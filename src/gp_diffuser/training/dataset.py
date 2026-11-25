"""Synthetic scene dataset and the offline safety-condition cache.

Why synthetic data: neither reference repository ships its dataset
(PathSafe-Ellipse git-ignores ``data/``; TrajSafe-Diffuser's configs point at
an external processed root).  The project therefore generates scenes that obey
the *documented* data contract of PathSafe-EllipseNet
(``results/metrics/data_contract.json``): 256x256 grid, ``resolution_m = 0.25``,
``occupancy`` in {0,1}, signed distance in metres with ``SDF_OUTSIDE_M = -1``,
16 LiDAR rays with ``max_range_m = 9.0``.

One map, one obstacle semantic (design document section 5.3): the raw obstacles
are dilated by the robot radius ONCE, and that dilated map is the single
``blocked`` map used by the search, the SDF, PathSafe-EllipseNet, the corridor
builder and the collision metrics.
"""
from __future__ import annotations

import heapq
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..frames import SceneTransform
from ..guidance.search import resample_polyline
from ..trajectory.bspline import BSplineCodec

FREE = 0
BLOCKED = 1

_HAVE_SCIPY_NDIMAGE = True
try:  # pragma: no cover - environment dependent
    from scipy import ndimage as _ndimage
except Exception:  # noqa: BLE001
    _HAVE_SCIPY_NDIMAGE = False
    _ndimage = None


# --------------------------------------------------------------------------- #
# distance field
# --------------------------------------------------------------------------- #
def sdf_from_occupancy(occupancy: np.ndarray, res_m: float) -> np.ndarray:
    """Signed distance in metres: positive in free space, negative inside.

    Exact Euclidean distance when ``scipy.ndimage`` is available, otherwise a
    two-pass 3-4 chamfer approximation (~2 % error, documented in the README).
    """
    blocked = np.asarray(occupancy) > 0.5
    if _HAVE_SCIPY_NDIMAGE:
        d_free = _ndimage.distance_transform_edt(~blocked) * float(res_m)
        d_blocked = _ndimage.distance_transform_edt(blocked) * float(res_m)
        return (d_free - d_blocked).astype(np.float32)
    return _chamfer_sdf(blocked, float(res_m))


def _chamfer_distance_to(target: np.ndarray) -> np.ndarray:
    """Two-pass 3-4 chamfer distance (in cells) to the nearest ``True`` cell."""
    height, width = target.shape
    big = 1e9
    dist = np.where(target, 0.0, big)
    d1, d2 = 1.0, float(np.sqrt(2.0))
    for r in range(height):
        row = dist[r]
        for c in range(width):
            v = row[c]
            if v == 0.0:
                continue
            best = v
            if r > 0:
                prev = dist[r - 1]
                best = min(best, prev[c] + d1)
                if c > 0:
                    best = min(best, prev[c - 1] + d2)
                if c + 1 < width:
                    best = min(best, prev[c + 1] + d2)
            if c > 0:
                best = min(best, row[c - 1] + d1)
            row[c] = best
    for r in range(height - 1, -1, -1):
        row = dist[r]
        for c in range(width - 1, -1, -1):
            v = row[c]
            if v == 0.0:
                continue
            best = v
            if r + 1 < height:
                nxt = dist[r + 1]
                best = min(best, nxt[c] + d1)
                if c > 0:
                    best = min(best, nxt[c - 1] + d2)
                if c + 1 < width:
                    best = min(best, nxt[c + 1] + d2)
            if c + 1 < width:
                best = min(best, row[c + 1] + d1)
            row[c] = best
    return dist


def _chamfer_sdf(blocked: np.ndarray, res_m: float) -> np.ndarray:
    """Signed chamfer field: positive in free space, negative inside obstacles."""
    d_to_blocked = _chamfer_distance_to(blocked)
    d_to_free = _chamfer_distance_to(~blocked)
    sdf = np.where(blocked, -d_to_free, d_to_blocked)
    return (sdf * float(res_m)).astype(np.float32)


def dilate(occupancy: np.ndarray, cells: int) -> np.ndarray:
    """Square-kernel dilation of the blocked set (robot radius on a grid)."""
    if cells <= 0:
        return np.asarray(occupancy).astype(np.int8)
    blocked = np.asarray(occupancy) > 0
    if _HAVE_SCIPY_NDIMAGE:
        out = _ndimage.binary_dilation(blocked, structure=np.ones((2 * cells + 1,) * 2, bool))
        return out.astype(np.int8)
    k = 2 * int(cells) + 1
    padded = np.pad(blocked, cells, mode="constant", constant_values=True)
    out = np.zeros_like(blocked)
    for dr in range(k):
        for dc in range(k):
            out |= padded[dr:dr + blocked.shape[0], dc:dc + blocked.shape[1]]
    return out.astype(np.int8)


# --------------------------------------------------------------------------- #
# grid A* used only to author the GT path labels
# --------------------------------------------------------------------------- #
def grid_astar(free: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
               max_expand: int = 400000) -> Optional[np.ndarray]:
    """4-neighbour A* on a boolean free map; returns ``(N,2)`` ``(row, col)`` ints.

    This authors the GT path LABEL for the synthetic dataset.  It is not the
    guidance search (that one is the migrated Astar-Diffusion code).
    """
    height, width = free.shape
    if not (free[start] and free[goal]):
        return None
    def h(i):
        return abs(i // width - goal[0]) + abs(i % width - goal[1])
    start_i, goal_i = start[0] * width + start[1], goal[0] * width + goal[1]
    g = {start_i: 0.0}
    parent = {start_i: -1}
    heap = [(h(start_i), 0, start_i)]
    closed = np.zeros(height * width, dtype=bool)
    expands = 0
    while heap:
        _, _, v = heapq.heappop(heap)
        if closed[v]:
            continue
        closed[v] = True
        expands += 1
        if expands > max_expand:
            return None
        if v == goal_i:
            nodes = [v]
            while parent[nodes[-1]] >= 0:
                nodes.append(parent[nodes[-1]])
            nodes.reverse()
            return np.stack([np.array(nodes) // width, np.array(nodes) % width], axis=-1)
        r, c = divmod(v, width)
        gv = g[v]
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < height and 0 <= cc < width) or not free[rr, cc]:
                continue
            u = rr * width + cc
            if closed[u]:
                continue
            ng = gv + 1.0
            if ng < g.get(u, 1e18):
                g[u] = ng
                parent[u] = v
                heapq.heappush(heap, (ng + h(u), int(u), u))
    return None


# --------------------------------------------------------------------------- #
# scene generation
# --------------------------------------------------------------------------- #
def make_scene(seed: int, cfg: dict) -> Dict:
    """One synthetic scene with GT path and GT B-spline controls (no safety cache)."""
    data_cfg = dict(cfg.get("dataset") or {})
    res_m = float(data_cfg.get("res_m", 0.25))
    grid = int(data_cfg.get("grid", 256))
    robot_cells = int(data_cfg.get("robot_radius_cells", 2))
    min_geodesic_m = float(data_cfg.get("min_geodesic_m", 20.0))
    rng = np.random.default_rng(int(seed))

    raw = np.zeros((grid, grid), dtype=np.int8)
    n_rect = int(rng.integers(6, 13))
    for _ in range(n_rect):
        w = int(rng.integers(int(grid * 0.05), int(grid * 0.20)))
        h = int(rng.integers(int(grid * 0.05), int(grid * 0.20)))
        r = int(rng.integers(0, grid - h))
        c = int(rng.integers(0, grid - w))
        raw[r:r + h, c:c + w] = BLOCKED
    # a couple of long walls to create real detours
    for _ in range(int(rng.integers(1, 4))):
        if rng.random() < 0.5:
            c = int(rng.integers(int(grid * 0.2), int(grid * 0.8)))
            r0 = int(rng.integers(0, grid // 2))
            r1 = r0 + int(rng.integers(grid // 4, grid // 2))
            raw[r0:min(r1, grid), c:c + 2] = BLOCKED
        else:
            r = int(rng.integers(int(grid * 0.2), int(grid * 0.8)))
            c0 = int(rng.integers(0, grid // 2))
            c1 = c0 + int(rng.integers(grid // 4, grid // 2))
            raw[r:r + 2, c0:min(c1, grid)] = BLOCKED
    raw[:3, :] = BLOCKED
    raw[-3:, :] = BLOCKED
    raw[:, :3] = BLOCKED
    raw[:, -3:] = BLOCKED

    blocked = dilate(raw, robot_cells)
    free = blocked == FREE

    # start / goal: far enough apart (geodesic, 4-neighbour BFS on a coarse grid)
    for _ in range(60):
        s_rc = (int(rng.integers(8, grid - 8)), int(rng.integers(8, grid - 8)))
        g_rc = (int(rng.integers(8, grid - 8)), int(rng.integers(8, grid - 8)))
        if not (free[s_rc] and free[g_rc]):
            continue
        if np.hypot(s_rc[0] - g_rc[0], s_rc[1] - g_rc[1]) * res_m < min_geodesic_m:
            continue
        path_rc = grid_astar(free, s_rc, g_rc)
        if path_rc is None or len(path_rc) * res_m < min_geodesic_m:
            continue
        break
    else:
        raise RuntimeError("scene %d: could not place a valid start/goal pair" % seed)

    path_m = np.stack([path_rc[:, 1] * res_m, path_rc[:, 0] * res_m], axis=-1)
    gt_path_m = resample_polyline(path_m, int(data_cfg.get("path_points", 128)))
    sdf = sdf_from_occupancy(blocked, res_m)

    return {
        "seed": int(seed),
        "occupancy": blocked.astype(np.float32),
        "raw_occupancy": raw.astype(np.float32),
        "sdf": sdf,
        "start_m": path_m[0].astype(np.float32),
        "goal_m": path_m[-1].astype(np.float32),
        "start_goal_m": np.stack([path_m[0], path_m[-1]]).astype(np.float32),
        "gt_path_m": gt_path_m.astype(np.float32),
        "res_m": res_m,
        "grid": grid,
    }


def _tangent(points: np.ndarray, at_start: bool) -> np.ndarray:
    if len(points) < 2:
        return np.array([1.0, 0.0], dtype=np.float32)
    delta = (points[1] - points[0]) if at_start else (points[-1] - points[-2])
    norm = float(np.linalg.norm(delta))
    if norm < 1e-9:
        return np.array([1.0, 0.0], dtype=np.float32)
    return (delta / norm).astype(np.float32)


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
class SyntheticTrajectoryDataset(Dataset):
    """Scenes + GT B-spline controls in the scene ``[-1,1]`` frame."""

    def __init__(self, cfg: dict, split: str = "train", size: Optional[int] = None,
                 seed_offset: int = 0):
        self.cfg = cfg
        data_cfg = dict(cfg.get("dataset") or {})
        self.res_m = float(data_cfg.get("res_m", 0.25))
        self.grid = int(data_cfg.get("grid", 256))
        self.extent_m = float(self.res_m * self.grid)
        self.transform = SceneTransform(self.extent_m, self.grid)
        self.scene_scale = self.transform.scale
        self.split = split
        default_size = {"train": int(data_cfg.get("train_size", 64)),
                        "val": int(data_cfg.get("val_size", 16)),
                        "test": int(data_cfg.get("test_size", 16))}.get(split, 16)
        self.size = int(size if size is not None else default_size)
        offsets = {"train": 0, "val": 100000, "test": 200000}
        self.seed_offset = int(offsets.get(split, 300000)) + int(seed_offset)
        traj_cfg = dict(cfg.get("trajectory") or {})
        self.codec = BSplineCodec(degree=int(traj_cfg.get("degree", 3)),
                                  num_controls=int(traj_cfg.get("num_controls", 32)),
                                  curve_points=int(traj_cfg.get("curve_points", 128)))
        self._cache: Dict[int, Dict] = {}

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if index in self._cache:
            return self._cache[index]
        scene = make_scene(self.seed_offset + index, self.cfg)
        tf = self.transform
        start_goal_scene = tf.to_scene(scene["start_goal_m"])
        curve = torch.as_tensor(tf.to_scene(scene["gt_path_m"]), dtype=torch.float32)[None]
        controls = self.codec.fit_curve_to_controls(
            curve, torch.as_tensor(start_goal_scene[0], dtype=torch.float32)[None],
            torch.as_tensor(start_goal_scene[1], dtype=torch.float32)[None])[0]
        heads = np.stack([_tangent(scene["gt_path_m"], True),
                          _tangent(scene["gt_path_m"], False)])
        item = {
            "occupancy": torch.as_tensor(scene["occupancy"], dtype=torch.float32)[None],
            "sdf": torch.as_tensor(scene["sdf"], dtype=torch.float32)[None],
            "start_goal": torch.as_tensor(start_goal_scene, dtype=torch.float32),
            "heading": torch.as_tensor(heads, dtype=torch.float32),
            "heading_valid": torch.ones(2, dtype=torch.bool),
            "controls_gt": controls,
            "path_gt_scene": curve[0],
            "path_gt_m": torch.as_tensor(scene["gt_path_m"], dtype=torch.float32),
            "start_m": torch.as_tensor(scene["start_m"], dtype=torch.float32),
            "goal_m": torch.as_tensor(scene["goal_m"], dtype=torch.float32),
            "seed": torch.tensor(int(scene["seed"]), dtype=torch.long),
            "res_m": torch.tensor(self.res_m),
            "scene_scale": torch.tensor(tf.scale),
        }
        if len(self._cache) < 8:
            self._cache[index] = item
        return item

    def raw_scene(self, index: int) -> Dict:
        """The numpy scene (for caching / visualisation)."""
        return make_scene(self.seed_offset + index, self.cfg)


def collate_scenes(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = ["occupancy", "sdf", "start_goal", "heading", "heading_valid",
            "controls_gt", "path_gt_scene", "path_gt_m", "start_m", "goal_m",
            "seed", "res_m", "scene_scale"]
    out = {}
    for key in keys:
        values = [b[key] for b in batch]
        if torch.is_tensor(values[0]) and values[0].dtype == torch.bool:
            out[key] = torch.stack(values)
        else:
            out[key] = torch.stack(values)
    return out
