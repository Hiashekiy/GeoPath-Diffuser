"""Project adapter around the Astar-Diffusion guidance-field search.

The solver itself is the migrated upstream code (``grid`` / ``sources`` /
``field`` / ``propagation`` / ``policies`` / ``paths`` / ``collision`` /
``normalize``).  This module only supplies what the new project needs:

* a world-metre -> grid-cell conversion for the unified safe map,
* guide-trajectory -> legal free-space sources (upstream ``M0`` drop rule),
* one call that returns a single feasible path, arc-length resampled to 128
  points, with the plain A* fallback the design document asks for (section 5.4),
* a debug payload (field, expansion count) for the visualisation tool.

Default policy is ``B3_linear_fusion`` with ``alpha = 0.25`` on a ``linear``
field weight: that is the configuration the upstream ``docs/results.md``
reports as the best expansion ratio (0.204 x A*, paths 4.2 % longer).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .collision import project_to_nearest_free
from .config import FREE, OCCUPIED
from .field import build_weighted_field
from .grid import Grid
from .normalize import prepare_scalars
from .paths import astar
from .policies import POLICY_CLASSES, astar_search, priority_search, path_is_valid
from .sources import build_sources_drop, build_sources_project


def inflate_occupancy(occupancy: np.ndarray, cells: int) -> np.ndarray:
    """Binary dilation of the blocked set by ``cells`` (Chebyshev/square kernel).

    Used only when the dataset's ``blocked`` map is a raw obstacle map rather
    than a robot-centre feasible map (design document section 5.3).  The
    dataset used here already stores a centre-feasible map, so the default
    inflation is 0 and this is a no-op.
    """
    occ = np.asarray(occupancy)
    if cells <= 0:
        return occ.astype(np.int8, copy=False)
    blocked = occ > 0
    k = 2 * int(cells) + 1
    padded = np.pad(blocked, cells, mode="constant", constant_values=True)
    out = np.zeros_like(blocked)
    for dr in range(k):
        for dc in range(k):
            out |= padded[dr:dr + blocked.shape[0], dc:dc + blocked.shape[1]]
    return out.astype(np.int8)


class GuidanceFieldSearch:
    """Weighted guidance-field search on a ``[H,W]`` occupancy map."""

    def __init__(self, cfg: Optional[dict] = None, res_m: float = 0.25):
        cfg = dict(cfg or {})
        self.res_m = float(res_m)
        self.lam = float(cfg.get("lam", 0.01))
        self.weight = str(cfg.get("weight", "linear"))
        self.policy_name = str(cfg.get("policy", "B3_linear_fusion"))
        self.alpha = float(cfg.get("alpha", 0.25))
        self.gamma = float(cfg.get("gamma", 0.5))
        self.num_sources = int(cfg.get("num_sources", 64))
        self.path_points = int(cfg.get("path_points", 128))
        self.source_method = str(cfg.get("source_method", "M0"))
        self.inflation_cells = int(cfg.get("inflation_cells", 0))
        self.projection_radius = int(cfg.get("projection_radius", 4))

    # ------------------------------------------------------------------ helpers
    def world_to_cell(self, points_xy: np.ndarray) -> np.ndarray:
        """World metres -> float cell coordinates ``(row, col)``.

        Cell ``(row, col)`` has its centre at world ``(col*res, row*res)``, which
        is the convention used by PathSafe-EllipseNet's coordinate queries.
        """
        pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        return np.stack([pts[:, 1] / self.res_m, pts[:, 0] / self.res_m], axis=-1)

    def cell_to_world(self, cells: np.ndarray) -> np.ndarray:
        cells = np.asarray(cells, dtype=np.float64).reshape(-1, 2)
        return np.stack([cells[:, 1] * self.res_m, cells[:, 0] * self.res_m], axis=-1)

    @staticmethod
    def _project_free(grid: Grid, cell: Tuple[int, int], radius: int) -> Optional[Tuple[int, int]]:
        if grid.is_free(int(cell[0]), int(cell[1])):
            return int(cell[0]), int(cell[1])
        found = project_to_nearest_free(grid, (int(cell[0]), int(cell[1])), radius)
        return None if found is None else found[0]

    def _sources(self, grid: Grid, guide_cells: np.ndarray, cfg_dummy=None) -> dict:
        """Legal free-space sources from the guide trajectory (upstream M0/M1)."""
        if self.source_method == "M1":
            return build_sources_project(grid, guide_cells, self.num_sources, _Cfg(self.projection_radius))
        return build_sources_drop(grid, guide_cells, self.num_sources, _Cfg(self.projection_radius))

    def _policy(self, scalars: dict):
        cls = POLICY_CLASSES.get(self.policy_name)
        if cls is None:
            raise KeyError("unknown policy %r (one of %s)"
                           % (self.policy_name, ", ".join(POLICY_CLASSES)))
        if cls.__name__ in ("LinearFusionPolicy",):
            return cls(scalars, alpha=self.alpha)
        if cls.__name__ in ("NonlinearFusionPolicy",):
            return cls(scalars, alpha=self.alpha, gamma=self.gamma)
        if cls.__name__ in ("AdaptiveFusionPolicy",):
            return cls(scalars, alpha_far=self.alpha, alpha_near=1.0 - self.alpha,
                       eta=1.0, gamma=self.gamma)
        return cls(scalars)

    # ------------------------------------------------------------------- search
    def search(self, occupancy: np.ndarray, guide_xy: np.ndarray,
               start_xy, goal_xy, return_debug: bool = False) -> Dict:
        """One guided search -> a single feasible path in world metres.

        ``occupancy`` is ``[H,W]`` with ``1 = blocked``.  ``guide_xy`` is the
        guide trajectory (GT path in training, the map-only diffusion prediction
        at inference).  The returned ``path_xy`` has ``path_points`` samples,
        uniform in arc length, with the exact start/goal as first/last point.
        """
        occ = inflate_occupancy(occupancy, self.inflation_cells)
        occ = np.where(occ > 0, OCCUPIED, FREE).astype(np.int8)
        grid = Grid(occ)

        start_cell = self._project_free(grid, tuple(np.rint(self.world_to_cell(np.asarray(start_xy))[0]).astype(int)), self.projection_radius)
        goal_cell = self._project_free(grid, tuple(np.rint(self.world_to_cell(np.asarray(goal_xy))[0]).astype(int)), self.projection_radius)
        if start_cell is None or goal_cell is None:
            return {"status": "invalid_endpoints", "path_xy": None, "fallback": None,
                    "sources": 0, "expanded": 0, "valid": False}

        guide_cells = self.world_to_cell(guide_xy)
        # keep the guide inside the map so the source sampler stays meaningful
        guide_cells[:, 0] = np.clip(guide_cells[:, 0], 0.0, grid.height - 1.0)
        guide_cells[:, 1] = np.clip(guide_cells[:, 1], 0.0, grid.width - 1.0)
        sources = self._sources(grid, guide_cells)

        debug = {"grid": grid, "sources": sources, "field": None, "scalars": None}
        status, nodes, expanded, fallback = "unreachable", [], 0, None
        if len(sources["positions_rc"]) > 0:
            field = build_weighted_field(grid, sources, self.weight, self.lam,
                                         method="scipy" if _have_scipy() else "heap")
            scalars = prepare_scalars(grid, field["field"], self.weight, goal_cell)
            policy = self._policy(scalars)
            result = priority_search(grid, start_cell, goal_cell, policy)
            status, nodes, expanded = result["status"], result["path"], result["expanded"]
            debug.update({"field": field["field"], "scalars": scalars,
                          "policy": policy.describe()})

        if status != "success" or not nodes:
            base = astar_search(grid, start_cell, goal_cell)
            fallback = base["status"]
            status, nodes = base["status"], base["path"]
            expanded = base["expanded"]

        if status != "success" or not nodes:
            return {"status": status, "path_xy": None, "fallback": fallback,
                    "sources": int(len(sources["positions_rc"])), "expanded": expanded,
                    "valid": False}

        check = path_is_valid(grid, nodes, start_cell, goal_cell)
        world = self.cell_to_world(np.array([grid.rc(int(n)) for n in nodes], dtype=np.float64))
        world[0] = np.asarray(start_xy, dtype=np.float64).reshape(2)
        world[-1] = np.asarray(goal_xy, dtype=np.float64).reshape(2)
        path = resample_polyline(world, self.path_points)

        out = {
            "status": status,
            "fallback": fallback,
            "path_xy": path,                       # [128,2] world metres
            "path_nodes": nodes,
            "sources": int(len(sources["positions_rc"])),
            "expanded": int(expanded),
            "valid": bool(check["valid"]),
            "valid_reason": check.get("reason", ""),
            "start_cell": start_cell,
            "goal_cell": goal_cell,
        }
        if return_debug:
            out["debug"] = debug
        return out


class _Cfg:
    """Minimal ``Config`` stand-in for the upstream source builders."""

    def __init__(self, projection_radius: int = 4):
        self.projection_radius = int(projection_radius)
        self.bridge_roi_margins = (4, 8, 16, 32)
        self.num_source_samples = 64


def _have_scipy() -> bool:
    try:
        import scipy.sparse.csgraph  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def resample_polyline(points: np.ndarray, count: int) -> np.ndarray:
    """Uniform arc-length resampling of a world polyline (endpoints preserved)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        raise ValueError("polyline needs at least two points")
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= 1e-12:
        return np.repeat(pts[:1], count, axis=0)
    target = np.linspace(0.0, total, int(count))
    idx = np.clip(np.searchsorted(cum, target, side="right") - 1, 0, len(pts) - 2)
    span = (cum[idx + 1] - cum[idx]).clip(min=1e-12)
    frac = ((target - cum[idx]) / span).clip(0.0, 1.0)
    out = pts[idx] + frac[:, None] * (pts[idx + 1] - pts[idx])
    out[0], out[-1] = pts[0], pts[-1]
    return out


def astar_fallback(occupancy: np.ndarray, start_xy, goal_xy, res_m: float = 0.25,
                   count: int = 128) -> Optional[np.ndarray]:
    """Plain A* path in world metres (used as a standalone baseline)."""
    grid = Grid(np.where(np.asarray(occupancy) > 0, OCCUPIED, FREE).astype(np.int8))
    s = tuple(np.rint(np.array([start_xy[1], start_xy[0]]) / res_m).astype(int))
    g = tuple(np.rint(np.array([goal_xy[1], goal_xy[0]]) / res_m).astype(int))
    nodes = astar(grid, grid.idx(*s), grid.idx(*g))
    if not nodes:
        return None
    world = np.stack([np.array([grid.rc(int(n)) for n in nodes], dtype=np.float64)[:, 1],
                      np.array([grid.rc(int(n)) for n in nodes], dtype=np.float64)[:, 0]], axis=-1) * res_m
    return resample_polyline(world, count)
