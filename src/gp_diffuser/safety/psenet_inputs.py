"""LiDAR ray features for PathSafe-EllipseNet, reproduced from its own spec.

PathSafe-EllipseNet consumes ``lidar`` (range per ray), ``ray_dir`` and
``ray_valid``.  The loader that produced those arrays is NOT in the
PathSafe-Ellipse repository (``data/`` is git-ignored), so the semantics are
reproduced here from what the repository *does* document:

* ``docs/report.md`` §5: ``angle_k = 2*pi*k/16``, ``u = (cos, sin)``,
  ``ray_directions`` derived by the loader as a ``(16,2)`` array;
* ``docs/report.md`` §5 (measured): ``lidar_ranges`` is the distance from the
  path point to the *boundary of the first blocked cell* along the ray, i.e.
  "how far a point robot can walk in that direction";
* ``docs/report.md`` §5: ``ray_valid = lidar < max_range - 1 mm`` distinguishes a
  real hit from a range truncation;
* ``results/metrics/data_contract.json``: ``max_range_m = 9.0``, cell
  ``resolution_m = 0.25``, cell ``(row, col)`` centre at world ``(col*res, row*res)``
  (the same convention as ``geometry/sampling.py``).

Outside the map counts as blocked, matching ``sample_sdf``'s
``SDF_OUTSIDE_M`` rule.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch

RAY_COUNT = 16
MAX_RANGE_M = 9.0
RAY_HIT_EPS_M = 1e-3

_EPS = 1e-12


def ray_directions(num_rays: int = RAY_COUNT) -> np.ndarray:
    """``angle_k = 2*pi*k/num_rays`` -> ``(R, 2)`` unit directions."""
    angles = 2.0 * math.pi * np.arange(int(num_rays), dtype=np.float64) / float(num_rays)
    return np.stack([np.cos(angles), np.sin(angles)], axis=-1)


def _cast_one(occupancy: np.ndarray, point_cell: Tuple[float, float],
              direction: np.ndarray, res_m: float, max_range_m: float) -> float:
    """Amanatides-Woo march from one world point; returns metres to the first wall."""
    height, width = occupancy.shape
    # cell coordinates: cell (row, col) spans [row-0.5, row+0.5] x [col-0.5, col+0.5]
    x = float(point_cell[1])
    y = float(point_cell[0])
    dx, dy = float(direction[0]), float(direction[1])
    ix, iy = int(math.floor(x + 0.5)), int(math.floor(y + 0.5))

    def blocked(i: int, j: int) -> bool:
        if not (0 <= i < height and 0 <= j < width):
            return True  # outside the map is blocked
        return occupancy[i, j] > 0

    if blocked(iy, ix):
        return 0.0

    step_x = 1 if dx > _EPS else (-1 if dx < -_EPS else 0)
    step_y = 1 if dy > _EPS else (-1 if dy < -_EPS else 0)
    inf = float("inf")
    if step_x > 0:
        t_max_x, t_delta_x = (ix + 0.5 - x) / dx, 1.0 / dx
    elif step_x < 0:
        t_max_x, t_delta_x = (ix - 0.5 - x) / dx, -1.0 / dx
    else:
        t_max_x, t_delta_x = inf, inf
    if step_y > 0:
        t_max_y, t_delta_y = (iy + 0.5 - y) / dy, 1.0 / dy
    elif step_y < 0:
        t_max_y, t_delta_y = (iy - 0.5 - y) / dy, -1.0 / dy
    else:
        t_max_y, t_delta_y = inf, inf

    limit_cells = max_range_m / max(res_m, _EPS)
    travelled = 0.0
    for _ in range(int(4 * limit_cells) + 8):
        if t_max_x < t_max_y:
            travelled = t_max_x
            ix += step_x
            t_max_x += t_delta_x
        else:
            travelled = t_max_y
            iy += step_y
            t_max_y += t_delta_y
        if travelled >= limit_cells:
            return float(max_range_m)
        if blocked(iy, ix):
            return float(min(travelled * res_m, max_range_m))
    return float(max_range_m)


def cast_rays(occupancy: np.ndarray, points_xy: np.ndarray, res_m: float,
              num_rays: int = RAY_COUNT, max_range_m: float = MAX_RANGE_M
              ) -> dict:
    """Ray features for ``points_xy [N,2]`` (world metres) on a ``[H,W]`` map.

    Returns ``{"lidar": [N,R] metres, "ray_dir": [R,2], "ray_valid": [N,R]}``.
    """
    occ = np.asarray(occupancy)
    if occ.ndim == 3:
        occ = occ[0]
    occ = (occ > 0.5).astype(np.int8)
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    dirs = ray_directions(num_rays)
    cell_pts = pts / float(res_m)                 # (row, col) float cell coordinates
    lidar = np.full((len(pts), int(num_rays)), float(max_range_m), dtype=np.float32)
    for n in range(len(pts)):
        y = cell_pts[n, 1]
        x = cell_pts[n, 0]
        for k in range(int(num_rays)):
            lidar[n, k] = _cast_one(occ, (y, x), dirs[k], float(res_m), float(max_range_m))
    ray_valid = (lidar < float(max_range_m) - RAY_HIT_EPS_M).astype(np.float32)
    return {
        "lidar": torch.from_numpy(lidar),
        "ray_dir": torch.from_numpy(dirs.astype(np.float32)),
        "ray_valid": torch.from_numpy(ray_valid),
    }
