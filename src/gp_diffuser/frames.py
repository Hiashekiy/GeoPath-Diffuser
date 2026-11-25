"""Coordinate frames used across the project.

Three frames appear and must never be mixed up:

``metres``
    The dataset / occupancy-grid frame.  Cell ``(row, col)`` has its centre at
    ``(col*res, row*res)``.  PathSafe-EllipseNet consumes this frame
    (``path_xy``, ``lidar``, ``sdf`` all in metres).
``norm``
    ``[-1, 1]^2`` pixel-centre normalised coordinates.  This is the SAME
    convention as ``geometry/sampling.py::world_to_grid``
    (``gx = 2*(x/res + 0.5)/W - 1``) and as TrajSafe's
    ``convex_region._obstacle_boundary_points``
    (``x = -1 + (col + 0.5) * 2/W``), so one transform keeps the occupancy grid,
    the migrated region builder and ``F.grid_sample`` aligned to the same cell
    centre.  The trajectory/B-spline/ALM state lives in this frame.
``cells``
    Integer grid indices used by the Astar-Diffusion search (``(row, col)``).
"""
from __future__ import annotations

import numpy as np


class SceneTransform:
    """Bidirectional metres <-> normalised ``[-1,1]`` conversion for one square map.

    ``norm = 2 * metres / extent + (-1 + 1/grid)`` and its exact inverse, so that
    ``norm = 0`` is the centre of the map and every cell centre maps onto the
    ``grid_sample`` coordinate of that cell.
    """

    def __init__(self, extent_m: float, grid: int):
        self.extent_m = float(extent_m)
        self.grid = int(grid)
        if self.extent_m <= 0 or self.grid <= 0:
            raise ValueError("extent_m and grid must be positive")
        self.scale = self.extent_m / 2.0            # metres per normalised unit
        self.offset = -1.0 + 1.0 / float(self.grid)  # half-pixel shift

    def to_scene(self, xy_m) -> np.ndarray:
        return 2.0 * np.asarray(xy_m, dtype=np.float64) / self.extent_m + self.offset

    def to_meters(self, xy_scene) -> np.ndarray:
        return (np.asarray(xy_scene, dtype=np.float64) - self.offset) * self.scale

    def lengths_to_scene(self, lengths_m) -> np.ndarray:
        return np.asarray(lengths_m, dtype=np.float64) / self.scale

    def lengths_to_meters(self, lengths_scene) -> np.ndarray:
        return np.asarray(lengths_scene, dtype=np.float64) * self.scale

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "SceneTransform(extent_m=%g, grid=%d)" % (self.extent_m, self.grid)


def resize_cells(occupancy: np.ndarray, grid: int) -> np.ndarray:
    """Nearest-neighbour resize of an occupancy map to ``grid`` x ``grid``."""
    occ = np.asarray(occupancy)
    if occ.shape[-1] == grid and occ.shape[-2] == grid:
        return occ
    rows = (np.arange(grid) + 0.5) * (occ.shape[-2] / float(grid))
    cols = (np.arange(grid) + 0.5) * (occ.shape[-1] / float(grid))
    return occ[np.clip(rows.astype(int), 0, occ.shape[-2] - 1)][
        :, np.clip(cols.astype(int), 0, occ.shape[-1] - 1)]
