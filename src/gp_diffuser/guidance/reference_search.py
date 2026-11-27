"""Adapter: run the *vendored* Astar-Diffusion library instead of our own port.

``src/gp_diffuser/vendor/field_guided_search`` is a verbatim copy of
``references/Astar-Diffusion/field-guided-search/src/field_guided_search`` (the
packaged library the upstream project ships).  This module keeps the interface
the rest of GeoPath-Diffuser already uses

    search(occupancy[H,W] 1=blocked, guide_xy metres, start_m, goal_m) -> dict

and delegates every geometric decision to that library:

* map convention: upstream takes ``0 = free``; we pass ``(occupancy == 0)``;
* coordinates    : upstream works in ``[row, col]`` cells; we convert with the
  PathSafe convention ``col = x / res``, ``row = y / res`` (cell centres);
* guidance field : ``GuidanceField.build(guide_rc)`` (their
  ``MAX_i[w(u_i) - lambda d]`` construction, their source projection);
* search         : ``GuidanceField.search(start, goal, alpha=...)`` (their
  B3 linear fusion / A* fallback) plus their ``astar``/``path_is_valid``.

Our own ``guidance/search.py`` stays in the tree but is no longer used by the
pipeline.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..vendor.field_guided_search import GuidanceField


def _as_path(value) -> np.ndarray:
    """Normalise upstream `path` (may be None / [] / ndarray) to `[K,2]`."""
    if value is None:
        return np.zeros((0, 2), dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    return arr.reshape(-1, 2)


class ReferenceGuidanceSearch:
    """Thin wrapper over ``field_guided_search.GuidanceField`` (vendored)."""

    def __init__(self, cfg: Optional[dict] = None, res_m: float = 0.25):
        cfg = dict(cfg or {})
        self.cfg = cfg
        self.res_m = float(res_m)
        self.lam = float(cfg.get("lam", 0.01))
        self.weight = str(cfg.get("weight", "linear"))
        self.alpha = float(cfg.get("alpha", 0.25))
        self.num_sources = int(cfg.get("num_sources", cfg.get("num_samples", 64)))
        # upstream names the two source policies 'drop' / 'project'; the design doc
        # of this project calls them M0 / M1
        raw_method = str(cfg.get("source_method", "drop"))
        self.source_method = {"M0": "drop", "M1": "project"}.get(raw_method, raw_method)
        if self.source_method not in ("drop", "project"):
            self.source_method = "drop"
        self.projection_radius = int(cfg.get("projection_radius", 4))
        self.path_points = int(cfg.get("path_points", 128))
        self.grid_cache_size = int(cfg.get("grid_cache_size", 64))
        self._field_cache: Dict[bytes, GuidanceField] = {}

    # ------------------------------------------------------------------ utils
    def _to_cell(self, points) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        return np.stack([np.rint(pts[:, 1] / self.res_m),
                         np.rint(pts[:, 0] / self.res_m)], axis=1)     # [row, col]

    def _to_metres(self, cells) -> np.ndarray:
        cells = np.asarray(cells, dtype=np.float64).reshape(-1, 2)
        return np.stack([cells[:, 1] * self.res_m, cells[:, 0] * self.res_m], axis=1)

    def _resample(self, path_m: np.ndarray, count: int) -> np.ndarray:
        if len(path_m) == 1:
            return np.repeat(path_m, count, axis=0)
        seg = np.linalg.norm(np.diff(path_m, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        total = s[-1]
        if total <= 1e-9:
            return np.repeat(path_m[:1], count, axis=0)
        target = np.linspace(0.0, total, int(count))
        return np.stack([np.interp(target, s, path_m[:, 0]),
                         np.interp(target, s, path_m[:, 1])], axis=1)

    # ----------------------------------------------------------------- search
    def search(self, occupancy: np.ndarray, guide_xy: np.ndarray,
               start_xy, goal_xy, return_debug: bool = False) -> Dict:
        occ = np.asarray(occupancy)
        if occ.ndim == 3:
            occ = occ[0]
        # upstream Grid convention is `free = (occupancy == 0)`, which is exactly
        # ours (1 = blocked), so the map is handed over unchanged
        free_map = (occ != 0).astype(np.uint8)

        start_rc = self._to_cell(np.asarray(start_xy).reshape(1, 2))[0]
        goal_rc = self._to_cell(np.asarray(goal_xy).reshape(1, 2))[0]
        guide_rc = self._to_cell(guide_xy)

        height, width = free_map.shape
        start_rc = np.clip(start_rc, 0, [height - 1, width - 1])
        goal_rc = np.clip(goal_rc, 0, [height - 1, width - 1])
        result: Dict = {
            "status": "failed", "path_xy": None, "valid": False,
            "fallback": None, "sources": 0, "expanded": 0,
            "endpoint_projected": False, "path_verified": False,
        }

        # their GuidanceField refuses obstacles as start/goal: project first
        free_bool = (occ == 0)
        for name, cell in (("start", start_rc), ("goal", goal_rc)):
            if not free_bool[int(cell[0]), int(cell[1])]:
                result["fallback"] = "%s_in_obstacle" % name
                cand = np.argwhere(free_bool)
                if len(cand) == 0:
                    return result
                d = ((cand - cell) ** 2).sum(axis=1)
                cell[:] = cand[int(np.argmin(d))]
                result["endpoint_projected"] = True
                if name == "start":
                    start_rc = cell
                else:
                    goal_rc = cell

        gfield = GuidanceField(free_map, lam=self.lam, weight=self.weight,
                               num_samples=self.num_sources,
                               source_method=self.source_method,
                               projection_radius=self.projection_radius)
        try:
            gfield.build(guide_rc.astype(np.float64))
        except Exception as error:                                   # noqa: BLE001
            # no legal source on the guide -> their plain A* is still a valid route
            astar = gfield.astar(start_rc.astype(int), goal_rc.astype(int))
            result.update({"status": astar.get("status"), "fallback": "no_source_astar",
                           "expanded": int(astar.get("expanded", 0) or 0)})
            path = _as_path(astar.get("path"))
            if len(path) >= 2:
                result["path_xy"] = self._resample(self._to_metres(path), self.path_points)
                result["valid"] = True
                result["path_verified"] = bool(
                    gfield.is_valid(path, start_rc.astype(int), goal_rc.astype(int)).get("valid", True))
            return result

        out = gfield.search(start_rc.astype(int), goal_rc.astype(int), alpha=self.alpha)
        status = str(out.get("status", "failed"))
        path = _as_path(out.get("path"))
        result["status"] = status
        result["expanded"] = int(out.get("expanded", 0) or 0)
        result["sources"] = int(len((gfield.sources or {}).get("positions_rc", [])))
        if len(path) < 2:
            astar = gfield.astar(start_rc.astype(int), goal_rc.astype(int))
            path = _as_path(astar.get("path"))
            result["fallback"] = "astar"
            result["status"] = str(astar.get("status", status))
            result["expanded"] = int(astar.get("expanded", 0) or 0)
        if len(path) >= 2:
            check = gfield.is_valid(path, start_rc.astype(int), goal_rc.astype(int))
            result["path_verified"] = bool(check.get("valid", True))
            result["path_xy"] = self._resample(self._to_metres(path), self.path_points)
            result["valid"] = True
        if return_debug:
            result["field"] = gfield.field
            result["source_positions_rc"] = np.asarray(
                (gfield.sources or {}).get("positions_rc", []), dtype=np.float64)
        return result
