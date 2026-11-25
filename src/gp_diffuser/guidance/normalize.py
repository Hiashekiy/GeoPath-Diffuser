"""Normalisation of the guidance field and of the goal distance (plan §3.4, §4).

Both are mapped into [0, 1] with *explicit, reproducible* rules so that one alpha means the
same fusion ratio on every map and for every field.  The field clip bounds are read from the
stage-2 configuration (``FIELD_CLIP``); no constant is duplicated here.
"""

from __future__ import annotations

import numpy as np

from .grid import Grid
from .config import FIELD_CLIP, field_clip_bounds

EPS = 1e-12



def normalize_field(field: np.ndarray, weight: str) -> np.ndarray:
    """F_hat = (clip(F, Fmin, Fmax) - Fmin) / (Fmax - Fmin); NaN stays NaN."""
    lo, hi = field_clip_bounds(weight)
    span = (hi - lo) or 1.0
    clipped = np.clip(field, lo, hi)
    return (clipped - lo) / span


def manhattan_distance_map(shape: tuple[int, int], goal: tuple[int, int]) -> np.ndarray:
    rr, cc = np.mgrid[0 : shape[0], 0 : shape[1]]
    return np.abs(rr - goal[0]) + np.abs(cc - goal[1])


def normalize_distance(grid: Grid, goal: tuple[int, int]) -> tuple[np.ndarray, float]:
    """d_hat = d_G / (Dmax + eps), with Dmax over the free cells of this map."""
    d = manhattan_distance_map((grid.height, grid.width), goal).astype(np.float64)
    d_max = float(d[grid.free].max()) if grid.free.any() else 0.0
    return d / (d_max + EPS), d_max


def prepare_scalars(grid: Grid, field: np.ndarray, weight: str, goal: tuple[int, int]) -> dict:
    """Everything the search policies need, computed once per (map, field)."""
    f_hat = normalize_field(field, weight)
    d_hat, d_max = normalize_distance(grid, goal)
    return {
        "field": field,
        "F_hat": f_hat,
        "d_hat": d_hat,
        "distance": manhattan_distance_map((grid.height, grid.width), goal).astype(np.float64),
        "D_max": d_max,
        "clip": field_clip_bounds(weight),
    }
