"""Guidance-field construction: source weights and weighted multi-source propagation.

    F(x) = max_i [ w(u_i) - lam * d_free(x, s_i) ]

Weights come from each source's **original** trajectory progress ``u``; when obstacle
samples are dropped or bridged the remaining ``u`` values are never re-normalised, so a
collision interval leaves a genuine hole in the progress axis.

The propagation backends live in :mod:`guidance_field.propagation`; ``scipy`` (sparse
Dijkstra) is the default because it is exact and 20x+ faster than a Python heap here.
"""

from __future__ import annotations

import numpy as np

from .config import FIELD_CLIP, WEIGHT_KINDS, field_clip_bounds
from .grid import Grid
from .propagation import PROPAGATORS, propagate as propagate_field


def sample_progress(m: int) -> np.ndarray:
    """Normalised trajectory progress u_i = i / (M-1)."""
    return np.linspace(0.0, 1.0, m)


def weight_at_progress(u, kind: str) -> np.ndarray:
    """Continuous weight function w(u)."""
    u = np.asarray(u, dtype=np.float64)
    if kind == "uniform":
        return np.ones_like(u)
    if kind == "linear":
        return u.copy()
    if kind == "quadratic":
        return u**2
    raise ValueError(f"unknown weight kind {kind!r}, expected one of {WEIGHT_KINDS}")


def source_weights(progress, kind: str) -> np.ndarray:
    """Weights from the sources' stored original progress (no re-normalisation)."""
    return np.asarray(weight_at_progress(np.asarray(progress, dtype=np.float64), kind), dtype=np.float64)


def build_weighted_field(grid: Grid, sources: dict, kind: str, lam: float,
                         method: str = "scipy") -> dict:
    """One field for one (source set, weighting) pair.

    ``sources`` needs ``positions_rc`` (K, 2) and ``progress`` (K,).  ``method`` selects the
    propagator; the ``heap`` backend is used automatically when the dominant source is
    requested, because the sparse backends do not track ownership.
    """
    positions = np.asarray(sources["positions_rc"], dtype=int).reshape(-1, 2)
    progress = np.asarray(sources["progress"], dtype=np.float64).reshape(-1)
    if len(positions) == 0:
        return {
            "field": np.full((grid.height, grid.width), np.nan, dtype=np.float64),
            "owner": np.full((grid.height, grid.width), -1, dtype=np.int32),
            "weights": np.zeros(0),
            "num_sources": 0,
            "empty": True,
        }
    weights = source_weights(progress, kind)
    field, owner = propagate_field(grid, positions, weights, lam, method=method, return_owner=True)
    return {
        "field": np.asarray(field, dtype=np.float64),
        "owner": owner,
        "weights": weights,
        "num_sources": int(len(positions)),
        "empty": False,
    }


def build_all_weight_variants(grid: Grid, sources: dict, lam: float,
                              method: str = "scipy") -> dict[str, dict]:
    """uniform / linear / quadratic fields for the same source set."""
    return {kind: build_weighted_field(grid, sources, kind, lam, method=method) for kind in WEIGHT_KINDS}


__all__ = [
    "FIELD_CLIP",
    "PROPAGATORS",
    "build_all_weight_variants",
    "build_weighted_field",
    "field_clip_bounds",
    "propagate_field",
    "sample_progress",
    "source_weights",
    "weight_at_progress",
]
