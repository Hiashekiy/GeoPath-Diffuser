"""Collision intervals on a continuous curve, and their mapping to ellipse queries.

Two defects of the first implementation are fixed here:

1. **Point-only collision detection** missed obstacles between two consecutive
   samples (128 decoded points are ~0.35 m apart on a 0.25 m grid).  The curve is
   now swept SEGMENT by SEGMENT with the Amanatides-Woo traversal already
   migrated from Astar-Diffusion (``guidance.collision.trace_segment_cells``),
   which is the same test the search uses.

2. **Index-proportional mapping** assumed the predicted curve's parameter
   progress equals the searched path's arc-length progress.  The collision
   locations are now PROJECTED onto the searched path and the query index is
   taken from the projected arc length, so a collision at position A always
   constrains the ellipses near A.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..guidance.collision import arc_lengths, trace_segment_cells

__all__ = ["curve_collision_intervals", "collision_spans_on_path",
           "project_points_to_path"]

_EPS = 1e-9


def _scene_to_cell(points_scene: np.ndarray, height: int, width: int) -> np.ndarray:
    """Normalised scene coords -> continuous cell coordinates ``(row, col)``.

    Cell ``(row, col)`` has its centre at index ``(row, col)``, i.e. the inverse
    of ``frames.SceneTransform.to_scene`` (``scene = 2*(index+0.5)/N - 1``).
    """
    pts = np.asarray(points_scene, dtype=np.float64).reshape(-1, 2)
    cols = (pts[:, 0] + 1.0) * 0.5 * width - 0.5
    rows = (pts[:, 1] + 1.0) * 0.5 * height - 0.5
    return np.stack([rows, cols], axis=-1)


def curve_collision_intervals(occupancy, curve_scene, outside_is_blocked: bool = True,
                              min_penetration_m: float = 0.0) -> Dict:
    """Segment-level collision intervals of one curve.

    ``occupancy`` is ``[H,W]`` (1 = blocked), ``curve_scene`` ``[N,2]`` in the
    normalised scene frame.  Returns intervals in both arc length (scene units)
    and normalised curve parameter ``u``.
    """
    occ = (occupancy.detach().cpu().numpy() if hasattr(occupancy, "detach")
           else np.asarray(occupancy))
    occ = np.squeeze(occ)
    if occ.ndim != 2:
        raise ValueError("occupancy must reduce to [H,W], got %s" % (occ.shape,))
    height, width = occ.shape
    curve = np.asarray(curve_scene, dtype=np.float64).reshape(-1, 2)
    if len(curve) < 2:
        raise ValueError("curve needs at least two points")

    cells = _scene_to_cell(curve, height, width)
    cum = arc_lengths(curve)
    total = float(cum[-1])

    raw: List[Tuple[float, float]] = []
    for i in range(len(curve) - 1):
        segment_length = float(cum[i + 1] - cum[i])
        if segment_length <= _EPS:
            continue
        for row, col, t_in, t_out in trace_segment_cells(cells[i], cells[i + 1]):
            if t_out - t_in <= _EPS:
                continue
            blocked = True
            if 0 <= row < height and 0 <= col < width:
                blocked = bool(occ[row, col] > 0.5)
            elif not outside_is_blocked:
                blocked = False
            if not blocked:
                continue
            s_in = float(cum[i] + t_in * segment_length)
            s_out = float(cum[i] + t_out * segment_length)
            if (s_out - s_in) < float(min_penetration_m):
                continue
            raw.append((s_in, s_out))

    if total <= _EPS:
        # a degenerate (zero-length) curve carries no segment to sweep, so fall
        # back to the cell containing its single location
        row = int(round(cells[0, 0]))
        col = int(round(cells[0, 1]))
        blocked = True
        if 0 <= row < height and 0 <= col < width:
            blocked = bool(occ[row, col] > 0.5)
        elif not outside_is_blocked:
            blocked = False
        return {
            "intervals": ([{"s_in": 0.0, "s_out": 0.0, "u_in": 0.0,
                            "u_out": 0.0, "length": 0.0}] if blocked else []),
            "path_length": 0.0,
            "collision_length": 0.0,
            "collision_fraction": 1.0 if blocked else 0.0,
            "arc": cum,
            "degenerate": True,
        }

    merged = _merge(raw)
    intervals = [{
        "s_in": s_in, "s_out": s_out,
        "u_in": (s_in / total) if total > 0 else 0.0,
        "u_out": (s_out / total) if total > 0 else 0.0,
        "length": s_out - s_in,
    } for s_in, s_out in merged]
    collision_length = float(sum(iv["length"] for iv in intervals))
    return {
        "intervals": intervals,
        "path_length": total,
        "collision_length": collision_length,
        "collision_fraction": (collision_length / total) if total > 0 else 0.0,
        "arc": cum,
    }


def _merge(intervals: Sequence[Tuple[float, float]], gap: float = 1e-7):
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [list(ordered[0])]
    for lo, hi in ordered[1:]:
        if lo <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(float(a), float(b)) for a, b in merged]


def project_points_to_path(path_scene: np.ndarray, points_scene: np.ndarray,
                           samples: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
    """Project points onto a polyline -> ``(arc_fraction, distance_scene)``.

    The polyline is densified to ``samples`` arc-uniform stations once; each point
    is matched to its nearest station, so the returned fraction is the arc-length
    progress of the closest point on the path.
    """
    path = np.asarray(path_scene, dtype=np.float64).reshape(-1, 2)
    pts = np.asarray(points_scene, dtype=np.float64).reshape(-1, 2)
    if len(path) < 2:
        return np.zeros(len(pts)), np.full(len(pts), np.inf)
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= _EPS:
        return np.zeros(len(pts)), np.full(len(pts), np.inf)
    target = np.linspace(0.0, total, int(samples))
    dense = np.stack([np.interp(target, cum, path[:, 0]),
                      np.interp(target, cum, path[:, 1])], axis=-1)
    d = np.linalg.norm(pts[:, None, :] - dense[None, :, :], axis=-1)
    idx = np.argmin(d, axis=1)
    return target[idx] / total, d[np.arange(len(pts)), idx]


def _point_on_curve(curve_scene: np.ndarray, cum: np.ndarray, u: float) -> np.ndarray:
    s = float(np.clip(u, 0.0, 1.0)) * float(cum[-1])
    j = int(np.searchsorted(cum, s, side="right") - 1)
    j = max(0, min(j, len(curve_scene) - 2))
    span = cum[j + 1] - cum[j]
    frac = 0.0 if span <= _EPS else (s - cum[j]) / span
    return curve_scene[j] + frac * (curve_scene[j + 1] - curve_scene[j])


def collision_spans_on_path(occupancy, curve_scene, path_scene, n_query: int,
                            pad: int = 4, min_q: int = 8, max_spans: int = 3,
                            scene_scale: float = 32.0,
                            min_penetration_m: float = 0.0) -> Dict:
    """Collision intervals of ``curve_scene`` mapped onto query spans of ``path_scene``.

    Returns ``spans`` (query index ranges), the raw intervals, and the number /
    length of spans dropped by ``max_spans`` (never silently ignored).
    """
    occ_np = (occupancy.detach().cpu().numpy() if hasattr(occupancy, "detach")
              else np.asarray(occupancy))
    occ_np = np.squeeze(occ_np)
    curve = np.asarray(curve_scene, dtype=np.float64).reshape(-1, 2)
    path = np.asarray(path_scene, dtype=np.float64).reshape(-1, 2)
    info = curve_collision_intervals(occ_np, curve,
                                     min_penetration_m=0.0)
    cum = arc_lengths(curve)
    q = max(int(n_query), 2)
    spans: List[Tuple[int, int]] = []
    for iv in info["intervals"]:
        # sample the interval sparsely (endpoints + interior) and project each onto
        # the searched path; the query index follows the PROJECTED arc length
        us = np.linspace(iv["u_in"], iv["u_out"],
                         max(3, int(np.ceil((iv["u_out"] - iv["u_in"]) * 64)) + 1))
        pts = np.stack([_point_on_curve(curve, cum, float(u)) for u in us])
        frac, _dist = project_points_to_path(path, pts)
        idx = np.rint(frac * (q - 1)).astype(int)
        qa = int(idx.min()) - int(pad)
        qb = int(idx.max()) + int(pad)
        qa, qb = max(0, qa), min(q - 1, qb)
        if qb - qa + 1 < int(min_q):
            need = int(min_q) - (qb - qa + 1)
            qa = max(0, qa - need // 2)
            qb = min(q - 1, qa + int(min_q) - 1)
            qa = max(0, qb - int(min_q) + 1)
        spans.append((qa, qb))

    spans = _merge_spans(spans)
    dropped = spans[max(0, int(max_spans)):]
    kept = spans[:max(0, int(max_spans))]
    # collision length that falls outside the kept spans (metres, for the report)
    total_len = float(info["path_length"])
    dropped_length_scene = 0.0
    if dropped:
        for a, b in dropped:
            lo, hi = a / max(q - 1, 1), b / max(q - 1, 1)
            for iv in info["intervals"]:
                overlap = max(0.0, min(iv["u_out"], hi) - max(iv["u_in"], lo))
                dropped_length_scene += overlap * total_len
    return {
        "spans": kept,
        "dropped_spans": len(dropped),
        "dropped_length_m": dropped_length_scene * float(scene_scale),
        "intervals": info["intervals"],
        "collision_fraction": info["collision_fraction"],
        "collision_length_m": info["collision_length"] * float(scene_scale),
        "path_length_m": total_len * float(scene_scale),
        "n_intervals": len(info["intervals"]),
    }


def _merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not spans:
        return []
    spans = sorted(spans)
    merged = [list(spans[0])]
    for a, b in spans[1:]:
        if a <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(int(a), int(b)) for a, b in merged]
