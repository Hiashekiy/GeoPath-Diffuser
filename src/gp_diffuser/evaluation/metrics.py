"""Evaluation metrics (design document section 14, main indicator list).

Everything is computed in metres from the occupancy grid / SDF so the numbers are
comparable across the ablation configurations.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch


def dense_points(points_m: np.ndarray, step_m: float = 0.05) -> np.ndarray:
    """Arc-length densification of a polyline (metres)."""
    pts = np.asarray(points_m, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    out = [pts[0]]
    for i, length in enumerate(seg):
        n = max(1, int(np.ceil(length / max(step_m, 1e-6))))
        for k in range(1, n + 1):
            out.append(pts[i] + (k / n) * (pts[i + 1] - pts[i]))
    return np.asarray(out)


def sample_sdf(sdf: np.ndarray, points_m: np.ndarray, res_m: float) -> np.ndarray:
    """Bilinear SDF lookup in metres, outside the map counts as blocked."""
    field = np.asarray(sdf, dtype=np.float64)
    if field.ndim == 3:
        field = field[0]
    height, width = field.shape
    cols = points_m[:, 0] / res_m
    rows = points_m[:, 1] / res_m
    inside = (cols >= 0) & (cols <= width - 1) & (rows >= 0) & (rows <= height - 1)
    out = np.full(len(points_m), -1.0, dtype=np.float64)
    r0 = np.clip(np.floor(rows).astype(int), 0, height - 1)
    c0 = np.clip(np.floor(cols).astype(int), 0, width - 1)
    r1 = np.clip(r0 + 1, 0, height - 1)
    c1 = np.clip(c0 + 1, 0, width - 1)
    fr = np.clip(rows - r0, 0, 1)
    fc = np.clip(cols - c0, 0, 1)
    value = (field[r0, c0] * (1 - fr) * (1 - fc) + field[r1, c0] * fr * (1 - fc)
             + field[r0, c1] * (1 - fr) * fc + field[r1, c1] * fr * fc)
    out[inside] = value[inside]
    return out


def trajectory_metrics(points_m: np.ndarray, occupancy: np.ndarray, sdf: np.ndarray,
                       res_m: float, step_m: float = 0.05) -> Dict[str, float]:
    """Collision rate, clearance, length, smoothness of one trajectory."""
    pts = np.asarray(points_m, dtype=np.float64).reshape(-1, 2)
    dense = dense_points(pts, step_m)
    occ = np.asarray(occupancy)
    if occ.ndim == 3:
        occ = occ[0]
    height, width = occ.shape
    rows = np.rint(dense[:, 1] / res_m).astype(int)
    cols = np.rint(dense[:, 0] / res_m).astype(int)
    outside = (rows < 0) | (rows >= height) | (cols < 0) | (cols >= width)
    blocked = np.ones(len(dense), dtype=bool)
    inside = ~outside
    blocked[inside] = occ[rows[inside], cols[inside]] > 0.5
    clearance = sample_sdf(sdf, dense, res_m)

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    length = float(seg.sum())
    # curvature on the duplicate-free polyline (a B-spline with clustered
    # controls can decode to repeated points; those carry no direction)
    keep = np.concatenate([[True], seg > 1e-6])
    clean = pts[keep]
    if len(clean) >= 3:
        v1 = clean[1:-1] - clean[:-2]
        v2 = clean[2:] - clean[1:-1]
        n1 = np.linalg.norm(v1, axis=1)
        n2 = np.linalg.norm(v2, axis=1)
        cosang = np.clip((v1 * v2).sum(axis=1) / (n1 * n2).clip(min=1e-12), -1.0, 1.0)
        angle = np.arccos(cosang)
        curvature = angle / (0.5 * (n1 + n2)).clip(min=1e-6)
        curvature = curvature[np.isfinite(curvature)]
    else:
        curvature = np.zeros(0)
    return {
        "collision_rate": float(blocked.mean()),
        "collision_free": bool(not blocked.any()),
        "min_clearance_m": float(clearance.min()),
        "mean_clearance_m": float(clearance.mean()),
        "length_m": length,
        "mean_curvature": float(curvature.mean()) if len(curvature) else 0.0,
        "max_curvature": float(curvature.max()) if len(curvature) else 0.0,
        "end_to_end_distance_m": float(np.linalg.norm(pts[0] - pts[-1])) if len(pts) > 1 else 0.0,
        "num_dense_samples": int(len(dense)),
    }


def corridor_violation_m(points_m: np.ndarray, pack, scene_scale: float,
                         params: Optional[np.ndarray] = None) -> float:
    """Largest half-space violation of the frozen corridor along the trajectory.

    Each piece of the pack owns one parameter interval of the curve, so a point is
    only tested against the faces of ITS piece (testing every piece against every
    point would report meaningless "violations" from far-away cells).
    """
    if pack is None or pack.piece_mask.shape[1] == 0:
        return 0.0
    pts_scene = np.asarray(points_m, dtype=np.float64).reshape(-1, 2) / float(scene_scale)
    n = len(pts_scene)
    if n == 0:
        return 0.0
    u = np.linspace(0.0, 1.0, n) if params is None else np.asarray(params, dtype=np.float64)
    intervals = pack.intervals.detach().cpu().numpy()
    piece_mask = pack.piece_mask.detach().cpu().numpy()
    face_mask = pack.face_mask.detach().cpu().numpy()
    A = pack.piece_A.detach().cpu().numpy()
    b = pack.piece_b.detach().cpu().numpy()
    worst = 0.0
    for sample in range(pack.piece_mask.shape[0]):
        for piece in range(pack.piece_mask.shape[1]):
            if not piece_mask[sample, piece]:
                continue
            lo, hi = float(intervals[sample, piece, 0]), float(intervals[sample, piece, 1])
            select = (u >= lo - 1e-9) & (u <= hi + 1e-9)
            if not select.any():
                continue
            faces = face_mask[sample, piece]
            if not faces.any():
                continue
            g = pts_scene[select] @ A[sample, piece][faces].T - b[sample, piece][faces]
            worst = max(worst, float(np.max(g)))
    return worst * float(scene_scale)


def aggregate(rows: List[Dict[str, float]]) -> Dict[str, float]:
    """Mean of every numeric key over a list of per-sample metric dicts."""
    if not rows:
        return {}
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float)) and not isinstance(rows[0][k], bool)]
    out = {}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row]
        out[key] = float(np.mean(values)) if values else 0.0
    out["num_samples"] = len(rows)
    out["success_rate"] = float(np.mean([1.0 if row.get("collision_free", False) else 0.0
                                         for row in rows]))
    return out
