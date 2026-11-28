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
    """Collision rate, clearance, length, smoothness of one trajectory.

    The collision rate is computed with the SEGMENT sweep
    (``safety.collision_query.curve_collision_intervals``) on a 0.05 m densified
    curve: a point-only test misses a 0.25 m wall between two samples, which is
    exactly the defect P1-1 of the code review.
    """
    from ..frames import SceneTransform
    from ..safety.collision_query import curve_collision_intervals

    pts = np.asarray(points_m, dtype=np.float64).reshape(-1, 2)
    dense = dense_points(pts, step_m)
    occ = np.asarray(occupancy)
    if occ.ndim == 3:
        occ = occ[0]
    grid = int(occ.shape[-1])
    transform = SceneTransform(float(grid) * float(res_m), grid)
    sweep = curve_collision_intervals(occ, transform.to_scene(dense))
    blocked = np.zeros(len(dense), dtype=bool)
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
        "collision_rate": float(sweep["collision_fraction"]),
        "collision_free": bool(sweep["collision_fraction"] <= 1e-9),
        "collision_intervals": int(len(sweep["intervals"])),
        "min_clearance_m": float(clearance.min()),
        "mean_clearance_m": float(clearance.mean()),
        "length_m": length,
        "mean_curvature": float(curvature.mean()) if len(curvature) else 0.0,
        "max_curvature": float(curvature.max()) if len(curvature) else 0.0,
        "end_to_end_distance_m": float(np.linalg.norm(pts[0] - pts[-1])) if len(pts) > 1 else 0.0,
        "num_dense_samples": int(len(dense)),
    }


def corridor_violation_m(points_m: np.ndarray, pack, scene_scale: float,
                         params: Optional[np.ndarray] = None,
                         offset: float = 0.0) -> float:
    """Largest half-space violation of the frozen constraints along a curve.

    Two pack flavours are understood:

    * :class:`~gp_diffuser.safety.bspline_sample_constraints.BSplineSampleConstraintPack`
      (the inference pack): every point is judged by the cell of the SAMPLE whose
      arc-length progress is closest to its own progress.  A sample point owns
      exactly one cell, so this is the same "responsibility" rule the guidance and
      the QP use - never "test every cell against every point".
    * the historical Bezier pack (``piece_mask`` / ``intervals``): a point is only
      tested against the piece whose parameter interval contains it.

    ``SceneTransform.to_scene`` is ``x / scene_scale + offset`` (``offset =
    -1 + 1/grid``), so ``offset`` MUST be passed: dropping it shifts every test
    point by one whole scene unit (32 m here) and fabricates a huge violation.
    """
    if pack is None:
        return 0.0
    if hasattr(pack, "sample_mask"):                       # sampled pack
        return _sampled_violation_m(points_m, pack, scene_scale, params, offset)
    if pack.piece_mask.shape[1] == 0:
        return 0.0
    pts_scene = (np.asarray(points_m, dtype=np.float64).reshape(-1, 2)
                 / float(scene_scale) + float(offset))
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


def _sampled_violation_m(points_m, pack, scene_scale: float,
                         params: Optional[np.ndarray] = None,
                         offset: float = 0.0) -> float:
    """Violation of a curve against the sampled pack (one cell per sample point).

    Each curve point inherits the cell of the pack sample whose FROZEN parameter
    ``u_l`` is closest to its own, which is the same responsibility rule CPCG and
    the QP use.  It is a diagnostic, not a certificate.
    """
    pts_scene = (np.asarray(points_m, dtype=np.float64).reshape(-1, 2)
                 / float(scene_scale) + float(offset))
    n = len(pts_scene)
    if n == 0 or pack.sample_mask.shape[1] == 0:
        return 0.0
    u = (np.linspace(0.0, 1.0, n) if params is None
         else np.asarray(params, dtype=np.float64).reshape(-1))
    worst = 0.0
    for sample in range(pack.sample_mask.shape[0]):
        smask = pack.sample_mask[sample].detach().cpu().numpy()
        if not smask.any():
            continue
        sample_u = pack.sample_u[sample].detach().cpu().numpy()
        face = pack.face_mask[sample].detach().cpu().numpy()
        A = pack.sample_A[sample].detach().cpu().numpy()
        b = pack.sample_b[sample].detach().cpu().numpy()
        owners = np.flatnonzero(smask)
        nearest = owners[np.argmin(np.abs(sample_u[owners][None, :] - u[:, None]),
                                   axis=1)]
        for point, l in zip(pts_scene, nearest):
            faces = face[l]
            if not faces.any():
                continue
            g = A[l][faces] @ point - b[l][faces]
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
