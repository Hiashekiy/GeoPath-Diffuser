"""Local convex safety regions and the frozen B-spline constraint pack.

Design document sections 6.4 / 7.4: ellipses are generated along the whole
path, but the expensive convex-region construction runs ONLY in the collision
intervals of the current trajectory (plus ``pad_queries`` on each side, at least
``min_queries`` wide).  Everywhere else the corridor keeps an unconstrained
"stay inside the map" box cell, which costs four halfspaces and cannot drag the
curve anywhere.

The construction itself is the migrated upstream code, used unchanged:
``EllipseRegionBuilder.build_from_ellipse`` -> ``build_safety_corridor``
(overlap check + point-seeded bridge) -> ``build_constraint_pack_from_regions``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .convex_region import EllipseRegionBuilder
from .safety_corridor import build_safety_corridor

__all__ = ["collision_mask", "collision_intervals", "build_region_table",
           "build_local_constraint_pack", "box_cell_faces", "pad_region_tables"]


def collision_mask(occupancy, curve, outside_is_blocked: bool = True
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Discrete collision per dense curve point (both in scene coordinates).

    Copied from TrajSafe-Diffuser ``src/diffusion/alm_local.py``: a verification
    statistic, never a projection.
    """
    occ = (occupancy.detach().cpu().numpy() if torch.is_tensor(occupancy)
           else np.asarray(occupancy))
    occ = np.squeeze(occ)
    if occ.ndim != 2:
        raise ValueError("occupancy must reduce to [H,W], got %s" % (occ.shape,))
    curve = (curve.detach().cpu().numpy() if torch.is_tensor(curve)
             else np.asarray(curve))
    curve = curve.reshape(-1, 2)
    res = occ.shape[-1]
    px = np.rint((curve[:, 0] + 1.0) / 2.0 * res - 0.5).astype(int)
    py = np.rint((curve[:, 1] + 1.0) / 2.0 * res - 0.5).astype(int)
    ok = (px >= 0) & (px < res) & (py >= 0) & (py < res)
    hit = (np.ones(len(curve), dtype=bool) if outside_is_blocked
           else np.zeros(len(curve), dtype=bool))
    hit[ok] = occ[py[ok], px[ok]].astype(bool)
    return hit, (~ok)


def collision_intervals(hit: np.ndarray, pad: int = 4, min_q: int = 8,
                        n_query: int = 64) -> List[Tuple[int, int]]:
    """Contiguous collision runs mapped to ellipse-query index spans.

    Same padding / minimum-width rule as the reference sampler
    (``src/diffusion/alm_local.py``).
    """
    idx = np.flatnonzero(hit)
    if len(idx) == 0:
        return []
    length = len(hit)
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i != prev + 1:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    q = max(int(n_query), 2)
    spans = []
    for a, b in runs:
        qa = int(np.floor(a / max(length - 1, 1) * (q - 1))) - int(pad)
        qb = int(np.ceil(b / max(length - 1, 1) * (q - 1))) + int(pad)
        spans.append((max(0, qa), min(q - 1, qb)))
    widened = []
    for a, b in spans:
        width = b - a + 1
        if width < int(min_q):
            need = int(min_q) - width
            a = max(0, a - need // 2)
            b = min(q - 1, a + int(min_q) - 1)
            a = max(0, b - int(min_q) + 1)
        widened.append((a, b))
    return _merge_spans(widened)


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


def box_cell_faces(margin: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """The unconstrained cell used outside collision intervals: the map box."""
    inset = 1.0 - float(margin)
    A = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], dtype=np.float64)
    b = np.full(4, inset, dtype=np.float64)
    return A, b


def build_region_table(occupancy_scene: torch.Tensor,
                       centers_scene: torch.Tensor,
                       shape4: torch.Tensor,
                       anchors: torch.Tensor,
                       curve_scene: Optional[torch.Tensor] = None,
                       gamma_scene: Optional[torch.Tensor] = None,
                       config: Optional[dict] = None,
                       scope: str = "local"):
    """Padded half-space table ``A x <= b`` for one sample.

    ``scope="local"`` builds local convex regions only inside the collision
    intervals of ``curve_scene`` (inference, design document section 6.4).
    ``scope="full"`` builds one region per ellipse (offline training cache: the
    GT curve is collision-free by construction, so a local scope would produce
    an empty constraint set and a dead safety loss).

    Returns ``(table, info)`` with ``table = {"cell_A": [R,F,2], "cell_b": [R,F],
    "cell_mask": [R], "anchors": [R]}``.
    """
    cfg = dict(config or {})
    region_cfg = dict(cfg.get("region") or {})
    local_cfg = dict(cfg.get("local") or {})
    pad = int(local_cfg.get("pad_queries", 4))
    min_q = int(local_cfg.get("min_queries", 8))
    max_spans = int(local_cfg.get("max_spans", 3))

    centers_scene = torch.as_tensor(centers_scene, dtype=torch.float32)
    shape4 = torch.as_tensor(shape4, dtype=torch.float32)
    anchors = torch.as_tensor(anchors, dtype=torch.float64).reshape(-1)
    Q = int(anchors.numel())

    info: Dict = {"scope": scope, "spans": [], "local_cells": 0, "box_cells": 0,
                  "corridor_valid": [], "corridor_failure": [], "bridges": 0,
                  "collision_fraction": 0.0}

    if scope == "full":
        spans = [(0, Q - 1)]
    else:
        if curve_scene is None:
            spans = []
        else:
            hit, _ = collision_mask(occupancy_scene, curve_scene)
            info["collision_fraction"] = float(hit.mean()) if len(hit) else 0.0
            spans = collision_intervals(hit, pad=pad, min_q=min_q, n_query=Q)[:max_spans]
    info["spans"] = [list(s) for s in spans]

    cells: List[Tuple[float, np.ndarray, np.ndarray]] = []
    if spans:
        builder = EllipseRegionBuilder(occupancy_scene.float(), region_cfg)
        for a, b in spans:
            corridor = build_safety_corridor(
                builder, centers_scene[a:b + 1], shape4[a:b + 1],
                anchors[a:b + 1], gamma=gamma_scene,
                gamma_lengths=(int(gamma_scene.shape[0]) if gamma_scene is not None else None),
                config=cfg)
            info["corridor_valid"].append(bool(corridor.valid))
            info["corridor_failure"].append(corridor.failure_reason)
            if not corridor.valid:
                continue
            info["bridges"] += int(corridor.bridge_cell_count)
            for cell in corridor.cells:
                cells.append((float(cell.anchor_s),
                              np.asarray(cell.A, dtype=np.float64),
                              np.asarray(cell.b, dtype=np.float64)))
        info["local_cells"] = len(cells)

    covered = [anchor for anchor, _, _ in cells]
    margin = float(region_cfg.get("map_boundary_margin",
                                 region_cfg.get("safety_margin", 0.02)))
    box_A, box_b = box_cell_faces(margin)
    for i in range(Q):
        s = float(anchors[i])
        if any(abs(s - c) < 1e-9 for c in covered):
            continue
        cells.append((s, box_A, box_b))
    info["box_cells"] = Q - info["local_cells"]

    cells.sort(key=lambda item: item[0])
    max_faces = max(len(b) for _, _, b in cells)
    cell_A = np.zeros((Q, max_faces, 2), dtype=np.float64)
    cell_b = np.full((Q, max_faces), np.inf, dtype=np.float64)
    cell_mask = np.zeros(Q, dtype=bool)
    cell_anchors = np.zeros(Q, dtype=np.float64)
    for i, (s, A, b) in enumerate(cells[:Q]):
        cell_A[i, :len(b)] = A
        cell_b[i, :len(b)] = b
        cell_mask[i] = True
        cell_anchors[i] = s

    table = {"cell_A": cell_A, "cell_b": cell_b, "cell_mask": cell_mask,
             "anchors": cell_anchors}
    return table, info


def build_local_constraint_pack(codec, occupancy_scene: torch.Tensor,
                                centers_scene: torch.Tensor,
                                shape4: torch.Tensor,
                                anchors: torch.Tensor,
                                curve_scene: torch.Tensor,
                                gamma_scene: Optional[torch.Tensor] = None,
                                config: Optional[dict] = None,
                                margin: float = 0.0,
                                scope: str = "local"):
    """Table -> frozen :class:`BSplineConstraintPack` for ONE sample."""
    from .bspline_constraints import build_constraint_pack_from_regions

    table, info = build_region_table(occupancy_scene, centers_scene, shape4,
                                     anchors, curve_scene, gamma_scene,
                                     config, scope=scope)
    return pack_from_tables(codec, [table], margin=margin), info


def pack_from_tables(codec, tables: List[Dict], margin: float = 0.0):
    """Build ONE batched pack from per-sample tables (pad faces with ``A=0, b=inf``)."""
    from .bspline_constraints import build_constraint_pack_from_regions

    max_faces = max(t["cell_A"].shape[1] for t in tables)
    batch = len(tables)
    R = tables[0]["cell_A"].shape[0]
    cell_A = np.zeros((batch, R, max_faces, 2), dtype=np.float64)
    cell_b = np.full((batch, R, max_faces), np.inf, dtype=np.float64)
    cell_mask = np.zeros((batch, R), dtype=bool)
    anchors = np.zeros(R, dtype=np.float64)
    for i, table in enumerate(tables):
        faces = table["cell_A"].shape[1]
        cell_A[i, :, :faces] = table["cell_A"]
        cell_b[i, :, :faces] = table["cell_b"]
        cell_mask[i] = table["cell_mask"]
        anchors = np.asarray(table["anchors"], dtype=np.float64)
    return build_constraint_pack_from_regions(
        codec,
        cell_A=torch.as_tensor(cell_A),
        cell_b=torch.as_tensor(cell_b),
        cell_mask=torch.as_tensor(cell_mask),
        anchors=anchors,
        margin=float(margin),
        dtype=torch.float32)


def pad_region_tables(tables: List[Dict]) -> Dict[str, torch.Tensor]:
    """Stack per-sample tables into batch tensors (faces padded with ``b=+inf``)."""
    max_faces = max(t["cell_A"].shape[1] for t in tables)
    batch = len(tables)
    R = tables[0]["cell_A"].shape[0]
    cell_A = torch.zeros(batch, R, max_faces, 2, dtype=torch.float64)
    cell_b = torch.full((batch, R, max_faces), float("inf"), dtype=torch.float64)
    cell_mask = torch.zeros(batch, R, dtype=torch.bool)
    anchors = torch.zeros(batch, R, dtype=torch.float64)
    for i, table in enumerate(tables):
        faces = table["cell_A"].shape[1]
        cell_A[i, :, :faces] = torch.as_tensor(table["cell_A"])
        cell_b[i, :, :faces] = torch.as_tensor(table["cell_b"])
        cell_mask[i] = torch.as_tensor(table["cell_mask"]).bool()
        anchors[i] = torch.as_tensor(table["anchors"])
    return {"cell_A": cell_A, "cell_b": cell_b, "cell_mask": cell_mask,
            "anchors": anchors}
