"""Local convex safety regions and the frozen B-spline constraint pack.

Design document sections 6.4 / 7.4: ellipses are generated along the whole path,
but the expensive convex-region construction runs ONLY in the collision intervals
of the current trajectory (plus ``pad_queries`` on each side, at least
``min_queries`` wide).  Everywhere else the corridor keeps an unconstrained
"stay inside the map" box cell (four halfspaces, cannot drag the curve).

Fixes applied after the code review (see ``docs/code_review_verification.md``):

* **P0-3** the region table has ``len(cells)`` rows, not ``Q``: bridge cells are
  never truncated;
* **P0-4** tables are packed per sample and merged with padding, so each sample
  keeps its own anchors;
* **P1-1** collision intervals come from a SEGMENT-level sweep, not from
  point samples (``safety/collision_query.py``);
* **P1-6** the intervals are projected onto the searched path, so the query index
  follows the real geometry instead of a curve-index ratio;
* **P1-7** there is no cap on the number of collision intervals: every merged
  interval gets its local convex regions built;
* box cells are flagged, so the training corridor loss can ignore them.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .collision_query import collision_spans_on_path, curve_collision_intervals
from .convex_region import EllipseRegionBuilder
from .pack_utils import merge_packs
from .safety_corridor import build_safety_corridor

__all__ = ["collision_mask", "collision_intervals", "build_region_table",
           "build_local_constraint_pack", "box_cell_faces", "pad_region_tables",
           "pack_from_tables"]


# --------------------------------------------------------------------------- #
# legacy point-sampled statistics (kept for diagnostics / tests)
# --------------------------------------------------------------------------- #
def collision_mask(occupancy, curve, outside_is_blocked: bool = True
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Discrete collision per dense curve point (both in scene coordinates).

    Kept as a cheap diagnostic.  The authoritative collision test used for the
    safety geometry is the SEGMENT sweep in :mod:`collision_query`.
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
    """Legacy index-ratio mapping (kept for the unit tests of the padding rule).

    ``collision_spans_on_path`` is what the pipeline uses; this one assumes the
    curve parameter progress equals the path arc-length progress, which is only
    true when both are the same polyline.
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


# --------------------------------------------------------------------------- #
# region table
# --------------------------------------------------------------------------- #
def build_region_table(occupancy_scene: torch.Tensor,
                       centers_scene: torch.Tensor,
                       shape4_scene: torch.Tensor,
                       anchors: torch.Tensor,
                       curve_scene: Optional[torch.Tensor] = None,
                       gamma_scene: Optional[torch.Tensor] = None,
                       config: Optional[dict] = None,
                       scope: str = "local",
                       scene_scale: float = 32.0,
                       spans: Optional[List[Tuple[int, int]]] = None) -> Tuple[Dict, Dict]:
    """Padded half-space table ``A x <= b`` for one sample.

    Every argument is in the normalised scene frame -- in particular
    ``shape4_scene`` is ``[log a_s, log b_s, cos2t, sin2t]`` with the axes already
    divided by ``scene_scale`` (see ``safety_features.ellipse_shape4_to_scene``).

    ``scope="local"`` builds convex regions only inside the collision intervals of
    ``curve_scene``, projected onto ``gamma_scene`` (the searched path).
    ``scope="full"`` builds one region per ellipse (offline training cache).

    Returns ``(table, info)``; ``table`` holds ``cell_A [R,F,2]``,
    ``cell_b [R,F]``, ``cell_mask [R]``, ``cell_is_box [R]``, ``anchors [R]``
    with ``R = len(cells) = Q + bridge cells``.
    """
    cfg = dict(config or {})
    region_cfg = dict(cfg.get("region") or {})
    local_cfg = dict(cfg.get("local") or {})
    pad = int(local_cfg.get("pad_queries", 4))
    min_q = int(local_cfg.get("min_queries", 8))

    centers_scene = torch.as_tensor(centers_scene, dtype=torch.float32)
    shape4_scene = torch.as_tensor(shape4_scene, dtype=torch.float32)
    anchors = torch.as_tensor(anchors, dtype=torch.float64).reshape(-1)
    Q = int(anchors.numel())

    info: Dict = {"scope": scope, "spans": [], "local_cells": 0, "box_cells": 0,
                  "corridor_valid": [], "corridor_failure": [], "bridges": 0,
                  "collision_fraction": 0.0, "collision_intervals": 0,
                  "collision_length_m": 0.0, "num_rows": 0}

    if spans is not None:
        span_list = [tuple(int(v) for v in s) for s in spans]
    elif scope == "full":
        span_list = [(0, Q - 1)]
    else:
        span_list = []
        if curve_scene is not None and gamma_scene is not None:
            query = collision_spans_on_path(
                occupancy_scene, torch.as_tensor(curve_scene).detach().cpu().numpy(),
                torch.as_tensor(gamma_scene).detach().cpu().numpy(), Q,
                pad=pad, min_q=min_q,
                scene_scale=float(scene_scale), min_penetration_m=0.0)
            span_list = list(query["spans"])
            info.update({
                "collision_fraction": float(query["collision_fraction"]),
                "collision_intervals": int(query["n_intervals"]),
                "collision_length_m": float(query["collision_length_m"]),
            })
        elif curve_scene is not None:
            # no path to project onto: fall back to the exact segment sweep and
            # the legacy index mapping, and say so
            sweep = curve_collision_intervals(occupancy_scene, curve_scene)
            info["collision_fraction"] = float(sweep["collision_fraction"])
            info["collision_intervals"] = len(sweep["intervals"])
            info["collision_length_m"] = float(sweep["collision_length"]) * float(scene_scale)
            info["fallback"] = "index_mapping_no_path"
            spans_idx = []
            for iv in sweep["intervals"]:
                a = int(np.floor(iv["u_in"] * (Q - 1))) - pad
                b = int(np.ceil(iv["u_out"] * (Q - 1))) + pad
                spans_idx.append((max(0, a), min(Q - 1, b)))
            span_list = _merge_spans(spans_idx)
    info["spans"] = [list(s) for s in span_list]

    cells: List[Tuple[float, np.ndarray, np.ndarray, bool]] = []
    if span_list:
        builder = EllipseRegionBuilder(occupancy_scene.float(), region_cfg)
        for a, b in span_list:
            a = max(0, int(a))
            b = min(Q - 1, int(b))
            corridor = build_safety_corridor(
                builder, centers_scene[a:b + 1], shape4_scene[a:b + 1],
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
                              np.asarray(cell.b, dtype=np.float64),
                              False))     # False = not a map-box cell
        info["local_cells"] = len(cells)

    covered = [anchor for anchor, _, _, _ in cells]
    margin = float(region_cfg.get("map_boundary_margin",
                                  region_cfg.get("safety_margin", 0.02)))
    box_A, box_b = box_cell_faces(margin)
    for i in range(Q):
        s = float(anchors[i])
        if any(abs(s - c) < 1e-9 for c in covered):
            continue
        cells.append((s, box_A, box_b, True))    # True = map-box cell
    info["box_cells"] = int(sum(1 for c in cells if c[3]))

    cells.sort(key=lambda item: item[0])
    rows = len(cells)
    max_faces = max(len(item[2]) for item in cells)
    cell_A = np.zeros((rows, max_faces, 2), dtype=np.float64)
    cell_b = np.full((rows, max_faces), np.inf, dtype=np.float64)
    cell_mask = np.zeros(rows, dtype=bool)
    cell_is_box = np.zeros(rows, dtype=bool)
    cell_anchors = np.zeros(rows, dtype=np.float64)
    for i, (s, A, b, is_box) in enumerate(cells):
        cell_A[i, :len(b)] = A
        cell_b[i, :len(b)] = b
        cell_mask[i] = True
        cell_anchors[i] = s
        cell_is_box[i] = bool(is_box)

    info["num_rows"] = int(rows)
    table = {"cell_A": cell_A, "cell_b": cell_b, "cell_mask": cell_mask,
             "cell_is_box": cell_is_box, "anchors": cell_anchors}
    return table, info


# --------------------------------------------------------------------------- #
# packs
# --------------------------------------------------------------------------- #
def pack_from_tables(codec, tables: List[Dict], margin: float = 0.0,
                     include_box: bool = True, device=None):
    """One batched pack from per-sample tables (each keeps its own anchors)."""
    from .bspline_constraints import build_constraint_pack_from_regions

    packs = []
    for table in tables:
        mask = np.asarray(table["cell_mask"], dtype=bool).copy()
        if not include_box and "cell_is_box" in table:
            mask &= ~np.asarray(table["cell_is_box"], dtype=bool)
        if not mask.any():
            packs.append(None)
            continue
        packs.append(build_constraint_pack_from_regions(
            codec,
            cell_A=torch.as_tensor(table["cell_A"])[None],
            cell_b=torch.as_tensor(table["cell_b"])[None],
            cell_mask=torch.as_tensor(mask)[None],
            anchors=np.asarray(table["anchors"], dtype=np.float64),
            margin=float(margin),
            dtype=torch.float32,
            device=device))
    return merge_packs(packs, codec.num_controls, device=device)


def build_local_constraint_pack(codec, occupancy_scene: torch.Tensor,
                                centers_scene: torch.Tensor,
                                shape4_scene: torch.Tensor,
                                anchors: torch.Tensor,
                                curve_scene: torch.Tensor,
                                gamma_scene: Optional[torch.Tensor] = None,
                                config: Optional[dict] = None,
                                margin: float = 0.0,
                                scope: str = "local",
                                scene_scale: float = 32.0,
                                include_box: bool = True):
    """Table -> frozen :class:`BSplineConstraintPack` for ONE sample."""
    table, info = build_region_table(occupancy_scene, centers_scene, shape4_scene,
                                     anchors, curve_scene, gamma_scene, config,
                                     scope=scope, scene_scale=scene_scale)
    return pack_from_tables(codec, [table], margin=margin,
                            include_box=include_box), info


def pad_region_tables(tables: List[Dict]) -> Dict[str, torch.Tensor]:
    """Stack per-sample tables into batch tensors (rows/faces padded, never used)."""
    max_rows = max(t["cell_A"].shape[0] for t in tables)
    max_faces = max(t["cell_A"].shape[1] for t in tables)
    batch = len(tables)
    cell_A = torch.zeros(batch, max_rows, max_faces, 2, dtype=torch.float64)
    cell_b = torch.full((batch, max_rows, max_faces), float("inf"), dtype=torch.float64)
    cell_mask = torch.zeros(batch, max_rows, dtype=torch.bool)
    cell_is_box = torch.zeros(batch, max_rows, dtype=torch.bool)
    anchors = torch.zeros(batch, max_rows, dtype=torch.float64)
    for i, table in enumerate(tables):
        rows = table["cell_A"].shape[0]
        faces = table["cell_A"].shape[1]
        cell_A[i, :rows, :faces] = torch.as_tensor(table["cell_A"])
        cell_b[i, :rows, :faces] = torch.as_tensor(table["cell_b"])
        cell_mask[i, :rows] = torch.as_tensor(table["cell_mask"]).bool()
        if "cell_is_box" in table:
            cell_is_box[i, :rows] = torch.as_tensor(table["cell_is_box"]).bool()
        anchors[i, :rows] = torch.as_tensor(table["anchors"])
    return {"cell_A": cell_A, "cell_b": cell_b, "cell_mask": cell_mask,
            "cell_is_box": cell_is_box, "anchors": anchors}
