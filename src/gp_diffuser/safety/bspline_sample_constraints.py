"""Uniform-arc-length sampled B-spline constraint pack (inference CPCG / QP).

The only optimisation variable of the inference-time safety correction is the
GLOBAL B-spline control polygon ``Q in R^{C x 2}``::

    P(u) = N(u) Q

The constraints are evaluated on ``M`` curve points sampled once, at uniform ARC
LENGTH along the FROZEN reference curve ``P_ref = N(u) Q_ref`` (the clean
prediction that generated the safety geometry):

    1. u_grid = linspace(0, 1, L)              L = arc_length_lookup_points
    2. P_grid = N(u_grid) Q_ref, ell_grid = cumulative polyline length
    3. s_grid = ell_grid / ell_grid[-1]
    4. s_l = l / (M - 1),  u_l = interp(s_l, s_grid, u_grid)      l = 0..M-1
    5. N_l = N(u_l)                            frozen for the whole run

Because ``u_l`` is frozen, every constraint stays LINEAR in ``Q`` even though the
sample points are equidistant in arc length.  Re-sampling per denoising step
would make the constraint matrix depend on the current ``Q`` and destroy that
property, so it is forbidden.

Each sample point is assigned to EXACTLY ONE convex cell of the frozen corridor,
by its arc-length progress ``s_l`` and the cell anchors ``s_j`` (bridge cells
included)::

    tau_0 = 0,  tau_R = 1,  tau_j = (s_j + s_{j+1}) / 2
    j(l) = clip(searchsorted(tau, s_l, side="right") - 1, 0, R - 1)

Region membership is therefore decided ONCE, by the frozen anchors and the frozen
``s_l`` - never by the (moving) position of the optimised sample point.

The resulting linear constraint is

    A_{j(l)} N_l Q <= b_{j(l)}

which CPCG (squared hinge) and the terminal QP both consume from the SAME pack.

Deliberately NOT here: Bezier extraction, Bezier control points, knot insertion,
cutting planes, ALM.  The pack constrains ``M`` points of the real curve; it is
NOT a continuous certificate - the independent dense collision validation in the
sampler stays the final authority.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..trajectory.bspline import BSplineCodec, bspline_basis_matrix
from .bspline_constraints import responsibility_intervals

__all__ = [
    "BSplineSampleConstraintPack",
    "uniform_arc_length_params",
    "max_parameter_gap",
    "needs_uniform_fallback",
    "assign_sample_regions",
    "halfspace_tables_from_cells",
    "region_polygons",
    "sampled_points",
    "build_sampled_constraint_pack",
    "build_sampled_constraint_pack_from_tables",
    "sample_pack_is_empty",
    "empty_sample_pack",
]

#: a reference curve shorter than this (scene units) has no usable arc length
MIN_CURVE_LENGTH = 1e-6

#: If the arc-length inversion leaves a gap wider than this many sample spacings,
#: the reference curve is so unevenly parameterised (its length is concentrated in
#: a small part of ``u``, e.g. an early denoising prediction that is still a
#: collapsed spike) that most of the PARAMETER range - and therefore most of the
#: curve at the later steps - receives no constraint sample at all.  Example seen
#: on a real scene: 109 samples in ``u in [0, 0.1]``, 147 in ``u in [0.9, 1]`` and
#: ZERO in between.  Such a pack is not "uniform arc-length sampling", it is a
#: hole, so the sampler falls back to uniform-u (recorded in the diagnostics).
MAX_U_GAP_FACTOR = 8.0

#: default number of constraint sample points (``sampling.constraint_points``)
DEFAULT_CONSTRAINT_POINTS = 256
#: default resolution of the arc-length lookup table
DEFAULT_LOOKUP_POINTS = 1024


# ---------------------------------------------------------------------------
# basis rows (cached: the lookup table basis is the expensive part)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=32)
def _cached_basis(knots_bytes: bytes, num_controls: int, degree: int,
                  params_bytes: bytes) -> np.ndarray:
    knots = np.frombuffer(knots_bytes, dtype=np.float64).copy()
    params = np.frombuffer(params_bytes, dtype=np.float64).copy()
    return bspline_basis_matrix(knots, num_controls, degree, params)


def sample_basis_rows(codec: BSplineCodec, params) -> np.ndarray:
    """``N(u)`` -> ``[len(params), C]`` as float64 numpy (cached per parameter set)."""
    values = np.ascontiguousarray(np.asarray(params, dtype=np.float64).reshape(-1))
    knots = np.ascontiguousarray(np.asarray(codec._knots64, dtype=np.float64))
    return _cached_basis(knots.tobytes(), int(codec.num_controls),
                         int(codec.degree), values.tobytes())


def _as_numpy(value, dtype=np.float64) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


# ---------------------------------------------------------------------------
# 1. uniform arc length sampling of the reference curve
# ---------------------------------------------------------------------------


def max_parameter_gap(u_l) -> float:
    """Largest ``u`` jump between consecutive samples (1.0 for ``M < 2``)."""
    u = np.asarray(u_l, dtype=np.float64).reshape(-1)
    if u.size < 2:
        return 1.0
    return float(np.diff(u).max())


def needs_uniform_fallback(u_l) -> bool:
    """True when the arc-length sampling would leave most of the curve uncovered.

    ``u_l`` is expected to be ``M`` sorted parameters of an equal-arc-length
    sampling.  A gap wider than ``MAX_U_GAP_FACTOR`` sample spacings means the
    length of the reference curve is concentrated in a tiny part of the parameter
    range, so the samples bunch up there and the rest of the curve gets no
    constraint at all.
    """
    u = np.asarray(u_l, dtype=np.float64).reshape(-1)
    M = int(u.size)
    if M < 2:
        return True
    return max_parameter_gap(u) > MAX_U_GAP_FACTOR / float(M - 1)


def uniform_arc_length_params(codec: BSplineCodec, reference_controls,
                              num_samples: int = DEFAULT_CONSTRAINT_POINTS,
                              lookup_points: int = DEFAULT_LOOKUP_POINTS) -> Dict:
    """Frozen ``(u_l, s_l, N_l)`` of ONE reference control polygon.

    ``reference_controls`` is ``[C,2]`` (numpy / torch, any device).  Returns a
    dict with ``u [M]``, ``s [M]``, ``basis [M,C]``, ``length``, ``fallback`` and
    ``fallback_reason``.

    Two degenerate cases fall back to ``u_l = s_l = linspace(0,1,M)``:

    ``degenerate_reference_curve``  the reference curve is (numerically) a point,
                                    so no arc-length ordering exists;
    ``skewed_arc_length``           the arc length is concentrated in a small part
                                    of ``u`` (see :data:`MAX_U_GAP_FACTOR`), so the
                                    equal-arc-length samples would leave the rest
                                    of the curve unconstrained.

    Both cases are recorded instead of silently pretending the samples are
    equidistant along a meaningful curve.
    """
    M = max(int(num_samples), 2)
    q = _as_numpy(reference_controls).reshape(-1, 2)
    C = int(codec.num_controls)
    if q.shape[0] != C:
        raise ValueError("reference_controls must be [%d,2], got %s"
                         % (C, (q.shape,)))
    L = max(int(lookup_points), 2)
    u_grid = np.linspace(0.0, 1.0, L)
    points = sample_basis_rows(codec, u_grid) @ q                      # [L,2]
    segment = np.linalg.norm(np.diff(points, axis=0), axis=-1)
    ell = np.concatenate([[0.0], np.cumsum(segment)])
    total = float(ell[-1])
    s_l = np.linspace(0.0, 1.0, M)

    fallback_reason = None
    raw_gap = 1.0
    if (not np.isfinite(total)) or total <= MIN_CURVE_LENGTH:
        fallback_reason = "degenerate_reference_curve"
        raw = s_l.copy()
    else:
        s_grid = ell / total
        # np.interp needs a strictly increasing abscissa: a stationary piece of
        # the reference curve repeats the same arc-length progress
        keep = np.concatenate([[True], np.diff(s_grid) > 0.0])
        raw = np.interp(s_l, s_grid[keep], u_grid[keep])
        raw = np.maximum.accumulate(np.clip(raw, 0.0, 1.0))
        raw[0], raw[-1] = 0.0, 1.0
        raw_gap = max_parameter_gap(raw)
        if needs_uniform_fallback(raw):
            fallback_reason = "skewed_arc_length"

    u_l = s_l.copy() if fallback_reason else raw
    u_l = np.maximum.accumulate(np.clip(u_l, 0.0, 1.0))
    u_l[0], u_l[-1] = 0.0, 1.0

    return {"u": u_l, "s": s_l, "basis": sample_basis_rows(codec, u_l),
            "length": total, "fallback": fallback_reason is not None,
            "fallback_reason": fallback_reason,
            "raw_max_u_gap": raw_gap,
            "num_samples": M}


def assign_sample_regions(tau, sample_s, num_regions: int) -> np.ndarray:
    """``j(l) = clip(searchsorted(tau, s_l, side='right') - 1, 0, R-1)``.

    A point exactly ON an interior boundary belongs to the region on its RIGHT
    (``side='right'``); ``s = 0`` and ``s = 1`` fall into the first / last
    region after the clip.  Every sample gets exactly one region id.
    """
    tau = np.asarray(tau, dtype=np.float64).reshape(-1)
    s = np.asarray(sample_s, dtype=np.float64).reshape(-1)
    idx = np.searchsorted(tau, s, side="right") - 1
    return np.clip(idx, 0, max(int(num_regions) - 1, 0)).astype(np.int64)


# ---------------------------------------------------------------------------
# 2. half-space tables of the cells
# ---------------------------------------------------------------------------


def halfspace_tables_from_cells(cells: Sequence) -> Tuple[np.ndarray, np.ndarray,
                                                          np.ndarray, np.ndarray]:
    """Padded ``(A [R,F,2], b [R,F], face_mask [R,F], is_box [R])`` from cells.

    ``cells`` are objects with ``.A`` / ``.b`` (``CorridorCell``) or 4-tuples
    ``(anchor_s, A, b, is_box)``.  ``is_box`` marks the map-box cells of
    ``legacy_local``, which the local pack can ignore.
    """
    rows = list(cells)
    R = len(rows)
    faces = [len(_as_numpy(cell.b).reshape(-1)) if hasattr(cell, "A")
             else len(_as_numpy(cell[2]).reshape(-1)) for cell in rows]
    fmax = max(faces, default=0)
    A = np.zeros((R, fmax, 2), dtype=np.float64)
    b = np.full((R, fmax), np.inf, dtype=np.float64)
    mask = np.zeros((R, fmax), dtype=bool)
    is_box = np.zeros(R, dtype=bool)
    for i, cell in enumerate(rows):
        if hasattr(cell, "A"):
            cell_A = _as_numpy(cell.A).reshape(-1, 2)
            cell_b = _as_numpy(cell.b).reshape(-1)
            box = False
        else:
            cell_A = _as_numpy(cell[1]).reshape(-1, 2)
            cell_b = _as_numpy(cell[2]).reshape(-1)
            box = bool(cell[3])
        f = int(min(len(cell_b), cell_A.shape[0]))
        if f == 0:
            continue
        finite = np.isfinite(cell_b[:f])
        A[i, :f] = cell_A[:f]
        b[i, :f] = cell_b[:f]
        mask[i, :f] = finite
        is_box[i] = box
    return A, b, mask, is_box


# ---------------------------------------------------------------------------
# 3. pack
# ---------------------------------------------------------------------------


@dataclass
class BSplineSampleConstraintPack:
    """Frozen constraint pack on ``M`` uniform-arc-length curve samples.

    ``A_{j(l)} N_l Q <= b_{j(l)}`` for every valid sample ``l`` and face ``f``.
    """

    basis: torch.Tensor            # [B,M,C]   frozen N(u_l)
    sample_s: torch.Tensor         # [B,M]     uniform arc-length progress
    sample_u: torch.Tensor         # [B,M]     parameter recovered from Q_ref
    sample_A: torch.Tensor         # [B,M,F,2] normals of the assigned cell
    sample_b: torch.Tensor         # [B,M,F]   bounds of the assigned cell
    face_mask: torch.Tensor        # [B,M,F]   bool
    sample_mask: torch.Tensor      # [B,M]     bool
    region_id: torch.Tensor        # [B,M]     long, unique region per sample
    sample_is_box: torch.Tensor    # [B,M]     bool, map-box cell flag
    diagnostics: Dict = field(default_factory=dict)

    # ------------------------------------------------------------------ shape
    @property
    def num_controls(self) -> int:
        return int(self.basis.shape[-1])

    @property
    def num_samples(self) -> int:
        return int(self.basis.shape[1])

    @property
    def num_faces(self) -> int:
        return int(self.sample_A.shape[2])

    @property
    def num_constraint_rows(self) -> int:
        """Number of ACTIVE scalar inequalities ``sample x face``."""
        return int((self.sample_mask[:, :, None] & self.face_mask).sum())

    # ---------------------------------------------------------------- convert
    def to(self, device=None, dtype=None) -> "BSplineSampleConstraintPack":
        def cast(x):
            if x.dtype.is_floating_point:
                return x.to(device=device, dtype=dtype)
            return x.to(device=device)

        return BSplineSampleConstraintPack(
            basis=cast(self.basis), sample_s=cast(self.sample_s),
            sample_u=cast(self.sample_u), sample_A=cast(self.sample_A),
            sample_b=cast(self.sample_b), face_mask=cast(self.face_mask),
            sample_mask=cast(self.sample_mask), region_id=cast(self.region_id),
            sample_is_box=cast(self.sample_is_box), diagnostics=self.diagnostics)

    def index_select(self, rows: torch.Tensor) -> "BSplineSampleConstraintPack":
        return BSplineSampleConstraintPack(
            basis=self.basis.index_select(0, rows),
            sample_s=self.sample_s.index_select(0, rows),
            sample_u=self.sample_u.index_select(0, rows),
            sample_A=self.sample_A.index_select(0, rows),
            sample_b=self.sample_b.index_select(0, rows),
            face_mask=self.face_mask.index_select(0, rows),
            sample_mask=self.sample_mask.index_select(0, rows),
            region_id=self.region_id.index_select(0, rows),
            sample_is_box=self.sample_is_box.index_select(0, rows),
            diagnostics=self.diagnostics)

    def summary(self) -> Dict:
        diag = self.diagnostics or {}
        return {
            "num_constraint_samples": int(self.num_samples),
            "num_samples_valid": int(self.sample_mask.sum()),
            "num_constraint_rows": self.num_constraint_rows,
            "num_faces_max": self.num_faces,
            "num_controls": self.num_controls,
            "arc_length_fallback": diag.get("arc_length_fallback"),
            "arc_fallback_reason": diag.get("arc_fallback_reason"),
            "raw_max_u_gap": diag.get("raw_max_u_gap"),
            "uncovered_region_count": diag.get("uncovered_region_count"),
            "min_samples_per_region": diag.get("min_samples_per_region"),
            "sampling_coverage_ok": diag.get("sampling_coverage_ok"),
            "warnings": list(diag.get("warnings") or []),
        }


def sample_pack_is_empty(pack) -> bool:
    """``True`` when the pack constrains nothing (no sample, no face, no mask)."""
    if pack is None:
        return True
    if int(pack.basis.shape[1]) == 0 or int(pack.sample_A.shape[2]) == 0:
        return True
    return not bool(pack.sample_mask.any() and pack.face_mask.any())


def empty_sample_pack(num_controls: int, num_samples: int = 0,
                      dtype=torch.float32, device=None) -> BSplineSampleConstraintPack:
    """A pack that constrains nothing (used when no geometry could be built)."""
    B, M, C = 1, int(num_samples), int(num_controls)
    return BSplineSampleConstraintPack(
        basis=torch.zeros(B, M, C, dtype=dtype, device=device),
        sample_s=torch.zeros(B, M, dtype=dtype, device=device),
        sample_u=torch.zeros(B, M, dtype=dtype, device=device),
        sample_A=torch.zeros(B, M, 0, 2, dtype=dtype, device=device),
        sample_b=torch.zeros(B, M, 0, dtype=dtype, device=device),
        face_mask=torch.zeros(B, M, 0, dtype=torch.bool, device=device),
        sample_mask=torch.zeros(B, M, dtype=torch.bool, device=device),
        region_id=torch.full((B, M), -1, dtype=torch.long, device=device),
        sample_is_box=torch.zeros(B, M, dtype=torch.bool, device=device),
        diagnostics={"warnings": ["empty pack"]})


# ---------------------------------------------------------------------------
# 4. per-sample construction
# ---------------------------------------------------------------------------


def _sample_pack_one(codec: BSplineCodec,
                     cell_A, cell_b, cell_face_mask, cell_anchors,
                     reference_controls, num_samples: int, lookup_points: int,
                     cell_is_box=None, margin: float = 0.0,
                     include_box: bool = True) -> Optional[Dict]:
    """One sample's arc-length sampling + unique region assignment."""
    anchors = _as_numpy(cell_anchors).reshape(-1)
    R = int(anchors.shape[0])
    A_reg = np.asarray(cell_A, dtype=np.float64).reshape(R, -1, 2)
    b_reg = np.asarray(cell_b, dtype=np.float64).reshape(R, -1)
    mask_reg = np.asarray(cell_face_mask, dtype=bool).reshape(b_reg.shape)
    box_reg = (np.zeros(R, dtype=bool) if cell_is_box is None
               else np.asarray(cell_is_box, dtype=bool).reshape(R))
    if not include_box:
        mask_reg = mask_reg & ~box_reg[:, None]
    if R == 0 or not mask_reg.any():
        return None

    # ---- region ordering: bridges included, ascending anchor progress ------
    tau, order = responsibility_intervals(anchors)
    A_reg, b_reg = A_reg[order], b_reg[order]
    mask_reg, box_reg = mask_reg[order], box_reg[order]

    arc = uniform_arc_length_params(codec, reference_controls,
                                    num_samples=num_samples,
                                    lookup_points=lookup_points)
    s_l = arc["s"]
    region = assign_sample_regions(tau, s_l, R)

    sample_A = A_reg[region]                              # [M,F,2]
    sample_b = b_reg[region] - float(margin)              # [M,F]
    face = mask_reg[region]                               # [M,F]
    is_box = box_reg[region]                              # [M]
    has_face = face.any(axis=-1)
    sample_mask = has_face & np.isfinite(sample_b).any(axis=-1)

    # ---- coverage statistics (per REGION, not per sample) -----------------
    counts = np.bincount(region[sample_mask], minlength=R).astype(np.int64)
    covered_regions = np.flatnonzero(mask_reg.any(axis=1))
    counts_all = counts[covered_regions] if covered_regions.size else counts
    uncovered = int((counts_all == 0).sum())
    min_per = int(counts_all.min()) if counts_all.size else 0
    warnings: List[str] = []
    if uncovered > 0:
        warnings.append("%d of %d constrained regions receive NO sample point"
                        % (uncovered, int(covered_regions.size)))
    if arc["fallback"]:
        if arc["fallback_reason"] == "skewed_arc_length":
            warnings.append(
                "reference arc length is concentrated in u (max u gap %.3f): "
                "uniform-u fallback instead of equal-arc-length samples"
                % arc["raw_max_u_gap"])
        else:
            warnings.append("degenerate reference curve (length %.3g): uniform-u "
                            "fallback" % arc["length"])

    return {
        "basis": arc["basis"], "u": arc["u"], "s": s_l,
        "A": sample_A, "b": sample_b, "face_mask": face,
        "sample_mask": sample_mask, "region_id": region,
        "sample_is_box": is_box,
        "arc_fallback": bool(arc["fallback"]),
        "arc_fallback_reason": arc["fallback_reason"],
        "raw_max_u_gap": float(arc["raw_max_u_gap"]),
        "curve_length": float(arc["length"]),
        "region_sample_counts": counts.tolist(),
        "uncovered_region_count": uncovered,
        "min_samples_per_region": min_per,
        "num_regions": int(R),
        "warnings": warnings,
    }


def _assemble_sample_pack(per_sample: List[Optional[Dict]], num_controls: int,
                          device=None, dtype=torch.float32
                          ) -> Optional[BSplineSampleConstraintPack]:
    """Pad per-sample dicts (``None`` = fully masked row) into one pack."""
    if not per_sample or all(p is None for p in per_sample):
        return None
    B = len(per_sample)
    C = int(num_controls)
    M = max(p["basis"].shape[0] for p in per_sample if p is not None)
    F = max(p["A"].shape[1] for p in per_sample if p is not None)
    M = max(M, 1)
    F = max(F, 1)

    basis = torch.zeros(B, M, C, dtype=dtype, device=device)
    sample_s = torch.zeros(B, M, dtype=dtype, device=device)
    sample_u = torch.zeros(B, M, dtype=dtype, device=device)
    sample_A = torch.zeros(B, M, F, 2, dtype=dtype, device=device)
    sample_b = torch.zeros(B, M, F, dtype=dtype, device=device)
    face_mask = torch.zeros(B, M, F, dtype=torch.bool, device=device)
    sample_mask = torch.zeros(B, M, dtype=torch.bool, device=device)
    region_id = torch.full((B, M), -1, dtype=torch.long, device=device)
    sample_is_box = torch.zeros(B, M, dtype=torch.bool, device=device)

    diag = {"arc_length_fallback": [], "arc_fallback_reason": [],
            "raw_max_u_gap": [], "uncovered_region_count": [],
            "min_samples_per_region": [], "region_sample_counts": [],
            "num_regions": [], "curve_length": [], "warnings": []}
    for i, pack in enumerate(per_sample):
        if pack is None:
            diag["arc_length_fallback"].append(None)
            diag["arc_fallback_reason"].append(None)
            diag["raw_max_u_gap"].append(None)
            diag["uncovered_region_count"].append(None)
            diag["min_samples_per_region"].append(None)
            diag["region_sample_counts"].append([])
            diag["num_regions"].append(0)
            diag["curve_length"].append(0.0)
            continue
        m, f = pack["basis"].shape[0], pack["A"].shape[1]
        basis[i, :m] = torch.as_tensor(pack["basis"], dtype=dtype, device=device)
        sample_s[i, :m] = torch.as_tensor(pack["s"], dtype=dtype, device=device)
        sample_u[i, :m] = torch.as_tensor(pack["u"], dtype=dtype, device=device)
        sample_A[i, :m, :f] = torch.as_tensor(pack["A"], dtype=dtype, device=device)
        sample_b[i, :m, :f] = torch.as_tensor(pack["b"], dtype=dtype, device=device)
        face_mask[i, :m, :f] = torch.as_tensor(pack["face_mask"], device=device)
        sample_mask[i, :m] = torch.as_tensor(pack["sample_mask"], device=device)
        region_id[i, :m] = torch.as_tensor(pack["region_id"], device=device).long()
        sample_is_box[i, :m] = torch.as_tensor(pack["sample_is_box"], device=device)
        diag["arc_length_fallback"].append(pack["arc_fallback"])
        diag["arc_fallback_reason"].append(pack["arc_fallback_reason"])
        diag["raw_max_u_gap"].append(pack["raw_max_u_gap"])
        diag["uncovered_region_count"].append(pack["uncovered_region_count"])
        diag["min_samples_per_region"].append(pack["min_samples_per_region"])
        diag["region_sample_counts"].append(pack["region_sample_counts"])
        diag["num_regions"].append(pack["num_regions"])
        diag["curve_length"].append(pack["curve_length"])
        diag["warnings"].extend("[sample %d] %s" % (i, w) for w in pack["warnings"])

    pack = BSplineSampleConstraintPack(
        basis=basis, sample_s=sample_s, sample_u=sample_u, sample_A=sample_A,
        sample_b=sample_b, face_mask=face_mask, sample_mask=sample_mask,
        region_id=region_id, sample_is_box=sample_is_box, diagnostics=diag)
    diag["num_constraint_samples"] = int(pack.sample_mask.sum())
    diag["num_constraint_rows"] = pack.num_constraint_rows
    diag["num_faces_max"] = F
    diag["sampling_coverage_ok"] = bool(diag["num_constraint_rows"] > 0
                                        and not any(v for v in
                                                    diag["uncovered_region_count"]
                                                    if v))
    return pack


# ---------------------------------------------------------------------------
# 5. builders
# ---------------------------------------------------------------------------


def _reference_rows(reference_controls, index: int) -> np.ndarray:
    """``[C,2]`` reference polygon of sample ``index`` (batch or single)."""
    q = _as_numpy(reference_controls)
    if q.ndim == 2:
        return q
    if index >= q.shape[0]:
        raise ValueError("reference_controls has %d rows, needed row %d"
                         % (q.shape[0], index))
    return q[index]


def build_sampled_constraint_pack(
    codec: BSplineCodec,
    corridors: Sequence[Optional[object]],
    reference_controls,
    constraint_points: int = DEFAULT_CONSTRAINT_POINTS,
    arc_length_lookup_points: int = DEFAULT_LOOKUP_POINTS,
    margin: float = 0.0,
    device=None,
    dtype=torch.float32,
) -> Optional[BSplineSampleConstraintPack]:
    """``trajsafe`` path: one pack from a batch of frozen ``SafetyCorridor``s.

    ``corridors[b]`` may be ``None`` / invalid (failed search): that row becomes
    fully masked, so the guidance and the QP are no-ops for it instead of
    raising.  ``reference_controls`` is the clean B-spline polygon the geometry
    was generated from (``[B,C,2]`` or a shared ``[C,2]``).
    """
    per_sample: List[Optional[Dict]] = []
    for b, corridor in enumerate(corridors):
        if corridor is None or not getattr(corridor, "valid", False) \
                or getattr(corridor, "num_cells", 0) < 1:
            per_sample.append(None)
            continue
        cells = list(corridor.cells)
        A, bb, mask, is_box = halfspace_tables_from_cells(cells)
        anchors = np.asarray([float(c.anchor_s) for c in cells], dtype=np.float64)
        per_sample.append(_sample_pack_one(
            codec, A, bb, mask, anchors, _reference_rows(reference_controls, b),
            constraint_points, arc_length_lookup_points, cell_is_box=is_box,
            margin=margin))
    return _assemble_sample_pack(per_sample, codec.num_controls, device, dtype)


def build_sampled_constraint_pack_from_tables(
    codec: BSplineCodec,
    tables: Sequence[Optional[Dict]],
    reference_controls,
    constraint_points: int = DEFAULT_CONSTRAINT_POINTS,
    arc_length_lookup_points: int = DEFAULT_LOOKUP_POINTS,
    margin: float = 0.0,
    include_box: bool = True,
    device=None,
    dtype=torch.float32,
) -> Optional[BSplineSampleConstraintPack]:
    """``legacy_local`` path: one pack from per-sample region tables.

    Each table is the dict :func:`gp_diffuser.safety.local_regions.build_region_table`
    returns (``cell_A [R,F,2]``, ``cell_b [R,F]``, ``cell_mask [R]``,
    ``cell_is_box [R]``, ``anchors [R]``).  ``tables`` has one entry per batch
    row and may contain ``None``.  ``include_box=False`` drops the map-box cells,
    which is what the LOCAL violation statistics use.
    """
    per_sample: List[Optional[Dict]] = []
    for b, table in enumerate(tables):
        if table is None:
            per_sample.append(None)
            continue
        cell_mask = np.asarray(table["cell_mask"], dtype=bool).reshape(-1)
        if not cell_mask.any():
            per_sample.append(None)
            continue
        A = np.asarray(table["cell_A"], dtype=np.float64)
        bb = np.asarray(table["cell_b"], dtype=np.float64)
        face = cell_mask[:, None] & np.isfinite(bb) & (np.abs(A).sum(axis=-1) > 0.0)
        if not face.any():
            per_sample.append(None)
            continue
        anchors = np.asarray(table["anchors"], dtype=np.float64).reshape(-1)
        is_box = (np.asarray(table["cell_is_box"], dtype=bool).reshape(-1)
                  if "cell_is_box" in table else np.zeros(len(anchors), dtype=bool))
        per_sample.append(_sample_pack_one(
            codec, A, bb, face, anchors, _reference_rows(reference_controls, b),
            constraint_points, arc_length_lookup_points, cell_is_box=is_box,
            margin=margin, include_box=include_box))
    return _assemble_sample_pack(per_sample, codec.num_controls, device, dtype)


# ---------------------------------------------------------------------------
# 6. inspection helpers (visualisation / diagnostics)
# ---------------------------------------------------------------------------


def sampled_points(pack: BSplineSampleConstraintPack, controls,
                   sample: int = 0) -> np.ndarray:
    """``P_l = N_l Q`` -> ``[M,2]`` numpy for one sample of the batch."""
    q = _as_numpy(controls)
    q = q.reshape(-1, 2) if q.ndim == 2 else q[0]
    basis = _as_numpy(pack.basis[sample])
    return basis @ q


def region_polygons(pack: BSplineSampleConstraintPack, sample: int = 0
                    ) -> List[Tuple[int, np.ndarray, np.ndarray]]:
    """Unique ``(region_id, A [f,2], b [f])`` of the regions actually assigned.

    Only the regions that own at least one sample point are returned, plus the
    number of samples each one owns (useful for the coverage figure).
    """
    if pack is None:
        return []
    ids = _as_numpy(pack.region_id[sample], np.int64)
    smask = _as_numpy(pack.sample_mask[sample], bool)
    A = _as_numpy(pack.sample_A[sample])
    b = _as_numpy(pack.sample_b[sample])
    fmask = _as_numpy(pack.face_mask[sample], bool)
    out: List[Tuple[int, np.ndarray, np.ndarray]] = []
    seen = set()
    for l in range(len(ids)):
        if not smask[l]:
            continue
        rid = int(ids[l])
        if rid in seen:
            continue
        seen.add(rid)
        if not fmask[l].any():
            continue
        out.append((rid, A[l][fmask[l]], b[l][fmask[l]]))
    return out
