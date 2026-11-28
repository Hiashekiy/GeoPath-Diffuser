"""Uniform-arc-length B-spline sample pack tests (spec sections 3 / 9).

Covers: the frozen arc-length parameters, the ONE-region-per-sample assignment
(bridge cells included), batch independence, face/padding masks and the coverage
statistics.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from gp_diffuser.safety.bspline_sample_constraints import (
    assign_sample_regions, build_sampled_constraint_pack,
    build_sampled_constraint_pack_from_tables, empty_sample_pack,
    needs_uniform_fallback, region_polygons, sample_pack_is_empty,
    uniform_arc_length_params)
from gp_diffuser.safety.local_regions import box_cell_faces


def _codec(num_controls: int = 32):
    from gp_diffuser.trajectory.bspline import BSplineCodec
    return BSplineCodec(degree=3, num_controls=num_controls, curve_points=128)


def _straight(num_controls: int = 32, hump: float = 0.0) -> torch.Tensor:
    q = torch.zeros(1, num_controls, 2)
    q[0, :, 0] = torch.linspace(-0.8, 0.8, num_controls)
    if hump:
        q[0, num_controls // 3: num_controls // 2, 1] = hump
    return q


def _table(num_cells: int, anchors=None, is_box: bool = False,
           faces=None) -> dict:
    """Uniform box cells; ``faces`` optionally replaces A/b for every cell."""
    A, b = box_cell_faces(0.0) if faces is None else faces
    n = int(num_cells)
    return {"cell_A": np.repeat(np.asarray(A, dtype=np.float64)[None], n, axis=0),
            "cell_b": np.repeat(np.asarray(b, dtype=np.float64)[None], n, axis=0),
            "cell_mask": np.ones(n, dtype=bool),
            "cell_is_box": np.full(n, bool(is_box)),
            "anchors": (np.linspace(0.0, 1.0, n) if anchors is None
                        else np.asarray(anchors, dtype=np.float64))}


# --------------------------------------------------------------------------- #
# region assignment
# --------------------------------------------------------------------------- #
def test_anchor_midpoints_define_the_boundaries():
    tau = np.array([0.0, 0.125, 0.375, 0.625, 0.875, 1.0])   # 5 regions
    s = np.array([0.0, 0.124, 0.125, 0.2, 0.375, 0.5, 0.625, 0.9, 1.0])
    rid = assign_sample_regions(tau, s, 5)
    assert rid.tolist() == [0, 0, 1, 1, 2, 2, 3, 4, 4], rid.tolist()


def test_every_sample_gets_exactly_one_region():
    codec = _codec()
    ref = _straight()
    table = _table(11)
    pack = build_sampled_constraint_pack_from_tables(
        codec, [table], ref, constraint_points=97, arc_length_lookup_points=256)
    rid = pack.region_id[0].numpy()
    assert rid.shape == (97,)
    assert rid.min() >= 0
    assert rid.max() <= 10
    assert pack.sample_mask[0].all()


def test_boundary_points_go_right_and_ends_to_first_last():
    tau = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    rid = assign_sample_regions(tau, np.array([0.0, 0.25, 0.5, 0.75, 1.0]), 4)
    assert rid.tolist() == [0, 1, 2, 3, 3]


def test_bridge_cells_participate_in_the_order():
    """Unsorted anchors (a bridge inserted in the middle) must be re-ordered."""
    codec = _codec()
    ref = _straight()
    anchors = np.array([0.0, 0.2, 0.5, 0.35, 1.0])          # 0.35 = bridge
    table = _table(5, anchors=anchors)
    pack = build_sampled_constraint_pack_from_tables(
        codec, [table], ref, constraint_points=41, arc_length_lookup_points=128)
    counts = pack.diagnostics["region_sample_counts"][0]
    assert len(counts) == 5
    assert sum(counts) == 41
    # the region sorted third (anchor 0.35) owns the samples near s = 0.35
    rid = pack.region_id[0].numpy()
    s = pack.sample_s[0].numpy()
    near = np.abs(s - 0.35) < 0.05
    assert near.any() and (np.unique(rid[near]) == 2).all()


def test_batch_rows_are_independent():
    codec = _codec()
    ref = torch.cat([_straight(32), _straight(32, hump=0.6)], dim=0)
    tables = [_table(8), _table(12)]
    pack = build_sampled_constraint_pack_from_tables(
        codec, tables, ref, constraint_points=40, arc_length_lookup_points=256)
    assert pack.basis.shape[0] == 2
    assert not torch.allclose(pack.sample_u[0], pack.sample_u[1]), \
        "a different reference polygon must give a different u_l"
    assert pack.region_id[0].max() <= 7 and pack.region_id[1].max() <= 11
    assert pack.sample_A.shape[2] == max(pack.face_mask[0].sum(-1).max().item(),
                                         pack.face_mask[1].sum(-1).max().item())


# --------------------------------------------------------------------------- #
# arc length parameters
# --------------------------------------------------------------------------- #
def test_arc_length_parameters_are_monotone_and_span_the_curve():
    codec = _codec()
    for hump in (0.0, 0.3, -0.5):
        arc = uniform_arc_length_params(codec, _straight(32, hump),
                                        num_samples=64, lookup_points=512)
        u = arc["u"]
        assert np.all(np.diff(u) > 0.0)
        assert u[0] == 0.0 and u[-1] == 1.0
        assert not arc["fallback"]


def test_samples_are_equidistant_in_arc_length():
    codec = _codec()
    ref = _straight(32, hump=0.5)
    arc = uniform_arc_length_params(codec, ref, num_samples=128,
                                    lookup_points=2048)
    points = arc["basis"] @ ref[0].numpy()
    segments = np.linalg.norm(np.diff(points, axis=0), axis=-1)
    assert segments.max() / max(segments.min(), 1e-12) < 1.15, \
        "uniform arc-length samples must be ~equidistant along the curve"


def test_degenerate_curve_falls_back_to_uniform_u():
    codec = _codec()
    ref = torch.zeros(1, 32, 2)
    arc = uniform_arc_length_params(codec, ref, num_samples=16, lookup_points=64)
    assert arc["fallback"]
    assert np.allclose(arc["u"], np.linspace(0.0, 1.0, 16))

    table = _table(4)
    pack = build_sampled_constraint_pack_from_tables(
        codec, [table], ref, constraint_points=16, arc_length_lookup_points=64)
    assert pack.diagnostics["arc_length_fallback"][0] is True
    assert pack.diagnostics["warnings"], "the fallback must be recorded"


def test_skewed_arc_length_falls_back_to_uniform_u():
    """A reference whose length is concentrated in a corner of u must not be used.

    Real case that motivated the guard: the clean prediction of the FIRST followed
    step is still a wild spike, so equal-arc-length sampling put 109 samples into
    the first u-decile, 147 into the last one and NONE in between - i.e. 80 % of
    the curve carried no constraint at all.
    """
    codec = _codec()
    ref = torch.zeros(1, 32, 2)
    ref[0, 0] = torch.tensor([-0.9, 0.0])
    ref[0, -1] = torch.tensor([0.9, 0.0])          # far ends, collapsed middle
    arc = uniform_arc_length_params(codec, ref, num_samples=256,
                                    lookup_points=1024)
    assert arc["fallback"] and arc["fallback_reason"] == "skewed_arc_length"
    assert arc["raw_max_u_gap"] > 0.5, arc["raw_max_u_gap"]
    assert np.allclose(arc["u"], np.linspace(0.0, 1.0, 256))

    pack = build_sampled_constraint_pack_from_tables(
        codec, [_table(8)], ref, constraint_points=256,
        arc_length_lookup_points=1024)
    hist = np.histogram(pack.sample_u[0].numpy(), bins=10, range=(0.0, 1.0))[0]
    assert (hist > 0).all(), \
        "every u-decile must own at least one sample: %s" % (hist.tolist(),)
    assert pack.diagnostics["arc_fallback_reason"][0] == "skewed_arc_length"
    assert pack.diagnostics["warnings"], "the fallback must be recorded"


def test_uniform_fallback_helper_threshold():
    M = 256
    assert needs_uniform_fallback(np.linspace(0.0, 1.0, M)) is False
    skewed = np.concatenate([np.linspace(0.0, 0.06, 200),
                             np.linspace(0.94, 1.0, 56)])
    assert needs_uniform_fallback(skewed) is True
    assert needs_uniform_fallback(np.array([0.5])) is True, "M < 2 cannot be sampled"
    # two samples are always within the threshold: nothing to judge there
    assert needs_uniform_fallback(np.array([0.0, 1.0])) is False


def test_pack_parameters_are_frozen_and_reused():
    codec = _codec()
    ref = _straight(32, hump=0.4)
    pack = build_sampled_constraint_pack_from_tables(
        codec, [_table(9)], ref, constraint_points=64, arc_length_lookup_points=256)
    u0 = pack.sample_u.clone()
    rid0 = pack.region_id.clone()
    # evaluating a MOVED control polygon must not change the sampling or the
    # region assignment (they belong to the reference curve only)
    from gp_diffuser.diffusion.clean_guidance import sample_constraint_values
    moved = ref.clone()
    moved[0, 5:15, 1] += 0.2
    sample_constraint_values(moved, pack)
    assert torch.equal(pack.sample_u, u0)
    assert torch.equal(pack.region_id, rid0)


# --------------------------------------------------------------------------- #
# masks and coverage
# --------------------------------------------------------------------------- #
def test_map_box_and_padded_faces_are_masked():
    codec = _codec()
    table = _table(6)
    # one padded face per cell: b = +inf, A = 0
    table["cell_b"] = np.concatenate(
        [table["cell_b"], np.full((6, 1), np.inf)], axis=1)
    table["cell_A"] = np.concatenate(
        [table["cell_A"], np.zeros((6, 1, 2))], axis=1)
    pack = build_sampled_constraint_pack_from_tables(
        codec, [table], _straight(), constraint_points=32,
        arc_length_lookup_points=128)
    assert pack.face_mask[0].sum(dim=-1).max().item() == 4, \
        "the padded face must never be a valid constraint"
    assert pack.num_constraint_rows == 4 * 32

    boxed = build_sampled_constraint_pack_from_tables(
        codec, [_table(6, is_box=True)], _straight(), constraint_points=32,
        arc_length_lookup_points=128)
    assert boxed.sample_is_box[0].all()
    assert int(boxed.diagnostics["num_constraint_rows"]) == 4 * 32
    no_box = build_sampled_constraint_pack_from_tables(
        codec, [_table(6, is_box=True)], _straight(), constraint_points=32,
        arc_length_lookup_points=128, include_box=False)
    assert no_box is None, "box cells must not constrain the local pack"


def test_uncovered_regions_are_counted():
    codec = _codec()
    # 64 regions but only 8 sample points -> many regions own nothing
    pack = build_sampled_constraint_pack_from_tables(
        codec, [_table(64)], _straight(), constraint_points=8,
        arc_length_lookup_points=128)
    counts = pack.diagnostics["region_sample_counts"][0]
    expected = int(np.sum(np.asarray(counts) == 0))
    assert pack.diagnostics["uncovered_region_count"][0] == expected > 0
    assert pack.diagnostics["sampling_coverage_ok"] is False
    assert pack.diagnostics["min_samples_per_region"][0] == 0


def test_full_coverage_is_reported_as_ok():
    codec = _codec()
    pack = build_sampled_constraint_pack_from_tables(
        codec, [_table(4)], _straight(), constraint_points=64,
        arc_length_lookup_points=256)
    assert pack.diagnostics["uncovered_region_count"][0] == 0
    assert pack.diagnostics["sampling_coverage_ok"] is True
    assert pack.diagnostics["min_samples_per_region"][0] > 0


def test_missing_rows_and_empty_packs():
    codec = _codec()
    ref = _straight().repeat(3, 1, 1)
    pack = build_sampled_constraint_pack_from_tables(
        codec, [None, _table(4), None], ref, constraint_points=16,
        arc_length_lookup_points=64)
    assert pack.basis.shape[0] == 3
    assert pack.sample_mask[0].sum() == 0 and pack.sample_mask[2].sum() == 0
    assert pack.sample_mask[1].all()

    assert build_sampled_constraint_pack_from_tables(
        codec, [None], _straight(), constraint_points=8) is None
    empty = empty_sample_pack(32, num_samples=4)
    assert sample_pack_is_empty(empty)
    assert not sample_pack_is_empty(pack)


# --------------------------------------------------------------------------- #
# corridor path
# --------------------------------------------------------------------------- #
class _Cell:
    def __init__(self, anchor_s, A, b):
        self.anchor_s = float(anchor_s)
        self.A = np.asarray(A, dtype=np.float64)
        self.b = np.asarray(b, dtype=np.float64)


class _Corridor:
    def __init__(self, cells, valid=True):
        self.cells = cells
        self.valid = bool(valid)

    @property
    def num_cells(self):
        return len(self.cells)


def test_corridor_path_and_table_path_agree():
    codec = _codec()
    A, b = box_cell_faces(0.0)
    anchors = np.linspace(0.0, 1.0, 8)
    cells = [_Cell(s, A, b) for s in anchors]
    from_corridor = build_sampled_constraint_pack(
        codec, [_Corridor(cells)], _straight(), constraint_points=32,
        arc_length_lookup_points=128)
    from_table = build_sampled_constraint_pack_from_tables(
        codec, [_table(8)], _straight(), constraint_points=32,
        arc_length_lookup_points=128)
    assert torch.allclose(from_corridor.basis, from_table.basis)
    assert torch.equal(from_corridor.region_id, from_table.region_id)
    assert torch.allclose(from_corridor.sample_b, from_table.sample_b)


def test_invalid_corridor_becomes_a_masked_row():
    codec = _codec()
    A, b = box_cell_faces(0.0)
    cells = [_Cell(s, A, b) for s in np.linspace(0.0, 1.0, 4)]
    pack = build_sampled_constraint_pack(
        codec, [None, _Corridor(cells, valid=False), _Corridor(cells)],
        _straight().repeat(3, 1, 1), constraint_points=16,
        arc_length_lookup_points=64)
    assert pack.sample_mask[0].sum() == 0
    assert pack.sample_mask[1].sum() == 0
    assert pack.sample_mask[2].all()


def test_region_polygons_lists_one_cell_per_region():
    codec = _codec()
    pack = build_sampled_constraint_pack_from_tables(
        codec, [_table(6)], _straight(), constraint_points=48,
        arc_length_lookup_points=192)
    polys = region_polygons(pack, 0)
    ids = [p[0] for p in polys]
    assert len(ids) == len(set(ids)) == 6
    assert all(p[1].shape[0] == 4 for p in polys)
