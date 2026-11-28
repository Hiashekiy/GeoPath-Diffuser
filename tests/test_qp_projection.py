"""Terminal convex QP on the sampled B-spline constraints (spec sections 6 / 9).

The QP variable is ALWAYS the global control polygon (``2 * num_controls``
variables); the rows are ``N_l[k] * a_{j(l),f,d}`` so ``G @ vec(Q) <= h`` is
exactly ``a_{j(l),f} . (N_l Q) <= b_{j(l),f}`` for the sampled curve points.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from gp_diffuser.diffusion.clean_guidance import (constraint_loss_and_grad,
                                                sample_constraint_values)
from gp_diffuser.safety.bspline_sample_constraints import (
    build_sampled_constraint_pack_from_tables, empty_sample_pack)
from gp_diffuser.safety.local_regions import box_cell_faces
from gp_diffuser.safety.qp_projection import (endpoint_violation,
                                              project_controls_to_corridor_qp,
                                              sampled_constraint_rows)

SOLVER = {"eps_abs": 1e-7, "eps_rel": 1e-7, "max_iter": 20000, "polish": True,
          "feasibility_tol_scene": 1e-4, "scene_to_meter": 32.0}


def _require_osqp():
    pytest.importorskip("osqp")


def _codec(num_controls: int = 32):
    from gp_diffuser.trajectory.bspline import BSplineCodec
    return BSplineCodec(degree=3, num_controls=num_controls, curve_points=128)


def _straight(num_controls: int = 32, hump: float = 0.0) -> torch.Tensor:
    q = torch.zeros(1, num_controls, 2)
    q[0, :, 0] = torch.linspace(-0.8, 0.8, num_controls)
    if hump:
        q[0, num_controls // 4: num_controls // 2, 1] = hump
    return q


def _box_table(num_cells: int = 32, b=None):
    A, bb = box_cell_faces(0.0)
    bb = np.asarray(bb, dtype=np.float64) if b is None else np.asarray(b, float)
    return {"cell_A": np.repeat(A[None], num_cells, axis=0),
            "cell_b": np.repeat(bb[None], num_cells, axis=0),
            "cell_mask": np.ones(num_cells, dtype=bool),
            "cell_is_box": np.ones(num_cells, dtype=bool),
            "anchors": np.linspace(0.0, 1.0, num_cells)}


def _box_pack(num_controls: int = 32, reference=None, points: int = 64,
              table=None):
    codec = _codec(num_controls)
    ref = _straight(num_controls) if reference is None else reference
    pack = build_sampled_constraint_pack_from_tables(
        codec, [table if table is not None else _box_table(32)], ref,
        constraint_points=points, arc_length_lookup_points=256)
    return codec, pack


def _outside_controls(num_controls: int = 32) -> torch.Tensor:
    q = torch.zeros(1, num_controls, 2)
    q[0, num_controls // 4: num_controls // 2] = torch.tensor([1.35, -1.25])
    return q


# --------------------------------------------------------------------------- #
# 9.3.1 / 9.3.2 feasible box corridor + exact endpoints
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("num_controls", [32, 64])
def test_qp_projects_into_the_box_and_keeps_endpoints(num_controls):
    _require_osqp()
    codec, pack = _box_pack(num_controls)
    q = _outside_controls(num_controls)
    start_goal = torch.tensor([[[-0.4, -0.4], [0.4, 0.4]]])

    projected, info = project_controls_to_corridor_qp(q, pack, start_goal,
                                                     solver_cfg=SOLVER)
    assert info["status"][0] in ("solved", "solved inaccurate"), info["status"]
    assert bool(info["certified"][0])
    assert float(info["max_violation_scene"][0]) <= 1e-4
    assert torch.allclose(projected[:, 0], start_goal[:, 0], atol=1e-9)
    assert torch.allclose(projected[:, -1], start_goal[:, 1], atol=1e-9)
    assert projected.shape == q.shape, "the QP variable is the global control polygon"

    # the same violation measure the guidance uses agrees
    _, _, stats = constraint_loss_and_grad(projected, pack)
    assert float(stats["max_violation"][0]) <= 1e-4

    # all valid SAMPLE constraints hold, point by point
    _, raw, valid = sample_constraint_values(projected, pack)
    assert float((torch.relu(raw) * valid).max()) <= 1e-4


# --------------------------------------------------------------------------- #
# 9.3.3 heading tangent constraints
# --------------------------------------------------------------------------- #
def test_qp_respects_a_valid_heading():
    _require_osqp()
    codec, pack = _box_pack(32)
    q = _outside_controls(32)
    start_goal = torch.tensor([[[-0.5, -0.5], [0.5, 0.5]]])
    heading = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    valid = torch.ones(1, 2, dtype=torch.bool)

    projected, info = project_controls_to_corridor_qp(
        q, pack, start_goal, heading=heading, heading_valid=valid, solver_cfg=SOLVER)
    assert bool(info["certified"][0])

    start_dir = projected[0, 1] - projected[0, 0]
    goal_dir = projected[0, -1] - projected[0, -2]
    assert abs(float(start_dir[1])) < 1e-6, "start tangent must lose the vertical part"
    assert float(start_dir[0]) >= -1e-9
    assert abs(float(goal_dir[0])) < 1e-6
    assert float(goal_dir[1]) >= -1e-9


def test_invalid_heading_adds_no_tangent_constraint():
    _require_osqp()
    codec, pack = _box_pack(32)
    q = _outside_controls(32)
    start_goal = torch.tensor([[[-0.5, -0.5], [0.5, 0.5]]])
    heading = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    valid = torch.zeros(1, 2, dtype=torch.bool)
    with_invalid, info = project_controls_to_corridor_qp(
        q, pack, start_goal, heading=heading, heading_valid=valid, solver_cfg=SOLVER)
    without, info_none = project_controls_to_corridor_qp(
        q, pack, start_goal, heading=None, solver_cfg=SOLVER)
    assert bool(info["certified"][0]) and bool(info_none["certified"][0])
    assert torch.allclose(with_invalid, without, atol=1e-6), \
        "an invalid heading must behave exactly like no heading at all"


# --------------------------------------------------------------------------- #
# 9.3.5 empty / invalid pack is never certified
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["none", "empty"])
def test_empty_pack_is_never_certified(kind):
    _require_osqp()
    codec = _codec(32)
    q = _outside_controls(32)
    start_goal = torch.tensor([[[-0.4, -0.4], [0.4, 0.4]]])
    pack = None if kind == "none" else empty_sample_pack(32, num_samples=16)
    projected, info = project_controls_to_corridor_qp(q, pack, start_goal,
                                                     solver_cfg=SOLVER)
    assert not info["all_certified"]
    assert not bool(info["certified"].any())
    assert torch.equal(projected, q), "an uncertified sample keeps its controls"
    assert all(s == "no_constraints" for s in info["status"])


# --------------------------------------------------------------------------- #
# 9.3.6 infeasible constraints must be reported
# --------------------------------------------------------------------------- #
def test_infeasible_constraints_are_reported_not_hidden():
    _require_osqp()
    codec, pack = _box_pack(32)
    q = _outside_controls(32)
    # start == goal == outside the box: the endpoint equality and the sampled
    # constraints cannot hold at the same time
    start_goal = torch.tensor([[[1.5, 1.5], [1.5, 1.5]]])
    projected, info = project_controls_to_corridor_qp(q, pack, start_goal,
                                                     solver_cfg=SOLVER)
    assert not bool(info["certified"][0]), info["status"]
    assert torch.equal(projected, q), "an infeasible sample must not be rewritten"
    assert info["per_sample"][0].get("failure_reason") in (
        "endpoints_outside_their_region", "sampled_constraints_infeasible",
        "residual_violation_after_projection"), info["per_sample"][0]
    assert float(info["endpoint_violation_scene"][0]) > 1e-3, \
        "the endpoint diagnostic must flag an endpoint outside its region"


# --------------------------------------------------------------------------- #
# constraint assembly (no solver needed)
# --------------------------------------------------------------------------- #
def test_constraint_rows_match_the_pack_inequalities():
    codec, pack = _box_pack(32)
    G, h = sampled_constraint_rows(pack, 0)
    valid = int((pack.sample_mask[0][:, None] & pack.face_mask[0]).sum())
    assert G is not None and G.shape == (valid, 2 * codec.num_controls)
    assert len(h) == valid

    q = _outside_controls(32).numpy()[0]
    g = G @ q.reshape(-1) - h
    _, raw, mask = sample_constraint_values(torch.as_tensor(q)[None],
                                            pack.to(dtype=torch.float32))
    assert np.allclose(np.sort(g), np.sort(raw[mask].detach().numpy().astype(np.float64)),
                       atol=1e-5)


def test_rows_are_built_from_the_basis_times_the_region_normals():
    codec, pack = _box_pack(32, points=32)
    G, h = sampled_constraint_rows(pack, 0)
    N = pack.basis[0].numpy()
    A = pack.sample_A[0].numpy()
    b = pack.sample_b[0].numpy()
    fmask = pack.face_mask[0].numpy()
    samples, faces = np.nonzero(fmask)
    C = codec.num_controls
    # check three random rows element by element
    for r in (0, len(h) // 2, len(h) - 1):
        l, f = samples[r], faces[r]
        expected = np.zeros(2 * C)
        expected[0::2] = N[l] * A[l, f, 0]
        expected[1::2] = N[l] * A[l, f, 1]
        assert np.allclose(G[r], expected, atol=1e-7)
        assert h[r] == pytest.approx(b[l, f], abs=1e-9)


def test_padded_faces_are_not_emitted():
    codec = _codec(32)
    table = _box_table(32)
    table["cell_b"] = np.concatenate(
        [table["cell_b"][:, :3], np.full((32, 1), np.inf)], axis=1)
    table["cell_A"] = np.concatenate(
        [table["cell_A"][:, :3], np.zeros((32, 1, 2))], axis=1)
    _, pack = _box_pack(32, table=table)
    G, h = sampled_constraint_rows(pack, 0)
    assert G is not None
    assert len(h) == 3 * int(pack.sample_mask[0].sum())


def test_qp_variable_dimension_is_always_two_times_the_controls():
    for num_controls in (32, 64):
        codec, pack = _box_pack(num_controls)
        G, _ = sampled_constraint_rows(pack, 0)
        assert G.shape[1] == 2 * codec.num_controls == 2 * pack.num_controls


# --------------------------------------------------------------------------- #
# endpoint diagnostics: only the FIRST / LAST sample's own region
# --------------------------------------------------------------------------- #
def test_endpoint_diagnostics_use_only_the_first_and_last_region():
    codec = _codec(32)
    n = 4
    A, _ = box_cell_faces(0.0)
    # region 0 = a small box around the start, region 3 = one around the goal,
    # the middle regions are wide boxes
    left = np.array([-0.4, 0.6, 0.1, 0.1])          # -0.5 <= x <= -0.4, |y| <= 0.1
    right = np.array([0.6, -0.4, 0.1, 0.1])         #  0.4 <= x <=  0.5, |y| <= 0.1
    wide = np.array([2.0, 2.0, 2.0, 2.0])
    cell_b = np.stack([left, wide, wide, right])
    table = {"cell_A": np.repeat(A[None], n, axis=0), "cell_b": cell_b,
             "cell_mask": np.ones(n, dtype=bool),
             "cell_is_box": np.zeros(n, dtype=bool),
             "anchors": np.array([0.0, 0.4, 0.7, 1.0])}
    _, pack = _box_pack(32, table=table, points=32)
    start = np.array([-0.45, 0.0])
    goal = np.array([0.45, 0.0])
    info = endpoint_violation(pack, 0, start, goal)
    assert info["start"] <= 1e-9 and info["goal"] <= 1e-9
    # the SAME start point tested against the goal's region would be a violation:
    # proving the diagnostic does not use all regions
    A_last = pack.sample_A[0, -1].numpy()
    b_last = pack.sample_b[0, -1].numpy()
    fmask = pack.face_mask[0, -1].numpy()
    assert float(np.max(A_last[fmask] @ start - b_last[fmask])) > 0.5

    far = np.array([3.0, 0.0])
    assert endpoint_violation(pack, 0, far, goal)["start"] > 1.0
