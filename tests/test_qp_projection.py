"""Terminal convex QP projection tests (design document section 9.3)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from gp_diffuser.diffusion.clean_guidance import constraint_loss_and_grad
from gp_diffuser.safety.local_regions import box_cell_faces, pack_from_tables
from gp_diffuser.safety.pack_utils import empty_pack, merge_packs
from gp_diffuser.safety.qp_projection import (corridor_constraint_rows,
                                              project_controls_to_corridor_qp)

SOLVER = {"eps_abs": 1e-7, "eps_rel": 1e-7, "max_iter": 20000, "polish": True,
          "feasibility_tol_scene": 1e-4, "scene_to_meter": 32.0}


def _codec(num_controls: int):
    from gp_diffuser.trajectory.bspline import BSplineCodec
    return BSplineCodec(degree=3, num_controls=num_controls, curve_points=128)


def _box_table(num_cells: int = 32):
    A, b = box_cell_faces(0.0)
    return {"cell_A": np.repeat(A[None], num_cells, axis=0),
            "cell_b": np.repeat(b[None], num_cells, axis=0),
            "cell_mask": np.ones(num_cells, dtype=bool),
            "cell_is_box": np.ones(num_cells, dtype=bool),
            "anchors": np.linspace(0.0, 1.0, num_cells)}


def _box_pack(num_controls: int):
    codec = _codec(num_controls)
    return codec, pack_from_tables(codec, [_box_table(32)])


def _outside_controls(num_controls: int) -> torch.Tensor:
    q = torch.zeros(1, num_controls, 2)
    q[0, num_controls // 4: num_controls // 2] = torch.tensor([1.35, -1.25])
    return q


# --------------------------------------------------------------------------- #
# 9.3.1 / 9.3.2 feasible box corridor + exact endpoints
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("num_controls", [32, 64])
def test_qp_projects_into_the_box_and_keeps_endpoints(num_controls):
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

    # the same violation measure the guidance uses agrees
    _, _, stats = constraint_loss_and_grad(projected, pack)
    assert float(stats["max_violation"][0]) <= 1e-4
    assert projected.shape == q.shape, "the QP variable is the global control polygon"


# --------------------------------------------------------------------------- #
# 9.3.3 heading tangent constraints
# --------------------------------------------------------------------------- #
def test_qp_respects_a_valid_heading():
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
@pytest.mark.parametrize("kind", ["none", "empty", "all_masked"])
def test_empty_pack_is_never_certified(kind):
    codec = _codec(32)
    q = _outside_controls(32)
    start_goal = torch.tensor([[[-0.4, -0.4], [0.4, 0.4]]])
    if kind == "none":
        pack = None
    elif kind == "empty":
        pack = empty_pack(32)
    else:
        pack = merge_packs([None], 32)
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
    codec, pack = _box_pack(32)
    q = _outside_controls(32)
    # start == goal == outside the box: the endpoint equality and the corridor
    # cannot hold at the same time
    start_goal = torch.tensor([[[1.5, 1.5], [1.5, 1.5]]])
    projected, info = project_controls_to_corridor_qp(q, pack, start_goal,
                                                     solver_cfg=SOLVER)
    assert not bool(info["certified"][0]), info["status"]
    assert info["status"][0] not in ("solved",), info["status"]
    assert torch.equal(projected, q), "an infeasible sample must not be rewritten"


# --------------------------------------------------------------------------- #
# constraint assembly
# --------------------------------------------------------------------------- #
def test_constraint_rows_match_the_pack_inequalities():
    codec, pack = _box_pack(32)
    G, h = corridor_constraint_rows(pack, 0)
    valid_faces = pack.face_mask[0][pack.piece_mask[0]].sum().item()
    assert G is not None and len(h) == 4 * valid_faces
    q = _outside_controls(32).numpy()[0]
    g = G @ q.reshape(-1) - h
    from gp_diffuser.diffusion.clean_guidance import constraint_values
    _, raw, valid = constraint_values(torch.as_tensor(q)[None],
                                     pack.to(dtype=torch.float32))
    assert np.allclose(np.sort(g), np.sort(raw[valid].numpy().astype(np.float64)),
                       atol=1e-5)


def test_padded_faces_are_not_emitted():
    codec = _codec(32)
    table = _box_table(32)
    # make ONE face of every cell a padding face (b = +inf, A = 0)
    table["cell_b"] = np.concatenate(
        [table["cell_b"][:, :3], np.full((32, 1), np.inf)], axis=1)
    table["cell_A"] = np.concatenate(
        [table["cell_A"][:, :3], np.zeros((32, 1, 2))], axis=1)
    pack = pack_from_tables(codec, [table])
    G, h = corridor_constraint_rows(pack, 0)
    assert G is not None
    assert len(h) == 4 * 3 * pack.num_pieces[0].item()
