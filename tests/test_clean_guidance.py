"""CPCG unit tests: analytic gradient, masking, trust region, DDIM consistency."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from gp_diffuser.diffusion.clean_guidance import (clean_prediction_guidance,
                                                constraint_loss_and_grad)
from gp_diffuser.diffusion.sampler_conditional_dit import ddim_step
from gp_diffuser.models.boundary import BoundaryDecoder
from gp_diffuser.safety.pack_utils import empty_pack, merge_packs
from gp_diffuser.safety.local_regions import box_cell_faces, pack_from_tables


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _box_table(num_cells: int = 32, margin: float = 0.0, valid: bool = True):
    A, b = box_cell_faces(margin)
    cell_A = np.zeros((num_cells, 4, 2))
    cell_b = np.full((num_cells, 4), np.inf)
    mask = np.zeros(num_cells, dtype=bool)
    if valid:
        cell_A[:] = A
        cell_b[:] = b
        mask[:] = True
    return {"cell_A": cell_A, "cell_b": cell_b, "cell_mask": mask,
            "cell_is_box": np.ones(num_cells, dtype=bool),
            "anchors": np.linspace(0.0, 1.0, num_cells)}


def _box_pack(codec, margin: float = 0.0, valid: bool = True):
    return pack_from_tables(codec, [_box_table(32, margin, valid)])


def _violating_controls(num_controls: int = 32) -> torch.Tensor:
    q = torch.zeros(1, num_controls, 2)
    q[0, num_controls // 4: num_controls // 2] = torch.tensor([1.45, 0.0])
    return q


def _codec(num_controls: int):
    from gp_diffuser.trajectory.bspline import BSplineCodec
    return BSplineCodec(degree=3, num_controls=num_controls, curve_points=128)


# --------------------------------------------------------------------------- #
# 9.1.1 finite differences
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k, d", [(10, 0), (12, 1), (20, 0)])
def test_analytic_gradient_matches_finite_difference(k, d):
    codec = _codec(32)
    pack = _box_pack(codec)
    q = _violating_controls(32)
    _, grad, _ = constraint_loss_and_grad(q, pack)

    eps = 1e-4
    plus = q.clone()
    plus[0, k, d] += eps
    minus = q.clone()
    minus[0, k, d] -= eps
    fd = ((constraint_loss_and_grad(plus, pack)[0]
           - constraint_loss_and_grad(minus, pack)[0]) / (2.0 * eps))
    assert torch.allclose(grad[0, k, d], fd[0], atol=1e-4, rtol=1e-3), \
        "analytic %r vs finite difference %r" % (float(grad[0, k, d]), float(fd[0]))


def test_gradient_step_reduces_the_safety_cost():
    codec = _codec(32)
    pack = _box_pack(codec)
    q = _violating_controls(32)
    before, grad, _ = constraint_loss_and_grad(q, pack)
    after, _, _ = constraint_loss_and_grad(q - 0.05 * grad, pack)
    assert float(after) < float(before)


# --------------------------------------------------------------------------- #
# 9.1.3 satisfied constraints -> exactly zero guidance
# --------------------------------------------------------------------------- #
def test_no_violation_means_zero_gradient_and_no_motion():
    codec = _codec(32)
    pack = _box_pack(codec)
    q = torch.zeros(1, 32, 2)
    q[0, :, 0] = torch.linspace(-0.5, 0.5, 32)
    loss, grad, stats = constraint_loss_and_grad(q, pack)
    assert float(loss) == 0.0
    assert torch.count_nonzero(grad) == 0
    assert not bool(stats["has_violation"].any())

    decoder = BoundaryDecoder(dtype=torch.float32)
    start_goal = torch.tensor([[[-0.5, 0.0], [0.5, 0.0]]])
    guided, gstats = clean_prediction_guidance(q, pack, decoder, start_goal,
                                               step_size=1.0, max_delta_scene=0.1)
    assert torch.equal(guided, q), "a satisfied pack must return q0 bit-for-bit"
    assert not bool(gstats["guidance_applied"].any())


# --------------------------------------------------------------------------- #
# 9.1.4 masked piece / face must not contribute
# --------------------------------------------------------------------------- #
def test_padding_pieces_and_faces_produce_no_gradient():
    codec = _codec(32)
    pack = _box_pack(codec)
    q = _violating_controls(32)

    def clone(p):
        return p.__class__(
            extraction=p.extraction.clone(), piece_A=p.piece_A.clone(),
            piece_b=p.piece_b.clone(), face_mask=p.face_mask.clone(),
            piece_mask=p.piece_mask.clone(),
            piece_region_id=p.piece_region_id.clone(),
            intervals=p.intervals.clone(), num_pieces=p.num_pieces.clone())

    # a piece marked invalid must contribute nothing, whatever its bounds say
    masked_piece = clone(pack)
    masked_piece.piece_b[0, -1] = -100.0        # brutally violated if it counted
    masked_piece.piece_mask[0, -1] = False
    loss_piece, grad_piece, _ = constraint_loss_and_grad(q, masked_piece)
    _, grad_clean, _ = constraint_loss_and_grad(q, pack)
    assert torch.allclose(grad_piece, grad_clean, atol=1e-7)
    assert float(loss_piece) == pytest.approx(float(constraint_loss_and_grad(q, pack)[0]))

    # a face flagged invalid must be skipped even when its bound is absurd
    brute = clone(pack)
    brute.piece_b[0, :, -1] = -100.0
    loss_brute, _, _ = constraint_loss_and_grad(q, brute)
    flagged = clone(brute)
    flagged.face_mask[0, :, -1] = False
    loss_flagged, grad_flagged, _ = constraint_loss_and_grad(q, flagged)
    loss_reference, grad_reference, _ = constraint_loss_and_grad(q, pack)
    assert float(loss_brute) > float(loss_reference)
    assert float(loss_flagged) == pytest.approx(float(loss_reference), abs=1e-6)
    assert torch.allclose(grad_flagged, grad_reference, atol=1e-6)


# --------------------------------------------------------------------------- #
# 9.1.5 empty pack
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["none", "zero_pieces", "all_false"])
def test_empty_pack_is_safe(kind):
    codec = _codec(32)
    q = _violating_controls(32)
    if kind == "none":
        pack = None
    elif kind == "zero_pieces":
        pack = empty_pack(32)
    else:
        pack = merge_packs([None], 32)
    loss, grad, stats = constraint_loss_and_grad(q, pack)
    assert torch.isfinite(grad).all()
    assert torch.count_nonzero(grad) == 0
    assert float(loss.sum()) == 0.0

    decoder = BoundaryDecoder(dtype=torch.float32)
    start_goal = torch.tensor([[[-0.9, -0.9], [0.9, 0.9]]])
    guided, gstats = clean_prediction_guidance(q, pack, decoder, start_goal,
                                               step_size=0.1, max_delta_scene=0.01)
    assert torch.isfinite(guided).all()
    assert torch.equal(guided, q)
    assert all(torch.isfinite(v).all() for v in gstats.values())


# --------------------------------------------------------------------------- #
# 9.1.6 batch independence
# --------------------------------------------------------------------------- #
def test_batch_rows_are_independent():
    codec = _codec(32)
    good_pack = _box_pack(codec)
    q_violating = _violating_controls(32)
    q_clean = torch.zeros(1, 32, 2)
    q_clean[0, :, 0] = torch.linspace(-0.5, 0.5, 32)
    batch = torch.cat([q_violating, q_clean], dim=0)

    pack_a = merge_packs([good_pack, good_pack], 32)
    pack_b = merge_packs([good_pack, None], 32)
    _, grad_a, _ = constraint_loss_and_grad(batch, pack_a)
    _, grad_b, _ = constraint_loss_and_grad(batch, pack_b)
    assert torch.allclose(grad_a[0], grad_b[0], atol=1e-7)
    assert torch.allclose(grad_a[1], grad_b[1], atol=1e-7)
    assert torch.count_nonzero(grad_b[1]) == 0, "the empty row must stay untouched"


# --------------------------------------------------------------------------- #
# 9.1.7 C is dynamic
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("num_controls", [32, 64])
def test_control_count_is_not_hardcoded(num_controls):
    codec = _codec(num_controls)
    pack = _box_pack(codec)
    q = _violating_controls(num_controls)
    loss, grad, _ = constraint_loss_and_grad(q, pack)
    assert float(loss) > 0.0
    assert grad.shape == (1, num_controls, 2)
    assert pack.num_controls == num_controls

    decoder = BoundaryDecoder(dtype=torch.float32)
    start_goal = torch.tensor([[[-0.9, 0.0], [0.9, 0.0]]])
    guided, stats = clean_prediction_guidance(q, pack, decoder, start_goal,
                                              step_size=0.1, max_delta_scene=0.01)
    assert guided.shape == (1, num_controls, 2)
    assert float(stats["safe_loss_after"]) <= float(stats["safe_loss_before"]) + 1e-6


# --------------------------------------------------------------------------- #
# 9.1.8 endpoints / heading survive the step
# --------------------------------------------------------------------------- #
def test_endpoints_and_heading_are_preserved():
    codec = _codec(32)
    pack = _box_pack(codec)
    decoder = BoundaryDecoder(dtype=torch.float32)
    start_goal = torch.tensor([[[-0.5, -0.5], [0.5, 0.5]]])
    heading = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])
    valid = torch.ones(1, 2, dtype=torch.bool)

    q = _violating_controls(32)
    q = decoder(q, start_goal, heading, valid)
    guided, _ = clean_prediction_guidance(q, pack, decoder, start_goal, heading,
                                          valid, step_size=0.5, max_delta_scene=0.02)
    assert torch.allclose(guided[:, 0], start_goal[:, 0], atol=1e-6)
    assert torch.allclose(guided[:, -1], start_goal[:, 1], atol=1e-6)

    def tangent_error(c):
        basis = codec.basis_derivative_at([0.0, 1.0]).to(dtype=c.dtype)
        t = torch.einsum("pk,bkd->bpd", basis, c)
        t = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        return (t - heading).abs().amax()

    assert float(tangent_error(guided)) <= float(tangent_error(q)) + 1e-5


# --------------------------------------------------------------------------- #
# 9.1.9 trust region
# --------------------------------------------------------------------------- #
def test_trust_region_bounds_every_control():
    codec = _codec(32)
    pack = _box_pack(codec)
    decoder = BoundaryDecoder(dtype=torch.float32)
    start_goal = torch.tensor([[[-0.9, -0.9], [0.9, 0.9]]])
    q = _violating_controls(32)
    q = decoder(q, start_goal)
    max_delta = 0.004
    guided, stats = clean_prediction_guidance(q, pack, decoder, start_goal,
                                              step_size=5.0,
                                              max_delta_scene=max_delta)
    moved = (guided - q).norm(dim=-1)
    assert float(moved.max()) <= max_delta + 1e-6
    assert float(stats["delta_max_scene"]) <= max_delta + 1e-6


# --------------------------------------------------------------------------- #
# 9.2 DDIM algebra
# --------------------------------------------------------------------------- #
def test_ddim_reconstructs_eps_from_the_guided_clean_prediction(cfg):
    from gp_diffuser.factory import build_schedule

    schedule = build_schedule(cfg)
    torch.manual_seed(0)
    q_t = torch.randn(2, 32, 2)
    q0 = torch.randn(2, 32, 2) * 0.3
    t_cur, t_prev = 700, 690
    t = torch.full((2,), t_cur, dtype=torch.long)

    eps = schedule.predict_eps_from_x0(q_t, t, q0).float()
    sqrt_ab = schedule.sqrt_alphas_cumprod[t_cur].float()
    sqrt_1ma = schedule.sqrt_one_minus_alphas_cumprod[t_cur].float()
    assert torch.allclose(sqrt_ab * q0 + sqrt_1ma * eps, q_t, atol=1e-5)

    sqrt_ab_all = schedule.sqrt_alphas_cumprod.float()
    sqrt_1ma_all = schedule.sqrt_one_minus_alphas_cumprod.float()
    q_next = ddim_step(q_t, q0, sqrt_ab_all, sqrt_1ma_all, t_cur, t_prev)

    sa_s, s1_s = sqrt_ab_all[t_prev], sqrt_1ma_all[t_prev]
    assert torch.allclose(q_next, sa_s * q0 + s1_s * eps, atol=1e-5)
    # the last step returns the clean prediction itself
    assert torch.equal(ddim_step(q_t, q0, sqrt_ab_all, sqrt_1ma_all, t_cur, -1), q0)


def test_guidance_only_changes_q0_and_leaves_ddim_untouched(cfg):
    """The guided prediction enters through q0_used, nothing else changes."""
    from gp_diffuser.factory import build_schedule

    schedule = build_schedule(cfg)
    torch.manual_seed(1)
    q_t = torch.randn(1, 32, 2)
    q0_raw = torch.randn(1, 32, 2) * 0.2
    q0_guided = q0_raw - 0.01            # any clean-space correction
    sqrt_ab = schedule.sqrt_alphas_cumprod.float()
    sqrt_1ma = schedule.sqrt_one_minus_alphas_cumprod.float()
    a = ddim_step(q_t, q0_raw, sqrt_ab, sqrt_1ma, 500, 490)
    b = ddim_step(q_t, q0_guided, sqrt_ab, sqrt_1ma, 500, 490)
    assert not torch.allclose(a, b)
    # same formula, different q0_used: the delta is exactly sa_s * d + s1_s * deps
    t = torch.full((1,), 500, dtype=torch.long)
    eps_raw = schedule.predict_eps_from_x0(q_t, t, q0_raw).float()
    eps_guided = schedule.predict_eps_from_x0(q_t, t, q0_guided).float()
    expected = (sqrt_ab[490] * q0_guided + sqrt_1ma[490] * eps_guided)
    assert torch.allclose(b, expected, atol=1e-5)
    assert not torch.allclose(eps_raw, eps_guided)
