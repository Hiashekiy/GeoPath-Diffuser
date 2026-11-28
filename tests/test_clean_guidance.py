"""CPCG unit tests on the uniform-arc-length sample pack.

Analytic gradient vs finite differences, masking, one-region-per-sample,
backtracking acceptance, dynamic control count and DDIM consistency.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from gp_diffuser.diffusion.clean_guidance import (clean_prediction_guidance,
                                                constraint_loss_and_grad,
                                                sample_constraint_values)
from gp_diffuser.diffusion.sampler_conditional_dit import ddim_step
from gp_diffuser.models.boundary import BoundaryDecoder
from gp_diffuser.safety.bspline_sample_constraints import (
    build_sampled_constraint_pack_from_tables, empty_sample_pack, region_polygons)
from gp_diffuser.safety.local_regions import box_cell_faces


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _codec(num_controls: int = 32):
    from gp_diffuser.trajectory.bspline import BSplineCodec
    return BSplineCodec(degree=3, num_controls=num_controls, curve_points=128)


def _straight(num_controls: int = 32, hump: float = 0.0) -> torch.Tensor:
    q = torch.zeros(1, num_controls, 2)
    q[0, :, 0] = torch.linspace(-0.8, 0.8, num_controls)
    if hump:
        q[0, num_controls // 4: num_controls // 2, 1] = hump
    return q


def _box_table(num_cells: int = 32, margin: float = 0.0, valid: bool = True,
               bounds=None):
    A, b = box_cell_faces(margin)
    b = np.asarray(b, dtype=np.float64) if bounds is None else np.asarray(bounds)
    cell_A = np.zeros((num_cells, len(b), 2))
    cell_b = np.full((num_cells, len(b)), np.inf)
    mask = np.zeros(num_cells, dtype=bool)
    if valid:
        cell_A[:] = A[:len(b)]
        cell_b[:] = b
        mask[:] = True
    return {"cell_A": cell_A, "cell_b": cell_b, "cell_mask": mask,
            "cell_is_box": np.ones(num_cells, dtype=bool),
            "anchors": np.linspace(0.0, 1.0, num_cells)}


def _box_pack(codec, margin: float = 0.0, valid: bool = True, points: int = 64,
              reference=None, dtype=torch.float32):
    table = _box_table(32, margin, valid)
    ref = _straight(codec.num_controls) if reference is None else reference
    pack = build_sampled_constraint_pack_from_tables(
        codec, [table], ref, constraint_points=points,
        arc_length_lookup_points=256, dtype=dtype)
    return pack


def _violating_controls(num_controls: int = 32) -> torch.Tensor:
    q = torch.zeros(1, num_controls, 2)
    q[0, num_controls // 4: num_controls // 2] = torch.tensor([1.45, 0.0])
    return q


def _masked(pack, **changes):
    fields = dict(basis=pack.basis.clone(), sample_s=pack.sample_s.clone(),
                  sample_u=pack.sample_u.clone(), sample_A=pack.sample_A.clone(),
                  sample_b=pack.sample_b.clone(), face_mask=pack.face_mask.clone(),
                  sample_mask=pack.sample_mask.clone(),
                  region_id=pack.region_id.clone(),
                  sample_is_box=pack.sample_is_box.clone(),
                  diagnostics=pack.diagnostics)
    fields.update(changes)
    return pack.__class__(**fields)


# --------------------------------------------------------------------------- #
# 9.1.1 finite differences
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("k, d", [(3, 0), (8, 0), (12, 1)])
def test_analytic_gradient_matches_finite_difference(k, d):
    codec = _codec(32)
    pack = _box_pack(codec, dtype=torch.float64)
    q = _violating_controls(32).double()
    _, grad, _ = constraint_loss_and_grad(q, pack)
    assert float(grad.abs().max()) > 0.0

    eps = 1e-6
    plus = q.clone()
    plus[0, k, d] += eps
    minus = q.clone()
    minus[0, k, d] -= eps
    fd = ((constraint_loss_and_grad(plus, pack)[0]
           - constraint_loss_and_grad(minus, pack)[0]) / (2.0 * eps))
    assert torch.allclose(grad[0, k, d], fd[0], atol=1e-8, rtol=1e-4), \
        "analytic %r vs finite difference %r" % (float(grad[0, k, d]), float(fd[0]))


def test_gradient_step_reduces_the_safety_cost():
    codec = _codec(32)
    pack = _box_pack(codec)
    q = _violating_controls(32)
    before, grad, _ = constraint_loss_and_grad(q, pack)
    after, _, _ = constraint_loss_and_grad(q - 5.0 * grad, pack)
    assert float(after) < float(before)


# --------------------------------------------------------------------------- #
# 9.1.3 satisfied constraints -> exactly zero guidance
# --------------------------------------------------------------------------- #
def test_no_violation_means_zero_gradient_and_no_motion():
    codec = _codec(32)
    pack = _box_pack(codec)
    q = _straight(32)
    loss, grad, stats = constraint_loss_and_grad(q, pack)
    assert float(loss) == 0.0
    assert torch.count_nonzero(grad) == 0
    assert not bool(stats["has_violation"].any())

    decoder = BoundaryDecoder(dtype=torch.float32)
    start_goal = torch.tensor([[[-0.5, 0.0], [0.5, 0.0]]])
    guided, gstats = clean_prediction_guidance(q, pack, decoder, start_goal,
                                               step_size=50.0)
    assert torch.equal(guided, q), "a satisfied pack must return q0 bit-for-bit"
    assert not bool(gstats["guidance_applied"].any())


# --------------------------------------------------------------------------- #
# 9.1.4 masked sample / face must not contribute
# --------------------------------------------------------------------------- #
def test_masked_samples_and_faces_produce_no_gradient():
    codec = _codec(32)
    pack = _box_pack(codec)
    q = _violating_controls(32)
    loss_reference, grad_reference, _ = constraint_loss_and_grad(q, pack)

    # a masked SAMPLE must contribute nothing, whatever its bounds say
    masked_sample = _masked(pack)
    masked_sample.sample_b[0, 3] = -100.0
    masked_sample.sample_mask[0, 3] = False
    loss, grad, _ = constraint_loss_and_grad(q, masked_sample)
    assert torch.allclose(grad, grad_reference, atol=1e-7)
    assert float(loss) == pytest.approx(float(loss_reference))

    # a face flagged invalid must be skipped even when its bound is absurd
    brute = _masked(pack)
    brute.sample_b[0, :, -1] = -100.0
    loss_brute, _, _ = constraint_loss_and_grad(q, brute)
    flagged = _masked(brute, face_mask=brute.face_mask.clone())
    flagged.face_mask[0, :, -1] = False
    loss_flagged, grad_flagged, _ = constraint_loss_and_grad(q, flagged)
    assert float(loss_brute) > float(loss_reference)
    assert float(loss_flagged) == pytest.approx(float(loss_reference), abs=1e-6)
    assert torch.allclose(grad_flagged, grad_reference, atol=1e-6)


# --------------------------------------------------------------------------- #
# 9.1.5 empty pack
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["none", "empty"])
def test_empty_pack_is_safe(kind):
    codec = _codec(32)
    q = _violating_controls(32)
    pack = None if kind == "none" else empty_sample_pack(32, num_samples=16)
    loss, grad, stats = constraint_loss_and_grad(q, pack)
    assert torch.isfinite(grad).all()
    assert torch.count_nonzero(grad) == 0
    assert float(loss.sum()) == 0.0

    decoder = BoundaryDecoder(dtype=torch.float32)
    start_goal = torch.tensor([[[-0.9, -0.9], [0.9, 0.9]]])
    guided, gstats = clean_prediction_guidance(q, pack, decoder, start_goal,
                                              step_size=50.0)
    assert torch.isfinite(guided).all()
    assert torch.equal(guided, q)
    assert all(torch.isfinite(v).all() for v in gstats.values())


# --------------------------------------------------------------------------- #
# 9.1.6 batch independence
# --------------------------------------------------------------------------- #
def test_batch_rows_are_independent():
    codec = _codec(32)
    q_clean = _straight(32)
    q_bad = _violating_controls(32)
    batch = torch.cat([q_bad, q_clean], dim=0)
    # both packs sample the SAME reference for row 0, so the comparison isolates
    # the per-sample masking instead of the (legitimate) arc-length difference
    table = _box_table(32)
    pack_both = build_sampled_constraint_pack_from_tables(
        codec, [table, table], torch.cat([q_bad, q_clean], dim=0),
        constraint_points=64, arc_length_lookup_points=256)
    _, grad_a, _ = constraint_loss_and_grad(batch, pack_both)
    single = build_sampled_constraint_pack_from_tables(
        codec, [table], q_bad, constraint_points=64, arc_length_lookup_points=256)
    _, grad_b, _ = constraint_loss_and_grad(batch[:1], single)
    assert float(grad_a[0].abs().max()) > 0.0
    assert torch.allclose(grad_a[0], grad_b[0], atol=1e-6)
    assert torch.count_nonzero(grad_a[1]) == 0, "the clean row must stay untouched"

    # a fully masked row must not move either
    masked = _masked(pack_both, sample_mask=torch.zeros_like(pack_both.sample_mask))
    _, grad_masked, _ = constraint_loss_and_grad(batch, masked)
    assert torch.count_nonzero(grad_masked) == 0


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
                                              step_size=20.0)
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
                                          valid, step_size=30.0)
    assert torch.allclose(guided[:, 0], start_goal[:, 0], atol=1e-6)
    assert torch.allclose(guided[:, -1], start_goal[:, 1], atol=1e-6)

    def tangent_error(c):
        basis = codec.basis_derivative_at([0.0, 1.0]).to(dtype=c.dtype)
        t = torch.einsum("pk,bkd->bpd", basis, c)
        t = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        return (t - heading).abs().amax()

    assert float(tangent_error(guided)) <= float(tangent_error(q)) + 1e-5


# --------------------------------------------------------------------------- #
# 9.2 DDIM algebra (unchanged)
# --------------------------------------------------------------------------- #
def test_ddim_reconstructs_eps_from_the_guided_clean_prediction(cfg):
    from gp_diffuser.factory import build_schedule

    schedule = build_schedule(cfg)
    torch.manual_seed(0)
    q_t = torch.randn(2, 32, 2)
    q0 = torch.randn(2, 32, 2) * 0.3
    table = int(schedule.num_timesteps)
    t_cur, t_prev = max(table // 2, 1), max(table // 2 - 1, 0)
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
    assert torch.equal(ddim_step(q_t, q0, sqrt_ab_all, sqrt_1ma_all, t_cur, -1), q0)


def test_guidance_only_changes_q0_and_leaves_ddim_untouched(cfg):
    """The guided prediction enters through q0_used, nothing else changes."""
    from gp_diffuser.factory import build_schedule

    schedule = build_schedule(cfg)
    torch.manual_seed(1)
    q_t = torch.randn(1, 32, 2)
    q0_raw = torch.randn(1, 32, 2) * 0.2
    q0_guided = q0_raw - 0.01
    sqrt_ab = schedule.sqrt_alphas_cumprod.float()
    sqrt_1ma = schedule.sqrt_one_minus_alphas_cumprod.float()
    table = int(schedule.num_timesteps)
    t_cur, t_prev = max(table // 2, 1), max(table // 2 - 1, 0)
    a = ddim_step(q_t, q0_raw, sqrt_ab, sqrt_1ma, t_cur, t_prev)
    b = ddim_step(q_t, q0_guided, sqrt_ab, sqrt_1ma, t_cur, t_prev)
    assert not torch.allclose(a, b)
    t = torch.full((1,), t_cur, dtype=torch.long)
    eps_raw = schedule.predict_eps_from_x0(q_t, t, q0_raw).float()
    eps_guided = schedule.predict_eps_from_x0(q_t, t, q0_guided).float()
    expected = (sqrt_ab[t_prev] * q0_guided + sqrt_1ma[t_prev] * eps_guided)
    assert torch.allclose(b, expected, atol=1e-5)
    assert not torch.allclose(eps_raw, eps_guided)


# --------------------------------------------------------------------------- #
# cost definition: fixed denominator M, no violated-count normalisation
# --------------------------------------------------------------------------- #
def test_cost_is_the_fixed_denominator_hinge():
    codec = _codec(32)
    pack = _box_pack(codec, points=64)
    q = _violating_controls(32)
    loss, _, stats = constraint_loss_and_grad(q, pack)
    _, raw, valid = sample_constraint_values(q, pack)
    v = torch.relu(raw) * valid
    M = pack.basis.shape[1]
    assert torch.allclose(loss, 0.5 * v.square().sum(dim=(1, 2)) / M, atol=1e-7)
    assert M == 64
    assert float(stats["violated_count"][0]) > 1


def test_cost_is_continuous_when_a_new_constraint_starts_violating():
    """Deepening a violation must never LOWER the cost."""
    codec = _codec(32)
    pack = _box_pack(codec)
    ref = _straight(32)
    depths = [1.0, 1.05, 1.10, 1.15, 1.20, 1.30, 1.45]
    losses, counts = [], []
    for depth in depths:
        q = ref.clone()
        q[0, 6:12, 0] = 1.45
        q[0, 20:26, 1] = depth
        loss, _, stats = constraint_loss_and_grad(q, pack)
        losses.append(float(loss))
        counts.append(float(stats["violated_count"][0]))
    assert max(counts) > min(counts), "the violated count must actually change here"
    for i in range(len(depths) - 1):
        assert losses[i + 1] >= losses[i] - 1e-9, (
            "cost dropped from %.6f to %.6f when the count went %.0f -> %.0f"
            % (losses[i], losses[i + 1], counts[i], counts[i + 1]))


def test_a_sample_is_constrained_only_by_its_own_region():
    """Region 0 is impossible, region 1 is the plain box: only region 0 violates."""
    codec = _codec(32)
    ref = _straight(32)
    impossible = np.array([-1.0, -1.0, -1.0, -1.0])     # x<=-1 and -x<=-1
    ok = np.array([1.0, 1.0, 1.0, 1.0])
    n = 32
    cell_A = np.repeat(box_cell_faces(0.0)[0][None], n, axis=0)
    cell_b = np.repeat(ok[None], n, axis=0)
    cell_b[: n // 2] = impossible                       # only the first half
    table = {"cell_A": cell_A, "cell_b": cell_b,
             "cell_mask": np.ones(n, dtype=bool),
             "cell_is_box": np.zeros(n, dtype=bool),
             "anchors": np.linspace(0.0, 1.0, n)}
    pack = build_sampled_constraint_pack_from_tables(
        codec, [table], ref, constraint_points=64, arc_length_lookup_points=256)
    _, raw, valid = sample_constraint_values(ref, pack)
    violated = (torch.relu(raw) > 0) & valid
    per_sample = violated.any(dim=-1)[0]
    rid = pack.region_id[0]
    assert per_sample.any() and not per_sample.all()
    assert bool((rid[per_sample] < n // 2).all()), \
        "a violated sample must live in the impossible region"
    safe = pack.sample_mask[0] & ~per_sample
    assert bool((rid[safe] >= n // 2).all()), \
        "a safe sample must be judged by its own (satisfiable) region"


# --------------------------------------------------------------------------- #
# backtracking acceptance
# --------------------------------------------------------------------------- #
def test_a_step_that_cannot_improve_is_rejected():
    """Strict decrease or nothing: a hopeless trial leaves Q alone."""
    codec = _codec(32)
    pack = _box_pack(codec)
    decoder = BoundaryDecoder(dtype=torch.float32)
    start_goal = torch.tensor([[[-0.9, -0.9], [0.9, 0.9]]])
    q = decoder(_violating_controls(32), start_goal)
    guided, stats = clean_prediction_guidance(q, pack, decoder, start_goal,
                                              step_size=1.0,
                                              backtrack_factors=(1e-12,))
    assert torch.equal(guided, q), "no decrease -> the guidance must be a no-op"
    assert not bool(stats["accepted"][0])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_acceptance_never_increases_the_cost(seed):
    codec = _codec(32)
    pack = _box_pack(codec)
    decoder = BoundaryDecoder(dtype=torch.float32)
    torch.manual_seed(seed)
    q = torch.randn(4, 32, 2) * 0.6
    start_goal = torch.tensor([[[-0.7, -0.7], [0.7, 0.7]]]).repeat(4, 1, 1)
    q = decoder(q, start_goal)
    before, _, _ = constraint_loss_and_grad(q, pack)
    guided, stats = clean_prediction_guidance(q, pack, decoder, start_goal,
                                              step_size=50.0)
    after, _, _ = constraint_loss_and_grad(guided, pack)
    accepted = stats["accepted"]
    assert bool(accepted.any()), "at least one sample should improve"
    assert torch.all(after[accepted] < before[accepted] + 1e-9)
    rejected = stats["rejected"]
    assert torch.equal(guided[rejected], q[rejected]), "rejected rows stay untouched"


def test_zero_step_size_produces_no_motion():
    codec = _codec(32)
    pack = _box_pack(codec)
    decoder = BoundaryDecoder(dtype=torch.float32)
    start_goal = torch.tensor([[[-0.9, -0.9], [0.9, 0.9]]])
    q = decoder(_violating_controls(32), start_goal)
    guided, stats = clean_prediction_guidance(q, pack, decoder, start_goal,
                                              step_size=0.0)
    assert torch.equal(guided, q)
    assert not bool(stats["guidance_applied"].any())


# --------------------------------------------------------------------------- #
# the gradient step is the REAL step (no |delta|_max re-normalisation)
# --------------------------------------------------------------------------- #
def test_step_is_the_real_gradient_step():
    codec = _codec(32)
    pack = _box_pack(codec)
    decoder = BoundaryDecoder(dtype=torch.float32)
    start_goal = torch.tensor([[[-0.9, -0.9], [0.9, 0.9]]])
    q = decoder(_violating_controls(32), start_goal)
    step_size = 50.0
    guided, stats = clean_prediction_guidance(q, pack, decoder, start_goal,
                                              step_size=step_size)
    moved = float((guided - q).norm(dim=-1).max())
    assert moved > 0.0
    assert moved <= step_size * float(stats["grad_norm"][0]) + 1e-6, \
        "the displacement must not exceed the raw gradient step (no re-scaling)"


def test_repeated_guidance_reduces_the_violation_monotonically():
    """The point of CPCG: every accepted step must shrink the violation.

    The frozen pack is sampled from the ORIGINAL (violating) reference curve, so
    the sample parameters never change; only ``Q`` moves, and the sampled
    violation must keep falling until the corridor is reached.
    """
    codec = _codec(32)
    decoder = BoundaryDecoder(dtype=torch.float32)
    start_goal = torch.tensor([[[-0.8, 0.0], [0.8, 0.0]]])
    q = decoder(_violating_controls(32), start_goal)
    pack = _box_pack(codec, reference=decoder(_violating_controls(32), start_goal))

    trace = []
    for _ in range(40):
        before = float(constraint_loss_and_grad(q, pack)[0])
        q, stats = clean_prediction_guidance(q, pack, decoder, start_goal,
                                             step_size=20.0)
        trace.append((float(stats["max_violation_before"][0]),
                      float(stats["max_violation_after"][0]),
                      float(stats["safe_loss_after"][0])))
        assert trace[-1][2] <= before + 1e-6, "the accepted step must not raise J"
        assert trace[-1][1] <= trace[-1][0] + 1e-6, "a step must not deepen it"

    violations = [t[1] for t in trace]
    assert violations[0] > 0.0 and violations[-1] < violations[0]
    for a, b in zip(violations, violations[1:]):
        assert b <= a + 1e-6, "the violation track must be monotone: %s" % (violations,)
    assert violations[-1] < 1e-3, "guidance must converge into the corridor: %s" % (
        violations[-5:],)
