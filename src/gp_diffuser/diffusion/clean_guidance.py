"""Clean-Prediction Constraint Guidance (CPCG) on uniform-arc-length samples.

The constraint cost is defined on the CLEAN B-spline controls the network just
predicted, never on the noisy state ``q_t``, and it is evaluated on ``M`` real
curve samples of the frozen
:class:`~gp_diffuser.safety.bspline_sample_constraints.BSplineSampleConstraintPack`::

    P_l          = N_l Q                          sample points of the curve
    g[l,f]       = a_{j(l),f} . P_l - b_{j(l),f}  half-space value (<= 0 is safe)
    J(Q)         = 1/(2M) sum_{l,f} m[l,f] relu(g + delta)^2

Sample ``l`` is constrained ONLY by the single convex cell ``j(l)`` it was
assigned to when the safety geometry was built.  One sample point never carries
all regions, and the assignment never changes with ``Q``: it is frozen by the
reference arc-length progress ``s_l`` and the frozen cell anchors.

Because ``N_l``, ``a_{j(l)}``, ``b_{j(l)}`` are constants, the gradient is
analytic and exact (no autograd, no backward pass through the DiT)::

    d g[l,f] / dQ[k,d] = a_{j(l),f,d} N_l[k]
    dJ/dQ[k,d]         = 1/M sum_{l,f} relu(g+delta) m a_{j(l),f,d} N_l[k]

The denominator is the FIXED sample count ``M``; dividing by the number of
currently violated rows would make the cost discontinuous at every constraint
crossing and would shrink the step by the violation count.

Update rule (descent -> boundary projection -> backtracking)::

    d        = -dJ/dQ                                  (endpoints zeroed)
    for a in (1, 0.5, 0.25, 0.125, 0.0625):
        Q_try = BoundaryDecoder(Q + a * step_size * d)
        accept the FIRST trial with J(Q_try) < J(Q)
    otherwise REJECT and return ``Q`` bit-for-bit

``step_size`` is the real initial gradient step length in scene units per unit of
gradient; there is NO re-normalisation of the displacement to a fixed
``max_delta_m`` (that used to amplify tiny gradients into a fixed 0.5 m jump).
Backtracking is what keeps an overshooting step inside the corridor, so
``step_size`` must be re-scanned whenever the loss scale changes.

The guided clean prediction enters the sampler through the EXISTING ``q0_used``
interface: the DDIM update rebuilds ``eps`` from ``(q_t, q0_used)``, so no extra
forward, no extra noise and no extra scheduler math are needed.

Deliberately NOT here: Bezier extraction / Bezier control points, ALM, dual
variables, penalties on ``q_t``, and anything that treats sample points as
independent optimisation variables.  The only primal variable is the global
B-spline control polygon.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch

from ..safety.bspline_sample_constraints import sample_pack_is_empty

__all__ = [
    "sample_constraint_values",
    "constraint_loss_and_grad",
    "violation_stats",
    "clean_prediction_guidance",
    "DEFAULT_BACKTRACK_FACTORS",
]

#: line-search factors tried in order; the first strict decrease wins
DEFAULT_BACKTRACK_FACTORS: Tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625)


def _empty_stats(q0: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Values of a pack that constrains nothing."""
    B = int(q0.shape[0])
    zero = q0.new_zeros(B)
    return {"loss": zero.clone(), "max_violation": zero.clone(),
            "violated_count": zero.clone(), "valid_count": zero.clone(),
            "grad_norm": zero.clone(),
            "has_violation": torch.zeros(B, dtype=torch.bool, device=q0.device)}


def sample_constraint_values(q0: torch.Tensor, pack, margin_scene: float = 0.0
                             ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(points [B,M,2], raw [B,M,F], valid [B,M,F])`` of the sampled constraints.

    ``raw = a_{j(l)} . P_l - b_{j(l)} + margin_scene``: positive means that curve
    sample lies outside its responsible convex cell (plus the extra margin).
    """
    points = torch.einsum("bmc,bcd->bmd", pack.basis, q0)
    g = torch.einsum("bmfd,bmd->bmf", pack.sample_A, points) - pack.sample_b
    valid = (pack.sample_mask[:, :, None] & pack.face_mask).expand_as(g)
    return points, g + float(margin_scene), valid


def violation_stats(raw: torch.Tensor, valid: torch.Tensor, loss: torch.Tensor,
                    grad_norm: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """Per-sample scalars of one constraint evaluation ``[B]`` each."""
    violated = valid & (raw > 0.0)
    with torch.no_grad():
        max_violation = (torch.relu(raw) * valid).flatten(1).max(dim=1).values \
            if raw.shape[1] else torch.zeros(raw.shape[0], dtype=raw.dtype,
                                             device=raw.device)
        stats = {
            "loss": loss,
            "max_violation": max_violation,
            "violated_count": violated.sum(dim=(1, 2)).to(raw.dtype),
            "valid_count": valid.sum(dim=(1, 2)).to(raw.dtype),
            "has_violation": violated.any(dim=(1, 2)),
        }
    if grad_norm is not None:
        stats["grad_norm"] = grad_norm
    return stats


def constraint_loss_and_grad(q0: torch.Tensor, pack, margin_scene: float = 0.0
                             ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """``(loss [B], grad [B,C,2], stats)`` of the frozen sample pack.

    ``J = 1/(2M) sum relu(g + delta)^2`` with the FIXED sample count ``M``:
    continuous in ``Q`` and with the natural scale of the violation.  A sample
    with nothing violated gets an exactly zero gradient, so guidance cannot move
    it through numerical noise.
    """
    if sample_pack_is_empty(pack):
        zeros = torch.zeros_like(q0)
        return q0.new_zeros(int(q0.shape[0])), zeros, _empty_stats(q0)

    pack = pack.to(device=q0.device, dtype=q0.dtype)
    _, raw, valid = sample_constraint_values(q0, pack, margin_scene)
    v = torch.relu(raw) * valid                       # [B,M,F]
    M = max(int(pack.basis.shape[1]), 1)
    loss = 0.5 * v.square().sum(dim=(1, 2)) / float(M)
    grad = torch.einsum("bmf,bmfd,bmc->bcd", v, pack.sample_A, pack.basis) / float(M)
    stats = violation_stats(raw, valid, loss.detach(),
                            grad_norm=grad.detach().norm(dim=(1, 2)))
    return loss, grad, stats


def _loss_at(q0: torch.Tensor, pack, margin_scene: float) -> torch.Tensor:
    """``J(Q)`` without its gradient (used by the backtracking line search)."""
    if sample_pack_is_empty(pack):
        return q0.new_zeros(int(q0.shape[0]))
    _, raw, valid = sample_constraint_values(q0, pack, margin_scene)
    M = max(int(pack.basis.shape[1]), 1)
    return 0.5 * (torch.relu(raw) * valid).square().sum(dim=(1, 2)) / float(M)


def clean_prediction_guidance(
    q0: torch.Tensor,
    pack,
    boundary_decoder,
    start_goal: torch.Tensor,
    heading: Optional[torch.Tensor] = None,
    heading_valid: Optional[torch.Tensor] = None,
    step_size: float = 10.0,
    margin_scene: float = 0.0,
    backtrack_factors: Optional[Sequence[float]] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """One accepted descent step on the clean control polygon.

    ``step_size`` is the initial gradient step (scene units); backtracking over
    ``backtrack_factors`` shrinks an overshooting step, and only a trial with a
    STRICT decrease of ``J`` is accepted - otherwise ``q0`` comes back unchanged.
    """
    factors = tuple(float(a) for a in (backtrack_factors or DEFAULT_BACKTRACK_FACTORS))
    empty = sample_pack_is_empty(pack)
    if not empty:
        pack = pack.to(device=q0.device, dtype=q0.dtype)
    loss_before, grad, stats_before = constraint_loss_and_grad(q0, pack, margin_scene)
    grad = grad.clone()
    if grad.shape[1] >= 2:
        grad[:, 0] = 0.0
        grad[:, -1] = 0.0

    has = stats_before["has_violation"]
    B = int(q0.shape[0])
    zero = q0.new_zeros(B)

    # ---- descent direction (endpoints are hard conditions, not variables) ---
    direction = -float(step_size) * grad
    direction_max = direction.norm(dim=-1).amax(dim=-1)                # [B]
    usable = has & (direction_max > 0.0)

    # ---- backtracking line search, per sample ------------------------------
    accepted = torch.zeros(B, dtype=torch.bool, device=q0.device)
    alpha = zero.clone()
    best = q0.clone()
    best_loss = loss_before.detach().clone()
    trials = 0
    for a in factors:
        pending = usable & ~accepted
        if not bool(pending.any()):
            break
        trials += 1
        candidate = boundary_decoder(q0 + a * direction, start_goal, heading,
                                     heading_valid)
        loss_try = _loss_at(candidate, pack, margin_scene)
        improve = pending & (loss_try < best_loss)
        if bool(improve.any()):
            best = torch.where(improve[:, None, None], candidate, best)
            best_loss = torch.where(improve, loss_try, best_loss)
            alpha = torch.where(improve, torch.full_like(alpha, a), alpha)
            accepted = accepted | improve

    q0_guided = torch.where(accepted[:, None, None], best, q0)

    if empty:
        stats_after = _empty_stats(q0)
    else:
        _, raw, valid = sample_constraint_values(q0_guided, pack, margin_scene)
        stats_after = violation_stats(raw, valid, best_loss.detach())

    delta = q0_guided - q0
    delta_norm = delta.norm(dim=-1)                                    # [B,C]
    stats = {
        "safe_loss_before": loss_before.detach(),
        "safe_loss_after": best_loss,
        "safe_loss_ratio": torch.where(loss_before.detach() > 0,
                                       best_loss / loss_before.detach().clamp_min(1e-12),
                                       torch.ones_like(loss_before.detach())),
        "max_violation_before": stats_before["max_violation"],
        "max_violation_after": stats_after["max_violation"],
        "violated_count_before": stats_before["violated_count"],
        "violated_count_after": stats_after["violated_count"],
        "grad_norm": stats_before["grad_norm"],
        "step_size": torch.full_like(zero, float(step_size)),
        "delta_rms_scene": delta_norm.square().mean(dim=-1).sqrt(),
        "delta_max_scene": delta_norm.amax(dim=-1),
        "accepted": accepted,
        "rejected": usable & ~accepted,
        "alpha": alpha,
        "backtrack_trials": torch.full_like(zero, float(trials)),
        "guidance_applied": accepted,
    }
    return q0_guided, stats
