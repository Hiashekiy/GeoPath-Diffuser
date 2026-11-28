"""Clean-Prediction Constraint Guidance (CPCG).

The constraint cost is defined on the CLEAN B-spline controls the network just
predicted, never on the noisy state ``q_t``:

    beta[p,r]   = E[p,r] Q                        exact cubic Bezier controls
    g[p,r,f]    = A[p,f] beta[p,r] - b[p,f]       half-space value (g <= 0 is safe)
    J(Q)        = 0.5 * sum M[p,r,f] * max(0, g + margin)^2

**No violated-count normalisation.**  Dividing by ``N_viol(Q)`` makes the cost
discontinuous whenever a constraint crosses zero (one new violation halves the
whole cost even if its own value is ~0), and it shrinks the gradient by the
number of violated scalars - which on a real corridor is several hundred, i.e.
millimetre-sized control steps.  The un-normalised sum is continuous, has the
same minimiser (the corridor), and its gradient has the natural scale of the
violation itself.

Because every ``E`` / ``A`` / ``b`` is a frozen constant, ``dJ/dQ`` is analytic:

    d g[p,r,f]/dQ[k,d] = A[p,f,d] E[p,r,k]
    dJ/dQ = sum_{p,r,f} relu(g+margin) * M * A[p,f,:] * E[p,r,:]

so there is no autograd and no backward pass through the DiT.  ``sample()``
keeps its ``@torch.no_grad()``.

Update rule (direction -> trust region -> backtracking):

    1. d       = -dJ/dQ                                  (endpoints zeroed)
    2. Q_tmp   = BoundaryDecoder(Q + step_size * d)      (endpoints/heading kept)
    3. d_TR    = (Q_tmp - Q) normalised to |.|_max = max_delta_scene
    4. for a in (1, 0.5, 0.25, ...):  Q_try = BoundaryDecoder(Q + a * d_TR)
       accept the first ``a`` with J(Q_try) < J(Q); otherwise REJECT the step
Q_try is re-projected through the decoder before ``J`` is evaluated, so the
measured decrease belongs to the polygon that would really be used.

The guided clean prediction enters the sampler through the EXISTING ``q0_used``
interface: the DDIM update already rebuilds ``eps`` from ``(q_t, q0_used)``, so
no extra forward, no extra noise and no extra scheduler math are needed.

Deliberately NOT here: ALM, dual variables, collision penalties on ``q_t``, and
anything that treats the individual Bezier controls as independent optimisation
variables.  The only primal variable is the global B-spline control polygon.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch

__all__ = [
    "constraint_values",
    "constraint_loss_and_grad",
    "violation_stats",
    "clean_prediction_guidance",
    "DEFAULT_BACKTRACK_FACTORS",
]

#: line-search factors tried in order; the first strict decrease wins
DEFAULT_BACKTRACK_FACTORS: Tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625)


def _empty_stats(q0: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Values of a pack that constrains nothing (``None``, no piece, no face)."""
    B = int(q0.shape[0])
    zero = q0.new_zeros(B)
    return {"loss": zero.clone(), "max_violation": zero.clone(),
            "violated_count": zero.clone(), "valid_count": zero.clone(),
            "grad_norm": zero.clone(),
            "has_violation": torch.zeros(B, dtype=torch.bool, device=q0.device)}


def _is_empty(pack, q0: torch.Tensor) -> bool:
    return (pack is None or pack.extraction.shape[1] == 0
            or pack.piece_A.shape[2] == 0
            or int(pack.piece_mask.sum()) == 0)


def constraint_values(q0: torch.Tensor, pack, margin_scene: float = 0.0
                      ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(beta [B,P,4,2], raw [B,P,4,F], valid [B,P,4,F])`` for a clean polygon.

    ``raw = g + margin_scene``: positive means the Bezier control is outside its
    responsible convex region (plus the extra margin, if any).
    """
    beta = torch.einsum("bprk,bkd->bprd", pack.extraction, q0)
    g = (torch.einsum("bpfd,bprd->bprf", pack.piece_A, beta)
         - pack.piece_b[:, :, None, :])
    valid = (pack.piece_mask[:, :, None, None]
             & pack.face_mask[:, :, None, :]).expand_as(g)
    return beta, g + float(margin_scene), valid


def violation_stats(raw: torch.Tensor, valid: torch.Tensor, loss: torch.Tensor,
                    grad_norm: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """Per-sample scalars of one constraint evaluation ``[B]`` each."""
    violated = valid & (raw > 0.0)
    has = violated.any(dim=(1, 2, 3)) if raw.dim() == 4 else \
        torch.zeros(raw.shape[0], dtype=torch.bool, device=raw.device)
    stats = {
        "loss": loss,
        "max_violation": (torch.relu(raw) * valid).flatten(1).max(dim=1).values
        if raw.dim() == 4 and raw.shape[1] and raw.shape[3]
        else torch.zeros(raw.shape[0], dtype=raw.dtype, device=raw.device),
        "violated_count": violated.sum(dim=(1, 2, 3)).to(raw.dtype)
        if raw.dim() == 4 else torch.zeros(raw.shape[0], dtype=raw.dtype,
                                           device=raw.device),
        "valid_count": valid.sum(dim=(1, 2, 3)).to(raw.dtype)
        if raw.dim() == 4 else torch.zeros(raw.shape[0], dtype=raw.dtype,
                                           device=raw.device),
        "has_violation": has,
    }
    if grad_norm is not None:
        stats["grad_norm"] = grad_norm
    return stats


def constraint_loss_and_grad(q0: torch.Tensor, pack, margin_scene: float = 0.0
                             ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """Masked squared hinge of the frozen pack and its analytic gradient.

    ``loss_per_sample [B]``, ``grad_q0 [B,C,2]`` and a stats dict.  The sum runs
    over EVERY valid violated scalar with NO count normalisation: the cost is
    continuous in ``Q`` (adding one more violation changes ``J`` by that
    violation's own quadratic term only), and its gradient is not diluted by the
    hundreds of faces a real corridor carries.

    A sample with no violated constraint gets an exactly zero gradient (and a
    zero loss), so guidance cannot move it through numerical noise.
    """
    if _is_empty(pack, q0):
        zeros = torch.zeros_like(q0)
        return q0.new_zeros(int(q0.shape[0])), zeros, _empty_stats(q0)

    pack = pack.to(device=q0.device, dtype=q0.dtype)
    _, raw, valid = constraint_values(q0, pack, margin_scene)
    violated = valid & (raw > 0.0)
    v = torch.relu(raw) * valid                     # [B,P,4,F], masked hinge value
    loss = 0.5 * v.square().sum(dim=(1, 2, 3))

    grad = torch.einsum("bprf,bpfd,bprk->bkd", v, pack.piece_A, pack.extraction)
    has = violated.any(dim=(1, 2, 3))
    grad = torch.where(has[:, None, None], grad, torch.zeros_like(grad))
    stats = violation_stats(raw, valid, loss, grad_norm=grad.norm(dim=(1, 2)))
    return loss, grad, stats


def _loss_at(q0: torch.Tensor, pack, margin_scene: float) -> torch.Tensor:
    """``J(Q)`` without its gradient (used by the backtracking line search)."""
    _, raw, valid = constraint_values(q0, pack, margin_scene)
    return 0.5 * (torch.relu(raw) * valid).square().sum(dim=(1, 2, 3))


def clean_prediction_guidance(
    q0: torch.Tensor,
    pack,
    boundary_decoder,
    start_goal: torch.Tensor,
    heading: Optional[torch.Tensor] = None,
    heading_valid: Optional[torch.Tensor] = None,
    step_size: float = 1.0,
    max_delta_scene: float = 0.01,
    margin_scene: float = 0.0,
    backtrack_factors: Optional[Sequence[float]] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """One accepted descent step on the clean polygon.

    1. analytic gradient of the corridor cost at ``q0`` (endpoints zeroed - they
       are hard conditions, not free variables);
    2. ``Q + step_size * (-grad)`` re-projected through ``BoundaryDecoder`` so the
       endpoint / heading conditions survive;
    3. the resulting displacement is NORMALISED so its largest control move is
       exactly ``max_delta_scene`` - the trust region is the step length, not a
       cap that is never reached;
    4. backtracking over ``backtrack_factors``; every trial is re-projected
       through the decoder and its real cost is measured;
    5. the first trial with ``J < J_before`` is accepted, otherwise the step is
       REJECTED and ``q0`` comes back bit-for-bit.

    Returns ``(q0_guided, stats)``.  No autograd, no network call, no ALM.
    """
    factors = tuple(float(a) for a in (backtrack_factors or DEFAULT_BACKTRACK_FACTORS))
    empty = _is_empty(pack, q0)
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

    # ---- 1-3: descent direction -> boundary projection -> trust-region length
    q0_tmp = q0 - float(step_size) * grad
    delta_eff = boundary_decoder(q0_tmp, start_goal, heading, heading_valid) - q0
    delta_max = delta_eff.norm(dim=-1).amax(dim=-1)                    # [B]
    usable = has & (delta_max > 0.0)
    tr_scale = torch.where(usable,
                           float(max_delta_scene) / (delta_max + 1e-12),
                           torch.zeros_like(delta_max))
    step = delta_eff * tr_scale[:, None, None]                         # |.|_max = max_delta

    # ---- 4: backtracking line search, per sample
    accepted = torch.zeros(B, dtype=torch.bool, device=q0.device)
    alpha = zero.clone()
    best = q0.clone()
    best_loss = loss_before.clone()
    trials = 0
    for a in factors:
        pending = usable & ~accepted
        if not bool(pending.any()):
            break
        trials += 1
        candidate = boundary_decoder(q0 + a * step, start_goal, heading, heading_valid)
        loss_try = _loss_at(candidate, pack, margin_scene)
        improve = pending & (loss_try < loss_before)
        if bool(improve.any()):
            best = torch.where(improve[:, None, None], candidate, best)
            best_loss = torch.where(improve, loss_try, best_loss)
            alpha = torch.where(improve, torch.full_like(alpha, a), alpha)
            accepted = accepted | improve

    q0_guided = torch.where(accepted[:, None, None], best, q0)

    if empty:
        stats_after = _empty_stats(q0)
    else:
        stats_after = violation_stats(*constraint_values(
            q0_guided, pack, margin_scene)[1:], loss=best_loss)

    delta = q0_guided - q0
    delta_norm = delta.norm(dim=-1)                                    # [B,C]
    stats = {
        "safe_loss_before": loss_before,
        "safe_loss_after": best_loss,
        "safe_loss_ratio": torch.where(loss_before > 0,
                                       best_loss / loss_before.clamp_min(1e-12),
                                       torch.ones_like(loss_before)),
        "max_violation_before": stats_before["max_violation"],
        "max_violation_after": stats_after["max_violation"],
        "violated_count_before": stats_before["violated_count"],
        "violated_count_after": stats_after["violated_count"],
        "grad_norm": stats_before["grad_norm"],
        "delta_rms_scene": delta_norm.square().mean(dim=-1).sqrt(),
        "delta_max_scene": delta_norm.amax(dim=-1),
        "trust_region_scale": tr_scale,
        "accepted": accepted,
        "rejected": usable & ~accepted,
        "alpha": alpha,
        "backtrack_trials": torch.full_like(zero, float(trials)),
        "guidance_applied": accepted,
    }
    return q0_guided, stats
