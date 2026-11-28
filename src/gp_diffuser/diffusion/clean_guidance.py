"""Clean-Prediction Constraint Guidance (CPCG).

The constraint cost is defined on the CLEAN B-spline controls the network just
predicted, never on the noisy state ``q_t``:

    beta[p,r]   = E[p,r] Q                        exact cubic Bezier controls
    g[p,r,f]    = A[p,f] beta[p,r] - b[p,f]       half-space value (g <= 0 is safe)
    J(Q)        = 0.5 sum_{violated} relu(g)^2 / count(violated)

Because every ``E`` / ``A`` / ``b`` is a frozen constant, ``dJ/dQ`` is analytic:

    d g[p,r,f]/dQ[k,d] = A[p,f,d] E[p,r,k]
    dJ/dQ = sum_{violated} relu(g) A[p,f,:] E[p,r,:] / count

so there is no autograd and no backward pass through the DiT.  ``sample()``
keeps its ``@torch.no_grad()``.

The guided clean prediction enters the sampler through the EXISTING ``q0_used``
interface: the DDIM update already rebuilds ``eps`` from ``(q_t, q0_used)``, so
no extra forward, no extra noise and no extra scheduler math are needed.

Deliberately NOT here: ALM, dual variables, collision penalties on ``q_t``, and
anything that treats the individual Bezier controls as independent optimisation
variables.  The only primal variable is the global B-spline control polygon.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

__all__ = [
    "constraint_values",
    "constraint_loss_and_grad",
    "violation_stats",
    "clean_prediction_guidance",
]


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

    ``loss_per_sample [B]``, ``grad_q0 [B,C,2]`` and a stats dict.  ``count`` is
    the number of CURRENTLY VIOLATED valid scalars: averaging over every face of
    the corridor would dilute the few real violations into nothing.

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
    count = violated.sum(dim=(1, 2, 3)).clamp_min(1).to(q0.dtype)
    loss = 0.5 * v.square().sum(dim=(1, 2, 3)) / count

    grad = torch.einsum("bprf,bpfd,bprk->bkd", v, pack.piece_A, pack.extraction)
    grad = grad / count[:, None, None]
    has = violated.any(dim=(1, 2, 3))
    grad = torch.where(has[:, None, None], grad, torch.zeros_like(grad))
    stats = violation_stats(raw, valid, loss, grad_norm=grad.norm(dim=(1, 2)))
    return loss, grad, stats


def clean_prediction_guidance(
    q0: torch.Tensor,
    pack,
    boundary_decoder,
    start_goal: torch.Tensor,
    heading: Optional[torch.Tensor] = None,
    heading_valid: Optional[torch.Tensor] = None,
    step_size: float = 0.1,
    max_delta_scene: float = 0.01,
    margin_scene: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """One trust-region-limited analytic gradient step on the clean polygon.

    Order matters and is the documented one:

        1. analytic gradient of the corridor cost at ``q0``
        2. zero the ENDPOINT rows (they are hard conditions, not free variables)
        3. ``q0 + (-step_size * grad)`` then ``BoundaryDecoder`` again, so the
           endpoint / heading conditions survive the step
        4. trust region on the EFFECTIVE delta (the one the decoder actually
           produced), ``max_delta_scene`` per control point
        5. no violation -> ``q0`` is returned bit-for-bit unchanged

    Returns ``(q0_guided, stats)``.  No autograd, no network call, no ALM.
    """
    empty = _is_empty(pack, q0)
    if not empty:
        pack = pack.to(device=q0.device, dtype=q0.dtype)
    loss_before, grad, stats_before = constraint_loss_and_grad(q0, pack, margin_scene)
    grad = grad.clone()
    if grad.shape[1] >= 2:
        grad[:, 0] = 0.0
        grad[:, -1] = 0.0

    has = stats_before["has_violation"]
    q0_tmp = q0 - float(step_size) * grad
    q0_boundary = boundary_decoder(q0_tmp, start_goal, heading, heading_valid)
    delta_eff = q0_boundary - q0

    delta_max = delta_eff.norm(dim=-1).amax(dim=-1)                 # [B]
    scale = torch.minimum(torch.ones_like(delta_max),
                          float(max_delta_scene) / (delta_max + 1e-12))
    delta = scale[:, None, None] * delta_eff
    delta = torch.where(has[:, None, None], delta, torch.zeros_like(delta))
    q0_guided = q0 + delta

    if empty:
        stats_after = _empty_stats(q0)
    else:
        _, raw_after, valid_after = constraint_values(q0_guided, pack, margin_scene)
        violated = valid_after & (raw_after > 0.0)
        v = torch.relu(raw_after) * valid_after
        count = violated.sum(dim=(1, 2, 3)).clamp_min(1).to(q0.dtype)
        loss_after = 0.5 * v.square().sum(dim=(1, 2, 3)) / count
        stats_after = violation_stats(raw_after, valid_after, loss_after)

    delta_norm = delta.norm(dim=-1)                                  # [B,C]
    stats = {
        "safe_loss_before": stats_before["loss"],
        "safe_loss_after": stats_after["loss"],
        "max_violation_before": stats_before["max_violation"],
        "max_violation_after": stats_after["max_violation"],
        "violated_count_before": stats_before["violated_count"],
        "violated_count_after": stats_after["violated_count"],
        "grad_norm": stats_before["grad_norm"],
        "delta_rms_scene": delta_norm.square().mean(dim=-1).sqrt(),
        "delta_max_scene": delta_max * scale,
        "trust_region_scale": scale,
        "guidance_applied": has & (delta_norm.amax(dim=-1) > 0.0),
    }
    return q0_guided, stats
