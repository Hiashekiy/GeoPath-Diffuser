"""Losses for the dual-branch joint objective, aligned with TrajSafe-Diffuser.

    L = lambda_c   * L_ctrl      (INTERIOR B-spline controls vs GT)
      + lambda_u   * L_ctrl      (same, Map-only branch)
      + lambda_align * L_align   (dual-branch: Map-only chases the safety branch)
      + lambda_bound * L_boundary (RAW control polygon near both endpoints)
      + lambda_smooth * L_smooth  (2nd/3rd differences OF THE CONTROLS)
      + lambda_curve  * L_curve   (2nd/3rd differences of the DECODED curve)
      + lambda_safe   * L_safe    (THE continuous safety loss)

Two things are deliberately the same as the validated TrajSafe design:

1. ``L_ctrl`` uses the **INTERIOR** controls ``Q_1..Q_{C-2}`` only.  The endpoint
   controls are produced by the free network output and are made exact by the
   parameter-free :class:`~gp_diffuser.models.boundary.BoundaryDecoder`; putting
   them into the MSE as well would double-count them and pull ``Q_1..Q_3``
   towards the start (this is what collapsed the first training run).
2. ``L_safe`` is the **only** safety term and it is the exact object the
   inference ALM projects onto: the continuous Bezier convex-hull violation of the
   control polygon against the frozen corridor pack (``A beta <= b``), in scene
   units, LINEAR, reduced with ``max``.  There is no SDF/margin penalty and no
   decoded-curve corridor penalty any more: the old ``relu(0.30 m - sdf)`` term
   dominated the gradient by ~48x and fought the GT (66% of GT paths sit closer
   than 0.30 m to the inflated obstacles).

``L_smooth`` / ``L_curve`` give the network the smoothness the first run had no
penalty for, and ``L_boundary`` supervises the *shape* of the local control
polygon around both endpoints (GT polygon translated to this sample's S/G), so
the network has to produce a sane polygon instead of relying on the decoder.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

__all__ = [
    "control_x0_loss", "control_loss", "alignment_loss", "control_smoothness_loss",
    "curve_smoothness_loss", "boundary_control_loss", "pack_violation",
    "pack_max_violation", "continuous_safety_loss", "total_loss",
    "DEFAULT_LOSS_WEIGHTS",
]

DEFAULT_LOSS_WEIGHTS: Dict[str, float] = {
    "lambda_c": 1.0,
    "lambda_u": 1.0,
    "lambda_align": 0.5,
    "lambda_boundary": 0.2,
    "lambda_smooth": 0.15,
    "lambda_curve": 0.15,
    "lambda_safe": 0.15,
    # privileged supervision: the same corridor set on the MAP-ONLY branch
    # (0.0 = off, the historical objective)
    "lambda_safe_u": 0.0,
}


def _masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return values.mean()
    mask = mask.to(values.dtype)
    while mask.dim() < values.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(values)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


# --------------------------------------------------------------------------- #
# regression on the control polygon
# --------------------------------------------------------------------------- #
def control_x0_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample MSE on the INTERIOR controls ``Q_1..Q_{C-2}`` -> ``[B]``.

    The two endpoint controls are excluded: they are made exact by the fixed
    BoundaryDecoder, so supervising them here would double-count the endpoints
    and drag the neighbouring controls towards the start.
    """
    if pred.shape != target.shape:
        raise ValueError("control shapes differ: %s vs %s"
                         % (tuple(pred.shape), tuple(target.shape)))
    interior_pred = pred[:, 1:-1]
    interior_gt = target[:, 1:-1]
    if interior_pred.numel() == 0:                      # degenerate C == 2
        return pred.reshape(pred.shape[0], -1).sum(dim=1) * 0.0
    return ((interior_pred - interior_gt) ** 2).mean(dim=(1, 2))


control_loss = control_x0_loss                              # backwards-compatible alias


def alignment_loss(pred_map_only: torch.Tensor,
                   pred_safety: torch.Tensor) -> torch.Tensor:
    """``||Q_u - sg(Q_c)||^2`` on the interior controls (dual-branch design)."""
    return control_x0_loss(pred_map_only, pred_safety.detach())


# --------------------------------------------------------------------------- #
# smoothness (the first run had none -> triangular / hooked curves)
# --------------------------------------------------------------------------- #
def _difference_penalty(points: torch.Tensor, gt_points: torch.Tensor,
                        acc_weight: float, jerk_weight: float,
                        eps: float = 1e-3) -> torch.Tensor:
    """2nd/3rd differences scaled by the detached GT step length (batch scalar)."""
    velocity = points[:, 1:] - points[:, :-1]
    acceleration = velocity[:, 1:] - velocity[:, :-1]
    jerk = acceleration[:, 1:] - acceleration[:, :-1]
    gt_velocity = gt_points[:, 1:] - gt_points[:, :-1]
    step = gt_velocity.norm(dim=-1).mean().detach().clamp_min(1e-6)
    acceleration = acceleration / step
    jerk = jerk / step
    acc_norm = (acceleration.square().sum(dim=-1) + eps ** 2).sqrt().sub(eps)
    jerk_norm = (jerk.square().sum(dim=-1) + eps ** 2).sqrt().sub(eps)
    return float(acc_weight) * torch.log1p(acc_norm).mean() \
        + float(jerk_weight) * torch.log1p(jerk_norm).mean()


def control_smoothness_loss(pred: torch.Tensor, gt: torch.Tensor,
                            acc_weight: float = 0.25,
                            jerk_weight: float = 1.0) -> torch.Tensor:
    """``L_smooth``: 2nd and 3rd differences of the CONTROL polygon."""
    if pred.shape[1] < 4:
        return pred.sum() * 0.0
    return _difference_penalty(pred, gt, acc_weight, jerk_weight)


def curve_smoothness_loss(pred: torch.Tensor, gt: torch.Tensor, basis: torch.Tensor,
                          acc_weight: float = 0.25,
                          jerk_weight: float = 1.0) -> torch.Tensor:
    """``L_curve``: the same penalty on the DECODED curve ``P = B Q``."""
    p = torch.einsum("hk,bkd->bhd", basis.to(pred.dtype), pred)
    p_gt = torch.einsum("hk,bkd->bhd", basis.to(gt.dtype), gt)
    return _difference_penalty(p, p_gt, acc_weight, jerk_weight)


# --------------------------------------------------------------------------- #
# boundary (local polygon shape around both endpoints)
# --------------------------------------------------------------------------- #
def boundary_control_loss(q_raw: torch.Tensor, q_gt: torch.Tensor,
                          cond: torch.Tensor, ws: torch.Tensor,
                          wg: torch.Tensor) -> torch.Tensor:
    """Weighted local supervision of the RAW controls at both ends.

    Targets are the GT control polygon rigidly translated to THIS sample's
    start / goal, so the loss enforces the GT *shape* of the local polygon
    instead of a collapse onto S / G.
    """
    from ..models.boundary import boundary_targets

    if q_raw.shape != q_gt.shape:
        raise ValueError("control shapes differ: %s vs %s"
                         % (tuple(q_raw.shape), tuple(q_gt.shape)))
    target_s, target_g = boundary_targets(q_gt, cond)
    ws = ws.to(q_raw.dtype)[None, :, None]
    wg = wg.to(q_raw.dtype)[None, :, None]
    err_s = ((q_raw - target_s) ** 2).sum(dim=-1) * ws[:, :, 0]
    err_g = ((q_raw - target_g) ** 2).sum(dim=-1) * wg[:, :, 0]
    denom = (ws.sum() + wg.sum()).clamp_min(1e-9)
    return (err_s.sum(dim=1) + err_g.sum(dim=1)) / denom


# --------------------------------------------------------------------------- #
# the continuous constraint set: the single training-time safety term
# --------------------------------------------------------------------------- #
def pack_violation(controls: torch.Tensor, pack, margin: float = 0.0,
                   reduction: str = "max") -> torch.Tensor:
    """Per-sample violation of a control polygon against the corridor pack.

    ``beta = E Q`` are the exact cubic Bezier controls of every piece and the
    piece is inside its cell iff ``A beta <= b`` for all four Bezier controls and
    all faces.  Because the four controls lie in the (convex) cell, the whole
    piece does - a continuous certificate, not a 128-point sampling heuristic.

    ``"max"`` returns the deepest violation in SCENE units (what the ALM drives to
    zero); ``"mean"`` returns the mean positive violation over active faces.
    A pack with no active cell contributes exactly zero and stays graph-connected.
    """
    if controls.dim() != 3 or controls.shape[-1] != 2:
        raise ValueError("controls must be [B,C,2], got %s" % (tuple(controls.shape),))
    B0 = int(controls.shape[0])
    if pack is None:                     # no local corridor for this batch
        return controls.reshape(B0, -1).sum(dim=1) * 0.0
    if int(pack.extraction.shape[-1]) != int(controls.shape[1]):
        raise ValueError("pack expects %d controls, got %d"
                         % (int(pack.extraction.shape[-1]), int(controls.shape[1])))
    B = int(controls.shape[0])
    zero = controls.reshape(B, -1).sum(dim=1) * 0.0
    if int(pack.extraction.shape[1]) == 0 or int(pack.piece_A.shape[2]) == 0:
        return zero
    beta = torch.einsum("bprk,bkd->bprd", pack.extraction, controls)
    g = (torch.einsum("bpfd,bprd->bprf", pack.piece_A, beta)
         - pack.piece_b[:, :, None, :])
    mask = pack.piece_mask[:, :, None, None] & pack.face_mask[:, :, None, :]
    positive = torch.relu(g - float(margin)) * mask
    flat = positive.flatten(1)
    if flat.shape[1] == 0:
        return zero
    if reduction == "mean":
        denom = mask.expand_as(positive).flatten(1).sum(dim=1).clamp_min(1.0)
        return flat.sum(dim=1) / denom
    if reduction != "max":
        raise ValueError("reduction must be 'max' or 'mean', got %r" % (reduction,))
    return flat.amax(dim=1)


def pack_max_violation(controls: torch.Tensor, pack, margin: float = 0.0) -> torch.Tensor:
    return pack_violation(controls, pack, margin=margin, reduction="max")


def continuous_safety_loss(controls: torch.Tensor, pack, margin: float = 0.0,
                           sample_mask: Optional[torch.Tensor] = None,
                           reduction: str = "mean") -> torch.Tensor:
    """``L_safe``: continuous Bezier convex-hull violation (THE safety loss).

    A polygon already inside the corridor (a fixed point of the ALM) contributes
    exactly 0 and receives no gradient.
    """
    per = pack_violation(controls, pack, margin=margin, reduction="max")
    if reduction == "none":
        return per
    return _masked_mean(per, sample_mask)


def total_loss(loss_u: torch.Tensor, loss_c: torch.Tensor, loss_align: torch.Tensor,
               loss_boundary: torch.Tensor, loss_smooth: torch.Tensor,
               loss_curve: torch.Tensor, loss_safe: torch.Tensor,
               weights: Dict[str, float],
               loss_safe_u: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Weighted sum; every term is already a per-sample vector over the batch."""
    w = dict(DEFAULT_LOSS_WEIGHTS)
    w.update({k: float(v) for k, v in (weights or {}).items() if k in w})
    value = (w["lambda_c"] * loss_c
             + w["lambda_u"] * loss_u
             + w["lambda_align"] * loss_align
             + w["lambda_boundary"] * loss_boundary
             + w["lambda_smooth"] * loss_smooth
             + w["lambda_curve"] * loss_curve
             + w["lambda_safe"] * loss_safe)
    if loss_safe_u is not None:
        value = value + w["lambda_safe_u"] * loss_safe_u
    return value
