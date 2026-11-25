"""Losses for the dual-branch joint objective (design document section 9.3).

    L_total = lambda_c * L_c + lambda_u * L_u + lambda_a * L_align + L_safe^c

``L_safe^c = lambda_col * L_collision + lambda_corr * L_corridor``

``L_corridor`` is the *training-time twin* of the ALM: it evaluates the same
continuous constraint ``A_j E_{l,r} Q <= b_j`` on the exact Bezier controls of
every piece, which is why it needs no dense sampling of the curve.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def control_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample squared error in control space ``[B]``."""
    return ((pred - target) ** 2).mean(dim=(1, 2))


def alignment_loss(pred_map_only: torch.Tensor, pred_safety: torch.Tensor) -> torch.Tensor:
    """``||Q_u - sg(Q_c)||^2`` - the Map-only branch chases the safety branch."""
    return ((pred_map_only - pred_safety.detach()) ** 2).mean(dim=(1, 2))


def collision_loss(points_scene: torch.Tensor, sdf_scene: torch.Tensor,
                   margin_m: float = 0.3, scene_scale: float = 32.0) -> torch.Tensor:
    """Clearance penalty from the obstacle distance map (metres -> normalised)."""
    B, N, _ = points_scene.shape
    # scene coordinates are already the pixel-centre normalised frame, so they can
    # be handed to grid_sample directly (see gp_diffuser.frames.SceneTransform)
    grid = points_scene.reshape(B, 1, N, 2)
    sdf = sdf_scene.reshape(B, 1, *sdf_scene.shape[-2:]).to(points_scene.dtype)
    sampled = F.grid_sample(sdf, grid, mode="bilinear", padding_mode="border",
                            align_corners=False).reshape(B, N)
    # the sampled value can be negative inside obstacles; distances are metres
    margin_scene = float(margin_m) / float(scene_scale)
    violation = torch.relu(margin_scene - sampled)
    return (violation ** 2).mean(dim=1)


def corridor_loss(controls: torch.Tensor, pack) -> torch.Tensor:
    """Per-sample mean positive violation of the frozen half-space constraints."""
    if pack is None:
        return controls.new_zeros(controls.shape[0])
    beta = torch.einsum("bprk,bkd->bprd", pack.extraction, controls)
    g = (torch.einsum("bpfd,bprd->bprf", pack.piece_A, beta)
         - pack.piece_b[:, :, None, :])
    mask = pack.piece_mask[:, :, None, None] & pack.face_mask[:, :, None, :]
    positive = torch.relu(g) * mask
    count = mask.expand_as(g).sum(dim=(1, 2, 3)).clamp_min(1)
    return positive.sum(dim=(1, 2, 3)) / count


def total_loss(loss_u: torch.Tensor, loss_c: torch.Tensor,
               loss_align: torch.Tensor, loss_collision: torch.Tensor,
               loss_corridor: torch.Tensor, weights: Dict[str, float]) -> torch.Tensor:
    """Weighted sum; each term is already a per-sample mean over the batch."""
    return (float(weights.get("lambda_c", 1.0)) * loss_c
            + float(weights.get("lambda_u", 1.0)) * loss_u
            + float(weights.get("lambda_align", 0.5)) * loss_align
            + float(weights.get("lambda_collision", 1.0)) * loss_collision
            + float(weights.get("lambda_corridor", 1.0)) * loss_corridor)
