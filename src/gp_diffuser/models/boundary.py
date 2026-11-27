"""Parameter-free endpoint modulation for the control polygon (TrajSafe design).

The network outputs a FREE control polygon ``Q~ [B,C,2]``.  This decoder applies a
deterministic, constant correction profile so that the first and the last control
are exactly the start / goal **without teleporting them**::

    e_s = S - Q~_0                 e_g = G - Q~_{C-1}
    w^s = [1, 0.75, 0.5, 0.25, 0, ...]        (first ``span`` controls)
    w^g = [..., 0, 0.25, 0.5, 0.75, 1]        (mirrored at the end)

    Q*_i = Q~_i + w^s_i e_s + w^g_i e_g

so ``Q*_0 = S`` and ``Q*_{C-1} = G`` hold by construction for ANY prediction, while
the neighbouring controls move coherently with the endpoints instead of being
teleported.  Because the knot vector is clamped this is exactly the curve
endpoint condition ``C(0) = S``, ``C(1) = G``.

The profile is a constant buffer: no parameters, never trained, never touched by
the optimiser.  ``weights()`` is the single source of the profile for both this
correction and ``boundary_control_loss``, so the supervised local shape and the
applied correction can never drift apart.

``span`` is clamped to ``C // 2`` so a small control count degenerates gracefully
to plain endpoint hard-conditioning (span 1 == the old behaviour).
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

__all__ = ["BoundaryDecoder", "boundary_targets", "DEFAULT_BOUNDARY_PROFILE"]

DEFAULT_BOUNDARY_PROFILE: Tuple[float, ...] = (1.0, 0.75, 0.5, 0.25)


def boundary_targets(q_gt: torch.Tensor, cond: torch.Tensor):
    """GT local control-polygon shape rigidly translated to THIS sample's ends.

    ``T^s_i = S + (Q^GT_i - Q^GT_0)`` and ``T^g_j = G + (Q^GT_j - Q^GT_{C-1})``.
    One single definition, shared by the boundary loss and the metrics.
    """
    C = q_gt.shape[-2]
    off_s = (q_gt - q_gt[:, 0:1]) + cond[:, 0][:, None, :]
    off_g = (q_gt - q_gt[:, C - 1:C]) + cond[:, 1][:, None, :]
    return off_s, off_g


class BoundaryDecoder(nn.Module):
    """Parameter-free endpoint correction on the control polygon."""

    def __init__(self, profile: Sequence[float] | None = None, span: int | None = None,
                 dtype=torch.float32):
        super().__init__()
        if profile is None:
            n = int(span) if span else len(DEFAULT_BOUNDARY_PROFILE)
            if n < 1:
                raise ValueError("boundary span must be >= 1")
            base = list(DEFAULT_BOUNDARY_PROFILE)
            profile = base[:n] + [0.0] * max(0, n - len(base))
        prof = torch.as_tensor(list(profile), dtype=dtype).reshape(-1)
        if prof.numel() < 1:
            raise ValueError("boundary profile must not be empty")
        if float(prof[0]) != 1.0:
            raise ValueError("boundary profile[0] must be 1.0 so the first control is "
                             "exactly the start (got %s)" % float(prof[0]))
        self.register_buffer("profile", prof, persistent=False)
        self.span = int(prof.numel())

    def weights(self, num_controls: int, device=None, dtype=None):
        """``(w^s, w^g)`` constant vectors of length ``num_controls``."""
        C = int(num_controls)
        if C < 2:
            raise ValueError("num_controls must be >= 2, got %d" % C)
        dtype = dtype or self.profile.dtype
        device = device or self.profile.device
        k = max(1, min(self.span, C // 2))
        w = self.profile[:k].to(device=device, dtype=dtype)
        ws = torch.zeros(C, device=device, dtype=dtype)
        wg = torch.zeros(C, device=device, dtype=dtype)
        ws[:k] = w
        wg[C - k:] = torch.flip(w, dims=[0])
        return ws, wg

    def forward(self, q_raw: torch.Tensor, cond: torch.Tensor,
                heading: Optional[torch.Tensor] = None,
                heading_valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``q_raw [B,C,2]``, ``cond [B,2,2]`` -> ``q`` with exact endpoints."""
        if q_raw.dim() != 3 or q_raw.shape[-1] != 2:
            raise ValueError("q_raw must be [B,C,2], got %s" % (tuple(q_raw.shape),))
        C = int(q_raw.shape[-2])
        ws, wg = self.weights(C, device=q_raw.device, dtype=q_raw.dtype)
        e_s = cond[:, 0] - q_raw[:, 0]
        e_g = cond[:, 1] - q_raw[:, C - 1]
        q = (q_raw + ws[None, :, None] * e_s[:, None, :]
             + wg[None, :, None] * e_g[:, None, :])
        if heading is None:
            return q
        return self.align_tangents(q, heading, heading_valid)

    # ------------------------------------------------------------- tangents
    def tangent_weights(self, num_controls: int, device=None, dtype=None):
        """``(t^s, t^g)``: like ``weights`` but ZERO on the endpoint control.

        The endpoint itself must not move (it is already exact), so the tangency
        correction has to vanish at ``Q_0`` / ``Q_{C-1}`` and peak at its
        neighbour - that is what makes ``Q*_1 - Q*_0`` exactly the heading while
        ``Q*_0`` stays on the start point.
        """
        C = int(num_controls)
        dtype = dtype or self.profile.dtype
        device = device or self.profile.device
        k = max(2, min(self.span, C // 2))
        w = self.profile[:k].to(device=device, dtype=dtype)
        ts = torch.zeros(C, device=device, dtype=dtype)
        tg = torch.zeros(C, device=device, dtype=dtype)
        # weight 1.0 on the NEIGHBOUR so the segment direction is exact, then the
        # remaining profile entries fade the correction out for smoothness
        ts[1] = 1.0
        tg[C - 2] = 1.0
        if k > 2:
            ts[2:k] = w[2:k]
            # mirrored window: Q_{C-k} .. Q_{C-3} get the faded weights
            tg[C - k:C - 2] = torch.flip(w[2:k], dims=[0])
        return ts, tg

    def align_tangents(self, q: torch.Tensor, heading: torch.Tensor,
                       heading_valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Make the first/last control segment point along the requested headings.

        ``heading`` ``[B,2,2]`` is ``(start, goal)`` as a direction (not necessarily
        unit).  The correction is a pure translation of the neighbouring controls,
        scaled by the same decaying profile, so:

        * ``Q_0`` / ``Q_{C-1}`` do not move (position stays exact),
        * ``Q_1 - Q_0`` and ``Q_{C-1} - Q_{C-2}`` become exactly the headings,
        * the rest of the polygon only moves coherently (no kink at the ends).
        """
        if heading.dim() != 3 or heading.shape[-1] != 2:
            raise ValueError("heading must be [B,2,2], got %s" % (tuple(heading.shape),))
        C = int(q.shape[-2])
        if C < 3:
            return q
        ts, tg = self.tangent_weights(C, device=q.device, dtype=q.dtype)

        target = heading.to(q.dtype)
        target = target / target.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        if heading_valid is not None:
            valid = heading_valid.to(torch.bool)
            target = torch.where(valid[:, :, None], target,
                                 torch.zeros_like(target))

        d_s = q[:, 1] - q[:, 0]
        len_s = d_s.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        delta_s = len_s * target[:, 0] - d_s

        # `d_g` points BACKWARDS (C-2 -> C-1), while the requested heading is the
        # forward tangent at u=1, so the target direction is -h_g
        d_g = q[:, C - 2] - q[:, C - 1]
        len_g = d_g.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        delta_g = len_g * (-target[:, 1]) - d_g
        if heading_valid is not None:
            # an invalid end must be left EXACTLY as it was: zeroing only the target
            # would collapse the end segment onto the endpoint
            keep = heading_valid.to(torch.bool)
            delta_s = torch.where(keep[:, 0:1], delta_s, torch.zeros_like(delta_s))
            delta_g = torch.where(keep[:, 1:2], delta_g, torch.zeros_like(delta_g))

        q = q + ts[None, :, None] * delta_s[:, None, :] \
              + tg[None, :, None] * delta_g[:, None, :]
        return q
