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

from typing import Sequence, Tuple

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

    def forward(self, q_raw: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """``q_raw [B,C,2]``, ``cond [B,2,2]`` -> ``q`` with exact endpoints."""
        if q_raw.dim() != 3 or q_raw.shape[-1] != 2:
            raise ValueError("q_raw must be [B,C,2], got %s" % (tuple(q_raw.shape),))
        C = int(q_raw.shape[-2])
        ws, wg = self.weights(C, device=q_raw.device, dtype=q_raw.dtype)
        e_s = cond[:, 0] - q_raw[:, 0]
        e_g = cond[:, 1] - q_raw[:, C - 1]
        return (q_raw + ws[None, :, None] * e_s[:, None, :]
                + wg[None, :, None] * e_g[:, None, :])
