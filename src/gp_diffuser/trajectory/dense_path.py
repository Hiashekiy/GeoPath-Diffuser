"""Dense Skeleton gather helper used by the safety-corridor gap bridge.

Copied verbatim from TrajSafe-Diffuser ``src/models/trajsafe/geometry.py``
(branch ``v4-map16-fbdelta-safeloss``, commit ``14a5a97``).  Only
``dense_arclength`` + ``gather_dense_path_points`` are kept: the V5 model in
that file is NOT part of this project.  In the new pipeline the "dense
skeleton" is the searched path (design document section 6.4 / 7.4).
"""
from __future__ import annotations

import torch


def dense_arclength(coords: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """coords [B,G,2] padded, lengths [B] valid point count -> cum [B,G]."""
    B, G, _ = coords.shape
    if G < 2:
        return torch.zeros(B, 1, device=coords.device, dtype=coords.dtype)
    seg = torch.linalg.norm(coords[:, 1:] - coords[:, :-1], dim=-1)
    idx = torch.arange(G - 1, device=coords.device)[None]
    seg = seg * (idx < (lengths[:, None] - 1)).to(seg.dtype)
    zero = torch.zeros_like(seg[:, :1])
    return torch.cat([zero, torch.cumsum(seg, dim=1)], dim=1)


def gather_dense_path_points(coords: torch.Tensor, lengths: torch.Tensor,
                             s: torch.Tensor) -> torch.Tensor:
    """Gamma(s) on the dense chain: coords [B,G,2], lengths [B], s [B,K]."""
    B, G, _ = coords.shape
    lengths = lengths.clamp(min=1, max=G)
    cum = dense_arclength(coords, lengths)
    last = (lengths - 1).clamp(min=0).long()
    total = cum.gather(1, last[:, None]).clamp_min(1e-9)
    target = s.clamp(0.0, 1.0) * total
    idx = torch.searchsorted(cum.contiguous(), target.contiguous(), right=True)
    hi = last[:, None].expand_as(idx)
    idx = torch.minimum(torch.clamp(idx, min=1), hi.clamp_min(1))
    c0 = cum.gather(1, idx - 1)
    c1 = cum.gather(1, idx)
    t = ((target - c0) / (c1 - c0).clamp_min(1e-9)).clamp(0.0, 1.0).unsqueeze(-1)
    ii = idx.unsqueeze(-1).expand(-1, -1, 2)
    p0 = coords.gather(1, ii - 1)
    p1 = coords.gather(1, ii)
    return p0 + t * (p1 - p0)
