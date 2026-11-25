"""Condition encoders for the Conditional DiT.

Implements exactly the conditioning of the design document (section 4.3/4.4):

* ``MapFeatureProjection``: 1x1 convolution over the concatenated obstacle /
  SDF bottleneck features, plus a 2-D sinusoidal positional encoding, flattened
  to ``1024`` tokens of dimension ``D``.  This is a channel projection, NOT a
  second map encoder: the spatial resolution (32x32) is preserved.
* ``StartGoalEncoder``: start / goal / heading tokens with a validity mask (a
  missing heading contributes a learned "absent" vector, never a fake angle).
* ``SafetyConditionEncoder``: 128 path points (linear projection + 1-D
  positional encoding) and 64 ellipse geometries (MLP), each with its own
  modality embedding.

All three produce the Cross-Attention key/value memory; the trajectory tokens
are the queries.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def sincos_1d(length: int, dim: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Standard 1-D sinusoidal encoding ``[length, dim]`` (index based)."""
    position = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    half = dim // 2
    omega = torch.exp(-math.log(10000.0) * torch.arange(half, device=device,
                                                       dtype=torch.float32) / max(half - 1, 1))
    out = torch.cat([torch.sin(position * omega), torch.cos(position * omega)], dim=-1)
    if out.shape[-1] < dim:
        out = torch.cat([out, torch.zeros(length, dim - out.shape[-1], device=device)], dim=-1)
    return out[:, :dim].to(dtype)


def sincos_2d(height: int, width: int, dim: int, device=None,
              dtype=torch.float32) -> torch.Tensor:
    """2-D sinusoidal encoding ``[height*width, dim]`` (row/col split in halves)."""
    half = dim // 2
    emb_y = sincos_1d(height, half, device=device, dtype=dtype)
    emb_x = sincos_1d(width, half, device=device, dtype=dtype)
    emb_half = dim // 2
    grid_y = emb_y[:, None, :].expand(height, width, emb_half)
    grid_x = emb_x[None, :, :].expand(height, width, dim - emb_half)
    return torch.cat([grid_y, grid_x], dim=-1).reshape(height * width, dim)


def timestep_embedding(timesteps: torch.Tensor, dim: int,
                       max_period: float = 10000.0) -> torch.Tensor:
    """DiT-style sinusoidal timestep embedding ``[B, t] -> [B, dim]``."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) *
                      torch.arange(half, device=timesteps.device, dtype=torch.float32) / half)
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimestepEmbedder(nn.Module):
    """Sinusoidal embedding + 2-layer MLP -> ``[B, dim]`` (DiT ``t_embedder``)."""

    def __init__(self, dim: int, hidden: Optional[int] = None):
        super().__init__()
        hidden = int(hidden or dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(timestep_embedding(t, self.mlp[0].in_features))


class MapFeatureProjection(nn.Module):
    """``[occ_bottleneck ; sdf_bottleneck]`` -> ``[B, H*W, D]`` map memory."""

    def __init__(self, hidden_dim: int = 384, occ_channels: int = 192,
                 sdf_channels: int = 192, grid: int = 32):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.grid = int(grid)
        self.proj = nn.Conv2d(int(occ_channels) + int(sdf_channels),
                              int(hidden_dim), kernel_size=1)
        self.register_buffer("pos", sincos_2d(self.grid, self.grid, self.hidden_dim),
                             persistent=False)

    def forward(self, occ_bottleneck: torch.Tensor, sdf_bottleneck: torch.Tensor) -> torch.Tensor:
        if occ_bottleneck.dim() != 4:
            raise ValueError("expected [B,C,H,W] bottleneck features")
        if occ_bottleneck.shape[-2:] != (self.grid, self.grid):
            raise ValueError("bottleneck is %s, expected %d x %d"
                             % (tuple(occ_bottleneck.shape[-2:]), self.grid, self.grid))
        feat = torch.cat([occ_bottleneck, sdf_bottleneck], dim=1)
        feat = self.proj(feat)                                   # [B,D,32,32]
        tokens = feat.flatten(2).transpose(1, 2)                 # [B,1024,D]
        return tokens + self.pos.to(dtype=tokens.dtype, device=tokens.device)[None]


class StartGoalEncoder(nn.Module):
    """Start / goal / heading -> two condition tokens (with a heading mask)."""

    def __init__(self, hidden_dim: int = 384):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.pos_proj = nn.Linear(2, hidden_dim)
        self.heading_proj = nn.Linear(2, hidden_dim)
        self.null_heading = nn.Parameter(torch.zeros(hidden_dim))
        self.type_embed = nn.Parameter(torch.zeros(2, hidden_dim))
        nn.init.normal_(self.type_embed, std=0.02)
        nn.init.normal_(self.pos_proj.weight, std=0.02)
        nn.init.zeros_(self.pos_proj.bias)

    def forward(self, start_goal: torch.Tensor,
                heading: Optional[torch.Tensor] = None,
                heading_valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``start_goal [B,2,2]`` -> ``[B,2,D]``.

        ``heading [B,2,2]`` holds ``(cos, sin)`` per endpoint; rows whose
        ``heading_valid`` is False swap in a learned "absent" vector instead of
        a fabricated direction.
        """
        if start_goal.dim() != 3 or start_goal.shape[1] != 2:
            raise ValueError("start_goal must be [B,2,2]")
        tokens = self.pos_proj(start_goal)
        if heading is not None:
            head = self.heading_proj(heading)
            if heading_valid is not None:
                mask = heading_valid.to(head.dtype)[..., None]
                head = head * mask + self.null_heading.to(head.dtype) * (1.0 - mask)
            tokens = tokens + head
        else:
            tokens = tokens + self.null_heading.to(tokens.dtype)
        return tokens + self.type_embed.to(tokens.dtype)[None]


class SafetyConditionEncoder(nn.Module):
    """128 path points + 64 ellipses -> ``[B,192,D]`` safety tokens."""

    TYPE_PATH = 0
    TYPE_ELLIPSE = 1

    def __init__(self, hidden_dim: int = 384, path_points: int = 128,
                 num_ellipses: int = 64, ellipse_feature_dim: int = 6):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.path_points = int(path_points)
        self.num_ellipses = int(num_ellipses)
        self.path_proj = nn.Linear(2, hidden_dim)
        self.ellipse_proj = nn.Sequential(
            nn.Linear(int(ellipse_feature_dim), hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.type_embed = nn.Parameter(torch.zeros(2, hidden_dim))
        nn.init.normal_(self.type_embed, std=0.02)
        self.register_buffer("path_pos", sincos_1d(self.path_points, hidden_dim),
                             persistent=False)
        self.register_buffer("ellipse_pos", sincos_1d(self.num_ellipses, hidden_dim),
                             persistent=False)

    def forward(self, path: torch.Tensor, ellipse: torch.Tensor,
                valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        if path.shape[1] != self.path_points:
            raise ValueError("expected %d path points, got %d"
                             % (self.path_points, path.shape[1]))
        if ellipse.shape[1] != self.num_ellipses:
            raise ValueError("expected %d ellipses, got %d"
                             % (self.num_ellipses, ellipse.shape[1]))
        p = self.path_proj(path) + self.path_pos.to(path.dtype) + self.type_embed[self.TYPE_PATH]
        e = self.ellipse_proj(ellipse) + self.ellipse_pos.to(ellipse.dtype) \
            + self.type_embed[self.TYPE_ELLIPSE]
        tokens = torch.cat([p, e], dim=1)
        # Invalid rows are excluded by the cross-attention key mask the backbone
        # builds from ``valid``; they are not zeroed here so that a partially
        # valid condition never silently loses its learned projection.
        return tokens


class TrajectoryTokenEmbedder(nn.Module):
    """B-spline control points -> trajectory tokens ``[B,C,D]``."""

    def __init__(self, hidden_dim: int = 384, controls: int = 32):
        super().__init__()
        self.proj = nn.Linear(2, hidden_dim)
        self.register_buffer("pos", sincos_1d(int(controls), hidden_dim), persistent=False)

    def forward(self, controls: torch.Tensor) -> torch.Tensor:
        tokens = self.proj(controls)
        pos = self.pos.to(tokens.dtype)[None, :tokens.shape[1]]
        return tokens + pos
