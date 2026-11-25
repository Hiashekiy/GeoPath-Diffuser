"""Condition encoders for the Conditional DiT.

Three independent encoders (spec: 网络结构修改实施方案 §2 / §3):

``MapConditionEncoder``
    Two frozen bottleneck features (occupancy, SDF) each go through their OWN
    trainable adapter (1x1 conv -> GroupNorm(16) -> SiLU), are concatenated, and
    are fused by a residual spatial block (depthwise 3x3 -> GroupNorm -> SiLU ->
    pointwise 1x1 -> GroupNorm, added back to the concatenation).  The fused
    ``[B,384,32,32]`` map is flattened and gets a 2-D sinusoidal positional
    encoding, giving ``[B,1024,384]`` map memory.  The spatial resolution never
    changes and no second encoder is introduced.

``StartGoalEncoder``
    Start / goal / heading tokens with a validity mask (a missing heading
    contributes a learned "absent" vector, never a fake angle).

``SafetyConditionEncoder``
    ONLY the 64 ellipse geometries: ``Linear(6->384) -> SiLU -> Linear(384->384)``,
    plus a 1-D sinusoidal position encoding and the ellipse type embedding, giving
    ``[B,64,384]``.  The search path is *not* a diffusion token any more; it stays
    in the safety-geometry pipeline only.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# positional / timestep encodings
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# map condition: dual adapter + residual spatial fusion
# --------------------------------------------------------------------------- #
class MapAdapter(nn.Module):
    """One feature adapter: ``Conv2d(C, C, 1, bias=False) -> GroupNorm -> SiLU``."""

    def __init__(self, channels: int, groups: int = 16):
        super().__init__()
        if channels % groups:
            raise ValueError("channels %d must be divisible by groups %d"
                             % (channels, groups))
        self.conv = nn.Conv2d(int(channels), int(channels), kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(int(groups), int(channels))
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualSpatialFusion(nn.Module):
    """``x + GN(PW(SiLU(GN(DW3x3(x)))))`` at unchanged spatial resolution."""

    def __init__(self, channels: int, groups: int = 16, kernel_size: int = 3):
        super().__init__()
        if channels % groups:
            raise ValueError("channels %d must be divisible by groups %d"
                             % (channels, groups))
        kernel = int(kernel_size)
        padding = kernel // 2
        self.depthwise = nn.Conv2d(int(channels), int(channels), kernel_size=kernel,
                                  padding=padding, groups=int(channels), bias=False)
        self.norm1 = nn.GroupNorm(int(groups), int(channels))
        self.act = nn.SiLU()
        self.pointwise = nn.Conv2d(int(channels), int(channels), kernel_size=1, bias=False)
        self.norm2 = nn.GroupNorm(int(groups), int(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.depthwise(x)))
        h = self.norm2(self.pointwise(h))
        return x + h


class MapConditionEncoder(nn.Module):
    """Two frozen bottleneck features -> ``[B,1024,D]`` map memory (trainable).

    Replaces the previous single 1x1 ``MapFeatureProjection``: each modality gets
    its own adapter (they must not share parameters), the adapters' outputs are
    concatenated and then fused spatially before the flatten + 2-D position
    encoding.  The frozen ResUNets are NOT part of this module.
    """

    def __init__(self, hidden_dim: int = 384, bottleneck_channels: int = 192,
                 adapter_channels: Optional[int] = None, grid: int = 32,
                 groups: int = 16, kernel_size: int = 3):
        super().__init__()
        adapter_channels = int(adapter_channels or bottleneck_channels)
        if adapter_channels != int(bottleneck_channels):
            raise ValueError("adapter_channels must equal bottleneck_channels "
                             "(the adapters keep the bottleneck width)")
        if 2 * adapter_channels != int(hidden_dim):
            raise ValueError("2 x adapter_channels (%d) must equal hidden_dim (%d)"
                             % (2 * adapter_channels, int(hidden_dim)))
        self.hidden_dim = int(hidden_dim)
        self.grid = int(grid)
        self.occ_adapter = MapAdapter(adapter_channels, groups)
        self.sdf_adapter = MapAdapter(adapter_channels, groups)
        self.fusion = ResidualSpatialFusion(2 * adapter_channels, groups, kernel_size)
        self.register_buffer("pos", sincos_2d(self.grid, self.grid, self.hidden_dim),
                             persistent=False)

    def forward(self, occ_bottleneck: torch.Tensor,
                sdf_bottleneck: torch.Tensor) -> torch.Tensor:
        if occ_bottleneck.dim() != 4 or sdf_bottleneck.dim() != 4:
            raise ValueError("expected [B,C,H,W] bottleneck features")
        if occ_bottleneck.shape[-2:] != (self.grid, self.grid):
            raise ValueError("bottleneck is %s, expected %d x %d"
                             % (tuple(occ_bottleneck.shape[-2:]), self.grid, self.grid))
        # adapters are out-of-place: the frozen features are never modified
        occ_feat = self.occ_adapter(occ_bottleneck)
        sdf_feat = self.sdf_adapter(sdf_bottleneck)
        fused = self.fusion(torch.cat([occ_feat, sdf_feat], dim=1))   # [B,384,32,32]
        tokens = fused.flatten(2).transpose(1, 2)                    # [B,1024,384]
        return tokens + self.pos.to(dtype=tokens.dtype, device=tokens.device)[None]

    # ------------------------------------------------------------------ report
    def parameter_breakdown(self) -> dict:
        def count(module):
            return sum(p.numel() for p in module.parameters())
        return {
            "occ_adapter": count(self.occ_adapter),
            "sdf_adapter": count(self.sdf_adapter),
            "spatial_fusion": count(self.fusion),
            "total": count(self),
        }


# --------------------------------------------------------------------------- #
# start / goal
# --------------------------------------------------------------------------- #
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
        """``start_goal [B,2,2]`` -> ``[B,2,D]``."""
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


# --------------------------------------------------------------------------- #
# safety condition: ellipses only
# --------------------------------------------------------------------------- #
class SafetyConditionEncoder(nn.Module):
    """64 ellipses ``[B,64,6]`` -> safety memory ``[B,64,D]``.

    The search path is deliberately NOT encoded here (spec §3.1): it stays in the
    safety-geometry pipeline (ellipse centres, collision projection, corridor
    construction) but is no longer a diffusion condition token.
    """

    TYPE_ELLIPSE = 0

    def __init__(self, hidden_dim: int = 384, num_ellipses: int = 64,
                 ellipse_feature_dim: int = 6, add_positional_encoding: bool = True,
                 add_type_embedding: bool = True):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_ellipses = int(num_ellipses)
        self.ellipse_proj = nn.Sequential(
            nn.Linear(int(ellipse_feature_dim), hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.add_positional_encoding = bool(add_positional_encoding)
        self.add_type_embedding = bool(add_type_embedding)
        if self.add_type_embedding:
            self.type_embed = nn.Parameter(torch.zeros(1, hidden_dim))
            nn.init.normal_(self.type_embed, std=0.02)
        if self.add_positional_encoding:
            self.register_buffer("ellipse_pos", sincos_1d(self.num_ellipses, hidden_dim),
                                 persistent=False)

    def forward(self, ellipse: torch.Tensor) -> torch.Tensor:
        if ellipse.dim() != 3:
            raise ValueError("ellipse must be [B,64,6]")
        if ellipse.shape[1] != self.num_ellipses:
            raise ValueError("expected %d ellipses, got %d"
                             % (self.num_ellipses, ellipse.shape[1]))
        tokens = self.ellipse_proj(ellipse)
        if self.add_positional_encoding:
            tokens = tokens + self.ellipse_pos.to(dtype=tokens.dtype)
        if self.add_type_embedding:
            tokens = tokens + self.type_embed.to(dtype=tokens.dtype)
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
