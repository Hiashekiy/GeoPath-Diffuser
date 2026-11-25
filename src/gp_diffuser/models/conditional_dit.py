"""Standard Conditional Diffusion Transformer (DiT-S) for B-spline controls.

This is a NEW network.  It is not derived from TrajSafe-Diffuser V5: there is no
``TrajBlock``, no ``Control Backbone``, no ``Final Denoiser`` and no
``Skeleton Encoder`` here.  What it contains is the published DiT recipe:

* trajectory tokens = ``Linear(2 -> D)`` on the control points + 1-D sincos PE,
* timestep conditioning = sinusoidal embedding + MLP, injected through
  AdaLN-Zero (scale / shift / gate on every sub-block, zero-initialised gates),
* ``N`` identical blocks of ``[trajectory self-attention -> condition
  cross-attention -> MLP]``,
* ``LayerNorm -> Linear(D -> 2)`` output head predicting ``x0`` (clean controls).

Configuration (design document section 4.1): ``D=384``, 12 blocks, 6 heads,
``mlp_ratio=4`` (FFN width 1536).

Condition memory is ``[map tokens ; start/goal tokens ; optional safety tokens]``
so both conditions share one set of parameters; "no safety" simply means the
memory is 192 tokens shorter.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .condition_encoders import (SafetyConditionEncoder, StartGoalEncoder,
                                 TimestepEmbedder, TrajectoryTokenEmbedder)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN: ``x * (1 + scale) + shift`` with per-sample ``[B,D]`` parameters."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Attention(nn.Module):
    """Plain multi-head attention (the component DiT uses)."""

    def __init__(self, dim: int, num_heads: int = 6, qkv_bias: bool = True):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim %d not divisible by heads %d" % (dim, num_heads))
        self.heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None,
                key_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``x [B,N,D]``; ``context [B,M,D]`` for cross-attention."""
        source = x if context is None else context
        qkv_x = self.qkv(x)
        q = qkv_x.chunk(3, dim=-1)[0]
        kv = self.qkv(source)
        _, k, v = kv.chunk(3, dim=-1)
        B, N, _ = q.shape
        M = k.shape[1]
        q = q.reshape(B, N, self.heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, M, self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, M, self.heads, self.head_dim).transpose(1, 2)
        bias = None
        if key_mask is not None:
            bias = torch.zeros(B, 1, 1, M, dtype=q.dtype, device=q.device)
            bias = bias.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        out = out.transpose(1, 2).reshape(B, N, self.heads * self.head_dim)
        return self.proj(out)


class DiTBlock(nn.Module):
    """One Conditional DiT block with AdaLN-Zero modulation of all sub-blocks."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = Attention(dim, num_heads)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(approximate="tanh"),
                                 nn.Linear(hidden, dim))
        # 3 sub-blocks x (shift, scale, gate)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor,
                context: torch.Tensor,
                key_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        shift_sa, scale_sa, gate_sa, shift_ca, scale_ca, gate_ca, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(9, dim=-1)
        x = x + gate_sa.unsqueeze(1) * self.self_attn(modulate(self.norm1(x), shift_sa, scale_sa))
        x = x + gate_ca.unsqueeze(1) * self.cross_attn(
            modulate(self.norm2(x), shift_ca, scale_ca), context=context, key_mask=key_mask)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        return x


class ConditionalDiT(nn.Module):
    """Conditional DiT-S over B-spline control points, predicting ``x0``.

    ``forward`` accepts the interface the design document specifies
    (section 11.4)::

        pred = model(q_t=q_t, t=t, map_memory=map_memory, cond=start_goal,
                     heading=heading, heading_valid=heading_valid,
                     safety_condition=safety_condition)

    ``safety_condition=None`` is the Map-only branch; passing
    ``{"path": [B,128,2], "ellipse": [B,64,6], "valid": [B,192]}`` switches to
    the Safety-conditioned branch of the SAME weights.
    """

    def __init__(self, hidden_dim: int = 384, depth: int = 12, num_heads: int = 6,
                 mlp_ratio: float = 4.0, num_controls: int = 32,
                 path_points: int = 128, num_ellipses: int = 64,
                 ellipse_feature_dim: int = 6, map_grid: int = 32,
                 map_channels: int = 192, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.num_controls = int(num_controls)
        self.path_points = int(path_points)
        self.num_ellipses = int(num_ellipses)
        self.map_grid = int(map_grid)
        self.map_tokens = self.map_grid * self.map_grid

        self.traj_embed = TrajectoryTokenEmbedder(hidden_dim, num_controls)
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.start_goal = StartGoalEncoder(hidden_dim)
        self.safety_encoder = SafetyConditionEncoder(hidden_dim, path_points,
                                                     num_ellipses, ellipse_feature_dim)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_ratio) for _ in range(int(depth))])
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=True, eps=1e-6)
        self.head = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    # ------------------------------------------------------------------ memory
    def build_condition_memory(
            self,
            map_memory: torch.Tensor,
            cond: torch.Tensor,
            heading: Optional[torch.Tensor] = None,
            heading_valid: Optional[torch.Tensor] = None,
            safety_condition: Optional[Dict[str, torch.Tensor]] = None):
        """``[map ; start/goal ; (safety)]`` -> ``(memory, key_mask)``."""
        pieces = [map_memory, self.start_goal(cond, heading, heading_valid)]
        mask_pieces = [torch.ones(map_memory.shape[0], map_memory.shape[1],
                                  dtype=torch.bool, device=map_memory.device),
                       torch.ones(map_memory.shape[0], 2, dtype=torch.bool,
                                  device=map_memory.device)]
        if safety_condition is not None:
            safe = self.safety_encoder(safety_condition["path"],
                                       safety_condition["ellipse"])
            pieces.append(safe)
            valid = safety_condition.get("valid")
            if valid is None:
                mask_pieces.append(torch.ones(safe.shape[0], safe.shape[1],
                                              dtype=torch.bool, device=safe.device))
            else:
                mask_pieces.append(valid.to(torch.bool))
        memory = torch.cat(pieces, dim=1)
        key_mask = torch.cat(mask_pieces, dim=1)
        if bool(key_mask.all()):
            key_mask = None
        return memory, key_mask

    # ----------------------------------------------------------------- forward
    def forward(self, q_t: torch.Tensor, t: torch.Tensor,
                map_memory: torch.Tensor, cond: torch.Tensor,
                heading: Optional[torch.Tensor] = None,
                heading_valid: Optional[torch.Tensor] = None,
                safety_condition: Optional[Dict[str, torch.Tensor]] = None,
                condition_memory: Optional[torch.Tensor] = None,
                condition_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``q_t [B,C,2]`` + timestep -> predicted clean controls ``[B,C,2]``.

        ``condition_memory`` / ``condition_mask`` may be supplied precomputed so a
        single training iteration does not rebuild the (identical) condition
        memory for both branches.
        """
        if q_t.shape[1] != self.num_controls:
            raise ValueError("expected %d control points, got %d"
                             % (self.num_controls, q_t.shape[1]))
        if condition_memory is None:
            condition_memory, condition_mask = self.build_condition_memory(
                map_memory, cond, heading, heading_valid, safety_condition)
        x = self.traj_embed(q_t)
        c = self.t_embedder(t)
        for block in self.blocks:
            x = block(x, c, condition_memory, condition_mask)
        return self.head(self.final_norm(x))

    # ------------------------------------------------------------- convenience
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def config_dict(self) -> Dict:
        return {
            "backbone": "conditional_dit",
            "variant": "DiT-S",
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "num_controls": self.num_controls,
            "path_points": self.path_points,
            "num_ellipses": self.num_ellipses,
            "map_grid": self.map_grid,
            "parameters": self.num_parameters(),
        }
