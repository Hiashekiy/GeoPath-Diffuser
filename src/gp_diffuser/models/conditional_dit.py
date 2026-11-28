"""Standard Conditional Diffusion Transformer (DiT-S) for B-spline controls.

This is a NEW network, not a derivative of TrajSafe-Diffuser V5: no ``TrajBlock``,
no ``Control Backbone``, no ``Final Denoiser``, no ``Skeleton Encoder``.

Structure (spec: 网络结构修改实施方案 §4):

* trajectory tokens = ``Linear(2 -> D)`` on the control points + 1-D sincos PE;
* timestep conditioning = sinusoidal embedding + MLP, injected by AdaLN;
* every block = ``Self-Attention -> Map Cross-Attention -> (optional) Safety
  Cross-Attention -> MLP``, with **independent** parameters and LayerNorms for the
  two cross-attentions;
* Map Context (``1024`` map tokens + ``2`` start/goal tokens) is ALWAYS used and is
  identical in both modes;
* Safety Cross-Attention reads ONLY the 64 ellipse tokens and is injected through a
  per-block learnable gate ``alpha_l`` (init ``0.1``), not through an extra
  zero-initialised residual gate;
* Map-only mode skips the Safety Cross-Attention entirely (it is never faked with
  zero ellipse tokens);
* output head = ``LayerNorm -> Linear(D -> 2)`` predicting ``x0``.

Both conditional modes call the SAME instance and share every parameter except the
safety-only sub-modules, which are simply not executed in Map-only mode.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .condition_encoders import (MapConditionEncoder, SafetyConditionEncoder,
                                 StartGoalEncoder, TimestepEmbedder,
                                 TrajectoryTokenEmbedder)

ARCHITECTURE_VERSION = "conditional_dit_dual_map_fusion_dual_cross_attention_v2"

#: AdaLN outputs per block: sa(3) + map(3) + safety(2) + mlp(3)
_ADALN_CHUNKS = 11


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN: ``x * (1 + scale) + shift`` with per-sample ``[B,D]`` parameters."""
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Attention(nn.Module):
    """Plain multi-head attention (the component DiT uses).

    ``kv`` accepts a pre-projected ``(k, v)`` pair: the map context and the
    ellipse memory never change across DDIM steps, so their K/V are projected
    once per run (``project_kv``) instead of once per step per block.
    """

    def __init__(self, dim: int, num_heads: int = 6, qkv_bias: bool = True):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim %d not divisible by heads %d" % (dim, num_heads))
        self.heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def _heads(self, t: torch.Tensor) -> torch.Tensor:
        B, M, _ = t.shape
        return t.reshape(B, M, self.heads, self.head_dim).transpose(1, 2)

    def project_kv(self, context: torch.Tensor) -> tuple:
        """``context [B,M,D]`` -> head-shaped ``(k, v)`` for reuse across calls."""
        _, k, v = self.qkv(context).chunk(3, dim=-1)
        return self._heads(k), self._heads(v)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None,
                key_mask: Optional[torch.Tensor] = None,
                kv: Optional[tuple] = None) -> torch.Tensor:
        """``x [B,N,D]``; ``context [B,M,D]`` for cross-attention."""
        if context is None and kv is None:
            q, k, v = (self._heads(t) for t in self.qkv(x).chunk(3, dim=-1))
        else:
            q = self._heads(self.qkv(x).chunk(3, dim=-1)[0])
            k, v = self.project_kv(context) if kv is None else kv
        B, N = q.shape[0], q.shape[2]
        M = k.shape[2]
        bias = None
        if key_mask is not None:
            bias = torch.zeros(B, 1, 1, M, dtype=q.dtype, device=q.device)
            bias = bias.masked_fill(~key_mask[:, None, None, :], float("-inf"))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        out = out.transpose(1, 2).reshape(B, N, self.heads * self.head_dim)
        return self.proj(out)


def mask_plan(mask: Optional[torch.Tensor]) -> Tuple[str, Optional[torch.Tensor]]:
    """Decide ONCE per forward how a fixed ellipse mask is applied.

    ``"plain"``  no ellipse is masked out -> run the cross-attention unmasked;
    ``"none"``   every row is masked out  -> the residual is exactly zero;
    ``"subset"`` some rows are valid      -> gather those rows (index returned).

    Doing this per block costs one device->host sync per layer, per step: the
    mask never changes during sampling, so it is resolved a single time.
    """
    if mask is None:
        return "plain", None
    rows = mask.any(dim=-1)
    if not bool(rows.any()):
        return "none", None
    if bool(mask.all()):
        return "plain", None
    return "subset", rows.nonzero(as_tuple=False).squeeze(1)


class DiTBlock(nn.Module):
    """One Conditional DiT block: SA -> Map CA -> optional Safety CA -> MLP."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 safety_gate_init: float = 0.1):
        super().__init__()
        self.norm_sa = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(dim, num_heads)
        self.norm_map = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.map_cross_attn = Attention(dim, num_heads)
        self.norm_safe = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.safety_cross_attn = Attention(dim, num_heads)
        self.norm_mlp = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(approximate="tanh"),
                                 nn.Linear(hidden, dim))
        # AdaLN-Zero for SA / Map CA / MLP, plus an independent shift/scale for the
        # safety branch (its residual weight is the learnable alpha, see below)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(),
                                              nn.Linear(dim, _ADALN_CHUNKS * dim))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        # per-block learnable safety gate, small non-zero init (no second zero gate)
        self.safety_gate = nn.Parameter(torch.full((1,), float(safety_gate_init)))

    # --------------------------------------------------------------- safety CA
    def _safety_residual(self, x: torch.Tensor, shift: torch.Tensor,
                         scale: torch.Tensor, safety_kv: tuple,
                         safety_mask: Optional[torch.Tensor] = None,
                         safety_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``alpha_l * SafetyCA(AdaLN_safe(x), ellipse memory)``.

        Samples whose ellipse mask is entirely False are skipped by the caller, so
        their safety residual is exactly zero and the attention never sees an all
        ``-inf`` row (which would produce NaN).  The valid rows are computed in one
        batched call and scattered back; ``safety_idx`` is decided once per
        forward by :func:`mask_plan`, never per block.
        """
        query = modulate(self.norm_safe(x), shift, scale)
        if safety_idx is None:
            return self.safety_gate * self.safety_cross_attn(query, kv=safety_kv)
        selected = self.safety_cross_attn(
            query.index_select(0, safety_idx),
            key_mask=safety_mask.index_select(0, safety_idx),
            kv=tuple(t.index_select(0, safety_idx) for t in safety_kv))
        residual = torch.zeros_like(query).index_copy(0, safety_idx, selected)
        return self.safety_gate * residual

    # ----------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor, c: torch.Tensor,
                map_context: Optional[torch.Tensor] = None,
                safety_memory: Optional[torch.Tensor] = None,
                safety_mask: Optional[torch.Tensor] = None,
                map_kv: Optional[tuple] = None,
                safety_kv: Optional[tuple] = None,
                safety_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        (shift_sa, scale_sa, gate_sa,
         shift_map, scale_map, gate_map,
         shift_safe, scale_safe,
         shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(c).chunk(_ADALN_CHUNKS, dim=-1)

        x = x + gate_sa.unsqueeze(1) * self.self_attn(
            modulate(self.norm_sa(x), shift_sa, scale_sa))
        x = x + gate_map.unsqueeze(1) * self.map_cross_attn(
            modulate(self.norm_map(x), shift_map, scale_map),
            context=map_context, kv=map_kv)
        if safety_kv is None and safety_memory is not None:
            mode, safety_idx = mask_plan(safety_mask)
            if mode != "none":
                safety_kv = self.safety_cross_attn.project_kv(safety_memory)
        if safety_kv is not None:
            x = x + self._safety_residual(x, shift_safe, scale_safe,
                                          safety_kv, safety_mask, safety_idx)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm_mlp(x), shift_mlp, scale_mlp))
        return x


class ConditionalDiT(nn.Module):
    """Conditional DiT-S over B-spline control points, predicting ``x0``.

    Explicit interfaces (spec §7.2)::

        map_context = model.build_map_context(map_memory, start_goal, heading,
                                              heading_valid)      # [B,1026,D]
        safety_memory, safety_mask = model.build_safety_memory(ellipse, valid)
                                                                 # [B,64,D], [B,64]
        pred_u = model(q_t, t, map_context=map_context)
        pred_c = model(q_t, t, map_context=map_context,
                       safety_memory=safety_memory, safety_mask=safety_mask)
    """

    architecture_version = ARCHITECTURE_VERSION

    def __init__(self, hidden_dim: int = 384, depth: int = 12, num_heads: int = 6,
                 mlp_ratio: float = 4.0, num_controls: int = 32,
                 num_ellipses: int = 64, ellipse_feature_dim: int = 6,
                 map_grid: int = 32, map_channels: int = 192,
                 safety_gate_init: float = 0.1, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.num_controls = int(num_controls)
        self.num_ellipses = int(num_ellipses)
        self.map_grid = int(map_grid)
        self.map_tokens = self.map_grid * self.map_grid
        self.safety_gate_init = float(safety_gate_init)

        self.traj_embed = TrajectoryTokenEmbedder(hidden_dim, num_controls)
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.start_goal = StartGoalEncoder(hidden_dim)
        self.safety_encoder = SafetyConditionEncoder(hidden_dim, num_ellipses,
                                                     ellipse_feature_dim)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_ratio, safety_gate_init)
            for _ in range(int(depth))])
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=True, eps=1e-6)
        self.head = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    # ------------------------------------------------------------- conditions
    def build_map_context(self, map_memory: torch.Tensor,
                          start_goal: torch.Tensor,
                          heading: Optional[torch.Tensor] = None,
                          heading_valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``[map tokens ; start/goal tokens]`` -> ``[B,1026,D]`` (always on)."""
        if map_memory.shape[1] != self.map_tokens:
            raise ValueError("expected %d map tokens, got %d"
                             % (self.map_tokens, map_memory.shape[1]))
        sg = self.start_goal(start_goal, heading, heading_valid)
        return torch.cat([map_memory, sg], dim=1)

    def build_safety_memory(self, ellipse: torch.Tensor,
                            ellipse_valid: Optional[torch.Tensor] = None
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``ellipse [B,64,6]`` -> ``(memory [B,64,D], mask [B,64])``."""
        memory = self.safety_encoder(ellipse)
        if ellipse_valid is None:
            mask = torch.ones(memory.shape[0], memory.shape[1],
                              dtype=torch.bool, device=memory.device)
        else:
            mask = ellipse_valid.to(torch.bool)
            if tuple(mask.shape) != (memory.shape[0], memory.shape[1]):
                raise ValueError("ellipse_valid must be [B,%d], got %s"
                                 % (self.num_ellipses, tuple(mask.shape)))
        return memory, mask

    # ----------------------------------------------------- frozen K/V (inference)
    def build_map_kv(self, map_context: torch.Tensor) -> List[tuple]:
        """Per-block ``(k, v)`` of the map context, built ONCE per run.

        The map context is frozen for the whole reverse process, so its K/V do
        not change between DDIM steps: projecting them once removes the dominant
        per-step cost (12 x projection over 1026 tokens) without changing the
        result.
        """
        return [b.map_cross_attn.project_kv(map_context) for b in self.blocks]

    def build_safety_kv(self, safety_memory: torch.Tensor) -> List[tuple]:
        """Per-block ``(k, v)`` of the ellipse memory, built ONCE per run."""
        return [b.safety_cross_attn.project_kv(safety_memory) for b in self.blocks]

    # ---------------------------------------------------------------- forward
    def forward(self, q_t: torch.Tensor, t: torch.Tensor,
                map_context: Optional[torch.Tensor] = None,
                safety_memory: Optional[torch.Tensor] = None,
                safety_mask: Optional[torch.Tensor] = None,
                map_kv: Optional[List[tuple]] = None,
                safety_kv: Optional[List[tuple]] = None) -> torch.Tensor:
        """``q_t [B,C,2]`` + timestep -> predicted clean controls ``[B,C,2]``.

        ``safety_memory=None`` is the Map-only mode: the Safety Cross-Attention is
        not executed at all.  Passing it executes every block's safety path with
        the ellipse memory only.

        ``map_kv`` / ``safety_kv`` accept the pre-projected pairs from
        :meth:`build_map_kv` / :meth:`build_safety_kv` (inference only: they are
        constant across steps).  ``map_context`` may be omitted in that case.
        """
        if q_t.shape[1] != self.num_controls:
            raise ValueError("expected %d control points, got %d"
                             % (self.num_controls, q_t.shape[1]))
        expected = self.map_tokens + 2
        if map_context is not None and map_context.shape[1] != expected:
            raise ValueError("map_context must have %d tokens, got %d"
                             % (expected, map_context.shape[1]))
        if map_context is None and map_kv is None:
            raise ValueError("one of map_context / map_kv is required")
        if safety_memory is not None and safety_memory.shape[1] != self.num_ellipses:
            raise ValueError("safety_memory must have %d ellipse tokens, got %d"
                             % (self.num_ellipses, safety_memory.shape[1]))

        # the ellipse mask is fixed for the whole run: resolve it once, not per
        # block (an all-valid mask is the same as no mask at all)
        safety_idx = None
        if safety_kv is not None or safety_memory is not None:
            mode, safety_idx = mask_plan(safety_mask)
            if mode == "none":
                safety_kv, safety_memory, safety_mask = None, None, None
            elif mode == "plain":
                safety_mask = None

        x = self.traj_embed(q_t)
        c = self.t_embedder(t)
        for i, block in enumerate(self.blocks):
            x = block(x, c, map_context, safety_memory, safety_mask,
                      map_kv=None if map_kv is None else map_kv[i],
                      safety_kv=None if safety_kv is None else safety_kv[i],
                      safety_idx=safety_idx)
        return self.head(self.final_norm(x))

    # ------------------------------------------------------------- reporting
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def safety_parameter_breakdown(self) -> Dict:
        """Parameter split of the safety-only sub-modules (spec §11.2)."""
        def count(module):
            return sum(p.numel() for p in module.parameters())
        return {
            "safety_condition_encoder": count(self.safety_encoder),
            "safety_cross_attention": sum(count(b.safety_cross_attn) for b in self.blocks),
            "safety_gates": sum(b.safety_gate.numel() for b in self.blocks),
            "map_cross_attention": sum(count(b.map_cross_attn) for b in self.blocks),
            "self_attention": sum(count(b.self_attn) for b in self.blocks),
            "mlp": sum(count(b.mlp) for b in self.blocks),
            "adaln_modulation": sum(count(b.adaLN_modulation) for b in self.blocks),
            "total": self.num_parameters(),
        }

    def config_dict(self) -> Dict:
        return {
            "architecture_version": ARCHITECTURE_VERSION,
            "backbone": "conditional_dit",
            "variant": "DiT-S",
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "num_controls": self.num_controls,
            "num_ellipses": self.num_ellipses,
            "map_grid": self.map_grid,
            "map_tokens": self.map_tokens,
            "map_context_tokens": self.map_tokens + 2,
            "condition_fusion": "separate_map_safety_cross_attention",
            "safety_residual_gate_init": self.safety_gate_init,
            "parameters": self.num_parameters(),
        }
