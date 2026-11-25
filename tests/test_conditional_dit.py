"""Conditional DiT acceptance tests (spec: 网络结构修改实施方案 §9.3).

The condition interface is now explicit:

    map_context  = model.build_map_context(map_memory, start_goal, heading, valid)
    safety_memory, safety_mask = model.build_safety_memory(ellipse, ellipse_valid)
    pred = model(q_t, t, map_context=..., safety_memory=..., safety_mask=...)
"""
from __future__ import annotations

import inspect

import pytest
import torch

from gp_diffuser.models.conditional_dit import (ARCHITECTURE_VERSION, ConditionalDiT,
                                                DiTBlock, modulate)


def _conditions(pipeline, item, monkeypatch=None):
    occ = item["occupancy"][None]
    sdf = item["sdf"][None]
    feats = pipeline["map_encoder"].encode(occ, sdf)
    mem = pipeline["map_condition_encoder"](feats["occ_bottleneck"],
                                            feats["sdf_bottleneck"])
    model = pipeline["model"]
    ctx = model.build_map_context(mem, item["start_goal"][None],
                                  item["heading"][None], item["heading_valid"][None])
    return model, ctx


def _ellipse(batch: int = 1) -> torch.Tensor:
    ell = torch.zeros(batch, 64, 6)
    ell[..., 0:2] = 0.05
    ell[..., 2] = 0.03
    ell[..., 3] = 0.01
    ell[..., 4] = 1.0
    return ell


def test_dit_s_configuration(pipeline):
    model = pipeline["model"]
    assert isinstance(model, ConditionalDiT)
    assert model.architecture_version == ARCHITECTURE_VERSION
    assert model.hidden_dim == 384
    assert model.depth == 12
    assert model.num_heads == 6
    assert len(model.blocks) == 12
    assert all(isinstance(b, DiTBlock) for b in model.blocks)
    block = model.blocks[0]
    assert block.mlp[0].out_features == 1536                 # mlp_ratio 4
    assert block.adaLN_modulation[-1].out_features == 11 * 384
    assert torch.count_nonzero(block.adaLN_modulation[-1].weight) == 0
    assert float(block.safety_gate.item()) == pytest.approx(0.1, abs=1e-6)


def test_four_subblocks_and_two_independent_cross_attentions(pipeline):
    block = pipeline["model"].blocks[0]
    for name in ("self_attn", "map_cross_attn", "safety_cross_attn"):
        assert hasattr(block, name)
    for name in ("norm_sa", "norm_map", "norm_safe", "norm_mlp"):
        assert hasattr(block, name)
    map_ids = {id(p) for p in block.map_cross_attn.parameters()}
    safe_ids = {id(p) for p in block.safety_cross_attn.parameters()}
    assert not (map_ids & safe_ids), "Map and Safety cross-attention must not share params"


def test_predicts_bspline_controls(pipeline, sample):
    item, _ = sample
    model, ctx = _conditions(pipeline, item)
    with torch.no_grad():
        pred = model(torch.randn(1, 32, 2), torch.tensor([500]), map_context=ctx)
    assert tuple(pred.shape) == (1, 32, 2)


def test_map_context_and_safety_memory_shapes(pipeline, sample):
    item, _ = sample
    model, ctx = _conditions(pipeline, item)
    assert tuple(ctx.shape) == (1, 1026, 384)          # 1024 map + 2 start/goal
    memory, mask = model.build_safety_memory(_ellipse(2))
    assert tuple(memory.shape) == (2, 64, 384)
    assert tuple(mask.shape) == (2, 64) and mask.dtype == torch.bool
    assert bool(mask.all())


def test_safety_encoder_has_no_path_tokens(pipeline):
    enc = pipeline["model"].safety_encoder
    assert not hasattr(enc, "path_proj")
    assert not hasattr(enc, "path_pos")
    params = inspect.signature(enc.forward).parameters
    assert list(params) == ["ellipse"], "the encoder must take ellipses only"


def test_map_only_skips_safety_cross_attention(pipeline, sample, monkeypatch):
    item, _ = sample
    model, ctx = _conditions(pipeline, item)
    calls = {"n": 0}
    original = type(model.blocks[0]).__dict__["_safety_residual"]

    def counting(self, *args, **kwargs):
        calls["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(model.blocks[0]), "_safety_residual", counting)
    q = torch.randn(1, 32, 2)
    t = torch.tensor([700])
    with torch.no_grad():
        model(q, t, map_context=ctx)
        assert calls["n"] == 0, "Map-only mode must not run the safety branch"
        memory, mask = model.build_safety_memory(_ellipse(1))
        model(q, t, map_context=ctx, safety_memory=memory, safety_mask=mask)
        assert calls["n"] == 12, "the safety branch must run in every block"


def test_safety_residual_is_zero_for_an_all_false_mask(pipeline, sample):
    item, _ = sample
    model, ctx = _conditions(pipeline, item)
    q = torch.randn(1, 32, 2)
    t = torch.tensor([700])
    memory, _ = model.build_safety_memory(_ellipse(1))
    empty = torch.zeros(1, 64, dtype=torch.bool)
    with torch.no_grad():
        out_u = model(q, t, map_context=ctx)
        out_e = model(q, t, map_context=ctx, safety_memory=memory, safety_mask=empty)
    assert torch.allclose(out_u, out_e, atol=1e-6)
    assert torch.isfinite(out_e).all()


def test_partial_mask_excludes_invalid_ellipses(pipeline, sample):
    """Changing an invalid ellipse must not change the prediction."""
    item, _ = sample
    model, ctx = _conditions(pipeline, item)
    mask = torch.zeros(1, 64, dtype=torch.bool)
    mask[0, :32] = True
    ellipse = _ellipse(1)
    perturbed = ellipse.clone()
    perturbed[0, 32:] = 5.0                     # only the masked-out rows change
    with torch.no_grad():
        memory_a, mask_a = model.build_safety_memory(ellipse, mask)
        memory_b, _ = model.build_safety_memory(perturbed, mask)
        q = torch.randn(1, 32, 2)
        t = torch.tensor([700])
        out_a = model(q, t, map_context=ctx, safety_memory=memory_a, safety_mask=mask_a)
        out_b = model(q, t, map_context=ctx, safety_memory=memory_b, safety_mask=mask_a)
    assert torch.allclose(out_a, out_b, atol=1e-6)
    assert torch.isfinite(out_a).all()


def test_modes_share_one_set_of_parameters(pipeline, sample):
    item, _ = sample
    model, ctx = _conditions(pipeline, item)
    before = {name: id(p) for name, p in model.named_parameters()}
    memory, mask = model.build_safety_memory(_ellipse(1))
    q = torch.randn(1, 32, 2)
    t = torch.tensor([900])
    with torch.no_grad():
        out_u = model(q, t, map_context=ctx)
        out_c = model(q, t, map_context=ctx, safety_memory=memory, safety_mask=mask)
    assert out_u.shape == out_c.shape == (1, 32, 2)
    assert {name: id(p) for name, p in model.named_parameters()} == before
    # the two modes differ at the block level (the output head is zero-initialised)
    block = model.blocks[0]
    x = model.traj_embed(q)
    c = model.t_embedder(t)
    y_u = block(x, c, ctx)
    y_c = block(x, c, ctx, memory, mask)
    assert not torch.allclose(y_u, y_c)


def test_safety_branch_receives_gradient(pipeline, sample):
    item, _ = sample
    model, ctx = _conditions(pipeline, item)
    memory, mask = model.build_safety_memory(_ellipse(1))
    # a non-zero head so the safety path influences the loss
    torch.nn.init.normal_(model.head.weight, std=0.02)
    pred = model(torch.randn(1, 32, 2), torch.tensor([500]), map_context=ctx,
                 safety_memory=memory, safety_mask=mask)
    pred.square().mean().backward()
    block = model.blocks[0]
    assert block.safety_gate.grad is not None and float(block.safety_gate.grad.abs()) > 0
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in block.safety_cross_attn.parameters())
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in model.safety_encoder.parameters())
    model.zero_grad(set_to_none=True)


def test_no_v5_modules_are_used(pipeline):
    model = pipeline["model"]
    source = inspect.getsource(type(model))
    for banned in ("traj_blocks", "final_blocks", "skeleton_blocks", "SkeletonEncoder",
                   "ControlBackbone", "FinalDenoiser", "TrajBlock"):
        assert banned not in source
    names = {name for name, _ in model.named_modules()}
    for banned in ("traj_blocks", "final_blocks", "skeleton"):
        assert not any(banned in n for n in names)


def test_map_context_length_is_validated(pipeline, sample):
    item, _ = sample
    model, _ = _conditions(pipeline, item)
    with pytest.raises(ValueError):
        model(torch.randn(1, 32, 2), torch.tensor([1]), map_context=torch.randn(1, 100, 384))
    with pytest.raises(ValueError):
        model(torch.randn(1, 32, 2), torch.tensor([1]),
              map_context=torch.randn(1, 1026, 384),
              safety_memory=torch.randn(1, 32, 384))
