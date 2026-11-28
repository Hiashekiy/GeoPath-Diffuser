"""Structure / shape / gradient tests for the new condition network.

Covers spec §9.1 (MapConditionEncoder), §9.2 (SafetyConditionEncoder),
§9.4 (dual-branch trainer) and §9.5 (inference sampler), plus the checkpoint
version gate of §8.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from gp_diffuser.models.conditional_dit import ARCHITECTURE_VERSION
from gp_diffuser.models.condition_encoders import MapConditionEncoder
from gp_diffuser.training.dataset import SyntheticTrajectoryDataset
from gp_diffuser.training.dual_branch_trainer import DualBranchTrainer
from gp_diffuser.training.safety_cache import build_safety_cache, load_safety_cache

CACHE_SPLIT = "train_pytest_structure"


def _features(pipeline, item, batch: int = 1):
    occ = item["occupancy"][None].repeat(batch, 1, 1, 1)
    sdf = item["sdf"][None].repeat(batch, 1, 1, 1)
    return pipeline["map_encoder"].encode(occ, sdf), occ, sdf


def _ellipse(batch: int = 1) -> torch.Tensor:
    ell = torch.zeros(batch, 64, 6)
    ell[..., 0:2] = 0.05
    ell[..., 2] = 0.03
    ell[..., 3] = 0.01
    ell[..., 4] = 1.0
    return ell


# --------------------------------------------------------------------------- #
# §9.1 MapConditionEncoder
# --------------------------------------------------------------------------- #
def test_map_condition_encoder_shapes(pipeline, sample):
    item, _ = sample
    feats, _, _ = _features(pipeline, item, batch=2)
    encoder = pipeline["map_condition_encoder"]
    assert tuple(feats["occ_bottleneck"].shape) == (2, 192, 32, 32)
    assert tuple(feats["sdf_bottleneck"].shape) == (2, 192, 32, 32)
    memory = encoder(feats["occ_bottleneck"], feats["sdf_bottleneck"])
    assert tuple(memory.shape) == (2, 1024, 384)
    # adapters and fusion keep the 32x32 resolution
    occ_feat = encoder.occ_adapter(feats["occ_bottleneck"])
    sdf_feat = encoder.sdf_adapter(feats["sdf_bottleneck"])
    fused = encoder.fusion(torch.cat([occ_feat, sdf_feat], dim=1))
    assert tuple(occ_feat.shape) == (2, 192, 32, 32)
    assert tuple(sdf_feat.shape) == (2, 192, 32, 32)
    assert tuple(fused.shape) == (2, 384, 32, 32)


def test_adapters_are_independent(pipeline):
    encoder = pipeline["map_condition_encoder"]
    occ_ids = {id(p) for p in encoder.occ_adapter.parameters()}
    sdf_ids = {id(p) for p in encoder.sdf_adapter.parameters()}
    assert occ_ids and sdf_ids and not (occ_ids & sdf_ids)
    assert type(encoder.occ_adapter) is type(encoder.sdf_adapter)


def test_adapters_and_fusion_receive_gradient(pipeline, sample):
    item, _ = sample
    feats, _, _ = _features(pipeline, item)
    encoder = pipeline["map_condition_encoder"]
    encoder.zero_grad(set_to_none=True)
    memory = encoder(feats["occ_bottleneck"], feats["sdf_bottleneck"])
    memory.square().mean().backward()

    def has_grad(module):
        return any(p.grad is not None and float(p.grad.abs().sum()) > 0
                   for p in module.parameters())

    assert has_grad(encoder.occ_adapter)
    assert has_grad(encoder.sdf_adapter)
    assert has_grad(encoder.fusion)
    encoder.zero_grad(set_to_none=True)


def test_frozen_map_encoder_and_inputs_are_untouched(pipeline, sample):
    item, _ = sample
    encoder = pipeline["map_condition_encoder"]
    feats, _, _ = _features(pipeline, item)
    occ_before = feats["occ_bottleneck"].clone()
    sdf_before = feats["sdf_bottleneck"].clone()
    memory = encoder(feats["occ_bottleneck"], feats["sdf_bottleneck"])
    memory.square().mean().backward()
    assert torch.equal(feats["occ_bottleneck"], occ_before), "input must not be modified"
    assert torch.equal(feats["sdf_bottleneck"], sdf_before)
    assert all(p.grad is None for p in pipeline["map_encoder"].encoder_parameters())
    encoder.zero_grad(set_to_none=True)


def test_token_order_matches_the_2d_position_encoding(pipeline):
    """Token ``r*32+c`` must correspond to spatial cell ``(r, c)``."""
    encoder = pipeline["map_condition_encoder"].eval()
    occ = torch.zeros(1, 192, 32, 32)
    sdf = torch.zeros(1, 192, 32, 32)
    r0, c0 = 5, 21
    occ[0, :, r0, c0] = 5.0                      # a single hot cell
    with torch.no_grad():
        memory = encoder(occ, sdf)
        # remove the positional encoding to isolate the spatial source
        tokens = memory[0] - encoder.pos.to(memory.dtype)
    norms = tokens.norm(dim=-1)
    assert int(norms.argmax().item()) == r0 * 32 + c0


def test_map_condition_encoder_matches_the_spec_configuration(pipeline):
    encoder = pipeline["map_condition_encoder"]
    assert isinstance(encoder, MapConditionEncoder)
    assert isinstance(encoder.occ_adapter.conv, torch.nn.Conv2d)
    assert encoder.occ_adapter.conv.kernel_size == (1, 1)
    assert encoder.occ_adapter.conv.bias is None
    assert isinstance(encoder.occ_adapter.norm, torch.nn.GroupNorm)
    assert encoder.occ_adapter.norm.num_groups == 16
    assert isinstance(encoder.fusion.depthwise.conv if hasattr(encoder.fusion.depthwise, "conv")
                      else encoder.fusion.depthwise, torch.nn.Conv2d)
    assert encoder.fusion.depthwise.groups == 384
    assert encoder.fusion.depthwise.kernel_size == (3, 3)
    assert encoder.fusion.pointwise.kernel_size == (1, 1)
    assert encoder.pos.shape == (1024, 384)


# --------------------------------------------------------------------------- #
# §9.2 SafetyConditionEncoder
# --------------------------------------------------------------------------- #
def test_safety_encoder_shapes_and_mask(pipeline):
    model = pipeline["model"]
    ellipse = _ellipse(3)
    memory, mask = model.build_safety_memory(ellipse)
    assert tuple(memory.shape) == (3, 64, 384)
    assert tuple(mask.shape) == (3, 64) and mask.dtype == torch.bool
    partial = torch.zeros(3, 64, dtype=torch.bool)
    partial[1, :10] = True
    memory2, mask2 = model.build_safety_memory(ellipse, partial)
    assert tuple(mask2.shape) == (3, 64)
    assert int(mask2.sum()) == 10
    with pytest.raises(ValueError):
        model.build_safety_memory(torch.zeros(3, 32, 6))


def test_safety_memory_has_no_path_tokens(pipeline, sample):
    item, _ = sample
    model = pipeline["model"]
    memory, _ = model.build_safety_memory(_ellipse(1))
    assert memory.shape[1] == 64, "safety memory must be ellipse tokens only"
    assert not hasattr(model.safety_encoder, "path_proj")


def test_all_false_mask_produces_no_nan(pipeline, sample):
    item, _ = sample
    model = pipeline["model"]
    feats, _, _ = _features(pipeline, item)
    memory_map = pipeline["map_condition_encoder"](feats["occ_bottleneck"],
                                                   feats["sdf_bottleneck"])
    ctx = model.build_map_context(memory_map, item["start_goal"][None],
                                  item["heading"][None], item["heading_valid"][None])
    safety, _ = model.build_safety_memory(_ellipse(1))
    with torch.no_grad():
        out = model(torch.randn(1, 32, 2), torch.tensor([400]), map_context=ctx,
                    safety_memory=safety,
                    safety_mask=torch.zeros(1, 64, dtype=torch.bool))
    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------- #
# §9.4 dual-branch trainer
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def two_sample_cache(pipeline, cfg):
    dataset = SyntheticTrajectoryDataset(cfg, "test")
    build_safety_cache(dataset, pipeline["net"], pipeline["safety_generator"],
                       pipeline["guidance"], cfg, split=CACHE_SPLIT, verbose=False)
    cache = load_safety_cache(cfg, CACHE_SPLIT)
    assert cache is not None
    return dataset, cache


def _trainer(pipeline, cfg):
    """Own DiT + MapConditionEncoder + optimizer (no shared-state pollution)."""
    from gp_diffuser.factory import (build_codec, build_map_condition_encoder,
                                     build_model, build_schedule)
    torch.manual_seed(0)
    model = build_model(cfg)
    encoder = build_map_condition_encoder(cfg)
    return DualBranchTrainer(cfg, model, encoder, pipeline["map_encoder"],
                             build_codec(cfg), build_schedule(cfg), device="cpu",
                             scene_scale=pipeline["transform"].scale)


def test_trainer_shares_map_context_and_ellipse_memory(pipeline, cfg, two_sample_cache):
    dataset, cache = two_sample_cache
    trainer = _trainer(pipeline, cfg)
    items = [dataset[i] for i in range(len(dataset))]
    index = np.arange(len(items))
    seen = {"q_t": [], "t": [], "ctx": [], "safety": []}
    model = trainer.model
    original = type(model).forward

    def spy(self, q_t=None, t=None, map_context=None, safety_memory=None,
            safety_mask=None, **kw):
        seen["q_t"].append(q_t)
        seen["t"].append(t)
        seen["ctx"].append(id(map_context))
        seen["safety"].append(None if safety_memory is None else tuple(safety_memory.shape))
        return original(self, q_t=q_t, t=t, map_context=map_context,
                        safety_memory=safety_memory, safety_mask=safety_mask, **kw)

    type(model).forward = spy
    try:
        trainer.train_step(items, cache, index)
    finally:
        type(model).forward = original
    assert len(seen["q_t"]) == 2
    assert torch.equal(seen["q_t"][0], seen["q_t"][1])
    assert torch.equal(seen["t"][0], seen["t"][1])
    assert seen["ctx"][0] == seen["ctx"][1], "both branches share one map context"
    assert seen["safety"][0] is None, "Map-only must not receive safety memory"
    assert seen["safety"][1] == (len(items), 64, 384)


def test_trainer_optimizer_and_gradient_coverage(pipeline, cfg, two_sample_cache):
    dataset, cache = two_sample_cache
    trainer = _trainer(pipeline, cfg)
    items = [dataset[i] for i in range(len(dataset))]
    index = np.arange(len(items))
    optimizer_ids = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    for param in trainer.map_condition_encoder.parameters():
        assert id(param) in optimizer_ids, "MapConditionEncoder must be optimised"
    for param in pipeline["map_encoder"].encoder_parameters():
        assert id(param) not in optimizer_ids, "frozen encoder must not be optimised"

    clipped = {}

    def spy_clip(parameters, max_norm, *a, **kw):
        clipped["ids"] = {id(p) for p in parameters}
        return original_clip(parameters, max_norm, *a, **kw)

    original_clip = torch.nn.utils.clip_grad_norm_
    torch.nn.utils.clip_grad_norm_ = spy_clip
    try:
        # the AdaLN-Zero gates start at exactly 0: step 1 only trains the head,
        # step 2 opens the gates (and trains the safety branch), and only step 3
        # gives the Map Cross-Attention / MapConditionEncoder gradient.  See
        # tests/test_gradient_onset.py for the measured onset.
        for _ in range(3):
            trainer.train_step(items, cache, index)
    finally:
        torch.nn.utils.clip_grad_norm_ = original_clip
    encoder_ids = {id(p) for p in trainer.map_condition_encoder.parameters()}
    model_ids = {id(p) for p in trainer.model.parameters() if p.requires_grad}
    assert clipped["ids"] == encoder_ids | model_ids, "clip must cover DiT + map encoder"
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0
               for p in trainer.map_condition_encoder.parameters()), (
        "the map encoder must receive gradient once the AdaLN gates are non-zero")
    assert all(p.grad is None for p in pipeline["map_encoder"].encoder_parameters())


def test_adaln_zero_gates_delay_map_gradient(cfg, pipeline, sample):
    """AdaLN-Zero + safety gate 0.1: why the map encoder starts at zero gradient.

    Every block's SA / Map CA / MLP residual is gated by an AdaLN-Zero parameter
    that starts at exactly 0, so on the very first backward those branches (and the
    MapConditionEncoder feeding Map CA) receive zero gradient, while the safety
    path -- gated by the learnable ``alpha_l`` initialised to 0.1 -- already does.
    Once a step has made the gates non-zero the map encoder receives gradient as
    well.  This is the documented behaviour of the chosen initialisation, not a
    wiring bug.
    """
    from gp_diffuser.factory import build_map_condition_encoder, build_model

    item, _ = sample
    model = build_model(cfg)
    encoder = build_map_condition_encoder(cfg)
    feats = pipeline["map_encoder"].encode(item["occupancy"][None], item["sdf"][None])
    ellipse = _ellipse(1)
    q = torch.randn(1, 32, 2)
    t = torch.tensor([500])

    def run():
        memory = encoder(feats["occ_bottleneck"], feats["sdf_bottleneck"])
        ctx = model.build_map_context(memory, item["start_goal"][None],
                                     item["heading"][None], item["heading_valid"][None])
        safety, mask = model.build_safety_memory(ellipse, None)
        pred = model(q, t, map_context=ctx, safety_memory=safety, safety_mask=mask)
        pred.square().mean().backward()
        grad = {name: (0.0 if p.grad is None else float(p.grad.abs().sum()))
                for name, p in encoder.named_parameters()}
        safety_grad = float(sum(p.grad.abs().sum() for p in model.safety_encoder.parameters()
                                if p.grad is not None))
        model.zero_grad(set_to_none=True)
        encoder.zero_grad(set_to_none=True)
        return grad, safety_grad

    init_grad, _ = run()
    assert max(init_grad.values()) == 0.0, "zero-init head + zero gates => no gradient"

    torch.nn.init.normal_(model.head.weight, std=0.02)
    gated_grad, safety_grad = run()
    assert max(gated_grad.values()) == 0.0, "AdaLN-Zero gates still block the map path"
    assert safety_grad > 0.0, "the safety path is gated by alpha=0.1, not by zero"

    for block in model.blocks:                 # simulate one optimizer step
        with torch.no_grad():
            block.adaLN_modulation[-1].bias.fill_(0.1)
    open_grad, _ = run()
    assert max(open_grad.values()) > 0.0, "with non-zero gates the map encoder trains"


# --------------------------------------------------------------------------- #
# §9.5 inference sampler
# --------------------------------------------------------------------------- #
def test_sampler_builds_conditions_once(pipeline, sample, monkeypatch):
    item, _ = sample
    model = pipeline["model"]
    counts = {"map_ctx": 0, "safety": 0}
    original_ctx = type(model).build_map_context
    original_safety = type(model).build_safety_memory

    def ctx_spy(self, *a, **kw):
        counts["map_ctx"] += 1
        return original_ctx(self, *a, **kw)

    def safety_spy(self, *a, **kw):
        counts["safety"] += 1
        return original_safety(self, *a, **kw)

    monkeypatch.setattr(type(model), "build_map_context", ctx_spy)
    monkeypatch.setattr(type(model), "build_safety_memory", safety_spy)
    out = pipeline["sampler"].sample(item["occupancy"][None], item["sdf"][None],
                                     item["start_goal"][None],
                                     heading=item["heading"][None],
                                     heading_valid=item["heading_valid"][None],
                                     steps=8, seed=0)
    assert counts["map_ctx"] == 1, "Map Context is built once per run"
    assert counts["safety"] <= 1, "ellipse memory is encoded at most once"
    assert out["map_context_tokens"] == 1026
    if out["safety_memory"] is not None:
        assert tuple(out["safety_memory"].shape)[1:] == (64, 384)
        assert tuple(out["safety_mask"].shape)[1:] == (64,)


def test_safety_condition_has_no_path_tokens(pipeline, sample):
    item, _ = sample
    out = pipeline["sampler"].sample(item["occupancy"][None], item["sdf"][None],
                                     item["start_goal"][None],
                                     heading=item["heading"][None],
                                     heading_valid=item["heading_valid"][None],
                                     steps=8, seed=0)
    if out["safety_condition"] is None:
        pytest.skip("all searches failed for this sample")
    assert set(out["safety_condition"]) == {"ellipse", "valid"}
    assert out["search_path_m"] is not None, "the path still feeds the geometry"


def test_sampler_branch_switch_and_guidance_interface(pipeline, sample):
    item, _ = sample
    out = pipeline["sampler"].sample(item["occupancy"][None], item["sdf"][None],
                                     item["start_goal"][None],
                                     heading=item["heading"][None],
                                     heading_valid=item["heading_valid"][None],
                                     steps=8, seed=0)
    branches = [step["branch"] for step in out["steps_trace"]]
    assert branches[0] == "map_only" and branches[-1] in ("map_only", "safety")
    for key in ("controls", "trajectory_m", "valid", "dense_collision_fraction",
                "guidance_stats", "final_qp_status", "corridor_certified",
                "pack", "timing"):
        assert key in out, "the sampler output interface must stay unchanged"


# --------------------------------------------------------------------------- #
# §8 checkpoint compatibility
# --------------------------------------------------------------------------- #
def test_checkpoint_version_gate(pipeline, tmp_path):
    from gp_diffuser.factory import load_checkpoint
    model = pipeline["model"]
    encoder = pipeline["map_condition_encoder"]
    good = tmp_path / "good.pt"
    torch.save({"architecture_version": ARCHITECTURE_VERSION,
                "model": model.state_dict(),
                "map_condition_encoder": encoder.state_dict(),
                "config": model.config_dict()}, good)
    payload = load_checkpoint(str(good), model, encoder)
    assert payload["architecture_version"] == ARCHITECTURE_VERSION

    legacy = tmp_path / "legacy.pt"
    torch.save({"model": model.state_dict(),
                "map_projection": encoder.state_dict(),
                "config": {"backbone": "conditional_dit"}}, legacy)
    with pytest.raises(RuntimeError) as error:
        load_checkpoint(str(legacy), model, encoder)
    assert "retrain" in str(error.value).lower()
