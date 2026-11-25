"""Conditional DiT acceptance tests (design document section 13.1 / 4.x)."""
from __future__ import annotations

import inspect

import torch

from gp_diffuser.models.conditional_dit import ConditionalDiT, DiTBlock


def test_dit_s_configuration(pipeline):
    model = pipeline["model"]
    assert isinstance(model, ConditionalDiT)
    assert model.hidden_dim == 384
    assert model.depth == 12
    assert model.num_heads == 6
    assert len(model.blocks) == 12
    assert all(isinstance(b, DiTBlock) for b in model.blocks)
    # FFN width = mlp_ratio * D
    assert model.blocks[0].mlp[0].out_features == 1536
    # AdaLN-Zero: 9 modulation parameters, zero-initialised
    assert model.blocks[0].adaLN_modulation[-1].out_features == 9 * 384
    assert torch.count_nonzero(model.blocks[0].adaLN_modulation[-1].weight) == 0


def test_predicts_bspline_controls(pipeline, sample):
    item, _ = sample
    model = pipeline["model"]
    feats = pipeline["map_encoder"].encode(item["occupancy"][None], item["sdf"][None])
    mem = pipeline["map_projection"](feats["occ_bottleneck"], feats["sdf_bottleneck"])
    q_t = torch.randn(1, 32, 2)
    with torch.no_grad():
        pred = model(q_t=q_t, t=torch.tensor([500]), map_memory=mem,
                     cond=item["start_goal"][None])
    assert tuple(pred.shape) == (1, 32, 2)


def test_output_head_is_zero_initialised(pipeline):
    model = pipeline["model"]
    assert torch.count_nonzero(model.head.weight) == 0
    assert torch.count_nonzero(model.head.bias) == 0


def test_two_modes_share_one_set_of_parameters(pipeline, sample):
    item, _ = sample
    model = pipeline["model"]
    feats = pipeline["map_encoder"].encode(item["occupancy"][None], item["sdf"][None])
    mem = pipeline["map_projection"](feats["occ_bottleneck"], feats["sdf_bottleneck"])
    safety = {
        "path": torch.randn(1, 128, 2) * 0.1,
        "ellipse": torch.randn(1, 64, 6) * 0.1,
        "valid": torch.ones(1, 192, dtype=torch.bool),
    }
    ids_before = {id(p) for p in model.parameters()}
    with torch.no_grad():
        out_u = model(q_t=torch.randn(1, 32, 2), t=torch.tensor([900]), map_memory=mem,
                      cond=item["start_goal"][None])
        out_c = model(q_t=torch.randn(1, 32, 2), t=torch.tensor([900]), map_memory=mem,
                      cond=item["start_goal"][None], safety_condition=safety)
    assert out_u.shape == out_c.shape == (1, 32, 2)
    assert {id(p) for p in model.parameters()} == ids_before
    assert model.config_dict()["parameters"] > 40_000_000


def test_condition_memory_layout(pipeline, sample):
    item, _ = sample
    model = pipeline["model"]
    mem = torch.zeros(1, 1024, 384)
    cond = item["start_goal"][None]
    mem_u, _ = model.build_condition_memory(mem, cond)
    assert mem_u.shape == (1, 1026, 384)          # map + start/goal
    safety = {"path": torch.zeros(1, 128, 2), "ellipse": torch.zeros(1, 64, 6)}
    mem_c, _ = model.build_condition_memory(mem, cond, safety_condition=safety)
    assert mem_c.shape == (1, 1026 + 192, 384)    # + 128 path + 64 ellipse


def test_safety_tokens_carry_modality_embeddings(pipeline):
    enc = pipeline["model"].safety_encoder
    assert enc.type_embed.shape == (2, 384)
    a = enc(torch.zeros(1, 128, 2), torch.zeros(1, 64, 6))
    b = enc(torch.ones(1, 128, 2), torch.zeros(1, 64, 6))
    assert a.shape == b.shape == (1, 192, 384)
    # path and ellipse blocks are distinguishable purely by their type embedding
    assert not torch.allclose(a[0, 0], a[0, -1])


def test_no_v5_modules_are_used(pipeline):
    model = pipeline["model"]
    source = inspect.getsource(type(model))
    for banned in ("traj_blocks", "final_blocks", "skeleton_blocks", "SkeletonEncoder",
                   "ControlBackbone", "FinalDenoiser", "TrajBlock"):
        assert banned not in source
    names = {name for name, _ in model.named_modules()}
    for banned in ("traj_blocks", "final_blocks", "skeleton"):
        assert not any(banned in n for n in names)


def test_masked_condition_tokens_are_excluded(pipeline, sample):
    """An all-False safety mask must make the safety branch equal the map-only one."""
    item, _ = sample
    model = pipeline["model"].eval()
    feats = pipeline["map_encoder"].encode(item["occupancy"][None], item["sdf"][None])
    mem = pipeline["map_projection"](feats["occ_bottleneck"], feats["sdf_bottleneck"])
    cond = item["start_goal"][None]
    q_t = torch.randn(1, 32, 2)
    safety = {"path": torch.randn(1, 128, 2), "ellipse": torch.randn(1, 64, 6),
              "valid": torch.zeros(1, 192, dtype=torch.bool)}
    with torch.no_grad():
        out_u = model(q_t=q_t, t=torch.tensor([700]), map_memory=mem, cond=cond)
        out_c = model(q_t=q_t, t=torch.tensor([700]), map_memory=mem, cond=cond,
                      safety_condition=safety)
    assert torch.allclose(out_u, out_c, atol=1e-6)
