"""Gradient-onset verification on a FRESH model, with real optimiser steps.

No parameter is hand-edited and no step is simulated: a freshly built
Conditional DiT + MapConditionEncoder is trained for three real
forward/backward/optimizer.step cycles, and the gradients are snapshotted inside
``optimizer.step`` (i.e. after ``backward`` and before the weights move).

Why the onset looks like this (all three effects are by design, not bugs):

* step 1 — the output head is zero-initialised, so ``dL/dx_final =
  head.weight^T dL/dpred = 0``: only ``final_norm`` and ``head`` receive
  gradient.  Every residual branch is additionally gated by an AdaLN-Zero
  parameter that is exactly 0.
* step 2 — after step 1 the head is non-zero, so ``dL/dx_final != 0`` and the
  AdaLN parameters (which own the shift/scale/gate outputs) now receive
  gradient.  The sublayer *weights* are still multiplied by ``gate == 0``, so
  they stay at zero.  The safety path is gated by ``alpha_l = 0.1`` (not by an
  AdaLN gate), so it already receives gradient.
* step 3 — step 2 made the gates non-zero, so Self-Attention, Map
  Cross-Attention, the MLP, the timestep embedder and therefore the
  MapConditionEncoder start receiving gradient.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from gp_diffuser.factory import (build_codec, build_map_condition_encoder, build_model,
                                 build_schedule)
from gp_diffuser.training.dual_branch_trainer import DualBranchTrainer
from gp_diffuser.training.dataset import SyntheticTrajectoryDataset
from gp_diffuser.training.safety_cache import build_safety_cache, load_safety_cache

CACHE_SPLIT = "train_pytest_onset"
ZERO = 1e-12


@pytest.fixture(scope="module")
def onset_cache(pipeline, cfg):
    dataset = SyntheticTrajectoryDataset(cfg, "test", size=2)
    build_safety_cache(dataset, pipeline["net"], pipeline["safety_generator"],
                       pipeline["guidance"], cfg, split=CACHE_SPLIT, verbose=False)
    cache = load_safety_cache(cfg, CACHE_SPLIT)
    assert cache is not None
    return dataset, cache


def _fresh_trainer(cfg, pipeline):
    """Fresh DiT + fresh MapConditionEncoder; only the frozen encoder is shared."""
    torch.manual_seed(0)
    model = build_model(cfg)
    encoder = build_map_condition_encoder(cfg)
    trainer = DualBranchTrainer(cfg, model, encoder, pipeline["map_encoder"],
                                build_codec(cfg), build_schedule(cfg),
                                device="cpu",
                                scene_scale=pipeline["transform"].scale)
    return model, encoder, trainer


def _grad_norm(modules) -> float:
    total = 0.0
    for module in modules:
        for param in module.parameters():
            if param.grad is not None:
                total += float(param.grad.abs().sum())
    return total


def _snapshot(model, encoder, pipeline):
    blocks = list(model.blocks)
    return {
        "output_head": _grad_norm([model.final_norm, model.head]),
        "adaln": _grad_norm([b.adaLN_modulation for b in blocks]),
        "self_attn": _grad_norm([b.self_attn for b in blocks]),
        "map_cross_attn": _grad_norm([b.map_cross_attn for b in blocks]),
        "mlp": _grad_norm([b.mlp for b in blocks]),
        "traj_embed": _grad_norm([model.traj_embed]),
        "t_embedder": _grad_norm([model.t_embedder]),
        "start_goal": _grad_norm([model.start_goal]),
        "safety_encoder": _grad_norm([model.safety_encoder]),
        "safety_cross_attn": _grad_norm([b.safety_cross_attn for b in blocks]),
        "safety_gates": float(sum(float(b.safety_gate.grad.abs().sum())
                                  for b in blocks if b.safety_gate.grad is not None)),
        "map_condition_encoder": _grad_norm([encoder]),
        "frozen_occ_unet": _grad_norm([pipeline["net"].occ_unet]),
        "frozen_dist_unet": _grad_norm([pipeline["net"].dist_unet]),
    }


def test_three_step_gradient_onset(pipeline, cfg, onset_cache):
    dataset, cache = onset_cache
    model, encoder, trainer = _fresh_trainer(cfg, pipeline)
    items = [dataset[i] for i in range(2)]
    index = np.arange(2)

    # a freshly built model must still be in its documented initial state
    assert torch.count_nonzero(model.head.weight.detach()) == 0
    assert all(torch.count_nonzero(b.adaLN_modulation[-1].bias.detach()) == 0
               for b in model.blocks)
    assert all(abs(float(b.safety_gate.item()) - 0.1) < 1e-6 for b in model.blocks)

    steps = []
    original_step = trainer.optimizer.step

    def snapshotting_step(*args, **kwargs):
        steps.append(_snapshot(model, encoder, pipeline))
        return original_step(*args, **kwargs)

    trainer.optimizer.step = snapshotting_step
    try:
        for _ in range(3):
            trainer.train_step(items, cache, index)
    finally:
        trainer.optimizer.step = original_step
    assert len(steps) == 3

    s1, s2, s3 = steps

    # ---- step 1: only the output head (final_norm + head) trains ------------
    assert s1["output_head"] > 0.0
    assert s1["adaln"] < ZERO, "AdaLN depends on dL/dx_final, which is 0 while head = 0"
    for key in ("self_attn", "map_cross_attn", "mlp", "t_embedder", "start_goal",
                "safety_encoder", "safety_cross_attn", "safety_gates",
                "map_condition_encoder", "traj_embed"):
        assert s1[key] < ZERO, "step 1 must not train %s" % key

    # ---- step 2: AdaLN modulation + the safety branch ----------------------
    assert s2["adaln"] > 0.0
    assert s2["safety_cross_attn"] > 0.0
    assert s2["safety_encoder"] > 0.0
    assert s2["safety_gates"] > 0.0
    # the gated residual branches are still multiplied by gate == 0
    for key in ("self_attn", "map_cross_attn", "mlp", "t_embedder", "start_goal",
                "map_condition_encoder"):
        assert s2[key] < ZERO, "step 2 must still not train %s" % key

    # ---- step 3: Map CA and the MapConditionEncoder ------------------------
    assert s3["map_cross_attn"] > 0.0
    assert s3["map_condition_encoder"] > 0.0
    assert s3["self_attn"] > 0.0
    assert s3["t_embedder"] > 0.0

    # the frozen encoders never receive gradient, at any step
    for step in steps:
        assert step["frozen_occ_unet"] == 0.0
        assert step["frozen_dist_unet"] == 0.0
    assert all(param.grad is None for param in pipeline["net"].parameters())

    # the gate really did open: after step 2 the AdaLN bias is non-zero
    assert float(model.blocks[0].adaLN_modulation[-1].bias.detach().abs().sum()) > 0.0
    assert float(model.head.weight.detach().abs().sum()) > 0.0
