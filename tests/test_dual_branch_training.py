"""Dual-branch joint training tests (design document section 13.2)."""
from __future__ import annotations

import numpy as np
import torch

from gp_diffuser.training.dataset import SyntheticTrajectoryDataset
from gp_diffuser.training.dual_branch_trainer import DualBranchTrainer
from gp_diffuser.training.safety_cache import build_safety_cache, load_safety_cache

CACHE_SPLIT = "train_pytest_dual"


def _trainer(pipeline, cfg):
    """Own DiT + MapConditionEncoder + optimizer (no shared-state pollution)."""
    from gp_diffuser.factory import (build_codec, build_map_condition_encoder,
                                     build_model, build_schedule)
    import torch
    torch.manual_seed(0)
    model = build_model(cfg)
    encoder = build_map_condition_encoder(cfg)
    return DualBranchTrainer(cfg, model, encoder, pipeline["map_encoder"],
                             build_codec(cfg), build_schedule(cfg), device="cpu",
                             scene_scale=pipeline["transform"].scale)


def _cache(pipeline, cfg, dataset):
    build_safety_cache(dataset, pipeline["net"], pipeline["safety_generator"],
                       pipeline["guidance"], cfg, split=CACHE_SPLIT, verbose=False)
    return load_safety_cache(cfg, CACHE_SPLIT)


def test_both_branches_share_q_t_and_timestep(pipeline, cfg):
    dataset = SyntheticTrajectoryDataset(cfg, "test")
    cache = _cache(pipeline, cfg, dataset)
    trainer = _trainer(pipeline, cfg)
    items = [dataset[i] for i in range(len(dataset))]

    seen = []
    original = type(pipeline["model"]).forward

    def spy(self, *args, **kwargs):
        seen.append((kwargs["q_t"].clone(), kwargs["t"].clone()))
        return original(self, *args, **kwargs)

    type(pipeline["model"]).forward = spy
    try:
        trainer.train_step(items, cache, np.arange(len(items)))
    finally:
        type(pipeline["model"]).forward = original
    assert len(seen) == 2, "expected exactly two forwards (map-only + safety)"
    assert torch.equal(seen[0][0], seen[1][0]), "the two branches must share q_t"
    assert torch.equal(seen[0][1], seen[1][1]), "the two branches must share t"


def test_only_one_backward_and_one_update(pipeline, cfg):
    dataset = SyntheticTrajectoryDataset(cfg, "test")
    cache = _cache(pipeline, cfg, dataset)
    trainer = _trainer(pipeline, cfg)
    items = [dataset[i] for i in range(len(dataset))]

    counts = {"backward": 0, "step": 0}
    original_backward = torch.Tensor.backward
    original_step = trainer.optimizer.step

    def backward(self, *args, **kwargs):
        counts["backward"] += 1
        return original_backward(self, *args, **kwargs)

    def step(*args, **kwargs):
        counts["step"] += 1
        return original_step(*args, **kwargs)

    torch.Tensor.backward = backward
    trainer.optimizer.step = step
    try:
        stats = trainer.train_step(items, cache, np.arange(len(items)))
    finally:
        torch.Tensor.backward = original_backward
        trainer.optimizer.step = original_step
    assert counts == {"backward": 1, "step": 1}
    assert stats["loss"] > 0


def test_joint_loss_is_the_documented_weighted_sum(pipeline, cfg):
    dataset = SyntheticTrajectoryDataset(cfg, "test")
    cache = _cache(pipeline, cfg, dataset)
    trainer = _trainer(pipeline, cfg)
    items = [dataset[i] for i in range(len(dataset))]
    stats = trainer.train_step(items, cache, np.arange(len(items)))
    w = trainer.weights
    expected = (w["lambda_c"] * stats["loss_c"] + w["lambda_u"] * stats["loss_u"]
                + w["lambda_align"] * stats["loss_align"]
                + w["lambda_boundary"] * stats["loss_boundary"]
                + w["lambda_smooth"] * stats["loss_smooth"]
                + w["lambda_curve"] * stats["loss_curve"]
                + w["lambda_safe"] * stats["loss_safe"])
    assert abs(stats["loss"] - expected) < 1e-4


def test_alignment_target_is_stop_gradient(pipeline, cfg):
    model = pipeline["model"]
    pred_u = torch.randn(2, 32, 2, requires_grad=True)
    pred_c = torch.randn(2, 32, 2, requires_grad=True)
    from gp_diffuser.training.losses import alignment_loss
    loss = alignment_loss(pred_u, pred_c).mean()
    loss.backward()
    assert pred_c.grad is None, "the safety branch must be a stop-gradient target"
    assert pred_u.grad is not None
    # the descent direction (negative gradient) points towards the safety branch
    toward_target = (pred_c - pred_u).detach()
    assert (pred_u.grad * toward_target).sum() < 0, \
        "gradient must pull pred_u towards the stop-grad safety prediction"


def test_frozen_modules_get_no_gradient(pipeline, cfg):
    dataset = SyntheticTrajectoryDataset(cfg, "test")
    cache = _cache(pipeline, cfg, dataset)
    trainer = _trainer(pipeline, cfg)
    items = [dataset[i] for i in range(len(dataset))]
    trainer.train_step(items, cache, np.arange(len(items)))
    assert all(p.grad is None for p in trainer.map_encoder.encoder_parameters())
    assert all(p.grad is None for p in pipeline["net"].parameters())
    assert any(p.grad is not None for p in trainer.model.parameters())


def test_no_phased_training_switches(pipeline, cfg):
    """There is no warm-up / safety-activation flag anywhere in the trainer."""
    import inspect
    source = inspect.getsource(DualBranchTrainer)
    for banned in ("warmup_steps", "activation_delay", "base_only", "safety_activation",
                   "stage_", "phase"):
        assert banned not in source
    dataset = SyntheticTrajectoryDataset(cfg, "test")
    cache = _cache(pipeline, cfg, dataset)
    trainer = _trainer(pipeline, cfg)
    items = [dataset[i] for i in range(len(dataset))]
    trainer.train_step(items, cache, np.arange(len(items)))
    # both branches saw a GT loss in the very first iteration
    assert trainer.history[-1]["loss_u"] > 0
    assert trainer.history[-1]["loss_c"] > 0
