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

def test_map_only_arm_never_touches_the_safety_chain(pipeline, cfg, fresh_models):
    """``use_safety=False`` is the ablation arm: one map-only forward, no safety."""
    from gp_diffuser.factory import build_codec, build_schedule
    model, encoder = fresh_models
    trainer = DualBranchTrainer(cfg, model, encoder, pipeline["map_encoder"],
                                build_codec(cfg), build_schedule(cfg), device="cpu",
                                scene_scale=pipeline["transform"].scale,
                                use_safety=False)
    dataset = SyntheticTrajectoryDataset(cfg, "test")
    items = [dataset[i] for i in range(len(dataset))]

    seen = []
    original = type(model).forward

    def spy(self, *args, **kwargs):
        seen.append(kwargs.get("safety_memory"))
        return original(self, *args, **kwargs)

    type(model).forward = spy
    try:
        stats = trainer.train_step(items, None, np.arange(len(items)))
    finally:
        type(model).forward = original

    assert seen == [None], "the ablation arm must run exactly one map-only forward"
    assert stats["loss_c"] == 0.0 and stats["loss_align"] == 0.0
    assert stats["loss_safe"] == 0.0
    assert all(p.grad is None for p in model.safety_encoder.parameters())
    assert all(b.safety_gate.grad is None for b in model.blocks)
    assert all(p.grad is None for b in model.blocks
               for p in b.safety_cross_attn.parameters())
    # AdaLN-Zero: on step 1 only the output head moves (every gate is still 0),
    # which is exactly what the dual arm shows too -- the map-only path IS trained
    assert float(model.head.weight.grad.abs().sum()) > 0

def test_u_target_safety_removes_the_gt_anchor(pipeline, cfg, fresh_models):
    """``u_target='safety'``: the map-only branch is distilled, never regressed on GT."""
    from copy import deepcopy
    from gp_diffuser.factory import build_codec, build_schedule
    model, encoder = fresh_models
    run_cfg = deepcopy(cfg)
    run_cfg.setdefault("training", {})["u_target"] = "safety"
    run_cfg["training"]["align_space"] = "raw"
    trainer = DualBranchTrainer(run_cfg, model, encoder, pipeline["map_encoder"],
                                build_codec(run_cfg), build_schedule(run_cfg),
                                device="cpu", scene_scale=pipeline["transform"].scale)
    assert trainer.u_target == "safety" and trainer.align_space == "raw"
    dataset = SyntheticTrajectoryDataset(run_cfg, "test")
    cache = _cache(pipeline, run_cfg, dataset)
    items = [dataset[i] for i in range(len(dataset))]
    stats = trainer.train_step(items, cache, np.arange(len(items)))
    assert stats["loss_u"] == 0.0, "no GT term may act on the map-only branch"
    # step 1 cannot show a gap yet: the DiT output head is zero-initialised, so
    # both branches emit exactly the same (zero) polygon until it starts moving
    stats = trainer.train_step(items, cache, np.arange(len(items)))
    assert stats["loss_align"] > 0.0
    assert stats["loss_u"] == 0.0
    assert float(model.head.weight.grad.abs().sum()) > 0


def test_align_space_raw_ignores_the_endpoint_modulation(pipeline, cfg, fresh_models):
    """The alignment must be computed on the FREE polygons, not the fixed-up ones."""
    from gp_diffuser.factory import build_codec, build_schedule
    model, encoder = fresh_models
    trainer = DualBranchTrainer(cfg, model, encoder, pipeline["map_encoder"],
                                build_codec(cfg), build_schedule(cfg), device="cpu",
                                scene_scale=pipeline["transform"].scale)
    assert trainer.align_space == "raw"
    seen = []
    original = type(trainer.boundary).forward

    def spy(self, controls, cond, heading=None, heading_valid=None):
        out = original(self, controls, cond, heading, heading_valid)
        seen.append((controls.detach().clone(), out.detach().clone()))
        return out

    type(trainer.boundary).forward = spy
    dataset = SyntheticTrajectoryDataset(cfg, "test")
    cache = _cache(pipeline, cfg, dataset)
    items = [dataset[i] for i in range(len(dataset))]
    try:
        stats = trainer.train_step(items, cache, np.arange(len(items)))
    finally:
        type(trainer.boundary).forward = original
    raw_u, fixed_u = seen[0]
    # the two polygons must really differ, otherwise this test proves nothing
    assert not torch.allclose(raw_u, fixed_u)
    from gp_diffuser.training.losses import control_x0_loss
    expected = float(control_x0_loss(seen[0][0], seen[1][0]).mean())
    assert abs(stats["loss_align"] - expected) < 1e-5


def test_privileged_corridor_keeps_the_map_only_branch_safety_free(pipeline, cfg,
                                                                  fresh_models):
    """A corridor term on the map-only branch is a LABEL: its forward stays map-only."""
    from copy import deepcopy
    from gp_diffuser.factory import build_codec, build_schedule
    model, encoder = fresh_models
    run_cfg = deepcopy(cfg)
    run_cfg.setdefault("loss_weights", {})["lambda_safe_u"] = 0.15
    trainer = DualBranchTrainer(run_cfg, model, encoder, pipeline["map_encoder"],
                                build_codec(run_cfg), build_schedule(run_cfg),
                                device="cpu", scene_scale=pipeline["transform"].scale)
    dataset = SyntheticTrajectoryDataset(run_cfg, "test")
    cache = _cache(pipeline, run_cfg, dataset)
    items = [dataset[i] for i in range(len(dataset))]

    seen = []
    original = type(model).forward

    def spy(self, *args, **kwargs):
        seen.append(kwargs.get("safety_memory"))
        return original(self, *args, **kwargs)

    type(model).forward = spy
    try:
        stats = trainer.train_step(items, cache, np.arange(len(items)))
    finally:
        type(model).forward = original
    assert seen[0] is None, "the map-only forward must not receive ellipse tokens"
    assert seen[1] is not None, "the safety forward still does"
    assert stats["loss_safe_u"] >= 0.0


# --------------------------------------------------------------------------- #
# non-finite guard (a real divergence: one loss spike at step 215650 turned the
# run into NaN from step 215700 on, and ``last.pt`` was overwritten with 261
# non-finite tensors - gradient clipping cannot repair a NaN)
# --------------------------------------------------------------------------- #
def test_non_finite_loss_skips_the_optimizer_step(pipeline, cfg):
    dataset = SyntheticTrajectoryDataset(cfg, "test")
    cache = _cache(pipeline, cfg, dataset)
    trainer = _trainer(pipeline, cfg)
    items = [dataset[i] for i in range(len(dataset))]

    before = [p.detach().clone() for p in trainer.trainable_parameters]
    original = trainer._compute

    def poisoned(*args, **kwargs):
        loss, stats = original(*args, **kwargs)
        return loss * float("nan"), stats

    trainer._compute = poisoned
    try:
        stats = trainer.train_step(items, cache, np.arange(len(items)))
    finally:
        trainer._compute = original

    assert stats["skipped"] == 1.0
    assert trainer.skipped_steps == 1
    for old, new in zip(before, trainer.trainable_parameters):
        assert torch.equal(old, new), "a non-finite loss must not update the weights"
    assert trainer.weights_are_finite()


def test_non_finite_gradient_skips_the_optimizer_step(pipeline, cfg):
    dataset = SyntheticTrajectoryDataset(cfg, "test")
    cache = _cache(pipeline, cfg, dataset)
    trainer = _trainer(pipeline, cfg)
    items = [dataset[i] for i in range(len(dataset))]

    before = [p.detach().clone() for p in trainer.trainable_parameters]
    original_step = trainer.optimizer.step

    def must_not_run(*args, **kwargs):
        raise AssertionError("optimizer.step must not be reached with a NaN gradient")

    handle = trainer.model.head.weight.register_hook(
        lambda g: torch.full_like(g, float("nan")))
    trainer.optimizer.step = must_not_run
    try:
        stats = trainer.train_step(items, cache, np.arange(len(items)))
    finally:
        handle.remove()
        trainer.optimizer.step = original_step

    assert stats["skipped"] == 1.0
    for old, new in zip(before, trainer.trainable_parameters):
        assert torch.equal(old, new)
    assert trainer.weights_are_finite()


def test_checkpoint_refuses_non_finite_weights(pipeline, cfg, tmp_path):
    dataset = SyntheticTrajectoryDataset(cfg, "test")
    trainer = _trainer(pipeline, cfg)
    path = str(tmp_path / "should_not_exist.pt")
    trainer.model.head.weight.data[0, 0] = float("nan")
    assert trainer.weights_are_finite() is False
    assert trainer.save_checkpoint(path, epoch=1, steps=1) == ""
    import os
    assert not os.path.exists(path), "a NaN checkpoint must never be written"
