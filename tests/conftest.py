"""Shared pytest fixtures: project import path, config and cached pipeline."""
from __future__ import annotations

import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(PROJECT_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


@pytest.fixture(scope="session")
def project_root() -> str:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def cfg():
    from gp_diffuser.config import load_config
    config = load_config(os.path.join(PROJECT_ROOT, "configs", "base.yaml"))
    config["dataset"]["train_size"] = 2
    config["dataset"]["val_size"] = 1
    config["dataset"]["test_size"] = 2
    return config


@pytest.fixture(scope="session")
def pipeline(cfg):
    from gp_diffuser.factory import build_pipeline
    return build_pipeline(cfg, device="cpu")


@pytest.fixture(scope="session")
def sample(cfg):
    from gp_diffuser.training.dataset import SyntheticTrajectoryDataset
    dataset = SyntheticTrajectoryDataset(cfg, "test")
    return dataset[0], dataset


@pytest.fixture
def fresh_models(cfg):
    """Fresh Conditional DiT + MapConditionEncoder (function scoped).

    Tests that run ``optimizer.step`` or edit parameters MUST use this instead of
    the session-scoped ``pipeline``, otherwise they leak trained weights into every
    later test (order-dependent results).  Only the frozen map encoder is shared.
    """
    import torch
    from gp_diffuser.factory import build_map_condition_encoder, build_model
    torch.manual_seed(0)
    return build_model(cfg), build_map_condition_encoder(cfg)


@pytest.fixture
def fresh_trainer(cfg, pipeline, fresh_models):
    """A trainer with its own DiT / MapConditionEncoder / optimizer."""
    from gp_diffuser.factory import build_codec, build_schedule
    from gp_diffuser.training.dual_branch_trainer import DualBranchTrainer
    model, encoder = fresh_models
    return DualBranchTrainer(cfg, model, encoder, pipeline["map_encoder"],
                             build_codec(cfg), build_schedule(cfg), device="cpu",
                             scene_scale=pipeline["transform"].scale)


@pytest.fixture(autouse=True)
def _guard_shared_pipeline_state(request):
    """Fail if a test leaves the session-scoped model modified.

    The session ``pipeline`` must stay read-only: anything that runs
    ``optimizer.step`` or edits parameters has to use ``fresh_models`` /
    ``fresh_trainer``.  Only three tiny tensors are checked (head weight, block-0
    AdaLN bias, block-0 safety gate) so the guard costs microseconds.
    """
    if "pipeline" not in request.fixturenames:
        yield
        return
    import torch
    pipeline = request.getfixturevalue("pipeline")
    model = pipeline["model"]
    head = model.head.weight.detach().clone()
    adaln = model.blocks[0].adaLN_modulation[-1].bias.detach().clone()
    gate = model.blocks[0].safety_gate.detach().clone()
    yield
    if not torch.equal(model.head.weight.detach(), head):
        raise AssertionError("%s modified the shared pipeline head; use fresh_models"
                             % request.node.name)
    if not torch.equal(model.blocks[0].adaLN_modulation[-1].bias.detach(), adaln):
        raise AssertionError("%s modified the shared pipeline AdaLN; use fresh_models"
                             % request.node.name)
    if not torch.equal(model.blocks[0].safety_gate.detach(), gate):
        raise AssertionError("%s modified the shared pipeline safety gate; use fresh_models"
                             % request.node.name)
