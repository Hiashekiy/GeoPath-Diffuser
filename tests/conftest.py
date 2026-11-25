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
    config = load_config(os.path.join(PROJECT_ROOT, "configs", "default.yaml"))
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
