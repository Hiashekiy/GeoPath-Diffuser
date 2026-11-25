"""Shared bootstrap for the command-line scripts (``src`` on ``sys.path``)."""
from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(PROJECT_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import argparse  # noqa: E402


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs", "default.yaml"),
                        help="YAML config (defaults are merged in)")
    parser.add_argument("--device", default="cpu", help="torch device, e.g. cpu or cuda")
    return parser


def load(args):
    from gp_diffuser.config import load_config
    return load_config(args.config)


def data_batch(dataset, indices):
    """Stack dataset items by key (list of dicts -> dict of batched tensors)."""
    import torch
    items = [dataset[int(i)] for i in indices]
    keys = items[0].keys()
    return items, {k: torch.stack([it[k] for it in items]) for k in keys}
