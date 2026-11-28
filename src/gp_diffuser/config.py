"""Config loading and path resolution.

One YAML file (``configs/base.yaml``) carries every tunable; helpers here give
dotted access and a stable set of defaults so a partial config still runs.
"""
from __future__ import annotations

import copy
import os
from typing import Any, Dict

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "configs", "base.yaml")


def project_path(*parts: str) -> str:
    """Resolve ``parts`` against the project root."""
    return os.path.join(PROJECT_ROOT, *parts)


def load_config(path: str | None = None) -> Dict[str, Any]:
    """Load a YAML config and fill in the defaults from ``configs/base.yaml``."""
    base: Dict[str, Any] = {}
    if os.path.exists(DEFAULT_CONFIG):
        with open(DEFAULT_CONFIG, "r", encoding="utf-8") as fh:
            base = yaml.safe_load(fh) or {}
    if path is None:
        return base
    with open(path, "r", encoding="utf-8") as fh:
        override = yaml.safe_load(fh) or {}
    return deep_update(copy.deepcopy(base), override)


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def cfg_get(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    """``cfg_get(cfg, "diffusion.hidden_dim", 384)``."""
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def resolve_checkpoint(cfg: Dict[str, Any]) -> str:
    """Absolute path of the PathSafe checkpoint, searching the references tree."""
    rel = cfg_get(cfg, "map_encoder.checkpoint", "")
    candidates = [
        rel,
        project_path(rel) if rel else "",
        project_path("references", "PathSafe-Ellipse", "checkpoints", "v3_best.pt"),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(
        "PathSafe-EllipseNet checkpoint not found; expected "
        "references/PathSafe-Ellipse/checkpoints/v3_best.pt (config: "
        "map_encoder.checkpoint)"
    )
