"""Assemble the complete GeoPath-Diffuser pipeline from one config dict.

Everything is constructed once and shared: the frozen PathSafe-EllipseNet
instance backs BOTH the map encoder and the safety generator, so "the map is
encoded once" is structural rather than a convention.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import torch

from .config import cfg_get
from .diffusion.sampler_conditional_dit import ConditionalDiTSampler
from .diffusion.schedule import NoiseSchedule
from .frames import SceneTransform
from .guidance.search import GuidanceFieldSearch
from .map_encoder.frozen_encoder import FrozenMapEncoder
from .models.conditional_dit import ConditionalDiT
from .models.condition_encoders import MapFeatureProjection
from .safety.safety_features import (PathSafeEllipseNet, SafetyConditionGenerator,
                                     load_pathsafe)
from .trajectory.bspline import BSplineCodec


def build_transform(cfg: dict) -> SceneTransform:
    res_m = float(cfg_get(cfg, "dataset.res_m", 0.25))
    grid = int(cfg_get(cfg, "dataset.grid", 256))
    return SceneTransform(res_m * grid, grid)


def build_codec(cfg: dict) -> BSplineCodec:
    return BSplineCodec(degree=int(cfg_get(cfg, "trajectory.degree", 3)),
                        num_controls=int(cfg_get(cfg, "trajectory.num_controls", 32)),
                        curve_points=int(cfg_get(cfg, "trajectory.curve_points", 128)))


def build_schedule(cfg: dict) -> NoiseSchedule:
    return NoiseSchedule(int(cfg_get(cfg, "diffusion.train_diffusion_steps", 1000)),
                         beta_schedule=str(cfg_get(cfg, "diffusion.beta_schedule",
                                                  "squaredcos_cap_v2")))


def build_model(cfg: dict) -> ConditionalDiT:
    return ConditionalDiT(
        hidden_dim=int(cfg_get(cfg, "diffusion.hidden_dim", 384)),
        depth=int(cfg_get(cfg, "diffusion.depth", 12)),
        num_heads=int(cfg_get(cfg, "diffusion.num_heads", 6)),
        mlp_ratio=float(cfg_get(cfg, "diffusion.mlp_ratio", 4)),
        num_controls=int(cfg_get(cfg, "trajectory.num_controls", 32)),
        path_points=int(cfg_get(cfg, "safety_condition.path_points", 128)),
        num_ellipses=int(cfg_get(cfg, "safety_condition.num_ellipses", 64)),
        ellipse_feature_dim=int(cfg_get(cfg, "safety_condition.ellipse_feature_dim", 6)),
        map_grid=int(cfg_get(cfg, "map_condition.grid", 32)),
        map_channels=int(cfg_get(cfg, "map_condition.bottleneck_channels", 192)),
        dropout=float(cfg_get(cfg, "diffusion.dropout", 0.0)))


def build_map_projection(cfg: dict) -> MapFeatureProjection:
    return MapFeatureProjection(
        hidden_dim=int(cfg_get(cfg, "map_condition.projection_dim", 384)),
        occ_channels=int(cfg_get(cfg, "map_condition.bottleneck_channels", 192)),
        sdf_channels=int(cfg_get(cfg, "map_condition.bottleneck_channels", 192)),
        grid=int(cfg_get(cfg, "map_condition.grid", 32)))


def build_pipeline(cfg: dict, device: str = "cpu", with_pretrained: bool = True,
                   checkpoint: Optional[str] = None, seed: int = 0) -> Dict:
    """Construct encoders, safety generator, search, model and sampler."""
    torch.manual_seed(int(seed))
    transform = build_transform(cfg)
    codec = build_codec(cfg)
    # P2-2: the basis / pinv buffers must live on the compute device
    codec.to(device)
    schedule = build_schedule(cfg)
    guidance = GuidanceFieldSearch(cfg.get("guidance") or {},
                                  res_m=float(cfg_get(cfg, "dataset.res_m", 0.25)))

    net = None
    map_encoder = None
    safety_generator = None
    if with_pretrained:
        net = load_pathsafe(cfg, device=device)
        map_encoder = FrozenMapEncoder(net).to(device)
        safety_generator = SafetyConditionGenerator(
            net,
            num_queries=int(cfg_get(cfg, "safety_condition.num_ellipses", 64)),
            res_m=float(cfg_get(cfg, "dataset.res_m", 0.25)))
        safety_generator.to(device).eval()

    model = build_model(cfg).to(device)
    map_projection = build_map_projection(cfg).to(device)
    if checkpoint and os.path.exists(checkpoint):
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        if "map_projection" in payload:
            map_projection.load_state_dict(payload["map_projection"])

    sampler = ConditionalDiTSampler(
        model=model, codec=codec, schedule=schedule, map_projection=map_projection,
        map_encoder=map_encoder, safety_generator=safety_generator,
        guidance_search=guidance, cfg=cfg, device=device, transform=transform)

    return {
        "cfg": cfg,
        "device": device,
        "transform": transform,
        "codec": codec,
        "schedule": schedule,
        "guidance": guidance,
        "net": net,
        "map_encoder": map_encoder,
        "safety_generator": safety_generator,
        "model": model,
        "map_projection": map_projection,
        "sampler": sampler,
    }


def describe(pipeline: Dict) -> Dict:
    """Small summary used by the scripts and the report."""
    model = pipeline["model"]
    out = {
        "diffusion": model.config_dict(),
        "map_encoder_frozen": True,
        "map_encoder_trainable_parameters": (
            pipeline["map_encoder"].trainable_parameter_count()
            if pipeline["map_encoder"] is not None else None),
        "pathsafe_checkpoint": (pipeline["net"].checkpoint_meta
                                if pipeline["net"] is not None else None),
        "safety_generator_trainable_parameters": sum(
            p.numel() for p in pipeline["safety_generator"].parameters()
            if p.requires_grad) if pipeline["safety_generator"] is not None else 0,
        "trajectory": {"num_controls": pipeline["codec"].num_controls,
                       "degree": pipeline["codec"].degree,
                       "curve_points": pipeline["codec"].curve_points},
    }
    return out
