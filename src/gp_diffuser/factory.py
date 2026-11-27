"""Assemble the complete GeoPath-Diffuser pipeline from one config dict.

Everything is constructed once and shared: the frozen PathSafe-EllipseNet
instance backs BOTH the map encoder and the safety generator, so "the map is
encoded once" is structural rather than a convention.

Condition network (spec: 网络结构修改实施方案 §2 / §5):

    FrozenMapEncoder (frozen)  ->  occ/sdf bottleneck
    MapConditionEncoder (trainable)  ->  map memory [B,1024,384]
    ConditionalDiT.build_map_context(map_memory, start_goal, ...)  ->  [B,1026,384]
    ConditionalDiT.build_safety_memory(ellipse, valid)             ->  [B,64,384]
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import torch

from .config import cfg_get
from .diffusion.sampler_conditional_dit import ConditionalDiTSampler
from .diffusion.schedule import NoiseSchedule
from .frames import SceneTransform
from .guidance.reference_search import ReferenceGuidanceSearch
from .map_encoder.frozen_encoder import FrozenMapEncoder
from .models.conditional_dit import ARCHITECTURE_VERSION, ConditionalDiT
from .models.condition_encoders import MapConditionEncoder
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
        num_ellipses=int(cfg_get(cfg, "safety_condition.num_ellipses", 64)),
        ellipse_feature_dim=int(cfg_get(cfg, "safety_condition.ellipse_feature_dim", 6)),
        map_grid=int(cfg_get(cfg, "map_condition.grid", 32)),
        map_channels=int(cfg_get(cfg, "map_condition.bottleneck_channels", 192)),
        safety_gate_init=float(cfg_get(cfg, "diffusion.safety_residual_gate_init", 0.1)),
        dropout=float(cfg_get(cfg, "diffusion.dropout", 0.0)))


def build_map_condition_encoder(cfg: dict) -> MapConditionEncoder:
    """Dual-adapter + residual spatial fusion encoder (trainable)."""
    encoder_type = str(cfg_get(cfg, "map_condition.encoder_type",
                              "dual_adapter_residual_fusion"))
    if encoder_type != "dual_adapter_residual_fusion":
        raise ValueError("unsupported map_condition.encoder_type %r" % encoder_type)
    return MapConditionEncoder(
        hidden_dim=int(cfg_get(cfg, "map_condition.projection_dim", 384)),
        bottleneck_channels=int(cfg_get(cfg, "map_condition.bottleneck_channels", 192)),
        adapter_channels=int(cfg_get(cfg, "map_condition.adapter_channels", 192)),
        grid=int(cfg_get(cfg, "map_condition.grid", 32)),
        groups=int(cfg_get(cfg, "map_condition.adapter_groups", 16)),
        kernel_size=int(cfg_get(cfg, "map_condition.spatial_fusion.kernel_size", 3)))


def load_checkpoint(path: str, model: ConditionalDiT, map_condition_encoder,
                    device: str = "cpu") -> Dict:
    """Load a checkpoint, refusing anything not built by this architecture.

    The condition network changed (dual map adapters + separate Map/Safety cross
    attention), so an older checkpoint is incompatible: the error says so instead
    of silently ignoring missing parameters (spec §8).
    """
    payload = torch.load(path, map_location=device, weights_only=False)
    version = payload.get("architecture_version")
    if version is None:
        version = (payload.get("config") or {}).get("architecture_version")
    if version != ARCHITECTURE_VERSION:
        raise RuntimeError(
            "checkpoint '%s' was produced by architecture %r, but this code implements "
            "%r (dual map adapters + separate Map/Safety cross-attention).  The "
            "parameter sets are incompatible: retrain instead of loading it."
            % (path, version, ARCHITECTURE_VERSION))
    model.load_state_dict(payload["model"])
    encoder_state = payload.get("map_condition_encoder") or payload.get("map_projection")
    if encoder_state is None:
        raise RuntimeError("checkpoint '%s' has no MapConditionEncoder state" % path)
    map_condition_encoder.load_state_dict(encoder_state)
    return payload


def build_pipeline(cfg: dict, device: str = "cpu", with_pretrained: bool = True,
                   checkpoint: Optional[str] = None, seed: int = 0) -> Dict:
    """Construct encoders, safety generator, search, model and sampler."""
    torch.manual_seed(int(seed))
    transform = build_transform(cfg)
    codec = build_codec(cfg)
    # P2-2: the basis / pinv buffers must live on the compute device
    codec.to(device)
    schedule = build_schedule(cfg)
    # P2-2: q_sample indexes the schedule tables with the timestep tensor, so
    # they must live on the same device as t
    schedule.to(device)
    guidance = ReferenceGuidanceSearch(cfg.get("guidance") or {},
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
    map_condition_encoder = build_map_condition_encoder(cfg).to(device)
    if checkpoint and os.path.exists(checkpoint):
        load_checkpoint(checkpoint, model, map_condition_encoder, device=device)

    sampler = ConditionalDiTSampler(
        model=model, codec=codec, schedule=schedule,
        map_condition_encoder=map_condition_encoder,
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
        "map_condition_encoder": map_condition_encoder,
        "sampler": sampler,
    }


def describe(pipeline: Dict) -> Dict:
    """Small summary used by the scripts and the report (spec §11.2)."""
    model = pipeline["model"]
    encoder = pipeline["map_condition_encoder"]
    trainable = (sum(p.numel() for p in model.parameters() if p.requires_grad)
                 + sum(p.numel() for p in encoder.parameters() if p.requires_grad))
    return {
        "architecture_version": ARCHITECTURE_VERSION,
        "diffusion": model.config_dict(),
        "map_encoder_frozen": True,
        "map_encoder_parameters": (
            sum(p.numel() for p in pipeline["map_encoder"].encoder_parameters())
            if pipeline["map_encoder"] is not None else None),
        "map_encoder_trainable_parameters": (
            pipeline["map_encoder"].trainable_parameter_count()
            if pipeline["map_encoder"] is not None else None),
        "pathsafe_checkpoint": (pipeline["net"].checkpoint_meta
                                if pipeline["net"] is not None else None),
        "safety_generator_trainable_parameters": sum(
            p.numel() for p in pipeline["safety_generator"].parameters()
            if p.requires_grad) if pipeline["safety_generator"] is not None else 0,
        "map_condition_encoder": encoder.parameter_breakdown(),
        "safety_modules": model.safety_parameter_breakdown(),
        "trainable_parameters_total": int(trainable),
        "trajectory": {"num_controls": pipeline["codec"].num_controls,
                       "degree": pipeline["codec"].degree,
                       "curve_points": pipeline["codec"].curve_points},
    }
