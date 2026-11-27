"""PathSafe-EllipseNet as a frozen safety-feature generator.

Two things are added on top of the migrated network (``safety/psenet.py``):

1. ``encode_maps`` / ``forward_from_features`` - the interface the design
   document asks for (section 10, Task 1).  The network can consume features
   computed ONCE by the frozen map encoder instead of re-running the two
   ResUNets per call.  The forward maths below is a line-by-line mirror of
   ``PSENet.forward``; only the two encoder calls are replaced.
2. ``SafetyConditionGenerator`` - the project-facing adapter that turns a
   128-point search path into 64 ellipses plus the ``[cx,cy,a,b,cos2t,sin2t]``
   feature the diffusion backbone consumes (document section 6.2).

The ellipse heads, the 16-candidate arg-max rule and the geometric losses are
untouched; this module only generates geometry.
"""
from __future__ import annotations

import math
import os
from typing import Dict, Optional

import torch
import torch.nn as nn

from ..config import resolve_checkpoint
from .psenet import PSENet


class PathSafeEllipseNet(PSENet):
    """``PSENet`` + the externally-precomputed-feature entry points.

    Subclassing (instead of editing the migrated file) keeps the upstream class
    byte-for-byte intact: the shipped ``v3_best.pt`` state dict loads with
    ``strict=True`` because no parameter is added or renamed.
    """

    # ------------------------------------------------------------------ encode
    @torch.no_grad()
    def encode_maps(self, occupancy: torch.Tensor, sdf: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Both bottlenecks (for diffusion) and both full features (for PathSafe)."""
        occ_out = self.occ_unet.forward_with_features(occupancy)
        sdf_in = torch.clamp(sdf / float(self.d_clip), -1.0, 1.0)
        sdf_out = self.dist_unet.forward_with_features(sdf_in)
        return {
            "occ_bottleneck": occ_out["bottleneck"],
            "sdf_bottleneck": sdf_out["bottleneck"],
            "occ_full": occ_out["full"],
            "sdf_full": sdf_out["full"],
        }

    # --------------------------------------------------------------- features
    def forward_from_features(self, batch: Dict[str, torch.Tensor],
                              features: Dict[str, torch.Tensor],
                              tau: float = 1.0) -> Dict:
        """Mirror of ``PSENet.forward`` with the encoder calls removed."""
        path = batch["path_xy"]
        lidar = batch["lidar"]
        ray_dir = batch["ray_dir"]
        ray_valid = batch["ray_valid"]
        mask = batch["mask"]
        r_scene = batch["r_scene"]
        F_occ = features["occ_full"]
        F_dist = features["sdf_full"]
        sdf = batch["sdf"]

        B, N, _ = path.shape
        R = self.rays
        c_occ = self._occupancy_tokens(F_occ, path, r_scene)
        c_occ_f = c_occ if not self.ab_occ_point else torch.zeros_like(c_occ)
        R_tok = self._ray_tokens(F_dist, sdf, path, lidar, ray_dir, ray_valid,
                                 r_scene, c_occ_f)
        if self.ab_ray_attn:
            R_hat = R_tok
        else:
            R_hat = self.ray_attn(R_tok.reshape(B * N, R, self.C)).reshape(B, N, R, self.C)
        h_local = self.point_mlp(torch.cat([c_occ_f, R_hat.reshape(B, N, R * self.C)],
                                           dim=-1))
        from .sampling import normalized_arclength, sinusoidal_pe
        s = normalized_arclength(path, mask)
        h = h_local + sinusoidal_pe(s, self.C)
        if self.ab_path_attn:
            H_out = h
        else:
            H_out = self.path_attn(h, src_key_padding_mask=~mask)

        logits = self.dir_head(H_out)
        pi = torch.softmax(logits / max(float(tau), 1e-6), dim=-1)
        alpha = torch.sigmoid(self.ratio_head(H_out).squeeze(-1))
        raw = self.shape_head(H_out)
        a = self.a_min + (self.a_max - self.a_min) * torch.sigmoid(raw[..., 0])
        b = self.b_min + (a - self.b_min) * torch.sigmoid(raw[..., 1])
        theta = torch.atan2(raw[..., 3], raw[..., 2])
        d_eff = torch.clamp(lidar - self.ray_margin_m, min=0.0)
        q = path[:, :, None, :] + (alpha[:, :, None, None] * d_eff[..., None]) \
            * ray_dir[:, None, :, :]
        return {
            "logits": logits, "pi": pi, "alpha": alpha, "a": a, "b": b,
            "theta": theta, "q": q, "h_local": h_local, "h_global": H_out,
            "c_occ": c_occ, "R_hat": R_hat, "path_s": s,
        }

    @torch.no_grad()
    def predict_ellipses(self, batch: Dict[str, torch.Tensor],
                         features: Optional[Dict[str, torch.Tensor]] = None,
                         tau: float = 1.0) -> Dict[str, torch.Tensor]:
        """Ellipse geometry per query point (arg-max ray, never an average)."""
        out = (self.forward_from_features(batch, features, tau=tau)
               if features is not None else self.forward(batch, tau=tau))
        decoded = self.decode_argmax(out)
        theta = decoded["theta"]
        decoded["shape4"] = torch.stack([
            torch.log(decoded["a"].clamp_min(1e-6)),
            torch.log(decoded["b"].clamp_min(1e-6)),
            torch.cos(2.0 * theta), torch.sin(2.0 * theta)], dim=-1)
        return decoded


def load_pathsafe(cfg: dict, device: str = "cpu", dtype=torch.float32) -> PathSafeEllipseNet:
    """Load the trained PathSafe-EllipseNet (V3) weights.

    The checkpoint carries its own ``cfg`` (the exact training config), so the
    architecture is reconstructed from the checkpoint rather than guessed from
    the YAML.  ``v3_best.pt`` is 5,041,093 parameters on ``base=24`` /
    ``ch=128`` / ``depths=[1,1,1,1]``.
    """
    path = resolve_checkpoint(cfg)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "model" not in payload:
        raise KeyError("unexpected checkpoint layout in %s" % path)
    model_cfg = payload.get("cfg")
    if not isinstance(model_cfg, dict) or "model" not in model_cfg:
        raise KeyError("checkpoint %s has no embedded config" % path)
    # the network class comes from the vendored PathSafe-Ellipse package
    from .psenet_vendor import VendoredEllipseNet
    from ..vendor.ellipsenet.models.psenet import PSENet
    core = PSENet(model_cfg)
    missing, unexpected = core.load_state_dict(payload["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "PathSafe checkpoint does not match the network: missing=%s unexpected=%s"
            % (list(missing)[:8], list(unexpected)[:8]))
    core.to(device=device, dtype=dtype)
    core.eval()
    for param in core.parameters():
        param.requires_grad_(False)
    net = VendoredEllipseNet(core).to(device=device, dtype=dtype)
    net.eval()
    for param in net.parameters():
        param.requires_grad_(False)
    net.checkpoint_path = os.path.abspath(path)
    net.checkpoint_meta = {
        "path": os.path.abspath(path),
        "epoch": payload.get("epoch"),
        "step": payload.get("step"),
        "params": sum(p.numel() for p in net.parameters()),
        "implementation": "vendor/ellipsenet (upstream PSENet)",
    }
    return net


class SafetyConditionGenerator(nn.Module):
    """Search path + map features -> ellipses and safety tokens.

    Everything runs under ``no_grad``: the generator is a frozen inference
    module (design document section 9.5 - the safety geometry never receives
    gradient from the diffusion loss).
    """

    def __init__(self, net: PathSafeEllipseNet, num_queries: int = 64,
                 res_m: float = 0.25, max_range_m: float = 9.0):
        super().__init__()
        self.net = net
        self.num_queries = int(num_queries)
        self.res_m = float(res_m)
        self.max_range_m = float(max_range_m)

    @torch.no_grad()
    def forward(self, occupancy: torch.Tensor, sdf: torch.Tensor,
                path_xy: torch.Tensor, path_valid: Optional[torch.Tensor] = None,
                features: Optional[Dict[str, torch.Tensor]] = None,
                ray_features: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """``path_xy [B,128,2]`` (metres) -> ellipses at ``num_queries`` stations.

        The ``num_queries`` stations are taken by arc length along the search
        path, which is the same path the diffusion conditions see.
        """
        from .psenet_inputs import cast_rays
        import numpy as np

        B = path_xy.shape[0]
        device = path_xy.device
        query = resample_arclength(path_xy, self.num_queries)      # [B,Q,2]
        if path_valid is None:
            path_valid = torch.ones(B, self.num_queries, dtype=torch.bool, device=device)

        if ray_features is None:
            lids, rvalids = [], []
            dirs = None
            for b in range(B):
                occ_np = occupancy[b, 0].detach().cpu().numpy()
                ray = cast_rays(occ_np, query[b].detach().cpu().numpy(),
                                self.res_m, max_range_m=self.max_range_m)
                lids.append(ray["lidar"])
                rvalids.append(ray["ray_valid"])
                dirs = ray["ray_dir"]
            ray_features = {
                "lidar": torch.stack(lids).to(device),
                "ray_dir": dirs.to(device).unsqueeze(0).expand(B, -1, -1),
                "ray_valid": torch.stack(rvalids).to(device),
            }

        # upstream `r_scene` is the CELL SIZE in metres (their world_to_grid
        # computes col = x / r_scene then normalises by the grid width)
        r_scene = float(self.res_m)
        batch = {
            "occupancy": occupancy,
            "sdf": sdf,
            "path_xy": query,
            "lidar": ray_features["lidar"].to(device),
            "ray_dir": ray_features["ray_dir"].to(device),
            "ray_valid": ray_features["ray_valid"].to(device),
            "mask": path_valid,
            "r_scene": torch.full((B,), r_scene, device=device),
        }
        out = self.net.predict_ellipses(batch, features=features)
        theta = out["theta"]
        ellipse = torch.stack([
            out["center"][..., 0], out["center"][..., 1],
            out["a"], out["b"],
            torch.cos(2.0 * theta), torch.sin(2.0 * theta),
        ], dim=-1)
        return {
            "ellipse": ellipse,                 # [B,Q,6]
            "ellipse_center": out["center"],    # [B,Q,2]
            "ellipse_shape4": out["shape4"],    # [B,Q,4]
            "ellipse_prob": out["prob"],
            "path": path_xy,                    # [B,128,2]
            "query": query,                     # [B,Q,2]
            "valid": path_valid,
            "raw": out,
        }



def ellipse_shape4_to_scene(shape4_m: torch.Tensor, scene_scale: float) -> torch.Tensor:
    """Metre ``[log a_m, log b_m, cos2t, sin2t]`` -> scene units.

    The convex-region builder (``convex_region.ellipse_shape4_to_quadratic``)
    exponentiates the first two channels, so they must already be log-axes in the
    normalised scene frame.  Dividing the axes by ``scene_scale`` is exactly a
    constant shift of the log-axes, so this is done with a subtraction (no
    exp/log round trip, therefore no clamping loss).

    Code review P0-2: the first implementation converted the ellipse centres and
    the token axes but passed the raw metre ``shape4`` to the region builder,
    which made every ellipse 32x too large in scene coordinates.
    """
    if scene_scale <= 0:
        raise ValueError("scene_scale must be positive")
    out = shape4_m.clone()
    out[..., 0:2] = shape4_m[..., 0:2] - math.log(float(scene_scale))
    return out

def resample_arclength(points: torch.Tensor, count: int) -> torch.Tensor:
    """Uniform arc-length resampling of ``[B,N,2]`` polylines to ``[B,count,2]``."""
    B, N, _ = points.shape
    if N == count:
        return points
    seg = torch.linalg.norm(points[:, 1:] - points[:, :-1], dim=-1)
    cum = torch.cat([torch.zeros(B, 1, device=points.device, dtype=points.dtype),
                     torch.cumsum(seg, dim=1)], dim=1)
    total = cum[:, -1:].clamp_min(1e-9)
    target = torch.linspace(0.0, 1.0, count, device=points.device,
                            dtype=points.dtype)[None] * total
    idx = torch.searchsorted(cum.contiguous(), target.contiguous(), right=True) - 1
    idx = idx.clamp(0, N - 2)
    c0 = torch.gather(cum, 1, idx)
    c1 = torch.gather(cum, 1, idx + 1)
    frac = ((target - c0) / (c1 - c0).clamp_min(1e-9)).clamp(0.0, 1.0)
    p0 = torch.gather(points, 1, idx[..., None].expand(-1, -1, 2))
    p1 = torch.gather(points, 1, (idx + 1)[..., None].expand(-1, -1, 2))
    out = p0 + frac[..., None] * (p1 - p0)
    # exact endpoints (search paths start/end on S/G)
    out[:, 0] = points[:, 0]
    out[:, -1] = points[:, -1]
    return out
