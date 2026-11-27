"""Conditional-DiT DDIM sampler with one-shot mid-time safety injection.

Implements the inference flow of the design document section 8:

    Step 0  frozen map encoder, once  -> map memory (cached for the whole run)
    Step 1  map-only DDIM forwards    -> initial global guide trajectory
    Step 2  at ONE mid-time forward: guided search (Astar-Diffusion) ->
            PathSafe-EllipseNet ellipses -> local convex corridor + constraint
            pack.  Generated once, then frozen (never rebuilt per step).
    Step 3  safety-conditioned forwards; ALM corrects the current clean
            controls against the FROZEN pack before each DDIM update
    Step 4  dense validation of the final trajectory

There is no history feedback anywhere: the ALM output only ever enters the
current DDIM state update, never the network input.

Post-review fixes (``docs/code_review_verification.md``):

* **P0-2** the region builder receives ``shape4`` converted to scene units;
* **P0-3/P0-4** per-sample tables (bridges kept, own anchors) are packed and merged;
* **P1-1** collision detection is segment-level (no missed thin walls);
* **P1-3** the network prediction is endpoint-locked before decoding / ALM;
* **P1-2** ALM is skipped for samples without a LOCAL convex cell;
* **P1-4** a failed search no longer turns the guide trajectory into a "path";
* **P1-5** a correction is only accepted when it does not make things worse;
* **P2-1** start/goal heading errors are measured and reported;
* **P2-2** headings and the constraint pack follow the compute device.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..frames import SceneTransform
from ..models.boundary import BoundaryDecoder
from ..safety.collision_query import curve_collision_intervals
from ..safety.local_regions import pack_from_tables
from ..safety.safety_features import ellipse_shape4_to_scene
from .schedule import NoiseSchedule


def pick_times(num_timesteps: int, steps: Optional[int]):
    """Uniform sub-sampling of the training timesteps (upstream rule)."""
    if steps is None or steps >= num_timesteps:
        return None
    idx = torch.linspace(0, num_timesteps - 1, steps).long().tolist()
    times = sorted(set(int(v) for v in idx))
    if num_timesteps - 1 not in times:
        times.append(num_timesteps - 1)
    if 0 not in times:
        times.append(0)
    return sorted(times)


class ConditionalDiTSampler:
    """DDIM sampler around the Conditional DiT with frozen safety geometry."""

    def __init__(self, model, codec, schedule: NoiseSchedule, map_condition_encoder,
                 map_encoder, safety_generator, guidance_search, cfg: dict,
                 device: str = "cpu", transform: Optional[SceneTransform] = None,
                 scene_scale: float = 32.0):
        self.model = model
        self.codec = codec
        self.schedule = schedule
        self.map_condition_encoder = map_condition_encoder
        self.map_encoder = map_encoder
        self.safety_generator = safety_generator
        self.guidance = guidance_search
        self.cfg = dict(cfg or {})
        self.device = device
        boundary_cfg = dict(((self.cfg.get("model") or {}).get("boundary_decoder") or {}))
        self.boundary = BoundaryDecoder(profile=boundary_cfg.get("profile"),
                                        span=boundary_cfg.get("span"),
                                        dtype=torch.float32).to(device)
        if transform is None:
            grid = int((self.cfg.get("dataset") or {}).get("grid", 256))
            transform = SceneTransform(2.0 * scene_scale, grid)
        self.transform = transform

    # ------------------------------------------------------------------ helpers
    def _alm_cfg(self) -> dict:
        cfg = dict(self.cfg.get("alm") or {})
        # ALM reports its correction in metres; one normalised unit is `scale` m
        cfg.setdefault("scene_to_meter", self.transform.scale)
        return cfg

    def _alm_mode(self) -> str:
        """``"trajsafe"`` (default): TrajSafe's closed loop, full-ellipse corridor.

        ``"legacy_local"`` keeps the previous behaviour (regions only inside the
        collision intervals of the curve) for A/B comparison.
        """
        return str((self.cfg.get("alm") or {}).get("mode_impl", "trajsafe"))

    @torch.no_grad()
    def _decode(self, controls: torch.Tensor, points: Optional[int] = None) -> torch.Tensor:
        if points is None:
            return self.codec.decode_controls(controls)
        params = torch.linspace(0.0, 1.0, int(points), device=controls.device)
        basis = self.codec.basis_at(params).to(dtype=controls.dtype, device=controls.device)
        return torch.einsum("pk,bkd->bpd", basis, controls)

    def _collision_fractions(self, occupancy: torch.Tensor,
                             curve_scene: torch.Tensor) -> torch.Tensor:
        """Segment-level blocked fraction per sample ``[B]`` (P1-1)."""
        occ = occupancy.detach().cpu().numpy()
        curve = curve_scene.detach().cpu().numpy()
        out = []
        for b in range(curve.shape[0]):
            info = curve_collision_intervals(occ[b:b + 1], curve[b])
            out.append(float(info["collision_fraction"]))
        return torch.tensor(out, dtype=torch.float32, device=curve_scene.device)

    @staticmethod
    def _endpoint_error(curve_scene: torch.Tensor, start_goal: torch.Tensor) -> float:
        d0 = (curve_scene[:, 0] - start_goal[:, 0]).norm(dim=-1)
        d1 = (curve_scene[:, -1] - start_goal[:, 1]).norm(dim=-1)
        return float(torch.maximum(d0, d1).max().item())

    def _heading_errors_deg(self, controls: torch.Tensor,
                            heading: Optional[torch.Tensor],
                            heading_valid: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Angle between the curve tangent at u=0/1 and the requested heading.

        ``P'(u) = N'(u) Q`` is exact (upstream derivative-curve identity), so no
        finite differences are involved.  Returns ``[B,2]`` degrees or ``None``.
        """
        if heading is None:
            return None
        basis = self.codec.basis_derivative_at([0.0, 1.0]).to(
            dtype=controls.dtype, device=controls.device)          # [2,C]
        tangent = torch.einsum("pk,bkd->bpd", basis, controls)     # [B,2,2]
        tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        target = heading.to(controls.dtype)
        target = target / target.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        cos = (tangent * target).sum(dim=-1).clamp(-1.0, 1.0)
        err = torch.rad2deg(torch.arccos(cos))
        if heading_valid is not None:
            err = torch.where(heading_valid.to(torch.bool), err, torch.zeros_like(err))
        return err

    @staticmethod
    def accept_correction(active: torch.Tensor, collision_before: torch.Tensor,
                          collision_after: torch.Tensor,
                          violation_before: torch.Tensor,
                          violation_after: torch.Tensor) -> torch.Tensor:
        """Which ALM corrections may be written into the DDIM state (P1-5).

        A correction is accepted only for samples that have a LOCAL convex cell
        (``active``) and only when it does not increase the dense collision
        fraction or the worst half-space violation.  Rejected samples keep the raw
        prediction, so the ALM can never make a sample worse.
        """
        return (active.to(torch.bool)
                & (collision_after <= collision_before + 1e-12)
                & (violation_after <= violation_before + 1e-9))

    # ------------------------------------------------------------------- sample
    @torch.no_grad()
    def sample(self, occupancy: torch.Tensor, sdf: torch.Tensor,
               start_goal: torch.Tensor,
               heading: Optional[torch.Tensor] = None,
               heading_valid: Optional[torch.Tensor] = None,
               steps: Optional[int] = None,
               safety_mode: str = "full",
               alm: bool = True,
               seed: Optional[int] = None,
               return_trace: bool = True,
               record_curves: bool = False) -> Dict:
        """Full inference pipeline for a batch of scenes.

        ``occupancy`` ``[B,1,H,W]`` (1 = blocked, metres grid), ``sdf``
        ``[B,1,H,W]`` metres, ``start_goal`` ``[B,2,2]`` normalised coordinates.

        ``safety_mode``: ``"full"`` (map-only, then safety), ``"map_only"`` (the
        Map-only branch at every step: no search, no ellipse, no ALM) or
        ``"no_alm"`` (safety conditions, no ALM - ablation F).
        """
        if seed is not None:
            torch.manual_seed(int(seed))
        sampling_cfg = dict(self.cfg.get("sampling") or {})
        steps = int(steps or sampling_cfg.get("steps", 50))
        mid_fraction = float(sampling_cfg.get("mid_forward_fraction", 0.5))
        alm_cfg = self._alm_cfg()
        corridor_cfg = dict(self.cfg.get("corridor") or {})
        alm_on = bool(alm and alm_cfg.get("enabled", True) and safety_mode == "full")
        inherit_dual = bool(alm_cfg.get("inherit_dual_across_reverse_steps", True))
        dense_points = int(alm_cfg.get("dense_verify_points", 512))
        heading_tol = float(sampling_cfg.get("heading_tol_deg", 20.0))

        device = self.device
        occupancy = occupancy.to(device)
        sdf = sdf.to(device)
        start_goal = start_goal.to(device)
        if heading is not None:
            heading = heading.to(device)
        if heading_valid is not None:
            heading_valid = heading_valid.to(device)
        B = start_goal.shape[0]
        C = int(self.model.num_controls)

        timing: Dict[str, float] = {}
        t_all = time.perf_counter()

        # ---- Step 0: one map encoding for the whole run ---------------------
        t0 = time.perf_counter()
        features = self.map_encoder.encode(occupancy, sdf)
        map_memory = self.map_condition_encoder(features["occ_bottleneck"],
                                                features["sdf_bottleneck"])
        # Map Context = 1024 map tokens + 2 start/goal tokens, reused by every
        # reverse step of both modes (spec §6.1)
        map_context = self.model.build_map_context(map_memory, start_goal,
                                                   heading, heading_valid)
        timing["map_encode_s"] = time.perf_counter() - t0

        # ---- Step 1 setup: noise + hard endpoints --------------------------
        q = torch.randn(B, C, 2, device=device, dtype=torch.float32)
        q = self.boundary(q, start_goal, heading, heading_valid)
        T = int(self.schedule.num_timesteps)
        times = pick_times(T, steps) or list(range(T))
        reverse = list(reversed(times))
        k_safety = int(round(mid_fraction * (len(reverse) - 1)))

        sqrt_ab = self.schedule.sqrt_alphas_cumprod.to(device).float()
        sqrt_1ma = self.schedule.sqrt_one_minus_alphas_cumprod.to(device).float()

        safety_condition = None
        safety_memory = None
        safety_mask = None
        safety_raw = None
        guide_scene = None
        pack = None
        lam = None
        corridor_info: Dict = {}
        search_stats: List[Dict] = []
        alm_stats: List[Dict] = []
        per_step: List[Dict] = []
        search_failed = torch.zeros(B, dtype=torch.bool, device=device)
        local_cells = torch.zeros(B, dtype=torch.long, device=device)
        alm_accepted = 0
        alm_rejected = 0

        for i, t_cur in enumerate(reverse):
            t_prev = reverse[i + 1] if i + 1 < len(reverse) else -1
            use_safety = safety_condition is not None and safety_mode != "map_only"
            t_tensor = torch.full((B,), int(t_cur), device=device, dtype=torch.long)

            t_fwd = time.perf_counter()
            pred = self.model(q_t=q, t=t_tensor, map_context=map_context,
                              safety_memory=safety_memory if use_safety else None,
                              safety_mask=safety_mask if use_safety else None)
            # P1-3: the network has no hard endpoint conditioning, so lock the
            # predicted polygon BEFORE decoding, searching and projecting.
            pred = self.boundary(pred, start_goal, heading, heading_valid)
            dt_fwd = time.perf_counter() - t_fwd
            key = "guided_diffusion_s" if use_safety else "initial_diffusion_s"
            timing[key] = timing.get(key, 0.0) + dt_fwd

            # ---- Step 2: generate the safety geometry exactly once ---------
            if i == k_safety and safety_mode in ("full", "no_alm"):
                t0 = time.perf_counter()
                curve_scene = self._decode(pred)
                guide_scene = curve_scene.clone()   # the map-only guide trajectory
                safety_raw, search_stats = self._generate_safety(
                    occupancy, sdf, features, curve_scene, start_goal, corridor_cfg)
                if safety_raw is not None:
                    timing.update(safety_raw["timing"])
                timing["safety_total_s"] = time.perf_counter() - t0
                if safety_raw is None:
                    # every search failed: stay map-only and say so (P1-4)
                    search_failed[:] = True
                    safety_condition = None
                    safety_memory = None
                    safety_mask = None
                    pack = None
                    corridor_info = {"local_cells": 0, "spans": [],
                                     "reason": "search_failed_all"}
                else:
                    # safety condition = ellipse tokens only (spec §3.1)
                    safety_condition = {"ellipse": safety_raw["ellipse"],
                                        "valid": safety_raw["valid"]}
                    # encoded ONCE here and reused by every guided reverse step
                    safety_memory, safety_mask = self.model.build_safety_memory(
                        safety_condition["ellipse"], safety_condition["valid"])
                    pack = safety_raw["pack"]
                    local_cells = safety_raw["local_cells"]
                    search_failed = safety_raw["search_failed"]
                    corridor_info = safety_raw["corridor_info"]
                use_safety = False   # the generating forward itself is map-only

            q0_used = pred
            step_alm = None
            if use_safety and alm_on and pack is not None:
                # P1-2: only samples with a LOCAL convex cell are corrected
                active = local_cells > 0
                if bool(active.any()):
                    from .bspline_alm import bspline_alm_correct
                    t0 = time.perf_counter()
                    if self._alm_mode() == "trajsafe":
                        # VERBATIM TrajSafe usage (src/diffusion/sampler.py): the
                        # corrected prediction enters the DDIM update unguarded, the
                        # dual is warm started across reverse steps.  No extra
                        # acceptance test -- that was a GeoPath-only addition.
                        q_safe, lam_new, st = bspline_alm_correct(
                            pred, pack, self.codec,
                            lam=(lam if inherit_dual else None), config=alm_cfg,
                            inner_steps=int(alm_cfg.get("inner_steps", 6)))
                        if inherit_dual:
                            lam = lam_new
                        accept = active.clone()
                        q_safe = torch.where(active[:, None, None], q_safe, pred)
                        alm_accepted += int(active.sum().item())
                        step_alm = {
                            "active_samples": int(active.sum().item()),
                            "accepted_samples": int(active.sum().item()),
                            "max_violation_before": float(st["max_violation_before"].mean()),
                            "max_violation_after": float(st["max_violation_after"].mean()),
                            "feasible_rate": float(st["constraint_feasible_rate"].mean()),
                            "mean_correction_m": float(st["mean_curve_correction_m"].mean()),
                            "max_correction_m": float(st["max_curve_correction_m"].mean()),
                            "accepted_mask": [bool(v) for v in accept.detach().cpu()],
                        }
                        timing["alm_s"] = timing.get("alm_s", 0.0) + time.perf_counter() - t0
                    else:
                        q_safe, lam_new, stats = bspline_alm_correct(
                            pred, pack, self.codec, lam=lam if inherit_dual else None,
                            config=alm_cfg, inner_steps=int(alm_cfg.get("inner_steps", 6)))
                        timing["alm_s"] = timing.get("alm_s", 0.0) + time.perf_counter() - t0
                        before = self._decode(pred, dense_points)
                        after = self._decode(q_safe, dense_points)
                        col_before = self._collision_fractions(occupancy, before)
                        col_after = self._collision_fractions(occupancy, after)
                        viol_before = stats["max_violation_before"].to(device)
                        viol_after = stats["max_violation_after"].to(device)
                        # P1-5: never accept a correction that makes the sample worse
                        accept = self.accept_correction(active, col_before, col_after,
                                                        viol_before, viol_after)
                        q_safe = torch.where(accept[:, None, None], q_safe, pred)
                        if inherit_dual:
                            lam = lam_new
                        alm_accepted += int(accept.sum().item())
                        alm_rejected += int((active & ~accept).sum().item())
                        step_alm = {
                            "active_samples": int(active.sum().item()),
                            "accepted_samples": int(accept.sum().item()),
                            "max_violation_before": float(viol_before.mean().item()),
                            "max_violation_after": float(viol_after.mean().item()),
                            "feasible_rate": float(stats["constraint_feasible_rate"].mean()),
                            "mean_correction_m": float(stats["mean_curve_correction_m"].mean()),
                            "max_correction_m": float(stats["max_curve_correction_m"].mean()),
                            "collision_before": float(col_before.mean().item()),
                            "collision_after": float(col_after.mean().item()),
                            "accepted_mask": [bool(v) for v in accept.detach().cpu()],
                        }
                    alm_stats.append(step_alm)
                    q0_used = q_safe

            # ---- DDIM update -------------------------------------------------
            sa_t, s1_t = sqrt_ab[int(t_cur)], sqrt_1ma[int(t_cur)]
            if t_prev < 0:
                q = q0_used
            else:
                sa_s, s1_s = sqrt_ab[int(t_prev)], sqrt_1ma[int(t_prev)]
                eps = (q - sa_t * q0_used) / s1_t.clamp_min(1e-8)
                q = sa_s * q0_used + s1_s * eps
            q = self.boundary(q, start_goal, heading, heading_valid)
            if return_trace:
                entry = {"forward": i, "t": int(t_cur), "t_prev": int(t_prev),
                         "branch": "safety" if use_safety else "map_only",
                         "alm": step_alm}
                if record_curves:
                    # visualisation only: the x0 estimate before/after ALM at this step
                    entry["x0_scene"] = self._decode(pred).detach().cpu().numpy()
                    entry["x0_alm_scene"] = self._decode(q0_used).detach().cpu().numpy()
                per_step.append(entry)

        # ---- Step 4: final trajectory + dense validation -------------------
        t0 = time.perf_counter()
        curve_scene = self._decode(q)
        dense = self._decode(q, dense_points)
        collision = self._collision_fractions(occupancy, dense)
        endpoint_err = self._endpoint_error(curve_scene, start_goal)
        heading_err = self._heading_errors_deg(q, heading, heading_valid)
        timing["final_validation_s"] = time.perf_counter() - t0
        timing["total_s"] = time.perf_counter() - t_all

        collision_free = collision <= 1e-9
        heading_ok = (torch.ones(B, dtype=torch.bool, device=device)
                      if heading_err is None
                      else (heading_err <= heading_tol).all(dim=-1))
        valid = (collision_free & heading_ok & ~search_failed
                 & torch.full((B,), endpoint_err < 1e-4, dtype=torch.bool, device=device))

        return {
            "controls": q,
            "trajectory_scene": curve_scene,
            "trajectory_m": torch.as_tensor(self.transform.to_meters(
                curve_scene.detach().cpu().numpy()), dtype=torch.float32),
            "valid": bool(valid.all().item()),
            "valid_mask": valid.detach().cpu(),
            "dense_collision_fraction": float(collision.mean().item()),
            "dense_collision_fraction_per_sample": collision.detach().cpu(),
            "endpoint_error_scene": endpoint_err,
            "heading_error_deg": (None if heading_err is None
                                  else heading_err.detach().cpu()),
            "heading_tol_deg": heading_tol,
            "search_failed": search_failed.detach().cpu(),
            "local_cells_per_sample": local_cells.detach().cpu(),
            "safety_condition": safety_condition,
            "safety_memory": safety_memory,
            "safety_mask": safety_mask,
            "map_context_tokens": int(map_context.shape[1]),
            "ellipses": (None if safety_raw is None else safety_raw["ellipse"]),
            "ellipse_centers_scene": (None if safety_raw is None
                                      else safety_raw["ellipse_centers_scene"]),
            "search_path_m": (None if safety_raw is None else safety_raw["search_path_m"]),
            "guide_trajectory_scene": guide_scene,
            "guide_trajectory_m": (None if guide_scene is None else torch.as_tensor(
                self.transform.to_meters(guide_scene.detach().cpu().numpy()),
                dtype=torch.float32)),
            "search_stats": search_stats,
            "corridor_info": corridor_info,
            "pack": pack,
            "pack_local": (None if safety_raw is None else safety_raw.get("pack_local")),
            "alm_stats": alm_stats,
            "alm_accepted_steps": alm_accepted,
            "alm_rejected_steps": alm_rejected,
            "timing": timing,
            "steps_trace": per_step,
            "frames": {"k_safety": k_safety, "num_forwards": len(reverse),
                       "safety_generated_once": bool(safety_condition is not None)},
        }

    # -------------------------------------------------------- safety generation
    @torch.no_grad()
    def _generate_safety(self, occupancy, sdf, features, curve_scene, start_goal,
                         corridor_cfg) -> Tuple[Optional[Dict], List[Dict]]:
        """One guided search + one ellipse pass + one corridor build."""
        timing: Dict[str, float] = {}
        B = curve_scene.shape[0]
        tf = self.transform
        device = self.device

        timing["search_s"] = 0.0
        paths_m: List[Optional[torch.Tensor]] = []
        search_stats: List[Dict] = []
        for b in range(B):
            t0 = time.perf_counter()
            occ_np = occupancy[b, 0].detach().cpu().numpy()
            guide_m = tf.to_meters(curve_scene[b].detach().cpu().numpy())
            start_m = tf.to_meters(start_goal[b, 0].detach().cpu().numpy())
            goal_m = tf.to_meters(start_goal[b, 1].detach().cpu().numpy())
            result = self.guidance.search(occ_np, guide_m, start_m, goal_m)
            timing["search_s"] += time.perf_counter() - t0
            path = result.get("path_xy")
            # P1-4: a failed search is a FAILED sample, never a "path"
            ok = path is not None and bool(result.get("valid", True))
            paths_m.append(torch.as_tensor(np.asarray(path), dtype=torch.float32,
                                           device=device) if ok else None)
            search_stats.append({k: result.get(k) for k in
                                 ("status", "fallback", "sources", "expanded",
                                  "valid", "endpoint_projected", "path_verified")})

        ok_idx = [b for b in range(B) if paths_m[b] is not None]
        search_failed = torch.tensor([paths_m[b] is None for b in range(B)],
                                     dtype=torch.bool, device=device)
        local_cells = torch.zeros(B, dtype=torch.long, device=device)
        if not ok_idx:
            return None, search_stats

        path_m = torch.stack([paths_m[b] for b in ok_idx])
        n_ok = len(ok_idx)
        t0 = time.perf_counter()
        sub_features = {k: v[ok_idx] for k, v in features.items()}
        safety = self.safety_generator(occupancy[ok_idx], sdf[ok_idx], path_m,
                                       features=sub_features)
        timing["ellipse_s"] = time.perf_counter() - t0

        # ---- metres -> scene, including the AXES (P0-2) --------------------
        scale = tf.scale
        axes_m = torch.stack([safety["ellipse"][..., 2], safety["ellipse"][..., 3]],
                             dim=-1)
        centers_m = safety["ellipse_center"]
        centers_scene = torch.as_tensor(
            tf.to_scene(centers_m.detach().cpu().numpy()), dtype=torch.float32,
            device=device)
        axes_scene = torch.as_tensor(
            tf.lengths_to_scene(axes_m.detach().cpu().numpy()), dtype=torch.float32,
            device=device)
        theta = safety["ellipse"][..., 4:]
        ellipse_scene = torch.cat([centers_scene, axes_scene, theta], dim=-1)
        # the region builder exponentiates the log-axes, so it needs scene units
        shape4_scene = ellipse_shape4_to_scene(safety["ellipse_shape4"], scale)

        # ---- convex corridor + constraint pack -----------------------------
        t0 = time.perf_counter()
        Q = int(centers_scene.shape[1])
        anchors = torch.linspace(0.0, 1.0, Q, dtype=torch.float64, device=device)
        tables: List[Optional[Dict]] = []
        infos: List[Dict] = []
        path_scene = torch.as_tensor(
            tf.to_scene(path_m.detach().cpu().numpy()), dtype=torch.float32, device=device)
        if self._alm_mode() == "trajsafe":
            # VERBATIM TrajSafe construction (src/diffusion/alm_local.py): every
            # predicted ellipse -> convex cell, gap filling, then one closed
            # safety corridor per sample (overlap + bridge), then the Bezier
            # constraint pack.  No collision-interval restriction.
            pack, pack_local, corridors = self._full_corridor_pack(
                occupancy, centers_scene, shape4_scene, anchors, path_m,
                ok_idx, corridor_cfg)
            for local_i, b in enumerate(ok_idx):
                cor = corridors[local_i]
                local_cells[b] = int(getattr(pack, "num_pieces", torch.zeros(B))[b]
                                     if pack is not None else 0)
                infos.append({"index": int(b),
                              "local_cells": int(local_cells[b].item()),
                              "valid": bool(getattr(cor, "valid", False)),
                              "failure_reason": getattr(cor, "failure_reason", "") or "",
                              "spans": None})
            timing["corridor_s"] = time.perf_counter() - t0
        else:
            for local_i, b in enumerate(ok_idx):
                table, info = self._region_table(
                    occupancy[b:b + 1], centers_scene[local_i], shape4_scene[local_i],
                    anchors, curve_scene[b], path_scene[local_i],
                    corridor_cfg, scope="local")
                tables.append(table)
                local_cells[b] = int(info["local_cells"])
                infos.append({"index": int(b), **info})
            timing["corridor_s"] = time.perf_counter() - t0
            # the ALM gets the full pack (local cells + map box), while the reported
            # corridor violation and the training loss use the LOCAL cells only
            pack = pack_from_tables(self.codec, tables, include_box=True, device=device)
            pack_local = pack_from_tables(self.codec, tables, include_box=False,
                                          device=device)

        # ---- scatter the per-sample geometry back to the full batch -------
        path_len = int(path_m.shape[1])
        ellipse_scene_full = torch.zeros(B, Q, 6, dtype=torch.float32, device=device)
        centers_scene_full = torch.zeros(B, Q, 2, dtype=torch.float32, device=device)
        path_scene_full = torch.zeros(B, path_len, 2, dtype=torch.float32, device=device)
        path_m_full = torch.zeros(B, path_len, 2, dtype=torch.float32, device=device)
        # ellipse validity only: a failed-search sample keeps an all-False mask so
        # its safety residual is exactly zero (no fabricated ellipses)
        valid_rows = torch.zeros(B, Q, dtype=torch.bool, device=device)
        for local_i, b in enumerate(ok_idx):
            ellipse_scene_full[b] = ellipse_scene[local_i]
            centers_scene_full[b] = centers_scene[local_i]
            path_scene_full[b] = path_scene[local_i]
            path_m_full[b] = path_m[local_i]
            valid_rows[b] = True

        if len(infos) == 1:
            corridor_info: Dict = {**infos[0],
                                   "search_failed_samples": int(search_failed.sum().item()),
                                   "batch": infos}
        else:
            corridor_info = {
                "local_cells": int(local_cells.sum().item()),
                "spans": [i["spans"] for i in infos],
                "search_failed_samples": int(search_failed.sum().item()),
                "batch": infos,
            }

        raw = {
            "path_m": path_m_full,
            "path_scene": path_scene_full,
            "search_path_m": path_m_full,
            "ellipse": ellipse_scene_full,
            "ellipse_centers_scene": centers_scene_full,
            "valid": valid_rows,
            "pack": pack,
            "pack_local": pack_local,
            "local_cells": local_cells,
            "search_failed": search_failed,
            "corridor_info": corridor_info,
            "timing": timing,
        }
        return raw, search_stats

    def _full_corridor_pack(self, occupancy, centers_scene, shape4_scene, anchors,
                            path_m, ok_idx, corridor_cfg):
        """TrajSafe's FULL construction: every ellipse -> corridor -> pack.

        This is the sequence of ``vendor/trajsafe/alm_local.py`` (their
        ``alm_closed_loop``), minus the ALM call itself:

            EllipseRegionBuilder.build_from_ellipse -> per-query convex cells
            densify_ellipses                        -> fill the corridor gaps
            build_safety_corridor                   -> one closed corridor
            build_constraint_pack                   -> Bezier-extracted constraints

        Nothing here restricts the regions to the collision intervals; the whole
        predicted ellipse chain is used, which is what the ALM was designed for.
        """
        from ..safety.bspline_constraints import build_constraint_pack
        from ..safety.convex_region import EllipseRegionBuilder
        from ..safety.safety_corridor import build_safety_corridor
        from ..vendor.trajsafe.ellipse_densify import densify_ellipses

        alm_cfg = self._alm_cfg()
        region_cfg = dict(corridor_cfg.get("region") or {})
        builder_cfg = {k: alm_cfg[k] for k in
                       ("safety_margin", "obstacle_window_half", "obstacle_dilation",
                        "corridor_chunk_size", "ellipse_axis_min", "ellipse_axis_max")
                       if k in alm_cfg}
        builder_cfg = {**region_cfg, **builder_cfg}
        dens_cfg = dict(corridor_cfg.get("densify") or {})
        dens_on = bool(dens_cfg.get("enabled", True))
        path_scene_full = torch.as_tensor(
            self.transform.to_scene(path_m.detach().cpu().numpy()),
            dtype=torch.float32, device=self.device)

        B = int(centers_scene.shape[0])
        corridors: List[Optional[object]] = [None] * B
        for local_i, b in enumerate(ok_idx):
            builder = EllipseRegionBuilder(occupancy[b:b + 1].float(), builder_cfg)
            c_b, s_b, a_b = centers_scene[local_i], shape4_scene[local_i], anchors
            if dens_on:
                c_np, s_np, a_np, _ = densify_ellipses(
                    builder,
                    c_b.detach().cpu().numpy(), s_b.detach().cpu().numpy(),
                    a_b.detach().cpu().numpy(),
                    min_overlap=float(dens_cfg.get(
                        "min_overlap", corridor_cfg.get("min_overlap_ratio", 0.10))),
                    max_insert_per_gap=int(dens_cfg.get("max_insert_per_gap", 2)),
                    max_total=int(dens_cfg.get("max_total", 256)))
                c_b = torch.as_tensor(c_np, dtype=centers_scene.dtype, device=self.device)
                s_b = torch.as_tensor(s_np, dtype=shape4_scene.dtype, device=self.device)
                a_b = torch.as_tensor(a_np, dtype=anchors.dtype, device=self.device)
            corridors[b] = build_safety_corridor(
                builder, c_b, s_b, a_b, gamma=path_scene_full[b],
                gamma_lengths=max(int(path_scene_full.shape[1]), 1),
                config=corridor_cfg)

        pack = build_constraint_pack(
            self.codec, corridors, device=self.device, dtype=torch.float32,
            margin=float(alm_cfg.get("constraint_margin", 0.0)))
        return pack, pack, corridors

    def _region_table(self, occupancy_b, centers_scene, shape4_scene, anchors,
                      curve_scene, path_scene, corridor_cfg, scope="local"):
        """Thin wrapper so the sampling path and the cache build the same table."""
        from ..safety.local_regions import build_region_table
        return build_region_table(
            occupancy_b, centers_scene, shape4_scene, anchors,
            curve_scene=curve_scene, gamma_scene=path_scene, config=corridor_cfg,
            scope=scope, scene_scale=self.transform.scale)
