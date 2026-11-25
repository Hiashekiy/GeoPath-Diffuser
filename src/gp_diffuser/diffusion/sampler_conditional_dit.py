"""Conditional-DiT DDIM sampler with one-shot mid-time safety injection.

Implements the inference flow of the design document section 8 exactly:

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
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import torch

from ..frames import SceneTransform
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

    def __init__(self, model, codec, schedule: NoiseSchedule, map_projection,
                 map_encoder, safety_generator, guidance_search, cfg: dict,
                 device: str = "cpu", transform: Optional[SceneTransform] = None,
                 scene_scale: float = 32.0):
        self.model = model
        self.codec = codec
        self.schedule = schedule
        self.map_projection = map_projection
        self.map_encoder = map_encoder
        self.safety_generator = safety_generator
        self.guidance = guidance_search
        self.cfg = dict(cfg or {})
        self.device = device
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

    @torch.no_grad()
    def _decode(self, controls: torch.Tensor, points: Optional[int] = None) -> torch.Tensor:
        if points is None:
            return self.codec.decode_controls(controls)
        params = torch.linspace(0.0, 1.0, int(points), device=controls.device)
        basis = self.codec.basis_at(params).to(dtype=controls.dtype, device=controls.device)
        return torch.einsum("pk,bkd->bpd", basis, controls)

    @staticmethod
    def _collision_fraction(occupancy: torch.Tensor, curve_scene: torch.Tensor) -> float:
        """Mean blocked-sample fraction over the batch (dense curve check)."""
        from ..safety.local_regions import collision_mask
        fractions = []
        for b in range(curve_scene.shape[0]):
            hit, _outside = collision_mask(occupancy[b:b + 1], curve_scene[b])
            fractions.append(float(hit.mean()) if len(hit) else 0.0)
        return float(sum(fractions) / max(len(fractions), 1))

    @staticmethod
    def _endpoint_error(curve_scene: torch.Tensor, start_goal: torch.Tensor) -> float:
        d0 = (curve_scene[:, 0] - start_goal[:, 0]).norm(dim=-1)
        d1 = (curve_scene[:, -1] - start_goal[:, 1]).norm(dim=-1)
        return float(torch.maximum(d0, d1).max().item())

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
               return_trace: bool = True) -> Dict:
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

        device = self.device
        occupancy = occupancy.to(device)
        sdf = sdf.to(device)
        start_goal = start_goal.to(device)
        B = start_goal.shape[0]
        C = int(self.model.num_controls)

        timing: Dict[str, float] = {}
        t_all = time.perf_counter()

        # ---- Step 0: one map encoding for the whole run ---------------------
        t0 = time.perf_counter()
        features = self.map_encoder.encode(occupancy, sdf)
        map_memory = self.map_projection(features["occ_bottleneck"],
                                         features["sdf_bottleneck"])
        timing["map_encode_s"] = time.perf_counter() - t0

        # ---- Step 1 setup: noise + hard endpoints --------------------------
        q = torch.randn(B, C, 2, device=device, dtype=torch.float32)
        q = self.codec.hard_control_endpoints(q, start_goal)
        T = int(self.schedule.num_timesteps)
        times = pick_times(T, steps) or list(range(T))
        reverse = list(reversed(times))
        k_safety = int(round(mid_fraction * (len(reverse) - 1)))

        sqrt_ab = self.schedule.sqrt_alphas_cumprod.to(device).float()
        sqrt_1ma = self.schedule.sqrt_one_minus_alphas_cumprod.to(device).float()

        safety_condition = None
        safety_raw = None
        guide_scene = None
        pack = None
        lam = None
        corridor_info: Dict = {}
        search_stats: List[Dict] = []
        alm_stats: List[Dict] = []
        per_step: List[Dict] = []

        for i, t_cur in enumerate(reverse):
            t_prev = reverse[i + 1] if i + 1 < len(reverse) else -1
            use_safety = safety_condition is not None and safety_mode != "map_only"
            t_tensor = torch.full((B,), int(t_cur), device=device, dtype=torch.long)

            t_fwd = time.perf_counter()
            pred = self.model(q_t=q, t=t_tensor, map_memory=map_memory,
                              cond=start_goal, heading=heading,
                              heading_valid=heading_valid,
                              safety_condition=safety_condition if use_safety else None)
            dt_fwd = time.perf_counter() - t_fwd
            key = "guided_diffusion_s" if use_safety else "initial_diffusion_s"
            timing[key] = timing.get(key, 0.0) + dt_fwd

            # ---- Step 2: generate the safety geometry exactly once ---------
            if i == k_safety and safety_mode in ("full", "no_alm"):
                t0 = time.perf_counter()
                curve_scene = self._decode(pred)
                guide_scene = curve_scene.clone()      # the map-only guide trajectory
                safety_raw, search_stats = self._generate_safety(
                    occupancy, sdf, features, curve_scene, start_goal, corridor_cfg)
                safety_condition = {
                    "path": safety_raw["path_scene"],
                    "ellipse": safety_raw["ellipse"],
                    "valid": safety_raw["valid"],
                }
                pack = safety_raw["pack"]
                corridor_info = safety_raw["corridor_info"]
                timing.update(safety_raw["timing"])
                timing["safety_total_s"] = time.perf_counter() - t0
                use_safety = False   # the generating forward itself is map-only

            q0_used = pred
            step_alm = None
            if use_safety and alm_on and pack is not None and bool(int(pack.piece_mask.sum()) > 0):
                from .bspline_alm import bspline_alm_correct
                t0 = time.perf_counter()
                q_safe, lam_new, stats = bspline_alm_correct(
                    pred, pack, self.codec, lam=lam if inherit_dual else None,
                    config=alm_cfg, inner_steps=int(alm_cfg.get("inner_steps", 6)))
                timing["alm_s"] = timing.get("alm_s", 0.0) + time.perf_counter() - t0
                if inherit_dual:
                    lam = lam_new
                dense = self._decode(q_safe, int(alm_cfg.get("dense_verify_points", 512)))
                collision = self._collision_fraction(occupancy, dense)
                valid_after = (float(stats["constraint_feasible_rate"].mean()) >= 0.999
                               and collision <= 1e-9)
                step_alm = {
                    "max_violation_before": float(stats["max_violation_before"].mean()),
                    "max_violation_after": float(stats["max_violation_after"].mean()),
                    "feasible_rate": float(stats["constraint_feasible_rate"].mean()),
                    "mean_correction_m": float(stats["mean_curve_correction_m"].mean()),
                    "max_correction_m": float(stats["max_curve_correction_m"].mean()),
                    "dense_collision_fraction": collision,
                    "dense_validation_passed": bool(valid_after),
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
            q = self.codec.hard_control_endpoints(q, start_goal)
            if return_trace:
                per_step.append({"forward": i, "t": int(t_cur), "t_prev": int(t_prev),
                                 "branch": "safety" if use_safety else "map_only",
                                 "alm": step_alm})

        # ---- Step 4: final trajectory + dense validation -------------------
        t0 = time.perf_counter()
        curve_scene = self._decode(q)
        dense = self._decode(q, int(alm_cfg.get("dense_verify_points", 512)))
        collision = self._collision_fraction(occupancy, dense)
        endpoint_err = self._endpoint_error(curve_scene, start_goal)
        timing["final_validation_s"] = time.perf_counter() - t0
        timing["total_s"] = time.perf_counter() - t_all

        return {
            "controls": q,
            "trajectory_scene": curve_scene,
            "trajectory_m": torch.as_tensor(self.transform.to_meters(
                curve_scene.detach().cpu().numpy()), dtype=torch.float32),
            "valid": bool(collision <= 1e-9 and endpoint_err < 1e-4),
            "dense_collision_fraction": collision,
            "endpoint_error_scene": endpoint_err,
            "safety_condition": safety_condition,
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
            "alm_stats": alm_stats,
            "timing": timing,
            "steps_trace": per_step,
            "frames": {"k_safety": k_safety, "num_forwards": len(reverse),
                       "safety_generated_once": bool(safety_condition is not None)},
        }

    # -------------------------------------------------------- safety generation
    @torch.no_grad()
    def _generate_safety(self, occupancy, sdf, features, curve_scene, start_goal,
                         corridor_cfg) -> tuple:
        """One guided search + one ellipse pass + one corridor build (per sample)."""
        timing: Dict[str, float] = {}
        B = curve_scene.shape[0]
        tf = self.transform
        path_m_list, path_scene_list, search_stats = [], [], []
        timing["search_s"] = 0.0
        for b in range(B):
            t0 = time.perf_counter()
            occ_np = occupancy[b, 0].detach().cpu().numpy()
            guide_m = tf.to_meters(curve_scene[b].detach().cpu().numpy())
            start_m = tf.to_meters(start_goal[b, 0].detach().cpu().numpy())
            goal_m = tf.to_meters(start_goal[b, 1].detach().cpu().numpy())
            result = self.guidance.search(occ_np, guide_m, start_m, goal_m)
            timing["search_s"] += time.perf_counter() - t0
            path_m = result["path_xy"]
            status = result["status"]
            if path_m is None:
                path_m = guide_m            # flagged fallback: guide trajectory
                status = "fallback_to_guide"
            search_stats.append({k: result[k] for k in
                                 ("status", "fallback", "sources", "expanded", "valid")}
                                | {"status": status})
            path_m_list.append(torch.as_tensor(np.asarray(path_m), dtype=torch.float32,
                                               device=self.device))
            path_scene_list.append(torch.as_tensor(tf.to_scene(np.asarray(path_m)),
                                                   dtype=torch.float32,
                                                   device=self.device))
        path_m = torch.stack(path_m_list)
        path_scene = torch.stack(path_scene_list)

        t0 = time.perf_counter()
        safety = self.safety_generator(occupancy, sdf, path_m, features=features)
        timing["ellipse_s"] = time.perf_counter() - t0

        centers_m = safety["ellipse_center"]
        axes_m = torch.stack([safety["ellipse"][..., 2], safety["ellipse"][..., 3]], dim=-1)
        centers_scene = torch.as_tensor(tf.to_scene(centers_m.detach().cpu().numpy()),
                                        dtype=torch.float32, device=self.device)
        axes_scene = torch.as_tensor(tf.lengths_to_scene(axes_m.detach().cpu().numpy()),
                                     dtype=torch.float32, device=self.device)
        theta = safety["ellipse"][..., 4:]
        ellipse_scene = torch.cat([centers_scene, axes_scene, theta], dim=-1)
        shape4 = safety["ellipse_shape4"]

        t0 = time.perf_counter()
        from ..safety.local_regions import build_local_constraint_pack
        Q = int(centers_scene.shape[1])
        anchors = torch.linspace(0.0, 1.0, Q, dtype=torch.float64, device=self.device)
        packs, infos = [], []
        for b in range(B):
            pack, info = build_local_constraint_pack(
                self.codec, occupancy[b:b + 1], centers_scene[b], shape4[b],
                anchors, curve_scene[b], gamma_scene=path_scene[b],
                config=corridor_cfg, scope="local")
            packs.append(pack)
            infos.append(info)
        timing["corridor_s"] = time.perf_counter() - t0

        valid_rows = torch.ones(B, Q + 128, dtype=torch.bool, device=self.device)
        raw = {
            "path_m": path_m,
            "path_scene": path_scene,
            "search_path_m": path_m,
            "ellipse": ellipse_scene,
            "ellipse_centers_scene": centers_scene,
            "valid": valid_rows,
            "pack": (packs[0] if B == 1 else _merge_packs(packs)),
            "corridor_info": (infos[0] if B == 1 else {"batch": infos}),
            "timing": timing,
        }
        return raw, search_stats


def _merge_packs(packs):
    """Concatenate per-sample packs along the batch axis.

    The piece / face axes are padded to the batch maximum: per-sample packs have
    different numbers of pieces (different collision spans) and different face
    counts, so ``torch.cat`` cannot be used directly.
    """
    packs = [p for p in packs if p is not None]
    if not packs:
        return None
    if len(packs) == 1:
        return packs[0]
    from ..safety.bspline_constraints import BSplineConstraintPack

    max_pieces = max(p.extraction.shape[1] for p in packs)
    max_faces = max(p.piece_A.shape[2] for p in packs)
    controls = int(packs[0].extraction.shape[-1])
    batch = sum(int(p.extraction.shape[0]) for p in packs)
    extraction = torch.zeros(batch, max_pieces, 4, controls, dtype=packs[0].extraction.dtype)
    piece_A = torch.zeros(batch, max_pieces, max_faces, 2, dtype=packs[0].piece_A.dtype)
    piece_b = torch.zeros(batch, max_pieces, max_faces, dtype=packs[0].piece_b.dtype)
    face_mask = torch.zeros(batch, max_pieces, max_faces, dtype=torch.bool)
    piece_mask = torch.zeros(batch, max_pieces, dtype=torch.bool)
    piece_region_id = torch.full((batch, max_pieces), -1, dtype=torch.long)
    intervals = torch.zeros(batch, max_pieces, 2, dtype=packs[0].intervals.dtype)
    offset = 0
    for pack in packs:
        n, p, f = pack.extraction.shape[0], pack.extraction.shape[1], pack.piece_A.shape[2]
        extraction[offset:offset + n, :p] = pack.extraction
        piece_A[offset:offset + n, :p, :f] = pack.piece_A
        piece_b[offset:offset + n, :p, :f] = pack.piece_b
        face_mask[offset:offset + n, :p, :f] = pack.face_mask
        piece_mask[offset:offset + n, :p] = pack.piece_mask
        piece_region_id[offset:offset + n, :p] = pack.piece_region_id
        intervals[offset:offset + n, :p] = pack.intervals
        offset += n
    return BSplineConstraintPack(
        extraction=extraction, piece_A=piece_A, piece_b=piece_b,
        face_mask=face_mask, piece_mask=piece_mask,
        piece_region_id=piece_region_id, intervals=intervals,
        num_pieces=torch.cat([p.num_pieces for p in packs], dim=0))
