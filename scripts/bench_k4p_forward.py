"""Time the k4p pipeline: per-stage geometry build + one training forward/step.

Everything goes through the project's own modules; nothing is re-implemented:

    geometry  ReferenceGuidanceSearch / SafetyConditionGenerator /
              build_region_table / build_sampled_constraint_pack_from_tables
    training  factory.build_pipeline + DualBranchTrainer._compute / train_step

Usage::

    python scripts/bench_k4p_forward.py --device cuda --batch 4 --repeat 10
    python scripts/bench_k4p_forward.py --device cpu  --batch 4 --repeat 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import PROJECT_ROOT, base_parser, load  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gp_diffuser.factory import build_pipeline  # noqa: E402
from gp_diffuser.frames import SceneTransform  # noqa: E402
from gp_diffuser.guidance.reference_search import ReferenceGuidanceSearch  # noqa: E402
from gp_diffuser.safety.bspline_sample_constraints import (  # noqa: E402
    build_sampled_constraint_pack_from_tables)
from gp_diffuser.safety.local_regions import build_region_table  # noqa: E402
from gp_diffuser.safety.safety_features import (  # noqa: E402
    SafetyConditionGenerator, ellipse_shape4_to_scene)
from gp_diffuser.training.dual_branch_trainer import DualBranchTrainer  # noqa: E402
from gp_diffuser.training.real_dataset import RealTrajectoryDataset  # noqa: E402


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


class Timer:
    def __init__(self, device):
        self.device = device

    def __call__(self, fn, repeat=1):
        sync(self.device)
        t0 = time.perf_counter()
        for _ in range(repeat):
            out = fn()
        sync(self.device)
        return (time.perf_counter() - t0) / repeat, out


def build_geometry(cfg, dataset, count, net, codec, transform, device):
    """Per-sample geometry pass, with per-stage timings (seconds)."""
    occ_t_factory = lambda arr: torch.as_tensor(arr, dtype=torch.float32, device=device)[None, None]
    guidance = ReferenceGuidanceSearch(cfg.get("guidance") or {}, res_m=float(cfg["dataset"]["res_m"]))
    gen = SafetyConditionGenerator(net, num_queries=int(cfg["safety_condition"]["num_ellipses"]),
                                   res_m=float(cfg["dataset"]["res_m"]))
    tmr = Timer(device)
    rows, paths, ellipses, tables = [], [], [], []
    for i in range(count):
        scene = dataset.raw_scene(i)
        occ, sdf = np.asarray(scene["occupancy"], np.float32), np.asarray(scene["sdf"], np.float32)
        gt_m = np.asarray(scene["gt_path_m"], np.float64)
        start_m, goal_m = np.asarray(scene["start_m"]), np.asarray(scene["goal_m"])

        t_search, found = tmr(lambda: guidance.search(occ, gt_m, start_m, goal_m))
        path_m = np.asarray(found["path_xy"] if found.get("path_xy") is not None else gt_m,
                            dtype=np.float64)
        occ_t, sdf_t = occ_t_factory(occ), occ_t_factory(sdf)
        path_t = torch.as_tensor(path_m, dtype=torch.float32, device=device)[None]
        t_ellipse, safety = tmr(lambda: gen(occ_t, sdf_t, path_t))

        centers_m = safety["ellipse_center"][0].detach().cpu().numpy()
        axes_m = np.stack([safety["ellipse"][0, :, 2].detach().cpu().numpy(),
                           safety["ellipse"][0, :, 3].detach().cpu().numpy()], -1)
        theta = safety["ellipse"][0, :, 4:].detach().cpu().numpy()
        centers_scene = torch.as_tensor(transform.to_scene(centers_m), dtype=torch.float32, device=device)
        shape4_scene = ellipse_shape4_to_scene(safety["ellipse_shape4"], transform.scale)
        anchors = torch.linspace(0.0, 1.0, centers_scene.shape[0], dtype=torch.float64, device=device)
        path_scene = torch.as_tensor(transform.to_scene(path_m), dtype=torch.float32, device=device)
        curve_scene = torch.as_tensor(transform.to_scene(gt_m), dtype=torch.float32, device=device)

        t_region, (table, info) = tmr(lambda: build_region_table(
            occ_t, centers_scene, shape4_scene[0], anchors, curve_scene=curve_scene,
            gamma_scene=path_scene, config=cfg.get("corridor") or {}, scope="full",
            scene_scale=transform.scale))
        t_pack, pack = tmr(lambda: build_sampled_constraint_pack_from_tables(
            codec, [table], dataset[i]["controls_gt"][None],
            constraint_points=int(cfg["sampling"]["constraint_points"]),
            arc_length_lookup_points=int(cfg["sampling"]["arc_length_lookup_points"]),
            include_box=False, device=device))

        ellipses.append(np.concatenate([transform.to_scene(centers_m),
                                        transform.lengths_to_scene(axes_m), theta], -1))
        paths.append(transform.to_scene(path_m))
        tables.append(table)
        rows.append({"index": i, "search_s": t_search, "ellipse_s": t_ellipse,
                     "region_s": t_region, "pack_s": t_pack,
                     "total_s": t_search + t_ellipse + t_region + t_pack,
                     "cells": int(info.get("local_cells", 0)),
                     "pack_rows": int(pack.num_constraint_rows) if pack is not None else 0})
        print("  geometry %d: search %.3f | ellipse %.3f | region %.3f | pack %.3f | total %.3f s"
              % (i, t_search, t_ellipse, t_region, t_pack, rows[-1]["total_s"]), flush=True)

    # pack everything into the offline-cache layout DualBranchTrainer expects
    max_rows = max(t["cell_A"].shape[0] for t in tables)
    max_faces = max(t["cell_A"].shape[1] for t in tables)
    n = len(tables)
    cell_A = np.zeros((n, max_rows, max_faces, 2), dtype=np.float32)
    cell_b = np.full((n, max_rows, max_faces), np.inf, dtype=np.float32)
    cell_mask = np.zeros((n, max_rows), dtype=bool)
    cell_is_box = np.zeros((n, max_rows), dtype=bool)
    anchors = np.zeros((n, max_rows), dtype=np.float64)
    for i, table in enumerate(tables):
        r, f = table["cell_A"].shape[:2]
        cell_A[i, :r, :f] = table["cell_A"]
        cell_b[i, :r, :f] = table["cell_b"]
        cell_mask[i, :r] = table["cell_mask"]
        cell_is_box[i, :r] = table["cell_is_box"]
        anchors[i, :r] = table["anchors"]
    cache = {"path_scene": np.stack(paths).astype(np.float32),
             "ellipse_scene": np.stack(ellipses).astype(np.float32),
             "cell_A": cell_A, "cell_b": cell_b, "cell_mask": cell_mask,
             "cell_is_box": cell_is_box, "anchors": anchors}
    return rows, cache


def main() -> int:
    parser = base_parser("k4p: geometry + training forward timings")
    parser.add_argument("--data-root", default=os.path.join(PROJECT_ROOT, "data", "_demo_k4p"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--out", default=None)
    opts = parser.parse_args()

    cfg = load(opts)
    cfg["dataset"]["res_m"] = 0.625
    cfg["dataset"]["grid"] = 256
    cfg["dataset"]["real_root"] = opts.data_root
    dataset = RealTrajectoryDataset(cfg, opts.split, limit=opts.batch)
    transform = SceneTransform(0.625 * 256, 256)
    device = torch.device(opts.device)

    pipe = build_pipeline(cfg, device=opts.device, seed=0)
    codec = pipe["codec"]
    if codec.num_controls != dataset[0]["controls_gt"].shape[0]:
        raise SystemExit("config C=%d vs label C=%d" % (codec.num_controls,
                                                       dataset[0]["controls_gt"].shape[0]))
    trainer = DualBranchTrainer(cfg, pipe["model"], pipe["map_condition_encoder"],
                                pipe["map_encoder"], codec, pipe["schedule"],
                                device=opts.device, scene_scale=transform.scale,
                                use_safety=True)

    print("device=%s  batch=%d  grid=256  extent=160 m  C=%d  trainable=%d"
          % (opts.device, opts.batch, codec.num_controls, trainer.trainable_parameter_count()),
          flush=True)

    # ---- 1) geometry (offline path: search + frozen net + cells + pack) ----
    geo_rows, cache = build_geometry(cfg, dataset, opts.batch, pipe["net"], codec,
                                     transform, device)
    geo_mean = {k: float(np.mean([r[k] for r in geo_rows]))
                for k in ("search_s", "ellipse_s", "region_s", "pack_s", "total_s")}
    print("geometry per sample (mean): %s" % json.dumps({k: round(v, 4) for k, v in geo_mean.items()}),
          flush=True)

    # ---- 2) one training forward / step -----------------------------------
    items = [dataset[i] for i in range(opts.batch)]
    index = np.arange(opts.batch)
    tmr = Timer(device)

    with torch.no_grad():
        for _ in range(opts.warmup):
            trainer._compute(items, cache, index)
        t_fwd, _ = tmr(lambda: trainer._compute(items, cache, index), repeat=opts.repeat)

    # single-stage pieces of the forward, measured separately
    occupancy = torch.stack([it["occupancy"] for it in items]).to(device)
    sdf = torch.stack([it["sdf"] for it in items]).to(device)
    cond = torch.stack([it["start_goal"] for it in items]).to(device)
    heading = torch.stack([it["heading"] for it in items]).to(device)
    hv = torch.stack([it["heading_valid"] for it in items]).to(device)
    q0 = torch.stack([it["controls_gt"] for it in items]).to(device)
    with torch.no_grad():
        t_enc, feats = tmr(lambda: pipe["map_encoder"].encode(occupancy, sdf), repeat=opts.repeat)
        t_mc, map_mem = tmr(lambda: pipe["map_condition_encoder"](feats["occ_bottleneck"],
                                                                 feats["sdf_bottleneck"]),
                            repeat=opts.repeat)
        t_ctx, map_ctx = tmr(lambda: pipe["model"].build_map_context(map_mem, cond, heading, hv),
                             repeat=opts.repeat)
        ellipse = torch.as_tensor(cache["ellipse_scene"], dtype=torch.float32, device=device)
        valid = torch.ones(ellipse.shape[0], ellipse.shape[1], dtype=torch.bool, device=device)
        t_saf, (saf_mem, saf_mask) = tmr(lambda: pipe["model"].build_safety_memory(ellipse, valid),
                                        repeat=opts.repeat)
        t = torch.randint(0, int(pipe["schedule"].num_timesteps), (opts.batch,), device=device)
        q_t = pipe["schedule"].q_sample(q0, t, torch.randn_like(q0)).to(q0.dtype)
        t_dit_u, _ = tmr(lambda: pipe["model"](q_t=q_t, t=t, map_context=map_ctx),
                         repeat=opts.repeat)
        t_dit_c, _ = tmr(lambda: pipe["model"](q_t=q_t, t=t, map_context=map_ctx,
                                               safety_memory=saf_mem, safety_mask=saf_mask),
                         repeat=opts.repeat)

    for _ in range(opts.warmup):
        trainer.train_step(items, cache, index)
    t_step, _ = tmr(lambda: trainer.train_step(items, cache, index), repeat=opts.repeat)

    peak = (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else None
    report = {
        "device": opts.device,
        "batch": opts.batch,
        "geometry_per_sample_s": {k: round(v, 4) for k, v in geo_mean.items()},
        "forward_batch_s": {
            "frozen_map_encoder": round(t_enc, 4),
            "map_condition_encoder": round(t_mc, 4),
            "build_map_context": round(t_ctx, 4),
            "build_safety_memory": round(t_saf, 4),
            "dit_map_only_branch": round(t_dit_u, 4),
            "dit_safety_branch": round(t_dit_c, 4),
            "compute_total_forward_plus_loss": round(t_fwd, 4),
            "train_step_forward_backward_update": round(t_step, 4),
        },
        "forward_ms_per_sample": round(1000.0 * t_fwd / opts.batch, 2),
        "step_ms_per_sample": round(1000.0 * t_step / opts.batch, 2),
        "peak_gpu_memory_gb": round(peak, 2) if peak else None,
        "repeat": opts.repeat,
    }
    print(json.dumps(report, indent=2), flush=True)
    if opts.out:
        os.makedirs(os.path.dirname(os.path.abspath(opts.out)), exist_ok=True)
        with open(opts.out, "w", encoding="utf-8") as fh:
            json.dump({"report": report, "geometry": geo_rows}, fh, indent=2)
        print("written:", opts.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
