"""Staged, end-to-end demo of the safety-conditioned inference pipeline.

Runs ONE trained checkpoint through the five stages the design document prescribes,
reporting each stage's artefacts instead of only the final curve:

    stage 1  map-only branch  -> guidance trajectory (no safety features at all)
    stage 2  guidance-field search seeded by that trajectory
    stage 3  safety features: ellipse tokens + local convex corridor (A x <= b pack)
    stage 4  safety-injected diffusion (safety memory on, no correction =>
             safety_mode="safety_only")
    stage 5  CPCG + terminal convex QP (safety_mode="cpcg")

Every number printed here comes out of the production code path; only the plotting
glue is new.  Usage::

    python scripts/demo_full_pipeline.py --checkpoint <ckpt> --indices 0,1,2 --steps 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import PROJECT_ROOT, base_parser, load  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gp_diffuser.evaluation.metrics import trajectory_metrics  # noqa: E402
from gp_diffuser.factory import build_pipeline  # noqa: E402
from gp_diffuser.training.real_dataset import RealTrajectoryDataset  # noqa: E402

FIG_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures", "pipeline_demo")


def _chamfer_m(a, b) -> float:
    d = np.linalg.norm(np.asarray(a)[:, None, :] - np.asarray(b)[None, :, :], axis=-1)
    return float(0.5 * (d.min(axis=1).mean() + d.min(axis=0).mean()))


def _curve_stats(curve_m, item, res_m) -> dict:
    occ = item["occupancy"][0].numpy()
    sdf = item["sdf"][0].numpy()
    m = trajectory_metrics(np.asarray(curve_m), occ, sdf, res_m)
    return {"collision_rate": float(m["collision_rate"]),
            "min_clearance_m": float(m["min_clearance_m"]),
            "length_m": float(m["length_m"]),
            "chamfer_to_gt_m": _chamfer_m(curve_m, item["path_gt_m"].numpy())}


def run_scene(sampler, guidance, transform, dataset, index, steps, seed) -> dict:
    item = dataset[int(index)]
    occ = item["occupancy"][None].clone()
    sdf = item["sdf"][None].clone()
    sg = item["start_goal"][None].clone()
    hd = item["heading"][None].clone() if item.get("heading") is not None else None
    hv = (item["heading_valid"][None].clone()
          if item.get("heading_valid") is not None else None)
    res_m = float(dataset.res_m)
    scale = float(transform.scale)
    report = {"index": int(index)}

    # ---------------- stage 1: map-only => guidance trajectory --------------
    out_map = sampler.sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                             steps=steps, safety_mode="map_only", seed=seed,
                             record_curves=True)
    guide_m = out_map["trajectory_m"][0].numpy()
    report["stage1_map_only"] = _curve_stats(guide_m, item, res_m)
    trace = out_map.get("steps_trace") or []
    x0_curves = [np.asarray(e["x0_scene"]) for e in trace if "x0_scene" in e]
    if len(x0_curves) > 1:
        disp = [float(np.linalg.norm(a - b, axis=-1).mean())
                for a, b in zip(x0_curves, x0_curves[1:])]
        report["stage1_map_only"]["per_step_displacement_scene_mean"] = disp
        report["stage1_map_only"]["per_step_displacement_px_mean"] = [
            float(d * scale) for d in disp]
    report["stage1_map_only"]["num_forwards"] = int(out_map["frames"]["num_forwards"])

    # ---------------- stage 2: guidance-field search ------------------------
    occ_np = occ[0, 0].numpy()
    start_m = transform.to_meters(sg[0, 0].numpy())
    goal_m = transform.to_meters(sg[0, 1].numpy())
    found = guidance.search(occ_np, guide_m, start_m, goal_m)
    path = found.get("path_xy")
    report["stage2_search"] = {k: found.get(k) for k in
                               ("status", "fallback", "sources", "expanded", "valid",
                                "endpoint_projected", "path_verified")}
    report["stage2_search"]["path_points"] = 0 if path is None else int(len(path))

    # ---------------- stage 3: ellipse tokens + convex corridor pack --------
    features = sampler.map_encoder.encode(occ.to(sampler.device), sdf.to(sampler.device))
    curve_scene = out_map["trajectory_scene"][0:1]
    raw, search_stats = sampler._generate_safety(
        occ.to(sampler.device), sdf.to(sampler.device), features, curve_scene,
        sg.to(sampler.device), dict(sampler.cfg.get("corridor") or {}))
    if raw is None:
        report["stage3_safety_features"] = {"error": "safety generation returned None"}
        return report
    centers_m = transform.to_meters(raw["ellipse_centers_scene"][0].detach().cpu().numpy())
    axes_m = transform.lengths_to_meters(
        raw["ellipse"][0][:, 2:4].detach().cpu().numpy())
    pack = raw["pack"]
    pack_local = raw["pack_local"]
    report["stage3_safety_features"] = {
        "ellipse_tokens": int(raw["ellipse"].shape[1]),
        "ellipse_center_span_m": float(np.linalg.norm(
            centers_m.max(axis=0) - centers_m.min(axis=0))),
        "ellipse_axis_m_mean": [float(axes_m[:, 0].mean()), float(axes_m[:, 1].mean())],
        "ellipse_axis_m_min_max": [float(axes_m.min()), float(axes_m.max())],
        "local_convex_cells": int(raw["local_cells"][0].item()),
        "corridor_spans": (list(raw["corridor_info"].get("spans") or [])[:6]),
        "constraint_samples": (None if pack is None
                               else int(pack.basis.shape[1])),
        "constraint_inequalities": (None if pack is None
                                    else int(pack.num_constraint_rows)),
        "constraint_samples_local": (None if pack_local is None
                                     else int(pack_local.sample_mask.sum())),
        "constraint_inequalities_local": (None if pack_local is None
                                          else int(pack_local.num_constraint_rows)),
        "constraint_pack_warnings": ([] if pack is None
                                     else list(pack.summary()["warnings"])),
        "timing_s": {k: round(v, 3) for k, v in raw["timing"].items()},
        "search_stats": search_stats[0],
    }

    # ---------------- stage 4: safety-injected diffusion, no ALM ------------
    out_noalm = sampler.sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                               steps=steps, safety_mode="no_alm", seed=seed)
    noalm_m = out_noalm["trajectory_m"][0].numpy()
    report["stage4_safety_injected"] = _curve_stats(noalm_m, item, res_m)
    report["stage4_safety_injected"]["safety_generated_once"] = bool(
        out_noalm["frames"]["safety_generated_once"])

    # ---------------- stage 5: CPCG + terminal convex QP --------------------
    out_full = sampler.sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                              steps=steps, safety_mode="cpcg", seed=seed)
    full_m = out_full["trajectory_m"][0].numpy()
    report["stage5_cpcg_qp"] = _curve_stats(full_m, item, res_m)
    stats = out_full["guidance_stats"]
    report["stage5_cpcg_qp"]["guidance_steps"] = len(stats)
    report["stage5_cpcg_qp"]["qp_status"] = out_full["final_qp_status"]
    report["stage5_cpcg_qp"]["qp_certified"] = bool(
        out_full.get("corridor_certified_all", False))
    report["stage5_cpcg_qp"]["projection_correction_m"] = float(
        out_full.get("final_projection_correction_m", 0.0))
    if stats:
        report["stage5_cpcg_qp"]["violation_before_mean"] = float(
            np.mean([s["max_violation_before"] for s in stats]))
        report["stage5_cpcg_qp"]["violation_after_mean"] = float(
            np.mean([s["max_violation_after"] for s in stats]))
        report["stage5_cpcg_qp"]["delta_rms_scene_mean"] = float(
            np.mean([s["delta_rms_scene"] for s in stats]))
        report["stage5_cpcg_qp"]["delta_max_m_mean"] = float(
            np.mean([s["delta_max_m"] for s in stats]))
    report["final"] = {"guide_to_final_m": _chamfer_m(
        out_full["guide_trajectory_m"][0].numpy(), full_m)}
    return report


def plot_scene(sampler, guidance, transform, dataset, index, steps, seed, path_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    item = dataset[int(index)]
    occ = item["occupancy"][None].clone()
    sdf = item["sdf"][None].clone()
    sg = item["start_goal"][None].clone()
    hd = item["heading"][None].clone() if item.get("heading") is not None else None
    hv = (item["heading_valid"][None].clone()
          if item.get("heading_valid") is not None else None)
    res_m = float(dataset.res_m)
    raw_scene = dataset.raw_scene(int(index))
    occ_img = np.asarray(raw_scene["occupancy"])
    occ_img = occ_img[:, :, 0] if occ_img.ndim == 3 else occ_img

    out_map = sampler.sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                             steps=steps, safety_mode="map_only", seed=seed)
    guide_m = out_map["trajectory_m"][0].numpy()
    occ_np = occ[0, 0].numpy()
    start_m = transform.to_meters(sg[0, 0].numpy())
    goal_m = transform.to_meters(sg[0, 1].numpy())
    found = guidance.search(occ_np, guide_m, start_m, goal_m)
    features = sampler.map_encoder.encode(occ.to(sampler.device), sdf.to(sampler.device))
    raw, _ = sampler._generate_safety(
        occ.to(sampler.device), sdf.to(sampler.device), features,
        out_map["trajectory_scene"][0:1], sg.to(sampler.device),
        sampler._corridor_cfg())
    out_noalm = sampler.sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                               steps=steps, safety_mode="safety_only", seed=seed)
    out_full = sampler.sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                              steps=steps, safety_mode="cpcg", seed=seed)

    fig, axes = plt.subplots(1, 5, figsize=(26, 5.4))
    gt = raw_scene["gt_path_m"]

    def base(ax, title):
        ax.imshow(occ_img, origin="lower", cmap="Greys",
                  extent=[0, occ_img.shape[1] * res_m, 0, occ_img.shape[0] * res_m],
                  alpha=0.85)
        ax.plot(gt[:, 0], gt[:, 1], "-", color="#2ca02c", lw=1.4, alpha=0.7, label="GT")
        ax.plot(start_m[0], start_m[1], "o", color="black", ms=5)
        ax.plot(goal_m[0], goal_m[1], "s", color="black", ms=5)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, occ_img.shape[1] * res_m)
        ax.set_ylim(0, occ_img.shape[0] * res_m)
        ax.tick_params(labelsize=7)

    base(axes[0], "1. map-only (no safety) = guidance")
    axes[0].plot(guide_m[:, 0], guide_m[:, 1], "-", color="#1f77b4", lw=2.2,
                 label="guidance")
    axes[0].legend(fontsize=8)

    base(axes[1], "2. + guidance-field search")
    axes[1].plot(guide_m[:, 0], guide_m[:, 1], "--", color="#1f77b4", lw=1.2,
                 alpha=0.6, label="guidance (seed)")
    if found.get("path_xy") is not None:
        p = np.asarray(found["path_xy"])
        axes[1].plot(p[:, 0], p[:, 1], "-", color="#d62728", lw=2.2, label="searched path")
    axes[1].legend(fontsize=8)

    base(axes[2], "3. + ellipse tokens & convex cells")
    if raw is not None:
        centers_m = transform.to_meters(raw["ellipse_centers_scene"][0].detach().cpu().numpy())
        axes_m = transform.lengths_to_meters(raw["ellipse"][0][:, 2:4].detach().cpu().numpy())
        cos2t = raw["ellipse"][0][:, 4].detach().cpu().numpy()
        sin2t = raw["ellipse"][0][:, 5].detach().cpu().numpy()
        theta_deg = np.degrees(0.5 * np.arctan2(sin2t, cos2t))
        for c, ax_m, th in zip(centers_m, axes_m, theta_deg):
            axes[2].add_patch(Ellipse((c[0], c[1]), width=2 * ax_m[0], height=2 * ax_m[1],
                                      angle=th, fill=False, ec="#9467bd", lw=0.9,
                                      alpha=0.75))
        axes[2].plot(centers_m[:, 0], centers_m[:, 1], ".", color="#9467bd", ms=3)
    axes[2].legend(["ellipse corridor (64 tokens)"], fontsize=8)

    base(axes[3], "4. safety-injected diffusion (no correction)")
    axes[3].plot(out_noalm["guide_trajectory_m"][0].numpy()[:, 0],
                 out_noalm["guide_trajectory_m"][0].numpy()[:, 1], "--",
                 color="#1f77b4", lw=1.2, alpha=0.6, label="guidance")
    axes[3].plot(out_noalm["trajectory_m"][0].numpy()[:, 0],
                 out_noalm["trajectory_m"][0].numpy()[:, 1], "-", color="#ff7f0e",
                 lw=2.2, label="safety-only final")
    axes[3].legend(fontsize=8)

    base(axes[4], "5. + CPCG + terminal QP")
    axes[4].plot(out_full["controls_unprojected"][0].numpy()[:, 0],
                 out_full["controls_unprojected"][0].numpy()[:, 1], ":",
                 color="#ff7f0e", lw=1.3, label="before QP")
    axes[4].plot(out_full["trajectory_m"][0].numpy()[:, 0],
                 out_full["trajectory_m"][0].numpy()[:, 1], "-", color="#9467bd",
                 lw=2.4, label="CPCG + QP final")
    axes[4].legend(fontsize=8)

    fig.suptitle("val sample %d | staged pipeline | steps=%d" % (index, steps), fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path_png), exist_ok=True)
    fig.savefig(path_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path_png


def main() -> int:
    parser = base_parser("staged end-to-end pipeline demo")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--indices", default="0,1,2")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/reports/pipeline_demo.json")
    parser.add_argument("--figure-dir", default=FIG_DIR)
    opts = parser.parse_args()

    cfg = load(opts)
    dataset = RealTrajectoryDataset(cfg, opts.split)
    pipe = build_pipeline(cfg, device=opts.device, checkpoint=opts.checkpoint)
    sampler = pipe["sampler"]
    transform, guidance = pipe["transform"], pipe["guidance"]
    print("checkpoint:", opts.checkpoint, "| device:", opts.device,
          "| steps:", opts.steps, "| res_m:", dataset.res_m, flush=True)

    indices = [int(s) for s in opts.indices.split(",") if s.strip()]
    reports = []
    for index in indices:
        report = run_scene(sampler, guidance, transform, dataset, index,
                           opts.steps, opts.seed)
        reports.append(report)
        png = os.path.join(opts.figure_dir, "staged_val%03d.png" % index)
        plot_scene(sampler, guidance, transform, dataset, index, opts.steps,
                   opts.seed, png)
        report["figure"] = os.path.relpath(png, PROJECT_ROOT)
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)

    out_path = os.path.join(PROJECT_ROOT, opts.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"checkpoint": opts.checkpoint, "split": opts.split,
                   "steps": opts.steps, "seed": opts.seed,
                   "scenes": reports}, fh, indent=2, ensure_ascii=False)
    print("written:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
