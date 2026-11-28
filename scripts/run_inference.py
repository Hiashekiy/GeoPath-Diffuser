"""Full inference + evaluation over the test scenes.

Runs the pipeline in the three configurations the design document asks for

* ``map_only``    : the Map-only branch at every step (no search, no ellipse,
                    no CPCG, no QP)
* ``safety_only`` : safety cross-attention injected, CPCG and QP disabled
                    (legacy alias ``no_alm``)
* ``cpcg``        : safety cross-attention + clean-prediction constraint guidance
                    + ONE terminal convex QP (legacy alias ``full``)

and reports collision rate, clearance, length, smoothness, corridor violation,
the per-step guidance effect and the terminal QP diagnostics.  Figures and a
JSON/Markdown report go to ``outputs/``.

Example (the default 8 executed reverse steps, current checkpoint, no retraining)::

    python scripts/run_inference.py --dataset real --config configs/c64_longpath.yaml \
        --samples 8 --modes map_only,safety_only,cpcg --checkpoint <local checkpoint>
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import PROJECT_ROOT, base_parser, load  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gp_diffuser.evaluation.metrics import (aggregate, corridor_violation_m,
                                            trajectory_metrics)  # noqa: E402
from gp_diffuser.evaluation.visualize import render_sample  # noqa: E402
from gp_diffuser.factory import build_pipeline, describe  # noqa: E402
from gp_diffuser.diffusion.sampler_conditional_dit import MODES  # noqa: E402
from gp_diffuser.training.dataset import SyntheticTrajectoryDataset  # noqa: E402


def _mean(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else float("nan")


def run_mode(pipe, dataset, indices, cfg, mode: str, steps: int, seed: int) -> dict:
    items = [dataset[i] for i in indices]
    occupancy = torch.stack([it["occupancy"] for it in items])
    sdf = torch.stack([it["sdf"] for it in items])
    start_goal = torch.stack([it["start_goal"] for it in items])
    heading = torch.stack([it["heading"] for it in items])
    heading_valid = torch.stack([it["heading_valid"] for it in items])

    res_m = float(cfg["dataset"]["res_m"])
    scene_scale = pipe["transform"].scale
    scene_offset = pipe["transform"].offset
    rows, outputs = [], []
    for b in range(len(items)):
        t0 = time.perf_counter()
        out = pipe["sampler"].sample(
            occupancy[b:b + 1], sdf[b:b + 1], start_goal[b:b + 1],
            heading=heading[b:b + 1], heading_valid=heading_valid[b:b + 1],
            steps=steps, safety_mode=mode, seed=seed + b, record_curves=True)
        elapsed = time.perf_counter() - t0
        traj_m = out["trajectory_m"][0].detach().cpu().numpy()
        metrics = trajectory_metrics(traj_m, occupancy[b, 0].numpy(),
                                     sdf[b, 0].numpy(), res_m)
        # local (obstacle) constraints only: the map-box cells would always be
        # satisfied and would hide the actual corridor violation
        metrics["corridor_violation_m"] = corridor_violation_m(
            traj_m, out.get("pack_local"), scene_scale, offset=scene_offset)
        metrics["corridor_violation_full_m"] = corridor_violation_m(
            traj_m, out.get("pack"), scene_scale, offset=scene_offset)
        metrics["final_corridor_violation_m"] = float(
            out.get("final_qp_max_violation_scene", 0.0) * scene_scale)
        metrics["guidance_steps"] = int(out.get("guidance_steps", 0))
        metrics["guidance_max_violation_before_m"] = float(
            _mean([g["max_violation_before"] for g in out.get("guidance_stats", [])])
            * scene_scale)
        metrics["guidance_max_violation_after_m"] = float(
            _mean([g["max_violation_after"] for g in out.get("guidance_stats", [])])
            * scene_scale)
        metrics["guidance_delta_max_m"] = _mean(
            [g["delta_max_m"] for g in out.get("guidance_stats", [])])
        metrics["final_projection_correction_m"] = float(
            out.get("final_projection_correction_m", 0.0))
        metrics["final_qp_time_s"] = float(out.get("final_qp_time_s", 0.0))
        metrics["corridor_certified"] = bool(
            bool(out.get("corridor_certified_all", False)))
        metrics["qp_status"] = (out["final_qp_status"][0]
                                if out.get("final_qp_status") else "not_run")
        metrics["qp_iterations"] = (int(out["final_qp_iterations"][0])
                                    if out.get("final_qp_iterations") else 0)
        metrics["search_failed"] = bool(out["search_failed"][0])
        if out.get("heading_error_deg") is not None:
            metrics["heading_error_max_deg"] = float(
                out["heading_error_deg"].max().item())
        else:
            metrics["heading_error_max_deg"] = float("nan")
        metrics["inference_s"] = elapsed
        metrics["index"] = int(indices[b])
        metrics["search_status"] = (out["search_stats"][0]["status"]
                                    if out["search_stats"] else None)
        metrics["safety_generated"] = bool(out["frames"]["safety_generated_once"])
        metrics["local_cells"] = int(out["corridor_info"].get("local_cells", 0)) \
            if isinstance(out["corridor_info"], dict) else 0
        rows.append(metrics)

        # figure data
        scene = dataset.raw_scene(int(indices[b]))
        sample = {
            "occupancy_m": scene["occupancy"],
            "res_m": res_m,
            "transform": pipe["transform"],
            "gt_path_scene": items[b]["path_gt_scene"].numpy(),
            "start_goal_scene": items[b]["start_goal"].numpy(),
            "guide_path_m": (None if out["guide_trajectory_m"] is None
                             else out["guide_trajectory_m"][0].detach().cpu().numpy()),
            "scene_scale": scene_scale,
        }
        outputs.append((out, sample))
    return {"mode": mode, "rows": rows, "outputs": outputs,
            "aggregate": aggregate(rows), "timing": outputs[0][0]["timing"]}


def build_dataset(cfg, name: str, split: str):
    """``synthetic`` (self-contained) or ``real`` (geopath_v1 processed samples)."""
    if name == "real":
        from gp_diffuser.training.real_dataset import RealTrajectoryDataset
        return RealTrajectoryDataset(cfg, split)
    if name != "synthetic":
        raise ValueError("unknown --dataset %r" % name)
    return SyntheticTrajectoryDataset(cfg, split)


def main() -> int:
    parser = base_parser("inference + evaluation")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--dataset", default="synthetic",
                        choices=["synthetic", "real"],
                        help="real = data/geopath_v1 processed samples (needs the "
                             "c64_longpath config for real checkpoints)")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--steps", type=int, default=8,
                        help="executed reverse steps (default 8 = DDIM subsampling "
                             "of the trained 1000-step table; no retraining)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--modes", default="map_only,safety_only,cpcg",
                        help="comma separated subset of %s (legacy aliases "
                             "no_alm/full are accepted)" % (MODES,))
    opts = parser.parse_args()

    cfg = load(opts)
    pipe = build_pipeline(cfg, device=opts.device, checkpoint=opts.checkpoint)
    dataset = build_dataset(cfg, opts.dataset, opts.split)
    indices = list(range(min(opts.samples, len(dataset))))
    modes = [m.strip() for m in opts.modes.split(",") if m.strip()]

    report = {"config_summary": describe(pipe), "split": opts.split,
              "dataset": opts.dataset, "indices": indices, "steps": opts.steps,
              "checkpoint": opts.checkpoint, "modes": {}}
    figures_dir = cfg["evaluation"]["figure_dir"]
    for mode in modes:
        print("== mode %s" % mode, flush=True)
        result = run_mode(pipe, dataset, indices, cfg, mode, opts.steps, opts.seed)
        report["modes"][mode] = {"aggregate": result["aggregate"],
                                 "rows": result["rows"],
                                 "timing": result["timing"]}
        agg = result["aggregate"]
        print("   collision_rate=%.4f success=%.2f heading_err=%.1f deg "
              "corridor_viol=%.4f m guidance=%d steps qp=%s cert=%s"
              % (agg.get("collision_rate", 0.0), agg.get("success_rate", 0.0),
                 agg.get("heading_error_max_deg", float("nan")),
                 agg.get("corridor_violation_m", 0.0),
                 int(agg.get("guidance_steps", 0)),
                 result["rows"][0].get("qp_status") if result["rows"] else "n/a",
                 result["rows"][0].get("corridor_certified")
                 if result["rows"] else "n/a"))
        for out, sample in result["outputs"]:
            idx = int(sample.get("index", 0))
            name = "%s_%03d_%02d.png" % (mode, idx, opts.steps)
            path = os.path.join(PROJECT_ROOT, figures_dir, name)
            render_sample(out, sample, path=path,
                          title="%s (split=%s, index=%d, steps=%d)"
                                % (mode, opts.split, idx, opts.steps))

    os.makedirs(os.path.join(PROJECT_ROOT, "outputs", "reports"), exist_ok=True)
    out_json = os.path.join(PROJECT_ROOT, "outputs", "reports", "inference_report.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=float)

    lines = ["# Inference report (%s split, %d steps)" % (opts.split, opts.steps), "",
             "| mode | collision | success | heading err | corridor viol (m) | "
             "CPCG steps | QP status | QP cert | QP corr (m) | QP (s) | infer (s) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for mode in modes:
        agg = report["modes"][mode]["aggregate"]
        rows = report["modes"][mode]["rows"]
        lines.append(
            "| %s | %.4f | %.2f | %.1f | %.4f | %d | %s | %s | %.4f | %.3f | %.2f |"
            % (mode, agg.get("collision_rate", 0.0), agg.get("success_rate", 0.0),
               agg.get("heading_error_max_deg", float("nan")),
               agg.get("corridor_violation_m", 0.0),
               int(agg.get("guidance_steps", 0)),
               (rows[0].get("qp_status") if rows else "n/a"),
               ("yes" if (rows and rows[0].get("corridor_certified")) else "no"),
               agg.get("final_projection_correction_m", 0.0),
               agg.get("final_qp_time_s", 0.0),
               agg.get("inference_s", 0.0)))
    md_path = os.path.join(PROJECT_ROOT, "outputs", "reports", "inference_report.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("written:", out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
