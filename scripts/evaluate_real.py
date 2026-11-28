"""Evaluate a checkpoint on the REAL dataset (losses + closed-loop inference).

Two independent measurements:

1. teacher-forced losses via ``DualBranchTrainer.eval_step`` (no gradient) —
   comparable with the training curves;
2. closed-loop sampling (DDIM -> convex corridor -> ALM -> B-spline) on real
   val scenes, scored with the same metrics used for the GT audit: collision
   rate, min clearance, length error, heading error, ALM acceptance.

Usage::

    python scripts/evaluate_real.py --checkpoint outputs/checkpoints/geopath_v1_c64.pt \
        --split val --loss-samples 256 --sample-samples 32 --steps 20
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

from gp_diffuser.evaluation.metrics import trajectory_metrics  # noqa: E402
from gp_diffuser.factory import build_pipeline  # noqa: E402
from gp_diffuser.training.dual_branch_trainer import DualBranchTrainer  # noqa: E402
from gp_diffuser.training.real_dataset import RealTrajectoryDataset  # noqa: E402
from gp_diffuser.training.safety_cache import load_safety_cache  # noqa: E402


def _stats(values) -> dict:
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return {}
    return {"mean": float(arr.mean()), "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)), "max": float(arr.max()),
            "min": float(arr.min())}


def main() -> int:
    parser = base_parser("evaluate on the real dataset")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--loss-samples", type=int, default=256)
    parser.add_argument("--sample-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20, help="DDIM steps per sample")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/reports/real_eval.json")
    opts = parser.parse_args()

    cfg = load(opts)
    pipe = build_pipeline(cfg, device=opts.device, checkpoint=opts.checkpoint)
    dataset = RealTrajectoryDataset(cfg, opts.split)
    cache = load_safety_cache(cfg, opts.split, dataset=dataset)
    if cache is None:
        print("no safety cache for split %r" % opts.split)
        return 2
    report = {"checkpoint": opts.checkpoint, "split": opts.split,
              "dataset": dataset.summary(),
              "num_controls": int(pipe["codec"].num_controls), "device": opts.device}

    # ---- 1. teacher-forced losses -----------------------------------------
    trainer = DualBranchTrainer(cfg, pipe["model"], pipe["map_condition_encoder"],
                                pipe["map_encoder"], pipe["codec"], pipe["schedule"],
                                device=opts.device,
                                scene_scale=pipe["transform"].scale)
    torch.manual_seed(opts.seed)
    rows = []
    for start in range(0, min(opts.loss_samples, len(dataset)), opts.batch_size):
        idx = np.arange(start, min(start + opts.batch_size, len(dataset)))
        if len(idx) < 2:
            break
        rows.append(trainer.eval_step([dataset[int(i)] for i in idx], cache, idx))
    if rows:
        report["losses"] = {k: round(float(np.mean([r[k] for r in rows])), 5)
                            for k in rows[0] if isinstance(rows[0][k], (int, float))}
        report["loss_batches"] = len(rows)

    # ---- 2. closed-loop sampling ------------------------------------------
    sampler = pipe["sampler"]
    records = []
    started = time.time()
    for i in range(min(opts.sample_samples, len(dataset))):
        item = dataset[i]
        out = sampler.sample(item["occupancy"][None], item["sdf"][None],
                             item["start_goal"][None], heading=item["heading"][None],
                             heading_valid=item["heading_valid"][None],
                             steps=opts.steps, seed=opts.seed)
        traj_m = out["trajectory_m"][0].numpy()
        metrics = trajectory_metrics(traj_m, item["occupancy"][0].numpy(),
                                     item["sdf"][0].numpy(), dataset.res_m)
        gt_len = float(torch.linalg.norm(torch.diff(item["path_gt_m"], dim=0),
                                         dim=-1).sum())
        records.append({
            "index": i, "valid": bool(out["valid"]),
            "collision_rate": float(metrics["collision_rate"]),
            "min_clearance_m": float(metrics["min_clearance_m"]),
            "length_m": float(metrics["length_m"]), "gt_length_m": gt_len,
            "length_error_m": abs(float(metrics["length_m"]) - gt_len),
            "heading_error_deg_max": None if out["heading_error_deg"] is None
            else float(out["heading_error_deg"].max()),
            "guidance_steps": int(out.get("guidance_steps", 0)),
            "qp_status": (out["final_qp_status"][0]
                          if out.get("final_qp_status") else "not_run"),
            "qp_iterations": (int(out["final_qp_iterations"][0])
                              if out.get("final_qp_iterations") else 0),
            "projection_correction_m": float(
                out.get("final_projection_correction_m", 0.0)),
            "corridor_certified": bool(out.get("corridor_certified_all", False)),
            "dense_collision_fraction": float(out.get("dense_collision_fraction", 0.0)),
            "search_failed": bool(out.get("search_failed", False)),
            "local_cells": int(out["corridor_info"].get("local_cells", 0)),
            "seconds": float(out["timing"]["total_s"]),
        })
    report["inference"] = {
        "samples": len(records),
        "valid_rate": float(np.mean([r["valid"] for r in records])) if records else 0.0,
        "collision_free_rate": float(np.mean([r["collision_rate"] == 0 for r in records]))
        if records else 0.0,
        "corridor_certified_rate": float(np.mean([r["corridor_certified"]
                                                  for r in records])) if records else 0.0,
        "qp_solved_rate": float(np.mean([r["qp_status"] == "solved"
                                         for r in records])) if records else 0.0,
        "projection_correction_m": _stats([r["projection_correction_m"]
                                           for r in records]),
        "collision_rate": _stats([r["collision_rate"] for r in records]),
        "min_clearance_m": _stats([r["min_clearance_m"] for r in records]),
        "length_error_m": _stats([r["length_error_m"] for r in records]),
        "heading_error_deg_max": _stats([r["heading_error_deg_max"] for r in records]),
        "seconds_per_sample": float(np.mean([r["seconds"] for r in records]))
        if records else 0.0,
        "elapsed_s": round(time.time() - started, 1),
    }
    report["records"] = records

    out_path = os.path.join(PROJECT_ROOT, opts.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print(json.dumps({k: report[k] for k in ("losses", "inference") if k in report},
                     indent=2, ensure_ascii=False))
    print("written:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
