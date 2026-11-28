"""Where does the wiggle come from: the network, CPCG, or the terminal QP?

For each sample the same run is sampled once (cpcg, record_curves) and three
curves are compared against GT:

    fwd0 raw      network's clean prediction before any geometry exists
    fwd1 guided   same step after the first CPCG correction
    FINAL         after the terminal convex QP

Roughness = mean ||p[i-1] - 2 p[i] + p[i+1]|| in metres (second difference of the
128-point curve), plus the 95th percentile and the mean turning angle.  The GT
curve gives the reference value for the same data.

Usage::

    python scripts/diag_k4p_smoothness.py --split test --samples 12
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

from gp_diffuser.factory import build_pipeline  # noqa: E402
from gp_diffuser.training.real_dataset import RealTrajectoryDataset  # noqa: E402


def roughness(pts: np.ndarray) -> dict:
    """Second-difference / turning-angle statistics of a 2-D polyline (metres)."""
    p = np.asarray(pts, dtype=np.float64)
    if len(p) < 3:
        return {"d2_mean_m": float("nan"), "d2_p95_m": float("nan"),
                "turn_deg": float("nan"), "spacing_m": float("nan")}
    d2 = np.linalg.norm(p[:-2] - 2.0 * p[1:-1] + p[2:], axis=-1)
    seg = p[1:] - p[:-1]
    step = np.linalg.norm(seg, axis=-1).clip(1e-9)
    unit = seg / step[:, None]
    cosang = np.clip((unit[:-1] * unit[1:]).sum(-1), -1.0, 1.0)
    return {"d2_mean_m": float(d2.mean()), "d2_p95_m": float(np.percentile(d2, 95)),
            "turn_deg": float(np.degrees(np.arccos(cosang)).mean()),
            "spacing_m": float(step.mean())}


def main() -> int:
    parser = base_parser("k4p: smoothness attribution (network vs CPCG vs QP)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", default=os.path.join(PROJECT_ROOT, "outputs", "reports",
                                                      "diag_k4p_smoothness.json"))
    opts = parser.parse_args()

    cfg = load(opts)
    print("loss weights: %s" % json.dumps(cfg.get("loss_weights") or {}), flush=True)
    pipe = build_pipeline(cfg, device=opts.device, checkpoint=opts.checkpoint)
    dataset = RealTrajectoryDataset(cfg, opts.split)
    sampler, tf = pipe["sampler"], pipe["transform"]
    steps = int(opts.steps or cfg["sampling"]["steps"])

    rows = []
    for index in range(min(int(opts.samples), len(dataset))):
        item = dataset[index]
        with torch.no_grad():
            out = sampler.sample(item["occupancy"][None], item["sdf"][None],
                                 item["start_goal"][None], heading=item["heading"][None],
                                 heading_valid=item["heading_valid"][None],
                                 steps=steps, safety_mode="cpcg", seed=opts.seed,
                                 record_curves=True)
        trace = out["steps_trace"]
        f0 = trace[0]["x0_scene"][0]
        f1_raw = trace[min(1, len(trace) - 1)]["x0_scene"][0]
        f1_gui = trace[min(1, len(trace) - 1)].get("x0_guided_scene")
        f1_gui = f1_raw if f1_gui is None else f1_gui[0]
        last_gui = trace[-1].get("x0_guided_scene")
        last_gui = trace[-1]["x0_scene"][0] if last_gui is None else last_gui[0]
        final = out["trajectory_m"][0].detach().cpu().numpy()
        gt = np.asarray(item["path_gt_m"])
        row = {"index": index, "gt": roughness(gt),
               "fwd0_raw": roughness(tf.to_meters(f0)),
               "fwd1_raw": roughness(tf.to_meters(f1_raw)),
               "fwd1_guided": roughness(tf.to_meters(f1_gui)),
               "last_guided": roughness(tf.to_meters(last_gui)),
               "final_qp": roughness(final)}
        rows.append(row)
        print("sample %2d | d2_mean m: gt %.4f | fwd0 %.4f | fwd1 %.4f | fwd1_cpcg %.4f | "
              "last_cpcg %.4f | FINAL %.4f" % (index, row["gt"]["d2_mean_m"],
                                              row["fwd0_raw"]["d2_mean_m"],
                                              row["fwd1_raw"]["d2_mean_m"],
                                              row["fwd1_guided"]["d2_mean_m"],
                                              row["last_guided"]["d2_mean_m"],
                                              row["final_qp"]["d2_mean_m"]), flush=True)

    keys = ("gt", "fwd0_raw", "fwd1_raw", "fwd1_guided", "last_guided", "final_qp")
    summary = {k: {m: round(float(np.mean([r[k][m] for r in rows])), 5)
                   for m in ("d2_mean_m", "d2_p95_m", "turn_deg", "spacing_m")}
               for k in keys}
    print("\n=== mean over %d samples ===" % len(rows))
    for k in keys:
        print("%-12s d2_mean %.4f m | d2_p95 %.4f | turn %.2f deg | spacing %.2f m"
              % (k, summary[k]["d2_mean_m"], summary[k]["d2_p95_m"],
                 summary[k]["turn_deg"], summary[k]["spacing_m"]))
    os.makedirs(os.path.dirname(os.path.abspath(opts.out)), exist_ok=True)
    with open(opts.out, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "rows": rows,
                   "loss_weights": cfg.get("loss_weights")}, fh, indent=2)
    print("written:", opts.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
