"""Where do the big unnecessary detours come from?  (not jitter - route shape)

For each sample the FINAL curve is compared with the three things that could own
its shape:

    GT curve          the demonstrated route
    guide path        the Astar-Diffusion search path (= the corridor spine)
    corridor faces    ``A P <= b`` of the frozen sample pack

Metrics per sample:

    turn_gt / turn_final     total |dtheta| (degrees) - "东拐西拐" is this number
    dev_guide_m              mean / max distance from the final curve to the guide
    dev_gt_m                 mean distance from the final curve to GT
    slack_m                  mean of A P - b over the ACTIVE constraint rows:
                             0  -> the curve rides the corridor boundary
                             <0 -> it sits strictly inside the corridor
    pred_dev_guide_m         the same distance for the RAW network prediction
                             (before CPCG/QP), to separate "network wander" from
                             "geometry pulled it out"

Usage::

    python scripts/diag_k4p_drift.py --split test --samples 8
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


def total_turn_deg(pts: np.ndarray) -> float:
    p = np.asarray(pts, dtype=np.float64)
    seg = np.diff(p, axis=0)
    step = np.linalg.norm(seg, axis=-1).clip(1e-9)
    unit = seg / step[:, None]
    cosang = np.clip((unit[:-1] * unit[1:]).sum(-1), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)).sum())


def dev_to_polyline(points: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Distance from every point of ``points`` to the polyline ``ref``."""
    pts = np.asarray(points, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    a, b = ref[:-1], ref[1:]           # [S,2]
    ab = b - a
    denom = (ab * ab).sum(-1).clip(1e-12)
    out = []
    for p in pts:
        t = np.clip(((p - a) * ab).sum(-1) / denom, 0.0, 1.0)
        proj = a + t[:, None] * ab
        out.append(np.linalg.norm(proj - p, axis=-1).min())
    return np.asarray(out)


def main() -> int:
    parser = base_parser("k4p: detour attribution (route shape)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", default=os.path.join(PROJECT_ROOT, "outputs", "reports",
                                                      "diag_k4p_drift.json"))
    opts = parser.parse_args()

    cfg = load(opts)
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
        gt = np.asarray(item["path_gt_m"], dtype=np.float64)
        final = out["trajectory_m"][0].detach().cpu().numpy().astype(np.float64)
        guide = (None if out.get("search_path_m") is None
                 else out["search_path_m"][0].detach().cpu().numpy().astype(np.float64))
        pred0 = tf.to_meters(out["steps_trace"][0]["x0_scene"][0])
        last_gui = out["steps_trace"][-1].get("x0_guided_scene")
        last_gui = (pred0 if last_gui is None else tf.to_meters(last_gui[0]))

        pack = out.get("pack")
        slack = None
        if pack is not None:
            basis = pack.basis[0].detach().cpu().numpy()          # [M,C]
            A = pack.sample_A[0].detach().cpu().numpy()           # [M,F,2]
            bb = pack.sample_b[0].detach().cpu().numpy()          # [M,F]
            smask = pack.sample_mask[0].detach().cpu().numpy()
            fmask = pack.face_mask[0].detach().cpu().numpy()
            q = out["controls"][0].detach().cpu().numpy().astype(np.float64)
            pts = basis @ q                                        # [M,2] scene
            g = np.einsum("mfk,mk->mf", A, pts) - bb
            active = smask[:, None] & fmask
            gm = np.where(active, g, -np.inf).max(axis=1)
            keep = np.isfinite(gm)
            slack = float(gm[keep].mean()) * tf.scale if keep.any() else None

        dev_guide = dev_to_polyline(final, guide) if guide is not None else None
        pred_dev = dev_to_polyline(pred0, guide) if guide is not None else None
        row = {"index": index,
               "turn_gt": total_turn_deg(gt), "turn_final": total_turn_deg(final),
               "turn_guide": (total_turn_deg(guide) if guide is not None else None),
               "turn_pred0": total_turn_deg(pred0),
               "turn_last_cpcg": total_turn_deg(last_gui),
               "dev_guide_mean_m": float(dev_guide.mean()) if dev_guide is not None else None,
               "dev_guide_max_m": float(dev_guide.max()) if dev_guide is not None else None,
               "pred_dev_guide_mean_m": float(pred_dev.mean()) if pred_dev is not None else None,
               "dev_gt_mean_m": float(dev_to_polyline(final, gt).mean()),
               "slack_m": slack,
               "length_ratio": float(np.linalg.norm(np.diff(final, axis=0), axis=1).sum()
                                     / max(np.linalg.norm(np.diff(gt, axis=0), axis=1).sum(), 1e-9))}
        rows.append(row)
        print("sample %2d | turn: gt %5.0f guide %5.0f pred %5.0f cpcg %5.0f FINAL %5.0f | dev-guide %.2f (pred %.2f) m | "
              "dev-gt %.2f m | slack %s m | len %.3f"
              % (index, row["turn_gt"], row["turn_guide"] or 0.0, row["turn_pred0"],
                 row["turn_last_cpcg"], row["turn_final"], row["dev_guide_mean_m"],
                 row["pred_dev_guide_mean_m"], row["dev_gt_mean_m"],
                 ("%.2f" % slack) if slack is not None else "-", row["length_ratio"]), flush=True)

    keys = [k for k in rows[0] if k != "index" and rows[0][k] is not None]
    summary = {k: round(float(np.mean([r[k] for r in rows if r[k] is not None])), 4)
               for k in keys}
    print("\n=== mean over %d samples ===" % len(rows))
    print(json.dumps(summary, indent=2))
    os.makedirs(os.path.dirname(os.path.abspath(opts.out)), exist_ok=True)
    with open(opts.out, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "rows": rows}, fh, indent=2)
    print("written:", opts.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
