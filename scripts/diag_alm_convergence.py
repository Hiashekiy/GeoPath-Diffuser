"""Is the constraint set actually safe, and does ONE converged projection fix it?

The user's argument: if the convex regions are obstacle-free, projecting a trajectory
onto A x <= b cannot make it worse -- so either the region set is not equivalent to
"collision free along the whole curve", or the projection simply never converges.

This script answers both, for one scene, with numbers:

  1. ``pack`` coverage: which parameter intervals the pieces constrain, and whether
     they cover [0,1] at all.
  2. Projection convergence: same prediction, inner_steps = 6 / 12 / 25 / 50 / 100 /
     200 with max_curve_step_scene = 0.20 and 0.05 -> violation, correction length,
     collision rate.
  3. The "one shot is enough" test: take the no-ALM final trajectory, project it
     ONCE with a large budget, and report violation + collision before/after.
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

from gp_diffuser.diffusion.bspline_alm import bspline_alm_correct, constraint_state  # noqa: E402
from gp_diffuser.evaluation.metrics import trajectory_metrics  # noqa: E402
from gp_diffuser.factory import build_pipeline  # noqa: E402
from gp_diffuser.training.real_dataset import RealTrajectoryDataset  # noqa: E402


def _collision(curve_m, item, res_m) -> float:
    return float(trajectory_metrics(np.asarray(curve_m), item["occupancy"][0].numpy(),
                                    item["sdf"][0].numpy(), res_m)["collision_rate"])


def main() -> int:
    parser = base_parser("ALM convergence / constraint-set sanity")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--out", default="outputs/reports/alm_convergence.json")
    opts = parser.parse_args()

    cfg = load(opts)
    dataset = RealTrajectoryDataset(cfg, "val")
    pipe = build_pipeline(cfg, device=opts.device, checkpoint=opts.checkpoint)
    sampler, transform, codec = pipe["sampler"], pipe["transform"], pipe["sampler"].codec
    res_m = float(dataset.res_m)

    item = dataset[opts.index]
    occ = item["occupancy"][None].clone()
    sdf = item["sdf"][None].clone()
    sg = item["start_goal"][None].clone()
    hd = item["heading"][None].clone() if item.get("heading") is not None else None
    hv = (item["heading_valid"][None].clone()
          if item.get("heading_valid") is not None else None)

    out_map = sampler.sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                             steps=opts.steps, safety_mode="map_only", seed=0)
    raw, _ = sampler._generate_safety(
        occ.to(sampler.device), sdf.to(sampler.device),
        sampler.map_encoder.encode(occ.to(sampler.device), sdf.to(sampler.device)),
        out_map["trajectory_scene"][0:1], sg.to(sampler.device),
        dict(sampler.cfg.get("corridor") or {}))
    pack = raw["pack"]

    # ---- 1. what the pieces constrain -------------------------------------
    iv = pack.intervals[0, :int(pack.num_pieces[0])].detach().cpu().numpy()
    faces = pack.face_mask[0, :int(pack.num_pieces[0])].sum(dim=-1).detach().cpu().numpy()
    print("pieces %d | faces/piece %d..%d | intervals cover [%.3f, %.3f]"
          % (len(iv), int(faces.min()), int(faces.max()), iv[:, 0].min(), iv[:, 1].max()))
    gaps = []
    order = np.argsort(iv[:, 0])
    for a, b in zip(order, order[1:]):
        if iv[b, 0] > iv[a, 1] + 1e-6:
            gaps.append((float(iv[a, 1]), float(iv[b, 0])))
    print("uncovered parameter gaps:", gaps if gaps else "none")

    # ---- 2. convergence of the projection --------------------------------
    alm_cfg = dict(sampler._alm_cfg())
    out_noalm = sampler.sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                               steps=opts.steps, safety_mode="no_alm", seed=0)
    q_ref = out_noalm["controls"][0:1].detach()
    curve_ref = out_noalm["trajectory_m"][0].numpy()
    print("\nno-ALM final: collision %.4f  length %.2f m"
          % (_collision(curve_ref, item, res_m),
             float(np.linalg.norm(np.diff(curve_ref, axis=0), axis=-1).sum())))

    rows = []
    for cap in (0.20, 0.05):
        for inner in (6, 12, 25, 50, 100, 200):
            cfg_i = dict(alm_cfg)
            cfg_i["max_curve_step_scene"] = cap
            q_safe, _lam, st = bspline_alm_correct(
                q_ref, pack, codec, lam=None, config=cfg_i, inner_steps=inner)
            curve = sampler._decode(q_safe)[0].detach().cpu().numpy()
            curve_m = transform.to_meters(curve)
            _, g, mask = constraint_state(q_safe, pack)
            viol = float(torch.relu(g).masked_fill(~mask.expand_as(g), 0).max())
            rows.append({
                "cap_scene": cap, "cap_m": cap * transform.scale, "inner_steps": inner,
                "violation_scene": viol, "violation_m": viol * transform.scale,
                "violation_trajsafe_stats": float(st["max_violation_after"].max()),
                "correction_m": float(st["mean_curve_correction_m"].max()) * transform.scale,
                "collision": _collision(curve_m, item, res_m),
                "length_m": float(np.linalg.norm(np.diff(curve_m, axis=0), axis=-1).sum()),
            })
            print("cap %.2f scene (%.1f m) inner %3d | violation %.4f scene (%.2f m) "
                  "| correction %.2f m | collision %.4f | length %.1f m"
                  % (cap, cap * transform.scale, inner, viol, viol * transform.scale,
                     rows[-1]["correction_m"], rows[-1]["collision"],
                     rows[-1]["length_m"]))

    out_path = os.path.join(PROJECT_ROOT, opts.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"index": opts.index, "pack_pieces": int(len(iv)),
                   "intervals": iv.tolist(), "gaps": gaps,
                   "no_alm_collision": _collision(curve_ref, item, res_m),
                   "rows": rows}, fh, indent=2)
    print("written:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
