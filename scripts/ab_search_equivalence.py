"""A/B: our vendored guidance-field search vs the packaged `field_guided_search`.

Both are supposed to be the same Astar-Diffusion algorithm.  This script runs them
on identical occupancy maps, identical guide curves and identical endpoints, then
reports path length, mutual deviation, expansion counts and collision rates.

    python scripts/ab_search_equivalence.py --checkpoint <ckpt> --samples 5
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import PROJECT_ROOT, base_parser, load  # noqa: E402

import numpy as np  # noqa: E402

from gp_diffuser.evaluation.metrics import trajectory_metrics  # noqa: E402
from gp_diffuser.factory import build_pipeline  # noqa: E402
from gp_diffuser.training.real_dataset import RealTrajectoryDataset  # noqa: E402

REF_SRC = os.path.join(PROJECT_ROOT, "references", "Astar-Diffusion",
                       "field-guided-search", "src")


def _length(p) -> float:
    p = np.asarray(p, dtype=np.float64).reshape(-1, 2)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=-1).sum())


def _max_gap(a, b) -> float:
    d = np.linalg.norm(np.asarray(a)[:, None, :] - np.asarray(b)[None, :, :], axis=-1)
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))


def main() -> int:
    parser = base_parser("A/B the two guidance-field search packagings")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--steps", type=int, default=50)
    opts = parser.parse_args()

    sys.path.insert(0, REF_SRC)
    from field_guided_search import GuidanceField  # noqa: E402  (reference package)

    cfg = load(opts)
    dataset = RealTrajectoryDataset(cfg, opts.split)
    pipe = build_pipeline(cfg, device=opts.device, checkpoint=opts.checkpoint)
    sampler, guidance, res_m = pipe["sampler"], pipe["guidance"], float(dataset.res_m)
    transform = pipe["transform"]
    print("ours   :", type(guidance).__module__ + "." + type(guidance).__name__)
    print("reference:", GuidanceField.__module__)
    print()

    rows = []
    for index in range(opts.samples):
        item = dataset[index]
        occ = item["occupancy"][None].clone()
        sdf = item["sdf"][None].clone()
        sg = item["start_goal"][None].clone()
        hd = item["heading"][None].clone() if item.get("heading") is not None else None
        hv = (item["heading_valid"][None].clone()
              if item.get("heading_valid") is not None else None)
        out = sampler.sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                             steps=opts.steps, safety_mode="map_only", seed=0)
        guide_m = out["trajectory_m"][0].numpy()
        occ_np = occ[0, 0].numpy()
        start_m = transform.to_meters(item["start_goal"].numpy()[0])
        goal_m = transform.to_meters(item["start_goal"].numpy()[1])

        ours = guidance.search(occ_np, guide_m, start_m, goal_m)

        gfield = GuidanceField((occ_np > 0).astype(np.int8), lam=guidance.lam,
                               weight=guidance.weight,
                               num_samples=guidance.num_sources)
        guide_rc = np.stack([guide_m[:, 1] / res_m, guide_m[:, 0] / res_m], axis=-1)
        gfield.build(guide_rc, num_samples=64)
        start_rc = (int(round(start_m[1] / res_m)), int(round(start_m[0] / res_m)))
        goal_rc = (int(round(goal_m[1] / res_m)), int(round(goal_m[0] / res_m)))
        ref = gfield.search(start_rc, goal_rc, alpha=0.25)
        ref_path = None
        if ref.get("path") is not None:
            prc = np.asarray(ref["path"], dtype=np.float64)
            ref_path = np.stack([prc[:, 1] * res_m, prc[:, 0] * res_m], axis=-1)

        stats = {
            "index": index,
            "ours_status": ours["status"],
            "ours_expanded": int(ours["expanded"]),
            "ours_valid": bool(ours["valid"]),
            "ours_len_m": _length(ours["path_xy"]) if ours["path_xy"] is not None else None,
            "ref_status": ref.get("status"),
            "ref_expanded": int(ref.get("expanded", 0)),
            "ref_len_m": None if ref_path is None else _length(ref_path),
            "max_gap_m": (None if ref_path is None or ours["path_xy"] is None
                          else _max_gap(ours["path_xy"], ref_path)),
        }
        for name, path in (("ours", ours["path_xy"]), ("ref", ref_path)):
            if path is not None:
                stats[name + "_collision"] = float(trajectory_metrics(
                    path, occ_np, item["sdf"][0].numpy(), res_m)["collision_rate"])
        rows.append(stats)
        print(stats, flush=True)

    keys = ("ours_len_m", "ref_len_m", "max_gap_m", "ours_expanded", "ref_expanded")
    print()
    for key in keys:
        vals = [r[key] for r in rows if r.get(key) is not None]
        if vals:
            print("%-14s mean %8.3f   min %8.3f   max %8.3f"
                  % (key, float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
