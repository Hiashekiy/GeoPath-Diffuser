"""Export one GeoPath case (map + diffusion guidance) for the reference libraries.

Writes an npz next to the reference repos so that THEIR OWN code can be run on our
data without any of our adapters:

    python scripts/export_case_for_refs.py --checkpoint <ckpt> --index 0

Output (default ``references/Astar-Diffusion/field-guided-search/data/``):
    occupancy  int8 [H,W]   1 = blocked (the upstream convention: 0 = free)
    guide_cells float [N,2] [row, col] of the map-only diffusion prediction
    start_cell int [2], goal_cell int [2]   same [row, col] convention
    res_m, extent_m
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import PROJECT_ROOT, base_parser, load  # noqa: E402

import numpy as np  # noqa: E402

from gp_diffuser.factory import build_pipeline  # noqa: E402
from gp_diffuser.training.real_dataset import RealTrajectoryDataset  # noqa: E402


def main() -> int:
    parser = base_parser("export a case for the reference libraries")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=os.path.join(
        "references", "Astar-Diffusion", "field-guided-search", "data"))
    opts = parser.parse_args()

    cfg = load(opts)
    dataset = RealTrajectoryDataset(cfg, opts.split)
    pipe = build_pipeline(cfg, device=opts.device, checkpoint=opts.checkpoint)
    sampler, transform = pipe["sampler"], pipe["transform"]
    res_m = float(dataset.res_m)

    item = dataset[opts.index]
    occ = item["occupancy"][None].clone()
    sdf = item["sdf"][None].clone()
    sg = item["start_goal"][None].clone()
    hd = item["heading"][None].clone() if item.get("heading") is not None else None
    hv = (item["heading_valid"][None].clone()
          if item.get("heading_valid") is not None else None)
    out = sampler.sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                         steps=opts.steps, safety_mode="map_only", seed=opts.seed)
    guide_m = out["trajectory_m"][0].numpy()
    start_m = transform.to_meters(sg[0, 0].numpy())
    goal_m = transform.to_meters(sg[0, 1].numpy())
    print("endpoint error: start %.4f m  goal %.4f m"
          % (np.linalg.norm(guide_m[0] - start_m),
             np.linalg.norm(guide_m[-1] - goal_m)), flush=True)

    occupancy = (occ[0, 0].numpy() > 0).astype(np.int8)
    guide_cells = np.stack([guide_m[:, 1] / res_m, guide_m[:, 0] / res_m], axis=-1)
    start_cell = np.rint([start_m[1] / res_m, start_m[0] / res_m]).astype(int)
    goal_cell = np.rint([goal_m[1] / res_m, goal_m[0] / res_m]).astype(int)
    gt_cells = np.stack([item["path_gt_m"].numpy()[:, 1] / res_m,
                         item["path_gt_m"].numpy()[:, 0] / res_m], axis=-1)

    out_dir = os.path.join(PROJECT_ROOT, opts.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "geopath_case%03d.npz" % opts.index)
    np.savez(path, occupancy=occupancy, guide_cells=guide_cells,
             guide_m=guide_m, start_cell=start_cell, goal_cell=goal_cell,
             start_m=start_m, goal_m=goal_m, gt_cells=gt_cells, res_m=res_m,
             extent_m=np.array(occupancy.shape[0] * res_m))
    print("occupancy", occupancy.shape, "blocked:", int(occupancy.sum()),
          "| guide points", len(guide_cells), "| start", start_cell, "goal", goal_cell)
    print("written:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
