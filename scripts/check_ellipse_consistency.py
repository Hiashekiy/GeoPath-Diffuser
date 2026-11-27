"""Are our ellipse centres / regions on the searched path, or systematically displaced?

Compares three things for one scene, all in metres:
  * the searched path our pipeline feeds to the safety generator,
  * the ellipse centres our pipeline predicts (after scene -> metre conversion),
  * the ellipse centres PathSafe-Ellipse predicts on the Astar-Diffusion path for
    the same map (references/PathSafe-Ellipse/results/geopath_demo/pathsafe_ellipses.json).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import PROJECT_ROOT, base_parser, load  # noqa: E402

import numpy as np  # noqa: E402

from gp_diffuser.factory import build_pipeline  # noqa: E402
from gp_diffuser.training.real_dataset import RealTrajectoryDataset  # noqa: E402


def dist_to_polyline(points, poly) -> np.ndarray:
    d = np.linalg.norm(np.asarray(points)[:, None, :] - np.asarray(poly)[None, :, :], axis=-1)
    return d.min(axis=1)


def main() -> int:
    parser = base_parser("ellipse centre consistency check")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--ref-json", default=os.path.join(
        "references", "PathSafe-Ellipse", "results", "geopath_demo",
        "pathsafe_ellipses.json"))
    opts = parser.parse_args()

    cfg = load(opts)
    dataset = RealTrajectoryDataset(cfg, "val")
    pipe = build_pipeline(cfg, device=opts.device, checkpoint=opts.checkpoint)
    sampler, transform = pipe["sampler"], pipe["transform"]

    item = dataset[opts.index]
    occ = item["occupancy"][None].clone()
    sdf = item["sdf"][None].clone()
    sg = item["start_goal"][None].clone()
    hd = item["heading"][None].clone() if item.get("heading") is not None else None
    hv = (item["heading_valid"][None].clone()
          if item.get("heading_valid") is not None else None)
    out_map = sampler.sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                             steps=opts.steps, safety_mode="map_only", seed=0)
    features = sampler.map_encoder.encode(occ.to(sampler.device), sdf.to(sampler.device))
    raw, stats = sampler._generate_safety(
        occ.to(sampler.device), sdf.to(sampler.device), features,
        out_map["trajectory_scene"][0:1], sg.to(sampler.device),
        dict(sampler.cfg.get("corridor") or {}))

    path_m = raw["search_path_m"][0].detach().cpu().numpy()
    centres_m = transform.to_meters(raw["ellipse_centers_scene"][0].detach().cpu().numpy())
    axes_m = transform.lengths_to_meters(raw["ellipse"][0][:, 2:4].detach().cpu().numpy())
    print("searched path: %d pts, from %s to %s"
          % (len(path_m), np.round(path_m[0], 2).tolist(), np.round(path_m[-1], 2).tolist()))
    print("our ellipse centres: first %s  mid %s  last %s"
          % (np.round(centres_m[0], 2).tolist(),
             np.round(centres_m[len(centres_m) // 2], 2).tolist(),
             np.round(centres_m[-1], 2).tolist()))
    d = dist_to_polyline(centres_m, path_m)
    print("our centre -> searched path distance (m): mean %.2f  p90 %.2f  max %.2f"
          % (d.mean(), np.percentile(d, 90), d.max()))
    print("our axes (m): a %.2f..%.2f  b %.2f..%.2f" %
          (axes_m[:, 0].min(), axes_m[:, 0].max(), axes_m[:, 1].min(), axes_m[:, 1].max()))

    ref_path = os.path.join(PROJECT_ROOT, opts.ref_json)
    if os.path.isfile(ref_path):
        ref = json.load(open(ref_path, encoding="utf-8"))
        ref_centres = np.asarray(ref["centres_m"])
        ref_a = np.asarray(ref["a_m"])
        ref_b = np.asarray(ref["b_m"])
        dr = dist_to_polyline(ref_centres, path_m)
        print("REFERENCE (PathSafe on Astar path) centres -> our path: mean %.2f max %.2f"
              % (dr.mean(), dr.max()))
        print("REFERENCE axes (m): a %.2f..%.2f  b %.2f..%.2f"
              % (ref_a.min(), ref_a.max(), ref_b.min(), ref_b.max()))
        n = min(len(ref_centres), len(centres_m))
        gap = np.linalg.norm(ref_centres[:n] - centres_m[:n], axis=-1)
        print("anchor-wise centre gap ours vs reference: mean %.2f m  median %.2f  max %.2f"
              % (gap.mean(), np.median(gap), gap.max()))
        print("a ratio (ours/ref): mean %.2f   b ratio: mean %.2f"
              % ((axes_m[:n, 0] / np.maximum(ref_a[:n], 1e-6)).mean(),
                 (axes_m[:n, 1] / np.maximum(ref_b[:n], 1e-6)).mean()))
    else:
        print("reference json not found:", ref_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
