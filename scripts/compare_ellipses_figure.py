"""Draw OUR ellipses and PathSafe's OWN ellipses on the same axes.

The numbers already agree (centre gap 0.22 m mean, axes ratio 0.98/0.96), so this
figure exists to show whether the *drawing* agrees too:

  panel 1: our pipeline's ellipses, drawn with matplotlib's Ellipse patch
  panel 2: PathSafe's ellipses for the same path, drawn with THEIR ellipse_xy()
  panel 3: both overlaid (ours dashed red, theirs solid blue) + our convex regions
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


def _poly_scene_to_m(poly, transform):
    return transform.to_meters(np.asarray(poly, dtype=np.float64))


def main() -> int:
    parser = base_parser("draw our vs PathSafe ellipses side by side")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--ref-json", default=os.path.join(
        "references", "PathSafe-Ellipse", "results", "geopath_demo",
        "pathsafe_ellipses.json"))
    parser.add_argument("--out", default="outputs/figures/pipeline_diag/ellipse_vs_pathsafe.png")
    opts = parser.parse_args()

    sys.path.insert(0, os.path.join(PROJECT_ROOT, "references", "PathSafe-Ellipse"))
    from ellipsenet.visualize import ellipse_xy  # their drawing primitive

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    cfg = load(opts)
    dataset = RealTrajectoryDataset(cfg, "val")
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
    out_map = sampler.sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                             steps=opts.steps, safety_mode="map_only", seed=0)
    features = sampler.map_encoder.encode(occ.to(sampler.device), sdf.to(sampler.device))
    raw, _ = sampler._generate_safety(
        occ.to(sampler.device), sdf.to(sampler.device), features,
        out_map["trajectory_scene"][0:1], sg.to(sampler.device),
        dict(sampler.cfg.get("corridor") or {}))

    path_m = raw["search_path_m"][0].detach().cpu().numpy()
    centres = transform.to_meters(raw["ellipse_centers_scene"][0].detach().cpu().numpy())
    a = transform.lengths_to_meters(raw["ellipse"][0][:, 2].detach().cpu().numpy())
    b = transform.lengths_to_meters(raw["ellipse"][0][:, 3].detach().cpu().numpy())
    c2 = raw["ellipse"][0][:, 4].detach().cpu().numpy()
    s2 = raw["ellipse"][0][:, 5].detach().cpu().numpy()
    theta = 0.5 * np.arctan2(s2, c2)

    ref = json.load(open(os.path.join(PROJECT_ROOT, opts.ref_json), encoding="utf-8"))
    rc = np.asarray(ref["centres_m"])
    ra = np.asarray(ref["a_m"])
    rb = np.asarray(ref["b_m"])
    rt = np.asarray(ref["theta_rad"])

    occ_np = occ[0, 0].numpy()
    extent = [0, occ_np.shape[1] * res_m, 0, occ_np.shape[0] * res_m]
    fig, axes = plt.subplots(1, 3, figsize=(24, 8.4))
    for ax in axes:
        ax.imshow(occ_np, origin="lower", cmap="Greys", extent=extent, alpha=0.92)
        ax.plot(path_m[:, 0], path_m[:, 1], "-", color="#ff2d55", lw=2.0,
                label="searched path (ours)")
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
        ax.tick_params(labelsize=8)

    for c, aa, bb, th in zip(centres, a, b, theta):
        axes[0].add_patch(Ellipse((c[0], c[1]), 2 * aa, 2 * bb,
                                  angle=np.degrees(th), fill=False, ec="#1f77b4",
                                  lw=0.8, alpha=0.7))
    axes[0].set_title("1) OUR ellipses (matplotlib patch, theta=0.5*atan2(sin2t,cos2t))",
                      fontsize=11)

    for c, aa, bb, th in zip(rc, ra, rb, rt):
        xy = ellipse_xy(c, aa, bb, th)
        if isinstance(xy, tuple):
            axes[1].plot(xy[0], xy[1], "-", color="#2ca02c", lw=0.8, alpha=0.7)
        else:
            xy = np.asarray(xy)
            axes[1].plot(xy[:, 0], xy[:, 1], "-", color="#2ca02c", lw=0.8, alpha=0.7)
    axes[1].set_title("2) PathSafe's ellipses, drawn with THEIR ellipse_xy()",
                      fontsize=11)

    for c, aa, bb, th in zip(centres, a, b, theta):
        axes[2].add_patch(Ellipse((c[0], c[1]), 2 * aa, 2 * bb,
                                  angle=np.degrees(th), fill=False, ec="#1f77b4",
                                  lw=0.8, alpha=0.55, ls="--"))
    for c, aa, bb, th in zip(rc, ra, rb, rt):
        xy = ellipse_xy(c, aa, bb, th)
        if isinstance(xy, tuple):
            axes[2].plot(xy[0], xy[1], "-", color="#2ca02c", lw=0.8, alpha=0.55)
        else:
            xy = np.asarray(xy)
            axes[2].plot(xy[:, 0], xy[:, 1], "-", color="#2ca02c", lw=0.8, alpha=0.55)
    axes[2].set_title("3) overlaid: ours (blue dashed) vs PathSafe (green solid)",
                      fontsize=11)

    fig.suptitle("val %d | our safety feature vs PathSafe's own output on the same path"
                 % opts.index, fontsize=13)
    fig.tight_layout()
    out = os.path.join(PROJECT_ROOT, opts.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=115, bbox_inches="tight")
    plt.close(fig)
    print("written:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
