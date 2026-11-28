"""Gallery: trajectories generated on an external low-density map, all on one plot."""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import PROJECT_ROOT, base_parser, load  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import ndimage  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

from gp_diffuser.evaluation.metrics import trajectory_metrics  # noqa: E402
from gp_diffuser.factory import build_pipeline  # noqa: E402
from gp_diffuser.frames import SceneTransform  # noqa: E402
from test_octile_map import load_octile, sdf_from_occupancy  # noqa: E402


def main() -> int:
    parser = base_parser("trajectory gallery on an external octile map")
    parser.add_argument("--map", required=True)
    parser.add_argument("--window", default="auto")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--length-m", default="18,25,32")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--checkpoint", default="outputs/checkpoints/last.pt")
    parser.add_argument("--outdir", default="outputs/figures/octile_gallery")
    opts = parser.parse_args()

    cfg = load(opts)
    grid, head = load_octile(opts.map)
    res = float(head.get("resolution", 0.25))
    if opts.window == "auto":
        best = None
        for y in range(0, grid.shape[0] - opts.size + 1, 8):
            for x in range(0, grid.shape[1] - opts.size + 1, 8):
                f = float(grid[y:y + opts.size, x:x + opts.size].mean())
                if best is None or f < best[0]:
                    best = (f, x, y)
        frac, x0, y0 = best
    else:
        x0, y0 = [int(v) for v in opts.window.split(",")]
        frac = float(grid[y0:y0 + opts.size, x0:x0 + opts.size].mean())
    occ = grid[y0:y0 + opts.size, x0:x0 + opts.size]
    sdf = sdf_from_occupancy(occ, res)
    clear = np.where(occ == 0, ndimage.distance_transform_edt(occ == 0) * res, 0.0)

    pipe = build_pipeline(cfg, device=opts.device, checkpoint=opts.checkpoint)
    sampler = pipe["sampler"]
    transform = SceneTransform(res * opts.size, opts.size)
    occ_t = torch.as_tensor(occ, dtype=torch.float32, device=opts.device)[None, None]
    sdf_t = torch.as_tensor(sdf, dtype=torch.float32, device=opts.device)[None, None]

    free = (occ == 0) & (clear >= 0.8)
    ys, xs = np.nonzero(free)
    pts = np.stack([xs, ys], 1).astype(np.float64)
    rng = np.random.default_rng(3)
    targets = [float(v) for v in opts.length_m.split(",")]
    results, used = [], []
    tries = 0
    while len(results) < opts.pairs and tries < 6000:
        tries += 1
        a, b = pts[rng.integers(0, len(pts))], pts[rng.integers(0, len(pts))]
        d = float(np.linalg.norm(b - a)) * res
        target = targets[len(results) % len(targets)]
        if abs(d - target) > 2.0:
            continue
        t = np.linspace(0, 1, 40)[:, None]
        line = a + t * (b - a)
        blocked = occ[np.clip(line[:, 1].astype(int), 0, opts.size - 1),
                      np.clip(line[:, 0].astype(int), 0, opts.size - 1)].mean()
        if blocked > 0.05:
            continue
        if any(np.linalg.norm(a - u) < 12 or np.linalg.norm(b - u) < 12 for u in used):
            continue
        used.extend([a, b])
        start_m, goal_m = a * res, b * res
        sg = torch.as_tensor(transform.to_scene(np.stack([start_m, goal_m])),
                             dtype=torch.float32, device=opts.device)[None]
        chord = goal_m - start_m
        hd = torch.as_tensor(np.stack([chord, chord]) / np.linalg.norm(chord),
                             dtype=torch.float32, device=opts.device)[None]
        out = sampler.sample(occ_t, sdf_t, sg, heading=hd,
                             heading_valid=torch.ones(1, 2, dtype=torch.bool,
                                                      device=opts.device),
                             steps=opts.steps, seed=0)
        traj = out["trajectory_m"][0].detach().cpu().numpy()
        m = trajectory_metrics(traj, occ.astype(np.float32), sdf.astype(np.float32), res)
        ellipses = (None if out.get("ellipses") is None
                    else out["ellipses"][0].detach().cpu().numpy())
        results.append({"start": start_m.tolist(), "goal": goal_m.tolist(),
                        "traj": traj, "ellipses": ellipses, "chord": d,
                        "collision": float(m["collision_rate"]),
                        "clearance": float(m["min_clearance_m"]),
                        "length": float(m["length_m"]),
                        "valid": bool(out["valid"]),
                        "guidance_steps": int(out.get("guidance_steps", 0)),
                        "qp_status": (out["final_qp_status"][0]
                                      if out.get("final_qp_status") else "not_run"),
                        "projection_correction_m": float(
                            out.get("final_projection_correction_m", 0.0)),
                        "cells": int((out["corridor_info"] or {}).get("local_cells", 0))})

    outdir = os.path.join(PROJECT_ROOT, opts.outdir)
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.4, 8.4))
    ax.imshow(occ, cmap="Greys", origin="lower", vmin=0, vmax=1, interpolation="nearest")
    colours = plt.cm.tab10(np.linspace(0, 0.9, max(len(results), 1)))
    scale = opts.size / 2.0
    for k, r in enumerate(results):
        p = r["traj"] / res
        ell = r.get("ellipses")
        if ell is not None:
            for cx, cy, a, b, c, s in ell:
                if not np.isfinite([cx, cy, a, b]).all() or a <= 0 or b <= 0:
                    continue
                ax.add_patch(Ellipse(((cx + 1.0) * scale - 0.5, (cy + 1.0) * scale - 0.5),
                                     width=2 * a * scale, height=2 * b * scale,
                                     angle=np.degrees(np.arctan2(s, c)), fill=False,
                                     edgecolor=colours[k], alpha=0.55, lw=0.9))
            ax.scatter((ell[:, 0] + 1.0) * scale - 0.5, (ell[:, 1] + 1.0) * scale - 0.5,
                       s=7, marker="x", color=colours[k], alpha=0.8, lw=0.7)
        ax.plot(p[:, 0], p[:, 1], color=colours[k], lw=2.2,
                label="#%d chord %.0fm len %.0fm coll %.2f" % (k, r["chord"], r["length"],
                                                              r["collision"]))
        s = np.array(r["start"]) / res
        g = np.array(r["goal"]) / res
        ax.plot([s[0], g[0]], [s[1], g[1]], color=colours[k], lw=0.8, ls=":", alpha=0.7)
        ax.scatter([s[0], g[0]], [s[1], g[1]], s=80, c=[colours[k]], edgecolors="k", zorder=6)
    ax.set_title("%s — window (%d,%d), blocked %.1f%% — %d trajectories + 64 ellipses each"
                 % (os.path.basename(opts.map), x0, y0, 100 * frac, len(results)), fontsize=10)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    png = os.path.join(outdir, "gallery.png")
    fig.savefig(png, dpi=110)
    plt.close(fig)
    with open(os.path.join(outdir, "gallery.json"), "w", encoding="utf-8") as fh:
        json.dump([{k: v for k, v in r.items() if k not in ("traj", "ellipses")}
                   for r in results],
                  fh, indent=2, ensure_ascii=False)
    for k, r in enumerate(results):
        print("#%d chord %.1f m  len %.1f m  coll %.3f  clear %+.2fm  valid %s  "
              "cells %d  cpcg %d steps  qp %s"
              % (k, r["chord"], r["length"], r["collision"], r["clearance"], r["valid"],
                 r["cells"], r["guidance_steps"], r["qp_status"]))
    print("figure:", os.path.relpath(png, PROJECT_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
