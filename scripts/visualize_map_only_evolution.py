"""No-safety prediction: the clean trajectory (x0) of EVERY reverse step.

Runs the sampler in ``map_only`` mode (no ellipses, no guided search, no ALM) and
draws, per scene,

    (a) all ``steps`` clean trajectories on one axis, coloured by the reverse step
        index, with the GT path underneath;
    (b) the snapshots that matter, labelled with their ``t``;
    (c) the final prediction next to the GT;

plus a contact sheet over the scenes.

Usage::

    python scripts/visualize_map_only_evolution.py --device cuda \
        --checkpoint outputs/checkpoints/ab_distill_safeu_e100/d2_ep080.pt \
        --indices 0,1,2,3,4,5 --steps 50
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import PROJECT_ROOT, base_parser, load  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gp_diffuser.evaluation.metrics import trajectory_metrics  # noqa: E402
from gp_diffuser.factory import build_pipeline  # noqa: E402
from gp_diffuser.training.real_dataset import RealTrajectoryDataset  # noqa: E402

SHOW_STEPS = 5


def _px(points, grid):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return (pts + 1.0) * 0.5 * grid - 0.5


def _base(ax, occ, grid, title=None, lim=None):
    ax.imshow(occ, cmap="Greys", origin="lower", vmin=0.0, vmax=1.0,
              interpolation="nearest")
    lo, hi = (lim if lim is not None else ((-0.5, -0.5), (grid - 0.5, grid - 0.5)))
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


def main() -> int:
    parser = base_parser("every clean trajectory of a map_only sample")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--indices", default=None)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--pick-seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", default="outputs/figures/map_only_evolution")
    opts = parser.parse_args()

    cfg = load(opts)
    pipe = build_pipeline(cfg, device=opts.device, checkpoint=opts.checkpoint)
    dataset = RealTrajectoryDataset(cfg, opts.split)
    grid = dataset.grid
    sampler = pipe["sampler"]
    outdir = os.path.join(PROJECT_ROOT, opts.outdir)
    os.makedirs(outdir, exist_ok=True)

    if opts.indices:
        indices = [int(v) for v in opts.indices.split(",") if v.strip()]
    else:
        rng = np.random.default_rng(opts.pick_seed)
        indices = sorted(rng.choice(len(dataset), size=min(opts.samples, len(dataset)),
                                    replace=False).tolist())
    items = [dataset[i] for i in indices]
    with torch.no_grad():
        out = sampler.sample(
            torch.stack([it["occupancy"] for it in items]),
            torch.stack([it["sdf"] for it in items]),
            torch.stack([it["start_goal"] for it in items]),
            heading=torch.stack([it["heading"] for it in items]),
            heading_valid=torch.stack([it["heading_valid"] for it in items]),
            steps=opts.steps, seed=opts.seed, safety_mode="map_only",
            record_curves=True)
    trace = out["steps_trace"]
    n_steps = len(trace)
    colours = plt.cm.viridis(np.linspace(0.05, 0.92, n_steps))
    picks = sorted({0, n_steps // 4, n_steps // 2, (3 * n_steps) // 4, n_steps - 1})
    picks = [min(p, n_steps - 1) for p in picks]

    records = []
    for k, index in enumerate(indices):
        item = items[k]
        occ = item["occupancy"][0].numpy()
        occ = occ[:, :, 0] if occ.ndim == 3 else occ
        gt = _px(item["path_gt_scene"].numpy(), grid)
        sg = _px(item["start_goal"].numpy(), grid)
        final = _px(out["trajectory_scene"][k].detach().cpu().numpy(), grid)
        metrics = trajectory_metrics(out["trajectory_m"][k].detach().cpu().numpy(), occ,
                                     item["sdf"][0].numpy(), dataset.res_m)
        curves = [_px(entry["x0_scene"][k], grid) for entry in trace]

        pts = np.concatenate([gt, final, sg] + curves, axis=0)
        lo = pts.min(axis=0) - 8.0
        hi = pts.max(axis=0) + 8.0
        lim = (lo, hi)
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6))
        ax = axes[0]
        _base(ax, occ, grid, "every clean trajectory (%d reverse steps, "
                             "colour = step)" % n_steps, lim)
        ax.plot(gt[:, 0], gt[:, 1], color="#1f77b4", lw=2.2, label="GT")
        for s in range(n_steps):
            ax.plot(curves[s][:, 0], curves[s][:, 1], color=colours[s], lw=0.9,
                    alpha=0.8)
        ax.legend(loc="lower left", fontsize=6)
        sm = plt.cm.ScalarMappable(cmap="viridis",
                                   norm=plt.Normalize(1, n_steps))
        cb = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label("reverse step", fontsize=7)
        cb.ax.tick_params(labelsize=6)

        ax = axes[1]
        _base(ax, occ, grid, "snapshots", lim)
        ax.plot(gt[:, 0], gt[:, 1], color="#1f77b4", lw=2.0, label="GT")
        for s in picks:
            ax.plot(curves[s][:, 0], curves[s][:, 1], color=colours[s], lw=1.8,
                    label="step %d (t=%d)" % (s + 1, trace[s]["t"]))
        ax.legend(loc="lower left", fontsize=6)

        ax = axes[2]
        _base(ax, occ, grid, "final x0 vs GT", lim)
        ax.plot(gt[:, 0], gt[:, 1], color="#1f77b4", lw=2.2, label="GT")
        ax.plot(final[:, 0], final[:, 1], color="#d62728", lw=2.4,
                label="final (step %d)" % n_steps)
        ax.scatter(sg[:, 0], sg[:, 1], s=70, c=["#2ca02c", "#d62728"],
                   edgecolors="k", zorder=6, label="start / goal")
        ax.set_title("final x0 vs GT — collision %.3f, len %.1f m"
                     % (float(metrics["collision_rate"]), float(metrics["length_m"])),
                     fontsize=8)
        ax.legend(loc="lower left", fontsize=6)

        fig.suptitle("no-safety (map_only) prediction — val sample %d, %s"
                     % (index, os.path.basename(opts.checkpoint or "default")),
                     fontsize=11)
        fig.tight_layout()
        path = os.path.join(outdir, "sample%02d_evolution.png" % index)
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)

        drift = [float(np.linalg.norm(curves[s] - curves[s - 1], axis=-1).mean())
                 for s in range(1, n_steps)]
        records.append({"index": index, "collision_rate": float(metrics["collision_rate"]),
                        "length_m": float(metrics["length_m"]),
                        "valid": bool(out["valid_mask"][k]),
                        "step_change_px": [round(v, 3) for v in drift],
                        "figure": os.path.relpath(path, PROJECT_ROOT)})
        print(json.dumps({kk: records[-1][kk] for kk in
                          ("index", "valid", "collision_rate", "length_m")}), flush=True)

    with open(os.path.join(outdir, "evolution.json"), "w", encoding="utf-8") as fh:
        json.dump({"checkpoint": opts.checkpoint, "steps": int(opts.steps),
                   "seed": int(opts.seed), "records": records}, fh, indent=2)
    print("figures:", outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
