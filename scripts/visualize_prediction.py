"""Full visualisation: safety geometry (corridor + ellipses + searched path) plus the
denoising process step by step.

Per sampled scene it writes

* ``sample<k>_safety.png``  1x3 : corridor + ellipses + searched path + GT ->
                                    final prediction -> the x0 of every step;
* ``sample<k>_steps.png``   one panel per reverse step, each with the corridor,
                                    ellipses, searched path, GT and that step's x0;
* ``safety_grid.png``       contact sheet over all scenes;
* ``summary.json``          per-scene metrics.

The corridor is rasterised exactly from the half-space cells the model sees
(``pack_local`` if present, else ``pack``), so what is drawn is the real feasible
set rather than an approximation.

Usage::

    python scripts/visualize_prediction.py --checkpoint <ckpt> --random 20 --steps 20
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
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

from gp_diffuser.evaluation.metrics import trajectory_metrics  # noqa: E402
from gp_diffuser.factory import build_pipeline  # noqa: E402
from gp_diffuser.training.real_dataset import RealTrajectoryDataset  # noqa: E402

CORRIDOR_CMAP = ListedColormap([[0, 0, 0, 0], [0.10, 0.60, 0.30, 0.22]])


def _np(t):
    if torch.is_tensor(t):
        t = t.detach().cpu().numpy()
    return np.asarray(t)


def _px(pts, grid):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    return (pts + 1.0) * 0.5 * grid - 0.5


def _corridor_mask(pack, grid):
    """Rasterise the feasible set of every masked half-space cell (scene coords).

    Works with the inference sample pack (one cell per uniform-arc-length sample
    point) and with the historical piece pack.
    """
    idx = np.arange(grid)
    scene = (idx + 0.5) / (grid / 2.0) - 1.0
    gx, gy = np.meshgrid(scene, scene)          # rows = y, cols = x
    mask = np.zeros((grid, grid), dtype=bool)
    if pack is None:
        return mask
    if hasattr(pack, "sample_A"):
        A = _np(pack.sample_A)[0]               # [M,F,2]
        b = _np(pack.sample_b)[0]               # [M,F]
        keep = _np(pack.sample_mask)[0] > 0
        faces = _np(pack.face_mask)[0] > 0      # [M,F] active half-spaces
    else:
        A = _np(pack.piece_A)[0]                # [P,F,2]
        b = _np(pack.piece_b)[0]                # [P,F]
        keep = _np(pack.piece_mask)[0] > 0
        faces = _np(pack.face_mask)[0] > 0      # [P,F] active half-spaces
    for p in np.nonzero(keep)[0]:
        act = faces[p]
        if not act.any():
            continue
        aa, bb = A[p][act], b[p][act]           # g = A x - b <= 0
        inside = (aa[:, 0][None, None, :] * gx[..., None]
                  + aa[:, 1][None, None, :] * gy[..., None]
                  - bb[None, None, :]) <= 1e-9
        mask |= inside.all(axis=-1)
    return mask


def _draw_base(ax, occ, mask, grid):
    ax.imshow(occ, cmap="Greys", origin="lower", vmin=0.0, vmax=1.0,
              interpolation="nearest")
    if mask is not None and mask.any():
        ax.imshow(mask.astype(np.float32), cmap=CORRIDOR_CMAP, origin="lower",
                  vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_xlim(-0.5, grid - 0.5)
    ax.set_ylim(-0.5, grid - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_ellipses(ax, ellipses, grid, colour="#7b2fbe", alpha=0.85, lw=1.0, centres=True):
    if ellipses is None:
        return
    e = _np(ellipses)[0]                        # [64,6] scene: cx,cy,a,b,cos,sin
    scale = grid / 2.0
    for cx, cy, a, b, c, s in e:
        if not np.isfinite([cx, cy, a, b]).all() or a <= 0 or b <= 0:
            continue
        px, py = _px(np.array([[cx, cy]]), grid)[0]
        ax.add_patch(Ellipse((px, py), width=2 * a * scale, height=2 * b * scale,
                             angle=np.degrees(np.arctan2(s, c)), fill=False,
                             edgecolor=colour, alpha=alpha, lw=lw))
    if centres:
        pxy = _px(e[:, :2], grid)
        ax.scatter(pxy[:, 0], pxy[:, 1], s=6, marker="x", c=colour, alpha=0.8, lw=0.6)


def main() -> int:
    parser = base_parser("visualise safety geometry and every DDIM step")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--indices", default=None)
    parser.add_argument("--random", type=int, default=20)
    parser.add_argument("--pick-seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--ncols", type=int, default=5)
    parser.add_argument("--safety-mode", default="cpcg",
                        choices=["map_only", "safety_only", "cpcg",
                                 "full", "no_alm"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", default="outputs/figures/predictions")
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
        indices = sorted(rng.choice(len(dataset), size=min(opts.random, len(dataset)),
                                    replace=False).tolist())

    panels, summary = [], []
    for index in indices:
        item = dataset[index]
        out = sampler.sample(item["occupancy"][None], item["sdf"][None],
                             item["start_goal"][None], heading=item["heading"][None],
                             heading_valid=item["heading_valid"][None],
                             steps=opts.steps, seed=opts.seed, record_curves=True,
                             safety_mode=opts.safety_mode)
        occ = _np(item["occupancy"][0])
        metrics = trajectory_metrics(_np(out["trajectory_m"][0]), occ, _np(item["sdf"][0]),
                                     dataset.res_m)
        # ONLY the local pack is enforced at sampling time; falling back to the
        # full pack would paint almost the whole map and hide the real geometry
        pack = out.get("pack_local")
        mask = _corridor_mask(pack, grid)
        gt_px = _px(_np(item["path_gt_scene"]), grid)
        pred_px = _px(_np(out["trajectory_scene"][0]), grid)
        search_px = (_px(_np(dataset.transform.to_scene(_np(out["search_path_m"][0]))), grid)
                     if out.get("search_path_m") is not None else None)
        sg_px = _px(_np(item["start_goal"]), grid)
        trace = out["steps_trace"]
        cols = plt.cm.viridis(np.linspace(0.05, 0.9, len(trace)))
        gt_len = float(torch.linalg.norm(torch.diff(item["path_gt_m"], dim=0), dim=-1).sum())
        info = out["corridor_info"] or {}

        # ---- safety geometry | final prediction | all x0 ---------------------
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4))
        ax = axes[0]
        _draw_base(ax, occ, mask, grid)
        _draw_ellipses(ax, out.get("ellipses"), grid)
        if search_px is not None:
            ax.plot(search_px[:, 0], search_px[:, 1], color="#ff7f0e", lw=2.0, ls="--",
                    label="searched path")
        ax.plot(gt_px[:, 0], gt_px[:, 1], color="#1f77b4", lw=2.2, label="GT")
        ax.scatter(sg_px[:, 0], sg_px[:, 1], s=80, c=["#2ca02c", "#d62728"],
                   edgecolors="k", zorder=6, label="start / goal")
        ax.set_title("safety geometry: corridor(green) + 64 ellipses + searched path\n"
                     "cells %d  spans %s  scope %s%s"
                     % (int(info.get("local_cells", 0)), info.get("spans"),
                        info.get("scope"),
                        "  [no local corridor -> ALM inactive]"
                        if int(info.get("local_cells", 0)) == 0 else ""), fontsize=8)
        ax.legend(loc="lower left", fontsize=6)

        ax = axes[1]
        _draw_base(ax, occ, mask, grid)
        _draw_ellipses(ax, out.get("ellipses"), grid, alpha=0.35, lw=0.6, centres=False)
        if search_px is not None:
            ax.plot(search_px[:, 0], search_px[:, 1], color="#ff7f0e", lw=1.4, ls="--")
        ax.plot(gt_px[:, 0], gt_px[:, 1], color="#1f77b4", lw=2.0, label="GT")
        ax.plot(pred_px[:, 0], pred_px[:, 1], color="#d62728", lw=2.4, label="prediction")
        ax.set_title("final prediction: valid=%s coll=%.3f clear=%.2fm\nlen %.1f / GT %.1f m"
                     % (bool(out["valid"]), float(metrics["collision_rate"]),
                        float(metrics["min_clearance_m"]), float(metrics["length_m"]), gt_len),
                     fontsize=8)
        ax.legend(loc="lower left", fontsize=6)

        ax = axes[2]
        _draw_base(ax, occ, mask, grid)
        _draw_ellipses(ax, out.get("ellipses"), grid, alpha=0.30, lw=0.5, centres=False)
        if search_px is not None:
            ax.plot(search_px[:, 0], search_px[:, 1], color="#ff7f0e", lw=1.2, ls="--")
        ax.plot(gt_px[:, 0], gt_px[:, 1], color="#1f77b4", lw=2.0, label="GT")
        for k, entry in enumerate(trace):
            cur = _px(_np(entry["x0_scene"][0]), grid)
            ax.plot(cur[:, 0], cur[:, 1], color=cols[k], lw=1.0, alpha=0.75)
        ax.plot([], [], color=cols[0], label="x0 step 1")
        ax.plot([], [], color=cols[-1], label="x0 step %d" % len(trace))
        ax.set_title("x0 of all %d reverse steps (viridis: early -> late)" % len(trace),
                     fontsize=8)
        ax.legend(loc="lower left", fontsize=6)
        fig.tight_layout()
        safety_png = os.path.join(outdir, "sample%02d_safety.png" % index)
        fig.savefig(safety_png, dpi=110)
        plt.close(fig)

        # ---- per-step panels with the full safety geometry ------------------
        ncols = max(1, int(opts.ncols))
        nrows = int(np.ceil(len(trace) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.9 * ncols, 2.9 * nrows))
        axes = np.atleast_2d(axes)
        for k, entry in enumerate(trace):
            ax = axes[k // ncols, k % ncols]
            _draw_base(ax, occ, mask, grid)
            _draw_ellipses(ax, out.get("ellipses"), grid, alpha=0.25, lw=0.5, centres=False)
            if search_px is not None:
                ax.plot(search_px[:, 0], search_px[:, 1], color="#ff7f0e", lw=1.0, ls="--")
            ax.plot(gt_px[:, 0], gt_px[:, 1], color="#1f77b4", lw=1.2)
            for j in range(k):
                old = _px(_np(trace[j]["x0_scene"][0]), grid)
                ax.plot(old[:, 0], old[:, 1], color=cols[j], lw=0.6, alpha=0.30)
            cur = _px(_np(entry["x0_scene"][0]), grid)
            ax.plot(cur[:, 0], cur[:, 1], color="#d62728", lw=1.6)
            if entry.get("x0_guided_scene") is not None:
                guided = _px(_np(entry["x0_guided_scene"][0]), grid)
                ax.plot(guided[:, 0], guided[:, 1], color="magenta", lw=1.0, ls=":")
            ax.scatter(sg_px[:, 0], sg_px[:, 1], s=14, c=["#2ca02c", "#d62728"],
                       zorder=5, edgecolors="k", linewidths=0.3)
            ax.set_title("%d/%d t=%d %s" % (k + 1, len(trace), entry["t"],
                                            "CPCG" if entry.get("constraint_guidance_applied")
                                            else entry["branch"][:3]), fontsize=6)
        for k in range(len(trace), nrows * ncols):
            axes[k // ncols, k % ncols].axis("off")
        fig.suptitle("sample %d — corridor(green) ellipses(purple) search(orange) GT(blue) "
                     "x0(red) after-CPCG(magenta)" % index, fontsize=10)
        fig.tight_layout()
        steps_png = os.path.join(outdir, "sample%02d_steps.png" % index)
        fig.savefig(steps_png, dpi=100)
        plt.close(fig)

        panels.append({"index": index, "occ": occ, "mask": mask, "gt": gt_px,
                       "search": search_px, "pred": pred_px,
                       "ellipses": out.get("ellipses"),
                       "collision": float(metrics["collision_rate"]),
                       "length": float(metrics["length_m"]), "gt_length": gt_len,
                       "valid": bool(out["valid"]),
                       "cells": int(info.get("local_cells", 0))})
        summary.append({"index": index, "valid": bool(out["valid"]),
                        "collision_rate": float(metrics["collision_rate"]),
                        "min_clearance_m": float(metrics["min_clearance_m"]),
                        "length_m": float(metrics["length_m"]), "gt_length_m": gt_len,
                        "corridor_cells": int(info.get("local_cells", 0)),
                        "corridor_scope": info.get("scope"),
                        "guidance_steps": int(out.get("guidance_steps", 0)),
                        "qp_status": out["final_qp_status"][0]
                        if out.get("final_qp_status") else "not_run",
                        "seconds": float(out["timing"]["total_s"]),
                        "safety_png": os.path.relpath(safety_png, PROJECT_ROOT),
                        "steps_png": os.path.relpath(steps_png, PROJECT_ROOT)})
        print(json.dumps(summary[-1], ensure_ascii=False), flush=True)

    # ---- contact sheet with the safety geometry ---------------------------
    if panels:
        ncols = 4
        nrows = int(np.ceil(len(panels) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.4 * nrows))
        axes = np.atleast_2d(axes)
        for k, panel in enumerate(panels):
            ax = axes[k // ncols, k % ncols]
            _draw_base(ax, panel["occ"], panel["mask"], grid)
            _draw_ellipses(ax, panel["ellipses"], grid, alpha=0.45, lw=0.5, centres=False)
            if panel["search"] is not None:
                ax.plot(panel["search"][:, 0], panel["search"][:, 1], color="#ff7f0e",
                        lw=1.1, ls="--")
            ax.plot(panel["gt"][:, 0], panel["gt"][:, 1], color="#1f77b4", lw=1.2)
            ax.plot(panel["pred"][:, 0], panel["pred"][:, 1], color="#d62728", lw=1.6)
            ax.set_title("#%d cells %d coll %.2f len %.0f/%.0f"
                         % (panel["index"], panel["cells"], panel["collision"],
                            panel["length"], panel["gt_length"]), fontsize=6)
        for k in range(len(panels), nrows * ncols):
            axes[k // ncols, k % ncols].axis("off")
        fig.suptitle("corridor(green) + ellipses(purple) + searched path(orange) + GT(blue) "
                     "vs prediction(red) — %d %s scenes" % (len(panels), opts.split), fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "safety_grid.png"), dpi=100)
        plt.close(fig)

    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    if summary:
        print("scenes %d  valid %.1f%%  collision-free %.1f%%  mean collision %.3f  "
              "corridor cells median %d"
              % (len(summary), 100 * np.mean([s["valid"] for s in summary]),
                 100 * np.mean([s["collision_rate"] == 0 for s in summary]),
                 float(np.mean([s["collision_rate"] for s in summary])),
                 int(np.median([s["corridor_cells"] for s in summary]))))
    print("figures:", outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
