"""Inference test on an external low-density octile map (no training, read-only).

The pipeline expects 0.25 m/cell over a 64 m (256x256) window, so a bigger octile
map is *cropped* (never resized): the crop keeps the real metric scale and feeds
the same frozen encoders / DiT / ALM as the training scenes.

Usage::

    python scripts/test_octile_map.py --map London_1_256_part_0_0.map \
        --window auto --samples 4 --steps 20
"""
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

from gp_diffuser.evaluation.metrics import trajectory_metrics  # noqa: E402
from gp_diffuser.factory import build_pipeline  # noqa: E402
from gp_diffuser.frames import SceneTransform  # noqa: E402

TILE_FREE = ".G"
TILE_BLOCKED = "@OS"


def load_octile(path: str):
    lines = open(path, "r", encoding="utf-8", errors="replace").read().splitlines()
    head = {}
    i = 0
    while i < len(lines) and lines[i].strip() != "map":
        parts = lines[i].split()
        if len(parts) == 2:
            head[parts[0]] = parts[1]
        i += 1
    rows = [ln for ln in lines[i + 1:] if len(ln.strip()) >= int(head.get("width", 0))]
    grid = np.array([[1 if ch in TILE_BLOCKED else 0 for ch in row[:int(head["width"])]]
                     for row in rows[:int(head["height"])]], dtype=np.uint8)
    return grid, head


def sdf_from_occupancy(occ: np.ndarray, res_m: float) -> np.ndarray:
    """Metres-signed distance, same convention as the PathSafe distance fields.

    Free space positive, blocked negative, outside the map treated as blocked.
    """
    blocked = occ > 0
    free = ~blocked
    d_free = ndimage.distance_transform_edt(free) * res_m      # distance to blocked
    d_blocked = ndimage.distance_transform_edt(blocked) * res_m
    sdf = np.where(free, d_free, -d_blocked)
    # pad the outside as blocked (distance from the border inward)
    border = np.minimum.reduce([np.arange(occ.shape[0])[:, None].repeat(occ.shape[1], 1),
                                np.arange(occ.shape[0])[::-1][:, None].repeat(occ.shape[1], 1),
                                np.arange(occ.shape[1])[None, :].repeat(occ.shape[0], 0),
                                np.arange(occ.shape[1])[::-1][None, :].repeat(occ.shape[0], 0)])
    return np.where(border == 0, np.minimum(sdf, -0.5 * res_m), sdf)


def pick_window(grid: np.ndarray, size: int = 256, stride: int = 8):
    best = None
    h, w = grid.shape
    for y in range(0, h - size + 1, stride):
        for x in range(0, w - size + 1, stride):
            frac = float(grid[y:y + size, x:x + size].mean())
            if best is None or frac < best[0]:
                best = (frac, x, y)
    return best


def pick_endpoints(occ: np.ndarray, clearance: np.ndarray, res_m: float,
                   target_m: float = 30.0, min_clear_m: float = 0.6, seed: int = 0,
                   require_blocked: bool = False):
    """A start/goal pair with a realistic chord and a mostly-free direct line."""
    free = (occ == 0) & (clearance >= min_clear_m)
    ys, xs = np.nonzero(free)
    if len(ys) < 2:
        return None
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    rng = np.random.default_rng(seed)
    best, best_score = None, 1e9
    n = len(pts)
    for _ in range(20000):
        i, j = rng.integers(0, n, 2)
        a, b = pts[i], pts[j]
        d = float(np.linalg.norm(b - a)) * res_m
        score = abs(d - target_m)
        if score >= best_score:
            continue
        t = np.linspace(0.0, 1.0, 40)[:, None]
        line = a + t * (b - a)
        cols = np.clip(line[:, 0].astype(int), 0, occ.shape[1] - 1)
        rows = np.clip(line[:, 1].astype(int), 0, occ.shape[0] - 1)
        blocked_frac = float(occ[rows, cols].mean())
        if require_blocked:
            # the chord must cross obstacles, so the model has to bend around them
            if not (0.12 <= blocked_frac <= 0.60):
                continue
            score = abs(d - target_m) + abs(blocked_frac - 0.3)
        elif blocked_frac > 0.05:
            continue
        best, best_score = (a, b), score
    return best


def main() -> int:
    parser = base_parser("inference test on an external octile map")
    parser.add_argument("--map", required=True)
    parser.add_argument("--window", default="auto", help='"auto" or "x,y"')
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--checkpoint", default="outputs/checkpoints/last.pt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-len-m", type=float, default=30.0)
    parser.add_argument("--blocked-chord", action="store_true",
                        help="force the straight chord to cross obstacles")
    parser.add_argument("--outdir", default="outputs/figures/octile_test")
    opts = parser.parse_args()

    cfg = load(opts)
    grid, head = load_octile(opts.map)
    res_m = float(head.get("resolution", head.get("cell_length_m", 0.25)))
    print("octile map %s: %dx%d @ %.4f m/blocked %.3f"
          % (os.path.basename(opts.map), grid.shape[1], grid.shape[0], res_m, grid.mean()))

    if opts.window == "auto":
        frac, x0, y0 = pick_window(grid, opts.size)
    else:
        x0, y0 = [int(v) for v in opts.window.split(",")]
        frac = float(grid[y0:y0 + opts.size, x0:x0 + opts.size].mean())
    occ = grid[y0:y0 + opts.size, x0:x0 + opts.size]
    print("window (%d,%d) %dx%d: blocked %.3f" % (x0, y0, opts.size, opts.size, frac))

    sdf = sdf_from_occupancy(occ, res_m)
    clearance = np.where(occ == 0, ndimage.distance_transform_edt(occ == 0) * res_m, 0.0)

    transform = SceneTransform(res_m * opts.size, opts.size)
    pipe = build_pipeline(cfg, device=opts.device, checkpoint=opts.checkpoint)
    sampler = pipe["sampler"]

    ends = pick_endpoints(occ, clearance, res_m, target_m=opts.target_len_m,
                          require_blocked=opts.blocked_chord)
    if ends is None:
        print("no usable start/goal in this window")
        return 2
    a, b = ends
    start_m = np.array([a[0], a[1]]) * res_m
    goal_m = np.array([b[0], b[1]]) * res_m
    sg = torch.as_tensor(transform.to_scene(np.stack([start_m, goal_m])),
                         dtype=torch.float32, device=opts.device)[None]
    chord = goal_m - start_m
    heading = torch.as_tensor(np.stack([chord, chord]) / np.linalg.norm(chord),
                              dtype=torch.float32, device=opts.device)[None]
    print("start %.1f,%.1f -> goal %.1f,%.1f (chord %.1f m)"
          % (start_m[0], start_m[1], goal_m[0], goal_m[1], np.linalg.norm(chord)))

    occ_t = torch.as_tensor(occ, dtype=torch.float32, device=opts.device)[None, None]
    sdf_t = torch.as_tensor(sdf, dtype=torch.float32, device=opts.device)[None, None]
    outdir = os.path.join(PROJECT_ROOT, opts.outdir)
    os.makedirs(outdir, exist_ok=True)

    rows = []
    for k in range(opts.samples):
        out = sampler.sample(occ_t, sdf_t, sg, heading=heading,
                             heading_valid=torch.ones(1, 2, dtype=torch.bool,
                                                      device=opts.device),
                             steps=opts.steps, seed=opts.seed + k, record_curves=True)
        traj_m = out["trajectory_m"][0].detach().cpu().numpy()
        metrics = trajectory_metrics(traj_m, occ.astype(np.float32),
                                     sdf.astype(np.float32), res_m)
        rows.append({"sample": k, "valid": bool(out["valid"]),
                     "collision_rate": float(metrics["collision_rate"]),
                     "min_clearance_m": float(metrics["min_clearance_m"]),
                     "length_m": float(metrics["length_m"]),
                     "endpoint_error_m": float(np.linalg.norm(traj_m[0] - start_m)
                                               + np.linalg.norm(traj_m[-1] - goal_m)),
                     "alm_steps": len(out["alm_stats"]),
                     "corridor_cells": int((out["corridor_info"] or {}).get("local_cells", 0)),
                     "seconds": float(out["timing"]["total_s"]),
                     "trajectory_m": traj_m, "trace": out["steps_trace"],
                     "search": (None if out.get("search_path_m") is None
                                else out["search_path_m"][0].detach().cpu().numpy()),
                     "ellipse_centres": (None if out.get("ellipse_centers_scene") is None
                                         else out["ellipse_centers_scene"][0].detach().cpu().numpy())})

    cols = min(opts.samples, 4)
    rows_n = int(np.ceil(opts.samples / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.4 * cols, 4.4 * rows_n))
    axes = np.atleast_2d(axes)
    for i, row in enumerate(rows):
        ax = axes[i // cols, i % cols]
        ax.imshow(occ, cmap="Greys", origin="lower", vmin=0, vmax=1, interpolation="nearest")
        s_px = (start_m / res_m, goal_m / res_m)
        ax.plot(row["trajectory_m"][:, 0] / res_m, row["trajectory_m"][:, 1] / res_m,
                color="#d62728", lw=2.0, label="prediction")
        ax.plot([s_px[0][0], s_px[1][0]], [s_px[0][1], s_px[1][1]],
                color="#1f77b4", lw=1.0, ls=":", label="chord")
        ax.scatter([s_px[0][0], s_px[1][0]], [s_px[0][1], s_px[1][1]], s=70,
                   c=["#2ca02c", "#d62728"], edgecolors="k", zorder=6)
        ax.set_title("seed %d  coll %.3f  clear %.2fm  len %.1fm  cells %d"
                     % (row["sample"], row["collision_rate"], row["min_clearance_m"],
                        row["length_m"], row["corridor_cells"]), fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=6, loc="lower left")
    for i in range(len(rows), rows_n * cols):
        axes[i // cols, i % cols].axis("off")
    fig.suptitle("external octile map %s — window (%d,%d) blocked %.1f%% — %d seeds"
                 % (os.path.basename(opts.map), x0, y0, 100 * frac, opts.samples), fontsize=10)
    fig.tight_layout()
    png = os.path.join(outdir, "octile_predictions.png")
    fig.savefig(png, dpi=110)
    plt.close(fig)

    summary = {"map": os.path.basename(opts.map), "map_shape": list(grid.shape),
               "map_blocked": float(grid.mean()), "window": [x0, y0], "window_blocked": frac,
               "resolution_m": res_m, "chord_m": float(np.linalg.norm(chord)),
               "samples": [{k: v for k, v in r.items() if k not in ("trajectory_m", "trace",
                                                                    "search", "ellipse_centres")}
                           for r in rows],
               "figure": os.path.relpath(png, PROJECT_ROOT)}
    with open(os.path.join(outdir, "octile_test.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary["samples"], indent=2, ensure_ascii=False))
    print("figure:", summary["figure"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
