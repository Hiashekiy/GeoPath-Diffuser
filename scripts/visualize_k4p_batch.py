"""Step-by-step overview of many k4p test samples in one figure.

Same source data as ``scripts/visualize_cpcg_steps.py`` (the sampler's own
``steps_trace`` / ``x0_guided_scene`` / terminal QP), just laid out as
"one row per sample, four snapshots per row" so a batch can be scanned at once:

    col 1  forward 0      map-only prediction, no safety geometry yet
    col 2  forward 1      first safety step: geometry built, CPCG applied
    col 3  forward k      middle of the reverse loop (k = len//2)
    col 4  final          after the terminal convex QP

Collision is the project's segment-level test (``curve_collision_intervals``),
the same one ``visualize_cpcg_steps`` prints.

Usage::

    python scripts/visualize_k4p_batch.py --split test --count 12 --steps 12 \
        --checkpoint outputs/checkpoints/k4p_d2_t12_d4/d2_last.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import PROJECT_ROOT, base_parser, load  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from gp_diffuser.factory import build_pipeline  # noqa: E402
from gp_diffuser.training.real_dataset import RealTrajectoryDataset  # noqa: E402
from visualize_cpcg_steps import _collision  # noqa: E402


def pick_indices(n_total: int, count: int, extra: list) -> list:
    idx = sorted({int(round(v)) for v in np.linspace(0, n_total - 1, int(count))})
    for e in extra:
        if 0 <= e < n_total and e not in idx:
            idx.append(int(e))
    return idx


def run_one(sampler, transform, item, steps: int, seed: int) -> dict:
    with torch.no_grad():
        out = sampler.sample(item["occupancy"][None], item["sdf"][None],
                             item["start_goal"][None], heading=item["heading"][None],
                             heading_valid=item["heading_valid"][None],
                             steps=steps, safety_mode="cpcg", seed=seed,
                             record_curves=True)
    occ = item["occupancy"][0].numpy()
    trace = out["steps_trace"]
    frames = []
    for entry in trace:
        raw = entry.get("x0_scene")
        guided = entry.get("x0_guided_scene")
        frames.append({
            "mode": entry.get("model_branch"),
            "collision": (_collision(occ, raw[0])["fraction"] if raw is not None else None),
            "guided_collision": (_collision(occ, guided[0])["fraction"]
                                 if guided is not None else None),
            "raw": (None if raw is None else transform.to_meters(raw[0])),
            "guided": (None if guided is None else transform.to_meters(guided[0])),
        })
    final_m = out["trajectory_m"][0].detach().cpu().numpy()
    final = {"collision": _collision(occ, transform.to_scene(final_m))["fraction"],
             "min_clearance_m": (float(out["min_clearance_m"][0])
                                 if out.get("min_clearance_m") is not None else None),
             "corridor_violation_m": (float(out["final_corridor_violation_scene"].flatten()[0]
                                               * transform.scale)
                                      if out.get("final_corridor_violation_scene") is not None else None),
             "qp": (out.get("final_qp_status") or ["-"])[0],
             "curve": final_m}
    guide = (None if out.get("search_path_m") is None
             else transform.to_meters(out["search_path_m"][0].detach().cpu().numpy()))
    ellipses = []
    if out.get("ellipses") is not None:
        for cx, cy, ax_, by_, c2, s2 in out["ellipses"][0].detach().cpu().numpy():
            if not np.isfinite([cx, cy, ax_, by_]).all() or ax_ <= 0 or by_ <= 0:
                continue
            ellipses.append((cx, cy, ax_, by_, 0.5 * np.arctan2(s2, c2)))
    return {"frames": frames, "final": final, "guide": guide, "ellipses": ellipses,
            "gt_m": np.asarray(item["path_gt_m"])}


def draw(ax, geometry, curve, transform, title, colour="red"):
    occ = geometry["occ"] if "occ" in geometry else None
    extent = transform.extent_m
    if occ is not None:
        ax.imshow(occ, origin="lower", extent=(0, extent, 0, extent), cmap="Greys",
                  vmin=0.0, vmax=1.4, interpolation="nearest")
    ax.plot(geometry["gt_m"][:, 0], geometry["gt_m"][:, 1], color="white", lw=1.6,
            ls="--", alpha=0.85)
    if geometry["guide"] is not None:
        ax.plot(geometry["guide"][:, 0], geometry["guide"][:, 1], color="deepskyblue",
                lw=1.0, alpha=0.9)
    for cx, cy, ax_, by_, th in geometry["ellipses"]:
        phi = np.linspace(0.0, 2.0 * np.pi, 48)
        local = np.stack([ax_ * np.cos(phi), by_ * np.sin(phi)], -1)
        rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        pts = local @ rot.T + np.array([cx, cy])
        ax.plot(pts[:, 0], pts[:, 1], color="orange", lw=0.4, alpha=0.45)
    ax.plot(curve[:, 0], curve[:, 1], color=colour, lw=1.9)
    lo = np.minimum(geometry["gt_m"].min(0), curve.min(0)) - 12.0
    hi = np.maximum(geometry["gt_m"].max(0), curve.max(0)) + 12.0
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_title(title, fontsize=7.5, loc="left")
    ax.tick_params(labelsize=6)
    ax.set_aspect("equal")


def main() -> int:
    parser = base_parser("k4p: multi-sample step-by-step overview")
    parser.add_argument("--split", default="test")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--indices", default="", help="extra explicit indices")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--outdir", default=os.path.join(PROJECT_ROOT, "outputs", "figures",
                                                         "k4p_batch"))
    opts = parser.parse_args()

    cfg = load(opts)
    pipe = build_pipeline(cfg, device=opts.device, checkpoint=opts.checkpoint)
    dataset = RealTrajectoryDataset(cfg, opts.split)
    steps = int(opts.steps or cfg["sampling"]["steps"])
    indices = pick_indices(len(dataset), opts.count,
                           [int(s) for s in opts.indices.split(",") if s.strip()])
    transform = pipe["transform"]

    results = []
    for index in indices:
        item = dataset[index]
        res = run_one(pipe["sampler"], transform, item, steps, opts.seed)
        res["index"] = index
        res["occ"] = item["occupancy"][0].numpy()
        results.append(res)
        f = res["frames"]
        print("sample %3d | fwd0 %.3f | guided0 %.3f | mid %.3f | final %.4f | "
              "clearance %s | corridor %s | qp %s"
              % (index, f[0]["collision"] or 0.0, (f[1]["guided_collision"] or 0.0),
                 (f[len(f) // 2]["guided_collision"] or 0.0), res["final"]["collision"],
                 ("%.2f" % res["final"]["min_clearance_m"]
                  if res["final"]["min_clearance_m"] is not None else "-"),
                 ("%.2f" % res["final"]["corridor_violation_m"]
                  if res["final"]["corridor_violation_m"] is not None else "-"),
                 res["final"]["qp"]), flush=True)

    rows = len(results)
    fig, axes = plt.subplots(rows, 4, figsize=(13.0, 3.1 * rows), squeeze=False)
    for r, res in enumerate(results):
        frames = res["frames"]
        mid = len(frames) // 2
        picks = [(frames[0], "raw", "fwd0  map-only prediction, no geometry"),
                 (frames[min(1, len(frames) - 1)], "guided",
                  "fwd1  geometry built + CPCG"),
                 (frames[mid], "guided", "fwd%d  mid reverse loop (CPCG)" % mid),
                 (None, None, "FINAL  after terminal QP")]
        for c, (frame, key, label) in enumerate(picks):
            ax = axes[r][c]
            geometry = {"occ": res["occ"], "gt_m": res["gt_m"], "guide": res["guide"],
                        "ellipses": res["ellipses"]}
            if frame is None:
                curve, coll, colour = res["final"]["curve"], res["final"]["collision"], "lime"
            else:
                curve = frame[key]
                coll = frame["collision"] if key == "raw" else frame["guided_collision"]
                colour = "red" if key == "raw" else "orange"
            if curve is None:
                ax.axis("off")
                continue
            draw(ax, geometry, curve, transform,
                 "%s\ncollision %.1f%%" % (label, 100.0 * (coll or 0.0)),
                 colour=colour)
            if c == 0:
                ax.set_ylabel("sample %d" % res["index"], fontsize=8)
    fig.suptitle("k4p test samples, 12 reverse steps + terminal QP "
                 "(white dashed = GT, cyan = guidance search path, orange = 64 ellipses)",
                 fontsize=10, y=0.998)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    os.makedirs(opts.outdir, exist_ok=True)
    png = os.path.join(opts.outdir, "overview_%s_%d_samples.png" % (opts.split, rows))
    fig.savefig(png, dpi=125)
    payload = {"split": opts.split, "steps": steps, "checkpoint": opts.checkpoint,
               "indices": [r["index"] for r in results],
               "rows": [{"index": r["index"],
                         "final": {k: v for k, v in r["final"].items() if k != "curve"},
                         "frames": [{"mode": f["mode"], "collision": f["collision"],
                                     "guided_collision": f["guided_collision"]}
                                    for f in r["frames"]]} for r in results]}
    with open(os.path.join(opts.outdir, "overview_%s.json" % opts.split), "w",
              encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print("figure:", png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
