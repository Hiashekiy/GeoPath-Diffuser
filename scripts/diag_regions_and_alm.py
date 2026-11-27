"""Diagnose two things the staged demo did NOT show: the convex REGIONS and the ALM.

1. Draws the actual convex corridors the ALM projects onto (``pack.piece_A/b``,
   one piece = one convex cell in scene coordinates) instead of only ellipses.
2. Measures whether those regions are trustworthy: for each piece it samples the
   polygon and reports the fraction of sampled area that sits on BLOCKED cells.
3. Traces the ALM step by step (violation before/after, correction magnitude) and
   the geometry drift it causes versus the same run with ALM off.

    python scripts/diag_regions_and_alm.py --checkpoint <ckpt> --indices 0,1
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

from gp_diffuser.evaluation.metrics import trajectory_metrics  # noqa: E402
from gp_diffuser.factory import build_pipeline  # noqa: E402
from gp_diffuser.training.real_dataset import RealTrajectoryDataset  # noqa: E402

FIG_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures", "pipeline_diag")


def _chamfer(a, b) -> float:
    d = np.linalg.norm(np.asarray(a)[:, None, :] - np.asarray(b)[None, :, :], axis=-1)
    return float(0.5 * (d.min(axis=1).mean() + d.min(axis=0).mean()))


def _clip_polygon(poly: np.ndarray, normal: np.ndarray, offset: float) -> np.ndarray:
    """Sutherland-Hodgman clip of a convex polygon by one half-plane ``n . x <= b``."""
    if len(poly) == 0:
        return poly
    out = []
    n = len(poly)
    for i in range(n):
        cur, nxt = poly[i], poly[(i + 1) % n]
        dc = float(normal @ cur - offset)
        dn = float(normal @ nxt - offset)
        if dc <= 0:
            out.append(cur)
        if (dc <= 0) != (dn <= 0):
            t = dc / (dc - dn)
            out.append(cur + t * (nxt - cur))
    return np.asarray(out, dtype=np.float64).reshape(-1, 2)


def piece_polygons(pack, sample: int = 0) -> list:
    """One convex polygon (scene coords) per REAL constraint piece.

    A piece with no active face constrains nothing; it must not be drawn (the
    earlier version clipped nothing and reported the whole map as a "region").
    """
    A = pack.piece_A[sample].detach().cpu().numpy()      # [P,F,2]
    b = pack.piece_b[sample].detach().cpu().numpy()      # [P,F]
    face = pack.face_mask[sample].detach().cpu().numpy()  # [P,F]
    piece = pack.piece_mask[sample].detach().cpu().numpy()  # [P]
    box = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    polys = []
    for p in range(len(piece)):
        if not piece[p] or not face[p].any():
            continue
        poly = box
        for f in range(A.shape[1]):
            if not face[p, f]:
                continue
            poly = _clip_polygon(poly, A[p, f], b[p, f])
            if len(poly) == 0:
                break
        if len(poly) >= 3:
            polys.append(poly)
    return polys


def _shoelace(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_stats(poly: np.ndarray, occ: np.ndarray, res_m: float,
                  scene_scale: float) -> dict:
    """Area (m^2) and the fraction of the polygon that sits on blocked cells."""
    x0, y0 = poly.min(axis=0)
    x1, y1 = poly.max(axis=0)
    if x1 <= x0 or y1 <= y0:
        return {"area_m2": 0.0, "blocked_fraction": None, "samples": 0}
    gx = np.linspace(x0, x1, 24)
    gy = np.linspace(y0, y1, 24)
    grid = np.stack(np.meshgrid(gx, gy), axis=-1).reshape(-1, 2)
    # point-in-convex-polygon via signed cross products
    inside = np.ones(len(grid), dtype=bool)
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        edge = b - a
        inside &= (edge[0] * (grid[:, 1] - a[1]) - edge[1] * (grid[:, 0] - a[0])) >= -1e-12
    pts = grid[inside]
    if len(pts) == 0:
        return {"area_m2": 0.0, "blocked_fraction": None, "samples": 0}
    scene = np.stack([(pts[:, 0] + 1.0) * 0.5, (pts[:, 1] + 1.0) * 0.5], axis=-1)
    height, width = occ.shape
    cols = np.clip((scene[:, 0] * width).astype(int), 0, width - 1)
    rows = np.clip((scene[:, 1] * height).astype(int), 0, height - 1)
    blocked = occ[rows, cols] > 0
    return {"area_m2": _shoelace(poly) * float(scene_scale) ** 2,
            "blocked_fraction": float(blocked.mean()),
            "samples": int(len(pts))}


def main() -> int:
    parser = base_parser("regions + ALM diagnostics")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--indices", default="0,1,2")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="outputs/reports/pipeline_diag.json")
    opts = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    cfg = load(opts)
    dataset = RealTrajectoryDataset(cfg, opts.split)
    pipe = build_pipeline(cfg, device=opts.device, checkpoint=opts.checkpoint)
    sampler, guidance, transform = pipe["sampler"], pipe["guidance"], pipe["transform"]
    res_m = float(dataset.res_m)
    alm_cfg = sampler._alm_cfg()
    print("alm config:", json.dumps(alm_cfg, ensure_ascii=False), flush=True)
    print("1 scene unit = %.1f m -> max_curve_step_scene %.2f = %.2f m"
          % (transform.scale, alm_cfg.get("max_curve_step_scene", 0.0),
             alm_cfg.get("max_curve_step_scene", 0.0) * transform.scale), flush=True)

    results = []
    for index in [int(s) for s in opts.indices.split(",") if s.strip()]:
        item = dataset[index]
        occ = item["occupancy"][None].clone()
        sdf = item["sdf"][None].clone()
        sg = item["start_goal"][None].clone()
        hd = item["heading"][None].clone() if item.get("heading") is not None else None
        hv = (item["heading_valid"][None].clone()
              if item.get("heading_valid") is not None else None)
        occ_np = occ[0, 0].numpy()
        gt_m = item["path_gt_m"].numpy()

        out_map = sampler.sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                                 steps=opts.steps, safety_mode="map_only", seed=opts.seed)
        guide_m = out_map["trajectory_m"][0].numpy()
        features = sampler.map_encoder.encode(occ.to(sampler.device), sdf.to(sampler.device))
        raw, _ = sampler._generate_safety(
            occ.to(sampler.device), sdf.to(sampler.device), features,
            out_map["trajectory_scene"][0:1], sg.to(sampler.device),
            dict(sampler.cfg.get("corridor") or {}))
        pack = raw["pack_local"]

        regions = []
        if pack is not None:
            for poly in piece_polygons(pack):
                stat = polygon_stats(poly, occ_np, res_m, transform.scale)
                stat["polygon_scene"] = poly.tolist()
                regions.append(stat)

        out_noalm = sampler.sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                                   steps=opts.steps, safety_mode="no_alm", seed=opts.seed)
        out_full = sampler.sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                                  steps=opts.steps, safety_mode="full", seed=opts.seed)
        noalm_m = out_noalm["trajectory_m"][0].numpy()
        full_m = out_full["trajectory_m"][0].numpy()

        def stats(curve):
            m = trajectory_metrics(curve, occ_np, item["sdf"][0].numpy(), res_m)
            return {"collision_rate": float(m["collision_rate"]),
                    "length_m": float(m["length_m"]),
                    "chamfer_gt_m": _chamfer(curve, gt_m)}

        trace = [e for e in (out_full.get("steps_trace") or []) if e.get("alm")]
        drift = [e for e in trace if e["alm"]]
        report = {
            "index": int(index),
            "guide": stats(guide_m),
            "no_alm": stats(noalm_m),
            "alm": stats(full_m),
            "alm_vs_no_alm_correction_m": _chamfer(full_m, noalm_m),
            "alm_steps": len(drift),
            "per_step": [{"forward": e["forward"],
                          "viol_before": e["alm"]["max_violation_before"],
                          "viol_after": e["alm"]["max_violation_after"],
                          "corr_mean_m": e["alm"]["mean_correction_m"],
                          "corr_max_m": e["alm"]["max_correction_m"],
                          "coll_before": e["alm"].get("collision_before"),
                          "coll_after": e["alm"].get("collision_after"),
                          "accepted": sum(e["alm"]["accepted_mask"])}
                         for e in drift],
            "regions": [{k: v for k, v in r.items() if k != "polygon_scene"}
                        for r in regions],
            "region_blocked_fraction_mean": (
                float(np.mean([r["blocked_fraction"] for r in regions
                               if r["blocked_fraction"] is not None])) if regions else None),
        }
        results.append(report)
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)

        # ---------------------------- figure --------------------------------
        fig, axes = plt.subplots(2, 3, figsize=(21, 12))
        axes = axes.ravel()
        scale_m = occ_np.shape[1] * res_m

        def base(ax, title):
            ax.imshow(occ_np, origin="lower", cmap="Greys",
                      extent=[0, scale_m, 0, scale_m], alpha=0.9)
            ax.plot(gt_m[:, 0], gt_m[:, 1], "-", color="#2ca02c", lw=1.2, alpha=0.65,
                    label="GT")
            s_m = transform.to_meters(sg[0, 0].numpy())
            g_m = transform.to_meters(sg[0, 1].numpy())
            ax.plot(*s_m, "o", color="k", ms=6, label="start")
            ax.plot(*g_m, "s", color="k", ms=6, label="goal")
            ax.set_title(title, fontsize=11)
            ax.set_xlim(0, scale_m); ax.set_ylim(0, scale_m)
            ax.tick_params(labelsize=8)

        base(axes[0], "1) map-only guidance (no safety features)")
        axes[0].plot(guide_m[:, 0], guide_m[:, 1], "-", color="#1f77b4", lw=2.4,
                     label="guidance")

        base(axes[1], "2) guidance-field search (Astar-Diffusion module)")
        axes[1].plot(guide_m[:, 0], guide_m[:, 1], "--", color="#1f77b4", lw=1.0,
                     alpha=0.5, label="guidance (seed)")
        if raw["search_path_m"] is not None:
            p = raw["search_path_m"][0].detach().cpu().numpy()
            axes[1].plot(p[:, 0], p[:, 1], "-", color="#d62728", lw=2.4,
                         label="searched path")

        base(axes[2], "3) convex regions (ALM constraints) / obstacles inside them")
        for r in regions:
            poly = np.asarray(r["polygon_scene"], dtype=np.float64)
            xy = np.stack([(poly[:, 0] + 1.0) * 0.5 * scale_m,
                           (poly[:, 1] + 1.0) * 0.5 * scale_m], axis=-1)
            bad = (r["blocked_fraction"] or 0.0) > 0.02
            axes[2].fill(xy[:, 0], xy[:, 1], facecolor="#d62728" if bad else "#9467bd",
                         alpha=0.20, edgecolor="#4b1f6f", lw=0.8)
        if raw["ellipse_centers_scene"] is not None:
            centers_m = transform.to_meters(raw["ellipse_centers_scene"][0].detach().cpu().numpy())
            axes_m = transform.lengths_to_meters(
                raw["ellipse"][0][:, 2:4].detach().cpu().numpy())
            c2 = raw["ellipse"][0][:, 4].detach().cpu().numpy()
            s2 = raw["ellipse"][0][:, 5].detach().cpu().numpy()
            for c, ax_m, cc, ss in zip(centers_m, axes_m, c2, s2):
                axes[2].add_patch(Ellipse((c[0], c[1]), 2 * ax_m[0], 2 * ax_m[1],
                                          angle=np.degrees(0.5 * np.arctan2(ss, cc)),
                                          fill=False, ec="#1f77b4", lw=0.6, alpha=0.5))
        axes[2].plot([], [], "-", color="#9467bd", lw=6, alpha=0.3, label="free region")
        axes[2].plot([], [], "-", color="#d62728", lw=6, alpha=0.3, label="region>2% blocked")
        axes[2].legend(fontsize=8)

        base(axes[3], "4) safety-injected diffusion (no ALM)")
        axes[3].plot(guide_m[:, 0], guide_m[:, 1], "--", color="#1f77b4", lw=1.0,
                     alpha=0.5, label="guidance")
        axes[3].plot(noalm_m[:, 0], noalm_m[:, 1], "-", color="#ff7f0e", lw=2.4,
                     label="no-ALM")

        base(axes[4], "5) with ALM convex projection")
        axes[4].plot(noalm_m[:, 0], noalm_m[:, 1], "--", color="#ff7f0e", lw=1.4,
                     alpha=0.7, label="no-ALM")
        axes[4].plot(full_m[:, 0], full_m[:, 1], "-", color="#6a3d9a", lw=2.6,
                     label="ALM")

        ax = axes[5]
        if report["per_step"]:
            xs = [p["forward"] for p in report["per_step"]]
            ax.plot(xs, [p["viol_before"] for p in report["per_step"]], "-",
                    color="#d62728", label="violation before (scene)")
            ax.plot(xs, [p["viol_after"] for p in report["per_step"]], "-",
                    color="#2ca02c", label="violation after (scene)")
            ax.set_xlabel("guided reverse step"); ax.set_ylabel("max violation (scene)")
            ax2 = ax.twinx()
            ax2.plot(xs, [p["corr_mean_m"] for p in report["per_step"]], "--",
                     color="#1f77b4", label="mean correction (m)")
            ax2.plot(xs, [p["corr_max_m"] for p in report["per_step"]], ":",
                     color="#9467bd", label="max correction (m)")
            ax2.set_ylabel("correction (m)")
            h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="best")
            ax.set_title("6) ALM trace: what each projection does", fontsize=11)
        for ax_ in axes:
            ax_.tick_params(labelsize=8)
        fig.suptitle("val %d | regions & ALM diagnostics | steps=%d" % (index, opts.steps),
                     fontsize=13)
        fig.tight_layout()
        png = os.path.join(FIG_DIR, "regions_alm_val%03d.png" % index)
        os.makedirs(FIG_DIR, exist_ok=True)
        fig.savefig(png, dpi=115, bbox_inches="tight")
        plt.close(fig)
        report["figure"] = os.path.relpath(png, PROJECT_ROOT)
        print("figure:", report["figure"], flush=True)

    out_path = os.path.join(PROJECT_ROOT, opts.out)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"checkpoint": opts.checkpoint, "steps": opts.steps,
                   "scenes": results}, fh, indent=2, ensure_ascii=False)
    print("written:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
