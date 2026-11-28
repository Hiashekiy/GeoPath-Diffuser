"""Per-denoising-step diagnostic figure for ONE (colliding) sample.

Panels: one per EXECUTED reverse step plus the final post-QP trajectory.  Every
panel shows the map, the GT path, the guidance-field search path, the 64 PathSafe
ellipses, the convex corridor the constraint pack was built from, the clean
prediction of THAT step, and - in the title - that step's own collision amount.

The final panel shows the trajectory after the terminal convex QP together with
its QP status / certificate / correction.

Usage::

    python scripts/visualize_cpcg_steps.py --config configs/c64_t12.yaml \
        --checkpoint outputs/checkpoints/cpcg_t12_100ep/t12_ep070.pt \
        --split test --scan 40 --mode cpcg --device cuda

Writes ``outputs/figures/cpcg_steps/<tag>.png`` and ``<tag>_metrics.json``.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import PROJECT_ROOT, base_parser, load  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gp_diffuser.factory import build_pipeline  # noqa: E402
from gp_diffuser.evaluation.metrics import corridor_violation_m  # noqa: E402
from gp_diffuser.safety.collision_query import curve_collision_intervals  # noqa: E402
from gp_diffuser.safety.convex_region import halfspaces_to_vertices  # noqa: E402
from gp_diffuser.training.real_dataset import RealTrajectoryDataset  # noqa: E402
from gp_diffuser.training.dataset import SyntheticTrajectoryDataset  # noqa: E402


def _collision(occ_np: np.ndarray, curve_scene: np.ndarray) -> dict:
    """Segment-level collision of one curve (the project's authoritative test)."""
    info = curve_collision_intervals(occ_np, np.asarray(curve_scene, dtype=np.float64))
    return {"fraction": float(info["collision_fraction"]),
            "intervals": int(len(info["intervals"])),
            "length_scene": float(info.get("collision_length", 0.0))}


def _regions_from_pack(pack, sample: int = 0):
    """Corridor polygons ``[(region_id, vertices)]``, one per constrained region.

    Every piece of a region carries the SAME half-space set (the region is
    convex), so one piece per ``piece_region_id`` reconstructs the cell exactly.
    """
    if pack is None:
        return []
    ids = pack.piece_region_id[sample].detach().cpu().numpy()
    pmask = pack.piece_mask[sample].detach().cpu().numpy()
    A = pack.piece_A[sample].detach().cpu().numpy()
    b = pack.piece_b[sample].detach().cpu().numpy()
    fmask = pack.face_mask[sample].detach().cpu().numpy()
    seen, out = set(), []
    for p in range(len(ids)):
        if not pmask[p]:
            continue
        rid = int(ids[p])
        if rid in seen:
            continue
        seen.add(rid)
        faces = fmask[p]
        if not faces.any():
            continue
        poly = halfspaces_to_vertices(A[p][faces].astype(np.float64),
                                      b[p][faces].astype(np.float64))
        if poly is not None and len(poly) >= 3:
            out.append((rid, poly))
    return out


def _cells_near(polys_m, curves_m, radius_m: float):
    """Keep only the cells a curve actually travels through (else 124 fills blur)."""
    keep = []
    for rid, poly in polys_m:
        hit = False
        for curve in curves_m:
            if curve is None or len(curve) == 0:
                continue
            d = np.linalg.norm(curve[:, None, :] - poly[None, :, :], axis=-1)
            if float(d.min()) <= radius_m:
                hit = True
                break
        if hit:
            keep.append((rid, poly))
    return keep


def _panel(ax, occ_np, res_m, gt_m, start_m, goal_m, *, ellipses_m, regions_m,
           search_m, curves, title, legend=False):
    extent = occ_np.shape[-1] * res_m
    ax.imshow(occ_np, origin="lower", cmap="Greys", alpha=0.85,
              extent=(0.0, extent, 0.0, extent))
    ax.set_xlim(0, extent)
    ax.set_ylim(0, extent)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    for _, poly in regions_m:
        ax.fill(poly[:, 0], poly[:, 1], facecolor="#17becf", alpha=0.18, lw=0.0)
    for e in ellipses_m:
        ax.plot(e[:, 0], e[:, 1], color="#9467bd", lw=0.45, alpha=0.55)
    if search_m is not None:
        ax.plot(search_m[:, 0], search_m[:, 1], color="#ff7f0e", lw=1.4, ls="--")
    ax.plot(gt_m[:, 0], gt_m[:, 1], color="#1f77b4", lw=1.6, alpha=0.85)

    for xy, style in curves:
        ax.plot(xy[:, 0], xy[:, 1], **style)
    ax.plot(*start_m, "o", color="#2ca02c", ms=7, zorder=6)
    ax.plot(*goal_m, "s", color="#d62728", ms=7, zorder=6)
    ax.set_title(title, fontsize=7.5)


def main() -> int:
    parser = base_parser("per-step CPCG diagnostic figure of one colliding sample")
    parser.add_argument("--dataset", default="real", choices=["real", "synthetic"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--indices", default="", help="explicit sample indices")
    parser.add_argument("--scan", type=int, default=40, help="how many samples to screen")
    parser.add_argument("--steps", type=int, default=None, help="executed reverse steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", default="cpcg",
                        choices=["map_only", "safety_only", "cpcg"])
    parser.add_argument("--projection", default="on", choices=["on", "off"],
                        help="run the terminal QP (needed for the final panel)")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--pick", default="final",
                        choices=["final", "preprojection"],
                        help="final = keep a sample whose OUTPUT still collides; "
                             "preprojection = a sample that collided before the QP")
    parser.add_argument("--tag", default="")
    parser.add_argument("--outdir", default="outputs/figures/cpcg_steps")
    opts = parser.parse_args()

    cfg = load(opts)
    pipe = build_pipeline(cfg, device=opts.device, checkpoint=opts.checkpoint)
    dataset = (RealTrajectoryDataset(cfg, opts.split) if opts.dataset == "real"
               else SyntheticTrajectoryDataset(cfg, opts.split))
    sampler, transform = pipe["sampler"], pipe["transform"]
    res_m = float(cfg["dataset"]["res_m"])
    steps = opts.steps or int(cfg["sampling"]["steps"])
    projection = opts.projection == "on"

    explicit = [int(s) for s in opts.indices.split(",") if s.strip()]
    candidates = explicit or list(range(min(opts.scan, len(dataset))))

    chosen = None
    for index in candidates:
        item = dataset[index]
        out = sampler.sample(
            item["occupancy"][None], item["sdf"][None], item["start_goal"][None],
            heading=item["heading"][None], heading_valid=item["heading_valid"][None],
            steps=steps, safety_mode=opts.mode, seed=opts.seed,
            record_curves=True,
            final_projection=projection if opts.mode == "cpcg" else None)
        occ_np = item["occupancy"][0].numpy()
        final_m = out["trajectory_m"][0].detach().cpu().numpy()
        pre_m = transform.to_meters(out["controls_unprojected"][0].detach().cpu().numpy())
        final_coll = _collision(occ_np, transform.to_scene(final_m))["fraction"]
        pre_coll = _collision(occ_np, transform.to_scene(pre_m))["fraction"]
        per_step = [_collision(occ_np, c["x0_scene"][0])["fraction"]
                    for c in out["steps_trace"] if c.get("x0_scene") is not None]
        print("sample %3d | output collision %.4f | pre-QP %.4f | per-step max %.4f"
              % (index, final_coll, pre_coll, max(per_step) if per_step else 0.0),
              flush=True)
        hit = (final_coll > 1e-9) if opts.pick == "final" else (pre_coll > 1e-9)
        if hit:
            chosen = (index, item, out)
            break

    if chosen is None:
        print("no colliding sample found in %d candidates - pass --indices or --pick "
              "preprojection" % len(candidates))
        return 2

    index, item, out = chosen
    occ_np = item["occupancy"][0].numpy()
    gt_m = np.asarray(item["path_gt_m"], dtype=np.float64)
    start_m = transform.to_meters(item["start_goal"][0].numpy())
    goal_m = transform.to_meters(item["start_goal"][1].numpy())

    search_m = None
    if out.get("search_path_m") is not None:
        search_m = transform.to_meters(out["search_path_m"][0].detach().cpu().numpy())
    ellipses_m = []
    if out.get("ellipses") is not None:
        ell = out["ellipses"][0].detach().cpu().numpy()
        for cx, cy, ax_, by_, c2, s2 in ell:
            if not np.isfinite([cx, cy, ax_, by_]).all() or ax_ <= 0 or by_ <= 0:
                continue
            theta = 0.5 * np.arctan2(s2, c2)
            t = np.linspace(0.0, 2.0 * np.pi, 40)
            local = np.stack([ax_ * np.cos(t), by_ * np.sin(t)], axis=-1)
            rot = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
            ellipses_m.append(transform.to_meters(local @ rot.T + np.array([cx, cy])))
    regions_m = [(rid, transform.to_meters(poly))
                 for rid, poly in _regions_from_pack(out.get("pack"))]

    trace = [c for c in out["steps_trace"] if c.get("x0_scene") is not None]
    # the safety geometry (search path / ellipses / corridor) only EXISTS from the
    # creation forward on - drawing it in the earlier map-only panels would claim
    # the correction stage saw something it did not.
    creation_forward = int(out["frames"]["k_safety"])
    final_m = out["trajectory_m"][0].detach().cpu().numpy()
    rows = {
        "sample_index": int(index),
        "mode": opts.mode,
        "steps": int(steps),
        "projection": bool(projection),
        "checkpoint": opts.checkpoint,
        "per_step": [],
        "final": {},
    }
    for c in trace:
        curve_scene = c["x0_scene"][0]
        curve_m = transform.to_meters(curve_scene)
        coll = _collision(occ_np, curve_scene)
        guided = c.get("x0_guided_scene")
        rows["per_step"].append({
            "forward": int(c["forward"]),
            "t": int(c["t"]),
            "t_prev": int(c["t_prev"]),
            "model_branch": c.get("model_branch"),
            "guidance_applied": bool(c.get("constraint_guidance_applied")),
            "collision_fraction": coll["fraction"],
            "collision_intervals": coll["intervals"],
            "corridor_violation_m": float(corridor_violation_m(
                curve_m, out.get("pack"), float(transform.scale),
                offset=transform.offset)),
            "length_m": float(np.linalg.norm(np.diff(curve_m, axis=0), axis=1).sum()),
            "guidance": c.get("guidance"),
            "guided_collision_fraction": (
                None if guided is None
                else _collision(occ_np, guided[0])["fraction"]),
        })

    fin_coll = _collision(occ_np, transform.to_scene(final_m))
    rows["final"] = {
        "collision_fraction": fin_coll["fraction"],
        "collision_intervals": fin_coll["intervals"],
        "min_clearance_m": float(out["minimum_clearance_m"]),
        "endpoint_error_scene": float(out["endpoint_error_scene"]),
        "heading_error_deg": (None if out["heading_error_deg"] is None
                              else out["heading_error_deg"].tolist()),
        "qp_status": out.get("final_qp_status"),
        "qp_iterations": out.get("final_qp_iterations"),
        "qp_certified": bool(out.get("corridor_certified_all", False)),
        "qp_correction_m": float(out.get("final_projection_correction_m", 0.0)),
        "valid": bool(out["valid"]),
    }

    # ---------------------------------------------------------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n_panels = len(trace) + 1
    ncols = min(4, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.1 * ncols, 3.9 * nrows),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for k, (c, meta) in enumerate(zip(trace, rows["per_step"])):
        curves = [(_to_m(transform, c["x0_scene"][0]),
                   dict(color="#d62728", lw=2.0, label="x0_pred"))]
        if c.get("x0_guided_scene") is not None and meta["guidance_applied"]:
            curves.append((_to_m(transform, c["x0_guided_scene"][0]),
                           dict(color="#2ca02c", lw=1.9, ls="--", label="x0_guided")))
        built = meta["forward"] >= creation_forward
        cells = _cells_near(regions_m, [curves[0][0]], radius_m=3.0) if built else []
        extra = "" if built else "\ngeometry not built yet (no search/ellipses/corridor)"
        title = ("step %d/%d  t=%d  %s\ncollision %.2f %% (%d seg)  ·  corridor %.3f m%s"
                 % (k + 1, len(trace), meta["t"],
                    "CPCG" if meta["guidance_applied"] else meta["model_branch"][:11],
                    100.0 * meta["collision_fraction"], meta["collision_intervals"],
                    meta["corridor_violation_m"], extra))
        _panel(axes[k], occ_np, res_m, gt_m, start_m, goal_m,
               ellipses_m=ellipses_m if built else [],
               regions_m=cells,
               search_m=search_m if built else None,
               curves=curves, title=title,
               legend=(k == 0))
        if meta["guidance"]:
            axes[k].text(0.02, 0.02,
                         "J %.4f→%.4f\nΔmax %.2f m  α=%.2f"
                         % (meta["guidance"]["safe_loss_before"],
                            meta["guidance"]["safe_loss_after"],
                            meta["guidance"]["delta_max_m"],
                            meta["guidance"]["alpha"]),
                         transform=axes[k].transAxes, fontsize=6, va="bottom",
                         bbox=dict(fc="white", alpha=0.8, lw=0))

    ax = axes[len(trace)]
    title = ("FINAL (after %s)\ncollision %.2f %% (%d seg) · QP %s · corr %.2f m"
             % ("terminal QP" if projection else "DDIM only",
                100.0 * fin_coll["fraction"], fin_coll["intervals"],
                (out.get("final_qp_status") or ["not_run"])[0],
                rows["final"]["qp_correction_m"]))
    final_cells = _cells_near(regions_m, [final_m], radius_m=3.0)
    _panel(ax, occ_np, res_m, gt_m, start_m, goal_m,
           ellipses_m=ellipses_m, regions_m=final_cells, search_m=search_m,
           curves=[(final_m, dict(color="#6a3d9a", lw=2.6, label="final")),
                   (_to_m(transform,
                          out["controls_unprojected"][0].detach().cpu().numpy()),
                    dict(color="#ff7f0e", lw=1.3, ls=":", label="pre-QP"))],
           title=title)
    ax.text(0.02, 0.02,
            "certified=%s\nvalid=%s\nmin clearance %.2f m"
            % (rows["final"]["qp_certified"], rows["final"]["valid"],
               rows["final"]["min_clearance_m"]),
            transform=ax.transAxes, fontsize=6.5, va="bottom",
            bbox=dict(fc="white", alpha=0.85, lw=0))

    handles = [Line2D([], [], color="#1f77b4", lw=1.6, label="GT path"),
               Line2D([], [], color="#ff7f0e", lw=1.4, ls="--", label="searched path"),
               Line2D([], [], color="#9467bd", lw=0.8, label="PathSafe ellipses"),
               Line2D([], [], color="#17becf", lw=6, alpha=0.3,
                      label="convex corridor cells (near the curve)"),
               Line2D([], [], color="#d62728", lw=2.0, label="x0_pred (clean)"),
               Line2D([], [], color="#2ca02c", lw=1.6, ls="--", label="x0_guided (CPCG)"),
               Line2D([], [], color="#6a3d9a", lw=2.6, label="final trajectory")]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 1.02))
    for ax_idle in axes[n_panels:]:
        ax_idle.axis("off")
    fig.suptitle("%s | %s sample %d | %d executed steps | checkpoint %s"
                 % (opts.mode, opts.dataset + "/" + opts.split, index, len(trace),
                    os.path.basename(opts.checkpoint or "none")), fontsize=11)

    outdir = os.path.join(PROJECT_ROOT, opts.outdir)
    os.makedirs(outdir, exist_ok=True)
    tag = opts.tag or ("%s_%s_%03d" % (opts.mode, opts.split, index))
    png = os.path.join(outdir, tag + ".png")
    fig.savefig(png, dpi=115)
    plt.close(fig)
    rows["figure"] = os.path.relpath(png, PROJECT_ROOT)

    # collision-vs-step curve (the same numbers as the panel titles, readable)
    fig2, ax2 = plt.subplots(figsize=(6.4, 2.8), constrained_layout=True)
    xs = [m["forward"] for m in rows["per_step"]]
    ax2.plot(xs, [100 * m["collision_fraction"] for m in rows["per_step"]],
             "-o", color="#d62728", ms=3, label="x0_pred per step")
    ax2.plot(xs, [100 * (m["guided_collision_fraction"] or 0.0)
                  for m in rows["per_step"]],
             "--o", color="#2ca02c", ms=3, label="x0_guided per step")
    ax2.axhline(100 * fin_coll["fraction"], color="#6a3d9a", lw=1.2,
                label="final (after QP)")
    ax2.set_xlabel("reverse forward index")
    ax2.set_ylabel("collision (%)")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)
    ax2.set_title("collision amount per denoising step", fontsize=9)
    png2 = os.path.join(outdir, tag + "_collision.png")
    fig2.savefig(png2, dpi=120)
    plt.close(fig2)
    rows["collision_figure"] = os.path.relpath(png2, PROJECT_ROOT)

    with open(os.path.join(outdir, tag + "_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False, default=float)

    print(json.dumps({"figure": rows["figure"],
                      "collision_figure": rows["collision_figure"],
                      "per_step_collision": [round(m["collision_fraction"], 4)
                                             for m in rows["per_step"]],
                      "final": rows["final"]}, indent=2, ensure_ascii=False))
    return 0


def _to_m(transform, curve_scene):
    return transform.to_meters(np.asarray(curve_scene, dtype=np.float64))


if __name__ == "__main__":
    raise SystemExit(main())
