"""Run the project's safety-geometry pipeline on a few k4p samples and plot it.

Per sample, exactly the order ``safety_cache.build_safety_cache`` /
``sampler_conditional_dit`` use (nothing is re-implemented here):

    GT curve (metres)  -> ReferenceGuidanceSearch.search        -> search path
    search path        -> SafetyConditionGenerator (frozen)     -> 64 ellipses
    ellipses + map     -> local_regions.build_region_table      -> convex cells
    cells + Q_ref      -> build_sampled_constraint_pack_from_tables -> M-point pack
    pack               -> A_j(l) N_l Q <= b_j(l)                -> violations

The figures show metres.  ``--show-labels`` also draws the dataset's own
``ellipse_shape4_gt_q64`` labels (centres = 64 uniform-arc-length points of
``pdg_path_points``, the convention the upstream label pass documents).

Usage::

    python scripts/prepare_k4p_dataset.py --limit 4 --out data/_demo_k4p
    python scripts/demo_k4p_geometry.py --data-root data/_demo_k4p --limit 4
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

from gp_diffuser.frames import SceneTransform  # noqa: E402
from gp_diffuser.guidance.reference_search import ReferenceGuidanceSearch  # noqa: E402
from gp_diffuser.safety.bspline_sample_constraints import (  # noqa: E402
    build_sampled_constraint_pack_from_tables, sampled_points)
from gp_diffuser.safety.convex_region import halfspaces_to_vertices  # noqa: E402
from gp_diffuser.safety.local_regions import build_region_table  # noqa: E402
from gp_diffuser.safety.safety_features import (  # noqa: E402
    SafetyConditionGenerator, ellipse_shape4_to_scene, load_pathsafe,
    resample_arclength)
from gp_diffuser.trajectory.bspline import BSplineCodec  # noqa: E402
from gp_diffuser.training.real_dataset import RealTrajectoryDataset  # noqa: E402

SOURCE_ROOT = os.path.join(PROJECT_ROOT, "data", "160m-k4p-c32q64")


def ellipse_xy(center, axes, theta, count: int = 72) -> np.ndarray:
    """``center + R(theta) [a cos, b sin]`` -> ``[count,2]`` closed outline."""
    phi = np.linspace(0.0, 2.0 * np.pi, int(count))
    local = np.stack([axes[0] * np.cos(phi), axes[1] * np.sin(phi)], axis=-1)
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    return local @ rot.T + np.asarray(center)[None]


def draw_polygon(ax, A, b, transform, **kwargs) -> bool:
    """Half-spaces are in scene units; the axes are metres."""
    verts = halfspaces_to_vertices(A, b)
    if verts is None or len(verts) < 3:
        return False
    ax.add_patch(plt.Polygon(transform.to_meters(verts), closed=True, **kwargs))
    return True


def load_labels(split: str, index: int, source_index: int):
    """Dataset's own q64 label, centred on 64 arc-length points of the pdg path."""
    folder = os.path.join(SOURCE_ROOT, split)
    shape4 = np.load(os.path.join(folder, "ellipse_shape4_gt_q64.npy"), mmap_mode="r")[source_index]
    valid = np.load(os.path.join(folder, "shape_valid_q64.npy"), mmap_mode="r")[source_index]
    path = np.load(os.path.join(folder, "pdg_path_points.npy"), mmap_mode="r")[source_index]
    return np.asarray(shape4), np.asarray(valid), np.asarray(path)


def run_sample(cfg, dataset, index, net, codec, transform, scope, show_labels):
    scene = dataset.raw_scene(index)
    occ = np.asarray(scene["occupancy"], dtype=np.float32)
    sdf = np.asarray(scene["sdf"], dtype=np.float32)
    gt_m = np.asarray(scene["gt_path_m"], dtype=np.float64)
    start_m, goal_m = np.asarray(scene["start_m"]), np.asarray(scene["goal_m"])

    guidance = ReferenceGuidanceSearch(cfg.get("guidance") or {},
                                       res_m=float(cfg["dataset"]["res_m"]))
    found = guidance.search(occ, gt_m, start_m, goal_m)
    search_ok = found.get("path_xy") is not None and bool(found.get("valid", True))
    path_m = np.asarray(found["path_xy"] if search_ok else gt_m, dtype=np.float64)

    occ_t = torch.as_tensor(occ, dtype=torch.float32)[None, None]
    sdf_t = torch.as_tensor(sdf, dtype=torch.float32)[None, None]
    gen = SafetyConditionGenerator(net, num_queries=int(cfg["safety_condition"]["num_ellipses"]),
                                   res_m=float(cfg["dataset"]["res_m"]))
    safety = gen(occ_t, sdf_t, torch.as_tensor(path_m, dtype=torch.float32)[None])

    centers_m = safety["ellipse_center"][0].numpy()
    axes_m = np.stack([safety["ellipse"][0, :, 2].numpy(),
                       safety["ellipse"][0, :, 3].numpy()], axis=-1)
    theta_scene = np.arctan2(safety["ellipse"][0, :, 5].numpy(),
                             safety["ellipse"][0, :, 4].numpy()) / 2.0
    centers_scene = torch.as_tensor(transform.to_scene(centers_m), dtype=torch.float32)
    shape4_scene = ellipse_shape4_to_scene(safety["ellipse_shape4"], transform.scale)
    anchors = torch.linspace(0.0, 1.0, centers_scene.shape[0], dtype=torch.float64)
    path_scene = torch.as_tensor(transform.to_scene(path_m), dtype=torch.float32)
    curve_scene = torch.as_tensor(transform.to_scene(gt_m), dtype=torch.float32)

    table, info = build_region_table(
        occ_t, centers_scene, shape4_scene[0], anchors, curve_scene=curve_scene,
        gamma_scene=path_scene, config=cfg.get("corridor") or {}, scope=scope,
        scene_scale=transform.scale)

    controls_scene = dataset[index]["controls_gt"][None]
    pack = build_sampled_constraint_pack_from_tables(
        codec, [table], controls_scene,
        constraint_points=int(cfg["sampling"]["constraint_points"]),
        arc_length_lookup_points=int(cfg["sampling"]["arc_length_lookup_points"]),
        margin=float((cfg.get("corridor") or {}).get("constraint_margin_scene", 0.0)),
        include_box=False)

    points = sampled_points(pack, controls_scene, 0) if pack is not None else None
    violation = None
    if pack is not None:
        A = pack.sample_A[0].numpy()
        b = pack.sample_b[0].numpy()
        active = pack.sample_mask[0].numpy()[:, None] & pack.face_mask[0].numpy()
        g = np.einsum("mfk,mk->mf", A, points) - b
        g = np.where(active, g, -np.inf)
        worst = g.max(axis=1)
        violation = np.where(np.isfinite(worst), worst, 0.0) * transform.scale
    stats = {
        "index": int(index),
        "search_status": found.get("status"),
        "search_valid": bool(search_ok),
        "search_expanded": int(found.get("expanded", 0)),
        "path_len_m": round(float(np.linalg.norm(np.diff(path_m, axis=0), axis=1).sum()), 1),
        "gt_len_m": round(float(np.linalg.norm(np.diff(gt_m, axis=0), axis=1).sum()), 1),
        "ellipse_a_m": [round(float(np.mean(axes_m[:, 0])), 2),
                        round(float(np.max(axes_m[:, 0])), 2)],
        "ellipse_b_m": [round(float(np.mean(axes_m[:, 1])), 2),
                        round(float(np.max(axes_m[:, 1])), 2)],
        "cells_local": int(info.get("local_cells", 0)),
        "cells_box": int(info.get("box_cells", 0)),
        "corridor_valid": bool(all(info.get("corridor_valid") or [False])),
        "pack_rows": int(pack.num_constraint_rows) if pack is not None else 0,
        "violated_samples": int((violation > 1e-9).sum()) if violation is not None else -1,
        "max_violation_m": round(float(violation.max()), 3) if violation is not None else None,
    }
    geometry = {"occ": occ, "gt_m": gt_m, "path_m": path_m, "centers_m": centers_m,
                "axes_m": axes_m, "theta": theta_scene, "table": table,
                "points": points, "violation": violation}
    if show_labels:
        shape4, valid, pdg = load_labels(dataset.split, index,
                                        int(dataset.records[index]["seed"]))
        centres = resample_arclength(
            torch.as_tensor(np.array(pdg, dtype=np.float32), dtype=torch.float32)[None],
            shape4.shape[0])[0].numpy()
        geometry["label"] = {"centres": transform.to_meters(centres),
                            "axes": np.stack([np.exp(shape4[:, 0]) * transform.scale,
                                              np.exp(shape4[:, 1]) * transform.scale], axis=-1),
                            "theta": np.arctan2(shape4[:, 3], shape4[:, 2]) / 2.0,
                            "valid": valid}
    return stats, geometry


def draw(ax_full, ax_zoom, geometry, stats, transform):
    occ, gt_m, path_m = geometry["occ"], geometry["gt_m"], geometry["path_m"]
    extent_m = transform.extent_m
    base = dict(origin="lower", extent=(0, extent_m, 0, extent_m), cmap="Greys",
                vmin=0.0, vmax=1.4, interpolation="nearest")
    for ax in (ax_full, ax_zoom):
        ax.imshow(occ, **base)
        ax.plot(gt_m[:, 0], gt_m[:, 1], color="white", lw=2.0, ls="--", label="GT curve")
        ax.plot(path_m[:, 0], path_m[:, 1], color="deepskyblue", lw=1.6,
                label="guidance search path")
        for c, axes, th in zip(geometry["centers_m"], geometry["axes_m"], geometry["theta"]):
            outline = ellipse_xy(c, axes, th)
            ax.plot(outline[:, 0], outline[:, 1], color="orange", lw=0.5, alpha=0.6)
        table = geometry["table"]
        drawn = 0
        for row in range(table["cell_A"].shape[0]):
            if not bool(table["cell_mask"][row]) or bool(table["cell_is_box"][row]):
                continue
            b_row = table["cell_b"][row]
            keep = np.isfinite(b_row) & (np.abs(table["cell_A"][row]).sum(-1) > 0.0)
            if draw_polygon(ax, table["cell_A"][row][keep], b_row[keep], transform,
                            facecolor="lime", edgecolor="green", lw=1.0, alpha=0.22):
                drawn += 1
        if "label" in geometry:
            lab = geometry["label"]
            for c, axes, th, ok in zip(lab["centres"], lab["axes"], lab["theta"], lab["valid"]):
                if not ok:
                    continue
                outline = ellipse_xy(c, axes, th)
                ax.plot(outline[:, 0], outline[:, 1], color="magenta", lw=0.8,
                        ls=":", alpha=0.9)
        if geometry["points"] is not None:
            pts = transform.to_meters(geometry["points"])
            v = geometry["violation"]
            ax.scatter(pts[:, 0], pts[:, 1], s=20, zorder=6, linewidths=0.4, edgecolors="white",
                       c=np.where(v > 1e-9, "red", "royalblue"),
                       label="constraint samples (M=%d)" % len(pts))
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")
    lo = np.minimum(gt_m.min(axis=0), path_m.min(axis=0)) - 15.0
    hi = np.maximum(gt_m.max(axis=0), path_m.max(axis=0)) + 15.0
    ax_zoom.set_xlim(lo[0], hi[0])
    ax_zoom.set_ylim(lo[1], hi[1])
    ax_full.set_title("%s | cells=%d valid=%s | rows=%d | violations=%d (max %.2f m)"
                      % ("sample %d" % stats["index"], stats["cells_local"],
                         stats["corridor_valid"], stats["pack_rows"],
                         stats["violated_samples"], stats["max_violation_m"] or 0.0),
                      fontsize=9, loc="left")
    ax_zoom.set_title("zoom | ellipse a=%.2f/%.2f m b=%.2f/%.2f m | search %s"
                      % (stats["ellipse_a_m"][0], stats["ellipse_a_m"][1],
                         stats["ellipse_b_m"][0], stats["ellipse_b_m"][1],
                         stats["search_status"]), fontsize=9, loc="left")
    ax_full.legend(fontsize=7, loc="upper right")


def main() -> int:
    parser = base_parser("k4p: GT-guided search -> ellipses -> convex cells -> constraint pack")
    parser.add_argument("--data-root", default=os.path.join(PROJECT_ROOT, "data", "_demo_k4p"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--out", default=os.path.join(PROJECT_ROOT, "outputs", "figures",
                                                     "k4p_geometry_demo.png"))
    parser.add_argument("--show-labels", action="store_true",
                        help="also draw the dataset's own ellipse_shape4_gt_q64 labels")
    opts = parser.parse_args()

    cfg = load(opts)
    cfg["dataset"]["res_m"] = 0.625          # 160 m over 256 cells
    cfg["dataset"]["grid"] = 256
    cfg["dataset"]["real_root"] = opts.data_root
    dataset = RealTrajectoryDataset(cfg, opts.split, limit=opts.offset + opts.limit)
    transform = SceneTransform(float(cfg["dataset"]["res_m"]) * int(cfg["dataset"]["grid"]),
                               int(cfg["dataset"]["grid"]))
    codec = BSplineCodec(degree=int(cfg["trajectory"]["degree"]),
                         num_controls=int(cfg["trajectory"]["num_controls"]),
                         curve_points=int(cfg["trajectory"]["curve_points"]))
    if codec.num_controls != dataset[0]["controls_gt"].shape[0]:
        raise SystemExit("config num_controls=%d but the labels have C=%d"
                         % (codec.num_controls, dataset[0]["controls_gt"].shape[0]))
    net = load_pathsafe(cfg, device="cpu")
    scope = str((cfg.get("corridor") or {}).get("training_scope", "full"))
    print("scope=%r  res_m=%.3f  extent=%.0f m  samples=%d"
          % (scope, cfg["dataset"]["res_m"], transform.extent_m, len(dataset)), flush=True)

    rows = []
    for i in range(opts.offset, opts.offset + opts.limit):
        stats, geometry = run_sample(cfg, dataset, i, net, codec, transform, scope,
                                     opts.show_labels)
        print(json.dumps(stats), flush=True)
        rows.append((stats, geometry))

    fig, axes = plt.subplots(len(rows), 2, figsize=(15.0, 6.4 * len(rows)), squeeze=False)
    for r, (stats, geometry) in enumerate(rows):
        draw(axes[r][0], axes[r][1], geometry, stats, transform)
    fig.suptitle("k4p 160 m/0.625 m: GT-guided Astar-Diffusion search -> frozen "
                 "PathSafe ellipses -> convex cells -> sampled constraint pack"
                 + ("  (magenta dotted = dataset q64 labels)" if opts.show_labels else ""),
                 fontsize=10, y=0.996)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    os.makedirs(os.path.dirname(os.path.abspath(opts.out)), exist_ok=True)
    fig.savefig(opts.out, dpi=130)
    print("figure:", opts.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
