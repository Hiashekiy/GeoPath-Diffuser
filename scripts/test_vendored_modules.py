"""Test + visualise the two vendored reference modules on the ORIGINAL data.

Everything here runs the *upstream* code that we now vendor:

* guidance  -> ``vendor/field_guided_search`` (their ``GuidanceField``)
* ellipses  -> ``vendor/ellipsenet`` (their ``PSENet``, checkpoint ``v3_best.pt``)
* corridor  -> the same ``build_region_table`` the safety cache uses, fed with the
  vendored ellipses

and compares against the labels shipped with the data (``ellipse_ab`` /
``ellipse_angle`` / ``center_offset`` / ``path_xy``), which are exactly what the
ellipse network was trained on.

Usage::

    python scripts/test_vendored_modules.py --samples 3
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
from matplotlib.patches import Ellipse  # noqa: E402

from gp_diffuser.factory import build_pipeline, build_transform  # noqa: E402
from gp_diffuser.frames import SceneTransform  # noqa: E402
from gp_diffuser.guidance.reference_search import ReferenceGuidanceSearch  # noqa: E402
from gp_diffuser.safety.local_regions import build_region_table, pack_from_tables  # noqa: E402
from gp_diffuser.safety.psenet_inputs import ray_directions  # noqa: E402
from gp_diffuser.safety.safety_features import ellipse_shape4_to_scene  # noqa: E402

# the corridor rasteriser from the prediction visualiser is reused unchanged
from visualize_prediction import _corridor_mask, _px  # noqa: E402


def load_raw(entry, data_root="data"):
    traj = np.load(os.path.join(data_root, "trajectories", entry["trajectory"]))
    sdf = np.load(os.path.join(data_root, "distance_fields", "sdf",
                               entry["map_key"] + ".npz"))["signed_distance_m"]
    return traj, np.asarray(sdf, dtype=np.float32)


def main() -> int:
    parser = base_parser("test the vendored guidance / ellipse / corridor modules")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--outdir", default="outputs/figures/vendored_modules")
    opts = parser.parse_args()

    cfg = load(opts)
    pipe = build_pipeline(cfg, device=opts.device)
    transform: SceneTransform = pipe["transform"]
    codec = pipe["codec"]
    net = pipe["net"]
    search = ReferenceGuidanceSearch(cfg.get("guidance") or {}, res_m=0.25)
    corridor_cfg = dict(cfg.get("corridor") or {})
    grid = int((cfg.get("dataset") or {}).get("grid", 256))
    res_m = float((cfg.get("dataset") or {}).get("res_m", 0.25))

    index = [json.loads(l) for l in open(os.path.join(PROJECT_ROOT, "data",
                                                      "distance_fields", "index.jsonl"),
                                         encoding="utf-8")]
    entries = [e for e in index if abs(float(e["resolution_m"]) - 0.25) < 1e-9]
    # prefer trajectories with a decent number of stations and a verified label
    entries = [e for e in entries if 12 <= int(e["n_points"]) <= 40][:opts.samples]
    outdir = os.path.join(PROJECT_ROOT, opts.outdir)
    os.makedirs(outdir, exist_ok=True)

    summary = []
    for k, entry in enumerate(entries):
        traj, sdf = load_raw(entry)
        occ = np.asarray(traj["occupancy"]).astype(np.float32)
        path_m = np.asarray(traj["path_xy"], dtype=np.float64)
        start_m = np.asarray(traj["start_xy"], dtype=np.float64).reshape(2)
        goal_m = np.asarray(traj["goal_xy"], dtype=np.float64).reshape(2)
        ab_gt = np.asarray(traj["ellipse_ab"], dtype=np.float64)
        angle_gt = np.asarray(traj["ellipse_angle"], dtype=np.float64)
        off_gt = np.asarray(traj["center_offset"], dtype=np.float64)
        centre_gt = path_m + off_gt

        # ---------------- 1. vendored guidance search ---------------------
        res = search.search(occ, path_m, start_m, goal_m, return_debug=True)
        path_search = res.get("path_xy")
        if path_search is None:
            print("[%s] search failed (%s)" % (entry["trajectory"], res.get("status")))
            continue
        field = res.get("field")
        sources = res.get("source_positions_rc")

        # ---------------- 2. vendored ellipse network ---------------------
        occ_t = torch.as_tensor(occ, device=opts.device)[None, None]
        sdf_t = torch.as_tensor(sdf, device=opts.device)[None, None]
        path_t = torch.as_tensor(path_search, dtype=torch.float32,
                                 device=opts.device)[None]
        safety = pipe["safety_generator"](occ_t, sdf_t, path_t)
        centres_m = safety["ellipse_center"][0].detach().cpu().numpy()
        axes_m = np.stack([safety["ellipse"][0, :, 2].detach().cpu().numpy(),
                           safety["ellipse"][0, :, 3].detach().cpu().numpy()], axis=-1)
        theta = np.arctan2(safety["ellipse"][0, :, 5].detach().cpu().numpy(),
                           safety["ellipse"][0, :, 4].detach().cpu().numpy()) * 0.5

        # the same net, but driven by THEIR stored LiDAR at THEIR stations: this is
        # the only apples-to-apples comparison against the shipped labels
        lidar_np = np.asarray(traj["lidar_ranges"], dtype=np.float32)
        n_gt = lidar_np.shape[0]
        batch = {
            "occupancy": occ_t, "sdf": sdf_t,
            "path_xy": torch.as_tensor(path_m, dtype=torch.float32,
                                      device=opts.device)[None],
            "lidar": torch.as_tensor(lidar_np, device=opts.device)[None],
            "ray_dir": torch.as_tensor(ray_directions(16).astype(np.float32),
                                    device=opts.device)[None],
            "ray_valid": torch.as_tensor(lidar_np < 9.0 - 1e-3, device=opts.device)[None],
            "mask": torch.ones(1, n_gt, dtype=torch.bool, device=opts.device),
            "r_scene": torch.full((1,), float(res_m), device=opts.device),
        }
        with torch.no_grad():
            dec = net.predict_ellipses(batch)
        ab_net = np.stack([dec["a"][0].float().cpu().numpy(),
                           dec["b"][0].float().cpu().numpy()], axis=-1)
        angle_net = dec["theta"][0].float().cpu().numpy()

        # ---------------- 3. convex corridor from those ellipses ----------
        centres_scene = transform.to_scene(centres_m)
        ellipse_scene = np.concatenate([centres_scene,
                                        transform.lengths_to_scene(axes_m),
                                        np.stack([safety["ellipse"][0, :, 4].detach().cpu().numpy(),
                                                  safety["ellipse"][0, :, 5].detach().cpu().numpy()],
                                                 axis=-1)], axis=-1)
        shape4_scene = ellipse_shape4_to_scene(safety["ellipse_shape4"], transform.scale)
        anchors = torch.linspace(0.0, 1.0, int(centres_scene.shape[0]),
                                 dtype=torch.float64, device=opts.device)
        table, info = build_region_table(
            occ_t, torch.as_tensor(centres_scene, dtype=torch.float32, device=opts.device),
            shape4_scene[0], anchors,
            curve_scene=torch.as_tensor(transform.to_scene(path_m), dtype=torch.float32,
                                       device=opts.device)[None],
            gamma_scene=torch.as_tensor(transform.to_scene(path_search),
                                        dtype=torch.float32, device=opts.device),
            config=corridor_cfg, scope="full", scene_scale=transform.scale)
        pack = pack_from_tables(codec, [table], include_box=False, device="cpu")
        mask = _corridor_mask(pack, grid)

        # ---------------- 4. figure ---------------------------------------
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
        gt_px = _px(transform.to_scene(path_m), grid)
        se_px = _px(transform.to_scene(path_search), grid)
        sg_px = _px(transform.to_scene(np.stack([start_m, goal_m])), grid)

        ax = axes[0]
        ax.imshow(occ, cmap="Greys", origin="lower", vmin=0, vmax=1)
        if field is not None:
            finite = np.isfinite(field)
            im = ax.imshow(np.where(finite, field, np.nan), cmap="viridis", origin="lower",
                           alpha=0.55)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="guidance F(r,c)")
            if sources is not None and len(sources):
                src = np.asarray(sources, dtype=np.float64)
                ax.scatter(src[:, 1], src[:, 0], s=12, c="white", edgecolors="k",
                           linewidths=0.3, label="sources")
        ax.plot(gt_px[:, 0], gt_px[:, 1], color="#1f77b4", lw=2.2, label="GT path")
        ax.plot(se_px[:, 0], se_px[:, 1], color="#ff7f0e", lw=2.2, ls="--",
                label="vendored search")
        ax.scatter(sg_px[:, 0], sg_px[:, 1], s=70, c=["#2ca02c", "#d62728"],
                   edgecolors="k", zorder=6)
        ax.set_title("guidance field + search\nstatus %s  |GT| %.1f m  |search| %.1f m"
                     % (res.get("status"), _length(path_m), _length(path_search)), fontsize=8)
        ax.legend(fontsize=6, loc="lower left")
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[1]
        ax.imshow(occ, cmap="Greys", origin="lower", vmin=0, vmax=1)
        ax.plot(gt_px[:, 0], gt_px[:, 1], color="#1f77b4", lw=1.6, label="GT path")
        # GT labels live at their own stations (path point + centre_offset); the net
        # ran on exactly those stations, so both are drawn at the same centres
        net_centres_m = dec["center"][0].float().cpu().numpy()
        _ellipses(ax, centre_gt, ab_gt, angle_gt, grid, res_m, "#ff7f0e", 0.95,
                  "GT labels (%d)" % len(ab_gt))
        _ellipses(ax, net_centres_m, ab_net, angle_net, grid, res_m, "#2ca02c", 0.95,
                  "vendored net, same stations (%d)" % len(ab_net))
        ax.set_title("ellipses: GT labels vs vendored net\nmedian a %.2f/%.2f m  b %.2f/%.2f m"
                     % (np.median(ab_gt[:, 0]), np.median(ab_net[:, 0]),
                        np.median(ab_gt[:, 1]), np.median(ab_net[:, 1])), fontsize=8)
        ax.legend(fontsize=6, loc="lower left")
        ax.set_xticks([]); ax.set_yticks([])

        ax = axes[2]
        ax.imshow(occ, cmap="Greys", origin="lower", vmin=0, vmax=1)
        if mask.any():
            ax.imshow(np.where(mask, 1.0, np.nan), cmap="Greens", origin="lower", alpha=0.45)
        _ellipses(ax, centres_m, axes_m, theta, grid, res_m, "#9467bd", 0.7,
                  "pipeline ellipses (%d)" % len(axes_m))
        ax.plot(se_px[:, 0], se_px[:, 1], color="#ff7f0e", lw=1.6, ls="--")
        ax.set_title("convex corridor from those ellipses\ncells %d  valid %s  bridges %d"
                     % (int(info.get("local_cells", 0)), info.get("corridor_valid"),
                        int(info.get("bridges", 0))), fontsize=8)
        ax.legend(fontsize=6, loc="lower left")
        ax.set_xticks([]); ax.set_yticks([])

        fig.suptitle("%s  (res %.2f m, %d stations)" % (entry["trajectory"], res_m, n_gt),
                     fontsize=10)
        fig.tight_layout()
        png = os.path.join(outdir, "sample%02d.png" % k)
        fig.savefig(png, dpi=110)
        plt.close(fig)

        row = {"trajectory": entry["trajectory"], "status": res.get("status"),
               "gt_length_m": round(_length(path_m), 2),
               "search_length_m": round(_length(path_search), 2),
               "search_sources": int(res.get("sources", 0) or 0),
               "gt_a_median_m": float(np.median(ab_gt[:, 0])),
               "gt_b_median_m": float(np.median(ab_gt[:, 1])),
               "net_a_median_m": float(np.median(ab_net[:, 0])),
               "net_b_median_m": float(np.median(ab_net[:, 1])),
               "pipeline_a_median_m": float(np.median(axes_m[:, 0])),
               "pipeline_b_median_m": float(np.median(axes_m[:, 1])),
               "corridor_cells": int(info.get("local_cells", 0)),
               "corridor_valid": bool(info.get("corridor_valid")),
               "bridges": int(info.get("bridges", 0)),
               "png": os.path.relpath(png, PROJECT_ROOT)}
        summary.append(row)
        print(json.dumps(row, ensure_ascii=False))

    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print("figures:", outdir)
    return 0


def _length(path) -> float:
    p = np.asarray(path, dtype=np.float64)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


def _ellipses(ax, centres_m, ab_m, angle, grid, res_m, colour, alpha, label):
    """Draw ellipses given centres/axes in metres onto the pixel frame."""
    scale = 1.0 / res_m
    for (cx, cy), (a, b), th in zip(np.asarray(centres_m), np.asarray(ab_m), np.asarray(angle)):
        if not np.all(np.isfinite([cx, cy, a, b, th])) or a <= 0 or b <= 0:
            continue
        ax.add_patch(Ellipse((cx * scale, cy * scale), width=2 * a * scale,
                             height=2 * b * scale, angle=np.degrees(th), fill=False,
                             edgecolor=colour, alpha=alpha, lw=1.0))
    ax.plot([], [], color=colour, lw=1.0, label=label)


if __name__ == "__main__":
    raise SystemExit(main())
