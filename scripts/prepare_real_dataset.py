"""Stage 1b: split by map_group + GT conversion + B-spline quality audit.

Implements《GeoPath-Diffuser_数据集制作方案.md》§4–§7:

* split strictly by ``map_group`` (80/10/10, seed 0) — never per trajectory;
* 128-point arc-length resampling in metres, ``SceneTransform`` to scene coords;
* 32 cubic B-spline controls fitted with the existing ``BSplineCodec``;
* per-sample quality metrics (RMSE, max error, endpoint/heading error, dense
  collision, min clearance, length, curvature) and a data-driven error cut;
* processed NPZ per sample plus ``train/val/test.jsonl`` indexes.

Usage::

    # pilot (stage 1 acceptance)
    python scripts/prepare_real_dataset.py --pilot --limit-train 1000 --limit-val 200
    # full
    python scripts/prepare_real_dataset.py
    # one split only / resumable
    python scripts/prepare_real_dataset.py --split val
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import PROJECT_ROOT, base_parser, load  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gp_diffuser.evaluation.metrics import trajectory_metrics  # noqa: E402
from gp_diffuser.factory import build_codec, build_transform  # noqa: E402
from gp_diffuser.guidance.search import resample_polyline  # noqa: E402

DATASET_VERSION = "geopath_v1"


# --------------------------------------------------------------------------- #
# split
# --------------------------------------------------------------------------- #
def split_by_group(records, frac=(0.8, 0.1, 0.1), seed: int = 0):
    """Deterministic map_group split (same semantics as PathSafe ``split_by_group``).

    Groups (not trajectories) are shuffled with ``seed`` and cut at the fraction
    boundaries, so every trajectory of a group lands in exactly one split.
    """
    groups = sorted({str(r["map_group"]) for r in records})
    rng = np.random.default_rng(int(seed))
    order = np.asarray(groups, dtype=object)[rng.permutation(len(groups))]
    n_train = int(round(len(groups) * float(frac[0])))
    n_val = int(round(len(groups) * float(frac[1])))
    n_train = max(0, min(n_train, len(groups)))
    n_val = max(0, min(n_val, len(groups) - n_train))
    assign = {}
    for i, group in enumerate(order):
        assign[str(group)] = "train" if i < n_train else (
            "val" if i < n_train + n_val else "test")
    return assign


def stratified_pilot(records, assign, per_split: dict, seed: int = 0):
    """Group-stratified subset of a split (all groups of that split represented)."""
    rng = np.random.default_rng(int(seed))
    out = []
    for split, limit in per_split.items():
        rows = [r for r in records if assign[str(r["map_group"])] == split]
        if limit is None or limit >= len(rows):
            out.extend(rows)
            continue
        by_group = defaultdict(list)
        for row in rows:
            by_group[str(row["map_group"])].append(row)
        groups = sorted(by_group)
        # proportional allocation with at least one sample per group
        quota = {g: max(1, int(round(limit * len(by_group[g]) / len(rows))))
                 for g in groups}
        chosen = []
        for group in groups:
            pool = by_group[group]
            idx = rng.permutation(len(pool))[:min(quota[group], len(pool))]
            chosen.extend(pool[i] for i in idx)
        if len(chosen) > limit:
            idx = rng.permutation(len(chosen))[:limit]
            chosen = [chosen[i] for i in idx]
        out.extend(chosen)
    return out


# --------------------------------------------------------------------------- #
# GT conversion
# --------------------------------------------------------------------------- #
def heading_at(path: np.ndarray, at_start: bool, min_step_m: float = 0.05) -> np.ndarray:
    """Unit tangent from the original path ends (walk inward until non-degenerate)."""
    pts = np.asarray(path, dtype=np.float64)
    if len(pts) < 2:
        return np.array([1.0, 0.0])
    order = range(len(pts) - 1) if at_start else range(len(pts) - 1, 0, -1)
    for k in order:
        a, b = (pts[0], pts[k + 1]) if at_start else (pts[k - 1], pts[-1])
        delta = b - a
        norm = float(np.linalg.norm(delta))
        if norm >= min_step_m:
            return delta / norm
    delta = (pts[1] - pts[0]) if at_start else (pts[-1] - pts[-2])
    norm = float(np.linalg.norm(delta))
    return delta / norm if norm > 1e-9 else np.array([1.0, 0.0])


def convert_sample(record, data_root: str, transform, codec, curve_points: int,
                   control_points: int) -> dict:
    """GT conversion + quality metrics for one eligible trajectory."""
    path = os.path.join(data_root, "trajectories", record["trajectory"])
    with np.load(path) as data:
        occupancy = np.asarray(data["occupancy"]).astype(np.float32)
        path_xy = np.asarray(data["path_xy"], dtype=np.float64)
        start_xy = np.asarray(data["start_xy"], dtype=np.float64).reshape(2)
        goal_xy = np.asarray(data["goal_xy"], dtype=np.float64).reshape(2)
        meta = json.loads(str(data["meta_json"]))
    sdf_path = os.path.join(data_root, "distance_fields", "sdf",
                            "%s.npz" % record["map_key"])
    with np.load(sdf_path) as sdf_data:
        sdf = np.asarray(sdf_data["signed_distance_m"]).astype(np.float32)
        valid_mask = np.asarray(sdf_data["valid_mask"]).astype(np.uint8)

    path_m = resample_polyline(path_xy, curve_points)            # [128,2] metres
    path_scene = transform.to_scene(path_m)
    start_goal_m = np.stack([path_m[0], path_m[-1]])
    start_goal_scene = transform.to_scene(start_goal_m)
    heading = np.stack([heading_at(path_xy, True), heading_at(path_xy, False)])
    original_length = float(np.linalg.norm(np.diff(path_xy, axis=0), axis=1).sum())

    curve = torch.as_tensor(path_scene, dtype=torch.float32)[None]
    cond = torch.as_tensor(start_goal_scene, dtype=torch.float32)
    controls = codec.fit_curve_to_controls(curve, cond[0:1], cond[1:2])[0]
    decoded = codec.decode_controls(controls[None])[0]
    decoded_m = transform.to_meters(decoded.numpy())

    # NOTE: ``decoded`` and ``curve`` live in scene units; report metres
    scene_to_m = float(transform.scale)
    fit_error_scene = torch.linalg.norm(decoded - curve[0], dim=-1)
    fit_error = fit_error_scene * scene_to_m
    metrics = trajectory_metrics(decoded_m, occupancy, sdf, res_m=float(transform.extent_m
                                                                       / transform.grid))
    # heading error from the exact derivative basis
    basis = codec.basis_derivative_at([0.0, 1.0]).to(dtype=controls.dtype)
    tangents = torch.einsum("pk,bkd->bpd", basis, controls[None])[0]
    tangents = tangents / tangents.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    target = torch.as_tensor(heading, dtype=torch.float32)
    target = target / target.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    cos = (tangents * target).sum(dim=-1).clamp(-1.0, 1.0)
    heading_err = torch.rad2deg(torch.arccos(cos)).numpy()
    endpoint_scene = float(max(float((decoded[0] - curve[0, 0]).norm()),
                              float((decoded[-1] - curve[0, -1]).norm())))
    endpoint_m = endpoint_scene * scene_to_m

    return {
        "occupancy": occupancy,
        "sdf": sdf,
        "valid_mask": valid_mask,
        "path_xy_m": path_m.astype(np.float32),
        "path_xy_scene": path_scene.astype(np.float32),
        "controls_scene": controls.numpy().astype(np.float32),
        "start_goal_m": start_goal_m.astype(np.float32),
        "start_goal_scene": start_goal_scene.astype(np.float32),
        "heading": heading.astype(np.float32),
        "heading_valid": np.ones(2, dtype=bool),
        "quality": {
            "fit_rmse_m": float(fit_error.pow(2).mean().sqrt()),
            "fit_max_error_m": float(fit_error.max()),
            "endpoint_error_scene": endpoint_scene,
            "endpoint_error_m": endpoint_m,
            "heading_error_start_deg": float(heading_err[0]),
            "heading_error_goal_deg": float(heading_err[1]),
            "collision_rate": float(metrics["collision_rate"]),
            "min_clearance_m": float(metrics["min_clearance_m"]),
            "decoded_length_m": float(metrics["length_m"]),
            "original_length_m": original_length,
            "mean_curvature": float(metrics["mean_curvature"]),
            "max_curvature": float(metrics["max_curvature"]),
            "n_points_original": int(len(path_xy)),
        },
        "meta": {
            "dataset_version": DATASET_VERSION,
            "sample_id": record["sample_id"],
            "scene_id": record.get("scene_id"),
            "map_group": record.get("map_group"),
            "map_key": record.get("map_key"),
            "path_id": record.get("path_id"),
            "resolution_m": float(record.get("resolution_m", 0.0)),
            "physical_extent_m": float(transform.extent_m),
            "curve_points": int(curve_points),
            "control_points": int(control_points),
            "source_trajectory": record["trajectory"],
            "source_flags": {k: record.get(k) for k in
                             ("label_valid", "safety_verified", "continuity_verified")},
            "seed": int(record.get("seed", 0)),
            "source_meta_seed": meta.get("seed"),
        },
    }


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = base_parser("split + GT conversion + B-spline audit for the real dataset")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", default="data/geopath_v1")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--limit-train", type=int, default=1000)
    parser.add_argument("--limit-val", type=int, default=200)
    parser.add_argument("--split", default=None,
                        choices=["train", "val", "test", "pilot"])
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--split-frac", default="0.8,0.1,0.1")
    parser.add_argument("--fit-cut-m", type=float, default=None,
                        help="override the data-driven max-error cut")
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    opts = parser.parse_args()

    cfg = load(opts)
    frac = tuple(float(v) for v in opts.split_frac.split(","))
    out_root = os.path.join(PROJECT_ROOT, opts.out)
    data_root = os.path.join(PROJECT_ROOT, opts.data_root)
    reports = os.path.join(out_root, "reports")
    os.makedirs(reports, exist_ok=True)

    eligible_path = os.path.join(out_root, "eligible_samples.jsonl")
    if not os.path.exists(eligible_path):
        raise SystemExit("run scripts/audit_real_dataset.py first (%s)" % eligible_path)
    with open(eligible_path, "r", encoding="utf-8") as fh:
        eligible = [json.loads(line) for line in fh if line.strip()]
    print("eligible samples: %d" % len(eligible), flush=True)

    assign = split_by_group(eligible, frac, opts.split_seed)
    groups_by_split = defaultdict(list)
    for group, split in sorted(assign.items()):
        groups_by_split[split].append(group)
    counts = Counter(assign[str(r["map_group"])] for r in eligible)
    split_manifest = {
        "dataset_version": DATASET_VERSION,
        "split_seed": int(opts.split_seed),
        "split_frac": list(frac),
        "group_counts": {k: len(v) for k, v in groups_by_split.items()},
        "sample_counts": dict(counts),
        "groups": {k: v for k, v in groups_by_split.items()},
    }
    with open(os.path.join(out_root, "split_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(split_manifest, fh, indent=2, ensure_ascii=False)
    print("split by map_group: %s" % dict(counts), flush=True)

    transform = build_transform(cfg)
    codec = build_codec(cfg)
    curve_points = int(codec.curve_points)
    control_points = int(codec.num_controls)

    if opts.pilot:
        per_split = {"train": opts.limit_train, "val": opts.limit_val}
        selected = stratified_pilot(eligible, assign, per_split, seed=opts.split_seed)
        dataset_tag = "pilot"
    else:
        selected = eligible
        dataset_tag = "full"

    pending = selected
    if opts.split and opts.split != "pilot":
        pending = [r for r in selected if assign[str(r["map_group"])] == opts.split]
    print("converting %d samples (%s)" % (len(pending), dataset_tag), flush=True)

    os.makedirs(os.path.join(out_root, "reports"), exist_ok=True)
    started = time.time()
    results = []
    rejections = []
    for i, record in enumerate(pending):
        split = assign[str(record["map_group"])]
        processed_dir = os.path.join(out_root, "processed", split)
        os.makedirs(processed_dir, exist_ok=True)
        npz_path = os.path.join(processed_dir, "%s.npz" % record["sample_id"])
        try:
            converted = convert_sample(record, data_root, transform, codec,
                                       curve_points, control_points)
        except Exception as error:                                  # noqa: BLE001
            rejections.append({**record, "reason": "conversion_failed",
                               "detail": str(error)[:160]})
            continue
        quality = converted["quality"]
        if quality["collision_rate"] > 0.0:
            rejections.append({**record, "reason": "fit_collides",
                               "collision_rate": quality["collision_rate"],
                               "fit_max_error_m": quality["fit_max_error_m"]})
            continue
        payload = {
            "occupancy": converted["occupancy"].astype(np.uint8),
            "sdf": converted["sdf"],
            "valid_mask": converted["valid_mask"],
            "path_xy_m": converted["path_xy_m"],
            "path_xy_scene": converted["path_xy_scene"],
            "controls_scene": converted["controls_scene"],
            "start_goal_m": converted["start_goal_m"],
            "start_goal_scene": converted["start_goal_scene"],
            "heading": converted["heading"],
            "heading_valid": converted["heading_valid"],
            "meta_json": np.asarray(json.dumps({**converted["meta"], "quality": quality},
                                               ensure_ascii=False)),
        }
        if opts.overwrite or not os.path.exists(npz_path):
            np.savez_compressed(npz_path, **payload)
        results.append({"split": split, "npz": npz_path, "record": record,
                        "quality": quality})
        if opts.progress_every and (i + 1) % opts.progress_every == 0:
            print("  converted %d/%d  %.0fs" % (i + 1, len(pending),
                                                time.time() - started), flush=True)

    # ---- data-driven fit-error cut -------------------------------------
    errors = np.array([r["quality"]["fit_max_error_m"] for r in results])
    cut = opts.fit_cut_m
    if cut is None:
        cut = float(max(0.5, np.percentile(errors, 99))) if len(errors) else 0.5
    kept, over_cut = [], []
    for row in results:
        if row["quality"]["fit_max_error_m"] > cut:
            over_cut.append(row)
            os.remove(row["npz"])
        else:
            kept.append(row)

    # ---- indexes -------------------------------------------------------
    def record_of(row):
        rec = row["record"]
        return {**{k: rec.get(k) for k in ("sample_id", "scene_id", "map_group",
                                           "map_key", "path_id", "resolution_m")},
                "split": row["split"],
                "processed": os.path.relpath(row["npz"], PROJECT_ROOT).replace("\\", "/"),
                "path_length_m": rec.get("path_length_m"),
                "n_points": rec.get("n_points"),
                "seed": rec.get("seed", 0),
                "quality": row["quality"]}

    by_split = defaultdict(list)
    for row in kept:
        by_split[row["split"]].append(record_of(row))
    for split, rows in by_split.items():
        with open(os.path.join(out_root, "%s.jsonl" % split), "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(os.path.join(reports, "conversion_rejections.jsonl"), "w",
              encoding="utf-8") as fh:
        for row in rejections + [{**r["record"], "reason": "fit_max_error_over_cut",
                                  "fit_max_error_m": r["quality"]["fit_max_error_m"],
                                  "cut_m": cut} for r in over_cut]:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    fit_report = {
        "dataset_version": DATASET_VERSION,
        "dataset_tag": dataset_tag,
        "curve_points": curve_points,
        "control_points": control_points,
        "samples_converted": len(results),
        "kept": len(kept),
        "rejected_collision": len(rejections),
        "rejected_over_cut": len(over_cut),
        "fit_cut_m": cut,
        "distributions": {},
    }
    if kept:
        q = {k: np.array([r["quality"][k] for r in kept], dtype=np.float64)
             for k in kept[0]["quality"]}
        for key, values in q.items():
            fit_report["distributions"][key] = {
                "mean": float(values.mean()),
                "p50": float(np.percentile(values, 50)),
                "p90": float(np.percentile(values, 90)),
                "p99": float(np.percentile(values, 99)),
                "max": float(values.max()),
            }
    by_split_counts = Counter(r["split"] for r in kept)
    fit_report["per_split"] = dict(by_split_counts)
    fit_report["per_split_groups"] = {s: len({r["record"]["map_group"]
                                              for r in kept if r["split"] == s})
                                      for s in by_split_counts}
    with open(os.path.join(reports, "bspline_fit.json"), "w", encoding="utf-8") as fh:
        json.dump(fit_report, fh, indent=2, ensure_ascii=False)

    collision_report = {
        "samples_checked": len(results),
        "sample_collision_rate": float(np.mean([1.0 if r["quality"]["collision_rate"] > 0 else 0.0
                                                for r in results])) if results else 0.0,
        "mean_collision_rate": float(np.mean([r["quality"]["collision_rate"]
                                              for r in results])) if results else 0.0,
        "min_clearance_m": {
            "min": float(min(r["quality"]["min_clearance_m"] for r in results)) if results else None,
            "p1": float(np.percentile([r["quality"]["min_clearance_m"] for r in results], 1)) if results else None,
        },
        "endpoint_error_scene_max": float(max((r["quality"]["endpoint_error_scene"]
                                               for r in results), default=0.0)),
        "heading_error_deg_max": float(max((max(r["quality"]["heading_error_start_deg"],
                                                r["quality"]["heading_error_goal_deg"])
                                            for r in results), default=0.0)),
        "excluded_collision": len(rejections),
        "excluded_over_cut": len(over_cut),
        "fit_cut_m": cut,
    }
    with open(os.path.join(reports, "collision_check.json"), "w", encoding="utf-8") as fh:
        json.dump(collision_report, fh, indent=2, ensure_ascii=False)

    lines = ["# GT 转换与 B-spline 质量审计（%s）" % dataset_tag, "",
             "* 样本版本：`%s`，curve_points=%d，control_points=%d"
             % (DATASET_VERSION, curve_points, control_points),
             "* 转换成功：%d，保留：%d，碰撞排除：%d，超过误差阈值排除：%d"
             % (len(results), len(kept), len(rejections), len(over_cut)),
             "* 误差阈值（p99 与 0.5 m 取大）：%.4f m" % cut,
             "* 各 split 保留样本：%s" % dict(by_split_counts), ""]
    if kept:
        lines += ["| 指标 | mean | p50 | p90 | p99 | max |", "|---|---|---|---|---|---|"]
        for key, stats in fit_report["distributions"].items():
            lines.append("| %s | %.4f | %.4f | %.4f | %.4f | %.4f |"
                         % (key, stats["mean"], stats["p50"], stats["p90"],
                            stats["p99"], stats["max"]))
    with open(os.path.join(reports, "dataset_summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(json.dumps({"converted": len(results), "kept": len(kept),
                      "rejected_collision": len(rejections),
                      "rejected_over_cut": len(over_cut), "fit_cut_m": cut,
                      "per_split": dict(by_split_counts),
                      "elapsed_s": round(time.time() - started, 1)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
