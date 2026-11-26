"""Stage 1: inventory + map-spec filter + SDF consistency for the real dataset.

Implements《GeoPath-Diffuser_数据集制作方案.md》§3:

* stage 1  full inventory of ``data/trajectories`` + ``data/distance_fields``;
* stage 2  keep only 0.25 m/cell, 256x256, 64 m samples — decided from the SDF
  index and the actual arrays, never from the file name;
* stage 3  quality flags (readable, required fields, finite, path length,
  endpoints in range) with an explicit reason for every rejection.

Outputs (``data/geopath_v1/``)::

    manifest.json            counts + map spec + audit version
    eligible_samples.jsonl   one record per eligible trajectory
    rejected_samples.jsonl   one record per rejection with reason
    reports/data_audit.json
    reports/data_audit.md

Usage::

    python scripts/audit_real_dataset.py                # full inventory
    python scripts/audit_real_dataset.py --limit 2000    # quick sampling
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import PROJECT_ROOT, base_parser, load  # noqa: E402

import numpy as np  # noqa: E402

AUDIT_VERSION = "geopath_v1_audit-1"
RESOLUTION_M = 0.25
GRID = 256
EXTENT_M = 64.0
TOL = 1e-6


def _load_index(path: str) -> list:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _percentiles(values, points=(50, 90, 99, 100)) -> dict:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {"p%d" % p: float(np.percentile(arr, p)) for p in points}


def main() -> int:
    parser = base_parser("audit the staged real dataset (stage 1)")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out", default="data/geopath_v1")
    parser.add_argument("--limit", type=int, default=None,
                        help="only inspect the first N candidate trajectories")
    parser.add_argument("--progress-every", type=int, default=2000)
    opts = parser.parse_args()

    data_root = os.path.join(PROJECT_ROOT, opts.data_root)
    out_root = os.path.join(PROJECT_ROOT, opts.out)
    reports = os.path.join(out_root, "reports")
    os.makedirs(reports, exist_ok=True)

    traj_dir = os.path.join(data_root, "trajectories")
    sdf_dir = os.path.join(data_root, "distance_fields")
    index = _load_index(os.path.join(sdf_dir, "index.jsonl"))
    sdf_manifest = json.load(open(os.path.join(sdf_dir, "manifest.json"),
                                  encoding="utf-8"))

    inventory = {
        "audit_version": AUDIT_VERSION,
        "data_root": data_root,
        "trajectory_files_on_disk": len([n for n in os.listdir(traj_dir)
                                         if n.endswith(".npz")]),
        "index_records": len(index),
        "sdf_manifest_version": sdf_manifest.get("sdf_version"),
        "sdf_unique_maps": sdf_manifest.get("n_unique_maps"),
        "resolution_census_all": dict(Counter(round(float(r["resolution_m"]), 4)
                                             for r in index)),
        "map_groups": len({r["map_group"] for r in index}),
    }

    candidates = [r for r in index if abs(float(r["resolution_m"]) - RESOLUTION_M) < TOL]
    if opts.limit is not None:
        candidates = candidates[:int(opts.limit)]
    print("index %d records, %d candidates at %.4f m/cell"
          % (len(index), len(candidates), RESOLUTION_M), flush=True)

    sdf_cache: dict = {}
    eligible, rejected = [], []
    reasons = Counter()
    per_group = Counter()
    per_group_maps = defaultdict(set)
    n_points_all, lengths_all, sdf_checks = [], [], Counter()
    started = time.time()

    for i, record in enumerate(candidates):
        name = record["trajectory"]
        sample_id = os.path.splitext(os.path.basename(name))[0]
        path = os.path.join(traj_dir, name)
        base = {"sample_id": sample_id, "trajectory": name,
                "map_key": record.get("map_key"), "scene_id": record.get("scene_id"),
                "map_group": record.get("map_group"), "path_id": record.get("path_id"),
                "resolution_m": float(record.get("resolution_m")),
                "n_points": int(record.get("n_points", -1))}

        def reject(reason, **extra):
            reasons[reason] += 1
            rejected.append({**base, "reason": reason, **extra})

        if not os.path.exists(path):
            reject("trajectory_file_missing")
            continue
        try:
            with np.load(path) as data:
                fields = set(data.files)
                missing = {"occupancy", "path_xy", "start_xy", "goal_xy"} - fields
                if missing:
                    reject("required_field_missing", missing=sorted(missing))
                    continue
                occupancy = np.asarray(data["occupancy"])
                path_xy = np.asarray(data["path_xy"], dtype=np.float64)
                start_xy = np.asarray(data["start_xy"], dtype=np.float64).reshape(2)
                goal_xy = np.asarray(data["goal_xy"], dtype=np.float64).reshape(2)
                flags = {k: int(np.asarray(data[k]).max()) if k in fields else None
                         for k in ("label_valid", "safety_verified",
                                   "continuity_verified")}
                meta = json.loads(str(data["meta_json"])) if "meta_json" in fields else {}
        except Exception as error:                                  # noqa: BLE001
            reject("npz_unreadable", detail=str(error)[:120])
            continue

        if occupancy.shape != (GRID, GRID):
            reject("map_shape", shape=list(occupancy.shape))
            continue
        if path_xy.ndim != 2 or path_xy.shape[1] != 2 or len(path_xy) < 2:
            reject("path_shape", shape=list(path_xy.shape))
            continue
        if not (np.isfinite(path_xy).all() and np.isfinite(start_xy).all()
                and np.isfinite(goal_xy).all()):
            reject("non_finite")
            continue
        physical = float(meta.get("physical_field_m", -1.0))
        source_res = float(meta.get("source_resolution_m", -1.0))
        if abs(physical - EXTENT_M) > 1e-3:
            reject("meta_extent", physical_field_m=physical)
            continue
        if abs(source_res - RESOLUTION_M) > TOL:
            reject("meta_resolution", source_resolution_m=source_res)
            continue

        length = float(np.linalg.norm(np.diff(path_xy, axis=0), axis=1).sum())
        if length <= 1e-3:
            reject("degenerate_path", length_m=length)
            continue
        if (path_xy.min() < 0.0 or path_xy.max() > EXTENT_M - RESOLUTION_M + 1e-6):
            reject("path_out_of_map", min=float(path_xy.min()),
                   max=float(path_xy.max()))
            continue

        # ---- SDF: match by map_key, verify resolution and occupancy ----------
        map_key = str(record["map_key"])
        if map_key not in sdf_cache:
            sdf_path = os.path.join(sdf_dir, "sdf", "%s.npz" % map_key)
            if not os.path.exists(sdf_path):
                sdf_cache[map_key] = None
            else:
                with np.load(sdf_path) as sdf_data:
                    sdf_cache[map_key] = {
                        "sdf": np.asarray(sdf_data["signed_distance_m"]),
                        "blocked": np.asarray(sdf_data["blocked"]),
                        "valid_mask": np.asarray(sdf_data["valid_mask"]),
                        "resolution_m": float(sdf_data["resolution_m"]),
                    }
        sdf_entry = sdf_cache[map_key]
        if sdf_entry is None:
            reject("sdf_missing", map_key=map_key)
            continue
        if sdf_entry["sdf"].shape != (GRID, GRID):
            reject("sdf_shape", shape=list(sdf_entry["sdf"].shape))
            continue
        if abs(sdf_entry["resolution_m"] - RESOLUTION_M) > TOL:
            reject("sdf_resolution", resolution_m=sdf_entry["resolution_m"])
            continue
        if not np.array_equal(sdf_entry["blocked"].astype(np.uint8),
                              occupancy.astype(np.uint8)):
            reject("occupancy_sdf_mismatch")
            continue
        sdf_checks["matched"] += 1

        per_group[str(record["map_group"])] += 1
        per_group_maps[str(record["map_group"])].add(map_key)
        n_points_all.append(int(len(path_xy)))
        lengths_all.append(length)
        eligible.append({**base, "path_length_m": round(length, 4),
                         "n_points_actual": int(len(path_xy)),
                         "label_valid": flags["label_valid"],
                         "safety_verified": flags["safety_verified"],
                         "continuity_verified": flags["continuity_verified"],
                         "seed": int(meta.get("seed", 0))})

        if opts.progress_every and (i + 1) % opts.progress_every == 0:
            print("  scanned %d/%d  eligible %d  rejected %d  %.0fs"
                  % (i + 1, len(candidates), len(eligible), len(rejected),
                     time.time() - started), flush=True)

    elapsed = time.time() - started
    summary = {
        "audit_version": AUDIT_VERSION,
        "map_spec": {"grid": GRID, "resolution_m": RESOLUTION_M,
                     "physical_extent_m": EXTENT_M},
        "inventory": inventory,
        "candidates": len(candidates),
        "eligible": len(eligible),
        "rejected": len(rejected),
        "rejection_reasons": dict(reasons),
        "sdf_consistency": dict(sdf_checks),
        "map_groups_eligible": len(per_group),
        "eligible_per_group": dict(sorted(per_group.items())),
        "maps_per_group": {k: len(v) for k, v in sorted(per_group_maps.items())},
        "path_points": _percentiles(n_points_all),
        "path_length_m": _percentiles(lengths_all),
        "elapsed_s": round(elapsed, 1),
    }

    def write_jsonl(path, rows):
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_jsonl(os.path.join(out_root, "eligible_samples.jsonl"), eligible)
    write_jsonl(os.path.join(out_root, "rejected_samples.jsonl"), rejected)
    with open(os.path.join(out_root, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    with open(os.path.join(reports, "data_audit.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    lines = [
        "# 真实数据集盘点与筛选（stage 1）", "",
        "| 项 | 值 |", "|---|---|",
        "| 轨迹文件（磁盘） | %d |" % inventory["trajectory_files_on_disk"],
        "| SDF 索引记录 | %d |" % inventory["index_records"],
        "| 唯一地图（SDF） | %s |" % inventory["sdf_unique_maps"],
        "| 分辨率分布（全部） | %s |" % inventory["resolution_census_all"],
        "| 0.25 m 候选 | %d |" % len(candidates),
        "| **合格样本** | **%d** |" % len(eligible),
        "| 排除样本 | %d |" % len(rejected),
        "| 合格 map_group | %d |" % len(per_group),
        "| SDF 与 occupancy 一致 | %d |" % sdf_checks["matched"],
        "", "## 排除原因", "", "| 原因 | 数量 |", "|---|---|",
    ]
    for reason, count in reasons.most_common():
        lines.append("| %s | %d |" % (reason, count))
    lines += ["", "## 分布", "",
              "* 路径点数：%s" % summary["path_points"],
              "* 路径长度 (m)：%s" % summary["path_length_m"],
              "* 耗时：%.1f s" % elapsed]
    with open(os.path.join(reports, "data_audit.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(json.dumps({k: summary[k] for k in
                      ("candidates", "eligible", "rejected", "rejection_reasons",
                       "map_groups_eligible", "elapsed_s")}, indent=2,
                     ensure_ascii=False))
    print("written:", out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
