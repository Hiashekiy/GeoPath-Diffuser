"""Convert the array-based root ``data/160m-k4p-c32q64`` into the processed
``geopath_v1`` layout that ``RealTrajectoryDataset`` reads.

Source (per split, column-oriented .npy)::

    occupancy.npy               [N,256,256] uint8    1 = blocked, already ERODED
                                                      by 4 cells (k4p step 05)
    conditions.npy              [N,2,2]  f32         start/goal, scene [-1,1]
    curve_gt.npy                [N,128,2] f32        demo curve, scene
    control_gt.npy              [N,32,2] f32         cubic B-spline controls, scene
    heading.npy / heading_valid [N,2,2] / [N,2]
    episode_id / sample_id      [N]

Target (per sample .npz, exactly the keys ``RealTrajectoryDataset._load`` reads)::

    occupancy  sdf  start_goal_scene  heading  heading_valid
    controls_scene  path_xy_scene  path_xy_m  start_goal_m  meta_json

Conversion rules (see data/160m-k4p-c32q64/NEW_LABELS_REPORT.md):

* frames -- the source uses the same half-pixel convention as ``SceneTransform``
  (``world_to_grid``), so the map is ``res_m = 160/256 = 0.625`` and
  ``metres = (scene - (-1 + 1/256)) * 80``.  Nothing is resampled.
* occupancy -- the stored bitmap is the *eroded* map and carries no robot radius,
  while this project's contract is "the single blocked map, inflated ONCE".
  It is inflated here by an exact Euclidean radius (default 1.1 m = the 0.8 m
  robot + 0.3 m margin used by the upstream label pass).
* sdf -- recomputed from the inflated bitmap (the source has no sdf).
* trajectory -- ``control_gt`` / ``curve_gt`` are already scene coordinates and
  already satisfy degree 3 / C=32 / 128 points, so they are copied; ``--num-controls``
  refits them with ``BSplineCodec`` only when a different C is requested.

Usage::

    python scripts/prepare_k4p_dataset.py --limit 8 --splits train   # smoke
    python scripts/prepare_k4p_dataset.py                            # full
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import PROJECT_ROOT  # noqa: E402

import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
from gp_diffuser.frames import SceneTransform  # noqa: E402
from gp_diffuser.training.dataset import sdf_from_occupancy  # noqa: E402

SOURCE_ROOT = os.path.join(PROJECT_ROOT, "data", "160m-k4p-c32q64")
SOURCE_RES_M = 0.625
SOURCE_GRID = 256
OUT_ROOT = os.path.join(PROJECT_ROOT, "data", "geopath_v1_k4p")
SPLITS = ("train", "val", "test")


def transform() -> SceneTransform:
    return SceneTransform(SOURCE_RES_M * SOURCE_GRID, SOURCE_GRID)


def inflate(occupancy: np.ndarray, radius_m: float, res_m: float) -> np.ndarray:
    """Blocked cells within ``radius_m`` of an obstacle (exact Euclidean, metres)."""
    blocked = np.asarray(occupancy) > 0.5
    if radius_m <= 0:
        return blocked.astype(np.float32)
    distance_m = ndimage.distance_transform_edt(~blocked) * float(res_m)
    return (distance_m <= float(radius_m)).astype(np.float32)


def map_key(occ: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(occ, dtype=np.uint8).tobytes()).hexdigest()


def load_split(split: str) -> dict:
    folder = os.path.join(SOURCE_ROOT, split)
    out = {}
    for name in ("occupancy", "conditions", "curve_gt", "control_gt", "heading",
                 "heading_valid", "episode_id", "sample_id"):
        path = os.path.join(folder, "%s.npy" % name)
        out[name] = np.load(path, mmap_mode="r") if name == "occupancy" else np.load(path)
    return out


def load_manifest() -> dict:
    path = os.path.join(SOURCE_ROOT, "clean_manifest.jsonl")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return {int(r["sample_id"]): r for r in map(json.loads, fh) if r.get("sample_id")}


def convert_split(split: str, args, codec) -> dict:
    data = load_split(split)
    manifest = load_manifest()
    tf = transform()
    n = len(data["sample_id"]) if args.limit is None else min(args.limit, len(data["sample_id"]))
    out_split = os.path.join(OUT_ROOT, "processed", split)
    os.makedirs(out_split, exist_ok=True)

    # the map is shared by ~48 samples: inflate + sdf once per distinct bitmap
    maps, records = {}, []
    started = time.time()
    for i in range(n):
        raw = np.asarray(data["occupancy"][i])
        key = map_key(raw)
        if key not in maps:
            blocked = inflate(raw, args.robot_radius_m, SOURCE_RES_M)
            maps[key] = {"occupancy": blocked, "sdf": sdf_from_occupancy(blocked, SOURCE_RES_M)}

        cond = np.asarray(data["conditions"][i], dtype=np.float32)
        curve_scene = np.asarray(data["curve_gt"][i], dtype=np.float32)
        controls_scene = np.asarray(data["control_gt"][i], dtype=np.float32)
        if args.num_controls is not None and args.num_controls != controls_scene.shape[0]:
            import torch
            curve = torch.as_tensor(curve_scene, dtype=torch.float32)[None]
            controls_scene = codec.fit_curve_to_controls(
                curve, torch.as_tensor(cond[0], dtype=torch.float32)[None],
                torch.as_tensor(cond[1], dtype=torch.float32)[None])[0].numpy()

        sample_id = int(data["sample_id"][i])
        meta = manifest.get(sample_id, {})
        record = {"sample_id": sample_id,
                  "map_key": key,
                  "map_group": int(data["episode_id"][i]),
                  "scene_id": meta.get("scene_id", int(data["episode_id"][i])),
                  "town": meta.get("town"),
                  "tier": meta.get("tier"),
                  "seed": i}
        records.append(record)
        np.savez_compressed(
            os.path.join(out_split, "%d.npz" % sample_id),
            occupancy=maps[key]["occupancy"],
            sdf=maps[key]["sdf"],
            start_goal_scene=cond,
            start_goal_m=tf.to_meters(cond).astype(np.float32),
            heading=np.asarray(data["heading"][i], dtype=np.float32),
            heading_valid=np.asarray(data["heading_valid"][i], dtype=bool),
            controls_scene=controls_scene,
            path_xy_scene=curve_scene,
            path_xy_m=tf.to_meters(curve_scene).astype(np.float32),
            meta_json=np.array(json.dumps({**record, "res_m": SOURCE_RES_M,
                                           "source": os.path.basename(SOURCE_ROOT)})),
        )
        if args.verbose and (i + 1) % 200 == 0:
            print("  %s: %d/%d (%.1fs)" % (split, i + 1, n, time.time() - started), flush=True)

    with open(os.path.join(OUT_ROOT, "%s.jsonl" % split), "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    summary = {"split": split, "samples": n, "maps": len(maps),
               "seconds": round(time.time() - started, 2)}
    print(json.dumps(summary), flush=True)
    return summary


def main() -> int:
    global OUT_ROOT
    parser = argparse.ArgumentParser(description="array root -> geopath_v1 layout")
    parser.add_argument("--out", default=OUT_ROOT)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N samples per split (smoke runs)")
    parser.add_argument("--robot-radius-m", type=float, default=1.1,
                        help="inflation applied to the stored (eroded) bitmap; "
                             "1.1 = 0.8 robot + 0.3 margin used upstream")
    parser.add_argument("--num-controls", type=int, default=None,
                        help="refit the controls to this C (default: keep the source C)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    OUT_ROOT = args.out if os.path.isabs(args.out) else os.path.join(PROJECT_ROOT, args.out)
    os.makedirs(OUT_ROOT, exist_ok=True)

    codec = None
    if args.num_controls is not None:
        from gp_diffuser.factory import build_codec
        codec = build_codec({"trajectory": {"degree": 3, "num_controls": args.num_controls,
                                            "curve_points": 128}})

    summaries = [convert_split(s, args, codec) for s in args.splits.split(",") if s]
    manifest = {"dataset_version": "geopath_v1_k4p",
                "source": os.path.relpath(SOURCE_ROOT, PROJECT_ROOT),
                "res_m": SOURCE_RES_M, "grid": SOURCE_GRID,
                "extent_m": SOURCE_RES_M * SOURCE_GRID,
                "robot_radius_m": args.robot_radius_m,
                "splits": {row["split"]: {"samples": row["samples"], "maps": row["maps"]}
                           for row in summaries}}
    with open(os.path.join(OUT_ROOT, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
