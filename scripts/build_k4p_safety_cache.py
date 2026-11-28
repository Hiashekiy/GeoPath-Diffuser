"""Build the offline safety cache for the k4p root **from its own labels**.

No network, no search: the k4p dataset already ships the geometry this project's
trainer consumes (``NEW_LABELS_REPORT.md``), so it is mapped 1:1 onto the cache
layout ``dual_branch_trainer._safety_conditions`` reads:

    path_scene    <- pdg_path_points                     [N,128,2]  (A* path, scene)
    ellipse_scene <- [cx,cy,a,b,cos2t,sin2t]             [N,64,6]
                     centres = resample_arclength(pdg_path_points, 64)
                     a,b     = exp(label log-axes)  -- ALREADY scene units (cap 0.25
                               = 20 m), so do NOT divide by scene_scale again
    ellipse_mask  <- shape_valid_q64                     [N,64]
    cell_A/cell_b <- alm_cell_a / alm_cell_b             [N,128,F,2] / [N,128,F]
    cell_mask     <- alm_cell_valid  (which of the 128 anchors has a cell)
    anchors       <- alm_anchor_s    (linspace(0,1,128), same convention as the code)
    cell_is_box   <- zeros                               (no map-box cells)

The ALM cells are anchored on ``candidate_xy[topology_best]``, NOT on the pdg
path: measured slack ``A c - b`` is <= -0.019 for every valid face on the
candidate polyline and up to +0.14 on the pdg path.  They are used as-is.

Usage::

    python scripts/build_k4p_safety_cache.py --config configs/k4p_160m_t12_c32.yaml \
        --splits train,val
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

from gp_diffuser.safety.safety_features import resample_arclength  # noqa: E402
from gp_diffuser.training.real_dataset import RealTrajectoryDataset  # noqa: E402
from gp_diffuser.training.safety_cache import (cache_path, dataset_descriptor,
                                               fingerprint)  # noqa: E402

SOURCE_ROOT = os.path.join(PROJECT_ROOT, "data", "160m-k4p-c32q64")


def build_split(cfg: dict, split: str, source_root: str) -> dict:
    dataset = RealTrajectoryDataset(cfg, split)
    n = len(dataset)
    folder = os.path.join(source_root, split)
    arrays = {name: np.load(os.path.join(folder, "%s.npy" % name))
              for name in ("pdg_path_points", "ellipse_shape4_gt_q64", "shape_valid_q64",
                           "alm_cell_a", "alm_cell_b", "alm_cell_valid", "alm_anchor_s")}
    if len(arrays["pdg_path_points"]) != n or int(arrays["alm_cell_a"].shape[0]) != n:
        raise RuntimeError("%s: label rows (%d) != dataset samples (%d)"
                           % (split, len(arrays["pdg_path_points"]), n))

    pdg = np.ascontiguousarray(arrays["pdg_path_points"], dtype=np.float32)
    shape4 = np.asarray(arrays["ellipse_shape4_gt_q64"], dtype=np.float32)   # [N,64,4] scene
    valid64 = np.asarray(arrays["shape_valid_q64"], dtype=bool)
    q = shape4.shape[1]
    centres = resample_arclength(torch.as_tensor(pdg), q).numpy()
    ellipse_scene = np.concatenate([centres,
                                    np.exp(shape4[..., 0:1]),      # a, scene units
                                    np.exp(shape4[..., 1:2]),      # b, scene units
                                    shape4[..., 2:4]], axis=-1).astype(np.float32)

    cell_A = np.asarray(arrays["alm_cell_a"], dtype=np.float32)
    cell_b = np.asarray(arrays["alm_cell_b"], dtype=np.float32)
    cell_mask = np.asarray(arrays["alm_cell_valid"], dtype=bool)
    anchors = np.asarray(arrays["alm_anchor_s"], dtype=np.float64)
    cell_is_box = np.zeros_like(cell_mask)

    out = cache_path(cfg, split)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(out, path_scene=pdg, ellipse_scene=ellipse_scene,
                        ellipse_mask=valid64, cell_A=cell_A, cell_b=cell_b,
                        cell_mask=cell_mask, cell_is_box=cell_is_box, anchors=anchors)
    meta = {"split": split, "size": int(n), "scope": "k4p_labels",
            "path_points": int(pdg.shape[1]), "num_ellipses": int(q),
            "max_rows": int(cell_A.shape[1]), "max_faces": int(cell_A.shape[2]),
            "scene_grid": int(cfg["dataset"]["grid"]), "res_m": float(cfg["dataset"]["res_m"]),
            "source": os.path.relpath(source_root, PROJECT_ROOT),
            "anchors_on": "candidate_xy[topology_best] (upstream ALM cells)",
            "ellipse_centres": "resample_arclength(pdg_path_points, %d)" % q,
            "fingerprint": fingerprint(cfg, split, n, dataset=dataset),
            "dataset": dataset_descriptor(dataset)}
    with open(out.replace(".npz", "_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=float)
    stats = {"split": split, "samples": int(n), "ellipses": int(q), "rows": int(cell_A.shape[1]),
             "faces": int(cell_A.shape[2]), "shape_valid": round(float(valid64.mean()), 4),
             "cell_valid": round(float(cell_mask.mean()), 4),
             "a_m": [round(float(np.exp(shape4[..., 0]).mean() * cfg["dataset"]["res_m"]
                                 * cfg["dataset"]["grid"] / 2.0), 2),
                     round(float(np.percentile(np.exp(shape4[..., 0]), 95)
                                 * cfg["dataset"]["res_m"] * cfg["dataset"]["grid"] / 2.0), 2)],
             "cache": out, "fingerprint": meta["fingerprint"]}
    print(json.dumps(stats), flush=True)
    return stats


def main() -> int:
    parser = base_parser("k4p: offline safety cache from the dataset labels")
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--source-root", default=SOURCE_ROOT)
    opts = parser.parse_args()
    cfg = load(opts)
    for split in [s.strip() for s in opts.splits.split(",") if s.strip()]:
        build_split(cfg, split, opts.source_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
