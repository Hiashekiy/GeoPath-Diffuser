"""Prove the corridor config migration (ALM -> corridor) changed no geometry.

The retired ``alm`` block used to feed a few PURE GEOMETRY parameters into the
corridor builder and ``alm.constraint_margin`` into the constraint pack.  This
script rebuilds the same scene twice:

* ``legacy``  - the old mapping: ``{**corridor.region, **<alm geometry keys>}``
  with ``margin = alm.constraint_margin`` (read from git's HEAD config);
* ``current`` - the new mapping: ``corridor.region`` only, with
  ``margin = corridor.constraint_margin_scene``.

and compares corridor cells (anchors / halfspaces), the whole
:class:`BSplineConstraintPack` summary and the number of active half-spaces.
Any difference means the migration changed the geometry, which is forbidden.

    python scripts/verify_corridor_equivalence.py --index 2 --steps 6
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import PROJECT_ROOT, base_parser, load  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from gp_diffuser.factory import build_pipeline  # noqa: E402
from gp_diffuser.safety.bspline_constraints import build_constraint_pack  # noqa: E402
from gp_diffuser.safety.convex_region import EllipseRegionBuilder  # noqa: E402
from gp_diffuser.safety.safety_corridor import build_safety_corridor  # noqa: E402
from gp_diffuser.training.dataset import SyntheticTrajectoryDataset  # noqa: E402
from gp_diffuser.vendor.trajsafe.ellipse_densify import densify_ellipses  # noqa: E402

GEOMETRY_KEYS = ("safety_margin", "obstacle_window_half", "obstacle_dilation",
                 "corridor_chunk_size", "ellipse_axis_min", "ellipse_axis_max",
                 "map_boundary_faces")


def legacy_alm_config() -> dict:
    """The ``alm`` block of the reviewed commit (the pre-CPCG config)."""
    text = subprocess.run(["git", "-C", PROJECT_ROOT, "show",
                           "HEAD:configs/default.yaml"],
                          capture_output=True, check=True).stdout.decode("utf-8")
    return dict((yaml.safe_load(text) or {}).get("alm") or {})


def build(cfg, pipe, occ, centers, shape4, anchors, path_scene, builder_cfg,
          margin):
    builder = EllipseRegionBuilder(occ.float(), builder_cfg)
    dens = dict((cfg.get("corridor") or {}).get("densify") or {})
    if bool(dens.get("enabled", True)):
        c_np, s_np, a_np, _ = densify_ellipses(
            builder, centers.numpy(), shape4.numpy(), anchors.numpy(),
            min_overlap=float(dens.get("min_overlap", 0.10)),
            max_insert_per_gap=int(dens.get("max_insert_per_gap", 2)),
            max_total=int(dens.get("max_total", 256)))
        centers = torch.as_tensor(c_np, dtype=centers.dtype)
        shape4 = torch.as_tensor(s_np, dtype=shape4.dtype)
        anchors = torch.as_tensor(a_np, dtype=anchors.dtype)
    corridor = build_safety_corridor(builder, centers, shape4, anchors,
                                     gamma=path_scene,
                                     gamma_lengths=int(path_scene.shape[0]),
                                     config=cfg["corridor"])
    pack = build_constraint_pack(pipe["codec"], [corridor], device="cpu",
                                 dtype=torch.float32, margin=margin)
    return corridor, pack


def main() -> int:
    parser = base_parser("corridor config migration equivalence")
    parser.add_argument("--index", type=int, default=2)
    parser.add_argument("--steps", type=int, default=6)
    opts = parser.parse_args()

    cfg = load(opts)
    pipe = build_pipeline(cfg, device="cpu",
                          checkpoint=getattr(opts, "checkpoint", None))
    dataset = SyntheticTrajectoryDataset(cfg, "test")
    item = dataset[opts.index]
    occ = item["occupancy"][None]
    sdf = item["sdf"][None]
    sg = item["start_goal"][None]
    hd = item["heading"][None]
    hv = item["heading_valid"][None]

    out = pipe["sampler"].sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                                 steps=opts.steps, safety_mode="safety_only",
                                 seed=0)
    raw, _ = pipe["sampler"]._generate_safety(
        occ, sdf, pipe["map_encoder"].encode(occ, sdf),
        out["guide_trajectory_scene"], sg, pipe["sampler"]._corridor_cfg())
    centers = raw["ellipse_centers_scene"][0]
    shapes = raw["shape4_scene"][0]
    anchors = torch.as_tensor(raw["anchors"], dtype=torch.float64)
    path_scene = torch.as_tensor(pipe["transform"].to_scene(
        raw["search_path_m"][0].numpy()), dtype=torch.float32)

    alm = legacy_alm_config()
    legacy_builder = dict((cfg.get("corridor") or {}).get("region") or {})
    legacy_builder.update({k: alm[k] for k in GEOMETRY_KEYS if k in alm})
    legacy_margin = float(alm.get("constraint_margin", 0.0))

    corridor_new = dict((cfg.get("corridor") or {}).get("region") or {})
    current_margin = float((cfg.get("corridor") or {}).get("constraint_margin_scene",
                                                         0.0))

    old_corridor, old_pack = build(cfg, pipe, occ, centers, shapes, anchors,
                                   path_scene, legacy_builder, legacy_margin)
    new_corridor, new_pack = build(cfg, pipe, occ, centers, shapes, anchors,
                                   path_scene, corridor_new, current_margin)

    same_anchors = np.allclose(old_corridor.anchors(), new_corridor.anchors(),
                              atol=0.0)
    same_cells = (old_corridor.num_cells == new_corridor.num_cells
                  and old_corridor.valid == new_corridor.valid
                  and old_corridor.bridge_cell_count == new_corridor.bridge_cell_count)
    same_faces = all(
        np.array_equal(np.asarray(a.A), np.asarray(b.A))
        and np.array_equal(np.asarray(a.b), np.asarray(b.b))
        for a, b in zip(old_corridor.cells, new_corridor.cells))
    same_pack = (old_pack.summary() == new_pack.summary()
                 and int(old_pack.num_constraints) == int(new_pack.num_constraints))
    same_extraction = torch.equal(old_pack.extraction, new_pack.extraction)
    same_bounds = (torch.equal(old_pack.piece_A, new_pack.piece_A)
                   and torch.equal(old_pack.piece_b, new_pack.piece_b)
                   and torch.equal(old_pack.face_mask, new_pack.face_mask))

    report = {
        "index": int(opts.index),
        "legacy_builder_config": {k: legacy_builder[k] for k in sorted(legacy_builder)},
        "current_builder_config": {k: corridor_new[k] for k in sorted(corridor_new)},
        "legacy_margin": legacy_margin,
        "current_margin": current_margin,
        "corridor_valid": (bool(old_corridor.valid), bool(new_corridor.valid)),
        "corridor_cells": (int(old_corridor.num_cells), int(new_corridor.num_cells)),
        "bridge_cells": (int(old_corridor.bridge_cell_count),
                         int(new_corridor.bridge_cell_count)),
        "pack_summary": old_pack.summary(),
        "active_halfspaces": (int(old_pack.num_constraints),
                              int(new_pack.num_constraints)),
        "identical": {"anchors": bool(same_anchors), "cells": bool(same_cells),
                      "halfspaces": bool(same_faces), "pack": bool(same_pack),
                      "extraction": bool(same_extraction),
                      "bounds": bool(same_bounds)},
    }
    print(yaml.safe_dump(report, allow_unicode=True, sort_keys=False))
    ok = all(report["identical"].values())
    print("GEOMETRY UNCHANGED" if ok else "GEOMETRY CHANGED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
