"""Offline safety-condition cache (design document section 9.1).

For every training sample the safety condition is produced ONCE by the frozen
modules and cached:

    GT path label
      -> Astar-Diffusion guided search  -> training search path (128 points)
      -> frozen PathSafe-EllipseNet      -> 64 ellipses
      -> convex-region table             -> half-spaces, assignment, mask

The cache is a single ``.npz`` per split plus a small JSON with the metadata.
Nothing here is differentiated: the encoders and the ellipse network are frozen
and drop out of the graph entirely.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch

from ..frames import SceneTransform
from ..guidance.search import resample_polyline
from ..safety.local_regions import build_region_table


def cache_path(cfg: dict, split: str) -> str:
    from ..config import project_path
    cache_cfg = dict(cfg.get("dataset") or {})
    directory = cache_cfg.get("cache_dir", "outputs/cache")
    return project_path(directory, "safety_%s.npz" % split)


@torch.no_grad()
def build_safety_cache(dataset, net, safety_generator, guidance, cfg: dict,
                       split: str = "train", limit: Optional[int] = None,
                       verbose: bool = True) -> Dict:
    """Run the frozen pipeline over ``dataset`` and persist the safety cache."""
    from ..config import project_path  # noqa: F401  (kept for path resolution)

    cache_cfg = dict(cfg.get("dataset") or {})
    scope = str((cfg.get("corridor") or {}).get("training_scope", "full"))
    res_m = float(cache_cfg.get("res_m", 0.25))
    grid = int(cache_cfg.get("grid", 256))
    transform = SceneTransform(res_m * grid, grid)
    corridor_cfg = dict(cfg.get("corridor") or {})

    n = len(dataset) if limit is None else min(int(limit), len(dataset))
    paths_scene: List[np.ndarray] = []
    ellipses_scene: List[np.ndarray] = []
    tables: List[Dict] = []
    infos: List[Dict] = []
    search_stats: List[Dict] = []

    for i in range(n):
        scene = dataset.raw_scene(i)
        occ = scene["occupancy"]
        sdf = scene["sdf"]
        # 1) training guide source = the GT path label (section 5.1)
        guide_m = np.asarray(scene["gt_path_m"], dtype=np.float64)
        start_m = np.asarray(scene["start_m"], dtype=np.float64)
        goal_m = np.asarray(scene["goal_m"], dtype=np.float64)
        result = guidance.search(occ, guide_m, start_m, goal_m)
        search_m = result["path_xy"]
        status = result["status"]
        if search_m is None:
            search_m = guide_m
            status = "fallback_to_guide"
        search_stats.append({"status": status, "fallback": result.get("fallback"),
                             "expanded": int(result.get("expanded", 0)),
                             "valid": bool(result.get("valid", False))})

        path_m_t = torch.as_tensor(np.asarray(search_m), dtype=torch.float32)[None]
        occ_t = torch.as_tensor(occ, dtype=torch.float32)[None, None]
        sdf_t = torch.as_tensor(sdf, dtype=torch.float32)[None, None]
        safety = safety_generator(occ_t, sdf_t, path_m_t)

        centers_m = safety["ellipse_center"][0].numpy()
        axes_m = np.stack([safety["ellipse"][0, :, 2].numpy(),
                           safety["ellipse"][0, :, 3].numpy()], axis=-1)
        theta = safety["ellipse"][0, :, 4:].numpy()
        centers_scene = transform.to_scene(centers_m)
        ellipse_scene = np.concatenate([centers_scene,
                                        transform.lengths_to_scene(axes_m), theta], axis=-1)

        # 2) convex region table (scope from config: local at inference, full here)
        Q = int(centers_scene.shape[0])
        anchors = np.linspace(0.0, 1.0, Q)
        gt_curve_scene = torch.as_tensor(transform.to_scene(scene["gt_path_m"]),
                                        dtype=torch.float32)[None]
        table, info = build_region_table(
            occ_t, torch.as_tensor(centers_scene, dtype=torch.float32),
            safety["ellipse_shape4"][0], torch.as_tensor(anchors),
            curve_scene=gt_curve_scene,
            gamma_scene=torch.as_tensor(transform.to_scene(search_m), dtype=torch.float32),
            config=corridor_cfg, scope=scope)

        paths_scene.append(transform.to_scene(search_m).astype(np.float32))
        ellipses_scene.append(ellipse_scene.astype(np.float32))
        tables.append(table)
        infos.append(info)
        if verbose and (i + 1) % 8 == 0:
            print("  safety cache %s: %d/%d" % (split, i + 1, n), flush=True)

    max_faces = max(t["cell_A"].shape[1] for t in tables)
    R = tables[0]["cell_A"].shape[0]
    cell_A = np.zeros((n, R, max_faces, 2), dtype=np.float32)
    cell_b = np.full((n, R, max_faces), np.inf, dtype=np.float32)
    cell_mask = np.zeros((n, R), dtype=bool)
    anchors = np.zeros((n, R), dtype=np.float64)
    for i, table in enumerate(tables):
        faces = table["cell_A"].shape[1]
        cell_A[i, :, :faces] = table["cell_A"].astype(np.float32)
        cell_b[i, :, :faces] = table["cell_b"].astype(np.float32)
        cell_mask[i] = table["cell_mask"]
        anchors[i] = table["anchors"]

    out = cache_path(cfg, split)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out,
                        path_scene=np.stack(paths_scene),
                        ellipse_scene=np.stack(ellipses_scene),
                        cell_A=cell_A, cell_b=cell_b, cell_mask=cell_mask,
                        anchors=anchors)
    meta = {
        "split": split,
        "size": int(n),
        "scope": scope,
        "path_points": int(paths_scene[0].shape[0]),
        "num_ellipses": int(R),
        "max_faces": int(max_faces),
        "scene_grid": int(grid),
        "res_m": res_m,
        "search": search_stats,
        "corridor": infos,
    }
    with open(out.replace(".npz", "_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=float)
    print("safety cache written: %s (%d samples)" % (out, n), flush=True)
    return {"path": out, "meta": meta}


def load_safety_cache(cfg: dict, split: str) -> Optional[Dict[str, np.ndarray]]:
    path = cache_path(cfg, split)
    if not os.path.exists(path):
        return None
    with np.load(path) as data:
        return {k: np.asarray(data[k]) for k in data.files}
