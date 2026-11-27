"""Offline safety-condition cache (design document section 9.1).

For every training sample the safety condition is produced ONCE by the frozen
modules and cached:

    GT path label
      -> Astar-Diffusion guided search  -> training search path (128 points)
      -> frozen PathSafe-EllipseNet      -> 64 ellipses
      -> convex-region table             -> half-spaces, assignment, mask

Nothing here is differentiated: the encoders and the ellipse network are frozen
and drop out of the graph entirely.

Post-review changes (``docs/code_review_verification.md``):

* **P0-2** the table gets ``shape4`` in SCENE units (the region builder
  exponentiates the log-axes);
* **P2-2** every tensor is built on the device of the frozen network;
* **P2-3** the cache records a fingerprint (config subset + dataset shape +
  checkpoint identity) and :func:`load_safety_cache` refuses a stale file.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch

from ..frames import SceneTransform
from ..safety.local_regions import build_region_table
from ..safety.safety_features import ellipse_shape4_to_scene


def cache_path(cfg: dict, split: str) -> str:
    from ..config import project_path
    cache_cfg = dict(cfg.get("dataset") or {})
    directory = cache_cfg.get("cache_dir", "outputs/cache")
    return project_path(directory, "safety_%s.npz" % split)


def dataset_descriptor(dataset) -> Optional[dict]:
    """Identity of the sample set, if the dataset exposes one (real datasets do).

    ``None`` for the synthetic dataset, so existing caches stay valid.
    """
    digest = getattr(dataset, "digest", None)
    if digest is None:
        return None
    return {"kind": type(dataset).__name__, "digest": str(digest),
            "samples": len(dataset), "root": getattr(dataset, "root", None),
            "split": getattr(dataset, "split", None)}


def fingerprint(cfg: dict, split: str, size: int, checkpoint: Optional[str] = None,
                dataset=None) -> str:
    """Stable hash of everything that changes the cached geometry.

    ``checkpoint`` defaults to the resolved PathSafe checkpoint, so a builder and a
    loader compute the same value without passing it around.  ``dataset`` adds the
    sample-set identity (split digest) when the dataset exposes one.
    """
    if checkpoint is None:
        try:
            from ..config import resolve_checkpoint
            checkpoint = resolve_checkpoint(cfg)
        except Exception:  # noqa: BLE001 - a missing checkpoint only weakens the hash
            checkpoint = None
    payload = {
        "split": split,
        "size": int(size),
        # identity of the code that produced the geometry: the cached ellipses and
        # corridors are only valid for the generator that built them
        "geometry_generator": "vendor/field_guided_search+vendor/ellipsenet/PSENet;r_scene=res_m",
        "dataset": {k: cfg.get("dataset", {}).get(k) for k in
                    ("res_m", "grid", "extent_m", "robot_radius_cells",
                     "min_geodesic_m", "path_points")},
        "corridor": cfg.get("corridor"),
        "safety_condition": cfg.get("safety_condition"),
        "trajectory": cfg.get("trajectory"),
        "guidance": cfg.get("guidance"),
        "checkpoint": None,
        "dataset": dataset_descriptor(dataset),
    }
    if checkpoint and os.path.exists(checkpoint):
        stat = os.stat(checkpoint)
        payload["checkpoint"] = {"path": os.path.abspath(checkpoint),
                                 "size": int(stat.st_size),
                                 "mtime": int(stat.st_mtime)}
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


@torch.no_grad()
def build_safety_cache(dataset, net, safety_generator, guidance, cfg: dict,
                       split: str = "train", limit: Optional[int] = None,
                       verbose: bool = True) -> Dict:
    """Run the frozen pipeline over ``dataset`` and persist the safety cache."""
    cache_cfg = dict(cfg.get("dataset") or {})
    scope = str((cfg.get("corridor") or {}).get("training_scope", "full"))
    res_m = float(cache_cfg.get("res_m", 0.25))
    grid = int(cache_cfg.get("grid", 256))
    transform = SceneTransform(res_m * grid, grid)
    corridor_cfg = dict(cfg.get("corridor") or {})
    device = next(net.parameters()).device if net is not None else torch.device("cpu")

    n = len(dataset) if limit is None else min(int(limit), len(dataset))
    paths_scene: List[np.ndarray] = []
    ellipses_scene: List[np.ndarray] = []
    tables: List[Dict] = []
    infos: List[Dict] = []
    search_stats: List[Dict] = []

    for i in range(n):
        scene = dataset.raw_scene(i)
        occ_np, sdf_np = scene["occupancy"], scene["sdf"]
        # 1) training guide source = the GT path label (section 5.1)
        guide_m = np.asarray(scene["gt_path_m"], dtype=np.float64)
        start_m = np.asarray(scene["start_m"], dtype=np.float64)
        goal_m = np.asarray(scene["goal_m"], dtype=np.float64)
        result = guidance.search(occ_np, guide_m, start_m, goal_m)
        search_m = result["path_xy"]
        status = result["status"]
        ok = search_m is not None and bool(result.get("valid", True))
        if not ok:
            search_m = guide_m
            status = "search_failed_guide_used_for_safety_only"
        search_stats.append({"index": int(i), "status": status,
                             "search_ok": bool(ok),
                             "fallback": result.get("fallback"),
                             "expanded": int(result.get("expanded", 0)),
                             "valid": bool(result.get("valid", False))})

        occ_t = torch.as_tensor(occ_np, dtype=torch.float32, device=device)[None, None]
        sdf_t = torch.as_tensor(sdf_np, dtype=torch.float32, device=device)[None, None]
        path_t = torch.as_tensor(search_m, dtype=torch.float32, device=device)[None]
        safety = safety_generator(occ_t, sdf_t, path_t)

        centers_m = safety["ellipse_center"][0].detach().cpu().numpy()
        axes_m = np.stack([safety["ellipse"][0, :, 2].detach().cpu().numpy(),
                           safety["ellipse"][0, :, 3].detach().cpu().numpy()], axis=-1)
        theta = safety["ellipse"][0, :, 4:].detach().cpu().numpy()
        centers_scene = transform.to_scene(centers_m)
        ellipse_scene = np.concatenate([centers_scene,
                                        transform.lengths_to_scene(axes_m), theta], axis=-1)
        # P0-2: the region builder needs SCENE log-axes
        shape4_scene = ellipse_shape4_to_scene(safety["ellipse_shape4"], transform.scale)

        Q = int(centers_scene.shape[0])
        anchors = torch.linspace(0.0, 1.0, Q, dtype=torch.float64, device=device)
        gt_curve_scene = torch.as_tensor(transform.to_scene(scene["gt_path_m"]),
                                         dtype=torch.float32, device=device)[None]
        table, info = build_region_table(
            occ_t, torch.as_tensor(centers_scene, dtype=torch.float32, device=device),
            shape4_scene[0], anchors,
            curve_scene=gt_curve_scene,
            gamma_scene=torch.as_tensor(transform.to_scene(search_m),
                                        dtype=torch.float32, device=device),
            config=corridor_cfg, scope=scope, scene_scale=transform.scale)

        paths_scene.append(transform.to_scene(search_m).astype(np.float32))
        ellipses_scene.append(ellipse_scene.astype(np.float32))
        tables.append({k: (np.asarray(v) if isinstance(v, np.ndarray) else v)
                       for k, v in table.items()})
        infos.append({"index": int(i), **info})
        if verbose and (i + 1) % 8 == 0:
            print("  safety cache %s: %d/%d" % (split, i + 1, n), flush=True)

    max_rows = max(t["cell_A"].shape[0] for t in tables)
    max_faces = max(t["cell_A"].shape[1] for t in tables)
    cell_A = np.zeros((n, max_rows, max_faces, 2), dtype=np.float32)
    cell_b = np.full((n, max_rows, max_faces), np.inf, dtype=np.float32)
    cell_mask = np.zeros((n, max_rows), dtype=bool)
    cell_is_box = np.zeros((n, max_rows), dtype=bool)
    anchors = np.zeros((n, max_rows), dtype=np.float64)
    for i, table in enumerate(tables):
        rows, faces = table["cell_A"].shape[:2]
        cell_A[i, :rows, :faces] = table["cell_A"].astype(np.float32)
        cell_b[i, :rows, :faces] = table["cell_b"].astype(np.float32)
        cell_mask[i, :rows] = table["cell_mask"]
        cell_is_box[i, :rows] = table.get("cell_is_box", np.zeros(rows, bool))
        anchors[i, :rows] = table["anchors"]

    out = cache_path(cfg, split)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out,
                        path_scene=np.stack(paths_scene),
                        ellipse_scene=np.stack(ellipses_scene),
                        cell_A=cell_A, cell_b=cell_b, cell_mask=cell_mask,
                        cell_is_box=cell_is_box, anchors=anchors)
    checkpoint = (getattr(net, "checkpoint_meta", None) or {}).get("path") \
        if net is not None else None
    descriptor = dataset_descriptor(dataset)
    meta = {
        "split": split,
        "size": int(n),
        "scope": scope,
        "path_points": int(paths_scene[0].shape[0]),
        "num_ellipses": int(cell_mask.shape[1]),
        "max_rows": int(max_rows),
        "max_faces": int(max_faces),
        "scene_grid": int(grid),
        "res_m": res_m,
        "fingerprint": fingerprint(cfg, split, n, checkpoint, dataset=dataset),
        "dataset": descriptor,
        "search": search_stats,
        "corridor": infos,
    }
    with open(out.replace(".npz", "_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=float)
    print("safety cache written: %s (%d samples, fingerprint %s)"
          % (out, n, meta["fingerprint"]), flush=True)
    return {"path": out, "meta": meta}


def load_safety_cache(cfg: dict, split: str, verify: bool = True,
                      verbose: bool = True, dataset=None
                      ) -> Optional[Dict[str, np.ndarray]]:
    """Load a cache, refusing one whose fingerprint no longer matches the config."""
    path = cache_path(cfg, split)
    if not os.path.exists(path):
        return None
    meta_path = path.replace(".npz", "_meta.json")
    recorded = None
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as fh:
            recorded = json.load(fh)
    if verify:
        if recorded is None:
            print("[cache] %s has no meta file - rebuilding" % path)
            return None
        size = int(recorded.get("size", -1))
        current = fingerprint(cfg, split, size, dataset=dataset)
        if recorded.get("fingerprint") != current:
            print("[cache] stale cache (%s): fingerprint %s != %s - rebuild with "
                  "scripts/build_safety_cache.py"
                  % (path, recorded.get("fingerprint"), current))
            return None
        if verbose:
            print("[cache] %s ok (fingerprint %s)" % (path, current))
    with np.load(path) as data:
        return {k: np.asarray(data[k]) for k in data.files}
