"""Real-dataset loader for the processed GeoPath-Diffuser v1 dataset.

The processed layout is produced by ``scripts/prepare_real_dataset.py``::

    data/geopath_v1/
    ├── manifest.json            dataset version + map spec + counts
    ├── split_manifest.json      map_group -> split assignment
    ├── train.jsonl / val.jsonl / test.jsonl
    ├── processed/<split>/<sample_id>.npz
    └── reports/

One processed NPZ holds the whole training sample (occupancy, sdf, 128-point GT
path in metres and scene coordinates, 32 B-spline controls, start/goal, heading,
quality metrics).  Maps are NOT duplicated per trajectory inside the archive:
the same occupancy/sdf pair is shared by every path of one ``map_key``, and the
loader keeps a small LRU cache.

Interfaces follow the existing synthetic dataset exactly, so
``DualBranchTrainer`` / ``build_safety_cache`` work unchanged:

    __getitem__(i) -> dict of tensors (occupancy, sdf, start_goal, heading,
                      heading_valid, controls_gt, path_gt_scene, path_gt_m,
                      start_m, goal_m, seed, res_m, scene_scale)
    raw_scene(i)   -> dict of numpy (occupancy, sdf, gt_path_m, start_m, goal_m,
                      scene_id, map_group)
"""
from __future__ import annotations

import json
import os
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import project_path
from ..frames import SceneTransform

__all__ = ["RealTrajectoryDataset", "load_samples", "dataset_root",
           "split_digest"]


def dataset_root(cfg: dict) -> str:
    data_cfg = dict(cfg.get("dataset") or {})
    return project_path(data_cfg.get("real_root", "data/geopath_v1"))


def load_samples(root: str, split: str) -> List[Dict]:
    """Read ``<root>/<split>.jsonl`` (one JSON record per eligible trajectory)."""
    path = os.path.join(root, "%s.jsonl" % split)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "missing %s - run scripts/prepare_real_dataset.py first" % path)
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def split_digest(records: List[Dict]) -> str:
    """Stable hash of the sample ids of a split (goes into the cache fingerprint)."""
    import hashlib
    blob = "\n".join(sorted(str(r["sample_id"]) for r in records)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


class RealTrajectoryDataset(Dataset):
    """Processed samples of one split, with a per-``map_key`` map LRU cache."""

    def __init__(self, cfg: dict, split: str = "train", root: Optional[str] = None,
                 limit: Optional[int] = None, map_cache: int = 128):
        self.cfg = cfg
        data_cfg = dict(cfg.get("dataset") or {})
        self.res_m = float(data_cfg.get("res_m", 0.25))
        self.grid = int(data_cfg.get("grid", 256))
        self.extent_m = float(self.res_m * self.grid)
        self.transform = SceneTransform(self.extent_m, self.grid)
        self.scene_scale = self.transform.scale
        self.split = split
        self.root = root or dataset_root(cfg)
        self.records = load_samples(self.root, split)
        if limit is not None:
            self.records = self.records[:int(limit)]
        if not self.records:
            raise RuntimeError("split %r is empty (%s)" % (split, self.root))
        self.sample_ids = [str(r["sample_id"]) for r in self.records]
        self._map_cache: "OrderedDict[str, Dict[str, np.ndarray]]" = OrderedDict()
        self._map_cache_size = int(map_cache)
        self.digest = split_digest(self.records)

    # ------------------------------------------------------------------ helpers
    def __len__(self) -> int:
        return len(self.records)

    def path(self, index: int) -> str:
        record = self.records[index]
        return os.path.join(self.root, "processed", self.split,
                            "%s.npz" % record["sample_id"])

    def _load(self, index: int) -> Dict:
        with np.load(self.path(index)) as data:
            item = {k: np.asarray(data[k]) for k in data.files}
        item["meta"] = json.loads(str(item["meta_json"])) if "meta_json" in item else {}
        return item

    def map_of(self, index: int) -> Dict[str, np.ndarray]:
        """occupancy + sdf for the sample's ``map_key`` (LRU cached)."""
        key = str(self.records[index]["map_key"])
        if key in self._map_cache:
            self._map_cache.move_to_end(key)
            return self._map_cache[key]
        with np.load(self.path(index)) as data:
            entry = {"occupancy": np.asarray(data["occupancy"]),
                     "sdf": np.asarray(data["sdf"])}
        self._map_cache[key] = entry
        if len(self._map_cache) > self._map_cache_size:
            self._map_cache.popitem(last=False)
        return entry

    # ------------------------------------------------------------- trainer API
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        record = self.records[index]
        item = self._load(index)
        occupancy = torch.as_tensor(item["occupancy"], dtype=torch.float32)[None]
        sdf = torch.as_tensor(item["sdf"], dtype=torch.float32)[None]
        return {
            "occupancy": occupancy,
            "sdf": sdf,
            "start_goal": torch.as_tensor(item["start_goal_scene"], dtype=torch.float32),
            "heading": torch.as_tensor(item["heading"], dtype=torch.float32),
            "heading_valid": torch.as_tensor(item["heading_valid"], dtype=torch.bool),
            "controls_gt": torch.as_tensor(item["controls_scene"], dtype=torch.float32),
            "path_gt_scene": torch.as_tensor(item["path_xy_scene"], dtype=torch.float32),
            "path_gt_m": torch.as_tensor(item["path_xy_m"], dtype=torch.float32),
            "start_m": torch.as_tensor(item["start_goal_m"][0], dtype=torch.float32),
            "goal_m": torch.as_tensor(item["start_goal_m"][1], dtype=torch.float32),
            "seed": torch.tensor(int(record.get("seed", index)), dtype=torch.long),
            "res_m": torch.tensor(self.res_m),
            "scene_scale": torch.tensor(self.scene_scale),
        }

    # ---------------------------------------------------------- safety-cache API
    def raw_scene(self, index: int) -> Dict:
        record = self.records[index]
        item = self._load(index)
        return {
            "occupancy": np.asarray(item["occupancy"], dtype=np.float32),
            "sdf": np.asarray(item["sdf"], dtype=np.float32),
            "gt_path_m": np.asarray(item["path_xy_m"], dtype=np.float64),
            "start_m": np.asarray(item["start_goal_m"][0], dtype=np.float64),
            "goal_m": np.asarray(item["start_goal_m"][1], dtype=np.float64),
            "scene_id": record.get("scene_id"),
            "map_group": record.get("map_group"),
            "map_key": record.get("map_key"),
            "sample_id": record.get("sample_id"),
            "res_m": self.res_m,
            "grid": self.grid,
        }

    # ------------------------------------------------------------------ report
    def summary(self) -> Dict:
        groups = {str(r.get("map_group")) for r in self.records}
        maps = {str(r.get("map_key")) for r in self.records}
        return {"split": self.split, "samples": len(self.records),
                "map_groups": len(groups), "maps": len(maps),
                "digest": self.digest, "root": self.root}
