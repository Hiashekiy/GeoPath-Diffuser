"""Generate the safety-condition cache for the REAL dataset.

Runs《GeoPath-Diffuser_数据集制作方案.md》§8 on a prepared split:

    GT path -> Astar-Diffusion guide search -> searched path
            -> frozen PathSafe-EllipseNet -> 64 ellipses
            -> convex regions -> half-space table -> safety cache

The geometry code is the existing pipeline (``build_safety_cache``); this script
only supplies the real loader, the split, and the dataset-aware fingerprint.

Usage::

    python scripts/build_real_safety_cache.py --split train
    python scripts/build_real_safety_cache.py --split val --limit 200
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import PROJECT_ROOT, base_parser, load  # noqa: E402

from gp_diffuser.factory import build_pipeline  # noqa: E402
from gp_diffuser.training.real_dataset import RealTrajectoryDataset  # noqa: E402
from gp_diffuser.training.safety_cache import build_safety_cache  # noqa: E402


def main() -> int:
    parser = base_parser("build the safety cache for a real split")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None)
    opts = parser.parse_args()

    cfg = load(opts)
    pipe = build_pipeline(cfg, device=opts.device)
    dataset = RealTrajectoryDataset(cfg, opts.split, limit=opts.limit)
    print("split=%s samples=%d groups=%d digest=%s"
          % (opts.split, len(dataset), len({r["map_group"] for r in dataset.records}),
             dataset.digest), flush=True)

    started = time.time()
    info = build_safety_cache(dataset, pipe["net"], pipe["safety_generator"],
                              pipe["guidance"], cfg, split=opts.split,
                              limit=opts.limit)
    meta = info["meta"]
    corridor = meta["corridor"]
    search = meta["search"]

    def rate(flag):
        return float(sum(1 for s in search if bool(s.get(flag)))) / max(len(search), 1)

    summary = {
        "split": opts.split,
        "samples": meta["size"],
        "dataset": meta.get("dataset"),
        "fingerprint": meta["fingerprint"],
        "scope": meta["scope"],
        "search_ok_rate": rate("search_ok"),
        "search_status": {
            status: sum(1 for s in search if s.get("status") == status)
            for status in {s.get("status") for s in search}},
        "local_cells_mean": float(sum(c["local_cells"] for c in corridor)) / max(len(corridor), 1),
        "corridor_valid_rate": float(sum(1 for c in corridor if c["local_cells"] > 0))
        / max(len(corridor), 1),
        "bridges_total": int(sum(c["bridges"] for c in corridor)),
        "convex_regions_total": int(sum(c["local_cells"] for c in corridor)),
        "elapsed_s": round(time.time() - started, 1),
    }
    reports = os.path.join(dataset.root, "reports")
    os.makedirs(reports, exist_ok=True)
    path = os.path.join(reports, "safety_cache_audit_%s.json" % opts.split)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("written:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
