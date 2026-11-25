"""Build the offline safety-condition cache for one dataset split.

    GT path label -> Astar-Diffusion guided search -> search path (128 pts)
                  -> frozen PathSafe-EllipseNet -> 64 ellipses
                  -> convex-region table (half-spaces + assignment + mask)

Example::

    python scripts/build_safety_cache.py --split train --size 32
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import base_parser, load  # noqa: E402

from gp_diffuser.factory import build_pipeline  # noqa: E402
from gp_diffuser.training.dataset import SyntheticTrajectoryDataset  # noqa: E402
from gp_diffuser.training.safety_cache import build_safety_cache  # noqa: E402


def main() -> int:
    parser = base_parser("build the safety-condition cache")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--size", type=int, default=None, help="override the split size")
    opts = parser.parse_args()

    cfg = load(opts)
    pipe = build_pipeline(cfg, device=opts.device)
    dataset = SyntheticTrajectoryDataset(cfg, opts.split, size=opts.size)
    print("building safety cache: split=%s size=%d scope=%s"
          % (opts.split, len(dataset), cfg["corridor"].get("training_scope")))
    info = build_safety_cache(dataset, pipe["net"], pipe["safety_generator"],
                              pipe["guidance"], cfg, split=opts.split,
                              limit=opts.limit)
    meta = info["meta"]
    valid = sum(1 for c in meta["corridor"] if c["local_cells"] > 0)
    total_cells = sum(c["local_cells"] for c in meta["corridor"])
    bridges = sum(c["bridges"] for c in meta["corridor"])
    print("corridor built for %d/%d samples; mean local cells %.1f; bridges %d"
          % (valid, meta["size"], total_cells / max(meta["size"], 1), bridges))
    failures = {}
    for c in meta["corridor"]:
        for reason in c["corridor_failure"]:
            if reason:
                failures[reason.split(":")[0]] = failures.get(reason.split(":")[0], 0) + 1
    if failures:
        print("corridor failure reasons:", failures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
