"""Dual-branch joint training (one backward + one update per iteration).

The safety conditions come from the offline cache (``build_safety_cache.py``);
the frozen map encoder runs inside the loop but never receives gradient.

Example::

    python scripts/build_safety_cache.py --split train --size 16
    python scripts/train_dual_branch.py --steps 20 --batch-size 4
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import PROJECT_ROOT, base_parser, load  # noqa: E402

import numpy as np  # noqa: E402

from gp_diffuser.evaluation.visualize import render_training_curves  # noqa: E402
from gp_diffuser.factory import build_pipeline  # noqa: E402
from gp_diffuser.training.dataset import SyntheticTrajectoryDataset  # noqa: E402
from gp_diffuser.training.dual_branch_trainer import DualBranchTrainer  # noqa: E402
from gp_diffuser.training.safety_cache import load_safety_cache  # noqa: E402


def main() -> int:
    parser = base_parser("dual-branch joint training")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--save", default=None, help="checkpoint path (.pt)")
    opts = parser.parse_args()

    cfg = load(opts)
    if opts.size is not None:
        cfg["dataset"]["train_size"] = int(opts.size)
    if opts.lr is not None:
        cfg["training"]["lr"] = float(opts.lr)

    cache = load_safety_cache(cfg, "train")
    if cache is None:
        print("no safety cache found - run scripts/build_safety_cache.py first")
        return 2

    pipe = build_pipeline(cfg, device=opts.device)
    dataset = SyntheticTrajectoryDataset(cfg, "train")
    if len(dataset) != cache["path_scene"].shape[0]:
        print("cache covers %d samples, dataset has %d - rebuild the cache"
              % (cache["path_scene"].shape[0], len(dataset)))
        return 2
    items = [dataset[i] for i in range(len(dataset))]

    trainer = DualBranchTrainer(cfg, pipe["model"], pipe["map_projection"],
                                pipe["map_encoder"], pipe["codec"], pipe["schedule"],
                                device=opts.device, scene_scale=pipe["transform"].scale)
    print("trainable parameters: %d (frozen encoders: %d trainable)"
          % (trainer.trainable_parameter_count(),
             pipe["map_encoder"].trainable_parameter_count()))
    summary = trainer.fit(items, cache, epochs=opts.epochs, batch_size=opts.batch_size,
                          log_every=opts.log_every, seed=1,
                          max_steps=opts.steps,
                          ckpt_dir=cfg["training"]["ckpt_dir"])
    print(json.dumps(summary, indent=2))

    curve = render_training_curves(
        trainer.history,
        os.path.join(cfg["evaluation"]["figure_dir"], "training_curves.png"))
    if curve:
        print("figure:", curve)

    if opts.save:
        os.makedirs(os.path.dirname(os.path.abspath(opts.save)), exist_ok=True)
        import torch
        torch.save({"model": pipe["model"].state_dict(),
                    "map_projection": pipe["map_projection"].state_dict(),
                    "config": pipe["model"].config_dict(),
                    "history": trainer.history}, opts.save)
        print("checkpoint:", opts.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
