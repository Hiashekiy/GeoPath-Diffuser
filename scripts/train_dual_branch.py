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
import torch  # noqa: E402

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
    parser.add_argument("--dataset", default="synthetic", choices=["synthetic", "real"],
                        help="synthetic scenes or the processed geopath_v1 dataset")
    parser.add_argument("--real-split", default="train",
                        choices=["train", "val", "test"])
    parser.add_argument("--val-split", default=None,
                        help="optional split evaluated after training (e.g. val)")
    parser.add_argument("--limit", type=int, default=None,
                        help="only use the first N samples of the split")
    opts = parser.parse_args()

    cfg = load(opts)
    if opts.size is not None:
        cfg["dataset"]["train_size"] = int(opts.size)
    if opts.lr is not None:
        cfg["training"]["lr"] = float(opts.lr)

    # ---- dataset + cache (real or synthetic) -----------------------------
    cache_split = opts.real_split if opts.dataset == "real" else "train"
    if opts.dataset == "real":
        from gp_diffuser.training.real_dataset import RealTrajectoryDataset
        dataset = RealTrajectoryDataset(cfg, opts.real_split, limit=opts.limit)
        print("real dataset: %s" % json.dumps(dataset.summary(), ensure_ascii=False))
    else:
        dataset = SyntheticTrajectoryDataset(cfg, "train")
    cache = load_safety_cache(cfg, cache_split, dataset=dataset)
    if cache is None:
        print("no usable safety cache for split %r - run "
              "scripts/build_real_safety_cache.py (real) or "
              "scripts/build_safety_cache.py (synthetic) first" % cache_split)
        return 2

    pipe = build_pipeline(cfg, device=opts.device)
    if len(dataset) != cache["path_scene"].shape[0]:
        print("cache covers %d samples, dataset has %d - rebuild the cache"
              % (cache["path_scene"].shape[0], len(dataset)))
        return 2
    # keep the dataset lazy: a full real split is ~13k x (occupancy+sdf) and would
    # not fit in memory if materialised here
    items = dataset if opts.dataset == "real" else [dataset[i] for i in range(len(dataset))]

    trainer = DualBranchTrainer(cfg, pipe["model"],
                                pipe["map_condition_encoder"],
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
    if summary.get("last"):
        os.makedirs(os.path.join(PROJECT_ROOT, "outputs", "reports"), exist_ok=True)
        with open(os.path.join(PROJECT_ROOT, "outputs", "reports",
                               "train_history.json"), "w", encoding="utf-8") as fh:
            json.dump({"dataset": opts.dataset, "split": cache_split,
                       "config": {"num_controls": pipe["codec"].num_controls,
                                  "batch_size": opts.batch_size,
                                  "steps": summary["steps"],
                                  "lr": cfg["training"]["lr"]},
                       "summary": summary, "history": trainer.history},
                      fh, indent=2)

    # ---- optional validation pass (no gradient, no state change) ---------
    if opts.val_split:
        from gp_diffuser.training.real_dataset import RealTrajectoryDataset as _Real
        val = _Real(cfg, opts.val_split, limit=opts.limit)
        val_cache = load_safety_cache(cfg, opts.val_split, dataset=val)
        if val_cache is None:
            print("[val] no cache for split %r - skipped" % opts.val_split)
        else:
            rows = []
            for start in range(0, min(len(val), 128), opts.batch_size):
                idx = np.arange(start, min(start + opts.batch_size, len(val)))
                if len(idx) < 2:
                    break
                rows.append(trainer.eval_step([val[int(i)] for i in idx],
                                              val_cache, idx))
            if rows:
                mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
                print("[val] %s" % json.dumps({k: round(v, 5) for k, v in mean.items()}))
                with open(os.path.join(PROJECT_ROOT, "outputs", "reports",
                                       "val_metrics.json"), "w", encoding="utf-8") as fh:
                    json.dump({"split": opts.val_split, "batches": len(rows),
                               "mean": mean}, fh, indent=2)

    curve = render_training_curves(
        trainer.history,
        os.path.join(cfg["evaluation"]["figure_dir"], "training_curves.png"))
    if curve:
        print("figure:", curve)

    if opts.save:
        os.makedirs(os.path.dirname(os.path.abspath(opts.save)), exist_ok=True)
        import torch
        torch.save({"architecture_version": pipe["model"].architecture_version,
                    "model": pipe["model"].state_dict(),
                    "map_condition_encoder": pipe["map_condition_encoder"].state_dict(),
                    "config": pipe["model"].config_dict(),
                    "history": trainer.history}, opts.save)
        print("checkpoint:", opts.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
