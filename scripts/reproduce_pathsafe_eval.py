"""Reproduce PathSafe-Ellipse's own ``visualize_grid`` evaluation with the vendored code.

It runs **their** evaluation script end to end (their ``collect``, their
``exact_trajectory``, their ``build_polytopes``, their plotting/metrics) against
the same checkpoint, split and sampling seed as the committed reference run in
``references/PathSafe-Ellipse/results/figures/grid``, and writes the result next to
it so the numbers can be diffed directly.

The only things this script supplies are the two paths the upstream config points
at a machine that no longer exists (``data/trajectories`` and
``data/distance_fields``), and the missing ``ellipsenet.data.dataset`` module
(restored in ``vendor/ellipsenet/data/dataset.py``).

Usage::

    python scripts/reproduce_pathsafe_eval.py --n 24 --seed 11
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import PROJECT_ROOT, base_parser  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

VENDOR = os.path.join(PROJECT_ROOT, "src", "gp_diffuser", "vendor")
sys.path.insert(0, VENDOR)

from ellipsenet.data.dataset import build_loader, load_index, split_by_group  # noqa: E402
from ellipsenet.models.psenet import PSENet  # noqa: E402
from ellipsenet.visualize_grid import collect  # noqa: E402


def main() -> int:
    parser = base_parser("reproduce the PathSafe-Ellipse grid evaluation")
    parser.add_argument("--checkpoint",
                        default="references/PathSafe-Ellipse/checkpoints/v3_best.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--out", default="outputs/reports/pathsafe_repro")
    parser.add_argument("--reference", default="references/PathSafe-Ellipse/results/"
                                              "figures/grid/summary.json")
    opts = parser.parse_args()

    device = torch.device(opts.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = os.path.join(PROJECT_ROOT, opts.checkpoint)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    model = PSENet(cfg).to(device).eval()
    model.load_state_dict(ck["model"])

    # upstream config points at the original machine's data roots
    data_cfg = cfg["data"]
    data_cfg["trajectory_dir"] = os.path.join(PROJECT_ROOT, "data", "trajectories")
    data_cfg["sdf_dir"] = os.path.join(PROJECT_ROOT, "data", "distance_fields")

    entries = load_index(data_cfg["sdf_dir"])
    tr, va, te = split_by_group(entries, frac=tuple(data_cfg["split_frac"]),
                                seed=int(data_cfg["split_seed"]))
    pick = {"train": tr, "val": va, "test": te}[opts.split]
    pick = [i for i in pick if int(entries[i]["n_points"]) <= int(data_cfg["max_points"])]
    rng = np.random.default_rng(opts.seed)
    pick = [pick[i] for i in sorted(rng.choice(len(pick), size=min(opts.n, len(pick)),
                                              replace=False))]
    loader = build_loader([entries[i] for i in pick], data_cfg["trajectory_dir"],
                          data_cfg["sdf_dir"], opts.batch_size, cfg,
                          augment=False, shuffle=False)

    rows = collect(model, loader, device, regions=True, window_margin=3.0)

    outdir = os.path.join(PROJECT_ROOT, opts.out)
    os.makedirs(outdir, exist_ok=True)

    def _nanmean(values):
        arr = np.asarray([v for v in values], dtype=np.float64)
        return float(np.nanmean(arr)) if arr.size else float("nan")

    safe_pt = np.concatenate([r["safe"] for r in rows]) if rows else np.array([])
    safe_tr = [float(r["safe"].mean()) for r in rows]
    summary = {
        "checkpoint": ckpt,
        "split": opts.split,
        "n_trajectories": len(rows),
        "n_points": int(sum(r["n_points"] for r in rows)),
        "exact_safe_rate_per_point": float(safe_pt.mean()) if safe_pt.size else float("nan"),
        "exact_safe_rate_per_trajectory": _nanmean(safe_tr),
        "trajectories_fully_safe": float(np.mean([r["corridor_ok"] for r in rows])) if rows else 0.0,
        "corridor_success_rate": float(np.mean([r["corridor_ok"] for r in rows])) if rows else 0.0,
        # exactly upstream's definitions: continuity = mean(pair_ok),
        # cover = mean(pair_cover), area = median over trajectories of
        # median(pi a b) per trajectory
        "midpoint_continuity": _nanmean([np.mean(r["pair_ok"])
                                         for r in rows if len(r["pair_ok"])]),
        "boundary_cover": _nanmean([np.mean(r["pair_cover"])
                                    for r in rows if len(r["pair_cover"])]),
        "area_p50_m2": float(np.median([np.median(np.pi * r["a"] * r["b"])
                                        for r in rows])),
        "per_trajectory": [
            {"scene_id": r["scene_id"], "file": r["file"], "n_points": r["n_points"],
             "resolution_m": r["resolution_m"],
             "exact_safe_rate": float(r["safe"].mean()),
             "midpoint_continuity": float(np.nanmean(r["pair_cover"]))
             if len(r["pair_cover"]) else None,
             "corridor_success": int(bool(r["corridor_ok"])),
             "regions": int(len(r["regions"]) if r["regions"] else 0),
             "regions_ok": int(np.sum(r["regions_ok"])) if r["regions_ok"] is not None else 0,
             "regions_full": int(np.sum(r["regions_full"])) if r["regions_full"] is not None else 0}
            for r in rows],
    }
    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print(json.dumps({k: v for k, v in summary.items() if k != "per_trajectory"},
                     indent=2, ensure_ascii=False))

    ref_path = os.path.join(PROJECT_ROOT, opts.reference)
    if os.path.exists(ref_path):
        ref = json.load(open(ref_path, encoding="utf-8"))
        print("\n%-34s %12s %12s %10s" % ("metric", "reference", "reproduced", "delta"))
        for key in ("n_trajectories", "n_points", "exact_safe_rate_per_point",
                    "exact_safe_rate_per_trajectory", "corridor_success_rate",
                    "midpoint_continuity", "boundary_cover", "area_p50_m2"):
            a, b = ref.get(key), summary.get(key)
            if a is None or b is None:
                continue
            print("%-34s %12.4f %12.4f %10.4f" % (key, a, b, b - a))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
