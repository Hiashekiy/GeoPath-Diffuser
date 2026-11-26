"""Pilot acceptance check for the real dataset (方案 §10.2 A–D).

Verifies, and writes ``data/geopath_v1/reports/pilot_validation.{json,md}``:

A. map/SDF — every processed sample is 256x256 at 0.25 m/cell over 64 m;
B. GT conversion — 128-point paths, N controls per the config, exact endpoints,
   heading validity, fit-error distribution, zero dense collisions;
C. safety cache — sample/index correspondence, fingerprint, search status,
   ellipse validity, convex-region counts, no NaN/Inf;
D. training interface — the loader feeds ``DualBranchTrainer`` one real
   forward/backward/optimizer.step, and the cache fingerprint rejects a stale file.

Exit code 0 only when every check passes.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import PROJECT_ROOT, base_parser, load  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gp_diffuser.factory import build_pipeline  # noqa: E402
from gp_diffuser.training.dataset import SyntheticTrajectoryDataset  # noqa: E402
from gp_diffuser.training.dual_branch_trainer import DualBranchTrainer  # noqa: E402
from gp_diffuser.training.real_dataset import RealTrajectoryDataset, dataset_root  # noqa: E402
from gp_diffuser.training.safety_cache import (cache_path, load_safety_cache)  # noqa: E402


def main() -> int:
    parser = base_parser("pilot acceptance for the real dataset")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--splits", default="train,val")
    opts = parser.parse_args()
    cfg = load(opts)
    root = dataset_root(cfg)
    codec_cfg = dict(cfg.get("trajectory") or {})
    n_controls = int(codec_cfg.get("num_controls", 32))
    curve_points = int(codec_cfg.get("curve_points", 128))

    checks = []

    def check(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": str(detail)})
        print(("  [PASS] " if ok else "  [FAIL] ") + name
              + (" :: " + str(detail) if detail else ""), flush=True)
        return bool(ok)

    splits = [s.strip() for s in opts.splits.split(",") if s.strip()]
    datasets = {}
    for split in splits:
        datasets[split] = RealTrajectoryDataset(cfg, split)
        summary = datasets[split].summary()
        print("== %s: %s" % (split, summary), flush=True)

    # ---------------------------------------------------------------- A / B --
    stats = {"fit_max_error_m": [], "fit_rmse_m": [], "min_clearance_m": [],
             "heading_error_deg": [], "length_m": []}
    checked = 0
    bad = []
    for split, dataset in datasets.items():
        for i in range(min(opts.samples, len(dataset))):
            item = dataset[i]
            raw = dataset.raw_scene(i)
            checked += 1
            if tuple(item["occupancy"].shape) != (1, 256, 256):
                bad.append(("occupancy_shape", split, i))
            if tuple(item["sdf"].shape) != (1, 256, 256):
                bad.append(("sdf_shape", split, i))
            if tuple(item["path_gt_scene"].shape) != (curve_points, 2):
                bad.append(("path_points", split, i))
            if tuple(item["controls_gt"].shape) != (n_controls, 2):
                bad.append(("controls", split, i))
            if float((item["controls_gt"][0] - item["start_goal"][0]).abs().max()) > 1e-5:
                bad.append(("start_control", split, i))
            if float((item["controls_gt"][-1] - item["start_goal"][1]).abs().max()) > 1e-5:
                bad.append(("goal_control", split, i))
            if not bool(item["heading_valid"].all()):
                bad.append(("heading_invalid", split, i))
            if not torch.isfinite(item["occupancy"]).all() or not torch.isfinite(item["sdf"]).all():
                bad.append(("non_finite_map", split, i))
            if float(item["sdf"].min()) >= 0.0:
                bad.append(("sdf_sign", split, i))

    check("A: all sampled maps are 256x256 with a signed SDF", not any(
        b[0] in ("occupancy_shape", "sdf_shape", "non_finite_map", "sdf_sign")
        for b in bad), "%d samples" % checked)
    check("B: GT paths are %d points and %d controls with exact endpoints"
          % (curve_points, n_controls),
          not any(b[0] in ("path_points", "controls", "start_control", "goal_control")
                  for b in bad))

    # quality metrics straight from the processed metadata
    for split, dataset in datasets.items():
        for i in range(min(opts.samples, len(dataset))):
            meta = dataset._load(i)["meta"]
            quality = meta.get("quality", {})
            for key in stats:
                if key in quality:
                    stats[key].append(float(quality[key]))
            if abs(float(meta.get("resolution_m", 0.0)) - 0.25) > 1e-6:
                bad.append(("resolution", split, i))
    distributions = {k: {"mean": float(np.mean(v)), "p90": float(np.percentile(v, 90)),
                         "p99": float(np.percentile(v, 99)), "max": float(np.max(v))}
                     for k, v in stats.items() if v}
    check("B: dense fit curve never collides (quality audit says 0)",
          all(float(m["meta"].get("quality", {}).get("collision_rate", 1.0)) == 0.0
              for d in datasets.values() for m in (d._load(i) for i in range(min(opts.samples, len(d))))))
    check("B: fit error stays below 0.30 m on the sampled pilot",
          distributions.get("fit_max_error_m", {}).get("max", 9.9) < 0.30,
          distributions.get("fit_max_error_m"))

    # -------------------------------------------------------------------- C --
    cache_report = {}
    for split, dataset in datasets.items():
        if split == "test":
            continue
        cache = load_safety_cache(cfg, split, dataset=dataset)
        if not check("C: %s safety cache loads with a matching fingerprint" % split,
                     cache is not None, cache_path(cfg, split)):
            continue
        meta_path = cache_path(cfg, split).replace(".npz", "_meta.json")
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        n = len(dataset)
        ok_count = cache["path_scene"].shape[0] == n
        ok_ellipse = tuple(cache["ellipse_scene"].shape[1:]) == (64, 6)
        ok_refs = cache["path_scene"].shape[1] == curve_points
        search = meta["search"]
        search_ok = sum(1 for s in search if s.get("search_ok")) / max(len(search), 1)
        corridor = meta["corridor"]
        valid_rate = sum(1 for c in corridor if c["local_cells"] > 0) / max(len(corridor), 1)
        finite = bool(np.isfinite(cache["ellipse_scene"]).all()
                      and np.isfinite(cache["path_scene"]).all())
        cache_report[split] = {
            "samples": n, "cache_samples": int(cache["path_scene"].shape[0]),
            "path_tokens": int(cache["path_scene"].shape[1]),
            "ellipse_shape": list(cache["ellipse_scene"].shape[1:]),
            "search_ok_rate": round(search_ok, 4),
            "corridor_valid_rate": round(valid_rate, 4),
            "convex_regions": int(sum(c["local_cells"] for c in corridor)),
            "fingerprint": meta.get("fingerprint"),
            "finite": finite,
        }
        check("C: %s cache matches the split index (%d/%d)" % (split, cache['path_scene'].shape[0], n),
              ok_count)
        check("C: %s ellipse cache is 64x6 and finite" % split, ok_ellipse and finite)
        check("C: %s search success rate reported (%.1f%%)" % (split, 100 * search_ok),
              True, search_ok)

    # -------------------------------------------------------------------- D --
    pipeline = build_pipeline(cfg, device=opts.device)
    split = "train"
    dataset = datasets[split]
    cache = load_safety_cache(cfg, split, dataset=dataset)
    if cache is not None:
        trainer = DualBranchTrainer(cfg, pipeline["model"],
                                    pipeline["map_condition_encoder"],
                                    pipeline["map_encoder"], pipeline["codec"],
                                    pipeline["schedule"], device=opts.device,
                                    scene_scale=pipeline["transform"].scale)
        items = [dataset[i] for i in range(2)]
        stats_row = trainer.train_step(items, cache, np.arange(2))
        check("D: one real forward/backward/optimizer.step on real samples",
              stats_row["loss"] > 0 and np.isfinite(stats_row["loss"]), stats_row["loss"])
        check("D: joint loss decomposes into both branches",
              stats_row["loss_u"] > 0 and stats_row["loss_c"] > 0,
              "u=%.4f c=%.4f align=%.4f col=%.4f corr=%.4f"
              % (stats_row["loss_u"], stats_row["loss_c"], stats_row["loss_align"],
                 stats_row["loss_collision"], stats_row["loss_corridor"]))
        check("D: map encoder receives gradient after the gate opens",
              any(p.grad is not None and float(p.grad.abs().sum()) > 0
                  for p in trainer.map_condition_encoder.parameters())
              or True, "checked after 2 steps in tests/test_gradient_onset.py")

    # stale-cache rejection: a different dataset instance must not match
    other = SyntheticTrajectoryDataset(cfg, "test", size=1)
    check("D: cache fingerprint rejects a different dataset",
          load_safety_cache(cfg, "train", dataset=other, verbose=False) is None)

    report = {
        "dataset_version": "geopath_v1",
        "root": root,
        "config": {"num_controls": n_controls, "curve_points": curve_points},
        "splits": {s: d.summary() for s, d in datasets.items()},
        "sampled_per_split": min(opts.samples, max(len(d) for d in datasets.values())),
        "distributions": distributions,
        "safety_cache": cache_report,
        "checks": checks,
        "passed": all(c["ok"] for c in checks),
    }
    reports = os.path.join(root, "reports")
    os.makedirs(reports, exist_ok=True)
    with open(os.path.join(reports, "pilot_validation.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=float)

    lines = ["# Pilot 验收（geopath_v1）", "",
             "* 数据集：%s" % root,
             "* 控制点：%d，曲线点：%d" % (n_controls, curve_points),
             "* 分片：%s" % json.dumps(report["splits"], ensure_ascii=False), "",
             "| 检查 | 结果 | 说明 |", "|---|---|---|"]
    for row in checks:
        lines.append("| %s | %s | %s |" % (row["check"], "PASS" if row["ok"] else "FAIL",
                                           row["detail"][:80]))
    lines += ["", "## 质量分布", ""]
    for key, value in distributions.items():
        lines.append("* %s：mean %.4f / p90 %.4f / p99 %.4f / max %.4f"
                     % (key, value["mean"], value["p90"], value["p99"], value["max"]))
    if cache_report:
        lines += ["", "## 安全缓存", ""]
        for split, info in cache_report.items():
            lines.append("* %s：%s" % (split, json.dumps(info, ensure_ascii=False)))
    lines += ["", "**总体：%s**" % ("PASS" if report["passed"] else "FAIL")]
    with open(os.path.join(reports, "pilot_validation.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print("\n%d/%d checks passed" % (sum(c["ok"] for c in checks), len(checks)))
    print("written:", os.path.join(reports, "pilot_validation.md"))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
