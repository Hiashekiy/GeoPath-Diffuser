"""A/B comparison of the two training arms, both queried WITHOUT safety.

The question this script answers:

    a model that was trained WITH the safety chain, but is queried without it,
    does it beat a model that never saw the safety chain at all?

Both arms are sampled in ``safety_mode="map_only"``: the Map-only branch at every
reverse step, no guided search, no ellipse tokens, no ALM - the same inference
code path for both, same samples, same initial noise, same DDIM schedule.  The
only difference is the checkpoint.

Usage::

    python scripts/compare_safety_ablation.py \
        --arm dual=outputs/checkpoints/ab_dual/last.pt \
        --arm map_only=outputs/checkpoints/ab_map_only/last.pt \
        --split val --samples 64 --batch 8 --steps 50 --device cuda
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import PROJECT_ROOT, base_parser, load  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gp_diffuser.evaluation.metrics import trajectory_metrics  # noqa: E402
from gp_diffuser.factory import build_pipeline  # noqa: E402
from gp_diffuser.training.real_dataset import RealTrajectoryDataset  # noqa: E402

FIGURE_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures", "ablation_safety")


def _stats(values):
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return {}
    return {"mean": float(arr.mean()), "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)), "max": float(arr.max())}


def _chamfer_m(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric mean nearest-neighbour distance between two polylines (m)."""
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float(0.5 * (d.min(axis=1).mean() + d.min(axis=0).mean()))


def _stack(items, key):
    return torch.stack([it[key] for it in items])


def _homotopy_witness(guide, final, occupancy, sdf, res_m, blends=11):
    """Sufficient test that final lies in the guidance's own homotopy class.

    Straight-line interpolation ``(1-s)*guide + s*final`` sampled at ``s in [0,1]``
    is a free homotopy *witness*: if every blended curve is collision free, the two
    curves are provably connected without crossing an obstacle.  Collisions on the
    witness family do NOT prove the classes differ (a bad witness can collide even
    for identical classes), so this is a one-sided indicator.
    """
    worst = 0.0
    for s in np.linspace(0.0, 1.0, blends):
        blend = (1.0 - s) * np.asarray(guide) + s * np.asarray(final)
        worst = max(worst, float(trajectory_metrics(
            blend, occupancy, sdf, res_m)["collision_rate"]))
    return {"ok": bool(worst == 0.0), "worst_collision": worst}


def run_arm(name: str, checkpoint: str, cfg: dict, dataset, indices, opts) -> dict:
    # the ablation arm never trained its safety sub-modules (their weights are
    # still the initialisation), so a safety-enabled run would inject garbage
    trained_with_safety = torch.load(checkpoint, map_location="cpu",
                                     weights_only=False).get("use_safety")
    if trained_with_safety is False and opts.safety_mode != "map_only":
        raise SystemExit("arm %r was trained without the safety chain "
                         "(use_safety=False): it can only be sampled with "
                         "--safety-mode map_only, not %r"
                         % (name, opts.safety_mode))
    pipe = build_pipeline(cfg, device=opts.device, checkpoint=checkpoint)
    sampler = pipe["sampler"]
    scale = float(pipe["transform"].scale)
    res_m = float(dataset.res_m)
    records = []
    started = time.time()
    for begin in range(0, len(indices), opts.batch):
        chunk = list(indices[begin:begin + opts.batch])
        items = [dataset[int(i)] for i in chunk]
        with torch.no_grad():
            out = sampler.sample(
                _stack(items, "occupancy"), _stack(items, "sdf"),
                _stack(items, "start_goal"), heading=_stack(items, "heading"),
                heading_valid=_stack(items, "heading_valid"),
                steps=opts.steps, safety_mode=opts.safety_mode, seed=opts.seed)
        traj = out["trajectory_m"].numpy()
        controls = out["controls"].detach().cpu().numpy()
        guide = out.get("guide_trajectory_m")
        guide = None if guide is None else guide.detach().cpu().numpy()
        expanded = int(sum(int(s.get("expanded") or 0)
                           for s in (out.get("search_stats") or [])))
        heading_err = (None if out["heading_error_deg"] is None
                       else out["heading_error_deg"].numpy())
        for k, idx in enumerate(chunk):
            item = items[k]
            gt_m = item["path_gt_m"].numpy()
            metrics = trajectory_metrics(traj[k], item["occupancy"][0].numpy(),
                                         item["sdf"][0].numpy(), res_m)
            gt_len = float(np.linalg.norm(np.diff(gt_m, axis=0), axis=-1).sum())
            interior = slice(1, -1)
            ctrl_rmse = float(np.sqrt(np.mean(
                (controls[k][interior] - item["controls_gt"].numpy()[interior]) ** 2)) * scale)
            sg_m = item["start_goal"].numpy() * scale
            endpoint_m = float(max(np.linalg.norm(traj[k][0] - sg_m[0]),
                                   np.linalg.norm(traj[k][-1] - sg_m[1])))
            guide_metrics = (None if guide is None else
                             trajectory_metrics(guide[k], item["occupancy"][0].numpy(),
                                                item["sdf"][0].numpy(), res_m))
            witness = (None if guide is None else
                       _homotopy_witness(guide[k], traj[k], item["occupancy"][0].numpy(),
                                         item["sdf"][0].numpy(), res_m))
            records.append({
                "index": int(idx),
                "valid": bool(out["valid_mask"][k]),
                "collision_rate": float(metrics["collision_rate"]),
                "min_clearance_m": float(metrics["min_clearance_m"]),
                "length_m": float(metrics["length_m"]),
                "gt_length_m": gt_len,
                "length_ratio": float(metrics["length_m"]) / max(gt_len, 1e-6),
                "length_error": abs(float(metrics["length_m"]) / max(gt_len, 1e-6) - 1.0),
                "chamfer_m": _chamfer_m(traj[k], gt_m),
                "guide_collision_rate": (None if guide_metrics is None
                                         else float(guide_metrics["collision_rate"])),
                "guide_chamfer_to_gt_m": (None if guide is None
                                          else _chamfer_m(guide[k], gt_m)),
                "homotopy_witness_ok": (None if witness is None else witness["ok"]),
                "homotopy_witness_collision": (None if witness is None
                                               else witness["worst_collision"]),
                "guide_chamfer_to_final_m": (None if guide is None
                                             else _chamfer_m(guide[k], traj[k])),
                "search_expanded": expanded,
                "control_rmse_m": ctrl_rmse,
                "endpoint_error_m": endpoint_m,
                "heading_error_deg_max": (None if heading_err is None
                                          else float(heading_err[k].max())),
                "curve_m": traj[k].tolist(),
            })
    summary = {
        "arm": name, "checkpoint": os.path.relpath(checkpoint, PROJECT_ROOT),
        "trained_with_safety": trained_with_safety,
        "safety_mode": opts.safety_mode, "steps": int(opts.steps),
        "seed": int(opts.seed), "samples": len(records),
        "valid_rate": float(np.mean([r["valid"] for r in records])),
        "collision_free_rate": float(np.mean([r["collision_rate"] == 0
                                              for r in records])),
        "collision_rate": _stats([r["collision_rate"] for r in records]),
        "min_clearance_m": _stats([r["min_clearance_m"] for r in records]),
        "chamfer_m": _stats([r["chamfer_m"] for r in records]),
        "control_rmse_m": _stats([r["control_rmse_m"] for r in records]),
        "length_ratio": _stats([r["length_ratio"] for r in records]),
        "length_error": _stats([r["length_error"] for r in records]),
        "heading_error_deg_max": _stats([r["heading_error_deg_max"] for r in records]),
        "endpoint_error_m": _stats([r["endpoint_error_m"] for r in records]),
        "guide_collision_rate": _stats([r["guide_collision_rate"] for r in records]),
        "guide_chamfer_to_gt_m": _stats([r["guide_chamfer_to_gt_m"] for r in records]),
        "guide_chamfer_to_final_m": _stats([r["guide_chamfer_to_final_m"]
                                            for r in records]),
        "homotopy_witness_rate": (
            None if records[0]["homotopy_witness_ok"] is None
            else float(np.mean([r["homotopy_witness_ok"] for r in records]))),
        "seconds": round(time.time() - started, 1),
    }
    return {"summary": summary, "records": records}


def _paired(arms, key: str, lower_is_better: bool = True, other: int = 1):
    """Per-sample win/loss counts: the first arm against arm ``other``."""
    if len(arms) <= other:
        return {}
    a = {r["index"]: r for r in arms[0]["records"]}
    b = {r["index"]: r for r in arms[other]["records"]}
    common = sorted(set(a) & set(b))
    first = second = tie = 0
    for idx in common:
        va, vb = a[idx][key], b[idx][key]
        if abs(va - vb) < 1e-12:
            tie += 1
        elif (va < vb) == lower_is_better:
            first += 1
        else:
            second += 1
    return {"metric": key, "n": len(common), "first_better": first,
            "second_better": second, "tie": tie}


def _montage(arms, dataset, summaries, path, show=6):
    """Same scenes, both arms: occupancy + GT + predicted curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [a["summary"]["arm"] for a in arms]
    by_index = [{r["index"]: r for r in a["records"]} for a in arms]
    common = sorted(set.intersection(*[set(b) for b in by_index]))
    both_collide = [i for i in common
                    if all(by_index[k][i]["collision_rate"] > 0 for k in range(len(arms)))]
    disagree = sorted((i for i in common if len({by_index[k][i]["collision_rate"] > 0
                                                 for k in range(len(arms))}) > 1),
                      key=lambda i: -max(by_index[k][i]["collision_rate"]
                                         for k in range(len(arms))))
    clean = [i for i in common
             if all(by_index[k][i]["collision_rate"] == 0 for k in range(len(arms)))]
    order = (both_collide[:2] + disagree[:show - 4] + clean[:1] + disagree[show - 4:])
    picked = list(dict.fromkeys(order))[:show]
    if not picked:
        picked = common[:show]

    cols = 3
    rows = int(np.ceil(len(picked) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 4.4 * rows),
                             squeeze=False)
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    for ax, idx in zip(axes.ravel(), picked):
        raw = dataset.raw_scene(int(idx))
        occ = raw["occupancy"]
        ax.imshow(np.asarray(occ)[:, :, 0] if np.asarray(occ).ndim == 3 else occ,
                  origin="lower", cmap="Greys", extent=[0, occ.shape[-1], 0, occ.shape[-2]],
                  alpha=0.85)
        gt = raw["gt_path_m"]
        ax.plot(gt[:, 0] / raw["res_m"], gt[:, 1] / raw["res_m"], "-",
                color="#2ca02c", lw=1.6, alpha=0.75, label="GT")
        for k, arm in enumerate(arms):
            curve = np.asarray(by_index[k][idx]["curve_m"])
            label = "%s (coll %.2f)" % (names[k], by_index[k][idx]["collision_rate"])
            ax.plot(curve[:, 0] / raw["res_m"], curve[:, 1] / raw["res_m"], "-",
                    color=colors[k % len(colors)], lw=1.8, label=label)
        ax.plot(raw["start_m"][0] / raw["res_m"], raw["start_m"][1] / raw["res_m"], "o",
                color="black", ms=5)
        ax.plot(raw["goal_m"][0] / raw["res_m"], raw["goal_m"][1] / raw["res_m"], "s",
                color="black", ms=5)
        ax.set_title("val sample %d" % idx, fontsize=9)
        ax.legend(fontsize=6, loc="best")
        ax.tick_params(labelsize=6)
    for ax in axes.ravel()[len(picked):]:
        ax.axis("off")
    fig.suptitle("safety-free inference (map_only): %s" % " vs ".join(names), fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def _bars(arms, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [a["summary"]["arm"] for a in arms]
    panels = [
        ("valid rate", [a["summary"]["valid_rate"] for a in arms]),
        ("collision-free rate", [a["summary"]["collision_free_rate"] for a in arms]),
        ("mean collision fraction",
         [a["summary"]["collision_rate"].get("mean", 0.0) for a in arms]),
        ("chamfer vs GT (m)", [a["summary"]["chamfer_m"].get("mean", 0.0) for a in arms]),
        ("control RMSE (m)",
         [a["summary"]["control_rmse_m"].get("mean", 0.0) for a in arms]),
        ("length ratio", [a["summary"]["length_ratio"].get("mean", 0.0) for a in arms]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, (title, values) in zip(axes.ravel(), panels):
        bars = ax.bar(names, values, color=["#1f77b4", "#d62728", "#2ca02c", "#9467bd"][:len(names)])
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=8)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, "%.4g" % value,
                    ha="center", va="bottom", fontsize=8)
    fig.suptitle("safety-free inference (map_only)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    parser = base_parser("A/B: safety-trained vs never-safety-trained, both map_only")
    parser.add_argument("--arm", action="append", required=True,
                        help="name=checkpoint, repeatable (1..N)")
    parser.add_argument("--split", default="val")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--safety-mode", default="map_only",
                        choices=["map_only", "full", "no_alm"])
    parser.add_argument("--out", default="outputs/reports/ablation_safety_ablation.json")
    parser.add_argument("--figure-dir", default=FIGURE_DIR)
    parser.add_argument("--no-figures", action="store_true")
    opts = parser.parse_args()

    arms_in = []
    for spec in opts.arm:
        name, _, ckpt = spec.partition("=")
        if not ckpt or not os.path.exists(ckpt):
            print("missing checkpoint for arm %r: %r" % (name, ckpt))
            return 2
        arms_in.append((name, ckpt))
    if len(arms_in) < 1:
        print("give at least one --arm name=checkpoint")
        return 2

    cfg = load(opts)
    dataset = RealTrajectoryDataset(cfg, opts.split)
    indices = list(range(min(opts.samples, len(dataset))))
    print("split %s: %d samples, evaluating %d"
          % (opts.split, len(dataset), len(indices)))

    arms = []
    for name, ckpt in arms_in:
        print("[arm %s] %s" % (name, ckpt), flush=True)
        arms.append(run_arm(name, ckpt, cfg, dataset, indices, opts))
        print("  -> %s" % json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                     for k, v in arms[-1]["summary"].items()
                                     if k in ("valid_rate", "collision_free_rate",
                                              "seconds")}), flush=True)

    payload = {"split": opts.split, "safety_mode": opts.safety_mode,
               "steps": int(opts.steps), "seed": int(opts.seed),
               "samples": len(indices),
               "arms": [a["summary"] for a in arms],
               "paired": [dict(_paired(arms, k, other=j),
                               versus=arms[j]["summary"]["arm"])
                          for k in ("collision_rate", "chamfer_m", "length_error")
                          for j in range(1, len(arms))],
               "records": {a["summary"]["arm"]: a["records"] for a in arms}}
    out_path = os.path.join(PROJECT_ROOT, opts.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    if not opts.no_figures:
        os.makedirs(opts.figure_dir, exist_ok=True)
        print("figure:", _bars(arms, os.path.join(opts.figure_dir, "summary.png")))
        print("figure:", _montage(arms, dataset, payload["arms"],
                                  os.path.join(opts.figure_dir, "montage.png")))

    header = "%-12s %7s %7s %7s %7s %7s %7s" % (
        "arm", "valid", "free", "coll", "chamfer", "ctrl_rm", "len")
    print("\n" + header)
    for summary in payload["arms"]:
        print("%-12s %7.3f %7.3f %7.4f %7.3f %7.3f %7.3f" % (
            summary["arm"], summary["valid_rate"], summary["collision_free_rate"],
            summary["collision_rate"].get("mean", 0.0),
            summary["chamfer_m"].get("mean", 0.0),
            summary["control_rmse_m"].get("mean", 0.0),
            summary["length_ratio"].get("mean", 0.0)))
    for row in payload["paired"]:
        if row:
            print("paired %-14s %s better %d / %s better %d / tie %d (n=%d)"
                  % (row["metric"], payload["arms"][0]["arm"], row["first_better"],
                     row["versus"], row["second_better"], row["tie"],
                     row["n"]))
    if payload["arms"][0]["guide_chamfer_to_gt_m"]:
        print("\n%-12s %9s %9s %9s %8s" % ("arm (guidance)", "g_free", "g_cham_gt",
                                           "g_to_final", "homotopy"))
        for summary in payload["arms"]:
            print("%-12s %9.3f %9.3f %9.3f %8s" % (
                summary["arm"],
                1.0 - summary["guide_collision_rate"].get("mean", 0.0),
                summary["guide_chamfer_to_gt_m"].get("mean", 0.0),
                summary["guide_chamfer_to_final_m"].get("mean", 0.0),
                ("n/a" if summary["homotopy_witness_rate"] is None
                 else "%.3f" % summary["homotopy_witness_rate"])))
    print("written:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

