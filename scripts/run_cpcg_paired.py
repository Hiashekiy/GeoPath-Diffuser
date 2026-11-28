"""Paired A/B/C/D experiment for CPCG (design document section 10, Phase C).

Everything except the correction mechanism is held fixed: same checkpoint, same
scene, same start/goal/heading, same initial noise, the same executed reverse
steps (8 by default = DDIM subsampling of the trained 1000-step table, no
retraining) and therefore the same search / ellipses / corridor pack (asserted
bitwise below).

    A  map_only            no safety conditions at all
    B  safety_only         safety cross-attention, no correction
    C  safety + CPCG       guidance on, terminal QP OFF (isolates the guidance)
    D  safety + CPCG + QP  the complete proposal

The acceptance criterion is NOT "D has zero violation" (that only proves the QP
works).  It is: CPCG must lower the corridor violation of the FOLLOW-UP clean
predictions, or at least reduce how much the terminal QP has to move the curve.

    python scripts/run_cpcg_paired.py --config configs/c64_longpath.yaml \
        --dataset real --split test --indices 1,4 \
        --checkpoint outputs/checkpoints/ab_distill_safeu_e100/d2_ep100.pt --device cuda
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
import yaml  # noqa: E402

from gp_diffuser.evaluation.metrics import (corridor_violation_m,
                                            trajectory_metrics)  # noqa: E402
from gp_diffuser.factory import build_pipeline  # noqa: E402
from gp_diffuser.training.dataset import SyntheticTrajectoryDataset  # noqa: E402
from gp_diffuser.training.real_dataset import RealTrajectoryDataset  # noqa: E402

GROUPS = {
    "A": {"name": "map_only", "safety_mode": "map_only",
          "guidance": None, "projection": None},
    "B": {"name": "safety_only", "safety_mode": "safety_only",
          "guidance": None, "projection": None},
    "C": {"name": "cpcg_no_projection", "safety_mode": "cpcg",
          "guidance": True, "projection": False},
    "D": {"name": "cpcg_full", "safety_mode": "cpcg",
          "guidance": True, "projection": True},
}


def _steps_series(trace, pack, transform, *, guided_key: str, offset: float) -> list:
    """Corridor violation (m) of the clean prediction of every recorded step."""
    out = []
    for entry in trace:
        curve = entry.get(guided_key)
        if curve is None:
            continue
        curve_m = transform.to_meters(np.asarray(curve[0], dtype=np.float64))
        out.append(float(corridor_violation_m(curve_m, pack,
                                              float(transform.scale),
                                              offset=offset)))
    return out


def yaml_load_value(text: str):
    """Parse a CLI value the way YAML would (float/int/bool, else string)."""
    return yaml.safe_load(text)


def _stats(values) -> dict:
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not values:
        return {"mean": None, "max": None, "min": None, "n": 0}
    return {"mean": float(np.mean(values)), "max": float(np.max(values)),
            "min": float(np.min(values)), "n": len(values)}


def _pack_equals(a, b) -> bool:
    """True when two runs froze the SAME sample pack (same arc length, same cells)."""
    if a is None or b is None:
        return a is None and b is None
    return (torch.equal(a.basis, b.basis)
            and torch.equal(a.sample_u, b.sample_u)
            and torch.equal(a.region_id, b.region_id)
            and torch.equal(a.sample_A, b.sample_A)
            and torch.equal(a.sample_b, b.sample_b)
            and torch.equal(a.face_mask, b.face_mask)
            and torch.equal(a.sample_mask, b.sample_mask))


def run_group(pipe, item, group: dict, steps: int, seed: int, res_m: float) -> dict:
    sampler = pipe["sampler"]
    transform = pipe["transform"]
    t0 = time.perf_counter()
    out = sampler.sample(
        item["occupancy"][None], item["sdf"][None], item["start_goal"][None],
        heading=item["heading"][None], heading_valid=item["heading_valid"][None],
        steps=steps, safety_mode=group["safety_mode"], seed=seed,
        record_curves=True,
        clean_guidance=group["guidance"], final_projection=group["projection"])
    elapsed = time.perf_counter() - t0

    traj_m = out["trajectory_m"][0].detach().cpu().numpy()
    metrics = trajectory_metrics(traj_m, item["occupancy"][0].numpy(),
                                 item["sdf"][0].numpy(), res_m)
    pack = out.get("pack")
    violations = _steps_series(out["steps_trace"], pack, transform,
                               guided_key="x0_guided_scene",
                               offset=transform.offset)
    guidance = out.get("guidance_stats") or []
    qp = out.get("final_qp") or {}
    return {
        "group": group["name"],
        "valid": bool(out["valid"]),
        "corridor_certified": bool(out.get("corridor_certified_all", False)),
        "collision_rate": float(metrics["collision_rate"]),
        "min_clearance_m": float(metrics["min_clearance_m"]),
        "length_m": float(metrics["length_m"]),
        "mean_curvature": float(metrics["mean_curvature"]),
        "max_curvature": float(metrics["max_curvature"]),
        "endpoint_error_scene": float(out["endpoint_error_scene"]),
        "heading_error_max_deg": (None if out.get("heading_error_deg") is None
                                  else float(out["heading_error_deg"].max())),
        "corridor_violation_full_m": float(corridor_violation_m(
            traj_m, pack, float(transform.scale), offset=transform.offset)),
        "corridor_violation_local_m": float(corridor_violation_m(
            traj_m, out.get("pack_local"), float(transform.scale),
            offset=transform.offset)),
        "step_violation_m": _stats(violations),
        "guidance_steps": int(out.get("guidance_steps", 0)),
        "guidance_max_violation_before_m": _stats(
            [g["max_violation_before"] * transform.scale for g in guidance]),
        "guidance_max_violation_after_m": _stats(
            [g["max_violation_after"] * transform.scale for g in guidance]),
        "guidance_delta_max_m": _stats([g["delta_max_m"] for g in guidance]),
        "guidance_delta_rms_scene": _stats([g["delta_rms_scene"] for g in guidance]),
        "final_qp_status": (qp.get("status") if qp else []),
        "final_qp_failure_reason": (None if not qp
                                    else qp.get("per_sample", [{}])[0].get(
                                        "failure_reason")),
        "endpoint_violation_scene": (None if not qp
                                     else qp.get("endpoint_violation_scene")),
        "final_qp_iterations": (qp.get("iterations") if qp else []),
        "final_qp_time_s": float(out.get("final_qp_time_s", 0.0)),
        "final_projection_correction_m": float(
            out.get("final_projection_correction_m", 0.0)),
        "final_corridor_violation_m": float(
            out.get("final_qp_max_violation_scene", 0.0) * transform.scale),
        # ---- sampled-constraint semantics (NOT a continuous certificate) ----
        "sampled_constraint_max_violation_m": float(
            out.get("sampled_constraint_max_violation", 0.0)) * transform.scale,
        "sampled_constraints_satisfied": bool(
            out.get("sampled_constraints_satisfied", False)),
        "sampled_region_coverage_ok": bool(
            out.get("sampled_region_coverage_ok", False)),
        "num_constraint_samples": int(out.get("num_constraint_samples", 0)),
        "num_constraint_rows": int(out.get("num_constraint_rows", 0)),
        "dense_collision_fraction": float(out.get("dense_collision_fraction", 0.0)),
        "dense_collision_free": bool(out.get("dense_collision_free", False)),
        "timing_s": {k: round(v, 4) for k, v in out["timing"].items()},
        "inference_s": elapsed,
        "trajectory_m": traj_m,
        "_pack": pack,
        "_out": out,
    }


def main() -> int:
    parser = base_parser("CPCG paired A/B/C/D experiment")
    parser.add_argument("--dataset", default="real", choices=["real", "synthetic"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--indices", default="1,4")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--groups", default="A,B,C,D")
    parser.add_argument("--set", action="append", default=[],
                        metavar="dotted.key=value",
                        help="config override, e.g. --set clean_guidance.step_size=0.2 "
                             "(repeatable; used for the step-size / trust-region sweep)")
    parser.add_argument("--out", default="outputs/reports/cpcg_paired.json")
    parser.add_argument("--no-figures", action="store_true",
                        help="skip the per-scene figure (sweeps over many scenes)")
    parser.add_argument("--tag", default="")
    opts = parser.parse_args()

    cfg = load(opts)
    for spec in opts.set:
        if "=" not in spec:
            raise SystemExit("--set expects dotted.key=value, got %r" % spec)
        key, value = spec.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        try:
            parsed = yaml_load_value(value)
        except Exception:
            parsed = value
        node[parts[-1]] = parsed
    pipe = build_pipeline(cfg, device=opts.device, checkpoint=opts.checkpoint)
    dataset = (RealTrajectoryDataset(cfg, opts.split) if opts.dataset == "real"
               else SyntheticTrajectoryDataset(cfg, opts.split))
    res_m = float(cfg["dataset"]["res_m"])
    indices = [int(s) for s in opts.indices.split(",") if s.strip()]
    groups = [GROUPS[g.strip().upper()] for g in opts.groups.split(",") if g.strip()]

    report = {"checkpoint": opts.checkpoint, "dataset": opts.dataset,
              "split": opts.split, "steps": int(opts.steps), "seed": int(opts.seed),
              "overrides": list(opts.set), "tag": opts.tag,
              "clean_guidance": dict(cfg.get("clean_guidance") or {}),
              "final_projection": {k: v for k, v in
                                   (cfg.get("final_projection") or {}).items()
                                   if k != "scene_to_meter"},
              "scene_scale": float(pipe["transform"].scale), "samples": []}
    for index in indices:
        item = dataset[index]
        print("== scene %d" % index, flush=True)
        per_group = {}
        for group in groups:
            result = run_group(pipe, item, group, opts.steps, opts.seed, res_m)
            per_group[group["name"]] = result
            step = result["step_violation_m"]
            print("   %-20s coll=%.4f viol(final)=%.3f m step_viol(mean/max)=%s/%s "
                  "cpcg=%d qp=%s corr=%.3f m %.2fs"
                  % (group["name"], result["collision_rate"],
                     result["corridor_violation_full_m"],
                     ("-" if step["mean"] is None else "%.3f" % step["mean"]),
                     ("-" if step["max"] is None else "%.3f" % step["max"]),
                     result["guidance_steps"],
                     (result["final_qp_status"][0]
                      if result["final_qp_status"] else "not_run"),
                     result["final_projection_correction_m"],
                     result["inference_s"]), flush=True)

        # the comparison is only clean if B/C/D shared ONE corridor pack
        packs = {k: v["_pack"] for k, v in per_group.items() if k != "map_only"}
        shared = None
        if packs:
            keys = list(packs)
            shared = all(_pack_equals(packs[keys[0]], packs[k]) for k in keys[1:])
        sample_report = {"index": int(index),
                         "shared_pack_across_conditioned_groups": bool(shared),
                         "groups": {}}
        for name, result in per_group.items():
            cleaned = {k: v for k, v in result.items()
                       if not k.startswith("_") and k != "trajectory_m"}
            sample_report["groups"][name] = cleaned
        report["samples"].append(sample_report)

        # figure: the four trajectories on one map
        if not opts.no_figures:
            try:
                _plot(pipe, item, per_group, index, opts)
            except Exception as error:                    # plotting is optional
                print("   (figure skipped: %s)" % error, flush=True)

    out_path = os.path.join(PROJECT_ROOT, opts.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    report["aggregate"] = _aggregate(report["samples"])
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=float, ensure_ascii=False)
    _write_markdown(report, out_path.replace(".json", ".md"))
    _print_aggregate(report["aggregate"])
    print("written:", out_path)
    return 0


def _aggregate(samples: list) -> dict:
    """Per-group means/counts over all scenes (the sample is the unit)."""
    names = sorted({n for s in samples for n in s["groups"]})
    out = {}
    for name in names:
        rows = [s["groups"][name] for s in samples if name in s["groups"]]
        if not rows:
            continue
        def mean(key, pred=lambda v: True):
            vals = [float(r[key]) for r in rows if r[key] is not None and pred(r[key])]
            return None if not vals else float(np.mean(vals))
        step_mean = [r["step_violation_m"]["mean"] for r in rows
                     if r["step_violation_m"]["mean"] is not None]
        step_max = [r["step_violation_m"]["max"] for r in rows
                    if r["step_violation_m"]["max"] is not None]
        out[name] = {
            "scenes": len(rows),
            "collision_rate": mean("collision_rate"),
            "collision_free_scenes": int(sum(1 for r in rows
                                             if r["collision_rate"] <= 1e-9)),
            "min_clearance_m": mean("min_clearance_m"),
            "step_violation_mean_m": (None if not step_mean
                                      else float(np.mean(step_mean))),
            "step_violation_max_m": (None if not step_max
                                     else float(np.mean(step_max))),
            "final_violation_m": mean("corridor_violation_full_m"),
            "guidance_steps_mean": mean("guidance_steps"),
            "qp_solved_scenes": int(sum(1 for r in rows
                                        if r["final_qp_status"]
                                        and r["final_qp_status"][0] == "solved")),
            "certified_scenes": int(sum(1 for r in rows
                                        if r.get("corridor_certified"))),
            "projection_correction_m": mean("final_projection_correction_m",
                                            lambda v: v > 0.0),
            "sampled_violation_m": mean("sampled_constraint_max_violation_m"),
            "sampled_satisfied_scenes": int(sum(
                1 for r in rows if r.get("sampled_constraints_satisfied"))),
            "coverage_ok_scenes": int(sum(
                1 for r in rows if r.get("sampled_region_coverage_ok"))),
            "dense_collision_free_scenes": int(sum(
                1 for r in rows if r.get("dense_collision_free"))),
            "constraint_rows_mean": mean("num_constraint_rows"),
            "inference_s": mean("inference_s"),
        }
    return out


def _print_aggregate(agg: dict) -> None:
    for name, row in agg.items():
        print("   [%s] scenes=%d collision=%.4f free=%d sampled_sat=%d coverage=%d "
              "step_viol mean/max=%s/%s "
              "final_viol=%s qp_solved=%d cert=%d corr=%s t=%.2fs"
              % (name, row["scenes"], row["collision_rate"],
                 row["collision_free_scenes"],
                 row["sampled_satisfied_scenes"], row["coverage_ok_scenes"],
                 "-" if row["step_violation_mean_m"] is None
                 else "%.3f" % row["step_violation_mean_m"],
                 "-" if row["step_violation_max_m"] is None
                 else "%.3f" % row["step_violation_max_m"],
                 "-" if row["final_violation_m"] is None
                 else "%.3f" % row["final_violation_m"],
                 row["qp_solved_scenes"], row["certified_scenes"],
                 "-" if row["projection_correction_m"] is None
                 else "%.3f" % row["projection_correction_m"],
                 row["inference_s"]), flush=True)


def _plot(pipe, item, per_group, index, opts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    transform = pipe["transform"]
    scene = item["occupancy"][0].numpy()
    extent = scene.shape[-1] * float(pipe["cfg"]["dataset"]["res_m"])
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.2), constrained_layout=True)
    colours = {"map_only": "#1f77b4", "safety_only": "#ff7f0e",
               "cpcg_no_projection": "#2ca02c", "cpcg_full": "#d62728"}

    ax = axes[0]
    ax.imshow(scene, origin="lower", cmap="Greys", alpha=0.85,
              extent=(0.0, extent, 0.0, extent))
    ax.set_xlim(0, extent)
    ax.set_ylim(0, extent)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("A/B/C/D trajectories (scene %d, %d steps)" % (index, opts.steps))
    for name, result in per_group.items():
        traj = result["trajectory_m"]
        ax.plot(traj[:, 0], traj[:, 1], color=colours.get(name, "k"), lw=1.9,
                label=name)
    ax.legend(loc="lower left", fontsize=8)

    ax = axes[1]
    ax.set_title("per-step clean-prediction corridor violation (m)")
    for name, result in per_group.items():
        out = result["_out"]
        series = _steps_series(out["steps_trace"], result["_pack"], transform,
                              guided_key="x0_guided_scene",
                              offset=transform.offset)
        if series:
            ax.plot(np.arange(len(series)), series, color=colours.get(name, "k"),
                    lw=1.1, label=name)
    ax.set_xlabel("reverse forward index")
    ax.set_ylabel("max half-space violation (m)")
    ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    path = os.path.join(PROJECT_ROOT, "outputs", "figures", "cpcg_paired",
                        "scene%03d.png" % index)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print("   figure:", os.path.relpath(path, PROJECT_ROOT), flush=True)


def _write_markdown(report: dict, path: str) -> None:
    lines = ["# CPCG paired A/B/C/D experiment", "",
             "- checkpoint: `%s`" % report["checkpoint"],
             "- dataset: %s / %s, steps = %d, seed = %d"
             % (report["dataset"], report["split"], report["steps"], report["seed"]),
             "- overrides: %s" % (", ".join(report.get("overrides") or []) or "none"),
             ""]
    agg = report.get("aggregate") or {}
    if agg:
        lines += ["## Aggregate over scenes", "",
                  "| group | scenes | collision | collision-free | step viol mean/max (m) "
                  "| final viol (m) | CPCG steps | QP solved | certified | QP corr (m) | s |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, row in agg.items():
            lines.append("| %s | %d | %.4f | %d | %s / %s | %s | %s | %d | %d | %s | %.2f |"
                         % (name, row["scenes"], row["collision_rate"],
                            row["collision_free_scenes"],
                            "-" if row["step_violation_mean_m"] is None
                            else "%.3f" % row["step_violation_mean_m"],
                            "-" if row["step_violation_max_m"] is None
                            else "%.3f" % row["step_violation_max_m"],
                            "-" if row["final_violation_m"] is None
                            else "%.3f" % row["final_violation_m"],
                            "-" if row["guidance_steps_mean"] is None
                            else "%.1f" % row["guidance_steps_mean"],
                            row["qp_solved_scenes"], row["certified_scenes"],
                            "-" if row["projection_correction_m"] is None
                            else "%.3f" % row["projection_correction_m"],
                            row["inference_s"]))
        lines.append("")
    lines += ["## Per scene", "",
              "| scene | group | collision | min clearance (m) | step violation mean/max (m) "
              "| final violation (m) | CPCG steps | QP | QP correction (m) | cert |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for sample in report["samples"]:
        for name, row in sample["groups"].items():
            step = row["step_violation_m"]
            lines.append("| %d | %s | %.4f | %.3f | %s / %s | %.3f | %d | %s | %.3f | %s |"
                         % (sample["index"], name, row["collision_rate"],
                            row["min_clearance_m"],
                            "-" if step["mean"] is None else "%.3f" % step["mean"],
                            "-" if step["max"] is None else "%.3f" % step["max"],
                            row["corridor_violation_full_m"], row["guidance_steps"],
                            (row["final_qp_status"][0]
                             if row["final_qp_status"] else "not_run"),
                            row["final_projection_correction_m"],
                            row["corridor_certified"]))
        lines.append("| %d | shared pack across B/C/D | %s | | | | | | | |"
                     % (sample["index"],
                        sample["shared_pack_across_conditioned_groups"]))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
