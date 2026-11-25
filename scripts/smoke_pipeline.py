"""Minimal end-to-end verification of the whole pipeline on ONE sample.

Prints a checklist for every stage the design document requires (section 6,
stage 4 / section 13 acceptance):

    map encoding (once) -> map memory -> map-only trajectory -> guidance search
    -> PathSafe ellipses -> local convex corridor + constraint pack -> ALM
    -> DDIM finish -> dense validation

Exit code 0 means every check passed.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import PROJECT_ROOT, base_parser, load  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from gp_diffuser.evaluation.metrics import trajectory_metrics  # noqa: E402
from gp_diffuser.evaluation.visualize import render_sample  # noqa: E402
from gp_diffuser.factory import build_pipeline, describe  # noqa: E402
from gp_diffuser.training.dataset import SyntheticTrajectoryDataset  # noqa: E402


def main() -> int:
    parser = base_parser("end-to-end smoke check")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--figure", default="outputs/figures/smoke_pipeline.png")
    opts = parser.parse_args()

    cfg = load(opts)
    pipe = build_pipeline(cfg, device=opts.device)
    checks = []

    def check(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": str(detail)})
        print(("  [PASS] " if ok else "  [FAIL] ") + name + (" :: " + str(detail) if detail else ""))

    print("== pipeline summary")
    print(json.dumps(describe(pipe), indent=2))

    dataset = SyntheticTrajectoryDataset(cfg, "test")
    item = dataset[opts.index]
    scene = dataset.raw_scene(opts.index)
    occ = item["occupancy"][None]
    sdf = item["sdf"][None]
    sg = item["start_goal"][None]

    print("== stage 0: frozen map encoding")
    net = pipe["net"]
    feats = pipe["map_encoder"].encode(occ, sdf)
    check("occupancy bottleneck is 32x32", tuple(feats["occ_bottleneck"].shape[-2:]) == (32, 32),
          tuple(feats["occ_bottleneck"].shape))
    check("bottleneck channels == 192", feats["occ_bottleneck"].shape[1] == 192,
          feats["occ_bottleneck"].shape[1])
    check("full feature is 256x256", tuple(feats["occ_full"].shape[-2:]) == (256, 256),
          tuple(feats["occ_full"].shape))
    check("all PathSafe parameters frozen",
          all(not p.requires_grad for p in net.parameters()))

    mem = pipe["map_projection"](feats["occ_bottleneck"], feats["sdf_bottleneck"])
    check("map memory is 1024 x 384", tuple(mem.shape[1:]) == (1024, 384), tuple(mem.shape))

    print("== stage 1-4: full sampling run")
    out = pipe["sampler"].sample(occ, sdf, sg, heading=item["heading"][None],
                                 heading_valid=item["heading_valid"][None],
                                 steps=opts.steps, safety_mode="full", seed=0)
    check("safety geometry generated exactly once", out["frames"]["safety_generated_once"])
    check("first forwards are map-only, later ones safety-conditioned",
          out["steps_trace"][0]["branch"] == "map_only"
          and out["steps_trace"][-1]["branch"] == "safety")
    check("one guided search for the sample", len(out["search_stats"]) == 1,
          out["search_stats"])
    check("search produced a feasible path",
          bool(out["search_stats"][0]["valid"]), out["search_stats"][0]["status"])
    check("64 ellipses generated", out["ellipses"] is not None and out["ellipses"].shape[1] == 64,
          None if out["ellipses"] is None else tuple(out["ellipses"].shape))
    check("safety tokens are 128 path + 64 ellipse",
          out["safety_condition"] is not None
          and out["safety_condition"]["path"].shape[1] == 128
          and out["safety_condition"]["ellipse"].shape[1] == 64)
    info = out["corridor_info"] or {}
    check("corridor pack built", out["pack"] is not None)
    check("corridor scope is local (collision intervals only)",
          info.get("scope") == "local", info.get("spans"))
    check("ALM executed on the frozen pack", len(out["alm_stats"]) > 0,
          "%d steps" % len(out["alm_stats"]))
    check("ALM never fed back to the network (no feedback key in the trace)",
          all("feedback" not in step for step in out["steps_trace"]))

    traj_m = out["trajectory_m"][0].detach().cpu().numpy()
    metrics = trajectory_metrics(traj_m, scene["occupancy"], scene["sdf"],
                                 float(cfg["dataset"]["res_m"]))
    check("endpoints match start/goal",
          float(np.abs(traj_m[0] - scene["start_m"]).max()) < 1e-3
          and float(np.abs(traj_m[-1] - scene["goal_m"]).max()) < 1e-3,
          "start err %.4f, goal err %.4f" % (np.abs(traj_m[0] - scene["start_m"]).max(),
                                             np.abs(traj_m[-1] - scene["goal_m"]).max()))
    print("== metrics on this sample")
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v
                      for k, v in metrics.items()}, indent=2))
    print("== timings")
    print(json.dumps({k: round(v, 4) for k, v in out["timing"].items()}, indent=2))

    figure = os.path.join(PROJECT_ROOT, opts.figure)
    sample = {"occupancy_m": scene["occupancy"], "res_m": float(cfg["dataset"]["res_m"]),
              "transform": pipe["transform"],
              "gt_path_scene": item["path_gt_scene"].numpy(),
              "start_goal_scene": item["start_goal"].numpy(),
              "scene_scale": pipe["transform"].scale}
    render_sample(out, sample, path=figure,
                  title="smoke pipeline | index=%d | steps=%d" % (opts.index, opts.steps))
    print("figure:", figure)

    ok = all(c["ok"] for c in checks)
    os.makedirs(os.path.join(PROJECT_ROOT, "outputs", "reports"), exist_ok=True)
    with open(os.path.join(PROJECT_ROOT, "outputs", "reports", "smoke_checks.json"),
              "w", encoding="utf-8") as fh:
        json.dump({"checks": checks, "metrics": metrics,
                   "timing": out["timing"]}, fh, indent=2, default=float)
    print("\n%d/%d checks passed" % (sum(c["ok"] for c in checks), len(checks)))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
