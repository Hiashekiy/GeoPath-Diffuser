"""Time one full inference run of the k4p samples (8 DDIM reverse steps).

``sampling.steps = 8`` means ``pick_times(1000, 8)`` -> 8 executed reverse steps,
i.e. exactly 8 denoiser forwards per sample (map encoding and the safety geometry
are built ONCE and reused).  This script measures that end to end through
``ConditionalDiTSampler.sample`` for the three modes.

Weights are random (no checkpoint exists) -- timings are representative, the
trajectories are not.

Usage::

    python scripts/bench_k4p_sample.py --device cuda --samples 4 --repeat 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bootstrap import PROJECT_ROOT, base_parser, load  # noqa: E402

import torch  # noqa: E402

from gp_diffuser.diffusion.sampler_conditional_dit import pick_times  # noqa: E402
from gp_diffuser.factory import build_pipeline  # noqa: E402
from gp_diffuser.training.real_dataset import RealTrajectoryDataset  # noqa: E402


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def main() -> int:
    parser = base_parser("k4p: inference wall clock (8 reverse steps)")
    parser.add_argument("--data-root", default=os.path.join(PROJECT_ROOT, "data", "_demo_k4p"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--steps", type=int, default=None,
                        help="override sampling.steps (default: the config value)")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--modes", default="map_only,safety_only,cpcg")
    parser.add_argument("--out", default=os.path.join(PROJECT_ROOT, "outputs", "reports",
                                                      "bench_k4p_sample.json"))
    opts = parser.parse_args()

    cfg = load(opts)
    cfg["dataset"]["res_m"] = 0.625
    cfg["dataset"]["grid"] = 256
    cfg["dataset"]["real_root"] = opts.data_root
    device = opts.device
    pipe = build_pipeline(cfg, device=device, seed=0)
    dataset = RealTrajectoryDataset(cfg, opts.split, limit=opts.samples)
    items = [dataset[i] for i in range(opts.samples)]
    occupancy = torch.stack([it["occupancy"] for it in items])
    sdf = torch.stack([it["sdf"] for it in items])
    cond = torch.stack([it["start_goal"] for it in items])
    heading = torch.stack([it["heading"] for it in items])
    heading_valid = torch.stack([it["heading_valid"] for it in items])

    steps = int(opts.steps or cfg["sampling"]["steps"])
    times = pick_times(int(cfg["diffusion"]["train_diffusion_steps"]), steps)
    if times is None:      # steps >= train_diffusion_steps -> the full table
        times = list(range(int(cfg["diffusion"]["train_diffusion_steps"])))
    print("train_diffusion_steps=%d  sampling.steps=%d -> pick_times -> %s (%d forwards)"
          % (int(cfg["diffusion"]["train_diffusion_steps"]), steps, times, len(times)),
          flush=True)

    report = {"device": device, "samples": opts.samples, "steps": len(times),
              "timesteps": times, "modes": {}}
    for mode in [m.strip() for m in opts.modes.split(",") if m.strip()]:
        pipe["sampler"].sample(occupancy, sdf, cond, heading, heading_valid,
                               steps=steps, safety_mode=mode, seed=0)   # warmup
        sync(device)
        t0 = time.perf_counter()
        for _ in range(int(opts.repeat)):
            out = pipe["sampler"].sample(occupancy, sdf, cond, heading, heading_valid,
                                         steps=steps, safety_mode=mode, seed=0)
        sync(device)
        total = (time.perf_counter() - t0) / int(opts.repeat)
        timing = {k: round(float(v), 4) for k, v in (out.get("timing") or {}).items()
                  if isinstance(v, (int, float))}
        per_step = out.get("frames") or {}
        report["modes"][mode] = {
            "total_s_per_batch": round(total, 4),
            "total_ms_per_sample": round(1000.0 * total / opts.samples, 2),
            "timing": timing,
            "trace_len": len(per_step.get("controls") or []),
        }
        print("%-12s %6.2f ms/sample  (batch %.3f s)  %s"
              % (mode, 1000.0 * total / opts.samples, total, json.dumps(timing)), flush=True)

    peak = torch.cuda.max_memory_allocated() / 1e9 if str(device).startswith("cuda") else None
    report["peak_gpu_memory_gb"] = round(peak, 2) if peak else None
    os.makedirs(os.path.dirname(os.path.abspath(opts.out)), exist_ok=True)
    with open(opts.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    print("written:", opts.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
