"""Time ONE denoising step of the Conditional DiT - network forward only.

Measures exactly one thing: ``model(q_t, t, map_context[, safety_memory,
safety_mask])`` for a **single sample** (B = 1).  No batching (batching would
amortise kernel launch cost and fake a speed-up), no boundary decoder, no ALM
correction, no convex-region / corridor construction.

Two variants are timed, because the map context and the ellipse memory never
change during the reverse process:

  raw  K/V projected inside every forward (12 x per step)
  kv   K/V projected once per run (``build_map_kv`` / ``build_safety_kv``)

The variants are interleaved round-robin and each call is timed with CUDA
events, so clock drift / other GPU load hits all of them equally.

    python scripts/bench_denoise_step.py --device cuda --reps 200
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

import torch

from gp_diffuser.config import load_config
from gp_diffuser.factory import build_map_condition_encoder, build_model


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def measure(variants: dict, reps: int, warmup: int, device: str,
            rounds: int = 3) -> dict:
    """Round-robin timing of single calls, never a batch of calls.

    ``amortised`` = ``reps`` calls back to back with one sync (steady-state cost
    per call, removes the per-call sync that a 200-kernel B=1 forward would
    otherwise pay); ``latency`` = one call + sync, median over ``reps`` calls.
    Each round times every variant, so drift/background load hits all equally.
    """
    cuda = device.startswith("cuda")
    out = {k: {"amortised": [], "latency": []} for k in variants}

    def one_round():
        torch.cuda.synchronize() if cuda else None
        for key, fn in variants.items():
            if cuda:
                t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
                t0.record()
                for _ in range(reps):
                    fn()
                t1.record()
                torch.cuda.synchronize()
                out[key]["amortised"].append(t0.elapsed_time(t1) / reps)
            else:
                t_start = time.perf_counter()
                for _ in range(reps):
                    fn()
                out[key]["amortised"].append((time.perf_counter() - t_start) * 1e3 / reps)
        for key, fn in variants.items():
            samples = []
            if cuda:
                starts = [torch.cuda.Event(True) for _ in range(reps)]
                ends = [torch.cuda.Event(True) for _ in range(reps)]
                for i in range(reps):
                    starts[i].record()
                    fn()
                    ends[i].record()
                torch.cuda.synchronize()
                samples = [s.elapsed_time(e) for s, e in zip(starts, ends)]
            else:
                for _ in range(reps):
                    t0 = time.perf_counter()
                    fn()
                    samples.append((time.perf_counter() - t0) * 1e3)
            out[key]["latency"].append(statistics.median(samples))

    with torch.inference_mode():
        for _ in range(warmup):
            for fn in variants.values():
                fn()
        for _ in range(rounds):
            one_round()
    stats = {}
    for key, d in out.items():
        stats[key] = {"amortised": statistics.median(d["amortised"]),
                      "amortised_min": min(d["amortised"]),
                      "latency": statistics.median(d["latency"]),
                      "rounds": d["latency"]}
    return stats


def main() -> int:
    parser = _bootstrap.base_parser("Time one Conditional-DiT denoising forward")
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--batch", type=int, default=1,
                        help="keep at 1: batching changes the per-sample cost")
    args = parser.parse_args()

    cfg = _bootstrap.load(args)
    device = args.device
    torch.manual_seed(0)

    model = build_model(cfg).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    build_map_condition_encoder(cfg)          # keeps the config honest
    info = model.config_dict()

    B = int(args.batch)
    C = model.num_controls
    D = model.hidden_dim
    map_memory = torch.randn(B, model.map_tokens, D, device=device)
    start_goal = torch.randn(B, 2, 2, device=device)
    q = torch.randn(B, C, 2, device=device)
    t = torch.full((B,), 500, device=device, dtype=torch.long)
    ellipse = torch.randn(B, model.num_ellipses, 6, device=device)
    ellipse_valid = torch.ones(B, model.num_ellipses, dtype=torch.bool, device=device)

    map_context = model.build_map_context(map_memory, start_goal)
    safety_memory, safety_mask = model.build_safety_memory(ellipse, ellipse_valid)
    map_kv = model.build_map_kv(map_context)
    safety_kv = model.build_safety_kv(safety_memory)

    variants = {
        "map_only": lambda: model(q_t=q, t=t, map_context=map_context),
        "map_only_kv": lambda: model(q_t=q, t=t, map_kv=map_kv),
        "safety": lambda: model(q_t=q, t=t, map_context=map_context,
                                safety_memory=safety_memory, safety_mask=safety_mask),
        "safety_kv": lambda: model(q_t=q, t=t, map_kv=map_kv, safety_kv=safety_kv,
                                   safety_mask=safety_mask),
    }
    results = measure(variants, args.reps, args.warmup, device)

    print("device            : %s" % device)
    if device.startswith("cuda"):
        print("gpu               : %s" % torch.cuda.get_device_name(0))
    print("torch             : %s" % torch.__version__)
    print("architecture      : %s" % info["architecture_version"])
    print("diffusion         : %s" % {k: info[k] for k in
                                      ("backbone", "variant", "hidden_dim", "depth",
                                       "num_heads", "mlp_ratio", "parameters")})
    print("map_context tokens: %d (+%d ellipse keys, query = %d control tokens)"
          % (model.map_tokens + 2, model.num_ellipses, C))
    print("batch             : %d (no batching)" % B)
    print("reps              : %d/round, %d rounds (warmup %d)"
          % (args.reps, 3, args.warmup))
    print("")
    print("%-12s %10s %10s %10s" % ("variant", "amortised", "amort.min", "latency"))
    for key, st in results.items():
        print("%-12s %10.2f %10.2f %10.2f  ms"
              % (key, st["amortised"], st["amortised_min"], st["latency"]))

    # whole-run extrapolation for the configured schedule: the safety geometry is
    # generated at the mid forward, so half the steps are map-only (no ALM, no
    # search / ellipse / corridor - those happen once and are excluded here)
    steps = int((cfg.get("sampling") or {}).get("steps", 50))
    mid = float((cfg.get("sampling") or {}).get("mid_forward_fraction", 0.5))
    k = int(round(mid * (steps - 1)))
    n_safety = steps - k - 1
    print("")
    print("schedule          : %d steps, safety geometry generated at forward #%d"
          % (steps, k))
    for tag, a, b in (("raw", "map_only", "safety"), ("kv ", "map_only_kv", "safety_kv")):
        one, two = results[a]["amortised"], results[b]["amortised"]
        print("network total [%s]: %.1f ms  (%d map-only x %.2f + %d safety x %.2f)"
              % (tag, (k + 1) * one + n_safety * two, k + 1, one, n_safety, two))
    return 0


if __name__ == "__main__":
    sys.exit(main())
