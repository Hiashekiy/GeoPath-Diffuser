"""Probe: does the Map-only branch actually move towards the safety branch?

Teacher-forced (no sampling, no ALM): one forward per branch at the same
``q_t / t``, then compare, in metres,

    gap   = RMS(u - c)     the quantity ``L_align`` measures
    err_u = RMS(u - gt)
    err_c = RMS(c - gt)

plus the collision rate of the decoded curves and the projection coefficient
``k`` of ``(u - gt)`` onto ``(c - gt)`` (k = 1: u ignores c, k = 0: u copies c).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/src")

import numpy as np
import torch

from gp_diffuser.evaluation.metrics import trajectory_metrics
from gp_diffuser.factory import build_pipeline
from gp_diffuser.training.real_dataset import RealTrajectoryDataset
from gp_diffuser.training.safety_cache import load_safety_cache

CKPT = "outputs/checkpoints/ab_dual/last.pt"
SPLIT = "val"
BATCH = 16
SAMPLES = 64
TIMESTEPS = (250, 500)


def rms_m(a, b, scale):
    return float(torch.sqrt(((a - b) ** 2).mean()).item()) * scale


def main() -> int:
    torch.manual_seed(0)
    cfg = __import__("gp_diffuser.config", fromlist=["load_config"]).load_config(
        "configs/c64_longpath.yaml")
    pipe = build_pipeline(cfg, device="cuda", checkpoint=CKPT)
    dataset = RealTrajectoryDataset(cfg, SPLIT)
    cache = load_safety_cache(cfg, SPLIT, dataset=dataset)
    model, codec, sched = pipe["model"], pipe["codec"], pipe["schedule"]
    scale = float(pipe["transform"].scale)
    boundary = pipe["sampler"].boundary
    model.eval()

    mask = cache.get("ellipse_mask")
    if mask is not None:
        per = np.asarray(mask).reshape(len(mask), -1).sum(axis=1)
        print("cache ellipse_mask: samples with a valid ellipse %d/%d, mean valid %d"
              % (int((per > 0).sum()), len(per), int(per.mean())))
    else:
        print("cache has no ellipse_mask -> every ellipse counted as valid")

    for t_scalar in TIMESTEPS:
        agg = {k: [] for k in ("gap", "err_u", "err_c", "cgap", "cerr_u", "cerr_c",
                               "k", "coll_u", "coll_c")}
        for begin in range(0, min(SAMPLES, len(dataset)), BATCH):
            idx = np.arange(begin, min(begin + BATCH, len(dataset)))
            items = [dataset[int(i)] for i in idx]
            occ = torch.stack([it["occupancy"] for it in items]).cuda()
            sdf = torch.stack([it["sdf"] for it in items]).cuda()
            cond = torch.stack([it["start_goal"] for it in items]).cuda()
            heading = torch.stack([it["heading"] for it in items]).cuda()
            hv = torch.stack([it["heading_valid"] for it in items]).cuda()
            q0 = torch.stack([it["controls_gt"] for it in items]).cuda()
            ellipse = torch.as_tensor(cache["ellipse_scene"][idx], dtype=torch.float32,
                                      device="cuda")
            valid = (torch.ones(ellipse.shape[0], ellipse.shape[1], dtype=torch.bool,
                                device="cuda") if mask is None else
                     torch.as_tensor(np.asarray(mask)[idx], dtype=torch.bool, device="cuda"))
            with torch.no_grad():
                feats = pipe["map_encoder"].encode(occ, sdf)
                memory = pipe["map_condition_encoder"](feats["occ_bottleneck"],
                                                       feats["sdf_bottleneck"])
                ctx = model.build_map_context(memory, cond, heading, hv)
                safety_memory, safety_mask = model.build_safety_memory(ellipse, valid)
                T = int(sched.num_timesteps)
                t = torch.full((q0.shape[0],), int(t_scalar), device="cuda", dtype=torch.long)
                q_t = sched.q_sample(q0, t, torch.randn_like(q0)).float()
                raw_u = model(q_t=q_t, t=t, map_context=ctx)
                raw_c = model(q_t=q_t, t=t, map_context=ctx,
                              safety_memory=safety_memory, safety_mask=safety_mask)
                u = boundary(raw_u, cond, heading, hv)
                c = boundary(raw_c, cond, heading, hv)
                basis = codec.basis.to("cuda")
                curve_u = torch.einsum("hk,bkd->bhd", basis, u)
                curve_c = torch.einsum("hk,bkd->bhd", basis, c)
                curve_gt = torch.einsum("hk,bkd->bhd", basis, q0)
                du, dc = u - q0, c - q0
                k = float((du * dc).sum() / dc.square().sum().clamp_min(1e-12))
                agg["gap"].append(rms_m(u, c, scale))
                agg["err_u"].append(rms_m(u, q0, scale))
                agg["err_c"].append(rms_m(c, q0, scale))
                agg["cgap"].append(rms_m(curve_u, curve_c, scale))
                agg["cerr_u"].append(rms_m(curve_u, curve_gt, scale))
                agg["cerr_c"].append(rms_m(curve_c, curve_gt, scale))
                agg["k"].append(k)
                for tag, curve in (("coll_u", curve_u), ("coll_c", curve_c)):
                    pts = pipe["transform"].to_meters(curve.cpu().numpy())
                    for b in range(pts.shape[0]):
                        m = trajectory_metrics(pts[b], items[b]["occupancy"][0].numpy(),
                                               items[b]["sdf"][0].numpy(), dataset.res_m)
                        agg[tag].append(float(m["collision_rate"]))

        print("t=%3d | controls gap %.2f err_u %.2f err_c %.2f m "
              "| curve gap %.2f err_u %.2f err_c %.2f m | k=%.2f "
              "| coll_u %.3f coll_c %.3f"
              % (t_scalar, np.mean(agg["gap"]), np.mean(agg["err_u"]),
                 np.mean(agg["err_c"]), np.mean(agg["cgap"]),
                 np.mean(agg["cerr_u"]), np.mean(agg["cerr_c"]), np.mean(agg["k"]),
                 np.mean(agg["coll_u"]), np.mean(agg["coll_c"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
