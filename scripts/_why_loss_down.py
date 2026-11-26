"""Why can the loss go down while the predictions are worse than a straight line?"""
import json
import sys

sys.path.insert(0, "src")
import numpy as np
import torch

from gp_diffuser.config import load_config
from gp_diffuser.factory import build_pipeline
from gp_diffuser.training.losses import collision_loss, control_loss
from gp_diffuser.training.real_dataset import RealTrajectoryDataset
from gp_diffuser.training.safety_cache import load_safety_cache

cfg = load_config("configs/c64_longpath.yaml")
h = json.load(open("outputs/reports/train_history.json", encoding="utf-8"))["history"]
keys = ["loss", "loss_u", "loss_c", "loss_align", "loss_collision", "loss_corridor"]
head = {k: float(np.mean([r[k] for r in h[:60]])) for k in keys}
tail = {k: float(np.mean([r[k] for r in h[-60:]])) for k in keys}
print("== 1. which term actually dropped (first 60 vs last 60 steps) ==")
for k in keys:
    print("  %-14s %8.3f -> %8.3f   delta %+8.3f  (%+.0f%% of the total drop)"
          % (k, head[k], tail[k], tail[k] - head[k],
             100 * (tail[k] - head[k]) / (tail["loss"] - head["loss"])))

pipe = build_pipeline(cfg, device="cuda", checkpoint="outputs/checkpoints/geopath_v1_c64.pt")
model, codec = pipe["model"], pipe["codec"]
ds = RealTrajectoryDataset(cfg, "val")
cache = load_safety_cache(cfg, "val", dataset=ds, verbose=False)
N = 64
occ = torch.stack([ds[i]["occupancy"] for i in range(N)]).cuda()
sdf = torch.stack([ds[i]["sdf"] for i in range(N)]).cuda()
sg = torch.stack([ds[i]["start_goal"] for i in range(N)]).cuda()
gt = torch.stack([ds[i]["controls_gt"] for i in range(N)]).cuda()
with torch.no_grad():
    f = pipe["map_encoder"].encode(occ, sdf)
    ctx = model.build_map_context(pipe["map_condition_encoder"](f["occ_bottleneck"],
                                                               f["sdf_bottleneck"]),
                                  sg, None, None)


def spread(x):
    """Mean per-sample RMS radius around the curve's own centroid (scene units)."""
    c = x.mean(dim=1, keepdim=True)
    return float(((x - c) ** 2).sum(dim=-1).mean().sqrt())


print("\n== 2. does the prediction look like GT, like noise, or like a blown-up curve? ==")
print("  %-6s %9s %9s %9s %9s %9s" % ("t", "MSE", "corr(p,q0)", "corr(p,q_t)",
                                     "spread(p)", "spread(q0)"))
model.eval()
torch.manual_seed(0)
for t_lo in (0, 250, 500, 950):
    mses, c0, ct, sp, sg0 = [], [], [], [], []
    with torch.no_grad():
        for s in range(0, N, 32):
            q0 = gt[s:s + 32]
            tt = torch.full((q0.shape[0],), t_lo, device="cuda")
            qq = pipe["schedule"].q_sample(q0, tt, torch.randn_like(q0)).float()
            p = model(q_t=qq, t=tt, map_context=ctx[s:s + 32])
            mses.append(((p - q0) ** 2).mean(dim=(1, 2)))
            a = (p * q0).mean(dim=(1, 2))
            b = (p * qq).mean(dim=(1, 2))
            c0.append(a / (p.pow(2).mean(dim=(1, 2)).sqrt() * q0.pow(2).mean(dim=(1, 2)).sqrt()))
            ct.append(b / (p.pow(2).mean(dim=(1, 2)).sqrt() * qq.pow(2).mean(dim=(1, 2)).sqrt()))
            sp.append([spread(x[None]) for x in p])
            sg0.append([spread(x[None]) for x in q0])
    print("  %-6d %9.4f %9.3f %9.3f %9.4f %9.4f"
          % (t_lo, float(torch.cat(mses).mean()), float(torch.cat(c0).mean()),
             float(torch.cat(ct).mean()), float(np.mean(sp)), float(np.mean(sg0))))

print("\n== 3. collision term: value and gradient along the way ==")
pred_curve = codec.decode_controls(model(q_t=pipe["schedule"].q_sample(
    gt, torch.full((N,), 500, device="cuda"), torch.randn_like(gt)).float(),
    t=torch.full((N,), 500, device="cuda"), map_context=ctx))
print("  L_collision(pred) %.4f m^2   (mse %.4f)" %
      (float(collision_loss(pred_curve, sdf, margin_m=0.3, scene_scale=32).mean()),
       float(control_loss(model(q_t=pipe["schedule"].q_sample(
           gt, torch.full((N,), 500, device="cuda"), torch.randn_like(gt)).float(),
           t=torch.full((N,), 500, device="cuda"), map_context=ctx), gt).mean())))
print("  L_collision(GT)   %.4f m^2" %
      float(collision_loss(codec.decode_controls(gt), sdf, margin_m=0.3,
                           scene_scale=32).mean()))
print("  |pred| mean %.4f vs |GT| mean %.4f  (scene units; ratio %.2fx)"
      % (float(pred_curve.abs().mean()), float(codec.decode_controls(gt).abs().mean()),
         float(pred_curve.abs().mean() / codec.decode_controls(gt).abs().mean())))
