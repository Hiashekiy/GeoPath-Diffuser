"""Read-only verification of the review claims (no parameter update, no training).

A. do the GT trajectories themselves satisfy the training-time safety terms?
B. what is the gradient norm of each loss term at one fixed batch / seed?
C. how well does the model regress x0 per noise level, vs the straight line?
"""
import sys

sys.path.insert(0, "src")
import numpy as np
import torch

from gp_diffuser.config import load_config
from gp_diffuser.evaluation.metrics import sample_sdf
from gp_diffuser.factory import build_pipeline
from gp_diffuser.safety.local_regions import pack_from_tables
from gp_diffuser.training.dual_branch_trainer import DualBranchTrainer
from gp_diffuser.training.losses import (alignment_loss, collision_loss, control_loss,
                                         corridor_loss)
from gp_diffuser.training.real_dataset import RealTrajectoryDataset
from gp_diffuser.training.safety_cache import load_safety_cache

cfg = load_config("configs/c64_longpath.yaml")
pipe = build_pipeline(cfg, device="cuda", checkpoint="outputs/checkpoints/geopath_v1_c64.pt")
model, codec, enc = pipe["model"], pipe["codec"], pipe["map_condition_encoder"]
ds = RealTrajectoryDataset(cfg, "val")
cache = load_safety_cache(cfg, "val", dataset=ds, verbose=False)
trainer = DualBranchTrainer(cfg, model, enc, pipe["map_encoder"], codec,
                            pipe["schedule"], device="cuda",
                            scene_scale=pipe["transform"].scale)
N = 128
occ = torch.stack([ds[i]["occupancy"] for i in range(N)]).cuda()
sdf = torch.stack([ds[i]["sdf"] for i in range(N)]).cuda()
sg = torch.stack([ds[i]["start_goal"] for i in range(N)]).cuda()
gt = torch.stack([ds[i]["controls_gt"] for i in range(N)]).cuda()
gt_scene = torch.stack([ds[i]["path_gt_scene"] for i in range(N)]).cuda()
head = torch.stack([ds[i]["heading"] for i in range(N)]).cuda()
hvalid = torch.stack([ds[i]["heading_valid"] for i in range(N)]).cuda()
with torch.no_grad():
    feats = pipe["map_encoder"].encode(occ, sdf)
    mem = enc(feats["occ_bottleneck"], feats["sdf_bottleneck"])
    ctx = model.build_map_context(mem, sg, head, hvalid)

print("== A. GT under the training-time safety terms (128 val samples) ==")
for margin in (0.0, 0.05, 0.1, 0.3):
    loss = collision_loss(gt_scene, sdf, margin_m=margin, scene_scale=32.0)
    print("  L_collision(Q_GT, margin %.2f m): mean %.5f m^2  frac>0 %.2f  RMSE %.3f m"
          % (margin, float(loss.mean()), float((loss > 1e-12).float().mean()),
             float(loss.mean()) ** 0.5))

corr_vals = []
for i in range(N):
    _, tables = trainer._safety_conditions(cache, np.array([i]))
    pack = pack_from_tables(codec, tables, include_box=False, device="cuda")
    if pack is None or int(pack.piece_mask.sum()) == 0:
        continue
    with torch.no_grad():
        corr_vals.append(float(corridor_loss(gt[i:i + 1], pack).item()))
corr_vals = np.asarray(corr_vals)
print("  L_corridor(Q_GT, local cells): mean %.5f scene (%.3f m)  frac>0 %.2f  cells>0 in %d/%d"
      % (corr_vals.mean(), corr_vals.mean() * 32.0, float((corr_vals > 0).mean()),
         len(corr_vals), N))

clear_m = np.array([float(sample_sdf(ds[i]["sdf"][0].numpy(), ds[i]["path_gt_m"].numpy(), 0.25).min())
                    for i in range(N)])
print("  GT clearance: min %.2f  p1 %.2f  p50 %.2f  frac<0.30 m %.2f  frac<0.64 m %.2f"
      % (clear_m.min(), np.percentile(clear_m, 1), np.median(clear_m),
         float((clear_m < 0.30).mean()), float((clear_m < 0.64).mean())))

print("\n== B. gradient norm per loss term (one batch, one seed) ==")
torch.manual_seed(0)
B = 4
t = torch.randint(0, int(pipe["schedule"].num_timesteps), (B,), device="cuda")
eps = torch.randn_like(gt[:B])
q_t = pipe["schedule"].q_sample(gt[:B], t, eps).float()
pu = model(q_t=q_t, t=t, map_context=ctx[:B])
pc = model(q_t=q_t, t=t, map_context=ctx[:B])
_, tables_b = trainer._safety_conditions(cache, np.arange(B))
pack_b = pack_from_tables(codec, tables_b, include_box=False, device="cuda")
terms = {
    "L_u": control_loss(pu, gt[:B]).mean(),
    "L_c": control_loss(pc, gt[:B]).mean(),
    "L_align*0.5": 0.5 * alignment_loss(pu, pc).mean(),
    "L_collision": collision_loss(codec.decode_controls(pu), sdf[:B], margin_m=0.3,
                                  scene_scale=32.0).mean(),
    "L_corridor": corridor_loss(pc, pack_b).mean(),
}
dit = [p for p in model.parameters() if p.requires_grad]
for name, value in terms.items():
    model.zero_grad(set_to_none=True)
    value.backward(retain_graph=True)
    norm = torch.sqrt(sum((p.grad ** 2).sum() for p in dit if p.grad is not None))
    print("  %-13s value %9.4f   ||grad|| %11.4f" % (name, float(value), float(norm)))
model.zero_grad(set_to_none=True)

print("\n== C. x0 regression per noise level (128 val samples) ==")
C = codec.num_controls
w = torch.linspace(0, 1, C, device="cuda")[None, :, None]
straight = sg[:, 0:1] + w * (sg[:, 1:2] - sg[:, 0:1])
base = float(((straight - gt) ** 2).mean())
print("  straight-line baseline: MSE %.5f  RMSE %.1f m" % (base, base ** 0.5 * 32.0))
model.eval()
for t_lo in (0, 100, 250, 500, 750, 950):
    errs = []
    with torch.no_grad():
        for s in range(0, N, 32):
            q0 = gt[s:s + 32]
            tt = torch.full((q0.shape[0],), t_lo, device="cuda")
            qq = pipe["schedule"].q_sample(q0, tt, torch.randn_like(q0)).float()
            pp = model(q_t=qq, t=tt, map_context=ctx[s:s + 32])
            errs.append(((pp - q0) ** 2).mean(dim=(1, 2)))
    e = float(torch.cat(errs).mean())
    print("  t=%3d : MSE %.5f  RMSE %.1f m" % (t_lo, e, e ** 0.5 * 32.0))
