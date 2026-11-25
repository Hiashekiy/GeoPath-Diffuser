"""Single-stage dual-branch joint training (design document section 9).

One iteration, one backward pass, one optimizer step:

    q_t common to both branches (same GT, same noise, same timestep)
    branch U : map memory + start/goal                      -> pred_u
    branch C : map memory + start/goal + safety tokens      -> pred_c
    L = lc*L_c + lu*L_u + la*L_align + (lcol*L_collision + lcorr*L_corridor)

There is no warm-up phase, no "safety activation" phase and no second network:
both branches call the same Conditional DiT weights inside the same iteration.
The frozen encoders / ellipse network never enter the autograd graph.
"""
from __future__ import annotations

import math
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from ..models.conditional_dit import ConditionalDiT
from ..safety.local_regions import pad_region_tables, pack_from_tables
from .losses import (alignment_loss, collision_loss, control_loss, corridor_loss,
                     total_loss)


class DualBranchTrainer:
    """Joint trainer for the two conditional modes of one shared DiT."""

    def __init__(self, cfg: dict, model: ConditionalDiT, map_projection,
                 map_encoder, codec, schedule, device: str = "cpu",
                 scene_scale: float = 32.0):
        self.cfg = dict(cfg or {})
        self.model = model.to(device)
        self.map_projection = map_projection.to(device)
        self.map_encoder = map_encoder.to(device)
        self.map_encoder.eval()
        self.codec = codec
        self.schedule = schedule
        self.device = device
        self.scene_scale = float(scene_scale)
        train_cfg = dict(self.cfg.get("training") or {})
        weights = dict(self.cfg.get("loss_weights") or {})
        self.weights = {
            "lambda_u": float(weights.get("lambda_u", 1.0)),
            "lambda_c": float(weights.get("lambda_c", 1.0)),
            "lambda_align": float(weights.get("lambda_align", 0.5)),
            "lambda_collision": float(weights.get("lambda_collision", 1.0)),
            "lambda_corridor": float(weights.get("lambda_corridor", 1.0)),
        }
        self.collision_margin_m = float(train_cfg.get("collision_margin_m", 0.3))
        params = list(self.model.parameters()) + list(self.map_projection.parameters())
        self.optimizer = torch.optim.AdamW(
            [p for p in params if p.requires_grad],
            lr=float(train_cfg.get("lr", 2e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)))
        self.grad_clip = float(train_cfg.get("grad_clip", 1.0))
        self.history: List[Dict] = []

    # ------------------------------------------------------------------ helpers
    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad) \
            + sum(p.numel() for p in self.map_projection.parameters() if p.requires_grad)

    def _batch_tensor(self, batch: List[Dict], key: str) -> torch.Tensor:
        return torch.stack([item[key] for item in batch]).to(self.device)

    def _safety_conditions(self, cache: Dict[str, np.ndarray], index: np.ndarray):
        path = torch.as_tensor(cache["path_scene"][index], dtype=torch.float32,
                               device=self.device)
        ellipse = torch.as_tensor(cache["ellipse_scene"][index], dtype=torch.float32,
                                  device=self.device)
        valid = torch.ones(path.shape[0], path.shape[1] + ellipse.shape[1],
                           dtype=torch.bool, device=self.device)
        tables = [{"cell_A": cache["cell_A"][i], "cell_b": cache["cell_b"][i],
                   "cell_mask": cache["cell_mask"][i], "anchors": cache["anchors"][i]}
                  for i in index]
        return {"path": path, "ellipse": ellipse, "valid": valid}, tables

    # -------------------------------------------------------------- train step
    def train_step(self, batch: List[Dict], cache: Dict[str, np.ndarray],
                   index: np.ndarray) -> Dict[str, float]:
        self.model.train()
        self.map_projection.train()
        self.map_encoder.eval()
        occupancy = self._batch_tensor(batch, "occupancy")
        sdf = self._batch_tensor(batch, "sdf")
        cond = self._batch_tensor(batch, "start_goal")
        heading = self._batch_tensor(batch, "heading")
        heading_valid = self._batch_tensor(batch, "heading_valid")
        q0 = self._batch_tensor(batch, "controls_gt")
        B = q0.shape[0]

        with torch.no_grad():
            features = self.map_encoder.encode(occupancy, sdf)
        map_memory = self.map_projection(features["occ_bottleneck"],
                                         features["sdf_bottleneck"])

        safety_condition, tables = self._safety_conditions(cache, index)
        pack = pack_from_tables(self.codec, tables)

        # ---- the SAME noise and timestep for both branches ----------------
        T = int(self.schedule.num_timesteps)
        t = torch.randint(0, T, (B,), device=self.device)
        eps = torch.randn_like(q0)
        q_t = self.schedule.q_sample(q0, t, eps).to(q0.dtype)

        mem_u, mask_u = self.model.build_condition_memory(
            map_memory, cond, heading, heading_valid, None)
        mem_c, mask_c = self.model.build_condition_memory(
            map_memory, cond, heading, heading_valid, safety_condition)

        pred_u = self.model(q_t=q_t, t=t, map_memory=map_memory, cond=cond,
                            heading=heading, heading_valid=heading_valid,
                            condition_memory=mem_u, condition_mask=mask_u)
        pred_c = self.model(q_t=q_t, t=t, map_memory=map_memory, cond=cond,
                            heading=heading, heading_valid=heading_valid,
                            condition_memory=mem_c, condition_mask=mask_c)

        # ---- losses -------------------------------------------------------
        loss_u = control_loss(pred_u, q0)
        loss_c = control_loss(pred_c, q0)
        loss_align = alignment_loss(pred_u, pred_c)
        curve_c = self.codec.decode_controls(pred_c)
        loss_col = collision_loss(curve_c, sdf, margin_m=self.collision_margin_m,
                                  scene_scale=self.scene_scale)
        loss_corr = corridor_loss(pred_c, pack)
        # every term is a per-sample vector; the joint objective is its batch mean
        loss = total_loss(loss_u, loss_c, loss_align, loss_col, loss_corr,
                          self.weights).mean()

        # ---- exactly ONE backward + ONE update ----------------------------
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], self.grad_clip)
        self.optimizer.step()

        stats = {
            "loss": float(loss.item()),
            "loss_u": float(loss_u.mean().item()),
            "loss_c": float(loss_c.mean().item()),
            "loss_align": float(loss_align.mean().item()),
            "loss_collision": float(loss_col.mean().item()),
            "loss_corridor": float(loss_corr.mean().item()),
            "timestep_mean": float(t.float().mean().item()),
            "batch": int(B),
        }
        self.history.append(stats)
        return stats

    # --------------------------------------------------------------------- fit
    def fit(self, train_items: List[Dict], cache: Dict[str, np.ndarray],
            epochs: int = 1, batch_size: int = 4, log_every: int = 10,
            seed: int = 0, max_steps: Optional[int] = None,
            val_fn=None, ckpt_dir: Optional[str] = None) -> Dict:
        """Iterate the joint objective over ``train_items``."""
        torch.manual_seed(int(seed))
        generator = np.random.default_rng(int(seed))
        n = len(train_items)
        steps = 0
        start = time.perf_counter()
        for epoch in range(int(epochs)):
            order = generator.permutation(n)
            for begin in range(0, n, int(batch_size)):
                index = order[begin:begin + int(batch_size)]
                if len(index) < 2:
                    continue
                batch = [train_items[int(i)] for i in index]
                stats = self.train_step(batch, cache, index)
                steps += 1
                if steps % max(1, int(log_every)) == 0:
                    print("epoch %d step %4d | loss %.5f (u %.5f c %.5f align %.5f "
                          "col %.5f corr %.5f) %.1fs"
                          % (epoch + 1, steps, stats["loss"], stats["loss_u"],
                             stats["loss_c"], stats["loss_align"],
                             stats["loss_collision"], stats["loss_corridor"],
                             time.perf_counter() - start), flush=True)
                if max_steps and steps >= int(max_steps):
                    break
            if ckpt_dir:
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save({"model": self.model.state_dict(),
                            "map_projection": self.map_projection.state_dict(),
                            "config": self.model.config_dict(),
                            "epoch": epoch + 1,
                            "steps": steps}, os.path.join(ckpt_dir, "last.pt"))
            if max_steps and steps >= int(max_steps):
                break
        summary = {
            "steps": steps,
            "epochs": int(epochs),
            "seconds": time.perf_counter() - start,
            "last": self.history[-1] if self.history else None,
            "trainable_parameters": self.trainable_parameter_count(),
        }
        return summary
