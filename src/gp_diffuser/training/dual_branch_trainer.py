"""Single-stage dual-branch joint training (design document section 9).

One iteration, one backward pass, one optimizer step (spec §5):

    q_t common to both branches (same GT, same noise, same timestep)
    map_memory  = MapConditionEncoder(frozen occ/sdf bottleneck)     [B,1024,384]
    map_context = model.build_map_context(map_memory, start_goal, ...) [B,1026,384]
    branch U    = model(q_t, t, map_context)                      -> pred_u
    branch C    = model(q_t, t, map_context, safety_memory, mask)  -> pred_c
    L = lc*L_c + lu*L_u + la*L_align + (lcol*L_collision + lcorr*L_corridor)

There is no warm-up phase, no "safety activation" phase and no second network:
both branches call the same Conditional DiT weights inside the same iteration.
The frozen encoders / ellipse network never enter the autograd graph, while the
MapConditionEncoder is trainable and included in the optimizer AND in the gradient
clipping (spec §2.3 / §5.4).
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch

from ..models.boundary import BoundaryDecoder
from ..models.conditional_dit import ConditionalDiT
from ..safety.local_regions import pack_from_tables
from .losses import (DEFAULT_LOSS_WEIGHTS, alignment_loss, boundary_control_loss,
                     continuous_safety_loss, control_smoothness_loss,
                     control_x0_loss, curve_smoothness_loss, total_loss)


class DualBranchTrainer:
    """Joint trainer for the two conditional modes of one shared DiT."""

    def __init__(self, cfg: dict, model: ConditionalDiT, map_condition_encoder,
                 map_encoder, codec, schedule, device: str = "cpu",
                 scene_scale: float = 32.0, use_safety: bool = True):
        self.cfg = dict(cfg or {})
        # ``use_safety=False`` is the ablation arm: the Map-only branch alone
        # (no ellipse tokens, no alignment, no corridor term).  It shares every
        # other term with the dual objective so the two arms are comparable.
        self.use_safety = bool(use_safety)
        self.model = model.to(device)
        self.map_condition_encoder = map_condition_encoder.to(device)
        self.map_encoder = map_encoder.to(device)
        self.map_encoder.eval()
        self.codec = codec
        self.schedule = schedule.to(device)   # q_sample indexes these tables with t on the device
        self.device = device
        self.scene_scale = float(scene_scale)
        train_cfg = dict(self.cfg.get("training") or {})
        # "gt": the map-only branch is regressed on GT (original objective)
        # "safety": it is distilled from the safety branch only (no GT at all)
        self.u_target = str(train_cfg.get("u_target", "gt"))
        if self.u_target not in ("gt", "safety"):
            raise ValueError("training.u_target must be 'gt' or 'safety', got %r"
                             % (self.u_target,))
        # "raw": align the FREE network polygons (before the endpoint modulation)
        # "final": align the endpoint-corrected polygons
        self.align_space = str(train_cfg.get("align_space", "raw"))
        if self.align_space not in ("raw", "final"):
            raise ValueError("training.align_space must be 'raw' or 'final', got %r"
                             % (self.align_space,))
        weights = dict(DEFAULT_LOSS_WEIGHTS)
        weights.update({k: float(v) for k, v in (self.cfg.get("loss_weights") or {}).items()
                        if k in weights})
        self.weights = weights
        # parameter-free endpoint modulation, shared with the sampler (TrajSafe)
        boundary_cfg = dict(((self.cfg.get("model") or {}).get("boundary_decoder") or {}))
        self.boundary = BoundaryDecoder(profile=boundary_cfg.get("profile"),
                                        span=boundary_cfg.get("span"),
                                        dtype=torch.float32).to(device)
        self.trainable_parameters = (
            [p for p in self.model.parameters() if p.requires_grad]
            + [p for p in self.map_condition_encoder.parameters() if p.requires_grad])
        self.optimizer = torch.optim.AdamW(
            self.trainable_parameters,
            lr=float(train_cfg.get("lr", 2e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)))
        self.grad_clip = float(train_cfg.get("grad_clip", 1.0))
        self.history: List[Dict] = []
        # steps skipped because the loss / a gradient was not finite
        self.skipped_steps: int = 0

    # ------------------------------------------------------------------ helpers
    def trainable_parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.trainable_parameters))

    def _batch_tensor(self, batch: List[Dict], key: str) -> torch.Tensor:
        return torch.stack([item[key] for item in batch]).to(self.device)

    def _safety_conditions(self, cache: Dict[str, np.ndarray], index: np.ndarray):
        """Ellipse geometry + validity for the batch (no path tokens, spec §3.1)."""
        ellipse = torch.as_tensor(cache["ellipse_scene"][index], dtype=torch.float32,
                                  device=self.device)
        mask = cache.get("ellipse_mask")
        if mask is None:
            valid = torch.ones(ellipse.shape[0], ellipse.shape[1], dtype=torch.bool,
                               device=self.device)
        else:
            valid = torch.as_tensor(np.asarray(mask)[index], dtype=torch.bool,
                                    device=self.device)
        is_box = cache.get("cell_is_box")
        tables = []
        for i in index:
            table = {"cell_A": cache["cell_A"][i], "cell_b": cache["cell_b"][i],
                     "cell_mask": cache["cell_mask"][i], "anchors": cache["anchors"][i]}
            if is_box is not None:
                table["cell_is_box"] = is_box[i]
            tables.append(table)
        return {"ellipse": ellipse, "valid": valid}, tables

    # ------------------------------------------------------------- computation
    def _compute(self, batch: List[Dict], cache: Dict[str, np.ndarray],
                 index: np.ndarray):
        """Shared forward pass + joint loss (used by train and eval)."""
        occupancy = self._batch_tensor(batch, "occupancy")
        sdf = self._batch_tensor(batch, "sdf")
        cond = self._batch_tensor(batch, "start_goal")
        heading = self._batch_tensor(batch, "heading")
        heading_valid = self._batch_tensor(batch, "heading_valid")
        q0 = self._batch_tensor(batch, "controls_gt")
        B = q0.shape[0]

        # torch.no_grad() wraps ONLY the frozen map encoder (spec §2.3)
        with torch.no_grad():
            features = self.map_encoder.encode(occupancy, sdf)
        map_memory = self.map_condition_encoder(features["occ_bottleneck"],
                                                features["sdf_bottleneck"])
        map_context = self.model.build_map_context(map_memory, cond, heading,
                                                  heading_valid)

        safety_memory = safety_mask = None
        pack = None
        if self.use_safety:
            safety_condition, tables = self._safety_conditions(cache, index)
            safety_memory, safety_mask = self.model.build_safety_memory(
                safety_condition["ellipse"], safety_condition["valid"])
            # P1-2: the corridor loss must see the OBSTACLE cells only.  The
            # map-box cells are a fallback that would teach the model to hug the
            # map border.
            pack = pack_from_tables(self.codec, tables, include_box=False,
                                    device=self.device)

        # ---- the SAME noise and timestep for both branches ----------------
        T = int(self.schedule.num_timesteps)
        t = torch.randint(0, T, (B,), device=self.device)
        eps = torch.randn_like(q0)
        q_t = self.schedule.q_sample(q0, t, eps).to(q0.dtype)

        # ---- two forward passes of the SAME model -------------------------
        # the network emits a FREE control polygon; the fixed boundary decoder
        # turns it into an exact-endpoint polygon (never a teleport of Q_0/Q_{C-1})
        raw_u = self.model(q_t=q_t, t=t, map_context=map_context)
        pred_u = self.boundary(raw_u, cond, heading, heading_valid)
        ws, wg = self.boundary.weights(q0.shape[1], device=self.device, dtype=q0.dtype)
        basis = self.codec.basis.to(self.device)
        zero = q0.reshape(B, -1).sum(dim=1) * 0.0

        if self.use_safety:
            raw_c = self.model(q_t=q_t, t=t, map_context=map_context,
                               safety_memory=safety_memory, safety_mask=safety_mask)
            pred_c = self.boundary(raw_c, cond, heading, heading_valid)
            # ---- losses (TrajSafe-aligned) --------------------------------
            loss_c = control_x0_loss(pred_c, q0)
            # the alignment lives on the FREE polygons by default: the endpoint
            # modulation is a post-hoc fix-up, so aligning after it only aligns
            # the fix-up (both branches are dragged to the same S/G there)
            if self.align_space == "raw":
                loss_align = control_x0_loss(raw_u, raw_c.detach())
            else:
                loss_align = alignment_loss(pred_u, pred_c)
            if self.u_target == "safety":
                # distillation only: the map-only branch is supervised by the
                # safety branch, never by GT (no two-target average any more)
                loss_u = zero
                loss_bound = boundary_control_loss(raw_c, q0, cond, ws, wg)
            else:
                loss_u = control_x0_loss(pred_u, q0)      # interior controls only
                loss_bound = 0.5 * (boundary_control_loss(raw_u, q0, cond, ws, wg)
                                    + boundary_control_loss(raw_c, q0, cond, ws, wg))
            loss_smooth = 0.5 * (control_smoothness_loss(pred_u, q0)
                                 + control_smoothness_loss(pred_c, q0))
            loss_curve = 0.5 * (curve_smoothness_loss(pred_u, q0, basis)
                                + curve_smoothness_loss(pred_c, q0, basis))
            # ONE safety term: the same continuous constraint set the ALM projects onto
            loss_safe = continuous_safety_loss(pred_c, pack, reduction="none")
            # privileged supervision: the SAME constraint set, on the map-only
            # branch (a label, not an input - that branch still sees no ellipse)
            loss_safe_u = (continuous_safety_loss(pred_u, pack, reduction="none")
                           if self.weights.get("lambda_safe_u", 0.0) > 0 else None)
            loss = total_loss(loss_u, loss_c, loss_align, loss_bound, loss_smooth,
                              loss_curve, loss_safe, self.weights,
                              loss_safe_u=loss_safe_u).mean()
        else:
            # ablation arm: the Map-only branch with the SAME non-safety terms
            loss_c = loss_align = loss_safe = zero
            loss_u = control_x0_loss(pred_u, q0)
            loss_bound = boundary_control_loss(raw_u, q0, cond, ws, wg)
            loss_smooth = control_smoothness_loss(pred_u, q0)
            loss_curve = curve_smoothness_loss(pred_u, q0, basis)
            w = self.weights
            loss = (w["lambda_u"] * loss_u + w["lambda_boundary"] * loss_bound
                    + w["lambda_smooth"] * loss_smooth
                    + w["lambda_curve"] * loss_curve).mean()

        stats = {
            "loss": float(loss.item()),
            "loss_u": float(loss_u.mean().item()),
            "loss_c": float(loss_c.mean().item()),
            "loss_align": float(loss_align.mean().item()),
            "loss_boundary": float(loss_bound.mean().item()),
            "loss_smooth": float(loss_smooth.mean().item()),
            "loss_curve": float(loss_curve.mean().item()),
            "loss_safe": float(loss_safe.mean().item()),
            "loss_safe_max": float(loss_safe.max().item()),
            "loss_safe_u": (0.0 if self.use_safety is False
                            or self.weights.get("lambda_safe_u", 0.0) <= 0
                            else float(loss_safe_u.mean().item())),
            "corridor_pieces": int(pack.piece_mask.sum().item()) if pack is not None else 0,
            "corridor_samples": int((pack.piece_mask.any(dim=1)).sum().item())
                               if pack is not None else 0,
            "safety_gate_mean": float(np.mean([float(b.safety_gate.detach().item())
                                               for b in self.model.blocks])),
            "timestep_mean": float(t.float().mean().item()),
            "batch": int(B),
        }
        return loss, stats

    # -------------------------------------------------------------- train step
    def train_step(self, batch: List[Dict], cache: Dict[str, np.ndarray],
                   index: np.ndarray) -> Dict[str, float]:
        self.model.train()
        self.map_condition_encoder.train()
        self.map_encoder.eval()
        loss, stats = self._compute(batch, cache, index)

        # ---- non-finite guard (measured: one loss spike at step 215650 turned
        # the run into NaN at step 215700 and it kept "training" for 9 epochs,
        # overwriting the checkpoint with NaN weights.  Grad clipping cannot fix
        # NaN - it propagates it - so the step is skipped instead.)
        if not torch.isfinite(loss):
            self.skipped_steps += 1
            stats["skipped"] = 1.0
            self.history.append(stats)
            return stats

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not all(torch.isfinite(p.grad).all()
                   for p in self.trainable_parameters if p.grad is not None):
            self.optimizer.zero_grad(set_to_none=True)
            self.skipped_steps += 1
            stats["skipped"] = 1.0
            self.history.append(stats)
            return stats
        if self.grad_clip > 0:
            # the clip must cover the DiT AND the new map encoder (spec §5.4)
            torch.nn.utils.clip_grad_norm_(self.trainable_parameters, self.grad_clip)
        self.optimizer.step()

        stats["skipped"] = 0.0
        self.history.append(stats)
        return stats

    def weights_are_finite(self) -> bool:
        """``True`` when every trainable parameter is finite (checkpoint gate)."""
        return all(bool(torch.isfinite(p).all()) for p in self.trainable_parameters)

    # --------------------------------------------------------------- eval step
    @torch.no_grad()
    def eval_step(self, batch: List[Dict], cache: Dict[str, np.ndarray],
                  index: np.ndarray) -> Dict[str, float]:
        """Same objective without gradient (validation only, no state change)."""
        self.model.eval()
        self.map_condition_encoder.eval()
        self.map_encoder.eval()
        _loss, stats = self._compute(batch, cache, index)
        return stats

    # --------------------------------------------------------------------- ckpt
    def save_checkpoint(self, path: str, epoch: int, steps: int,
                        with_optimizer: bool = False) -> str:
        if not self.weights_are_finite():
            # never write a NaN checkpoint over a good one (measured: a diverged
            # run replaced last.pt with 261 non-finite tensors)
            print("[guard] REFUSING to save %s: non-finite weights at epoch %d"
                  % (path, int(epoch)), flush=True)
            return ""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {"architecture_version": self.model.architecture_version,
                   "model": self.model.state_dict(),
                   "map_condition_encoder": self.map_condition_encoder.state_dict(),
                   "config": self.model.config_dict(),
                   "use_safety": self.use_safety,
                   "u_target": self.u_target,
                   "align_space": self.align_space,
                   "epoch": int(epoch),
                   "steps": int(steps)}
        if with_optimizer:      # ~440 MB on top: only ``last.pt`` carries it
            payload["optimizer"] = self.optimizer.state_dict()
        torch.save(payload, path)
        return path

    def load_optimizer_state(self, payload: Dict) -> bool:
        """Restore AdamW moments when the checkpoint carries them."""
        state = (payload or {}).get("optimizer")
        if not state:
            return False
        try:
            self.optimizer.load_state_dict(state)
            return True
        except (ValueError, KeyError):
            return False

    # --------------------------------------------------------------------- fit
    def fit(self, train_items: List[Dict], cache: Dict[str, np.ndarray],
            epochs: int = 1, batch_size: int = 4, log_every: int = 10,
            seed: int = 0, max_steps: Optional[int] = None,
            val_fn=None, ckpt_dir: Optional[str] = None,
            ckpt_every: int = 0, ckpt_prefix: str = "ckpt",
            start_epoch: int = 0) -> Dict:
        """Iterate the joint objective over ``train_items``.

        ``start_epoch`` > 0 continues a previous run: the RNG streams are offset
        by it so the continued epochs see fresh sample orders instead of
        replaying the first ones.
        """
        torch.manual_seed(int(seed) + int(start_epoch))
        generator = np.random.default_rng(int(seed) + int(start_epoch))
        n = len(train_items)
        steps = 0
        start = time.perf_counter()
        for epoch in range(int(start_epoch), int(start_epoch) + int(epochs)):
            order = generator.permutation(n)
            for begin in range(0, n, int(batch_size)):
                index = order[begin:begin + int(batch_size)]
                if len(index) < 2:
                    continue
                batch = [train_items[int(i)] for i in index]
                stats = self.train_step(batch, cache, index)
                steps += 1
                if stats.get("skipped"):
                    print("epoch %d step %4d | [guard] non-finite loss/gradient - "
                          "step skipped (%d so far) %.1fs"
                          % (epoch + 1, steps, self.skipped_steps,
                             time.perf_counter() - start), flush=True)
                elif steps % max(1, int(log_every)) == 0:
                    print("epoch %d step %4d | loss %.5f (u %.4f c %.4f align %.4f "
                          "bnd %.4f sm %.4f crv %.4f safe %.4f/max %.3f su %.4f) "
                          "gate %.3f %.1fs"
                          % (epoch + 1, steps, stats["loss"], stats["loss_u"],
                             stats["loss_c"], stats["loss_align"],
                             stats["loss_boundary"], stats["loss_smooth"],
                             stats["loss_curve"], stats["loss_safe"],
                             stats["loss_safe_max"], stats["loss_safe_u"],
                             stats["safety_gate_mean"],
                             time.perf_counter() - start), flush=True)
                if max_steps and steps >= int(max_steps):
                    break
            if ckpt_dir:
                self.save_checkpoint(os.path.join(ckpt_dir, "last.pt"),
                                     epoch + 1, steps, with_optimizer=True)
                if ckpt_every and (epoch + 1) % int(ckpt_every) == 0:
                    self.save_checkpoint(
                        os.path.join(ckpt_dir, "%s_ep%03d.pt"
                                     % (ckpt_prefix, epoch + 1)), epoch + 1, steps)
            if max_steps and steps >= int(max_steps):
                break
        summary = {
            "steps": steps,
            "epochs": int(epochs),
            "seconds": time.perf_counter() - start,
            "skipped_steps": int(self.skipped_steps),
            "weights_finite": bool(self.weights_are_finite()),
            "last": self.history[-1] if self.history else None,
            "trainable_parameters": self.trainable_parameter_count(),
        }
        return summary
