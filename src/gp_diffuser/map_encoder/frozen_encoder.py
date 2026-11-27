"""Frozen PathSafe-EllipseNet map encoder.

Design document section 3: the two ResUNets of PathSafe-EllipseNet are reused
as-is, frozen, and encoded exactly once per scene.  Their ``enc3`` bottleneck
feeds the diffusion map memory; the full-resolution feature is kept for
PathSafe-EllipseNet's own local queries.

Nothing here instantiates a second map encoder and nothing here re-trains the
encoder: the weights come from ``checkpoints/v3_best.pt``.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


def _resunet_features(net, x):
    """Bottleneck (`enc3` output) + full-resolution output of a vendored ResUNet.

    The vendored PathSafe-Ellipse `ResUNet.forward` returns only the final
    tensor, so the bottleneck the DiT needs is captured with a forward hook
    instead of re-implementing their encoder.
    """
    captured = {}

    def _hook(_module, _inp, out):
        captured["bottleneck"] = out

    handle = net.enc3.register_forward_hook(_hook)
    try:
        full = net(x)
    finally:
        handle.remove()
    return {"bottleneck": captured["bottleneck"], "full": full}


class FrozenMapEncoder(nn.Module):
    """Read-only view over a (shared) :class:`PathSafeEllipseNet` instance.

    ``net`` is the SAME instance the safety generator uses, so "map features are
    encoded once" is structural, not a convention: there is only one pair of
    ResUNets in the process.
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.freeze()

    # ------------------------------------------------------------------ freeze
    def freeze(self) -> None:
        """``requires_grad=False`` + ``eval()`` on both encoders."""
        self.net.eval()
        for module in (self.net.occ_unet, self.net.dist_unet):
            module.eval()
            for param in module.parameters():
                param.requires_grad_(False)
        # the safety heads are frozen too (section 9.5): they only generate
        # geometry, they are never optimised by the diffusion loss
        for param in self.net.parameters():
            param.requires_grad_(False)
        for module in self.net.modules():
            module.eval()

    def train(self, mode: bool = True):  # noqa: D401 - never train the encoders
        """Keep the frozen modules in eval mode even if the parent model trains."""
        super().train(False)
        self.net.eval()
        return self

    # ------------------------------------------------------------------ encode
    @torch.no_grad()
    def encode(self, occupancy: torch.Tensor, sdf: torch.Tensor) -> Dict[str, torch.Tensor]:
        """One forward pass -> both bottlenecks and both full-resolution features.

        ``occupancy`` ``[B,1,H,W]`` in {0,1}; ``sdf`` ``[B,1,H,W]`` in metres
        (positive in free space).  Returns::

            {
              "occ_bottleneck": [B,192,32,32],   # Diffusion map condition
              "sdf_bottleneck": [B,192,32,32],
              "occ_full":       [B,128,256,256], # PathSafe local queries
              "sdf_full":       [B,128,256,256],
            }
        """
        occ_out = _resunet_features(self.net.occ_unet, occupancy)
        # the SDF encoder sees the same clipped input PSENet.forward uses
        sdf_in = torch.clamp(sdf / float(self.net.d_clip), -1.0, 1.0)
        sdf_out = _resunet_features(self.net.dist_unet, sdf_in)
        features = {
            "occ_bottleneck": occ_out["bottleneck"].detach(),
            "sdf_bottleneck": sdf_out["bottleneck"].detach(),
            "occ_full": occ_out["full"].detach(),
            "sdf_full": sdf_out["full"].detach(),
        }
        return features

    # ------------------------------------------------------------------ checks
    def encoder_parameters(self):
        return list(self.net.occ_unet.parameters()) + list(self.net.dist_unet.parameters())

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.encoder_parameters() if p.requires_grad)
