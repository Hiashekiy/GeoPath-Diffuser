"""Adapter around the *vendored* PathSafe-Ellipse network (their code, their weights).

``src/gp_diffuser/vendor/ellipsenet`` is a verbatim copy of
``references/PathSafe-Ellipse/ellipsenet``; the only added file is
``vendor/ellipsenet/data/sdf.py`` (that package is gitignored upstream; the
constants are restored from upstream's own manifest/report).

The rest of the pipeline expects ``net.predict_ellipses(batch)`` returning
``{center, a, b, theta, prob, shape4}``, which is what this adapter exposes on top
of their ``PSENet.forward`` + ``PSENet.decode_argmax``.  Nothing about the network
is re-implemented: the argparse-max ray rule, the heads and the ResUNets are
theirs.

``features`` (the DiT's cached map features) is accepted for API compatibility
but ignored: their forward runs its own encoders.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import torch
import torch.nn as nn

from ..vendor.ellipsenet.models.psenet import PSENet, count_parameters

__all__ = ["VendoredEllipseNet", "PSENet", "count_parameters"]


class VendoredEllipseNet(nn.Module):
    """Their ``PSENet`` + the ``predict_ellipses`` call our pipeline uses."""

    def __init__(self, net: PSENet):
        super().__init__()
        self.net = net
        self.num_queries = int(getattr(net, "num_queries", 0)) or None

    @torch.no_grad()
    def predict_ellipses(self, batch: Dict[str, torch.Tensor],
                         features: Optional[Dict[str, torch.Tensor]] = None,
                         tau: float = 1.0) -> Dict[str, torch.Tensor]:
        out = self.net(batch, tau=tau)
        decoded = self.net.decode_argmax(out)
        theta = decoded["theta"]
        decoded["shape4"] = torch.stack([
            torch.log(decoded["a"].clamp_min(1e-6)),
            torch.log(decoded["b"].clamp_min(1e-6)),
            torch.cos(2.0 * theta), torch.sin(2.0 * theta)], dim=-1)
        return decoded

    def forward(self, batch, tau: float = 1.0):
        return self.net(batch, tau=tau)

    def __getattr__(self, name):
        """Transparent access to the wrapped network (``occ_unet``, ``cfg``, ...)."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            core = self.__dict__.get("_modules", {}).get("net")
            if core is None:
                raise
            return getattr(core, name)
