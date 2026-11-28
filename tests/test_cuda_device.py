"""Device-handling test (code review P2-2).

Skipped on a CPU-only machine, but the assertions are the ones that failed
statically before the fix: the B-spline codec buffers, the headings and the
constraint pack must all follow the compute device.
"""
from __future__ import annotations

import pytest
import torch

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(),
                                   reason="needs a CUDA device")


@cuda_required
def test_pipeline_shapes_and_devices_on_cuda(cfg):
    from gp_diffuser.factory import build_pipeline
    from gp_diffuser.training.dataset import SyntheticTrajectoryDataset

    device = "cuda"
    pipeline = build_pipeline(cfg, device=device)
    dataset = SyntheticTrajectoryDataset(cfg, "test")
    item = dataset[0]

    assert pipeline["codec"].basis.device.type == "cuda"
    assert next(pipeline["model"].parameters()).device.type == "cuda"

    out = pipeline["sampler"].sample(
        item["occupancy"][None], item["sdf"][None], item["start_goal"][None],
        heading=item["heading"][None], heading_valid=item["heading_valid"][None],
        steps=4, seed=0)
    assert torch.isfinite(out["controls"]).all()
    assert out["controls"].device.type == "cuda"
    # the schedule tables must be on the device too (q_sample indexes them with t)
    assert pipeline["schedule"].sqrt_alphas_cumprod.device.type == "cuda"
    q0 = torch.zeros(2, 32, 2, device="cuda")
    table = int(pipeline["schedule"].num_timesteps)
    t = torch.tensor([0, table - 1], device="cuda")
    x_t = pipeline["schedule"].q_sample(q0, t)
    assert x_t.device.type == "cuda" and torch.isfinite(x_t).all()
    if out["pack"] is not None:
        assert out["pack"].basis.device.type == "cuda"
    if out["heading_error_deg"] is not None:
        assert out["heading_error_deg"].device.type == "cpu"
