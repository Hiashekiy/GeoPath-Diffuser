"""End-to-end inference tests (design document section 13.4 / 8)."""
from __future__ import annotations

import numpy as np
import torch

from gp_diffuser.diffusion import sampler_conditional_dit as sampler_mod
from gp_diffuser.map_encoder.resunet import ResUNet


def _batch(item):
    return (item["occupancy"][None], item["sdf"][None], item["start_goal"][None],
            item["heading"][None], item["heading_valid"][None])


def test_map_encoded_once_per_run(pipeline, sample, monkeypatch):
    item, _ = sample
    calls = {"occ": 0, "sdf": 0}
    original = ResUNet.forward_with_features

    def counting(self, x):
        calls["occ" if self is pipeline["net"].occ_unet else "sdf"] += 1
        return original(self, x)

    monkeypatch.setattr(ResUNet, "forward_with_features", counting)
    occ, sdf, sg, hd, hv = _batch(item)
    pipeline["sampler"].sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                               steps=6, seed=0)
    assert calls == {"occ": 1, "sdf": 1}


def test_safety_geometry_is_generated_once(pipeline, sample):
    item, _ = sample
    occ, sdf, sg, hd, hv = _batch(item)
    out = pipeline["sampler"].sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                                     steps=10, seed=0)
    assert out["frames"]["safety_generated_once"]
    assert len(out["search_stats"]) == 1
    k = int(out["frames"]["k_safety"])
    assert 0 < k < out["frames"]["num_forwards"], "safety must switch mid-schedule"
    branches = [step["branch"] for step in out["steps_trace"]]
    # forwards 0..k are map-only (the k-th generates the geometry), then safety
    assert branches[:k + 1] == ["map_only"] * (k + 1)
    assert set(branches[k + 1:]) == {"safety"}


def test_search_and_ellipses_are_never_recomputed(pipeline, sample, monkeypatch):
    item, _ = sample
    occ, sdf, sg, hd, hv = _batch(item)
    search_calls = {"n": 0}
    gen_calls = {"n": 0}
    original_search = pipeline["guidance"].search
    original_gen = type(pipeline["safety_generator"]).forward

    def search(*args, **kwargs):
        search_calls["n"] += 1
        return original_search(*args, **kwargs)

    def gen(self, *args, **kwargs):
        gen_calls["n"] += 1
        return original_gen(self, *args, **kwargs)

    monkeypatch.setattr(pipeline["guidance"], "search", search)
    monkeypatch.setattr(type(pipeline["safety_generator"]), "forward", gen)
    pipeline["sampler"].sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                              steps=8, seed=0)
    assert search_calls["n"] == 1, "the guidance search must run exactly once"
    assert gen_calls["n"] == 1, "the ellipse network must run exactly once"


def test_same_pack_is_reused_by_every_guided_step(pipeline, cfg, monkeypatch):
    """Test scene 2 has local convex cells, so the ALM actually runs."""
    from gp_diffuser.training.dataset import SyntheticTrajectoryDataset
    item = SyntheticTrajectoryDataset(cfg, "test")[2]
    occ, sdf, sg, hd, hv = _batch(item)
    seen = []
    original = sampler_mod.__dict__["pick_times"]   # touch the module for import

    from gp_diffuser.diffusion import bspline_alm as alm_mod
    original_alm = alm_mod.bspline_alm_correct

    def spy(q_ref, pack, codec, **kwargs):
        seen.append(id(pack))
        return original_alm(q_ref, pack, codec, **kwargs)

    monkeypatch.setattr(alm_mod, "bspline_alm_correct", spy)
    out = pipeline["sampler"].sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                                     steps=10, seed=0)
    assert int(out["corridor_info"]["local_cells"]) > 0, "scene 2 must have local cells"
    assert len(seen) >= 2, "ALM must run on several guided steps"
    assert len(set(seen)) == 1, "the frozen pack must be the same object every step"


def test_no_history_feedback_reaches_the_network(pipeline, sample, monkeypatch):
    item, _ = sample
    occ, sdf, sg, hd, hv = _batch(item)
    kwargs_seen = []
    model = pipeline["model"]
    original = type(model).forward

    def spy(self, *args, **kwargs):
        kwargs_seen.append(set(kwargs.keys()))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(model), "forward", spy)
    pipeline["sampler"].sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                               steps=8, seed=0)
    for keys in kwargs_seen:
        assert not (keys & {"feedback", "history", "delta", "correction"})


def test_safety_condition_object_is_stable_across_steps(pipeline, sample, monkeypatch):
    item, _ = sample
    occ, sdf, sg, hd, hv = _batch(item)
    conds = []
    model = pipeline["model"]
    original = type(model).forward

    def spy(self, *args, **kwargs):
        safety = kwargs.get("safety_condition")
        conds.append(None if safety is None else id(safety))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(model), "forward", spy)
    pipeline["sampler"].sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                               steps=8, seed=0)
    non_null = [c for c in conds if c is not None]
    assert non_null and len(set(non_null)) == 1


def test_map_only_mode_has_no_safety(pipeline, sample):
    item, _ = sample
    occ, sdf, sg, hd, hv = _batch(item)
    out = pipeline["sampler"].sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                                     steps=6, safety_mode="map_only", seed=0)
    assert out["safety_condition"] is None
    assert out["pack"] is None
    assert out["alm_stats"] == []
    assert all(step["branch"] == "map_only" for step in out["steps_trace"])


def test_outputs_are_finite_and_endpoints_exact(pipeline, sample):
    item, _ = sample
    occ, sdf, sg, hd, hv = _batch(item)
    out = pipeline["sampler"].sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                                     steps=6, seed=1)
    assert torch.isfinite(out["controls"]).all()
    assert out["controls"].shape == (1, 32, 2)
    assert out["trajectory_m"].shape == (1, 128, 2)
    assert torch.allclose(out["controls"][:, 0], sg[:, 0], atol=1e-6)
    assert torch.allclose(out["controls"][:, -1], sg[:, 1], atol=1e-6)


def test_steps_are_configurable(pipeline, sample):
    item, _ = sample
    occ, sdf, sg, hd, hv = _batch(item)
    for steps in (4, 12):
        out = pipeline["sampler"].sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                                         steps=steps, seed=0)
        assert out["frames"]["num_forwards"] == steps
