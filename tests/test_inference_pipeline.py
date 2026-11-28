"""End-to-end inference tests (design document section 13.4 / 8 / 9.4)."""
from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from gp_diffuser.diffusion import sampler_conditional_dit as sampler_mod
from gp_diffuser.safety import qp_projection as qp_mod
from gp_diffuser.training.dataset import SyntheticTrajectoryDataset
from gp_diffuser.vendor.ellipsenet.models.resunet import ResUNet


def _batch(item):
    return (item["occupancy"][None], item["sdf"][None], item["start_goal"][None],
            item["heading"][None], item["heading_valid"][None])


def _scene_with_cells(cfg, index: int = 2):
    """Test scene 2 has local convex cells, so the pack really constrains."""
    return SyntheticTrajectoryDataset(cfg, "test")[index]


def test_map_encoded_once_per_run(pipeline, sample, monkeypatch):
    item, _ = sample
    calls = {"occ": 0, "sdf": 0}
    original = ResUNet.forward

    def counting(self, x):
        calls["occ" if self is pipeline["net"].occ_unet else "sdf"] += 1
        return original(self, x)

    monkeypatch.setattr(ResUNet, "forward", counting)
    occ, sdf, sg, hd, hv = _batch(item)
    pipeline["sampler"].sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                               steps=6, seed=0)
    # one pass for the DiT map condition, one inside the vendored PSENet: their
    # forward owns its encoders, so the features cannot be injected
    assert calls == {"occ": 2, "sdf": 2}


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
    # model_branch and the guidance flag are reported separately
    assert out["steps_trace"][k]["model_branch"] == "map_only"


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


def test_frozen_pack_and_safety_memory_are_reused(pipeline, cfg, monkeypatch):
    item = _scene_with_cells(cfg)
    occ, sdf, sg, hd, hv = _batch(item)
    seen_packs = []
    seen_memory = []
    original = type(pipeline["model"]).forward
    original_guidance = sampler_mod.clean_prediction_guidance

    def spy(self, *args, **kwargs):
        seen_memory.append(kwargs.get("safety_memory"))
        return original(self, *args, **kwargs)

    def spy_guidance(q0, pack, *args, **kwargs):
        seen_packs.append(id(pack))
        return original_guidance(q0, pack, *args, **kwargs)

    monkeypatch.setattr(type(pipeline["model"]), "forward", spy)
    monkeypatch.setattr(sampler_mod, "clean_prediction_guidance", spy_guidance)
    out = pipeline["sampler"].sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                                     steps=10, seed=0)
    assert int(out["corridor_info"]["local_cells"]) > 0, "scene 2 must have local cells"
    assert len(seen_packs) >= 2, "CPCG must run on several guided steps"
    assert len(set(seen_packs)) == 1, "the frozen pack must be the same object"
    memory = [m for m in seen_memory if m is not None]
    assert memory and len({id(m) for m in memory}) == 1, \
        "safety memory must be built once"


def test_cpcg_adds_no_network_forward(pipeline, cfg, monkeypatch):
    item = _scene_with_cells(cfg)
    occ, sdf, sg, hd, hv = _batch(item)
    forwards = {"n": 0}
    original = type(pipeline["model"]).forward

    def spy(self, *args, **kwargs):
        forwards["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(pipeline["model"]), "forward", spy)
    out = pipeline["sampler"].sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                                     steps=8, safety_mode="cpcg", seed=0)
    assert forwards["n"] == out["frames"]["num_forwards"] == 8
    assert out["guidance_steps"] > 0


def test_sampler_never_calls_the_retired_alm(pipeline, sample, monkeypatch):
    from gp_diffuser.diffusion import bspline_alm as alm_mod

    calls = []
    monkeypatch.setattr(alm_mod, "bspline_alm_correct",
                        lambda *a, **k: calls.append(1) or (a[0], None, {}))
    item, _ = sample
    occ, sdf, sg, hd, hv = _batch(item)
    pipeline["sampler"].sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                               steps=8, seed=0)
    assert calls == []
    source = inspect.getsource(sampler_mod)
    assert "bspline_alm" not in source, "the active sampler must not import/call the ALM"
    assert "lam" not in [k for k in inspect.signature(
        sampler_mod.ConditionalDiTSampler.sample).parameters], "no dual variables"


def test_guidance_starts_only_after_the_pack_exists(pipeline, cfg):
    item = _scene_with_cells(cfg)
    occ, sdf, sg, hd, hv = _batch(item)
    out = pipeline["sampler"].sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                                     steps=10, seed=0)
    k = int(out["frames"]["k_safety"])
    applied = [step["constraint_guidance_applied"] for step in out["steps_trace"]]
    assert applied[:k] == [False] * k, "no CPCG before the pack exists"
    assert applied[k], "the pack-creating forward guides the same step"
    assert all(applied[k:]), "every later reverse step guides"


def test_guided_clean_prediction_enters_the_ddim_update(pipeline, cfg, monkeypatch):
    item = _scene_with_cells(cfg)
    occ, sdf, sg, hd, hv = _batch(item)
    guided, used = [], []
    original_guidance = sampler_mod.clean_prediction_guidance
    original_step = sampler_mod.ddim_step

    def spy_guidance(q0, *args, **kwargs):
        out, stats = original_guidance(q0, *args, **kwargs)
        guided.append(out.detach().clone())
        return out, stats

    def spy_step(q, q0_used, *args, **kwargs):
        used.append(q0_used.detach().clone())
        return original_step(q, q0_used, *args, **kwargs)

    monkeypatch.setattr(sampler_mod, "clean_prediction_guidance", spy_guidance)
    monkeypatch.setattr(sampler_mod, "ddim_step", spy_step)
    pipeline["sampler"].sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                               steps=8, seed=0)
    assert len(guided) >= 2
    matched = sum(any(torch.allclose(g, u) for u in used) for g in guided)
    assert matched == len(guided), "each guided q0 must reach the DDIM update"


def test_terminal_qp_runs_exactly_once(pipeline, cfg, monkeypatch):
    item = _scene_with_cells(cfg)
    occ, sdf, sg, hd, hv = _batch(item)
    calls = []
    original_qp = qp_mod.project_controls_to_corridor_qp

    def spy(*args, **kwargs):
        calls.append(1)
        return original_qp(*args, **kwargs)

    monkeypatch.setattr(qp_mod, "project_controls_to_corridor_qp", spy)
    out = pipeline["sampler"].sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                                     steps=8, seed=0)
    assert len(calls) == 1, "the QP must run once, after the whole reverse loop"
    assert out["final_qp_status"], out["final_qp_status"]


def test_dense_validation_runs_on_the_projected_curve(pipeline, cfg):
    item = _scene_with_cells(cfg)
    occ, sdf, sg, hd, hv = _batch(item)
    out = pipeline["sampler"].sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                                     steps=8, seed=0)
    controls = out["controls"]
    dense = pipeline["sampler"]._decode(controls, 512)
    expected = pipeline["transform"].to_meters(dense[0].detach().cpu().numpy())
    reported = out["trajectory_dense_m"][0].detach().cpu().numpy()
    assert np.allclose(expected, reported, atol=1e-4), \
        "the 512-point validation curve must belong to the FINAL controls"
    assert out["trajectory_dense_m"].shape == (1, 512, 2)


def test_map_only_mode_has_no_safety(pipeline, sample):
    item, _ = sample
    occ, sdf, sg, hd, hv = _batch(item)
    out = pipeline["sampler"].sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                                     steps=6, safety_mode="map_only", seed=0)
    assert out["safety_condition"] is None
    assert out["pack"] is None
    assert out["guidance_stats"] == []
    assert out["guidance_enabled"] is False
    assert out["projection_enabled"] is False
    assert out["final_qp_status"] == []
    assert all(step["branch"] == "map_only" for step in out["steps_trace"])
    assert not bool(out["corridor_certified"].any()), "no geometry, no certificate"


def test_safety_only_mode_has_conditions_but_no_correction(pipeline, cfg):
    item = _scene_with_cells(cfg)
    occ, sdf, sg, hd, hv = _batch(item)
    out = pipeline["sampler"].sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                                     steps=8, safety_mode="safety_only", seed=0)
    assert out["safety_condition"] is not None
    assert out["pack"] is not None
    assert out["guidance_stats"] == []
    assert out["final_qp_status"] == []
    assert not bool(out["corridor_certified"].any())
    assert set(step["model_branch"] for step in out["steps_trace"]) == \
        {"map_only", "safety_conditioned"}


def test_cpcg_mode_runs_guidance_and_projection(pipeline, cfg):
    item = _scene_with_cells(cfg)
    occ, sdf, sg, hd, hv = _batch(item)
    out = pipeline["sampler"].sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                                     steps=8, safety_mode="cpcg", seed=0)
    assert out["guidance_enabled"] and out["projection_enabled"]
    assert out["guidance_steps"] > 0
    assert out["final_qp_status"], "the terminal QP must report a status"
    assert out["controls"].shape == (1, 32, 2)
    for key in ("controls", "trajectory_m", "valid", "dense_collision_fraction",
                "pack", "timing"):
        assert key in out, "the sampler output interface must stay unchanged"


def test_legacy_mode_aliases(pipeline, sample):
    item, _ = sample
    occ, sdf, sg, hd, hv = _batch(item)
    alias = pipeline["sampler"].sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                                       steps=4, safety_mode="no_alm", seed=0)
    assert alias["safety_mode"] == "safety_only"
    alias = pipeline["sampler"].sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                                       steps=4, safety_mode="full", seed=0)
    assert alias["safety_mode"] == "cpcg"
    with pytest.raises(ValueError):
        pipeline["sampler"].sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                                   steps=2, safety_mode="alm", seed=0)


def test_failed_search_marks_the_sample_invalid(pipeline, sample, monkeypatch):
    item, _ = sample
    monkeypatch.setattr(pipeline["guidance"], "search",
                        lambda *a, **k: {"status": "unreachable", "path_xy": None,
                                         "fallback": "unreachable", "sources": 0,
                                         "expanded": 0, "valid": False})
    occ, sdf, sg, hd, hv = _batch(item)
    out = pipeline["sampler"].sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                                     steps=8, seed=0)
    assert out["search_path_m"] is None
    assert out["safety_condition"] is None
    assert bool(out["search_failed"][0])
    assert out["valid"] is False
    assert out["guidance_stats"] == []
    assert not bool(out["corridor_certified"].any())


def test_uncertified_projection_never_rewrites_the_trajectory(pipeline, cfg,
                                                             monkeypatch):
    item = _scene_with_cells(cfg)
    occ, sdf, sg, hd, hv = _batch(item)

    def failing(*args, **kwargs):
        q = args[0]
        return q.clone(), {"per_sample": [], "solver": "osqp",
                           "status": ["primal infeasible"], "iterations": [0],
                           "solve_time_s": [0.0], "max_violation_scene": [float("nan")],
                           "correction_m": [0.0], "num_corridor_rows": [0],
                           "certified": torch.zeros(q.shape[0], dtype=torch.bool),
                           "all_certified": False, "total_solve_time_s": 0.0}

    monkeypatch.setattr(qp_mod, "project_controls_to_corridor_qp", failing)
    out = pipeline["sampler"].sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                                     steps=8, seed=0)
    assert out["final_qp_status"] == ["primal infeasible"]
    assert torch.equal(out["controls"], out["controls_unprojected"])
    assert not bool(out["corridor_certified"].any())
    assert not out["corridor_certified_all"], \
        "a failed QP must never be reported as a safety certificate"


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
        safety = kwargs.get("safety_memory")
        conds.append(None if safety is None else id(safety))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(model), "forward", spy)
    pipeline["sampler"].sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                               steps=8, seed=0)
    non_null = [c for c in conds if c is not None]
    assert non_null and len(set(non_null)) == 1


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


def test_default_is_8_reverse_steps(pipeline, sample: object):
    """The repo's inference default is 8 EXECUTED steps (no retraining involved).

    Eight steps are a uniform subsampling of the trained 1000-entry noise table,
    so the same weights are used and the reverse loop runs 8 forwards.  Running
    the whole table stays possible but is never the default.
    """
    from gp_diffuser.diffusion.sampler_conditional_dit import pick_times

    assert len(pick_times(1000, 8)) == 8
    assert pick_times(1000, 8)[0] == 0 and pick_times(1000, 8)[-1] == 999
    assert len(pick_times(1000, 50)) == 50
    assert pick_times(1000, 1000) is None, "the full table is opt-in only"

    item, _ = sample
    occ, sdf, sg, hd, hv = _batch(item)
    sampler = pipeline["sampler"]
    assert int(sampler.cfg["sampling"]["steps"]) == 8
    out = sampler.sample(occ, sdf, sg, heading=hd, heading_valid=hv, seed=0)
    assert out["frames"]["num_forwards"] == 8


def test_safety_geometry_is_built_at_the_configured_forward(pipeline, sample: object):
    """``sampling.geometry_forward`` places the one-shot geometry (0-based).

    The shipped default builds it at the SECOND executed step, so CPCG guides
    every step after the first one instead of only the second half.
    """
    item, _ = sample
    occ, sdf, sg, hd, hv = _batch(item)
    out = pipeline["sampler"].sample(occ, sdf, sg, heading=hd, heading_valid=hv,
                                     seed=0)
    assert out["frames"]["k_safety"] == int(
        pipeline["sampler"].cfg["sampling"]["geometry_forward"])
    assert out["frames"]["k_safety"] == 1
    assert int(out["guidance_steps"]) >= 1
