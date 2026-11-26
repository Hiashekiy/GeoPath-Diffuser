"""Regression tests for the defects found in the code review.

Each test encodes the behaviour REQUIRED after the fix and must PASS.  The defect
ids match ``docs/code_review_verification.md``:

    P0-1  collision_loss mixed metres with normalised units
    P0-2  ellipse shape4 (metres) was fed to the convex-region builder unscaled
    P0-3  bridge cells were truncated by cells[:Q]
    P0-4  pack_from_tables kept only the LAST sample's anchors
    P1-1  collision detection sampled points only -> thin walls missed
    P1-2  ALM ran although no local convex cell existed (box cells only)
    P1-3  predicted endpoints were not locked before the ALM
    P1-4  a failed guidance search silently reused the (colliding) guide
    P1-5  an ALM correction was accepted even when it made the sample worse
    P2-1  heading was a network condition but never verified
    P2-3  the safety cache had no fingerprint
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest
import torch

from gp_diffuser.safety import local_regions
from gp_diffuser.safety.collision_query import curve_collision_intervals
from gp_diffuser.safety.convex_region import EllipseRegionBuilder
from gp_diffuser.safety.local_regions import (build_region_table, pack_from_tables)
from gp_diffuser.safety.safety_corridor import build_safety_corridor
from gp_diffuser.safety.safety_features import ellipse_shape4_to_scene
from gp_diffuser.training.dataset import SyntheticTrajectoryDataset
from gp_diffuser.training.losses import (continuous_safety_loss,
                                          pack_violation)


# --------------------------------------------------------------------------- #
# P0-1 — the SDF/margin collision term is gone; the single safety term is the
# continuous convex-hull violation against the corridor pack (TrajSafe design).
# --------------------------------------------------------------------------- #
def _unit_pack(controls_per_sample=8, offset=0.0):
    """A single half-space ``x <= 0`` applied to every Bezier control."""
    B, C = 1, controls_per_sample
    piece = C - 3
    A = torch.zeros(B, piece, 1, 2)
    A[..., 0] = 1.0
    b = torch.full((B, piece, 1), float(offset))
    return pack_stub(A, b, B, piece, C)


def pack_stub(A, b, B, piece, C):
    class _Pack:
        pass

    pack = _Pack()
    pack.piece_A = A
    pack.piece_b = b
    pack.piece_mask = torch.ones(B, piece, dtype=torch.bool)
    pack.face_mask = torch.ones(B, piece, A.shape[2], dtype=torch.bool)
    extraction = torch.zeros(B, piece, 4, C)
    for i in range(piece):
        for j in range(4):
            extraction[:, i, j, i + j] = 1.0
    pack.extraction = extraction
    return pack


def test_continuous_safety_loss_zero_inside_and_positive_outside():
    pack = _unit_pack(offset=0.0)
    inside = torch.full((1, 8, 2), -0.5)
    assert float(pack_violation(inside, pack).sum()) == 0.0
    outside = torch.full((1, 8, 2), 0.25)
    value = float(pack_violation(outside, pack)[0])
    assert value == pytest.approx(0.25, rel=1e-5), "max violation in scene units"


def test_continuous_safety_loss_ignores_box_cells():
    pack = _unit_pack(offset=-1.0)          # trivially satisfied
    pack.piece_mask[:] = False              # no local cell at all
    controls = torch.rand(1, 8, 2)
    assert float(continuous_safety_loss(controls, pack).sum()) == 0.0


# --------------------------------------------------------------------------- #
# P0-2
# --------------------------------------------------------------------------- #
def test_ellipse_shape4_to_scene_units():
    axes_m = torch.tensor([1.0, 0.13])
    shape4_m = torch.tensor([[math.log(axes_m[0]), math.log(axes_m[1]), 1.0, 0.0]])
    shape4_s = ellipse_shape4_to_scene(shape4_m, 32.0)
    recovered = shape4_s[:, :2].exp()
    assert torch.allclose(recovered, axes_m / 32.0, atol=1e-6)
    # the orientation channels are untouched
    assert torch.allclose(shape4_s[:, 2:], shape4_m[:, 2:])


def test_region_builder_receives_scene_axes(pipeline, sample, monkeypatch):
    item, dataset = sample
    scene = dataset.raw_scene(0)
    transform = pipeline["transform"]
    captured = {}
    original = local_regions.build_region_table

    def spy(occupancy, centers, shape4, anchors, *args, **kwargs):
        captured["shape4"] = torch.as_tensor(shape4).detach().clone()
        return original(occupancy, centers, shape4, anchors, *args, **kwargs)

    monkeypatch.setattr(local_regions, "build_region_table", spy)
    occ = torch.as_tensor(scene["occupancy"])[None, None]
    sdf = torch.as_tensor(scene["sdf"])[None, None]
    features = pipeline["map_encoder"].encode(occ, sdf)
    pipeline["sampler"]._generate_safety(
        occ, sdf, features, torch.as_tensor(item["path_gt_scene"])[None],
        torch.as_tensor(item["start_goal"])[None], pipeline["cfg"]["corridor"])
    axes_scene = captured["shape4"][:, :2].clamp(-6.0, 0.7).exp()
    # 64 m map -> a scene axis can never exceed 1.0; the metres value (~0.78) would
    assert float(axes_scene.max()) < 0.1
    assert float(axes_scene.max()) > 0.0
    assert transform.scale == 32.0


def test_shape4_metre_frame_hits_the_axis_clamp():
    """Why P0-2 materially changes geometry: the clamp is applied in the wrong unit.

    ``ellipse_shape4_to_quadratic`` clamps the log-axes to ``(-6.0, 0.7)`` and then
    the metres axes to ``axis_max = 2.0``.  Feeding METRE log-axes therefore
    truncates every semi-axis above ``exp(0.7) = 2.01 m``, while the scene frame
    (``axis_max = 2.0`` scene units = 64 m) never touches it on this map.
    """
    from gp_diffuser.safety.convex_region import ellipse_shape4_to_quadratic

    a_m = 2.5                       # a perfectly legal PathSafe prediction
    shape4_m = torch.tensor([[math.log(a_m), math.log(0.5), 1.0, 0.0]])
    shape4_s = ellipse_shape4_to_scene(shape4_m, 32.0)

    q_metres = ellipse_shape4_to_quadratic(shape4_m, 0.002, 2.0)[0]
    q_scene = ellipse_shape4_to_quadratic(shape4_s, 0.002, 2.0)[0]
    a_from_metres = float((1.0 / q_metres[0, 0]).sqrt())
    a_from_scene = float((1.0 / q_scene[0, 0]).sqrt()) * 32.0

    assert abs(a_from_metres - 2.0) < 0.02, "the metre frame truncates a=2.5 m to 2.0"
    assert abs(a_from_scene - a_m) < 0.02, "the scene frame keeps a=2.5 m"


def test_shape4_scene_axes_are_physically_consistent(pipeline, sample):
    """The axes the builder sees must equal the metres axes divided by the scale."""
    _, dataset = sample
    scene = dataset.raw_scene(0)
    transform = pipeline["transform"]
    result = pipeline["guidance"].search(scene["occupancy"], scene["gt_path_m"],
                                         scene["start_m"], scene["goal_m"])
    safety = pipeline["safety_generator"](
        torch.as_tensor(scene["occupancy"])[None, None],
        torch.as_tensor(scene["sdf"])[None, None],
        torch.as_tensor(result["path_xy"], dtype=torch.float32)[None])
    axes_m = torch.stack([safety["ellipse"][..., 2], safety["ellipse"][..., 3]], dim=-1)
    scene_axes = ellipse_shape4_to_scene(safety["ellipse_shape4"], transform.scale)[..., :2].exp()
    assert torch.allclose(scene_axes, axes_m / transform.scale, atol=1e-5)
    # the map is 64 m across, so no scene-frame semi-axis may approach 1.0
    assert float(scene_axes.max()) < 0.2


# --------------------------------------------------------------------------- #
# P0-3 / P0-4
# --------------------------------------------------------------------------- #
def _scene_table(pipeline, scene, cfg, scope="full"):
    transform = pipeline["transform"]
    guide = pipeline["guidance"]
    result = guide.search(scene["occupancy"], scene["gt_path_m"],
                          scene["start_m"], scene["goal_m"])
    safety = pipeline["safety_generator"](
        torch.as_tensor(scene["occupancy"])[None, None],
        torch.as_tensor(scene["sdf"])[None, None],
        torch.as_tensor(result["path_xy"], dtype=torch.float32)[None])
    centers = torch.as_tensor(transform.to_scene(safety["ellipse_center"][0].numpy()),
                              dtype=torch.float32)
    shape4_scene = ellipse_shape4_to_scene(safety["ellipse_shape4"], transform.scale)
    anchors = torch.linspace(0.0, 1.0, 64, dtype=torch.float64)
    gamma = torch.as_tensor(transform.to_scene(result["path_xy"]), dtype=torch.float32)
    table, info = build_region_table(
        torch.as_tensor(scene["occupancy"])[None, None], centers, shape4_scene[0],
        anchors, curve_scene=None, gamma_scene=gamma, config=cfg["corridor"],
        scope=scope, scene_scale=transform.scale)
    return table, info, centers, shape4_scene, anchors, gamma, result


def test_no_bridge_cell_is_truncated(pipeline, sample, cfg):
    _, dataset = sample
    scene = dataset.raw_scene(0)
    table, _info, centers, shape4, anchors, gamma, _ = _scene_table(
        pipeline, scene, cfg, scope="full")
    corridor = build_safety_corridor(
        EllipseRegionBuilder(torch.as_tensor(scene["occupancy"])[None, None].float(),
                             cfg["corridor"]["region"]),
        centers, shape4[0], anchors, gamma=gamma, gamma_lengths=int(gamma.shape[0]),
        config=cfg["corridor"])
    assert corridor.num_cells > 64, "this scene needs at least one bridge cell"
    assert table["cell_A"].shape[0] == corridor.num_cells
    assert table["cell_A"].shape[0] > 64, "bridge rows must survive"


def test_batch_pack_uses_per_sample_anchors(pipeline, sample, cfg):
    _, dataset = sample
    scene = dataset.raw_scene(0)
    table, *_ = _scene_table(pipeline, scene, cfg, scope="full")
    table_a = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in table.items()}
    table_b = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in table.items()}
    table_b["anchors"] = np.linspace(0.0, 1.0, table_b["anchors"].shape[0]) ** 0.35

    pack_a = pack_from_tables(pipeline["codec"], [table_a])
    pack_b = pack_from_tables(pipeline["codec"], [table_b])
    pack_batch = pack_from_tables(pipeline["codec"], [table_a, table_b])
    assert torch.allclose(pack_batch.piece_b[0], pack_a.piece_b[0])
    assert not torch.allclose(pack_batch.piece_b[0], pack_b.piece_b[0])


def test_different_row_counts_merge(pipeline, sample, cfg):
    """Per-sample packs with different piece/face counts must merge cleanly."""
    _, dataset = sample
    scene = dataset.raw_scene(0)
    table, *_ = _scene_table(pipeline, scene, cfg, scope="full")
    trimmed = {k: (v[:32].copy() if isinstance(v, np.ndarray) else v)
               for k, v in table.items()}
    pack = pack_from_tables(pipeline["codec"], [table, trimmed])
    assert pack.extraction.shape[0] == 2
    assert int(pack.num_pieces[0]) > int(pack.num_pieces[1])
    assert pack.piece_mask[1].sum() > 0


# --------------------------------------------------------------------------- #
# P1-1
# --------------------------------------------------------------------------- #
def test_thin_wall_detected_by_segment_sweep():
    """A 0.25 m wall between two 0.7 m-apart samples must be detected."""
    occ = np.zeros((256, 256))
    occ[100, 128] = 1.0
    grid = 256
    y = 2.0 * 100 / grid - 1.0 + 1.0 / grid
    x0 = 2.0 * 127 / grid - 1.0 + 1.0 / grid
    x1 = 2.0 * 129 / grid - 1.0 + 1.0 / grid
    curve = np.array([[x0, y], [x1, y]])
    info = curve_collision_intervals(occ, curve)
    assert info["collision_fraction"] > 0.0
    assert len(info["intervals"]) == 1


def test_sampler_collision_is_segment_level(pipeline):
    occ = torch.zeros(1, 1, 256, 256)
    occ[0, 0, 100, 128] = 1.0
    grid = 256
    y = 2.0 * 100 / grid - 1.0 + 1.0 / grid
    x0 = 2.0 * 127 / grid - 1.0 + 1.0 / grid
    x1 = 2.0 * 129 / grid - 1.0 + 1.0 / grid
    curve = torch.tensor([[[x0, y], [x1, y]]])
    fraction = pipeline["sampler"]._collision_fractions(occ, curve)
    assert float(fraction[0]) > 0.0


def test_local_spans_follow_the_projected_position(pipeline, sample, cfg):
    """A collision must constrain the ellipses near THAT position, not near the
    equally-indexed position of the search path (P1-6)."""
    _, dataset = sample
    scene = dataset.raw_scene(0)
    transform = pipeline["transform"]
    result = pipeline["guidance"].search(scene["occupancy"], scene["gt_path_m"],
                                         scene["start_m"], scene["goal_m"])
    path_scene = torch.as_tensor(transform.to_scene(result["path_xy"]),
                                 dtype=torch.float32)
    # collide the curve at ~30 % of its length
    curve = path_scene.clone()
    blocked = np.argwhere(scene["occupancy"] > 0.5)
    grid = scene["occupancy"].shape[-1]
    row, col = blocked[len(blocked) // 3]
    obstacle_scene = torch.tensor([2.0 * col / grid - 1.0 + 1.0 / grid,
                                   2.0 * row / grid - 1.0 + 1.0 / grid])
    k = int(0.3 * (len(curve) - 1))
    curve[k - 2:k + 3] = obstacle_scene
    table, info = build_region_table(
        torch.as_tensor(scene["occupancy"])[None, None],
        torch.as_tensor(transform.to_scene(np.zeros((64, 2))), dtype=torch.float32),
        torch.zeros(64, 4), torch.linspace(0.0, 1.0, 64, dtype=torch.float64),
        curve_scene=curve[None], gamma_scene=path_scene, config=cfg["corridor"],
        scope="local", scene_scale=transform.scale)
    assert info["collision_intervals"] >= 1
    # the forced collision is at 30 % of the curve; the span must be centred near
    # the projected arc-length position, i.e. in the first half of the queries
    if info["spans"]:
        a, b = info["spans"][0]
        assert a <= 32 <= b + 10


def test_all_collision_intervals_are_used(cfg):
    """P1-7: no cap on the number of collision intervals.

    Four independent walls sit on one straight trajectory.  All four collision
    intervals must be detected, projected onto the searched path as four disjoint
    query spans, and each span must reach the convex-region builder -- nothing is
    truncated (the old ``max_spans = 3`` dropped the fourth region).
    """
    from gp_diffuser.frames import SceneTransform
    from gp_diffuser.safety.collision_query import collision_spans_on_path

    grid, res_m = 128, 0.25
    extent = grid * res_m                     # 32 m
    transform = SceneTransform(extent, grid)
    occ = np.zeros((grid, grid))
    wall_cols = [24, 48, 72, 96]              # four separate walls
    for col in wall_cols:
        occ[40:88, col] = 1.0

    xs = np.arange(2.0, 30.0, 0.35)
    curve_m = np.stack([xs, np.full_like(xs, 16.0)], axis=-1)
    curve_scene = transform.to_scene(curve_m)

    info = collision_spans_on_path(occ, curve_scene, curve_scene, n_query=64,
                                   pad=4, min_q=8, scene_scale=transform.scale)
    assert info["n_intervals"] == 4, "four independent collision intervals expected"
    assert len(info["spans"]) == 4, "no interval may be dropped"
    for (a1, b1), (a2, b2) in zip(info["spans"], info["spans"][1:]):
        assert b1 < a2, "the four spans must stay disjoint"

    # every span must enter the convex-region builder (one attempt per span)
    q = int(curve_scene.shape[0])
    centers = torch.as_tensor(curve_scene[::max(1, q // 64)][:64], dtype=torch.float32)
    if centers.shape[0] != 64:                          # pad by repeating the tail
        centers = torch.cat([centers, centers[-1:].repeat(64 - centers.shape[0], 1)])
    shape4 = torch.tensor([[math.log(0.03), math.log(0.01), 1.0, 0.0]]).repeat(64, 1)
    table, tinfo = build_region_table(
        torch.as_tensor(occ)[None, None], centers, shape4,
        torch.linspace(0.0, 1.0, 64, dtype=torch.float64),
        curve_scene=torch.as_tensor(curve_scene, dtype=torch.float32)[None],
        gamma_scene=torch.as_tensor(curve_scene, dtype=torch.float32),
        config=cfg["corridor"], scope="local", scene_scale=transform.scale)
    assert len(tinfo["spans"]) == 4
    assert len(tinfo["corridor_valid"]) == 4, "each span must be attempted"
    assert len(tinfo["corridor_failure"]) == 4
    assert table["cell_A"].shape[0] > 0


# --------------------------------------------------------------------------- #
# P1-2
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("index", [0, 1])
def test_alm_runs_only_with_local_convex_cells(pipeline, cfg, index):
    dataset = SyntheticTrajectoryDataset(cfg, "test")
    item = dataset[index]
    out = pipeline["sampler"].sample(
        item["occupancy"][None], item["sdf"][None], item["start_goal"][None],
        heading=item["heading"][None], heading_valid=item["heading_valid"][None],
        steps=8, seed=0)
    local = int(out["corridor_info"].get("local_cells", 0))
    assert (local > 0) == (len(out["alm_stats"]) > 0), (
        "local_cells=%d but alm_stats=%d" % (local, len(out["alm_stats"])))


def test_corridor_loss_ignores_box_cells(pipeline, sample, cfg):
    """A table of map-box cells only must produce no corridor constraint at all."""
    _, dataset = sample
    scene = dataset.raw_scene(0)
    # local scope with no collision interval -> every row is a map-box cell
    table, info, *_ = _scene_table(pipeline, scene, cfg, scope="local")
    assert info["box_cells"] == table["cell_A"].shape[0]
    box_only = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in table.items()}
    box_only["cell_mask"] = np.asarray(table["cell_is_box"], dtype=bool).copy()
    assert box_only["cell_mask"].any()
    pack_all = pack_from_tables(pipeline["codec"], [box_only], include_box=True)
    pack_local = pack_from_tables(pipeline["codec"], [box_only], include_box=False)
    assert pack_all is not None and int(pack_all.piece_mask.sum()) > 0
    assert pack_local is None, "box cells must not become corridor constraints"
    controls = torch.zeros(1, 32, 2)
    # no local pack at all -> the single safety term contributes exactly zero
    loss = continuous_safety_loss(controls, pack_local)
    assert float(loss.sum()) == 0.0 and loss.requires_grad


# --------------------------------------------------------------------------- #
# P1-3
# --------------------------------------------------------------------------- #
def test_prediction_endpoints_locked(pipeline, sample, monkeypatch):
    """A drifted network prediction must be endpoint-locked before it is decoded."""
    item, _ = sample
    start_goal = item["start_goal"][None]
    model = pipeline["model"]
    original = type(model).forward

    def drifted(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        return out + 0.3                      # every control point drifts

    monkeypatch.setattr(type(model), "forward", drifted)
    out = pipeline["sampler"].sample(
        item["occupancy"][None], item["sdf"][None], start_goal,
        heading=item["heading"][None], heading_valid=item["heading_valid"][None],
        steps=8, seed=0)
    guide = out["guide_trajectory_scene"][0]
    assert float((guide[0] - start_goal[0, 0]).norm()) < 1e-6
    assert float((guide[-1] - start_goal[0, 1]).norm()) < 1e-6
    assert float((out["controls"][0, 0] - start_goal[0, 0]).norm()) < 1e-6


# --------------------------------------------------------------------------- #
# P1-4
# --------------------------------------------------------------------------- #
def test_failed_search_is_not_a_path(pipeline, sample, monkeypatch):
    item, _ = sample
    monkeypatch.setattr(pipeline["guidance"], "search",
                        lambda *a, **k: {"status": "unreachable", "path_xy": None,
                                         "fallback": "unreachable", "sources": 0,
                                         "expanded": 0, "valid": False})
    out = pipeline["sampler"].sample(
        item["occupancy"][None], item["sdf"][None], item["start_goal"][None],
        heading=item["heading"][None], heading_valid=item["heading_valid"][None],
        steps=8, seed=0)
    assert out["search_path_m"] is None
    assert out["safety_condition"] is None
    assert bool(out["search_failed"][0])
    assert out["valid"] is False
    assert out["alm_stats"] == []


# --------------------------------------------------------------------------- #
# P1-5
# --------------------------------------------------------------------------- #
def test_accept_correction_rejects_worse():
    from gp_diffuser.diffusion.sampler_conditional_dit import ConditionalDiTSampler
    active = torch.tensor([True, True, True, False])
    col_before = torch.tensor([0.1, 0.0, 0.0, 0.0])
    col_after = torch.tensor([0.0, 0.2, 0.0, 0.0])     # sample 1 gets worse
    viol_before = torch.tensor([1.0, 1.0, 0.5, 1.0])
    viol_after = torch.tensor([0.0, 0.0, 0.9, 0.0])    # sample 2 violation grows
    accept = ConditionalDiTSampler.accept_correction(active, col_before, col_after,
                                                     viol_before, viol_after)
    assert accept.tolist() == [True, False, False, False]


# --------------------------------------------------------------------------- #
# P2-1
# --------------------------------------------------------------------------- #
def test_heading_errors_measured(pipeline):
    sampler = pipeline["sampler"]
    codec = pipeline["codec"]
    controls = torch.zeros(1, 32, 2)
    controls[0, :, 0] = torch.linspace(0.0, 1.0, 32)   # straight +x
    heading = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])  # both along +x
    err = sampler._heading_errors_deg(controls, heading, None)
    assert err is not None and err.shape == (1, 2)
    assert float(err.max()) < 1.0
    heading_y = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])
    err_y = sampler._heading_errors_deg(controls, heading_y, None)
    assert abs(float(err_y.mean()) - 90.0) < 2.0
    assert codec.basis_derivative_at([0.0, 1.0]).shape == (2, 32)


# --------------------------------------------------------------------------- #
# P2-2 / P2-3 / search verification
# --------------------------------------------------------------------------- #
def test_search_flags_blocked_endpoint(cfg):
    from gp_diffuser.guidance.search import GuidanceFieldSearch
    search = GuidanceFieldSearch(cfg["guidance"], res_m=0.25)
    occ = np.zeros((128, 128), dtype=np.int8)
    occ[10:12, 10:12] = 1
    start = np.array([11 * 0.25, 11 * 0.25])          # inside the wall
    goal = np.array([27.0, 27.0])
    guide = np.stack([np.linspace(start[0], goal[0], 64),
                      np.linspace(start[1], goal[1], 64)], axis=-1)
    result = search.search(occ, guide, start, goal)
    assert result["endpoint_projected"] is True
    assert result["valid"] is False


def test_cache_fingerprint_rejects_stale(pipeline, cfg, tmp_path):
    from gp_diffuser.training import safety_cache
    local_cfg = json.loads(json.dumps(cfg))
    local_cfg["dataset"] = dict(cfg["dataset"])
    local_cfg["dataset"]["cache_dir"] = str(tmp_path)
    local_cfg["dataset"]["test_size"] = 2
    dataset = SyntheticTrajectoryDataset(local_cfg, "train", size=2)
    info = safety_cache.build_safety_cache(
        dataset, pipeline["net"], pipeline["safety_generator"],
        pipeline["guidance"], local_cfg, split="fp", verbose=False)
    cache = safety_cache.load_safety_cache(local_cfg, "fp")
    assert cache is not None and "cell_is_box" in cache

    meta_path = info["path"].replace(".npz", "_meta.json")
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    meta["fingerprint"] = "stale"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    assert safety_cache.load_safety_cache(local_cfg, "fp") is None

    # a config change alone must also invalidate it
    meta["fingerprint"] = info["meta"]["fingerprint"]
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    local_cfg["corridor"]["local"]["pad_queries"] = 9
    assert safety_cache.load_safety_cache(local_cfg, "fp") is None
