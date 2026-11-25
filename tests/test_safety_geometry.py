"""Safety-geometry tests: ellipses, local convex regions, constraint pack."""
from __future__ import annotations

import numpy as np
import torch

from gp_diffuser.frames import SceneTransform
from gp_diffuser.safety.local_regions import (build_region_table, collision_intervals,
                                              collision_mask, pad_region_tables,
                                              pack_from_tables)


def _safety(pipeline, item):
    occ = item["occupancy"][None]
    sdf = item["sdf"][None]
    path_m = item["path_gt_m"][None]
    return pipeline["safety_generator"](occ, sdf, path_m)


def test_ellipse_generation_uses_the_frozen_network(pipeline, sample):
    item, _ = sample
    out = _safety(pipeline, item)
    assert out["ellipse"].shape == (1, 64, 6)
    assert out["ellipse_center"].shape == (1, 64, 2)
    assert out["ellipse_shape4"].shape == (1, 64, 4)
    # cos2t / sin2t is a unit vector
    norm = (out["ellipse"][..., 4] ** 2 + out["ellipse"][..., 5] ** 2).sqrt()
    assert float((norm - 1.0).abs().max()) < 1e-5
    # axes obey the network's own bounds
    a = out["ellipse"][..., 2]
    b = out["ellipse"][..., 3]
    assert float(a.min()) >= 0.15 - 1e-4 and float(a.max()) <= 6.0 + 1e-4
    assert float((a - b).min()) >= -1e-4
    assert float(b.min()) >= 0.05 - 1e-4


def test_ellipse_centres_lie_on_the_search_path(pipeline, sample):
    item, _ = sample
    out = _safety(pipeline, item)
    query = out["query"][0].numpy()
    assert query.shape == (64, 2)
    # arc-length querying must hit both endpoints exactly
    assert np.allclose(query[0], item["path_gt_m"][0].numpy(), atol=1e-3)
    assert np.allclose(query[-1], item["path_gt_m"][-1].numpy(), atol=1e-3)


def test_no_collision_means_box_cells_only(pipeline, sample, cfg):
    item, _ = sample
    out = _safety(pipeline, item)
    transform = pipeline["transform"]
    centers = transform.to_scene(out["ellipse_center"][0].numpy())
    anchors = torch.linspace(0.0, 1.0, 64, dtype=torch.float64)
    table, info = build_region_table(
        item["occupancy"][None], torch.as_tensor(centers, dtype=torch.float32),
        out["ellipse_shape4"][0], anchors,
        curve_scene=item["path_gt_scene"][None],
        gamma_scene=torch.as_tensor(transform.to_scene(item["path_gt_m"].numpy()),
                                    dtype=torch.float32),
        config=cfg["corridor"], scope="local")
    assert info["spans"] == [], "a collision-free curve must not trigger local regions"
    assert info["box_cells"] == 64
    assert float(np.isfinite(table["cell_b"]).sum()) == 64 * 4


def test_colliding_curve_triggers_local_spans(pipeline, sample, cfg):
    """A curve pushed into an obstacle must produce local spans (padded)."""
    item, _ = sample
    out = _safety(pipeline, item)
    transform = pipeline["transform"]
    centers = transform.to_scene(out["ellipse_center"][0].numpy())
    anchors = torch.linspace(0.0, 1.0, 64, dtype=torch.float64)
    # push the middle of the curve into the nearest blocked cell
    curve = item["path_gt_scene"].clone()
    occ = item["occupancy"][0].numpy()
    grid = occ.shape[-1]
    blocked = np.argwhere(occ > 0.5)
    mid = blocked[len(blocked) // 2]
    scene_pt = torch.tensor([2.0 * mid[1] / grid - 1.0 + 1.0 / grid,
                             2.0 * mid[0] / grid - 1.0 + 1.0 / grid],
                            dtype=torch.float32)
    curve[55:70] = scene_pt
    hit, _ = collision_mask(item["occupancy"][None], curve)
    assert hit.any()
    spans = collision_intervals(hit, pad=4, min_q=8, n_query=64)
    assert spans, "collision intervals must map to query spans"
    for a, b in spans:
        assert b - a + 1 >= 8, "min_queries must widen short spans"
    table, info = build_region_table(
        item["occupancy"][None], torch.as_tensor(centers, dtype=torch.float32),
        out["ellipse_shape4"][0], anchors, curve_scene=curve[None],
        gamma_scene=torch.as_tensor(transform.to_scene(item["path_gt_m"].numpy()),
                                    dtype=torch.float32),
        config=cfg["corridor"], scope="local")
    assert info["spans"], "local scope must report the collision spans"


def test_pack_masks_invalid_faces(pipeline, sample, cfg):
    item, _ = sample
    out = _safety(pipeline, item)
    transform = pipeline["transform"]
    centers = transform.to_scene(out["ellipse_center"][0].numpy())
    anchors = torch.linspace(0.0, 1.0, 64, dtype=torch.float64)
    table, _ = build_region_table(
        item["occupancy"][None], torch.as_tensor(centers, dtype=torch.float32),
        out["ellipse_shape4"][0], anchors,
        curve_scene=item["path_gt_scene"][None],
        gamma_scene=torch.as_tensor(transform.to_scene(item["path_gt_m"].numpy()),
                                    dtype=torch.float32),
        config=cfg["corridor"], scope="full")
    pack = pack_from_tables(pipeline["codec"], [table])
    assert pack.piece_mask.any()
    # every unmasked face must have a finite normal and a finite rhs
    A = pack.piece_A[pack.face_mask]
    b = pack.piece_b[pack.face_mask]
    assert torch.isfinite(A).all() and torch.isfinite(b).all()
    # unit normals (the region builder normalises every separator)
    assert float((A.norm(dim=-1) - 1.0).abs().max()) < 1e-4


def test_padding_does_not_change_the_pack(pipeline, sample, cfg):
    """Per-sample tables with different face counts must stack without effect."""
    item, _ = sample
    out = _safety(pipeline, item)
    transform = pipeline["transform"]
    centers = transform.to_scene(out["ellipse_center"][0].numpy())
    anchors = torch.linspace(0.0, 1.0, 64, dtype=torch.float64)
    table, _ = build_region_table(
        item["occupancy"][None], torch.as_tensor(centers, dtype=torch.float32),
        out["ellipse_shape4"][0], anchors,
        curve_scene=item["path_gt_scene"][None],
        gamma_scene=torch.as_tensor(transform.to_scene(item["path_gt_m"].numpy()),
                                    dtype=torch.float32),
        config=cfg["corridor"], scope="full")
    small = {"cell_A": table["cell_A"][:, :4], "cell_b": table["cell_b"][:, :4],
             "cell_mask": table["cell_mask"], "anchors": table["anchors"]}
    stacked = pad_region_tables([small, table])
    assert stacked["cell_A"].shape[0] == 2
    assert stacked["cell_A"].shape[2] == table["cell_A"].shape[1]
    # the padded rows carry b=+inf, so they can never bind
    padded = stacked["cell_b"][0, :, 4:]
    assert torch.isinf(padded).all()


def test_scene_transform_roundtrip():
    tf = SceneTransform(64.0, 256)
    pts = np.array([[0.0, 0.0], [32.0, 32.0], [63.75, 63.75], [10.25, 40.5]])
    assert np.allclose(tf.to_meters(tf.to_scene(pts)), pts)
    # a cell centre maps onto the grid_sample coordinate of that cell
    col = 100
    scene_x = tf.to_scene(np.array([[col * 0.25, 0.0]]))[0, 0]
    assert abs(scene_x - (2.0 * (col + 0.5) / 256 - 1.0)) < 1e-9
