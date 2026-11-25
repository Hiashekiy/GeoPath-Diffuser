"""ALM tests: reuse, dual inheritance, masking, dense verification."""
from __future__ import annotations

import numpy as np
import torch

from gp_diffuser.diffusion.bspline_alm import bspline_alm_correct
from gp_diffuser.safety.local_regions import build_region_table, pack_from_tables


def _box_table(margin=0.0, valid=True):
    from gp_diffuser.safety.local_regions import box_cell_faces
    A, b = box_cell_faces(margin)
    cell_A = np.zeros((32, 4, 2))
    cell_b = np.full((32, 4), np.inf)
    mask = np.zeros(32, dtype=bool)
    if valid:
        cell_A[:] = A
        cell_b[:] = b
        mask[:] = True
    return {"cell_A": cell_A, "cell_b": cell_b, "cell_mask": mask,
            "cell_is_box": np.ones(32, dtype=bool),
            "anchors": np.linspace(0.0, 1.0, 32)}


def _box_pack(codec, margin=0.0, valid=True):
    return pack_from_tables(codec, [_box_table(margin, valid)], margin=0.0)


def test_alm_reduces_violation(pipeline):
    codec = pipeline["codec"]
    pack = _box_pack(codec)
    q = torch.randn(1, 32, 2) * 0.05
    q[0, 8:16] = torch.tensor([1.45, 0.0])          # clearly outside the map box
    cfg = dict(pipeline["cfg"]["alm"])
    q_safe, lam, stats = bspline_alm_correct(q, pack, codec, lam=None, config=cfg,
                                             inner_steps=6)
    before = float(stats["max_violation_before"].mean())
    after = float(stats["max_violation_after"].mean())
    assert before > 0.2
    assert after < before
    assert float(stats["mean_curve_correction_m"].mean()) > 0.0


def test_alm_keeps_endpoints(pipeline):
    codec = pipeline["codec"]
    pack = _box_pack(codec)
    q = torch.randn(1, 32, 2) * 0.05
    q[0, 5:12] = torch.tensor([1.3, 0.2])
    q_safe, _, _ = bspline_alm_correct(q, pack, codec, config=dict(pipeline["cfg"]["alm"]),
                                       inner_steps=4)
    assert torch.allclose(q_safe[:, 0], q[:, 0])
    assert torch.allclose(q_safe[:, -1], q[:, -1])


def test_dual_variables_are_returned_for_inheritance(pipeline):
    codec = pipeline["codec"]
    pack = _box_pack(codec)
    q = torch.randn(1, 32, 2) * 0.05
    q[0, 10:20] = torch.tensor([1.5, 0.0])
    cfg = dict(pipeline["cfg"]["alm"])
    _, lam1, _ = bspline_alm_correct(q, pack, codec, lam=None, config=cfg, inner_steps=5)
    assert float(lam1.abs().sum()) > 0, "active constraints must produce a dual"
    # warm start with the inherited dual must run and stay finite
    q2 = q.clone()
    q_safe2, lam2, stats2 = bspline_alm_correct(q2, pack, codec, lam=lam1,
                                                config=cfg, inner_steps=5)
    assert torch.isfinite(q_safe2).all() and torch.isfinite(lam2).all()
    assert float(stats2["max_violation_after"].mean()) <= float(
        stats2["max_violation_before"].mean()) + 1e-6


def test_fully_masked_pack_is_a_no_op(pipeline):
    """No cell -> no pack; an explicitly empty pack leaves the controls untouched."""
    from gp_diffuser.safety.pack_utils import empty_pack
    codec = pipeline["codec"]
    assert _box_pack(codec, valid=False) is None, "an all-masked table has no pack"
    pack = empty_pack(codec.num_controls)
    q = torch.randn(1, 32, 2)
    q_safe, lam, stats = bspline_alm_correct(q, pack, codec,
                                             config=dict(pipeline["cfg"]["alm"]),
                                             inner_steps=5)
    assert torch.allclose(q_safe, q)
    assert float(lam.abs().sum()) == 0.0
    assert float(stats["max_violation_after"].mean()) == 0.0


def test_padding_rows_are_ignored(pipeline):
    """Adding masked piece rows must not change the corrected controls."""
    codec = pipeline["codec"]
    pack = _box_pack(codec)
    q = torch.randn(1, 32, 2) * 0.05
    q[0, 12:18] = torch.tensor([1.35, -0.1])
    cfg = dict(pipeline["cfg"]["alm"])
    q_a, _, _ = bspline_alm_correct(q, pack, codec, config=cfg, inner_steps=5)

    padded = type(pack)(**{f: getattr(pack, f) for f in (
        "extraction", "piece_A", "piece_b", "face_mask", "piece_mask",
        "piece_region_id", "intervals", "num_pieces")})
    extra = 7
    padded.extraction = torch.cat(
        [padded.extraction, torch.zeros(1, extra, 4, padded.extraction.shape[-1])], dim=1)
    padded.piece_A = torch.cat(
        [padded.piece_A, torch.zeros(1, extra, padded.piece_A.shape[2], 2)], dim=1)
    padded.piece_b = torch.cat(
        [padded.piece_b, torch.zeros(1, extra, padded.piece_b.shape[2])], dim=1)
    padded.face_mask = torch.cat(
        [padded.face_mask, torch.zeros(1, extra, padded.face_mask.shape[2], dtype=torch.bool)],
        dim=1)
    padded.piece_mask = torch.cat(
        [padded.piece_mask, torch.zeros(1, extra, dtype=torch.bool)], dim=1)
    padded.piece_region_id = torch.cat(
        [padded.piece_region_id, torch.full((1, extra), -1, dtype=torch.long)], dim=1)
    padded.intervals = torch.cat(
        [padded.intervals, torch.zeros(1, extra, 2)], dim=1)
    q_b, _, _ = bspline_alm_correct(q, padded, codec, config=cfg, inner_steps=5)
    assert torch.allclose(q_a, q_b, atol=1e-6)


def test_dense_validation_catches_collisions(pipeline, sample):
    item, _ = sample
    occ = item["occupancy"][None]
    good = item["path_gt_scene"][None]
    sampler = pipeline["sampler"]
    assert float(sampler._collision_fractions(occ, good)[0]) == 0.0
    occ_np = item["occupancy"][0].numpy()
    blocked = np.argwhere(occ_np > 0.5)
    grid = occ_np.shape[-1]
    row, col = blocked[len(blocked) // 3]
    bad = torch.tensor([2.0 * col / grid - 1.0 + 1.0 / grid,
                        2.0 * row / grid - 1.0 + 1.0 / grid]).repeat(1, 128, 1)
    assert float(sampler._collision_fractions(occ, bad)[0]) == 1.0


def test_pack_from_region_table_matches_batch_build(pipeline, sample, cfg):
    item, _ = sample
    out = pipeline["safety_generator"](item["occupancy"][None], item["sdf"][None],
                                       item["path_gt_m"][None])
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
    pack_single = pack_from_tables(pipeline["codec"], [table])
    pack_pair = pack_from_tables(pipeline["codec"], [table, table])
    assert pack_single.num_pieces[0].item() == pack_pair.num_pieces[0].item()
    assert torch.allclose(pack_single.piece_b[0], pack_pair.piece_b[0])
