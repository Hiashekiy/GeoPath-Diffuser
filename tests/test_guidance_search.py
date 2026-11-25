"""Guidance-field search tests (design document section 13.3)."""
from __future__ import annotations

import numpy as np

from gp_diffuser.guidance.search import GuidanceFieldSearch, resample_polyline


def _occ_with_wall(height=128, width=128, gap_row=64, gap=10):
    occ = np.zeros((height, width), dtype=np.int8)
    occ[:, width // 2] = 1
    occ[gap_row - gap // 2:gap_row + gap // 2, width // 2] = 0
    return occ


def test_search_path_is_free_and_resampled(cfg):
    res_m = float(cfg["dataset"]["res_m"])
    occ = _occ_with_wall()
    search = GuidanceFieldSearch(cfg["guidance"], res_m=res_m)
    start = np.array([5.0, 16.0])
    goal = np.array([27.0, 16.0])
    guide = np.stack([np.linspace(start[0], goal[0], 64), np.full(64, start[1])], axis=-1)
    result = search.search(occ, guide, start, goal)
    assert result["status"] == "success"
    path = result["path_xy"]
    assert path.shape == (128, 2)
    assert np.allclose(path[0], start) and np.allclose(path[-1], goal)
    cells = np.rint(path / res_m).astype(int)
    assert occ[cells[:, 1], cells[:, 0]].sum() == 0, "searched path must stay in free space"


def test_arc_length_resampling_preserves_endpoints():
    line = np.stack([np.linspace(0.0, 4.0, 40), np.zeros(40)], axis=-1)
    out = resample_polyline(line, 64)
    assert out.shape == (64, 2)
    assert np.allclose(out[0], line[0]) and np.allclose(out[-1], line[-1])
    seg = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert seg.std() / seg.mean() < 1e-4, "a straight line must resample uniformly"

    # a corner turns one chord slightly shorter than the arc it spans
    corner = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 3.0]])
    out = resample_polyline(corner, 64)
    assert np.allclose(out[0], corner[0]) and np.allclose(out[-1], corner[-1])
    seg = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert seg.std() / seg.mean() < 0.1


def _two_wall_map(height=128, width=128, gaps=((52, 62), (92, 102))):
    """Two vertical walls, each with two openings (upper / lower)."""
    occ = np.zeros((height, width), dtype=np.int8)
    for col in (40, 84):
        occ[:, col:col + 4] = 1
        for lo, hi in gaps:
            occ[lo:hi, col:col + 4] = 0
    return occ


def test_fallback_to_plain_astar_when_guidance_is_absent(cfg):
    """A guide trajectory that lies inside obstacles still yields a feasible path."""
    res_m = float(cfg["dataset"]["res_m"])
    occ = _two_wall_map()
    search = GuidanceFieldSearch(cfg["guidance"], res_m=res_m)
    start = np.array([4.0, 4.0])
    goal = np.array([27.0, 27.0])
    # guide parked entirely inside the first wall (illegal as a source)
    guide = np.stack([np.full(48, 10.2), np.linspace(4.0, 12.0, 48)], axis=-1)
    result = search.search(occ, guide, start, goal)
    assert result["path_xy"] is not None
    assert result["status"] == "success"
    assert result["valid"]


def test_unreachable_map_returns_no_path(cfg):
    res_m = float(cfg["dataset"]["res_m"])
    occ = np.zeros((64, 64), dtype=np.int8)
    occ[:, 32] = 1                          # solid wall, no gap
    search = GuidanceFieldSearch(cfg["guidance"], res_m=res_m)
    start = np.array([4.0, 4.0])
    goal = np.array([12.0, 4.0])
    guide = np.stack([np.linspace(4.0, 12.0, 32), np.full(32, 4.0)], axis=-1)
    result = search.search(occ, guide, start, goal)
    assert result["path_xy"] is None
    assert result["status"] in ("unreachable", "invalid_endpoints")


def test_guidance_field_follows_the_guide_corridor(cfg):
    """Both openings are feasible; the guide decides which one is taken."""
    res_m = float(cfg["dataset"]["res_m"])
    occ = _two_wall_map()                  # openings around y = 13 m and y = 24 m
    search = GuidanceFieldSearch(cfg["guidance"], res_m=res_m)
    start = np.array([4.0, 4.0])
    goal = np.array([27.0, 27.0])
    upper = np.stack([np.linspace(4.0, 27.0, 96),
                      np.concatenate([np.linspace(4.0, 13.0, 48),
                                      np.linspace(13.0, 27.0, 48)])], axis=-1)
    result = search.search(occ, upper, start, goal)
    assert result["path_xy"] is not None
    path = result["path_xy"]
    # crossing the first wall (x = 10..11 m) must happen near the UPPER opening
    crossing = path[(path[:, 0] > 9.0) & (path[:, 0] < 12.5)]
    assert len(crossing) > 0
    assert np.abs(crossing[:, 1] - 13.0).min() < 3.0
