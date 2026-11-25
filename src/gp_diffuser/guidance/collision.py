"""Collision geometry between a continuous trajectory and the occupancy grid.

Conventions (fixed here and relied on everywhere else):

* a grid cell ``(r, c)`` is the unit square centred at ``(r, c)``, i.e. the square
  ``[r-0.5, r+0.5] x [c-0.5, c+0.5]``;
* a trajectory is a float ``(N, 2)`` array of ``(row, col)`` points;
* the trajectory collides as soon as it passes through *any* obstacle cell, so a
  segment whose two endpoints are both free can still collide;
* a point lying exactly on a cell boundary is attributed to the cell with the larger
  coordinate (conservative, deterministic rule; covered by a unit test).
"""

from __future__ import annotations

import math

import numpy as np

from .grid import Grid

EPS = 1e-9
INF = float("inf")


# --------------------------------------------------------------------------- #
# basic polyline utilities
# --------------------------------------------------------------------------- #
def polyline_length(path: np.ndarray) -> float:
    path = np.asarray(path, dtype=np.float64)
    if len(path) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def arc_lengths(path: np.ndarray) -> np.ndarray:
    """Cumulative arc length at every vertex."""
    path = np.asarray(path, dtype=np.float64)
    if len(path) == 0:
        return np.zeros(0)
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def point_at_arc_length(path: np.ndarray, cum: np.ndarray, s: float) -> np.ndarray:
    """Linear interpolation along the polyline at arc length ``s``."""
    path = np.asarray(path, dtype=np.float64)
    if len(path) == 1:
        return path[0].copy()
    s = float(np.clip(s, 0.0, cum[-1]))
    j = int(np.searchsorted(cum, s, side="right") - 1)
    j = max(0, min(j, len(path) - 2))
    seg = cum[j + 1] - cum[j]
    frac = 0.0 if seg <= EPS else (s - cum[j]) / seg
    return path[j] + frac * (path[j + 1] - path[j])


def densify_polyline(path: np.ndarray, max_step: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Dense sampling of the polyline with a step no larger than ``max_step``.

    Returns ``(points, arc)`` where ``arc`` holds the arc length of every point.
    """
    path = np.asarray(path, dtype=np.float64)
    if len(path) < 2:
        return path.copy(), np.zeros(len(path))
    cum = arc_lengths(path)
    out_pts = [path[0]]
    out_arc = [0.0]
    for j in range(len(path) - 1):
        seg = cum[j + 1] - cum[j]
        n = max(1, int(math.ceil(seg / max_step)))
        for k in range(1, n + 1):
            t = k / n
            out_pts.append(path[j] + t * (path[j + 1] - path[j]))
            out_arc.append(cum[j] + t * seg)
    return np.array(out_pts), np.array(out_arc)


def cell_of(point) -> tuple[int, int]:
    """Grid cell that contains a continuous point."""
    return int(round(float(point[0]))), int(round(float(point[1])))


def cell_center(cell) -> np.ndarray:
    return np.array([float(cell[0]), float(cell[1])])


# --------------------------------------------------------------------------- #
# segment <-> cell traversal
# --------------------------------------------------------------------------- #
def trace_segment_cells(p0, p1) -> list[tuple[int, int, float, float]]:
    """Every cell the segment crosses, with the parameter interval spent inside it.

    Amanatides-Woo voxel traversal in the shifted space where cell ``(r, c)`` occupies
    ``[c, c+1) x [r, r+1)``; parameters are normalised to ``[0, 1]`` along the segment.
    A corner crossing is reported as the two edge-adjacent cells (the corner cell is
    touched on a measure-zero set).
    """
    r0, c0 = float(p0[0]), float(p0[1])
    r1, c1 = float(p1[0]), float(p1[1])
    x0, y0 = c0 + 0.5, r0 + 0.5
    x1, y1 = c1 + 0.5, r1 + 0.5
    dx, dy = x1 - x0, y1 - y0
    ix, iy = int(math.floor(x0 + EPS)), int(math.floor(y0 + EPS))
    if abs(dx) < EPS and abs(dy) < EPS:
        return [(iy, ix, 0.0, 0.0)]

    stepx = 1 if dx > EPS else (-1 if dx < -EPS else 0)
    stepy = 1 if dy > EPS else (-1 if dy < -EPS else 0)
    if stepx > 0:
        tmax_x, tdelta_x = (ix + 1 - x0) / dx, 1.0 / dx
    elif stepx < 0:
        tmax_x, tdelta_x = (ix - x0) / dx, -1.0 / dx
    else:
        tmax_x, tdelta_x = INF, INF
    if stepy > 0:
        tmax_y, tdelta_y = (iy + 1 - y0) / dy, 1.0 / dy
    elif stepy < 0:
        tmax_y, tdelta_y = (iy - y0) / dy, -1.0 / dy
    else:
        tmax_y, tdelta_y = INF, INF

    end_x, end_y = int(math.floor(x1 + EPS)), int(math.floor(y1 + EPS))
    guard = 4 * (abs(end_x - ix) + abs(end_y - iy)) + 8
    cells: list[tuple[int, int, float, float]] = []
    t_enter = 0.0
    for _ in range(guard):
        t_exit = min(tmax_x, tmax_y, 1.0)
        cells.append((iy, ix, t_enter, t_exit))
        if t_exit >= 1.0 - EPS:
            break
        if tmax_x < tmax_y:
            t_enter = tmax_x
            ix += stepx
            tmax_x += tdelta_x
        else:
            t_enter = tmax_y
            iy += stepy
            tmax_y += tdelta_y
    return cells


def segment_collision_intervals(grid: Grid, p0, p1) -> list[tuple[float, float]]:
    """Parameter intervals of one segment that run through obstacles."""
    out = []
    for row, col, t_in, t_out in trace_segment_cells(p0, p1):
        if t_out <= t_in:
            continue
        if not (0 <= row < grid.height and 0 <= col < grid.width):
            continue  # leaving the map counts as no collision (validated elsewhere)
        if not grid.free[row, col]:
            out.append((t_in, t_out))
    return _merge_intervals(out)


def _merge_intervals(intervals, gap: float = 1e-7) -> list[tuple[float, float]]:
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [list(ordered[0])]
    for lo, hi in ordered[1:]:
        if lo <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(lo, hi) for lo, hi in merged]


# --------------------------------------------------------------------------- #
# whole-trajectory collision analysis
# --------------------------------------------------------------------------- #
def _free_cell_near(grid: Grid, path, cum, s: float, direction: int) -> tuple[int, int] | None:
    """Nearest free cell found by walking along the trajectory from ``s``."""
    total = float(cum[-1])
    for k in range(1, 41):
        probe = s + direction * 0.05 * k
        probe = float(np.clip(probe, 0.0, total))
        r, c = cell_of(point_at_arc_length(path, cum, probe))
        if 0 <= r < grid.height and 0 <= c < grid.width and grid.free[r, c]:
            return r, c
        if probe in (0.0, total):
            break
    r, c = cell_of(point_at_arc_length(path, cum, s))
    projected = project_to_nearest_free(grid, (r, c), radius=4)
    return projected[0] if projected is not None else None


def detect_collision_intervals(grid: Grid, path: np.ndarray) -> dict:
    """All collision intervals of the raw trajectory, in arc-length coordinates."""
    path = np.asarray(path, dtype=np.float64)
    if len(path) < 2:
        raise ValueError("trajectory needs at least two points")
    if not np.all(np.isfinite(path)):
        raise ValueError("trajectory contains NaN/Inf")
    cum = arc_lengths(path)
    total = float(cum[-1])
    raw: list[tuple[float, float]] = []
    per_segment = []
    for j in range(len(path) - 1):
        seg = cum[j + 1] - cum[j]
        if seg <= EPS:
            per_segment.append([])
            continue
        local = segment_collision_intervals(grid, path[j], path[j + 1])
        per_segment.append(local)
        raw.extend((cum[j] + t0 * seg, cum[j] + t1 * seg) for t0, t1 in local)
    merged = _merge_intervals(raw)

    intervals = []
    for idx, (s_in, s_out) in enumerate(merged):
        entry_point = point_at_arc_length(path, cum, s_in)
        exit_point = point_at_arc_length(path, cum, s_out)
        intervals.append(
            {
                "index": idx,
                "s_in": float(s_in),
                "s_out": float(s_out),
                "u_in": float(s_in / total) if total > 0 else 0.0,
                "u_out": float(s_out / total) if total > 0 else 0.0,
                "length": float(s_out - s_in),
                "entry_point_rc": [float(v) for v in entry_point],
                "exit_point_rc": [float(v) for v in exit_point],
                "entry_cell": [int(v) for v in (_free_cell_near(grid, path, cum, s_in, -1) or cell_of(entry_point))],
                "exit_cell": [int(v) for v in (_free_cell_near(grid, path, cum, s_out, +1) or cell_of(exit_point))],
            }
        )
    collision_length = float(sum(iv["length"] for iv in intervals))
    return {
        "path_length": total,
        "collision_length": collision_length,
        "collision_ratio": float(collision_length / total) if total > 0 else 0.0,
        "collision_interval_count": len(intervals),
        "max_collision_interval_length": float(max((iv["length"] for iv in intervals), default=0.0)),
        "intervals": intervals,
        "arc": cum,
        "per_segment_intervals": per_segment,
    }


def measure_collision_ratio(grid: Grid, path: np.ndarray, step: float = 0.2) -> dict:
    """Collision statistics used to grade a trajectory (``collision_ratio`` is primary)."""
    info = detect_collision_intervals(grid, path)
    points, _arc = densify_polyline(path, step)
    rows = np.rint(points[:, 0]).astype(int)
    cols = np.rint(points[:, 1]).astype(int)
    inside = (rows >= 0) & (rows < grid.height) & (cols >= 0) & (cols < grid.width)
    blocked = np.zeros(len(points), dtype=bool)
    blocked[inside] = ~grid.free[rows[inside], cols[inside]]
    out = {
        "path_length": info["path_length"],
        "collision_length": info["collision_length"],
        "collision_ratio": info["collision_ratio"],
        "collision_interval_count": info["collision_interval_count"],
        "max_collision_interval_length": info["max_collision_interval_length"],
        "collision_sample_ratio": float(blocked.mean()) if len(points) else 0.0,
        "num_detection_samples": int(len(points)),
        "intervals": info["intervals"],
    }
    return out


# --------------------------------------------------------------------------- #
# repair primitive shared by M1 and by entry/exit anchor lookup
# --------------------------------------------------------------------------- #
def project_to_nearest_free(grid: Grid, cell: tuple[int, int], radius: float) -> tuple[tuple[int, int], float] | None:
    """Euclidean-nearest free cell within ``radius``; ``None`` if none qualifies.

    Ties are broken by ``(dr, dc)`` so the result is deterministic.
    """
    r0, c0 = int(round(cell[0])), int(round(cell[1]))
    best = None
    best_d = INF
    r_lo, r_hi = max(0, r0 - int(math.ceil(radius))), min(grid.height - 1, r0 + int(math.ceil(radius)))
    c_lo, c_hi = max(0, c0 - int(math.ceil(radius))), min(grid.width - 1, c0 + int(math.ceil(radius)))
    for r in range(r_lo, r_hi + 1):
        for c in range(c_lo, c_hi + 1):
            if not grid.free[r, c]:
                continue
            d = math.hypot(r - r0, c - c0)
            if d > radius + EPS:
                continue
            if d < best_d - EPS or (abs(d - best_d) <= EPS and best is not None and (r, c) < best):
                best_d = d
                best = (r, c)
    if best is None:
        return None
    return best, float(best_d)


def traversed_obstacle_cells(grid: Grid, path: np.ndarray, min_penetration: float = 0.0) -> list[tuple[int, int, float]]:
    """Obstacle cells the trajectory actually passes through, with how deep it goes.

    ``penetration`` is the path length spent inside the cell.  This is the geometric
    evidence behind a red "collision" segment: a red line drawn over a thin dark seam is
    hard to read, so the figures mark exactly these cells.
    """
    path = np.asarray(path, dtype=np.float64)
    cum = arc_lengths(path)
    found: dict[tuple[int, int], float] = {}
    for j in range(len(path) - 1):
        seg = float(cum[j + 1] - cum[j])
        if seg <= 0:
            continue
        for row, col, t_in, t_out in trace_segment_cells(path[j], path[j + 1]):
            if not (0 <= row < grid.height and 0 <= col < grid.width):
                continue
            if grid.free[row, col]:
                continue
            depth = (t_out - t_in) * seg
            if depth <= min_penetration:
                continue
            key = (row, col)
            found[key] = max(found.get(key, 0.0), depth)
    return [(r, c, d) for (r, c), d in sorted(found.items())]


def count_obstacle_cells(grid: Grid) -> int:
    return int((~grid.free).sum())
