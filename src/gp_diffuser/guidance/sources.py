"""Turning a (possibly colliding) raw trajectory into legal free-space diffusion sources.

Three methods, always on the *same* map, the *same* raw trajectory and the *same* progress
definition:

* ``M0`` drop  -- keep only samples that already sit in free space (forward progress kept)
* ``M1`` project -- snap obstacle samples to the Euclidean-nearest free cell (<= radius)
* ``M2`` bridge -- keep free parts of the trajectory, and replace each collision interval
  by a local ROI A* that runs from the interval's entry anchor to its exit anchor; the
  bridge inherits the interval's original progress range

The progress ``u`` of a source is always the arc-length fraction along the **raw**
trajectory.  After dropping/projecting/bridging nothing is re-normalised to the remaining
samples; a collision interval inside the trajectory leaves a genuine hole in the ``u`` axis.
"""

from __future__ import annotations

import numpy as np

from .grid import Grid
from .paths import astar
from .collision import (
    arc_lengths,
    cell_of,
    detect_collision_intervals,
    point_at_arc_length,
    polyline_length,
    project_to_nearest_free,
)

VALID_ORIGINS = ("original_free", "local_bridge")


# --------------------------------------------------------------------------- #
# raw trajectory sampling
# --------------------------------------------------------------------------- #
def sample_raw_trajectory(raw_path: np.ndarray, num_samples: int) -> dict:
    """Arc-length uniform samples of the raw trajectory with their progress ``u``."""
    raw = np.asarray(raw_path, dtype=np.float64)
    cum = arc_lengths(raw)
    total = float(cum[-1])
    s = np.linspace(0.0, total, num_samples)
    points = np.stack([point_at_arc_length(raw, cum, t) for t in s])
    cells = np.rint(points).astype(int)
    return {
        "positions_rc_float": points,
        "cells": cells,
        "progress": s / total if total > 0 else np.zeros(num_samples),
        "arc": s,
        "total_length": total,
        "cum": cum,
    }


def _empty_result(num_samples: int, method: str) -> dict:
    return {
        "positions_rc": np.zeros((0, 2), dtype=int),
        "progress": np.zeros(0, dtype=np.float64),
        "origin": np.zeros(0, dtype=object),
        "interval_id": np.zeros(0, dtype=int),
        "metadata": {
            "method": method,
            "num_raw_samples": int(num_samples),
            "num_valid_sources": 0,
            "valid_source_ratio": 0.0,
            "num_duplicate_source_cells": 0,
            "num_invalid_sources": 0,
            "mean_projection_distance": None,
            "max_projection_distance": None,
            "num_collision_intervals": 0,
            "num_bridged_intervals": None,
            "bridge_success_rate": None,
            "bridge_total_length": None,
            "bridge_detour_ratio": None,
            "roi_expansion_count": None,
            "failures": [],
        },
    }


def _finalise(positions, progress, origin, interval_id, meta, grid: Grid, method: str) -> dict:
    """Drop any non-free source (last safety net) and count it."""
    positions = np.asarray(positions, dtype=int).reshape(-1, 2)
    progress = np.asarray(progress, dtype=np.float64).reshape(-1)
    origin = np.asarray(origin, dtype=object).reshape(-1)
    interval_id = np.asarray(interval_id, dtype=int).reshape(-1)
    keep = np.array(
        [grid.free[r, c] if (0 <= r < grid.height and 0 <= c < grid.width) else False for r, c in positions],
        dtype=bool,
    ) if len(positions) else np.zeros(0, dtype=bool)
    dropped = int((~keep).sum())
    positions, progress = positions[keep], progress[keep]
    origin, interval_id = origin[keep], interval_id[keep]
    cells = [tuple(int(v) for v in p) for p in positions]
    meta = dict(meta)
    meta.update(
        {
            "method": method,
            "num_valid_sources": int(len(positions)),
            "valid_source_ratio": float(len(positions) / meta["num_raw_samples"]) if meta["num_raw_samples"] else 0.0,
            "num_duplicate_source_cells": int(len(cells) - len(set(cells))),
            "num_invalid_sources": int(meta.get("num_invalid_sources", 0) + dropped),
        }
    )
    order = np.argsort(progress, kind="stable")
    return {
        "positions_rc": positions[order],
        "progress": progress[order],
        "origin": origin[order],
        "interval_id": interval_id[order],
        "metadata": meta,
        "support_path": None,
    }


# --------------------------------------------------------------------------- #
# M0 -- drop samples that fall inside obstacles
# --------------------------------------------------------------------------- #
def build_sources_drop(grid: Grid, raw_path: np.ndarray, num_samples: int, cfg) -> dict:
    sampled = sample_raw_trajectory(raw_path, num_samples)
    cells = sampled["cells"]
    progress = sampled["progress"]
    free = np.array(
        [grid.free[r, c] if (0 <= r < grid.height and 0 <= c < grid.width) else False for r, c in cells],
        dtype=bool,
    )
    info = detect_collision_intervals(grid, raw_path)
    meta = {
        "num_raw_samples": int(num_samples),
        "num_invalid_sources": 0,
        "mean_projection_distance": None,
        "max_projection_distance": None,
        "num_collision_intervals": int(info["collision_interval_count"]),
        "num_bridged_intervals": None,
        "bridge_success_rate": None,
        "bridge_total_length": None,
        "bridge_detour_ratio": None,
        "roi_expansion_count": None,
        "failures": [],
    }
    return _finalise(
        cells[free],
        progress[free],
        np.array(["original_free"] * int(free.sum()), dtype=object),
        np.full(int(free.sum()), -1, dtype=int),
        meta,
        grid,
        "M0",
    )


# --------------------------------------------------------------------------- #
# M1 -- project obstacle samples onto the nearest free cell
# --------------------------------------------------------------------------- #
def build_sources_project(grid: Grid, raw_path: np.ndarray, num_samples: int, cfg) -> dict:
    sampled = sample_raw_trajectory(raw_path, num_samples)
    cells = sampled["cells"]
    progress = sampled["progress"]
    positions, origin, interval_id = [], [], []
    progress_kept = []
    distances, failures = [], []
    for (r, c), u in zip(cells, progress):
        in_map = 0 <= r < grid.height and 0 <= c < grid.width
        if in_map and grid.free[r, c]:
            positions.append((int(r), int(c)))
            origin.append("original_free")
            interval_id.append(-1)
            progress_kept.append(float(u))
            distances.append(0.0)
            continue
        projected = project_to_nearest_free(grid, (int(r), int(c)), cfg.projection_radius)
        if projected is None:
            failures.append({"cell": [int(r), int(c)], "reason": "no free cell within projection radius"})
            continue
        (pr, pc), dist = projected
        positions.append((int(pr), int(pc)))
        origin.append("projected")
        interval_id.append(-1)
        progress_kept.append(float(u))
        distances.append(float(dist))
    info = detect_collision_intervals(grid, raw_path)
    meta = {
        "num_raw_samples": int(num_samples),
        "num_invalid_sources": len(failures),
        "mean_projection_distance": float(np.mean(distances)) if distances else None,
        "max_projection_distance": float(np.max(distances)) if distances else None,
        "num_projected_sources": int(sum(1 for o in origin if o == "projected")),
        "num_collision_intervals": int(info["collision_interval_count"]),
        "num_bridged_intervals": None,
        "bridge_success_rate": None,
        "bridge_total_length": None,
        "bridge_detour_ratio": None,
        "roi_expansion_count": None,
        "failures": failures,
    }
    return _finalise(positions, progress_kept, origin, interval_id, meta, grid, "M1")


# --------------------------------------------------------------------------- #
# M2 -- local free-space bridging of collision intervals
# --------------------------------------------------------------------------- #
def _subpath(raw: np.ndarray, cum: np.ndarray, s0: float, s1: float, step: float = 0.5) -> np.ndarray:
    if s1 <= s0:
        return np.array([point_at_arc_length(raw, cum, s0)])
    n = max(2, int(np.ceil((s1 - s0) / step)) + 1)
    return np.stack([point_at_arc_length(raw, cum, t) for t in np.linspace(s0, s1, n)])


def build_local_bridge(grid: Grid, entry, exit_, margins, cfg) -> dict | None:
    """A* between two free anchors inside a progressively enlarged ROI around them."""
    occ = grid.occupancy
    r0, r1 = sorted((int(entry[0]), int(exit_[0])))
    c0, c1 = sorted((int(entry[1]), int(exit_[1])))
    attempts = list(margins) + [None]
    for attempt_index, margin in enumerate(attempts):
        if margin is None:
            rr0, rr1, cc0, cc1 = 0, grid.height - 1, 0, grid.width - 1
        else:
            rr0 = max(0, r0 - margin)
            rr1 = min(grid.height - 1, r1 + margin)
            cc0 = max(0, c0 - margin)
            cc1 = min(grid.width - 1, c1 + margin)
        local_occ = occ[rr0 : rr1 + 1, cc0 : cc1 + 1]
        local_grid = Grid(local_occ)
        le = (int(entry[0]) - rr0, int(entry[1]) - cc0)
        lx = (int(exit_[0]) - rr0, int(exit_[1]) - cc0)
        if not (local_grid.is_free(*le) and local_grid.is_free(*lx)):
            continue
        nodes = astar(local_grid, local_grid.idx(*le), local_grid.idx(*lx))
        if not nodes:
            continue
        pts = np.array(
            [[divmod(int(i), local_grid.width)[0] + rr0, divmod(int(i), local_grid.width)[1] + cc0] for i in nodes],
            dtype=np.float64,
        )
        straight = float(np.linalg.norm(np.array(entry, float) - np.array(exit_, float)))
        return {
            "polyline": pts,
            "roi": [int(rr0), int(rr1), int(cc0), int(cc1)],
            "margin": margin if margin is not None else "full_map",
            "roi_attempt": attempt_index,
            "length": polyline_length(pts),
            "straight_distance": straight,
            "detour_ratio": float(polyline_length(pts) / straight) if straight > 0 else None,
        }
    return None


def resample_support_path_with_progress(pieces, num_samples: int) -> dict:
    """Equal-arc sampling of the stitched support path, with inherited progress."""
    pts: list[np.ndarray] = []
    prog: list[float] = []
    for piece in pieces:
        P = np.asarray(piece["points"], dtype=np.float64)
        if len(P) == 0:
            continue
        u0, u1 = float(piece["u_start"]), float(piece["u_end"])
        frac = np.linspace(0.0, 1.0, len(P))
        pr = u0 + frac * (u1 - u0)
        if pts and np.linalg.norm(P[0] - pts[-1]) < 1e-9:
            pts.extend(P[1:])
            prog.extend(pr[1:])
        else:
            pts.extend(P)
            prog.extend(pr)
    if len(pts) < 2:
        raise ValueError("support path has fewer than two distinct points")
    P = np.asarray(pts, dtype=np.float64)
    U = np.maximum.accumulate(np.asarray(prog, dtype=np.float64))  # progress never goes back
    cum = arc_lengths(P)
    s = np.linspace(0.0, cum[-1], num_samples)
    out_pts = np.stack([point_at_arc_length(P, cum, t) for t in s])
    out_u = np.interp(s, cum, U)
    # origin: which piece each sample came from
    piece_id = np.searchsorted(np.concatenate([[0.0], cum]), s, side="right") - 1
    piece_id = np.clip(piece_id, 0, len(P) - 1)
    return {"points": out_pts, "progress": out_u, "support": P, "support_progress": U, "cum": cum, "piece_id": piece_id}


def build_sources_bridge(grid: Grid, raw_path: np.ndarray, num_samples: int, cfg) -> dict:
    raw = np.asarray(raw_path, dtype=np.float64)
    cum = arc_lengths(raw)
    total = float(cum[-1])
    info = detect_collision_intervals(grid, raw)
    intervals = info["intervals"]
    delta = 0.05

    pieces: list[dict] = []
    bridge_meta = []
    failures = []
    cursor = 0.0
    cursor_point = raw[0]
    bridged = 0
    for iv in intervals:
        free_end = max(cursor, iv["s_in"] - delta)
        free_pts = _subpath(raw, cum, cursor, free_end)
        entry_center = np.array(iv["entry_cell"], dtype=np.float64)
        free_pts = np.vstack([free_pts, entry_center])
        pieces.append({"points": free_pts, "u_start": cursor / total, "u_end": iv["u_in"]})

        bridge = build_local_bridge(grid, iv["entry_cell"], iv["exit_cell"], cfg.bridge_roi_margins, cfg)
        if bridge is None:
            failures.append({"interval": iv["index"], "reason": "no local bridge even at full map"})
            exit_center = np.array(iv["exit_cell"], dtype=np.float64)
            pieces.append(
                {"points": np.vstack([entry_center, exit_center]), "u_start": iv["u_in"], "u_end": iv["u_out"]}
            )
        else:
            bridged += 1
            pieces.append({"points": bridge["polyline"], "u_start": iv["u_in"], "u_end": iv["u_out"]})
            bridge_meta.append(
                {
                    "interval": iv["index"],
                    "u_in": round(iv["u_in"], 4),
                    "u_out": round(iv["u_out"], 4),
                    "roi": bridge["roi"],
                    "margin": bridge["margin"],
                    "roi_attempt": bridge["roi_attempt"],
                    "length": round(bridge["length"], 3),
                    "straight_distance": round(bridge["straight_distance"], 3),
                    "detour_ratio": None if bridge["detour_ratio"] is None else round(bridge["detour_ratio"], 3),
                }
            )
        cursor = min(total, iv["s_out"] + delta)
        cursor_point = np.array(iv["exit_cell"], dtype=np.float64)
    tail = _subpath(raw, cum, cursor, total)
    pieces.append({"points": np.vstack([cursor_point, tail]), "u_start": min(1.0, cursor / total), "u_end": 1.0})

    resampled = resample_support_path_with_progress(pieces, num_samples)
    cells = np.rint(resampled["points"]).astype(int)
    # origin per source: any source whose progress lies inside a bridge interval is a bridge source
    origin = np.array(["original_free"] * len(cells), dtype=object)
    interval_id = np.full(len(cells), -1, dtype=int)
    for b in bridge_meta:
        inside = (resampled["progress"] > b["u_in"] + 1e-9) & (resampled["progress"] < b["u_out"] - 1e-9)
        origin[inside] = "local_bridge"
        interval_id[inside] = b["interval"]

    total_bridge = sum(b["length"] for b in bridge_meta)
    total_detour = [b["detour_ratio"] for b in bridge_meta if b["detour_ratio"] is not None]
    meta = {
        "num_raw_samples": int(num_samples),
        "num_invalid_sources": 0,
        "mean_projection_distance": None,
        "max_projection_distance": None,
        "num_collision_intervals": int(len(intervals)),
        "num_bridged_intervals": int(bridged),
        "bridge_success_rate": float(bridged / len(intervals)) if intervals else None,
        "bridge_total_length": round(float(total_bridge), 3) if bridge_meta else None,
        "bridge_detour_ratio": round(float(np.mean(total_detour)), 3) if total_detour else None,
        "roi_expansion_count": int(sum(b["roi_attempt"] for b in bridge_meta)),
        "failures": failures,
        "bridges": bridge_meta,
        "support_length": round(float(polyline_length(resampled["support"])), 3),
        "raw_length": round(total, 3),
    }
    result = _finalise(cells, resampled["progress"], origin, interval_id, meta, grid, "M2")
    result["support_path"] = resampled["support"]
    return result


BUILDERS = {
    "M0": build_sources_drop,
    "M1": build_sources_project,
    "M2": build_sources_bridge,
}


def build_all_methods(grid: Grid, raw_path: np.ndarray, cfg) -> dict[str, dict]:
    """Run M0/M1/M2 on the same trajectory with identical progress and sample count."""
    return {name: BUILDERS[name](grid, raw_path, cfg.num_source_samples, cfg) for name in ("M0", "M1", "M2")}
