"""Traditional graph-search path pool, arc-length resampling and quality grading.

Nothing here is learned: the whole point of stage 1 is that the *input* trajectory
can come from anywhere (later a diffusion prior), so we only need a spread of
feasible trajectories of known optimality ratio ``rho = L(P) / L(P*)``.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from .config import BUCKETS
from .grid import Grid

INF = float("inf")


# --------------------------------------------------------------------------- #
# single-source graph search
# --------------------------------------------------------------------------- #
def astar(
    grid: Grid,
    start: int,
    goal: int,
    h_weight: float = 1.0,
    noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> list[int] | None:
    """A* with 4-neighbour unit costs.

    ``h_weight > 1`` gives Weighted A* (bound: rho <= h_weight on a consistent
    heuristic).  ``noise`` perturbs the heuristic multiplicatively per node, which
    yields *different but near-optimal* routes -- exactly what is needed to fill the
    Excellent / Good grades with routes that are not the same corridor as A*.
    """
    if start == goal:
        return [start]
    if not (grid.free.ravel()[start] and grid.free.ravel()[goal]):
        return None
    width = grid.width
    gr, gc = divmod(goal, width)
    xi: dict[int, float] = {}

    def h(i: int) -> float:
        r, c = divmod(i, width)
        base = abs(r - gr) + abs(c - gc)
        if noise <= 0.0:
            return base
        if i not in xi:
            xi[i] = 1.0 + noise * float(rng.uniform(-1.0, 1.0))
        return base * xi[i]

    gscore = {start: 0.0}
    parent = {start: -1}
    open_heap = [(h(start) * max(h_weight, 1e-9), start)]
    closed = np.zeros(grid.size, dtype=bool)
    while open_heap:
        _, v = heapq.heappop(open_heap)
        if closed[v]:
            continue
        if v == goal:
            return _reconstruct(parent, goal)
        closed[v] = True
        gv = gscore[v]
        for u in grid.neighbors(v):
            u = int(u)
            if closed[u]:
                continue
            ng = gv + 1.0
            if ng < gscore.get(u, INF):
                gscore[u] = ng
                parent[u] = v
                heapq.heappush(open_heap, (ng + h_weight * h(u), u))
    return None


def greedy_best_first(grid: Grid, start: int, goal: int) -> list[int] | None:
    """f = h only: ignores accumulated cost, so it dives into wrong branches."""
    if start == goal:
        return [start]
    return astar(grid, start, goal, h_weight=1.0e6)


def _reconstruct(parent: dict[int, int], node: int) -> list[int]:
    out = [node]
    while True:
        p = parent[out[-1]]
        if p < 0:
            break
        out.append(p)
    out.reverse()
    return out


def path_length(nodes: list[int]) -> int:
    return len(nodes) - 1


def concat(*parts: list[int]) -> list[int]:
    """Join path pieces, dropping the duplicated junction node."""
    out: list[int] = []
    for part in parts:
        if not part:
            return []
        if out and out[-1] == part[0]:
            out.extend(part[1:])
        else:
            out.extend(part)
    return out


def backtrack_count(nodes: list[int]) -> int:
    """Number of repeated nodes; >0 means the trajectory doubles back on itself."""
    return len(nodes) - len(set(nodes))


# --------------------------------------------------------------------------- #
# arc-length resampling
# --------------------------------------------------------------------------- #
def resample_arc_length(nodes: list[int], width: int, m: int) -> np.ndarray:
    """``m`` samples evenly spaced *along the arc*, snapped to grid nodes.

    Positions are spaced ``L / m`` apart, so a node index can only repeat when the
    trajectory doubles back through the same cell -- a legitimate (and informative)
    property of a poor trajectory, handled by the propagation step.
    """
    if len(nodes) < 2:
        raise ValueError("path needs at least two nodes")
    rc = np.array([divmod(int(i), width) for i in nodes], dtype=np.float64)
    s = np.arange(len(nodes), dtype=np.float64)  # unit edge costs -> arc length
    total = s[-1]
    t = (np.arange(m, dtype=np.float64) + 0.5) * (total / m)
    t[0], t[-1] = 0.0, total
    idx = np.clip(np.searchsorted(s, t, side="right") - 1, 0, len(nodes) - 2)
    frac = (t - s[idx]) / (s[idx + 1] - s[idx])
    pos = rc[idx] + frac[:, None] * (rc[idx + 1] - rc[idx])
    return np.rint(pos).astype(np.int64)


# --------------------------------------------------------------------------- #
# candidate pool
# --------------------------------------------------------------------------- #
@dataclass
class Candidate:
    label: str
    nodes: list[int]
    ratio: float
    backtracks: int
    kind: str

    @property
    def length(self) -> int:
        return len(self.nodes) - 1

    @property
    def backtrack_fraction(self) -> float:
        return self.backtracks / max(len(self.nodes), 1)

    def key(self) -> tuple[int, ...]:
        return tuple(self.nodes)


def _add(pool: list[Candidate], seen: set, label: str, kind: str, nodes: list[int] | None, best_len: int) -> None:
    if not nodes or len(nodes) < 2:
        return
    k = tuple(nodes)
    if k in seen:
        return
    seen.add(k)
    pool.append(
        Candidate(label=label, nodes=nodes, ratio=path_length(nodes) / best_len, backtracks=backtrack_count(nodes), kind=kind)
    )


def build_candidate_pool(
    grid: Grid,
    start: tuple[int, int],
    goal: tuple[int, int],
    best_len: int,
    detour: np.ndarray | None,
    cfg,
    rng: np.random.Generator,
) -> list[Candidate]:
    """All feasible trajectories of the map, generated by traditional search only."""
    s_idx, g_idx = grid.idx(*start), grid.idx(*goal)
    pool: list[Candidate] = []
    seen: set = set()

    _add(pool, seen, "astar", "optimal", astar(grid, s_idx, g_idx), best_len)
    for hw in (1.15, 1.5, 2.0, 3.0):
        _add(pool, seen, f"weighted_astar_{hw:g}", "weighted", astar(grid, s_idx, g_idx, h_weight=hw), best_len)
    _add(pool, seen, "greedy_best_first", "greedy", greedy_best_first(grid, s_idx, g_idx), best_len)

    # near-optimal but topologically different routes
    for k, noise in enumerate((0.05, 0.1, 0.2)):
        for rep in range(6):
            nodes = astar(grid, s_idx, g_idx, h_weight=1.0, noise=noise, rng=rng)
            _add(pool, seen, f"astar_aug_{noise:g}_{rep}", "noisy_astar", nodes, best_len)

    # waypoint-constrained A*: S -> W -> G, W drawn from deep pockets and free space
    free = grid.free_idx
    pocket_cells = []
    if detour is not None:
        flat = detour.ravel()
        good = free[np.isfinite(flat[free]) & (flat[free] >= 10.0)]
        pocket_cells = list(good)
    sources = pocket_cells if pocket_cells else list(free)
    n_wp = cfg.waypoint_candidates
    for k in range(n_wp):
        use_pocket = pocket_cells and (k % 2 == 0)
        pool_cells = pocket_cells if use_pocket else list(free)
        w = int(pool_cells[int(rng.integers(len(pool_cells)))])
        if w in (s_idx, g_idx):
            continue
        leg1 = astar(grid, s_idx, w)
        leg2 = astar(grid, w, g_idx)
        if not leg1 or not leg2:
            continue
        _add(pool, seen, f"waypoint_{k:02d}", "waypoint", concat(leg1, leg2), best_len)

    # two-waypoint chains open up really long detours (needed for the Poor grade)
    for k in range(n_wp // 2):
        w1 = int(free[int(rng.integers(len(free)))])
        w2 = int(free[int(rng.integers(len(free)))])
        if len({w1, w2, s_idx, g_idx}) < 4:
            continue
        legs = [astar(grid, s_idx, w1), astar(grid, w1, w2), astar(grid, w2, g_idx)]
        if not all(legs):
            continue
        _add(pool, seen, f"waypoint2_{k:02d}", "waypoint_chain", concat(*legs), best_len)

    return pool


def select_by_grade(pool: list[Candidate]) -> dict[str, dict]:
    """Pick one candidate per quality grade, grading by rho rather than by label.

    Ties are broken towards trajectories that do *not* double back: a dead-end
    excursion retraces cells the field already covers, so it would show up much less
    clearly than a trajectory that genuinely takes a different corridor.
    """
    def score(c: Candidate, center: float) -> float:
        return abs(c.ratio - center) + 1.5 * c.backtrack_fraction

    chosen: dict[str, Candidate] = {}
    used: set = set()
    for name, lo, hi, center in BUCKETS:
        within = [
            c
            for c in pool
            if c.key() not in used and c.ratio >= lo and (hi is None or c.ratio < hi)
        ]
        if within:
            best = min(within, key=lambda c: (score(c, center), c.length))
            chosen[name] = best
            used.add(best.key())
    # fall back (and flag it) for any grade the pool could not hit exactly
    for name, lo, hi, center in BUCKETS:
        if name in chosen:
            continue
        rest = [c for c in pool if c.key() not in used]
        if not rest:
            continue
        best = min(rest, key=lambda c: (score(c, center), c.length))
        chosen[name] = best
        used.add(best.key())
    out = {}
    for name, lo, hi, center in BUCKETS:
        cand = chosen.get(name)
        if cand is None:
            continue
        in_grade = cand.ratio >= lo and (hi is None or cand.ratio < hi)
        out[name] = {
            "label": cand.label,
            "kind": cand.kind,
            "nodes": cand.nodes,
            "ratio": float(cand.ratio),
            "length": cand.length,
            "num_waypoints": {"waypoint": 1, "waypoint_chain": 2}.get(cand.kind, 0),
            "backtracks": cand.backtracks,
            "backtrack_fraction": float(cand.backtrack_fraction),
            "in_grade": bool(in_grade),
        }
    return out
