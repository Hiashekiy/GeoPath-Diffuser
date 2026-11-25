"""Priority policies and the best-first search loops (plan §5, §7).

Every policy exposes ``priority(r, c)`` computed from precomputed scalar maps only.  Ball
``B0`` is the exception: it is traditional A* with ``g + h`` and lives in its own loop.
The non-``g`` policies physically cannot see ``g``, which is what makes "B1-B5 must not use
the accumulated cost" a property that can be tested rather than trusted.
"""

from __future__ import annotations

import heapq
import time

import numpy as np

from .grid import Grid

NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1))  # fixed traversal order (reproducible)
INF = float("inf")


# --------------------------------------------------------------------------- #
# policies
# --------------------------------------------------------------------------- #
class PriorityPolicy:
    """J(n) from precomputed maps only -- no access to g(n) by construction."""

    name = "base"
    uses_field = True
    uses_distance = True

    def __init__(self, scalars: dict, **params):
        self.F = scalars["F_hat"]
        self.D = scalars["d_hat"]
        self.params = params

    def priority(self, r: int, c: int) -> float:  # pragma: no cover - overridden
        raise NotImplementedError

    def describe(self) -> dict:
        return {"method": self.name, **self.params}


class DistancePolicy(PriorityPolicy):
    """B1: J = d_hat."""

    name = "B1_distance_gbfs"
    uses_field = False

    def priority(self, r, c):
        return float(self.D[r, c])


class FieldOnlyPolicy(PriorityPolicy):
    """B2: J = -F_hat."""

    name = "B2_field_only"
    uses_distance = False

    def priority(self, r, c):
        return -float(self.F[r, c])


class LinearFusionPolicy(PriorityPolicy):
    """B3: J = alpha*d_hat - (1-alpha)*F_hat."""

    name = "B3_linear_fusion"

    def __init__(self, scalars, alpha: float, **params):
        super().__init__(scalars, alpha=alpha, **params)
        self.alpha = float(alpha)

    def priority(self, r, c):
        return self.alpha * float(self.D[r, c]) - (1.0 - self.alpha) * float(self.F[r, c])


class NonlinearFusionPolicy(PriorityPolicy):
    """B4: J = alpha*d_hat**gamma - (1-alpha)*F_hat."""

    name = "B4_nonlinear_fusion"

    def __init__(self, scalars, alpha: float, gamma: float, **params):
        super().__init__(scalars, alpha=alpha, gamma=gamma, **params)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        # gamma < 1 amplifies distance differences close to the goal (plan §5 B4)
        self.D_pow = np.power(np.clip(self.D, 0.0, 1.0), self.gamma)

    def priority(self, r, c):
        return self.alpha * float(self.D_pow[r, c]) - (1.0 - self.alpha) * float(self.F[r, c])


class AdaptiveFusionPolicy(PriorityPolicy):
    """B5: alpha(r,c) grows as the node approaches the goal."""

    name = "B5_adaptive_fusion"

    def __init__(self, scalars, alpha_far: float, alpha_near: float, eta: float, gamma: float, **params):
        super().__init__(scalars, alpha_far=alpha_far, alpha_near=alpha_near, eta=eta, gamma=gamma, **params)
        self.D_pow = np.power(np.clip(self.D, 0.0, 1.0), float(gamma))
        self.alpha_map = alpha_far + (alpha_near - alpha_far) * np.power(1.0 - self.D, float(eta))

    def priority(self, r, c):
        a = float(self.alpha_map[r, c])
        return a * float(self.D_pow[r, c]) - (1.0 - a) * float(self.F[r, c])


POLICY_CLASSES = {
    "B1_distance_gbfs": DistancePolicy,
    "B2_field_only": FieldOnlyPolicy,
    "B3_linear_fusion": LinearFusionPolicy,
    "B4_nonlinear_fusion": NonlinearFusionPolicy,
    "B5_adaptive_fusion": AdaptiveFusionPolicy,
}


# --------------------------------------------------------------------------- #
# search loops
# --------------------------------------------------------------------------- #
def _reconstruct(parent: dict, node: int) -> list[int]:
    out = [node]
    while parent[out[-1]] >= 0:
        out.append(parent[out[-1]])
    out.reverse()
    return out


def astar_search(grid: Grid, start: tuple[int, int], goal: tuple[int, int], cutoff: int = 0) -> dict:
    """B0 baseline: f = g + h, unit edge cost, Manhattan heuristic (admissible)."""
    t0 = time.perf_counter()
    s_idx, g_idx = grid.idx(*start), grid.idx(*goal)
    gr, gc = goal
    width = grid.width

    def h(i):
        r, c = divmod(i, width)
        return abs(r - gr) + abs(c - gc)

    gscore = {s_idx: 0.0}
    parent = {s_idx: -1}
    closed = np.zeros(grid.size, dtype=bool)
    counter = 0
    open_heap = [(h(s_idx), 0, s_idx)]
    expanded = 0
    status = "unreachable"
    trace = []
    while open_heap:
        _, _, v = heapq.heappop(open_heap)
        if closed[v]:
            continue
        closed[v] = True
        expanded += 1
        trace.append(v)
        if cutoff and expanded > cutoff:
            status = "cutoff"
            break
        if v == g_idx:
            status = "success"
            break
        gv = gscore[v]
        for u in grid.neighbors(int(v)):
            u = int(u)
            if closed[u]:
                continue
            ng = gv + 1.0
            if ng < gscore.get(u, INF):
                gscore[u] = ng
                parent[u] = v
                counter += 1
                heapq.heappush(open_heap, (ng + h(u), counter, u))
    elapsed = time.perf_counter() - t0
    path = _reconstruct(parent, g_idx) if status == "success" else []
    return {
        "status": status,
        "expanded": expanded,
        "generated": len(parent),
        "search_time": elapsed,
        "path": path,
        "g_cost": gscore.get(g_idx) if status == "success" else None,
        "trace": trace,
    }


def priority_search(
    grid: Grid,
    start: tuple[int, int],
    goal: tuple[int, int],
    policy: PriorityPolicy,
    cutoff: int = 0,
    record_trace: bool = True,
) -> dict:
    """Generic best-first search for B1-B5: pop the smallest J, record the first discovery.

    No ``g`` value is computed or consulted: a node is enqueued once, on first discovery,
    which is exactly the node management the plan asks for (§7.2).
    """
    t0 = time.perf_counter()
    s_idx, g_idx = grid.idx(*start), grid.idx(*goal)
    width = grid.width
    parent = {s_idx: -1}
    queued = {s_idx: float(policy.priority(*divmod(s_idx, width)))}
    counter = 0
    open_heap = [(queued[s_idx], 0, s_idx)]
    closed = np.zeros(grid.size, dtype=bool)
    expanded = 0
    trace = []
    status = "unreachable"
    while open_heap:
        _, _, v = heapq.heappop(open_heap)
        if closed[v]:
            continue
        closed[v] = True
        expanded += 1
        if record_trace:
            trace.append(v)
        if cutoff and expanded > cutoff:
            status = "cutoff"
            break
        if v == g_idx:
            status = "success"
            break
        r, c = divmod(v, width)
        for dr, dc in NEIGHBOURS:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < grid.height and 0 <= cc < grid.width):
                continue
            if not grid.free[rr, cc]:
                continue
            u = rr * width + cc
            if u in parent:
                continue
            parent[u] = v
            counter += 1
            j = float(policy.priority(rr, cc))
            queued[u] = j
            heapq.heappush(open_heap, (j, counter, u))
    elapsed = time.perf_counter() - t0
    path = _reconstruct(parent, g_idx) if status == "success" else []
    return {
        "status": status,
        "expanded": expanded,
        "generated": len(parent),
        "search_time": elapsed,
        "path": path,
        "g_cost": None,
        "trace": trace,
    }


def dijkstra_length(grid: Grid, start: tuple[int, int], goal: tuple[int, int]) -> float | None:
    """Shortest path length for offline evaluation only (never fed to the searchers)."""
    dist = {grid.idx(*start): 0.0}
    heap = [(0.0, grid.idx(*start))]
    closed = set()
    g_idx = grid.idx(*goal)
    while heap:
        d, v = heapq.heappop(heap)
        if v in closed:
            continue
        closed.add(v)
        if v == g_idx:
            return d
        for u in grid.neighbors(int(v)):
            u = int(u)
            if u in closed:
                continue
            nd = d + 1.0
            if nd < dist.get(u, INF):
                dist[u] = nd
                heapq.heappush(heap, (nd, u))
    return None


def path_is_valid(grid: Grid, path: list[int], start: tuple[int, int], goal: tuple[int, int]) -> dict:
    """Continuity + obstacle + endpoint check on a returned path."""
    if not path:
        return {"valid": False, "reason": "empty path"}
    if path[0] != grid.idx(*start) or path[-1] != grid.idx(*goal):
        return {"valid": False, "reason": "endpoints do not match S/G"}
    rows_cols = [divmod(int(i), grid.width) for i in path]
    for r, c in rows_cols:
        if not grid.free[r, c]:
            return {"valid": False, "reason": f"cell {(r, c)} is an obstacle"}
    for (r0, c0), (r1, c1) in zip(rows_cols, rows_cols[1:]):
        if abs(r0 - r1) + abs(c0 - c1) != 1:
            return {"valid": False, "reason": "path is not 4-connected"}
    return {"valid": True, "length": len(rows_cols) - 1, "collision_free": True}
