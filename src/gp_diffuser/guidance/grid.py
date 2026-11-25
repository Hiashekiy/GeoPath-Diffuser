"""Occupancy-grid primitives: 4-neighbourhood geodesics, dead-end peeling, cycle rank."""

from __future__ import annotations

from collections import deque

import numpy as np

from .config import FREE

NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


class Grid:
    """Immutable occupancy grid addressed by flat index ``r * width + c``."""

    def __init__(self, occupancy) -> None:
        occ = np.asarray(occupancy, dtype=np.int8)
        if occ.ndim != 2:
            raise ValueError("occupancy must be 2-D")
        self.occupancy = occ
        self.height, self.width = occ.shape
        self.free = occ == FREE
        self.size = occ.size
        # Precomputed neighbour table, -1 where the neighbour is not free space.
        nbr = np.full((self.size, 4), -1, dtype=np.int32)
        for r in range(self.height):
            for c in range(self.width):
                i = r * self.width + c
                if not self.free[r, c]:
                    continue
                for k, (dr, dc) in enumerate(NEIGHBORS4):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < self.height and 0 <= cc < self.width and self.free[rr, cc]:
                        nbr[i, k] = rr * self.width + cc
        self._nbr = nbr
        self.free_idx = np.flatnonzero(self.free.ravel())

    # ---- addressing -----------------------------------------------------
    def idx(self, r: int, c: int) -> int:
        return r * self.width + c

    def rc(self, i: int) -> tuple[int, int]:
        return divmod(int(i), self.width)

    def is_free(self, r: int, c: int) -> bool:
        return 0 <= r < self.height and 0 <= c < self.width and bool(self.free[r, c])

    def neighbors(self, i: int) -> np.ndarray:
        """Free 4-neighbours of a flat cell index (-1 entries dropped)."""
        row = self._nbr[i]
        return row[row >= 0]

    def neighbors_rc(self, r: int, c: int) -> list[tuple[int, int]]:
        out = []
        for dr, dc in NEIGHBORS4:
            rr, cc = r + dr, c + dc
            if self.is_free(rr, cc):
                out.append((rr, cc))
        return out

    # ---- topology -------------------------------------------------------
    def geodesic_distance(self, seeds, max_dist: int | None = None) -> np.ndarray:
        """Unit-cost 4-neighbour BFS distance from ``seeds``; -1 = unreachable."""
        dist = np.full(self.size, -1, dtype=np.int32)
        dq: deque[int] = deque()
        for s in np.atleast_1d(np.asarray(seeds, dtype=np.int64)):
            s = int(s)
            if self.free.ravel()[s] and dist[s] < 0:
                dist[s] = 0
                dq.append(s)
        while dq:
            v = dq.popleft()
            nd = int(dist[v]) + 1
            if max_dist is not None and nd > max_dist:
                continue
            for u in self._nbr[v]:
                if u >= 0 and dist[u] < 0:
                    dist[u] = nd
                    dq.append(u)
        return dist

    def distance_map(self, seeds) -> np.ndarray:
        """Geodesic distance as float with ``inf`` outside the source component."""
        dist = self.geodesic_distance(seeds).astype(np.float64)
        dist[dist < 0] = np.inf
        return dist.reshape(self.height, self.width)

    def components(self) -> tuple[np.ndarray, int]:
        """Connected-component labels over free cells (-1 on obstacles)."""
        labels = np.full(self.size, -1, dtype=np.int32)
        count = 0
        for s in self.free_idx:
            if labels[s] >= 0:
                continue
            labels[s] = count
            dq = deque([int(s)])
            while dq:
                v = dq.popleft()
                for u in self._nbr[v]:
                    if u >= 0 and labels[u] < 0:
                        labels[u] = count
                        dq.append(u)
            count += 1
        return labels.reshape(self.height, self.width), count

    def cycle_rank(self) -> int:
        """Independent cycles of the free-space graph: E - V + C."""
        edges = int((self._nbr >= 0).sum()) // 2
        _, comps = self.components()
        return edges - len(self.free_idx) + comps

    def leaf_prune(self) -> tuple[np.ndarray, np.ndarray]:
        """Peel free cells that are not on any cycle (dead-end interiors).

        Returns ``(dead_end_mask, depth)`` where ``depth`` is the peeling layer,
        i.e. how far inside a dead-end pocket a cell sits (0 on the spine).
        """
        alive = self.free.ravel().copy()
        deg = np.where(alive, (self._nbr >= 0).sum(axis=1), 0).astype(np.int32)
        depth = np.zeros(self.size, dtype=np.int32)
        dq = deque(int(i) for i in self.free_idx if deg[i] <= 1)
        layer = 0
        pruned = np.zeros(self.size, dtype=bool)
        while dq:
            layer += 1
            batch = list(dq)
            dq.clear()
            for v in batch:
                if not alive[v]:
                    continue
                alive[v] = False
                pruned[v] = True
                depth[v] = layer
                for u in self._nbr[v]:
                    if u >= 0 and alive[u]:
                        deg[u] -= 1
                        if deg[u] <= 1:
                            dq.append(int(u))
        return pruned.reshape(self.height, self.width), depth.reshape(self.height, self.width)

    # ---- convenience ----------------------------------------------------
    def geodesic_ball(self, center: tuple[int, int], radius: int) -> np.ndarray:
        """Free cells within geodesic ``radius`` of ``center`` (bool mask).

        A Euclidean disk would reach across walls, which pollutes any "region around
        S/G" statistic in a maze; the geodesic ball is the meaningful region.
        """
        dist = self.geodesic_distance([self.idx(*center)], max_dist=radius)
        mask = (dist >= 0) & (dist <= radius)
        return mask.reshape(self.height, self.width)

    def free_fraction(self) -> float:
        return float(self.free.sum()) / self.size


def path_mask(shape: tuple[int, int], path_nodes) -> np.ndarray:
    """Boolean mask marking the flat-index path nodes."""
    mask = np.zeros(shape, dtype=bool)
    for i in path_nodes:
        r, c = divmod(int(i), shape[1])
        mask[r, c] = True
    return mask


def detour_profile(grid: Grid, start: tuple[int, int], goal: tuple[int, int]) -> dict:
    """Cost of routing through a cell, relative to the optimal route.

    ``detour(x) = d(S, x) + d(x, G) - d(S, G)`` is 0 exactly on the cells of some
    optimal route and grows with how far off-route a cell sits (a deep dead-end
    pocket gets a large value).  It depends only on the map, not on a sampled path,
    which makes it the fair label for "correct channel vs. wrong channel".
    """
    s_idx, g_idx = grid.idx(*start), grid.idx(*goal)
    d_s = grid.geodesic_distance([s_idx]).astype(np.float64)
    d_g = grid.geodesic_distance([g_idx]).astype(np.float64)
    d_s[d_s < 0] = np.inf
    d_g[d_g < 0] = np.inf
    d_sg = float(d_s[g_idx])
    detour = d_s + d_g - d_sg
    detour[~np.isfinite(detour)] = np.inf
    shape = (grid.height, grid.width)
    return {
        "d_start": d_s.reshape(shape),
        "d_goal": d_g.reshape(shape),
        "detour": detour.reshape(shape),
        "d_sg": d_sg,
    }
