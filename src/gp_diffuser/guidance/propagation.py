"""Exact and approximate propagators for the max-plus guidance field.

The target field is F(x) = max_i [w_i - lam * d_free(x, s_i)].

Implementations:
- heap: exact Python max-Dijkstra, optionally returns dominant-source ownership.
- vectorized: exact synchronous max-plus relaxation; best on shallow/open maps.
- bucket: quantized Dial-style propagation.
- scipy: exact compiled sparse-graph Dijkstra with a virtual super-source.
- auto: use compiled sparse Dijkstra for field-only queries; heap when ownership is requested.

The sparse graph is cached per immutable Grid instance, so repeated fields on one map
only pay for the source-specific virtual edges and shortest-path solve.
"""
from __future__ import annotations

import heapq
import weakref
from weakref import WeakKeyDictionary

import numpy as np

# ADAPTATION (GeoPath-Diffuser): scipy is an OPTIONAL backend.  The
# ``scipy`` propagator below is unchanged; when scipy is missing this module
# still imports and the exact ``heap`` backend stays available (it is the
# same max-Dijkstra, only slower).
try:  # pragma: no cover - depends on the environment
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    _HAVE_SCIPY = True
except Exception:  # noqa: BLE001
    csr_matrix = None
    dijkstra = None
    _HAVE_SCIPY = False

from .grid import Grid

NEG_INF = -np.inf
_CSR_CACHE = weakref.WeakKeyDictionary()  # auto-evicts when a Grid is collected


def _validate_inputs(grid: Grid, samples, weights, lam: float):
    samples = np.asarray(samples, dtype=np.int64).reshape(-1, 2)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(samples) != len(weights):
        raise ValueError("samples and weights must have the same length")
    if not np.isfinite(weights).all():
        raise ValueError("weights must be finite")
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("lam must be finite and strictly positive")
    if len(samples):
        r, c = samples[:, 0], samples[:, 1]
        if np.any(r < 0) or np.any(r >= grid.height) or np.any(c < 0) or np.any(c >= grid.width):
            raise ValueError("source coordinate outside grid")
        if not np.all(grid.free[r, c]):
            raise ValueError("all propagation sources must lie in free space")
    return samples, weights


def _seed(grid: Grid, samples, weights):
    field = np.full((grid.height, grid.width), NEG_INF, dtype=np.float64)
    owner = np.full((grid.height, grid.width), -1, dtype=np.int32)
    for i, (r, c) in enumerate(samples):
        if weights[i] > field[r, c]:
            field[r, c] = weights[i]
            owner[r, c] = i
    return field, owner


def _finalise(field, owner, free, return_owner):
    out = np.where(free & np.isfinite(field), field, np.nan)
    if not return_owner:
        return out
    own = np.where(free & (owner >= 0), owner, -1) if owner is not None else np.full(free.shape, -1, np.int32)
    return out, own


def propagate_heap(grid: Grid, samples, weights, lam: float, return_owner: bool = False):
    """Exact multi-source max-Dijkstra on a Python max-heap."""
    samples, weights = _validate_inputs(grid, samples, weights, lam)
    field2d, owner2d = _seed(grid, samples, weights)
    field, owner = field2d.ravel().copy(), owner2d.ravel().copy()
    heap = []
    queued = set()
    for r, c in samples:
        idx = grid.idx(int(r), int(c))
        if idx not in queued and field[idx] > NEG_INF:
            queued.add(idx)
            heap.append((-field[idx], idx))
    heapq.heapify(heap)
    nbr = grid._nbr
    while heap:
        neg_v, v = heapq.heappop(heap)
        value = -neg_v
        if value < field[v]:
            continue
        candidate = value - lam
        for u in nbr[v]:
            if u >= 0 and candidate > field[u]:
                field[u] = candidate
                owner[u] = owner[v]
                heapq.heappush(heap, (-candidate, int(u)))
    return _finalise(field.reshape(field2d.shape), owner.reshape(field2d.shape), grid.free, return_owner)


def propagate_vectorized(grid: Grid, samples, weights, lam: float, return_owner: bool = False,
                         steps_per_round: int = 4, max_rounds: int = 200000):
    """Exact synchronous max-plus relaxation; runtime scales with propagation eccentricity × area."""
    if return_owner:
        raise ValueError("propagate_vectorized cannot report dominant-source ownership")
    samples, weights = _validate_inputs(grid, samples, weights, lam)
    field, _ = _seed(grid, samples, weights)
    free, work = grid.free, np.empty_like(field)
    converged = False
    for _round in range(max_rounds):
        batch_start = field.copy()
        for _step in range(max(1, steps_per_round)):
            np.copyto(work, field)
            np.maximum(work[1:, :], field[:-1, :] - lam, out=work[1:, :])
            np.maximum(work[:-1, :], field[1:, :] - lam, out=work[:-1, :])
            np.maximum(work[:, 1:], field[:, :-1] - lam, out=work[:, 1:])
            np.maximum(work[:, :-1], field[:, 1:] - lam, out=work[:, :-1])
            work[~free] = NEG_INF
            field, work = work, field
        if np.array_equal(field, batch_start):
            converged = True
            break
    if not converged:
        raise RuntimeError(f"vectorized propagation did not converge within {max_rounds} rounds")
    return _finalise(field, None, free, False)


def _free_graph(grid: Grid):
    """Return cached CSR adjacency for the free-space grid graph (unit edge costs)."""
    graph = _CSR_CACHE.get(grid)
    if graph is not None:
        return graph
    nbr = grid._nbr
    valid = nbr >= 0
    counts = valid.sum(axis=1).astype(np.int64)
    indptr = np.empty(grid.size + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    indices = nbr[valid].astype(np.int32, copy=False)
    data = np.ones(indices.size, dtype=np.float64)
    graph = csr_matrix((data, indices, indptr), shape=(grid.size, grid.size))
    _CSR_CACHE[grid] = graph
    return graph


def propagate_scipy(grid: Grid, samples, weights, lam: float, return_owner: bool = False):
    """Exact compiled sparse Dijkstra using a virtual super-source.

    For w_max=max(w_i), minimise d_free(x,s_i)+(w_max-w_i)/lam.
    A virtual source connects to each seed with that nonnegative offset plus 1.
    The common +1 does not change the minimiser and avoids zero-weight sparse edges.
    """
    if return_owner:
        return propagate_heap(grid, samples, weights, lam, return_owner=True)
    samples, weights = _validate_inputs(grid, samples, weights, lam)
    if not len(samples):
        return np.full((grid.height, grid.width), np.nan, dtype=np.float64)
    graph = _free_graph(grid)
    w_max = float(weights.max())
    seed_ids = samples[:, 0] * grid.width + samples[:, 1]
    # Multiple trajectory samples may occupy the same cell: keep the strongest.
    unique_ids, inverse = np.unique(seed_ids, return_inverse=True)
    unique_weights = np.full(len(unique_ids), NEG_INF, dtype=np.float64)
    np.maximum.at(unique_weights, inverse, weights)
    offsets = (w_max - unique_weights) / lam + 1.0

    n = grid.size
    indptr = np.concatenate((graph.indptr.astype(np.int64, copy=False),
                             np.array([graph.nnz + len(unique_ids)], dtype=np.int64)))
    indices = np.concatenate((graph.indices.astype(np.int32, copy=False), unique_ids.astype(np.int32)))
    data = np.concatenate((graph.data, offsets))
    augmented = csr_matrix((data, indices, indptr), shape=(n + 1, n + 1))
    distances = dijkstra(augmented, directed=True, indices=n)
    values = w_max - lam * (distances[:n] - 1.0)
    reached = grid.free.ravel() & np.isfinite(distances[:n])
    out = np.full(n, np.nan, dtype=np.float64)
    out[reached] = values[reached]
    return out.reshape(grid.height, grid.width)


def propagate_bucket(grid: Grid, samples, weights, lam: float, return_owner: bool = False, scale: int = 16):
    """Quantized Dial-style propagation; maximum field error is lam/(2*scale)."""
    samples, weights = _validate_inputs(grid, samples, weights, lam)
    free_flat, size = grid.free.ravel(), grid.size
    big = np.iinfo(np.int64).max
    key = np.full(size, big, dtype=np.int64)
    owner = np.full(size, -1, dtype=np.int32)
    levels, level_heap = {}, []

    def push(level, node):
        bucket = levels.get(level)
        if bucket is None:
            levels[level] = [node]
            heapq.heappush(level_heap, level)
        else:
            bucket.append(node)

    for i, (r, c) in enumerate(samples):
        idx = grid.idx(int(r), int(c))
        offset = int(round(scale * (1.0 - float(weights[i])) / lam))
        if offset < key[idx]:
            key[idx], owner[idx] = offset, i
            push(offset, idx)

    settled = np.zeros(size, dtype=bool)
    for _ in iter(int, 1):
        if not level_heap:
            break
        level = heapq.heappop(level_heap)
        bucket = levels.pop(level)
        for v in bucket:
            if settled[v] or key[v] != level:
                continue
            settled[v] = True
            nxt = level + scale
            for u in grid._nbr[v]:
                if u >= 0 and not settled[u] and nxt < key[u]:
                    key[u], owner[u] = nxt, owner[v]
                    push(nxt, int(u))
    reached = settled & free_flat
    field = np.full(size, NEG_INF, dtype=np.float64)
    field[reached] = 1.0 - lam * (key[reached].astype(np.float64) / scale)
    return _finalise(field.reshape(grid.height, grid.width), owner.reshape(grid.height, grid.width),
                     grid.free, return_owner)


def propagate_auto(grid: Grid, samples, weights, lam: float, return_owner: bool = False, **kwargs):
    """Use compiled exact sparse Dijkstra for field-only queries; retain heap for ownership."""
    if return_owner:
        return propagate_heap(grid, samples, weights, lam, return_owner=True)
    if not _HAVE_SCIPY:  # ADAPTATION: optional scipy backend
        return propagate_heap(grid, samples, weights, lam)
    return propagate_scipy(grid, samples, weights, lam)


PROPAGATORS = {
    "heap": propagate_heap,
    "vectorized": propagate_vectorized,
    "bucket": propagate_bucket,
    "scipy": propagate_scipy,
    "auto": propagate_auto,
}


def propagate(grid: Grid, samples, weights, lam: float, method: str = "auto",
              return_owner: bool = False, **kwargs):
    if method not in PROPAGATORS:
        raise KeyError(f"unknown propagation method {method!r}")
    return PROPAGATORS[method](grid, samples, weights, lam, return_owner=return_owner, **kwargs)
