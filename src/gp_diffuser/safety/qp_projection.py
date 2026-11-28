"""Terminal convex QP projection of the clean controls into the frozen corridor.

CPCG is soft guidance: it changes the reverse direction but certifies nothing.
After the LAST reverse step the clean control polygon therefore gets one standard
convex QP, solved per sample against the SAME frozen
:class:`~gp_diffuser.safety.bspline_constraints.BSplineConstraintPack` the
guidance used:

    minimize    0.5 * || Q - Q_target ||_2^2

    subject to  A[p,f] · E[p,r] · Q <= b[p,f]      for every valid (p, r, f)
                Q[0]   = start,   Q[C-1] = goal
                (optional, when a heading is valid)
                    normal( Q_1 - Q_0 )     = 0,  forward( Q_1 - Q_0 )     >= 0
                    normal( Q_C-1 - Q_C-2 ) = 0,  forward( Q_C-1 - Q_C-2 ) >= 0

The optimisation variable is ALWAYS the global B-spline control polygon ``Q
[C,2]``; the four Bezier controls of a piece are ``E[p,r] Q``, i.e. linear
functions of it, never independent variables.

Because the four Bezier controls of a piece are in the (convex) responsible
region, the whole curve piece is: a successful QP is a CONTINUOUS certificate
for the intervals the pack actually covers.  It says nothing about intervals the
corridor never covered - that is what the independent dense collision validation
in the sampler is for.

Nothing here rebuilds a corridor, re-extracts a Bezier matrix or calls the ALM.
"""
from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch

__all__ = ["project_controls_to_corridor_qp", "corridor_constraint_rows"]

# OSQP statuses that carry a usable solution
_SOLVED = ("solved", "solved inaccurate")


def _as_numpy(torch_or_array, dtype=np.float64) -> np.ndarray:
    if torch.is_tensor(torch_or_array):
        torch_or_array = torch_or_array.detach().cpu().numpy()
    return np.asarray(torch_or_array, dtype=dtype)


def corridor_constraint_rows(pack, sample: int
                             ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """``(G [M,2C], h [M])`` for ``A E Q <= b`` of one sample, valid rows only.

    The row of ``(p, r, f)`` is ``G[m, 2k+d] = E[p,r,k] * A[p,f,d]`` and
    ``h[m] = b[p,f]``, i.e. exactly ``A[p,f] . (E[p,r] Q) <= b[p,f]`` flattened
    into the control-space ordering ``(k, d)``.  Padded pieces and padded faces
    are never emitted.
    """
    piece_mask = _as_numpy(pack.piece_mask[sample], bool)
    face_mask = _as_numpy(pack.face_mask[sample], bool)
    if not piece_mask.any():
        return None, None
    pieces, faces = np.nonzero(piece_mask[:, None] & face_mask)
    if len(pieces) == 0:
        return None, None

    E = _as_numpy(pack.extraction[sample])            # [P,4,C]
    A = _as_numpy(pack.piece_A[sample])               # [P,F,2]
    b = _as_numpy(pack.piece_b[sample])               # [P,F]
    C = int(E.shape[-1])

    E_sel = np.concatenate([E[pieces, r] for r in range(4)], axis=0)   # [4M,C]
    A_sel = np.tile(A[pieces, faces], (4, 1))                          # [4M,2]
    h_sel = np.tile(b[pieces, faces], 4)                               # [4M]
    G = (E_sel[:, :, None] * A_sel[:, None, :]).reshape(-1, 2 * C)
    return G, h_sel


def _tangent_rows(C: int, start_dir: np.ndarray, goal_dir: np.ndarray):
    """Linear tangent rows, using the BoundaryDecoder's control convention.

    ``Q_1 - Q_0`` and ``Q_{C-1} - Q_{C-2}`` are the forward tangents at ``u=0``
    and ``u=1`` up to a POSITIVE scale (clamped uniform knots), which is the same
    convention :class:`~gp_diffuser.models.boundary.BoundaryDecoder` aligns, so a
    QP solution and a decoded ``q0_guided`` agree on what "heading" means.
    """
    eq = np.zeros((2, 2 * C))
    ineq = np.zeros((2, 2 * C))
    pairs = ((1, 0, start_dir, 0), (C - 1, C - 2, goal_dir, 1))
    for i, j, direction, slot in pairs:
        if direction is None:
            continue
        d = np.asarray(direction, dtype=np.float64).reshape(2)
        norm = float(np.linalg.norm(d))
        if norm <= 1e-9:
            continue
        d = d / norm
        n = np.array([-d[1], d[0]])
        for vec, rows in ((n, eq), (d, ineq)):
            rows[slot, 2 * i] = vec[0]
            rows[slot, 2 * i + 1] = vec[1]
            rows[slot, 2 * j] = -vec[0]
            rows[slot, 2 * j + 1] = -vec[1]
    keep_eq = np.abs(eq).sum(axis=1) > 0.0
    keep_ineq = np.abs(ineq).sum(axis=1) > 0.0
    return eq[keep_eq], ineq[keep_ineq]


def _solve_one(target: np.ndarray, G: Optional[np.ndarray], h: Optional[np.ndarray],
               start: np.ndarray, goal: np.ndarray,
               start_dir: Optional[np.ndarray], goal_dir: Optional[np.ndarray],
               cfg: dict) -> Tuple[np.ndarray, Dict]:
    """One convex QP; returns the (possibly snapped) controls and diagnostics."""
    import osqp
    import scipy.sparse as sp

    C = int(target.shape[0])
    n_var = 2 * C
    blocks, lower, upper = [], [], []
    n_corridor = 0
    if G is not None and h is not None and len(h):
        blocks.append(G)
        lower.append(np.full(len(h), -np.inf))
        upper.append(h)
        n_corridor = int(len(h))

    A_eq = np.zeros((4, n_var))
    A_eq[0, 0] = 1.0
    A_eq[1, 1] = 1.0
    A_eq[2, 2 * (C - 1)] = 1.0
    A_eq[3, 2 * (C - 1) + 1] = 1.0
    b_eq = np.array([start[0], start[1], goal[0], goal[1]], dtype=np.float64)
    blocks.append(A_eq)
    lower.append(b_eq)
    upper.append(b_eq)

    if start_dir is not None or goal_dir is not None:
        t_eq, t_ineq = _tangent_rows(C, start_dir, goal_dir)
        if len(t_eq):
            blocks.append(t_eq)
            lower.append(np.zeros(len(t_eq)))
            upper.append(np.zeros(len(t_eq)))
        if len(t_ineq):
            blocks.append(t_ineq)
            lower.append(np.zeros(len(t_ineq)))
            upper.append(np.full(len(t_ineq), np.inf))

    A_osqp = sp.csc_matrix(np.vstack(blocks))
    l_vec = np.concatenate(lower)
    u_vec = np.concatenate(upper)
    P = sp.identity(n_var, format="csc")
    q_vec = -np.asarray(target, dtype=np.float64).reshape(-1)

    info: Dict = {"status": "not_run", "iterations": 0, "solve_time_s": 0.0,
                  "num_corridor_rows": n_corridor, "certified": False}
    # Pre-check: the endpoint equalities can only be met inside the corridor if
    # the endpoints themselves are in it.  Reported explicitly so an infeasible
    # QP says WHETHER the corridor covered the endpoints (spec 5.4).
    endpoint_violation = float("nan")
    if G is not None and len(h):
        snapped = np.array(target, dtype=np.float64).reshape(-1, 2).copy()
        snapped[0] = start
        snapped[C - 1] = goal
        endpoint_violation = float(np.max(G @ snapped.reshape(-1) - h))
    info["endpoint_violation_scene"] = endpoint_violation
    t0 = time.perf_counter()
    setup_kwargs = dict(
        P=P, q=q_vec, A=A_osqp, l=l_vec, u=u_vec, verbose=False,
        eps_abs=float(cfg.get("eps_abs", 1e-5)),
        eps_rel=float(cfg.get("eps_rel", 1e-5)),
        max_iter=int(cfg.get("max_iter", 10000)))
    try:
        solver = osqp.OSQP()
        try:
            # OSQP >= 1.0 renamed the setting
            solver.setup(polishing=bool(cfg.get("polish", True)), **setup_kwargs)
        except TypeError:
            solver.setup(polish=bool(cfg.get("polish", True)), **setup_kwargs)
        try:
            result = solver.solve(raise_error=False)
        except TypeError:
            result = solver.solve()
        x = np.asarray(result.x, dtype=np.float64) if result.x is not None else None
        info["status"] = str(result.info.status)
        info["iterations"] = int(result.info.iter)
        info["solve_time_s"] = float(getattr(result.info, "run_time", 0.0) or 0.0)
    except Exception as error:                                  # pragma: no cover
        x = None
        info["status"] = "solver_error:%s" % error
    info["wall_time_s"] = time.perf_counter() - t0

    if x is None or not np.all(np.isfinite(x)) or info["status"] not in _SOLVED:
        # an infeasible / unconverged iterate is NOT a projection: never report it
        # (or a correction distance computed from it)
        info["max_violation_scene"] = float("nan")
        info["certified"] = False
        if info["status"] not in _SOLVED:
            tol = float(cfg.get("feasibility_tol_scene", 1e-4))
            info["failure_reason"] = (
                "endpoints_outside_corridor"
                if np.isfinite(endpoint_violation) and endpoint_violation > tol
                else "corridor_infeasible")
        return np.array(target, dtype=np.float64, copy=True), info

    controls = x.reshape(C, 2)
    # the endpoint rows are hard conditions: re-impose them exactly (the solver
    # satisfies them to ~1e-9, this removes even that) and report feasibility of
    # the polygon that is actually returned
    controls[0] = start
    controls[C - 1] = goal
    if G is not None and len(h):
        violation = float(np.max(G @ controls.reshape(-1) - h))
    else:
        violation = 0.0
    info["max_violation_scene"] = violation
    info["certified"] = bool(n_corridor > 0
                             and violation <= float(cfg.get("feasibility_tol_scene",
                                                            1e-4)))
    if not info["certified"]:
        info["failure_reason"] = ("no_corridor_constraints" if n_corridor == 0
                                  else "residual_violation_after_projection")
    return controls, info


def project_controls_to_corridor_qp(
    q_target: torch.Tensor,
    pack,
    start_goal: torch.Tensor,
    heading: Optional[torch.Tensor] = None,
    heading_valid: Optional[torch.Tensor] = None,
    solver_cfg: Optional[dict] = None,
) -> Tuple[torch.Tensor, Dict]:
    """One terminal convex QP per sample.  Returns ``(q_projected, info)``.

    ``info`` carries per-sample ``status`` / ``iterations`` / ``solve_time_s`` /
    ``max_violation_scene`` / ``certified`` plus the aggregate keys the sampler
    and the report use.  A sample without any corridor constraint is returned
    UNCHANGED with ``certified == False``: an empty pack is not a certificate.
    """
    cfg = dict(solver_cfg or {})
    scene_to_meter = float(cfg.get("scene_to_meter", 1.0))
    B, C = int(q_target.shape[0]), int(q_target.shape[1])

    projected = q_target.detach().clone()
    per_sample: list = []
    if pack is None:
        per_sample = [{"status": "no_constraints", "iterations": 0,
                       "solve_time_s": 0.0, "num_corridor_rows": 0,
                       "max_violation_scene": float("nan"),
                       "correction_m": 0.0, "certified": False}] * B
    else:
        for b in range(B):
            target = _as_numpy(q_target[b])
            start = _as_numpy(start_goal[b, 0])
            goal = _as_numpy(start_goal[b, 1])
            G, h = corridor_constraint_rows(pack, b)
            if G is None:
                per_sample.append({"status": "no_constraints", "iterations": 0,
                                   "solve_time_s": 0.0, "num_corridor_rows": 0,
                                   "max_violation_scene": float("nan"),
                                   "correction_m": 0.0, "certified": False})
                continue
            start_dir = goal_dir = None
            if heading is not None:
                hd = _as_numpy(heading[b]).reshape(-1, 2)
                if heading_valid is None:
                    start_dir, goal_dir = hd[0], hd[1]
                else:
                    hv = _as_numpy(heading_valid[b], bool).reshape(-1)
                    for k in range(min(2, len(hv))):
                        if bool(hv[k]):
                            if k == 0:
                                start_dir = hd[0]
                            else:
                                goal_dir = hd[1]
                # an invalid end simply gets no tangent rows (never a silent drop
                # of a VALID heading)
            controls, info = _solve_one(target, G, h, start, goal,
                                        start_dir, goal_dir, cfg)
            info["correction_m"] = float(
                np.abs(controls - target).max() * scene_to_meter)
            per_sample.append(info)
            if info["certified"]:
                projected[b] = torch.as_tensor(controls, dtype=q_target.dtype,
                                               device=q_target.device)
            else:
                # never present an uncertified projection as the safe trajectory:
                # the sampler keeps the un-projected controls and the failure flag
                projected[b] = q_target[b]

    def _col(key: str, default):
        return [info.get(key, default) for info in per_sample]

    certified = torch.tensor([bool(item.get("certified", False))
                              for item in per_sample], dtype=torch.bool,
                             device=q_target.device)
    summary = {
        "solver": str(cfg.get("solver", "osqp")),
        "per_sample": per_sample,
        "status": _col("status", "not_run"),
        "iterations": _col("iterations", 0),
        "solve_time_s": _col("solve_time_s", 0.0),
        "max_violation_scene": _col("max_violation_scene", float("nan")),
        "correction_m": _col("correction_m", 0.0),
        "num_corridor_rows": _col("num_corridor_rows", 0),
        "certified": certified.detach().cpu(),
        "all_certified": bool(certified.all().item()),
        "total_solve_time_s": float(sum(item.get("solve_time_s", 0.0)
                                        for item in per_sample)),
    }
    return projected, summary
