"""Visualisation: one figure that shows every stage of the pipeline.

Panels (per sample):

* the unified safe map with the start/goal and the GT path,
* the initial Map-only diffusion trajectory (the search guide),
* the single searched path returned by the guidance field,
* the 64 PathSafe ellipses and the local convex regions actually built
  (box cells outside the collision intervals are drawn faintly),
* the final trajectory with its B-spline control polygon.

Usage::

    from gp_diffuser.evaluation.visualize import render_sample
    render_sample(out, sample, path="outputs/figures/sample_00.png")
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np


def _convex_polygon(A, b):
    from ..safety.convex_region import halfspaces_to_vertices
    return halfspaces_to_vertices(np.asarray(A, dtype=np.float64),
                                  np.asarray(b, dtype=np.float64))


def _ellipse_polygon(center, a, b, theta_cos2, theta_sin2, n: int = 64):
    theta = 0.5 * np.arctan2(theta_sin2, theta_cos2)
    t = np.linspace(0.0, 2.0 * np.pi, n)
    local = np.stack([a * np.cos(t), b * np.sin(t)], axis=-1)
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    return local @ rot.T + np.asarray(center)


def render_sample(out: Dict, sample: Dict, path: Optional[str] = None,
                  title: str = "", dpi: int = 130):
    """Render the stage-by-stage figure for ONE sample."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    occupancy = np.asarray(sample["occupancy_m"], dtype=np.float64)
    if occupancy.ndim == 3:
        occupancy = occupancy[0]
    res_m = float(sample["res_m"])
    extent = occupancy.shape[-1] * res_m
    transform = sample.get("transform")

    fig, axes = plt.subplots(1, 3, figsize=(19.0, 6.4), constrained_layout=True)
    for ax in axes:
        ax.set_xlim(0.0, extent)
        ax.set_ylim(0.0, extent)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(occupancy, origin="lower", cmap="Greys",
                  extent=(0.0, extent, 0.0, extent), alpha=0.85)

    def to_m(xy):
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        return transform.to_meters(xy) if transform is not None else xy

    gt = to_m(sample["gt_path_scene"])
    start = to_m(sample["start_goal_scene"])[0]
    goal = to_m(sample["start_goal_scene"])[1]

    # ---------------- panel 1: guide + searched path -----------------------
    ax = axes[0]
    ax.set_title("(a) Map-only guide -> guidance-field search")
    ax.plot(gt[:, 0], gt[:, 1], color="#1f77b4", lw=2.4, label="GT path")
    guide = sample.get("guide_path_m")
    if guide is not None:
        guide = np.asarray(guide, dtype=np.float64)
        ax.plot(guide[:, 0], guide[:, 1], color="#ff7f0e", lw=2.0, ls="--",
                label="initial diffusion guide")
    search = out.get("search_path_m")
    if search is not None:
        search = np.asarray(search[0] if np.ndim(search) == 3 else search,
                            dtype=np.float64)
        ax.plot(search[:, 0], search[:, 1], color="#2ca02c", lw=2.2,
                label="searched path (single)")
    ax.plot(*start, "o", color="#2ca02c", ms=9, label="start")
    ax.plot(*goal, "s", color="#d62728", ms=9, label="goal")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    # ---------------- panel 2: ellipses + convex regions ------------------
    ax = axes[1]
    ax.set_title("(b) PathSafe ellipses + local convex regions")
    ellipses = out.get("ellipses")
    scene_scale = float(sample.get("scene_scale", 32.0))
    if ellipses is not None:
        ell = np.asarray(ellipses[0] if np.ndim(ellipses) == 3 else ellipses,
                         dtype=np.float64)
        centers_m = transform.to_meters(ell[:, 0:2]) if transform is not None else ell[:, 0:2]
        axes_len_m = ell[:, 2:4] * (scene_scale if transform is None else transform.scale)
        for i in range(len(ell)):
            poly = _ellipse_polygon(centers_m[i], axes_len_m[i, 0], axes_len_m[i, 1],
                                    ell[i, 4], ell[i, 5], n=48)
            ax.plot(poly[:, 0], poly[:, 1], color="#9467bd", lw=0.7, alpha=0.75)

    regions = sample.get("region_table")
    if regions is not None:
        cell_A = regions["cell_A"]
        cell_b = regions["cell_b"]
        cell_mask = regions["cell_mask"]
        for i in range(len(cell_mask)):
            if not cell_mask[i]:
                continue
            faces = np.isfinite(cell_b[i])
            A = cell_A[i][faces]
            bb = cell_b[i][faces]
            box_like = (len(bb) == 4 and np.allclose(np.abs(A), np.abs(
                np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]))))
            poly = _convex_polygon(A, bb)
            if poly is None or len(poly) < 3:
                continue
            poly_m = transform.to_meters(poly) if transform is not None else poly
            color = "#cccccc" if box_like else "#17becf"
            alpha = 0.12 if box_like else 0.30
            ax.fill(poly_m[:, 0], poly_m[:, 1], color=color, alpha=alpha,
                    lw=0.0)
    ax.plot(*start, "o", color="#2ca02c", ms=9)
    ax.plot(*goal, "s", color="#d62728", ms=9)
    ax.text(0.02, 0.98, "cyan = local convex cell\n grey = map box cell",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(fc="white", alpha=0.8, lw=0))

    # ---------------- panel 3: final trajectory ---------------------------
    ax = axes[2]
    ax.set_title("(c) Final trajectory (after ALM + DDIM)")
    traj = np.asarray(out["trajectory_m"], dtype=np.float64)
    traj = traj[0] if traj.ndim == 3 else traj
    ax.plot(gt[:, 0], gt[:, 1], color="#1f77b4", lw=2.0, alpha=0.7, label="GT path")
    dense = out.get("trajectory_dense_m")
    if dense is not None:
        dense = np.asarray(dense[0] if np.ndim(dense) == 3 else dense, dtype=np.float64)
        ax.plot(dense[:, 0], dense[:, 1], color="#d62728", lw=2.4, label="prediction (512 pts)")
    else:
        ax.plot(traj[:, 0], traj[:, 1], color="#d62728", lw=2.4, label="prediction (128 pts)")
    controls = out["controls"]
    controls = controls[0] if np.ndim(controls) == 3 else controls
    ctrl_m = transform.to_meters(np.asarray(controls, dtype=np.float64)) \
        if transform is not None else np.asarray(controls)
    ax.plot(ctrl_m[:, 0], ctrl_m[:, 1], "-", color="#7f7f7f", lw=0.9, marker="o",
            ms=2.4, label="control polygon")
    ax.plot(*start, "o", color="#2ca02c", ms=9)
    ax.plot(*goal, "s", color="#d62728", ms=9)
    valid = out.get("valid")
    collision = out.get("dense_collision_fraction", 0.0)
    ax.text(0.02, 0.98,
            "valid=%s\ncollision=%.3f\nALM steps=%d"
            % (valid, collision, len(out.get("alm_stats", []))),
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(fc="white", alpha=0.85, lw=0))
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    if title:
        fig.suptitle(title, fontsize=11)
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)
        return path
    return fig


def render_training_curves(history, path: str, dpi: int = 130):
    """Loss curves for the joint objective and its components."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not history:
        return None
    keys = ["loss", "loss_u", "loss_c", "loss_align", "loss_collision", "loss_corridor"]
    fig, ax = plt.subplots(figsize=(8.0, 4.6), constrained_layout=True)
    steps = np.arange(1, len(history) + 1)
    for key in keys:
        values = [row.get(key, np.nan) for row in history]
        ax.plot(steps, values, label=key, lw=1.6)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("dual-branch joint objective")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
