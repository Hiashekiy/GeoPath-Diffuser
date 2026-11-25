"""Configuration and shared constants for the guidance-field library.

One dataclass carries every tunable of the whole pipeline (map generation, collision
grading, source handling, propagation and search), so a single object can be threaded
through the API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# grid values
FREE = 0
OCCUPIED = 1

# source-weight families: F(x) = max_i [ w(u_i) - lam * d_free(x, s_i) ]
WEIGHT_KINDS = ("uniform", "linear", "quadratic")

# Canonical clip bounds per weighting. Used both for figure colour ranges and for the
# search-stage normalisation, so one alpha means the same thing everywhere.
FIELD_CLIP = {
    "uniform": (0.50, 1.00),
    "linear": (-0.10, 1.00),
    "quadratic": (-0.10, 1.00),
}

SOURCE_METHODS = ("M0", "M1", "M2")

MAP_TYPES = ("A", "B", "C", "D")
MAP_TYPE_NAMES = {
    "A": "random_clutter",
    "B": "rooms_and_corridors",
    "C": "multi_branch_maze",
    "D": "structured_topology",
}
# Free-space fraction band per map family: the families are deliberately not forced into
# one shared band (a maze is denser than a cluttered open field).
FREE_FRACTION_BANDS = {
    "A": (0.60, 0.88),
    "B": (0.50, 0.80),
    "C": (0.30, 0.62),
    "D": (0.42, 0.78),
}

# Path-quality grades by optimality ratio rho = L(P) / L(P*): (name, lo, hi, centre).
BUCKETS = (
    ("excellent", 1.00, 1.05, 1.02),
    ("good", 1.05, 1.15, 1.10),
    ("medium", 1.15, 1.30, 1.22),
    ("poor", 1.30, None, 1.45),
)

TRAJECTORY_GRADES = ("feasible", "mild_collision", "moderate_collision", "severe_collision")
# Collision grades are assigned from the *measured* collision ratio, never from the name of
# the generator that produced the trajectory.
COLLISION_BANDS = {
    "mild_collision": (0.01, 0.05),
    "moderate_collision": (0.05, 0.15),
    "severe_collision": (0.15, 0.30),
}

METHODS = (
    "B0_astar",
    "B1_distance_gbfs",
    "B2_field_only",
    "B3_linear_fusion",
    "B4_nonlinear_fusion",
    "B5_adaptive_fusion",
)
METHOD_LABELS = {
    "B0_astar": "A*  (g + h)",
    "B1_distance_gbfs": "Distance-GBFS",
    "B2_field_only": "Field-Only",
    "B3_linear_fusion": "Linear Fusion",
    "B4_nonlinear_fusion": "Nonlinear Fusion",
    "B5_adaptive_fusion": "Adaptive Fusion",
}
# Methods whose priority must not touch the accumulated cost g(n).
NON_G_METHODS = METHODS[1:]
SEARCH_STATUSES = ("success", "unreachable", "cutoff")


def field_clip_bounds(weight: str) -> tuple[float, float]:
    """Lower/upper clip of the field for a weighting (the single source of truth)."""
    if weight not in FIELD_CLIP:
        raise KeyError(f"no clip bounds configured for weight {weight!r}")
    return FIELD_CLIP[weight]


@dataclass(frozen=True)
class Config:
    """Every tunable of the pipeline, with the defaults used in the published results."""

    # --- maps -------------------------------------------------------------
    grid_size: int = 64
    instances_per_type: int = 5
    seed: int = 20260214
    min_geodesic_sg: int = 30
    max_scene_attempts: int = 120

    # --- collision trajectories -------------------------------------------
    collision_detection_step: float = 0.2
    num_shortcut_candidates: int = 90
    num_bezier_candidates: int = 60
    num_directed_candidates: int = 40

    # --- sources and fields ------------------------------------------------
    source_method: str = "M0"
    num_source_samples: int = 64
    lam: float = 0.02
    ball_radius: int = 3
    projection_radius: int = 3
    bridge_roi_margins: tuple[int, ...] = (4, 8, 16, 32)
    field_weights: tuple[str, ...] = ("linear", "quadratic")
    propagation: str = "auto"  # auto | scipy | vectorized | heap | bucket

    # --- search ------------------------------------------------------------
    alphas: tuple[float, ...] = (0.25, 0.50, 0.75)
    gamma_main: float = 0.5
    alpha_far: float = 0.25
    alpha_near: float = 0.75
    adaptive_eta: float = 1.0
    adaptive_gamma: float = 0.5
    node_cutoff: int = 0  # 0 disables the safety cutoff

    # --- region statistics -------------------------------------------------
    detour_near: float = 2.0
    detour_far: float = 10.0

    # --- experiment bookkeeping -------------------------------------------
    core_grade: str = "moderate_collision"
    figure_grades: tuple[str, ...] = TRAJECTORY_GRADES

    def collision_bands(self) -> dict:
        return dict(COLLISION_BANDS)

    def as_dict(self) -> dict:
        return asdict(self)
