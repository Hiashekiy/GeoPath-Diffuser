"""Trajectory representation: B-spline codec, control fits and dense-path queries."""
from .bspline import BSplineCodec, TrajectoryToControlHead, bspline_basis_matrix
from .dense_path import gather_dense_path_points

__all__ = ["BSplineCodec", "TrajectoryToControlHead", "bspline_basis_matrix",
           "gather_dense_path_points"]
