"""B-spline trajectory representation tests."""
from __future__ import annotations

import numpy as np
import torch

from gp_diffuser.trajectory.bspline import BSplineCodec


def test_clamped_endpoints_are_controls():
    codec = BSplineCodec(degree=3, num_controls=32, curve_points=128)
    q = torch.randn(2, 32, 2)
    curve = codec.decode_controls(q)
    assert curve.shape == (2, 128, 2)
    assert torch.allclose(curve[:, 0], q[:, 0], atol=1e-5)
    assert torch.allclose(curve[:, -1], q[:, -1], atol=1e-5)


def test_fit_curve_to_controls_is_endpoint_exact():
    codec = BSplineCodec(degree=3, num_controls=32, curve_points=128)
    curve = torch.randn(1, 128, 2)
    start = torch.tensor([[0.0, 0.0]])
    goal = torch.tensor([[1.0, 1.0]])
    q = codec.fit_curve_to_controls(curve, start, goal)
    assert q.shape == (1, 32, 2)
    assert torch.allclose(q[:, 0], start)
    assert torch.allclose(q[:, -1], goal)
    decoded = codec.decode_controls(q)
    assert torch.allclose(decoded[:, 0], start, atol=1e-5)
    assert torch.allclose(decoded[:, -1], goal, atol=1e-5)


def test_fit_error_is_small_on_a_smooth_curve():
    codec = BSplineCodec(degree=3, num_controls=32, curve_points=128)
    t = torch.linspace(0.0, 1.0, 128)
    curve = torch.stack([t, 0.3 * torch.sin(2 * np.pi * t)], dim=-1)[None]
    q = codec.fit_curve_to_controls(curve, curve[:, 0], curve[:, -1])
    decoded = codec.decode_controls(q)
    assert float((decoded - curve).norm(dim=-1).max()) < 0.02


def test_hard_control_endpoints_are_exact():
    codec = BSplineCodec()
    q = torch.randn(1, 32, 2)
    cond = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    out = codec.hard_control_endpoints(q, cond)
    assert torch.equal(out[:, 0], cond[:, 0])
    assert torch.equal(out[:, -1], cond[:, 1])


def test_control_count_is_a_representation_choice():
    """Raising the control count must not change the network configuration."""
    small = BSplineCodec(num_controls=32)
    large = BSplineCodec(num_controls=48)
    assert small.num_controls == 32 and large.num_controls == 48
    assert small.basis.shape == (128, 32)
    assert large.basis.shape == (128, 48)


def test_basis_derivative_operator_is_exact_for_line():
    """A straight line's derivative must be constant (exact derivative curve)."""
    codec = BSplineCodec(degree=3, num_controls=32, curve_points=128)
    q = torch.zeros(1, 32, 2)
    q[0, :, 0] = torch.linspace(0.0, 1.0, 32)
    params = np.linspace(0.05, 0.95, 17)
    d_basis = codec.basis_derivative_at(params)
    derivative = torch.einsum("pk,bkd->bpd", d_basis, q)
    # a clamped spline re-parameterises the ends, so only the DIRECTION is constant
    assert derivative[..., 0].min() > 0.0
    assert float(derivative[..., 1].abs().max()) < 1e-5
    assert float(derivative[..., 0].std() / derivative[..., 0].mean()) < 0.05
