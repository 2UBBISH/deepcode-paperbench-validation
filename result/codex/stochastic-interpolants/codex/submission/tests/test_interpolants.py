"""Boundary conditions and basic properties of Definition 3.1."""

import math

import pytest
import torch

from si_couplings.interpolants import (
    LinearInterpolant,
    VPInterpolant,
    build_interpolant,
)


@pytest.mark.parametrize("name", ["linear", "linear_zero_gamma", "vp"])
def test_boundary_conditions(name):
    """alpha_0 = beta_1 = 1 and alpha_1 = beta_0 = gamma_0 = gamma_1 = 0."""
    sch = build_interpolant(name)
    x0 = torch.randn(8, 4)
    x1 = torch.randn(8, 4)
    z = torch.randn(8, 4)
    i0 = sch.interpolate(torch.zeros(8), x0, x1, z)
    i1 = sch.interpolate(torch.ones(8), x0, x1, z)
    assert torch.allclose(i0, x0, atol=1e-6)
    assert torch.allclose(i1, x1, atol=1e-6)


def test_linear_coefficients():
    sch = LinearInterpolant()
    t = torch.tensor([0.0, 0.25, 0.5, 1.0])
    c = sch.coefficients(t)
    assert torch.allclose(c.alpha, 1 - t)
    assert torch.allclose(c.beta, t)
    assert torch.allclose(c.gamma, torch.sqrt(2 * t * (1 - t)))
    assert torch.allclose(c.dalpha, -torch.ones_like(t))
    assert torch.allclose(c.dbeta, torch.ones_like(t))


def test_positive_variance_condition():
    """alpha_t^2 + beta_t^2 + gamma_t^2 > 0 on [0, 1]."""
    for name in ["linear", "linear_zero_gamma", "vp"]:
        sch = build_interpolant(name)
        t = torch.linspace(0, 1, 101)
        c = sch.coefficients(t)
        total = c.alpha**2 + c.beta**2 + c.gamma**2
        assert (total > 0).all()


def test_reversed_schedule_is_time_reversal():
    """The schedule printed in Section 4.1 is the time-reversed one."""
    fwd = LinearInterpolant(with_variance=False)
    rev = LinearInterpolant(with_variance=False, reverse=True)
    x0, x1 = torch.randn(5, 3), torch.randn(5, 3)
    t = torch.rand(5)
    a = fwd.interpolate(t, x0, x1)
    b = rev.interpolate(1 - t, x0, x1)
    assert torch.allclose(a, b, atol=1e-6)


def test_vp_interpolant_is_orthogonal():
    sch = VPInterpolant()
    t = torch.rand(6)
    c = sch.coefficients(t)
    assert torch.allclose(c.alpha**2 + c.beta**2, torch.ones_like(t), atol=1e-6)


def test_shapes_for_images_and_vectors():
    sch = build_interpolant("linear")
    t = torch.rand(4)
    for shape in [(4, 2), (4, 3, 8, 8)]:
        x0, x1, z = (torch.randn(*shape) for _ in range(3))
        out = sch.interpolate(t, x0, x1, z)
        assert out.shape == shape
        assert sch.interpolate_velocity(t, x0, x1, z).shape == shape


def test_effective_gamma_of_coupled_interpolant():
    """gamma_tilde_t = alpha_t * sigma for x_0 = m(x_1) + sigma zeta."""
    sch = build_interpolant("linear_zero_gamma")
    t = torch.tensor([0.0, 0.5, 1.0])
    sigma = 0.3
    eff = sch.effective_gamma(t, sigma)
    assert torch.allclose(eff, (1 - t) * sigma)
