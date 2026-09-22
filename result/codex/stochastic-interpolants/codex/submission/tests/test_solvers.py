"""Accuracy of the ODE solvers used for sampling (Appendix B)."""

import math

import torch

from si_couplings.solvers import dopri5_integrate, euler_integrate, heun_integrate, odeint


def _exponential_rhs(x, t):
    return -x


def test_dopri5_solves_linear_ode():
    """dx/dt = -x from 0 to 1 gives x(1) = x(0) e^{-1}."""
    x0 = torch.ones(4, 3)
    out = dopri5_integrate(_exponential_rhs, x0, atol=1e-9, rtol=1e-9)
    assert torch.allclose(out, x0 * math.exp(-1), atol=1e-6)


def test_euler_matches_algorithm_2():
    x0 = torch.ones(2, 5)
    out = euler_integrate(_exponential_rhs, x0, steps=2000)
    assert torch.allclose(out, x0 * math.exp(-1), atol=2e-3)


def test_heun_more_accurate_than_euler():
    x0 = torch.ones(2, 5)
    exact = x0 * math.exp(-1)
    e = (euler_integrate(_exponential_rhs, x0, steps=100) - exact).abs().max()
    h = (heun_integrate(_exponential_rhs, x0, steps=100) - exact).abs().max()
    assert h < e


def test_odeint_dispatch():
    x0 = torch.ones(2, 2)
    for method in ["dopri5", "euler", "heun"]:
        out = odeint(_exponential_rhs, x0, method=method, steps=100)
        assert out.shape == x0.shape


def test_dopri5_adaptive_step_count_is_bounded():
    """A stiff-ish problem should still be integrated in a bounded number of steps."""
    x0 = torch.linspace(-1, 1, 100).reshape(50, 2)

    def rhs(x, t):
        return torch.sin(5 * x)

    out = dopri5_integrate(rhs, x0, atol=1e-6, rtol=1e-6, max_steps=2000)
    assert torch.isfinite(out).all()
