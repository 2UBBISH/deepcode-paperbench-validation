"""Tests for the forward/reverse SDE machinery in :mod:`npse.src.sde`.

These tests validate the closed-form transition kernels, conditional scores,
reverse drifts, probability-flow ODE velocities, prior sampling, and helper
functions against the analytic formulas used in the NPSE paper.
"""

from __future__ import annotations

import math

import pytest
import torch

from npse.src.sde import (
    SDE,
    VESDE,
    VPSDE,
    ensure_tensor,
    estimate_sigma_max,
    get_sde,
    isotropic_gaussian_logp,
    reverse_sde_step,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ve(sigma_min: float = 0.01, sigma_max: float = 1.0) -> VESDE:
    return VESDE(sigma_min=sigma_min, sigma_max=sigma_max, T=1.0)


def _make_vp(beta_min: float = 0.1, beta_max: float = 11.0) -> VPSDE:
    return VPSDE(beta_min=beta_min, beta_max=beta_max, T=1.0)


# ---------------------------------------------------------------------------
# VE SDE
# ---------------------------------------------------------------------------

def test_ve_sigma_and_g_match_closed_form():
    sde = _make_ve(sigma_min=0.01, sigma_max=1.0)
    t = 0.5
    expected_sigma = 0.01 * (1.0 / 0.01) ** 0.5  # 0.1
    expected_g = expected_sigma * math.sqrt(2.0 * math.log(1.0 / 0.01))

    assert torch.allclose(sde.sigma(t), torch.tensor(expected_sigma, dtype=torch.float32), atol=1e-6)
    assert torch.allclose(sde.g(t), torch.tensor(expected_g, dtype=torch.float32), atol=1e-5)


def test_ve_marginal_mean_var_std():
    sde = _make_ve(sigma_min=0.05, sigma_max=2.0)
    theta0 = torch.randn(8, 3)
    t = 0.7

    sigma = sde.sigma(t)
    assert torch.allclose(sde.marginal_std(t), sigma)
    assert torch.allclose(sde.marginal_var(t), sigma**2)
    assert torch.allclose(sde.marginal_mean(theta0, t), theta0)
    # Zero drift for VE.
    assert torch.allclose(sde.f(theta0, t), torch.zeros_like(theta0))


def test_ve_transition_sample_and_score():
    sde = _make_ve(sigma_min=0.01, sigma_max=1.0)
    torch.manual_seed(0)
    theta0 = torch.randn(12, 4)
    t = 0.35

    sample = sde.transition_sample(theta0, t)
    assert sample.shape == theta0.shape

    var = sde.marginal_var(t)
    score = sde.transition_score(sample, theta0, t)
    expected_score = -(sample - theta0) / var
    assert torch.allclose(score, expected_score, atol=1e-6)

    # The transition is exactly theta0 + sqrt(var) * eps, so the mean over many
    # draws must approach theta0.
    many = torch.stack([sde.transition_sample(theta0, t) for _ in range(200)])
    assert torch.allclose(many.mean(dim=0), theta0, atol=0.05)
    assert torch.allclose(many.var(dim=0), torch.full_like(theta0, float(var)), atol=0.05)


def test_ve_score_finite_at_zero():
    sde = _make_ve()
    theta0 = torch.randn(5, 2)
    theta_t = sde.transition_sample(theta0, 0.0)
    score = sde.transition_score(theta_t, theta0, 0.0)
    assert torch.isfinite(score).all()


def test_ve_prior_sample_and_logp():
    sde = _make_ve(sigma_min=0.01, sigma_max=1.0)
    prior = sde.prior_sample(20000, 4, torch.device("cpu"))
    assert prior.shape == (20000, 4)

    # VE prior at t=T is N(0, sigma_max^2 I).
    assert prior.mean(dim=0).abs().max().item() < 0.05
    assert torch.allclose(prior.var(dim=0), torch.tensor(1.0), atol=0.05)

    logp = sde.prior_logp(prior[:128])
    expected = isotropic_gaussian_logp(prior[:128], 0.0, sde.sigma(sde.T) ** 2)
    assert torch.allclose(logp, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# VP SDE
# ---------------------------------------------------------------------------

def test_vp_beta_f_g():
    sde = _make_vp(beta_min=0.1, beta_max=11.0)
    t = 0.3
    beta = 0.1 + 0.3 * (11.0 - 0.1)
    theta = torch.randn(5, 3)

    assert torch.allclose(sde.beta(t), torch.tensor(beta, dtype=torch.float32), atol=1e-6)
    assert torch.allclose(sde.g(t), torch.tensor(math.sqrt(beta), dtype=torch.float32), atol=1e-6)
    assert torch.allclose(sde.f(theta, t), -0.5 * beta * theta, atol=1e-5)


def test_vp_beta_integral_and_marginals():
    sde = _make_vp(beta_min=0.1, beta_max=11.0)
    t = 0.3
    theta0 = torch.randn(6, 3)

    beta_int = 0.1 * t + 0.5 * (11.0 - 0.1) * t**2
    assert torch.allclose(sde.beta_int(t), torch.tensor(beta_int, dtype=torch.float32), atol=1e-6)

    expected_mean = math.exp(-0.5 * beta_int) * theta0
    expected_var = 1.0 - math.exp(-beta_int)
    assert torch.allclose(sde.marginal_mean(theta0, t), expected_mean, atol=1e-5)
    assert torch.allclose(sde.marginal_var(t), torch.tensor(expected_var, dtype=torch.float32), atol=1e-6)
    assert torch.allclose(sde.marginal_std(t), torch.sqrt(sde.marginal_var(t)), atol=1e-6)


def test_vp_transition_sample_and_score():
    sde = _make_vp()
    torch.manual_seed(1)
    theta0 = torch.randn(16, 3)
    t = 0.55

    sample = sde.transition_sample(theta0, t)
    assert sample.shape == theta0.shape

    mean = sde.marginal_mean(theta0, t)
    var = sde.marginal_var(t)
    score = sde.transition_score(sample, theta0, t)
    expected_score = -(sample - mean) / var
    assert torch.allclose(score, expected_score, atol=1e-6)

    many = torch.stack([sde.transition_sample(theta0, t) for _ in range(200)])
    assert torch.allclose(many.mean(dim=0), mean, atol=0.05)
    assert torch.allclose(many.var(dim=0), torch.full_like(theta0, float(var)), atol=0.05)


def test_vp_prior_sample_and_logp():
    sde = _make_vp()
    prior = sde.prior_sample(20000, 3, torch.device("cpu"))
    assert prior.shape == (20000, 3)

    # VP prior at t=T is close to standard normal.
    assert prior.mean(dim=0).abs().max().item() < 0.05
    assert torch.allclose(prior.var(dim=0), torch.tensor(1.0), atol=0.05)

    logp = sde.prior_logp(prior[:64])
    expected = isotropic_gaussian_logp(prior[:64], 0.0, sde.marginal_var(sde.T))
    assert torch.allclose(logp, expected, atol=1e-4)


# ---------------------------------------------------------------------------
# Reverse drift / probability flow velocity
# ---------------------------------------------------------------------------

def test_reverse_drift_and_ode_velocity():
    sde = _make_vp()
    theta = torch.randn(7, 2, requires_grad=False)
    t = 0.4
    score = torch.randn_like(theta)

    g2 = sde.g(t) ** 2
    drift = sde.f(theta, t)

    expected_reverse = drift - g2 * score
    expected_ode = drift - 0.5 * g2 * score

    assert torch.allclose(sde.reverse_drift(theta, t, score), expected_reverse, atol=1e-6)
    assert torch.allclose(sde.ode_velocity(theta, t, score), expected_ode, atol=1e-6)


def test_reverse_sde_step_deterministic():
    sde = _make_vp()
    theta = torch.randn(4, 3)
    t = 0.6
    dt = 0.01
    score = torch.randn_like(theta)
    noise = torch.zeros_like(theta)

    step = reverse_sde_step(sde, theta, t, dt, score, noise=noise)
    expected = theta - sde.reverse_drift(theta, t, score) * dt
    assert step.shape == theta.shape
    assert torch.allclose(step, expected, atol=1e-6)


def test_reverse_sde_step_shape_with_random_noise():
    sde = _make_ve()
    theta = torch.randn(5, 2)
    step = reverse_sde_step(sde, theta, 0.4, 0.02, torch.randn_like(theta), noise=None)
    assert step.shape == theta.shape
    assert torch.isfinite(step).all()


# ---------------------------------------------------------------------------
# Factory / helpers
# ---------------------------------------------------------------------------

def test_get_sde_factory():
    ve = get_sde("ve", sigma_min=0.01, sigma_max=1.0)
    vp = get_sde("VP", beta_min=0.1, beta_max=11.0)
    assert isinstance(ve, VESDE)
    assert isinstance(vp, VPSDE)

    with pytest.raises((ValueError, KeyError)):
        get_sde("not_a_sde")


def test_estimate_sigma_max():
    samples = torch.tensor([[0.0, 0.0], [3.0, 4.0], [1.0, 1.0]])
    sigma_max = estimate_sigma_max(samples, max_samples=1000)
    assert isinstance(sigma_max, float)
    assert math.isclose(sigma_max, 5.0, rel_tol=1e-5)


def test_isotropic_gaussian_logp():
    x = torch.randn(6, 3)
    mean = torch.zeros(3)
    var = 2.0
    logp = isotropic_gaussian_logp(x, mean, var)

    expected = -0.5 * ((x - mean) ** 2 / var).sum(dim=-1) - 0.5 * 3.0 * math.log(2.0 * math.pi * var)
    assert logp.shape == (6,)
    assert torch.allclose(logp, expected, atol=1e-5)


def test_ensure_tensor():
    t = ensure_tensor(0.5, torch.device("cpu"), torch.float32)
    assert isinstance(t, torch.Tensor)
    assert t.dim() == 0
    assert t.dtype == torch.float32

    arr = ensure_tensor([1.0, 2.0, 3.0], torch.device("cpu"), torch.float64)
    assert arr.shape == (3,)
    assert arr.dtype == torch.float64


# ---------------------------------------------------------------------------
# Simple manual runner (usable without pytest if desired)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {fn.__name__}: {exc!r}")
    if failures:
        raise SystemExit(f"{failures} test(s) failed")
    print(f"All {len(tests)} tests passed.")
