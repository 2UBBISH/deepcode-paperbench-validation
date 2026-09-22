"""Unit tests for probability-flow ODE posterior sampling.

These tests validate the public sampling interfaces in
:mod:`npse.src.sampling` without requiring a trained score network. They
cover shape handling for single and batched observations, finiteness of the
returned tensors, the explicit Euler fallback, and the torchdiffeq-based RK45
path (which is skipped automatically when ``torchdiffeq`` is unavailable).
"""

from __future__ import annotations

import pytest
import torch

from npse.src.sampling import (
    ProbabilityFlowSampler,
    euler_probability_flow_sample,
    sample_probability_flow,
)
from npse.src.sde import VESDE, VPSDE

try:  # pragma: no cover - availability check
    import torchdiffeq  # noqa: F401

    HAS_TORCHDIFFEQ = True
except Exception:  # pragma: no cover
    HAS_TORCHDIFFEQ = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_ve(sigma_min: float = 0.01, sigma_max: float = 1.0) -> VESDE:
    """Construct a small VE SDE with known noise schedule."""
    return VESDE(sigma_min=sigma_min, sigma_max=sigma_max, T=1.0)


def _make_vp() -> VPSDE:
    """Construct the default linear-schedule VP SDE."""
    return VPSDE(beta_min=0.1, beta_max=11.0, T=1.0)


def _zero_score(theta: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Score function that returns zeros for every input."""
    return torch.zeros_like(theta)


def _linear_score(theta: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """A simple deterministic score used to exercise non-zero ODE velocities."""
    return theta


# ---------------------------------------------------------------------------
# Explicit Euler fallback
# ---------------------------------------------------------------------------
def test_euler_single_observation_shape():
    sde = _make_ve()
    x_obs = torch.randn(2)
    out = euler_probability_flow_sample(
        sde=sde,
        score_fn=_zero_score,
        n_samples=7,
        x_obs=x_obs,
        theta_dim=3,
        device="cpu",
        dtype=torch.float32,
        steps=20,
    )
    assert isinstance(out, torch.Tensor)
    assert out.shape == (7, 3)
    assert out.dtype == torch.float32


def test_euler_batched_observation_shape():
    sde = _make_ve()
    x_obs = torch.randn(4, 2)
    out = euler_probability_flow_sample(
        sde=sde,
        score_fn=_zero_score,
        n_samples=5,
        x_obs=x_obs,
        theta_dim=3,
        device="cpu",
        dtype=torch.float32,
        steps=20,
    )
    assert out.shape == (4, 5, 3)


def test_euler_output_is_finite():
    sde = _make_ve()
    x_obs = torch.randn(2)
    out = euler_probability_flow_sample(
        sde=sde,
        score_fn=_linear_score,
        n_samples=16,
        x_obs=x_obs,
        theta_dim=2,
        device="cpu",
        dtype=torch.float32,
        steps=50,
    )
    assert torch.isfinite(out).all()


def test_euler_zero_score_matches_ve_prior_distribution():
    # For a VE SDE with zero score the probability-flow velocity is zero, so
    # the ODE solution equals the initial prior draw theta_T ~ N(0, sigma_max^2).
    sde = _make_ve(sigma_min=0.01, sigma_max=1.0)
    x_obs = torch.randn(2)
    n = 6000
    out = euler_probability_flow_sample(
        sde=sde,
        score_fn=_zero_score,
        n_samples=n,
        x_obs=x_obs,
        theta_dim=2,
        device="cpu",
        dtype=torch.float32,
        steps=20,
    )
    assert out.shape == (n, 2)
    mean = out.mean().item()
    std = out.std().item()
    assert abs(mean) < 0.2
    assert 0.7 < std < 1.3


def test_euler_vp_finite_and_shape():
    sde = _make_vp()
    x_obs = torch.randn(2)
    out = euler_probability_flow_sample(
        sde=sde,
        score_fn=_linear_score,
        n_samples=8,
        x_obs=x_obs,
        theta_dim=2,
        device="cpu",
        dtype=torch.float32,
        steps=30,
    )
    assert out.shape == (8, 2)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# RK45 probability-flow integration (requires torchdiffeq)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAS_TORCHDIFFEQ, reason="torchdiffeq is not installed")
def test_probability_flow_single_observation_shape():
    sde = _make_ve()
    x_obs = torch.randn(2)
    out = sample_probability_flow(
        sde=sde,
        score_fn=_zero_score,
        n_samples=5,
        x_obs=x_obs,
        theta_dim=3,
        device="cpu",
        dtype=torch.float32,
        rtol=1e-4,
        atol=1e-4,
        method="rk45",
    )
    assert out.shape == (5, 3)


@pytest.mark.skipif(not HAS_TORCHDIFFEQ, reason="torchdiffeq is not installed")
def test_probability_flow_batched_observation_shape():
    sde = _make_ve()
    x_obs = torch.randn(3, 2)
    out = sample_probability_flow(
        sde=sde,
        score_fn=_zero_score,
        n_samples=4,
        x_obs=x_obs,
        theta_dim=2,
        device="cpu",
        dtype=torch.float32,
        rtol=1e-4,
        atol=1e-4,
        method="rk45",
    )
    assert out.shape == (3, 4, 2)


@pytest.mark.skipif(not HAS_TORCHDIFFEQ, reason="torchdiffeq is not installed")
def test_probability_flow_output_is_finite():
    sde = _make_ve()
    x_obs = torch.randn(2)
    out = sample_probability_flow(
        sde=sde,
        score_fn=_linear_score,
        n_samples=8,
        x_obs=x_obs,
        theta_dim=2,
        device="cpu",
        dtype=torch.float32,
        rtol=1e-4,
        atol=1e-4,
        method="rk45",
    )
    assert out.shape == (8, 2)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# ProbabilityFlowSampler class
# ---------------------------------------------------------------------------
def test_probability_flow_sampler_construction():
    sde = _make_ve()
    sampler = ProbabilityFlowSampler(sde=sde, score_fn=_zero_score, rtol=1e-4, atol=1e-4, method="rk45")
    assert hasattr(sampler, "sample")
    assert callable(sampler.sample)
    assert hasattr(sampler, "score")
    assert callable(sampler.score)


# ---------------------------------------------------------------------------
# Manual runner (usable without pytest)
# ---------------------------------------------------------------------------
def _run_all() -> None:
    """Run all tests manually and report pass/fail status."""
    tests = [
        test_euler_single_observation_shape,
        test_euler_batched_observation_shape,
        test_euler_output_is_finite,
        test_euler_zero_score_matches_ve_prior_distribution,
        test_euler_vp_finite_and_shape,
        test_probability_flow_single_observation_shape,
        test_probability_flow_batched_observation_shape,
        test_probability_flow_output_is_finite,
        test_probability_flow_sampler_construction,
    ]
    failures = []
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures.append((test.__name__, exc))
            print(f"FAIL {test.__name__}: {exc}")
    if failures:
        raise SystemExit(f"{len(failures)} test(s) failed")


if __name__ == "__main__":
    _run_all()
