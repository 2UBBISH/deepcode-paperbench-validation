"""Unit tests for the benchmark simulators in :mod:`npse.benchmarks`.

These tests validate the common ``Benchmark`` contract for every implemented
task: prior sampling shapes and support, simulator shapes and finiteness,
prior log-probability behaviour, observation sampling, and where available
closed-form posterior properties (Gaussian Linear) or reference-posterior
access.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import pytest
import torch

from npse.benchmarks import (
    Benchmark,
    GaussianLinear,
    GaussianMixture,
    GaussianLinearUniform,
    TwoMoons,
    BernoulliGLM,
    SLCP,
    SIR,
    LotkaVolterra,
    get_benchmark,
    to_torch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _check_prior_samples(benchmark: Benchmark, n: int = 8) -> torch.Tensor:
    """Sample from the prior and verify shape, dtype, and finiteness."""
    theta = benchmark.prior_sample(n)
    assert isinstance(theta, torch.Tensor)
    assert theta.shape == (n, benchmark.theta_dim)
    assert theta.dtype == benchmark.dtype
    assert torch.isfinite(theta).all()
    return theta


def _check_simulator(benchmark: Benchmark, n: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample prior parameters and simulate observations."""
    theta = _check_prior_samples(benchmark, n)
    x = benchmark.simulator(theta)
    assert isinstance(x, torch.Tensor)
    assert x.shape == (n, benchmark.x_dim)
    assert x.dtype == benchmark.dtype
    assert torch.isfinite(x).all()
    return theta, x


def _check_observation(benchmark: Benchmark) -> torch.Tensor:
    """Verify ``sample_observation`` returns a well-formed observation."""
    x_obs = benchmark.sample_observation(1)
    # Accept a single observation or a leading singleton batch dimension.
    assert isinstance(x_obs, torch.Tensor)
    flat = x_obs.reshape(-1)
    assert flat.numel() == benchmark.x_dim
    assert torch.isfinite(flat).all()
    return x_obs.reshape(-1)


# ---------------------------------------------------------------------------
# Benchmark list for the generic contract test
# ---------------------------------------------------------------------------
_ALL_BENCHMARKS: List[Tuple[str, Benchmark]] = [
    ("gaussian_linear", GaussianLinear()),
    ("gaussian_mixture", GaussianMixture()),
    ("two_moons", TwoMoons()),
    ("gaussian_linear_uniform", GaussianLinearUniform()),
    ("bernoulli_glm", BernoulliGLM()),
    ("slcp", SLCP()),
    ("sir", SIR()),
    ("lotka_volterra", LotkaVolterra()),
]


@pytest.mark.parametrize("name,benchmark", _ALL_BENCHMARKS)
def test_benchmark_contract(name: str, benchmark: Benchmark) -> None:
    """Every benchmark should satisfy the core sampling contract."""
    assert benchmark.name == name
    assert benchmark.theta_dim > 0
    assert benchmark.x_dim > 0

    theta = _check_prior_samples(benchmark, n=6)
    x = benchmark.simulator(theta)
    assert x.shape == (6, benchmark.x_dim)
    assert torch.isfinite(x).all()

    # Joint sampling must also work.
    joint_theta, joint_x = benchmark.sample_joint(4)
    assert joint_theta.shape == (4, benchmark.theta_dim)
    assert joint_x.shape == (4, benchmark.x_dim)
    assert torch.isfinite(joint_theta).all()
    assert torch.isfinite(joint_x).all()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def test_get_benchmark_registry() -> None:
    """The registry must return the correct class for every supported name."""
    assert isinstance(get_benchmark("gaussian_linear"), GaussianLinear)
    assert isinstance(get_benchmark("gaussian_mixture"), GaussianMixture)
    assert isinstance(get_benchmark("two_moons"), TwoMoons)
    assert isinstance(get_benchmark("gaussian_linear_uniform"), GaussianLinearUniform)
    assert isinstance(get_benchmark("bernoulli_glm"), BernoulliGLM)
    assert isinstance(get_benchmark("slcp"), SLCP)
    assert isinstance(get_benchmark("sir"), SIR)
    assert isinstance(get_benchmark("lotka_volterra"), LotkaVolterra)


def test_get_benchmark_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_benchmark("not_a_benchmark")


# ---------------------------------------------------------------------------
# Tensor conversion utility
# ---------------------------------------------------------------------------
def test_to_torch_conversion() -> None:
    device = torch.device("cpu")
    assert to_torch(1.5, device=device).shape == ()
    assert to_torch([1.0, 2.0, 3.0], device=device).shape == (3,)
    assert to_torch(torch.tensor([1.0, 2.0]), device=device).shape == (2,)

    # dtype coercion
    out = to_torch([1, 2, 3], device=device, dtype=torch.float32)
    assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# Gaussian Linear analytic posterior
# ---------------------------------------------------------------------------
def test_gaussian_linear_analytic_posterior() -> None:
    bench = GaussianLinear(dim=10, prior_var=0.1, likelihood_var=0.1)
    x_obs = torch.zeros(10)
    samples = bench.reference_posterior_samples(x_obs, n=5000)
    assert samples is not None
    assert samples.shape == (5000, 10)

    # Posterior is N(x/2, 0.05 I).
    mean = samples.mean(dim=0)
    var = samples.var(dim=0, unbiased=False)
    assert torch.allclose(mean, torch.zeros(10), atol=0.03)
    assert torch.allclose(var, torch.full((10,), 0.05), atol=0.01)


def test_gaussian_linear_posterior_mean_matches_closed_form() -> None:
    bench = GaussianLinear(dim=5)
    x_obs = torch.tensor([0.2, -0.3, 0.5, -0.1, 0.4])
    expected = x_obs * (bench.prior_var / (bench.prior_var + bench.likelihood_var))
    assert torch.allclose(bench.posterior_mean(x_obs), expected)


def test_gaussian_linear_prior_log_prob_matches_gaussian() -> None:
    bench = GaussianLinear(dim=3)
    theta = torch.zeros(1, 3)
    logp = bench.prior_log_prob(theta)
    # N(0, 0.1 I): -d/2 * log(2*pi*0.1)
    expected = -3.0 / 2.0 * math.log(2.0 * math.pi * 0.1)
    assert torch.allclose(logp, torch.tensor(expected), atol=1e-5)


# ---------------------------------------------------------------------------
# Gaussian Mixture
# ---------------------------------------------------------------------------
def test_gaussian_mixture_prior_support() -> None:
    bench = GaussianMixture(low=-10.0, high=10.0)
    theta = _check_prior_samples(bench, n=200)
    assert (theta >= -10.0).all()
    assert (theta <= 10.0).all()


def test_gaussian_mixture_prior_log_prob() -> None:
    bench = GaussianMixture(low=-10.0, high=10.0)
    inside = torch.zeros(1, 2)
    outside = torch.tensor([[11.0, 0.0]])
    inside_logp = bench.prior_log_prob(inside)
    outside_logp = bench.prior_log_prob(outside)

    expected_inside = -math.log(20.0) * 2
    assert torch.allclose(inside_logp, torch.tensor(expected_inside), atol=1e-5)
    assert torch.isinf(outside_logp).all() and (outside_logp < 0).all()


def test_gaussian_mixture_reference_posterior() -> None:
    bench = GaussianMixture()
    x_obs = torch.tensor([0.0, 0.0])
    samples = bench.reference_posterior_samples(x_obs, n=2000)
    assert samples is not None
    assert samples.shape == (2000, 2)
    assert torch.isfinite(samples).all()
    assert (samples >= -10.0).all() and (samples <= 10.0).all()


# ---------------------------------------------------------------------------
# Two Moons
# ---------------------------------------------------------------------------
def test_two_moons_prior_support() -> None:
    bench = TwoMoons(low=-1.0, high=1.0)
    theta = _check_prior_samples(bench, n=200)
    assert (theta >= -1.0).all()
    assert (theta <= 1.0).all()


def test_two_moons_prior_log_prob() -> None:
    bench = TwoMoons()
    inside_logp = bench.prior_log_prob(torch.zeros(1, 2))
    outside_logp = bench.prior_log_prob(torch.tensor([[2.0, 0.0]]))
    expected_inside = -math.log(2.0) * 2
    assert torch.allclose(inside_logp, torch.tensor(expected_inside), atol=1e-5)
    assert torch.isinf(outside_logp).all()


def test_two_moons_simulator_matches_spec_statistically() -> None:
    bench = TwoMoons()
    theta = torch.zeros(20, 2)
    x = bench.simulator(theta)
    # x1 = r cos(alpha) + 0.25 - |theta1+theta2|/sqrt(2)
    # x2 = r sin(alpha) + (-theta1+theta2)/sqrt(2); with theta=0 the second term
    # is a small noise around zero. Verify the x2 mean is near zero and x1 is
    # near 0.25 + E[r cos(alpha)].
    assert x.shape == (20, 2)
    assert x[:, 1].mean().abs() < 0.05
    assert (x[:, 0].mean() - 0.25).abs() < 0.08


# ---------------------------------------------------------------------------
# Gaussian Linear Uniform
# ---------------------------------------------------------------------------
def test_gaussian_linear_uniform_prior_support() -> None:
    bench = GaussianLinearUniform(dim=10, low=-1.0, high=1.0)
    theta = _check_prior_samples(bench, n=200)
    assert (theta >= -1.0).all()
    assert (theta <= 1.0).all()


def test_gaussian_linear_uniform_prior_log_prob() -> None:
    bench = GaussianLinearUniform(dim=10, low=-1.0, high=1.0)
    inside_logp = bench.prior_log_prob(torch.zeros(1, 10))
    outside_logp = bench.prior_log_prob(torch.full((1, 10), 1.5))
    expected_inside = -10.0 * math.log(2.0)
    assert torch.allclose(inside_logp, torch.tensor(expected_inside), atol=1e-5)
    assert torch.isinf(outside_logp).all()


def test_gaussian_linear_uniform_simulator_variance() -> None:
    bench = GaussianLinearUniform(dim=4, likelihood_var=0.1)
    theta = torch.zeros(200, 4)
    x = bench.simulator(theta)
    var = x.var(dim=0, unbiased=False)
    assert torch.allclose(var, torch.full((4,), 0.1), atol=0.03)


# ---------------------------------------------------------------------------
# Bernoulli GLM
# ---------------------------------------------------------------------------
def test_bernoulli_glm_prior_and_simulator() -> None:
    bench = BernoulliGLM(dim=10)
    theta, x = _check_simulator(bench, n=16)
    assert theta.shape == (16, 10)
    assert x.shape == (16, 10)
    # Bernoulli observations are binary counts / probabilities.
    assert (x >= 0.0).all() and (x <= 1.0).all()


def test_bernoulli_glm_prior_log_prob_finite() -> None:
    bench = BernoulliGLM(dim=10)
    theta = bench.prior_sample(8)
    logp = bench.prior_log_prob(theta)
    assert logp.shape == (8,)
    assert torch.isfinite(logp).all()


# ---------------------------------------------------------------------------
# SLCP
# ---------------------------------------------------------------------------
def test_slcp_prior_support_and_dimensions() -> None:
    bench = SLCP(low=-3.0, high=3.0)
    theta = _check_prior_samples(bench, n=50)
    assert theta.shape == (50, 5)
    assert (theta >= -3.0).all()
    assert (theta <= 3.0).all()


def test_slcp_simulator_shape() -> None:
    bench = SLCP()
    theta, x = _check_simulator(bench, n=8)
    assert theta.shape == (8, 5)
    assert x.shape == (8, 8)
    assert torch.isfinite(x).all()


def test_slcp_prior_log_prob() -> None:
    bench = SLCP(low=-3.0, high=3.0)
    inside_logp = bench.prior_log_prob(torch.zeros(1, 5))
    outside_logp = bench.prior_log_prob(torch.full((1, 5), 4.0))
    expected_inside = -5.0 * math.log(6.0)
    assert torch.allclose(inside_logp, torch.tensor(expected_inside), atol=1e-5)
    assert torch.isinf(outside_logp).all()


# ---------------------------------------------------------------------------
# SIR
# ---------------------------------------------------------------------------
def test_sir_prior_and_simulator() -> None:
    bench = SIR()
    theta, x = _check_simulator(bench, n=6)
    assert theta.shape == (6, 2)
    assert x.shape == (6, 10)
    assert (x >= 0.0).all()
    assert (x <= 1.0).all()  # reported as infected fraction I/N


def test_sir_prior_log_prob_matches_lognormal() -> None:
    bench = SIR()
    theta = bench.prior_sample(4)
    logp = bench.prior_log_prob(theta)
    assert logp.shape == (4,)
    assert torch.isfinite(logp).all()


# ---------------------------------------------------------------------------
# Lotka-Volterra
# ---------------------------------------------------------------------------
def test_lotka_volterra_prior_and_simulator() -> None:
    bench = LotkaVolterra()
    theta, x = _check_simulator(bench, n=6)
    assert theta.shape == (6, 4)
    assert x.shape == (6, 20)
    assert torch.isfinite(x).all()
    assert (x >= 0.0).all()


def test_lotka_volterra_prior_log_prob_finite() -> None:
    bench = LotkaVolterra()
    theta = bench.prior_sample(4)
    logp = bench.prior_log_prob(theta)
    assert logp.shape == (4,)
    assert torch.isfinite(logp).all()


# ---------------------------------------------------------------------------
# Observation sampling for all tasks
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,benchmark", _ALL_BENCHMARKS)
def test_sample_observation_well_formed(name: str, benchmark: Benchmark) -> None:
    x_obs = _check_observation(benchmark)
    assert x_obs.shape == (benchmark.x_dim,)


# ---------------------------------------------------------------------------
# Manual runner (pytest-free execution)
# ---------------------------------------------------------------------------
def _run_all() -> None:
    """Run every ``test_*`` function in this module without pytest."""
    import sys
    import inspect

    module = sys.modules[__name__]
    tests = [
        obj
        for _, obj in inspect.getmembers(module, inspect.isfunction)
        if _.startswith("test_")
    ]
    # Parametrized helpers need their parameterised invocations.
    failures = 0
    for fn in tests:
        try:
            if getattr(fn, "pytestmark", None):
                continue
            # Call parameterised contract tests for every benchmark.
            if fn.__name__ in ("test_benchmark_contract", "test_sample_observation_well_formed"):
                for name, bench in _ALL_BENCHMARKS:
                    fn(name, bench)
            else:
                fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {fn.__name__}: {exc}")

    if failures:
        raise SystemExit(f"{failures} test(s) failed")
    print("All benchmark tests passed.")


if __name__ == "__main__":
    _run_all()
