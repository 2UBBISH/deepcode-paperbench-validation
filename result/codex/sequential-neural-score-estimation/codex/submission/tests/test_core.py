"""Unit tests for the core (TS)NPSE machinery.

These tests are intentionally small: they verify the mathematical machinery
(forward SDE transitions, the instantaneous change-of-variables formula, the
HPR truncation) and that NPSE / TSNPSE run end-to-end on toy problems.
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snpse import (  # noqa: E402
    DiffusionConfig,
    DiffusionPosterior,
    NPSE,
    ScoreNetwork,
    Standardizer,
    TSNPSE,
    TSNPSEConfig,
    TrainingConfig,
    VESDE,
    VPSDE,
    compute_sigma_max,
)
from snpse.normalization import StandardizedDistribution  # noqa: E402
from snpse.proposals import TruncatedProposal, estimate_truncation_boundary  # noqa: E402


class GaussianLinearToy:
    """``p(theta) = N(0, s0^2 I)`` and ``p(x | theta) = N(theta, s1^2 I)``."""

    def __init__(self, dim: int = 2, s0: float = 1.0, s1: float = 0.5) -> None:
        self.dim = dim
        self.s0, self.s1 = s0, s1

    def get_prior_dist(self):
        return torch.distributions.Independent(
            torch.distributions.Normal(torch.zeros(self.dim), self.s0), 1
        )

    def sample_prior(self, n: int) -> torch.Tensor:
        return self.get_prior_dist().sample((n,))

    def get_simulator(self):
        s1 = self.s1

        def simulator(theta: torch.Tensor) -> torch.Tensor:
            return theta + s1 * torch.randn_like(theta)

        return simulator

    def reference_posterior_samples(self, x_obs: torch.Tensor, n: int) -> torch.Tensor:
        prec = 1.0 / self.s0 ** 2 + 1.0 / self.s1 ** 2
        var = 1.0 / prec
        mean = var * x_obs.reshape(-1) / self.s1 ** 2
        return mean + math.sqrt(var) * torch.randn(n, self.dim)


def test_vp_sde_transition_matches_analytic():
    sde = VPSDE(beta_min=0.1, beta_max=11.0)
    theta0 = torch.randn(4, 3)
    t = torch.tensor([0.0, 0.25, 0.5, 1.0])
    mean, std = sde.transition(theta0, t)
    integral = sde.beta_min * t + 0.5 * (sde.beta_max - sde.beta_min) * t ** 2
    assert torch.allclose(mean, theta0 * torch.exp(-0.5 * integral).reshape(-1, 1), atol=1e-6)
    assert torch.allclose(std.reshape(-1), torch.sqrt(1.0 - torch.exp(-integral)), atol=1e-6)


def test_ve_sde_transition_and_sigma_max():
    sde = VESDE(sigma_min=0.01, sigma_max=2.0)
    t = torch.tensor([0.0, 1.0])
    _, std = sde.transition(torch.zeros(2, 1), t)
    assert torch.allclose(std.reshape(-1), torch.tensor([0.01, 2.0]), atol=1e-6)
    theta = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    assert abs(compute_sigma_max(theta, technique=1) - 5.0) < 1e-6


def _diffusion_posterior_with_analytic_velocity(sde, mu, sd, dim):
    """A ``DiffusionPosterior`` whose velocity is the probability flow of the
    Gaussian path ``N(mu (1-t), (sd + (1-sd) t)^2)`` with reference ``N(0, 1)``."""
    net = ScoreNetwork(dim, dim)
    dp = DiffusionPosterior(net, sde, dim, DiffusionConfig(rtol=1e-7, atol=1e-7))

    def velocity(z, x, t):
        tt = float(t.reshape(-1)[0])
        a = mu * (1 - tt)
        b = sd + (1 - sd) * tt
        return -mu + ((1 - sd) / b) * (z - a)

    dp.velocity = velocity
    return dp


def test_change_of_variables_matches_analytic_gaussian():
    dim = 2
    sde = VESDE(sigma_min=0.01, sigma_max=1.0)
    mu = torch.tensor([1.0, -2.0])
    sd = torch.tensor([0.5, 0.25])
    dp = _diffusion_posterior_with_analytic_velocity(sde, mu, sd, dim)
    z = mu + sd * torch.randn(8, dim)
    lp = dp.log_prob(torch.zeros(1), z)
    analytic = (-0.5 * torch.log(2 * torch.pi * sd ** 2) - 0.5 * ((z - mu) / sd) ** 2).sum(-1)
    offset = lp - analytic
    # the difference must be constant in z (it is the log normalising constant)
    assert float(offset.std()) < 1e-3
    # and, because the reference distribution used in Eq. (5) coincides exactly
    # with p_T for this path, the normalising constant must vanish: this checks
    # that the density returned by the change-of-variables formula is normalised
    assert float(offset.abs().max()) < 1e-2


def test_sampling_and_log_prob_agree():
    dim = 2
    sde = VESDE(sigma_min=0.01, sigma_max=1.0)
    mu = torch.tensor([1.0, -2.0])
    sd = torch.tensor([0.5, 0.25])
    dp = _diffusion_posterior_with_analytic_velocity(sde, mu, sd, dim)
    z, lp_gen = dp.sample_and_log_prob(torch.zeros(1), 16, seed=0)
    lp_fwd = dp.log_prob(torch.zeros(1), z)
    assert float((lp_gen - lp_fwd).abs().max()) < 1e-3


def test_npse_recovers_gaussian_posterior():
    torch.manual_seed(0)
    problem = GaussianLinearToy(dim=2, s0=1.0, s1=0.5)
    x_obs = torch.tensor([[1.0, -1.0]])
    config = TrainingConfig(max_iters=1500, patience=500, batch_size=128)
    est = NPSE(2, 2, problem.get_prior_dist(), sde="ve", sigma_min=0.05, training_config=config)
    theta = problem.sample_prior(5000)
    x = problem.get_simulator()(theta)
    est.fit(theta, x, verbose=False)
    samples = est.sample(x_obs, 4000, seed=0)
    ref = problem.reference_posterior_samples(x_obs, 4000)
    assert float((samples.mean(0) - ref.mean(0)).norm()) < 0.25
    # the posterior standard deviation should be within a factor of ~1.6
    ratio = (samples.std(0) / ref.std(0)).mean()
    assert 0.6 < float(ratio) < 1.6


def test_tsnpse_runs_and_truncates():
    torch.manual_seed(0)
    problem = GaussianLinearToy(dim=1, s0=1.0, s1=0.5)
    config = TSNPSEConfig(num_rounds=2, num_simulations=200, hpr_num_samples=200, hpr_eps=5e-4)
    training = TrainingConfig(max_iters=100, patience=50, batch_size=64)
    est = TSNPSE(
        1,
        1,
        problem.get_prior_dist(),
        problem.get_simulator(),
        sde="ve",
        sigma_min=0.05,
        config=config,
        training_config=training,
        seed=0,
    )
    x_obs = torch.tensor([[0.5]])
    est.run(x_obs, verbose=False)
    assert len(est.diagnostics) == 2
    assert len(est.boundaries) == 1
    assert math.isfinite(est.boundaries[0].kappa)
    samples = est.npse.sample(x_obs, 500, seed=0)
    ref = problem.reference_posterior_samples(x_obs, 500)
    assert float((samples.mean(0) - ref.mean(0)).norm()) < 0.6


def test_truncated_proposal_rejection_sampling():
    """The truncated proposal must only return samples above the threshold."""
    torch.manual_seed(0)
    sde = VESDE(sigma_min=0.01, sigma_max=1.0)

    class _AnalyticNet(torch.nn.Module):
        """Exact score of a perturbed Gaussian posterior under the VE SDE."""

        def __init__(self, mu, sd):
            super().__init__()
            self.mu, self.sd = mu, sd

        def forward(self, theta_t, x, t):
            var = self.sd ** 2 + sde.std(t.reshape(-1)).reshape(-1, 1) ** 2
            return -(theta_t - self.mu) / var

    mu = torch.tensor([0.5])
    sd = torch.tensor([0.4])
    dp = DiffusionPosterior(_AnalyticNet(mu, sd), sde, 1, DiffusionConfig(rtol=1e-6, atol=1e-6))
    boundary = estimate_truncation_boundary(dp, torch.zeros(1), num_samples=2000, eps=5e-4)

    prior = StandardizedDistribution(
        torch.distributions.Independent(torch.distributions.Normal(torch.zeros(1), 1.0), 1),
        Standardizer(mean=torch.zeros(1), std=torch.ones(1)),
    )
    prop = TruncatedProposal(prior, boundary, lambda z: dp.log_prob(torch.zeros(1), z))
    samples = prop.sample(200, batch_size=2048, max_attempts=200)
    lp = dp.log_prob(torch.zeros(1), samples)
    assert float(lp.min()) > boundary.kappa
    assert prop.boundary.acceptance_rate < 1.0


def test_standardized_prior_density_is_proportional_to_prior():
    """Proposition 3.1: inside a uniform prior the truncated proposal has a
    constant density, i.e. it is proportional to the prior."""
    """Proposition 3.1 requires the truncated proposal to be proportional to the
    prior on the support of the posterior; inside a uniform prior this reduces
    to a constant density."""
    prior = torch.distributions.Independent(
        torch.distributions.Uniform(torch.tensor([-3.0]), torch.tensor([3.0])), 1
    )
    scaler = Standardizer(mean=torch.zeros(1), std=torch.ones(1))
    standardised = StandardizedDistribution(prior, scaler)
    z = torch.linspace(-2.0, 2.0, 11).reshape(-1, 1)
    lp = standardised.log_prob(z)
    assert float(lp.std()) < 1e-5


# --------------------------------------------------------------------------- #
# Alternative sequential methods (Section 3.2 / Appendix C)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("variant", ["a", "b", "c"])
def test_snpse_variants_run(variant):
    from snpse import SNPSEA, SNPSEB, SNPSEC, VariantConfig

    torch.manual_seed(0)
    problem = GaussianLinearToy(dim=1, s0=1.0, s1=0.5)
    config = VariantConfig(num_rounds=2, num_simulations=100, verbose=False)
    training = TrainingConfig(max_iters=40, patience=20, batch_size=64)
    cls = {"a": SNPSEA, "b": SNPSEB, "c": SNPSEC}[variant]
    est = cls(
        1,
        1,
        problem.get_prior_dist(),
        problem.get_simulator(),
        sde="ve",
        sigma_min=0.05,
        config=config,
        training_config=training,
        seed=0,
    )
    x_obs = torch.tensor([[0.5]])
    est.run(x_obs, verbose=False)
    assert len(est.diagnostics) == 2
    # every variant must be able to produce posterior samples
    samples = est.sample(x_obs, 50) if variant == "a" else est.npse.sample(x_obs, 50, seed=0)
    assert samples.shape == (50, 1)
    assert torch.isfinite(samples).all()
