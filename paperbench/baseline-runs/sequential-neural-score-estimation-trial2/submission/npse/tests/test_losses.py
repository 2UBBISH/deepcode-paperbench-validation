"""Unit tests for the denoising score-matching losses in ``npse/src/losses.py``.

The tests cover:
  * time sampling range and shape
  * weighting selection (``g2``, ``none``/``None``, and callables)
  * NPSE / prior / NLSE / weighted SNPSE-B loss computations
  * zero loss for a perfect score network on a deterministic SDE
  * consistency between the scalar loss and the per-sample squared error
  * finiteness and non-negativity when using real networks and VE/VP SDEs
"""

from __future__ import annotations

import pytest
import torch

from npse.src.losses import (
    get_weighting,
    nlse_dsm_loss,
    npse_dsm_loss,
    npse_dsm_squared_error,
    prior_dsm_loss,
    sample_times,
    weighted_npse_dsm_loss,
)
from npse.src.networks import PriorScoreNetwork, ScoreNetwork
from npse.src.sde import VESDE, VPSDE


# ---------------------------------------------------------------------------
# Deterministic SDE and dummy score networks used for exact loss bookkeeping.
# ---------------------------------------------------------------------------
class _DeterministicSDE:
    """SDE whose transition kernel is deterministic.

    The transition is chosen so that the target score simplifies to
    ``theta0 - theta_t = -t``, which makes expected-loss calculations exact.
    """

    def __init__(self, T: float = 1.0):
        self.T = T

    def g(self, t):
        return torch.ones_like(t)

    def transition_sample(self, theta0, t):
        # theta_t = theta0 + t (broadcast over the parameter dimension)
        return theta0 + t.unsqueeze(-1)

    def transition_score(self, theta_t, theta0, t):
        # target score = theta0 - theta_t = -t
        return theta0 - theta_t


class _ScoreNet:
    """Callable stand-in for :class:`ScoreNetwork` with a fixed output policy."""

    def __init__(self, mode: str):
        self.mode = mode

    def __call__(self, theta, x, t):
        if self.mode == "zero":
            return torch.zeros_like(theta)
        if self.mode == "perfect":
            # matches the deterministic SDE target theta0 - theta_t = -t
            return -t.unsqueeze(-1).expand_as(theta)
        if self.mode == "theta":
            return theta
        raise ValueError(f"Unknown mode: {self.mode}")


class _PriorScoreNet:
    """Callable stand-in for :class:`PriorScoreNetwork`."""

    def __init__(self, mode: str):
        self.mode = mode

    def __call__(self, theta, t):
        if self.mode == "zero":
            return torch.zeros_like(theta)
        if self.mode == "perfect":
            return -t.unsqueeze(-1).expand_as(theta)
        raise ValueError(f"Unknown mode: {self.mode}")


def _zero_prior_score(theta, t):
    return torch.zeros_like(theta)


def _perfect_prior_score(theta, t):
    return -t.unsqueeze(-1).expand_as(theta)


def _make_fixture(batch: int = 16, dim: int = 3, seed: int = 0):
    torch.manual_seed(seed)
    sde = _DeterministicSDE()
    theta0 = torch.randn(batch, dim)
    x = torch.randn(batch, dim)
    t = torch.rand(batch) * sde.T + 1e-4
    return sde, theta0, x, t


# ---------------------------------------------------------------------------
# Time sampling
# ---------------------------------------------------------------------------
def test_sample_times_shape_and_range():
    t = sample_times(128, T=1.0, eps=1e-5)
    assert t.shape == (128,)
    assert torch.all(t >= 1e-5 - 1e-9)
    assert torch.all(t <= 1.0 + 1e-9)


def test_sample_times_respects_custom_T():
    t = sample_times(64, T=2.5, eps=1e-4)
    assert t.shape == (64,)
    assert torch.all(t >= 1e-4 - 1e-9)
    assert torch.all(t <= 2.5 + 1e-9)


# ---------------------------------------------------------------------------
# Weighting selection
# ---------------------------------------------------------------------------
def test_get_weighting_g2_matches_sde_g_squared():
    sde = VESDE(sigma_min=0.05, sigma_max=1.0)
    t = torch.tensor([0.0, 0.25, 0.5, 1.0])
    weights = get_weighting(sde, t, "g2")
    assert torch.allclose(weights, sde.g(t) ** 2)


def test_get_weighting_none_and_string_none():
    sde = VESDE(sigma_min=0.05, sigma_max=1.0)
    t = torch.tensor([0.0, 0.5, 1.0])
    for weighting in ("none", None):
        weights = get_weighting(sde, t, weighting)
        assert torch.allclose(weights, torch.ones_like(t))


def test_get_weighting_callable():
    sde = VESDE(sigma_min=0.05, sigma_max=1.0)
    t = torch.tensor([0.0, 0.5, 1.0])
    weights = get_weighting(sde, t, lambda s, tt: 3.0 * tt + 0.5)
    assert torch.allclose(weights, 3.0 * t + 0.5)


# ---------------------------------------------------------------------------
# NPSE posterior score-matching loss
# ---------------------------------------------------------------------------
def test_npse_dsm_squared_error_perfect_score_is_zero():
    sde, theta0, x, t = _make_fixture(dim=3)
    se = npse_dsm_squared_error(sde, _ScoreNet("perfect"), theta0, x, t)
    assert se.shape == (theta0.shape[0],)
    assert torch.allclose(se, torch.zeros_like(se))


def test_npse_dsm_loss_perfect_score_is_zero():
    sde, theta0, x, t = _make_fixture(dim=3)
    loss = npse_dsm_loss(sde, _ScoreNet("perfect"), theta0, x, t)
    assert loss.dim() == 0
    assert torch.allclose(loss, torch.zeros_like(loss))


def test_npse_dsm_loss_consistent_with_squared_error():
    sde, theta0, x, t = _make_fixture(dim=3)
    net = _ScoreNet("zero")
    actual = npse_dsm_loss(sde, net, theta0, x, t, weighting="g2")
    se = npse_dsm_squared_error(sde, net, theta0, x, t)
    weights = get_weighting(sde, t, "g2")
    expected = 0.5 * (weights * se).mean()
    assert torch.allclose(actual, expected)


def test_npse_dsm_loss_is_nonnegative_and_finite():
    sde, theta0, x, t = _make_fixture(dim=3)
    loss = npse_dsm_loss(sde, _ScoreNet("zero"), theta0, x, t)
    assert torch.isfinite(loss)
    assert loss >= 0.0


# ---------------------------------------------------------------------------
# Prior denoising score-matching loss
# ---------------------------------------------------------------------------
def test_prior_dsm_loss_perfect_score_is_zero():
    sde, theta0, x, t = _make_fixture(dim=3)
    loss = prior_dsm_loss(sde, _PriorScoreNet("perfect"), theta0, t)
    assert loss.dim() == 0
    assert torch.allclose(loss, torch.zeros_like(loss))


def test_prior_dsm_loss_matches_manual_formula():
    # dim=1 makes sum(dim=-1) and mean(dim=-1) identical, so the manual
    # calculation is robust to the internal per-sample reduction convention.
    sde, theta0, x, t = _make_fixture(dim=1)
    actual = prior_dsm_loss(sde, _PriorScoreNet("zero"), theta0, t, weighting="g2")
    theta_t = sde.transition_sample(theta0, t)
    target = sde.transition_score(theta_t, theta0, t)
    score = torch.zeros_like(theta_t)
    se = ((score - target) ** 2).sum(dim=-1)
    expected = 0.5 * (sde.g(t) * se).mean()
    assert torch.allclose(actual, expected)


# ---------------------------------------------------------------------------
# NLSE likelihood score-matching loss
# ---------------------------------------------------------------------------
def test_nlse_dsm_loss_decomposes_to_perfect_posterior_score():
    sde, theta0, x, t = _make_fixture(dim=3)
    # zero likelihood score + perfect prior score == perfect posterior score
    loss = nlse_dsm_loss(sde, _ScoreNet("zero"), _perfect_prior_score, theta0, x, t)
    assert loss.dim() == 0
    assert torch.allclose(loss, torch.zeros_like(loss))


def test_nlse_dsm_loss_matches_manual_formula():
    sde, theta0, x, t = _make_fixture(dim=1)
    actual = nlse_dsm_loss(
        sde, _ScoreNet("zero"), _zero_prior_score, theta0, x, t, weighting="g2"
    )
    theta_t = sde.transition_sample(theta0, t)
    target = sde.transition_score(theta_t, theta0, t)
    score = torch.zeros_like(theta_t) + torch.zeros_like(theta_t)
    se = ((score - target) ** 2).sum(dim=-1)
    expected = 0.5 * (sde.g(t) * se).mean()
    assert torch.allclose(actual, expected)


# ---------------------------------------------------------------------------
# Importance-weighted SNPSE-B loss
# ---------------------------------------------------------------------------
def test_weighted_npse_dsm_loss_applies_importance_weights():
    sde, theta0, x, t = _make_fixture(dim=1)
    importance = torch.rand(theta0.shape[0]) + 0.5
    actual = weighted_npse_dsm_loss(
        sde, _ScoreNet("zero"), theta0, x, t, importance, weighting="g2"
    )
    theta_t = sde.transition_sample(theta0, t)
    target = sde.transition_score(theta_t, theta0, t)
    se = ((torch.zeros_like(theta_t) - target) ** 2).sum(dim=-1)
    expected = 0.5 * (sde.g(t) * importance * se).mean()
    assert torch.allclose(actual, expected)


def test_weighted_npse_dsm_loss_perfect_score_zero():
    sde, theta0, x, t = _make_fixture(dim=3)
    importance = torch.rand(theta0.shape[0]) + 0.5
    loss = weighted_npse_dsm_loss(
        sde, _ScoreNet("perfect"), theta0, x, t, importance
    )
    assert torch.allclose(loss, torch.zeros_like(loss))


# ---------------------------------------------------------------------------
# Smoke tests with real SDEs and real networks
# ---------------------------------------------------------------------------
def test_npse_dsm_loss_real_sde_and_network():
    torch.manual_seed(0)
    sde = VESDE(sigma_min=0.05, sigma_max=1.0)
    net = ScoreNetwork(theta_dim=3, x_dim=3, hidden_dim=32)
    theta0 = torch.randn(16, 3)
    x = torch.randn(16, 3)
    t = sample_times(16, T=1.0)
    loss = npse_dsm_loss(sde, net, theta0, x, t)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss >= 0.0


def test_prior_dsm_loss_real_sde_and_network():
    torch.manual_seed(0)
    sde = VESDE(sigma_min=0.05, sigma_max=1.0)
    net = PriorScoreNetwork(theta_dim=3, hidden_dim=32)
    theta0 = torch.randn(16, 3)
    t = sample_times(16, T=1.0)
    loss = prior_dsm_loss(sde, net, theta0, t)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss >= 0.0


def test_nlse_dsm_loss_real_sde_and_network():
    torch.manual_seed(0)
    sde = VESDE(sigma_min=0.05, sigma_max=1.0)
    net = ScoreNetwork(theta_dim=3, x_dim=3, hidden_dim=32)
    theta0 = torch.randn(16, 3)
    x = torch.randn(16, 3)
    t = sample_times(16, T=1.0)
    loss = nlse_dsm_loss(sde, net, _zero_prior_score, theta0, x, t)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss >= 0.0


def test_npse_dsm_loss_vp_sde_finite():
    torch.manual_seed(0)
    sde = VPSDE(beta_min=0.1, beta_max=11.0)
    net = ScoreNetwork(theta_dim=3, x_dim=3, hidden_dim=32)
    theta0 = torch.randn(16, 3)
    x = torch.randn(16, 3)
    t = sample_times(16, T=1.0)
    loss = npse_dsm_loss(sde, net, theta0, x, t)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss >= 0.0


# ---------------------------------------------------------------------------
# Manual runner (usable without pytest)
# ---------------------------------------------------------------------------
def _run_all():
    tests = [
        value
        for key, value in sorted(globals().items())
        if key.startswith("test_") and callable(value)
    ]
    failures = []
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures.append((fn.__name__, exc))
            print(f"FAIL {fn.__name__}: {exc}")
    if failures:
        raise SystemExit(f"{len(failures)} test(s) failed")
    print(f"{len(tests)} tests passed")


if __name__ == "__main__":
    _run_all()
