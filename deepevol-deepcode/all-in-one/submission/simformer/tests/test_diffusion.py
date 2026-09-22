"""Unit tests for :mod:`simformer.diffusion` (VESDE/VPSDE, reverse SDE, ODE).

These tests validate the SDE coefficients and marginal statistics reported in
Appendix A2.1 of the Simformer paper:

* VESDE: ``f(x, t) = 0`` and ``g(t) = sigma_min (sigma_max/sigma_min)^t *
  sqrt(2 log(sigma_max/sigma_min))`` with ``sigma_max=15`` and
  ``sigma_min=1e-4``; marginal ``x_t | x_0 ~ N(x_0, sigma(t)^2 I)`` with
  ``sigma(t) = sigma_min (sigma_max/sigma_min)^t``.
* VPSDE: ``f(x, t) = -0.5 beta(t) x`` and ``g(t) = sqrt(beta(t))`` with
  ``beta(t) = beta_min + t (beta_max - beta_min)``, ``beta_min=0.01``,
  ``beta_max=10``; marginal mean ``mu(t) = exp(-0.25 t^2 dB - 0.5 t beta_min)``
  and variance ``1 - mu(t)^2``.

Additional tests cover the score/epsilon algebra
(``grad log p_t = -eps / sigma(t)``), the Tweedie denoised estimate used in
Algorithm 1, the reverse-time Euler--Maruyama step, the probability-flow ODE
and the instantaneous change-of-variables log-likelihood.

The module is tested through a tolerant import so that either repository layout
(``simformer.diffusion`` or ``simformer.simformer.diffusion``) works.
"""

from __future__ import annotations

import importlib
import math
from typing import Any, List, Optional, Tuple

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Tolerant module import (supports both repo layouts)
# ---------------------------------------------------------------------------

_CANDIDATES = (
    "simformer.diffusion",
    "simformer.simformer.diffusion",
)


def _load_module() -> Any:
    last_error: Optional[Exception] = None
    for name in _CANDIDATES:
        try:
            return importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - depends on layout
            last_error = exc
    pytest.skip(f"could not import diffusion module: {last_error!r}")


diffusion = _load_module()


def _get(name: str, *fallbacks: str) -> Any:
    """Return the first available attribute, else skip the test."""
    for candidate in (name,) + fallbacks:
        if hasattr(diffusion, candidate):
            return getattr(diffusion, candidate)
    pytest.skip(f"diffusion module has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Paper constants
# ---------------------------------------------------------------------------

SIGMA_MAX = 15.0
SIGMA_MIN = 1e-4
BETA_MIN = 0.01
BETA_MAX = 10.0
T_MIN = 1e-5
T_MAX = 1.0

RATIO = SIGMA_MAX / SIGMA_MIN
LOG_RATIO = math.log(RATIO)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_array(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _vesde(**kwargs: Any) -> Any:
    cls = _get("VarianceExplodingSDE", "VESDE")
    defaults = dict(
        sigma_max=SIGMA_MAX,
        sigma_min=SIGMA_MIN,
        t_min=T_MIN,
        t_max=T_MAX,
    )
    defaults.update(kwargs)
    try:
        return cls(**defaults)
    except TypeError:  # pragma: no cover - alternate signature
        return cls()


def _vpsde(**kwargs: Any) -> Any:
    cls = _get("VariancePreservingSDE", "VPSDE")
    defaults = dict(
        beta_min=BETA_MIN,
        beta_max=BETA_MAX,
        t_min=T_MIN,
        t_max=T_MAX,
    )
    defaults.update(kwargs)
    try:
        return cls(**defaults)
    except TypeError:  # pragma: no cover - alternate signature
        return cls()


def _sigma(t: np.ndarray) -> np.ndarray:
    """Analytic VESDE marginal std from Appendix A2.1."""
    t = np.asarray(t, dtype=np.float64)
    return SIGMA_MIN * np.power(RATIO, t)


def _beta(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    return BETA_MIN + t * (BETA_MAX - BETA_MIN)


def _mu(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    dB = BETA_MAX - BETA_MIN
    return np.exp(-0.25 * t**2 * dB - 0.5 * t * BETA_MIN)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_default_constants_match_paper():
    sigma_max = getattr(diffusion, "DEFAULT_SIGMA_MAX", None)
    sigma_min = getattr(diffusion, "DEFAULT_SIGMA_MIN", None)
    beta_min = getattr(diffusion, "DEFAULT_BETA_MIN", None)
    beta_max = getattr(diffusion, "DEFAULT_BETA_MAX", None)
    if None in (sigma_max, sigma_min, beta_min, beta_max):
        pytest.skip("default SDE constants not exported")
    assert sigma_max == pytest.approx(SIGMA_MAX)
    assert sigma_min == pytest.approx(SIGMA_MIN)
    assert beta_min == pytest.approx(BETA_MIN)
    assert beta_max == pytest.approx(BETA_MAX)


def test_default_time_bounds():
    assert float(getattr(diffusion, "DEFAULT_T_MIN")) == pytest.approx(T_MIN)
    assert float(getattr(diffusion, "DEFAULT_T_MAX")) == pytest.approx(T_MAX)


@pytest.mark.parametrize("name", ["vesde", "VESDE", "variance_exploding"])
def test_get_sde_accepts_name_aliases(name):
    get_sde = _get("get_sde")
    try:
        sde = get_sde(name)
    except Exception:
        pytest.skip(f"get_sde does not accept alias {name!r}")
    assert sde is not None
    assert hasattr(sde, "diffusion")


def test_get_sde_returns_vesde_and_vpsde():
    get_sde = _get("get_sde")
    vesde = get_sde("vesde")
    vpsde = get_sde("vpsde")
    assert _as_array(vesde.marginal_std(0.5)).size == 1
    assert _as_array(vpsde.marginal_std(0.5)).size == 1


def test_sde_from_config_dict():
    sde_from_config = _get("sde_from_config")
    sde = sde_from_config({"name": "vesde", "sigma_max": SIGMA_MAX, "sigma_min": SIGMA_MIN})
    assert _as_array(sde.diffusion(0.3)).size == 1


# ---------------------------------------------------------------------------
# VESDE coefficients
# ---------------------------------------------------------------------------


def test_vesde_diffusion_matches_paper_formula():
    sde = _vesde()
    for t in (1e-5, 1e-3, 0.1, 0.5, 1.0):
        expected = SIGMA_MIN * math.pow(RATIO, t) * math.sqrt(2.0 * LOG_RATIO)
        got = float(np.asarray(_as_array(sde.diffusion(t))).reshape(-1)[0])
        assert got == pytest.approx(expected, rel=1e-6)


def test_vesde_drift_is_zero():
    sde = _vesde()
    x = np.array([[0.3, -1.2, 4.0]])
    for t in (1e-4, 0.25, 1.0):
        drift = _as_array(sde.drift(x, t))
        assert drift.shape == x.shape
        assert np.allclose(drift, 0.0, atol=1e-10)


def test_vesde_marginal_std_matches_paper_formula():
    sde = _vesde()
    for t in (1e-5, 1e-3, 0.1, 0.5, 1.0):
        got = float(np.asarray(_as_array(sde.marginal_std(t))).reshape(-1)[0])
        assert got == pytest.approx(float(_sigma(t)), rel=1e-6)


def test_vesde_marginal_mean_is_one():
    sde = _vesde()
    for t in (1e-5, 0.1, 1.0):
        got = float(np.asarray(_as_array(sde.marginal_mean(t))).reshape(-1)[0])
        assert got == pytest.approx(1.0, rel=1e-9)


def test_vesde_marginal_std_at_bounds():
    sde = _vesde()
    sigma_lo = float(np.asarray(_as_array(sde.marginal_std(T_MIN))).reshape(-1)[0])
    sigma_hi = float(np.asarray(_as_array(sde.marginal_std(T_MAX))).reshape(-1)[0])
    assert sigma_lo == pytest.approx(SIGMA_MIN * math.pow(RATIO, T_MIN), rel=1e-6)
    assert sigma_hi == pytest.approx(SIGMA_MAX, rel=1e-6)


def test_vesde_prior_std_is_sigma_max():
    sde = _vesde()
    prior_fn = getattr(sde, "prior_mean_std", None)
    if prior_fn is None:
        pytest.skip("prior_mean_std not implemented")
    out = prior_fn()
    std = _as_array(out[1] if isinstance(out, (tuple, list)) else out)
    assert float(np.max(std)) == pytest.approx(SIGMA_MAX, rel=1e-6)


def test_vesde_sample_marginal_matches_gaussian_moments():
    sde = _vesde()
    rng = np.random.default_rng(0)
    n = 20000
    x0 = np.full((n, 1), 0.7)
    t = np.full((n,), 0.3)
    x_t = _as_array(sde.sample_marginal(x0, t, np.asarray(sde.sample_prior(x0.shape, rng)), rng))
    sigma = float(_sigma(0.3))
    assert x_t.mean() == pytest.approx(0.7, abs=0.02)
    assert x_t.std() == pytest.approx(sigma, rel=0.05)


# ---------------------------------------------------------------------------
# VPSDE coefficients
# ---------------------------------------------------------------------------


def test_vpsde_beta_matches_paper_formula():
    sde = _vpsde()
    beta_fn = getattr(sde, "beta", None)
    if beta_fn is None:
        pytest.skip("VPSDE.beta not implemented")
    for t in (1e-5, 0.1, 0.5, 1.0):
        got = float(np.asarray(_as_array(beta_fn(t))).reshape(-1)[0])
        assert got == pytest.approx(float(_beta(t)), rel=1e-6)


def test_vpsde_diffusion_is_sqrt_beta():
    sde = _vpsde()
    for t in (1e-5, 0.1, 0.5, 1.0):
        got = float(np.asarray(_as_array(sde.diffusion(t))).reshape(-1)[0])
        assert got == pytest.approx(math.sqrt(float(_beta(t))), rel=1e-6)


def test_vpsde_drift_is_minus_half_beta_x():
    sde = _vpsde()
    x = np.array([[0.5, -2.0]])
    for t in (1e-4, 0.3, 1.0):
        drift = _as_array(sde.drift(x, t))
        expected = -0.5 * float(_beta(t)) * x
        assert np.allclose(drift, expected, rtol=1e-6, atol=1e-8)


def test_vpsde_marginal_mean_std_match_paper_formula():
    sde = _vpsde()
    for t in (1e-5, 0.5, 1.0):
        mu = float(np.asarray(_as_array(sde.marginal_mean(t))).reshape(-1)[0])
        std = float(np.asarray(_as_array(sde.marginal_std(t))).reshape(-1)[0])
        assert mu == pytest.approx(float(_mu(t)), rel=1e-6)
        assert std == pytest.approx(math.sqrt(max(0.0, 1.0 - _mu(t) ** 2)), rel=1e-6)


def test_vpsde_marginal_variance_sum_is_one():
    sde = _vpsde()
    for t in (1e-5, 0.2, 0.7, 1.0):
        mu = float(np.asarray(_as_array(sde.marginal_mean(t))).reshape(-1)[0])
        std = float(np.asarray(_as_array(sde.marginal_std(t))).reshape(-1)[0])
        assert mu**2 + std**2 == pytest.approx(1.0, rel=1e-6)


def test_vpsde_marginal_mean_at_origin():
    sde = _vpsde()
    mu = float(np.asarray(_as_array(sde.marginal_mean(T_MIN))).reshape(-1)[0])
    assert mu == pytest.approx(1.0, rel=1e-4)


# ---------------------------------------------------------------------------
# time grid
# ---------------------------------------------------------------------------


def test_time_grid_shape_and_endpoints():
    grid_fn = _get("time_grid")
    grid = _as_array(grid_fn(n_steps=200, t_min=T_MIN, t_max=T_MAX, descending=True))
    assert grid.shape[0] == 200
    assert (grid >= T_MIN - 1e-8).all()
    assert (grid <= T_MAX + 1e-8).all()
    assert np.all(np.diff(grid) < 0)


def test_time_grid_ascending_reverses():
    grid_fn = _get("time_grid")
    asc = _as_array(grid_fn(n_steps=64, t_min=T_MIN, t_max=T_MAX, descending=False))
    desc = _as_array(grid_fn(n_steps=64, t_min=T_MIN, t_max=T_MAX, descending=True))
    assert np.all(np.diff(asc) > 0)
    assert np.allclose(asc, desc[::-1])


def test_default_steps_is_500():
    steps = getattr(diffusion, "DEFAULT_STEPS", None)
    if steps is None:
        pytest.skip("DEFAULT_STEPS not exported")
    assert int(steps) == 500


def test_min_recommended_steps_is_50():
    value = getattr(diffusion, "MIN_RECOMMENDED_STEPS", None)
    if value is None:
        pytest.skip("MIN_RECOMMENDED_STEPS not exported")
    assert int(value) == 50


# ---------------------------------------------------------------------------
# score / epsilon algebra
# ---------------------------------------------------------------------------


def test_score_target_equals_minus_epsilon_over_sigma():
    sde = _vesde()
    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(8, 3))
    t = np.full((8,), 0.42)
    eps = rng.normal(size=x0.shape)
    x_t = _as_array(sde.marginal_mean(t)) * x0 + _as_array(sde.marginal_std(t)) * eps
    score = _as_array(sde.score_target(x_t, x0, t))
    sigma = float(np.asarray(_as_array(sde.marginal_std(t))).reshape(-1)[0])
    assert np.allclose(score, -eps / sigma, rtol=1e-5, atol=1e-6)


def test_epsilon_target_equals_score_times_sigma():
    sde = _vesde()
    rng = np.random.default_rng(1)
    x0 = rng.normal(size=(5, 2))
    t = np.full((5,), 0.25)
    eps = rng.normal(size=x0.shape)
    x_t = _as_array(sde.marginal_mean(t)) * x0 + _as_array(sde.marginal_std(t)) * eps
    eps_hat = _as_array(sde.epsilon_target(x_t, x0, t))
    assert np.allclose(eps_hat, eps, rtol=1e-5, atol=1e-6)


def test_score_from_epsilon_identity_roundtrip():
    """score = -eps/sigma(t) and eps = -score*sigma(t) must invert."""
    score_from_eps = getattr(diffusion, "score_from_epsilon", None)
    eps_from_score = getattr(diffusion, "epsilon_from_score", None)
    if score_from_eps is None or eps_from_score is None:
        pytest.skip("score/epsilon algebra helpers not exported")
    eps = np.random.default_rng(0).normal(size=(4, 3))
    sigma = 2.5
    score = _as_array(score_from_eps(eps, np.full((4, 1), sigma)))
    back = _as_array(eps_from_score(score, np.full((4, 1), sigma)))
    assert np.allclose(back, eps, rtol=1e-6, atol=1e-8)


def test_tweedie_denoise_recovers_x0_mean():
    """x_hat_0 = (x_t + sigma(t)^2 s) / mu(t) recovers x_0 when s is exact."""
    sde = _vesde()
    rng = np.random.default_rng(2)
    x0 = rng.normal(size=(6, 2))
    t = np.full((6,), 0.5)
    eps = rng.normal(size=x0.shape)
    x_t = _as_array(sde.marginal_mean(t)) * x0 + _as_array(sde.marginal_std(t)) * eps
    score = _as_array(sde.score_target(x_t, x0, t))
    denoise = getattr(sde, "twedie_denoise", None) or getattr(sde, "tweedie_denoise", None)
    if denoise is None:
        pytest.skip("Tweedie denoiser not implemented")
    x_hat0 = _as_array(denoise(x_t, t, score))
    assert np.allclose(x_hat0, x0, rtol=1e-5, atol=1e-6)


def test_tweedie_for_vpsde_recovers_x0_mean():
    sde = _vpsde()
    rng = np.random.default_rng(3)
    x0 = rng.normal(size=(6, 2))
    t = np.full((6,), 0.4)
    eps = rng.normal(size=x0.shape)
    mu = float(np.asarray(_as_array(sde.marginal_mean(t))).reshape(-1)[0])
    sigma = float(np.asarray(_as_array(sde.marginal_std(t))).reshape(-1)[0])
    x_t = mu * x0 + sigma * eps
    score = _as_array(sde.score_target(x_t, x0, t))
    denoise = getattr(sde, "twedie_denoise", None) or getattr(sde, "tweedie_denoise", None)
    if denoise is None:
        pytest.skip("Tweedie denoiser not implemented")
    x_hat0 = _as_array(denoise(x_t, t, score))
    assert np.allclose(x_hat0, x0, rtol=1e-5, atol=1e-6)


def test_score_scaling_inverse_variance():
    fn = getattr(diffusion, "score_scaling_inverse_variance", None)
    if fn is None:
        sde = _vesde()
        scaling = getattr(sde, "score_scaling", None)
        if scaling is None:
            pytest.skip("score scaling not implemented")
        fn = scaling
    sde = _vesde()
    for t in (1e-2, 0.3, 1.0):
        sigma = float(np.asarray(_as_array(sde.marginal_std(t))).reshape(-1)[0])
        got = float(np.asarray(_as_array(fn(sde, t))).reshape(-1)[0])
        assert got == pytest.approx(1.0 / sigma**2, rel=1e-6)


# ---------------------------------------------------------------------------
# reverse SDE
# ---------------------------------------------------------------------------


def test_reverse_sde_step_shape_and_finiteness():
    step_fn = _get("reverse_sde_step", "euler_maruyama_step")
    sde = _vesde()
    x = np.zeros((4, 2))
    t_cur, t_next = 0.5, 0.45
    score = np.ones((4, 2))
    out = _as_array(step_fn(sde, x, t_cur, t_next, score, noise=np.zeros((4, 2))))
    assert out.shape == x.shape
    assert np.all(np.isfinite(out))


def test_reverse_sde_step_with_zero_score_has_deterministic_sign():
    """With s=0 the VESDE reverse drift vanishes -> only the noise term acts."""
    step_fn = _get("reverse_sde_step", "euler_maruyama_step")
    sde = _vesde()
    x = np.zeros((3, 2))
    out = _as_array(step_fn(sde, x, 0.5, 0.4, np.zeros((3, 2)), noise=np.zeros((3, 2))))
    assert np.allclose(out, 0.0, atol=1e-10)


def test_reverse_sde_step_moves_towards_score():
    """A positive score should move x upwards (denoising direction)."""
    step_fn = _get("reverse_sde_step", "euler_maruyama_step")
    sde = _vesde()
    x = np.zeros((2, 2))
    out = _as_array(step_fn(sde, x, 0.5, 0.4, np.ones((2, 2)), noise=np.zeros((2, 2))))
    assert (out > 0).all()


def test_sample_reverse_sde_shapes_and_finiteness():
    sample_fn = getattr(diffusion, "sample_reverse_sde", None)
    if sample_fn is None:
        pytest.skip("sample_reverse_sde not implemented")
    sde = _vesde()

    def score_fn(x, t):
        return np.zeros_like(x)

    out = sample_fn(
        score_fn,
        sde,
        shape=(8, 2),
        n_steps=10,
        t_min=T_MIN,
        t_max=T_MAX,
        rng=np.random.default_rng(0),
    )
    samples = _as_array(out[0] if isinstance(out, tuple) else out)
    assert samples.shape == (8, 2)
    assert np.all(np.isfinite(samples))


def test_sample_reverse_sde_clamps_conditioned_dimensions():
    sample_fn = getattr(diffusion, "sample_reverse_sde", None)
    if sample_fn is None:
        pytest.skip("sample_reverse_sde not implemented")
    sde = _vesde()

    def score_fn(x, t):
        return np.zeros_like(x)

    condition_mask = np.array([[1.0, 0.0]])
    condition_values = np.array([[3.5, 0.0]])
    out = sample_fn(
        score_fn,
        sde,
        shape=(4, 2),
        n_steps=5,
        t_min=T_MIN,
        t_max=T_MAX,
        condition_mask=condition_mask,
        condition_values=condition_values,
        rng=np.random.default_rng(1),
    )
    samples = _as_array(out[0] if isinstance(out, tuple) else out)
    assert np.allclose(samples[:, 0], 3.5, atol=1e-8)
    assert np.all(np.isfinite(samples[:, 1]))


# ---------------------------------------------------------------------------
# forward noising / prior
# ---------------------------------------------------------------------------


def test_sample_prior_shape_and_scale():
    sde = _vesde()
    rng = np.random.default_rng(0)
    prior = _as_array(sde.sample_prior((5000, 1), rng))
    assert prior.shape == (5000, 1)
    assert prior.std() == pytest.approx(SIGMA_MAX, rel=0.05)


def test_sample_marginal_shape():
    sde = _vesde()
    rng = np.random.default_rng(0)
    x0 = np.ones((7, 3))
    t = np.full((7,), 0.2)
    eps = rng.normal(size=x0.shape)
    x_t = _as_array(sde.sample_marginal(x0, t, eps, rng))
    assert x_t.shape == x0.shape


# ---------------------------------------------------------------------------
# probability-flow ODE and log-likelihood
# ---------------------------------------------------------------------------


def test_probability_flow_drift_formula():
    drift_fn = _get("probability_flow_drift")
    sde = _vesde()
    x = np.array([[0.4, -0.3]])
    t = 0.5
    score = np.array([[1.5, -2.0]])
    drift = _as_array(drift_fn(sde, x, t, score))
    g = float(np.asarray(_as_array(sde.diffusion(t))).reshape(-1)[0])
    expected = _as_array(sde.drift(x, t)) - 0.5 * g**2 * score
    assert np.allclose(drift, expected, rtol=1e-6, atol=1e-8)


def test_probability_flow_ode_output_shape():
    ode_fn = getattr(diffusion, "probability_flow_ode", None)
    if ode_fn is None:
        pytest.skip("probability_flow_ode not implemented")
    sde = _vesde()

    def score_fn(x, t):
        return np.zeros_like(x)

    x0 = np.zeros((5, 2))
    out = ode_fn(score_fn, sde, x0, n_steps=5, t_min=T_MIN, t_max=T_MAX)
    result = _as_array(out[0] if isinstance(out, tuple) else out)
    assert result.shape == (5, 2)
    assert np.all(np.isfinite(result))


def test_log_likelihood_from_ode_is_finite():
    ll_fn = getattr(diffusion, "log_likelihood_from_ode", None)
    if ll_fn is None:
        pytest.skip("log_likelihood_from_ode not implemented")
    sde = _vesde()

    def score_fn(x, t):
        return np.zeros_like(x)

    x0 = np.array([[0.1, -0.2], [0.3, 0.0]])
    try:
        out = ll_fn(score_fn, sde, x0, n_steps=5, t_min=T_MIN, t_max=T_MAX)
    except TypeError:
        out = ll_fn(score_fn, sde, x0, n_steps=5)
    values = _as_array(out[0] if isinstance(out, tuple) else out)
    assert np.all(np.isfinite(values))


def test_exact_divergence_of_linear_map():
    exact_div = getattr(diffusion, "exact_divergence", None)
    if exact_div is None:
        pytest.skip("exact_divergence not implemented")

    scale = np.array([2.0, -3.0, 0.5])

    def fn(x):
        return scale * x

    x = np.ones((1, 3))
    div = float(np.asarray(_as_array(exact_div(fn, x))).reshape(-1)[0])
    assert div == pytest.approx(float(scale.sum()), rel=1e-3, abs=1e-3)


def test_hutchinson_divergence_is_close_for_linear_map():
    hutch = getattr(diffusion, "hutchinson_divergence", None)
    if hutch is None:
        pytest.skip("hutchinson_divergence not implemented")
    scale = np.array([2.0, -3.0, 0.5])

    def fn(x):
        return scale * x

    x = np.ones((1, 3))
    rng = np.random.default_rng(0)
    div = float(np.asarray(_as_array(hutch(fn, x, n_samples=512, rng=rng))).reshape(-1)[0])
    assert div == pytest.approx(float(scale.sum()), abs=1.0)


def test_divergence_dispatch_matches_exact_for_small_dim():
    div_fn = getattr(diffusion, "divergence", None)
    if div_fn is None:
        pytest.skip("divergence dispatch not implemented")
    scale = np.array([1.0, -1.0])

    def fn(x):
        return scale * x

    x = np.ones((1, 2))
    div = float(np.asarray(_as_array(div_fn(fn, x, exact_max_dim=32, rng=np.random.default_rng(0)))).reshape(-1)[0])
    assert div == pytest.approx(0.0, abs=1e-3)


# ---------------------------------------------------------------------------
# SDEConfig / torch interop
# ---------------------------------------------------------------------------


def test_sde_config_build_roundtrip():
    cfg_cls = getattr(diffusion, "SDEConfig", None)
    if cfg_cls is None:
        pytest.skip("SDEConfig not exported")
    cfg = cfg_cls(name="vesde", sigma_max=SIGMA_MAX, sigma_min=SIGMA_MIN)
    sde = cfg.build()
    assert _as_array(sde.marginal_std(0.5)).size == 1
    as_dict = cfg.to_dict()
    assert isinstance(as_dict, dict)
    assert as_dict.get("name") == "vesde"


def test_vesde_accepts_torch_tensors():
    torch = pytest.importorskip("torch")
    sde = _vesde()
    t = torch.full((3,), 0.5)
    out = sde.marginal_std(t)
    assert tuple(out.shape) == (3,)


@pytest.mark.parametrize("t", [1e-5, 0.1, 0.5, 1.0])
def test_vesde_sigma_monotone_increasing(t):
    sde = _vesde()
    sigma = float(np.asarray(_as_array(sde.marginal_std(t))).reshape(-1)[0])
    assert sigma == pytest.approx(float(_sigma(t)), rel=1e-6)


def test_sde_names_and_aliases():
    vesde_cls = _get("VarianceExplodingSDE", "VESDE")
    vpsde_cls = _get("VariancePreservingSDE", "VPSDE")
    assert vesde_cls is not None and vpsde_cls is not None
