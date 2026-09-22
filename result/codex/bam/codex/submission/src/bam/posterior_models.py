"""The three Bayesian models of Section 5.2, transcribed from their Stan code.

The paper uses targets from posteriordb (Magnusson et al., 2022) and obtains the
gradients of the log density with BridgeStan.  Here the same three models are
transcribed into JAX straight from their Stan sources
(https://github.com/stan-dev/posteriordb/tree/master/posterior_database/models/stan),
so that the log density (and, by automatic differentiation, its gradient) can be
evaluated without compiling Stan:

* ``arK``                    -- autoregressive time series, D = 7  (nearly Gaussian)
* ``gp_pois_regr``           -- GP Poisson regression, D = 13      (non-Gaussian)
* ``eight_schools_centered`` -- 8 schools hierarchical model, D = 10 (non-Gaussian)

The parameter *order* follows the Stan declaration order, which is also the
column order of the reference draws shipped in ``data/posteriordb``.

Constraint handling: parameters such as ``sigma > 0`` or ``rho > 0`` make the
posterior density zero outside an open set of R^D.  We therefore return
``-inf`` for infeasible points and report the score of the *smooth* part of the
log density, zeroed out outside the support (in this way the score never
contains the NaNs that a naive ``grad(-inf)`` would produce).  For all three
posteriors the posterior mass is far from the boundary, so samples drawn by the
variational algorithms are feasible in practice.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import jax.numpy as jnp
import numpy as np

from .targets import PosteriorTarget

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "data", "posteriordb")

_LOG2PI = float(np.log(2.0 * np.pi))


def _normal_lpdf(x, mu, sigma):
    # ``log |sigma|`` is the smooth extension of the density outside ``sigma > 0``
    return -0.5 * (_LOG2PI + 2.0 * jnp.log(jnp.abs(sigma)) + ((x - mu) / sigma) ** 2)


def _sum_normal_lpdf(x, mu, sigma):
    """Vectorized ``normal_lpdf`` summed over the batch (all arguments batched)."""
    return -0.5 * jnp.sum(_LOG2PI + 2.0 * jnp.log(jnp.abs(sigma)) + ((x - mu) / sigma) ** 2)


def _half_cauchy_lpdf(x, scale):
    """``cauchy(0, scale)`` for an ``x > 0`` parameter (Stan's ``<lower=0>``)."""
    return jnp.log(2.0) - jnp.log(jnp.pi) - jnp.log(scale) - jnp.log1p((x / scale) ** 2)


# ----------------------------------------------------------------------------
# arK
# ----------------------------------------------------------------------------


def make_ark_target(data: Dict, ref_mean=None, ref_sd=None,
                    parameterization: str = "unconstrained") -> PosteriorTarget:
    K = int(data["K"])
    y = jnp.asarray(np.asarray(data["y"], dtype=np.float64))
    T = int(data["T"])
    # y[t-1] (0-based) is the observation at time t; the likelihood runs over
    # t = K+1..T and regresses y_t on y_{t-1}, ..., y_{t-K}
    lags = jnp.stack([y[K - k - 1:T - k - 1] for k in range(K)], axis=1)  # (T-K, K)
    target_y = y[K:T]                                                    # (T-K,)

    def log_joint(z):
        alpha = z[0]
        beta = z[1:1 + K]
        sigma = z[1 + K]
        mu_t = alpha + lags @ beta
        lp = _normal_lpdf(alpha, 0.0, 10.0)
        lp = lp + _sum_normal_lpdf(beta, 0.0, 10.0)
        lp = lp + _half_cauchy_lpdf(sigma, 2.5)
        lp = lp + _sum_normal_lpdf(target_y, mu_t, sigma)
        return lp

    feasible = lambda z: z[1 + K] > 0
    return PosteriorTarget(dim=1 + K + 1, log_joint=log_joint, name="arK", feasible=feasible,
                           ref_mean=ref_mean, ref_sd=ref_sd,
                           param_names=["alpha"] + [f"beta[{k}]" for k in range(1, K + 1)] + ["sigma"],
                           positive_dims=[1 + K], parameterization=parameterization)


# ----------------------------------------------------------------------------
# gp_pois_regr
# ----------------------------------------------------------------------------


def exp_quad_cov(x: jnp.ndarray, alpha, rho) -> jnp.ndarray:
    """Stan's ``gp_exp_quad_cov(x, alpha, rho)``: ``alpha^2 exp(-d^2 / (2 rho^2))``."""
    d2 = (x[:, None] - x[None, :]) ** 2
    return alpha**2 * jnp.exp(-d2 / (2.0 * rho**2))


def make_gp_pois_regr_target(data: Dict, ref_mean=None, ref_sd=None,
                             parameterization: str = "unconstrained",
                             variable: str = "f") -> PosteriorTarget:
    """GP Poisson regression.

    ``variable='f'`` (default) uses the *transformed* parameter ``f = L(rho, alpha) f_tilde``
    of Stan's ``transformed parameters`` block as the variable of the inference
    problem, i.e. the posterior

        p(rho, alpha, f | k) ∝ p(rho) p(alpha) N(L^{-1} f; 0, I) / det L
                                  * prod_n Poisson(k_n ; exp(f_n)),

    which is the pushforward of the true posterior through ``f`` and is the
    parameterization listed by posteriordb for this model (``rho: 1, alpha: 1,
    f: 11``), the one in which the reference draws are stored, and the one in
    which the paper's standard-Gaussian initialization is meaningful (in the
    ``f_tilde`` parameterization, samples from ``N(0, I)`` map to
    ``f = L f_tilde`` with ``|f| ~ 10``, where the Poisson likelihood is
    astronomically small).

    ``variable='f_tilde'`` gives the density over the model's actual parameters.
    """
    x = jnp.asarray(np.asarray(data["x"], dtype=np.float64))
    k_obs = jnp.asarray(np.asarray(data["k"], dtype=np.float64))
    N = int(data["N"])

    def _prior_and_chol(rho, alpha):
        cov = exp_quad_cov(x, alpha, rho) + 1e-10 * jnp.eye(N)
        L = jnp.linalg.cholesky(cov)
        lp = (25.0 - 1.0) * jnp.log(jnp.abs(rho)) - 4.0 * rho  # gamma(25, 4) up to a constant
        lp = lp + _normal_lpdf(alpha, 0.0, 2.0)
        return lp, L

    if variable == "f":
        def log_joint(z):
            rho, alpha = z[0], z[1]
            f = z[2:]
            lp, L = _prior_and_chol(rho, alpha)
            f_tilde = jnp.linalg.solve(L, f)
            lp = lp - 0.5 * jnp.sum(f_tilde**2)
            lp = lp - jnp.sum(jnp.log(jnp.diag(L)))   # change of variables f = L f_tilde
            lp = lp + jnp.sum(k_obs * f - jnp.exp(f))  # poisson_log up to log(k!)
            return lp
    elif variable == "f_tilde":
        def log_joint(z):
            rho, alpha = z[0], z[1]
            f_tilde = z[2:]
            lp, L = _prior_and_chol(rho, alpha)
            f = L @ f_tilde
            lp = lp - 0.5 * jnp.sum(f_tilde**2)
            lp = lp + jnp.sum(k_obs * f - jnp.exp(f))
            return lp
    else:
        raise ValueError(variable)

    feasible = lambda z: jnp.all(jnp.array([z[0] > 0, z[1] > 0]))
    names = ["rho", "alpha"] + ([f"f[{i}]" for i in range(1, N + 1)] if variable == "f"
                                else [f"f_tilde[{i}]" for i in range(1, N + 1)])
    return PosteriorTarget(dim=2 + N, log_joint=log_joint, name="gp_pois_regr", feasible=feasible,
                           ref_mean=ref_mean, ref_sd=ref_sd, param_names=names,
                           positive_dims=[0, 1], parameterization=parameterization)


# ----------------------------------------------------------------------------
# eight_schools_centered
# ----------------------------------------------------------------------------


def make_eight_schools_target(data: Dict, ref_mean=None, ref_sd=None,
                              parameterization: str = "unconstrained") -> PosteriorTarget:
    y = jnp.asarray(np.asarray(data["y"], dtype=np.float64))
    sigma = jnp.asarray(np.asarray(data["sigma"], dtype=np.float64))
    J = int(data["J"])

    def log_joint(z):
        theta = z[:J]
        mu = z[J]
        tau = z[J + 1]
        lp = _half_cauchy_lpdf(tau, 5.0)
        lp = lp + _sum_normal_lpdf(theta, mu, tau)
        lp = lp + _sum_normal_lpdf(y, theta, sigma)
        lp = lp + _normal_lpdf(mu, 0.0, 5.0)
        return lp

    feasible = lambda z: z[J + 1] > 0
    return PosteriorTarget(dim=J + 2, log_joint=log_joint, name="eight_schools_centered", feasible=feasible,
                           ref_mean=ref_mean, ref_sd=ref_sd,
                           param_names=[f"theta[{j}]" for j in range(1, J + 1)] + ["mu", "tau"],
                           positive_dims=[J + 1], parameterization=parameterization)


BUILDERS = {
    "arK": make_ark_target,
    "gp_pois_regr": make_gp_pois_regr_target,
    "eight_schools_centered": make_eight_schools_target,
}


def load_posteriordb_target(name: str, data_dir: Optional[str] = None,
                            parameterization: str = "unconstrained",
                            variable: Optional[str] = None) -> PosteriorTarget:
    """Load one of the three Section 5.2 targets with its HMC reference summaries.

    ``parameterization='unconstrained'`` (the default) is the space Stan and
    BridgeStan work in, i.e. it includes the log-Jacobian of the positivity
    transforms.  ``'constrained'`` gives the density over the model parameters
    themselves.

    ``variable`` selects, for ``gp_pois_regr``, whether the GP values enter as
    the transformed parameter ``f`` (default, as listed by posteriordb) or as the
    model parameter ``f_tilde``; see :func:`make_gp_pois_regr_target`.
    """
    data_dir = DATA_DIR if data_dir is None else data_dir
    with open(os.path.join(data_dir, f"{name}_data.json")) as fh:
        data = json.load(fh)
    suffix = ""
    if name == "gp_pois_regr" and (variable or "f") == "f_tilde":
        suffix = "_ftilde"
    with open(os.path.join(data_dir, f"{name}{suffix}_ref_stats.json")) as fh:
        ref = json.load(fh)
    kwargs = {"parameterization": parameterization}
    if name == "gp_pois_regr":
        kwargs["variable"] = variable or "f"
    return BUILDERS[name](data, ref_mean=np.asarray(ref["mean"]), ref_sd=np.asarray(ref["sd"]), **kwargs)


def reference_draws_in_target_space(name: str, data_dir: Optional[str] = None,
                                    parameterization: str = "unconstrained",
                                    variable: Optional[str] = None) -> np.ndarray:
    """Reference HMC draws mapped into the target's parameterization."""
    suffix = ""
    if name == "gp_pois_regr" and (variable or "f") == "f_tilde":
        suffix = "_ftilde"
    data_dir = DATA_DIR if data_dir is None else data_dir
    with np.load(os.path.join(data_dir, f"{name}{suffix}_draws.npz")) as f:
        draws = np.asarray(f["draws"], dtype=np.float64)
    if parameterization == "unconstrained":
        target = load_posteriordb_target(name, data_dir, parameterization="constrained", variable=variable)
        out = draws.copy()
        if target.positive_dims:
            out[:, list(target.positive_dims)] = np.log(out[:, list(target.positive_dims)])
        return out
    return draws


def load_reference_draws(name: str, data_dir: Optional[str] = None) -> np.ndarray:
    """Reference HMC draws (``n_draws x D``) for one of the Section 5.2 targets."""
    data_dir = DATA_DIR if data_dir is None else data_dir
    with np.load(os.path.join(data_dir, f"{name}_draws.npz")) as f:
        return np.asarray(f["draws"], dtype=np.float64)


__all__ = [
    "load_posteriordb_target",
    "load_reference_draws",
    "reference_draws_in_target_space",
    "make_ark_target",
    "make_gp_pois_regr_target",
    "make_eight_schools_target",
    "exp_quad_cov",
]
