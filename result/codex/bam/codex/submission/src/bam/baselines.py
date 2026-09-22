"""Baseline algorithms.

* ``ADVI``  -- ELBO maximization with Adam (Algorithm 2 of Appendix E.1).
* ``Score`` -- the same stochastic-gradient scheme, but with the score-based
  divergence of eq. (2) in place of the negative ELBO.
* ``Fisher``-- the same scheme with the Fisher divergence in place of the ELBO.
* ``GSM``   -- Gaussian score matching (Algorithm 3 of Appendix E.1), the
  special case of BaM with ``B = 1`` and ``lambda -> inf``.

All methods share the same Gaussian variational family (full covariance) and
the same initialization, and all of them spend exactly ``B`` target-score (or
log-density) gradient evaluations per iteration, so the x-axis of the paper's
figures ("number of gradient evaluations") is ``B * n_iters`` for every method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np

from .adam import Adam
from .linalg import symmetrize
from .targets import Target, gaussian_cholesky, sample_gaussian


# ----------------------------------------------------------------------------
# parametrization helpers: theta = (mu, L) with Sigma = L L^T, L lower triangular
# ----------------------------------------------------------------------------


def _tril_indices(D: int):
    return np.tril_indices(D)


def unpack_params(params: np.ndarray, D: int):
    """Split a flat parameter vector into ``(mu, L)`` with ``Sigma = L L^T``.

    The strict lower-triangular part of ``L`` is unconstrained and the diagonal
    is parametrized through ``exp`` so that ``L`` is always invertible.
    """
    mu = params[:D]
    raw = params[D:].reshape(D, D)
    L = np.tril(raw, -1) + np.diag(np.exp(np.diag(raw)))
    return mu, L


def pack_params(mu: np.ndarray, L: np.ndarray) -> np.ndarray:
    D = mu.shape[0]
    raw = np.tril(L, -1) + np.diag(np.log(np.diag(L)))
    return np.concatenate([mu, raw.ravel()])


def n_params(D: int) -> int:
    return D + D * D


# ----------------------------------------------------------------------------
# differentiable objectives (ELBO / score / Fisher) for q = N(mu, Sigma)
# ----------------------------------------------------------------------------


def make_objective(target: Target, loss: str):
    """Return ``loss_fn(params, eps) -> scalar``, differentiable with JAX.

    ``eps`` are the standardized samples used by the reparameterization trick,
    ``z = mu + L eps``.
    """
    D = int(target.dim)

    def _mu_L(params):
        mu = params[:D]
        raw = params[D:].reshape(D, D)
        L = jnp.tril(raw, -1) + jnp.diag(jnp.exp(jnp.diag(raw)))
        return mu, L

    def elbo(params, eps):
        mu, L = _mu_L(params)
        Z = mu[None, :] + eps @ L.T
        Sigma = L @ L.T
        Sigma_inv = jnp.linalg.inv(Sigma)
        diff = Z - mu
        log_q = -0.5 * (D * jnp.log(2.0 * jnp.pi) + 2.0 * jnp.sum(jnp.log(jnp.diag(L)))) \
                - 0.5 * jnp.einsum("bi,ij,bj->b", diff, Sigma_inv, diff)
        log_p = target.log_density_jax(Z)
        return -jnp.mean(log_p - log_q)

    def score_loss(params, eps):
        mu, L = _mu_L(params)
        Z = mu[None, :] + eps @ L.T
        Sigma = L @ L.T
        s_q = -(Z - mu) @ jnp.linalg.inv(Sigma).T
        s_p = target.score_jax(Z)
        d = s_q - s_p
        return jnp.mean(jnp.sum((d @ L.T) ** 2, axis=1))  # ||d||^2_{L L^T}

    def fisher_loss(params, eps):
        mu, L = _mu_L(params)
        Z = mu[None, :] + eps @ L.T
        Sigma = L @ L.T
        s_q = -(Z - mu) @ jnp.linalg.inv(Sigma).T
        s_p = target.score_jax(Z)
        return jnp.mean(jnp.sum((s_q - s_p) ** 2, axis=1))

    fn = {"elbo": elbo, "score": score_loss, "fisher": fisher_loss}[loss]

    @jax.jit
    def value_and_grad(params, eps):
        return jax.value_and_grad(fn)(params, eps)

    return value_and_grad


# ----------------------------------------------------------------------------
# ADVI / Score / Fisher
# ----------------------------------------------------------------------------


@dataclass
class GradientVIConfig:
    batch_size: int = 2
    learning_rate: float = 0.01
    loss: str = "elbo"  # 'elbo' | 'score' | 'fisher'
    n_iters: int = 1000
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999

    @property
    def name(self) -> str:
        return {"elbo": "ADVI", "score": "Score", "fisher": "Fisher"}[self.loss]


class GradientVI:
    """Stochastic-gradient variational inference (Algorithm 2 of the paper)."""

    def __init__(self, target: Target, config: GradientVIConfig):
        self.target = target
        self.config = config
        self._obj = make_objective(target, config.loss)

    def run(self, key, n_iters: Optional[int] = None, mu0: Optional[np.ndarray] = None,
            Sigma0: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        cfg = self.config
        D = int(self.target.dim)
        B = int(cfg.batch_size)
        T = int(cfg.n_iters if n_iters is None else n_iters)
        if mu0 is None:
            mu0 = np.zeros(D)
        if Sigma0 is None:
            Sigma0 = np.eye(D)
        params = pack_params(np.asarray(mu0, dtype=np.float64),
                             gaussian_cholesky(Sigma0))
        opt = Adam(params, cfg.learning_rate, cfg.adam_beta1, cfg.adam_beta2)

        mus = np.zeros((T + 1, D))
        Sigmas = np.zeros((T + 1, D, D))
        grad_evals = np.zeros(T + 1, dtype=np.int64)
        losses = np.zeros(T + 1)
        mu, L = unpack_params(params, D)
        mus[0], Sigmas[0] = mu, symmetrize(L @ L.T)

        keys = jax.random.split(key, T)
        for t in range(T):
            eps = jax.random.normal(keys[t], (B, D))
            loss, grad = self._obj(jnp.asarray(params), eps)
            params = opt.update(params, np.asarray(grad))
            mu, L = unpack_params(params, D)
            mus[t + 1], Sigmas[t + 1] = mu, symmetrize(L @ L.T)
            grad_evals[t + 1] = grad_evals[t] + B
            losses[t + 1] = float(loss)
        return {"mu": mus, "Sigma": Sigmas, "grad_evals": grad_evals, "loss": losses,
                "n_iters": T, "batch_size": B, "name": cfg.name}


# ----------------------------------------------------------------------------
# GSM (Modi et al., 2023) -- Algorithm 3
# ----------------------------------------------------------------------------


def gsm_step(target: Target, mu: np.ndarray, Sigma: np.ndarray, Z: np.ndarray, S: np.ndarray,
             ensure_pd: bool = True):
    """One GSM update from a batch of samples ``Z`` and target scores ``S``."""
    B, D = Z.shape
    delta_mu = np.zeros((B, D))
    delta_Sigma = np.zeros((B, D, D))
    for b in range(B):
        s_b = S[b]
        diff = mu - Z[b]
        c = float(s_b @ Sigma @ s_b + (diff @ s_b) ** 2)
        rho = 0.5 * (-1.0 + np.sqrt(1.0 + 4.0 * c))
        eps_b = Sigma @ s_b - diff
        denom = 1.0 + rho + float(diff @ s_b)
        M = np.eye(D) - np.outer(diff, s_b) / denom
        d_mu = M @ eps_b / (1.0 + rho)
        mu_tilde = mu + d_mu
        delta_mu[b] = d_mu
        delta_Sigma[b] = np.outer(diff, diff) - np.outer(mu_tilde - Z[b], mu_tilde - Z[b])
    mu_new = mu + delta_mu.mean(axis=0)
    Sigma_new = symmetrize(Sigma + delta_Sigma.mean(axis=0))
    if ensure_pd:
        w, Q = np.linalg.eigh(Sigma_new)
        w = np.maximum(w, 1e-12)
        Sigma_new = symmetrize((Q * w) @ Q.T)
    return mu_new, Sigma_new


class GSM:
    """Gaussian score matching (Algorithm 3)."""

    def __init__(self, target: Target, batch_size: int = 2, ensure_pd: bool = True):
        self.target = target
        self.batch_size = int(batch_size)
        self.ensure_pd = ensure_pd

    def run(self, key, n_iters: int, mu0: Optional[np.ndarray] = None,
            Sigma0: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        D = int(self.target.dim)
        mu = np.zeros(D) if mu0 is None else np.asarray(mu0, dtype=np.float64).ravel().copy()
        Sigma = np.eye(D) if Sigma0 is None else symmetrize(np.asarray(Sigma0, dtype=np.float64)).copy()
        T = int(n_iters)
        mus = np.zeros((T + 1, D))
        Sigmas = np.zeros((T + 1, D, D))
        grad_evals = np.zeros(T + 1, dtype=np.int64)
        mus[0], Sigmas[0] = mu, Sigma
        keys = jax.random.split(key, T)
        for t in range(T):
            Z = sample_gaussian(keys[t], mu, gaussian_cholesky(Sigma), self.batch_size)
            S = np.asarray(self.target.score(Z))
            mu, Sigma = gsm_step(self.target, mu, Sigma, Z, S, self.ensure_pd)
            mus[t + 1], Sigmas[t + 1] = mu, Sigma
            grad_evals[t + 1] = grad_evals[t] + self.batch_size
        return {"mu": mus, "Sigma": Sigmas, "grad_evals": grad_evals, "n_iters": T,
                "batch_size": self.batch_size, "name": "GSM"}


__all__ = [
    "GradientVI",
    "GradientVIConfig",
    "GSM",
    "gsm_step",
    "make_objective",
    "pack_params",
    "unpack_params",
]
