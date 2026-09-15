"""Baseline variational-inference algorithms compared against Batch-and-Match.

This module implements the four baselines from the paper:

* ADVI (full-covariance Gaussian variational family optimized with ADAM),
  using the reparameterization (pathwise) gradient so that only the target
  score ``grad_z log p(z)`` is required -- this is essential for black-box
  targets such as BridgeStan posteriors that are not JAX-traceable.
* The score-based baseline: the empirical score divergence is minimized with
  ADAM instead of the negative ELBO.
* The Fisher baseline: the empirical Fisher divergence is minimized with ADAM.
* GSM (Gaussian score matching), the per-sample closed-form update described
  in Algorithm 3 of the paper.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

__all__ = [
    "AdamState",
    "GradientStepResult",
    "GsmStepResult",
    "adam_init",
    "adam_step",
    "advi_step",
    "score_step",
    "fisher_step",
    "gsm_update",
    "gsm_step",
    "run_advi",
    "run_score",
    "run_fisher",
    "run_gsm",
]


# ---------------------------------------------------------------------------
# ADAM optimizer for pytree parameters
# ---------------------------------------------------------------------------
class AdamState(NamedTuple):
    """State of the ADAM optimizer.

    ``m`` and ``v`` are pytrees with the same structure as the parameters.
    ``t`` is the iteration counter.
    """

    m: Any
    v: Any
    t: int


def adam_init(params: Any) -> AdamState:
    """Initialize ADAM state for a parameter pytree."""
    m = tree_map(lambda x: jnp.zeros_like(x), params)
    v = tree_map(lambda x: jnp.zeros_like(x), params)
    return AdamState(m=m, v=v, t=0)


def adam_step(
    params: Any,
    grads: Any,
    state: AdamState,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[Any, AdamState]:
    """Perform one ADAM update.

    Parameters
    ----------
    params:
        Current parameter pytree.
    grads:
        Gradient pytree with the same structure as ``params``.
    state:
        Current ADAM state.
    learning_rate:
        Step size.
    beta1, beta2, eps:
        ADAM hyperparameters.

    Returns
    -------
    (new_params, new_state)
    """
    t = state.t + 1
    m = tree_map(
        lambda mm, gg: beta1 * mm + (1.0 - beta1) * gg, state.m, grads
    )
    v = tree_map(
        lambda vv, gg: beta2 * vv + (1.0 - beta2) * (gg * gg), state.v, grads
    )
    m_hat = tree_map(lambda x: x / (1.0 - beta1**t), m)
    v_hat = tree_map(lambda x: x / (1.0 - beta2**t), v)
    new_params = tree_map(
        lambda p, mh, vh: p - learning_rate * mh / (jnp.sqrt(vh) + eps),
        params,
        m_hat,
        v_hat,
    )
    return new_params, AdamState(m=m, v=v, t=t)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pack_params(mu: jnp.ndarray, L: jnp.ndarray) -> dict[str, jnp.ndarray]:
    return {"mu": mu, "L": L}


def _sample_eps(key: jax.Array, batch_size: int, dim: int) -> jnp.ndarray:
    return jax.random.normal(key, (batch_size, dim))


def _sigma_from_L(L: jnp.ndarray, jitter: float = 1e-10) -> jnp.ndarray:
    """Reconstruct covariance from its lower-triangular factor."""
    L = jnp.tril(L)
    D = L.shape[0]
    return L @ L.T + jitter * jnp.eye(D)


def _grad_log_q(
    z: jnp.ndarray, mu: jnp.ndarray, Sigma: jnp.ndarray
) -> jnp.ndarray:
    """Vectorized ``grad_z log q(z) = -Sigma^{-1}(z - mu)`` for Gaussian q."""
    return -jnp.linalg.solve(Sigma, (z - mu).T).T


# ---------------------------------------------------------------------------
# Step results
# ---------------------------------------------------------------------------
class GradientStepResult(NamedTuple):
    """One iteration of a gradient-based Gaussian baseline."""

    mu: jnp.ndarray
    L: jnp.ndarray
    Sigma: jnp.ndarray
    z: jnp.ndarray
    loss: Any
    adam_state: AdamState


class GsmStepResult(NamedTuple):
    """One iteration of the GSM baseline."""

    mu: jnp.ndarray
    Sigma: jnp.ndarray
    z: jnp.ndarray
    g: jnp.ndarray


# ---------------------------------------------------------------------------
# ADVI (negative ELBO with reparameterization gradients)
# ---------------------------------------------------------------------------
def advi_step(
    key: jax.Array,
    mu: jnp.ndarray,
    L: jnp.ndarray,
    target_score: Callable[[jnp.ndarray], jnp.ndarray],
    batch_size: int,
    adam_state: AdamState,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    target_log_prob: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
) -> GradientStepResult:
    """Perform one ADVI iteration.

    The variational family is a full-covariance Gaussian with
    ``Sigma = L L^T`` where ``L`` is lower triangular. The ELBO gradient is
    estimated with the reparameterization trick:

        grad_mu ELBO = E_eps[ grad_z log p(z) - grad_z log q(z) ]
        grad_L  ELBO = E_eps[ (grad_z log p(z) - grad_z log q(z)) eps^T ]

    so the implementation only needs a (vectorized) target score function,
    not a JAX-traceable target density. The optional ``target_log_prob`` is
    used solely to report the negative ELBO as ``loss``.

    ``target_score`` is expected to be vectorized, i.e. it maps an array of
    shape ``(B, D)`` to an array of scores of shape ``(B, D)``.
    """
    D = mu.shape[0]
    eps_samples = _sample_eps(key, batch_size, D)
    L_tril = jnp.tril(L)
    Sigma = _sigma_from_L(L_tril)

    z = mu + (L_tril @ eps_samples.T).T
    s = target_score(z)  # grad_z log p(z_b)

    diff = s - _grad_log_q(z, mu, Sigma)  # (B, D)

    grad_mu = jnp.mean(diff, axis=0)
    grad_L_full = jnp.mean(diff[:, :, None] * eps_samples[:, None, :], axis=0)
    grad_L = jnp.tril(grad_L_full)

    params = _pack_params(mu, L)
    grads = _pack_params(grad_mu, grad_L)
    new_params, new_adam_state = adam_step(
        params, grads, adam_state, learning_rate, beta1, beta2, eps
    )

    mu_new = new_params["mu"]
    L_new = jnp.tril(new_params["L"])
    Sigma_new = _sigma_from_L(L_new)

    loss = None
    if target_log_prob is not None:
        log_p = jax.vmap(target_log_prob)(z)
        log_q = -0.5 * D * jnp.log(2.0 * jnp.pi) - 0.5 * jnp.linalg.slogdet(
            Sigma
        )[1] - 0.5 * jnp.sum(
            jnp.linalg.solve(Sigma, (z - mu).T).T * (z - mu), axis=1
        )
        loss = -jnp.mean(log_p - log_q)

    return GradientStepResult(mu_new, L_new, Sigma_new, z, loss, new_adam_state)


# ---------------------------------------------------------------------------
# Score-divergence baseline
# ---------------------------------------------------------------------------
def _score_divergence_loss(
    params: dict[str, jnp.ndarray],
    eps_samples: jnp.ndarray,
    target_score: Callable[[jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    mu = params["mu"]
    L = jnp.tril(params["L"])
    Sigma = _sigma_from_L(L)
    z = mu + (L @ eps_samples.T).T
    grad_log_q = _grad_log_q(z, mu, Sigma)
    grad_log_p = target_score(z)
    diff = grad_log_q - grad_log_p
    # ||diff_b||^2_{Sigma} = diff_b^T Sigma diff_b
    return jnp.mean(jnp.einsum("bi,ij,bj->b", diff, Sigma, diff))


def score_step(
    key: jax.Array,
    mu: jnp.ndarray,
    L: jnp.ndarray,
    target_score: Callable[[jnp.ndarray], jnp.ndarray],
    batch_size: int,
    adam_state: AdamState,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> GradientStepResult:
    """Perform one score-divergence minimization step with ADAM."""
    D = mu.shape[0]
    eps_samples = _sample_eps(key, batch_size, D)
    params = _pack_params(mu, L)
    loss, grads = jax.value_and_grad(_score_divergence_loss)(
        params, eps_samples, target_score
    )
    new_params, new_adam_state = adam_step(
        params, grads, adam_state, learning_rate, beta1, beta2, eps
    )

    mu_new = new_params["mu"]
    L_new = jnp.tril(new_params["L"])
    Sigma_new = _sigma_from_L(L_new)
    L_tril = jnp.tril(L)
    z = mu + (L_tril @ eps_samples.T).T
    return GradientStepResult(mu_new, L_new, Sigma_new, z, loss, new_adam_state)


# ---------------------------------------------------------------------------
# Fisher-divergence baseline
# ---------------------------------------------------------------------------
def _fisher_divergence_loss(
    params: dict[str, jnp.ndarray],
    eps_samples: jnp.ndarray,
    target_score: Callable[[jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    mu = params["mu"]
    L = jnp.tril(params["L"])
    Sigma = _sigma_from_L(L)
    z = mu + (L @ eps_samples.T).T
    grad_log_q = _grad_log_q(z, mu, Sigma)
    grad_log_p = target_score(z)
    diff = grad_log_q - grad_log_p
    return jnp.mean(jnp.sum(diff * diff, axis=1))


def fisher_step(
    key: jax.Array,
    mu: jnp.ndarray,
    L: jnp.ndarray,
    target_score: Callable[[jnp.ndarray], jnp.ndarray],
    batch_size: int,
    adam_state: AdamState,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> GradientStepResult:
    """Perform one Fisher-divergence minimization step with ADAM."""
    D = mu.shape[0]
    eps_samples = _sample_eps(key, batch_size, D)
    params = _pack_params(mu, L)
    loss, grads = jax.value_and_grad(_fisher_divergence_loss)(
        params, eps_samples, target_score
    )
    new_params, new_adam_state = adam_step(
        params, grads, adam_state, learning_rate, beta1, beta2, eps
    )

    mu_new = new_params["mu"]
    L_new = jnp.tril(new_params["L"])
    Sigma_new = _sigma_from_L(L_new)
    L_tril = jnp.tril(L)
    z = mu + (L_tril @ eps_samples.T).T
    return GradientStepResult(mu_new, L_new, Sigma_new, z, loss, new_adam_state)


# ---------------------------------------------------------------------------
# GSM (Gaussian score matching), Algorithm 3 of the paper
# ---------------------------------------------------------------------------
def gsm_update(
    mu: jnp.ndarray,
    Sigma: jnp.ndarray,
    z: jnp.ndarray,
    g: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply the per-sample GSM update to a batch of samples and scores.

    For each sample ``(z_b, g_b = grad_z log p(z_b))`` the update is::

        delta_b  = mu - z_b
        a_b      = g_b^T Sigma g_b + (delta_b^T g_b)^2
        rho_b    = (-1 + sqrt(1 + 4 a_b)) / 2
        eps_b    = Sigma g_b - mu + z_b
        delta_mu_b = (1 / (1 + rho_b)) *
                     [I - delta_b g_b^T / (1 + rho_b + delta_b^T g_b)] eps_b
        tilde_mu_b = mu + delta_mu_b
        delta_Sigma_b = delta_b delta_b^T
                        - (tilde_mu_b - z_b)(tilde_mu_b - z_b)^T

    and the new parameters are the batch averages:

        mu_new     = mu + mean_b delta_mu_b
        Sigma_new  = Sigma + mean_b delta_Sigma_b
    """
    def per_sample(z_b: jnp.ndarray, g_b: jnp.ndarray):
        s = g_b
        delta = mu - z_b
        a = s @ Sigma @ s + (delta @ s) ** 2
        rho = 0.5 * (-1.0 + jnp.sqrt(jnp.maximum(0.0, 1.0 + 4.0 * a)))
        eps_vec = Sigma @ s + z_b - mu
        denom = 1.0 + rho + delta @ s
        # Avoid division by (near) zero.
        denom_safe = jnp.where(jnp.abs(denom) < 1e-12, 1e-12, denom)
        delta_mu = (1.0 / (1.0 + rho)) * (
            eps_vec - delta * (s @ eps_vec) / denom_safe
        )
        tilde_mu = mu + delta_mu
        delta_Sigma = jnp.outer(delta, delta) - jnp.outer(
            tilde_mu - z_b, tilde_mu - z_b
        )
        return delta_mu, delta_Sigma

    delta_mus, delta_Sigmas = jax.vmap(per_sample)(z, g)
    mu_new = mu + jnp.mean(delta_mus, axis=0)
    Sigma_new = Sigma + jnp.mean(delta_Sigmas, axis=0)
    Sigma_new = 0.5 * (Sigma_new + Sigma_new.T)
    return mu_new, Sigma_new


def gsm_step(
    key: jax.Array,
    mu: jnp.ndarray,
    Sigma: jnp.ndarray,
    target_score: Callable[[jnp.ndarray], jnp.ndarray],
    batch_size: int,
) -> GsmStepResult:
    """Sample from the current Gaussian, score the target, and apply GSM."""
    D = mu.shape[0]
    Sigma_sym = 0.5 * (Sigma + Sigma.T)
    L = jnp.linalg.cholesky(Sigma_sym + 1e-10 * jnp.eye(D))
    eps_samples = _sample_eps(key, batch_size, D)
    z = mu + (L @ eps_samples.T).T
    g = target_score(z)
    mu_new, Sigma_new = gsm_update(mu, Sigma_sym, z, g)
    return GsmStepResult(mu_new, Sigma_new, z, g)


# ---------------------------------------------------------------------------
# High-level runners
# ---------------------------------------------------------------------------
def run_advi(
    key: jax.Array,
    mu0: jnp.ndarray,
    L0: jnp.ndarray,
    target_score: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    T: int = 100,
    batch_size: int = 10,
    learning_rate: float = 1e-2,
    learning_rate_fn: Optional[Callable[[int], float]] = None,
    target_log_prob: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    adam_state: Optional[AdamState] = None,
    metric_fn: Optional[Callable[[jnp.ndarray, jnp.ndarray], Any]] = None,
    return_history: bool = True,
) -> dict[str, Any]:
    """Run ADVI for ``T`` iterations and return final parameters/history."""
    mu = jnp.asarray(mu0)
    L = jnp.tril(jnp.asarray(L0))
    if adam_state is None:
        adam_state = adam_init(_pack_params(mu, L))

    mu_history: list[jnp.ndarray] = []
    Sigma_history: list[jnp.ndarray] = []
    losses: list[Any] = []
    metrics: list[Any] = []

    for t in range(T):
        key, subkey = jax.random.split(key)
        lr = learning_rate_fn(t) if learning_rate_fn is not None else learning_rate
        result = advi_step(
            subkey,
            mu,
            L,
            target_score,
            batch_size,
            adam_state,
            lr,
            target_log_prob=target_log_prob,
        )
        mu = result.mu
        L = result.L
        adam_state = result.adam_state
        if return_history:
            mu_history.append(mu)
            Sigma_history.append(result.Sigma)
            losses.append(result.loss)
        if metric_fn is not None:
            metrics.append(metric_fn(mu, result.Sigma))

    out: dict[str, Any] = {"mu": mu, "L": L, "Sigma": result.Sigma}
    if return_history:
        out["mu_history"] = jnp.stack(mu_history)
        out["Sigma_history"] = jnp.stack(Sigma_history)
        out["losses"] = losses
    if metric_fn is not None:
        out["metrics"] = metrics
    return out


def run_score(
    key: jax.Array,
    mu0: jnp.ndarray,
    L0: jnp.ndarray,
    target_score: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    T: int = 100,
    batch_size: int = 10,
    learning_rate: float = 1e-2,
    learning_rate_fn: Optional[Callable[[int], float]] = None,
    adam_state: Optional[AdamState] = None,
    metric_fn: Optional[Callable[[jnp.ndarray, jnp.ndarray], Any]] = None,
    return_history: bool = True,
) -> dict[str, Any]:
    """Run the score-divergence baseline for ``T`` iterations."""
    mu = jnp.asarray(mu0)
    L = jnp.tril(jnp.asarray(L0))
    if adam_state is None:
        adam_state = adam_init(_pack_params(mu, L))

    mu_history: list[jnp.ndarray] = []
    Sigma_history: list[jnp.ndarray] = []
    losses: list[Any] = []
    metrics: list[Any] = []

    for t in range(T):
        key, subkey = jax.random.split(key)
        lr = learning_rate_fn(t) if learning_rate_fn is not None else learning_rate
        result = score_step(
            subkey, mu, L, target_score, batch_size, adam_state, lr
        )
        mu = result.mu
        L = result.L
        adam_state = result.adam_state
        if return_history:
            mu_history.append(mu)
            Sigma_history.append(result.Sigma)
            losses.append(result.loss)
        if metric_fn is not None:
            metrics.append(metric_fn(mu, result.Sigma))

    out: dict[str, Any] = {"mu": mu, "L": L, "Sigma": result.Sigma}
    if return_history:
        out["mu_history"] = jnp.stack(mu_history)
        out["Sigma_history"] = jnp.stack(Sigma_history)
        out["losses"] = losses
    if metric_fn is not None:
        out["metrics"] = metrics
    return out


def run_fisher(
    key: jax.Array,
    mu0: jnp.ndarray,
    L0: jnp.ndarray,
    target_score: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    T: int = 100,
    batch_size: int = 10,
    learning_rate: float = 1e-2,
    learning_rate_fn: Optional[Callable[[int], float]] = None,
    adam_state: Optional[AdamState] = None,
    metric_fn: Optional[Callable[[jnp.ndarray, jnp.ndarray], Any]] = None,
    return_history: bool = True,
) -> dict[str, Any]:
    """Run the Fisher-divergence baseline for ``T`` iterations."""
    mu = jnp.asarray(mu0)
    L = jnp.tril(jnp.asarray(L0))
    if adam_state is None:
        adam_state = adam_init(_pack_params(mu, L))

    mu_history: list[jnp.ndarray] = []
    Sigma_history: list[jnp.ndarray] = []
    losses: list[Any] = []
    metrics: list[Any] = []

    for t in range(T):
        key, subkey = jax.random.split(key)
        lr = learning_rate_fn(t) if learning_rate_fn is not None else learning_rate
        result = fisher_step(
            subkey, mu, L, target_score, batch_size, adam_state, lr
        )
        mu = result.mu
        L = result.L
        adam_state = result.adam_state
        if return_history:
            mu_history.append(mu)
            Sigma_history.append(result.Sigma)
            losses.append(result.loss)
        if metric_fn is not None:
            metrics.append(metric_fn(mu, result.Sigma))

    out: dict[str, Any] = {"mu": mu, "L": L, "Sigma": result.Sigma}
    if return_history:
        out["mu_history"] = jnp.stack(mu_history)
        out["Sigma_history"] = jnp.stack(Sigma_history)
        out["losses"] = losses
    if metric_fn is not None:
        out["metrics"] = metrics
    return out


def run_gsm(
    key: jax.Array,
    mu0: jnp.ndarray,
    Sigma0: jnp.ndarray,
    target_score: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    T: int = 100,
    batch_size: int = 10,
    metric_fn: Optional[Callable[[jnp.ndarray, jnp.ndarray], Any]] = None,
    return_history: bool = True,
) -> dict[str, Any]:
    """Run GSM for ``T`` iterations and return final parameters/history."""
    mu = jnp.asarray(mu0)
    Sigma = jnp.asarray(Sigma0)

    mu_history: list[jnp.ndarray] = []
    Sigma_history: list[jnp.ndarray] = []
    metrics: list[Any] = []

    for t in range(T):
        key, subkey = jax.random.split(key)
        result = gsm_step(subkey, mu, Sigma, target_score, batch_size)
        mu = result.mu
        Sigma = result.Sigma
        if return_history:
            mu_history.append(mu)
            Sigma_history.append(Sigma)
        if metric_fn is not None:
            metrics.append(metric_fn(mu, Sigma))

    out: dict[str, Any] = {"mu": mu, "Sigma": Sigma}
    if return_history:
        out["mu_history"] = jnp.stack(mu_history)
        out["Sigma_history"] = jnp.stack(Sigma_history)
    if metric_fn is not None:
        out["metrics"] = metrics
    return out
