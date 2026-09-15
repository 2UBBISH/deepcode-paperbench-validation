"""Batch-and-Match (BaM) algorithm.

Implements Algorithm 1 from the paper:

    for t = 0, ..., T-1:
        1. sample z_1, ..., z_B ~ N(mu_t, Sigma_t)
        2. compute scores g_b = grad_z log p(z_b)
        3. compute batch statistics (zbar, gbar, C, Gamma)
        4. update covariance by solving  Sigma U Sigma + Sigma = V
        5. update mean in closed form

The actual closed-form mean/covariance updates live in
:mod:`bam.match_update`; this module handles sampling, score evaluation,
the iteration loop, and learning-rate schedules.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp

from .batch_stats import BatchStats, compute_batch_stats
from .divergences import symmetrize
from .match_update import match_update

__all__ = [
    "BaMStepResult",
    "sample_mvn",
    "compute_scores",
    "bam_step",
    "run_bam",
    "constant_learning_rate",
    "decaying_learning_rate",
]


class BaMStepResult(NamedTuple):
    """Everything produced by one BaM iteration."""

    mu: jnp.ndarray
    Sigma: jnp.ndarray
    z: jnp.ndarray
    g: jnp.ndarray
    stats: BatchStats
    lam: float


def sample_mvn(
    key: jax.Array,
    mu: jnp.ndarray,
    Sigma: jnp.ndarray,
    num_samples: int,
) -> jnp.ndarray:
    """Draw ``num_samples`` samples from N(mu, Sigma).

    Uses a Cholesky factorization of a symmetrized covariance matrix with a
    small diagonal jitter for numerical stability.
    """
    mu = jnp.asarray(mu)
    Sigma = jnp.asarray(Sigma)

    if mu.ndim != 1:
        raise ValueError("mu must be a one-dimensional vector.")
    if Sigma.shape != (mu.shape[0], mu.shape[0]):
        raise ValueError("Sigma must be a square matrix matching mu.")

    dim = mu.shape[0]
    Sigma_sym = symmetrize(Sigma)
    jitter = 1e-6 if Sigma_sym.dtype == jnp.float32 else 1e-10
    chol = jnp.linalg.cholesky(
        Sigma_sym + jitter * jnp.eye(dim, dtype=Sigma_sym.dtype)
    )
    eps = jax.random.normal(key, (num_samples, dim), dtype=mu.dtype)
    return mu[None, :] + eps @ chol.T


def compute_scores(
    z: jnp.ndarray,
    target_log_prob: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    target_score: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
) -> jnp.ndarray:
    """Compute target scores ``grad_z log p(z)`` for a batch of samples.

    If ``target_score`` is supplied it is assumed to be a vectorized function
    mapping a batch ``(B, D)`` to a batch of score vectors ``(B, D)``. This is
    the appropriate entry point for non-JAX or externally differentiated
    targets (e.g. BridgeStan posteriors).

    Otherwise ``target_log_prob`` must be a JAX-differentiable function from a
    single vector ``z`` (shape ``(D,)``) to a scalar. Scores are obtained by
    ``jax.vmap(jax.grad(target_log_prob))``.
    """
    if target_score is not None:
        return jnp.asarray(target_score(z))
    if target_log_prob is None:
        raise ValueError(
            "Either target_log_prob or target_score must be provided."
        )
    return jax.vmap(jax.grad(target_log_prob))(z)


def bam_step(
    key: jax.Array,
    mu_t: jnp.ndarray,
    Sigma_t: jnp.ndarray,
    *,
    target_log_prob: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    target_score: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    B: int = 10,
    lam: float = 1.0,
    rank_tol: float = 1e-10,
) -> BaMStepResult:
    """Execute one BaM iteration.

    Args:
        key: JAX PRNG key used to sample the current variational Gaussian.
        mu_t: Current variational mean, shape ``(D,)``.
        Sigma_t: Current variational covariance, shape ``(D, D)``.
        target_log_prob: JAX-differentiable log density ``R^D -> R``.
        target_score: Vectorized score function ``(B, D) -> (B, D)``.
        B: Batch size.
        lam: Learning-rate parameter ``lambda_t``.
        rank_tol: Rank tolerance forwarded to the quadratic matrix solver.
    """
    z = sample_mvn(key, mu_t, Sigma_t, B)
    g = compute_scores(z, target_log_prob, target_score)
    stats = compute_batch_stats(z, g)
    mu_new, Sigma_new = match_update(
        mu_t,
        Sigma_t,
        stats,
        lam,
        rank_tol=rank_tol,
    )
    return BaMStepResult(
        mu=mu_new,
        Sigma=Sigma_new,
        z=z,
        g=g,
        stats=stats,
        lam=float(lam),
    )


def constant_learning_rate(B: int, D: int) -> Callable[[int], float]:
    """Constant schedule ``lambda_t = B * D`` used for Gaussian targets."""
    return lambda t: float(B * D)


def decaying_learning_rate(B: int, D: int) -> Callable[[int], float]:
    """Decaying schedule ``lambda_t = B * D / (t + 1)``.

    Used for non-Gaussian and hierarchical posterior targets.
    """
    return lambda t: float(B * D) / float(t + 1)


def run_bam(
    key: jax.Array,
    mu0: jnp.ndarray,
    Sigma0: jnp.ndarray,
    *,
    target_log_prob: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    target_score: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    T: int = 100,
    B: int = 10,
    lam: float = 1.0,
    learning_rate_fn: Optional[Callable[[int], float]] = None,
    rank_tol: float = 1e-10,
    metric_fn: Optional[Callable[..., Any]] = None,
    return_history: bool = True,
) -> dict[str, Any]:
    """Run ``T`` iterations of BaM.

    Args:
        key: JAX PRNG key.
        mu0: Initial variational mean.
        Sigma0: Initial variational covariance.
        target_log_prob: JAX-differentiable log density ``R^D -> R``.
        target_score: Vectorized score function ``(B, D) -> (B, D)``.
        T: Number of iterations.
        B: Batch size.
        lam: Constant learning-rate parameter used when ``learning_rate_fn``
            is not supplied.
        learning_rate_fn: Callable ``t -> lambda_t`` overriding ``lam``.
        rank_tol: Rank tolerance for the quadratic covariance solver.
        metric_fn: Optional callable ``(mu, Sigma, step_result, t) -> value``
            invoked after each iteration. Useful for recording KL, errors, etc.
        return_history: If ``True``, store and return the full covariance
            trajectory (which is memory-intensive for large ``D``).

    Returns:
        Dictionary with keys:
            ``mu`` (final mean),
            ``Sigma`` (final covariance),
            ``mu_history`` (shape ``(T+1, D)``, including the initial value),
            ``Sigma_history`` (shape ``(T+1, D, D)`` or ``None``),
            ``metrics`` (list of metric_fn results),
            ``lambda_history`` (length ``T``).
    """
    mu = jnp.asarray(mu0)
    Sigma = jnp.asarray(Sigma0)

    keys = jax.random.split(key, max(T, 1))

    mu_history = [mu]
    sigma_history = [Sigma] if return_history else []
    metrics_history: list[Any] = []
    lambda_history: list[float] = []

    for t in range(T):
        lam_t = lam if learning_rate_fn is None else learning_rate_fn(t)
        lam_t = float(lam_t)

        result = bam_step(
            keys[t],
            mu,
            Sigma,
            target_log_prob=target_log_prob,
            target_score=target_score,
            B=B,
            lam=lam_t,
            rank_tol=rank_tol,
        )

        mu = result.mu
        Sigma = result.Sigma

        mu_history.append(mu)
        if return_history:
            sigma_history.append(Sigma)
        lambda_history.append(lam_t)

        if metric_fn is not None:
            metrics_history.append(metric_fn(mu, Sigma, result, t))

    return {
        "mu": mu,
        "Sigma": Sigma,
        "mu_history": jnp.stack(mu_history),
        "Sigma_history": jnp.stack(sigma_history) if return_history else None,
        "metrics": metrics_history,
        "lambda_history": lambda_history,
    }
