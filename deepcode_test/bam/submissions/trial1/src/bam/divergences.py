"""Score-based divergence and related diagnostics for Gaussian variational families.

This module implements the score-based divergence used by Batch-and-Match
(BaM), together with the Gaussian specializations that make the BaM update
tractable.

Definition
----------
For a variational density ``q`` and target density ``p`` the score-based
divergence is

    D(q; p) = E_q[
        ( grad_z log q(z) - grad_z log p(z) )^T
        Gamma_q^{-1}
        ( grad_z log q(z) - grad_z log p(z) )
    ]

where ``Gamma_q = E_q[(grad_z log q)(grad_z log q)^T]``.  For a Gaussian
variational family ``q = N(nu, Psi)`` we have ``Gamma_q = Psi^{-1}`` and
therefore

    D(q; p) = E_q[
        (grad_z log p(z) - grad_z log q(z))^T
        Psi
        (grad_z log p(z) - grad_z log q(z))
    ].

The empirical batch estimate evaluated on samples ``z_1, ..., z_B`` and
target scores ``g_b = grad_z log p(z_b)`` for a candidate ``q = N(mu, Sigma)``
is

    D_hat(q; p) = (1 / B) sum_b
        || grad_z log q(z_b) - g_b ||_Sigma^2,

which expands to

    tr(Gamma Sigma) + tr(C Sigma^{-1})
    + ||mu - zbar - Sigma gbar||_{Sigma^{-1}}^2 + const.
"""

from __future__ import annotations

from typing import Callable, Union

import jax.numpy as jnp

__all__ = [
    "score_divergence_gaussian",
    "score_divergence_batch_form",
    "score_divergence_gaussian_closed_form",
    "score_divergence_general",
    "fisher_divergence_estimate",
    "fisher_divergence_gaussian_closed_form",
    "kl_gaussian",
    "reverse_kl_gaussian",
    "symmetrize",
    "tilt_gaussian_mean",
]


def symmetrize(a: jnp.ndarray, jitter: float = 0.0) -> jnp.ndarray:
    """Symmetrize a square matrix and optionally add diagonal jitter."""
    a = 0.5 * (a + a.T)
    if jitter:
        d = a.shape[0]
        a = a + jitter * jnp.eye(d, dtype=a.dtype)
    return a


def _as_scores(
    target_score: Union[Callable[[jnp.ndarray], jnp.ndarray], jnp.ndarray],
    z: jnp.ndarray,
) -> jnp.ndarray:
    """Return target scores evaluated at ``z`` from either a callable or array."""
    if callable(target_score):
        return jnp.asarray(target_score(z))
    return jnp.asarray(target_score)


def score_divergence_gaussian(
    mu: jnp.ndarray,
    Sigma: jnp.ndarray,
    target_score: Union[Callable[[jnp.ndarray], jnp.ndarray], jnp.ndarray],
    z_samples: jnp.ndarray,
) -> jnp.ndarray:
    """Empirical score-based divergence for a Gaussian candidate ``q = N(mu, Sigma)``.

    Computes

        D_hat(q; p) = (1 / B) sum_b || grad_z log q(z_b) - g_b ||_Sigma^2.

    Parameters
    ----------
    mu:
        Mean of the candidate Gaussian, shape ``(D,)``.
    Sigma:
        Covariance of the candidate Gaussian, shape ``(D, D)``.
    target_score:
        Either target scores ``g_b = grad_z log p(z_b)`` with shape ``(B, D)``,
        or a callable ``z -> grad_z log p(z)``.
    z_samples:
        Samples ``z_b`` with shape ``(B, D)``.

    Returns
    -------
    Scalar divergence estimate.
    """
    mu = jnp.asarray(mu)
    Sigma = symmetrize(jnp.asarray(Sigma))
    z = jnp.asarray(z_samples)
    g = _as_scores(target_score, z)

    Sigma_inv = jnp.linalg.inv(Sigma)
    # grad_z log q(z_b) = -Sigma^{-1}(z_b - mu)
    grads_q = -(z - mu) @ Sigma_inv
    diff = grads_q - g
    # ||diff_b||_Sigma^2 = diff_b^T Sigma diff_b
    weighted = diff @ Sigma
    sq = jnp.sum(weighted * diff, axis=1)
    return jnp.mean(sq)


def score_divergence_batch_form(
    mu: jnp.ndarray,
    Sigma: jnp.ndarray,
    zbar: jnp.ndarray,
    gbar: jnp.ndarray,
    C: jnp.ndarray,
    Gamma: jnp.ndarray,
    z: jnp.ndarray | None = None,
    g: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Batch-statistics expansion of the empirical score-based divergence.

    Returns

        tr(Gamma Sigma) + tr(C Sigma^{-1})
        + ||mu - zbar - Sigma gbar||_{Sigma^{-1}}^2 + const,

    where ``const = (2/B) sum_b g_b^T (z_b - zbar)`` is independent of the
    candidate parameters ``(mu, Sigma)``.  When ``z`` and ``g`` are supplied,
    the constant is included so that this matches
    :func:`score_divergence_gaussian` exactly; otherwise it is omitted.
    """
    mu = jnp.asarray(mu)
    Sigma = symmetrize(jnp.asarray(Sigma))
    zbar = jnp.asarray(zbar)
    gbar = jnp.asarray(gbar)
    C = symmetrize(jnp.asarray(C))
    Gamma = symmetrize(jnp.asarray(Gamma))

    Sigma_inv = jnp.linalg.inv(Sigma)
    value = jnp.trace(Gamma @ Sigma) + jnp.trace(C @ Sigma_inv)
    residual = mu - zbar - Sigma @ gbar
    value = value + residual @ Sigma_inv @ residual

    if z is not None and g is not None:
        z = jnp.asarray(z)
        g = jnp.asarray(g)
        const = 2.0 * (jnp.mean(jnp.sum(g * z, axis=1)) - jnp.dot(gbar, zbar))
        value = value + const
    return value


def score_divergence_gaussian_closed_form(
    mu_q: jnp.ndarray,
    Sigma_q: jnp.ndarray,
    mu_p: jnp.ndarray,
    Sigma_p: jnp.ndarray,
) -> jnp.ndarray:
    """Closed-form score-based divergence between two Gaussians.

    For ``q = N(mu_q, Sigma_q)`` and ``p = N(mu_p, Sigma_p)``:

        D(q; p) = tr[(I - Sigma_q Sigma_p^{-1})^2]
                  + (mu_q - mu_p)^T Sigma_p^{-1} Sigma_q Sigma_p^{-1}
                    (mu_q - mu_p).
    """
    mu_q = jnp.asarray(mu_q)
    mu_p = jnp.asarray(mu_p)
    Sigma_q = symmetrize(jnp.asarray(Sigma_q))
    Sigma_p = symmetrize(jnp.asarray(Sigma_p))

    d = mu_q.shape[0]
    Sigma_p_inv = jnp.linalg.inv(Sigma_p)
    m = Sigma_q @ Sigma_p_inv
    eye = jnp.eye(d, dtype=m.dtype)
    trace_term = jnp.trace((eye - m) @ (eye - m))
    diff = mu_q - mu_p
    mean_term = diff @ Sigma_p_inv @ Sigma_q @ Sigma_p_inv @ diff
    return trace_term + mean_term


def score_divergence_general(
    scores_q: jnp.ndarray,
    scores_p: jnp.ndarray,
    Gamma_q: jnp.ndarray,
) -> jnp.ndarray:
    """Generic Monte-Carlo score-based divergence.

    ``scores_q`` and ``scores_p`` are ``(B, D)`` arrays of score values and
    ``Gamma_q`` is the score covariance ``E_q[(grad log q)(grad log q)^T]``.
    """
    diff = jnp.asarray(scores_q) - jnp.asarray(scores_p)
    Gamma_q = symmetrize(jnp.asarray(Gamma_q))
    Gamma_q_inv = jnp.linalg.inv(Gamma_q)
    weighted = diff @ Gamma_q_inv
    return jnp.mean(jnp.sum(weighted * diff, axis=1))


def fisher_divergence_estimate(
    mu: jnp.ndarray,
    Sigma: jnp.ndarray,
    target_score: Union[Callable[[jnp.ndarray], jnp.ndarray], jnp.ndarray],
    z_samples: jnp.ndarray,
) -> jnp.ndarray:
    """Empirical Fisher divergence ``E_q[||grad log q - grad log p||^2]``.

    Unlike the score-based divergence, this uses the Euclidean norm without
    the inverse score-covariance weighting.
    """
    mu = jnp.asarray(mu)
    Sigma = symmetrize(jnp.asarray(Sigma))
    z = jnp.asarray(z_samples)
    g = _as_scores(target_score, z)

    Sigma_inv = jnp.linalg.inv(Sigma)
    grads_q = -(z - mu) @ Sigma_inv
    diff = grads_q - g
    return jnp.mean(jnp.sum(diff * diff, axis=1))


def fisher_divergence_gaussian_closed_form(
    mu_q: jnp.ndarray,
    Sigma_q: jnp.ndarray,
    mu_p: jnp.ndarray,
    Sigma_p: jnp.ndarray,
) -> jnp.ndarray:
    """Closed-form Fisher divergence between two Gaussians.

    For ``q = N(mu_q, Sigma_q)`` and ``p = N(mu_p, Sigma_p)``:

        F(q; p) = tr(A Sigma_q A)
                  + (mu_q - mu_p)^T Sigma_p^{-2} (mu_q - mu_p),

    with ``A = Sigma_p^{-1} - Sigma_q^{-1}``.
    """
    mu_q = jnp.asarray(mu_q)
    mu_p = jnp.asarray(mu_p)
    Sigma_q = symmetrize(jnp.asarray(Sigma_q))
    Sigma_p = symmetrize(jnp.asarray(Sigma_p))

    Sigma_q_inv = jnp.linalg.inv(Sigma_q)
    Sigma_p_inv = jnp.linalg.inv(Sigma_p)
    a = Sigma_p_inv - Sigma_q_inv
    trace_term = jnp.trace(a @ Sigma_q @ a)
    diff = mu_q - mu_p
    mean_term = diff @ Sigma_p_inv @ Sigma_p_inv @ diff
    return trace_term + mean_term


def kl_gaussian(
    mu0: jnp.ndarray,
    Sigma0: jnp.ndarray,
    mu1: jnp.ndarray,
    Sigma1: jnp.ndarray,
) -> jnp.ndarray:
    """Kullback-Leibler divergence ``KL(N(mu0, Sigma0) || N(mu1, Sigma1))``."""
    mu0 = jnp.asarray(mu0)
    mu1 = jnp.asarray(mu1)
    Sigma0 = symmetrize(jnp.asarray(Sigma0))
    Sigma1 = symmetrize(jnp.asarray(Sigma1))

    d = mu0.shape[0]
    Sigma1_inv = jnp.linalg.inv(Sigma1)
    diff = mu1 - mu0
    value = jnp.trace(Sigma1_inv @ Sigma0) + diff @ Sigma1_inv @ diff - d
    _, logdet0 = jnp.linalg.slogdet(Sigma0)
    _, logdet1 = jnp.linalg.slogdet(Sigma1)
    value = value + logdet1 - logdet0
    return 0.5 * value


def reverse_kl_gaussian(
    mu_q: jnp.ndarray,
    Sigma_q: jnp.ndarray,
    mu_p: jnp.ndarray,
    Sigma_p: jnp.ndarray,
) -> jnp.ndarray:
    """Reverse KL ``KL(p || q)`` for Gaussian ``q`` and ``p``."""
    return kl_gaussian(mu_p, Sigma_p, mu_q, Sigma_q)


def tilt_gaussian_mean(
    mu: jnp.ndarray, Sigma: jnp.ndarray, s: jnp.ndarray
) -> jnp.ndarray:
    """Mean of a Gaussian tilted by a linear exponential ``exp(s^T z)``.

    If ``p(z) ∝ N(z; mu, Sigma) exp(s^T z)`` then
    ``p(z) = N(z; mu + Sigma s, Sigma)``.
    """
    mu = jnp.asarray(mu)
    Sigma = symmetrize(jnp.asarray(Sigma))
    s = jnp.asarray(s)
    return mu + Sigma @ s
