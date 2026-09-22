"""The score-based divergence of Section 2 and Appendix A, and related divergences.

Definition A.2:

    D(q ; p) = E_q[ (grad log q/p)^T Gamma_q^{-1} (grad log q/p) ],
    Gamma_q   = E_q[ (grad log q)(grad log q)^T ].

For a Gaussian variational family ``Gamma_q = Cov(q)^{-1}`` (eq. (35)), so that

    D(q ; p) = E_q[ || grad log(q/p) ||^2_{Cov(q)} ],      (eq. (2))

which is the definition used in the main text, and which is the quantity that
BaM estimates with a batch of samples (eq. (3)-(4)).

The module also provides

* the closed-form divergence between two Gaussians (Proposition A.7),
* the special cases of annealing (Theorem A.5) and exponential tilting
  (Theorem A.6), used in the unit tests, and
* the Fisher divergence and the KL divergence estimators used for evaluation.
"""

from __future__ import annotations

import jax
import numpy as np

from .linalg import psd_pinv, symmetrize
from .targets import Target, gaussian_cholesky, gaussian_logpdf, sample_gaussian


# ----------------------------------------------------------------------------
# Monte-Carlo estimation of the score-based divergence (eq. (3))
# ----------------------------------------------------------------------------


def score_based_divergence(target: Target, mu: np.ndarray, Sigma: np.ndarray, n_samples: int,
                           key, weight: str = "cov") -> float:
    """Monte-Carlo estimate of ``D(q ; p)`` for ``q = N(mu, Sigma)``.

    ``weight='cov'`` gives the score-based divergence (weighted by
    ``Cov(q) = Sigma``); ``weight='identity'`` gives the (non affine invariant)
    Fisher divergence.
    """
    mu = np.asarray(mu, dtype=np.float64).ravel()
    Sigma = symmetrize(np.asarray(Sigma, dtype=np.float64))
    chol = gaussian_cholesky(Sigma)
    Z = sample_gaussian(key, mu, chol, n_samples)
    s_p = np.asarray(target.score(Z))
    s_q = (mu[None, :] - Z) @ np.linalg.inv(Sigma).T  # grad log q(z)
    diff = s_q - s_p
    if weight == "cov":
        vals = np.einsum("bi,ij,bj->b", diff, Sigma, diff)
    else:
        vals = np.sum(diff**2, axis=1)
    return float(np.mean(vals))


def fisher_divergence(target: Target, mu: np.ndarray, Sigma: np.ndarray, n_samples: int, key) -> float:
    """Monte-Carlo estimate of the Fisher divergence ``E_q[||grad log q/p||^2]``."""
    return score_based_divergence(target, mu, Sigma, n_samples, key, weight="identity")


# ----------------------------------------------------------------------------
# closed forms
# ----------------------------------------------------------------------------


def gaussian_score_divergence(mu_p: np.ndarray, Sigma_p: np.ndarray, mu_q: np.ndarray,
                              Sigma_q: np.ndarray) -> float:
    """``D(q ; p)`` for Gaussian ``p`` and ``q`` (Proposition A.7).

        D(q ; p) = tr[(I - Psi Sigma^{-1})^2] + (nu - mu)^T Sigma^{-1} Psi Sigma^{-1} (nu - mu)
    """
    Sigma_p = symmetrize(np.asarray(Sigma_p, dtype=np.float64))
    Sigma_q = symmetrize(np.asarray(Sigma_q, dtype=np.float64))
    mu_p = np.asarray(mu_p, dtype=np.float64).ravel()
    mu_q = np.asarray(mu_q, dtype=np.float64).ravel()
    D = Sigma_p.shape[0]
    prec = np.linalg.inv(Sigma_p)
    M = np.eye(D) - Sigma_q @ prec
    tr = np.trace(M @ M)
    diff = mu_q - mu_p
    quad = diff @ prec @ Sigma_q @ prec @ diff
    return float(tr + quad)


def annealing_divergence(beta: float, dim: int) -> float:
    """``D(q ; p) = D (beta - 1)^2`` when ``p ~ q^beta`` (Theorem A.5)."""
    return float(dim * (beta - 1.0) ** 2)


def exponential_tilting_divergence(theta: np.ndarray, Sigma_q: np.ndarray) -> float:
    """``D(q ; p) = theta^T Gamma_q^{-1} theta`` for ``p(z) ~ q(z) exp(theta^T z)``.

    For a Gaussian ``q`` with covariance ``Psi`` we have ``Gamma_q = Psi^{-1}``,
    hence ``Gamma_q^{-1} = Psi`` (Theorem A.6).
    """
    theta = np.asarray(theta, dtype=np.float64).ravel()
    Sigma_q = symmetrize(np.asarray(Sigma_q, dtype=np.float64))
    return float(theta @ Sigma_q @ theta)


# ----------------------------------------------------------------------------
# empirical (batch) estimate used inside BaM
# ----------------------------------------------------------------------------


def empirical_score_divergence(mu: np.ndarray, Sigma: np.ndarray, zbar: np.ndarray, gbar: np.ndarray,
                               C: np.ndarray, Gamma: np.ndarray) -> float:
    """The batch estimate ``D_hat_{q_t}(q ; p)`` of eq. (4) and eq. (98).

        D_hat(q ; p) = tr(Gamma Sigma) + tr(C Sigma^{-1}) + ||mu - zbar - Sigma gbar||^2_{Sigma^{-1}}
                       + const.

    (the additive constant, ``tr(Gamma_centered-free terms)``, is dropped since
    it does not depend on ``q``).  This is the quantity that the MATCH step
    minimizes, up to the KL regularizer.
    """
    mu = np.asarray(mu, dtype=np.float64).ravel()
    Sigma = symmetrize(np.asarray(Sigma, dtype=np.float64))
    Sigma_inv = psd_pinv(Sigma)
    r = mu - np.asarray(zbar).ravel() - Sigma @ np.asarray(gbar).ravel()
    return float(np.trace(np.asarray(Gamma) @ Sigma) + np.trace(np.asarray(C) @ Sigma_inv) + r @ Sigma_inv @ r)


def kl_gaussian(mu_q: np.ndarray, Sigma_q: np.ndarray, mu_p: np.ndarray, Sigma_p: np.ndarray) -> float:
    """``KL(q ; p)`` between two multivariate Gaussians (in nats)."""
    mu_q = np.asarray(mu_q, dtype=np.float64).ravel()
    mu_p = np.asarray(mu_p, dtype=np.float64).ravel()
    Sigma_q = symmetrize(np.asarray(Sigma_q, dtype=np.float64))
    Sigma_p = symmetrize(np.asarray(Sigma_p, dtype=np.float64))
    D = mu_q.shape[0]
    Lp = np.linalg.cholesky(Sigma_p)
    Lq = np.linalg.cholesky(Sigma_q)
    diff = mu_p - mu_q
    sol = np.linalg.solve(Lp, diff)
    logdet_p = 2.0 * np.sum(np.log(np.diag(Lp)))
    logdet_q = 2.0 * np.sum(np.log(np.diag(Lq)))
    tr = np.trace(np.linalg.solve(Sigma_p, Sigma_q))
    return float(0.5 * (tr + sol @ sol - D + logdet_p - logdet_q))


def kl_gaussian_samples(target: Target, mu: np.ndarray, Sigma: np.ndarray, n_samples: int, key,
                        direction: str = "forward") -> float:
    """Monte-Carlo estimate of ``KL(p ; q)`` (forward) or ``KL(q ; p)`` (reverse).

    ``q`` is the Gaussian ``N(mu, Sigma)`` and ``p`` is the (normalized) target.
    These are the evaluation metrics of Section 5.1.
    """
    mu = np.asarray(mu, dtype=np.float64).ravel()
    Sigma = symmetrize(np.asarray(Sigma, dtype=np.float64))
    chol = gaussian_cholesky(Sigma)
    if direction == "forward":  # E_p[log p - log q]
        key_z, key_q = jax.random.split(key)
        Z = target.sample(key_z, n_samples)
    elif direction == "reverse":  # E_q[log q - log p]
        Z = sample_gaussian(key, mu, chol, n_samples)
    else:
        raise ValueError(direction)
    log_q = np.asarray(gaussian_logpdf(jax.numpy.asarray(Z), jax.numpy.asarray(mu), jax.numpy.asarray(chol)))
    log_p = np.asarray(target.log_density(jax.numpy.asarray(Z)))
    if direction == "forward":
        return float(np.mean(log_p - log_q))
    return float(np.mean(log_q - log_p))


__all__ = [
    "score_based_divergence",
    "fisher_divergence",
    "gaussian_score_divergence",
    "annealing_divergence",
    "exponential_tilting_divergence",
    "empirical_score_divergence",
    "kl_gaussian",
    "kl_gaussian_samples",
]
