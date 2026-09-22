"""Analytic (infinite batch) recursions for the convergence analysis of Section 3.2.

For a Gaussian target ``p = N(mu_*, Sigma_*)`` and ``B -> inf``, the batch
statistics of Algorithm 1 converge (Lemma D.2) to

    zbar -> mu_t,   C -> Sigma_t,
    gbar -> Sigma_*^{-1} (mu_* - mu_t),   Gamma -> Sigma_*^{-1} Sigma_t Sigma_*^{-1},

and the algorithm reduces to the deterministic recursions of Propositions D.3
and D.4:

    Sigma_{t+1} = solution of  Sigma U Sigma + Sigma = V
                  with U = lambda Sigma_*^{-1} Sigma_t Sigma_*^{-1}
                           + lambda/(1+lambda) gbar gbar^T,
                       V = (1 + lambda) Sigma_t
    mu_{t+1} = (mu_t + lambda (Sigma_{t+1} gbar + mu_t)) / (1 + lambda)

in terms of the normalized errors ``eps_t = Sigma_*^{-1/2}(mu_t - mu_*)`` and
``Delta_t = Sigma_*^{-1/2}(Sigma_t - Sigma_*)Sigma_*^{-1/2}``.  Theorem 3.1
states that, with

    alpha = lambda_min(Sigma_*^{-1/2} Sigma_0 Sigma_*^{-1/2}),
    beta  = min(alpha, (1 + lambda) / (1 + lambda + ||eps_0||^2)),
    delta = lambda beta / (1 + lambda),

we have ``||eps_t|| <= (1-delta)^t ||eps_0||`` and
``||Delta_t|| <= (1-delta)^t ||Delta_0|| + t (1-delta)^{t-1} ||eps_0||^2``.

This module implements those recursions so that the theorem can be checked
numerically (see ``experiments/verify_theorem31.py``), and so that the infinite
batch limit can be compared against finite-batch runs of Algorithm 1.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from .linalg import symmetrize, solve_quadratic_matrix_eq


def infinite_batch_step(mu: np.ndarray, Sigma: np.ndarray, mu_star: np.ndarray, Sigma_star: np.ndarray,
                        lam: float) -> Tuple[np.ndarray, np.ndarray]:
    """One iteration of Algorithm 1 in the limit ``B -> inf`` (Gaussian target)."""
    mu = np.asarray(mu, dtype=np.float64).ravel()
    Sigma = symmetrize(np.asarray(Sigma, dtype=np.float64))
    mu_star = np.asarray(mu_star, dtype=np.float64).ravel()
    Sigma_star = symmetrize(np.asarray(Sigma_star, dtype=np.float64))
    prec = np.linalg.inv(Sigma_star)
    gbar = prec @ (mu_star - mu)               # lim gbar
    C = Sigma                                  # lim C
    Gamma = prec @ Sigma @ prec                # lim Gamma
    U = lam * Gamma + (lam / (1.0 + lam)) * np.outer(gbar, gbar)
    V = Sigma + lam * C                        # the (mu_t - zbar) term vanishes
    Sigma_new = solve_quadratic_matrix_eq(U, V)
    mu_new = (mu + lam * (Sigma_new @ gbar + mu)) / (1.0 + lam)
    return mu_new, Sigma_new


def run_infinite_batch(mu0: np.ndarray, Sigma0: np.ndarray, mu_star: np.ndarray, Sigma_star: np.ndarray,
                       lam: float, n_iters: int) -> Dict[str, np.ndarray]:
    """Run the infinite-batch recursion and report the normalized errors of eqs. (14-15)."""
    Sigma_star = symmetrize(np.asarray(Sigma_star, dtype=np.float64))
    mu_star = np.asarray(mu_star, dtype=np.float64).ravel()
    S_half = _sqrtm(Sigma_star)
    S_inv_half = np.linalg.inv(S_half)

    mu = np.asarray(mu0, dtype=np.float64).ravel().copy()
    Sigma = symmetrize(np.asarray(Sigma0, dtype=np.float64)).copy()
    eps = np.zeros((n_iters + 1, S_half.shape[0]))
    Delta = np.zeros((n_iters + 1, S_half.shape[0], S_half.shape[0]))
    eps_norm = np.zeros(n_iters + 1)
    Delta_norm = np.zeros(n_iters + 1)
    for t in range(n_iters + 1):
        if t > 0:
            mu, Sigma = infinite_batch_step(mu, Sigma, mu_star, Sigma_star, lam)
        eps[t] = S_inv_half @ (mu - mu_star)
        Delta[t] = S_inv_half @ (Sigma - Sigma_star) @ S_inv_half
        eps_norm[t] = np.linalg.norm(eps[t])
        Delta_norm[t] = np.linalg.norm(Delta[t])
    return {"mu": mu, "Sigma": Sigma, "eps": eps, "Delta": Delta,
            "eps_norm": eps_norm, "Delta_norm": Delta_norm}


def theoretical_bounds(mu0: np.ndarray, Sigma0: np.ndarray, mu_star: np.ndarray, Sigma_star: np.ndarray,
                       lam: float, n_iters: int) -> Dict[str, np.ndarray]:
    """The multiplicative factors ``(1-delta)^t`` and the bounds of Theorem 3.1."""
    Sigma_star = symmetrize(np.asarray(Sigma_star, dtype=np.float64))
    S_half = _sqrtm(Sigma_star)
    S_inv_half = np.linalg.inv(S_half)
    M = S_inv_half @ symmetrize(np.asarray(Sigma0, dtype=np.float64)) @ S_inv_half
    alpha = float(np.min(np.linalg.eigvalsh(M)))
    eps0 = S_inv_half @ (np.asarray(mu0, dtype=np.float64).ravel() - np.asarray(mu_star).ravel())
    eps0_norm2 = float(eps0 @ eps0)
    beta = min(alpha, (1.0 + lam) / (1.0 + lam + eps0_norm2))
    delta = lam * beta / (1.0 + lam)
    t = np.arange(n_iters + 1)
    decay = (1.0 - delta) ** t
    return {
        "alpha": alpha,
        "beta": beta,
        "delta": delta,
        "eps_bound": decay * np.linalg.norm(eps0),
        "Delta_bound": decay * np.linalg.norm(
            S_inv_half @ (symmetrize(np.asarray(Sigma0, dtype=np.float64)) - Sigma_star) @ S_inv_half)
            + t * (1.0 - delta) ** np.maximum(t - 1, 0) * eps0_norm2,
        "decay": decay,
    }


def _sqrtm(A: np.ndarray) -> np.ndarray:
    w, Q = np.linalg.eigh(symmetrize(A))
    return (Q * np.sqrt(np.maximum(w, 0.0))) @ Q.T
