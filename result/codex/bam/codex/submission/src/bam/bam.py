"""The batch and match (BaM) algorithm of Section 3.1 (Algorithm 1).

One iteration consists of

* a BATCH step: draw ``z_1..z_B ~ q_t``, evaluate the target scores
  ``g_b = grad log p(z_b)``, and form the batch statistics
  ``zbar``, ``gbar``, ``C``, ``Gamma``; and
* a MATCH step: minimize

      L_BaM(q) = D_hat_{q_t}(q ; p) + (2 / lambda_t) KL(q_t ; q)

  in closed form.  With

      U = lambda_t Gamma + lambda_t/(1+lambda_t) gbar gbar^T
      V = Sigma_t + lambda_t C + lambda_t/(1+lambda_t) (mu_t - zbar)(mu_t - zbar)^T

  the updated covariance solves the quadratic matrix equation

      Sigma_{t+1} U Sigma_{t+1} + Sigma_{t+1} = V        (eq. (9))

  whose solution is ``Sigma_{t+1} = 2 V [I + (I + 4 U V)^{1/2}]^{-1}``
  (eq. (12), Lemma B.1), and the updated mean is

      mu_{t+1} = 1/(1+lambda_t) mu_t + lambda_t/(1+lambda_t) (Sigma_{t+1} gbar + zbar)
                                                          (eq. (13))

  which must be computed *after* the covariance update.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import jax
import numpy as np

from .linalg import (
    ensure_positive_definite,
    low_rank_factor_of_U,
    symmetrize,
    solve_quadratic_matrix_eq,
    solve_quadratic_matrix_eq_low_rank,
)
from .targets import Target, gaussian_cholesky, sample_gaussian


def lambda_constant(value: float) -> Callable[[int], float]:
    """Constant inverse regularization ``lambda_t = value``."""
    return lambda t: float(value)


def lambda_decay(value: float, power: float = 1.0) -> Callable[[int], float]:
    """Decaying inverse regularization ``lambda_t = value / (t + 1)^power``.

    The paper uses ``lambda_t = B D / (t + 1)`` (power 1) for non-Gaussian
    targets (Section 5.1) and for the posterior inference applications
    (Section 5.2), and reports ``BD / sqrt(t+1)`` as an alternative schedule in
    Appendix E.4.
    """
    return lambda t: float(value) / (float(t) + 1.0) ** float(power)


@dataclass
class BaMConfig:
    batch_size: int = 10
    lam: Callable[[int], float] = lambda_constant(10.0)
    #: 'auto' uses the low-rank solver of Lemma B.3 when it is cheaper (B < D)
    solver: str = "auto"
    #: numerical floor applied to the variational covariance after each update
    jitter: float = 1e-12
    #: relative/absolute eigenvalue floors used to keep the covariance positive
    #: definite (the closed-form update is PD in exact arithmetic; on
    #: ill-conditioned targets round-off can make it slightly indefinite)
    min_eig_abs: float = 1e-12
    min_eig_rel: float = 1e-12
    #: if an update is not finite (which can happen when the current
    #: approximation has drifted into a region where the target score
    #: overflows), keep the previous iterate instead of failing
    skip_nonfinite_updates: bool = True


class BaM:
    """Batch and match variational inference."""

    def __init__(self, target: Target, config: BaMConfig):
        self.target = target
        self.config = config

    # -- single iteration ----------------------------------------------------
    def step(self, mu: np.ndarray, Sigma: np.ndarray, key, t: int) -> Dict[str, np.ndarray]:
        cfg = self.config
        B = int(cfg.batch_size)
        D = int(self.target.dim)
        lam = float(cfg.lam(t))

        # ---- BATCH step ----------------------------------------------------
        Z = sample_gaussian(key, mu, gaussian_cholesky(Sigma), B)  # (B, D)
        G = np.asarray(self.target.score(Z))                        # (B, D)
        zbar = Z.mean(axis=0)
        gbar = G.mean(axis=0)
        Zc = Z - zbar
        Gc = G - gbar
        C = (Zc.T @ Zc) / B
        Gamma = (Gc.T @ Gc) / B

        # ---- MATCH step ----------------------------------------------------
        U = lam * Gamma + (lam / (1.0 + lam)) * np.outer(gbar, gbar)
        V = Sigma + lam * C + (lam / (1.0 + lam)) * np.outer(mu - zbar, mu - zbar)

        solver = cfg.solver
        if solver == "auto":
            solver = "low_rank" if B <= D else "cholesky"
        Sigma_new = None
        try:
            if solver == "low_rank":
                Q = low_rank_factor_of_U(Gamma, gbar, lam, batch=G)
                Sigma_new = solve_quadratic_matrix_eq_low_rank(Q, V)
            elif solver in ("cholesky", "direct"):
                Sigma_new = solve_quadratic_matrix_eq(U, V)
            else:
                raise ValueError(f"unknown solver {solver!r}")
        except (np.linalg.LinAlgError, ValueError):
            if not cfg.skip_nonfinite_updates:
                raise
            Sigma_new = None   # the batch statistics were not numerically usable

        nonfinite = Sigma_new is None or not (np.isfinite(Sigma_new).all() and np.isfinite(V).all()
                                              and np.isfinite(np.asarray(mu)).all()
                                              and np.isfinite(np.asarray(gbar)).all())
        if nonfinite:
            if not cfg.skip_nonfinite_updates:
                raise FloatingPointError("non-finite BaM update")
            # the current approximation sits in a region where the target score
            # overflows; freeze the iterate instead of propagating NaNs
            mu_new, Sigma_new = mu.copy(), Sigma.copy()
        else:
            Sigma_new = ensure_positive_definite(symmetrize(Sigma_new), cfg.min_eig_abs, cfg.min_eig_rel)
            Sigma_new = Sigma_new + cfg.jitter * np.eye(D)
            mu_new = (mu + lam * (Sigma_new @ gbar + zbar)) / (1.0 + lam)
            if not (np.isfinite(mu_new).all() and np.isfinite(Sigma_new).all()):
                mu_new, Sigma_new = mu.copy(), Sigma.copy()
                nonfinite = True
        return {"mu": mu_new, "Sigma": Sigma_new, "U": U, "V": V, "zbar": zbar, "gbar": gbar,
                "C": C, "Gamma": Gamma, "lam": lam, "nonfinite": nonfinite}

    # -- full run ------------------------------------------------------------
    def run(self, key, n_iters: int, mu0: np.ndarray, Sigma0: np.ndarray,
            n_diag_samples: int = 0, diag_every: int = 1, diag_key=None) -> Dict[str, np.ndarray]:
        """Run BaM for ``n_iters`` iterations.

        Returns a dictionary with the iterates (``mu``, ``Sigma`` of shape
        ``(n_iters + 1, ...)``) and the cumulative number of *gradient
        evaluations* of the target score (``B`` per iteration), which is the
        x-axis used throughout the paper's figures.
        """
        D = int(self.target.dim)
        mu = np.asarray(mu0, dtype=np.float64).ravel().copy()
        Sigma = symmetrize(np.asarray(Sigma0, dtype=np.float64)).copy()

        mus = np.zeros((n_iters + 1, D))
        Sigmas = np.zeros((n_iters + 1, D, D))
        grad_evals = np.zeros(n_iters + 1, dtype=np.int64)
        div_hist = []
        nonfinite_steps = 0
        mus[0], Sigmas[0] = mu, Sigma
        if n_diag_samples and diag_key is not None:
            from .divergence import score_based_divergence
            div_hist.append(score_based_divergence(self.target, mu, Sigma, n_diag_samples, diag_key))
        else:
            div_hist.append(np.nan)

        keys = jax.random.split(key, n_iters)
        for t in range(n_iters):
            out = self.step(mu, Sigma, keys[t], t)
            mu, Sigma = out["mu"], out["Sigma"]
            nonfinite_steps += int(bool(out.get("nonfinite", False)))
            mus[t + 1], Sigmas[t + 1] = mu, Sigma
            grad_evals[t + 1] = grad_evals[t] + int(self.config.batch_size)
            if n_diag_samples and diag_key is not None and ((t + 1) % diag_every == 0):
                from .divergence import score_based_divergence
                div_hist.append(score_based_divergence(self.target, mu, Sigma, n_diag_samples, diag_key))
            elif n_diag_samples:
                div_hist.append(np.nan)
        return {
            "mu": mus,
            "Sigma": Sigmas,
            "grad_evals": grad_evals,
            "divergence": np.asarray(div_hist, dtype=np.float64),
            "n_iters": n_iters,
            "batch_size": int(self.config.batch_size),
            "nonfinite_steps": nonfinite_steps,
        }


def run_bam(target: Target, key, n_iters: int, batch_size: int, lam: Callable[[int], float],
            mu0: Optional[np.ndarray] = None, Sigma0: Optional[np.ndarray] = None,
            solver: str = "auto", **kwargs) -> Dict[str, np.ndarray]:
    """Convenience wrapper: run BaM from the paper's default initialization.

    Unless specified otherwise the experiments initialize
    ``mu_0 ~ Uniform[0, 0.1]^D`` and ``Sigma_0 = I`` (Appendices E.3 and E.5).
    """
    D = int(target.dim)
    if mu0 is None:
        key, sub = jax.random.split(key)
        mu0 = 0.1 * np.asarray(jax.random.uniform(sub, (D,)))
    if Sigma0 is None:
        Sigma0 = np.eye(D)
    bam = BaM(target, BaMConfig(batch_size=batch_size, lam=lam, solver=solver))
    return bam.run(key, n_iters, mu0, Sigma0, **kwargs)
