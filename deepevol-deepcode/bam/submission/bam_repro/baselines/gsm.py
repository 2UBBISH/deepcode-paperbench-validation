"""GSM baseline: Gaussian Score Matching (Modi et al., 2023), Algorithm 3.

This module implements the Gaussian Score Matching (GSM) baseline that is
compared against BaM in Sections 5.1--5.3 (Figures 5.1, 5.2, 5.3, 5.4).

Algorithm 3 (paper, Appendix E.1)::

    Input: Iterations T, batch size B, unnormalized target p~, initial mean
           mu_0, initial covariance Sigma_0
    for t = 0, ..., T-1 do
        Sample z_1, ..., z_B ~ q_t = N(mu_t, Sigma_t)
        for b = 1, ..., B do
            s_b = grad_z log p~(z_b)
            eps_b = Sigma_t s_b - mu_t + z_b
            solve  rho (1 + rho) = s_b^T Sigma_t s_b + [(mu_t - z_b)^T s_b]^2  for rho > 0
            delta_mu_b = 1/(1+rho) [ I - (mu_t - z_b) s_b^T / (1 + rho + (mu_t - z_b)^T s_b) ] eps_b
            delta_Sigma_b = (mu_t - z_b)(mu_t - z_b)^T
                            - (mu~_b - z_b)(mu~_b - z_b)^T,   mu~_b = mu_t + delta_mu_b
        end for
        mu_{t+1}    = mu_t    + (1/B) sum_b delta_mu_b
        Sigma_{t+1} = Sigma_t + (1/B) sum_b delta_Sigma_b
    end for

Notes
-----
* ``rho`` is the positive root of ``rho^2 + rho - c = 0`` with
  ``c = s^T Sigma s + [(mu - z)^T s]^2 >= 0``, i.e.
  ``rho = (sqrt(1 + 4 c) - 1) / 2 >= 0`` (positivity guard included).
* GSM is a *per-sample* update, so each iteration costs exactly ``B``
  gradient evaluations.  This is tracked exactly like in the other
  baselines so experiments can plot metrics against gradient evaluations.
* The covariance update ``delta_Sigma_b`` is not guaranteed to keep
  ``Sigma`` positive definite; following the paper's implementation the
  updated covariance is re-project onto ``S_{++}^D`` (symmetrize + clip
  eigenvalues / jittered Cholesky).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..bam.matrix_equations import (
    _HAS_JAX,
    array_namespace,
    ensure_spd,
    symmetrize,
)
from ..bam.vi_base import standard_normal

__all__ = [
    "GSM",
    "GSMResult",
    "gsm_step",
    "gsm_sample_update",
    "gsm_batch_update",
    "solve_rho",
    "gsm_fit",
    "gsm_divergence",
    "GSMResult",
]

_JITTER = 1e-12


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _default_xp(*arrays):
    """Return the array namespace of ``arrays`` (NumPy fallback)."""
    try:
        return array_namespace(*arrays)
    except Exception:  # pragma: no cover - defensive
        return np


def _as_2d_vector(v, xp):
    v = xp.reshape(v, (-1,))
    return v


def solve_rho(s_b, Sigma_t, mu_t, z_b, xp=None):
    """Positive root of ``rho (1 + rho) = s^T Sigma s + [(mu - z)^T s]^2``.

    Parameters
    ----------
    s_b : (D,) array
        Score of the target at ``z_b``.
    Sigma_t : (D, D) array
        Current variational covariance.
    mu_t : (D,) array
        Current variational mean.
    z_b : (D,) array
        Sample used for this update.

    Returns
    -------
    rho : scalar array
        The non-negative root ``(sqrt(1 + 4 c) - 1) / 2``.
    """
    if xp is None:
        xp = _default_xp(s_b, Sigma_t, mu_t, z_b)
    s_b = _as_2d_vector(s_b, xp)
    mu_t = _as_2d_vector(mu_t, xp)
    z_b = _as_2d_vector(z_b, xp)
    d = mu_t - z_b
    quad = s_b @ (Sigma_t @ s_b)
    cross = d @ s_b
    c = quad + cross ** 2
    c = xp.maximum(c, 0.0)
    return 0.5 * (xp.sqrt(1.0 + 4.0 * c) - 1.0)


def gsm_sample_update(mu_t, Sigma_t, z_b, s_b, xp=None, return_rho=False):
    """Per-sample GSM update ``(delta_mu_b, delta_Sigma_b)`` (Algorithm 3)."""
    if xp is None:
        xp = _default_xp(mu_t, Sigma_t, z_b, s_b)
    mu_t = _as_2d_vector(mu_t, xp)
    z_b = _as_2d_vector(z_b, xp)
    s_b = _as_2d_vector(s_b, xp)
    D = mu_t.shape[0]
    eye = xp.eye(D, dtype=mu_t.dtype)

    rho = solve_rho(s_b, Sigma_t, mu_t, z_b, xp=xp)

    # eps_b = Sigma_t s_b - mu_t + z_b
    eps_b = Sigma_t @ s_b - mu_t + z_b

    diff = mu_t - z_b                      # (mu_t - z_b)
    cross = diff @ s_b                     # (mu_t - z_b)^T s_b

    denom = 1.0 + xp.asarray(rho) + cross
    # guard against (numerically) zero denominator
    denom = xp.where(xp.abs(denom) < _JITTER, xp.sign(denom) * _JITTER + _JITTER,
                     denom)

    # I - diff s^T / (1 + rho + diff^T s)
    M = eye - xp.outer(diff, s_b) / denom
    delta_mu_b = (M @ eps_b) / (1.0 + xp.asarray(rho))

    mu_tilde_b = mu_t + delta_mu_b
    outer_old = xp.outer(diff, diff)
    outer_new = xp.outer(mu_tilde_b - z_b, mu_tilde_b - z_b)
    delta_Sigma_b = outer_old - outer_new

    if return_rho:
        return delta_mu_b, delta_Sigma_b, rho
    return delta_mu_b, delta_Sigma_b


def gsm_batch_update(mu_t, Sigma_t, z, g, xp=None, return_deltas=False):
    """Average the per-sample GSM updates over a batch ``z`` with scores ``g``.

    ``z`` and ``g`` have shape ``(B, D)``.
    """
    if xp is None:
        xp = _default_xp(mu_t, Sigma_t, z, g)
    z = xp.asarray(z)
    g = xp.asarray(g)
    if z.ndim == 1:
        z = z[None, :]
    if g.ndim == 1:
        g = g[None, :]
    B = z.shape[0]

    d_mu = None
    d_Sig = None
    deltas_mu: List[Any] = []
    deltas_Sig: List[Any] = []
    for b in range(B):
        dmu_b, dSig_b = gsm_sample_update(mu_t, Sigma_t, z[b], g[b], xp=xp)
        d_mu = dmu_b if d_mu is None else d_mu + dmu_b
        d_Sig = dSig_b if d_Sig is None else d_Sig + dSig_b
        if return_deltas:
            deltas_mu.append(dmu_b)
            deltas_Sig.append(dSig_b)
    d_mu = d_mu / B
    d_Sig = d_Sig / B
    mu_next = mu_t + d_mu
    Sigma_next = Sigma_t + d_Sig
    if return_deltas:
        return mu_next, Sigma_next, deltas_mu, deltas_Sig
    return mu_next, Sigma_next


def gsm_step(mu_t, Sigma_t, z, g, xp=None, check_spd=True, jitter=_JITTER):
    """One full GSM iteration: batch updates + SPD re-projection.

    Returns ``(mu_next, Sigma_next)``.
    """
    if xp is None:
        xp = _default_xp(mu_t, Sigma_t, z, g)
    mu_next, Sigma_next = gsm_batch_update(mu_t, Sigma_t, z, g, xp=xp)
    Sigma_next = symmetrize(xp.asarray(Sigma_next))
    if check_spd:
        Sigma_next = xp.asarray(ensure_spd(Sigma_next, jitter=jitter))
    return mu_next, Sigma_next


def gsm_divergence(mu, Sigma, target, n_samples=4096, rng=None, xp=None):
    """Monte-Carlo score-based divergence ``D(q; p)`` for a GSM iterate.

    Uses the Gaussian form of the weighted score-based divergence
    (Definition A.2 with ``Gamma_q = Cov(q)^{-1}``, eq. (2) in Section 2.2).
    """
    if xp is None:
        xp = _default_xp(mu, Sigma)
    mu = xp.asarray(mu)
    Sigma = xp.asarray(Sigma)
    D = mu.shape[0]
    L = xp.linalg.cholesky(ensure_spd(Sigma, jitter=_JITTER))
    eps = standard_normal((n_samples, D), xp=xp, rng=rng)
    if D == 1:
        z = mu + (L @ eps.T).T
    else:
        z = mu + eps @ L.T
    g = xp.asarray(target.score(z) if hasattr(target, "score") else target(z))
    diff = g - (z - mu) @ xp.linalg.inv(Sigma)  # grad log p - grad log q  (sign: d log q = -Sigma^{-1}(z-mu))
    # grad log q = -Sigma^{-1}(z - mu), so grad log(q/p) = grad log q - grad log p
    r = -diff
    weighted = xp.einsum("bi,ij,bj->b", r, Sigma, r)
    return float(xp.mean(weighted))


# ---------------------------------------------------------------------------
# result container
# ---------------------------------------------------------------------------
@dataclass
class GSMResult:
    """Result of a GSM run."""

    mu: Any
    Sigma: Any
    n_iter: int
    grad_evals: int
    batch_size: int = 1
    mu_history: Optional[List[Any]] = None
    Sigma_history: Optional[List[Any]] = None
    divergences: Optional[List[float]] = None
    name: str = "gsm"

    @property
    def dim(self) -> int:
        return int(np.asarray(self.mu).reshape(-1).shape[0])

    @property
    def grad_evaluations(self) -> int:
        return int(self.grad_evals)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_iter": int(self.n_iter),
            "grad_evals": int(self.grad_evals),
            "batch_size": int(self.batch_size),
            "mu": np.asarray(self.mu),
            "Sigma": np.asarray(self.Sigma),
        }


# ---------------------------------------------------------------------------
# GSM optimizer
# ---------------------------------------------------------------------------
class GSM:
    """Gaussian Score Matching (Modi et al., 2023) — Algorithm 3.

    Parameters
    ----------
    mu0, Sigma0 : array
        Initial variational parameters.
    score_fn : callable, optional
        ``z -> grad log p(z)``; may take a single ``(D,)`` vector or a
        ``(B, D)`` batch of samples.
    target : object, optional
        Object exposing ``.score(z)`` (used if ``score_fn`` is ``None``).
    batch_size : int
        Number of samples (and gradient evaluations) per iteration.
    seed, rng : int / np.random.Generator
        Randomness control.
    check_spd : bool
        Re-project the covariance onto the SPD cone after each update.
    """

    name = "gsm"

    def __init__(
        self,
        mu0,
        Sigma0,
        score_fn: Optional[Callable] = None,
        target: Any = None,
        batch_size: int = 2,
        seed: int = 0,
        rng: Optional[np.random.Generator] = None,
        xp: Any = None,
        dtype: Any = None,
        track_history: bool = False,
        history_every: int = 1,
        check_spd: bool = True,
        jitter: float = _JITTER,
        **kwargs: Any,
    ) -> None:
        if rng is None:
            rng = np.random.default_rng(seed)
        self.rng = rng
        self.seed = seed
        self.xp = xp if xp is not None else np
        self.dtype = dtype if dtype is not None else np.float64

        if score_fn is None:
            if target is None:
                raise ValueError("GSM requires either `score_fn` or `target`.")
            score_fn = getattr(target, "score", None) or getattr(target, "grad_log_prob", None)
            if score_fn is None:
                raise ValueError("`target` does not expose `score`/`grad_log_prob`.")
        self.score_fn = score_fn
        self.target = target

        self.mu = np.asarray(mu0, dtype=np.float64).reshape(-1)
        self.Sigma = ensure_spd(symmetrize(np.asarray(Sigma0, dtype=np.float64)))
        self.batch_size = int(batch_size)
        self.t = 0
        self.grad_evals = 0
        self.track_history = bool(track_history)
        self.history_every = max(1, int(history_every))
        self.check_spd = bool(check_spd)
        self.jitter = float(jitter)
        self.mu_history: List[np.ndarray] = []
        self.Sigma_history: List[np.ndarray] = []
        self._record()

    # -- basics -------------------------------------------------------------
    @property
    def dim(self) -> int:
        return int(self.mu.shape[0])

    @property
    def mean(self):
        return self.mu

    @property
    def covariance(self):
        return self.Sigma

    def _record(self) -> None:
        if self.track_history and (self.t % self.history_every == 0):
            self.mu_history.append(self.mu.copy())
            self.Sigma_history.append(self.Sigma.copy())

    # -- sampling / scoring -------------------------------------------------
    def sample(self, batch_size: Optional[int] = None):
        """Sample ``z ~ N(mu, Sigma)`` with reparameterization."""
        B = int(batch_size) if batch_size is not None else self.batch_size
        D = self.dim
        L = np.linalg.cholesky(ensure_spd(self.Sigma, jitter=self.jitter))
        eps = standard_normal((B, D), xp=np, rng=self.rng)
        return self.mu + eps @ L.T

    def evaluate_scores(self, z):
        """Evaluate the target score at samples ``z`` of shape ``(B, D)``."""
        z = np.asarray(z)
        try:
            g = self.score_fn(z)
        except Exception:
            g = np.stack([np.asarray(self.score_fn(zb)) for zb in z])
        return np.asarray(g)

    # -- main loop ----------------------------------------------------------
    def step(self):
        """Perform one GSM iteration; returns ``(mu_{t+1}, Sigma_{t+1})``."""
        z = self.sample()
        g = self.evaluate_scores(z)
        self.grad_evals += int(np.asarray(z).shape[0])

        B = z.shape[0]
        d_mu = np.zeros(self.dim)
        d_Sig = np.zeros((self.dim, self.dim))
        for b in range(B):
            dmu_b, dSig_b = gsm_sample_update(
                self.mu, self.Sigma, z[b], g[b], xp=np
            )
            d_mu = d_mu + dmu_b
            d_Sig = d_Sig + dSig_b
        d_mu = d_mu / B
        d_Sig = d_Sig / B

        self.mu = self.mu + d_mu
        Sigma_new = symmetrize(self.Sigma + d_Sig)
        if self.check_spd:
            Sigma_new = ensure_spd(Sigma_new, jitter=self.jitter)
        self.Sigma = np.asarray(Sigma_new)

        self.t += 1
        self._record()
        return self.mu, self.Sigma

    def run(self, T: int):
        """Run ``T`` iterations and return a :class:`GSMResult`."""
        for _ in range(int(T)):
            self.step()
        return self.result()

    def result(self) -> GSMResult:
        return GSMResult(
            mu=self.mu.copy(),
            Sigma=self.Sigma.copy(),
            n_iter=int(self.t),
            grad_evals=int(self.grad_evals),
            batch_size=int(self.batch_size),
            mu_history=list(self.mu_history) if self.mu_history else None,
            Sigma_history=list(self.Sigma_history) if self.Sigma_history else None,
            name=self.name,
        )

    def fit(self, T: int) -> GSMResult:
        return self.run(T)


# ---------------------------------------------------------------------------
# functional interface
# ---------------------------------------------------------------------------
def gsm_fit(
    mu0,
    Sigma0,
    score_fn: Optional[Callable] = None,
    target: Any = None,
    T: int = 1000,
    batch_size: int = 2,
    seed: int = 0,
    track_history: bool = False,
    history_every: int = 1,
    **kwargs: Any,
) -> GSMResult:
    """Functional wrapper: run GSM for ``T`` iterations."""
    opt = GSM(
        mu0,
        Sigma0,
        score_fn=score_fn,
        target=target,
        batch_size=batch_size,
        seed=seed,
        track_history=track_history,
        history_every=history_every,
        **kwargs,
    )
    return opt.run(T)
