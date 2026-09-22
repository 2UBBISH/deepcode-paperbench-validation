"""Batch and Match (BaM): black-box VI with a score-based divergence.

This module implements Algorithm 1 of

    "Batch and Match: Black-Box Variational Inference with a Score-Based
    Divergence"

Each iteration alternates between

* the **batch step** (Section 3.1, Appendix C.1): draw ``z_b ~ N(mu_t, Sigma_t)``,
  evaluate the target scores ``g_b = grad log p(z_b)``, and form the batch
  statistics

      z_bar = (1/B) sum_b z_b,        C     = (1/B) sum_b (z_b - z_bar)(z_b - z_bar)^T
      g_bar = (1/B) sum_b g_b,        Gamma = (1/B) sum_b (g_b - g_bar)(g_b - g_bar)^T

  which give the empirical score-based divergence (Appendix C.1, eq. (98))

      D_hat(q_t; p) = tr(Gamma Sigma) + tr(C Sigma^{-1})
                      + ||mu - z_bar - Sigma g_bar||^2_{Sigma^{-1}} + const.

* the **match step** (Section 3.1, Appendix C.2): minimize the regularized
  objective ``L^BaM(q) = D_hat(q_t; p) + (2/lambda_t) KL(q_t; q)`` in closed
  form.  The optimal covariance solves the quadratic matrix equation

      Sigma_{t+1} U Sigma_{t+1} + Sigma_{t+1} = V                (eq. (9)/(11))

  with (eq. (8)/(10))

      U = lambda_t Gamma + (lambda_t / (1 + lambda_t)) g_bar g_bar^T
      V = Sigma_t + lambda_t C + (lambda_t / (1 + lambda_t)) (mu_t - z_bar)(mu_t - z_bar)^T

  whose symmetric positive definite solution is (eq. (12))

      Sigma_{t+1} = 2 V [ I + (I + 4 U V)^{1/2} ]^{-1}

  and the optimal mean is (eq. (13))

      mu_{t+1} = (1 / (1 + lambda_t)) mu_t
                 + (lambda_t / (1 + lambda_t)) (Sigma_{t+1} g_bar + z_bar)

  The mean update depends on the *new* covariance, so the two updates must be
  performed in the order shown above.

Limiting cases (Section 3.1):

* ``lambda_t -> 0``:   ``Sigma_{t+1} = Sigma_t`` and ``mu_{t+1} = mu_t`` (no movement);
* ``B = 1`` and ``lambda_t -> inf``: BaM reduces to Gaussian score matching (GSM);
* ``B -> inf`` then ``lambda_0 -> inf``: one-step convergence for Gaussian targets
  (Corollary D.5).

The dense covariance update costs ``O(D^3)``.  When the batch size is small
(``B << D''), ``U = Q Q^T`` is low rank and Lemma B.3 gives an ``O(D^2 B + B^3)``
update, which is exposed here as ``low_rank=True``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Union

import numpy as np

from .matrix_equations import (
    _HAS_JAX,
    array_namespace,
    ensure_spd,
    inverse_spd,
    solve_quadratic_matrix_equation,
    solve_quadratic_matrix_equation_low_rank,
    symmetrize,
)
from .vi_base import standard_normal

if _HAS_JAX:  # pragma: no cover - optional dependency
    import jax
    import jax.numpy as jnp
else:  # pragma: no cover
    jax = None
    jnp = None


__all__ = [
    "BatchStatistics",
    "MatchStepResult",
    "BaMResult",
    "batch_statistics",
    "low_rank_factor",
    "outer",
    "bam_match_step",
    "match_step",
    "empirical_divergence",
    "resolve_lambda",
    "BaM",
    "bam_fit",
]


_NEGATIVE_EIG_TOL = 1e-12


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _is_jax_namespace(xp) -> bool:
    return getattr(xp, "__name__", "").startswith("jax")


def _eye(A, xp):
    n = A.shape[0]
    return xp.eye(n, dtype=A.dtype)


def outer(a, b, xp):
    """Outer product ``a b^T`` (works for numpy and jax.numpy)."""
    return xp.reshape(a, (-1, 1)) * xp.reshape(b, (1, -1))


def safe_cholesky(A, xp=None, jitter: float = 1e-12, max_tries: int = 10):
    """Cholesky factor of a symmetric matrix, adding jitter if necessary."""
    if xp is None:
        xp = array_namespace(A)
    A = symmetrize(A)
    shift = jitter * float(np.max(np.abs(np.asarray(A)))) + jitter
    if shift <= 0.0:
        shift = jitter
    B = A
    for _ in range(max_tries):
        try:
            return xp.linalg.cholesky(B)
        except Exception:  # pragma: no cover - backend specific
            B = A + shift * _eye(A, xp)
            shift *= 10.0
    return xp.linalg.cholesky(ensure_spd(A, jitter=jitter))


def _finalize_covariance(Sigma, xp, jitter: float = 1e-12, check_spd: bool = True):
    """Symmetrize (and if needed project) an updated covariance matrix."""
    Sigma = symmetrize(Sigma)
    if not check_spd:
        return Sigma
    try:
        min_eig = float(xp.min(xp.linalg.eigvalsh(Sigma)))
    except Exception:  # pragma: no cover
        return Sigma
    if min_eig <= jitter:
        Sigma = ensure_spd(Sigma, jitter=max(jitter, 1e-10), min_eig=max(jitter, 1e-8))
    return Sigma


# ---------------------------------------------------------------------------
# batch step
# ---------------------------------------------------------------------------
@dataclass
class BatchStatistics:
    """Batch statistics of Algorithm 1, step 5 (Appendix C.1)."""

    z_bar: Any
    g_bar: Any
    C: Any
    Gamma: Any
    batch_size: int

    @property
    def dim(self) -> int:
        return int(self.z_bar.shape[0])

    def as_dict(self):
        return {
            "z_bar": self.z_bar,
            "g_bar": self.g_bar,
            "C": self.C,
            "Gamma": self.Gamma,
            "B": self.batch_size,
        }


def batch_statistics(z, g, xp=None) -> BatchStatistics:
    """Compute ``z_bar, g_bar, C, Gamma`` from samples ``z`` and scores ``g``.

    Parameters
    ----------
    z : array of shape ``(B, D)``
        Samples from the current variational distribution ``q_t``.
    g : array of shape ``(B, D)``
        Target scores ``grad log p(z_b)`` at those samples.

    Returns
    -------
    BatchStatistics
    """
    if xp is None:
        xp = array_namespace(z, g)
    z = xp.asarray(z)
    g = xp.asarray(g)
    B = int(z.shape[0])
    z_bar = xp.mean(z, axis=0)
    g_bar = xp.mean(g, axis=0)
    dz = z - z_bar
    dg = g - g_bar
    C = symmetrize(xp.einsum("bi,bj->ij", dz, dz) / B)
    Gamma = symmetrize(xp.einsum("bi,bj->ij", dg, dg) / B)
    return BatchStatistics(z_bar=z_bar, g_bar=g_bar, C=C, Gamma=Gamma, batch_size=B)


def low_rank_factor(g, g_bar, lam: float, xp=None):
    """Return ``Q`` with ``Q Q^T = lambda_t Gamma + (lambda_t/(1+lambda_t)) g_bar g_bar^T``.

    ``Gamma = (1/B) sum_b (g_b - g_bar)(g_b - g_bar)^T`` has rank at most
    ``B - 1``, so ``U`` is low rank and Lemma B.3 applies.  The returned factor
    has ``B + 1`` columns.
    """
    if xp is None:
        xp = array_namespace(g)
    B = int(g.shape[0])
    dg = g - g_bar
    # ``U = (lam / B) sum_b (g_b - g_bar)(g_b - g_bar)^T + c g_bar g_bar^T``
    Q1 = xp.sqrt(xp.asarray(lam)) / xp.sqrt(xp.asarray(float(B))) * dg.T
    c = lam / (1.0 + lam)
    Q2 = xp.sqrt(xp.asarray(c)) * xp.reshape(g_bar, (-1, 1))
    return xp.concatenate([Q1, Q2], axis=1)


# ---------------------------------------------------------------------------
# match step
# ---------------------------------------------------------------------------
@dataclass
class MatchStepResult:
    """Result of the closed-form match step."""

    mu: Any
    Sigma: Any
    U: Any
    V: Any
    lam: float


def bam_match_step(
    mu_t,
    Sigma_t,
    z=None,
    g=None,
    lam: float = 1.0,
    xp=None,
    stats: Optional[BatchStatistics] = None,
    low_rank: Optional[bool] = None,
    jitter: float = 0.0,
    check_spd: bool = True,
    return_result: bool = False,
):
    """One closed-form match step of BaM (Section 3.1, eqs. (8)-(13)).

    Parameters
    ----------
    mu_t, Sigma_t : array, array
        Current variational mean and covariance (``Sigma_t`` must be SPD).
    z, g : array of shape ``(B, D)``, optional
        Batch of samples and target scores.  May be replaced by ``stats``.
    lam : float
        Inverse regularization parameter ``lambda_t``.
    low_rank : bool, optional
        Use the ``O(D^2 B + B^3)`` low-rank solver of Lemma B.3.  If ``None``,
        it is selected automatically when ``B + 1 < D``.
    return_result : bool
        If ``True``, also return the ``U``/``V`` matrices and ``lambda_t``.

    Returns
    -------
    mu_next, Sigma_next (or a :class:`MatchStepResult`).
    """
    if xp is None:
        xp = array_namespace(mu_t, Sigma_t, *( () if z is None else (z,)))
    mu_t = xp.asarray(mu_t)
    Sigma_t = symmetrize(xp.asarray(Sigma_t))
    lam = float(lam)

    if stats is None:
        if z is None or g is None:
            raise ValueError("either `stats` or both `z` and `g` must be provided")
        stats = batch_statistics(z, g, xp)

    z_bar = stats.z_bar
    g_bar = stats.g_bar
    c = lam / (1.0 + lam)  # lambda_t / (1 + lambda_t)

    U = lam * stats.Gamma + c * outer(g_bar, g_bar, xp)
    diff = mu_t - z_bar
    V = Sigma_t + lam * stats.C + c * outer(diff, diff, xp)
    U = symmetrize(U)
    V = symmetrize(V)

    if low_rank is None:
        low_rank = (stats.batch_size + 1) < int(mu_t.shape[0])

    if low_rank:
        Q = low_rank_factor(g, g_bar, lam, xp)
        Sigma_next = solve_quadratic_matrix_equation_low_rank(V, Q, jitter=jitter)
    else:
        Sigma_next = solve_quadratic_matrix_equation(U, V, jitter=jitter)

    Sigma_next = _finalize_covariance(Sigma_next, xp, check_spd=check_spd)

    # mean update uses the *updated* covariance (eq. (13))
    mu_next = (1.0 / (1.0 + lam)) * mu_t + c * (Sigma_next @ g_bar + z_bar)

    if return_result:
        return MatchStepResult(mu=mu_next, Sigma=Sigma_next, U=U, V=V, lam=lam)
    return mu_next, Sigma_next


# alias used by tests / experiments
match_step = bam_match_step


def empirical_divergence(mu, Sigma, z, g, xp=None, stats: Optional[BatchStatistics] = None):
    """Empirical score-based divergence ``D_hat_{q_t}(q; p)`` (Appendix C.1, eq. (98)).

    ``tr(Gamma Sigma) + tr(C Sigma^{-1}) + ||mu - z_bar - Sigma g_bar||^2_{Sigma^{-1}}``
    (additive constants omitted).
    """
    if xp is None:
        xp = array_namespace(mu, Sigma)
    if stats is None:
        stats = batch_statistics(z, g, xp)
    Sigma = symmetrize(Sigma)
    Sigma_inv = inverse_spd(Sigma)
    term1 = xp.sum(stats.Gamma * Sigma)
    term2 = xp.sum(stats.C * Sigma_inv)
    resid_vec = mu - stats.z_bar - Sigma @ stats.g_bar
    term3 = float(resid_vec @ (Sigma_inv @ resid_vec))
    return term1 + term2 + term3


# ---------------------------------------------------------------------------
# learning-rate handling
# ---------------------------------------------------------------------------
def resolve_lambda(
    lam,
    t: int,
    dim: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> float:
    """Evaluate ``lambda_t`` from a float, callable, schedule name, or sequence.

    Supported forms
    ---------------
    * ``None``      -> ``B * D`` (the constant schedule used for Gaussian targets);
    * a number      -> that number;
    * a callable    -> ``lam(t)`` (or ``lam.value(t)``);
    * a string      -> a named schedule from :mod:`bam.learning_rate`;
    * a sequence    -> ``lam[t]``.
    """
    if lam is None:
        if batch_size is None or dim is None:
            return 1.0
        return float(batch_size) * float(dim)
    if isinstance(lam, str):
        from .learning_rate import make_schedule

        schedule = make_schedule(lam, batch_size=batch_size, dim=dim)
        return float(schedule(t))
    if isinstance(lam, (int, float, np.floating, np.integer)):
        return float(lam)
    if callable(lam):
        return float(lam(t))
    if hasattr(lam, "value") and callable(getattr(lam, "value")):
        return float(lam.value(t))
    try:
        return float(lam[t])
    except Exception as exc:  # pragma: no cover
        raise TypeError(f"unsupported learning-rate specification: {type(lam)!r}") from exc


# ---------------------------------------------------------------------------
# the BaM algorithm
# ---------------------------------------------------------------------------
@dataclass
class BaMResult:
    """Output of :meth:`BaM.run`."""

    mu: Any
    Sigma: Any
    mu_history: List[Any] = field(default_factory=list)
    lam_history: List[float] = field(default_factory=list)
    Sigma_history: List[Any] = field(default_factory=list)
    grad_evals: int = 0
    n_iter: int = 0


class BaM:
    """Batch and Match variational inference (Algorithm 1).

    Parameters
    ----------
    mu0, Sigma0 : array, array
        Initial variational mean and covariance.
    score_fn : callable
        Target score ``s(z) = grad log p(z)``.  May accept an ``(B, D)`` array
        (vectorized) or a single ``(D,)`` array (it is then applied per sample).
    batch_size : int
        Number of samples per iteration ``B``.
    lam : float | callable | str | None
        Inverse regularization parameter.  ``None`` means the default constant
        value ``B * D``.  Strings are resolved by :mod:`bam.learning_rate`
        (e.g. ``"constant"``, ``"decay"``, ``"sqrt_decay"``, ``"decay_b"``).
    rng : numpy.random.Generator, optional
        Source of randomness for the NumPy backend.
    key : jax.random.PRNGKey, optional
        Source of randomness for the JAX backend.
    low_rank : bool, optional
        Force (or forbid) the low-rank covariance update.
    check_spd : bool
        Re-project the updated covariance if it loses positive definiteness.
    """

    def __init__(
        self,
        mu0,
        Sigma0,
        score_fn: Callable,
        batch_size: int = 20,
        lam: Union[float, Callable, str, None] = None,
        rng=None,
        key=None,
        xp=None,
        low_rank: Optional[bool] = None,
        jitter: float = 0.0,
        dtype=None,
        seed: int = 0,
        track_history: bool = False,
        history_every: int = 1,
        callback: Optional[Callable] = None,
        check_spd: bool = True,
    ) -> None:
        if xp is None:
            xp = array_namespace(mu0, Sigma0)
        self.xp = xp
        self.mu = xp.asarray(mu0, dtype=dtype)
        self.Sigma = symmetrize(xp.asarray(Sigma0, dtype=dtype))
        self.score_fn = score_fn
        self.batch_size = int(batch_size)
        self.lam_spec = lam
        self.low_rank = low_rank
        self.jitter = float(jitter)
        self.dtype = dtype
        self.seed = int(seed)
        self.track_history = bool(track_history)
        self.history_every = max(1, int(history_every))
        self.callback = callback
        self.check_spd = bool(check_spd)

        self.t = 0
        self.grad_evals = 0
        self._is_jax = _is_jax_namespace(xp)
        if self._is_jax:
            if key is None:
                key = jax.random.PRNGKey(self.seed)
            self.key = key
            self.rng = None
        else:
            if rng is None:
                rng = np.random.default_rng(self.seed)
            self.rng = rng
            self.key = None

        self.mu_history: List[Any] = [self.mu]
        self.lam_history: List[float] = []
        self.Sigma_history: List[Any] = [self.Sigma] if self.track_history else []
        self.stats: Optional[BatchStatistics] = None
        self.last_result: Optional[MatchStepResult] = None

    # -- basics ---------------------------------------------------------
    @property
    def dim(self) -> int:
        return int(self.mu.shape[0])

    def lam_t(self, t: Optional[int] = None) -> float:
        if t is None:
            t = self.t
        return resolve_lambda(self.lam_spec, t, dim=self.dim, batch_size=self.batch_size)

    # -- sampling -------------------------------------------------------
    def sample(self, batch_size: Optional[int] = None):
        """Draw ``z_b ~ N(mu_t, Sigma_t)`` via reparameterization."""
        xp = self.xp
        B = int(self.batch_size if batch_size is None else batch_size)
        D = self.dim
        L = safe_cholesky(self.Sigma, xp, jitter=max(self.jitter, 1e-12))
        if self._is_jax:
            self.key, subkey = jax.random.split(self.key)
            eps = standard_normal((B, D), xp=xp, key=subkey, dtype=self.dtype)
        else:
            eps = standard_normal((B, D), xp=xp, rng=self.rng, dtype=self.dtype)
        # z = mu + eps L^T  (reparameterization)
        return self.mu + eps @ L.T

    # -- scores ---------------------------------------------------------
    def evaluate_scores(self, z):
        """Evaluate ``g_b = grad log p(z_b)``, vectorized if possible."""
        xp = self.xp
        if self._is_jax:
            try:
                g = self.score_fn(z)
            except Exception:  # pragma: no cover
                g = jax.vmap(self.score_fn)(z)
        else:
            try:
                g = self.score_fn(z)
            except Exception:  # pragma: no cover
                g = np.stack([np.asarray(self.score_fn(zb)) for zb in np.asarray(z)])
        g = xp.asarray(g)
        if g.shape != z.shape:
            rows = [xp.asarray(self.score_fn(zb)) for zb in z]
            g = xp.stack(rows)
        return g

    # -- one iteration --------------------------------------------------
    def step(self, lam: Optional[float] = None):
        """Perform one BaM iteration; returns ``(mu_next, Sigma_next)``."""
        if lam is None:
            lam = self.lam_t(self.t)
        z = self.sample()
        g = self.evaluate_scores(z)
        self.stats = batch_statistics(z, g, self.xp)
        result = bam_match_step(
            self.mu,
            self.Sigma,
            z,
            g,
            lam=lam,
            xp=self.xp,
            stats=self.stats,
            low_rank=self.low_rank,
            jitter=self.jitter,
            check_spd=self.check_spd,
            return_result=True,
        )
        self.last_result = result
        self.mu = result.mu
        self.Sigma = _finalize_covariance(result.Sigma, self.xp, check_spd=self.check_spd)
        self.t += 1
        self.grad_evals += self.batch_size
        self.lam_history.append(float(lam))
        self.mu_history.append(self.mu)
        if self.track_history and (self.t % self.history_every == 0):
            self.Sigma_history.append(self.Sigma)
        if self.callback is not None:
            self.callback(self.t, self.mu, self.Sigma)
        return self.mu, self.Sigma

    def run(self, T: int) -> BaMResult:
        """Run ``T`` iterations of Algorithm 1."""
        for _ in range(int(T)):
            self.step()
        return BaMResult(
            mu=self.mu,
            Sigma=self.Sigma,
            mu_history=self.mu_history,
            lam_history=self.lam_history,
            Sigma_history=self.Sigma_history,
            grad_evals=self.grad_evals,
            n_iter=self.t,
        )


def bam_fit(
    mu0,
    Sigma0,
    score_fn: Callable,
    T: int,
    batch_size: int = 20,
    lam: Union[float, Callable, str, None] = None,
    rng=None,
    key=None,
    xp=None,
    low_rank: Optional[bool] = None,
    jitter: float = 0.0,
    dtype=None,
    seed: int = 0,
    track_history: bool = False,
    history_every: int = 1,
    callback: Optional[Callable] = None,
    check_spd: bool = True,
) -> BaMResult:
    """Convenience functional wrapper around :class:`BaM`."""
    algo = BaM(
        mu0,
        Sigma0,
        score_fn,
        batch_size=batch_size,
        lam=lam,
        rng=rng,
        key=key,
        xp=xp,
        low_rank=low_rank,
        jitter=jitter,
        dtype=dtype,
        seed=seed,
        track_history=track_history,
        history_every=history_every,
        callback=callback,
        check_spd=check_spd,
    )
    return algo.run(T)
