"""ADVI baseline (Algorithm 2 of the paper).

The paper (Appendix E.1, Algorithm 2) describes the ADVI baseline used in all
experiments as follows::

    Input: Iterations T, batch size B, unnormalized target p~, learning rate
           lambda_t > 0, initial variational mean mu_0 in R^D, initial
           variational covariance Sigma_0 in S^D_{++}
    for t = 0, ..., T-1 do
        Sample z_1, ..., z_B ~ q_t = N(mu_t, Sigma_t)
        Compute a stochastic estimate of the (negative) ELBO
            L_ELBO(z_{1:B}) = -sum_b [log p~(z_b) - log q_t(z_b)]
        Update the variational parameters w_t := (mu_t, Sigma_t) with gradient
            w_{t+1} = w_t - lambda_t grad_w L_ELBO(z_{1:B})
        # Our implementation uses the ADAM update.
    end for
    Output: variational parameters mu_T, Sigma_T

This module provides

* :class:`GradientVI` -- a generic, full-covariance reparameterized Gaussian
  VI engine that performs ADAM updates of ``(mu, L)`` where ``Sigma = L L^T``
  is the Cholesky factorisation of the variational covariance.  Several losses
  are available: the (negative) ELBO (Algorithm 2), the score-based divergence
  D(q ; p) of Definition A.2 / eq. (34) and the (unweighted) Fisher divergence
  E_q[||grad log q - grad log p||^2_I].
* :class:`ADVI` -- the exact baseline of Algorithm 2 (negative ELBO + ADAM).
* :class:`GradientVIResult` -- container with the variational parameters and
  the diagnostics (per-iteration losses, number of gradient evaluations, ...).

Estimator conventions (documented defaults, since the paper only spells out the
*objective*, not the estimator)

* ELBO: the standard *reparameterised* (pathwise) gradient is used.  With
  ``z_b = mu + L eps_b`` one has ``log q(z_b) = const - log|L| - 0.5||eps_b||^2``
  and therefore (for the negative ELBO ``L = -ELBO``)

      grad_mu L = -(1/B) sum_b s_b
      grad_L  L = -(1/B) sum_b s_b eps_b^T - L^{-T}

  where ``s_b = grad_z log p~(z_b)``.  Only first-order scores of the target are
  required, which is the interface exposed by every target in ``bam_repro``.
* Score-based divergence D(q ; p) and Fisher divergence: the empirical losses
  are, with ``g_b = Sigma^{-1}(z_b - mu)`` and ``h_b = s_b + g_b``,

      D_hat = (1/B) sum_b h_b^T Sigma h_b          (weighted by Cov(q))
      F_hat = (1/B) sum_b h_b^T h_b                (weighted by I)

  Their gradients w.r.t. ``(mu, Sigma)`` are computed *holding the sampled
  scores fixed* (the standard Hessian-free estimator; the pathwise term
  ``d s_b / d z_b`` is the Hessian of the target log-density and is omitted):

      grad_mu D_hat = -(2/B) sum_b h_b
      grad_Sigma D_hat = (1/B) sum_b (s_b s_b^T - g_b g_b^T)
      grad_mu F_hat = -(2/B) sum_b Sigma^{-1} h_b
      grad_Sigma F_hat = -(1/B) sum_b (g_b h_b^T + h_b g_b^T)

  Both losses vanish iff ``grad log q = grad log p`` on average, so the fixed
  point is the correct one for the score-based divergence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..bam.matrix_equations import (
    _HAS_JAX,
    array_namespace,
    ensure_spd,
    inverse_spd,
    symmetrize,
)
from ..bam.vi_base import gaussian_log_density, standard_normal

__all__ = [
    "GradientVI",
    "GradientVIResult",
    "ADVI",
    "ADVIResult",
    "advi_fit",
    "LOSSES",
]


LOSSES: Tuple[str, ...] = (
    "elbo",
    "score",
    "score_divergence",
    "fisher",
    "fisher_divergence",
)

# ADAM hyper-parameters (Kingma & Ba defaults; the paper only specifies that
# ADAM is used, so the standard values are taken).
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8

_MIN_DIAG = 1e-12


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _cholesky_lower(Sigma: np.ndarray, jitter: float = 0.0) -> np.ndarray:
    """Lower-triangular Cholesky factor of a (symmetric) positive-definite A.

    A small jitter is added adaptively if the plain factorisation fails, and a
    symmetric eigendecomposition is used as a last resort so that the
    variational covariance always stays positive definite.
    """
    xp = array_namespace(Sigma)
    Sigma = symmetrize(xp.asarray(Sigma))
    D = Sigma.shape[0]
    eye = xp.eye(D, dtype=Sigma.dtype)
    try:
        return xp.linalg.cholesky(Sigma + jitter * eye)
    except Exception:
        pass
    for scale in (1e-10, 1e-8, 1e-6, 1e-4):
        try:
            return xp.linalg.cholesky(Sigma + scale * eye)
        except Exception:
            continue
    # eigen fallback
    w, V = xp.linalg.eigh(Sigma)
    w = xp.clip(w, 1e-10, None)
    Sig_pd = (V * w) @ V.T
    Sig_pd = symmetrize(Sig_pd)
    return xp.linalg.cholesky(Sig_pd + 1e-12 * eye)


def _inverse_from_chol(L: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(Sigma^{-1}, L^{-T})`` from a lower-triangular Cholesky ``L``."""
    xp = array_namespace(L)
    D = L.shape[0]
    Linv = xp.linalg.solve(L, xp.eye(D, dtype=L.dtype))
    LinvT = Linv.T
    Sig_inv = LinvT @ Linv
    return symmetrize(Sig_inv), LinvT


def _resolve_score_fn(
    score_fn: Optional[Callable] = None,
    target: Any = None,
) -> Callable:
    """Return a callable ``z -> grad_z log p(z)`` from a function or a target."""
    if score_fn is not None:
        return score_fn
    if target is None:
        raise ValueError("one of `score_fn` or `target` must be provided")
    for attr in ("score", "grad_log_prob", "gradient_log_prob"):
        fn = getattr(target, attr, None)
        if callable(fn):
            return fn
    raise ValueError(
        "target does not expose a score function (expected `.score` or "
        "`.grad_log_prob`)"
    )


def _resolve_log_prob_fn(log_prob_fn: Optional[Callable], target: Any) -> Optional[Callable]:
    if log_prob_fn is not None:
        return log_prob_fn
    if target is None:
        return None
    for attr in ("log_prob", "log_density", "logp", "log_joint"):
        fn = getattr(target, attr, None)
        if callable(fn):
            return fn
    return None


def _as_lr_schedule(lr: Union[float, Callable[[int], float]]):
    if callable(lr):
        return lr
    value = float(lr)

    def _fn(t: int) -> float:  # noqa: ARG001 - constant schedule
        return value

    return _fn


def _sym_grad_to_chol(L: np.ndarray, G_Sigma: np.ndarray, mask_lower: bool = True) -> np.ndarray:
    """Convert ``grad_Sigma`` (symmetric) to ``grad_L`` for ``Sigma = L L^T``.

    For symmetric ``G`` one has ``tr(G dSigma) = 2 tr(L^T G dL)`` and, since only
    the lower-triangular entries of ``L`` are free parameters, the gradient with
    respect to those parameters is the lower triangle of ``2 L^T G``.
    """
    G_L = 2.0 * (L.T @ symmetrize(G_Sigma))
    if mask_lower:
        G_L = np.tril(G_L)
    return G_L


def _mask_lower(A: np.ndarray) -> np.ndarray:
    return np.tril(A)


# ----------------------------------------------------------------------------
# result container
# ----------------------------------------------------------------------------
@dataclass
class GradientVIResult:
    """Outcome of a gradient-based VI run (ADVI, Score-ADVI, Fisher-ADVI)."""

    mu: np.ndarray
    Sigma: np.ndarray
    L: Optional[np.ndarray] = None
    n_iter: int = 0
    grad_evals: int = 0
    batch_size: int = 0
    losses: List[float] = field(default_factory=list)
    mu_history: List[np.ndarray] = field(default_factory=list)
    Sigma_history: List[np.ndarray] = field(default_factory=list)
    lam_history: List[float] = field(default_factory=list)
    name: str = "advi"

    @property
    def dim(self) -> int:
        return int(np.asarray(self.mu).shape[0])

    @property
    def grad_evaluations(self) -> np.ndarray:
        """Number of gradient (score) evaluations after each iteration."""
        return np.arange(1, self.n_iter + 1) * int(self.batch_size)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "mu": np.asarray(self.mu),
            "Sigma": np.asarray(self.Sigma),
            "n_iter": int(self.n_iter),
            "grad_evals": int(self.grad_evals),
            "batch_size": int(self.batch_size),
            "losses": list(self.losses),
            "mu_history": list(self.mu_history),
            "Sigma_history": list(self.Sigma_history),
            "lam_history": list(self.lam_history),
        }


# backwards/forwards compatible alias
ADVIResult = GradientVIResult


# ----------------------------------------------------------------------------
# generic gradient VI engine
# ----------------------------------------------------------------------------
class GradientVI:
    """Full-covariance Gaussian VI with ADAM updates (gradient-based baselines).

    Parameters
    ----------
    mu0, Sigma0 : array-like
        Initial variational mean and covariance (``Sigma_0`` is projected onto
        the positive-definite cone when necessary).
    score_fn : callable, optional
        ``z (B, D) -> grad_z log p~(z) (B, D)``.  Alternatively provide
        ``target`` exposing ``.score`` / ``.grad_log_prob``.
    target : object, optional
        Target distribution object (used to fetch the score and log-density).
    log_prob_fn : callable, optional
        ``z (B, D) -> log p~(z) (B,)``; used for reporting the loss value and
        for the ELBO objective.  Not needed for the gradients.
    loss : str
        One of ``"elbo"``, ``"score"``/``"score_divergence"``,
        ``"fisher"``/``"fisher_divergence"``.
    batch_size : int
        Number of samples/gradient evaluations per iteration.
    learning_rate : float or callable
        ADAM step size ``lambda_t`` (the ``lambda_t`` of Algorithm 2).  A
        callable is evaluated at the iteration index ``t`` (0-based).
    """

    loss_kind: str = "elbo"
    name: str = "advi"

    def __init__(
        self,
        mu0: np.ndarray,
        Sigma0: np.ndarray,
        score_fn: Optional[Callable] = None,
        target: Any = None,
        log_prob_fn: Optional[Callable] = None,
        loss: Optional[str] = None,
        batch_size: int = 2,
        learning_rate: Union[float, Callable[[int], float]] = 0.01,
        beta1: float = ADAM_BETA1,
        beta2: float = ADAM_BETA2,
        adam_eps: float = ADAM_EPS,
        seed: int = 0,
        rng: Optional[np.random.Generator] = None,
        dtype: Any = np.float64,
        track_history: bool = False,
        history_every: int = 1,
        jitter: float = 0.0,
        check_spd: bool = True,
    ) -> None:
        self.mu = np.asarray(mu0, dtype=dtype).reshape(-1).copy()
        Sigma0 = np.asarray(Sigma0, dtype=dtype)
        if check_spd:
            Sigma0 = np.asarray(ensure_spd(Sigma0))
        self.L = _cholesky_lower(np.asarray(Sigma0, dtype=dtype), jitter=jitter)

        self.dim = int(self.mu.shape[0])
        self.score_fn = _resolve_score_fn(score_fn, target)
        self.log_prob_fn = _resolve_log_prob_fn(log_prob_fn, target)
        self.target = target

        self.loss_kind = str(loss if loss is not None else self.loss_kind).lower()
        if self.loss_kind in ("score_divergence", "weighted_fisher", "df"):
            self.loss_kind = "score"
        if self.loss_kind in ("fisher_divergence",):
            self.loss_kind = "fisher"
        if self.loss_kind not in ("elbo", "score", "fisher"):
            raise ValueError(
                f"unknown loss '{loss}'; expected one of {LOSSES}"
            )

        self.batch_size = int(batch_size)
        self._lr_fn = _as_lr_schedule(learning_rate)
        self.learning_rate = float(learning_rate) if not callable(learning_rate) else None
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.adam_eps = float(adam_eps)

        self.rng = rng if rng is not None else np.random.default_rng(seed)
        self.seed = int(seed)
        self.dtype = dtype
        self.jitter = float(jitter)
        self.check_spd = bool(check_spd)

        self.t = 0
        self.grad_evals = 0
        self.track_history = bool(track_history)
        self.history_every = max(1, int(history_every))
        self.losses: List[float] = []
        self.mu_history: List[np.ndarray] = []
        self.Sigma_history: List[np.ndarray] = []
        self.lam_history: List[float] = []

        # ADAM state
        self._m_mu = np.zeros_like(self.mu)
        self._v_mu = np.zeros_like(self.mu)
        self._m_L = np.zeros_like(self.L)
        self._v_L = np.zeros_like(self.L)
        self._adam_steps = 0

    # -- state -------------------------------------------------------------
    @property
    def mean(self) -> np.ndarray:
        return self.mu

    @property
    def covariance(self) -> np.ndarray:
        return symmetrize(self.L @ self.L.T)

    @property
    def Sigma(self) -> np.ndarray:
        return self.covariance

    def lam_t(self, t: Optional[int] = None) -> float:
        """Learning rate ``lambda_t`` used at iteration ``t``."""
        if t is None:
            t = self.t
        return float(self._lr_fn(int(t)))

    # -- sampling ----------------------------------------------------------
    def sample(self, batch_size: Optional[int] = None) -> np.ndarray:
        """Reparameterised samples ``z = mu + L eps``."""
        B = int(self.batch_size if batch_size is None else batch_size)
        eps = standard_normal((B, self.dim), xp=np, rng=self.rng, dtype=self.dtype)
        return self.mu[None, :] + eps @ self.L.T

    def evaluate_scores(self, z: np.ndarray) -> np.ndarray:
        """Target scores ``s_b = grad log p~(z_b)``."""
        s = np.asarray(self.score_fn(z), dtype=self.dtype)
        if s.ndim == 1:
            s = s.reshape(1, -1)
        return s

    def evaluate_log_prob(self, z: np.ndarray) -> Optional[np.ndarray]:
        if self.log_prob_fn is None:
            return None
        lp = np.asarray(self.log_prob_fn(z), dtype=self.dtype)
        return lp.reshape(-1)

    def variational_log_prob(self, z: np.ndarray) -> np.ndarray:
        return gaussian_log_density(z, self.mu, self.covariance)

    # -- gradients ---------------------------------------------------------
    def gradients(
        self,
        z: np.ndarray,
        eps: np.ndarray,
        s: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return ``(grad_mu, grad_L, loss_value)`` for the current state."""
        if self.loss_kind == "elbo":
            return self._grads_elbo(z, eps, s)
        if self.loss_kind == "score":
            return self._grads_score_divergence(z, s)
        return self._grads_fisher(z, s)

    def _grads_elbo(
        self, z: np.ndarray, eps: np.ndarray, s: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Pathwise (reparameterised) gradient of the negative ELBO."""
        B = int(z.shape[0])
        L = self.L
        _, LinvT = _inverse_from_chol(L)
        grad_mu = -s.mean(axis=0)
        grad_L = _mask_lower(-(s.T @ eps) / B - LinvT)

        # loss value (monitoring only)
        lp = self.evaluate_log_prob(z)
        lq = self.variational_log_prob(z)
        if lp is None:
            loss = float("nan")
        else:
            loss = float(-(lp - lq).mean())
        return grad_mu, grad_L, loss

    def _grads_score_divergence(
        self, z: np.ndarray, s: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Gradients of the empirical score-based divergence D_hat."""
        B = int(z.shape[0])
        Sig_inv, _ = _inverse_from_chol(self.L)
        g = (z - self.mu[None, :]) @ Sig_inv  # rows g_b = Sigma^{-1}(z_b - mu)
        h = s + g
        grad_mu = -2.0 * h.mean(axis=0)
        G_Sigma = (s.T @ s - g.T @ g) / B
        grad_L = _sym_grad_to_chol(self.L, G_Sigma)
        # D_hat = (1/B) sum_b h_b^T Sigma h_b
        loss = float(np.mean(np.einsum("bi,ij,bj->b", h, self.covariance, h)))
        return grad_mu, grad_L, loss

    def _grads_fisher(
        self, z: np.ndarray, s: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Gradients of the empirical (unweighted) Fisher divergence."""
        B = int(z.shape[0])
        Sig_inv, _ = _inverse_from_chol(self.L)
        g = (z - self.mu[None, :]) @ Sig_inv
        h = s + g
        Sig_inv_h = h @ Sig_inv
        grad_mu = -2.0 * Sig_inv_h.mean(axis=0)
        G_Sigma = -(g.T @ h + h.T @ g) / B
        grad_L = _sym_grad_to_chol(self.L, G_Sigma)
        loss = float(np.mean(np.sum(h * h, axis=1)))
        return grad_mu, grad_L, loss

    # -- optimizer ---------------------------------------------------------
    def _adam_update(self, grad_mu: np.ndarray, grad_L: np.ndarray, lr: float) -> None:
        self._adam_steps += 1
        t = self._adam_steps
        b1, b2 = self.beta1, self.beta2
        self._m_mu = b1 * self._m_mu + (1.0 - b1) * grad_mu
        self._v_mu = b2 * self._v_mu + (1.0 - b2) * (grad_mu ** 2)
        self._m_L = b1 * self._m_L + (1.0 - b1) * grad_L
        self._v_L = b2 * self._v_L + (1.0 - b2) * (grad_L ** 2)
        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t
        self.mu = self.mu - lr * (self._m_mu / bc1) / (np.sqrt(self._v_mu / bc2) + self.adam_eps)
        self.L = self.L - lr * (self._m_L / bc1) / (np.sqrt(self._v_L / bc2) + self.adam_eps)
        self._finalize()

    def _finalize(self) -> None:
        """Keep ``L`` lower triangular with a strictly positive diagonal."""
        self.L = np.tril(self.L)
        diag = np.diag(self.L).copy()
        diag = np.where(np.isfinite(diag), diag, _MIN_DIAG)
        diag = np.maximum(np.abs(diag), _MIN_DIAG)
        np.fill_diagonal(self.L, diag)

    # -- iterations --------------------------------------------------------
    def step(self, lr: Optional[float] = None) -> float:
        """One ADAM step; returns the loss value for the iteration."""
        B = self.batch_size
        eps = standard_normal((B, self.dim), xp=np, rng=self.rng, dtype=self.dtype)
        z = self.mu[None, :] + eps @ self.L.T
        s = self.evaluate_scores(z)
        grad_mu, grad_L, loss = self.gradients(z, eps, s)
        lr_t = self.lam_t(self.t) if lr is None else float(lr)
        self._adam_update(grad_mu, grad_L, lr_t)
        self.grad_evals += B
        self.t += 1
        self.losses.append(loss)
        self.lam_history.append(lr_t)
        if self.track_history and (self.t % self.history_every == 0):
            self.mu_history.append(self.mu.copy())
            self.Sigma_history.append(self.covariance.copy())
        return loss

    def run(self, T: int) -> GradientVIResult:
        """Run ``T`` iterations of the algorithm."""
        for _ in range(int(T)):
            self.step()
        return self.result()

    def result(self) -> GradientVIResult:
        return GradientVIResult(
            mu=self.mu.copy(),
            Sigma=self.covariance.copy(),
            L=self.L.copy(),
            n_iter=int(self.t),
            grad_evals=int(self.grad_evals),
            batch_size=int(self.batch_size),
            losses=list(self.losses),
            mu_history=[m.copy() for m in self.mu_history],
            Sigma_history=[S.copy() for S in self.Sigma_history],
            lam_history=list(self.lam_history),
            name=self.name,
        )

    # convenience ----------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{type(self).__name__}(dim={self.dim}, B={self.batch_size}, "
            f"loss='{self.loss_kind}', lr={self._lr_fn(0):.4g})"
        )


# ----------------------------------------------------------------------------
# Algorithm 2: ADVI (negative ELBO + ADAM)
# ----------------------------------------------------------------------------
class ADVI(GradientVI):
    """ADVI baseline (Algorithm 2): reparameterised Gaussian + negative ELBO.

    ``w_{t+1} = w_t - lambda_t grad_w L_ELBO`` implemented with ADAM.
    """

    loss_kind = "elbo"
    name = "advi"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("loss", "elbo")
        super().__init__(*args, **kwargs)


def advi_fit(
    mu0: np.ndarray,
    Sigma0: np.ndarray,
    score_fn: Optional[Callable] = None,
    target: Any = None,
    T: int = 1000,
    batch_size: int = 2,
    learning_rate: Union[float, Callable[[int], float]] = 0.01,
    loss: str = "elbo",
    track_history: bool = False,
    history_every: int = 1,
    seed: int = 0,
    **kwargs: Any,
) -> GradientVIResult:
    """Functional wrapper: run a gradient-based VI baseline for ``T`` steps."""
    opt = GradientVI(
        mu0=mu0,
        Sigma0=Sigma0,
        score_fn=score_fn,
        target=target,
        loss=loss,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        track_history=track_history,
        history_every=history_every,
        **kwargs,
    )
    return opt.run(T)
