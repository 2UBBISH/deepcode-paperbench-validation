"""KL-divergence metrics for the BaM reproduction.

The paper evaluates every synthetic experiment with *empirical estimates of the KL
divergence in both directions* (Section 5.1):

    forward KL:  KL(p ; q) = E_p[ log p(z) - log q(z) ]
    reverse KL:  KL(q ; p) = E_q[ log q(z) - log p(z) ]

For Gaussian targets both quantities are available in closed form (see the
Gaussian KL identities, eq. (15)/(16) of Appendix A), whereas for the
sinh-arcsinh target (Section 5.1 / Appendix E.4) they are estimated with Monte
Carlo using the known (analytic) target log density and samples drawn from the
target or from the variational distribution.

This module therefore exposes

* :func:`forward_kl` / :func:`reverse_kl` / :func:`kl_divergence` -- dispatching
  helpers that pick the closed-form Gaussian formula when the target provides
  one, and Monte Carlo otherwise;
* :func:`mc_forward_kl` / :func:`mc_reverse_kl` -- explicit Monte Carlo
  estimators (also usable for purely empirical ``log p`` / ``log q`` functions);
* :func:`kl_curve` -- evaluate a whole optimisation trace (the x-axis of Figures
  5.1/5.2 is the number of *gradient evaluations*, not iterations, so the curve
  helper accepts the matching ``grad_evals`` array).

The whole module is written to work in NumPy and, when available, inside JAX
(arrays are dispatched through ``array_namespace``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

from ..bam.matrix_equations import (
    _HAS_JAX,
    array_namespace,
    ensure_spd,
    symmetrize,
)

if _HAS_JAX:  # pragma: no cover - optional dependency
    import jax  # noqa: F401
    import jax.numpy as jnp

__all__ = [
    "KLResult",
    "DEFAULT_KL_SAMPLES",
    "gaussian_forward_kl",
    "gaussian_reverse_kl",
    "mc_forward_kl",
    "mc_reverse_kl",
    "forward_kl",
    "reverse_kl",
    "kl_divergence",
    "kl_curve",
    "empirical_log_ratio",
    "is_gaussian_target",
    "has_closed_form_kl",
    "kl_pair",
]


# ---------------------------------------------------------------------------
# constants / small helpers
# ---------------------------------------------------------------------------

DEFAULT_KL_SAMPLES = 4096
"""Number of Monte Carlo samples used when no closed form is available."""

_MIN_EIG = 1e-12


@dataclass
class KLResult:
    """Value of a KL divergence together with its Monte Carlo standard error.

    ``closed_form`` is ``True`` when the value came from an analytic formula, in
    which case ``stderr`` is ``0.0``.
    """

    value: float
    stderr: float = 0.0
    direction: str = "forward"
    closed_form: bool = False
    n_samples: int = 0

    def __float__(self) -> float:  # convenience: allow np.asarray(...) semantics
        return float(self.value)

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "stderr": self.stderr,
            "direction": self.direction,
            "closed_form": self.closed_form,
            "n_samples": self.n_samples,
        }


def _is_jax_namespace(xp) -> bool:
    return _HAS_JAX and xp is jnp


def _asarray(a, xp=np):
    if xp is None:
        xp = np
    return xp.asarray(a)


def _sym_spd(A, xp):
    """Symmetrise and (in NumPy) project a matrix onto the SPD cone."""
    A = symmetrize(A)
    if _is_jax_namespace(xp):
        return A
    A = np.asarray(A)
    eigs = np.linalg.eigvalsh(0.5 * (A + A.T))
    min_eig = float(np.min(eigs)) if eigs.size else 0.0
    if not np.isfinite(min_eig) or min_eig < _MIN_EIG:
        A = ensure_spd(A, jitter=1e-10, min_eig=1e-10)
    return A


# ---------------------------------------------------------------------------
# closed-form Gaussian KL
# ---------------------------------------------------------------------------

def gaussian_forward_kl(
    mu_star,
    Sigma_star,
    mu,
    Sigma,
    jitter: float = 1e-12,
    xp=np,
) -> float:
    """``KL(N(mu*, Sigma*) ; N(mu, Sigma))`` -- the forward KL ``KL(p ; q)``.

    Analytic formula::

        KL(p ; q) = 1/2 [ log|Sigma| - log|Sigma*|
                          - D + tr(Sigma^{-1} Sigma*)
                          + (mu - mu*) Sigma^{-1} (mu - mu*) ]

    where ``p = N(mu*, Sigma*)`` is the target and ``q = N(mu, Sigma)`` the
    variational approximation.
    """
    xp = array_namespace(mu_star, Sigma_star, mu, Sigma) if xp is np else xp
    mu_star = xp.asarray(mu_star).reshape(-1)
    mu = xp.asarray(mu).reshape(-1)
    Sigma_star = _sym_spd(xp.asarray(Sigma_star), xp)
    Sigma = _sym_spd(xp.asarray(Sigma), xp)
    D = int(mu.shape[0])

    L_q = xp.linalg.cholesky(Sigma + jitter * xp.eye(D) if _is_jax_namespace(xp) else Sigma + jitter * np.eye(D))
    log_det_q = 2.0 * xp.sum(xp.log(xp.diagonal(L_q)))

    if _is_jax_namespace(xp):
        L_p = xp.linalg.cholesky(Sigma_star + jitter * xp.eye(D))
        log_det_p = 2.0 * xp.sum(xp.log(xp.diagonal(L_p)))
        # tr(Sigma^{-1} Sigma*)
        Sigma_inv = xp.linalg.inv(Sigma)
        trace_term = xp.trace(Sigma_inv @ Sigma_star)
        diff = mu - mu_star
        quad = diff @ (Sigma_inv @ diff)
        # Sigma_inv is nonsymmetric for jnp; take the symmetric part of the quadratic.
        quad = 0.5 * (quad + diff @ (Sigma_inv.T @ diff))
    else:
        L_p = np.linalg.cholesky(np.asarray(Sigma_star) + jitter * np.eye(D))
        log_det_p = 2.0 * np.sum(np.log(np.diag(L_p)))
        Sigma_inv = np.linalg.inv(np.asarray(Sigma))
        trace_term = np.trace(Sigma_inv @ np.asarray(Sigma_star))
        diff = np.asarray(mu) - np.asarray(mu_star)
        quad = float(diff @ (Sigma_inv @ diff))

    kl = 0.5 * (log_det_q - log_det_p - D + trace_term + quad)
    return float(np.asarray(kl))


def gaussian_reverse_kl(
    mu_star,
    Sigma_star,
    mu,
    Sigma,
    jitter: float = 1e-12,
    xp=np,
) -> float:
    """``KL(N(mu, Sigma) ; N(mu*, Sigma*))`` -- the reverse KL ``KL(q ; p)``."""
    # KL(q || p) is just the forward KL with the arguments swapped.
    return gaussian_forward_kl(mu, Sigma, mu_star, Sigma_star, jitter=jitter, xp=xp)


# ---------------------------------------------------------------------------
# Monte Carlo estimators
# ---------------------------------------------------------------------------

def empirical_log_ratio(
    log_p: Union[np.ndarray, Any],
    log_q: Union[np.ndarray, Any],
    positive: bool = True,
) -> Tuple[float, float]:
    """Return ``mean(log_p - log_q)`` and its standard error.

    ``positive`` selects the forward convention, ``E_p[log p - log q]``; set
    ``positive=False`` for the reverse one, ``E_q[log q - log p]``.
    """
    log_p = np.asarray(log_p, dtype=np.float64).reshape(-1)
    log_q = np.asarray(log_q, dtype=np.float64).reshape(-1)
    if log_p.shape != log_q.shape:
        raise ValueError(
            f"log_p and log_q must have the same shape, got {log_p.shape} and {log_q.shape}"
        )
    ratio = log_p - log_q
    if not positive:
        ratio = -ratio
    n = ratio.shape[0]
    if n == 0:
        return float("nan"), float("nan")
    value = float(np.mean(ratio))
    if n > 1:
        stderr = float(np.std(ratio, ddof=1) / np.sqrt(n))
    else:
        stderr = 0.0
    return value, stderr


def _gaussian_log_prob_batch(z, mu, Sigma, xp=np, jitter: float = 0.0):
    """Vectorised ``log N(z ; mu, Sigma)`` for ``z`` of shape ``(B, D)``."""
    z = xp.asarray(z)
    mu = xp.asarray(mu).reshape(-1)
    Sigma = _sym_spd(xp.asarray(Sigma), xp)
    D = int(mu.shape[0])
    eye = xp.eye(D)
    L = xp.linalg.cholesky(Sigma + jitter * eye)
    diff = z - mu
    # solve L w = diff^T  ->  w = L^{-1} diff^T
    w = xp.linalg.solve(L, diff.T)
    quad = xp.sum(w * w, axis=0)
    log_det = 2.0 * xp.sum(xp.log(xp.diagonal(L)))
    const = D * float(np.log(2.0 * np.pi))
    return -0.5 * (const + log_det + quad)


# ---------------------------------------------------------------------------
# target / q duck-typing helpers
# ---------------------------------------------------------------------------

def is_gaussian_target(target: Any) -> bool:
    """``True`` when *target* is a Gaussian target (closed-form KL available)."""
    if target is None:
        return False
    if hasattr(target, "forward_kl") and hasattr(target, "reverse_kl") and hasattr(target, "precision"):
        return True
    return getattr(target, "name", "") == "gaussian"


def has_closed_form_kl(target: Any) -> bool:
    return is_gaussian_target(target)


def _target_moments(target: Any, xp):
    """Extract ``(mu_star, Sigma_star)`` from a Gaussian-like target."""
    for mean_attr in ("mean", "mu", "mu_star", "mean_"):
        if hasattr(target, mean_attr):
            mu_star = getattr(target, mean_attr)
            for cov_attr in ("cov", "covariance", "Sigma", "Sigma_star"):
                if hasattr(target, cov_attr):
                    return xp.asarray(mu_star).reshape(-1), xp.asarray(getattr(target, cov_attr))
    raise TypeError("could not extract (mean, covariance) from the target")


def _sample_target(target: Any, n: int, rng=None, key=None, xp=np):
    """Draw ``n`` samples from a target, tolerating a few calling conventions."""
    attempts = (
        lambda: target.sample(n, rng=rng, key=key),
        lambda: target.sample(n, rng=rng),
        lambda: target.sample(n, key=key),
        lambda: target.sample(n),
        lambda: target.sample(n_samples=n),
    )
    last_error: Optional[Exception] = None
    for attempt in attempts:
        try:
            return xp.asarray(attempt())
        except TypeError as err:  # signature mismatch, try next form
            last_error = err
        except Exception as err:  # pragma: no cover - re-raised below
            last_error = err
            break
    raise TypeError(f"could not sample from target: {last_error}")


def _target_log_prob(target: Any, z, xp):
    """Evaluate ``log p(z)`` for an array of samples."""
    for name in ("log_prob", "log_density", "logp"):
        fn = getattr(target, name, None)
        if fn is None:
            continue
        try:
            return xp.asarray(fn(z))
        except Exception:
            try:
                out = [fn(xp.asarray(zi)) for zi in xp.asarray(z)]
                return xp.asarray(out)
            except Exception:
                continue
    raise TypeError("target does not expose a usable log density")


def _q_moments(q, xp):
    """Return ``(mu, Sigma)`` for a Gaussian variational object or 2-tuple."""
    if isinstance(q, (tuple, list)) and len(q) == 2:
        return xp.asarray(q[0]).reshape(-1), xp.asarray(q[1])
    for mu_attr in ("mean", "mu"):
        if hasattr(q, mu_attr):
            mu = getattr(q, mu_attr)
            if callable(mu):
                mu = mu()
            for cov_attr in ("covariance", "cov", "Sigma"):
                if hasattr(q, cov_attr):
                    cov = getattr(q, cov_attr)
                    if callable(cov):
                        cov = cov()
                    return xp.asarray(mu).reshape(-1), xp.asarray(cov)
    raise TypeError("could not extract (mean, covariance) from the variational distribution")


def _q_log_prob(q, z, xp, jitter: float = 0.0):
    """Evaluate ``log q(z)`` for an array of samples ``(B, D)``."""
    if isinstance(q, (tuple, list)):
        mu, Sigma = _q_moments(q, xp)
        return _gaussian_log_prob_batch(z, mu, Sigma, xp=xp, jitter=jitter)
    fn = getattr(q, "log_prob", None)
    if fn is not None:
        try:
            out = xp.asarray(fn(z))
            if out.ndim > 0:
                return out
        except Exception:
            pass
        try:
            return xp.asarray([fn(xp.asarray(zi)) for zi in xp.asarray(z)])
        except Exception:
            pass
    mu, Sigma = _q_moments(q, xp)
    return _gaussian_log_prob_batch(z, mu, Sigma, xp=xp, jitter=jitter)


def _q_sample(q, n: int, rng=None, key=None, xp=np):
    """Draw ``n`` samples from the variational distribution."""
    if isinstance(q, (tuple, list)):
        mu, Sigma = _q_moments(q, xp)
        if _is_jax_namespace(xp):
            L = xp.linalg.cholesky(Sigma)
            eps = jax.random.normal(key, (n, mu.shape[0]))
            return mu + eps @ L.T
        L = np.linalg.cholesky(np.asarray(Sigma))
        eps = np.asarray(rng if rng is not None else np.random.default_rng()).normal(size=(n, mu.shape[0]))
        return mu + eps @ L.T
    attempts = (
        lambda: q.sample(n, rng=rng, key=key),
        lambda: q.sample(n, rng=rng),
        lambda: q.sample(n, key=key),
        lambda: q.sample(n),
        lambda: q.sample(batch_size=n),
    )
    last_error: Optional[Exception] = None
    for attempt in attempts:
        try:
            return xp.asarray(attempt())
        except TypeError as err:
            last_error = err
        except Exception as err:  # pragma: no cover
            last_error = err
            break
    raise TypeError(f"could not sample from the variational distribution: {last_error}")


# ---------------------------------------------------------------------------
# Monte Carlo KL
# ---------------------------------------------------------------------------

def mc_forward_kl(
    target: Any,
    q: Any,
    n_samples: int = DEFAULT_KL_SAMPLES,
    rng=None,
    key=None,
    batch_size: Optional[int] = None,
    xp=np,
) -> KLResult:
    """Monte Carlo estimate of ``KL(p ; q) = E_p[log p(z) - log q(z)]``.

    Samples are drawn from the **target** ``p``.  When ``q`` is a Gaussian
    variational object the Gaussian log density is evaluated in closed form, so
    the estimator is exact up to the Monte Carlo error of the expectation.
    """
    if xp is np:
        xp = array_namespace(getattr(q, "mean", 0.0)) if _HAS_JAX else np
    if xp is None:
        xp = np
    if rng is None and key is None and not _is_jax_namespace(xp):
        rng = np.random.default_rng()

    if _is_jax_namespace(xp):
        z = _sample_target(target, n_samples, rng=rng, key=key, xp=xp)
        log_p = _target_log_prob(target, z, xp)
        log_q = _q_log_prob(q, z, xp)
        ratio = np.asarray(log_p) - np.asarray(log_q)
        value = float(np.mean(ratio))
        stderr = float(np.std(ratio, ddof=1) / np.sqrt(ratio.shape[0])) if ratio.shape[0] > 1 else 0.0
        return KLResult(value=value, stderr=stderr, direction="forward", closed_form=False,
                        n_samples=int(n_samples))

    z = _sample_target(target, n_samples, rng=rng, key=key, xp=xp)
    z = np.asarray(z)
    log_p = np.asarray(_target_log_prob(target, z, xp))
    log_q = np.asarray(_q_log_prob(q, z, xp))
    value, stderr = empirical_log_ratio(log_p, log_q, positive=True)
    return KLResult(value=value, stderr=stderr, direction="forward", closed_form=False,
                    n_samples=int(n_samples))


def mc_reverse_kl(
    target: Any,
    q: Any,
    n_samples: int = DEFAULT_KL_SAMPLES,
    rng=None,
    key=None,
    batch_size: Optional[int] = None,
    xp=np,
) -> KLResult:
    """Monte Carlo estimate of ``KL(q ; p) = E_q[log q(z) - log p(z)]``."""
    if xp is None:
        xp = np
    if rng is None and key is None and not _is_jax_namespace(xp):
        rng = np.random.default_rng()

    z = _q_sample(q, n_samples, rng=rng, key=key, xp=xp)
    if _is_jax_namespace(xp):
        z = xp.asarray(z)
        log_q = _q_log_prob(q, z, xp)
        log_p = _target_log_prob(target, z, xp)
        ratio = np.asarray(log_q) - np.asarray(log_p)
        value = float(np.mean(ratio))
        stderr = float(np.std(ratio, ddof=1) / np.sqrt(ratio.shape[0])) if ratio.shape[0] > 1 else 0.0
        return KLResult(value=value, stderr=stderr, direction="reverse", closed_form=False,
                        n_samples=int(n_samples))

    z = np.asarray(z)
    log_q = np.asarray(_q_log_prob(q, z, xp))
    log_p = np.asarray(_target_log_prob(target, z, xp))
    value, stderr = empirical_log_ratio(log_q, log_p, positive=True)
    return KLResult(value=value, stderr=stderr, direction="reverse", closed_form=False,
                    n_samples=int(n_samples))


# ---------------------------------------------------------------------------
# dispatching front-ends
# ---------------------------------------------------------------------------

def _closed_form_kl(target, q, direction: str, xp):
    """Closed-form Gaussian KL, preferring the target's own implementation."""
    mu, Sigma = _q_moments(q, xp)
    if direction == "forward":
        fn = getattr(target, "forward_kl", None)
        if callable(fn):
            try:
                return float(np.asarray(fn(mu, Sigma)))
            except Exception:
                pass
        mu_star, Sigma_star = _target_moments(target, xp)
        return gaussian_forward_kl(mu_star, Sigma_star, mu, Sigma, xp=xp)
    fn = getattr(target, "reverse_kl", None)
    if callable(fn):
        try:
            return float(np.asarray(fn(mu, Sigma)))
        except Exception:
            pass
    mu_star, Sigma_star = _target_moments(target, xp)
    return gaussian_reverse_kl(mu_star, Sigma_star, mu, Sigma, xp=xp)


def kl_divergence(
    target: Any,
    q: Any,
    direction: str = "forward",
    method: str = "auto",
    n_samples: int = DEFAULT_KL_SAMPLES,
    rng=None,
    key=None,
    xp=np,
) -> KLResult:
    """KL divergence between a target ``p`` and a Gaussian ``q``.

    Parameters
    ----------
    target:
        The target distribution; Gaussian targets give a closed-form result.
    q:
        A Gaussian variational object (``GaussianVariational``) or a ``(mu,
        Sigma)`` pair.
    direction:
        ``"forward"`` for ``KL(p ; q)`` (the metric of Figure 5.1) or
        ``"reverse"`` for ``KL(q ; p)`` (Figure E.3).
    method:
        ``"auto"`` (closed form whenever the target is Gaussian, Monte Carlo
        otherwise), ``"closed_form"`` to force the analytic formula, or
        ``"mc"`` to force Monte Carlo.
    """
    direction = str(direction).lower()
    if direction in ("fwd", "forward", "kl(p;q)", "kl_p_q"):
        direction = "forward"
    elif direction in ("rev", "reverse", "backward", "kl(q;p)", "kl_q_p"):
        direction = "reverse"
    else:
        raise ValueError(f"unknown KL direction {direction!r}")

    method = "auto" if method is None else str(method).lower()
    if xp is np:
        q_xp = array_namespace(getattr(q, "mean", 0.0)) if _HAS_JAX else np
        xp = q_xp if q_xp is not None else np

    use_closed_form = method in ("closed_form", "closed", "analytic", "exact") or (
        method == "auto" and has_closed_form_kl(target)
    )
    if use_closed_form:
        value = _closed_form_kl(target, q, direction, xp)
        return KLResult(value=value, stderr=0.0, direction=direction, closed_form=True,
                        n_samples=0)

    mc = mc_forward_kl if direction == "forward" else mc_reverse_kl
    return mc(target, q, n_samples=n_samples, rng=rng, key=key, xp=xp)


def forward_kl(target: Any, q: Any, **kwargs) -> KLResult:
    """``KL(p ; q)`` -- see :func:`kl_divergence`."""
    return kl_divergence(target, q, direction="forward", **kwargs)


def reverse_kl(target: Any, q: Any, **kwargs) -> KLResult:
    """``KL(q ; p)`` -- see :func:`kl_divergence`."""
    return kl_divergence(target, q, direction="reverse", **kwargs)


def kl_pair(
    target: Any,
    q: Any,
    n_samples: int = DEFAULT_KL_SAMPLES,
    rng=None,
    key=None,
    xp=np,
) -> Tuple[KLResult, KLResult]:
    """Return ``(forward, reverse)`` KL results for a single variational fit."""
    return (
        forward_kl(target, q, n_samples=n_samples, rng=rng, key=key, xp=xp),
        reverse_kl(target, q, n_samples=n_samples, rng=rng, key=key, xp=xp),
    )


# ---------------------------------------------------------------------------
# curve helpers for the figures
# ---------------------------------------------------------------------------

def kl_curve(
    target: Any,
    mu_history: Sequence[Any],
    Sigma_history: Sequence[Any],
    direction: str = "forward",
    grad_evals: Optional[Sequence[float]] = None,
    grid: Optional[Sequence[int]] = None,
    n_samples: int = DEFAULT_KL_SAMPLES,
    method: str = "auto",
    rng=None,
    seed: int = 0,
    xp=np,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the KL divergence along a variational trace.

    Returns ``(grad_evals, kl, kl_stderr)`` as NumPy arrays; the first entry is
    the abscissa actually used for plotting (the supplied ``grad_evals`` when
    available, otherwise ``0, 1, 2, ...``).

    ``mu_history[i]`` / ``Sigma_history[i]`` describe the variational state after
    iteration ``i``.  Because the experiments plot metrics against the number of
    gradient evaluations, callers should pass ``grad_evals``
    (``batch_size * t`` for BaM and the baselines).
    """
    mu_history = list(mu_history)
    Sigma_history = list(Sigma_history)
    n = min(len(mu_history), len(Sigma_history))
    if n == 0:
        raise ValueError("empty variational trace")

    if grid is None:
        grid = list(range(n))
    grid = [int(i) for i in grid if 0 <= int(i) < n]

    if grad_evals is None:
        xs = np.asarray(grid, dtype=np.float64)
    else:
        grad_evals = np.asarray(grad_evals, dtype=np.float64)
        xs = grad_evals[np.asarray(grid, dtype=int)]

    rng = np.random.default_rng(seed) if rng is None else rng
    values = np.empty(len(grid), dtype=np.float64)
    errors = np.empty(len(grid), dtype=np.float64)
    for j, i in enumerate(grid):
        res = kl_divergence(
            target,
            (mu_history[i], Sigma_history[i]),
            direction=direction,
            method=method,
            n_samples=n_samples,
            rng=rng,
            xp=xp,
        )
        values[j] = res.value
        errors[j] = res.stderr
    return xs, values, errors


def kl_from_fit(
    target: Any,
    mu,
    Sigma,
    direction: str = "forward",
    method: str = "auto",
    n_samples: int = DEFAULT_KL_SAMPLES,
    rng=None,
    xp=np,
) -> float:
    """Scalar KL value for one variational state (convenience wrapper)."""
    return kl_divergence(
        target, (mu, Sigma), direction=direction, method=method,
        n_samples=n_samples, rng=rng, xp=xp,
    ).value
