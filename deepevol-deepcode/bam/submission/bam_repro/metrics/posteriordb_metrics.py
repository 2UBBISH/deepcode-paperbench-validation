"""Metrics for the real-application experiments (Section 5.2 and Section 5.3).

The paper (Section 5.2, Appendix E.5) evaluates BaM / ADVI / GSM on posteriordb
targets using the *relative mean error* and the *relative SD error* with respect
to HMC reference samples::

    relative mean error = || (mu - mu_hat) / sigma ||_2
    relative SD   error = || (sigma - sigma_hat) / sigma ||_2

where ``mu, sigma`` are the posterior mean / standard deviation estimated from
the HMC reference samples and ``mu_hat, sigma_hat`` are the corresponding
quantities of the variational distribution ``q``.

Section 5.3 evaluates a deep generative model by how well a test image ``x'`` is
reconstructed from the posterior expectation ``E[z' | x']`` fed through the
network ``Omega(., theta_hat)``; the quality is reported with the mean squared
error (MSE).  ``reconstruction_mse`` below implements that metric.

All functions work with NumPy arrays (and, where cheap, with any array module
that supports the handful of operations used).  The module deliberately has no
hard dependency on posteriordb / BridgeStan so that it can be imported in any
environment; the reference moments are simply passed in as arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    # model bookkeeping
    "PAPER_POSTERIORDB_MODELS",
    "PAPER_POSTERIORDB_NAMES",
    "paper_posteriordb_models",
    # reference moments
    "posterior_moments",
    "hmc_reference_moments",
    "variational_moments",
    # core metrics
    "relative_mean_error",
    "relative_sd_error",
    "relative_errors",
    "ErrorResult",
    # curves / reporting
    "error_curve",
    "relative_error_curve",
    "summarize_errors",
    # VAE metric
    "reconstruction_mse",
    "reconstruction_mse_from_z",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The three posteriordb models used in Section 5.2 together with their
#: dimensionality (Section 5.2: ark D=7, gp-pois-regr D=13,
#: eight-schools-centered D=10).
PAPER_POSTERIORDB_MODELS: Dict[str, Dict[str, Any]] = {
    "ark": {
        "name": "ark",
        "dim": 7,
        "description": "nearly Gaussian target",
        "gaussian": True,
    },
    "gp-pois-regr": {
        "name": "gp-pois-regr",
        "dim": 13,
        "description": "Gaussian process Poisson regression (non-Gaussian)",
        "gaussian": False,
    },
    "eight-schools-centered": {
        "name": "eight-schools-centered",
        "dim": 10,
        "description": "8-schools hierarchical Bayesian model (non-Gaussian)",
        "gaussian": False,
    },
}

#: Model names in the order in which the paper presents them.
PAPER_POSTERIORDB_NAMES: Tuple[str, ...] = tuple(PAPER_POSTERIORDB_MODELS.keys())

#: Batch sizes used in Figure 5.3 (Section 5.2).
PAPER_POSTERIORDB_BATCH_SIZES: Tuple[int, ...] = (8, 32)

#: Number of runs averaged in the posteriordb experiments (Section 5.2).
PAPER_POSTERIORDB_N_RUNS: int = 5

#: Number of runs averaged in the synthetic experiments.
PAPER_SYNTHETIC_N_RUNS: int = 10

_TINY = 1e-300


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _asarray(a, dtype=np.float64) -> np.ndarray:
    """Convert ``a`` to a NumPy array without copying when possible."""
    if isinstance(a, np.ndarray) and a.dtype == dtype:
        return a
    return np.asarray(a, dtype=dtype)


def _get(obj, names: Iterable[str], default: Any = None) -> Any:
    """Return the first attribute/key of ``obj`` found in ``names``."""
    for name in names:
        if isinstance(obj, Mapping):
            if name in obj:
                return obj[name]
        else:
            if hasattr(obj, name):
                value = getattr(obj, name)
                if value is not None:
                    return value
    return default


def paper_posteriordb_models() -> Dict[str, int]:
    """Mapping ``model name -> dimension`` for the three Section 5.2 models."""
    return {name: int(info["dim"]) for name, info in PAPER_POSTERIORDB_MODELS.items()}


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Elementwise ratio guarding against zero denominators."""
    denom = np.where(np.abs(denominator) < _TINY, _TINY, denominator)
    return numerator / denom


def _to_std(sigma_or_cov: Any) -> np.ndarray:
    """Return a vector of standard deviations from a covariance *or* a SD vector.

    A 2-D input is interpreted as a covariance matrix and its diagonal square
    root is returned; a 1-D input is interpreted as already being a vector of
    standard deviations (negative entries are mapped through ``|.|``).
    """
    arr = _asarray(sigma_or_cov)
    if arr.ndim == 2:
        diag = np.diag(arr)
        return np.sqrt(np.clip(diag, 0.0, None))
    return np.sqrt(np.clip(arr, 0.0, None)) if np.all(arr >= 0) else np.abs(arr)


def _to_mean_array(x: Any) -> np.ndarray:
    """Coerce the many possible "fit result" objects into a mean vector."""
    if isinstance(x, (tuple, list)) and len(x) == 2:
        # (mu, Sigma) pair
        return _asarray(x[0]).reshape(-1)
    if hasattr(x, "mean"):
        value = x.mean
        return _asarray(value() if callable(value) else value).reshape(-1)
    if hasattr(x, "mu"):
        return _asarray(x.mu).reshape(-1)
    arr = _asarray(x)
    if arr.ndim == 2:
        # a matrix of posterior samples -> empirical mean
        return arr.mean(axis=0)
    return arr.reshape(-1)


def posterior_moments(samples: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Posterior mean and standard deviation estimated from samples.

    Parameters
    ----------
    samples:
        Array of shape ``(N, D)`` (``N`` HMC reference draws) or ``(N,)`` for
        a scalar parameter.  A ``(mu, Sigma)`` pair is also accepted, in which
        case the moments are read off analytically.

    Returns
    -------
    (mean, std):
        Vectors of length ``D``.
    """
    if isinstance(samples, (tuple, list)) and len(samples) == 2:
        mu, cov = samples
        return _asarray(mu).reshape(-1), _to_std(cov)
    arr = _asarray(samples)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=0)
    return mean.reshape(-1), std.reshape(-1)


#: Alias used by the posteriordb experiment code.
hmc_reference_moments = posterior_moments


def variational_moments(q: Any, mean: Any = None, cov: Any = None) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and per-coordinate SD of a variational approximation.

    Accepts a ``GaussianVariational`` object, a ``(mu, Sigma)`` pair, a result
    dataclass with ``mu`` / ``Sigma`` fields, or explicit ``mean`` / ``cov``
    arguments.
    """
    if q is not None:
        if isinstance(q, (tuple, list)) and len(q) == 2:
            mean, cov = q
        else:
            if mean is None:
                mean = _get(q, ("mu", "mean", "loc"))
            if cov is None:
                cov = _get(q, ("Sigma", "cov", "covariance"))
    if mean is None:
        raise ValueError("variational_moments: could not determine the variational mean")
    mean_vec = _asarray(mean).reshape(-1)
    if cov is None:
        # assume a mean-only object: SD of one
        return mean_vec, np.ones_like(mean_vec)
    return mean_vec, _to_std(cov)


# ---------------------------------------------------------------------------
# Core metrics (Section 5.2 / Appendix E.5, eq. in E.5)
# ---------------------------------------------------------------------------


def relative_mean_error(
    mu: Any,
    mu_hat: Any = None,
    sigma: Any = None,
    q: Any = None,
) -> float:
    r"""Relative mean error :math:`\|(\mu - \hat\mu)/\sigma\|_2`.

    Parameters
    ----------
    mu:
        Posterior mean from the HMC reference samples (or the reference sample
        matrix, from which the mean is computed).
    mu_hat:
        Variational mean.  If omitted, ``q`` is used instead.
    sigma:
        Posterior standard deviation from the HMC reference samples (vector of
        standard deviations or the covariance matrix).
    q:
        Optional variational object / ``(mu, Sigma)`` pair used when ``mu_hat``
        is not supplied.
    """
    if mu_hat is None and q is not None:
        mu_vec, sig_vec_q = variational_moments(q)
        if sigma is None:
            sigma = sig_vec_q
    else:
        mu_vec = _to_mean_array(mu) if not isinstance(mu, (tuple, list)) or len(mu) != 2 else _asarray(mu[0]).reshape(-1)
        mu_hat_vec = _to_mean_array(mu_hat)
        return _relative_mean_error_from_vectors(mu_vec, mu_hat_vec, sigma)

    mu_hat_vec = _to_mean_array(mu_hat) if mu_hat is not None else mu_vec
    return _relative_mean_error_from_vectors(mu_vec, mu_hat_vec, sigma)


def _relative_mean_error_from_vectors(mu: np.ndarray, mu_hat: np.ndarray, sigma: Any) -> float:
    if sigma is None:
        raise ValueError("relative_mean_error requires the reference standard deviations `sigma`")
    sig = _to_std(sigma)
    mu = _asarray(mu).reshape(-1)
    mu_hat = _asarray(mu_hat).reshape(-1)
    if sig.shape != mu.shape:
        sig = np.broadcast_to(sig, mu.shape)
    return float(np.linalg.norm(_safe_divide(mu - mu_hat, sig), ord=2))


def relative_sd_error(
    sigma: Any,
    sigma_hat: Any = None,
    q: Any = None,
) -> float:
    r"""Relative SD error :math:`\|(\sigma - \hat\sigma)/\sigma\|_2`.

    ``sigma`` is the reference standard deviation (vector or covariance) and
    ``sigma_hat`` the variational one; if ``sigma_hat`` is omitted it is read
    from ``q``.
    """
    sig = _to_std(sigma)
    if sigma_hat is None:
        if q is None:
            raise ValueError("relative_sd_error requires `sigma_hat` or a variational object `q`")
        _, sig_hat = variational_moments(q)
    else:
        sig_hat = _to_std(sigma_hat)
    sig = _asarray(sig).reshape(-1)
    sig_hat = _asarray(sig_hat).reshape(-1)
    if sig_hat.shape != sig.shape:
        sig_hat = np.broadcast_to(sig_hat, sig.shape)
    return float(np.linalg.norm(_safe_divide(sig - sig_hat, sig), ord=2))


@dataclass
class ErrorResult:
    """Container for the two Section 5.2 error metrics."""

    relative_mean_error: float
    relative_sd_error: float
    n_params: int = 0
    model: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relative_mean_error": self.relative_mean_error,
            "relative_sd_error": self.relative_sd_error,
            "n_params": self.n_params,
            "model": self.model,
        }

    def __iter__(self):
        # allows ``rme, rsde = relative_errors(...)``
        yield self.relative_mean_error
        yield self.relative_sd_error


def relative_errors(
    reference: Any,
    q: Any = None,
    mu_hat: Any = None,
    cov_hat: Any = None,
    model: Optional[str] = None,
) -> ErrorResult:
    """Compute both relative errors for a single variational fit.

    Parameters
    ----------
    reference:
        HMC reference samples ``(N, D)``, a ``(mu, sigma)`` pair, or a
        ``(mu, Sigma)`` pair; the posterior moments are derived from it.
    q:
        Variational approximation (``GaussianVariational``, result dataclass or
        ``(mu, Sigma)`` pair).
    mu_hat, cov_hat:
        Explicit variational moments; override the values taken from ``q``.
    """
    mu_ref, sd_ref = posterior_moments(reference)

    if mu_hat is None and q is not None:
        mu_hat, sd_hat = variational_moments(q)
    else:
        mu_hat = _to_mean_array(mu_hat)
        sd_hat = _to_std(cov_hat) if cov_hat is not None else np.ones_like(mu_hat)

    rme = _relative_mean_error_from_vectors(mu_ref, mu_hat, sd_ref)
    sd_ref = _to_std(sd_ref)
    if sd_hat.shape != sd_ref.shape:
        sd_hat = np.broadcast_to(sd_hat, sd_ref.shape)
    rsde = float(np.linalg.norm(_safe_divide(sd_ref - sd_hat, sd_ref), ord=2))
    return ErrorResult(rme, rsde, n_params=int(mu_ref.size), model=model)


# ---------------------------------------------------------------------------
# Curves and aggregation
# ---------------------------------------------------------------------------


def error_curve(
    reference: Any,
    mu_history: Sequence[Any],
    Sigma_history: Optional[Sequence[Any]] = None,
    grad_evals: Optional[Sequence[int]] = None,
    model: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Relative mean/SD error along a sequence of variational iterates.

    Parameters
    ----------
    reference:
        HMC reference samples (or reference moments) used to define the target
        posterior mean and SD.
    mu_history:
        Iterable of mean vectors; may also be an iterable of
        ``GaussianVariational`` objects / ``(mu, Sigma)`` pairs, in which case
        ``Sigma_history`` is optional.
    Sigma_history:
        Iterable of covariance matrices aligned with ``mu_history``.
    grad_evals:
        Gradient evaluations at each recorded iterate; defaults to
        ``0, 1, ..., T-1``.

    Returns
    -------
    dict with keys ``grad_evals``, ``relative_mean_error``, ``relative_sd_error``.
    """
    mu_ref, sd_ref = posterior_moments(reference)
    sig_ref = _to_std(sd_ref)

    if Sigma_history is None:
        mus = []
        sigmas = []
        for item in mu_history:
            if hasattr(item, "mu") or (isinstance(item, (tuple, list)) and len(item) == 2):
                m, s = variational_moments(item)
            else:
                m, s = _to_mean_array(item), np.ones_like(mu_ref)
            mus.append(m)
            sigmas.append(s)
        mu_arr = np.asarray(mus, dtype=np.float64)
        sd_arr = np.asarray(sigmas, dtype=np.float64)
    else:
        mu_arr = np.asarray([_to_mean_array(m) for m in mu_history], dtype=np.float64)
        sd_arr = np.asarray([_to_std(s) for s in Sigma_history], dtype=np.float64)

    if mu_arr.ndim == 1:
        mu_arr = mu_arr.reshape(1, -1)
    if sd_arr.ndim == 1:
        sd_arr = sd_arr.reshape(1, -1)

    n = mu_arr.shape[0]
    rmse_curve = np.empty(n, dtype=np.float64)
    rsde_curve = np.empty(n, dtype=np.float64)
    for i in range(n):
        m = mu_arr[i]
        s = sd_arr[i]
        if s.shape != sig_ref.shape:
            s = np.broadcast_to(s, sig_ref.shape)
        rmse_curve[i] = np.linalg.norm(_safe_divide(mu_ref - m, sig_ref), ord=2)
        rsde_curve[i] = np.linalg.norm(_safe_divide(sig_ref - s, sig_ref), ord=2)

    if grad_evals is None:
        x = np.asarray(np.arange(n), dtype=np.float64)
    else:
        x = np.asarray(list(grad_evals), dtype=np.float64).reshape(-1)
        if x.size != n:
            x = np.resize(x, n)

    return {
        "grad_evals": x,
        "relative_mean_error": rmse_curve,
        "relative_sd_error": rsde_curve,
        "model": np.asarray([model] * n) if model is not None else np.asarray([None] * n),
    }


#: Alias with a more explicit name.
relative_error_curve = error_curve


def summarize_errors(
    results: Sequence[Union[ErrorResult, Mapping[str, float], Any]],
    ddof: int = 1,
) -> Dict[str, Dict[str, float]]:
    """Mean / standard error over independent runs (Section 5.2 protocol).

    ``results`` is a sequence of ``ErrorResult`` (or mappings with the two
    metric keys).  ``ddof=1`` gives the sample standard deviation used for the
    standard error of the mean, matching the paper's shaded regions.
    """
    rme = []
    rsde = []
    for r in results:
        if isinstance(r, ErrorResult):
            rme.append(r.relative_mean_error)
            rsde.append(r.relative_sd_error)
        elif isinstance(r, Mapping):
            rme.append(float(r["relative_mean_error"]))
            rsde.append(float(r["relative_sd_error"]))
        elif isinstance(r, (tuple, list)) and len(r) == 2:
            rme.append(float(r[0]))
            rsde.append(float(r[1]))
        else:  # fall back to attribute access
            rme.append(float(getattr(r, "relative_mean_error")))
            rsde.append(float(getattr(r, "relative_sd_error")))

    def _stat(values: Sequence[float]) -> Dict[str, float]:
        arr = np.asarray(values, dtype=np.float64)
        n = arr.size
        mean = float(arr.mean()) if n else float("nan")
        if n > 1:
            std = float(arr.std(ddof=ddof))
            stderr = std / np.sqrt(n)
        else:
            std = 0.0
            stderr = 0.0
        return {"mean": mean, "std": std, "stderr": float(stderr), "n": int(n)}

    return {"relative_mean_error": _stat(rme), "relative_sd_error": _stat(rsde)}


# ---------------------------------------------------------------------------
# Section 5.3 - reconstruction MSE for the deep generative model
# ---------------------------------------------------------------------------


def reconstruction_mse(x: Any, x_hat: Any, reduction: str = "mean") -> Union[float, np.ndarray]:
    """Mean squared reconstruction error between images ``x`` and ``x_hat``.

    ``x`` and ``x_hat`` are arrays of shape ``(N, 3072)`` (CIFAR-10 flattened)
    or ``(N, C, H, W)``; the MSE is taken over all pixels of all images.
    ``reduction`` may be ``"mean"``, ``"sum"`` or ``"none"`` (per-image MSEs).
    """
    a = _asarray(x)
    b = _asarray(x_hat)
    if a.shape != b.shape:
        b = np.reshape(b, a.shape)
    diff = a - b
    per_image = np.mean(diff ** 2, axis=tuple(range(1, a.ndim))) if a.ndim > 1 else diff ** 2
    if reduction == "mean":
        return float(np.mean(per_image))
    if reduction == "sum":
        return float(np.sum(per_image))
    if reduction == "none":
        return per_image
    raise ValueError(f"unknown reduction {reduction!r}")


def reconstruction_mse_from_z(
    z_mean: Any,
    decoder: Any,
    x: Any,
    reduction: str = "mean",
) -> Union[float, np.ndarray]:
    """Reconstruct ``x`` by passing ``E[z|x]`` through the decoder ``Omega``.

    Parameters
    ----------
    z_mean:
        Posterior expectation of the latent code, shape ``(N, 256)`` (or a
        single code of shape ``(256,)``, which is promoted to a batch of one).
    decoder:
        Callable implementing ``Omega(., theta_hat)``; takes a batch of latent
        codes and returns the likelihood mean in image space.
    x:
        Target images, shape ``(N, 3072)`` or ``(N, C, H, W)``.
    """
    z = _asarray(z_mean)
    if z.ndim == 1:
        z = z[None, :]
    x_hat = decoder(z)
    x_hat = np.asarray(x_hat)
    return reconstruction_mse(x, x_hat, reduction=reduction)
