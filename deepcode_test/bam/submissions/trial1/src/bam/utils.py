"""Shared utilities for the Batch-and-Match (BaM) reproduction.

This module collects small, reusable helpers used across experiments and
notebooks:

* JAX pseudo-random-number-generator (PRNG) key management
* wall-clock timing helpers
* closed-form forward/reverse KL between Gaussian distributions
* normalized and relative error metrics used in the paper's experiments
* lightweight plotting/saving helpers

The implementations here are intentionally dependency-light: only ``jax`` and
``numpy`` are imported at module level.  Matplotlib is imported lazily inside
plotting helpers so that importing :mod:`bam.utils` never requires a display or
the full plotting stack.
"""

from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    # PRNG helpers
    "seed_key",
    "split_key",
    "fold_in",
    "key_stream",
    "KeyGen",
    # timing
    "Timer",
    "time_block",
    "wallclock",
    # Gaussian metrics
    "symmetrize",
    "sym_inv_sqrt",
    "logdet",
    "kl_forward_gaussian",
    "kl_reverse_gaussian",
    "forward_kl_gaussian",
    "reverse_kl_gaussian",
    "gaussian_kl",
    "normalized_mean_error",
    "normalized_covariance_error",
    "mean_sd_from_draws",
    "relative_mean_error",
    "relative_sd_error",
    "posterior_error_summary",
    # plotting / file helpers
    "ensure_directory",
    "as_numpy",
    "mean_sem",
    "set_plot_style",
    "save_figure",
    "new_figure",
    "plot_mean_sem",
]


# ---------------------------------------------------------------------------
# PRNG helpers
# ---------------------------------------------------------------------------

def seed_key(seed: Union[int, jax.Array]) -> jax.Array:
    """Return a JAX PRNG key from an integer seed.

    Parameters
    ----------
    seed:
        Integer seed or an existing JAX key.  Passing an existing key is a
        no-op, which makes this convenient for default-argument seeding.

    Returns
    -------
    jax.Array
        A JAX PRNG key.
    """
    if isinstance(seed, jax.Array):
        return seed
    return jax.random.PRNGKey(int(seed))


def split_key(key: jax.Array, n: int = 2) -> Tuple[jax.Array, ...]:
    """Split a PRNG key into ``n`` independent keys."""
    keys = jax.random.split(key, max(int(n), 1))
    return tuple(keys[i] for i in range(int(n)))


def fold_in(key: jax.Array, data: Any) -> jax.Array:
    """Deterministically fold arbitrary data into a PRNG key.

    ``data`` is converted to an integer hash using Python's built-in ``hash``,
    which is deterministic within a process (and usually across runs for
    strings/ints).  For fully reproducible experiments, callers should prefer
    explicit ``jax.random.fold_in`` with integer counters when strict
    cross-process determinism is required.
    """
    return jax.random.fold_in(key, hash(data))


def key_stream(seed: Union[int, jax.Array], count: Optional[int] = None) -> Iterator[jax.Array]:
    """Yield a stream of independent PRNG keys.

    Parameters
    ----------
    seed:
        Seed or parent key.
    count:
        Number of keys to yield, or ``None`` for an infinite stream.
    """
    key = seed_key(seed)
    produced = 0
    while count is None or produced < count:
        key, subkey = jax.random.split(key)
        yield subkey
        produced += 1


class KeyGen:
    """Small stateful PRNG-key generator.

    Example
    -------
    >>> rng = KeyGen(0)
    >>> key1 = rng()
    >>> key2 = rng()
    """

    def __init__(self, seed: Union[int, jax.Array]) -> None:
        self._key = seed_key(seed)

    def __call__(self) -> jax.Array:
        self._key, subkey = jax.random.split(self._key)
        return subkey

    @property
    def key(self) -> jax.Array:
        """Return the current internal key without advancing it."""
        return self._key


# ---------------------------------------------------------------------------
# Wall-clock timing
# ---------------------------------------------------------------------------

class Timer:
    """Simple wall-clock timer usable both as an object and a context manager.

    Examples
    --------
    >>> timer = Timer()
    >>> with timer:
    ...     pass
    >>> print(timer.elapsed)
    """

    def __init__(self) -> None:
        self._start: Optional[float] = None
        self._end: Optional[float] = None

    def start(self) -> "Timer":
        """Start or restart the timer."""
        self._start = time.perf_counter()
        self._end = None
        return self

    def stop(self) -> float:
        """Stop the timer and return elapsed seconds."""
        self._end = time.perf_counter()
        return self.elapsed

    @property
    def elapsed(self) -> float:
        """Elapsed seconds, or 0.0 if never started."""
        if self._start is None:
            return 0.0
        end = self._end if self._end is not None else time.perf_counter()
        return end - self._start

    def __enter__(self) -> "Timer":
        return self.start()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.stop()


@contextlib.contextmanager
def time_block(
    label: Optional[str] = None,
    *,
    verbose: bool = True,
) -> Iterator[Timer]:
    """Context manager that times a code block and optionally prints the result.

    Parameters
    ----------
    label:
        Human-readable block label.  If ``None``, the block is silently timed.
    verbose:
        If ``True`` and ``label`` is not ``None``, print elapsed seconds on exit.
    """
    timer = Timer().start()
    try:
        yield timer
    finally:
        timer.stop()
        if verbose and label is not None:
            print(f"[{label}] {timer.elapsed:.4f}s")


def wallclock(fn: Callable) -> Callable:
    """Decorator that times a function and returns ``(result, elapsed_seconds)``.

    The wrapped function returns a tuple so that callers can transparently
    access both the original result and the wall-clock duration.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Tuple[Any, float]:
        timer = Timer().start()
        result = fn(*args, **kwargs)
        timer.stop()
        return result, timer.elapsed

    wrapper.__name__ = getattr(fn, "__name__", "wrapped")
    wrapper.__doc__ = getattr(fn, "__doc__", None)
    return wrapper


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

def symmetrize(a: jnp.ndarray, jitter: float = 0.0) -> jnp.ndarray:
    """Symmetrize a square matrix and optionally add diagonal jitter.

    This mirrors the stabilizers used throughout the BaM codebase.
    """
    a = jnp.asarray(a)
    out = 0.5 * (a + a.T)
    if jitter:
        out = out + jitter * jnp.eye(a.shape[-1], dtype=a.dtype)
    return out


def sym_inv_sqrt(a: jnp.ndarray, jitter: float = 1e-10) -> jnp.ndarray:
    """Compute the symmetric inverse square root of a PSD matrix.

    Returns ``S`` such that ``S @ a @ S = I`` for invertible ``a``.
    """
    a = symmetrize(jnp.asarray(a), jitter=jitter)
    evals, evecs = jnp.linalg.eigh(a)
    evals = jnp.clip(evals, a_min=jitter)
    return (evecs * (1.0 / jnp.sqrt(evals))[None, :]) @ evecs.T


def logdet(a: jnp.ndarray, jitter: float = 1e-10) -> jnp.ndarray:
    """Log absolute determinant of a symmetric positive semidefinite matrix."""
    a = symmetrize(jnp.asarray(a), jitter=jitter)
    evals = jnp.linalg.eigvalsh(a)
    return jnp.sum(jnp.log(jnp.clip(evals, a_min=jitter)))


# ---------------------------------------------------------------------------
# Gaussian divergences
# ---------------------------------------------------------------------------

def kl_forward_gaussian(
    mu0: jnp.ndarray,
    Sigma0: jnp.ndarray,
    mu1: jnp.ndarray,
    Sigma1: jnp.ndarray,
    jitter: float = 1e-10,
) -> jnp.ndarray:
    """Forward KL ``KL(N(mu0, Sigma0) || N(mu1, Sigma1))``.

    This is the same closed form used in :mod:`bam.divergences`, reimplemented
    here so :mod:`bam.utils` can be imported independently.
    """
    mu0 = jnp.asarray(mu0)
    mu1 = jnp.asarray(mu1)
    Sigma0 = symmetrize(jnp.asarray(Sigma0), jitter=jitter)
    Sigma1 = symmetrize(jnp.asarray(Sigma1), jitter=jitter)

    d = mu0.shape[-1]
    Sigma1_inv = jnp.linalg.inv(Sigma1)
    diff = mu1 - mu0
    return 0.5 * (
        jnp.trace(Sigma1_inv @ Sigma0)
        + diff @ Sigma1_inv @ diff
        - d
        + logdet(Sigma1, jitter=jitter)
        - logdet(Sigma0, jitter=jitter)
    )


def kl_reverse_gaussian(
    mu0: jnp.ndarray,
    Sigma0: jnp.ndarray,
    mu1: jnp.ndarray,
    Sigma1: jnp.ndarray,
    jitter: float = 1e-10,
) -> jnp.ndarray:
    """Reverse KL ``KL(N(mu1, Sigma1) || N(mu0, Sigma0))``.

    The argument order is intentionally the same as ``kl_forward_gaussian`` for
    convenience: the first two arguments define the first Gaussian and the last
    two define the second Gaussian.
    """
    return kl_forward_gaussian(mu1, Sigma1, mu0, Sigma0, jitter=jitter)


def forward_kl_gaussian(
    mu_q: jnp.ndarray,
    Sigma_q: jnp.ndarray,
    mu_p: jnp.ndarray,
    Sigma_p: jnp.ndarray,
    jitter: float = 1e-10,
) -> jnp.ndarray:
    """Alias for :func:`kl_forward_gaussian` using ``q``/``p`` names."""
    return kl_forward_gaussian(mu_q, Sigma_q, mu_p, Sigma_p, jitter=jitter)


def reverse_kl_gaussian(
    mu_q: jnp.ndarray,
    Sigma_q: jnp.ndarray,
    mu_p: jnp.ndarray,
    Sigma_p: jnp.ndarray,
    jitter: float = 1e-10,
) -> jnp.ndarray:
    """Alias for :func:`kl_reverse_gaussian` using ``q``/``p`` names."""
    return kl_reverse_gaussian(mu_q, Sigma_q, mu_p, Sigma_p, jitter=jitter)


def gaussian_kl(
    mu_q: jnp.ndarray,
    Sigma_q: jnp.ndarray,
    mu_p: jnp.ndarray,
    Sigma_p: jnp.ndarray,
    *,
    reverse: bool = False,
    jitter: float = 1e-10,
) -> jnp.ndarray:
    """Return forward or reverse Gaussian KL in one convenience function."""
    if reverse:
        return kl_reverse_gaussian(mu_q, Sigma_q, mu_p, Sigma_p, jitter=jitter)
    return kl_forward_gaussian(mu_q, Sigma_q, mu_p, Sigma_p, jitter=jitter)


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

def normalized_mean_error(
    mu: jnp.ndarray,
    mu_star: jnp.ndarray,
    Sigma_star: jnp.ndarray,
    jitter: float = 1e-10,
) -> jnp.ndarray:
    """Normalized mean error ``|| Sigma_*^{-1/2} (mu - mu_*) ||``.

    This is the scalar diagnostic used to verify Theorem 1's convergence bound.
    """
    inv_sqrt = sym_inv_sqrt(Sigma_star, jitter=jitter)
    diff = jnp.asarray(mu) - jnp.asarray(mu_star)
    return jnp.linalg.norm(inv_sqrt @ diff)


def normalized_covariance_error(
    Sigma: jnp.ndarray,
    Sigma_star: jnp.ndarray,
    jitter: float = 1e-10,
    ord: Optional[Union[int, str]] = "fro",
) -> jnp.ndarray:
    """Normalized covariance error ``|| Sigma_*^{-1/2} Sigma Sigma_*^{-1/2} - I ||``."""
    inv_sqrt = sym_inv_sqrt(Sigma_star, jitter=jitter)
    centered = inv_sqrt @ symmetrize(Sigma, jitter=jitter) @ inv_sqrt
    d = centered.shape[-1]
    err = centered - jnp.eye(d, dtype=centered.dtype)
    if ord == "fro":
        return jnp.linalg.norm(err, ord="fro")
    return jnp.linalg.norm(err, ord=ord)


def mean_sd_from_draws(draws: jnp.ndarray, axis: int = 0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return empirical mean and standard deviation from posterior draws."""
    draws = jnp.asarray(draws)
    return jnp.mean(draws, axis=axis), jnp.std(draws, axis=axis)


def relative_mean_error(
    mu: jnp.ndarray,
    reference_draws: jnp.ndarray,
    *,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """Relative posterior mean error with respect to reference draws.

    The error is ``||mu - mean(reference)|| / (||mean(reference)|| + eps)``.
    """
    mu = jnp.asarray(mu)
    ref = jnp.asarray(reference_draws)
    ref_mean = jnp.mean(ref, axis=0)
    denom = jnp.linalg.norm(ref_mean) + eps
    return jnp.linalg.norm(mu - ref_mean) / denom


def relative_sd_error(
    sd: jnp.ndarray,
    reference_draws: jnp.ndarray,
    *,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """Relative posterior standard-deviation error with respect to reference draws.

    The error is ``||sd - std(reference)|| / (||std(reference)|| + eps)``.
    """
    sd = jnp.asarray(sd)
    ref = jnp.asarray(reference_draws)
    ref_sd = jnp.std(ref, axis=0)
    denom = jnp.linalg.norm(ref_sd) + eps
    return jnp.linalg.norm(sd - ref_sd) / denom


def posterior_error_summary(
    mu: jnp.ndarray,
    sd: jnp.ndarray,
    reference_draws: jnp.ndarray,
    *,
    eps: float = 1e-12,
) -> Dict[str, jnp.ndarray]:
    """Return a small dictionary with posterior relative mean and SD errors."""
    return {
        "relative_mean_error": relative_mean_error(mu, reference_draws, eps=eps),
        "relative_sd_error": relative_sd_error(sd, reference_draws, eps=eps),
    }


# ---------------------------------------------------------------------------
# File and plotting helpers
# ---------------------------------------------------------------------------

def ensure_directory(path: Union[str, os.PathLike]) -> Path:
    """Create ``path`` as a directory if needed and return it as ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def as_numpy(x: Any) -> np.ndarray:
    """Convert a JAX array or array-like object to a NumPy array."""
    return np.asarray(jax.device_get(x))


def mean_sem(
    values: Sequence[Union[float, int, np.ndarray, jnp.ndarray]],
    axis: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(mean, standard error of the mean)`` for a sequence of values.

    If ``values`` is a sequence of scalars, the returned arrays are 0-dimensional
    NumPy arrays.
    """
    arr = np.asarray([as_numpy(v) for v in values])
    mean = np.mean(arr, axis=axis)
    std = np.std(arr, axis=axis)
    n = arr.shape[axis] if arr.ndim > 0 else len(arr)
    sem = std / np.sqrt(max(n, 1))
    return mean, sem


def set_plot_style(*, seaborn: bool = False) -> None:
    """Apply a consistent Matplotlib style for paper-style plots.

    Matplotlib is imported lazily so this module remains lightweight.
    """
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - only used for plotting
        raise ImportError("Matplotlib is required for plotting helpers.") from exc

    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    if seaborn:
        try:
            import seaborn as sns  # type: ignore

            sns.set_theme(style="whitegrid")
        except Exception:
            pass


def new_figure(
    figsize: Tuple[float, float] = (6.5, 4.0),
    nrows: int = 1,
    ncols: int = 1,
    **subplot_kwargs: Any,
) -> Tuple[Any, Any]:
    """Create a new Matplotlib figure and axes, applying default style lazily."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise ImportError("Matplotlib is required for plotting helpers.") from exc

    set_plot_style()
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **subplot_kwargs)
    return fig, axes


def save_figure(
    fig: Any,
    path: Union[str, os.PathLike],
    *,
    dpi: int = 300,
    bbox_inches: str = "tight",
    **kwargs: Any,
) -> Path:
    """Save a Matplotlib figure to ``path``, creating parent directories."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise ImportError("Matplotlib is required for plotting helpers.") from exc

    out = Path(path)
    ensure_directory(out.parent)
    fig.savefig(out, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
    plt.close(fig)
    return out


def plot_mean_sem(
    ax: Any,
    x: Sequence[Union[float, int]],
    values: Sequence[Sequence[Union[float, int, np.ndarray, jnp.ndarray]]],
    *,
    label: Optional[str] = None,
    color: Optional[str] = None,
    marker: Optional[str] = None,
    linestyle: Optional[str] = "-",
    linewidth: float = 1.6,
    alpha: float = 0.2,
    **kwargs: Any,
) -> None:
    """Plot mean curves with shaded standard-error bands on ``ax``.

    Parameters
    ----------
    ax:
        Matplotlib axes.
    x:
        One-dimensional sequence of x locations.
    values:
        Sequence of repetitions; ``values[i]`` should contain one or more
        observations at ``x[i]``.  Scalars, arrays, and nested sequences are
        accepted.
    """
    x_arr = np.asarray(x, dtype=float)
    means = []
    sems = []
    for rep in values:
        m, s = mean_sem(rep)
        means.append(float(np.asarray(m).squeeze()))
        sems.append(float(np.asarray(s).squeeze()))
    means_arr = np.asarray(means)
    sems_arr = np.asarray(sems)

    plot_kwargs: Dict[str, Any] = {
        "linewidth": linewidth,
        "linestyle": linestyle,
    }
    if label is not None:
        plot_kwargs["label"] = label
    if color is not None:
        plot_kwargs["color"] = color
    if marker is not None:
        plot_kwargs["marker"] = marker
    plot_kwargs.update(kwargs)

    ax.plot(x_arr, means_arr, **plot_kwargs)
    fill_kwargs = {"alpha": alpha}
    if color is not None:
        fill_kwargs["color"] = color
    ax.fill_between(x_arr, means_arr - sems_arr, means_arr + sems_arr, **fill_kwargs)
