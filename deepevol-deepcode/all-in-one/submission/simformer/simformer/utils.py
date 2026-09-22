"""Utility functions for Simformer.

Contents
--------
* random Gaussian Fourier features used to embed the diffusion time ``t``
  (128-dimensional, see Appendix Sec. A2.1) and the index set of
  function-valued parameters (Sec. 3.1).
* small diagonal Gaussian / Student-t distributions used throughout the
  code base (tasks, reference distributions, evaluation).
* score <-> epsilon conversions together with the algebra that links the
  learned score to the noise-prediction parameterisation used in training.

The paper uses JAX as backbone (Bradbury et al., 2018).  This reproduction
implements the *identical* mathematics on top of a framework agnostic numpy
core with a PyTorch neural-network backend (see ``transformer.py``); the
numerical definitions below are shared by both parts.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Random Fourier features
# ---------------------------------------------------------------------------


class GaussianFourierFeatures:
    """Random Gaussian Fourier embedding of a scalar (the diffusion time).

    The embedding has ``out_dim`` dimensions (``out_dim`` must be even) and is

    .. math::
        \\gamma(t) = \\big[\\sin(2\\pi w t), \\cos(2\\pi w t)\\big],
        \\qquad w \\sim \\mathcal{N}(0, \\sigma^{2})

    with ``sigma = scale``.  The frequencies ``w`` are *fixed* random
    directions drawn once at construction time (as done in Song et al. (2021b)
    / the official Simformer implementation).
    """

    def __init__(self, out_dim: int = 128, scale: float = 16.0, seed: int = 0):
        if out_dim % 2 != 0:
            raise ValueError("out_dim must be even")
        self.out_dim = out_dim
        self.scale = scale
        rng = np.random.default_rng(seed)
        self.frequencies = rng.normal(0.0, scale, size=(out_dim // 2,)).astype(np.float64)

    def __call__(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64).reshape(-1)
        phase = 2.0 * np.pi * t[:, None] * self.frequencies[None, :]
        return np.concatenate([np.sin(phase), np.cos(phase)], axis=-1)

    @property
    def dimension(self) -> int:
        return self.out_dim

    def state_dict(self) -> dict:
        return {"out_dim": self.out_dim, "scale": self.scale, "frequencies": self.frequencies}


def random_fourier_features(
    x: np.ndarray, out_dim: int = 32, lengthscale: float = 0.2, seed: int = 0
) -> np.ndarray:
    """Random Fourier embedding of an index set element (Sec. 3.1).

    Used for function-valued parameters (e.g. the time-dependent contact rate
    of the SIRD task or the population trajectories of Lotka-Volterra): the
    *node identifier* is a shared learnable embedding vector plus this
    embedding of the index (time) at which the variable is evaluated.

    Shapes: ``x`` -> ``(..., )``, output -> ``(..., out_dim)``.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    squeeze = x.ndim == 0
    x = x.reshape(-1)
    w = rng.normal(0.0, 1.0 / lengthscale, size=(out_dim // 2,))
    phase = x[:, None] * w[None, :]
    out = np.concatenate([np.sin(phase), np.cos(phase)], axis=-1)
    if squeeze:
        out = out[0]
    return out


# ---------------------------------------------------------------------------
# Probability distribution helpers (numpy)
# ---------------------------------------------------------------------------


def diag_gaussian_sample(
    mean: np.ndarray, std: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    return mean + std * rng.standard_normal(size=np.shape(mean))


def diag_gaussian_logpdf(
    x: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Log density of a diagonal Gaussian (sum over the last axis)."""
    var = np.square(std)
    return np.sum(
        -0.5 * np.log(2.0 * np.pi * var) - 0.5 * np.square(x - mean) / var, axis=-1
    )


def student_t_logpdf(
    x: np.ndarray, df: float, loc: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    z = (x - loc) / scale
    return np.sum(
        math.lgamma((df + 1.0) / 2.0)
        - math.lgamma(df / 2.0)
        - 0.5 * math.log(df * math.pi)
        - np.log(scale)
        - ((df + 1.0) / 2.0) * np.log1p(np.square(z) / df),
        axis=-1,
    )


def log_uniform(x: float, low: float, high: float) -> float:
    if low <= x <= high:
        return -math.log(high - low)
    return -np.inf


# ---------------------------------------------------------------------------
# Score algebra
# ---------------------------------------------------------------------------


def score_from_epsilon(eps: np.ndarray, sigma_t: np.ndarray) -> np.ndarray:
    """``grad log p_t(x_t | x0) = -eps / sigma_t`` (Appendix A2.1)."""
    sigma_t = np.asarray(sigma_t)
    while sigma_t.ndim < eps.ndim:
        sigma_t = sigma_t[..., None] if sigma_t.shape and sigma_t.shape[0] == eps.shape[0] else sigma_t[..., None]
    return -eps / sigma_t


def epsilon_from_score(score: np.ndarray, sigma_t: np.ndarray) -> np.ndarray:
    return -score * sigma_t


def denoise_from_score(
    x_t: np.ndarray, score: np.ndarray, mu_t: np.ndarray, sigma_t: np.ndarray
) -> np.ndarray:
    r"""One-step (Tweedie) denoised estimate.

    .. math::
        \hat{x}_0 = (x_t + \sigma(t)^2 s(x_t, t)) / \mu(t)

    which is exactly the "Denoise" line of Algorithm 1 in Appendix A3.3.
    """
    return (x_t + np.square(sigma_t) * score) / mu_t


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def sym_mask(mask: np.ndarray) -> np.ndarray:
    """Make an attention mask undirected (Sec. 3.2: symmetrization)."""
    mask = np.asarray(mask)
    return ((mask + mask.T) > 0).astype(np.float64)


def as_mask(mask) -> np.ndarray:
    """Coerce a boolean / float mask to a float 0/1 matrix."""
    return (np.asarray(mask) > 0).astype(np.float64)


def nan_to_num(x: np.ndarray, value: float = 0.0) -> np.ndarray:
    return np.nan_to_num(x, nan=value, posinf=value, neginf=value)


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes", "y", "t")


def batch_split(array: np.ndarray, batch_size: int):
    """Yield successive slices of ``array`` along axis 0."""
    n = np.shape(array)[0]
    for start in range(0, n, batch_size):
        yield array[start : start + batch_size]
