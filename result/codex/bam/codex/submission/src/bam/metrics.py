"""Evaluation metrics used in the experiments.

* forward / reverse KL divergences to a known target (Section 5.1),
* relative mean and standard deviation errors with respect to HMC reference
  samples (Section 5.2 and Figure E.6), and
* image reconstruction error for the deep generative model (Section 5.3).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def relative_mean_error(mean_hat: np.ndarray, ref_mean: np.ndarray, ref_sd: np.ndarray) -> float:
    """``|| (mu_hat - mu) / sigma ||_2`` (Appendix E.5)."""
    mean_hat = np.asarray(mean_hat, dtype=np.float64).ravel()
    ref_mean = np.asarray(ref_mean, dtype=np.float64).ravel()
    ref_sd = np.asarray(ref_sd, dtype=np.float64).ravel()
    return float(np.linalg.norm((mean_hat - ref_mean) / ref_sd))


def relative_sd_error(sd_hat: np.ndarray, ref_sd: np.ndarray) -> float:
    """``|| (sigma_hat - sigma) / sigma ||_2`` (Appendix E.5)."""
    sd_hat = np.asarray(sd_hat, dtype=np.float64).ravel()
    ref_sd = np.asarray(ref_sd, dtype=np.float64).ravel()
    return float(np.linalg.norm((sd_hat - ref_sd) / ref_sd))


def reconstruction_mse(x: np.ndarray, x_hat: np.ndarray, sigma2: float = 0.1) -> float:
    """Mean squared reconstruction error ||x - Omega(E[z])||^2 / D (Section 5.3)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    x_hat = np.asarray(x_hat, dtype=np.float64).ravel()
    return float(np.mean((x - x_hat) ** 2))


def summarize_runs(values: np.ndarray, axis: int = 0) -> Dict[str, np.ndarray]:
    """Mean and standard error of a set of independent runs."""
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[axis]
    return {
        "mean": values.mean(axis=axis),
        "se": values.std(axis=axis, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(values.mean(axis=axis)),
    }
