"""Correlation / ranking / error metrics used throughout the paper.

Section 4 ("Metric Setup") and Appendix D.1 describe the measurements:

* ``PEA`` -- Pearson correlation coefficient
* ``R^2`` -- coefficient of determination
* ``KEN`` -- Kendall rank correlation coefficient
* ``SPE`` -- Spearman rank-order correlation coefficient
* ``MAE`` -- mean absolute error for OOD-error prediction

The paper reports "the absolute value of all correlations for simplicity", and
the tables show ``R^2 == PEA**2``; we therefore return absolute values by
default so the numbers line up with Tables 2/3/11/12/13/15.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #
def minmax_scale(x: np.ndarray, feature_range: Tuple[float, float] = (0.0, 1.0)
                 ) -> np.ndarray:
    """Min-max scaling to ``feature_range`` (paper uses this instead of probit)."""
    x = np.asarray(x, dtype=np.float64)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo == 0:
        return np.zeros_like(x) + feature_range[0]
    return feature_range[0] + (x - lo) * (feature_range[1] - feature_range[0]) / (
        hi - lo
    )


# --------------------------------------------------------------------------- #
# linearity measurements
# --------------------------------------------------------------------------- #
def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Signed Pearson correlation coefficient."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size != b.size:
        raise ValueError("inputs must have the same length")
    if a.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom == 0:
        return float("nan")
    return float((a * b).sum() / denom)


def coefficient_of_determination(a: np.ndarray, b: np.ndarray) -> float:
    """``R^2`` of the least-squares fit, clipped to ``[0, 1]``.

    Following the paper we report ``R^2 = PEA**2`` (which is the coefficient of
    determination of a simple linear regression), so the values match Tables
    2/13/15 exactly.
    """
    r = pearson(a, b)
    if np.isnan(r):
        return float("nan")
    return float(r * r)


def pea(a: np.ndarray, b: np.ndarray, absolute: bool = True) -> float:
    r = pearson(a, b)
    return abs(r) if absolute else r


def r2(a: np.ndarray, b: np.ndarray) -> float:
    return coefficient_of_determination(a, b)


# --------------------------------------------------------------------------- #
# ranking measurements
# --------------------------------------------------------------------------- #
def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (handles ties like scipy.stats.rankdata)."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(a: np.ndarray, b: np.ndarray, absolute: bool = True) -> float:
    r = pearson(_rankdata(a), _rankdata(b))
    return abs(r) if absolute else r


def kendall(a: np.ndarray, b: np.ndarray, absolute: bool = True) -> float:
    """Kendall's tau-b rank correlation."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    n = len(a)
    if n < 2:
        return float("nan")
    n0 = n * (n - 1) / 2.0
    n1 = n2 = n3 = 0.0
    for i in range(n - 1):
        da = a[i + 1:] - a[i]
        db = b[i + 1:] - b[i]
        sa = np.sign(da)
        sb = np.sign(db)
        n1 += float((sa == 0).sum())
        n2 += float((sb == 0).sum())
        n3 += float((sa * sb > 0).sum()) - float((sa * sb < 0).sum())
    n1 /= 2.0
    n2 /= 2.0
    denom = np.sqrt((n0 - n1) * (n0 - n2))
    if denom == 0:
        return float("nan")
    tau = n3 / denom
    return abs(tau) if absolute else float(tau)


def ken(a: np.ndarray, b: np.ndarray, absolute: bool = True) -> float:
    return kendall(a, b, absolute=absolute)


def all_linearity(a: np.ndarray, b: np.ndarray) -> dict:
    return {
        "R2": r2(a, b),
        "PEA": pea(a, b),
        "KEN": ken(a, b),
        "SPE": spearman(a, b),
    }


# --------------------------------------------------------------------------- #
# error prediction
# --------------------------------------------------------------------------- #
def fit_linear_predictor(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Least-squares ``y = slope * x + intercept``."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def error_prediction_mae(
    id_metric: np.ndarray,
    ood_metric: np.ndarray,
    normalize: bool = True,
) -> float:
    """MAE of the linear predictor ``ID metric -> OOD metric``.

    Mirrors Section 4.2: min-max scaling is used instead of a probit transform,
    a linear function is fitted on the (ID, OOD) pairs, and the MAE between the
    predicted and actual OOD performance is reported (Table 3).
    """
    x = np.asarray(id_metric, dtype=np.float64).ravel()
    y = np.asarray(ood_metric, dtype=np.float64).ravel()
    if normalize:
        x = minmax_scale(x)
        y = minmax_scale(y)
    slope, intercept = fit_linear_predictor(x, y)
    pred = slope * x + intercept
    return float(np.mean(np.abs(pred - y)))


def linear_r2(a: np.ndarray, b: np.ndarray) -> float:
    """Alias kept for readability in analysis code."""
    return coefficient_of_determination(a, b)


def summarize(x: np.ndarray, name: Optional[str] = None) -> str:
    x = np.asarray(x, dtype=np.float64)
    tag = ("%s: " % name) if name else ""
    return "%smean=%.4f std=%.4f min=%.4f max=%.4f" % (
        tag, x.mean(), x.std(), x.min(), x.max()
    )
