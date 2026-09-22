"""Correlation, ranking and prediction metrics for LCA-on-the-Line.

Implements the measurement suite described in:

* Section D.1.1 ("Linearity Measurement"): the coefficient of determination
  ``R^2`` and the Pearson correlation coefficient ``PEA``;
* Section D.1.2 ("Ranking measurement"): the Kendall rank correlation
  coefficient ``KEN`` (tau) and the Spearman rank-order correlation
  coefficient ``SPE`` (rho);
* Section 4 ("Metric Setup"): MAE (Mean Absolute Error) for the prediction
  experiments;
* Section 4.2 ("Predicting OOD Performance via ID LCA"): a linear function of
  the in-distribution LCA distance is fitted to predict OOD Top-1 accuracy.
  Following the paper, the features are **min-max scaled** instead of applying
  the probit transform used by (Baek et al., 2022) and (Miller et al., 2021),
  "because LCA does not fall within the [0,1] range".

Because the LCA distance *decreases* when accuracy *increases*, the reported
correlation coefficients are sign-flipped to their absolute values
(``abs_values=True``), as done in the paper's tables.

All functions accept array-like inputs (lists, numpy arrays or torch tensors)
and are implemented in plain numpy so that the module stays importable without
scipy / sklearn (both are used when available for numerically equivalent
results).
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "EPS",
    "DEFAULT_SCALER",
    "DEFAULT_FEATURE_RANGE",
    "SCALERS",
    # scaling helpers
    "min_max_scale",
    "inverse_min_max_scale",
    "probit",
    "inverse_probit",
    "rank_data",
    "apply_scaler",
    # individual metrics
    "r2_score",
    "pearson_correlation",
    "kendall_tau",
    "spearman_rho",
    "mean_absolute_error",
    "root_mean_squared_error",
    "top1_error",
    # fitting
    "LinearFit",
    "linear_regression",
    "fit_linear",
    "fit_predict",
    "regression_report",
    # aggregate reporting
    "CorrelationResults",
    "correlation_metrics",
    "compute_correlations",
    "correlation_table",
    "correlation_table_from_dataframe",
    "format_table",
    # constants referenced by the reproduction targets
    "TABLE2_CORRELATION_TARGETS",
    "TABLE3_MAE_TARGETS",
]

EPS = 1e-12
DEFAULT_SCALER = "minmax"
DEFAULT_FEATURE_RANGE: Tuple[float, float] = (0.0, 1.0)

# ---------------------------------------------------------------------------
# Expected values from the paper (used for validation / reporting only).
# ---------------------------------------------------------------------------
# Table 2: R^2 / PEA of ID LCA distance against OOD Top-1 accuracy (75 models).
TABLE2_CORRELATION_TARGETS: Dict[str, Dict[str, float]] = {
    "v2": {"r2": 0.339, "pea": 0.582},
    "s": {"r2": 0.816, "pea": 0.903},
    "r": {"r2": 0.779, "pea": 0.883},
    "a": {"r2": 0.704, "pea": 0.839},
    "objectnet": {"r2": 0.915, "pea": 0.956},
}

# Table 3: MAE of the OOD Top-1 accuracy predictor (ID LCA vs ID Top1 baselines).
TABLE3_MAE_TARGETS: Dict[str, Dict[str, float]] = {
    "v2": {"lca": 0.162, "id_top1": 0.230},
    "s": {"lca": 0.093, "id_top1": 0.277},
    "r": {"lca": 0.114, "id_top1": 0.192},
    "a": {"lca": 0.103, "id_top1": 0.178},
    "objectnet": {"lca": 0.048},
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _as_float_array(values: Any) -> np.ndarray:
    """Convert lists / numpy arrays / torch tensors to a 1-D float64 array."""
    if values is None:
        return np.zeros(0, dtype=np.float64)
    if hasattr(values, "detach"):  # torch tensor
        values = values.detach().cpu().numpy()
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.reshape(-1)


def _paired(x: Any, y: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Return aligned finite (x, y) arrays, dropping non-finite entries."""
    xa = _as_float_array(x)
    ya = _as_float_array(y)
    if xa.shape != ya.shape:
        raise ValueError(f"x and y must have the same length ({xa.shape} vs {ya.shape})")
    mask = np.isfinite(xa) & np.isfinite(ya)
    if mask.sum() < len(mask):
        logger.debug("dropping %d non-finite pairs", int((~mask).sum()))
    return xa[mask], ya[mask]


def _sign(result: float, abs_values: bool) -> float:
    if result is None or (isinstance(result, float) and math.isnan(result)):
        return float("nan")
    return float(abs(result)) if abs_values else float(result)


# ---------------------------------------------------------------------------
# Scaling: min-max (paper's choice for LCA) and probit (accuracy baselines)
# ---------------------------------------------------------------------------
def min_max_scale(
    x: Any,
    feature_range: Tuple[float, float] = DEFAULT_FEATURE_RANGE,
    axis: Optional[int] = None,
) -> np.ndarray:
    """Sklearn-style min-max scaling onto ``feature_range`` (default ``[0, 1]``).

    ``x_min`` / ``x_max`` are the observed extremes of the array (or along
    ``axis``).  A constant input is mapped to the lower bound of the range.
    """
    arr = np.asarray(x, dtype=np.float64)
    lo, hi = float(feature_range[0]), float(feature_range[1])
    x_min = np.min(arr, axis=axis, keepdims=axis is not None)
    x_max = np.max(arr, axis=axis, keepdims=axis is not None)
    span = x_max - x_min
    span = np.where(np.abs(span) < EPS, 1.0, span)
    scaled = (arr - x_min) / span
    out = scaled * (hi - lo) + lo
    # constant rows collapse exactly onto the lower bound
    out = np.where(np.abs(x_max - x_min) < EPS, lo, out)
    return out if axis is not None else np.asarray(out, dtype=np.float64).reshape(arr.shape)


def inverse_min_max_scale(
    scaled: Any,
    x_min: float,
    x_max: float,
    feature_range: Tuple[float, float] = DEFAULT_FEATURE_RANGE,
) -> np.ndarray:
    """Undo :func:`min_max_scale` given the original extremes."""
    arr = np.asarray(scaled, dtype=np.float64)
    lo, hi = float(feature_range[0]), float(feature_range[1])
    span = hi - lo
    span = span if abs(span) > EPS else 1.0
    return (arr - lo) / span * (float(x_max) - float(x_min)) + float(x_min)


def probit(x: Any, eps: float = 1e-6) -> np.ndarray:
    """Probit (inverse normal CDF) transform used for accuracy baselines.

    Accuracy values are first clipped to ``[eps, 1 - eps]``; scipy is used when
    available and an ``erfinv``-free rational approximation otherwise.
    """
    arr = np.asarray(x, dtype=np.float64)
    arr = np.clip(arr, float(eps), 1.0 - float(eps))
    try:  # pragma: no cover - depends on scipy availability
        from scipy.stats import norm

        return norm.ppf(arr)
    except Exception:  # pragma: no cover
        # Acklam / Wichura style rational approximation of the inverse normal CDF.
        a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
             1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
        b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
             6.680131188771972e01, -1.328068155288572e01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
             -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
        d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
             3.754408661907416e00]
        p_low, p_high = 0.02425, 1.0 - 0.02425
        out = np.zeros_like(arr)
        low = arr < p_low
        high = arr > p_high
        mid = ~(low | high)
        if np.any(low):
            q = np.sqrt(-2.0 * np.log(arr[low]))
            out[low] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                       ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        if np.any(high):
            q = np.sqrt(-2.0 * np.log(1.0 - arr[high]))
            out[high] = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        if np.any(mid):
            q = arr[mid] - 0.5
            r = q * q
            out[mid] = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
                       (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        return out


def inverse_probit(z: Any, eps: float = 1e-6) -> np.ndarray:
    """Inverse of :func:`probit` (normal CDF), clipped back into ``[eps, 1-eps]``."""
    arr = np.asarray(z, dtype=np.float64)
    try:  # pragma: no cover
        from scipy.stats import norm

        out = norm.cdf(arr)
    except Exception:  # pragma: no cover
        out = 0.5 * (1.0 + np.vectorize(math.erf)(arr / math.sqrt(2.0)))
    return np.clip(out, float(eps), 1.0 - float(eps))


def apply_scaler(x: Any, scaler: str = DEFAULT_SCALER) -> np.ndarray:
    """Apply a named scaler: ``"none"``, ``"minmax"`` or ``"probit"``."""
    name = (scaler or "none").lower()
    if name in ("none", "", "identity", "raw"):
        return _as_float_array(x)
    if name in ("minmax", "min_max", "min-max"):
        return min_max_scale(x)
    if name in ("probit", "norm", "gaussian"):
        return probit(x)
    raise ValueError(f"unknown scaler '{scaler}'")


SCALERS: Dict[str, Callable[[Any], np.ndarray]] = {
    "none": lambda x: _as_float_array(x),
    "minmax": min_max_scale,
    "probit": probit,
}


def rank_data(values: Any, method: str = "average") -> np.ndarray:
    """Rank transform (ties averaged) used by the Spearman coefficient."""
    arr = _as_float_array(values)
    if arr.size == 0:
        return arr
    try:  # pragma: no cover - scipy is the preferred path
        from scipy.stats import rankdata

        return rankdata(arr, method=method).astype(np.float64)
    except Exception:  # pragma: no cover
        order = np.argsort(arr, kind="mergesort")
        ranks = np.empty(arr.size, dtype=np.float64)
        ranks[order] = np.arange(1, arr.size + 1, dtype=np.float64)
        # average tied ranks
        sorted_vals = arr[order]
        i = 0
        while i < sorted_vals.size:
            j = i + 1
            while j < sorted_vals.size and sorted_vals[j] == sorted_vals[i]:
                j += 1
            if j - i > 1:
                ranks[order[i:j]] = np.mean(ranks[order[i:j]])
            i = j
        return ranks


# ---------------------------------------------------------------------------
# Individual metrics (Section D.1.1 / D.1.2 / Section 4)
# ---------------------------------------------------------------------------
def r2_score(y_true: Any, y_pred: Any, sample_weight: Optional[Any] = None) -> float:
    """Coefficient of determination, Equation (1) of Section D.1.1.

    ``R^2 = 1 - sum_i (y_i - f(x_i))^2 / sum_i (y_i - y_bar)^2``
    """
    yt, yp = _paired(y_true, y_pred)
    n = yt.size
    if n == 0:
        return float("nan")
    if sample_weight is not None:
        w = _as_float_array(sample_weight)[:n]
        total_w = w.sum()
        if total_w <= 0:
            return float("nan")
        y_bar = float(np.sum(w * yt) / total_w)
        ss_res = float(np.sum(w * (yt - yp) ** 2))
        ss_tot = float(np.sum(w * (yt - y_bar) ** 2))
    else:
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot < EPS:
        # sklearn convention: perfect prediction -> 1.0, else 0.0
        return 1.0 if ss_res < EPS else 0.0
    return float(1.0 - ss_res / ss_tot)


def pearson_correlation(x: Any, y: Any, abs_values: bool = True) -> float:
    """Pearson correlation coefficient, Equation (2) of Section D.1.1."""
    xa, ya = _paired(x, y)
    n = xa.size
    if n < 2:
        return float("nan")
    xm = xa - np.mean(xa)
    ym = ya - np.mean(ya)
    denom = math.sqrt(float(np.sum(xm ** 2)) * float(np.sum(ym ** 2)))
    if denom < EPS:
        return float("nan")
    return _sign(float(np.sum(xm * ym) / denom), abs_values)


def kendall_tau(x: Any, y: Any, abs_values: bool = True) -> float:
    """Kendall rank correlation coefficient (tau), Section D.1.2.

    ``tau = (#concordant - #discordant) / (0.5 * n * (n - 1))``
    """
    xa, ya = _paired(x, y)
    n = xa.size
    if n < 2:
        return float("nan")
    try:  # pragma: no cover - scipy path
        from scipy.stats import kendalltau

        tau = float(kendalltau(xa, ya).correlation)
        if math.isnan(tau):
            raise ValueError("nan tau")
        return _sign(tau, abs_values)
    except Exception:
        return _sign(_kendall_tau_python(xa, ya), abs_values)


def _kendall_tau_python(xa: np.ndarray, ya: np.ndarray) -> float:
    """Tau-a computed directly from concordant / discordant pair counts."""
    n = xa.size
    concordant = 0
    discordant = 0
    for i in range(n - 1):
        dx = xa[i + 1:] - xa[i]
        dy = ya[i + 1:] - ya[i]
        prod = dx * dy
        concordant += int(np.sum(prod > 0))
        discordant += int(np.sum(prod < 0))
    denom = 0.5 * n * (n - 1)
    if denom <= 0:
        return float("nan")
    return float(concordant - discordant) / denom


def spearman_rho(x: Any, y: Any, abs_values: bool = True) -> float:
    """Spearman rank-order correlation coefficient, Section D.1.2."""
    xa, ya = _paired(x, y)
    n = xa.size
    if n < 2:
        return float("nan")
    try:  # pragma: no cover - scipy path
        from scipy.stats import spearmanr

        rho = float(spearmanr(xa, ya).correlation)
        if not math.isnan(rho):
            return _sign(rho, abs_values)
    except Exception:
        pass
    rx = rank_data(xa)
    ry = rank_data(ya)
    rho = pearson_correlation(rx, ry, abs_values=False)
    if math.isnan(rho):
        # Fall back to the textbook closed form (no ties):
        # rho = 1 - 6 * sum d_i^2 / (n * (n^2 - 1))
        d = rx - ry
        denom = n * (n ** 2 - 1)
        rho = float("nan") if denom == 0 else 1.0 - 6.0 * float(np.sum(d ** 2)) / denom
    return _sign(rho, abs_values)


def mean_absolute_error(y_true: Any, y_pred: Any) -> float:
    """MAE used by the prediction experiments (Section 4)."""
    yt, yp = _paired(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    return float(np.mean(np.abs(yt - yp)))


def root_mean_squared_error(y_true: Any, y_pred: Any) -> float:
    yt, yp = _paired(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    return float(math.sqrt(float(np.mean((yt - yp) ** 2))))


def top1_error(accuracy: Any) -> np.ndarray:
    """``1 - accuracy`` (the error scale used by Table 7 / Figure 5)."""
    return 1.0 - _as_float_array(accuracy)


# ---------------------------------------------------------------------------
# Linear fit with min-max (or probit) feature scaling
# ---------------------------------------------------------------------------
@dataclass
class LinearFit:
    """An ordinary-least-squares line fitted on a *scaled* feature.

    ``y ~ slope * scaler(x) + intercept`` -- equivalently, using the paper's
    min-max scaling, ``y ~ a * (x - x_min) / (x_max - x_min) + b``.
    """

    slope: float = float("nan")
    intercept: float = float("nan")
    scaler: str = DEFAULT_SCALER
    feature_range: Tuple[float, float] = DEFAULT_FEATURE_RANGE
    x_min: float = 0.0
    x_max: float = 1.0
    n: int = 0
    target: str = ""

    # -- scaling ---------------------------------------------------------
    def transform(self, x: Any) -> np.ndarray:
        arr = _as_float_array(x)
        name = (self.scaler or "none").lower()
        if name in ("minmax", "min_max", "min-max"):
            span = self.x_max - self.x_min
            span = span if abs(span) > EPS else 1.0
            lo, hi = self.feature_range
            return (arr - self.x_min) / span * (hi - lo) + lo
        if name in ("probit", "norm", "gaussian"):
            return probit(arr)
        return arr

    def predict(self, x: Any) -> np.ndarray:
        """Predict the target in its original units."""
        return self.slope * self.transform(x) + self.intercept

    def predict_scaled(self, x: Any) -> np.ndarray:
        """Predict from already-scaled features."""
        return self.slope * _as_float_array(x) + self.intercept

    def predict_accuracy(self, x: Any) -> np.ndarray:
        """Inverse-probit the prediction (only meaningful for probit fits)."""
        return inverse_probit(self.predict(x))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _fit_scalar_fit(
    xs: np.ndarray,
    ys: np.ndarray,
    scaler: str,
    feature_range: Tuple[float, float],
    x_min: float,
    x_max: float,
    target: str = "",
) -> LinearFit:
    n = int(xs.size)
    if n < 2 or float(np.var(xs)) < EPS:
        intercept = float(np.mean(ys)) if ys.size else float("nan")
        return LinearFit(
            slope=0.0,
            intercept=intercept,
            scaler=scaler,
            feature_range=feature_range,
            x_min=x_min,
            x_max=x_max,
            n=n,
            target=target,
        )
    slope, intercept = np.polyfit(xs, ys, 1)
    return LinearFit(
        slope=float(slope),
        intercept=float(intercept),
        scaler=scaler,
        feature_range=feature_range,
        x_min=x_min,
        x_max=x_max,
        n=n,
        target=target,
    )


def linear_regression(
    x: Any,
    y: Any,
    scaler: str = DEFAULT_SCALER,
    feature_range: Tuple[float, float] = DEFAULT_FEATURE_RANGE,
    target: str = "",
) -> LinearFit:
    """Fit ``y = slope * scaler(x) + intercept`` by ordinary least squares.

    The paper min-max scales the LCA feature (``scaler="minmax"``); accuracy
    baselines use ``scaler="probit"`` with the min-max variant kept for
    reference (Section 4.2).
    """
    xa, ya = _paired(x, y)
    name = (scaler or "none").lower()
    if name in ("minmax", "min_max", "min-max"):
        x_min = float(np.min(xa)) if xa.size else 0.0
        x_max = float(np.max(xa)) if xa.size else 1.0
        span = x_max - x_min
        span = span if abs(span) > EPS else 1.0
        lo, hi = feature_range
        xs = (xa - x_min) / span * (hi - lo) + lo
        return _fit_scalar_fit(xs, ya, "minmax", feature_range, x_min, x_max, target)
    if name in ("probit", "norm", "gaussian"):
        xs = probit(xa)
        return _fit_scalar_fit(xs, ya, "probit", feature_range, 0.0, 1.0, target)
    xs = xa.astype(np.float64)
    return _fit_scalar_fit(xs, ya, "none", feature_range, 0.0, 1.0, target)


# Convenience aliases used by the scripts / OOD prediction module.
fit_linear = linear_regression


def fit_predict(
    x: Any,
    y: Any,
    scaler: str = DEFAULT_SCALER,
    target_transform: str = "identity",
) -> Dict[str, Any]:
    """Fit a linear model and report MAE / RMSE / R^2 / PEA on the training pairs.

    ``target_transform``:
      * ``"identity"`` -- predict the target as given (OOD Top-1 accuracy);
      * ``"error"``    -- fit/predict ``1 - y`` (OOD Top-1 error, Table 7);
      * ``"log"``      -- fit/predict ``log(y)``.
    """
    xa, ya = _paired(x, y)
    transform, inverse = _target_transform(target_transform)
    yt = transform(ya)
    fit = linear_regression(xa, yt, scaler=scaler, target=target_transform)
    pred_t = fit.predict(xa)
    pred = inverse(pred_t)
    return {
        "fit": fit,
        "n": int(xa.size),
        "scaler": (scaler or "none").lower(),
        "target_transform": target_transform,
        "slope": fit.slope,
        "intercept": fit.intercept,
        "x_min": fit.x_min,
        "x_max": fit.x_max,
        "predictions": pred,
        "predictions_fit_space": pred_t,
        "residuals": ya - pred,
        "mae": mean_absolute_error(ya, pred),
        "rmse": root_mean_squared_error(ya, pred),
        "r2": r2_score(ya, pred),
        "pea": pearson_correlation(ya, pred),
        "x": xa,
        "y": ya,
    }


def _target_transform(
    name: str,
    eps: float = 1e-6,
) -> Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    key = (name or "identity").lower()
    if key in ("identity", "none", "", "accuracy", "acc"):
        return (lambda v: v), (lambda v: v)
    if key in ("error", "err", "top1_error", "1-acc"):
        return (lambda v: 1.0 - v), (lambda v: 1.0 - v)
    if key == "log":
        return (lambda v: np.log(np.clip(v, eps, None))), (lambda v: np.exp(v))
    if key in ("logit", "probit"):
        return probit, inverse_probit
    raise ValueError(f"unknown target_transform '{name}'")


def regression_report(
    x: Any,
    y: Any,
    scaler: str = DEFAULT_SCALER,
    target_transform: str = "identity",
) -> Dict[str, Any]:
    """Alias of :func:`fit_predict` (kept for readability in scripts)."""
    return fit_predict(x, y, scaler=scaler, target_transform=target_transform)


# ---------------------------------------------------------------------------
# Aggregate reporting
# ---------------------------------------------------------------------------
@dataclass
class CorrelationResults:
    """Container for one (x, y) correlation measurement block."""

    n: int = 0
    r2: float = float("nan")
    pea: float = float("nan")
    ken: float = float("nan")
    spe: float = float("nan")
    mae: float = float("nan")
    slope: float = float("nan")
    intercept: float = float("nan")
    scaler: str = DEFAULT_SCALER
    x_name: str = "x"
    y_name: str = "y"

    def asdict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_row(self) -> Dict[str, float]:
        return {
            "R^2": self.r2,
            "PEA": self.pea,
            "KEN": self.ken,
            "SPE": self.spe,
            "MAE": self.mae,
            "Slope": self.slope,
            "Intercept": self.intercept,
            "N": self.n,
        }

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{self.x_name} vs {self.y_name}: R2={self.r2:.3f} PEA={self.pea:.3f} "
            f"KEN={self.ken:.3f} SPE={self.spe:.3f} MAE={self.mae:.4f} (n={self.n})"
        )


def correlation_metrics(
    x: Any,
    y: Any,
    abs_values: bool = True,
    scaler: str = DEFAULT_SCALER,
    x_name: str = "x",
    y_name: str = "y",
    fit_target: Optional[Any] = None,
) -> Dict[str, float]:
    """Compute the full metric block (R^2, PEA, KEN, SPE, MAE, slope, intercept).

    ``fit_target`` optionally overrides the target used for the linear fit
    (e.g. OOD error instead of OOD accuracy) while correlations are computed
    against ``y``.
    """
    xa, ya = _paired(x, y)
    if xa.size < 2:
        return CorrelationResults(n=int(xa.size), scaler=scaler, x_name=x_name, y_name=y_name).to_row()
    target = ya if fit_target is None else _as_float_array(fit_target)[: xa.size]
    fit = linear_regression(xa, target, scaler=scaler, target=x_name)
    pred = fit.predict(xa)
    res = CorrelationResults(
        n=int(xa.size),
        r2=_sign(r2_score(ya, pred) if fit_target is None else r2_score(ya, pred), False),
        pea=pearson_correlation(xa, ya, abs_values=abs_values),
        ken=kendall_tau(xa, ya, abs_values=abs_values),
        spe=spearman_rho(xa, ya, abs_values=abs_values),
        mae=mean_absolute_error(target, pred),
        slope=fit.slope,
        intercept=fit.intercept,
        scaler=(scaler or "none").lower(),
        x_name=x_name,
        y_name=y_name,
    )
    # R^2 is a variance-explained fraction and is reported unsigned.
    res.r2 = float(abs(res.r2)) if not math.isnan(res.r2) else res.r2
    return res.to_row()


compute_correlations = correlation_metrics


def correlation_table(
    metrics: Mapping[str, Mapping[str, float]],
    x_key: str,
    y_keys: Sequence[str],
    abs_values: bool = True,
    scaler: str = DEFAULT_SCALER,
    fit_on_error: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Build a Table-2-style table: one correlation block per target key.

    ``metrics`` maps model name -> {metric_key: value}.  ``x_key`` is the
    predictor (e.g. ``"id_lca"``) and ``y_keys`` the targets (e.g. the OOD
    Top-1 accuracies).  ``fit_on_error`` fits ``1 - y`` (paper Figure 5 uses
    error space) while reporting the correlations of ``y``.
    """
    names = list(metrics.keys())
    if not names:
        return {}
    if x_key not in next(iter(metrics.values())):
        raise KeyError(f"predictor '{x_key}' not present in metrics table")
    xs = np.array([float(metrics[n].get(x_key, np.nan)) for n in names], dtype=np.float64)
    table: Dict[str, Dict[str, float]] = {}
    for key in y_keys:
        if key not in next(iter(metrics.values())):
            logger.warning("target '%s' missing from metrics table -- skipped", key)
            continue
        ys = np.array([float(metrics[n].get(key, np.nan)) for n in names], dtype=np.float64)
        fit_target = (1.0 - ys) if fit_on_error else None
        table[key] = correlation_metrics(
            xs,
            ys,
            abs_values=abs_values,
            scaler=scaler,
            x_name=x_key,
            y_name=key,
            fit_target=fit_target,
        )
    return table


def correlation_table_from_dataframe(
    df: Any,
    x_key: str,
    y_keys: Sequence[str],
    abs_values: bool = True,
    scaler: str = DEFAULT_SCALER,
    fit_on_error: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Same as :func:`correlation_table` for a pandas ``DataFrame``."""
    if hasattr(df, "to_dict"):
        try:
            records = df.to_dict(orient="records")
        except TypeError:  # pragma: no cover - non-pandas mapping
            records = dict(df)
    else:
        records = dict(df)
    if isinstance(records, dict):
        return correlation_table(records, x_key=x_key, y_keys=y_keys,
                                 abs_values=abs_values, scaler=scaler,
                                 fit_on_error=fit_on_error)
    table: Dict[str, Dict[str, float]] = {}
    xs = np.asarray([float(r.get(x_key, np.nan)) for r in records], dtype=np.float64)
    for key in y_keys:
        ys = np.asarray([float(r.get(key, np.nan)) for r in records], dtype=np.float64)
        table[key] = correlation_metrics(
            xs, ys, abs_values=abs_values, scaler=scaler,
            x_name=x_key, y_name=key,
            fit_target=(1.0 - ys) if fit_on_error else None,
        )
    return table


def format_table(
    table: Mapping[str, Mapping[str, float]],
    columns: Sequence[str] = ("r2", "pea", "ken", "spe", "mae"),
    title: str = "",
) -> str:
    """Render a correlation table as a plain-text (markdown-ish) block."""
    cols = [c.lower() for c in columns]
    header_map = {"r2": "R^2", "pea": "PEA", "ken": "KEN", "spe": "SPE", "mae": "MAE",
                  "slope": "Slope", "intercept": "Intercept", "n": "N"}
    lines: List[str] = []
    if title:
        lines.append(title)
    lines.append("| target | " + " | ".join(header_map.get(c, c.upper()) for c in cols) + " |")
    lines.append("|" + "---|" * (len(cols) + 1))
    for key, row in table.items():
        vals = []
        for c in cols:
            v = row.get(c, float("nan"))
            vals.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        lines.append(f"| {key} | " + " | ".join(vals) + " |")
    return "\n".join(lines)
