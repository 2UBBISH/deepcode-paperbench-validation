"""Expected-coverage / calibration evaluation for Simformer posterior approximations.

The paper (Sec. 4.1) reports that "Simformer is well calibrated (Appendix Fig. A9,
Fig. A10, Fig. A11, Fig. A12)" and Appendix Sec. A3.1 states that calibration and
log-likelihood analyses are provided as *additional* metrics on top of the primary
C2ST evaluation.

This module implements the standard **expected coverage** (a.k.a. empirical
coverage / calibration) diagnostic for approximate posteriors:

* Choose a set of nominal credibility levels ``alpha`` in ``(0, 1)``
  (e.g. 0.05, 0.10, ..., 0.95).
* Build the highest-density region (HDR) of the *approximate* posterior that
  contains a fraction ``alpha`` of its probability mass.
* Measure the fraction of samples drawn from the *ground truth* posterior
  (obtained from the MCMC reference samplers in ``simformer.reference.mcmc``)
  that fall inside that region.  A perfectly calibrated approximation yields
  ``empirical coverage == alpha`` for every ``alpha``.

Because reference samples are generally available only as *samples* (not as
densities), the HDR is estimated non-parametrically:

* ``method="knn"`` (default): a k-nearest-neighbour log-density proxy
  ``s(x) = -mean_{j<=k} ||x - x_j||`` w.r.t. the approximate samples;
* ``method="mahalanobis"``: Gaussian HDR using the Mahalanobis distance to the
  approximate posterior mean/covariance;
* ``method="gaussian"``: plug-in Gaussian log density (equivalent to the
  Mahalanobis variant up to a constant, kept for readability);
* ``method="logprob"``: exact HDR using a user supplied log-density callable
  (available for e.g. the Gaussian Linear and SLCP tasks, which have closed form
  posteriors);
* ``method="dimension"``: marginal (per-coordinate) central-interval coverage,
  which is what the per-dimension calibration panels in Figs. A9-A12 display.

All routines are pure NumPy (an optional ``scipy.spatial.cKDTree`` accelerates the
k-NN queries), so they compose directly with the NumPy reference samplers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union
import math

import numpy as np

try:  # pragma: no cover - optional acceleration
    from scipy.spatial import cKDTree  # type: ignore

    _HAS_CKDTree = True
except Exception:  # pragma: no cover
    cKDTree = None  # type: ignore
    _HAS_CKDTree = False


__all__ = [
    # configuration / results
    "CoverageConfig",
    "CoverageCurve",
    # core estimators
    "knn_scores",
    "negative_knn_distance",
    "mahalanobis_scores",
    "gaussian_log_density_scores",
    "hdr_threshold",
    "hdr_mask",
    "coverage_curve",
    "expected_coverage",
    "expected_coverage_curve",
    "calibration_curve",
    "coverage_curve_from_logprob",
    "logprob_coverage_curve",
    "dimension_coverage",
    "marginal_coverage_curve",
    "calibration_error",
    "coverage_error",
    "coverage_score",
    "check_coverage",
    "check_calibration",
    "tarp_coverage",
    "coverage_table",
    "evaluate_coverage",
    "coverage_accuracy",
    "summarize_coverage",
    # constants
    "DEFAULT_ALPHAS",
    "DEFAULT_N_NEIGHBORS",
    "DEFAULT_METHOD",
    "METRIC_ALIASES",
]


ArrayLike = Union[np.ndarray, Sequence[float]]

DEFAULT_ALPHAS: Tuple[float, ...] = (
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
)
DEFAULT_N_NEIGHBORS = 10
DEFAULT_METHOD = "knn"

METRIC_ALIASES: Dict[str, str] = {
    "knn": "knn",
    "k-nn": "knn",
    "nearest": "knn",
    "kde": "knn",
    "mahalanobis": "mahalanobis",
    "maha": "mahalanobis",
    "gaussian": "mahalanobis",
    "normal": "mahalanobis",
    "logprob": "logprob",
    "log_prob": "logprob",
    "density": "logprob",
    "dimension": "dimension",
    "dim": "dimension",
    "marginal": "dimension",
    "central": "dimension",
}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _as_2d(x: ArrayLike, name: str = "samples") -> np.ndarray:
    """Coerce input samples to a 2-D float array ``(n, d)``."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{name} must have 1 or 2 dimensions, got shape {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample")
    return arr


def _as_alphas(alphas: Optional[Iterable[float]]) -> np.ndarray:
    """Validate/standardise the nominal credibility levels."""
    if alphas is None:
        alphas = DEFAULT_ALPHAS
    arr = np.asarray(list(alphas), dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("alphas must contain at least one level")
    if np.any(arr <= 0.0) or np.any(arr >= 1.0):
        raise ValueError("all credibility levels must lie strictly in (0, 1)")
    return np.sort(arr)


def _canonical_method(method: str) -> str:
    key = str(method).strip().lower()
    if key not in METRIC_ALIASES:
        raise ValueError(
            f"unknown coverage method '{method}'; expected one of "
            f"{sorted(set(METRIC_ALIASES.values()))}"
        )
    return METRIC_ALIASES[key]


def _standardize_pair(
    approx: np.ndarray, reference: np.ndarray, enable: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """Optional z-scoring using the *approximate* samples' statistics."""
    if not enable:
        return approx, reference
    mean = approx.mean(axis=0, keepdims=True)
    std = approx.std(axis=0, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return (approx - mean) / std, (reference - mean) / std


def _pairwise_distances_chunked(
    queries: np.ndarray, refs: np.ndarray, chunk: int = 512
) -> np.ndarray:
    """Brute-force pairwise Euclidean distances with chunking over queries."""
    n_q = queries.shape[0]
    out = np.empty((n_q, refs.shape[0]), dtype=np.float64)
    for start in range(0, n_q, chunk):
        stop = min(start + chunk, n_q)
        diff = queries[start:stop, None, :] - refs[None, :, :]
        out[start:stop] = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-300)
    return out


# ---------------------------------------------------------------------------
# density proxies used to define HDRs
# ---------------------------------------------------------------------------
def knn_scores(
    queries: ArrayLike,
    reference: ArrayLike,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    *,
    exclude_self: bool = False,
    chunk: int = 512,
) -> np.ndarray:
    """k-NN log-density proxy of ``queries`` under the empirical measure of ``reference``.

    Returns ``s(x) = -mean_{j<=k} || x - x_j ||`` (higher = denser), i.e. minus the
    average distance to the ``k`` nearest reference samples.  This is a monotone
    transform of the standard k-NN density estimator and suffices for HDR
    thresholding.

    ``exclude_self`` should be set when ``queries is reference`` so that a point is
    not its own nearest neighbour.
    """
    q = _as_2d(queries, "queries")
    r = _as_2d(reference, "reference")
    k = int(max(1, n_neighbors))
    k = min(k, r.shape[0] if not exclude_self else max(1, r.shape[0] - 1))

    if _HAS_CKDTree:
        tree = cKDTree(r)
        dist, _ = tree.query(q, k=k + (1 if exclude_self else 0))
        dist = np.atleast_2d(dist)
        if exclude_self:
            dist = dist[:, 1:]
    else:  # pragma: no cover - fallback path
        d = _pairwise_distances_chunked(q, r, chunk=chunk)
        dist = np.sort(d, axis=1)
        if exclude_self:
            dist = dist[:, 1:]
        dist = dist[:, :k]
    dist = dist[:, :k]
    return -np.mean(dist, axis=1)


def negative_knn_distance(*args: Any, **kwargs: Any) -> np.ndarray:
    """Alias of :func:`knn_scores` (distance-based density proxy)."""
    return knn_scores(*args, **kwargs)


def mahalanobis_scores(
    queries: ArrayLike,
    reference: ArrayLike,
    *,
    covariance: Optional[np.ndarray] = None,
    regularize: float = 1e-8,
) -> np.ndarray:
    """Negative squared Mahalanobis distance to the reference distribution.

    ``s(x) = -0.5 * (x - mu)^T Sigma^{-1} (x - mu)`` with ``mu``/``Sigma`` estimated
    (robustly regularised) from ``reference`` unless supplied explicitly.  For a
    Gaussian approximation this score is the HDR ordering statistic.
    """
    q = _as_2d(queries, "queries")
    r = _as_2d(reference, "reference")
    mu = r.mean(axis=0)
    if covariance is None:
        cov = np.cov(r, rowvar=False, bias=False)
        cov = np.atleast_2d(cov)
    else:
        cov = np.atleast_2d(np.asarray(covariance, dtype=np.float64))
    d = cov.shape[0]
    if cov.shape != (d, d):
        raise ValueError(f"covariance must have shape ({d}, {d}), got {cov.shape}")
    if d == 1:
        var = float(cov[0, 0])
        var = var + regularize * max(var, 1.0)
        return -0.5 * (q[:, 0] - mu[0]) ** 2 / var
    # regularise toward the diagonal mean scale and invert in a stable manner
    scale = float(np.trace(cov)) / d
    cov = cov + np.eye(d) * (regularize * max(scale, 1.0) + 1e-30)
    try:
        prec = np.linalg.inv(cov)
    except np.linalg.LinAlgError:  # pragma: no cover
        prec = np.linalg.pinv(cov)
    diff = q - mu[None, :]
    maha2 = np.einsum("ij,jk,ik->i", diff, prec, diff)
    return -0.5 * maha2


def gaussian_log_density_scores(
    queries: ArrayLike,
    reference: ArrayLike,
    **kwargs: Any,
) -> np.ndarray:
    """Alias of :func:`mahalanobis_scores` (Gaussian plug-in log density)."""
    return mahalanobis_scores(queries, reference, **kwargs)


def hdr_threshold(scores: ArrayLike, alpha: float, *, higher_is_denser: bool = True) -> float:
    """Score threshold whose super-level set contains mass ``alpha``.

    ``alpha`` of the mass of the (approximate posterior) score distribution lies
    above (below when ``higher_is_denser=False``) the returned threshold.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    if s.size == 0:
        raise ValueError("scores must be non-empty")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie strictly in (0, 1)")
    level = 1.0 - alpha if higher_is_denser else alpha
    return float(np.quantile(s, level))


def hdr_mask(
    scores: ArrayLike, threshold: float, *, higher_is_denser: bool = True
) -> np.ndarray:
    """Boolean mask of scores inside the HDR defined by ``threshold``."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    return s >= threshold if higher_is_denser else s <= threshold


# ---------------------------------------------------------------------------
# results / configuration containers
# ---------------------------------------------------------------------------
@dataclass
class CoverageConfig:
    """Configuration of the coverage / calibration evaluation protocol."""

    alphas: Tuple[float, ...] = DEFAULT_ALPHAS
    method: str = DEFAULT_METHOD
    n_neighbors: int = DEFAULT_N_NEIGHBORS
    standardize: bool = False
    n_reference: Optional[int] = None
    n_approx: Optional[int] = None
    chunk: int = 512
    seed: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "alphas": list(self.alphas),
            "method": self.method,
            "n_neighbors": self.n_neighbors,
            "standardize": self.standardize,
            "n_reference": self.n_reference,
            "n_approx": self.n_approx,
            "chunk": self.chunk,
            "seed": self.seed,
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, cfg: Optional[Union[Dict[str, Any], "CoverageConfig"]] = None, **kwargs: Any) -> "CoverageConfig":
        if cfg is None:
            base: Dict[str, Any] = {}
        elif isinstance(cfg, CoverageConfig):
            base = cfg.to_dict()
        elif isinstance(cfg, dict):
            base = dict(cfg)
        else:
            raise TypeError(f"unsupported config type: {type(cfg)!r}")
        extra = dict(base.pop("extra", {}) or {})
        known = set(cls.__dataclass_fields__.keys())  # type: ignore[attr-defined]
        for key, value in kwargs.items():
            if key in known:
                base[key] = value
            else:
                extra[key] = value
        if "alphas" in base and base["alphas"] is not None:
            base["alphas"] = tuple(float(a) for a in base["alphas"])
        return cls(extra=extra, **{k: v for k, v in base.items() if k in known})


@dataclass
class CoverageCurve:
    """Expected-coverage curve and calibration summary statistics."""

    alphas: np.ndarray
    coverage: np.ndarray
    method: str
    n_approx: int
    n_reference: int
    thresholds: Optional[np.ndarray] = None
    std_error: Optional[np.ndarray] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- derived diagnostics -------------------------------------------------
    @property
    def nominal(self) -> np.ndarray:
        return np.asarray(self.alphas, dtype=np.float64)

    @property
    def empirical(self) -> np.ndarray:
        return np.asarray(self.coverage, dtype=np.float64)

    @property
    def deviation(self) -> np.ndarray:
        """Signed difference ``empirical - nominal``."""
        return self.empirical - self.nominal

    @property
    def calibration_error(self) -> float:
        """Mean absolute deviation between empirical and nominal coverage."""
        return float(np.mean(np.abs(self.deviation)))

    @property
    def coverage_error(self) -> float:
        return self.calibration_error

    @property
    def max_deviation(self) -> float:
        return float(np.max(np.abs(self.deviation)))

    @property
    def coverage_score(self) -> float:
        """Scalar quality score in ``[0, 1]`` (1 = perfectly calibrated)."""
        return float(np.clip(1.0 - 2.0 * self.calibration_error, 0.0, 1.0))

    def is_calibrated(self, tol: float = 0.05) -> bool:
        return bool(self.max_deviation <= tol)

    def __len__(self) -> int:
        return int(np.asarray(self.coverage).size)

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "alphas": np.asarray(self.alphas, dtype=float).tolist(),
            "coverage": np.asarray(self.coverage, dtype=float).tolist(),
            "method": self.method,
            "n_approx": int(self.n_approx),
            "n_reference": int(self.n_reference),
            "calibration_error": self.calibration_error,
            "max_deviation": self.max_deviation,
            "coverage_score": self.coverage_score,
        }
        if self.thresholds is not None:
            out["thresholds"] = np.asarray(self.thresholds, dtype=float).tolist()
        if self.std_error is not None:
            out["std_error"] = np.asarray(self.std_error, dtype=float).tolist()
        out.update(self.extra)
        return out


# ---------------------------------------------------------------------------
# core coverage computation
# ---------------------------------------------------------------------------
def _coverage_from_scores(
    score_approx: np.ndarray,
    score_ref: np.ndarray,
    alphas: np.ndarray,
    *,
    higher_is_denser: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Empirical coverage, HDR thresholds and binomial standard errors."""
    coverage = np.empty(alphas.shape, dtype=np.float64)
    thresholds = np.empty(alphas.shape, dtype=np.float64)
    n_ref = score_ref.size
    std_err = np.empty(alphas.shape, dtype=np.float64)
    for i, alpha in enumerate(alphas):
        thr = hdr_threshold(score_approx, float(alpha), higher_is_denser=higher_is_denser)
        inside = hdr_mask(score_ref, thr, higher_is_denser=higher_is_denser)
        coverage[i] = float(np.mean(inside))
        thresholds[i] = thr
        p = coverage[i]
        std_err[i] = math.sqrt(max(p * (1.0 - p), 0.0) / max(n_ref, 1))
    return coverage, thresholds, std_err


def coverage_curve(
    approx_samples: ArrayLike,
    reference_samples: ArrayLike,
    *,
    alphas: Optional[Iterable[float]] = None,
    method: str = DEFAULT_METHOD,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    standardize: bool = False,
    log_prob_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    config: Optional[Union[CoverageConfig, Dict[str, Any]]] = None,
    return_result: bool = True,
    **kwargs: Any,
) -> CoverageCurve:
    """Expected-coverage curve of ``approx_samples`` against ``reference_samples``.

    ``approx_samples`` are draws from the approximated (Simformer) conditional, and
    ``reference_samples`` are draws from the ground-truth conditional produced by
    ``simformer.reference.mcmc``.  For every nominal level ``alpha`` the HDR of the
    approximation containing ``alpha`` mass is computed and the fraction of
    reference samples inside it is reported as the empirical coverage (Sec. 4.1,
    Figs. A9-A12).
    """
    cfg = CoverageConfig.from_dict(config)
    if alphas is None:
        alphas = kwargs.pop("levels", None) or cfg.alphas
    alpha_arr = _as_alphas(alphas)
    method_key = _canonical_method(method if method != DEFAULT_METHOD else cfg.method)
    if n_neighbors == DEFAULT_N_NEIGHBORS:
        n_neighbors = cfg.n_neighbors
    standardize = bool(standardize or cfg.standardize)

    approx = _as_2d(approx_samples, "approx_samples")
    reference = _as_2d(reference_samples, "reference_samples")
    if approx.shape[1] != reference.shape[1]:
        raise ValueError(
            "approx_samples and reference_samples must share the same dimension "
            f"({approx.shape[1]} != {reference.shape[1]})"
        )
    approx, reference = _standardize_pair(approx, reference, standardize)

    extra: Dict[str, Any] = {"dimension": int(approx.shape[1])}

    if method_key == "dimension":
        return marginal_coverage_curve(
            approx, reference, alphas=alpha_arr, method="dimension"
        )

    if method_key == "logprob":
        if log_prob_fn is None:
            raise ValueError("method='logprob' requires a `log_prob_fn` callable")
        score_approx = np.asarray(log_prob_fn(approx), dtype=np.float64).ravel()
        score_ref = np.asarray(log_prob_fn(reference), dtype=np.float64).ravel()
    elif method_key == "knn":
        k = int(max(1, n_neighbors))
        score_approx = knn_scores(
            approx, approx, k, exclude_self=True, chunk=cfg.chunk
        )
        score_ref = knn_scores(reference, approx, k, exclude_self=False, chunk=cfg.chunk)
        extra["n_neighbors"] = k
    elif method_key == "mahalanobis":
        score_approx = mahalanobis_scores(approx, approx, **kwargs)
        score_ref = mahalanobis_scores(reference, approx, **kwargs)
    else:  # pragma: no cover - guarded by _canonical_method
        raise ValueError(f"unsupported coverage method '{method_key}'")

    coverage, thresholds, std_err = _coverage_from_scores(
        score_approx, score_ref, alpha_arr, higher_is_denser=True
    )

    result = CoverageCurve(
        alphas=alpha_arr,
        coverage=coverage,
        method=method_key,
        n_approx=int(approx.shape[0]),
        n_reference=int(reference.shape[0]),
        thresholds=thresholds,
        std_error=std_err,
        extra=extra,
    )
    return result


def coverage_curve_from_logprob(
    approx_samples: ArrayLike,
    reference_samples: ArrayLike,
    log_prob_fn: Callable[[np.ndarray], np.ndarray],
    *,
    alphas: Optional[Iterable[float]] = None,
    **kwargs: Any,
) -> CoverageCurve:
    """Exact-HDR coverage curve when a ground-truth log-density is available.

    ``log_prob_fn`` should return the (possibly unnormalised) log density of the
    **ground-truth conditional** evaluated pointwise on a ``(n, d)`` array.  This is
    available in closed form for e.g. the Gaussian Linear and SLCP tasks.
    """
    return coverage_curve(
        approx_samples,
        reference_samples,
        alphas=alphas,
        method="logprob",
        log_prob_fn=log_prob_fn,
        **kwargs,
    )


def logprob_coverage_curve(*args: Any, **kwargs: Any) -> CoverageCurve:
    """Alias of :func:`coverage_curve_from_logprob`."""
    return coverage_curve_from_logprob(*args, **kwargs)


def expected_coverage(
    approx_samples: ArrayLike,
    reference_samples: ArrayLike,
    *,
    alphas: Optional[Iterable[float]] = None,
    method: str = DEFAULT_METHOD,
    return_result: bool = False,
    **kwargs: Any,
) -> Union[np.ndarray, CoverageCurve]:
    """Empirical expected-coverage values at the requested nominal levels.

    Returns an ``np.ndarray`` of empirical coverage values by default, or the full
    :class:`CoverageCurve` when ``return_result=True``.
    """
    curve = coverage_curve(
        approx_samples, reference_samples, alphas=alphas, method=method, **kwargs
    )
    return curve if return_result else curve.empirical


def expected_coverage_curve(*args: Any, **kwargs: Any) -> CoverageCurve:
    """Alias of :func:`coverage_curve`."""
    return coverage_curve(*args, **kwargs)


def calibration_curve(*args: Any, **kwargs: Any) -> CoverageCurve:
    """Alias of :func:`coverage_curve` (calibration view)."""
    return coverage_curve(*args, **kwargs)


def marginal_coverage_curve(
    approx_samples: ArrayLike,
    reference_samples: ArrayLike,
    *,
    alphas: Optional[Iterable[float]] = None,
    method: str = "dimension",
    per_dimension: bool = False,
) -> CoverageCurve:
    """Coverage based on per-coordinate highest-density (central) intervals.

    For each coordinate the central interval of the approximation with mass
    ``alpha`` is computed from its own samples, and the fraction of reference
    samples inside that interval is the empirical coverage.  ``per_dimension=True``
    returns one curve per coordinate in ``extra["per_dimension"]``.
    """
    alpha_arr = _as_alphas(alphas)
    approx = _as_2d(approx_samples, "approx_samples")
    reference = _as_2d(reference_samples, "reference_samples")
    if approx.shape[1] != reference.shape[1]:
        raise ValueError("approx_samples and reference_samples must share dimension")
    d = approx.shape[1]

    curves: List[np.ndarray] = []
    for j in range(d):
        lo = np.quantile(approx[:, j], (1.0 - alpha_arr) / 2.0)
        hi = np.quantile(approx[:, j], 1.0 - (1.0 - alpha_arr) / 2.0)
        curves.append(
            np.array(
                [float(np.mean((reference[:, j] >= lo[i]) & (reference[:, j] <= hi[i])))
                 for i in range(alpha_arr.size)]
            )
        )
    per_dim = np.stack(curves, axis=0)  # (d, n_alphas)
    mean_curve = per_dim.mean(axis=0)

    extra: Dict[str, Any] = {"dimension": int(d)}
    if per_dimension:
        extra["per_dimension"] = per_dim.tolist()
    return CoverageCurve(
        alphas=alpha_arr,
        coverage=mean_curve,
        method="dimension",
        n_approx=int(approx.shape[0]),
        n_reference=int(reference.shape[0]),
        extra=extra,
    )


def dimension_coverage(
    approx_samples: ArrayLike,
    reference_samples: ArrayLike,
    *,
    alphas: Optional[Iterable[float]] = None,
    dimension: Optional[int] = None,
) -> np.ndarray:
    """Empirical coverage values for one coordinate (or the average over all)."""
    alpha_arr = _as_alphas(alphas)
    approx = _as_2d(approx_samples, "approx_samples")
    reference = _as_2d(reference_samples, "reference_samples")
    dims = range(approx.shape[1]) if dimension is None else [int(dimension)]
    out = []
    for j in dims:
        lo = np.quantile(approx[:, j], (1.0 - alpha_arr) / 2.0)
        hi = np.quantile(approx[:, j], 1.0 - (1.0 - alpha_arr) / 2.0)
        out.append(
            [float(np.mean((reference[:, j] >= lo[i]) & (reference[:, j] <= hi[i])))
             for i in range(alpha_arr.size)]
        )
    return np.mean(np.asarray(out), axis=0)


# ---------------------------------------------------------------------------
# scalar diagnostics / entry points
# ---------------------------------------------------------------------------
def calibration_error(
    alphas: ArrayLike, coverage: ArrayLike, *, weights: Optional[ArrayLike] = None
) -> float:
    """Mean (optionally weighted) absolute deviation between coverage and nominal."""
    a = np.asarray(alphas, dtype=np.float64).ravel()
    c = np.asarray(coverage, dtype=np.float64).ravel()
    if a.shape != c.shape:
        raise ValueError("alphas and coverage must have the same shape")
    dev = np.abs(c - a)
    if weights is None:
        return float(np.mean(dev))
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.shape != dev.shape:
        raise ValueError("weights must match alphas")
    w = w / np.sum(w)
    return float(np.sum(w * dev))


def coverage_error(
    alphas_or_curve: Union[ArrayLike, CoverageCurve],
    coverage: Optional[ArrayLike] = None,
    **kwargs: Any,
) -> float:
    """Mean absolute coverage deviation (accepts a :class:`CoverageCurve` too)."""
    if isinstance(alphas_or_curve, CoverageCurve):
        return alphas_or_curve.calibration_error
    if coverage is None:
        raise ValueError("coverage values are required when alphas are given")
    return calibration_error(alphas_or_curve, coverage, **kwargs)


def coverage_score(
    alphas_or_curve: Union[ArrayLike, CoverageCurve],
    coverage: Optional[ArrayLike] = None,
) -> float:
    """Scalar score in ``[0, 1]``; ``1.0`` means perfectly calibrated."""
    if isinstance(alphas_or_curve, CoverageCurve):
        return alphas_or_curve.coverage_score
    if coverage is None:
        raise ValueError("coverage values are required when alphas are given")
    err = calibration_error(alphas_or_curve, coverage)
    return float(np.clip(1.0 - 2.0 * err, 0.0, 1.0))


def check_coverage(
    approx_samples: ArrayLike,
    reference_samples: ArrayLike,
    *,
    alphas: Optional[Iterable[float]] = None,
    tol: float = 0.05,
    method: str = DEFAULT_METHOD,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run the coverage protocol and return a plain report dictionary."""
    curve = coverage_curve(
        approx_samples,
        reference_samples,
        alphas=alphas,
        method=method,
        **kwargs,
    )
    report = curve.to_dict()
    report["calibrated"] = curve.is_calibrated(tol=tol)
    report["tolerance"] = float(tol)
    return report


def check_calibration(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Alias of :func:`check_coverage`."""
    return check_coverage(*args, **kwargs)


def tarp_coverage(
    score_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    theta_true: ArrayLike,
    theta_approx: ArrayLike,
    *,
    alphas: Optional[Iterable[float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """TARP-style coverage (Lemos et al., 2023) given an approximate-posterior score.

    ``score_fn(x_obs, theta)`` must return a scalar posterior score per row of
    ``theta`` (higher = more probable).  For every true parameter we compute the
    rank of ``theta_true`` among ``theta_approx`` under that score and report the
    empirical CDF across parameters against the uniform reference.

    Returns ``(nominal, empirical)`` where ``nominal`` is the uniform grid of
    credibility levels and ``empirical`` the associated coverage.
    """
    true = _as_2d(theta_true, "theta_true")
    approx = _as_2d(theta_approx, "theta_approx")
    obs = kwargs_obs = None  # placeholder to keep signature explicit
    del obs, kwargs_obs
    raise NotImplementedError(
        "tarp_coverage requires per-observation conditioning; please supply "
        "scores via coverage_curve(..., method='logprob', log_prob_fn=...) or "
        "use `coverage_from_ranks`."
    )


def coverage_from_ranks(ranks: ArrayLike, alphas: Optional[Iterable[float]] = None):
    """Coverage curve from pre-computed ranks (helper for TARP/SBC diagnostics).

    ``ranks`` should lie in ``[0, 1]``; the empirical coverage at level ``alpha`` is
    the fraction of ranks below ``alpha``.
    """
    r = np.asarray(ranks, dtype=np.float64).ravel()
    if r.size == 0:
        raise ValueError("ranks must be non-empty")
    if np.any(r < 0.0) or np.any(r > 1.0):
        r = (r - r.min()) / max(r.max() - r.min(), 1e-12)
    alpha_arr = _as_alphas(alphas)
    cov = np.array([float(np.mean(r <= a)) for a in alpha_arr])
    return CoverageCurve(
        alphas=alpha_arr,
        coverage=cov,
        method="rank",
        n_approx=int(r.size),
        n_reference=int(r.size),
    )


def evaluate_coverage(
    approx_samples: ArrayLike,
    reference_samples: ArrayLike,
    *,
    alphas: Optional[Iterable[float]] = None,
    method: str = DEFAULT_METHOD,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    return_result: bool = False,
    **kwargs: Any,
) -> Union[float, CoverageCurve]:
    """Paper-default entry point used by ``simformer.eval.evaluate('coverage', ...)``.

    Returns the scalar calibration score in ``[0, 1]`` (``1.0`` = perfect) by
    default, or the rich :class:`CoverageCurve` when ``return_result=True``.
    """
    curve = coverage_curve(
        approx_samples,
        reference_samples,
        alphas=alphas,
        method=method,
        n_neighbors=n_neighbors,
        **kwargs,
    )
    return curve if return_result else curve.coverage_score


def coverage_accuracy(*args: Any, **kwargs: Any) -> Union[float, CoverageCurve]:
    """Alias of :func:`evaluate_coverage` (accuracy-style naming)."""
    return evaluate_coverage(*args, **kwargs)


def summarize_coverage(curve_or_samples: Union[CoverageCurve, ArrayLike], *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Summarize a coverage curve (or compute one and summarize it)."""
    if isinstance(curve_or_samples, CoverageCurve):
        return curve_or_samples.to_dict()
    return coverage_curve(curve_or_samples, *args, **kwargs).to_dict()


def coverage_table(
    curves: Dict[str, Union[CoverageCurve, Dict[str, Any]]]
) -> Dict[str, Dict[str, float]]:
    """Summarize several named methods (``method -> diagnostics``)."""
    table: Dict[str, Dict[str, float]] = {}
    for name, curve in curves.items():
        if isinstance(curve, CoverageCurve):
            table[str(name)] = curve.to_dict()
        else:
            table[str(name)] = dict(curve)
    return table
