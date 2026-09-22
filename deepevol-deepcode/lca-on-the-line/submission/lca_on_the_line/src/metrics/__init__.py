"""Metrics package for LCA-on-the-Line.

Exposes the two metric families used by the reproduction:

* :mod:`src.metrics.lca_metric` -- dataset level mistake-severity metrics
  ``D_LCA`` (information-content LCA distance over misclassified samples) and
  ``D_ELCA`` (expected LCA distance over the full softmax), plus Top-1/Top-5.
* :mod:`src.metrics.correlation` -- R^2 / Pearson / Kendall / Spearman / MAE,
  min-max & probit scaling and the linear ``ID LCA -> OOD accuracy`` fit.

The package initializer is intentionally dependency-free: symbols are resolved
lazily through PEP 562 ``__getattr__`` so that ``import src.metrics`` never
eagerly pulls numpy/scipy/torch.  Heavy imports happen only when a consumer
touches one of the exported names.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional, Tuple

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Lazy export map: public symbol -> providing submodule
# ---------------------------------------------------------------------------
_EXPORTS: Dict[str, str] = {
    # ---- src/metrics/lca_metric.py -------------------------------------
    "LcaMetric": "lca_metric",
    "ModelMetrics": "lca_metric",
    "evaluate_model_outputs": "lca_metric",
    "lca_distance_dataset": "lca_metric",
    "elca_distance_dataset": "lca_metric",
    "expected_lca_distance_dataset": "lca_metric",
    "softmax": "lca_metric",
    "topk_accuracy": "lca_metric",
    "top1_accuracy": "lca_metric",
    "top5_accuracy": "lca_metric",
    "DEFAULT_SOFTMAX_TEMPERATURE": "lca_metric",
    # ---- src/metrics/correlation.py -------------------------------------
    "LinearFit": "correlation",
    "CorrelationResults": "correlation",
    "correlation_metrics": "correlation",
    "compute_correlations": "correlation",
    "correlation_table": "correlation",
    "correlation_table_from_dataframe": "correlation",
    "format_table": "correlation",
    "fit_linear": "correlation",
    "linear_regression": "correlation",
    "fit_predict": "correlation",
    "regression_report": "correlation",
    "r2_score": "correlation",
    "pearson_correlation": "correlation",
    "kendall_tau": "correlation",
    "spearman_rho": "correlation",
    "mean_absolute_error": "correlation",
    "root_mean_squared_error": "correlation",
    "top1_error": "correlation",
    "min_max_scale": "correlation",
    "inverse_min_max_scale": "correlation",
    "probit": "correlation",
    "inverse_probit": "correlation",
    "apply_scaler": "correlation",
    "rank_data": "correlation",
    "SCALERS": "correlation",
    "TABLE2_CORRELATION_TARGETS": "correlation",
    "TABLE3_MAE_TARGETS": "correlation",
    "DEFAULT_SCALER": "correlation",
}

_SUBMODULES: Tuple[str, ...] = ("lca_metric", "correlation")

# Aliases used by downstream code / scripts (paper naming).
_ALIASES: Dict[str, str] = {
    "PearsonCorrelation": "pearson_correlation",
    "SpearmanCorrelation": "spearman_rho",
    "KendallCorrelation": "kendall_tau",
    "R2": "r2_score",
    "MAE": "mean_absolute_error",
    "RMSE": "root_mean_squared_error",
    "compute_correlation_table": "correlation_table",
    "DatasetLca": "lca_distance_dataset",
    "DatasetElca": "elca_distance_dataset",
}


def _import_submodule(name: str) -> Any:
    """Import a sibling submodule tolerating several ``sys.path`` layouts."""
    candidates = (
        f"{__name__}.{name}",
        f"src.metrics.{name}",
        f"metrics.{name}",
        name,
    )
    last_error: Optional[BaseException] = None
    for candidate in candidates:
        try:
            return importlib.import_module(candidate)
        except Exception as exc:  # pragma: no cover - defensive
            last_error = exc
    if last_error is not None:
        raise last_error
    raise ImportError(f"cannot import metrics submodule {name!r}")


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute access for submodules and public symbols."""
    if name in _SUBMODULES:
        return _import_submodule(name)

    symbol = _ALIASES.get(name, name)
    module_name = _EXPORTS.get(symbol)
    if module_name is not None:
        module = _import_submodule(module_name)
        try:
            value = getattr(module, symbol)
        except AttributeError as exc:  # pragma: no cover - defensive
            raise AttributeError(
                f"module {module_name!r} has no attribute {symbol!r}"
            ) from exc
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(list(globals()) + list(_SUBMODULES) + list(_EXPORTS) + list(_ALIASES)))


def __getattr_many__() -> Dict[str, Any]:  # pragma: no cover - convenience
    """Eagerly resolve every exported symbol (used by tests/introspection)."""
    resolved: Dict[str, Any] = {}
    for name in _EXPORTS:
        resolved[name] = getattr(__import__(__name__, fromlist=[name]), name)
    return resolved


__all__ = ["__version__"] + list(_SUBMODULES) + sorted(_EXPORTS) + sorted(_ALIASES)
