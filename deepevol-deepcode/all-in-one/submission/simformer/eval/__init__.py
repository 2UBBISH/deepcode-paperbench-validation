"""Evaluation utilities for Simformer.

This package gathers the metrics used in Section 4.1 / Appendix A3.1 of the
Simformer paper: 

* :mod:`simformer.eval.c2st` -- classifier two-sample tests (random forest with
  100 trees, 0.5 = indistinguishable) comparing amortized samples against
  ground-truth MCMC references.
* :mod:`simformer.eval.coverage` -- expected coverage / calibration of the
  estimated posterior (rank statistics and HDR coverage).
* :mod:`simformer.eval.nll` -- log-likelihood of held-out joint data computed
  with the probability-flow ODE (Appendix A3.1), converted to nats.

Sub-modules that require heavyweight optional dependencies (scikit-learn) are
imported lazily so that the lighter modules (coverage / NLL) remain usable in a
minimal environment.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List

__all__ = [
    # c2st
    "c2st",
    "c2st_accuracy",
    "c2st_score",
    "evaluate_c2st",
    "c2st_matrix",
    # coverage
    "expected_coverage",
    "coverage_from_distances",
    "rank_statistic",
    "coverage_curve",
    "coverage_error",
    "hdr_coverage_levels",
    "evaluate_coverage",
    # nll
    "joint_log_likelihood",
    "joint_nll",
    "probability_flow_nll",
    "evaluate_nll",
    "nll_from_samples",
    # dispatch helper
    "evaluate",
]

_SUBMODULES: Dict[str, str] = {
    # name -> module path
    "c2st": "simformer.eval.c2st",
    "c2st_accuracy": "simformer.eval.c2st",
    "c2st_score": "simformer.eval.c2st",
    "evaluate_c2st": "simformer.eval.c2st",
    "c2st_matrix": "simformer.eval.c2st",
    "expected_coverage": "simformer.eval.coverage",
    "coverage_from_distances": "simformer.eval.coverage",
    "rank_statistic": "simformer.eval.coverage",
    "coverage_curve": "simformer.eval.coverage",
    "coverage_error": "simformer.eval.coverage",
    "hdr_coverage_levels": "simformer.eval.coverage",
    "evaluate_coverage": "simformer.eval.coverage",
    "joint_log_likelihood": "simformer.eval.nll",
    "joint_nll": "simformer.eval.nll",
    "probability_flow_nll": "simformer.eval.nll",
    "evaluate_nll": "simformer.eval.nll",
    "nll_from_samples": "simformer.eval.nll",
}

DEFAULT_N_FOREST_TREES = 100
DEFAULT_C2ST_SEED = 0

_MODULE_CACHE: Dict[str, Any] = {}


def _load_module(path: str) -> Any:
    """Import (and cache) a sub-module of :mod:`simformer.eval`."""
    module = _MODULE_CACHE.get(path)
    if module is None:
        try:
            module = importlib.import_module(path)
        except ImportError:
            module = importlib.import_module(path.rsplit(".", 1)[-1])
        _MODULE_CACHE[path] = module
    return module


def __getattr__(name: str) -> Any:  # pragma: no cover - thin delegation
    path = _SUBMODULES.get(name)
    if path is None:
        raise AttributeError(f"module 'simformer.eval' has no attribute {name!r}")
    value = getattr(_load_module(path), name, None)
    if value is None:  # pragma: no cover - defensive
        raise AttributeError(f"{path} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(__all__))


def evaluate(
    metric: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Dispatch to one of the evaluation metrics by name.

    ``metric`` may be one of ``"c2st"``, ``"coverage"`` or ``"nll"`` (case
    insensitive, ``"-"``/``"_"`` insensitive).  Extra positional and keyword
    arguments are forwarded to the corresponding evaluation function.
    """
    key = metric.lower().replace("-", "_").replace(" ", "_")
    dispatch = {
        "c2st": "evaluate_c2st",
        "classifier_two_sample_test": "evaluate_c2st",
        "two_sample": "evaluate_c2st",
        "coverage": "evaluate_coverage",
        "calibration": "evaluate_coverage",
        "expected_coverage": "evaluate_coverage",
        "nll": "evaluate_nll",
        "log_likelihood": "evaluate_nll",
        "loglikelihood": "evaluate_nll",
    }
    target = dispatch.get(key)
    if target is None:
        raise ValueError(
            f"unknown metric {metric!r}; expected one of {sorted(set(dispatch))}"
        )
    return globals()[target](*args, **kwargs)


def available_metrics() -> List[str]:
    """Return the canonical metric names understood by :func:`evaluate`."""
    return ["c2st", "coverage", "nll"]
