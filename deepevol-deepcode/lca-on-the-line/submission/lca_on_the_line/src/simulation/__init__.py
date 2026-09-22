"""Simulation package for *LCA-on-the-Line* (Appendix C / Table 7).

Exposes the self-contained Gaussian-mixture study that demonstrates the LCA
hypothesis (a model that is *better* in-distribution can be *worse*
out-of-distribution, and ID LCA tracks the OOD gap).

The package is intentionally import-light: no heavy dependency (numpy is only
pulled in when a symbol from :mod:`src.simulation.simulated_lca` is actually
accessed).  Resolution is performed through PEP 562 ``__getattr__`` so that
``import src.simulation`` stays cheap and can be used freely from tests and
scripts, and so that multiple ``sys.path`` layouts (package vs. flat ``src``)
both work.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional, Tuple

__version__ = "0.1.0"

__all__ = ["__version__"] + [
    # submodules
    "simulated_lca",
    # classes
    "NumpySoftmaxClassifier",
    "SimulationResult",
    # matrix / hierarchy helpers
    "simulated_lca_matrix",
    "build_simulated_hierarchy",
    "hierarchy_matches_analytic_matrix",
    "hierarchy_lca_matrix",
    "build_lca_metric",
    # data generation
    "class_means",
    "sample_mixture",
    "select_features",
    # modelling
    "fit_logistic_regression",
    "model_predict",
    "model_predict_proba",
    "model_view",
    "ood_feature_view",
    # metrics
    "accuracy",
    "top1_error",
    "elca_of_predictions",
    "lca_all_of_predictions",
    "lca_of_predictions",
    "evaluate_split",
    "train_models",
    "evaluate_trial",
    "run_simulation",
    "sanity_check",
    "bayes_errors",
    # reporting / cli
    "make_figure",
    "build_arg_parser",
    "main",
    # constants
    "CAUSAL_FEATURES",
    "CLASS_MEANS",
    "CONFOUNDING_FEATURES",
    "DEFAULT_NUM_SAMPLES",
    "DEFAULT_NUM_TRIALS",
    "DEFAULT_SEED",
    "ID_FEATURES",
    "LCA_MATRIX",
    "MODEL_NAMES",
    "NUM_CLASSES",
    "OOD_FEATURES",
    "TABLE7_REFERENCE",
]


# --------------------------------------------------------------------------- #
# Lazy import plumbing
# --------------------------------------------------------------------------- #
_SUBMODULES: Tuple[str, ...] = ("simulated_lca",)

_EXPORTS: Dict[str, str] = {
    # classes
    "NumpySoftmaxClassifier": "simulated_lca",
    "SimulationResult": "simulated_lca",
    # matrix / hierarchy
    "simulated_lca_matrix": "simulated_lca",
    "build_simulated_hierarchy": "simulated_lca",
    "hierarchy_matches_analytic_matrix": "simulated_lca",
    "hierarchy_lca_matrix": "simulated_lca",
    "build_lca_metric": "simulated_lca",
    # data
    "class_means": "simulated_lca",
    "sample_mixture": "simulated_lca",
    "select_features": "simulated_lca",
    # modelling
    "fit_logistic_regression": "simulated_lca",
    "model_predict": "simulated_lca",
    "model_predict_proba": "simulated_lca",
    "model_view": "simulated_lca",
    "ood_feature_view": "simulated_lca",
    # metrics
    "accuracy": "simulated_lca",
    "top1_error": "simulated_lca",
    "elca_of_predictions": "simulated_lca",
    "lca_all_of_predictions": "simulated_lca",
    "lca_of_predictions": "simulated_lca",
    "evaluate_split": "simulated_lca",
    "train_models": "simulated_lca",
    "evaluate_trial": "simulated_lca",
    "run_simulation": "simulated_lca",
    "sanity_check": "simulated_lca",
    "bayes_errors": "simulated_lca",
    # reporting / cli
    "make_figure": "simulated_lca",
    "build_arg_parser": "simulated_lca",
    "main": "simulated_lca",
    # constants
    "CAUSAL_FEATURES": "simulated_lca",
    "CLASS_MEANS": "simulated_lca",
    "CONFOUNDING_FEATURES": "simulated_lca",
    "DEFAULT_NUM_SAMPLES": "simulated_lca",
    "DEFAULT_NUM_TRIALS": "simulated_lca",
    "DEFAULT_SEED": "simulated_lca",
    "ID_FEATURES": "simulated_lca",
    "LCA_MATRIX": "simulated_lca",
    "MODEL_NAMES": "simulated_lca",
    "NUM_CLASSES": "simulated_lca",
    "OOD_FEATURES": "simulated_lca",
    "TABLE7_REFERENCE": "simulated_lca",
}

_ALIASES: Dict[str, str] = {
    # paper-facing naming convenience
    "LCAMatrix": "simulated_lca_matrix",
    "simulate": "run_simulation",
    "run_table7": "run_simulation",
    "SimulationResultTable7": "SimulationResult",
}


def _import_submodule(name: str) -> Any:
    """Import a sibling submodule tolerating several ``sys.path`` layouts."""
    candidates = (
        f"{__name__}.{name}",
        f"src.simulation.{name}",
        f"simulation.{name}",
        name,
    )
    last_error: Optional[BaseException] = None
    for candidate in candidates:
        try:
            return importlib.import_module(candidate)
        except Exception as exc:  # pragma: no cover - layout dependent
            last_error = exc
    if last_error is not None:  # pragma: no cover - layout dependent
        raise last_error
    raise ImportError(f"cannot import simulation submodule {name!r}")  # pragma: no cover


def _resolve(name: str) -> Any:
    """Resolve ``name`` to a submodule, an exported symbol, or an alias target."""
    if name in _SUBMODULES:
        return _import_submodule(name)

    if name in _EXPORTS:
        module = _import_submodule(_EXPORTS[name])
        return getattr(module, name)

    if name in _ALIASES:
        target = _ALIASES[name]
        return _resolve(target) if target in _EXPORTS else getattr(
            _import_submodule("simulated_lca"), target
        )

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __getattr__(name: str) -> Any:  # PEP 562
    value = _resolve(name)
    globals()[name] = value  # cache for subsequent access
    return value


def __dir__() -> List[str]:
    return sorted(set(list(globals().keys()) + list(_SUBMODULES) + list(_EXPORTS) + list(_ALIASES)))


def __getattr_many__() -> Dict[str, Any]:  # pragma: no cover - convenience/testing
    """Eagerly resolve every exported symbol (useful for tests/introspection)."""
    resolved: Dict[str, Any] = {}
    for name in _EXPORTS:
        try:
            resolved[name] = _resolve(name)
        except Exception as exc:  # pragma: no cover - optional dependency
            resolved[name] = exc
    return resolved
