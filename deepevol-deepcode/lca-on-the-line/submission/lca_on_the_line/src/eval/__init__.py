"""Evaluation package for the LCA-on-the-Line reproduction.

This package bundles the two evaluation layers used by the paper:

``evaluate_models``
    The model-evaluation driver (Section 4.1) that computes ID/OOD
    Top-1/Top-5/LCA/ELCA for every model in the 75-model zoo and caches the
    per-model penultimate features ``M(X)`` for downstream stages.

``ood_prediction``
    OOD performance prediction and baselines (Section 4.2 / Table 3):
    ID-LCA, ID Top-1, Average Confidence, Aline-D and Aline-S.

Consistent with the other sub-packages of this repository, the initializer is
deliberately dependency-free: submodules and their public symbols are resolved
lazily through :pep:`562` ``__getattr__`` so that ``import src.eval`` does not
eagerly pull numpy/torch or any model code.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional, Tuple

__version__ = "0.1.0"

_SUBMODULES: Tuple[str, ...] = ("evaluate_models", "ood_prediction")

# Public symbol -> providing submodule.
_EXPORTS: Dict[str, str] = {
    # ---- evaluate_models (Section 4.1; Tables 1, 2, 8) -------------------
    "EvaluationConfig": "evaluate_models",
    "ModelRecord": "evaluate_models",
    "collect_outputs": "evaluate_models",
    "collect_outputs_cached": "evaluate_models",
    "save_outputs_npz": "evaluate_models",
    "load_outputs_npz": "evaluate_models",
    "evaluate_model_on_dataset": "evaluate_models",
    "evaluate_model": "evaluate_models",
    "evaluate_zoo": "evaluate_models",
    "evaluate_zoo_from_cache": "evaluate_models",
    "records_to_rows": "evaluate_models",
    "records_to_dataframe": "evaluate_models",
    "summary_table": "evaluate_models",
    "save_results": "evaluate_models",
    "load_results": "evaluate_models",
    "build_hierarchy": "evaluate_models",
    "build_evaluation_loaders": "evaluate_models",
    "main": "evaluate_models",
    "TABLE8_REFERENCE": "evaluate_models",
    # ---- ood_prediction (Section 4.2; Table 3) ---------------------------
    "TemperatureScaler": "ood_prediction",
    "OodPredictionResult": "ood_prediction",
    "OodPredictionTable": "ood_prediction",
    "evaluate_ood_prediction": "ood_prediction",
    "predict_linear": "ood_prediction",
    "fit_temperature": "ood_prediction",
    "average_confidence_predictions": "ood_prediction",
    "soft_agreement": "ood_prediction",
    "hard_agreement": "ood_prediction",
    "pairwise_agreement": "ood_prediction",
    "agreement_matrix": "ood_prediction",
    "aline_predictions": "ood_prediction",
    "build_model_pairs": "ood_prediction",
    "resolve_cached_output": "ood_prediction",
    "load_cached_output": "ood_prediction",
    "summarize": "ood_prediction",
    "load_records": "ood_prediction",
    "build_arg_parser": "ood_prediction",
    "TABLE3_REFERENCE": "ood_prediction",
    "DEFAULT_METHODS": "ood_prediction",
    "DEFAULT_OOD_DATASETS": "ood_prediction",
}

# Paper-facing aliases.
_ALIASES: Dict[str, str] = {
    "EvalConfig": "EvaluationConfig",
    "Table8Reference": "TABLE8_REFERENCE",
    "Table3Reference": "TABLE3_REFERENCE",
    "OODPredictionTable": "OodPredictionTable",
    "OODPredictionResult": "OodPredictionResult",
    "AlinePredictions": "aline_predictions",
    "EvaluateZoo": "evaluate_zoo",
    "EvaluateZooFromCache": "evaluate_zoo_from_cache",
}

__all__ = ["__version__"] + list(_SUBMODULES) + sorted(_EXPORTS) + sorted(_ALIASES)


def _import_submodule(name: str) -> Any:
    """Import a sibling evaluation submodule, tolerating several layouts."""
    candidates = (
        f"{__name__}.{name}",
        f"src.eval.{name}",
        f"eval.{name}",
        name,
    )
    last_error: Optional[BaseException] = None
    for candidate in candidates:
        try:
            return importlib.import_module(candidate)
        except Exception as exc:  # pragma: no cover - layout dependent
            last_error = exc
    if last_error is not None:
        raise last_error
    raise ImportError(f"cannot import evaluation submodule {name!r}")


def _resolve(name: str) -> Any:
    """Resolve ``name`` to a submodule, an exported symbol or an alias."""
    if name in _SUBMODULES:
        return _import_submodule(name)
    module_name = _EXPORTS.get(name)
    if module_name is not None:
        return getattr(_import_submodule(module_name), name)
    # Aliases may reference either a submodule attribute or a re-exported symbol.
    target = _ALIASES.get(name)
    if target is not None:
        if target in _EXPORTS:
            module_name = _EXPORTS[target]
            return getattr(_import_submodule(module_name), target)
        if target in _SUBMODULES:
            return _import_submodule(target)
        # Last resort: look for the symbol on either submodule.
        for submodule in _SUBMODULES:
            try:
                return getattr(_import_submodule(submodule), target)
            except AttributeError:
                continue
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute hook (caches resolved objects)."""
    value = _resolve(name)
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(_SUBMODULES) | set(_EXPORTS) | set(_ALIASES))


def __getattr_many__() -> Dict[str, Any]:
    """Resolve every exported symbol; failures are returned instead of raised."""
    resolved: Dict[str, Any] = {}
    for name in list(_EXPORTS) + list(_ALIASES) + list(_SUBMODULES):
        try:
            resolved[name] = _resolve(name)
        except Exception as exc:  # pragma: no cover - introspection helper
            resolved[name] = exc
    return resolved
