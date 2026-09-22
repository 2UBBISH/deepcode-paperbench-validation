"""Taxonomy-alignment package for ``LCA-on-the-Line`` (paper Section 4.3).

This package groups the three taxonomy-alignment experiments:

* :mod:`src.alignment.soft_loss` -- Algorithm 1 (``LCA_ALIGNMENT_LOSS``): the
  taxonomy-aligned soft loss built from ``reverse_LCA_matrix = 1 - M_LCA``.
* :mod:`src.alignment.linear_probe` -- linear probing over frozen features with
  CE-only vs. CE + soft-LCA losses and Wortsman-style weight interpolation
  (Tables 5/6/9/10).
* :mod:`src.alignment.prompt_engineering` -- taxonomy-aware prompt engineering
  for zero-shot CLIP-ViT32 (Table 14).

The initializer is intentionally *dependency free*: importing
``src.alignment`` must not pull in torch, numpy, CLIP or OpenCLIP.  All heavy
symbols are resolved lazily through PEP 562 ``__getattr__`` on first access and
memoized into the module globals afterwards, mirroring the pattern used by the
sibling packages (``src.hierarchy``, ``src.metrics``, ``src.models``,
``src.data``, ``src.eval``, ``src.simulation``).
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional, Tuple

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Sub-modules of this package
# ---------------------------------------------------------------------------
_SUBMODULES: Tuple[str, ...] = ("soft_loss", "linear_probe", "prompt_engineering")

# ---------------------------------------------------------------------------
# Symbol -> providing sub-module map
# ---------------------------------------------------------------------------
# NOTE: ``build_soft_loss`` exists in both ``soft_loss`` and ``linear_probe``;
# the first sub-module listed here wins (``soft_loss``), matching the order of
# ``_SUBMODULES``.
_EXPORTS: Dict[str, str] = {
    # ---- soft_loss.py (Algorithm 1) --------------------------------------
    "LCAAlignmentLoss": "soft_loss",
    "LcaAlignmentLoss": "soft_loss",
    "lca_alignment_loss": "soft_loss",
    "LCA_ALIGNMENT_LOSS": "soft_loss",
    "lca_soft_loss": "soft_loss",
    "taxonomy_alignment_loss": "soft_loss",
    "reverse_lca_matrix": "soft_loss",
    "build_reverse_lca_matrix": "soft_loss",
    "build_alignment_targets": "soft_loss",
    "one_hot_encode": "soft_loss",
    "log_softmax": "soft_loss",
    "softmax": "soft_loss",
    "standard_cross_entropy": "soft_loss",
    "normalize_mode": "soft_loss",
    "build_soft_loss": "soft_loss",
    "wordnet_soft_loss": "soft_loss",
    "latent_hierarchy_soft_loss": "soft_loss",
    "numpy_lca_alignment_loss": "soft_loss",
    "SoftLossConfig": "soft_loss",
    "DEFAULT_LAMBDA_WEIGHT": "soft_loss",
    "DEFAULT_TEMPERATURE": "soft_loss",
    "DEFAULT_ALIGNMENT_MODE": "soft_loss",
    "DEFAULT_REDUCTION": "soft_loss",
    "ALIGNMENT_MODES": "soft_loss",
    "REDUCTIONS": "soft_loss",
    # ---- linear_probe.py (Section 4.3.2) ---------------------------------
    "ProbeConfig": "linear_probe",
    "ProbeData": "linear_probe",
    "ProbeTrainResult": "linear_probe",
    "InterpolationPoint": "linear_probe",
    "InterpolationResult": "linear_probe",
    "ProbeExperimentResult": "linear_probe",
    "set_seed": "linear_probe",
    "build_linear_warmup_cosine_scheduler": "linear_probe",
    "build_optimizer": "linear_probe",
    "make_labeled_folds": "linear_probe",
    "make_feature_loader": "linear_probe",
    "build_linear_probe": "linear_probe",
    "get_probe_weights": "linear_probe",
    "get_probe_bias": "linear_probe",
    "set_probe_weights": "linear_probe",
    "build_hierarchy_soft_loss": "linear_probe",
    "probe_logits": "linear_probe",
    "probe_accuracy": "linear_probe",
    "probe_top5_accuracy": "linear_probe",
    "evaluate_probe": "linear_probe",
    "train_linear_probe": "linear_probe",
    "interpolate_probes": "linear_probe",
    "interpolation_grid": "linear_probe",
    "interpolate_and_evaluate": "linear_probe",
    "run_linear_probe_experiment": "linear_probe",
    "load_probe_data_from_cache": "linear_probe",
    "DEFAULT_LEARNING_RATE": "linear_probe",
    "DEFAULT_BATCH_SIZE": "linear_probe",
    "DEFAULT_EPOCHS": "linear_probe",
    "DEFAULT_WEIGHT_DECAY": "linear_probe",
    "DEFAULT_WARMUP_TYPE": "linear_probe",
    "DEFAULT_WARMUP_LR": "linear_probe",
    "DEFAULT_WARMUP_RATIO": "linear_probe",
    "DEFAULT_SCHEDULER": "linear_probe",
    "DEFAULT_INTERP_GRID": "linear_probe",
    # ---- prompt_engineering.py (Section 4.3.3 / Table 14) ----------------
    "PromptTemplate": "prompt_engineering",
    "PromptConfig": "prompt_engineering",
    "PromptEncoder": "prompt_engineering",
    "OpenAICLIPEncoder": "prompt_engineering",
    "OpenCLIPEncoder": "prompt_engineering",
    "CallableTextEncoder": "prompt_engineering",
    "sanitize_class_name": "prompt_engineering",
    "node_display_name": "prompt_engineering",
    "class_ancestor_names": "prompt_engineering",
    "build_shuffled_ancestors": "prompt_engineering",
    "format_prompt": "prompt_engineering",
    "build_prompt": "prompt_engineering",
    "build_prompt_texts": "prompt_engineering",
    "build_all_prompt_texts": "prompt_engineering",
    "build_encoder": "prompt_engineering",
    "encode_text_prompts": "prompt_engineering",
    "encode_image_loader": "prompt_engineering",
    "zero_shot_logits": "prompt_engineering",
    "zero_shot_metrics": "prompt_engineering",
    "evaluate_prompts": "prompt_engineering",
    "run_prompt_evaluation": "prompt_engineering",
    "check_against_table14": "prompt_engineering",
    "format_table14": "prompt_engineering",
    "PROMPT_TEMPLATES": "prompt_engineering",
    "DEFAULT_TEMPLATES": "prompt_engineering",
    "TABLE14_REFERENCE": "prompt_engineering",
    "PROMPT_DATASETS": "prompt_engineering",
    "DISPLAY_NAMES": "prompt_engineering",
    "DEFAULT_MAX_ANCESTORS": "prompt_engineering",
    "DEFAULT_LOGIT_SCALE": "prompt_engineering",
    "DEFAULT_SHUFFLE_SEED": "prompt_engineering",
}

# Symbols that may live in more than one sub-module: resolved in
# ``_SUBMODULES`` order, first present definition wins.
_COMMON_EXPORTS: Tuple[str, ...] = (
    "process_lca_matrix",  # soft_loss re-export / fallback
)

# Paper-facing / convenience aliases.
_ALIASES: Dict[str, str] = {
    # Algorithm 1 naming
    "Algorithm1Loss": "LCAAlignmentLoss",
    "LcaSoftLoss": "lca_alignment_loss",
    "TaxonomyAlignmentLoss": "lca_alignment_loss",
    "ReverseLCAMatrix": "reverse_lca_matrix",
    # linear probe naming
    "LinearProbeConfig": "ProbeConfig",
    "TrainLinearProbe": "train_linear_probe",
    "RunProbeExperiment": "run_linear_probe_experiment",
    "WeightInterpolation": "interpolate_and_evaluate",
    # prompt engineering naming
    "BuildPrompts": "build_all_prompt_texts",
    "RunPromptEvaluation": "run_prompt_evaluation",
    "Table14Reference": "TABLE14_REFERENCE",
    "PromptTemplates": "PROMPT_TEMPLATES",
}


# ---------------------------------------------------------------------------
# Lazy import machinery
# ---------------------------------------------------------------------------
def _import_submodule(name: str) -> Any:
    """Import a sibling sub-module, tolerating several ``sys.path`` layouts.

    Tries, in order:
      * ``src.alignment.<name>``
      * ``alignment.<name>``
      * ``<name>`` (flat layout with ``src`` itself on ``sys.path``)
    """
    candidates = (
        f"{__name__}.{name}",
        f"src.alignment.{name}",
        f"alignment.{name}",
        name,
    )
    last_error: Optional[BaseException] = None
    for candidate in candidates:
        try:  # pragma: no cover - layout dependent
            return importlib.import_module(candidate)
        except ImportError as exc:  # pragma: no cover - layout dependent
            last_error = exc
        except ModuleNotFoundError as exc:  # pragma: no cover
            last_error = exc
    if last_error is not None:  # pragma: no cover
        raise last_error
    raise ImportError(f"cannot import sub-module {name!r} of {__name__!r}")


def _resolve(name: str) -> Any:
    """Resolve ``name`` to a sub-module, an exported symbol or an alias."""
    if name in _SUBMODULES:
        return _import_submodule(name)

    alias = _ALIASES.get(name)
    if alias is not None:
        return _resolve(alias)

    # Explicit export table (first sub-module wins for duplicated names).
    provider = _EXPORTS.get(name)
    if provider is not None:
        module = _import_submodule(provider)
        if hasattr(module, name):
            return getattr(module, name)
        # fall through: the symbol may live elsewhere (interface drift)

    # Common helpers present in more than one sub-module.
    if name in _COMMON_EXPORTS:
        for submodule in _SUBMODULES:
            module = _import_submodule(submodule)
            if hasattr(module, name):
                return getattr(module, name)

    # Last resort: probe every sub-module in order.
    for submodule in _SUBMODULES:
        module = _import_submodule(submodule)
        if hasattr(module, name):
            return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __getattr__(name: str) -> Any:  # PEP 562
    """Lazily resolve attributes and memoize them in the module globals."""
    value = _resolve(name)
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    """Advertise lazily-resolved names for ``dir()`` / tab-completion."""
    names = set(globals()) | set(_SUBMODULES) | set(_EXPORTS) | set(_ALIASES)
    names |= set(_COMMON_EXPORTS)
    return sorted(names)


def __getattr_many__() -> Dict[str, Any]:
    """Eagerly resolve every advertised export (tests / introspection).

    Resolution failures are returned in-place rather than raised so callers can
    report partial progress.  Marked ``no cover`` because it is test-only.
    """
    resolved: Dict[str, Any] = {}
    for name in sorted(set(_EXPORTS) | set(_ALIASES) | set(_COMMON_EXPORTS)):
        try:  # pragma: no cover - test helper
            resolved[name] = __getattr__(name)
        except Exception as exc:  # pragma: no cover - test helper
            resolved[name] = exc
    return resolved


__all__ = (
    ["__version__"]
    + list(_SUBMODULES)
    + sorted(_EXPORTS)
    + sorted(_ALIASES)
    + list(_COMMON_EXPORTS)
)
