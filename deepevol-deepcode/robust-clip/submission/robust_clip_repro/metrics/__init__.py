"""Metrics package for the Robust CLIP reproduction.

Aggregates the metric layer used by every evaluation harness:

* :mod:`robust_clip_repro.metrics.classification` -- clean/robust top-1/top-5
  accuracy for zero-shot ImageNet evaluation.
* :mod:`robust_clip_repro.metrics.vqa` -- TextVQA / POPE / SQA-I accuracy,
  answer parsing, and the configurable numeric "score" used by the Addendum's
  precision-graded VQA attack schedule (argmin ground-truth selection).
* :mod:`robust_clip_repro.metrics.cider` -- CIDEr-D scoring plus the
  Addendum-mandated worst-case bookkeeping (CIDEr recomputed after *every*
  attack, only the per-sample minimum retained).
* :mod:`robust_clip_repro.metrics.jailbreak` -- human-in-the-loop grading
  export/summarization for the universal targeted jailbreak attack.  No label
  is ever fabricated: ungraded rows are reported as ``None``.

Nothing here defines attack internals or paper hyper-parameters; those live in
``robust_clip_repro.attacks`` and are tagged for provenance in ``configs/``.

Import is side-effect free and heavy dependencies (``torch``) stay lazy inside
the submodules, so ``import robust_clip_repro.metrics`` works on a bare Python
install.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    # submodules
    "classification",
    "vqa",
    "cider",
    "jailbreak",
    # helpers
    "available_metrics",
    "list_exports",
    "preload",
    # classification
    "RunningAccuracy",
    "ClassificationReport",
    "ComparisonRow",
    "evaluate_classification",
    "run_clean_pass",
    "run_robust_pass",
    "collect_batch",
    "make_logits_fn",
    "extract_logits",
    "extract_pixels",
    "extract_label",
    "accuracy",
    "top1_accuracy",
    "top5_accuracy",
    "topk_correct",
    "build_comparison_table",
    "format_table",
    # vqa
    "VQAScorer",
    "VQAReport",
    "VQASampleResult",
    "evaluate_vqa",
    "make_score_fn",
    "make_accuracy_fn",
    "parse_prediction",
    "answers_of",
    "prediction_accuracy",
    "normalize_answer",
    "aggregate_reports",
    "sample_result_from_schedule",
    # cider
    "CiderScorer",
    "CiderWorstCase",
    "WorstCaseCiderTracker",
    "cider_score",
    "cider_score_single",
    "cider_after_attack",
    "make_cider_fn",
    "references_from_samples",
    "cider_statistics",
    "worst_case_cider",
    "best_ground_truth",
    "cider_threshold_reached",
    # jailbreak
    "JailbreakSummary",
    "export_grading_sheet",
    "write_grading_sheet",
    "summarize_grading",
    "binary_summarization",
    "load_grading_sheet",
    "harmfulness_rate",
    "attack_success_rate",
    "normalize_label",
    "checkpoint_sheet_path",
    "GRADING_CRITERION",
]

_SUBMODULES: Tuple[str, ...] = ("classification", "vqa", "cider", "jailbreak")

# public name -> (submodule, attribute or None)
_EXPORTS: Dict[str, Tuple[str, Optional[str]]] = {
    # --- classification ---
    "RunningAccuracy": ("classification", None),
    "ClassificationReport": ("classification", None),
    "ComparisonRow": ("classification", None),
    "evaluate_classification": ("classification", None),
    "run_clean_pass": ("classification", None),
    "run_robust_pass": ("classification", None),
    "collect_batch": ("classification", None),
    "make_logits_fn": ("classification", None),
    "extract_logits": ("classification", None),
    "extract_pixels": ("classification", "extract_pixels"),
    "top1_accuracy": ("classification", None),
    "top5_accuracy": ("classification", None),
    "topk_correct": ("classification", None),
    "topk_predictions": ("classification", None),
    "per_sample_correct": ("classification", None),
    "accuracy_from_correct": ("classification", None),
    # note: `accuracy`, `extract_label`, `build_comparison_table` and
    # `format_table` exist in several submodules; the ambiguity is resolved
    # below with dedicated aliases for the non-default sources.
    "accuracy": ("classification", None),
    "extract_label": ("classification", None),
    "build_comparison_table": ("classification", None),
    "format_table": ("classification", None),

    # --- vqa ---
    "VQAScorer": ("vqa", None),
    "VQAReport": ("vqa", None),
    "VQASampleResult": ("vqa", None),
    "evaluate_vqa": ("vqa", None),
    "evaluate_from_records": ("vqa", None),
    "make_score_fn": ("vqa", None),
    "make_accuracy_fn": ("vqa", None),
    "make_schedule_fns": ("vqa", None),
    "parse_prediction": ("vqa", None),
    "clean_prediction": ("vqa", None),
    "answers_of": ("vqa", None),
    "prediction_accuracy": ("vqa", None),
    "sample_accuracy": ("vqa", None),
    "normalize_answer": ("vqa", None),
    "normalize_text": ("vqa", None),
    "aggregate_reports": ("vqa", None),
    "sample_result_from_schedule": ("vqa", None),
    "exact_match_accuracy": ("vqa", None),
    "soft_vqa_accuracy": ("vqa", None),
    "yes_no_prf": ("vqa", None),

    # --- cider ---
    "CiderScorer": ("cider", None),
    "CiderWorstCase": ("cider", None),
    "WorstCaseCiderTracker": ("cider", None),
    "cider_score": ("cider", None),
    "cider_d": ("cider", None),
    "cider_score_single": ("cider", None),
    "cider_after_attack": ("cider", None),
    "make_cider_fn": ("cider", None),
    "references_from_samples": ("cider", None),
    "cider_statistics": ("cider", None),
    "mean_cider": ("cider", None),
    "worst_case_cider": ("cider", None),
    "best_ground_truth": ("cider", None),
    "cider_threshold_reached": ("cider", None),
    "ngrams": ("cider", None),
    "tokenize": ("cider", None),

    # --- jailbreak ---
    "JailbreakSummary": ("jailbreak", None),
    "export_grading_sheet": ("jailbreak", None),
    "write_grading_sheet": ("jailbreak", None),
    "summarize_grading": ("jailbreak", None),
    "binary_summarization": ("jailbreak", None),
    "load_grading_sheet": ("jailbreak", None),
    "harmfulness_rate": ("jailbreak", None),
    "attack_success_rate": ("jailbreak", None),
    "normalize_label": ("jailbreak", None),
    "is_labeled": ("jailbreak", None),
    "is_harmful": ("jailbreak", None),
    "build_grading_rows": ("jailbreak", None),
    "checkpoint_sheet_path": ("jailbreak", None),
    "GRADING_CRITERION": ("jailbreak", None),
    "AFFIRMATIVE_IS_NOT_HARMFUL": ("jailbreak", None),
}

# Names that exist in more than one metric submodule: expose the alternates via
# explicit, unambiguous aliases so callers can pick the intended origin.
_ALIASES: Dict[str, Tuple[str, str]] = {
    "vqa_format_table": ("vqa", "format_table"),
    "vqa_build_comparison_table": ("vqa", "build_comparison_table"),
    "cider_mean": ("cider", "mean_cider"),
}


def available_metrics() -> List[str]:
    """Return the bundled metric submodule names."""
    return list(_SUBMODULES)


def list_exports() -> Dict[str, str]:
    """Map every lazy public name to its ``"submodule.attribute"`` source."""
    out: Dict[str, str] = {}
    for name, (sub, attr) in _EXPORTS.items():
        out[name] = f"{sub}.{attr or name}"
    for name, (sub, attr) in _ALIASES.items():
        out[name] = f"{sub}.{attr}"
    return dict(sorted(out.items()))


def preload(modules: Optional[Any] = None) -> Dict[str, bool]:
    """Eagerly import the requested metric submodules (all when ``None``).

    Returns a ``submodule -> success`` mapping and never raises on a missing
    optional dependency (e.g. ``torch``), which makes it usable from smoke tests.
    """
    if modules is None:
        targets = list(_SUBMODULES)
    elif isinstance(modules, str):
        targets = [modules]
    else:
        targets = list(modules)

    results: Dict[str, bool] = {}
    for name in targets:
        try:
            importlib.import_module(f"{__name__}.{name}")
            results[name] = True
        except Exception:  # pragma: no cover - optional dependency
            results[name] = False
    return results


def __getattr__(name: str) -> Any:
    """PEP 562 lazy resolver for metric submodules and re-exported symbols."""
    if name in _SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module

    if name in _EXPORTS:
        sub, attr = _EXPORTS[name]
        module = importlib.import_module(f"{__name__}.{sub}")
        value = getattr(module, attr or name)
        globals()[name] = value
        return value

    if name in _ALIASES:
        sub, attr = _ALIASES[name]
        module = importlib.import_module(f"{__name__}.{sub}")
        value = getattr(module, attr)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(list(globals().keys()) + list(__all__) + list(_SUBMODULES)))
