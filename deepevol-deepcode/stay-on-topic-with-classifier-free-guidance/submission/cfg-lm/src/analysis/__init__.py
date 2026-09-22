"""CFG-LM analysis package (Section 5).

Re-exports the four Section 5 analysis modules -- sampling entropy
(``entropy.py``), top-p token overlap (``overlap.py``), continuation-only
perplexity (``perplexity.py``) and the per-step vocabulary re-ranking
walkthrough of Table 3 (``visualize.py``) -- into a single ``src.analysis``
namespace.

Every submodule import is guarded so that the pure-NumPy math layers stay
importable (and unit-testable on CPU) even when ``torch`` / ``transformers`` /
``scipy`` / a model checkpoint are unavailable.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

__version__ = "0.1.0"

__all__: List[str] = [
    "available_exports",
    "__version__",
]

# ---------------------------------------------------------------------------
# entropy.py
# ---------------------------------------------------------------------------
_ENTROPY_EXPORTS = (
    # classes
    "EntropyStats",
    "EntropyComparison",
    "EntropyAnalyzer",
    # functions
    "entropy",
    "shannon_entropy",
    "entropy_from_logits",
    "entropy_from_logprobs",
    "top_p_token_count",
    "effective_vocab_size",
    "mean_entropy",
    "sample_entropy_of_distributions",
    "concatenate_stats",
    "batch_mean_entropy",
    "entropy_report",
    "format_entropy_table",
    "check_against_paper",
    "summarize_entropy",
    # constants
    "ANALYSIS_GAMMA",
    "TOP_P",
    "CFG_ENTROPY_MEAN",
    "VANILLA_ENTROPY_MEAN",
    "ENTROPY_TOLERANCE",
    "ENTROPY_MODES",
    "MODE_GAMMAS",
    "MODE_LABELS",
)

# ---------------------------------------------------------------------------
# overlap.py
# ---------------------------------------------------------------------------
_OVERLAP_EXPORTS = (
    # classes
    "TopPSetStats",
    "OverlapStats",
    "OverlapComparison",
    "OverlapAnalyzer",
    # functions
    "guided_distribution",
    "top_p_token_set",
    "top_p_token_sets",
    "top_p_token_count",
    "overlap_fraction",
    "jaccard",
    "intersection_size",
    "top_p_token_overlap",
    "rankdata",
    "pearson",
    "spearman_correlation",
    "spearman_with_pvalue",
    "compare_token_sets",
    "overlap_series",
    "spearman_difficulty",
    "difficulty_correlation",
    "spearman_by_length_bin",
    "dataset_similarity_table",
    "most_least_similar_examples",
    "format_overlap_table",
    "summarize_overlap",
    "overlap_report",
    "check_against_paper",
    "per_sample_entropy_proxy",
    # constants
    "CFG_VANILLA_OVERLAP",
    "SPEARMAN_THRESHOLD",
    "OVERLAP_TOLERANCE",
    "OVERLAP_MODES",
)

# ---------------------------------------------------------------------------
# perplexity.py
# ---------------------------------------------------------------------------
_PERPLEXITY_EXPORTS = (
    # classes
    "PerplexityStats",
    "PerplexityComparison",
    "PerplexityAnalyzer",
    # functions
    "log_probs_from_logits",
    "perplexity_from_logprobs",
    "token_perplexities",
    "mean_nll",
    "pearson",
    "spearman_correlation",
    "rankdata",
    "regression_slope",
    "correlation_table",
    "perplexity_report",
    "check_against_paper",
    "format_ppl_table",
    "summarize_ppl",
    # constants
    "CFG_PPL_VANILLA_CORR",
    "CFG_PPL_INSTRUCT_CORR",
    "PPL_TOLERANCE",
    "PPL_MODES",
    "CORRELATION_PAIRS",
)

# ---------------------------------------------------------------------------
# visualize.py
# ---------------------------------------------------------------------------
_VISUALIZE_EXPORTS = (
    # classes
    "TokenRanking",
    "StepRanking",
    "VocabReorderer",
    # functions
    "rank_vocabulary",
    "encouragement_scores",
    "paper_difference",
    "guided_logprobs",
    "reference_logprobs",
    "top_and_bottom_ids",
    "rank_step",
    "rank_from_logits",
    "run_table3_walkthrough",
    "table3_columns",
    "format_table3",
    "table3_dataframe",
    "count_expected_hits",
    "ranking_report",
    "check_against_paper",
    "summarize_rankings",
    "save_rankings",
    "load_rankings",
    "plot_rankings",
    "last_position",
    "decode_token_ids",
    # constants
    "DRAGON_PROMPT",
    "DRAGON_NEGATIVE_PROMPT",
    "TABLE3_GAMMA",
    "TOP_K_DISPLAY",
    "TABLE3_STEPS",
    "EXPECTED_ENCOURAGED",
    "EXPECTED_DISCOURAGED",
    "REFERENCE_MODES",
)


def _reexport(module_name: str, names) -> bool:
    """Import ``src.analysis.<module_name>`` and copy its available names.

    Returns ``True`` when the submodule imported successfully.
    """
    try:
        import importlib

        module = importlib.import_module(f"{__name__}.{module_name}")
    except Exception as exc:  # pragma: no cover - depends on environment
        logger.debug("analysis.%s unavailable: %s", module_name, exc)
        return False
    for name in names:
        if hasattr(module, name):
            globals()[name] = getattr(module, name)
            if name not in __all__:
                __all__.append(name)
    return True


_AVAILABLE: Dict[str, bool] = {
    "entropy": _reexport("entropy", _ENTROPY_EXPORTS),
    "overlap": _reexport("overlap", _OVERLAP_EXPORTS),
    "perplexity": _reexport("perplexity", _PERPLEXITY_EXPORTS),
    "visualize": _reexport("visualize", _VISUALIZE_EXPORTS),
}


def available_exports() -> Dict[str, bool]:
    """Report which analysis submodules imported successfully.

    Scripts can call this to warn (instead of crash) when optional
    dependencies such as ``torch`` / ``scipy`` / ``matplotlib`` are missing.
    """
    return dict(_AVAILABLE)


def get_analysis_anchors() -> Dict[str, Any]:
    """Return the paper reference values used to validate Section 5 results.

    Keys mirror the constants defined in the individual analysis modules and
    fall back to the paper's stated numbers when a module is unavailable.
    """
    anchors: Dict[str, Any] = {
        "cfg_entropy_mean": globals().get("CFG_ENTROPY_MEAN", 4.7),
        "vanilla_entropy_mean": globals().get("VANILLA_ENTROPY_MEAN", 5.49),
        "cfg_vanilla_overlap": globals().get("CFG_VANILLA_OVERLAP", 0.5),
        "cfg_ppl_vanilla_corr": globals().get("CFG_PPL_VANILLA_CORR", 0.94),
        "cfg_ppl_instruct_corr": globals().get("CFG_PPL_INSTRUCT_CORR", 0.70),
        "analysis_gamma": globals().get("ANALYSIS_GAMMA", 1.5),
    }
    return anchors


__all__.extend(["get_analysis_anchors"])
