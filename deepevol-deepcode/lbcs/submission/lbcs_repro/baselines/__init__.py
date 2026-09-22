"""Coreset-selection baselines used by the LBCS reproduction.

This package collects every competitor that LBCS is compared against:

* ``base``            -- shared ``BaselineSelector`` interface, score->mask helpers,
                         per-sample losses/gradient norms, reference-model training.
* ``uniform``         -- Uniform random selection (Appendix D.1).
* ``el2n``            -- EL2N: norm of ``softmax(h(x)) - onehot(y)`` (Appendix D.1).
* ``grand``           -- GraNd: norm of the per-example loss gradient (Appendix D.1).
* ``moderate``        -- Moderate: distance-to-class-center scores near the median.
* ``influential``     -- Influential: influence-function generalization-gap scores.
* ``ccs``             -- CCS: one-shot coverage-plus-importance selection.
* ``probabilistic``   -- Zhou et al. (2022) probabilistic bilevel coreset
                         (Figure 1 comparator and a Section 5.2 baseline).
* ``weighted_bilevel``-- Eq. (3) fixed-size and Eq. (4) weighted trivial bilevel
                         formulations used to reproduce the Figure 1 failure modes.

Besides re-exporting the individual selectors, this module exposes a small
registry (``BASELINE_REGISTRY``) and factory (``get_baseline`` / ``make_baseline``)
so experiment drivers can iterate the baseline suite by name instead of hard-coding
imports.

Scope note: ImageNet-1k (Section 5.4), continual learning (Appendix E.5) and
streaming (Appendix E.6) are out of scope for this reproduction.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

# --------------------------------------------------------------------------------------
# Shared infrastructure
# --------------------------------------------------------------------------------------
from .base import (
    BaselineSelector,
    ScoreBaseline,
    bottomk_indices,
    collect_predictions,
    forward_logits,
    gather_scores_by_index,
    indices_to_mask,
    median_indices,
    normalize_vector,
    per_sample_grad_norms,
    per_sample_losses,
    resolve_seed,
    set_seed,
    stratified_topk_indices,
    to_float_scores,
    topk_indices,
    train_reference_model,
)

# --------------------------------------------------------------------------------------
# Uniform random selection
# --------------------------------------------------------------------------------------
from .uniform import (
    UniformSampling,
    UniformSelector,
    uniform_indices,
    uniform_mask,
)

# --------------------------------------------------------------------------------------
# EL2N (Paul et al., 2021)
# --------------------------------------------------------------------------------------
from .el2n import (
    EL2N,
    EL2NSelector,
    el2n_indices,
    el2n_mask,
    el2n_scores,
    el2n_scores_batch,
    el2n_scores_from_logits,
    error_vectors,
)

# --------------------------------------------------------------------------------------
# GraNd (Paul et al., 2021)
# --------------------------------------------------------------------------------------
from .grand import (
    GraNd,
    GraNdSelector,
    grad_norm_scores,
    grand_indices,
    grand_mask,
    grand_scores,
    grand_scores_from_model,
)

# --------------------------------------------------------------------------------------
# Moderate (Xia et al., 2023)
# --------------------------------------------------------------------------------------
from .moderate import (
    Moderate,
    ModerateSelector,
    class_center_distances,
    class_centers,
    extract_features,
    median_closeness_order,
    moderate_indices,
    moderate_mask,
    moderate_scores,
)

# --------------------------------------------------------------------------------------
# Influential (Yang et al., 2023)
# --------------------------------------------------------------------------------------
from .influential import (
    Influential,
    InfluentialSelector,
    conjugate_gradient,
    generalization_gap_scores,
    hessian_diagonal,
    hessian_vector_product,
    influential_indices,
    influential_mask,
    influential_scores,
    inverse_hessian_vector,
    per_sample_grad_vectors,
    validation_grad_vector,
)

# --------------------------------------------------------------------------------------
# CCS (Zheng et al., 2023)
# --------------------------------------------------------------------------------------
from .ccs import (
    CCS,
    CCSSelector,
    allocate_class_budget,
    ccs_indices,
    ccs_mask,
    ccs_scores,
    ccs_select_indices,
    kcenter_greedy,
    min_distance_to_selected,
    normalize01,
    preserved_class_budgets,
)

# --------------------------------------------------------------------------------------
# Probabilistic bilevel coreset (Zhou et al., 2022) -- Figure 1 comparator / Section 5.2
# --------------------------------------------------------------------------------------
from .probabilistic import (
    Probabilistic,
    ProbabilisticBilevelSelector,
    ProbabilisticCoreset,
    ProbabilisticResult,
    ProbabilisticSelector,
    bernoulli_probabilities,
    clip_probabilities,
    expected_coreset_size,
    figure1_curves,
    gradient_norm_analysis,
    log_prob,
    mask_probability,
    probabilistic_bilevel,
    sample_mask,
    sample_masks,
    score_function_gradient,
    score_function_log_prob_gradient,
    zeta1,
    zeta2,
)

# --------------------------------------------------------------------------------------
# Trivial bilevel formulations of Section 2.1 / Figure 1: Eq. (3) and Eq. (4)
# --------------------------------------------------------------------------------------
from .weighted_bilevel import (
    Eq3Baseline,
    Eq3Selector,
    Eq4Baseline,
    Eq4Selector,
    TrivialBilevelResult,
    WeightedBilevelBaseline,
    WeightedBilevelSelector,
    combined_value,
    diagnose_failure_mode,
    eq3_curves,
    eq3_value,
    eq4_curves,
    eq4_value,
    figure1_comparison,
    trivial_bilevel_curves,
    weighted_bilevel,
)


# --------------------------------------------------------------------------------------
# Registry / factory
# --------------------------------------------------------------------------------------
#: Name -> selector class.  The canonical names are the ones printed in Tables 2/3.
BASELINE_REGISTRY: Dict[str, Type[BaselineSelector]] = {
    "uniform": UniformSelector,
    "el2n": EL2NSelector,
    "grand": GraNdSelector,
    "moderate": ModerateSelector,
    "influential": InfluentialSelector,
    "ccs": CCSSelector,
    "probabilistic": ProbabilisticSelector,
    "eq3": Eq3Selector,
    "eq4": Eq4Selector,
}

#: Lower-case aliases accepted by :func:`get_baseline` (paper names and shorthands).
BASELINE_ALIASES: Dict[str, str] = {
    "random": "uniform",
    "uniform random": "uniform",
    "el2n": "el2n",
    "grand": "grand",
    "gradient norm": "grand",
    "grad_norm": "grand",
    "moderate": "moderate",
    "influential": "influential",
    "influence": "influential",
    "ccs": "ccs",
    "coverage": "ccs",
    "prob": "probabilistic",
    "probabilistic": "probabilistic",
    "weighted": "probabilistic",
    "zhou": "probabilistic",
    "eq. (3)": "eq3",
    "eq.(3)": "eq3",
    "eq3": "eq3",
    "eq. (4)": "eq4",
    "eq.(4)": "eq4",
    "eq4": "eq4",
}

#: Suite used by Section 5.2 / 5.3 comparisons (score-based baselines + LBCS itself
#: is run separately by the drivers).
COMPARISON_SUITE: List[str] = [
    "Uniform",
    "EL2N",
    "GraNd",
    "Moderate",
    "Influential",
    "CCS",
    "Probabilistic",
]


def available_baselines() -> List[str]:
    """Return the canonical names registered in :data:`BASELINE_REGISTRY`."""
    return list(BASELINE_REGISTRY.keys())


def resolve_baseline_name(name: str) -> str:
    """Map a paper name / shorthand onto a canonical registry key.

    Raises
    ------
    KeyError
        If ``name`` is neither a canonical key nor a known alias.
    """
    if not isinstance(name, str):
        raise KeyError(f"Baseline name must be a string, got {type(name)!r}")
    key = name.strip().lower()
    if key in BASELINE_REGISTRY:
        return key
    if key in BASELINE_ALIASES:
        return BASELINE_ALIASES[key]
    raise KeyError(
        f"Unknown baseline {name!r}. Available: {sorted(BASELINE_REGISTRY)} "
        f"(aliases: {sorted(BASELINE_ALIASES)})"
    )


def get_baseline(name: str) -> Type[BaselineSelector]:
    """Return the selector *class* registered under ``name``."""
    return BASELINE_REGISTRY[resolve_baseline_name(name)]


def make_baseline(name: str, **kwargs: Any) -> BaselineSelector:
    """Instantiate the baseline registered under ``name``.

    Keyword arguments are forwarded verbatim to the selector constructor, so
    e.g. ``make_baseline("moderate", num_classes=10, seed=0)`` works.
    """
    return get_baseline(name)(**kwargs)


def build_baselines(
    names: Optional[List[str]] = None,
    **kwargs: Any,
) -> Dict[str, BaselineSelector]:
    """Instantiate several baselines at once, keyed by canonical name.

    Parameters
    ----------
    names:
        Baseline names (paper names or aliases).  Defaults to
        :data:`COMPARISON_SUITE`.
    **kwargs:
        Forwarded to every selector constructor; selectors that do not accept a
        given keyword must tolerate ``**kwargs`` (all LBCS baselines do).
    """
    selected = COMPARISON_SUITE if names is None else names
    instances: Dict[str, BaselineSelector] = {}
    for name in selected:
        key = resolve_baseline_name(name)
        if key in instances:
            continue
        instances[key] = BASELINE_REGISTRY[key](**kwargs)
    return instances


__all__ = [
    # registry / factory
    "BASELINE_REGISTRY",
    "BASELINE_ALIASES",
    "COMPARISON_SUITE",
    "available_baselines",
    "resolve_baseline_name",
    "get_baseline",
    "make_baseline",
    "build_baselines",
    # base infrastructure
    "BaselineSelector",
    "ScoreBaseline",
    "indices_to_mask",
    "topk_indices",
    "bottomk_indices",
    "median_indices",
    "stratified_topk_indices",
    "to_float_scores",
    "forward_logits",
    "per_sample_losses",
    "per_sample_grad_norms",
    "normalize_vector",
    "set_seed",
    "resolve_seed",
    "train_reference_model",
    "collect_predictions",
    "gather_scores_by_index",
    # uniform
    "UniformSelector",
    "UniformSampling",
    "uniform_indices",
    "uniform_mask",
    # EL2N
    "EL2NSelector",
    "EL2N",
    "el2n_scores",
    "el2n_scores_batch",
    "el2n_scores_from_logits",
    "el2n_indices",
    "el2n_mask",
    "error_vectors",
    # GraNd
    "GraNdSelector",
    "GraNd",
    "grand_scores",
    "grand_scores_from_model",
    "grand_indices",
    "grand_mask",
    "grad_norm_scores",
    # Moderate
    "ModerateSelector",
    "Moderate",
    "moderate_scores",
    "moderate_indices",
    "moderate_mask",
    "median_closeness_order",
    "class_centers",
    "class_center_distances",
    "extract_features",
    # Influential
    "InfluentialSelector",
    "Influential",
    "influential_scores",
    "influential_indices",
    "influential_mask",
    "per_sample_grad_vectors",
    "validation_grad_vector",
    "hessian_diagonal",
    "hessian_vector_product",
    "conjugate_gradient",
    "inverse_hessian_vector",
    "generalization_gap_scores",
    # CCS
    "CCSSelector",
    "CCS",
    "ccs_select_indices",
    "ccs_indices",
    "ccs_mask",
    "ccs_scores",
    "normalize01",
    "allocate_class_budget",
    "preserved_class_budgets",
    "min_distance_to_selected",
    "kcenter_greedy",
    # probabilistic bilevel (Zhou et al. 2022)
    "ProbabilisticSelector",
    "Probabilistic",
    "ProbabilisticCoreset",
    "ProbabilisticBilevelSelector",
    "ProbabilisticResult",
    "probabilistic_bilevel",
    "figure1_curves",
    "clip_probabilities",
    "bernoulli_probabilities",
    "sample_mask",
    "sample_masks",
    "mask_probability",
    "log_prob",
    "score_function_log_prob_gradient",
    "score_function_gradient",
    "expected_coreset_size",
    "zeta1",
    "zeta2",
    "gradient_norm_analysis",
    # trivial bilevel formulations of Section 2.1 / Figure 1
    "WeightedBilevelBaseline",
    "WeightedBilevelSelector",
    "Eq3Baseline",
    "Eq4Baseline",
    "Eq3Selector",
    "Eq4Selector",
    "TrivialBilevelResult",
    "eq3_value",
    "eq4_value",
    "combined_value",
    "weighted_bilevel",
    "trivial_bilevel_curves",
    "eq3_curves",
    "eq4_curves",
    "figure1_comparison",
    "diagnose_failure_mode",
]
