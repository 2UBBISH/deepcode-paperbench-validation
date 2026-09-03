"""Baseline algorithms compared against Functional Reward Encodings (FRE).

This package collects re-implementations/adapters for the methods used in the
paper's empirical comparison:

- Forward-Backward (FB) representations (:mod:`fre.baselines.fb`)
- Successor Features (SF) with ICM features (:mod:`fre.baselines.sf`)
- OPAL offline skill discovery (:mod:`fre.baselines.opal`)

The actual evaluation utilities live in
:mod:`fre.pipeline.evaluate_baselines` and are re-exported through
:mod:`fre.baselines.baseline_eval` for a stable public API.
"""

from fre.baselines.fb import FB, ForwardBackward, train_fb_agent
from fre.baselines.opal import OPAL, train_opal_agent
from fre.baselines.sf import ICMFeatures, SF, SuccessorFeatures, train_sf_agent

# Re-export the unified evaluation facade so callers can do:
#   from fre.baselines import evaluate_baseline, evaluate_all_baselines
from fre.baselines.baseline_eval import (
    _CallablePolicyAdapter,
    _FixedConditionAgent,
    build_parser,
    evaluate_all_baselines,
    evaluate_baseline,
    evaluate_fb_agent,
    evaluate_gc_agent,
    evaluate_opal_agent,
    evaluate_regression_agent,
    evaluate_sf_agent,
    main,
)

__all__ = [
    # FB
    "ForwardBackward",
    "FB",
    "train_fb_agent",
    # SF
    "SuccessorFeatures",
    "SF",
    "ICMFeatures",
    "train_sf_agent",
    # OPAL
    "OPAL",
    "train_opal_agent",
    # Evaluation facade
    "evaluate_baseline",
    "evaluate_all_baselines",
    "evaluate_fb_agent",
    "evaluate_sf_agent",
    "evaluate_gc_agent",
    "evaluate_opal_agent",
    "evaluate_regression_agent",
    "_CallablePolicyAdapter",
    "_FixedConditionAgent",
    "build_parser",
    "main",
]
