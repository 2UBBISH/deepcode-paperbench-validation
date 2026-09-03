"""Unified baseline evaluation utilities for the FRE paper baselines.

This module is the canonical public API for evaluating the comparison methods
from Table 1 of the paper: GC-IQL, GC-BC, OPAL, FB, and SF.  The actual
evaluation loops live in :mod:`fre.pipeline.evaluate_baselines`; this module
re-exports them so downstream code can use ``from fre.baselines.baseline_eval
import evaluate_baseline`` without depending on pipeline-internal paths.
"""

from __future__ import annotations

from fre.pipeline.evaluate_baselines import (
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
    "evaluate_baseline",
    "evaluate_all_baselines",
    "evaluate_gc_agent",
    "evaluate_regression_agent",
    "evaluate_fb_agent",
    "evaluate_sf_agent",
    "evaluate_opal_agent",
    "_FixedConditionAgent",
    "_CallablePolicyAdapter",
    "build_parser",
    "main",
]
