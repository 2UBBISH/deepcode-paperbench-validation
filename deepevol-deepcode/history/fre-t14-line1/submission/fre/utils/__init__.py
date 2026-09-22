"""Utility subpackage for the FRE reproduction.

This package bundles dependency-light helpers shared by the training and
evaluation code:

* :mod:`fre.utils.normalization` -- return normalization to the paper's 0-100
  scale, mean/std aggregation across seeds, and comparisons against the
  published Table 1 / Table 4 reference numbers.
* :mod:`fre.utils.logging` -- console/JSONL/TensorBoard metric logging, run
  directory creation and JSON/CSV/NPZ artifact persistence.

The package is intentionally import-light (numpy only, TensorBoard optional) so
that reporting/evaluation scripts can import it without pulling in torch/gym.

Usage::

    from fre.utils import normalize_return, ResultAccumulator, MetricLogger
    from fre.utils import TABLE1_FRE_TARGETS, aggregate_rows
"""

from __future__ import annotations

from typing import List

from fre.utils.normalization import (
    NORMALIZED_MAX,
    NORMALIZED_MIN,
    SUCCESS_BONUS,
    TABLE1_AGGREGATE_TARGETS,
    TABLE1_FRE_TARGETS,
    TABLE4_FRE_TARGETS,
    ResultAccumulator,
    aggregate_domain_scores,
    aggregate_evaluations,
    aggregate_rows,
    aggregate_seeds,
    aggregate_task_scores,
    coefficient_of_variation,
    compare_to_reference,
    denormalize_return,
    format_mean_std,
    format_score,
    mean_std,
    normalize_return,
    normalize_returns,
    normalize_with_bounds,
    score_within_band,
    seeded_rng,
    task_bounds,
    to_markdown_table,
)
from fre.utils.logging import (
    AverageMeter,
    MetricLogger,
    TableLogger,
    Timer,
    as_scalar,
    configure_logging,
    flatten_dict,
    format_metrics,
    get_logger,
    load_json,
    make_run_dir,
    result_row,
    save_csv,
    save_json,
    save_numpy,
    summarize_seeds,
)

__all__: List[str] = [
    # --- fre.utils.normalization -------------------------------------------------
    "normalize_return",
    "normalize_returns",
    "normalize_with_bounds",
    "denormalize_return",
    "task_bounds",
    "mean_std",
    "coefficient_of_variation",
    "aggregate_seeds",
    "aggregate_evaluations",
    "aggregate_task_scores",
    "aggregate_domain_scores",
    "aggregate_rows",
    "ResultAccumulator",
    "format_mean_std",
    "format_score",
    "compare_to_reference",
    "score_within_band",
    "to_markdown_table",
    "seeded_rng",
    "NORMALIZED_MIN",
    "NORMALIZED_MAX",
    "SUCCESS_BONUS",
    "TABLE1_FRE_TARGETS",
    "TABLE1_AGGREGATE_TARGETS",
    "TABLE4_FRE_TARGETS",
    # --- fre.utils.logging -------------------------------------------------------
    "MetricLogger",
    "AverageMeter",
    "TableLogger",
    "Timer",
    "configure_logging",
    "get_logger",
    "as_scalar",
    "flatten_dict",
    "format_metrics",
    "make_run_dir",
    "save_json",
    "load_json",
    "save_csv",
    "save_numpy",
    "summarize_seeds",
    "result_row",
]
