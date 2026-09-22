"""LBCS reproduction utilities package.

This subpackage aggregates the glue layer that every experiment driver, the CLI
(``main.py``) and the orchestrator (``scripts/run_all.py``) rely on:

* ``lbcs_repro.utils.metrics``      -- objectives ``f1``/``f2``, accuracy, coreset size,
  per-data-point accuracy and ``mean +- std`` aggregation (paper report protocol).
* ``lbcs_repro.utils.seed``         -- deterministic seeding, per-repeat seed derivation,
  seeded DataLoader generators and RNG scoping.
* ``lbcs_repro.utils.logging``      -- logging setup, metric tracking and artifact persistence.
* ``lbcs_repro.utils.checkpoint``   -- checkpoint save/load helpers (optional; imported
  defensively so the package still imports before/without it).

None of these modules contain paper specific formulas; the objectives themselves live in
``lbcs_repro.lbcs.objectives``.  This package simply re-exports the shared measurement,
reproducibility and telemetry surface.

Scope note: ImageNet-1k (Sec. 5.4), continual learning (Appendix E.5) and streaming
(Appendix E.6) are explicitly out of scope for this reproduction and are not implemented
here.
"""

from __future__ import annotations

from .logging import (  # noqa: F401
    ARTIFACT_SUFFIXES,
    DEFAULT_DATE_FORMAT,
    DEFAULT_LOG_FORMAT,
    DEFAULT_LOG_LEVEL,
    DEFAULT_OUTPUT_DIR,
    ExperimentLogger,
    LoggingConfig,
    MetricRecord,
    MetricTracker,
    ROOT_LOGGER_NAME,
    configure_logging,
    describe_config,
    format_value_table,
    get_logger,
    log_config,
    log_kv,
    log_section,
    setup_logging,
    to_jsonable,
)
from .metrics import (  # noqa: F401
    ACCURACY_LABEL,
    AggregateMeasurement,
    LBCS_LABEL,
    MEASUREMENT_COLUMNS,
    Measurement,
    PAPER_REPEATS,
    PER_POINT_LABEL,
    SIZE_LABEL,
    accuracy,
    accuracy_gap,
    accuracy_per_datapoint,
    aggregate_measurements,
    aggregate_records,
    average_accuracy_per_datapoint,
    best_method,
    binarize_mask,
    compressed_to_full_mask,
    confidence_interval,
    coreset_size,
    count_failures,
    decreased,
    evaluate_accuracy,
    f1_value,
    f2_value,
    format_mean_std,
    mask_size,
    mean_of,
    mean_std,
    non_decreasing,
    non_increasing,
    objective_vector,
    per_point_accuracy,
    per_sample_cross_entropy,
    rank_methods,
    records_to_rows,
    relative_size_reduction,
    render_table,
    selected_indices,
    size_reduction,
    std_error,
    summarize,
    top1_accuracy,
)
from .seed import (  # noqa: F401
    DEFAULT_SEED,
    describe_seed,
    make_generator,
    repeat_seeds,
    resolve_seed,
    seed_everything,
    seed_scope,
    set_deterministic,
    set_seed,
    torch_available,
    worker_init_fn,
)

# ``checkpoint`` is imported defensively: it is part of the glue layer and its helpers are
# convenience-only (drivers can still run without them).  Any failure (module not yet
# written, missing optional dependency) must not break ``import lbcs_repro.utils``.
try:  # pragma: no cover - defensive import
    from .checkpoint import (  # noqa: F401
        CheckpointManager,
        checkpoint_state,
        list_checkpoints,
        load_checkpoint,
        load_model_state,
        restore_model,
        save_checkpoint,
        save_model_state,
        save_results,
    )

    _CHECKPOINT_AVAILABLE = True
except Exception:  # pragma: no cover - defensive import
    CheckpointManager = None  # type: ignore[assignment]
    checkpoint_state = None  # type: ignore[assignment]
    list_checkpoints = None  # type: ignore[assignment]
    load_checkpoint = None  # type: ignore[assignment]
    load_model_state = None  # type: ignore[assignment]
    restore_model = None  # type: ignore[assignment]
    save_checkpoint = None  # type: ignore[assignment]
    save_model_state = None  # type: ignore[assignment]
    save_results = None  # type: ignore[assignment]
    _CHECKPOINT_AVAILABLE = False


TORCH_AVAILABLE = torch_available()


__all__ = [
    # submodules
    "logging",
    "metrics",
    "seed",
    "checkpoint",
    # metrics: objective / measurement
    "Measurement",
    "AggregateMeasurement",
    "binarize_mask",
    "coreset_size",
    "f1_value",
    "f2_value",
    "mask_size",
    "selected_indices",
    "compressed_to_full_mask",
    "objective_vector",
    "size_reduction",
    "relative_size_reduction",
    "top1_accuracy",
    "evaluate_accuracy",
    "accuracy",
    "per_sample_cross_entropy",
    "accuracy_per_datapoint",
    "average_accuracy_per_datapoint",
    "per_point_accuracy",
    # metrics: aggregation / reporting
    "mean_std",
    "format_mean_std",
    "summarize",
    "std_error",
    "confidence_interval",
    "aggregate_records",
    "aggregate_measurements",
    "mean_of",
    "count_failures",
    "decreased",
    "non_increasing",
    "non_decreasing",
    "best_method",
    "rank_methods",
    "accuracy_gap",
    "render_table",
    "records_to_rows",
    "MEASUREMENT_COLUMNS",
    "PAPER_REPEATS",
    "LBCS_LABEL",
    "SIZE_LABEL",
    "ACCURACY_LABEL",
    "PER_POINT_LABEL",
    # seeding
    "set_seed",
    "seed_everything",
    "set_deterministic",
    "resolve_seed",
    "repeat_seeds",
    "make_generator",
    "worker_init_fn",
    "seed_scope",
    "describe_seed",
    "torch_available",
    "DEFAULT_SEED",
    "TORCH_AVAILABLE",
    # logging / telemetry
    "setup_logging",
    "configure_logging",
    "get_logger",
    "log_section",
    "log_kv",
    "log_config",
    "describe_config",
    "to_jsonable",
    "format_value_table",
    "LoggingConfig",
    "MetricRecord",
    "MetricTracker",
    "ExperimentLogger",
    "DEFAULT_LOG_FORMAT",
    "DEFAULT_DATE_FORMAT",
    "DEFAULT_LOG_LEVEL",
    "ROOT_LOGGER_NAME",
    "DEFAULT_OUTPUT_DIR",
    "ARTIFACT_SUFFIXES",
    # checkpoint (may be None if the module is unavailable)
    "CheckpointManager",
    "checkpoint_state",
    "list_checkpoints",
    "load_checkpoint",
    "load_model_state",
    "restore_model",
    "save_checkpoint",
    "save_model_state",
    "save_results",
]
