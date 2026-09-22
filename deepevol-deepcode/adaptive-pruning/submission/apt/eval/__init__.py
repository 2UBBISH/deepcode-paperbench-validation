"""APT evaluation package.

Aggregates the evaluation utilities used to reproduce the APT paper's tables
and figures:

* :mod:`apt.eval.metrics` -- end-task metrics (GLUE accuracy/F1/MCC/Spearman,
  SQuAD v2 EM/F1, CNN/DM ROUGE) plus the paper's reference scores and
  relative-accuracy helpers.
* :mod:`apt.eval.efficiency` -- training/inference efficiency (time-to-accuracy,
  peak memory via ``torch.cuda.max_memory_allocated()``, inference latency /
  throughput) and FT normalization.
* :mod:`apt.eval.run_eval` -- top-level evaluation harness (paper Sec. 5.3).

Each submodule is imported defensively so that importing :mod:`apt.eval` still
works when an optional heavy dependency (``sklearn``, ``scipy``,
``rouge_score``, ``torch``) is missing, or before a submodule has been written.
``AVAILABLE`` records which submodules imported successfully.
"""

from __future__ import annotations

from typing import Dict, List

__all__: List[str] = ["AVAILABLE", "available"]

AVAILABLE: Dict[str, bool] = {}


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
try:  # pragma: no cover - availability depends on environment
    from . import metrics  # noqa: F401
    from .metrics import (  # noqa: F401
        FT_REFERENCES,
        GLUE_ACCURACY_TASKS,
        GLUE_BIG_TASKS,
        GLUE_F1_TASKS,
        GLUE_PRIMARY_METRIC,
        GLUE_REGRESSION_TASKS,
        GLUE_SMALL_TASKS,
        GLUE_TASKS,
        RAW_EFFICIENCY,
        ROUGE_KEYS,
        SEQ2SEQ_TASKS,
        SQUAD_TASKS,
        ReferenceScores,
        accuracy,
        aggregate_glue_metrics,
        binary_f1,
        compute_cnndm_metrics,
        compute_glue_metrics,
        compute_metrics,
        compute_rouge,
        compute_squad_metrics,
        format_percent,
        glue_average,
        is_big_glue_task,
        is_glue_task,
        is_regression_task,
        is_seq2seq_task,
        is_squad_task,
        matthews_corrcoef,
        metric_for_display,
        normalize_task_name,
        primary_metric,
        primary_metric_name,
        relative_accuracy,
        relative_accuracy_sst2_mnli,
        spearman_correlation,
        squad_exact_match,
        squad_f1,
    )

    AVAILABLE["metrics"] = True
    __all__ += [
        "metrics",
        "FT_REFERENCES",
        "GLUE_ACCURACY_TASKS",
        "GLUE_BIG_TASKS",
        "GLUE_F1_TASKS",
        "GLUE_PRIMARY_METRIC",
        "GLUE_REGRESSION_TASKS",
        "GLUE_SMALL_TASKS",
        "GLUE_TASKS",
        "RAW_EFFICIENCY",
        "ROUGE_KEYS",
        "SEQ2SEQ_TASKS",
        "SQUAD_TASKS",
        "ReferenceScores",
        "accuracy",
        "aggregate_glue_metrics",
        "binary_f1",
        "compute_cnndm_metrics",
        "compute_glue_metrics",
        "compute_metrics",
        "compute_rouge",
        "compute_squad_metrics",
        "format_percent",
        "glue_average",
        "is_big_glue_task",
        "is_glue_task",
        "is_regression_task",
        "is_seq2seq_task",
        "is_squad_task",
        "matthews_corrcoef",
        "metric_for_display",
        "normalize_task_name",
        "primary_metric",
        "primary_metric_name",
        "relative_accuracy",
        "relative_accuracy_sst2_mnli",
        "spearman_correlation",
        "squad_exact_match",
        "squad_f1",
    ]
except Exception as exc:  # noqa: BLE001
    import warnings

    warnings.warn(f"apt.eval.metrics unavailable: {exc}")
    AVAILABLE["metrics"] = False


# ---------------------------------------------------------------------------
# efficiency
# ---------------------------------------------------------------------------
try:  # pragma: no cover - availability depends on environment
    from . import efficiency  # noqa: F401
    from .efficiency import (  # noqa: F401
        DEFAULT_SEQUENCE_LENGTH,
        INFERENCE_BATCH_SIZES,
        LLAMA_7B_INFERENCE_BATCH_SIZE,
        LLAMA_13B_INFERENCE_BATCH_SIZE,
        METRIC_KEYS,
        SMALL_MODEL_INFERENCE_BATCH_SIZE,
        TABLE11_RAW,
        TABLE2_RELATIVE,
        TTA_FRACTION,
        EfficiencyResult,
        PeakMemoryTracker,
        TimeToAccuracy,
        TimeToAccuracyTracker,
        Timer,
        TrainingEfficiencyTracker,
        benchmark_inference,
        cuda_available,
        current_memory_mb,
        efficiency_row,
        efficiency_summary,
        format_efficiency,
        free_memory_mb,
        inference_batch_size_for,
        make_dummy_batch,
        measure_inference,
        measure_inference_latency,
        measure_inference_memory,
        measure_inference_throughput,
        measure_peak_memory,
        measure_runtime,
        measure_training_efficiency,
        model_size_breakdown,
        model_size_mb,
        normalize_efficiency,
        parameter_count,
        peak_memory_bytes,
        peak_memory_mb,
        relative_from_table11,
        relative_metric,
        reset_peak_memory,
        speedup,
        synchronize,
        time_to_accuracy,
    )

    AVAILABLE["efficiency"] = True
    __all__ += [
        "efficiency",
        "DEFAULT_SEQUENCE_LENGTH",
        "INFERENCE_BATCH_SIZES",
        "LLAMA_7B_INFERENCE_BATCH_SIZE",
        "LLAMA_13B_INFERENCE_BATCH_SIZE",
        "METRIC_KEYS",
        "SMALL_MODEL_INFERENCE_BATCH_SIZE",
        "TABLE11_RAW",
        "TABLE2_RELATIVE",
        "TTA_FRACTION",
        "EfficiencyResult",
        "PeakMemoryTracker",
        "TimeToAccuracy",
        "TimeToAccuracyTracker",
        "Timer",
        "TrainingEfficiencyTracker",
        "benchmark_inference",
        "cuda_available",
        "current_memory_mb",
        "efficiency_row",
        "efficiency_summary",
        "format_efficiency",
        "free_memory_mb",
        "inference_batch_size_for",
        "make_dummy_batch",
        "measure_inference",
        "measure_inference_latency",
        "measure_inference_memory",
        "measure_inference_throughput",
        "measure_peak_memory",
        "measure_runtime",
        "measure_training_efficiency",
        "model_size_breakdown",
        "model_size_mb",
        "normalize_efficiency",
        "parameter_count",
        "peak_memory_bytes",
        "peak_memory_mb",
        "relative_from_table11",
        "relative_metric",
        "reset_peak_memory",
        "speedup",
        "synchronize",
        "time_to_accuracy",
    ]
except Exception as exc:  # noqa: BLE001
    import warnings

    warnings.warn(f"apt.eval.efficiency unavailable: {exc}")
    AVAILABLE["efficiency"] = False


# ---------------------------------------------------------------------------
# run_eval (top-level harness, paper Sec. 5.3)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - availability depends on environment
    from . import run_eval  # noqa: F401
    from .run_eval import (  # noqa: F401
        ATTACK_BASELINE_METHODS,
        BASELINE_METHODS,
        METHOD_ORDER,
        TABLE2_REFERENCES,
        EvalConfig,
        EvalOutcome,
        EvaluationResult,
        evaluate_apt_model,
        evaluate_method,
        evaluate_model,
        evaluate_predictions,
        evaluate_task,
        format_results_table,
        load_reference_constants,
        relative_to_ft,
        run_all_tasks,
        run_evaluation,
        save_results,
    )

    AVAILABLE["run_eval"] = True
    __all__ += [
        "run_eval",
        "ATTACK_BASELINE_METHODS",
        "BASELINE_METHODS",
        "METHOD_ORDER",
        "TABLE2_REFERENCES",
        "EvalConfig",
        "EvalOutcome",
        "EvaluationResult",
        "evaluate_apt_model",
        "evaluate_method",
        "evaluate_model",
        "evaluate_predictions",
        "evaluate_task",
        "format_results_table",
        "load_reference_constants",
        "relative_to_ft",
        "run_all_tasks",
        "run_evaluation",
        "save_results",
    ]
except Exception as exc:  # noqa: BLE001
    import warnings

    warnings.warn(f"apt.eval.run_eval unavailable: {exc}")
    AVAILABLE["run_eval"] = False


def available() -> Dict[str, bool]:
    """Return a copy of the submodule availability map."""
    return dict(AVAILABLE)
