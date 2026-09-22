"""Evaluation package for BBox-Adapter.

This package aggregates the three evaluation modules described in the plan
(:mod:`bbox_adapter.eval.metrics`, :mod:`bbox_adapter.eval.cost` and
:mod:`bbox_adapter.eval.vram`) behind a single, stable import surface.

* :mod:`~bbox_adapter.eval.metrics` -- Accuracy, True + Info, Toxic %,
  Toxicity Prob %, Delta % (paper Sections 4.2-4.7, Appendix E, Table 4/6/7).
* :mod:`~bbox_adapter.eval.cost` -- token accounting and dollar costs per 1k
  questions (Section 4.4, Table 4).
* :mod:`~bbox_adapter.eval.vram` -- peak GPU VRAM measurement for the 0.1B
  adapter only (Table 6 + Addendum).

Everything here is a thin re-export layer; the heavy lifting lives in the
sibling modules.  Imports are defensive so that a partially installed
environment (e.g. no ``torch`` / no ``transformers``) still lets the metric
utilities be used for offline regression checks.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

__all__: List[str] = []


def _extend(names: Optional[List[str]]) -> None:
    """Append re-exported symbol names to ``__all__`` without duplicates."""
    if not names:
        return
    for name in names:
        if name not in __all__:
            __all__.append(name)


# ---------------------------------------------------------------------------
# metrics.py -- answer scoring, judges, paper reference tables
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised implicitly through the real environment
    from .metrics import (  # noqa: F401
        DATASET_METRIC,
        LOWER_IS_BETTER,
        METRIC_LABELS,
        PAPER_AVERAGE_DELTA,
        PAPER_COST_RATIOS,
        PAPER_COST_RATIOS_SINGLE_STEP,
        PAPER_FIG3,
        PAPER_TABLE2,
        PAPER_TABLE3,
        PAPER_TABLE4,
        PAPER_TABLE5,
        PAPER_TABLE6,
        PAPER_TABLE7,
        MetricReport,
        MetricsConfig,
        RobertaToxicityJudge,
        accuracy,
        accuracy_from_pairs,
        aggregate_results,
        aggregate_seeds,
        average_metric,
        compare_to_paper,
        compute_deltas,
        delta_percent,
        evaluate,
        evaluate_generations,
        format_report,
        format_results_table,
        improvement_ratio,
        is_lower_better,
        lookup_paper,
        mean,
        mock_toxicity_judge,
        population_std,
        reference_table,
        sample_stderr,
        sample_std,
        save_results,
        signed_delta,
        table_row,
        toxic_percentage,
        toxicity_metrics,
        toxicity_probability,
        true_info,
        truth_info,
    )

    _extend(
        [
            "DATASET_METRIC",
            "LOWER_IS_BETTER",
            "METRIC_LABELS",
            "PAPER_AVERAGE_DELTA",
            "PAPER_COST_RATIOS",
            "PAPER_COST_RATIOS_SINGLE_STEP",
            "PAPER_FIG3",
            "PAPER_TABLE2",
            "PAPER_TABLE3",
            "PAPER_TABLE4",
            "PAPER_TABLE5",
            "PAPER_TABLE6",
            "PAPER_TABLE7",
            "MetricReport",
            "MetricsConfig",
            "RobertaToxicityJudge",
            "accuracy",
            "accuracy_from_pairs",
            "aggregate_results",
            "aggregate_seeds",
            "average_metric",
            "compare_to_paper",
            "compute_deltas",
            "delta_percent",
            "evaluate",
            "evaluate_generations",
            "format_report",
            "format_results_table",
            "improvement_ratio",
            "is_lower_better",
            "lookup_paper",
            "mean",
            "mock_toxicity_judge",
            "population_std",
            "reference_table",
            "sample_stderr",
            "sample_std",
            "save_results",
            "signed_delta",
            "table_row",
            "toxic_percentage",
            "toxicity_metrics",
            "toxicity_probability",
            "true_info",
            "truth_info",
        ]
    )
    _METRICS_AVAILABLE = True
except Exception:  # pragma: no cover - defensive import guard
    _METRICS_AVAILABLE = False


# ---------------------------------------------------------------------------
# cost.py -- token accounting, $/1k questions, Table 4
# ---------------------------------------------------------------------------
try:  # pragma: no cover
    from .cost import (  # noqa: F401
        DEFAULT_PER_QUESTIONS,
        DEFAULT_PRICING_MODEL,
        PAPER_COST_RATIOS as COST_PAPER_COST_RATIOS,
        PAPER_PERFORMANCE_GAINS,
        PAPER_TABLE4,
        PHASE_INFERENCE,
        PHASE_TOTAL,
        PHASE_TRAINING,
        PHASES,
        PRICING,
        USD_PER_1K_INPUT_TOKENS,
        USD_PER_1K_OUTPUT_TOKENS,
        CostConfig,
        CostLedger,
        ExperimentCost,
        PhaseView,
        TokenLedger,
        TokenUsage,
        average,
        compare_cost_to_paper,
        cost_per_1k_questions,
        cost_per_question,
        cost_ratio,
        cost_ratios_from_table,
        estimate_azure_sft_cost,
        estimate_candidate_sampling_cost,
        estimate_inference_cost_per_1k,
        estimate_training_cost,
        format_cost_table,
        inference_cost_per_1k,
        model_pricing,
        paper_table4_row,
        performance_gain_from_table,
        price_tokens,
        ratio,
        reference_table as cost_reference_table,
        resolve_pricing,
        save_cost_report,
        speedup_ratio,
        summarize_costs,
        times_less,
        tokens_to_usd,
        training_cost_per_1k,
    )

    _extend(
        [
            "DEFAULT_PER_QUESTIONS",
            "DEFAULT_PRICING_MODEL",
            "COST_PAPER_COST_RATIOS",
            "PAPER_PERFORMANCE_GAINS",
            "PAPER_TABLE4",
            "PHASE_INFERENCE",
            "PHASE_TOTAL",
            "PHASE_TRAINING",
            "PHASES",
            "PRICING",
            "USD_PER_1K_INPUT_TOKENS",
            "USD_PER_1K_OUTPUT_TOKENS",
            "CostConfig",
            "CostLedger",
            "ExperimentCost",
            "PhaseView",
            "TokenLedger",
            "TokenUsage",
            "average",
            "compare_cost_to_paper",
            "cost_per_1k_questions",
            "cost_per_question",
            "cost_ratio",
            "cost_ratios_from_table",
            "estimate_azure_sft_cost",
            "estimate_candidate_sampling_cost",
            "estimate_inference_cost_per_1k",
            "estimate_training_cost",
            "format_cost_table",
            "inference_cost_per_1k",
            "model_pricing",
            "paper_table4_row",
            "performance_gain_from_table",
            "price_tokens",
            "ratio",
            "cost_reference_table",
            "resolve_pricing",
            "save_cost_report",
            "speedup_ratio",
            "summarize_costs",
            "times_less",
            "tokens_to_usd",
            "training_cost_per_1k",
        ]
    )
    _COST_AVAILABLE = True
except Exception:  # pragma: no cover - defensive import guard
    _COST_AVAILABLE = False


# ---------------------------------------------------------------------------
# vram.py -- peak GPU memory for the 0.1B adapter (Table 6)
# ---------------------------------------------------------------------------
try:  # pragma: no cover
    from .vram import (  # noqa: F401
        ADAPTER_SIZES,
        MIXTRAL_HALF_PRECISION_GIB,
        MIXTRAL_PARAMS_B,
        PAPER_ACCURACY_TABLE6,
        PAPER_STRATEGYQA_BASE_ACC,
        PAPER_TABLE6,
        PAPER_VRAM_TABLE6,
        REPORT_ONLY_0_1B,
        VRAM_METHOD_NAMES,
        VramConfig,
        VramMeasurement,
        VramMeter,
        VramReport,
        VramReporter,
        VramTracker,
        compare_to_paper as vram_compare_to_paper,
        cuda_available,
        current_memory_gib,
        describe_device,
        device_memory_summary,
        estimate_bbox_adapter_vram_gib,
        estimate_lora_vram_gib,
        estimate_model_vram_gib,
        estimate_vram_for_method,
        format_vram_table,
        measure_phase,
        measure_vram,
        paper_table6_row,
        peak_memory_bytes,
        peak_memory_gib,
        reference_table as vram_reference_table,
        reset_peak_memory,
        save_vram_report,
        set_vram_logger,
        summarize,
        torch_available,
        total_memory_gib,
    )

    _extend(
        [
            "ADAPTER_SIZES",
            "MIXTRAL_HALF_PRECISION_GIB",
            "MIXTRAL_PARAMS_B",
            "PAPER_ACCURACY_TABLE6",
            "PAPER_STRATEGYQA_BASE_ACC",
            "PAPER_TABLE6",
            "PAPER_VRAM_TABLE6",
            "REPORT_ONLY_0_1B",
            "VRAM_METHOD_NAMES",
            "VramConfig",
            "VramMeasurement",
            "VramMeter",
            "VramReport",
            "VramReporter",
            "VramTracker",
            "vram_compare_to_paper",
            "cuda_available",
            "current_memory_gib",
            "describe_device",
            "device_memory_summary",
            "estimate_bbox_adapter_vram_gib",
            "estimate_lora_vram_gib",
            "estimate_model_vram_gib",
            "estimate_vram_for_method",
            "format_vram_table",
            "measure_phase",
            "measure_vram",
            "paper_table6_row",
            "peak_memory_bytes",
            "peak_memory_gib",
            "vram_reference_table",
            "reset_peak_memory",
            "save_vram_report",
            "set_vram_logger",
            "summarize",
            "torch_available",
            "total_memory_gib",
        ]
    )
    _VRAM_AVAILABLE = True
except Exception:  # pragma: no cover - defensive import guard
    _VRAM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Convenience factories / metadata
# ---------------------------------------------------------------------------


def evaluate_dataset(
    generations: List[str],
    *,
    dataset: str,
    golds: Optional[List[Any]] = None,
    examples: Optional[List[Any]] = None,
    answer_type: Optional[str] = None,
    choices_list: Optional[List[Any]] = None,
    base_value: Optional[float] = None,
    metric: Optional[str] = None,
    config: Optional[Any] = None,
    judge: Optional[Any] = None,
    toxicity_judge: Optional[Any] = None,
    return_details: bool = False,
    **kwargs: Any,
) -> Any:
    """Evaluate generations for a named dataset via :func:`eval.metrics.evaluate_generations`.

    This is a thin, dependency-light wrapper so script entry points can do
    ``from bbox_adapter.eval import evaluate_dataset`` without caring about the
    concrete metric dispatch (Accuracy / True+Info / Toxicity).
    """
    if not _METRICS_AVAILABLE:  # pragma: no cover
        raise ImportError(
            "bbox_adapter.eval.metrics is unavailable in this environment; "
            "install torch/transformers or import the metrics module directly."
        )
    from . import metrics as _metrics  # noqa: WPS433 (local import by design)

    return _metrics.evaluate_generations(
        generations,
        dataset=dataset,
        golds=golds,
        examples=examples,
        answer_type=answer_type,
        choices_list=choices_list,
        base_value=base_value,
        metric=metric,
        config=config,
        judge=judge,
        toxicity_judge=toxicity_judge,
        return_details=return_details,
        **kwargs,
    )


def results_table(rows: List[Dict[str, Any]], *, title: str = "") -> str:
    """Render a paper-style results table (delegates to :mod:`eval.metrics`)."""
    if not _METRICS_AVAILABLE:  # pragma: no cover
        return "\n".join(str(r) for r in rows)
    from .metrics import format_results_table

    return format_results_table(rows, title=title)


def cost_table(rows: List[Dict[str, Any]], *, title: str = "") -> str:
    """Render the cost table (Table 4) delegating to :mod:`eval.cost`."""
    if not _COST_AVAILABLE:  # pragma: no cover
        return "\n".join(str(r) for r in rows)
    from .cost import format_cost_table

    return format_cost_table(rows, title=title)


def vram_table(rows: List[Any], *, title: str = "", digits: int = 0) -> str:
    """Render the VRAM table (Table 6) delegating to :mod:`eval.vram`."""
    if not _VRAM_AVAILABLE:  # pragma: no cover
        return "\n".join(str(r) for r in rows)
    from .vram import format_vram_table

    return format_vram_table(rows, title=title, digits=digits)


def make_cost_ledger(config: Optional[Any] = None, **kwargs: Any) -> Any:
    """Construct a :class:`eval.cost.CostLedger` from a config mapping/dataclass."""
    if not _COST_AVAILABLE:  # pragma: no cover
        raise ImportError("bbox_adapter.eval.cost is unavailable in this environment.")
    from .cost import CostLedger

    if config is not None:
        try:
            return CostLedger.from_config(config, **kwargs)
        except Exception:
            return CostLedger(**kwargs)
    return CostLedger(**kwargs)


def describe() -> Dict[str, Any]:
    """Return metadata describing the evaluation package and its availability."""
    return {
        "module": "bbox_adapter.eval",
        "paper": "Lightweight Adapting for Black-Box Large Language Models",
        "sections": {
            "metrics": "4.2-4.7, Appendix E (Accuracy / True+Info / Toxic% / Toxicity Prob% / Delta%)",
            "cost": "4.4, Table 4 (training & inference $ per 1k questions)",
            "vram": "Table 6 + Addendum (0.1B adapter only)",
        },
        "available": {
            "metrics": _METRICS_AVAILABLE,
            "cost": _COST_AVAILABLE,
            "vram": _VRAM_AVAILABLE,
        },
    }


def _self_test() -> Dict[str, Any]:
    """Dependency-free smoke test of the package surface."""
    result: Dict[str, Any] = {"metrics": _METRICS_AVAILABLE, "cost": _COST_AVAILABLE, "vram": _VRAM_AVAILABLE}
    assert isinstance(describe(), dict)
    if _METRICS_AVAILABLE:
        assert callable(evaluate_dataset)  # type: ignore[name-defined]
        assert PAPER_TABLE2  # type: ignore[name-defined]
    if _COST_AVAILABLE:
        assert PRICING  # type: ignore[name-defined]
        assert callable(cost_table)
    if _VRAM_AVAILABLE:
        assert PAPER_TABLE6  # type: ignore[name-defined]
        assert callable(vram_table)
    result["ok"] = True
    return result


if __name__ == "__main__":  # pragma: no cover
    import json

    print(json.dumps({"describe": describe(), "self_test": _self_test()}, indent=2, default=str))
