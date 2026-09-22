"""Evaluation subpackage for DPMs-ANT.

Aggregates the public evaluation API:

* :mod:`dpm_ant.evaluation.intra_lpips` -- Intra-LPIPS diversity metric (§5.2)
* :mod:`dpm_ant.evaluation.fid`         -- FID against 2.5k/2.7k target sets (§5.2)
* :mod:`dpm_ant.evaluation.metrics`     -- efficiency metrics (Table 1/8, §5.3)
* :mod:`dpm_ant.evaluation.evaluate`    -- end-to-end evaluation pipeline

The symbols are resolved lazily (PEP 562) so that ``import dpm_ant.evaluation``
stays cheap and does not require the optional ``lpips`` / ``clean-fid`` /
``torchvision`` dependencies unless a metric is actually used.
"""
from __future__ import annotations

import importlib
from typing import Any, Dict, List

__all__: List[str] = [
    # --- intra_lpips ---
    "LPIPSMetric",
    "IntraLPIPS",
    "IntraLPIPSConfig",
    "compute_intra_lpips",
    "pairwise_distances",
    "assign_to_nearest",
    "cluster_pairwise_mean",
    "build_lpips_metric",
    "load_reference_images",
    # --- fid ---
    "FIDConfig",
    "FIDResult",
    "FIDMetric",
    "compute_fid",
    "compute_fid_from_dirs",
    "build_fid_metric",
    "frechet_distance",
    "activation_statistics",
    "load_fid_reference",
    "TARGET_FID_SIZES",
    "PAPER_FID_REFERENCE",
    # --- metrics ---
    "EfficiencyConfig",
    "EfficiencyMetrics",
    "efficiency_report",
    "adaptor_parameter_rate",
    "parameter_rate",
    "count_parameters",
    "measure_gpu_memory",
    "measure_time",
    "benchmark_iterations",
    "GPUMemoryTracker",
    "Timer",
    "compare_to_paper",
    "format_report",
    "save_report",
    "PAPER_PARAM_RATE",
    "PAPER_REFERENCE_TABLES",
    # --- evaluate ---
    "EvalConfig",
    "EvalResult",
    "Evaluator",
    "evaluate_model",
    "evaluate_from_dirs",
    "run_evaluation",
    "aggregate_seeds",
    "load_report",
    "load_images_from_dir",
    "resolve_reference_dir",
    "PAPER_INTRA_LPIPS",
    "PAPER_FID",
    "PAPER_ABLATION_FID",
    "PAPER_SENSITIVITY",
    "PAPER_TIME_HOURS",
    "PAPER_CLASSIFIER_ABLATION",
]

_EXPORTS: Dict[str, str] = {
    # intra_lpips
    "LPIPSMetric": "intra_lpips",
    "IntraLPIPS": "intra_lpips",
    "IntraLPIPSConfig": "intra_lpips",
    "compute_intra_lpips": "intra_lpips",
    "pairwise_distances": "intra_lpips",
    "assign_to_nearest": "intra_lpips",
    "cluster_pairwise_mean": "intra_lpips",
    "build_lpips_metric": "intra_lpips",
    "load_reference_images": "intra_lpips",
    # fid
    "FIDConfig": "fid",
    "FIDResult": "fid",
    "FIDMetric": "fid",
    "compute_fid": "fid",
    "compute_fid_from_dirs": "fid",
    "build_fid_metric": "fid",
    "frechet_distance": "fid",
    "activation_statistics": "fid",
    "load_fid_reference": "fid",
    "TARGET_FID_SIZES": "fid",
    "PAPER_FID_REFERENCE": "fid",
    # metrics
    "EfficiencyConfig": "metrics",
    "EfficiencyMetrics": "metrics",
    "efficiency_report": "metrics",
    "adaptor_parameter_rate": "metrics",
    "parameter_rate": "metrics",
    "count_parameters": "metrics",
    "measure_gpu_memory": "metrics",
    "measure_time": "metrics",
    "benchmark_iterations": "metrics",
    "GPUMemoryTracker": "metrics",
    "Timer": "metrics",
    "compare_to_paper": "metrics",
    "format_report": "metrics",
    "save_report": "metrics",
    "PAPER_PARAM_RATE": "metrics",
    "PAPER_REFERENCE_TABLES": "metrics",
    # evaluate
    "EvalConfig": "evaluate",
    "EvalResult": "evaluate",
    "Evaluator": "evaluate",
    "evaluate_model": "evaluate",
    "evaluate_from_dirs": "evaluate",
    "run_evaluation": "evaluate",
    "aggregate_seeds": "evaluate",
    "load_report": "evaluate",
    "load_images_from_dir": "evaluate",
    "resolve_reference_dir": "evaluate",
    "PAPER_INTRA_LPIPS": "evaluate",
    "PAPER_FID": "evaluate",
    "PAPER_ABLATION_FID": "evaluate",
    "PAPER_SENSITIVITY": "evaluate",
    "PAPER_TIME_HOURS": "evaluate",
    "PAPER_CLASSIFIER_ABLATION": "evaluate",
}


def __getattr__(name: str) -> Any:  # pragma: no cover - simple dispatch
    """Lazily resolve the public evaluation API."""
    if name in _EXPORTS:
        module = importlib.import_module(f".{_EXPORTS[name]}", __name__)
        try:
            value = getattr(module, name)
        except AttributeError as exc:  # pragma: no cover
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r} "
                f"(submodule {_EXPORTS[name]!r} does not define it)"
            ) from exc
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:  # pragma: no cover
    return sorted(set(globals()) | set(__all__))
