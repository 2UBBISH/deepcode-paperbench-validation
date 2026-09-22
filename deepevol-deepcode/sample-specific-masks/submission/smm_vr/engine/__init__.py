"""Training / evaluation engine for Sample-specific Multi-channel Masks (SMM).

This package aggregates the pieces required to reproduce the paper:

* :mod:`smm_vr.engine.seeds` -- the three-seed ``{0, 1, 2}`` reproducibility protocol.
* :mod:`smm_vr.engine.metrics` -- top-1/top-k accuracy, mean +- std aggregation and
  paper-style table rendering.
* :mod:`smm_vr.engine.train_smm` -- Algorithm 1 (joint training of the shared pattern
  ``delta`` and the lightweight mask generator ``phi`` through the frozen classifier).
* :mod:`smm_vr.engine.evaluate` -- the SMM evaluation protocol (top-1 test accuracy,
  label-mapping aware decoding, feature extraction for t-SNE).

The re-exports below are intentionally guarded so that ``import smm_vr.engine`` never
fails while the package is being built incrementally.
"""

from __future__ import annotations

__all__ = []


def _extend(names):
    for name in names:
        if name not in __all__:
            __all__.append(name)


# --------------------------------------------------------------------------------------
# seeds -- reproducibility (three seeds, deterministic cuDNN, loader worker seeding)
# --------------------------------------------------------------------------------------
try:  # pragma: no cover - import guard
    from .seeds import (  # noqa: F401
        DEFAULT_CUDNN_BENCHMARK,
        DEFAULT_DETERMINISTIC,
        DEFAULT_SEEDS,
        N_SEEDS,
        SEEDS,
        capture_rng_state,
        dataloader_seed_kwargs,
        enumerate_seeds,
        get_seed,
        make_generator,
        make_worker_init_fn,
        resolve_seeds,
        restore_rng_state,
        seed_all,
        seed_everything,
        seed_worker,
        set_seed,
        temporary_seed,
        worker_kwargs,
    )

    _extend(
        [
            "SEEDS",
            "DEFAULT_SEEDS",
            "N_SEEDS",
            "DEFAULT_DETERMINISTIC",
            "DEFAULT_CUDNN_BENCHMARK",
            "get_seed",
            "set_seed",
            "seed_everything",
            "seed_all",
            "make_generator",
            "seed_worker",
            "make_worker_init_fn",
            "worker_kwargs",
            "dataloader_seed_kwargs",
            "capture_rng_state",
            "restore_rng_state",
            "temporary_seed",
            "resolve_seeds",
            "enumerate_seeds",
        ]
    )
except ImportError:  # pragma: no cover
    pass


# --------------------------------------------------------------------------------------
# metrics -- accuracy bookkeeping and reporting
# --------------------------------------------------------------------------------------
try:  # pragma: no cover - import guard
    from .metrics import (  # noqa: F401
        REFERENCE_TABLES,
        TABLE1_RESNET18,
        TABLE1_RESNET50,
        TABLE2_VIT_B32,
        TABLE3_ABLATIONS,
        AccuracyMeter,
        MetricTracker,
        RunResult,
        accuracy,
        aggregate_results,
        aggregate_seeds,
        compare_run_to_reference,
        compare_with_reference,
        dump_results_json,
        format_mean_std,
        format_results_table,
        mean_over_datasets,
        resolve_reference_table,
        top1_accuracy,
        topk_accuracy,
    )

    _extend(
        [
            "AccuracyMeter",
            "MetricTracker",
            "RunResult",
            "accuracy",
            "top1_accuracy",
            "topk_accuracy",
            "aggregate_seeds",
            "aggregate_results",
            "format_mean_std",
            "mean_over_datasets",
            "format_results_table",
            "compare_with_reference",
            "compare_run_to_reference",
            "resolve_reference_table",
            "dump_results_json",
            "TABLE1_RESNET18",
            "TABLE1_RESNET50",
            "TABLE2_VIT_B32",
            "TABLE3_ABLATIONS",
            "REFERENCE_TABLES",
        ]
    )
except ImportError:  # pragma: no cover
    pass


# --------------------------------------------------------------------------------------
# train_smm -- Algorithm 1
# --------------------------------------------------------------------------------------
try:  # pragma: no cover - import guard
    from .train_smm import (  # noqa: F401
        DEFAULT_ALPHA_DELTA,
        DEFAULT_ALPHA_MASK_5,
        DEFAULT_ALPHA_MASK_6,
        DEFAULT_BATCH_SIZE,
        DEFAULT_EPOCHS,
        DEFAULT_GAMMA_DELTA,
        DEFAULT_GAMMA_MASK_5,
        DEFAULT_GAMMA_MASK_6,
        DEFAULT_MILESTONES,
        MASK_LAYERS_BY_BACKBONE,
        SMALL_BATCH_DATASETS,
        EpochStats,
        SMMTrainConfig,
        TrainingHistory,
        build_label_mapping,
        build_optimizers,
        build_scheduler,
        evaluate_epoch,
        train_accuracy,
        train_one_seed,
        train_smm,
        train_with_seeds,
    )

    _extend(
        [
            "DEFAULT_EPOCHS",
            "DEFAULT_MILESTONES",
            "DEFAULT_ALPHA_DELTA",
            "DEFAULT_GAMMA_DELTA",
            "DEFAULT_ALPHA_MASK_5",
            "DEFAULT_GAMMA_MASK_5",
            "DEFAULT_ALPHA_MASK_6",
            "DEFAULT_GAMMA_MASK_6",
            "DEFAULT_BATCH_SIZE",
            "SMALL_BATCH_DATASETS",
            "MASK_LAYERS_BY_BACKBONE",
            "SMMTrainConfig",
            "EpochStats",
            "TrainingHistory",
            "build_optimizers",
            "build_scheduler",
            "evaluate_epoch",
            "train_accuracy",
            "train_smm",
            "train_one_seed",
            "train_with_seeds",
            "build_label_mapping",
        ]
    )
except ImportError:  # pragma: no cover
    pass


# --------------------------------------------------------------------------------------
# evaluate -- the SMM evaluation protocol
# --------------------------------------------------------------------------------------
try:  # pragma: no cover - import guard
    from .evaluate import (  # noqa: F401
        EVAL_BATCH_SIZE,
        TSNE_SAMPLES_PER_DATASET,
        EvaluationConfig,
        EvaluationResult,
        collect_logits,
        decode_predictions,
        evaluate,
        evaluate_many_datasets,
        evaluate_method,
        evaluate_seeds,
        extract_features,
        extract_features_for_tsne,
        save_results,
        summarize_results,
    )

    _extend(
        [
            "EVAL_BATCH_SIZE",
            "TSNE_SAMPLES_PER_DATASET",
            "EvaluationConfig",
            "EvaluationResult",
            "decode_predictions",
            "collect_logits",
            "evaluate",
            "extract_features",
            "extract_features_for_tsne",
            "evaluate_seeds",
            "evaluate_method",
            "summarize_results",
            "evaluate_many_datasets",
            "save_results",
        ]
    )
except ImportError:  # pragma: no cover
    pass
