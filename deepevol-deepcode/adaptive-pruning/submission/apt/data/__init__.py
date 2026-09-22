"""Data pipelines for APT.

This package aggregates the three data pipelines used in the reproduction:

* :mod:`apt.data.glue` -- GLUE big/small tasks (MNLI, SST2, QNLI, QQP, MRPC,
  CoLA, RTE, STSB) with Table 6 hyper-parameters and metric helpers.
* :mod:`apt.data.squad` -- SQuAD v2.0 with sliding-window features, span
  post-processing and official-style EM/F1 metrics.
* :mod:`apt.data.cnndm` -- CNN/DailyMail summarization for T5 with ROUGE.

Imports are deliberately defensive: each submodule is optional so that a
partial install (e.g. no ``datasets``/``rouge_score``) can still import the
package. Use :data:`AVAILABLE` to discover which pipelines loaded.
"""

from __future__ import annotations

from typing import Dict, List

__all__: List[str] = []
AVAILABLE: Dict[str, bool] = {}

# ---------------------------------------------------------------------------
# GLUE
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import guard
    from .glue import (  # noqa: F401
        GLUE_BIG_TASKS,
        GLUE_SMALL_TASKS,
        GLUE_TASKS,
        TABLE6,
        GlueCollator,
        GlueDataset,
        GlueTaskSpec,
        HParams,
        T5GlueDataset,
        build_glue_dataset,
        build_glue_datasets,
        canonical_task_name,
        collator_for_task,
        compute_glue_metrics,
        get_hparams,
        get_task_spec,
        is_big_task,
        make_glue_dataloaders,
        pareto_big_small_order,
    )

    AVAILABLE["glue"] = True
    __all__ += [
        "GLUE_BIG_TASKS",
        "GLUE_SMALL_TASKS",
        "GLUE_TASKS",
        "TABLE6",
        "GlueCollator",
        "GlueDataset",
        "GlueTaskSpec",
        "HParams",
        "T5GlueDataset",
        "build_glue_dataset",
        "build_glue_datasets",
        "canonical_task_name",
        "collator_for_task",
        "compute_glue_metrics",
        "get_hparams",
        "get_task_spec",
        "is_big_task",
        "make_glue_dataloaders",
        "pareto_big_small_order",
    ]
except Exception:  # pragma: no cover
    AVAILABLE["glue"] = False

# ---------------------------------------------------------------------------
# SQuAD v2.0
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import guard
    from .squad import (  # noqa: F401
        SquadCollator,
        SquadDataset,
        SquadExample,
        SquadFeatures,
        build_squad_features,
        compute_squad_metrics,
        convert_examples_to_features,
        get_squad_hparams,
        load_squad_examples,
        make_squad_dataloaders,
        squad_metric_summary,
        span_logits_to_answers,
        write_predictions,
    )

    AVAILABLE["squad"] = True
    __all__ += [
        "SquadCollator",
        "SquadDataset",
        "SquadExample",
        "SquadFeatures",
        "build_squad_features",
        "compute_squad_metrics",
        "convert_examples_to_features",
        "get_squad_hparams",
        "load_squad_examples",
        "make_squad_dataloaders",
        "squad_metric_summary",
        "span_logits_to_answers",
        "write_predictions",
    ]
except Exception:  # pragma: no cover
    AVAILABLE["squad"] = False

# ---------------------------------------------------------------------------
# CNN/DailyMail
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import guard
    from .cnndm import (  # noqa: F401
        CNN_DM_TABLE6,
        CNNDM_SPEC,
        CnndmCollator,
        CnndmDataset,
        CnndmSpec,
        build_cnndm_dataset,
        compute_cnndm_metrics,
        compute_rouge,
        decode_predictions,
        evaluate_cnndm,
        generate_summaries,
        get_cnndm_hparams,
        make_cnndm_dataloaders,
        rouge_metric_summary,
    )

    AVAILABLE["cnndm"] = True
    __all__ += [
        "CNN_DM_TABLE6",
        "CNNDM_SPEC",
        "CnndmCollator",
        "CnndmDataset",
        "CnndmSpec",
        "build_cnndm_dataset",
        "compute_cnndm_metrics",
        "compute_rouge",
        "decode_predictions",
        "evaluate_cnndm",
        "generate_summaries",
        "get_cnndm_hparams",
        "make_cnndm_dataloaders",
        "rouge_metric_summary",
    ]
except Exception:  # pragma: no cover
    AVAILABLE["cnndm"] = False


# ---------------------------------------------------------------------------
# Convenience unified loader
# ---------------------------------------------------------------------------
def make_dataloaders(
    task: str,
    tokenizer,
    model_type: str = "encoder",
    **kwargs,
):
    """Dispatch to the correct pipeline based on the task name.

    ``task`` may be a GLUE task (``"sst2"``, ``"mnli"``, ...), ``"squad"`` /
    ``"squad_v2"``, or ``"cnndm"`` / ``"cnn_dailymail"``.
    """
    name = str(task).lower().replace("-", "").replace("_", "")
    if name in {"squad", "squad2", "squadv2"}:
        return make_squad_dataloaders(tokenizer, **kwargs)
    if name in {"cnndm", "cnndailymail", "cnn", "dailymail"}:
        return make_cnndm_dataloaders(tokenizer, **kwargs)
    return make_glue_dataloaders(task, tokenizer, model_type=model_type, **kwargs)


def compute_metrics(task: str, predictions, references, **kwargs):
    """Dispatch metric computation for a canonical task name."""
    name = str(task).lower().replace("-", "").replace("_", "")
    if name in {"squad", "squad2", "squadv2"}:
        return compute_squad_metrics(predictions, references, **kwargs)
    if name in {"cnndm", "cnndailymail", "cnn", "dailymail"}:
        return compute_cnndm_metrics(predictions, references, **kwargs)
    return compute_glue_metrics(task, predictions, references)


__all__ += ["available", "make_dataloaders", "compute_metrics", "AVAILABLE"]


def available() -> Dict[str, bool]:
    """Return which data pipelines imported successfully."""
    return dict(AVAILABLE)
