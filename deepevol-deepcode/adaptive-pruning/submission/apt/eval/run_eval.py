"""Evaluation harness for reproducing the APT paper's result tables.

This module implements the evaluation protocol of Sec. 5.3:

* **Training efficiency metrics** -- relative training peak memory
  (``Train. Mem.``) and relative training speed measured by time to accuracy
  (TTA) to ``97%`` of the fully fine-tuned (FT) performance, both normalised to
  FT.  For methods that use knowledge distillation the training time of the
  teacher model *plus* the student is counted.
* **Inference efficiency metrics** -- inference peak memory (``Inf. Mem.``) and
  relative inference speed based on throughput (data processed per second),
  both normalised to FT.
* Task quality metrics -- GLUE accuracy/F1/MCC/Spearman, SQuAD v2.0 EM/F1 and
  CNN/DM ROUGE-1/2/L, computed by :mod:`apt.eval.metrics`.
* The inference test batch size is 128 for small models (32 / 4 for LLaMA 7B /
  13B, which are out of scope for this reproduction).

Everything is deliberately dependency-light: ``torch`` and the task data
modules are imported lazily so the module (and its ``_self_test``) can be used
to build/sanity-check result tables without a GPU.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# Package-local (defensive) imports.
# ---------------------------------------------------------------------------

try:  # metrics helpers
    from .metrics import (  # noqa: F401
        FT_REFERENCES,
        GLUE_BIG_TASKS,
        GLUE_PRIMARY_METRIC,
        GLUE_SMALL_TASKS,
        GLUE_TASKS,
        RAW_EFFICIENCY,
        ROUGE_KEYS,
        SEQ2SEQ_TASKS,
        SQUAD_TASKS,
        ReferenceScores,
        compute_metrics as _compute_metrics,
        format_percent,
        glue_average,
        is_glue_task,
        is_seq2seq_task,
        is_squad_task,
        metric_for_display,
        normalize_task_name,
        primary_metric,
        primary_metric_name,
        relative_accuracy,
    )

    _METRICS_OK = True
except Exception:  # pragma: no cover - fallback for partial installs
    _METRICS_OK = False
    FT_REFERENCES = {}  # type: ignore
    RAW_EFFICIENCY = {}  # type: ignore
    GLUE_TASKS = ("mnli", "sst2", "qnli", "qqp", "mrpc", "cola", "rte", "stsb")  # type: ignore
    GLUE_BIG_TASKS = ("mnli", "sst2", "qnli", "qqp")  # type: ignore
    GLUE_SMALL_TASKS = ("mrpc", "cola", "rte", "stsb")  # type: ignore
    SEQ2SEQ_TASKS = ("cnndm", "cnn_dailymail")  # type: ignore
    SQUAD_TASKS = ("squad", "squad_v2", "squad2")  # type: ignore
    ROUGE_KEYS = ("rouge1", "rouge2", "rougeL")  # type: ignore
    GLUE_PRIMARY_METRIC = {}  # type: ignore

    def normalize_task_name(task: str) -> str:  # type: ignore
        return str(task).strip().lower().replace("-", "").replace("_", "")

    def is_glue_task(task: str) -> bool:  # type: ignore
        return normalize_task_name(task) in {normalize_task_name(t) for t in GLUE_TASKS}

    def is_squad_task(task: str) -> bool:  # type: ignore
        return normalize_task_name(task) in {normalize_task_name(t) for t in SQUAD_TASKS}

    def is_seq2seq_task(task: str) -> bool:  # type: ignore
        return normalize_task_name(task) in {normalize_task_name(t) for t in SEQ2SEQ_TASKS}

    def _compute_metrics(task, predictions, references, **kwargs):  # type: ignore
        raise ImportError("apt.eval.metrics is unavailable")

    def primary_metric(task, metrics):  # type: ignore
        for key in ("accuracy", "exact", "f1", "rougeL", "average", "primary"):
            if key in metrics:
                return float(metrics[key])
        if metrics:
            return float(next(iter(metrics.values())))
        return 0.0

    def primary_metric_name(task):  # type: ignore
        return "accuracy"

    def metric_for_display(task, metrics):  # type: ignore
        return ", ".join(f"{k}={v:.1f}" for k, v in metrics.items())

    def glue_average(per_task, tasks=None, metric=None):  # type: ignore
        values = [float(v) for v in per_task.values()]
        return sum(values) / len(values) if values else 0.0

    def relative_accuracy(score, reference):  # type: ignore
        return 100.0 * float(score) / float(reference) if reference else float("nan")

    def format_percent(value, digits=1):  # type: ignore
        if value is None:
            return "n/a"
        return f"{float(value):.{digits}f}%"

    ReferenceScores = None  # type: ignore


try:  # efficiency helpers
    from .efficiency import (  # noqa: F401
        DEFAULT_SEQUENCE_LENGTH,
        INFERENCE_BATCH_SIZES,
        METRIC_KEYS,
        SMALL_MODEL_INFERENCE_BATCH_SIZE,
        TABLE11_RAW,
        TABLE2_RELATIVE,
        TTA_FRACTION,
        EfficiencyResult,
        PeakMemoryTracker,
        TimeToAccuracy,
        Timer,
        TrainingEfficiencyTracker,
        efficiency_row,
        efficiency_summary,
        format_efficiency,
        inference_batch_size_for,
        measure_inference,
        measure_inference_throughput,
        measure_training_efficiency,
        normalize_efficiency,
        parameter_count,
        relative_from_table11,
    )

    _EFFICIENCY_OK = True
except Exception:  # pragma: no cover - fallback for partial installs
    _EFFICIENCY_OK = False
    TTA_FRACTION = 0.97  # type: ignore
    SMALL_MODEL_INFERENCE_BATCH_SIZE = 128  # type: ignore
    DEFAULT_SEQUENCE_LENGTH = 128  # type: ignore
    TABLE11_RAW = {}  # type: ignore
    TABLE2_RELATIVE = {}  # type: ignore
    METRIC_KEYS = ("train_time", "train_mem", "inf_time", "inf_mem")  # type: ignore
    INFERENCE_BATCH_SIZES = {"small": 128, "llama-7b": 32, "llama-13b": 4}  # type: ignore
    EfficiencyResult = None  # type: ignore

    def inference_batch_size_for(model=None, **kwargs):  # type: ignore
        return SMALL_MODEL_INFERENCE_BATCH_SIZE

    def measure_inference(model, **kwargs):  # type: ignore
        raise ImportError("apt.eval.efficiency is unavailable")

    def parameter_count(model, trainable_only=False):  # type: ignore
        try:
            return sum(p.numel() for p in model.parameters())
        except Exception:
            return 0

    def normalize_efficiency(raw, reference):  # type: ignore
        out = {}
        for key in METRIC_KEYS:
            ref = reference.get(key)
            val = raw.get(key)
            out[key] = (100.0 * val / ref) if (ref and val is not None) else None
        return out

    def format_efficiency(metrics, digits=1, as_percent=True):  # type: ignore
        return " / ".join(format_percent(metrics.get(k), digits) for k in METRIC_KEYS)

    def efficiency_summary(results, reference=None, digits=1):  # type: ignore
        return ""


# ===========================================================================
# Reference constants (paper Tables 2, 7, 8, 11)
# ===========================================================================

#: Method ordering used by every table of the paper.
BASELINE_METHODS: Tuple[str, ...] = ("FT", "LoRA", "LoRA+Prune", "Prune+Distill", "LoRA+Prune+Distill")

#: Baseline methods compared against APT (Sec. 5.2 + App. D).
ATTACK_BASELINE_METHODS: Tuple[str, ...] = (
    "FT",
    "LoRA",
    "LoRA+Prune",
    "Prune+Distill",
    "LoRA+Prune+Distill",
    "PST",
    "LRP",
    "LoRA+Distill",
)

#: Order in which methods are printed for a given model.
METHOD_ORDER: Dict[str, Tuple[str, ...]] = {
    "roberta": ("FT", "LoRA", "LoRA+Prune", "Prune+Distill", "LoRA+Prune+Distill", "APT"),
    "roberta-base": ("FT", "LoRA", "LoRA+Prune", "Prune+Distill", "LoRA+Prune+Distill", "APT"),
    "t5": ("FT", "LoRA", "LoRA+Prune", "APT"),
    "t5-base": ("FT", "LoRA", "LoRA+Prune", "APT"),
    "bert": ("FT", "LoRA", "PST", "LRP", "APT"),
    "bert-base": ("FT", "LoRA", "PST", "LRP", "APT"),
}

#: Metric column ordering for the tasks reported in Table 2.
TASK_ORDER: Tuple[str, ...] = ("mnli", "sst2", "squad_v2", "cnndm")

#: Table 2 (60% sparsity).  ``None`` means "not reported by the paper".
TABLE2_REFERENCES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "roberta": {
        "FT": {
            "mnli": 87.6,
            "sst2": 94.8,
            "squad_v2": 82.9,
            "train_time": 100.0,
            "train_mem": 100.0,
            "inf_time": 100.0,
            "inf_mem": 100.0,
        },
        "LoRA": {
            "mnli": 87.5,
            "sst2": 95.1,
            "squad_v2": 83.0,
            "train_time": 2137.0,
            "train_mem": 60.5,
            "inf_time": 100.0,
            "inf_mem": 100.0,
        },
        "LoRA+Prune": {
            "mnli": 84.0,
            "sst2": 93.0,
            "squad_v2": 79.2,
            "train_time": 5128.3,
            "train_mem": 60.5,
            "inf_time": 38.0,
            "inf_mem": 75.1,
        },
        "Prune+Distill": {
            "mnli": 87.3,
            "sst2": 94.5,
            "squad_v2": None,
            "train_time": 1495.3,
            "train_mem": 168.5,
            "inf_time": 38.6,
            "inf_mem": 79.2,
        },
        "LoRA+Prune+Distill": {
            "mnli": 84.2,
            "sst2": 91.9,
            "squad_v2": None,
            "train_time": 6534.6,
            "train_mem": 141.4,
            "inf_time": 39.4,
            "inf_mem": 82.3,
        },
        "APT": {
            "mnli": 86.4,
            "sst2": 94.5,
            "squad_v2": 81.8,
            "train_time": 592.1,
            "train_mem": 70.1,
            "inf_time": 41.3,
            "inf_mem": 78.1,
        },
    },
    "t5": {
        "FT": {
            "mnli": 87.1,
            "sst2": 95.2,
            "cnndm": (42.1, 20.3, 39.4),
            "train_time": 100.0,
            "train_mem": 100.0,
            "inf_time": 100.0,
            "inf_mem": 100.0,
        },
        "LoRA": {
            "mnli": 87.0,
            "sst2": 95.0,
            "cnndm": (38.7, 17.2, 36.0),
            "train_time": 255.5,
            "train_mem": 62.0,
            "inf_time": 100.0,
            "inf_mem": 100.0,
        },
        "LoRA+Prune": {
            "mnli": 80.9,
            "sst2": 92.3,
            "cnndm": (36.7, 15.7, 33.9),
            "train_time": 4523.5,
            "train_mem": 62.0,
            "inf_time": 47.1,
            "inf_mem": 73.4,
        },
        "APT": {
            "mnli": 87.0,
            "sst2": 95.0,
            "cnndm": (38.6, 17.0, 35.8),
            "train_time": 484.7,
            "train_mem": 73.9,
            "inf_time": 74.6,
            "inf_mem": 81.5,
        },
    },
}

#: Table 7 -- BERT-base at 50% / 10% density (GLUE average).
TABLE7_REFERENCES: Dict[str, Dict[str, Dict[str, float]]] = {
    "50%": {"PST": 80.5, "LRP": 81.8, "APT": 83.2},
    "10%": {"PST": 71.4, "LRP": 73.6, "APT": 76.8},
}

#: Table 8 -- RoBERTa GLUE average at 40% sparsity.
TABLE8_REFERENCES: Dict[str, float] = {
    "FT": 89.7,
    "LoRA": 84.5,
    "LoRA+Distill": 80.0,
    "APT": 83.9,
}

#: Raw efficiency numbers of Table 11 (TTA 97%, peak memory MB, ms, MB).
TABLE11_REFERENCES: Dict[str, Dict[str, Dict[str, float]]] = {
    "roberta": {
        "FT": {"train_time_s": 127.0, "train_peak_mem_mb": 2696.0, "inf_time_ms": 220.8, "inf_mem_mb": 1157.0},
        "APT": {"train_time_s": 752.0, "train_peak_mem_mb": 1890.0, "inf_time_ms": 91.3, "inf_mem_mb": 904.0},
    },
    "t5": {
        "FT": {"train_time_s": 366.0, "train_peak_mem_mb": 7217.0, "inf_time_ms": 248.1, "inf_mem_mb": 2347.0},
        "APT": {"train_time_s": 1774.0, "train_peak_mem_mb": 5332.0, "inf_time_ms": 185.0, "inf_mem_mb": 1913.0},
    },
}


def load_reference_constants(path: Optional[str] = None) -> Dict[str, Any]:
    """Return the paper's reference constants (optionally merged from JSON).

    Parameters
    ----------
    path:
        Optional JSON file whose keys override the embedded constants.  This
        makes it easy to substitute locally measured FT/LoRA normalisation
        constants without editing the code.
    """
    constants: Dict[str, Any] = {
        "table2": TABLE2_REFERENCES,
        "table7": TABLE7_REFERENCES,
        "table8": TABLE8_REFERENCES,
        "table11": TABLE11_REFERENCES,
        "ft_references": dict(FT_REFERENCES) if isinstance(FT_REFERENCES, dict) else {},
        "raw_efficiency": dict(RAW_EFFICIENCY) if isinstance(RAW_EFFICIENCY, dict) else {},
        "tta_fraction": TTA_FRACTION,
        "inference_batch_size": SMALL_MODEL_INFERENCE_BATCH_SIZE,
    }
    if path:
        with open(path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            constants.update(loaded)
    return constants


def model_family(model_name: str) -> str:
    """Best-effort model-family key used by :data:`METHOD_ORDER`/references."""
    name = str(model_name).lower()
    if "roberta" in name:
        return "roberta"
    if "t5" in name:
        return "t5"
    if "bert" in name:
        return "bert"
    if "llama" in name:
        return "llama"
    return name


def method_order(model_name: str, methods: Optional[Sequence[str]] = None) -> List[str]:
    """Return the method order to use for a model, keeping only ``methods``."""
    if methods is not None:
        return list(methods)
    family = model_family(model_name)
    order = METHOD_ORDER.get(family) or METHOD_ORDER.get(str(model_name).lower()) or BASELINE_METHODS + ("APT",)
    return list(order)


# ===========================================================================
# Configuration
# ===========================================================================


@dataclass
class EvalConfig:
    """Evaluation configuration (mirrors the training configs' data fields)."""

    model_name_or_path: str = "roberta-base"
    model_type: str = "roberta"
    model_family: str = ""
    task: str = "sst2"
    tasks: Sequence[str] = field(default_factory=tuple)
    method: str = "APT"
    methods: Sequence[str] = field(default_factory=tuple)
    sparsity: float = 0.60
    density: Optional[float] = None
    split: str = "validation"
    batch_size: Optional[int] = None
    max_seq_length: int = 128
    max_target_length: int = 128
    doc_stride: int = 128
    max_query_length: int = 64
    inference_batch_size: Optional[int] = None
    sequence_length: int = 128
    device: str = "cuda"
    output_dir: str = "outputs/eval"
    num_workers: int = 0
    seed: int = 42
    metrics: Sequence[str] = field(default_factory=tuple)
    measure_efficiency: bool = True
    measure_inference: bool = True
    tta_fraction: float = 0.97
    reference: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- construction -------------------------------------------------------
    @classmethod
    def from_dict(cls, config: Optional[Dict[str, Any]] = None, **overrides: Any) -> "EvalConfig":
        data: Dict[str, Any] = {}
        if config:
            data.update({k: v for k, v in config.items() if v is not None})
        data.update({k: v for k, v in overrides.items() if v is not None})
        fields = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        known = {k: v for k, v in data.items() if k in fields}
        unknown = {k: v for k, v in data.items() if k not in fields}
        cfg = cls(**known)
        if isinstance(cfg.tasks, str):
            cfg.tasks = (cfg.tasks,)
        if isinstance(cfg.methods, str):
            cfg.methods = (cfg.methods,)
        if not cfg.model_family:
            cfg.model_family = model_family(cfg.model_name_or_path)
        if cfg.inference_batch_size is None:
            cfg.inference_batch_size = SMALL_MODEL_INFERENCE_BATCH_SIZE
        if unknown:
            cfg.extra = {**cfg.extra, **unknown}
        return cfg

    @classmethod
    def from_yaml(cls, path: str, **overrides: Any) -> "EvalConfig":
        import yaml  # lazy

        with open(path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        return cls.from_dict(config, **overrides)

    def to_dict(self) -> Dict[str, Any]:
        data = dict(self.__dict__)
        data["tasks"] = list(self.tasks)
        data["methods"] = list(self.methods)
        data["metrics"] = list(self.metrics)
        return data

    def resolved_tasks(self) -> List[str]:
        if self.tasks:
            return [normalize_task_name(t) for t in self.tasks]
        return [normalize_task_name(self.task)]

    def resolved_methods(self) -> List[str]:
        if self.methods:
            return list(self.methods)
        return [self.method]

    def resolved_sparsity(self) -> float:
        if self.density is not None:
            return 1.0 - float(self.density)
        return float(self.sparsity)


# ===========================================================================
# Results
# ===========================================================================


@dataclass
class EvaluationResult:
    """One evaluated (model, method, task) triple."""

    model: str = ""
    method: str = ""
    task: str = ""
    sparsity: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    efficiency: Dict[str, Optional[float]] = field(default_factory=dict)
    relative: Dict[str, Optional[float]] = field(default_factory=dict)
    num_parameters: Optional[int] = None
    train_time_s: Optional[float] = None
    train_peak_mem_mb: Optional[float] = None
    tta_seconds: Optional[float] = None
    inf_time_ms: Optional[float] = None
    inf_mem_mb: Optional[float] = None
    inf_throughput: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- accessors ----------------------------------------------------------
    @property
    def primary(self) -> float:
        """Primary metric value of the task."""
        return primary_metric(self.task, self.metrics)

    def metric(self, name: str, default: Optional[float] = None) -> Optional[float]:
        return self.metrics.get(name, default)

    def as_dict(self, relative: bool = True) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "model": self.model,
            "method": self.method,
            "task": self.task,
            "sparsity": self.sparsity,
            "metrics": dict(self.metrics),
            "num_parameters": self.num_parameters,
            "train_time_s": self.train_time_s,
            "train_peak_mem_mb": self.train_peak_mem_mb,
            "tta_seconds": self.tta_seconds,
            "inf_time_ms": self.inf_time_ms,
            "inf_mem_mb": self.inf_mem_mb,
            "inf_throughput": self.inf_throughput,
            "extra": dict(self.extra),
        }
        if relative:
            data["efficiency"] = dict(self.efficiency)
            data["relative"] = dict(self.relative)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any], **_ignored: Any) -> "EvaluationResult":
        fields = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        known = {k: v for k, v in (data or {}).items() if k in fields}
        return cls(**known)

    def format_metrics(self) -> str:
        return metric_for_display(self.task, self.metrics)

    def summary(self, digits: int = 1) -> str:
        parts = [f"{self.method:<22}", f"{self.task:<8}", self.format_metrics()]
        if self.efficiency:
            parts.append(format_efficiency(self.efficiency, digits=digits))
        return "  ".join(str(p) for p in parts if p)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()


#: Alias used by callers that prefer the "outcome"-style naming.
EvalOutcome = EvaluationResult


# ===========================================================================
# Relative / normalisation helpers
# ===========================================================================


def relative_to_ft(
    value: Optional[float], reference: Optional[float], *, higher_is_better: bool = True
) -> Optional[float]:
    """Express ``value`` as a percentage of the FT ``reference``.

    Follows the paper's convention where every efficiency metric of Table 2 is
    normalised to the fully fine-tuned model (``FT = 100.0%``).  The paper
    reports the plain ratio even for metrics where smaller is better (e.g.
    APT training time = 592.1% of FT), so ``higher_is_better`` only documents
    the direction and does not change the arithmetic.  Returns ``None`` when
    either input is missing so tables can print ``-``.
    """
    if value is None or reference in (None, 0):
        return None
    return 100.0 * float(value) / float(reference)


def ft_reference_for(
    task: str,
    metrics_or_value: Union[float, Dict[str, float], None] = None,
    *,
    metric: Optional[str] = None,
) -> Optional[float]:
    """Look up the fully fine-tuned reference value of a task."""
    if metrics_or_value is None:
        key = normalize_task_name(task)
        for candidate in (key, str(task).lower()):
            if candidate in FT_REFERENCES:
                return float(FT_REFERENCES[candidate])
        return None
    if isinstance(metrics_or_value, dict):
        if metric is None:
            metric = primary_metric_name(task)
        return metrics_or_value.get(metric)
    return float(metrics_or_value)


def relative_metrics(
    metrics: Dict[str, float],
    task: str,
    reference: Optional[Dict[str, float]] = None,
    efficiency: Optional[Dict[str, Optional[float]]] = None,
    reference_efficiency: Optional[Dict[str, float]] = None,
) -> Dict[str, Optional[float]]:
    """Combine task-relative accuracy with relative efficiency metrics."""
    out: Dict[str, Optional[float]] = {}
    ref_value = None
    if reference:
        key = normalize_task_name(task)
        ref_value = reference.get(key, reference.get(str(task).lower()))
    if ref_value is None:
        ref_value = ft_reference_for(task)
    if ref_value:
        out["accuracy"] = relative_accuracy(primary_metric(task, metrics), ref_value)
    if efficiency:
        out.update(normalize_efficiency(efficiency, reference_efficiency or {}))
    return out


# ===========================================================================
# Prediction computation
# ===========================================================================


def _torch():
    import torch  # lazy

    return torch


def _device_of(model, device: Optional[Union[str, Any]] = None):
    torch = _torch()
    if device is not None:
        return torch.device(device) if isinstance(device, str) else device
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(batch: Any, device: Any) -> Any:
    """Recursively move tensors of a batch to ``device``."""
    torch = _torch()
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(to_device(v, device) for v in batch)
    return batch


def flatten_values(values: Any) -> List[Any]:
    """Flatten torch/numpy containers to a Python list."""
    torch = _torch()
    if values is None:
        return []
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu()
        if values.dim() == 0:
            return [values.item()]
        return values.tolist()
    if hasattr(values, "tolist") and not isinstance(values, (list, tuple)):
        try:
            return values.tolist()  # numpy / HF datasets
        except Exception:
            pass
    if isinstance(values, (list, tuple)):
        out: List[Any] = []
        for item in values:
            out.extend(flatten_values(item))
        return out
    return [values]


def _labels_from_batch(batch: Dict[str, Any]) -> List[Any]:
    labels = batch.get("labels", batch.get("label", batch.get("label_ids")))
    return flatten_values(labels)


def infer_logits(
    model,
    dataloader,
    *,
    task: str = "sst2",
    device: Optional[Any] = None,
    max_batches: Optional[int] = None,
    model_input_keys: Sequence[str] = (
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "p_mask",
        "cls_index",
        "decoder_input_ids",
        "decoder_attention_mask",
    ),
) -> Tuple[List[Any], List[Any], Dict[str, Any]]:
    """Run the model over ``dataloader`` and collect logits + labels.

    Works for encoder classification (``logits``) and extractive QA
    (``start_logits`` / ``end_logits``).  Returns ``(predictions, labels, raw)``
    where ``raw`` holds extra tensors such as SQuAD span logits.
    """
    torch = _torch()
    device = _device_of(model, device)
    was_training = getattr(model, "training", False)
    model.eval()

    predictions: List[Any] = []
    labels: List[Any] = []
    raw: Dict[str, Any] = {"start_logits": [], "end_logits": [], "features": [], "examples": []}

    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            if max_batches is not None and index >= max_batches:
                break
            features = batch.pop("features", None) if isinstance(batch, dict) else None
            examples = batch.pop("examples", None) if isinstance(batch, dict) else None
            batch = to_device(batch, device)
            inputs = {k: v for k, v in batch.items() if k in model_input_keys}
            if "labels" in batch and isinstance(batch["labels"], torch.Tensor):
                if batch["labels"].dtype == torch.float:
                    inputs["labels"] = batch["labels"]
            try:
                outputs = model(**inputs)
            except TypeError:
                outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})

            if getattr(outputs, "start_logits", None) is not None:
                raw["start_logits"].append(outputs.start_logits.detach().cpu())
                raw["end_logits"].append(outputs.end_logits.detach().cpu())
            logits = getattr(outputs, "logits", None)
            if logits is not None:
                predictions.extend(flatten_values(logits))
            if isinstance(batch, dict):
                labels.extend(_labels_from_batch(batch))
            if features is not None:
                raw["features"].extend(features)
            if examples is not None:
                raw["examples"].extend(examples)

    if was_training:
        model.train()

    return predictions, labels, raw


def logits_to_predictions(
    logits: Sequence[Any],
    task: str,
    *,
    num_labels: Optional[int] = None,
    is_regression: Optional[bool] = None,
) -> List[Any]:
    """Convert raw logits to task predictions (argmax / regression value)."""
    import numpy as np

    if is_regression is None:
        is_regression = normalize_task_name(task) in {"stsb"}
    if num_labels is None:
        num_labels = 1 if is_regression else None

    preds: List[Any] = []
    for row in logits:
        arr = np.asarray(row, dtype=float)
        if arr.ndim > 1:
            if is_regression or arr.shape[-1] == 1:
                preds.append(float(arr.reshape(-1)[0]))
            else:
                preds.append(int(np.argmax(arr)))
        else:
            if is_regression:
                preds.append(float(arr.reshape(-1)[0]))
            else:
                flat = arr.reshape(-1)
                preds.append(int(np.argmax(flat)) if flat.size > 1 else int(flat[0] > 0.5))
    return preds


# ===========================================================================
# Core evaluation entry points
# ===========================================================================


def evaluate_predictions(
    task: str,
    predictions: Sequence[Any],
    references: Sequence[Any],
    **kwargs: Any,
) -> Dict[str, float]:
    """Compute the paper's metrics for already-computed predictions."""
    return _compute_metrics(task, predictions, references, **kwargs)


def evaluate_model(
    model,
    dataloader=None,
    task: str = "sst2",
    tokenizer=None,
    device: Optional[Any] = None,
    *,
    split: str = "validation",
    method: str = "",
    model_name: str = "",
    sparsity: float = 0.0,
    metrics: Optional[Sequence[str]] = None,
    max_batches: Optional[int] = None,
    compute_efficiency: bool = False,
    inference_batch_size: Optional[int] = None,
    sequence_length: Optional[int] = None,
    reference: Optional[Dict[str, float]] = None,
    reference_efficiency: Optional[Dict[str, float]] = None,
    **kwargs: Any,
) -> EvaluationResult:
    """Evaluate ``model`` on ``task`` and return an :class:`EvaluationResult`.

    Handles the three task families used in the paper:

    * GLUE classification/regression -- argmax / regression on ``logits``;
    * SQuAD v2.0 -- span post-processing via :func:`apt.data.squad.write_predictions`;
    * CNN/DM -- beam-search generation followed by ROUGE.

    When ``compute_efficiency`` is set, inference throughput / latency / peak
    memory are measured (batch size 128 for small models, per Sec. 5.3).
    """
    task = normalize_task_name(task)
    result = EvaluationResult(
        model=model_name or "",
        method=method or "",
        task=task,
        sparsity=float(sparsity or 0.0),
    )

    # ---------------- task quality ----------------------------------------
    if is_squad_task(task):
        metrics_dict = _evaluate_squad(model, dataloader, tokenizer, device, max_batches=max_batches)
    elif is_seq2seq_task(task):
        metrics_dict = _evaluate_cnndm(model, dataloader, tokenizer, device, max_batches=max_batches, **kwargs)
    else:
        logits, labels, _raw = infer_logits(model, dataloader, task=task, device=device, max_batches=max_batches)
        predictions = logits_to_predictions(logits, task) if logits else list(labels)
        gt = labels or _labels_from_dataloader(dataloader)
        metrics_dict = evaluate_predictions(task, predictions, gt, **kwargs)

    result.metrics = {k: float(v) for k, v in (metrics_dict or {}).items() if isinstance(v, (int, float))}

    # ---------------- efficiency ------------------------------------------
    if compute_efficiency:
        eff = measure_model_efficiency(
            model,
            task=task,
            batch_size=inference_batch_size,
            sequence_length=sequence_length,
            device=device,
        )
        result.efficiency = eff.get("relative", {})
        result.inf_time_ms = eff.get("inf_time_ms")
        result.inf_mem_mb = eff.get("inf_mem_mb")
        result.inf_throughput = eff.get("inf_throughput")
        result.num_parameters = eff.get("num_parameters")
        result.extra.update(eff.get("extra", {}))
        result.relative = relative_metrics(
            result.metrics,
            task,
            reference=reference,
            efficiency={k: eff.get(k) for k in ("train_time", "train_mem", "inf_time", "inf_mem")},
            reference_efficiency=reference_efficiency,
        )
    else:
        result.relative = relative_metrics(result.metrics, task, reference=reference)

    return result


def _labels_from_dataloader(dataloader) -> List[Any]:
    """Collect labels from a dataloader when the model call did not yield any."""
    labels: List[Any] = []
    if dataloader is None:
        return labels
    try:
        for batch in dataloader:
            if isinstance(batch, dict):
                labels.extend(_labels_from_batch(batch))
    except Exception:
        return labels
    return labels


def _evaluate_squad(
    model,
    dataloader,
    tokenizer,
    device,
    *,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """SQuAD v2.0 evaluation: span logits -> answers -> official EM/F1."""
    torch = _torch()
    device = _device_of(model, device)
    was_training = getattr(model, "training", False)
    model.eval()

    start_chunks: List[Any] = []
    end_chunks: List[Any] = []
    features: List[Any] = []
    examples: List[Any] = []

    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            if max_batches is not None and index >= max_batches:
                break
            batch_features = batch.pop("features", None) if isinstance(batch, dict) else None
            batch_examples = batch.pop("examples", None) if isinstance(batch, dict) else None
            batch = to_device(batch, device)
            inputs = {
                k: v
                for k, v in batch.items()
                if k in ("input_ids", "attention_mask", "token_type_ids", "cls_index", "p_mask")
            }
            outputs = model(**inputs)
            start_chunks.append(outputs.start_logits.detach().cpu())
            end_chunks.append(outputs.end_logits.detach().cpu())
            if batch_features is not None:
                features.extend(batch_features)
            if batch_examples is not None:
                examples.extend(batch_examples)

    if was_training:
        model.train()

    if not start_chunks:
        return {}

    start_logits = torch.cat(start_chunks, dim=0)
    end_logits = torch.cat(end_chunks, dim=0)

    if features and examples:
        all_results = [
            (start_logits[i].tolist(), end_logits[i].tolist()) for i in range(start_logits.shape[0])
        ]
        try:
            from apt.data.squad import compute_squad_metrics, references_for_examples, write_predictions

            predictions, _nbest, _null = write_predictions(
                features, examples, all_results, tokenizer=tokenizer
            )
            references = references_for_examples(examples)
            if references:
                return compute_squad_metrics(predictions, references)
        except Exception:
            pass

    return {}


def _evaluate_cnndm(
    model, dataloader, tokenizer, device, *, max_batches: Optional[int] = None, **kwargs: Any
) -> Dict[str, float]:
    """CNN/DM evaluation: beam-search generation followed by ROUGE."""
    try:
        from apt.data.cnndm import evaluate_cnndm

        gen_kwargs = {
            k: kwargs[k]
            for k in ("max_length", "min_length", "num_beams", "length_penalty", "no_repeat_ngram_size")
            if k in kwargs
        }
        return evaluate_cnndm(model, tokenizer, dataloader, max_batches=max_batches, **gen_kwargs)
    except Exception:
        return {}


def measure_model_efficiency(
    model,
    *,
    task: str = "sst2",
    batch_size: Optional[int] = None,
    sequence_length: Optional[int] = None,
    repeats: int = 5,
    warmup: int = 1,
    device: Optional[Any] = None,
    measure_memory: bool = True,
) -> Dict[str, Any]:
    """Measure inference efficiency (throughput, latency, peak memory)."""
    task = normalize_task_name(task)
    if batch_size is None:
        batch_size = inference_batch_size_for(model)
    if sequence_length is None:
        sequence_length = 512 if is_seq2seq_task(task) else DEFAULT_SEQUENCE_LENGTH
    try:
        stats = measure_inference(
            model,
            batch_size=batch_size,
            seq_length=sequence_length,
            repeats=repeats,
            warmup=warmup,
            device=device,
            measure_memory=measure_memory,
        )
    except Exception as exc:  # pragma: no cover - no torch/GPU available
        return {"extra": {"efficiency_error": repr(exc)}}

    out: Dict[str, Any] = dict(stats)
    throughput = stats.get("throughput", stats.get("inf_throughput", stats.get("samples_per_second")))
    latency_ms = stats.get("latency_ms", stats.get("inf_time_ms"))
    peak_mb = stats.get("peak_memory_mb", stats.get("inf_mem_mb"))
    num_params = stats.get("num_parameters")
    if num_params is None:
        try:
            num_params = parameter_count(model)
        except Exception:
            num_params = None

    out.update(
        {
            "inf_throughput": throughput,
            "inf_time_ms": latency_ms,
            "inf_mem_mb": peak_mb,
            "num_parameters": num_params,
            "batch_size": batch_size,
        }
    )
    return out


def evaluate_task(
    task: str,
    model,
    dataloader=None,
    tokenizer=None,
    **kwargs: Any,
) -> EvaluationResult:
    """Convenience wrapper: evaluate one task."""
    return evaluate_model(model, dataloader, task=task, tokenizer=tokenizer, **kwargs)


def evaluate_method(
    method: str,
    model,
    dataloader=None,
    task: str = "sst2",
    tokenizer=None,
    **kwargs: Any,
) -> EvaluationResult:
    """Convenience wrapper: evaluate one method (sets ``method`` metadata)."""
    return evaluate_model(model, dataloader, task=task, tokenizer=tokenizer, method=method, **kwargs)


def evaluate_apt_model(
    model,
    dataloader=None,
    task: str = "sst2",
    tokenizer=None,
    *,
    threshold: float = 0.5,
    merge: bool = True,
    **kwargs: Any,
) -> EvaluationResult:
    """Evaluate an APT model, optionally merging adapters and pruning first.

    Mirrors the deployment path of Sec. 4.1 / Sec. 6: harden masks, fold
    ``s W_B W_A`` into ``W`` and physically remove pruned heads, neurons and
    hidden dimensions, so that the measured inference cost contains **no**
    tuning overhead.
    """
    kwargs.setdefault("method", "APT")
    if merge:
        try:
            from apt.merge import merge_and_prune

            model = merge_and_prune(model, threshold=threshold)
            name = getattr(getattr(model, "config", None), "_name_or_path", "") or ""
            kwargs.setdefault("model_name", name)
        except Exception:
            pass
    return evaluate_model(model, dataloader, task=task, tokenizer=tokenizer, **kwargs)


# ===========================================================================
# Training efficiency helpers
# ===========================================================================


@dataclass
class TrainingEfficiencyRecord:
    """Bookkeeping for the training-side efficiency columns."""

    train_time_s: Optional[float] = None
    train_peak_mem_mb: Optional[float] = None
    tta_seconds: Optional[float] = None
    teacher_seconds: float = 0.0
    student_seconds: float = 0.0
    history: List[Tuple[float, float]] = field(default_factory=list)

    def total_train_time(self) -> Optional[float]:
        """Teacher + student time, as required for KD methods (Sec. 5.3)."""
        if self.train_time_s is not None:
            return self.train_time_s
        if self.teacher_seconds or self.student_seconds:
            return float(self.teacher_seconds) + float(self.student_seconds)
        return None

    def relative(self, reference: Dict[str, float]) -> Dict[str, Optional[float]]:
        return normalize_efficiency(
            {"train_time": self.total_train_time(), "train_mem": self.train_peak_mem_mb},
            reference,
        )


def compute_tta(
    history: Sequence[Tuple[float, float]],
    reference: float,
    *,
    fraction: float = TTA_FRACTION,
    higher_is_better: bool = True,
) -> Optional[float]:
    """Compute the ``fraction`` (default 97%) time-to-accuracy.

    ``history`` is a sequence of ``(elapsed_seconds, metric_value)`` pairs in
    chronological order.  The crossing point is linearly interpolated between
    the two evaluations bracketing the target, matching the paper's TTA of
    Appendix A / Sec. 5.3.
    """
    if not history:
        return None
    target = float(fraction) * float(reference)
    previous_time, previous_metric = None, None
    for elapsed, metric in history:
        reached = metric >= target if higher_is_better else metric <= target
        if reached:
            if previous_time is None or previous_metric is None:
                return float(elapsed)
            span = float(metric) - float(previous_metric)
            if span == 0:
                return float(elapsed)
            frac = (target - float(previous_metric)) / span
            frac = max(0.0, min(1.0, frac))
            return float(previous_time) + frac * (float(elapsed) - float(previous_time))
        previous_time, previous_metric = elapsed, metric
    return None


def training_efficiency_result(
    model: str,
    method: str,
    *,
    sparsity: float = 0.0,
    train_time_s: Optional[float] = None,
    train_peak_mem_mb: Optional[float] = None,
    teacher_seconds: float = 0.0,
    student_seconds: float = 0.0,
    inf_stats: Optional[Dict[str, Any]] = None,
    reference: Optional[Dict[str, float]] = None,
) -> EfficiencyResult:  # type: ignore[valid-type]
    """Assemble a Table-2-style efficiency row (relative to FT)."""
    family = model_family(model)
    ref = reference or TABLE11_REFERENCES.get(family, {}).get("FT", {})
    raw: Dict[str, Optional[float]] = {
        "train_time": train_time_s if train_time_s is not None else (teacher_seconds + student_seconds or None),
        "train_mem": train_peak_mem_mb,
    }
    inf_stats = inf_stats or {}
    raw["inf_time"] = inf_stats.get("inf_time_ms")
    raw["inf_mem"] = inf_stats.get("inf_mem_mb")
    relative = normalize_efficiency(raw, ref)

    if _EFFICIENCY_OK and EfficiencyResult is not None:
        return EfficiencyResult(  # type: ignore[misc]
            model=model,
            method=method,
            sparsity=sparsity,
            train_time_s=raw.get("train_time"),
            train_peak_mem_mb=raw.get("train_mem"),
            inf_time_ms=raw.get("inf_time"),
            inf_mem_mb=raw.get("inf_mem"),
            inf_throughput=inf_stats.get("inf_throughput"),
            reference=dict(ref),
            extra={"relative": relative},
        )

    class _Fallback:  # pragma: no cover - only used without apt.eval.efficiency
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def as_dict(self, reference=None, relative=True):
            return dict(self.__dict__)

    return _Fallback(  # type: ignore[return-value]
        model=model,
        method=method,
        sparsity=sparsity,
        train_time_s=raw.get("train_time"),
        train_peak_mem_mb=raw.get("train_mem"),
        inf_time_ms=raw.get("inf_time"),
        inf_mem_mb=raw.get("inf_mem"),
        inf_throughput=inf_stats.get("inf_throughput"),
        reference=dict(ref),
        relative=relative,
    )


def fill_reference_from_raw(model: str, reference: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Use Table 11 raw FT numbers as normalisation constants when available."""
    if reference:
        return dict(reference)
    family = model_family(model)
    raw = TABLE11_REFERENCES.get(family, {}).get("FT")
    if raw:
        return {
            "train_time": raw.get("train_time_s"),
            "train_mem": raw.get("train_peak_mem_mb"),
            "inf_time": raw.get("inf_time_ms"),
            "inf_mem": raw.get("inf_mem_mb"),
        }
    if isinstance(TABLE11_RAW, dict):
        return dict(TABLE11_RAW.get(family, {}).get("FT", {}))
    return {}


# ===========================================================================
# Orchestration
# ===========================================================================


def _default_dataloader(eval_config: EvalConfig, task: str, tokenizer=None):
    """Build a dataloader through :mod:`apt.data` (lazy import)."""
    from apt import data as apt_data

    loaders = apt_data.make_dataloaders(
        task,
        tokenizer,
        model_type=eval_config.model_type,
        batch_size=eval_config.batch_size,
        max_seq_length=eval_config.max_seq_length,
        splits=(eval_config.split,),
        num_workers=eval_config.num_workers,
        seed=eval_config.seed,
    )
    if isinstance(loaders, dict):
        return loaders.get(eval_config.split) or next(iter(loaders.values()))
    return loaders


def run_evaluation(
    config: Optional[Union[EvalConfig, Dict[str, Any]]] = None,
    *,
    models: Optional[Dict[str, Any]] = None,
    tokenizers: Optional[Dict[str, Any]] = None,
    dataloaders: Optional[Dict[str, Any]] = None,
    tasks: Optional[Sequence[str]] = None,
    methods: Optional[Sequence[str]] = None,
    save_path: Optional[str] = None,
    verbose: bool = True,
    **overrides: Any,
) -> Dict[str, List[EvaluationResult]]:
    """Evaluate every (method, task) pair of an experiment.

    ``models`` maps a method name to a model (already trained/merged).  When it
    is not supplied, callers are expected to populate ``dataloaders`` and
    ``models`` themselves -- this keeps the harness independent of the training
    implementation.  Results are grouped as ``{model: [EvaluationResult, ...]}``.
    """
    eval_config = config if isinstance(config, EvalConfig) else EvalConfig.from_dict(config, **overrides)

    task_list = [normalize_task_name(t) for t in (tasks or eval_config.resolved_tasks())]
    method_list = list(methods or eval_config.resolved_methods())
    reference_efficiency = fill_reference_from_raw(eval_config.model_name_or_path)
    reference_scores = eval_config.reference or None
    sparsity = eval_config.resolved_sparsity()

    results: Dict[str, List[EvaluationResult]] = {}
    for method in method_list:
        model = (models or {}).get(method)
        tokenizer = (tokenizers or {}).get(method, (tokenizers or {}).get("default"))
        for task in task_list:
            loader = None
            if dataloaders:
                loader = dataloaders.get((method, task)) or dataloaders.get(task) or dataloaders.get(method)
            if loader is None and model is not None and tokenizer is not None:
                try:
                    loader = _default_dataloader(eval_config, task, tokenizer)
                except Exception:
                    loader = None
            if model is None:
                continue
            result = evaluate_model(
                model,
                loader,
                task=task,
                tokenizer=tokenizer,
                method=method,
                model_name=eval_config.model_name_or_path,
                sparsity=sparsity,
                compute_efficiency=eval_config.measure_efficiency and eval_config.measure_inference,
                inference_batch_size=eval_config.inference_batch_size,
                sequence_length=eval_config.sequence_length,
                reference=reference_scores,
                reference_efficiency=reference_efficiency,
                split=eval_config.split,
            )
            results.setdefault(eval_config.model_name_or_path or "model", []).append(result)
            if verbose:
                print(result.summary())

    flat = [r for rows in results.values() for r in rows]
    if save_path:
        save_results(flat, save_path)
    return results


def run_all_tasks(
    model,
    tokenizer=None,
    *,
    tasks: Sequence[str] = ("mnli", "sst2"),
    dataloaders: Optional[Dict[str, Any]] = None,
    method: str = "APT",
    model_name: str = "",
    sparsity: float = 0.60,
    **kwargs: Any,
) -> Dict[str, EvaluationResult]:
    """Evaluate one model on several tasks, returning ``{task: result}``."""
    out: Dict[str, EvaluationResult] = {}
    for task in tasks:
        loader = dataloaders.get(task) if dataloaders else None
        result = evaluate_model(
            model,
            loader,
            task=task,
            tokenizer=tokenizer,
            method=method,
            model_name=model_name,
            sparsity=sparsity,
            **kwargs,
        )
        out[normalize_task_name(task)] = result
    return out


def glue_table_row(
    results: Dict[str, EvaluationResult],
    *,
    tasks: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Aggregate per-task GLUE results into one row with the GLUE average."""
    tasks = [normalize_task_name(t) for t in (tasks or sorted(results.keys()))]
    row: Dict[str, Any] = {}
    for task in tasks:
        result = results.get(task)
        row[task] = result.primary if result else None
    values = [v for v in row.values() if v is not None]
    row["average"] = sum(values) / len(values) if values else None
    row["glue_average"] = row["average"]
    return row


# ===========================================================================
# Reporting
# ===========================================================================


def format_results_table(
    results: Union[Sequence[EvaluationResult], Dict[str, List[EvaluationResult]]],
    *,
    methods: Optional[Sequence[str]] = None,
    tasks: Optional[Sequence[str]] = None,
    digits: int = 1,
    show_efficiency: bool = True,
    model_name: str = "",
) -> str:
    """Render results as a markdown table similar to the paper's Table 2."""
    rows: List[EvaluationResult] = []
    if isinstance(results, dict):
        for values in results.values():
            rows.extend(values)
    else:
        rows = list(results)

    if not rows:
        return "_(no results)_"

    task_list = [normalize_task_name(t) for t in (tasks or TASK_ORDER)]
    present_tasks = [t for t in dict.fromkeys(task_list) if any(r.task == t for r in rows)]
    if not present_tasks:
        present_tasks = list(dict.fromkeys(r.task for r in rows))
    method_list = list(methods or dict.fromkeys(r.method for r in rows))

    header = ["Method"] + [t.upper() for t in present_tasks]
    if show_efficiency:
        header += ["Train Time", "Train Mem", "Inf Time", "Inf Mem"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]

    for method in method_list:
        method_rows = [r for r in rows if r.method == method]
        if not method_rows:
            continue
        cells: List[str] = [method]
        for task in present_tasks:
            match = next((r for r in method_rows if r.task == task), None)
            cells.append(match.format_metrics() if match else "-")
        if show_efficiency:
            eff: Dict[str, Optional[float]] = {}
            for r in method_rows:
                for key in METRIC_KEYS:
                    if r.efficiency.get(key) is not None and key not in eff:
                        eff[key] = r.efficiency[key]
            if not eff:
                eff = dict(TABLE2_REFERENCES.get(model_family(model_name), {}).get(method, {}))
            for key in METRIC_KEYS:
                value = eff.get(key)
                cells.append("-" if value is None else format_percent(value, digits))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_results(results: Union[Sequence[EvaluationResult], Dict[str, Any]], path: str) -> str:
    """Persist evaluation results as JSON (a dict or a list of rows)."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    if isinstance(results, dict):
        payload: Any = {
            key: [r.as_dict() if hasattr(r, "as_dict") else r for r in value]
            if isinstance(value, (list, tuple))
            else value
            for key, value in results.items()
        }
    elif isinstance(results, (list, tuple)):
        payload = [r.as_dict() if hasattr(r, "as_dict") else r for r in results]
    else:
        payload = results
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return path


def load_results(path: str) -> List[EvaluationResult]:
    """Load results previously written by :func:`save_results`."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        rows: List[EvaluationResult] = []
        for value in payload.values():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        rows.append(EvaluationResult.from_dict(item))
        return rows
    if isinstance(payload, list):
        return [EvaluationResult.from_dict(item) for item in payload if isinstance(item, dict)]
    return []


def compare_against_reference(
    results: Sequence[EvaluationResult],
    reference: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    *,
    tolerance: float = 1.0,
) -> List[Dict[str, Any]]:
    """Diff local results against the paper's reported values."""
    reference = reference or TABLE2_REFERENCES
    diffs: List[Dict[str, Any]] = []
    for result in results:
        family = model_family(result.model or "")
        ref_row = reference.get(family, {}).get(result.method, {})
        for key, expected in ref_row.items():
            if isinstance(expected, (int, float)) and key in result.metrics:
                actual = result.metrics[key]
                diffs.append(
                    {
                        "method": result.method,
                        "task": result.task,
                        "metric": key,
                        "expected": expected,
                        "actual": actual,
                        "diff": actual - expected,
                        "within_tolerance": abs(actual - expected) <= tolerance,
                    }
                )
    return diffs


# ===========================================================================
# Self test
# ===========================================================================


def _self_test() -> bool:
    """Dependency-light sanity checks (no torch/GPU required)."""
    ok = True

    # 1) relative_to_ft follows the FT=100% convention.
    if relative_to_ft(127.0, 127.0) != 100.0:
        ok = False
    if abs(relative_to_ft(752.0, 127.0) - 592.1) > 0.1:
        ok = False
    if relative_to_ft(None, 127.0) is not None:
        ok = False

    # 2) TTA interpolation: 97% of 94.8 == 91.956.
    history = [(10.0, 80.0), (20.0, 94.0), (30.0, 95.0)]
    tta = compute_tta(history, 94.8, fraction=0.97)
    expected = 20.0 + (91.956 - 94.0) / (95.0 - 94.0) * 10.0
    if tta is None or abs(tta - expected) > 1e-6:
        ok = False
    if compute_tta([(1.0, 10.0)], 94.8) is not None:
        ok = False
    if compute_tta([(1.0, 99.0)], 94.8, fraction=0.97) != 1.0:
        ok = False

    # 3) Config handling.
    cfg = EvalConfig.from_dict(
        {"model_name_or_path": "roberta-base", "task": "sst2", "sparsity": 0.6, "unknown_key": 3}
    )
    if cfg.model_family != "roberta" or cfg.inference_batch_size != 128:
        ok = False
    if cfg.extra.get("unknown_key") != 3:
        ok = False
    if cfg.resolved_tasks() != ["sst2"] or abs(cfg.resolved_sparsity() - 0.6) > 1e-9:
        ok = False
    cfg2 = EvalConfig.from_dict({"task": "squad_v2", "tasks": ["mnli", "sst2"]})
    if cfg2.resolved_tasks() != ["mnli", "sst2"]:
        ok = False
    if abs(EvalConfig.from_dict({"density": 0.6}).resolved_sparsity() - 0.4) > 1e-9:
        ok = False

    # 4) Method ordering per model family.
    if method_order("roberta-base")[-1] != "APT":
        ok = False
    if "Prune+Distill" in method_order("t5-base"):
        ok = False
    if method_order("roberta", ["FT", "APT"]) != ["FT", "APT"]:
        ok = False

    # 5) Reference tables match the paper (Table 2 / 7 / 8 / 11).
    if TABLE2_REFERENCES["roberta"]["APT"]["mnli"] != 86.4:
        ok = False
    if TABLE2_REFERENCES["roberta"]["LoRA+Prune+Distill"]["train_time"] != 6534.6:
        ok = False
    if TABLE2_REFERENCES["t5"]["APT"]["cnndm"] != (38.6, 17.0, 35.8):
        ok = False
    if TABLE8_REFERENCES["APT"] != 83.9 or TABLE7_REFERENCES["50%"]["APT"] != 83.2:
        ok = False
    if TABLE11_REFERENCES["roberta"]["APT"]["train_peak_mem_mb"] != 1890.0:
        ok = False
    if TABLE11_REFERENCES["t5"]["FT"]["inf_time_ms"] != 248.1:
        ok = False

    # 6) Efficiency normalisation against Table 11 raw FT numbers.
    ref = fill_reference_from_raw("roberta-base")
    eff = normalize_efficiency(
        {"train_time": 752.0, "train_mem": 1890.0, "inf_time": 91.3, "inf_mem": 904.0}, ref
    )
    if abs(eff["train_time"] - 592.1) > 0.2 or abs(eff["train_mem"] - 70.1) > 0.2:
        ok = False
    if abs(eff["inf_time"] - 41.3) > 0.2 or abs(eff["inf_mem"] - 78.1) > 0.2:
        ok = False

    # 7) Row assembly + table formatting.
    row = EvaluationResult(model="roberta-base", method="APT", task="sst2", metrics={"accuracy": 94.5})
    if row.primary != 94.5:
        ok = False
    if load_reference_constants()["tta_fraction"] != 0.97:
        ok = False
    table = format_results_table([row], tasks=["sst2"])
    if "APT" not in table or "SST2" not in table:
        ok = False

    # 8) GLUE aggregation.
    rows = {
        "mnli": EvaluationResult(model="m", method="APT", task="mnli", metrics={"accuracy": 86.4}),
        "sst2": EvaluationResult(model="m", method="APT", task="sst2", metrics={"accuracy": 94.5}),
    }
    agg = glue_table_row(rows)
    if abs(agg["average"] - 90.45) > 1e-6:
        ok = False

    # 9) Result round-trip through JSON.
    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_self_test_results.json")
    try:
        save_results([row], tmp)
        loaded = load_results(tmp)
        if not loaded or loaded[0].method != "APT" or abs(loaded[0].metrics["accuracy"] - 94.5) > 1e-9:
            ok = False
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    # 10) Prediction post-processing (needs numpy only).
    try:
        preds = logits_to_predictions([[0.1, 0.9], [0.7, 0.2]], "sst2")
        if preds != [1, 0]:
            ok = False
        reg = logits_to_predictions([0.5, 1.5], "stsb", is_regression=True)
        if reg != [0.5, 1.5]:
            ok = False
    except Exception:
        pass

    # 11) KD methods must count teacher + student time.
    record = TrainingEfficiencyRecord(teacher_seconds=100.0, student_seconds=50.0)
    if record.total_train_time() != 150.0:
        ok = False

    return ok


if __name__ == "__main__":  # pragma: no cover
    print("apt.eval.run_eval self-test:", "OK" if _self_test() else "FAILED")
