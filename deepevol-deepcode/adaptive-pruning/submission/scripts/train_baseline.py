#!/usr/bin/env python
"""Training entry point for the APT baselines (paper Section 5.2).

This script is the baseline counterpart of ``scripts/train_apt.py``.  It resolves a
configuration (``default.yaml`` -> task config -> explicit overrides), builds the
HuggingFace model/tokenizer and the task dataloaders, then dispatches to one of the
baseline implementations:

======================  ==========================================================
CLI / registry key      Implementation
======================  ==========================================================
``ft``                  ``apt.baselines.ft.train_ft``            (Table 2 "FT")
``lora``                ``apt.baselines.lora.train_lora``        (Table 2 "LoRA")
``mask_tuning``         ``apt.baselines.mask_tuning.train_mask_tuning``
                        (Table 2 "LoRA+Prune")
``cofi``                ``apt.baselines.cofi.train_cofi``        (Table 2 "Prune+Distill")
``lora_prune_distill``  ``apt.baselines.lora_prune_distill.train_lora_prune_distill``
                        (Table 2 "LoRA+Prune+Distill")
======================  ==========================================================

Usage examples::

    python scripts/train_baseline.py --method ft --model roberta-base --task sst2
    python scripts/train_baseline.py --method lora --config apt/configs/roberta_mnli.yaml
    python scripts/train_baseline.py --method cofi --task mnli --sparsity 0.6 --seeds 42 43 44
    python scripts/train_baseline.py --all --task sst2 --output-dir outputs/baselines

Table 6 learning rates / batch sizes / epoch counts are applied automatically for
the task group unless explicitly overridden on the command line.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------------------
# Optional / defensive imports: this module must stay importable (and ``--help`` must
# work) in a bare environment without torch/transformers.
# --------------------------------------------------------------------------------------

try:  # pragma: no cover - exercised only when the package is importable
    from apt.baselines import (
        AVAILABLE as BASELINE_AVAILABLE,
        BASELINE_METHODS,
        METHOD_DISPLAY_NAMES,
        available as baselines_available,
        external_methods,
        get_method,
        in_repo_methods,
        method_display_name,
    )

    _HAS_BASELINES = True
except Exception as _exc:  # pragma: no cover
    BASELINE_AVAILABLE: Dict[str, bool] = {}
    BASELINE_METHODS: Tuple[str, ...] = (
        "ft",
        "lora",
        "mask_tuning",
        "cofi",
        "lora_prune_distill",
    )
    METHOD_DISPLAY_NAMES: Dict[str, str] = {
        "ft": "FT",
        "lora": "LoRA",
        "mask_tuning": "LoRA+Prune",
        "cofi": "Prune+Distill",
        "lora_prune_distill": "LoRA+Prune+Distill",
        "apt": "APT",
    }
    _BASELINES_IMPORT_ERROR = _exc
    _HAS_BASELINES = False

    def baselines_available() -> Dict[str, bool]:  # type: ignore[misc]
        return dict(BASELINE_AVAILABLE)

    def in_repo_methods() -> List[str]:  # type: ignore[misc]
        return []

    def external_methods() -> List[str]:  # type: ignore[misc]
        return []

    def get_method(name: str):  # type: ignore[misc]
        raise RuntimeError(f"apt.baselines unavailable: {_exc}")

    def method_display_name(name: str) -> str:  # type: ignore[misc]
        return METHOD_DISPLAY_NAMES.get(name, name)


try:
    from apt.data import make_dataloaders

    _HAS_DATA = True
except Exception:  # pragma: no cover
    make_dataloaders = None  # type: ignore[assignment]
    _HAS_DATA = False

try:
    from apt.eval.metrics import (
        FT_REFERENCES,
        compute_metrics as _compute_metrics,
        metric_for_display,
        primary_metric as _primary_metric,
        normalize_task_name,
    )

    _HAS_METRICS = True
except Exception:  # pragma: no cover
    FT_REFERENCES: Dict[str, float] = {}
    _compute_metrics = None  # type: ignore[assignment]
    metric_for_display = None  # type: ignore[assignment]
    _primary_metric = None  # type: ignore[assignment]
    normalize_task_name = None  # type: ignore[assignment]
    _HAS_METRICS = False

try:
    import yaml  # type: ignore

    _HAS_YAML = True
except Exception:  # pragma: no cover
    yaml = None  # type: ignore[assignment]
    _HAS_YAML = False


def _torch():
    """Import torch lazily (returns ``None`` when unavailable)."""
    try:
        import torch  # type: ignore

        return torch
    except Exception:  # pragma: no cover
        return None


# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

CONFIG_DIR = os.path.join("apt", "configs")
DEFAULT_CONFIG_FILE = os.path.join(CONFIG_DIR, "default.yaml")

GLUE_BIG_TASKS = ("mnli", "sst2", "qnli", "qqp")
GLUE_SMALL_TASKS = ("mrpc", "cola", "rte", "stsb")
GLUE_TASKS = GLUE_BIG_TASKS + GLUE_SMALL_TASKS
SQUAD_TASKS = ("squad", "squad_v2", "squad2")
SEQ2SEQ_TASKS = ("cnndm", "cnn_dailymail", "xsum", "samsum")

# Config aliases mirroring ``scripts/train_apt.py`` / ``main.py``.
CONFIG_ALIASES: Dict[Tuple[str, str], str] = {
    ("roberta", "sst2"): "roberta_sst2.yaml",
    ("roberta", "mnli"): "roberta_mnli.yaml",
    ("roberta", "squad"): "squad.yaml",
    ("roberta", "squad_v2"): "squad.yaml",
    ("t5", "cnndm"): "t5_cnndm.yaml",
    ("t5", "cnn_dailymail"): "t5_cnndm.yaml",
}

TABLE6_GROUPS: Dict[str, Dict[str, Any]] = {
    "glue-big": {
        "learning_rate": 2.0e-4,
        "batch_size": 32,
        "epochs": 40,
        "distill_epochs": 20,
        "max_seq_length": 128,
        "max_target_length": 8,
    },
    "glue-small": {
        "learning_rate": 2.0e-4,
        "batch_size": 32,
        "epochs": 40,
        "distill_epochs": 20,
        "max_seq_length": 128,
        "max_target_length": 8,
    },
    "squad": {
        "learning_rate": 2.0e-4,
        "batch_size": 32,
        "epochs": 40,
        "distill_epochs": 20,
        "max_seq_length": 384,
        "doc_stride": 128,
        "max_query_length": 64,
        "max_target_length": 128,
    },
    "cnndm": {
        "learning_rate": 1.0e-4,
        "batch_size": 16,
        "epochs": 16,
        "distill_epochs": 6,
        "max_seq_length": 512,
        "max_target_length": 128,
    },
}

METHOD_ALIASES: Dict[str, str] = {
    "ft": "ft",
    "finetune": "ft",
    "fine_tune": "ft",
    "fine-tuning": "ft",
    "full": "ft",
    "lora": "lora",
    "lora_only": "lora",
    "mask_tuning": "mask_tuning",
    "mask-tuning": "mask_tuning",
    "lora+prune": "mask_tuning",
    "lora_prune": "mask_tuning",
    "retraining_free": "mask_tuning",
    "retraining-free-pruning": "mask_tuning",
    "masktuning": "mask_tuning",
    "cofi": "cofi",
    "cofipruning": "cofi",
    "cofi_pruning": "cofi",
    "prune+distill": "cofi",
    "prune_distill": "cofi",
    "lora_prune_distill": "lora_prune_distill",
    "lora+prune+distill": "lora_prune_distill",
    "lora_cofi": "lora_prune_distill",
}

MODEL_INPUT_KEYS = (
    "input_ids",
    "attention_mask",
    "token_type_ids",
    "position_ids",
    "decoder_input_ids",
    "decoder_attention_mask",
    "labels",
    "start_positions",
    "end_positions",
    "p_mask",
    "cls_index",
)


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def normalize_task(task: Optional[str]) -> str:
    """Canonicalise a task name (``SST-2`` -> ``sst2``, ``squad2`` -> ``squad_v2``)."""
    if task is None:
        return "sst2"
    if _HAS_METRICS and normalize_task_name is not None:
        try:
            return normalize_task_name(task)
        except Exception:
            pass
    name = str(task).strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "sst": "sst2",
        "sst2": "sst2",
        "mnli": "mnli",
        "mnlimm": "mnli",
        "qnli": "qnli",
        "qqp": "qqp",
        "mrpc": "mrpc",
        "cola": "cola",
        "rte": "rte",
        "stsb": "stsb",
        "squad": "squad_v2",
        "squadv2": "squad_v2",
        "squad2": "squad_v2",
        "cnndm": "cnndm",
        "cnndailymail": "cnndm",
    }
    return aliases.get(name, str(task).strip().lower())


def normalize_method(method: Optional[str]) -> str:
    """Normalise a baseline method name to its canonical registry key."""
    if not method:
        return "ft"
    key = str(method).strip().lower().replace("-", "_")
    if key in BASELINE_METHODS:
        return key
    if key in METHOD_ALIASES:
        return METHOD_ALIASES[key]
    # ``+``-joined paper names such as ``LoRA+Prune+Distill``
    key2 = str(method).strip().lower().replace("+", "_").replace("-", "_")
    key2 = "_".join(part for part in key2.split("_") if part)
    if key2 in BASELINE_METHODS:
        return key2
    if key2 in METHOD_ALIASES:
        return METHOD_ALIASES[key2]
    raise KeyError(
        f"Unknown baseline method {method!r}. "
        f"Valid: {', '.join(BASELINE_METHODS)} (aliases accepted)."
    )


def display_name(method: str) -> str:
    """Paper table label for a method key."""
    try:
        key = normalize_method(method)
    except KeyError:
        return str(method)
    return method_display_name(key)


def table6_group_for(model_type: Optional[str], task: Optional[str]) -> str:
    """Map (model, task) to a Table 6 hyper-parameter column."""
    task_name = normalize_task(task)
    if task_name in SQUAD_TASKS or task_name == "squad_v2":
        return "squad"
    if task_name in SEQ2SEQ_TASKS:
        return "cnndm"
    if task_name in GLUE_BIG_TASKS:
        return "glue-big"
    if task_name in GLUE_SMALL_TASKS:
        return "glue-small"
    # Unknown GLUE-ish task: default to the GLUE-big column.
    return "glue-big"


def is_seq2seq_task(task: Optional[str]) -> bool:
    return normalize_task(task) in SEQ2SEQ_TASKS


def is_squad_task(task: Optional[str]) -> bool:
    return normalize_task(task) in SQUAD_TASKS or normalize_task(task) == "squad_v2"


def is_glue_task(task: Optional[str]) -> bool:
    t = normalize_task(task)
    return t in GLUE_TASKS or t not in tuple(SEQ2SEQ_TASKS) + ("squad_v2",)


# --------------------------------------------------------------------------------------
# Config handling
# --------------------------------------------------------------------------------------


def load_yaml(path: Optional[str]) -> Dict[str, Any]:
    """Load a YAML config file, returning ``{}`` when unavailable."""
    if not path or not os.path.isfile(path):
        return {}
    if not _HAS_YAML:
        warnings.warn(f"pyyaml not installed; ignoring config {path}")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return dict(data) if isinstance(data, dict) else {}
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"Failed to read config {path}: {exc}")
        return {}


def resolve_config_path(
    path: Optional[str] = None,
    model: Optional[str] = None,
    task: Optional[str] = None,
) -> Optional[str]:
    """Resolve an explicit path / filename / alias / ``(model, task)`` pair."""
    if path:
        if os.path.isfile(path):
            return path
        candidates = [
            path,
            os.path.join(CONFIG_DIR, path),
            os.path.join(os.getcwd(), path),
        ]
        base = os.path.basename(path)
        if not base.endswith((".yaml", ".yml")):
            candidates += [
                os.path.join(CONFIG_DIR, base + ".yaml"),
                os.path.join(CONFIG_DIR, base + ".yml"),
            ]
        for cand in candidates:
            if os.path.isfile(cand):
                return cand
        if model and task:
            # Fall through to the (model, task) lookup below.
            pass
        else:
            warnings.warn(f"Config not found: {path}")
            return None

    if model and task:
        model_key = str(model).strip().lower()
        for prefix in ("facebook/", "roberta-", "bert-", "t5-", "google/"):
            if model_key.startswith(prefix):
                model_key = model_key[len(prefix) :]
        if model_key.startswith("roberta"):
            model_family = "roberta"
        elif model_key.startswith("t5") or model_key.startswith("mt5"):
            model_family = "t5"
        elif model_key.startswith("bert"):
            model_family = "bert"
        else:
            model_family = model_key.split("-")[0]
        task_key = normalize_task(task)
        alias = CONFIG_ALIASES.get((model_family, task_key))
        if alias:
            cand = os.path.join(CONFIG_DIR, alias)
            if os.path.isfile(cand):
                return cand
    return None


def merge_config(
    path: Optional[str] = None,
    explicit: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
    task: Optional[str] = None,
) -> Dict[str, Any]:
    """Merge ``default.yaml`` < task config < explicit overrides; fill Table 6 values."""
    config: Dict[str, Any] = {}
    config.update(load_yaml(DEFAULT_CONFIG_FILE))

    resolved = resolve_config_path(path, model, task)
    if resolved and os.path.abspath(resolved) != os.path.abspath(DEFAULT_CONFIG_FILE):
        config.update(load_yaml(resolved))
    config["_config_path"] = resolved

    explicit = {k: v for k, v in (explicit or {}).items() if v is not None}
    config.update(explicit)

    if model:
        config["model_name_or_path"] = model
    if task:
        config["task"] = normalize_task(task)

    model_name = str(config.get("model_name_or_path", "") or "")
    if not config.get("model_type"):
        lower = model_name.lower()
        if "roberta" in lower:
            config["model_type"] = "roberta"
        elif "t5" in lower:
            config["model_type"] = "t5"
        elif "bert" in lower:
            config["model_type"] = "bert"
        elif "electra" in lower:
            config["model_type"] = "electra"
        elif "deberta" in lower:
            config["model_type"] = "deberta"

    group = table6_group_for(config.get("model_type"), config.get("task"))
    config["table6_group"] = group
    defaults = TABLE6_GROUPS.get(group, {})
    for key, value in defaults.items():
        config.setdefault(key, value)

    config.setdefault("seed", 42)
    config.setdefault("device", "cuda")
    config.setdefault("weight_decay", 0.01)
    config.setdefault("warmup_ratio", 0.06)
    config.setdefault("optimizer", "adamw")
    config.setdefault("lr_kind", "linear")
    config.setdefault("max_grad_norm", 1.0)
    config.setdefault("inference_batch_size", 128)
    config.setdefault("output_dir", os.path.join("outputs", "baselines"))
    return config


# --------------------------------------------------------------------------------------
# Model / tokenizer / dataloaders
# --------------------------------------------------------------------------------------


def num_labels_for(task: str) -> int:
    """Number of classification labels for a GLUE task."""
    t = normalize_task(task)
    labels = {
        "sst2": 2,
        "mnli": 3,
        "qnli": 2,
        "qqp": 2,
        "mrpc": 2,
        "cola": 2,
        "rte": 2,
        "stsb": 1,
    }
    return labels.get(t, 2)


def is_regression_task(task: str) -> bool:
    return normalize_task(task) == "stsb"


def build_tokenizer(config: Dict[str, Any]):
    """Build the HuggingFace tokenizer for the configured model."""
    from transformers import AutoTokenizer  # lazy import

    name = config.get("model_name_or_path") or config.get("model") or "roberta-base"
    kwargs: Dict[str, Any] = {"use_fast": bool(config.get("use_fast", True))}
    if "t5" in str(name).lower():
        kwargs.setdefault("add_prefix_space", None) if False else None
    try:
        return AutoTokenizer.from_pretrained(name, **kwargs)
    except Exception:
        kwargs.pop("use_fast", None)
        return AutoTokenizer.from_pretrained(name, **kwargs)


def build_model(config: Dict[str, Any], tokenizer=None):
    """Build a task-appropriate HuggingFace model (classification / QA / seq2seq)."""
    from transformers import (  # lazy import
        AutoConfig,
        AutoModelForQuestionAnswering,
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
    )

    name = config.get("model_name_or_path") or config.get("model") or "roberta-base"
    task = normalize_task(config.get("task"))
    try:
        hf_config = AutoConfig.from_pretrained(name)
    except Exception:
        hf_config = None

    if is_squad_task(task):
        return AutoModelForQuestionAnswering.from_pretrained(name, config=hf_config)
    if is_seq2seq_task(task):
        return AutoModelForSeq2SeqLM.from_pretrained(name, config=hf_config)

    if hf_config is not None:
        hf_config.num_labels = num_labels_for(task)
        if is_regression_task(task):
            try:
                hf_config.problem_type = "regression"
            except Exception:
                pass
    return AutoModelForSequenceClassification.from_pretrained(name, config=hf_config)


def build_dataloaders(config: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    """Build ``{"train": ..., "validation": ...}`` dataloaders for the task."""
    if not _HAS_DATA or make_dataloaders is None:
        raise RuntimeError("apt.data is unavailable; cannot build dataloaders.")
    task = normalize_task(config.get("task"))
    model_type = "t5" if "t5" in str(config.get("model_type", "")).lower() else "encoder"
    kwargs: Dict[str, Any] = {
        "batch_size": config.get("batch_size"),
        "max_seq_length": config.get("max_seq_length"),
        "model_type": model_type,
        "seed": config.get("seed", 42),
        "num_workers": config.get("num_workers", 0),
        "dynamic_padding": config.get("dynamic_padding", False),
    }
    for key in (
        "max_target_length",
        "doc_stride",
        "max_query_length",
        "cache_dir",
        "data_dir",
        "use_datasets",
    ):
        if config.get(key) is not None:
            kwargs[key] = config[key]
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    loaders = make_dataloaders(task, tokenizer, **kwargs)
    if isinstance(loaders, dict):
        if "dev" in loaders and "validation" not in loaders:
            loaders["validation"] = loaders.pop("dev")
        return loaders
    return {"train": loaders}


def build_metrics_fn(task: str) -> Optional[Callable[..., Dict[str, float]]]:
    """Return a ``(predictions, references) -> metrics`` callable for the task."""
    if not _HAS_METRICS or _compute_metrics is None:
        return None
    task_name = normalize_task(task)

    def _fn(predictions, references, **kwargs):
        return _compute_metrics(task_name, predictions, references, **kwargs)

    return _fn


def primary_value(task: str, metrics: Optional[Dict[str, float]]) -> Optional[float]:
    """Primary paper metric for a task."""
    if not metrics:
        return None
    if _HAS_METRICS and _primary_metric is not None:
        try:
            value = _primary_metric(normalize_task(task), metrics)
            if value is not None:
                return float(value)
        except Exception:
            pass
    for key in ("accuracy", "f1", "exact", "rougeL", "rouge-l", "rouge_l", "pearson", "spearmanr"):
        if key in metrics:
            try:
                return float(metrics[key])
            except Exception:
                continue
    for value in metrics.values():
        try:
            return float(value)
        except Exception:
            continue
    return None


def format_metrics(task: str, metrics: Optional[Dict[str, float]]) -> str:
    """Human readable metric string (ROUGE rendered as ``42.1/20.3/39.4``)."""
    if not metrics:
        return "-"
    if _HAS_METRICS and metric_for_display is not None:
        try:
            return metric_for_display(normalize_task(task), metrics)
        except Exception:
            pass
    parts = [f"{k}={v:.4g}" for k, v in metrics.items()]
    return ", ".join(parts)


# --------------------------------------------------------------------------------------
# Method dispatch
# --------------------------------------------------------------------------------------


def build_method_config(method: str, config: Dict[str, Any]):
    """Construct the method-specific config dataclass and return ``(cfg, train_fn)``."""
    key = normalize_method(method)
    merged = {k: v for k, v in config.items() if not str(k).startswith("_")}

    if key == "ft":
        from apt.baselines.ft import FTConfig, train_ft

        return FTConfig.from_dict(merged), train_ft
    if key == "lora":
        from apt.baselines.lora import LoRAConfig, train_lora

        return LoRAConfig.from_dict(merged), train_lora
    if key == "mask_tuning":
        from apt.baselines.mask_tuning import MaskTuningConfig, train_mask_tuning

        return MaskTuningConfig.from_dict(merged), train_mask_tuning
    if key == "cofi":
        from apt.baselines.cofi import CoFiConfig, train_cofi

        return CoFiConfig.from_dict(merged), train_cofi
    if key == "lora_prune_distill":
        from apt.baselines.lora_prune_distill import (
            LoRAPruneDistillConfig,
            train_lora_prune_distill,
        )

        return LoRAPruneDistillConfig.from_dict(merged), train_lora_prune_distill
    raise KeyError(f"Unknown baseline method {method!r}")


def reference_metric_for(config: Dict[str, Any]) -> Optional[float]:
    """FT reference metric used for time-to-accuracy tracking."""
    explicit = config.get("reference_metric")
    if explicit is not None:
        try:
            return float(explicit)
        except Exception:
            return None
    task = normalize_task(config.get("task"))
    if _HAS_METRICS and FT_REFERENCES:
        for key in (task, task.replace("_v2", ""), task.replace("-", "_")):
            if key in FT_REFERENCES:
                try:
                    return float(FT_REFERENCES[key])
                except Exception:
                    continue
    return None


def _json_safe(obj: Any) -> Any:
    """Recursively convert a summary dict into JSON-serialisable objects."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {
            str(k): _json_safe(v)
            for k, v in obj.items()
            if str(k) not in ("trainer", "model", "tokenizer")
        }
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "as_dict") and callable(obj.as_dict):
        try:
            return _json_safe(obj.as_dict())
        except Exception:
            pass
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return _json_safe(obj.to_dict())
        except Exception:
            pass
    return str(obj)


def run_baseline(
    method: str,
    config: Optional[Dict[str, Any]] = None,
    *,
    model=None,
    tokenizer=None,
    train_dataloader=None,
    eval_dataloader=None,
    verbose: bool = True,
    evaluate_now: bool = True,
) -> Dict[str, Any]:
    """Train one baseline and return its (JSON-friendly) summary dictionary.

    Parameters
    ----------
    method:
        Baseline key (``ft``, ``lora``, ``mask_tuning``, ``cofi``,
        ``lora_prune_distill``) or any accepted alias (``"LoRA+Prune"`` ...).
    config:
        Flat configuration dictionary (as produced by :func:`merge_config`).

    Returns
    -------
    dict with at least ``method``, ``display_name``, ``task``, ``metrics``,
    ``primary``, ``train_time_s`` and efficiency fields when available.
    """
    key = normalize_method(method)
    config = dict(config or {})
    task = normalize_task(config.get("task"))
    model_name = config.get("model_name_or_path") or config.get("model") or ""

    if not _HAS_BASELINES:
        raise RuntimeError(
            f"apt.baselines is unavailable ({_BASELINE_IMPORT_ERROR}); "
            "install the package requirements first."
        )
    if not baselines_available().get(key, False):
        raise RuntimeError(
            f"Baseline {key!r} failed to import. In-repo: {in_repo_methods()}; "
            f"external (need cloned repos): {external_methods()}."
        )

    if verbose:
        print(f"\n=== {display_name(key)} | model={model_name} | task={task} ===")

    if model is None:
        model = build_model(config, tokenizer=tokenizer)
    if tokenizer is None:
        tokenizer = build_tokenizer(config)

    loaders = None
    if train_dataloader is None or eval_dataloader is None:
        try:
            loaders = build_dataloaders(config, tokenizer)
        except Exception as exc:
            warnings.warn(f"Could not build dataloaders ({exc}); relying on supplied ones.")
            loaders = {}
    if train_dataloader is None and loaders:
        train_dataloader = loaders.get("train")
    if eval_dataloader is None and loaders:
        eval_dataloader = loaders.get("validation") or loaders.get("dev") or loaders.get("test")

    method_config, train_fn = build_method_config(key, config)
    metrics_fn = build_metrics_fn(task)
    reference = reference_metric_for(config)
    seed = int(config.get("seed", 42) or 42)

    started = time.time()
    summary: Dict[str, Any] = {}
    call_kwargs: Dict[str, Any] = {}
    if key in ("lora", "mask_tuning", "cofi", "lora_prune_distill"):
        call_kwargs["evaluate_now"] = evaluate_now

    try:
        summary = train_fn(
            config=method_config,
            model=model,
            tokenizer=tokenizer,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            compute_metrics=metrics_fn,
            reference_metric=reference,
            seed=seed,
            verbose=verbose,
            **call_kwargs,
        ) or {}
    except TypeError:
        # Some entry points do not accept ``verbose``/``seed``/``evaluate_now``.
        summary = train_fn(
            config=method_config,
            model=model,
            tokenizer=tokenizer,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            compute_metrics=metrics_fn,
            reference_metric=reference,
        ) or {}
    elapsed = time.time() - started

    metrics = summary.get("metrics") or summary.get("eval_metrics") or {}
    primary = summary.get("primary")
    if primary is None:
        primary = primary_value(task, metrics)

    out: Dict[str, Any] = {
        "method": key,
        "display_name": display_name(key),
        "model": model_name or config.get("model_type", ""),
        "task": task,
        "sparsity": config.get("target_sparsity", config.get("sparsity", 0.0)),
        "seed": seed,
        "metrics": _json_safe(metrics),
        "primary": primary,
        "train_time_s": summary.get("train_time_s", elapsed),
        "train_peak_mem_mb": summary.get("train_peak_mem_mb"),
        "tta_seconds": summary.get("tta_seconds"),
        "inf_time_ms": summary.get("inf_time_ms"),
        "inf_mem_mb": summary.get("inf_mem_mb"),
        "inf_throughput": summary.get("inf_throughput"),
        "num_parameters": summary.get("num_parameters"),
        "num_tuning_parameters": summary.get("num_tuning_parameters"),
        "history": _json_safe(summary.get("history")),
        "summary": _json_safe(summary),
    }

    if verbose:
        print(
            f"  metrics: {format_metrics(task, metrics)}  "
            f"(primary={primary if primary is None else round(float(primary), 4)})"
        )
        print(f"  wall-clock: {elapsed:.1f}s")

    return out


def run_methods(
    methods: Sequence[str],
    config: Optional[Dict[str, Any]] = None,
    *,
    verbose: bool = True,
    **kwargs,
) -> Dict[str, Dict[str, Any]]:
    """Run several baselines sequentially with the same configuration."""
    results: Dict[str, Dict[str, Any]] = {}
    for method in methods:
        try:
            results[normalize_method(method)] = run_baseline(
                method, config, verbose=verbose, **kwargs
            )
        except Exception as exc:  # keep going; one baseline may be unavailable
            warnings.warn(f"Baseline {method!r} failed: {exc}")
            results[normalize_method(method)] = {"method": normalize_method(method), "error": str(exc)}
    return results


def run_seeds(
    method: str,
    config: Optional[Dict[str, Any]] = None,
    *,
    seeds: Sequence[int] = (42, 43, 44),
    verbose: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """Run one baseline over multiple seeds and aggregate mean/std of the primary metric."""
    config = dict(config or {})
    runs: List[Dict[str, Any]] = []
    for seed in seeds:
        seed_config = dict(config)
        seed_config["seed"] = int(seed)
        out = run_baseline(method, seed_config, verbose=verbose, **kwargs)
        runs.append(out)

    primaries = [r["primary"] for r in runs if r.get("primary") is not None]
    mean = sum(primaries) / len(primaries) if primaries else None
    if len(primaries) > 1:
        var = sum((p - mean) ** 2 for p in primaries) / (len(primaries) - 1)
        std = var**0.5
    else:
        std = 0.0 if primaries else None

    return {
        "method": normalize_method(method),
        "display_name": display_name(method),
        "task": normalize_task(config.get("task")),
        "seeds": list(seeds),
        "runs": runs,
        "primary_mean": mean,
        "primary_std": std,
    }


def print_comparison(results: Dict[str, Dict[str, Any]]) -> None:
    """Print a markdown comparison table of baseline results."""
    header = f"{'Method':<22}{'Task':<12}{'Metric (primary)':>20}{'Time (s)':>12}"
    print("\n" + header)
    print("-" * len(header))
    for key, res in results.items():
        if res.get("error"):
            print(f"{display_name(key):<22}{'':<12}{'ERROR: ' + res['error'][:40]:>20}")
            continue
        primary = res.get("primary")
        prim_str = "-" if primary is None else f"{float(primary):.2f}"
        print(
            f"{str(res.get('display_name', display_name(key))):<22}"
            f"{str(res.get('task', '')):<12}{prim_str:>20}"
            f"{(res.get('train_time_s') or 0):>12.1f}"
        )


def save_results(results: Any, path: str) -> str:
    """Persist results as JSON (creating parent directories as needed)."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(results), handle, indent=2)
    return path


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train APT baselines (FT / LoRA / LoRA+Prune / Prune+Distill / LoRA+Prune+Distill).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config path/filename/alias.")
    parser.add_argument("--model", type=str, default=None, help="HF model name or path.")
    parser.add_argument("--task", type=str, default=None, help="GLUE task / squad_v2 / cnndm.")
    parser.add_argument(
        "--method",
        type=str,
        default="ft",
        help=f"Baseline key or alias. Available: {', '.join(BASELINE_METHODS)}",
    )
    parser.add_argument("--methods", type=str, default=None, help="Comma-separated list of methods.")
    parser.add_argument("--all", action="store_true", help="Run every available baseline.")
    parser.add_argument("--lora-only", action="store_true", help="Restrict to in-repo baselines (ft, lora).")

    parser.add_argument("--lr", "--learning-rate", dest="learning_rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--distill-epochs", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--max-target-length", type=int, default=None)
    parser.add_argument("--doc-stride", type=int, default=None)
    parser.add_argument("--max-query-length", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--target-sparsity", "--sparsity", dest="target_sparsity", type=float, default=None)
    parser.add_argument("--reference-metric", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds; runs each seed and reports mean/std (paper uses 42,43,44).",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None, help="Debug: truncate training.")
    parser.add_argument("--max-eval-batches", type=int, default=None, help="Debug: truncate evaluation.")
    parser.add_argument("--no-evaluate", action="store_true", help="Skip post-training evaluation.")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--self-test", action="store_true", help="Run dependency-light self tests and exit.")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Collect non-None CLI values into a config-override dictionary."""
    keys = (
        "config",
        "model",
        "task",
        "learning_rate",
        "batch_size",
        "epochs",
        "distill_epochs",
        "weight_decay",
        "warmup_ratio",
        "max_seq_length",
        "max_target_length",
        "doc_stride",
        "max_query_length",
        "lora_rank",
        "lora_alpha",
        "target_sparsity",
        "reference_metric",
        "seed",
        "device",
        "output_dir",
        "num_workers",
        "max_train_batches",
        "max_eval_batches",
    )
    overrides: Dict[str, Any] = {}
    for key in keys:
        value = getattr(args, key, None)
        if value is not None and key not in ("config", "method"):
            overrides[key] = value
    if getattr(args, "fp16", False):
        overrides["fp16"] = True
    if getattr(args, "bf16", False):
        overrides["bf16"] = True
    return overrides


def methods_from_args(args: argparse.Namespace) -> List[str]:
    """Resolve the list of methods requested on the CLI."""
    if args.methods:
        raw = [part.strip() for part in str(args.methods).split(",") if part.strip()]
        return [normalize_method(part) for part in raw]
    if args.all:
        if args.lora_only:
            return ["ft", "lora"]
        return [m for m in BASELINE_METHODS if baselines_available().get(m, False)] or ["ft", "lora"]
    return [normalize_method(args.method)]


def main(config: Optional[Dict[str, Any]] = None, argv: Optional[List[str]] = None) -> int:
    """CLI / programmatic entry point used by ``main.py``."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.self_test:
        ok = _self_test()
        print("self-test:", "OK" if ok else "FAILED")
        return 0 if ok else 1

    overrides = cli_overrides(args)
    explicit: Dict[str, Any] = {}
    if config:
        explicit.update(config)
    explicit.update(overrides)

    merged = merge_config(
        path=args.config or (config or {}).get("_config_path"),
        explicit=explicit,
        model=args.model or (config or {}).get("model_name_or_path"),
        task=args.task or (config or {}).get("task"),
    )

    methods = methods_from_args(args)
    verbose = not args.quiet

    if verbose:
        print("Baselines:", ", ".join(methods))
        print("Config group:", merged.get("table6_group"), "| sparsity:", merged.get("target_sparsity"))

    seeds: Optional[List[int]] = None
    if args.seeds:
        try:
            seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
        except ValueError:
            seeds = None

    started = time.time()
    try:
        if seeds and len(seeds) > 1:
            all_results: Dict[str, Any] = {}
            for method in methods:
                all_results[method] = run_seeds(method, merged, seeds=seeds, verbose=verbose)
            results: Any = all_results
        else:
            results = run_methods(methods, merged, verbose=verbose)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if verbose:
        print_comparison(results if not seeds or len(seeds) <= 1 else results)

    output_dir = merged.get("output_dir") or os.path.join("outputs", "baselines")
    path = os.path.join(
        output_dir,
        f"baseline_{merged.get('model_type', 'model')}_{merged.get('task', 'task')}.json",
    )
    try:
        save_results(results, path)
        if verbose:
            print(f"\nResults saved to {path}")
    except Exception as exc:  # pragma: no cover
        warnings.warn(f"Could not save results: {exc}")

    if verbose:
        print(f"Total wall-clock: {time.time() - started:.1f}s")
    return 0


# --------------------------------------------------------------------------------------
# Dependency-light self test
# --------------------------------------------------------------------------------------


def _self_test() -> bool:
    """Validate config resolution, aliasing and dispatch tables without heavy deps.

    The tests that need torch/transformers are executed only when those packages are
    importable so this function is safe in a bare environment.
    """
    ok = True

    def check(condition: bool, message: str) -> None:
        nonlocal ok
        if not condition:
            ok = False
            print(f"  FAIL: {message}")

    # --- task normalisation ---------------------------------------------------------
    check(normalize_task("SST-2") == "sst2", "SST-2 -> sst2")
    check(normalize_task("squad2") == "squad_v2", "squad2 -> squad_v2")
    check(normalize_task("cnn_dailymail") == "cnndm", "cnn_dailymail -> cnndm")

    # --- method normalisation / aliases --------------------------------------------
    check(normalize_method("FT") == "ft", "FT -> ft")
    check(normalize_method("LoRA") == "lora", "LoRA -> lora")
    check(normalize_method("LoRA+Prune") == "mask_tuning", "LoRA+Prune -> mask_tuning")
    check(normalize_method("Prune+Distill") == "cofi", "Prune+Distill -> cofi")
    check(
        normalize_method("LoRA+Prune+Distill") == "lora_prune_distill",
        "LoRA+Prune+Distill -> lora_prune_distill",
    )
    try:
        normalize_method("definitely-not-a-method")
        check(False, "unknown method should raise KeyError")
    except KeyError:
        pass

    # --- display names ---------------------------------------------------------------
    check(display_name("mask_tuning") == "LoRA+Prune", "display name mask_tuning")
    check(display_name("cofi") == "Prune+Distill", "display name cofi")

    # --- Table 6 groups --------------------------------------------------------------
    check(table6_group_for("roberta", "sst2") == "glue-big", "sst2 -> glue-big")
    check(table6_group_for("roberta", "cola") == "glue-small", "cola -> glue-small")
    check(table6_group_for("roberta", "squad_v2") == "squad", "squad -> squad column")
    check(table6_group_for("t5", "cnndm") == "cnndm", "cnndm -> cnndm column")

    # --- config merging --------------------------------------------------------------
    merged = merge_config(explicit={"task": "sst2", "model_name_or_path": "roberta-base"})
    check(merged.get("model_type") == "roberta", "model_type inferred as roberta")
    check(abs(float(merged.get("learning_rate", 0)) - 2.0e-4) < 1e-12, "Table 6 LR for glue-big")
    check(int(merged.get("batch_size", 0)) == 32, "Table 6 batch size for glue-big")
    check(int(merged.get("epochs", 0)) == 40 and int(merged.get("distill_epochs", 0)) == 20, "Table 6 epochs")

    merged_squad = merge_config(explicit={"task": "squad_v2", "model_name_or_path": "roberta-base"})
    check(int(merged_squad.get("max_seq_length", 0)) == 384, "SQuAD max_seq_length")
    check(int(merged_squad.get("doc_stride", 0)) == 128, "SQuAD doc_stride")

    merged_cnn = merge_config(explicit={"task": "cnndm", "model_name_or_path": "t5-base"})
    check(merged_cnn.get("model_type") == "t5", "model_type inferred as t5")
    check(abs(float(merged_cnn.get("learning_rate", 0)) - 1.0e-4) < 1e-12, "CNN/DM LR")
    check(int(merged_cnn.get("batch_size", 0)) == 16, "CNN/DM batch size")

    # --- CLI overrides win -----------------------------------------------------------
    args = parse_args(["--method", "lora", "--task", "mnli", "--lr", "5e-5", "--seed", "43"])
    check(args.method == "lora" and args.seed == 43, "CLI parsing")
    ov = cli_overrides(args)
    check(abs(ov.get("learning_rate", 0) - 5e-5) < 1e-15, "CLI LR override collected")
    check("method" not in ov, "method excluded from config overrides")

    # --- json safety -----------------------------------------------------------------
    safe = _json_safe({"metrics": {"accuracy": 94.5}, "trainer": object(), "history": [1, 2]})
    check("trainer" not in safe and safe["metrics"]["accuracy"] == 94.5, "json-safe summary")

    # --- metric helpers --------------------------------------------------------------
    check(primary_value("sst2", {"accuracy": 94.5}) in (94.5, None), "primary metric from dict")
    check(format_metrics("sst2", {}) == "-", "empty metrics formatting")

    # --- optional heavy path: method config construction -----------------------------
    torch = _torch()
    if torch is not None and _HAS_BASELINES:
        for method in ("ft", "lora"):
            if not baselines_available().get(method, False):
                continue
            try:
                cfg, fn = build_method_config(method, merged)
                check(cfg is not None and callable(fn), f"build_method_config({method})")
                check(
                    abs(float(getattr(cfg, "learning_rate", 0.0)) - 2.0e-4) < 1e-12,
                    f"{method} config LR from Table 6",
                )
                check(int(getattr(cfg, "batch_size", 0)) == 32, f"{method} config batch size")
            except Exception as exc:
                check(False, f"build_method_config({method}) raised {exc}")

    return ok


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(argv=sys.argv[1:]))
