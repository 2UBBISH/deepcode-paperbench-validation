#!/usr/bin/env python
"""APT training entry point (Algorithm 1) for the reproduction codebase.

Implements the command-line driver for the *small-model* APT experiments:

    resolve config  ->  build tokenizer/model  ->  build task dataloaders
    ->  TrainConfig  ->  APTTrainer.fit()  (Algorithm 1, two stages)
    ->  merge/prune for inference  ->  evaluate  ->  save JSON summary

Paper references (verbatim):
  * Appendix A:  "we first prune and train the LM with the self-distillation
    objective, and then fine-tune the pruned LM to recover its end-task
    performance", cubic schedule ``gamma_t = gamma_T + (1 - gamma_T)
    (1 - t/T)^3``, masks decreased by ``alpha < 1`` instead of zeroed,
    EMA salience, adapter ranks initialised to 8 and linearly increased for
    salient layers, scaling factor 2, Table 6 hyper-parameters.
  * Appendix C / Algorithm 1: sort blocks by salience density, binary search the
    top-i blocks under the parameter constraint, ``alpha = 0.01`` gradual decay.
  * Section 6: the optimizer is reset every time the parameter size changes.

The script is intentionally dependency-light at import time (torch /
transformers / yaml are imported lazily) so that ``--help`` and unit tests work
in a bare environment, matching the conventions of the rest of the repository.

CLI examples
------------
    python scripts/train_apt.py --config apt/configs/roberta_sst2.yaml
    python scripts/train_apt.py --model roberta-base --task sst2 \
        --target_sparsity 0.6 --epochs 40 --distill_epochs 20
    python scripts/train_apt.py --config apt/configs/t5_cnndm.yaml --max-steps 20
    python scripts/train_apt.py --config apt/configs/roberta_sst2.yaml --ablation no_ds

Programmatic use (as required by ``main.py``)::

    from scripts.train_apt import main
    main({"model_name_or_path": "roberta-base", "task": "sst2"})
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from typing import Any, Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# optional heavy dependencies (lazy / guarded, repository convention)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - environment dependent
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


def _torch():
    """Import torch lazily; raise a helpful error when it is missing."""
    global torch
    if torch is None:
        try:  # pragma: no cover
            import torch as _t  # noqa: WPS433

            torch = _t
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "PyTorch is required to run APT training (pip install torch)"
            ) from exc
    return torch


# ---------------------------------------------------------------------------
# apt package imports (all guarded so the module always imports)
# ---------------------------------------------------------------------------
try:  # pragma: no cover
    from apt.training import (  # type: ignore
        APTTrainer,
        TrainConfig,
        set_seed as _set_seed,
        train_apt as _train_apt,
    )

    _HAS_TRAINING = True
except Exception as _exc:  # pragma: no cover
    APTTrainer = None  # type: ignore[assignment]
    TrainConfig = None  # type: ignore[assignment]
    _train_apt = None  # type: ignore[assignment]
    _set_seed = None  # type: ignore[assignment]
    _HAS_TRAINING = False
    _TRAINING_IMPORT_ERROR = _exc
else:
    _TRAINING_IMPORT_ERROR = None

try:  # pragma: no cover
    from apt.training import build_trainer as _build_trainer  # type: ignore

    _HAS_BUILD_TRAINER = True
except Exception:  # pragma: no cover
    _build_trainer = None  # type: ignore[assignment]
    _HAS_BUILD_TRAINER = False

try:  # pragma: no cover
    from apt.training import measure_inference_throughput as _measure_inf  # type: ignore

    _HAS_MEASURE_INF = True
except Exception:  # pragma: no cover
    _measure_inf = None  # type: ignore[assignment]
    _HAS_MEASURE_INF = False

try:  # pragma: no cover
    from apt.data import make_dataloaders as _make_dataloaders  # type: ignore

    _HAS_DATA = True
except Exception:  # pragma: no cover
    _make_dataloaders = None  # type: ignore[assignment]
    _HAS_DATA = False

try:  # pragma: no cover
    from apt.eval.metrics import (  # type: ignore
        compute_metrics as _compute_metrics,
        metric_for_display as _metric_for_display,
        primary_metric as _primary_metric,
    )

    _HAS_METRICS = True
except Exception:  # pragma: no cover
    _compute_metrics = None  # type: ignore[assignment]
    _metric_for_display = None  # type: ignore[assignment]
    _primary_metric = None  # type: ignore[assignment]
    _HAS_METRICS = False

try:  # pragma: no cover
    from apt.model_wrapper import (  # type: ignore
        apt_shape as _apt_shape,
        block_metadata as _block_metadata,
        is_wrapped as _is_wrapped,
        wrap_model as _wrap_model,
    )

    _HAS_WRAPPER = True
except Exception:  # pragma: no cover
    _apt_shape = None  # type: ignore[assignment]
    _block_metadata = None  # type: ignore[assignment]
    _is_wrapped = None  # type: ignore[assignment]
    _wrap_model = None  # type: ignore[assignment]
    _HAS_WRAPPER = False

try:  # pragma: no cover
    from apt.merge import merge_and_prune as _merge_and_prune  # type: ignore

    _HAS_MERGE = True
except Exception:  # pragma: no cover
    _merge_and_prune = None  # type: ignore[assignment]
    _HAS_MERGE = False

try:  # pragma: no cover
    from apt.eval.run_eval import evaluate_model as _evaluate_model  # type: ignore

    _HAS_RUN_EVAL = True
except Exception:  # pragma: no cover
    _evaluate_model = None  # type: ignore[assignment]
    _HAS_RUN_EVAL = False


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
CONFIG_DIR = os.path.join("apt", "configs")

#: default YAML merged underneath every task-specific config
DEFAULT_CONFIG_FILE = os.path.join(CONFIG_DIR, "default.yaml")

CONFIG_ALIASES: Dict[str, str] = {
    # (model, task) aliases -> config file
    "roberta:sst2": "roberta_sst2.yaml",
    "roberta:mnli": "roberta_mnli.yaml",
    "roberta:squad_v2": "squad.yaml",
    "roberta:squad": "squad.yaml",
    "t5:cnndm": "t5_cnndm.yaml",
    "t5:cnn_dailymail": "t5_cnndm.yaml",
    # bare config names
    "default": "default.yaml",
    "roberta_sst2": "roberta_sst2.yaml",
    "roberta_mnli": "roberta_mnli.yaml",
    "squad": "squad.yaml",
    "t5_cnndm": "t5_cnndm.yaml",
}

#: Table 6 hyper-parameters (Appendix A) used to fill defaults
TABLE6_GROUPS: Dict[str, Dict[str, float]] = {
    "glue-small": {"learning_rate": 2e-4, "batch_size": 32, "epochs": 40, "distill_epochs": 20},
    "glue-big": {"learning_rate": 2e-4, "batch_size": 32, "epochs": 40, "distill_epochs": 20},
    "squad": {"learning_rate": 2e-4, "batch_size": 32, "epochs": 40, "distill_epochs": 20},
    "cnndm": {"learning_rate": 1e-4, "batch_size": 16, "epochs": 16, "distill_epochs": 6},
}

GLUE_BIG_TASKS = ("mnli", "sst2", "qnli", "qqp")
GLUE_SMALL_TASKS = ("mrpc", "cola", "rte", "stsb")
SQUAD_TASKS = ("squad", "squad_v2", "squadv2", "squad2")
SEQ2SEQ_TASKS = ("cnndm", "cnn_dailymail", "xsum")

#: Table 4 ablation switches (remove exactly one component at a time, Sec. 5.6)
ABLATIONS: Dict[str, Dict[str, Any]] = {
    "none": {},
    # w/o A_P : no adaptive (gradual, re-selected) pruning -> instant masks
    "no_ap": {"mask_alpha": 1.0, "adjustment_interval": 10**9, "ablation": "no_adaptive_pruning"},
    # w/o salience : drop the outlier (kurtosis) aware salience term
    "no_salience": {"use_kurtosis": False, "ablation": "no_salience"},
    # w/o A_T : no adaptive tuning rank growth
    "no_at": {"use_adaptive_tuning": False, "ablation": "no_adaptive_tuning"},
    # w/o D_S : no self-knowledge distillation
    "no_ds": {"use_distillation": False, "ablation": "no_distillation"},
}

ABLATION_ALIASES: Dict[str, str] = {
    "none": "none",
    "": "none",
    "all": "none",
    "ap": "no_ap",
    "a_p": "no_ap",
    "no_ap": "no_ap",
    "no_adaptive_pruning": "no_ap",
    "wo_ap": "no_ap",
    "w/o a_p": "no_ap",
    "salience": "no_salience",
    "no_salience": "no_salience",
    "no_kurtosis": "no_salience",
    "wo_salience": "no_salience",
    "w/o salience": "no_salience",
    "at": "no_at",
    "a_t": "no_at",
    "no_at": "no_at",
    "no_adaptive_tuning": "no_at",
    "wo_at": "no_at",
    "w/o a_t": "no_at",
    "ds": "no_ds",
    "d_s": "no_ds",
    "no_ds": "no_ds",
    "no_distill": "no_ds",
    "no_distillation": "no_ds",
    "wo_ds": "no_ds",
    "w/o d_s": "no_ds",
}

MODEL_INPUT_KEYS = (
    "input_ids",
    "attention_mask",
    "token_type_ids",
    "position_ids",
    "decoder_input_ids",
    "decoder_attention_mask",
)


# ---------------------------------------------------------------------------
# helpers: tasks / configs
# ---------------------------------------------------------------------------
def normalize_task(task: Optional[str]) -> str:
    """Lower-case and canonicalise a task name."""
    if not task:
        return "sst2"
    name = str(task).strip().lower().replace("-", "_")
    if name in ("sst_2", "sst2", "sst"):
        return "sst2"
    if name in ("mnli", "mnli_matched", "mnli_mismatched", "mnli_mm"):
        return "mnli"
    if name in ("squad2", "squadv2", "squad_v2", "squad"):
        return "squad_v2" if name != "squad" else "squad"
    if name in ("cnn_dailymail", "cnndm", "cnn"):
        return "cnndm"
    return name


def table6_group_for(model_type: Optional[str], task: Optional[str]) -> str:
    """Pick the Table 6 column for a (model, task) pair."""
    t = normalize_task(task)
    if t in ("cnndm", "xsum"):
        return "cnndm"
    if t in ("squad", "squad_v2"):
        return "squad"
    if t in GLUE_SMALL_TASKS:
        return "glue-small"
    return "glue-big"


def resolve_ablation(name: Optional[str]) -> Dict[str, Any]:
    """Map an ablation label onto the Table 4 switch dictionary."""
    if not name:
        return {}
    key = ABLATION_ALIASES.get(str(name).strip().lower(), None)
    if key is None:
        raise KeyError(
            "unknown ablation %r; known: %s" % (name, sorted(set(ABLATION_ALIASES)))
        )
    return dict(ABLATIONS[key])


def resolve_config_path(
    path: Optional[str] = None,
    model: Optional[str] = None,
    task: Optional[str] = None,
) -> Optional[str]:
    """Resolve a config path / filename / alias / (model, task) pair to a file."""
    candidates: List[str] = []
    if path:
        candidates += [
            path,
            os.path.join(CONFIG_DIR, path),
            os.path.join(CONFIG_DIR, os.path.basename(path)),
        ]
        key = os.path.basename(path)
        if key.endswith(".yaml"):
            key = key[: -len(".yaml")]
        if key in CONFIG_ALIASES:
            candidates.append(os.path.join(CONFIG_DIR, CONFIG_ALIASES[key]))
    if model and task:
        key = "%s:%s" % (str(model).split("/")[-1].replace("-base", ""), normalize_task(task))
        if key in CONFIG_ALIASES:
            candidates.append(os.path.join(CONFIG_DIR, CONFIG_ALIASES[key]))
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def load_yaml(path: Optional[str]) -> Dict[str, Any]:
    """Load a YAML file into a flat dict (empty when unavailable)."""
    if not path or not os.path.isfile(path):
        return {}
    if yaml is None:
        warnings.warn("pyyaml is not installed; ignoring config file %s" % path)
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        return {}
    return dict(data)


def merge_config(path: Optional[str] = None, explicit: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Merge ``default.yaml`` <- task config <- explicit overrides."""
    merged: Dict[str, Any] = {}
    merged.update(load_yaml(DEFAULT_CONFIG_FILE))
    merged.update(load_yaml(path))
    if explicit:
        merged.update({k: v for k, v in explicit.items() if v is not None})
    # fill Table 6 numbers when absent
    group = table6_group_for(merged.get("model_type"), merged.get("task"))
    defaults = TABLE6_GROUPS.get(str(merged.get("table6_group") or group), TABLE6_GROUPS["glue-big"])
    for key, value in defaults.items():
        merged.setdefault(key, value)
    merged.setdefault("table6_group", group)
    merged.setdefault("target_sparsity", 0.60)
    merged.setdefault("initial_rank", 8)
    merged.setdefault("scaling", 2.0)
    merged.setdefault("mask_alpha", 0.01)
    merged.setdefault("ema_beta", 0.85)
    merged.setdefault("tau", 4)
    merged.setdefault("weight_decay", 0.01)
    merged.setdefault("warmup_ratio", 0.06)
    merged.setdefault("seed", 42)
    merged.setdefault("device", "cuda")
    merged.setdefault("max_seq_length", 128)
    merged.setdefault("max_target_length", 128)
    return merged


def split_known_keys(config: Dict[str, Any]) -> "tuple[Dict[str, Any], Dict[str, Any]]":
    """Split a config dict into TrainConfig fields and unknown extras."""
    if TrainConfig is None:
        return dict(config), {}
    try:
        import dataclasses

        known = {f.name for f in dataclasses.fields(TrainConfig)}
    except Exception:  # pragma: no cover
        return dict(config), {}
    main = {k: v for k, v in config.items() if k in known}
    extra = {k: v for k, v in config.items() if k not in known}
    return main, extra


def apply_ablation(config: Dict[str, Any], ablation: Optional[str]) -> Dict[str, Any]:
    """Apply a Table 4 ablation switch to the config dict (in place copy)."""
    out = dict(config)
    for key, value in resolve_ablation(ablation).items():
        out[key] = value
    return out


# ---------------------------------------------------------------------------
# model / tokenizer / data construction
# ---------------------------------------------------------------------------
def build_tokenizer(config: Dict[str, Any]):
    """Load the HuggingFace tokenizer for the configured checkpoint."""
    from transformers import AutoTokenizer  # lazy

    name = config.get("model_name_or_path") or "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_model(config: Dict[str, Any], tokenizer=None):
    """Build the task-appropriate HuggingFace model (Sec. 5.1 tasks)."""
    from transformers import (  # lazy
        AutoConfig,
        AutoModelForQuestionAnswering,
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
    )

    task = normalize_task(config.get("task"))
    name = config.get("model_name_or_path") or "roberta-base"
    hf_config = AutoConfig.from_pretrained(name)

    if task in SEQ2SEQ_TASKS:
        return AutoModelForSeq2SeqLM.from_pretrained(name, config=hf_config)
    if task in SQUAD_TASKS:
        return AutoModelForQuestionAnswering.from_pretrained(name, config=hf_config)

    # GLUE: regression task (STSB) vs. classification
    is_regression = task == "stsb"
    num_labels = 1 if is_regression else _num_labels_for(task)
    return AutoModelForSequenceClassification.from_pretrained(
        name,
        config=hf_config,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )


def _num_labels_for(task: str) -> int:
    if task == "mnli":
        return 3
    if task in ("stsb",):
        return 1
    return 2


def build_dataloaders(config: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    """Build train/validation dataloaders for the configured task."""
    if not _HAS_DATA:
        raise RuntimeError("apt.data is unavailable; cannot build dataloaders")
    task = normalize_task(config.get("task"))
    model_type = str(config.get("model_type") or "").lower()
    if not model_type:
        model_type = "t5" if "t5" in str(config.get("model_name_or_path", "")).lower() else "encoder"
    kwargs: Dict[str, Any] = {
        "batch_size": config.get("batch_size"),
        "max_seq_length": config.get("max_seq_length", 128),
        "num_workers": config.get("num_workers", 0),
        "seed": config.get("seed", 42),
        "model_type": model_type,
    }
    for key in ("max_target_length", "doc_stride", "max_query_length", "dynamic_padding", "data_dir", "cache_dir"):
        if config.get(key) is not None:
            kwargs[key] = config[key]
    try:
        return _make_dataloaders(task, tokenizer, **kwargs)
    except TypeError:
        # tolerate a narrower dispatcher signature
        return _make_dataloaders(task, tokenizer, batch_size=config.get("batch_size"))


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def build_metrics_fn(task: str) -> Optional[Callable[..., Dict[str, float]]]:
    """Wrap ``apt.eval.metrics.compute_metrics`` into a 2-argument callable."""
    if not _HAS_METRICS or _compute_metrics is None:
        return None

    def _fn(*args: Any, **kwargs: Any) -> Dict[str, float]:
        # (predictions, references)
        if len(args) >= 2:
            return dict(_compute_metrics(task, args[0], args[1], **kwargs))
        if len(args) == 1:
            payload = args[0]
            if isinstance(payload, dict):
                if "predictions" in payload and "references" in payload:
                    return dict(
                        _compute_metrics(task, payload["predictions"], payload["references"], **kwargs)
                    )
                # already aggregated metrics
                numeric = {k: float(v) for k, v in payload.items() if isinstance(v, (int, float))}
                if numeric:
                    return numeric
        raise ValueError("compute_metrics expects (predictions, references)")

    return _fn


def primary_value(task: str, metrics: Dict[str, float]) -> Optional[float]:
    """Primary paper metric for a task, from a metrics dict."""
    if not metrics:
        return None
    if _HAS_METRICS and _primary_metric is not None:
        try:
            value = _primary_metric(task, metrics)
            if value is not None:
                return float(value)
        except Exception:
            pass
    for key in ("accuracy", "f1", "exact", "rougeL", "rouge_l", "matthews_correlation", "spearmanr"):
        if key in metrics:
            return float(metrics[key])
    for value in metrics.values():
        if isinstance(value, (int, float)):
            return float(value)
    return None


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def wrap_with_adapters(model, config: Dict[str, Any], verbose: bool = False):
    """Inject APT adapters/masks into an HF model (Sec. 4.1)."""
    if not _HAS_WRAPPER or _wrap_model is None:
        return model
    try:
        if _is_wrapped is not None and _is_wrapped(model):
            return model
    except Exception:
        pass
    model_type = str(config.get("model_type") or "").lower() or None
    kwargs = {
        "rank": int(config.get("initial_rank", 8) or 8),
        "scaling": float(config.get("scaling", 2.0) or 2.0),
        "model_type": model_type,
    }
    if config.get("wrap_ffn") is not None:
        kwargs["wrap_ffn"] = bool(config["wrap_ffn"])
    if config.get("cache_for_salience") is not None:
        kwargs["cache_for_salience"] = bool(config["cache_for_salience"])
    try:
        return _wrap_model(model, verbose=verbose, **kwargs)
    except TypeError:
        return _wrap_model(model, **kwargs)


def run_training(
    config: Dict[str, Any],
    model=None,
    tokenizer=None,
    train_dataloader=None,
    eval_dataloader=None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute the full APT pipeline for one (model, task, sparsity) setting."""
    started = time.time()
    task = normalize_task(config.get("task"))

    if _set_seed is not None:
        try:
            _set_seed(int(config.get("seed", 42) or 42))
        except Exception:
            pass

    if tokenizer is None:
        tokenizer = build_tokenizer(config)
    if model is None:
        model = build_model(config, tokenizer)
    model = wrap_with_adapters(model, config, verbose=verbose)

    if train_dataloader is None or eval_dataloader is None:
        loaders = build_dataloaders(config, tokenizer)
        train_dataloader = train_dataloader or loaders.get("train")
        eval_dataloader = eval_dataloader or (
            loaders.get("validation") or loaders.get("dev") or loaders.get("test")
        )

    metrics_fn = build_metrics_fn(task)
    if not _HAS_TRAINING:
        raise RuntimeError(
            "apt.training is unavailable (%s); cannot run APT training" % _TRAINING_IMPORT_ERROR
        )

    main_keys, extra = split_known_keys(config)
    train_config = TrainConfig.from_dict(main_keys) if hasattr(TrainConfig, "from_dict") else TrainConfig(**main_keys)

    # Distillation / schedule switches that TrainConfig may not expose are routed
    # through as extra trainer kwargs (Sec. 4.3 adaptive tuning, Sec. 5.6 ablations).
    trainer_kwargs: Dict[str, Any] = {}
    for key in ("use_adaptive_tuning", "adjustment_interval", "ablation", "reference_model"):
        if key in extra and extra[key] is not None:
            trainer_kwargs[key] = extra[key]

    if _HAS_BUILD_TRAINER and _build_trainer is not None:
        try:
            trainer = _build_trainer(
                train_config,
                model=model,
                tokenizer=tokenizer,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                compute_metrics=metrics_fn,
                **trainer_kwargs,
            )
        except TypeError:
            trainer = _build_trainer(
                train_config, model=model, tokenizer=tokenizer,
                train_dataloader=train_dataloader, eval_dataloader=eval_dataloader,
            )
    else:  # pragma: no cover - fallback path
        trainer = APTTrainer(
            config=train_config,
            model=model,
            tokenizer=tokenizer,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            compute_metrics=metrics_fn,
            **trainer_kwargs,
        )

    max_steps = config.get("max_steps") or None
    fit_kwargs: Dict[str, Any] = {}
    if max_steps:
        fit_kwargs["max_steps"] = int(max_steps)
    try:
        train_out = trainer.fit(**fit_kwargs)
    except TypeError:
        train_out = trainer.fit()

    summary: Dict[str, Any] = dict(train_out) if isinstance(train_out, dict) else {"result": train_out}
    summary.setdefault("method", "APT")
    summary.setdefault("model", config.get("model_name_or_path"))
    summary.setdefault("task", task)
    summary.setdefault("sparsity", float(config.get("target_sparsity", 0.0) or 0.0))

    # ---- evaluation of the (merged, physically pruned) model -------------
    eval_metrics: Dict[str, float] = {}
    if isinstance(summary.get("metrics"), dict):
        eval_metrics = {k: float(v) for k, v in summary["metrics"].items() if isinstance(v, (int, float))}
    elif hasattr(trainer, "evaluate"):
        try:
            result = trainer.evaluate(eval_dataloader) if eval_dataloader is not None else trainer.evaluate()
            if isinstance(result, dict):
                eval_metrics = {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}
        except Exception as exc:  # pragma: no cover
            warnings.warn("evaluation failed: %s" % exc)

    merged_model = None
    if hasattr(trainer, "export_model"):
        try:
            threshold = float(config.get("merge_threshold", 0.5) or 0.5)
            merged_model = trainer.export_model(threshold=threshold)
        except Exception as exc:  # pragma: no cover
            warnings.warn("model export failed: %s" % exc)
    elif _HAS_MERGE and _merge_and_prune is not None:
        try:
            merged_model = _merge_and_prune(model, threshold=float(config.get("merge_threshold", 0.5) or 0.5))
        except Exception as exc:  # pragma: no cover
            warnings.warn("merge_and_prune failed: %s" % exc)

    summary["metrics"] = eval_metrics
    summary["primary"] = primary_value(task, eval_metrics)
    summary["train_time_s"] = float(summary.get("train_time_s") or (time.time() - started))
    summary["elapsed_s"] = time.time() - started
    summary["num_parameters"] = _parameter_count(merged_model if merged_model is not None else model)
    summary["sparsity_realised"] = _realised_sparsity(trainer)

    if _metric_for_display is not None and eval_metrics:
        try:
            summary["display"] = _metric_for_display(task, eval_metrics)
        except Exception:
            pass

    if config.get("measure_inference") and merged_model is not None and _HAS_MEASURE_INF:
        try:
            summary["inference"] = _measure_inf(
                merged_model,
                batch_size=int(config.get("inference_batch_size", 128) or 128),
                seq_length=int(config.get("max_seq_length", 128) or 128),
            )
        except Exception as exc:  # pragma: no cover
            warnings.warn("inference measurement failed: %s" % exc)

    # ---- persistence ------------------------------------------------------
    output_dir = config.get("output_dir") or "outputs/apt"
    try:
        os.makedirs(output_dir, exist_ok=True)
        tag = "%s_%s_s%s" % (
            str(config.get("model_name_or_path", "model")).split("/")[-1],
            task,
            str(config.get("target_sparsity", "0")).replace(".", ""),
        )
        result_path = os.path.join(output_dir, "apt_%s.json" % tag)
        with open(result_path, "w", encoding="utf-8") as handle:
            json.dump(_json_safe(summary), handle, indent=2)
        summary["result_path"] = result_path
    except Exception as exc:  # pragma: no cover
        warnings.warn("could not save results: %s" % exc)

    summary["trainer"] = trainer
    summary["model"] = merged_model if merged_model is not None else model
    if verbose:
        _print_summary(summary)
    return summary


def _parameter_count(model) -> Optional[int]:
    if model is None:
        return None
    try:
        return int(sum(p.numel() for p in model.parameters()))
    except Exception:  # pragma: no cover
        return None


def _realised_sparsity(trainer) -> Optional[float]:
    for attr in ("sparsity", "param_count"):
        if hasattr(trainer, attr):
            try:
                value = getattr(trainer, attr)
                value = value() if callable(value) else value
                if attr == "sparsity":
                    return float(value)
                if attr == "param_count":
                    continue
            except Exception:
                continue
    for attr in ("mask_manager", "masks"):
        manager = getattr(trainer, attr, None)
        if manager is not None and hasattr(manager, "sparsity"):
            try:
                return float(manager.sparsity())
            except Exception:
                continue
    return None


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items() if k not in ("trainer", "model")}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _print_summary(summary: Dict[str, Any]) -> None:
    print("-" * 72)
    print("APT  model=%s  task=%s  sparsity=%s" % (summary.get("model_name") or summary.get("model"), summary.get("task"), summary.get("sparsity")))
    if summary.get("metrics"):
        print("metrics       : %s" % summary["metrics"])
    if summary.get("display"):
        print("primary       : %s" % summary["display"])
    print("train_time_s  : %.1f" % float(summary.get("train_time_s") or 0.0))
    if summary.get("num_parameters"):
        print("parameters    : %d" % summary["num_parameters"])
    if summary.get("result_path"):
        print("saved         : %s" % summary["result_path"])
    print("-" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train_apt",
        description="Train APT (adaptive pruning + tuning) on small LMs (Table 2/4/7/8).",
    )
    parser.add_argument("--config", default=None, help="YAML config path or alias")
    parser.add_argument("--model", "--model_name_or_path", dest="model", default=None)
    parser.add_argument("--model_type", default=None, help="roberta | bert | t5 | ...")
    parser.add_argument("--task", default=None, help="sst2 | mnli | squad_v2 | cnndm | ...")
    parser.add_argument("--output_dir", default=None)

    # Table 6 hyper-parameters
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--distill_epochs", type=float, default=None)

    # Algorithm 1 / APT
    parser.add_argument("--target_sparsity", type=float, default=None)
    parser.add_argument("--initial_rank", type=int, default=None)
    parser.add_argument("--scaling", type=float, default=None)
    parser.add_argument("--mask_alpha", type=float, default=None)
    parser.add_argument("--ema_beta", type=float, default=None)
    parser.add_argument("--tau", type=int, default=None)
    parser.add_argument("--tuning_budget_initial", type=float, default=None)
    parser.add_argument("--tuning_budget_final", type=float, default=None)
    parser.add_argument("--pred_distill_weight", type=float, default=None)
    parser.add_argument("--layer_distill_weight", type=float, default=None)
    parser.add_argument("--top_fraction", type=float, default=None)
    parser.add_argument("--use_distillation", dest="use_distillation", action="store_true", default=None)
    parser.add_argument("--no_distillation", dest="use_distillation", action="store_false")
    parser.add_argument("--use_kurtosis", dest="use_kurtosis", action="store_true", default=None)
    parser.add_argument("--no_kurtosis", dest="use_kurtosis", action="store_false")
    parser.add_argument("--ablation", default=None, help="none | no_ap | no_salience | no_at | no_ds")

    # data / runtime
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--max_target_length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None, help="debug: cap global steps")
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument("--measure_inference", action="store_true", default=None)
    parser.add_argument("--dry-run", action="store_true", help="resolve config and exit")
    parser.add_argument("--quiet", action="store_true")
    return parser


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv if argv is not None else [])


def cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Collect non-None CLI values into a config override dict."""
    keys = (
        "model_name_or_path", "model_type", "task", "output_dir",
        "learning_rate", "batch_size", "epochs", "distill_epochs",
        "target_sparsity", "initial_rank", "scaling", "mask_alpha", "ema_beta", "tau",
        "tuning_budget_initial", "tuning_budget_final", "pred_distill_weight",
        "layer_distill_weight", "top_fraction", "use_distillation", "use_kurtosis",
        "max_seq_length", "max_target_length", "seed", "device", "num_workers",
        "max_steps", "max_train_batches", "max_eval_batches", "measure_inference",
    )
    overrides: Dict[str, Any] = {}
    for key in keys:
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    if args.model:
        overrides["model_name_or_path"] = args.model
    if args.ablation:
        overrides["ablation"] = args.ablation
    return overrides


def main(config: Optional[Dict[str, Any]] = None, argv: Optional[List[str]] = None) -> int:
    """Entry point used by ``main.py`` and by the command line."""
    args = parse_args(argv)

    # priority: default.yaml  <  config file  <  programmatic dict  <  CLI
    explicit: Dict[str, Any] = {}
    if config:
        explicit.update(config)
    explicit.update(cli_overrides(args))

    path = resolve_config_path(args.config, explicit.get("model_name_or_path"), explicit.get("task"))
    if args.config and path is None:
        print("warning: config %r not found; falling back to defaults" % args.config, file=sys.stderr)

    merged = merge_config(path, explicit)
    merged = apply_ablation(merged, merged.get("ablation"))

    if args.dry_run:
        print(json.dumps(_json_safe(merged), indent=2))
        return 0

    try:
        summary = run_training(merged, verbose=not args.quiet)
    except Exception as exc:  # pragma: no cover - surfaced to the CLI user
        print("APT training failed: %s" % exc, file=sys.stderr)
        return 1

    if summary.get("primary") is not None:
        print("APT %s %s -> primary=%.2f" % (merged.get("model_name_or_path"), merged.get("task"), summary["primary"]))
    return 0


def _self_test() -> bool:
    """Dependency-light sanity checks for task/config plumbing."""
    assert normalize_task("SST-2") == "sst2"
    assert normalize_task("squad2") == "squad_v2"
    assert normalize_task("cnn_dailymail") == "cnndm"
    assert table6_group_for("roberta", "mnli") == "glue-big"
    assert table6_group_for("roberta", "rte") == "glue-small"
    assert table6_group_for("t5", "cnndm") == "cnndm"

    merged = merge_config(None, {"model_name_or_path": "roberta-base", "task": "sst2"})
    assert merged["learning_rate"] == 2e-4 and merged["batch_size"] == 32
    assert merged["epochs"] == 40 and merged["distill_epochs"] == 20
    assert merged["target_sparsity"] == 0.60 and merged["initial_rank"] == 8

    abl = apply_ablation(merged, "w/o D_S")
    assert abl["use_distillation"] is False
    assert apply_ablation(merged, "no_salience")["use_kurtosis"] is False
    assert apply_ablation(merged, "w/o A_T")["use_adaptive_tuning"] is False
    assert apply_ablation(merged, "w/o A_P")["mask_alpha"] == 1.0

    args = parse_args(["--model", "roberta-base", "--task", "sst2", "--target_sparsity", "0.6"])
    overrides = cli_overrides(args)
    assert overrides["model_name_or_path"] == "roberta-base"
    assert overrides["task"] == "sst2"
    assert abs(overrides["target_sparsity"] - 0.6) < 1e-12
    return True


if __name__ == "__main__":  # pragma: no cover
    if "--self-test" in sys.argv:
        print("self-test:", "ok" if _self_test() else "failed")
        sys.exit(0)
    sys.exit(main())
