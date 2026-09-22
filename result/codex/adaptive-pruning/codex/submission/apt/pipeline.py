"""Experiment plumbing: datasets, models, tasks and the run entry points.

``run_apt``           -- the full APT recipe (Tables 2/3, Figures 3/4)
``run_apt_ablation``  -- the Table 4 / Table 5 ablations
``run_baseline``      -- FT / LoRA / LoRA+Prune / Prune+Distill /
                         LoRA+Prune+Distill comparison runs

Everything is driven by :class:`apt.trainer.APTConfig` plus the task name, so a
single script can reproduce a whole table.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Tuple

from data.tasks import GLUE_SPECS, build_task

from .trainer import APTConfig, APTTrainer, set_seed


# --------------------------------------------------------------------------- #
# Dataset / model factory
# --------------------------------------------------------------------------- #
TASK_SPECS: Dict[str, Dict[str, str]] = {
    "sst2": {"path": "nyu-mll/glue", "config": "sst2"},
    "mnli": {"path": "nyu-mll/glue", "config": "mnli"},
    "qqp": {"path": "nyu-mll/glue", "config": "qqp"},
    "qnli": {"path": "nyu-mll/glue", "config": "qnli"},
    "cola": {"path": "nyu-mll/glue", "config": "cola"},
    "mrpc": {"path": "nyu-mll/glue", "config": "mrpc"},
    "rte": {"path": "nyu-mll/glue", "config": "rte"},
    "stsb": {"path": "nyu-mll/glue", "config": "stsb"},
    "squad": {"path": "rajpurkar/squad_v2", "config": ""},
    "cnn_dm": {"path": "abisee/cnn_dailymail", "config": "3.0.0"},
}


def load_task_dataset(task: str, cache_dir: Optional[str] = None):
    from datasets import load_dataset

    spec = TASK_SPECS[task]
    if spec["config"]:
        return load_dataset(spec["path"], spec["config"], cache_dir=cache_dir)
    return load_dataset(spec["path"], cache_dir=cache_dir)


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def is_seq2seq(model_name: str) -> bool:
    return "t5" in model_name.lower()


def build_model(model_name: str, task: str, num_labels: int = 2):
    from transformers import (
        AutoModelForQuestionAnswering,
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
    )

    if task in {"squad", "squad_v2"}:
        return AutoModelForQuestionAnswering.from_pretrained(model_name)
    if is_seq2seq(model_name):
        return AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)


def prepare_features(task_obj, dataset, splits: Tuple[str, ...]) -> Dict[str, Any]:
    out = {}
    for split in splits:
        if split in dataset:
            out[split] = task_obj.tokenize(split)
    return out


@dataclass
class PreparedRun:
    tokenizer: Any
    dataset: Any
    task: Any
    train_features: Any
    eval_features: Any
    raw_eval: Any
    num_labels: int


def prepare(task: str, model_name: str, cache_dir: Optional[str] = None, limit_train: Optional[int] = None) -> PreparedRun:
    tokenizer = load_tokenizer(model_name)
    dataset = load_task_dataset(task, cache_dir=cache_dir)
    text_to_text = is_seq2seq(model_name) and task in GLUE_SPECS
    task_obj = build_task(task, dataset, tokenizer, text_to_text=text_to_text)
    num_labels = getattr(task_obj, "num_labels", 2)

    if task == "squad":
        train_split, eval_split = "train", "validation"
    else:
        train_split, eval_split = "train", task_obj.split_eval

    train_features = task_obj.tokenize(train_split)
    if limit_train:
        train_features = train_features.select(range(min(limit_train, len(train_features))))
    eval_features = task_obj.tokenize(eval_split)
    raw_eval = dataset[eval_split]
    return PreparedRun(tokenizer, dataset, task_obj, train_features, eval_features, raw_eval, num_labels)


# --------------------------------------------------------------------------- #
# APT
# --------------------------------------------------------------------------- #
def run_apt(
    task: str,
    model_name: str = "roberta-base",
    config: Optional[APTConfig] = None,
    cache_dir: Optional[str] = None,
    limit_train: Optional[int] = None,
    limit_eval: Optional[int] = None,
) -> Dict[str, Any]:
    cfg = config or APTConfig(model_name=model_name, task=task)
    cfg = replace(cfg, model_name=model_name, task=task)
    set_seed(cfg.seed)

    run = prepare(task, model_name, cache_dir=cache_dir, limit_train=limit_train)
    model = build_model(model_name, task, run.num_labels)

    eval_features = run.eval_features
    raw_eval = run.raw_eval
    if limit_eval:
        eval_features = eval_features.select(range(min(limit_eval, len(eval_features))))
        raw_eval = raw_eval.select(range(min(limit_eval, len(raw_eval))))

    trainer = APTTrainer(
        model,
        run.tokenizer,
        run.task,
        cfg,
        run.train_features,
        eval_features,
        raw_eval,
    )
    result = trainer.train()
    result["model_name"] = model_name
    result["task"] = task
    return result


# --------------------------------------------------------------------------- #
# Ablations (Tables 4 and 5)
# --------------------------------------------------------------------------- #
ABLATIONS = {
    "apt": {},
    # Section 5.6: "In these cases, we only train LMs with adaptive tuning
    # strategies with supervised finetuning objectives without distillation."
    "wo_adaptive_pruning": {"use_adaptive_pruning": False, "use_distillation": False},
    "wo_adaptive_tuning": {"use_adaptive_tuning": False},
    "wo_distillation": {"use_distillation": False},
    "wo_kurtosis": {"use_kurtosis": False},          # Table 5 (outlier term)
    "wo_salience": {"salience_based_allocation": False},
}


def run_apt_ablation(
    task: str,
    ablation: str,
    model_name: str = "roberta-base",
    base: Optional[APTConfig] = None,
    **kwargs,
) -> Dict[str, Any]:
    if ablation not in ABLATIONS:
        raise ValueError(f"unknown ablation '{ablation}'")
    cfg = base or APTConfig(model_name=model_name, task=task)
    cfg = replace(cfg, **ABLATIONS[ablation])
    return run_apt(task, model_name, cfg, **kwargs)


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def run_baseline(
    baseline: str,
    task: str,
    model_name: str = "roberta-base",
    target_sparsity: float = 0.6,
    config: Optional[APTConfig] = None,
    cache_dir: Optional[str] = None,
    limit_train: Optional[int] = None,
) -> Dict[str, Any]:
    """Dispatch to the baseline implementations in :mod:`baselines`."""
    from baselines import (
        run_cofi,
        run_finetune,
        run_lora,
        run_lora_prune,
        run_lora_prune_distill,
    )

    cfg = config or APTConfig(model_name=model_name, task=task, target_sparsity=target_sparsity)
    cfg = replace(cfg, model_name=model_name, task=task, target_sparsity=target_sparsity)
    run = prepare(task, model_name, cache_dir=cache_dir, limit_train=limit_train)
    model = build_model(model_name, task, run.num_labels)
    common = dict(
        model=model,
        tokenizer=run.tokenizer,
        task=run.task,
        config=cfg,
        train_features=run.train_features,
        eval_features=run.eval_features,
        raw_eval=run.raw_eval,
    )
    dispatch = {
        "ft": run_finetune,
        "finetune": run_finetune,
        "lora": run_lora,
        "lora_prune": run_lora_prune,
        "prune_distill": run_cofi,
        "cofi": run_cofi,
        "lora_prune_distill": run_lora_prune_distill,
    }
    if baseline not in dispatch:
        raise ValueError(f"unknown baseline '{baseline}'")
    result = dispatch[baseline](**common)
    result["baseline"] = baseline
    result["task"] = task
    return result


__all__ = [
    "TASK_SPECS",
    "APTConfig",
    "PreparedRun",
    "prepare",
    "build_model",
    "load_task_dataset",
    "load_tokenizer",
    "run_apt",
    "run_apt_ablation",
    "run_baseline",
    "ABLATIONS",
]
