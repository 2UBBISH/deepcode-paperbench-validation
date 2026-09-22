"""Full fine-tuning (FT) baseline for the APT reproduction.

This module implements the vanilla full-parameter fine-tuning baseline used in
Table 2 / Table 11 of *APT: Adaptive Pruning and Tuning Pretrained Language
Models for Efficient Training and Inference* (Section 5.2, "FT").

FT serves two purposes in the reproduction:

1. It is the accuracy upper bound / reference point for every task.
2. It defines the normalization constants of the efficiency tables: training
   time-to-accuracy (TTA) reaches 97% of the FT dev score (``TTA_FRACTION``),
   and every training/inference cost is reported relative to FT (100%).

The implementation is intentionally dependency-light: ``torch`` and
``transformers`` are imported lazily so that the module (and its self test) can
be imported in a bare environment.

Public API
----------
``FTConfig``          -- Table 6 hyper-parameters for the FT baseline
``FTModel``           -- HF model + tokenizer construction helper
``FTTrainer``         -- training / evaluation loop with TTA bookkeeping
``train_ft``          -- one-call FT entry point
``fine_tune``         -- alias of :func:`train_ft`
``build_ft_optimizer``-- AdamW with no-decay parameter groups
``freeze_all`` / ``unfreeze_all`` -- parameters helpers
"""

from __future__ import annotations

import json
import math
import os
import random
import time
import warnings
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "FTConfig",
    "FTModel",
    "FTTrainer",
    "build_ft_optimizer",
    "fine_tune",
    "freeze_all",
    "train_ft",
    "unfreeze_all",
    "ft_parameter_count",
    "evaluate_ft_model",
    "TTA_FRACTION",
    "GLUE_TASKS",
    "SQUAD_TASKS",
    "SEQ2SEQ_TASKS",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Time-to-accuracy target: 97% of the fully fine-tuned reference score.
TTA_FRACTION = 0.97

GLUE_TASKS = (
    "mnli",
    "sst2",
    "qnli",
    "qqp",
    "mrpc",
    "cola",
    "rte",
    "stsb",
)
SQUAD_TASKS = ("squad", "squad_v2", "squad2")
SEQ2SEQ_TASKS = ("cnndm", "cnn_dailymail", "xsum", "samsum")

#: Keys forwarded to a HuggingFace forward pass.
MODEL_INPUT_KEYS = (
    "input_ids",
    "attention_mask",
    "token_type_ids",
    "position_ids",
    "labels",
    "decoder_input_ids",
    "decoder_attention_mask",
    "start_positions",
    "end_positions",
)

#: Table 6 defaults, grouped exactly like the paper's table.
TABLE6_FT_DEFAULTS: Dict[str, Dict[str, float]] = {
    "glue-big": {"learning_rate": 2e-4, "batch_size": 32, "epochs": 40},
    "glue-small": {"learning_rate": 2e-4, "batch_size": 32, "epochs": 40},
    "squad": {"learning_rate": 2e-4, "batch_size": 32, "epochs": 40},
    "cnndm": {"learning_rate": 1e-4, "batch_size": 16, "epochs": 16},
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def canonical_task(task: Optional[str]) -> str:
    """Normalize a task name into a canonical key."""
    if not task:
        return "sst2"
    key = str(task).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "sst": "sst2",
        "sst_2": "sst2",
        "mnli_matched": "mnli",
        "mnli_mismatched": "mnli",
        "mnli_mm": "mnli",
        "stsb": "stsb",
        "sts_b": "stsb",
        "squad2": "squad_v2",
        "squadv2": "squad_v2",
        "squad_v2": "squad_v2",
        "cnn_dailymail": "cnndm",
        "cnn_dm": "cnndm",
        "cnndm": "cnndm",
    }
    return aliases.get(key, key)


def is_glue_task(task: str) -> bool:
    return canonical_task(task) in GLUE_TASKS


def is_squad_task(task: str) -> bool:
    return canonical_task(task) in ("squad", "squad_v2")


def is_seq2seq_task(task: str) -> bool:
    return canonical_task(task) in ("cnndm", "xsum", "samsum")


def table6_group_for(model_type: str, task: str) -> str:
    """Map (model, task) to a Table 6 column."""
    task = canonical_task(task)
    if is_seq2seq_task(task):
        return "cnndm"
    if is_squad_task(task):
        return "squad"
    if task in ("mnli", "sst2", "qnli", "qqp"):
        return "glue-big"
    return "glue-small"


@dataclass
class FTConfig:
    """Hyper-parameters of the full fine-tuning baseline (paper Table 6)."""

    model_name_or_path: str = "roberta-base"
    model_type: str = "roberta"
    task: str = "sst2"
    table6_group: Optional[str] = None

    # Table 6
    learning_rate: float = 2e-4
    batch_size: int = 32
    epochs: int = 40
    distill_epochs: int = 20  # unused by FT, kept for config compatibility

    # Optimization
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    warmup_ratio: float = 0.06
    lr_kind: str = "linear"
    max_grad_norm: float = 1.0
    seed: int = 42

    # Data
    max_seq_length: int = 128
    max_target_length: int = 128
    doc_stride: int = 128
    max_query_length: int = 64
    n_best_size: int = 20
    max_answer_length: int = 30
    null_score_diff_threshold: float = 0.0
    dynamic_padding: bool = False
    num_workers: int = 0

    # Runtime / evaluation
    device: str = "cuda"
    output_dir: str = "outputs/ft"
    logging_steps: int = 50
    eval_steps: int = 0  # 0 -> evaluate once per epoch
    save_steps: int = 0
    inference_batch_size: int = 128
    sequence_length: int = 128
    fp16: bool = False
    bf16: bool = False
    measure_efficiency: bool = True
    max_train_batches: int = 0  # 0 -> all
    max_eval_batches: int = 0

    extra: Dict[str, Any] = field(default_factory=dict)

    # -- construction helpers -------------------------------------------------
    def __post_init__(self) -> None:
        self.task = canonical_task(self.task)
        if self.table6_group is None:
            self.table6_group = table6_group_for(self.model_type, self.task)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]] = None, **overrides: Any) -> "FTConfig":
        data = dict(data or {})
        data.update(overrides)
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        kwargs: Dict[str, Any] = {}
        extra: Dict[str, Any] = dict(data.pop("extra", {}) or {})
        for key, value in data.items():
            if key in known:
                kwargs[key] = value
            else:
                extra[key] = value
        if extra:
            kwargs["extra"] = extra
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str, **overrides: Any) -> "FTConfig":
        import yaml  # lazy

        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return cls.from_dict(data, **overrides)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> str:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        return path

    @property
    def is_seq2seq(self) -> bool:
        return is_seq2seq_task(self.task)

    @property
    def is_squad(self) -> bool:
        return is_squad_task(self.task)

    @property
    def num_epochs(self) -> int:
        return int(self.epochs)


def apply_table6_defaults(config: FTConfig, table: Optional[Dict[str, Dict[str, float]]] = None) -> FTConfig:
    """Fill Table 6 numbers that were left at their constructor defaults."""
    table = table or TABLE6_FT_DEFAULTS
    column = table.get(config.table6_group or "", {}) if table else {}
    if column:
        defaults = FTConfig()
        for key in ("learning_rate", "batch_size", "epochs"):
            if key in column and getattr(config, key) == getattr(defaults, key):
                setattr(config, key, column[key])
    return config


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # pragma: no cover - numpy optional
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - torch optional
        pass
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def _torch():
    import torch  # lazy

    return torch


def resolve_device(device: Optional[str] = None):
    torch = _torch()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = str(device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    return torch.device(device)


def move_to_device(batch: Any, device) -> Any:
    torch = _torch()
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(v, device) for v in batch)
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    return batch


def model_inputs(batch: Dict[str, Any], keys: Sequence[str] = MODEL_INPUT_KEYS) -> Dict[str, Any]:
    return {k: v for k, v in batch.items() if k in keys and v is not None}


def num_labels_for_task(task: str) -> int:
    task = canonical_task(task)
    if task in ("mnli", "qqp", "mrpc", "rte", "qnli"):
        return 2
    if task in ("sst2", "cola"):
        return 2
    if task == "stsb":
        return 1
    return 2


def problem_type_for_task(task: str) -> str:
    return "regression" if canonical_task(task) == "stsb" else "single_label_classification"


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


class FTModel:
    """Builds the (model, tokenizer) pair appropriate for a task."""

    def __init__(self, config: Optional[FTConfig] = None, model=None, tokenizer=None, **kwargs: Any):
        if config is None:
            config = FTConfig.from_dict(kwargs.pop("config", None), **kwargs)
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        if self.model is None or self.tokenizer is None:
            built_model, built_tokenizer = self.build()
            self.model = self.model or built_model
            self.tokenizer = self.tokenizer or built_tokenizer

    # -- builders -------------------------------------------------------------
    def build(self, **kwargs: Any):
        from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification
        from transformers import AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM, AutoTokenizer

        config = self.config
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path, use_fast=True, **kwargs
        )
        model_kwargs: Dict[str, Any] = {}
        if config.model_type == "t5" or config.is_seq2seq:
            model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name_or_path, **model_kwargs)
        elif config.is_squad:
            model = AutoModelForQuestionAnswering.from_pretrained(config.model_name_or_path, **model_kwargs)
        else:
            num_labels = num_labels_for_task(config.task)
            try:
                hf_config = AutoConfig.from_pretrained(config.model_name_or_path)
                hf_config.num_labels = num_labels
                hf_config.problem_type = problem_type_for_task(config.task)
                model = AutoModelForSequenceClassification.from_pretrained(
                    config.model_name_or_path, config=hf_config, **model_kwargs
                )
            except Exception:
                model = AutoModelForSequenceClassification.from_pretrained(
                    config.model_name_or_path, num_labels=num_labels, **model_kwargs
                )
        return model, tokenizer

    def parameters(self):
        return list(self.model.parameters())

    def train(self):
        self.model.train()
        return self.model

    def eval(self):
        self.model.eval()
        return self.model

    def to(self, device):
        self.model.to(device)
        return self


# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------


def freeze_all(model) -> None:
    """Freeze every parameter of ``model`` (requires_grad = False)."""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_all(model) -> None:
    """Unfreeze every parameter of ``model`` (full fine-tuning)."""
    for param in model.parameters():
        param.requires_grad = True


def ft_parameter_count(model, trainable_only: bool = True) -> int:
    """Number of (trainable) parameters - FT never prunes anything."""
    total = 0
    for param in model.parameters():
        if trainable_only and not param.requires_grad:
            continue
        total += param.numel()
    return total


def trainable_parameter_list(model) -> List[Any]:
    seen = set()
    params = []
    for param in model.parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        params.append(param)
    return params


# ---------------------------------------------------------------------------
# Optimizer / scheduler
# ---------------------------------------------------------------------------


def build_ft_optimizer(
    model,
    *,
    lr: float = 2e-4,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    name: str = "adamw",
    parameters: Optional[Iterable[Any]] = None,
):
    """Build the FT optimizer with AdamW decay / no-decay parameter groups.

    Following the BERT/RoBERTa fine-tuning convention, bias and LayerNorm
    parameters are excluded from weight decay.
    """
    torch = _torch()
    params = list(parameters) if parameters is not None else trainable_parameter_list(model)
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight", "ln_", "norm.weight")
    decay, no_decay_params = [], []
    named = dict(model.named_parameters()) if hasattr(model, "named_parameters") else {}
    id_to_name = {id(p): n for n, p in named.items()}
    for index, param in enumerate(params):
        pname = id_to_name.get(id(param), f"param_{index}")
        if any(nd in pname for nd in no_decay):
            no_decay_params.append(param)
        else:
            decay.append(param)
    groups: List[Dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay, "lr": lr})
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0, "lr": lr})
    if not groups:
        groups = [{"params": params, "weight_decay": weight_decay, "lr": lr}]

    name = (name or "adamw").lower()
    if name in ("adamw", "adam_w"):
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)
    if name == "adam":
        return torch.optim.Adam(groups, lr=lr, betas=betas, eps=eps)
    if name in ("sgd",):
        return torch.optim.SGD(groups, lr=lr, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {name}")


def lr_factor(step: int, total_steps: int, *, warmup_steps: int = 0, kind: str = "linear") -> float:
    """Linear warmup followed by linear (or cosine) decay."""
    step = max(0, int(step))
    total_steps = max(1, int(total_steps))
    if warmup_steps > 0 and step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    progress = min(max(progress, 0.0), 1.0)
    if kind == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return max(0.0, 1.0 - progress)


# ---------------------------------------------------------------------------
# Time-to-accuracy bookkeeping
# ---------------------------------------------------------------------------


class TimeToAccuracyTracker:
    """Tracks wall-clock seconds needed to reach ``fraction`` x reference."""

    def __init__(self, reference: Optional[float], fraction: float = TTA_FRACTION, higher_is_better: bool = True):
        self.reference = reference
        self.fraction = fraction
        self.higher_is_better = higher_is_better
        self.history: List[Tuple[float, float]] = []
        self.tta: Optional[float] = None

    @property
    def target(self) -> Optional[float]:
        if self.reference is None:
            return None
        return self.reference * self.fraction

    def update(self, elapsed_seconds: float, metric_value: Optional[float]) -> Optional[float]:
        if metric_value is None:
            return self.tta
        self.history.append((float(elapsed_seconds), float(metric_value)))
        if self.tta is not None or self.target is None:
            return self.tta
        reached = (
            metric_value >= self.target if self.higher_is_better else metric_value <= self.target
        )
        if not reached:
            return None
        # linear interpolation between the previous and current observations
        if len(self.history) == 1:
            self.tta = float(elapsed_seconds)
            return self.tta
        t0, v0 = self.history[-2]
        t1, v1 = self.history[-1]
        tgt = self.target
        if v1 == v0:
            self.tta = float(t1)
            return self.tta
        ratio = (tgt - v0) / (v1 - v0)
        ratio = min(max(ratio, 0.0), 1.0)
        self.tta = float(t0 + ratio * (t1 - t0))
        return self.tta

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tta_seconds": self.tta,
            "tta_target": self.target,
            "tta_fraction": self.fraction,
            "tta_reference": self.reference,
            "history": list(self.history),
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class FTTrainer:
    """Full fine-tuning trainer (the reference baseline of the paper)."""

    def __init__(
        self,
        config: Optional[FTConfig] = None,
        model=None,
        tokenizer=None,
        train_dataloader=None,
        eval_dataloader=None,
        compute_metrics: Optional[Callable[..., Dict[str, float]]] = None,
        reference_metric: Optional[float] = None,
        **kwargs: Any,
    ):
        if config is None:
            config = FTConfig.from_dict(kwargs.pop("config", None), **kwargs)
        elif isinstance(config, dict):
            config = FTConfig.from_dict(config, **kwargs)
        self.config = apply_table6_defaults(config)

        if model is None:
            model, tokenizer = FTModel(self.config).model, FTModel(self.config).tokenizer
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.compute_metrics = compute_metrics
        self.reference_metric = reference_metric

        self.device = resolve_device(self.config.device)
        self.optimizer = None
        self.scheduler = None
        self.global_step = 0
        self.epoch = 0
        self.history: List[Dict[str, Any]] = []
        self.tta = TimeToAccuracyTracker(
            reference_metric, fraction=TTA_FRACTION, higher_is_better=True
        )
        self.train_seconds = 0.0
        self.train_peak_mem_mb: Optional[float] = None

    # -- setup ---------------------------------------------------------------
    def setup_model(self):
        unfreeze_all(self.model)
        self.model.to(self.device)
        return self.model

    def setup_optimizer(self):
        self.optimizer = build_ft_optimizer(
            self.model,
            lr=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
            betas=(float(self.config.adam_beta1), float(self.config.adam_beta2)),
            eps=float(self.config.adam_epsilon),
            name=self.config.optimizer,
        )
        return self.optimizer

    def setup_scheduler(self, total_steps: int):
        warmup_steps = int(round(total_steps * float(self.config.warmup_ratio)))
        torch = _torch()
        kind = self.config.lr_kind
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: lr_factor(step, total_steps, warmup_steps=warmup_steps, kind=kind),
        )
        return self.scheduler

    def steps_per_epoch(self) -> int:
        if self.train_dataloader is None:
            return 0
        try:
            return len(self.train_dataloader)
        except TypeError:  # pragma: no cover - iterable datasets
            return 0

    # -- train step ----------------------------------------------------------
    def training_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        torch = _torch()
        self.model.train()
        batch = move_to_device(batch, self.device)
        inputs = model_inputs(batch)
        outputs = self.model(**inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        if loss is None:
            raise RuntimeError("FT training step produced no loss (missing labels?).")
        loss.backward()
        if self.config.max_grad_norm and self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.max_grad_norm))
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1
        return {"loss": float(loss.detach().float().cpu().item())}

    # -- evaluation ----------------------------------------------------------
    def predict(self, dataloader=None) -> Tuple[List[Any], List[Any], Dict[str, Any]]:
        torch = _torch()
        dataloader = dataloader if dataloader is not None else self.eval_dataloader
        if dataloader is None:
            return [], [], {}
        self.model.eval()
        predictions: List[Any] = []
        references: List[Any] = []
        spans: Dict[str, Any] = {"start_logits": [], "end_logits": [], "features": []}
        max_batches = int(self.config.max_eval_batches or 0)
        with torch.no_grad():
            for index, batch in enumerate(dataloader):
                if max_batches and index >= max_batches:
                    break
                batch = move_to_device(batch, self.device)
                labels = batch.get("labels")
                inputs = model_inputs(batch)
                outputs = self.model(**inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                if self.config.is_squad:
                    spans["start_logits"].append(logits[0].detach().float().cpu())
                    spans["end_logits"].append(logits[1].detach().float().cpu())
                    features = batch.get("features") or batch.get("feature") or []
                    if features:
                        spans["features"].extend(features if isinstance(features, (list, tuple)) else list(features))
                else:
                    logits = logits.detach().float().cpu()
                    if logits.dim() > 1 and logits.size(-1) == 1:
                        predictions.extend(logits.view(-1).tolist())
                    else:
                        predictions.extend(logits.argmax(-1).tolist())
                if labels is not None:
                    references.extend(_flatten(labels))
        return predictions, references, spans

    def evaluate(self, dataloader=None, step: Optional[int] = None) -> Dict[str, float]:
        dataloader = dataloader if dataloader is not None else self.eval_dataloader
        if dataloader is None:
            return {}
        metrics: Dict[str, float] = {}
        if self.config.is_squad:
            metrics = self._evaluate_squad(dataloader)
        elif self.config.is_seq2seq:
            metrics = self._evaluate_seq2seq(dataloader)
        else:
            predictions, references, _ = self.predict(dataloader)
            metrics = self._compute_task_metrics(predictions, references)
        record = {"epoch": self.epoch, "step": step if step is not None else self.global_step}
        record.update(metrics)
        record["elapsed_seconds"] = self.train_seconds
        self.history.append(record)
        return metrics

    def _compute_task_metrics(self, predictions: List[Any], references: List[Any]) -> Dict[str, float]:
        if self.compute_metrics is not None:
            return dict(self.compute_metrics(predictions, references))
        try:
            from apt.eval.metrics import compute_metrics as task_metrics

            return dict(task_metrics(self.config.task, predictions, references))
        except Exception:
            total = len(references)
            if total == 0:
                return {}
            correct = sum(int(p == r) for p, r in zip(predictions, references))
            return {"accuracy": 100.0 * correct / total}

    def _evaluate_squad(self, dataloader) -> Dict[str, float]:
        try:
            from apt.data.squad import build_squad_features, compute_squad_metrics
            from apt.data.squad import references_for_examples, write_predictions
        except Exception:  # pragma: no cover - data module optional
            return {}
        # a dedicated SQuAD dataloader carries its features/examples on the dataset
        dataset = getattr(dataloader, "dataset", None)
        features = getattr(dataset, "features", None) if dataset is not None else None
        examples = getattr(dataset, "examples", None) if dataset is not None else None
        if features is None or examples is None:
            return {}
        predictions, _, spans = self.predict(dataloader)
        all_results = []
        for start, end in zip(spans["start_logits"], spans["end_logits"]):
            for row in range(start.size(0)):
                all_results.append(
                    {
                        "unique_id": int(features[len(all_results)].unique_id),
                        "start_logits": start[row].tolist(),
                        "end_logits": end[row].tolist(),
                    }
                )
        try:
            answers, _, _ = write_predictions(
                features,
                examples,
                all_results,
                n_best_size=self.config.n_best_size,
                max_answer_length=self.config.max_answer_length,
                null_score_diff_threshold=self.config.null_score_diff_threshold,
                tokenizer=self.tokenizer,
            )
            references = references_for_examples(examples)
            return dict(compute_squad_metrics(answers, references))
        except Exception:  # pragma: no cover - metrics optional
            return {}

    def _evaluate_seq2seq(self, dataloader) -> Dict[str, float]:
        try:
            from apt.data.cnndm import evaluate_cnndm

            return dict(
                evaluate_cnndm(
                    self.model,
                    self.tokenizer,
                    dataloader,
                    max_batches=self.config.max_eval_batches or None,
                    max_length=int(self.config.max_target_length),
                )
            )
        except Exception:  # pragma: no cover - data module optional
            return {}

    def primary_metric(self, metrics: Dict[str, float]) -> Optional[float]:
        if not metrics:
            return None
        try:
            from apt.eval.metrics import primary_metric as _primary

            value = _primary(self.config.task, metrics)
            if value is not None:
                return float(value)
        except Exception:
            pass
        for key in ("accuracy", "f1", "rougeL", "rouge_l", "score", "pearsonr"):
            if key in metrics:
                return float(metrics[key])
        for value in metrics.values():
            if isinstance(value, (int, float)):
                return float(value)
        return None

    # -- main loop -----------------------------------------------------------
    def fit(self, max_steps: Optional[int] = None) -> Dict[str, Any]:
        torch = _torch()
        self.setup_model()
        self.setup_optimizer()
        steps_per_epoch = self.steps_per_epoch()
        total_steps = steps_per_epoch * int(self.config.epochs)
        if max_steps:
            total_steps = min(total_steps, int(max_steps))
        self.setup_scheduler(total_steps)

        if not self.train_dataloader:
            warnings.warn("FITrainer.fit called without a training dataloader.")
            return {"history": self.history, "tta_seconds": None}

        if self.config.measure_efficiency and torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        started = time.time()
        self.model.train()
        stop = False
        for epoch in range(int(self.config.epochs)):
            self.epoch = epoch
            epoch_loss = 0.0
            n_batches = 0
            max_batches = int(self.config.max_train_batches or 0)
            for index, batch in enumerate(self.train_dataloader):
                if max_batches and index >= max_batches:
                    break
                stats = self.training_step(batch)
                epoch_loss += stats["loss"]
                n_batches += 1
                if self.config.logging_steps and self.global_step % self.config.logging_steps == 0:
                    print(
                        f"[FT] epoch {epoch + 1}/{self.config.epochs} step {self.global_step} "
                        f"loss {epoch_loss / max(1, n_batches):.4f}"
                    )
                if max_steps and self.global_step >= int(max_steps):
                    stop = True
                    break
            self.train_seconds = time.time() - started
            metrics = self.evaluate(step=self.global_step)
            primary = self.primary_metric(metrics)
            self.tta.update(self.train_seconds, primary)
            if metrics:
                print(f"[FT] epoch {epoch + 1} metrics: {metrics}")
            if stop:
                break

        self.train_seconds = time.time() - started
        if self.config.measure_efficiency and torch.cuda.is_available():
            try:
                self.train_peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            except Exception:
                self.train_peak_mem_mb = None

        return self.summary()

    def summary(self) -> Dict[str, Any]:
        final = self.history[-1] if self.history else {}
        primary = self.primary_metric(final) if final else None
        return {
            "method": "ft",
            "task": self.config.task,
            "model_name_or_path": self.config.model_name_or_path,
            "metrics": {k: v for k, v in final.items() if isinstance(v, (int, float))},
            "primary": primary,
            "train_time_s": self.train_seconds,
            "train_peak_mem_mb": self.train_peak_mem_mb,
            "tta_seconds": self.tta.tta,
            "tta_target": self.tta.target,
            "history": self.history,
            "global_step": self.global_step,
            "parameters": ft_parameter_count(self.model, trainable_only=False),
        }

    # -- export --------------------------------------------------------------
    def save(self, output_dir: Optional[str] = None) -> str:
        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if self.model is not None:
            try:
                self.model.save_pretrained(output_dir)
            except Exception:  # pragma: no cover
                pass
        if self.tokenizer is not None:
            try:
                self.tokenizer.save_pretrained(output_dir)
            except Exception:  # pragma: no cover
                pass
        self.config.save(os.path.join(output_dir, "ft_config.json"))
        with open(os.path.join(output_dir, "ft_summary.json"), "w", encoding="utf-8") as handle:
            json.dump(self.summary(), handle, indent=2, default=str)
        return output_dir


def _flatten(values: Any) -> List[Any]:
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        out: List[Any] = []
        for value in values:
            out.extend(_flatten(value))
        return out
    if hasattr(values, "detach"):  # torch tensor
        tensor = values.detach().cpu()
        if tensor.dim() == 0:
            return [tensor.item()]
        return tensor.reshape(-1).tolist()
    if hasattr(values, "tolist"):  # numpy array
        array = values.tolist()
        if isinstance(array, list):
            out = []
            for item in array:
                out.extend(_flatten(item))
            return out
        return [array]
    return [values]


# ---------------------------------------------------------------------------
# One-call entry points
# ---------------------------------------------------------------------------


def train_ft(
    config: Optional[FTConfig] = None,
    model=None,
    tokenizer=None,
    train_dataloader=None,
    eval_dataloader=None,
    compute_metrics: Optional[Callable[..., Dict[str, float]]] = None,
    reference_metric: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Train a fully fine-tuned model and return its summary."""
    trainer = FTTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        compute_metrics=compute_metrics,
        reference_metric=reference_metric,
        **kwargs,
    )
    summary = trainer.fit()
    summary["trainer"] = trainer
    return summary


# alias used in the reproduction plan / config docs
fine_tune = train_ft


def evaluate_ft_model(
    trainer: FTTrainer,
    dataloader=None,
    *,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate a (trained) FT model and optionally persist the metrics."""
    metrics = trainer.evaluate(dataloader)
    payload = {
        "task": trainer.config.task,
        "metrics": metrics,
        "primary": trainer.primary_metric(metrics),
    }
    if save_path:
        directory = os.path.dirname(os.path.abspath(save_path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    return payload


# ---------------------------------------------------------------------------
# Self test (dependency-light)
# ---------------------------------------------------------------------------


def _self_test() -> bool:
    """Sanity checks that do not require torch / transformers."""
    config = FTConfig.from_dict({"task": "SST-2", "model_name_or_path": "roberta-base"})
    assert config.task == "sst2", config.task
    assert config.table6_group == "glue-big", config.table6_group
    assert abs(config.learning_rate - 2e-4) < 1e-12

    squad = FTConfig.from_dict({"task": "squad_v2", "model_type": "roberta"})
    assert squad.is_squad and squad.table6_group == "squad"

    cnndm = FTConfig.from_dict({"task": "cnn_dailymail", "model_type": "t5"})
    assert cnndm.is_seq2seq and canonical_task(cnndm.task) == "cnndm"

    # Table 6 columns resolve correctly.
    assert table6_group_for("roberta", "mrpc") == "glue-small"
    assert table6_group_for("roberta", "qnli") == "glue-big"
    assert table6_group_for("roberta", "squad_v2") == "squad"
    assert table6_group_for("t5", "cnndm") == "cnndm"

    # TTA interpolation: 97% target reached between two evaluations.
    tracker = TimeToAccuracyTracker(reference=94.8, fraction=0.97)
    assert abs(tracker.target - 94.8 * 0.97) < 1e-9
    tracker.update(100.0, 90.0)
    assert tracker.tta is None, tracker.tta
    tracker.update(200.0, 95.0)
    assert tracker.tta is not None and 100.0 < tracker.tta < 200.0, tracker.tta
    interpolated = tracker.tta
    ratio = (94.8 * 0.97 - 90.0) / (95.0 - 90.0)
    assert abs(interpolated - (100.0 + ratio * 100.0)) < 1e-6

    # LR schedule: linear warmup then linear decay.
    assert abs(lr_factor(0, 100, warmup_steps=10) - 0.0) < 1e-12
    assert abs(lr_factor(10, 100, warmup_steps=10) - 1.0) < 1e-12
    assert abs(lr_factor(100, 100, warmup_steps=10) - 0.0) < 1e-12
    assert 0.0 < lr_factor(55, 100, warmup_steps=10) < 1.0

    # flatten helper
    assert _flatten([1, [2, 3]]) == [1, 2, 3]
    assert model_inputs({"input_ids": 1, "labels": 2, "foo": 3}) == {"input_ids": 1, "labels": 2}

    print("apt.baselines.ft self-test passed.")
    return True


if __name__ == "__main__":  # pragma: no cover
    _self_test()
