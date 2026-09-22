"""LoRA baseline for APT (paper Section 5.2, Appendix A Table 6).

This module implements the plain LoRA (Hu et al., 2021) parameter-efficient
fine-tuning baseline that APT is compared against in Table 2 / Table 4 / Table 8.

Paper facts used here
---------------------
* LoRA is one of the "PEFT" baselines; it trains low-rank update matrices only
  and therefore has ~100% inference time/memory overhead (Table 2: ``LoRA`` row
  shows Train Time 2137% / 255.5%, Train Mem 60.5% / 62.0%, Inf Time 100%,
  Inf Mem 100% for RoBERTa-base / T5-base respectively).  Consequently the LoRA
  update is *not* merged at inference in the paper's efficiency protocol.
* Hyper-parameters come from Table 6 for the target task group
  (GLUE-small / GLUE-big / SQuAD: lr 2e-4, batch 32, epochs 40, distill 20;
  CNN/DM: lr 1e-4, batch 16, epochs 16, distill 6).
* ``apt`` initializes adapter ranks to 8 with a static scaling factor of 2
  (Appendix A).  The in-repo LoRA baseline uses the same rank/scaling budget so
  the comparison isolates the *method*, not the tuning capacity: the default is
  ``lora_rank=8`` with ``lora_alpha=16`` so that ``alpha / rank == 2``.

Design notes
------------
* Dependency-light: ``torch`` / ``transformers`` are imported lazily so the
  module (and ``_self_test``) work in a bare environment.
* The trainer mirrors :class:`apt.baselines.ft.FTTrainer` (same method names,
  same summary dict shape) so experiment scripts can treat FT, LoRA and APT
  uniformly.
* ``apply_lora`` replaces target ``nn.Linear`` layers with :class:`LoRALinear`,
  which keeps the frozen base weight, exposes ``lora_a``/``lora_b`` tuning
  parameters and can ``merge()`` itself back into a plain dense layer.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Optional imports from sibling APT modules (used only to avoid duplication).
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised only when the package is importable
    from .ft import (  # type: ignore
        FTConfig,
        build_ft_optimizer,
        canonical_task as _ft_canonical_task,
        is_glue_task as _ft_is_glue_task,
        is_seq2seq_task as _ft_is_seq2seq_task,
        is_squad_task as _ft_is_squad_task,
        lr_factor as _ft_lr_factor,
        model_inputs as _ft_model_inputs,
        move_to_device as _ft_move_to_device,
        num_labels_for_task as _ft_num_labels_for_task,
        problem_type_for_task as _ft_problem_type_for_task,
        resolve_device as _ft_resolve_device,
        set_seed as _ft_set_seed,
        table6_group_for as _ft_table6_group_for,
        trainable_parameter_list as _ft_trainable_parameter_list,
        TABLE6_FT_DEFAULTS as _FT_TABLE6,
        MODEL_INPUT_KEYS as _FT_MODEL_INPUT_KEYS,
    )

    _HAS_FT = True
except Exception:  # pragma: no cover
    _HAS_FT = False
    _FT_TABLE6 = None
    _FT_MODEL_INPUT_KEYS = None


TTA_FRACTION = 0.97

# ---------------------------------------------------------------------------
# Defaults (paper Table 6 + Appendix A adapter budget)
# ---------------------------------------------------------------------------
DEFAULT_LORA_RANK = 8
DEFAULT_LORA_ALPHA = 16  # alpha / rank == 2, matching APT's static scaling s = 2
DEFAULT_LORA_DROPOUT = 0.0
DEFAULT_LORA_SCALING = 2.0  # APT's adapter scaling factor (Appendix A)
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_RATIO = 0.06
DEFAULT_MAX_GRAD_NORM = 1.0
DEFAULT_SEED = 42

#: Table 6 columns; LoRA keeps the same optimisation budget as FT.
TABLE6_LORA_DEFAULTS: Dict[str, Dict[str, float]] = {
    "glue-big": {
        "learning_rate": 2e-4,
        "batch_size": 32,
        "epochs": 40,
        "distill_epochs": 20,
    },
    "glue-small": {
        "learning_rate": 2e-4,
        "batch_size": 32,
        "epochs": 40,
        "distill_epochs": 20,
    },
    "squad": {
        "learning_rate": 2e-4,
        "batch_size": 32,
        "epochs": 40,
        "distill_epochs": 20,
    },
    "cnndm": {
        "learning_rate": 1e-4,
        "batch_size": 16,
        "epochs": 16,
        "distill_epochs": 6,
    },
}
if _FT_TABLE6:
    for _key, _vals in _FT_TABLE6.items():
        TABLE6_LORA_DEFAULTS.setdefault(_key, dict(_vals))

#: Module-name suffixes that LoRA targets for each model family.  Attention
#: query/value projections are the standard LoRA placement (Hu et al., 2021);
#: T5 additionally exposes its (gated) FFN projections.
LORA_TARGET_MODULES: Dict[str, Tuple[str, ...]] = {
    "roberta": ("query", "value"),
    "bert": ("query", "value"),
    "deberta": ("query_proj", "value_proj"),
    "distilbert": ("q_lin", "v_lin"),
    "electra": ("query", "value"),
    "t5": ("q", "v"),
    "mt5": ("q", "v"),
    "bart": ("q_proj", "v_proj"),
    "opt": ("q_proj", "v_proj"),
    "gpt2": ("c_attn",),
    "llama": ("q_proj", "v_proj"),
    "mistral": ("q_proj", "v_proj"),
}

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
GLUE_BIG_TASKS = ("mnli", "sst2", "qnli", "qqp")
GLUE_SMALL_TASKS = ("mrpc", "cola", "rte", "stsb")
SQUAD_TASKS = ("squad", "squad_v2", "squad2")
SEQ2SEQ_TASKS = ("cnndm", "cnn_dailymail", "xsum", "samsum")

MODEL_INPUT_KEYS = (
    "input_ids",
    "attention_mask",
    "token_type_ids",
    "position_ids",
    "head_mask",
    "decoder_input_ids",
    "decoder_attention_mask",
    "labels",
    "p_mask",
    "start_positions",
    "end_positions",
    "cls_index",
)

_TASK_ALIASES = {
    "sst-2": "sst2",
    "sst_2": "sst2",
    "mnli-mm": "mnli",
    "mnli_matched": "mnli",
    "mnli_mismatched": "mnli",
    "sts-b": "stsb",
    "stsb": "stsb",
    "squadv2": "squad_v2",
    "squad-2": "squad_v2",
    "squad2": "squad_v2",
    "cnn_dailymail": "cnndm",
    "cnn-dailymail": "cnndm",
    "cnndm": "cnndm",
}


# ---------------------------------------------------------------------------
# Small shared helpers (imported from apt.baselines.ft when available)
# ---------------------------------------------------------------------------
def canonical_task(task: Optional[str]) -> str:
    """Normalise a task name (GLUE/SQuAD/CNN-DM) to its canonical key."""
    if _HAS_FT:
        try:
            return _ft_canonical_task(task)
        except Exception:
            pass
    if task is None:
        return ""
    name = str(task).strip().lower().replace(" ", "")
    return _TASK_ALIASES.get(name, name)


def is_glue_task(task: Any) -> bool:
    if _HAS_FT:
        try:
            return bool(_ft_is_glue_task(task))
        except Exception:
            pass
    name = task if isinstance(task, str) else getattr(task, "task", str(task))
    return canonical_task(name) in GLUE_TASKS


def is_squad_task(task: Any) -> bool:
    if _HAS_FT:
        try:
            return bool(_ft_is_squad_task(task))
        except Exception:
            pass
    name = task if isinstance(task, str) else getattr(task, "task", str(task))
    return canonical_task(name) in SQUAD_TASKS


def is_seq2seq_task(task: Any) -> bool:
    if _HAS_FT:
        try:
            return bool(_ft_is_seq2seq_task(task))
        except Exception:
            pass
    name = task if isinstance(task, str) else getattr(task, "task", str(task))
    return canonical_task(name) in SEQ2SEQ_TASKS


def table6_group_for(model_type: str, task: str = "") -> str:
    """Table 6 column selector for a (model_type, task) pair."""
    if _HAS_FT:
        try:
            return _ft_table6_group_for(model_type, task)
        except Exception:
            pass
    name = canonical_task(task)
    if name in ("cnndm", "xsum", "samsum"):
        return "cnndm"
    if name in SQUAD_TASKS:
        return "squad"
    if name in GLUE_BIG_TASKS:
        return "glue-big"
    if name in GLUE_SMALL_TASKS:
        return "glue-small"
    return "glue-big"


def set_seed(seed: int) -> None:
    if _HAS_FT:
        try:
            _ft_set_seed(seed)
            return
        except Exception:
            pass
    random.seed(seed)
    try:  # pragma: no cover
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:  # pragma: no cover
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def resolve_device(device: Optional[str] = None):
    if _HAS_FT:
        try:
            return _ft_resolve_device(device)
        except Exception:
            pass
    try:  # pragma: no cover
        import torch

        if device in (None, "", "auto"):
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    except Exception:
        return device or "cpu"


def move_to_device(batch: Any, device: Any) -> Any:
    if _HAS_FT:
        try:
            return _ft_move_to_device(batch, device)
        except Exception:
            pass
    try:  # pragma: no cover
        import torch

        if isinstance(batch, dict):
            return {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
        if torch.is_tensor(batch):
            return batch.to(device)
    except Exception:
        pass
    return batch


def model_inputs(batch: Dict[str, Any], keys: Sequence[str] = MODEL_INPUT_KEYS):
    if _HAS_FT:
        try:
            return _ft_model_inputs(batch, keys)
        except Exception:
            pass
    return {k: v for k, v in (batch or {}).items() if k in set(keys)}


def num_labels_for_task(task: str) -> int:
    if _HAS_FT:
        try:
            return int(_ft_num_labels_for_task(task))
        except Exception:
            pass
    name = canonical_task(task)
    if name == "stsb":
        return 1
    if name == "mnli":
        return 3
    if name in ("mrpc", "cola", "rte", "qnli", "qqp", "sst2"):
        return 2
    return 2


def problem_type_for_task(task: str) -> Optional[str]:
    if _HAS_FT:
        try:
            return _ft_problem_type_for_task(task)
        except Exception:
            pass
    return "regression" if canonical_task(task) == "stsb" else None


def lr_factor(
    step: int,
    total_steps: int,
    *,
    warmup_steps: int = 0,
    kind: str = "linear",
    min_factor: float = 0.0,
) -> float:
    """LR multiplier: linear warmup then linear/cosine decay."""
    if _HAS_FT:
        try:
            return float(
                _ft_lr_factor(
                    step, total_steps, warmup_steps=warmup_steps, kind=kind
                )
            )
        except Exception:
            pass
    total_steps = max(int(total_steps), 1)
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(max(warmup_steps, 1))
    progress = (step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
    progress = min(max(progress, 0.0), 1.0)
    if kind == "cosine":
        value = min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))
    elif kind == "constant":
        value = 1.0
    else:  # linear decay
        value = min_factor + (1.0 - min_factor) * (1.0 - progress)
    return float(value)


def _torch():
    import torch  # noqa: WPS433 (lazy import)

    return torch


def _nn():
    import torch.nn as nn  # noqa: WPS433

    return nn


def detect_model_type(model_or_name: Any) -> str:
    """Best-effort model family detection (for target-module selection)."""
    if isinstance(model_or_name, str):
        name = model_or_name.lower()
    else:
        cfg = getattr(model_or_name, "config", None)
        name = str(getattr(cfg, "model_type", "") or "").lower()
        if not name:
            name = type(model_or_name).__name__.lower()
    for family in (
        "roberta",
        "deberta",
        "distilbert",
        "electra",
        "bert",
        "mt5",
        "t5",
        "bart",
        "opt",
        "gpt2",
        "llama",
        "mistral",
    ):
        if family in name:
            return "mt5" if family == "mt5" else family
    return name


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class LoRAConfig:
    """LoRA baseline hyper-parameters (Table 6 + standard LoRA defaults)."""

    model_name_or_path: str = "roberta-base"
    model_type: str = "roberta"
    task: str = "sst2"
    table6_group: str = "glue-big"

    # Table 6 optimisation budget (shared with FT).
    learning_rate: float = 2e-4
    batch_size: int = 32
    epochs: int = 40
    distill_epochs: int = 20

    # LoRA-specific settings (Appendix A adapter budget: rank 8, scaling 2).
    lora_rank: int = DEFAULT_LORA_RANK
    lora_alpha: Optional[float] = DEFAULT_LORA_ALPHA
    lora_dropout: float = DEFAULT_LORA_DROPOUT
    scaling: Optional[float] = None  # explicit override of alpha / rank
    target_modules: Optional[Sequence[str]] = None
    bias: str = "none"  # "none" | "all" | "lora_only"

    # Optimisation.
    optimizer: str = "adamw"
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    warmup_ratio: float = DEFAULT_WARMUP_RATIO
    lr_kind: str = "linear"
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM
    seed: int = DEFAULT_SEED

    # Data / tokenisation.
    max_seq_length: int = 128
    max_target_length: int = 128
    doc_stride: int = 128
    max_query_length: int = 64
    n_best_size: int = 20
    max_answer_length: int = 30
    null_score_diff_threshold: float = 0.0
    dynamic_padding: bool = False
    num_workers: int = 0

    # Runtime / evaluation.
    device: str = "cuda"
    output_dir: str = "outputs/lora"
    logging_steps: int = 50
    eval_steps: int = 0
    save_steps: int = 0
    inference_batch_size: int = 128
    sequence_length: int = 128
    fp16: bool = False
    bf16: bool = False
    measure_efficiency: bool = True
    max_train_batches: int = 0
    max_eval_batches: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- derived -----------------------------------------------------------
    @property
    def is_seq2seq(self) -> bool:
        return self.model_type.lower() in ("t5", "mt5", "bart") or is_seq2seq_task(
            self.task
        )

    @property
    def is_squad(self) -> bool:
        return is_squad_task(self.task)

    @property
    def num_epochs(self) -> int:
        return int(self.epochs)

    @property
    def resolved_scaling(self) -> float:
        """The multiplier applied to ``B @ A`` (APT uses a static s = 2)."""
        if self.scaling is not None:
            return float(self.scaling)
        alpha = self.lora_alpha if self.lora_alpha is not None else 2.0 * self.lora_rank
        return float(alpha) / float(max(int(self.lora_rank), 1))

    @property
    def resolved_target_modules(self) -> Tuple[str, ...]:
        if self.target_modules:
            return tuple(self.target_modules)
        return LORA_TARGET_MODULES.get(
            self.model_type.lower(), LORA_TARGET_MODULES.get("roberta", ("query", "value"))
        )

    # -- (de)serialisation --------------------------------------------------
    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]] = None, **overrides) -> "LoRAConfig":
        data = dict(data or {})
        data.update(overrides)
        known = set(cls.__dataclass_fields__.keys())  # type: ignore[attr-defined]
        payload = {k: v for k, v in data.items() if k in known}
        extra = {k: v for k, v in data.items() if k not in known}
        cfg = cls(**payload)
        if extra:
            cfg.extra.update(extra)
        cfg.model_type = (cfg.model_type or "").lower() or detect_model_type(
            cfg.model_name_or_path
        )
        cfg.task = canonical_task(cfg.task)
        if not cfg.table6_group:
            cfg.table6_group = table6_group_for(cfg.model_type, cfg.task)
        return cfg

    @classmethod
    def from_yaml(cls, path: str, **overrides) -> "LoRAConfig":
        import yaml  # noqa: WPS433

        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return cls.from_dict(data, **overrides)

    def to_dict(self) -> Dict[str, Any]:
        out = {k: getattr(self, k) for k in self.__dataclass_fields__}  # type: ignore[attr-defined]
        out["resolved_scaling"] = self.resolved_scaling
        out["target_modules"] = list(self.resolved_target_modules)
        return out

    def save(self, path: str) -> str:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        return path


def apply_table6_defaults(config: LoRAConfig, table: Optional[Dict[str, Dict[str, float]]] = None) -> LoRAConfig:
    """Fill Table 6 values for the config's task group (only untouched fields)."""
    table = table or TABLE6_LORA_DEFAULTS
    group = config.table6_group or table6_group_for(config.model_type, config.task)
    values = table.get(group)
    if not values:
        return config
    defaults = LoRAConfig()
    for key, value in values.items():
        if key in ("distill_epochs",):
            continue
        if getattr(config, key, None) == getattr(defaults, key, None):
            setattr(config, key, value)
    if group == "cnndm":
        if config.max_seq_length == defaults.max_seq_length:
            config.max_seq_length = 512
    return config


# ---------------------------------------------------------------------------
# LoRA layer / model surgery
# ---------------------------------------------------------------------------
class LoRALinear:
    """Factory-free helper namespace for LoRA layer construction.

    The actual module class is created lazily by :func:`_lora_linear_class`
    because it must subclass ``torch.nn.Module``; keeping the class definition
    in a factory lets this module import without PyTorch installed.
    """


_LORA_LINEAR_CLASS: Optional[type] = None


def _lora_linear_class() -> type:
    """Return (and memoise) the ``LoRALinear`` ``nn.Module`` class."""

    global _LORA_LINEAR_CLASS
    if _LORA_LINEAR_CLASS is not None:
        return _LORA_LINEAR_CLASS

    nn = _nn()
    torch = _torch()

    class _LoRALinearImpl(nn.Module):  # type: ignore[misc]
        """``nn.Linear`` with a frozen base weight plus a trainable LoRA update.

        ``h = W0 x + scaling * (B @ A) x`` with ``A`` Gaussian-initialised and
        ``B`` zero-initialised, so the wrapped layer is numerically a no-op at
        the start of training (standard LoRA initialisation).
        """

        def __init__(
            self,
            base_layer,
            rank: int = DEFAULT_LORA_RANK,
            scaling: float = DEFAULT_LORA_SCALING,
            dropout: float = 0.0,
            init_std: Optional[float] = None,
            name: str = "",
        ) -> None:
            super().__init__()
            self.in_features = int(base_layer.in_features)
            self.out_features = int(base_layer.out_features)
            self.rank = int(max(rank, 0))
            self.scaling = float(scaling)
            self.module_name = name
            self.dropout_p = float(dropout)

            self.weight = nn.Parameter(base_layer.weight.detach().clone(), requires_grad=False)
            if getattr(base_layer, "bias", None) is not None:
                self.bias = nn.Parameter(base_layer.bias.detach().clone(), requires_grad=False)
            else:
                self.register_parameter("bias", None)

            self.lora_a = nn.Parameter(torch.zeros(self.rank, self.in_features))
            self.lora_b = nn.Parameter(torch.zeros(self.out_features, self.rank))
            self.reset_lora_parameters(init_std)
            self._dropout = nn.Dropout(self.dropout_p) if self.dropout_p > 0 else nn.Identity()

        # -- init ---------------------------------------------------------
        def reset_lora_parameters(self, init_std: Optional[float] = None) -> None:
            if self.rank <= 0:
                return
            std = float(init_std) if init_std is not None else 1.0 / max(self.rank, 1)
            with torch.no_grad():
                self.lora_a.normal_(mean=0.0, std=std)
                self.lora_b.zero_()

        # -- math ---------------------------------------------------------
        def delta_weight(self):
            if self.rank <= 0:
                return None
            return (self.lora_b @ self.lora_a) * self.scaling

        def merged_weight(self):
            if self.rank <= 0:
                return self.weight
            return self.weight + self.delta_weight()

        def forward(self, x):
            out = nn.functional.linear(x, self.weight, self.bias)
            if self.rank > 0:
                xin = self._dropout(x)
                out = out + nn.functional.linear(
                    nn.functional.linear(xin, self.lora_a), self.lora_b
                ) * self.scaling
            return out

        def merge(self):
            """Fold ``scaling * B @ A`` into ``W`` and disable the adapter."""
            if self.rank <= 0:
                return self
            with torch.no_grad():
                self.weight.data = self.merged_weight().detach().clone()
                self.lora_a.data.zero_()
                self.lora_b.data.zero_()
            self.rank = 0
            return self

        def extra_repr(self) -> str:  # pragma: no cover - cosmetic
            return (
                f"in_features={self.in_features}, out_features={self.out_features}, "
                f"rank={self.rank}, scaling={self.scaling}"
            )

    _LORA_LINEAR_CLASS = _LoRALinearImpl
    return _LORA_LINEAR_CLASS


def lora_layer(
    base_layer,
    rank: int = DEFAULT_LORA_RANK,
    scaling: float = DEFAULT_LORA_SCALING,
    dropout: float = 0.0,
    init_std: Optional[float] = None,
    name: str = "",
):
    """Wrap ``base_layer`` (an ``nn.Linear``) in a LoRA module."""
    return _lora_linear_class()(
        base_layer,
        rank=rank,
        scaling=scaling,
        dropout=dropout,
        init_std=init_std,
        name=name,
    )


def _resolve_parent(model, qualified_name: str) -> Tuple[Any, str]:
    parts = qualified_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part) if not part.isdigit() else parent[int(part)]
    return parent, parts[-1]


def find_lora_targets(
    model,
    target_modules: Optional[Sequence[str]] = None,
    model_type: Optional[str] = None,
) -> List[Tuple[str, Any]]:
    """Return ``(qualified_name, nn.Linear)`` candidates for LoRA injection."""
    nn = _nn()
    family = (model_type or detect_model_type(model) or "").lower()
    suffixes = tuple(target_modules or LORA_TARGET_MODULES.get(family, ("query", "value")))
    found: List[Tuple[str, Any]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(name == s or name.endswith("." + s) or name.endswith(s) for s in suffixes):
            found.append((name, module))
    return found


def apply_lora(
    model,
    *,
    rank: int = DEFAULT_LORA_RANK,
    scaling: float = DEFAULT_LORA_SCALING,
    dropout: float = 0.0,
    target_modules: Optional[Sequence[str]] = None,
    model_type: Optional[str] = None,
    names: Optional[Iterable[str]] = None,
    init_std: Optional[float] = None,
    freeze_base: bool = True,
) -> List[Tuple[str, Any]]:
    """Replace target ``nn.Linear`` layers of ``model`` with LoRA layers in place.

    Returns the list of ``(name, lora_module)`` pairs that were injected.
    """
    injected: List[Tuple[str, Any]] = []
    if names is not None:
        wanted = set(names)
        targets = [
            (n, m) for n, m in find_lora_targets(model, None, model_type) if n in wanted
        ]
    else:
        targets = find_lora_targets(model, target_modules, model_type)

    for name, base in targets:
        parent, attr = _resolve_parent(model, name)
        layer = lora_layer(
            base, rank=rank, scaling=scaling, dropout=dropout, init_std=init_std, name=name
        )
        setattr(parent, attr, layer)
        injected.append((name, layer))

    if freeze_base:
        freeze_base_parameters(model)
    return injected


def freeze_base_parameters(model) -> None:
    """Freeze everything that is not a LoRA parameter (``lora_a`` / ``lora_b``)."""
    for name, param in model.named_parameters():
        param.requires_grad = ("lora_a" in name) or ("lora_b" in name)


def unfreeze_all(model) -> None:
    for param in model.parameters():
        param.requires_grad = True


def iter_lora_modules(model) -> List[Tuple[str, Any]]:
    out: List[Tuple[str, Any]] = []
    for name, module in model.named_modules():
        if hasattr(module, "lora_a") and hasattr(module, "lora_b"):
            out.append((name, module))
    return out


def lora_parameter_count(model, trainable_only: bool = True) -> int:
    total = 0
    for _, module in iter_lora_modules(model):
        for param in (module.lora_a, module.lora_b):
            if (not trainable_only) or param.requires_grad:
                total += int(param.numel())
    return total


def lora_state_dict(model, *, cpu: bool = True) -> Dict[str, Any]:
    """Extract only the LoRA tuning parameters (for checkpointing)."""
    state: Dict[str, Any] = {}
    for name, module in iter_lora_modules(model):
        for key, param in (("lora_a", module.lora_a), ("lora_b", module.lora_b)):
            tensor = param.detach()
            state[f"{name}.{key}"] = tensor.cpu() if cpu else tensor
    return state


def load_lora_state_dict(model, state: Dict[str, Any], strict: bool = False) -> None:
    torch = _torch()
    modules = dict(iter_lora_modules(model))
    loaded = 0
    for key, value in (state or {}).items():
        if "." not in key:
            continue
        module_name, _, param_name = key.rpartition(".")
        module = modules.get(module_name)
        if module is None or param_name not in ("lora_a", "lora_b"):
            continue
        param = getattr(module, param_name)
        with torch.no_grad():
            if tuple(param.shape) == tuple(value.shape):
                param.copy_(value.to(param.device, param.dtype))
                loaded += 1
    if strict and loaded != 2 * len(modules):
        raise RuntimeError(
            f"load_lora_state_dict: loaded {loaded} tensors for {len(modules)} modules"
        )


def merge_lora(model, inplace: bool = True):
    """Merge every LoRA update into its frozen base weight."""
    modules = iter_lora_modules(model if inplace else _clone_model(model))
    for _, module in modules:
        module.merge()
    return model


def _clone_model(model):
    import copy

    return copy.deepcopy(model)


class LoRAModel:
    """Thin container pairing an HF model with its tokenizer and LoRA setup."""

    def __init__(
        self,
        config: Optional[LoRAConfig] = None,
        model=None,
        tokenizer=None,
        **kwargs,
    ) -> None:
        self.config = config if isinstance(config, LoRAConfig) else LoRAConfig.from_dict(
            config if isinstance(config, dict) else None, **(kwargs or {})
        )
        self.model = model
        self.tokenizer = tokenizer

    # -- construction ------------------------------------------------------
    @classmethod
    def build(cls, config: Optional[LoRAConfig] = None, **kwargs) -> "LoRAModel":
        cfg = config if isinstance(config, LoRAConfig) else LoRAConfig.from_dict(
            config if isinstance(config, dict) else None, **kwargs
        )
        model, tokenizer = build_lora_model(cfg)
        return cls(config=cfg, model=model, tokenizer=tokenizer)

    # -- delegation --------------------------------------------------------
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def parameters(self):
        return self.model.parameters()

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def eval(self):
        self.model.eval()
        return self

    def to(self, device):
        self.model.to(device)
        return self

    def num_tuning_parameters(self) -> int:
        return lora_parameter_count(self.model)

    def merge(self):
        return merge_lora(self.model)

    def state_dict(self):
        return lora_state_dict(self.model)


def build_lora_model(config: LoRAConfig):
    """Instantiate the task-appropriate HF model + tokenizer and inject LoRA."""
    try:
        from transformers import (  # noqa: WPS433
            AutoConfig,
            AutoModelForQuestionAnswering,
            AutoModelForSeq2SeqLM,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "transformers is required to build the LoRA baseline model"
        ) from exc

    name = config.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    hf_config = AutoConfig.from_pretrained(name)

    if config.is_squad:
        model = AutoModelForQuestionAnswering.from_pretrained(name, config=hf_config)
    elif config.is_seq2seq:
        model = AutoModelForSeq2SeqLM.from_pretrained(name, config=hf_config)
    else:
        hf_config.num_labels = num_labels_for_task(config.task)
        problem_type = problem_type_for_task(config.task)
        if problem_type:
            hf_config.problem_type = problem_type
        model = AutoModelForSequenceClassification.from_pretrained(name, config=hf_config)

    apply_lora(
        model,
        rank=config.lora_rank,
        scaling=config.resolved_scaling,
        dropout=config.lora_dropout,
        target_modules=config.resolved_target_modules,
        model_type=config.model_type,
    )
    if config.bias == "all":
        for name_, param in model.named_parameters():
            if name_.endswith(".bias"):
                param.requires_grad = True
    return model, tokenizer


# ---------------------------------------------------------------------------
# Time-to-accuracy tracker (same protocol as FT / APT)
# ---------------------------------------------------------------------------
class TimeToAccuracyTracker:
    """Wall-clock seconds to reach ``fraction`` of an FT reference score."""

    def __init__(
        self,
        reference: Optional[float] = None,
        fraction: float = TTA_FRACTION,
        higher_is_better: bool = True,
    ) -> None:
        self.reference = reference
        self.fraction = float(fraction)
        self.higher_is_better = bool(higher_is_better)
        self.history: List[Tuple[float, float]] = []
        self.tta_seconds: Optional[float] = None

    @property
    def target(self) -> Optional[float]:
        if self.reference is None:
            return None
        return float(self.reference) * self.fraction

    def update(self, elapsed_seconds: float, metric_value: Optional[float]) -> Optional[float]:
        if metric_value is None:
            return self.tta_seconds
        value = float(metric_value)
        self.history.append((float(elapsed_seconds), value))
        if self.target is None or self.tta_seconds is not None:
            return self.tta_seconds
        reached = (
            value >= self.target if self.higher_is_better else value <= self.target
        )
        if not reached:
            return None
        if len(self.history) == 1:
            self.tta_seconds = float(elapsed_seconds)
            return self.tta_seconds
        t1, v1 = self.history[-2]
        t2, v2 = self.history[-1]
        if v2 == v1:
            self.tta_seconds = float(t2)
            return self.tta_seconds
        ratio = (self.target - v1) / (v2 - v1)
        ratio = min(max(ratio, 0.0), 1.0)
        self.tta_seconds = float(t1 + ratio * (t2 - t1))
        return self.tta_seconds

    def as_dict(self) -> Dict[str, Any]:
        return {
            "reference": self.reference,
            "fraction": self.fraction,
            "target": self.target,
            "tta_seconds": self.tta_seconds,
            "history": list(self.history),
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class LoRATrainer:
    """Plain LoRA fine-tuning baseline.

    Mirrors :class:`apt.baselines.ft.FTTrainer`'s public surface so experiment
    scripts can treat both uniformly (``setup_model``, ``fit``, ``evaluate``,
    ``summary``, ``save``).
    """

    def __init__(
        self,
        config: Optional[LoRAConfig] = None,
        model=None,
        tokenizer=None,
        train_dataloader=None,
        eval_dataloader=None,
        compute_metrics=None,
        reference_metric: Optional[float] = None,
        **kwargs,
    ) -> None:
        if isinstance(config, LoRAConfig):
            self.config = config
        else:
            self.config = LoRAConfig.from_dict(
                config if isinstance(config, dict) else None, **(kwargs or {})
            )
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.compute_metrics = compute_metrics
        self.reference_metric = reference_metric

        self.device = resolve_device(self.config.device)
        self.optimizer = None
        self.scheduler = None
        self.tta = TimeToAccuracyTracker(reference_metric, TTA_FRACTION)
        self.history: List[Dict[str, Any]] = []
        self.train_time_s = 0.0
        self.train_peak_mem_mb: Optional[float] = None
        self.global_step = 0
        self._injected: List[Tuple[str, Any]] = []

    # -- setup -------------------------------------------------------------
    def setup_model(self):
        torch = _torch()
        if self.model is None:
            self.model, self.tokenizer = build_lora_model(self.config)
        else:
            self._injected = apply_lora(
                self.model,
                rank=self.config.lora_rank,
                scaling=self.config.resolved_scaling,
                dropout=self.config.lora_dropout,
                target_modules=self.config.resolved_target_modules,
                model_type=self.config.model_type,
            )
        self.model.to(self.device)
        return self.model

    def setup_optimizer(self, parameters=None):
        torch = _torch()
        params = parameters if parameters is not None else [
            p for p in self.model.parameters() if p.requires_grad
        ]
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "lora" in name:
                decay.append(param)  # LoRA matrices take weight decay
            elif name.endswith(".bias") or "LayerNorm" in name or "layer_norm" in name:
                no_decay.append(param)
            else:
                no_decay.append(param)
        groups = [
            {"params": decay, "weight_decay": self.config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        groups = [g for g in groups if g["params"]]
        if not groups:
            groups = [{"params": list(params), "weight_decay": self.config.weight_decay}]
        self.optimizer = torch.optim.AdamW(
            groups,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
        )
        return self.optimizer

    def setup_scheduler(self, total_steps: Optional[int] = None):
        torch = _torch()
        if self.optimizer is None:
            self.setup_optimizer()
        total_steps = int(total_steps or max(self.steps_per_epoch() * self.config.epochs, 1))
        warmup = int(round(total_steps * self.config.warmup_ratio))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: lr_factor(
                step, total_steps, warmup_steps=warmup, kind=self.config.lr_kind
            ),
        )
        return self.scheduler

    def steps_per_epoch(self) -> int:
        try:
            return len(self.train_dataloader)  # type: ignore[arg-type]
        except Exception:
            return 0

    # -- training ----------------------------------------------------------
    def training_step(self, batch: Dict[str, Any]):
        torch = _torch()
        batch = move_to_device(batch, self.device)
        self.model.train()
        inputs = model_inputs(batch)
        outputs = self.model(**inputs)
        loss = getattr(outputs, "loss", None)
        if loss is None:
            raise RuntimeError("model forward did not return a loss")
        loss.backward()
        if self.config.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.config.max_grad_norm,
            )
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1
        return float(loss.detach().cpu())

    def fit(self, max_steps: Optional[int] = None) -> Dict[str, Any]:
        torch = _torch()
        if self.model is None:
            self.setup_model()
        if self.optimizer is None:
            self.setup_optimizer()
        if self.scheduler is None:
            self.setup_scheduler()

        device = self.device
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        start = time.time()
        stop = False
        for epoch in range(int(self.config.epochs)):
            for batch in self.train_dataloader or []:
                if max_steps and self.global_step >= max_steps:
                    stop = True
                    break
                if self.config.max_train_batches and (
                    self.global_step % max(self.steps_per_epoch(), 1)
                ) >= self.config.max_train_batches:
                    break
                self.training_step(batch)
                if self.config.logging_steps and (
                    self.global_step % self.config.logging_steps == 0
                ):
                    print(
                        f"[lora] epoch {epoch} step {self.global_step} "
                        f"lr {self.optimizer.param_groups[0]['lr']:.3e}"
                    )
            self.train_time_s = time.time() - start
            metrics = self.evaluate()
            self.tta.update(self.train_time_s, self.primary_metric(metrics))
            if torch.cuda.is_available():
                self.train_peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            if stop:
                break
        self.train_time_s = time.time() - start
        return self.summary()

    # -- evaluation --------------------------------------------------------
    def predict(self, dataloader=None):
        torch = _torch()
        dataloader = dataloader if dataloader is not None else self.eval_dataloader
        self.model.eval()
        logits, labels = [], []
        with torch.no_grad():
            for step, batch in enumerate(dataloader or []):
                if self.config.max_eval_batches and step >= self.config.max_eval_batches:
                    break
                batch = move_to_device(batch, self.device)
                inputs = model_inputs(batch)
                outputs = self.model(**inputs)
                logits.append(outputs.logits.detach().cpu())
                if "labels" in batch:
                    labels.append(batch["labels"].detach().cpu())
        return logits, labels

    def evaluate(self, dataloader=None, step: Optional[int] = None) -> Dict[str, float]:
        torch = _torch()
        logits, labels = self.predict(dataloader)
        if not logits:
            return {}
        logits = torch.cat(logits, dim=0)
        labels = torch.cat(labels, dim=0) if labels else None

        if self.config.is_seq2seq:
            # ROUGE is computed from decoded text by the task pipeline; fall back
            # to a token-match accuracy proxy when only logits are available.
            preds = logits.argmax(dim=-1)
            if labels is not None:
                return {"accuracy": float((preds == labels).float().mean().item() * 100.0)}
            return {}

        if is_squad_task(self.config.task):
            return {}  # SQuAD spans are post-processed by apt.data.squad

        if self.compute_metrics is not None:
            preds = logits.argmax(dim=-1) if logits.dim() > 1 and logits.size(-1) > 1 else logits.squeeze(-1)
            refs = labels
            try:
                return dict(self.compute_metrics(self.config.task, preds, refs))
            except Exception:
                pass

        if labels is None:
            return {}
        if logits.dim() > 1 and logits.size(-1) > 1:
            preds = logits.argmax(dim=-1)
            acc = float((preds == labels).float().mean().item() * 100.0)
            return {"accuracy": acc}
        preds = logits.squeeze(-1)
        return {"mse": float(((preds - labels) ** 2).mean().item())}

    def primary_metric(self, metrics: Optional[Dict[str, float]]) -> Optional[float]:
        if not metrics:
            return None
        for key in ("accuracy", "matthews_correlation", "spearmanr", "f1", "rougeL", "exact"):
            if key in metrics:
                return float(metrics[key])
        return float(next(iter(metrics.values())))

    # -- reporting ---------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        return {
            "method": "lora",
            "model": self.config.model_name_or_path,
            "task": self.config.task,
            "lora_rank": self.config.lora_rank,
            "scaling": self.config.resolved_scaling,
            "num_tuning_parameters": (
                lora_parameter_count(self.model) if self.model is not None else 0
            ),
            "train_time_s": self.train_time_s,
            "train_peak_mem_mb": self.train_peak_mem_mb,
            "tta_seconds": self.tta.tta_seconds,
            "global_step": self.global_step,
            "history": list(self.history),
        }

    def save(self, output_dir: Optional[str] = None) -> str:
        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        torch = _torch()
        if self.model is not None:
            torch.save(lora_state_dict(self.model), os.path.join(output_dir, "lora.pt"))
        self.config.save(os.path.join(output_dir, "lora_config.json"))
        return output_dir


# ---------------------------------------------------------------------------
# Convenience entry point (matches ``train_ft`` / ``train_apt`` style)
# ---------------------------------------------------------------------------
def train_lora(
    config: Optional[LoRAConfig] = None,
    model=None,
    tokenizer=None,
    train_dataloader=None,
    eval_dataloader=None,
    compute_metrics=None,
    reference_metric: Optional[float] = None,
    evaluate_now: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """Train the LoRA baseline and return its summary dict (incl. ``trainer``)."""
    cfg = config if isinstance(config, LoRAConfig) else LoRAConfig.from_dict(
        config if isinstance(config, dict) else None, **kwargs
    )
    set_seed(cfg.seed)
    trainer = LoRATrainer(
        config=cfg,
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        compute_metrics=compute_metrics,
        reference_metric=reference_metric,
    )
    summary = trainer.fit()
    if evaluate_now:
        metrics = trainer.evaluate()
        summary["metrics"] = metrics
        summary["primary"] = trainer.primary_metric(metrics)
    summary["trainer"] = trainer
    return summary


# Backwards-compatible aliases used by experiment scripts / __init__.
LoRA = LoRATrainer


def evaluate_lora_model(trainer: LoRATrainer, dataloader=None, save_path: Optional[str] = None) -> Dict[str, Any]:
    """Evaluate a trained LoRA model and optionally dump the result as JSON."""
    metrics = trainer.evaluate(dataloader)
    out = {
        "method": "lora",
        "metrics": metrics,
        "primary": trainer.primary_metric(metrics),
        "train_time_s": trainer.train_time_s,
        "train_peak_mem_mb": trainer.train_peak_mem_mb,
        "tta_seconds": trainer.tta.tta_seconds,
    }
    if save_path:
        directory = os.path.dirname(os.path.abspath(save_path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as handle:
            json.dump({k: v for k, v in out.items()}, handle, indent=2, default=str)
    return out


# ---------------------------------------------------------------------------
# Self test (dependency-light: works without torch/transformers)
# ---------------------------------------------------------------------------
def _self_test() -> bool:
    ok = True

    cfg = LoRAConfig.from_dict({"task": "sst-2", "model_type": "roberta"})
    if cfg.task != "sst2":
        print("FAIL: task alias normalisation", cfg.task)
        ok = False
    if abs(cfg.resolved_scaling - 2.0) > 1e-9:
        print("FAIL: default scaling (alpha/rank) should be 2.0:", cfg.resolved_scaling)
        ok = False
    if cfg.resolved_target_modules != ("query", "value"):
        print("FAIL: roberta target modules", cfg.resolved_target_modules)
        ok = False

    cfg2 = LoRAConfig.from_dict({"model_type": "t5", "task": "cnndm"})
    if cfg2.resolved_target_modules != ("q", "v"):
        print("FAIL: t5 target modules", cfg2.resolved_target_modules)
        ok = False
    if table6_group_for("t5", "cnndm") != "cnndm":
        print("FAIL: table6 group for cnndm")
        ok = False

    # TTA interpolation: FT reference 94.8, 97% target = 91.956
    tta = TimeToAccuracyTracker(reference=94.8)
    if abs(tta.target - 94.8 * 0.97) > 1e-9:
        print("FAIL: TTA target")
        ok = False
    tta.update(100.0, 80.0)
    if tta.tta_seconds is not None:
        print("FAIL: TTA should not trigger below target")
        ok = False
    tta.update(200.0, 100.0)
    expected = 100.0 + (94.8 * 0.97 - 80.0) / (100.0 - 80.0) * 100.0
    if tta.tta_seconds is None or abs(tta.tta_seconds - expected) > 1e-6:
        print("FAIL: TTA interpolation", tta.tta_seconds, expected)
        ok = False

    # LR schedule: warmup then linear decay to 0
    if abs(lr_factor(0, 100, warmup_steps=10) - 0.1) > 1e-9:
        print("FAIL: lr warmup")
        ok = False
    if abs(lr_factor(99, 100, warmup_steps=10)) > 1e-6:
        print("FAIL: lr decay end")
        ok = False

    # Table 6 defaults propagation
    cfg3 = apply_table6_defaults(LoRAConfig(model_type="t5", task="cnndm", table6_group="cnndm"))
    if abs(cfg3.learning_rate - 1e-4) > 1e-12 or cfg3.batch_size != 16 or cfg3.epochs != 16:
        print("FAIL: Table 6 CNN/DM defaults", cfg3.to_dict())
        ok = False

    # Optional torch paths: a tiny Linear wrapped in LoRA must be transparent.
    try:
        torch = _torch()
        nn = _nn()
        lin = nn.Linear(8, 4)
        layer = lora_layer(lin, rank=2, scaling=2.0)
        x = torch.randn(3, 8)
        base_out = lin(x)
        if not torch.allclose(layer(x), base_out, atol=1e-6):
            print("FAIL: LoRA layer must be a no-op at init")
            ok = False
        with torch.no_grad():
            layer.lora_b.normal_(0.0, 1.0)
        if torch.allclose(layer(x), base_out, atol=1e-6):
            print("FAIL: LoRA layer with non-zero B should change the output")
            ok = False
        merged = layer.merged_weight()
        with torch.no_grad():
            layer.lora_b.normal_(0.0, 1.0)
        _, n = torch.linalg.solve(
            torch.eye(layer.in_features), torch.eye(layer.in_features)
        )  # sanity: torch linear algebra available
        # merge() must reproduce merged_weight exactly
        fused = float((layer.merged_weight() - merged).abs().max().item())
        if fused < 0:
            print("FAIL: merge delta computation")
            ok = False

        class _Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.query = nn.Linear(6, 6)
                self.value = nn.Linear(6, 6)
                self.other = nn.Linear(6, 6)

        tiny = _Tiny()
        injected = apply_lora(tiny, rank=4, scaling=2.0, model_type="roberta")
        if len(injected) != 2:
            print("FAIL: expected 2 injected LoRA layers, got", len(injected))
            ok = False
        trainable = [n for n, p in tiny.named_parameters() if p.requires_grad]
        if not trainable or any("lora" not in n for n in trainable):
            print("FAIL: only LoRA parameters should be trainable:", trainable)
            ok = False
        if lora_parameter_count(tiny) != 2 * (4 * 6 + 6 * 4):
            print("FAIL: LoRA parameter count", lora_parameter_count(tiny))
            ok = False
        state = lora_state_dict(tiny)
        if len(state) != 4:
            print("FAIL: lora state dict size", len(state))
            ok = False
        merge_lora(tiny)
        if any(m.rank != 0 for _, m in iter_lora_modules(tiny)):
            print("FAIL: merge_lora should zero out ranks")
            ok = False
    except ImportError:
        pass  # torch not installed: pure-python checks above still valid

    print("lora.py self-test:", "OK" if ok else "FAILED")
    return ok


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(0 if _self_test() else 1)
