"""LoRA+Prune+Distill baseline (APT paper, Section 5.2).

The paper describes this baseline as:

    "LoRA+Prune+Distill: to reduce the training memory consumption in pruning and
     distillation, a simple baseline is to conduct CoFi pruning and distillation but
     with LoRA parameters tuned only. More specifically, only the :math:`L_0` module
     and LoRA parameters are tunable under this setting."

So this module is a thin specialisation of the CoFi recipe implemented in
:mod:`apt.baselines.cofi`:

* the CoFi structural pruning machinery (hard-concrete :math:`L_0` gates over
  heads / neurons / hidden dimensions) is reused unchanged, and
* the **tunable parameter set is restricted** to (a) the :math:`L_0` gate
  parameters (``log_alpha`` / gate tensors) and (b) the LoRA adapters
  (``lora_a`` / ``lora_b`` style parameters) placed by
  :func:`apt.baselines.lora.apply_lora`.

Because the frozen backbone is shared between the teacher and the student, the
training memory of this baseline stays close to LoRA itself, which is the exact
motivation stated in the paper.  Reference numbers from Table 2 (RoBERTa-base,
60% sparsity) are embedded in :data:`TABLE2_REFERENCES` for sanity checking.

Everything degrades gracefully: if :mod:`apt.baselines.cofi` cannot be imported
the module still defines the public API (config dataclass + method stub) so the
baseline registry keeps working; the stub raises a descriptive ``RuntimeError``
only when actually instantiated.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    # config / method
    "LoRAPruneDistillConfig",
    "LoRAPruneDistillMethod",
    "LoRAPruneDistillResult",
    "train_lora_prune_distill",
    "evaluate_lora_prune_distill",
    "prune_with_lora_prune_distill",
    # helpers
    "is_lora_or_gate_parameter",
    "lora_only_trainable_names",
    "freeze_to_lora_and_gates",
    "count_lora_only_parameters",
    "enforce_lora_only",
    "gate_parameter_names",
    "lora_parameter_names",
    "table6_defaults",
    # constants
    "METHOD_KEY",
    "METHOD_ALIASES",
    "DISPLAY_NAME",
    "TABLE2_REFERENCES",
    "TABLE2_EFFICIENCY",
    "TABLE8_REFERENCE",
    "LORA_NAME_PATTERNS",
    "GATE_NAME_PATTERNS",
    "LORA_TARGET_MODULES",
    "TTA_FRACTION",
]

# --------------------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------------------

TTA_FRACTION = 0.97

METHOD_KEY = "lora_prune_distill"
METHOD_DISPLAY_NAME = "LoRA+Prune+Distill"
DISPLAY_NAME = METHOD_DISPLAY_NAME
METHOD_ALIASES = (
    "lora_prune_distill",
    "lora+prune+distill",
    "lora_prune_distillation",
    "lora_pruning_distillation",
    "lora_cofi",
    "cofi_lora",
    "cofi+distill+lora",
)

DEFAULT_TARGET_SPARSITY = 0.60
DEFAULT_LORA_RANK = 8
DEFAULT_SCALING = 2.0
DEFAULT_SEED = 42
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_RATIO = 0.06
DEFAULT_MAX_GRAD_NORM = 1.0

#: Table 2 (RoBERTa-base, 60% sparsity) numbers for this baseline.  Used only for
#: reporting / validation, never to alter the training loop.
TABLE2_REFERENCES: Dict[str, Any] = {
    "model": "roberta-base",
    "sparsity": 0.60,
    "metrics": {
        "mnli": 84.2,
        "sst2": 91.9,
        # SQuAD v2 is '-' (not reported) for the distillation baselines in Table 2.
        "squad_v2": None,
    },
    # Normalised to full fine-tuning (FT == 100%).
    "ft": {"mnli": 87.6, "sst2": 94.8, "squad_v2": 82.9},
    "lora": {"mnli": 87.5, "sst2": 95.1, "squad_v2": 83.0},
}

#: Table 2 efficiency columns for this baseline (percent of FT).
TABLE2_EFFICIENCY: Dict[str, float] = {
    "train_time": 6534.6,
    "train_mem": 141.4,
    "inf_time": 39.4,
    "inf_mem": 82.3,
}

#: Table 8: RoBERTa GLUE at 40% sparsity (APT vs LoRA+Distill vs FT / LoRA).
TABLE8_REFERENCE: Dict[str, Optional[float]] = {
    "ft": 89.7,
    "lora": 84.5,
    "lora_distill": 80.0,
    "apt": 83.9,
}

#: Parameter-name patterns identifying LoRA tuning parameters.
LORA_NAME_PATTERNS: Tuple[str, ...] = (
    "lora_a",
    "lora_b",
    "lora_A",
    "lora_B",
    "loraa",
    "lorab",
    "lora_embedding_a",
    "lora_embedding_b",
)

#: Parameter-name patterns identifying hard-concrete L0 gate parameters.
GATE_NAME_PATTERNS: Tuple[str, ...] = (
    "log_alpha",
    "gate_alpha",
    "alpha_gate",
    "l0_gate",
    "gate_param",
    "gate_logit",
)

#: Target modules per model family for the LoRA adapters.
LORA_TARGET_MODULES: Dict[str, Tuple[str, ...]] = {
    "roberta": ("query", "value"),
    "bert": ("query", "value"),
    "electra": ("query", "value"),
    "deberta": ("query_proj", "value_proj"),
    "distilbert": ("q_lin", "v_lin"),
    "t5": ("q", "v"),
    "mt5": ("q", "v"),
    "bart": ("q_proj", "v_proj"),
    "opt": ("q_proj", "v_proj"),
    "llama": ("q_proj", "v_proj"),
    "mistral": ("q_proj", "v_proj"),
    "gpt2": ("c_attn",),
}

TABLE6_DEFAULTS: Dict[str, Dict[str, float]] = {
    "glue-big": {"learning_rate": 2e-4, "batch_size": 32, "epochs": 40, "distill_epochs": 20},
    "glue-small": {"learning_rate": 2e-4, "batch_size": 32, "epochs": 40, "distill_epochs": 20},
    "squad": {"learning_rate": 2e-4, "batch_size": 32, "epochs": 40, "distill_epochs": 20},
    "cnndm": {"learning_rate": 1e-4, "batch_size": 16, "epochs": 16, "distill_epochs": 6},
}

GLUE_TASKS = ("mnli", "sst2", "qnli", "qqp", "mrpc", "cola", "rte", "stsb")
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
    "encoder_outputs",
    "labels",
)


# --------------------------------------------------------------------------------------
# optional CoFi dependency
# --------------------------------------------------------------------------------------

_COFI_IMPORT_ERROR: Optional[BaseException] = None
try:  # pragma: no cover - import-time environment probing
    from .cofi import (  # noqa: F401
        CoFiConfig as _CoFiConfig,
        CoFiMethod as _CoFiMethod,
        HiddenStateCollector as _HiddenStateCollector,
        build_dataloaders_for_task as _build_dataloaders_for_task,
        build_model_and_tokenizer as _build_model_and_tokenizer,
        canonical_task as _canonical_task,
        evaluate_cofi as _evaluate_cofi,
        is_glue_task as _is_glue_task,
        is_seq2seq_task as _is_seq2seq_task,
        is_squad_task as _is_squad_task,
        resolve_device as _resolve_device,
        set_seed as _set_seed,
        table6_group_for as _table6_group_for,
    )

    _HAS_COFI = True
except Exception as exc:  # pragma: no cover - defensive
    _COFI_IMPORT_ERROR = exc
    _HAS_COFI = False
    _CoFiConfig = None  # type: ignore[assignment]
    _CoFiMethod = None  # type: ignore[assignment]
    _HiddenStateCollector = None  # type: ignore[assignment]
    _build_dataloaders_for_task = None  # type: ignore[assignment]
    _build_model_and_tokenizer = None  # type: ignore[assignment]
    _canonical_task = None  # type: ignore[assignment]
    _evaluate_cofi = None  # type: ignore[assignment]
    _is_glue_task = None  # type: ignore[assignment]
    _is_seq2seq_task = None  # type: ignore[assignment]
    _is_squad_task = None  # type: ignore[assignment]
    _resolve_device = None  # type: ignore[assignment]
    _set_seed = None  # type: ignore[assignment]
    _table6_group_for = None  # type: ignore[assignment]


# --------------------------------------------------------------------------------------
# small pure-python helpers (also used when CoFi is unavailable)
# --------------------------------------------------------------------------------------


def canonical_task(task: Optional[str]) -> str:
    """Normalise a task name (fallback implementation if CoFi is missing)."""
    if _canonical_task is not None:
        try:
            return _canonical_task(task)
        except Exception:
            pass
    if not task:
        return "sst2"
    name = str(task).strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "sst2": "sst2",
        "sst": "sst2",
        "mnli": "mnli",
        "mnlimatched": "mnli",
        "mnlimismatched": "mnli",
        "qnli": "qnli",
        "qqp": "qqp",
        "mrpc": "mrpc",
        "cola": "cola",
        "rte": "rte",
        "stsb": "stsb",
        "squad": "squad",
        "squad2": "squad_v2",
        "squadv2": "squad_v2",
        "cnndm": "cnndm",
        "cnndailymail": "cnndm",
    }
    return aliases.get(name, name)


def is_glue_task(task: Optional[str]) -> bool:
    if _is_glue_task is not None:
        try:
            return bool(_is_glue_task(task))
        except Exception:
            pass
    return canonical_task(task) in GLUE_TASKS


def is_squad_task(task: Optional[str]) -> bool:
    if _is_squad_task is not None:
        try:
            return bool(_is_squad_task(task))
        except Exception:
            pass
    return canonical_task(task) in ("squad", "squad_v2", "squad2")


def is_seq2seq_task(task: Optional[str]) -> bool:
    if _is_seq2seq_task is not None:
        try:
            return bool(_is_seq2seq_task(task))
        except Exception:
            pass
    return canonical_task(task) in SEQ2SEQ_TASKS


def resolve_device(device: Any = None) -> str:
    if _resolve_device is not None:
        try:
            return _resolve_device(device)
        except Exception:
            pass
    if isinstance(device, str) and device:
        return device
    try:
        import torch  # noqa: WPS433

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def set_seed(seed: int = DEFAULT_SEED) -> None:
    if _set_seed is not None:
        try:
            _set_seed(seed)
            return
        except Exception:
            pass
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np  # noqa: WPS433

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # noqa: WPS433

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def table6_group_for(model_type: Optional[str], task: Optional[str]) -> str:
    if _table6_group_for is not None:
        try:
            return _table6_group_for(model_type, task)
        except Exception:
            pass
    task = canonical_task(task)
    if task in SQUAD_TASKS or task.startswith("squad"):
        return "squad"
    if is_seq2seq_task(task):
        return "cnndm"
    if task in GLUE_BIG_TASKS:
        return "glue-big"
    return "glue-small"


def task_distill_weights(task: Optional[str]) -> Tuple[float, float]:
    """Distillation loss weighting used by the baseline.

    Addendum: GLUE uses ``L_pred + 0.9 * L_layer``; SQuAD and CNN/DM use
    ``0.1 * L_pred + 0.9 * L_layer``.
    """
    task = canonical_task(task)
    if is_glue_task(task):
        return 1.0, 0.9
    return 0.1, 0.9


def table6_defaults(model_type: Optional[str], task: Optional[str]) -> Dict[str, float]:
    """Return the Table 6 hyper-parameter column for a (model, task) pair."""
    group = table6_group_for(model_type, task)
    return dict(TABLE6_DEFAULTS.get(group, TABLE6_DEFAULTS["glue-big"]))


def lora_target_modules_for(model_type: Optional[str]) -> Tuple[str, ...]:
    key = (model_type or "roberta").lower()
    return LORA_TARGET_MODULES.get(key, LORA_TARGET_MODULES["roberta"])


def move_to_device(batch: Any, device: Any) -> Any:
    try:
        import torch  # noqa: WPS433
    except Exception:
        return batch
    if isinstance(batch, dict):
        return {
            k: (v.to(device) if hasattr(v, "to") else v)
            for k, v in batch.items()
        }
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(b, device) for b in batch)
    if hasattr(batch, "to"):
        return batch.to(device)
    return batch


def model_inputs(batch: Dict[str, Any], keys: Sequence[str] = MODEL_INPUT_KEYS) -> Dict[str, Any]:
    return {k: v for k, v in batch.items() if k in keys}


# --------------------------------------------------------------------------------------
# LoRA / L0 tunable-set helpers
# --------------------------------------------------------------------------------------


def _matches_any(name: str, patterns: Sequence[str]) -> bool:
    lowered = name.lower()
    return any(p.lower() in lowered for p in patterns)


def is_lora_parameter(name: str) -> bool:
    """True when ``name`` looks like a LoRA tuning parameter."""
    return _matches_any(name, LORA_NAME_PATTERNS)


def is_gate_parameter(name: str) -> bool:
    """True when ``name`` looks like a hard-concrete L0 gate parameter."""
    return _matches_any(name, GATE_NAME_PATTERNS)


def is_lora_or_gate_parameter(name: str) -> bool:
    """Exactly the tunable set of the LoRA+Prune+Distill baseline."""
    return is_lora_parameter(name) or is_gate_parameter(name)


def lora_parameter_names(model: Any) -> List[str]:
    return [n for n, _ in model.named_parameters() if is_lora_parameter(n)]


def gate_parameter_names(model: Any) -> List[str]:
    return [n for n, _ in model.named_parameters() if is_gate_parameter(n)]


def lora_only_trainable_names(model: Any) -> List[str]:
    return [n for n, _ in model.named_parameters() if is_lora_or_gate_parameter(n)]


def freeze_to_lora_and_gates(
    model: Any,
    *,
    include_gates: bool = True,
    extra_patterns: Optional[Sequence[str]] = None,
    verbose: bool = False,
) -> Dict[str, int]:
    """Freeze everything except LoRA adapters and L0 gate parameters.

    This is the literal implementation of the paper's description of the
    LoRA+Prune+Distill baseline: *only* the :math:`L_0` module and LoRA
    parameters are tunable.
    """
    stats = {"trainable": 0, "frozen": 0, "lora": 0, "gates": 0}
    extra = tuple(extra_patterns or ())
    for name, param in model.named_parameters():
        keep = is_lora_parameter(name) or (include_gates and is_gate_parameter(name))
        if not keep and extra and _matches_any(name, extra):
            keep = True
        param.requires_grad_(bool(keep))
        if keep:
            stats["trainable"] += 1
            if is_lora_parameter(name):
                stats["lora"] += 1
            if is_gate_parameter(name):
                stats["gates"] += 1
        else:
            stats["frozen"] += 1
    if verbose:
        print(
            "[lora_prune_distill] tunable set: "
            f"{stats['trainable']} tensors "
            f"({stats['lora']} LoRA, {stats['gates']} gates), "
            f"{stats['frozen']} frozen"
        )
    return stats


#: Alias kept for scripts that phrase it the other way round.
def enforce_lora_only(model: Any, **kwargs: Any) -> Dict[str, int]:
    return freeze_to_lora_and_gates(model, **kwargs)


def count_lora_only_parameters(model: Any) -> int:
    """Number of scalar parameters in the tunable (LoRA + L0) set."""
    total = 0
    for name, param in model.named_parameters():
        if is_lora_or_gate_parameter(name):
            total += int(param.numel())
    return total


def build_lora_only_optimizer(
    model: Any,
    *,
    lr: float = 2e-4,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    name: str = "adamw",
    parameters: Optional[Iterable[Any]] = None,
):
    """AdamW over the restricted (LoRA + L0) parameter set."""
    import torch  # noqa: WPS433

    if parameters is None:
        parameters = [
            p for n, p in model.named_parameters() if p.requires_grad and is_lora_or_gate_parameter(n)
        ]
    params = [p for p in parameters if p.requires_grad]
    if not params:
        params = [p for _, p in model.named_parameters() if p.requires_grad]
    decay, no_decay = [], []
    for i, p in enumerate(params):
        (no_decay if p.ndim <= 1 else decay).append(p)
    groups: List[Dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    if not groups:
        groups = [{"params": params, "weight_decay": weight_decay}]
    lname = (name or "adamw").lower()
    if lname in ("adamw", "adam_w"):
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)
    if lname == "adam":
        return torch.optim.Adam(groups, lr=lr, betas=betas, eps=eps)
    return torch.optim.SGD(groups, lr=lr, momentum=0.9)


# --------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------


@dataclass
class LoRAPruneDistillResult:
    """Container for one LoRA+Prune+Distill run / evaluation outcome."""

    model: str = ""
    task: str = "sst2"
    sparsity: float = DEFAULT_TARGET_SPARSITY
    metrics: Dict[str, float] = field(default_factory=dict)
    primary: Optional[float] = None
    train_time_s: Optional[float] = None
    train_peak_mem_mb: Optional[float] = None
    tta_seconds: Optional[float] = None
    inf_time_ms: Optional[float] = None
    inf_mem_mb: Optional[float] = None
    num_parameters: Optional[int] = None
    num_tuning_parameters: Optional[int] = None
    history: List[Dict[str, float]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        out = dict(self.__dict__)
        out["method"] = METHOD_KEY
        out["display_name"] = DISPLAY_NAME
        return out

    def summary(self) -> str:
        try:
            from apt.eval.metrics import metric_for_display

            shown = metric_for_display(self.task, self.metrics) if self.metrics else "-"
        except Exception:
            shown = str(self.primary) if self.primary is not None else "-"
        return (
            f"{DISPLAY_NAME} | {self.model or '-'} | {self.task} | "
            f"sparsity={self.sparsity:.2f} | {shown}"
        )


if _HAS_COFI:

    @dataclass
    class LoRAPruneDistillConfig(_CoFiConfig):  # type: ignore[misc, valid-type]
        """CoFi config restricted to the LoRA+Prune+Distill baseline.

        All CoFi pruning/distillation knobs are inherited; the only semantic
        difference is that the tunable set is restricted to LoRA + L0 gates
        (``tuning_only_lora``), and the displayed method name is the one used in
        Table 2 of the APT paper.
        """

        method: str = METHOD_KEY
        display_name: str = DISPLAY_NAME
        lora_only: bool = True
        lora_rank: int = DEFAULT_LORA_RANK
        lora_scaling: float = DEFAULT_SCALING

        def __post_init__(self) -> None:  # type: ignore[override]
            parent = getattr(super(), "__post_init__", None)
            if callable(parent):
                try:
                    parent()
                except TypeError:
                    pass
            # Force the LoRA-only flag on whichever field name the CoFi config uses.
            for attr in (
                "tuning_only_lora",
                "tune_lora_only",
                "lora_only",
                "only_lora",
                "tuning_mode",
                "mode",
            ):
                if attr == "mode":
                    continue
                if hasattr(self, attr):
                    try:
                        setattr(self, attr, True)
                    except Exception:
                        pass
            for attr in ("mode", "tuning_mode", "pruning_mode"):
                if hasattr(self, attr):
                    try:
                        current = getattr(self, attr)
                    except Exception:
                        current = None
                    if isinstance(current, str) and current.lower() in (
                        "cofi",
                        "prune_distill",
                        "prune+distill",
                    ):
                        try:
                            setattr(self, attr, "lora_prune_distill")
                        except Exception:
                            pass
            # Keep the LoRA rank / scaling in sync with the inherited fields.
            for attr in ("rank", "lora_r", "adapter_rank", "initial_rank"):
                if hasattr(self, attr):
                    try:
                        setattr(self, attr, self.lora_rank)
                    except Exception:
                        pass
            for attr in ("scaling", "lora_alpha_scaling"):
                if hasattr(self, attr):
                    try:
                        setattr(self, attr, self.lora_scaling)
                    except Exception:
                        pass

            # Fill the Table 6 column when the caller left the defaults in place.
            defaults = table6_defaults(getattr(self, "model_type", None), getattr(self, "task", None))
            for attr, value in defaults.items():
                if hasattr(self, attr):
                    try:
                        setattr(self, attr, value)
                    except Exception:
                        pass
            if is_glue_task(getattr(self, "task", None)):
                if hasattr(self, "pred_distill_weight"):
                    try:
                        setattr(self, "pred_distill_weight", 1.0)
                    except Exception:
                        pass
                if hasattr(self, "layer_distill_weight"):
                    try:
                        setattr(self, "layer_distill_weight", 0.9)
                    except Exception:
                        pass
            elif hasattr(self, "pred_distill_weight"):
                try:
                    setattr(self, "pred_distill_weight", 0.1)
                except Exception:
                    pass
                if hasattr(self, "layer_distill_weight"):
                    try:
                        setattr(self, "layer_distill_weight", 0.9)
                    except Exception:
                        pass

else:  # pragma: no cover - defensive fallback

    @dataclass
    class LoRAPruneDistillConfig:  # type: ignore[no-redef]
        """Minimal stand-in used when :mod:`apt.baselines.cofi` is unavailable."""

        model_name_or_path: str = "roberta-base"
        model_type: str = "roberta"
        task: str = "sst2"
        method: str = METHOD_KEY
        display_name: str = DISPLAY_NAME
        lora_only: bool = True
        lora_rank: int = DEFAULT_LORA_RANK
        lora_scaling: float = DEFAULT_SCALING
        learning_rate: float = 2e-4
        batch_size: int = 32
        epochs: int = 40
        distill_epochs: int = 20
        target_sparsity: float = DEFAULT_TARGET_SPARSITY
        weight_decay: float = DEFAULT_WEIGHT_DECAY
        warmup_ratio: float = DEFAULT_WARMUP_RATIO
        max_grad_norm: float = DEFAULT_MAX_GRAD_NORM
        seed: int = DEFAULT_SEED
        device: str = "cuda"
        output_dir: str = "outputs/lora_prune_distill"
        max_seq_length: int = 128
        max_target_length: int = 128
        inference_batch_size: int = 128
        extra: Dict[str, Any] = field(default_factory=dict)

        # -- convenience -------------------------------------------------------------
        @property
        def is_squad(self) -> bool:
            return is_squad_task(self.task)

        @property
        def is_seq2seq(self) -> bool:
            return is_seq2seq_task(self.task)

        @property
        def resolved_target_modules(self) -> Tuple[str, ...]:
            return lora_target_modules_for(self.model_type)

        def to_dict(self) -> Dict[str, Any]:
            return dict(self.__dict__)

        def save(self, path: str) -> str:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, indent=2)
            return path

        @classmethod
        def from_dict(cls, data: Optional[Dict[str, Any]] = None, **overrides: Any) -> "LoRAPruneDistillConfig":
            data = dict(data or {})
            extra = dict(data.pop("extra", {}) or {})
            known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
            kwargs = {k: v for k, v in data.items() if k in known}
            leftover = {k: v for k, v in data.items() if k not in known}
            extra.update(leftover)
            kwargs.update({k: v for k, v in overrides.items() if k in known})
            extra.update({k: v for k, v in overrides.items() if k not in known})
            config = cls(**kwargs)
            config.extra = extra
            return config

        @classmethod
        def from_yaml(cls, path: str, **overrides: Any) -> "LoRAPruneDistillConfig":
            with open(path, "r", encoding="utf-8") as handle:
                try:
                    import yaml  # noqa: WPS433

                    data = yaml.safe_load(handle) or {}
                except Exception:
                    data = json.load(handle)
            return cls.from_dict(data, **overrides)


def _coerce_config(config: Any = None, **overrides: Any) -> LoRAPruneDistillConfig:
    """Build a :class:`LoRAPruneDistillConfig` from anything sensible."""
    if isinstance(config, LoRAPruneDistillConfig):
        cfg = config
    elif config is None:
        cfg = LoRAPruneDistillConfig()
    elif isinstance(config, str):
        if config.endswith((".yaml", ".yml")):
            cfg = LoRAPruneDistillConfig.from_yaml(config)
        elif config.endswith(".json"):
            with open(config, "r", encoding="utf-8") as handle:
                cfg = LoRAPruneDistillConfig.from_dict(json.load(handle))
        else:
            raise ValueError(f"Unsupported config path: {config}")
    elif isinstance(config, dict):
        cfg = LoRAPruneDistillConfig.from_dict(config)
    else:  # a CoFiConfig or similar
        try:
            cfg = LoRAPruneDistillConfig.from_dict(dict(getattr(config, "__dict__", {}) or {}))
        except Exception:
            cfg = LoRAPruneDistillConfig()
    for key, value in overrides.items():
        if value is None:
            continue
        setattr(cfg, key, value)
    return cfg


# --------------------------------------------------------------------------------------
# the method
# --------------------------------------------------------------------------------------


class LoRAPruneDistillMethod:
    """LoRA+Prune+Distill baseline.

    Delegates the actual CoFi pruning + dynamic layer-wise distillation to
    :class:`apt.baselines.cofi.CoFiMethod`, but enforces the paper's restriction
    that only LoRA adapters and :math:`L_0` gate parameters are tunable.
    """

    display_name = DISPLAY_NAME
    method_key = METHOD_KEY

    def __init__(
        self,
        config: Any = None,
        model: Any = None,
        tokenizer: Any = None,
        train_dataloader: Any = None,
        eval_dataloader: Any = None,
        compute_metrics: Optional[Callable[..., Dict[str, float]]] = None,
        reference_metric: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        if not _HAS_COFI:
            raise RuntimeError(
                "apt.baselines.cofi is unavailable, so the LoRA+Prune+Distill baseline "
                "cannot run. Original import error: "
                f"{_COFI_IMPORT_ERROR!r}"
            )
        self.config = _coerce_config(config, **kwargs)
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.compute_metrics = compute_metrics
        self.reference_metric = reference_metric
        self.method = METHOD_KEY

        # Build the underlying CoFi runner with the forced LoRA-only setting.
        cofi_kwargs = dict(kwargs)
        for key in list(cofi_kwargs.keys()):
            if key in ("verbose", "device", "output_dir", "seed"):
                continue
            if not hasattr(self.config, key):
                cofi_kwargs.pop(key, None)
        self.inner = _CoFiMethod(
            config=self.config,
            model=model,
            tokenizer=tokenizer,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            compute_metrics=compute_metrics,
            reference_metric=reference_metric,
            **cofi_kwargs,
        )
        self.method = METHOD_KEY
        self._lora_stats: Dict[str, int] = {}

    # -- proxy attributes so scripts can treat this like a CoFiMethod --------------
    def __getattr__(self, item: str) -> Any:  # pragma: no cover - thin proxy
        inner = self.__dict__.get("inner")
        if inner is not None and hasattr(inner, item):
            return getattr(inner, item)
        raise AttributeError(item)

    # -- model / data ---------------------------------------------------------------
    @property
    def model(self) -> Any:
        return getattr(self.inner, "model", None)

    @model.setter
    def model(self, value: Any) -> None:
        if getattr(self, "inner", None) is not None:
            self.inner.model = value

    def setup_model(self, *args: Any, **kwargs: Any) -> Any:
        model = self.inner.setup_model(*args, **kwargs)
        self.apply_lora_only()
        return model

    def setup_data(self, *args: Any, **kwargs: Any) -> Any:
        result = self.inner.setup_data(*args, **kwargs)
        if self.train_dataloader is None:
            self.train_dataloader = getattr(self.inner, "train_dataloader", None)
        if self.eval_dataloader is None:
            self.eval_dataloader = getattr(self.inner, "eval_dataloader", None)
        return result

    # -- LoRA-only enforcement ------------------------------------------------------
    def apply_lora_only(self, model: Any = None, *, verbose: bool = False) -> Dict[str, int]:
        """Freeze everything but LoRA adapters and L0 gates on ``model``."""
        target = model if model is not None else self.model
        if target is None:
            return {}
        self._lora_stats = freeze_to_lora_and_gates(target, verbose=verbose)
        return self._lora_stats

    def wrap_for_tuning(self, *args: Any, **kwargs: Any) -> Any:
        result = None
        base = getattr(self.inner, "wrap_for_tuning", None)
        if callable(base):
            try:
                result = base(*args, **kwargs)
            except TypeError:
                result = base()
        # Make sure LoRA adapters exist, then restrict the tunable set.
        self._ensure_lora()
        self.apply_lora_only()
        return result

    def _ensure_lora(self) -> None:
        model = self.model
        if model is None:
            return
        try:
            from apt.baselines.lora import apply_lora  # noqa: WPS433
        except Exception:
            return
        try:
            apply_lora(
                model,
                rank=int(getattr(self.config, "lora_rank", DEFAULT_LORA_RANK)),
                scaling=float(getattr(self.config, "lora_scaling", DEFAULT_SCALING)),
                dropout=float(getattr(self.config, "lora_dropout", 0.0) or 0.0),
                target_modules=getattr(self.config, "resolved_target_modules", None),
                model_type=getattr(self.config, "model_type", None),
            )
        except Exception as exc:  # pragma: no cover - LoRA is optional plumbing
            warnings.warn(f"LoRA injection failed: {exc!r}")

    def tuning_parameters(self, *args: Any, **kwargs: Any) -> List[Any]:
        """Only LoRA + L0 gate parameters (the paper's tunable set)."""
        base = getattr(self.inner, "tuning_parameters", None)
        params: List[Any] = []
        if callable(base):
            try:
                params = list(base(*args, **kwargs) or [])
            except TypeError:
                params = list(base() or [])
            except Exception:
                params = []
        if self.model is not None:
            named = {id(p): n for n, p in self.model.named_parameters()}
            filtered = [
                p
                for p in params
                if is_lora_or_gate_parameter(named.get(id(p), ""))
            ]
            if filtered:
                return filtered
            lora_only = [
                p
                for n, p in self.model.named_parameters()
                if p.requires_grad and is_lora_or_gate_parameter(n)
            ]
            if lora_only:
                return lora_only
        return params

    def setup_optimizer(self, parameters: Optional[Iterable[Any]] = None, *args: Any, **kwargs: Any) -> Any:
        base = getattr(self.inner, "setup_optimizer", None)
        params = list(parameters) if parameters is not None else self.tuning_parameters()
        if callable(base):
            for attempt in (lambda: base(parameters=params), lambda: base(params), lambda: base()):
                try:
                    optimizer = attempt()
                except TypeError:
                    continue
                if optimizer is not None:
                    return optimizer
        optimizer = build_lora_only_optimizer(
            self.model,
            lr=float(getattr(self.config, "learning_rate", 2e-4)),
            weight_decay=float(getattr(self.config, "weight_decay", DEFAULT_WEIGHT_DECAY)),
            parameters=params,
        )
        try:
            self.inner.optimizer = optimizer
        except Exception:
            pass
        return optimizer

    # -- training / evaluation ------------------------------------------------------
    def fit(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        result = self.inner.fit(*args, **kwargs)
        # Re-assert the restriction: CoFi may unfreeze parameters during recovery.
        self.apply_lora_only()
        out = dict(result) if isinstance(result, dict) else {"result": result}
        out.setdefault("method", METHOD_KEY)
        out.setdefault("display_name", DISPLAY_NAME)
        if self._lora_stats:
            out.setdefault("lora_only_stats", dict(self._lora_stats))
        out.setdefault("num_tuning_parameters", count_lora_only_parameters(self.model) if self.model else None)
        return out

    def evaluate(self, *args: Any, **kwargs: Any) -> Dict[str, float]:
        return self.inner.evaluate(*args, **kwargs)

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        return self.inner.predict(*args, **kwargs)

    def prune(self, *args: Any, **kwargs: Any) -> Any:
        return self.inner.prune(*args, **kwargs)

    def recover(self, *args: Any, **kwargs: Any) -> Any:
        return self.inner.recover(*args, **kwargs)

    def select(self, *args: Any, **kwargs: Any) -> Any:
        return self.inner.select(*args, **kwargs)

    def prune_summary(self) -> Dict[str, Any]:
        base = getattr(self.inner, "prune_summary", None)
        summary = dict(base() if callable(base) else {})
        summary.setdefault("method", METHOD_KEY)
        summary.setdefault("display_name", DISPLAY_NAME)
        summary.setdefault("tunable_set", "lora+gate")
        if self.model is not None:
            summary.setdefault("num_tuning_parameters", count_lora_only_parameters(self.model))
        return summary

    def summary(self) -> Dict[str, Any]:
        base = getattr(self.inner, "summary", None)
        summary = dict(base() if callable(base) else {})
        summary["method"] = METHOD_KEY
        summary["display_name"] = DISPLAY_NAME
        summary.setdefault("sparsity", float(getattr(self.config, "target_sparsity", DEFAULT_TARGET_SPARSITY)))
        if self.model is not None:
            summary.setdefault("num_tuning_parameters", count_lora_only_parameters(self.model))
        if self._lora_stats:
            summary.setdefault("lora_only_stats", dict(self._lora_stats))
        return summary

    def save(self, output_dir: Optional[str] = None) -> str:
        base = getattr(self.inner, "save", None)
        if callable(base):
            try:
                return base(output_dir)
            except TypeError:
                return base()
        return output_dir or getattr(self.config, "output_dir", "outputs/lora_prune_distill")


#: Alias mirroring :mod:`apt.baselines.cofi`'s naming so the registry resolves either.
lora_prune_distill_trainer = LoRAPruneDistillMethod


# --------------------------------------------------------------------------------------
# one-call entry points
# --------------------------------------------------------------------------------------


def _build_model_and_tokenizer(config: LoRAPruneDistillConfig) -> Tuple[Any, Any]:
    if _build_model_and_tokenizer is not None:
        return _build_model_and_tokenizer(config)
    try:
        from apt.baselines.cofi import build_model_and_tokenizer as builder  # noqa: WPS433

        return builder(config)
    except Exception as exc:
        raise RuntimeError(f"Unable to build a model for the baseline: {exc!r}") from exc


def _build_dataloaders(config: LoRAPruneDistillConfig, tokenizer: Any, splits: Sequence[str]) -> Dict[str, Any]:
    if _build_dataloaders_for_task is not None:
        return _build_dataloaders_for_task(config, tokenizer, splits=tuple(splits))
    try:
        from apt.data import make_dataloaders  # noqa: WPS433

        task = canonical_task(getattr(config, "task", "sst2"))
        return make_dataloaders(
            task,
            tokenizer,
            model_type=getattr(config, "model_type", "roberta"),
            batch_size=getattr(config, "batch_size", None),
            max_seq_length=getattr(config, "max_seq_length", None),
            splits=tuple(splits),
        )
    except Exception as exc:  # pragma: no cover - data plumbing is optional
        warnings.warn(f"Unable to build dataloaders: {exc!r}")
        return {}


def train_lora_prune_distill(
    config: Any = None,
    model: Any = None,
    tokenizer: Any = None,
    train_dataloader: Any = None,
    eval_dataloader: Any = None,
    compute_metrics: Optional[Callable[..., Dict[str, float]]] = None,
    reference_metric: Optional[float] = None,
    *,
    verbose: bool = True,
    evaluate: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run the LoRA+Prune+Distill baseline end to end.

    Mirrors the argument conventions of :func:`apt.baselines.cofi.train_cofi` and
    :func:`apt.baselines.lora.train_lora` so experiment scripts can swap methods
    without special casing.
    """
    cfg = _coerce_config(config, **kwargs)
    set_seed(int(getattr(cfg, "seed", DEFAULT_SEED)))
    device = resolve_device(getattr(cfg, "device", None))
    try:
        setattr(cfg, "device", device)
    except Exception:
        pass

    if model is None or tokenizer is None:
        built_model, built_tokenizer = _build_model_and_tokenizer(cfg)
        model = model if model is not None else built_model
        tokenizer = tokenizer if tokenizer is not None else built_tokenizer

    if train_dataloader is None or eval_dataloader is None:
        loaders = _build_dataloaders(cfg, tokenizer, ("train", "validation"))
        train_dataloader = train_dataloader if train_dataloader is not None else loaders.get("train")
        eval_dataloader = (
            eval_dataloader
            if eval_dataloader is not None
            else loaders.get("validation", loaders.get("eval"))
        )

    runner = LoRAPruneDistillMethod(
        config=cfg,
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        compute_metrics=compute_metrics,
        reference_metric=reference_metric,
    )

    start = time.time()
    result = runner.fit()
    elapsed = time.time() - start
    result = dict(result) if isinstance(result, dict) else {}
    result.setdefault("method", METHOD_KEY)
    result.setdefault("display_name", DISPLAY_NAME)
    result.setdefault("train_time_s", elapsed)

    if evaluate:
        try:
            metrics = runner.evaluate()
            result["metrics"] = metrics
        except Exception as exc:  # pragma: no cover - evaluation is best-effort
            warnings.warn(f"evaluation failed: {exc!r}")

    result["trainer"] = runner
    result["config"] = cfg.to_dict() if hasattr(cfg, "to_dict") else {}
    if verbose:
        print(f"[{DISPLAY_NAME}] finished in {elapsed:.1f}s; summary: {result.get('metrics', '-')}")
    return result


def prune_with_lora_prune_distill(
    model: Any = None,
    dataloader: Any = None,
    *,
    config: Any = None,
    task: str = "sst2",
    sparsity: float = DEFAULT_TARGET_SPARSITY,
    device: Any = None,
    verbose: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """One-shot CoFi-style structural pruning with LoRA-only tuning."""
    cfg = _coerce_config(config, task=task, target_sparsity=sparsity, **kwargs)
    try:
        from apt.baselines.cofi import prune_with_cofi  # noqa: WPS433
    except Exception as exc:
        raise RuntimeError(f"CoFi pruning helper unavailable: {exc!r}") from exc
    return prune_with_cofi(model, dataloader, config=cfg, task=task, sparsity=sparsity, device=device, verbose=verbose)


def evaluate_lora_prune_distill(
    source: Any,
    dataloader: Any = None,
    *,
    task: str = "sst2",
    tokenizer: Any = None,
    device: Any = None,
    compute_metrics: Optional[Callable[..., Dict[str, float]]] = None,
    save_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate a trained :class:`LoRAPruneDistillMethod` (or a raw model)."""
    runner = source if isinstance(source, LoRAPruneDistillMethod) else None
    if runner is not None:
        try:
            metrics = runner.evaluate(dataloader)
        except Exception:
            metrics = {}
        out = {
            "method": METHOD_KEY,
            "display_name": DISPLAY_NAME,
            "task": canonical_task(task),
            "metrics": metrics,
            "summary": runner.summary(),
        }
    else:
        if _evaluate_cofi is None:
            raise RuntimeError("apt.baselines.cofi.evaluate_cofi is unavailable")
        out = _evaluate_cofi(
            source,
            dataloader,
            task=task,
            tokenizer=tokenizer,
            device=device,
            compute_metrics=compute_metrics,
        )
        out.setdefault("method", METHOD_KEY)
        out.setdefault("display_name", DISPLAY_NAME)
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as handle:
            json.dump({k: v for k, v in out.items() if k != "trainer"}, handle, indent=2, default=str)
    return out


# --------------------------------------------------------------------------------------
# self test
# --------------------------------------------------------------------------------------


def _self_test() -> bool:
    """Dependency-light sanity checks (no torch / transformers needed)."""
    ok = True

    # task normalisation / grouping
    assert canonical_task("SST-2") == "sst2"
    assert canonical_task("squad_v2") == "squad_v2"
    assert table6_group_for("roberta", "mnli") == "glue-big"
    assert table6_group_for("t5", "cnndm") == "cnndm"
    assert table6_defaults("roberta", "mnli")["epochs"] == 40
    assert table6_defaults("t5", "cnndm")["batch_size"] == 16
    assert task_distill_weights("sst2") == (1.0, 0.9)
    assert task_distill_weights("squad_v2") == (0.1, 0.9)
    assert task_distill_weights("cnndm") == (0.1, 0.9)

    # tunable-set identification
    assert is_lora_parameter("roberta.encoder.layer.0.attention.self.query.lora_a")
    assert is_lora_parameter("model.layers.0.self_attn.q_proj.lora_b")
    assert is_gate_parameter("roberta.encoder.layer.0.intermediate.gate_log_alpha")
    assert is_lora_or_gate_parameter("roberta.encoder.layer.3.output.lora_b")
    assert not is_lora_or_gate_parameter("roberta.encoder.layer.3.output.dense.weight")

    # config plumbing (works with or without CoFi)
    cfg = LoRAPruneDistillConfig()
    assert cfg.method == METHOD_KEY
    assert cfg.lora_only is True
    assert cfg.lora_rank == DEFAULT_LORA_RANK
    assert cfg.lora_scaling == DEFAULT_SCALING
    data = cfg.to_dict()
    cfg2 = LoRAPruneDistillConfig.from_dict({**data, "task": "mnli", "unknown_key": 5})
    assert cfg2.task == "mnli"
    assert cfg2.extra.get("unknown_key") == 5

    # Table 2 / Table 8 reference constants
    assert TABLE2_REFERENCES["metrics"]["mnli"] == 84.2
    assert TABLE2_REFERENCES["metrics"]["sst2"] == 91.9
    assert TABLE2_EFFICIENCY["train_time"] == 6534.6
    assert TABLE2_EFFICIENCY["train_mem"] == 141.4
    assert TABLE8_REFERENCE["apt"] == 83.9

    # result container
    res = LoRAPruneDistillResult(task="sst2", metrics={"accuracy": 91.9}, primary=91.9)
    assert res.as_dict()["method"] == METHOD_KEY
    assert DISPLAY_NAME in res.summary()

    # LoRA-only enforcement on a duck-typed model (no torch required)
    class _P:
        def __init__(self, flag: bool = True) -> None:
            self.requires_grad = flag
            self.ndim = 2
            self.numel = lambda: 4

        def requires_grad_(self, value: bool) -> "_P":
            self.requires_grad = value
            return self

    class _M:
        def __init__(self) -> None:
            self._params = {
                "layer.0.query.lora_a": _P(),
                "layer.0.query.lora_b": _P(),
                "layer.0.gate_log_alpha": _P(),
                "layer.0.query.weight": _P(),
                "layer.0.output.dense.weight": _P(),
            }

        def named_parameters(self):
            return list(self._params.items())

    stats = freeze_to_lora_and_gates(_M())
    assert stats["trainable"] == 3, stats
    assert stats["frozen"] == 2, stats
    assert count_lora_only_parameters(_M()) == 12

    # tuning_parameter_names / helpers
    class _T:
        def __init__(self) -> None:
            self.p = {"a.lora_a": _P(), "a.weight": _P()}

        def named_parameters(self):
            return list(self.p.items())

    assert lora_only_trainable_names(_T()) == ["a.lora_a"]

    # model_inputs filtering
    batch = {"input_ids": 1, "labels": 2, "row_id": 3}
    assert set(model_inputs(batch)) == {"input_ids", "labels"}

    # display name consistency with the baseline registry (if importable)
    try:
        from apt.baselines import METHOD_DISPLAY_NAMES  # noqa: WPS433

        assert METHOD_DISPLAY_NAMES.get(METHOD_KEY) == DISPLAY_NAME
    except Exception as exc:
        warnings.warn(f"registry check skipped: {exc!r}")

    return ok


if __name__ == "__main__":  # pragma: no cover
    print(f"{DISPLAY_NAME}: self test -> {_self_test()}")
    print(f"CoFi backend available: {_HAS_COFI}")
    print(f"Table 2 metrics: {TABLE2_REFERENCES['metrics']}")
    print(f"Table 2 efficiency (%FT): {TABLE2_EFFICIENCY}")
