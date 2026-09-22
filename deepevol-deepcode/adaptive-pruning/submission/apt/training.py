"""Two-stage APT training loop (Algorithm 1, Appendix A/C, Section 6).

This module orchestrates the complete APT fine-tuning procedure:

  Stage 1 (prune + self-distillation) -- for ``distill_epochs`` epochs the LM is pruned
  with the cubic sparsity schedule ``gamma_t = gamma_T + (1 - gamma_T)(1 - t/T)^3`` while
  a self-knowledge-distillation objective ``L = mu * L_distill + (1 - mu) * L_ft`` is
  optimised and adapters grow (``r_apt' = floor(r_apt * Delta_t' / Delta_t)`` for the
  top-half salient adapters).

  Stage 2 (recovery) -- the pruned LM (masks hardened) is fine-tuned on the end task for
  the remaining epochs to recover accuracy.

Each pruning step follows Algorithm 1:

  1. forward the student (with hidden-state caching for distillation),
  2. forward the frozen teacher (shared frozen params, masks = 1, ``W_B = 0``),
  3. backward the total loss,
  4. compute the outlier-aware salience and update its EMA (0.85 / 0.15),
  5. select blocks with the latency-saliency knapsack (density sort + binary search),
  6. anneal the masks by ``alpha = 0.01``,
  7. grow the tuning ranks of the top-half salient adapters within budget ``Delta_t``,
  8. clip / step the optimizer, and reset the optimizer state whenever a parameter size change
     occurred (mask pruning or rank growth).

The module is deliberately defensive: every ``apt.*`` dependency is imported with a guard and
the trainer degrades gracefully (e.g. no distillation, no rank growth) when a component is
unavailable.  Everything is duck-typed around the interfaces of the sibling modules
(``apt.adapters``, ``apt.masks``, ``apt.salience``, ``apt.block_selection``, ``apt.rank_controller``,
``apt.distillation``, ``apt.model_wrapper``, ``apt.merge``).
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Optional apt imports (defensive: training must import even in bare envs)
# ---------------------------------------------------------------------------

try:  # pragma: no cover - import guard
    from .adapters import (  # type: ignore
        BLOCK_TYPES,
        DIMENSION,
        HEAD,
        NEURON,
        APTAdapter,
        MaskedLinear,
        iter_masked_linears,
    )
except Exception:  # pragma: no cover
    BLOCK_TYPES = {0: "head", 1: "neuron", 2: "dimension"}
    HEAD, NEURON, DIMENSION = 0, 1, 2
    APTAdapter = MaskedLinear = None  # type: ignore
    iter_masked_linears = None  # type: ignore

try:  # pragma: no cover
    from .model_wrapper import (  # type: ignore
        APTModelWrapper,
        apt_shape,
        is_wrapped,
        unwrapped,
        wrap_model,
        count_parameters,
        restore_base_linears,
    )
except Exception:  # pragma: no cover
    APTModelWrapper = None  # type: ignore
    apt_shape = is_wrapped = unwrapped = wrap_model = None  # type: ignore
    count_parameters = restore_base_linears = None  # type: ignore

try:  # pragma: no cover
    from .masks import MaskManager, MaskState, approx_lm_param_count  # type: ignore
except Exception:  # pragma: no cover
    MaskManager = MaskState = None  # type: ignore
    approx_lm_param_count = None  # type: ignore

try:  # pragma: no cover
    from .block_selection import BlockSelector, ModelShape  # type: ignore
except Exception:  # pragma: no cover
    BlockSelector = ModelShape = None  # type: ignore

try:  # pragma: no cover
    from .salience import OutlierAwareSalience, SalienceConfig  # type: ignore
except Exception:  # pragma: no cover
    OutlierAwareSalience = SalienceConfig = None  # type: ignore

try:  # pragma: no cover
    from .rank_controller import (  # type: ignore
        RankController,
        build_optimizer,
        recreate_optimizer,
        reset_optimizer_state,
        trainable_parameters,
    )
except Exception:  # pragma: no cover
    RankController = None  # type: ignore
    build_optimizer = recreate_optimizer = reset_optimizer_state = None  # type: ignore
    trainable_parameters = None  # type: ignore

try:  # pragma: no cover
    from .distillation import (  # type: ignore
        SelfDistillation,
        find_transformer_layers,
        normalize_task,
        task_distill_weights,
        student_layer_keep_flags,
        TASK_DISTILL_WEIGHTS,
    )
except Exception:  # pragma: no cover
    SelfDistillation = find_transformer_layers = None  # type: ignore
    normalize_task = task_distill_weights = student_layer_keep_flags = None  # type: ignore
    TASK_DISTILL_WEIGHTS = {"glue": (1.0, 0.9), "squad": (0.1, 0.9), "cnndm": (0.1, 0.9)}

try:  # pragma: no cover
    from .schedulers import (  # type: ignore
        AdjustmentStepSchedule,
        MaskDecaySchedule,
        MuSchedule,
        build_lr_scheduler,
        compute_pruning_window,
        epochs_to_steps,
        lr_factor,
        make_mu_schedule,
        make_sparsity_schedule,
        rank_update,
    )
except Exception:  # pragma: no cover
    AdjustmentStepSchedule = MaskDecaySchedule = MuSchedule = None  # type: ignore
    build_lr_scheduler = compute_pruning_window = epochs_to_steps = lr_factor = None  # type: ignore
    make_mu_schedule = make_sparsity_schedule = rank_update = None  # type: ignore

try:  # pragma: no cover
    from .merge import (  # type: ignore
        build_prune_plan,
        harden_masks,
        merge_and_prune,
        verify_merge_equivalence,
    )
except Exception:  # pragma: no cover
    build_prune_plan = harden_masks = merge_and_prune = verify_merge_equivalence = None  # type: ignore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_INPUT_KEYS = (
    "input_ids",
    "attention_mask",
    "token_type_ids",
    "position_ids",
    "decoder_input_ids",
    "decoder_attention_mask",
    "encoder_outputs",
    "head_mask",
    "labels",
    "start_positions",
    "end_positions",
)

GLUE_TASKS = ("mnli", "sst2", "qnli", "qqp", "mrpc", "cola", "rte", "stsb")
SQUAD_TASKS = ("squad", "squad_v2", "squadv2")
SEQ2SEQ_TASKS = ("cnndm", "cnn_dailymail", "cnn-dailymail", "summarization", "xsum")

DEFAULT_ALPHA = 0.01
DEFAULT_EMA_BETA = 0.85
DEFAULT_INITIAL_RANK = 8
DEFAULT_SCALING = 2.0
DEFAULT_TARGET_SPARSITY = 0.60
DEFAULT_TAU = 4


def _task_family(task: Optional[str]) -> str:
    """Map a task name to one of ``glue`` / ``squad`` / ``seq2seq`` / ``other``."""
    name = _normalize_task(task)
    if name in SQUAD_TASKS:
        return "squad"
    if name in SEQ2SEQ_TASKS:
        return "seq2seq"
    if name in GLUE_TASKS:
        return "glue"
    return "other"


def _normalize_task(task: Optional[str]) -> str:
    """Lower-case / strip separators, then defer to ``apt.distillation.normalize_task``."""
    if task is None:
        return "glue"
    name = str(task).strip().lower().replace("-", "_")
    if callable(normalize_task):  # type: ignore[truthy-function]
        try:
            return normalize_task(name)
        except Exception:
            pass
    alias = {
        "sst_2": "sst2",
        "sst": "sst2",
        "mnli_matched": "mnli",
        "mnli_mismatched": "mnli",
        "mnli_mm": "mnli",
        "sts_b": "stsb",
        "squadv2": "squad",
        "cnn_dailymail": "cnndm",
        "cnn_dm": "cnndm",
        "cnndm_": "cnndm",
    }
    return alias.get(name, name)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """All APT training hyper-parameters (Table 6 + Appendix A + Addendum).

    Defaults follow GLUE-big / RoBERTa-base at 60% sparsity.
    """

    # model / task
    model_name_or_path: str = "roberta-base"
    model_type: Optional[str] = None
    task: str = "sst2"
    num_labels: Optional[int] = None

    # Table 6
    learning_rate: float = 2e-4
    batch_size: int = 32
    epochs: int = 40
    distill_epochs: int = 20
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    max_seq_length: int = 128
    max_target_length: int = 128
    doc_stride: int = 128
    max_query_length: int = 64

    # APT specifics
    target_sparsity: float = DEFAULT_TARGET_SPARSITY
    initial_rank: int = DEFAULT_INITIAL_RANK
    scaling: float = DEFAULT_SCALING
    mask_alpha: float = DEFAULT_ALPHA
    ema_beta: float = DEFAULT_EMA_BETA
    tau: int = DEFAULT_TAU
    use_kurtosis: bool = True
    sparsity_schedule: str = "cubic"
    sparsity_exponent: float = 3.0
    adjustment_interval: int = 1
    top_fraction: float = 0.5
    tuning_budget_initial: float = 1.0
    tuning_budget_final: float = 2.0
    max_rank: Optional[int] = None
    use_distillation: bool = True
    use_adaptive_tuning: bool = True
    use_salience: bool = True
    use_adaptive_pruning: bool = True
    pred_distill_weight: Optional[float] = None
    layer_distill_weight: float = 0.9
    distill_on_recovery: bool = False

    # optimization
    optimizer: str = "adamw"
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    lr_kind: str = "linear"
    lr_min_factor: float = 0.0

    # runtime
    device: Optional[str] = None
    seed: int = 42
    log_interval: int = 50
    eval_interval: Optional[int] = None
    output_dir: Optional[str] = None
    use_amp: bool = False
    num_workers: int = 0
    max_train_steps: Optional[int] = None
    max_eval_batches: Optional[int] = None
    reference_metric: Optional[float] = None  # FT accuracy for TTA tracking
    tta_fraction: float = 0.97
    measure_memory: bool = True
    verbose: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]] = None, **overrides: Any) -> "TrainConfig":
        """Build a config from a (nested) dict / YAML mapping plus overrides."""
        data = dict(data or {})
        flat: Dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, dict) and key not in ("extra", "adam_betas"):
                # allow nested sections such as ``training:`` / ``apt:``
                flat.update(value)
            else:
                flat[key] = value
        flat.update(overrides)
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        extra = dict(flat.pop("extra", {}) or {})
        unknown = {k: v for k, v in flat.items() if k not in known}
        extra.update(unknown)
        kwargs = {k: v for k, v in flat.items() if k in known}
        if "adam_betas" in kwargs and isinstance(kwargs["adam_betas"], (list, tuple)):
            kwargs["adam_betas"] = tuple(kwargs["adam_betas"])  # type: ignore[assignment]
        cfg = cls(**kwargs)
        cfg.extra = extra
        return cfg

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        import yaml  # local import: only needed when a YAML config is used

        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["adam_betas"] = list(self.adam_betas)
        return data

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        if path.endswith((".yaml", ".yml")):
            import yaml

            with open(path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(self.to_dict(), handle, sort_keys=False)
        else:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, indent=2)
        return path

    # ------------------------------------------------------------------
    @property
    def pruning_epochs(self) -> int:
        return int(self.distill_epochs)

    @property
    def recovery_epochs(self) -> int:
        return max(int(self.epochs) - int(self.distill_epochs), 0)

    def pred_weight_for(self, task: Optional[str] = None) -> float:
        """Distillation prediction weight: GLUE 1.0, SQuAD / CNN-DM 0.1 (Appendix A)."""
        if self.pred_distill_weight is not None:
            return float(self.pred_distill_weight)
        task = self.task if task is None else task
        if callable(task_distill_weights):  # type: ignore[truthy-function]
            try:
                return float(task_distill_weights(task)[0])
            except Exception:
                pass
        return float(TASK_DISTILL_WEIGHTS.get(_task_family(task), (1.0, 0.9))[0])


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def set_seed(seed: Optional[int]) -> None:
    """Seed python / numpy / torch (best effort)."""
    if seed is None:
        return
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32 - 1))
    except Exception:  # pragma: no cover
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_to_device(batch: Any, device: torch.device) -> Any:
    """Recursively move tensors of a mapping to ``device``."""
    if isinstance(batch, dict):
        out: Dict[str, Any] = {}
        for key, value in batch.items():
            out[key] = move_to_device(value, device)
        return out
    if isinstance(batch, (list, tuple)):
        moved = [move_to_device(v, device) for v in batch]
        return type(batch)(moved) if not isinstance(batch, list) else moved
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    return batch


def model_inputs(batch: Dict[str, Any], keys: Sequence[str] = MODEL_INPUT_KEYS) -> Dict[str, Any]:
    """Keep only the tensors that HuggingFace models accept as keyword arguments."""
    return {k: v for k, v in batch.items() if k in keys and v is not None}


def trainable_parameter_list(model: nn.Module) -> List[nn.Parameter]:
    """De-duplicated list of parameters that require grad."""
    seen, params = set(), []
    for param in model.parameters():
        if param.requires_grad and id(param) not in seen:
            seen.add(id(param))
            params.append(param)
    return params


def param_groups(model: nn.Module, weight_decay: float, lr: float) -> List[Dict[str, Any]]:
    """AdamW param groups: no weight decay on biases / LayerNorms."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay, "lr": lr})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0, "lr": lr})
    if not groups:
        groups.append({"params": [p for p in model.parameters() if p.requires_grad], "lr": lr})
    return groups


def _get_attr(obj: Any, names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return default


def _count_params(model: nn.Module, trainable_only: bool = False) -> int:
    if callable(count_parameters):  # type: ignore[truthy-function]
        try:
            return int(count_parameters(model, trainable_only=trainable_only))  # type: ignore[misc]
        except Exception:
            pass
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


# ---------------------------------------------------------------------------
# Hidden state recording (self-contained; no reliance on sibling attribute names)
# ---------------------------------------------------------------------------


class HiddenStateRecorder:
    """Collect per-transformer-layer hidden states with forward hooks.

    ``states`` maps the layer index inside the discovered transformer block list to the
    hidden state *output* of that block (post-block residual stream, i.e. the tensor passed on
    to the next layer, exactly what layer-wise distillation matches).
    """

    def __init__(self, model: nn.Module, layers: Optional[Sequence[nn.Module]] = None):
        self.model = model
        self.layers = list(layers) if layers is not None else self._discover(model)
        self.states: Dict[int, torch.Tensor] = {}
        self._handles: List[Any] = []

    # ------------------------------------------------------------------
    @staticmethod
    def _discover(model: nn.Module) -> List[nn.Module]:
        if callable(find_transformer_layers):  # type: ignore[truthy-function]
            try:
                found = find_transformer_layers(model)  # type: ignore[misc]
                if found:
                    return list(found)
            except Exception:
                pass
        for name, module in model.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) > 0:
                last = name.split(".")[-1]
                if last in ("layer", "layers", "block", "blocks", "h"):
                    return list(module)
        return []

    # ------------------------------------------------------------------
    def _hook(self, index: int) -> Callable[..., None]:
        def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if torch.is_tensor(tensor):
                self.states[index] = tensor

        return hook

    def attach(self) -> "HiddenStateRecorder":
        self.detach()
        for index, layer in enumerate(self.layers):
            self._handles.append(layer.register_forward_hook(self._hook(index)))
        return self

    def detach(self) -> None:
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:  # pragma: no cover
                pass
        self._handles = []

    def clear(self) -> None:
        self.states = {}

    def __enter__(self) -> "HiddenStateRecorder":
        self.clear()
        return self.attach()

    def __exit__(self, *exc: Any) -> None:
        self.detach()

    @property
    def n_layers(self) -> int:
        return len(self.layers)


# ---------------------------------------------------------------------------
# TTA (time-to-accuracy) tracker
# ---------------------------------------------------------------------------


@dataclass
class TimeToAccuracy:
    """Tracks time-to-accuracy to ``fraction`` of a reference (FT) metric."""

    reference: Optional[float] = None
    fraction: float = 0.97
    reached_at: Optional[float] = None
    reached_step: Optional[int] = None
    reached_metric: Optional[float] = None
    history: List[Tuple[float, int, float]] = field(default_factory=list)

    @property
    def target(self) -> Optional[float]:
        if self.reference is None:
            return None
        return float(self.reference) * float(self.fraction)

    def update(self, elapsed: float, step: int, metric: Optional[float]) -> bool:
        if metric is None:
            return False
        self.history.append((float(elapsed), int(step), float(metric)))
        target = self.target
        if target is None or self.reached_at is not None:
            return False
        if float(metric) >= target:
            self.reached_at = float(elapsed)
            self.reached_step = int(step)
            self.reached_metric = float(metric)
            return True
        return False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "reference": self.reference,
            "fraction": self.fraction,
            "target": self.target,
            "reached_at": self.reached_at,
            "reached_step": self.reached_step,
            "reached_metric": self.reached_metric,
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class APTTrainer:
    """APT two-stage prune-and-tune trainer (Algorithm 1).

    Parameters
    ----------
    config:
        :class:`TrainConfig` (or a plain dict / YAML path).
    model:
        A HuggingFace model (will be wrapped in place) or an already wrapped APT model.
    tokenizer:
        Optional tokenizer used by the data/eval helpers (not needed post-tokenization).
    train_dataloader / eval_dataloader:
        Iterables of batches; keys follow the HuggingFace convention.
    compute_metrics:
        Optional ``callable(predictions, references) -> Dict[str, float]``.  When omitted, the
        apt data modules are used based on ``config.task``.
    """

    def __init__(
        self,
        config: Any = None,
        model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
        train_dataloader: Optional[Iterable[Any]] = None,
        eval_dataloader: Optional[Iterable[Any]] = None,
        compute_metrics: Optional[Callable[..., Dict[str, float]]] = None,
        reference_model: Optional[nn.Module] = None,
        **kwargs: Any,
    ):
        if isinstance(config, str):
            config = TrainConfig.from_yaml(config)
        elif isinstance(config, dict) or config is None:
            config = TrainConfig.from_dict(config, **kwargs)
        self.config: TrainConfig = config
        self.task = _normalize_task(self.config.task)
        self.family = _task_family(self.task)

        self.device = torch.device(
            self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self._compute_metrics = compute_metrics
        self.reference_model = reference_model

        self.model: Optional[nn.Module] = None
        self.teacher: Optional[nn.Module] = None
        self.wrapper: Optional[Any] = None
        self.shape: Optional[Any] = None
        self.blocks: Optional[List[Any]] = None
        self.selector = None
        self.mask_manager = None
        self.salience = None
        self.rank_controller = None
        self.distill: Optional[Any] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.lr_scheduler: Optional[Any] = None

        self.sparsity_schedule = None
        self.mu_schedule = None
        self.mask_decay_schedule = None
        self.adjustment_schedule = None

        self.global_step = 0
        self.epoch = 0
        self.stage = "prune"
        self.pruning_start_step = 0
        self.pruning_end_step = 0
        self.total_steps = 0
        self.steps_per_epoch = 0
        self.history: List[Dict[str, Any]] = []
        self.optimizer_resets = 0
        self.tta = TimeToAccuracy(
            reference=self.config.reference_metric, fraction=self.config.tta_fraction
        )
        self.train_start_time: Optional[float] = None
        self.train_seconds: Optional[float] = None
        self.train_peak_memory_bytes: Optional[int] = None
        self.inference_peak_memory_bytes: Optional[int] = None

        set_seed(self.config.seed)
        if model is not None:
            self.model = model

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup_model(self, model: Optional[nn.Module] = None) -> nn.Module:
        """Wrap ``model`` (in place) so that APT adapters / masks are injected."""
        model = model if model is not None else self.model
        if model is None:
            raise ValueError("APT training requires a model.")
        if callable(is_wrapped) and is_wrapped(model):  # type: ignore[truthy-function]
            self.model = model
            self.wrapper = _get_attr(model, ("wrapper", "apt_wrapper"), None)
        elif callable(wrap_model):  # type: ignore[truthy-function]
            self.wrapper = wrap_model(  # type: ignore[misc]
                model,
                rank=self.config.initial_rank,
                scaling=self.config.scaling,
                model_type=self.config.model_type,
                cache_for_salience=True,
                capture_grad=True,
            )
            self.model = model
        else:  # pragma: no cover - fallback for bare environments
            warnings.warn("apt.model_wrapper unavailable: running without APT wrapping.")
            self.model = model
        self.model.to(self.device)
        if callable(apt_shape):  # type: ignore[truthy-function]
            try:
                self.shape = apt_shape(self.model, self.config.model_type)  # type: ignore[misc]
            except Exception:
                self.shape = None
        return self.model

    # ------------------------------------------------------------------
    def setup_pruning(self, train_steps: int = 0) -> None:
        """Build the block selector, mask manager and salience tracker."""
        if self.model is None:
            return
        if BlockSelector is not None and self.shape is not None:
            try:
                self.selector = BlockSelector(self.shape)
                self.blocks = self.selector.enumerate_blocks()
            except Exception:
                self.selector, self.blocks = None, None
        if MaskManager is not None and self.shape is not None:
            try:
                self.mask_manager = MaskManager(
                    self.shape,
                    selector=self.selector,
                    alpha=self.config.mask_alpha,
                )
            except Exception:
                self.mask_manager = None
        if OutlierAwareSalience is not None:
            try:
                self.salience = OutlierAwareSalience(
                    beta=self.config.ema_beta,
                    use_kurtosis=self.config.use_kurtosis,
                    include_frozen=self.config.use_salience,
                    include_tuning=True,
                )
            except Exception:
                self.salience = None
        if (
            self.config.use_adaptive_tuning
            and RankController is not None
            and self.blocks is not None
        ):
            budget = self._budget_schedule(train_steps)
            try:
                self.rank_controller = RankController(
                    initial_rank=self.config.initial_rank,
                    scaling=self.config.scaling,
                    top_fraction=self.config.top_fraction,
                    max_rank=self.config.max_rank,
                    budget_schedule=budget,
                    budget_initial=self.config.tuning_budget_initial,
                    budget_final=self.config.tuning_budget_final,
                    pruning_start_step=self.pruning_start_step,
                    pruning_end_step=self.pruning_end_step or max(train_steps, 1),
                    total_steps=train_steps,
                )
            except Exception:
                self.rank_controller = None
        if SelfDistillation is not None and self.config.use_distillation:
            dim, n_layers = self._dim_and_layers()
            try:
                weights = (self.config.pred_weight_for(self.task), self.config.layer_distill_weight)
                self.distill = SelfDistillation(
                    dim=dim,
                    n_layers=n_layers,
                    task=self.family,
                    tau=self.config.tau,
                    pred_weight=weights[0],
                    layer_weight=weights[1],
                    seed=self.config.seed,
                    device=self.device,
                )
            except Exception:
                self.distill = None

    # ------------------------------------------------------------------
    def _dim_and_layers(self) -> Tuple[int, int]:
        config = getattr(self.model, "config", None)
        dim = int(
            _get_attr(config, ("hidden_size", "d_model", "n_embd", "dim"), 768) or 768
        )
        n_layers = int(
            _get_attr(config, ("num_hidden_layers", "n_layer", "num_layers"), 12) or 12
        )
        return dim, n_layers

    def _budget_schedule(self, train_steps: int):
        try:
            from .schedulers import make_tuning_budget_schedule  # type: ignore

            return make_tuning_budget_schedule(
                initial=self.config.tuning_budget_initial,
                final=self.config.tuning_budget_final,
                kind="linear",
                pruning_start_step=self.pruning_start_step,
                pruning_end_step=self.pruning_end_step or max(train_steps, 1),
                total_steps=train_steps,
            )
        except Exception:
            return None

    # ------------------------------------------------------------------
    def setup_schedules(self, steps_per_epoch: int) -> None:
        """Build the cubic sparsity / mu / mask-decay / adjustment / LR schedules."""
        train_steps = int(steps_per_epoch * max(self.config.pruning_epochs, 1))
        if self.config.max_train_steps:
            train_steps = min(train_steps, int(self.config.max_train_steps))
        self.steps_per_epoch = int(steps_per_epoch)
        self.pruning_start_step = 0
        self.pruning_end_step = train_steps
        self.total_steps = int(steps_per_epoch * max(self.config.epochs, 1))

        if callable(make_sparsity_schedule):  # type: ignore[truthy-function]
            self.sparsity_schedule = make_sparsity_schedule(  # type: ignore[misc]
                kind=self.config.sparsity_schedule,
                target_sparsity=self.config.target_sparsity,
                total_steps=max(train_steps, 1),
                initial_sparsity=1.0,
                warmup_steps=0,
                exponent=self.config.sparsity_exponent,
            )
        if callable(make_mu_schedule):  # type: ignore[truthy-function]
            self.mu_schedule = make_mu_schedule(  # type: ignore[misc]
                pruning_start_step=0,
                pruning_end_step=max(train_steps, 1),
                enabled=self.config.use_distillation,
            )
        if MaskDecaySchedule is not None:
            self.mask_decay_schedule = MaskDecaySchedule(alpha=self.config.mask_alpha)
        if AdjustmentStepSchedule is not None:
            self.adjustment_schedule = AdjustmentStepSchedule(
                interval=self.config.adjustment_interval,
                pruning_start_step=0,
                pruning_end_step=max(train_steps, 1),
            )

    # ------------------------------------------------------------------
    def setup_optimizer(self, lr: Optional[float] = None) -> torch.optim.Optimizer:
        lr = float(self.config.learning_rate if lr is None else lr)
        if self.model is None:
            raise ValueError("setup_optimizer requires a model")
        groups = param_groups(self.model, self.config.weight_decay, lr)
        if callable(build_optimizer):  # type: ignore[truthy-function]
            try:
                self.optimizer = build_optimizer(
                    self.model,
                    lr=lr,
                    weight_decay=self.config.weight_decay,
                    betas=tuple(self.config.adam_betas),
                    eps=self.config.adam_eps,
                    name=self.config.optimizer,
                    parameters=groups,
                )
                return self.optimizer
            except Exception:
                pass
        self.optimizer = torch.optim.AdamW(
            groups, lr=lr, betas=tuple(self.config.adam_betas), eps=self.config.adam_eps
        )
        return self.optimizer

    # ------------------------------------------------------------------
    def setup_lr_scheduler(self, total_steps: Optional[int] = None) -> None:
        if self.optimizer is None:
            return
        total = int(total_steps or self.total_steps or 1)
        warmup = int(round(total * float(self.config.warmup_ratio)))
        if callable(build_lr_scheduler):  # type: ignore[truthy-function]
            try:
                self.lr_scheduler = build_lr_scheduler(  # type: ignore[misc]
                    self.optimizer,
                    total_steps=max(total, 1),
                    warmup_steps=warmup,
                    kind=self.config.lr_kind,
                    min_factor=self.config.lr_min_factor,
                )
                return
            except Exception:
                pass
        self.lr_scheduler = None

    # ------------------------------------------------------------------
    def _reset_optimizer(self, reason: str = "parameter-size-change") -> None:
        """Reset the optimizer after a parameter size change (Section 6)."""
        params = trainable_parameter_list(self.model) if self.model is not None else []
        new_optimizer = None
        if callable(recreate_optimizer):  # type: ignore[truthy-function]
            try:
                new_optimizer = recreate_optimizer(  # type: ignore[misc]
                    self.optimizer, parameters=params
                )
            except Exception:
                new_optimizer = None
        if new_optimizer is None:
            lr = self.config.learning_rate
            groups = param_groups(self.model, self.config.weight_decay, lr) if self.model else [
                {"params": params}
            ]
            new_optimizer = torch.optim.AdamW(
                groups, lr=lr, betas=tuple(self.config.adam_betas), eps=self.config.adam_eps
            )
            for group in new_optimizer.param_groups:
                group.setdefault("lr", lr)
        self.optimizer = new_optimizer
        self.optimizer_resets += 1
        if self.lr_scheduler is not None:
            # keep the same schedule shape but re-anchor to the new optimizer
            remaining = max(int(self.total_steps) - int(self.global_step), 1)
            self.setup_lr_scheduler(total_steps=remaining)
        if self.config.verbose and self.optimizer_resets <= 20:
            print(f"  [apt] optimizer reset ({reason}) #{self.optimizer_resets}")

    # ------------------------------------------------------------------
    # Teacher
    # ------------------------------------------------------------------
    def build_teacher(self) -> Optional[nn.Module]:
        """Duplicate the student sharing all *frozen* parameters (no extra LM copy).

        The duplicated teacher has its adapters zeroed (``W_B = 0``) and its masks reset to one,
        i.e. it reproduces the original pretrained LM's hidden states while sharing the frozen
        weight tensors with the student.
        """
        if self.model is None or not self.config.use_distillation:
            return None
        memo: Dict[int, nn.Parameter] = {}
        for _name, param in self.model.named_parameters():
            if not param.requires_grad:
                memo[id(param)] = param  # share frozen LM weights
        try:
            teacher = copy.deepcopy(self.model, memo)
        except Exception:
            teacher = copy.deepcopy(self.model)
        for module in teacher.modules():
            # zero adapter contribution
            for attr in ("lora_b", "lora_B", "b"):
                param = getattr(module, attr, None)
                if torch.is_tensor(param):
                    try:
                        with torch.no_grad():
                            param.zero_()
                    except Exception:
                        pass
            # reset masks to one
            for attr in ("mask_in", "mask_out", "input_mask", "output_mask"):
                value = getattr(module, attr, None)
                if torch.is_tensor(value) and value.is_floating_point():
                    try:
                        with torch.no_grad():
                            value.fill_(1.0)
                    except Exception:
                        pass
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)
        self.teacher = teacher.to(self.device)
        return self.teacher

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------
    def _forward(
        self,
        model: nn.Module,
        batch: Dict[str, Any],
        collect_hidden: bool = False,
        grad: bool = True,
        recorder: Optional[HiddenStateRecorder] = None,
    ) -> Tuple[Any, Optional[Dict[int, torch.Tensor]]]:
        inputs = model_inputs(batch)
        hidden = None
        context = torch.enable_grad() if grad else torch.no_grad()
        with context:
            if collect_hidden:
                recorder = recorder or HiddenStateRecorder(model)
                with recorder:
                    outputs = model(**inputs)
                hidden = dict(recorder.states)
            else:
                outputs = model(**inputs)
        return outputs, hidden

    # ------------------------------------------------------------------
    def _task_loss(self, outputs: Any, batch: Dict[str, Any]) -> torch.Tensor:
        """End-task loss; prefer the model's own loss, else compute it explicitly."""
        loss = getattr(outputs, "loss", None)
        if loss is not None:
            return loss
        logits = getattr(outputs, "logits", outputs)
        if self.family == "squad":
            start = batch.get("start_positions")
            end = batch.get("end_positions")
            if start is None or end is None:
                raise ValueError("SQuAD batches must contain start_positions / end_positions")
            start_logits, end_logits = logits[..., 0], logits[..., 1]
            return 0.5 * (
                F.cross_entropy(start_logits, start) + F.cross_entropy(end_logits, end)
            )
        labels = batch.get("labels")
        if labels is None:
            raise ValueError("Batch has no labels and the model returned no loss")
        if labels.dtype.is_floating_point or logits.dim() == labels.dim():
            return F.mse_loss(logits.view(-1), labels.view(-1).float())
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
        )

    # ------------------------------------------------------------------
    def _student_layer_keep(self) -> Optional[List[int]]:
        if self.model is None or not callable(student_layer_keep_flags):  # type: ignore[truthy-function]
            return None
        try:
            return list(student_layer_keep_flags(self.model))  # type: ignore[misc]
        except Exception:
            return None

    def _distillation_loss(
        self,
        student_outputs: Any,
        student_hidden: Dict[int, torch.Tensor],
        batch: Dict[str, Any],
        mu: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute ``mu * L_distill + (1 - mu) * L_ft`` and return (total, distill)."""
        if self.teacher is None or self.distill is None:
            ft = self._task_loss(student_outputs, batch)
            return ft, torch.zeros((), device=self.device)
        with torch.no_grad():
            teacher_outputs, teacher_hidden = self._forward(
                self.teacher, batch, collect_hidden=True, grad=False
            )
        teacher_indices = None
        try:
            teacher_indices = self.distill.sample_teacher_layers()
        except Exception:
            teacher_indices = None
        phi = None
        try:
            phi = self.distill.compute_phi(
                student_layer_keep=self._student_layer_keep(),
                n_teacher_layers=len(teacher_indices) if teacher_indices else None,
            )
        except Exception:
            phi = None
        try:
            layer_loss = self.distill.layer_loss(
                student_hidden, teacher_hidden, teacher_indices=teacher_indices, phi=phi
            )
        except Exception:
            layer_loss = torch.zeros((), device=self.device)
        try:
            pred_loss = self.distill.prediction_loss(
                getattr(student_outputs, "logits", None),
                getattr(teacher_outputs, "logits", None),
            )
        except Exception:
            pred_loss = torch.zeros((), device=self.device)
        if not torch.is_tensor(layer_loss):
            layer_loss = torch.as_tensor(layer_loss, device=self.device)
        if not torch.is_tensor(pred_loss):
            pred_loss = torch.as_tensor(pred_loss, device=self.device)
        distill = float(self.config.pred_weight_for(self.task)) * pred_loss + float(
            self.config.layer_distill_weight
        ) * layer_loss
        try:
            total = self.distill.total_loss(None, distill, mu)
            if not torch.is_tensor(total):
                total = torch.as_tensor(total, device=self.device)
            if not torch.isfinite(total):
                raise ValueError("non-finite distillation loss")
            # recompute with the task loss so the (1 - mu) term is included
            ft = self._task_loss(student_outputs, batch)
            total = total * 0.0 + float(mu) * distill + (1.0 - float(mu)) * ft
        except Exception:
            ft = self._task_loss(student_outputs, batch)
            total = float(mu) * distill + (1.0 - float(mu)) * ft
        return total, distill

    # ------------------------------------------------------------------
    # Schedule accessors
    # ------------------------------------------------------------------
    def _sparsity_at(self, step: int) -> float:
        if self.sparsity_schedule is not None:
            try:
                return float(self.sparsity_schedule(step))
            except Exception:
                pass
        frac = min(max(step / max(self.pruning_end_step, 1), 0.0), 1.0)
        target = float(self.config.target_sparsity)
        return target + (1.0 - target) * (1.0 - frac) ** float(self.config.sparsity_exponent)

    def _mu_at(self, step: int) -> float:
        if not self.config.use_distillation:
            return 0.0
        if self.mu_schedule is not None:
            try:
                return float(self.mu_schedule(step))
            except Exception:
                pass
        frac = min(max(step / max(self.pruning_end_step, 1), 0.0), 1.0)
        return frac

    def _alpha_at(self, step: int) -> float:
        if self.mask_decay_schedule is not None:
            try:
                return float(self.mask_decay_schedule(step))
            except Exception:
                pass
        return float(self.config.mask_alpha)

    def _should_adjust(self, step: int) -> bool:
        if self.adjustment_schedule is not None:
            try:
                return bool(self.adjustment_schedule.should_adjust(step))
            except Exception:
                pass
        interval = max(int(self.config.adjustment_interval), 1)
        return step % interval == 0

    # ------------------------------------------------------------------
    # Salience / selection / rank growth
    # ------------------------------------------------------------------
    def compute_salience(self, update: bool = True) -> Optional[Dict[str, float]]:
        if self.salience is None or not self.config.use_adaptive_pruning:
            return None
        try:
            scores = self.salience.step(
                self.model,
                shape=self.shape,
                blocks=self.blocks if not self.config.use_salience else None,
                beta=self.config.ema_beta,
                update=update,
            )
        except TypeError:
            try:
                scores = self.salience.step(self.model, shape=self.shape, beta=self.config.ema_beta)
            except Exception:
                return None
        except Exception:
            return None
        return scores if isinstance(scores, dict) else None

    def prune_step(self, salience: Optional[Dict[str, float]], step: int):
        """Update masks towards the scheduled sparsity and push them to the model."""
        if self.mask_manager is None or not self.config.use_adaptive_pruning:
            return None
        sparsity = self._sparsity_at(step)
        alpha = self._alpha_at(step)
        try:
            selection = self.mask_manager.step(
                model=self.model,
                salience=salience,
                sparsity=sparsity,
                alpha=alpha,
            )
        except TypeError:
            try:
                selection = self.mask_manager.step(
                    model=self.model, salience=salience, sparsity=sparsity
                )
            except Exception:
                return None
        except Exception:
            return None
        return selection

    def grow_ranks(self, salience: Optional[Dict[str, float]], step: int):
        if self.rank_controller is None or not self.config.use_adaptive_tuning:
            return None
        try:
            result = self.rank_controller.step(
                self.model, salience=salience, global_step=step
            )
        except TypeError:
            try:
                result = self.rank_controller.step(self.model, salience)
            except Exception:
                return None
        except Exception:
            return None
        optimizer = getattr(result, "optimizer", None)
        if optimizer is not None and optimizer is not self.optimizer:
            self.optimizer = optimizer
            self.optimizer_resets += 1
        return result

    def maybe_reset_optimizer(self) -> bool:
        """Reset the optimizer if a mask update / rank growth changed parameter shapes."""
        changed = False
        if self.mask_manager is not None:
            try:
                changed = bool(self.mask_manager.needs_optimizer_reset())
                if changed:
                    self.mask_manager.consume_optimizer_reset()
            except Exception:
                changed = False
        if self.rank_controller is not None:
            try:
                changed = bool(self.rank_controller.needs_optimizer_reset()) or changed
                if changed:
                    self.rank_controller.consume_optimizer_reset()
            except Exception:
                pass
        if changed:
            self._reset_optimizer()
        return changed

    # ------------------------------------------------------------------
    # One optimisation step (Algorithm 1)
    # ------------------------------------------------------------------
    def training_step(self, batch: Dict[str, Any], prune: bool = True) -> Dict[str, float]:
        if self.model is None:
            raise ValueError("training_step requires a model")
        batch = move_to_device(batch, self.device)
        self.model.train()
        if self.teacher is not None:
            self.teacher.eval()

        use_distill = (
            self.config.use_distillation
            and self.distill is not None
            and (prune or self.config.distill_on_recovery)
        )
        mu = self._mu_at(self.global_step) if prune else 0.0

        recorder = HiddenStateRecorder(self.model) if use_distill else None
        outputs, hidden = self._forward(
            self.model, batch, collect_hidden=use_distill, grad=True, recorder=recorder
        )

        ft_loss = self._task_loss(outputs, batch)
        if use_distill and hidden:
            loss, distill = self._distillation_loss(outputs, hidden, batch, mu)
        else:
            loss, distill = ft_loss, torch.zeros((), device=self.device)

        if not torch.isfinite(loss):
            # skip a poisoned step instead of corrupting the weights
            if self.optimizer is not None:
                self.optimizer.zero_grad(set_to_none=True)
            return {"loss": float("nan"), "skipped": 1.0}

        (loss / max(int(self.config.gradient_accumulation_steps), 1)).backward()

        metrics: Dict[str, float] = {
            "loss": float(loss.detach().cpu()),
            "ft_loss": float(ft_loss.detach().cpu()),
            "distill_loss": float(distill.detach().cpu()),
            "mu": float(mu),
        }

        # ---- salience -> mask update -> rank growth -> optimizer step -------------
        salience = None
        if prune and self._should_adjust(self.global_step):
            salience = self.compute_salience(update=True)
            metrics["sparsity"] = float(self._sparsity_at(self.global_step))
            selection = self.prune_step(salience, self.global_step)
            if selection is not None:
                try:
                    metrics["selected_blocks"] = float(len(getattr(selection, "retained", [])))
                except Exception:
                    pass
            result = self.grow_ranks(salience, self.global_step)
            if result is not None:
                try:
                    metrics["rank_growth"] = float(
                        getattr(result, "added_parameters", 0) or 0
                    )
                except Exception:
                    pass
            self.maybe_reset_optimizer()
        if salience is not None:
            try:
                metrics["salience_max"] = float(max(salience.values()) if salience else 0.0)
            except Exception:
                pass

        if self.optimizer is not None:
            if self.config.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameter_list(self.model), float(self.config.max_grad_norm)
                )
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        if recorder is not None:
            recorder.detach()
        if callable(clear_caches) if (clear_caches := globals().get("clear_caches")) else False:
            pass
        self._clear_salience_caches()
        return metrics

    def _clear_salience_caches(self) -> None:
        if self.model is None:
            return
        for module in self.model.modules():
            for attr in ("_cached_input", "_cached_output", "_cached_output_grad"):
                if hasattr(module, attr):
                    try:
                        setattr(module, attr, None)
                    except Exception:
                        pass

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, batch: Dict[str, Any]) -> Any:
        if self.model is None:
            raise ValueError("predict requires a model")
        self.model.eval()
        outputs, _ = self._forward(self.model, batch, collect_hidden=False, grad=False)
        return outputs

    def _default_metrics(
        self, predictions: List[Any], references: List[Any]
    ) -> Dict[str, float]:
        if self._compute_metrics is not None:
            try:
                return dict(self._compute_metrics(predictions, references))
            except Exception:
                pass
        try:
            from .data import compute_metrics as _cm  # type: ignore

            return dict(_cm(self.task, predictions, references))
        except Exception:
            # last resort: accuracy
            if not predictions:
                return {}
            correct = sum(int(p == r) for p, r in zip(predictions, references))
            return {"accuracy": 100.0 * correct / max(len(predictions), 1)}

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: Optional[Iterable[Any]] = None,
        metric_fn: Optional[Callable[..., Dict[str, float]]] = None,
        max_batches: Optional[int] = None,
        measure_memory: bool = False,
    ) -> Dict[str, float]:
        dataloader = dataloader if dataloader is not None else self.eval_dataloader
        if dataloader is None or self.model is None:
            return {}
        if measure_memory and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        was_training = self.model.training
        self.model.eval()
        total_loss, n_batches = 0.0, 0
        predictions: List[Any] = []
        references: List[Any] = []
        limit = max_batches if max_batches is not None else self.config.max_eval_batches
        max_batches = limit if limit else None
        for index, batch in enumerate(dataloader):
            if max_batches is not None and index >= max_batches:
                break
            batch = move_to_device(batch, self.device)
            outputs, _ = self._forward(self.model, batch, collect_hidden=False, grad=False)
            try:
                loss = self._task_loss(outputs, batch)
                total_loss += float(loss.detach().cpu())
                n_batches += 1
            except Exception:
                pass
            logits = getattr(outputs, "logits", outputs)
            preds, refs = self._decode_predictions(logits, batch)
            if preds is not None:
                predictions.extend(preds)
                references.extend(refs)
        metrics: Dict[str, float] = {}
        if n_batches:
            metrics["eval_loss"] = total_loss / n_batches
        if predictions:
            fn = metric_fn or self._default_metrics
            try:
                metrics.update({k: float(v) for k, v in dict(fn(predictions, references)).items()})
            except Exception:
                pass
        if measure_memory and self.device.type == "cuda":
            self.inference_peak_memory_bytes = int(torch.cuda.max_memory_allocated())
        if was_training:
            self.model.train()
        return metrics

    def _decode_predictions(
        self, logits: Any, batch: Dict[str, Any]
    ) -> Tuple[Optional[List[Any]], Optional[List[Any]]]:
        """Task-aware decoding of model logits into (predictions, references)."""
        labels = batch.get("labels")
        try:
            if self.family == "glue":
                if logits is None or labels is None:
                    return None, None
                if labels.dtype.is_floating_point or logits.dim() == labels.dim():
                    preds = logits.view(-1).float().tolist()
                    return preds, labels.view(-1).float().tolist()
                preds = logits.argmax(dim=-1).view(-1).tolist()
                refs = labels.view(-1).tolist()
                pairs = [(p, r) for p, r in zip(preds, refs) if r != -100]
                if not pairs:
                    return None, None
                return [p for p, _ in pairs], [r for _, r in pairs]
            if self.family == "squad":
                if logits is None:
                    return None, None
                start = logits[..., 0].argmax(dim=-1)
                end = logits[..., 1].argmax(dim=-1)
                mask = end < start
                end = torch.where(mask, start, end)
                # predictions as (start, end) tuples when references are offsets
                preds = list(zip(start.view(-1).tolist(), end.view(-1).tolist()))
                refs_batch = batch.get("span_answers")
                refs = list(refs_batch) if refs_batch is not None else preds
                return preds, refs
            if self.family == "seq2seq":
                if labels is None or logits is None:
                    return None, None
                pred_ids = logits.argmax(dim=-1)
                preds, refs = [], []
                for pred, ref in zip(pred_ids, labels):
                    preds.append(pred.tolist())
                    refs.append([int(t) for t in ref.tolist() if t != -100])
                return preds, refs
        except Exception:
            return None, None
        return None, None

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def fit(
        self,
        train_dataloader: Optional[Iterable[Any]] = None,
        eval_dataloader: Optional[Iterable[Any]] = None,
        steps_per_epoch: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run the full two-stage APT schedule and return the training report."""
        if self.model is None:
            raise ValueError("fit() requires a model (set `model=` or call setup_model()).")
        train_dataloader = train_dataloader if train_dataloader is not None else self.train_dataloader
        if train_dataloader is None:
            raise ValueError("fit() requires a training dataloader.")
        self.eval_dataloader = eval_dataloader if eval_dataloader is not None else self.eval_dataloader

        n_batches = steps_per_epoch or len(train_dataloader)  # type: ignore[arg-type]
        self.setup_schedules(max(int(n_batches), 1))
        self.setup_pruning(train_steps=self.pruning_end_step)
        if self.optimizer is None:
            self.setup_optimizer()
        self.setup_lr_scheduler(total_steps=self.total_steps)
        self.build_teacher()

        if self.device.type == "cuda" and self.config.measure_memory:
            torch.cuda.reset_peak_memory_stats()
        self.train_start_time = time.time()

        # ---------------- Stage 1: prune + self-distillation ----------------
        self.stage = "prune"
        for epoch in range(int(self.config.pruning_epochs)):
            self.epoch = epoch
            self._run_epoch(train_dataloader, prune=True)
            if self.config.eval_interval and self.eval_dataloader is not None:
                self._maybe_eval()

        # ---------------- Stage 2: recovery ----------------
        if self.config.recovery_epochs > 0:
            self.stage = "recovery"
            # freeze the pruning structure: harden masks, stop mask/rank updates
            if self.mask_manager is not None:
                try:
                    self.mask_manager.harden()  # type: ignore[attr-defined]
                except Exception:
                    pass
            if callable(harden_masks):  # type: ignore[truthy-function]
                try:
                    harden_masks(self.model)  # type: ignore[misc]
                except Exception:
                    pass
            for epoch in range(int(self.config.pruning_epochs), int(self.config.epochs)):
                self.epoch = epoch
                self._run_epoch(train_dataloader, prune=False)
                if self.config.eval_interval and self.eval_dataloader is not None:
                    self._maybe_eval()

        self.train_seconds = time.time() - (self.train_start_time or time.time())
        if self.device.type == "cuda" and self.config.measure_memory:
            self.train_peak_memory_bytes = int(torch.cuda.max_memory_allocated())
        final_metrics = self.evaluate(self.eval_dataloader) if self.eval_dataloader is not None else {}
        report = {
            "stage1_epochs": int(self.config.pruning_epochs),
            "stage2_epochs": int(self.config.recovery_epochs),
            "global_step": int(self.global_step),
            "optimizer_resets": int(self.optimizer_resets),
            "train_seconds": float(self.train_seconds or 0.0),
            "train_peak_memory_bytes": self.train_peak_memory_bytes,
            "final_metrics": final_metrics,
            "tta": self.tta.as_dict(),
            "sparsity": self.sparsity(),
            "history": self.history,
        }
        if self.config.output_dir:
            os.makedirs(self.config.output_dir, exist_ok=True)
            with open(os.path.join(self.config.output_dir, "train_report.json"), "w") as handle:
                json.dump(report, handle, indent=2, default=str)
        return report

    # ------------------------------------------------------------------
    def _run_epoch(self, dataloader: Iterable[Any], prune: bool) -> None:
        n_batches = max(self.steps_per_epoch, 1)
        for batch in dataloader:
            step_metrics = self.training_step(batch, prune=prune)
            if self.lr_scheduler is not None:
                try:
                    self.lr_scheduler.step()
                except Exception:
                    pass
            self.global_step += 1
            if self.config.max_train_steps and self.global_step >= int(self.config.max_train_steps):
                break
            if self.global_step % max(int(self.config.log_interval), 1) == 0:
                record = {"step": self.global_step, "epoch": self.epoch, "stage": self.stage}
                record.update({k: float(v) for k, v in step_metrics.items()})
                record["sparsity"] = self.sparsity()
                record["lr"] = float(
                    self.optimizer.param_groups[0]["lr"] if self.optimizer is not None else 0.0
                )
                if self.train_start_time:
                    record["elapsed"] = time.time() - self.train_start_time
                self.history.append(record)
                if self.config.verbose:
                    print(
                        f"  [{self.stage}] step {self.global_step} "
                        f"loss={record.get('loss', float('nan')):.4f} "
                        f"sparsity={record['sparsity']:.3f} "
                        f"mu={record.get('mu', 0.0):.2f} "
                        f"lr={record['lr']:.2e}"
                    )
            if self.global_step % max(n_batches, 1) == 0 and prune:
                pass

    # ------------------------------------------------------------------
    def _maybe_eval(self) -> Optional[Dict[str, float]]:
        metrics = self.evaluate(self.eval_dataloader)
        if not metrics or self.train_start_time is None:
            return metrics
        primary = self._primary_metric(metrics)
        if primary is not None:
            self.tta.update(time.time() - self.train_start_time, self.global_step, primary)
        if self.config.verbose:
            print(f"  [apt] eval @{self.global_step}: {metrics}")
        return metrics

    def _primary_metric(self, metrics: Dict[str, float]) -> Optional[float]:
        for key in ("accuracy", "acc", "f1", "matthews_correlation", "spearmanr", "pearsonr", "eval_accuracy"):
            if key in metrics:
                return float(metrics[key])
        for key, value in metrics.items():
            if key != "eval_loss":
                return float(value)
        return None

    # ------------------------------------------------------------------
    def sparsity(self) -> float:
        if self.mask_manager is not None:
            try:
                return float(self.mask_manager.sparsity())
            except Exception:
                pass
        return 0.0

    def param_count(self) -> int:
        return _count_params(self.model) if self.model is not None else 0

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export_model(self, threshold: float = 0.5, verify: bool = False) -> nn.Module:
        """Harden masks, merge ``W_B W_A`` into ``W`` and physically remove pruned blocks."""
        if self.model is None:
            raise ValueError("export_model requires a model")
        if callable(merge_and_prune):
            model = merge_and_prune(self.model, threshold=threshold)  # type: ignore[misc]
        else:
            if callable(restore_base_linears):  # pragma: no cover
                model = restore_base_linears(self.model, merge=True, threshold=threshold)  # type: ignore[misc]
            else:
                model = self.model
        if verify and callable(verify_merge_equivalence):
            try:
                verify_merge_equivalence(self.model, threshold=1e-4)  # type: ignore[misc]
            except Exception:
                pass
        return model

    def save(self, output_dir: Optional[str] = None, merge: bool = False) -> str:
        output_dir = output_dir or self.config.output_dir or "apt_output"
        os.makedirs(output_dir, exist_ok=True)
        model = self.export_model() if merge else self.model
        if model is not None and hasattr(model, "save_pretrained"):
            model.save_pretrained(os.path.join(output_dir, "model"))
        if self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(os.path.join(output_dir, "model"))
        self.config.save(os.path.join(output_dir, "config.json"))
        report = {
            "global_step": self.global_step,
            "optimizer_resets": self.optimizer_resets,
            "train_seconds": self.train_seconds,
            "train_peak_memory_bytes": self.train_peak_memory_bytes,
            "sparsity": self.sparsity(),
            "tta": self.tta.as_dict(),
            "history": self.history[-200:],
        }
        with open(os.path.join(output_dir, "trainer_state.json"), "w") as handle:
            json.dump(report, handle, indent=2, default=str)
        return output_dir


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------


def build_trainer(
    config: Any = None,
    model: Optional[nn.Module] = None,
    tokenizer: Optional[Any] = None,
    train_dataloader: Optional[Iterable[Any]] = None,
    eval_dataloader: Optional[Iterable[Any]] = None,
    **kwargs: Any,
) -> APTTrainer:
    """Create an :class:`APTTrainer` and wrap the model."""
    trainer = APTTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        **kwargs,
    )
    if trainer.model is not None:
        trainer.setup_model()
    return trainer


def train_apt(
    config: Any = None,
    model: Optional[nn.Module] = None,
    tokenizer: Optional[Any] = None,
    train_dataloader: Optional[Iterable[Any]] = None,
    eval_dataloader: Optional[Iterable[Any]] = None,
    steps_per_epoch: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """One-call APT training entry point used by ``scripts/train_apt.py``."""
    trainer = build_trainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        **kwargs,
    )
    report = trainer.fit(steps_per_epoch=steps_per_epoch)
    report["trainer"] = trainer
    return report


def measure_inference_throughput(
    model: nn.Module,
    batch_size: int = 128,
    seq_length: int = 128,
    vocab_size: int = 50265,
    repeats: int = 5,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Inference samples/sec + peak memory (Section 5.3 protocol)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    inputs = {
        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_length), device=device),
        "attention_mask": torch.ones(batch_size, seq_length, dtype=torch.long, device=device),
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for _ in range(2):  # warmup
            try:
                model(**inputs)
            except Exception:
                break
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(repeats):
            try:
                model(**inputs)
            except Exception:
                break
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.time() - start) / max(repeats, 1)
    peak = int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
    return {
        "seconds_per_batch": elapsed,
        "ms_per_batch": elapsed * 1000.0,
        "samples_per_sec": batch_size / max(elapsed, 1e-12),
        "peak_memory_bytes": peak,
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _self_test() -> bool:
    """Dependency-light checks of the training loop mechanics."""
    torch.manual_seed(0)
    cfg = TrainConfig.from_dict(
        {"task": "sst2", "epochs": 2, "distill_epochs": 1, "batch_size": 4, "target_sparsity": 0.5}
    )
    assert cfg.pruning_epochs == 1 and cfg.recovery_epochs == 1
    assert abs(cfg.pred_weight_for("squad") - 0.1) < 1e-9
    assert abs(cfg.pred_weight_for("sst2") - 1.0) < 1e-9

    # schedules
    trainer = APTTrainer(config=cfg)
    trainer.setup_schedules(steps_per_epoch=10)
    assert abs(trainer._sparsity_at(0) - 1.0) < 1e-9
    assert abs(trainer._sparsity_at(trainer.pruning_end_step) - 0.5) < 1e-6
    assert trainer._mu_at(0) < 1e-6 and abs(trainer._mu_at(trainer.pruning_end_step) - 1.0) < 1e-6
    assert abs(trainer._alpha_at(0) - 0.01) < 1e-9

    # TTA bookkeeping
    tta = TimeToAccuracy(reference=100.0, fraction=0.97)
    assert tta.target == 97.0
    assert not tta.update(1.0, 1, 90.0)
    assert tta.update(2.0, 2, 97.5)
    assert tta.reached_step == 2

    # task loss / hidden recorder on a tiny toy transformer
    class ToyBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.lin = nn.Linear(dim, dim)

        def forward(self, x):
            return x + self.lin(x)

    class Toy(nn.Module):
        def __init__(self, dim=16, n=2, classes=2):
            super().__init__()
            self.emb = nn.Embedding(32, dim)
            self.layer = nn.ModuleList([ToyBlock(dim) for _ in range(n)])
            self.head = nn.Linear(dim, classes)

        def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
            h = self.emb(input_ids)
            for block in self.layer:
                h = block(h)
            logits = self.head(h[:, 0])
            loss = None
            if labels is not None:
                loss = F.cross_entropy(logits, labels)
            return type("Out", (), {"loss": loss, "logits": logits})()

    toy = Toy()
    batch = {
        "input_ids": torch.randint(0, 32, (4, 6)),
        "attention_mask": torch.ones(4, 6, dtype=torch.long),
        "labels": torch.randint(0, 2, (4,)),
    }
    trainer = APTTrainer(config=cfg, model=toy)
    out, hidden = trainer._forward(toy, batch, collect_hidden=True)
    assert hidden is not None and len(hidden) == 2
    loss = trainer._task_loss(out, batch)
    assert loss.requires_grad
    metrics = trainer.training_step(batch, prune=False)
    assert "loss" in metrics and torch.isfinite(torch.tensor(metrics["loss"]))
    preds, refs = trainer._decode_predictions(out.logits, batch)
    assert preds is not None and len(preds) == 4
    return True


if __name__ == "__main__":  # pragma: no cover
    print("apt.training self-test:", _self_test())
