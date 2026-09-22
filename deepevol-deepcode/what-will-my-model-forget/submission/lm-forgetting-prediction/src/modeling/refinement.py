"""Model-refinement engine for ``What Will My Model Forget?``.

Implements the "Model Refinement" operator of Sec. 2 of the paper and the
hyper-parameter settings of Sec. 4.1 / Appendix B / Clarifications:

* Given the base PTLM ``f_0`` and one *online* (mispredicted) example
  ``<x_i, y_i> ~ D_R``, run ``K`` gradient steps on that single example and
  return the updated model ``f_i``.
* Three tuning setups:

  - ``head``    : fine-tune the LM head only. ``K = 100`` steps.
                  LR ``1e-3`` (BART0_L) / ``1e-4`` (FLAN-T5).
  - ``lora``    : low-rank adapters (Hu et al., 2021). ``K = 30`` steps.
                  LR ``1e-5`` (BART0_L) / ``1e-4`` (FLAN-T5).
  - ``full_ft`` : fine-tune the entire model. ``K = 30`` steps.
                  LR ``1e-5`` (BART0_L) / ``1e-4`` (FLAN-T5).

* Sequential refinement over a stream of online examples uses *smaller*
  learning rates of ``1e-6`` and ``1e-5`` (LoRA / full FT), or ``1e-3`` /
  ``1e-4`` (heads only) -- Appendix D.1 / Sec. 4.1 text.

* LoRA configuration is quoted verbatim from the benchmark *Clarifications*::

      LoraConfig(
          task_type=TaskType.SEQ_2_SEQ_LM,
          inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.1,
          bias="none", target_modules=['q', 'v'],
      )

* Replay hooks are provided here (selection callable + schedule) so that
  ``src/replay/refinement_replay.py`` can drive replay-based refinement with a
  distillation loss against the *base* PTLM outputs (Buzzega et al., 2020a).

The engine is deliberately model-agnostic: it only requires the wrapped
``BaseLM`` of ``src/modeling/base_lm.py`` (attributes ``model``,
``tokenizer``, ``model_key``, ``generate`` / ``predict`` / ``token_logits``).

Notes / defaults where the paper is silent
------------------------------------------
* Optimizer: AdamW (``betas=(0.9, 0.999)``, ``weight_decay=0.0``) -- the paper
  does not specify an optimizer; this is a documented default.
* Gradient clipping to norm 1.0 is applied as a stability default.
* For head-only tuning of T5, ``lm_head`` is *untied* from the shared embedding
  before training; otherwise "head only" would silently update the embedding
  matrix too.  (The paper does not discuss tying; documented choice.)
* Every call to :meth:`RefinementEngine.refine` starts from the pristine ``f_0``
  weights (single-error-at-a-time protocol of Sec. 2).  Sequential refinement
  (``sequential=True``) instead lets the model drift across online examples.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = [
    "MODES",
    "DEFAULT_STEPS",
    "DEFAULT_LRS",
    "DEFAULT_SEQUENTIAL_LRS",
    "DEFAULT_LORA_CONFIG",
    "resolve_steps",
    "resolve_lr",
    "lora_config_for",
    "build_lora_model",
    "head_only_parameters",
    "set_head_only",
    "build_optimizer",
    "resolve_model_key",
    "RefinementEngine",
    "build_refinement_engine",
    "refine_model",
    "clone_base_lm",
    "parse_args",
    "main",
]

# --------------------------------------------------------------------------------------
# Constants taken from the paper
# --------------------------------------------------------------------------------------

MODES: Tuple[str, ...] = ("head", "lora", "full_ft")

#: Sec. 4.1: "30 steps ... when we apply LoRA or full FT, and 100 steps when we
#: only fine-tune the heads".
DEFAULT_STEPS: Dict[str, int] = {"head": 100, "lora": 30, "full_ft": 30}

#: Sec. 4.1: LoRA / full FT -> 1e-5 for BART0 Large, 1e-4 for FLAN-T5;
#: heads only -> 1e-3 and 1e-4.
DEFAULT_LRS: Dict[str, Dict[str, float]] = {
    "BART0_L": {"head": 1e-3, "lora": 1e-5, "full_ft": 1e-5},
    "FLAN-T5_L": {"head": 1e-4, "lora": 1e-4, "full_ft": 1e-4},
    "FLAN-T5_3B": {"head": 1e-4, "lora": 1e-4, "full_ft": 1e-4},
    "FLAN-T5_small": {"head": 1e-4, "lora": 1e-4, "full_ft": 1e-4},
}

#: Sec. 4.1: "When sequentially fixing multiple errors, we use smaller learning
#: rates of 1e-6 and 1e-5.  When fine-tuning heads only, we use 1e-3 and 1e-4."
DEFAULT_SEQUENTIAL_LRS: Dict[str, Dict[str, float]] = {
    "BART0_L": {"head": 1e-3, "lora": 1e-6, "full_ft": 1e-6},
    "FLAN-T5_L": {"head": 1e-4, "lora": 1e-5, "full_ft": 1e-5},
    "FLAN-T5_3B": {"head": 1e-4, "lora": 1e-5, "full_ft": 1e-5},
    "FLAN-T5_small": {"head": 1e-4, "lora": 1e-5, "full_ft": 1e-5},
}

#: Clarifications: verbatim LoRA configuration used for the FLAN-T5 experiments.
DEFAULT_LORA_CONFIG: Dict[str, Any] = {
    "task_type": "SEQ_2_SEQ_LM",
    "inference_mode": False,
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "bias": "none",
    "target_modules": ["q", "v"],
}

DEFAULT_OPTIMIZER = "AdamW"
DEFAULT_BETAS: Tuple[float, float] = (0.9, 0.999)
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_MAX_GRAD_NORM = 1.0

#: Appendix D.2 (replay schedule): 8 replayed examples every 10 update steps for
#: BART0_L / FLAN-T5_L, and 4 every 5 steps for FLAN-T5_3B.
DEFAULT_REPLAY_BATCH = 8
DEFAULT_REPLAY_EVERY = 10
DEFAULT_REPLAY_BATCH_LARGE = 4
DEFAULT_REPLAY_EVERY_LARGE = 5

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config", "config.yaml"
)


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def resolve_model_key(model_key: Optional[str], base_lm: Any = None) -> str:
    """Best-effort resolution of the model-registry key (``BART0_L``, ...)."""
    if model_key:
        return str(model_key)
    for attr in ("model_key", "name"):
        value = getattr(base_lm, attr, None)
        if isinstance(value, str) and value:
            return value
    return "BART0_L"


def _load_yaml_config(path: Optional[str] = None) -> Dict[str, Any]:
    candidates = [path] if path else []
    candidates += [DEFAULT_CONFIG_PATH, "config/config.yaml"]
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            try:
                import yaml  # local import: yaml is optional

                with open(candidate, "r", encoding="utf-8") as handle:
                    data = yaml.safe_load(handle) or {}
                if isinstance(data, Mapping):
                    return dict(data)
            except Exception:  # pragma: no cover - config is optional here
                logger.debug("Could not parse config at %s", candidate)
    return {}


def resolve_steps(mode: str = "full_ft", steps: Optional[int] = None, config: Optional[Mapping[str, Any]] = None) -> int:
    """Number of gradient steps ``K`` for one online example (Sec. 4.1)."""
    if steps is not None:
        return int(steps)
    cfg = dict(config or _load_yaml_config())
    refinement = cfg.get("refinement") or {}
    key = {"head": "head_steps"}.get(mode, "steps")
    value = refinement.get(key)
    if value is None:
        return int(DEFAULT_STEPS.get(mode, 30))
    return int(value)


def resolve_lr(
    model_key: str,
    mode: str = "full_ft",
    sequential: bool = False,
    lr: Optional[float] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> float:
    """Learning rate for ``(model_key, mode)`` per Sec. 4.1 / Appendix D.1."""
    if lr is not None:
        return float(lr)
    cfg = dict(config or _load_yaml_config())
    refinement = cfg.get("refinement") or {}
    lrs = refinement.get("lr") or {}
    model_key = resolve_model_key(model_key)
    sub_key = "sequential" if sequential else "single"
    table = lrs.get(model_key) if isinstance(lrs, Mapping) else None
    if isinstance(table, Mapping):
        sub = table.get(sub_key) if isinstance(table.get(sub_key), Mapping) else table
        if isinstance(sub, Mapping) and sub.get(mode) is not None:
            return float(sub[mode])
    fallback = DEFAULT_SEQUENTIAL_LRS if sequential else DEFAULT_LRS
    return float(fallback.get(model_key, DEFAULT_LRS["BART0_L"]).get(mode, 1e-5))


def lora_config_for(model_key: str = "BART0_L", config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Return the LoRA configuration dict (Clarifications / Sec. 4)."""
    cfg = dict(config or _load_yaml_config())
    lora = dict(DEFAULT_LORA_CONFIG)
    refinement = cfg.get("refinement") or {}
    user_lora = refinement.get("lora")
    if isinstance(user_lora, Mapping):
        for key, value in user_lora.items():
            if value is not None:
                lora[key] = value
    return lora


def build_lora_model(model: nn.Module, model_key: str = "BART0_L", config: Optional[Mapping[str, Any]] = None) -> nn.Module:
    """Wrap ``model`` with PEFT LoRA adapters on the query/value projections.

    ``r=16``, ``alpha=32``, ``dropout=0.1``, ``bias="none"``,
    ``target_modules=['q', 'v']``, ``task_type=SEQ_2_SEQ_LM`` (Clarifications).
    """
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except Exception as exc:  # pragma: no cover - peft is an optional dependency
        raise ImportError(
            "LoRA refinement requires the `peft` package (pip install peft)."
        ) from exc

    lora = lora_config_for(model_key, config=config)
    task_type = lora.get("task_type", "SEQ_2_SEQ_LM")
    task_type = getattr(TaskType, str(task_type), TaskType.SEQ_2_SEQ_LM)
    lora_config = LoraConfig(
        task_type=task_type,
        inference_mode=bool(lora.get("inference_mode", False)),
        r=int(lora.get("r", 16)),
        lora_alpha=int(lora.get("lora_alpha", 32)),
        lora_dropout=float(lora.get("lora_dropout", 0.1)),
        bias=str(lora.get("bias", "none")),
        target_modules=list(lora.get("target_modules", ["q", "v"])),
    )
    return get_peft_model(model, lora_config)


def _head_param_names(model: nn.Module) -> List[str]:
    names = [
        name
        for name, _ in model.named_parameters()
        if "lm_head" in name or "final_logits_bias" in name or "score" == name.split(".")[-1]
    ]
    return names


def head_only_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Collect the LM-head parameters (BART ``lm_head`` / T5 ``lm_head``)."""
    names = _head_param_names(model)
    params = [p for name, p in model.named_parameters() if name in names]
    return params


def set_head_only(model: nn.Module, untie: bool = True) -> List[nn.Parameter]:
    """Freeze everything except the LM head; return the now-trainable parameters.

    For T5-style models ``lm_head.weight`` is tied to the input embedding, so the
    head is un-tied first (documented default: the paper does not discuss tying).
    """
    if untie:
        try:
            from transformers.pytorch_utils import find_pruneable_heads_and_indices  # noqa: F401
        except Exception:  # pragma: no cover
            pass

    head = getattr(model, "lm_head", None)
    if head is not None and getattr(head, "weight", None) is not None:
        weight = head.weight
        tied_to: Optional[nn.Parameter] = None
        for module in model.modules():
            if module is head:
                continue
            for param in module.parameters(recurse=False):
                if param is weight:
                    tied_to = weight
                    break
        if tied_to is not None:
            head.weight = nn.Parameter(head.weight.detach().clone())
            # break HF weight tying bookkeeping if present
            for attr in ("_tied_weights_keys", "_keys_to_ignore_on_save"):
                keys = getattr(model, attr, None)
                if isinstance(keys, (list, tuple, set)):
                    try:
                        setattr(model, attr, [k for k in keys if "lm_head" not in str(k)])
                    except Exception:  # pragma: no cover
                        pass

    for param in model.parameters():
        param.requires_grad = False
    trainable = head_only_parameters(model)
    for param in trainable:
        param.requires_grad = True
    if not trainable:  # pragma: no cover - unusual architectures
        logger.warning("No lm_head parameters found; falling back to last-module parameters.")
        modules = list(model.modules())
        for param in modules[-1].parameters():
            param.requires_grad = True
        trainable = [p for p in modules[-1].parameters()]
    return trainable


def build_optimizer(
    params: Iterable[nn.Parameter],
    lr: float,
    name: str = DEFAULT_OPTIMIZER,
    betas: Sequence[float] = DEFAULT_BETAS,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
) -> torch.optim.Optimizer:
    """AdamW optimizer (documented default; the paper does not specify one)."""
    params = [p for p in params if p.requires_grad]
    kwargs = dict(lr=lr, betas=tuple(float(b) for b in betas), weight_decay=float(weight_decay))
    if str(name).lower() in ("adamw", "adam_w"):
        return torch.optim.AdamW(params, **kwargs)
    if str(name).lower() == "adam":
        return torch.optim.Adam(params, **kwargs)
    if str(name).lower() == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=float(weight_decay))
    logger.warning("Unknown optimizer %r; falling back to AdamW.", name)
    return torch.optim.AdamW(params, **kwargs)


def clone_base_lm(base_lm: Any, share_tokenizer: bool = True) -> Any:
    """Deep-copy the wrapped HF model into a fresh ``BaseLM`` instance.

    Used to materialize ``f_i`` without destroying ``f_0`` (Sec. 2 protocol:
    every online example is applied to the *base* model).
    """
    wrapper_cls = type(base_lm)
    model = copy.deepcopy(getattr(base_lm, "model", base_lm))
    tokenizer = getattr(base_lm, "tokenizer", None) if share_tokenizer else None
    kwargs: Dict[str, Any] = {}
    if tokenizer is not None:
        for attr, default in (
            ("model_key", None),
            ("max_input_len", None),
            ("max_output_len", None),
        ):
            value = getattr(base_lm, attr, None)
            if value is not None:
                kwargs[attr] = value
        try:
            return wrapper_cls(model=model, tokenizer=tokenizer, **kwargs)
        except TypeError:  # pragma: no cover - tolerate other constructor signatures
            pass
    try:
        return wrapper_cls(model, tokenizer, **kwargs)  # type: ignore[call-arg]
    except Exception:  # pragma: no cover
        return model


# --------------------------------------------------------------------------------------
# The refinement engine
# --------------------------------------------------------------------------------------


class RefinementEngine:
    """Runs ``K`` gradient steps on a single online example (Sec. 2, Sec. 4.1).

    Parameters
    ----------
    base_lm:
        A ``src.modeling.base_lm.BaseLM`` wrapper around ``f_0`` (or a raw HF
        ``nn.Module``, in which case a minimal adapter is used).
    model_key:
        Registry key used for LR lookup (``BART0_L``, ``FLAN-T5_L``,
        ``FLAN-T5_3B``, ``FLAN-T5_small``).
    mode:
        One of :data:`MODES` (``head`` / ``lora`` / ``full_ft``).
    sequential:
        If ``True`` use the smaller sequential learning rates and do *not*
        reset the model to ``f_0`` before each online example.
    replay_selector:
        Optional callable ``selector(online_index, step) -> Sequence[example]``
        returning examples to replay at the current step.  ``None`` disables
        replay (vanilla refinement), which is the ``Random``/``Vanilla FT``
        comparison point of Tables 3 and 4.
    """

    def __init__(
        self,
        base_lm: Any,
        model_key: str = "BART0_L",
        mode: str = "full_ft",
        steps: Optional[int] = None,
        lr: Optional[float] = None,
        sequential: bool = False,
        optimizer: str = DEFAULT_OPTIMIZER,
        betas: Sequence[float] = DEFAULT_BETAS,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        max_grad_norm: Optional[float] = DEFAULT_MAX_GRAD_NORM,
        config: Optional[Mapping[str, Any]] = None,
        device: Optional[Any] = None,
        dtype: Optional[Any] = None,
        seed: int = 42,
        replay_selector: Optional[Callable[..., Sequence[Mapping[str, Any]]]] = None,
        replay_batch_size: int = DEFAULT_REPLAY_BATCH,
        replay_every_n_steps: int = DEFAULT_REPLAY_EVERY,
        distill_mode: str = "kl",
        distill_temperature: float = 1.0,
        distill_weight: float = 1.0,
        online_ce_weight: float = 1.0,
        cache_teacher_logits: bool = True,
        gradient_checkpointing: bool = False,
        verbose: bool = False,
    ) -> None:
        if mode not in MODES:
            raise ValueError("mode must be one of %s (got %r)" % (MODES, mode))

        self.config = dict(config) if config is not None else _load_yaml_config()
        self.base_lm = base_lm
        self.model_key = resolve_model_key(model_key, base_lm)
        self.mode = mode
        self.sequential = bool(sequential)
        self.steps = resolve_steps(mode, steps=steps, config=self.config)
        self.lr = resolve_lr(self.model_key, mode, sequential=self.sequential, lr=lr, config=self.config)
        self.optimizer_name = optimizer
        self.betas = tuple(float(b) for b in betas)
        self.weight_decay = float(weight_decay)
        self.max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)
        self.seed = int(seed)
        self.verbose = bool(verbose)

        self.replay_selector = replay_selector
        self.replay_batch_size = int(replay_batch_size)
        self.replay_every_n_steps = int(replay_every_n_steps)
        self.distill_mode = str(distill_mode)
        self.distill_temperature = float(distill_temperature)
        self.distill_weight = float(distill_weight)
        self.online_ce_weight = float(online_ce_weight)
        self.cache_teacher_logits = bool(cache_teacher_logits)

        # running statistics for diagnostics
        self.n_online_examples = 0
        self.n_steps_run = 0
        self.last_step_losses: List[float] = []

        model = getattr(base_lm, "model", base_lm)
        self.model: nn.Module = model
        self.tokenizer = getattr(base_lm, "tokenizer", None)

        if device is not None:
            self.model.to(device)
        self.device = getattr(base_lm, "device", None) or next(self.model.parameters()).device
        self.dtype = dtype if dtype is not None else getattr(base_lm, "dtype", torch.float32)

        if gradient_checkpointing:
            self.enable_gradient_checkpointing()

        # Adapter preparation (LoRA adds new modules; head-mode toggles requires_grad)
        self.lora_applied = False
        self.trainable_param_names: List[str] = []
        self.prepare()

        # Pristine f_0 snapshot (used to restore before every non-sequential edit)
        self._pristine_state: Optional[Dict[str, torch.Tensor]] = None
        self._snapshot_reference_state()

        # Frozen teacher used for replay distillation (base PTLM outputs)
        self._teacher: Optional[nn.Module] = None
        self._teacher_logit_cache: Dict[Any, torch.Tensor] = {}

    # ------------------------------------------------------------------ setup

    def prepare(self) -> None:
        """Attach LoRA / freeze parameters according to ``self.mode``."""
        self.model.train(self.mode != "head" or True)
        if self.mode == "lora" and not self.lora_applied:
            self.model = build_lora_model(self.model, self.model_key, config=self.config)
            self.lora_applied = True
            if hasattr(self.base_lm, "model"):
                try:
                    self.base_lm.model = self.model
                except Exception:  # pragma: no cover
                    pass
            self.model.train()
        elif self.mode == "head":
            set_head_only(self.model)
        else:  # full_ft
            for param in self.model.parameters():
                param.requires_grad = True
        self.trainable_param_names = [n for n, p in self.model.named_parameters() if p.requires_grad]

    def enable_gradient_checkpointing(self) -> None:
        try:
            self.model.gradient_checkpointing_enable()
        except Exception:  # pragma: no cover
            try:
                self.model.config.use_cache = False
                self.model.gradient_checkpointing_enable()
            except Exception:
                logger.warning("Gradient checkpointing not supported by this model.")

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]

    def n_trainable_parameters(self) -> int:
        return int(sum(p.numel() for p in self.trainable_parameters()))

    def n_total_parameters(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters()))

    # -------------------------------------------------------------- snapshots

    def _snapshot_reference_state(self) -> None:
        """Keep a CPU copy of the trainable parameters of the pristine ``f_0``."""
        self._pristine_state = {
            name: param.detach().to("cpu").clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    def reset(self) -> None:
        """Restore the pristine ``f_0`` trainable parameters (Sec. 2 protocol)."""
        if not self._pristine_state:
            return
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self._pristine_state:
                    param.copy_(self._pristine_state[name].to(param.device, param.dtype))
        self.n_online_examples = 0
        self._teacher_logit_cache.clear()

    def snapshot(self) -> Dict[str, torch.Tensor]:
        """Snapshot of the *current* trainable parameters (for sequential runs)."""
        return {n: p.detach().to("cpu").clone() for n, p in self.model.named_parameters() if p.requires_grad}

    def restore(self, state: Mapping[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in state:
                    param.copy_(state[name].to(param.device, param.dtype))

    def state_dict_trainable(self) -> Dict[str, torch.Tensor]:
        return self.snapshot()

    # ------------------------------------------------------- tokenisation / loss

    def _wrapper(self) -> Any:
        """Prefer the ``BaseLM`` wrapper's (re-)implemented methods when present."""
        return self.base_lm if self.base_lm is not None and hasattr(self.base_lm, "tokenize_inputs") else None

    def _tokenize_batch(self, examples: Sequence[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
        wrapper = self._wrapper()
        inputs = [str(ex.get("input", ex.get("x", ""))) for ex in examples]
        targets = [self._target_text(ex) for ex in examples]
        if wrapper is not None:
            try:
                return wrapper.tokenize_inputs(inputs, targets)
            except Exception:  # pragma: no cover - fall through to manual path
                logger.debug("BaseLM.tokenize_inputs failed; using manual tokenisation.")
        max_in = int(getattr(self.base_lm, "max_input_len", 512) or 512)
        max_out = int(getattr(self.base_lm, "max_output_len", 64) or 64)
        enc = self.tokenizer(
            inputs, max_length=max_in, truncation=True, padding=True, return_tensors="pt"
        )
        with self.tokenizer.as_target_tokenizer() if hasattr(self.tokenizer, "as_target_tokenizer") else _null_ctx():
            labels = self.tokenizer(text_target=targets, max_length=max_out, truncation=True, padding=True, return_tensors="pt")
        if "labels" not in labels and "input_ids" in labels:
            labels = {"labels": labels["input_ids"]}
        enc.update(labels)
        return enc

    @staticmethod
    def _target_text(example: Mapping[str, Any]) -> str:
        for key in ("target", "output", "y", "label"):
            if key in example and example[key] is not None:
                value = example[key]
                if isinstance(value, (list, tuple)):
                    return str(value[0])
                return str(value)
        refs = example.get("references")
        if isinstance(refs, (list, tuple)) and refs:
            return str(refs[0])
        return ""

    #: The engine's public alias for the supervised loss
    def compute_loss(self, examples: Sequence[Mapping[str, Any]]) -> torch.Tensor:
        """Cross-entropy loss of the online example(s), teacher forcing."""
        batch = self._tokenize_batch(examples)
        batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
        labels = batch.pop("labels", None)
        if labels is None:  # pragma: no cover - defensive
            raise ValueError("Tokenisation produced no labels for the refinement batch.")
        labels = labels.clone()
        labels[labels == (getattr(self.tokenizer, "pad_token_id", -100) or -100)] = -100
        batch["labels"] = labels
        out = self.model(**batch)
        return out.loss

    # alias used in some call sites / tests
    def supervised_loss(self, examples: Sequence[Mapping[str, Any]]) -> torch.Tensor:
        return self.compute_loss(examples)

    # ------------------------------------------------------------------ teacher

    def teacher(self) -> nn.Module:
        """Frozen copy of the *base* PTLM used for replay distillation."""
        if self._teacher is None:
            teacher = copy.deepcopy(self.model)
            for param in teacher.parameters():
                param.requires_grad = False
            teacher.eval()
            self._teacher = teacher
            if self.verbose:
                logger.info("Built frozen teacher for replay distillation.")
        return self._teacher

    def teacher_logits(self, examples: Sequence[Mapping[str, Any]], key: Any = None) -> torch.Tensor:
        """Base-PTLM (teacher) logits over the replayed examples' targets."""
        if self.cache_teacher_logits and key is not None and key in self._teacher_logit_cache:
            return self._teacher_logit_cache[key]
        teacher = self.teacher()
        batch = self._tokenize_batch(examples)
        batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
        labels = batch.pop("labels", None)
        with torch.no_grad():
            out = teacher(**batch)
            logits = out.logits
        if labels is not None and logits.shape[1] != labels.shape[1]:
            # seq2seq logits are shifted by one relative to labels
            logits = logits[:, : labels.shape[1], :]
        if self.cache_teacher_logits and key is not None:
            self._teacher_logit_cache[key] = logits.detach()
        return logits

    def _replay_loss(
        self,
        online_examples: Sequence[Mapping[str, Any]],
        online_loss: torch.Tensor,
        replay_examples: Sequence[Mapping[str, Any]],
        replay_keys: Optional[Sequence[Any]] = None,
    ) -> torch.Tensor:
        """Total refinement-step loss: online CE + distillation on replayed data."""
        if not replay_examples:
            return self.online_ce_weight * online_loss
        try:
            from ..forecasters.losses import replay_total_loss
        except Exception:  # pragma: no cover - allow running standalone
            replay_total_loss = None

        batch = self._tokenize_batch(replay_examples)
        batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
        labels = batch.pop("labels", None)
        student_logits = self.model(**batch).logits
        teacher_logits = self.teacher_logits(replay_examples, key=None)
        if teacher_logits.shape[1] != student_logits.shape[1]:
            length = min(teacher_logits.shape[1], student_logits.shape[1])
            teacher_logits = teacher_logits[:, :length, :]
            student_logits = student_logits[:, :length, :]

        if replay_total_loss is not None:
            online_batch = self._tokenize_batch(online_examples)
            online_batch = {k: v.to(self.device) for k, v in online_batch.items() if torch.is_tensor(v)}
            online_labels = online_batch.pop("labels", None)
            online_logits = self.model(**online_batch).logits
            if online_labels is not None and online_logits.shape[1] != online_labels.shape[1]:
                length = min(online_logits.shape[1], online_labels.shape[1])
                online_logits = online_logits[:, :length, :]
                online_labels = online_labels[:, :length]
            return replay_total_loss(
                online_logits,
                online_labels,
                student_logits,
                teacher_logits,
                replay_labels=labels,
                distill_weight=self.distill_weight,
                distill_temperature=self.distill_temperature,
                distill_mode=self.distill_mode,
                online_ce_weight=self.online_ce_weight,
            )

        # Fallback: local KL distillation (Buzzega et al., 2020a style).
        temperature = max(self.distill_temperature, 1e-6)
        log_probs = torch.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = torch.softmax(teacher_logits / temperature, dim=-1)
        loss = -(teacher_probs * log_probs).sum(dim=-1).mean() * (temperature ** 2)
        return self.online_ce_weight * online_loss + self.distill_weight * loss

    # --------------------------------------------------------------- optimise

    def refine(
        self,
        x_i: Any = None,
        y_i: Any = None,
        steps: Optional[int] = None,
        replay_selector: Optional[Callable[..., Sequence[Mapping[str, Any]]]] = None,
        reset: Optional[bool] = None,
        return_model: bool = True,
        online_index: Optional[int] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Fine-tune ``f_0`` on one online example and return ``f_i``.

        ``x_i`` may be an example mapping (``{"input":..., "target":...}``),
        an ``(input, target)`` tuple, or a plain input string (then ``y_i`` is
        required).  Returns ``(f_i, info)`` where ``info`` records the number of
        steps, the learning rate and the step losses.
        """
        example = self._normalize_online_example(x_i, y_i)
        if reset is None:
            reset = not self.sequential
        if reset:
            self.reset()

        model = self.model
        model.train()
        if hasattr(model, "config"):
            try:
                model.config.use_cache = False
            except Exception:  # pragma: no cover
                pass

        params = self.trainable_parameters()
        optimizer = build_optimizer(
            params, self.lr, name=self.optimizer_name, betas=self.betas, weight_decay=self.weight_decay
        )
        selector = replay_selector or self.replay_selector

        n_steps = int(steps if steps is not None else self.steps)
        losses: List[float] = []
        n_replayed = 0
        torch.manual_seed(self.seed + (online_index or 0))

        for step in range(n_steps):
            optimizer.zero_grad(set_to_none=True)
            online_loss = self.compute_loss([example])
            replay_examples: Sequence[Mapping[str, Any]] = ()
            if selector is not None and self.replay_every_n_steps > 0:
                if (step + 1) % self.replay_every_n_steps == 0:
                    replay_examples = self._select_replay(selector, online_index, step) or ()
                    n_replayed += len(replay_examples)
            loss = self._replay_loss([example], online_loss, replay_examples)
            loss.backward()
            if self.max_grad_norm is not None and self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach()))
            self.n_steps_run += 1

        self.n_online_examples += 1
        self.last_step_losses = losses
        if hasattr(model, "config"):
            try:
                model.config.use_cache = True
            except Exception:  # pragma: no cover
                pass

        info = {
            "online_index": online_index,
            "steps": n_steps,
            "lr": self.lr,
            "mode": self.mode,
            "model_key": self.model_key,
            "sequential": self.sequential,
            "n_replayed": n_replayed,
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
        }
        if self.verbose:
            logger.info(
                "Refined on online example (mode=%s, steps=%d, lr=%g): loss %.4f -> %.4f",
                self.mode, n_steps, self.lr, info["loss_first"] or 0.0, info["loss_last"] or 0.0,
            )
        f_i: Any = None
        if return_model:
            f_i = clone_base_lm(self.base_lm if self.base_lm is not None else self.model, share_tokenizer=True)
            f_i.model = self.model
            try:
                f_i.model_key = self.model_key
            except Exception:  # pragma: no cover
                pass
        return f_i, info

    #: aliases used by different call sites / scripts
    def refine_on_example(self, *args: Any, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        return self.refine(*args, **kwargs)

    def update(self, *args: Any, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        return self.refine(*args, **kwargs)

    def train_step(self, example: Mapping[str, Any], **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        return self.refine(example, **kwargs)

    def fit_on_example(self, example: Mapping[str, Any], **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
        return self.refine(example, **kwargs)

    def __call__(self, online: Any = None, y_i: Any = None, **kwargs: Any) -> Any:
        """Callable interface so the engine can be passed as ``refine_fn``.

        ``refine_fn`` in ``src/forgetting/ground_truth.py`` is an opaque
        callable that returns the refined model; we return ``f_i`` only.
        """
        f_i, _info = self.refine(online, y_i, **kwargs)
        return f_i

    @contextmanager
    def refine_context(self, example: Any, y_i: Any = None, **kwargs: Any) -> Iterator[Any]:
        """Context manager yielding ``f_i`` and restoring ``f_0`` on exit."""
        state = self.snapshot()
        try:
            f_i, _info = self.refine(example, y_i, reset=not self.sequential, **kwargs)
            yield f_i
        finally:
            self.restore(state)

    #: alias for the context manager (some call sites use ``refine_with``)
    def refine_with(self, example: Any, y_i: Any = None, **kwargs: Any):
        return self.refine_context(example, y_i, **kwargs)

    # ------------------------------------------------------------- sequential

    def refine_stream(
        self,
        online_examples: Sequence[Mapping[str, Any]],
        replay_selector: Optional[Callable[..., Sequence[Mapping[str, Any]]]] = None,
        progress: bool = False,
        reset: bool = True,
    ) -> Dict[str, Any]:
        """Sequentially fix a stream of errors (Sec. 5.2 / Figure 3 protocol)."""
        if reset:
            self.reset()
        if self.sequential is False:
            logger.warning(
                "refine_stream() called with sequential=False: using the single-example learning rate "
                "%g. Set sequential=True for the smaller sequential LRs of Sec. 4.1.",
                self.lr,
            )
        infos: List[Dict[str, Any]] = []
        iterator: Iterable[Any] = online_examples
        if progress:
            try:
                from tqdm import tqdm  # type: ignore

                iterator = tqdm(online_examples, desc="refine_stream")
            except Exception:  # pragma: no cover
                iterator = online_examples
        for idx, example in enumerate(iterator):
            _f_i, info = self.refine(
                example, reset=False, replay_selector=replay_selector, online_index=idx
            )
            infos.append(info)
        return {"n_examples": len(infos), "steps": self.n_steps_run, "infos": infos}

    # ---------------------------------------------------------------- helpers

    def _select_replay(
        self, selector: Callable[..., Sequence[Mapping[str, Any]]], online_index: Optional[int], step: int
    ) -> List[Mapping[str, Any]]:
        batch = self.replay_batch_size
        try:
            selected = selector(online_index, step)
        except TypeError:
            try:
                selected = selector(online_index)
            except TypeError:
                selected = selector()
        if selected is None:
            return []
        selected = list(selected)
        if batch and len(selected) > batch:
            rng = random.Random(self.seed + step)
            selected = rng.sample(selected, batch)
        return selected

    @staticmethod
    def _normalize_online_example(x_i: Any, y_i: Any = None) -> Mapping[str, Any]:
        if isinstance(x_i, Mapping):
            example = dict(x_i)
            if "target" not in example and y_i is not None:
                example["target"] = y_i
            return example
        if isinstance(x_i, (tuple, list)) and len(x_i) == 2 and y_i is None:
            return {"input": x_i[0], "target": x_i[1]}
        if isinstance(x_i, int) and y_i is None:
            raise ValueError(
                "RefinementEngine.refine() needs an example mapping or (input, target); "
                "got an index (%r). Pass `online_examples[idx]`." % (x_i,)
            )
        return {"input": "" if x_i is None else str(x_i), "target": "" if y_i is None else str(y_i)}

    # ------------------------------------------------------------ persistence

    def config_dict(self) -> Dict[str, Any]:
        return {
            "model_key": self.model_key,
            "mode": self.mode,
            "steps": self.steps,
            "lr": self.lr,
            "sequential": self.sequential,
            "optimizer": self.optimizer_name,
            "betas": list(self.betas),
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "seed": self.seed,
            "replay_batch_size": self.replay_batch_size,
            "replay_every_n_steps": self.replay_every_n_steps,
            "distill_mode": self.distill_mode,
            "distill_temperature": self.distill_temperature,
            "distill_weight": self.distill_weight,
            "online_ce_weight": self.online_ce_weight,
            "n_trainable_parameters": self.n_trainable_parameters(),
            "n_total_parameters": self.n_total_parameters(),
            "trainable_param_names": self.trainable_param_names[:50],
        }

    def save(self, path: str, extra: Optional[Mapping[str, Any]] = None) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "config": self.config_dict(),
            "state_dict": {k: v.detach().to("cpu") for k, v in self.model.state_dict().items()},
            "extra": dict(extra or {}),
        }
        torch.save(payload, path)
        return path

    def load_state(self, path: str, strict: bool = False) -> Dict[str, Any]:
        payload = torch.load(path, map_location="cpu")
        state = payload.get("state_dict", payload)
        missing, unexpected = self.model.load_state_dict(state, strict=strict)
        if missing or unexpected:
            logger.info("Loaded refinement state (missing=%d, unexpected=%d)", len(missing), len(unexpected))
        return payload


def _null_ctx():  # pragma: no cover - tiny helper for tokenizers without as_target_tokenizer
    import contextlib

    return contextlib.nullcontext()


def build_refinement_engine(
    base_lm: Any,
    model_key: str = "BART0_L",
    mode: str = "full_ft",
    config: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> RefinementEngine:
    """Factory used by the forgetting scripts.

    Reads defaults from ``config/config.yaml`` (``refinement`` section) unless
    explicit values are supplied.  Replay schedule defaults follow Appendix D.2:
    8 examples every 10 steps, or 4 every 5 steps for the 3B model.
    """
    cfg = dict(config) if config is not None else _load_yaml_config()
    refinement = cfg.get("refinement") or {}
    replay_cfg = cfg.get("replay") or {}

    model_key = resolve_model_key(model_key, base_lm)
    is_large_model = "3B" in model_key.upper()
    kwargs.setdefault("steps", None)
    kwargs.setdefault("lr", None)
    kwargs.setdefault("optimizer", refinement.get("optimizer", DEFAULT_OPTIMIZER))
    kwargs.setdefault("betas", refinement.get("betas", DEFAULT_BETAS))
    kwargs.setdefault("weight_decay", refinement.get("weight_decay", DEFAULT_WEIGHT_DECAY))
    kwargs.setdefault("max_grad_norm", refinement.get("max_grad_norm", DEFAULT_MAX_GRAD_NORM))
    kwargs.setdefault(
        "replay_batch_size",
        replay_cfg.get("batch_size_large" if is_large_model else "batch_size", DEFAULT_REPLAY_BATCH_LARGE if is_large_model else DEFAULT_REPLAY_BATCH),
    )
    kwargs.setdefault(
        "replay_every_n_steps",
        replay_cfg.get("every_n_steps_large" if is_large_model else "every_n_steps", DEFAULT_REPLAY_EVERY_LARGE if is_large_model else DEFAULT_REPLAY_EVERY),
    )
    kwargs.setdefault("distill_mode", replay_cfg.get("distillation", "kl"))
    kwargs.setdefault("distill_temperature", replay_cfg.get("distill_temperature", 1.0))
    kwargs.setdefault("distill_weight", replay_cfg.get("distill_weight", 1.0))
    kwargs.setdefault("config", cfg)
    return RefinementEngine(base_lm, model_key=model_key, mode=mode, **kwargs)


def refine_model(
    base_lm: Any,
    example: Mapping[str, Any],
    model_key: str = "BART0_L",
    mode: str = "full_ft",
    **kwargs: Any,
) -> Tuple[Any, Dict[str, Any]]:
    """One-shot convenience: refine ``f_0`` on ``example`` -> ``(f_i, info)``."""
    engine = build_refinement_engine(base_lm, model_key=model_key, mode=mode, **kwargs)
    return engine.refine(example, return_model=True)


# --------------------------------------------------------------------------------------
# CLI smoke test
# --------------------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model-refinement engine (Sec. 2 / Sec. 4.1).")
    parser.add_argument("--model-key", default="FLAN-T5_small", choices=["BART0_L", "FLAN-T5_L", "FLAN-T5_3B", "FLAN-T5_small"])
    parser.add_argument("--mode", default="head", choices=list(MODES))
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--replay-mode", default="none",
                        choices=["none", "random", "threshold", "logit", "representation", "gt"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--input", default="Question: What is the capital of France? Answer:")
    parser.add_argument("--target", default="Paris")
    parser.add_argument("--self-test", action="store_true", help="Run the tiny synthetic engine test (no LM needed).")
    return parser.parse_args(argv)


def _self_test() -> int:
    """Smoke-test the engine on a tiny randomly initialised T5 (no downloads)."""
    from transformers import T5Config, T5ForConditionalGeneration, T5TokenizerFast

    cfg = T5Config(vocab_size=128, d_model=32, d_ff=64, num_layers=2, num_decoder_layers=2,
                   num_heads=2, d_kv=16, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
    model = T5ForConditionalGeneration(cfg)
    tokenizer = T5TokenizerFast.from_pretrained("t5-small")
    tokenizer.pad_token = tokenizer.eos_token

    class _TinyWrapper:
        model_key = "FLAN-T5_small"
        max_input_len = 32
        max_output_len = 8

        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer
            self.device = torch.device("cpu")
            self.dtype = torch.float32

        def tokenize_inputs(self, inputs, targets):
            enc = self.tokenizer(inputs, max_length=32, truncation=True, padding=True, return_tensors="pt")
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(targets, max_length=8, truncation=True, padding=True, return_tensors="pt")
            enc["labels"] = labels["input_ids"]
            return enc

    wrapper = _TinyWrapper(model, tokenizer)
    example = {"input": "translate English to French: hello", "target": "bonjour"}
    for mode in MODES:
        engine = build_refinement_engine(
            wrapper, model_key="FLAN-T5_small", mode=mode, steps=2, lr=1e-3, config={}
        )
        f_i, info = engine.refine(example)
        assert info["steps"] == 2 and info["loss_last"] is not None
        print("mode=%-8s trainable=%-9d loss %.4f -> %.4f" % (mode, engine.n_trainable_parameters(), info["loss_first"], info["loss_last"]))
        del f_i, engine
    print("refinement self-test OK")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args(argv)
    if args.self_test:
        return _self_test()

    from .base_lm import load_base_lm

    base_lm = load_base_lm(args.model_key, device=args.device, dtype=args.dtype)
    engine = build_refinement_engine(base_lm, model_key=args.model_key, mode=args.mode,
                                    steps=args.steps, lr=args.lr, sequential=args.sequential)
    print("Refinement engine:", json.dumps(engine.config_dict(), indent=2, default=str))
    before = base_lm.predict([args.input], max_new_tokens=args.max_new_tokens)[0]
    f_i, info = engine.refine({"input": args.input, "target": args.target})
    engine.model.eval()
    after = base_lm.predict([args.input], max_new_tokens=args.max_new_tokens)[0]
    print("prediction before: %r" % (before,))
    print("prediction after : %r" % (after,))
    print("info:", json.dumps(info, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
