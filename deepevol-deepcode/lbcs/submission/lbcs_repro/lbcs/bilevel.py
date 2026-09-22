"""Algorithm 1: Lexicographic Bilevel Coreset Selection (LBCS) for RCS.

This module implements the paper's Algorithm 1 (Refined Coreset Selection)
together with the *inner-loop trainer hooks* it needs:

    |  Algorithm 1  Lexicographic bilevel coreset selection (LBCS) for RCS.
    |  Require: a network ``theta``, a dataset ``D``, a predefined size ``k``, and
    |           voluntary performance compromise ``epsilon``;
    |  1: Initialize masks ``m`` randomly with ``||m||_0 = k``;
    |  2: for training iteration ``t = 1, 2, ..., T`` do
    |  3:     Train the inner loop with ``D`` to converge, satisfying
    |          ``theta(m) <- arg min_theta L(m, theta)``;
    |  4:     Update masks ``m`` with ``theta(m)`` by lexicographic optimization
    |          as discussed in §3.2;
    |  5: end for
    |  Output: masks ``m`` after all training iterations.

The outer-loop mask update (step 4) is delegated to :class:`lbcs.lexiflow.LexiFlow`
(Algorithm 2, randomized direct search over the practical lexicographic
relations), and the train-then-evaluate primitive ``theta(m) -> F(m) = [f1, f2]``
is provided by :class:`lbcs.objectives.MaskObjectiveEvaluator`, which is fed the
inner trainer implemented here.

Paper facts implemented verbatim
--------------------------------
* Inner loop objective ``L(m, theta) = (1/||m||_0) * sum_i m_i * l(h(x_i; theta), y_i)``
  (Eq. (2)), returned by :func:`lbcs.objectives.inner_coreset_loss`.
* Step 3: the inner loop is trained *to convergence* (a fresh ``theta(m)`` for
  each queried mask — warm-started/finetuned when the §3.2 acceleration tricks
  are enabled).
* §5.2: inner loop = Adam, learning rate ``0.001``; ``epsilon = 0.2``, ``T = 500``.
* Figure 1 / Appendix C.3 (verbatim): "for the inner loop, the model is trained
  for 100 epochs using SGD with a learning rate of 0.1 and momentum of 0.9",
  and "for the outer loop, the probabilities are optimized by Adam with a
  learning rate of 2.5 and a cosine scheduler" (``T = 1000``).
* §3.2 acceleration: (i) first train a model with random masks, then finetune it
  with other masks; (ii) model sparsity / smaller models; (iii) group examples so
  that examples of a group share one mask entry.
* §3.2 lexicographic optimum sets:
  ``M1* = {m | f1(m) <= f1*(1+eps)}``, ``f1* = inf f1``;
  ``M2* = {m in M1* | f2(m) <= f2*}``, ``f2* = inf_{M1*} f2``.

Suggested defaults (NOT stated in the paper, kept in one place)
--------------------------------------------------------------
``delta_init = 0.1``, ``delta_lower = 1e-3`` (Algorithm 2 step size),
``batch_size = 128``, ``weight_decay = 0.0`` and, when a section does not state
``T``, ``T = 500``.  They are exposed through :class:`LBCSConfig` /
:class:`InnerTrainConfig` so that they can be changed without touching the
algorithm code.

PyTorch is imported lazily: the module stays importable (and its mask algebra
testable) in environments without PyTorch.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field, fields, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .discretize import discretize_mask
from .lexicographic import ThresholdTracker
from .masks import (
    Grouping,
    expand_mask,
    init_binary_mask,
    init_grouped_binary_mask,
    l0_norm,
    num_selected,
    reduce_mask,
    selected_indices,
)

# --------------------------------------------------------------------------- #
# Optional dependencies
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - exercised through the self-test when torch exists
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    DataLoader = None  # type: ignore
    Subset = None  # type: ignore
    _TORCH_AVAILABLE = False


LOGGER = logging.getLogger(__name__)

__all__ = [
    # configuration
    "InnerTrainConfig",
    "LBCSConfig",
    # trainer
    "TrainInfo",
    "InnerTrainer",
    "ModelStateBank",
    "build_optimizer",
    "build_scheduler",
    "make_inner_train_fn",
    # Algorithm 1
    "LBCS",
    "LexicographicBilevelCoresetSelection",
    "LBCSResult",
    "lbcs_run",
    "algorithm1",
    # constants
    "DEFAULT_T",
    "DEFAULT_EPSILON",
    "DEFAULT_DELTA_INIT",
    "DEFAULT_DELTA_LOWER",
    "FIGURE1_T",
    "SECTION52_T",
    "SECTION52_EPSILON",
    "_selftest",
]


# --------------------------------------------------------------------------- #
# Paper-stated and suggested constants
# --------------------------------------------------------------------------- #
DEFAULT_EPSILON = 0.2              # §5.2 (paper-stated)
DEFAULT_T = 500                    # §5.2 (paper-stated); suggested default elsewhere
SECTION52_EPSILON = 0.2
SECTION52_T = 500
FIGURE1_T = 1000                   # Figure 1 / Appendix C.3

DEFAULT_DELTA_INIT = 0.1           # SUGGESTED (Algorithm 2 step size)
DEFAULT_DELTA_LOWER = 1e-3         # SUGGESTED

#: inner-loop optimizers / schedules (paper-stated where noted)
SECTION52_INNER_OPTIMIZER = "adam"
SECTION52_INNER_LR = 0.001         # §5.2 paper-stated
FIGURE1_INNER_OPTIMIZER = "sgd"
FIGURE1_INNER_LR = 0.1             # Appendix C.3 paper-stated
FIGURE1_INNER_MOMENTUM = 0.9       # Appendix C.3 paper-stated
FIGURE1_INNER_EPOCHS = 100         # Appendix C.3 paper-stated
DEFAULT_INNER_EPOCHS = 100         # suggested default when unspecified
DEFAULT_BATCH_SIZE = 128           # suggested default
DEFAULT_WEIGHT_DECAY = 0.0         # suggested default


def _field_names(cls: type) -> set:
    try:
        return {f.name for f in fields(cls)}
    except TypeError:  # pragma: no cover - defensive
        return set()


# --------------------------------------------------------------------------- #
# Configuration objects
# --------------------------------------------------------------------------- #
@dataclass
class InnerTrainConfig:
    """Configuration of the inner loop ``theta(m) <- argmin L(m, theta)`` (step 3).

    Attributes
    ----------
    optimizer:
        ``"adam"`` or ``"sgd"`` (the paper uses Adam for §5.2 and SGD for Figure 1).
    lr, momentum, weight_decay:
        Optimizer hyper-parameters.
    epochs:
        Epochs of the inner loop on the coreset ("until convergence").
    batch_size, num_workers:
        DataLoader settings for the coreset.
    scheduler:
        ``None``, ``"cosine"``, ``"step"`` or ``"exponential"``.
    criterion:
        Optional loss object; defaults to ``torch.nn.CrossEntropyLoss()``.
    label_smoothing, drop_last, shuffle:
        Small training knobs (defaults follow the paper as closely as possible).
    seed:
        Optional seed for the coreset DataLoader shuffling.
    device:
        Torch device string/object; ``None`` means "auto-detect".
    max_batches:
        Truncate each epoch to this many batches (debug / smoke tests).
    early_stop_tol, early_stop_patience:
        Optional convergence criterion: stop when the relative improvement of the
        coreset loss is below ``early_stop_tol`` for ``early_stop_patience``
        epochs.  ``None`` disables early stopping (default: train ``epochs``).
    log_every:
        Log the loss every ``log_every`` epochs (0 disables logging).
    grouped:
        When ``True`` the mask is reduced to group space before the coreset is
        built (only meaningful together with a :class:`~lbcs.masks.Grouping`).
    """

    optimizer: str = SECTION52_INNER_OPTIMIZER
    lr: float = SECTION52_INNER_LR
    momentum: float = FIGURE1_INNER_MOMENTUM
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    epochs: int = DEFAULT_INNER_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    num_workers: int = 0
    scheduler: Optional[str] = None
    criterion: Any = None
    label_smoothing: float = 0.0
    shuffle: bool = True
    drop_last: bool = False
    seed: Optional[int] = None
    device: Any = None
    max_batches: Optional[int] = None
    early_stop_tol: Optional[float] = None
    early_stop_patience: int = 5
    log_every: int = 0
    grouped: bool = False

    # -- constructors for the paper's sections ----------------------------- #
    @classmethod
    def for_figure1(cls, **overrides: Any) -> "InnerTrainConfig":
        """Figure 1 inner loop: SGD, lr 0.1, momentum 0.9, 100 epochs (Appendix C.3)."""
        cfg = cls(
            optimizer=FIGURE1_INNER_OPTIMIZER,
            lr=FIGURE1_INNER_LR,
            momentum=FIGURE1_INNER_MOMENTUM,
            weight_decay=0.0,
            epochs=FIGURE1_INNER_EPOCHS,
            batch_size=DEFAULT_BATCH_SIZE,
        )
        return cfg.with_overrides(overrides)

    @classmethod
    def for_section52(cls, **overrides: Any) -> "InnerTrainConfig":
        """§5.2 inner loop: Adam with learning rate 0.001, 100 epochs."""
        cfg = cls(
            optimizer=SECTION52_INNER_OPTIMIZER,
            lr=SECTION52_INNER_LR,
            weight_decay=DEFAULT_WEIGHT_DECAY,
            epochs=DEFAULT_INNER_EPOCHS,
            batch_size=DEFAULT_BATCH_SIZE,
        )
        return cfg.with_overrides(overrides)

    @classmethod
    def from_dict(
        cls, data: Optional[Dict[str, Any]] = None, **overrides: Any
    ) -> "InnerTrainConfig":
        """Build a config from a (possibly partial) dictionary / YAML block."""
        merged = dict(data or {})
        merged.update(overrides)
        valid = {k: v for k, v in merged.items() if k in _field_names(cls)}
        return cls(**valid)

    # -- helpers ------------------------------------------------------------ #
    def with_overrides(
        self, overrides: Optional[Dict[str, Any]] = None
    ) -> "InnerTrainConfig":
        """Return a copy with the recognised ``overrides`` applied."""
        if not overrides:
            return self
        valid = {k: v for k, v in overrides.items() if k in _field_names(type(self))}
        return replace(self, **valid) if valid else self

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "criterion":
                out[f.name] = type(value).__name__ if value is not None else None
            else:
                out[f.name] = value
        return out


@dataclass
class LBCSConfig:
    """Configuration of Algorithm 1 (LBCS).

    ``k`` is the predefined coreset size, ``epsilon`` the voluntary performance
    compromise of ``f1`` and ``T`` the number of mask-update iterations.
    §5.2 uses ``epsilon = 0.2`` and ``T = 500``; Figure 1 uses ``T = 1000``.
    """

    k: int = 1000
    epsilon: float = DEFAULT_EPSILON
    T: int = DEFAULT_T
    # Algorithm 2 (outer search) hyper-parameters.
    delta_init: float = DEFAULT_DELTA_INIT
    delta_lower: float = DEFAULT_DELTA_LOWER
    # §3.2 acceleration tricks.
    warm_start: bool = True            # (i) train with random masks, then finetune
    pretrain_random_masks: bool = False
    pretrain_epochs: Optional[int] = None
    group_size: int = 1                # (iii) share one mask entry among a group
    grouped: bool = False
    sparsity: Optional[float] = None   # (ii) model sparsity / smaller models
    # bookkeeping.
    cache_evaluations: bool = True
    cache_theta: bool = False
    log_every: int = 0
    seed: Optional[int] = None
    device: Any = None
    evaluate_final: bool = True

    @classmethod
    def for_section52(cls, k: int = 1000, **overrides: Any) -> "LBCSConfig":
        cfg = cls(k=int(k), epsilon=SECTION52_EPSILON, T=SECTION52_T)
        return cfg.with_overrides(overrides)

    @classmethod
    def for_figure1(cls, k: int = 100, **overrides: Any) -> "LBCSConfig":
        cfg = cls(k=int(k), epsilon=DEFAULT_EPSILON, T=FIGURE1_T)
        return cfg.with_overrides(overrides)

    @classmethod
    def from_dict(
        cls, data: Optional[Dict[str, Any]] = None, **overrides: Any
    ) -> "LBCSConfig":
        merged = dict(data or {})
        merged.update(overrides)
        valid = {k: v for k, v in merged.items() if k in _field_names(cls)}
        return cls(**valid)

    def with_overrides(self, overrides: Optional[Dict[str, Any]] = None) -> "LBCSConfig":
        if not overrides:
            return self
        valid = {k: v for k, v in overrides.items() if k in _field_names(type(self))}
        return replace(self, **valid) if valid else self

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


# --------------------------------------------------------------------------- #
# Inner-loop trainer (Algorithm 1, step 3)
# --------------------------------------------------------------------------- #
@dataclass
class TrainInfo:
    """Result of one inner-loop training run ``theta(m)``."""

    model: Any = None
    mask: Optional[np.ndarray] = None
    num_examples: int = 0
    epochs_run: int = 0
    loss: float = float("nan")
    losses: List[float] = field(default_factory=list)
    best_loss: float = float("nan")
    num_steps: int = 0
    wall_time: float = 0.0
    optimizer: str = ""
    lr: float = 0.0
    key: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_examples": int(self.num_examples),
            "epochs_run": int(self.epochs_run),
            "loss": float(self.loss),
            "best_loss": float(self.best_loss),
            "num_steps": int(self.num_steps),
            "wall_time": float(self.wall_time),
            "optimizer": self.optimizer,
            "lr": float(self.lr),
            "key": self.key,
        }


class ModelStateBank:
    """Small LRU bank of ``theta(m)`` states, used for the §3.2 warm-start trick.

    The paper's first acceleration trick is to "first train a model with random
    masks and then finetune it with other different masks in Step 3".  Keeping a
    couple of recently trained states makes that finetuning cheap.
    """

    def __init__(self, max_entries: int = 2, to_cpu: bool = True) -> None:
        self.max_entries = max(1, int(max_entries))
        self.to_cpu = bool(to_cpu)
        self._states: Dict[str, Any] = {}
        self._order: List[str] = []

    # -- container protocol ------------------------------------------------- #
    def __len__(self) -> int:
        return len(self._states)

    def __contains__(self, key: str) -> bool:
        return key in self._states

    def keys(self) -> List[str]:
        return list(self._order)

    # -- API ---------------------------------------------------------------- #
    @staticmethod
    def _sanitize(state: Any, to_cpu: bool) -> Any:
        if not _TORCH_AVAILABLE or state is None:
            return state
        out = {}
        for name, value in state.items():
            if to_cpu and torch.is_tensor(value):
                out[name] = value.detach().to("cpu").clone()
            else:
                out[name] = value
        return out

    def save(self, key: Optional[str], model: Any) -> Optional[str]:
        """Store a copy of ``model``'s ``state_dict`` under ``key``."""
        if key is None or model is None or not _TORCH_AVAILABLE:
            return key
        if not hasattr(model, "state_dict"):
            return key
        self._states[key] = self._sanitize(copy.deepcopy(model.state_dict()), self.to_cpu)
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)
        while len(self._order) > self.max_entries:
            oldest = self._order.pop(0)
            self._states.pop(oldest, None)
        return key

    def load(self, key: Optional[str]) -> Optional[Any]:
        """Return a copy of the stored state for ``key`` (or ``None``)."""
        if key is None or key not in self._states:
            return None
        return copy.deepcopy(self._states[key])

    def latest(self) -> Optional[Any]:
        if not self._order:
            return None
        return self.load(self._order[-1])

    def latest_key(self) -> Optional[str]:
        return self._order[-1] if self._order else None

    def clear(self) -> None:
        self._states.clear()
        self._order.clear()


def build_optimizer(model: Any, config: InnerTrainConfig) -> Any:
    """Build the inner-loop optimizer (Adam for §5.2, SGD for Figure 1)."""
    if not _TORCH_AVAILABLE:  # pragma: no cover
        raise RuntimeError("PyTorch is required to build an optimizer.")
    name = (config.optimizer or "adam").lower()
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adam":
        return torch.optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    if name == "rmsprop":
        return torch.optim.RMSprop(
            params, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay
        )
    raise ValueError(f"Unsupported inner-loop optimizer: {config.optimizer!r}")


def build_scheduler(optimizer: Any, config: InnerTrainConfig, steps_per_epoch: int = 1) -> Any:
    """Build an optional LR scheduler (``"cosine"`` / ``"step"`` / ``"exponential"``)."""
    if not _TORCH_AVAILABLE or not config.scheduler:  # pragma: no cover
        return None
    name = str(config.scheduler).lower()
    total_steps = max(1, int(config.epochs) * max(1, int(steps_per_epoch)))
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, total_steps // 3), gamma=0.1
        )
    if name == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    if name in ("none", "null"):
        return None
    raise ValueError(f"Unsupported inner-loop scheduler: {config.scheduler!r}")


class InnerTrainer:
    """Trains ``theta(m)`` on the coreset selected by the mask ``m`` (step 3).

    The trainer builds a DataLoader over the *selected examples only* (an
    efficient equivalent of ``L(m, theta) = (1/||m||_0) sum_i m_i l(...)``),
    optimizes it until convergence, and optionally reuses a previously trained
    state as the initialization (finetuning, §3.2 acceleration trick (i)).

    Parameters
    ----------
    config:
        :class:`InnerTrainConfig` (Adam lr 0.001 for §5.2, SGD lr 0.1 momentum 0.9
        for Figure 1).
    device:
        Torch device (``None`` = auto-detect).
    criterion:
        Optional loss; defaults to cross entropy.
    state_bank:
        Optional :class:`ModelStateBank` for warm starts.
    grouping:
        Optional :class:`~lbcs.masks.Grouping`; when given, masks are reduced to
        group space before the coreset is materialized.
    """

    def __init__(
        self,
        config: Optional[InnerTrainConfig] = None,
        device: Any = None,
        criterion: Any = None,
        state_bank: Optional[ModelStateBank] = None,
        grouping: Optional[Grouping] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = config or InnerTrainConfig()
        self.device = _resolve_device(device if device is not None else self.config.device)
        self.criterion = criterion
        self.state_bank = state_bank
        self.grouping = grouping
        self.logger = logger or LOGGER
        self.train_log: List[Dict[str, Any]] = []

    # -- device / criterion ------------------------------------------------- #
    def resolve_criterion(self, override: Any = None) -> Any:
        if override is not None:
            return override
        if self.criterion is not None:
            return self.criterion
        if not _TORCH_AVAILABLE:  # pragma: no cover
            raise RuntimeError("PyTorch is required for inner-loop training.")
        ls = float(self.config.label_smoothing or 0.0)
        return nn.CrossEntropyLoss(label_smoothing=ls) if ls > 0 else nn.CrossEntropyLoss()

    # -- coreset materialization -------------------------------------------- #
    def binarize(self, mask: Any) -> np.ndarray:
        """Map a relaxed/grouped mask to an example-level binary mask."""
        if mask is None:
            return np.array([], dtype=np.float32)
        arr = np.asarray(_to_numpy(mask), dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return arr.astype(np.float32)
        is_binary = bool(np.all((np.abs(arr) < 1e-9) | (np.abs(arr - 1.0) < 1e-9)))
        binary = (
            arr.astype(np.float32)
            if is_binary
            else np.asarray(discretize_mask(arr, threshold=0.0), dtype=np.float32)
        )
        if self.grouping is not None and binary.shape[0] == self.grouping.num_groups:
            binary = np.asarray(expand_mask(binary, self.grouping), dtype=np.float32)
        return binary

    def coreset_indices(self, mask: Any, n: Optional[int] = None) -> np.ndarray:
        """Indices of the selected examples for ``mask`` (empty-safe)."""
        binary = self.binarize(mask)
        if n is not None and binary.shape[0] > n:
            binary = binary[:n]
        idx = np.flatnonzero(binary > 0)
        if idx.size == 0:
            # Degenerate candidate (e.g. an all-negative continuous mask): fall
            # back to the highest entries so an inner loop can still be trained.
            values = np.asarray(_to_numpy(mask), dtype=float).reshape(-1)
            if n is not None and values.size > n:
                values = values[:n]
            positive = int(np.count_nonzero(np.asarray(_to_numpy(mask), dtype=float) > 0))
            k = max(1, min(values.size, positive or 1))
            idx = np.argsort(-values)[:k]
        return np.asarray(idx, dtype=np.int64)

    def build_coreset_loader(
        self,
        mask: Any = None,
        dataset: Any = None,
        loader: Any = None,
        config: Optional[InnerTrainConfig] = None,
        n: Optional[int] = None,
    ) -> Tuple[Any, int]:
        """DataLoader over the selected examples (or over everything if ``mask`` is ``None``)."""
        cfg = config or self.config
        if not _TORCH_AVAILABLE:  # pragma: no cover
            raise RuntimeError("PyTorch is required to build a coreset loader.")

        generator = None
        if cfg.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(int(cfg.seed))

        def _loader_for(base: Any, shuffle: bool) -> Any:
            return DataLoader(
                base,
                batch_size=cfg.batch_size,
                shuffle=shuffle,
                num_workers=int(cfg.num_workers),
                drop_last=cfg.drop_last,
                generator=generator,
            )

        if mask is None:
            base = dataset if dataset is not None else getattr(loader, "dataset", None)
            if base is None:  # pragma: no cover - defensive
                raise ValueError("Either `dataset` or a loader with `.dataset` is required.")
            return _loader_for(base, cfg.shuffle), _dataset_len(base)

        base = dataset if dataset is not None else getattr(loader, "dataset", None)
        if base is None:  # pragma: no cover - defensive
            raise ValueError("A dataset (or a loader with `.dataset`) is required with a mask.")
        n_total = n or _dataset_len(base)
        idx = self.coreset_indices(mask, n=n_total)
        subset = _make_subset(base, idx)
        return _loader_for(subset, cfg.shuffle), int(idx.size)

    # -- training ----------------------------------------------------------- #
    def train(
        self,
        model: Any,
        mask: Any = None,
        dataset: Any = None,
        loader: Any = None,
        config: Optional[InnerTrainConfig] = None,
        warm_start_state: Any = None,
        return_info: bool = False,
        n: Optional[int] = None,
        **overrides: Any,
    ) -> Any:
        """Run the inner loop and return the trained ``theta(m)`` (step 3).

        Returns the model by default; with ``return_info=True`` returns a
        :class:`TrainInfo` whose ``.model`` is the trained network.
        """
        if not _TORCH_AVAILABLE:  # pragma: no cover
            raise RuntimeError("PyTorch is required for inner-loop training.")
        cfg = (config or self.config).with_overrides(overrides)
        device = _resolve_device(cfg.device if cfg.device is not None else self.device)

        if model is None:  # pragma: no cover - defensive
            raise ValueError("`model` is required for inner-loop training.")

        if warm_start_state is not None and hasattr(model, "load_state_dict"):
            try:
                model.load_state_dict(warm_start_state, strict=True)
            except Exception:  # pragma: no cover - architecture mismatch
                self.logger.debug("Warm start skipped: state_dict mismatch.")

        model.to(device)
        model.train()

        coreset_loader, num_examples = self.build_coreset_loader(
            mask=mask, dataset=dataset, loader=loader, config=cfg, n=n
        )
        criterion = self.resolve_criterion(cfg.criterion)
        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, cfg, max(1, len(coreset_loader)))

        start = time.time()
        losses: List[float] = []
        best_loss = float("inf")
        steps = 0
        epochs_run = 0
        stale = 0

        for epoch in range(int(cfg.epochs)):
            epoch_loss, epoch_steps = 0.0, 0
            for b_idx, batch in enumerate(coreset_loader):
                if cfg.max_batches is not None and b_idx >= int(cfg.max_batches):
                    break
                inputs, targets = _unpack_batch(batch)
                inputs = _to_device(inputs, device)
                targets = _to_device(targets, device)
                optimizer.zero_grad(set_to_none=True)
                logits = _forward(model, inputs)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                bs = _batch_size(inputs, targets, logits)
                epoch_loss += float(loss.detach().item()) * bs
                epoch_steps += bs
                steps += 1

            epochs_run = epoch + 1
            mean_loss = epoch_loss / max(1, epoch_steps)
            losses.append(mean_loss)
            if scheduler is not None:
                scheduler.step()

            if cfg.log_every and (epoch + 1) % int(cfg.log_every) == 0:
                self.logger.info(
                    "inner loop | epoch %d/%d | coreset=%d | loss=%.6f",
                    epoch + 1,
                    int(cfg.epochs),
                    num_examples,
                    mean_loss,
                )

            # Convergence check ("Train the inner loop ... to converge").
            if cfg.early_stop_tol is not None and cfg.early_stop_tol > 0:
                if best_loss == float("inf") or mean_loss < best_loss * (
                    1.0 - float(cfg.early_stop_tol)
                ):
                    stale = 0
                else:
                    stale += 1
                best_loss = min(best_loss, mean_loss)
                if stale >= max(1, int(cfg.early_stop_patience)):
                    break
            else:
                best_loss = min(best_loss, mean_loss)

        final_loss = losses[-1] if losses else float("nan")
        info = TrainInfo(
            model=model,
            mask=self.binarize(mask) if mask is not None else None,
            num_examples=int(num_examples),
            epochs_run=int(epochs_run),
            loss=float(final_loss),
            losses=losses,
            best_loss=float(best_loss if best_loss != float("inf") else float("nan")),
            num_steps=int(steps),
            wall_time=float(time.time() - start),
            optimizer=cfg.optimizer,
            lr=float(cfg.lr),
            key=mask_key(mask),
        )
        self.train_log.append(info.to_dict())
        return info if return_info else model


def make_inner_train_fn(
    trainer: InnerTrainer,
    model_factory: Callable[[], Any],
    dataset: Any = None,
    loader: Any = None,
    state_bank: Optional[ModelStateBank] = None,
    warm_start: bool = True,
    n: Optional[int] = None,
    keep_info: bool = False,
    info_sink: Optional[List[TrainInfo]] = None,
) -> Callable[..., Any]:
    """Build the ``inner_train_fn`` consumed by :class:`~lbcs.objectives.MaskObjectiveEvaluator`.

    The returned callable has the signature ``fn(mask, **train_kwargs) -> theta``
    (Algorithm 1, step 3).  A fresh model is produced by ``model_factory`` for
    every queried mask unless a ``model`` is supplied through the keyword
    arguments; when ``warm_start`` is enabled the state previously stored for that
    mask (or the most recent state) is loaded first — the paper's "train a model
    with random masks and then finetune it" acceleration.
    """

    def inner_train_fn(mask: Any, model: Any = None, **train_kwargs: Any) -> Any:
        base = model if model is not None else model_factory()
        bank = state_bank if state_bank is not None else trainer.state_bank
        warm_state = train_kwargs.pop("warm_start_state", None)
        if warm_state is None and warm_start and bank is not None:
            warm_state = bank.load(mask_key(mask))
            if warm_state is None:
                warm_state = bank.latest()
        info = trainer.train(
            base,
            mask=mask,
            dataset=dataset,
            loader=loader,
            warm_start_state=warm_state,
            return_info=True,
            n=n,
            **train_kwargs,
        )
        if bank is not None:
            bank.save(info.key if info.key is not None else mask_key(mask), base)
        if info_sink is not None:
            info_sink.append(info)
        return info if keep_info else base

    inner_train_fn.__name__ = "inner_train_fn"
    inner_train_fn.trainer = trainer  # type: ignore[attr-defined]
    return inner_train_fn


# --------------------------------------------------------------------------- #
# Algorithm 1: LBCS
# --------------------------------------------------------------------------- #
@dataclass
class LBCSResult:
    """Outcome of an Algorithm-1 run."""

    mask: Optional[np.ndarray] = None
    continuous_mask: Optional[np.ndarray] = None
    group_mask: Optional[np.ndarray] = None
    f1: float = float("nan")
    f2: float = float("nan")
    initial_f1: float = float("nan")
    initial_f2: float = float("nan")
    size: int = 0
    iterations: int = 0
    evaluations: int = 0
    restarts: int = 0
    delta_final: float = float("nan")
    epsilon: float = DEFAULT_EPSILON
    converged: bool = False
    stopped_reason: str = ""
    wall_time: float = 0.0
    theta: Any = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    thresholds: Optional[Dict[str, Any]] = None
    eval_history: Any = None
    mask_history: List[Any] = field(default_factory=list)
    group_size: int = 1
    n: int = 0
    k: int = 0

    # -- convenience -------------------------------------------------------- #
    @property
    def coreset_indices(self) -> np.ndarray:
        return (
            selected_indices(self.mask)
            if self.mask is not None
            else np.array([], dtype=np.int64)
        )

    def curve(self, key: str = "f1") -> np.ndarray:
        """Return a per-iteration curve (``"f1"``, ``"f2"``, ...)."""
        return np.asarray([h.get(key, np.nan) for h in self.history], dtype=float)

    def summary(self) -> Dict[str, Any]:
        return {
            "f1": float(self.f1),
            "f2": float(self.f2),
            "initial_f1": float(self.initial_f1),
            "initial_f2": float(self.initial_f2),
            "size": int(self.size),
            "k": int(self.k),
            "epsilon": float(self.epsilon),
            "iterations": int(self.iterations),
            "evaluations": int(self.evaluations),
            "restarts": int(self.restarts),
            "converged": bool(self.converged),
            "stopped_reason": self.stopped_reason,
            "wall_time": float(self.wall_time),
        }

    def to_dict(self) -> Dict[str, Any]:
        out = self.summary()
        out["mask"] = None if self.mask is None else np.asarray(self.mask).tolist()
        out["n"] = int(self.n)
        out["group_size"] = int(self.group_size)
        if self.thresholds is not None:
            out["thresholds"] = self.thresholds
        return out


class LBCS:
    """Lexicographic Bilevel Coreset Selection (Algorithm 1) for RCS.

    Parameters
    ----------
    model_factory:
        Callable returning a *fresh* network ``theta`` (the search network of the
        inner loop).  It is invoked once per queried mask; warm starting makes the
        repeated runs cheap.
    n:
        Number of candidate training examples (mask dimension).
    k:
        Predefined coreset size (``||m||_0 = k`` at initialization).
    epsilon:
        Voluntary performance compromise of ``f1`` (``0.2`` in §5.2).
    T:
        Number of mask-update iterations (``500`` in §5.2, ``1000`` for Figure 1).
    dataset:
        Training dataset used both as the candidate pool and for the inner loop.
    eval_loader:
        Loader over the *full* data used to evaluate ``f1`` (Eq. (1)).
    inner_config:
        :class:`InnerTrainConfig` of the inner loop (Adam 0.001 for §5.2, SGD 0.1
        momentum 0.9 for Figure 1).
    grouping:
        Optional :class:`~lbcs.masks.Grouping` implementing acceleration trick (iii).
    """

    def __init__(
        self,
        model_factory: Callable[[], Any],
        n: int,
        k: int,
        epsilon: float = DEFAULT_EPSILON,
        T: int = DEFAULT_T,
        dataset: Any = None,
        train_loader: Any = None,
        eval_loader: Any = None,
        inner_config: Optional[InnerTrainConfig] = None,
        criterion: Any = None,
        device: Any = None,
        seed: Optional[int] = None,
        grouping: Optional[Grouping] = None,
        group_size: int = 1,
        warm_start: bool = True,
        pretrain_random_masks: bool = False,
        pretrain_epochs: Optional[int] = None,
        cache_evaluations: bool = True,
        cache_theta: bool = False,
        delta_init: float = DEFAULT_DELTA_INIT,
        delta_lower: float = DEFAULT_DELTA_LOWER,
        log_every: int = 0,
        logger: Optional[logging.Logger] = None,
        config: Optional[LBCSConfig] = None,
    ) -> None:
        self.config = config or LBCSConfig(
            k=int(k),
            epsilon=float(epsilon),
            T=int(T),
            delta_init=float(delta_init),
            delta_lower=float(delta_lower),
            warm_start=bool(warm_start),
            pretrain_random_masks=bool(pretrain_random_masks),
            pretrain_epochs=pretrain_epochs,
            group_size=int(group_size),
            grouped=grouping is not None,
            cache_evaluations=bool(cache_evaluations),
            cache_theta=bool(cache_theta),
            log_every=int(log_every),
            seed=seed,
            device=device,
        )
        self.model_factory = model_factory
        self.n = int(n)
        self.k = int(min(max(1, int(k)), self.n))
        self.epsilon = float(epsilon)
        self.T = int(T)
        self.dataset = dataset
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.criterion = criterion
        self.device = _resolve_device(device)
        self.seed = seed
        self.grouping = grouping
        self.group_size = int(group_size) if grouping is None else int(grouping.group_size)
        self.logger = logger or LOGGER

        # Search space (group space when the grouping acceleration trick is used).
        if self.grouping is not None:
            self.search_dim = int(self.grouping.num_groups)
            self.group_size = int(self.grouping.group_size)
        else:
            self.search_dim = int(self.n)

        self.inner_config = inner_config or InnerTrainConfig(device=self.device, seed=seed)
        self.state_bank = ModelStateBank(max_entries=2)
        self.trainer = InnerTrainer(
            config=self.inner_config,
            device=self.device,
            criterion=criterion,
            state_bank=self.state_bank,
            grouping=self.grouping,
            logger=self.logger,
        )

        # Populated during :meth:`run`.
        self.evaluator: Any = None
        self.optimizer: Any = None
        self._objective_fn: Optional[Callable[[Any], np.ndarray]] = None
        self.history: List[Dict[str, Any]] = []
        self.threshold_tracker = ThresholdTracker(epsilon=self.epsilon)
        self.iteration = 0
        self.initial_mask: Optional[np.ndarray] = None
        self.current_mask: Optional[np.ndarray] = None
        self.last_evaluation: Any = None
        self.trace_entry: Dict[str, Any] = {}
        self.inner_infos: List[TrainInfo] = []
        self._rng = np.random.default_rng(self.seed if self.seed is not None else 0)
        self._pretrained = False

    # ------------------------------------------------------------------ #
    # Step 1: mask initialization -- "Initialize masks randomly with ||m||_0 = k"
    # ------------------------------------------------------------------ #
    def initialize_masks(self, k: Optional[int] = None, seed: Optional[int] = None) -> np.ndarray:
        """Random initialization with exactly ``||m||_0 = k`` (Algorithm 1, line 2)."""
        k = int(k if k is not None else self.k)
        s = self.seed if seed is None else seed
        if self.grouping is not None:
            grouped = init_grouped_binary_mask(self.grouping, k=k, seed=s)
            search_mask = np.asarray(
                reduce_mask(grouped, self.grouping, mode="any"), dtype=np.float32
            )
            self.initial_mask = search_mask
            self.current_mask = search_mask.copy()
            return search_mask
        m0 = init_binary_mask(self.n, k=min(k, self.n), seed=s)
        self.initial_mask = np.asarray(m0, dtype=np.float32)
        self.current_mask = self.initial_mask.copy()
        return self.initial_mask

    # ------------------------------------------------------------------ #
    # mask <-> search space / example space
    # ------------------------------------------------------------------ #
    def _as_search_space(self, mask: Any) -> np.ndarray:
        """Coerce an incoming mask to the search dimension (group space if grouped)."""
        arr = np.asarray(_to_numpy(mask), dtype=np.float64).reshape(-1)
        if arr.size == self.search_dim:
            return arr.astype(np.float64)
        if self.grouping is not None and arr.size == self.n:
            return np.asarray(reduce_mask(arr, self.grouping, mode="mean"), dtype=np.float64)
        if arr.size < self.search_dim:
            out = np.zeros(self.search_dim, dtype=np.float64)
            out[: arr.size] = arr
            return out
        return arr[: self.search_dim].astype(np.float64)

    def example_mask(self, mask: Any) -> np.ndarray:
        """Binary example-level mask ``m in {0,1}^n`` (``f2(m) = ||m||_0``)."""
        arr = self._as_search_space(mask)
        if np.all((np.abs(arr) < 1e-9) | (np.abs(arr - 1.0) < 1e-9)):
            binary = np.asarray(arr, dtype=np.float32)
        else:
            binary = np.asarray(discretize_mask(arr, threshold=0.0), dtype=np.float32)
        if self.grouping is not None:
            if binary.shape[0] == self.grouping.num_groups:
                binary = np.asarray(expand_mask(binary, self.grouping), dtype=np.float32)
            elif self.grouping.n is not None:
                binary = binary[: self.grouping.n]
        return binary

    # ------------------------------------------------------------------ #
    # Step 3 + f1: theta(m) -> F(m) = [f1(m), f2(m)]
    # ------------------------------------------------------------------ #
    def _build_objective(self) -> Callable[[Any], np.ndarray]:
        """Wire the inner trainer (step 3) into the full-data evaluator ``f1`` (Eq. (1))."""
        from .objectives import MaskObjectiveEvaluator

        inner_fn = make_inner_train_fn(
            trainer=self.trainer,
            model_factory=self.model_factory,
            dataset=self.dataset,
            loader=self.train_loader,
            state_bank=self.state_bank,
            warm_start=bool(self.config.warm_start),
            n=self.n,
            info_sink=self.inner_infos,
        )
        self.evaluator = MaskObjectiveEvaluator(
            inner_train_fn=inner_fn,
            full_loader=self.eval_loader,
            criterion=self.criterion,
            device=self.device,
            cache=bool(self.config.cache_evaluations),
            cache_theta=bool(self.config.cache_theta),
            logger=self.logger,
        )

        def objective(mask: Any) -> np.ndarray:
            search_mask = self._as_search_space(mask)
            # §3.2 grouping: search in group space but evaluate on example masks.
            example = self.example_mask(search_mask)
            key = mask_key(example)
            evaluation = self.evaluator.evaluate(
                example,
                force=False,
                return_theta=bool(self.config.cache_theta),
                iteration=self.iteration,
            )
            self.last_evaluation = evaluation
            self._record(evaluation, search_mask, key)
            return np.asarray([evaluation.f1, evaluation.f2], dtype=float)

        setattr(objective, "evaluate", lambda m: objective(m))
        setattr(objective, "example_mask", self.example_mask)
        return objective

    def _objective(self) -> Callable[[Any], np.ndarray]:
        if self._objective_fn is None:
            self._objective_fn = self._build_objective()
        return self._objective_fn

    def _record(self, evaluation: Any, search_mask: np.ndarray, key: str) -> None:
        """Add one evaluated point to the historical set ``H`` and the tracker ``F_H``."""
        f1 = float(getattr(evaluation, "f1", np.nan))
        f2 = float(getattr(evaluation, "f2", l0_norm(self.example_mask(search_mask))))
        self.threshold_tracker.add(f1=f1, f2=f2, key=key, mask=key, tag=f"t{self.iteration}")
        self.trace_entry = {
            "iteration": int(self.iteration),
            "f1": f1,
            "f2": f2,
            "key": key,
            "size": int(f2),
        }

    # ------------------------------------------------------------------ #
    # §3.2 acceleration trick (i): pretrain with random masks, then finetune
    # ------------------------------------------------------------------ #
    def pretrain(self, epochs: Optional[int] = None, mask: Any = None, **kwargs: Any) -> Any:
        """Train one model on a random-mask coreset to warm-start later finetuning."""
        if self._pretrained and mask is None:
            return self.state_bank.latest()
        if mask is None:
            m = self.initialize_masks() if self.initial_mask is None else self.initial_mask
        else:
            m = mask
        epochs_override = {
            "epochs": int(epochs) if epochs is not None else int(self.inner_config.epochs)
        }
        epochs_override.update(kwargs)
        example = self.example_mask(m)
        info = self.trainer.train(
            self.model_factory(),
            mask=example,
            dataset=self.dataset,
            loader=self.train_loader,
            config=self.inner_config,
            return_info=True,
            n=self.n,
            **epochs_override,
        )
        self.state_bank.save(mask_key(example), info.model)
        self._pretrained = True
        return info.model

    # ------------------------------------------------------------------ #
    # Step 4: lexicographic mask update (Algorithm 2 / LexiFlow)
    # ------------------------------------------------------------------ #
    def build_optimizer(self, **kwargs: Any) -> Any:
        """Create the Algorithm-2 randomized direct search over masks."""
        from .lexiflow import LexiFlow

        objective = self._objective()

        def callback(**info: Any) -> None:
            entry = dict(self.trace_entry) if self.trace_entry else {}
            entry.update({k: v for k, v in info.items() if not isinstance(v, np.ndarray)})
            for key in ("mask", "candidate"):
                if isinstance(info.get(key), np.ndarray):
                    entry[key] = np.asarray(info[key]).tolist()
            self.history.append(entry)
            self.iteration = int(entry.get("iteration", self.iteration + 1))

        kwargs.setdefault("epsilon", self.epsilon)
        kwargs.setdefault("delta_init", self.config.delta_init)
        kwargs.setdefault("delta_lower", self.config.delta_lower)
        kwargs.setdefault("dimension", self.search_dim)
        kwargs.setdefault("max_iters", self.T)
        kwargs.setdefault("seed", self.seed)
        kwargs.setdefault("tracker", self.threshold_tracker)
        kwargs.setdefault("callback", callback)
        kwargs.setdefault("log_every", int(self.config.log_every))
        self.optimizer = LexiFlow(objective, **kwargs)
        return self.optimizer

    # ------------------------------------------------------------------ #
    # Algorithm 1 driver
    # ------------------------------------------------------------------ #
    def run(
        self,
        T: Optional[int] = None,
        initial_mask: Any = None,
        **optimizer_kwargs: Any,
    ) -> LBCSResult:
        """Execute Algorithm 1 and return the final mask with ``F(m) = [f1, f2]``.

        The loop of ``T`` iterations is realized by the LexiFlow outer search:
        every iteration queries candidate masks, triggers the inner-loop training
        of step 3 and updates the incumbent ``m`` through the lexicographic
        relation of step 4.
        """
        start = time.time()
        if self.eval_loader is None:
            raise ValueError(
                "LBCS.run requires `eval_loader` (the full-data loader used for f1)."
            )
        if initial_mask is None:
            initial_mask = (
                self.initialize_masks() if self.initial_mask is None else self.initial_mask
            )

        # (i) optional pretraining with random masks, then finetuning.
        if bool(self.config.pretrain_random_masks):
            try:
                self.pretrain(epochs=self.config.pretrain_epochs)
            except Exception as exc:  # pragma: no cover - acceleration is optional
                self.logger.warning("Pre-training with random masks skipped: %s", exc)

        optimizer = self.build_optimizer(**optimizer_kwargs)
        initial_F = self._objective()(initial_mask)
        f1_init, f2_init = float(initial_F[0]), float(initial_F[1])

        result = optimizer.run(
            initial_mask, max_iters=int(T or self.T), initial_F=initial_F
        )

        best_continuous = getattr(result, "best_continuous", None)
        if best_continuous is None:
            best_continuous = getattr(result, "best_mask", initial_mask)
        best_search = self._as_search_space(best_continuous)

        group_mask = None
        if self.grouping is not None:
            group_binary = np.asarray(
                discretize_mask(best_search, threshold=0.0), dtype=np.float32
            )
            group_mask = group_binary

        final_mask = self.example_mask(best_search)

        # Final (fresh) evaluation of the discretized coreset mask.
        f1_final = float(result.best_F[0])
        f2_final = float(result.best_F[1])
        theta_final = None
        if bool(self.config.evaluate_final) and self.evaluator is not None:
            evaluation = self.evaluator.evaluate(
                final_mask,
                force=True,
                return_theta=True,
                iteration=int(getattr(result, "num_iterations", 0)),
            )
            f1_final = float(evaluation.f1)
            f2_final = float(evaluation.f2)
            theta_final = getattr(evaluation, "theta", None)

        thresholds = None
        try:
            thresholds = self.threshold_tracker.thresholds_info
        except Exception:  # pragma: no cover - defensive
            thresholds = None

        eval_history = None
        if self.evaluator is not None and hasattr(self.evaluator, "F_history"):
            try:
                eval_history = np.asarray(self.evaluator.F_history(), dtype=float)
            except Exception:  # pragma: no cover - defensive
                eval_history = None

        return LBCSResult(
            mask=final_mask,
            continuous_mask=best_search,
            group_mask=group_mask,
            f1=f1_final,
            f2=f2_final,
            initial_f1=f1_init,
            initial_f2=f2_init,
            size=num_selected(final_mask),
            iterations=int(getattr(result, "num_iterations", 0)),
            evaluations=int(getattr(result, "num_evaluations", 0)),
            restarts=int(getattr(result, "restarts", 0)),
            delta_final=float(getattr(result, "delta_final", float("nan"))),
            epsilon=self.epsilon,
            converged=bool(getattr(result, "converged", False)),
            stopped_reason=str(getattr(result, "stopped_reason", "")),
            wall_time=float(time.time() - start),
            theta=theta_final,
            history=list(self.history),
            thresholds=thresholds,
            eval_history=eval_history,
            mask_history=list(getattr(result, "masks_history", []) or []),
            group_size=int(self.group_size),
            n=int(self.n),
            k=int(self.k),
        )

    # Alias kept for readability in drivers / logs.
    fit = run

    # ------------------------------------------------------------------ #
    # helpers exposed for the experiment drivers
    # ------------------------------------------------------------------ #
    def evaluate_mask(self, mask: Any, force: bool = False) -> Any:
        """Evaluate one mask, returning the ``MaskEvaluation`` (train + f1)."""
        if self.evaluator is None:
            self._objective()
        return self.evaluator.evaluate(self.example_mask(mask), force=force)

    def coreset_mask(self) -> np.ndarray:
        """The current coreset as a binary example-level mask ``{0,1}^n``."""
        if self.current_mask is None:  # pragma: no cover - defensive
            raise RuntimeError("No mask yet: call initialize_masks()/run() first.")
        return self.example_mask(self.current_mask)

    def achieved_size(self) -> int:
        return int(num_selected(self.coreset_mask()))

    def thresholds(self) -> Any:
        """Current ``F_H = [f~1*, f~2*]`` (with the ``M_H^1``/``M_H^2`` sets)."""
        return self.threshold_tracker.thresholds

    def summary(self) -> Dict[str, Any]:
        info = self.threshold_tracker.thresholds_info
        return {
            "config": self.config.to_dict(),
            "inner_config": self.inner_config.to_dict(),
            "n": int(self.n),
            "k": int(self.k),
            "search_dim": int(self.search_dim),
            "group_size": int(self.group_size),
            "epsilon": float(self.epsilon),
            "T": int(self.T),
            "num_evaluations": int(len(self.history)),
            "F_H": [float(x) for x in np.asarray(info.get("F_H", [np.nan, np.nan])).astype(float)],
            "num_in_H": int(info.get("size_H", 0)),
            "num_M1": int(info.get("size_M1", 0)),
            "num_M2": int(info.get("size_M2", 0)),
        }


# Convenience aliases ------------------------------------------------------- #
LexicographicBilevelCoresetSelection = LBCS


def lbcs_run(
    model_factory: Callable[[], Any],
    n: int,
    k: int,
    eval_loader: Any = None,
    dataset: Any = None,
    train_loader: Any = None,
    inner_config: Optional[InnerTrainConfig] = None,
    epsilon: float = DEFAULT_EPSILON,
    T: int = DEFAULT_T,
    grouping: Optional[Grouping] = None,
    **kwargs: Any,
) -> LBCSResult:
    """Run Algorithm 1 once and return its :class:`LBCSResult`."""
    lbcs = LBCS(
        model_factory=model_factory,
        n=n,
        k=k,
        epsilon=epsilon,
        T=T,
        dataset=dataset,
        train_loader=train_loader,
        eval_loader=eval_loader,
        inner_config=inner_config,
        grouping=grouping,
        **kwargs,
    )
    return lbcs.run()


def algorithm1(
    model_factory: Callable[[], Any],
    n: int,
    k: int,
    eval_loader: Any = None,
    **kwargs: Any,
) -> LBCSResult:
    """Explicit spelling of :func:`lbcs_run` (Algorithm 1)."""
    return lbcs_run(model_factory=model_factory, n=n, k=k, eval_loader=eval_loader, **kwargs)


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #
def mask_key(mask: Any, decimals: int = 6) -> Optional[str]:
    """Stable key for a mask (delegates to ``lbcs.objectives.mask_key`` when possible)."""
    if mask is None:
        return None
    try:
        from .objectives import mask_key as _objective_mask_key

        return _objective_mask_key(mask, decimals=decimals)
    except Exception:  # pragma: no cover - fallback
        arr = np.asarray(_to_numpy(mask), dtype=float).reshape(-1)
        if np.all((np.abs(arr) < 1e-9) | (np.abs(arr - 1.0) < 1e-9)):
            bits = np.packbits((arr > 0.5).astype(np.uint8))
            return "b:" + bits.tobytes().hex()
        return "c:" + np.round(arr, decimals).tobytes().hex()


def _to_numpy(value: Any) -> np.ndarray:
    if _TORCH_AVAILABLE and torch.is_tensor(value):  # pragma: no cover - torch path
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _resolve_device(device: Any = None) -> Any:
    if not _TORCH_AVAILABLE:  # pragma: no cover
        return device
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device) if isinstance(device, str) else device


def _to_device(value: Any, device: Any) -> Any:
    if _TORCH_AVAILABLE and torch.is_tensor(value) and device is not None:
        return value.to(device)
    return value


def _dataset_len(dataset: Any) -> int:
    try:
        return int(len(dataset))
    except TypeError:  # pragma: no cover - defensive
        return int(len(getattr(dataset, "dataset", [])))


def _make_subset(dataset: Any, indices: np.ndarray) -> Any:
    """Build a subset view of ``dataset`` restricted to ``indices``."""
    try:
        from lbcs_repro.data.datasets import subset_dataset  # type: ignore

        return subset_dataset(dataset, indices)
    except Exception:  # pragma: no cover - fallback
        if not _TORCH_AVAILABLE:
            raise
        return Subset(dataset, [int(i) for i in indices])


def _unpack_batch(batch: Any) -> Tuple[Any, Any]:
    if isinstance(batch, (list, tuple)):
        if len(batch) >= 2:
            return batch[0], batch[1]
        raise ValueError("Batch must contain at least (inputs, targets).")
    if isinstance(batch, dict):  # pragma: no cover - defensive
        for xk, yk in (("inputs", "labels"), ("x", "y"), ("data", "targets")):
            if xk in batch and yk in batch:
                return batch[xk], batch[yk]
    raise ValueError("Unsupported batch format.")


def _forward(model: Any, inputs: Any) -> Any:
    out = model(inputs)
    if isinstance(out, (list, tuple)):
        return out[0]
    if hasattr(out, "logits"):  # pragma: no cover - HF-style outputs
        return out.logits
    return out


def _batch_size(inputs: Any, targets: Any, logits: Any) -> int:
    for value in (targets, logits, inputs):
        if _TORCH_AVAILABLE and torch.is_tensor(value) and value.dim() >= 1:
            return int(value.shape[0])
        if isinstance(value, np.ndarray) and value.ndim >= 1:
            return int(value.shape[0])
    return 1


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _selftest(verbose: bool = True) -> Dict[str, Any]:
    """Offline checks of the Algorithm-1 machinery (tiny sizes, CPU friendly)."""
    diagnostics: Dict[str, Any] = {"torch": bool(_TORCH_AVAILABLE)}

    # --- mask-level invariants (no torch needed) -------------------------- #
    m = init_binary_mask(50, k=20, seed=0)
    diagnostics["init_l0"] = int(l0_norm(m))
    diagnostics["init_ok"] = int(l0_norm(m)) == 20

    g = Grouping(50, group_size=5, seed=0)
    gm = np.asarray(reduce_mask(m, g, mode="any"), dtype=np.float32)
    diagnostics["group_dim_ok"] = int(gm.size) == int(g.num_groups)

    cfg52 = InnerTrainConfig.for_section52()
    diagnostics["inner_adam"] = (cfg52.optimizer, cfg52.lr, cfg52.epochs)
    cfg_f1 = InnerTrainConfig.for_figure1()
    diagnostics["inner_figure1"] = (
        cfg_f1.optimizer,
        cfg_f1.lr,
        cfg_f1.momentum,
        cfg_f1.epochs,
    )
    diagnostics["bank_len"] = len(ModelStateBank(max_entries=2))

    if _TORCH_AVAILABLE:
        # --- tiny end-to-end Algorithm 1 run ------------------------------ #
        torch.manual_seed(0)
        n, d, C = 60, 16, 3
        X = torch.randn(n, d)
        w = torch.randn(d, C)
        y = (X @ w).argmax(dim=1)

        class TinySet(torch.utils.data.Dataset):
            def __init__(self, features, labels):
                self.X, self.y = features, labels
                self.targets = labels

            def __len__(self):
                return self.X.shape[0]

            def __getitem__(self, i):
                return self.X[i], self.y[i]

        train_ds = TinySet(X, y)
        eval_loader = DataLoader(train_ds, batch_size=32, shuffle=False)

        def factory():
            return nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, C))

        inner = InnerTrainConfig(optimizer="adam", lr=0.01, epochs=3, batch_size=16)
        lbcs = LBCS(
            model_factory=factory,
            n=n,
            k=24,
            eval_loader=eval_loader,
            dataset=train_ds,
            inner_config=inner,
            epsilon=0.2,
            T=4,
            seed=0,
        )
        m0 = lbcs.initialize_masks()
        diagnostics["initial_size"] = int(num_selected(lbcs.example_mask(m0)))
        res = lbcs.run()
        diagnostics["final_f1"] = float(res.f1)
        diagnostics["final_f2"] = float(res.f2)
        diagnostics["initial_f1"] = float(res.initial_f1)
        diagnostics["history_len"] = int(len(res.history))
        diagnostics["thresholds"] = res.thresholds
        diagnostics["summary"] = res.summary()
        diagnostics["passed"] = bool(
            np.isfinite(res.f1) and res.f2 >= 1 and int(res.size) == int(res.f2)
        )
    else:  # pragma: no cover - numpy-only environment
        diagnostics["passed"] = bool(diagnostics["init_ok"] and diagnostics["group_dim_ok"])

    if verbose:
        for name, value in diagnostics.items():
            LOGGER.info("selftest | %s = %s", name, value)
    return diagnostics


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _selftest(verbose=True)
