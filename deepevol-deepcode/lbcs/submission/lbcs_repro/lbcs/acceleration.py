"""Acceleration helpers for Lexicographic Bilevel Coreset Selection (LBCS).

This module implements the three "tricks for acceleration" of Section 3.2
(*Algorithm flow and tricks for acceleration*) of the paper:

    "The computational consumption of Algorithm 1 originates from the model
    training in the inner loop (Step 3) and mask updates in the outer loop
    (Step 4).

    * To speed up the inner loop, we can first train a model with random masks
      and then finetune it with other different masks in Step 3.
    * Also, we can employ model sparsity and make the trained model smaller for
      faster training.
    * To accelerate the outer loop, the mask search space can be narrowed by
      treating several examples as a group. The examples in the same group
      share the same mask in coreset selection."

Accordingly the module exposes three focused components plus a facade:

    (i)   pretrain with random masks, then finetune   -> ``WarmStarter``,
          ``WarmStartSchedule``, ``warm_start_train``,
          ``pretrain_with_random_masks``, ``finetune_with_mask``
    (ii)  model sparsity / smaller trained models     -> ``MagnitudePruner``,
          ``SparseModelFactory``, ``apply_magnitude_pruning``,
          ``model_sparsity``, ``parameter_count``
    (iii) grouped (narrowed) outer-loop search space  -> ``GroupSearchSpace``,
          ``build_search_space``, ``search_space_dimension``,
          ``acceleration_ratio``
    facade                                            -> ``AccelerationConfig``,
          ``AccelerationManager``

Knobs tied to the paper's own experiment settings (inner-loop optimizers and
epochs) are re-used from ``lbcs.bilevel``'s constants: Adam with lr 0.001 for
Section 5.2 and SGD with lr 0.1 / momentum 0.9 for 100 epochs for Figure 1
(Appendix C.3).  Values the paper does *not* specify (number of pretraining
masks, sparsity amount, group size) are labelled ``SUGGESTED`` below.

PyTorch is a soft dependency: the grouped-search-space algebra, sparsity
bookkeeping and the configuration objects work with NumPy only, whereas
model-touching routines raise a clear error when torch is missing.
"""

from __future__ import annotations

import copy
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # torch is optional -----------------------------------------------------
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only without torch
    torch = None  # type: ignore
    nn = None  # type: ignore
    _TORCH_AVAILABLE = False


from .masks import (
    Grouping,
    expand_mask,
    init_binary_mask,
    init_continuous_mask,
    init_grouped_binary_mask,
    num_selected,
    reduce_mask,
)

LOGGER = logging.getLogger(__name__)

__all__ = [
    # configuration / facade
    "AccelerationConfig",
    "AccelerationManager",
    # (i) warm start
    "WarmStartSchedule",
    "WarmStarter",
    "warm_start_train",
    "pretrain_with_random_masks",
    "finetune_with_mask",
    # (ii) sparsity
    "SparsityConfig",
    "MagnitudePruner",
    "SparseModelFactory",
    "apply_magnitude_pruning",
    "model_sparsity",
    "parameter_count",
    "model_size_mb",
    "sparsity_report",
    "make_sparse_model_factory",
    # (iii) grouping
    "GroupSearchSpace",
    "build_search_space",
    "search_space_dimension",
    "acceleration_ratio",
    "group_mask_from_example_mask",
    "example_mask_from_group_mask",
    # constants
    "DEFAULT_GROUP_SIZE",
    "SUGGESTED_GROUP_SIZE",
    "SUGGESTED_SPARSITY_AMOUNT",
    "SUGGESTED_PRETRAIN_MASKS",
    "DEFAULT_PRETRAIN_EPOCHS",
]


# --------------------------------------------------------------------------- #
# Defaults.  ``SUGGESTED`` = not numerically specified by the paper.
# --------------------------------------------------------------------------- #

DEFAULT_GROUP_SIZE: int = 1
"""Ungrouped (identity) search; the grouping trick needs ``group_size > 1``."""

SUGGESTED_GROUP_SIZE: int = 8
"""SUGGESTED group size for the outer-loop acceleration trick.

The paper only says "several examples as a group" (Section 3.2); no value given.
"""

SUGGESTED_SPARSITY_AMOUNT: float = 0.5
"""SUGGESTED fraction of weights removed when "employing model sparsity".

Not numerically specified in the paper (Section 3.2).
"""

SUGGESTED_PRETRAIN_MASKS: int = 1
"""SUGGESTED number of random masks used by the initial (pretraining) training.

The paper says "first train a model with random masks" without a count.
"""

DEFAULT_PRETRAIN_EPOCHS: int = 100
"""SUGGESTED pretraining length, matching the paper's inner-loop budget."""


# --------------------------------------------------------------------------- #
# (iii) Grouped search space (torch-free)
# --------------------------------------------------------------------------- #


@dataclass
class GroupSearchSpace:
    """Narrow the outer-loop mask search space by grouping examples.

    "the mask search space can be narrowed by treating several examples as a
    group. The examples in the same group share the same mask in coreset
    selection." (Section 3.2)

    The decision dimension drops from ``n`` to ``G = ceil(n / group_size)`` while
    all evaluations stay at the example level (groups are expanded back to ``n``
    entries before the coreset loss ``L(m, theta)`` and the full-data objective
    ``f_1(m)`` are computed).
    """

    n: int
    group_size: int = DEFAULT_GROUP_SIZE
    seed: Optional[int] = None
    shuffle: bool = True

    def __post_init__(self) -> None:
        self.n = int(self.n)
        self.group_size = max(1, int(self.group_size))
        self.grouping = Grouping(
            self.n, group_size=self.group_size, seed=self.seed, shuffle=self.shuffle
        )

    # -- geometry ----------------------------------------------------------- #
    @property
    def num_groups(self) -> int:
        return int(self.grouping.num_groups)

    @property
    def G(self) -> int:
        """``G``: the (narrowed) search-space size seen by Algorithm 2."""
        return self.num_groups

    @property
    def dimension(self) -> int:
        return self.num_groups

    @property
    def is_grouped(self) -> bool:
        return self.num_groups != self.n

    @property
    def compression(self) -> float:
        return float(self.n) / float(self.num_groups) if self.num_groups else 1.0

    def group_of(self, index: int) -> int:
        return int(self.grouping.group_of(index))

    def group_sizes(self) -> np.ndarray:
        return np.asarray(self.grouping.group_sizes, dtype=np.int64)

    def __len__(self) -> int:
        return self.num_groups

    # -- mask conversion ---------------------------------------------------- #
    def expand(self, group_mask: Sequence[float]) -> np.ndarray:
        """Group-level mask -> example-level mask of length ``n``."""
        arr = np.asarray(group_mask, dtype=np.float32).reshape(-1)
        if not self.is_grouped:
            return arr[: self.n]
        return np.asarray(self.grouping.expand(arr), dtype=np.float32)

    def reduce(self, example_mask: Sequence[float], mode: str = "mean") -> np.ndarray:
        """Example-level mask -> group-level mask (``mean`` = group selection rate)."""
        arr = np.asarray(example_mask, dtype=np.float32).reshape(-1)
        if not self.is_grouped:
            return arr[: self.n]
        return np.asarray(self.grouping.reduce(arr, mode=mode), dtype=np.float32)

    # -- initialization ----------------------------------------------------- #
    def init_group_mask(
        self,
        k: Optional[int] = None,
        seed: Optional[int] = None,
        continuous: bool = False,
        dtype: Any = np.float32,
    ) -> np.ndarray:
        """Random group-level mask realising (approximately) ``||m||_0 = k``.

        With grouping, exact ``||m||_0 = k`` generally cannot be attained because
        whole groups share one entry; the achievable size is quantised by the
        group sizes.  ``k`` defaults to ``n // 2`` (Figure 1's random subset).
        """
        k = self.n // 2 if k is None else int(k)
        rng = np.random.default_rng(seed)
        if continuous:
            example = init_binary_mask(self.n, k=k, generator=rng, dtype=dtype)
            return self.reduce(example)
        example = init_grouped_binary_mask(self.grouping, k=k, generator=rng, dtype=dtype)
        return self.reduce(example)

    def init_example_mask(
        self,
        k: Optional[int] = None,
        seed: Optional[int] = None,
        continuous: bool = False,
        dtype: Any = np.float32,
    ) -> np.ndarray:
        """Random example-level mask with exactly ``||m||_0 = k``."""
        k = self.n // 2 if k is None else int(k)
        rng = np.random.default_rng(seed)
        if continuous:
            return init_continuous_mask(self.n, k=k, generator=rng, dtype=dtype)
        return init_binary_mask(self.n, k=k, generator=rng, dtype=dtype)

    # -- outer-loop hooks --------------------------------------------------- #
    def sample_direction(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Uniform direction on the unit sphere of the *group* space."""
        rng = np.random.default_rng() if rng is None else rng
        v = rng.normal(size=self.dimension)
        norm = float(np.linalg.norm(v))
        return v / norm if norm > 0 else v

    def selected_examples(self, group_mask: Sequence[float]) -> int:
        """Number of examples selected by a group-level mask (i.e. ``f_2``)."""
        return int(num_selected(self.expand(group_mask)))

    def size_report(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "group_size": self.group_size,
            "num_groups": self.num_groups,
            "dimension": self.dimension,
            "compression": self.compression,
            "acceleration_ratio": acceleration_ratio(self.n, self.group_size),
            "is_grouped": self.is_grouped,
        }

    def to_dict(self) -> Dict[str, Any]:
        return self.size_report()


def build_search_space(
    n: int, group_size: int = DEFAULT_GROUP_SIZE, seed: Optional[int] = None, **kwargs: Any
) -> GroupSearchSpace:
    """Create the narrowed outer-loop search space (identity when ``group_size == 1``)."""
    return GroupSearchSpace(n=n, group_size=group_size, seed=seed, **kwargs)


def search_space_dimension(n: int, group_size: int = DEFAULT_GROUP_SIZE) -> int:
    """Search-space dimension after grouping: ``ceil(n / group_size)``."""
    group_size = max(1, int(group_size))
    return int(math.ceil(int(n) / group_size))


def acceleration_ratio(n: int, group_size: int = DEFAULT_GROUP_SIZE) -> float:
    """How many times smaller the outer-loop decision problem becomes."""
    dim = search_space_dimension(n, group_size)
    return float(n) / float(dim) if dim > 0 else 1.0


def group_mask_from_example_mask(
    example_mask: Sequence[float], grouping: Optional[Grouping] = None, mode: str = "mean"
) -> np.ndarray:
    """Reduce an example-level mask to group level (identity without grouping)."""
    return np.asarray(reduce_mask(np.asarray(example_mask), grouping, mode=mode))


def example_mask_from_group_mask(
    group_mask: Sequence[float], grouping: Optional[Grouping] = None
) -> np.ndarray:
    """Expand a group-level mask back to the example level."""
    return np.asarray(expand_mask(np.asarray(group_mask), grouping))


# --------------------------------------------------------------------------- #
# (ii) Model sparsity / smaller trained models
# --------------------------------------------------------------------------- #


@dataclass
class SparsityConfig:
    """Configuration of the "employ model sparsity / smaller model" trick."""

    amount: float = 0.0
    """Fraction of weights zeroed out (``0.0`` = dense baseline)."""

    scope: str = "all"
    """One of ``"all"``, ``"hidden"`` (exclude the output layer), ``"linear"``, ``"conv"``."""

    include_bias: bool = False
    structural: bool = False
    """``True`` targets whole output neurons (channels) -> genuinely smaller model."""

    reapply_every_step: bool = True
    """Re-apply the pruning mask after each optimizer step to keep zeros frozen."""

    def enabled(self) -> bool:
        return self.amount > 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "amount": self.amount,
            "scope": self.scope,
            "include_bias": self.include_bias,
            "structural": self.structural,
            "reapply_every_step": self.reapply_every_step,
        }


class MagnitudePruner:
    """L1-magnitude (unstructured) pruner that keeps the *large* weights.

    Stores a boolean mask per parameter on
    ``model._lbcs_sparsity_masks`` so the zeros can be re-applied after every
    optimizer step ("model sparsity ... for faster training").
    """

    _ATTR = "_lbcs_sparsity_masks"

    def __init__(self, amount: float = 0.5, scope: str = "all", include_bias: bool = False):
        self.amount = float(min(max(amount, 0.0), 1.0))
        self.scope = str(scope)
        self.include_bias = bool(include_bias)

    # -- parameter selection ------------------------------------------------ #
    def _is_prunable(self, name: str, param: Any) -> bool:
        if not (_TORCH_AVAILABLE and isinstance(param, torch.Tensor)):  # pragma: no cover
            return False
        if param.dim() < 2:
            return False
        if "num_batches_tracked" in name:
            return False
        if not self.include_bias and name.endswith("bias"):
            return False
        if self.scope == "linear":
            return param.dim() == 2
        if self.scope == "conv":
            return param.dim() >= 3
        return True

    def named_prunable(self, model: Any) -> List[Tuple[str, Any]]:
        if not _TORCH_AVAILABLE:  # pragma: no cover
            raise RuntimeError("PyTorch is required for magnitude pruning")
        out: List[Tuple[str, Any]] = []
        for name, param in model.named_parameters():
            if not self._is_prunable(name, param):
                continue
            if self.scope == "hidden" and (
                "classifier" in name or name.startswith("fc") or name.startswith("linear")
            ):
                continue
            out.append((name, param))
        return out

    # -- pruning ------------------------------------------------------------ #
    def threshold(self, model: Any) -> float:
        """Global magnitude threshold so that ``amount`` of weights fall below it."""
        if not _TORCH_AVAILABLE:  # pragma: no cover
            raise RuntimeError("PyTorch is required for magnitude pruning")
        tensors = [
            p.detach().abs().reshape(-1).float().cpu() for _, p in self.named_prunable(model)
        ]
        if not tensors:
            return 0.0
        flat = torch.cat(tensors).numpy()
        if flat.size == 0:
            return 0.0
        keep = max(1, int(round((1.0 - self.amount) * flat.size)))
        if keep >= flat.size:
            return 0.0
        # threshold = (keep+1)-th smallest magnitude, i.e. keep weights stay > thr
        partitioned = np.partition(flat, flat.size - keep)
        return float(partitioned[flat.size - keep])

    def apply(self, model: Any) -> Dict[str, Any]:
        """Prune ``model`` in place and remember the masks used."""
        if not _TORCH_AVAILABLE:  # pragma: no cover
            raise RuntimeError("PyTorch is required for magnitude pruning")
        thr = self.threshold(model)
        masks: Dict[str, Any] = {}
        n_zero = 0
        n_total = 0
        with torch.no_grad():
            for name, param in self.named_prunable(model):
                mask = param.detach().abs() > thr
                if not bool(mask.any()):
                    # never delete an entire parameter tensor
                    mask = torch.ones_like(mask, dtype=torch.bool)
                masks[name] = mask.cpu()
                param.mul_(mask.to(param.dtype))
                n_zero += int((~mask).sum().item())
                n_total += int(mask.numel())
        setattr(
            model, self._ATTR, {"threshold": thr, "masks": masks, "amount": self.amount}
        )
        return {
            "threshold": thr,
            "zeroed": n_zero,
            "prunable": n_total,
            "realized_sparsity": (n_zero / n_total) if n_total else 0.0,
        }

    def enforce(self, model: Any) -> int:
        """Re-apply the stored pruning masks (call after each optimizer step)."""
        state = getattr(model, self._ATTR, None)
        if not state or not _TORCH_AVAILABLE:
            return 0
        masks: Dict[str, Any] = state.get("masks", {})
        if not masks:
            return 0
        n_zero = 0
        with torch.no_grad():
            for name, param in model.named_parameters():
                mask = masks.get(name)
                if mask is None:
                    continue
                mask = mask.to(param.device)
                z = int((~mask).sum().item())
                if z:
                    param.mul_(mask.to(param.dtype))
                    n_zero += z
        return n_zero

    def masks(self, model: Any) -> Dict[str, Any]:
        return dict(getattr(model, self._ATTR, {}).get("masks", {}))


def apply_magnitude_pruning(
    model: Any,
    amount: float = SUGGESTED_SPARSITY_AMOUNT,
    scope: str = "all",
    include_bias: bool = False,
    enforce: bool = True,
) -> Dict[str, Any]:
    """Prune ``model`` in place to ``amount`` sparsity and (optionally) freeze zeros."""
    pruner = MagnitudePruner(amount=amount, scope=scope, include_bias=include_bias)
    info = pruner.apply(model)
    if enforce:
        pruner.enforce(model)
    info.update({"amount": amount, "scope": scope, "sparsity": model_sparsity(model)})
    return info


class SparseModelFactory:
    """Wrap a model factory so that every produced model is pruned (smaller model)."""

    def __init__(
        self,
        factory: Callable[..., Any],
        amount: float = SUGGESTED_SPARSITY_AMOUNT,
        scope: str = "all",
        include_bias: bool = False,
        prune_on_create: bool = True,
    ):
        self.factory = factory
        self.pruner = MagnitudePruner(amount=amount, scope=scope, include_bias=include_bias)
        self.amount = float(amount)
        self.scope = scope
        self.prune_on_create = bool(prune_on_create)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        model = self.factory(*args, **kwargs)
        if self.prune_on_create and self.amount > 0:
            self.pruner.apply(model)
        return model

    def prune(self, model: Any) -> Dict[str, Any]:
        return self.pruner.apply(model)

    def enforce(self, model: Any) -> int:
        return self.pruner.enforce(model)

    def to_dict(self) -> Dict[str, Any]:
        return {"amount": self.amount, "scope": self.scope}


def make_sparse_model_factory(
    factory: Callable[..., Any],
    amount: float = SUGGESTED_SPARSITY_AMOUNT,
    scope: str = "all",
    **kwargs: Any,
) -> SparseModelFactory:
    """Convenience builder for :class:`SparseModelFactory`."""
    return SparseModelFactory(factory, amount=amount, scope=scope, **kwargs)


def parameter_count(model: Any, trainable_only: bool = False) -> int:
    """Number of (trainable) parameters."""
    if hasattr(model, "parameters"):
        params = list(model.parameters())
        if trainable_only:
            params = [p for p in params if getattr(p, "requires_grad", True)]
        return int(sum(int(p.numel()) for p in params))
    return 0  # pragma: no cover


def model_size_mb(model: Any, dtype_bytes: int = 4) -> float:
    """Dense storage footprint of the model's parameters, in MiB."""
    return parameter_count(model) * dtype_bytes / (1024.0 ** 2)


def model_sparsity(model: Any) -> float:
    """Fraction of exactly-zero parameters (``0.0`` for a dense model)."""
    if not (hasattr(model, "parameters")):  # pragma: no cover
        return 0.0
    n_zero = 0
    n_total = 0
    params = list(model.parameters())
    if _TORCH_AVAILABLE:
        with torch.no_grad():
            for p in params:
                if not isinstance(p, torch.Tensor) or p.numel() == 0:
                    continue
                n_zero += int((p == 0).sum().item())
                n_total += int(p.numel())
    else:  # pragma: no cover - numpy fallback
        for p in params:
            
            arr = np.asarray(p)
            n_zero += int(np.count_nonzero(arr == 0))
            n_total += int(arr.size)
    return float(n_zero) / float(n_total) if n_total else 0.0


def sparsity_report(model: Any, amount: float = 0.0) -> Dict[str, Any]:
    """Report on the (sparse) model size, for logging / ablation tables."""
    return {
        "parameters": parameter_count(model),
        "trainable_parameters": parameter_count(model, trainable_only=True),
        "size_mb": model_size_mb(model),
        "sparsity": model_sparsity(model),
        "target_sparsity": float(amount),
    }


# --------------------------------------------------------------------------- #
# (i) Warm start: pretrain with random masks, then finetune with other masks
# --------------------------------------------------------------------------- #


@dataclass
class WarmStartSchedule:
    """Schedule of the "pretrain with random masks, then finetune" trick (§3.2).

    The paper states the procedure but no budgets, so every omitted number is a
    SUGGESTED default.
    """

    pretrain_masks: int = SUGGESTED_PRETRAIN_MASKS
    pretrain_epochs: Optional[int] = None
    finetune_epochs: Optional[int] = None
    reuse_state: bool = True
    pretrain_k: Optional[int] = None
    """Coreset size of the random pretraining masks (``None`` -> the task's ``k``)."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pretrain_masks": self.pretrain_masks,
            "pretrain_epochs": self.pretrain_epochs,
            "finetune_epochs": self.finetune_epochs,
            "reuse_state": self.reuse_state,
            "pretrain_k": self.pretrain_k,
        }


def _train_call(
    train_fn: Callable[..., Any],
    model: Any,
    mask: np.ndarray,
    epochs: Optional[int],
    **kwargs: Any,
) -> Any:
    """Call an injected inner trainer, tolerating an absent ``epochs`` argument.

    ``train_fn`` may be any callable ``train_fn(model, mask, epochs=..., **kw)``
    returning the trained model or an object exposing ``.model``.
    """
    call_kwargs = dict(kwargs)
    if epochs is not None:
        call_kwargs["epochs"] = int(epochs)
    try:
        result = train_fn(model, mask, **call_kwargs)
    except TypeError:
        call_kwargs.pop("epochs", None)
        result = train_fn(model, mask, **call_kwargs)
    trained = getattr(result, "model", None)
    return trained if trained is not None else result


def pretrain_with_random_masks(
    train_fn: Callable[..., Any],
    model: Any,
    n: int,
    k: Optional[int] = None,
    epochs: Optional[int] = None,
    num_masks: int = SUGGESTED_PRETRAIN_MASKS,
    seed: Optional[int] = None,
    return_masks: bool = False,
) -> Any:
    """Trick (i), first half: "first train a model with random masks".

    Trains ``model`` on ``num_masks`` random coresets of size ``k`` (``epochs``
    each) and returns the warm-started model, or ``(model, masks)``.
    """
    n = int(n)
    k = n // 2 if k is None else int(k)
    rng = np.random.default_rng(seed)
    masks: List[np.ndarray] = []
    current = model
    for _ in range(max(1, int(num_masks))):
        mask = init_binary_mask(n, k=k, generator=rng)
        masks.append(mask)
        current = _train_call(train_fn, current, mask, epochs)
    if return_masks:
        return current, masks
    return current


def finetune_with_mask(
    train_fn: Callable[..., Any],
    model: Any,
    mask: Sequence[float],
    epochs: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """Trick (i), second half: "then finetune it with other different masks"."""
    return _train_call(train_fn, model, np.asarray(mask, dtype=np.float32), epochs, **kwargs)


class WarmStarter:
    """Stateful driver of the "pretrain with random masks / finetune" trick.

    It caches the most recent trained ``state_dict`` and reuses it across the
    mask updates of Algorithm 1, so each inner-loop call starts from a good
    initialisation instead of from scratch.
    """

    def __init__(
        self,
        schedule: Optional[WarmStartSchedule] = None,
        max_entries: int = 2,
        to_cpu: bool = True,
        logger: Optional[logging.Logger] = None,
        state_bank: Any = None,
    ):
        self.schedule = schedule or WarmStartSchedule()
        self.logger = logger or LOGGER
        self.to_cpu = bool(to_cpu)
        self.max_entries = int(max_entries)
        self._bank = state_bank
        self._pretrained = False
        self.pretrain_info: Dict[str, Any] = {}
        self.finetune_count = 0

    # -- state bank (delegates to lbcs.bilevel.ModelStateBank when available) - #
    @property
    def bank(self) -> Any:
        if self._bank is None:
            try:  # lazy import avoids a circular import at module load time
                from .bilevel import ModelStateBank

                self._bank = ModelStateBank(
                    max_entries=self.max_entries, to_cpu=self.to_cpu
                )
            except Exception:  # pragma: no cover - numpy fallback bank
                self._bank = _MiniStateBank(max_entries=self.max_entries, to_cpu=self.to_cpu)
        return self._bank

    # -- API ---------------------------------------------------------------- #
    def pretrain(
        self,
        train_fn: Callable[..., Any],
        model: Any,
        n: int,
        k: Optional[int] = None,
        seed: Optional[int] = None,
        force: bool = False,
    ) -> Any:
        """Run the random-mask pretraining once (idempotent unless ``force``)."""
        if self._pretrained and not force:
            return model
        sched = self.schedule
        t0 = time.time()
        model, masks = pretrain_with_random_masks(
            train_fn,
            model,
            n=n,
            k=sched.pretrain_k if sched.pretrain_k is not None else k,
            epochs=sched.pretrain_epochs,
            num_masks=sched.pretrain_masks,
            seed=seed,
            return_masks=True,
        )
        self._pretrained = True
        self.pretrain_info = {
            "num_masks": len(masks),
            "pretrain_epochs": sched.pretrain_epochs,
            "wall_time": time.time() - t0,
            "mask_sizes": [int(num_selected(m)) for m in masks],
        }
        if sched.reuse_state:
            self.save(model, key="pretrain")
        self.logger.debug(
            "warm start: pretrained with %d random mask(s) in %.1fs",
            len(masks),
            self.pretrain_info["wall_time"],
        )
        return model

    def finetune(
        self,
        train_fn: Callable[..., Any],
        model: Any,
        mask: Sequence[float],
        **kwargs: Any,
    ) -> Any:
        """Finetune the warm-started model on the coreset given by ``mask``."""
        if self.schedule.reuse_state:
            self.load(model, key="pretrain")
        model = finetune_with_mask(
            train_fn, model, mask, epochs=self.schedule.finetune_epochs, **kwargs
        )
        self.finetune_count += 1
        return model

    def warm_start(
        self,
        train_fn: Callable[..., Any],
        model: Any,
        mask: Sequence[float],
        n: int,
        k: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Any:
        """Full trick (i): pretrain on random masks, then finetune on ``mask``."""
        model = self.pretrain(train_fn, model, n=n, k=k, seed=seed)
        return self.finetune(train_fn, model, mask)

    # -- checkpointing ------------------------------------------------------ #
    def save(self, model: Any, key: Optional[str] = None) -> None:
        try:
            self.bank.save(model, key=key)
        except Exception:  # pragma: no cover - best effort
            pass

    def load(self, model: Any, key: Optional[str] = None) -> bool:
        try:
            return bool(self.bank.load(model, key=key))
        except Exception:  # pragma: no cover
            return False

    def reset(self) -> None:
        self._pretrained = False
        self.pretrain_info = {}
        self.finetune_count = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schedule": self.schedule.to_dict(),
            "pretrained": self._pretrained,
            "finetune_count": self.finetune_count,
            "pretrain_info": dict(self.pretrain_info),
        }


class _MiniStateBank:
    """Tiny state_dict bank used only when ``lbcs.bilevel.ModelStateBank`` is absent."""

    def __init__(self, max_entries: int = 2, to_cpu: bool = True):
        self.max_entries = max(1, int(max_entries))
        self.to_cpu = bool(to_cpu)
        self._store: Dict[Any, Dict[str, Any]] = {}
        self._order: List[Any] = []
        self._latest: Any = None

    @staticmethod
    def _key(key: Any) -> Any:
        return "latest" if key is None else key

    def save(self, model: Any, key: Any = None) -> Any:
        key = self._key(key)
        raw = model.state_dict() if hasattr(model, "state_dict") else dict(model)
        state: Dict[str, Any] = {}
        for k_, v_ in raw.items():
            if _TORCH_AVAILABLE and hasattr(v_, "detach"):
                state[k_] = v_.detach().cpu().clone() if self.to_cpu else v_.detach().clone()
            else:
                state[k_] = copy.deepcopy(v_)
        self._store[key] = state
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)
        while len(self._order) > self.max_entries:
            self._store.pop(self._order.pop(0), None)
        self._latest = key
        return key

    def load(self, model: Any, key: Any = None) -> bool:
        key = self._key(key) if key is not None else self._latest
        if key is None or key not in self._store:
            return False
        if hasattr(model, "load_state_dict"):
            try:
                model.load_state_dict(self._store[key], strict=False)
            except TypeError:  # pragma: no cover - minimal stubs
                model.load_state_dict(self._store[key])
        else:  # pragma: no cover
            model.update(self._store[key])
        return True

    def keys(self) -> List[Any]:
        return list(self._order)

    def __len__(self) -> int:
        return len(self._store)


def warm_start_train(
    train_fn: Callable[..., Any],
    model: Any,
    mask: Sequence[float],
    n: int,
    k: Optional[int] = None,
    pretrain_masks: int = SUGGESTED_PRETRAIN_MASKS,
    pretrain_epochs: Optional[int] = None,
    finetune_epochs: Optional[int] = None,
    seed: Optional[int] = None,
    starter: Optional[WarmStarter] = None,
) -> Any:
    """Functional wrapper for the complete warm-start trick (i)."""
    starter = starter or WarmStarter(
        WarmStartSchedule(
            pretrain_masks=pretrain_masks,
            pretrain_epochs=pretrain_epochs,
            finetune_epochs=finetune_epochs,
        )
    )
    return starter.warm_start(train_fn, model, mask, n=n, k=k, seed=seed)


# --------------------------------------------------------------------------- #
# Facade
# --------------------------------------------------------------------------- #


@dataclass
class AccelerationConfig:
    """Bundle of the three Section 3.2 acceleration tricks.

    All knobs live in one configuration object so the "suggested defaults" can be
    changed without editing algorithm code.

    Presets
    -------
    ``for_section52()`` : §5.2 inner loop, Adam with learning rate 0.001
        (paper-stated) with the acceleration tricks enabled.
    ``for_figure1()``   : Figure 1 / Appendix C.3 inner loop, SGD with learning
        rate 0.1 and momentum 0.9 for 100 epochs (paper-stated).
    ``disabled()``      : plain Algorithm 1 without any acceleration trick.
    """

    # (i) warm start
    warm_start: bool = True
    pretrain_masks: int = SUGGESTED_PRETRAIN_MASKS
    pretrain_epochs: Optional[int] = None
    finetune_epochs: Optional[int] = None

    # (ii) sparsity / smaller model
    sparsity: float = 0.0
    sparsity_scope: str = "all"
    sparsity_include_bias: bool = False

    # (iii) grouping
    group_size: int = DEFAULT_GROUP_SIZE
    group_seed: Optional[int] = None
    group_shuffle: bool = True

    # misc
    seed: Optional[int] = None
    log_every: int = 0

    # -- presets / construction --------------------------------------------- #
    @classmethod
    def for_section52(cls, **overrides: Any) -> "AccelerationConfig":
        """§5.2 preset (inner loop: Adam, lr 0.001, 100 epochs)."""
        cfg = cls(
            warm_start=True,
            pretrain_masks=SUGGESTED_PRETRAIN_MASKS,
            pretrain_epochs=None,
            finetune_epochs=None,
            sparsity=0.0,
            group_size=DEFAULT_GROUP_SIZE,
        )
        return cfg.with_overrides(**overrides)

    @classmethod
    def for_figure1(cls, **overrides: Any) -> "AccelerationConfig":
        """Figure 1 / Appendix C.3 preset (inner loop: SGD lr 0.1, momentum 0.9, 100 epochs)."""
        cfg = cls(
            warm_start=True,
            pretrain_masks=SUGGESTED_PRETRAIN_MASKS,
            pretrain_epochs=100,
            finetune_epochs=100,
            sparsity=0.0,
            group_size=DEFAULT_GROUP_SIZE,
        )
        return cfg.with_overrides(**overrides)

    @classmethod
    def disabled(cls) -> "AccelerationConfig":
        """No acceleration (plain Algorithm 1) - the natural ablation."""
        return cls(warm_start=False, sparsity=0.0, group_size=DEFAULT_GROUP_SIZE)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "AccelerationConfig":
        if not data:
            return cls()
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in dict(data).items() if k in known})

    def with_overrides(self, **overrides: Any) -> "AccelerationConfig":
        known = set(self.__dataclass_fields__)  # type: ignore[attr-defined]
        data = {**self.to_dict(), **{k: v for k, v in overrides.items() if k in known}}
        return AccelerationConfig.from_dict(data)

    # -- helpers ------------------------------------------------------------ #
    def enabled(self) -> bool:
        return bool(self.warm_start or self.sparsity > 0 or self.group_size > 1)

    def sparsity_config(self) -> SparsityConfig:
        return SparsityConfig(
            amount=self.sparsity,
            scope=self.sparsity_scope,
            include_bias=self.sparsity_include_bias,
        )

    def warm_start_schedule(self) -> WarmStartSchedule:
        return WarmStartSchedule(
            pretrain_masks=self.pretrain_masks,
            pretrain_epochs=self.pretrain_epochs,
            finetune_epochs=self.finetune_epochs,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "warm_start": self.warm_start,
            "pretrain_masks": self.pretrain_masks,
            "pretrain_epochs": self.pretrain_epochs,
            "finetune_epochs": self.finetune_epochs,
            "sparsity": self.sparsity,
            "sparsity_scope": self.sparsity_scope,
            "sparsity_include_bias": self.sparsity_include_bias,
            "group_size": self.group_size,
            "group_seed": self.group_seed,
            "group_shuffle": self.group_shuffle,
            "seed": self.seed,
            "log_every": self.log_every,
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "tricks": {
                "warm_start": bool(self.warm_start),
                "sparsity": self.sparsity > 0,
                "grouping": self.group_size > 1,
            },
            "config": self.to_dict(),
        }


class AccelerationManager:
    """Wire the three acceleration tricks into the Algorithm 1 pipeline.

    Usage (typical experiment driver)::

        acc = AccelerationManager(AccelerationConfig(group_size=8, sparsity=0.0))
        space = acc.search_space(n)                     # trick (iii): narrow M
        factory = acc.wrap_model_factory(make_model)    # trick (ii)
        model = acc.prepare_sparse(factory())           # trick (ii)
        theta = acc.train(train_fn, model, mask, n=n, k=k)  # trick (i)

    The manager never evaluates objectives itself: it only prepares the search
    space, the (possibly sparse) model, and the warm-started inner-loop start.
    """

    def __init__(
        self,
        config: Optional[AccelerationConfig] = None,
        n: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
        starter: Optional[WarmStarter] = None,
    ):
        self.config = config or AccelerationConfig()
        self.n = None if n is None else int(n)
        self.logger = logger or LOGGER
        self._space: Optional[GroupSearchSpace] = None
        self._starter: Optional[WarmStarter] = starter
        self.counters: Dict[str, int] = {
            "pretrains": 0,
            "finetunes": 0,
            "prune_applications": 0,
            "enforce_calls": 0,
        }

    # -- trick (iii): narrowed search space --------------------------------- #
    def search_space(
        self, n: Optional[int] = None, group_size: Optional[int] = None
    ) -> GroupSearchSpace:
        """Narrowed outer-loop search space (identity when ``group_size == 1``)."""
        if n is not None:
            self.n = int(n)
        if self._space is None or group_size is not None:
            if self.n is None:
                raise ValueError("AccelerationManager needs `n` to build the search space")
            self._space = build_search_space(
                self.n,
                group_size=self.config.group_size if group_size is None else group_size,
                seed=self.config.group_seed,
                shuffle=self.config.group_shuffle,
            )
        return self._space

    @property
    def grouping(self) -> Grouping:
        return self.search_space().grouping

    def dimension(
        self, n: Optional[int] = None, group_size: Optional[int] = None
    ) -> int:
        """Dimension of the outer-loop decision problem (Algorithm 2 space)."""
        return int(self.search_space(n, group_size).dimension)

    # -- trick (ii): sparsity ---------------------------------------------- #
    def wrap_model_factory(
        self, factory: Callable[..., Any], amount: Optional[float] = None
    ) -> Callable[..., Any]:
        """Return a factory whose models are pruned to ``sparsity``."""
        amt = self.config.sparsity if amount is None else float(amount)
        if amt <= 0:
            return factory
        return make_sparse_model_factory(
            factory, amount=amt, scope=self.config.sparsity_scope
        )

    def prepare_sparse(self, model: Any) -> Dict[str, Any]:
        """Prune an existing model in place (trick ii) and report its size."""
        if self.config.sparsity <= 0:
            return sparsity_report(model, amount=0.0)
        info = apply_magnitude_pruning(
            model,
            amount=self.config.sparsity,
            scope=self.config.sparsity_scope,
            include_bias=self.config.sparsity_include_bias,
        )
        self.counters["prune_applications"] += 1
        self.logger.debug("sparsity applied: %s", info)
        return info

    def enforce_sparsity(self, model: Any) -> int:
        """Freeze the pruned weights after an optimizer step."""
        if self.config.sparsity <= 0:
            return 0
        pruner = MagnitudePruner(
            amount=self.config.sparsity, scope=self.config.sparsity_scope
        )
        self.counters["enforce_calls"] += 1
        return pruner.enforce(model)

    # -- trick (i): warm start --------------------------------------------- #
    @property
    def starter(self) -> WarmStarter:
        if self._starter is None:
            self._starter = WarmStarter(
                schedule=self.config.warm_start_schedule(), logger=self.logger
            )
        return self._starter

    def pretrain(
        self,
        train_fn: Callable[..., Any],
        model: Any,
        n: Optional[int] = None,
        k: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Any:
        """Trick (i): train the model with random masks (once per run)."""
        if not self.config.warm_start:
            return model
        n_ = self.n if n is None else int(n)
        if n_ is None:
            raise ValueError("AccelerationManager needs `n` to pretrain")
        model = self.starter.pretrain(
            train_fn, model, n=n_, k=k, seed=self.config.seed if seed is None else seed
        )
        self.counters["pretrains"] += 1
        if self.config.sparsity > 0:
            self.prepare_sparse(model)
        return model

    def train(
        self,
        train_fn: Callable[..., Any],
        model: Any,
        mask: Sequence[float],
        n: Optional[int] = None,
        k: Optional[int] = None,
        seed: Optional[int] = None,
        pretrain: bool = True,
    ) -> Any:
        """Accelerated inner loop of Algorithm 1 (Step 3).

        With ``warm_start`` enabled this performs "pretrain with random masks,
        then finetune with the current mask"; otherwise it is a plain inner-loop
        training call.
        """
        n_ = self.n if n is None else int(n)
        if self.config.warm_start and pretrain:
            model = self.pretrain(train_fn, model, n=n_, k=k, seed=seed)
        if self.config.warm_start:
            model = self.starter.finetune(train_fn, model, mask)
            self.counters["finetunes"] += 1
        else:
            model = finetune_with_mask(
                train_fn, model, mask, epochs=self.config.finetune_epochs
            )
        if self.config.sparsity > 0:
            self.enforce_sparsity(model)
        return model

    # -- reporting ---------------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "config": self.config.to_dict(),
            "counters": dict(self.counters),
        }
        if self._space is not None:
            data["search_space"] = self._space.to_dict()
        if self._starter is not None:
            data["warm_start"] = self._starter.to_dict()
        return data

    def summary(self) -> Dict[str, Any]:
        return self.to_dict()


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #


def _selftest(verbose: bool = True) -> Dict[str, Any]:
    """Offline checks for the three acceleration tricks (torch optional)."""
    out: Dict[str, Any] = {}

    # (iii) grouping narrows the search space and expands back consistently
    n = 1000
    space = build_search_space(n, group_size=8, seed=0)
    out["num_groups"] = space.num_groups
    assert space.num_groups == search_space_dimension(n, 8) == 125
    assert abs(space.compression - 8.0) < 1e-9
    gmask = space.init_group_mask(k=400, seed=1, continuous=True)
    assert gmask.shape[0] == space.num_groups, gmask.shape
    ex = space.expand(gmask)
    assert ex.shape[0] == n, ex.shape
    assert np.all((ex >= 0) & (ex <= 1))
    ident = build_search_space(n, group_size=1)
    assert ident.num_groups == n and not ident.is_grouped
    assert np.allclose(ident.expand(np.ones(n)), np.ones(n))
    red = space.reduce(init_binary_mask(n, k=200, seed=2), mode="mean")
    assert red.shape[0] == space.num_groups and red.min() >= 0 and red.max() <= 1
    assert abs(acceleration_ratio(n, 8) - 8.0) < 1e-9
    assert abs(acceleration_ratio(n, 1) - 1.0) < 1e-9
    out["search_space_ok"] = True

    # (i) warm start with an injected toy trainer (works without torch)
    calls: List[Tuple[int, int]] = []

    class _ToyModel:
        def __init__(self) -> None:
            self.steps = 0

        def state_dict(self) -> Dict[str, Any]:
            return {"steps": self.steps}

        def load_state_dict(self, state: Dict[str, Any], strict: bool = True) -> None:
            self.steps = int(state.get("steps", 0))

    def toy_train(model: Any, mask: Any, epochs: int = 1) -> Any:
        calls.append((int(np.count_nonzero(mask)), int(epochs)))
        model.steps += int(epochs)
        return model

    starter = WarmStarter(
        WarmStartSchedule(pretrain_masks=1, pretrain_epochs=3, finetune_epochs=5)
    )
    model = _ToyModel()
    model = starter.warm_start(
        toy_train, model, init_binary_mask(n, k=100, seed=3), n=n, k=100, seed=4
    )
    assert len(calls) == 2, calls
    assert calls[0] == (100, 3) and calls[1] == (100, 5), calls
    out["warm_start_calls"] = calls
    before = len(calls)
    starter.pretrain(toy_train, model, n=n, k=100)
    assert len(calls) == before, "pretraining must only happen once"
    out["warm_start_ok"] = True

    # (ii) magnitude pruning / smaller model (needs torch)
    if _TORCH_AVAILABLE:
        net = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))
        dense_params = parameter_count(net)
        info = apply_magnitude_pruning(net, amount=0.5, scope="linear")
        sp = model_sparsity(net)
        out["prune_info"] = {
            "dense_params": dense_params,
            "sparsity": round(sp, 4),
            "realized": round(float(info["realized_sparsity"]), 4),
        }
        assert sp > 0.4, sp
        opt = torch.optim.SGD(net.parameters(), lr=0.1)
        loss = net(torch.randn(8, 16)).pow(2).mean()
        loss.backward()
        opt.step()
        MagnitudePruner(amount=0.5, scope="linear").enforce(net)
        assert model_sparsity(net) > 0.4
        factory = SparseModelFactory(
            lambda: nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2)), amount=0.5
        )
        m2 = factory()
        assert model_sparsity(m2) > 0.3
        assert parameter_count(m2) == 4 * 8 + 8 + 8 * 2 + 2
        out["sparsity_ok"] = True
    else:  # pragma: no cover
        out["sparsity_ok"] = "skipped (torch unavailable)"

    # facade round-trip
    cfg = AccelerationConfig.for_section52(group_size=8, sparsity=0.5)
    mgr = AccelerationManager(cfg, n=n)
    assert mgr.dimension() == 125
    assert mgr.config.warm_start is True
    assert AccelerationConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()
    assert AccelerationConfig.disabled().enabled() is False
    out["manager_summary"] = mgr.summary()
    out["facade_ok"] = True

    if verbose:
        for key, value in out.items():
            print(f"[acceleration] {key}: {value}")
    return out


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    _selftest(verbose=True)
