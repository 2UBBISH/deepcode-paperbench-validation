"""Table 8 (Appendix E.3): optimized coreset sizes under imperfect supervision.

Paper reference
---------------
Appendix E.3 *Optimized Coreset Sizes with Imperfect Supervision*:

    "In the main paper (5.3), we have shown the strength of the proposed method in
     coreset selection under imperfect supervision. Here we supplement the optimized
     coreset sizes by our method in these cases, which are provided in Table 8."

Table 8 (mean +- std of optimized coreset sizes by our method):

    Imperfect supervision          k=1000        k=2000        k=3000        k=4000
    With 30% corrupted labels   951.2 +- 4.9  1866.1 +- 8.3  2713.7 +- 10.8  3675.6 +- 17.0
    With 50% corrupted labels   934.5 +- 5.6  1856.5 +- 9.1  2708.8 +- 11.2  3668.4 +- 14.6
    With class-imbalanced data  988.4 +- 6.7  1893.8 +- 10.0 2762.7 +- 14.2  3757.4 +- 17.8

Section 5.3 protocol (reused verbatim from Section 5.2):

    * F-MNIST, LeNet proxy **and** LeNet target (Addendum "Useful details for
      Section 5.3": "the same proxy and target models are used as in section 5.2 for
      F-MNIST, i.e. a LeNet for both the proxy and target model").
    * 30% symmetric label noise (labels of 30% of the *training* data flipped), and a
      higher 50% noise level reported in Appendix E.2.
    * Exponential-type class imbalance with imbalance ratio 0.01 (Cao et al., 2019;
      code adapted from ``imbalanced-semi-self/dataset/imbalance_cifar.py``); the
      imbalance is "just injected into the training set, which does not include the
      test set".
    * k in {1000, 2000, 3000, 4000}; LBCS runs with epsilon = 0.2 and T = 500
      (Section 5.2 settings), 10 repeats.

This driver reports the **achieved** coreset size ``f2(m) = ||m||_0`` of the LBCS
solution under each imperfect-supervision condition and compares it to the paper's
Table 8 anchors.  It complements ``experiments/figure2_robustness.py`` (which reports
the accuracy side of Figure 2 / Table 8) by focusing on sizes.

Out of scope: ImageNet-1k (5.4), continual learning (Appendix E.5), streaming
(Appendix E.6).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # optional soft dependencies -------------------------------------------------
    import torch  # noqa: F401
    import torch.nn as nn  # noqa: F401

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch is optional
    torch = None  # type: ignore
    nn = None  # type: ignore
    _TORCH_AVAILABLE = False

try:
    import yaml  # type: ignore

    _YAML_AVAILABLE = True
except Exception:  # pragma: no cover
    yaml = None  # type: ignore
    _YAML_AVAILABLE = False

try:
    import matplotlib  # type: ignore

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore

    _MPL_AVAILABLE = True
except Exception:  # pragma: no cover
    plt = None  # type: ignore
    _MPL_AVAILABLE = False


LOGGER = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Paper constants
# --------------------------------------------------------------------------------------
PAPER_DATASET = "F-MNIST"
PAPER_KS: Tuple[int, ...] = (1000, 2000, 3000, 4000)
PAPER_EPSILON = 0.2
PAPER_T = 500
PAPER_REPEATS = 10
PAPER_NOISE_RATE_30 = 0.30
PAPER_NOISE_RATE_50 = 0.50
PAPER_IMBALANCE_RATIO = 0.01
PAPER_IMBALANCE_TYPE = "exp"

PAPER_INNER_OPTIMIZER = "adam"
PAPER_INNER_LR = 0.001
PAPER_INNER_EPOCHS = 100
PAPER_TARGET_OPTIMIZER = "adam"
PAPER_TARGET_LR = 0.001
PAPER_TARGET_EPOCHS = 100

LBCS_LABEL = "LBCS (ours)"
SIZE_LABEL = "Coreset size (ours)"

#: Conditions of Figure 2 / Table 8, in paper order.
CONDITIONS: Tuple[str, ...] = ("noise30", "noise50", "imbalance")

CONDITION_LABELS: Dict[str, str] = {
    "noise30": "With 30% corrupted labels",
    "noise50": "With 50% corrupted labels",
    "imbalance": "With class-imbalanced data",
}

#: Table 8 of the paper: condition -> k -> (mean, std) of optimized coreset size.
PAPER_TABLE8: Dict[str, Dict[int, Tuple[float, float]]] = {
    "noise30": {
        1000: (951.2, 4.9),
        2000: (1866.1, 8.3),
        3000: (2713.7, 10.8),
        4000: (3675.6, 17.0),
    },
    "noise50": {
        1000: (934.5, 5.6),
        2000: (1856.5, 9.1),
        3000: (2708.8, 11.2),
        4000: (3668.4, 14.6),
    },
    "imbalance": {
        1000: (988.4, 6.7),
        2000: (1893.8, 10.0),
        3000: (2762.7, 14.2),
        4000: (3757.4, 17.8),
    },
}

#: Paper reference for the accuracy side (Table 9 of the paper t-sweep at k=1000 is
#: unrelated; the figure-2 accuracy anchors act as a sanity reference only).
PAPER_ACCURACY_ANCHOR: Dict[str, Dict[int, float]] = {
    "noise30": {1000: 79.7},
    "imbalance": {1000: 79.4},
}

#: Suggested (non paper-stated) defaults -------------------------------------------
SUGGESTED_BATCH_SIZE = 128
SUGGESTED_EVAL_BATCH_SIZE = 256
SUGGESTED_WEIGHT_DECAY = 0.0
SUGGESTED_DELTA_INIT = 0.1
SUGGESTED_DELTA_LOWER = 1e-3
SUGGESTED_NUM_WORKERS = 0
SUGGESTED_SIZE_TOLERANCE = 25.0
DEFAULT_OUTPUT_DIR = os.path.join("results", "table8")


# --------------------------------------------------------------------------------------
# Small shared utilities (kept local so the driver is importable without torch/data)
# --------------------------------------------------------------------------------------
def set_seed(seed: Optional[int]) -> None:
    """Seed NumPy (and PyTorch when available) deterministically."""
    if seed is None:
        return
    import random

    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    if _TORCH_AVAILABLE:
        try:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
        except Exception:  # pragma: no cover
            pass


def resolve_seed(seed: Optional[int], repeat: int = 0, base: int = 0) -> int:
    """Deterministic per-repeat seed for the 10-repeat protocol."""
    s = base if seed is None else int(seed)
    if repeat <= 0:
        return s
    return int((s + repeat * 7919) % (2**31 - 1))


def binarize_mask(mask: Any) -> np.ndarray:
    """Project a mask to ``{0,1}`` following Appendix A (clamp to [-1,1], then
    ``[-1,0) -> 0``, ``[0,1] -> 1``)."""
    if mask is None:
        return np.zeros(0, dtype=np.float32)
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    arr = np.asarray(mask, dtype=np.float64).reshape(-1)
    arr = np.clip(arr, -1.0, 1.0)
    return (arr >= 0.0).astype(np.float32)


def mask_indices(mask: Any) -> np.ndarray:
    """Indices of selected examples (``m_i >= 0`` after projection)."""
    b = binarize_mask(mask)
    return np.flatnonzero(b > 0.5)


def mask_size(mask: Any) -> int:
    """``f2(m) = ||m||_0`` computed on the discretized mask."""
    return int(mask_indices(mask).size)


def canonical_condition(name: Optional[str]) -> str:
    """Normalize a condition name to ``noise30`` / ``noise50`` / ``imbalance``."""
    if not name:
        return CONDITIONS[0]
    key = str(name).strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    key = key.replace("labelnoise", "noise").replace("corrupted", "noise")
    key = key.replace("imbalanced", "imbalance").replace("noise", "noise")
    if key in ("30", "noise30", "0.3", "noise0.3", "30%", "noise30%"):
        return "noise30"
    if key in ("50", "noise50", "0.5", "noise0.5", "50%", "noise50%"):
        return "noise50"
    if key in ("imb", "imbalance", "imb0.01", "ratio0.01", "classimbalance"):
        return "imbalance"
    for cond in CONDITIONS:
        if cond in key:
            return cond
    raise KeyError(f"unknown condition {name!r}; expected one of {CONDITIONS}")


def canonical_dataset(name: Optional[str]) -> str:
    """Normalize dataset names to the paper spelling."""
    if not name:
        return PAPER_DATASET
    key = str(name).strip().upper().replace("_", "-")
    if key in ("F-MNIST", "FMNIST", "FASHION-MNIST", "FASHIONMNIST"):
        return "F-MNIST"
    if key in ("MNIST-S", "MNISTS"):
        return "MNIST-S"
    if key in ("MNIST",):
        return "MNIST"
    if key in ("SVHN",):
        return "SVHN"
    if key in ("CIFAR-10", "CIFAR10"):
        return "CIFAR-10"
    return str(name)


def mean_std(values: Sequence[float], ddof: int = 1) -> Tuple[float, float]:
    """Mean and sample standard deviation ignoring non-finite values."""
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 0.0
    if arr.size == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=ddof))


def format_mean_std(mean: float, std: float, decimals: int = 1) -> str:
    """Format ``mean +- std`` like the paper tables."""
    return f"{mean:.{decimals}f} +- {std:.{decimals}f}"


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
@dataclass
class Table8Config:
    """Protocol for the Table 8 (Appendix E.3) optimized-size experiment."""

    dataset: str = PAPER_DATASET
    conditions: Sequence[str] = field(default_factory=lambda: tuple(CONDITIONS))
    ks: Sequence[int] = field(default_factory=lambda: tuple(PAPER_KS))
    epsilon: float = PAPER_EPSILON
    T: int = PAPER_T
    repeats: int = PAPER_REPEATS

    noise_rate_30: float = PAPER_NOISE_RATE_30
    noise_rate_50: float = PAPER_NOISE_RATE_50
    imbalance_ratio: float = PAPER_IMBALANCE_RATIO
    imbalance_type: str = PAPER_IMBALANCE_TYPE

    inner_optimizer: str = PAPER_INNER_OPTIMIZER
    inner_lr: float = PAPER_INNER_LR
    inner_epochs: int = PAPER_INNER_EPOCHS
    target_optimizer: str = PAPER_TARGET_OPTIMIZER
    target_lr: float = PAPER_TARGET_LR
    target_epochs: int = PAPER_TARGET_EPOCHS

    weight_decay: float = SUGGESTED_WEIGHT_DECAY
    batch_size: int = SUGGESTED_BATCH_SIZE
    eval_batch_size: int = SUGGESTED_EVAL_BATCH_SIZE
    delta_init: float = SUGGESTED_DELTA_INIT
    delta_lower: float = SUGGESTED_DELTA_LOWER
    warm_start: bool = True
    group_size: int = 1
    num_workers: int = SUGGESTED_NUM_WORKERS

    num_classes: Optional[int] = None
    device: Optional[str] = None
    seed: int = 0
    log_every: int = 0
    output_dir: str = DEFAULT_OUTPUT_DIR
    data_root: Optional[str] = None
    save_artifacts: bool = True
    plot: bool = True
    f1_eval_split: str = "test"
    size_tolerance: float = SUGGESTED_SIZE_TOLERANCE
    verbose: bool = False

    # -- constructors ------------------------------------------------------------------
    @classmethod
    def paper(cls, **overrides: Any) -> "Table8Config":
        """Paper protocol: F-MNIST, k in {1000..4000}, epsilon=0.2, T=500, 10 repeats."""
        cfg = cls()
        return cfg.with_overrides(**overrides) if overrides else cfg

    @classmethod
    def smoke(cls, **overrides: Any) -> "Table8Config":
        """Cheap CPU sanity-check variant (not a paper setting)."""
        cfg = cls(
            conditions=("noise30",),
            ks=(200,),
            repeats=1,
            T=20,
            inner_epochs=1,
            target_epochs=1,
            batch_size=64,
            eval_batch_size=128,
            save_artifacts=False,
            plot=False,
            verbose=False,
        )
        return cfg.with_overrides(**overrides) if overrides else cfg

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Table8Config":
        """Build from a (possibly nested) config dict; accepts a ``table8`` block."""
        if not data:
            return cls()
        block = data.get("table8", data) if isinstance(data, dict) else data
        if not isinstance(block, dict):
            return cls()
        alias = {
            "eps": "epsilon",
            "t": "T",
            "num_repeats": "repeats",
            "noise_rate_30pct": "noise_rate_30",
            "noise_rate_50pct": "noise_rate_50",
            "imb_ratio": "imbalance_ratio",
            "eval_batch_size": "eval_batch_size",
            "output": "output_dir",
        }
        kwargs: Dict[str, Any] = {}
        for key, value in block.items():
            k = alias.get(str(key).lower(), str(key))
            if k in cls.__dataclass_fields__:
                kwargs[k] = value
        for key in ("ks", "conditions"):
            if key in kwargs and kwargs[key] is not None:
                kwargs[key] = tuple(kwargs[key])
        return cls(**kwargs)

    def with_overrides(self, **overrides: Any) -> "Table8Config":
        """Return a copy with overrides applied, ignoring ``None`` values."""
        clean = {k: v for k, v in overrides.items() if v is not None and k in self.__dataclass_fields__}
        return replace(self, **clean)

    def to_dict(self) -> Dict[str, Any]:
        out = {k: getattr(self, k) for k in self.__dataclass_fields__}
        for key in ("ks", "conditions"):
            out[key] = list(out[key])
        return out


# --------------------------------------------------------------------------------------
# Result cell
# --------------------------------------------------------------------------------------
@dataclass
class Table8Cell:
    """Aggregated optimized coreset size for one ``(condition, k)`` cell."""

    condition: str
    k: int
    size_mean: float = 0.0
    size_std: float = 0.0
    f1_mean: float = 0.0
    f1_std: float = 0.0
    accuracy_mean: float = 0.0
    accuracy_std: float = 0.0
    repeats: int = 0
    failures: int = 0
    raw: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def cell(self) -> Tuple[str, int]:
        return (self.condition, int(self.k))

    @property
    def label(self) -> str:
        return CONDITION_LABELS.get(self.condition, self.condition)

    def paper_reference(self) -> Optional[Tuple[float, float]]:
        return PAPER_TABLE8.get(self.condition, {}).get(int(self.k))

    def deviation(self) -> Optional[float]:
        ref = self.paper_reference()
        if ref is None:
            return None
        return float(self.size_mean - ref[0])

    def to_dict(self) -> Dict[str, Any]:
        ref = self.paper_reference()
        return {
            "condition": self.condition,
            "k": int(self.k),
            "size_mean": self.size_mean,
            "size_std": self.size_std,
            "size_str": format_mean_std(self.size_mean, self.size_std),
            "f1_mean": self.f1_mean,
            "f1_std": self.f1_std,
            "accuracy_mean": self.accuracy_mean,
            "accuracy_std": self.accuracy_std,
            "repeats": self.repeats,
            "failures": self.failures,
            "paper_size_mean": None if ref is None else ref[0],
            "paper_size_std": None if ref is None else ref[1],
            "size_deviation": self.deviation(),
        }


# --------------------------------------------------------------------------------------
# Context / data plumbing (soft dependencies, graceful degradation)
# --------------------------------------------------------------------------------------
def build_context(
    condition: str,
    dataset: Optional[str] = None,
    config: Optional[Table8Config] = None,
    repeat: int = 0,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Build corrupted train loaders (train split only) + clean test loader.

    Mirrors the Section 5.3 description: the imperfect supervision is injected into the
    *training* set only (Addendum: "the imbalance is just injected into the training
    set, which does not include the test set").
    """
    from lbcs_repro.data.datasets import get_dataset, get_targets, make_loader, num_classes, subset_dataset
    from lbcs_repro.data.robustness import (
        EXPONENTIAL_IMBALANCE_RATIO,
        get_imbalanced_loaders,
        get_noisy_loaders,
    )

    config = config or Table8Config()
    dataset = canonical_dataset(dataset or config.dataset)
    condition = canonical_condition(condition)
    seed = resolve_seed(config.seed, repeat)

    device = device or config.device
    if device is None and _TORCH_AVAILABLE and torch.cuda.is_available():
        device = "cuda"

    extra_seed = seed
    if condition in ("noise30", "noise50"):
        rate = config.noise_rate_30 if condition == "noise30" else config.noise_rate_50
        train_loader, test_loader = get_noisy_loaders(
            name=dataset,
            noise_rate=rate,
            batch_size=config.batch_size,
            root=config.data_root,
            num_workers=config.num_workers,
            seed=extra_seed,
            return_index=True,
        )
        noise_rate = rate
        imbalance_stats = None
    else:
        train_loader, test_loader = get_imbalanced_loaders(
            name=dataset,
            ratio=config.imbalance_ratio or EXPONENTIAL_IMBALANCE_RATIO,
            batch_size=config.batch_size,
            root=config.data_root,
            num_workers=config.num_workers,
            seed=extra_seed,
            imb_type=config.imbalance_type,
            return_index=True,
        )
        noise_rate = None
        imbalance_stats = None
        try:  # optional bookkeeping
            from lbcs_repro.data.robustness import imbalance_ratios

            train_ds = get_dataset(dataset, train=True, root=config.data_root)
            imbalance_stats = imbalance_ratios(train_ds, num_classes=num_classes(dataset))
        except Exception:  # pragma: no cover - diagnostics only
            imbalance_stats = None

    # Evaluation loaders always use the clean splits.
    eval_dataset = get_dataset(dataset, train=True, root=config.data_root)
    test_dataset = get_dataset(dataset, train=False, root=config.data_root)
    eval_loader = make_loader(
        eval_dataset if config.f1_eval_split != "test" else test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        return_index=False,
    )

    selection_dataset = eval_dataset  # clean pool used for scoring baselines
    targets = get_targets(selection_dataset)

    inner_factory, target_factory = _model_factories(dataset, config)

    context: Dict[str, Any] = {
        "condition": condition,
        "dataset": dataset,
        "config": config,
        "repeat": repeat,
        "seed": seed,
        "device": device,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "eval_loader": eval_loader,
        "selection_dataset": selection_dataset,
        "eval_dataset": eval_dataset,
        "targets": targets,
        "n": int(len(targets)),
        "num_classes": num_classes(dataset),
        "inner_factory": inner_factory,
        "target_factory": target_factory,
        "noise_rate": noise_rate,
        "imbalance_stats": imbalance_stats,
    }
    return context


def _model_factories(dataset: str, config: Table8Config) -> Tuple[Callable[..., Any], Callable[..., Any]]:
    """Resolve (inner/proxy, target) model factories; Addendum 5.3: LeNet for both."""
    inner_factory: Optional[Callable[..., Any]] = None
    target_factory: Optional[Callable[..., Any]] = None
    try:
        from lbcs_repro.models import default_model_for, model_factory

        try:
            inner_factory = model_factory(default_model_for(dataset, "inner"))
        except Exception:  # pragma: no cover
            inner_factory = None
        try:
            target_factory = model_factory(default_model_for(dataset, "target"))
        except Exception:  # pragma: no cover
            target_factory = None
    except Exception:  # pragma: no cover - registry unavailable
        inner_factory = None
        target_factory = None

    if inner_factory is None or target_factory is None:
        try:
            from lbcs_repro.models.lenet import lenet_factory

            inner_factory = inner_factory or lenet_factory
            target_factory = target_factory or lenet_factory
        except Exception:  # pragma: no cover
            pass
    return inner_factory, target_factory  # type: ignore[return-value]


def coreset_loader(
    context: Dict[str, Any],
    mask: Any,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> Any:
    """DataLoader restricted to the examples selected by ``mask``."""
    from lbcs_repro.data.datasets import make_loader, subset_dataset

    config: Table8Config = context.get("config") or Table8Config()
    dataset = context.get("selection_dataset") or context.get("eval_dataset")
    indices = mask_indices(mask)
    subset = subset_dataset(dataset, indices)
    return make_loader(
        subset,
        batch_size=batch_size or config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers if num_workers is None else num_workers,
        seed=seed,
    )


# --------------------------------------------------------------------------------------
# LBCS selection + target training (soft dependency on the core stack)
# --------------------------------------------------------------------------------------
def select_lbcs_mask(
    context: Dict[str, Any],
    k: int,
    config: Optional[Table8Config] = None,
    seed: Optional[int] = None,
    lbcs: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Run Algorithm 1 (LBCS) on the corrupted selection pool for a fixed size ``k``.

    Returns a dict with the continuous solution, its binarized mask, the achieved size
    ``f2(m)`` and the full-data objective ``f1(m)``.
    """
    config = config or context.get("config") or Table8Config()
    logger = logger or LOGGER
    seed = resolve_seed(config.seed, context.get("repeat", 0)) if seed is None else seed

    if lbcs is None:
        from lbcs_repro.lbcs.bilevel import LBCS, LBCSConfig, InnerTrainConfig

        inner_cfg = InnerTrainConfig(
            optimizer=config.inner_optimizer,
            lr=config.inner_lr,
            epochs=config.inner_epochs,
            batch_size=config.batch_size,
            weight_decay=config.weight_decay,
            device=context.get("device"),
            zero_init_residual=True,
        ) if hasattr(InnerTrainConfig, "zero_init_residual") else InnerTrainConfig(
            optimizer=config.inner_optimizer,
            lr=config.inner_lr,
            epochs=config.inner_epochs,
            batch_size=config.batch_size,
            weight_decay=config.weight_decay,
            device=context.get("device"),
        )

        lbcs_config = LBCSConfig(
            k=int(k),
            epsilon=config.epsilon,
            T=int(config.T),
            delta_init=config.delta_init,
            delta_lower=config.delta_lower,
            warm_start=config.warm_start,
            group_size=config.group_size,
            seed=seed,
            device=context.get("device"),
        ) if hasattr(LBCSConfig, "seed") else LBCSConfig(
            k=int(k), epsilon=config.epsilon, T=int(config.T)
        )

        lbcs = LBCS(
            model_factory=context.get("inner_factory"),
            n=int(context.get("n")),
            k=int(k),
            epsilon=config.epsilon,
            T=int(config.T),
            dataset=context.get("selection_dataset"),
            train_loader=context.get("train_loader"),
            eval_loader=context.get("eval_loader"),
            inner_config=inner_cfg,
            device=context.get("device"),
            seed=seed,
            config=lbcs_config,
        )

    if hasattr(lbcs, "run"):
        result = lbcs.run()
    elif callable(lbcs):
        result = lbcs()
    else:  # pragma: no cover - defensive
        raise TypeError("lbcs object must be callable or expose .run()")

    mask = getattr(result, "mask", None)
    if mask is None and isinstance(result, dict):
        mask = result.get("mask")
    if mask is None:  # pragma: no cover - defensive
        mask = getattr(result, "continuous_mask", None)

    size = int(getattr(result, "size", None) or getattr(result, "f2", None) or mask_size(mask))
    logger.info(
        "[table8] condition=%s k=%d -> achieved size %d (f1=%.4f)",
        context.get("condition"),
        k,
        size,
        float(getattr(result, "f1", float("nan"))),
    )
    return {
        "mask": binarize_mask(mask),
        "continuous_mask": getattr(result, "continuous_mask", None),
        "coreset_size": size,
        "f1": float(getattr(result, "f1", float("nan"))),
        "f2": float(size),
        "restarts": int(getattr(result, "restarts", 0) or 0),
        "wall_time": float(getattr(result, "wall_time", 0.0) or 0.0),
        "result": result,
    }


def evaluate_accuracy(model: Any, loader: Any, device: Optional[str] = None) -> float:
    """Top-1 accuracy (%) on ``loader``; falls back to a manual loop without torch."""
    try:
        from lbcs_repro.models.resnet18 import evaluate as _evaluate

        return float(_evaluate(model, loader, device=device))
    except Exception:
        pass
    if not _TORCH_AVAILABLE:  # pragma: no cover - no torch
        return float("nan")
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0], batch[1]
            else:  # pragma: no cover - defensive
                inputs, targets = batch, None
            if targets is None:  # pragma: no cover - defensive
                continue
            inputs = inputs.to(device) if device else inputs
            targets = targets.to(device) if device else targets
            logits = model(inputs)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            preds = logits.argmax(dim=1)
            correct += int((preds == targets).sum().item())
            total += int(targets.numel())
    return 100.0 * correct / total if total else float("nan")


def train_target_model(
    model: Any,
    train_loader: Any,
    test_loader: Any = None,
    epochs: int = PAPER_TARGET_EPOCHS,
    lr: float = PAPER_TARGET_LR,
    optimizer: str = PAPER_TARGET_OPTIMIZER,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
    device: Optional[str] = None,
    verbose: bool = False,
    log_every: int = 0,
) -> Tuple[Any, List[float]]:
    """Train the Section 5.2/5.3 target model (F-MNIST: LeNet, Adam, lr=0.001, 100 ep)."""
    if not _TORCH_AVAILABLE:  # pragma: no cover - no torch
        return model, []
    model = model.to(device) if device else model
    opt_name = str(optimizer).lower()
    if opt_name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif opt_name in ("adamw",):
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    history: List[float] = []
    for epoch in range(int(epochs)):
        model.train()
        for batch in train_loader:
            inputs, targets = (batch[0], batch[1]) if isinstance(batch, (list, tuple)) else (batch, None)
            if targets is None:  # pragma: no cover - defensive
                continue
            inputs = inputs.to(device) if device else inputs
            targets = targets.to(device) if device else targets
            opt.zero_grad()
            out = model(inputs)
            if isinstance(out, (list, tuple)):
                out = out[0]
            loss = criterion(out, targets)
            loss.backward()
            opt.step()
        if test_loader is not None:
            acc = evaluate_accuracy(model, test_loader, device=device)
            history.append(acc)
            if verbose and log_every and (epoch + 1) % log_every == 0:
                LOGGER.info("[table8] target epoch %d/%d test acc %.2f%%", epoch + 1, epochs, acc)
    return model, history


def train_and_evaluate(
    context: Dict[str, Any],
    mask: Any,
    seed: Optional[int] = None,
    config: Optional[Table8Config] = None,
    model: Optional[Any] = None,
    train_fn: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Train the target model on the constructed coreset and report clean test accuracy."""
    config = config or context.get("config") or Table8Config()
    if train_fn is not None:
        return train_fn(context, mask, config=config, seed=seed)

    size = mask_size(mask)
    if model is None and context.get("target_factory") is not None:
        try:
            model = context["target_factory"]()
        except Exception as exc:  # pragma: no cover - model unavailable
            LOGGER.warning("[table8] target model construction failed: %s", exc)
            return {"accuracy": float("nan"), "coreset_size": size, "accuracy_per_point": float("nan"), "history": []}

    if model is None or not _TORCH_AVAILABLE:  # pragma: no cover - offline
        return {"accuracy": float("nan"), "coreset_size": size, "accuracy_per_point": float("nan"), "history": []}

    loader = coreset_loader(context, mask, seed=seed)
    model, history = train_target_model(
        model,
        loader,
        test_loader=context.get("test_loader"),
        epochs=config.target_epochs,
        lr=config.target_lr,
        optimizer=config.target_optimizer,
        weight_decay=config.weight_decay,
        device=context.get("device"),
        verbose=config.verbose,
        log_every=config.log_every,
    )
    acc = evaluate_accuracy(model, context.get("test_loader"), device=context.get("device"))
    return {
        "accuracy": float(acc),
        "coreset_size": size,
        "accuracy_per_point": float(acc / size) if size else float("nan"),
        "history": history,
        "model": model,
    }


# --------------------------------------------------------------------------------------
# Sweep driver
# --------------------------------------------------------------------------------------
def run_single_cell(
    condition: str,
    k: int,
    repeat: int,
    config: Optional[Table8Config] = None,
    context: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    select_lbcs_fn: Optional[Callable[..., Any]] = None,
    train_eval_fn: Optional[Callable[..., Any]] = None,
    lbcs: Optional[Any] = None,
    context_builder: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Run one ``(condition, k, repeat)`` trial, recording the optimized coreset size."""
    config = config or Table8Config()
    logger = logger or LOGGER
    condition = canonical_condition(condition)
    seed = resolve_seed(config.seed, repeat) if seed is None else seed
    set_seed(seed)

    record: Dict[str, Any] = {
        "condition": condition,
        "k": int(k),
        "repeat": int(repeat),
        "seed": int(seed),
        "failed": False,
        "coreset_size": float("nan"),
        "f1": float("nan"),
        "accuracy": float("nan"),
    }

    try:
        if context is None:
            builder = context_builder or build_context
            context = builder(condition, config=config, repeat=repeat)

        select = select_lbcs_fn or select_lbcs_mask
        try:
            selection = select(context, int(k), config=config, seed=seed, lbcs=lbcs, logger=logger)
        except TypeError:  # tolerate narrower injected signatures
            selection = select(context, int(k))

        mask = selection.get("mask") if isinstance(selection, dict) else selection
        record["coreset_size"] = float(mask_size(mask))
        record["f1"] = float(np.asarray(selection.get("f1", np.nan)).reshape(-1)[0]) if isinstance(selection, dict) else float("nan")
        record["f2"] = record["coreset_size"]
        record["restarts"] = int(selection.get("restarts", 0)) if isinstance(selection, dict) else 0

        if train_eval_fn is not None or context.get("target_factory") is not None:
            evaluation = train_and_evaluate(context, mask, seed=seed, config=config, train_fn=train_eval_fn)
            record["accuracy"] = float(evaluation.get("accuracy", float("nan")))
            record["accuracy_per_point"] = float(evaluation.get("accuracy_per_point", float("nan")))
            record["coreset_size"] = float(evaluation.get("coreset_size", record["coreset_size"]))
        record["mask_size_check"] = int(mask_size(mask))
    except Exception as exc:  # keep long sweeps alive
        logger.warning("[table8] cell failed (condition=%s k=%s repeat=%s): %s", condition, k, repeat, exc)
        record["failed"] = True
        record["error"] = f"{type(exc).__name__}: {exc}"

    return record


def run_table8(
    config: Optional[Table8Config] = None,
    logger: Optional[logging.Logger] = None,
    cell_runner: Optional[Callable[..., Any]] = None,
    context_builder: Optional[Callable[..., Any]] = None,
    lbcs: Optional[Any] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Main Table 8 (Appendix E.3) driver: sweeps conditions x k x repeats."""
    if config is None and overrides:
        config = Table8Config.paper(**overrides)
    elif config is None:
        config = Table8Config.paper()
    elif overrides:
        config = config.with_overrides(**overrides)
    logger = logger or LOGGER

    logger.info(
        "[table8] dataset=%s conditions=%s ks=%s epsilon=%.2f T=%d repeats=%d",
        config.dataset,
        list(config.conditions),
        list(config.ks),
        config.epsilon,
        config.T,
        config.repeats,
    )

    runner = cell_runner or run_single_cell
    records: List[Dict[str, Any]] = []
    for condition in config.conditions:
        for k in config.ks:
            for repeat in range(int(config.repeats)):
                kwargs: Dict[str, Any] = {
                    "config": config,
                    "repeat": repeat,
                }
                if context_builder is not None:
                    kwargs["context_builder"] = context_builder
                if lbcs is not None:
                    kwargs["lbcs"] = lbcs
                try:
                    record = runner(canonical_condition(condition), int(k), **kwargs)
                except TypeError:
                    record = runner(canonical_condition(condition), int(k), repeat, config)
                records.append(record)

    cells = aggregate_results(records)
    checks = direction_checks(cells, config=config)
    table_text = format_table8(cells, config=config)
    artifacts = save_results(cells, records, checks, config) if config.save_artifacts else {}
    if config.plot:
        try:
            path = plot_table8(cells, os.path.join(config.output_dir, "table8_sizes.png"), config=config)
            if path:
                artifacts["plot"] = path
        except Exception as exc:  # pragma: no cover - plotting optional
            logger.warning("[table8] plotting failed: %s", exc)

    if config.verbose:
        print(table_text)

    return {
        "table": table_text,
        "config": config.to_dict(),
        "cells": cells,
        "records": records,
        "checks": checks,
        "artifacts": artifacts,
        "paper_reference": PAPER_TABLE8,
    }


def aggregate_results(records: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, int], Table8Cell]:
    """Group per-repeat records into ``(condition, k) -> Table8Cell``."""
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for rec in records:
        key = (canonical_condition(rec.get("condition")), int(rec.get("k", 0)))
        grouped.setdefault(key, []).append(rec)

    cells: Dict[Tuple[str, int], Table8Cell] = {}
    for (condition, k), recs in grouped.items():
        ok = [r for r in recs if not r.get("failed")]
        sizes = [r.get("coreset_size", np.nan) for r in ok]
        size_mean, size_std = mean_std(sizes)
        f1_mean, f1_std = mean_std([r.get("f1", np.nan) for r in ok])
        acc_mean, acc_std = mean_std([r.get("accuracy", np.nan) for r in ok])
        cells[(condition, k)] = Table8Cell(
            condition=condition,
            k=int(k),
            size_mean=size_mean,
            size_std=size_std,
            f1_mean=f1_mean,
            f1_std=f1_std,
            accuracy_mean=acc_mean,
            accuracy_std=acc_std,
            repeats=len(ok),
            failures=len(recs) - len(ok),
            raw=list(recs),
        )
    return cells


def direction_checks(
    cells: Dict[Tuple[str, int], Table8Cell],
    config: Optional[Table8Config] = None,
) -> Dict[str, Any]:
    """Validate the qualitative Appendix E.3 claims against the measured cells."""
    config = config or Table8Config()
    checks: Dict[str, Any] = {"per_cell": {}, "summary": {}, "ok": True}

    for (condition, k), cell in sorted(cells.items()):
        ref = cell.paper_reference()
        entry: Dict[str, Any] = {
            "size_mean": cell.size_mean,
            "size_std": cell.size_std,
            "k": int(k),
            "below_k": bool(cell.size_mean < k) if cell.repeats else None,
            "deviations": {},
        }
        if ref is not None:
            dev = cell.size_mean - ref[0]
            entry["deviations"]["size"] = {
                "measured": cell.size_mean,
                "paper": ref[0],
                "abs_diff": abs(dev),
                "within_tolerance": bool(abs(dev) <= config.size_tolerance),
            }
        checks["per_cell"][f"{condition}_k{k}"] = entry

    # (i) optimized sizes are always below the predefined k
    below = [
        checks["per_cell"][key]["below_k"]
        for key in checks["per_cell"]
        if checks["per_cell"][key]["below_k"] is not None
    ]
    checks["optimized_below_k"] = bool(below) and all(below)

    # (ii) larger k implies larger optimized size, per condition
    monotone: Dict[str, bool] = {}
    for condition in {cond for cond, _ in cells}:
        ks = sorted(k for (cond, k) in cells if cond == condition)
        sizes = [cells[(condition, k)].size_mean for k in ks]
        monotone[condition] = all(
            sizes[i + 1] >= sizes[i] - 1e-6 for i in range(len(sizes) - 1)
        ) if len(sizes) > 1 else True
    checks["size_increases_with_k"] = monotone

    # (iii) higher noise corrupts more, hence the optimized size is no larger
    if ("noise30" in {c for c, _ in cells}) and ("noise50" in {c for c, _ in cells}):
        common = sorted(
            k for k in {kk for c, kk in cells if c == "noise30"} & {kk for c, kk in cells if c == "noise50"}
        )
        checks["noise50_size_le_noise30"] = all(
            cells[("noise50", k)].size_mean <= cells[("noise30", k)].size_mean + 1.0 for k in common
        ) if common else None

    # (iv) class imbalance keeps more data than label noise (paper Table 8)
    if ("imbalance" in {c for c, _ in cells}) and ("noise30" in {c for c, _ in cells}):
        common = sorted(
            k for k in {kk for c, kk in cells if c == "imbalance"} & {kk for c, kk in cells if c == "noise30"}
        )
        checks["imbalance_size_ge_noise30"] = all(
            cells[("imbalance", k)].size_mean >= cells[("noise30", k)].size_mean - 1.0 for k in common
        ) if common else None

    checks["summary"] = {
        "optimized_below_k": checks["optimized_below_k"],
        "size_increases_with_k": monotone,
        "num_cells": len(cells),
        "failures": int(sum(c.failures for c in cells.values())),
    }
    reference_flags = [
        v["deviations"]["size"]["within_tolerance"]
        for v in checks["per_cell"].values()
        if v.get("deviations", {}).get("size")
    ]
    checks["matches_paper_within_tolerance"] = bool(reference_flags) and all(reference_flags)
    checks["ok"] = bool(checks["optimized_below_k"]) and (
        checks["matches_paper_within_tolerance"] if reference_flags else True
    )
    return checks


def format_table8(
    cells: Dict[Tuple[str, int], Table8Cell],
    config: Optional[Table8Config] = None,
) -> str:
    """Render the measured optimized coreset sizes in the layout of Table 8."""
    config = config or Table8Config()
    ks = sorted({int(k) for _, k in cells}) or list(config.ks)
    header = "Imperfect supervision".ljust(30) + "".join(f"k={k}".rjust(18) for k in ks)
    lines = [header, "-" * len(header)]
    for condition in CONDITIONS:
        row = CONDITION_LABELS.get(condition, condition).ljust(30)
        for k in ks:
            cell = cells.get((condition, int(k)))
            if cell is None:
                row += "".rjust(18)
            else:
                row += format_mean_std(cell.size_mean, cell.size_std).rjust(18)
        lines.append(row)

    lines.append("")
    lines.append("Paper reference (Appendix E.3, Table 8):")
    paper_header = "Imperfect supervision".ljust(30) + "".join(f"k={k}".rjust(18) for k in ks)
    lines.append(paper_header)
    lines.append("-" * len(paper_header))
    for condition in CONDITIONS:
        refs = PAPER_TABLE8.get(condition, {})
        if not refs:
            continue
        row = CONDITION_LABELS.get(condition, condition).ljust(30)
        for k in ks:
            ref = refs.get(int(k))
            row += (format_mean_std(*ref).rjust(18) if ref else "".rjust(18))
        lines.append(row)

    if cells:
        lines.append("")
        lines.append("Deviations from the paper (measured - paper mean):")
        for (condition, k), cell in sorted(cells.items()):
            dev = cell.deviation()
            if dev is None:
                continue
            lines.append(
                f"  {condition:<10} k={k:<5} measured={cell.size_mean:8.1f} "
                f"paper={cell.paper_reference()[0]:8.1f} diff={dev:+7.1f}"
            )
    return "\n".join(lines)


def plot_table8(
    cells: Dict[Tuple[str, int], Table8Cell],
    out_path: str,
    config: Optional[Table8Config] = None,
    dpi: int = 150,
) -> Optional[str]:
    """Plot optimized coreset size vs predefined size k for each condition."""
    if not _MPL_AVAILABLE or not cells:
        return None
    config = config or Table8Config()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ks = sorted({int(k) for _, k in cells})
    for condition in CONDITIONS:
        pts = [(k, cells[(condition, k)].size_mean, cells[(condition, k)].size_std) for k in ks if (condition, k) in cells]
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        es = [p[2] for p in pts]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=CONDITION_LABELS.get(condition, condition))
        refs = [(k, PAPER_TABLE8[condition][k][0]) for k in ks if k in PAPER_TABLE8.get(condition, {})]
        if refs:
            ax.plot([r[0] for r in refs], [r[1] for r in refs], linestyle="--", linewidth=1, alpha=0.6)
    ax.plot(ks, ks, color="grey", linestyle=":", label="k (predefined)")
    ax.set_xlabel("Predefined coreset size k")
    ax.set_ylabel("Optimized coreset size (ours)")
    ax.set_title("Table 8: optimized coreset sizes under imperfect supervision")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def save_results(
    cells: Dict[Tuple[str, int], Table8Cell],
    records: Sequence[Dict[str, Any]],
    checks: Dict[str, Any],
    config: Table8Config,
) -> Dict[str, str]:
    """Persist Table 8 artifacts (JSON / CSV / TXT / JSONL / checks)."""
    out_dir = config.output_dir
    os.makedirs(out_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    payload = {
        "config": config.to_dict(),
        "paper_reference": {k: {int(kk): list(vv) for kk, vv in v.items()} for k, v in PAPER_TABLE8.items()},
        "cells": [cell.to_dict() for _, cell in sorted(cells.items())],
        "checks": checks,
    }
    json_path = os.path.join(out_dir, "table8.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
    paths["json"] = json_path

    csv_path = os.path.join(out_dir, "table8.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "condition",
                "k",
                "size_mean",
                "size_std",
                "f1_mean",
                "f1_std",
                "accuracy_mean",
                "accuracy_std",
                "paper_size_mean",
                "paper_size_std",
                "repeats",
                "failures",
            ]
        )
        for _, cell in sorted(cells.items()):
            ref = cell.paper_reference() or (None, None)
            writer.writerow(
                [
                    cell.condition,
                    cell.k,
                    f"{cell.size_mean:.4f}",
                    f"{cell.size_std:.4f}",
                    f"{cell.f1_mean:.6f}",
                    f"{cell.f1_std:.6f}",
                    f"{cell.accuracy_mean:.4f}",
                    f"{cell.accuracy_std:.4f}",
                    "" if ref[0] is None else f"{ref[0]:.4f}",
                    "" if ref[1] is None else f"{ref[1]:.4f}",
                    cell.repeats,
                    cell.failures,
                ]
            )
    paths["csv"] = csv_path

    txt_path = os.path.join(out_dir, "table8.txt")
    with open(txt_path, "w", encoding="utf-8") as handle:
        handle.write(format_table8(cells, config=config))
        handle.write("\n\n")
        handle.write("Checks:\n")
        handle.write(json.dumps(checks, indent=2, default=_json_default))
        handle.write("\n")
    paths["txt"] = txt_path

    jsonl_path = os.path.join(out_dir, "table8_raw.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, default=_json_default) + "\n")
    paths["jsonl"] = jsonl_path

    checks_path = os.path.join(out_dir, "table8_checks.json")
    with open(checks_path, "w", encoding="utf-8") as handle:
        json.dump(checks, handle, indent=2, default=_json_default)
    paths["checks"] = checks_path

    return paths


def _json_default(obj: Any) -> Any:
    """JSON encoder fallback for NumPy scalars/arrays and torch tensors."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "detach"):
        try:
            return obj.detach().cpu().numpy().tolist()
        except Exception:  # pragma: no cover
            return str(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:  # pragma: no cover
            pass
    return str(obj)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Table 8 (Appendix E.3): optimized coreset sizes under imperfect supervision."
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config with a table8 block.")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--conditions", type=str, nargs="*", default=None)
    parser.add_argument("--ks", type=int, nargs="*", default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--inner-epochs", type=int, default=None)
    parser.add_argument("--target-epochs", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--paper", action="store_true", help="Use the paper protocol (default).")
    parser.add_argument("--smoke", action="store_true", help="Fast CPU sanity-check run.")
    parser.add_argument("--patience", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--selftest", action="store_true", help="Run the offline self-test and exit.")
    parser.add_argument("--no-save", action="store_true", help="Do not write artifacts.")
    parser.add_argument("--no-plot", action="store_true", help="Do not write the plot.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_argparser().parse_args(argv)

    if args.selftest:
        report = _selftest(verbose=True)
        return 0 if report.get("ok") else 1

    cfg = Table8Config.smoke() if args.smoke else Table8Config.paper()
    if args.config:
        if not _YAML_AVAILABLE:  # pragma: no cover
            LOGGER.warning("PyYAML unavailable; ignoring --config %s", args.config)
        else:
            with open(args.config, "r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
            cfg = Table8Config.from_dict(loaded).with_overrides(**cfg.to_dict())

    overrides = {
        "dataset": args.dataset,
        "conditions": tuple(args.conditions) if args.conditions else None,
        "ks": tuple(args.ks) if args.ks else None,
        "epsilon": args.epsilon,
        "T": args.T,
        "repeats": args.repeats,
        "inner_epochs": args.inner_epochs,
        "target_epochs": args.target_epochs,
        "output_dir": args.output_dir,
        "seed": args.seed,
        "device": args.device,
        "verbose": True if args.verbose else None,
        "save_artifacts": False if args.no_save else None,
        "plot": False if args.no_plot else None,
    }
    cfg = cfg.with_overrides(**overrides)

    results = run_table8(config=cfg)
    print(results["table"])
    return 0


# --------------------------------------------------------------------------------------
# Offline self-test (no torch / no datasets needed)
# --------------------------------------------------------------------------------------
class _SyntheticCellRunner:
    """Deterministic stand-in runner emulating LBCS shrinking the coreset.

    Not a substitute for the real experiment; used only so the aggregation/formatting/
    validation plumbing can be exercised offline.
    """

    def __init__(self, sizes: Optional[Dict[Tuple[str, int], float]] = None, noise: float = 0.0):
        self.sizes = sizes or {}
        self.noise = noise
        self.calls: List[Tuple[str, int, int]] = []

    def __call__(self, condition: str, k: int, repeat: int = 0, config: Optional[Table8Config] = None, **kwargs: Any) -> Dict[str, Any]:
        condition = canonical_condition(condition)
        self.calls.append((condition, int(k), int(repeat)))
        base = self.sizes.get((condition, int(k)))
        if base is None:
            base = PAPER_TABLE8.get(condition, {}).get(int(k), (float(k) * 0.95, 0.0))[0]
        rng = np.random.default_rng(abs(hash((condition, k, repeat))) % (2**31))
        size = base + (rng.normal(0.0, self.noise) if self.noise else 0.0)
        return {
            "condition": condition,
            "k": int(k),
            "repeat": int(repeat),
            "failed": False,
            "coreset_size": float(size),
            "f1": float(1.0 + 0.01 * k),
            "f2": float(size),
            "accuracy": 79.5,
        }


def _selftest(verbose: bool = True) -> Dict[str, Any]:
    """Offline verification of masks, aggregation, formatting and checks."""
    import tempfile

    report: Dict[str, Any] = {"ok": True, "checks": {}}

    def check(name: str, condition: bool, detail: Any = None) -> bool:
        report["checks"][name] = {"passed": bool(condition), "detail": detail}
        if not condition:
            report["ok"] = False
        return bool(condition)

    # 1) mask projection / size
    relaxed = np.array([-1.5, -0.2, 0.0, 0.4, 2.0], dtype=np.float64)
    b = binarize_mask(relaxed)
    check("binarize_rule", np.array_equal(b, np.array([0, 0, 1, 1, 1], dtype=np.float32)), b.tolist())
    check("mask_size", mask_size(relaxed) == 3, mask_size(relaxed))
    check("mask_indices", np.array_equal(mask_indices(relaxed), np.array([2, 3, 4])), mask_indices(relaxed).tolist())

    # 2) condition normalisation
    check("condition_aliases", canonical_condition("With 30% corrupted labels") == "noise30")
    check("condition_imbalance_alias", canonical_condition("class-imbalanced") == "imbalance")
    check("dataset_alias", canonical_dataset("fmnist") == "F-MNIST")

    # 3) config round-trip
    cfg = Table8Config.from_dict({"table8": {"ks": [1000, 2000], "epsilon": 0.3, "repeats": 3, "T": 250}})
    check("config_from_dict", list(cfg.ks) == [1000, 2000] and cfg.epsilon == 0.3 and cfg.repeats == 3 and cfg.T == 250,
          cfg.to_dict())
    check("config_paper_defaults", Table8Config.paper().epsilon == 0.2 and Table8Config.paper().T == 500)

    # 4) sweep with the synthetic runner reproduces table structure
    runner = _SyntheticCellRunner(noise=2.0)
    cfg_small = Table8Config(ks=(1000, 2000), conditions=("noise30", "noise50", "imbalance"), repeats=5,
                             save_artifacts=False, plot=False)
    records: List[Dict[str, Any]] = []
    for condition in cfg_small.conditions:
        for k in cfg_small.ks:
            for repeat in range(cfg_small.repeats):
                records.append(runner(condition, int(k), repeat, config=cfg_small))
    cells = aggregate_results(records)
    check("num_cells", len(cells) == 6, len(cells))
    check("repeats_per_cell", all(c.repeats == 5 for c in cells.values()), [c.repeats for c in cells.values()])

    # 5) mean/std close to the paper anchors for the noise30 row
    n30_1000 = cells[("noise30", 1000)]
    check("mean_near_paper", abs(n30_1000.size_mean - PAPER_TABLE8["noise30"][1000][0]) < 15.0,
          n30_1000.size_mean)

    # 6) direction checks
    checks = direction_checks(cells, config=cfg_small)
    check("direction_checks_ok", checks["optimized_below_k"] is True, checks.get("optimized_below_k"))
    check("size_increases_with_k", all(checks["size_increases_with_k"].values()),
          checks.get("size_increases_with_k"))

    # 7) formatting
    table_text = format_table8(cells, config=cfg_small)
    check("format_has_conditions", all(CONDITION_LABELS[c] in table_text for c in cfg_small.conditions))
    check("format_has_paper_block", "Paper reference" in table_text)

    # 8) artifacts
    with tempfile.TemporaryDirectory() as tmp:
        cfg_art = replace(cfg_small, output_dir=tmp, save_artifacts=True)
        paths = save_results(cells, records, checks, cfg_art)
        check("artifacts_written", all(os.path.exists(p) for p in paths.values()), list(paths.keys()))
        with open(paths["json"], "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        check("json_payload", "cells" in payload and len(payload["cells"]) == 6, len(payload.get("cells", [])))

    if verbose:
        print("Table 8 self-test:", "OK" if report["ok"] else "FAILED")
        for name, info in report["checks"].items():
            print(f"  [{'ok' if info['passed'] else 'FAIL'}] {name}: {info['detail']}")
    return report


#: Convenience aliases mirroring the other experiment drivers.
run = run_table8
table8_sizes = run_table8


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
