"""Figure 2 / Table 8: coreset selection under imperfect supervision (Section 5.3).

Reproduces the paper's Section 5.3 robustness study:

* **Figure 2(a)** -- test accuracy (%) of every coreset-selection method on
  F-MNIST whose training labels are corrupted with 30% symmetric noise
  (``k in {1000, 2000, 3000, 4000}``).
* **Figure 2(b)** -- test accuracy (%) when the training split follows an
  *exponential* class imbalance with imbalanced ratio 0.01 (Yin et al. 2019 /
  Cao et al. 2019 scheme, adapted to F-MNIST).
* **Appendix E.2** -- the same with 50% symmetric label noise.
* **Appendix E.3 / Table 8** -- the optimized coreset sizes produced by LBCS in
  these three imperfect-supervision cases.

Paper text (verbatim essentials)
--------------------------------
"We employ FMNIST here. We inject 30% symmetric label noise ... into the
original clean F-MNIST to generate the noisy version of F-MNIST. Namely, the
labels of 30% training data are flipped. The predefined coreset size k is set
to 1000, 2000, 3000, and 4000 respectively." ... "We also evaluate LBCS when
the noise level is higher, i.e., 50%."

"For the class-imbalanced experiment ... The exponential type of class
imbalance is used. The imbalanced ratio is set to 0.01."

Only the **training** split is corrupted; the test split stays clean (as
required by the reproduction plan: "Imbalance must affect only the training
set, never the test set.").
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Optional (soft) dependencies
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - exercised only when torch is present
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    _TORCH_AVAILABLE = False

try:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401

    _MPL_AVAILABLE = True
except Exception:  # pragma: no cover
    plt = None  # type: ignore
    _MPL_AVAILABLE = False

LOGGER = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Paper-stated constants
# --------------------------------------------------------------------------- #
PAPER_DATASET = "F-MNIST"
PAPER_KS: Tuple[int, ...] = (1000, 2000, 3000, 4000)
PAPER_EPSILON = 0.2
PAPER_T = 500
PAPER_REPEATS = 10
PAPER_NOISE_RATE_30 = 0.30
PAPER_NOISE_RATE_50 = 0.50
PAPER_IMBALANCE_RATIO = 0.01

#: Figure 2 uses the same baseline suite as Tables 2-3 (Appendix D.1).
METHOD_ORDER: Tuple[str, ...] = (
    "Uniform",
    "EL2N",
    "GraNd",
    "Influential",
    "Moderate",
    "CCS",
    "Probabilistic",
)
LBCS_LABEL = "LBCS (ours)"

#: Condition identifiers -> (display title, kind, severity)
CONDITIONS: Tuple[str, ...] = ("noise30", "noise50", "imbalance")

CONDITION_TITLES: Dict[str, str] = {
    "noise30": "30% corrupted labels",
    "noise50": "50% corrupted labels",
    "imbalance": "Class-imbalanced data (ratio 0.01)",
}

#: Appendix E.3 / Table 8 -- mean +- std of the *optimized coreset sizes*.
PAPER_TABLE8_SIZES: Dict[str, Dict[int, Tuple[float, float]]] = {
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

#: Target-model recipes (Section 5.2 / Appendix D.2) -- F-MNIST uses LeNet.
TARGET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "F-MNIST": {
        "model": "LeNet",
        "optimizer": "adam",
        "lr": 0.001,
        "epochs": 100,
        "batch_size": 128,
        "weight_decay": 0.0,
    },
}

#: Inner-loop (coreset selection) recipes -- Section 5.2 states Adam, lr 0.001.
INNER_CONFIGS: Dict[str, Dict[str, Any]] = {
    "F-MNIST": {
        "model": "LeNet",
        "optimizer": "adam",
        "lr": 0.001,
        "epochs": 100,
        "batch_size": 128,
    },
}

# Suggested defaults (NOT stated in the paper)
SUGGESTED_DELTA_INIT = 0.1
SUGGESTED_DELTA_LOWER = 1e-3
SUGGESTED_EVAL_BATCH_SIZE = 256
DEFAULT_OUTPUT_DIR = "results/figure2"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def set_seed(seed: Optional[int]) -> None:
    """Seed NumPy and (when available) PyTorch."""
    if seed is None:
        return
    np.random.seed(int(seed) % (2**32 - 1))
    if _TORCH_AVAILABLE:
        torch.manual_seed(int(seed) % (2**32 - 1))
        if torch.cuda.is_available():  # pragma: no cover
            torch.cuda.manual_seed_all(int(seed) % (2**32 - 1))


def resolve_seed(seed: Optional[int], repeat: int = 0, base: int = 0) -> int:
    """Deterministic per-repeat seed."""
    return int(base) + 10007 * int(repeat) + (0 if seed is None else int(seed))


def binarize_mask(mask: Any) -> np.ndarray:
    """Project a binary / relaxed ``[-1, 1]`` / probability mask to ``{0, 1}``.

    Implements the Appendix A rule: values in ``[-1, 0)`` become ``0`` and
    values in ``[0, 1]`` become ``1``.
    """
    if mask is None:
        raise ValueError("mask is None")
    if _TORCH_AVAILABLE and isinstance(mask, torch.Tensor):
        arr = mask.detach().cpu().numpy()
    else:
        arr = np.asarray(mask)
    arr = np.asarray(arr).reshape(-1).astype(np.float64)
    return (arr >= 0.0).astype(np.float32)


def mask_indices(mask: Any) -> np.ndarray:
    """Indices of the selected examples in a coreset mask."""
    return np.flatnonzero(binarize_mask(mask) > 0.5)


def mask_size(mask: Any) -> int:
    """``f2(m) = ||m||_0`` of a (possibly relaxed) mask."""
    return int(mask_indices(mask).size)


def canonical_condition(name: str) -> str:
    """Normalise a condition identifier (``30%noise`` -> ``noise30``)."""
    key = str(name).strip().lower().replace("%", "").replace("_", "").replace("-", "")
    table = {
        "noise30": "noise30",
        "noise03": "noise30",
        "noise0.30": "noise30",
        "30noise": "noise30",
        "noisy30": "noise30",
        "30": "noise30",
        "noise50": "noise50",
        "noise05": "noise50",
        "noise0.50": "noise50",
        "50noise": "noise50",
        "noisy50": "noise50",
        "50": "noise50",
        "imbalance": "imbalance",
        "imbalanced": "imbalance",
        "imb": "imbalance",
        "exp": "imbalance",
        "exponential": "imbalance",
    }
    if key not in table:
        raise KeyError(
            f"unknown condition {name!r}; available: {sorted(set(table.values()))}"
        )
    return table[key]


def noise_rate_for(condition: str) -> float:
    """Symmetric-noise rate of a condition (0.0 for the imbalance condition)."""
    condition = canonical_condition(condition)
    if condition == "noise30":
        return PAPER_NOISE_RATE_30
    if condition == "noise50":
        return PAPER_NOISE_RATE_50
    return 0.0


def canonical_dataset(name: str) -> str:
    """Normalise dataset names (``fmnist`` -> ``F-MNIST``)."""
    key = str(name).strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    if key in ("fmnist", "fashionmnist", "fashion"):
        return "F-MNIST"
    if key in ("mnists", "mnist"):
        return "MNIST-S"
    if key == "svhn":
        return "SVHN"
    if key in ("cifar10", "cifar"):
        return "CIFAR-10"
    return str(name)


# --------------------------------------------------------------------------- #
# Configuration containers
# --------------------------------------------------------------------------- #
@dataclass
class RobustnessConfig:
    """Protocol for the Section 5.3 / Figure 2 + Table 8 experiments."""

    dataset: str = PAPER_DATASET
    conditions: Sequence[str] = field(default_factory=lambda: tuple(CONDITIONS))
    ks: Sequence[int] = field(default_factory=lambda: tuple(PAPER_KS))
    methods: Sequence[str] = field(default_factory=lambda: tuple(METHOD_ORDER))
    epsilon: float = PAPER_EPSILON
    T: int = PAPER_T
    repeats: int = PAPER_REPEATS
    inner_epochs: int = 100
    inner_lr: float = 0.001
    inner_optimizer: str = "adam"
    inner_momentum: float = 0.9
    inner_weight_decay: float = 0.0
    batch_size: int = 128
    eval_batch_size: int = SUGGESTED_EVAL_BATCH_SIZE
    target_epochs: int = 100
    target_lr: float = 0.001
    target_optimizer: str = "adam"
    num_workers: int = 0
    delta_init: float = SUGGESTED_DELTA_INIT
    delta_lower: float = SUGGESTED_DELTA_LOWER
    warm_start: bool = True
    group_size: int = 1
    #: Which split is used to evaluate ``f1(m)`` inside the outer loop.
    f1_eval_split: str = "train"
    device: Optional[str] = None
    seed: int = 0
    log_every: int = 0
    data_root: Optional[str] = None
    output_dir: str = DEFAULT_OUTPUT_DIR
    save_artifacts: bool = True
    plot: bool = True
    verbose: bool = False

    # -- constructors ------------------------------------------------------ #
    @classmethod
    def paper(cls, **overrides: Any) -> "RobustnessConfig":
        """Exact Section 5.3 protocol (k in {1000..4000}, eps=0.2, T=500, 10 reps)."""
        cfg = cls()
        cfg.conditions = ("noise30", "noise50", "imbalance")
        cfg.ks = tuple(PAPER_KS)
        cfg.epsilon = PAPER_EPSILON
        cfg.T = PAPER_T
        cfg.repeats = PAPER_REPEATS
        cfg.f1_eval_split = "train"
        return cfg.with_overrides(**overrides)

    @classmethod
    def smoke(cls, **overrides: Any) -> "RobustnessConfig":
        """Tiny offline-friendly configuration."""
        cfg = cls(
            conditions=("noise30",),
            ks=(200,),
            repeats=1,
            T=3,
            inner_epochs=1,
            target_epochs=1,
            save_artifacts=False,
            plot=False,
        )
        return cfg.with_overrides(**overrides)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "RobustnessConfig":
        if not data:
            return cls()
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in dict(data).items() if k in known})

    def with_overrides(self, **overrides: Any) -> "RobustnessConfig":
        cfg = RobustnessConfig(**{k: getattr(self, k) for k in self.__dataclass_fields__})  # type: ignore[attr-defined]
        for key, value in overrides.items():
            if value is None:
                continue
            if key not in self.__dataclass_fields__:  # type: ignore[attr-defined]
                LOGGER.debug("ignoring unknown RobustnessConfig key %r", key)
                continue
            setattr(cfg, key, value)
        cfg.conditions = tuple(canonical_condition(c) for c in cfg.conditions)
        cfg.ks = tuple(int(k) for k in cfg.ks)
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key in self.__dataclass_fields__:  # type: ignore[attr-defined]
            value = getattr(self, key)
            out[key] = list(value) if isinstance(value, (tuple, list)) else value
        return out


@dataclass
class RobustnessContext:
    """Everything needed to evaluate one condition on one benchmark."""

    condition: str
    dataset: str
    train_loader: Any
    test_loader: Any
    eval_loader: Any
    n: int
    num_classes: int
    inner_factory: Callable[..., Any]
    target_factory: Callable[..., Any]
    noisy_targets: Optional[np.ndarray] = None
    clean_targets: Optional[np.ndarray] = None
    imbalance_stats: Optional[Dict[str, Any]] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition": self.condition,
            "dataset": self.dataset,
            "n": int(self.n),
            "num_classes": int(self.num_classes),
            "imbalance_stats": self.imbalance_stats,
            "extras": {k: v for k, v in self.extras.items() if isinstance(v, (int, float, str, bool))},
        }


@dataclass
class RobustnessCell:
    """Mean/std accuracy (and LBCS coreset size) for one ``(condition, k)`` cell."""

    condition: str
    k: int
    accuracy: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    coreset_size: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    accuracy_per_point: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    repeats: int = 0
    failures: int = 0

    def mean(self, method: str) -> float:
        return float(self.accuracy.get(method, (float("nan"), 0.0))[0])

    def size_mean(self, method: str = LBCS_LABEL) -> float:
        return float(self.coreset_size.get(method, (float("nan"), 0.0))[0])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition": self.condition,
            "k": int(self.k),
            "repeats": int(self.repeats),
            "failures": int(self.failures),
            "accuracy": {k: list(v) for k, v in self.accuracy.items()},
            "coreset_size": {k: list(v) for k, v in self.coreset_size.items()},
            "accuracy_per_point": {k: list(v) for k, v in self.accuracy_per_point.items()},
        }


# --------------------------------------------------------------------------- #
# Data / model plumbing
# --------------------------------------------------------------------------- #
def _model_factory_for(dataset: str, role: str = "inner", **kwargs: Any) -> Callable[..., Any]:
    """Resolve a fresh-network factory for ``role`` on ``dataset``."""
    dataset = canonical_dataset(dataset)
    try:  # preferred: the aggregated registry
        from lbcs_repro.models import default_model_for, model_factory

        name = default_model_for(dataset, role)
        return model_factory(name, **kwargs)
    except Exception as exc:  # pragma: no cover - fallback path
        LOGGER.debug("models registry unavailable (%s); falling back", exc)

    if dataset == "F-MNIST":
        from lbcs_repro.models.lenet import lenet_factory

        return lenet_factory(**kwargs)
    if dataset == "SVHN":
        from lbcs_repro.models.svhn_cnn import svhn_cnn_factory

        return svhn_cnn_factory(**kwargs)
    if dataset == "CIFAR-10":
        from lbcs_repro.models.cifar_cnn import cifar_cnn_factory

        return cifar_cnn_factory(**kwargs)
    from lbcs_repro.models.convnet import convnet_factory

    return convnet_factory(**kwargs)


def build_robustness_context(
    condition: str,
    config: Optional[RobustnessConfig] = None,
    repeat: int = 0,
    device: Optional[str] = None,
) -> RobustnessContext:
    """Build the corrupted (train-only) loaders and model factories for a condition."""
    config = config or RobustnessConfig()
    condition = canonical_condition(condition)
    dataset = canonical_dataset(config.dataset)
    seed = resolve_seed(config.seed, repeat)
    set_seed(seed)

    from lbcs_repro.data.robustness import (
        EXPONENTIAL_IMBALANCE_RATIO,
        get_imbalanced_loaders,
        get_noisy_loaders,
    )

    num_classes: Optional[int] = None
    try:
        from lbcs_repro.data.datasets import num_classes as _num_classes

        num_classes = int(_num_classes(dataset))
    except Exception:  # pragma: no cover
        num_classes = 10

    eval_split = str(getattr(config, "f1_eval_split", "train")).lower()
    extras: Dict[str, Any] = {"seed": seed}

    if condition in ("noise30", "noise50"):
        rate = noise_rate_for(condition)
        train_loader, test_loader = get_noisy_loaders(
            dataset,
            noise_rate=rate,
            batch_size=config.batch_size,
            root=config.data_root,
            num_workers=config.num_workers,
            seed=seed,
            num_classes=num_classes,
            return_index=True,
            shuffle_train=True,
        )
        extras["noise_rate"] = rate
        noisy_targets = getattr(train_loader.dataset, "targets", None)
        clean_targets = getattr(train_loader.dataset, "clean_targets", None)
        imbalance_stats = None
    elif condition == "imbalance":
        ratio = float(PAPER_IMBALANCE_RATIO)
        if ratio == float(EXPONENTIAL_IMBALANCE_RATIO):
            pass
        train_loader, test_loader = get_imbalanced_loaders(
            dataset,
            ratio=ratio,
            batch_size=config.batch_size,
            root=config.data_root,
            num_workers=config.num_workers,
            seed=seed,
            num_classes=num_classes,
            return_index=True,
            shuffle_train=True,
        )
        extras["imbalance_ratio"] = ratio
        imbalance_stats = getattr(train_loader.dataset, "imbalance_stats", None)
        noisy_targets = getattr(train_loader.dataset, "targets", None)
        clean_targets = None
    else:  # pragma: no cover - defensive
        raise KeyError(f"unsupported condition {condition!r}")

    # ``n`` is the size of the (corrupted) *selection pool*: the entire training
    # split as seen by the selector. Masks are defined on that pool.
    n = _resolve_n(train_loader)
    if imbalance_stats is None and condition != "imbalance":
        imbalance_stats = None
    else:
        imbalance_stats = imbalance_stats

    # f1(m) is measured over the full selection pool by default (Section 2 Eq. 1)
    eval_loader = test_loader if eval_split in ("test", "val") else train_loader

    return RobustnessContext(
        condition=condition,
        dataset=dataset,
        train_loader=train_loader,
        test_loader=test_loader,
        eval_loader=eval_loader,
        n=n,
        num_classes=int(num_classes or 10),
        inner_factory=_model_factory_for(dataset, "inner", num_classes=num_classes or 10),
        target_factory=_model_factory_for(dataset, "target", num_classes=num_classes or 10),
        noisy_targets=np.asarray(noisy_targets) if noisy_targets is not None else None,
        clean_targets=np.asarray(clean_targets) if clean_targets is not None else None,
        imbalance_stats=imbalance_stats,
        extras=extras,
    )


def _resolve_n(loader: Any) -> int:
    """Number of examples in the selection pool behind a loader."""
    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        raise ValueError("loader has no .dataset")
    for attr in ("targets", "labels", "tensors"):
        value = getattr(dataset, attr, None)
        if value is None:
            continue
        if attr == "tensors":
            try:
                return int(len(value[0]))
            except Exception:  # pragma: no cover
                continue
        try:
            return int(len(value))
        except TypeError:  # pragma: no cover
            continue
    try:
        return int(len(dataset))  # type: ignore[arg-type]
    except Exception:  # pragma: no cover
        return 0


def coreset_loader(
    context: RobustnessContext,
    mask: Any,
    batch_size: int = 128,
    num_workers: int = 0,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> Any:
    """DataLoader over the coreset selected by ``mask``.

    The corrupted labels of the training pool are used for post-selection target
    training (that is what "coreset selection with corrupted labels" means); the
    test split always stays clean.
    """
    indices = mask_indices(mask)
    from lbcs_repro.data.datasets import make_loader, subset_dataset

    subset = subset_dataset(context.train_loader.dataset, indices, return_index=False)
    return make_loader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        seed=seed,
        return_index=False,
    )


# --------------------------------------------------------------------------- #
# LBCS + baselines
# --------------------------------------------------------------------------- #
def select_lbcs_mask(
    context: RobustnessContext,
    k: int,
    config: RobustnessConfig,
    seed: Optional[int] = None,
    lbcs: Any = None,
) -> Dict[str, Any]:
    """Run Algorithm 1 (LBCS) on the (corrupted) selection pool."""
    from lbcs_repro.lbcs.bilevel import LBCS, LBCSConfig, InnerTrainConfig

    if lbcs is not None:  # dependency injection (offline tests)
        result = lbcs(context, k, seed)
        mask = result.get("mask") if isinstance(result, dict) else result
        return {
            "mask": binarize_mask(mask),
            "coreset_size": mask_size(mask),
            "f1": float(result.get("f1", np.nan)) if isinstance(result, dict) else np.nan,
            "f2": float(result.get("f2", mask_size(mask))) if isinstance(result, dict) else mask_size(mask),
            "wall_time": float(result.get("wall_time", 0.0)) if isinstance(result, dict) else 0.0,
        }

    inner = InnerTrainConfig(
        optimizer=config.inner_optimizer,
        lr=config.inner_lr,
        momentum=config.inner_momentum,
        weight_decay=config.inner_weight_decay,
        epochs=config.inner_epochs,
        batch_size=config.batch_size,
        device=config.device,
    )
    t0 = time.time()
    runner = LBCS(
        model_factory=context.inner_factory,
        n=context.n,
        k=int(k),
        epsilon=config.epsilon,
        T=config.T,
        dataset=context.eval_loader.dataset,
        inner_config=inner,
        device=config.device,
        seed=seed,
        warm_start=config.warm_start,
        group_size=config.group_size,
        delta_init=config.delta_init,
        delta_lower=config.delta_lower,
        log_every=config.log_every,
    )
    try:
        result = runner.run()
    except TypeError:  # pragma: no cover - tolerated signature drift
        result = LBCS(
            context.inner_factory,
            context.n,
            int(k),
            epsilon=config.epsilon,
            T=config.T,
            dataset=context.eval_loader.dataset,
            inner_config=inner,
            device=config.device,
            seed=seed,
        ).run()

    mask = getattr(result, "mask", None)
    return {
        "mask": binarize_mask(mask),
        "coreset_size": int(getattr(result, "size", mask_size(mask))),
        "f1": float(getattr(result, "f1", np.nan)),
        "f2": float(getattr(result, "f2", mask_size(mask))),
        "restarts": int(getattr(result, "restarts", 0)),
        "wall_time": float(getattr(result, "wall_time", time.time() - t0)),
        "result": result,
    }


def select_baseline_mask(
    name: str,
    context: RobustnessContext,
    k: int,
    config: RobustnessConfig,
    seed: Optional[int] = None,
    model: Any = None,
) -> np.ndarray:
    """Instantiate a baseline by name and produce a ``k``-sized coreset mask."""
    from lbcs_repro.baselines import make_baseline
    from lbcs_repro.baselines.base import indices_to_mask
    from lbcs_repro.baselines.uniform import uniform_indices

    n = int(context.n)
    try:
        selector = make_baseline(
            name,
            seed=seed,
            device=config.device,
            num_classes=context.num_classes,
            model=model,
        )
    except Exception as exc:  # pragma: no cover - degrade to uniform
        LOGGER.warning("baseline %s unavailable (%s); using Uniform", name, exc)
        return indices_to_mask(uniform_indices(n, int(k), seed=resolve_seed(seed, 0)), n)

    targets = context.noisy_targets
    dataset = getattr(context.train_loader, "dataset", None)
    loader = context.train_loader
    try:
        mask = selector.select_mask(
            n=n,
            k=int(k),
            dataset=dataset,
            targets=targets,
            num_classes=context.num_classes,
            seed=seed,
            model=model,
            model_kwargs={"num_classes": context.num_classes},
            train_loader=loader,
            loader=loader,
        )
        return binarize_mask(mask)
    except Exception as exc:  # pragma: no cover - fall through to uniform
        LOGGER.warning("baseline %s failed (%s); using Uniform", name, exc)
        return indices_to_mask(uniform_indices(n, int(k), seed=resolve_seed(seed, 0)), n)


# --------------------------------------------------------------------------- #
# Target-model training / evaluation
# --------------------------------------------------------------------------- #
def evaluate_accuracy(model: Any, loader: Any, device: Optional[str] = None) -> float:
    """Top-1 accuracy (%) of ``model`` on ``loader`` (clean test split)."""
    try:
        from lbcs_repro.models.resnet18 import evaluate as _evaluate

        return float(_evaluate(model, loader, device=device))
    except Exception:  # pragma: no cover
        pass
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to evaluate accuracy")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0], batch[1]
            x = x.to(dev)
            y = y.to(dev)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    return 100.0 * correct / max(total, 1)


def train_target_model(
    model: Any,
    train_loader: Any,
    test_loader: Optional[Any] = None,
    epochs: int = 100,
    lr: float = 0.001,
    optimizer: str = "adam",
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    device: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[Any, List[float]]:
    """Train the post-selection target model on a constructed coreset.

    F-MNIST recipe (Section 5.2 / Appendix D.2): LeNet, Adam, lr 0.001, 100 epochs.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for target-model training")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev)
    if str(optimizer).lower() == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif str(optimizer).lower() == "sgd":
        opt = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    else:  # pragma: no cover
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    history: List[float] = []
    for epoch in range(int(epochs)):
        model.train()
        for batch in train_loader:
            x, y = batch[0].to(dev), batch[1].to(dev)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
        acc = evaluate_accuracy(model, test_loader, device=str(dev)) if test_loader is not None else float("nan")
        history.append(float(acc))
        if verbose:
            LOGGER.info("target epoch %d acc=%.2f", epoch + 1, acc)
    return model, history


def train_and_evaluate(
    context: RobustnessContext,
    mask: Any,
    seed: Optional[int] = None,
    config: Optional[RobustnessConfig] = None,
    model: Any = None,
    train_fn: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Train a target model on the coreset and report its clean test accuracy."""
    config = config or RobustnessConfig()
    size = mask_size(mask)
    if size == 0:
        return {"accuracy": float("nan"), "coreset_size": 0, "accuracy_per_point": float("nan"), "model": None, "history": []}
    loader = coreset_loader(context, mask, batch_size=config.batch_size, seed=seed)
    set_seed(seed)
    if model is None:
        try:
            model = context.target_factory(num_classes=context.num_classes)
        except TypeError:  # pragma: no cover
            model = context.target_factory()
    if train_fn is not None:
        out = train_fn(model, loader, context.test_loader, seed)
        acc, history = out if isinstance(out, tuple) else (out, [])
    else:
        rec = TARGET_CONFIGS.get(context.dataset, TARGET_CONFIGS["F-MNIST"])
        model, history = train_target_model(
            model,
            loader,
            context.test_loader,
            epochs=int(config.target_epochs or rec["epochs"]),
            lr=float(config.target_lr or rec["lr"]),
            optimizer=str(config.target_optimizer or rec["optimizer"]),
            weight_decay=float(rec.get("weight_decay", 0.0)),
            device=config.device,
        )
        acc = evaluate_accuracy(model, context.test_loader, device=config.device)
    acc = float(acc)
    return {
        "accuracy": acc,
        "coreset_size": int(size),
        "accuracy_per_point": float(acc) / max(int(size), 1),
        "model": model,
        "history": [float(h) for h in history],
    }


# --------------------------------------------------------------------------- #
# Single cell (all methods once)
# --------------------------------------------------------------------------- #
def run_single_cell(
    condition: str,
    k: int,
    repeat: int,
    config: Optional[RobustnessConfig] = None,
    context: Optional[RobustnessContext] = None,
    seed: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    select_baseline_fn: Optional[Callable[..., np.ndarray]] = None,
    select_lbcs_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    train_eval_fn: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run every method once for one ``(condition, k, repeat)`` cell."""
    config = config or RobustnessConfig()
    logger = logger or LOGGER
    condition = canonical_condition(condition)
    cell_seed = seed if seed is not None else resolve_seed(config.seed, repeat)
    context = context or build_robustness_context(condition, config, repeat=repeat)

    record: Dict[str, Any] = {
        "condition": condition,
        "dataset": context.dataset,
        "k": int(k),
        "repeat": int(repeat),
        "seed": int(cell_seed),
        "n": int(context.n),
        "methods": {},
    }

    for name in config.methods:
        try:
            if select_baseline_fn is not None:
                mask = select_baseline_fn(name, context, k, config, cell_seed)
            else:
                mask = select_baseline_mask(name, context, k, config, seed=cell_seed)
            if train_eval_fn is not None:
                res = train_eval_fn(context, mask, cell_seed, config)
            else:
                res = train_and_evaluate(context, mask, seed=cell_seed, config=config)
            record["methods"][str(name)] = {
                "accuracy": float(res.get("accuracy", np.nan)),
                "coreset_size": int(res.get("coreset_size", mask_size(mask))),
                "accuracy_per_point": float(res.get("accuracy_per_point", np.nan)),
                "status": "ok",
            }
        except Exception as exc:  # keep long sweeps alive
            logger.warning("method %s failed on %s k=%s: %s", name, condition, k, exc)
            record["methods"][str(name)] = {
                "accuracy": float("nan"),
                "coreset_size": 0,
                "accuracy_per_point": float("nan"),
                "status": f"failed:{type(exc).__name__}",
            }

    # LBCS (Algorithm 1) ---------------------------------------------------- #
    try:
        if select_lbcs_fn is not None:
            lbcs_out = select_lbcs_fn(context, k, config, cell_seed)
        else:
            lbcs_out = select_lbcs_mask(context, k, config, seed=cell_seed)
        if train_eval_fn is not None:
            res = train_eval_fn(context, lbcs_out["mask"], cell_seed, config)
        else:
            res = train_and_evaluate(context, lbcs_out["mask"], seed=cell_seed, config=config)
        record["methods"][LBCS_LABEL] = {
            "accuracy": float(res.get("accuracy", np.nan)),
            "coreset_size": int(lbcs_out.get("coreset_size", res.get("coreset_size", 0))),
            "accuracy_per_point": float(res.get("accuracy_per_point", np.nan)),
            "f1": float(lbcs_out.get("f1", np.nan)),
            "f2": float(lbcs_out.get("f2", np.nan)),
            "wall_time": float(lbcs_out.get("wall_time", 0.0)),
            "status": "ok",
        }
    except Exception as exc:
        logger.warning("LBCS failed on %s k=%s: %s", condition, k, exc)
        record["methods"][LBCS_LABEL] = {
            "accuracy": float("nan"),
            "coreset_size": 0,
            "accuracy_per_point": float("nan"),
            "status": f"failed:{type(exc).__name__}",
        }

    return record


# --------------------------------------------------------------------------- #
# Aggregation / formatting
# --------------------------------------------------------------------------- #
def aggregate_results(records: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, int], RobustnessCell]:
    """Group per-repeat records into mean/std cells keyed by ``(condition, k)``."""
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for rec in records:
        grouped.setdefault((str(rec["condition"]), int(rec["k"])), []).append(rec)

    cells: Dict[Tuple[str, int], RobustnessCell] = {}
    for (condition, k), recs in grouped.items():
        cell = RobustnessCell(condition=condition, k=k, repeats=len(recs))
        names = set()
        for rec in recs:
            names.update(rec.get("methods", {}).keys())
        for name in names:
            accs, sizes, per_point = [], [], []
            for rec in recs:
                info = rec.get("methods", {}).get(name)
                if not info:
                    continue
                if info.get("status", "ok") != "ok" or not np.isfinite(info.get("accuracy", np.nan)):
                    cell.failures += 1
                    continue
                accs.append(float(info["accuracy"]))
                sizes.append(float(info.get("coreset_size", np.nan)))
                per_point.append(float(info.get("accuracy_per_point", np.nan)))
            if accs:
                cell.accuracy[name] = (float(np.mean(accs)), float(np.std(accs)))
                cell.coreset_size[name] = (float(np.nanmean(sizes)), float(np.nanstd(sizes)))
                cell.accuracy_per_point[name] = (
                    float(np.nanmean(per_point)),
                    float(np.nanstd(per_point)),
                )
        cells[(condition, k)] = cell
    return cells


def best_method(cell: RobustnessCell, methods: Optional[Sequence[str]] = None) -> str:
    """Method with the highest mean accuracy in a cell."""
    candidates = list(methods or cell.accuracy.keys())
    scored = [(cell.mean(m), m) for m in candidates if m in cell.accuracy and np.isfinite(cell.mean(m))]
    if not scored:
        return ""
    return max(scored)[1]


def format_figure2_table(
    cells: Dict[Tuple[str, int], RobustnessCell],
    config: Optional[RobustnessConfig] = None,
) -> str:
    """Render the Figure 2 (a)/(b) accuracy tables."""
    config = config or RobustnessConfig()
    conditions = [c for c in config.conditions if any(key[0] == c for key in cells)]
    lines: List[str] = []
    for condition in conditions:
        methods = list(config.methods) + [LBCS_LABEL]
        header = "Method".ljust(16) + "".join(f"| k={k:<12}" for k in config.ks)
        lines.append(f"--- Figure 2: {CONDITION_TITLES.get(condition, condition)} ---")
        lines.append(header)
        for name in methods:
            row = str(name).ljust(16)
            for k in config.ks:
                cell = cells.get((condition, int(k)))
                if cell is None or name not in cell.accuracy:
                    row += "| " + "n/a".ljust(13)
                else:
                    mean, std = cell.accuracy[name]
                    row += "| " + f"{mean:.1f} +- {std:.1f}".ljust(13)
            lines.append(row)
        lines.append("")
    return "\n".join(lines)


def format_table8(
    cells: Dict[Tuple[str, int], RobustnessCell],
    config: Optional[RobustnessConfig] = None,
) -> str:
    """Render Appendix E.3 Table 8 (optimized coreset sizes + paper reference)."""
    config = config or RobustnessConfig()
    lines = ["Table 8: Mean and standard deviation of optimized coreset sizes by our method", ""]
    header = "Imperfect supervision".ljust(30) + "".join(f"| k={k:<16}" for k in config.ks)
    lines.append(header)
    for condition in [c for c in config.conditions if any(key[0] == c for key in cells)]:
        row = CONDITION_TITLES.get(condition, condition).ljust(30)
        for k in config.ks:
            cell = cells.get((condition, int(k)))
            if cell is None or LBCS_LABEL not in cell.coreset_size:
                row += "| " + "n/a".ljust(17)
            else:
                mean, std = cell.coreset_size[LBCS_LABEL]
                row += "| " + f"{mean:.1f} +- {std:.1f}".ljust(17)
        lines.append(row)
    lines.append("")
    lines.append("Paper reference (Appendix E.3, Table 8):")
    for condition in [c for c in config.conditions if c in PAPER_TABLE8_SIZES]:
        row = CONDITION_TITLES.get(condition, condition).ljust(30)
        for k in config.ks:
            ref = PAPER_TABLE8_SIZES[condition].get(int(k))
            row += "| " + ("n/a" if ref is None else f"{ref[0]:.1f} +- {ref[1]:.1f}").ljust(17)
        lines.append(row)
    return "\n".join(lines)


def direction_checks(cells: Dict[Tuple[str, int], RobustnessCell]) -> Dict[str, Any]:
    """Qualitative Section 5.3 checks (LBCS superiority + size reduction)."""
    checks: Dict[str, Any] = {"lbcs_best": {}, "size_reduced": {}, "summary": {}}
    conditions = sorted({key[0] for key in cells})
    wins = 0
    total = 0
    for condition in conditions:
        for (cond, k), cell in sorted(cells.items()):
            if cond != condition:
                continue
            if LBCS_LABEL not in cell.accuracy:
                continue
            best = best_method(cell)
            lbcs_best = best == LBCS_LABEL
            checks["lbcs_best"][f"{condition}_k{k}"] = {
                "best_method": best,
                "lbcs_accuracy": cell.mean(LBCS_LABEL),
                "lbcs_is_best": bool(lbcs_best),
            }
            size_ok = cell.size_mean(LBCS_LABEL) <= 0.995 * float(k)
            checks["size_reduced"][f"{condition}_k{k}"] = {
                "predefined_k": int(k),
                "optimized_size": cell.size_mean(LBCS_LABEL),
                "reduced": bool(size_ok),
            }
            total += 1
            wins += 1 if lbcs_best else 0
    checks["summary"] = {
        "num_cells": total,
        "lbcs_wins": int(wins),
        "lbcs_win_rate": float(wins) / total if total else float("nan"),
        "paper_claim": "LBCS should outperform baselines under imperfect supervision",
    }
    return checks


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_figure2(
    cells: Dict[Tuple[str, int], RobustnessCell],
    out_path: str,
    config: Optional[RobustnessConfig] = None,
    dpi: int = 150,
) -> Optional[str]:
    """Plot Figure 2 panels: (a) 30% (and 50%) label noise, (b) class imbalance."""
    config = config or RobustnessConfig()
    if not _MPL_AVAILABLE:
        LOGGER.warning("matplotlib unavailable; skipping Figure 2 plot")
        return None
    panels = [c for c in config.conditions if any(key[0] == c for key in cells)]
    if not panels:
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 4.0), squeeze=False)
    methods = list(config.methods) + [LBCS_LABEL]
    ks = [int(k) for k in config.ks]
    for ax, condition in zip(axes[0], panels):
        for name in methods:
            ys = []
            for k in ks:
                cell = cells.get((condition, k))
                ys.append(cell.mean(name) if cell is not None and name in cell.accuracy else np.nan)
            style = dict(marker="o", linewidth=2.0)
            if name == LBCS_LABEL:
                style.update(color="black", linestyle="--", zorder=5)
            ax.plot(ks, ys, label=str(name), **style)
        ax.set_title(CONDITION_TITLES.get(condition, condition))
        ax.set_xlabel("coreset size k")
        ax.set_ylabel("test accuracy (%)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Figure 2: coreset selection under imperfect supervision")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    if _MPL_AVAILABLE:
        plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Top-level driver
# --------------------------------------------------------------------------- #
def run_figure2(
    config: Optional[RobustnessConfig] = None,
    logger: Optional[logging.Logger] = None,
    cell_runner: Optional[Callable[..., Dict[str, Any]]] = None,
    context_builder: Optional[Callable[..., RobustnessContext]] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Run the full Section 5.3 study and return tables, checks and artifacts."""
    config = (config or RobustnessConfig()).with_overrides(**overrides)
    logger = logger or LOGGER
    logger.info(
        "Figure 2 / Table 8: dataset=%s conditions=%s ks=%s T=%d eps=%.2f repeats=%d",
        config.dataset,
        list(config.conditions),
        list(config.ks),
        config.T,
        config.epsilon,
        config.repeats,
    )

    records: List[Dict[str, Any]] = []
    for condition in config.conditions:
        try:
            context = (
                context_builder(condition, config)
                if context_builder is not None
                else build_robustness_context(condition, config, repeat=0)
            )
        except Exception as exc:
            logger.warning("cannot build context for %s: %s", condition, exc)
            continue
        for k in config.ks:
            for repeat in range(int(config.repeats)):
                cell_seed = resolve_seed(config.seed, repeat)
                if cell_runner is not None:
                    rec = cell_runner(condition, k, repeat, config, context, cell_seed)
                else:
                    rec = run_single_cell(
                        condition,
                        k,
                        repeat,
                        config=config,
                        context=context,
                        seed=cell_seed,
                        logger=logger,
                    )
                records.append(rec)
            logger.info("%s k=%d done", condition, k)

    cells = aggregate_results(records)
    checks = direction_checks(cells)
    table8_text = format_table8(cells, config)
    figure2_text = format_figure2_table(cells, config)
    artifacts: Dict[str, str] = {}

    if config.save_artifacts and config.output_dir:
        artifacts = save_results(cells, records, checks, config, table8_text, figure2_text)
    if config.plot and config.save_artifacts and config.output_dir:
        path = plot_figure2(cells, os.path.join(config.output_dir, "figure2.png"), config)
        if path:
            artifacts["figure2_png"] = path

    return {
        "cells": cells,
        "records": records,
        "checks": checks,
        "figure2": figure2_text,
        "table8": table8_text,
        "config": config.to_dict(),
        "artifacts": artifacts,
    }


def save_results(
    cells: Dict[Tuple[str, int], RobustnessCell],
    records: Sequence[Dict[str, Any]],
    checks: Dict[str, Any],
    config: RobustnessConfig,
    table8_text: str,
    figure2_text: str,
) -> Dict[str, str]:
    """Persist Figure 2 / Table 8 artifacts (json, csv, txt, jsonl)."""
    out_dir = config.output_dir or DEFAULT_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    table8_path = os.path.join(out_dir, "table8_sizes.txt")
    with open(table8_path, "w", encoding="utf-8") as fh:
        fh.write(table8_text + "\n")
    paths["table8_txt"] = table8_path

    figure2_path = os.path.join(out_dir, "figure2_accuracies.txt")
    with open(figure2_path, "w", encoding="utf-8") as fh:
        fh.write(figure2_text + "\n")
    paths["figure2_txt"] = figure2_path

    json_path = os.path.join(out_dir, "figure2_table8.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "cells": {f"{cond}|{k}": cell.to_dict() for (cond, k), cell in cells.items()},
                "checks": checks,
                "config": config.to_dict(),
            },
            fh,
            indent=2,
            default=str,
        )
    paths["json"] = json_path

    csv_path = os.path.join(out_dir, "figure2_table8.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["condition", "k", "method", "accuracy_mean", "accuracy_std", "size_mean", "size_std"])
        for (condition, k), cell in sorted(cells.items()):
            for name in list(cell.accuracy.keys()):
                acc = cell.accuracy.get(name, (np.nan, np.nan))
                size = cell.coreset_size.get(name, (np.nan, np.nan))
                writer.writerow([condition, k, name, acc[0], acc[1], size[0], size[1]])
    paths["csv"] = csv_path

    raw_path = os.path.join(out_dir, "figure2_raw.jsonl")
    with open(raw_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, default=str) + "\n")
    paths["raw_jsonl"] = raw_path

    checks_path = os.path.join(out_dir, "figure2_checks.json")
    with open(checks_path, "w", encoding="utf-8") as fh:
        json.dump(checks, fh, indent=2, default=str)
    paths["checks"] = checks_path

    return paths


#: Aliases expected by the experiment registry / CLI.
run = run_figure2
run_robustness = run_figure2
figure2_robustness = run_figure2
run_table8 = run_figure2


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Figure 2 / Table 8: coreset selection under imperfect supervision (Section 5.3)"
    )
    parser.add_argument("--paper", action="store_true", help="use the exact Section 5.3 protocol")
    parser.add_argument("--smoke", action="store_true", help="tiny offline configuration")
    parser.add_argument("--selftest", action="store_true", help="run offline checks and exit")
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--ks", nargs="+", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--inner-epochs", type=int, default=None)
    parser.add_argument("--target-epochs", type=int, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_argparser().parse_args(list(argv) if argv is not None else None)

    if args.selftest:
        report = _selftest(verbose=True)
        return 0 if report.get("ok") else 1

    config = RobustnessConfig.paper() if args.paper else RobustnessConfig()
    if args.smoke:
        config = RobustnessConfig.smoke()
    if args.config:
        try:
            import yaml

            with open(args.config, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            config = config.with_overrides(**data)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("could not load config %s: %s", args.config, exc)
    if args.conditions:
        config.conditions = tuple(args.conditions)
    if args.ks:
        config.ks = tuple(args.ks)
    if args.repeats is not None:
        config.repeats = args.repeats
    if args.T is not None:
        config.T = args.T
    if args.epsilon is not None:
        config.epsilon = args.epsilon
    if args.inner_epochs is not None:
        config.inner_epochs = args.inner_epochs
    if args.target_epochs is not None:
        config.target_epochs = args.target_epochs
    if args.dataset:
        config.dataset = args.dataset
    if args.device:
        config.device = args.device
    if args.seed is not None:
        config.seed = args.seed
    if args.data_root:
        config.data_root = args.data_root
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.no_plot:
        config.plot = False

    result = run_figure2(config)
    print(result["figure2"])
    print(result["table8"])
    print("checks:", json.dumps(result["checks"]["summary"], indent=2, default=str))
    return 0


# --------------------------------------------------------------------------- #
# Offline self-test
# --------------------------------------------------------------------------- #
def _selftest(verbose: bool = True) -> Dict[str, Any]:
    """Offline checks: masks, aggregation, formatting, checks, driver dry-run."""
    report: Dict[str, Any] = {"ok": True, "notes": []}

    # masks / projection ---------------------------------------------------- #
    mask = np.array([-1.0, -0.5, -1e-12, 0.0, 1e-12, 0.5, 1.0])
    proj = binarize_mask(mask)
    assert proj.tolist() == [0, 0, 0, 1, 1, 1, 1], proj.tolist()
    assert mask_size(mask) == 4
    assert mask_indices(mask).tolist() == [3, 4, 5, 6]
    report["projection"] = True

    # condition normalisation ----------------------------------------------- #
    assert canonical_condition("30%") == "noise30"
    assert canonical_condition("noise_50") == "noise50"
    assert canonical_condition("Imbalanced") == "imbalance"
    assert noise_rate_for("noise30") == 0.30 and noise_rate_for("noise50") == 0.50
    assert noise_rate_for("imbalance") == 0.0
    report["conditions"] = True

    # synthetic records -> aggregation -------------------------------------- #
    rng = np.random.default_rng(0)
    records: List[Dict[str, Any]] = []
    for condition in ("noise30", "noise50", "imbalance"):
        for k in (1000, 2000):
            for repeat in range(3):
                methods = {}
                for name in METHOD_ORDER:
                    methods[name] = {
                        "accuracy": float(60 + rng.normal(0, 0.5)),
                        "coreset_size": int(k),
                        "accuracy_per_point": 0.06,
                        "status": "ok",
                    }
                methods[LBCS_LABEL] = {
                    "accuracy": float(70 + rng.normal(0, 0.5)),
                    "coreset_size": int(0.95 * k),
                    "accuracy_per_point": 0.07,
                    "f1": 0.5,
                    "f2": 0.95 * k,
                    "wall_time": 1.0,
                    "status": "ok",
                }
                records.append(
                    {
                        "condition": condition,
                        "dataset": "F-MNIST",
                        "k": k,
                        "repeat": repeat,
                        "seed": repeat,
                        "n": 60000,
                        "methods": methods,
                    }
                )
    cells = aggregate_results(records)
    assert len(cells) == 6, len(cells)
    cell = cells[("noise30", 1000)]
    assert cell.repeats == 3
    assert abs(cell.mean(LBCS_LABEL) - cell.mean("Uniform")) > 5.0
    assert cell.size_mean(LBCS_LABEL) < 1000.0
    report["aggregation"] = True

    cfg = RobustnessConfig(conditions=("noise30", "noise50", "imbalance"), ks=(1000, 2000), repeats=3)
    text8 = format_table8(cells, cfg)
    assert "Table 8" in text8 and "951.2" in text8
    text2 = format_figure2_table(cells, cfg)
    assert "Figure 2" in text2 and LBCS_LABEL in text2
    report["formatting"] = True

    checks = direction_checks(cells)
    assert checks["summary"]["num_cells"] == 6
    assert checks["summary"]["lbcs_wins"] == 6, checks["summary"]
    assert all(entry["reduced"] for entry in checks["size_reduced"].values())
    report["direction_checks"] = True

    # config round-trip ------------------------------------------------------ #
    paper = RobustnessConfig.paper()
    assert tuple(paper.ks) == tuple(PAPER_KS) and paper.epsilon == 0.2 and paper.T == 500
    assert paper.repeats == 10
    assert RobustnessConfig.from_dict(paper.to_dict()).ks == tuple(PAPER_KS)
    report["config"] = True

    # driver dry-run with injected cell runner ------------------------------- #
    def fake_cell_runner(condition, k, repeat, config, context, seed):
        methods = {
            name: {"accuracy": 60.0, "coreset_size": int(k), "accuracy_per_point": 0.06, "status": "ok"}
            for name in METHOD_ORDER
        }
        methods[LBCS_LABEL] = {
            "accuracy": 72.0,
            "coreset_size": int(0.96 * k),
            "accuracy_per_point": 0.07,
            "status": "ok",
        }
        return {
            "condition": condition,
            "dataset": "F-MNIST",
            "k": int(k),
            "repeat": int(repeat),
            "seed": int(seed),
            "n": 60000,
            "methods": methods,
        }

    dry = run_figure2(
        RobustnessConfig(
            conditions=("noise30",),
            ks=(300,),
            repeats=1,
            save_artifacts=False,
            plot=False,
        ),
        cell_runner=fake_cell_runner,
    )
    assert dry["checks"]["summary"]["lbcs_wins"] == 1
    assert "figure2" in dry and "table8" in dry
    report["dry_run"] = True

    if _TORCH_AVAILABLE:
        try:
            from lbcs_repro.models.lenet import lenet_factory

            model = lenet_factory(num_classes=10)()
            assert model is not None
            report["model_factory"] = True
        except Exception as exc:  # pragma: no cover
            report["notes"].append(f"model factory check skipped: {exc}")
    else:
        report["notes"].append("torch unavailable: model checks skipped")

    report["notes"].append("out of scope: ImageNet-1k (5.4), continual learning (E.5), streaming (E.6)")

    if verbose:
        print("Figure 2 / Table 8 self-test")
        for key, value in report.items():
            if key == "notes":
                continue
            print(f"  {key}: {value}")
        for note in report["notes"]:
            print(f"  note: {note}")
    return report


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
