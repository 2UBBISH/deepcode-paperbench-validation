"""Table 6 experiment driver: cross network architecture evaluation (Section 6).

Section 6 of the paper ("Cross network architecture evaluation") states:

    "Here we demonstrate that the proposed method is not limited to specific network
     architectures. We employ SVHN and use ViTsmall (Dosovitskiy et al., 2021) and
     WideResNet (abbreviated as W-NET) (Zagoruyko & Komodakis, 2016) for training on
     the constructed coreset. The other experimental settings are not changed.
     Results are provided in Table 6. As can be seen, with ViT, our method is still
     superior to the competitors with respect to test accuracy and coreset sizes
     (the exact coreset sizes of our method can be checked in Table 2). With W-NET,
     our LBCS gets the best test accuracy when k = 1000, k = 3000, and k = 4000 with
     smaller coreset sizes. In other cases, i.e., k = 2000, LBCS can achieve
     competitive test accuracy compared with baselines but with a smaller coreset size."

So this driver:
  * uses the SVHN benchmark (same inner-loop coreset-selection settings as Section 5.2,
    i.e. Adam lr 0.001 inner loop, epsilon = 0.2, T = 500, ten repeats);
  * keeps *everything* unchanged from the Section 5.2 comparison (same candidate pool,
    same baselines, same predefined coreset sizes k in {1000, 2000, 3000, 4000}), and only
    swaps the post-selection *target* network for ViT-small and WideResNet (W-NET);
  * reports mean +- std test accuracy (%) per (architecture, k) cell;
  * reuses the LBCS coreset sizes measured in Table 2 (SVHN), because the paper says the
    exact coreset sizes obtained by LBCS "can be checked in Table 2".

Everything that the paper does not numerically specify (architecture widths for ViT-small /
W-NET, batch size, weight decay, ...) is labelled ``SUGGESTED_*`` and lives in the config
dataclass so it can be overridden from ``configs/section6.yaml`` without touching driver code.

Out of scope for this reproduction: ImageNet-1k (Section 5.4), continual learning
(Appendix E.5) and streaming (Appendix E.6). None of those are referenced here.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import torch
    import torch.nn as nn  # noqa: F401
    from torch.utils.data import DataLoader, Subset

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch-free environments
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    Subset = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


# --------------------------------------------------------------------------------------
# Paper-stated constants (Section 6 / Section 5.2 / Table 2 / Table 6)
# --------------------------------------------------------------------------------------

#: The Section 6 cross-architecture study is conducted on SVHN.
PAPER_DATASET = "SVHN"

#: Predefined coreset sizes used throughout Section 5.2 and re-used in Section 6.
PAPER_KS: Tuple[int, ...] = (1000, 2000, 3000, 4000)

#: The two target architectures of Table 6, in the table's row-block order.
PAPER_ARCHITECTURES: Tuple[str, ...] = ("ViTSmall", "WideResNet")

#: "The parameters epsilon and T are set to 0.2 and 500." (Section 5.2, unchanged in §6)
PAPER_EPSILON = 0.2
PAPER_T = 500

#: "All experiments are repeated ten times ... with PyTorch." (Section 5.2)
PAPER_REPEATS = 10

#: "An Adam optimizer is used with a learning rate of 0.001 for the inner loop."
PAPER_INNER_OPTIMIZER = "adam"
PAPER_INNER_LR = 0.001

#: "for F-MNIST and SVHN, an Adam optimizer is used with a learning rate of 0.001 and
#:  100 epochs." (post-selection target training; SVHN branch)
PAPER_TARGET_OPTIMIZER = "adam"
PAPER_TARGET_LR = 0.001
PAPER_TARGET_EPOCHS = 100

#: Comparison methods of Section 5.2 ("The other experimental settings are not changed").
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
SIZE_LABEL = "Coreset size (ours)"

#: LBCS coreset sizes on SVHN, taken from Table 2 (mean +- std), k = 1000 .. 4000.
PAPER_TABLE2_SVHN_SIZES: Dict[int, Tuple[float, float]] = {
    1000: (970.0, 4.8),
    2000: (1902.3, 10.3),
    3000: (2712.6, 15.0),
    4000: (3804.2, 16.4),
}

#: Table 6 of the paper: architecture -> k -> method -> (mean, std) test accuracy (%).
PAPER_TABLE6: Dict[str, Dict[int, Dict[str, Tuple[float, float]]]] = {
    "ViTSmall": {
        1000: {
            "Uniform": (28.5, 3.1),
            "EL2N": (22.7, 3.5),
            "GraNd": (24.0, 2.2),
            "Influential": (31.5, 1.8),
            "Moderate": (32.8, 1.5),
            "CCS": (31.7, 1.6),
            "Probabilistic": (29.6, 0.3),
            LBCS_LABEL: (33.9, 0.8),
        },
        2000: {
            "Uniform": (46.6, 2.7),
            "EL2N": (40.9, 2.6),
            "GraNd": (38.8, 0.6),
            "Influential": (42.2, 1.7),
            "Moderate": (45.5, 2.3),
            "CCS": (46.1, 1.8),
            "Probabilistic": (46.6, 2.0),
            LBCS_LABEL: (47.5, 2.2),
        },
        3000: {
            "Uniform": (50.0, 2.2),
            "EL2N": (46.7, 3.0),
            "GraNd": (47.9, 2.4),
            "Influential": (50.8, 0.7),
            "Moderate": (51.0, 2.9),
            "CCS": (50.4, 1.6),
            "Probabilistic": (50.5, 1.9),
            LBCS_LABEL: (51.3, 0.6),
        },
        4000: {
            "Uniform": (54.0, 3.3),
            "EL2N": (49.9, 2.8),
            "GraNd": (50.8, 0.9),
            "Influential": (53.3, 0.9),
            "Moderate": (54.9, 1.9),
            "CCS": (56.2, 2.1),
            "Probabilistic": (55.3, 1.5),
            LBCS_LABEL: (57.7, 0.4),
        },
    },
    "WideResNet": {
        1000: {
            "Uniform": (78.8, 1.5),
            "EL2N": (67.9, 2.7),
            "GraNd": (70.5, 3.0),
            "Influential": (79.3, 2.8),
            "Moderate": (80.0, 0.4),
            "CCS": (79.8, 0.9),
            "Probabilistic": (80.1, 1.3),
            LBCS_LABEL: (80.3, 1.2),
        },
        2000: {
            "Uniform": (87.2, 1.2),
            "EL2N": (69.5, 3.3),
            "GraNd": (73.4, 2.6),
            "Influential": (87.1, 0.8),
            "Moderate": (88.0, 0.3),
            "CCS": (88.7, 0.6),
            "Probabilistic": (87.0, 1.0),
            LBCS_LABEL: (87.8, 1.1),
        },
        3000: {
            "Uniform": (89.1, 0.9),
            "EL2N": (76.6, 1.2),
            "GraNd": (78.8, 3.2),
            "Influential": (90.3, 0.7),
            "Moderate": (90.3, 0.4),
            "CCS": (90.2, 0.4),
            "Probabilistic": (89.3, 0.6),
            LBCS_LABEL: (90.7, 0.5),
        },
        4000: {
            "Uniform": (90.2, 1.9),
            "EL2N": (80.3, 1.9),
            "GraNd": (83.4, 1.7),
            "Influential": (90.9, 1.1),
            "Moderate": (90.8, 0.5),
            "CCS": (91.1, 1.0),
            "Probabilistic": (90.6, 0.5),
            LBCS_LABEL: (91.4, 0.9),
        },
    },
}

#: Architecture-name normalisation (configs may use paper spellings).
_ARCH_ALIASES: Dict[str, str] = {
    "vit": "ViTSmall",
    "vitsmall": "ViTSmall",
    "vit-small": "ViTSmall",
    "vit_small": "ViTSmall",
    "svhnvits": "ViTSmall",
    "wnet": "WideResNet",
    "w-net": "WideResNet",
    "wideresnet": "WideResNet",
    "wide_resnet": "WideResNet",
    "wrn": "WideResNet",
}

# --------------------------------------------------------------------------------------
# SUGGESTED defaults (NOT stated in the paper; centralised so configs can override)
# --------------------------------------------------------------------------------------

SUGGESTED_INNER_EPOCHS = 100
SUGGESTED_BATCH_SIZE = 128
SUGGESTED_EVAL_BATCH_SIZE = 256
SUGGESTED_WEIGHT_DECAY = 5e-4
SUGGESTED_DELTA_INIT = 0.1
SUGGESTED_DELTA_LOWER = 1e-3
SUGGESTED_NUM_WORKERS = 0
SUGGESTED_EVAL_SPLIT = "train"
DEFAULT_OUTPUT_DIR = os.path.join("results", "table6")


# --------------------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------------------


def set_seed(seed: Optional[int]) -> None:
    """Seed NumPy (and PyTorch when available) for reproducible repeats."""
    if seed is None:
        return
    np.random.seed(int(seed) % (2 ** 32))
    try:
        import random

        random.seed(int(seed))
    except Exception:  # pragma: no cover
        pass
    if _TORCH_AVAILABLE:
        try:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
        except Exception:  # pragma: no cover
            pass


def resolve_seed(seed: Optional[int], repeat: int = 0, base: int = 0) -> int:
    """Deterministic per-repeat seed (mirrors ``utils.seed.resolve_seed``)."""
    base_seed = base if seed is None else int(seed)
    if repeat == 0:
        return base_seed
    return int((base_seed + repeat * 7919) % (2 ** 31 - 1))


def binarize_mask(mask: Any) -> np.ndarray:
    """Project a mask to {0,1} using the Appendix A clip/projection rule.

    Values below -1 become -1, values above 1 become 1, [-1, 0) -> 0, [0, 1] -> 1.
    """
    arr = np.asarray(mask, dtype=np.float64).ravel()
    if arr.size == 0:
        return arr.astype(np.float32)
    arr = np.clip(arr, -1.0, 1.0)
    return (arr >= 0.0).astype(np.float32)


def mask_indices(mask: Any) -> np.ndarray:
    """Indices selected by ``mask`` (binary/relaxed/probability masks accepted)."""
    return np.flatnonzero(binarize_mask(mask)).astype(np.int64)


def mask_size(mask: Any) -> int:
    """``f_2(m) = ||m||_0`` computed on the discretised mask."""
    return int(mask_indices(mask).size)


def canonical_dataset(name: Optional[str]) -> str:
    """Normalise dataset aliases to the paper spelling."""
    if not name:
        return PAPER_DATASET
    text = str(name).strip().lower().replace("_", "-")
    if text in {"svhn", "street view house numbers"}:
        return "SVHN"
    if text in {"f-mnist", "fmnist", "fashion-mnist", "fashionmnist"}:
        return "F-MNIST"
    if text in {"cifar-10", "cifar10"}:
        return "CIFAR-10"
    if text in {"mnist-s", "mnists", "mnist"}:
        return "MNIST-S"
    return str(name)


def canonical_architecture(name: Optional[str]) -> str:
    """Normalise architecture names to ``ViTSmall`` / ``WideResNet``."""
    if not name:
        return PAPER_ARCHITECTURES[0]
    key = str(name).strip().lower().replace(" ", "")
    return _ARCH_ALIASES.get(key, _ARCH_ALIASES.get(key.replace("-", ""), str(name)))


def mean_std(values: Sequence[float], ddof: int = 1) -> Tuple[float, float]:
    """Mean and sample standard deviation, ignoring non-finite values."""
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 0.0
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=ddof))


def format_mean_std(mean: float, std: float, decimals: int = 1) -> str:
    """Render ``mean +- std`` the way the paper's tables do."""
    return f"{mean:.{decimals}f} +- {std:.{decimals}f}"


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


@dataclass
class Table6Config:
    """Protocol for the Section 6 cross-architecture study (Table 6)."""

    dataset: str = PAPER_DATASET
    ks: Tuple[int, ...] = PAPER_KS
    architectures: Tuple[str, ...] = PAPER_ARCHITECTURES
    methods: Tuple[str, ...] = METHOD_ORDER
    epsilon: float = PAPER_EPSILON
    T: int = PAPER_T
    repeats: int = PAPER_REPEATS

    # Inner loop (coreset selection) -- Section 5.2 settings, unchanged in Section 6.
    inner_optimizer: str = PAPER_INNER_OPTIMIZER
    inner_lr: float = PAPER_INNER_LR
    inner_epochs: int = SUGGESTED_INNER_EPOCHS

    # Post-selection target training -- SVHN branch: Adam, lr 0.001, 100 epochs.
    target_optimizer: str = PAPER_TARGET_OPTIMIZER
    target_lr: float = PAPER_TARGET_LR
    target_epochs: int = PAPER_TARGET_EPOCHS
    weight_decay: float = SUGGESTED_WEIGHT_DECAY
    momentum: float = 0.9

    batch_size: int = SUGGESTED_BATCH_SIZE
    eval_batch_size: int = SUGGESTED_EVAL_BATCH_SIZE
    num_workers: int = SUGGESTED_NUM_WORKERS

    delta_init: float = SUGGESTED_DELTA_INIT
    delta_lower: float = SUGGESTED_DELTA_LOWER
    warm_start: bool = True
    group_size: int = 1

    device: Optional[str] = None
    seed: int = 0
    log_every: int = 0
    data_root: Optional[str] = None
    output_dir: str = DEFAULT_OUTPUT_DIR
    save_artifacts: bool = True
    plot: bool = True
    f1_eval_split: str = SUGGESTED_EVAL_SPLIT
    verbose: bool = False

    # -- constructors ------------------------------------------------------------------

    @classmethod
    def paper(cls, **overrides: Any) -> "Table6Config":
        """Paper-stated protocol: SVHN, k in {1000..4000}, eps 0.2, T 500, 10 repeats."""
        return cls(**overrides)

    @classmethod
    def smoke(cls, **overrides: Any) -> "Table6Config":
        """Tiny configuration for a fast end-to-end smoke run."""
        defaults: Dict[str, Any] = dict(
            ks=(32,),
            repeats=1,
            T=3,
            inner_epochs=1,
            target_epochs=1,
            batch_size=16,
            eval_batch_size=32,
            save_artifacts=False,
            plot=False,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Table6Config":
        """Build a config from a (nested) mapping, tolerating extra keys."""
        if not data:
            return cls()
        flat: Dict[str, Any] = {}
        for key, value in dict(data).items():
            if isinstance(value, dict) and key in {"table6", "cross_arch", "section6"}:
                flat.update(value)
            else:
                flat[key] = value
        fields = set(cls.__dataclass_fields__.keys())  # type: ignore[attr-defined]
        kwargs: Dict[str, Any] = {}
        for key, value in flat.items():
            if key not in fields:
                continue
            if key in {"ks", "architectures", "methods"} and isinstance(value, (list, tuple)):
                value = tuple(value)
            kwargs[key] = value
        return cls(**kwargs)

    def with_overrides(self, **overrides: Any) -> "Table6Config":
        """Return a copy with ``overrides`` applied for recognised fields."""
        fields = set(self.__dataclass_fields__.keys())  # type: ignore[attr-defined]
        data = {k: v for k, v in overrides.items() if k in fields and v is not None}
        for key in ("ks", "architectures", "methods"):
            if key in data and isinstance(data[key], (list, tuple)):
                data[key] = tuple(data[key])
        return Table6Config(**{**self.to_dict(), **data})

    def to_dict(self) -> Dict[str, Any]:
        """Serialisable view (tuples -> lists) for JSON artifacts."""
        out: Dict[str, Any] = {}
        for key, value in self.__dict__.items():
            out[key] = list(value) if isinstance(value, tuple) else value
        return out


@dataclass
class Table6Cell:
    """Aggregated (mean +- std) result for one (architecture, k) cell."""

    architecture: str
    k: int
    accuracy: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    coreset_size: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    f1: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    repeats: int = 0
    failures: int = 0
    raw: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def cell(self) -> Tuple[str, int]:
        return (self.architecture, int(self.k))

    def accuracy_mean(self, method: str) -> Optional[float]:
        entry = self.accuracy.get(method)
        return None if entry is None else float(entry[0])

    def size_mean(self, method: str = LBCS_LABEL) -> Optional[float]:
        entry = self.coreset_size.get(method)
        return None if entry is None else float(entry[0])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "architecture": self.architecture,
            "k": int(self.k),
            "accuracy": {m: [float(v[0]), float(v[1])] for m, v in self.accuracy.items()},
            "coreset_size": {m: [float(v[0]), float(v[1])] for m, v in self.coreset_size.items()},
            "f1": {m: [float(v[0]), float(v[1])] for m, v in self.f1.items()},
            "repeats": int(self.repeats),
            "failures": int(self.failures),
        }


# --------------------------------------------------------------------------------------
# Data / model plumbing (soft imports keep the module importable without torch)
# --------------------------------------------------------------------------------------


def _import_datasets():
    from lbcs_repro.data import datasets as datasets_mod

    return datasets_mod


def _model_factory_for(architecture: str, num_classes: int = 10) -> Callable[..., Any]:
    """Resolve a fresh-model factory for the requested cross-architecture target."""
    arch = canonical_architecture(architecture)
    if arch == "ViTSmall":
        from lbcs_repro.models.vit_small import vit_small_factory

        return vit_small_factory(num_classes=num_classes)
    if arch == "WideResNet":
        from lbcs_repro.models.wide_resnet import wide_resnet_factory

        return wide_resnet_factory(num_classes=num_classes)
    # Defensive fallback through the model registry.
    from lbcs_repro.models import model_factory as registry_factory

    return registry_factory(architecture)


def _inner_factory_for(dataset: str, num_classes: int = 10) -> Callable[..., Any]:
    """Inner-loop (coreset-selection) proxy: simple CNN for SVHN (Section 5.2)."""
    try:
        from lbcs_repro.models import default_model_for, model_factory

        return model_factory(default_model_for(dataset, "inner"))
    except Exception:  # pragma: no cover - fall back to the explicit import
        from lbcs_repro.models.svhn_cnn import svhn_cnn_factory

        return svhn_cnn_factory(num_classes=num_classes)


def build_context(
    architecture: str,
    dataset: Optional[str] = None,
    config: Optional[Table6Config] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Build datasets, loaders, model factories and metadata for one architecture."""
    cfg = config or Table6Config()
    ds_name = canonical_dataset(dataset or cfg.dataset)
    data_mod = _import_datasets()

    train_dataset = data_mod.get_dataset(
        ds_name, train=True, root=cfg.data_root, fallback_synthetic=not _TORCH_AVAILABLE
    )
    test_dataset = data_mod.get_dataset(
        ds_name, train=False, root=cfg.data_root, fallback_synthetic=not _TORCH_AVAILABLE
    )
    num_classes = int(data_mod.num_classes(ds_name))
    targets = np.asarray(data_mod.get_targets(train_dataset))
    n = int(targets.shape[0])

    test_loader = data_mod.make_loader(
        test_dataset, batch_size=cfg.eval_batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    eval_split = str(getattr(cfg, "f1_eval_split", "train")).lower()
    eval_dataset = train_dataset if eval_split.startswith("train") else test_dataset

    return {
        "dataset": ds_name,
        "architecture": canonical_architecture(architecture),
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "dataset_for_inner": data_mod.subset_dataset(train_dataset, np.arange(n)),
        "test_loader": test_loader,
        "eval_dataset": eval_dataset,
        "targets": targets,
        "n": n,
        "num_classes": num_classes,
        "inner_factory": _inner_factory_for(ds_name, num_classes),
        "target_factory": _model_factory_for(architecture, num_classes),
        "device": device or (cfg.device if cfg.device else "cpu"),
        "config": cfg,
    }


def coreset_loader(
    context: Dict[str, Any],
    mask: Any,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> Any:
    """DataLoader restricted to the examples selected by ``mask`` (post-selection training)."""
    cfg: Table6Config = context.get("config", Table6Config())
    data_mod = _import_datasets()
    indices = mask_indices(mask)
    subset = data_mod.subset_dataset(context["train_dataset"], indices)
    kwargs: Dict[str, Any] = dict(
        batch_size=int(batch_size or cfg.batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers if num_workers is not None else cfg.num_workers),
    )
    if _TORCH_AVAILABLE and shuffle and seed is not None:
        try:
            generator = torch.Generator()
            generator.manual_seed(int(seed))
            kwargs["generator"] = generator
        except Exception:  # pragma: no cover
            pass
    try:
        return data_mod.make_loader(subset, **kwargs)
    except Exception:  # pragma: no cover - minimal fallback
        return DataLoader(subset, **kwargs)


def evaluate_accuracy(model: Any, loader: Any, device: Optional[str] = None) -> float:
    """Top-1 accuracy (%) of ``model`` on ``loader``."""
    try:
        from lbcs_repro.models.wide_resnet import evaluate as _evaluate

        return float(_evaluate(model, loader, device=device))
    except Exception:
        pass
    if not _TORCH_AVAILABLE:  # pragma: no cover
        raise RuntimeError("evaluate_accuracy requires PyTorch")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0], batch[1]
            else:  # pragma: no cover - dict batches
                inputs, targets = batch["inputs"], batch["targets"]
            inputs = inputs.to(dev)
            targets = targets.to(dev)
            logits = model(inputs)
            if isinstance(logits, (tuple, list)):  # pragma: no cover
                logits = logits[0]
            preds = logits.argmax(dim=1)
            correct += int((preds == targets).sum().item())
            total += int(targets.numel())
    return 100.0 * correct / max(total, 1)


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
    """Train the post-selection target model on the constructed coreset.

    Section 5.2 (unchanged in Section 6) for SVHN: Adam, lr 0.001, 100 epochs.
    """
    if not _TORCH_AVAILABLE:  # pragma: no cover
        raise RuntimeError("train_target_model requires PyTorch")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev)
    criterion = torch.nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    opt_name = str(optimizer).lower()
    if opt_name == "sgd":
        opt = torch.optim.SGD(
            params, lr=float(lr), momentum=float(momentum), weight_decay=float(weight_decay)
        )
    elif opt_name in {"adamw", "adam_w"}:
        opt = torch.optim.AdamW(params, lr=float(lr), weight_decay=float(weight_decay))
    else:
        opt = torch.optim.Adam(params, lr=float(lr), weight_decay=float(weight_decay))

    history: List[float] = []
    for epoch in range(int(epochs)):
        model.train()
        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0], batch[1]
            else:  # pragma: no cover
                inputs, targets = batch["inputs"], batch["targets"]
            inputs = inputs.to(dev)
            targets = targets.to(dev)
            opt.zero_grad()
            logits = model(inputs)
            if isinstance(logits, (tuple, list)):  # pragma: no cover
                logits = logits[0]
            loss = criterion(logits, targets)
            loss.backward()
            opt.step()
        if test_loader is not None:
            acc = evaluate_accuracy(model, test_loader, device=str(dev))
            history.append(acc)
            if verbose and (log_every <= 0 or (epoch + 1) % max(log_every, 1) == 0):
                LOGGER.info("epoch %d/%d  eval accuracy %.2f", epoch + 1, epochs, acc)
    return model, history


# --------------------------------------------------------------------------------------
# Coreset selection: LBCS and baselines (identical settings to Section 5.2)
# --------------------------------------------------------------------------------------


def select_lbcs_mask(
    context: Dict[str, Any],
    k: int,
    config: Optional[Table6Config] = None,
    seed: Optional[int] = None,
    lbcs: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Run LBCS / Algorithm 1 for predefined size ``k`` on the SVHN candidate pool."""
    cfg = config or context.get("config") or Table6Config()
    log = logger or LOGGER
    if lbcs is not None and callable(lbcs):
        result = lbcs(context=context, k=k, config=cfg, seed=seed)
        if isinstance(result, dict):
            return dict(result)
        return {"mask": result, "coreset_size": mask_size(result)}

    from lbcs_repro.lbcs.bilevel import InnerTrainConfig, LBCS, LBCSConfig

    inner = InnerTrainConfig.from_dict(
        {
            "optimizer": cfg.inner_optimizer,
            "lr": cfg.inner_lr,
            "epochs": cfg.inner_epochs,
            "batch_size": cfg.batch_size,
        }
    )
    lbcs_config = LBCSConfig(
        k=int(k),
        epsilon=float(cfg.epsilon),
        T=int(cfg.T),
        delta_init=float(cfg.delta_init),
        delta_lower=float(cfg.delta_lower),
        warm_start=bool(cfg.warm_start),
        group_size=int(cfg.group_size),
        seed=seed,
        device=cfg.device,
    )
    runner = LBCS(
        model_factory=context["inner_factory"],
        n=int(context["n"]),
        k=int(k),
        dataset=context["dataset_for_inner"],
        eval_loader=context["test_loader"],
        inner_config=inner,
        config=lbcs_config,
        device=context.get("device"),
        seed=seed,
        log_every=int(cfg.log_every),
        logger=log,
    )
    result = runner.run()
    mask = getattr(result, "mask", None)
    if mask is None:  # pragma: no cover - defensive
        mask = runner.coreset_mask()
    return {
        "mask": np.asarray(mask).ravel(),
        "coreset_size": mask_size(mask),
        "f1": getattr(result, "f1", None),
        "f2": getattr(result, "f2", None),
        "restarts": getattr(result, "restarts", None),
        "wall_time": getattr(result, "wall_time", None),
        "result": result,
    }


def select_baseline_mask(
    name: str,
    context: Dict[str, Any],
    k: int,
    config: Optional[Table6Config] = None,
    seed: Optional[int] = None,
    model: Optional[Any] = None,
) -> Any:
    """Run a Section 5.2 baseline with a fixed coreset size ``k``; fall back to Uniform."""
    n = int(context["n"])
    targets = context.get("targets")
    try:
        from lbcs_repro.baselines import make_baseline

        selector = make_baseline(
            name,
            seed=seed,
            device=context.get("device"),
            num_classes=context.get("num_classes"),
        )
        kwargs: Dict[str, Any] = dict(
            n=n, k=int(k), dataset=context["train_dataset"], targets=targets
        )
        if getattr(selector, "requires_model", False):
            kwargs["model"] = model
            kwargs["loader"] = context.get("train_loader")
        try:
            return selector.select_mask(**kwargs)
        except TypeError:
            return selector.select_mask(n, int(k))
    except Exception as exc:  # pragma: no cover - long sweeps must not die
        LOGGER.warning("baseline %s failed (%s); falling back to Uniform", name, exc)
        from lbcs_repro.baselines.uniform import uniform_indices

        idx = np.asarray(uniform_indices(n, int(k), seed=seed), dtype=np.int64)
        out = np.zeros(n, dtype=np.float32)
        out[idx] = 1.0
        return out


def train_and_evaluate(
    context: Dict[str, Any],
    mask: Any,
    seed: Optional[int] = None,
    config: Optional[Table6Config] = None,
    model: Optional[Any] = None,
    train_fn: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Train the cross-architecture target model on the coreset and evaluate it."""
    cfg = config or context.get("config") or Table6Config()
    size = mask_size(mask)
    if size == 0:  # pragma: no cover - degenerate mask
        return {
            "accuracy": 0.0,
            "coreset_size": 0,
            "accuracy_per_point": 0.0,
            "history": [],
            "model": None,
        }
    loader = coreset_loader(context, mask, batch_size=cfg.batch_size, seed=seed)
    if train_fn is not None and callable(train_fn):
        return dict(train_fn(context=context, loader=loader, mask=mask, seed=seed, config=cfg))
    target = model if model is not None else context["target_factory"]()
    trained, history = train_target_model(
        target,
        loader,
        test_loader=context.get("test_loader"),
        epochs=int(cfg.target_epochs),
        lr=float(cfg.target_lr),
        optimizer=str(cfg.target_optimizer),
        weight_decay=float(cfg.weight_decay),
        momentum=float(cfg.momentum),
        device=context.get("device"),
        log_every=int(cfg.log_every),
    )
    acc = evaluate_accuracy(trained, context["test_loader"], device=context.get("device"))
    return {
        "accuracy": float(acc),
        "coreset_size": int(size),
        "accuracy_per_point": float(acc) / max(size, 1),
        "history": history,
        "model": trained,
    }


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------


def run_single_cell(
    architecture: str,
    k: int,
    repeat: int,
    config: Optional[Table6Config] = None,
    context: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    select_lbcs_fn: Optional[Callable[..., Any]] = None,
    select_baseline_fn: Optional[Callable[..., Any]] = None,
    train_eval_fn: Optional[Callable[..., Any]] = None,
    lbcs: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run every method once for one (architecture, k, repeat) triple."""
    cfg = config or Table6Config()
    log = logger or LOGGER
    ctx = context if context is not None else build_context(architecture, config=cfg)
    cell_seed = seed if seed is not None else resolve_seed(cfg.seed, repeat)
    set_seed(cell_seed)

    record: Dict[str, Any] = {
        "architecture": canonical_architecture(architecture),
        "k": int(k),
        "repeat": int(repeat),
        "seed": int(cell_seed),
        "dataset": ctx.get("dataset", cfg.dataset),
        "methods": {},
        "failed": False,
    }
    per_method: Dict[str, Dict[str, Any]] = {}

    if select_lbcs_fn is not None:
        lbcs_fn = select_lbcs_fn
    else:
        lbcs_fn = lambda **kw: select_lbcs_mask(**kw, lbcs=lbcs)  # noqa: E731
    base_fn = select_baseline_fn or select_baseline_mask

    # -- LBCS (ours) -----------------------------------------------------------------
    lbcs_size = None
    try:
        out = lbcs_fn(context=ctx, k=int(k), config=cfg, seed=cell_seed, logger=log)
        mask = out.get("mask") if isinstance(out, dict) else out
        lbcs_size = (
            int(out.get("coreset_size", mask_size(mask))) if isinstance(out, dict) else mask_size(mask)
        )
        measured = (
            train_eval_fn(context=ctx, mask=mask, seed=cell_seed, config=cfg)
            if train_eval_fn is not None
            else train_and_evaluate(ctx, mask, seed=cell_seed, config=cfg)
        )
        entry = {
            "accuracy": float(measured.get("accuracy", float("nan"))),
            "coreset_size": int(measured.get("coreset_size", lbcs_size)),
            "f1": out.get("f1") if isinstance(out, dict) else None,
            "f2": out.get("f2") if isinstance(out, dict) else None,
        }
    except Exception as exc:  # pragma: no cover - keep the sweep alive
        log.warning("LBCS failed for %s k=%s repeat=%s: %s", architecture, k, repeat, exc)
        record["failed"] = True
        entry = {"accuracy": float("nan"), "coreset_size": int(k), "f1": None, "f2": None}
    per_method[LBCS_LABEL] = entry

    # -- Baselines (same predefined size k; "other experimental settings are not changed")
    for name in cfg.methods:
        try:
            mask = base_fn(name, ctx, int(k), cfg, cell_seed)
            measured = (
                train_eval_fn(context=ctx, mask=mask, seed=cell_seed, config=cfg)
                if train_eval_fn is not None
                else train_and_evaluate(ctx, mask, seed=cell_seed, config=cfg)
            )
            per_method[name] = {
                "accuracy": float(measured.get("accuracy", float("nan"))),
                "coreset_size": int(measured.get("coreset_size", mask_size(mask))),
                "f1": None,
                "f2": None,
            }
        except Exception as exc:  # pragma: no cover
            log.warning("baseline %s failed for %s k=%s: %s", name, architecture, k, exc)
            per_method[name] = {
                "accuracy": float("nan"),
                "coreset_size": int(k),
                "f1": None,
                "f2": None,
            }

    record["methods"] = per_method
    record["lbcs_coreset_size"] = lbcs_size
    return record


def run_table6(
    config: Optional[Table6Config] = None,
    logger: Optional[logging.Logger] = None,
    cell_runner: Optional[Callable[..., Any]] = None,
    context_builder: Optional[Callable[..., Any]] = None,
    lbcs: Optional[Any] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Run the Section 6 cross-architecture (Table 6) experiment."""
    cfg = config or Table6Config()
    if overrides:
        cfg = cfg.with_overrides(**overrides)
    log = logger or LOGGER

    architectures = tuple(canonical_architecture(a) for a in cfg.architectures)
    records: List[Dict[str, Any]] = []
    contexts: Dict[str, Dict[str, Any]] = {}

    for architecture in architectures:
        if context_builder is not None:
            ctx = context_builder(architecture=architecture, config=cfg)
        else:
            try:
                ctx = build_context(architecture, config=cfg)
            except Exception as exc:  # pragma: no cover - offline tolerance
                log.warning("could not build context for %s: %s", architecture, exc)
                continue
        contexts[architecture] = ctx
        for k in cfg.ks:
            for repeat in range(int(cfg.repeats)):
                if cell_runner is not None:
                    record = cell_runner(
                        architecture=architecture, k=int(k), repeat=repeat, config=cfg, context=ctx
                    )
                else:
                    record = run_single_cell(
                        architecture=architecture,
                        k=int(k),
                        repeat=repeat,
                        config=cfg,
                        context=ctx,
                        lbcs=lbcs,
                        logger=log,
                    )
                records.append(record)

    cells = aggregate_results(records)
    checks = direction_checks(cells, config=cfg)
    table_text = format_table6(cells, config=cfg)

    artifacts: Dict[str, str] = {}
    if cfg.save_artifacts:
        artifacts = save_results(cells, records, checks, cfg, table_text=table_text)
        try:
            plotted = plot_table6(cells, os.path.join(cfg.output_dir, "table6.png"), config=cfg)
            if plotted:
                artifacts["png"] = plotted
        except Exception as exc:  # pragma: no cover
            log.warning("plotting failed: %s", exc)

    return {
        "table": table_text,
        "config": cfg.to_dict(),
        "cells": cells,
        "records": records,
        "checks": checks,
        "artifacts": artifacts,
        "paper_table6": PAPER_TABLE6,
        "paper_lbcs_sizes": PAPER_TABLE2_SVHN_SIZES,
    }


# --------------------------------------------------------------------------------------
# Aggregation / reporting
# --------------------------------------------------------------------------------------


def aggregate_results(records: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, int], Table6Cell]:
    """Group per-repeat records into mean/std cells keyed by (architecture, k)."""
    acc_buckets: Dict[Tuple[str, int], Dict[str, List[float]]] = {}
    size_buckets: Dict[Tuple[str, int], Dict[str, List[float]]] = {}
    f1_buckets: Dict[Tuple[str, int], Dict[str, List[float]]] = {}
    counts: Dict[Tuple[str, int], int] = {}
    failures: Dict[Tuple[str, int], int] = {}
    raws: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}

    for record in records:
        key = (canonical_architecture(record.get("architecture")), int(record.get("k", 0)))
        acc_buckets.setdefault(key, {})
        size_buckets.setdefault(key, {})
        f1_buckets.setdefault(key, {})
        counts[key] = counts.get(key, 0) + 1
        failures[key] = failures.get(key, 0) + (1 if record.get("failed") else 0)
        raws.setdefault(key, []).append(record)
        for method, entry in (record.get("methods") or {}).items():
            acc = entry.get("accuracy")
            if acc is not None and np.isfinite(float(acc)):
                acc_buckets[key].setdefault(method, []).append(float(acc))
            size = entry.get("coreset_size")
            if size is not None and np.isfinite(float(size)):
                size_buckets[key].setdefault(method, []).append(float(size))
            f1 = entry.get("f1")
            if f1 is not None and np.isfinite(float(f1)):
                f1_buckets[key].setdefault(method, []).append(float(f1))

    cells: Dict[Tuple[str, int], Table6Cell] = {}
    for key in sorted(acc_buckets.keys(), key=lambda item: (item[0], item[1])):
        architecture, k = key
        cell = Table6Cell(architecture=architecture, k=int(k))
        for method, values in acc_buckets[key].items():
            cell.accuracy[method] = mean_std(values)
        for method, values in size_buckets[key].items():
            cell.coreset_size[method] = mean_std(values)
        for method, values in f1_buckets[key].items():
            cell.f1[method] = mean_std(values)
        cell.repeats = int(counts.get(key, 0))
        cell.failures = int(failures.get(key, 0))
        cell.raw = raws.get(key, [])
        cells[key] = cell
    return cells


def direction_checks(
    cells: Dict[Tuple[str, int], Table6Cell], config: Optional[Table6Config] = None
) -> Dict[str, Any]:
    """Validate the Section 6 qualitative claims of Table 6."""
    checks: Dict[str, Any] = {"details": {}}

    vit_best = 0
    wnet_best = 0
    size_ok = True
    at_least_uniform = 0
    total_cells = 0

    for (architecture, k), cell in cells.items():
        accs = {m: v[0] for m, v in cell.accuracy.items() if np.isfinite(v[0])}
        if not accs:
            continue
        best_method = max(accs, key=lambda m: accs[m])
        total_cells += 1
        if best_method == LBCS_LABEL:
            if architecture == "ViTSmall":
                vit_best += 1
            if architecture == "WideResNet":
                wnet_best += 1
        size = cell.size_mean(LBCS_LABEL)
        if size is not None and size > float(k) + 1e-6:
            size_ok = False
        uniform_acc = accs.get("Uniform")
        lbcs_acc = accs.get(LBCS_LABEL)
        if uniform_acc is not None and lbcs_acc is not None and lbcs_acc >= uniform_acc:
            at_least_uniform += 1

        checks["details"][f"{architecture}|k={k}"] = {
            "best_method": best_method,
            "best_accuracy": float(accs[best_method]),
            "lbcs_accuracy": None if lbcs_acc is None else float(lbcs_acc),
            "lbcs_coreset_size": size,
            "repeats": int(cell.repeats),
        }

    checks["vit_cells_lbcs_best"] = vit_best
    checks["vit_cells_total"] = sum(1 for (a, _) in cells if a == "ViTSmall")
    checks["wnet_cells_lbcs_best"] = wnet_best
    checks["wnet_cells_total"] = sum(1 for (a, _) in cells if a == "WideResNet")
    checks["lbcs_size_le_k"] = bool(size_ok)
    checks["cells_lbcs_at_least_uniform"] = at_least_uniform
    checks["cells_total"] = total_cells
    checks["summary"] = (
        "Section 6 asks LBCS to remain superior/competitive under ViT-small and W-NET "
        "target networks while using a smaller-than-k coreset."
    )
    return checks


def format_table6(cells: Dict[Tuple[str, int], Table6Cell], config: Optional[Table6Config] = None) -> str:
    """Render the Table 6 layout (accuracy per method + LBCS coreset size)."""
    cfg = config or Table6Config()
    methods = list(cfg.methods)
    lines: List[str] = []
    header = ["Architecture", "k"] + methods + [LBCS_LABEL, SIZE_LABEL]
    lines.append(" | ".join(header))
    lines.append("-" * len(lines[-1]))
    for (architecture, k), cell in sorted(cells.items(), key=lambda item: (item[0][0], item[0][1])):
        row = [architecture, str(int(k))]
        for method in methods:
            entry = cell.accuracy.get(method)
            row.append("-" if entry is None else format_mean_std(entry[0], entry[1]))
        lbcs_entry = cell.accuracy.get(LBCS_LABEL)
        row.append("-" if lbcs_entry is None else format_mean_std(lbcs_entry[0], lbcs_entry[1]))
        size = cell.size_mean(LBCS_LABEL)
        row.append("-" if size is None else f"{size:.1f}")
        lines.append(" | ".join(row))

    lines.append("")
    lines.append("Paper reference (Table 6, mean +- std test accuracy %):")
    for architecture in PAPER_ARCHITECTURES:
        for k in PAPER_KS:
            block = PAPER_TABLE6.get(architecture, {}).get(k)
            if not block:
                continue
            ref = " | ".join(f"{m}: {format_mean_std(*block[m])}" for m in block.keys())
            lines.append(f"  {architecture} k={k}: {ref}")
    return "\n".join(lines)


def plot_table6(
    cells: Dict[Tuple[str, int], Table6Cell],
    out_path: str,
    config: Optional[Table6Config] = None,
    dpi: int = 150,
) -> Optional[str]:
    """Plot test accuracy vs predefined coreset size for each target architecture."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - plotting optional
        return None

    architectures = sorted({architecture for (architecture, _) in cells})
    if not architectures:
        return None
    fig, axes = plt.subplots(
        1, len(architectures), figsize=(6 * len(architectures), 4), squeeze=False
    )
    for ax, architecture in zip(axes[0], architectures):
        ks = sorted({k for (a, k) in cells if a == architecture})
        for method in list((config or Table6Config()).methods) + [LBCS_LABEL]:
            ys = []
            for k in ks:
                entry = cells.get((architecture, k))
                value = None if entry is None else entry.accuracy.get(method)
                ys.append(float("nan") if value is None else value[0])
            ax.plot(ks, ys, marker="o", label=method)
        ax.set_title(architecture)
        ax.set_xlabel("predefined coreset size k")
        ax.set_ylabel("test accuracy (%)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def save_results(
    cells: Dict[Tuple[str, int], Table6Cell],
    records: Sequence[Dict[str, Any]],
    checks: Dict[str, Any],
    config: Table6Config,
    table_text: Optional[str] = None,
) -> Dict[str, str]:
    """Persist Table 6 artifacts (JSON/CSV/TXT/JSONL/checks)."""
    out_dir = config.output_dir
    os.makedirs(out_dir, exist_ok=True)
    artifacts: Dict[str, str] = {}

    payload = {
        "table6": {f"{a}|k={k}": cell.to_dict() for (a, k) in sorted(cells.keys())},
        "paper_table6": PAPER_TABLE6,
        "paper_lbcs_sizes": {str(k): list(v) for k, v in PAPER_TABLE2_SVHN_SIZES.items()},
        "checks": checks,
        "config": config.to_dict(),
    }
    json_path = os.path.join(out_dir, "table6.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2)
    artifacts["json"] = json_path

    text = table_text if table_text is not None else format_table6(cells, config=config)
    txt_path = os.path.join(out_dir, "table6.txt")
    with open(txt_path, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    artifacts["txt"] = txt_path

    csv_path = os.path.join(out_dir, "table6.csv")
    headers = [
        "architecture",
        "k",
        "method",
        "accuracy_mean",
        "accuracy_std",
        "coreset_size_mean",
        "coreset_size_std",
    ]
    rows: List[List[Any]] = []
    for (architecture, k), cell in sorted(cells.items(), key=lambda item: (item[0][0], item[0][1])):
        for method in list(cell.accuracy.keys()):
            acc = cell.accuracy.get(method, (float("nan"), float("nan")))
            size = cell.coreset_size.get(method, (float("nan"), float("nan")))
            rows.append([architecture, int(k), method, acc[0], acc[1], size[0], size[1]])
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)
    artifacts["csv"] = csv_path

    jsonl_path = os.path.join(out_dir, "table6_raw.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_jsonable(record)) + "\n")
    artifacts["jsonl"] = jsonl_path

    checks_path = os.path.join(out_dir, "table6_checks.json")
    with open(checks_path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(checks), handle, indent=2)
    artifacts["checks"] = checks_path
    return artifacts


def _jsonable(obj: Any) -> Any:
    """Recursively convert results to JSON-serialisable objects."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if hasattr(obj, "to_dict"):
        try:
            return _jsonable(obj.to_dict())
        except Exception:  # pragma: no cover
            return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Table 6: Section 6 cross-network-architecture study (SVHN)"
    )
    parser.add_argument("--dataset", type=str, default=PAPER_DATASET)
    parser.add_argument("--architectures", type=str, nargs="*", default=list(PAPER_ARCHITECTURES))
    parser.add_argument("--ks", type=int, nargs="*", default=list(PAPER_KS))
    parser.add_argument("--epsilon", type=float, default=PAPER_EPSILON)
    parser.add_argument("--T", type=int, default=PAPER_T)
    parser.add_argument("--repeats", type=int, default=PAPER_REPEATS)
    parser.add_argument("--inner-epochs", type=int, default=SUGGESTED_INNER_EPOCHS)
    parser.add_argument("--target-epochs", type=int, default=PAPER_TARGET_EPOCHS)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=str, default=None, help="optional YAML config path")
    parser.add_argument("--paper", action="store_true", help="run the paper protocol")
    parser.add_argument("--smoke", action="store_true", help="run a tiny smoke configuration")
    parser.add_argument("--selftest", action="store_true", help="offline self-test (no torch/data)")
    return parser


def _config_from_args(args: argparse.Namespace) -> Table6Config:
    if args.smoke:
        cfg = Table6Config.smoke()
    elif args.paper:
        cfg = Table6Config()
    else:
        cfg = Table6Config(
            dataset=canonical_dataset(args.dataset),
            architectures=tuple(canonical_architecture(a) for a in args.architectures),
            ks=tuple(int(k) for k in args.ks),
            epsilon=float(args.epsilon),
            T=int(args.T),
            repeats=int(args.repeats),
            inner_epochs=int(args.inner_epochs),
            target_epochs=int(args.target_epochs),
            device=args.device,
            seed=int(args.seed),
            output_dir=args.output_dir,
        )
    if args.config:
        try:
            import yaml

            with open(args.config, "r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
            section = data.get("table6") or data.get("section6") or data
            cfg = cfg.with_overrides(**dict(section))
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("could not load config %s: %s", args.config, exc)
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.selftest:
        report = _selftest(verbose=True)
        return 0 if report.get("ok") else 1

    cfg = _config_from_args(args)
    logging.getLogger(__name__).info("Table 6 (cross architecture) with %s", cfg.to_dict())
    result = run_table6(config=cfg)
    print(result["table"])
    return 0


# --------------------------------------------------------------------------------------
# Offline self-test / dry-run
# --------------------------------------------------------------------------------------


class _SyntheticCellRunner:
    """Deterministic offline stand-in for a full cross-architecture cell run."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)

    def __call__(
        self,
        architecture: str,
        k: int,
        repeat: int,
        config: Any = None,
        context: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        rng = np.random.default_rng(resolve_seed(self.seed, repeat) + int(k))
        base = 30.0 if canonical_architecture(architecture) == "ViTSmall" else 78.0
        methods: Dict[str, Dict[str, Any]] = {}
        for method in METHOD_ORDER:
            methods[method] = {
                "accuracy": float(base + 10.0 * np.log10(max(k, 1)) + rng.normal(0, 0.2)),
                "coreset_size": int(k),
            }
        methods[LBCS_LABEL] = {
            "accuracy": float(base + 10.0 * np.log10(max(k, 1)) + 0.5 + rng.normal(0, 0.1)),
            "coreset_size": int(round(k * 0.95)),
            "f1": float(1.0 + rng.random() * 0.1),
            "f2": float(round(k * 0.95)),
        }
        return {
            "architecture": canonical_architecture(architecture),
            "k": int(k),
            "repeat": int(repeat),
            "seed": int(resolve_seed(self.seed, repeat)),
            "dataset": PAPER_DATASET,
            "methods": methods,
            "failed": False,
        }


def _selftest(verbose: bool = True) -> Dict[str, Any]:
    """Offline checks: masks, aggregation, formatting, checks, config, dry run."""
    report: Dict[str, Any] = {"ok": True, "checks": {}}

    # Mask algebra / projection (Appendix A rule).
    mask = np.array([-1.5, -0.2, 0.0, 0.4, 1.7], dtype=np.float64)
    report["checks"]["binarize"] = bool(
        np.array_equal(binarize_mask(mask), np.array([0, 0, 1, 1, 1], dtype=np.float32))
    )
    report["checks"]["size"] = mask_size(mask) == 4

    # Canonicalisation.
    report["checks"]["arch_alias"] = canonical_architecture("vit-small") == "ViTSmall"
    report["checks"]["arch_alias2"] = canonical_architecture("W-NET") == "WideResNet"
    report["checks"]["dataset_alias"] = canonical_dataset("svhn") == "SVHN"

    # Config round-trips.
    cfg = Table6Config.paper()
    report["checks"]["config_paper"] = (
        cfg.epsilon == 0.2 and cfg.T == 500 and tuple(cfg.ks) == PAPER_KS
    )
    round_trip = Table6Config.from_dict(cfg.to_dict()).to_dict()
    report["checks"]["config_round_trip"] = round_trip["ks"] == list(PAPER_KS)
    report["checks"]["config_override"] = Table6Config().with_overrides(ks=(512,)).ks == (512,)

    # Aggregation + formatting + checks (dry run, fully offline).
    runner = _SyntheticCellRunner()
    records = [
        runner(architecture=arch, k=k, repeat=r)
        for arch in PAPER_ARCHITECTURES
        for k in (1000, 2000)
        for r in range(2)
    ]
    cells = aggregate_results(records)
    report["checks"]["num_cells"] = len(cells) == 4
    sample = cells[("ViTSmall", 1000)]
    report["checks"]["cell_has_lbcs"] = LBCS_LABEL in sample.accuracy
    report["checks"]["cell_repeats"] = sample.repeats == 2
    text = format_table6(cells, config=Table6Config())
    report["checks"]["format_has_header"] = "Architecture" in text and "k" in text
    checks = direction_checks(cells, config=Table6Config())
    report["checks"]["size_le_k"] = bool(checks.get("lbcs_size_le_k", False))

    # Reference-table sanity.
    report["checks"]["paper_table6_complete"] = all(
        LBCS_LABEL in PAPER_TABLE6[arch][k] for arch in PAPER_ARCHITECTURES for k in PAPER_KS
    )
    report["checks"]["paper_sizes_complete"] = all(k in PAPER_TABLE2_SVHN_SIZES for k in PAPER_KS)

    report["ok"] = all(bool(v) for v in report["checks"].values())
    if verbose:
        print("Table 6 (cross architecture) self-test")
        for key, value in report["checks"].items():
            print(f"  {key:<28} {value}")
        print(f"  {'OK':<28} {report['ok']}")
    return report


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())
