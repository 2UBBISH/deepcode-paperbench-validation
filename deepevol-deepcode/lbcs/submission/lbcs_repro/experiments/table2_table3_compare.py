"""Section 5.2 experiment driver: Tables 2 and 3 (comparison with the competitors).

This module reproduces the two kinds of comparison described in Section 5.2 of
the LBCS paper ("Refined Coreset Selection"):

* **Table 2 -- first kind of comparison.**  Every coreset-selection method is
  given the same *predefined* coreset size ``k``.  After coreset selection a
  target network is trained on the constructed coreset and the measurements are
  (a) the model test accuracy on test data and (b) the coreset size.  For LBCS
  (the only method that also minimizes the size) the *optimized* coreset size
  ``||m||_0`` is reported next to the accuracy: a higher accuracy with a smaller
  coreset size is better.  The average accuracy brought by per data point within
  the coreset (Appendix E.1) is computed as well.

* **Table 3 -- second kind of comparison.**  The coreset size *achieved by LBCS*
  is applied to the baselines; their coreset selection and model training then
  start from that size and the only measurement is the test accuracy under the
  same coreset size (higher accuracy = superior coreset selection).

Paper settings reproduced here (Section 5.2):

* benchmarks: F-MNIST, SVHN, CIFAR-10;
* predefined coreset sizes ``k in {1000, 2000, 3000, 4000}``;
* coreset selection (inner loop): LeNet for F-MNIST and simple CNNs for SVHN and
  CIFAR-10 with an Adam optimizer at learning rate ``0.001``;
* ``epsilon = 0.2`` and ``T = 500`` for LBCS;
* every experiment is repeated **ten** times; mean and std are reported;
* post-selection target training: LeNet for F-MNIST, a CNN for SVHN and
  ResNet-18 for CIFAR-10; Adam with learning rate ``0.001`` for ``100`` epochs
  on F-MNIST/SVHN and SGD with initial learning rate ``0.1`` and a cosine rate
  scheduler for ``200`` epochs on CIFAR-10.

Everything the paper does *not* state (batch size, weight decay, DataLoader
workers, ...) is an explicitly labelled ``SUGGESTED`` default stored in
:data:`TARGET_CONFIGS` / :class:`Table2Config` so it can be changed from YAML
without touching the experiment logic.

The paper's own Table 2/3 numbers are stored in :data:`PAPER_TABLE2` /
:data:`PAPER_TABLE3` and are used *only* to validate the reproduction
(qualitative direction checks); they are never an input to any algorithm.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # PyTorch is a soft dependency: mask algebra works without it.
    import torch
    import torch.nn as nn

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch-free environments
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

LOGGER = logging.getLogger("lbcs_repro.experiments.table2_table3")

# --------------------------------------------------------------------------------------
# Paper constants and reported reference values (Section 5.2)
# --------------------------------------------------------------------------------------

PAPER_DATASETS: Tuple[str, ...] = ("F-MNIST", "SVHN", "CIFAR-10")
PAPER_KS: Tuple[int, ...] = (1000, 2000, 3000, 4000)
PAPER_EPSILON: float = 0.2
PAPER_T: int = 500
PAPER_REPEATS: int = 10

#: Competitor order of the paper tables; LBCS/size columns are appended last.
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

#: Paper-reported Table 2: ``dataset -> k -> method -> (mean %, std)``.
PAPER_TABLE2: Dict[str, Dict[int, Dict[str, Tuple[float, float]]]] = {
    "F-MNIST": {
        1000: {
            "Uniform": (76.9, 2.5), "EL2N": (71.8, 2.9), "GraNd": (70.7, 4.0),
            "Influential": (78.9, 2.0), "Moderate": (77.0, 0.6), "CCS": (76.7, 3.5),
            "Probabilistic": (80.3, 0.6), LBCS_LABEL: (79.7, 0.7),
            SIZE_LABEL: (956.7, 3.5),
        },
        2000: {
            "Uniform": (80.0, 2.4), "EL2N": (73.7, 1.6), "GraNd": (71.7, 2.3),
            "Influential": (80.4, 0.8), "Moderate": (80.3, 0.4), "CCS": (81.4, 0.6),
            "Probabilistic": (82.6, 0.2), LBCS_LABEL: (82.8, 0.6),
            SIZE_LABEL: (1915.3, 6.6),
        },
        3000: {
            "Uniform": (81.7, 1.7), "EL2N": (75.3, 2.3), "GraNd": (73.3, 1.8),
            "Influential": (81.5, 1.2), "Moderate": (81.7, 0.5), "CCS": (82.6, 1.2),
            "Probabilistic": (83.7, 0.9), LBCS_LABEL: (84.0, 0.6),
            SIZE_LABEL: (2831.6, 10.9),
        },
        4000: {
            "Uniform": (83.0, 1.7), "EL2N": (77.0, 1.0), "GraNd": (75.9, 2.1),
            "Influential": (82.4, 1.3), "Moderate": (82.4, 0.3), "CCS": (84.1, 0.6),
            "Probabilistic": (84.2, 0.7), LBCS_LABEL: (84.5, 0.4),
            SIZE_LABEL: (3745.4, 15.6),
        },
    },
    "SVHN": {
        1000: {
            "Uniform": (67.1, 3.3), "EL2N": (56.8, 1.3), "GraNd": (60.7, 1.1),
            "Influential": (70.3, 0.8), "Moderate": (68.4, 2.0), "CCS": (66.9, 1.9),
            "Probabilistic": (67.8, 0.4), LBCS_LABEL: (70.6, 0.3),
            SIZE_LABEL: (970.0, 4.8),
        },
        2000: {
            "Uniform": (75.9, 1.0), "EL2N": (64.8, 0.6), "GraNd": (67.3, 2.0),
            "Influential": (76.2, 1.3), "Moderate": (77.9, 0.7), "CCS": (77.3, 0.8),
            "Probabilistic": (76.6, 1.3), LBCS_LABEL: (78.3, 0.7),
            SIZE_LABEL: (1902.3, 10.3),
        },
        3000: {
            "Uniform": (80.3, 1.2), "EL2N": (72.1, 2.8), "GraNd": (75.2, 1.6),
            "Influential": (80.8, 1.5), "Moderate": (81.8, 0.7), "CCS": (81.9, 0.6),
            "Probabilistic": (80.9, 1.1), LBCS_LABEL: (82.3, 0.7),
            SIZE_LABEL: (2712.6, 15.0),
        },
        4000: {
            "Uniform": (83.9, 0.8), "EL2N": (75.8, 1.9), "GraNd": (79.1, 2.4),
            "Influential": (83.6, 1.8), "Moderate": (83.9, 0.6), "CCS": (84.1, 0.3),
            "Probabilistic": (84.3, 1.4), LBCS_LABEL: (84.6, 0.6),
            SIZE_LABEL: (3804.2, 16.4),
        },
    },
    "CIFAR-10": {
        1000: {
            "Uniform": (46.9, 1.8), "EL2N": (36.8, 1.2), "GraNd": (41.6, 2.0),
            "Influential": (45.7, 1.1), "Moderate": (48.1, 2.2), "CCS": (47.6, 1.6),
            "Probabilistic": (48.2, 0.9), LBCS_LABEL: (48.3, 1.2),
            SIZE_LABEL: (970.4, 2.9),
        },
        2000: {
            "Uniform": (58.1, 2.0), "EL2N": (47.9, 0.7), "GraNd": (52.3, 2.4),
            "Influential": (57.7, 1.3), "Moderate": (58.5, 1.3), "CCS": (59.3, 1.4),
            "Probabilistic": (60.1, 0.8), LBCS_LABEL: (60.4, 1.0),
            SIZE_LABEL: (1955.2, 5.3),
        },
        3000: {
            "Uniform": (65.7, 2.3), "EL2N": (56.1, 1.9), "GraNd": (61.9, 1.7),
            "Influential": (67.5, 1.6), "Moderate": (69.2, 2.6), "CCS": (67.6, 1.6),
            "Probabilistic": (68.7, 1.1), LBCS_LABEL: (69.5, 0.9),
            SIZE_LABEL: (2913.8, 9.6),
        },
        4000: {
            "Uniform": (70.9, 2.5), "EL2N": (63.0, 2.0), "GraNd": (67.9, 1.3),
            "Influential": (71.7, 2.4), "Moderate": (73.9, 0.4), "CCS": (73.0, 0.9),
            "Probabilistic": (73.6, 0.2), LBCS_LABEL: (73.4, 0.5),
            SIZE_LABEL: (3736.0, 14.2),
        },
    },
}

#: Paper-reported Table 3: accuracy at the coreset sizes achieved by LBCS.
PAPER_TABLE3: Dict[str, Dict[int, Dict[str, Tuple[float, float]]]] = {
    "F-MNIST": {
        956: {
            "Uniform": (76.5, 1.8), "EL2N": (71.3, 3.1), "GraNd": (70.8, 1.1),
            "Influential": (78.2, 0.9), "Moderate": (76.3, 0.5), "CCS": (75.4, 1.1),
            "Probabilistic": (79.2, 0.9), LBCS_LABEL: (79.7, 0.5),
        },
        1935: {
            "Uniform": (79.8, 2.1), "EL2N": (73.2, 1.3), "GraNd": (71.2, 1.5),
            "Influential": (80.0, 1.9), "Moderate": (79.7, 0.5), "CCS": (80.3, 0.6),
            "Probabilistic": (81.7, 0.7), LBCS_LABEL: (82.8, 0.4),
        },
        2832: {
            "Uniform": (81.2, 1.3), "EL2N": (75.0, 1.6), "GraNd": (73.2, 1.1),
            "Influential": (81.0, 0.7), "Moderate": (81.4, 0.3), "CCS": (82.5, 0.7),
            "Probabilistic": (83.4, 0.6), LBCS_LABEL: (84.0, 0.4),
        },
        3746: {
            "Uniform": (82.8, 1.5), "EL2N": (77.0, 2.2), "GraNd": (75.1, 1.6),
            "Influential": (82.1, 1.0), "Moderate": (82.2, 0.4), "CCS": (83.6, 1.0),
            "Probabilistic": (83.8, 0.5), LBCS_LABEL: (84.5, 0.3),
        },
    },
    "SVHN": {
        970: {
            "Uniform": (66.7, 2.6), "EL2N": (57.2, 0.5), "GraNd": (60.6, 1.7),
            "Influential": (70.3, 1.2), "Moderate": (68.4, 1.8), "CCS": (65.1, 1.1),
            "Probabilistic": (67.6, 1.3), LBCS_LABEL: (70.6, 0.3),
        },
        1902: {
            "Uniform": (75.7, 1.8), "EL2N": (65.0, 0.7), "GraNd": (67.0, 1.2),
            "Influential": (75.5, 0.9), "Moderate": (77.7, 1.2), "CCS": (75.9, 1.4),
            "Probabilistic": (76.1, 0.7), LBCS_LABEL: (78.3, 0.7),
        },
        2713: {
            "Uniform": (79.5, 2.6), "EL2N": (72.3, 0.5), "GraNd": (74.8, 1.1),
            "Influential": (80.0, 1.9), "Moderate": (81.4, 1.1), "CCS": (81.1, 1.0),
            "Probabilistic": (80.5, 0.4), LBCS_LABEL: (82.3, 0.8),
        },
        3805: {
            "Uniform": (83.6, 1.2), "EL2N": (75.5, 1.8), "GraNd": (78.2, 1.3),
            "Influential": (82.8, 1.6), "Moderate": (83.6, 0.6), "CCS": (84.2, 0.3),
            "Probabilistic": (83.5, 1.2), LBCS_LABEL: (84.6, 0.6),
        },
    },
    "CIFAR-10": {
        970: {
            "Uniform": (46.8, 1.2), "EL2N": (36.7, 1.1), "GraNd": (41.4, 1.9),
            "Influential": (44.8, 1.5), "Moderate": (46.2, 1.9), "CCS": (45.4, 1.0),
            "Probabilistic": (47.8, 1.1), LBCS_LABEL: (48.3, 1.2),
        },
        1955: {
            "Uniform": (58.0, 1.3), "EL2N": (48.3, 1.9), "GraNd": (52.5, 1.2),
            "Influential": (57.6, 1.9), "Moderate": (57.4, 0.8), "CCS": (58.6, 1.4),
            "Probabilistic": (59.4, 1.2), LBCS_LABEL: (60.4, 1.0),
        },
        2914: {
            "Uniform": (65.5, 1.9), "EL2N": (55.0, 3.2), "GraNd": (67.7, 1.8),
            "Influential": (67.2, 1.0), "Moderate": (68.2, 2.1), "CCS": (66.5, 1.0),
            "Probabilistic": (68.0, 0.8), LBCS_LABEL: (69.5, 0.9),
        },
        3736: {
            "Uniform": (70.6, 2.4), "EL2N": (58.8, 1.9), "GraNd": (72.8, 1.1),
            "Influential": (70.2, 3.5), "Moderate": (73.0, 1.2), "CCS": (72.8, 0.9),
            "Probabilistic": (73.4, 0.5), LBCS_LABEL: (73.4, 0.5),
        },
    },
}

# --------------------------------------------------------------------------------------
# Coreset-selection (inner loop) and target-training protocols
# --------------------------------------------------------------------------------------

#: Inner-loop protocols: LeNet for F-MNIST, simple CNNs for SVHN / CIFAR-10,
#: Adam with lr 0.001 -- all paper-stated (Section 5.2).  Remaining fields are
#: SUGGESTED defaults.
INNER_CONFIGS: Dict[str, Dict[str, Any]] = {
    "F-MNIST": {"model": "LeNet", "optimizer": "adam", "lr": 0.001,
                "momentum": 0.9, "epochs": 100, "batch_size": 128, "weight_decay": 0.0},
    "SVHN": {"model": "SVHNCNN", "optimizer": "adam", "lr": 0.001,
             "momentum": 0.9, "epochs": 100, "batch_size": 128, "weight_decay": 0.0},
    "CIFAR-10": {"model": "CIFARCNN", "optimizer": "adam", "lr": 0.001,
                 "momentum": 0.9, "epochs": 100, "batch_size": 128, "weight_decay": 0.0},
}

#: Post-selection target training (Section 5.2).  ``model``/``optimizer``/``lr``/
#: ``epochs``/``scheduler`` are paper-stated; the remaining entries are SUGGESTED.
TARGET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "F-MNIST": {
        "model": "LeNet", "optimizer": "adam", "lr": 0.001, "momentum": 0.9,
        "epochs": 100, "scheduler": None,
        "batch_size": 128, "weight_decay": 0.0, "augment": False,
    },
    "SVHN": {
        "model": "SVHNCNNTarget", "optimizer": "adam", "lr": 0.001, "momentum": 0.9,
        "epochs": 100, "scheduler": None,
        "batch_size": 128, "weight_decay": 0.0, "augment": False,
    },
    "CIFAR-10": {
        "model": "ResNet18", "optimizer": "sgd", "lr": 0.1, "momentum": 0.9,
        "epochs": 200, "scheduler": "cosine",
        "batch_size": 128, "weight_decay": 5e-4, "augment": True,
    },
}

SUGGESTED_EVAL_BATCH_SIZE = 256
SUGGESTED_NUM_WORKERS = 0
SUGGESTED_DELTA_INIT = 0.1
SUGGESTED_DELTA_LOWER = 1e-3
SUGGESTED_VISIBLE_GPU = 0
DEFAULT_OUTPUT_DIR = os.path.join("results", "table2_table3")


# --------------------------------------------------------------------------------------
# Configuration containers
# --------------------------------------------------------------------------------------


@dataclass
class TargetTrainConfig:
    """Post-selection target-training recipe for one benchmark."""

    model: str = "LeNet"
    optimizer: str = "adam"
    lr: float = 0.001
    momentum: float = 0.9
    epochs: int = 100
    scheduler: Optional[str] = None
    batch_size: int = 128
    weight_decay: float = 0.0
    augment: bool = False

    @classmethod
    def for_dataset(cls, dataset: str, **overrides: Any) -> "TargetTrainConfig":
        spec = dict(TARGET_CONFIGS.get(canonical_dataset(dataset), TARGET_CONFIGS["F-MNIST"]))
        spec.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**spec)

    def with_overrides(self, **overrides: Any) -> "TargetTrainConfig":
        data = self.to_dict()
        data.update({k: v for k, v in overrides.items() if v is not None})
        return TargetTrainConfig(**data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model, "optimizer": self.optimizer, "lr": self.lr,
            "momentum": self.momentum, "epochs": self.epochs, "scheduler": self.scheduler,
            "batch_size": self.batch_size, "weight_decay": self.weight_decay,
            "augment": self.augment,
        }


@dataclass
class InnerTrainSpec:
    """Inner-loop (coreset selection) recipe for one benchmark."""

    model: str = "LeNet"
    optimizer: str = "adam"
    lr: float = 0.001
    momentum: float = 0.9
    epochs: int = 100
    batch_size: int = 128
    weight_decay: float = 0.0

    @classmethod
    def for_dataset(cls, dataset: str, **overrides: Any) -> "InnerTrainSpec":
        spec = dict(INNER_CONFIGS.get(canonical_dataset(dataset), INNER_CONFIGS["F-MNIST"]))
        spec.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**spec)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model, "optimizer": self.optimizer, "lr": self.lr,
            "momentum": self.momentum, "epochs": self.epochs,
            "batch_size": self.batch_size, "weight_decay": self.weight_decay,
        }


@dataclass
class Table2Config:
    """Configuration of the Section 5.2 (Tables 2-3) protocol."""

    datasets: Tuple[str, ...] = PAPER_DATASETS
    ks: Tuple[int, ...] = PAPER_KS
    methods: Tuple[str, ...] = METHOD_ORDER
    epsilon: float = PAPER_EPSILON
    T: int = PAPER_T
    repeats: int = PAPER_REPEATS
    inner_epochs: int = 100
    batch_size: int = 128
    eval_batch_size: int = SUGGESTED_EVAL_BATCH_SIZE
    num_workers: int = SUGGESTED_NUM_WORKERS
    delta_init: float = SUGGESTED_DELTA_INIT
    delta_lower: float = SUGGESTED_DELTA_LOWER
    warm_start: bool = True
    group_size: int = 1
    device: Optional[str] = None
    seed: int = 0
    log_every: int = 0
    output_dir: str = DEFAULT_OUTPUT_DIR
    data_root: Optional[str] = None
    save_artifacts: bool = True
    table3: bool = True
    f1_eval_split: str = "train"  # "train" (selection pool) or "test"
    verbose: bool = False

    # -- constructors -------------------------------------------------------------
    @classmethod
    def paper(cls, **overrides: Any) -> "Table2Config":
        return cls().with_overrides(**overrides)

    @classmethod
    def smoke(cls, **overrides: Any) -> "Table2Config":
        """Tiny configuration for an end-to-end smoke run."""
        return cls(
            datasets=("F-MNIST",),
            ks=(1000,),
            methods=("Uniform", "Moderate"),
            T=10,
            repeats=1,
            inner_epochs=1,
            batch_size=64,
            eval_batch_size=128,
            output_dir=os.path.join("results", "table2_smoke"),
            table3=False,
        ).with_overrides(**overrides)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Table2Config":
        cfg = cls()
        if not data:
            return cfg
        allowed = set(cfg.to_dict().keys())
        clean: Dict[str, Any] = {}
        for key, value in data.items():
            if key in allowed and value is not None:
                clean[key] = value
        return cfg.with_overrides(**clean)

    def with_overrides(self, **overrides: Any) -> "Table2Config":
        data = self.to_dict()
        for key, value in overrides.items():
            if value is None or key not in data:
                continue
            if key in ("datasets", "ks", "methods") and not isinstance(value, tuple):
                value = tuple(value)
            data[key] = value
        return Table2Config(**data)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "datasets": tuple(self.datasets), "ks": tuple(self.ks), "methods": tuple(self.methods),
            "epsilon": self.epsilon, "T": self.T, "repeats": self.repeats,
            "inner_epochs": self.inner_epochs, "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size, "num_workers": self.num_workers,
            "delta_init": self.delta_init, "delta_lower": self.delta_lower,
            "warm_start": self.warm_start, "group_size": self.group_size,
            "device": self.device, "seed": self.seed, "log_every": self.log_every,
            "output_dir": self.output_dir, "data_root": self.data_root,
            "save_artifacts": self.save_artifacts, "table3": self.table3,
            "f1_eval_split": self.f1_eval_split, "verbose": self.verbose,
        }


@dataclass
class CellResult:
    """Aggregated (mean +/- std) results of one ``(dataset, k)`` cell."""

    dataset: str
    k: int
    accuracy: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    coreset_size: Optional[Tuple[float, float]] = None
    accuracy_per_point: Dict[str, float] = field(default_factory=dict)
    repeats: int = 0
    failures: Dict[str, int] = field(default_factory=dict)

    def mean(self, method: str) -> Optional[float]:
        value = self.accuracy.get(method)
        return None if value is None else float(value[0])

    def size_mean(self) -> Optional[float]:
        return None if self.coreset_size is None else float(self.coreset_size[0])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "k": self.k,
            "accuracy": {m: {"mean": v[0], "std": v[1]} for m, v in self.accuracy.items()},
            "coreset_size": None if self.coreset_size is None
            else {"mean": self.coreset_size[0], "std": self.coreset_size[1]},
            "accuracy_per_point": dict(self.accuracy_per_point),
            "repeats": self.repeats,
            "failures": dict(self.failures),
        }


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------


def canonical_dataset(name: str) -> str:
    """Normalise a dataset name to the paper's spelling."""
    key = str(name).strip().lower().replace("_", "-").replace(" ", "")
    if key in ("f-mnist", "fmnist", "fashion-mnist", "fashionmnist", "fashion-mnist", "fashion"):
        return "F-MNIST"
    if key == "svhn":
        return "SVHN"
    if key in ("cifar-10", "cifar10", "cifar"):
        return "CIFAR-10"
    if key in ("mnist", "mnist-s", "mnists"):
        return "MNIST-S"
    return str(name)


def binarize_mask(mask: Any) -> np.ndarray:
    """Project a binary / relaxed ``[-1,1]`` / probability mask to ``{0,1}``.

    Appendix A projection rule: values in ``[-1, 0)`` become ``0`` and values in
    ``[0, 1]`` become ``1``.
    """
    if _TORCH_AVAILABLE and hasattr(mask, "detach"):
        arr = mask.detach().cpu().numpy()
    else:
        arr = np.asarray(mask)
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    if arr.size and bool(np.all((arr == 0.0) | (arr == 1.0))):
        return (arr > 0.5).astype(np.float32)
    return (arr >= 0.0).astype(np.float32)


def mask_indices(mask: Any) -> np.ndarray:
    """Indices of the examples selected by a (possibly relaxed) mask."""
    return np.flatnonzero(binarize_mask(mask) > 0.5)


def mask_size(mask: Any) -> int:
    """``f2(m) = ||m||_0`` for a binary or relaxed mask."""
    return int(mask_indices(mask).size)


def resolve_seed(seed: Optional[int], repeat: int = 0, base: int = 0) -> int:
    """Deterministic per-repeat seed (matches ``baselines.base.resolve_seed``)."""
    if seed is None:
        seed = base
    return int((int(seed) + 100003 * int(repeat)) % (2 ** 31 - 1))


def set_seed(seed: Optional[int]) -> None:
    """Seed NumPy and, when available, PyTorch/CUDA."""
    if seed is not None:
        np.random.seed(int(seed) % (2 ** 32))
    if _TORCH_AVAILABLE:
        try:
            torch.manual_seed(int(seed) if seed is not None else 0)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed) if seed is not None else 0)
        except Exception:  # pragma: no cover
            pass


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def _filter_kwargs(fn: Callable[..., Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only the keyword arguments accepted by ``fn``."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):  # builtins / C callables
        return dict(kwargs)
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _resolve_device(device: Optional[str]) -> Optional[str]:
    if device is not None:
        return device
    if _TORCH_AVAILABLE and torch.cuda.is_available():
        return "cuda"
    return "cpu" if _TORCH_AVAILABLE else None


def _model_factory_for(model_name: str, num_classes: int) -> Callable[..., Any]:
    """Return a fresh-model factory for the requested architecture name."""
    try:
        from lbcs_repro.models import model_factory

        return model_factory(model_name, num_classes=num_classes)
    except Exception:  # pragma: no cover - fallback path
        from lbcs_repro.models import (
            cifar_cnn_factory, convnet_factory, lenet_factory, resnet18_factory, svhn_cnn_factory,
        )

        table: Dict[str, Callable[..., Any]] = {
            "convnet": convnet_factory, "lenet": lenet_factory, "resnet18": resnet18_factory,
            "cifarcnn": cifar_cnn_factory, "svhncnn": svhn_cnn_factory,
        }
        key = str(model_name).lower().replace("-", "").replace("_", "")
        return table.get(key, convnet_factory)(num_classes=num_classes)


# --------------------------------------------------------------------------------------
# Per-benchmark context
# --------------------------------------------------------------------------------------


@dataclass
class DatasetContext:
    """Loaders, model factories and target recipe for one benchmark."""

    dataset: str
    selection_dataset: Any
    selection_loader: Any
    f1_eval_loader: Any
    test_loader: Any
    n: int
    num_classes: int
    targets: np.ndarray
    inner_spec: InnerTrainSpec
    target_config: TargetTrainConfig
    device: Optional[str] = None
    data_root: Optional[str] = None
    test_dataset: Any = None

    def inner_factory(self) -> Callable[..., Any]:
        return _model_factory_for(self.inner_spec.model, self.num_classes)

    def target_factory(self) -> Callable[..., Any]:
        return _model_factory_for(self.target_config.model, self.num_classes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset, "n": self.n, "num_classes": self.num_classes,
            "inner": self.inner_spec.to_dict(), "target_train": self.target_config.to_dict(),
        }


def build_dataset_context(
    dataset: str,
    config: Table2Config,
    repeat: int = 0,
    device: Optional[str] = None,
) -> DatasetContext:
    """Build loaders and model factories for one benchmark.

    Coreset selection happens on the *training* split; ``f1(m)`` is evaluated on
    the full-data loader (the training pool by default, or the test split when
    ``config.f1_eval_split == "test"``) and the reported measurement is the new
    target model's accuracy on the untouched test split.
    """
    from lbcs_repro.data.datasets import get_dataset, get_targets, make_loader, num_classes as _num_classes

    name = canonical_dataset(dataset)
    device = _resolve_device(device if device is not None else config.device)
    root = config.data_root

    train_ds = get_dataset(name, train=True, root=root, download=True)
    test_ds = get_dataset(name, train=False, root=root, download=True)

    targets = np.asarray(get_targets(train_ds))
    n = int(targets.size)
    n_cls = int(_num_classes(name))

    selection_loader = make_loader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, seed=resolve_seed(config.seed, repeat),
        drop_last=False, return_index=False,
    )
    f1_source = test_ds if str(config.f1_eval_split).lower() == "test" else train_ds
    f1_eval_loader = make_loader(
        f1_source, batch_size=config.eval_batch_size, shuffle=False, num_workers=config.num_workers,
    )
    test_loader = make_loader(
        test_ds, batch_size=config.eval_batch_size, shuffle=False, num_workers=config.num_workers,
    )

    return DatasetContext(
        dataset=name,
        selection_dataset=train_ds,
        selection_loader=selection_loader,
        f1_eval_loader=f1_eval_loader,
        test_loader=test_loader,
        n=n,
        num_classes=n_cls,
        targets=targets,
        inner_spec=InnerTrainSpec.for_dataset(name, epochs=config.inner_epochs, batch_size=config.batch_size),
        target_config=TargetTrainConfig.for_dataset(name),
        device=device,
        data_root=root,
        test_dataset=test_ds,
    )


def coreset_loader(
    context: DatasetContext,
    mask: Any,
    batch_size: int = 128,
    num_workers: int = 0,
    augment: bool = False,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> Any:
    """DataLoader over the examples selected by ``mask`` (the constructed coreset)."""
    from lbcs_repro.data.datasets import get_dataset, make_loader, subset_dataset

    indices = mask_indices(mask)
    if indices.size == 0:
        raise ValueError("coreset is empty: the mask selects no example")

    source = context.selection_dataset
    if augment:
        try:  # augmentation is reserved for post-selection target training
            source = get_dataset(context.dataset, train=True, root=context.data_root,
                                 download=True, augment=True)
        except Exception:  # pragma: no cover - best effort
            source = context.selection_dataset
    subset = subset_dataset(source, indices)
    return make_loader(
        subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        seed=seed, drop_last=False,
    )


# --------------------------------------------------------------------------------------
# Target-model training / evaluation
# --------------------------------------------------------------------------------------


def _forward_logits(model: Any, inputs: Any) -> Any:
    out = model(inputs)
    if isinstance(out, (tuple, list)):
        return out[0]
    return getattr(out, "logits", out)


def evaluate_accuracy(model: Any, loader: Any, device: Optional[str] = None) -> float:
    """Top-1 test accuracy (%) of ``model`` over ``loader``."""
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to evaluate a target model")
    device = _resolve_device(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            inputs = batch[0].to(device)
            targets = torch.as_tensor(batch[1]).to(device)
            logits = _forward_logits(model, inputs)
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            total += int(targets.numel())
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
    scheduler: Optional[str] = None,
    device: Optional[str] = None,
    verbose: bool = False,
    log_every: int = 0,
) -> Tuple[Any, List[float]]:
    """Post-selection target training (Section 5.2); returns model + accuracy history."""
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to train a target model")
    device = _resolve_device(device)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    opt_name = str(optimizer).lower()
    if opt_name == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name in ("sgd", "momentum"):
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"unsupported optimizer {optimizer!r}")

    sched = None
    if scheduler:
        sched_name = str(scheduler).lower()
        if sched_name == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(int(epochs), 1))
        elif sched_name == "step":
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(int(epochs) // 3, 1), gamma=0.1)

    history: List[float] = []
    for epoch in range(int(epochs)):
        model.train()
        for batch in train_loader:
            inputs = batch[0].to(device)
            targets = torch.as_tensor(batch[1]).to(device)
            opt.zero_grad()
            loss = criterion(_forward_logits(model, inputs), targets)
            loss.backward()
            opt.step()
        if sched is not None:
            sched.step()
        if test_loader is not None:
            acc = evaluate_accuracy(model, test_loader, device=device)
            history.append(float(acc))
            if verbose or (log_every and (epoch + 1) % int(log_every) == 0):
                LOGGER.info("%s: epoch %d/%d test acc %.2f", _model_name(model), epoch + 1, epochs, acc)
    return model, history


def _model_name(model: Any) -> str:
    return type(model).__name__ if model is not None else "model"


def train_and_evaluate(
    context: DatasetContext,
    mask: Any,
    seed: Optional[int] = None,
    config: Optional[Table2Config] = None,
    model: Any = None,
) -> Dict[str, Any]:
    """Train a fresh target model on the coreset ``mask`` and report test accuracy."""
    config = config or Table2Config()
    tc = context.target_config
    set_seed(seed)
    if model is None:
        model = context.target_factory()()
    loader = coreset_loader(
        context, mask, batch_size=tc.batch_size, num_workers=config.num_workers,
        augment=bool(tc.augment), shuffle=True, seed=seed,
    )
    model, history = train_target_model(
        model, loader, test_loader=context.test_loader,
        epochs=tc.epochs, lr=tc.lr, optimizer=tc.optimizer, momentum=tc.momentum,
        weight_decay=tc.weight_decay, scheduler=tc.scheduler, device=context.device,
        verbose=config.verbose, log_every=config.log_every,
    )
    acc = float(history[-1]) if history else float("nan")
    size = mask_size(mask)
    return {
        "accuracy": acc,
        "coreset_size": int(size),
        "accuracy_per_point": acc / max(size, 1) if not math.isnan(acc) else float("nan"),
        "history": history,
        "model": model,
    }


# --------------------------------------------------------------------------------------
# Coreset selection: LBCS and the Appendix D.1 baselines
# --------------------------------------------------------------------------------------


def select_lbcs_mask(
    context: DatasetContext,
    k: int,
    config: Table2Config,
    seed: Optional[int] = None,
    lbcs: Any = None,
) -> Dict[str, Any]:
    """Run Algorithm 1 (LBCS) for one ``(dataset, k)`` and return mask + diagnostics."""
    if lbcs is None:
        from lbcs_repro.lbcs.bilevel import LBCS, LBCSConfig, InnerTrainConfig

        spec = context.inner_spec
        inner = InnerTrainConfig.for_section52()
        inner.epochs = int(spec.epochs)
        inner.batch_size = int(spec.batch_size)
        inner.lr = float(spec.lr)
        inner.optimizer = str(spec.optimizer)
        inner.momentum = float(spec.momentum)
        inner.weight_decay = float(spec.weight_decay)

        lbcs_config = LBCSConfig.for_section52()
        lbcs_config.k = int(k)
        lbcs_config.epsilon = float(config.epsilon)
        lbcs_config.T = int(config.T)
        lbcs_config.delta_init = float(config.delta_init)
        lbcs_config.delta_lower = float(config.delta_lower)
        lbcs_config.warm_start = bool(config.warm_start)
        lbcs_config.group_size = int(config.group_size)
        lbcs_config.device = context.device
        lbcs_config.seed = seed

        kwargs: Dict[str, Any] = {
            "model_factory": context.inner_factory(),
            "n": int(context.n),
            "k": int(k),
            "dataset": context.selection_dataset,
            "train_loader": context.selection_loader,
            "eval_loader": context.f1_eval_loader,
            "inner_config": inner,
            "epsilon": float(config.epsilon),
            "T": int(config.T),
            "device": context.device,
            "seed": seed,
            "group_size": int(config.group_size),
            "warm_start": bool(config.warm_start),
            "delta_init": float(config.delta_init),
            "delta_lower": float(config.delta_lower),
            "log_every": int(config.log_every),
            "config": lbcs_config,
        }
        lbcs = LBCS(**_filter_kwargs(LBCS.__init__, kwargs))

    set_seed(seed)
    t0 = time.time()
    result = lbcs.run() if hasattr(lbcs, "run") else lbcs()
    wall = time.time() - t0

    mask = getattr(result, "mask", None)
    if mask is None:
        mask = getattr(result, "continuous_mask", None)
    if mask is None:
        raise RuntimeError("LBCS returned no mask")

    binarized = binarize_mask(mask)
    size = int(getattr(result, "size", 0) or mask_size(binarized))
    return {
        "mask": binarized,
        "coreset_size": size,
        "f1": float(getattr(result, "f1", float("nan"))),
        "f2": float(getattr(result, "f2", size)),
        "restarts": int(getattr(result, "restarts", 0) or 0),
        "wall_time": float(wall),
        "result": result,
    }


def _build_selector(name: str, context: DatasetContext, k: int, seed: Optional[int], model: Any) -> Any:
    """Instantiate a baseline selector by paper name, with a tolerant fallback."""
    from lbcs_repro.baselines import make_baseline

    ctor_kwargs: Dict[str, Any] = {
        "seed": seed, "device": context.device, "num_classes": context.num_classes,
        "model": model, "k": int(k), "n": int(context.n),
    }
    try:
        return make_baseline(name, **ctor_kwargs)
    except TypeError:
        return make_baseline(name, **_filter_kwargs(make_baseline, {}))


def select_baseline_mask(
    name: str,
    context: DatasetContext,
    k: int,
    config: Table2Config,
    seed: Optional[int] = None,
    model: Any = None,
) -> np.ndarray:
    """Select a fixed-size ``k`` coreset with one of the Appendix D.1 baselines."""
    from lbcs_repro.baselines.base import indices_to_mask, topk_indices

    n = int(context.n)
    targets = context.targets
    seed_now = resolve_seed(config.seed, 0) if seed is None else seed

    selector = None
    try:
        selector = _build_selector(name, context, k, seed_now, model)
    except Exception as exc:  # pragma: no cover - keep long sweeps alive
        LOGGER.warning("could not instantiate baseline %s (%s); using uniform fallback", name, exc)

    if selector is not None:
        select_fn = getattr(selector, "select_mask", None) or getattr(selector, "mask", None)
        if select_fn is not None:
            call_kwargs: Dict[str, Any] = {
                "n": n, "k": int(k), "dataset": context.selection_dataset, "targets": targets,
                "num_classes": context.num_classes, "seed": seed_now, "device": context.device,
                "model": model, "loader": context.selection_loader,
                "train_loader": context.selection_loader,
            }
            filtered = _filter_kwargs(select_fn, call_kwargs)
            filtered["n"] = n
            filtered["k"] = int(k)
            try:
                mask = select_fn(**filtered)
                if mask is not None:
                    return binarize_mask(mask).astype(np.float32)
            except Exception as exc:  # pragma: no cover
                LOGGER.debug("baseline %s select_mask failed (%s); trying score path", name, exc)

        # Score path: rank the baseline's own scores and keep the top-k.
        if hasattr(selector, "compute_scores"):
            score_kwargs = _filter_kwargs(
                selector.compute_scores,
                {
                    "dataset": context.selection_dataset, "targets": targets, "n": n, "seed": seed_now,
                    "model": model, "loader": context.selection_loader,
                    "train_loader": context.selection_loader,
                    "num_classes": context.num_classes, "device": context.device,
                },
            )
            try:
                scores = selector.compute_scores(**score_kwargs)
                if scores is not None:
                    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
                    if scores.size >= n:
                        return indices_to_mask(topk_indices(scores, int(k), largest=True), n).astype(np.float32)
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("baseline %s compute_scores failed (%s)", name, exc)

    from lbcs_repro.baselines.uniform import uniform_indices

    return indices_to_mask(uniform_indices(n, int(k), seed=seed_now), n).astype(np.float32)


# --------------------------------------------------------------------------------------
# Single-cell execution
# --------------------------------------------------------------------------------------


def run_single_cell(
    dataset: str,
    k: int,
    repeat: int,
    config: Table2Config,
    context: Optional[DatasetContext] = None,
    seed: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    select_baseline_fn: Optional[Callable[..., Any]] = None,
    select_lbcs_fn: Optional[Callable[..., Any]] = None,
    train_eval_fn: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Run every method once for one ``(dataset, k, repeat)`` triple (Table 2)."""
    logger = logger or LOGGER
    seed = resolve_seed(config.seed, repeat) if seed is None else seed
    name = canonical_dataset(dataset)
    context = context or build_dataset_context(name, config, repeat=repeat, device=config.device)

    select_baseline_fn = select_baseline_fn or select_baseline_mask
    select_lbcs_fn = select_lbcs_fn or select_lbcs_mask
    train_eval_fn = train_eval_fn or train_and_evaluate

    record: Dict[str, Any] = {
        "dataset": name, "k": int(k), "repeat": int(repeat), "seed": int(seed),
        "methods": {}, "failures": {}, "lbcs": None,
    }

    for method in config.methods:
        t0 = time.time()
        try:
            mask = select_baseline_fn(method, context, int(k), config, seed=seed)
            metrics = train_eval_fn(context, mask, seed=seed, config=config)
            record["methods"][method] = {
                "accuracy": float(metrics.get("accuracy", float("nan"))),
                "coreset_size": int(metrics.get("coreset_size", mask_size(mask))),
                "accuracy_per_point": float(metrics.get("accuracy_per_point", float("nan"))),
                "wall_time": float(time.time() - t0),
                "requested_size": int(k),
            }
        except Exception as exc:  # pragma: no cover - long sweeps must survive
            logger.warning("[%s k=%d rep=%d] baseline %s failed: %s", name, k, repeat, method, exc)
            record["failures"][method] = str(exc)

    # LBCS is evaluated at the same predefined size k, and additionally reports
    # the optimized size ||m||_0 (Table 2, last column).
    t0 = time.time()
    try:
        info = select_lbcs_fn(context, int(k), config, seed=seed)
        mask = info["mask"]
        metrics = train_eval_fn(context, mask, seed=seed, config=config)
        size = int(info.get("coreset_size", mask_size(mask)))
        record["lbcs"] = {
            "accuracy": float(metrics.get("accuracy", float("nan"))),
            "coreset_size": size,
            "accuracy_per_point": float(metrics.get("accuracy_per_point", float("nan"))),
            "f1": float(info.get("f1", float("nan"))),
            "f2": float(info.get("f2", size)),
            "restarts": int(info.get("restarts", 0) or 0),
            "wall_time": float(time.time() - t0),
            "requested_size": int(k),
        }
    except Exception as exc:  # pragma: no cover
        logger.warning("[%s k=%d rep=%d] LBCS failed: %s", name, k, repeat, exc)
        record["failures"][LBCS_LABEL] = str(exc)

    return record


def run_single_cell_table3(
    dataset: str,
    k: int,
    lbcs_size: int,
    repeat: int,
    config: Table2Config,
    context: Optional[DatasetContext] = None,
    seed: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    select_baseline_fn: Optional[Callable[..., Any]] = None,
    train_eval_fn: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Run the baselines at the LBCS-achieved size (Table 3, second comparison)."""
    logger = logger or LOGGER
    seed = resolve_seed(config.seed, repeat) if seed is None else seed
    name = canonical_dataset(dataset)
    context = context or build_dataset_context(name, config, repeat=repeat, device=config.device)

    select_baseline_fn = select_baseline_fn or select_baseline_mask
    train_eval_fn = train_eval_fn or train_and_evaluate

    record: Dict[str, Any] = {
        "dataset": name, "k": int(k), "lbcs_size": int(lbcs_size), "repeat": int(repeat),
        "seed": int(seed), "methods": {}, "failures": {}, "lbcs": None,
    }
    for method in config.methods:
        try:
            mask = select_baseline_fn(method, context, int(lbcs_size), config, seed=seed)
            metrics = train_eval_fn(context, mask, seed=seed, config=config)
            record["methods"][method] = {
                "accuracy": float(metrics.get("accuracy", float("nan"))),
                "coreset_size": int(metrics.get("coreset_size", mask_size(mask))),
                "accuracy_per_point": float(metrics.get("accuracy_per_point", float("nan"))),
                "requested_size": int(lbcs_size),
            }
        except Exception as exc:  # pragma: no cover
            logger.warning("[%s k=%d size=%d rep=%d] baseline %s failed: %s",
                           name, k, lbcs_size, repeat, method, exc)
            record["failures"][method] = str(exc)
    return record


# --------------------------------------------------------------------------------------
# Aggregation / formatting
# --------------------------------------------------------------------------------------


def aggregate_results(records: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, int], CellResult]:
    """Group per-repeat records into mean +/- std cells."""
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for rec in records:
        grouped.setdefault((canonical_dataset(rec["dataset"]), int(rec["k"])), []).append(rec)

    cells: Dict[Tuple[str, int], CellResult] = {}
    for key, recs in grouped.items():
        dataset, k = key
        cell = CellResult(dataset=dataset, k=k, repeats=len(recs))
        accuracy: Dict[str, List[float]] = {}
        per_point: Dict[str, List[float]] = {}
        sizes: List[float] = []
        failures: Dict[str, int] = {}

        for rec in recs:
            for method, values in rec.get("methods", {}).items():
                accuracy.setdefault(method, []).append(float(values["accuracy"]))
                per_point.setdefault(method, []).append(float(values.get("accuracy_per_point", float("nan"))))
            lbcs = rec.get("lbcs")
            if lbcs:
                accuracy.setdefault(LBCS_LABEL, []).append(float(lbcs["accuracy"]))
                per_point.setdefault(LBCS_LABEL, []).append(float(lbcs.get("accuracy_per_point", float("nan"))))
                sizes.append(float(lbcs["coreset_size"]))
            for method in rec.get("failures", {}):
                failures[method] = failures.get(method, 0) + 1

        for method, values in accuracy.items():
            cell.accuracy[method] = _mean_std(values)
            pts = np.asarray(per_point.get(method, []), dtype=np.float64)
            pts = pts[~np.isnan(pts)]
            cell.accuracy_per_point[method] = float(pts.mean()) if pts.size else float("nan")
        if sizes:
            cell.coreset_size = _mean_std(sizes)
        cell.failures = failures
        cells[key] = cell
    return cells


def _ordered_methods(cells: Dict[Tuple[str, int], CellResult], methods: Sequence[str]) -> List[str]:
    present: List[str] = []
    for method in methods:
        if any(method in cell.accuracy for cell in cells.values()):
            present.append(method)
    if any(LBCS_LABEL in cell.accuracy for cell in cells.values()):
        present.append(LBCS_LABEL)
    return present


def _sorted_datasets(cells: Dict[Tuple[str, int], CellResult]) -> List[str]:
    datasets = sorted({ds for ds, _ in cells})
    return sorted(datasets, key=lambda d: (PAPER_DATASETS.index(d) if d in PAPER_DATASETS else 99, d))


def _render(rows: List[List[str]], title: str) -> str:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = [title]
    lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(rows[0])))
    lines.append("  ".join("-" * w for w in widths))
    for row in rows[1:]:
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
    return "\n".join(lines) + "\n"


def format_table2(cells: Dict[Tuple[str, int], CellResult], config: Optional[Table2Config] = None) -> str:
    """Render Table 2 (same predefined ``k`` for every method)."""
    methods = _ordered_methods(cells, METHOD_ORDER)
    if not methods:
        return "Table 2: (no results)\n"
    include_size = any(cell.coreset_size is not None for cell in cells.values())
    header = ["Dataset", "k"] + methods + ([SIZE_LABEL] if include_size else [])
    rows = [header]
    for dataset in _sorted_datasets(cells):
        for k in sorted(k for ds, k in cells if ds == dataset):
            cell = cells[(dataset, k)]
            row = [dataset, str(k)]
            for method in methods:
                value = cell.accuracy.get(method)
                row.append("n/a" if value is None else f"{value[0]:.1f} +/- {value[1]:.1f}")
            if include_size:
                size = cell.coreset_size
                row.append("n/a" if size is None else f"{size[0]:.1f} +/- {size[1]:.1f}")
            rows.append(row)
    return _render(
        rows,
        "Table 2: Mean and standard deviation of test accuracy (%) with various predefined coreset sizes.",
    )


def format_table3(
    cells: Dict[Tuple[str, int], CellResult],
    lbcs_sizes: Optional[Dict[Tuple[str, int], int]] = None,
    config: Optional[Table2Config] = None,
) -> str:
    """Render Table 3 (coreset sizes achieved by LBCS applied to the baselines)."""
    lbcs_sizes = lbcs_sizes or {}
    methods = _ordered_methods(cells, METHOD_ORDER)
    if not methods:
        return "Table 3: (no results)\n"
    rows = [["Dataset", "k (LBCS size)"] + methods]
    for dataset in _sorted_datasets(cells):
        for k in sorted(k for ds, k in cells if ds == dataset):
            cell = cells[(dataset, k)]
            size = int(lbcs_sizes.get((dataset, k), k))
            row = [dataset, str(size)]
            for method in methods:
                value = cell.accuracy.get(method)
                row.append("n/a" if value is None else f"{value[0]:.1f} +/- {value[1]:.1f}")
            rows.append(row)
    return _render(
        rows,
        "Table 3: Mean and standard deviation of test accuracy (%) with coreset sizes achieved by LBCS.",
    )


def format_accuracy_per_point(cells: Dict[Tuple[str, int], CellResult]) -> str:
    """Render the Appendix E.1 average accuracy-brought-by-per-data-point view."""
    methods = _ordered_methods(cells, METHOD_ORDER)
    if not methods:
        return "Appendix E.1: (no results)\n"
    rows = [["Dataset", "k"] + methods]
    for dataset in _sorted_datasets(cells):
        for k in sorted(k for ds, k in cells if ds == dataset):
            cell = cells[(dataset, k)]
            row = [dataset, str(k)]
            for method in methods:
                value = cell.accuracy_per_point.get(method, float("nan"))
                row.append("n/a" if math.isnan(value) else f"{value:.4f}")
            rows.append(row)
    return _render(
        rows,
        "Appendix E.1: Average accuracy (%) brought by per data point within the selected coreset.",
    )


# --------------------------------------------------------------------------------------
# Validation of the paper's claims
# --------------------------------------------------------------------------------------


def direction_checks(cells: Dict[Tuple[str, int], CellResult]) -> Dict[str, Any]:
    """Check the qualitative Section 5.2 claims for every reproduced cell."""
    checks: Dict[str, Any] = {}
    for (dataset, k), cell in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        size = cell.size_mean()
        lbcs_acc = cell.mean(LBCS_LABEL)
        best_name, best_value = None, None
        for method in METHOD_ORDER:
            value = cell.mean(method)
            if value is None:
                continue
            if best_value is None or value > best_value:
                best_name, best_value = method, value
        paper_ref = PAPER_TABLE2.get(dataset, {}).get(k, {})
        checks[f"{dataset}@{k}"] = {
            "lbcs_accuracy": lbcs_acc,
            "best_baseline": best_value,
            "best_baseline_name": best_name,
            "lbcs_beats_best_baseline": None
            if (lbcs_acc is None or best_value is None) else bool(lbcs_acc >= best_value),
            "lbcs_competitive": None
            if (lbcs_acc is None or best_value is None) else bool(lbcs_acc >= best_value - 2.0),
            "optimized_size": size,
            "size_below_k": None if size is None else bool(size <= float(k)),
            "paper_lbcs_accuracy": paper_ref.get(LBCS_LABEL, (None, None))[0],
            "paper_size": paper_ref.get(SIZE_LABEL, (None, None))[0],
        }
    measured = [v for key, v in checks.items() if key != "summary"]
    checks["summary"] = {
        "num_cells": len(measured),
        "size_reduced": int(sum(1 for v in measured if v["size_below_k"])),
        "beats_best": int(sum(1 for v in measured if v["lbcs_beats_best_baseline"])),
        "competitive": int(sum(1 for v in measured if v["lbcs_competitive"])),
    }
    return checks


# --------------------------------------------------------------------------------------
# Artifact saving
# --------------------------------------------------------------------------------------


def _serialise_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rec in records:
        clean = {k: v for k, v in rec.items() if k not in ("lbcs",)}
        clean["methods"] = {
            m: {k: v for k, v in values.items() if k != "mask"}
            for m, values in rec.get("methods", {}).items()
        }
        lbcs = rec.get("lbcs")
        if lbcs:
            clean["lbcs"] = {k: v for k, v in lbcs.items() if k != "mask"}
        out.append(clean)
    return out


def save_results(
    cells: Dict[Tuple[str, int], CellResult],
    records: Sequence[Dict[str, Any]],
    checks: Dict[str, Any],
    config: Table2Config,
    table3_cells: Optional[Dict[Tuple[str, int], CellResult]] = None,
    lbcs_sizes: Optional[Dict[Tuple[str, int], int]] = None,
) -> Dict[str, str]:
    """Write JSON / CSV / TXT artifacts under ``config.output_dir``."""
    os.makedirs(config.output_dir, exist_ok=True)
    artifacts: Dict[str, str] = {}

    with open(os.path.join(config.output_dir, "table2.txt"), "w", encoding="utf-8") as handle:
        handle.write(format_table2(cells, config))
    artifacts["table2_txt"] = os.path.join(config.output_dir, "table2.txt")

    with open(os.path.join(config.output_dir, "appendix_e1_per_point.txt"), "w", encoding="utf-8") as handle:
        handle.write(format_accuracy_per_point(cells))
    artifacts["per_point_txt"] = os.path.join(config.output_dir, "appendix_e1_per_point.txt")

    with open(os.path.join(config.output_dir, "table2.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": config.to_dict(),
                "cells": [cell.to_dict() for cell in cells.values()],
                "checks": checks,
            },
            handle, indent=2, default=str,
        )
    artifacts["table2_json"] = os.path.join(config.output_dir, "table2.json")

    methods = _ordered_methods(cells, METHOD_ORDER)
    with open(os.path.join(config.output_dir, "table2.csv"), "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "k"] + [f"{m}_mean" for m in methods] + [f"{m}_std" for m in methods]
                         + ["lbcs_size_mean", "lbcs_size_std"])
        for (dataset, k), cell in sorted(cells.items()):
            means = ["" if cell.accuracy.get(m) is None else f"{cell.accuracy[m][0]:.4f}" for m in methods]
            stds = ["" if cell.accuracy.get(m) is None else f"{cell.accuracy[m][1]:.4f}" for m in methods]
            size = cell.coreset_size
            writer.writerow([dataset, k] + means + stds
                            + (["", ""] if size is None else [f"{size[0]:.2f}", f"{size[1]:.2f}"]))
    artifacts["table2_csv"] = os.path.join(config.output_dir, "table2.csv")

    raw_path = os.path.join(config.output_dir, "table2_raw.jsonl")
    with open(raw_path, "w", encoding="utf-8") as handle:
        for rec in _serialise_records(records):
            handle.write(json.dumps(rec, default=str) + "\n")
    artifacts["table2_raw"] = raw_path

    if table3_cells:
        sizes = lbcs_sizes or {}
        with open(os.path.join(config.output_dir, "table3.txt"), "w", encoding="utf-8") as handle:
            handle.write(format_table3(table3_cells, sizes, config))
        artifacts["table3_txt"] = os.path.join(config.output_dir, "table3.txt")
        with open(os.path.join(config.output_dir, "table3.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "lbcs_sizes": {f"{ds}@{k}": int(v) for (ds, k), v in sizes.items()},
                    "cells": [cell.to_dict() for cell in table3_cells.values()],
                },
                handle, indent=2, default=str,
            )
        artifacts["table3_json"] = os.path.join(config.output_dir, "table3.json")

    return artifacts


# --------------------------------------------------------------------------------------
# Top-level drivers
# --------------------------------------------------------------------------------------


def run_table2(
    config: Optional[Table2Config] = None,
    cell_runner: Optional[Callable[..., Dict[str, Any]]] = None,
    logger: Optional[logging.Logger] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Run Table 2 (first kind of comparison) for every dataset / ``k`` / repeat.

    ``cell_runner(dataset=..., k=..., repeat=..., config=..., seed=...)`` may be
    injected to replace the expensive real pipeline, which is how the offline
    self-test validates aggregation and reporting.
    """
    logger = logger or LOGGER
    config = (config or Table2Config.paper()).with_overrides(**overrides)
    logger.info("Table 2: datasets=%s ks=%s repeats=%d epsilon=%.2f T=%d",
                list(config.datasets), list(config.ks), config.repeats, config.epsilon, config.T)

    records: List[Dict[str, Any]] = []
    for dataset in config.datasets:
        for k in config.ks:
            for repeat in range(int(config.repeats)):
                seed = resolve_seed(config.seed, repeat)
                t0 = time.time()
                if cell_runner is not None:
                    rec = cell_runner(dataset=dataset, k=int(k), repeat=int(repeat), config=config, seed=seed)
                else:
                    rec = run_single_cell(dataset, int(k), int(repeat), config, seed=seed, logger=logger)
                rec.setdefault("dataset", canonical_dataset(dataset))
                rec.setdefault("k", int(k))
                rec.setdefault("repeat", int(repeat))
                rec.setdefault("wall_time", float(time.time() - t0))
                records.append(rec)
                logger.info("[%s k=%d rep=%d] finished in %.1fs", rec["dataset"], k, repeat, rec["wall_time"])

    cells = aggregate_results(records)
    checks = direction_checks(cells)
    lbcs_sizes: Dict[Tuple[str, int], int] = {
        key: int(round(cell.coreset_size[0]))
        for key, cell in cells.items() if cell.coreset_size is not None
    }

    artifacts: Dict[str, str] = {}
    if config.save_artifacts:
        artifacts = save_results(cells, records, checks, config, lbcs_sizes=lbcs_sizes)

    return {
        "table2": format_table2(cells, config),
        "cells": cells,
        "records": records,
        "checks": checks,
        "lbcs_sizes": lbcs_sizes,
        "per_point": format_accuracy_per_point(cells),
        "config": config,
        "artifacts": artifacts,
    }


def run_table3(
    config: Optional[Table2Config] = None,
    lbcs_sizes: Optional[Dict[Tuple[str, int], int]] = None,
    lbcs_accuracies: Optional[Dict[Tuple[str, int], float]] = None,
    cell_runner: Optional[Callable[..., Dict[str, Any]]] = None,
    logger: Optional[logging.Logger] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Run Table 3 (second kind of comparison).

    The coreset size obtained by LBCS is applied to every baseline; their
    selection and target training then start from that size, and only the test
    accuracy is compared.  LBCS accuracies are carried over from Table 2 since
    the LBCS coreset (and hence its trained model) is unchanged.
    """
    logger = logger or LOGGER
    config = (config or Table2Config.paper()).with_overrides(**overrides)

    if lbcs_sizes is None or lbcs_accuracies is None:
        table2 = run_table2(config=config, cell_runner=cell_runner, logger=logger)
        lbcs_sizes = lbcs_sizes or table2["lbcs_sizes"]
        lbcs_accuracies = lbcs_accuracies or {
            key: cell.accuracy[LBCS_LABEL][0]
            for key, cell in table2["cells"].items() if LBCS_LABEL in cell.accuracy
        }
    lbcs_sizes = lbcs_sizes or {}
    lbcs_accuracies = lbcs_accuracies or {}

    records: List[Dict[str, Any]] = []
    for dataset in config.datasets:
        for k in config.ks:
            key = (canonical_dataset(dataset), int(k))
            size = int(lbcs_sizes.get(key, k))
            for repeat in range(int(config.repeats)):
                seed = resolve_seed(config.seed, repeat)
                if cell_runner is not None:
                    rec = cell_runner(dataset=dataset, k=int(k), repeat=int(repeat), config=config,
                                      seed=seed, lbcs_size=size)
                else:
                    rec = run_single_cell_table3(dataset, int(k), size, int(repeat), config,
                                                 seed=seed, logger=logger)
                records.append(rec)
                logger.info("[%s k=%d size=%d rep=%d] table-3 cell done", key[0], k, size, repeat)

    cells = aggregate_results(records)
    for key, acc in lbcs_accuracies.items():
        if key in cells:
            cells[key].accuracy[LBCS_LABEL] = (float(acc), 0.0)

    artifacts: Dict[str, str] = {}
    if config.save_artifacts:
        os.makedirs(config.output_dir, exist_ok=True)
        path = os.path.join(config.output_dir, "table3.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(format_table3(cells, lbcs_sizes, config))
        artifacts["table3_txt"] = path

    return {
        "table3": format_table3(cells, lbcs_sizes, config),
        "cells": cells,
        "records": records,
        "lbcs_sizes": lbcs_sizes,
        "config": config,
        "artifacts": artifacts,
    }


def run_table2_table3(
    config: Optional[Table2Config] = None,
    logger: Optional[logging.Logger] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Run both kinds of comparison (Tables 2 and 3) and return all artifacts."""
    config = (config or Table2Config.paper()).with_overrides(**overrides)
    table2 = run_table2(config=config, logger=logger)
    summary: Dict[str, Any] = {
        "table2": table2["table2"],
        "cells": table2["cells"],
        "records": table2["records"],
        "checks": table2["checks"],
        "lbcs_sizes": table2["lbcs_sizes"],
        "per_point": table2["per_point"],
        "config": config,
        "artifacts": dict(table2["artifacts"]),
    }
    if config.table3:
        table3 = run_table3(
            config=config,
            lbcs_sizes=table2["lbcs_sizes"],
            lbcs_accuracies={
                key: cell.accuracy[LBCS_LABEL][0]
                for key, cell in table2["cells"].items() if LBCS_LABEL in cell.accuracy
            },
            logger=logger,
        )
        summary["table3"] = table3["table3"]
        summary["table3_cells"] = table3["cells"]
        summary["artifacts"].update(table3["artifacts"])
    return summary


#: Entry-point aliases used by the experiment registry / CLI.
run = run_table2_table3
table2_table3_compare = run_table2_table3


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LBCS Section 5.2 / Tables 2-3 driver")
    parser.add_argument("--paper", action="store_true", help="paper protocol (10 repeats, all datasets)")
    parser.add_argument("--smoke", action="store_true", help="quick single-repeat smoke run")
    parser.add_argument("--selftest", action="store_true", help="offline self-test (no torch, no data)")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--ks", nargs="+", type=int, default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--inner-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--table3", action="store_true", help="also run Table 3")
    parser.add_argument("--no-save", action="store_true", help="do not write artifacts")
    parser.add_argument("--config", default=None, help="path to a YAML config file")
    return parser


def _config_from_args(args: argparse.Namespace) -> Table2Config:
    overrides: Dict[str, Any] = {}
    if args.config:
        try:
            import yaml  # type: ignore

            with open(args.config, "r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
            block = loaded.get("table2", loaded)
            overrides.update(Table2Config.from_dict(block).to_dict())
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("could not load config %s: %s", args.config, exc)

    for name, value in (
        ("datasets", args.datasets), ("ks", args.ks), ("methods", args.methods),
        ("epsilon", args.epsilon), ("T", args.T), ("repeats", args.repeats),
        ("inner_epochs", args.inner_epochs), ("batch_size", args.batch_size),
        ("device", args.device), ("seed", args.seed), ("output_dir", args.output_dir),
    ):
        if value is not None:
            overrides[name] = value
    if args.table3:
        overrides["table3"] = True
    if args.no_save:
        overrides["save_artifacts"] = False

    base = Table2Config.smoke() if args.smoke else Table2Config.paper()
    return base.with_overrides(**overrides)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.selftest:
        report = _selftest(verbose=True)
        return 0 if report.get("ok") else 1

    result = run_table2_table3(config=_config_from_args(args))
    print(result["table2"])
    if "table3" in result:
        print(result["table3"])
    return 0


# --------------------------------------------------------------------------------------
# Offline self-test
# --------------------------------------------------------------------------------------


def _selftest(verbose: bool = True) -> Dict[str, Any]:
    """Offline validation of masks, aggregation, formatting, checks and drivers."""
    report: Dict[str, Any] = {"ok": True, "checks": {}}

    def check(name: str, condition: bool, detail: Any = None) -> None:
        report["checks"][name] = {"ok": bool(condition), "detail": detail}
        if not condition:
            report["ok"] = False

    # --- masks / sizes -------------------------------------------------------------
    mask = np.zeros(10, dtype=np.float32)
    mask[[1, 3, 5]] = 1.0
    check("mask_size", mask_size(mask) == 3, mask_size(mask))
    check("mask_indices", list(mask_indices(mask)) == [1, 3, 5], mask_indices(mask).tolist())
    relaxed = np.array([-1.0, -1e-12, 0.0, 1e-12, 1.0], dtype=np.float32)
    check("binarize_projection", list(binarize_mask(relaxed)) == [0.0, 0.0, 1.0, 1.0, 1.0],
          binarize_mask(relaxed).tolist())

    # --- aggregation ---------------------------------------------------------------
    records = []
    for repeat in range(4):
        records.append({
            "dataset": "F-MNIST", "k": 1000, "repeat": repeat, "seed": repeat,
            "methods": {
                "Uniform": {"accuracy": 70.0 + repeat, "coreset_size": 1000,
                            "accuracy_per_point": 0.07, "requested_size": 1000},
                "Moderate": {"accuracy": 75.0 + repeat, "coreset_size": 1000,
                             "accuracy_per_point": 0.075, "requested_size": 1000},
            },
            "lbcs": {"accuracy": 79.0 + repeat, "coreset_size": 950 + repeat,
                     "accuracy_per_point": 0.083, "f1": 1.0, "f2": float(950 + repeat),
                     "requested_size": 1000},
            "failures": {},
        })
    cells = aggregate_results(records)
    cell = cells[("F-MNIST", 1000)]
    check("aggregate_uniform_mean", abs(cell.mean("Uniform") - 71.5) < 1e-9, cell.mean("Uniform"))
    check("aggregate_lbcs_mean", abs(cell.mean(LBCS_LABEL) - 80.5) < 1e-9, cell.mean(LBCS_LABEL))
    check("aggregate_size_mean", abs(cell.size_mean() - 951.5) < 1e-9, cell.size_mean())
    check("aggregate_repeats", cell.repeats == 4, cell.repeats)
    check("aggregate_std_nonzero", cell.accuracy["Uniform"][1] > 0, cell.accuracy["Uniform"][1])
    check("aggregate_per_point", abs(cell.accuracy_per_point[LBCS_LABEL] - 0.083) < 1e-9,
          cell.accuracy_per_point[LBCS_LABEL])

    # --- formatting ----------------------------------------------------------------
    table2_text = format_table2(cells)
    check("format_table2", "Table 2" in table2_text and "Uniform" in table2_text, None)
    check("format_table2_size_column", SIZE_LABEL.split(" ")[0] in table2_text, None)
    table3_text = format_table3(cells, {("F-MNIST", 1000): 950})
    check("format_table3", "Table 3" in table3_text and "950" in table3_text, None)
    check("format_per_point", "per data point" in format_accuracy_per_point(cells), None)

    # --- qualitative checks --------------------------------------------------------
    checks = direction_checks(cells)
    entry = checks["F-MNIST@1000"]
    check("direction_size_below_k", entry["size_below_k"] is True, entry)
    check("direction_beats_best", entry["lbcs_beats_best_baseline"] is True, entry)
    check("direction_summary", checks["summary"]["num_cells"] == 1, checks["summary"])

    # --- configuration -------------------------------------------------------------
    cfg = Table2Config.paper(repeats=3, ks=[1000, 2000])
    check("config_overrides", cfg.repeats == 3 and tuple(cfg.ks) == (1000, 2000), cfg.to_dict())
    check("config_epsilon", abs(cfg.epsilon - PAPER_EPSILON) < 1e-12, cfg.epsilon)
    check("config_T", cfg.T == PAPER_T, cfg.T)
    restored = Table2Config.from_dict(cfg.to_dict())
    check("config_from_dict", tuple(restored.ks) == tuple(cfg.ks) and restored.repeats == cfg.repeats,
          restored.to_dict())
    smoke = Table2Config.smoke()
    check("config_smoke", smoke.repeats == 1 and smoke.table3 is False, smoke.to_dict())
    check("target_cifar", TargetTrainConfig.for_dataset("CIFAR-10").epochs == 200
          and TargetTrainConfig.for_dataset("CIFAR-10").optimizer == "sgd"
          and TargetTrainConfig.for_dataset("CIFAR-10").scheduler == "cosine",
          TargetTrainConfig.for_dataset("CIFAR-10").to_dict())
    check("target_fmnist", TargetTrainConfig.for_dataset("F-MNIST").optimizer == "adam"
          and TargetTrainConfig.for_dataset("F-MNIST").epochs == 100,
          TargetTrainConfig.for_dataset("F-MNIST").to_dict())
    check("target_svhn", TargetTrainConfig.for_dataset("SVHN").optimizer == "adam"
          and TargetTrainConfig.for_dataset("SVHN").epochs == 100,
          TargetTrainConfig.for_dataset("SVHN").to_dict())
    check("inner_adam_lr", abs(InnerTrainSpec.for_dataset("F-MNIST").lr - 0.001) < 1e-12
          and InnerTrainSpec.for_dataset("F-MNIST").model == "LeNet",
          InnerTrainSpec.for_dataset("F-MNIST").to_dict())

    # --- reference tables / helpers -------------------------------------------------
    check("paper_table2_complete",
          all(k in PAPER_TABLE2[ds] for ds in PAPER_DATASETS for k in PAPER_KS),
          {ds: sorted(PAPER_TABLE2[ds]) for ds in PAPER_DATASETS})
    check("paper_table3_present", all(ds in PAPER_TABLE3 for ds in PAPER_DATASETS), sorted(PAPER_TABLE3))
    check("dataset_aliases", canonical_dataset("fashion_mnist") == "F-MNIST"
          and canonical_dataset("cifar10") == "CIFAR-10"
          and canonical_dataset("SVHN") == "SVHN", None)
    check("resolve_seed_deterministic", resolve_seed(0, 3) == resolve_seed(0, 3), resolve_seed(0, 3))
    check("mean_std_single", _mean_std([5.0]) == (5.0, 0.0), _mean_std([5.0]))

    # --- end-to-end dry runs with injected synthetic runners ------------------------
    def fake_cell_runner(dataset: str, k: int, repeat: int, config: Table2Config, seed: int, **_: Any) -> Dict[str, Any]:
        rng = np.random.default_rng(seed)
        methods = {
            method: {
                "accuracy": float(60.0 + 5.0 * rng.random()),
                "coreset_size": int(k),
                "accuracy_per_point": float((60.0 + 5.0 * rng.random()) / max(int(k), 1)),
                "requested_size": int(k),
            }
            for method in config.methods
        }
        return {
            "dataset": canonical_dataset(dataset), "k": int(k), "repeat": int(repeat), "seed": int(seed),
            "methods": methods,
            "lbcs": {
                "accuracy": float(70.0 + 5.0 * rng.random()), "coreset_size": int(k) - 3,
                "accuracy_per_point": 70.0 / max(int(k) - 3, 1), "f1": 1.0, "f2": float(int(k) - 3),
                "requested_size": int(k),
            },
            "failures": {},
        }

    dry = run_table2(config=Table2Config.smoke(save_artifacts=False), cell_runner=fake_cell_runner)
    check("run_table2_records", len(dry["records"]) == 1, len(dry["records"]))
    check("run_table2_table", "Table 2" in dry["table2"], None)
    check("run_table2_lbcs_sizes", dry["lbcs_sizes"][("F-MNIST", 1000)] == 997, dry["lbcs_sizes"])

    dry3 = run_table3(
        config=Table2Config.smoke(save_artifacts=False, table3=True),
        lbcs_sizes={("F-MNIST", 1000): 997},
        lbcs_accuracies={("F-MNIST", 1000): 75.0},
        cell_runner=fake_cell_runner,
    )
    check("run_table3_table", "Table 3" in dry3["table3"] and "997" in dry3["table3"], None)
    check("run_table3_lbcs_carried",
          abs(dry3["cells"][("F-MNIST", 1000)].mean(LBCS_LABEL) - 75.0) < 1e-9,
          dry3["cells"][("F-MNIST", 1000)].mean(LBCS_LABEL))

    if verbose:
        print("table2_table3_compare self-test")
        for name, info in report["checks"].items():
            print(f"  [{'ok ' if info['ok'] else 'FAIL'}] {name}")
        print(f"overall: {'ok' if report['ok'] else 'FAILED'}")
    return report


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
