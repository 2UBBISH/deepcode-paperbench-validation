"""Dataset statistics and reference metadata for the SMM target tasks.

This module is the single source of truth for the numbers reported in the
SMM paper (ICML 2024), transcribed from:

* Appendix C, Table 6 -- detailed dataset information (original image size,
  training/testing set sizes and number of classes for the 11 target tasks),
* Appendix C, Table 9 -- per-dataset batch size and mask-generator training
  hyper-parameters (milestones, initial learning rate, decay) for the 5-layer
  (ResNet-18/ResNet-50) and 6-layer (ViT-B32) mask generators,
* Section 5, Tables 1/2/3 and Appendix D, Table 10 -- reference accuracies
  used to check reproduced results.

It also offers utilities to *measure* the statistics of the datasets that are
actually materialised on disk (sizes, number of classes, class distribution)
and to compare them against the paper's reference values.

Nothing in this module requires a GPU and torchvision datasets are only built
lazily, so importing it is cheap and safe from configuration scripts.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    # Table 6 metadata
    "TABLE6_DATASET_INFO",
    "DatasetInfo",
    "dataset_info",
    "original_image_size",
    "reference_train_size",
    "reference_test_size",
    "reference_num_classes",
    "reference_total_size",
    "DATASET_STATS",
    # Table 9 training parameters
    "TABLE9_TRAINING_PARAMS",
    "TrainingParams",
    "training_params",
    "milestones_for",
    "BATCH_SIZES",
    # Reference result tables
    "TABLE1_RESNET",
    "TABLE1_RESNET18",
    "TABLE1_RESNET50",
    "TABLE1_AVERAGES",
    "TABLE2_VIT_B32",
    "TABLE2_AVERAGES",
    "TABLE3_ABLATIONS",
    "TABLE3_AVERAGES",
    "TABLE6_MAIN_DATASETS",
    "TABLE10_LABEL_MAPPINGS",
    "TABLE10_AVERAGES",
    "TABLE11_SCALING",
    "TABLE12_STANFORDCARS",
    "TABLE13_LORA",
    "TABLE14_FINETUNE_FC",
    "REFERENCE_TABLES",
    # Measurement utilities
    "DatasetStatistics",
    "compute_dataset_statistics",
    "dataset_statistics",
    "class_distribution",
    "summarise_datasets",
    "format_statistics_table",
    "verify_dataset_statistics",
    "compare_with_reference",
    "dump_statistics_json",
    # Aggregate helpers
    "paper_summary",
]

Tensor = Any  # torch.Tensor, kept untyped to avoid importing torch here.


# ---------------------------------------------------------------------------
# Appendix C, Table 6 -- Detailed Dataset Information
# ---------------------------------------------------------------------------
# NOTE: the original image sizes are 32x32 (CIFAR10/CIFAR100/SVHN/GTSRB) and
# 128x128 for the remaining nine tasks.  Training/testing sizes are the ones
# used by Chen et al. (2023) and reported in Table 6.

TABLE6_DATASET_INFO: Dict[str, Tuple[int, int, int, int]] = {
    # canonical name: (original image size, train size, test size, num classes)
    "cifar10": (32, 50000, 10000, 10),
    "cifar100": (32, 50000, 10000, 100),
    "svhn": (32, 73257, 26032, 10),
    "gtsrb": (32, 39209, 12630, 43),
    "flowers102": (128, 4093, 2463, 102),
    "dtd": (128, 2820, 1692, 47),
    "ucf101": (128, 7639, 3783, 101),
    "food101": (128, 50500, 30300, 101),
    "sun397": (128, 15888, 19850, 397),
    "eurosat": (128, 13500, 8100, 10),
    "oxfordpets": (128, 2944, 3669, 37),
    # Appendix D.4 failure case (not part of the 11 main tasks)
    "stanfordcars": (128, 8144, 8041, 196),
}

#: Order used in every table of the paper (Table 1, 2, 3 and 10).
TABLE6_MAIN_DATASETS: Tuple[str, ...] = (
    "cifar10",
    "cifar100",
    "svhn",
    "gtsrb",
    "flowers102",
    "dtd",
    "ucf101",
    "food101",
    "sun397",
    "eurosat",
    "oxfordpets",
)

#: Pretty names used in the paper's tables.
DISPLAY_NAMES: Dict[str, str] = {
    "cifar10": "CIFAR10",
    "cifar100": "CIFAR100",
    "svhn": "SVHN",
    "gtsrb": "GTSRB",
    "flowers102": "Flowers102",
    "dtd": "DTD",
    "ucf101": "UCF101",
    "food101": "Food101",
    "sun397": "SUN397",
    "eurosat": "EuroSAT",
    "oxfordpets": "OxfordPets",
    "stanfordcars": "StanfordCars",
}

#: Aliases tolerated when users pass dataset names around.
_ALIASES: Dict[str, str] = {
    "cifar-10": "cifar10",
    "cifar_10": "cifar10",
    "cifar10": "cifar10",
    "cifar-100": "cifar100",
    "cifar_100": "cifar100",
    "cifar100": "cifar100",
    "svhn": "svhn",
    "gtsrb": "gtsrb",
    "flowers102": "flowers102",
    "flowers-102": "flowers102",
    "oxford_flowers102": "flowers102",
    "dtd": "dtd",
    "ucf101": "ucf101",
    "food101": "food101",
    "sun397": "sun397",
    "eurosat": "eurosat",
    "oxfordpets": "oxfordpets",
    "oxford-iiit-pet": "oxfordpets",
    "oxford_pets": "oxfordpets",
    "stanfordcars": "stanfordcars",
    "stanford_cars": "stanfordcars",
    "cars": "stanfordcars",
}


def _canonical(name: str) -> str:
    """Normalise a dataset name/alias to its canonical Table 6 key."""
    key = str(name).strip().lower().replace(" ", "").replace("-", "").replace("_", "").replace(".", "")
    if key in _ALIASES:
        return _ALIASES[key]
    if key in TABLE6_DATASET_INFO:
        return key
    # fuzzy fallback: strip separators again from alias keys
    for alias, canon in _ALIASES.items():
        if alias.replace("-", "").replace("_", "") == key:
            return canon
    raise KeyError(f"Unknown dataset {name!r}; known: {sorted(TABLE6_DATASET_INFO)}")


@dataclass(frozen=True)
class DatasetInfo:
    """Static description of one target task (Appendix C, Table 6)."""

    name: str
    original_size: int
    train_size: int
    test_size: int
    num_classes: int

    # -- convenience -------------------------------------------------------
    @property
    def display_name(self) -> str:
        return DISPLAY_NAMES.get(self.name, self.name)

    @property
    def total_size(self) -> int:
        return self.train_size + self.test_size

    @property
    def original_shape(self) -> Tuple[int, int]:
        return (self.original_size, self.original_size)

    @property
    def is_low_resolution(self) -> bool:
        """True for 32x32 tasks (used for the finetuning comparison, App. E)."""
        return self.original_size <= 32

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.name,
            "display_name": self.display_name,
            "original_image_size": self.original_size,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "num_classes": self.num_classes,
        }


def dataset_info(name: str) -> DatasetInfo:
    """Return the :class:`DatasetInfo` for ``name`` (Table 6)."""
    canon = _canonical(name)
    size, train, test, classes = TABLE6_DATASET_INFO[canon]
    return DatasetInfo(
        name=canon,
        original_size=int(size),
        train_size=int(train),
        test_size=int(test),
        num_classes=int(classes),
    )


def original_image_size(name: str) -> int:
    return dataset_info(name).original_size


def reference_train_size(name: str) -> int:
    return dataset_info(name).train_size


def reference_test_size(name: str) -> int:
    return dataset_info(name).test_size


def reference_num_classes(name: str) -> int:
    return dataset_info(name).num_classes


def reference_total_size(name: str) -> int:
    return dataset_info(name).total_size


def get_original_size(name: str) -> int:
    """Alias of :func:`original_image_size` (matches ``datasets.dataset_image_size``)."""
    return original_image_size(name)


#: Mapping form of Table 6 for quick programmatic access.
DATASET_STATS: Dict[str, Dict[str, Any]] = {
    name: dataset_info(name).as_dict() for name in TABLE6_DATASET_INFO
}


# ---------------------------------------------------------------------------
# Appendix C, Table 9 -- Training parameter settings of the mask generator
# ---------------------------------------------------------------------------
# Batch sizes: 256 everywhere except DTD and OxfordPets (64).
# Milestones are [0, 100, 145] in the paper; the leading 0 is an artifact of the
# table (the schedule is the 100th/145th epoch milestones of Chen et al., 2023)
# and is dropped here.  The optimizer is not specified by the paper: the
# engine defaults to SGD with these learning rates and milestones.

DEFAULT_MILESTONES: Tuple[int, ...] = (100, 145)
DEFAULT_EPOCHS: int = 200


@dataclass(frozen=True)
class TrainingParams:
    """Per-dataset training parameters (Appendix C, Table 9)."""

    name: str
    batch_size: int
    milestones: Tuple[int, ...]
    # 5-layer mask generator (ResNet-18 / ResNet-50)
    alpha_5layer: float
    gamma_5layer: float
    # 6-layer mask generator (ViT-B32)
    alpha_6layer: float
    gamma_6layer: float
    epochs: int = DEFAULT_EPOCHS

    @property
    def display_name(self) -> str:
        return DISPLAY_NAMES.get(self.name, self.name)

    def alpha(self, num_layers: int = 5) -> float:
        return self.alpha_5layer if num_layers < 6 else self.alpha_6layer

    def gamma(self, num_layers: int = 5) -> float:
        return self.gamma_5layer if num_layers < 6 else self.gamma_6layer

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.name,
            "batch_size": self.batch_size,
            "milestones": list(self.milestones),
            "epochs": self.epochs,
            "alpha_5layer": self.alpha_5layer,
            "gamma_5layer": self.gamma_5layer,
            "alpha_6layer": self.alpha_6layer,
            "gamma_6layer": self.gamma_6layer,
        }


#: Table 9 transposed into a per-dataset dictionary.  ``(b, milestones, a5, g5, a6, g6)``
_TABLE9_RAW: Dict[str, Tuple[int, Tuple[int, ...], float, float, float, float]] = {
    "cifar10": (256, DEFAULT_MILESTONES, 0.01, 0.1, 0.001, 1.0),
    "cifar100": (256, DEFAULT_MILESTONES, 0.01, 0.1, 0.001, 1.0),
    "svhn": (256, DEFAULT_MILESTONES, 0.01, 0.1, 0.001, 1.0),
    "gtsrb": (256, DEFAULT_MILESTONES, 0.01, 0.1, 0.001, 1.0),
    "flowers102": (256, DEFAULT_MILESTONES, 0.01, 0.1, 0.001, 1.0),
    "dtd": (64, DEFAULT_MILESTONES, 0.01, 0.1, 0.001, 1.0),
    "ucf101": (256, DEFAULT_MILESTONES, 0.01, 0.1, 0.001, 1.0),
    "food101": (256, DEFAULT_MILESTONES, 0.01, 0.1, 0.001, 1.0),
    "sun397": (256, DEFAULT_MILESTONES, 0.01, 0.1, 0.001, 1.0),
    "eurosat": (256, DEFAULT_MILESTONES, 0.01, 0.1, 0.001, 1.0),
    "oxfordpets": (64, DEFAULT_MILESTONES, 0.01, 0.1, 0.001, 1.0),
    # StanfordCars is only used for the failure-case table; training details
    # follow the same protocol as the other 128x128 tasks.
    "stanfordcars": (256, DEFAULT_MILESTONES, 0.01, 0.1, 0.001, 1.0),
}

#: Initial learning rate of the shared pattern ``delta`` (Chen et al., 2023).
DEFAULT_ALPHA_DELTA: float = 0.01
DEFAULT_GAMMA_DELTA: float = 0.1

#: Batch size used for every dataset (with the DTD/OxfordPets exceptions).
DEFAULT_BATCH_SIZE: int = 256


def training_params(name: str) -> TrainingParams:
    """Return the Table 9 training parameters for ``name``."""
    canon = _canonical(name)
    b, milestones, a5, g5, a6, g6 = _TABLE9_RAW[canon]
    return TrainingParams(
        name=canon,
        batch_size=int(b),
        milestones=tuple(int(m) for m in milestones),
        alpha_5layer=float(a5),
        gamma_5layer=float(g5),
        alpha_6layer=float(a6),
        gamma_6layer=float(g6),
    )


def milestones_for(name: str) -> Tuple[int, ...]:
    return training_params(name).milestones


def batch_size_for(name: str) -> int:
    return training_params(name).batch_size


def alpha_for(name: str, num_layers: int = 5) -> float:
    return training_params(name).alpha(num_layers)


def gamma_for(name: str, num_layers: int = 5) -> float:
    return training_params(name).gamma(num_layers)


#: Table 9 batch sizes only, keyed by canonical dataset name.
BATCH_SIZES: Dict[str, int] = {name: v.as_dict()["batch_size"] for name, v in
                               ((n, training_params(n)) for n in TABLE6_DATASET_INFO)}

#: Full Table 9 mapping name -> dict.
TABLE9_TRAINING_PARAMS: Dict[str, Dict[str, Any]] = {
    name: training_params(name).as_dict() for name in TABLE6_DATASET_INFO
}


# ---------------------------------------------------------------------------
# Reference results reported in the paper (used for verification/reporting)
# ---------------------------------------------------------------------------

#: Table 2 -- ViT-B32 (ImageNet-1K) accuracy (%) per method.
#: Columns: Pad, Narrow, Medium, Full, Ours (SMM).
TABLE2_VIT_B32: Dict[str, Dict[str, float]] = {
    "cifar10": {"pad": 62.4, "narrow": 96.6, "medium": 96.5, "full": 95.8, "ours": 97.4},
    "cifar100": {"pad": 31.6, "narrow": 74.4, "medium": 75.3, "full": 75.0, "ours": 82.6},
    "svhn": {"pad": 80.2, "narrow": 85.0, "medium": 87.4, "full": 87.8, "ours": 89.7},
    "gtsrb": {"pad": 62.3, "narrow": 57.8, "medium": 68.6, "full": 75.5, "ours": 80.5},
    "flowers102": {"pad": 57.3, "narrow": 55.3, "medium": 56.6, "full": 55.9, "ours": 79.1},
    "dtd": {"pad": 43.7, "narrow": 37.3, "medium": 38.5, "full": 37.7, "ours": 45.6},
    "ucf101": {"pad": 33.6, "narrow": 44.5, "medium": 44.8, "full": 40.9, "ours": 42.6},
    "food101": {"pad": 37.4, "narrow": 47.3, "medium": 48.6, "full": 49.4, "ours": 64.8},
    "sun397": {"pad": 21.8, "narrow": 29.0, "medium": 29.4, "full": 28.8, "ours": 36.7},
    "eurosat": {"pad": 95.9, "narrow": 90.9, "medium": 90.9, "full": 89.1, "ours": 93.5},
    "oxfordpets": {"pad": 57.6, "narrow": 82.5, "medium": 81.0, "full": 75.3, "ours": 83.8},
}

TABLE2_AVERAGES: Dict[str, float] = {
    "pad": 53.1, "narrow": 63.7, "medium": 65.2, "full": 64.7, "ours": 72.4,
}

#: Table 1 -- ResNet-18/ResNet-50 accuracy (%) per method.
#: Columns: Pad, Narrow, Medium, Full, Ours (SMM).
TABLE1_RESNET18: Dict[str, Dict[str, float]] = {
    "cifar10": {"pad": 65.6, "narrow": 68.7, "medium": 65.3, "full": 68.9, "ours": 72.8},
    "cifar100": {"pad": 31.0, "narrow": 30.4, "medium": 33.5, "full": 33.8, "ours": 39.4},
    "svhn": {"pad": 68.9, "narrow": 43.2, "medium": 49.6, "full": 78.3, "ours": 84.4},
    "gtsrb": {"pad": 70.2, "narrow": 63.6, "medium": 60.3, "full": 76.8, "ours": 80.4},
    "flowers102": {"pad": 20.8, "narrow": 19.0, "medium": 22.0, "full": 23.2, "ours": 38.7},
    "dtd": {"pad": 35.0, "narrow": 29.0, "medium": 27.2, "full": 29.0, "ours": 33.6},
    "ucf101": {"pad": 27.0, "narrow": 22.2, "medium": 25.3, "full": 24.4, "ours": 28.7},
    "food101": {"pad": 10.9, "narrow": 9.9, "medium": 11.3, "full": 13.2, "ours": 17.5},
    "sun397": {"pad": 15.2, "narrow": 10.5, "medium": 12.1, "full": 13.4, "ours": 16.0},
    "eurosat": {"pad": 80.6, "narrow": 57.4, "medium": 66.2, "full": 84.3, "ours": 92.2},
    "oxfordpets": {"pad": 57.4, "narrow": 45.8, "medium": 54.8, "full": 70.0, "ours": 74.1},
}

TABLE1_RESNET50: Dict[str, Dict[str, float]] = {
    "cifar10": {"pad": 68.4, "narrow": 74.7, "medium": 70.5, "full": 75.1, "ours": 77.3},
    "cifar100": {"pad": 35.7, "narrow": 35.2, "medium": 38.3, "full": 38.6, "ours": 44.0},
    "svhn": {"pad": 79.4, "narrow": 56.5, "medium": 58.6, "full": 81.6, "ours": 84.7},
    "gtsrb": {"pad": 64.8, "narrow": 60.4, "medium": 63.6, "full": 75.1, "ours": 82.3},
    "flowers102": {"pad": 22.9, "narrow": 20.9, "medium": 22.2, "full": 25.5, "ours": 41.6},
    "dtd": {"pad": 36.1, "narrow": 34.0, "medium": 31.7, "full": 32.6, "ours": 39.1},
    "ucf101": {"pad": 29.3, "narrow": 24.4, "medium": 26.5, "full": 27.6, "ours": 31.4},
    "food101": {"pad": 9.8, "narrow": 9.6, "medium": 10.6, "full": 11.4, "ours": 22.9},
    "sun397": {"pad": 12.7, "narrow": 12.5, "medium": 12.7, "full": 14.7, "ours": 17.3},
    "eurosat": {"pad": 82.1, "narrow": 58.4, "medium": 68.3, "full": 85.7, "ours": 94.0},
    "oxfordpets": {"pad": 68.0, "narrow": 51.0, "medium": 62.3, "full": 72.7, "ours": 84.9},
}

TABLE1_RESNET: Dict[str, Dict[str, Dict[str, float]]] = {
    "resnet18": TABLE1_RESNET18,
    "resnet50": TABLE1_RESNET50,
}

TABLE1_AVERAGES: Dict[str, Dict[str, float]] = {
    "resnet18": {"pad": 43.91, "narrow": 43.48, "medium": 45.04, "full": 46.85, "ours": 52.53},
    "resnet50": {"pad": 49.15, "narrow": 46.76, "medium": 49.39, "full": 52.10, "ours": 56.35},
}

#: Table 3 -- ablation on the masking strategy (ResNet-18, mean +- std %).
#: Columns: only delta, only f_mask, single-channel f_mask^s, ours.
TABLE3_ABLATIONS: Dict[str, Dict[str, float]] = {
    "cifar10": {"only_delta": 68.9, "only_fmask": 59.0, "single_channel": 72.6, "ours": 72.8},
    "cifar100": {"only_delta": 33.8, "only_fmask": 32.1, "single_channel": 38.0, "ours": 39.4},
    "svhn": {"only_delta": 78.3, "only_fmask": 51.1, "single_channel": 78.4, "ours": 84.4},
    "gtsrb": {"only_delta": 76.8, "only_fmask": 55.7, "single_channel": 70.7, "ours": 80.4},
    "flowers102": {"only_delta": 23.2, "only_fmask": 32.2, "single_channel": 30.2, "ours": 38.7},
    "dtd": {"only_delta": 29.0, "only_fmask": 27.2, "single_channel": 32.7, "ours": 33.6},
    "ucf101": {"only_delta": 24.4, "only_fmask": 25.7, "single_channel": 28.0, "ours": 28.7},
    "food101": {"only_delta": 13.2, "only_fmask": 13.3, "single_channel": 15.8, "ours": 17.5},
    "sun397": {"only_delta": 13.4, "only_fmask": 10.5, "single_channel": 15.9, "ours": 16.0},
    "eurosat": {"only_delta": 84.3, "only_fmask": 89.2, "single_channel": 90.6, "ours": 92.2},
    "oxfordpets": {"only_delta": 70.0, "only_fmask": 72.5, "single_channel": 73.8, "ours": 74.1},
}

#: Table 3 standard deviations (%) for the same runs.
TABLE3_ABLATION_STD: Dict[str, Dict[str, float]] = {
    "cifar10": {"only_delta": 0.4, "only_fmask": 1.6, "single_channel": 2.6, "ours": 0.7},
    "cifar100": {"only_delta": 0.2, "only_fmask": 0.3, "single_channel": 0.6, "ours": 0.6},
    "svhn": {"only_delta": 0.3, "only_fmask": 3.1, "single_channel": 0.2, "ours": 2.0},
    "gtsrb": {"only_delta": 0.9, "only_fmask": 1.2, "single_channel": 0.8, "ours": 1.2},
    "flowers102": {"only_delta": 0.5, "only_fmask": 0.4, "single_channel": 0.4, "ours": 0.7},
    "dtd": {"only_delta": 0.7, "only_fmask": 0.5, "single_channel": 0.5, "ours": 0.4},
    "ucf101": {"only_delta": 0.9, "only_fmask": 0.3, "single_channel": 0.3, "ours": 0.8},
    "food101": {"only_delta": 0.1, "only_fmask": 0.1, "single_channel": 0.1, "ours": 0.1},
    "sun397": {"only_delta": 0.2, "only_fmask": 0.1, "single_channel": 0.1, "ours": 0.3},
    "eurosat": {"only_delta": 0.5, "only_fmask": 0.9, "single_channel": 0.5, "ours": 0.2},
    "oxfordpets": {"only_delta": 0.6, "only_fmask": 0.3, "single_channel": 0.6, "ours": 0.4},
}

TABLE3_AVERAGES: Dict[str, float] = {
    "only_delta": 46.85, "only_fmask": 42.59, "single_channel": 49.70, "ours": 52.53,
}

#: Table 10 -- improvement of SMM when applied on top of different f_out.
#: ``label_mapping`` ->  {"without_smm": ..., "with_smm": [...], "improve": [...]}
TABLE10_LABEL_MAPPINGS: Dict[str, Dict[str, List[float]]] = {
    "ilm": {
        "without_smm": [68.90, 33.80, 78.30, 76.80, 23.20, 29.00, 24.40, 13.20, 13.40, 84.30, 70.00],
        "with_smm": [72.80, 39.40, 84.40, 80.40, 38.70, 33.60, 28.00, 17.50, 16.00, 92.20, 74.10],
        "improve": [3.90, 5.60, 6.10, 3.60, 15.50, 4.60, 4.30, 4.30, 2.60, 7.90, 4.10],
    },
    "flm": {
        "without_smm": [71.79, 29.79, 78.78, 74.76, 17.78, 30.14, 22.71, 11.58, 13.45, 86.00, 69.66],
        "with_smm": [72.75, 32.35, 83.73, 80.90, 32.16, 34.28, 25.72, 15.21, 15.45, 92.67, 72.83],
        "improve": [0.96, 2.56, 4.95, 6.14, 14.37, 4.14, 3.01, 3.62, 1.99, 6.67, 3.16],
    },
    "rlm": {
        "without_smm": [65.68, 16.99, 77.44, 69.60, 12.34, 14.00, 9.04, 7.15, 1.05, 84.49, 8.89],
        "with_smm": [69.71, 23.47, 85.37, 82.38, 37.68, 19.74, 16.71, 15.86, 3.35, 94.47, 16.84],
        "improve": [4.03, 6.48, 7.92, 12.79, 25.33, 5.14, 7.67, 8.71, 2.29, 9.98, 7.96],
    },
}

TABLE10_AVERAGES: Dict[str, Dict[str, float]] = {
    "ilm": {"without_smm": 46.85, "with_smm": 52.53, "improve": 5.68},
    "flm": {"without_smm": 46.04, "with_smm": 50.73, "improve": 4.69},
    "rlm": {"without_smm": 33.39, "with_smm": 42.32, "improve": 8.94},
}

#: Appendix D.3, Table 11 -- scaling study of the mask generator
#: (EuroSAT + ResNet-18).  ``channels`` are the intermediate widths.
TABLE11_SCALING: Dict[str, Dict[str, Any]] = {
    "baseline": {"channels": [8, 16, 32, 64], "num_params": 26499, "accuracy": 92.2},
    "x2": {"channels": [16, 32, 64, 128], "num_params": None, "accuracy": None},
    "x4": {"channels": [32, 64, 128, 256], "num_params": None, "accuracy": None},
    "x8": {"channels": [64, 128, 256, 512], "num_params": None, "accuracy": None},
}

#: Appendix D.4, Table 12 -- StanfordCars failure case (all methods < 10 %).
TABLE12_STANFORDCARS: Dict[str, float] = {
    "pad": None, "narrow": None, "medium": None, "full": None, "ours": None,
    "note": "All VR methods remain below 10% accuracy; SMM does not rescue this task.",
}

#: Appendix E.1, Table 13 -- SMM vs LoRA for ViT-Large (rank 6, lr 0.01, 10 epochs).
TABLE13_LORA: Dict[str, Dict[str, float]] = {}
#: Appendix E.2, Table 14 -- Finetuning-FC with/without SMM.
TABLE14_FINETUNE_FC: Dict[str, Dict[str, float]] = {}

#: Registry to iterate over reference tables from reporting code.
REFERENCE_TABLES: Dict[str, Any] = {
    "table1_resnet18": TABLE1_RESNET18,
    "table1_resnet50": TABLE1_RESNET50,
    "table2_vit_b32": TABLE2_VIT_B32,
    "table3_ablations": TABLE3_ABLATIONS,
    "table6_datasets": DATASET_STATS,
    "table9_training": TABLE9_TRAINING_PARAMS,
    "table10_label_mappings": TABLE10_LABEL_MAPPINGS,
}

#: Reference average accuracies (%).  Useful for automated verification.
REFERENCE_AVERAGES: Dict[str, Any] = {
    "resnet18": TABLE1_AVERAGES["resnet18"],
    "resnet50": TABLE1_AVERAGES["resnet50"],
    "vit_b32": TABLE2_AVERAGES,
    "ablations": TABLE3_AVERAGES,
    "label_mappings": TABLE10_AVERAGES,
}


# ---------------------------------------------------------------------------
# Measured dataset statistics
# ---------------------------------------------------------------------------

@dataclass
class DatasetStatistics:
    """Statistics measured from a materialised dataset (not the paper values)."""

    name: str
    train_size: int
    test_size: int
    num_classes: int
    original_size: Optional[int] = None
    train_class_counts: Optional[Dict[int, int]] = None
    test_class_counts: Optional[Dict[int, int]] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_size(self) -> int:
        return self.train_size + self.test_size

    @property
    def display_name(self) -> str:
        return DISPLAY_NAMES.get(self.name, self.name)

    def as_dict(self, include_distribution: bool = False) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "dataset": self.name,
            "display_name": self.display_name,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "num_classes": self.num_classes,
        }
        if self.original_size is not None:
            out["original_image_size"] = self.original_size
        if include_distribution:
            out["train_class_counts"] = self.train_class_counts
            out["test_class_counts"] = self.test_class_counts
        if self.extra:
            out.update(self.extra)
        return out

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"DatasetStatistics(name={self.name!r}, train={self.train_size}, "
            f"test={self.test_size}, classes={self.num_classes})"
        )


def _num_classes_of(dataset: Any, fallback: Optional[int] = None) -> int:
    """Best-effort extraction of the number of classes from a dataset object."""
    for attr in ("classes", ):
        value = getattr(dataset, attr, None)
        if value is not None and not callable(value):
            try:
                return len(value)
            except TypeError:
                pass
    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = getattr(dataset, "labels", None)
    if targets is not None:
        try:
            return len(set(int(t) for t in targets))
        except TypeError:
            pass
    if fallback is not None:
        return int(fallback)
    spec_classes = None
    name = getattr(dataset, "name", None)
    if isinstance(name, str):
        try:
            spec_classes = reference_num_classes(name)
        except KeyError:
            spec_classes = None
    if spec_classes is not None:
        return int(spec_classes)
    return -1


def class_distribution(dataset: Any, max_classes: Optional[int] = None) -> Dict[int, int]:
    """Count samples per class in ``dataset``.

    Uses ``targets``/``labels`` when available (torchvision convention) and
    falls back to iterating the dataset otherwise.  ``max_classes`` truncates
    the returned dictionary (useful for datasets with ~400 classes).
    """
    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = getattr(dataset, "labels", None)
    if targets is None:
        targets = getattr(dataset, "_labels", None)

    if targets is None:
        counts: Dict[int, int] = {}
        for index in range(len(dataset)):
            item = dataset[index]
            label = item[1] if isinstance(item, (tuple, list)) and len(item) > 1 else None
            if label is None:
                break
            label = int(label)
            counts[label] = counts.get(label, 0) + 1
        if max_classes is not None:
            counts = dict(sorted(counts.items())[:max_classes])
        return counts

    counts = {}
    for label in targets:
        # one-hot rows (e.g. some torchvision datasets) are supported
        if hasattr(label, "argmax") and not isinstance(label, int):
            label = int(label.argmax())
        else:
            label = int(label)
        counts[label] = counts.get(label, 0) + 1
    if max_classes is not None:
        counts = dict(sorted(counts.items())[:max_classes])
    return counts


def compute_dataset_statistics(
    name: str,
    train_dataset: Any,
    test_dataset: Any,
    *,
    include_distribution: bool = True,
    original_size: Optional[int] = None,
) -> DatasetStatistics:
    """Measure the statistics of a concrete train/test dataset pair."""
    canon = _canonical(name)
    info = dataset_info(canon)

    train_counts = None
    test_counts = None
    num_classes = None
    if include_distribution:
        try:
            train_counts = class_distribution(train_dataset)
            test_counts = class_distribution(test_dataset)
            num_classes = len(set(train_counts) | set(test_counts))
        except Exception:  # pragma: no cover - defensive
            train_counts = test_counts = None
    if not num_classes:
        num_classes = _num_classes_of(train_dataset, fallback=info.num_classes)

    return DatasetStatistics(
        name=canon,
        train_size=len(train_dataset),
        test_size=len(test_dataset),
        num_classes=int(num_classes) if num_classes and num_classes > 0 else info.num_classes,
        original_size=original_size if original_size is not None else info.original_size,
        train_class_counts=train_counts,
        test_class_counts=test_counts,
    )


def dataset_statistics(
    name: str,
    *,
    root: Optional[str] = None,
    data_root: Optional[str] = None,
    backbone: Optional[str] = None,
    imgsize: Optional[int] = None,
    download: bool = False,
    include_distribution: bool = True,
    train_fraction: Optional[float] = None,
    split_seed: int = 0,
) -> DatasetStatistics:
    """Build a dataset (lazily importing :mod:`smm_vr.data.datasets`) and measure it."""
    from .datasets import build_datasets  # local import: avoids circular import

    train_ds, test_ds, _spec = build_datasets(
        name,
        backbone=backbone,
        root=root,
        data_root=data_root,
        imgsize=imgsize,
        download=download,
        train_fraction=train_fraction,
        split_seed=split_seed,
    )
    return compute_dataset_statistics(
        name,
        train_ds,
        test_ds,
        include_distribution=include_distribution,
    )


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def summarise_datasets(
    names: Optional[Sequence[str]] = None,
    *,
    reference: bool = True,
) -> List[Dict[str, Any]]:
    """Return a list of per-dataset summary dictionaries.

    With ``reference=True`` (default) the values come from Appendix C, Table 6
    and no dataset has to be downloaded; this is what config/report code needs.
    """
    if names is None:
        names = list(TABLE6_MAIN_DATASETS)
    rows: List[Dict[str, Any]] = []
    for name in names:
        canon = _canonical(name)
        if reference:
            row = dataset_info(canon).as_dict()
            row.update(training_params(canon).as_dict())
        else:
            stats = dataset_statistics(canon)
            row = stats.as_dict()
        rows.append(row)
    return rows


def format_statistics_table(rows: Optional[Iterable[Mapping[str, Any]]] = None) -> str:
    """Render dataset statistics as a fixed-width text table (Table 6 style)."""
    if rows is None:
        rows = summarise_datasets()
    rows = list(rows)
    header = (
        f"{'Dataset':<13}{'OrigSize':>9}{'Train':>9}{'Test':>8}"
        f"{'Classes':>9}{'Batch':>7}{'Alpha5':>8}{'Gamma5':>8}{'Alpha6':>8}{'Gamma6':>8}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{str(row.get('display_name', row.get('dataset'))):<13}"
            f"{int(row.get('original_image_size', 0)):>9}"
            f"{int(row.get('train_size', 0)):>9}"
            f"{int(row.get('test_size', 0)):>8}"
            f"{int(row.get('num_classes', 0)):>9}"
            f"{int(row.get('batch_size', 0)):>7}"
            f"{float(row.get('alpha_5layer', 0.0)):>8.4g}"
            f"{float(row.get('gamma_5layer', 0.0)):>8.4g}"
            f"{float(row.get('alpha_6layer', 0.0)):>8.4g}"
            f"{float(row.get('gamma_6layer', 0.0)):>8.4g}"
        )
    return "\n".join(lines)


def verify_dataset_statistics(
    name: str,
    stats: DatasetStatistics,
    *,
    tolerance: float = 0.02,
    raise_on_mismatch: bool = False,
) -> Dict[str, Any]:
    """Compare measured statistics against Table 6 within ``tolerance``.

    Returns a dictionary with the reference and measured values, relative
    errors and a boolean ``matches`` flag (sizes within ``tolerance`` and an
    exact match of the number of classes).
    """
    info = dataset_info(name)
    result: Dict[str, Any] = {
        "dataset": info.name,
        "reference": info.as_dict(),
        "measured": stats.as_dict(),
    }

    def _rel(measured: int, reference: int) -> float:
        if reference == 0:
            return 0.0 if measured == 0 else float("inf")
        return abs(measured - reference) / float(reference)

    train_rel = _rel(stats.train_size, info.train_size)
    test_rel = _rel(stats.test_size, info.test_size)
    classes_ok = int(stats.num_classes) == int(info.num_classes)
    sizes_ok = train_rel <= tolerance and test_rel <= tolerance

    result.update(
        {
            "train_rel_error": train_rel,
            "test_rel_error": test_rel,
            "classes_match": classes_ok,
            "sizes_within_tolerance": sizes_ok,
            "tolerance": tolerance,
            "matches": bool(sizes_ok and classes_ok),
        }
    )
    if raise_on_mismatch and not result["matches"]:
        raise AssertionError(
            f"Dataset {info.name}: measured (train={stats.train_size}, "
            f"test={stats.test_size}, classes={stats.num_classes}) deviates from "
            f"Table 6 (train={info.train_size}, test={info.test_size}, "
            f"classes={info.num_classes}) beyond tolerance {tolerance}."
        )
    return result


def compare_with_reference(
    results: Mapping[str, Mapping[str, float]],
    *,
    table: str = "table2_vit_b32",
    key: str = "ours",
    verbose: bool = False,
) -> Dict[str, Any]:
    """Compare measured accuracies with a reference table from the paper.

    ``results`` maps canonical dataset names to ``{method: accuracy}`` (or a
    plain number, in which case ``key`` selects the method column of the
    reference table).  Returns per-dataset deltas and the average delta.
    """
    if table.startswith("table1"):
        backbone = table.replace("table1_", "")
        reference = TABLE1_RESNET[backbone]
    elif table.startswith("table2"):
        reference = TABLE2_VIT_B32
    elif table.startswith("table3"):
        reference = TABLE3_ABLATIONS
    else:
        raise KeyError(f"Unknown reference table {table!r}")

    per_dataset: Dict[str, Any] = {}
    deltas: List[float] = []
    for name, value in results.items():
        canon = _canonical(name)
        if canon not in reference or key not in reference[canon]:
            continue
        expected = float(reference[canon][key])
        measured = value.get(key) if isinstance(value, Mapping) else float(value)
        if measured is None:
            continue
        measured = float(measured)
        delta = measured - expected
        per_dataset[canon] = {
            "expected": expected,
            "measured": measured,
            "delta": delta,
        }
        deltas.append(delta)

    summary = {
        "table": table,
        "key": key,
        "per_dataset": per_dataset,
        "mean_delta": (sum(deltas) / len(deltas)) if deltas else None,
        "mean_abs_delta": (sum(abs(d) for d in deltas) / len(deltas)) if deltas else None,
        "num_datasets": len(deltas),
    }
    if verbose:  # pragma: no cover - convenience
        for name, row in per_dataset.items():
            print(f"{name:<13} expected={row['expected']:6.2f} "
                  f"measured={row['measured']:6.2f} delta={row['delta']:+.2f}")
        if summary["mean_delta"] is not None:
            print(f"mean delta: {summary['mean_delta']:+.2f}")
    return summary


def dump_statistics_json(path: str, rows: Optional[Iterable[Mapping[str, Any]]] = None) -> str:
    """Write dataset statistics (Table 6 + Table 9) to ``path`` as JSON."""
    if rows is None:
        rows = summarise_datasets()
    rows = list(rows)
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, sort_keys=False)
    return path


def paper_summary() -> Dict[str, Any]:
    """Compact summary of every reference table stored in this module."""
    return {
        "datasets": DATASET_STATS,
        "training": TABLE9_TRAINING_PARAMS,
        "averages": REFERENCE_AVERAGES,
        "num_main_datasets": len(TABLE6_MAIN_DATASETS),
        "datasets_order": list(TABLE6_MAIN_DATASETS),
    }


if __name__ == "__main__":  # pragma: no cover - manual inspection
    print(format_statistics_table())
    print()
    print("Reference averages:")
    for key, value in REFERENCE_AVERAGES.items():
        print(f"  {key}: {value}")
