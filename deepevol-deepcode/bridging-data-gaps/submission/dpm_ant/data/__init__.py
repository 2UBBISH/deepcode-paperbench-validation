"""Data pipeline for DPMs-ANT (few-shot adaptation data & evaluation sets).

This package initializer re-exports the public data-loading API from
:mod:`dpm_ant.data.datasets` (and, when present, :mod:`dpm_ant.data.prepare_data`).

Heavy imports (torch, PIL, ...) are performed lazily via PEP 562 so that
``import dpm_ant.data`` stays cheap and works in partial checkouts.
"""

from __future__ import annotations

import importlib
from typing import Dict, List

__all__: List[str] = [
    # registries
    "SOURCE_DATASETS",
    "TARGET_DATASETS",
    "FID_TARGET_SETS",
    "DATASET_ALIASES",
    # dataset classes
    "ImageFolderDataset",
    "FewShotDataset",
    "TensorDataset",
    "LoopedDataset",
    # factory / loader helpers
    "build_dataset",
    "build_target_dataset",
    "build_source_dataset",
    "build_fid_dataset",
    "build_target_loader",
    "infinite_loader",
    "sample_source_target_batch",
    # io / utils
    "list_images",
    "load_image",
    "normalize_images",
    "denormalize_images",
    "to_latents",
    "resolve_split",
    "dataset_size",
]

_EXPORTS: Dict[str, str] = {
    "SOURCE_DATASETS": "datasets",
    "TARGET_DATASETS": "datasets",
    "FID_TARGET_SETS": "datasets",
    "DATASET_ALIASES": "datasets",
    "ImageFolderDataset": "datasets",
    "FewShotDataset": "datasets",
    "TensorDataset": "datasets",
    "LoopedDataset": "datasets",
    "build_dataset": "datasets",
    "build_target_dataset": "datasets",
    "build_source_dataset": "datasets",
    "build_fid_dataset": "datasets",
    "build_target_loader": "datasets",
    "infinite_loader": "datasets",
    "sample_source_target_batch": "datasets",
    "list_images": "datasets",
    "load_image": "datasets",
    "normalize_images": "datasets",
    "denormalize_images": "datasets",
    "to_latents": "datasets",
    "resolve_split": "datasets",
    "dataset_size": "datasets",
}


def __getattr__(name: str):  # pragma: no cover - trivial dispatch
    """Resolve a public symbol lazily from its defining submodule."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{module_name}", __name__)
    try:
        value = getattr(module, name)
    except AttributeError as exc:  # pragma: no cover
        raise AttributeError(
            f"module {module.__name__!r} does not define {name!r}"
        ) from exc
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    return sorted(set(list(globals().keys()) + list(__all__)))
