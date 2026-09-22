"""Data package for LCA-on-the-Line.

Exposes the in-distribution (ImageNet-1k) and out-of-distribution loaders used
by the evaluation pipeline (Section 4 of the paper):

* :mod:`src.data.imagenet`     -- ImageNet-1k ID loader + canonical label mapping
* :mod:`src.data.ood_datasets` -- ImageNet-v2 (MatchedFrequency), ImageNet-S,
  ImageNet-R, ImageNet-A and ObjectNet loaders, all remapped onto the canonical
  1000-class WordNet ordering.

The initializer is intentionally dependency free: heavy packages (torch,
torchvision, HuggingFace ``datasets``, numpy) are only imported when a symbol is
actually accessed, via PEP 562 ``__getattr__``.  This keeps
``import src.data`` cheap for tests and metadata-only utilities.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional, Tuple

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Sub-modules and their public exports
# ---------------------------------------------------------------------------

_SUBMODULES: Tuple[str, ...] = ("imagenet", "ood_datasets")

_IMAGENET_EXPORTS: Tuple[str, ...] = (
    "ImageNetUnavailable",
    "LabelMapping",
    "ImageNetIDDataset",
    "build_transform",
    "load_imagenet_wnids",
    "load_imagenet_class_names",
    "build_label_mapping",
    "load_imagenet_hf",
    "load_imagenet_local",
    "build_imagenet_dataset",
    "subset_dataset",
    "build_imagenet_loader",
    "synthetic_imagenet_like",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "DEFAULT_RESOLUTION",
    "DEFAULT_CROP_PCT",
    "DEFAULT_SPLIT",
    "DEFAULT_HF_NAME",
    "IMAGENET_NUM_CLASSES",
)

_OOD_EXPORTS: Tuple[str, ...] = (
    "OODDataset",
    "WnidOrderedDataset",
    "NpyClassDataset",
    "ObjectNetDataset",
    "build_ood_transform",
    "build_imagenet_v2_dataset",
    "build_imagenet_sketch_dataset",
    "build_imagenet_r_dataset",
    "build_imagenet_a_dataset",
    "build_objectnet_dataset",
    "build_ood_dataset",
    "build_all_ood_datasets",
    "normalize_ood_name",
    "display_name",
    "subsample",
    "load_imagenet_v2_local",
    "load_imagenet_v2_hf",
    "load_imagenet_sketch_local",
    "load_imagenet_sketch_hf",
    "load_wnid_ordered_folder",
    "build_wnid_label_mapping",
    "parse_objectnet_mapping",
    "DEFAULT_OOD_DATASETS",
    "OOD_DISPLAY_NAMES",
    "OOD_ALIASES",
    "HF_DATASETS",
    "IMAGENET_V2_HF_REPO",
    "IMAGENET_V2_COMMIT",
    "IMAGENET_V2_VARIANT",
    "IMAGENET_A_WNIDS",
    "OBJECTNET_MAPPING_FILE",
    "BUILDERS",
)

_EXPORTS: Dict[str, str] = {}
for _name in _IMAGENET_EXPORTS:
    _EXPORTS[_name] = "imagenet"
for _name in _OOD_EXPORTS:
    _EXPORTS[_name] = "ood_datasets"

# Symbols that may live in either submodule (resolved from the first match).
_COMMON_EXPORTS: Tuple[str, ...] = (
    "build_transform",
    "build_ood_transform",
    "subset_dataset",
    "subsample",
)

# Paper / addendum facing aliases -> canonical symbol names.
_ALIASES: Dict[str, str] = {
    "ImageNet": "build_imagenet_dataset",
    "LoadImageNet": "build_imagenet_dataset",
    "ImageNetLoader": "build_imagenet_loader",
    "LabelMap": "LabelMapping",
    "OODDatasetBuilder": "build_ood_dataset",
    "BuildOODDatasets": "build_all_ood_datasets",
    "OOD_DATASETS": "DEFAULT_OOD_DATASETS",
    "ImagenetV2Dataset": "build_imagenet_v2_dataset",
    "ImageNetSketchDataset": "build_imagenet_sketch_dataset",
    "ImageNetRDataset": "build_imagenet_r_dataset",
    "ImageNetADataset": "build_imagenet_a_dataset",
    "ObjectNetLoader": "build_objectnet_dataset",
}

__all__ = (
    ["__version__"]
    + list(_SUBMODULES)
    + sorted(_EXPORTS)
    + sorted(_ALIASES)
)


# ---------------------------------------------------------------------------
# Lazy import helpers
# ---------------------------------------------------------------------------


def _import_submodule(name: str) -> Any:
    """Import a sibling submodule, tolerating several ``sys.path`` layouts."""
    candidates = (
        f"{__name__}.{name}",
        f"src.data.{name}",
        f"data.{name}",
        name,
    )
    last_error: Optional[BaseException] = None
    for candidate in candidates:
        try:
            return importlib.import_module(candidate)
        except Exception as exc:  # noqa: BLE001 - try the next layout
            last_error = exc
    if last_error is not None:
        raise last_error
    raise ImportError(f"could not import submodule {name!r}")


def _resolve(name: str) -> Any:
    """Resolve ``name`` to a submodule, an exported symbol or an alias."""
    if name in _SUBMODULES:
        return _import_submodule(name)

    target = _ALIASES.get(name, name)

    if target in _SUBMODULES:
        return _import_submodule(target)

    module_name = _EXPORTS.get(target)
    candidates: List[str]
    if module_name is not None:
        candidates = [module_name]
    else:
        candidates = list(_SUBMODULES)

    for mod_name in candidates:
        try:
            module = _import_submodule(mod_name)
        except Exception:  # noqa: BLE001 - fall through to the next module
            continue
        if hasattr(module, target):
            return getattr(module, target)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __getattr__(name: str) -> Any:  # PEP 562
    value = _resolve(name)
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(_SUBMODULES) | set(_EXPORTS) | set(_ALIASES))


def __getattr_many__() -> Dict[str, Any]:
    """Resolve every advertised export; failures are returned as exceptions."""
    resolved: Dict[str, Any] = {}
    for name in list(_EXPORTS) + list(_ALIASES):
        try:
            resolved[name] = getattr(__import__(__name__, fromlist=["_"]), name)
        except Exception as exc:  # noqa: BLE001 - report, do not raise
            resolved[name] = exc
    return resolved
