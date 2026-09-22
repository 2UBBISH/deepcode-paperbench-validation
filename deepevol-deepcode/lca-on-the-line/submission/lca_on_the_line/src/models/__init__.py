"""Model zoos for the LCA-on-the-Line reproduction (Appendix A).

This package exposes two interchangeable model zoos:

* :mod:`src.models.vm_zoo` -- the 36 torchvision vision models (VMs).
* :mod:`src.models.vlm_zoo` -- the 39 vision-language models (VLMs):
  ALBEF, BLIP, 7 OpenAI CLIP checkpoints and 30 OpenCLIP configs.

Both zoos share the same contract used by the evaluation driver
``src/eval/evaluate_models.py``:

    logits(x)   -> (B, K) zero-shot / classification logits
    features(x) -> (B, D) penultimate representation ``M(X)`` (paper Sec 4.3.1)

Importing this package is intentionally dependency free (no torch,
torchvision, open_clip, clip or transformers import happens at module load
time); submodules and their public symbols are resolved lazily on first
attribute access via PEP 562 ``__getattr__``.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional, Tuple

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Submodules
# ---------------------------------------------------------------------------
_SUBMODULES: Tuple[str, ...] = ("vm_zoo", "vlm_zoo")

# ---------------------------------------------------------------------------
# Public symbols re-exported from each submodule (name -> providing module)
# ---------------------------------------------------------------------------
_EXPORTS: Dict[str, str] = {
    # ---------------------------------------------------------------- vm_zoo
    "ModelSpec": "vm_zoo",
    "VisionModelWrapper": "vm_zoo",
    "VISION_MODEL_SPECS": "vm_zoo",
    "VM_NAMES": "vm_zoo",
    "VM_ALIASES": "vm_zoo",
    "SPECS_BY_NAME": "vm_zoo",
    "list_vm_names": "vm_zoo",
    "specs_by_family": "vm_zoo",
    "get_model_spec": "vm_zoo",
    "get_weights_enum": "vm_zoo",
    "resolve_weights": "vm_zoo",
    "find_classifier_module": "vm_zoo",
    "build_torchvision_model": "vm_zoo",
    "create_vm": "vm_zoo",
    "build_vm_zoo": "vm_zoo",
    "release_vm_zoo": "vm_zoo",
    "get_eval_transform": "vm_zoo",
    "get_train_transform": "vm_zoo",
    "create_dummy_vm": "vm_zoo",
    # --------------------------------------------------------------- vlm_zoo
    "VlmSpec": "vlm_zoo",
    "VlmModelWrapper": "vlm_zoo",
    "VLM_MODEL_SPECS": "vlm_zoo",
    "VLM_NAMES": "vlm_zoo",
    "VLM_ALIASES": "vlm_zoo",
    "SPECS_BY_NAME_VLM": "vlm_zoo",
    "list_vlm_names": "vlm_zoo",
    "build_openclip_specs": "vlm_zoo",
    "class_names_for_hierarchy": "vlm_zoo",
    "format_prompts": "vlm_zoo",
    "create_vlm": "vlm_zoo",
    "build_vlm_zoo": "vlm_zoo",
    "release_vlm_zoo": "vlm_zoo",
    "create_dummy_vlm": "vlm_zoo",
    "IMAGENET_PROMPT_TEMPLATES": "vlm_zoo",
    "SIMPLE_PROMPT_TEMPLATE": "vlm_zoo",
}

# Common helpers that exist in both zoos -- resolved from ``vm_zoo`` first
# (identical signatures in vlm_zoo) so downstream code can import them from
# the package root.
_COMMON_EXPORTS: Tuple[str, ...] = (
    "extract_outputs",
    "extract_and_cache",
    "cache_path",
    "save_outputs",
    "load_outputs",
)

# Paper/addendum naming -> canonical registry key aliases.
_ALIASES: Dict[str, str] = {
    "VMZoo": "build_vm_zoo",
    "VLMZoo": "build_vlm_zoo",
    "get_vm": "create_vm",
    "get_vlm": "create_vlm",
    "IMAGENET_NUM_CLASSES": "IMAGENET_NUM_CLASSES",
}


def _import_submodule(name: str) -> Any:
    """Import a sibling submodule tolerating several ``sys.path`` layouts."""
    candidates = (
        f"{__name__}.{name}",
        f"src.models.{name}",
        f"models.{name}",
        name,
    )
    last_error: Optional[BaseException] = None
    for candidate in candidates:
        try:
            return importlib.import_module(candidate)
        except Exception as exc:  # pragma: no cover - defensive
            last_error = exc
    # pragma: no cover - defensive
    assert last_error is not None
    raise last_error


def _resolve(name: str) -> Any:
    """Resolve ``name`` to an object from one of the zoo submodules or a helper."""
    if name in _SUBMODULES:
        return _import_submodule(name)

    if name in _COMMON_EXPORTS:
        for module_name in _SUBMODULES:
            module = _import_submodule(module_name)
            if hasattr(module, name):
                return getattr(module, name)
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    target = _ALIASES.get(name, name)
    module_name = _EXPORTS.get(target)
    if module_name is not None:
        module = _import_submodule(module_name)
        if hasattr(module, target):
            return getattr(module, target)
        raise AttributeError(f"module {module_name!r} has no attribute {target!r}")

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __getattr__(name: str) -> Any:  # PEP 562
    value = _resolve(name)
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(_SUBMODULES) | set(_EXPORTS) | set(_ALIASES))


__all__ = (
    ["__version__"]
    + list(_SUBMODULES)
    + sorted(_EXPORTS)
    + sorted(_COMMON_EXPORTS)
    + sorted(_ALIASES)
)
