"""``robust_clip_repro.models`` — victim models and the trainable vision encoder.

This package aggregates the victim-model wrappers used by every evaluation
harness plus the trainable OpenCLIP vision encoder used for Robust CLIP
unsupervised adversarial fine-tuning:

* :mod:`~robust_clip_repro.models.clip_vision_encoder`
  — OpenCLIP ``ViT-L-14``/``openai`` vision tower at 224 px; the module that is
  fine-tuned/trained by Robust CLIP and shared by the CLIP/LLaVA wrappers.
* :mod:`~robust_clip_repro.models.llava_openclip`
  — LLaVA-1.5 7B patched to consume the **OpenCLIP** CLIP implementation with
  the OpenAI ``ViT-L/14@224`` vision encoder (instead of LLaVA's default HF
  ``ViT-L/14@336``), as mandated by the Addendum.
* :mod:`~robust_clip_repro.models.openflamingo_wrapper`
  — OpenFlamingo victim wrapper mirroring the LLaVA interface.  The Addendum
  only pins the upstream repository, so version/vision-encoder/language-backbone
  are exposed as configuration and tagged ``UNSPECIFIED_BY_ADDENDUM``.

Importing this package is intentionally side-effect free: ``torch``,
``open_clip``, ``transformers``, ``llava`` and ``open_flamingo`` are all
imported lazily on first attribute access (PEP 562), so schema/glue helpers can
be used in environments without the heavy GPU stack installed.

Provenance policy inherited from :mod:`robust_clip_repro`
--------------------------------------------------------
Nothing the paper body/Addendum do not state is invented here.  The only
hard-coded values are the Addendum-mandated ones:

* LLaVA uses the OpenAI CLIP ``ViT-L-14`` vision encoder at **224 px** through
  ``open_clip`` (``use_openclip=True``, ``patch_hf_clip=True``).
* The language backbone is frozen for robustness evaluation.
* Perturbations follow the int16 (half-precision) / int32 (single-precision)
  policy from :mod:`robust_clip_repro.utils.precision`.

Everything else (OpenFlamingo version, dtype, device map, generation settings,
prompt templates, checkpoint paths) is exposed through config dataclasses and
surfaced via each wrapper's ``external_defaults()`` / ``summary()``.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    # submodules
    "clip_vision_encoder",
    "llava_openclip",
    "openflamingo_wrapper",
    # introspection helpers
    "available_models",
    "list_exports",
    "preload",
    "build_model",
    "SUBMODULES",
    "MODEL_SELECTORS",
]

#: Submodules bundled by this package (imported lazily).
SUBMODULES: Tuple[str, ...] = (
    "clip_vision_encoder",
    "llava_openclip",
    "openflamingo_wrapper",
)

# Backwards/forwards compatible alias used by introspection helpers.
_SUBMODULES: Tuple[str, ...] = SUBMODULES

#: Mapping of the ``model_name`` selector strings used throughout the
#: evaluation configs (``configs/models.yaml``, ``configs/*.yaml``) to the
#: factory function inside the corresponding submodule.
MODEL_SELECTORS: Dict[str, Tuple[str, str]] = {
    # generic OpenCLIP vision encoder (vanilla CLIP / Robust CLIP backbone)
    "clip": ("clip_vision_encoder", "build_clip_vision_encoder"),
    "openclip": ("clip_vision_encoder", "build_clip_vision_encoder"),
    "robust_clip": ("clip_vision_encoder", "build_clip_vision_encoder"),
    "vision_encoder": ("clip_vision_encoder", "build_clip_vision_encoder"),
    # LLaVA-1.5 7B with OpenCLIP ViT-L/14@224
    "llava": ("llava_openclip", "build_llava_victim"),
    "llava-1.5": ("llava_openclip", "build_llava_victim"),
    "llava_openclip": ("llava_openclip", "build_llava_victim"),
    # OpenFlamingo
    "openflamingo": ("openflamingo_wrapper", "build_openflamingo_victim"),
    "open_flamingo": ("openflamingo_wrapper", "build_openflamingo_victim"),
    "flamingo": ("openflamingo_wrapper", "build_openflamingo_victim"),
}

#: Lazy re-exports: ``public name -> (submodule, attribute name)``.
#: ``None`` as attribute means "same name as the public name".
_EXPORTS: Dict[str, Tuple[str, Optional[str]]] = {
    # ------------------------------------------------------------------ #
    # clip_vision_encoder — trainable OpenCLIP ViT-L/14@224 vision tower  #
    # ------------------------------------------------------------------ #
    "CLIPVisionConfig": ("clip_vision_encoder", None),
    "CLIPVisionEncoder": ("clip_vision_encoder", None),
    "RandomVisionEncoder": ("clip_vision_encoder", None),
    "build_clip_vision_encoder": ("clip_vision_encoder", None),
    "load_vision_encoder_weights": ("clip_vision_encoder", None),
    "DEFAULT_MODEL_NAME": ("clip_vision_encoder", None),
    "DEFAULT_PRETRAINED": ("clip_vision_encoder", None),
    "DEFAULT_RESOLUTION": ("clip_vision_encoder", None),
    "DEFAULT_IMAGE_SIZE": ("clip_vision_encoder", None),
    "DEFAULT_CONTEXT_LENGTH": ("clip_vision_encoder", None),
    "CLIP_MEAN": ("clip_vision_encoder", None),
    "CLIP_STD": ("clip_vision_encoder", None),
    "ROBUST_CLIP_REPO_URL": ("clip_vision_encoder", None),
    "OPENAI_CLIP_WEIGHTS_URL": ("clip_vision_encoder", None),
    # ------------------------------------------------------------------ #
    # llava_openclip — LLaVA-1.5 7B with OpenCLIP ViT-L/14@224            #
    # ------------------------------------------------------------------ #
    "LLaVAOpenCLIPConfig": ("llava_openclip", None),
    "LLaVAVictim": ("llava_openclip", None),
    "LLaVACaptioner": ("llava_openclip", None),
    "DummyLLaVAVictim": ("llava_openclip", None),
    "build_llava_victim": ("llava_openclip", None),
    "load_llava_openclip": ("llava_openclip", None),
    "patch_hf_clip_to_openclip": ("llava_openclip", None),
    "vqa_prompt": ("llava_openclip", None),
    "caption_prompt": ("llava_openclip", None),
    "LLAVA_REPO_URL": ("llava_openclip", None),
    "LLAVA_15_7B_HF_ID": ("llava_openclip", None),
    "DEFAULT_MODEL_PATH": ("llava_openclip", None),
    "DEFAULT_CLIP_MODEL": ("llava_openclip", None),
    "DEFAULT_CLIP_PRETRAINED": ("llava_openclip", None),
    "DEFAULT_CLIP_RESOLUTION": ("llava_openclip", None),
    "DEFAULT_DTYPE": ("llava_openclip", None),
    # ------------------------------------------------------------------ #
    # openflamingo_wrapper — OpenFlamingo victim (repo pinned only)       #
    # ------------------------------------------------------------------ #
    "OpenFlamingoConfig": ("openflamingo_wrapper", None),
    "OpenFlamingoVictim": ("openflamingo_wrapper", None),
    "OpenFlamingoCaptioner": ("openflamingo_wrapper", None),
    "DummyOpenFlamingoVictim": ("openflamingo_wrapper", None),
    "build_openflamingo_victim": ("openflamingo_wrapper", None),
    "load_openflamingo": ("openflamingo_wrapper", None),
}


def available_models() -> List[str]:
    """Return the bundled victim/vision-encoder submodule names."""
    return list(SUBMODULES)


def list_exports() -> Dict[str, str]:
    """Return ``public name -> "submodule.attribute"`` for every lazy export.

    Useful for logging/discovery: the evaluation harnesses can log exactly
    which model entry points are available in the current environment.
    """
    return {
        name: f"{mod}.{attr or name}"
        for name, (mod, attr) in sorted(_EXPORTS.items())
    }


def preload(modules: Optional[Any] = None) -> Dict[str, bool]:
    """Eagerly import the requested model submodules.

    Parameters
    ----------
    modules:
        ``None`` (default) to import every submodule, a single submodule name
        (``str``) or an iterable of names.

    Returns
    -------
    Dict[str, bool]
        ``submodule -> success`` mapping.  Missing optional dependencies
        (``torch``, ``open_clip``, ``llava``, ``open_flamingo``) are reported
        as ``False`` instead of raising, which is what the smoke tests rely on.
    """
    if modules is None:
        names: List[str] = list(SUBMODULES)
    elif isinstance(modules, str):
        names = [modules]
    else:
        names = list(modules)

    results: Dict[str, bool] = {}
    for name in names:
        if name not in SUBMODULES:
            results[name] = False
            continue
        try:
            importlib.import_module(f".{name}", __name__)
            results[name] = True
        except Exception:  # pragma: no cover - optional heavy dependencies
            results[name] = False
    return results


def build_model(model_name: str, config: Any = None, **kwargs: Any) -> Any:
    """Instantiate a victim / vision encoder by its config selector string.

    Parameters
    ----------
    model_name:
        One of :data:`MODEL_SELECTORS` (e.g. ``"llava"``, ``"openflamingo"``,
        ``"clip"``/``"robust_clip"``).  Comparison is case/format insensitive.
    config:
        A config dataclass, mapping, or ``None``.  Passed positionally to the
        corresponding factory function when supported.
    **kwargs:
        Forwarded to the factory function.

    Returns
    -------
    The object produced by the factory (a victim wrapper or a vision encoder).
    """
    key = str(model_name).strip().lower().replace("-", "_")
    if key not in MODEL_SELECTORS:
        # tolerate "llava-1.5" style spellings that only differ by dashes
        key = key.replace(".", "_")
        for alias in MODEL_SELECTORS:
            if alias.replace("-", "_") == key:
                key = alias
                break
    if key not in MODEL_SELECTORS:
        raise ValueError(
            f"unknown model selector {model_name!r}; "
            f"expected one of {sorted(MODEL_SELECTORS)}"
        )

    submodule_name, factory_name = MODEL_SELECTORS[key]
    submodule = importlib.import_module(f".{submodule_name}", __name__)
    factory = getattr(submodule, factory_name)

    if config is None:
        return factory(**kwargs)
    try:
        return factory(config, **kwargs)
    except TypeError:
        # factory does not accept a positional config (e.g. OpenFlamingo
        # wrapper keyword-only signature) — retry keyword-style.
        return factory(config=config, **kwargs)


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute resolution (submodules and re-exports)."""
    if name in SUBMODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module

    if name in _EXPORTS:
        submodule_name, attr = _EXPORTS[name]
        module = importlib.import_module(f".{submodule_name}", __name__)
        value = getattr(module, attr or name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(__all__) | set(_SUBMODULES))
