"""Prompt templates for the Robust CLIP reproduction package.

This package exposes every LLaVA / OpenFlamingo prompt template used by the
VQA, captioning and jailbreak evaluation harnesses, together with the builders
that format them with an optional LLaVA ``<image>`` placeholder.

Design notes
------------
* The Addendum (and the paper body) never include the literal prompt strings,
  so the plan requires them to be sourced from the upstream repositories
  (``haotian-liu/LLaVA`` and ``mlfoundations/open_flamingo``) and kept in a
  single module.  Every template is therefore tagged
  ``UNSPECIFIED_BY_ADDENDUM`` through :data:`PROMPT_SOURCES` so evaluation
  configs can log prompt provenance instead of pretending the strings came
  from the paper.
* Importing this package is intentionally cheap and side-effect free: the
  concrete templates live in :mod:`robust_clip_repro.prompts.templates` and are
  resolved lazily through PEP 562 ``__getattr__`` (same pattern as
  ``robust_clip_repro.attacks``, ``...data``, ``...metrics`` and
  ``...models``).  That keeps schema/glue code importable without torch.

Public surface
--------------
Submodule
    ``templates`` -- the literal templates and builders.
Constants
    ``DEFAULT_IMAGE_TOKEN``, ``QUESTION_ANSWER_SEPARATOR``, ``UNSPECIFIED``,
    ``VQA_QUESTION``, ``POPE_QUESTION``, ``SQA_QUESTION``, ``CAPTION_PROMPT``,
    ``TASK_PROMPTS``, ``PROMPT_SOURCES``, ``MODEL_PROMPT_KEYS`` and the
    OpenFlamingo variants.
Builders
    :func:`build_prompt`, :func:`build_question_prompt`,
    :func:`build_caption_prompt`, :func:`openflamingo_prompt`,
    :func:`get_prompt`, ...
Helpers
    :func:`registered_prompts`, :func:`prompt_provenance`,
    :func:`has_image_token`, :func:`ensure_image_token`.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional, Tuple

__all__: List[str] = [
    # submodule
    "templates",
    # introspection helpers
    "available_prompts",
    "list_exports",
    "preload",
    # constants / types
    "DEFAULT_IMAGE_TOKEN",
    "QUESTION_ANSWER_SEPARATOR",
    "UNSPECIFIED",
    "VQA_QUESTION",
    "POPE_QUESTION",
    "SQA_QUESTION",
    "CAPTION_PROMPT",
    "VQA_PROMPT",
    "POPE_PROMPT",
    "SQA_PROMPT",
    "LLAVA_VQA_PROMPT",
    "LLAVA_POPE_PROMPT",
    "LLAVA_SQA_PROMPT",
    "LLAVA_CAPTION_PROMPT",
    "OPENFLAMINGO_VQA_QUESTION",
    "OPENFLAMINGO_VQA_PROMPT",
    "OPENFLAMINGO_POPE_QUESTION",
    "OPENFLAMINGO_POPE_PROMPT",
    "OPENFLAMINGO_CAPTION_PROMPT",
    "TASK_PROMPTS",
    "PROMPT_SOURCES",
    "MODEL_PROMPT_KEYS",
    # builders
    "build_task_prompt",
    "build_question_prompt",
    "build_vqa_prompt",
    "build_pope_prompt",
    "build_sqa_prompt",
    "build_caption_prompt",
    "build_prompt",
    "question_prompt",
    "pope_prompt",
    "sqa_prompt",
    "caption_prompt",
    "llava_caption_prompt",
    "openflamingo_vqa_prompt",
    "openflamingo_pope_prompt",
    "openflamingo_caption_prompt",
    "openflamingo_prompt",
    # helpers
    "has_image_token",
    "ensure_image_token",
    "registered_prompts",
    "get_prompt",
    "prompt_provenance",
    # CLI
    "build_arg_parser",
    "main",
]

#: Bundled prompt submodules (only one, kept as a tuple for interface parity
#: with the other package ``__init__`` files in this repository).
_SUBMODULES: Tuple[str, ...] = ("templates",)

#: ``public name -> "submodule.attribute"`` resolution table used by
#: :func:`__getattr__`.  Names not listed here are resolved as bare submodules.
_EXPORTS: Dict[str, str] = {
    # ---- constants / types -------------------------------------------------
    "DEFAULT_IMAGE_TOKEN": "templates.DEFAULT_IMAGE_TOKEN",
    "QUESTION_ANSWER_SEPARATOR": "templates.QUESTION_ANSWER_SEPARATOR",
    "UNSPECIFIED": "templates.UNSPECIFIED",
    "VQA_QUESTION": "templates.VQA_QUESTION",
    "POPE_QUESTION": "templates.POPE_QUESTION",
    "SQA_QUESTION": "templates.SQA_QUESTION",
    "CAPTION_PROMPT": "templates.CAPTION_PROMPT",
    "VQA_PROMPT": "templates.VQA_PROMPT",
    "POPE_PROMPT": "templates.POPE_PROMPT",
    "SQA_PROMPT": "templates.SQA_PROMPT",
    "LLAVA_VQA_PROMPT": "templates.LLAVA_VQA_PROMPT",
    "LLAVA_POPE_PROMPT": "templates.LLAVA_POPE_PROMPT",
    "LLAVA_SQA_PROMPT": "templates.LLAVA_SQA_PROMPT",
    "LLAVA_CAPTION_PROMPT": "templates.LLAVA_CAPTION_PROMPT",
    "OPENFLAMINGO_VQA_QUESTION": "templates.OPENFLAMINGO_VQA_QUESTION",
    "OPENFLAMINGO_VQA_PROMPT": "templates.OPENFLAMINGO_VQA_PROMPT",
    "OPENFLAMINGO_POPE_QUESTION": "templates.OPENFLAMINGO_POPE_QUESTION",
    "OPENFLAMINGO_POPE_PROMPT": "templates.OPENFLAMINGO_POPE_PROMPT",
    "OPENFLAMINGO_CAPTION_PROMPT": "templates.OPENFLAMINGO_CAPTION_PROMPT",
    "TASK_PROMPTS": "templates.TASK_PROMPTS",
    "PROMPT_SOURCES": "templates.PROMPT_SOURCES",
    "MODEL_PROMPT_KEYS": "templates.MODEL_PROMPT_KEYS",
    # ---- builders ----------------------------------------------------------
    "build_task_prompt": "templates.build_task_prompt",
    "build_question_prompt": "templates.build_question_prompt",
    "build_vqa_prompt": "templates.build_vqa_prompt",
    "build_pope_prompt": "templates.build_pope_prompt",
    "build_sqa_prompt": "templates.build_sqa_prompt",
    "build_caption_prompt": "templates.build_caption_prompt",
    "build_prompt": "templates.build_prompt",
    "question_prompt": "templates.question_prompt",
    "pope_prompt": "templates.pope_prompt",
    "sqa_prompt": "templates.sqa_prompt",
    "caption_prompt": "templates.caption_prompt",
    "llava_caption_prompt": "templates.llava_caption_prompt",
    "openflamingo_vqa_prompt": "templates.openflamingo_vqa_prompt",
    "openflamingo_pope_prompt": "templates.openflamingo_pope_prompt",
    "openflamingo_caption_prompt": "templates.openflamingo_caption_prompt",
    "openflamingo_prompt": "templates.openflamingo_prompt",
    # ---- helpers -----------------------------------------------------------
    "has_image_token": "templates.has_image_token",
    "ensure_image_token": "templates.ensure_image_token",
    "registered_prompts": "templates.registered_prompts",
    "get_prompt": "templates.get_prompt",
    "prompt_provenance": "templates.prompt_provenance",
    # ---- CLI ---------------------------------------------------------------
    "build_arg_parser": "templates.build_arg_parser",
    "main": "templates.main",
}


def available_prompts() -> List[str]:
    """Return the bundled prompt submodule names."""
    return list(_SUBMODULES)


def list_exports() -> Dict[str, str]:
    """Return ``public name -> "submodule.attribute"`` for every lazy export."""
    return dict(sorted(_EXPORTS.items()))


def preload(modules: Optional[Any] = None) -> Dict[str, bool]:
    """Eagerly import prompt submodules.

    Parameters
    ----------
    modules:
        ``None`` (default) imports every bundled submodule; a single string or
        an iterable of names imports just those.

    Returns
    -------
    dict
        ``submodule -> success`` mapping.  Failures are reported as ``False``
        rather than raised, so smoke tests can detect missing optional
        dependencies (mirrors the other package ``__init__`` files).
    """
    if modules is None:
        names: List[str] = list(_SUBMODULES)
    elif isinstance(modules, str):
        names = [modules]
    else:
        names = list(modules)

    results: Dict[str, bool] = {}
    for name in names:
        try:
            importlib.import_module(f"{__name__}.{name}")
            results[name] = True
        except Exception:  # pragma: no cover - optional dependency guard
            results[name] = False
    return results


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute resolution for submodules and re-exports."""
    if name in _SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module

    target = _EXPORTS.get(name)
    if target is not None:
        module_name, _, attr = target.partition(".")
        module = importlib.import_module(f"{__name__}.{module_name}")
        value = getattr(module, attr)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(__all__) | set(_SUBMODULES))
