"""Prompt templates for the Robust CLIP reproduction harnesses.

Provenance
----------
The Addendum states (page 1)::

    Details about prompts for LLaVA and OpenFlamingo — when grading, the rubric
    will check for implementations of things asked for from the code below.

and that the LLaVA harness is taken from https://github.com/haotian-liu/LLaVA
and the OpenFlamingo model/harness from
https://github.com/mlfoundations/open_flamingo.

The Addendum **does not** include literal prompt strings.  Therefore every
template in this module is tagged as either

* ``UPSTREAM``            — inherited from haotian-liu/LLaVA or
  mlfoundations/open_flamingo (the sources the Addendum pins), or
* ``UNSPECIFIED``         — the Addendum is silent about the exact wording; the
  value is an externally supplied default and is reported as such.

No literal template here is invented "as if from the paper"; the default text
mirrors the well-known upstream LLaVA-1.5 / OpenFlamingo conventions and can be
overridden through ``configs/models.yaml`` (``prompts.*``) or the harness CLIs.

Canonical upstream references
-----------------------------
LLaVA (haotian-liu/LLaVA)
    * ``llava/conversation.py``  — ``conv_vicuna_v1`` / ``SeparatorStyle.VICUNA_V1``
    * ``llava/eval/model_vqa_loader.py`` / ``model_vqa.py`` — VQA question text
    * ``llava/eval/model_vqa_science.py`` — ScienceQA (SQA-I) question text
    * ``llava/eval/eval_pope.py`` — POPE judging
    * ``llava/train/train.py`` — ``DEFAULT_IMAGE_TOKEN = "<image>"``

OpenFlamingo (mlfoundations/open_flamingo)
    * ``open_flamingo/eval/models/open_flamingo.py`` — ``<image>`` prefix and
      ``"<image>Output:"`` / ``"<image>A short caption:"`` style prompts
    * ``open_flamingo/eval/eval_datasets.py`` — COCO / Flickr30k captions

Public interface (what the harnesses import)
--------------------------------------------
LLaVA / generic:
    ``task_prompt``, ``question_prompt``
    (plus the aliases ``build_question_prompt``, ``build_vqa_prompt``)
    ``pope_prompt``, ``sqa_prompt``, ``build_pope_prompt``, ``build_sqa_prompt``
    ``caption_prompt``, ``build_caption_prompt``, ``llava_caption_prompt``
    ``build_prompt``

OpenFlamingo:
    ``openflamingo_vqa_prompt``, ``openflamingo_pope_prompt``,
    ``openflamingo_caption_prompt``, ``openflamingo_prompt``

Constants:
    ``DEFAULT_IMAGE_TOKEN`` (``"<image>"``), ``QUESTION_ANSWER_SEPARATOR``,
    ``CAPTION_PROMPT``, ``POPE_PROMPT`` (deprecated alias — actually the VQA
    question phrase), ``SQA_PROMPT``, ``VQA_PROMPT``, ``OPENFLAMINGO_*``,
    ``LLAVA_*``, ``TASK_PROMPTS``, ``PROMPT_SOURCES``, ``UNSPECIFIED``

CLI::

    python -m robust_clip_repro.prompts.templates --self-test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

LOGGER = logging.getLogger("robust_clip_repro.prompts.templates")

__all__ = [
    # tokens / separators
    "DEFAULT_IMAGE_TOKEN",
    "QUESTION_ANSWER_SEPARATOR",
    "UNSPECIFIED",
    # LLaVA / generic templates
    "VQA_QUESTION",
    "VQA_PROMPT",
    "POPE_QUESTION",
    "SQA_QUESTION",
    "CAPTION_PROMPT",
    "POPE_PROMPT",
    "SQA_PROMPT",
    "LLAVA_VQA_PROMPT",
    "LLAVA_POPE_PROMPT",
    "LLAVA_SQA_PROMPT",
    "LLAVA_CAPTION_PROMPT",
    # OpenFlamingo templates
    "OPENFLAMINGO_VQA_QUESTION",
    "OPENFLAMINGO_VQA_PROMPT",
    "OPENFLAMINGO_POPE_QUESTION",
    "OPENFLAMINGO_POPE_PROMPT",
    "OPENFLAMINGO_CAPTION_PROMPT",
    # registries
    "TASK_PROMPTS",
    "PROMPT_SOURCES",
    "MODEL_PROMPT_KEYS",
    # helpers
    "has_image_token",
    "ensure_image_token",
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
    "registered_prompts",
    "get_prompt",
    "prompt_provenance",
    "main",
]

# --------------------------------------------------------------------------- #
# Provenance markers                                                          #
# --------------------------------------------------------------------------- #

UNSPECIFIED = "UNSPECIFIED_BY_ADDENDUM"
"""Tag meaning: the Addendum does not state this value (external default)."""

_LLAVA_SOURCE = "haotian-liu/LLaVA (llava/conversation.py, llava/eval/model_vqa_loader.py)"
_OF_SOURCE = "mlfoundations/open_flamingo (open_flamingo/eval/models/open_flamingo.py)"

PROMPT_SOURCES: Dict[str, Dict[str, str]] = {
    "llava_vqa": {"source": _LLAVA_SOURCE, "provenance": UNSPECIFIED},
    "llava_pope": {"source": _LLAVA_SOURCE, "provenance": UNSPECIFIED},
    "llava_sqa": {
        "source": "haotian-liu/LLaVA (llava/eval/model_vqa_science.py)",
        "provenance": UNSPECIFIED,
    },
    "llava_caption": {"source": _LLAVA_SOURCE, "provenance": UNSPECIFIED},
    "openflamingo_vqa": {"source": _OF_SOURCE, "provenance": UNSPECIFIED},
    "openflamingo_pope": {"source": _OF_SOURCE, "provenance": UNSPECIFIED},
    "openflamingo_caption": {"source": _OF_SOURCE, "provenance": UNSPECIFIED},
}
"""Where each template is inherited from, and whether the Addendum states it.

The Addendum pins the two upstream repositories but contains no literal prompt
strings, so every entry here stays ``UNSPECIFIED`` until a value is supplied by
``configs/models.yaml`` or the caller.
"""

# --------------------------------------------------------------------------- #
# Tokens                                                                      #
# --------------------------------------------------------------------------- #

DEFAULT_IMAGE_TOKEN = "<image>"
"""LLaVA's multimodal image placeholder (haotian-liu/LLaVA)."""

QUESTION_ANSWER_SEPARATOR = "\nAnswer the question using a single word or phrase."
"""Suffix LLaVA-1.5 appends when the answer must be short (VQA/POPE/SQA)."""

# --------------------------------------------------------------------------- #
# LLaVA / generic prompt templates                                            #
# --------------------------------------------------------------------------- #

VQA_QUESTION = (
    "{question}\n"
    "Answer the question using a single word or phrase."
)
"""TextVQA/VQA question text (LLaVA ``model_vqa_loader``)."""

VQA_PROMPT = VQA_QUESTION  # backwards-compatible alias

POPE_QUESTION = (
    "{question}\n"
    "Answer the question using a single word or phrase."
)
"""POPE asks a yes/no question; LLaVA uses the same short-answer suffix."""

SQA_QUESTION = (
    "{question}\n"
    "Answer with the option's letter from the given choices directly."
)
"""ScienceQA/SQA-I question text (LLaVA ``model_vqa_science``)."""

CAPTION_PROMPT = "A short caption:"
"""OpenFlamingo/LLaVA captioning continuation prompt."""

# Deprecated alias names kept for harness compatibility.  ``POPE_PROMPT`` is the
# *question* phrase used for POPE, not the POPE judging rule.
POPE_PROMPT = POPE_QUESTION
SQA_PROMPT = SQA_QUESTION

LLAVA_VQA_PROMPT = VQA_QUESTION
LLAVA_POPE_PROMPT = POPE_QUESTION
LLAVA_SQA_PROMPT = SQA_QUESTION
LLAVA_CAPTION_PROMPT = CAPTION_PROMPT

# --------------------------------------------------------------------------- #
# OpenFlamingo prompt templates                                              #
# --------------------------------------------------------------------------- #

OPENFLAMINGO_VQA_QUESTION = (
    "{question}\n"
    "Answer the question using a single word or phrase."
)
"""OpenFlamingo VQA prompt; the ``<image>`` token prefix is added by the builder."""

OPENFLAMINGO_VQA_PROMPT = OPENFLAMINGO_VQA_QUESTION

OPENFLAMINGO_POPE_QUESTION = (
    "{question}\n"
    "Answer the question using a single word or phrase."
)
"""OpenFlamingo POPE prompt (yes/no short answer)."""

OPENFLAMINGO_POPE_PROMPT = OPENFLAMINGO_POPE_QUESTION

OPENFLAMINGO_CAPTION_PROMPT = "A short caption:"
"""OpenFlamingo captioning prompt (``<image>`` prefix added by the builder)."""

# --------------------------------------------------------------------------- #
# Registries                                                                  #
# --------------------------------------------------------------------------- #

TASK_PROMPTS: Dict[str, str] = {
    "vqa": VQA_QUESTION,
    "pope": POPE_QUESTION,
    "sqa": SQA_QUESTION,
    "caption": CAPTION_PROMPT,
}
"""Task name -> default template (LLaVA-style)."""

MODEL_PROMPT_KEYS: Dict[str, Dict[str, str]] = {
    "llava": {
        "vqa": "llava.vqa",
        "pope": "llava.pope",
        "sqa": "llava.sqa",
        "caption": "llava.caption",
    },
    "openflamingo": {
        "vqa": "openflamingo.vqa",
        "pope": "openflamingo.pope",
        "caption": "openflamingo.caption",
    },
}
"""Which ``configs/models.yaml`` ``prompts.*`` key holds each task template."""

_TASK_ALIASES: Dict[str, str] = {
    "vqa": "vqa",
    "vqa_loader": "vqa",
    "textvqa": "vqa",
    "text_vqa": "vqa",
    "pope": "pope",
    "polling": "pope",
    "sqa": "sqa",
    "sqa-i": "sqa",
    "sqa_i": "sqa",
    "scienceqa": "sqa",
    "science_qa": "sqa",
    "caption": "caption",
    "captioning": "caption",
    "coco": "caption",
    "flickr30k": "caption",
}

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _normalize_task(task: Optional[str], default: Optional[str] = None) -> str:
    """Map a task/dataset name onto ``vqa|pope|sqa|caption``."""
    if task is None:
        if default is None:
            return "vqa"
        return _normalize_task(default)
    key = str(task).strip().lower().replace(" ", "").replace("_", "_")
    if key in _TASK_ALIASES:
        return _TASK_ALIASES[key]
    squeezed = key.replace("-", "").replace("_", "")
    for alias, canonical in _TASK_ALIASES.items():
        if alias.replace("-", "").replace("_", "") == squeezed:
            return canonical
    LOGGER.debug("Unknown task name %r; falling back to 'vqa'", task)
    return "vqa" if default is None else _normalize_task(default)


def _normalize_model(model: Optional[str]) -> str:
    """Map a victim name onto ``llava`` or ``openflamingo``."""
    if not model:
        return "llava"
    key = str(model).strip().lower().replace("-", "").replace("_", "")
    if key.startswith("openflam") or key == "of":
        return "openflamingo"
    return "llava"


def _template_for(
    task: str,
    model: str,
    template: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> str:
    """Resolve the default template for ``(model, task)``."""
    if template:
        return template
    if dataset_name:
        task = _normalize_task(dataset_name, default=task)
    if model == "openflamingo":
        return {
            "vqa": OPENFLAMINGO_VQA_QUESTION,
            "pope": OPENFLAMINGO_POPE_QUESTION,
            "sqa": OPENFLAMINGO_VQA_QUESTION,
            "caption": OPENFLAMINGO_CAPTION_PROMPT,
        }[task]
    return {
        "vqa": VQA_QUESTION,
        "pope": POPE_QUESTION,
        "sqa": SQA_QUESTION,
        "caption": CAPTION_PROMPT,
    }[task]


def has_image_token(prompt: str) -> bool:
    """True when ``prompt`` already contains LLaVA's ``<image>`` placeholder."""
    return DEFAULT_IMAGE_TOKEN in str(prompt)


def ensure_image_token(prompt: str) -> str:
    """Prefix ``<image>\\n`` when the placeholder is missing (upstream behaviour)."""
    prompt = str(prompt)
    if has_image_token(prompt):
        return prompt
    return f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"


def _format(template: Optional[str], sample: Any, **fields: Any) -> Optional[str]:
    """Format ``template`` defensively, returning ``None`` when not applicable.

    ``sample`` may be a mapping (``{"question": ...}``), an object with a
    ``question``/``prompt`` attribute, a ``(question, [choices])`` tuple, or a
    bare question string.
    """
    if template is None:
        return None

    question = fields.pop("question", None)
    choices = fields.pop("choices", None)

    if isinstance(sample, Mapping):
        question = question if question is not None else sample.get("question")
        choices = choices if choices is not None else sample.get("choices")
    elif isinstance(sample, (tuple, list)):
        if len(sample) >= 1 and sample[0] is not None:
            question = question if question is not None else sample[0]
        if len(sample) >= 2 and sample[1] is not None:
            choices = choices if choices is not None else sample[1]
    elif sample is not None:
        for attr in ("question", "prompt", "query", "text"):
            if hasattr(sample, attr):
                value = getattr(sample, attr)
                if value:
                    question = question if question is not None else value
                    break
        if choices is None and hasattr(sample, "choices"):
            choices = getattr(sample, "choices")

    if not question and "{" in template:
        return None

    base = str(question) if question else ""
    if choices is not None and "sqa" in template.lower() and "answer" not in base.lower():
        choice_text = _format_choices(choices)
        if choice_text:
            base = f"{base}\n{choice_text}" if base else choice_text

    if "{question}" not in template:
        # Template without a slot: still honour the image token convention.
        return template

    return template.replace("{question}", base)


def _format_choices(choices: Any) -> str:
    """Render SQA choices as ``A. ... B. ...`` lines (LLaVA convention)."""
    if choices is None:
        return ""
    if isinstance(choices, (str, bytes)):
        return str(choices)
    try:
        items: Sequence[Any] = list(choices)
    except TypeError:
        return str(choices)
    if not items:
        return ""
    if len(items) == 1 and isinstance(items[0], (list, tuple)):
        items = list(items[0])
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines: List[str] = []
    for index, choice in enumerate(items):
        if index >= len(letters):
            break
        lines.append(f"{letters[index]}. {choice}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Public builders                                                             #
# --------------------------------------------------------------------------- #


def build_task_prompt(
    sample: Any = None,
    *,
    question: Optional[str] = None,
    choices: Any = None,
    task: Optional[str] = None,
    model: str = "llava",
    dataset_name: Optional[str] = None,
    template: Optional[str] = None,
) -> Optional[str]:
    """Build the prompt for ``task`` with the default model conventions.

    Returns ``None`` when the sample carries no question (captioning callers pass
    ``question=None`` and are expected to fall back to
    :func:`build_caption_prompt`).
    """
    resolved_task = _normalize_task(task, default=dataset_name)
    resolved_model = _normalize_model(model)
    effective = _template_for(
        resolved_task, resolved_model, template=template, dataset_name=dataset_name
    )
    fields: Dict[str, Any] = {}
    if question is not None:
        fields["question"] = question
    if choices is not None:
        fields["choices"] = choices
    prompt = _format(effective, sample, **fields)
    if prompt is None:
        return None
    return ensure_image_token(prompt)


def build_question_prompt(
    sample: Any = None,
    *,
    question: Optional[str] = None,
    choices: Any = None,
    model: str = "llava",
    dataset_name: Optional[str] = None,
    template: Optional[str] = None,
    task: Optional[str] = None,
    **kwargs: Any,
) -> Optional[str]:
    """Question prompt for VQA-style tasks (TextVQA/POPE/SQA-I).

    This is the entry point used by ``data/benchmarks.py::build_prompt`` and by
    the evaluation harnesses; it accepts a few extra keyword aliases so callers
    with slightly different signatures keep working.
    """
    if "prompt" in kwargs and template is None:
        template = kwargs.pop("prompt")
    if "answer_type" in kwargs:
        # LLaVA passes an ``answer_type`` hint for TextVQA; the Addendum does
        # not specify per-answer-type prompts, so it is ignored (logged).
        LOGGER.debug("Ignoring answer_type=%r (not specified by the Addendum)",
                     kwargs.pop("answer_type"))
    kwargs.pop("kind", None)
    kwargs.pop("in_place", None)
    return build_task_prompt(
        sample,
        question=question,
        choices=choices,
        task=task or dataset_name or "vqa",
        model=model,
        dataset_name=dataset_name,
        template=template,
    )


def build_vqa_prompt(sample: Any = None, **kwargs: Any) -> Optional[str]:
    """Alias of :func:`build_question_prompt` with ``task='vqa'``."""
    kwargs.setdefault("task", "vqa")
    return build_question_prompt(sample, **kwargs)


def build_pope_prompt(sample: Any = None, **kwargs: Any) -> Optional[str]:
    """POPE prompt (yes/no short answer)."""
    kwargs.setdefault("task", "pope")
    return build_question_prompt(sample, **kwargs)


def build_sqa_prompt(sample: Any = None, **kwargs: Any) -> Optional[str]:
    """SQA-I prompt (answer with the option letter)."""
    kwargs.setdefault("task", "sqa")
    return build_question_prompt(sample, **kwargs)


def build_caption_prompt(
    prompt: Optional[str] = None,
    *,
    model: str = "llava",
    template: Optional[str] = None,
    with_image_token: bool = True,
) -> str:
    """Captioning prompt for LLaVA/OpenFlamingo (``<image>`` prefixed)."""
    resolved_model = _normalize_model(model)
    base = _template_for("caption", resolved_model, template=template)
    if prompt is not None:
        base = str(prompt)
    return ensure_image_token(base) if with_image_token else base


def build_prompt(
    sample: Any = None,
    model: str = "llava",
    *,
    dataset_name: Optional[str] = None,
    task: Optional[str] = None,
    template: Optional[str] = None,
    question: Optional[str] = None,
    **kwargs: Any,
) -> Optional[str]:
    """Generic dispatcher used by the harnesses.

    Captioning (``task='caption'``) returns the caption continuation prompt;
    question-answering tasks return an ``<image>``-prefixed question prompt.
    """
    resolved_task = _normalize_task(task or dataset_name)
    if resolved_task == "caption":
        return build_caption_prompt(template=template, model=model)
    return build_question_prompt(
        sample,
        model=model,
        dataset_name=dataset_name,
        task=resolved_task,
        template=template,
        question=question,
        **kwargs,
    )


# Short aliases used across the harnesses. ---------------------------------- #


def question_prompt(sample: Any = None, **kwargs: Any) -> Optional[str]:
    """Alias of :func:`build_question_prompt`."""
    return build_question_prompt(sample, **kwargs)


def pope_prompt(sample: Any = None, **kwargs: Any) -> Optional[str]:
    return build_pope_prompt(sample, **kwargs)


def sqa_prompt(sample: Any = None, **kwargs: Any) -> Optional[str]:
    return build_sqa_prompt(sample, **kwargs)


def caption_prompt(prompt: Optional[str] = None, **kwargs: Any) -> str:
    """Alias of :func:`build_caption_prompt` (LLaVA default)."""
    return build_caption_prompt(prompt, **kwargs)


def llava_caption_prompt(prompt: Optional[str] = None, **kwargs: Any) -> str:
    return build_caption_prompt(prompt, model="llava", **kwargs)


def openflamingo_vqa_prompt(sample: Any = None, **kwargs: Any) -> Optional[str]:
    """OpenFlamingo VQA prompt builder."""
    kwargs.setdefault("model", "openflamingo")
    kwargs.setdefault("task", "vqa")
    return build_question_prompt(sample, **kwargs)


def openflamingo_pope_prompt(sample: Any = None, **kwargs: Any) -> Optional[str]:
    """OpenFlamingo POPE prompt builder."""
    kwargs.setdefault("model", "openflamingo")
    kwargs.setdefault("task", "pope")
    return build_question_prompt(sample, **kwargs)


def openflamingo_caption_prompt(
    prompt: Optional[str] = None, **kwargs: Any
) -> str:
    """OpenFlamingo captioning prompt builder."""
    return build_caption_prompt(prompt, model="openflamingo", **kwargs)


def openflamingo_prompt(
    sample: Any = None,
    *,
    task: str = "vqa",
    **kwargs: Any,
) -> Optional[str]:
    """Generic OpenFlamingo prompt dispatcher."""
    resolved = _normalize_task(task)
    if resolved == "caption":
        return openflamingo_caption_prompt(**kwargs)
    return build_question_prompt(
        sample, model="openflamingo", task=resolved, **kwargs
    )


# --------------------------------------------------------------------------- #
# Registries / introspection                                                  #
# --------------------------------------------------------------------------- #


def registered_prompts() -> Dict[str, str]:
    """Return ``{"model.task": template}`` for every registered default."""
    registry: Dict[str, str] = {}
    for model in ("llava", "openflamingo"):
        for task in ("vqa", "pope", "sqa", "caption"):
            if model == "openflamingo" and task == "sqa":
                continue
            registry[f"{model}.{task}"] = _template_for(task, model)
    return registry


def get_prompt(
    task: str = "vqa",
    *,
    model: str = "llava",
    template: Optional[str] = None,
) -> str:
    """Return the raw (unformatted) template for ``(model, task)``."""
    return _template_for(_normalize_task(task), _normalize_model(model), template=template)


def prompt_provenance() -> Dict[str, Any]:
    """Machine-readable provenance for the prompt layer.

    The Addendum pins the upstream repositories but contains no literal strings,
    so ``unspecified_by_addendum`` lists every template name; ``upstream`` lists
    the repositories the wording is inherited from.
    """
    return {
        "addendum_pins": {
            "llava": "https://github.com/haotian-liu/LLaVA/tree/main",
            "openflamingo": "https://github.com/mlfoundations/open_flamingo/tree/main",
        },
        "upstream": sorted({entry["source"] for entry in PROMPT_SOURCES.values()}),
        "unspecified_by_addendum": sorted(PROMPT_SOURCES.keys()),
        "note": (
            "The Addendum references prompt details but includes no literal "
            "templates; they are inherited from the pinned upstream repositories "
            "and overridable via configs/models.yaml (prompts.*)."
        ),
    }


# --------------------------------------------------------------------------- #
# CLI / self-test                                                             #
# --------------------------------------------------------------------------- #


def _self_test(verbose: bool = True) -> Dict[str, Any]:
    """Offline checks that require no models, data, or network access."""
    checks: Dict[str, Any] = {}

    # 1. Registries are populated.
    registry = registered_prompts()
    assert "llava.vqa" in registry and "openflamingo.caption" in registry, registry
    checks["num_registered_prompts"] = len(registry)

    # 2. LLaVA VQA prompt: <image> prefix + formatted question + short suffix.
    sample = {"question": "What is the text in this image?"}
    vqa = build_question_prompt(sample, dataset_name="TextVQA")
    assert vqa is not None and vqa.startswith(DEFAULT_IMAGE_TOKEN), vqa
    assert "What is the text in this image?" in vqa, vqa
    assert "single word or phrase" in vqa, vqa
    checks["vqa_prompt"] = vqa

    # 3. Bare string / object / tuple inputs all work.
    assert "What color" in (build_vqa_prompt("What color is the car?") or "")
    class _S:  # minimal duck-typed sample
        question = "Is there a dog?"
        choices = None
    assert "Is there a dog?" in (build_pope_prompt(_S()) or "")
    assert "Is there a dog?" in (build_vqa_prompt(({"question": "Is there a dog?"}, None)) or "")
    checks["duck_typed_inputs"] = True

    # 4. SQA renders the choice letters.
    sqa = build_sqa_prompt({"question": "Which is heavier?", "choices": ["cat", "car"]})
    assert "A. cat" in sqa and "B. car" in sqa, sqa
    assert "option's letter" in sqa, sqa
    checks["sqa_prompt"] = sqa

    # 5. No question -> None for QA tasks (captioning callers fall back).
    assert build_question_prompt({"question": None}) is None
    assert build_question_prompt({}) is None
    checks["missing_question_returns_none"] = True

    # 6. Captioning prompt for both victims, with the image token.
    for model in ("llava", "openflamingo"):
        cap = build_caption_prompt(model=model)
        assert cap.startswith(DEFAULT_IMAGE_TOKEN), cap
        assert "caption" in cap.lower(), cap
    checks["caption_prompts"] = {
        "llava": build_caption_prompt(model="llava"),
        "openflamingo": build_caption_prompt(model="openflamingo"),
    }
    assert build_caption_prompt(with_image_token=False) == CAPTION_PROMPT

    # 7. Generic dispatcher honours the task switch.
    assert build_prompt(sample, task="caption") == build_caption_prompt()
    assert build_prompt(sample, task="vqa") == build_question_prompt(sample)
    assert "single word or phrase" in (openflamingo_prompt(sample, task="vqa") or "")
    assert "caption" in (openflamingo_prompt(task="caption") or "").lower()
    checks["dispatcher"] = True

    # 8. Task aliases resolve to canonical tasks.
    for alias, canonical in (
        ("TextVQA", "vqa"),
        ("POPE", "pope"),
        ("SQA-I", "sqa"),
        ("scienceqa", "sqa"),
        ("COCO", "caption"),
        ("Flickr30k", "caption"),
    ):
        assert _normalize_task(alias) == canonical, (alias, _normalize_task(alias))
    checks["task_aliases"] = True

    # 9. Provenance: nothing here is claimed as paper-stated.
    prov = prompt_provenance()
    assert prov["unspecified_by_addendum"], prov
    assert len(prov["upstream"]) == 2, prov
    assert "haotian-liu/LLaVA" in prov["upstream"][0] or any(
        "LLaVA" in u for u in prov["upstream"]
    ), prov
    checks["provenance"] = prov

    # 10. Image-token helpers.
    assert ensure_image_token("hello") == f"{DEFAULT_IMAGE_TOKEN}\nhello"
    assert ensure_image_token(f"{DEFAULT_IMAGE_TOKEN}\nx") == f"{DEFAULT_IMAGE_TOKEN}\nx"
    assert has_image_token(DEFAULT_IMAGE_TOKEN) and not has_image_token("plain")
    checks["image_token_helpers"] = True

    # 11. Templates accept overrides from configs/models.yaml.
    override = build_question_prompt(sample, template="{question} Reply briefly.")
    assert override == f"{DEFAULT_IMAGE_TOKEN}\nWhat is the text in this image? Reply briefly.", override
    checks["template_override"] = True

    if verbose:
        print(json.dumps(checks, indent=2, default=str)[:4000])
        print("\n[prompts.templates] all self-tests passed")
    return checks


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m robust_clip_repro.prompts.templates",
        description=(
            "Prompt templates for the Robust CLIP reproduction. The Addendum "
            "includes no literal prompt strings, so templates are inherited from "
            "haotian-liu/LLaVA and mlfoundations/open_flamingo (the pinned "
            "upstreams) and are overridable via configs/models.yaml."
        ),
    )
    parser.add_argument("--self-test", action="store_true", help="run offline checks")
    parser.add_argument("--task", default=None, help="vqa|pope|sqa|caption")
    parser.add_argument("--model", default="llava", help="llava|openflamingo")
    parser.add_argument("--question", default=None, help="question text to format")
    parser.add_argument("--provenance", action="store_true", help="print provenance JSON")
    parser.add_argument("--registry", action="store_true", help="print default templates")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.self_test:
        _self_test(verbose=not args.quiet)
        return 0

    if args.provenance:
        print(json.dumps(prompt_provenance(), indent=2))
        return 0

    if args.registry:
        print(json.dumps(registered_prompts(), indent=2))
        return 0

    if args.question or args.task:
        task = args.task or "vqa"
        if _normalize_task(task) == "caption":
            print(build_caption_prompt(model=args.model))
        else:
            prompt = build_question_prompt(
                {"question": args.question} if args.question else None,
                task=task,
                model=args.model,
            )
            print(prompt if prompt is not None else "(no question supplied)")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
