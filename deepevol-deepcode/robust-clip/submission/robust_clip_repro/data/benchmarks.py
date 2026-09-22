"""TextVQA / POPE / SQA-I benchmark loaders for the Robust CLIP reproduction.

Addendum (in scope, implemented verbatim here):
  * "The implementations for the POPE benchmark and the SQA-I benchmarks are also
    taken from the repository" -- i.e. https://github.com/haotian-liu/LLaVA/tree/main.
    The dataset/split names, image handling and answer normalisation below mirror the
    LLaVA harnesses of that repository.
  * "For visual-question answering, low-precision attacks are performed on the top 5
    frequent ground truth ..." -- so this module exposes the *top-k most frequent
    ground truths* of a benchmark plus the per-sample scoring hooks the VQA attack
    scheduler (``attacks/vqa_schedule.py``) needs.
  * The attack with target "Word" "is not done on TextVQA" -- therefore the module
    exposes :func:`should_skip_word_attack` and marks TextVQA samples accordingly.

Nothing here invents a paper hyperparameter: the definition of the numeric VQA
"score" used to pick the ground truth with the *lowest* score is **not** given by the
Addendum, so it is exposed as a configurable hook (:func:`make_score_fn`) whose
default is documented as externally supplied (``UNSPECIFIED_BY_ADDENDUM``).

Only light dependencies are imported eagerly (torch is imported lazily so that the
answer-normalisation helpers can be unit tested without a GPU); the heavy
``datasets`` download happens lazily inside :func:`load_benchmark`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import string
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

LOGGER = logging.getLogger("robust_clip_repro.data.benchmarks")

# --------------------------------------------------------------------------------------
# Dataset identity
# --------------------------------------------------------------------------------------

TEXT_VQA = "TextVQA"
POPE = "POPE"
SQA_I = "SQA-I"

DATASET_NAMES: Tuple[str, ...] = (TEXT_VQA, POPE, SQA_I)

#: Case/format-insensitive aliases (roster also accepts the raw LLaVA names).
DATASET_ALIASES: Dict[str, str] = {
    "textvqa": TEXT_VQA,
    "text_vqa": TEXT_VQA,
    "text-vqa": TEXT_VQA,
    "textvqa_0.5.1_val": TEXT_VQA,
    "pope": POPE,
    "pope_adv": POPE,
    "pope_random": POPE,
    "pope_popular": POPE,
    "sqa": SQA_I,
    "sqa-i": SQA_I,
    "sqa_i": SQA_I,
    "scienceqa": SQA_I,
    "science_qa": SQA_I,
    "science-qa": SQA_I,
}

#: Splits used by the LLaVA harnesses shipped with the paper's repo.
DEFAULT_SPLITS: Dict[str, str] = {
    TEXT_VQA: "validation",
    POPE: "test",
    SQA_I: "test",
}

#: HuggingFace dataset ids. ``hf_kwargs`` (e.g. an explicit ``config``/``name``) can be
#: supplied per dataset because the Addendum does not pin the Hub revision.
HF_DATASET_IDS: Dict[str, str] = {
    TEXT_VQA: "lmms-lab/TextVQA",
    POPE: "lmms-lab/POPE",
    SQA_I: "lmms-lab/ScienceQA",
}

HF_KWARGS: Dict[str, Dict[str, Any]] = {
    TEXT_VQA: {},
    POPE: {},
    SQA_I: {},
}

#: LLaVA-repo annotation filenames (used when loading local files instead of the Hub).
LOCAL_ANNOTATIONS: Dict[str, str] = {
    TEXT_VQA: "textvqa_val.json",
    POPE: "coco_pope_random.json",
    SQA_I: "scienceqa_test.json",
}
LOCAL_IMAGES: Dict[str, Optional[str]] = {
    TEXT_VQA: "train_val_images",
    POPE: "val2014",
    SQA_I: "images",
}

#: Addendum: the "Word" targeted attack is skipped on TextVQA.
WORD_ATTACK_SKIPPED: Tuple[str, ...] = (TEXT_VQA,)

TOP_K_GROUND_TRUTHS = 5
UNSPECIFIED = "UNSPECIFIED_BY_ADDENDUM"

# --------------------------------------------------------------------------------------
# Shared ground-truth frequency type (defined once, in the attack scheduler)
# --------------------------------------------------------------------------------------

try:  # pragma: no cover - trivial import path
    from ..attacks.vqa_schedule import (  # type: ignore
        TOP_K_GROUND_TRUTHS as _TOP_K,
        GroundTruthFrequency,
        most_frequent_ground_truth,
        most_frequent_ground_truths,
    )

    TOP_K_GROUND_TRUTHS = int(_TOP_K)
    _SHARED_FREQUENCY_TYPE = True
except Exception:  # pragma: no cover - fallback keeps this module importable alone
    _SHARED_FREQUENCY_TYPE = False

    @dataclass(frozen=True)
    class GroundTruthFrequency:  # type: ignore[no-redef]
        """Frequency of one ground-truth string (fallback definition)."""

        answer: str
        count: int

        def as_dict(self) -> Dict[str, Any]:
            return {"answer": self.answer, "count": self.count}

    def most_frequent_ground_truths(
        samples: Sequence[Any], k: int = TOP_K_GROUND_TRUTHS
    ) -> List["GroundTruthFrequency"]:
        counts = ground_truth_counts(samples)
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return [GroundTruthFrequency(answer=a, count=c) for a, c in ranked[: int(k)]]

    def most_frequent_ground_truth(samples: Sequence[Any]) -> Optional[str]:
        ranked = most_frequent_ground_truths(samples, k=1)
        return ranked[0].answer if ranked else None


# --------------------------------------------------------------------------------------
# Answer normalisation (LLaVA / VQA-v2 conventions)
# --------------------------------------------------------------------------------------

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: Any) -> str:
    """VQA-style answer normalisation: lowercase, strip punctuation/articles/space."""

    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower().strip()
    text = text.translate(_PUNCT_TABLE)
    text = _ARTICLES.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text


def normalize_dataset_name(name: Any) -> str:
    """Map any accepted alias to one of :data:`DATASET_NAMES`."""

    if name is None:
        raise ValueError(
            f"dataset name is required; expected one of {DATASET_NAMES}"
        )
    raw = str(name).strip()
    key = raw.lower().replace(" ", "")
    if raw in DATASET_NAMES:
        return raw
    if key in DATASET_ALIASES:
        return DATASET_ALIASES[key]
    # Loose matching ("SQuAD"-style substrings) for convenience.
    for canonical in DATASET_NAMES:
        if canonical.lower().replace("-", "").replace("_", "") in key.replace("-", "").replace("_", ""):
            return canonical
    raise ValueError(f"unknown VQA benchmark {name!r}; expected one of {DATASET_NAMES}")


def is_textvqa(dataset_name: Any) -> bool:
    """True for TextVQA (drives the Addendum's "skip the `Word` attack" rule)."""

    try:
        return normalize_dataset_name(dataset_name) == TEXT_VQA
    except ValueError:
        return False


def should_skip_word_attack(dataset_name: Any) -> bool:
    """Addendum: "the attack with Word is not done on TextVQA"."""

    return is_textvqa(dataset_name)


# --------------------------------------------------------------------------------------
# Sample container
# --------------------------------------------------------------------------------------


@dataclass
class VQASample:
    """Canonical, model-agnostic VQA sample.

    ``answers`` holds every acceptable ground-truth string (the frequency statistics
    used by the Addendum consume one *primary* ground truth per sample, see
    :meth:`ground_truth`).
    """

    sample_id: Union[int, str]
    dataset: str
    question: str
    answers: List[str] = field(default_factory=list)
    image: Any = None
    image_id: Optional[Union[int, str]] = None
    image_path: Optional[str] = None
    choices: Optional[List[str]] = None
    answer_index: Optional[int] = None
    prompt: Optional[str] = None
    split: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- convenience ------------------------------------------------------------------
    @property
    def ground_truth(self) -> str:
        """Primary ground truth; for multiple-choice tasks the chosen choice."""

        if self.choices is not None and self.answer_index is not None:
            try:
                return str(self.choices[int(self.answer_index)])
            except (IndexError, ValueError, TypeError):
                pass
        return self.answers[0] if self.answers else ""

    #: alias so callers may use either spelling
    @property
    def ground_truths(self) -> List[str]:
        return list(self.answers)

    @property
    def answer(self) -> str:
        return self.ground_truth

    @property
    def skip_word_attack(self) -> bool:
        return should_skip_word_attack(self.dataset)

    def to_dict(self, include_image: bool = False) -> Dict[str, Any]:
        payload = asdict(self) if include_image else {
            k: v for k, v in asdict(self).items() if k != "image"
        }
        payload["ground_truth"] = self.ground_truth
        payload.pop("image", None) if not include_image else None
        if include_image:
            img = self.image
            payload["image"] = None if img is None else type(img).__name__
        return payload


def ground_truths_of(sample: Any) -> List[str]:
    """All acceptable ground truths for a sample (dict or :class:`VQASample`)."""

    if isinstance(sample, VQASample):
        return list(sample.answers) or ([sample.ground_truth] if sample.ground_truth else [])
    if isinstance(sample, dict):
        if sample.get("answers"):
            return _coerce_answer_list(sample["answers"])
        gt = sample.get("ground_truth")
        return [gt] if isinstance(gt, str) and gt else []
    if isinstance(sample, str):
        return [sample]
    return []


def ground_truth_of(sample: Any) -> str:
    """Primary ground truth of a sample (dict or :class:`VQASample`)."""

    if isinstance(sample, VQASample):
        return sample.ground_truth
    if isinstance(sample, dict):
        if sample.get("ground_truth"):
            return str(sample["ground_truth"])
        if sample.get("choices") is not None and sample.get("answer_index") is not None:
            try:
                return str(sample["choices"][int(sample["answer_index"])])
            except (IndexError, ValueError, TypeError):
                pass
        answers = ground_truths_of(sample)
        return answers[0] if answers else ""
    return str(sample)


def ground_truth_counts(samples: Sequence[Any]) -> Dict[str, int]:
    """Frequency of each primary ground truth across ``samples``."""

    counts: Dict[str, int] = {}
    for sample in samples:
        gt = ground_truth_of(sample)
        if gt == "":
            continue
        counts[gt] = counts.get(gt, 0) + 1
    return counts


def top_k_ground_truths(
    samples: Sequence[Any], k: int = TOP_K_GROUND_TRUTHS
) -> List[GroundTruthFrequency]:
    """The ``k`` most frequent ground truths.

    Addendum: "low-precision attacks are performed on the top 5 frequent ground
    truth".  Ties are broken by first appearance so the ordering is deterministic
    (the shared scheduler implementation does the same).
    """

    try:
        return most_frequent_ground_truths(samples, k=k)  # type: ignore[misc]
    except TypeError:  # pragma: no cover - older signature
        counts = ground_truth_counts(samples)
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        return [GroundTruthFrequency(answer=a, count=c) for a, c in ranked[: int(k)]]


def top_ground_truth(samples: Sequence[Any]) -> Optional[str]:
    """The single most frequent ground truth (target of the "maybe"/"Word" attacks)."""

    try:
        return most_frequent_ground_truth(samples)  # type: ignore[misc]
    except TypeError:  # pragma: no cover
        ranked = top_k_ground_truths(samples, k=1)
        return ranked[0].answer if ranked else None


def answer_vocabulary(samples: Sequence[Any]) -> List[str]:
    """All distinct acceptable answers, in first-appearance order."""

    seen: List[str] = []
    seen_set = set()
    for sample in samples:
        for ans in ground_truths_of(sample) or [ground_truth_of(sample)]:
            if ans and ans not in seen_set:
                seen_set.add(ans)
                seen.append(ans)
    return seen


def select_lowest_scoring_ground_truth(
    ground_truths: Sequence[str], scores: Sequence[float]
) -> Tuple[str, int]:
    """Addendum step (2): the ground truth that led to the *lowest* score."""

    if not ground_truths:
        raise ValueError("select_lowest_scoring_ground_truth requires ground truths")
    if len(scores) != len(ground_truths):
        raise ValueError(
            f"scores ({len(scores)}) and ground truths ({len(ground_truths)}) differ"
        )
    best_idx = int(min(range(len(scores)), key=lambda i: (float(scores[i]), i)))
    return str(ground_truths[best_idx]), best_idx


def argmin_ground_truth(
    ground_truths: Sequence[str], scores: Sequence[float]
) -> str:
    """Convenience wrapper returning only the arg-min ground truth."""

    return select_lowest_scoring_ground_truth(ground_truths, scores)[0]


# --------------------------------------------------------------------------------------
# Raw schema normalisation
# --------------------------------------------------------------------------------------

_ANSWER_KEYS = ("answers", "answer", "ground_truth", "gt", "label", "target")
_QUESTION_KEYS = ("question", "query", "problem", "text", "prompt", "instruction")
_IMAGE_KEYS = ("image", "image_path", "img", "image_file", "file_name", "image_id")


def _coerce_answer_list(value: Any) -> List[str]:
    """Normalise the many `answers` encodings into a list of strings."""

    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        # LLaVA TextVQA v0.5.1 stores {"answer": [...], "question_id": ...}
        for key in ("answer", "answers", "text"):
            if key in value:
                return _coerce_answer_list(value[key])
        return []
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, dict):
                for key in ("answer", "text", "caption", "label"):
                    if key in item:
                        out.extend(_coerce_answer_list(item[key]))
                        break
            elif isinstance(item, (list, tuple)):
                out.extend(_coerce_answer_list(item))
            else:
                out.append(str(item))
        # order-preserving dedup
        seen, uniq = set(), []
        for item in out:
            if item not in seen:
                seen.add(item)
                uniq.append(item)
        return uniq
    return [str(value)]


def _first_present(example: Dict[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for key in keys:
        if key in example and example[key] not in (None, ""):
            return example[key]
    return None


def extract_image(example: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
    """Return ``(image_or_None, image_path_or_None)`` for a raw example."""

    for key in _IMAGE_KEYS:
        value = example.get(key)
        if value is None:
            continue
        if isinstance(value, (str, os.PathLike)):
            return None, str(value)
        return value, None  # PIL.Image or dict with "bytes"/"path"
    return None, None


def normalize_sample(
    example: Dict[str, Any],
    dataset_name: Any,
    index: int = 0,
    *,
    split: Optional[str] = None,
    image_root: Optional[Union[str, os.PathLike]] = None,
) -> VQASample:
    """Convert a raw Hub/annotation example into a :class:`VQASample`.

    Supported schemas (LLaVA repository conventions):

    * **TextVQA** -- ``{"question", "answers": [{"answer": ...}] | [...], "image"}``
    * **POPE** -- ``{"question", "answer"/"label": "yes"|"no", "image"}``
    * **SQA-I** (ScienceQA) -- ``{"question", "choices", "answer": <index>, "image"}``
    """

    dataset = normalize_dataset_name(dataset_name)

    question = _first_present(example, _QUESTION_KEYS)
    if question is None:
        text_choices = example.get("choices")
        question = example.get("question") or (
            " ".join(str(c) for c in text_choices) if text_choices else ""
        )
    question = "" if question is None else str(question).strip()

    image, image_path = extract_image(example)
    if image_path is not None and image_root is not None:
        candidate = Path(image_root) / image_path
        image_path = str(candidate) if candidate.exists() else image_path

    choices = example.get("choices")
    choices = [str(c) for c in choices] if isinstance(choices, (list, tuple)) else None
    answer_index: Optional[int] = None

    answers: List[str] = []
    if dataset == SQA_I and choices is not None:
        raw_answer = example.get("answer", example.get("solution", example.get("label")))
        if isinstance(raw_answer, str) and raw_answer in choices:
            answer_index = choices.index(raw_answer)
            answers = [raw_answer]
        else:
            try:
                answer_index = int(raw_answer)
                answers = [choices[answer_index]]
            except (TypeError, ValueError, IndexError):
                answers = _coerce_answer_list(raw_answer)
    else:
        raw_answer = _first_present(example, _ANSWER_KEYS)
        if isinstance(example.get("answers"), dict):
            raw_answer = example["answers"]
        answers = _coerce_answer_list(raw_answer)
        for key in _ANSWER_KEYS:  # merge alternative answer fields
            extra = example.get(key)
            if key == "answers" or extra is None:
                continue
            extra_list = _coerce_answer_list(extra)
            for item in extra_list:
                if not answers:
                    answers.append(item)
        seen, uniq = set(), []
        for item in answers:
            if item not in seen:
                seen.add(item)
                uniq.append(item)
        answers = uniq

    sample_id: Union[int, str] = index
    for key in ("question_id", "id", "sample_id", "qid", "problem_id", "image_id"):
        if key in example and example[key] is not None and key != "image_id":
            sample_id = example[key]
            break

    metadata = {
        k: v
        for k, v in example.items()
        if k not in ("image",) and isinstance(v, (int, float, str, bool, type(None)))
    }

    return VQASample(
        sample_id=sample_id,
        dataset=dataset,
        question=question,
        answers=answers,
        image=image,
        image_id=example.get("image_id"),
        image_path=image_path,
        choices=choices,
        answer_index=answer_index,
        split=split,
        metadata=metadata,
    )


# --------------------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------------------

#: Fallback prompts mirroring the two upstream repositories.  The Addendum says prompt
#: details live in the graders' rubric ("Details about prompts for LLaVA and
#: OpenFlamingo ... the rubric will check for implementations of things asked for from
#: the code below"), so these are *defaults* and any value supplied by
#: ``prompts/templates.py`` takes precedence.
DEFAULT_TEMPLATES: Dict[str, str] = {
    "llava": "{question}\nAnswer the question using a single word or phrase.",
    "openflamingo": "<image>{question}\nAnswer:",
}

PROMPT_TASK_DEFAULT = UNSPECIFIED


def build_prompt(
    sample: Union[VQASample, Dict[str, Any]],
    model_name: str = "llava",
    *,
    dataset_name: Optional[str] = None,
    template: Optional[str] = None,
) -> str:
    """Build the task prompt for one sample.

    Delegates to ``prompts/templates.py`` when that module provides a matching helper,
    otherwise falls back to :data:`DEFAULT_TEMPLATES`.
    """

    question = sample.question if isinstance(sample, VQASample) else str(
        sample.get("question", "")
    )
    dataset = dataset_name or (
        sample.dataset if isinstance(sample, VQASample) else None
    )
    key = str(model_name).lower()

    if template is None:
        try:  # pragma: no cover - optional dependency on the prompt module
            from ..prompts import templates as _templates  # type: ignore

            for fn_name in ("build_question_prompt", "build_vqa_prompt", "question_prompt"):
                fn = getattr(_templates, fn_name, None)
                if callable(fn):
                    try:
                        return fn(question, model_name=key, dataset_name=dataset)
                    except TypeError:
                        try:
                            return fn(question, dataset_name=dataset)
                        except TypeError:
                            continue
        except Exception:  # pragma: no cover - prompt module optional
            pass

    fmt = DEFAULT_TEMPLATES.get(key) or DEFAULT_TEMPLATES["llava"]
    return fmt.format(question=question)


def attach_prompts(
    samples: Sequence[VQASample], model_name: str = "llava", *, in_place: bool = True
) -> List[VQASample]:
    """Fill ``sample.prompt`` for every sample."""

    out = list(samples) if in_place else [
        VQASample(**{**asdict(s), "image": s.image}) for s in samples
    ]
    for sample in out:
        sample.prompt = build_prompt(sample, model_name)
    return out


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def _resolve_split(dataset: str, split: Optional[str]) -> str:
    if split:
        return str(split)
    return DEFAULT_SPLITS.get(dataset, "test")


def _load_hf_dataset(
    dataset: str,
    split: str,
    *,
    dataset_id: Optional[str] = None,
    hf_kwargs: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
    trust_remote_code: bool = True,
    streaming: bool = False,
):
    """``datasets.load_dataset`` wrapper (lazy import, non-interactive)."""

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Loading the VQA benchmarks requires the HuggingFace `datasets` package "
            "(`pip install datasets`). See the Addendum: the HF loader is used with "
            "trust_remote_code=True to avoid waiting for stdin."
        ) from exc

    kwargs: Dict[str, Any] = dict(hf_kwargs or HF_KWARGS.get(dataset, {}))
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if trust_remote_code:
        # Addendum guidance (stated for imagenet-1k): avoids waiting for stdin.
        kwargs.setdefault("trust_remote_code", True)
    if streaming:
        kwargs["streaming"] = True

    ds_id = dataset_id or HF_DATASET_IDS[dataset]
    LOGGER.info("loading HF dataset %s split=%s kwargs=%s", ds_id, split, sorted(kwargs))
    return load_dataset(ds_id, split=split, **kwargs)


def _load_local_annotations(path: Union[str, os.PathLike]) -> List[Dict[str, Any]]:
    """Read a LLaVA-repo style annotation JSON (``{"data": [...]}`` or a list)."""

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        for key in ("data", "questions", "annotations", "examples"):
            if isinstance(payload.get(key), list):
                return list(payload[key])
        # TextVQA official format: {"data": [...]}, POPE: list, SQA: {"data": [...]}
        return [payload]
    if isinstance(payload, list):
        return list(payload)
    raise ValueError(f"unsupported annotation file layout in {path}")


def load_benchmark(
    dataset_name: Any,
    *,
    split: Optional[str] = None,
    num_samples: Optional[int] = None,
    dataset_id: Optional[str] = None,
    hf_kwargs: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
    local_path: Optional[Union[str, os.PathLike]] = None,
    image_root: Optional[Union[str, os.PathLike]] = None,
    trust_remote_code: bool = True,
    streaming: bool = False,
    shuffle: bool = False,
    seed: int = 0,
    model_name: Optional[str] = None,
    verbose: bool = True,
) -> List[VQASample]:
    """Load one VQA benchmark and normalise it to :class:`VQASample` objects."""

    dataset = normalize_dataset_name(dataset_name)
    resolved_split = _resolve_split(dataset, split)

    if local_path is not None:
        examples = _load_local_annotations(local_path)
        if verbose:
            LOGGER.info("loaded %d raw examples from %s", len(examples), local_path)
    else:
        hf_ds = _load_hf_dataset(
            dataset,
            resolved_split,
            dataset_id=dataset_id,
            hf_kwargs=hf_kwargs,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            streaming=streaming,
        )
        if streaming:
            examples = list(hf_ds.take(int(num_samples))) if num_samples else list(hf_ds)
        else:
            examples = list(hf_ds)

    if shuffle:
        import random

        rng = random.Random(seed)
        rng.shuffle(examples)

    if num_samples is not None:
        examples = examples[: int(num_samples)]

    samples = [
        normalize_sample(
            ex if isinstance(ex, dict) else dict(ex),
            dataset,
            index=i,
            split=resolved_split,
            image_root=image_root,
        )
        for i, ex in enumerate(examples)
    ]

    if model_name:
        attach_prompts(samples, model_name)

    if verbose:
        counts = ground_truth_counts(samples)
        top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_K_GROUND_TRUTHS]
        LOGGER.info(
            "%s[%s]: %d samples, %d distinct answers, top-%d GTs=%s, skip Word attack=%s",
            dataset,
            resolved_split,
            len(samples),
            len(counts),
            TOP_K_GROUND_TRUTHS,
            top,
            should_skip_word_attack(dataset),
        )
    return samples


def load_textvqa(**kwargs: Any) -> List[VQASample]:
    return load_benchmark(TEXT_VQA, **kwargs)


def load_pope(**kwargs: Any) -> List[VQASample]:
    return load_benchmark(POPE, **kwargs)


def load_sqa_i(**kwargs: Any) -> List[VQASample]:
    return load_benchmark(SQA_I, **kwargs)


LOADERS: Dict[str, Callable[..., List[VQASample]]] = {
    TEXT_VQA: load_textvqa,
    POPE: load_pope,
    SQA_I: load_sqa_i,
}


def load_benchmarks(
    datasets: Sequence[Any] = DATASET_NAMES, **kwargs: Any
) -> Dict[str, List[VQASample]]:
    """Load several benchmarks; returns ``{canonical_name: samples}``."""

    out: Dict[str, List[VQASample]] = {}
    for name in datasets:
        canonical = normalize_dataset_name(name)
        out[canonical] = LOADERS[canonical](**kwargs)
    return out


# --------------------------------------------------------------------------------------
# Per-sample scoring hooks (consumed by attacks/vqa_schedule.py)
# --------------------------------------------------------------------------------------

#: The Addendum does not define the numeric "score" used to select the arg-min ground
#: truth, therefore the default below is explicitly recorded as externally supplied.
SCORE_KINDS: Tuple[str, ...] = (
    "accuracy",            # 1.0 if the model answer matches the GT else 0.0 (default)
    "negative_logprob",    # -mean log prob of the GT answer tokens
    "negative_logit",      # -max logit of the GT answer tokens
    "negative_prob",       # -probability the model assigns to the GT answer
    "edit_distance",       # -normalised similarity between prediction and GT
)
DEFAULT_SCORE_KIND = "accuracy"
SCORE_KIND_PROVENANCE = UNSPECIFIED


def vqa_accuracy(
    prediction: Any,
    ground_truths: Union[Any, Sequence[Any]],
    dataset_name: Optional[Any] = None,
    *,
    max_answers: Optional[int] = None,
) -> float:
    """VQA-v2 style soft accuracy (``min(#matches / 3, 1)``) / exact match.

    LLaVA's harnesses use the soft accuracy for TextVQA and exact match for
    POPE/SQA-I; with a single ground truth both coincide, so the soft form is the
    default here and reduces to exact match automatically.
    """

    answers = _coerce_answer_list(ground_truths)
    if not answers:
        return 0.0
    if max_answers is not None:
        answers = answers[: int(max_answers)]
    pred = normalize_answer(prediction)
    if pred == "":
        return 0.0
    matches = sum(1 for ans in answers if normalize_answer(ans) == pred)
    if len(answers) == 1:
        return 1.0 if matches else 0.0
    return min(matches / 3.0, 1.0)


def isolated_accuracy(prediction: Any, ground_truths: Union[Any, Sequence[Any]]) -> float:
    """Exact-match accuracy (LLaVA's POPE / SQA-I convention)."""

    answers = _coerce_answer_list(ground_truths)
    if not answers:
        return 0.0
    pred = normalize_answer(prediction)
    return 1.0 if any(pred == normalize_answer(a) for a in answers) else 0.0


def sample_accuracy(
    prediction: Any, sample: Union[VQASample, Dict[str, Any], str], dataset_name: Optional[Any] = None
) -> float:
    """Per-sample accuracy hook, dispatching on the benchmark."""

    if isinstance(sample, VQASample):
        dataset = dataset_name or sample.dataset
        truths = sample.answers or [sample.ground_truth]
    elif isinstance(sample, dict):
        dataset = dataset_name or sample.get("dataset")
        truths = ground_truths_of(sample)
    else:
        dataset = dataset_name
        truths = [sample]

    if dataset is not None and normalize_dataset_name(dataset) in (POPE, SQA_I):
        return isolated_accuracy(prediction, truths)
    return vqa_accuracy(prediction, truths)


def _extract_answer(value: Any) -> Tuple[str, Optional[float]]:
    """Pull ``(answer_text, score_or_None)`` out of a victim's return value."""

    if isinstance(value, dict):
        text = value.get("answer") or value.get("text") or value.get("response") or ""
        score = value.get("logprob", value.get("score", value.get("prob")))
        return str(text), (float(score) if isinstance(score, (int, float)) else None)
    return str(value), None


def make_score_fn(
    victim: Callable[..., Any],
    *,
    kind: str = DEFAULT_SCORE_KIND,
    dataset_name: Optional[Any] = None,
    answer_parser: Optional[Callable[[str], str]] = None,
    **victim_kwargs: Any,
) -> Callable[[Any, str], float]:
    """Build the ``score_fn(pixels, ground_truth) -> float`` hook.

    The scheduler performs the Addendum's step (2) -- "a high-precision attack is done
    on the ground truth that led to the lowest score for each sample" -- by taking the
    arg-min over the top-5 low-precision results, so **lower must mean worse**.

    ``victim`` is any callable ``victim(pixels, question, **kwargs) -> str | dict``.
    The Addendum does not define this score; ``kind`` therefore defaults to
    ``"accuracy"`` and is recorded as externally supplied
    (:data:`SCORE_KIND_PROVENANCE`).
    """

    if kind not in SCORE_KINDS:
        raise ValueError(
            f"unknown VQA score kind {kind!r}; expected one of {SCORE_KINDS} "
            f"(the Addendum leaves the score definition unspecified: {UNSPECIFIED})"
        )
    question = victim_kwargs.pop("question", None)

    def score_fn(pixels: Any, ground_truth: str, question: Optional[str] = None) -> float:  # type: ignore[assignment]
        raw = victim(pixels, question, **victim_kwargs) if question is not None else victim(pixels)
        prediction, model_score = _extract_answer(raw)
        if answer_parser is not None:
            prediction = answer_parser(prediction)

        if kind == "accuracy":
            return sample_accuracy(prediction, ground_truth, dataset_name)
        if kind in ("negative_logprob", "negative_prob", "negative_logit"):
            if model_score is None:
                raise ValueError(
                    f"score kind {kind!r} requires the victim to return a numeric "
                    "'logprob'/'prob'/'score' field"
                )
            return -float(model_score)
        if kind == "edit_distance":
            pred, truth = normalize_answer(prediction), normalize_answer(ground_truth)
            import difflib

            return -float(difflib.SequenceMatcher(None, pred, truth).ratio())
        raise ValueError(f"unreachable score kind {kind!r}")  # pragma: no cover

    score_fn.kind = kind  # type: ignore[attr-defined]
    score_fn.provenance = SCORE_KIND_PROVENANCE  # type: ignore[attr-defined]
    return score_fn


def make_constant_score_fn(value: float = 0.0) -> Callable[[Any, str], float]:
    """Model-free score hook (for unit tests of the scheduler ordering)."""

    def score_fn(pixels: Any, ground_truth: str) -> float:
        return float(value)

    score_fn.kind = "constant"  # type: ignore[attr-defined]
    score_fn.provenance = UNSPECIFIED  # type: ignore[attr-defined]
    return score_fn


# --------------------------------------------------------------------------------------
# Image / pixel-space helpers (PGD projects in NON-normalised pixel space)
# --------------------------------------------------------------------------------------


def image_to_pixels(
    image: Any,
    *,
    resolution: Optional[int] = 224,
    device: Any = None,
    dtype: Any = None,
    in01: bool = True,
) -> Any:
    """Convert a PIL image / tensor / path to ``(1, 3, H, W)`` float pixels in [0, 1]."""

    import torch  # lazy

    if isinstance(image, (str, os.PathLike)):
        from PIL import Image  # lazy

        image = Image.open(image).convert("RGB")
    if isinstance(image, torch.Tensor):
        tensor = image.detach().float()
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[1] not in (1, 3) and tensor.shape[-1] in (1, 3):
            tensor = tensor.permute(0, 3, 1, 2)
        if tensor.max() > 1.5:
            tensor = tensor / 255.0
        pixels = tensor
    else:
        try:
            import numpy as np  # lazy

            array = np.asarray(image)
        except Exception as exc:  # pragma: no cover
            raise TypeError(f"unsupported image type {type(image)!r}") from exc
        arr = torch.from_numpy(array).float()
        if arr.dim() == 2:
            arr = arr.unsqueeze(-1).repeat(1, 1, 3)
        if arr.dim() == 3 and arr.shape[-1] in (1, 3, 4):
            arr = arr[..., :3]
        arr = arr.permute(2, 0, 1).unsqueeze(0)
        if arr.max() > 1.5:
            arr = arr / 255.0
        pixels = arr

    if pixels.shape[1] == 1:
        pixels = pixels.repeat(1, 3, 1, 1) if False else torch.cat([pixels] * 3, dim=1)

    if resolution is not None and (pixels.shape[-1] != int(resolution) or pixels.shape[-2] != int(resolution)):
        import torch.nn.functional as F  # lazy

        pixels = F.interpolate(
            pixels, size=(int(resolution), int(resolution)), mode="bilinear", align_corners=False
        )
    if not in01:
        pixels = pixels * 255.0
    if dtype is not None:
        pixels = pixels.to(dtype)
    if device is not None:
        pixels = pixels.to(device)
    return pixels.contiguous()


def sample_to_pixels(
    sample: Union[VQASample, Dict[str, Any]],
    *,
    resolution: Optional[int] = 224,
    device: Any = None,
    dtype: Any = None,
    image: Any = None,
    in01: bool = True,
) -> Any:
    """Raw (non-normalised) pixel tensor for one sample -- the PGD l_inf ball space."""

    image = image if image is not None else (
        sample.image if isinstance(sample, VQASample) else sample.get("image")
    )
    path = sample.image_path if isinstance(sample, VQASample) else sample.get("image_path")
    if image is None and path:
        image = path
    if image is None:
        raise ValueError("sample has neither an image object nor an image path")
    return image_to_pixels(image, resolution=resolution, device=device, dtype=dtype, in01=in01)


def normalize_pixels(pixels: Any, mean: Sequence[float], std: Sequence[float]) -> Any:
    """Apply model normalisation while leaving the raw pixels untouched for PGD."""

    import torch  # lazy

    mean_t = torch.as_tensor(mean, dtype=pixels.dtype, device=pixels.device).view(1, -1, 1, 1)
    std_t = torch.as_tensor(std, dtype=pixels.dtype, device=pixels.device).view(1, -1, 1, 1)
    return (pixels - mean_t) / std_t


# --------------------------------------------------------------------------------------
# Aggregation helpers used by eval_vqa.py
# --------------------------------------------------------------------------------------


def dataset_statistics(samples: Sequence[VQASample], k: int = TOP_K_GROUND_TRUTHS) -> Dict[str, Any]:
    """Summary statistics a harness can log (frequency table, skip rules, splits)."""

    counts = ground_truth_counts(samples)
    return {
        "dataset": samples[0].dataset if samples else None,
        "num_samples": len(samples),
        "num_distinct_ground_truths": len(counts),
        "top_k_ground_truths": [f.as_dict() for f in top_k_ground_truths(samples, k=k)],
        "most_frequent_ground_truth": top_ground_truth(samples),
        "skip_word_attack": should_skip_word_attack(
            samples[0].dataset if samples else None
        ),
        "top_k": int(k),
    }


# --------------------------------------------------------------------------------------
# Self-test / CLI
# --------------------------------------------------------------------------------------


def _synthetic_samples() -> Dict[str, List[VQASample]]:
    textvqa = [
        normalize_sample(
            {"question_id": i, "question": f"q{i}", "answers": [{"answer": a}]},
            TEXT_VQA,
            index=i,
        )
        for i, a in enumerate(["yes", "yes", "yes", "no", "no", "maybe"])
    ]
    pope = [
        normalize_sample({"question": f"p{i}", "answer": a, "image": f"{i}.jpg"}, POPE, index=i)
        for i, a in enumerate(["yes", "yes", "no"])
    ]
    sqa = [
        normalize_sample(
            {
                "question": f"s{i}",
                "choices": ["a", "b", "c"],
                "answer": i % 3,
                "image": None,
            },
            SQA_I,
            index=i,
        )
        for i in range(3)
    ]
    return {TEXT_VQA: textvqa, POPE: pope, SQA_I: sqa}


def _self_test(verbose: bool = True) -> Dict[str, Any]:
    """Model- and download-free checks of the schemas, statistics and hooks."""

    results: Dict[str, Any] = {}
    samples = _synthetic_samples()

    # --- aliases / skip rule ---------------------------------------------------------
    assert normalize_dataset_name("textvqa") == TEXT_VQA
    assert normalize_dataset_name("scienceqa") == SQA_I
    assert is_textvqa("TextVQA") and should_skip_word_attack("text_vqa")
    assert not should_skip_word_attack(POPE)

    # --- textvqa schema --------------------------------------------------------------
    tv = samples[TEXT_VQA]
    assert tv[0].answers == ["yes"], tv[0].answers
    assert tv[0].ground_truth == "yes"
    stats = dataset_statistics(tv)
    assert stats["most_frequent_ground_truth"] == "yes"
    assert [f.answer for f in top_k_ground_truths(tv, k=5)][:2] == ["yes", "no"]
    assert stats["skip_word_attack"] is True
    results["textvqa"] = stats

    # --- pope schema -----------------------------------------------------------------
    pope = samples[POPE]
    assert all(s.answers for s in pope)
    assert pope[0].image_path == "0.jpg"
    assert dataset_statistics(pope)["most_frequent_ground_truth"] == "yes"
    results["pope"] = dataset_statistics(pope)

    # --- sqa-i schema ----------------------------------------------------------------
    sqa = samples[SQA_I]
    assert sqa[1].choices == ["a", "b", "c"]
    assert sqa[1].answer_index == 1 and sqa[1].ground_truth == "b"
    results["sqa_i"] = dataset_statistics(sqa)

    # --- scoring hooks ---------------------------------------------------------------
    assert vqa_accuracy("Yes", ["yes"]) == 1.0
    assert vqa_accuracy("yes", ["yes", "yes", "yes", "yes"]) == 1.0
    assert vqa_accuracy("maybe", ["yes", "yes", "yes"]) == 0.0
    assert isolated_accuracy(" no ", ["no"]) == 1.0
    assert sample_accuracy("yes", pope[0]) == 1.0

    # arg-min selection ("ground truth that led to the lowest score")
    gts = ["yes", "no", "maybe"]
    gt, idx = select_lowest_scoring_ground_truth(gts, [1.0, 0.5, 0.0])
    assert (gt, idx) == ("maybe", 2)
    assert argmin_ground_truth(gts, [0.0, 1.0, 0.5]) == "yes"

    const_score = make_constant_score_fn(0.25)
    assert const_score(None, "yes") == 0.25

    def _victim(pixels, question=None, **kw):
        return {"answer": "yes", "logprob": -0.2}

    acc_score = make_score_fn(_victim, kind="accuracy")
    assert acc_score(None, "yes") == 1.0 and acc_score(None, "no") == 0.0
    lp_score = make_score_fn(_victim, kind="negative_logprob")
    assert abs(lp_score(None, "yes") - 0.2) < 1e-9
    try:
        make_score_fn(_victim, kind="not-a-score")
        raise AssertionError("unknown score kind must raise")
    except ValueError:
        pass
    results["score_hooks"] = {"default_kind": DEFAULT_SCORE_KIND, "provenance": SCORE_KIND_PROVENANCE}

    # --- prompts ---------------------------------------------------------------------
    prompt = build_prompt(tv[0], "llava")
    assert "q0" in prompt
    results["prompt_example"] = prompt

    # --- pixels (raw, non-normalised) ------------------------------------------------
    try:
        import torch  # noqa: F401
        from PIL import Image

        img = Image.new("RGB", (64, 48), color=(10, 20, 30))
        pixels = image_to_pixels(img, resolution=224)
        assert tuple(pixels.shape) == (1, 3, 224, 224)
        assert float(pixels.max()) <= 1.0 and float(pixels.min()) >= 0.0
        normed = normalize_pixels(pixels, [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        assert normed.shape == pixels.shape and float(normed.abs().max()) > 1.0
        sample = VQASample(sample_id=0, dataset=TEXT_VQA, question="q", answers=["a"], image=img)
        raw = sample_to_pixels(sample, resolution=224)
        assert torch.allclose(raw, pixels)
        results["pixels"] = {"shape": tuple(pixels.shape), "in01": True}
    except ImportError:  # pragma: no cover - torch/PIL not installed
        results["pixels"] = "skipped (torch/PIL unavailable)"

    if verbose:
        LOGGER.info("self-test passed: %s", json.dumps(results, default=str)[:400])
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TextVQA/POPE/SQA-I loaders for the Robust CLIP reproduction."
    )
    parser.add_argument("--dataset", default=None, help="TextVQA | POPE | SQA-I")
    parser.add_argument("--split", default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=TOP_K_GROUND_TRUTHS)
    parser.add_argument("--local-path", default=None, help="LLaVA-style annotation JSON")
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--model-name", default=None, help="attach model prompts (llava/openflamingo)")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)

    if args.self_test or args.dataset is None:
        payload = _self_test(verbose=True)
        print(json.dumps(payload, indent=2, default=str))
        return 0

    samples = load_benchmark(
        args.dataset,
        split=args.split,
        num_samples=args.num_samples,
        local_path=args.local_path,
        image_root=args.image_root,
        model_name=args.model_name,
        verbose=True,
    )
    stats = dataset_statistics(samples, k=args.top_k)
    example = samples[0].to_dict() if samples else None
    print(json.dumps({"statistics": stats, "example": example}, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
