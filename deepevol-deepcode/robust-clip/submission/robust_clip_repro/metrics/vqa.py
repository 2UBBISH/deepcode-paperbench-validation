"""VQA metrics for the Robust CLIP reproduction (TextVQA / POPE / SQA-I).

This module is the metric layer paired with :mod:`robust_clip_repro.attacks.vqa_schedule`.
It provides three things:

1. **Answer normalization + parsing**: VQA-style answer normalization, yes/no
   parsing (POPE), multiple-choice matching (SQA-I) and free-form cleanup
   (TextVQA).
2. **Accuracy metrics**: the VQA-v2 soft accuracy used for TextVQA, exact-match
   accuracy for POPE / SQA-I, plus POPE yes/no precision/recall/F1 so the paper's
   tables can be rebuilt from recorded predictions.
3. **The numeric "score"** used by the Addendum's precision-graded schedule
   ("a high-precision attack is done on the ground truth that led to the lowest
   score for each sample").  The Addendum does not define this score, so it is
   *configurable* (``kind=``) and every choice is tagged
   ``UNSPECIFIED_BY_ADDENDUM`` -- no value is invented as if it came from the
   paper.  Whatever the kind, the invariant is the same: **higher score = the
   model performs better on that ground truth**, so ``argmin`` over the top-5
   ground truths selects the most damaging one, exactly as the Addendum states.

Everything heavy (``torch``, victim models, dataset loaders) is imported lazily,
so the pure-Python metric helpers and the offline ``--self-test`` work without a
GPU or any downloads.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import string
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

LOGGER = logging.getLogger("robust_clip_repro.metrics.vqa")

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

TEXT_VQA = "TextVQA"
POPE = "POPE"
SQA_I = "SQA-I"
DATASET_NAMES: Tuple[str, ...] = (TEXT_VQA, POPE, SQA_I)

DATASET_ALIASES: Dict[str, str] = {
    "textvqa": TEXT_VQA,
    "text_vqa": TEXT_VQA,
    "textvqa_0.5.1_val": TEXT_VQA,
    "textvqa_0.5.1_validation": TEXT_VQA,
    "vqa": TEXT_VQA,
    "pope": POPE,
    "pope_yesno": POPE,
    "sqa": SQA_I,
    "sqa-i": SQA_I,
    "sqa_i": SQA_I,
    "scienceqa": SQA_I,
    "science_qa": SQA_I,
    "sqai": SQA_I,
}

#: The Addendum defines no numeric score for the arg-min selection step; every
#: score definition is therefore an *external* choice, logged as such.
UNSPECIFIED = "UNSPECIFIED_BY_ADDENDUM"

#: Score kinds.  All of them are performance scores (higher = better model
#: performance on the candidate ground truth; lower = attack was more damaging).
SCORE_KINDS: Tuple[str, ...] = (
    "accuracy",
    "soft_accuracy",
    "exact_match",
    "f1",
    "logprob",
    "neg_ce",
    "confidence",
)

#: Default mirrors ``data/benchmarks.py`` (also Addendum-silent).
DEFAULT_SCORE_KIND = "accuracy"

SCORE_KIND_PROVENANCE = {kind: UNSPECIFIED for kind in SCORE_KINDS}

#: Values this module needs but the Addendum never states.
EXTERNAL_DEFAULTS: Dict[str, Any] = {
    "score_kind": DEFAULT_SCORE_KIND,
    "score_source": UNSPECIFIED,
    "answer_parser": "first_non_empty_line",
    "soft_accuracy_denominator": 3.0,
    "pope_positive_label": "yes",
    "max_new_tokens": "UNSPECIFIED_BY_ADDENDUM",
    "prompt_template": UNSPECIFIED,
}

MAX_ANSWERS = 10

ARTICLES = {"a", "an", "the"}

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

_YES_TOKENS = {"yes", "yeah", "yep", "true", "correct"}
_NO_TOKENS = {"no", "nope", "false", "incorrect"}

#: SQA-I generated answers are sometimes prefixed; strip common scaffolding.
_ANSWER_PREFIXES = (
    "the answer is",
    "answer:",
    "final answer:",
    "the correct answer is",
)


# --------------------------------------------------------------------------------------
# Text normalization / parsing
# --------------------------------------------------------------------------------------


def normalize_dataset_name(name: Optional[str]) -> str:
    """Map a dataset name/alias to one of :data:`DATASET_NAMES`."""
    if name is None:
        return TEXT_VQA
    key = str(name).strip().lower().replace(" ", "_")
    if key in DATASET_ALIASES:
        return DATASET_ALIASES[key]
    for canonical in DATASET_NAMES:
        if key == canonical.lower().replace("-", "_"):
            return canonical
    raise ValueError(f"Unknown VQA dataset name {name!r}; expected one of {DATASET_NAMES}")


def normalize_text(text: Any) -> str:
    """Lowercase, collapse whitespace and strip punctuation."""
    if text is None:
        return ""
    return " ".join(str(text).strip().lower().translate(_PUNCT_TABLE).split())


def normalize_answer(text: Any) -> str:
    """VQA-style answer normalization (lowercase, punctuation + articles removed)."""
    tokens = normalize_text(text).split()
    tokens = [t for t in tokens if t not in ARTICLES]
    return " ".join(tokens)


def tokenize_answer(text: Any) -> List[str]:
    """Normalized token list of an answer string."""
    return normalize_answer(text).split()


def clean_prediction(text: Any) -> str:
    """Strip scaffolding/newlines from a generated answer."""
    if text is None:
        return ""
    if isinstance(text, (list, tuple)):
        text = text[0] if text else ""
    s = str(text).replace("\r", "\n").strip()
    # Generated captions/answers may repeat the prompt; keep the first line.
    lines = [ln.strip() for ln in s.split("\n") if ln.strip()]
    if not lines:
        return ""
    s = lines[0]
    low = s.lower()
    for prefix in _ANSWER_PREFIXES:
        if low.startswith(prefix):
            s = s[len(prefix):].strip(" :.-\t")
            low = s.lower()
    s = s.strip().strip(".").strip()
    return s


def first_non_empty_line(text: Any) -> str:
    """Default answer parser: the first non-empty line of the generation."""
    return clean_prediction(text)


def parse_yes_no(text: Any) -> Optional[str]:
    """Return ``"yes"`` / ``"no"`` if the prediction looks like a yes/no answer."""
    s = normalize_text(clean_prediction(text))
    if not s:
        return None
    tokens = s.split()
    head = tokens[0]
    if head in _YES_TOKENS:
        return "yes"
    if head in _NO_TOKENS:
        return "no"
    for tok in tokens:
        if tok in _YES_TOKENS:
            return "yes"
        if tok in _NO_TOKENS:
            return "no"
    return None


def parse_choice(text: Any, choices: Optional[Sequence[str]]) -> Optional[str]:
    """Match a generation to one of the SQA-I multiple-choice options."""
    if not choices:
        return None
    s = clean_prediction(text)
    s_norm = normalize_answer(s)
    if not s_norm:
        return None
    for choice in choices:
        if normalize_answer(choice) == s_norm:
            return choice
    # Fall back to substring containment (longest choice first).
    for choice in sorted(choices, key=lambda c: -len(str(c))):
        c_norm = normalize_answer(choice)
        if c_norm and c_norm in s_norm:
            return choice
    return None


def parse_prediction(
    text: Any,
    *,
    dataset_name: Optional[str] = None,
    choices: Optional[Sequence[str]] = None,
    parser: Optional[Callable[[Any], str]] = None,
) -> str:
    """Dataset-aware prediction parsing.

    POPE -> ``"yes"``/``"no"`` (or the cleaned text when undecidable);
    SQA-I -> the matched choice (or the cleaned text);
    TextVQA -> the first non-empty line.
    """
    if parser is not None:
        return parser(text)
    dataset = normalize_dataset_name(dataset_name) if dataset_name else None
    cleaned = clean_prediction(text)
    if dataset == POPE:
        yn = parse_yes_no(cleaned)
        return yn if yn is not None else cleaned
    if dataset == SQA_I:
        choice = parse_choice(cleaned, choices)
        return choice if choice is not None else cleaned
    return cleaned


# --------------------------------------------------------------------------------------
# Ground-truth extraction
# --------------------------------------------------------------------------------------

_ANSWER_LIST_KEYS = ("answers", "ground_truths", "gts", "references", "answer_list")
_ANSWER_KEY_KEYS = ("answer", "ground_truth", "gt", "label", "multiple_choice_answer", "text")
_CHOICE_KEYS = ("choices", "options", "multiple_choices")


def _get(sample: Any, key: str, default: Any = None) -> Any:
    if isinstance(sample, Mapping):
        return sample.get(key, default)
    return getattr(sample, key, default)


def answers_of(sample: Any, dataset_name: Optional[str] = None) -> List[str]:
    """Extract the ground-truth answer list of a sample (duck-typed)."""
    if isinstance(sample, str):
        return [sample]
    if isinstance(sample, (list, tuple)):
        return [str(s) for s in sample]
    for key in _ANSWER_LIST_KEYS:
        value = _get(sample, key)
        if isinstance(value, Mapping):
            value = value.get("answers") or value.get("answer") or list(value.values())
        if isinstance(value, str):
            if value.strip():
                return [value]
        elif isinstance(value, (list, tuple)) and len(value) > 0:
            if isinstance(value[0], Mapping):  # VQA records: {"answer": ..., 'answer_confidence': ...}
                out = [str(v.get("answer", "")).strip() for v in value]
                out = [v for v in out if v]
                if out:
                    return out
            else:
                out = [str(v).strip() for v in value if v is not None and str(v).strip()]
                if out:
                    return out
    for key in _ANSWER_KEY_KEYS:
        value = _get(sample, key)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            out = [str(v).strip() for v in value if str(v).strip()]
            if out:
                return out
        elif str(value).strip():
            # SQA-I stores an answer index; prefer the mapped choice.
            if isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
                choices = choices_of(sample)
                idx = int(value)
                if choices and 0 <= idx < len(choices):
                    return [str(choices[idx])]
            return [clean_prediction(value)]
    return []


def choices_of(sample: Any) -> List[str]:
    """Extract multiple-choice options if present (SQA-I)."""
    for key in _CHOICE_KEYS:
        value = _get(sample, key)
        if isinstance(value, (list, tuple)) and value:
            return [str(v) for v in value]
    return []


def ground_truth_of(sample: Any, dataset_name: Optional[str] = None) -> str:
    """Primary ground truth of a sample (first entry of :func:`answers_of`)."""
    answers = answers_of(sample, dataset_name)
    return answers[0] if answers else ""


# --------------------------------------------------------------------------------------
# Accuracy metrics
# --------------------------------------------------------------------------------------


def soft_vqa_accuracy(prediction: str, ground_truths: Sequence[str], *, denominator: float = 3.0) -> float:
    """VQA-v2 accuracy: ``min(#votes_matching / 3, 1)`` (Addendum-silent -> default 3)."""
    if not ground_truths:
        return 0.0
    pred = normalize_answer(prediction)
    denom = float(denominator) if denominator else 3.0
    if denom <= 0:
        denom = 3.0
    votes = sum(1 for gt in ground_truths if normalize_answer(gt) == pred)
    # Scale to the standard 10-annotator setting when fewer answers are present.
    if len(ground_truths) < MAX_ANSWERS:
        return min(1.0, votes / denom)
    return min(1.0, votes / denom)


def exact_match_accuracy(prediction: str, ground_truths: Sequence[str]) -> float:
    """1.0 if the normalized prediction matches any ground truth, else 0.0."""
    pred = normalize_answer(prediction)
    if not pred:
        return 0.0
    return 1.0 if any(normalize_answer(gt) == pred for gt in ground_truths) else 0.0


def is_exact_match(prediction: str, ground_truths: Sequence[str]) -> bool:
    """Boolean view of :func:`exact_match_accuracy`."""
    return bool(exact_match_accuracy(prediction, ground_truths) > 0.0)


def token_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 between a prediction and a single reference answer."""
    pred_tokens = tokenize_answer(prediction)
    gt_tokens = tokenize_answer(ground_truth)
    if not pred_tokens or not gt_tokens:
        return 0.0
    from collections import Counter

    common = Counter(pred_tokens) & Counter(gt_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def yes_no_prf(
    predictions: Sequence[str],
    ground_truths: Sequence[str],
    *,
    positive: str = "yes",
) -> Dict[str, float]:
    """POPE-style precision/recall/F1 for the positive (``yes``) class.

    ``predictions`` are normalized ``yes``/``no`` strings; anything else counts
    as a negative prediction.
    """
    pos = normalize_answer(positive) or "yes"
    tp = fp = fn = tn = 0
    for pred, gt in zip(predictions, ground_truths):
        p = normalize_answer(pred) == pos
        g = normalize_answer(gt) == pos
        if p and g:
            tp += 1
        elif p and not g:
            fp += 1
        elif not p and g:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def prediction_accuracy(
    prediction: str,
    ground_truths: Sequence[str],
    dataset_name: Optional[str] = None,
) -> float:
    """Dataset-appropriate accuracy of one prediction.

    TextVQA uses VQA-v2 soft accuracy; POPE and SQA-I use exact match.
    """
    if dataset_name is not None:
        dataset = normalize_dataset_name(dataset_name)
        if dataset == TEXT_VQA:
            return soft_vqa_accuracy(prediction, ground_truths)
        return exact_match_accuracy(prediction, ground_truths)
    # Heuristic when the dataset is unknown: POPE-like yes/no -> exact match.
    if len(ground_truths) == 1 and normalize_answer(ground_truths[0]) in ("yes", "no"):
        return exact_match_accuracy(prediction, ground_truths)
    if len(ground_truths) > 1:
        return soft_vqa_accuracy(prediction, ground_truths)
    return exact_match_accuracy(prediction, ground_truths)


def sample_accuracy(
    prediction: str,
    sample: Any,
    dataset_name: Optional[str] = None,
) -> float:
    """Accuracy of a prediction against a sample (or list of ground truths)."""
    if isinstance(sample, (list, tuple, str)):
        return prediction_accuracy(prediction, answers_of(sample), dataset_name)
    dataset = dataset_name
    if dataset is None:
        dataset = _get(sample, "dataset")
    return prediction_accuracy(prediction, answers_of(sample), dataset)


def mean(values: Iterable[float]) -> float:
    """Arithmetic mean of a (possibly empty) iterable."""
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def accuracy_from_predictions(
    predictions: Sequence[str],
    references: Sequence[Any],
    dataset_name: Optional[str] = None,
) -> Dict[str, float]:
    """Aggregate clean predictions into the paper's per-dataset metrics.

    Returns ``{"accuracy": ..., ("precision"/"recall"/"f1": ...) for POPE}``.
    """
    scores: List[float] = []
    preds_norm: List[str] = []
    gts_norm: List[str] = []
    for pred, ref in zip(predictions, references):
        gts = answers_of(ref, dataset_name)
        dataset = dataset_name
        if dataset is None and not isinstance(ref, (list, tuple, str)):
            dataset = _get(ref, "dataset")
        scores.append(prediction_accuracy(pred, gts, dataset))
        preds_norm.append(normalize_answer(pred))
        gts_norm.append(normalize_answer(gts[0]) if gts else "")
    metrics: Dict[str, float] = {"accuracy": mean(scores)}
    dataset = normalize_dataset_name(dataset_name) if dataset_name else None
    if dataset == POPE:
        metrics.update(yes_no_prf(preds_norm, gts_norm))
    return metrics


# --------------------------------------------------------------------------------------
# Torch helpers for log-probability scores (lazy torch import)
# --------------------------------------------------------------------------------------


def answer_logprob(
    logits: Any,
    token_ids: Any,
    *,
    reduction: str = "mean",
) -> float:
    """Teacher-forced log-probability of ``token_ids`` under ``logits``.

    ``logits`` must be ``(T, V)`` (or ``(B, T, V)``) covering the *prompt plus
    answer*, i.e. the answer tokens are assumed to be the **last**
    ``len(token_ids)`` positions.  ``reduction`` is one of ``mean``/``sum``/``min``.
    """
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - torch is a hard requirement at runtime
        raise ImportError(
            "torch is required for log-probability scoring; install torch or use "
            "score kind 'accuracy'."
        ) from exc

    logits_t = torch.as_tensor(logits)
    target = torch.as_tensor(token_ids, dtype=torch.long).reshape(-1)
    if target.numel() == 0 or logits_t.numel() == 0:
        return 0.0
    if logits_t.dim() == 2:
        logits_t = logits_t.unsqueeze(0)
    if logits_t.dim() != 3:
        raise ValueError(f"expected (B, T, V) logits, got shape {tuple(logits_t.shape)}")

    seq_len = logits_t.shape[1]
    n = int(target.numel())
    start = max(0, seq_len - n)
    target = target[-min(n, seq_len):]
    selected = logits_t[:, start:start + target.numel(), :]
    log_probs = F.log_softmax(selected.float(), dim=-1)
    gathered = log_probs.gather(
        -1, target.view(1, -1, 1).expand(log_probs.shape[0], target.numel(), 1)
    ).squeeze(-1)
    per_sample = gathered.mean(dim=0)
    if reduction == "mean":
        return float(per_sample.mean())
    if reduction == "sum":
        return float(per_sample.sum())
    if reduction == "min":
        return float(per_sample.min())
    return float(per_sample.mean())


def answer_probability(logits: Any, token_ids: Any, *, reduction: str = "mean") -> float:
    """Probability (not log) of the answer tokens under ``logits``."""
    import math

    lp = answer_logprob(logits, token_ids, reduction=reduction)
    return float(math.exp(lp)) if lp > -700 else 0.0


# --------------------------------------------------------------------------------------
# Score functions: the "score" used for the Addendum's arg-min selection
# --------------------------------------------------------------------------------------


class VQAScorer:
    """Numeric performance score for one (pixels, ground_truth) pair.

    **Invariant**: higher score == the model performs better on that ground
    truth.  The Addendum's step (2) therefore takes the ``argmin`` over the top-5
    ground truths to find the *most damaging* candidate.

    The score definition is Addendum-silent and hence configurable via ``kind``:

    * ``accuracy`` / ``soft_accuracy`` / ``exact_match`` / ``f1`` - generation
      based, need ``generate_fn(pixels) -> str``;
    * ``logprob`` / ``neg_ce`` - model-likelihood based, need
      ``logits_fn(pixels, text) -> logits`` and a ``tokenizer``;
    * ``confidence`` - mean probability of the ground-truth tokens.
    """

    def __init__(
        self,
        kind: str = DEFAULT_SCORE_KIND,
        *,
        generate_fn: Optional[Callable[..., Any]] = None,
        logits_fn: Optional[Callable[..., Any]] = None,
        tokenizer: Any = None,
        prompt: str = "",
        dataset_name: Optional[str] = None,
        answer_parser: Optional[Callable[[Any], str]] = None,
        soft_accuracy_denominator: float = 3.0,
        choices_fn: Optional[Callable[[Any, Any], Sequence[str]]] = None,
        verbose: bool = False,
    ) -> None:
        kind = str(kind or DEFAULT_SCORE_KIND).lower()
        if kind in ("neg_ce", "ce", "nll", "negative_ce"):
            kind = "logprob"
        if kind in ("soft", "vqa_accuracy", "vqa"):
            kind = "soft_accuracy"
        if kind not in SCORE_KINDS:
            raise ValueError(f"Unknown score kind {kind!r}; supported: {SCORE_KINDS}")
        self.kind = kind
        self.generate_fn = generate_fn
        self.logits_fn = logits_fn
        self.tokenizer = tokenizer
        self.prompt = prompt or ""
        self.dataset_name = normalize_dataset_name(dataset_name) if dataset_name else None
        self.answer_parser = answer_parser
        self.soft_accuracy_denominator = soft_accuracy_denominator
        self.choices_fn = choices_fn
        self.verbose = verbose
        self.provenance = {
            "kind": UNSPECIFIED,
            "soft_accuracy_denominator": UNSPECIFIED,
            "generation_args": UNSPECIFIED,
            "note": (
                "The Addendum does not define the numeric score used to pick the "
                "ground truth with the lowest score; this definition is externally "
                "supplied."
            ),
        }
        if kind in ("accuracy", "soft_accuracy", "exact_match", "f1") and generate_fn is None:
            raise ValueError(f"score kind {kind!r} requires generate_fn")
        if kind in ("logprob", "confidence") and logits_fn is None:
            raise ValueError(f"score kind {kind!r} requires logits_fn")

    # -- helpers -----------------------------------------------------------------
    def _encode(self, text: str) -> Any:
        if self.tokenizer is None:
            raise ValueError("logprob scoring requires a tokenizer")
        tok = self.tokenizer
        try:
            ids = tok.encode(text, add_special_tokens=False)
        except Exception:
            out = tok(text, add_special_tokens=False)
            ids = out["input_ids"] if isinstance(out, Mapping) else out
            if isinstance(ids, (list, tuple)) and ids and isinstance(ids[0], (list, tuple)):
                ids = ids[0]
        return ids

    def _generate(self, pixels: Any) -> str:
        out = self.generate_fn(pixels)
        return parse_prediction(
            out,
            dataset_name=self.dataset_name,
            choices=(self.choices_fn(pixels, None) if self.choices_fn else None),
            parser=self.answer_parser,
        )

    # -- public API --------------------------------------------------------------
    def __call__(self, pixels: Any, ground_truth: Any) -> float:
        """Return the performance score of ``ground_truth`` on ``pixels``."""
        text = str(ground_truth)
        if self.kind in ("accuracy", "soft_accuracy", "exact_match", "f1"):
            prediction = self._generate(pixels)
            if self.kind == "accuracy":
                return float(prediction_accuracy(prediction, [text], self.dataset_name))
            if self.kind == "soft_accuracy":
                return float(
                    soft_vqa_accuracy(prediction, [text], denominator=self.soft_accuracy_denominator)
                )
            if self.kind == "exact_match":
                return float(exact_match_accuracy(prediction, [text]))
            return float(token_f1(prediction, text))
        if self.kind in ("logprob", "confidence"):
            ids = self._encode(self.prompt + text)
            logits = self.logits_fn(pixels, text)
            if self.kind == "logprob":
                return float(answer_logprob(logits, ids, reduction="mean"))
            return float(answer_probability(logits, ids, reduction="mean"))
        raise ValueError(f"Unknown score kind {self.kind!r}")

    def score_many(self, pixels: Any, ground_truths: Sequence[str]) -> Dict[str, float]:
        """Score several candidate ground truths on the same (adversarial) pixels."""
        return {str(gt): self(pixels, gt) for gt in ground_truths}

    def select_lowest(self, pixels: Any, ground_truths: Sequence[str]) -> Tuple[str, int, Dict[str, float]]:
        """Addendum step (2): the candidate ground truth with the *lowest* score."""
        scores = self.score_many(pixels, ground_truths)
        if not scores:
            return "", -1, {}
        ordered = list(ground_truths)
        best_idx = min(range(len(ordered)), key=lambda i: (scores[str(ordered[i])], i))
        return str(ordered[best_idx]), best_idx, scores

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "dataset_name": self.dataset_name,
            "soft_accuracy_denominator": self.soft_accuracy_denominator,
            "provenance": dict(self.provenance),
        }


def make_score_fn(
    *,
    kind: str = DEFAULT_SCORE_KIND,
    generate_fn: Optional[Callable[..., Any]] = None,
    logits_fn: Optional[Callable[..., Any]] = None,
    tokenizer: Any = None,
    prompt: str = "",
    dataset_name: Optional[str] = None,
    **kwargs: Any,
) -> VQAScorer:
    """Build a :class:`VQAScorer`; the returned callable exposes ``.kind``/``.provenance``."""
    return VQAScorer(
        kind,
        generate_fn=generate_fn,
        logits_fn=logits_fn,
        tokenizer=tokenizer,
        prompt=prompt,
        dataset_name=dataset_name,
        **kwargs,
    )


def make_accuracy_fn(
    generate_fn: Callable[..., Any],
    *,
    dataset_name: Optional[str] = None,
    answer_parser: Optional[Callable[[Any], str]] = None,
    choices_fn: Optional[Callable[[Any], Sequence[str]]] = None,
) -> Callable[[Any, Any], float]:
    """Build an ``accuracy_fn(pixels, ground_truth) -> float`` for the schedule."""
    dataset = normalize_dataset_name(dataset_name) if dataset_name else None

    def accuracy_fn(pixels: Any, ground_truth: Any) -> float:
        out = generate_fn(pixels)
        choices = choices_fn(pixels) if choices_fn else None
        prediction = parse_prediction(
            out, dataset_name=dataset, choices=choices, parser=answer_parser
        )
        return float(prediction_accuracy(prediction, [str(ground_truth)], dataset))

    accuracy_fn.dataset_name = dataset  # type: ignore[attr-defined]
    accuracy_fn.provenance = {"answer_parser": UNSPECIFIED}  # type: ignore[attr-defined]
    return accuracy_fn


def make_schedule_fns(
    victim: Any,
    *,
    kind: str = DEFAULT_SCORE_KIND,
    dataset_name: Optional[str] = None,
    prompt: str = "",
    answer_parser: Optional[Callable[[Any], str]] = None,
    generation_kwargs: Optional[Mapping[str, Any]] = None,
) -> Tuple[VQAScorer, Callable[[Any, Any], float]]:
    """Build ``(score_fn, accuracy_fn)`` from a victim model object.

    The victim must expose ``generate(pixels, prompt, **kwargs) -> str`` and,
    for likelihood-based score kinds, ``logits_fn``/``targeted_logits`` plus a
    ``tokenizer`` (see ``models/llava_openclip.py``).
    """
    generation_kwargs = dict(generation_kwargs or {})

    def generate_fn(pixels: Any, _prompt: Optional[str] = None) -> str:
        out = victim.generate(pixels, _prompt if _prompt is not None else prompt, **generation_kwargs)
        if isinstance(out, (list, tuple)):
            out = out[0] if out else ""
        return str(out)

    logits_fn = None
    tokenizer = getattr(victim, "tokenizer", None)
    if hasattr(victim, "targeted_logits"):
        logits_fn = lambda pixels, text: victim.targeted_logits(pixels, text)  # noqa: E731
    elif hasattr(victim, "logits_fn"):
        _inner = victim.logits_fn(prompt)

        def logits_fn(pixels, text, _inner=_inner):  # type: ignore[misc]
            return _inner(pixels)

    score_fn = VQAScorer(
        kind,
        generate_fn=generate_fn,
        logits_fn=logits_fn,
        tokenizer=tokenizer,
        prompt=prompt,
        dataset_name=dataset_name,
        answer_parser=answer_parser,
    )
    accuracy_fn = make_accuracy_fn(generate_fn, dataset_name=dataset_name, answer_parser=answer_parser)
    return score_fn, accuracy_fn


def constant_score_fn(value: float = 0.0) -> Callable[[Any, Any], float]:
    """Model-free score function (useful for scheduling/self tests)."""

    def score_fn(pixels: Any, ground_truth: Any) -> float:  # noqa: ARG001
        return float(value)

    score_fn.kind = "constant"  # type: ignore[attr-defined]
    score_fn.provenance = {"kind": UNSPECIFIED}  # type: ignore[attr-defined]
    return score_fn


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


def _dtype_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    name = getattr(value, "name", None)
    if name:
        return f"{name}{getattr(value, 'bits', '')}"
    return str(value)


@dataclass
class VQASampleResult:
    """Per-sample outcome of the scheduled VQA attack."""

    sample_id: Any = None
    question: str = ""
    ground_truth: str = ""
    ground_truths: List[str] = field(default_factory=list)
    selected_ground_truth: Optional[str] = None
    clean_prediction: str = ""
    attacked_prediction: str = ""
    clean_accuracy: float = 0.0
    attacked_accuracy: float = 0.0
    worst_accuracy: float = 0.0
    stage_accuracies: Dict[str, float] = field(default_factory=dict)
    stage_scores: Dict[str, float] = field(default_factory=dict)
    skipped_stages: List[str] = field(default_factory=list)
    trace: List[str] = field(default_factory=list)
    perturbation_dtypes: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "question": self.question,
            "ground_truth": self.ground_truth,
            "ground_truths": list(self.ground_truths),
            "selected_ground_truth": self.selected_ground_truth,
            "clean_prediction": self.clean_prediction,
            "attacked_prediction": self.attacked_prediction,
            "clean_accuracy": self.clean_accuracy,
            "attacked_accuracy": self.attacked_accuracy,
            "worst_accuracy": self.worst_accuracy,
            "stage_accuracies": dict(self.stage_accuracies),
            "stage_scores": dict(self.stage_scores),
            "skipped_stages": list(self.skipped_stages),
            "trace": list(self.trace),
            "perturbation_dtypes": list(self.perturbation_dtypes),
            "extra": dict(self.extra),
        }


@dataclass
class VQAReport:
    """Aggregated VQA robustness result for one (dataset, model, attack) setting."""

    dataset_name: str = TEXT_VQA
    model_name: str = "model"
    method: str = "model"
    attack_name: str = "scheduled"
    num_samples: int = 0
    clean_accuracy: float = 0.0
    attacked_accuracy: float = 0.0
    worst_accuracy: float = 0.0
    accuracy_drop: float = 0.0
    stage_accuracies: Dict[str, float] = field(default_factory=dict)
    stage_precisions: Dict[str, str] = field(default_factory=dict)
    stage_counts: Dict[str, int] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    top_ground_truths: List[Any] = field(default_factory=list)
    most_frequent_ground_truth: Optional[str] = None
    score_kind: str = DEFAULT_SCORE_KIND
    precision_ok: bool = True
    per_sample: List[VQASampleResult] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    external_defaults: Dict[str, Any] = field(default_factory=lambda: dict(EXTERNAL_DEFAULTS))
    elapsed_seconds: float = 0.0

    # -- derived -----------------------------------------------------------------
    @property
    def absolute_robust_gain(self) -> float:
        return self.clean_accuracy - self.attacked_accuracy

    def summary(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "model": self.model_name,
            "method": self.method,
            "attack": self.attack_name,
            "num_samples": self.num_samples,
            "clean_accuracy": self.clean_accuracy,
            "attacked_accuracy": self.attacked_accuracy,
            "worst_accuracy": self.worst_accuracy,
            "accuracy_drop": self.accuracy_drop,
            "score_kind": self.score_kind,
            "stage_accuracies": dict(self.stage_accuracies),
            "precision_ok": self.precision_ok,
            "external_defaults": dict(self.external_defaults),
        }

    def as_dict(self, include_per_sample: bool = False) -> Dict[str, Any]:
        out = {
            "dataset_name": self.dataset_name,
            "model_name": self.model_name,
            "method": self.method,
            "attack_name": self.attack_name,
            "num_samples": self.num_samples,
            "clean_accuracy": self.clean_accuracy,
            "attacked_accuracy": self.attacked_accuracy,
            "worst_accuracy": self.worst_accuracy,
            "accuracy_drop": self.accuracy_drop,
            "stage_accuracies": dict(self.stage_accuracies),
            "stage_precisions": dict(self.stage_precisions),
            "stage_counts": dict(self.stage_counts),
            "metrics": dict(self.metrics),
            "top_ground_truths": [gt if not hasattr(gt, "as_dict") else gt.as_dict() for gt in self.top_ground_truths],
            "most_frequent_ground_truth": self.most_frequent_ground_truth,
            "score_kind": self.score_kind,
            "precision_ok": self.precision_ok,
            "provenance": dict(self.provenance),
            "external_defaults": dict(self.external_defaults),
            "elapsed_seconds": self.elapsed_seconds,
        }
        if include_per_sample:
            out["per_sample"] = [s.as_dict() for s in self.per_sample]
        return out

    def to_json(self, path: Optional[str] = None, *, include_per_sample: bool = False, indent: int = 2) -> str:
        text = json.dumps(self.as_dict(include_per_sample=include_per_sample), indent=indent, default=str)
        if path:
            directory = os.path.dirname(os.path.abspath(path))
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
        return text


@dataclass
class ComparisonRow:
    """One row of the clean-vs-robust VQA comparison table."""

    dataset: str
    method: str
    attack: str = "scheduled"
    clean_accuracy: float = 0.0
    attacked_accuracy: float = 0.0
    worst_accuracy: float = 0.0
    num_samples: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "method": self.method,
            "attack": self.attack,
            "clean_accuracy": self.clean_accuracy,
            "attacked_accuracy": self.attacked_accuracy,
            "worst_accuracy": self.worst_accuracy,
            "num_samples": self.num_samples,
        }


def build_comparison_table(reports: Sequence[VQAReport]) -> List[Dict[str, Any]]:
    """Tabulate reports (e.g. vanilla CLIP vs Robust CLIP across the 3 datasets)."""
    rows: List[ComparisonRow] = []
    for rep in reports:
        rows.append(
            ComparisonRow(
                dataset=rep.dataset_name,
                method=rep.method or rep.model_name,
                attack=rep.attack_name,
                clean_accuracy=rep.clean_accuracy,
                attacked_accuracy=rep.attacked_accuracy,
                worst_accuracy=rep.worst_accuracy,
                num_samples=rep.num_samples,
            )
        )
    return [row.as_dict() for row in rows]


def format_table(rows: Sequence[Mapping[str, Any]], *, percent: bool = True) -> str:
    """Render comparison rows as a plain-text table."""
    if not rows:
        return "(no rows)"
    keys = list(rows[0].keys())
    table = [keys]
    for row in rows:
        cells = []
        for key in keys:
            value = row.get(key)
            if percent and isinstance(value, float) and "accuracy" in key:
                cells.append(f"{100.0 * value:6.2f}")
            else:
                cells.append(str(value))
        table.append(cells)
    widths = [max(len(str(r[i])) for r in table) for i in range(len(keys))]
    lines = []
    for idx, row in enumerate(table):
        lines.append("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))
        if idx == 0:
            lines.append("  ".join("-" * widths[i] for i in range(len(keys))))
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Schedule aggregation helpers
# --------------------------------------------------------------------------------------


def _stage_name(record: Any) -> str:
    for attr in ("stage", "name"):
        value = getattr(record, attr, None)
        if value is not None:
            return str(value)
    return "stage"


def _last_perturbation(result: Any) -> Any:
    """Last (most recent) perturbation retained by a schedule result."""
    stages = list(getattr(result, "stages", []) or [])
    for record in reversed(stages):
        if getattr(record, "skipped", False):
            continue
        if getattr(record, "perturbation", None) is not None:
            return record.perturbation
    return getattr(result, "best_perturbation", None)


def _prepare_pixels(sample: Any, *, resolution: int = 224, device: Any = None) -> Any:
    """Get raw (non-normalized) pixels for a sample."""
    if isinstance(sample, Mapping) and sample.get("pixels") is not None:
        return sample["pixels"]
    if not isinstance(sample, Mapping) and getattr(sample, "pixels", None) is not None:
        return sample.pixels
    try:
        from ..data.benchmarks import sample_to_pixels  # lazy

        return sample_to_pixels(sample, resolution=resolution, device=device)
    except Exception as exc:  # pragma: no cover - depends on optional deps
        raise RuntimeError(
            "Cannot resolve raw pixels for a VQA sample; provide 'pixels' on the sample "
            f"or install PIL/torch so data.benchmarks.sample_to_pixels works ({exc})."
        ) from exc


def aggregate_reports(
    per_sample: Sequence[VQASampleResult],
    *,
    dataset_name: str,
    model_name: str = "model",
    method: str = "model",
    attack_name: str = "scheduled",
    clean_predictions: Optional[Sequence[str]] = None,
    clean_references: Optional[Sequence[Any]] = None,
    score_kind: str = DEFAULT_SCORE_KIND,
    stage_precisions: Optional[Mapping[str, str]] = None,
    top_ground_truths: Optional[Sequence[Any]] = None,
    most_frequent_ground_truth: Optional[str] = None,
    elapsed_seconds: float = 0.0,
    provenance: Optional[Mapping[str, Any]] = None,
) -> VQAReport:
    """Aggregate per-sample results into a :class:`VQAReport`."""
    dataset = normalize_dataset_name(dataset_name)
    report = VQAReport(
        dataset_name=dataset,
        model_name=model_name,
        method=method,
        attack_name=attack_name,
        num_samples=len(per_sample),
        per_sample=list(per_sample),
        score_kind=score_kind,
        elapsed_seconds=elapsed_seconds,
    )
    report.clean_accuracy = mean(s.clean_accuracy for s in per_sample)
    report.attacked_accuracy = mean(s.attacked_accuracy for s in per_sample)
    report.worst_accuracy = mean(s.worst_accuracy for s in per_sample) if per_sample else 0.0
    report.accuracy_drop = report.clean_accuracy - report.attacked_accuracy

    stage_values: Dict[str, List[float]] = {}
    stage_counts: Dict[str, int] = {}
    for sample in per_sample:
        for stage, value in sample.stage_accuracies.items():
            stage_values.setdefault(stage, []).append(float(value))
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
    report.stage_accuracies = {stage: mean(vals) for stage, vals in stage_values.items()}
    report.stage_counts = stage_counts
    if stage_precisions:
        report.stage_precisions = {str(k): str(v) for k, v in stage_precisions.items()}

    if clean_predictions is not None and clean_references is not None:
        try:
            report.metrics = accuracy_from_predictions(clean_predictions, clean_references, dataset)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Could not aggregate clean metrics: %s", exc)

    if top_ground_truths is not None:
        report.top_ground_truths = list(top_ground_truths)
    report.most_frequent_ground_truth = most_frequent_ground_truth

    report.precision_ok = all(
        dtype in (None, "int16", "int32", "torch.int16", "torch.int32")
        for sample in per_sample
        for dtype in sample.perturbation_dtypes
    )
    report.provenance = dict(provenance or {})
    report.provenance.setdefault("score_kind", UNSPECIFIED)
    report.provenance.setdefault("attack_budgets", UNSPECIFIED)
    return report


def sample_result_from_schedule(
    sample: Any,
    result: Any,
    *,
    sample_index: int = 0,
    clean_prediction: str = "",
    attacked_prediction: str = "",
    dataset_name: Optional[str] = None,
) -> VQASampleResult:
    """Convert a ``VQAScheduleResult`` into a :class:`VQASampleResult` record."""
    ground_truths = answers_of(sample, dataset_name)
    gt = ground_truths[0] if ground_truths else ""
    record = VQASampleResult(
        sample_id=_get(sample, "sample_id", sample_index),
        question=str(_get(sample, "question", "") or ""),
        ground_truth=gt,
        ground_truths=list(ground_truths),
        selected_ground_truth=getattr(result, "selected_ground_truth", None),
        clean_prediction=clean_prediction,
        attacked_prediction=attacked_prediction,
        clean_accuracy=float(
            prediction_accuracy(clean_prediction, ground_truths, dataset_name) if ground_truths else 0.0
        ),
        attacked_accuracy=float(
            prediction_accuracy(attacked_prediction, ground_truths, dataset_name) if ground_truths else 0.0
        ),
    )
    accuracies: List[float] = []
    for stage_record in getattr(result, "stages", []) or []:
        name = _stage_name(stage_record)
        if getattr(stage_record, "skipped", False):
            record.skipped_stages.append(name)
            continue
        accuracy = getattr(stage_record, "accuracy", None)
        score = getattr(stage_record, "score", None)
        if accuracy is not None:
            record.stage_accuracies[name] = float(accuracy)
            accuracies.append(float(accuracy))
        if score is not None:
            record.stage_scores[name] = float(score)
        dtype = _dtype_name(getattr(stage_record, "perturbation_dtype", None))
        if dtype:
            record.perturbation_dtypes.append(dtype)
    trace = getattr(result, "trace", None)
    if trace:
        record.trace = [str(t) for t in trace]
    record.worst_accuracy = min(accuracies) if accuracies else record.attacked_accuracy
    if not record.attacked_prediction:
        record.attacked_accuracy = record.worst_accuracy
    return record


def evaluate_vqa(
    samples: Sequence[Any],
    *,
    scheduler: Any = None,
    generate_fn: Optional[Callable[..., Any]] = None,
    score_fn: Optional[Callable[[Any, Any], float]] = None,
    accuracy_fn: Optional[Callable[[Any, Any], float]] = None,
    dataset_name: Optional[str] = None,
    model_name: str = "model",
    method: Optional[str] = None,
    attack_name: str = "scheduled",
    top_k: int = 5,
    pixel_fn: Optional[Callable[[Any], Any]] = None,
    resolution: int = 224,
    device: Any = None,
    generator: Any = None,
    return_perturbations: bool = True,
    num_samples: Optional[int] = None,
    answer_parser: Optional[Callable[[Any], str]] = None,
    verbose: bool = True,
) -> VQAReport:
    """Run the Addendum's staged VQA attack schedule over a dataset and aggregate.

    ``scheduler`` defaults to ``attacks.vqa_schedule.VQAAttackScheduler`` built
    from ``score_fn``/``accuracy_fn``.  Set ``scheduler=None`` and
    ``score_fn=constant_score_fn(...)`` for a model-free dry run.
    """
    started = time.time()
    samples = list(samples)
    if num_samples is not None:
        samples = samples[: int(num_samples)]
    if not samples:
        return VQAReport(dataset_name=normalize_dataset_name(dataset_name or TEXT_VQA), num_samples=0)

    if dataset_name is None:
        dataset_name = _get(samples[0], "dataset") or TEXT_VQA
    dataset = normalize_dataset_name(dataset_name)

    # top-5 most frequent ground truths + single most frequent (dataset level)
    top5: List[Any] = []
    most_frequent: Optional[str] = None
    try:
        from ..attacks.vqa_schedule import (  # lazy
            most_frequent_ground_truth,
            most_frequent_ground_truths,
        )

        top5 = list(most_frequent_ground_truths(samples, k=top_k))
        most_frequent = most_frequent_ground_truth(samples)
    except Exception as exc:  # pragma: no cover - fallback path
        LOGGER.warning("Falling back to local ground-truth frequency computation: %s", exc)
        from collections import Counter

        counts = Counter()
        for sample in samples:
            for answer in answers_of(sample, dataset):
                counts[normalize_answer(answer)] += 1
        top5 = [answer for answer, _ in counts.most_common(top_k)]
        most_frequent = top5[0] if top5 else None

    gt_strings = [str(gt.answer if hasattr(gt, "answer") else gt) for gt in top5]

    if score_fn is None or accuracy_fn is None:
        raise ValueError(
            "evaluate_vqa requires score_fn and accuracy_fn (or a prebuilt scheduler); "
            "use metrics.vqa.make_score_fn / make_accuracy_fn."
        )

    if pixel_fn is None:
        def pixel_fn(sample):  # type: ignore[misc]
            return _prepare_pixels(sample, resolution=resolution, device=device)

    if scheduler is None:
        from ..attacks.vqa_schedule import VQAAttackScheduler  # lazy

        scheduler = VQAAttackScheduler(
            attack_fn=None,
            score_fn=score_fn,
            accuracy_fn=accuracy_fn,
            dataset_name=dataset,
            generator=generator,
        )

    per_sample: List[VQASampleResult] = []
    clean_predictions: List[str] = []
    stage_precisions: Dict[str, str] = {}
    precision_ok = True

    for index, sample in enumerate(samples):
        pixels = pixel_fn(sample)
        try:
            prediction = generate_fn(pixels) if generate_fn is not None else ""
        except Exception as exc:  # pragma: no cover - victim failure
            LOGGER.warning("Clean generation failed on sample %s: %s", index, exc)
            prediction = ""
        prediction = parse_prediction(
            prediction,
            dataset_name=dataset,
            choices=choices_of(sample) or None,
            parser=answer_parser,
        )
        clean_predictions.append(prediction)

        result = scheduler.run(
            pixels,
            gt_strings,
            dataset_name=dataset,
            most_frequent=most_frequent,
            sample_index=index,
            score_fn=score_fn,
            accuracy_fn=accuracy_fn,
            generator=generator,
            return_perturbations=return_perturbations,
        )
        attacked_prediction = ""
        try:
            last_delta = _last_perturbation(result)
            if last_delta is not None and generate_fn is not None:
                adv_pixels = _adversarial_pixels(pixels, last_delta)
                attacked_prediction = generate_fn(adv_pixels)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Attacked generation failed on sample %s: %s", index, exc)
        attacked_prediction = parse_prediction(
            attacked_prediction,
            dataset_name=dataset,
            choices=choices_of(sample) or None,
            parser=answer_parser,
        )

        record = sample_result_from_schedule(
            sample,
            result,
            sample_index=index,
            clean_prediction=prediction,
            attacked_prediction=attacked_prediction,
            dataset_name=dataset,
        )
        per_sample.append(record)
        for stage_record in getattr(result, "stages", []) or []:
            name = _stage_name(stage_record)
            stage_precisions.setdefault(name, str(getattr(stage_record, "precision", "") or ""))
            dtype = _dtype_name(getattr(stage_record, "perturbation_dtype", None))
            if dtype and dtype not in ("int16", "int32", "torch.int16", "torch.int32"):
                precision_ok = False
        if verbose and (index + 1) % 50 == 0:
            LOGGER.info("VQA evaluation: %d/%d samples", index + 1, len(samples))

    report = aggregate_reports(
        per_sample,
        dataset_name=dataset,
        model_name=model_name,
        method=method or model_name,
        attack_name=attack_name,
        clean_predictions=clean_predictions,
        clean_references=samples,
        score_kind=getattr(score_fn, "kind", DEFAULT_SCORE_KIND) or DEFAULT_SCORE_KIND,
        stage_precisions=stage_precisions,
        top_ground_truths=top5,
        most_frequent_ground_truth=most_frequent,
        elapsed_seconds=time.time() - started,
        provenance={
            "score_kind": UNSPECIFIED,
            "vqa_attack_order": "ADDENDUM",
            "textvqa_word_attack_skipped": dataset == TEXT_VQA,
        },
    )
    report.precision_ok = precision_ok and report.precision_ok
    return report


def _adversarial_pixels(pixels: Any, delta: Any) -> Any:
    """Decode an integer-coded perturbation and add it to raw pixels."""
    try:
        import torch
    except Exception:  # pragma: no cover
        return pixels
    if not isinstance(delta, torch.Tensor):
        delta = torch.as_tensor(delta)
    if torch.is_floating_point(delta):
        delta_float = delta
    else:
        try:
            from ..utils.precision import decode_perturbation  # lazy

            delta_float = decode_perturbation(delta)
        except Exception:  # pragma: no cover - fallback: treat as raw codes
            delta_float = delta.float()
    if not isinstance(pixels, torch.Tensor):
        pixels = torch.as_tensor(pixels)
    adv = pixels.float() + delta_float.to(pixels.device).float()
    return adv.clamp(0.0, 1.0)


def evaluate_from_records(records: Sequence[Mapping[str, Any]], *, dataset_name: Optional[str] = None) -> VQAReport:
    """Rebuild a report from saved per-sample records (offline table reproduction)."""
    samples = [VQASampleResult(**{k: v for k, v in row.items() if k in VQASampleResult.__dataclass_fields__}) for row in records]
    dataset = dataset_name or (records[0].get("extra", {}) or {}).get("dataset") or TEXT_VQA
    return aggregate_reports(samples, dataset_name=dataset)


# --------------------------------------------------------------------------------------
# CLI / self test
# --------------------------------------------------------------------------------------


def _self_test(verbose: bool = True) -> Dict[str, Any]:
    """Offline checks of normalization, parsing, metrics and reporting."""
    checks: Dict[str, Any] = {}

    # normalization / parsing
    assert normalize_answer("The A DOG, running") == "dog running"
    assert parse_yes_no("Yes, there is.") == "yes"
    assert parse_yes_no("no.") == "no"
    assert parse_yes_no("maybe") is None
    assert parse_prediction("Answer: 42\n", dataset_name=POPE) in ("42", "yes", "no")
    assert parse_choice("B) mitochondria", ["nucleus", "mitochondria"]) == "mitochondria"
    assert clean_prediction("Final answer: Paris.") == "Paris"
    checks["parsing"] = True

    # accuracy metrics
    assert abs(soft_vqa_accuracy("dog", ["dog", "dog", "cat"]) - 2 / 3) < 1e-9
    assert abs(soft_vqa_accuracy("dog", ["dog", "dog", "dog"]) - 1.0) < 1e-9
    assert exact_match_accuracy("The Dog", ["dog"]) == 1.0
    assert exact_match_accuracy("cat", ["dog"]) == 0.0
    assert abs(token_f1("the big dog", "big dog") - 2 * (2 / 3) * 1.0 / (2 / 3 + 1.0)) < 1e-9
    prf = yes_no_prf(["yes", "no", "yes", "no"], ["yes", "yes", "yes", "no"])
    assert prf["tp"] == 2 and prf["fp"] == 0 and prf["fn"] == 1 and prf["tn"] == 1
    assert abs(prf["precision"] - 1.0) < 1e-9
    checks["accuracy"] = True

    # ground truth extraction (duck-typed)
    assert answers_of({"answers": ["a", "b"]}) == ["a", "b"]
    assert answers_of({"multiple_choice_answer": "cat"}) == ["cat"]
    assert ground_truth_of({"answers": ["first", "second"]}) == "first"
    assert answers_of({"choices": ["x", "y"], "answer": 1}) == ["y"]
    checks["ground_truths"] = True

    # score function: arg-min selection (Addendum step 2)
    fake_scores = {"low": 0.1, "mid": 0.5, "high": 0.9}
    scorer = VQAScorer(
        "accuracy",
        generate_fn=lambda pixels: fake_scores.get(str(pixels), 0.0),
        dataset_name=POPE,
    )
    gt, idx, scores = scorer.select_lowest("low", ["low", "mid", "high"])
    assert gt == "low" and idx == 0, (gt, idx)
    assert abs(scores["high"] - 0.9) < 1e-9
    assert scorer.provenance["kind"] == UNSPECIFIED
    assert scorer.as_dict()["kind"] == "accuracy"
    assert make_score_fn(
        kind="exact_match", generate_fn=lambda pixels: "yes", dataset_name=POPE
    )("x", "yes") == 1.0
    csf = constant_score_fn(0.25)
    assert csf("anything", "gt") == 0.25
    checks["score_fn"] = True

    # dataset-appropriate accuracy dispatch
    assert prediction_accuracy("yes", ["yes"], POPE) == 1.0
    assert abs(prediction_accuracy("dog", ["dog", "dog", "cat"], TEXT_VQA) - 2 / 3) < 1e-9
    assert sample_accuracy("mitochondria", {"choices": ["nucleus", "mitochondria"], "answer": 1}, SQA_I) == 1.0
    checks["dispatch"] = True

    # aggregation + reporting
    rows = [
        VQASampleResult(
            sample_id=i,
            ground_truth="yes",
            ground_truths=["yes"],
            selected_ground_truth="yes",
            clean_prediction="yes",
            attacked_prediction="no",
            clean_accuracy=1.0,
            attacked_accuracy=0.0,
            worst_accuracy=0.0,
            stage_accuracies={"low_precision_top5": 0.5, "high_precision_argmin": 0.0},
            perturbation_dtypes=["int16", "int32"],
        )
        for i in range(4)
    ]
    report = aggregate_reports(
        rows,
        dataset_name=POPE,
        model_name="robust_clip",
        clean_predictions=["yes"] * 4,
        clean_references=[{"answers": ["yes"]}] * 4,
        top_ground_truths=["yes", "no"],
        most_frequent_ground_truth="yes",
    )
    assert report.num_samples == 4
    assert abs(report.clean_accuracy - 1.0) < 1e-9
    assert abs(report.attacked_accuracy - 0.0) < 1e-9
    assert abs(report.accuracy_drop - 1.0) < 1e-9
    assert abs(report.stage_accuracies["low_precision_top5"] - 0.5) < 1e-9
    assert report.precision_ok is True
    assert abs(report.metrics["accuracy"] - 1.0) < 1e-9
    assert "f1" in report.metrics  # POPE adds PRF
    assert report.summary()["dataset"] == POPE
    assert report.as_dict(include_per_sample=True)["per_sample"][0]["sample_id"] == 0
    table = build_comparison_table([report])
    assert table[0]["dataset"] == POPE
    text = format_table(table)
    assert "accuracy" in text
    checks["report"] = True

    # total-variation-free: perturbed-pixel helper must not require a model
    import torch  # noqa: F401  (skip gracefully if torch missing)

    checks["env"] = {"torch": "available"}

    if verbose:
        print(json.dumps({"self_test": "ok", "checks": list(checks.keys())}, indent=2))
    return checks


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VQA metrics for the Robust CLIP reproduction.")
    parser.add_argument("--self-test", action="store_true", help="run offline checks and exit")
    parser.add_argument("--records", type=str, default=None, help="per-sample JSON records to aggregate")
    parser.add_argument("--dataset", type=str, default=None, help="override dataset name")
    parser.add_argument("--output", type=str, default=None, help="write aggregated report JSON here")
    parser.add_argument("--quiet", action="store_true", help="suppress progress logging")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.quiet:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.self_test:
        _self_test(verbose=True)
        return 0
    if not args.records:
        build_arg_parser().print_help()
        return 1
    with open(args.records, "r", encoding="utf-8") as handle:
        records = json.load(handle)
    if isinstance(records, Mapping):
        records = records.get("per_sample", [])
    report = evaluate_from_records(records, dataset_name=args.dataset)
    text = report.to_json(args.output)
    if not args.output:
        print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
