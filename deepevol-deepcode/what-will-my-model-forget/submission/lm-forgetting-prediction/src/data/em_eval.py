"""SQuAD-2.0-style Exact Match (EM) grading.

The paper (Sec. 2) defines the Exact Match score of a model ``f`` on a dataset ``D`` as

    EM_{D,f} := |{<x, y> in D | f(x) = y}| / |D|

and the Clarifications state that a prediction is graded "using the exact match metric
using the evaluation script of SQuAD 2.0"
(https://rajpurkar.github.io/SQuAD-explorer/).

The SQuAD 2.0 evaluation script normalizes a string by:
  * lower-casing,
  * stripping punctuation (``string.punctuation``),
  * removing articles (``a``, ``an``, ``the``),
  * collapsing runs of whitespace.

A prediction is counted as correct when its normalization equals *any* of the
reference normalizations (max-over-references).

This module is used for (a) collecting ``D_R`` (mispredicted examples), (b) the base-EM
sanity numbers on ``D_PT`` (Appendix, Table 7: BART0_L ~ 50.50, FLAN-T5_L ~ 47.47,
FLAN-T5_3B ~ 51.31) and (c) the Edit Success Rate reported in Tables 3/4.
"""

from __future__ import annotations

import re
import string
from typing import Iterable, List, Sequence, Union

__all__ = [
    "normalize_answer",
    "metric_max_over_ground_truths",
    "exact_match_score",
    "em_score",
    "is_correct",
    "EM",
    "EM_percent",
]


# --------------------------------------------------------------------------------------
# Normalization (verbatim behaviour of the SQuAD 2.0 ``evaluate`` script)
# --------------------------------------------------------------------------------------
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize_answer(s: str) -> str:
    """Lower-case, remove punctuation/articles and collapse whitespace.

    Mirrors ``normalize_answer`` from the official SQuAD 2.0 evaluation script
    (``squad_v2/evaluate.py``).
    """
    if s is None:
        return ""

    def remove_articles(text: str) -> str:
        return _ARTICLES_RE.sub(" ", text)

    def white_space_fix(text: str) -> str:
        return _WS_RE.sub(" ", text).strip()

    def remove_punc(text: str) -> str:
        return text.translate(_PUNCT_TABLE)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))


def _tokenize(s: str) -> List[str]:
    """Whitespace tokenization used by the official F1/EM comparison."""
    return normalize_answer(s).split()


def exact_match_score(prediction: str, ground_truth: str) -> int:
    """1 if ``prediction`` normalizes to ``ground_truth``, else 0."""
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(
    metric_fn, prediction: str, ground_truths: Union[str, Sequence[str]]
) -> int:
    """Max of ``metric_fn(prediction, gt)`` over all references ``gt``.

    ``D_PT`` / ``D_R`` examples often carry multiple reference answers, so we take
    the maximum over them, exactly like the SQuAD 2.0 script.
    """
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    ground_truths = [g for g in ground_truths if g is not None]
    if len(ground_truths) == 0:
        return 0
    return max(int(metric_fn(prediction, gt)) for gt in ground_truths)


def em_score(prediction: str, references: Union[str, Sequence[str]]) -> int:
    """Convenience wrapper: max-over-references exact match (0/1)."""
    return metric_max_over_ground_truths(exact_match_score, prediction, references)


def is_correct(prediction: str, references: Union[str, Sequence[str]]) -> bool:
    """Boolean variant used by the ``D_R`` collectors / Edit-Success bookkeeping."""
    return bool(em_score(prediction, references))


def EM(
    dataset: Iterable[dict],
    predictions: Sequence[str],
    pred_key: str = "prediction",
    target_key: str = "target",
) -> float:
    """Exact Match of ``predictions`` on ``dataset`` (fraction in ``[0, 1]``).

    Parameters
    ----------
    dataset:
        Iterable of example dicts. Each dict is expected to hold a ``target``
        (str or list of str) field, or one of the tolerated aliases below.
    predictions:
        Model output strings, aligned 1:1 with ``dataset`` (or dicts carrying
        ``pred_key``).
    pred_key, target_key:
        Field names used to extract the prediction / reference from dict items.

    Returns
    -------
    float
        ``EM_{D,f}`` in ``[0, 1]`` (the paper reports the same quantity multiplied by
        100 in Table 7 - see :func:`EM_percent`).
    """
    examples = list(dataset)
    if len(examples) != len(predictions):
        raise ValueError(
            f"EM: found {len(predictions)} predictions for {len(examples)} examples"
        )
    if len(examples) == 0:
        return 0.0

    n_correct = 0
    for ex, pred in zip(examples, predictions):
        if isinstance(pred, dict):
            pred = pred.get(pred_key, "")
        refs = extract_references(ex, target_key)
        n_correct += em_score(pred, refs)
    return n_correct / len(examples)


def extract_references(example: dict, target_key: str = "target") -> List[str]:
    """Pull all reference strings out of an example dict (robust to key variants)."""
    for key in (
        target_key,
        "targets",
        "target",
        "answers",
        "answer",
        "label",
        "references",
    ):
        if key in example and example[key] is not None:
            value = example[key]
            if isinstance(value, (list, tuple)):
                flat: List[str] = []
                for v in value:
                    if isinstance(v, dict):  # e.g. SQuAD-style {"text": ...}
                        flat.append(str(v.get("text", "")))
                    else:
                        flat.append(str(v))
                return flat
            return [str(value)]
    raise KeyError(
        f"Could not find a reference answer in example keys={list(example.keys())}"
    )


def EM_percent(*args, **kwargs) -> float:
    """EM in percent (``100 * EM``) - the scale used by the paper's tables."""
    return 100.0 * EM(*args, **kwargs)
