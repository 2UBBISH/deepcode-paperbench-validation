"""Metrics used throughout the paper.

* Exact Match is computed with the official SQuAD 2.0 evaluation script's answer
  normalisation.  The addendum states that "a prediction is graded using the
  exact match metric using the evaluation script of SQuAD 2.0".
* ``binary_f1_scores`` implements the F1 of the forgetting-forecasting task
  (Sec. 4.1: "We report F1 scores for binary forgetting prediction").
* ``edit_success_rate`` and ``em_drop_ratio`` follow the definitions in Sec. 2.
"""
from __future__ import annotations

import collections
import re
import string
from typing import Dict, Iterable, List, Sequence, Tuple

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)


# --------------------------------------------------------------------------------------
# SQuAD 2.0 normalisation (ported from the official evaluate_v2.py)
# --------------------------------------------------------------------------------------
def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = _ARTICLES_RE.sub(" ", s)
    return " ".join(s.split())


def is_correct(prediction: str, reference: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(reference)


def exact_match(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Exact Match as defined in Sec. 2 of the paper."""
    assert len(predictions) == len(references), "predictions/references length mismatch"
    if not predictions:
        return 0.0
    hits = sum(is_correct(p, r) for p, r in zip(predictions, references))
    return hits / len(predictions)


def edit_success_rate(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Proportion of examples of ``D_R`` that the updated model answers correctly.

    ``Edit Success Rate = |{<x_i, y_i> in D_R : f_i(x_i) = y_i}| / |D_R|`` (Sec. 2).
    """
    return exact_match(predictions, references)


def em_drop_ratio(em_updated: float, em_base: float) -> float:
    """``(EM_{D_PT, f_i} - EM_{D_PT, f_0}) / EM_{D_PT, f_0}`` (Sec. 2).

    Tables 3 and 4 of the paper report this value in percent; callers multiply by
    100 themselves (see ``wwmf.evaluation.report``).
    """
    if em_base == 0:
        return float("nan")
    return (em_updated - em_base) / em_base


# --------------------------------------------------------------------------------------
# binary F1 for forgetting forecasting
# --------------------------------------------------------------------------------------
def precision_recall_f1(y_true: Iterable[int], y_pred: Iterable[int]) -> Tuple[float, float, float]:
    """Precision / Recall / F1 of the positive (forgotten) class, in percent."""
    y_true = [int(v) for v in y_true]
    y_pred = [int(v) for v in y_pred]
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return 100.0 * precision, 100.0 * recall, 100.0 * f1


def binary_f1_scores(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, float]:
    precision, recall, f1 = precision_recall_f1(y_true, y_pred)
    return {"precision": precision, "recall": recall, "f1": f1}


def average_precision(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Area under the precision-recall curve (diagnostic for threshold tuning)."""
    order = sorted(range(len(y_score)), key=lambda i: -y_score[i])
    tp = fp = 0
    n_pos = sum(1 for v in y_true if v == 1)
    if n_pos == 0:
        return 0.0
    ap = 0.0
    prev_recall = 0.0
    for idx in order:
        if y_true[idx] == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / n_pos
        precision = tp / (tp + fp)
        ap += precision * (recall - prev_recall)
        prev_recall = recall
    return 100.0 * ap


def forget_frequency(labels: Sequence[Sequence[int]]) -> List[float]:
    """Frequency of forgetting of every upstream example.

    ``labels[i][j] == z_ij``; the returned list is indexed by ``j`` (upstream
    example).  This is the statistic used by threshold-based forecasting (Sec. 3.1).
    """
    if not labels:
        return []
    n_j = len(labels[0])
    counts: collections.Counter = collections.Counter()
    for row in labels:
        assert len(row) == n_j
        for j, v in enumerate(row):
            if v:
                counts[j] += 1
    return [counts[j] / len(labels) for j in range(n_j)]
