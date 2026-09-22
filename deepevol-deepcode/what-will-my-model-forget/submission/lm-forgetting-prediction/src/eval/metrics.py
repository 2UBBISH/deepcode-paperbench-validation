"""Evaluation metrics for "What Will My Model Forget?" (forecasting forgotten examples).

Implements the metric definitions of Sec. 2 / Sec. 4.1 of the paper:

* Exact Match of a model ``f`` on a dataset ``D`` (Sec. 2)::

      EM_{D, f} := |{<x, y> in D | f(x) = y}| / |D|

  (grading uses the SQuAD-2.0-style normalization implemented in
  ``src/data/em_eval.py``).

* **F1 / Precision / Recall** of the binary forgetting forecast
  ``g(<x_i, y_i>, <x_j, y_j>) -> z_ij in {0, 1}`` ("Metrics" of Sec. 4.1:
  "We report F1 scores for binary forgetting prediction").  The paper's tables
  (Table 1, Table 2, Figure 3) report these numbers on a 0-100 scale
  (e.g. BART0 Head Representation F1 = 79.32), so all public metric helpers
  return *percent* values by default; pass ``percent=False`` data to obtain
  fractions in ``[0, 1]`` -- every helper accepts a ``percent`` flag.

* **Edit Success Rate** (Sec. 2)::

      EditSuccess := |{<x_i, y_i> in D_R | f_i(x_i) = y_i}| / |D_R|

  i.e. the proportion of mispredicted examples that produce the correct answer
  after the model update.

* **EM Drop Ratio** (Sec. 2)::

      EMDropRatio := (EM_{D_PT, f_i} - EM_{D_PT, f_0}) / EM_{D_PT, f_0}

  Note the sign convention of the paper: since refinement *degrades* upstream
  performance, ``EM_{D_PT, f_i} < EM_{D_PT, f_0}`` and the ratio is negative;
  the paper's tables report the magnitude (Table 3 "EM Drop %": Vanilla FT
  BART0 = 9.274, i.e. ``-9.274%``).  Both sign conventions are exposed via the
  ``as_magnitude`` argument so that tables can be reproduced directly
  while the raw (signed) quantity is also available.

Additionally provides helpers used by the rest of the code base (the threshold,
logit-based and representation-based forecasters probe this module for metric
functions):

* ``forecast_metrics`` / ``compute_forecast_metrics`` / ``binary_metrics``
* ``binary_f1`` / ``f1_score_binary`` / ``forecast_f1``
* ``precision_recall_f1`` / ``f1_from_counts`` / ``precision_from_counts`` /
  ``recall_from_counts``

and the continual-stream aggregation helper ``average_metrics_up_to_step``
used by ``scripts/run_continual_stream.py`` to reproduce Figure 3 (F1 /
Precision / Recall averaged up to each time step).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# SQuAD-2.0-style exact match (re-exported from the data layer)
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - exercised indirectly
    from ..data.em_eval import (  # type: ignore
        EM as _em_fraction,
        EM_percent as _em_percent,
        em_score,
        exact_match_score,
        extract_references,
        is_correct,
        normalize_answer,
    )

    _HAVE_EM_EVAL = True
except Exception:  # pragma: no cover - standalone fallback
    _HAVE_EM_EVAL = False

    def normalize_answer(s: Optional[str]) -> str:  # type: ignore[misc]
        import re
        import string

        if s is None:
            return ""
        s = str(s).lower()
        s = s.translate(str.maketrans("", "", string.punctuation))
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        return " ".join(s.split())

    def exact_match_score(prediction: Optional[str], ground_truth: Optional[str]) -> int:  # type: ignore[misc]
        return int(normalize_answer(prediction) == normalize_answer(ground_truth))

    def em_score(prediction: Optional[str], references: Any) -> int:  # type: ignore[misc]
        if references is None:
            return 0
        if isinstance(references, str):
            refs = [references]
        else:
            refs = [r for r in references if r is not None]
        if not refs:
            return 0
        return max(exact_match_score(prediction, r) for r in refs)

    def is_correct(prediction: Optional[str], references: Any) -> bool:  # type: ignore[misc]
        return bool(em_score(prediction, references))

    def extract_references(example: Mapping[str, Any], target_key: str = "target") -> List[str]:  # type: ignore[misc]
        for key in (target_key, "targets", "references", "answers", "label"):
            if key in example and example[key] is not None:
                value = example[key]
                if isinstance(value, str):
                    return [value]
                if isinstance(value, dict) and "text" in value:
                    return [str(value["text"])]
                if isinstance(value, (list, tuple)):
                    out: List[str] = []
                    for item in value:
                        if isinstance(item, dict) and "text" in item:
                            out.append(str(item["text"]))
                        else:
                            out.append(str(item))
                    if out:
                        return out
        raise KeyError(f"No reference field found in example with keys {sorted(example.keys())}")

    def EM(  # type: ignore[misc]
        dataset: Iterable[Mapping[str, Any]],
        predictions: Sequence[Optional[str]],
        pred_key: str = "prediction",
        target_key: str = "target",
    ) -> float:
        examples = list(dataset)
        if len(examples) != len(predictions):
            raise ValueError(f"EM: {len(examples)} examples vs {len(predictions)} predictions")
        if not examples:
            return 0.0
        total = 0.0
        for example, prediction in zip(examples, predictions):
            if prediction is None and pred_key in example:
                prediction = example[pred_key]
            refs = extract_references(example, target_key=target_key)
            total += em_score(prediction, refs)
        return total / len(examples)

    def EM_percent(*args: Any, **kwargs: Any) -> float:  # type: ignore[misc]
        return 100.0 * EM(*args, **kwargs)


logger = logging.getLogger("eval.metrics")

__all__ = [
    # EM
    "normalize_answer",
    "exact_match_score",
    "em_score",
    "is_correct",
    "extract_references",
    "exact_match",
    "exact_match_percent",
    "EM_DROP_RATIO_EPS",
    # low level counting
    "precision_from_counts",
    "recall_from_counts",
    "f1_from_counts",
    "precision_recall_f1",
    "confusion_counts",
    "prediction_prevalence",
    "true_prevalence",
    "accuracy",
    # binary forecasting metrics
    "binary_metrics",
    "compute_forecast_metrics",
    "forecast_metrics",
    "forecast_f1",
    "binary_f1",
    "f1_score_binary",
    "precision_score_binary",
    "recall_score_binary",
    "coerce_binary_labels",
    # refinement metrics
    "edit_success_rate",
    "edit_success_rate_from_records",
    "edit_success_flags",
    "em_drop_ratio",
    "em_drop_ratio_from_predictions",
    "em_drop_ratio_from_pairs",
    "predictions_to_examples",
    # aggregation / reporting
    "average_metrics_up_to_step",
    "aggregate_metric_histories",
    "per_task_metrics",
    "summarize_refinement",
    "metrics_to_table_row",
    "parse_args",
    "main",
]

# Numerical guard for the EM Drop Ratio denominator (EM_{D_PT, f0} > 0 in the
# paper: 50.50 / 47.47 / 51.31 for BART0_L / FLAN-T5_L / FLAN-T5_3B, Table 7).
EM_DROP_RATIO_EPS = 1e-8


# --------------------------------------------------------------------------- #
# Low-level counting helpers
# --------------------------------------------------------------------------- #
def precision_from_counts(tp: float, fp: float, zero_division: float = 0.0) -> float:
    """Precision = tp / (tp + fp) with a configurable zero-division value."""
    denom = tp + fp
    return float(tp / denom) if denom > 0 else float(zero_division)


def recall_from_counts(tp: float, fn: float, zero_division: float = 0.0) -> float:
    """Recall = tp / (tp + fn) with a configurable zero-division value."""
    denom = tp + fn
    return float(tp / denom) if denom > 0 else float(zero_division)


def f1_from_counts(
    tp: float,
    fp: float,
    fn: float,
    zero_division: float = 0.0,
    precision: Optional[float] = None,
    recall: Optional[float] = None,
) -> float:
    """F1 = 2PR / (P + R) computed from confusion counts (or given P/R)."""
    p = precision_from_counts(tp, fp, zero_division=zero_division) if precision is None else float(precision)
    r = recall_from_counts(tp, fn, zero_division=zero_division) if recall is None else float(recall)
    if (p + r) <= 0.0:
        return float(zero_division)
    return float(2.0 * p * r / (p + r))


def precision_recall_f1(tp: float, fp: float, fn: float, zero_division: float = 0.0) -> Dict[str, float]:
    """Return ``{"precision", "recall", "f1"}`` (fractions in ``[0, 1]``)."""
    p = precision_from_counts(tp, fp, zero_division=zero_division)
    r = recall_from_counts(tp, fn, zero_division=zero_division)
    return {
        "precision": p,
        "recall": r,
        "f1": f1_from_counts(tp, fp, fn, zero_division=zero_division, precision=p, recall=r),
    }


def accuracy(tp: float, tn: float, fp: float, fn: float, zero_division: float = 0.0) -> float:
    """Accuracy = (tp + tn) / total."""
    total = tp + tn + fp + fn
    return float((tp + tn) / total) if total > 0 else float(zero_division)


def coerce_binary_labels(values: Iterable[Any]) -> List[int]:
    """Coerce an iterable of (bool / float / int / tensor-like) labels to ``0/1`` ints."""
    out: List[int] = []
    for value in values:
        if value is None:
            raise ValueError("coerce_binary_labels: encountered None label")
        if hasattr(value, "item") and not isinstance(value, (str, bytes)):
            try:
                value = value.item()
            except Exception:  # pragma: no cover
                value = float(value)
        if isinstance(value, str):
            stripped = value.strip().lower()
            if stripped in {"1", "true", "yes", "pos", "positive", "forgotten", "forget"}:
                out.append(1)
            elif stripped in {"0", "false", "no", "neg", "negative", "not_forgotten", "kept"}:
                out.append(0)
            else:
                out.append(int(float(stripped) != 0.0))
            continue
        out.append(int(float(value) != 0.0))
    return out


def confusion_counts(z_true: Iterable[Any], z_pred: Iterable[Any]) -> Tuple[int, int, int, int]:
    """Return ``(tp, fp, fn, tn)`` for binary labels with positive class = 1."""
    y = coerce_binary_labels(z_true)
    y_hat = coerce_binary_labels(z_pred)
    if len(y) != len(y_hat):
        raise ValueError(f"confusion_counts: {len(y)} labels vs {len(y_hat)} predictions")
    tp = fp = fn = tn = 0
    for t, p in zip(y, y_hat):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 1 and p == 0:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def prediction_prevalence(z_pred: Iterable[Any]) -> float:
    """Fraction of positive forecasts (how many upstream examples are predicted forgotten)."""
    labels = coerce_binary_labels(z_pred)
    if not labels:
        return 0.0
    return float(sum(labels) / len(labels))


def true_prevalence(z_true: Iterable[Any]) -> float:
    """Fraction of truly forgotten pairs (the paper's 1%-10% minority prevalence)."""
    return prediction_prevalence(z_true)


# --------------------------------------------------------------------------- #
# Binary forecasting metrics (Sec. 4.1 "we report F1 scores for binary
# forgetting prediction"; tables report them on a 0-100 scale)
# --------------------------------------------------------------------------- #
def binary_metrics(
    z_true: Iterable[Any],
    z_pred: Iterable[Any],
    percent: bool = True,
    zero_division: float = 0.0,
    with_counts: bool = True,
) -> Dict[str, float]:
    """Precision / Recall / F1 (+ accuracy / counts) of forgetting forecasts.

    Args:
        z_true: ground-truth labels ``z_ij in {0, 1}``.
        z_pred: forecast labels ``z_hat_ij in {0, 1}``.
        percent: if ``True`` (default) metrics are returned on the paper's
            0-100 scale (Table 1 reports e.g. 79.32); otherwise fractions.
        zero_division: value used when a precision/recall denominator is 0.
        with_counts: include tp/fp/fn/tn and prevalences in the output.
    """
    tp, fp, fn, tn = confusion_counts(z_true, z_pred)
    core = precision_recall_f1(tp, fp, fn, zero_division=zero_division)
    scale = 100.0 if percent else 1.0
    result: Dict[str, float] = {
        "precision": core["precision"] * scale,
        "recall": core["recall"] * scale,
        "f1": core["f1"] * scale,
        "accuracy": accuracy(tp, tn, fp, fn, zero_division=zero_division) * scale,
    }
    if with_counts:
        n = tp + fp + fn + tn
        result.update(
            {
                "tp": float(tp),
                "fp": float(fp),
                "fn": float(fn),
                "tn": float(tn),
                "n": float(n),
                "predicted_prevalence": float((tp + fp) / n) if n else 0.0,
                "true_prevalence": float((tp + fn) / n) if n else 0.0,
            }
        )
    return result


# Aliases probed by the forecaster modules / scripts.
compute_forecast_metrics = binary_metrics
forecast_metrics = binary_metrics


def forecast_f1(z_true: Iterable[Any], z_pred: Iterable[Any], percent: bool = True, **kwargs: Any) -> float:
    """F1 of binary forgetting prediction (0-100 by default, as in Table 1)."""
    return float(binary_metrics(z_true, z_pred, percent=percent, **kwargs)["f1"])


# Names other modules look for (see ``logit_based.evaluate_labels`` etc.).
binary_f1 = forecast_f1
f1_score_binary = forecast_f1


def precision_score_binary(z_true: Iterable[Any], z_pred: Iterable[Any], percent: bool = True, **kwargs: Any) -> float:
    """Precision of binary forgetting prediction."""
    return float(binary_metrics(z_true, z_pred, percent=percent, **kwargs)["precision"])


def recall_score_binary(z_true: Iterable[Any], z_pred: Iterable[Any], percent: bool = True, **kwargs: Any) -> float:
    """Recall of binary forgetting prediction."""
    return float(binary_metrics(z_true, z_pred, percent=percent, **kwargs)["recall"])


# --------------------------------------------------------------------------- #
# Exact Match helpers (Sec. 2 definition, Table 7 sanity numbers)
# --------------------------------------------------------------------------- #
def exact_match(
    examples: Iterable[Mapping[str, Any]],
    predictions: Sequence[Optional[str]],
    percent: bool = False,
    pred_key: str = "prediction",
    target_key: str = "target",
) -> float:
    """``EM_{D, f} = |{<x, y> in D | f(x) = y}| / |D|`` (Sec. 2)."""
    value = _em_fraction(examples, predictions, pred_key=pred_key, target_key=target_key)
    return float(value * 100.0) if percent else float(value)


def exact_match_percent(
    examples: Iterable[Mapping[str, Any]],
    predictions: Sequence[Optional[str]],
    **kwargs: Any,
) -> float:
    """EM on the 0-100 scale used by Table 7 (BART0_L 50.50, FLAN-T5_L 47.47, ...)."""
    kwargs.pop("percent", None)
    return exact_match(examples, predictions, percent=True, **kwargs)


# --------------------------------------------------------------------------- #
# Edit Success Rate (Sec. 2)
# --------------------------------------------------------------------------- #
def edit_success_flags(
    online_examples: Sequence[Mapping[str, Any]],
    predictions: Optional[Sequence[Optional[str]]] = None,
    pred_key: str = "fi_prediction",
    target_key: str = "target",
) -> List[int]:
    """Per-example ``1[f_i(x_i) = y_i]`` flags for the online (refinement) examples.

    Predictions may be supplied explicitly or read from the example dicts under
    ``pred_key`` (``fi_prediction`` / ``prediction`` / ``f_i_prediction``).
    """
    flags: List[int] = []
    for index, example in enumerate(online_examples):
        prediction: Optional[str] = None
        if predictions is not None:
            prediction = predictions[index]
        else:
            for key in (pred_key, "fi_prediction", "f_i_prediction", "prediction", "edit_prediction"):
                if key in example and example[key] is not None:
                    prediction = example[key]
                    break
        if prediction is None:
            raise ValueError(
                "edit_success_flags: no post-refinement prediction found; pass `predictions` "
                "or store it under 'fi_prediction'/'prediction'"
            )
        refs = extract_references(example, target_key=target_key)
        flags.append(int(is_correct(prediction, refs)))
    return flags


def edit_success_rate(
    online_examples: Optional[Sequence[Mapping[str, Any]]] = None,
    predictions: Optional[Sequence[Optional[str]]] = None,
    flags: Optional[Iterable[Any]] = None,
    percent: bool = True,
    pred_key: str = "fi_prediction",
    target_key: str = "target",
) -> float:
    """Edit Success Rate = ``|{<x_i, y_i> in D_R | f_i(x_i) = y_i}| / |D_R|`` (Sec. 2).

    Accepts either explicit success ``flags`` or (examples, predictions).
    The paper reports the rate in percent (Table 3: Vanilla FT BART0 90.4).
    """
    if flags is not None:
        values = coerce_binary_labels(flags)
        if not values:
            return 0.0
        rate = sum(values) / len(values)
    else:
        if online_examples is None:
            raise ValueError("edit_success_rate: provide `flags` or `online_examples`")
        computed = edit_success_flags(
            online_examples, predictions=predictions, pred_key=pred_key, target_key=target_key
        )
        if not computed:
            return 0.0
        rate = sum(computed) / len(computed)
    return float(rate * 100.0) if percent else float(rate)


def edit_success_rate_from_records(records: Sequence[Any], percent: bool = True, key: str = "edit_success") -> float:
    """Edit Success Rate from ground-truth ``OnlineRecord``-like objects/dicts."""
    flags: List[int] = []
    for record in records:
        if isinstance(record, Mapping):
            if key in record and record[key] is not None:
                flags.append(int(bool(record[key])))
                continue
            # fall back to correctness fields produced by ground_truth.OnlineRecord
            if "fi_correct" in record and record["fi_correct"] is not None:
                flags.append(int(bool(record["fi_correct"])))
                continue
        else:
            if hasattr(record, key) and getattr(record, key) is not None:
                flags.append(int(bool(getattr(record, key))))
                continue
            if hasattr(record, "fi_correct") and getattr(record, "fi_correct") is not None:
                flags.append(int(bool(getattr(record, "fi_correct"))))
                continue
        raise ValueError(f"edit_success_rate_from_records: could not find '{key}'/'fi_correct' in record {record!r}")
    if not flags:
        return 0.0
    rate = sum(flags) / len(flags)
    return float(rate * 100.0) if percent else float(rate)


# --------------------------------------------------------------------------- #
# EM Drop Ratio (Sec. 2)
# --------------------------------------------------------------------------- #
def em_drop_ratio(
    em_before: float,
    em_after: float,
    percent: bool = True,
    as_magnitude: bool = True,
    eps: float = EM_DROP_RATIO_EPS,
) -> float:
    """``(EM_{D_PT,f_i} - EM_{D_PT,f_0}) / EM_{D_PT,f_0}`` (Sec. 2).

    Args:
        em_before: ``EM_{D_PT, f_0}`` (base PTLM upstream EM, Table 7).
        em_after: ``EM_{D_PT, f_i}`` (upstream EM after refinement).
        percent: scale the ratio by 100 (the paper's "EM Drop %" rows).
        as_magnitude: return the non-negative magnitude reported in the tables
            (Table 3: Vanilla FT BART0 = 9.274).  If ``False`` the signed ratio
            is returned (negative when forgetting occurs).
    """
    if em_before is None or em_after is None:
        raise ValueError("em_drop_ratio: em_before and em_after must both be provided")
    if abs(float(em_before)) <= eps:
        raise ValueError(f"em_drop_ratio: EM before refinement is ~0 ({em_before}); ratio undefined")
    ratio = (float(em_after) - float(em_before)) / float(em_before)
    if percent:
        ratio *= 100.0
    return float(abs(ratio)) if as_magnitude else float(ratio)


def predictions_to_examples(
    examples: Sequence[Mapping[str, Any]],
    predictions: Sequence[Optional[str]],
    pred_key: str = "prediction",
) -> List[Dict[str, Any]]:
    """Attach predictions to shallow copies of ``examples`` (helper for EM)."""
    if len(examples) != len(predictions):
        raise ValueError(f"predictions_to_examples: {len(examples)} examples vs {len(predictions)} predictions")
    out: List[Dict[str, Any]] = []
    for example, prediction in zip(examples, predictions):
        merged = dict(example)
        merged[pred_key] = prediction
        out.append(merged)
    return out


def em_drop_ratio_from_predictions(
    upstream_examples: Sequence[Mapping[str, Any]],
    em_predictions_before: Sequence[Optional[str]],
    em_predictions_after: Sequence[Optional[str]],
    percent: bool = True,
    as_magnitude: bool = True,
    target_key: str = "target",
) -> Dict[str, float]:
    """EM Drop Ratio from two prediction lists over the same ``D_PT``.

    Returns a dict with ``em_before``, ``em_after`` and ``em_drop`` (the ratio,
    already scaled/signed per the ``percent``/``as_magnitude`` arguments).
    """
    em_before = exact_match(upstream_examples, em_predictions_before, percent=True, target_key=target_key)
    em_after = exact_match(upstream_examples, em_predictions_after, percent=True, target_key=target_key)
    return {
        "em_before": float(em_before),
        "em_after": float(em_after),
        "em_drop": em_drop_ratio(em_before, em_after, percent=percent, as_magnitude=as_magnitude),
        "n_upstream": float(len(upstream_examples)),
    }


def em_drop_ratio_from_pairs(
    pair_records: Sequence[Any],
    percent: bool = True,
    as_magnitude: bool = True,
    base_em: Optional[float] = None,
) -> Dict[str, float]:
    """EM Drop Ratio estimated from labelled pairs ``z_ij = 1[f_i(x_j) != y_j]``.

    Because every pair in ``D_PT_hat`` is correctly answered by ``f_0``
    (``EM_{D_PT_hat, f_0} = 100%``), ``EM_{D_PT_hat, f_i} = 100% - prevalence``
    (in percent), so the drop ratio can be computed from ground-truth pairs
    without re-running inference.  If ``base_em`` is given (the EM of ``f_0`` on
    the *unfiltered* ``D_PT``, e.g. 50.50 for BART0_L) it is used as the
    denominator instead, matching the paper's ``EM_{D_PT, f_0}``.
    """
    z = []
    for record in pair_records:
        if isinstance(record, Mapping):
            z.append(record.get("z", record.get("label", 0)))
        else:
            z.append(getattr(record, "z", getattr(record, "label", 0)))
    labels = coerce_binary_labels(z)
    if not labels:
        raise ValueError("em_drop_ratio_from_pairs: no pair records supplied")
    em_after = 100.0 * (1.0 - sum(labels) / len(labels))
    em_before = 100.0 if base_em is None else float(base_em)
    return {
        "em_before": float(em_before),
        "em_after": float(em_after),
        "em_drop": em_drop_ratio(em_before, em_after, percent=percent, as_magnitude=as_magnitude),
        "n_pairs": float(len(labels)),
        "forgotten_prevalence": float(sum(labels) / len(labels)),
    }


# --------------------------------------------------------------------------- #
# Continual-stream aggregation (Figure 3)
# --------------------------------------------------------------------------- #
def average_metrics_up_to_step(history: Sequence[Mapping[str, Any]]) -> List[Dict[str, float]]:
    """Running averages of F1 / Precision / Recall "averaged up to each time step".

    ``history[k]`` holds the metrics measured at time step ``k`` (cumulative
    confusion counts are accepted via ``tp``/``fp``/``fn``).  Figure 3 plots the
    average up to the end of the stream, so element ``k`` of the returned list
    is the average over steps ``0..k``.
    """
    running: List[Dict[str, float]] = []
    sums: Dict[str, float] = {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    counts = 0
    for step_metrics in history:
        counts += 1
        if all(key in step_metrics for key in ("tp", "fp", "fn")):
            derived = precision_recall_f1(
                float(step_metrics["tp"]), float(step_metrics["fp"]), float(step_metrics["fn"])
            )
            for key, value in derived.items():
                sums[key] += value
        else:
            for key in ("f1", "precision", "recall"):
                value = step_metrics.get(key)
                if value is None:
                    raise ValueError(f"average_metrics_up_to_step: missing '{key}' in step metrics {step_metrics!r}")
                value = float(value)
                # accept 0-100 inputs and normalise to a fraction internally
                sums[key] += value / 100.0 if value > 1.0 else value
        running.append(
            {
                "step": float(len(running)),
                "f1": 100.0 * sums["f1"] / counts,
                "precision": 100.0 * sums["precision"] / counts,
                "recall": 100.0 * sums["recall"] / counts,
            }
        )
    return running


def aggregate_metric_histories(histories: Sequence[Sequence[Mapping[str, Any]]]) -> List[Dict[str, float]]:
    """Average several stream runs (e.g. multiple error streams) step by step.

    Histories may have different lengths; shorter runs stop contributing after
    their last step.  Returns one averaged ``{step, f1, precision, recall}``
    record per step index, with metrics on the 0-100 scale.
    """
    normalized: List[List[Dict[str, float]]] = [average_metrics_up_to_step(h) for h in histories if h]
    if not normalized:
        return []
    max_len = max(len(run) for run in normalized)
    out: List[Dict[str, float]] = []
    for step in range(max_len):
        present = [run[step] for run in normalized if step < len(run)]
        count = len(present)
        out.append(
            {
                "step": float(step),
                "f1": sum(item["f1"] for item in present) / count,
                "precision": sum(item["precision"] for item in present) / count,
                "recall": sum(item["recall"] for item in present) / count,
                "n_runs": float(count),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #
def per_task_metrics(
    z_true: Sequence[Any],
    z_pred: Sequence[Any],
    tasks: Sequence[str],
    percent: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Grouped precision/recall/F1 per upstream task (Table 2 ID/OOD buckets use this)."""
    if not (len(z_true) == len(z_pred) == len(tasks)):
        raise ValueError("per_task_metrics: z_true, z_pred and tasks must have equal length")
    buckets: Dict[str, List[int]] = {}
    for index, task in enumerate(tasks):
        buckets.setdefault(str(task), []).append(index)
    out: Dict[str, Dict[str, float]] = {}
    for task, indices in sorted(buckets.items()):
        y = [z_true[i] for i in indices]
        y_hat = [z_pred[i] for i in indices]
        out[task] = binary_metrics(y, y_hat, percent=percent)
    return out


def summarize_refinement(
    records: Sequence[Any],
    base_em: Optional[float] = None,
    percent: bool = True,
) -> Dict[str, float]:
    """Edit Success Rate + EM Drop Ratio summary for a refinement run.

    ``records`` are ``OnlineRecord``-like objects (``edit_success``,
    ``f0_correct``/``fi_correct``) or mappings with a ``z``/``forgotten`` field.
    """
    summary: Dict[str, float] = {}
    edit_rate = edit_success_rate_from_records(records, percent=percent)
    summary["edit_success"] = edit_rate
    flags: List[float] = []
    for record in records:
        if isinstance(record, Mapping):
            if "z" in record:
                flags.append(float(record["z"]))
            elif "forgotten" in record:
                flags.append(float(bool(record["forgotten"])))
        else:
            if hasattr(record, "z"):
                flags.append(float(getattr(record, "z")))
            elif hasattr(record, "forgotten"):
                flags.append(float(bool(getattr(record, "forgotten"))))
    if flags:
        em_after = 100.0 * (1.0 - sum(flags) / len(flags))
        em_before = 100.0 if base_em is None else float(base_em)
        summary["em_before"] = float(em_before)
        summary["em_after"] = float(em_after)
        summary["em_drop"] = em_drop_ratio(em_before, em_after, percent=percent, as_magnitude=True)
        summary["forgotten_prevalence"] = float(sum(flags) / len(flags))
    return summary


def metrics_to_table_row(name: str, metrics: Mapping[str, Any]) -> str:
    """Format one metrics dict as a compact table row (percent scale)."""
    return (
        f"{name:<22} P={float(metrics.get('precision', 0.0)):6.2f} "
        f"R={float(metrics.get('recall', 0.0)):6.2f} "
        f"F1={float(metrics.get('f1', 0.0)):6.2f}"
    )


# --------------------------------------------------------------------------- #
# CLI / self-test
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluation metrics for forgetting forecasting")
    parser.add_argument("--self-test", action="store_true", help="run built-in metric checks and exit")
    parser.add_argument("--pairs", default=None, help="JSONL file of labelled pairs (fields: z, z_hat) to score")
    parser.add_argument("--pred-jsonl", default=None, help="JSONL file of forecasts (fields: z, z_hat)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = parse_args(argv)

    if args.self_test or (args.pairs is None and args.pred_jsonl is None):
        # z_true / z_pred with tp=3, fp=1, fn=2, tn=14 -> P=75, R=60, F1=66.667
        y = [1, 1, 1, 1, 1, 0] * 3 + [0] * 2
        y_hat = [1, 1, 1, 0, 0, 1] + [0] * 5 + [0] * 3 + [0] * 2
        metrics = binary_metrics(y, y_hat)
        logger.info("self-test binary metrics: %s", json.dumps(metrics, sort_keys=True))
        assert abs(metrics["f1"] - 2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"])) < 1e-6
        assert abs(f1_from_counts(3, 1, 2) - 2 * 0.75 * 0.6 / (0.75 + 0.6)) < 1e-9
        assert abs(binary_f1([1, 0], [1, 1]) - 66.66666666666666) < 1e-6
        # Edit success: 3 of 4 corrected
        rate = edit_success_rate(flags=[1, 1, 0, 1])
        assert abs(rate - 75.0) < 1e-9, rate
        # EM Drop Ratio: 50.50 -> 45.82 (Vanilla FT BART0 magnitude ~9.27)
        assert abs(em_drop_ratio(50.50, 50.50 * (1 - 0.09274)) - 9.274) < 1e-6
        assert em_drop_ratio(50.50, 50.50 * (1 - 0.09274), as_magnitude=False) < 0
        # Continual-stream running averages
        hist = [{"tp": 2, "fp": 1, "fn": 1}, {"tp": 1, "fp": 2, "fn": 2}]
        avg = average_metrics_up_to_step(hist)
        assert len(avg) == 2 and avg[1]["f1"] <= avg[0]["f1"] + 1e-9
        # EM helper on tiny examples
        examples = [{"target": "The cat."}, {"target": "dog"}]
        assert abs(exact_match(examples, ["cat", "cat"], percent=True) - 50.0) < 1e-9
        logger.info("metrics self-test passed")
        return 0

    path = args.pairs or args.pred_jsonl
    if not os.path.isfile(path):
        raise FileNotFoundError(f"metrics: no such file: {path}")
    z_true: List[int] = []
    z_pred: List[int] = []
    for line in open(path, "r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if isinstance(record, Mapping):
            if "z" in record and "z_hat" in record:
                z_true.append(int(record["z"]))
                z_pred.append(int(record["z_hat"]))
            elif "results" in record:
                for item in record["results"]:
                    z_true.append(int(item.get("z", item.get("z_true", 0))))
                    z_pred.append(int(item.get("z_hat", item.get("z_pred", 0))))
    metrics = binary_metrics(z_true, z_pred)
    logger.info(metrics_to_table_row("forecast", metrics))
    logger.info("counts: tp=%.0f fp=%.0f fn=%.0f tn=%.0f n=%.0f", metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"], metrics["n"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
