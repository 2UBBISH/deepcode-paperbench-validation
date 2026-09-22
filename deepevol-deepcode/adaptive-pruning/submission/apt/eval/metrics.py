"""Task metrics used by the APT evaluation harness.

This module centralises every end-task metric reported in the paper (Section 5.3
and Tables 2/4/7/8):

* GLUE classification/regression metrics -- accuracy, F1, Matthews correlation,
  Spearman correlation and the GLUE-average aggregation.
* SQuAD v2.0 exact match / F1 (delegated to :mod:`apt.data.squad` when it is
  importable, with a dependency-light fallback).
* CNN/DailyMail ROUGE-1/2/L (delegated to :mod:`apt.data.cnndm` when available,
  with a self-contained fallback).

The module is intentionally dependency-light: heavy packages (``sklearn``,
``scipy``, ``rouge_score``) are imported lazily and every metric has a NumPy or
pure-Python fallback so the module can be imported and unit-tested in a bare
environment.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# Task families
# ---------------------------------------------------------------------------

GLUE_TASKS: Tuple[str, ...] = (
    "mnli",
    "sst2",
    "qnli",
    "qqp",
    "mrpc",
    "cola",
    "rte",
    "stsb",
)
GLUE_BIG_TASKS: Tuple[str, ...] = ("mnli", "sst2", "qnli", "qqp")
GLUE_SMALL_TASKS: Tuple[str, ...] = ("mrpc", "cola", "rte", "stsb")

SQUAD_TASKS: Tuple[str, ...] = ("squad", "squad_v2", "squadv2")
SEQ2SEQ_TASKS: Tuple[str, ...] = ("cnndm", "cnn_dailymail", "cnn-dailymail", "summarization")

#: GLUE tasks whose primary metric is accuracy-like (not F1).
GLUE_ACCURACY_TASKS: Tuple[str, ...] = ("mnli", "mnli-mm", "sst2", "qnli", "rte", "cola")
GLUE_F1_TASKS: Tuple[str, ...] = ("qqp", "mrpc")
GLUE_REGRESSION_TASKS: Tuple[str, ...] = ("stsb", "sts-b")

#: Primary metric name per canonical GLUE task (mirrors the HF `glue` metric).
GLUE_PRIMARY_METRIC: Dict[str, str] = {
    "mnli": "accuracy",
    "sst2": "accuracy",
    "qnli": "accuracy",
    "rte": "accuracy",
    "cola": "matthews_correlation",
    "qqp": "f1",
    "mrpc": "f1",
    "stsb": "spearmanr",
}

#: Paper-order metric keys for the CNN/DM table column ("42.1/20.3/39.4").
ROUGE_KEYS: Tuple[str, ...] = ("rouge1", "rouge2", "rougeL")


# ---------------------------------------------------------------------------
# Task-name helpers
# ---------------------------------------------------------------------------

def normalize_task_name(task: str) -> str:
    """Lower-case / strip separators so aliases resolve to canonical keys."""
    if task is None:
        return ""
    key = str(task).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "sst_2": "sst2",
        "mnli_matched": "mnli",
        "mnli_mismatched": "mnli",
        "mnli_mm": "mnli",
        "sts_b": "stsb",
        "squadv2": "squad_v2",
        "squad2": "squad_v2",
        "squadv1": "squad",
        "squad1": "squad",
        "cnn_dailymail": "cnndm",
        "cnn_dm": "cnndm",
        "cnndm3": "cnndm",
        "summarization": "cnndm",
    }
    return aliases.get(key, key)


def is_glue_task(task: str) -> bool:
    return normalize_task_name(task) in GLUE_TASKS


def is_squad_task(task: str) -> bool:
    return normalize_task_name(task) in ("squad", "squad_v2")


def is_seq2seq_task(task: str) -> bool:
    return normalize_task_name(task) in ("cnndm",) or normalize_task_name(task) in SEQ2SEQ_TASKS


def is_big_glue_task(task: str) -> bool:
    return normalize_task_name(task) in GLUE_BIG_TASKS


def is_regression_task(task: str) -> bool:
    return normalize_task_name(task) in GLUE_REGRESSION_TASKS


def primary_metric_name(task: str) -> str:
    """Name of the metric the paper reports for a task."""
    key = normalize_task_name(task)
    if is_glue_task(key):
        return GLUE_PRIMARY_METRIC.get(key, "accuracy")
    if is_squad_task(key):
        return "f1"
    if key in ("cnndm", "cnn_dailymail", "cnn_dm") or key in SEQ2SEQ_TASKS:
        return "rougeL"
    return "accuracy"


# ---------------------------------------------------------------------------
# Generic metric helpers
# ---------------------------------------------------------------------------

def _to_list(values: Any) -> List[Any]:
    """Convert numpy/torch containers to plain python lists."""
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        return list(values)
    if hasattr(values, "detach"):  # torch tensor
        values = values.detach().cpu()
    if hasattr(values, "tolist"):
        try:
            return list(values.tolist())
        except Exception:  # pragma: no cover - defensive
            pass
    try:
        return list(values)
    except TypeError:
        return [values]


def _flatten(values: Any) -> List[float]:
    out: List[float] = []
    for item in _to_list(values):
        if isinstance(item, (list, tuple)):
            out.extend(float(v) for v in item)
        elif hasattr(item, "tolist") and not isinstance(item, (int, float, str)):
            sub = item.tolist()
            if isinstance(sub, list):
                out.extend(float(v) for v in sub)
            else:
                out.append(float(sub))
        else:
            out.append(float(item))
    return out


def accuracy(predictions: Any, references: Any, ignore_index: Optional[int] = -100) -> float:
    """Plain accuracy in ``[0, 100]``."""
    preds = _to_list(predictions)
    refs = _to_list(references)
    if not preds:
        return 0.0
    n = min(len(preds), len(refs))
    correct = 0
    total = 0
    for p, r in zip(preds[:n], refs[:n]):
        if ignore_index is not None and r == ignore_index:
            continue
        total += 1
        if p == r:
            correct += 1
    return 100.0 * correct / total if total else 0.0


def binary_f1(predictions: Any, references: Any, positive_label: int = 1) -> float:
    """Binary F1 in ``[0, 100]`` using ``positive_label`` as the positive class."""
    preds = _to_list(predictions)
    refs = _to_list(references)
    n = min(len(preds), len(refs))
    tp = fp = fn = 0
    for p, r in zip(preds[:n], refs[:n]):
        p_pos = int(p) == positive_label
        r_pos = int(r) == positive_label
        if p_pos and r_pos:
            tp += 1
        elif p_pos and not r_pos:
            fp += 1
        elif r_pos and not p_pos:
            fn += 1
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 100.0 * 2 * precision * recall / (precision + recall)


def matthews_corrcoef(predictions: Any, references: Any) -> float:
    """Matthews correlation coefficient (optionally via sklearn)."""
    preds = _to_list(predictions)
    refs = _to_list(references)
    n = min(len(preds), len(refs))
    preds, refs = preds[:n], refs[:n]
    try:  # pragma: no cover - optional dependency
        from sklearn.metrics import matthews_corrcoef as _mcc

        return 100.0 * float(_mcc(refs, preds))
    except Exception:
        pass
    tp = tn = fp = fn = 0
    for p, r in zip(preds, refs):
        if p == r == 1:
            tp += 1
        elif p == r == 0:
            tn += 1
        elif p != r and p == 1:
            fp += 1
        else:
            fn += 1
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return 0.0
    return 100.0 * (tp * tn - fp * fn) / denom


def spearman_correlation(predictions: Any, references: Any) -> float:
    """Spearman rank correlation x100 (optionally via scipy)."""
    preds = _flatten(predictions)
    refs = _flatten(references)
    n = min(len(preds), len(refs))
    preds, refs = preds[:n], refs[:n]
    if n < 2:
        return 0.0
    try:  # pragma: no cover - optional dependency
        from scipy.stats import spearmanr

        return 100.0 * float(spearmanr(preds, refs).correlation)
    except Exception:
        pass
    return 100.0 * _rank_correlation(preds, refs)


def _ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _rank_correlation(a: Sequence[float], b: Sequence[float]) -> float:
    ra, rb = _ranks(a), _ranks(b)
    n = len(ra)
    if n < 2:
        return 0.0
    mean_a = sum(ra) / n
    mean_b = sum(rb) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb))
    var_a = sum((x - mean_a) ** 2 for x in ra)
    var_b = sum((y - mean_b) ** 2 for y in rb)
    denom = math.sqrt(var_a * var_b)
    return cov / denom if denom else 0.0


# ---------------------------------------------------------------------------
# GLUE / classification metric bundle
# ---------------------------------------------------------------------------

def compute_glue_metrics(
    task: str,
    predictions: Any,
    references: Any,
    label_names: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Primary GLUE metric plus ``accuracy`` for non-accuracy tasks.

    Values are scaled x100 to match the paper's tables (e.g. SST2 ``94.8``).
    """
    key = normalize_task_name(task)
    preds = _to_list(predictions)
    refs = _to_list(references)
    if is_regression_task(key):
        # `predictions` may be a mapping {"predictions": [...]} so unwrap it.
        metrics: Dict[str, float] = {}
        if isinstance(predictions, dict):
            preds = _to_list(predictions.get("predictions", predictions))
        metrics["spearmanr"] = spearman_correlation(preds, refs)
        metrics["pearsonr"] = _pearson(preds, refs) * 100.0
        try:
            mse = sum((float(p) - float(r)) ** 2 for p, r in zip(preds, refs)) / max(1, len(preds))
        except Exception:  # pragma: no cover - defensive
            mse = 0.0
        metrics["mse"] = mse
        return metrics

    metrics = {}
    if key in GLUE_F1_TASKS:
        metrics["f1"] = binary_f1(preds, refs, positive_label=1)
        metrics["accuracy"] = accuracy(preds, refs)
        metrics["precision"], metrics["recall"] = _precision_recall(preds, refs)
    elif key in GLUE_ACCURACY_TASKS:
        metrics["accuracy"] = accuracy(preds, refs)
    else:
        metrics["accuracy"] = accuracy(preds, refs)

    if key == "cola":
        metrics["matthews_correlation"] = matthews_corrcoef(preds, refs)
    primary = GLUE_PRIMARY_METRIC.get(key, "accuracy")
    metrics["primary"] = metrics.get(primary, metrics.get("accuracy", 0.0))
    return metrics


def _precision_recall(predictions: Any, references: Any, positive_label: int = 1) -> Tuple[float, float]:
    preds = _to_list(predictions)
    refs = _to_list(references)
    n = min(len(preds), len(refs))
    tp = fp = fn = 0
    for p, r in zip(preds[:n], refs[:n]):
        p_pos = int(p) == positive_label
        r_pos = int(r) == positive_label
        if p_pos and r_pos:
            tp += 1
        elif p_pos:
            fp += 1
        elif r_pos:
            fn += 1
    precision = 100.0 * tp / (tp + fp) if (tp + fp) else 0.0
    recall = 100.0 * tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall


def _pearson(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    x, y = list(a[:n]), list(b[:n])
    mx, my = sum(x) / n, sum(y) / n
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    vx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    vy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    return cov / (vx * vy) if vx and vy else 0.0


# ---------------------------------------------------------------------------
# GLUE average (Table 7 / Table 8)
# ---------------------------------------------------------------------------

def glue_average(
    per_task: Dict[str, Union[float, Dict[str, float]]],
    tasks: Optional[Sequence[str]] = None,
    metric: Optional[Union[str, Dict[str, str]]] = None,
) -> float:
    """Average the primary metric over GLUE tasks (paper GLUE Avg, e.g. 83.2)."""
    task_list = list(tasks) if tasks is not None else list(per_task.keys())
    values: List[float] = []
    for task in task_list:
        key = normalize_task_name(task)
        if key not in per_task:
            # tolerate raw/alias keys stored in the dict
            match = next((k for k in per_task if normalize_task_name(k) == key), None)
            if match is None:
                continue
            key = match
        value = per_task[key]
        if isinstance(value, dict):
            name = metric
            if isinstance(metric, dict):
                name = metric.get(normalize_task_name(task), GLUE_PRIMARY_METRIC.get(normalize_task_name(task), "accuracy"))
            name = name or GLUE_PRIMARY_METRIC.get(normalize_task_name(key), "accuracy")
            value = value.get(name, value.get("primary", value.get("accuracy", 0.0)))
        values.append(float(value))
    if not values:
        return 0.0
    return sum(values) / len(values)


def aggregate_glue_metrics(
    metrics_by_task: Dict[str, Dict[str, float]],
    tasks: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """Return ``{task: primary}`` plus an ``"average"`` entry."""
    tasks = tasks or GLUE_TASKS
    out: Dict[str, float] = {}
    for task in tasks:
        key = normalize_task_name(task)
        if key not in metrics_by_task:
            continue
        per_task = metrics_by_task[key]
        out[key] = float(per_task.get("primary", per_task.get("accuracy", per_task.get("f1", 0.0))))
    out["average"] = glue_average(out, tasks=[t for t in tasks if normalize_task_name(t) in out])
    return out


# ---------------------------------------------------------------------------
# SQuAD
# ---------------------------------------------------------------------------

def _normalize_answer(text: str) -> str:
    """Official SQuAD answer normalisation."""
    if text is None:
        return ""

    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_punc(s: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in s if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(str(text).lower())))


def squad_exact_match(prediction: str, ground_truth: Union[str, Sequence[str]]) -> float:
    pred = _normalize_answer(prediction)
    truths = [ground_truth] if isinstance(ground_truth, str) else list(ground_truth)
    return float(any(pred == _normalize_answer(t) for t in truths))


def squad_f1(prediction: str, ground_truth: Union[str, Sequence[str]]) -> float:
    pred = _normalize_answer(prediction)
    truths = [ground_truth] if isinstance(ground_truth, str) else list(ground_truth)
    if not pred:
        return 0.0
    best = 0.0
    for truth in truths:
        truth_norm = _normalize_answer(truth)
        if not truth_norm:
            continue
        pred_tokens = pred.split()
        truth_tokens = truth_norm.split()
        common = Counter(pred_tokens) & Counter(truth_tokens)
        n_common = sum(common.values())
        if n_common == 0:
            continue
        precision = n_common / len(pred_tokens)
        recall = n_common / len(truth_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return 100.0 * best


def compute_squad_metrics(
    predictions: Union[Dict[str, str], Sequence[str]],
    references: Union[Dict[str, Any], Sequence[Any]],
    no_answer_probs: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """SQuAD v2.0 exact match / F1 in ``[0, 100]``.

    Delegates to :mod:`apt.data.squad` when possible (identical implementation),
    otherwise uses the local fallbacks above.
    """
    try:  # pragma: no cover - prefer the canonical implementation
        from apt.data.squad import compute_squad_metrics as _squad

        return _squad(predictions, references, no_answer_probs=no_answer_probs)
    except Exception:
        pass

    ems: List[float] = []
    f1s: List[float] = []
    has_ans_em: List[float] = []
    has_ans_f1: List[float] = []
    no_ans: List[float] = []
    total = 0

    if isinstance(predictions, dict):
        ids = list(predictions.keys())
    else:
        ids = list(range(len(predictions)))

    for i, key in enumerate(ids):
        pred = predictions[key] if isinstance(predictions, dict) else predictions[i]
        ref = references[key] if isinstance(references, dict) else references[i]
        if isinstance(ref, dict):
            if ref.get("no_answer", False) or not ref.get("answers"):
                gold_answers = [""]
                is_no_answer = True
            else:
                gold_answers = list(ref.get("answers", []))
                is_no_answer = False
        else:
            gold_answers = [ref]
            is_no_answer = False
        em = squad_exact_match(pred, gold_answers)
        f1 = squad_f1(pred, gold_answers)
        ems.append(em)
        f1s.append(f1)
        total += 1
        if is_no_answer:
            no_ans.append(em)
        else:
            has_ans_em.append(em)
            has_ans_f1.append(f1)

    def mean(xs: Sequence[float]) -> float:
        return 100.0 * sum(xs) / len(xs) if xs else 0.0

    metrics = {
        "exact": mean(ems),
        "f1": mean(f1s),
        "total": float(total),
        "HasAns_exact": mean(has_ans_em),
        "HasAns_f1": mean(has_ans_f1),
        "NoAns_exact": mean(no_ans),
        "NoAns_f1": mean(no_ans),
    }
    if no_answer_probs:
        best_em, best_f1, best_thresh = 0.0, 0.0, 0.0
        for thresh_i in range(0, 1000, 4):
            thresh = thresh_i / 1000.0
            ems_t, f1s_t = [], []
            for i, key in enumerate(ids):
                pred = predictions[key] if isinstance(predictions, dict) else predictions[i]
                ref = references[key] if isinstance(references, dict) else references[i]
                gold = list(ref.get("answers", [])) if isinstance(ref, dict) else [ref]
                if no_answer_probs.get(key, 0.0) >= thresh:
                    pred = ""
                ems_t.append(squad_exact_match(pred, gold))
                f1s_t.append(squad_f1(pred, gold))
            if mean(ems_t) + mean(f1s_t) > best_em + best_f1:
                best_em, best_f1, best_thresh = mean(ems_t), mean(f1s_t), thresh
        metrics["best_exact"] = best_em
        metrics["best_f1"] = best_f1
        metrics["best_exact_thresh"] = best_thresh
        metrics["best_f1_thresh"] = best_thresh
    return metrics


# ---------------------------------------------------------------------------
# CNN/DM ROUGE
# ---------------------------------------------------------------------------

def _tokenize_for_rouge(text: str) -> List[str]:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t]


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _f1(overlap: int, pred_total: int, ref_total: int) -> float:
    if overlap == 0 or pred_total == 0 or ref_total == 0:
        return 0.0
    precision = overlap / pred_total
    recall = overlap / ref_total
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _rouge_n_f1(prediction: str, reference: str, n: int) -> float:
    pred_tokens = _tokenize_for_rouge(prediction)
    ref_tokens = _tokenize_for_rouge(reference)
    pred_ngrams = _ngrams(pred_tokens, n)
    ref_ngrams = _ngrams(ref_tokens, n)
    overlap = sum((pred_ngrams & ref_ngrams).values())
    return 100.0 * _f1(overlap, sum(pred_ngrams.values()), sum(ref_ngrams.values()))


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def _rouge_l_f1(prediction: str, reference: str) -> float:
    pred_tokens = _tokenize_for_rouge(prediction)
    ref_tokens = _tokenize_for_rouge(reference)
    lcs = _lcs_length(pred_tokens, ref_tokens)
    return 100.0 * _f1(lcs, len(pred_tokens), len(ref_tokens))


def compute_rouge(
    predictions: Sequence[str],
    references: Sequence[Any],
    use_stemmer: bool = True,
) -> Dict[str, float]:
    """ROUGE-1/2/L F1 x100, preferring the official ``rouge_score`` package."""
    try:  # pragma: no cover - optional dependency
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=use_stemmer)
        totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        n = 0
        for pred, ref in zip(predictions, references):
            refs = ref if isinstance(ref, (list, tuple)) else [ref]
            best = None
            for r in refs:
                scores = scorer.score(str(r), str(pred))
                if best is None:
                    best = scores
                else:
                    best = {k: max(best[k].fmeasure, scores[k].fmeasure) and best[k] for k in best}
            if best is None:
                continue
            for k in totals:
                totals[k] += float(best[k].fmeasure) * 100.0
            n += 1
        return {k: (v / n if n else 0.0) for k, v in totals.items()}
    except Exception:
        pass

    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    n = len(predictions)
    if n == 0:
        return totals
    for pred, ref in zip(predictions, references):
        refs = ref if isinstance(ref, (list, tuple)) else [ref]
        r1 = max((_rouge_n_f1(pred, r, 1) for r in refs), default=0.0)
        r2 = max((_rouge_n_f1(pred, r, 2) for r in refs), default=0.0)
        rl = max((_rouge_l_f1(pred, r) for r in refs), default=0.0)
        totals["rouge1"] += r1
        totals["rouge2"] += r2
        totals["rougeL"] += rl
    return {k: v / n for k, v in totals.items()}


def compute_cnndm_metrics(
    predictions: Sequence[str],
    references: Sequence[Any],
    use_stemmer: bool = True,
) -> Dict[str, float]:
    """CNN/DM ROUGE bundle (delegates to :mod:`apt.data.cnndm` when available)."""
    try:  # pragma: no cover - prefer the canonical implementation
        from apt.data.cnndm import compute_cnndm_metrics as _rouge

        return _rouge(predictions, references, use_stemmer=use_stemmer)
    except Exception:
        return compute_rouge(predictions, references, use_stemmer=use_stemmer)


# ---------------------------------------------------------------------------
# Unified dispatch
# ---------------------------------------------------------------------------

def compute_metrics(task: str, predictions: Any, references: Any, **kwargs) -> Dict[str, float]:
    """Dispatch to the task-specific metric implementation."""
    key = normalize_task_name(task)
    if is_squad_task(key):
        return compute_squad_metrics(predictions, references, **kwargs)
    if key in ("cnndm", "cnn_dailymail", "cnn_dm") or key in SEQ2SEQ_TASKS:
        return compute_cnndm_metrics(predictions, references, **kwargs)
    return compute_glue_metrics(key, predictions, references, **kwargs)


def primary_metric(task: str, metrics: Dict[str, float]) -> float:
    """Pick the paper-reported metric value out of a metrics dict."""
    key = normalize_task_name(task)
    name = primary_metric_name(key)
    if name in metrics:
        return float(metrics[name])
    for fallback in ("primary", "accuracy", "f1", "exact", "rougeL", "rouge_l"):
        if fallback in metrics:
            return float(metrics[fallback])
    return 0.0


def metric_for_display(task: str, metrics: Dict[str, float]) -> str:
    """Human-readable metric string used in tables (e.g. ``42.1/20.3/39.4``)."""
    key = normalize_task_name(task)
    if key in ("cnndm", "cnn_dailymail", "cnn_dm") or key in SEQ2SEQ_TASKS:
        ordered = []
        for k in ROUGE_KEYS:
            value = metrics.get(k, metrics.get(k.lower(), metrics.get(k.upper(), 0.0)))
            ordered.append(f"{float(value):.1f}")
        return "/".join(ordered)
    return f"{primary_metric(key, metrics):.1f}"


# ---------------------------------------------------------------------------
# Paper reference values + relative-quality helpers
# ---------------------------------------------------------------------------

@dataclass
class ReferenceScores:
    """Fully fine-tuned (FT) reference scores used to normalise APT results."""

    task: str
    value: float
    higher_is_better: bool = True

    def relative(self, score: float) -> float:
        """Score as a percentage of the FT reference (paper's relative accuracy)."""
        if self.value == 0:
            return 0.0
        return 100.0 * float(score) / float(self.value)

    def reached(self, score: float, fraction: float = 0.97) -> bool:
        """Whether ``score`` reaches ``fraction`` x FT (used for 97% TTA)."""
        target = fraction * self.value
        return score >= target if self.higher_is_better else score <= target


#: Table 2 FT references (RoBERTa-base / T5-base).
FT_REFERENCES: Dict[str, float] = {
    "roberta-base:mnli": 87.6,
    "roberta-base:sst2": 94.8,
    "roberta-base:squad_v2": 82.9,
    "t5-base:mnli": 87.1,
    "t5-base:sst2": 95.2,
    "t5-base:cnndm": 42.1,
}

#: Table 11 raw efficiency numbers (RoBERTa FT / APT, T5 FT / APT).
RAW_EFFICIENCY: Dict[str, Dict[str, float]] = {
    "roberta-base:ft": {"tta_seconds": 127.0, "train_mem_mb": 2696.0, "inf_time_ms": 220.8, "inf_mem_mb": 1157.0},
    "roberta-base:apt": {"tta_seconds": 752.0, "train_mem_mb": 1890.0, "inf_time_ms": 91.3, "inf_mem_mb": 904.0},
    "t5-base:ft": {"tta_seconds": 366.0, "train_mem_mb": 7217.0, "inf_time_ms": 248.1, "inf_mem_mb": 2347.0},
    "t5-base:apt": {"tta_seconds": 1774.0, "train_mem_mb": 5332.0, "inf_time_ms": 185.0, "inf_mem_mb": 1913.0},
}


def relative_accuracy(score: float, reference: float) -> float:
    """Relative accuracy in % (Section 5.5 reports the SST2/MNLI average)."""
    if reference == 0:
        return 0.0
    return 100.0 * float(score) / float(reference)


def relative_accuracy_sst2_mnli(
    sst2: float,
    mnli: float,
    sst2_ft: float = 94.8,
    mnli_ft: float = 87.6,
) -> float:
    """Average of SST2 and MNLI relative accuracy w.r.t. FT (Section 5.5)."""
    return 0.5 * (relative_accuracy(sst2, sst2_ft) + relative_accuracy(mnli, mnli_ft))


def format_percent(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}%"


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> bool:  # pragma: no cover - manual sanity check
    assert abs(accuracy([1, 0, 1, 1], [1, 0, 0, 1]) - 75.0) < 1e-6
    assert abs(binary_f1([1, 0, 1], [1, 0, 0]) - 80.0) < 1e-6
    assert abs(squad_f1("the cat sat", "cat sat") - 100.0 * 2 * (2 / 3) * 1.0 / (2 / 3 + 1.0)) < 0.5
    assert abs(squad_exact_match("The cat.", "cat") - 1.0) < 1e-6
    assert compute_rouge(["the cat sat on the mat"], ["the cat sat on the mat"])["rouge1"] > 99.0
    assert abs(glue_average({"sst2": 94.5, "mnli": 86.4}) - 90.45) < 1e-6
    assert abs(relative_accuracy_sst2_mnli(94.5, 86.4) - 0.5 * (94.5 / 94.8 * 100 + 86.4 / 87.6 * 100)) < 1e-6
    assert primary_metric_name("squad_v2") == "f1"
    assert metric_for_display("cnndm", {"rouge1": 42.1, "rouge2": 20.3, "rougeL": 39.4}) == "42.1/20.3/39.4"
    print("apt.eval.metrics self-test passed")
    return True


if __name__ == "__main__":  # pragma: no cover
    _self_test()
