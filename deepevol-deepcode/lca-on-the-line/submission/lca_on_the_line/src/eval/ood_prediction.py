"""Out-of-distribution performance prediction and baselines (paper Section 4.2, Table 3, Section F.2).

The paper estimates a model's OOD performance with a linear function derived from its
*in-distribution* LCA distance and compares against four competitive baselines:

1. ``id_top1``  - "Accuracy on the Line" (Miller et al., 2021): in-distribution Top-1
   accuracy, with a probit transform (as done in Miller et al., 2021 and Baek et al., 2022).
2. ``ac``       - Average Confidence (Hendrycks & Gimpel, 2017): the OOD logits are
   temperature-scaled (temperature fitted on the labelled ID validation split) and the
   mean maximum softmax probability is used as predicted OOD accuracy.
3. ``aline_d``  - Agreement-on-the-Line with *different depth* model pairs (Baek et al., 2022).
4. ``aline_s``  - Agreement-on-the-Line with *different size/width* model pairs (Baek et al., 2022).
5. ``id_lca``   - (Ours) in-distribution LCA distance.

Because LCA does not fall inside ``[0, 1]`` the paper uses **min-max scaling** instead of the
probit transform for the LCA predictor; accuracy-based baselines keep the probit transform.
All reported errors are the mean absolute error (MAE) between the predicted and the true OOD
Top-1 accuracy of every model, matching Table 3 of the paper.

Notes
-----
* AC and both Aline baselines consume *unlabelled* OOD data (logits/agreements); ID Top-1 and
  ID LCA only need the in-distribution measurements.
* The Aline implementations mirror the ones referenced in the paper's Addendum, copied from
  ``https://github.com/kebaek/Agreement-on-the-line/blob/main/agreement_trajectory.ipynb``:
  the soft agreement between a pair of models is the average over samples/classes of
  ``p1 * p2 + (1 - p1) * (1 - p2)`` and the hard agreement is the fraction of samples on which
  the two models' arg-max predictions coincide.  A linear relation ``accuracy ~ agreement`` is
  fitted on in-distribution data and then applied to the OOD agreements.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

LOG = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Optional reuse of src/metrics/correlation.py (metric definitions live there; local
# fallbacks below keep this module usable on its own).
# --------------------------------------------------------------------------------------
try:  # pragma: no cover - import bookkeeping only
    from ..metrics.correlation import (  # type: ignore
        mean_absolute_error as _corr_mae,
        pearson_correlation as _corr_pea,
        probit as _corr_probit,
        inverse_probit as _corr_inv_probit,
        r2_score as _corr_r2,
    )

    _HAS_CORRELATION = True
except Exception:  # pragma: no cover
    try:
        from metrics.correlation import (  # type: ignore
            mean_absolute_error as _corr_mae,
            pearson_correlation as _corr_pea,
            probit as _corr_probit,
            inverse_probit as _corr_inv_probit,
            r2_score as _corr_r2,
        )

        _HAS_CORRELATION = True
    except Exception:
        _HAS_CORRELATION = False
        _corr_mae = _corr_pea = _corr_probit = _corr_inv_probit = _corr_r2 = None


# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
DEFAULT_OOD_DATASETS: Tuple[str, ...] = ("v2", "s", "r", "a", "objectnet")

#: Paper Table 3: MAE of each predictor, per OOD dataset (bold/underline values).
TABLE3_REFERENCE: Dict[str, Dict[str, float]] = {
    "id_top1": {"v2": 0.040, "s": 0.230, "r": 0.277, "a": 0.192, "objectnet": 0.178},
    "ac": {"v2": 0.043, "s": 0.124, "r": 0.113, "a": 0.324, "objectnet": 0.127},
    "aline_d": {"v2": 0.121, "s": 0.270, "r": 0.167, "a": 0.409, "objectnet": 0.265},
    "aline_s": {"v2": 0.072, "s": 0.143, "r": 0.201, "a": 0.165, "objectnet": 0.131},
    "id_lca": {"v2": 0.162, "s": 0.093, "r": 0.114, "a": 0.103, "objectnet": 0.048},
}

#: Human readable method names (Table 3 row labels).
METHOD_LABELS: Dict[str, str] = {
    "id_top1": "ID Top1 (Miller et al., 2021)",
    "ac": "AC (Hendrycks & Gimpel, 2017)",
    "aline_d": "Aline-D (Baek et al., 2022)",
    "aline_s": "Aline-S (Baek et al., 2022)",
    "id_lca": "(Ours) ID LCA",
}

DEFAULT_METHODS: Tuple[str, ...] = ("id_top1", "ac", "aline_d", "aline_s", "id_lca")

#: Datasets that require OOD logits (AC / Aline) and those that only need ID measurements.
ID_ONLY_METHODS: Tuple[str, ...] = ("id_top1", "id_lca")
OOD_LOGIT_METHODS: Tuple[str, ...] = ("ac", "aline_d", "aline_s")

ID_DATASET_ALIASES: Tuple[str, ...] = ("id", "imagenet", "imagenet-1k", "imagenet1k", "val", "validation")

#: Agreement flavour used by each Aline variant.  Pairs sharing one architecture family and
#: therefore comparable arg-max labels use the hard agreement; pairs that differ in
#: size/width use the (more stable) soft agreement.
ALINE_AGREEMENT: Dict[str, str] = {"aline_d": "hard", "aline_s": "soft"}

DEFAULT_TEMPERATURE_BOUNDS: Tuple[float, float] = (1e-2, 1e2)
DEFAULT_SUBSAMPLE: int = 20000

#: Explicit model-pair groups for the two Aline flavours (names follow the VM/VLM zoos).
DEPTH_PAIR_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152"),
    ("vgg11", "vgg13", "vgg16", "vgg19"),
    ("densenet121", "densenet161", "densenet169", "densenet201"),
    ("mnasnet0_5", "mnasnet0_75", "mnasnet1_0", "mnasnet1_3"),
    ("regnet_y_400mf", "regnet_y_800mf", "regnet_y_1_6gf"),
    ("clip_rn50", "clip_rn101"),
    ("clip_vit_b_32", "clip_vit_l_14"),
    ("clip_vit_b_16", "clip_vit_l_14"),
    ("convnext_tiny", "convnext_small", "convnext_base"),
    ("swin_t", "swin_s", "swin_b"),
)

WIDTH_PAIR_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("squeezenet1_0", "squeezenet1_1"),
    ("mobilenet_v3_small", "mobilenet_v3_large"),
    ("shufflenet_v2_x1_0", "shufflenet_v2_x2_0"),
    ("resnet50", "wide_resnet50_2"),
    ("resnet101", "wide_resnet101_2"),
    ("clip_rn50", "clip_rn50x4"),
    ("clip_vit_b_32", "clip_vit_b_16"),
    ("convnext_tiny", "convnext_small"),
    ("swin_t", "swin_b"),
    ("efficientnet_b0", "efficientnet_b4"),
)

_SUBSET_ALIASES: Dict[str, str] = {
    "all": "all",
    "vm": "vm",
    "vms": "vm",
    "vision": "vm",
    "vlm": "vlm",
    "vlms": "vlm",
    "vision-language": "vlm",
}


# --------------------------------------------------------------------------------------
# Small numeric helpers (numpy only, so the module never hard-depends on scipy/sklearn)
# --------------------------------------------------------------------------------------
def _as_float_array(x: Any) -> np.ndarray:
    """Convert lists / numpy arrays / torch tensors to a float64 numpy array."""
    if hasattr(x, "detach"):  # torch tensor
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def mean_absolute_error(y_true: Any, y_pred: Any) -> float:
    """Mean absolute error (delegates to ``metrics.correlation`` when available)."""
    y_true = _as_float_array(y_true).ravel()
    y_pred = _as_float_array(y_pred).ravel()
    if _HAS_CORRELATION and _corr_mae is not None:
        try:
            return float(_corr_mae(y_true, y_pred))
        except Exception:  # pragma: no cover - defensive
            pass
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))


def probit(x: Any, eps: float = 1e-6) -> np.ndarray:
    """Probit (inverse normal CDF) transform used for accuracy baselines."""
    if _HAS_CORRELATION and _corr_probit is not None:
        try:
            return _as_float_array(_corr_probit(_as_float_array(x), eps=eps))
        except Exception:  # pragma: no cover
            pass
    x = np.clip(_as_float_array(x), eps, 1.0 - eps)
    # Acklam's rational approximation of the inverse normal CDF.
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1.0 - 0.02425
    out = np.zeros_like(x)
    lo = x < plow
    hi = x > phigh
    mid = ~(lo | hi)
    if np.any(lo):
        q = np.sqrt(-2.0 * np.log(x[lo]))
        out[lo] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if np.any(hi):
        q = np.sqrt(-2.0 * np.log(1.0 - x[hi]))
        out[hi] = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if np.any(mid):
        q = x[mid] - 0.5
        r = q * q
        out[mid] = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        )
    return out


def inverse_probit(z: Any) -> np.ndarray:
    """Inverse of :func:`probit` (the standard normal CDF)."""
    if _HAS_CORRELATION and _corr_inv_probit is not None:
        try:
            return _as_float_array(_corr_inv_probit(_as_float_array(z)))
        except Exception:  # pragma: no cover
            pass
    z = _as_float_array(z)
    return 0.5 * (1.0 + np.vectorize(math.erf, otypes=[np.float64])(z / math.sqrt(2.0)))


def min_max_scale(x: Any, feature_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """Min-max scaling used for the LCA predictor (LCA is not in ``[0, 1]``)."""
    x = _as_float_array(x)
    lo, hi = float(np.min(x)), float(np.max(x))
    a, b = feature_range
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo <= 1e-12:
        return np.full_like(x, a)
    return (x - lo) / (hi - lo) * (b - a) + a


def pearson_correlation(x: Any, y: Any, abs_values: bool = True) -> float:
    if _HAS_CORRELATION and _corr_pea is not None:
        try:
            return float(_corr_pea(_as_float_array(x), _as_float_array(y), abs_values=abs_values))
        except Exception:  # pragma: no cover
            pass
    x, y = _as_float_array(x).ravel(), _as_float_array(y).ravel()
    if x.size < 2:
        return float("nan")
    xc, yc = x - x.mean(), y - y.mean()
    denom = float(np.sqrt(np.sum(xc ** 2) * np.sum(yc ** 2)))
    if denom <= 1e-12:
        return float("nan")
    r = float(np.sum(xc * yc) / denom)
    return abs(r) if abs_values else r


def r2_score(y_true: Any, y_pred: Any) -> float:
    if _HAS_CORRELATION and _corr_r2 is not None:
        try:
            return float(_corr_r2(_as_float_array(y_true), _as_float_array(y_pred)))
        except Exception:  # pragma: no cover
            pass
    y_true, y_pred = _as_float_array(y_true).ravel(), _as_float_array(y_pred).ravel()
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


# --------------------------------------------------------------------------------------
# Softmax / temperature scaling / average confidence
# --------------------------------------------------------------------------------------
def softmax(logits: Any, temperature: float = 1.0, axis: int = -1) -> np.ndarray:
    """Numerically stable temperature-scaled softmax."""
    logits = _as_float_array(logits)
    temperature = float(temperature) if temperature else 1.0
    z = logits / max(temperature, 1e-8)
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def _nll(logits: np.ndarray, targets: np.ndarray, temperature: float) -> float:
    probs = softmax(logits, temperature=temperature)
    rows = np.arange(targets.shape[0])
    p = np.clip(probs[rows, targets.astype(np.int64)], 1e-12, 1.0)
    return float(-np.mean(np.log(p)))


@dataclass
class TemperatureScaler:
    """Temperature scaling fitted by minimising the ID validation NLL."""

    temperature: float = 1.0
    fitted: bool = False
    nll: float = float("nan")
    num_samples: int = 0

    def predict_proba(self, logits: Any) -> np.ndarray:
        return softmax(logits, temperature=self.temperature)

    def average_confidence(self, logits: Any) -> np.ndarray:
        """Per-sample maximum softmax probability."""
        return np.max(self.predict_proba(logits), axis=-1)

    def predicted_accuracy(self, logits: Any) -> float:
        """Average confidence over the dataset == predicted Top-1 accuracy."""
        conf = self.average_confidence(logits)
        return float(np.mean(conf)) if conf.size else float("nan")

    def to_dict(self) -> Dict[str, Any]:
        return {"temperature": self.temperature, "fitted": self.fitted,
                "nll": self.nll, "num_samples": self.num_samples}


def fit_temperature(
    logits: Any,
    targets: Any,
    bounds: Tuple[float, float] = DEFAULT_TEMPERATURE_BOUNDS,
    subsample: Optional[int] = DEFAULT_SUBSAMPLE,
    seed: int = 0,
) -> TemperatureScaler:
    """Fit a single temperature on labelled in-distribution logits (Hendrycks & Gimpel)."""
    logits = _as_float_array(logits)
    targets = _as_float_array(targets).astype(np.int64).ravel()
    n = min(logits.shape[0], targets.shape[0])
    logits, targets = logits[:n], targets[:n]
    if subsample and n > subsample:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=int(subsample), replace=False))
        logits, targets = logits[idx], targets[idx]
    if logits.size == 0:
        LOG.warning("Cannot fit temperature on an empty logit array; using T=1.0")
        return TemperatureScaler(temperature=1.0, fitted=False)

    lo, hi = math.log(bounds[0]), math.log(bounds[1])

    def objective(log_t: float) -> float:
        return _nll(logits, targets, math.exp(float(log_t)))

    best_log_t = 0.0
    try:  # scipy fast path
        from scipy.optimize import minimize_scalar  # type: ignore

        res = minimize_scalar(objective, bounds=(lo, hi), method="bounded")
        best_log_t = float(res.x)
    except Exception:
        # Golden-section search (deterministic, dependency free).
        gr = (math.sqrt(5.0) - 1.0) / 2.0
        a, b = lo, hi
        c, d = b - gr * (b - a), a + gr * (b - a)
        fc, fd = objective(c), objective(d)
        for _ in range(80):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - gr * (b - a)
                fc = objective(c)
            else:
                a, c, fc = c, d, fd
                d = a + gr * (b - a)
                fd = objective(d)
            if abs(b - a) < 1e-5:
                break
        best_log_t = 0.5 * (a + b)

    temperature = float(math.exp(best_log_t))
    return TemperatureScaler(
        temperature=temperature,
        fitted=True,
        nll=objective(best_log_t),
        num_samples=int(logits.shape[0]),
    )


def average_confidence_predictions(
    logits: Any,
    temperature: float = 1.0,
    calibrator: Optional[TemperatureScaler] = None,
) -> Tuple[float, np.ndarray]:
    """Predicted accuracy (mean max-prob) and the per-sample confidences."""
    scaler = calibrator if calibrator is not None else TemperatureScaler(temperature=temperature)
    conf = scaler.average_confidence(logits)
    return (float(np.mean(conf)) if conf.size else float("nan")), conf


# --------------------------------------------------------------------------------------
# Agreement-on-the-Line (Aline-D / Aline-S)
# --------------------------------------------------------------------------------------
def soft_agreement(probs_a: Any, probs_b: Any) -> float:
    """Soft agreement between two probabilistic predictions.

    ``mean_{n,k} [ p_a(k) * p_b(k) + (1 - p_a(k)) * (1 - p_b(k)) ]``
    (the formulation used in the Agreement-on-the-Line notebook), i.e. the agreement of the
    independent binary decisions "is class k the label?".
    """
    a, b = _as_float_array(probs_a), _as_float_array(probs_b)
    if a.size == 0 or a.shape != b.shape:
        return float("nan")
    per_sample = np.mean(a * b + (1.0 - a) * (1.0 - b), axis=-1)
    return float(np.mean(per_sample))


def hard_agreement(probs_a: Any, probs_b: Any) -> float:
    """Fraction of samples on which the two models' arg-max predictions coincide."""
    a, b = _as_float_array(probs_a), _as_float_array(probs_b)
    if a.size == 0 or a.shape != b.shape:
        return float("nan")
    return float(np.mean(np.argmax(a, axis=-1) == np.argmax(b, axis=-1)))


def pairwise_agreement(probs_a: Any, probs_b: Any, mode: str = "soft") -> float:
    mode = (mode or "soft").lower()
    if mode in ("hard", "argmax"):
        return hard_agreement(probs_a, probs_b)
    return soft_agreement(probs_a, probs_b)


def agreement_matrix(
    probs_by_model: Mapping[str, np.ndarray],
    pairs: Sequence[Tuple[str, str]],
    mode: str = "soft",
) -> Dict[Tuple[str, str], float]:
    """Agreement for every requested pair (pairs with missing models are skipped)."""
    out: Dict[Tuple[str, str], float] = {}
    for m1, m2 in pairs:
        if m1 not in probs_by_model or m2 not in probs_by_model:
            continue
        value = pairwise_agreement(probs_by_model[m1], probs_by_model[m2], mode=mode)
        if np.isfinite(value):
            out[(m1, m2)] = value
    return out


# ---------------------------------------------------------------- model pair construction
def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


def _all_combinations(group: Sequence[str]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            pairs.append((group[i], group[j]))
    return pairs


def build_model_pairs(
    names: Optional[Sequence[str]] = None,
    pair_type: str = "depth",
    groups: Optional[Sequence[Sequence[str]]] = None,
    max_pairs: Optional[int] = None,
) -> List[Tuple[str, str]]:
    """Construct model pairs for Aline-D (``pair_type="depth"``) / Aline-S (``"size"``).

    Explicit groups take precedence; when none of them matches the available model names a
    name-based heuristic groups models by their architecture stem and pairs all members with
    a different variant suffix.
    """
    pair_type = (pair_type or "depth").lower()
    if groups is None:
        groups = DEPTH_PAIR_GROUPS if pair_type.startswith("depth") else WIDTH_PAIR_GROUPS

    available = {_normalize_name(n): n for n in names} if names is not None else None

    def _keep(name: str) -> bool:
        return available is None or _normalize_name(name) in available

    def _resolve(name: str) -> str:
        if available is None:
            return name
        return available.get(_normalize_name(name), name)

    pairs: List[Tuple[str, str]] = []
    for group in groups:
        members = [_resolve(g) for g in group if _keep(g)]
        if len(members) > 1:
            pairs.extend(_all_combinations(members))

    if not pairs and names is not None:
        # Heuristic fallback: group by the non-numeric stem of the model name.
        import re

        buckets: Dict[str, List[str]] = {}
        for name in names:
            stem = re.sub(r"[0-9].*$", "", _normalize_name(name)) or _normalize_name(name)
            buckets.setdefault(stem, []).append(name)
        group_list = [members for members in buckets.values() if len(members) > 1]
        for group in group_list:
            pairs.extend(_all_combinations(group))
        if not pairs:
            # Last resort: pair the first model with every other model.
            ordered = list(names)
            if len(ordered) > 1:
                pairs = [(ordered[0], m) for m in ordered[1:]]

    # Deduplicate while preserving order.
    seen, unique = set(), []
    for p in pairs:
        key = tuple(sorted(p))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    if max_pairs is not None:
        unique = unique[: int(max_pairs)]
    return unique


def _fit_linear(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Ordinary least squares ``y ~ slope * x + intercept`` (numpy only)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.size < 2 or np.allclose(x, x[0]):
        return 0.0, float(np.mean(y)) if y.size else float("nan")
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope), float(intercept)


def aline_predictions(
    probs_id: Mapping[str, np.ndarray],
    id_accuracies: Mapping[str, float],
    probs_ood: Mapping[str, np.ndarray],
    pairs: Sequence[Tuple[str, str]],
    mode: str = "soft",
    return_fit: bool = False,
) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, float]]]:
    """Agreement-on-the-Line prediction of every model's OOD accuracy.

    The linear relation ``accuracy ~ agreement`` is fitted on the *in-distribution* data using
    the average accuracy of each pair, then applied to the OOD agreements.  The prediction of a
    single model is the mean over all pairs in which it participates.
    """
    id_agree = agreement_matrix(probs_id, pairs, mode=mode)
    usable = [p for p in pairs if p in id_agree and p[0] in id_accuracies and p[1] in id_accuracies]
    if len(usable) < 2:
        LOG.warning("Aline: not enough usable pairs (%d); returning empty predictions", len(usable))
        return ({}, {"slope": float("nan"), "intercept": float("nan"), "n_pairs": len(usable)})
    x = np.array([id_agree[p] for p in usable], dtype=np.float64)
    y = np.array([0.5 * (id_accuracies[p[0]] + id_accuracies[p[1]]) for p in usable], dtype=np.float64)
    slope, intercept = _fit_linear(x, y)
    fit = {"slope": slope, "intercept": intercept, "n_pairs": len(usable),
           "pea_id": pearson_correlation(x, y)}

    ood_agree = agreement_matrix(probs_ood, usable, mode=mode)
    if not ood_agree:
        return ({}, fit)
    per_model: Dict[str, List[float]] = {}
    for (m1, m2), agr in ood_agree.items():
        pred = slope * agr + intercept
        per_model.setdefault(m1, []).append(pred)
        per_model.setdefault(m2, []).append(pred)
    predictions = {m: float(np.mean(v)) for m, v in per_model.items() if v}
    return (predictions, fit) if return_fit else predictions


# --------------------------------------------------------------------------------------
# In-distribution linear predictors (ID LCA with min-max scaling, ID Top-1 with probit)
# --------------------------------------------------------------------------------------
def predict_linear(
    x: Any,
    y: Any,
    scaler: str = "minmax",
    target_transform: str = "identity",
    leave_one_out: bool = False,
) -> Dict[str, Any]:
    """Fit ``y ~ f(x)`` and return predictions for the same models.

    ``scaler="minmax"`` is used for LCA (which is unbounded), ``target_transform="probit"``
    for accuracy-based baselines.  With ``leave_one_out=True`` every model is predicted from a
    fit that excludes it, which gives an honest estimate of the predictor's error.
    """
    x = _as_float_array(x).ravel()
    y = _as_float_array(y).ravel()
    scaler = (scaler or "none").lower()
    target_transform = (target_transform or "identity").lower()

    def _forward_target(v: np.ndarray) -> np.ndarray:
        if target_transform in ("probit", "logit", "norm"):
            return probit(v)
        if target_transform == "log":
            return np.log(np.clip(v, 1e-8, None))
        return v

    def _inverse_target(v: np.ndarray) -> np.ndarray:
        if target_transform in ("probit", "logit", "norm"):
            return inverse_probit(v)
        if target_transform == "log":
            return np.exp(v)
        return v

    def _transform_x(v: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
        if scaler == "minmax":
            if x_max - x_min <= 1e-12:
                return np.zeros_like(v)
            return (v - x_min) / (x_max - x_min)
        if scaler in ("probit", "logit", "norm"):
            return probit(v)
        return v

    x_min, x_max = float(np.min(x)) if x.size else 0.0, float(np.max(x)) if x.size else 1.0
    xs = _transform_x(x, x_min, x_max)
    ys = _forward_target(y)

    if leave_one_out and x.size > 2:
        preds = np.full_like(y, np.nan)
        for i in range(x.size):
            mask = np.ones(x.size, dtype=bool)
            mask[i] = False
            slope_i, intercept_i = _fit_linear(xs[mask], ys[mask])
            preds[i] = slope_i * xs[i] + intercept_i
        slope, intercept = _fit_linear(xs, ys)
        pred_scaled = preds
    else:
        slope, intercept = _fit_linear(xs, ys)
        pred_scaled = slope * xs + intercept

    predictions = _inverse_target(pred_scaled)
    return {
        "predictions": predictions,
        "slope": slope,
        "intercept": intercept,
        "x_min": x_min,
        "x_max": x_max,
        "scaler": scaler,
        "target_transform": target_transform,
        "fit": (slope, intercept),
        "mae": mean_absolute_error(y, predictions),
        "rmse": float(np.sqrt(np.mean((y - predictions) ** 2))) if y.size else float("nan"),
        "r2": r2_score(y, predictions),
        "pea": pearson_correlation(y, predictions),
        "n": int(y.size),
    }


# --------------------------------------------------------------------------------------
# Result containers
# --------------------------------------------------------------------------------------
@dataclass
class OodPredictionResult:
    """One (predictor, OOD dataset) cell of the Table 3 error-prediction matrix."""

    method: str
    dataset: str
    mae: float
    rmse: float = float("nan")
    r2: float = float("nan")
    pea: float = float("nan")
    n: int = 0
    predictions: Dict[str, float] = field(default_factory=dict)
    targets: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def asdict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Keep the payload JSON friendly.
        d["predictions"] = {str(k): float(v) for k, v in self.predictions.items()}
        d["targets"] = {str(k): float(v) for k, v in self.targets.items()}
        return d


@dataclass
class OodPredictionTable:
    """Predictor x dataset table of prediction errors (Table 3 / Table 12)."""

    results: Dict[str, Dict[str, OodPredictionResult]] = field(default_factory=dict)
    datasets: Tuple[str, ...] = DEFAULT_OOD_DATASETS
    methods: Tuple[str, ...] = DEFAULT_METHODS
    subset: str = "all"
    metric: str = "top1"

    # ---------------------------------------------------------------- table views
    def mae_table(self) -> Dict[str, Dict[str, float]]:
        return {m: {d: self.results[m][d].mae for d in self.datasets if d in self.results.get(m, {})}
                for m in self.methods if m in self.results}

    def get(self, method: str, dataset: str) -> Optional[OodPredictionResult]:
        return self.results.get(method, {}).get(dataset)

    def best_method(self, dataset: str) -> Optional[str]:
        cells = [(m, self.results[m][dataset].mae) for m in self.methods
                 if dataset in self.results.get(m, {}) and np.isfinite(self.results[m][dataset].mae)]
        return min(cells, key=lambda kv: kv[1])[0] if cells else None

    def asdict(self) -> Dict[str, Any]:
        return {
            "subset": self.subset,
            "metric": self.metric,
            "datasets": list(self.datasets),
            "methods": list(self.methods),
            "mae": self.mae_table(),
            "cells": {m: {d: r.asdict() for d, r in per.items()} for m, per in self.results.items()},
        }

    def check_against_table3(self, tolerance: float = 0.10) -> List[str]:
        """Compare the produced MAE values with the paper's Table 3."""
        messages: List[str] = []
        for method, per_dataset in TABLE3_REFERENCE.items():
            for dataset, reference in per_dataset.items():
                cell = self.get(method, dataset)
                if cell is None or not np.isfinite(cell.mae):
                    messages.append(f"MISSING {method}/{dataset} (paper {reference:.3f})")
                    continue
                delta = abs(cell.mae - reference)
                flag = "OK " if delta <= tolerance else "OFF"
                messages.append(
                    f"{flag} {method:>8s}/{dataset:<9s} mae={cell.mae:.3f} paper={reference:.3f} "
                    f"|d|={delta:.3f}"
                )
        # The paper's central claim: ID LCA beats ID Top-1 on the four severe shifts.
        for dataset in ("s", "r", "a", "objectnet"):
            ours, baseline = self.get("id_lca", dataset), self.get("id_top1", dataset)
            if ours is None or baseline is None:
                continue
            ok = np.isfinite(ours.mae) and np.isfinite(baseline.mae) and ours.mae < baseline.mae
            messages.append(
                f"{'OK ' if ok else 'FAIL'} ID LCA < ID Top1 on {dataset}: "
                f"{ours.mae:.3f} vs {baseline.mae:.3f}"
            )
        return messages

    def format_table(self, decimals: int = 3) -> str:
        methods = [m for m in self.methods if m in self.results]
        header_methods = [METHOD_LABELS.get(m, m) for m in methods]
        width = max([len(h) for h in header_methods] + [12])
        head = f"{'Methods':<{width}}" + "".join(f"{d:>11s}" for d in self.datasets)
        lines = [head, "-" * len(head)]
        for method, label in zip(methods, header_methods):
            row = f"{label:<{width}}"
            for dataset in self.datasets:
                cell = self.get(method, dataset)
                row += f"{cell.mae:>11.3f}" if cell is not None and np.isfinite(cell.mae) else f"{'-':>11s}"
            lines.append(row)
        return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Score-table normalisation (accepts ModelRecord objects, dicts or DataFrames)
# --------------------------------------------------------------------------------------
def _normalize_records(records: Any) -> Dict[str, Dict[str, Any]]:
    """Coerce the various score-table representations into ``{model: {field: value}}``."""
    out: Dict[str, Dict[str, Any]] = {}
    if records is None:
        return out

    if isinstance(records, Mapping):
        for name, record in records.items():
            if hasattr(record, "asdict"):
                fields = record.asdict()
            elif isinstance(record, Mapping):
                fields = dict(record)
            else:  # object with attributes
                fields = {k: v for k, v in vars(record).items() if not k.startswith("_")}
            fields.setdefault("name", name)
            out[str(name)] = fields
        return out

    if hasattr(records, "to_dict"):  # pandas DataFrame
        try:
            records = records.to_dict("records")
        except Exception:  # pragma: no cover
            records = list(records)

    if isinstance(records, (list, tuple)):
        for record in records:
            if hasattr(record, "asdict"):
                fields = record.asdict()
            elif isinstance(record, Mapping):
                fields = dict(record)
            else:
                fields = {k: v for k, v in vars(record).items() if not k.startswith("_")}
            name = fields.get("name") or fields.get("model")
            if name is None:
                continue
            fields.setdefault("name", name)
            out[str(name)] = fields
    return out


def model_id_metric(fields: Mapping[str, Any], field: str = "id_top1") -> Optional[float]:
    """Extract an ID measurement (``id_top1``/``id_lca``/...) from a normalised record."""
    for key in (field, field.replace("_", ""), field.lower()):
        if key in fields:
            try:
                return float(fields[key])
            except (TypeError, ValueError):
                return None
    return None


def model_ood_metric(
    fields: Mapping[str, Any],
    dataset: str,
    metric: str = "top1",
    record: Any = None,
) -> Optional[float]:
    """Extract an OOD measurement for ``dataset``/``metric`` from a normalised record."""
    ds = str(dataset)
    candidates = [
        f"ood_{metric}_{ds}",
        f"{metric}_{ds}",
        f"{ds}_{metric}",
        f"ood_{ds}_{metric}",
        f"top1_{ds}" if metric == "top1" else f"{metric}_{ds}",
    ]
    for key in candidates:
        if key in fields and fields[key] is not None:
            try:
                value = float(fields[key])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value) or not np.isnan(value):
                return value
    # Nested mapping: fields[dataset][metric]
    nested = fields.get(ds)
    if isinstance(nested, Mapping):
        for key in (metric, f"top1", f"ood_{metric}"):
            if key in nested and nested[key] is not None:
                try:
                    return float(nested[key])
                except (TypeError, ValueError):
                    pass
    # ModelRecord-style accessor: record.get(dataset, metric)
    if record is not None and hasattr(record, "get"):
        try:
            value = record.get(ds, metric)
            if value is not None:
                return float(value)
        except Exception:  # pragma: no cover - defensive
            pass
    # Aliases used by src.data.ood_datasets.display_name (ImgN-S -> s, ...)
    aliases = {"s": ("imagenet_s", "imagens", "sketch", "imagenet_sketch"),
               "r": ("imagenet_r", "imagenetr", "rendition"),
               "a": ("imagenet_a", "imageneta", "adversarial"),
               "v2": ("imagenet_v2", "imagenetv2"),
               "objectnet": ("object_net", "objnet")}
    for alias in aliases.get(ds, ()):  # pragma: no cover - convenience
        for key in (f"ood_{metric}_{alias}", f"{metric}_{alias}"):
            if key in fields and fields[key] is not None:
                try:
                    return float(fields[key])
                except (TypeError, ValueError):
                    pass
    return None


def _family_of(fields: Mapping[str, Any]) -> str:
    family = fields.get("family") or fields.get("modality") or ""
    family = str(family).lower()
    if "vlm" in family or "clip" in family or "language" in family:
        return "vlm"
    if "vm" in family or family in ("cnn", "vit", "vision"):
        return "vm"
    name = str(fields.get("name", "")).lower()
    if name.startswith("clip") or "blip" in name or "albef" in name:
        return "vlm"
    return family or "unknown"


def subset_model_names(records: Mapping[str, Dict[str, Any]], subset: str = "all") -> List[str]:
    """Filter model names to ``all``/``vm``/``vlm`` (Tables 11 and 12 of the paper)."""
    subset = _SUBSET_ALIASES.get(str(subset).lower(), "all")
    if subset == "all":
        return list(records)
    return [name for name, fields in records.items() if _family_of(fields) == subset]


# --------------------------------------------------------------------------------------
# Cached logits access (produced by src/eval/evaluate_models.py)
# --------------------------------------------------------------------------------------
def _candidate_cache_paths(cache_dir: str, model_name: str, dataset_name: str) -> List[str]:
    names = [f"{model_name}__{dataset_name}.npz", f"{model_name}_{dataset_name}.npz",
             f"{model_name}-{dataset_name}.npz"]
    subdirs = ("outputs", "", "features", "logits", "cache")
    paths: List[str] = []
    for sub in subdirs:
        base = os.path.join(cache_dir, sub) if sub else cache_dir
        for name in names:
            paths.append(os.path.join(base, name))
        paths.append(os.path.join(base, model_name, f"{dataset_name}.npz"))
    return paths


def resolve_cached_output(cache_dir: str, model_name: str, dataset_name: str) -> Optional[str]:
    """Locate the cached ``.npz`` with logits/features/targets for a model+dataset pair."""
    if not cache_dir or not os.path.isdir(cache_dir):
        return None
    for path in _candidate_cache_paths(cache_dir, model_name, dataset_name):
        if os.path.isfile(path):
            return path
    return None


def load_cached_output(path: str) -> Dict[str, np.ndarray]:
    """Load a cached ``.npz`` (keys ``logits``/``features``/``targets``)."""
    out: Dict[str, np.ndarray] = {}
    try:
        with np.load(path, allow_pickle=True) as data:
            for key in data.files:
                try:
                    out[key] = np.asarray(data[key])
                except Exception:  # pragma: no cover - defensive
                    continue
    except Exception as exc:  # pragma: no cover - corrupted cache
        LOG.warning("Failed to load cached outputs %s (%s)", path, exc)
    return out


def _lookup_logits(
    source: Optional[Any],
    model_name: str,
    dataset_name: str,
    cache_dir: Optional[str] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """Fetch cached outputs from an in-memory mapping, a directory or a file path."""
    if source is not None:
        if isinstance(source, Mapping):
            for key in ((model_name, dataset_name), f"{model_name}__{dataset_name}",
                        f"{model_name}/{dataset_name}"):
                if key in source:
                    value = source[key]
                    return {"logits": np.asarray(value)} if not isinstance(value, Mapping) else dict(value)
            nested = source.get(model_name)
            if isinstance(nested, Mapping) and dataset_name in nested:
                value = nested[dataset_name]
                return {"logits": np.asarray(value)} if not isinstance(value, Mapping) else dict(value)
        elif isinstance(source, str):
            if os.path.isdir(source):
                path = resolve_cached_output(source, model_name, dataset_name)
                return load_cached_output(path) if path else None
            if os.path.isfile(source):
                return load_cached_output(source)
    if cache_dir:
        path = resolve_cached_output(cache_dir, model_name, dataset_name)
        if path:
            return load_cached_output(path)
    return None


def _find_id_logits(
    source: Optional[Any],
    model_name: str,
    cache_dir: Optional[str] = None,
) -> Optional[Dict[str, np.ndarray]]:
    for alias in ID_DATASET_ALIASES:
        found = _lookup_logits(source, model_name, alias, cache_dir=cache_dir)
        if found and "logits" in found:
            return found
    return None


# --------------------------------------------------------------------------------------
# Main evaluation entry point
# --------------------------------------------------------------------------------------
def evaluate_ood_prediction(
    records: Any,
    datasets: Sequence[str] = DEFAULT_OOD_DATASETS,
    methods: Sequence[str] = DEFAULT_METHODS,
    metric: str = "top1",
    subset: str = "all",
    cache_dir: Optional[str] = None,
    logits: Optional[Any] = None,
    pairs_d: Optional[Sequence[Tuple[str, str]]] = None,
    pairs_s: Optional[Sequence[Tuple[str, str]]] = None,
    agreement_d: str = "hard",
    agreement_s: str = "soft",
    temperature: Optional[float] = None,
    calibrate_temperature: bool = True,
    leave_one_out: bool = False,
    max_logit_samples: Optional[int] = DEFAULT_SUBSAMPLE,
    verbose: bool = True,
) -> OodPredictionTable:
    """Predict OOD Top-1 accuracy with ID LCA and four baselines (paper Table 3).

    Parameters
    ----------
    records:
        The 75-model score table, either ``{name: ModelRecord}`` (as returned by
        :mod:`src.eval.evaluate_models`), a list of dicts or a DataFrame.
    cache_dir / logits:
        Cached model outputs (``logits``) required by the AC and Aline baselines.  ``logits``
        may be a mapping ``{(model, dataset): logits}`` or ``{model: {dataset: logits}}``.
    """
    fields_by_model = _normalize_records(records)
    names = subset_model_names(fields_by_model, subset=subset)
    if verbose:
        LOG.info("OOD prediction: %d models (subset=%s), datasets=%s", len(names), subset, list(datasets))

    table = OodPredictionTable(results={}, datasets=tuple(datasets), methods=tuple(methods),
                               subset=subset, metric=metric)

    # ------------------------------------------------------------ in-distribution predictors
    for method in methods:
        if method not in ID_ONLY_METHODS:
            continue
        x_field = {"id_top1": "id_top1", "id_lca": "id_lca"}.get(method, method)
        x_values = [model_id_metric(fields_by_model[n], x_field) for n in names]
        scaler = "minmax" if method == "id_lca" else "probit"
        target_transform = "identity" if method == "id_lca" else "probit"
        for dataset in datasets:
            y_values = [model_ood_metric(fields_by_model[n], dataset, metric,
                                         record=_raw_record(records, n)) for n in names]
            xs, ys, kept = [], [], []
            for name, xv, yv in zip(names, x_values, y_values):
                if xv is None or yv is None or not np.isfinite(xv) or not np.isfinite(yv):
                    continue
                xs.append(xv)
                ys.append(yv)
                kept.append(name)
            if len(kept) < 3:
                LOG.warning("%s/%s: not enough models (%d) with valid scores", method, dataset, len(kept))
                table.results.setdefault(method, {})[dataset] = OodPredictionResult(
                    method=method, dataset=dataset, mae=float("nan"), n=len(kept)
                )
                continue
            fit = predict_linear(np.asarray(xs), np.asarray(ys), scaler=scaler,
                                 target_transform=target_transform, leave_one_out=leave_one_out)
            table.results.setdefault(method, {})[dataset] = OodPredictionResult(
                method=method,
                dataset=dataset,
                mae=fit["mae"],
                rmse=fit["rmse"],
                r2=fit["r2"],
                pea=fit["pea"],
                n=int(fit["n"]),
                predictions={n: float(p) for n, p in zip(kept, fit["predictions"])},
                targets={n: float(t) for n, t in zip(kept, ys)},
                extra={"slope": fit["slope"], "intercept": fit["intercept"], "scaler": scaler,
                       "target_transform": target_transform},
            )
            if verbose:
                LOG.info("%-8s %-9s MAE=%.3f (R2=%.3f)", method, dataset, fit["mae"], fit["r2"])

    # ------------------------------------------------------------ AC and Aline (need logits)
    needs_logits = [m for m in methods if m in OOD_LOGIT_METHODS]
    if needs_logits:
        probs_by_dataset: Dict[str, Dict[str, np.ndarray]] = {}
        id_probs: Dict[str, np.ndarray] = {}
        id_targets: Optional[np.ndarray] = None
        for name in names:
            id_out = _find_id_logits(logits, name, cache_dir=cache_dir)
            if not id_out or "logits" not in id_out:
                continue
            logits_id = _as_float_array(id_out["logits"])
            if "targets" in id_out:
                id_targets = _as_float_array(id_out["targets"]).astype(np.int64).ravel()
            temperature_used = temperature
            if calibrate_temperature and temperature_used is None:
                if id_targets is not None and id_targets.shape[0] == logits_id.shape[0]:
                    scaler_obj = fit_temperature(logits_id, id_targets, subsample=max_logit_samples)
                    temperature_used = scaler_obj.temperature
                else:
                    LOG.warning("No ID targets cached for %s; falling back to T=1.0", name)
                    temperature_used = 1.0
            probs_id = softmax(logits_id, temperature=temperature_used if temperature_used else 1.0)
            if max_logit_samples and probs_id.shape[0] > max_logit_samples:
                probs_id = probs_id[: int(max_logit_samples)]
            id_probs[name] = probs_id

            if "ac" in needs_logits:
                for dataset in datasets:
                    out = _lookup_logits(logits, name, dataset, cache_dir=cache_dir)
                    if not out or "logits" not in out:
                        continue
                    logits_ood = _as_float_array(out["logits"])
                    if max_logit_samples and logits_ood.shape[0] > max_logit_samples:
                        logits_ood = logits_ood[: int(max_logit_samples)]
                    probs_ood = softmax(logits_ood, temperature=temperature_used if temperature_used else 1.0)
                    probs_by_dataset.setdefault(dataset, {})[name] = probs_ood

        if "ac" in needs_logits:
            for dataset in datasets:
                preds, targets, kept = {}, {}, []
                for name, probs in probs_by_dataset.get(dataset, {}).items():
                    true_acc = model_ood_metric(fields_by_model[name], dataset, metric,
                                                record=_raw_record(records, name))
                    if true_acc is None or not np.isfinite(true_acc):
                        continue
                    preds[name] = float(np.mean(np.max(probs, axis=-1)))
                    targets[name] = float(true_acc)
                    kept.append(name)
                if len(kept) < 3:
                    table.results.setdefault("ac", {})[dataset] = OodPredictionResult(
                        method="ac", dataset=dataset, mae=float("nan"), n=len(kept))
                    continue
                y_true = np.array([targets[n] for n in kept])
                y_pred = np.array([preds[n] for n in kept])
                table.results.setdefault("ac", {})[dataset] = OodPredictionResult(
                    method="ac", dataset=dataset,
                    mae=mean_absolute_error(y_true, y_pred),
                    rmse=float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
                    r2=r2_score(y_true, y_pred),
                    pea=pearson_correlation(y_true, y_pred),
                    n=len(kept), predictions=preds, targets=targets,
                    extra={"temperature": temperature},
                )
                if verbose:
                    LOG.info("ac       %-9s MAE=%.3f", dataset,
                             table.results["ac"][dataset].mae)

        for method, pair_type, mode in (("aline_d", "depth", agreement_d),
                                        ("aline_s", "size", agreement_s)):
            if method not in needs_logits:
                continue
            pairs = pairs_d if method == "aline_d" and pairs_d is not None else None
            if method == "aline_s" and pairs_s is not None:
                pairs = pairs_s
            if pairs is None:
                pairs = build_model_pairs(sorted(id_probs.keys()), pair_type=pair_type)
            pairs = [(a, b) for a, b in pairs if a in id_probs and b in id_probs]
            id_accuracies = {}
            for name in id_probs:
                value = model_id_metric(fields_by_model[name], "id_top1")
                if value is not None and np.isfinite(value):
                    id_accuracies[name] = float(value)
            for dataset in datasets:
                probs_ood = probs_by_dataset.get(dataset, {})
                if not probs_ood or not pairs:
                    table.results.setdefault(method, {})[dataset] = OodPredictionResult(
                        method=method, dataset=dataset, mae=float("nan"), n=0)
                    continue
                preds, fit = aline_predictions(id_probs, id_accuracies, probs_ood, pairs,
                                               mode=mode, return_fit=True)
                targets, kept = {}, []
                for name, value in preds.items():
                    true_acc = model_ood_metric(fields_by_model[name], dataset, metric,
                                                record=_raw_record(records, name))
                    if true_acc is None or not np.isfinite(true_acc):
                        continue
                    targets[name] = float(true_acc)
                    kept.append(name)
                if len(kept) < 3:
                    table.results.setdefault(method, {})[dataset] = OodPredictionResult(
                        method=method, dataset=dataset, mae=float("nan"), n=len(kept))
                    continue
                y_true = np.array([targets[n] for n in kept])
                y_pred = np.array([preds[n] for n in kept])
                table.results.setdefault(method, {})[dataset] = OodPredictionResult(
                    method=method, dataset=dataset,
                    mae=mean_absolute_error(y_true, y_pred),
                    rmse=float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
                    r2=r2_score(y_true, y_pred),
                    pea=pearson_correlation(y_true, y_pred),
                    n=len(kept),
                    predictions={n: float(preds[n]) for n in kept},
                    targets=targets,
                    extra={"agreement": mode, "n_pairs": int(fit.get("n_pairs", 0)),
                           "slope": fit.get("slope"), "intercept": fit.get("intercept")},
                )
                if verbose:
                    LOG.info("%-8s %-9s MAE=%.3f (pairs=%d)", method, dataset,
                             table.results[method][dataset].mae, len(pairs))

    return table


def _raw_record(records: Any, name: str) -> Any:
    """Best-effort access to the original record object (for ``ModelRecord.get``)."""
    if isinstance(records, Mapping):
        return records.get(name)
    return None


# --------------------------------------------------------------------------------------
# Convenience helpers for the reproduction script
# --------------------------------------------------------------------------------------
def summarize(table: OodPredictionTable, datasets: Sequence[str] = DEFAULT_OOD_DATASETS) -> Dict[str, Any]:
    """Compact success summary: per dataset best method and the LCA-vs-baseline comparison."""
    per_dataset: Dict[str, Any] = {}
    for dataset in datasets:
        best = table.best_method(dataset)
        ours = table.get("id_lca", dataset)
        baseline = table.get("id_top1", dataset)
        per_dataset[dataset] = {
            "best_method": best,
            "best_mae": (table.get(best, dataset).mae if best else float("nan")),
            "lca_mae": (ours.mae if ours is not None else float("nan")),
            "id_top1_mae": (baseline.mae if baseline is not None else float("nan")),
            "lca_beats_id_top1": bool(
                ours is not None and baseline is not None
                and np.isfinite(ours.mae) and np.isfinite(baseline.mae) and ours.mae < baseline.mae
            ),
        }
    severe = ("s", "r", "a", "objectnet")
    return {
        "subset": table.subset,
        "metric": table.metric,
        "per_dataset": per_dataset,
        "lca_best_or_competitive_on_severe_shift": all(
            per_dataset[d]["lca_beats_id_top1"] for d in severe if d in per_dataset
        ),
    }


def save_results(path: str, table: OodPredictionTable, **meta: Any) -> str:
    """Persist the prediction table (plus metadata) as JSON."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = {"meta": meta, "table": table.asdict(), "table3_reference": TABLE3_REFERENCE}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=float)
    return path


def load_records(path: str) -> Dict[str, Dict[str, Any]]:
    """Load a score table previously written by ``evaluate_models.save_results``."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, Mapping) and "records" in payload:
        payload = payload["records"]
    return _normalize_records(payload)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce Table 3: OOD error prediction.")
    parser.add_argument("--scores-json", default=None,
                        help="score table written by scripts/run_correlation.py / evaluate_models")
    parser.add_argument("--cache-dir", default=None, help="directory with cached model outputs (.npz)")
    parser.add_argument("--datasets", nargs="*", default=list(DEFAULT_OOD_DATASETS))
    parser.add_argument("--methods", nargs="*", default=list(DEFAULT_METHODS))
    parser.add_argument("--metric", default="top1", choices=["top1", "top5"])
    parser.add_argument("--subset", default="all", choices=["all", "vm", "vlm"])
    parser.add_argument("--temperature", type=float, default=None,
                        help="fixed softmax temperature for AC (default: fitted on ID)")
    parser.add_argument("--no-calibration", action="store_true", help="disable temperature calibration")
    parser.add_argument("--leave-one-out", action="store_true",
                        help="predict each model from a fit that excludes it")
    parser.add_argument("--max-logit-samples", type=int, default=DEFAULT_SUBSAMPLE)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--tolerance", type=float, default=0.10)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    records: Any = None
    if args.scores_json and os.path.isfile(args.scores_json):
        records = load_records(args.scores_json)
    if not records and args.cache_dir:
        from .evaluate_models import build_hierarchy, evaluate_zoo_from_cache  # noqa: F401

        model_names = sorted({os.path.basename(p).split("__")[0]
                              for p in os.listdir(os.path.join(args.cache_dir, "outputs"))
                              if p.endswith(".npz")}) if os.path.isdir(
            os.path.join(args.cache_dir, "outputs")) else []
        if model_names:
            hierarchy = build_hierarchy(allow_synthetic=True)
            records = evaluate_zoo_from_cache(model_names, args.cache_dir, hierarchy=hierarchy,
                                              ood_names=args.datasets)
    if not records:
        LOG.error("No score table available; pass --scores-json or --cache-dir")
        return 2

    table = evaluate_ood_prediction(
        records,
        datasets=args.datasets,
        methods=args.methods,
        metric=args.metric,
        subset=args.subset,
        cache_dir=args.cache_dir,
        temperature=args.temperature,
        calibrate_temperature=not args.no_calibration,
        leave_one_out=args.leave_one_out,
        max_logit_samples=args.max_logit_samples,
        verbose=True,
    )
    print("\nTable 3 (error prediction MAE):")
    print(table.format_table())
    print("\nValidation:")
    for line in table.check_against_table3(tolerance=args.tolerance):
        print("  " + line)
    summary = summarize(table)
    os.makedirs(args.results_dir, exist_ok=True)
    save_results(os.path.join(args.results_dir, "table3_ood_prediction.json"), table, **summary)
    LOG.info("Saved results to %s", os.path.join(args.results_dir, "table3_ood_prediction.json"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    raise SystemExit(main())
