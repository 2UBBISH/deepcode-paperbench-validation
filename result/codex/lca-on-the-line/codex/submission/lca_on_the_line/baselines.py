"""Baselines for predicting OOD performance from ID measurements (Section 4.2).

Implemented here:

* ``ID Top1``  -- Accuracy-on-the-Line (Miller et al., 2021)
* ``AC``       -- Average Confidence after temperature scaling
                  (Hendrycks & Gimpel, 2017)
* ``Aline-S`` / ``Aline-D`` -- Agreement-on-the-Line (Baek et al., 2022)

The ``aline`` implementation is a faithful port of
``Agreement-on-the-line/agreement_trajectory.ipynb`` (the notebook referenced by
the paper addendum): pairwise agreements are probit-transformed, a linear fit is
performed from ID agreement to OOD agreement, and a least-squares solve turns
the per-pair estimates into per-model OOD accuracies.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .metrics import error_prediction_mae, minmax_scale


# --------------------------------------------------------------------------- #
# rescaling + linear fit
# --------------------------------------------------------------------------- #
def rescale(data, scaling: str = "probit") -> np.ndarray:
    data = np.asarray(data, dtype=np.float64)
    if scaling == "probit":
        from scipy.stats import norm

        return norm.ppf(np.clip(data, 1e-6, 1 - 1e-6))
    if scaling == "logit":
        data = np.clip(data, 1e-6, 1 - 1e-6)
        return np.log(data / (1 - data))
    if scaling in ("linear", "minmax"):
        return data if scaling == "linear" else minmax_scale(data)
    raise NotImplementedError(scaling)


def compute_linear_fit(x, y) -> Tuple[Tuple[float, float], float]:
    """OLS fit ``y = bias + slope * x``; returns ``((bias, slope), r2)``."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    design = np.stack([np.ones_like(x), x], axis=1)
    params, *_ = np.linalg.lstsq(design, y, rcond=None)
    pred = design @ params
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return (float(params[0]), float(params[1])), r2


# --------------------------------------------------------------------------- #
# Agreement-on-the-Line (Baek et al., 2022)
# --------------------------------------------------------------------------- #
def aline(
    test_preds: Sequence[np.ndarray],
    test_accuracy: Sequence[float],
    shift_preds: Sequence[np.ndarray],
    lower: float = 0.05,
    upper: float = 0.98,
    verbose: bool = False,
):
    """Port of ``aline`` from the Agreement-on-the-line notebook.

    Returns ``(pred_s, pred_d), bias, slope`` where ``pred_s`` (Aline-S) and
    ``pred_d`` (Aline-D) are the two OOD-accuracy estimates described in the
    notebook.
    """
    test_accuracy = rescale(test_accuracy, "probit")
    n = len(test_preds)
    A, test_agrs, shift_agrs, test_accs = [], [], [], []
    for i in range(n):
        for j in range(i, n):
            a = np.zeros(n)
            a[i] = 0.5
            a[j] = 0.5
            A.append(a)
            test_agrs.append(float(np.mean(test_preds[i] == test_preds[j])))
            shift_agrs.append(float(np.mean(shift_preds[i] == shift_preds[j])))
            test_accs.append(0.5 * test_accuracy[i] + 0.5 * test_accuracy[j])

    A = np.array(A)
    test_agrs = np.array(test_agrs)
    shift_agrs = np.array(shift_agrs)
    test_accs = np.array(test_accs)

    keep = (
        (test_agrs <= upper) & (test_agrs >= lower)
        & (shift_agrs <= upper) & (shift_agrs >= lower)
    )
    A, test_agrs, shift_agrs, test_accs = (
        A[keep], test_agrs[keep], shift_agrs[keep], test_accs[keep]
    )

    test_agrs = rescale(test_agrs, "probit")
    shift_agrs = rescale(shift_agrs, "probit")
    (bias, slope), fit_r2 = compute_linear_fit(test_agrs, shift_agrs)
    b = shift_agrs + slope * (test_accs - test_agrs)
    w, *_ = np.linalg.lstsq(A, b, rcond=None)
    pred_s = slope * test_accuracy + bias
    from scipy.stats import norm

    pred_d = norm.cdf(w)
    if verbose:
        print("[aline] slope=%.4f bias=%.4f r2=%.4f" % (slope, bias, fit_r2))
    return (pred_s, pred_d), bias, slope


def get_predictions(evaluations) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper around a list of per-model evaluation dicts."""
    test_preds = [e["test_pred"] for e in evaluations]
    test_accuracy = [e["test_accuracy"] for e in evaluations]
    shift_preds = [e["shift_pred"] for e in evaluations]
    (pred_s, pred_d), _, _ = aline(test_preds, test_accuracy, shift_preds)
    return pred_s, pred_d


# --------------------------------------------------------------------------- #
# Average Confidence (Hendrycks & Gimpel, 2017)
# --------------------------------------------------------------------------- #
def max_confidence(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim == 1:
        return probs
    return probs.max(axis=-1)


def average_confidence(probs: np.ndarray) -> float:
    """Mean maximum probability (the AC predictor)."""
    return float(np.mean(max_confidence(probs)))


def fit_temperature(logits: np.ndarray, targets: Sequence[int],
                    max_iter: int = 200) -> float:
    """Fit a temperature on ID logits by minimising the NLL (temperature scaling)."""
    import torch

    logits_t = torch.tensor(np.asarray(logits, dtype=np.float64), dtype=torch.float64)
    targets_t = torch.tensor(np.asarray(targets), dtype=torch.long)
    log_temp = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimiser = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=max_iter)
    criterion = torch.nn.CrossEntropyLoss()

    def closure():
        optimiser.zero_grad()
        loss = criterion(logits_t / torch.exp(log_temp), targets_t)
        loss.backward()
        return loss

    optimiser.step(closure)
    return float(torch.exp(log_temp).item())


def softmax_np(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64) / max(temperature, 1e-8)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


# --------------------------------------------------------------------------- #
# Table 3: error-prediction comparison
# --------------------------------------------------------------------------- #
def table3_row(
    id_metric: np.ndarray,
    ood_accuracy: np.ndarray,
    normalize: str = "minmax",
) -> float:
    return error_prediction_mae(
        id_metric, ood_accuracy, normalize=(normalize == "minmax")
    )


def evaluate_error_predictors(
    id_accuracy: np.ndarray,
    id_lca: np.ndarray,
    ood_accuracy: np.ndarray,
    id_preds: Sequence[np.ndarray],
    ood_preds: Sequence[np.ndarray],
    ood_confidence: Optional[Sequence[float]] = None,
    id_probs: Optional[Sequence[np.ndarray]] = None,
    ood_probs: Optional[Sequence[np.ndarray]] = None,
    normalize_agreement: bool = True,
) -> Dict[str, float]:
    """MAE (lower is better) for every baseline of Table 3.

    ``normalize_agreement`` applies min-max scaling (the paper's choice) instead
    of the probit transform for the accuracy-style baselines.

    ``ood_confidence`` (the per-model average confidence, AC) may be supplied
    directly to avoid keeping every model's probability matrix in memory; if it
    is omitted the confidence is derived from ``ood_probs``.
    """
    results: Dict[str, float] = {}
    results["ID Top1 (Miller et al., 2021)"] = table3_row(id_accuracy, ood_accuracy)

    if ood_confidence is None:
        if ood_probs is None:
            raise ValueError("provide either `ood_confidence` or `ood_probs`")
        ood_confidence = [average_confidence(p) for p in ood_probs]
    results["AC (Hendrycks & Gimpel, 2017)"] = error_prediction_mae(
        np.asarray(ood_confidence), ood_accuracy
    )

    if normalize_agreement:
        # min-max variant of agreement-on-the-line
        (pred_s, pred_d), _, _ = aline_minmax(
            id_preds, id_accuracy, ood_preds
        )
    else:
        (pred_s, pred_d), _, _ = aline(id_preds, id_accuracy, ood_preds)
    results["Aline-D (Baek et al., 2022)"] = float(
        np.mean(np.abs(minmax_scale(pred_d) - minmax_scale(ood_accuracy)))
    )
    results["Aline-S (Baek et al., 2022)"] = float(
        np.mean(np.abs(minmax_scale(pred_s) - minmax_scale(ood_accuracy)))
    )
    results["(Ours) ID LCA"] = table3_row(id_lca, ood_accuracy)
    return results


def aline_minmax(
    test_preds: Sequence[np.ndarray],
    test_accuracy: Sequence[float],
    shift_preds: Sequence[np.ndarray],
    lower: float = 0.05,
    upper: float = 0.98,
):
    """Same as :func:`aline` but with min-max scaling instead of probit."""
    test_accuracy = minmax_scale(np.asarray(test_accuracy, dtype=np.float64))
    n = len(test_preds)
    A, test_agrs, shift_agrs, test_accs = [], [], [], []
    for i in range(n):
        for j in range(i, n):
            a = np.zeros(n)
            a[i] = 0.5
            a[j] = 0.5
            A.append(a)
            test_agrs.append(float(np.mean(test_preds[i] == test_preds[j])))
            shift_agrs.append(float(np.mean(shift_preds[i] == shift_preds[j])))
            test_accs.append(0.5 * test_accuracy[i] + 0.5 * test_accuracy[j])
    A = np.array(A)
    test_agrs = np.array(test_agrs)
    shift_agrs = np.array(shift_agrs)
    test_accs = np.array(test_accs)
    keep = (
        (test_agrs <= upper) & (test_agrs >= lower)
        & (shift_agrs <= upper) & (shift_agrs >= lower)
    )
    A, test_agrs, shift_agrs, test_accs = (
        A[keep], test_agrs[keep], shift_agrs[keep], test_accs[keep]
    )
    (bias, slope), _ = compute_linear_fit(test_agrs, shift_agrs)
    b = shift_agrs + slope * (test_accs - test_agrs)
    w, *_ = np.linalg.lstsq(A, b, rcond=None)
    pred_s = slope * test_accuracy + bias
    pred_d = w
    return (pred_s, pred_d), bias, slope
