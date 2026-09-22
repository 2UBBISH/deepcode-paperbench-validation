"""Forward transfer metric used for the robotic manipulation experiments.

Appendix F::

    ForwardTransfer := (AUC - AUC_b) / (1 - AUC_b)
    AUC := 1/T * integral_0^T p(t) dt

where ``p(t)`` is the success rate of the fine-tuned model at time ``t`` and
``AUC_b`` is the area under the success-rate curve of a model trained from
scratch.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def auc(steps: Sequence[float], success_rate: Sequence[float], total_steps: Optional[float] = None) -> float:
    """Area under the success-rate curve, normalised by the training length."""

    steps = np.asarray(steps, dtype=np.float64)
    success_rate = np.asarray(success_rate, dtype=np.float64)
    if total_steps is None:
        total_steps = float(steps[-1]) if len(steps) else 1.0
    if len(steps) < 2:
        return float(success_rate.mean()) if len(success_rate) else float("nan")
    integral = np.trapz(success_rate, steps)
    return float(integral / total_steps)


def forward_transfer(
    steps: Sequence[float],
    success_rate: Sequence[float],
    baseline_steps: Sequence[float],
    baseline_success_rate: Sequence[float],
    total_steps: Optional[float] = None,
) -> float:
    """``(AUC - AUC_b) / (1 - AUC_b)``."""

    a = auc(steps, success_rate, total_steps)
    b = auc(baseline_steps, baseline_success_rate, total_steps)
    if abs(1.0 - b) < 1e-12:
        return float("nan")
    return float((a - b) / (1.0 - b))


def confidence_interval(values: Sequence[float], confidence: float = 0.9) -> tuple[float, float]:
    """Mean and half-width of a ``confidence``-level normal CI (Appendix B.3)."""

    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return float(values.mean()) if len(values) else float("nan"), 0.0
    from scipy import stats

    mean = float(values.mean())
    sem = float(values.std(ddof=1) / np.sqrt(len(values)))
    z = float(stats.norm.ppf(0.5 + confidence / 2.0))
    return mean, z * sem
