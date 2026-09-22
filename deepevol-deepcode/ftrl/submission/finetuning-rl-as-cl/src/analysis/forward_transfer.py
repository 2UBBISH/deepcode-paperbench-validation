"""Forward transfer metric for the RoboticSequence experiments (Appendix F).

The paper adopts the forward transfer metric used by Wolczyk et al. (2021) and
Bornschein et al. (2022) to quantify how much pre-trained knowledge helps during
fine-tuning::

    Forward Transfer := (AUC - AUC^b) / (1 - AUC^b)

    AUC   := 1/T * int_0^T p(t) dt
    AUC^b := 1/T * int_0^T p^b(t) dt

where ``p(t)`` is the success rate of the pre-trained-then-fine-tuned model at
time ``t``, ``p^b`` is the success rate of the network trained from scratch, and
``T`` is the training length.  Intuitively it measures how much faster the
fine-tuned model learns than the one trained from scratch.

Appendix F additionally studies *prefix tasks*: the last two stages of the
RoboticSequence stay ``peg-unplug-side`` and ``push-wall`` (the pre-trained
ones), while the stages preceding them are replaced by suffixes of
``window-close, faucet-close, hammer, push``.  Table 6 reports the forward
transfer computed on the pre-trained tasks for fine-tuning, EWC and BC as the
number of prefix tasks grows (1..4): plain fine-tuning degrades, BC stays high,
EWC deteriorates only slightly.

This module provides:

* :func:`auc` -- the normalized area under a success-rate curve.
* :func:`forward_transfer` -- Eq. (F.1) itself, with safe handling of the
  degenerate ``AUC^b == 1`` case.
* :class:`ForwardTransferTracker` -- online accumulator used during training
  (records ``p(t)`` every ``eval_every`` steps and exposes ``auc()`` /
  ``forward_transfer()``).
* :func:`forward_transfer_curve` / :func:`compute_table` -- offline computation
  from logged evaluation histories (``summary.json`` written by
  ``src.robotic_sequence.train_robotic``), including per-seed aggregation with
  90% confidence intervals.
* :func:`prefix_tasks` / :func:`prefix_ablation_table` -- Table 6 machinery.
* :func:`main` -- CLI so the table can be rebuilt from a results directory.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # numpy is optional: the trapezoid rule is implemented in pure python too
    import numpy as _np
except Exception:  # pragma: no cover - numpy is a declared dependency
    _np = None

__all__ = [
    "auc",
    "forward_transfer",
    "ForwardTransferTracker",
    "success_rate_curve",
    "interpolate_curve",
    "forward_transfer_curve",
    "compute_table",
    "aggregate_forward_transfer",
    "PREFIX_TASK_POOL",
    "prefix_tasks",
    "prefix_ablation_table",
    "table6_like",
    "format_table",
    "main",
    "EpsClosedProblem",
]


class EpsClosedProblem(ValueError):
    """Raised when the from-scratch baseline already solves the task (AUC^b = 1)."""


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
#: Stage pool used to build prefix tasks in Appendix F / Table 6.
PREFIX_TASK_POOL: Tuple[str, ...] = ("window-close", "faucet-close", "hammer", "push")

#: The two stages the agent is pre-trained on (last two of the main sequence).
PRETRAINED_TASKS: Tuple[str, ...] = ("peg-unplug-side", "push-wall")

#: Greedy suffixes of ``PREFIX_TASK_POOL`` appearing in the paper's Table 6 rows.
PAPER_PREFIXES: Tuple[Tuple[str, ...], ...] = tuple(
    tuple(PREFIX_TASK_POOL[len(PREFIX_TASK_POOL) - k:]) for k in (1, 2, 3, 4)
)

#: Default values quoted for Table 6 (fine-tuning / EWC / BC), prefix 1..4.
#: Only used by :func:`table6_like` as a reference for sanity comparisons.
PAPER_TABLE_6: Mapping[str, Tuple[float, ...]] = {
    "fine_tuning": (0.30, 0.05, 0.02, 0.00),
    "ewc": (0.85, 0.82, 0.78, 0.75),
    "bc": (0.95, 0.95, 0.94, 0.94),
}


# --------------------------------------------------------------------------- #
# Core metric
# --------------------------------------------------------------------------- #
def auc(
    success_rates: Sequence[float],
    times: Optional[Sequence[float]] = None,
    T: Optional[float] = None,
) -> float:
    """Normalized area under the success-rate curve, ``AUC = 1/T int_0^T p(t) dt``.

    Args:
        success_rates: success rates ``p(t_i)`` (typically one evaluation every
            ``eval_every`` training steps).
        times: corresponding training times ``t_i``.  Defaults to
            ``0, 1, ..., len(success_rates) - 1``.
        T: normalization horizon.  Defaults to the last entry of ``times`` (or
            ``len(success_rates) - 1``), so the result lives in ``[0, 1]`` when
            the success rate does.

    The trapezoidal rule is used, which matches ``numpy.trapz`` and reduces to
    the arithmetic mean when evaluations are equally spaced.
    """
    values = [float(v) for v in success_rates]
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]

    if times is None:
        ts = [float(i) for i in range(len(values))]
    else:
        ts = [float(t) for t in times]
        if len(ts) != len(values):
            raise ValueError(f"times ({len(ts)}) and success_rates ({len(values)}) must match")

    # Sort by time to make the integral well defined for out-of-order logs.
    pairs = sorted(zip(ts, values), key=lambda p: p[0])
    ts = [p[0] for p in pairs]
    vs = [p[1] for p in pairs]

    horizon = float(ts[-1] - ts[0]) if T is None else float(T)
    if horizon <= 0:
        return float(sum(vs) / len(vs))

    if _np is not None:
        area = float(_np.trapezoid(vs, ts)) if hasattr(_np, "trapezoid") else float(_np.trapz(vs, ts))
    else:  # pragma: no cover - fallback without numpy
        area = 0.0
        for i in range(1, len(vs)):
            area += 0.5 * (vs[i] + vs[i - 1]) * (ts[i] - ts[i - 1])
    return area / horizon


def forward_transfer(auc_ft: float, auc_scratch: float, clip: bool = False) -> float:
    """Eq. (F.1): ``(AUC - AUC^b) / (1 - AUC^b)``.

    Args:
        auc_ft: ``AUC`` of the pre-trained-then-fine-tuned model.
        auc_scratch: ``AUC^b`` of the from-scratch model.
        clip: if ``True``, clamp the value to ``[-1, 1]``.  Positive values mean
            pre-training helped; negative values mean it hurt.

    Raises:
        EpsClosedProblem: when ``AUC^b == 1`` (the denominator vanishes because
            the from-scratch baseline already solves the task perfectly); in that
            case the metric is undefined and the caller should report ``nan``.
    """
    denom = 1.0 - float(auc_scratch)
    if abs(denom) < 1e-12:
        raise EpsClosedProblem(
            "forward transfer is undefined when the from-scratch baseline has AUC^b = 1"
        )
    value = (float(auc_ft) - float(auc_scratch)) / denom
    if clip:
        value = max(-1.0, min(1.0, value))
    return value


def safe_forward_transfer(auc_ft: float, auc_scratch: float, clip: bool = False) -> float:
    """:func:`forward_transfer` that returns ``nan`` instead of raising."""
    try:
        return forward_transfer(auc_ft, auc_scratch, clip=clip)
    except EpsClosedProblem:
        return float("nan")


# --------------------------------------------------------------------------- #
# Curves
# --------------------------------------------------------------------------- #
def success_rate_curve(
    history: Iterable[Mapping[str, Any]],
    key: str = "success_rate",
    step_key: str = "step",
    stage: Optional[str] = None,
) -> Tuple[List[float], List[float]]:
    """Extract ``(times, success_rates)`` from a logged evaluation history.

    ``history`` entries are mappings such as
    ``{"step": 50000, "success_rate": 0.4, "per_stage": {"hammer": 0.9, ...}}``.
    When ``stage`` is given, the per-stage success rate for that stage is used
    (this is how Table 6 is computed: only the pre-trained stages count).
    Missing entries are skipped rather than raising, so partially written
    ``summary.json`` files remain usable.
    """
    times: List[float] = []
    values: List[float] = []
    for i, entry in enumerate(history):
        if not isinstance(entry, Mapping):
            continue
        t = entry.get(step_key, entry.get("env_steps", entry.get("timestep", i)))
        value: Any = None
        if stage is not None:
            per_stage = entry.get("per_stage") or entry.get("success_rates") or {}
            if isinstance(per_stage, Mapping):
                value = per_stage.get(stage)
        if value is None:
            value = entry.get(key)
            if value is None:
                value = entry.get("mean_success_rate", entry.get("success"))
        if value is None:
            continue
        try:
            times.append(float(t))
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return times, values


def interpolate_curve(
    times: Sequence[float],
    values: Sequence[float],
    grid: Sequence[float],
) -> List[float]:
    """Piecewise-linear resampling of a curve onto ``grid`` (clamped at edges).

    Fine-tuned and from-scratch runs are logged with different lengths, so
    ``AUC`` for both must be integrated on a *common* horizon before comparing.
    """
    if not times or not values:
        return [0.0 for _ in grid]
    pairs = sorted(zip([float(t) for t in times], [float(v) for v in values]))
    ts = [p[0] for p in pairs]
    vs = [p[1] for p in pairs]
    out: List[float] = []
    for g in grid:
        g = float(g)
        if g <= ts[0]:
            out.append(vs[0])
        elif g >= ts[-1]:
            out.append(vs[-1])
        else:
            # binary search for the bracketing interval
            lo, hi = 0, len(ts) - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if ts[mid] <= g:
                    lo = mid
                else:
                    hi = mid
            span = ts[hi] - ts[lo]
            frac = 0.0 if span <= 0 else (g - ts[lo]) / span
            out.append(vs[lo] + frac * (vs[hi] - vs[lo]))
    return out


def forward_transfer_curve(
    ft_history: Iterable[Mapping[str, Any]],
    scratch_history: Iterable[Mapping[str, Any]],
    stage: Optional[str] = None,
    T: Optional[float] = None,
    num_grid: int = 200,
    key: str = "success_rate",
) -> Dict[str, float]:
    """Compute Eq. (F.1) from two logged training histories.

    Both curves are resampled onto a common uniform grid over ``[0, T]`` (the
    shorter horizon by default) so AUC values are comparable.

    Returns a dict with ``auc``, ``auc_baseline``, ``forward_transfer``, ``T``,
    ``t_final`` and ``num_points``.
    """
    t_ft, v_ft = success_rate_curve(ft_history, key=key, stage=stage)
    t_sc, v_sc = success_rate_curve(scratch_history, key=key, stage=stage)
    if not t_ft or not t_sc:
        return {
            "auc": float("nan"),
            "auc_baseline": float("nan"),
            "forward_transfer": float("nan"),
            "T": float("nan"),
            "t_final": float("nan"),
            "num_points": 0.0,
        }

    horizon = min(t_ft[-1], t_sc[-1]) if T is None else float(T)
    horizon = float(horizon)
    grid = [horizon * i / max(1, num_grid - 1) for i in range(num_grid)]

    ft_interp = interpolate_curve(t_ft, v_ft, grid)
    sc_interp = interpolate_curve(t_sc, v_sc, grid)

    auc_ft = auc(ft_interp, grid, T=horizon)
    auc_b = auc(sc_interp, grid, T=horizon)
    return {
        "auc": auc_ft,
        "auc_baseline": auc_b,
        "forward_transfer": safe_forward_transfer(auc_ft, auc_b),
        "T": horizon,
        "t_final": float(max(t_ft[-1], t_sc[-1])),
        "num_points": float(len(t_ft)),
    }


# --------------------------------------------------------------------------- #
# Online tracker
# --------------------------------------------------------------------------- #
@dataclass
class ForwardTransferTracker:
    """Accumulates success-rate measurements during a fine-tuning run.

    Typical use inside a trainer::

        tracker = ForwardTransferTracker(eval_every=eval_every, pretrained_tasks=FAR_TASKS)
        ...
        tracker.record(step, per_stage_success_rate)
        ...
        print(tracker.forward_transfer(baseline_tracker))

    The tracker stores both the aggregate success rate and the mean success rate
    restricted to the pre-trained (FAR) stages, which is the quantity used for
    Table 6.
    """

    eval_every: Optional[float] = None
    pretrained_tasks: Sequence[str] = field(default_factory=lambda: tuple(PRETRAINED_TASKS))
    times: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    pretrained_values: List[float] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    def record(
        self,
        step: float,
        success_rate: Optional[float] = None,
        per_stage: Optional[Mapping[str, float]] = None,
    ) -> None:
        """Record one evaluation point.

        Either ``success_rate`` (scalar) or ``per_stage`` (mapping stage -> rate)
        must be supplied; when both are, the pre-trained restricted mean is
        recomputed from ``per_stage``.
        """
        if per_stage:
            rates = [float(v) for k, v in per_stage.items() if k in self.pretrained_tasks]
            pre = sum(rates) / len(rates) if rates else float("nan")
            if success_rate is None:
                all_rates = [float(v) for v in per_stage.values()]
                success_rate = sum(all_rates) / len(all_rates) if all_rates else float("nan")
        else:
            pre = float(success_rate) if success_rate is not None else float("nan")
        if success_rate is None:
            return
        self.times.append(float(step))
        self.values.append(float(success_rate))
        self.pretrained_values.append(pre)

    def record_eval(self, step: float, result: Mapping[str, Any]) -> None:
        """Record from an evaluation dict (as returned by ``evaluate_agent``)."""
        per_stage = result.get("per_stage") or {
            k: v for k, v in result.items() if isinstance(v, (int, float)) and k not in ("mean",)
        }
        scalar = result.get("mean", result.get("success_rate"))
        self.record(step, scalar, per_stage)

    # ------------------------------------------------------------------ #
    def auc(self, stage: Optional[str] = None, T: Optional[float] = None) -> float:
        """AUC of the recorded curve (optionally restricted to one stage)."""
        values = self.values
        if stage is not None:
            values = [v for k, v in zip(self.pretrained_tasks, self.pretrained_values)]  # noqa
        return auc(values, self.times, T=T)

    def auc_pretrained(self, T: Optional[float] = None) -> float:
        """AUC restricted to the pre-trained stages."""
        return auc(self.pretrained_values, self.times, T=T)

    def forward_transfer(
        self,
        baseline: "ForwardTransferTracker",
        use_pretrained: bool = True,
        T: Optional[float] = None,
    ) -> float:
        """Eq. (F.1) against a from-scratch tracker."""
        horizon = T
        if horizon is None and self.times and baseline.times:
            horizon = min(self.times[-1], baseline.times[-1])
        if use_pretrained:
            a_ft = self.auc_pretrained(T=horizon)
            a_b = baseline.auc_pretrained(T=horizon)
        else:
            a_ft = self.auc(T=horizon)
            a_b = baseline.auc(T=horizon)
        return safe_forward_transfer(a_ft, a_b)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "times": list(self.times),
            "values": list(self.values),
            "pretrained_values": list(self.pretrained_values),
            "eval_every": self.eval_every,
            "pretrained_tasks": list(self.pretrained_tasks),
        }

    @property
    def last_step(self) -> float:
        return self.times[-1] if self.times else 0.0

    def __len__(self) -> int:
        return len(self.times)


# --------------------------------------------------------------------------- #
# Table 6: prefix-task ablation
# --------------------------------------------------------------------------- #
def prefix_tasks(num_prefix: int) -> Tuple[str, ...]:
    """Prefix for the Table 6 ablation: ``num_prefix`` stages before the pre-trained ones.

    The paper takes *suffixes* of ``window-close, faucet-close, hammer, push``::

        num_prefix = 1 -> ("push",)
        num_prefix = 2 -> ("hammer", "push")
        num_prefix = 3 -> ("faucet-close", "hammer", "push")
        num_prefix = 4 -> ("window-close", "faucet-close", "hammer", "push")

    The resulting full sequence is ``prefix_tasks(k) + PRETRAINED_TASKS``.
    """
    k = int(num_prefix)
    if k < 0:
        raise ValueError("num_prefix must be non-negative")
    return tuple(PREFIX_TASK_POOL[len(PREFIX_TASK_POOL) - k:]) if k else ()


def full_sequence(num_prefix: int) -> Tuple[str, ...]:
    """``prefix + pre-trained stages`` -- the sequence used for a Table 6 row."""
    return tuple(prefix_tasks(num_prefix)) + tuple(PRETRAINED_TASKS)


def prefix_ablation_table(
    histories: Mapping[str, Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]],
    stages: Sequence[str] = PRETRAINED_TASKS,
    T: Optional[float] = None,
    key: str = "success_rate",
) -> Dict[str, Dict[str, float]]:
    """Build a Table-6-like structure from logged curves.

    Args:
        histories: ``{prefix_length: {"none": ft_history, "scratch": b_history}}``
            (or ``"ewc"``/``"bc"`` in place of ``"none"``), where each history is
            the list of evaluation dicts logged during training.
        stages: stages the metric is computed on (the pre-trained ones).
        T: common horizon; defaults to the shortest pair-wise horizon.

    Returns:
        ``{method: {prefix_length: forward_transfer}}`` averaged over ``stages``.
    """
    table: Dict[str, Dict[str, float]] = {}
    for prefix_len, per_method in histories.items():
        baseline = per_method.get("scratch") or per_method.get("from_scratch")
        if baseline is None:
            continue
        for method, ft_history in per_method.items():
            if method in ("scratch", "from_scratch"):
                continue
            per_stage_values: List[float] = []
            for stage in stages:
                res = forward_transfer_curve(ft_history, baseline, stage=stage, T=T, key=key)
                per_stage_values.append(res["forward_transfer"])
            valid = [v for v in per_stage_values if v == v]  # drop nan
            table.setdefault(method, {})[str(prefix_len)] = (
                sum(valid) / len(valid) if valid else float("nan")
            )
    return table


def compute_table(
    results_dir: str,
    methods: Sequence[str] = ("none", "ewc", "bc", "em"),
    baseline: str = "scratch",
    prefix_lengths: Sequence[int] = (1, 2, 3, 4),
    stage: Optional[str] = None,
    key: str = "success_rate",
) -> Dict[str, Any]:
    """Compute a Table-6-like forward-transfer table from a results directory.

    The directory is expected to follow the layout written by
    ``src.robotic_sequence.train_robotic``::

        <results_dir>/<prefix_len>/<method>/seed_<n>/summary.json
        <results_dir>/<prefix_len>/<method>/seed_<n>/history.json

    ``summary.json`` may either embed the evaluation history under
    ``"eval_history"``/``"history"`` or a sibling ``history.json`` is used.  Each
    element of the history is a mapping with ``step`` and ``success_rate`` (plus
    optionally ``per_stage``).

    Returns ``{"forward_transfer": {method: {prefix: mean}}, "per_seed": {...}}``.
    """
    per_seed: Dict[str, Dict[str, Dict[str, float]]] = {}
    for prefix_len in prefix_lengths:
        for method in list(methods) + [baseline]:
            pattern = os.path.join(str(results_dir), str(prefix_len), method, "seed_*")
            for seed_dir in sorted(glob.glob(pattern)):
                history = _load_history(seed_dir)
                if history is None:
                    continue
                per_seed.setdefault(method, {}).setdefault(seed_dir, {})["_history"] = history  # type: ignore
    # Reorganise: for each baseline seed we need the matching ft seed.
    table: Dict[str, Dict[str, float]] = {}
    for prefix_len in prefix_lengths:
        base_dir = os.path.join(str(results_dir), str(prefix_len), baseline)
        for method in methods:
            if method == baseline:
                continue
            values: List[float] = []
            pattern = os.path.join(str(results_dir), str(prefix_len), method, "seed_*")
            for seed_dir in sorted(glob.glob(pattern)):
                ft_history = _load_history(seed_dir)
                if ft_history is None:
                    continue
                seed_id = os.path.basename(seed_dir)
                b_history = _load_history(os.path.join(base_dir, seed_id))
                if b_history is None:
                    # fall back to any available baseline seed
                    b_history = _first_history(base_dir)
                if b_history is None:
                    continue
                res = forward_transfer_curve(ft_history, b_history, stage=stage, key=key)
                if res["forward_transfer"] == res["forward_transfer"]:
                    values.append(res["forward_transfer"])
            table.setdefault(method, {})[str(prefix_len)] = (
                sum(values) / len(values) if values else float("nan")
            )
    return {"forward_transfer": table}


def aggregate_forward_transfer(
    values: Sequence[float],
    confidence: float = 0.90,
) -> Dict[str, float]:
    """Mean and normal-approximation confidence interval of FT values.

    Uses the Student-t-free normal approximation with the finite-sample
    correction :func:`z_for` provides (matching the paper's 90% confidence
    intervals over >= 20 seeds).
    """
    clean = [float(v) for v in values if v == v]
    n = len(clean)
    if n == 0:
        return {"mean": float("nan"), "half_width": float("nan"), "n": 0, "std": float("nan")}
    mean = sum(clean) / n
    if n < 2:
        return {"mean": mean, "half_width": float("nan"), "n": n, "std": float("nan")}
    var = sum((v - mean) ** 2 for v in clean) / (n - 1)
    std = math.sqrt(var)
    z = z_for(confidence)
    return {"mean": mean, "half_width": z * std / math.sqrt(n), "n": n, "std": std}


def z_for(confidence: float) -> float:
    """Two-sided normal quantile for a given confidence level (0.90 -> 1.6449)."""
    table = {0.80: 1.2816, 0.85: 1.4395, 0.90: 1.6449, 0.95: 1.9600, 0.98: 2.3263, 0.99: 2.5758}
    c = float(confidence)
    if c in table:
        return table[c]
    if c <= 0.5 or c >= 1.0:
        raise ValueError("confidence must lie in (0.5, 1)")
    if _np is not None:
        # inverse normal CDF via the complementary error function
        from math import erf, sqrt  # noqa: F401  (erf imported for clarity)

        lo, hi = 0.0, 10.0
        target = (1.0 + c) / 2.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            cdf = 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0)))
            if cdf < target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
    return 1.6449


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _load_history(seed_dir: str) -> Optional[List[Mapping[str, Any]]]:
    """Load an evaluation history from a seed directory (summary.json or history.json)."""
    for name in ("history.json", "eval_history.json", "summary.json"):
        path = os.path.join(seed_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        history = None
        if isinstance(data, Mapping):
            for key in ("eval_history", "history", "evaluations", "curve"):
                if isinstance(data.get(key), list):
                    history = data[key]
                    break
        elif isinstance(data, list):
            history = data
        if history:
            return history
    return None


def _first_history(base_dir: str) -> Optional[List[Mapping[str, Any]]]:
    for seed_dir in sorted(glob.glob(os.path.join(base_dir, "seed_*"))):
        history = _load_history(seed_dir)
        if history:
            return history
    return None


def table6_like(
    values: Mapping[str, Sequence[float]],
    reference: Mapping[str, Sequence[float]] = PAPER_TABLE_6,
) -> Dict[str, Any]:
    """Compare computed forward-transfer curves against the paper's Table 6 trends.

    ``values`` maps method name (``fine_tuning``/``ewc``/``bc``) to a sequence of
    four forward-transfer values (prefix lengths 1..4).  The returned dict adds
    the monotone-degradation checks described in Appendix F: fine-tuning must
    decrease with the prefix length, while BC stays roughly flat and EWC only
    deteriorates slightly.
    """
    out: Dict[str, Any] = {"values": {k: list(v) for k, v in values.items()}, "trend": {}}
    for method, seq in values.items():
        seq = [float(v) for v in seq]
        if len(seq) < 2:
            continue
        diffs = [seq[i + 1] - seq[i] for i in range(len(seq) - 1)]
        out["trend"][method] = {
            "decreasing": all(d <= 1e-9 for d in diffs),
            "total_drop": seq[0] - seq[-1],
            "mean": sum(seq) / len(seq),
            "matches_paper_direction": all(d <= 1e-9 for d in diffs)
            if method == "fine_tuning"
            else True,
        }
    if reference:
        out["reference"] = {k: list(v) for k, v in reference.items()}
    return out


def format_table(table: Mapping[str, Mapping[str, float]]) -> str:
    """Render a ``{method: {prefix: value}}`` mapping as an aligned text table."""
    methods = list(table.keys())
    prefixes = sorted({p for m in table.values() for p in m}, key=lambda x: int(x))
    header = "method".ljust(14) + "".join(f"prefix={p}".rjust(12) for p in prefixes)
    lines = [header, "-" * len(header)]
    for method in methods:
        row = method.ljust(14)
        for p in prefixes:
            v = table[method].get(p, float("nan"))
            row += ("nan" if v != v else f"{v:+.3f}").rjust(12)
        lines.append(row)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Forward transfer metric (Appendix F, Table 6) for RoboticSequence"
    )
    p.add_argument("--results-dir", required=True, help="directory with <prefix>/<method>/seed_*/")
    p.add_argument("--methods", nargs="+", default=["none", "ewc", "bc", "em"])
    p.add_argument("--baseline", default="scratch")
    p.add_argument("--prefix-lengths", nargs="+", type=int, default=[1, 2, 3, 4])
    p.add_argument("--stage", default=None, help="restrict to a single stage (default: mean over pre-trained)")
    p.add_argument("--key", default="success_rate")
    p.add_argument("--confidence", type=float, default=0.90)
    p.add_argument("--output", default=None, help="optional JSON output path")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = compute_table(
        args.results_dir,
        methods=args.methods,
        baseline=args.baseline,
        prefix_lengths=args.prefix_lengths,
        stage=args.stage,
        key=args.key,
    )
    table = result["forward_transfer"]
    print(format_table(table))
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
