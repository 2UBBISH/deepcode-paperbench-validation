"""Return-distribution analysis for the fine-tuning / retention experiments.

This module implements the *final-return distribution* analysis used for the main
comparison figures of Wołczyk et al. (2024):

* Figure 3 (main results): fine-tuning (vanilla) vs. from-scratch vs. the
  knowledge-retention variants (EWC / BC / KS / EM) on NetHack, Montezuma's
  Revenge and RoboticSequence.  Each method contributes a distribution of
  evaluation returns (one value per evaluation episode, per seed); the figure
  reports the mean with a confidence interval and the spread of the individual
  episodes.
* Appendix E (Montezuma) / Appendix D (NetHack) return distributions per method.

Everything here is deliberately dependency-light: the numeric bookkeeping works
with plain Python lists, while NumPy and matplotlib are imported lazily so the
module can be imported (and unit-tested) in headless / minimal environments.
That mirrors the design of the sibling analysis modules
(:mod:`src.analysis.plotting`, :mod:`src.analysis.density_plots`).

Typical usage
-------------
>>> tracker = ReturnTracker(method="bc", seed=0)
>>> tracker.add_episodes(eval_returns)          # list of per-episode returns
>>> tracker.add_scalars(mean_return=7610.0, std_return=1200.0, n_episodes=1000)
>>> payload = tracker.to_dict()

>>> dist = ReturnDistribution.from_summaries([s1, s2, s3])   # three seeds
>>> dist.mean, dist.half_width

The module also provides

* :func:`collect_return_summaries` / :func:`load_return_records`: discovery of
  ``summary.json`` / ``returns.json`` files written by the trainers, following
  the same ``<results_dir>/<method>/seed_<seed>/`` convention as
  :mod:`src.analysis.forward_transfer` and :mod:`src.analysis.density_plots`.
* :func:`compare_methods`: pairwise Welch tests + Cohen's *d* against a reference
  method (``scratch`` by default), which is how the paper's "does retention beat
  vanilla / from scratch" claims were checked.
* :func:`plot_return_distribution` / :func:`plot_return_violin` /
  :func:`plot_return_grid` / :func:`plot_learning_curve_returns`: Figure-3 style
  rendering helpers.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

try:  # optional NumPy fast path
    import numpy as _np
except Exception:  # pragma: no cover - NumPy is expected in practice
    _np = None


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_CONFIDENCE = 0.90
DEFAULT_BINS = 30
DEFAULT_POINT_SIZE = 4.0
DEFAULT_COLORMAP = "viridis"
FIGURE_DPI = 150

#: Evaluation cadence mentioned in the reproduction plan for NetHack
#: (per-level evaluation every 25M environment steps, 200 episodes per level).
EVAL_EVERY = 25_000_000
#: Number of episodes used for the NetHack full evaluation in the paper.
NUM_EVAL_EPISODES = 1000
#: Number of evaluation episodes for Montezuma / RoboticSequence evaluations.
NUM_EVAL_EPISODES_MONTEZUMA = 100
NUM_EVAL_EPISODES_ROBOTIC = 20

#: Method ordering used throughout the paper's figures.
METHOD_ORDER: Tuple[str, ...] = ("scratch", "none", "ewc", "bc", "ks", "em")

METHOD_LABELS: Dict[str, str] = {
    "scratch": "from scratch",
    "from_scratch": "from scratch",
    "none": "fine-tuning",
    "vanilla": "fine-tuning",
    "finetune": "fine-tuning",
    "ewc": "fine-tuning + EWC",
    "bc": "fine-tuning + BC",
    "ks": "fine-tuning + KS",
    "kickstarting": "fine-tuning + KS",
    "em": "fine-tuning + EM",
    "episodic_memory": "fine-tuning + EM",
    "expert": "expert (AutoAscend)",
    "pi_star": "$\\pi_*$",
}

#: Reference numbers from the paper, used as loose reproduction sanity checks.
#: Only qualitative / order-of-magnitude comparisons are meaningful because the
#: exact numbers depend on the number of seeds and evaluation episodes.
PAPER_REFERENCES: Dict[str, Dict[str, Any]] = {
    "nethack": {
        "metric": "score",
        "pi_star": 5000.0,          # ~5K for the released Human Monk LSTM
        "scratch": 776.0,           # from-scratch baseline stays around 776
        "none": 1000.0,             # vanilla fine-tuning degrades on FAR states
        "ks": 10588.0,              # +/- 672 over a 1000-episode evaluation
        "ks_std": 672.0,
        "bc": 7610.0,
        "ewc": 3976.0,
    },
    "montezuma": {
        "metric": "episode_return",
        "pi_star": 7000.0,          # M1 reaches ~7000 cumulative reward
        "scratch": 0.0,
        "bc": 4000.0,               # BC/EWC reach a higher final return
        "ewc": 2500.0,
    },
    "robotic_sequence": {
        "metric": "success_rate",
        "pi_star": 1.0,             # pi_* solves peg-unplug-side / push-wall at 100%
        "none": 0.20,               # vanilla forgets FAR stages
        "bc": 0.80,                 # BC solves all four stages ~80% of the time
        "ewc": 0.40,
        "em": 0.40,
    },
}

SUMMARY_KEYS = (
    "return",
    "returns",
    "eval_return",
    "mean_return",
    "episode_return",
    "cumulative_reward",
    "score",
    "success_rate",
)
EPISODE_KEYS = ("episode_returns", "eval_returns", "returns", "scores", "episodes")


# --------------------------------------------------------------------------- #
# Small numeric helpers (no NumPy required)
# --------------------------------------------------------------------------- #


def z_for(confidence: float = DEFAULT_CONFIDENCE) -> float:
    """Two-sided normal quantile for a confidence level.

    Uses a small lookup table for the levels used in the paper (0.68/0.90/0.95)
    and falls back to Acklam's rational approximation of the inverse normal CDF,
    which is accurate to ~1e-9.  The paper reports 90% confidence intervals over
    at least 20 seeds, hence the 0.90 default.
    """
    table = {0.5: 0.6744897501960817, 0.68: 0.994457883209753, 0.8: 1.2815515655446004,
             0.9: 1.6448536269514722, 0.95: 1.959963984540054, 0.98: 2.3263478740408408,
             0.99: 2.5758293035489004}
    try:
        key = round(float(confidence), 4)
    except (TypeError, ValueError):
        return table[0.9]
    if key in table:
        return table[key]
    if key <= 0.0 or key >= 1.0:
        return 0.0
    # Acklam's algorithm for the inverse normal CDF.
    p = 1.0 - (1.0 - key) / 2.0
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return not (isinstance(value, float) and math.isnan(value))
    if _np is not None and isinstance(value, _np.number):
        return True
    return False


def as_float_list(values: Any) -> List[float]:
    """Best-effort conversion of anything array-like into a list of floats."""
    if values is None:
        return []
    if _is_number(values):
        return [float(values)]
    if _np is not None:
        try:
            arr = _np.asarray(values, dtype=float).ravel()
            return [float(v) for v in arr if not math.isnan(float(v))]
        except Exception:
            pass
    out: List[float] = []
    if hasattr(values, "tolist"):
        try:
            return as_float_list(values.tolist())
        except Exception:
            pass
    try:
        iterator = list(values)
    except TypeError:
        return []
    for item in iterator:
        if _is_number(item):
            out.append(float(item))
        elif isinstance(item, (list, tuple)) and item:
            # (step, value) pairs
            last = item[-1]
            if _is_number(last):
                out.append(float(last))
    return out


def as_numpy(values: Any) -> Any:
    """Convert to a NumPy array (or a list when NumPy is unavailable)."""
    flat = as_float_list(values)
    if _np is None:
        return flat
    return _np.asarray(flat, dtype=float)


def mean_of(values: Any) -> float:
    flat = as_float_list(values)
    return float(sum(flat) / len(flat)) if flat else float("nan")


def std_of(values: Any, ddof: int = 1) -> float:
    flat = as_float_list(values)
    n = len(flat)
    if n <= ddof:
        return 0.0 if n else float("nan")
    m = sum(flat) / n
    var = sum((v - m) ** 2 for v in flat) / (n - ddof)
    return math.sqrt(max(var, 0.0))


def percentiles(values: Any, quantiles: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95)) -> Dict[str, float]:
    """Linear-interpolation percentiles of a sample (used for box plots)."""
    flat = sorted(as_float_list(values))
    if not flat:
        return {}
    out: Dict[str, float] = {}
    n = len(flat)
    for q in quantiles:
        q = min(max(float(q), 0.0), 1.0)
        pos = q * (n - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        frac = pos - lo
        out[f"p{int(round(q * 100)):02d}"] = flat[lo] * (1.0 - frac) + flat[hi] * frac
    return out


def summarize(values: Any, confidence: float = DEFAULT_CONFIDENCE) -> Dict[str, float]:
    """Mean + normal-approximation confidence interval over seeds."""
    flat = as_float_list(values)
    n = len(flat)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "half_width": float("nan"),
                "n": 0, "lo": float("nan"), "hi": float("nan")}
    mean = sum(flat) / n
    std = std_of(flat)
    half = z_for(confidence) * std / math.sqrt(n) if n > 1 else 0.0
    return {"mean": float(mean), "std": float(std), "half_width": float(half), "n": int(n),
            "lo": float(mean - half), "hi": float(mean + half)}


def bootstrap_ci(values: Any, confidence: float = DEFAULT_CONFIDENCE, num_samples: int = 2000,
                 seed: int = 0) -> Dict[str, float]:
    """Percentile bootstrap CI (useful for skewed return distributions)."""
    flat = as_float_list(values)
    n = len(flat)
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0,
                "half_width": float("nan")}
    if n == 1:
        return {"mean": flat[0], "lo": flat[0], "hi": flat[0], "n": 1, "half_width": 0.0}
    rng = _np.random.default_rng(seed) if _np is not None else None
    if rng is not None:
        arr = _np.asarray(flat, dtype=float)
        idx = rng.integers(0, n, size=(num_samples, n))
        means = arr[idx].mean(axis=1)
        alpha = (1.0 - confidence) / 2.0
        lo, hi = _np.quantile(means, [alpha, 1.0 - alpha])
    else:  # pragma: no cover - deterministic fallback
        import random as _random
        rnd = _random.Random(seed)
        means = sorted(sum(rnd.choice(flat) for _ in range(n)) / n for _ in range(num_samples))
        alpha = (1.0 - confidence) / 2.0
        lo = means[max(0, int(alpha * num_samples) - 1)]
        hi = means[min(num_samples - 1, int((1.0 - alpha) * num_samples))]
    mean = sum(flat) / n
    return {"mean": float(mean), "lo": float(lo), "hi": float(hi), "n": int(n),
            "half_width": float((hi - lo) / 2.0)}


def welch_ttest(a: Any, b: Any) -> Dict[str, float]:
    """Welch's t-test without SciPy (returns t, dof and a two-sided p-value)."""
    x = as_float_list(a)
    y = as_float_list(b)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return {"t": float("nan"), "dof": float("nan"), "p_value": float("nan"),
                "mean_diff": mean_of(x) - mean_of(y)}
    mx, my = sum(x) / nx, sum(y) / ny
    vx, vy = std_of(x) ** 2, std_of(y) ** 2
    se2 = vx / nx + vy / ny
    if se2 <= 0.0:
        return {"t": 0.0, "dof": float(nx + ny - 2), "p_value": 1.0, "mean_diff": mx - my}
    t = (mx - my) / math.sqrt(se2)
    dof = se2 ** 2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    p = 2.0 * (1.0 - _student_t_cdf(abs(t), dof))
    return {"t": float(t), "dof": float(dof), "p_value": float(max(0.0, min(1.0, p))),
            "mean_diff": float(mx - my)}


def _student_t_cdf(t: float, dof: float) -> float:
    """Student-t CDF via the regularised incomplete beta function."""
    if dof <= 0 or math.isnan(dof):
        return 0.5
    x = dof / (dof + t * t)
    ib = _betainc(0.5 * dof, 0.5, x)
    return 1.0 - 0.5 * ib if t >= 0 else 0.5 * ib


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function (continued-fraction, Numerical Recipes)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta) / a
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x)
    return 1.0 - math.exp(math.log(1.0 - x) * b + math.log(x) * a - lbeta) / b * _betacf(b, a, 1.0 - x)


def _betacf(a: float, b: float, x: float, max_iter: int = 200, eps: float = 3e-12) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def cohens_d(a: Any, b: Any) -> float:
    """Standardised mean difference (pooled SD) between two samples."""
    x, y = as_float_list(a), as_float_list(b)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    vx, vy = std_of(x) ** 2, std_of(y) ** 2
    pooled = math.sqrt(((nx - 1) * vx + (ny - 1) * vy) / max(nx + ny - 2, 1))
    if pooled <= 0.0:
        return 0.0
    return float((mean_of(x) - mean_of(y)) / pooled)


def histogram(values: Any, bins: int = DEFAULT_BINS, weights: Any = None,
              density: bool = False) -> Dict[str, List[float]]:
    """Histogram of a sample -> ``{"counts", "edges", "centers"}``.

    Implemented directly so the analysis works without NumPy/matplotlib, and
    fall back to NumPy when available for speed.
    """
    flat = as_float_list(values)
    w = as_float_list(weights) if weights is not None else None
    if not flat:
        return {"counts": [], "edges": [], "centers": []}
    if _np is not None:
        arr = _np.asarray(flat, dtype=float)
        counts, edges = _np.histogram(arr, bins=bins, weights=(_np.asarray(w) if w else None),
                                      density=density)
        centers = (edges[:-1] + edges[1:]) / 2.0
        return {"counts": [float(c) for c in counts], "edges": [float(e) for e in edges],
                "centers": [float(c) for c in centers]}
    lo, hi = min(flat), max(flat)
    if hi <= lo:
        lo, hi = lo - 0.5, hi + 0.5
    width = (hi - lo) / max(bins, 1)
    edges = [lo + i * width for i in range(bins + 1)]
    counts = [0.0] * bins
    for i, v in enumerate(flat):
        idx = int((v - lo) / width)
        idx = min(max(idx, 0), bins - 1)
        counts[idx] += (w[i] if w else 1.0)
    if density:
        total = sum(counts) or 1.0
        counts = [c / total / width for c in counts]
    centers = [(edges[i] + edges[i + 1]) / 2.0 for i in range(bins)]
    return {"counts": counts, "edges": edges, "centers": centers}


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #


@dataclass
class ReturnRecord:
    """One evaluation result: (method, seed, step) -> return sample or summary."""

    method: str
    seed: int = 0
    step: int = 0
    returns: List[float] = field(default_factory=list)
    mean_return: Optional[float] = None
    std_return: Optional[float] = None
    n_episodes: Optional[int] = None
    metric: str = "return"
    task: Optional[str] = None
    success_rate: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- helpers ---------------------------------------------------------- #
    @property
    def effective_mean(self) -> float:
        if self.mean_return is not None:
            return float(self.mean_return)
        return mean_of(self.returns)

    @property
    def effective_std(self) -> float:
        if self.std_return is not None:
            return float(self.std_return)
        return std_of(self.returns)

    @property
    def effective_n(self) -> int:
        if self.n_episodes:
            return int(self.n_episodes)
        return len(self.returns)

    @property
    def half_width(self) -> float:
        n = self.effective_n
        if n <= 1:
            return 0.0
        return z_for(DEFAULT_CONFIDENCE) * self.effective_std / math.sqrt(n)

    def add_returns(self, values: Any) -> "ReturnRecord":
        flat = as_float_list(values)
        self.returns.extend(flat)
        if flat and self.mean_return is None:
            self.mean_return = mean_of(flat)
            self.std_return = std_of(flat)
            self.n_episodes = len(flat)
        return self

    def as_dict(self, include_returns: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "method": self.method,
            "seed": self.seed,
            "step": self.step,
            "metric": self.metric,
            "mean_return": self.effective_mean,
            "std_return": self.effective_std,
            "n_episodes": self.effective_n,
            "half_width": self.half_width,
        }
        if include_returns and self.returns:
            payload["returns"] = list(self.returns)
        for key in ("task", "success_rate"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReturnRecord":
        payload = dict(payload or {})
        method = str(payload.get("method", payload.get("name", "none")))
        seed = int(payload.get("seed", 0) or 0)
        step = int(payload.get("step", payload.get("global_step", 0)) or 0)
        returns: List[float] = []
        for key in EPISODE_KEYS:
            if key in payload:
                returns = as_float_list(payload[key])
                if returns:
                    break
        mean_return = payload.get("mean_return", payload.get("eval_return"))
        std_return = payload.get("std_return")
        n_episodes = payload.get("n_episodes", payload.get("num_episodes"))
        if mean_return is None:
            for key in ("return", "episode_return", "mean", "score", "eval_mean"):
                if _is_number(payload.get(key)):
                    mean_return = float(payload[key])
                    break
        if std_return is None:
            for key in ("return_std", "std", "eval_std"):
                if _is_number(payload.get(key)):
                    std_return = float(payload[key])
                    break
        if n_episodes is None:
            n_episodes = len(returns) if returns else None
        return cls(
            method=method,
            seed=seed,
            step=step,
            returns=returns,
            mean_return=float(mean_return) if _is_number(mean_return) else None,
            std_return=float(std_return) if _is_number(std_return) else None,
            n_episodes=int(n_episodes) if n_episodes else None,
            metric=str(payload.get("metric", "return")),
            task=payload.get("task"),
            success_rate=float(payload["success_rate"]) if _is_number(payload.get("success_rate")) else None,
            metadata={k: v for k, v in payload.items()
                      if k not in {"method", "seed", "step", "returns", "mean_return",
                                   "std_return", "n_episodes", "metric", "task",
                                   "success_rate", "episode_returns"}},
        )


class ReturnTracker:
    """Online accumulator of evaluation returns for a single (method, seed) run.

    The trainers call :meth:`add_episodes` (when the per-episode returns are
    available, e.g. NetHack) or :meth:`add_scalars` (when only the mean / std of
    an evaluation were logged, e.g. some Montezuma runs).
    """

    def __init__(self, method: str = "none", seed: int = 0, metric: str = "return",
                 task: Optional[str] = None, every: Optional[float] = EVAL_EVERY,
                 name: str = "returns") -> None:
        self.method = str(method)
        self.seed = int(seed)
        self.metric = str(metric)
        self.task = task
        self.every = every
        self.name = name
        self.records: List[ReturnRecord] = []
        self._last_step: Optional[int] = None

    # -- recording ------------------------------------------------------- #
    def should_record(self, step: int) -> bool:
        """True when ``step`` crosses an evaluation boundary (default 25M)."""
        if self.every is None:
            return True
        try:
            step = int(step)
        except (TypeError, ValueError):
            return True
        if self._last_step is None:
            return True
        return (step - self._last_step) >= float(self.every)

    def add_episodes(self, returns: Any, step: int = 0, reset: bool = False) -> ReturnRecord:
        """Record a full set of per-episode evaluation returns."""
        flat = as_float_list(returns)
        if reset or self._last_step is None or step > self._last_step:
            rec = ReturnRecord(method=self.method, seed=self.seed, step=int(step),
                               returns=flat, metric=self.metric, task=self.task)
            self.records.append(rec)
            self._last_step = int(step)
            return rec
        # Same evaluation step: extend the existing record.
        self.records[-1].add_returns(flat)
        return self.records[-1]

    def add_scalars(self, mean_return: float, std_return: Optional[float] = None,
                    n_episodes: Optional[int] = None, step: int = 0,
                    success_rate: Optional[float] = None, **metadata: Any) -> ReturnRecord:
        """Record only aggregate statistics of an evaluation."""
        rec = ReturnRecord(method=self.method, seed=self.seed, step=int(step),
                           mean_return=float(mean_return),
                           std_return=float(std_return) if std_return is not None else None,
                           n_episodes=int(n_episodes) if n_episodes else None,
                           metric=self.metric, task=self.task, success_rate=success_rate,
                           metadata=dict(metadata))
        self.records.append(rec)
        self._last_step = int(step)
        return rec

    def record(self, result: Mapping[str, Any] | ReturnRecord, step: Optional[int] = None) -> ReturnRecord:
        """Record from a dict produced by the trainers' evaluation loop.

        Accepts either per-episode lists (``episode_returns``/``returns``) or
        scalars (``mean_return``/``eval_return``/``score``).
        """
        if isinstance(result, ReturnRecord):
            self.records.append(result)
            self._last_step = result.step
            return result
        payload = dict(result or {})
        if step is not None:
            payload.setdefault("step", int(step))
        payload.setdefault("method", self.method)
        payload.setdefault("seed", self.seed)
        payload.setdefault("metric", self.metric)
        rec = ReturnRecord.from_dict(payload)
        self.records.append(rec)
        self._last_step = rec.step
        return rec

    # -- queries --------------------------------------------------------- #
    @property
    def last_step(self) -> int:
        return self._last_step if self._last_step is not None else 0

    def curve(self, key: str = "step") -> Tuple[List[float], List[float]]:
        """Return ``(steps, values)`` for the recorded evaluations."""
        steps = [float(r.step) for r in self.records]
        if key in ("mean", "return", "mean_return"):
            values = [r.effective_mean for r in self.records]
        elif key in ("std", "std_return"):
            values = [r.effective_std for r in self.records]
        elif key in ("success_rate", "success"):
            values = [float(r.success_rate if r.success_rate is not None else float("nan"))
                      for r in self.records]
        else:
            values = [float(r.metadata.get(key, float("nan"))) for r in self.records]
        return steps, values

    def final(self, index: int = -1) -> Optional[ReturnRecord]:
        """Last (or ``index``-th from the end) evaluation record."""
        if not self.records:
            return None
        try:
            return self.records[index]
        except IndexError:
            return None

    def all_returns(self, last_k: Optional[int] = None) -> List[float]:
        """Concatenate per-episode returns (optionally only the last k evals)."""
        records = self.records[-last_k:] if last_k else self.records
        out: List[float] = []
        for rec in records:
            out.extend(rec.returns)
        return out

    def forgetting(self, reference: Optional[float] = None, key: str = "mean") -> float:
        """Relative drop from the pre-trained / best reference value.

        ``forgetting = (ref - final) / ref`` clipped to [0, 1] when ``ref > 0``,
        matching the paper's qualitative "how much of the pre-trained capability
        is lost" reading of the FAR-state performance.
        """
        rec = self.final()
        if rec is None:
            return float("nan")
        value = rec.effective_mean if key == "mean" else (
            rec.success_rate if rec.success_rate is not None else float("nan"))
        if reference is None:
            reference = max((r.effective_mean for r in self.records), default=float("nan"))
        if not _is_number(reference) or not _is_number(value) or reference == 0:
            return float("nan")
        return float((reference - value) / abs(reference))

    def summary(self, confidence: float = DEFAULT_CONFIDENCE) -> Dict[str, Any]:
        rec = self.final()
        payload: Dict[str, Any] = {
            "method": self.method,
            "seed": self.seed,
            "metric": self.metric,
            "steps": [r.step for r in self.records],
            "means": [r.effective_mean for r in self.records],
            "final_mean": rec.effective_mean if rec else float("nan"),
            "final_std": rec.effective_std if rec else float("nan"),
            "final_half_width": rec.half_width if rec else float("nan"),
            "n_episodes": rec.effective_n if rec else 0,
            "confidence": confidence,
        }
        if rec is not None and rec.success_rate is not None:
            payload["final_success_rate"] = rec.success_rate
        return payload

    def to_dict(self, include_records: bool = True) -> Dict[str, Any]:
        payload = self.summary()
        if include_records:
            payload["records"] = [r.as_dict() for r in self.records]
        return payload

    def save(self, path: str, include_records: bool = True) -> str:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w") as handle:
            json.dump(self.to_dict(include_records=include_records), handle, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "ReturnTracker":
        with open(path, "r") as handle:
            payload = json.load(handle)
        tracker = cls(method=payload.get("method", "none"), seed=payload.get("seed", 0),
                      metric=payload.get("metric", "return"))
        for rec in payload.get("records", []):
            tracker.records.append(ReturnRecord.from_dict(rec))
        if not tracker.records and _is_number(payload.get("final_mean")):
            tracker.add_scalars(payload["final_mean"], payload.get("final_std"),
                                payload.get("n_episodes"), payload.get("step", 0))
        return tracker


@dataclass
class ReturnDistribution:
    """Aggregated return distribution of one method across seeds."""

    method: str
    values: List[float] = field(default_factory=list)          # final return per seed
    episode_values: List[float] = field(default_factory=list)  # all eval episodes pooled
    per_seed: List[float] = field(default_factory=list)
    mean: float = float("nan")
    std: float = float("nan")
    half_width: float = float("nan")
    n: int = 0
    confidence: float = DEFAULT_CONFIDENCE
    quantiles: Dict[str, float] = field(default_factory=dict)
    metric: str = "return"
    step: int = 0

    def as_dict(self, include_episodes: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "method": self.method,
            "mean": self.mean,
            "std": self.std,
            "half_width": self.half_width,
            "n": self.n,
            "confidence": self.confidence,
            "metric": self.metric,
            "step": self.step,
            "quantiles": dict(self.quantiles),
        }
        if self.values:
            payload["values"] = list(self.values)
        if include_episodes and self.episode_values:
            payload["episode_values"] = list(self.episode_values)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReturnDistribution":
        payload = dict(payload or {})
        values = as_float_list(payload.get("values") or payload.get("per_seed") or [])
        if not values and _is_number(payload.get("mean")):
            values = [float(payload["mean"])]
        return cls(
            method=str(payload.get("method", "none")),
            values=values,
            episode_values=as_float_list(payload.get("episode_values") or []),
            per_seed=as_float_list(payload.get("per_seed") or []),
            mean=float(payload.get("mean", mean_of(values))),
            std=float(payload.get("std", std_of(values))),
            half_width=float(payload.get("half_width", float("nan"))),
            n=int(payload.get("n", len(values))),
            confidence=float(payload.get("confidence", DEFAULT_CONFIDENCE)),
            quantiles=dict(payload.get("quantiles") or {}),
            metric=str(payload.get("metric", "return")),
            step=int(payload.get("step", 0) or 0),
        )

    @classmethod
    def from_tracker(cls, tracker: ReturnTracker, final_only: bool = True,
                     confidence: float = DEFAULT_CONFIDENCE) -> "ReturnDistribution":
        if final_only:
            records = [tracker.final()] if tracker.final() else []
        else:
            records = tracker.records
        records = [r for r in records if r is not None]
        values, episodes = [], []
        for rec in records:
            values.append(rec.effective_mean)
            episodes.extend(rec.returns)
        payload = summarize(values, confidence=confidence)
        return cls(
            method=tracker.method,
            values=values,
            episode_values=episodes,
            per_seed=list(values),
            mean=payload["mean"],
            std=payload["std"],
            half_width=payload["half_width"],
            n=payload["n"],
            confidence=confidence,
            quantiles=percentiles(episodes or values),
            metric=tracker.metric,
            step=tracker.last_step,
        )

    @classmethod
    def from_summaries(cls, summaries: Iterable[Union[Mapping[str, Any], ReturnTracker, "ReturnDistribution"]],
                       method: Optional[str] = None, confidence: float = DEFAULT_CONFIDENCE,
                       use_episodes: bool = False) -> "ReturnDistribution":
        """Aggregate per-seed summaries into a method-level distribution."""
        values: List[float] = []
        episodes: List[float] = []
        name = method
        step = 0
        max_n = 0
        for item in summaries:
            if isinstance(item, ReturnTracker):
                name = name or item.method
                rec = item.final()
                if rec is None:
                    continue
                values.append(rec.effective_mean)
                episodes.extend(rec.returns)
                step = max(step, rec.step)
                max_n = max(max_n, rec.effective_n)
            elif isinstance(item, ReturnDistribution):
                name = name or item.method
                values.extend(item.values or [item.mean])
                episodes.extend(item.episode_values)
                step = max(step, item.step)
                max_n = max(max_n, item.n)
            else:
                payload = dict(item or {})
                name = name or str(payload.get("method", "none"))
                for key in EPISODE_KEYS:
                    if payload.get(key):
                        episodes.extend(as_float_list(payload[key]))
                mean_value = None
                for key in ("mean_return", "final_mean", "eval_return", "return", "mean", "score"):
                    if _is_number(payload.get(key)):
                        mean_value = float(payload[key])
                        break
                if mean_value is None:
                    mean_value = mean_of(episodes[-1:]) if episodes else float("nan")
                if _is_number(mean_value):
                    values.append(mean_value)
                step = max(step, int(payload.get("step", 0) or 0))
                n_val = payload.get("n_episodes", payload.get("num_episodes"))
                if _is_number(n_val):
                    max_n = max(max_n, int(n_val))
        payload = summarize(values, confidence=confidence)
        return cls(
            method=name or "none",
            values=values,
            episode_values=episodes,
            per_seed=list(values),
            mean=payload["mean"],
            std=payload["std"],
            half_width=payload["half_width"],
            n=payload["n"],
            confidence=confidence,
            quantiles=percentiles(episodes if use_episodes and episodes else values),
            step=step,
            metadata_n=max_n,  # type: ignore[call-arg]
        ) if False else cls(
            method=name or "none",
            values=values,
            episode_values=episodes,
            per_seed=list(values),
            mean=payload["mean"],
            std=payload["std"],
            half_width=payload["half_width"],
            n=payload["n"],
            confidence=confidence,
            quantiles=percentiles(episodes if use_episodes and episodes else values),
            step=step,
        )


# --------------------------------------------------------------------------- #
# Aggregation across seeds / methods
# --------------------------------------------------------------------------- #


def aggregate_seed_returns(seed_values: Any, confidence: float = DEFAULT_CONFIDENCE) -> Dict[str, float]:
    """Mean + CI over seeds (each seed contributes its final return)."""
    return summarize(seed_values, confidence=confidence)


def aggregate_return_records(records: Iterable[Union[ReturnRecord, Mapping[str, Any]]],
                             confidence: float = DEFAULT_CONFIDENCE,
                             step: Optional[Union[int, str]] = None,
                             method: Optional[str] = None) -> ReturnDistribution:
    """Aggregate records of one method into a :class:`ReturnDistribution`.

    ``step`` selects a specific evaluation step; ``"final"`` (or ``None``) picks
    the deepest step recorded for each seed, which is what the paper's main
    figures report.
    """
    parsed: List[ReturnRecord] = []
    for rec in records:
        parsed.append(rec if isinstance(rec, ReturnRecord) else ReturnRecord.from_dict(rec))
    if method is not None:
        parsed = [r for r in parsed if r.method == method]
    if not parsed:
        return ReturnDistribution(method=method or "none", confidence=confidence)

    if step == "final" or step is None:
        by_seed: Dict[int, ReturnRecord] = {}
        for rec in parsed:
            prev = by_seed.get(rec.seed)
            if prev is None or rec.step >= prev.step:
                by_seed[rec.seed] = rec
        selected = list(by_seed.values())
    else:
        selected = [r for r in parsed if r.step == int(step)]
        if not selected:
            selected = parsed

    values = [r.effective_mean for r in selected]
    episodes: List[float] = []
    for rec in selected:
        episodes.extend(rec.returns)
    payload = summarize(values, confidence=confidence)
    return ReturnDistribution(
        method=(method or selected[0].method),
        values=values,
        episode_values=episodes,
        per_seed=list(values),
        mean=payload["mean"],
        std=payload["std"],
        half_width=payload["half_width"],
        n=payload["n"],
        confidence=confidence,
        quantiles=percentiles(episodes if episodes else values),
        metric=selected[0].metric,
        step=max(r.step for r in selected),
    )


def aggregate_methods(records: Iterable[Union[ReturnRecord, Mapping[str, Any]]],
                      methods: Optional[Sequence[str]] = None,
                      confidence: float = DEFAULT_CONFIDENCE,
                      step: Optional[Union[int, str]] = "final") -> "OrderedDict[str, ReturnDistribution]":
    """Aggregate a mixed record collection into ``{method: ReturnDistribution}``."""
    parsed = [r if isinstance(r, ReturnRecord) else ReturnRecord.from_dict(r) for r in records]
    if not methods:
        seen: List[str] = []
        for rec in parsed:
            if rec.method not in seen:
                seen.append(rec.method)
        methods = tuple(sorted(seen, key=lambda m: (METHOD_ORDER.index(m) if m in METHOD_ORDER
                                                    else len(METHOD_ORDER), m)))
    out: "OrderedDict[str, ReturnDistribution]" = OrderedDict()
    for method in methods:
        out[method] = aggregate_return_records(parsed, confidence=confidence, step=step, method=method)
    return out


def compare_methods(distributions: Mapping[str, Union[ReturnDistribution, Any]],
                    reference: str = "scratch",
                    confidence: float = DEFAULT_CONFIDENCE) -> Dict[str, Dict[str, float]]:
    """Pairwise comparisons of every method against a reference.

    Reports the absolute difference of means, the relative improvement, a Welch
    t-test and Cohen's *d*.  This is how the paper's claims
    ("retention > vanilla > scratch") are substantiated in the appendix tables.
    """
    ref = distributions.get(reference)
    ref_values = as_float_list(getattr(ref, "values", ref)) if ref is not None else []
    out: Dict[str, Dict[str, float]] = {}
    for name, dist in distributions.items():
        values = as_float_list(getattr(dist, "values", dist))
        if not values:
            continue
        stats = summarize(values, confidence=confidence)
        payload: Dict[str, float] = {
            "mean": stats["mean"],
            "half_width": stats["half_width"],
            "n": float(stats["n"]),
        }
        if ref_values and name != reference:
            diff = mean_of(values) - mean_of(ref_values)
            base = mean_of(ref_values)
            payload["diff_vs_reference"] = diff
            payload["relative_improvement"] = diff / abs(base) if base else float("nan")
            payload["cohens_d"] = cohens_d(values, ref_values)
            payload.update({f"ttest_{k}": v for k, v in welch_ttest(values, ref_values).items()})
        out[name] = payload
    return out


def method_ordering(distributions: Mapping[str, Union[ReturnDistribution, float]],
                    descending: bool = True) -> List[str]:
    """Order methods by mean return (used to check the paper's ordering claims)."""
    pairs = []
    for name, dist in distributions.items():
        value = dist if _is_number(dist) else getattr(dist, "mean", float("nan"))
        pairs.append((name, float(value)))
    pairs.sort(key=lambda kv: kv[1], reverse=descending)
    return [name for name, _ in pairs]


def matches_paper_ordering(distributions: Mapping[str, Union[ReturnDistribution, float]],
                           env: str = "nethack") -> Dict[str, Any]:
    """Loose check of the paper's qualitative ordering for the given env.

    NetHack:   KS > BC > EWC > vanilla  (and all retention > vanilla)
    Montezuma: BC/EWC > scratch, BC separates from vanilla after ~20M steps
    RoboticSequence: BC > EM/EWC > vanilla  (vanilla ~ scratch)
    """
    order = method_ordering(distributions)
    expected = {
        "nethack": ["ks", "bc", "ewc", "none"],
        "montezuma": ["bc", "ewc", "none", "scratch"],
        "robotic_sequence": ["bc", "em", "ewc", "none"],
    }.get(env, [])
    present = [m for m in expected if m in order]
    observed = [m for m in order if m in present]
    return {
        "env": env,
        "observed_order": observed,
        "expected_order": present,
        "matches": observed == present if len(present) > 1 else None,
    }


# --------------------------------------------------------------------------- #
# IO: discovery of trainer outputs
# --------------------------------------------------------------------------- #


def _get_field(payload: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return default


def return_record_from_summary(payload: Mapping[str, Any], method: Optional[str] = None,
                               seed: Optional[int] = None, step: Optional[int] = None) -> ReturnRecord:
    """Build a :class:`ReturnRecord` from a trainer ``summary.json`` payload."""
    payload = dict(payload or {})
    rec = ReturnRecord.from_dict(payload)
    if method is not None:
        rec.method = method
    if seed is not None:
        rec.seed = int(seed)
    if step is not None:
        rec.step = int(step)
    return rec


def iter_seed_files(results_dir: str, methods: Optional[Sequence[str]] = None,
                    filenames: Sequence[str] = ("summary.json", "returns.json", "return_distribution.json"),
                    layout: str = "auto") -> Iterable[Tuple[str, int, str]]:
    """Yield ``(method, seed, path)`` triples discovered under ``results_dir``.

    Supported layouts::

        <results_dir>/<method>/seed_<seed>/<file>
        <results_dir>/seed_<seed>/<method>/<file>
        <results_dir>/<method>/<seed>/<file>
        <results_dir>/<method>_seed<seed>/<file>
        <results_dir>/<method>/<file>            (single seed, seed=0)
        <results_dir>/<prefix>/<method>/seed_<seed>/<file>   (prefix ablations)
    """
    patterns: List[str] = []
    for name in filenames:
        patterns.extend([
            os.path.join(results_dir, "*", "seed_*", name),
            os.path.join(results_dir, "*", "*", "seed_*", name),
            os.path.join(results_dir, "*", "seed*", "*", name),
            os.path.join(results_dir, "*", "*", name),
            os.path.join(results_dir, "*_seed*", name),
        ])
    seen: set = set()
    files: List[str] = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            real = os.path.realpath(path)
            if real in seen:
                continue
            seen.add(real)
            files.append(path)

    for path in files:
        parts = os.path.normpath(path).split(os.sep)
        method, seed = None, None
        for part in parts:
            lowered = part.lower()
            if lowered.startswith("seed_") or lowered.startswith("seed"):
                digits = "".join(ch for ch in part if ch.isdigit())
                if digits:
                    seed = int(digits)
        method_seed_names = {m for m in (methods or METHOD_ORDER)}
        for part in parts:
            base = part.replace("-", "_").lower()
            for cand in method_seed_names:
                if base == cand.replace("-", "_").lower():
                    method = cand
        if method is None:
            # derive from a "<method>_seed<k>" directory or the file's parent
            parent = os.path.basename(os.path.dirname(path))
            if "_seed" in parent:
                method = parent.split("_seed")[0]
            elif parent not in {"seed_0", "seed_1", ""} and not parent.startswith("seed"):
                method = parent
            else:
                grand = os.path.basename(os.path.dirname(os.path.dirname(path)))
                method = grand if grand and not grand.startswith("seed") else "none"
        if seed is None:
            seed = 0
        if methods and method not in methods:
            continue
        yield method, seed, path


def load_return_records(results_dir: str, methods: Optional[Sequence[str]] = None,
                        include_final: bool = True) -> List[ReturnRecord]:
    """Load all return records found under ``results_dir``."""
    records: List[ReturnRecord] = []
    if not os.path.isdir(results_dir):
        return records
    for method, seed, path in iter_seed_files(results_dir, methods=methods):
        try:
            with open(path, "r") as handle:
                payload = json.load(handle)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        found = _records_from_payload(payload, method=method, seed=seed)
        if not found and include_final:
            rec = return_record_from_summary(payload, method=method, seed=seed)
            if _is_number(rec.mean_return) or rec.returns:
                found = [rec]
        records.extend(found)
    return records


def _records_from_payload(payload: Mapping[str, Any], method: str, seed: int) -> List[ReturnRecord]:
    """Extract return records from a summary payload (several known schemas)."""
    out: List[ReturnRecord] = []
    for key in ("returns", "return_records", "return_distribution", "eval_returns"):
        block = payload.get(key)
        if isinstance(block, list) and block and isinstance(block[0], dict):
            for item in block:
                rec = ReturnRecord.from_dict(item)
                rec.method = method
                rec.seed = seed if not item.get("seed") else rec.seed
                out.append(rec)
            return out
    history = payload.get("history") or payload.get("eval_history")
    if isinstance(history, list):
        for item in history:
            if not isinstance(item, dict):
                continue
            has_value = any(_is_number(item.get(k)) for k in SUMMARY_KEYS)
            if not has_value:
                continue
            rec = ReturnRecord.from_dict(item)
            rec.method = method
            rec.seed = seed
            out.append(rec)
        if out:
            return out
    # single-evaluation summary
    rec = ReturnRecord.from_dict(payload)
    rec.method = method
    rec.seed = seed
    if _is_number(rec.mean_return) or rec.returns:
        out.append(rec)
    return out


def collect_return_summaries(results_dir: str, methods: Optional[Sequence[str]] = None,
                             confidence: float = DEFAULT_CONFIDENCE,
                             step: Optional[Union[int, str]] = "final") -> "OrderedDict[str, ReturnDistribution]":
    """One-shot ``results_dir`` -> ``{method: ReturnDistribution}`` helper."""
    records = load_return_records(results_dir, methods=methods)
    return aggregate_methods(records, methods=methods, confidence=confidence, step=step)


def return_table(distributions: Mapping[str, Union[ReturnDistribution, Mapping[str, Any]]],
                 reference: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Compact, JSON-serialisable table of the per-method return statistics."""
    table: Dict[str, Dict[str, Any]] = {}
    for name, dist in distributions.items():
        if isinstance(dist, ReturnDistribution):
            payload = dist.as_dict()
        else:
            payload = dict(dist or {})
        entry = {
            "mean": payload.get("mean"),
            "std": payload.get("std"),
            "half_width": payload.get("half_width"),
            "n": payload.get("n"),
            "quantiles": payload.get("quantiles", {}),
        }
        table[name] = entry
    if reference and reference in table:
        base = table[reference]["mean"]
        for name, entry in table.items():
            if name == reference or not _is_number(entry.get("mean")) or not base:
                continue
            entry["relative_to_reference"] = float(entry["mean"] / base)
    return table


def format_table(table: Mapping[str, Mapping[str, Any]], confidence: float = DEFAULT_CONFIDENCE) -> str:
    if not table:
        return "(no data)"
    width = max(len(str(k)) for k in table)
    lines = [f"{'method'.ljust(width)}  {'mean':>12}  {'+/-':>10}  {'n':>4}"]
    for name, entry in table.items():
        mean = entry.get("mean")
        half = entry.get("half_width")
        n = entry.get("n")
        mean_s = f"{mean:12.2f}" if _is_number(mean) else f"{'nan':>12}"
        half_s = f"{half:10.2f}" if _is_number(half) else f"{'nan':>10}"
        lines.append(f"{str(name).ljust(width)}  {mean_s}  {half_s}  {str(n):>4}")
    lines.append(f"(confidence intervals at {int(confidence * 100)}%)")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Plotting (matplotlib optional / lazy)
# --------------------------------------------------------------------------- #


def _import_pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        return plt
    except Exception:  # pragma: no cover - plotting is optional
        return None


def _save_figure(fig: Any, path: Optional[str]) -> Any:
    if fig is not None and path:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        try:
            fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        except Exception:
            pass
    return fig


def plot_return_distribution(distributions: Union[Mapping[str, Any], Any],
                             path: Optional[str] = None,
                             bins: int = DEFAULT_BINS,
                             kind: str = "violin",
                             confidence: float = DEFAULT_CONFIDENCE,
                             reference: Optional[str] = None,
                             title: str = "return distribution",
                             xlabel: str = "return",
                             ylabel: Optional[str] = None,
                             order: Optional[Sequence[str]] = None,
                             horizontal: bool = True,
                             show: bool = False,
                             ax: Any = None,
                             colormap: str = DEFAULT_COLORMAP,
                             **kwargs: Any) -> Any:
    """Per-method return distribution (Figure 3 style).

    ``kind`` selects ``"violin"`` (default, shows the spread of the evaluation
    episodes), ``"box"``, ``"hist"`` (overlaid histograms) or ``"bar"`` (mean
    with a confidence interval).  When only per-seed means are available the
    function transparently degrades to a bar chart.
    """
    plt = _import_pyplot()
    if plt is None:  # pragma: no cover
        return None
    if isinstance(distributions, (list, tuple)):
        # Accept a sequence of ReturnDistribution / dicts.
        distributions = OrderedDict(
            (getattr(d, "method", None) or (d.get("method") if isinstance(d, Mapping) else str(i)), d)
            for i, d in enumerate(distributions))
    elif not isinstance(distributions, Mapping):
        distributions = {getattr(distributions, "method", "method"): distributions}

    names = list(order) if order else list(distributions.keys())

    def _samples(dist: Any) -> List[float]:
        if isinstance(dist, ReturnDistribution):
            return dist.episode_values or dist.values
        if isinstance(dist, Mapping):
            for key in EPISODE_KEYS + ("values", "per_seed"):
                if dist.get(key):
                    return as_float_list(dist[key])
            return []
        return as_float_list(getattr(dist, "episode_values", None) or
                             getattr(dist, "values", None))

    def _mean_ci(dist: Any) -> Tuple[float, float]:
        if isinstance(dist, ReturnDistribution):
            return dist.mean, (dist.half_width if _is_number(dist.half_width)
                               else summarize(dist.values, confidence)["half_width"])
        if isinstance(dist, Mapping):
            values = as_float_list(dist.get("values") or dist.get("per_seed") or [])
            stats = summarize(values, confidence)
            mean = dist.get("mean", stats["mean"])
            return float(mean), float(dist.get("half_width", stats["half_width"]))
        values = as_float_list(getattr(dist, "values", []))
        stats = summarize(values, confidence)
        return float(getattr(dist, "mean", stats["mean"])), stats["half_width"]

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, max(2.4, 0.85 * len(names) + 1.6)))
    colors = None
    try:
        cmap = plt.get_cmap(colormap)
        colors = [cmap(i / max(len(names) - 1, 1)) for i in range(len(names))]
    except Exception:
        colors = None

    samples = [_samples(distributions[n]) for n in names]
    means = [_mean_ci(distributions[n]) for n in names]
    labels = [METHOD_LABELS.get(n, n) for n in names]

    if kind in ("violin", "box") and any(len(s) > 1 for s in samples):
        positions = list(range(1, len(names) + 1))
        plot_samples = [s if len(s) > 1 else [means[i][0], means[i][0]] for i, s in enumerate(samples)]
        if horizontal:
            parts = (ax.violinplot(plot_samples, positions=positions, vert=False, showmeans=True,
                                   showextrema=True, widths=0.8)
                     if kind == "violin" else
                     ax.boxplot(plot_samples, positions=positions, vert=False, patch_artist=True,
                                widths=0.6))
        else:
            parts = (ax.violinplot(plot_samples, positions=positions, showmeans=True,
                                   showextrema=True, widths=0.8)
                     if kind == "violin" else
                     ax.boxplot(plot_samples, positions=positions, patch_artist=True, widths=0.6))
        bodies = parts.get("bodies") if isinstance(parts, dict) else None
        if bodies is not None and colors is not None:
            for body, color in zip(bodies, colors):
                try:
                    body.set_facecolor(color)
                    body.set_alpha(0.7)
                except Exception:
                    pass
        elif isinstance(parts, dict) and "boxes" in parts and colors is not None:
            for box, color in zip(parts["boxes"], colors):
                try:
                    box.set_facecolor(color)
                except Exception:
                    pass
        positions_axis = positions
        setter = ax.set_yticks if horizontal else ax.set_xticks
        setter(positions_axis, labels)
        if horizontal:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel or "")
        else:
            ax.set_ylabel(xlabel)
            ax.set_xlabel(ylabel or "")
    elif kind == "hist":
        all_values = [v for s in samples for v in s] or [m[0] for m in means]
        lo, hi = min(all_values), max(all_values)
        edges = [lo + (hi - lo) * i / max(bins, 1) for i in range(bins + 1)] if hi > lo else None
        for i, name in enumerate(names):
            values = samples[i] or [means[i][0]]
            try:
                ax.hist(values, bins=edges or bins, alpha=0.55, density=False,
                        label=labels[i], color=(colors[i] if colors else None), **kwargs)
            except Exception:
                continue
        ax.legend(fontsize=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel or "episodes")
    else:  # bar chart of the mean with CI
        ypos = list(range(len(names)))
        values = [m[0] for m in means]
        errors = [m[1] for m in means]
        ax.barh(ypos, values, xerr=errors, color=colors, alpha=0.85) if horizontal else \
            ax.bar(ypos, values, yerr=errors, color=colors, alpha=0.85)
        if horizontal:
            ax.set_yticks(ypos, labels)
            ax.set_xlabel(xlabel)
        else:
            ax.set_xticks(ypos, labels, rotation=20)
            ax.set_ylabel(xlabel)

    if reference and reference in distributions:
        ref_mean, _ = _mean_ci(distributions[reference])
        if _is_number(ref_mean):
            try:
                if kind not in ("violin", "box"):
                    (ax.axvline if horizontal else ax.axhline)(ref_mean, ls="--", color="gray", lw=1.2,
                                                              label=METHOD_LABELS.get(reference, reference))
                    ax.legend(fontsize=8)
            except Exception:
                pass
    ax.set_title(title)
    if path or show:
        try:
            fig.tight_layout()
        except Exception:
            pass
    _save_figure(fig, path)
    if show:
        try:
            plt.show()
        except Exception:
            pass
    return fig


def plot_return_violin(distributions: Union[Mapping[str, Any], Any], path: Optional[str] = None,
                       **kwargs: Any) -> Any:
    """Alias of :func:`plot_return_distribution` with ``kind="violin"``."""
    kwargs.setdefault("kind", "violin")
    return plot_return_distribution(distributions, path=path, **kwargs)


def plot_return_box(distributions: Union[Mapping[str, Any], Any], path: Optional[str] = None,
                    **kwargs: Any) -> Any:
    kwargs.setdefault("kind", "box")
    return plot_return_distribution(distributions, path=path, **kwargs)


def plot_return_grid(distributions: Union[Mapping[str, Any], Any], path: Optional[str] = None,
                     kinds: Sequence[str] = ("violin", "bar"), ncols: int = 2,
                     suptitle: Optional[str] = None, show: bool = False, **kwargs: Any) -> Any:
    """Grid with several views of the same per-method return distributions."""
    plt = _import_pyplot()
    if plt is None:  # pragma: no cover
        return None
    kinds = list(kinds)
    nrows = int(math.ceil(len(kinds) / max(ncols, 1)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 3.6 * nrows), squeeze=False)
    for i, kind in enumerate(kinds):
        ax = axes[i // ncols][i % ncols]
        plot_return_distribution(distributions, kind=kind, ax=ax, show=False,
                                 title=kind, **kwargs)
    for j in range(len(kinds), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    if suptitle:
        fig.suptitle(suptitle)
    try:
        fig.tight_layout()
    except Exception:
        pass
    _save_figure(fig, path)
    if show:
        try:
            plt.show()
        except Exception:
            pass
    return fig


def plot_learning_curve_returns(curves: Mapping[str, Any], path: Optional[str] = None,
                                confidence: float = DEFAULT_CONFIDENCE,
                                xlabel: str = "environment steps", ylabel: str = "return",
                                title: str = "fine-tuning return", log_x: bool = False,
                                show: bool = False, ax: Any = None,
                                reference: Optional[float] = None,
                                order: Optional[Sequence[str]] = None) -> Any:
    """Mean evaluation return vs. environment steps per method, with CI bands.

    ``curves`` may map a method to a :class:`ReturnTracker`, a list of
    ``(step, value)`` pairs, a mapping with ``steps``/``means``, or a list of
    per-seed curves (in which case the mean curve and a normal-approx CI band are
    drawn).
    """
    plt = _import_pyplot()
    if plt is None:  # pragma: no cover
        return None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
    else:
        fig = None

    names = list(order) if order else list(curves.keys())
    try:
        cmap = plt.get_cmap(DEFAULT_COLORMAP)
        colors = [cmap(i / max(len(names) - 1, 1)) for i in range(len(names))]
    except Exception:
        colors = [None] * len(names)

    for idx, name in enumerate(names):
        entry = curves[name]
        steps, means, band = _curve_triplet(entry)
        if not steps:
            continue
        color = colors[idx]
        label = METHOD_LABELS.get(name, name)
        ax.plot(steps, means, color=color, label=label, lw=1.8)
        if band is not None:
            lo, hi = band
            try:
                ax.fill_between(steps, lo, hi, color=color, alpha=0.18, lw=0)
            except Exception:
                pass
    if reference is not None and _is_number(reference):
        ax.axhline(float(reference), ls="--", color="gray", lw=1.2, label="reference")
    if log_x:
        try:
            ax.set_xscale("log")
        except Exception:
            pass
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    if fig is not None:
        try:
            fig.tight_layout()
        except Exception:
            pass
    _save_figure(fig, path)
    if show:
        try:
            plt.show()
        except Exception:
            pass
    return fig


def _curve_triplet(entry: Any) -> Tuple[List[float], List[float], Optional[Tuple[List[float], List[float]]]]:
    """Normalise a curve entry into ``(steps, means, (lo, hi) or None)``."""
    if isinstance(entry, ReturnTracker):
        steps, values = entry.curve("mean")
        return steps, values, None
    if isinstance(entry, ReturnDistribution):
        return [float(entry.step)], [entry.mean], None
    if isinstance(entry, Mapping):
        steps = as_float_list(entry.get("steps") or entry.get("x") or [])
        means = as_float_list(entry.get("means") or entry.get("values") or entry.get("y") or [])
        lo = as_float_list(entry.get("lo") or [])
        hi = as_float_list(entry.get("hi") or [])
        if steps and means and len(steps) == len(means):
            return steps, means, (lo, hi) if lo and hi else None
        # seeds x steps matrix
        seeds = entry.get("curves") or entry.get("per_seed")
        if seeds:
            return _stack_seed_curves(seeds)
        return [], [], None
    if isinstance(entry, (list, tuple)):
        if entry and isinstance(entry[0], (list, tuple)) and len(entry[0]) == 2 and all(
                _is_number(v) for v in entry[0]):
            pairs = [(float(a), float(b)) for a, b in entry]
            pairs.sort(key=lambda p: p[0])
            return [p[0] for p in pairs], [p[1] for p in pairs], None
        if entry and isinstance(entry[0], (list, tuple)):
            return _stack_seed_curves(entry)
        values = as_float_list(entry)
        return [float(i) for i in range(len(values))], values, None
    return [], [], None


def _stack_seed_curves(seeds: Sequence[Any], confidence: float = DEFAULT_CONFIDENCE
                       ) -> Tuple[List[float], List[float], Optional[Tuple[List[float], List[float]]]]:
    per_seed: List[Tuple[List[float], List[float]]] = []
    for seed in seeds:
        if isinstance(seed, Mapping):
            steps = as_float_list(seed.get("steps") or [])
            values = as_float_list(seed.get("means") or seed.get("values") or [])
        elif isinstance(seed, ReturnTracker):
            steps, values = seed.curve("mean")
        else:
            steps, values = _curve_triplet(seed)[:2]
        if steps and values:
            per_seed.append((steps, values))
    if not per_seed:
        return [], [], None
    grid = sorted({s for steps, _ in per_seed for s in steps})
    means, los, his = [], [], []
    z = z_for(confidence)
    for step in grid:
        vals = [_interp(steps, values, step) for steps, values in per_seed]
        vals = [v for v in vals if not math.isnan(v)]
        if not vals:
            means.append(float("nan"))
            los.append(float("nan"))
            his.append(float("nan"))
            continue
        m = sum(vals) / len(vals)
        sd = std_of(vals)
        half = z * sd / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
        means.append(m)
        los.append(m - half)
        his.append(m + half)
    return grid, means, (los, his)


def _interp(steps: Sequence[float], values: Sequence[float], t: float) -> float:
    if not steps:
        return float("nan")
    if t <= steps[0]:
        return float(values[0])
    if t >= steps[-1]:
        return float(values[-1])
    for i in range(1, len(steps)):
        if steps[i] >= t:
            x0, x1 = steps[i - 1], steps[i]
            y0, y1 = values[i - 1], values[i]
            if x1 == x0:
                return float(y1)
            return float(y0 + (y1 - y0) * (t - x0) / (x1 - x0))
    return float(values[-1])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate and plot per-method return distributions "
                    "(Figure 3 / Appendix D-E of the reproduction plan).")
    parser.add_argument("--results-dir", default=None, help="root results directory")
    parser.add_argument("--methods", nargs="*", default=None,
                        help="retention variants to include (default: all found)")
    parser.add_argument("--reference", default="scratch",
                        help="reference method for the pairwise comparison table")
    parser.add_argument("--step", default="final",
                        help='"final" (default) or an evaluation step index')
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--env", default=None, help="environment name (for ordering checks)")
    parser.add_argument("--kind", default="violin",
                        choices=["violin", "box", "hist", "bar"])
    parser.add_argument("--output", default=None, help="figure output path")
    parser.add_argument("--json", default=None, help="JSON output path for the table")
    parser.add_argument("--show", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.results_dir:
        parser.print_help()
        return 0
    step: Union[int, str] = args.step
    if isinstance(step, str) and step.isdigit():
        step = int(step)
    distributions = collect_return_summaries(args.results_dir, methods=args.methods,
                                             confidence=args.confidence, step=step)
    table = return_table(distributions, reference=args.reference)
    print(format_table(table, confidence=args.confidence))
    comparison = compare_methods(distributions, reference=args.reference,
                                confidence=args.confidence)
    if comparison:
        print("\npairwise comparisons vs. %s:" % args.reference)
        for name, stats in comparison.items():
            if "diff_vs_reference" in stats:
                print("  %-8s diff=%+.2f  rel=%+.3f  d=%+.2f  p=%.3g" % (
                    name, stats["diff_vs_reference"], stats.get("relative_improvement", float("nan")),
                    stats.get("cohens_d", float("nan")), stats.get("ttest_p_value", float("nan"))))
    if args.env:
        print("\nordering check:", json.dumps(matches_paper_ordering(distributions, args.env)))
    if args.json:
        directory = os.path.dirname(os.path.abspath(args.json))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.json, "w") as handle:
            json.dump({"table": table, "comparison": comparison}, handle, indent=2)
    if args.output:
        plot_return_distribution(distributions, path=args.output, kind=args.kind,
                                 confidence=args.confidence, title=args.env or "return distribution",
                                 show=args.show)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ReturnRecord",
    "ReturnTracker",
    "ReturnDistribution",
    "aggregate_seed_returns",
    "aggregate_return_records",
    "aggregate_methods",
    "compare_methods",
    "method_ordering",
    "matches_paper_ordering",
    "load_return_records",
    "collect_return_summaries",
    "iter_seed_files",
    "return_record_from_summary",
    "return_table",
    "format_table",
    "plot_return_distribution",
    "plot_return_violin",
    "plot_return_box",
    "plot_return_grid",
    "plot_learning_curve_returns",
    "histogram",
    "percentiles",
    "summarize",
    "bootstrap_ci",
    "welch_ttest",
    "cohens_d",
    "z_for",
    "main",
    "build_parser",
    "METHOD_ORDER",
    "METHOD_LABELS",
    "PAPER_REFERENCES",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_BINS",
    "EVAL_EVERY",
    "NUM_EVAL_EPISODES",
]
