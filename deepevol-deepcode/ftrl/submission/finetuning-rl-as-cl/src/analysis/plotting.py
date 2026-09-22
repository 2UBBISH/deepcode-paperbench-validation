"""Generic plotting utilities shared by all analysis modules.

This module implements the *figure-level* plotting machinery used to produce
the paper's figures (Wołczyk et al., 2024, "Fine-tuning Reinforcement Learning
Models is Secretly a Forgetting Mitigation Problem"):

* learning curves averaged over seeds with confidence intervals (Figures 3, 5),
* per-stage success-rate bars (Figure 7),
* prefix-length / forward-transfer tables (Table 6, Figure 22),
* knowledge-retention / forgetting traces (Figure 8),
* support for the density plots (Figure 4) and return distributions (Figure 3).

Everything that depends on ``matplotlib`` is imported lazily so the module can
be imported (and its metric bookkeeping used) in a headless, matplotlib-free
environment.  The numeric helpers (``mean_ci``, ``bootstrap_ci``, ``z_for``,
``moving_average``, ``smooth_curve``, ``interpolate``) have no third-party
dependency beyond NumPy (optional; pure-Python fallbacks are provided).

The paper reports 90% confidence intervals over at least 20 seeds for the
RoboticSequence experiments and 1000-episode evaluations for NetHack.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None  # type: ignore

__all__ = [
    # styles / constants
    "DEFAULT_CONFIDENCE",
    "METHOD_STYLES",
    "METHOD_ORDER",
    "METHOD_LABELS",
    "FIGURE_DPI",
    "method_style",
    "set_plot_style",
    "get_cmap_colors",
    # statistics
    "z_for",
    "mean_ci",
    "bootstrap_ci",
    "summarize",
    "aggregate_curves",
    "CurveStats",
    # curve helpers
    "moving_average",
    "smooth_curve",
    "interpolate",
    "resample_curve",
    "pad_curve",
    "downsample_curve",
    "normalize_curve",
    # core plots
    "plot_curve",
    "plot_curves",
    "plot_metric_grid",
    "plot_bar_comparison",
    "plot_success_rates",
    "plot_table_heatmap",
    "plot_forward_transfer",
    "plot_retention_summary",
    "save_figure",
    # io helpers
    "load_history",
    "load_summaries",
    "collect_histories",
    "main",
]


# --------------------------------------------------------------------------- #
# Constants / styles
# --------------------------------------------------------------------------- #

DEFAULT_CONFIDENCE: float = 0.90  # the paper uses 90% CIs
FIGURE_DPI: int = 150

#: order used for the retention-method comparisons throughout the paper
METHOD_ORDER: Tuple[str, ...] = (
    "scratch",
    "from_scratch",
    "none",
    "vanilla",
    "vanilla_fine_tuning",
    "ewc",
    "em",
    "em",
    "bc",
    "ks",
)

#: pretty labels for the retention methods
METHOD_LABELS: Dict[str, str] = {
    "scratch": "from scratch",
    "from_scratch": "from scratch",
    "none": "fine-tuning",
    "vanilla": "fine-tuning",
    "vanilla_fine_tuning": "fine-tuning",
    "ewc": "EWC",
    "em": "EM",
    "em": "EM",
    "bc": "BC",
    "ks": "KS",
}

#: (color, linestyle) per method.  Chosen to be colour-blind friendly.
METHOD_STYLES: Dict[str, Dict[str, Any]] = {
    "scratch": {"color": "#444444", "linestyle": "--", "marker": None},
    "from_scratch": {"color": "#444444", "linestyle": "--", "marker": None},
    "none": {"color": "#d62728", "linestyle": "-", "marker": "o"},
    "vanilla": {"color": "#d62728", "linestyle": "-", "marker": "o"},
    "vanilla_fine_tuning": {"color": "#d62728", "linestyle": "-", "marker": "o"},
    "ewc": {"color": "#1f77b4", "linestyle": "-", "marker": "s"},
    "em": {"color": "#9467bd", "linestyle": "-", "marker": "^"},
    "bc": {"color": "#2ca02c", "linestyle": "-", "marker": "D"},
    "ks": {"color": "#ff7f0e", "linestyle": "-", "marker": "v"},
}

_DEFAULT_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                   "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf")


def method_style(method: str, index: int = 0, **overrides: Any) -> Dict[str, Any]:
    """Return a matplotlib style dict for ``method`` (falling back to a palette)."""
    key = str(method).strip().lower().replace(" ", "_").replace("+", "")
    style = dict(METHOD_STYLES.get(key, {}))
    if not style:
        style = {"color": _DEFAULT_COLORS[index % len(_DEFAULT_COLORS)],
                 "linestyle": "-", "marker": None}
    style.update(overrides)
    return style


def set_plot_style(style: str = "paper", dpi: int = FIGURE_DPI) -> None:
    """Best-effort matplotlib rcParams styling; silently no-ops without matplotlib."""
    try:  # pragma: no cover - optional dependency
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover
        return
    if style == "paper":
        plt.rcParams.update({
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.autolayout": False,
        })
    else:  # pragma: no cover
        plt.rcParams.update({"figure.dpi": dpi, "savefig.dpi": dpi})


def get_cmap_colors(name: str, n: int) -> List[Any]:
    """Return ``n`` colours sampled from a matplotlib colormap as RGBA tuples."""
    n = max(1, int(n))
    try:  # pragma: no cover - optional dependency
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap(name)
        if n == 1:
            return [cmap(0.5)]
        return [cmap(i / (n - 1)) for i in range(n)]
    except Exception:  # pragma: no cover
        return [None] * n


# --------------------------------------------------------------------------- #
# Statistics helpers
# --------------------------------------------------------------------------- #

_Z_TABLE: Dict[float, float] = {
    0.50: 0.6745,
    0.68: 0.9945,
    0.80: 1.2816,
    0.90: 1.6449,
    0.95: 1.9600,
    0.98: 2.3263,
    0.99: 2.5758,
}


def z_for(confidence: float = DEFAULT_CONFIDENCE) -> float:
    """Two-sided normal quantile for a confidence level (0.90 -> 1.6449)."""
    confidence = float(confidence)
    for level, z in _Z_TABLE.items():
        if abs(confidence - level) < 1e-12:
            return z
    if confidence <= 0.0 or confidence >= 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if _np is not None:
        try:
            # ``np.percentile`` of the normal distribution == inverse CDF
            from math import erf

            target = 0.5 + confidence / 2.0
            lo, hi = 0.0, 40.0
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                cdf = 0.5 * (1.0 + erf(mid / math.sqrt(2.0)))
                if cdf < target:
                    lo = mid
                else:
                    hi = mid
            return 0.5 * (lo + hi)
        except Exception:  # pragma: no cover
            pass
    # Acklam rational approximation for the inverse normal CDF
    p = 0.5 + confidence / 2.0
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def _as_float_list(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for v in values:
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(fv):
            continue
        out.append(fv)
    return out


def mean_ci(values: Sequence[float], confidence: float = DEFAULT_CONFIDENCE) -> Dict[str, float]:
    """Normal-approximation confidence interval of the mean.

    Returns ``{"mean", "half_width", "std", "n", "lo", "hi"}``.  ``half_width``
    is the value used for the shaded bands in the paper's figures.
    """
    data = _as_float_list(values)
    n = len(data)
    if n == 0:
        return {"mean": float("nan"), "half_width": float("nan"), "std": float("nan"),
                "n": 0, "lo": float("nan"), "hi": float("nan")}
    mean = sum(data) / n
    if n == 1:
        return {"mean": mean, "half_width": 0.0, "std": 0.0, "n": 1,
                "lo": mean, "hi": mean}
    var = sum((x - mean) ** 2 for x in data) / (n - 1)
    std = math.sqrt(max(0.0, var))
    half = z_for(confidence) * std / math.sqrt(n)
    return {"mean": mean, "half_width": half, "std": std, "n": n,
            "lo": mean - half, "hi": mean + half}


def bootstrap_ci(values: Sequence[float], confidence: float = DEFAULT_CONFIDENCE,
                 num_samples: int = 2000, seed: int = 0,
                 statistic: str = "mean") -> Dict[str, float]:
    """Percentile bootstrap CI (used for non-normal metrics such as scores)."""
    data = _as_float_list(values)
    n = len(data)
    if n == 0:
        return {"mean": float("nan"), "half_width": float("nan"), "std": float("nan"),
                "n": 0, "lo": float("nan"), "hi": float("nan")}

    def _stat(sample: Sequence[float]) -> float:
        if statistic == "mean":
            return sum(sample) / len(sample)
        if statistic == "median":
            s = sorted(sample)
            mid = len(s) // 2
            return s[mid] if len(s) % 2 else 0.5 * (s[mid - 1] + s[mid])
        raise ValueError(f"unknown statistic: {statistic}")

    point = _stat(data)
    if n == 1:
        return {"mean": point, "half_width": 0.0, "std": 0.0, "n": 1,
                "lo": point, "hi": point}
    rng = _np.random.default_rng(seed) if _np is not None else None
    stats: List[float] = []
    for _ in range(int(num_samples)):
        if rng is not None:
            idx = rng.integers(0, n, size=n)
            sample = [data[int(i)] for i in idx]
        else:
            import random as _random

            sample = [_random.choice(data) for _ in range(n)]
        stats.append(_stat(sample))
    stats.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = stats[max(0, min(len(stats) - 1, int(alpha * len(stats))))]
    hi = stats[max(0, min(len(stats) - 1, int((1 - alpha) * len(stats)) - 1))]
    std = math.sqrt(sum((x - point) ** 2 for x in data) / (n - 1))
    return {"mean": point, "half_width": 0.5 * (hi - lo), "std": std, "n": n,
            "lo": lo, "hi": hi}


def summarize(values: Sequence[float], confidence: float = DEFAULT_CONFIDENCE,
              bootstrap: bool = False) -> Dict[str, float]:
    """Mean + confidence interval, using the bootstrap when requested."""
    if bootstrap:
        return bootstrap_ci(values, confidence=confidence)
    return mean_ci(values, confidence=confidence)


@dataclass
class CurveStats:
    """Aggregated learning curve (mean + CI band) over seeds."""

    steps: List[float] = field(default_factory=list)
    mean: List[float] = field(default_factory=list)
    lo: List[float] = field(default_factory=list)
    hi: List[float] = field(default_factory=list)
    std: List[float] = field(default_factory=list)
    n: List[int] = field(default_factory=list)
    confidence: float = DEFAULT_CONFIDENCE

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.steps)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "steps": list(self.steps),
            "mean": list(self.mean),
            "lo": list(self.lo),
            "hi": list(self.hi),
            "std": list(self.std),
            "n": list(self.n),
            "confidence": self.confidence,
        }


def _curve_pairs(curve: Any) -> Tuple[List[float], List[float]]:
    """Normalise the many history shapes into ``(steps, values)`` pairs."""
    if curve is None:
        return [], []
    if isinstance(curve, Mapping):
        for skey in ("steps", "step", "times", "time", "x", "env_steps", "global_step"):
            if skey in curve:
                steps = list(curve[skey])
                for vkey in ("values", "value", "success_rate", "returns", "score",
                             "mean_return", "y"):
                    if vkey in curve:
                        return _as_float_list(steps), _as_float_list(curve[vkey])
        # single summary dict -> single point
        for vkey in ("mean", "value", "success_rate", "return", "score"):
            if vkey in curve:
                return [float(curve.get("step", 0))], _as_float_list([curve[vkey]])
        return [], []
    pairs = list(curve)
    if not pairs:
        return [], []
    first = pairs[0]
    if isinstance(first, (list, tuple)) and len(first) >= 2:
        steps = _as_float_list(p[0] for p in pairs)
        vals = _as_float_list(p[1] for p in pairs)
        return steps, vals
    return [float(i) for i in range(len(pairs))], _as_float_list(pairs)


def aggregate_curves(curves: Sequence[Any], confidence: float = DEFAULT_CONFIDENCE,
                     grid: Optional[Sequence[float]] = None,
                     num_grid: int = 100, normalize_steps: bool = False) -> CurveStats:
    """Average several per-seed learning curves onto a common grid.

    Each curve may be a ``(steps, values)`` mapping, a list of ``(step, value)``
    tuples or a plain list of values (steps then default to the index).
    """
    prepared: List[Tuple[List[float], List[float]]] = []
    for curve in curves:
        steps, values = _curve_pairs(curve)
        if not values:
            continue
        if not steps or len(steps) != len(values):
            if normalize_steps and len(values) > 1:
                steps = [i / (len(values) - 1) for i in range(len(values))]
            else:
                steps = [float(i) for i in range(len(values))]
        prepared.append((steps, values))
    if not prepared:
        return CurveStats(confidence=confidence)

    if grid is None:
        t_min = min(s[0] for s, _ in prepared)
        t_max = max(s[-1] for s, _ in prepared)
        if t_max <= t_min:
            grid = [t_min]
        else:
            n = max(2, int(num_grid))
            grid = [t_min + (t_max - t_min) * i / (n - 1) for i in range(n)]
    grid = list(grid)

    stats = CurveStats(steps=list(grid), confidence=confidence)
    for t in grid:
        samples = [interpolate(steps, values, t) for steps, values in prepared]
        summary = mean_ci(samples, confidence=confidence)
        stats.mean.append(summary["mean"])
        stats.lo.append(summary["lo"])
        stats.hi.append(summary["hi"])
        stats.std.append(summary["std"])
        stats.n.append(int(summary["n"]))
    return stats


# --------------------------------------------------------------------------- #
# Curve manipulation helpers
# --------------------------------------------------------------------------- #

def moving_average(values: Sequence[float], window: int = 5) -> List[float]:
    """Causal (trailing) moving average used to smooth noisy training curves."""
    data = _as_float_list(values)
    window = max(1, int(window))
    if window == 1 or not data:
        return data
    out: List[float] = []
    running = 0.0
    for i, v in enumerate(data):
        running += v
        if i >= window:
            running -= data[i - window]
        out.append(running / min(i + 1, window))
    return out


def smooth_curve(steps: Sequence[float], values: Sequence[float],
                 num_points: int = 200) -> Tuple[List[float], List[float]]:
    """Resample a curve onto a uniform grid (piecewise-linear)."""
    s = _as_float_list(steps)
    v = _as_float_list(values)
    if len(s) != len(v) or not v:
        return [], []
    if len(v) == 1:
        return list(s), list(v)
    t_min, t_max = s[0], s[-1]
    n = max(2, int(num_points))
    grid = [t_min + (t_max - t_min) * i / (n - 1) for i in range(n)]
    return grid, [interpolate(s, v, t) for t in grid]


def interpolate(steps: Sequence[float], values: Sequence[float], t: float) -> float:
    """Piecewise-linear interpolation with clamped extrapolation."""
    s = list(steps)
    v = list(values)
    if not v:
        return float("nan")
    if len(v) == 1 or len(s) < 2:
        return float(v[0])
    if t <= s[0]:
        return float(v[0])
    if t >= s[-1]:
        return float(v[-1])
    lo, hi = 0, len(s) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if s[mid] <= t:
            lo = mid
        else:
            hi = mid
    span = s[hi] - s[lo]
    if span <= 0:
        return float(v[hi])
    w = (t - s[lo]) / span
    return float(v[lo] * (1.0 - w) + v[hi] * w)


def resample_curve(curve: Any, grid: Sequence[float]) -> List[float]:
    """Interpolate any supported curve shape onto ``grid``."""
    steps, values = _curve_pairs(curve)
    if not values:
        return [float("nan")] * len(grid)
    return [interpolate(steps, values, t) for t in grid]


def pad_curve(values: Sequence[float], length: Optional[int] = None,
              pad_value: Optional[float] = None) -> List[float]:
    """Right-pad a curve with its last value (e.g. NetHack 1000-episode runs)."""
    data = _as_float_list(values)
    if length is None or len(data) >= length:
        return data
    fill = pad_value if pad_value is not None else (data[-1] if data else float("nan"))
    return data + [fill] * (length - len(data))


def downsample_curve(steps: Sequence[float], values: Sequence[float],
                     every: int) -> Tuple[List[float], List[float]]:
    """Keep every ``every``-th point (used for rarely evaluated metrics)."""
    every = max(1, int(every))
    s = _as_float_list(steps)
    v = _as_float_list(values)
    n = min(len(s), len(v))
    return s[:n:every], v[:n:every]


def normalize_curve(values: Sequence[float], reference: Optional[float] = None) -> List[float]:
    """Divide a curve by a reference value (e.g. pre-trained log-likelihood)."""
    data = _as_float_list(values)
    ref = reference if reference is not None else (data[0] if data else 1.0)
    if not ref:
        return list(data)
    return [x / ref for x in data]


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def _plt():  # pragma: no cover - thin wrapper
    import matplotlib.pyplot as plt

    return plt


def _as_numeric(x: Any) -> Any:
    try:
        return _np.asarray(x, dtype=float) if _np is not None else x
    except Exception:  # pragma: no cover
        return x


def save_figure(fig: Any, path: str, dpi: int = FIGURE_DPI) -> Optional[str]:
    """Save a matplotlib figure, creating parent directories as needed."""
    if fig is None:
        return None
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
    try:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    except Exception:  # pragma: no cover
        return None
    return path


def plot_curve(steps: Sequence[float], values: Sequence[float],
               ax: Any = None, label: Optional[str] = None,
               color: Any = None, linestyle: str = "-", marker: Any = None,
               ci: Optional[Mapping[str, Sequence[float]]] = None,
               alpha: float = 0.2, log_x: bool = False,
               xlabel: Optional[str] = None, ylabel: Optional[str] = None,
               title: Optional[str] = None, smooth: int = 0,
               path: Optional[str] = None, show: bool = False,
               xlim: Optional[Tuple[float, float]] = None,
               ylim: Optional[Tuple[float, float]] = None,
               marker_every: int = 1, **plot_kwargs: Any) -> Any:
    """Plot a single curve, optionally with a CI band (Figure 3/5 style)."""
    plt = _plt()
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=plot_kwargs.pop("figsize", (5.5, 3.5)))
    else:
        fig = ax.figure
    s = _as_float_list(steps)
    v = _as_float_list(values)
    if smooth and smooth > 1:
        s, v = smooth_curve(s, v, num_points=smooth)
    kwargs: Dict[str, Any] = {"label": label, "linestyle": linestyle}
    if color is not None:
        kwargs["color"] = color
    if marker is not None:
        kwargs["marker"] = marker
        kwargs["markevery"] = max(1, int(marker_every))
    kwargs.update(plot_kwargs)
    line, = ax.plot(_as_numeric(s), _as_numeric(v), **kwargs)
    if ci is not None:
        lo = _as_float_list(ci.get("lo", []))
        hi = _as_float_list(ci.get("hi", []))
        csteps = _as_float_list(ci.get("steps", s))
        if lo and hi and len(lo) == len(hi):
            band_color = kwargs.get("color", None) or line.get_color()
            ax.fill_between(_as_numeric(csteps), _as_numeric(lo), _as_numeric(hi),
                            color=band_color, alpha=alpha, linewidth=0)
    if log_x:
        ax.set_xscale("log")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    if label:
        ax.legend(loc="best", frameon=False)
    if path:
        save_figure(fig, path)
    if show:  # pragma: no cover
        plt.show()
    return fig if created else ax


def plot_curves(curves: Mapping[str, Any],
                confidence: float = DEFAULT_CONFIDENCE,
                grid: Optional[Sequence[float]] = None,
                xlabel: str = "environment steps",
                ylabel: str = "return",
                title: Optional[str] = None,
                log_x: bool = False,
                path: Optional[str] = None,
                show: bool = False,
                normalize_steps: bool = False,
                num_grid: int = 100,
                methods_order: Optional[Sequence[str]] = None,
                **plot_kwargs: Any) -> Any:
    """Plot several per-method seed collections with 90% CI bands.

    ``curves`` maps a method name to either:

    * a list of per-seed learning curves (each a ``(steps, values)`` mapping,
      a list of ``(step, value)`` tuples or a plain value list), or
    * a single learning curve.
    """
    plt = _plt()
    fig, ax = plt.subplots(figsize=plot_kwargs.pop("figsize", (6.0, 4.0)))
    names = list(curves.keys())
    if methods_order:
        ordered = [m for m in methods_order if m in curves]
        ordered += [m for m in names if m not in ordered]
        names = ordered
    else:
        names = sorted(names, key=lambda m: (METHOD_ORDER.index(m)
                                             if m in METHOD_ORDER else len(METHOD_ORDER)
                                             ) * 100 + names.index(m))
    for i, name in enumerate(names):
        payload = curves[name]
        is_collection = isinstance(payload, (list, tuple)) and payload and not (
            isinstance(payload[0], (int, float)))
        style = method_style(name, i)
        label = METHOD_LABELS.get(str(name).lower(), str(name))
        if is_collection:
            stats = aggregate_curves(payload, confidence=confidence, grid=grid,
                                     num_grid=num_grid,
                                     normalize_steps=normalize_steps)
            if not len(stats):
                continue
            plot_curve(stats.steps, stats.mean, ax=ax, label=label,
                       ci=stats.as_dict(), **{**style, **plot_kwargs})
        else:
            steps, values = _curve_pairs(payload)
            if not values:
                continue
            plot_curve(steps, values, ax=ax, label=label, **{**style, **plot_kwargs})
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(loc="best", frameon=False)
    if path:
        save_figure(fig, path)
    if show:  # pragma: no cover
        plt.show()
    return fig


def plot_metric_grid(histories: Mapping[str, Any],
                     metrics: Sequence[str] = ("success_rate", "return"),
                     confidence: float = DEFAULT_CONFIDENCE,
                     ncols: int = 2, path: Optional[str] = None,
                     show: bool = False,
                     xlabel: str = "environment steps", **plot_kwargs: Any) -> Any:
    """Grid of per-metric comparison plots (one panel per metric)."""
    plt = _plt()
    metrics = list(metrics)
    ncols = max(1, int(ncols))
    nrows = max(1, math.ceil(len(metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=plot_kwargs.pop("figsize",
                                                     (5.0 * ncols, 3.4 * nrows)),
                             squeeze=False)
    for idx, metric in enumerate(metrics):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        series = {method: _extract_metric(payload, metric)
                  for method, payload in histories.items()}
        series = {k: v for k, v in series.items() if v}
        names = sorted(series.keys(), key=lambda m: (METHOD_ORDER.index(m)
                                                     if m in METHOD_ORDER
                                                     else len(METHOD_ORDER)))
        for i, name in enumerate(names):
            style = method_style(name, i)
            label = METHOD_LABELS.get(str(name).lower(), str(name))
            payload = series[name]
            is_collection = isinstance(payload, (list, tuple)) and payload and not (
                isinstance(payload[0], (int, float)))
            if is_collection:
                stats = aggregate_curves(payload, confidence=confidence)
                if not len(stats):
                    continue
                plot_curve(stats.steps, stats.mean, ax=ax, label=label,
                           ci=stats.as_dict(), **{**style, **plot_kwargs})
            else:
                steps, values = _curve_pairs(payload)
                plot_curve(steps, values, ax=ax, label=label, **{**style, **plot_kwargs})
        ax.set_title(metric)
        ax.set_xlabel(xlabel)
        if names:
            ax.legend(loc="best", frameon=False)
    for idx in range(len(metrics), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")
    if path:
        save_figure(fig, path)
    if show:  # pragma: no cover
        plt.show()
    return fig


def _extract_metric(history: Any, metric: str) -> Any:
    """Pull one logged metric out of an evaluation-history container."""
    if history is None:
        return None
    if isinstance(history, Mapping):
        if metric in history and not isinstance(history.get(metric), Mapping):
            value = history[metric]
            if isinstance(value, (list, tuple)) and value and isinstance(value[0], (int, float)):
                steps = history.get("steps") or history.get("step") or history.get("times")
                if steps is not None and len(steps) == len(value):
                    return {"steps": list(steps), "values": list(value)}
            return value
        # nested ``{seed: history}``
        return {k: _extract_metric(v, metric) for k, v in history.items()}
    if isinstance(history, (list, tuple)):
        out = []
        for entry in history:
            if isinstance(entry, Mapping):
                stats = entry.get(metric, entry.get("mean"))
                if stats is None:
                    continue
                step = entry.get("step", entry.get("steps"))
                out.append((step if step is not None else len(out), stats))
            else:
                out.append(entry)
        return out
    return None


def plot_bar_comparison(values: Mapping[str, Any],
                        errors: Optional[Mapping[str, float]] = None,
                        confidence: float = DEFAULT_CONFIDENCE,
                        xlabel: str = "method", ylabel: str = "success rate",
                        title: Optional[str] = None,
                        path: Optional[str] = None, show: bool = False,
                        rotate: int = 0, **bar_kwargs: Any) -> Any:
    """Bar chart comparing methods (Figure 7 / Figure 3c style)."""
    plt = _plt()
    names = list(values.keys())
    names = sorted(names, key=lambda m: (METHOD_ORDER.index(m)
                                        if m in METHOD_ORDER else len(METHOD_ORDER)))
    means: List[float] = []
    errs: List[float] = []
    colors: List[Any] = []
    for i, name in enumerate(names):
        payload = values[name]
        if isinstance(payload, Mapping):
            payload = payload.get("mean", payload)
        if isinstance(payload, (list, tuple)) and payload:
            summary = mean_ci([v for v in payload if isinstance(v, (int, float))],
                              confidence=confidence)
            means.append(summary["mean"])
            errs.append(summary["half_width"])
        else:
            means.append(float(payload))
            errs.append(float((errors or {}).get(name, 0.0)))
        colors.append(method_style(name, i).get("color"))
    fig, ax = plt.subplots(figsize=bar_kwargs.pop("figsize", (5.0, 3.5)))
    xs = list(range(len(names)))
    ax.bar(xs, means, yerr=errs, color=colors, capsize=3,
           tick_label=[METHOD_LABELS.get(str(n).lower(), str(n)) for n in names],
           **bar_kwargs)
    if rotate:
        ax.set_xticklabels([METHOD_LABELS.get(str(n).lower(), str(n)) for n in names],
                           rotation=rotate, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if path:
        save_figure(fig, path)
    if show:  # pragma: no cover
        plt.show()
    return fig


def plot_success_rates(success_rates: Mapping[str, Any],
                       stages: Optional[Sequence[str]] = None,
                       confidence: float = DEFAULT_CONFIDENCE,
                       title: str = "per-stage success rate",
                       path: Optional[str] = None, show: bool = False,
                       **kwargs: Any) -> Any:
    """Grouped bar chart of per-stage success rates (Figure 7).

    ``success_rates`` maps ``method -> {stage: value}`` where each per-stage
    value may be a scalar or a list over seeds.
    """
    plt = _plt()
    methods = sorted(success_rates.keys(),
                     key=lambda m: (METHOD_ORDER.index(m)
                                    if m in METHOD_ORDER else len(METHOD_ORDER)))
    if stages is None:
        stages = []
        for method in methods:
            payload = success_rates[method]
            if isinstance(payload, Mapping):
                for stage in payload:
                    if stage not in stages:
                        stages.append(stage)
    stages = list(stages)
    n_methods, n_stages = len(methods), max(1, len(stages))
    width = 0.8 / n_methods
    fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (2.6 + 1.1 * n_stages, 3.6)))
    for i, method in enumerate(methods):
        payload = success_rates.get(method, {})
        means: List[float] = []
        errs: List[float] = []
        for stage in stages:
            value = payload.get(stage) if isinstance(payload, Mapping) else None
            if isinstance(value, (list, tuple)) and value:
                summary = mean_ci(value, confidence=confidence)
                means.append(summary["mean"])
                errs.append(summary["half_width"])
            elif value is None:
                means.append(0.0)
                errs.append(0.0)
            else:
                means.append(float(value))
                errs.append(0.0)
        style = method_style(method, i)
        xs = [s + (i - (n_methods - 1) / 2.0) * width for s in range(n_stages)]
        ax.bar(xs, means, width=width, yerr=errs, capsize=2,
               color=style.get("color"),
               label=METHOD_LABELS.get(str(method).lower(), str(method)))
    ax.set_xticks(range(n_stages))
    ax.set_xticklabels([str(s) for s in stages], rotation=20, ha="right")
    ax.set_ylabel("success rate")
    ax.set_title(title)
    if methods:
        ax.legend(loc="best", frameon=False)
    if path:
        save_figure(fig, path)
    if show:  # pragma: no cover
        plt.show()
    return fig


def plot_table_heatmap(table: Mapping[str, Mapping[str, Any]],
                       title: Optional[str] = None,
                       xlabel: str = "prefix length",
                       ylabel: str = "method",
                       cmap: str = "viridis",
                       path: Optional[str] = None, show: bool = False,
                       annotate: bool = True, vmin: float = 0.0,
                       vmax: float = 1.0, **kwargs: Any) -> Any:
    """Heatmap of a method x prefix table (Table 6 style)."""
    plt = _plt()
    methods = sorted(table.keys())
    columns: List[str] = []
    for method in methods:
        for col in table[method]:
            if str(col) not in columns:
                columns.append(str(col))
    try:
        columns = sorted(columns, key=lambda c: float(c))
    except ValueError:  # pragma: no cover - non-numeric columns
        pass
    matrix = [[float(table[m].get(c, table[m].get(c.replace(".0", ""), float("nan"))))
               for c in columns] for m in methods]
    fig, ax = plt.subplots(figsize=kwargs.pop("figsize",
                                              (1.2 + 0.7 * len(columns),
                                               1.2 + 0.5 * len(methods))))
    data = _as_numeric(matrix)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([str(c) for c in columns])
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([METHOD_LABELS.get(str(m).lower(), str(m)) for m in methods])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if annotate:
        for i in range(len(methods)):
            for j in range(len(columns)):
                val = matrix[i][j]
                if isinstance(val, float) and math.isnan(val):
                    continue
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="white" if val < 0.55 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if path:
        save_figure(fig, path)
    if show:  # pragma: no cover
        plt.show()
    return fig


def plot_forward_transfer(table: Mapping[str, Mapping[str, Any]],
                          baseline: Optional[Mapping[str, Any]] = None,
                          confidence: float = DEFAULT_CONFIDENCE,
                          title: str = "forward transfer",
                          xlabel: str = "number of prefix tasks",
                          ylabel: str = "forward transfer",
                          path: Optional[str] = None, show: bool = False,
                          ylim: Tuple[float, float] = (-0.05, 1.05),
                          **kwargs: Any) -> Any:
    """Line plot of forward transfer vs. prefix length (Table 6 / Figure 22)."""
    plt = _plt()
    methods = sorted(table.keys(),
                     key=lambda m: (METHOD_ORDER.index(m)
                                    if m in METHOD_ORDER else len(METHOD_ORDER)))
    fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (5.5, 3.6)))
    for i, method in enumerate(methods):
        payload = table[method]
        try:
            xs = sorted(payload.keys(), key=lambda k: float(k))
        except (TypeError, ValueError):  # pragma: no cover
            xs = list(payload.keys())
        ys: List[float] = []
        errs: List[float] = []
        for x in xs:
            value = payload[x]
            if isinstance(value, (list, tuple)) and value:
                summary = mean_ci(value, confidence=confidence)
                ys.append(summary["mean"])
                errs.append(summary["half_width"])
            elif isinstance(value, Mapping):
                ys.append(float(value.get("mean", float("nan"))))
                errs.append(float(value.get("half_width", 0.0)))
            else:
                ys.append(float(value))
                errs.append(0.0)
        style = method_style(method, i)
        plot_curve([float(x) if _is_number(x) else idx for idx, x in enumerate(xs)],
                   ys, ax=ax, label=METHOD_LABELS.get(str(method).lower(), str(method)),
                   color=style.get("color"), linestyle=style.get("linestyle", "-"),
                   marker=style.get("marker"))
        if any(errs):
            ax.errorbar(range(len(xs)), _as_numeric(ys), yerr=_as_numeric(errs),
                        fmt="none", ecolor=style.get("color"), capsize=3, alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    if methods:
        ax.legend(loc="best", frameon=False)
    if path:
        save_figure(fig, path)
    if show:  # pragma: no cover
        plt.show()
    return fig


def plot_retention_summary(summaries: Mapping[str, Mapping[str, Any]],
                           metrics: Sequence[str] = ("return", "forgetting",
                                                     "loglikelihood"),
                           confidence: float = DEFAULT_CONFIDENCE,
                           path: Optional[str] = None, show: bool = False,
                           **kwargs: Any) -> Any:
    """Compact summary of retention methods across several metrics.

    ``summaries`` maps ``method -> {metric: value-or-seed-list}``.
    """
    return plot_metric_grid(
        {method: {metric: payload.get(metric) for metric in metrics}
         for method, payload in summaries.items()},
        metrics=list(metrics), confidence=confidence, path=path, show=show,
        **kwargs)


def _is_number(x: Any) -> bool:
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #

def load_history(path: str) -> Any:
    """Load a JSON history file (``history.json`` / ``summary.json``)."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _find_history(directory: str) -> Optional[str]:
    for name in ("history.json", "summary.json", "metrics.json", "eval_history.json"):
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate):
            return candidate
    return None


def collect_histories(results_dir: str, methods: Sequence[str] = (
        "none", "ewc", "bc", "em", "ks", "scratch"),
        metric_key: Optional[str] = None,
        layouts: Sequence[str] = ("{method}/seed_{seed}",
                                  "{method}/seed{seed}",
                                  "{seed}/{method}",
                                  "seed_{seed}/{method}")) -> Dict[str, List[Any]]:
    """Scan a results directory and group per-seed histories by method.

    Expected layout (written by the trainers)::

        <results_dir>/<method>/seed_<seed>/(history|summary).json

    Returns ``{method: [history, ...]}`` (per-seed histories in seed order).
    """
    grouped: Dict[str, List[Any]] = {m: [] for m in methods}
    if not os.path.isdir(results_dir):
        return grouped
    for method in methods:
        method_dir = os.path.join(results_dir, method)
        seed_entries: List[str] = []
        for name in os.listdir(method_dir) if os.path.isdir(method_dir) else []:
            full = os.path.join(method_dir, name)
            if os.path.isdir(full):
                seed_entries.append(full)
        if not seed_entries:  # flat layout: <results_dir>/<method>.json
            candidate = _find_history(method_dir)
            if candidate:
                grouped[method].append(load_history(candidate))
            continue
        seed_entries.sort(key=_seed_sort_key)
        for entry in seed_entries:
            path = _find_history(entry)
            if path:
                history = load_history(path)
                if metric_key and isinstance(history, Mapping) and metric_key in history:
                    history = {metric_key: history[metric_key]}
                grouped[method].append(history)
    return grouped


def load_summaries(results_dir: str, methods: Sequence[str] = (
        "none", "ewc", "bc", "em", "ks", "scratch")) -> Dict[str, Any]:
    """Alias of :func:`collect_histories` for the ``summary.json`` layout."""
    return collect_histories(results_dir, methods=methods)


def _seed_sort_key(path: str) -> Tuple[int, str]:
    base = os.path.basename(path)
    digits = "".join(ch for ch in base if ch.isdigit())
    try:
        return int(digits), base
    except ValueError:  # pragma: no cover
        return 10 ** 9, base


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser():  # pragma: no cover - CLI plumbing
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot learning curves / retention summaries from a results directory.")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--metric", default="success_rate")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--xlabel", default="environment steps")
    parser.add_argument("--ylabel", default=None)
    parser.add_argument("--log-x", action="store_true")
    parser.add_argument("--per-method", action="store_true",
                        help="write one figure per method instead of a comparison")
    parser.add_argument("--grid", type=int, default=100)
    parser.add_argument("--output", default=None)
    parser.add_argument("-q", "--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    args = build_parser().parse_args(argv)
    methods = args.methods or ["scratch", "none", "ewc", "bc", "em", "ks"]
    histories = collect_histories(args.results_dir, methods=methods)
    histories = {m: h for m, h in histories.items() if h}
    if not histories:
        if not args.quiet:
            print(f"no histories found under {args.results_dir}")
        return 1

    per_method: Dict[str, Any] = {}
    for method, runs in histories.items():
        per_method[method] = _extract_metric(runs, args.metric)

    if args.per_method:
        output_dir = args.output or os.path.join(args.results_dir, "figures")
        os.makedirs(output_dir, exist_ok=True)
        for method, curve in per_method.items():
            stats = aggregate_curves(curve, confidence=args.confidence,
                                     num_grid=args.grid)
            path = os.path.join(output_dir, f"curve_{method}_{args.metric}.png")
            plot_curves({method: stats.as_dict()}, confidence=args.confidence,
                        xlabel=args.xlabel, ylabel=args.ylabel or args.metric,
                        log_x=args.log_x, path=path)
        if not args.quiet:
            print(f"wrote figures to {output_dir}")
        return 0

    output = args.output or os.path.join(args.results_dir,
                                         f"comparison_{args.metric}.png")
    plot_metric_grid(per_method, metrics=[args.metric], confidence=args.confidence,
                     path=output, xlabel=args.xlabel)
    if not args.quiet:
        print(f"wrote {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
