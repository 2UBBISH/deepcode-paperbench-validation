"""Shared plotting utilities for the DPO / toxicity mechanistic analysis.

This module centralises all figure styling used to reproduce Figures 1-7 of
*"A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and
Toxicity"*.

Implemented conventions (from the paper text):

* **Figure 3** – residual streams before / after DPO (the shift
  :math:`\\delta_{\\mathbf{x}}` is an offset that lets :math:`\\text{GPT2}_{DPO}`
  bypass regions that previously triggered toxic value vectors).
* **Figure 4** – each point is a residual stream sampled from either
  :math:`\\mathbf{x}_{GPT}` or :math:`\\mathbf{x}_{DPO}`, projected onto
  1) :math:`\\bar{\\delta}_{\\mathbf{x}}` (mean difference in residual streams)
  and 2) the principal component of the residual streams.  Dotted lines join
  samples from the same prompt.  Colours indicate whether each point activates
  ``MLP.v_770^19``.
* **Figure 5** – **blue** areas: percentage of value vectors with a cosine
  similarity score against :math:`\\delta_{\\mathbf{x}}^{19}` (Eq. 2);
  **orange** areas: percentage of value vectors with a mean activation during
  the forward pass of the 1,199 RealToxicityPrompts.
* **Figure 7 (Appendix C/D)** – the same two plots for other layers.

No matplotlib import happens at module import time: figures are only created
inside the helper functions, so ``import src.analysis.plots`` stays cheap.

Author clarifications honoured here:

* Toxicity is measured with ``unitary/unbiased-toxic-roberta`` (not the
  Perspective API); this module only *draws* results, it never scores them.
* Llama2 figures are out of scope for the reproduction; the styling helpers are
  nonetheless model-agnostic.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    # colours / markers
    "BASE_COLOR",
    "ALIGNED_COLOR",
    "MODEL_COLORS",
    "MODEL_MARKERS",
    "ACTIVE_COLOR",
    "INACTIVE_COLOR",
    "HIST_COSINE_COLOR",
    "HIST_ACTIVATION_COLOR",
    "TOXIC_COLOR",
    "NEUTRAL_COLOR",
    "GPT2_NAME",
    "GPT2_DPO_NAME",
    "color_for_model",
    "marker_for_model",
    # figure lifecycle
    "set_style",
    "make_figure",
    "save_figure",
    # low level drawing
    "percentage_histogram",
    "scatter_groups",
    "paired_lines",
    "bar_with_errors",
    "line_series",
    "annotate_bars",
    "add_legend",
    "shade_region",
    "density_lines",
    # higher level composites
    "plot_shift_arrows",
    "plot_activation_bars",
    "plot_histogram_pair",
]

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
#: Colour used for the *unaligned* model (GPT2, blue-ish).
BASE_COLOR = "#1f77b4"
#: Colour used for the *aligned* model (GPT2_DPO, orange-ish).
ALIGNED_COLOR = "#ff7f0e"

#: Per-model colours, keyed by the names used across the analysis modules.
MODEL_COLORS: Dict[str, str] = {
    "gpt2": BASE_COLOR,
    "gpt2-medium": BASE_COLOR,
    "gpt2_medium": BASE_COLOR,
    "base": BASE_COLOR,
    "before": BASE_COLOR,
    "unaligned": BASE_COLOR,
    "gpt2_dpo": ALIGNED_COLOR,
    "gpt2-dpo": ALIGNED_COLOR,
    "dpo": ALIGNED_COLOR,
    "after": ALIGNED_COLOR,
    "aligned": ALIGNED_COLOR,
    "unaligned_10x": "#d62728",
}

#: Per-model markers so that GPT2 vs GPT2_DPO remain distinguishable in
#: greyscale (Figure 4 uses both colour *and* shape).
MODEL_MARKERS: Dict[str, str] = {
    "gpt2": "o",
    "gpt2-medium": "o",
    "gpt2_medium": "o",
    "base": "o",
    "before": "o",
    "unaligned": "o",
    "gpt2_dpo": "^",
    "gpt2-dpo": "^",
    "dpo": "^",
    "after": "^",
    "aligned": "^",
    "unaligned_10x": "s",
}

#: Colour for points/tokens that activate the toxic value vector.
ACTIVE_COLOR = "#d62728"
#: Colour for points/tokens that do NOT activate the toxic value vector.
INACTIVE_COLOR = "#7f7f7f"

#: Figure 5 blue areas: percentage of value vectors by cosine similarity
#: against ``delta_x``.
HIST_COSINE_COLOR = "#4c72b0"
#: Figure 5 orange areas: percentage of value vectors by mean activation.
HIST_ACTIVATION_COLOR = "#dd8452"

#: Misc semantic colours.
TOXIC_COLOR = "#c44e52"
NEUTRAL_COLOR = "#55a868"

#: Canonical model names (mirror :mod:`src.model_utils`).
GPT2_NAME = "gpt2"
GPT2_DPO_NAME = "gpt2_dpo"

_DEFAULT_FIGSIZE: Tuple[float, float] = (7.0, 4.0)
_DEFAULT_DPI = 150


# --------------------------------------------------------------------------- #
# Naming helpers
# --------------------------------------------------------------------------- #
def _normalise(name: Optional[str]) -> str:
    """Lower-case and strip a model label so palette lookups are forgiving."""
    if name is None:
        return ""
    return str(name).strip().lower().replace(" ", "")


def color_for_model(name: Optional[str], default: str = BASE_COLOR) -> str:
    """Return the palette colour for ``name`` (:data:`MODEL_COLORS`)."""
    key = _normalise(name)
    if key in MODEL_COLORS:
        return MODEL_COLORS[key]
    # tolerate e.g. "openai-community/gpt2-medium"
    for token in ("dpo", "aligned"):
        if token in key:
            return ALIGNED_COLOR
    for token in ("gpt2", "base", "before", "unaligned"):
        if token in key:
            return BASE_COLOR
    return default


def marker_for_model(name: Optional[str], default: str = "o") -> str:
    """Return the plotting marker for ``name`` (:data:`MODEL_MARKERS`)."""
    key = _normalise(name)
    if key in MODEL_MARKERS:
        return MODEL_MARKERS[key]
    for token in ("dpo", "aligned", "after"):
        if token in key:
            return "^"
    for token in ("gpt2", "base", "before", "unaligned"):
        if token in key:
            return "o"
    return default


# --------------------------------------------------------------------------- #
# Figure lifecycle
# --------------------------------------------------------------------------- #
def set_style(style: str = "seaborn-v0_8-whitegrid", font_scale: float = 1.0) -> None:
    """Best-effort seaborn/matplotlib style setup (never raises)."""
    try:  # pragma: no cover - depends on seaborn availability
        import seaborn as sns  # type: ignore

        try:
            sns.set_theme(style=style, context="paper", font_scale=font_scale)
        except Exception:
            sns.set(style="whitegrid")
    except Exception:
        try:
            import matplotlib as mpl  # type: ignore

            for candidate in (style, "seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
                if candidate in getattr(mpl.style, "available", []):
                    mpl.style.use(candidate)
                    break
        except Exception:
            pass


def _pyplot():
    """Import and configure :mod:`matplotlib.pyplot` lazily."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt  # type: ignore

    return plt


def make_figure(
    figsize: Tuple[float, float] = _DEFAULT_FIGSIZE,
    nrows: int = 1,
    ncols: int = 1,
    squeeze: bool = True,
    dpi: int = _DEFAULT_DPI,
    style: Optional[str] = None,
):
    """Create ``(fig, axes)`` with the shared style.

    Parameters
    ----------
    figsize:
        Figure size in inches.
    nrows, ncols:
        Sub-plot grid.  For a single sub-plot the returned ``axes`` is a single
        :class:`~matplotlib.axes.Axes` (``squeeze=True``) as opposed to a
        1-element array.
    """
    if style is not None:
        set_style(style)
    plt = _pyplot()
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi, squeeze=squeeze)
    return fig, axes


def save_figure(fig, out_path: Optional[str], dpi: int = _DEFAULT_DPI, close: bool = True, **kwargs) -> Optional[str]:
    """Save ``fig`` to ``out_path`` (creating directories) and optionally close it.

    Returns the path written, or ``None`` when ``out_path`` is ``None``.
    """
    if out_path is None:
        if close:
            _pyplot().close(fig)
        return None
    out_path = str(out_path)
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", **kwargs)
    if close:
        _pyplot().close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Low level drawing helpers
# --------------------------------------------------------------------------- #
def percentage_histogram(
    values,
    bins: int = 50,
    value_range: Tuple[float, float] = (-1.0, 1.0),
    density: bool = False,
    weights=None,
    eps: float = 1e-12,
):
    """Histogram expressed as a **percentage** of the inputs.

    This matches the Figure 5 convention where the y-axis of both the blue and
    the orange areas is the *percentage* of value vectors falling into each
    bin.  Non-finite values are dropped.

    Returns
    -------
    (percent, centers) : Tuple[np.ndarray, np.ndarray]
        ``percent[k]`` is the percentage of samples in bin ``k``;
        ``centers[k]`` is the bin centre.
    """
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        edges = np.linspace(value_range[0], value_range[1], int(bins) + 1)
        return np.zeros(int(bins)), 0.5 * (edges[:-1] + edges[1:])
    hist, edges = np.histogram(arr, bins=int(bins), range=tuple(value_range), weights=weights)
    total = float(hist.sum())
    if total <= eps:
        total = float(arr.size)
    percent = 100.0 * hist / total
    centers = 0.5 * (edges[:-1] + edges[1:])
    return percent.astype(np.float64), centers.astype(np.float64)


def scatter_groups(
    ax,
    groups: Dict[str, Dict[str, object]],
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    alpha: float = 0.6,
    s: float = 14.0,
    legend: bool = True,
    zorder: int = 2,
):
    """Scatter several labelled groups onto ``ax``.

    ``groups`` maps a legend label to a dict with keys ``x``, ``y`` and optional
    ``color``, ``marker``, ``alpha``, ``s``.
    """
    handles: List[object] = []
    labels: List[str] = []
    for label, data in groups.items():
        x = np.asarray(data["x"], dtype=np.float64).reshape(-1)
        y = np.asarray(data["y"], dtype=np.float64).reshape(-1)
        if x.size == 0 or y.size == 0:
            continue
        color = data.get("color", color_for_model(label))
        marker = data.get("marker", marker_for_model(label))
        h = ax.scatter(
            x,
            y,
            s=float(data.get("s", s)),
            alpha=float(data.get("alpha", alpha)),
            c=color,
            marker=marker,
            linewidths=0.0,
            label=label,
            zorder=zorder,
        )
        handles.append(h)
        labels.append(str(label))
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if legend and handles:
        ax.legend(handles, labels, frameon=True, fontsize="small")
    return ax


def paired_lines(
    ax,
    before_xy,
    after_xy,
    color: str = "#999999",
    alpha: float = 0.35,
    linewidth: float = 0.8,
    linestyle: str = ":",
    zorder: int = 1,
):
    """Draw dotted connector lines between paired (same-prompt) points.

    Figure 4: *"Dotted lines indicate samples from the same prompt."*
    ``before_xy`` / ``after_xy`` are arrays of shape ``[n, 2]``.
    """
    before = np.asarray(before_xy, dtype=np.float64)
    after = np.asarray(after_xy, dtype=np.float64)
    if before.ndim == 1:
        before = before.reshape(-1, 2)
    if after.ndim == 1:
        after = after.reshape(-1, 2)
    n = min(len(before), len(after))
    for k in range(n):
        ax.plot(
            [before[k, 0], after[k, 0]],
            [before[k, 1], after[k, 1]],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            linestyle=linestyle,
            zorder=zorder,
        )
    return ax


def bar_with_errors(
    ax,
    labels: Sequence[str],
    values: Sequence[float],
    errors: Optional[Sequence[float]] = None,
    colors: Optional[Sequence[str]] = None,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    rotate: float = 0.0,
    annotate: bool = False,
    fmt: str = "{:.3f}",
    width: float = 0.7,
):
    """Grouped bar chart used for Table 2 / Figure 2 style comparisons."""
    x = np.arange(len(labels))
    vals = np.asarray(values, dtype=np.float64)
    errs = None if errors is None else np.asarray(errors, dtype=np.float64)
    if colors is None:
        colors = [color_for_model(lab) for lab in labels]
    bars = ax.bar(x, vals, width=width, yerr=errs, color=colors, capsize=3, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(lab) for lab in labels], rotation=rotate, ha="right" if rotate else "center")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if annotate:
        annotate_bars(ax, bars, fmt=fmt)
    return bars


def annotate_bars(ax, bars, fmt: str = "{:.3f}", dy: float = 0.01, fontsize: int = 8) -> None:
    """Write the value of each bar above it."""
    for bar in bars:
        height = bar.get_height()
        try:
            ax.annotate(
                fmt.format(float(height)),
                xy=(bar.get_x() + bar.get_width() / 2.0, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=fontsize,
            )
        except Exception:  # pragma: no cover - defensive
            continue
    return None


def line_series(
    ax,
    xs: Sequence[float],
    series: Dict[str, Sequence[float]],
    colors: Optional[Dict[str, str]] = None,
    markers: Optional[Dict[str, str]] = None,
    xlabel: str = "",
    ylabel: str = "",
    title: str = "",
    linestyles: Optional[Dict[str, str]] = None,
    legend: bool = True,
    linewidth: float = 1.6,
):
    """Plot several named 1-D series (used by the logit-lens / activation curves)."""
    x = np.asarray(xs)
    for label, ys in series.items():
        color = (colors or {}).get(label, color_for_model(label))
        marker = (markers or {}).get(label, marker_for_model(label))
        ls = (linestyles or {}).get(label, "-")
        ax.plot(
            x,
            np.asarray(ys, dtype=np.float64),
            color=color,
            marker=marker,
            markersize=3.5,
            linewidth=linewidth,
            linestyle=ls,
            label=str(label),
        )
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if legend and len(series) > 0 and any(ax.get_legend_handles_labels()[0]):
        ax.legend(frameon=True, fontsize="small")
    return ax


def add_legend(ax, loc: str = "best", fontsize: str = "small", **kwargs):
    """Attach a legend when at least one labelled artist exists."""
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc=loc, fontsize=fontsize, frameon=True, **kwargs)
    return ax


def shade_region(ax, x0: float, x1: float, color: str = "#cccccc", alpha: float = 0.25, label: Optional[str] = None, zorder: int = 0):
    """Shade an interval (e.g. the negative-cosine region of Figure 5)."""
    return ax.axvspan(x0, x1, color=color, alpha=alpha, label=label, zorder=zorder)


def density_lines(ax, x: Sequence[float], ys: Dict[str, Sequence[float]], colors: Dict[str, str], label: str = "", linewidth: float = 1.4) -> None:
    """Overlay distribution outlines (Figure 5 histograms drawn as step lines)."""
    x = np.asarray(x, dtype=np.float64)
    for name, y in ys.items():
        y = np.asarray(y, dtype=np.float64)
        ax.plot(x, y, color=colors.get(name, HIST_COSINE_COLOR), linewidth=linewidth, label=name)
    if label:
        ax.set_ylabel(label)
    return None


# --------------------------------------------------------------------------- #
# Composite figures
# --------------------------------------------------------------------------- #
def plot_shift_arrows(
    ax,
    before_xy,
    after_xy,
    color: str = "#333333",
    alpha: float = 0.4,
    linewidth: float = 0.9,
    max_arrows: int = 60,
    seed: int = 0,
):
    """Draw arrows ``before -> after`` for a sample of prompts (Figure 3/4 motif)."""
    before = np.asarray(before_xy, dtype=np.float64).reshape(-1, 2)
    after = np.asarray(after_xy, dtype=np.float64).reshape(-1, 2)
    n = min(len(before), len(after))
    if n == 0:
        return ax
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(int(max_arrows), n), replace=False) if n > max_arrows else np.arange(n)
    for k in idx:
        ax.annotate(
            "",
            xy=(after[k, 0], after[k, 1]),
            xytext=(before[k, 0], before[k, 1]),
            arrowprops=dict(arrowstyle="->", color=color, alpha=alpha, linewidth=linewidth),
        )
    return ax


def plot_activation_bars(
    ax,
    result_before,
    result_after,
    labels: Optional[Sequence[str]] = None,
    layer: Optional[int] = None,
    before_label: str = "GPT2",
    after_label: str = "GPT2_DPO",
    title: str = "Mean MLP activations $m_i$",
    ylabel: str = "$m_i = \\sigma(x^\\ell \\cdot MLP.k_i^\\ell)$",
):
    """Grouped bars of the mean activations ``m_i`` before/after DPO (Figure 2).

    ``result_before`` / ``result_after`` are :class:`src.analysis.activations.MeanActivationResult`
    instances (duck-typed via ``indices`` and ``mean``).
    """
    idx = list(getattr(result_before, "indices", []))
    mean_b = np.asarray(getattr(result_before, "mean", []), dtype=np.float64)
    mean_a = np.asarray(getattr(result_after, "mean", []), dtype=np.float64)
    n = min(len(idx), mean_b.size, mean_a.size)
    idx, mean_b, mean_a = idx[:n], mean_b[:n], mean_a[:n]
    if labels is None:
        labels = [f"MLP.v$_{{{i}}}^{{{l}}}$" for (l, i) in idx]
    x = np.arange(n)
    width = 0.38
    ax.bar(x - width / 2.0, mean_b, width, color=BASE_COLOR, label=before_label, zorder=2)
    ax.bar(x + width / 2.0, mean_a, width, color=ALIGNED_COLOR, label=after_label, zorder=2)
    ax.axhline(0.0, color="#444444", linewidth=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(list(labels), rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title if layer is None else f"{title} (layer {layer})")
    ax.legend(frameon=True, fontsize="small")
    return ax


def plot_histogram_pair(
    ax,
    cosine_percent: Sequence[float],
    centers: Sequence[float],
    activation_percent: Optional[Sequence[float]] = None,
    title: str = "",
    xlabel: str = "cosine similarity with $\\delta_x$",
    ylabel: str = "% of value vectors",
    cosine_label: str = "$\\delta_{MLP.v}$ vs $\\delta_x$",
    activation_label: str = "mean activation",
    alpha: float = 0.55,
    annotate_decile: bool = False,
):
    """Figure 5: blue cosine-similarity histogram with the orange activation overlay.

    Both series are *percentages* of value vectors; the x-axis of the orange
    overlay is the mean activation and the x-axis of the blue histogram is the
    cosine similarity (this is the paper's side-by-side convention).
    """
    centers = np.asarray(centers, dtype=np.float64)
    ax.fill_between(
        centers,
        0.0,
        np.asarray(cosine_percent, dtype=np.float64),
        color=HIST_COSINE_COLOR,
        alpha=alpha,
        label=cosine_label,
        zorder=2,
    )
    if activation_percent is not None:
        act = np.asarray(activation_percent, dtype=np.float64)
        m = min(act.size, centers.size)
        ax.fill_between(
            centers[:m],
            0.0,
            act[:m],
            color=HIST_ACTIVATION_COLOR,
            alpha=alpha,
            label=activation_label,
            zorder=3,
        )
    ax.axvline(0.0, color="#444444", linewidth=0.8, linestyle="--", zorder=1)
    if annotate_decile:
        # fraction of value vectors with a negative cosine similarity
        total = float(np.sum(cosine_percent))
        if total > 0:
            neg = float(np.sum(np.asarray(cosine_percent)[np.asarray(centers) < 0.0]))
            ax.text(0.02, 0.94, f"{100.0 * neg / total:.1f}% negative", transform=ax.transAxes, fontsize=8, va="top")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(frameon=True, fontsize="small")
    return ax


# --------------------------------------------------------------------------- #
# Convenience: layer-wise multi-panel figure (Appendix C / D, Figure 7)
# --------------------------------------------------------------------------- #
def make_layer_grid(
    layers: Iterable[int],
    ncols: int = 3,
    figsize: Optional[Tuple[float, float]] = None,
    subplot_size: Tuple[float, float] = (4.0, 3.0),
    dpi: int = _DEFAULT_DPI,
):
    """Create a grid of axes, one per requested layer (Figure 7 helper)."""
    layers = list(layers)
    n = max(len(layers), 1)
    ncols = max(1, min(int(ncols), n))
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (subplot_size[0] * ncols, subplot_size[1] * nrows)
    fig, axes = make_figure(figsize=figsize, nrows=nrows, ncols=ncols, squeeze=False, dpi=dpi)
    flat = np.asarray(axes).reshape(-1)
    for ax in flat[n:]:
        ax.axis("off")
    return fig, axes, layers
