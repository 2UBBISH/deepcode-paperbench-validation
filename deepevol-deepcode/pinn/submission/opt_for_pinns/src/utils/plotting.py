"""Figure helpers for the "Challenges in Training PINNs" reproduction.

This module is *glue*: it does not implement any algorithm of the paper, it only
turns the numbers produced by the experiment runners into the paper's figures.

Figures targeted
----------------
* Figure 1  - loss vs. iteration on the wave PDE (Adam slow, L-BFGS stalls,
              NNCG continues to decrease)             -> :func:`plot_training_curves`
* Figure 2  - loss vs. L2RE scatter (lower loss <-> lower error)
                                                      -> :func:`plot_loss_vs_l2re`
* Figure 3  - top: spectral density of ``H_L(w)`` for the three PDEs;
              bottom: per-component (residual / initial / boundary) densities
                                                      -> :func:`plot_spectral_density`,
                                                         :func:`plot_component_densities`
* Figure 4  - NNCG / GD fine-tuning: loss and gradient norm vs. iteration
                                                      -> :func:`plot_finetune`
* Figure 5  - NNCG fine-tuning L2RE vs. iteration   -> :func:`plot_finetune` (l2re panel)
* Figure 7  - L-BFGS-preconditioned densities, per PDE and per component
                                                      -> :func:`plot_preconditioned_densities`
* Figure 8  - optimizer comparison per network width -> :func:`plot_width_sweep`

Everything is written with a lazy matplotlib import so that the training code
never requires a working display and so that the package imports cleanly on
machines without matplotlib.

Typical usage
-------------
>>> from src.utils.plotting import plot_spectral_density
>>> plot_spectral_density(estimates, outfile="results/fig3_top.pdf")
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "PLOTTING_AVAILABLE",
    "set_plot_style",
    "save_figure",
    "plot_training_curves",
    "plot_loss_vs_l2re",
    "plot_spectral_density",
    "plot_component_densities",
    "plot_preconditioned_densities",
    "plot_condition_numbers",
    "plot_finetune",
    "plot_optimizer_comparison",
    "plot_width_sweep",
    "plot_histogram_spectrum",
    "make_figures_from_summary",
    "PDE_LABELS",
    "COMPONENT_LABELS",
    "COLORS",
]


# --------------------------------------------------------------------------- #
# Optional dependency handling
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - depends on environment
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

try:  # pragma: no cover - depends on environment
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as _plt

    PLOTTING_AVAILABLE = True
except Exception:  # pragma: no cover
    _plt = None
    PLOTTING_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Cosmetic constants
# --------------------------------------------------------------------------- #
PDE_LABELS: Dict[str, str] = {
    "convection": "Convection",
    "reaction": "Reaction",
    "wave": "Wave",
}

COMPONENT_LABELS: Dict[str, str] = {
    "residual": "Residual",
    "initial": "Initial condition",
    "boundary": "Boundary condition",
    "ic": "Initial condition",
    "bc": "Boundary condition",
    "total": "Total",
}

COLORS: Dict[str, str] = {
    "adam": "#1f77b4",
    "lbfgs": "#ff7f0e",
    "adam+lbfgs": "#2ca02c",
    "combined": "#2ca02c",
    "nncg": "#d62728",
    "gd": "#7f7f7f",
    "residual": "#1f77b4",
    "initial": "#ff7f0e",
    "boundary": "#2ca02c",
}

_OPT_LABEL = {
    "adam": "Adam",
    "lbfgs": "L-BFGS",
    "l-bfgs": "L-BFGS",
    "adam+lbfgs": "Adam+L-BFGS",
    "combined": "Adam+L-BFGS",
    "nncg": "NNCG",
    "gd": "GD",
}


def _require_matplotlib():
    """Return the pyplot module, raising a helpful error when unavailable."""
    if not PLOTTING_AVAILABLE or _plt is None:
        raise ImportError(
            "matplotlib is required for plotting helpers; install it with "
            "`pip install matplotlib` (the training code does not need it)."
        )
    return _plt


def set_plot_style(style: Optional[str] = "seaborn-v0_8-whitegrid") -> None:
    """Apply a compact, publication-friendly matplotlib style (best effort)."""
    plt = _require_matplotlib()
    if style:
        try:
            plt.style.use(style)
        except Exception:
            pass
    matplotlib.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "mathtext.fontset": "dejavusans",
        }
    )


# --------------------------------------------------------------------------- #
# Generic utilities
# --------------------------------------------------------------------------- #
def _is_density_result(obj: Any) -> bool:
    """True for spectral_density.DensityResult-like objects."""
    return hasattr(obj, "grid") and hasattr(obj, "density")


def _density_arrays(res: Any) -> Tuple[Any, Any]:
    """Extract ``(grid, density)`` from a DensityResult / dict / tuple."""
    if _is_density_result(res):
        grid, dens = res.grid, res.density
    elif isinstance(res, Mapping):
        grid = res.get("grid")
        dens = res.get("density")
    elif isinstance(res, (tuple, list)) and len(res) >= 2:
        grid, dens = res[0], res[1]
    else:  # pragma: no cover - defensive
        raise TypeError(f"cannot interpret {type(res)!r} as a spectral density")

    if _np is not None:
        return _np.asarray(grid), _np.asarray(dens)
    return grid, dens  # pragma: no cover


def _as_scalar(x: Any) -> float:
    """Best-effort conversion of tensor/array/scalar to python float."""
    if x is None:
        return float("nan")
    if _np is not None and isinstance(x, _np.ndarray) and x.size == 1:
        return float(x.reshape(-1)[0])
    if hasattr(x, "detach"):
        try:
            return float(x.detach().cpu().reshape(-1)[0])
        except Exception:
            pass
    try:
        return float(x)
    except Exception:
        return float("nan")


def _as_sequence(values: Any) -> List[float]:
    """Convert an iterable/tensor/array of numbers to a list of floats."""
    if values is None:
        return []
    if hasattr(values, "detach"):
        values = values.detach().cpu()
        try:
            values = values.reshape(-1).tolist()
        except Exception:
            values = values.tolist()
    elif _np is not None and isinstance(values, _np.ndarray):
        values = values.reshape(-1).tolist()
    elif not isinstance(values, (list, tuple)):
        try:
            values = list(values)
        except TypeError:
            values = [values]
    return [_as_scalar(v) for v in values]


def _prepare_outfile(outfile: Optional[Any]) -> Optional[str]:
    if outfile is None:
        return None
    path = Path(str(outfile))
    if path.parent and str(path.parent) not in ("", "."):
        path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def save_figure(fig, outfile: Optional[Any] = None, show: bool = False, close: bool = True):
    """Save (and/or show) a figure, creating parent directories as needed.

    Returns the resolved file path (or ``None`` when the figure was not saved).
    """
    if outfile is not None:
        path = _prepare_outfile(outfile)
        fig.savefig(path)
    if show:  # pragma: no cover - interactive only
        _require_matplotlib().show()
    if close:
        _require_matplotlib().close(fig)
    return _prepare_outfile(outfile)


def _log_ticks(ax, axis: str = "y") -> None:
    """Use scientific-ish log ticks without hiding small values."""
    from matplotlib.ticker import LogLocator, NullFormatter

    if axis == "y":
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
        ax.xaxis.set_minor_formatter(NullFormatter())


def _opt_label(name: Any) -> str:
    key = str(name).lower()
    return _OPT_LABEL.get(key, str(name))


def _opt_color(name: Any) -> str:
    key = str(name).lower()
    if key in COLORS:
        return COLORS[key]
    if "+" in key:
        return COLORS["combined"]
    return "#333333"


# --------------------------------------------------------------------------- #
# Figure 1 / Figure 4 / Figure 5 : training curves
# --------------------------------------------------------------------------- #
def plot_training_curves(
    histories: Mapping[str, Any],
    *,
    key: str = "losses",
    steps_key: str = "steps",
    xlabel: str = "Iteration",
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    logy: bool = True,
    outfile: Optional[str] = None,
    show: bool = False,
    ax=None,
    **plot_kwargs: Any,
):
    """Plot one curve per optimizer (Figure 1 / Figure 8 style).

    Parameters
    ----------
    histories:
        Mapping ``{optimizer_name: history}`` where ``history`` is either a
        ``TrainingHistory``-like object (attributes ``losses`` / ``steps``), a
        dict with those keys, or a tuple ``(steps, values)``.
    key:
        Which series of the history to plot (``"losses"``, ``"l2re"``,
        ``"grad_norms"``).
    """
    plt = _require_matplotlib()
    if ylabel is None:
        ylabel = {
            "losses": "Loss",
            "l2re": "L2RE",
            "grad_norms": r"$\|\nabla L\|_2$",
        }.get(key, key)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(5.0, 3.4))
    else:
        fig = ax.figure

    for name, hist in histories.items():
        if hist is None:
            continue
        if isinstance(hist, (tuple, list)) and len(hist) == 2 and not isinstance(hist, Mapping):
            xs, ys = _as_sequence(hist[0]), _as_sequence(hist[1])
        elif isinstance(hist, Mapping):
            ys = _as_sequence(hist.get(key))
            xs = _as_sequence(hist.get(steps_key)) or list(range(1, len(ys) + 1))
        else:
            ys = _as_sequence(getattr(hist, key, None))
            xs = _as_sequence(getattr(hist, steps_key, None)) or list(range(1, len(ys) + 1))

        n = min(len(xs), len(ys))
        if n == 0:
            continue
        ax.plot(
            xs[:n],
            ys[:n],
            label=_opt_label(name),
            color=_opt_color(name),
            lw=1.6,
            **plot_kwargs,
        )

    if logy:
        ax.set_yscale("log")
        _log_ticks(ax, "y")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(frameon=True)

    if own_fig:
        return save_figure(fig, outfile, show=show)
    return fig


def plot_finetune(
    histories: Mapping[str, Any],
    *,
    outfile: Optional[str] = None,
    title: Optional[str] = None,
    show: bool = False,
) -> Dict[str, Optional[str]]:
    """Figure 4/5: NNCG vs. GD fine-tuning (loss, gradient norm, L2RE).

    Produces a 1x3 panel figure when the histories carry gradient norms and
    L2RE values; falls back to a loss-only panel otherwise.
    """
    plt = _require_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.4))

    panels = [
        ("losses", "Loss", True),
        ("grad_norms", r"$\|\nabla L\|_2$", True),
        ("l2re", "L2RE", True),
    ]
    out: Dict[str, Optional[str]] = {}
    for ax, (key, ylabel, logy) in zip(axes, panels):
        plotted = False
        for name, hist in (histories or {}).items():
            if hist is None:
                continue
            if isinstance(hist, Mapping):
                ys = _as_sequence(hist.get(key))
                xs = _as_sequence(hist.get("steps")) or list(range(1, len(ys) + 1))
            else:
                ys = _as_sequence(getattr(hist, key, None))
                xs = _as_sequence(getattr(hist, "steps", None)) or list(range(1, len(ys) + 1))
            n = min(len(xs), len(ys))
            if n == 0:
                continue
            finite = [v for v in ys[:n] if math.isfinite(v) and v > 0]
            if not finite and logy:
                continue
            ax.plot(xs[:n], ys[:n], label=_opt_label(name), color=_opt_color(name), lw=1.6)
            plotted = True
        if logy:
            ax.set_yscale("log")
            _log_ticks(ax, "y")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel if key != "l2re" else "L2RE")
        if plotted:
            ax.legend(frameon=True)

    if title:
        fig.suptitle(title)
    path = save_figure(fig, outfile, show=show)
    out["finetune"] = path
    return out


# --------------------------------------------------------------------------- #
# Figure 2 : loss vs L2RE
# --------------------------------------------------------------------------- #
def plot_loss_vs_l2re(
    records: Any,
    *,
    loss_key: str = "loss",
    l2re_key: str = "l2re",
    group_key: str = "optimizer",
    outfile: Optional[str] = None,
    title: Optional[str] = None,
    logx: bool = True,
    logy: bool = True,
    annotate: bool = True,
    show: bool = False,
):
    """Figure 2: scatter of loss against L2RE, one series per optimizer.

    ``records`` may be:
      * a mapping ``{optimizer: (losses, l2res)}``,
      * a mapping ``{optimizer: {"loss": [...], "l2re": [...]}}``,
      * a flat sequence of dicts with the ``loss_key``/``l2re_key``/``group_key``
        fields (e.g. the rows of ``Table 1``).
    """
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(5.2, 3.8))

    def _scatter(name: str, xs: Sequence[float], ys: Sequence[float]):
        if not xs:
            return
        ax.scatter(
            xs,
            ys,
            s=26,
            alpha=0.85,
            label=_opt_label(name),
            color=_opt_color(name),
            edgecolors="none" if not annotate else "#222222",
            linewidths=0.0 if not annotate else 0.4,
        )

    if isinstance(records, Mapping):
        for name, payload in records.items():
            if isinstance(payload, Mapping):
                xs = _as_sequence(payload.get(loss_key, payload.get("final_loss")))
                ys = _as_sequence(payload.get(l2re_key))
            elif isinstance(payload, (tuple, list)) and len(payload) == 2:
                xs, ys = _as_sequence(payload[0]), _as_sequence(payload[1])
            else:  # pragma: no cover - defensive
                continue
            _scatter(name, xs, ys)
    else:
        groups: Dict[str, Tuple[List[float], List[float]]] = {}
        for row in records or []:
            name = row.get(group_key, "optimizer") if isinstance(row, Mapping) else group_key
            xs, ys = groups.setdefault(name, ([], []))
            xs.append(_as_scalar(row.get(loss_key)))
            ys.append(_as_scalar(row.get(l2re_key)))
        for name, (xs, ys) in groups.items():
            _scatter(name, xs, ys)

    if logx:
        ax.set_xscale("log")
        _log_ticks(ax, "x")
    if logy:
        ax.set_yscale("log")
        _log_ticks(ax, "y")
    ax.set_xlabel("Loss")
    ax.set_ylabel("L2 relative error")
    if title:
        ax.set_title(title)
    ax.legend(frameon=True, title="Optimizer")
    return save_figure(fig, outfile, show=show)


# --------------------------------------------------------------------------- #
# Figure 3 (top) / Figure 7 : spectral densities
# --------------------------------------------------------------------------- #
def plot_spectral_density(
    estimates: Mapping[str, Any],
    *,
    outfile: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: str = "Eigenvalue",
    ylabel: str = "Spectral density",
    logy: bool = False,
    smooth: bool = True,
    xlim: Optional[Tuple[float, float]] = None,
    ax=None,
    show: bool = False,
    ncols: Optional[int] = None,
):
    """Figure 3 (top): density of ``H_L`` per PDE (one curve per PDE).

    ``estimates`` maps ``pde_name -> DensityResult`` (or ``(grid, density)``).
    """
    plt = _require_matplotlib()
    own_fig = ax is None
    if own_fig:
        names = [k for k, v in (estimates or {}).items() if v is not None]
        ncols = ncols or max(1, min(3, len(names)))
        nrows = int(math.ceil(len(names) / ncols)) or 1
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.3 * nrows), squeeze=False)
        fig.suptitle(title or "Hessian spectral density of the PINN loss")
        for i, name in enumerate(names):
            ax_i = axes[i // ncols][i % ncols]
            _draw_density(ax_i, estimates[name], label=PDE_LABELS.get(name, name),
                          color=None, xlabel=xlabel, ylabel=ylabel, logy=logy,
                          smooth=smooth, xlim=xlim)
            ax_i.set_title(PDE_LABELS.get(name, name))
        for j in range(len(names), nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")
        return save_figure(fig, outfile, show=show)

    for name, res in (estimates or {}).items():
        if res is None:
            continue
        _draw_density(ax, res, label=_opt_label(name), color=_opt_color(name),
                      xlabel=xlabel, ylabel=ylabel, logy=logy, smooth=smooth, xlim=xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(frameon=True)
    return fig


def _draw_density(ax, res, *, label: str, color: Optional[str], xlabel: str, ylabel: str,
                  logy: bool, smooth: bool, xlim: Optional[Tuple[float, float]]):
    """Draw a single density curve, optionally Gaussian-smoothed."""
    grid, dens = _density_arrays(res)
    if _np is not None and smooth and dens.size > 8:
        kernel = _np.array([1, 4, 6, 4, 1], dtype=float)
        kernel /= kernel.sum()
        dens = _np.convolve(dens, kernel, mode="same")
    ax.plot(grid, dens, label=label, color=color, lw=1.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
        _log_ticks(ax, "y")
    if xlim is not None:
        ax.set_xlim(*xlim)


def plot_component_densities(
    densities: Mapping[str, Mapping[str, Any]],
    *,
    outfile: Optional[str] = None,
    title: Optional[str] = None,
    logy: bool = False,
    xlim: Optional[Tuple[float, float]] = None,
    show: bool = False,
):
    """Figure 3 (bottom): per-component densities ``{pde: {component: density}}``."""
    plt = _require_matplotlib()
    pdes = [p for p, d in (densities or {}).items() if d]
    if not pdes:
        raise ValueError("plot_component_densities received no densities")
    fig, axes = plt.subplots(1, len(pdes), figsize=(4.8 * len(pdes), 3.4), squeeze=False)
    components = ["residual", "initial", "boundary"]
    for i, pde in enumerate(pdes):
        ax = axes[0][i]
        comp_map = densities[pde]
        for comp in components:
            res = comp_map.get(comp)
            if res is None:
                continue
            _draw_density(ax, res, label=COMPONENT_LABELS.get(comp, comp),
                          color=COLORS.get(comp), xlabel="Eigenvalue",
                          ylabel="Spectral density", logy=logy, smooth=True, xlim=xlim)
        ax.set_title(PDE_LABELS.get(pde, pde))
        ax.legend(frameon=True)
    if title:
        fig.suptitle(title)
    return save_figure(fig, outfile, show=show)


def plot_preconditioned_densities(
    densities: Mapping[str, Any],
    *,
    outfile: Optional[str] = None,
    title: Optional[str] = None,
    logy: bool = True,
    show: bool = False,
):
    """Figure 7: ``H_L`` vs. preconditioned ``H~^T H_L H~`` per PDE/component.

    ``densities`` is a nested mapping
    ``{pde: {component_or_"total": {"plain": density, "preconditioned": density}}}``.
    """
    plt = _require_matplotlib()
    pdes = list((densities or {}).keys())
    if not pdes:
        raise ValueError("plot_preconditioned_densities received no densities")

    fig, axes = plt.subplots(1, len(pdes), figsize=(4.8 * len(pdes), 3.4), squeeze=False)
    for i, pde in enumerate(pdes):
        ax = axes[0][i]
        entry = densities[pde] or {}
        plain = entry.get("plain") if "plain" in entry else entry.get("total", {}).get("plain")
        pre = (
            entry.get("preconditioned")
            if "preconditioned" in entry
            else entry.get("total", {}).get("preconditioned")
        )
        if plain is not None:
            _draw_density(ax, plain, label="H_L", color=COLORS["adam"],
                          xlabel="Eigenvalue", ylabel="Spectral density",
                          logy=logy, smooth=True, xlim=None)
        if pre is not None:
            _draw_density(ax, pre, label=r"$\tilde{H}^T H_L \tilde{H}$",
                          color=COLORS["combined"], xlabel="Eigenvalue",
                          ylabel="Spectral density", logy=logy, smooth=True, xlim=None)
        ax.set_title(PDE_LABELS.get(pde, pde))
        ax.set_xlabel("Eigenvalue")
        ax.legend(frameon=True)
    if title:
        fig.suptitle(title)
    return save_figure(fig, outfile, show=show)


def plot_condition_numbers(
    report: Mapping[str, Any],
    *,
    outfile: Optional[str] = None,
    title: Optional[str] = None,
    show: bool = False,
):
    """Bar chart of condition numbers for ``H_L`` vs. preconditioned operator.

    ``report`` maps ``pde -> {"plain": cond, "preconditioned": cond}`` (any
    missing entry is skipped). A log y-axis highlights the >=1e3 reduction that
    the paper reports.
    """
    plt = _require_matplotlib()
    pdes = list((report or {}).keys())
    if not pdes:
        raise ValueError("plot_condition_numbers received no report")

    plain = [_as_scalar((report[p] or {}).get("plain", (report[p] or {}).get("condition_number")))
             for p in pdes]
    pre = [_as_scalar((report[p] or {}).get("preconditioned",
                                            (report[p] or {}).get("preconditioned_condition_number")))
           for p in pdes]

    xs = _np.arange(len(pdes)) if _np is not None else list(range(len(pdes)))
    width = 0.38
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.bar([x - width / 2 for x in xs], plain, width=width, label=r"$\kappa(H_L)$",
           color=COLORS["adam"])
    ax.bar([x + width / 2 for x in xs], pre, width=width,
           label=r"$\kappa(\tilde{H}^T H_L \tilde{H})$", color=COLORS["combined"])
    ax.set_xticks(list(xs))
    ax.set_xticklabels([PDE_LABELS.get(p, p) for p in pdes])
    ax.set_yscale("log")
    _log_ticks(ax, "y")
    ax.set_ylabel("Condition number")
    if title:
        ax.set_title(title)
    ax.legend(frameon=True)
    return save_figure(fig, outfile, show=show)


def plot_histogram_spectrum(
    eigenvalues: Any,
    *,
    outfile: Optional[str] = None,
    title: Optional[str] = None,
    bins: int = 50,
    logx: bool = True,
    logy: bool = False,
    label: Optional[str] = None,
    show: bool = False,
):
    """Histogram of a set of (outlier) eigenvalues, e.g. from Lanczos Ritz values."""
    plt = _require_matplotlib()
    vals = _as_sequence(eigenvalues)
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    if logx:
        vals = [abs(v) for v in vals if v != 0]
        edges = None
        if _np is not None and vals:
            lo = max(min(vals), 1e-30)
            edges = _np.logspace(math.log10(lo), math.log10(max(vals)), bins + 1)
        ax.hist(vals, bins=edges if edges is not None else bins, label=label, color=COLORS["adam"])
        ax.set_xscale("log")
        _log_ticks(ax, "x")
    else:
        ax.hist(vals, bins=bins, label=label, color=COLORS["adam"])
    if logy:
        ax.set_yscale("log")
        _log_ticks(ax, "y")
    ax.set_xlabel("Eigenvalue (magnitude)")
    ax.set_ylabel("Count")
    if title:
        ax.set_title(title)
    if label:
        ax.legend(frameon=True)
    return save_figure(fig, outfile, show=show)


# --------------------------------------------------------------------------- #
# Figure 2/8 : optimizer comparison across widths
# --------------------------------------------------------------------------- #
def plot_optimizer_comparison(
    results: Mapping[str, Mapping[str, Any]],
    *,
    metric: str = "l2re",
    outfile: Optional[str] = None,
    title: Optional[str] = None,
    width_key: str = "width",
    group_key: str = "optimizer",
    logy: bool = True,
    pde: Optional[str] = None,
    show: bool = False,
):
    """Figure 2/8: best metric per (width, optimizer), one line per optimizer.

    ``results`` accepts either
      * ``{pde: [row, ...]}`` with rows carrying ``width``/``optimizer``/``metric``
        (e.g. Table 1 rows), or
      * ``{width: {optimizer: value}}`` for a single PDE.
    """
    plt = _require_matplotlib()

    def _panel(ax, rows: Sequence[Mapping[str, Any]], panel_title: Optional[str]):
        grouped: Dict[str, Dict[float, float]] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            w = _as_scalar(row.get(width_key))
            name = row.get(group_key, "optimizer")
            val = _as_scalar(row.get(metric, row.get(f"best_{metric}", row.get(f"{metric}_min"))))
            if not math.isfinite(w) or not math.isfinite(val):
                continue
            grouped.setdefault(str(name), {})[w] = val
        for name, series in grouped.items():
            items = sorted(series.items())
            ax.plot([w for w, _ in items], [v for _, v in items],
                    marker="o", ms=4, lw=1.5, label=_opt_label(name), color=_opt_color(name))
        if logy:
            ax.set_yscale("log")
            _log_ticks(ax, "y")
        ax.set_xlabel("Network width")
        ax.set_ylabel({"l2re": "L2RE", "loss": "Loss"}.get(metric, metric))
        ax.set_title(panel_title or "")
        ax.legend(frameon=True)

    if isinstance(results, Mapping) and all(isinstance(v, (list, tuple)) for v in results.values()):
        pdes = [pde] if pde else list(results.keys())
        fig, axes = plt.subplots(1, len(pdes), figsize=(4.6 * len(pdes), 3.4), squeeze=False)
        for i, name in enumerate(pdes):
            _panel(axes[0][i], results.get(name, []), PDE_LABELS.get(name, name))
        if title:
            fig.suptitle(title)
        return save_figure(fig, outfile, show=show)

    # {width: {optimizer: value}} form -> single panel
    rows: List[Dict[str, Any]] = []
    for w, per_opt in (results or {}).items():
        if not isinstance(per_opt, Mapping):
            continue
        for name, val in per_opt.items():
            rows.append({width_key: _as_scalar(w), group_key: name, metric: _as_scalar(val)})
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    _panel(ax, rows, title or PDE_LABELS.get(str(pde), pde))
    return save_figure(fig, outfile, show=show)


def plot_width_sweep(
    results: Mapping[str, Mapping[str, Any]],
    *,
    metric: str = "l2re",
    outfile: Optional[str] = None,
    show: bool = False,
):
    """Thin wrapper around :func:`plot_optimizer_comparison` for Figure 8."""
    return plot_optimizer_comparison(results, metric=metric, outfile=outfile, show=show)


# --------------------------------------------------------------------------- #
# Batch helper used by the runners
# --------------------------------------------------------------------------- #
def make_figures_from_summary(
    summary: Mapping[str, Any],
    outdir: Any,
    *,
    prefix: str = "",
) -> Dict[str, Optional[str]]:
    """Create every figure that the given ``summary.json`` payload supports.

    The runner passes the dict written to ``results/<study>/summary.json``; this
    helper inspects its keys and silently skips the panels whose data is absent.
    Figures that fail for any reason are skipped (plotting is best-effort glue).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    produced: Dict[str, Optional[str]] = {}
    if not PLOTTING_AVAILABLE:
        return produced

    def _try(name: str, fn, *args, **kwargs):
        target = outdir / f"{prefix}{name}.png"
        try:
            produced[name] = fn(*args, outfile=str(target), show=False, **kwargs)
        except Exception as exc:  # pragma: no cover - plotting is best effort
            produced[name] = None
            produced[f"{name}__error"] = f"{type(exc).__name__}: {exc}"

    summary = summary or {}

    # per-PDE loss / l2re histories
    if summary.get("histories"):
        _try("loss_curves", plot_training_curves, summary["histories"])
    if summary.get("finetune_histories"):
        _try("finetune", plot_finetune, summary["finetune_histories"])
    if summary.get("records"):
        _try("loss_vs_l2re", plot_loss_vs_l2re, summary["records"])
    if summary.get("density"):
        _try("spectral_density", plot_spectral_density, summary["density"])
    if summary.get("component_density"):
        _try("component_densities", plot_component_densities, summary["component_density"])
    if summary.get("preconditioned_density"):
        _try(
            "preconditioned_densities",
            plot_preconditioned_densities,
            summary["preconditioned_density"],
        )
    if summary.get("condition_numbers"):
        _try("condition_numbers", plot_condition_numbers, summary["condition_numbers"])
    if summary.get("table1"):
        _try(
            "optimizer_comparison",
            plot_optimizer_comparison,
            summary["table1"],
            metric="l2re",
        )
    if summary.get("outlier_eigenvalues"):
        _try("outlier_eigenvalues", plot_histogram_spectrum, summary["outlier_eigenvalues"])
    return produced


def _self_test(outdir: str = "/tmp/pinn_plot_test") -> List[str]:  # pragma: no cover
    """Quick smoke test of the figure helpers on synthetic data."""
    import random

    if not PLOTTING_AVAILABLE:
        print("matplotlib unavailable; skipping plotting self-test")
        return []

    os.makedirs(outdir, exist_ok=True)
    grid = [10 ** (i / 40 - 6) for i in range(241)]
    dens = [math.exp(-((math.log10(g) + 2.0) ** 2) / 0.4) for g in grid]
    written = []

    written.append(
        plot_spectral_density(
            {"convection": (grid, dens), "reaction": (grid, dens), "wave": (grid, dens)},
            outfile=os.path.join(outdir, "fig3_top.png"),
        )
    )
    written.append(
        plot_component_densities(
            {"convection": {"residual": (grid, dens), "initial": (grid, dens), "boundary": (grid, dens)}},
            outfile=os.path.join(outdir, "fig3_bottom.png"),
        )
    )
    written.append(
        plot_preconditioned_densities(
            {"convection": {"plain": (grid, dens), "preconditioned": (grid, dens)}},
            outfile=os.path.join(outdir, "fig7.png"),
        )
    )
    written.append(
        plot_condition_numbers(
            {"convection": {"plain": 1e8, "preconditioned": 1e3},
             "reaction": {"plain": 1e6, "preconditioned": 1e3},
             "wave": {"plain": 1e10, "preconditioned": 1e4}},
            outfile=os.path.join(outdir, "cond.png"),
        )
    )
    hist = {"steps": list(range(0, 41000, 500)),
            "losses": [10 ** (-1 - 3 * i / 81) for i in range(82)],
            "grad_norms": [10 ** (-0.5 - 2 * i / 81) for i in range(82)]}
    written.append(
        plot_training_curves(
            {"adam": hist, "l-bfgs": hist, "adam+lbfgs": hist, "nncg": hist},
            outfile=os.path.join(outdir, "fig1.png"),
        )
    )
    written.append(
        plot_loss_vs_l2re(
            {"adam": {"loss": [1e-4, 1e-5], "l2re": [5e-2, 8e-3]},
             "l-bfgs": {"loss": [1.5e-5], "l2re": [8e-3]},
             "adam+lbfgs": {"loss": [6e-6], "l2re": [4e-3]}},
            outfile=os.path.join(outdir, "fig2.png"),
        )
    )
    written.append(
        plot_finetune(
            {"nncg": {"steps": list(range(2000)), "losses": [1e-3 * 0.99 ** i for i in range(2000)],
                      "grad_norms": [1e-2 * 0.99 ** i for i in range(2000)],
                      "l2re": [5e-2 * 0.995 ** i for i in range(2000)]},
             "gd": {"steps": list(range(2000)), "losses": [1e-3] * 2000,
                    "grad_norms": [1e-2] * 2000, "l2re": [5e-2] * 2000}},
            outfile=os.path.join(outdir, "fig4.png"),
        )
    )
    rows = {
        "convection": [
            {"width": w, "optimizer": opt, "l2re": 10 ** (-1 - random.random())}
            for w in (50, 100, 200, 400)
            for opt in ("adam", "l-bfgs", "adam+lbfgs")
        ]
    }
    written.append(
        plot_optimizer_comparison(rows, outfile=os.path.join(outdir, "fig8.png"))
    )
    written.append(
        plot_histogram_spectrum([2e4, 1e4, 5e3, 1e2, 1e1], outfile=os.path.join(outdir, "hist.png"))
    )
    written.append(
        json.dumps(make_figures_from_summary({}, outdir)) and os.path.join(outdir, "summary_ok")
    )
    print("plotting self-test wrote:", [w for w in written if w])
    return [w for w in written if w]


if __name__ == "__main__":  # pragma: no cover
    _self_test()
