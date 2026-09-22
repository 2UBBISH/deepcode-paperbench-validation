"""Plotting helpers for the figures of the paper."""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PDE_TITLES = {"convection": "Convection", "reaction": "Reaction", "wave": "Wave"}
COMPONENT_TITLES = {
    "residual": "Residual",
    "ic": "Initial condition",
    "bc": "Boundary condition",
}


def _save(fig, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _all_non_negative(curves: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> bool:
    """True if every curve is defined on an essentially non-negative range."""
    for grids, _ in curves.values():
        if len(grids) and float(np.min(grids)) < -1e-2 * max(abs(float(np.max(grids))), 1.0):
            return False
    return True


# ---------------------------------------------------------------------- #
# Figure 2: final L2RE vs final loss
# ---------------------------------------------------------------------- #
def plot_loss_vs_l2re(records: Sequence[dict], path: str) -> str:
    pdes = [p for p in ("convection", "reaction", "wave") if any(r["pde"] == p for r in records)]
    fig, axes = plt.subplots(1, len(pdes), figsize=(4.2 * len(pdes), 3.6), squeeze=False)
    markers = {"adam": "o", "lbfgs": "s", "adam_lbfgs": "^"}
    for ax, pde in zip(axes[0], pdes):
        for r in records:
            if r["pde"] != pde:
                continue
            ax.scatter(
                r["final_loss"],
                r["final_l2re"],
                s=18,
                color=plt.cm.viridis(r["width"] / 400.0),
                marker=markers.get(r["optimizer"], "o"),
                edgecolors="none",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Final loss")
        ax.set_ylabel("L2RE")
        ax.set_title(PDE_TITLES.get(pde, pde))
        ax.grid(alpha=0.3)
    return _save(fig, path)


# ---------------------------------------------------------------------- #
# Figures 3 and 7: Hessian spectral densities
# ---------------------------------------------------------------------- #
def plot_spectral_density(
    curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str,
    path: str,
    top_eigenvalues: Optional[Dict[str, float]] = None,
) -> str:
    """``curves`` maps a label to ``(grids, density)``."""
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    top = 0.0
    for label, (grids, density) in curves.items():
        style = "--" if "precond" in label else "-"
        ax.plot(grids, density, style, label=label, linewidth=1.6)
        if len(density):
            top = max(top, float(np.max(density)))
    ax.set_xscale("symlog", linthresh=1e-3 if _all_non_negative(curves) else 1.0)
    ax.set_yscale("log")
    if top > 0:
        ax.set_ylim(top * 1e-6, top * 5.0)
    ax.set_xlabel("Eigenvalue")
    ax.set_ylabel("Spectral density")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    if top_eigenvalues:
        txt = "\n".join(f"{k}: {v:.2e}" for k, v in top_eigenvalues.items())
        ax.text(
            0.02,
            0.02,
            txt,
            transform=ax.transAxes,
            fontsize=7,
            va="bottom",
            bbox=dict(fc="white", alpha=0.7, ec="0.7"),
        )
    return _save(fig, path)


def plot_spectral_density_panels(
    panels: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    suptitle: str,
    path: str,
) -> str:
    """One panel per loss component, each with Hessian + preconditioned curves."""
    names = list(panels)
    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 3.8), squeeze=False)
    for ax, name in zip(axes[0], names):
        top = 0.0
        for label, (grids, density) in panels[name].items():
            style = "--" if "precond" in label else "-"
            ax.plot(grids, density, style, label=label, linewidth=1.5)
            if len(density):
                top = max(top, float(np.max(density)))
        ax.set_xscale(
            "symlog",
            linthresh=1e-3 if _all_non_negative(panels[name]) else 1.0,
        )
        ax.set_yscale("log")
        if top > 0:
            ax.set_ylim(top * 1e-6, top * 5.0)
        ax.set_title(COMPONENT_TITLES.get(name, name))
        ax.set_xlabel("Eigenvalue")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("Spectral density")
    fig.suptitle(suptitle)
    return _save(fig, path)


# ---------------------------------------------------------------------- #
# Figure 4: under-optimization (NNCG / GD after Adam+L-BFGS)
# ---------------------------------------------------------------------- #
def plot_underoptimization(
    runs: Dict[str, Dict[str, List[float]]],
    path: str,
    loss_key: str = "loss",
    grad_key: str = "grad_norm",
) -> str:
    """``runs`` maps "<pde>:<optimizer>" to a trace of the fine-tuning phase."""
    pdes = sorted({k.split(":")[0] for k in runs})
    fig, axes = plt.subplots(2, len(pdes), figsize=(4.4 * len(pdes), 7.0), squeeze=False)
    for j, pde in enumerate(pdes):
        for label, trace in runs.items():
            if not label.startswith(pde + ":"):
                continue
            name = label.split(":", 1)[1]
            it = trace["iteration"]
            axes[0][j].plot(it, trace[loss_key], label=name)
            axes[1][j].plot(it, trace[grad_key], label=name)
        axes[0][j].set_yscale("log")
        axes[1][j].set_yscale("log")
        axes[0][j].set_title(PDE_TITLES.get(pde, pde))
        axes[0][j].set_ylabel("Loss")
        axes[1][j].set_ylabel(r"$\|\nabla L\|_2$")
        axes[1][j].set_xlabel("Iteration")
        for a in (axes[0][j], axes[1][j]):
            a.grid(alpha=0.3)
            a.legend(fontsize=8)
    return _save(fig, path)


def plot_optimizer_curves(
    curves: Dict[str, Dict[str, List[float]]],
    pde: str,
    path: str,
    x_key: str = "iteration",
    y_key: str = "loss",
) -> str:
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for label, tr in curves.items():
        ax.plot(tr[x_key], tr[y_key], label=label)
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss" if y_key == "loss" else y_key)
    ax.set_title(PDE_TITLES.get(pde, pde))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_phase_curves(
    phases: Dict[str, Tuple[Sequence[float], Sequence[float]]],
    pde: str,
    path: str,
    ylabel: str = "Loss",
) -> str:
    """Figure 1: the loss over the Adam, Adam+L-BFGS and NNCG phases."""
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    offset = 0.0
    for label, (it, vals) in phases.items():
        it = np.asarray(it, dtype=float) + offset
        ax.plot(it, vals, label=label)
        if len(it):
            offset = it[-1]
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(PDE_TITLES.get(pde, pde))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, path)


# ---------------------------------------------------------------------- #
# Figure 5: absolute error heat maps
# ---------------------------------------------------------------------- #
def plot_error_heatmaps(
    panels: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    path: str,
    ncols: int = 3,
) -> str:
    """``panels`` is a list of ``(title, x, t, abs_error)``."""
    n = len(panels)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.4 * nrows), squeeze=False)
    for ax in axes.reshape(-1)[n:]:
        ax.axis("off")
    for ax, (title, x, t, err) in zip(axes.reshape(-1), panels):
        im = ax.pcolormesh(x, t, err.T, shading="auto", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        fig.colorbar(im, ax=ax, label="absolute error")
    return _save(fig, path)


# ---------------------------------------------------------------------- #
# Figure 8: loss / L2RE vs network width
# ---------------------------------------------------------------------- #
def plot_bars(
    stats: Dict[str, Dict[str, List[float]]],
    metric: str,
    pde: str,
    path: str,
    widths: Sequence[int] = (50, 100, 200, 400),
) -> str:
    """``stats`` maps an optimizer label to ``{"min": [...], "median": [...], "max": [...]}``."""
    widths = WIDTHS if widths is None else widths
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    labels = list(stats)
    n = len(next(iter(stats.values()))["median"]) if labels else 0
    x = np.arange(n)
    width = 0.8 / max(len(labels), 1)
    for i, label in enumerate(labels):
        med = np.array(stats[label]["median"])
        lo = np.maximum(np.array(stats[label]["min"]), 1e-300)
        hi = np.array(stats[label]["max"])
        ax.bar(x + i * width, med, width, label=label)
        ax.errorbar(
            x + i * width,
            med,
            yerr=[np.abs(med - lo), np.abs(hi - med)],
            fmt="none",
            ecolor="k",
            elinewidth=0.8,
            capsize=2,
        )
    ax.set_yscale("log")
    ax.set_xticks(x + width * (len(labels) - 1) / 2)
    ax.set_xticklabels([f"w={w}" for w in list(widths)[:n]])
    ax.set_ylabel(metric)
    ax.set_title(PDE_TITLES.get(pde, pde))
    if labels:
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, path)


WIDTHS = [50, 100, 200, 400]
