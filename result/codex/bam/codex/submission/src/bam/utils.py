"""Shared utilities for the experiment scripts (experiment loops, plotting)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import jax
import numpy as np


def split_keys(key, n: int):
    """Deterministic list of ``n`` sub-keys."""
    return list(jax.random.split(key, n))


@dataclass
class RunResult:
    """The outcome of a single (algorithm, hyper-parameter, seed) run."""

    method: str
    grad_evals: np.ndarray
    metric: np.ndarray
    meta: Dict = field(default_factory=dict)


def plot_curves(curves: Sequence[Dict], xlabel: str, ylabel: str, title: str, path: str,
                logy: bool = True, logx: bool = True):
    """Plot mean curves with standard-error bands (the style of the paper)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for c in curves:
        x, y = np.asarray(c["x"], dtype=float), np.asarray(c["y"], dtype=float)
        se = np.asarray(c.get("se", np.zeros_like(y)), dtype=float)
        if logy:
            # KL divergences can be a few ulps negative from round-off; clip for
            # the log-scale plots (the paper reports divergences on a log axis)
            y = np.maximum(y, 1e-16)
        line, = ax.plot(x, y, label=c["label"], color=c.get("color"))
        if c.get("linestyle"):
            line.set_linestyle(c["linestyle"])
        if c.get("band", True) and np.any(se > 0):
            ax.fill_between(x, np.maximum(y - se, 1e-16) if logy else y - se, y + se,
                            alpha=0.25, color=line.get_color(), lw=0)
        if c.get("individual") is not None:
            for run in np.asarray(c["individual"], dtype=float):
                ax.plot(x, np.maximum(run, 1e-16) if logy else run, color=line.get_color(), lw=0.5, alpha=0.25)
    if logy:
        ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
