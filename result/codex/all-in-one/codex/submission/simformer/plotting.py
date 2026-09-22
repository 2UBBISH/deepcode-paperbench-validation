"""Plotting helpers for the figures of the paper."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_posterior_predictive(samples: np.ndarray, x_obs: np.ndarray,
                              observation_indices: Sequence[int],
                              output: Path, true_trajectory: Optional[np.ndarray] = None,
                              title: str = "posterior predictive"):
    """Fig. 5a/b: posterior predictive samples of a time series."""
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    quantiles = np.quantile(samples, [0.005, 0.5, 0.995], axis=0)
    time = np.arange(quantiles.shape[-1])
    ax[0].fill_between(time, quantiles[0], quantiles[2], alpha=0.3,
                       label="Simformer 99% quantiles")
    ax[0].plot(time, quantiles[1], color="C0", label="Simformer median")
    if true_trajectory is not None:
        ax[0].plot(time, true_trajectory, color="k", label="ground truth")
    ax[0].scatter(observation_indices, x_obs, color="C2", marker="x",
                  label="observations")
    ax[0].set_xlabel("time")
    ax[0].set_ylabel("population density")
    ax[0].set_title(title)
    ax[0].legend(fontsize=7)
    ax[1].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_posterior_marginals(samples: np.ndarray, names: Sequence[str],
                             output: Path, true_values: Optional[np.ndarray] = None,
                             title: str = "posterior"):
    """Fig. 6/7: marginals of the inferred posterior."""
    plt = _plt()
    n = samples.shape[-1]
    n_cols = min(4, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows),
                             squeeze=False)
    for i in range(n):
        ax = axes[i // n_cols][i % n_cols]
        ax.hist(samples[:, i], bins=40, density=True, color="C0", alpha=0.7)
        if true_values is not None:
            ax.axvline(true_values[i], color="k", linestyle="--")
        ax.set_title(names[i] if i < len(names) else f"var {i}", fontsize=9)
    for j in range(n, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].set_visible(False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_function_posterior(samples: np.ndarray, times: np.ndarray, output: Path,
                            true_values: Optional[np.ndarray] = None,
                            title: str = "posterior of a time dependent parameter"):
    """Fig. 6a (upper right): posterior of a function valued parameter."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5, 3))
    mean = samples.mean(axis=0)
    low, high = np.quantile(samples, [0.005, 0.995], axis=0)
    ax.fill_between(times, low, high, alpha=0.3, label="99% quantiles")
    ax.plot(times, mean, color="C0", label="posterior mean")
    if true_values is not None:
        ax.plot(times, true_values, "k--", label="ground truth")
    ax.set_xlabel("time")
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_energy_posterior_predictive(energy_samples: np.ndarray,
                                     simulator_energy: Optional[np.ndarray],
                                     output: Path, threshold: Optional[float] = None,
                                     title: str = "energy consumption"):
    """Fig. 7c/f: posterior predictive of the energy consumption."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(energy_samples, bins=40, density=True, alpha=0.6, color="C0",
            label="Simformer")
    if simulator_energy is not None:
        ax.hist(simulator_energy, bins=40, density=True, histtype="step",
                color="C2", linewidth=2, label="simulator")
    if threshold is not None:
        ax.axvline(threshold, color="r", linestyle="--",
                   label="energy constraint")
    ax.set_xlabel(r"energy [$\mu J / s$]")
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_c2st_vs_simulations(results: dict, output: Path):
    """Fig. 4a: C2ST as a function of the number of simulations."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4))
    for task, values in results.items():
        n_sims = sorted(values["simformer"].keys())
        ax.plot(n_sims, [values["simformer"][n] for n in n_sims], "o-",
                label=f"{task} (Simformer)")
        if "npe" in values:
            ax.plot(n_sims, [values["npe"][n] for n in n_sims], "s--",
                    label=f"{task} (NPE)")
    ax.axhline(0.5, color="grey", linestyle=":", label="perfect")
    ax.set_xscale("log")
    ax.set_xlabel("number of simulations")
    ax.set_ylabel("C2ST accuracy")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
