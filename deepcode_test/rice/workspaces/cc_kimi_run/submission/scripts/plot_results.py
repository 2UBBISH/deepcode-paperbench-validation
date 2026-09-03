"""Plotting utilities for RICE experiment results."""
import json
import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_experiment_i(
    results: Dict[str, Any],
    output_path: str = "results/fidelity.png",
    title: str = "Fidelity Scores",
) -> None:
    """Bar plot of fidelity scores across explanation methods and K values."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    methods = list(results.keys())
    k_values = sorted(
        [float(k.split("=")[1]) for k in results[methods[0]].keys()],
        key=float,
    )
    x = np.arange(len(k_values))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, method in enumerate(methods):
        means = []
        stds = []
        for k in k_values:
            entry = results[method][f"k={k}"]
            means.append(entry["mean"])
            stds.append(entry["std"])
        ax.bar(x + i * width, means, width, yerr=stds, label=method)

    ax.set_xlabel("Top-K Fraction")
    ax.set_ylabel("Fidelity Score")
    ax.set_title(title)
    ax.set_xticks(x + width)
    ax.set_xticklabels([str(k) for k in k_values])
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_experiment_ii_or_iii(
    results: Dict[str, Dict[str, float]],
    output_path: str = "results/refining.png",
    title: str = "Refining Performance",
) -> None:
    """Bar plot of final reward after refining for each method."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    methods = list(results.keys())
    means = [results[m]["mean"] for m in methods]
    stds = [results[m]["std"] for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(methods, means, yerr=stds)
    ax.set_ylabel("Final Reward")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_experiment_v(
    results: Dict[str, Dict[str, Dict[str, float]]],
    output_path: str = "results/sensitivity.png",
    title: str = "Hyper-parameter Sensitivity",
) -> None:
    """Plot sensitivity to p, lambda, and alpha."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    params = ["p", "lambda", "alpha"]
    for ax, param in zip(axes, params):
        values = sorted([float(k) for k in results[param].keys()], key=float)
        means = [results[param][str(v)]["mean"] for v in values]
        stds = [results[param][str(v)]["std"] for v in values]
        ax.errorbar(values, means, yerr=stds, marker="o")
        ax.set_xlabel(param)
        ax.set_ylabel("Final Reward")
        ax.set_title(f"Sensitivity to {param}")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def load_and_plot(result_dir: str) -> None:
    """Load result JSON files from a directory and generate plots."""
    for name, plot_fn in [
        ("experiment_i.json", plot_experiment_i),
        ("experiment_ii.json", plot_experiment_ii_or_iii),
        ("experiment_iii.json", plot_experiment_ii_or_iii),
        ("experiment_v.json", plot_experiment_v),
    ]:
        path = os.path.join(result_dir, name)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        output = os.path.join(result_dir, name.replace(".json", ".png"))
        plot_fn(data, output_path=output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=str, required=True)
    args = parser.parse_args()
    load_and_plot(args.result_dir)
