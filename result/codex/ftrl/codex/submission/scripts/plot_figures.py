#!/usr/bin/env python3
"""Reproduce the paper's figures from the result files produced by the trainers.

The script consumes the JSON/npz artefacts written by
``scripts/train_nethack.py``, ``scripts/train_montezuma.py`` and
``scripts/train_metaworld.py`` and writes the figures of the paper:

======================  ==================================================
Figure 3a / Table 5     NetHack score across methods
Figure 3b / Figure 17b  Montezuma return across methods
Figure 3c / Figure 7    RoboticSequence success rate, overall and per stage
Figure 5                NetHack per-level average return (level 4, Sokoban)
Figure 6                Montezuma Room-7 success rate every 5M steps
Figure 8                expert log-likelihood + PCA projection
Figure 14               additional NetHack metrics
Figure 16 / 18          level/room visitation densities
Figure 20               CKA between pre-training and fine-tuned activations
======================  ==================================================

Example::

    python scripts/plot_figures.py --results results --out results/figures
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def _load_json(path):
    with open(path) as handle:
        return json.load(handle)


def plot_figure3a(nethack_dir: str, out_dir: str) -> None:
    """NetHack score of every method (Figure 3a)."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for path in sorted(glob.glob(os.path.join(nethack_dir, "*.json"))):
        method = os.path.basename(path).split("-")[0]
        history = _load_json(path)
        if "steps" in history and "score" in history:
            ax.plot(history["steps"], history["score"], label=method)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("NetHack score")
    ax.set_title("NetHack (Figure 3a)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figure3a_nethack.png"), dpi=150)
    plt.close(fig)


def plot_figure3b(montezuma_dir: str, out_dir: str) -> None:
    """Montezuma's Revenge average return (Figure 3b)."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for path in sorted(glob.glob(os.path.join(montezuma_dir, "finetune-*.json"))):
        method = os.path.basename(path).split("-")[1]
        history = _load_json(path)
        ax.plot(history["steps"], history["mean_extrinsic_return"], label=method)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("episode return")
    ax.set_title("Montezuma's Revenge (Figure 3b)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figure3b_montezuma.png"), dpi=150)
    plt.close(fig)


def plot_figure6(montezuma_dir: str, out_dir: str) -> None:
    """Room-7 success rate every 5M steps (Figure 6)."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for path in sorted(glob.glob(os.path.join(montezuma_dir, "room7-*.json"))):
        method = os.path.basename(path).split("-")[1].replace(".json", "")
        data = _load_json(path)
        ax.plot(data["steps"], data["room_success_rate"], marker="o", label=method)
    ax.set_xlabel("environment steps")
    ax.set_ylabel("Room 7 success rate")
    ax.set_title("Montezuma's Revenge, Room 7 (Figure 6)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figure6_room7.png"), dpi=150)
    plt.close(fig)


def plot_figure7(metaworld_dir: str, out_dir: str) -> None:
    """Per-stage success rate on RoboticSequence (Figure 7)."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = os.path.join(metaworld_dir, "grid.json")
    if not os.path.exists(path):
        print(f"[skip] {path} not found")
        return
    curves = _load_json(path)
    methods = list(curves)
    stages = list(curves[methods[0]])
    fig, axes = plt.subplots(1, len(stages), figsize=(4 * len(stages), 3.5), sharey=True)
    if len(stages) == 1:
        axes = [axes]
    for ax, stage in zip(axes, stages):
        for method in methods:
            runs = np.asarray(curves[method][stage], dtype=np.float64)
            if runs.size == 0:
                continue
            mean = runs.mean(axis=0)
            half = 1.64 * runs.std(axis=0) / np.sqrt(max(runs.shape[0], 1))
            steps = np.linspace(0, 1, len(mean))
            ax.plot(steps, mean, label=method)
            ax.fill_between(steps, mean - half, mean + half, alpha=0.2)
        ax.set_title(stage)
        ax.set_xlabel("training progress")
    axes[0].set_ylabel("success rate")
    axes[-1].legend()
    fig.suptitle("RoboticSequence per-stage success (Figure 7)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figure7_robotic_sequence.png"), dpi=150)
    plt.close(fig)


def plot_figure8(metaworld_dir: str, out_dir: str) -> None:
    """Expert log-likelihoods and their PCA projection (Figure 8)."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = os.path.join(metaworld_dir, "expert_loglikelihoods.npz")
    if not os.path.exists(path):
        print(f"[skip] {path} not found")
        return
    data = np.load(path)
    checkpoints = [key for key in data.files if key.startswith("checkpoint_")]
    fig, axes = plt.subplots(2, len(checkpoints), figsize=(3 * len(checkpoints), 6))
    for column, name in enumerate(sorted(checkpoints)):
        likelihoods = data[name]
        projection = data[name.replace("checkpoint_", "pca_")]
        axes[1][column].scatter(projection[:, 0], projection[:, 1], c=likelihoods, s=6, cmap="viridis")
        axes[1][column].set_title(name)
        axes[0][column].hist(likelihoods, bins=30)
    fig.suptitle("expert log-likelihood under the fine-tuned policy (Figure 8)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figure8_loglikelihoods.png"), dpi=150)
    plt.close(fig)


def plot_figure20(metaworld_dir: str, out_dir: str) -> None:
    """CKA between pre-training and fine-tuned activations (Figure 20)."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = os.path.join(metaworld_dir, "cka.json")
    if not os.path.exists(path):
        print(f"[skip] {path} not found")
        return
    data = _load_json(path)
    fig, ax = plt.subplots(figsize=(6, 4))
    for layer, values in data.items():
        ax.plot(values["steps"], values["cka"], label=layer)
    ax.set_xlabel("fine-tuning steps")
    ax.set_ylabel("CKA")
    ax.set_title("Representation similarity during fine-tuning (Figure 20)")
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figure20_cka.png"), dpi=150)
    plt.close(fig)


def plot_visitation(nethack_dir: str, montezuma_dir: str, out_dir: str) -> None:
    """Level / room visitation densities (Figures 4, 16, 18)."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for path in sorted(glob.glob(os.path.join(nethack_dir, "visitation-*.json"))):
        method = os.path.basename(path).split("-")[1].replace(".json", "")
        data = _load_json(path)
        axes[0].hist2d(data["turns"], data["max_level"], bins=(30, 20), cmap="magma", alpha=0.7)
        axes[0].set_title("NetHack level visitation")
        axes[0].set_xlabel("turns")
        axes[0].set_ylabel("max dungeon level")
    for path in sorted(glob.glob(os.path.join(montezuma_dir, "visitation-*.json"))):
        data = _load_json(path)
        rooms = sorted(int(k) for k in data)
        axes[1].plot(rooms, [data[str(r)] for r in rooms], marker="o")
        axes[1].set_title("Montezuma room visitation")
        axes[1].set_xlabel("room")
        axes[1].set_ylabel("steps spent")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "visitation.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/figures")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    nethack_dir = os.path.join(args.results, "nethack")
    montezuma_dir = os.path.join(args.results, "montezuma")
    metaworld_dir = os.path.join(args.results, "metaworld")
    for directory in (nethack_dir, montezuma_dir, metaworld_dir):
        os.makedirs(directory, exist_ok=True)

    plot_figure3a(nethack_dir, args.out)
    plot_figure3b(montezuma_dir, args.out)
    plot_figure6(montezuma_dir, args.out)
    plot_figure7(metaworld_dir, args.out)
    plot_figure8(metaworld_dir, args.out)
    plot_figure20(metaworld_dir, args.out)
    plot_visitation(nethack_dir, montezuma_dir, args.out)
    print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
