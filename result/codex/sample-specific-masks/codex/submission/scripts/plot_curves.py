#!/usr/bin/env python3
"""Plot training curves (Appendix D.2) and mask statistics from run JSONs.

    python scripts/plot_curves.py --runs runs --dataset cifar10 --out figs/curves_cifar10.png

Every run stores the per-epoch training loss/accuracy and (every ``eval_every``
epochs) the test accuracy/loss, so the curves of Figure 11/12 can be redrawn
from the JSON files alone.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional


def load_runs(runs_dir: str, dataset: str, backbone: Optional[str] = None) -> Dict[str, List[dict]]:
    runs: Dict[str, List[dict]] = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(runs_dir, dataset, "*.json"))):
        with open(path) as fh:
            record = json.load(fh)
        if backbone and record.get("config", {}).get("backbone") != backbone:
            continue
        label = record.get("method", "?")
        runs[label].append(record)
    return runs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    runs = load_runs(args.runs, args.dataset, args.backbone)
    if not runs:
        print(f"no runs found under {args.runs}/{args.dataset}")
        return 1

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for label, records in runs.items():
        for record in records:
            history = record.get("history", [])
            epochs = [h["epoch"] for h in history]
            axes[0].plot(epochs, [h["train_loss"] for h in history], label=f"{label} (seed {record['seed']})")
            axes[1].plot(epochs, [h["train_accuracy"] for h in history], label=label)
            test = [(h["epoch"], h["test_accuracy"]) for h in history if "test_accuracy" in h]
            if test:
                axes[2].plot([t[0] for t in test], [t[1] for t in test], marker="o", label=label)
    for ax, title in zip(axes, ["training loss", "training accuracy (%)", "test accuracy (%)"]):
        ax.set_title(f"{args.dataset}: {title}")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    out = args.out or os.path.join("figs", f"curves_{args.dataset}.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
