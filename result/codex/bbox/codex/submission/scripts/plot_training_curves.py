#!/usr/bin/env python
"""Plot the learning curves logged by ``AdapterTrainer`` (Appendix K style).

``train_bbox_adapter.py`` writes ``iteration_<t>/train_curves.csv`` with the loss
and the mean positive/negative energies; this script turns them into a figure so
that the behaviour reported in Appendix K (positive energy rising above negative
energy) can be inspected without opening a notebook.
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List


def read_curves(path: str) -> Dict[str, List[float]]:
    columns: Dict[str, List[float]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for key, value in row.items():
                columns.setdefault(key, []).append(float(value))
    return columns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Directory containing iteration_*/train_curves.csv")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    curve_paths = []
    for name in sorted(os.listdir(args.run_dir)):
        path = os.path.join(args.run_dir, name, "train_curves.csv")
        if os.path.exists(path):
            curve_paths.append(path)
    if not curve_paths:
        raise SystemExit(f"No train_curves.csv found under {args.run_dir}")

    output = args.output or os.path.join(args.run_dir, "learning_curves.png")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for path in curve_paths:
        curves = read_curves(path)
        label = os.path.basename(os.path.dirname(path))
        axes[0].plot(curves["step"], curves["loss"], label=label)
        axes[1].plot(curves["step"], curves["positive_energy"], label=f"{label} positive")
        axes[1].plot(curves["step"], curves["negative_energy"], linestyle="--",
                     label=f"{label} negative")
    axes[0].set_xlabel("adapter update"); axes[0].set_ylabel("loss"); axes[0].legend()
    axes[1].set_xlabel("adapter update"); axes[1].set_ylabel("energy g_theta"); axes[1].legend()
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
