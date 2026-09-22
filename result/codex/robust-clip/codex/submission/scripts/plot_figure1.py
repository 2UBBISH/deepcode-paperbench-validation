#!/usr/bin/env python
"""Radar plot of Fig. 1 (App. B.2) from the result files of the evaluation scripts.

The radial axes are the metrics of the zero-shot / LVLM tasks, each running from
0 to the maximum over the compared models, and both TeCoA and FARE are compared
at the l_inf radius of 2/255 (Robust) performance of LLaVA-1.5 on vision-language
tasks and zero-shot (robust) classification for different CLIP models.

Usage::

    python scripts/plot_figure1.py --results results/llava_coco_fare2.json ... \
        --labels CLIP TeCoA FARE --output results/figure1.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description="radar plot of Figure 1")
    parser.add_argument("--results", nargs="+", required=True, help="result JSON files produced by the eval scripts")
    parser.add_argument("--labels", nargs="+", required=True, help="model names (one per result file)")
    parser.add_argument("--output", default="results/figure1.png")
    return parser.parse_args()


def flatten(prefix: str, payload, out: dict) -> None:
    for key, value in payload.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            flatten(name, value, out)
        elif isinstance(value, (int, float)):
            out[name] = float(value)


def main():
    args = parse_args()
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional dependency
        raise SystemExit("matplotlib and numpy are required for the plot")

    metrics = {}
    names = []
    for path, label in zip(args.results, args.labels):
        with open(path) as handle:
            payload = json.load(handle)
        series = {}
        flatten("", payload.get("results", payload), series)
        metrics[label] = series
        names.extend(key for key in series if key not in metrics.get(label, {}))

    keys = sorted({key for series in metrics.values() for key in series})
    maximum = {key: max(metrics[label].get(key, 0.0) for label in metrics) for key in keys}
    angles = [2 * math.pi * i / len(keys) for i in range(len(keys))]
    angles += angles[:1]

    figure, axis = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    for label, series in metrics.items():
        values = [series.get(key, 0.0) / max(maximum[key], 1e-9) for key in keys]
        values += values[:1]
        axis.plot(angles, values, label=label)
        axis.fill(angles, values, alpha=0.1)
    axis.set_xticks(angles[:-1])
    axis.set_xticklabels([f"{key}\n(max {maximum[key]:.1f})" for key in keys], fontsize=7)
    axis.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    axis.set_title("Figure 1: normalised (robust) performance per task")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    figure.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
