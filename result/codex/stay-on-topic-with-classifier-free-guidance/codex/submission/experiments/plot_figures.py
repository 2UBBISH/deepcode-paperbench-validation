#!/usr/bin/env python3
"""Regenerate the paper's figures from the JSON written by the experiment
scripts.

Produces:

* Figure 2 / 17 -- GSM8K and AQuA accuracy and % valid chains vs gamma
  (``run_cot.py`` output);
* Figure 6 / 7 -- benchmark accuracy vs gamma for the GPT-2 and Pythia
  families (``run_zeroshot.py`` output);
* Figure 9 -- accuracy vs inference FLOPs per token (via
  ``run_flops_ancova.py --make-plots``);
* Figure 18 -- entropy and top-p size distributions from the Section 5
  per-example rows (``run_section5.py`` output).

Example
-------
    python experiments/plot_figures.py --zeroshot results/zeroshot.json \
        --cot results/cot.json --section5 results/section5/per_example.json \
        --output-dir results/figures
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_gamma_curves(records, value_key, out_path, title, group_key="task"):
    plt = _import_matplotlib()
    groups = defaultdict(list)
    for record in records:
        groups[record[group_key]].append(record)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, items in sorted(groups.items()):
        items = sorted(items, key=lambda r: r["gamma"])
        ax.plot([r["gamma"] for r in items], [r[value_key] for r in items], marker="o", label=name)
    ax.set_xlabel("gamma")
    ax.set_ylabel(value_key)
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_model_family(records, out_path, title):
    """One subplot per task, one line per model size (Figures 6/7)."""
    plt = _import_matplotlib()
    tasks = sorted({r["task"] for r in records})
    ncols = 3
    nrows = int(np.ceil(len(tasks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    for ax, task in zip(axes.ravel(), tasks):
        subset = [r for r in records if r["task"] == task and r.get("acc") is not None]
        by_model = defaultdict(list)
        for record in subset:
            by_model[record["model"]].append(record)
        for model, items in sorted(by_model.items()):
            items = sorted(items, key=lambda r: r["gamma"])
            ax.plot([r["gamma"] for r in items], [r["acc"] for r in items], marker="o", label=model)
        ax.set_title(task, fontsize=9)
        ax.set_xlabel("gamma", fontsize=8)
        ax.set_ylabel("accuracy", fontsize=8)
        ax.tick_params(labelsize=7)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_entropy_histograms(rows, out_path):
    plt = _import_matplotlib()
    keys = [k for k in ("entropy_prompted", "entropy_unprompted", "entropy_cfg", "entropy_instruct") if any(k in r for r in rows)]
    fig, ax = plt.subplots(figsize=(6, 4))
    for key in keys:
        values = [r[key] for r in rows if key in r]
        ax.hist(values, bins=50, histtype="step", label=f"{key} (mean={np.mean(values):.2f})")
    ax.set_xlabel("entropy (nats)")
    ax.set_ylabel("# datapoints")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zeroshot", default=None)
    parser.add_argument("--cot", default=None)
    parser.add_argument("--section5", default=None)
    parser.add_argument("--output-dir", default="results/figures")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.zeroshot and os.path.exists(args.zeroshot):
        with open(args.zeroshot) as fh:
            records = json.load(fh)
        gpt2 = [r for r in records if r.get("family") == "gpt2"]
        pythia = [r for r in records if r.get("family") == "pythia"]
        if gpt2:
            plot_model_family(gpt2, os.path.join(args.output_dir, "fig6_gpt2.png"),
                              "Standard benchmarks over CFG strengths (GPT-2)")
        if pythia:
            plot_model_family(pythia, os.path.join(args.output_dir, "fig7_pythia.png"),
                              "Standard benchmarks over CFG strengths (Pythia)")

    if args.cot and os.path.exists(args.cot):
        with open(args.cot) as fh:
            cot = json.load(fh)
        for dataset in sorted({r["dataset"] for r in cot}):
            subset = [dict(r, task=f"{r['model']}") for r in cot if r["dataset"] == dataset]
            plot_gamma_curves(subset, "accuracy",
                              os.path.join(args.output_dir, f"cot_{dataset}_accuracy.png"),
                              f"CoT accuracy vs gamma ({dataset})")
            plot_gamma_curves(subset, "valid_fraction",
                              os.path.join(args.output_dir, f"cot_{dataset}_valid.png"),
                              f"CoT % valid chains vs gamma ({dataset})")

    if args.section5 and os.path.exists(args.section5):
        with open(args.section5) as fh:
            rows = json.load(fh)
        plot_entropy_histograms(rows, os.path.join(args.output_dir, "fig18a_entropy.png"))


if __name__ == "__main__":
    main()
