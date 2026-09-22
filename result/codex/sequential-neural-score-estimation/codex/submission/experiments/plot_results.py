#!/usr/bin/env python3
"""Plot the C2ST results in the layout of Figures 2 and 3.

Figure 2 of the paper shows the non-sequential methods (NPSE with VE / VP SDEs
vs. NPE); Figure 3 shows the sequential methods (TSNPSE with VE / VP SDEs vs.
SNPE-C and TSNPE).  Both figures plot the C2ST score (lower is better; 0.5 is a
perfect posterior approximation) against the simulation budget for each of the
eight benchmark tasks.

Example
-------
::

    python experiments/plot_results.py results/benchmark.jsonl results/baselines.jsonl \
        --output figures/c2st
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import TASK_SPECS, load_results  # noqa: E402


NON_SEQUENTIAL = {
    "npse_ve": ("NPSE (VE SDE)", "tab:blue"),
    "npse_vp": ("NPSE (VP SDE)", "tab:cyan"),
    "npe": ("NPE", "tab:gray"),
}

SEQUENTIAL = {
    "tsnpse_ve": ("TSNPSE (VE SDE)", "tab:red"),
    "tsnpse_vp": ("TSNPSE (VP SDE)", "tab:orange"),
    "snpe_c": ("SNPE-C", "tab:gray"),
    "tsnpe": ("TSNPE", "tab:green"),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+")
    p.add_argument("--output", type=str, default="figures/c2st")
    return p.parse_args(argv)


def _panel(records, methods, out_path, title):
    tasks = [t for t in TASK_SPECS if any(r["task"] == t for r in records)]
    if not tasks:
        print(f"no records for {title}")
        return
    ncols = 4
    nrows = (len(tasks) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    for ax, task in zip(axes.reshape(-1), tasks):
        plotted = False
        for method, (label, colour) in methods.items():
            pts = sorted(
                (r["budget"], r["c2st"]) for r in records if r["task"] == task and r["method"] == method
            )
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", label=label, color=colour)
            plotted = True
        ax.set_xscale("log")
        ax.set_title(TASK_SPECS[task].display, fontsize=10)
        ax.set_ylim(0.45, 1.02)
        ax.axhline(0.5, color="k", lw=0.5, ls="--")
        ax.set_xlabel("simulations")
        ax.set_ylabel("C2ST")
        if plotted:
            ax.legend(fontsize=7)
    for ax in axes.reshape(-1)[len(tasks):]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_path}_{title.split()[0].lower()}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}_*.png/pdf")


def main(argv=None) -> int:
    args = parse_args(argv)
    records = []
    for path in args.files:
        records.extend(load_results(path))
    if not records:
        print("no results found")
        return 1
    _panel(records, NON_SEQUENTIAL, args.output, "Figure 2: non-sequential methods")
    _panel(records, SEQUENTIAL, args.output, "Figure 3: sequential methods")

    # aggregate view: mean C2ST across tasks per method and budget
    agg = collections.defaultdict(list)
    for r in records:
        agg[(r["method"], r["budget"])].append(r["c2st"])
    print("\nmean C2ST across tasks")
    for (method, budget), vals in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        print(f"  {method:12s} N={budget:7d}  {sum(vals) / len(vals):.4f}  ({len(vals)} tasks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
