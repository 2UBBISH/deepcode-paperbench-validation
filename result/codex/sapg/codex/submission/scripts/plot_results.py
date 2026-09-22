#!/usr/bin/env python3
"""Build the paper's figures and Table 1 from a directory of runs.

Example::

    python scripts/plot_results.py --root runs --out figures --results results
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sapg.envs import BENCHMARK_TASKS  # noqa: E402
from sapg.utils.plotting import (  # noqa: E402
    plot_curves,
    summarise_final,
    summary_to_latex,
    summary_to_markdown,
)
from sapg.utils.results import collect_runs, default_metric  # noqa: E402

DEFAULT_METHODS = ["sapg", "ppo", "pbt", "pql"]
DEFAULT_ABLATIONS = [
    "sapg",
    "sapg_entropy0.003",
    "sapg_entropy0.005",
    "sapg_no_offpolicy",
    "sapg_high_offpolicy",
    "sapg_symmetric",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SAPG results")
    parser.add_argument("--root", default="runs")
    parser.add_argument("--out", default="figures")
    parser.add_argument("--results", default="results")
    parser.add_argument("--tasks", nargs="*", default=list(BENCHMARK_TASKS))
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--ablations", nargs="*", default=DEFAULT_ABLATIONS)
    parser.add_argument("--smooth", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = collect_runs(args.root, tasks=args.tasks)
    if not runs:
        print(f"[plot_results] no runs found under '{args.root}'", file=sys.stderr)
        return

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.results, exist_ok=True)
    # population-based methods report their best member, SAPG reports its leader
    reduce_map = {"pbt": "max", "dexpbt": "max"}

    table: dict = {}
    for task, task_runs in runs.items():
        metric = default_metric(task)
        methods = {label: dirs for label, dirs in task_runs.items() if label in args.methods}
        if methods:
            out_path = os.path.join(args.out, f"fig5_{task}.png")
            plot_curves(
                methods,
                out_path,
                metric=metric,
                title=f"{task}: SAPG vs baselines",
                ylabel="successes" if "successes" in metric else "episode reward",
                smooth_window=args.smooth,
                reduce_map=reduce_map,
            )
            print(f"[plot_results] wrote {out_path}")

        ablations = {label: dirs for label, dirs in task_runs.items() if label in args.ablations}
        if ablations:
            out_path = os.path.join(args.out, f"fig6_{task}_ablations.png")
            plot_curves(
                ablations,
                out_path,
                metric=metric,
                title=f"{task}: ablations",
                ylabel="successes" if "successes" in metric else "episode reward",
                smooth_window=args.smooth,
            )
            print(f"[plot_results] wrote {out_path}")

        summary = summarise_final(task_runs, metric, reduce_map={**reduce_map, **{k: "mean" for k in task_runs}})
        table[task] = summary

    with open(os.path.join(args.results, "table1.md"), "w") as handle:
        for task, summary in table.items():
            handle.write(f"### {task}\n\n{summary_to_markdown(summary)}\n\n")
    with open(os.path.join(args.results, "table1.tex"), "w") as handle:
        for task, summary in table.items():
            handle.write(f"% {task}\n{summary_to_latex(summary)}\n\n")
    with open(os.path.join(args.results, "table1.json"), "w") as handle:
        json.dump(table, handle, indent=2)
    print(f"[plot_results] wrote {os.path.join(args.results, 'table1.md')}")


if __name__ == "__main__":
    main()
