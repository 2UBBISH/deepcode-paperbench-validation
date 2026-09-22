#!/usr/bin/env python3
"""Aggregate the JSONL result files into the tables underlying Figures 2-3.

Reads one or more result files produced by ``experiments/run_benchmark.py``
and ``experiments/run_baselines.py`` and prints a markdown table of C2ST scores
with rows ``(task, budget)`` and one column per method, mirroring the layout of
Figures 2 (non-sequential) and 3 (sequential) of the paper.

Example
-------
::

    python experiments/collect_results.py results/benchmark.jsonl results/baselines.jsonl
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import load_results  # noqa: E402

NON_SEQUENTIAL = ["npse_ve", "npse_vp", "npe"]
SEQUENTIAL = ["tsnpse_ve", "tsnpse_vp", "snpe_c", "tsnpe", "snpse_a", "snpse_b", "snpse_c"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+")
    p.add_argument("--csv", type=str, default=None, help="optionally write the table to a CSV file")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    records = []
    for path in args.files:
        records.extend(load_results(path))
    if not records:
        print("no results found")
        return 1

    table = collections.defaultdict(dict)
    for r in records:
        key = (r["task"], r["budget"])
        table[key][r["method"]] = r["c2st"]

    methods = [m for m in NON_SEQUENTIAL + SEQUENTIAL if any(m in v for v in table.values())]
    keys = sorted(table, key=lambda k: (k[1], k[0]))

    header = "| task | budget | " + " | ".join(methods) + " |"
    sep = "|" + "---|" * (len(methods) + 2)
    print(header)
    print(sep)
    for task, budget in keys:
        row = [task, str(budget)]
        for m in methods:
            v = table[(task, budget)].get(m)
            row.append(f"{v:.4f}" if isinstance(v, float) else "-")
        print("| " + " | ".join(row) + " |")

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w") as fh:
            fh.write("task,budget," + ",".join(methods) + "\n")
            for task, budget in keys:
                vals = [
                    f"{table[(task, budget)][m]:.6f}" if m in table[(task, budget)] else ""
                    for m in methods
                ]
                fh.write(f"{task},{budget}," + ",".join(vals) + "\n")
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
