"""Aggregate run results into the tables of the paper.

``python -m smm.aggregate --runs runs --backbone resnet18 --table 1``

prints mean +/- std test accuracy per (dataset, method) like Table 1 (ResNets),
Table 2 (ViT-B/32) and Table 3 (ablations), and can dump a CSV/JSON summary.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

from .datasets import DATASET_SPECS

METHOD_ORDER = ["pad", "narrow", "medium", "full", "smm"]
ABLATION_ORDER = ["only_delta", "only_mask", "single_channel", "smm"]
PRETTY = {
    "pad": "Pad", "narrow": "Narrow", "medium": "Medium", "full": "Full",
    "smm": "Ours", "only_delta": "Only delta", "only_mask": "Only f_mask",
    "single_channel": "Single-channel f_mask",
}


def collect(runs_dir: str, backbone: Optional[str] = None) -> Dict[str, Dict[str, List[float]]]:
    table: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for root, _dirs, files in os.walk(runs_dir):
        for fname in files:
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(root, fname)) as fh:
                try:
                    rec = json.load(fh)
                except json.JSONDecodeError:  # pragma: no cover
                    continue
            dataset = rec.get("dataset")
            method = rec.get("method")
            acc = rec.get("test_accuracy")
            if dataset is None or method is None or acc is None:
                continue
            if backbone is not None and rec.get("config", {}).get("backbone") != backbone:
                continue
            table[dataset][method].append(float(acc))
    return table


def format_table(table: Dict[str, Dict[str, List[float]]], order: Iterable[str]) -> str:
    order = list(order)
    header = "dataset".ljust(12) + "".join(PRETTY.get(m, m).rjust(20) for m in order) + "average".rjust(14)
    lines = [header, "-" * len(header)]
    datasets = [d for d in DATASET_SPECS if d in table] + [d for d in sorted(table) if d not in DATASET_SPECS]
    averages = defaultdict(list)
    for dataset in datasets:
        row = dataset.ljust(12)
        per_method = []
        for method in order:
            values = table[dataset].get(method, [])
            if values:
                mean = statistics.fmean(values)
                std = statistics.stdev(values) if len(values) > 1 else 0.0
                per_method.append(mean)
                averages[method].append(mean)
                row += f"{mean:8.2f} +/- {std:6.2f}".rjust(20)
            else:
                row += "-".rjust(20)
        row += (f"{statistics.fmean(per_method):8.2f}" if per_method else "-").rjust(14)
        lines.append(row)
    avg_row = "average".ljust(12)
    for method in order:
        values = averages.get(method, [])
        avg_row += (f"{statistics.fmean(values):8.2f}" if values else "-").rjust(20)
    lines.append("-" * len(header))
    lines.append(avg_row)
    return "\n".join(lines)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="aggregate SMM runs into paper tables")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--backbone", default=None, help="filter by backbone name")
    parser.add_argument("--table", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)

    table = collect(args.runs, args.backbone)
    order = ABLATION_ORDER if args.table == 3 else METHOD_ORDER
    print(format_table(table, order))
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({d: {m: v for m, v in per.items()} for d, per in table.items()}, fh, indent=2)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
