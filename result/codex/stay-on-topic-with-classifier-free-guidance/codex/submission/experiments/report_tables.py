#!/usr/bin/env python3
"""Render the paper's tables from the JSON produced by the experiments.

* **Table 5** -- zero-shot benchmarks, one row per model, each cell showing
  the ``gamma = 1`` baseline and the best CFG result (the paper shows
  ``gamma = 1.5``).
* **Table 2 / 7 / 8 / 9** -- CodeGen pass@k per guidance strength and
  temperature.
* **Table 6** -- ANCOVA p-values from ``run_flops_ancova.py``.

Example
-------
    python experiments/report_tables.py \
        --zeroshot results/zeroshot.json \
        --humaneval results/humaneval.json \
        --ancova results/flops/ancova_table.json \
        --output results/tables.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cfglm.tasks import TABLE5_TASKS  # noqa: E402

TASK_HEADERS = {
    "arc_challenge": "ARC-c",
    "arc_easy": "ARC-e",
    "boolq": "BoolQ",
    "hellaswag": "HellaSwag",
    "piqa": "PiQA",
    "sciq": "SciQ",
    "triviaqa": "TriviaQA",
    "winogrande": "WinoGrande",
    "lambada_openai": "LAMBADA",
}


def _metric(record: dict) -> Optional[float]:
    return record.get("acc") if record.get("acc") is not None else record.get("substring_match")


def table5(records: List[dict], headline_gamma: float = 1.5) -> str:
    """The zero-shot table (Table 5) in the paper's layout."""
    by_model: Dict[str, List[dict]] = defaultdict(list)
    for record in records:
        by_model[record["model"]].append(record)

    header = ["Model"] + [TASK_HEADERS.get(t, t) for t in TABLE5_TASKS]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for model in sorted(by_model):
        cells = [model]
        for task in TABLE5_TASKS:
            items = [r for r in by_model[model] if r["task"] == task]
            if not items:
                cells.append("-")
                continue
            baseline = next((_metric(r) for r in items if r["gamma"] == 1.0), None)
            ours = next((_metric(r) for r in items if r["gamma"] == headline_gamma), None)
            if ours is None:
                best = max(items, key=lambda r: _metric(r) or 0)
                ours = _metric(best)
            def fmt(value):
                return "-" if value is None else f"{100 * value:.1f}"
            delta = "" if (baseline is None or ours is None) else ("*" if ours > baseline else "")
            cells.append(f"{fmt(baseline)} / {fmt(ours)}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def humaneval_table(records: List[dict], temperature: float = 0.2, ks=(1, 10, 100)) -> str:
    """CodeGen pass@k table (Table 2 for temperature 0.2)."""
    rows = [r for r in records if abs(r["temperature"] - temperature) < 1e-9]
    models = sorted({r["model"] for r in rows})
    gammas = sorted({r["gamma"] for r in rows})
    header = ["gamma"] + [f"{m} k={k}" for m in models for k in ks]
    lines = [f"temperature = {temperature}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for gamma in gammas:
        cells = [f"{gamma}"]
        for model in models:
            match = next((r for r in rows if r["model"] == model and r["gamma"] == gamma), None)
            for k in ks:
                value = match["pass@k"].get(str(k), match["pass@k"].get(k)) if match else None
                cells.append("-" if value is None else f"{100 * float(value):.1f}%")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def ancova_table(path: str) -> str:
    """Table 6: ANCOVA p-values and which side wins."""
    with open(path) as fh:
        rows = json.load(fh)
    lines = ["| Task | p-value | Win |", "|---|---|---|"]
    for row in sorted(rows, key=lambda r: r["p_value"]):
        if row["p_value"] != row["p_value"]:  # NaN
            verdict = "n/a"
        elif row["p_value"] > 0.01:
            verdict = "p>.01"
        else:
            verdict = row["winner"]
        lines.append(f"| {row['task']} | {row['p_value']:.3f} | {verdict} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zeroshot", default=None)
    parser.add_argument("--humaneval", default=None)
    parser.add_argument("--ancova", default=None)
    parser.add_argument("--headline-gamma", type=float, default=1.5)
    parser.add_argument("--output", default="results/tables.md")
    args = parser.parse_args()

    sections: List[str] = []
    if args.zeroshot and os.path.exists(args.zeroshot):
        with open(args.zeroshot) as fh:
            sections.append("# Table 5 -- general natural language benchmarks\n\n" + table5(json.load(fh), args.headline_gamma))
    if args.humaneval and os.path.exists(args.humaneval):
        with open(args.humaneval) as fh:
            records = json.load(fh)
        body = [f"# Table 2 / 7-9 -- CodeGen HumanEval pass@k\n"]
        for temperature in sorted({r["temperature"] for r in records}):
            body.append(humaneval_table(records, temperature))
        sections.append("\n\n".join(body))
    if args.ancova and os.path.exists(args.ancova):
        sections.append("# Table 6 -- ANCOVA p-values\n\n" + ancova_table(args.ancova))

    text = "\n\n".join(sections) if sections else "no inputs given"
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as fh:
        fh.write(text + "\n")
    print(text)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
