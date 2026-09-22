#!/usr/bin/env python
"""Collect the run outputs into tables that mirror the paper's layout.

Reads the artifacts produced by the other scripts and writes markdown tables:

* ``table2_main_results.md``  -- accuracy and delta per method/dataset (Section 4.2)
* ``table3_plug_and_play.md`` -- plug-and-play deltas (Section 4.3)
* ``table4_cost.md``          -- accuracy vs. training/inference cost (Section 4.4)
* ``table5_ablation.md``      -- MLM vs NCE loss (Section 4.5)
* ``table6_vram.md``          -- Mixtral-8x7B accuracy and VRAM (Sections 4.6/4.7)
* ``figure3_scale.md``        -- beams / iterations (Section 4.6)
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

SETTING_LABELS = {
    "ground_truth": "BBox-Adapter (Ground-Truth)",
    "ai_feedback": "BBox-Adapter (AI Feedback)",
    "combined": "BBox-Adapter (Combined)",
}


def read_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def best_per_setting(rows: List[Dict], dataset: str) -> Dict[str, float]:
    """Table 2 reports the best of the 0.1B and 0.3B adapters per setting."""

    best: Dict[str, float] = {}
    for row in rows:
        if row.get("dataset") != dataset:
            continue
        setting = row.get("setting")
        value = row.get("accuracy")
        if value is None:
            continue
        if setting not in best or value > best[setting]:
            best[setting] = value
    return best


def format_delta(base: Optional[float], value: Optional[float]) -> str:
    if base is None or value is None:
        return ""
    return f"{value - base:+.2f}"


def table2(runs_root: str, datasets: List[str]) -> str:
    summary = read_json(os.path.join(runs_root, "table2_summary.json")) or []
    lines = [
        "| Method | " + " | ".join(datasets) + " |",
        "|" + "---|" * (len(datasets) + 1),
    ]
    cot: Dict[str, Optional[float]] = {}
    for dataset in datasets:
        payload = read_json(os.path.join(runs_root, "baselines", dataset, "cot_results.json"))
        cot[dataset] = payload.get("accuracy") if payload else None
    lines.append(
        "| gpt-3.5-turbo | "
        + " | ".join(f"{cot[d]:.2f}" if cot[d] is not None else "n/a" for d in datasets)
        + " |"
    )
    for setting, label in SETTING_LABELS.items():
        cells = []
        for dataset in datasets:
            best = best_per_setting(summary, dataset).get(setting)
            cells.append(
                f"{best:.2f} ({format_delta(cot[dataset], best)})" if best is not None else "n/a"
            )
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def simple_table(path: str, title: str) -> str:
    payload = read_json(path)
    if not payload:
        return f"_{title}: {path} not found_\n"
    if isinstance(payload, dict):
        payload = [{"key": key, "value": value} for key, value in payload.items()]
    headers = list(payload[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in payload:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=str, default="runs/table2")
    parser.add_argument("--output-dir", type=str, default="runs/tables")
    parser.add_argument("--datasets", type=str,
                        default="strategyqa,gsm8k,truthfulqa,scienceqa")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    outputs = {
        "table2_main_results.md": table2(args.runs_root, datasets),
        "table3_plug_and_play.md": simple_table(
            os.path.join(args.runs_root, "plug_and_play.json"), "Table 3"),
        "table4_cost.md": simple_table(
            os.path.join(args.runs_root, "cost_results.json"), "Table 4"),
        "table5_ablation.md": simple_table(
            os.path.join(args.runs_root, "ablation_results.json"), "Table 5"),
        "table6_vram.md": simple_table(
            os.path.join(args.runs_root, "vram_results.json"), "Table 6"),
        "figure3_scale.md": simple_table(
            os.path.join(args.runs_root, "scale_results.json"), "Figure 3"),
    }
    for name, content in outputs.items():
        path = os.path.join(args.output_dir, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
        print(f"[tables] wrote {path}")


if __name__ == "__main__":
    main()
