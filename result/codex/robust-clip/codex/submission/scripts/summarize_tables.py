#!/usr/bin/env python
"""Aggregate the per-experiment result files into the tables of the paper.

Reads the JSON files written by ``eval_lvlm.py`` / ``eval_zeroshot.py`` and prints
Table 1 style (dataset x encoder, clean + robust) and Table 4 style summaries,
including the average over the datasets (Table 1 averages the four LVLM tasks,
Table 4 averages the zero-shot datasets without ImageNet).

Usage::

    python scripts/summarize_tables.py --encoders CLIP=results/llava_coco_clip.json \
        TeCoA4=results/llava_coco_tecoa4.json FARE4=results/llava_coco_fare4.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description="summarize the LVLM / zero-shot results")
    parser.add_argument("--encoders", nargs="+", required=True, help="NAME=path/to/result.json")
    parser.add_argument("--table", choices=["lvlm", "zeroshot"], default="lvlm")
    return parser.parse_args()


def read(path: str):
    with open(path) as handle:
        payload = json.load(handle)
    return payload.get("results", payload)


def main():
    args = parse_args()
    rows = {}
    for entry in args.encoders:
        name, path = entry.split("=", 1)
        payload = read(path)
        if args.table == "zeroshot":
            rows[name] = {key: value for key, value in payload.items() if isinstance(value, dict)}
        else:
            row = {"clean": payload.get("clean", {}), "robust": payload.get("robust", {})}
            rows[name] = row

    if args.table == "zeroshot":
        datasets = sorted({key for row in rows.values() for key in row if key != "average_zero_shot"})
        header = ["encoder"] + datasets + ["average (w/o ImageNet)"]
        print(" | ".join(header))
        for name, row in rows.items():
            clean = [row[d]["clean"] for d in datasets if d in row]
            values = [f"{sum(clean) / max(len(clean), 1):.1f} (clean)"]
            print(" | ".join([name] + [f"{row[d]['clean']:.1f}" for d in datasets] + values))
        print()
        print("robust accuracy:")
        print(" | ".join(["encoder"] + datasets + ["average"]))
        for name, row in rows.items():
            for key in sorted({k for d in datasets for k in row.get(d, {}) if k.startswith("robust_")}):
                values = [row[d].get(key, float("nan")) for d in datasets]
                valid = [v for v in values if v == v]
                print(" | ".join([f"{name} {key}"] + [f"{v:.1f}" for v in values] + [f"{sum(valid) / max(len(valid), 1):.1f}"]))
        return

    print("LVLM results (metric: cider / accuracy)")
    for name, row in rows.items():
        clean = row["clean"]
        print(f"{name}: clean={clean}")
        for key, values in row["robust"].items():
            print(f"    {key}: {values}")


if __name__ == "__main__":
    main()
