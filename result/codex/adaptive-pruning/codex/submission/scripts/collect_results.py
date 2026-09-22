#!/usr/bin/env python3
"""Collect ``result.json`` files into a single table (one row per run)."""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List


def flatten(prefix: str, obj: Any, out: Dict[str, Any]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(f"{prefix}.{k}" if prefix else k, v, out)
    else:
        out[prefix] = obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="runs/*/result.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(args.glob)):
        with open(path) as fh:
            data = json.load(fh)
        row: Dict[str, Any] = {"run": os.path.dirname(path)}
        flatten("", data.get("config", {}), row)
        flatten("", {k: v for k, v in data.items() if k not in {"config", "logs", "tta_history"}}, row)
        rows.append(row)

    if not rows:
        print("no result.json found")
        return 1
    keys = sorted({k for r in rows for k in r})
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(rows, fh, indent=2)
    try:
        import pandas as pd

        df = pd.DataFrame(rows)[keys]
        print(df.to_string(index=False))
    except Exception:
        for r in rows:
            print(json.dumps(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
