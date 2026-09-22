#!/usr/bin/env python3
"""Offline re-analysis of a JudgeEval results.json -> confusion matrix + lenient/strict bias.

No API calls, no cost. Usage:
    python analyze_judge_eval_bias.py <results.json> [<data_dir>]
data_dir defaults to ./data/judge_eval
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def leaves(node: dict, require_valid: bool = False) -> list[dict]:
    if require_valid and not node.get("valid_score", True):
        return []
    if not node.get("sub_tasks"):
        return [node]
    out: list[dict] = []
    for s in node["sub_tasks"]:
        out.extend(leaves(s, require_valid))
    return out


def analyze(results_path: Path, data_dir: Path) -> None:
    res = json.loads(results_path.read_text())
    print(f"judge_type={res['judge_type']}  code_only={res['code_only']}")
    print(f"model={res['judge_kwargs'].get('completer_config', {}).get('model')}")

    grand: Counter[str] = Counter()
    for r in res["results"]:
        exp = json.loads(
            (data_dir / r["example_id"] / "grading" / "expected_result.json").read_text()
        )
        truth = {n["id"]: n["score"] for n in leaves(exp)}

        cm: Counter[str] = Counter()
        rows = []
        for leaf in leaves(r["graded_task_tree"], require_valid=True):
            pred, gt = leaf["score"], truth[leaf["id"]]
            cell = ("TP" if gt else "FP") if pred else ("FN" if gt else "TN")
            cm[cell] += 1
            rows.append((leaf["id"], cell, leaf.get("task_category"), leaf.get("requirements")))
        grand.update(cm)
        report(r["example_id"], cm)
        (results_path.parent / f"leaf_verdicts_{r['example_id'].replace('/', '_')}.json").write_text(
            json.dumps(rows, indent=1)
        )

    if len(res["results"]) > 1:
        report("POOLED", grand)


def report(label: str, cm: Counter[str]) -> None:
    tp, fp, fn, tn = cm["TP"], cm["FP"], cm["FN"], cm["TN"]
    n = tp + fp + fn + tn
    print(f"\n--- {label}  n={n}")
    print(f"  TP={tp}  FP={fp} (LENIENT errors)  FN={fn} (STRICT errors)  TN={tn}")
    print(f"  accuracy      {(tp + tn) / n:.4f}")
    print(f"  sensitivity   {tp / (tp + fn):.4f}   (recall on human-PASS leaves)")
    print(f"  specificity   {tn / (tn + fp):.4f}   (recall on human-FAIL leaves)")
    f1p = 2 * tp / (2 * tp + fp + fn)
    f1n = 2 * tn / (2 * tn + fn + fp)
    print(f"  macro F1      {(f1p + f1n) / 2:.4f}   (f1_pass={f1p:.4f} f1_fail={f1n:.4f})")
    print(f"  judge pass rate {(tp + fp) / n:.4f}  vs human pass rate {(tp + fn) / n:.4f}")
    net = fp - fn
    print(f"  NET BIAS: {'LENIENT' if net > 0 else 'STRICT'} by {abs(net)} leaves "
          f"({abs(net) / n * 100:.1f} pp of the leaf set)")


if __name__ == "__main__":
    analyze(Path(sys.argv[1]), Path(sys.argv[2] if len(sys.argv) > 2 else "data/judge_eval"))
