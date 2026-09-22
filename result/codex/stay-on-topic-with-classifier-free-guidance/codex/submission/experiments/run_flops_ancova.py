#!/usr/bin/env python3
"""Section 4 / Appendix C.2 -- accuracy vs inference FLOPs and ANCOVA.

Consumes the JSON produced by ``run_zeroshot.py`` and asks the question the
paper asks: *for the same inference compute, does a smaller model with CFG
perform as well as a model twice its size?*

For each benchmark the script

1. converts every (model, gamma) point into inference FLOPs per token
   (``cfglm/flops.py``, following the ELECTRA script) and the harness
   accuracy;
2. fits a logistic regression of accuracy on ``log`` FLOPs separately for the
   vanilla and CFG groups (the curves of Figure 9);
3. runs the ANCOVA of ``accuracy ~ log_flops + group`` and reports the
   p-value of the group term, together with which side wins (Table 6), using
   the paper's ``p = .01`` cutoff;
4. counts how many of the nine tasks show a statistically insignificant
   difference ("5 out of 9" in the paper).

Example
-------
    python experiments/run_flops_ancova.py --input results/zeroshot.json \
        --output-dir results/flops
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cfglm.stats import ancova  # noqa: E402


def load_points(path: str) -> Dict[str, List[dict]]:
    with open(path) as fh:
        records = json.load(fh)
    by_task: Dict[str, List[dict]] = defaultdict(list)
    for record in records:
        if record.get("acc") is None:
            continue
        by_task[record["task"]].append(record)
    return by_task


def logistic_regression(x: np.ndarray, y: np.ndarray, steps: int = 500, lr: float = 0.1):
    """Simple gradient-descent logistic fit (avoids a scikit-learn dependency)."""
    x = (x - x.mean()) / (x.std() + 1e-12)
    X = np.column_stack([np.ones_like(x), x])
    w = np.zeros(2)
    for _ in range(steps):
        z = X @ w
        p = 1 / (1 + np.exp(-z))
        grad = X.T @ (p - y) / len(y)
        w -= lr * grad
    return w, x.mean(), x.std()


def predict_line(w, mean, std, x_raw):
    x = (x_raw - mean) / (std + 1e-12)
    z = w[0] + w[1] * x
    return 1 / (1 + np.exp(-z))


def analyse_task(task: str, points: List[dict]) -> dict:
    x = np.log(np.array([p["flops_per_token"] for p in points], dtype=float))
    y = np.array([p["acc"] for p in points], dtype=float)
    group = np.array([1 if p["cfg"] else 0 for p in points], dtype=int)
    if len(points) < 4 or len(set(group.tolist())) < 2:
        return {
            "p_value": float("nan"),
            "coef_group": float("nan"),
            "winner": "n/a",
            "insufficient_data": True,
        }
    result = ancova(x, y, group, interaction=True)
    # direction of the effect: which side has the higher fitted accuracy at
    # the mean log-FLOP of the CFG group
    mean_cfg_flops = float(np.mean(x[group == 1])) if (group == 1).any() else float(np.mean(x))
    result["winner"] = "CFG" if result["coef_group"] > 0 else "Vanilla"
    result["mean_cfg_logflops"] = mean_cfg_flops
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="results/zeroshot.json")
    parser.add_argument("--output-dir", default="results/flops")
    parser.add_argument("--cutoff", type=float, default=0.01)
    parser.add_argument("--make-plots", action="store_true", help="write the Figure 9 style plots")
    args = parser.parse_args()

    by_task = load_points(args.input)
    os.makedirs(args.output_dir, exist_ok=True)

    table = []
    significant = 0
    for task in sorted(by_task):
        points = by_task[task]
        if len(points) < 4:
            continue
        res = analyse_task(task, points)
        significant += int(res["p_value"] <= args.cutoff)
        verdict = (
            f"{res['winner']} (p={res['p_value']:.3f})"
            if res["p_value"] <= args.cutoff
            else f"p>.01 ({res['p_value']:.3f})"
        )
        table.append((task, res["p_value"], res["winner"], verdict))
        print(f"{task:<16} p={res['p_value']:.4f}  winner={res['winner']:<8} {verdict}")

        if args.make_plots:
            try:
                _plot_task(task, points, os.path.join(args.output_dir, f"accuracy_vs_flops_{task}.png"))
            except Exception as exc:  # pragma: no cover - plotting is optional
                print(f"[flops] could not plot {task}: {exc}")

    n_tasks = len(table)
    print(
        f"\n{significant}/{n_tasks} tasks are significantly different at p={args.cutoff}; "
        f"{n_tasks - significant}/{n_tasks} show no significant difference "
        "(the paper reports 5/9)."
    )

    with open(os.path.join(args.output_dir, "ancova_table.json"), "w") as fh:
        json.dump(
            [{"task": t, "p_value": p, "winner": w, "verdict": v} for t, p, w, v in table],
            fh,
            indent=2,
        )


def _plot_task(task: str, points: List[dict], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    for label, is_cfg, colour in (("Vanilla", False, "tab:blue"), ("CFG", True, "tab:red")):
        pts = [p for p in points if p["cfg"] == is_cfg]
        if not pts:
            continue
        x = np.log(np.array([p["flops_per_token"] for p in pts], dtype=float))
        y = np.array([p["acc"] for p in pts], dtype=float)
        ax.scatter(x, y, s=18, label=label, color=colour)
        if len(pts) >= 3:
            w, mean, std = logistic_regression(x, y)
            grid = np.linspace(x.min(), x.max(), 50)
            ax.plot(grid, predict_line(w, mean, std, grid), "--", color=colour)
    ax.set_xlabel("log FLOPs per token (inference)")
    ax.set_ylabel("accuracy")
    ax.set_title(task)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
