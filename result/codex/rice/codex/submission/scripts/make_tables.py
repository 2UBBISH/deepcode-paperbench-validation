#!/usr/bin/env python3
"""Print the tables of the paper side by side with the reproduction results.

    python scripts/make_tables.py --results results

The reference numbers (Table 1, Table 4, ...) come from
``docs/paper_reference.json`` so that the general trends can be checked
directly against the values reported in the paper.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load(path):
    with open(path) as handle:
        return json.load(handle)


def _fmt(value, digits=2):
    if value is None:
        return "-"
    try:
        return "{:.{}f}".format(float(value), digits)
    except (TypeError, ValueError):
        return str(value)


def load_refining(results_dir):
    out = {}
    for path in glob.glob(os.path.join(results_dir, "**", "*_refining.json"), recursive=True):
        data = _load(path)
        out[data["env"]] = data
    return out


def load_fidelity(results_dir):
    out = {}
    for path in glob.glob(os.path.join(results_dir, "**", "*_fidelity.json"), recursive=True):
        data = _load(path)
        out[data["env"]] = data
    return out


def load_run_one(results_dir):
    """Merge the single-job files written by ``scripts/run_one.py``."""
    merged = {}
    pattern = os.path.join(results_dir, "**", "*_seed*.json")
    for path in glob.glob(pattern, recursive=True):
        data = _load(path)
        if not all(key in data for key in ("env", "method", "seed", "final_return")):
            continue
        key = (data["env"], data.get("explanation", "ours"), data["method"])
        merged.setdefault(key, []).append(float(data["final_return"]))
    return merged


def table_run_one(runs):
    if not runs:
        return ""
    envs = sorted({key[0] for key in runs})
    lines = [
        "| Task | Explanation | rice | ppo | jsrl | statemask_r |",
        "|---|---|---|---|---|---|",
    ]
    for env in envs:
        explanations = sorted({key[1] for key in runs if key[0] == env})
        for explanation in explanations:
            row = [env, explanation]
            for method in ("rice", "ppo", "jsrl", "statemask_r"):
                values = runs.get((env, explanation, method), [])
                row.append(
                    "{:.2f} (n={})".format(sum(values) / len(values), len(values))
                    if values
                    else "-"
                )
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def table_1(refining, reference):
    header = (
        "| Task | No Refine | PPO | JSRL | StateMask-R | Ours | Random expl | "
        "StateMask expl | paper: No Refine -> Ours |"
    )
    lines = [header, "|" + "---|" * 9]
    paper = reference["table_1_agent_refining_performance"]["rows"]
    for env, data in sorted(refining.items()):
        vary_refine = data.get("vary_refine", {})
        vary_expl = data.get("vary_explanation", {})
        no_refine = data.get("no_refine", {}).get("mean")
        row = [
            env,
            _fmt(no_refine),
            _fmt(vary_refine.get("ppo", {}).get("mean")),
            _fmt(vary_refine.get("jsrl", {}).get("mean")),
            _fmt(vary_refine.get("statemask_r", {}).get("mean")),
            _fmt(vary_refine.get("ours", {}).get("mean")),
            _fmt(vary_expl.get("random", {}).get("mean")),
            _fmt(vary_expl.get("statemask", {}).get("mean")),
        ]
        if env in paper:
            row.append(
                "{} -> {}".format(_fmt(paper[env][0]), _fmt(paper[env][4]))
            )
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def table_fidelity(fidelity):
    lines = ["| Task | Explanation | " + " | ".join(
        "fidelity K={:.0%}".format(k) for k in (0.1, 0.2, 0.3, 0.4)
    ) + " |", "|" + "---|" * 6]
    for env, data in sorted(fidelity.items()):
        for method, values in sorted(data["fidelity"].items()):
            cells = [
                "{:.2f} ± {:.2f}".format(m, s)
                for m, s in zip(values["mean"], values["std"])
            ]
            lines.append(
                "| {} | {} | {} |".format(env, method, " | ".join(cells))
            )
    return "\n".join(lines)


def table_4(fidelity, reference):
    ref = reference["table_4_mask_training_time_seconds"]
    lines = [
        "| Task | samples | StateMask (s) | Ours (s) | reduction (%) | paper reduction (%) |",
        "|---|---|---|---|---|---|",
    ]
    for env, data in sorted(fidelity.items()):
        efficiency = data.get("efficiency", {})
        ours = efficiency.get("ours", {}).get("wall_time")
        sm = efficiency.get("statemask", {}).get("wall_time")
        samples = efficiency.get("ours", {}).get("samples") or ref["samples"].get(env)
        reduction = efficiency.get("time_reduction_percent")
        if reduction is None and ours and sm:
            reduction = 100.0 * (1.0 - ours / sm)
        paper_reduction = None
        if env in ref["Ours"] and env in ref["StateMask"]:
            paper_reduction = 100.0 * (
                1.0 - ref["Ours"][env] / ref["StateMask"][env]
            )
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                env,
                samples,
                _fmt(sm, 1),
                _fmt(ours, 1),
                _fmt(reduction, 1),
                _fmt(paper_reduction, 1),
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument(
        "--reference", default=os.path.join("docs", "paper_reference.json")
    )
    args = parser.parse_args()

    reference = _load(args.reference)
    refining = load_refining(args.results)
    fidelity = load_fidelity(args.results)

    print("## Table 1 -- agent refining performance (reproduction vs paper)\n")
    print(table_1(refining, reference) if refining else "(no refining results yet)")
    print("\n## Fidelity scores (Experiment I)\n")
    print(table_fidelity(fidelity) if fidelity else "(no fidelity results yet)")
    print("\n## Table 4 -- mask training cost (Experiment I)\n")
    print(table_4(fidelity, reference) if fidelity else "(no fidelity results yet)")
    runs = load_run_one(args.results)
    if runs:
        print("\n## Single jobs (scripts/run_one.py), mean final reward\n")
        print(table_run_one(runs))


if __name__ == "__main__":
    main()
