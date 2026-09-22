#!/usr/bin/env python
"""Aggregate per-seed evaluation JSON files into the paper's result tables.

Reads files named ``<domain>-<prior>-s<seed>.json`` (as produced by
``evaluate_fre.py``), averages each task suite across seeds, and emits
Table 1 / Table 4 style summaries (mean +/- standard deviation over 5 seeds,
with 20 evaluation episodes per seed, matching the paper).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np


NAME_RE = re.compile(r"^(?P<domain>[a-z]+)-(?P<prior>.+)-s(?P<seed>\d+)\.json$")


def load_results(input_dir: str) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Return ``{prior: {seed: {suite: score}}}``.

    Suite names are unique across the three domains, so evaluations from all
    domains are merged into a single table; this is what makes the combined
    ``exorl-all`` and ``all`` rows of Table 1 possible.
    """
    results: Dict[str, Dict[int, Dict[str, float]]] = defaultdict(dict)
    for fname in sorted(os.listdir(input_dir)):
        match = NAME_RE.match(fname)
        if not match:
            continue
        prior, seed = match.group("prior"), int(match.group("seed"))
        with open(os.path.join(input_dir, fname)) as f:
            payload = json.load(f)
        per_suite = results[prior].setdefault(seed, {})
        for suite_name, suite in payload.items():
            if isinstance(suite, dict) and "mean" in suite:
                per_suite[suite_name] = float(suite["mean"])
    return results


def aggregate(results) -> Dict[str, Dict[str, Dict[str, float]]]:
    """``{prior: {suite: {mean, std, num_seeds}}}``."""
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for prior, seeds in results.items():
        suites = defaultdict(list)
        for _seed, per_suite in seeds.items():
            for suite, score in per_suite.items():
                suites[suite].append(score)
        out[prior] = {
            suite: {
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
                "num_seeds": len(v),
            }
            for suite, v in suites.items()
        }
    return out


# The four AntMaze task sets shown in Figure 5 (path-all is the average of the
# three corridor tasks).
FIGURE_5_SUITES = ["ant-goal-reaching", "ant-directional", "ant-random-simplex", "ant-path-all"]


def figure5_normalize(
    aggregated: Dict[str, Dict[str, Dict[str, float]]],
    suites: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Normalise scores the way Figure 5 does.

    The addendum: "the returns are normalized by dividing by the maximum return
    that any agent scores on that task set.  Thus there are four columns that
    have a normalized return of 1 (one for each task set)."  Figure 5 compares
    the FRE variants (FRE-all, FRE-goals, FRE-lin, FRE-mlp, FRE-lin-mlp,
    FRE-goal-mlp, FRE-goal-lin) on the four AntMaze task sets.
    """
    priors = aggregated
    suites = list(suites or FIGURE_5_SUITES)
    suites = [s for s in suites if any(s in p for p in priors.values())]
    maxima = {
        suite: max(priors[p][suite]["mean"] for p in priors if suite in priors[p])
        for suite in suites
    }
    return {
        prior: {
            suite: (entry["mean"] / maxima[suite] if maxima[suite] else 0.0)
            for suite, entry in suites_of.items()
            if suite in suites
        }
        for prior, suites_of in priors.items()
    }


# The row groupings used by Table 1: each combined row is the mean of the
# suites that make it up.
TABLE_1_ROWS = {
    "antmaze-all": [
        "ant-goal-reaching",
        "ant-directional",
        "ant-random-simplex",
        "ant-path-loop",
        "ant-path-edges",
        "ant-path-center",
    ],
    "exorl-all": [
        "exorl-walker-goals",
        "exorl-cheetah-goals",
        "exorl-walker-velocity",
        "exorl-cheetah-velocity",
    ],
    "kitchen-all": ["kitchen"],
}


def combined_rows(prior_scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Average the per-suite scores into the combined rows of Table 1."""
    rows: Dict[str, float] = {}
    for row, suites in TABLE_1_ROWS.items():
        present = [prior_scores[s]["mean"] for s in suites if s in prior_scores]
        if present:
            rows[row] = float(np.mean(present))
    # The final row averages the AntMaze, ExORL and Kitchen groups.
    groups = [rows[k] for k in ("antmaze-all", "exorl-all", "kitchen-all") if k in rows]
    if groups:
        rows["all"] = float(np.mean(groups))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="eval")
    parser.add_argument("--output", default="eval/table1.json")
    parser.add_argument("--figure5", action="store_true",
                        help="also emit the Figure 5 (max-normalised) AntMaze numbers")
    args = parser.parse_args()

    aggregated = aggregate(load_results(args.input_dir))
    # Add the combined Table 1 rows (antmaze-all / exorl-all / kitchen-all / all)
    # to every prior's entry.
    for prior, suites in aggregated.items():
        if not suites:
            continue
        num_seeds = suites[next(iter(suites))]["num_seeds"]
        for row, value in combined_rows(suites).items():
            suites[row] = {"mean": value, "std": float("nan"), "num_seeds": num_seeds}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(aggregated, f, indent=2, default=float)

    # Print a compact Table-1 style summary.
    priors = sorted(aggregated)
    all_suites = sorted({s for p in aggregated.values() for s in p})
    print(f"\n=== Table 1 / Table 4 style summary ===")
    print(f"{'suite':<28}" + "".join(f"{p:>17}" for p in priors))
    for suite in all_suites:
        row = f"{suite:<28}"
        for prior in priors:
            entry = aggregated[prior].get(suite)
            row += f"{'--':>17}" if entry is None else f"{entry['mean']:>11.1f}±{entry['std']:<5.1f}"
        print(row)

    if args.figure5:
        f5 = figure5_normalize(aggregated)
        out_path = os.path.splitext(args.output)[0] + "_figure5.json"
        with open(out_path, "w") as f:
            json.dump(f5, f, indent=2, default=float)
        print("\n=== Figure 5 (AntMaze, max-normalised) ===")
        for prior, scores in f5.items():
            row = f"{prior:<16}" + "".join(f"{v:>14.2f}" for v in scores.values())
            print(row)


if __name__ == "__main__":
    main()
