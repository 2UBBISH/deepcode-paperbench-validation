#!/usr/bin/env python3
"""Regenerate the paper's tables and figures from a results directory.

    python scripts/reproduce_tables.py --results-dir results --out-dir tables

Produces:

* ``table1.csv``          -- model performance vs mistake severity
* ``table2.csv`` + text   -- ID LCA/Top1 vs OOD Top1/Top5 (R^2 / PEA)
* ``table3.csv``          -- error-prediction MAE of every baseline
* ``figure1.png``         -- LCA unifies VMs and VLMs
* ``figure5.png``         -- the four severely shifted OOD datasets
* ``figure9.png``         -- LCA vs accuracy on the same dataset

No dataset or model is loaded: everything is computed from the cached logits
written by ``lca_on_the_line.evaluate``.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lca_on_the_line.analysis import (  # noqa: E402
    compute_ranking_table,
    compute_table1,
    compute_table2,
    compute_table3,
    figure1,
    figure5,
    figure9,
    format_ranking_table,
    format_table2,
    load_metrics,
)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="tables")
    parser.add_argument("--skip-table3", action="store_true")
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_metrics(args.results_dir)
    print("loaded %d rows for %d models" % (len(df), df["model"].nunique()))

    table1 = compute_table1(df)
    table1.to_csv(os.path.join(args.out_dir, "table1.csv"))
    print("\n== Table 1 ==\n", table1.round(4))

    table2 = compute_table2(df)
    print("\n== Table 2 (R2 / PEA) ==\n", format_table2(table2))
    with open(os.path.join(args.out_dir, "table2.txt"), "w") as fh:
        fh.write(format_table2(table2) + "\n")

    ranking = compute_ranking_table(df)
    print("\n== Ranking measures (KEN / SPE) ==\n", format_ranking_table(ranking))
    with open(os.path.join(args.out_dir, "ranking_measures.txt"), "w") as fh:
        fh.write(format_ranking_table(ranking) + "\n")

    if not args.skip_table3:
        table3 = compute_table3(df, args.results_dir)
        table3.to_csv(os.path.join(args.out_dir, "table3.csv"))
        print("\n== Table 3 (MAE, lower is better) ==\n", table3.round(4))

    figure1(df, os.path.join(args.out_dir, "figure1.png"))
    figure5(df, os.path.join(args.out_dir, "figure5.png"))
    figure9(df, os.path.join(args.out_dir, "figure9.png"))
    print("\nwrote tables and figures to", args.out_dir)


if __name__ == "__main__":
    main()
