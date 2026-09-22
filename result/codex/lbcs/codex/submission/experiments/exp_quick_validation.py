"""Reduced-budget validation of LBCS on MNIST-S (a smoke test of Table 1).

The full configuration of Section 5.1 (20 repetitions of `T = 1000` outer
iterations with a 100-step inner loop) is far too expensive to run as a smoke
test, so this script runs the *same* pipeline with a reduced budget and checks
the qualitative claims of the paper:

* the optimised objectives are lower than the initialised ones
  (`f_1` improves and `f_2`, the coreset size, becomes smaller than `k`);
* a larger voluntary compromise `eps` leads to a smaller coreset.

Run it with e.g.

    python -m experiments.exp_quick_validation --T 100 --epochs 10
"""

from __future__ import annotations

import os

import torch

from lbcs.lbcs import LBCS

from .common import (SPECS, base_argparser, inner_cfg, make_bundle, markdown_table,
                     model_factory, objective_cfg, resolve_cli, save_json,
                     write_rows_csv)
from .exp_table2_table3 import lbcs_config


def main(argv=None) -> int:
    parser = base_argparser("Quick validation of LBCS on MNIST-S")
    parser.add_argument("--ks", type=int, nargs="+", default=[200, 400])
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.2, 0.4])
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--mnist-s-size", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args(argv)

    device = resolve_cli(args)["device"]
    if args.dry_run:
        args.T, args.mnist_s_size, args.ks = 5, 300, [50]
        args.epochs = args.epochs or 5
    if args.epochs is None:
        args.epochs = 10

    bundle = make_bundle("mnist-s", args.data_root,
                         mnist_s_size=args.mnist_s_size)
    spec = SPECS["mnist-s"]
    records = []
    for k in args.ks:
        for epsilon in args.epsilons:
            for seed in range(args.seeds or args.repeats):
                cfg = lbcs_config(k, seed, spec, args.T, args.epochs,
                                  epsilon=epsilon, args=args)
                info = LBCS(bundle, model_factory(spec.proxy_model), device,
                            cfg).select()
                init_f1 = float(info["init_objectives"][0])
                init_f2 = float(info["init_objectives"][1])
                records.append({
                    "k": k, "epsilon": epsilon, "seed": seed, "T": args.T,
                    "inner_epochs": args.epochs,
                    "init_f1": init_f1, "init_f2": init_f2,
                    "final_f1": info["f1"], "final_f2": info["f2"],
                    "f1_improved": info["f1"] < init_f1,
                    "f2_reduced": info["f2"] < init_f2,
                    "num_queries": info["num_queries"],
                })
                print(f"k={k} eps={epsilon} seed={seed}: "
                      f"f1 {init_f1:.3f} -> {info['f1']:.3f}, "
                      f"f2 {init_f2:.0f} -> {info['f2']:.0f}")
                write_rows_csv(os.path.join(args.results_dir,
                                            "quick_validation.csv"), records)

    save_json(os.path.join(args.results_dir, "quick_validation.json"),
              {"records": records})
    print()
    print(markdown_table(
        [{"k": r["k"], "eps": r["epsilon"],
          "f1 (init)": f"{r['init_f1']:.3f}",
          "f1 (final)": f"{r['final_f1']:.3f}",
          "f2 (init)": f"{r['init_f2']:.0f}",
          "f2 (final)": f"{r['final_f2']:.0f}"} for r in records],
        ["k", "eps", "f1 (init)", "f1 (final)", "f2 (init)", "f2 (final)"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
