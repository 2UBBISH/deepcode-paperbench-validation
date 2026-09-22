"""Table 5 of the paper: influence of the mask initialisation (Section 6).

"LBCS+Moderate" means that the mask is initialised by the Moderate coreset and
then refined by LBCS, instead of starting from a random mask.  The experiment
is run on F-MNIST with the predefined sizes k = 1000, 2000, 3000, 4000 and ten
repetitions.
"""

from __future__ import annotations

import os

import torch

from lbcs.baselines.pipeline import BaselineSelector, ScoreConfig
from lbcs.inner_loop import InnerLoopConfig
from lbcs.lbcs import LBCS, evaluate_coreset

from .common import (SPECS, base_argparser, make_bundle, mean_std,
                     model_factory, resolve_cli, save_json, write_rows_csv)
from .exp_table2_table3 import lbcs_config, target_cfg_for


def main(argv=None) -> int:
    parser = base_argparser("Table 5: LBCS with Moderate mask initialisation")
    parser.add_argument("--ks", type=int, nargs="+",
                        default=[1000, 2000, 3000, 4000])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--T", type=int, default=500)
    args = parser.parse_args(argv)

    device = resolve_cli(args)["device"]
    spec = SPECS["fmnist"]
    repeats = args.seeds or args.repeats
    if args.dry_run:
        args.ks, args.T, args.epochs, repeats = [50], 3, 1, 1

    bundle = make_bundle("fmnist", args.data_root)
    proxy_cfg = ScoreConfig(epochs=args.epochs or spec.inner_epochs,
                            batch_size=spec.batch_size,
                            optimizer=spec.inner_optimizer, lr=spec.inner_lr,
                            num_models=1)
    records = []
    for seed in range(repeats):
        selector = BaselineSelector(bundle, model_factory(spec.proxy_model),
                                    device, proxy_cfg, seed=seed)
        for k in args.ks:
            moderate_idx = selector.select("moderate", k)
            for variant, init_idx, init_label in (
                    ("LBCS", None, "random"),
                    ("LBCS+Moderate", moderate_idx, "moderate")):
                cfg = lbcs_config(k, seed, spec, args.T,
                                  args.epochs or spec.inner_epochs,
                                  epsilon=args.epsilon,
                                  init_indices=init_idx,
                                  init_label=init_label, args=args)
                info = LBCS(bundle, model_factory(spec.proxy_model), device,
                            cfg).select()
                metrics = evaluate_coreset(bundle, info["mask"],
                                           model_factory(spec.target_model),
                                           device, target_cfg_for(args, spec), seed=seed)
                records.append({"k": k, "method": variant, "seed": seed,
                                "test_accuracy": metrics["test_accuracy"],
                                "coreset_size": metrics["coreset_size"]})
                print(f"  k={k} {variant}: acc={metrics['test_accuracy']:.2f} "
                      f"size={metrics['coreset_size']}")
                write_rows_csv(os.path.join(args.results_dir,
                                            "table5_moderate_init.csv"),
                               records)

    payload = {"records": records}
    save_json(os.path.join(args.results_dir, "table5_moderate_init.json"),
              payload)
    print("\n=== Table 5 (F-MNIST) ===")
    print("| k | LBCS | LBCS+Moderate |")
    print("|---|---|---|")
    for k in args.ks:
        cells = []
        for variant in ("LBCS", "LBCS+Moderate"):
            vals = [r["test_accuracy"] for r in records
                    if r["k"] == k and r["method"] == variant]
            stats = mean_std(vals)
            cells.append(f"{stats['mean']:.1f} ± {stats['std']:.1f}")
        print(f"| {k} | {cells[0]} | {cells[1]} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
