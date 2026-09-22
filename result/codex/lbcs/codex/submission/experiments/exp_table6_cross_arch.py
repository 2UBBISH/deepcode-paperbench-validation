"""Table 6 of the paper: cross network architecture evaluation (Section 6).

SVHN coresets constructed with the proxy CNN are used to train two *other*
target architectures -- ViT-small and WideResNet (W-NET) -- to show that the
proposed selection is not tied to a specific network.  All other settings are
unchanged with respect to Section 5.2.

Row groups of Table 6 correspond to the two target architectures; the exact
coreset sizes of LBCS are the ones reported in Table 2.
"""

from __future__ import annotations

import os

import torch

from lbcs.baselines import probabilistic_select
from lbcs.baselines.pipeline import BaselineSelector, ScoreConfig
from lbcs.baselines.probabilistic import ProbabilisticConfig
from lbcs.inner_loop import InnerLoopConfig
from lbcs.lbcs import LBCS, evaluate_coreset
from lbcs.utils import discretize

from .common import (SPECS, base_argparser, inner_cfg, make_bundle, mean_std,
                     model_factory, objective_cfg, resolve_cli, save_json,
                     write_rows_csv)
from .exp_table2_table3 import ALL_METHODS, BASELINE_NAMES, lbcs_config


def main(argv=None) -> int:
    parser = base_argparser("Table 6: cross architecture evaluation on SVHN")
    parser.add_argument("--ks", type=int, nargs="+",
                        default=[1000, 2000, 3000, 4000])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS)
    parser.add_argument("--targets", nargs="+", default=["vit", "wideresnet"])
    parser.add_argument("--dataset", default="svhn",
                        help="benchmark of the coreset selection (the paper "
                             "uses SVHN)")
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--T", type=int, default=500)
    parser.add_argument("--prob-T", dest="prob_T", type=int, default=500)
    args = parser.parse_args(argv)

    device = resolve_cli(args)["device"]
    spec = SPECS[args.dataset]
    repeats = args.seeds or args.repeats
    if args.epochs is None:
        args.epochs = spec.inner_epochs
    if args.dry_run:
        args.ks, args.T, args.prob_T, args.epochs, repeats = [50], 3, 3, 1, 1

    bundle = make_bundle(args.dataset, args.data_root)
    proxy_cfg = ScoreConfig(epochs=args.epochs, batch_size=spec.batch_size,
                            optimizer=spec.inner_optimizer, lr=spec.inner_lr,
                            num_models=1)
    records = []
    for seed in range(repeats):
        selector = BaselineSelector(bundle, model_factory(spec.proxy_model),
                                    device, proxy_cfg, seed=seed)
        for k in args.ks:
            for method in args.methods:
                if method == "LBCS":
                    info = LBCS(bundle, model_factory(spec.proxy_model), device,
                                lbcs_config(k, seed, spec, args.T, args.epochs,
                                            epsilon=args.epsilon,
                                            args=args)).select()
                    mask = info["mask"]
                elif method == "Probabilistic":
                    indices = probabilistic_select(
                        bundle, k, model_factory(spec.proxy_model), device,
                        ProbabilisticConfig(
                            k=k, T=args.prob_T, C=1, lam=None, seed=seed,
                            inner=inner_cfg(spec, args.epochs, args),
                            objective=objective_cfg(args, spec)))
                    mask = torch.zeros(bundle.n).scatter_(0, indices, 1.0)
                else:
                    indices = selector.select(method, k)
                    mask = torch.zeros(bundle.n).scatter_(0, indices, 1.0)

                for target in args.targets:
                    from lbcs.lbcs import TargetTrainingConfig
                    tcfg = TargetTrainingConfig(epochs=args.epochs,
                                                batch_size=spec.batch_size,
                                                optimizer=spec.target_optimizer,
                                                lr=spec.target_lr,
                                                momentum=0.9)
                    metrics = evaluate_coreset(
                        bundle, mask, model_factory(target), device, tcfg,
                        seed=seed)
                    records.append({"k": k, "method": method, "seed": seed,
                                    "target": target,
                                    "test_accuracy": metrics["test_accuracy"],
                                    "coreset_size": metrics["coreset_size"]})
                    print(f"  k={k} {method} -> {target}: "
                          f"acc={metrics['test_accuracy']:.2f}")
                    write_rows_csv(os.path.join(args.results_dir,
                                                "table6_cross_arch.csv"),
                                   records)

    save_json(os.path.join(args.results_dir, "table6_cross_arch.json"),
              {"records": records})
    print("\n=== Table 6 (SVHN) ===")
    for target in args.targets:
        print(f"\n-- target = {target}")
        print("| k | " + " | ".join(ALL_METHODS) + " |")
        print("|---|" + "|".join(["---"] * len(ALL_METHODS)) + "|")
        for k in args.ks:
            cells = []
            for method in ALL_METHODS:
                vals = [r["test_accuracy"] for r in records
                        if r["k"] == k and r["method"] == method
                        and r["target"] == target]
                stats = mean_std(vals)
                cells.append(f"{stats['mean']:.1f} ± {stats['std']:.1f}"
                             if stats["n"] else "")
            print(f"| {k} | " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
