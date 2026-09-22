"""Table 1 of the paper: preliminary presentation of the algorithm's
superiority (Section 5.1).

Setup: MNIST-S (1,000 examples randomly sampled from MNIST), the ConvNet of
Zhou et al. (2022), predefined coreset sizes k = 200 and 400, voluntary
performance compromises eps in {0.2, 0.3, 0.4}, 20 repetitions.

Reported for each configuration:

* the initial objectives f1(m) / f2(m) (one random mask of size k),
* the objectives after lexicographic bilevel coreset selection (mean +/- std).

The expected trend is that both objectives are lower after selection than at
initialisation, and that a larger eps yields a smaller f2.
"""

from __future__ import annotations

import os

import torch

from lbcs.inner_loop import InnerLoopConfig
from lbcs.lbcs import LBCS, LBCSConfig
from lbcs.lexiflow import LexiFlowConfig
from lbcs.objectives import ObjectiveConfig

from .common import (SPECS, base_argparser, inner_cfg, make_bundle, mean_std,
                     model_factory, objective_cfg, resolve_cli, save_json)


def build_config(k: int, epsilon: float, seed: int, T: int, inner_epochs: int,
                 spec, args) -> LBCSConfig:
    return LBCSConfig(
        k=k, epsilon=epsilon, T=T, seed=seed,
        inner=inner_cfg(spec, inner_epochs, args),
        objective=objective_cfg(args, spec),
        lexiflow=LexiFlowConfig(epsilon=epsilon, delta_init=1.0,
                                delta_lower=1e-3,
                                step_decay_patience=10, seed=seed),
    )


def main(argv=None) -> int:
    parser = base_argparser("Table 1: LBCS on MNIST-S")
    parser.add_argument("--ks", type=int, nargs="+", default=[200, 400])
    parser.add_argument("--epsilons", type=float, nargs="+",
                        default=[0.2, 0.3, 0.4])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--mnist-s-size", type=int, default=1000)
    parser.add_argument("--mnist-s-seed", type=int, default=0)
    args = parser.parse_args(argv)

    device = resolve_cli(args)["device"]
    repeats = args.seeds or args.repeats
    if args.dry_run:
        repeats, args.T, args.mnist_s_size = 1, 4, 300
        args.epochs = args.epochs or 5
    inner_epochs = args.epochs or 100

    bundle = make_bundle("mnist-s", args.data_root,
                         mnist_s_size=args.mnist_s_size,
                         mnist_s_seed=args.mnist_s_seed)
    spec = SPECS["mnist-s"]

    summary = {"dataset": "mnist-s", "n": bundle.n, "T": args.T,
               "repeats": repeats, "results": {}}
    for k in args.ks:
        for epsilon in args.epsilons:
            f1s, f2s, init_f1s, init_f2s = [], [], [], []
            for seed in range(repeats):
                cfg = build_config(k, epsilon, seed, args.T, inner_epochs,
                                   spec, args)
                info = LBCS(bundle, model_factory(spec.proxy_model), device,
                            cfg).select()
                f1s.append(info["f1"])
                f2s.append(info["f2"])
                init_f1s.append(float(info["init_objectives"][0]))
                init_f2s.append(float(info["init_objectives"][1]))
                print(f"  k={k} eps={epsilon} seed={seed}: "
                      f"f1={info['f1']:.3f} f2={info['f2']:.0f} "
                      f"(init f1={init_f1s[-1]:.3f})")
            key = f"k{k}_eps{epsilon}"
            summary["results"][key] = {
                "k": k, "epsilon": epsilon,
                "init_f1": mean_std(init_f1s), "init_f2": mean_std(init_f2s),
                "f1": mean_std(f1s), "f2": mean_std(f2s),
            }
    save_json(os.path.join(args.results_dir, "table1_mnist_s.json"), summary)
    _write_table(os.path.join(args.results_dir, "table1_mnist_s.md"),
                 summary["results"], args.ks, args.epsilons)

    for k in args.ks:
        for epsilon in args.epsilons:
            res = summary["results"][f"k{k}_eps{epsilon}"]
            print(f"k={k} eps={epsilon}: f1 {_fmt(res['f1'])} "
                  f"f2 {_fmt(res['f2'], 1)}")
    return 0


def _fmt(stats, digits=2):
    return f"{stats['mean']:.{digits}f} ± {stats['std']:.{digits}f}"


def _write_table(path, results, ks, epsilons):
    lines = ["| k | Objectives | Initial | " +
             " | ".join(f"eps={e}" for e in epsilons) + " |",
             "|---|---|---|" + "|".join(["---"] * len(epsilons)) + "|"]
    for k in ks:
        first = results[f"k{k}_eps{epsilons[0]}"]
        lines.append(f"| {k} | f1 | {first['init_f1']['mean']:.2f} | " +
                     " | ".join(_fmt(results[f"k{k}_eps{e}"]["f1"])
                                for e in epsilons) + " |")
        lines.append(f"| {k} | f2 | {first['init_f2']['mean']:.0f} | " +
                     " | ".join(_fmt(results[f"k{k}_eps{e}"]["f2"], 1)
                                for e in epsilons) + " |")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
