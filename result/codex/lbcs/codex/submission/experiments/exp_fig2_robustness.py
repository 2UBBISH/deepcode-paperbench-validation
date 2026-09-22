"""Figure 2 of the paper: robustness against imperfect supervision (Section 5.3).

Two panels on F-MNIST:

* (a) coreset selection with 30% symmetric label noise -- the labels of 30% of
  the training data are flipped;
* (b) coreset selection with class-imbalanced data -- exponential class
  imbalance with an imbalanced ratio of 0.01.

The roles of the LBCS's voluntary compromise eps (Remark 2) are the point of
the figure: a small compromise of f1 prevents the coreset selection from
overfitting the corrupted supervision, which helps generalisation.

All methods are evaluated with the same protocol as Section 5.2 (LeNet proxy
and LeNet target, Adam 1e-3, 100 epochs, ten repetitions).
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import torch

from lbcs.baselines.pipeline import BaselineSelector, ScoreConfig
from lbcs.baselines.probabilistic import ProbabilisticConfig
from lbcs.baselines import probabilistic_select
from lbcs.data import make_class_imbalanced, make_noisy_bundle
from lbcs.lbcs import LBCS, evaluate_coreset, TargetTrainingConfig
from lbcs.utils import discretize

from .common import (SPECS, base_argparser, inner_cfg, make_bundle, mean_std,
                     model_factory, objective_cfg, resolve_cli, save_json,
                     write_rows_csv)
from .exp_table2_table3 import (ALL_METHODS, BASELINE_NAMES, lbcs_config,
                                target_cfg_for)


def run_condition(bundle, tag: str, args, device, results_dir: str) -> Dict:
    dataset = "fmnist"
    spec = SPECS[dataset]
    repeats = args.seeds or args.repeats
    if args.dry_run:
        repeats = 1
    records: List[dict] = []
    proxy_cfg = ScoreConfig(epochs=args.epochs or spec.inner_epochs,
                            batch_size=spec.batch_size,
                            optimizer=spec.inner_optimizer, lr=spec.inner_lr,
                            num_models=args.proxy_models)
    for seed in range(repeats):
        selector = BaselineSelector(bundle, model_factory(spec.proxy_model),
                                    device, proxy_cfg, seed=seed)
        for k in args.ks:
            for method in args.methods:
                if method == "LBCS":
                    info = LBCS(bundle, model_factory(spec.proxy_model), device,
                                lbcs_config(k, seed, spec, args.T,
                                            args.epochs or spec.inner_epochs,
                                            epsilon=args.epsilon, args=args)
                                ).select()
                    mask = info["mask"]
                elif method == "Probabilistic":
                    indices = probabilistic_select(
                        bundle, k, model_factory(spec.proxy_model), device,
                        ProbabilisticConfig(
                            k=k, T=args.prob_T, C=1, lam=None, seed=seed,
                            inner=inner_cfg(
                                spec, args.epochs or spec.inner_epochs, args),
                            objective=objective_cfg(args, spec)))
                    mask = torch.zeros(bundle.n).scatter_(0, indices, 1.0)
                else:
                    indices = selector.select(method, k)
                    mask = torch.zeros(bundle.n).scatter_(0, indices, 1.0)
                metrics = evaluate_coreset(bundle, mask,
                                           model_factory(spec.target_model),
                                           device, target_cfg_for(args, spec), seed=seed)
                records.append({
                    "condition": tag, "k": k, "method": method, "seed": seed,
                    "test_accuracy": metrics["test_accuracy"],
                    "coreset_size": metrics["coreset_size"],
                })
                print(f"  [{tag}] k={k} {method}: "
                      f"acc={metrics['test_accuracy']:.2f} "
                      f"size={metrics['coreset_size']}")
                write_rows_csv(os.path.join(results_dir,
                                            f"fig2_{tag}.csv"), records)
    return {"condition": tag, "records": records}


def main(argv=None) -> int:
    parser = base_argparser("Figure 2: robustness to imperfect supervision")
    parser.add_argument("--ks", type=int, nargs="+",
                        default=[1000, 2000, 3000, 4000])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS)
    parser.add_argument("--noise-rate", dest="noise_rate", type=float,
                        default=0.3)
    parser.add_argument("--imbalance-ratio", dest="imbalance_ratio",
                        type=float, default=0.01)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--T", type=int, default=500)
    parser.add_argument("--prob-T", dest="prob_T", type=int, default=500)
    parser.add_argument("--proxy-models", dest="proxy_models", type=int,
                        default=1)
    args = parser.parse_args(argv)

    device = resolve_cli(args)["device"]
    if args.dry_run:
        args.ks, args.T, args.prob_T, args.epochs = [50], 3, 3, 1

    clean = make_bundle("fmnist", args.data_root)
    noisy = make_noisy_bundle(clean, args.noise_rate, seed=0)
    imbalanced = make_bundle("fmnist", args.data_root)
    x, y = make_class_imbalanced(imbalanced.train_x, imbalanced.train_y,
                                 imbalanced.num_classes, args.imbalance_ratio,
                                 seed=0)
    from lbcs.data import DatasetBundle
    imbalanced = DatasetBundle("fmnist-imb", x, y, imbalanced.test_x,
                               imbalanced.test_y, imbalanced.num_classes)

    payload = {}
    conditions = [("noise30", noisy), ("imbalanced", imbalanced)] \
        if args.noise_rate == 0.3 else \
        [(f"noise{int(args.noise_rate * 100)}", noisy),
         ("imbalanced", imbalanced)]
    for tag, bundle in conditions:
        print(f"[fig2] condition={tag} n={bundle.n}")
        payload[tag] = run_condition(bundle, tag, args, device,
                                     args.results_dir)
    save_json(os.path.join(args.results_dir, "fig2_robustness.json"), payload)
    _plot(payload, args.ks, args.results_dir)
    return 0


def _plot(payload, ks, results_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                       # pragma: no cover
        print(f"[fig2] matplotlib unavailable ({exc}); skipping plot")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (tag, out) in zip(axes, payload.items()):
        for method in ALL_METHODS:
            ys = []
            for k in ks:
                vals = [r["test_accuracy"] for r in out["records"]
                        if r["k"] == k and r["method"] == method]
                ys.append(mean_std(vals)["mean"] if vals else np.nan)
            ax.plot(ks, ys, marker="o", label=method)
        ax.set_xlabel("predefined coreset size k")
        ax.set_ylabel("test accuracy (%)")
        ax.set_title(tag)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(results_dir, "fig2_robustness.png")
    fig.savefig(path, dpi=150)
    print(f"[fig2] wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
