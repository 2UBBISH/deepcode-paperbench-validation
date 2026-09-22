"""Tables 2 and 3 of the paper: comparison with the competitors (Section 5.2).

Table 2 (same *predefined* coreset size).  For every benchmark
(F-MNIST, SVHN, CIFAR-10), every predefined size k in {1000, 2000, 3000, 4000}
and every method (Uniform, EL2N, GraNd, Influential, Moderate, CCS,
Probabilistic, LBCS) the coreset is constructed, a *target* network is trained
on the coreset and the test accuracy is measured.  For LBCS the optimised
coreset size is reported as well.

Table 3 (the *same* coreset size for everybody).  The coreset size found by
LBCS is applied to all the baselines, so that the comparison isolates the
quality of the selection.

Settings (Section 5.2):

* coreset selection networks: LeNet for F-MNIST, simple CNNs for SVHN and
  CIFAR-10, Adam with learning rate 1e-3 for the inner loop, eps = 0.2, T = 500;
* target networks: LeNet (F-MNIST), CNN (SVHN), ResNet-18 (CIFAR-10);
* target training: Adam lr 1e-3 / 100 epochs for F-MNIST and SVHN, SGD lr 0.1
  with a cosine schedule / 200 epochs for CIFAR-10;
* all experiments are repeated ten times.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch

from lbcs.baselines import probabilistic_select
from lbcs.baselines.pipeline import BaselineSelector, ScoreConfig
from lbcs.baselines.probabilistic import ProbabilisticConfig
from lbcs.data import DatasetBundle
from lbcs.inner_loop import InnerLoopConfig
from lbcs.lbcs import LBCS, LBCSConfig, evaluate_coreset, TargetTrainingConfig
from lbcs.lexiflow import LexiFlowConfig
from lbcs.objectives import ObjectiveConfig
from lbcs.utils import discretize

from .common import (SPECS, base_argparser, make_bundle, mean_std,
                     inner_cfg, model_factory, objective_cfg, resolve_cli,
                     save_json, target_epochs, write_rows_csv)

BASELINE_NAMES = ["Uniform", "EL2N", "GraNd", "Influential", "Moderate",
                  "CCS", "Probabilistic"]
ALL_METHODS = BASELINE_NAMES + ["LBCS"]


def lbcs_config(k: int, seed: int, spec, T: int, inner_epochs: int,
                epsilon: float = 0.2, group_size: int = 1,
                init_indices: Optional[torch.Tensor] = None,
                init_label: str = "random", args=None) -> LBCSConfig:
    return LBCSConfig(
        k=k, epsilon=epsilon, T=T, seed=seed, group_size=group_size,
        inner=inner_cfg(spec, inner_epochs, args),
        objective=objective_cfg(args, spec),
        lexiflow=LexiFlowConfig(epsilon=epsilon, delta_init=1.0,
                                delta_lower=1e-3, step_decay_patience=10,
                                seed=seed),
        init_indices=init_indices, init_label=init_label,
    )


def target_cfg(spec) -> TargetTrainingConfig:
    return TargetTrainingConfig(epochs=spec.target_epochs,
                                batch_size=spec.batch_size,
                                optimizer=spec.target_optimizer,
                                lr=spec.target_lr,
                                momentum=0.9,
                                weight_decay=spec.target_weight_decay,
                                scheduler=spec.target_scheduler)


def target_cfg_for(args, spec) -> TargetTrainingConfig:
    """Target-training configuration honouring ``--target-epochs``."""
    cfg = target_cfg(spec)
    cfg.epochs = target_epochs(args, spec)
    return cfg


def run_one_dataset(dataset: str, args, device, results_dir: str) -> Dict:
    bundle = make_bundle(dataset, args.data_root)
    spec = SPECS[dataset]
    repeats = args.seeds or args.repeats
    if args.dry_run:
        repeats = 1
    proxy_cfg = ScoreConfig(epochs=(args.proxy_epochs or args.epochs
                                    or spec.inner_epochs),
                            batch_size=spec.batch_size,
                            optimizer=spec.inner_optimizer, lr=spec.inner_lr,
                            num_models=args.proxy_models)

    records: List[dict] = []
    lbcs_masks: Dict[int, List[float]] = defaultdict(list)

    for seed in range(repeats):
        print(f"[{dataset}] repetition {seed + 1}/{repeats}")
        selector = None
        need_scores = [m for m in args.methods
                       if m in BASELINE_NAMES and m != "Probabilistic"]
        if need_scores:
            selector = BaselineSelector(bundle, model_factory(spec.proxy_model),
                                        device, proxy_cfg, seed=seed)

        for k in args.ks:
            for method in args.methods:
                if method == "LBCS":
                    info = LBCS(bundle, model_factory(spec.proxy_model), device,
                                lbcs_config(k, seed, spec, args.T,
                                            args.epochs or spec.inner_epochs,
                                            epsilon=args.epsilon,
                                            group_size=args.group_size,
                                            args=args)
                                ).select()
                    indices = torch.nonzero(
                        info["mask"].reshape(-1) > 0.5,
                        as_tuple=False).flatten()
                    lbcs_masks[k].append(float(indices.numel()))
                elif method == "Probabilistic":
                    pcfg = ProbabilisticConfig(
                        k=k, T=args.prob_T, C=1, lam=None, seed=seed,
                        inner=inner_cfg(spec, args.epochs or spec.inner_epochs,
                                        args))
                    indices = probabilistic_select(bundle,
                                                   k,
                                                   model_factory(spec.proxy_model),
                                                   device, pcfg)
                else:
                    indices = selector.select(method, k)

                metrics = evaluate_coreset(bundle, torch.zeros(bundle.n).scatter_(
                    0, indices, 1.0), model_factory(spec.target_model), device,
                    target_cfg_for(args, spec), seed=seed)
                records.append({
                    "dataset": dataset, "k": k, "method": method, "seed": seed,
                    "repetition": seed, "table": "table2",
                    "test_accuracy": metrics["test_accuracy"],
                    "coreset_size": metrics["coreset_size"],
                    "accuracy_per_1000_datapoints":
                        metrics["accuracy_per_datapoint"],
                })
                print(f"    k={k} {method}: acc={metrics['test_accuracy']:.2f} "
                      f"size={metrics['coreset_size']}")
                _flush(results_dir, dataset, records)

    # ---- Table 3: rerun the baselines at the LBCS coreset sizes ---------
    if not args.skip_table3:
        for k in args.ks:
            sizes = lbcs_masks[k]
            if not sizes:
                continue
            k_ours = int(round(float(np.mean(sizes))))
            for seed in range(repeats):
                selector = BaselineSelector(
                    bundle, model_factory(spec.proxy_model), device, proxy_cfg,
                    seed=seed)
                for method in [m for m in args.methods
                               if m in BASELINE_NAMES
                               and m != "Probabilistic"]:
                    indices = selector.select(method, k_ours)
                    metrics = evaluate_coreset(
                        bundle,
                        torch.zeros(bundle.n).scatter_(0, indices, 1.0),
                        model_factory(spec.target_model), device,
                        target_cfg_for(args, spec), seed=seed)
                    records.append({
                        "dataset": dataset, "k": k, "method": method,
                        "seed": seed, "repetition": seed, "table": "table3",
                        "reference_size_k_ours": k_ours,
                        "test_accuracy": metrics["test_accuracy"],
                        "coreset_size": metrics["coreset_size"],
                    })
                    _flush(results_dir, dataset, records)

    return {"dataset": dataset, "records": records,
            "lbcs_sizes": {k: mean_std(v) for k, v in lbcs_masks.items()}}


def _flush(results_dir: str, dataset: str, records: List[dict]) -> None:
    write_rows_csv(os.path.join(results_dir, f"table2_table3_{dataset}.csv"),
                   records)


def summarise(dataset: str, records: List[dict], ks: List[int]) -> None:
    print(f"\n=== {dataset} (Table 2) ===")
    rows = []
    for k in ks:
        row = {"k": k}
        for method in ALL_METHODS:
            vals = [r["test_accuracy"] for r in records
                    if r["table"] == "table2" and r["k"] == k
                    and r["method"] == method]
            if vals:
                row[method] = f"{mean_std(vals)['mean']:.1f} ± "
                row[method] += f"{mean_std(vals)['std']:.1f}"
        sizes = [r["coreset_size"] for r in records
                 if r["table"] == "table2" and r["k"] == k
                 and r["method"] == "LBCS"]
        if sizes:
            row["Coreset size (ours)"] = f"{mean_std(sizes)['mean']:.1f} ± "
            row["Coreset size (ours)"] += f"{mean_std(sizes)['std']:.1f}"
        rows.append(row)
    columns = ["k"] + ALL_METHODS + ["Coreset size (ours)"]
    print("| " + " | ".join(columns) + " |")
    for row in rows:
        print("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")


def main(argv=None) -> int:
    parser = base_argparser("Tables 2 and 3: comparison with competitors")
    parser.add_argument("--datasets", nargs="+",
                        default=["fmnist", "svhn", "cifar10"])
    parser.add_argument("--ks", type=int, nargs="+",
                        default=[1000, 2000, 3000, 4000])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--T", type=int, default=500)
    parser.add_argument("--prob-T", dest="prob_T", type=int, default=500)
    parser.add_argument("--group-size", dest="group_size", type=int, default=1)
    parser.add_argument("--proxy-epochs", dest="proxy_epochs", type=int,
                        default=None)
    parser.add_argument("--proxy-models", dest="proxy_models", type=int,
                        default=1)
    parser.add_argument("--skip-table3", action="store_true")
    args = parser.parse_args(argv)

    device = resolve_cli(args)["device"]
    if args.dry_run:
        args.ks = [50, 100]
        args.T, args.prob_T = 3, 3
        args.epochs = 1

    payload = {}
    for dataset in args.datasets:
        out = run_one_dataset(dataset, args, device, args.results_dir)
        payload[dataset] = out
        summarise(dataset, out["records"], args.ks)
    save_json(os.path.join(args.results_dir, "table2_table3.json"), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
