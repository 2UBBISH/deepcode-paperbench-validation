"""Construct coresets for one dataset/size/method and store them on disk.

This decouples *coreset selection* (expensive: it solves the inner loop for
every queried mask) from *target training* (see
``lbcs.lbcs.evaluate_coreset``), which is useful when the two phases run in
different jobs.  Selected coresets are written as ``torch.save``-ed index
tensors next to a small JSON manifest.
"""

from __future__ import annotations

import json
import os

import torch

from lbcs.baselines import BASELINES, probabilistic_select
from lbcs.baselines.pipeline import BaselineSelector, ScoreConfig
from lbcs.baselines.probabilistic import ProbabilisticConfig
from lbcs.lbcs import LBCS, evaluate_coreset
from lbcs.utils import discretize

from .common import SPECS, base_argparser, make_bundle, model_factory, resolve_cli
from .exp_table2_table3 import BASELINE_NAMES, lbcs_config


def select_indices(method, bundle, k, seed, spec, args, device):
    if method == "LBCS":
        info = LBCS(bundle, model_factory(spec.proxy_model), device,
                    lbcs_config(k, seed, spec, args.T,
                                args.epochs or spec.inner_epochs,
                                epsilon=args.epsilon, args=args)).select()
        return torch.nonzero(info["mask"].reshape(-1) > 0.5,
                             as_tuple=False).flatten(), info
    if method == "Probabilistic":
        indices = probabilistic_select(
            bundle, k, model_factory(spec.proxy_model), device,
            ProbabilisticConfig(k=k, T=args.prob_T, C=1, lam=None, seed=seed))
        return indices, None
    selector = BaselineSelector(bundle, model_factory(spec.proxy_model),
                                device,
                                ScoreConfig(epochs=args.epochs or spec.inner_epochs,
                                            batch_size=spec.batch_size,
                                            optimizer=spec.inner_optimizer,
                                            lr=spec.inner_lr),
                                seed=seed)
    return selector.select(method, k), None


def main(argv=None) -> int:
    parser = base_argparser("Select coresets and store them on disk")
    parser.add_argument("--dataset", default="fmnist")
    parser.add_argument("--ks", type=int, nargs="+", default=[1000])
    parser.add_argument("--methods", nargs="+",
                        default=BASELINE_NAMES + ["LBCS"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--T", type=int, default=500)
    parser.add_argument("--prob-T", dest="prob_T", type=int, default=500)
    parser.add_argument("--train-target", action="store_true",
                        help="also train the target model on every coreset")
    args = parser.parse_args(argv)

    device = resolve_cli(args)["device"]
    spec = SPECS[args.dataset]
    bundle = make_bundle(args.dataset, args.data_root)
    out_dir = os.path.join(args.results_dir, "coresets", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    if args.dry_run:
        args.ks, args.T, args.prob_T, args.epochs = [20], 2, 2, 1

    manifest = []
    for seed in range(args.seeds or args.repeats):
        for k in args.ks:
            for method in args.methods:
                indices, info = select_indices(method, bundle, k, seed, spec,
                                               args, device)
                name = f"{method}_k{k}_seed{seed}.pt"
                torch.save(indices.cpu(), os.path.join(out_dir, name))
                entry = {"dataset": args.dataset, "method": method, "k": k,
                         "seed": seed, "size": int(indices.numel()),
                         "file": name}
                if info is not None:
                    entry["f1"] = info["f1"]
                    entry["f2"] = info["f2"]
                if args.train_target:
                    from .exp_table2_table3 import target_cfg_for
                    mask = torch.zeros(bundle.n).scatter_(0, indices, 1.0)
                    metrics = evaluate_coreset(bundle, mask,
                                               model_factory(spec.target_model),
                                               device, target_cfg_for(args, spec),
                                               seed=seed)
                    entry.update(metrics)
                manifest.append(entry)
                print(f"  {entry}")
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
