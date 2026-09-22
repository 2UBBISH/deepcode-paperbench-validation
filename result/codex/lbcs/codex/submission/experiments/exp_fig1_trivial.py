"""Figure 1 of the paper: phenomena of the trivial solutions (Section 2.1).

Four panels, all on a random subset of MNIST with the ConvNet of
Zhou et al. (2022):

    (a) f1(m) vs. outer iterations with equation (3)  [f1 only]
    (b) f2(m) vs. outer iterations with equation (3)
    (c) f1(m) vs. outer iterations with equation (4)  [weighted combination]
    (d) f2(m) vs. outer iterations with equation (4)

Settings (Appendix C.3 + addendum):

* MNIST-S, i.e. an arbitrary random subset of MNIST;
* inner loop: 100 epochs of SGD, learning rate 0.1, momentum 0.9;
* outer loop: Adam with learning rate 2.5 and a cosine scheduler;
* lambda = 0.5 for equation (4);
* T = 1000 iterations of the outer loop.

The figure is expected to show that the un-modified bilevel method minimises
f1 while keeping the coreset size near the predefined k (panels a and b),
whereas the naive weighted combination collapses the coreset size at the price
of a large f1 (panels c and d).
"""

from __future__ import annotations

import os

import torch

from lbcs.baselines.probabilistic import (ProbabilisticConfig,
                                          ProbabilisticCoreset)
from lbcs.objectives import ObjectiveConfig
from lbcs.utils import set_seed

from .common import (SPECS, base_argparser, inner_cfg, make_bundle,
                     markdown_table, model_factory, resolve_cli, save_json,
                     write_rows_csv)


def run_condition(bundle, device, lam, k, T, seed, inner, verbose,
                  outer_lr: float = 2.5):
    cfg = ProbabilisticConfig(
        k=k, T=T, C=1, lam=lam, s_lr=outer_lr, s_scheduler="cosine", seed=seed,
        inner=inner,
        objective=ObjectiveConfig(eval_batch_size=512),
        verbose=verbose,
    )
    set_seed(seed)
    runner = ProbabilisticCoreset(bundle, model_factory("convnet"), device, cfg)
    return runner.run()


def main(argv=None) -> int:
    parser = base_argparser("Figure 1: trivial solutions of RCS")
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--lambda-value", type=float, default=0.5)
    parser.add_argument("--outer-lr", dest="outer_lr", type=float, default=2.5,
                        help="Adam learning rate on the probabilities "
                             "(Appendix C.3 states 2.5; the reference "
                             "implementation of Zhou et al. (2022) uses 5e-2)")
    parser.add_argument("--mnist-s-size", type=int, default=1000)
    parser.add_argument("--mnist-s-seed", type=int, default=0)
    parser.add_argument("--paper-inner-loop", dest="paper_inner_loop",
                        action="store_true",
                        help="use the literal Appendix C.3 inner loop "
                             "(SGD, lr 0.1, momentum 0.9, 100 epochs)")
    args = parser.parse_args(argv)

    device = resolve_cli(args)["device"]
    if args.dry_run:
        args.T, args.mnist_s_size = 4, 200
        args.epochs = args.epochs or 5
    inner_epochs = args.epochs or 100
    if args.paper_inner_loop:
        args.inner_optimizer, args.inner_lr = "sgd", 0.1
    inner = inner_cfg(SPECS["mnist-s"], inner_epochs, args, warm_start=False)
    results_dir = args.results_dir

    bundle = make_bundle("mnist-s", args.data_root,
                         mnist_s_size=args.mnist_s_size,
                         mnist_s_seed=args.mnist_s_seed)

    runs = {}
    for tag, lam in (("eq3", None), ("eq4", args.lambda_value)):
        result = run_condition(bundle, device, lam, args.k, args.T, seed=0,
                               inner=inner, verbose=not args.dry_run,
                               outer_lr=args.outer_lr)
        runs[tag] = result
        write_rows_csv(os.path.join(results_dir, f"fig1_{tag}_trace.csv"),
                       result["trace"])

    save_json(os.path.join(results_dir, "fig1_summary.json"), {
        "k": args.k, "T": args.T, "lambda": args.lambda_value,
        "mnist_s_size": args.mnist_s_size,
        "eq3_final": {"f1": runs["eq3"]["trace"][-1]["f1"],
                      "f2": runs["eq3"]["trace"][-1]["f2"]},
        "eq4_final": {"f1": runs["eq4"]["trace"][-1]["f1"],
                      "f2": runs["eq4"]["trace"][-1]["f2"]},
    })
    _plot(runs, results_dir, args.k)

    rows = [
        {"equation": "(3) f1 only", "f1": f"{runs['eq3']['trace'][-1]['f1']:.3f}",
         "f2": f"{runs['eq3']['trace'][-1]['f2']:.1f}",
         "expected size": f"{runs['eq3']['trace'][-1]['f2_expected']:.1f}"},
        {"equation": f"(4) lambda={args.lambda_value}",
         "f1": f"{runs['eq4']['trace'][-1]['f1']:.3f}",
         "f2": f"{runs['eq4']['trace'][-1]['f2']:.1f}",
         "expected size": f"{runs['eq4']['trace'][-1]['f2_expected']:.1f}"},
    ]
    print(markdown_table(rows, ["equation", "f1", "f2", "expected size"]))
    return 0


def _plot(runs, results_dir, k):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                       # pragma: no cover
        print(f"[fig1] matplotlib unavailable ({exc}); skipping plot")
        return
    os.makedirs(results_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for col, (tag, label, lam) in enumerate(
            [("eq3", "equation (3): f1 only", None),
             ("eq4", "equation (4): weighted", 0.5)]):
        trace = runs[tag]["trace"]
        iters = [t["iter"] for t in trace]
        axes[0][col].plot(iters, [t["f1"] for t in trace])
        axes[0][col].set_title(f"({'ac'[col]}) f1, {label}")
        axes[0][col].set_xlabel("outer iteration")
        axes[0][col].set_ylabel("f1(m)")
        axes[1][col].plot(iters, [t["f2"] for t in trace], label="sampled |m|_0")
        axes[1][col].plot(iters, [t["f2_expected"] for t in trace],
                          label="E|m|_0 = sum s")
        axes[1][col].axhline(k, ls="--", c="grey", label=f"predefined k={k}")
        axes[1][col].set_title(f"({'bd'[col]}) f2, {label}")
        axes[1][col].set_xlabel("outer iteration")
        axes[1][col].set_ylabel("f2(m)")
        axes[1][col].legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(results_dir, "fig1_trivial_solutions.png")
    fig.savefig(path, dpi=150)
    print(f"[fig1] wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
