"""Gradient-norm analysis of the trivial solution (Section 2.1, Appendix C.2).

Appendix C.2 derives the gradient of the outer loop of the weighted-combination
objective (equation (4)) for the probabilistic (Bernoulli) reparameterisation
of the mask:

    grad_s [ E f_1 + E f_2 ] = E[ f_1(m) (m - s) / (s (1 - s)) ] + 1

so the two gradient-norm scales that compete in the update are

    zeta_1(lambda) = (1 - lambda) || f_1(m) (m - s) / (s (1 - s)) ||_2
    zeta_2(lambda) = lambda * sqrt(n)

and, because the second one equals ``sqrt(n) / 2`` at ``lambda = 1/2`` while
``n`` is large in coreset selection, the minimisation of ``f_2`` dominates:
the coreset size is driven down while ``f_1`` stays large.  This script
measures both quantities on MNIST-S with the ConvNet proxy and reports
``zeta_2 / zeta_1`` for a grid of ``lambda`` values.
"""

from __future__ import annotations

import math
import os

import torch

from lbcs.baselines.probabilistic import ProbabilisticCoreset, ProbabilisticConfig
from lbcs.inner_loop import InnerLoopConfig
from lbcs.objectives import ObjectiveConfig
from lbcs.utils import set_seed

from .common import (base_argparser, make_bundle, markdown_table, model_factory,
                     resolve_cli, save_json, write_rows_csv)


def main(argv=None) -> int:
    parser = base_argparser("Gradient-norm analysis of equation (4)")
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument("--lambdas", type=float, nargs="+",
                        default=[0.1, 0.25, 0.5])
    parser.add_argument("--mnist-s-size", type=int, default=1000)
    parser.add_argument("--mnist-s-seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20,
                        help="outer steps observed before measuring")
    args = parser.parse_args(argv)

    device = resolve_cli(args)["device"]
    if args.dry_run:
        args.mnist_s_size, args.steps, args.epochs = 200, 2, 1

    bundle = make_bundle("mnist-s", args.data_root,
                         mnist_s_size=args.mnist_s_size,
                         mnist_s_seed=args.mnist_s_seed)
    n = bundle.n

    cfg = ProbabilisticConfig(
        k=args.k, T=args.steps, C=1, lam=None, seed=0,
        inner=InnerLoopConfig(epochs=args.epochs or 100, batch_size=128,
                              optimizer="sgd", lr=0.1, momentum=0.9,
                              warm_start=False),
        objective=ObjectiveConfig(eval_batch_size=512))
    set_seed(0)
    runner = ProbabilisticCoreset(bundle, model_factory("convnet"), device, cfg)
    result = runner.run()
    s = result["s"]
    mask = result["mask"]
    f1 = result["trace"][-1]["f1"]

    # the sampled/recovered mask and the current probabilities
    score = f1 * (mask - s) / (s * (1.0 - s) + 1e-12)
    norm_score = float(score.norm().item())

    rows = []
    for lam in args.lambdas:
        zeta1 = (1.0 - lam) * norm_score
        zeta2 = lam * math.sqrt(n)
        rows.append({"lambda": lam, "zeta1": zeta1, "zeta2": zeta2,
                     "zeta2/zeta1": zeta2 / max(zeta1, 1e-12)})
    write_rows_csv(os.path.join(args.results_dir, "gradient_analysis.csv"),
                   rows)
    save_json(os.path.join(args.results_dir, "gradient_analysis.json"),
              {"n": n, "f1": f1, "grad_norm_f1_term": norm_score,
               "sqrt_n": math.sqrt(n), "rows": rows,
               "lbc_result": {"coreset_size": result["coreset_size"],
                              "f1": f1}})
    print(markdown_table(
        [{"lambda": f"{r['lambda']:.2f}", "zeta_1(lambda)": f"{r['zeta1']:.3f}",
          "zeta_2(lambda)": f"{r['zeta2']:.3f}",
          "zeta_2 / zeta_1": f"{r['zeta2 / zeta1']:.2f}"} for r in rows],
        ["lambda", "zeta_1(lambda)", "zeta_2(lambda)", "zeta_2 / zeta_1"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
