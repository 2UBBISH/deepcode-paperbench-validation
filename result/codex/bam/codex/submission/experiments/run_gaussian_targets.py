"""Section 5.1 (first part) -- Gaussian targets of increasing dimension.

Reproduces Figure 5.1 (forward KL divergence vs. number of gradient evaluations)
and Figure E.3 (the same for the reverse KL divergence).

Protocol (paper + Appendix E.3):

* targets: ``p = N(0, Sigma_*)`` with ``Sigma_* = A A^T`` and ``A`` a random
  ``D x D`` matrix, for ``D = 4, 16, 64, 256``;
* initialization ``mu_0 ~ Uniform[0, 0.1]``, ``Sigma_0 = I``;
* BaM uses the constant inverse regularization ``lambda_t = B D``;
* ADVI / Score / Fisher / GSM use batch size ``B = 2`` (their learning rates are
  the ones selected by the grid search reported in Appendix E.3, and can be
  re-derived with ``--grid-search``);
* 10 independent runs per configuration, curves show the mean;
* the metrics are the forward and reverse KL divergences between ``q`` and ``p``.

Example
-------
    python experiments/run_gaussian_targets.py --dims 4 16 --n-runs 2 --quick
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import jax

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from bam.grid_search import PAPER_GRIDS, PAPER_SELECTED, grid_search  # noqa: E402
from bam.runner import (  # noqa: E402
    bam_spec,
    gradient_spec,
    gsm_spec,
    plot_curves,
    run_method,
    run_repeats,
    summarise_curves,
)
from bam.targets import random_gaussian_target  # noqa: E402
from bam.utils import save_json  # noqa: E402

DEFAULT_DIMS = (4, 16, 64, 256)
DEFAULT_BAM_BATCHES = (2, 5, 10, 20, 40)
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def build_specs(dim: int, bam_batches, gradient_batches: int, max_grad_evals: int, bam_iter_cap: int,
                selected: dict | None = None):
    """Method specifications for one target dimension."""
    sel = PAPER_SELECTED["gaussian"] if selected is None else selected
    specs = []
    for B in bam_batches:
        n_iters = int(min(max_grad_evals // B, bam_iter_cap))
        specs.append(bam_spec(B, n_iters, lam_value=float(B * dim)))
    grad_iters = int(max_grad_evals // gradient_batches)
    score_lr = sel["Score"]
    specs.append(gradient_spec("ADVI", gradient_batches, grad_iters, sel["ADVI"]))
    specs.append(gradient_spec("Fisher", gradient_batches, grad_iters, sel["Fisher"]))
    specs.append(gradient_spec("Score", gradient_batches, grad_iters,
                               score_lr[dim] if isinstance(score_lr, dict) else score_lr))
    specs.append(gsm_spec(gradient_batches, grad_iters))
    return specs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dims", type=int, nargs="+", default=list(DEFAULT_DIMS))
    ap.add_argument("--bam-batches", type=int, nargs="+", default=list(DEFAULT_BAM_BATCHES))
    ap.add_argument("--gradient-batch", type=int, default=2,
                    help="batch size of ADVI / Score / Fisher / GSM (paper: 2)")
    ap.add_argument("--max-grad-evals", type=int, default=100000)
    ap.add_argument("--bam-iter-cap", type=int, default=2000)
    ap.add_argument("--n-runs", type=int, default=10)
    ap.add_argument("--n-eval-samples", type=int, default=20000,
                    help="Monte-Carlo samples for non-Gaussian metrics")
    ap.add_argument("--target-scale", type=float, default=1.0 / np.sqrt(2.0),
                    help="scale of the random matrix A in Sigma_* = A A^T (Appendix E.3); the "
                         "default gives marginal variances of order D/2, as for unscaled A A^T")
    ap.add_argument("--grid-search", action="store_true",
                    help="re-derive the learning rates of the gradient-based methods by grid search")
    ap.add_argument("--outdir", default=RESULTS)
    ap.add_argument("--quick", action="store_true", help="tiny run for smoke testing")
    args = ap.parse_args()

    if args.quick:
        args.dims = args.dims[:2]
        args.n_runs = min(args.n_runs, 1)
        args.max_grad_evals = 2000
        args.bam_iter_cap = 200
        args.bam_batches = args.bam_batches[:2]

    os.makedirs(args.outdir, exist_ok=True)
    summary = {"config": vars(args), "curves": {}}

    for dim in args.dims:
        selected = None
        if args.grid_search:
            # a short pilot run for each candidate learning rate on the first seed
            target = random_gaussian_target(dim, seed=0, scale=args.target_scale)
            from bam.runner import evaluate_metric, random_init

            def make_run(loss):
                def run(lr):
                    spec = gradient_spec({"elbo": "ADVI", "score": "Score", "fisher": "Fisher"}[loss],
                                         args.gradient_batch, max(50, args.max_grad_evals // 20),
                                         lr)
                    return run_method(target, spec, jax.random.PRNGKey(0),
                                      random_init(jax.random.PRNGKey(1), dim), np.eye(dim))

                return run

            def score_fn(metric):
                return lambda out: float(evaluate_metric(target, out, metric, jax.random.PRNGKey(2))[-1])

            best = {}
            for loss, metric in (("elbo", "forward_kl"), ("score", "score_div"), ("fisher", "forward_kl")):
                lr, table = grid_search(PAPER_GRIDS["gaussian_default"], make_run(loss), score_fn(metric))
                name = {"elbo": "ADVI", "score": "Score", "fisher": "Fisher"}[loss]
                best[name] = lr
                summary["curves"].setdefault(f"D{dim}-grid-{name}", table)
                print(f"D={dim} {name}: selected learning rate {lr} (paper: {PAPER_SELECTED['gaussian'][name]})")
            selected = best

        specs = build_specs(dim, args.bam_batches, args.gradient_batch, args.max_grad_evals,
                            args.bam_iter_cap, selected)
        for metric in ("forward_kl", "reverse_kl"):
            curves = []
            for spec in specs:
                t0 = time.time()
                res = run_repeats(
                    lambda seed, d=dim, sc=args.target_scale: random_gaussian_target(d, seed=seed, scale=sc),
                    spec, metric, n_runs=args.n_runs, n_samples=args.n_eval_samples,
                    stride=max(1, spec.n_iters // 200))
                s = summarise_curves(res)
                # the paper shows the individual runs as transparent curves
                curves.append({"label": spec.label, "x": s["x"], "y": s["y"], "se": s["se"],
                               "individual": s["runs"] if args.n_runs <= 10 else None})
                print(f"D={dim:4d} {spec.label:18s} {metric:11s} "
                      f"final={s['y'][-1]:.4g}  ({time.time() - t0:.1f}s)")
                key = f"D{dim}-{spec.name}-B{spec.batch_size}-{metric}"
                summary["curves"][key] = {"x": s["x"].tolist(), "y": s["y"].tolist(), "se": s["se"].tolist()}
            path = os.path.join(args.outdir, f"gaussian_D{dim}_{metric}.png")
            plot_curves(curves, "number of gradient evaluations", metric.replace("_", " ").upper(),
                        f"Gaussian target, D={dim}", path)
            print(f"  saved {path}")

    save_json(os.path.join(args.outdir, "gaussian_targets_summary.json"), summary)


if __name__ == "__main__":
    main()
