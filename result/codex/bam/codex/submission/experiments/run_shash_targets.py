"""Section 5.1 (second part) -- non-Gaussian targets with varying skew and tails.

Reproduces Figure 5.2 (forward KL) and Figure E.4 (reverse KL) for
sinh-arcsinh normal targets with ``D = 10``:

* varying skew ``s = 0.2, 1.0, 1.8`` with normal tails (``tau = 1``), and
* varying tail weight ``tau = 0.1, 0.9, 1.7`` with no skew (``s = 0``).

Protocol (Section 5.1 and Appendix E.4):

* BaM uses the decaying inverse regularization ``lambda_t = B D / (t + 1)``,
  since "some decay is necessary for BaM to converge when the target
  distribution is non-Gaussian";
* ADVI / Score / Fisher / GSM use batch size ``B = 5`` with the learning rates
  selected by the grid search reported in Appendix E.4;
* 10 runs, curves show the mean, shaded regions the standard error.

Assumption (documented in the README): the base Gaussian of the sinh-arcsinh
transformation is the standard normal, ``y ~ N(0, I)``; the paper does not state
the base distribution explicitly.
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
    evaluate_metric,
    gradient_spec,
    gsm_spec,
    plot_curves,
    random_init,
    run_method,
    run_repeats,
    summarise_curves,
)
from bam.targets import SinhArcsinhTarget  # noqa: E402
from bam.utils import save_json  # noqa: E402

DEFAULT_SKEWS = (0.2, 1.0, 1.8)
DEFAULT_TAILS = (0.1, 0.9, 1.7)
DEFAULT_BAM_BATCHES = (2, 5, 10, 20, 40)
DIM = 10
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def make_target(kind: str, param: float, dim: int = DIM, seed: int = 0) -> SinhArcsinhTarget:
    """Sinh-arcsinh target with a standard-normal base and uniform skew/tail."""
    skew, tail = (param, 1.0) if kind == "skew" else (0.0, param)
    return SinhArcsinhTarget(np.zeros(dim), np.eye(dim), skew=skew, tail=tail,
                             name=f"shash-{kind}{param}")


def build_specs(kind: str, param: float, dim: int, bam_batches, gradient_batch: int,
                max_grad_evals: int, bam_iter_cap: int, selected=None):
    sel_skew = PAPER_SELECTED["shash_skew"]
    sel_tail = PAPER_SELECTED["shash_tail"]
    sel = (sel_skew if kind == "skew" else sel_tail) if selected is None else selected
    specs = []
    for B in bam_batches:
        n_iters = int(min(max_grad_evals // B, bam_iter_cap))
        specs.append(bam_spec(B, n_iters, lam_value=float(B * dim), decay=True))
    grad_iters = int(max_grad_evals // gradient_batch)
    score_lr = sel["Score"]
    specs.append(gradient_spec("ADVI", gradient_batch, grad_iters, sel["ADVI"]))
    specs.append(gradient_spec("Fisher", gradient_batch, grad_iters, sel["Fisher"]))
    specs.append(gradient_spec("Score", gradient_batch, grad_iters,
                               score_lr[param] if isinstance(score_lr, dict) else score_lr))
    specs.append(gsm_spec(gradient_batch, grad_iters))
    return specs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skews", type=float, nargs="+", default=list(DEFAULT_SKEWS))
    ap.add_argument("--tails", type=float, nargs="+", default=list(DEFAULT_TAILS))
    ap.add_argument("--bam-batches", type=int, nargs="+", default=list(DEFAULT_BAM_BATCHES))
    ap.add_argument("--gradient-batch", type=int, default=5)
    ap.add_argument("--max-grad-evals", type=int, default=20000)
    ap.add_argument("--bam-iter-cap", type=int, default=2000)
    ap.add_argument("--n-runs", type=int, default=10)
    ap.add_argument("--n-eval-samples", type=int, default=20000)
    ap.add_argument("--eval-points", type=int, default=200,
                    help="number of iterates at which the (Monte-Carlo) metrics are evaluated")
    ap.add_argument("--grid-search", action="store_true",
                    help="re-derive the learning rates of ADVI / Score / Fisher by grid search")
    ap.add_argument("--outdir", default=RESULTS)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.skews, args.tails = args.skews[:1], args.tails[:1]
        args.n_runs, args.max_grad_evals, args.bam_iter_cap = 1, 1000, 100
        args.bam_batches = args.bam_batches[:2]
        args.n_eval_samples = min(args.n_eval_samples, 2000)
        args.eval_points = min(args.eval_points, 25)

    os.makedirs(args.outdir, exist_ok=True)
    summary = {"config": {k: v for k, v in vars(args).items() if k != "outdir"}, "curves": {}}

    panels = ([("skew", s) for s in args.skews], [("tail", t) for t in args.tails])
    for metric in ("forward_kl", "reverse_kl"):
        for panel, entries in enumerate(panels):
            curves = []
            for kind, param in entries:
                selected = None
                if args.grid_search:
                    target = make_target(kind, param)

                    def make_run(loss):
                        def run(lr):
                            spec = gradient_spec({"elbo": "ADVI", "score": "Score", "fisher": "Fisher"}[loss],
                                                 args.gradient_batch, max(50, args.max_grad_evals // 10), lr)
                            return run_method(target, spec, jax.random.PRNGKey(0),
                                              random_init(jax.random.PRNGKey(1), DIM), np.eye(DIM))

                        return run

                    best = {}
                    for loss, m in (("elbo", "reverse_kl"), ("score", "reverse_kl"), ("fisher", "reverse_kl")):
                        lr, table = grid_search(PAPER_GRIDS["shash"], make_run(loss),
                                                lambda out: float(evaluate_metric(
                                                    make_target(kind, param), out, m,
                                                    jax.random.PRNGKey(3), n_samples=5000,
                                                    stride=max(1, out["mu"].shape[0] // 20))[-1]))
                        best[{"elbo": "ADVI", "score": "Score", "fisher": "Fisher"}[loss]] = lr
                        summary["curves"][f"grid-{kind}{param}-{loss}"] = table
                    selected = best

                specs = build_specs(kind, param, DIM, args.bam_batches, args.gradient_batch,
                                    args.max_grad_evals, args.bam_iter_cap, selected)
                for spec in specs:
                    t0 = time.time()
                    res = run_repeats(lambda seed, k=kind, p=param: make_target(k, p, seed=seed), spec,
                                      metric, n_runs=args.n_runs, n_samples=args.n_eval_samples,
                                      stride=max(1, spec.n_iters // args.eval_points))
                    s = summarise_curves(res)
                    label = f"{kind}={param} " + spec.label
                    curves.append({"label": label, "x": s["x"], "y": s["y"], "se": s["se"]})
                    print(f"{kind}={param} {spec.label:18s} {metric:11s} final={s['y'][-1]:.4g} "
                          f"({time.time() - t0:.1f}s)", flush=True)
                    summary["curves"][f"{kind}{param}-{spec.name}-B{spec.batch_size}-{metric}"] = {
                        "x": s["x"].tolist(), "y": s["y"].tolist(), "se": s["se"].tolist()}
            path = os.path.join(args.outdir,
                                f"shash_{'skew' if panel == 0 else 'tail'}_{metric}.png")
            plot_curves(curves, "number of gradient evaluations",
                        metric.replace("_", " ").upper(),
                        f"sinh-arcsinh targets, D={DIM}, "
                        f"{'varying skew' if panel == 0 else 'varying tails'}", path)
            print(f"  saved {path}")

    save_json(os.path.join(args.outdir, "shash_targets_summary.json"), summary)


if __name__ == "__main__":
    main()
