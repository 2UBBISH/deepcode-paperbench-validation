"""Section 5.2 -- posterior inference in hierarchical Bayesian models.

Reproduces Figure 5.3 (relative posterior mean error) and Figure E.6 (relative
posterior standard-deviation error) for the three posteriordb targets

    arK                    D = 7   (nearly Gaussian)
    gp_pois_regr           D = 13  (GP Poisson regression)
    eight_schools_centered D = 10  (8-schools hierarchical model)

Protocol (Section 5.2 and Appendix E.5):

* initial variational mean ``mu_0 ~ Uniform[0, 0.1]`` and ``Sigma_0 = I``;
* BaM uses the decaying inverse regularization ``lambda_t = B D / (t + 1)``;
* batch sizes ``B = 8`` (dashed) and ``B = 32`` (solid) for BaM, ADVI and GSM;
* 5 runs, curves show the mean, shaded regions the standard error;
* metrics: relative errors of the posterior mean and SD with respect to HMC
  reference samples, with the variational summaries mapped back to the
  constrained (model) parameter space.

The gradients of the posterior log densities are obtained by automatic
differentiation of the JAX transcription of the Stan models (BridgeStan in the
paper); see ``bam/posterior_models.py``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from bam.grid_search import PAPER_GRIDS, grid_search  # noqa: E402
from bam.bam import lambda_constant, lambda_decay  # noqa: E402
from bam.posterior_models import load_posteriordb_target  # noqa: E402
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
from bam.utils import save_json  # noqa: E402

MODELS = ("arK", "gp_pois_regr", "eight_schools_centered")
DEFAULT_BATCHES = (8, 32)
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def bam_schedule(kind: str, value: float):
    """BaM inverse-regularization schedule.

    ``paper``   -- ``lambda_t = B D / (t + 1)``, the schedule reported in
                   Section 5.2 of the paper;
    ``sqrt``    -- ``lambda_t = B D / sqrt(t + 1)``;
    ``constant``-- ``lambda_t = B D``.

    See the README ("known discrepancy on gp_pois_regr") for why the alternative
    schedules are provided: with the reported ``BD/(t+1)`` schedule our BaM run
    stalls at a large relative mean error on ``gp_pois_regr``, while the slower
    decaying schedules reproduce the paper's reported convergence.
    """
    if kind == "paper":
        return lambda_decay(value, 1.0)
    if kind == "sqrt":
        return lambda_decay(value, 0.5)
    if kind == "constant":
        return lambda_constant(value)
    raise ValueError(kind)


def build_specs(dim: int, batches, max_grad_evals: int, iter_cap: int, advi_lr: float,
                schedule: str = "paper"):
    specs = []
    for B in batches:
        n_iters = int(min(max_grad_evals // B, iter_cap))
        lam = bam_schedule(schedule, float(B * dim))
        specs.append(bam_spec(B, n_iters, lam_value=float(B * dim), decay=(schedule != "constant")))
        specs[-1].lam = lam
    for B in batches:
        n_iters = int(min(max_grad_evals // B, iter_cap))
        specs.append(gradient_spec("ADVI", B, n_iters, advi_lr))
        specs.append(gsm_spec(B, n_iters))
    return specs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--batches", type=int, nargs="+", default=list(DEFAULT_BATCHES))
    ap.add_argument("--max-grad-evals", type=int, default=20000)
    ap.add_argument("--iter-cap", type=int, default=2000)
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--advi-lr", type=float, default=0.01)
    ap.add_argument("--bam-schedule", default="paper", choices=["paper", "sqrt", "constant"],
                    help="BaM inverse-regularization schedule (default: the BD/(t+1) schedule "
                         "reported in the paper; see the README for the gp_pois_regr caveat)")
    ap.add_argument("--gp-variable", default="f_tilde", choices=["f", "f_tilde"],
                    help="parameterization of the GP values for gp_pois_regr: the model parameter "
                         "f_tilde (default; the space in which the model is actually defined) or "
                         "the transformed parameter f listed by posteriordb")
    ap.add_argument("--grid-search", action="store_true",
                    help="re-derive ADVI's learning rate by grid search (as the paper does)")
    ap.add_argument("--parameterization", default="constrained", choices=["unconstrained", "constrained"],
                    help="target space for the VI algorithms.  The default, 'constrained', is the "
                         "space of the model's own parameters (the space in which the paper's "
                         "initialization mu_0 ~ U[0, 0.1], Sigma_0 = I is meaningful and in which the "
                         "HMC reference summaries live); the density is smoothly extended outside "
                         "the support of the model.  'unconstrained' uses Stan's unconstrained "
                         "parameterization (log-Jacobians included).")
    ap.add_argument("--outdir", default=RESULTS)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.models = args.models[:1]
        args.batches = args.batches[:1]
        args.max_grad_evals, args.iter_cap, args.n_runs = 2000, 100, 1

    os.makedirs(args.outdir, exist_ok=True)
    summary = {"config": {k: v for k, v in vars(args).items() if k != "outdir"}, "curves": {}}

    for model in args.models:
        def target_factory(seed, m=model):
            return load_posteriordb_target(m, parameterization=args.parameterization,
                                           variable=args.gp_variable if m == "gp_pois_regr" else None)

        dim = target_factory(0).dim
        advi_lr = args.advi_lr
        if args.grid_search:
            import jax

            target = target_factory(0)

            def run(lr):
                spec = gradient_spec("ADVI", args.batches[-1], max(50, args.max_grad_evals // 10), lr)
                return run_method(target, spec, jax.random.PRNGKey(0),
                                  random_init(jax.random.PRNGKey(1), dim), np.eye(dim))

            advi_lr, table = grid_search(PAPER_GRIDS["gaussian_default"], run,
                                         lambda out: float(evaluate_metric(target, out, "rel_mean",
                                                                           jax.random.PRNGKey(2))[-1]))
            summary["curves"][f"{model}-grid-ADVI"] = table
            print(f"{model}: selected ADVI learning rate {advi_lr}")

        specs = build_specs(dim, args.batches, args.max_grad_evals, args.iter_cap, advi_lr,
                            args.bam_schedule)
        for metric in ("rel_mean", "rel_sd"):
            curves = []
            for spec in specs:
                t0 = time.time()
                try:
                    res = run_repeats(target_factory, spec, metric, n_runs=args.n_runs,
                                      stride=max(1, spec.n_iters // 200))
                except Exception as exc:   # keep going if one method diverges badly
                    print(f"{model:24s} {spec.label:16s} {metric:9s} FAILED: "
                          f"{type(exc).__name__}: {exc}", flush=True)
                    summary["curves"][f"{model}-{spec.name}-B{spec.batch_size}-{metric}"] = {"error": str(exc)}
                    continue
                s = summarise_curves(res)
                label = f"{spec.label}" + (" (solid)" if spec.batch_size == max(args.batches) else " (dashed)")
                curves.append({"label": label, "x": s["x"], "y": s["y"], "se": s["se"],
                               "linestyle": "-" if spec.batch_size == max(args.batches) else "--"})
                print(f"{model:24s} {spec.label:16s} {metric:9s} final={s['y'][-1]:.4g} "
                      f"({time.time() - t0:.1f}s)", flush=True)
                summary["curves"][f"{model}-{spec.name}-B{spec.batch_size}-{metric}"] = {
                    "x": s["x"].tolist(), "y": s["y"].tolist(), "se": s["se"].tolist()}
            path = os.path.join(args.outdir, f"posteriordb_{model}_{metric}.png")
            plot_curves(curves, "number of gradient evaluations",
                        "relative mean error" if metric == "rel_mean" else "relative SD error",
                        f"{model} (D={dim})", path)
            print(f"  saved {path}")

    save_json(os.path.join(args.outdir, "posteriordb_summary.json"), summary)


if __name__ == "__main__":
    main()
