"""Section 3.2 -- numerical check of Theorem 3.1 (exponential convergence).

The theorem concerns the limit of infinite batch size with a Gaussian target
``p = N(mu_*, Sigma_*)``.  This script

1. runs the exact infinite-batch recursion of Propositions D.3/D.4 (implemented
   in ``bam.theory``) and checks the bounds

       ||eps_t||   <= (1 - delta)^t ||eps_0||
       ||Delta_t|| <= (1 - delta)^t ||Delta_0|| + t (1 - delta)^{t-1} ||eps_0||^2,

   for several regularization levels ``lambda`` (including very small and very
   large ones), and the per-iteration inequalities eqs. (19-20);
2. verifies that the infinite-batch recursion agrees with an actual run of
   Algorithm 1 with a large batch size;
3. verifies Corollary D.5 (one-step convergence for ``B -> inf`` then
   ``lambda_0 -> inf``);
4. saves a figure of the errors and their bounds.

    python experiments/verify_theorem31.py --dim 16 --n-iters 200
"""

from __future__ import annotations

import argparse
import os
import sys

import jax
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from bam.bam import BaM, BaMConfig, lambda_constant  # noqa: E402
from bam.targets import GaussianTarget, random_gaussian_target  # noqa: E402
from bam.theory import run_infinite_batch, theoretical_bounds  # noqa: E402
from bam.utils import plot_curves, save_json  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--n-iters", type=int, default=200)
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.1, 1.0, 10.0, 100.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-check", type=int, default=20,
                    help="number of iterations for the finite-batch comparison")
    ap.add_argument("--outdir", default=RESULTS)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    summary = {"config": vars(args), "checks": {}}
    rng = np.random.default_rng(args.seed)
    target = random_gaussian_target(args.dim, seed=args.seed)
    mu_star, Sigma_star = target.mu, target.Sigma
    # a deliberately poor initialization: mean far from the target mean,
    # covariance with a small minimum eigenvalue (alpha << 1)
    mu0 = rng.normal(size=args.dim) * 2.0
    A = rng.normal(size=(args.dim, args.dim)) / np.sqrt(args.dim)
    Sigma0 = A @ A.T + 1e-2 * np.eye(args.dim)

    curves = []
    for lam in args.lambdas:
        res = run_infinite_batch(mu0, Sigma0, mu_star, Sigma_star, lam, args.n_iters)
        bnd = theoretical_bounds(mu0, Sigma0, mu_star, Sigma_star, lam, args.n_iters)
        t = np.arange(args.n_iters + 1)
        eps_ok = bool(np.all(res["eps_norm"] <= bnd["eps_bound"] + 1e-12))
        Delta_ok = bool(np.all(res["Delta_norm"] <= bnd["Delta_bound"] + 1e-10))
        # per-iteration inequalities, eqs. (19-20)
        step_eps = res["eps_norm"][1:] <= (1 - bnd["delta"]) * res["eps_norm"][:-1] + 1e-12
        steps = res["Delta_norm"][1:] <= (1 - bnd["delta"]) * res["Delta_norm"][:-1] + res["eps_norm"][:-1] ** 2 + 1e-10
        summary["checks"][f"lambda={lam}"] = {
            "alpha": bnd["alpha"], "beta": bnd["beta"], "delta": bnd["delta"],
            "eps_bound_holds": eps_ok, "Delta_bound_holds": Delta_ok,
            "per_iteration_eps_holds": bool(np.all(step_eps)),
            "per_iteration_Delta_holds": bool(np.all(steps)),
            "final_eps_norm": float(res["eps_norm"][-1]),
            "final_Delta_norm": float(res["Delta_norm"][-1]),
        }
        print(f"lambda={lam:8g} alpha={bnd['alpha']:.3g} beta={bnd['beta']:.3g} delta={bnd['delta']:.3g} "
              f"| eps bound {eps_ok} | Delta bound {Delta_ok} | "
              f"per-iteration {bool(np.all(step_eps))}/{bool(np.all(steps))} | "
              f"final ||eps||={res['eps_norm'][-1]:.2e} ||Delta||={res['Delta_norm'][-1]:.2e}")
        if lam == args.lambdas[0]:
            curves.append({"label": r"$\|\varepsilon_t\|$", "x": t, "y": res["eps_norm"], "band": False})
            curves.append({"label": r"bound on $\|\varepsilon_t\|$", "x": t, "y": bnd["eps_bound"],
                           "band": False})
            curves.append({"label": r"$\|\Delta_t\|$", "x": t, "y": np.maximum(res["Delta_norm"], 1e-16),
                           "band": False})
            curves.append({"label": r"bound on $\|\Delta_t\|$", "x": t,
                           "y": np.maximum(bnd["Delta_bound"], 1e-16), "band": False})
    path = os.path.join(args.outdir, "theorem31_errors.png")
    plot_curves(curves, "iteration $t$", "normalized error", "Theorem 3.1: errors and bounds", path,
                logy=True, logx=False)
    print(f"  saved {path}")

    # ---- 2. finite batch comparison ----------------------------------------
    B = 4096
    lam = 1.0
    bam = BaM(target, BaMConfig(batch_size=B, lam=lambda_constant(lam)))
    out = bam.run(jax.random.PRNGKey(0), args.batch_check, mu0, Sigma0)
    inf_res = run_infinite_batch(mu0, Sigma0, mu_star, Sigma_star, lam, args.batch_check)
    finite_err = np.asarray([np.linalg.norm(np.linalg.solve(np.linalg.cholesky(Sigma_star), out["mu"][i] - mu_star))
                             for i in range(args.batch_check + 1)])
    rel = np.abs(finite_err - inf_res["eps_norm"]) / np.maximum(inf_res["eps_norm"], 1e-12)
    summary["finite_batch"] = {"batch_size": B, "lambda": lam,
                               "infinite_batch_eps": inf_res["eps_norm"].tolist(),
                               "finite_batch_eps": finite_err.tolist(),
                               "max_relative_difference": float(rel.max())}
    print(f"finite batch (B={B}) vs infinite batch: max relative difference in ||eps_t|| = {rel.max():.3f}")

    # ---- 3. one-step convergence (Corollary D.5) ---------------------------
    one_step = {}
    for lam_big in (1e2, 1e4, 1e6):
        res = run_infinite_batch(mu0, Sigma0, mu_star, Sigma_star, float(lam_big), 1)
        one_step[f"lambda={lam_big:g}"] = {"eps_1": float(res["eps_norm"][1]),
                                           "Delta_1": float(res["Delta_norm"][1])}
    summary["one_step_convergence"] = one_step
    print("Corollary D.5 (one step, B -> inf then lambda -> inf):")
    for k, v in one_step.items():
        print(f"  {k}: ||eps_1||={v['eps_1']:.3e}  ||Delta_1||={v['Delta_1']:.3e}")

    save_json(os.path.join(args.outdir, "theorem31_checks.json"), summary)


if __name__ == "__main__":
    main()
