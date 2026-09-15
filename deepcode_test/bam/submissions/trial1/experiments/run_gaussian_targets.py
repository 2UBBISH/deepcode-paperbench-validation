"""Section 5.1 -- Gaussian targets experiments.

Runs Batch-and-Match (BaM) and the baselines ADVI, Score, Fisher, and GSM on
random multivariate Gaussian targets. The script reproduces the main
quantitative comparisons of the paper:

  * dimensions D in {4, 16, 64, 128, 256}
  * initialization mu0 ~ Uniform[0, 0.1], Sigma0 = I
  * BaM learning rate lambda_t = B * D
  * metrics: forward KL(q_t || p), reverse KL(p || q_t),
    number of gradient evaluations to reach a forward-KL threshold,
    and wallclock time

Results are written as CSV so that ``experiments/plot_results.py`` can
reproduce the corresponding figures.

Example
-------
    python experiments/run_gaussian_targets.py \
        --dims 4,16 --algos bam,advi,gsm --T 200 --B 10 \
        --runs 3 --seed 0 --out results/gaussian_targets.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

# Make project root importable when the script is run directly.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.bam.algorithm import (  # noqa: E402
    bam_step,
    constant_learning_rate,
    decaying_learning_rate,
)
from src.bam.baselines import (  # noqa: E402
    adam_init,
    advi_step,
    fisher_step,
    gsm_step,
    score_step,
)
from src.bam.divergences import kl_gaussian, reverse_kl_gaussian  # noqa: E402
from src.bam.targets import (  # noqa: E402
    GaussianTarget,
    make_initial_params,
    random_gaussian_target,
)


@dataclass
class RunResult:
    """A single-algorithm single-run result."""

    dimension: int
    algorithm: str
    run: int
    T: int
    B: int
    final_forward_kl: float
    final_reverse_kl: float
    grad_evals_to_threshold: int
    wallclock_seconds: float
    final_mean_error: float
    final_cov_error: float


def _symmetrize(a: jnp.ndarray) -> jnp.ndarray:
    """Symmetrize a square matrix."""
    return 0.5 * (a + a.T)


def _normalized_errors(
    mu: jnp.ndarray, Sigma: jnp.ndarray, target: GaussianTarget
) -> Tuple[float, float]:
    """Return normalized mean and covariance errors w.r.t. a Gaussian target.

    The normalized errors are defined as
        || Sigma_*^{-1/2} (mu - mu_*) ||_2
    and
        || Sigma_*^{-1/2} Sigma Sigma_*^{-1/2} - I ||_F.
    """
    # Eigen-decomposition square root of the target covariance.
    evals, evecs = jnp.linalg.eigh(_symmetrize(target.Sigma))
    evals = jnp.clip(evals, a_min=1e-12)
    inv_sqrt = (evecs * (1.0 / jnp.sqrt(evals))[None, :]) @ evecs.T
    inv_sqrt = _symmetrize(inv_sqrt)

    mean_err = float(jnp.linalg.norm(inv_sqrt @ (mu - target.mu)))
    cov_err = float(
        jnp.linalg.norm(inv_sqrt @ Sigma @ inv_sqrt - jnp.eye(mu.shape[0]))
    )
    return mean_err, cov_err


def _record_metrics(
    mu: jnp.ndarray,
    Sigma: jnp.ndarray,
    target: GaussianTarget,
    fkl_hist: List[float],
    rkl_hist: List[float],
) -> None:
    """Append forward/reverse KL estimates to history lists."""
    fkl = float(kl_gaussian(mu, Sigma, target.mu, target.Sigma))
    rkl = float(reverse_kl_gaussian(mu, Sigma, target.mu, target.Sigma))
    fkl_hist.append(fkl)
    rkl_hist.append(rkl)


def _grad_evals_to_threshold(
    fkl_hist: Sequence[float], threshold: float, B: int
) -> int:
    """Gradient evaluations used until the forward-KL threshold is reached.

    Each iteration consumes ``B`` target-score evaluations. If the threshold is
    never reached we report ``T * B`` (the total gradient budget).
    """
    for t, fkl in enumerate(fkl_hist):
        if fkl <= threshold:
            return (t + 1) * B
    return len(fkl_hist) * B


def _run_one_algorithm(
    *,
    algorithm: str,
    target: GaussianTarget,
    mu0: jnp.ndarray,
    Sigma0: jnp.ndarray,
    key: jax.Array,
    T: int,
    B: int,
    threshold: float,
    advi_lr: float,
    score_lr: float,
    fisher_lr: float,
    bam_schedule: Optional[Callable[[int], float]] = None,
) -> Dict[str, Any]:
    """Run a single algorithm on a Gaussian target and return raw results."""
    D = target.mu.shape[0]
    target_score = target.score_fn()

    mu = mu0
    Sigma = Sigma0

    fkl_hist: List[float] = []
    rkl_hist: List[float] = []

    t_start = time.perf_counter()

    if algorithm == "bam":
        schedule = bam_schedule if bam_schedule is not None else constant_learning_rate(B, D)
        for t in range(T):
            key, subkey = jax.random.split(key)
            result = bam_step(
                subkey,
                mu,
                Sigma,
                target_score=target_score,
                B=B,
                lam=float(schedule(t)),
            )
            mu, Sigma = result.mu, result.Sigma
            _record_metrics(mu, Sigma, target, fkl_hist, rkl_hist)

    elif algorithm in ("advi", "score", "fisher"):
        # Gradient baselines parameterize the covariance through a Cholesky
        # factor L, with Sigma = L L^T.
        L = jnp.linalg.cholesky(
            _symmetrize(Sigma0) + 1e-8 * jnp.eye(D)
        )
        params = (mu, L)
        adam_state = adam_init(params)
        lr = {"advi": advi_lr, "score": score_lr, "fisher": fisher_lr}[algorithm]

        for t in range(T):
            key, subkey = jax.random.split(key)
            if algorithm == "advi":
                result = advi_step(subkey, mu, L, target_score, B, adam_state, lr)
            elif algorithm == "score":
                result = score_step(subkey, mu, L, target_score, B, adam_state, lr)
            else:
                result = fisher_step(subkey, mu, L, target_score, B, adam_state, lr)
            mu, L = result.mu, result.L
            adam_state = result.adam_state
            Sigma = _symmetrize(L @ L.T)
            _record_metrics(mu, Sigma, target, fkl_hist, rkl_hist)

    elif algorithm == "gsm":
        for t in range(T):
            key, subkey = jax.random.split(key)
            result = gsm_step(subkey, mu, Sigma, target_score, B)
            mu, Sigma = result.mu, result.Sigma
            _record_metrics(mu, Sigma, target, fkl_hist, rkl_hist)

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    wallclock = time.perf_counter() - t_start
    mean_err, cov_err = _normalized_errors(mu, Sigma, target)

    return {
        "final_mu": mu,
        "final_Sigma": Sigma,
        "fkl_hist": fkl_hist,
        "rkl_hist": rkl_hist,
        "grad_evals_to_threshold": _grad_evals_to_threshold(fkl_hist, threshold, B),
        "wallclock_seconds": wallclock,
        "final_mean_error": mean_err,
        "final_cov_error": cov_err,
    }


def run_gaussian_targets_experiment(
    *,
    dims: Sequence[int],
    algorithms: Sequence[str],
    T: int,
    B: int,
    runs: int,
    seed: int,
    threshold: float,
    advi_lr: float,
    score_lr: float,
    fisher_lr: float,
) -> pd.DataFrame:
    """Run the Gaussian-target comparison and return a tidy results frame."""
    rows: List[RunResult] = []
    master_key = jax.random.PRNGKey(seed)

    for D in dims:
        for run_idx in range(runs):
            # Split the random key deterministically across dimensions/runs.
            master_key, target_key, init_key, algo_key = jax.random.split(master_key, 4)
            target = random_gaussian_target(target_key, D)
            mu0, Sigma0 = make_initial_params(init_key, D)

            for algo in algorithms:
                key = jax.random.fold_in(algo_key, hash(algo) & 0x7FFFFFFF)
                result = _run_one_algorithm(
                    algorithm=algo,
                    target=target,
                    mu0=mu0,
                    Sigma0=Sigma0,
                    key=key,
                    T=T,
                    B=B,
                    threshold=threshold,
                    advi_lr=advi_lr,
                    score_lr=score_lr,
                    fisher_lr=fisher_lr,
                )
                rows.append(
                    RunResult(
                        dimension=D,
                        algorithm=algo,
                        run=run_idx,
                        T=T,
                        B=B,
                        final_forward_kl=float(result["fkl_hist"][-1]),
                        final_reverse_kl=float(result["rkl_hist"][-1]),
                        grad_evals_to_threshold=int(result["grad_evals_to_threshold"]),
                        wallclock_seconds=float(result["wallclock_seconds"]),
                        final_mean_error=float(result["final_mean_error"]),
                        final_cov_error=float(result["final_cov_error"]),
                    )
                )

    return pd.DataFrame([asdict(r) for r in rows])


def _parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _parse_str_list(value: str) -> List[str]:
    return [x.strip().lower() for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run BaM and baselines on Gaussian targets (Section 5.1)."
    )
    parser.add_argument(
        "--dims",
        type=_parse_int_list,
        default=[4, 16, 64, 128, 256],
        help="Comma-separated dimensions.",
    )
    parser.add_argument(
        "--algos",
        type=_parse_str_list,
        default=["bam", "advi", "score", "fisher", "gsm"],
        help="Comma-separated algorithms.",
    )
    parser.add_argument("--T", type=int, default=200, help="Number of iterations.")
    parser.add_argument("--B", type=int, default=10, help="Batch size.")
    parser.add_argument("--runs", type=int, default=1, help="Number of repeated runs.")
    parser.add_argument("--seed", type=int, default=0, help="Master random seed.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Forward-KL threshold for gradient-evaluation counts.",
    )
    parser.add_argument("--advi-lr", type=float, default=1e-2)
    parser.add_argument("--score-lr", type=float, default=1e-2)
    parser.add_argument("--fisher-lr", type=float, default=1e-2)
    parser.add_argument(
        "--out",
        type=str,
        default="results/gaussian_targets.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    frame = run_gaussian_targets_experiment(
        dims=args.dims,
        algorithms=args.algos,
        T=args.T,
        B=args.B,
        runs=args.runs,
        seed=args.seed,
        threshold=args.threshold,
        advi_lr=args.advi_lr,
        score_lr=args.score_lr,
        fisher_lr=args.fisher_lr,
    )

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    frame.to_csv(args.out, index=False)

    # Compact per-dimension/algo summary.
    summary = (
        frame.groupby(["dimension", "algorithm"])
        .agg(
            final_forward_kl=("final_forward_kl", "mean"),
            final_reverse_kl=("final_reverse_kl", "mean"),
            grad_evals_to_threshold=("grad_evals_to_threshold", "mean"),
            wallclock_seconds=("wallclock_seconds", "mean"),
        )
        .reset_index()
    )
    print("=" * 80)
    print("Gaussian-target results")
    print("=" * 80)
    print(summary.to_string(index=False))
    print(f"\nSaved raw results to {args.out}")


if __name__ == "__main__":
    main()
