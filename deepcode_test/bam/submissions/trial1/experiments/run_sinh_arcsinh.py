"""Section 5.1 non-Gaussian target experiments: sinh-arcsinh normal targets.

This script reproduces the skew and tail-weight experiments described in the
"Batch and Match" paper (Section 5.1, non-Gaussian targets).  It compares
Batch-and-Match (BaM) with ADVI, the score-divergence baseline, the Fisher
baseline, and Gaussian score matching (GSM).

Skew cases use ``tailweight = 1`` and ``skew in {0.2, 1.0, 1.8}``; tail cases
use ``skew = 0`` and ``tailweight in {0.1, 0.9, 1.7}``.  BaM uses the decaying
learning rate ``lambda_t = B * D / (t + 1)``.

Because the sinh-arcsinh target is non-Gaussian, forward and reverse KL
divergences are estimated by Monte Carlo using samples from the variational
Gaussian and from the target, respectively.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import pandas as pd

from src.bam.algorithm import bam_step, decaying_learning_rate
from src.bam.baselines import (
    adam_init,
    advi_step,
    fisher_step,
    gsm_step,
    score_step,
)
from src.bam.targets import SinhArcsinhTarget, make_initial_params, sample_mvn, symmetrize

# ---------------------------------------------------------------------------
# Monte-Carlo KL helpers for a Gaussian variational family and a general target
# ---------------------------------------------------------------------------


def _log_gaussian_pdf(z: jnp.ndarray, mu: jnp.ndarray, Sigma: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the log density of N(mu, Sigma) for a batch of points ``z``."""
    z = jnp.atleast_2d(z)
    d = int(z.shape[-1])
    Sigma_sym = symmetrize(Sigma, jitter=1e-8)
    sign, log_det = jnp.linalg.slogdet(Sigma_sym)
    diff = z - mu
    sol = jnp.linalg.solve(Sigma_sym, diff.T).T
    quad = jnp.sum(diff * sol, axis=-1)
    return -0.5 * (d * jnp.log(2.0 * jnp.pi) + log_det + quad)


def _mc_forward_kl(
    mu: jnp.ndarray,
    Sigma: jnp.ndarray,
    target: SinhArcsinhTarget,
    key: jax.Array,
    num_samples: int = 2000,
) -> float:
    """Monte-Carlo estimate of KL(q || p) with q = N(mu, Sigma)."""
    z = sample_mvn(key, mu, Sigma, num_samples)
    log_q = _log_gaussian_pdf(z, mu, Sigma)
    log_p = target.log_prob(z)
    return float(jnp.mean(log_q - log_p))


def _mc_reverse_kl(
    mu: jnp.ndarray,
    Sigma: jnp.ndarray,
    target: SinhArcsinhTarget,
    key: jax.Array,
    num_samples: int = 2000,
) -> float:
    """Monte-Carlo estimate of KL(p || q) with q = N(mu, Sigma)."""
    z = target.sample(key, num_samples)
    log_q = _log_gaussian_pdf(z, mu, Sigma)
    log_p = target.log_prob(z)
    return float(jnp.mean(log_p - log_q))


def _grad_evals_to_threshold(hist: Sequence[float], threshold: float, batch_size: int) -> int:
    """Return the number of gradient/score evaluations until ``hist`` crosses
    ``threshold`` (inclusive).  If the threshold is never reached, return the
    full evaluation budget ``len(hist) * batch_size``."""
    for i, value in enumerate(hist):
        if float(value) <= threshold:
            return (i + 1) * batch_size
    return len(hist) * batch_size


# ---------------------------------------------------------------------------
# Per-algorithm driver
# ---------------------------------------------------------------------------

_GRADIENT_ALGOS = {"advi", "score", "fisher"}


def _run_one_algorithm(
    *,
    algorithm: str,
    target: SinhArcsinhTarget,
    mu0: jnp.ndarray,
    Sigma0: jnp.ndarray,
    key: jax.Array,
    T: int,
    B: int,
    threshold: float,
    kl_samples: int,
    advi_lr: float,
    score_lr: float,
    fisher_lr: float,
) -> Dict[str, Any]:
    """Run one algorithm for ``T`` iterations and return history + metrics."""
    D = int(mu0.shape[0])
    key, key_init = jax.random.split(key)

    if algorithm in _GRADIENT_ALGOS:
        mu = mu0
        L = jnp.linalg.cholesky(symmetrize(Sigma0, jitter=1e-8))
        adam_state = adam_init((mu, L))
        Sigma = L @ L.T
    else:
        mu = mu0
        Sigma = Sigma0
        L = None  # type: ignore[assignment]

    score_fn: Callable[[jnp.ndarray], jnp.ndarray] = target.score_fn()

    fkl_hist: List[float] = []
    rkl_hist: List[float] = []
    iter_seconds: List[float] = []

    lr_fn = decaying_learning_rate(B, D)
    start_wall = time.perf_counter()

    for t in range(T):
        key, key_step = jax.random.split(key)

        if algorithm == "bam":
            lam = float(lr_fn(t))
            res = bam_step(key_step, mu, Sigma, target_score=score_fn, B=B, lam=lam)
            mu = res.mu
            Sigma = res.Sigma
        elif algorithm == "advi":
            res = advi_step(
                key_step,
                mu,
                L,
                score_fn,
                batch_size=B,
                adam_state=adam_state,
                learning_rate=advi_lr,
            )
            mu = res.mu
            L = res.L
            Sigma = res.Sigma
            adam_state = res.adam_state
        elif algorithm == "score":
            res = score_step(
                key_step,
                mu,
                L,
                score_fn,
                batch_size=B,
                adam_state=adam_state,
                learning_rate=score_lr,
            )
            mu = res.mu
            L = res.L
            Sigma = res.Sigma
            adam_state = res.adam_state
        elif algorithm == "fisher":
            res = fisher_step(
                key_step,
                mu,
                L,
                score_fn,
                batch_size=B,
                adam_state=adam_state,
                learning_rate=fisher_lr,
            )
            mu = res.mu
            L = res.L
            Sigma = res.Sigma
            adam_state = res.adam_state
        elif algorithm == "gsm":
            res = gsm_step(key_step, mu, Sigma, score_fn, batch_size=B)
            mu = res.mu
            Sigma = res.Sigma
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Evaluate divergences on a fixed budget of extra samples.
        key, key_kl = jax.random.split(key)
        fkl = _mc_forward_kl(mu, Sigma, target, key_kl, kl_samples)
        rkl = _mc_reverse_kl(mu, Sigma, target, key_kl, kl_samples)
        fkl_hist.append(fkl)
        rkl_hist.append(rkl)
        iter_seconds.append(time.perf_counter() - start_wall)

    return {
        "mu": mu,
        "Sigma": Sigma,
        "forward_kl_history": fkl_hist,
        "reverse_kl_history": rkl_hist,
        "final_forward_kl": fkl_hist[-1],
        "final_reverse_kl": rkl_hist[-1],
        "grad_evals_to_threshold": _grad_evals_to_threshold(fkl_hist, threshold, B),
        "wallclock_seconds": iter_seconds[-1],
    }


# ---------------------------------------------------------------------------
# Experiment matrix
# ---------------------------------------------------------------------------


def run_sinh_arcsinh_experiment(
    *,
    dims: Sequence[int] = (4,),
    skew_cases: Sequence[float] = (0.2, 1.0, 1.8),
    tail_cases: Sequence[float] = (0.1, 0.9, 1.7),
    algorithms: Sequence[str] = ("bam", "advi", "score", "fisher", "gsm"),
    T: int = 200,
    B: int = 32,
    runs: int = 5,
    seed: int = 0,
    threshold: float = 0.1,
    kl_samples: int = 2000,
    advi_lr: float = 0.01,
    score_lr: float = 0.01,
    fisher_lr: float = 0.01,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the Section 5.1 sinh-arcsinh experiment matrix and return a tidy
    results :class:`pandas.DataFrame`."""
    master_key = jax.random.PRNGKey(seed)
    rows: List[Dict[str, Any]] = []

    for D in dims:
        # Base Gaussian is a standard normal; skew/tail transforms are applied
        # on top of it, isolating the effect of the sinh-arcsinh parameters.
        base_mu = jnp.zeros(D)
        base_Sigma = jnp.eye(D)

        cases: List[Tuple[str, str, float]] = []
        for s in skew_cases:
            cases.append(("skew", "skew", float(s)))
        for tau in tail_cases:
            cases.append(("tail", "tailweight", float(tau)))

        for case_type, param_name, param_value in cases:
            if case_type == "skew":
                target = SinhArcsinhTarget(base_mu, base_Sigma, skew=param_value, tailweight=1.0)
            else:
                target = SinhArcsinhTarget(base_mu, base_Sigma, skew=0.0, tailweight=param_value)

            for run in range(runs):
                master_key, key = jax.random.split(master_key)
                mu0, Sigma0 = make_initial_params(key, D, mu_low=0.0, mu_high=0.1, sigma0=1.0)

                for algorithm in algorithms:
                    key, algo_key = jax.random.split(key)
                    result = _run_one_algorithm(
                        algorithm=algorithm,
                        target=target,
                        mu0=mu0,
                        Sigma0=Sigma0,
                        key=algo_key,
                        T=T,
                        B=B,
                        threshold=threshold,
                        kl_samples=kl_samples,
                        advi_lr=advi_lr,
                        score_lr=score_lr,
                        fisher_lr=fisher_lr,
                    )
                    rows.append(
                        {
                            "dimension": D,
                            "case_type": case_type,
                            "param_name": param_name,
                            "param_value": param_value,
                            "algorithm": algorithm,
                            "run": run,
                            "T": T,
                            "B": B,
                            "final_forward_kl": result["final_forward_kl"],
                            "final_reverse_kl": result["final_reverse_kl"],
                            "grad_evals_to_threshold": result["grad_evals_to_threshold"],
                            "wallclock_seconds": result["wallclock_seconds"],
                        }
                    )
                    if verbose:
                        print(
                            f"[D={D}] {case_type}={param_value:.2f} run={run} "
                            f"{algorithm:6s} fKL={result['final_forward_kl']:.4f} "
                            f"rKL={result['final_reverse_kl']:.4f} "
                            f"evals={result['grad_evals_to_threshold']}"
                        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_float_list(value: str) -> List[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def _parse_int_list(value: str) -> List[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def _parse_str_list(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Section 5.1 sinh-arcsinh non-Gaussian target experiments."
    )
    parser.add_argument("--dims", type=str, default="4", help="Comma-separated dimensions.")
    parser.add_argument("--skew-cases", type=str, default="0.2,1.0,1.8")
    parser.add_argument("--tail-cases", type=str, default="0.1,0.9,1.7")
    parser.add_argument(
        "--algos",
        type=str,
        default="bam,advi,score,fisher,gsm",
        help="Comma-separated algorithm names.",
    )
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--B", type=int, default=32)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--kl-samples", type=int, default=2000)
    parser.add_argument("--advi-lr", type=float, default=0.01)
    parser.add_argument("--score-lr", type=float, default=0.01)
    parser.add_argument("--fisher-lr", type=float, default=0.01)
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join("results", "sinh_arcsinh.csv"),
        help="Output CSV path.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    df = run_sinh_arcsinh_experiment(
        dims=_parse_int_list(args.dims),
        skew_cases=_parse_float_list(args.skew_cases),
        tail_cases=_parse_float_list(args.tail_cases),
        algorithms=_parse_str_list(args.algos),
        T=args.T,
        B=args.B,
        runs=args.runs,
        seed=args.seed,
        threshold=args.threshold,
        kl_samples=args.kl_samples,
        advi_lr=args.advi_lr,
        score_lr=args.score_lr,
        fisher_lr=args.fisher_lr,
        verbose=not args.quiet,
    )

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
