"""Section 5.2 hierarchical posterior experiments.

Reproduces the posteriordb experiments from

    Batch and Match: Score-Based Black-Box Variational Inference

comparing Batch-and-Match (BaM), full-covariance ADVI, and Gaussian Score
Matching (GSM) on three hierarchical Bayesian posteriors:

    * ark                    (D = 7)
    * gp-pois-regr           (D = 13)
    * eight-schools-centered (D = 10)

The script evaluates unnormalized log posteriors and their gradients through
BridgeStan, runs each algorithm with batch sizes B = 8 and B = 32, and reports
the relative error of the variational posterior mean and standard deviation
estimates against HMC reference draws from posteriordb.

BaM uses the decaying learning-rate schedule ``lambda_t = B * D / (t + 1)``
as specified in the paper for non-Gaussian and posterior targets.

Example
-------
    python experiments/run_posteriordb.py --T 200 --runs 5 --out results/posteriordb.csv

Use ``--synthetic`` to run with cheap Gaussian stand-ins when posteriordb /
BridgeStan are unavailable (for smoke-testing the pipeline only).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

# Make the ``src`` directory importable as the ``bam`` package.
_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from bam.algorithm import bam_step, decaying_learning_rate  # noqa: E402
from bam.baselines import adam_init, advi_step, gsm_step  # noqa: E402
from bam.posterior_targets import (  # noqa: E402
    PosteriorTarget,
    load_posterior_target,
    posterior_relative_errors,
)
from bam.targets import GaussianTarget  # noqa: E402

DEFAULT_MODELS: Tuple[str, ...] = (
    "ark",
    "gp-pois-regr",
    "eight-schools-centered",
)

DEFAULT_ALGORITHMS: Tuple[str, ...] = ("bam", "advi", "gsm")

DEFAULT_BATCH_SIZES: Tuple[int, ...] = (8, 32)

# Synthetic stand-in dimensions for smoke-testing without posteriordb.
_SYNTHETIC_DIMS: Dict[str, int] = {
    "ark": 7,
    "gp-pois-regr": 13,
    "eight-schools-centered": 10,
}


def _target_dim(target: PosteriorTarget) -> int:
    """Return the target dimensionality, accommodating naming differences."""
    if hasattr(target, "dim"):
        return int(target.dim)
    if hasattr(target, "d"):
        return int(target.d)
    ref = target.reference_draws()
    if ref is not None:
        return int(np.asarray(ref).shape[-1])
    raise ValueError("Could not determine target dimensionality.")


def _extract_mu_sigma(result: Any) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Extract ``(mu, Sigma)`` from a step/run result object or dict."""
    if isinstance(result, dict):
        mu = result["mu"]
        if "Sigma" in result and result["Sigma"] is not None:
            sigma = result["Sigma"]
        elif "L" in result and result["L"] is not None:
            sigma = result["L"] @ result["L"].T
        else:
            raise KeyError("Result must contain 'Sigma' or 'L'.")
    else:
        mu = result.mu
        if hasattr(result, "Sigma") and result.Sigma is not None:
            sigma = result.Sigma
        elif hasattr(result, "L") and result.L is not None:
            sigma = result.L @ result.L.T
        else:
            raise AttributeError("Result must have 'Sigma' or 'L'.")
    return mu, (sigma + sigma.T) / 2.0


def _relative_errors(
    mu: jnp.ndarray,
    Sigma: jnp.ndarray,
    target: PosteriorTarget,
) -> Tuple[float, float]:
    """Relative posterior mean/SD errors against HMC reference draws."""
    ref = target.reference_draws()
    if ref is None:
        return float("nan"), float("nan")
    sd = jnp.sqrt(jnp.clip(jnp.diag(Sigma), 0.0, None))
    mean_err, sd_err = posterior_relative_errors(mu, sd, ref)
    return float(mean_err), float(sd_err)


def _run_bam(
    mu0: jnp.ndarray,
    Sigma0: jnp.ndarray,
    target: PosteriorTarget,
    B: int,
    T: int,
    key: jnp.ndarray,
) -> Dict[str, Any]:
    """Run BaM with the decaying posterior learning-rate schedule."""
    mu = mu0
    Sigma = Sigma0
    D = int(mu0.shape[0])
    score_fn = target.score_fn()
    mean_errs: List[float] = []
    sd_errs: List[float] = []

    t0 = time.time()
    for t in range(T):
        lam = float(B * D) / float(t + 1)
        key, subkey = jax.random.split(key)
        step = bam_step(
            subkey,
            mu,
            Sigma,
            target_score=score_fn,
            B=B,
            lam=lam,
        )
        mu, Sigma = step.mu, step.Sigma
        me, se = _relative_errors(mu, Sigma, target)
        mean_errs.append(me)
        sd_errs.append(se)
    wall = time.time() - t0

    return {
        "mu": mu,
        "Sigma": Sigma,
        "mean_err_hist": mean_errs,
        "sd_err_hist": sd_errs,
        "wallclock_seconds": wall,
    }


def _run_advi(
    mu0: jnp.ndarray,
    Sigma0: jnp.ndarray,
    target: PosteriorTarget,
    B: int,
    T: int,
    learning_rate: float,
    key: jnp.ndarray,
) -> Dict[str, Any]:
    """Run full-covariance ADVI with ADAM and reparameterized gradients."""
    D = int(mu0.shape[0])
    mu = mu0
    L = jnp.linalg.cholesky(Sigma0 + 1e-8 * jnp.eye(D))
    score_fn = target.score_fn()
    log_prob_fn = target.log_prob_fn()
    adam_state = adam_init((mu, L))

    mean_errs: List[float] = []
    sd_errs: List[float] = []

    t0 = time.time()
    for _ in range(T):
        key, subkey = jax.random.split(key)
        step = advi_step(
            subkey,
            mu,
            L,
            target_score=score_fn,
            batch_size=B,
            adam_state=adam_state,
            learning_rate=learning_rate,
            target_log_prob=log_prob_fn,
        )
        mu, L = step.mu, step.L
        adam_state = step.adam_state
        Sigma = L @ L.T
        me, se = _relative_errors(mu, Sigma, target)
        mean_errs.append(me)
        sd_errs.append(se)
    wall = time.time() - t0

    return {
        "mu": mu,
        "Sigma": Sigma,
        "mean_err_hist": mean_errs,
        "sd_err_hist": sd_errs,
        "wallclock_seconds": wall,
    }


def _run_gsm(
    mu0: jnp.ndarray,
    Sigma0: jnp.ndarray,
    target: PosteriorTarget,
    B: int,
    T: int,
    key: jnp.ndarray,
) -> Dict[str, Any]:
    """Run Gaussian Score Matching (Algorithm 3) with batch-averaged updates."""
    mu = mu0
    Sigma = Sigma0
    score_fn = target.score_fn()

    mean_errs: List[float] = []
    sd_errs: List[float] = []

    t0 = time.time()
    for _ in range(T):
        key, subkey = jax.random.split(key)
        step = gsm_step(subkey, mu, Sigma, target_score=score_fn, batch_size=B)
        mu, Sigma = step.mu, step.Sigma
        me, se = _relative_errors(mu, Sigma, target)
        mean_errs.append(me)
        sd_errs.append(se)
    wall = time.time() - t0

    return {
        "mu": mu,
        "Sigma": Sigma,
        "mean_err_hist": mean_errs,
        "sd_err_hist": sd_errs,
        "wallclock_seconds": wall,
    }


def _make_synthetic_target(name: str, key: jnp.ndarray) -> PosteriorTarget:
    """Construct a cheap Gaussian stand-in posterior for smoke tests."""
    dim = _SYNTHETIC_DIMS.get(name, 5)
    k1, k2, k3 = jax.random.split(key, 3)
    mu = 0.4 * jax.random.normal(k1, (dim,))
    A = 0.7 * jax.random.normal(k2, (dim, dim))
    Sigma = A @ A.T + 0.6 * jnp.eye(dim)
    Sigma = (Sigma + Sigma.T) / 2.0

    gt = GaussianTarget(mu=mu, Sigma=Sigma)
    ref = gt.sample(k3, 4000)

    return PosteriorTarget(
        name=name,
        dim=dim,
        log_density_fn=gt.log_prob_fn(),
        log_density_grad_fn=gt.score_fn(),
        reference_draws=ref,
    )


def _load_target(
    name: str,
    key: jnp.ndarray,
    synthetic: bool,
    pdb_path: Optional[str],
    cache_dir: Optional[str],
) -> PosteriorTarget:
    if synthetic:
        return _make_synthetic_target(name, key)
    return load_posterior_target(
        name,
        pdb_path=pdb_path,
        cache_dir=cache_dir,
        load_reference_draws=True,
    )


def _run_one_algorithm(
    algorithm: str,
    mu0: jnp.ndarray,
    Sigma0: jnp.ndarray,
    target: PosteriorTarget,
    B: int,
    T: int,
    advi_lr: float,
    key: jnp.ndarray,
) -> Dict[str, Any]:
    if algorithm == "bam":
        return _run_bam(mu0, Sigma0, target, B, T, key)
    if algorithm == "advi":
        return _run_advi(mu0, Sigma0, target, B, T, advi_lr, key)
    if algorithm == "gsm":
        return _run_gsm(mu0, Sigma0, target, B, T, key)
    raise ValueError(f"Unknown algorithm: {algorithm}")


def run_posteriordb_experiment(
    *,
    models: Sequence[str] = DEFAULT_MODELS,
    algorithms: Sequence[str] = DEFAULT_ALGORITHMS,
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
    T: int = 200,
    runs: int = 5,
    seed: int = 0,
    advi_lr: float = 0.01,
    synthetic: bool = False,
    pdb_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the full Section 5.2 posteriordb experiment matrix.

    Returns a long-format DataFrame with one row per (model, algorithm, batch
    size, run) containing final and minimum relative posterior mean/SD errors
    and wall-clock time.
    """
    rows: List[Dict[str, Any]] = []
    master_key = jax.random.PRNGKey(seed)
    counter = 0

    for model in models:
        model_key, master_key = jax.random.split(master_key)
        target = _load_target(model, model_key, synthetic, pdb_path, cache_dir)
        dim = _target_dim(target)

        if verbose:
            print(f"[run_posteriordb] model={model} dim={dim}")

        for run in range(runs):
            run_key, master_key = jax.random.split(master_key)
            mu0 = jnp.zeros(dim)
            Sigma0 = jnp.eye(dim)

            for algorithm in algorithms:
                for B in batch_sizes:
                    counter += 1
                    alg_key = jax.random.fold_in(run_key, counter)

                    result = _run_one_algorithm(
                        algorithm,
                        mu0,
                        Sigma0,
                        target,
                        B,
                        T,
                        advi_lr,
                        alg_key,
                    )

                    final_mean_err, final_sd_err = _relative_errors(
                        result["mu"], result["Sigma"], target
                    )
                    mean_err_hist = result["mean_err_hist"]
                    sd_err_hist = result["sd_err_hist"]

                    rows.append(
                        {
                            "model": model,
                            "dimension": dim,
                            "algorithm": algorithm,
                            "B": B,
                            "batch_size": B,
                            "run": run,
                            "T": T,
                            "grad_evals": T * B,
                            "final_mean_rel_err": final_mean_err,
                            "final_sd_rel_err": final_sd_err,
                            "min_mean_rel_err": float(
                                np.nanmin(mean_err_hist)
                            )
                            if len(mean_err_hist)
                            else float("nan"),
                            "min_sd_rel_err": float(np.nanmin(sd_err_hist))
                            if len(sd_err_hist)
                            else float("nan"),
                            "wallclock_seconds": result["wallclock_seconds"],
                        }
                    )

                    if verbose:
                        print(
                            f"  {algorithm:5s} B={B:<3d} run={run} "
                            f"mean_err={final_mean_err:.4f} "
                            f"sd_err={final_sd_err:.4f}"
                        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Section 5.2 posteriordb hierarchical posterior experiments."
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated posteriordb model names.",
    )
    parser.add_argument(
        "--algos",
        type=str,
        default=",".join(DEFAULT_ALGORITHMS),
        help="Comma-separated algorithms: bam,advi,gsm.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default=",".join(str(b) for b in DEFAULT_BATCH_SIZES),
        help="Comma-separated batch sizes.",
    )
    parser.add_argument("--T", type=int, default=200, help="Number of iterations.")
    parser.add_argument("--runs", type=int, default=5, help="Monte-Carlo repetitions.")
    parser.add_argument("--seed", type=int, default=0, help="Master PRNG seed.")
    parser.add_argument("--advi-lr", type=float, default=0.01, help="ADVI learning rate.")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic Gaussian stand-ins instead of posteriordb.",
    )
    parser.add_argument("--pdb-path", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "results" / "posteriordb.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    models = tuple(m.strip() for m in args.models.split(",") if m.strip())
    algorithms = tuple(a.strip() for a in args.algos.split(",") if a.strip())
    batch_sizes = tuple(int(b) for b in args.batch_sizes.split(",") if b.strip())

    df = run_posteriordb_experiment(
        models=models,
        algorithms=algorithms,
        batch_sizes=batch_sizes,
        T=args.T,
        runs=args.runs,
        seed=args.seed,
        advi_lr=args.advi_lr,
        synthetic=args.synthetic,
        pdb_path=args.pdb_path,
        cache_dir=args.cache_dir,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
