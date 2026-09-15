"""Section 5.3 CIFAR-10 deep generative model experiments.

This script trains (or loads) a convolutional VAE with a 256-dimensional latent
space and a Gaussian decoder with fixed variance ``sigma^2 = 0.1``.  For a set
of held-out test images it then runs full-covariance posterior inference with
Batch-and-Match (BaM), full-covariance ADVI, and Gaussian Score Matching (GSM),
plus a factorized amortized variational inference (AVI) baseline.

The script reproduces the paper's empirical protocol:

* a short ``T_pilot`` pilot run is used to select learning rates for ADVI and
  BaM;
* final runs use ``T_final`` iterations;
* a fixed-gradient-budget comparison (3000 gradient evaluations) contrasts
  ADVI (B=10, T=300) against BaM (B=300, T=10).

Metrics are reconstruction MSE on the test image and wall-clock time, written
as a tidy CSV consumed by ``experiments/plot_results.py``.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

# Make the local ``bam`` package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bam.algorithm import bam_step  # noqa: E402
from bam.baselines import adam_init, advi_step, gsm_step  # noqa: E402
from bam.deep_generative import (  # noqa: E402
    DeepGenerativeTarget,
    avi_posterior,
    decoder_forward,
    download_cifar10,
    flatten_images,
    initial_advi_factor,
    initial_posterior_params,
    load_cifar10,
    reconstruction_mse,
    train_vae,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_CSV = PROJECT_ROOT / "results" / "cifar.csv"
DEFAULT_RECON_NPZ = PROJECT_ROOT / "results" / "cifar_reconstructions.npz"
DEFAULT_DATA_DIR = str(Path.home() / ".bam_data" / "cifar10")
DEFAULT_CHECKPOINT = str(Path.home() / ".bam_data" / "cifar10_vae_params.pkl")

OBS_DIM = 32 * 32 * 3
LATENT_DIM = 256
SIGMA2 = 0.1


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def _reconstruction_mse(
    decoder_params: Dict[str, Any], z_mean: jnp.ndarray, x_obs: jnp.ndarray
) -> float:
    """Robust mean squared reconstruction error.

    ``z_mean`` may be either ``(D,)`` or ``(N, D)`` and ``x_obs`` may be
    either ``(3072,)``, ``(N, 3072)``, or an image-shaped array; both are
    flattened internally so the comparison is shape-agnostic.
    """
    if z_mean.ndim == 1:
        z_mean = z_mean[None, :]
    recon = decoder_forward(decoder_params, z_mean)  # (N, 32, 32, 3)
    recon_flat = jnp.reshape(recon, (recon.shape[0], -1))

    if x_obs.ndim == 1:
        x_obs = jnp.reshape(x_obs, (1, -1))
    else:
        x_obs = jnp.reshape(x_obs, (x_obs.shape[0], -1))

    return float(jnp.mean(jnp.sum((recon_flat - x_obs) ** 2, axis=-1)))


def _decode_image(decoder_params: Dict[str, Any], z: jnp.ndarray) -> np.ndarray:
    """Return a ``(32, 32, 3)`` NumPy image decoded from a latent mean."""
    recon = decoder_forward(decoder_params, z[None, :])
    return np.asarray(recon[0])


# ---------------------------------------------------------------------------
# Algorithm runners
# ---------------------------------------------------------------------------
def _run_bam(
    key: jax.Array,
    score_fn: Callable[[jnp.ndarray], jnp.ndarray],
    mu0: jnp.ndarray,
    Sigma0: jnp.ndarray,
    B: int,
    T: int,
    lam_fn: Callable[[int], float],
    mse_fn: Callable[[jnp.ndarray], float],
) -> Dict[str, Any]:
    """Run BaM with a (possibly scheduled) learning rate."""
    mu, Sigma = mu0, Sigma0
    mse_hist: List[float] = []
    t0 = time.perf_counter()
    for t in range(T):
        key, subkey = jax.random.split(key)
        result = bam_step(
            subkey,
            mu,
            Sigma,
            target_score=score_fn,
            B=B,
            lam=lam_fn(t),
        )
        mu, Sigma = result.mu, result.Sigma
        mse_hist.append(mse_fn(mu))
    wallclock = time.perf_counter() - t0
    return {
        "mu": mu,
        "Sigma": Sigma,
        "mse_hist": mse_hist,
        "wallclock_seconds": wallclock,
        "grad_evals": B * T,
    }


def _run_advi(
    key: jax.Array,
    score_fn: Callable[[jnp.ndarray], jnp.ndarray],
    mu0: jnp.ndarray,
    L0: jnp.ndarray,
    B: int,
    T: int,
    learning_rate: float,
    mse_fn: Callable[[jnp.ndarray], float],
) -> Dict[str, Any]:
    """Run full-covariance ADVI with ADAM and reparameterized gradients."""
    mu, L = mu0, L0
    adam_state = adam_init((mu, L))
    mse_hist: List[float] = []
    t0 = time.perf_counter()
    for _ in range(T):
        key, subkey = jax.random.split(key)
        result = advi_step(
            subkey,
            mu,
            L,
            score_fn,
            batch_size=B,
            adam_state=adam_state,
            learning_rate=learning_rate,
        )
        mu, L, adam_state = result.mu, result.L, result.adam_state
        mse_hist.append(mse_fn(mu))
    wallclock = time.perf_counter() - t0
    return {
        "mu": mu,
        "L": L,
        "mse_hist": mse_hist,
        "wallclock_seconds": wallclock,
        "grad_evals": B * T,
    }


def _run_gsm(
    key: jax.Array,
    score_fn: Callable[[jnp.ndarray], jnp.ndarray],
    mu0: jnp.ndarray,
    Sigma0: jnp.ndarray,
    B: int,
    T: int,
    mse_fn: Callable[[jnp.ndarray], float],
) -> Dict[str, Any]:
    """Run Gaussian Score Matching."""
    mu, Sigma = mu0, Sigma0
    mse_hist: List[float] = []
    t0 = time.perf_counter()
    for _ in range(T):
        key, subkey = jax.random.split(key)
        result = gsm_step(subkey, mu, Sigma, score_fn, batch_size=B)
        mu, Sigma = result.mu, result.Sigma
        mse_hist.append(mse_fn(mu))
    wallclock = time.perf_counter() - t0
    return {
        "mu": mu,
        "Sigma": Sigma,
        "mse_hist": mse_hist,
        "wallclock_seconds": wallclock,
        "grad_evals": B * T,
    }


# ---------------------------------------------------------------------------
# Pilot learning-rate selection
# ---------------------------------------------------------------------------
def _pilot_bam_learning_rate(
    key: jax.Array,
    score_fn: Callable[[jnp.ndarray], jnp.ndarray],
    mu0: jnp.ndarray,
    Sigma0: jnp.ndarray,
    B: int,
    D: int,
    T_pilot: int,
    mse_fn: Callable[[jnp.ndarray], float],
) -> float:
    """Select a constant BaM learning rate on a short pilot run."""
    base = float(B * D)
    candidates = [0.01 * base, 0.1 * base, base]
    best_lam = candidates[0]
    best_mse = float("inf")
    for lam in candidates:
        key, subkey = jax.random.split(key)
        out = _run_bam(
            subkey, score_fn, mu0, Sigma0, B, T_pilot, lambda t, lam=lam: lam, mse_fn
        )
        final_mse = out["mse_hist"][-1]
        if final_mse < best_mse:
            best_mse = final_mse
            best_lam = lam
    return best_lam


def _pilot_advi_learning_rate(
    key: jax.Array,
    score_fn: Callable[[jnp.ndarray], jnp.ndarray],
    mu0: jnp.ndarray,
    L0: jnp.ndarray,
    B: int,
    T_pilot: int,
    mse_fn: Callable[[jnp.ndarray], float],
) -> float:
    """Select an ADVI learning rate on a short pilot run."""
    candidates = [1e-4, 1e-3, 1e-2]
    best_lr = candidates[0]
    best_mse = float("inf")
    for lr in candidates:
        key, subkey = jax.random.split(key)
        out = _run_advi(subkey, score_fn, mu0, L0, B, T_pilot, lr, mse_fn)
        final_mse = out["mse_hist"][-1]
        if final_mse < best_mse:
            best_mse = final_mse
            best_lr = lr
    return best_lr


# ---------------------------------------------------------------------------
# VAE parameter management
# ---------------------------------------------------------------------------
def _get_vae_params(
    key: jax.Array,
    train_x: jnp.ndarray,
    checkpoint_path: Optional[str],
    epochs: int,
    latent_dim: int,
    sigma2: float,
    verbose: bool,
) -> Dict[str, Any]:
    """Load a cached VAE or train one from scratch."""
    if checkpoint_path and checkpoint_path.lower() not in ("none", ""):
        ckpt = Path(checkpoint_path).expanduser()
        if ckpt.exists():
            if verbose:
                print(f"[cifar] Loading VAE parameters from {ckpt}")
            with open(ckpt, "rb") as f:
                return pickle.load(f)

    if verbose:
        print(
            f"[cifar] Training VAE (latent_dim={latent_dim}, sigma2={sigma2}, "
            f"epochs={epochs})"
        )
    params = train_vae(
        key,
        train_x,
        epochs=epochs,
        batch_size=128,
        learning_rate=1e-3,
        latent_dim=latent_dim,
        sigma2=sigma2,
        verbose=verbose,
    )

    if checkpoint_path and checkpoint_path.lower() not in ("none", ""):
        ckpt = Path(checkpoint_path).expanduser()
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        with open(ckpt, "wb") as f:
            pickle.dump(params, f)
        if verbose:
            print(f"[cifar] Saved VAE parameters to {ckpt}")
    return params


# ---------------------------------------------------------------------------
# Main experiment driver
# ---------------------------------------------------------------------------
def run_cifar_experiment(
    *,
    test_size: int = 5,
    batch_sizes: Sequence[int] = (10, 300),
    advi_batch_size: int = 10,
    gsm_batch_size: int = 10,
    T_pilot: int = 100,
    T_final: int = 1000,
    seed: int = 0,
    epochs: int = 10,
    sigma2: float = SIGMA2,
    latent_dim: int = LATENT_DIM,
    data_dir: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    budget_evals: int = 3000,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the Section 5.3 CIFAR-10 deep generative model experiments.

    Returns a tidy ``pd.DataFrame`` with one row per
    (test image, algorithm, configuration).
    """
    key = jax.random.PRNGKey(seed)
    data_dir = data_dir or DEFAULT_DATA_DIR
    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINT

    # --- data -------------------------------------------------------------
    try:
        download_cifar10(data_dir)
    except Exception as exc:  # pragma: no cover - network optional
        if verbose:
            print(f"[cifar] CIFAR download skipped or failed: {exc}")

    train_x, test_x = load_cifar10(data_dir=data_dir)
    train_x = jnp.asarray(train_x)
    test_x = jnp.asarray(test_x)
    test_x_flat = flatten_images(test_x)
    n_test = int(test_x.shape[0])
    test_indices = list(range(min(int(test_size), n_test)))

    # --- VAE --------------------------------------------------------------
    key, vae_key = jax.random.split(key)
    params = _get_vae_params(
        vae_key, train_x, checkpoint_path, epochs, latent_dim, sigma2, verbose
    )
    encoder_params = params["encoder"]
    decoder_params = params["decoder"]

    # --- pilot learning rates on the first test image ---------------------
    pilot_idx = test_indices[0]
    pilot_img = test_x[pilot_idx]
    pilot_flat = test_x_flat[pilot_idx]
    pilot_target = DeepGenerativeTarget(
        x_obs=jnp.asarray(pilot_flat), decoder_params=decoder_params, sigma2=sigma2
    )
    pilot_score_fn = pilot_target.score_fn()
    pilot_mu, pilot_Sigma = initial_posterior_params(encoder_params, pilot_img)
    pilot_mu_advi, pilot_L0 = initial_advi_factor(encoder_params, pilot_img)
    pilot_mse_fn = lambda mu: _reconstruction_mse(  # noqa: E731
        decoder_params, mu, pilot_flat
    )

    if verbose:
        print("[cifar] Running pilot for learning-rate selection")
    best_lams: Dict[int, float] = {}
    for B in batch_sizes:
        key, subkey = jax.random.split(key)
        best_lams[B] = _pilot_bam_learning_rate(
            subkey, pilot_score_fn, pilot_mu, pilot_Sigma, B, latent_dim, T_pilot, pilot_mse_fn
        )
        if verbose:
            print(f"[cifar]   BaM B={B}: selected lambda={best_lams[B]:.4g}")
    key, subkey = jax.random.split(key)
    best_lr = _pilot_advi_learning_rate(
        subkey, pilot_score_fn, pilot_mu_advi, pilot_L0, advi_batch_size, T_pilot, pilot_mse_fn
    )
    if verbose:
        print(f"[cifar]   ADVI B={advi_batch_size}: selected lr={best_lr:.4g}")

    # --- final runs -------------------------------------------------------
    rows: List[Dict[str, Any]] = []
    recon_originals: List[np.ndarray] = []
    recon_bam: List[np.ndarray] = []
    recon_advi: List[np.ndarray] = []
    recon_avi: List[np.ndarray] = []

    max_bam_B = max(batch_sizes)

    for idx in test_indices:
        x_img = test_x[idx]
        x_flat = test_x_flat[idx]
        if verbose:
            print(f"[cifar] Inference on test image {idx}")

        target = DeepGenerativeTarget(
            x_obs=jnp.asarray(x_flat), decoder_params=decoder_params, sigma2=sigma2
        )
        score_fn = target.score_fn()
        mu0, Sigma0 = initial_posterior_params(encoder_params, x_img)
        mu0_advi, L0 = initial_advi_factor(encoder_params, x_img)
        mse_fn = lambda mu: _reconstruction_mse(  # noqa: E731
            decoder_params, mu, x_flat
        )

        # AVI (factorized encoder) baseline.
        avi_mu, _ = avi_posterior(encoder_params, x_img)
        avi_mse = _reconstruction_mse(decoder_params, avi_mu, x_flat)
        rows.append(
            {
                "test_index": idx,
                "algorithm": "avi",
                "batch_size": 0,
                "T": 0,
                "scenario": "avi",
                "learning_rate": np.nan,
                "reconstruction_mse": avi_mse,
                "wallclock_seconds": 0.0,
                "grad_evals": 0,
            }
        )

        bam_mu_for_plot: Optional[jnp.ndarray] = None
        advi_mu_for_plot: Optional[jnp.ndarray] = None

        # BaM for each requested batch size.
        for B in batch_sizes:
            key, subkey = jax.random.split(key)
            out = _run_bam(
                subkey,
                score_fn,
                mu0,
                Sigma0,
                B,
                T_final,
                lambda t, lam=best_lams[B]: lam,
                mse_fn,
            )
            final_mse = out["mse_hist"][-1]
            rows.append(
                {
                    "test_index": idx,
                    "algorithm": "bam",
                    "batch_size": B,
                    "T": T_final,
                    "scenario": "full",
                    "learning_rate": best_lams[B],
                    "reconstruction_mse": final_mse,
                    "wallclock_seconds": out["wallclock_seconds"],
                    "grad_evals": out["grad_evals"],
                }
            )
            if B == max_bam_B:
                bam_mu_for_plot = out["mu"]

        # Full-covariance ADVI.
        key, subkey = jax.random.split(key)
        out = _run_advi(
            subkey, score_fn, mu0_advi, L0, advi_batch_size, T_final, best_lr, mse_fn
        )
        rows.append(
            {
                "test_index": idx,
                "algorithm": "advi",
                "batch_size": advi_batch_size,
                "T": T_final,
                "scenario": "full",
                "learning_rate": best_lr,
                "reconstruction_mse": out["mse_hist"][-1],
                "wallclock_seconds": out["wallclock_seconds"],
                "grad_evals": out["grad_evals"],
            }
        )
        advi_mu_for_plot = out["mu"]

        # Gaussian Score Matching.
        key, subkey = jax.random.split(key)
        out = _run_gsm(subkey, score_fn, mu0, Sigma0, gsm_batch_size, T_final, mse_fn)
        rows.append(
            {
                "test_index": idx,
                "algorithm": "gsm",
                "batch_size": gsm_batch_size,
                "T": T_final,
                "scenario": "full",
                "learning_rate": np.nan,
                "reconstruction_mse": out["mse_hist"][-1],
                "wallclock_seconds": out["wallclock_seconds"],
                "grad_evals": out["grad_evals"],
            }
        )

        # Fixed-gradient-budget comparison: ADVI B=10,T=300 vs BaM B=300,T=10.
        budget_B_bam = max_bam_B
        advi_budget_T = max(1, budget_evals // advi_batch_size)
        key, subkey = jax.random.split(key)
        out = _run_advi(
            subkey,
            score_fn,
            mu0_advi,
            L0,
            advi_batch_size,
            advi_budget_T,
            best_lr,
            mse_fn,
        )
        rows.append(
            {
                "test_index": idx,
                "algorithm": "advi",
                "batch_size": advi_batch_size,
                "T": advi_budget_T,
                "scenario": "budget",
                "learning_rate": best_lr,
                "reconstruction_mse": out["mse_hist"][-1],
                "wallclock_seconds": out["wallclock_seconds"],
                "grad_evals": out["grad_evals"],
            }
        )

        bam_budget_T = max(1, budget_evals // budget_B_bam)
        key, subkey = jax.random.split(key)
        out = _run_bam(
            subkey,
            score_fn,
            mu0,
            Sigma0,
            budget_B_bam,
            bam_budget_T,
            lambda t, lam=best_lams[budget_B_bam]: lam,
            mse_fn,
        )
        rows.append(
            {
                "test_index": idx,
                "algorithm": "bam",
                "batch_size": budget_B_bam,
                "T": bam_budget_T,
                "scenario": "budget",
                "learning_rate": best_lams[budget_B_bam],
                "reconstruction_mse": out["mse_hist"][-1],
                "wallclock_seconds": out["wallclock_seconds"],
                "grad_evals": out["grad_evals"],
            }
        )

        # Store images for the reconstruction figure bundle.
        recon_originals.append(np.asarray(jnp.reshape(x_flat, (32, 32, 3))))
        recon_avi.append(_decode_image(decoder_params, avi_mu))
        if bam_mu_for_plot is not None:
            recon_bam.append(_decode_image(decoder_params, bam_mu_for_plot))
        if advi_mu_for_plot is not None:
            recon_advi.append(_decode_image(decoder_params, advi_mu_for_plot))

    # Save optional reconstruction bundle for plot_results.py.
    DEFAULT_RECON_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        DEFAULT_RECON_NPZ,
        test_indices=np.asarray(test_indices),
        originals=np.asarray(recon_originals),
        reconstructions_bam=np.asarray(recon_bam) if recon_bam else np.zeros((0, 32, 32, 3)),
        reconstructions_advi=np.asarray(recon_advi) if recon_advi else np.zeros((0, 32, 32, 3)),
        reconstructions_avi=np.asarray(recon_avi) if recon_avi else np.zeros((0, 32, 32, 3)),
    )
    if verbose:
        print(f"[cifar] Saved reconstruction bundle to {DEFAULT_RECON_NPZ}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_int_list(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Section 5.3 CIFAR-10 deep generative model experiments"
    )
    parser.add_argument("--test-size", type=int, default=5)
    parser.add_argument(
        "--batch-sizes", type=_parse_int_list, default=[10, 300], help="comma-separated BaM batch sizes"
    )
    parser.add_argument("--advi-batch-size", type=int, default=10)
    parser.add_argument("--gsm-batch-size", type=int, default=10)
    parser.add_argument("--T-pilot", type=int, default=100)
    parser.add_argument("--T-final", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--sigma2", type=float, default=SIGMA2)
    parser.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="VAE params pickle path; 'none' disables caching"
    )
    parser.add_argument("--budget-evals", type=int, default=3000)
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT_CSV))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    df = run_cifar_experiment(
        test_size=args.test_size,
        batch_sizes=tuple(args.batch_sizes),
        advi_batch_size=args.advi_batch_size,
        gsm_batch_size=args.gsm_batch_size,
        T_pilot=args.T_pilot,
        T_final=args.T_final,
        seed=args.seed,
        epochs=args.epochs,
        sigma2=args.sigma2,
        latent_dim=args.latent_dim,
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        budget_evals=args.budget_evals,
        verbose=not args.quiet,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    if not args.quiet:
        print(f"[cifar] Wrote results to {out_path}")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
