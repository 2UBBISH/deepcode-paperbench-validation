"""Section 5.3 -- posterior inference in a deep generative model (CIFAR-10 VAE).

Given a pre-trained decoder (see ``train_vae_cifar10.py``) and a test image
``x'``, the target is ``p(z' | x') ∝ N(z'; 0, I) N(x'; Omega(z'), sigma^2 I)``
with ``z' in R^256`` and ``sigma^2 = 0.1``.  We compare

* BaM (batch sizes ``B = 10, 100, 300``),
* ADVI with the same batch sizes,
* GSM and the amortized encoder (AVI) as reference points,

and measure the reconstruction error obtained by feeding the posterior mean
``E[z' | x']`` through the decoder (Figure 5.4 plots the MSE against the number of
gradient evaluations, Figure E.7 against wallclock time).

Protocol (Section 5.3 and Appendix E.6):

* all VI algorithms start from the standard Gaussian;
* ADVI and BaM run a pilot experiment of ``T = 100`` iterations to select the
  learning rate for each batch size, after which the algorithms are run for
  ``T = 1000`` iterations;
* the learning rates selected in the paper are recorded in
  ``bam.grid_search.PAPER_SELECTED['vae_bam']`` (BaM) and ``['vae_advi']`` (ADVI)
  and used as defaults when the pilot search is skipped.

    python experiments/run_vae_posterior.py --checkpoint checkpoints/vae_cifar10.npz \\
        --batch-sizes 10 100 300 --n-iters 1000
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import jax
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from bam.bam import BaM, BaMConfig, lambda_constant  # noqa: E402
from bam.baselines import GSM, GradientVI, GradientVIConfig  # noqa: E402
from bam.cifar10 import DEFAULT_ROOT, load_cifar10  # noqa: E402
from bam.deep_generative import DeepGenerativePosterior, amortized_posterior  # noqa: E402
from bam.grid_search import PAPER_GRIDS, PAPER_SELECTED, grid_search  # noqa: E402
from bam.utils import plot_curves, save_json  # noqa: E402
from bam.vae import VAEConfig, load_params  # noqa: E402

DEFAULT_BATCHES = (10, 100, 300)
RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def mse_curve(target: DeepGenerativePosterior, out) -> np.ndarray:
    return np.asarray([target.reconstruction_mse(out["mu"][i]) for i in range(out["mu"].shape[0])])


def run_bam_mse(target, B: int, n_iters: int, lam: float, key):
    bam = BaM(target, BaMConfig(batch_size=B, lam=lambda_constant(lam)))
    out = bam.run(key, n_iters, np.zeros(target.dim), np.eye(target.dim))
    return out, mse_curve(target, out)


def run_advi_mse(target, B: int, n_iters: int, lr: float, key):
    vi = GradientVI(target, GradientVIConfig(batch_size=B, learning_rate=lr, loss="elbo", n_iters=n_iters))
    out = vi.run(key, mu0=np.zeros(target.dim), Sigma0=np.eye(target.dim))
    return out, mse_curve(target, out)


def run_gsm_mse(target, B: int, n_iters: int, key):
    out = GSM(target, batch_size=B).run(key, n_iters, mu0=np.zeros(target.dim), Sigma0=np.eye(target.dim))
    return out, mse_curve(target, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="VAE checkpoint written by train_vae_cifar10.py")
    ap.add_argument("--data-dir", default=DEFAULT_ROOT)
    ap.add_argument("--test-index", type=int, default=0, help="index of x' in the CIFAR-10 test set")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=list(DEFAULT_BATCHES))
    ap.add_argument("--n-iters", type=int, default=1000)
    ap.add_argument("--pilot-iters", type=int, default=100)
    ap.add_argument("--budget", type=int, default=3000,
                    help="gradient-evaluation budget used for the 'best MSE under a fixed budget' comparison")
    ap.add_argument("--latent-dim", type=int, default=256)
    ap.add_argument("--c-hid", type=int, default=64)
    ap.add_argument("--sigma2", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--random-image", action="store_true",
                    help="use a random image instead of a CIFAR-10 test image (for smoke tests; "
                         "the decoder checkpoint must match the configuration)")
    ap.add_argument("--skip-pilot", action="store_true",
                    help="use the learning rates reported in the paper instead of re-running the pilot search")
    ap.add_argument("--outdir", default=RESULTS)
    ap.add_argument("--quick", action="store_true", help="tiny run for smoke testing (random decoder)")
    args = ap.parse_args()

    config = VAEConfig(latent_dim=args.latent_dim, c_hid=args.c_hid, sigma2=args.sigma2)
    key = jax.random.PRNGKey(args.seed)
    params = load_params(args.checkpoint, config, key)
    if args.random_image:
        x_obs = np.asarray(jax.random.uniform(key, config.image_shape) * 2.0 - 1.0)
    else:
        x_test, _ = load_cifar10(args.data_dir, "test", normalize=True)
        x_obs = x_test[args.test_index]
    target = DeepGenerativePosterior(params["dec"], config, x_obs, name=f"cifar10_test{args.test_index}")

    if args.quick:
        args.n_iters, args.pilot_iters = 10, 3

    os.makedirs(args.outdir, exist_ok=True)
    summary = {"config": {k: v for k, v in vars(args).items() if k != "outdir"}, "curves": {}}

    # ---- amortized variational inference (encoder) -------------------------
    avi = amortized_posterior(params["enc"], params["dec"], config, x_obs)
    avi_mse = target.reconstruction_mse(avi["mu"])
    print(f"AVI reconstruction MSE: {avi_mse:.5f}")
    summary["avi_mse"] = avi_mse

    curves, wallclock = [], []
    for B in args.batch_sizes:
        # ---- BaM ----------------------------------------------------------
        if args.skip_pilot:
            lam_bam = PAPER_SELECTED["vae_bam"].get(B, 50.0)
            lr_advi = PAPER_SELECTED["vae_advi"]["ADVI"]
            pilot_bam = pilot_advi = None
        else:
            candidates = (0.01, 0.1, 0.2, 10.0) if B <= 10 else (
                (2.0, 20.0, 50.0, 100.0, 200.0) if B <= 100 else (1000.0, 5000.0, 7500.0, 10000.0))
            lam_bam, pilot_bam = grid_search(
                candidates, lambda lam: run_bam_mse(target, B, args.pilot_iters, lam, key)[1][-1],
                lambda v: float(v))
            lr_advi, pilot_advi = grid_search(
                PAPER_GRIDS["vae_advi"], lambda lr: run_advi_mse(target, B, args.pilot_iters, lr, key)[1][-1],
                lambda v: float(v))
            summary["curves"][f"pilot-BaM-B{B}"] = pilot_bam
            summary["curves"][f"pilot-ADVI-B{B}"] = pilot_advi
        print(f"B={B}: BaM lambda={lam_bam}, ADVI lr={lr_advi}")

        t0 = time.time()
        out_bam, mse_bam = run_bam_mse(target, B, args.n_iters, lam_bam, key)
        t_bam = time.time() - t0
        curves.append({"label": f"BaM (B={B})", "x": out_bam["grad_evals"], "y": mse_bam})
        wallclock.append({"label": f"BaM (B={B})", "x": np.linspace(0, t_bam, len(mse_bam)), "y": mse_bam})

        t0 = time.time()
        out_advi, mse_advi = run_advi_mse(target, B, args.n_iters, lr_advi, key)
        t_advi = time.time() - t0
        curves.append({"label": f"ADVI (B={B})", "x": out_advi["grad_evals"], "y": mse_advi})
        wallclock.append({"label": f"ADVI (B={B})", "x": np.linspace(0, t_advi, len(mse_advi)), "y": mse_advi})

        t0 = time.time()
        out_gsm, mse_gsm = run_gsm_mse(target, B, args.n_iters, key)
        t_gsm = time.time() - t0
        curves.append({"label": f"GSM (B={B})", "x": out_gsm["grad_evals"], "y": mse_gsm})
        wallclock.append({"label": f"GSM (B={B})", "x": np.linspace(0, t_gsm, len(mse_gsm)), "y": mse_gsm})

        print(f"B={B}: final MSE BaM {mse_bam[-1]:.5f} (best {mse_bam.min():.5f} at {int(np.argmin(mse_bam)) * B} "
              f"grad evals), ADVI {mse_advi[-1]:.5f} (best {mse_advi.min():.5f}), GSM {mse_gsm[-1]:.5f}")
        # the paper's "computational budget" comparison (Section 5.3): the best
        # reconstruction error achieved within a fixed budget of gradient
        # evaluations (3,000 in the paper) for each method and batch size
        for name, mse, outs in (("BaM", mse_bam, out_bam), ("ADVI", mse_advi, out_advi), ("GSM", mse_gsm, out_gsm)):
            ge = np.asarray(outs["grad_evals"])
            mask = ge <= args.budget
            if mask.any():
                idx = int(np.argmin(np.where(mask, mse, np.inf)))
                summary.setdefault("budget_best", {})[f"{name}-B{B}"] = {
                    "budget": args.budget, "best_mse": float(mse[idx]), "grad_evals": int(ge[idx]),
                    "iterations": int(idx)}
                print(f"    budget {args.budget}: {name} (B={B}) best MSE {mse[idx]:.5f} "
                      f"after {int(ge[idx])} gradient evaluations")
        summary["curves"][f"B{B}"] = {
            "BaM": {"x": out_bam["grad_evals"].tolist(), "mse": mse_bam.tolist()},
            "ADVI": {"x": out_advi["grad_evals"].tolist(), "mse": mse_advi.tolist()},
            "GSM": {"x": out_gsm["grad_evals"].tolist(), "mse": mse_gsm.tolist()},
        }

    plot_curves(curves, "number of gradient evaluations", "reconstruction MSE",
                "CIFAR-10 VAE posterior", os.path.join(args.outdir, "vae_posterior_mse_vs_grad.png"),
                logy=False, logx=True)
    plot_curves(wallclock, "wallclock time (s)", "reconstruction MSE",
                "CIFAR-10 VAE posterior", os.path.join(args.outdir, "vae_posterior_mse_vs_time.png"),
                logy=False, logx=True)
    save_json(os.path.join(args.outdir, "vae_posterior_summary.json"), summary)


if __name__ == "__main__":
    main()
