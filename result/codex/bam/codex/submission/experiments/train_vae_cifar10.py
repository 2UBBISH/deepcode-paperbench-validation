"""Pre-train the VAE used in Section 5.3 on CIFAR-10 (Appendix E.6).

The decoder ``Omega(., theta)`` is learned by variational expectation
maximization: the ELBO is maximized over the decoder weights and a factorized
Gaussian ``q(z | x)`` parameterized by the convolutional encoder.  Optimization
uses Adam with a linear warmup from 0 to ``1e-4`` over 100 batches followed by a
linear decay to ``1e-5`` over 500 batches, one Monte-Carlo sample per ELBO
estimate, GELU activations, no dropout and no normalization (see the addendum to
the paper).  Training runs for 100 epochs.

    python experiments/train_vae_cifar10.py --epochs 100 --out checkpoints/vae_cifar10.npz

Training on CPU is slow (about a second per batch at ``c_hid=64``); the
checkpoint produced here is the input of ``run_vae_posterior.py``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import jax
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from bam.cifar10 import DEFAULT_ROOT, load_cifar10  # noqa: E402
from bam.utils import save_json  # noqa: E402
from bam.vae import VAEConfig, save_params, train_vae  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=DEFAULT_ROOT)
    ap.add_argument("--out", default="checkpoints/vae_cifar10.npz")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--latent-dim", type=int, default=256)
    ap.add_argument("--c-hid", type=int, default=64)
    ap.add_argument("--sigma2", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subset", type=int, default=0, help="use only the first N training images (debugging)")
    ap.add_argument("--synthetic", action="store_true",
                    help="train on synthetic smooth images instead of downloading CIFAR-10 "
                         "(smoke test of the Section 5.3 pipeline; the decoder produced here is not "
                         "the one the paper uses)")
    args = ap.parse_args()

    if args.synthetic:
        # random low-frequency images in [-1, 1]: the same tensor layout as CIFAR-10,
        # enough to exercise the training loop and the posterior inference code
        rng = np.random.default_rng(args.seed)
        n = args.subset or 512
        low = rng.normal(size=(n, 8, 8, 3))
        x_train = np.repeat(np.repeat(low, 4, axis=1), 4, axis=2)
        x_train = np.clip(x_train / (np.abs(x_train).max() + 1e-8), -1.0, 1.0).astype(np.float64)
    else:
        x_train, _ = load_cifar10(args.data_dir, "train", normalize=True)
    if args.subset:
        x_train = x_train[: args.subset]
    print(f"training on {x_train.shape} images")

    config = VAEConfig(latent_dim=args.latent_dim, c_hid=args.c_hid, sigma2=args.sigma2)
    t0 = time.time()
    result = train_vae(x_train, jax.random.PRNGKey(args.seed), config, n_epochs=args.epochs,
                       batch_size=args.batch_size, verbose=True)
    print(f"training took {time.time() - t0:.1f}s")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_params(args.out, result["params"])
    np.savetxt(os.path.join(os.path.dirname(os.path.abspath(args.out)), "vae_elbo_history.csv"),
               result["history"], delimiter=",")
    save_json(os.path.join(os.path.dirname(os.path.abspath(args.out)), "vae_config.json"),
              {"latent_dim": args.latent_dim, "c_hid": args.c_hid, "sigma2": args.sigma2,
               "batch_size": args.batch_size, "epochs": args.epochs, "seed": args.seed,
               "final_elbo": float(-result["history"][-1])})
    print(f"saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
