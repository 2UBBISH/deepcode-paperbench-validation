#!/usr/bin/env python
"""Train a Simformer on one of the tasks of the paper.

Example
-------
```bash
python scripts/train_simformer.py --task two_moons --mask-mode directed \
    --n-simulations 10000 --sde vesde --out results/two_moons_directed
```

The number of simulations, batch size, learning rate, optimizer and the early
stopping criterion follow Appendix A2.1.  The trained model is stored as a torch
checkpoint together with the (generated) training data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from common import RESULTS, build_model, ensure_dir, generate_simulations, \
    save_checkpoint, save_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--mask-mode", default="dense",
                        choices=["dense", "undirected", "directed"])
    parser.add_argument("--sde", default="vesde", choices=["vesde", "vpsde"])
    parser.add_argument("--n-simulations", type=int, default=10000)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--self-recurrence", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--condition-mask-options", default="all",
                        choices=["all", "posterior"])
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    out = Path(args.out) if args.out else (
        RESULTS / f"{args.task}_{args.mask_mode}_{args.sde}_{args.n_simulations}")
    ensure_dir(out)
    print(f"[train] task={args.task} mask={args.mask_mode} sde={args.sde} "
          f"n_simulations={args.n_simulations} -> {out}")

    task, model = build_model(
        args.task, mask_mode=args.mask_mode, sde=args.sde,
        n_layers=args.n_layers, seed=args.seed, num_steps=args.num_steps,
        self_recurrence=args.self_recurrence,
        condition_mask_options=args.condition_mask_options)

    theta, x, index, metadata = generate_simulations(
        task, args.n_simulations, seed=args.seed)
    np.savez_compressed(out / "simulations.npz", theta=theta, x=x,
                        index=index)
    print(f"[train] generated {args.n_simulations} simulations of "
          f"{task.n_variables} variables each")

    model.fit(theta, x, index, metadata, batch_size=args.batch_size,
              lr=args.lr, max_epochs=args.max_epochs, patience=args.patience,
              max_steps=args.max_steps, verbose=True)

    save_checkpoint(model, out / "model.pt")
    save_config(out / "config.json", model.config,
                {"task": args.task, "n_simulations": args.n_simulations})
    np.savez_compressed(out / "history.npz",
                        **{k: np.array([h[k] for h in model.history])
                           for k in ("epoch", "step", "train_loss", "val_loss")})
    print(f"[train] done. best validation loss: "
          f"{min(h['val_loss'] for h in model.history):.4f}")


if __name__ == "__main__":
    main()
