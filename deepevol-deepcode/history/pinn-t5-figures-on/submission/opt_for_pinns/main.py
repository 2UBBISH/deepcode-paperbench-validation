"""Entry point for the ``opt_for_pinns`` project.

Reproduces "Challenges in Training PINNs: A Loss Landscape Perspective".

This script provides a unified CLI to:

* train a single PINN configuration (PDE + optimizer) and report loss / L2RE,
* dispatch to any of the experiment scripts under ``experiments/`` that
  reproduce the paper's figures and tables.

Examples
--------
Train convection with Adam+L-BFGS::

    python main.py train --pde convection --optimizer adam_lbfgs --width 200

Run the optimizer comparison sweep (Table 1 / Figure 8)::

    python main.py experiment optimizer_comparison

Run every experiment::

    python main.py experiment all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Make the package importable when executed as a script from the repo root.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import torch  # noqa: E402

from opt_for_pinns.src.data import build_data  # noqa: E402
from opt_for_pinns.src.loss import evaluate  # noqa: E402
from opt_for_pinns.src.model import build_model  # noqa: E402
from opt_for_pinns.src.pdes import get_pde  # noqa: E402
from opt_for_pinns.src.train import train  # noqa: E402
from opt_for_pinns.src.utils import (  # noqa: E402
    ensure_dir,
    get_logger,
    save_json,
    set_seed,
)

# ---------------------------------------------------------------------------
# Defaults (mirroring the paper's Section 2.2 / 6.1 protocol)
# ---------------------------------------------------------------------------
DEFAULT_PDES: List[str] = ["convection", "reaction", "wave"]
DEFAULT_WIDTH = 200
DEFAULT_DEPTH = 3
DEFAULT_SEED = 345
DEFAULT_TOTAL_ITERS = 41000
DEFAULT_SWITCH_ITER = 11000
DEFAULT_ADAM_LR = 1e-3

# Per-PDE best configurations from Section 6.1 (used by the fine-tuning,
# spectral-density and wall-clock experiments).
BEST_CONFIG: Dict[str, Dict[str, object]] = {
    "convection": {"width": 200, "adam_lr": 1e-4, "switch_iter": 11000, "seed": 345},
    "reaction": {"width": 200, "adam_lr": 1e-3, "switch_iter": 11000, "seed": 456},
    "wave": {"width": 200, "adam_lr": 1e-3, "switch_iter": 11000, "seed": 567},
}

EXPERIMENTS = [
    "optimizer_comparison",
    "loss_vs_l2re",
    "spectral_density",
    "nncg_finetune",
    "wallclock",
]


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------
def run_single(
    pde_name: str = "convection",
    optimizer: str = "adam_lbfgs",
    width: int = DEFAULT_WIDTH,
    depth: int = DEFAULT_DEPTH,
    seed: int = DEFAULT_SEED,
    adam_lr: float = DEFAULT_ADAM_LR,
    switch_iter: int = DEFAULT_SWITCH_ITER,
    total_iters: int = DEFAULT_TOTAL_ITERS,
    n_finetune_steps: int = 2000,
    mu: float = 1e-2,
    device: str = "cpu",
    outdir: Optional[str] = None,
    log_every: int = 0,
    logger=None,
) -> Dict:
    """Train a single PINN configuration and return a result dictionary.

    Parameters
    ----------
    pde_name : {"convection", "reaction", "wave"}
    optimizer : {"adam", "lbfgs", "adam_lbfgs", "nncg", "gd"}
    width, depth : MLP architecture (paper: 3 hidden layers, width in {50,100,200,400}).
    seed : RNG seed (paper uses 5 seeds: 123, 234, 345, 456, 567).
    adam_lr : Adam learning rate (grid {1e-5, 1e-4, 1e-3, 1e-2, 1e-1}).
    switch_iter : Adam -> L-BFGS switch point (grid {1000, 11000, 31000}).
    total_iters : total Adam+L-BFGS iterations (paper: 41000).
    n_finetune_steps : NNCG / GD fine-tuning steps (paper: 2000).
    mu : NNCG damping parameter (tuned in {1e-5..1e-1}).
    """
    logger = logger or get_logger()
    set_seed(seed)

    pde = get_pde(pde_name)
    model = build_model(width=width, depth=depth, seed=seed, device=device)
    data = build_data(pde, seed=seed)
    data = data.to(device=device)

    kwargs: Dict[str, object] = {}
    if optimizer == "adam":
        kwargs = {"lr": adam_lr, "n_iters": total_iters}
    elif optimizer == "lbfgs":
        kwargs = {"lr": 1.0, "n_iters": total_iters, "history_size": 100}
    elif optimizer == "adam_lbfgs":
        kwargs = {
            "adam_lr": adam_lr,
            "switch_iter": switch_iter,
            "total_iters": total_iters,
            "lbfgs_lr": 1.0,
            "history_size": 100,
        }
    elif optimizer == "nncg":
        # NNCG is a fine-tuning method: warm up with Adam+L-BFGS first.
        warm = train(
            pde,
            model,
            data,
            optimizer_name="adam_lbfgs",
            adam_lr=adam_lr,
            switch_iter=switch_iter,
            total_iters=total_iters,
            log_every=log_every,
            verbose=bool(log_every),
        )
        kwargs = {"n_steps": n_finetune_steps, "mu": mu}
    elif optimizer == "gd":
        warm = train(
            pde,
            model,
            data,
            optimizer_name="adam_lbfgs",
            adam_lr=adam_lr,
            switch_iter=switch_iter,
            total_iters=total_iters,
            log_every=log_every,
            verbose=bool(log_every),
        )
        kwargs = {"n_steps": n_finetune_steps, "lr": 1e-3}
    else:
        raise ValueError(f"Unknown optimizer: {optimizer!r}")

    result = train(
        pde,
        model,
        data,
        optimizer_name=optimizer,
        log_every=log_every,
        verbose=bool(log_every),
        **kwargs,
    )

    record = result.as_dict()
    record.update(
        {
            "pde": pde_name,
            "optimizer": optimizer,
            "width": width,
            "depth": depth,
            "seed": seed,
            "adam_lr": adam_lr,
            "switch_iter": switch_iter,
            "total_iters": total_iters,
        }
    )

    if logger:
        logger.info(
            "[%s | %s | w=%d | seed=%d] loss=%.6e l2re=%.6e grad=%.6e",
            pde_name,
            optimizer,
            width,
            seed,
            record.get("loss", float("nan")),
            record.get("l2re", float("nan")),
            record.get("grad_norm", float("nan")),
        )

    if outdir:
        ensure_dir(outdir)
        save_json(record, os.path.join(outdir, f"{pde_name}_{optimizer}_w{width}_s{seed}.json"))

    return record


# ---------------------------------------------------------------------------
# Experiment dispatch
# ---------------------------------------------------------------------------
def run_experiment(name: str, outdir: str, logger=None, **kwargs) -> Dict:
    """Dispatch to one of the experiment scripts under ``experiments/``."""
    logger = logger or get_logger()
    exp_outdir = os.path.join(outdir, name)
    ensure_dir(exp_outdir)

    if name == "optimizer_comparison":
        from opt_for_pinns.experiments import run_optimizer_comparison as mod
    elif name == "loss_vs_l2re":
        from opt_for_pinns.experiments import run_loss_vs_l2re as mod
    elif name == "spectral_density":
        from opt_for_pinns.experiments import run_spectral_density as mod
    elif name == "nncg_finetune":
        from opt_for_pinns.experiments import run_nncg_finetune as mod
    elif name == "wallclock":
        from opt_for_pinns.experiments import run_wallclock as mod
    else:
        raise ValueError(f"Unknown experiment: {name!r}. Choose from {EXPERIMENTS}.")

    logger.info("Running experiment: %s -> %s", name, exp_outdir)
    return mod.run(outdir=exp_outdir, logger=logger, **kwargs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="opt_for_pinns",
        description=(
            "Reproduction of 'Challenges in Training PINNs: A Loss Landscape Perspective'."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- train subcommand -------------------------------------------------
    p_train = sub.add_parser("train", help="Train a single PINN configuration.")
    p_train.add_argument("--pde", default="convection", choices=DEFAULT_PDES)
    p_train.add_argument(
        "--optimizer",
        default="adam_lbfgs",
        choices=["adam", "lbfgs", "adam_lbfgs", "nncg", "gd"],
    )
    p_train.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    p_train.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    p_train.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_train.add_argument("--adam-lr", type=float, default=DEFAULT_ADAM_LR)
    p_train.add_argument("--switch-iter", type=int, default=DEFAULT_SWITCH_ITER)
    p_train.add_argument("--total-iters", type=int, default=DEFAULT_TOTAL_ITERS)
    p_train.add_argument("--n-finetune-steps", type=int, default=2000)
    p_train.add_argument("--mu", type=float, default=1e-2)
    p_train.add_argument("--device", default="cpu")
    p_train.add_argument("--outdir", default="results/single")
    p_train.add_argument("--log-every", type=int, default=0)

    # --- experiment subcommand -------------------------------------------
    p_exp = sub.add_parser("experiment", help="Run one of the paper's experiments.")
    p_exp.add_argument("name", choices=EXPERIMENTS + ["all"])
    p_exp.add_argument("--outdir", default="results")
    p_exp.add_argument("--total-iters", type=int, default=DEFAULT_TOTAL_ITERS)
    p_exp.add_argument("--log-every", type=int, default=0)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = get_logger()

    if args.command == "train":
        run_single(
            pde_name=args.pde,
            optimizer=args.optimizer,
            width=args.width,
            depth=args.depth,
            seed=args.seed,
            adam_lr=args.adam_lr,
            switch_iter=args.switch_iter,
            total_iters=args.total_iters,
            n_finetune_steps=args.n_finetune_steps,
            mu=args.mu,
            device=args.device,
            outdir=args.outdir,
            log_every=args.log_every,
            logger=logger,
        )
        return 0

    if args.command == "experiment":
        names = EXPERIMENTS if args.name == "all" else [args.name]
        for name in names:
            run_experiment(
                name,
                outdir=args.outdir,
                logger=logger,
                total_iters=args.total_iters,
                log_every=args.log_every,
            )
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
