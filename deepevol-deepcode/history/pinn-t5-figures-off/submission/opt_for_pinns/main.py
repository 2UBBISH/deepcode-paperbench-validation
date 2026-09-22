"""Entry point CLI for the opt_for_pinns codebase.

Provides subcommands to:
  - train   : train a PINN with a chosen optimizer (adam / lbfgs / adam_lbfgs / nncg)
  - eval    : evaluate a trained checkpoint (loss, L2RE, grad norm)
  - spectral: estimate the Hessian spectral density (full or per-component)

This mirrors the experiment structure of the paper
"Challenges in Training PINNs: A Loss Landscape Perspective".

Usage examples
--------------
    python main.py train --pde convection --optimizer adam_lbfgs --width 100 --seed 0
    python main.py eval  --pde convection --width 100 --checkpoint results/.../model.pt
    python main.py spectral --pde convection --width 200 --switch-iter 11000
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, Optional

import torch

# Make `src` importable when running `python main.py` from the package root.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from src.data import build_data, make_eval_points  # noqa: E402
from src.loss import make_loss_fn, component_loss_fns  # noqa: E402
from src.metrics import evaluate, l2_relative_error_from_data  # noqa: E402
from src.model import build_model, flatten_parameters, set_flat_parameters  # noqa: E402
from src.pdes import get_pde  # noqa: E402
from src.utils import (  # noqa: E402
    ensure_dir,
    get_device,
    get_logger,
    load_config,
    merge_configs,
    results_dir,
    save_checkpoint,
    save_json,
    set_seed,
)

LOGGER = get_logger("opt_for_pinns.main")

# Default learning rates per PDE (best configs from the paper's tuning grid).
DEFAULT_LRS: Dict[str, float] = {
    "convection": 1e-4,
    "reaction": 1e-3,
    "wave": 1e-3,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_pde_config(pde_name: str, config_path: Optional[str]) -> Dict[str, Any]:
    """Load a PDE config from `configs/<pde>.yaml` (or an explicit path)."""
    if config_path is None:
        config_path = os.path.join(_HERE, "configs", f"{pde_name}.yaml")
    if os.path.exists(config_path):
        try:
            return load_config(config_path)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Failed to load config %s: %s", config_path, exc)
    return {}


def _build_problem(pde_name: str, width: int, seed: int, device: torch.device):
    """Instantiate PDE, model, and data for a given configuration."""
    pde = get_pde(pde_name)
    model = build_model(width=width, depth=3, seed=seed, device=device)
    data = build_data(pde, seed=seed, device=device)
    return pde, model, data


def _resolve_lr(pde_name: str, cfg: Dict[str, Any], cli_lr: Optional[float]) -> float:
    if cli_lr is not None:
        return cli_lr
    best = cfg.get("best_lr")
    if best is not None:
        return float(best)
    return DEFAULT_LRS.get(pde_name, 1e-3)


# ---------------------------------------------------------------------------
# Subcommand: train
# ---------------------------------------------------------------------------
def cmd_train(args: argparse.Namespace) -> int:
    set_seed(args.seed)
    device = get_device(prefer_cuda=not args.cpu)
    cfg = _load_pde_config(args.pde, args.config)
    lr = _resolve_lr(args.pde, cfg, args.lr)

    LOGGER.info(
        "Training PDE=%s optimizer=%s width=%d seed=%d lr=%g device=%s",
        args.pde, args.optimizer, args.width, args.seed, lr, device,
    )

    pde, model, data = _build_problem(args.pde, args.width, args.seed, device)
    loss_fn = make_loss_fn(model, pde, data)

    history: Dict[str, Any] = {}
    lbfgs_history = None
    t0 = time.time()

    if args.optimizer == "adam":
        from src.optimizers.adam_lbfgs import train_adam

        model, history = train_adam(
            model, loss_fn, lr=lr, total_iters=args.total_iters,
            verbose=args.verbose, log_every=args.log_every,
        )
    elif args.optimizer == "lbfgs":
        from src.optimizers.adam_lbfgs import train_lbfgs

        model, history = train_lbfgs(
            model, loss_fn, lr=1.0, total_iters=args.total_iters,
            memory=args.lbfgs_memory, verbose=args.verbose, log_every=args.log_every,
        )
    elif args.optimizer == "adam_lbfgs":
        from src.optimizers.adam_lbfgs import train_adam_lbfgs

        model, history, lbfgs_history = train_adam_lbfgs(
            model, loss_fn, adam_lr=lr, switch_iter=args.switch_iter,
            total_iters=args.total_iters, lbfgs_memory=args.lbfgs_memory,
            record_curvature=args.record_curvature,
            verbose=args.verbose, log_every=args.log_every,
        )
    elif args.optimizer == "nncg":
        # NNCG is a fine-tuning optimizer: run Adam+L-BFGS first, then NNCG.
        from src.optimizers.adam_lbfgs import train_adam_lbfgs
        from src.optimizers.nncg import nncg_minimize

        model, history, lbfgs_history = train_adam_lbfgs(
            model, loss_fn, adam_lr=lr, switch_iter=args.switch_iter,
            total_iters=args.total_iters, lbfgs_memory=args.lbfgs_memory,
            record_curvature=args.record_curvature,
            verbose=args.verbose, log_every=args.log_every,
        )
        model, nncg_summary, nncg_hist = nncg_minimize(
            model, loss_fn, mu=args.mu, K=args.nncg_iters,
            verbose=args.verbose, log_every=args.log_every,
        )
        history["nncg"] = nncg_hist
        history["nncg_summary"] = nncg_summary
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    elapsed = time.time() - t0

    # Final evaluation.
    metrics = evaluate(model, pde, data)
    metrics["wall_clock_s"] = elapsed
    metrics["optimizer"] = args.optimizer
    metrics["pde"] = args.pde
    metrics["width"] = args.width
    metrics["seed"] = args.seed
    metrics["lr"] = lr

    LOGGER.info(
        "Final: loss=%.6e L2RE=%.6e grad_norm=%.6e (%.1fs)",
        metrics["loss"], metrics["l2re"], metrics.get("grad_norm", float("nan")), elapsed,
    )

    # Persist results.
    out_dir = results_dir(
        args.results_root, args.pde, args.optimizer, f"w{args.width}_s{args.seed}"
    )
    save_checkpoint(
        {
            "model_state": model.state_dict(),
            "width": args.width,
            "depth": 3,
            "pde": args.pde,
            "optimizer": args.optimizer,
            "seed": args.seed,
            "lr": lr,
        },
        os.path.join(out_dir, "model.pt"),
    )
    save_json(metrics, os.path.join(out_dir, "metrics.json"))
    save_json(history, os.path.join(out_dir, "history.json"))

    if lbfgs_history is not None and args.record_curvature:
        try:
            save_checkpoint(
                lbfgs_history.to_dict(),
                os.path.join(out_dir, "lbfgs_history.pt"),
            )
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Could not save L-BFGS history: %s", exc)

    LOGGER.info("Saved results to %s", out_dir)
    return 0


# ---------------------------------------------------------------------------
# Subcommand: eval
# ---------------------------------------------------------------------------
def cmd_eval(args: argparse.Namespace) -> int:
    set_seed(args.seed)
    device = get_device(prefer_cuda=not args.cpu)
    pde, model, data = _build_problem(args.pde, args.width, args.seed, device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state)
        LOGGER.info("Loaded checkpoint %s", args.checkpoint)

    metrics = evaluate(model, pde, data)
    LOGGER.info(
        "Eval: loss=%.6e L2RE=%.6e grad_norm=%.6e",
        metrics["loss"], metrics["l2re"], metrics.get("grad_norm", float("nan")),
    )
    print(metrics)
    return 0


# ---------------------------------------------------------------------------
# Subcommand: spectral
# ---------------------------------------------------------------------------
def cmd_spectral(args: argparse.Namespace) -> int:
    set_seed(args.seed)
    device = get_device(prefer_cuda=not args.cpu)
    cfg = _load_pde_config(args.pde, args.config)
    lr = _resolve_lr(args.pde, cfg, args.lr)

    LOGGER.info(
        "Spectral density: PDE=%s width=%d seed=%d switch_iter=%d",
        args.pde, args.width, args.seed, args.switch_iter,
    )

    pde, model, data = _build_problem(args.pde, args.width, args.seed, device)
    loss_fn = make_loss_fn(model, pde, data)

    # Train up to the switch point (Adam+L-BFGS) to reach the analysis point.
    from src.optimizers.adam_lbfgs import train_adam_lbfgs

    model, history, lbfgs_history = train_adam_lbfgs(
        model, loss_fn, adam_lr=lr, switch_iter=args.switch_iter,
        total_iters=args.switch_iter, lbfgs_memory=args.lbfgs_memory,
        record_curvature=True, verbose=args.verbose, log_every=args.log_every,
    )

    from src.hessian.hvp import HVPOperator
    from src.hessian.spectral_density import spectral_density

    p = sum(pp.numel() for pp in model.parameters())
    results: Dict[str, Any] = {}

    # Full-loss spectral density.
    op = HVPOperator(loss_fn, model)
    results["full"] = spectral_density(
        op, p, num_matvecs=args.num_matvecs, num_repeats=args.num_repeats,
        num_bins=args.num_bins, device=device,
    )

    # Per-component spectral densities (residual / ic / bc).
    if args.per_component:
        comp_fns = component_loss_fns(model, pde, data)
        for name, fn in comp_fns.items():
            comp_op = HVPOperator(fn, model)
            results[name] = spectral_density(
                comp_op, p, num_matvecs=args.num_matvecs, num_repeats=args.num_repeats,
                num_bins=args.num_bins, device=device,
            )

    out_dir = results_dir(
        args.results_root, "spectral", args.pde, f"w{args.width}_s{args.seed}"
    )
    # Convert tensors to lists for JSON serialization.
    serializable = {
        k: {kk: (vv.tolist() if torch.is_tensor(vv) else vv) for kk, vv in v.items()}
        for k, v in results.items()
    }
    save_json(serializable, os.path.join(out_dir, "spectral_density.json"))
    LOGGER.info("Saved spectral density to %s", out_dir)
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="opt_for_pinns",
        description="Training and analysis tools for PINNs (loss landscape perspective).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--pde", choices=["convection", "reaction", "wave"], required=True)
    common.add_argument("--width", type=int, default=100, choices=[50, 100, 200, 400])
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--config", type=str, default=None, help="Path to a YAML config.")
    common.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    common.add_argument("--verbose", action="store_true")
    common.add_argument("--log-every", type=int, default=1000)
    common.add_argument("--results-root", type=str, default="results")

    # train
    p_train = sub.add_parser("train", parents=[common], help="Train a PINN.")
    p_train.add_argument(
        "--optimizer",
        choices=["adam", "lbfgs", "adam_lbfgs", "nncg"],
        default="adam_lbfgs",
    )
    p_train.add_argument("--lr", type=float, default=None, help="Adam learning rate.")
    p_train.add_argument("--switch-iter", type=int, default=11000)
    p_train.add_argument("--total-iters", type=int, default=41000)
    p_train.add_argument("--lbfgs-memory", type=int, default=100)
    p_train.add_argument("--record-curvature", action="store_true")
    p_train.add_argument("--mu", type=float, default=1e-2, help="NNCG damping.")
    p_train.add_argument("--nncg-iters", type=int, default=2000)
    p_train.set_defaults(func=cmd_train)

    # eval
    p_eval = sub.add_parser("eval", parents=[common], help="Evaluate a checkpoint.")
    p_eval.add_argument("--checkpoint", type=str, default=None)
    p_eval.set_defaults(func=cmd_eval)

    # spectral
    p_spec = sub.add_parser("spectral", parents=[common], help="Hessian spectral density.")
    p_spec.add_argument("--lr", type=float, default=None)
    p_spec.add_argument("--switch-iter", type=int, default=11000)
    p_spec.add_argument("--lbfgs-memory", type=int, default=100)
    p_spec.add_argument("--num-matvecs", type=int, default=100)
    p_spec.add_argument("--num-repeats", type=int, default=1)
    p_spec.add_argument("--num-bins", type=int, default=200)
    p_spec.add_argument("--per-component", action="store_true")
    p_spec.set_defaults(func=cmd_spectral)

    return parser


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
