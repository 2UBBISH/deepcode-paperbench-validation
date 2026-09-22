"""Experiment orchestration for the opt_for_pinns codebase.

This module runs the full experiment grid described in the paper
"Challenges in Training PINNs: A Loss Landscape Perspective":

    PDE x optimizer x width x seed

It produces the raw results (loss, L2RE, gradient norm, histories) that the
``scripts/reproduce_*.py`` files consume to build Tables 1-3 and Figures 1-8.

Design notes
------------
* Each (pde, optimizer, width, seed) run is fully self-contained and writes its
  results to ``results/<pde>/<optimizer>/w<width>_s<seed>/``.
* Learning rates are tuned per (pde, optimizer) by selecting the value from the
  configured grid that yields the smallest final L2RE (matching the paper's
  selection process).  The best lr is cached in ``results/<pde>/best_lr.json``.
* Optimizers supported: ``adam``, ``lbfgs``, ``adam_lbfgs`` (and ``nncg`` as a
  fine-tuning stage on top of ``adam_lbfgs``).
* The default grid matches the paper: widths {50, 100, 200, 400}, 5 seeds,
  41000 total iterations for Adam+L-BFGS.

Usage
-----
    python run_experiments.py --pdes convection reaction wave \
        --optimizers adam lbfgs adam_lbfgs \
        --widths 50 100 200 400 --seeds 0 1 2 3 4

    # Quick smoke test
    python run_experiments.py --pdes convection --optimizers adam_lbfgs \
        --widths 50 --seeds 0 --total-iters 200
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

# Allow running as a script from the package root.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from src.data import build_data, make_eval_points  # noqa: E402
from src.loss import make_loss_fn  # noqa: E402
from src.metrics import evaluate, l2_relative_error_from_data  # noqa: E402
from src.model import build_model  # noqa: E402
from src.pdes import get_pde  # noqa: E402
from src.utils import (  # noqa: E402
    ensure_dir,
    get_device,
    get_logger,
    load_config,
    results_dir,
    save_json,
    set_seed,
)
from src.optimizers.adam_lbfgs import (  # noqa: E402
    train_adam,
    train_adam_lbfgs,
    train_lbfgs,
)

logger = get_logger("run_experiments")


# ---------------------------------------------------------------------------
# Defaults (paper Section 2.2 / Section 4)
# ---------------------------------------------------------------------------

DEFAULT_PDES = ["convection", "reaction", "wave"]
DEFAULT_OPTIMIZERS = ["adam", "lbfgs", "adam_lbfgs"]
DEFAULT_WIDTHS = [50, 100, 200, 400]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]

# Learning-rate grids used for tuning (paper Section 4).
ADAM_LR_GRID = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
LBFGS_LR_GRID = [1.0]
ADAM_LBFGS_LR_GRID = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

# Paper defaults.
TOTAL_ITERS = 41000
SWITCH_ITERS = [1000, 11000, 31000]
DEFAULT_SWITCH = 11000
LBFGS_MEMORY = 100

# Fallback best lr per PDE (used when tuning is skipped).
DEFAULT_BEST_LR = {
    "convection": 1e-4,
    "reaction": 1e-3,
    "wave": 1e-3,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pde_config_path(pde_name: str, config_dir: str) -> str:
    return os.path.join(config_dir, f"{pde_name}.yaml")


def _load_pde_config(pde_name: str, config_dir: str) -> Dict[str, Any]:
    path = _pde_config_path(pde_name, config_dir)
    if os.path.exists(path):
        return load_config(path)
    return {}


def _build_problem(
    pde_name: str,
    width: int,
    seed: int,
    device: torch.device,
    pde_kwargs: Optional[Dict[str, Any]] = None,
):
    """Instantiate (pde, model, data) for a single run."""
    pde = get_pde(pde_name, **(pde_kwargs or {}))
    model = build_model(width=width, depth=3, seed=seed, device=device)
    data = build_data(pde, seed=seed, device=device)
    return pde, model, data


def _run_dir(root: str, pde_name: str, optimizer: str, width: int, seed: int) -> str:
    return str(results_dir(root, pde_name, optimizer, f"w{width}_s{seed}"))


def _train_once(
    pde_name: str,
    optimizer: str,
    width: int,
    seed: int,
    lr: float,
    total_iters: int,
    switch_iter: int,
    device: torch.device,
    pde_kwargs: Optional[Dict[str, Any]] = None,
    record_curvature: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Train a single PINN and return a result dict with metrics + history."""
    set_seed(seed)
    pde, model, data = _build_problem(pde_name, width, seed, device, pde_kwargs)
    loss_fn = make_loss_fn(model, pde, data)

    t0 = time.time()
    curvature = None
    if optimizer == "adam":
        model, history = train_adam(
            model, loss_fn, lr=lr, total_iters=total_iters, verbose=verbose
        )
    elif optimizer == "lbfgs":
        model, history = train_lbfgs(
            model,
            loss_fn,
            lr=lr,
            total_iters=total_iters,
            memory=LBFGS_MEMORY,
            verbose=verbose,
        )
    elif optimizer == "adam_lbfgs":
        model, history, curvature = train_adam_lbfgs(
            model,
            loss_fn,
            adam_lr=lr,
            switch_iter=switch_iter,
            total_iters=total_iters,
            lbfgs_memory=LBFGS_MEMORY,
            record_curvature=record_curvature,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    wall_time = time.time() - t0

    metrics = evaluate(model, pde, data)
    result: Dict[str, Any] = {
        "pde": pde_name,
        "optimizer": optimizer,
        "width": width,
        "seed": seed,
        "lr": lr,
        "total_iters": total_iters,
        "switch_iter": switch_iter if optimizer == "adam_lbfgs" else None,
        "wall_time": wall_time,
        "loss": metrics["loss"],
        "l2re": metrics["l2re"],
        "grad_norm": metrics.get("grad_norm"),
        "loss_residual": metrics.get("loss_residual"),
        "loss_ic": metrics.get("loss_ic"),
        "loss_bc": metrics.get("loss_bc"),
        "history": history,
    }

    if curvature is not None:
        result["curvature"] = curvature.to_dict()

    return result


def _save_run(result: Dict[str, Any], root: str) -> str:
    out_dir = _run_dir(
        root, result["pde"], result["optimizer"], result["width"], result["seed"]
    )
    ensure_dir(out_dir)
    save_json(result, os.path.join(out_dir, "result.json"))
    return out_dir


# ---------------------------------------------------------------------------
# Learning-rate tuning
# ---------------------------------------------------------------------------

def tune_lr(
    pde_name: str,
    optimizer: str,
    width: int,
    seed: int,
    lr_grid: Sequence[float],
    total_iters: int,
    switch_iter: int,
    device: torch.device,
    pde_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Select the lr from ``lr_grid`` with the smallest final L2RE."""
    trials: List[Dict[str, Any]] = []
    best_lr = lr_grid[0]
    best_l2re = float("inf")
    for lr in lr_grid:
        res = _train_once(
            pde_name,
            optimizer,
            width,
            seed,
            lr,
            total_iters,
            switch_iter,
            device,
            pde_kwargs,
            verbose=verbose,
        )
        trials.append({"lr": lr, "loss": res["loss"], "l2re": res["l2re"]})
        if res["l2re"] < best_l2re:
            best_l2re = res["l2re"]
            best_lr = lr
        logger.info(
            "[tune] pde=%s opt=%s w=%d s=%d lr=%.1e -> loss=%.3e l2re=%.3e",
            pde_name, optimizer, width, seed, lr, res["loss"], res["l2re"],
        )
    return best_lr, trials


def resolve_best_lr(
    pde_name: str,
    optimizer: str,
    root: str,
    tune: bool,
    tune_width: int,
    tune_seed: int,
    total_iters: int,
    switch_iter: int,
    device: torch.device,
    pde_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> float:
    """Return the best lr for (pde, optimizer), tuning if requested."""
    cache_path = os.path.join(root, pde_name, "best_lr.json")
    cache: Dict[str, Any] = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as fh:
                cache = json.load(fh)
        except Exception:
            cache = {}

    key = optimizer
    if not tune and key in cache:
        return float(cache[key])

    if not tune:
        return DEFAULT_BEST_LR.get(pde_name, 1e-3)

    if optimizer == "adam":
        grid = ADAM_LR_GRID
    elif optimizer == "lbfgs":
        grid = LBFGS_LR_GRID
    else:
        grid = ADAM_LBFGS_LR_GRID

    best_lr, trials = tune_lr(
        pde_name,
        optimizer,
        tune_width,
        tune_seed,
        grid,
        total_iters,
        switch_iter,
        device,
        pde_kwargs,
        verbose=verbose,
    )
    cache[key] = best_lr
    cache[f"{key}_trials"] = trials
    ensure_dir(os.path.dirname(cache_path))
    save_json(cache, cache_path)
    logger.info("[tune] pde=%s opt=%s best_lr=%.1e", pde_name, optimizer, best_lr)
    return best_lr


# ---------------------------------------------------------------------------
# Grid orchestration
# ---------------------------------------------------------------------------

def run_grid(
    pdes: Sequence[str],
    optimizers: Sequence[str],
    widths: Sequence[int],
    seeds: Sequence[int],
    root: str = "results",
    config_dir: str = "configs",
    total_iters: int = TOTAL_ITERS,
    switch_iter: int = DEFAULT_SWITCH,
    tune: bool = True,
    tune_width: int = 100,
    tune_seed: int = 0,
    device: Optional[torch.device] = None,
    verbose: bool = False,
    skip_existing: bool = True,
) -> List[Dict[str, Any]]:
    """Run the full experiment grid and return the list of result dicts."""
    device = device or get_device()
    ensure_dir(root)
    all_results: List[Dict[str, Any]] = []

    for pde_name in pdes:
        cfg = _load_pde_config(pde_name, config_dir)
        pde_kwargs = cfg.get("pde_kwargs", {}) if isinstance(cfg, dict) else {}

        for optimizer in optimizers:
            best_lr = resolve_best_lr(
                pde_name,
                optimizer,
                root,
                tune,
                tune_width,
                tune_seed,
                total_iters,
                switch_iter,
                device,
                pde_kwargs,
                verbose=verbose,
            )

            for width, seed in itertools.product(widths, seeds):
                out_dir = _run_dir(root, pde_name, optimizer, width, seed)
                result_path = os.path.join(out_dir, "result.json")
                if skip_existing and os.path.exists(result_path):
                    try:
                        with open(result_path, "r") as fh:
                            all_results.append(json.load(fh))
                        logger.info("[skip] %s", result_path)
                        continue
                    except Exception:
                        pass

                logger.info(
                    "[run] pde=%s opt=%s w=%d s=%d lr=%.1e",
                    pde_name, optimizer, width, seed, best_lr,
                )
                result = _train_once(
                    pde_name,
                    optimizer,
                    width,
                    seed,
                    best_lr,
                    total_iters,
                    switch_iter,
                    device,
                    pde_kwargs,
                    verbose=verbose,
                )
                _save_run(result, root)
                all_results.append(result)
                logger.info(
                    "[done] pde=%s opt=%s w=%d s=%d loss=%.3e l2re=%.3e (%.1fs)",
                    pde_name, optimizer, width, seed,
                    result["loss"], result["l2re"], result["wall_time"],
                )

    # Persist an aggregate summary for downstream scripts.
    summary = [
        {
            "pde": r["pde"],
            "optimizer": r["optimizer"],
            "width": r["width"],
            "seed": r["seed"],
            "lr": r["lr"],
            "loss": r["loss"],
            "l2re": r["l2re"],
            "grad_norm": r.get("grad_norm"),
            "wall_time": r.get("wall_time"),
        }
        for r in all_results
    ]
    save_json(summary, os.path.join(root, "summary.json"))
    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the PINN optimizer experiment grid."
    )
    p.add_argument("--pdes", nargs="+", default=DEFAULT_PDES)
    p.add_argument("--optimizers", nargs="+", default=DEFAULT_OPTIMIZERS)
    p.add_argument("--widths", nargs="+", type=int, default=DEFAULT_WIDTHS)
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--root", type=str, default="results")
    p.add_argument("--config-dir", type=str, default="configs")
    p.add_argument("--total-iters", type=int, default=TOTAL_ITERS)
    p.add_argument("--switch-iter", type=int, default=DEFAULT_SWITCH)
    p.add_argument("--tune", action="store_true", default=False,
                   help="Tune the learning rate per (pde, optimizer).")
    p.add_argument("--tune-width", type=int, default=100)
    p.add_argument("--tune-seed", type=int, default=0)
    p.add_argument("--no-skip-existing", action="store_true", default=False)
    p.add_argument("--verbose", action="store_true", default=False)
    p.add_argument("--cpu", action="store_true", default=False)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device("cpu") if args.cpu else get_device()
    logger.info("Using device: %s", device)

    run_grid(
        pdes=args.pdes,
        optimizers=args.optimizers,
        widths=args.widths,
        seeds=args.seeds,
        root=args.root,
        config_dir=args.config_dir,
        total_iters=args.total_iters,
        switch_iter=args.switch_iter,
        tune=args.tune,
        tune_width=args.tune_width,
        tune_seed=args.tune_seed,
        device=device,
        verbose=args.verbose,
        skip_existing=not args.no_skip_existing,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
