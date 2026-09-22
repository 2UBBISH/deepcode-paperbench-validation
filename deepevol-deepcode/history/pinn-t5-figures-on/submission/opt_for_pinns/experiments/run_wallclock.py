"""Wall-clock timing experiment (Table 3, Section 7.4).

Measures the per-iteration wall-clock time of L-BFGS and NNCG for each PDE
(convection, reaction, wave).  The paper reports:

    L-BFGS : {4.6e-2, 3.6e-2, 9.0e-2} s   (convection, reaction, wave)
    NNCG   : {2.5e-1, 7.2e-1, 2.9e1} s
    ratio  : {5.43, 20, 322.22}

The wave equation is by far the slowest for NNCG because its residual requires
second-order derivatives (u_tt, u_xx), making each Hessian-vector product
substantially more expensive.

This script:
  1. Trains a PINN with Adam+L-BFGS to a reasonable point (per-PDE best config).
  2. Times a fixed number of L-BFGS iterations (per-iteration wall clock).
  3. Times a fixed number of NNCG iterations (per-iteration wall clock).
  4. Reports the ratio NNCG / L-BFGS and writes ``results.json`` + a bar plot.

Usage
-----
    python -m opt_for_pinns.experiments.run_wallclock --outdir results/wallclock
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch

# Allow running as a script from the repository root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from opt_for_pinns.src.data import build_data
from opt_for_pinns.src.model import build_model
from opt_for_pinns.src.pdes import get_pde
from opt_for_pinns.src.train import (
    finetune_nncg,
    train_adam_lbfgs,
    train_lbfgs,
)
from opt_for_pinns.src.utils import ensure_dir, get_logger, save_json, set_seed

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_PDES = ["convection", "reaction", "wave"]
DEFAULT_SEEDS = [123, 234, 345, 456, 567]

# Per-PDE best configuration (width, Adam lr, switch iter) from Section 6.1.
BEST_CONFIG = {
    "convection": {"width": 200, "adam_lr": 1e-4, "switch_iter": 11000, "seed": 345},
    "reaction": {"width": 200, "adam_lr": 1e-3, "switch_iter": 11000, "seed": 456},
    "wave": {"width": 200, "adam_lr": 1e-3, "switch_iter": 11000, "seed": 567},
}

TOTAL_ITERS = 41000

# Number of iterations to time for each optimizer.  Kept small so the
# experiment finishes in reasonable time while still giving a stable estimate.
N_TIMED_LBFGS = 20
N_TIMED_NNCG = 20

# NNCG hyperparameters (Section 7.2 / Appendix E.2).
NNCG_KWARGS = dict(
    n_steps=N_TIMED_NNCG,
    s=60,
    F=20,
    mu=1e-2,
    eps=1e-16,
    M=1000,
    eta=1.0,
    alpha=0.1,
    beta=0.5,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _summarize(values: List[float]) -> Dict[str, float]:
    """Return min/median/max/mean of a list of floats."""
    if not values:
        return {"min": float("nan"), "median": float("nan"),
                "max": float("nan"), "mean": float("nan")}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _time_lbfgs(pde_name: str, width: int, seed: int, adam_lr: float,
                switch_iter: int, total_iters: int, n_timed: int,
                logger=None) -> Dict:
    """Train Adam+L-BFGS, then time ``n_timed`` L-BFGS iterations."""
    set_seed(seed)
    pde = get_pde(pde_name)
    model = build_model(width=width, depth=3, seed=seed)
    data = build_data(pde, seed=seed)

    # Warm up with Adam + L-BFGS to reach a realistic operating point.
    res = train_adam_lbfgs(
        pde, model, data,
        adam_lr=adam_lr,
        switch_iter=switch_iter,
        total_iters=total_iters,
        log_every=0,
        verbose=False,
    )

    # Now time a fresh L-BFGS phase (continuing from the trained model).
    t0 = time.perf_counter()
    timed = train_lbfgs(
        pde, model, data,
        lr=1.0,
        n_iters=n_timed,
        history_size=100,
        line_search_fn="strong_wolfe",
        log_every=0,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0

    per_iter = elapsed / max(1, n_timed)
    if logger is not None:
        logger.info(
            "[%s] L-BFGS: %.4e s/iter (total %.3f s over %d iters)",
            pde_name, per_iter, elapsed, n_timed,
        )
    return {
        "pde": pde_name,
        "optimizer": "lbfgs",
        "width": width,
        "seed": seed,
        "n_timed": n_timed,
        "total_time": elapsed,
        "per_iter_time": per_iter,
        "loss": float(timed.loss),
        "l2re": float(timed.l2re),
    }


def _time_nncg(pde_name: str, width: int, seed: int, adam_lr: float,
               switch_iter: int, total_iters: int, n_timed: int,
               logger=None) -> Dict:
    """Train Adam+L-BFGS, then time ``n_timed`` NNCG iterations."""
    set_seed(seed)
    pde = get_pde(pde_name)
    model = build_model(width=width, depth=3, seed=seed)
    data = build_data(pde, seed=seed)

    # Warm up with Adam + L-BFGS to reach a realistic operating point.
    train_adam_lbfgs(
        pde, model, data,
        adam_lr=adam_lr,
        switch_iter=switch_iter,
        total_iters=total_iters,
        log_every=0,
        verbose=False,
    )

    # Time NNCG fine-tuning.
    kwargs = dict(NNCG_KWARGS)
    kwargs["n_steps"] = n_timed
    t0 = time.perf_counter()
    timed = finetune_nncg(pde, model, data, log_every=0, verbose=False, **kwargs)
    elapsed = time.perf_counter() - t0

    per_iter = elapsed / max(1, n_timed)
    if logger is not None:
        logger.info(
            "[%s] NNCG: %.4e s/iter (total %.3f s over %d iters)",
            pde_name, per_iter, elapsed, n_timed,
        )
    return {
        "pde": pde_name,
        "optimizer": "nncg",
        "width": width,
        "seed": seed,
        "n_timed": n_timed,
        "total_time": elapsed,
        "per_iter_time": per_iter,
        "loss": float(timed.loss),
        "l2re": float(timed.l2re),
    }


def _plot_wallclock(records: List[Dict], outdir: str, logger=None) -> None:
    """Bar plot comparing per-iteration times of L-BFGS vs NNCG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        if logger is not None:
            logger.warning("matplotlib unavailable, skipping plot: %s", exc)
        return

    pdes = sorted({r["pde"] for r in records})
    lbfgs_times = []
    nncg_times = []
    for pde in pdes:
        lb = [r["per_iter_time"] for r in records
              if r["pde"] == pde and r["optimizer"] == "lbfgs"]
        nn = [r["per_iter_time"] for r in records
              if r["pde"] == pde and r["optimizer"] == "nncg"]
        lbfgs_times.append(float(np.median(lb)) if lb else float("nan"))
        nncg_times.append(float(np.median(nn)) if nn else float("nan"))

    x = np.arange(len(pdes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width / 2, lbfgs_times, width, label="L-BFGS")
    ax.bar(x + width / 2, nncg_times, width, label="NNCG")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(pdes)
    ax.set_ylabel("Per-iteration wall-clock time (s)")
    ax.set_title("Per-iteration cost: L-BFGS vs NNCG")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(outdir, "table3_wallclock.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    if logger is not None:
        logger.info("Saved wall-clock plot to %s", path)


# ---------------------------------------------------------------------------
# Main experiment driver
# ---------------------------------------------------------------------------


def run(pdes: Optional[List[str]] = None,
        seeds: Optional[List[int]] = None,
        outdir: str = "results/wallclock",
        total_iters: int = TOTAL_ITERS,
        n_timed_lbfgs: int = N_TIMED_LBFGS,
        n_timed_nncg: int = N_TIMED_NNCG,
        log_every: int = 0,
        logger=None) -> Dict:
    """Run the wall-clock timing experiment (Table 3)."""
    pdes = list(pdes) if pdes else list(DEFAULT_PDES)
    seeds = list(seeds) if seeds else list(DEFAULT_SEEDS)
    ensure_dir(outdir)
    if logger is None:
        logger = get_logger("run_wallclock")

    records: List[Dict] = []
    for pde_name in pdes:
        cfg = BEST_CONFIG.get(pde_name, {"width": 200, "adam_lr": 1e-3,
                                         "switch_iter": 11000, "seed": seeds[0]})
        width = cfg["width"]
        adam_lr = cfg["adam_lr"]
        switch_iter = cfg["switch_iter"]

        # Use the per-PDE best seed first, then any additional seeds requested.
        run_seeds = [cfg["seed"]] + [s for s in seeds if s != cfg["seed"]]

        for seed in run_seeds:
            logger.info("Timing L-BFGS on %s (width=%d, seed=%d)", pde_name, width, seed)
            records.append(_time_lbfgs(
                pde_name, width, seed, adam_lr, switch_iter,
                total_iters, n_timed_lbfgs, logger=logger,
            ))

            logger.info("Timing NNCG on %s (width=%d, seed=%d)", pde_name, width, seed)
            records.append(_time_nncg(
                pde_name, width, seed, adam_lr, switch_iter,
                total_iters, n_timed_nncg, logger=logger,
            ))

    # Aggregate per (pde, optimizer).
    summary: Dict[str, Dict] = {}
    for pde_name in pdes:
        summary[pde_name] = {}
        for opt in ("lbfgs", "nncg"):
            vals = [r["per_iter_time"] for r in records
                    if r["pde"] == pde_name and r["optimizer"] == opt]
            summary[pde_name][opt] = _summarize(vals)
        lb = summary[pde_name]["lbfgs"]["median"]
        nn = summary[pde_name]["nncg"]["median"]
        summary[pde_name]["ratio"] = (nn / lb) if lb and lb > 0 else float("nan")

    payload = {"records": records, "summary": summary}
    save_json(payload, os.path.join(outdir, "results.json"))
    _plot_wallclock(records, outdir, logger=logger)

    # Print a Table-3-like summary.
    logger.info("=" * 60)
    logger.info("Per-iteration wall-clock time (seconds)")
    logger.info("%-12s %-12s %-12s %-10s", "PDE", "L-BFGS", "NNCG", "ratio")
    for pde_name in pdes:
        lb = summary[pde_name]["lbfgs"]["median"]
        nn = summary[pde_name]["nncg"]["median"]
        ratio = summary[pde_name]["ratio"]
        logger.info("%-12s %-12.4e %-12.4e %-10.2f", pde_name, lb, nn, ratio)
    logger.info("=" * 60)

    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Wall-clock timing experiment (Table 3)."
    )
    parser.add_argument("--outdir", type=str, default="results/wallclock",
                        help="Output directory for results and figures.")
    parser.add_argument("--pdes", type=str, nargs="+", default=DEFAULT_PDES,
                        help="PDEs to time.")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help="Random seeds.")
    parser.add_argument("--total-iters", type=int, default=TOTAL_ITERS,
                        help="Total Adam+L-BFGS iterations before timing.")
    parser.add_argument("--n-timed-lbfgs", type=int, default=N_TIMED_LBFGS,
                        help="Number of L-BFGS iterations to time.")
    parser.add_argument("--n-timed-nncg", type=int, default=N_TIMED_NNCG,
                        help="Number of NNCG iterations to time.")
    parser.add_argument("--log-every", type=int, default=0,
                        help="Logging frequency (0 = silent).")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logger = get_logger("run_wallclock")
    run(
        pdes=args.pdes,
        seeds=args.seeds,
        outdir=args.outdir,
        total_iters=args.total_iters,
        n_timed_lbfgs=args.n_timed_lbfgs,
        n_timed_nncg=args.n_timed_nncg,
        log_every=args.log_every,
        logger=logger,
    )


if __name__ == "__main__":
    main()
