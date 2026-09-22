"""Experiment: optimizer comparison (Table 1, Figure 8).

Compares Adam, L-BFGS, and Adam+L-BFGS across the three PDEs and four widths,
using 5 random seeds per configuration. Reports min/median/max of final loss and
L2RE, and produces a figure of loss vs. iteration for the best configuration.

Paper reference: Section 6.1, Table 1, Figure 8.

Usage:
    python -m opt_for_pinns.experiments.run_optimizer_comparison \
        --outdir results/optimizer_comparison \
        --pdes convection reaction wave \
        --widths 50 100 200 400 \
        --seeds 123 234 345 456 567
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch

# Allow running as a script from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from opt_for_pinns.src.data import build_data
from opt_for_pinns.src.model import build_model
from opt_for_pinns.src.pdes import get_pde
from opt_for_pinns.src.train import train
from opt_for_pinns.src.utils import ensure_dir, get_logger, save_json, set_seed, summarize

# Adam learning-rate grid searched per (PDE, width, seed).
ADAM_LR_GRID = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
# Switch points (Adam -> L-BFGS) searched for the combined optimizer.
SWITCH_GRID = [1000, 11000, 31000]

TOTAL_ITERS = 41000


def _run_single(pde_name: str, width: int, seed: int, optimizer: str,
                adam_lr: float = 1e-3, switch_iter: int = 11000,
                total_iters: int = TOTAL_ITERS, log_every: int = 0,
                logger=None) -> Dict:
    """Train one (PDE, width, seed, optimizer) configuration and return metrics."""
    set_seed(seed)
    pde = get_pde(pde_name)
    model = build_model(width=width, depth=3, seed=seed)
    data = build_data(pde, seed=seed)

    if optimizer == "adam":
        res = train(pde, model, data, optimizer_name="adam",
                    lr=adam_lr, n_iters=total_iters, log_every=log_every)
    elif optimizer == "lbfgs":
        res = train(pde, model, data, optimizer_name="lbfgs",
                    lr=1.0, n_iters=total_iters, log_every=log_every)
    elif optimizer == "adam_lbfgs":
        res = train(pde, model, data, optimizer_name="adam_lbfgs",
                    adam_lr=adam_lr, switch_iter=switch_iter,
                    total_iters=total_iters, log_every=log_every)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    out = res.as_dict()
    out.update({"pde": pde_name, "width": width, "seed": seed,
                "optimizer": optimizer, "adam_lr": adam_lr,
                "switch_iter": switch_iter})
    if logger is not None:
        logger.info(
            "[%s w=%d seed=%d %s] loss=%.4e l2re=%.4e grad=%.4e",
            pde_name, width, seed, optimizer, out["loss"], out["l2re"],
            out.get("grad_norm", float("nan")),
        )
    return out


def _best_adam_lr(pde_name: str, width: int, seed: int, logger=None) -> float:
    """Grid-search Adam lr for a given (PDE, width, seed); return best lr."""
    best_lr, best_loss = ADAM_LR_GRID[0], float("inf")
    for lr in ADAM_LR_GRID:
        r = _run_single(pde_name, width, seed, "adam", adam_lr=lr,
                        total_iters=TOTAL_ITERS, logger=None)
        if r["loss"] < best_loss:
            best_loss, best_lr = r["loss"], lr
    if logger is not None:
        logger.info("Best Adam lr for %s w=%d seed=%d: %g (loss=%.4e)",
                    pde_name, width, seed, best_lr, best_loss)
    return best_lr


def _best_switch(pde_name: str, width: int, seed: int, adam_lr: float,
                 logger=None) -> int:
    """Grid-search the Adam->L-BFGS switch point; return best switch iter."""
    best_sw, best_loss = SWITCH_GRID[0], float("inf")
    for sw in SWITCH_GRID:
        r = _run_single(pde_name, width, seed, "adam_lbfgs", adam_lr=adam_lr,
                        switch_iter=sw, total_iters=TOTAL_ITERS, logger=None)
        if r["loss"] < best_loss:
            best_loss, best_sw = r["loss"], sw
    if logger is not None:
        logger.info("Best switch for %s w=%d seed=%d: %d (loss=%.4e)",
                    pde_name, width, seed, best_sw, best_loss)
    return best_sw


def run(pdes: List[str], widths: List[int], seeds: List[int],
        outdir: str, total_iters: int = TOTAL_ITERS,
        log_every: int = 0, logger=None) -> Dict:
    """Run the full optimizer-comparison sweep and persist results."""
    ensure_dir(outdir)
    all_records: List[Dict] = []

    for pde_name, width, seed in itertools.product(pdes, widths, seeds):
        # 1) Adam: search lr grid.
        adam_lr = _best_adam_lr(pde_name, width, seed, logger=logger)
        rec_adam = _run_single(pde_name, width, seed, "adam",
                               adam_lr=adam_lr, total_iters=total_iters,
                               log_every=log_every, logger=logger)
        all_records.append(rec_adam)

        # 2) L-BFGS (lr=1.0, memory=100).
        rec_lbfgs = _run_single(pde_name, width, seed, "lbfgs",
                                total_iters=total_iters,
                                log_every=log_every, logger=logger)
        all_records.append(rec_lbfgs)

        # 3) Adam+L-BFGS: search switch point.
        switch = _best_switch(pde_name, width, seed, adam_lr, logger=logger)
        rec_comb = _run_single(pde_name, width, seed, "adam_lbfgs",
                               adam_lr=adam_lr, switch_iter=switch,
                               total_iters=total_iters,
                               log_every=log_every, logger=logger)
        all_records.append(rec_comb)

    # Aggregate min/median/max per (PDE, optimizer) over widths & seeds.
    summary: Dict[str, Dict] = {}
    for pde_name in pdes:
        summary[pde_name] = {}
        for opt in ["adam", "lbfgs", "adam_lbfgs"]:
            recs = [r for r in all_records
                    if r["pde"] == pde_name and r["optimizer"] == opt]
            if not recs:
                continue
            summary[pde_name][opt] = {
                "loss": summarize([r["loss"] for r in recs]),
                "l2re": summarize([r["l2re"] for r in recs]),
                "grad_norm": summarize([r.get("grad_norm", float("nan"))
                                        for r in recs]),
                "n": len(recs),
            }

    save_json({"records": all_records, "summary": summary},
              os.path.join(outdir, "results.json"))

    # Print a compact table (min loss / min L2RE per PDE & optimizer).
    if logger is not None:
        logger.info("=" * 72)
        logger.info("Optimizer comparison (min loss / min L2RE)")
        logger.info("%-12s %-14s %-14s %-14s", "PDE", "Adam", "L-BFGS",
                    "Adam+L-BFGS")
        for pde_name in pdes:
            row = [pde_name]
            for opt in ["adam", "lbfgs", "adam_lbfgs"]:
                s = summary.get(pde_name, {}).get(opt)
                if s is None:
                    row.append("n/a")
                else:
                    row.append(f"{s['loss']['min']:.2e}/{s['l2re']['min']:.2e}")
            logger.info("%-12s %-14s %-14s %-14s", *row)
        logger.info("=" * 72)

    _plot_loss_curves(all_records, outdir, logger=logger)
    return {"records": all_records, "summary": summary}


def _plot_loss_curves(records: List[Dict], outdir: str, logger=None) -> None:
    """Figure 8: loss vs. iteration for the best config per optimizer."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        if logger is not None:
            logger.warning("matplotlib unavailable, skipping figure: %s", exc)
        return

    # We only have final metrics in records; plot a bar-style summary instead
    # of full curves unless history is present.
    pdes = sorted({r["pde"] for r in records})
    fig, axes = plt.subplots(1, len(pdes), figsize=(5 * len(pdes), 4),
                             squeeze=False)
    for ax, pde_name in zip(axes[0], pdes):
        opts = ["adam", "lbfgs", "adam_lbfgs"]
        vals = []
        for opt in opts:
            recs = [r for r in records
                    if r["pde"] == pde_name and r["optimizer"] == opt]
            vals.append(min((r["loss"] for r in recs), default=float("nan")))
        ax.bar(range(len(opts)), vals)
        ax.set_yscale("log")
        ax.set_xticks(range(len(opts)))
        ax.set_xticklabels(["Adam", "L-BFGS", "Adam+L-BFGS"], rotation=15)
        ax.set_title(pde_name)
        ax.set_ylabel("final loss")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "figure8_optimizer_comparison.png"), dpi=150)
    plt.close(fig)
    if logger is not None:
        logger.info("Saved figure to %s", os.path.join(
            outdir, "figure8_optimizer_comparison.png"))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Optimizer comparison (Table 1, Fig 8)")
    p.add_argument("--outdir", type=str, default="results/optimizer_comparison")
    p.add_argument("--pdes", nargs="+", default=["convection", "reaction", "wave"])
    p.add_argument("--widths", nargs="+", type=int, default=[50, 100, 200, 400])
    p.add_argument("--seeds", nargs="+", type=int, default=[123, 234, 345, 456, 567])
    p.add_argument("--total-iters", type=int, default=TOTAL_ITERS)
    p.add_argument("--log-every", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logger = get_logger("run_optimizer_comparison")
    run(args.pdes, args.widths, args.seeds, args.outdir,
        total_iters=args.total_iters, log_every=args.log_every, logger=logger)


if __name__ == "__main__":
    main()
