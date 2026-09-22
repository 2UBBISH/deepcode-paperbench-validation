"""Experiment: Loss vs L2RE scatter (Figure 2).

Reproduces Figure 2 of "Challenges in Training PINNs: A Loss Landscape
Perspective".  The figure plots the final training loss against the L2 relative
error (L2RE) for many PINN training runs, showing that lower loss generally
implies lower L2RE (a monotone trend across PDEs).

The experiment reuses the optimizer-comparison sweep: for each PDE we train
PINNs with Adam, L-BFGS and Adam+L-BFGS across several widths / seeds / learning
rates, and record the (loss, L2RE) pair for every run.  The scatter plot is then
produced from these pairs.

Usage
-----
    python -m opt_for_pinns.experiments.run_loss_vs_l2re \
        --outdir results/loss_vs_l2re

The script writes:
    - ``results.json``  : all (loss, l2re) records plus per-PDE summaries
    - ``figure2_loss_vs_l2re.png`` : the scatter plot
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

# Allow running as a plain script from the repository root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from opt_for_pinns.src.data import build_data
from opt_for_pinns.src.model import build_model
from opt_for_pinns.src.pdes import get_pde
from opt_for_pinns.src.train import train
from opt_for_pinns.src.utils import ensure_dir, get_logger, save_json, set_seed


# ---------------------------------------------------------------------------
# Defaults (paper Section 2.2 / 6.1)
# ---------------------------------------------------------------------------
DEFAULT_PDES = ["convection", "reaction", "wave"]
DEFAULT_WIDTHS = [50, 100, 200, 400]
DEFAULT_SEEDS = [123, 234, 345, 456, 567]
ADAM_LR_GRID = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
SWITCH_GRID = [1000, 11000, 31000]
TOTAL_ITERS = 41000


# ---------------------------------------------------------------------------
# Single-run helper
# ---------------------------------------------------------------------------
def _run_single(
    pde_name: str,
    width: int,
    seed: int,
    optimizer: str,
    adam_lr: float = 1e-3,
    switch_iter: int = 11000,
    total_iters: int = TOTAL_ITERS,
    log_every: int = 0,
    logger=None,
) -> Dict:
    """Train a single PINN configuration and return its (loss, l2re) record."""
    set_seed(seed)
    pde = get_pde(pde_name)
    model = build_model(width=width, depth=3, seed=seed)
    data = build_data(pde, seed=seed)

    if optimizer == "adam":
        res = train(pde, model, data, optimizer_name="adam", lr=adam_lr,
                    n_iters=total_iters, log_every=log_every)
    elif optimizer == "lbfgs":
        res = train(pde, model, data, optimizer_name="lbfgs", lr=1.0,
                    n_iters=total_iters, log_every=log_every)
    elif optimizer == "adam_lbfgs":
        res = train(pde, model, data, optimizer_name="adam_lbfgs",
                    adam_lr=adam_lr, switch_iter=switch_iter,
                    total_iters=total_iters, log_every=log_every)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    record = {
        "pde": pde_name,
        "width": width,
        "seed": seed,
        "optimizer": optimizer,
        "adam_lr": adam_lr,
        "switch_iter": switch_iter,
        "loss": float(res.loss),
        "l2re": float(res.l2re),
        "grad_norm": float(res.grad_norm),
    }
    if logger is not None:
        logger.info(
            "[loss_vs_l2re] pde=%s width=%d seed=%d opt=%s lr=%g switch=%d "
            "-> loss=%.4e l2re=%.4e",
            pde_name, width, seed, optimizer, adam_lr, switch_iter,
            record["loss"], record["l2re"],
        )
    return record


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run(
    pdes: Optional[List[str]] = None,
    widths: Optional[List[int]] = None,
    seeds: Optional[List[int]] = None,
    outdir: str = "results/loss_vs_l2re",
    total_iters: int = TOTAL_ITERS,
    log_every: int = 0,
    logger=None,
) -> Dict:
    """Run the loss-vs-L2RE sweep and produce Figure 2.

    For each (PDE, width, seed) we run Adam (best lr from the grid), L-BFGS, and
    Adam+L-BFGS (best switch point).  Every run contributes one (loss, L2RE)
    point to the scatter plot.
    """
    pdes = pdes or DEFAULT_PDES
    widths = widths or DEFAULT_WIDTHS
    seeds = seeds or DEFAULT_SEEDS

    ensure_dir(outdir)
    records: List[Dict] = []

    for pde_name in pdes:
        for width in widths:
            for seed in seeds:
                # --- Adam: pick best lr from the grid ---------------------
                best_adam = None
                for lr in ADAM_LR_GRID:
                    rec = _run_single(pde_name, width, seed, "adam",
                                      adam_lr=lr, total_iters=total_iters,
                                      log_every=log_every, logger=logger)
                    if best_adam is None or rec["loss"] < best_adam["loss"]:
                        best_adam = rec
                records.append(best_adam)

                # --- L-BFGS ----------------------------------------------
                rec = _run_single(pde_name, width, seed, "lbfgs",
                                  total_iters=total_iters,
                                  log_every=log_every, logger=logger)
                records.append(rec)

                # --- Adam+L-BFGS: pick best switch point -----------------
                best_switch = None
                for sw in SWITCH_GRID:
                    rec = _run_single(pde_name, width, seed, "adam_lbfgs",
                                      adam_lr=best_adam["adam_lr"],
                                      switch_iter=sw, total_iters=total_iters,
                                      log_every=log_every, logger=logger)
                    if best_switch is None or rec["loss"] < best_switch["loss"]:
                        best_switch = rec
                records.append(best_switch)

    # --- Summaries --------------------------------------------------------
    summary: Dict[str, Dict] = {}
    for pde_name in pdes:
        pde_recs = [r for r in records if r["pde"] == pde_name]
        summary[pde_name] = {
            "n_runs": len(pde_recs),
            "loss": _summarize([r["loss"] for r in pde_recs]),
            "l2re": _summarize([r["l2re"] for r in pde_recs]),
            "corr_loss_l2re": _correlation(
                [r["loss"] for r in pde_recs], [r["l2re"] for r in pde_recs]
            ),
        }

    payload = {"records": records, "summary": summary}
    save_json(payload, os.path.join(outdir, "results.json"))

    _plot_scatter(records, outdir, logger=logger)

    if logger is not None:
        logger.info("[loss_vs_l2re] wrote results to %s", outdir)
    return payload


def _summarize(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _correlation(xs: List[float], ys: List[float]) -> float:
    """Pearson correlation between loss and L2RE (log-space, robust)."""
    x = np.log10(np.asarray(xs, dtype=np.float64) + 1e-30)
    y = np.log10(np.asarray(ys, dtype=np.float64) + 1e-30)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _plot_scatter(records: List[Dict], outdir: str, logger=None) -> None:
    """Render Figure 2: loss vs L2RE scatter, one colour per PDE."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional
        if logger is not None:
            logger.warning("matplotlib unavailable, skipping figure: %s", exc)
        return

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    colors = {"convection": "tab:blue", "reaction": "tab:orange", "wave": "tab:green"}
    markers = {"adam": "o", "lbfgs": "s", "adam_lbfgs": "^"}

    for pde_name in sorted({r["pde"] for r in records}):
        for opt in ["adam", "lbfgs", "adam_lbfgs"]:
            pts = [r for r in records if r["pde"] == pde_name and r["optimizer"] == opt]
            if not pts:
                continue
            xs = [max(r["loss"], 1e-30) for r in pts]
            ys = [max(r["l2re"], 1e-30) for r in pts]
            ax.scatter(
                xs, ys,
                s=28,
                alpha=0.75,
                color=colors.get(pde_name, "gray"),
                marker=markers.get(opt, "o"),
                label=f"{pde_name} / {opt}",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training loss")
    ax.set_ylabel("L2 relative error")
    ax.set_title("Loss vs L2RE (Figure 2)")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()

    out_path = os.path.join(outdir, "figure2_loss_vs_l2re.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    if logger is not None:
        logger.info("[loss_vs_l2re] saved figure to %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Loss vs L2RE scatter experiment (Figure 2)."
    )
    parser.add_argument("--outdir", type=str, default="results/loss_vs_l2re")
    parser.add_argument("--pdes", type=str, nargs="+", default=DEFAULT_PDES)
    parser.add_argument("--widths", type=int, nargs="+", default=DEFAULT_WIDTHS)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--total-iters", type=int, default=TOTAL_ITERS)
    parser.add_argument("--log-every", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logger = get_logger("loss_vs_l2re")
    run(
        pdes=args.pdes,
        widths=args.widths,
        seeds=args.seeds,
        outdir=args.outdir,
        total_iters=args.total_iters,
        log_every=args.log_every,
        logger=logger,
    )


if __name__ == "__main__":
    main()
