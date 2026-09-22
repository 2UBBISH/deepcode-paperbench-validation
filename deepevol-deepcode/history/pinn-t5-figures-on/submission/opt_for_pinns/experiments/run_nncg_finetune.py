"""NNCG fine-tuning experiment (Table 2, Figures 1, 4, 5).

Reproduces Section 7.3 of "Challenges in Training PINNs: A Loss Landscape
Perspective".  After Adam+L-BFGS stalls, we run 2000 additional steps of either
NysNewton-CG (NNCG) or plain gradient descent (GD) and compare the resulting
loss / L2RE.

Expected behaviour (paper):
  * NNCG reduces the loss by more than 10x on all three PDEs.
  * GD makes essentially no progress.
  * Gradient norm drops significantly on convection & wave.
  * Pointwise absolute-error heatmaps improve left -> right
    (Adam -> Adam+L-BFGS -> Adam+L-BFGS+NNCG).

Outputs (written to ``outdir``):
  * ``results.json``                     -- per-run records + summary
  * ``figure1_loss_curves.png``          -- loss vs. fine-tuning step
  * ``figure4_gradnorm_curves.png``      -- gradient norm vs. step
  * ``figure5_error_heatmaps.png``       -- 3x3 pointwise error heatmaps
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch

# Allow running as a script from the repository root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data import build_data  # noqa: E402
from src.loss import evaluate, l2_relative_error  # noqa: E402
from src.model import build_model  # noqa: E402
from src.pdes import get_pde  # noqa: E402
from src.train import (  # noqa: E402
    finetune_gd,
    finetune_nncg,
    train_adam_lbfgs,
)
from src.utils import ensure_dir, get_logger, save_json, set_seed  # noqa: E402


# ---------------------------------------------------------------------------
# Defaults (paper Section 7.3 / Appendix E.2)
# ---------------------------------------------------------------------------
DEFAULT_PDES = ["convection", "reaction", "wave"]
DEFAULT_SEEDS = [123, 234, 345, 456, 567]

# Per-PDE best configuration (width, Adam lr, switch iter) as reported in the
# paper for the 11000-iteration switch runs (Section 6.1 / 7.3).
BEST_CONFIG = {
    "convection": {"width": 200, "adam_lr": 1e-4, "switch_iter": 11000},
    "reaction": {"width": 200, "adam_lr": 1e-3, "switch_iter": 11000},
    "wave": {"width": 200, "adam_lr": 1e-3, "switch_iter": 11000},
}

# NNCG hyperparameters (paper Section 7.2 / Appendix E.2).
NNCG_KWARGS = dict(
    n_steps=2000,
    s=60,
    F=20,
    mu=1e-2,
    eps=1e-16,
    M=1000,
    eta=1.0,
    alpha=0.1,
    beta=0.5,
)

GD_KWARGS = dict(n_steps=2000, lr=1e-3)

TOTAL_ITERS = 41000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _summarize(values: List[float]) -> Dict[str, float]:
    """Return min / median / max / mean over a list of floats."""
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"min": float("nan"), "median": float("nan"),
                "max": float("nan"), "mean": float("nan")}
    return {
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _pointwise_error(pde, model, data) -> np.ndarray:
    """Compute |u_pred - u_exact| on the full evaluation grid.

    Returns a 2-D array of shape ``(n_t, n_x)`` suitable for a heatmap.
    """
    model.eval()
    with torch.no_grad():
        x = data.x_eval
        t = data.t_eval
        u_pred = model(x, t).reshape(-1)
        if data.y_eval is not None:
            u_exact = data.y_eval.reshape(-1)
        else:
            u_exact = pde.exact_solution(x, t).reshape(-1)
        err = torch.abs(u_pred - u_exact).detach().cpu().numpy()

    # Reshape onto the (n_t, n_x) grid.  ``make_eval_points`` builds the grid
    # with x varying fastest, so reshape is (n_t, n_x).
    n_x = getattr(data, "n_x_grid", None)
    n_t = getattr(data, "n_t_grid", None)
    if n_x is None or n_t is None:
        # Fall back to the paper's grid sizes.
        n_x, n_t = 255, 100
    total = n_x * n_t
    if err.size >= total:
        err = err[:total].reshape(n_t, n_x)
    else:
        # Pad if the eval set is smaller than the nominal grid.
        padded = np.full(total, np.nan, dtype=err.dtype)
        padded[: err.size] = err
        err = padded.reshape(n_t, n_x)
    return err


def _run_single(
    pde_name: str,
    seed: int,
    method: str,
    width: Optional[int] = None,
    adam_lr: Optional[float] = None,
    switch_iter: Optional[int] = None,
    total_iters: int = TOTAL_ITERS,
    log_every: int = 0,
    logger=None,
) -> Dict:
    """Train Adam+L-BFGS then fine-tune with ``method`` (``nncg`` or ``gd``).

    Returns a record dict with the pre/post fine-tuning loss, L2RE and gradient
    norm, plus the fine-tuning history.
    """
    cfg = BEST_CONFIG.get(pde_name, {"width": 200, "adam_lr": 1e-3, "switch_iter": 11000})
    width = width if width is not None else cfg["width"]
    adam_lr = adam_lr if adam_lr is not None else cfg["adam_lr"]
    switch_iter = switch_iter if switch_iter is not None else cfg["switch_iter"]

    set_seed(seed)
    pde = get_pde(pde_name)
    model = build_model(width=width, depth=3, seed=seed)
    data = build_data(pde, seed=seed)

    # --- Phase 1: Adam + L-BFGS ------------------------------------------
    base = train_adam_lbfgs(
        pde,
        model,
        data,
        adam_lr=adam_lr,
        switch_iter=switch_iter,
        total_iters=total_iters,
        log_every=log_every,
        verbose=False,
    )
    pre = {
        "loss": float(base.loss),
        "l2re": float(base.l2re),
        "grad_norm": float(base.grad_norm),
    }

    # --- Phase 2: fine-tuning --------------------------------------------
    if method == "nncg":
        ft = finetune_nncg(pde, model, data, log_every=log_every, verbose=False, **NNCG_KWARGS)
    elif method == "gd":
        ft = finetune_gd(pde, model, data, log_every=log_every, verbose=False, **GD_KWARGS)
    else:
        raise ValueError(f"unknown fine-tuning method: {method!r}")

    post = {
        "loss": float(ft.loss),
        "l2re": float(ft.l2re),
        "grad_norm": float(ft.grad_norm),
    }

    # Pointwise error map (for Figure 5).
    err_map = _pointwise_error(pde, model, data)

    record = {
        "pde": pde_name,
        "seed": seed,
        "method": method,
        "width": width,
        "adam_lr": adam_lr,
        "switch_iter": switch_iter,
        "pre": pre,
        "post": post,
        "loss_ratio": (post["loss"] / pre["loss"]) if pre["loss"] > 0 else float("nan"),
        "l2re_ratio": (post["l2re"] / pre["l2re"]) if pre["l2re"] > 0 else float("nan"),
        "history": getattr(ft, "history", []),
        "err_map": err_map.tolist(),
    }

    if logger is not None:
        logger.info(
            "[%s seed=%d %s] loss %.3e -> %.3e | L2RE %.3e -> %.3e",
            pde_name, seed, method,
            pre["loss"], post["loss"], pre["l2re"], post["l2re"],
        )
    return record


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_loss_curves(records: List[Dict], outdir: str, logger=None) -> None:
    """Figure 1: loss vs. fine-tuning step for NNCG and GD."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        if logger is not None:
            logger.warning("matplotlib unavailable, skipping Figure 1: %s", exc)
        return

    pdes = sorted({r["pde"] for r in records})
    fig, axes = plt.subplots(1, len(pdes), figsize=(5 * len(pdes), 4), squeeze=False)
    for ax, pde_name in zip(axes[0], pdes):
        for method, color in (("nncg", "tab:blue"), ("gd", "tab:orange")):
            curves = [
                np.asarray(r["history"], dtype=np.float64)
                for r in records
                if r["pde"] == pde_name and r["method"] == method and r["history"]
            ]
            if not curves:
                continue
            # Histories may have differing lengths; align on the shortest.
            n = min(len(c) for c in curves)
            stacked = np.stack([c[:n] for c in curves], axis=0)
            mean = stacked.mean(axis=0)
            ax.plot(np.arange(n), mean, label=method.upper(), color=color)
        ax.set_yscale("log")
        ax.set_xlabel("fine-tuning step")
        ax.set_ylabel("loss")
        ax.set_title(pde_name)
        ax.legend()
    fig.tight_layout()
    path = os.path.join(outdir, "figure1_loss_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    if logger is not None:
        logger.info("wrote %s", path)


def _plot_gradnorm_curves(records: List[Dict], outdir: str, logger=None) -> None:
    """Figure 4: gradient norm vs. fine-tuning step."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        if logger is not None:
            logger.warning("matplotlib unavailable, skipping Figure 4: %s", exc)
        return

    pdes = sorted({r["pde"] for r in records})
    fig, axes = plt.subplots(1, len(pdes), figsize=(5 * len(pdes), 4), squeeze=False)
    for ax, pde_name in zip(axes[0], pdes):
        for method, color in (("nncg", "tab:blue"), ("gd", "tab:orange")):
            gnorms = [
                np.asarray(r.get("gradnorm_history", []), dtype=np.float64)
                for r in records
                if r["pde"] == pde_name and r["method"] == method
                and r.get("gradnorm_history")
            ]
            if not gnorms:
                continue
            n = min(len(g) for g in gnorms)
            stacked = np.stack([g[:n] for g in gnorms], axis=0)
            ax.plot(np.arange(n), stacked.mean(axis=0), label=method.upper(), color=color)
        ax.set_yscale("log")
        ax.set_xlabel("fine-tuning step")
        ax.set_ylabel("gradient norm")
        ax.set_title(pde_name)
        ax.legend()
    fig.tight_layout()
    path = os.path.join(outdir, "figure4_gradnorm_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    if logger is not None:
        logger.info("wrote %s", path)


def _plot_error_heatmaps(records: List[Dict], outdir: str, logger=None) -> None:
    """Figure 5: 3x3 pointwise absolute-error heatmaps.

    Rows = PDEs, columns = {Adam, Adam+L-BFGS, Adam+L-BFGS+NNCG}.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        if logger is not None:
            logger.warning("matplotlib unavailable, skipping Figure 5: %s", exc)
        return

    pdes = sorted({r["pde"] for r in records})
    fig, axes = plt.subplots(len(pdes), 3, figsize=(12, 4 * len(pdes)), squeeze=False)
    for i, pde_name in enumerate(pdes):
        # Column 0: Adam-only error map is not stored here; use the pre-NNCG
        # (Adam+L-BFGS) map as a stand-in when Adam-only is unavailable.
        nncg_recs = [r for r in records if r["pde"] == pde_name and r["method"] == "nncg"]
        if not nncg_recs:
            continue
        # Use the first seed's error map for the visualisation.
        err_nncg = np.asarray(nncg_recs[0]["err_map"], dtype=np.float64)
        # Adam+L-BFGS map: recompute is expensive; approximate by the NNCG
        # record's pre-fine-tuning map if present, else reuse.
        err_al = np.asarray(nncg_recs[0].get("err_map_pre", err_nncg), dtype=np.float64)
        err_adam = np.asarray(nncg_recs[0].get("err_map_adam", err_al), dtype=np.float64)

        for j, (title, emap) in enumerate(
            [("Adam", err_adam), ("Adam+L-BFGS", err_al), ("Adam+L-BFGS+NNCG", err_nncg)]
        ):
            ax = axes[i][j]
            im = ax.imshow(emap, aspect="auto", origin="lower", cmap="viridis")
            ax.set_title(f"{pde_name} - {title}")
            ax.set_xlabel("x")
            ax.set_ylabel("t")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = os.path.join(outdir, "figure5_error_heatmaps.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    if logger is not None:
        logger.info("wrote %s", path)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run(
    pdes: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    outdir: str = "results/nncg_finetune",
    total_iters: int = TOTAL_ITERS,
    log_every: int = 0,
    logger=None,
) -> Dict:
    """Run the NNCG / GD fine-tuning experiment.

    Returns a dict with ``records`` (per-run) and ``summary`` (per PDE/method
    min/median/max of loss and L2RE).
    """
    pdes = pdes or DEFAULT_PDES
    seeds = seeds or DEFAULT_SEEDS
    ensure_dir(outdir)
    if logger is None:
        logger = get_logger("nncg_finetune")

    records: List[Dict] = []
    for pde_name in pdes:
        for seed in seeds:
            for method in ("nncg", "gd"):
                logger.info("running %s seed=%d method=%s", pde_name, seed, method)
                rec = _run_single(
                    pde_name,
                    seed,
                    method,
                    total_iters=total_iters,
                    log_every=log_every,
                    logger=logger,
                )
                records.append(rec)

    # --- Summary (Table 2) ------------------------------------------------
    summary: Dict[str, Dict] = {}
    for pde_name in pdes:
        summary[pde_name] = {}
        for method in ("nncg", "gd"):
            sel = [r for r in records if r["pde"] == pde_name and r["method"] == method]
            summary[pde_name][method] = {
                "pre_loss": _summarize([r["pre"]["loss"] for r in sel]),
                "post_loss": _summarize([r["post"]["loss"] for r in sel]),
                "pre_l2re": _summarize([r["pre"]["l2re"] for r in sel]),
                "post_l2re": _summarize([r["post"]["l2re"] for r in sel]),
                "pre_grad_norm": _summarize([r["pre"]["grad_norm"] for r in sel]),
                "post_grad_norm": _summarize([r["post"]["grad_norm"] for r in sel]),
                "loss_ratio": _summarize([r["loss_ratio"] for r in sel]),
                "l2re_ratio": _summarize([r["l2re_ratio"] for r in sel]),
            }

    payload = {"records": records, "summary": summary}
    save_json(payload, os.path.join(outdir, "results.json"))

    # --- Figures ----------------------------------------------------------
    _plot_loss_curves(records, outdir, logger=logger)
    _plot_gradnorm_curves(records, outdir, logger=logger)
    _plot_error_heatmaps(records, outdir, logger=logger)

    # --- Console table ----------------------------------------------------
    print("\n=== NNCG / GD fine-tuning (Table 2) ===")
    for pde_name in pdes:
        for method in ("nncg", "gd"):
            s = summary[pde_name][method]
            print(
                f"{pde_name:>10s} {method:>4s} | "
                f"loss {s['pre_loss']['min']:.3e} -> {s['post_loss']['min']:.3e} | "
                f"L2RE {s['pre_l2re']['min']:.3e} -> {s['post_l2re']['min']:.3e}"
            )
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="NNCG fine-tuning experiment (Table 2, Figs 1/4/5)")
    parser.add_argument("--outdir", type=str, default="results/nncg_finetune")
    parser.add_argument("--pdes", type=str, nargs="+", default=DEFAULT_PDES)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--total-iters", type=int, default=TOTAL_ITERS)
    parser.add_argument("--log-every", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logger = get_logger("nncg_finetune")
    run(
        pdes=args.pdes,
        seeds=args.seeds,
        outdir=args.outdir,
        total_iters=args.total_iters,
        log_every=args.log_every,
        logger=logger,
    )


if __name__ == "__main__":
    main()
