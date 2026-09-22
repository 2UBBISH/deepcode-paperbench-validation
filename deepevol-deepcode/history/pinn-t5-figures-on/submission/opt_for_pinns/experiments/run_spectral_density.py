"""Spectral density experiment (Figures 3 & 7).

Reproduces Section 5 of "Challenges in Training PINNs: A Loss Landscape
Perspective".  For each PDE we train a PINN with Adam+L-BFGS (switching at
iteration 11000, using the per-PDE best config: width=200 and the Adam lr that
yields the smallest L2RE), then estimate the spectral density of

    * the Hessian of the total loss (solid curves), and
    * the L-BFGS-preconditioned Hessian (dashed curves),

using Stochastic Lanczos Quadrature (SLQ).  We additionally report the spectral
density of each individual loss component (residual / IC / BC) for the
convection problem (Figure 3 bottom) and for reaction / wave (Figure 7).

Expected qualitative results:
    * Large outlier eigenvalues (>1e4 convection, >1e3 reaction, >1e5 wave)
      with the bulk of the density near zero -> ill-conditioned Hessian.
    * The residual component is the most ill-conditioned.
    * L-BFGS preconditioning reduces the top eigenvalue by >= 1e3.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

# Allow running as a script from the repository root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data import build_data  # noqa: E402
from src.hessian import (  # noqa: E402
    flatten_params,
    make_hvp,
    slq_spectral_density,
    unflatten_params,
)
from src.lbfgs_precond import build_preconditioned_matvec  # noqa: E402
from src.loss import bc_loss, ic_loss, pinn_loss, residual_loss  # noqa: E402
from src.model import build_model  # noqa: E402
from src.pdes import get_pde  # noqa: E402
from src.train import train_adam_lbfgs  # noqa: E402
from src.utils import ensure_dir, get_logger, save_json, set_seed  # noqa: E402


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_PDES = ["convection", "reaction", "wave"]
DEFAULT_SEEDS = [123, 234, 345, 456, 567]

# Per-PDE best configs from Section 6.1 (width=200, switch at 11000 iters).
BEST_CONFIG: Dict[str, Dict[str, object]] = {
    "convection": {"width": 200, "adam_lr": 1e-4, "switch_iter": 11000, "seed": 345},
    "reaction": {"width": 200, "adam_lr": 1e-3, "switch_iter": 11000, "seed": 456},
    "wave": {"width": 200, "adam_lr": 1e-3, "switch_iter": 11000, "seed": 567},
}

TOTAL_ITERS = 41000
SWITCH_ITER = 11000

# SLQ settings.
SLQ_ITERS = 100
SLQ_PROBES = 1
SLQ_BINS = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_component_loss_fns(pde, model, data) -> Dict[str, Callable[[], torch.Tensor]]:
    """Return zero-arg loss callables for total / residual / ic / bc terms."""

    def total():
        return pinn_loss(
            pde, model, data.x_res, data.t_res, data.x_ic, data.t_ic, data.x_bc, data.t_bc
        )

    def residual():
        return residual_loss(pde, model, data.x_res, data.t_res)

    def ic():
        return ic_loss(pde, model, data.x_ic, data.t_ic)

    def bc():
        return bc_loss(pde, model, data.x_bc, data.t_bc)

    return {"total": total, "residual": residual, "ic": ic, "bc": bc}


def _collect_lbfgs_curvature(model, data, pde, switch_iter: int, total_iters: int):
    """Run Adam+L-BFGS and capture the final L-BFGS curvature pairs.

    We re-run the L-BFGS phase with a custom loop so that we can record the
    ``(s_i, y_i, rho_i)`` pairs produced by the strong-Wolfe line search.  The
    returned lists are used to build the L-BFGS preconditioner (Algorithms 2/3).
    """
    import torch.optim as optim

    params = [p for p in model.parameters() if p.requires_grad]
    s_list: List[torch.Tensor] = []
    y_list: List[torch.Tensor] = []
    rho_list: List[float] = []

    prev_params = flatten_params(params).detach().clone()
    prev_grad: Optional[torch.Tensor] = None

    optimizer = optim.LBFGS(
        params,
        lr=1.0,
        max_iter=1,
        max_eval=1,
        history_size=100,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        loss = pinn_loss(
            pde, model, data.x_res, data.t_res, data.x_ic, data.t_ic, data.x_bc, data.t_bc
        )
        loss.backward()
        return loss

    n_lbfgs = max(0, total_iters - switch_iter)
    for _ in range(n_lbfgs):
        optimizer.step(closure)
        cur_params = flatten_params(params).detach().clone()
        cur_grad = torch.cat(
            [p.grad.detach().reshape(-1) for p in params if p.grad is not None]
        ).clone()
        if prev_grad is not None:
            s = (cur_params - prev_params).to(torch.float64)
            y = (cur_grad - prev_grad).to(torch.float64)
            sy = torch.dot(s, y)
            if sy > 1e-12:
                s_list.append(s)
                y_list.append(y)
                rho_list.append(float(1.0 / sy))
                if len(s_list) > 100:
                    s_list.pop(0)
                    y_list.pop(0)
                    rho_list.pop(0)
        prev_params = cur_params
        prev_grad = cur_grad

    return s_list, y_list, rho_list


def _spectral_density_for_loss(
    loss_fn: Callable[[], torch.Tensor],
    params,
    p: int,
    dtype: torch.dtype,
    device: torch.device,
    grid: Optional[torch.Tensor] = None,
) -> Dict[str, object]:
    """Estimate the spectral density of the Hessian of ``loss_fn`` via SLQ."""
    hvp = make_hvp(loss_fn, params)
    out = slq_spectral_density(
        hvp,
        p,
        n_iters=SLQ_ITERS,
        n_probes=SLQ_PROBES,
        n_bins=SLQ_BINS,
        grid=grid,
        dtype=dtype,
        device=device,
    )
    return {
        "grid": out["grid"].detach().cpu().numpy().tolist(),
        "density": out["density"].detach().cpu().numpy().tolist(),
        "lambda_max": float(out["lambda_max"]),
        "lambda_min": float(out["lambda_min"]),
    }


def _preconditioned_spectral_density(
    loss_fn: Callable[[], torch.Tensor],
    params,
    p: int,
    s_list,
    y_list,
    rho_list,
    dtype: torch.dtype,
    device: torch.device,
    grid: Optional[torch.Tensor] = None,
) -> Dict[str, object]:
    """Estimate the spectral density of the L-BFGS-preconditioned Hessian."""
    hvp = make_hvp(loss_fn, params)
    if len(s_list) == 0:
        return {"grid": [], "density": [], "lambda_max": float("nan"), "lambda_min": float("nan")}

    precond_matvec = build_preconditioned_matvec(
        hvp, s_list, y_list, rho_list, dtype=dtype, device=device
    )
    # The preconditioned operator lives in the augmented space of dimension p+m.
    m = len(s_list)
    out = slq_spectral_density(
        precond_matvec,
        p + m,
        n_iters=SLQ_ITERS,
        n_probes=SLQ_PROBES,
        n_bins=SLQ_BINS,
        grid=grid,
        dtype=dtype,
        device=device,
    )
    return {
        "grid": out["grid"].detach().cpu().numpy().tolist(),
        "density": out["density"].detach().cpu().numpy().tolist(),
        "lambda_max": float(out["lambda_max"]),
        "lambda_min": float(out["lambda_min"]),
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def run(
    pdes: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    outdir: str = "results/spectral_density",
    total_iters: int = TOTAL_ITERS,
    log_every: int = 0,
    logger=None,
) -> Dict:
    """Run the spectral-density experiment for the requested PDEs."""
    pdes = pdes or DEFAULT_PDES
    seeds = seeds or DEFAULT_SEEDS
    logger = logger or get_logger("spectral_density")
    ensure_dir(outdir)

    dtype = torch.float64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    records: List[Dict] = []
    summary: Dict[str, Dict] = {}

    for pde_name in pdes:
        cfg = BEST_CONFIG.get(pde_name, {"width": 200, "adam_lr": 1e-3, "switch_iter": SWITCH_ITER, "seed": seeds[0]})
        width = int(cfg["width"])
        adam_lr = float(cfg["adam_lr"])
        switch_iter = int(cfg["switch_iter"])
        seed = int(cfg["seed"])

        logger.info(
            "=== %s: width=%d adam_lr=%g switch=%d seed=%d ===",
            pde_name, width, adam_lr, switch_iter, seed,
        )

        set_seed(seed)
        pde = get_pde(pde_name)
        model = build_model(width=width, depth=3, seed=seed, dtype=torch.float32, device=device)
        data = build_data(pde, seed=seed, dtype=torch.float32)
        data = data.to(device=device, dtype=torch.float32)

        # --- Train with Adam + L-BFGS (switch at 11000) -------------------
        res = train_adam_lbfgs(
            pde, model, data,
            adam_lr=adam_lr,
            switch_iter=switch_iter,
            total_iters=total_iters,
            log_every=log_every,
            verbose=False,
        )
        logger.info(
            "  trained: loss=%.4e l2re=%.4e grad_norm=%.4e",
            res.loss, res.l2re, res.grad_norm,
        )

        # --- Collect L-BFGS curvature pairs for the preconditioner --------
        # Re-run the L-BFGS phase to capture (s, y, rho) pairs.
        set_seed(seed)
        model2 = build_model(width=width, depth=3, seed=seed, dtype=torch.float32, device=device)
        data2 = build_data(pde, seed=seed, dtype=torch.float32).to(device=device, dtype=torch.float32)
        # Warm-start model2 from the trained model.
        model2.load_state_dict(model.state_dict())
        s_list, y_list, rho_list = _collect_lbfgs_curvature(
            model2, data2, pde, switch_iter=switch_iter, total_iters=total_iters
        )
        logger.info("  collected %d L-BFGS curvature pairs", len(s_list))

        # --- Move to float64 for Hessian / SLQ ----------------------------
        model64 = build_model(width=width, depth=3, seed=seed, dtype=dtype, device=device)
        model64.load_state_dict({k: v.to(dtype) for k, v in model.state_dict().items()})
        data64 = build_data(pde, seed=seed, dtype=dtype).to(device=device, dtype=dtype)

        params = [p for p in model64.parameters() if p.requires_grad]
        p_dim = int(sum(p.numel() for p in params))

        loss_fns = _make_component_loss_fns(pde, model64, data64)

        # Common grid for all densities of this PDE (log-spaced).
        grid = torch.logspace(-2, 8, SLQ_BINS, dtype=dtype, device=device)

        # Total-loss Hessian spectral density (solid).
        total_density = _spectral_density_for_loss(
            loss_fns["total"], params, p_dim, dtype, device, grid=grid
        )
        # Total-loss preconditioned Hessian spectral density (dashed).
        total_precond = _preconditioned_spectral_density(
            loss_fns["total"], params, p_dim, s_list, y_list, rho_list, dtype, device, grid=grid
        )

        # Per-component Hessian spectral densities.
        comp_densities: Dict[str, Dict] = {}
        for comp_name in ("residual", "ic", "bc"):
            comp_densities[comp_name] = _spectral_density_for_loss(
                loss_fns[comp_name], params, p_dim, dtype, device, grid=grid
            )

        record = {
            "pde": pde_name,
            "width": width,
            "adam_lr": adam_lr,
            "switch_iter": switch_iter,
            "seed": seed,
            "loss": float(res.loss),
            "l2re": float(res.l2re),
            "grad_norm": float(res.grad_norm),
            "n_params": p_dim,
            "n_curvature_pairs": len(s_list),
            "total_hessian": total_density,
            "total_preconditioned": total_precond,
            "components": comp_densities,
        }
        records.append(record)

        summary[pde_name] = {
            "hessian_lambda_max": total_density["lambda_max"],
            "hessian_lambda_min": total_density["lambda_min"],
            "precond_lambda_max": total_precond["lambda_max"],
            "precond_lambda_min": total_precond["lambda_min"],
            "reduction_factor": (
                total_density["lambda_max"] / total_precond["lambda_max"]
                if total_precond["lambda_max"] and np.isfinite(total_precond["lambda_max"])
                else float("nan")
            ),
            "residual_lambda_max": comp_densities["residual"]["lambda_max"],
            "ic_lambda_max": comp_densities["ic"]["lambda_max"],
            "bc_lambda_max": comp_densities["bc"]["lambda_max"],
        }
        logger.info(
            "  Hessian lambda_max=%.4e lambda_min=%.4e | precond lambda_max=%.4e (reduction %.2fx)",
            summary[pde_name]["hessian_lambda_max"],
            summary[pde_name]["hessian_lambda_min"],
            summary[pde_name]["precond_lambda_max"],
            summary[pde_name]["reduction_factor"],
        )

    payload = {"records": records, "summary": summary}
    save_json(payload, os.path.join(outdir, "results.json"))

    _plot_spectral_density(records, outdir, logger=logger)
    _plot_component_density(records, outdir, logger=logger)

    return payload


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_spectral_density(records, outdir, logger=None):
    """Figure 3 (top) / Figure 7: total-loss Hessian vs preconditioned density."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        if logger:
            logger.warning("matplotlib unavailable, skipping plots: %s", exc)
        return

    n = len(records)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for ax, rec in zip(axes[0], records):
        h = rec["total_hessian"]
        pc = rec["total_preconditioned"]
        if h["grid"]:
            ax.plot(h["grid"], h["density"], "-", label="Hessian", color="C0")
        if pc["grid"]:
            ax.plot(pc["grid"], pc["density"], "--", label="L-BFGS precond.", color="C1")
        ax.set_xscale("log")
        ax.set_xlabel("eigenvalue")
        ax.set_ylabel("spectral density")
        ax.set_title(rec["pde"])
        ax.legend()
    fig.tight_layout()
    path = os.path.join(outdir, "figure3_spectral_density.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    if logger:
        logger.info("wrote %s", path)


def _plot_component_density(records, outdir, logger=None):
    """Figure 3 (bottom) / Figure 7: per-component Hessian spectral density."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        if logger:
            logger.warning("matplotlib unavailable, skipping plots: %s", exc)
        return

    n = len(records)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for ax, rec in zip(axes[0], records):
        for comp_name, color in (("residual", "C0"), ("ic", "C2"), ("bc", "C3")):
            d = rec["components"].get(comp_name)
            if d and d["grid"]:
                ax.plot(d["grid"], d["density"], "-", label=comp_name, color=color)
        ax.set_xscale("log")
        ax.set_xlabel("eigenvalue")
        ax.set_ylabel("spectral density")
        ax.set_title(rec["pde"])
        ax.legend()
    fig.tight_layout()
    path = os.path.join(outdir, "figure7_component_density.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    if logger:
        logger.info("wrote %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="PINN Hessian spectral density (Figures 3 & 7)")
    parser.add_argument("--outdir", type=str, default="results/spectral_density")
    parser.add_argument("--pdes", type=str, nargs="+", default=DEFAULT_PDES)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--total-iters", type=int, default=TOTAL_ITERS)
    parser.add_argument("--log-every", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logger = get_logger("spectral_density")
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
