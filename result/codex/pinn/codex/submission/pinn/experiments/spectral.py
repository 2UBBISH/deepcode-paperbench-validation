"""Spectral density of the Hessian and of the L-BFGS-preconditioned Hessian.

Reproduces Figures 3 and 7 of the paper (Section 5).  Per the addendum only
Adam+L-BFGS runs that switch at 11000 iterations are considered, and the
network/learning-rate/seed configuration is the one with the smallest L2RE.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..data import build_dataset
from ..hessian.lbfgs_precond import LBFGSHistory, LBFGSPreconditioner
from ..hessian.spectral import spectral_density
from ..hessian.hvp import hvp
from ..models import build_model
from ..optim.objective import Objective
from ..optim.trainers import TrainConfig, train
from ..plotting import plot_spectral_density, plot_spectral_density_panels
from ..problems import build_problem
from .common import Paths, RunSpec

COMPONENTS = ("residual", "ic", "bc")


def train_adam_lbfgs(
    pde: str,
    width: int,
    seed: int,
    lr: float,
    switch: int = 11000,
    iters: int = 41000,
    log_every: int = 500,
    progress: bool = True,
):
    """Train one Adam+L-BFGS model and keep its L-BFGS correction pairs."""
    problem = build_problem(pde)
    ds = build_dataset(problem, seed=seed)
    net = build_model(width=width, n_layers=3, seed=seed)
    obj = Objective(problem, net, ds)
    cfg = TrainConfig(optimizer="adam_lbfgs", lr=lr, switch_iter=switch, iters=iters, log_every=log_every)
    res = train(problem, net, ds, cfg, seed=seed, progress=progress)
    return problem, ds, net, obj, res


def hessian_spectrum(
    obj: Objective,
    component: Optional[str] = None,
    preconditioner: Optional[LBFGSPreconditioner] = None,
    n_runs: int = 1,
    n_iter: int = 100,
    seed: int = 0,
    verbose: bool = False,
) -> Dict[str, np.ndarray]:
    """Spectral density of ``H`` (or of ``H_k H`` if a preconditioner is given)."""
    if component is None:
        loss_fn = obj.loss
    else:
        loss_fn = lambda: obj.component_loss(component)  # noqa: E731

    def matvec(v):
        return hvp(loss_fn, obj.params, v)

    if preconditioner is None:
        mv, dim = matvec, obj.n_params
    else:
        mv, dim = (lambda v: preconditioner.matvec(v, matvec)), preconditioner.dim
    grids, density, eigenvalues = spectral_density(
        mv, dim, n_runs=n_runs, n_iter=n_iter, seed=seed, verbose=verbose
    )
    return {
        "grids": grids,
        "density": density,
        "eigenvalues": eigenvalues,
        "max_eigenvalue": float(np.max(eigenvalues)),
        "min_eigenvalue": float(np.min(eigenvalues)),
    }


def run_spectral_experiment(
    pde: str,
    width: int,
    seed: int,
    lr: float,
    outdir: Paths,
    switch: int = 11000,
    iters: int = 41000,
    n_runs: int = 1,
    n_iter: int = 100,
    components: bool = True,
    progress: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Compute every spectral density appearing in Figures 3 and 7 for one PDE."""
    outdir.makedirs()
    problem, ds, net, obj, res = train_adam_lbfgs(
        pde, width, seed, lr, switch=switch, iters=iters, progress=progress
    )
    history = res.lbfgs_history
    pre = LBFGSPreconditioner(history) if history is not None else None

    curves: Dict[str, Dict[str, np.ndarray]] = {}
    keys: List[Tuple[str, Optional[str]]] = [("full", None)]
    if components:
        keys += [(c, c) for c in COMPONENTS]
    for name, comp in keys:
        curves[f"{name}:hessian"] = hessian_spectrum(
            obj, comp, None, n_runs=n_runs, n_iter=n_iter, seed=seed, verbose=progress
        )
        if pre is not None:
            curves[f"{name}:precond"] = hessian_spectrum(
                obj, comp, pre, n_runs=n_runs, n_iter=n_iter, seed=seed + 1, verbose=progress
            )

    # ---- persist raw spectra ---------------------------------------- #
    npz_path = os.path.join(outdir.tables, f"spectra_{pde}.npz")
    np.savez(
        npz_path,
        **{
            f"{k}__{field}": value
            for k, spec in curves.items()
            for field, value in spec.items()
        },
    )
    summary = {
        k: {
            "max_eigenvalue": spec["max_eigenvalue"],
            "min_eigenvalue": spec["min_eigenvalue"],
        }
        for k, spec in curves.items()
    }
    summary["run"] = {
        "pde": pde,
        "width": width,
        "seed": seed,
        "lr": lr,
        "switch": switch,
        "iters": iters,
        "final_loss": res.final_loss,
        "final_l2re": res.final_l2re,
        "conditioning_improvement": (
            summary["full:hessian"]["max_eigenvalue"] / summary["full:precond"]["max_eigenvalue"]
            if "full:precond" in summary
            else None
        ),
    }
    with open(os.path.join(outdir.tables, f"spectra_{pde}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # ---- figures ----------------------------------------------------- #
    top = {
        "Hessian": curves["full:hessian"]["max_eigenvalue"],
    }
    if "full:precond" in curves:
        top["preconditioned"] = curves["full:precond"]["max_eigenvalue"]
    plot_spectral_density(
        {
            "Hessian": (curves["full:hessian"]["grids"], curves["full:hessian"]["density"]),
            **(
                {
                    "preconditioned Hessian": (
                        curves["full:precond"]["grids"],
                        curves["full:precond"]["density"],
                    )
                }
                if "full:precond" in curves
                else {}
            ),
        },
        f"{pde}: full PINN loss",
        outdir.figure(f"figure3_top_{pde}.png"),
        top_eigenvalues=top,
    )
    if components:
        panels = {}
        for comp in COMPONENTS:
            h = curves[f"{comp}:hessian"]
            p = curves.get(f"{comp}:precond")
            panels[comp] = {"Hessian": (h["grids"], h["density"])}
            if p is not None:
                panels[comp]["preconditioned Hessian"] = (p["grids"], p["density"])
        name = "figure3_bottom_convection.png" if pde == "convection" else f"figure7_{pde}.png"
        plot_spectral_density_panels(
            panels,
            f"{pde}: Hessian of each loss component after Adam+L-BFGS",
            outdir.figure(name),
        )
    return curves
