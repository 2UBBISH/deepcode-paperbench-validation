"""Fine-tuning Adam+L-BFGS with NNCG or gradient descent (Section 7).

Reproduces Figure 1 (loss curve on the wave problem), Figure 4 (loss and
gradient norms after the Adam+L-BFGS run), Figure 5 (pointwise absolute errors
at the optimiser switch points), Table 2 (loss/L2RE after fine-tuning) and
Table 3 (per-iteration wall-clock times).

"In section 7.3, training was continued for an additional 2000 steps for each
of the GD and NNCG optimizers." (addendum)
"""

from __future__ import annotations

import copy
import csv
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..data import build_dataset
from ..metrics import l2_relative_error
from ..models import build_model
from ..optim.nncg import NNCG, NNCGConfig, gd_finetune
from ..optim.objective import Objective
from ..optim.trainers import TrainConfig, train
from ..plotting import (
    plot_error_heatmaps,
    plot_optimizer_curves,
    plot_phase_curves,
    plot_underoptimization,
)
from ..problems import build_problem
from .common import Paths

MU_GRID: Sequence[float] = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)


def _state_copy(net) -> dict:
    return {k: v.detach().clone() for k, v in net.state_dict().items()}


@torch.no_grad()
def _error_grid(net, problem, nx: int = 255, nt: int = 100) -> np.ndarray:
    xy = problem.interior_grid()
    dtype = next(net.parameters()).dtype
    pred = net(xy.to(dtype)).reshape(nx, nt)
    true = problem.exact(xy.to(dtype)).reshape(nx, nt)
    return (pred - true).abs().numpy()


def run_finetune_experiment(
    pde: str,
    width: int,
    seed: int,
    lr: float,
    outdir: Paths,
    switch: int = 11000,
    iters: int = 41000,
    nncg_iters: int = 2000,
    gd_iters: int = 2000,
    gd_lr: Optional[float] = None,
    mu_grid: Sequence[float] = MU_GRID,
    nncg_cfg: Optional[dict] = None,
    progress: bool = True,
) -> dict:
    outdir.makedirs()
    problem = build_problem(pde)
    ds = build_dataset(problem, seed=seed)
    net = build_model(width=width, n_layers=3, seed=seed)
    cfg = TrainConfig(
        optimizer="adam_lbfgs",
        lr=lr,
        switch_iter=switch,
        iters=iters,
        log_every=max(iters // 40, 1),
        snapshot_iters=(switch,),
    )
    res = train(problem, net, ds, cfg, seed=seed, progress=progress)
    state_after_adam_lbfgs = _state_copy(net)
    state_after_adam = res.snapshots.get(switch)

    baseline = {
        "loss": res.final_loss,
        "l2re": res.final_l2re,
        "grad_norm": res.final_grad_norm,
    }
    lbfgs_seconds_per_iter = res.wall_clock / max(res.lbfgs_steps, 1)

    # ---- NNCG, tuned over the damping parameter mu ------------------- #
    nncg_cfg = dict(nncg_cfg or {})
    results: List[dict] = []
    best: Optional[dict] = None
    for mu in mu_grid:
        net.load_state_dict(state_after_adam_lbfgs)
        obj = Objective(problem, net, ds)
        config = NNCGConfig(iters=nncg_iters, mu=mu, **nncg_cfg)
        nncg_res = NNCG(obj, config, problem=problem).run(progress=progress)
        row = {
            "pde": pde,
            "width": width,
            "seed": seed,
            "lr": lr,
            "optimizer": "Adam+L-BFGS+NNCG",
            "mu": mu,
            "loss": nncg_res.final_loss,
            "l2re": nncg_res.final_l2re,
            "grad_norm": nncg_res.final_grad_norm,
            "seconds_per_iteration": nncg_res.seconds_per_iteration,
            "wall_clock": nncg_res.wall_clock,
            "mean_cg_iterations": float(np.mean(nncg_res.cg_iterations)),
        }
        results.append(row)
        if best is None or row["loss"] < best["row"]["loss"]:
            best = {"row": row, "res": nncg_res, "state": _state_copy(net)}
        if progress:
            print(
                f"    mu={mu:g}: loss={row['loss']:.4e} l2re={row['l2re']:.4e} "
                f"({row['seconds_per_iteration']:.3f}s/it)",
                flush=True,
            )

    # ---- gradient descent baseline ----------------------------------- #
    net.load_state_dict(state_after_adam_lbfgs)
    obj = Objective(problem, net, ds)
    gd_res = gd_finetune(
        obj,
        iters=gd_iters,
        lr=(lr if gd_lr is None else gd_lr),
        problem=problem,
        progress=progress,
    )
    gd_row = {
        "pde": pde,
        "width": width,
        "seed": seed,
        "lr": lr,
        "optimizer": "Adam+L-BFGS+GD",
        "mu": None,
        "loss": gd_res.final_loss,
        "l2re": gd_res.final_l2re,
        "grad_norm": gd_res.final_grad_norm,
        "seconds_per_iteration": gd_res.seconds_per_iteration,
        "wall_clock": gd_res.wall_clock,
        "mean_cg_iterations": None,
    }

    # ---- Table 2 (loss / L2RE after fine-tuning) --------------------- #
    rows = [
        {
            "pde": pde,
            "optimizer": "Adam+L-BFGS",
            "loss": baseline["loss"],
            "l2re": baseline["l2re"],
            "grad_norm": baseline["grad_norm"],
        },
        {
            "pde": pde,
            "optimizer": "Adam+L-BFGS+NNCG",
            "loss": best["row"]["loss"],
            "l2re": best["row"]["l2re"],
            "grad_norm": best["row"]["grad_norm"],
        },
        {
            "pde": pde,
            "optimizer": "Adam+L-BFGS+GD",
            "loss": gd_row["loss"],
            "l2re": gd_row["l2re"],
            "grad_norm": gd_row["grad_norm"],
        },
    ]
    table2_path = outdir.table("table2_finetune.csv")
    _append_csv(rows, table2_path)

    # ---- Table 3 (per-iteration wall-clock times) -------------------- #
    table3_path = outdir.table("table3_timing.csv")
    _append_csv(
        [
            {
                "pde": pde,
                "optimizer": "L-BFGS",
                "seconds_per_iteration": lbfgs_seconds_per_iter,
            },
            {
                "pde": pde,
                "optimizer": "NNCG",
                "seconds_per_iteration": best["row"]["seconds_per_iteration"],
            },
            {
                "pde": pde,
                "optimizer": "Time Ratio (NNCG / L-BFGS)",
                "seconds_per_iteration": best["row"]["seconds_per_iteration"]
                / max(lbfgs_seconds_per_iter, 1e-12),
            },
        ],
        table3_path,
    )

    # ---- figures ----------------------------------------------------- #
    plot_underoptimization(
        {
            f"{pde}:NNCG (mu={best['row']['mu']:g})": best["res"].trace,
            f"{pde}:GD": gd_res.trace,
        },
        outdir.figure(f"figure4_{pde}.png"),
    )
    plot_optimizer_curves(
        {f"NNCG (mu={best['row']['mu']:g})": best["res"].trace, "GD": gd_res.trace},
        pde,
        outdir.figure(f"figure4_{pde}_loss.png"),
    )
    # Figure 1: the loss through the Adam -> Adam+L-BFGS -> NNCG pipeline
    adam_it = res.trace["adam_iteration"]
    adam_loss = res.trace["adam_loss"]
    lbfgs_mask = [i >= switch for i in res.trace["iteration"]]
    plot_phase_curves(
        {
            "Adam": (adam_it, adam_loss),
            "Adam+L-BFGS": (
                [i for i, m in zip(res.trace["iteration"], lbfgs_mask) if m],
                [l for l, m in zip(res.trace["loss"], lbfgs_mask) if m],
            ),
            f"NNCG (mu={best['row']['mu']:g})": (
                best["res"].trace["iteration"],
                best["res"].trace["loss"],
            ),
        },
        pde,
        outdir.figure(f"figure1_loss_curve_{pde}.png"),
    )

    # error maps: after Adam, after Adam+L-BFGS, after NNCG
    panels = []
    if state_after_adam is not None:
        net.load_state_dict(state_after_adam)
        panels.append(
            (
                "after Adam",
                np.linspace(problem.x_domain[0], problem.x_domain[1], 255),
                np.linspace(problem.t_domain[0], problem.t_domain[1], 100),
                _error_grid(net, problem),
            )
        )
    net.load_state_dict(state_after_adam_lbfgs)
    panels.append(
        (
            "after Adam+L-BFGS",
            np.linspace(problem.x_domain[0], problem.x_domain[1], 255),
            np.linspace(problem.t_domain[0], problem.t_domain[1], 100),
            _error_grid(net, problem),
        )
    )
    net.load_state_dict(best["state"])
    panels.append(
        (
            "after Adam+L-BFGS+NNCG",
            np.linspace(problem.x_domain[0], problem.x_domain[1], 255),
            np.linspace(problem.t_domain[0], problem.t_domain[1], 100),
            _error_grid(net, problem),
        )
    )
    plot_error_heatmaps(panels, outdir.figure(f"figure5_{pde}.png"))

    npz_path = outdir.table(f"finetune_{pde}.npz")
    np.savez(
        npz_path,
        **{
            "nncg_iteration": np.asarray(best["res"].trace["iteration"]),
            "nncg_loss": np.asarray(best["res"].trace["loss"]),
            "nncg_grad_norm": np.asarray(best["res"].trace["grad_norm"]),
            "gd_iteration": np.asarray(gd_res.trace["iteration"]),
            "gd_loss": np.asarray(gd_res.trace["loss"]),
            "gd_grad_norm": np.asarray(gd_res.trace["grad_norm"]),
        },
    )
    summary = {
        "pde": pde,
        "width": width,
        "seed": seed,
        "lr": lr,
        "switch": switch,
        "iters": iters,
        "baseline": baseline,
        "nncg": best["row"],
        "nncg_all_mu": results,
        "gd": gd_row,
        "lbfgs_seconds_per_iteration": lbfgs_seconds_per_iter,
    }
    with open(os.path.join(outdir.tables, f"finetune_{pde}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def _append_csv(rows: Sequence[dict], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    exists = os.path.exists(path)
    keys = list(rows[0])
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        if not exists:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return path
