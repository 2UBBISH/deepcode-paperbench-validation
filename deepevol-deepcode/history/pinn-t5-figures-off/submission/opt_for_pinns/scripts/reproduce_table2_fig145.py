"""Reproduce Table 2 and Figures 1, 4, 5 of the paper.

This script implements the NNCG fine-tuning experiments described in Section 7:

  * Table 2: Loss & L2RE after NNCG / GD fine-tuning (2000 steps) starting from an
    Adam+L-BFGS solution.  Expected: NNCG reduces loss >10x and improves L2RE,
    while plain gradient descent makes essentially no progress.
  * Figure 1: Wave PDE convergence curves (Adam slow; Adam+L-BFGS stalls; NNCG
    improves further).
  * Figure 4: Loss & gradient norm vs iterations for NNCG vs GD after Adam+L-BFGS.
  * Figure 5: Absolute-error heatmaps at the switch points (after Adam / +L-BFGS /
    +NNCG).

The script is read-only by default: it consumes the artifacts produced by
``run_experiments.py`` (or trains on demand with ``--run``) and writes plots /
JSON summaries under ``results/``.

Usage
-----
    python scripts/reproduce_table2_fig145.py --run            # train + evaluate
    python scripts/reproduce_table2_fig145.py                  # use cached results
    python scripts/reproduce_table2_fig145.py --pdes wave      # single PDE
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Make ``src`` importable when running the script directly.
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data import build_data, make_eval_grid  # noqa: E402
from src.loss import make_loss_fn  # noqa: E402
from src.metrics import evaluate, l2_relative_error_from_data  # noqa: E402
from src.model import build_model, flatten_parameters, set_flat_parameters  # noqa: E402
from src.optimizers.adam_lbfgs import train_adam, train_adam_lbfgs  # noqa: E402
from src.optimizers.nncg import nncg_minimize  # noqa: E402
from src.pdes import get_pde  # noqa: E402
from src.utils import (  # noqa: E402
    ensure_dir,
    get_device,
    get_logger,
    load_config,
    load_json,
    results_dir,
    save_json,
    set_seed,
)

logger = get_logger("reproduce_table2_fig145")

# ---------------------------------------------------------------------------
# Defaults (mirroring the paper / configs)
# ---------------------------------------------------------------------------
DEFAULT_PDES = ["convection", "reaction", "wave"]
DEFAULT_WIDTH = 200
DEFAULT_SEED = 0
DEFAULT_SWITCH = 11000
DEFAULT_TOTAL_ITERS = 41000
DEFAULT_LBFGS_MEMORY = 100
DEFAULT_FINETUNE_STEPS = 2000
DEFAULT_MU_GRID = [1e-2, 1e-1]
DEFAULT_BEST_LR = {"convection": 1e-4, "reaction": 1e-3, "wave": 1e-3}

# Reference values from the paper (Table 2) for qualitative comparison.
PAPER_TABLE2 = {
    "convection": {
        "adam_lbfgs": {"loss": 5.95e-6, "l2re": 4.19e-3},
        "nncg": {"loss": 3.63e-6, "l2re": 1.94e-3},
    },
    "reaction": {
        "adam_lbfgs": {"loss": 5.26e-6, "l2re": 1.92e-2},
        "nncg": {"loss": 2.89e-7, "l2re": 9.92e-3},
    },
    "wave": {
        "adam_lbfgs": {"loss": 1.12e-3, "l2re": 5.52e-2},
        "nncg": {"loss": 6.13e-5, "l2re": 1.27e-2},
    },
}

OPT_LABEL = {
    "adam": "Adam",
    "lbfgs": "L-BFGS",
    "adam_lbfgs": "Adam+L-BFGS",
    "nncg": "NNCG",
    "gd": "GD",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_pde_config(pde_name: str, config_dir: str = "configs") -> Dict[str, Any]:
    """Load ``configs/<pde>.yaml`` if present, else return an empty dict."""
    path = os.path.join(_ROOT, config_dir, f"{pde_name}.yaml")
    if os.path.isfile(path):
        try:
            return load_config(path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to load config %s: %s", path, exc)
    return {}


def _resolve_lr(pde_name: str, cfg: Dict[str, Any], cli_lr: Optional[float]) -> float:
    if cli_lr is not None:
        return float(cli_lr)
    best = cfg.get("best_lr", {}) if cfg else {}
    if isinstance(best, dict) and "adam_lbfgs" in best:
        return float(best["adam_lbfgs"])
    return float(DEFAULT_BEST_LR.get(pde_name, 1e-3))


def _build_problem(
    pde_name: str,
    width: int,
    seed: int,
    device: torch.device,
    cfg: Optional[Dict[str, Any]] = None,
):
    """Instantiate (pde, model, data) for a given PDE/width/seed."""
    cfg = cfg or {}
    pde_kwargs = cfg.get("pde_kwargs", {}) or {}
    pde = get_pde(pde_name, **pde_kwargs)
    model = build_model(width=width, depth=3, seed=seed, device=device)
    data = build_data(pde, seed=seed, device=device)
    return pde, model, data


def _grad_norm(loss_fn, model) -> float:
    """Compute ||grad L(w)||_2 without touching ``.grad`` buffers."""
    params = [p for p in model.parameters() if p.requires_grad]
    loss = loss_fn()
    grads = torch.autograd.grad(loss, params, retain_graph=False, create_graph=False,
                                allow_unused=True)
    total = 0.0
    for g in grads:
        if g is not None:
            total += float(g.detach().pow(2).sum().item())
    return total ** 0.5


def _clone_model(model):
    """Deep-copy a model (architecture + weights)."""
    import copy

    return copy.deepcopy(model)


# ---------------------------------------------------------------------------
# Fine-tuning routines
# ---------------------------------------------------------------------------
def finetune_nncg(
    model,
    loss_fn,
    steps: int = DEFAULT_FINETUNE_STEPS,
    mu: float = 1e-2,
    s: int = 60,
    F: int = 20,
    M: int = 1000,
    eps: float = 1e-16,
    eta: float = 1.0,
    alpha: float = 0.1,
    beta: float = 0.5,
    verbose: bool = False,
    log_every: int = 100,
) -> Tuple[Any, Dict[str, List[float]]]:
    """Run NNCG fine-tuning for ``steps`` iterations.

    Returns the (in-place updated) model and a history dict with keys
    ``loss``, ``grad_norm``, ``step``.
    """
    _, summary, history = nncg_minimize(
        model,
        loss_fn,
        mu=mu,
        s=s,
        F=F,
        K=steps,
        M=M,
        eps=eps,
        eta=eta,
        alpha=alpha,
        beta=beta,
        verbose=verbose,
        log_every=log_every,
    )
    return model, history


def finetune_gd(
    model,
    loss_fn,
    steps: int = DEFAULT_FINETUNE_STEPS,
    lr: float = 1e-3,
    verbose: bool = False,
    log_every: int = 100,
) -> Tuple[Any, Dict[str, List[float]]]:
    """Plain gradient-descent fine-tuning baseline (2000 steps).

    Uses ``torch.optim.SGD`` with the given learning rate.  The paper reports
    that GD makes essentially no progress from an Adam+L-BFGS solution.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr)
    history: Dict[str, List[float]] = {"loss": [], "grad_norm": [], "step": []}
    for k in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn()
        loss.backward()
        opt.step()
        if k % log_every == 0 or k == steps - 1:
            history["loss"].append(float(loss.detach().item()))
            history["grad_norm"].append(_grad_norm(loss_fn, model))
            history["step"].append(k)
            if verbose:
                logger.info("GD step %d loss=%.6e", k, history["loss"][-1])
    return model, history


# ---------------------------------------------------------------------------
# Per-PDE experiment
# ---------------------------------------------------------------------------
def run_pde_experiment(
    pde_name: str,
    width: int = DEFAULT_WIDTH,
    seed: int = DEFAULT_SEED,
    lr: Optional[float] = None,
    switch: int = DEFAULT_SWITCH,
    total_iters: int = DEFAULT_TOTAL_ITERS,
    lbfgs_memory: int = DEFAULT_LBFGS_MEMORY,
    finetune_steps: int = DEFAULT_FINETUNE_STEPS,
    mu_grid: Optional[List[float]] = None,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Train Adam+L-BFGS then fine-tune with NNCG and GD.

    Returns a dict with the loss/L2RE at each stage, the fine-tuning histories,
    and the per-iteration wall-clock timings.
    """
    device = device or get_device()
    mu_grid = mu_grid or DEFAULT_MU_GRID
    cfg = _load_pde_config(pde_name)
    lr = _resolve_lr(pde_name, cfg, lr)

    set_seed(seed)
    pde, model, data = _build_problem(pde_name, width, seed, device, cfg)
    loss_fn = make_loss_fn(model, pde, data)

    result: Dict[str, Any] = {
        "pde": pde_name,
        "width": width,
        "seed": seed,
        "lr": lr,
        "switch": switch,
        "finetune_steps": finetune_steps,
    }

    # --- Stage 1: Adam only (up to switch) ---------------------------------
    t0 = time.time()
    train_adam(model, loss_fn, lr=lr, total_iters=switch, verbose=verbose,
               log_every=max(1, switch // 10))
    adam_time = time.time() - t0
    adam_metrics = evaluate(model, pde, data)
    result["adam"] = {
        "loss": adam_metrics["loss"],
        "l2re": adam_metrics["l2re"],
        "grad_norm": adam_metrics.get("grad_norm"),
        "time": adam_time,
    }
    adam_state = _clone_model(model)

    # --- Stage 2: Adam + L-BFGS (to total_iters) ---------------------------
    t0 = time.time()
    train_adam_lbfgs(
        model,
        loss_fn,
        adam_lr=lr,
        switch_iter=switch,
        total_iters=total_iters,
        lbfgs_memory=lbfgs_memory,
        verbose=verbose,
        log_every=max(1, (total_iters - switch) // 10),
    )
    lbfgs_time = time.time() - t0
    al_metrics = evaluate(model, pde, data)
    result["adam_lbfgs"] = {
        "loss": al_metrics["loss"],
        "l2re": al_metrics["l2re"],
        "grad_norm": al_metrics.get("grad_norm"),
        "time": lbfgs_time,
    }
    al_state = _clone_model(model)

    # --- Stage 3a: NNCG fine-tuning (tune mu by smallest final loss) -------
    best_nncg: Optional[Dict[str, Any]] = None
    for mu in mu_grid:
        trial = _clone_model(al_state)
        trial_loss_fn = make_loss_fn(trial, pde, data)
        t0 = time.time()
        _, hist = finetune_nncg(
            trial, trial_loss_fn, steps=finetune_steps, mu=mu, verbose=verbose
        )
        nncg_time = time.time() - t0
        metrics = evaluate(trial, pde, data)
        record = {
            "mu": mu,
            "loss": metrics["loss"],
            "l2re": metrics["l2re"],
            "grad_norm": metrics.get("grad_norm"),
            "time": nncg_time,
            "time_per_iter": nncg_time / max(1, finetune_steps),
            "history": hist,
        }
        if best_nncg is None or record["loss"] < best_nncg["loss"]:
            best_nncg = record
        if verbose:
            logger.info(
                "[%s] NNCG mu=%.1e loss=%.6e l2re=%.6e",
                pde_name, mu, record["loss"], record["l2re"],
            )
    result["nncg"] = best_nncg

    # --- Stage 3b: GD fine-tuning baseline ---------------------------------
    gd_model = _clone_model(al_state)
    gd_loss_fn = make_loss_fn(gd_model, pde, data)
    t0 = time.time()
    _, gd_hist = finetune_gd(
        gd_model, gd_loss_fn, steps=finetune_steps, lr=lr, verbose=verbose
    )
    gd_time = time.time() - t0
    gd_metrics = evaluate(gd_model, pde, data)
    result["gd"] = {
        "loss": gd_metrics["loss"],
        "l2re": gd_metrics["l2re"],
        "grad_norm": gd_metrics.get("grad_norm"),
        "time": gd_time,
        "time_per_iter": gd_time / max(1, finetune_steps),
        "history": gd_hist,
    }

    # --- Stage 4: absolute-error heatmaps (Figure 5) -----------------------
    result["heatmaps"] = _compute_heatmaps(pde, adam_state, al_state, model, data, device)

    # --- Stage 5: convergence curves (Figures 1 & 4) -----------------------
    result["convergence"] = _compute_convergence(
        pde, model, data, al_state, lr, finetune_steps, device, verbose
    )

    return result


def _compute_heatmaps(pde, adam_state, al_state, nncg_state, data, device):
    """Compute absolute-error grids for Adam / +L-BFGS / +NNCG (Figure 5)."""
    x_grid, t_grid = make_eval_grid(pde, device=device)
    out: Dict[str, Any] = {}
    for label, mdl in (("adam", adam_state), ("adam_lbfgs", al_state), ("nncg", nncg_state)):
        with torch.no_grad():
            pred = mdl(x_grid, t_grid).reshape(-1)
            exact = pde.exact(x_grid, t_grid).reshape(-1)
            err = (pred - exact).abs()
        out[label] = {
            "x": x_grid.reshape(-1).detach().cpu().tolist(),
            "t": t_grid.reshape(-1).detach().cpu().tolist(),
            "error": err.detach().cpu().tolist(),
            "max_error": float(err.max().item()),
        }
    return out


def _compute_convergence(
    pde, nncg_model, data, al_state, lr, finetune_steps, device, verbose
):
    """Record loss/grad-norm trajectories for NNCG vs GD (Figures 1 & 4)."""
    # NNCG trajectory
    nncg_trial = _clone_model(al_state)
    nncg_loss_fn = make_loss_fn(nncg_trial, pde, data)
    _, nncg_hist = finetune_nncg(
        nncg_trial, nncg_loss_fn, steps=finetune_steps, mu=1e-2, verbose=verbose
    )
    # GD trajectory
    gd_trial = _clone_model(al_state)
    gd_loss_fn = make_loss_fn(gd_trial, pde, data)
    _, gd_hist = finetune_gd(gd_trial, gd_loss_fn, steps=finetune_steps, lr=lr,
                             verbose=verbose)
    return {"nncg": nncg_hist, "gd": gd_hist}


# ---------------------------------------------------------------------------
# Table 2
# ---------------------------------------------------------------------------
def build_table2(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Build the Table 2 structure: {pde: {stage: {loss, l2re}}}."""
    table: Dict[str, Dict[str, Dict[str, float]]] = {}
    for r in results:
        pde = r["pde"]
        table[pde] = {
            "adam_lbfgs": {
                "loss": r["adam_lbfgs"]["loss"],
                "l2re": r["adam_lbfgs"]["l2re"],
            },
            "nncg": {"loss": r["nncg"]["loss"], "l2re": r["nncg"]["l2re"]},
            "gd": {"loss": r["gd"]["loss"], "l2re": r["gd"]["l2re"]},
        }
    return table


def format_table2(table: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    lines = []
    header = f"{'PDE':<12} {'Stage':<14} {'Loss':>12} {'L2RE':>12} {'Loss ratio':>12}"
    lines.append(header)
    lines.append("-" * len(header))
    for pde, stages in table.items():
        base = stages["adam_lbfgs"]["loss"]
        for stage in ("adam_lbfgs", "nncg", "gd"):
            s = stages[stage]
            ratio = base / s["loss"] if s["loss"] > 0 else float("inf")
            lines.append(
                f"{pde:<12} {OPT_LABEL.get(stage, stage):<14} "
                f"{s['loss']:>12.4e} {s['l2re']:>12.4e} {ratio:>12.2f}"
            )
        lines.append("")
    return "\n".join(lines)


def format_table2_comparison(table: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    lines = ["Comparison against paper Table 2 (reference values):", ""]
    for pde, ref in PAPER_TABLE2.items():
        if pde not in table:
            continue
        lines.append(f"[{pde}]")
        for stage in ("adam_lbfgs", "nncg"):
            got = table[pde][stage]
            exp = ref[stage]
            lines.append(
                f"  {OPT_LABEL.get(stage, stage):<14} "
                f"loss={got['loss']:.3e} (paper {exp['loss']:.3e})  "
                f"l2re={got['l2re']:.3e} (paper {exp['l2re']:.3e})"
            )
        lines.append("")
    return "\n".join(lines)


def check_table2_trends(table: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    lines = ["Qualitative trend checks (Table 2):"]
    for pde, stages in table.items():
        base = stages["adam_lbfgs"]["loss"]
        nncg = stages["nncg"]["loss"]
        gd = stages["gd"]["loss"]
        improved = nncg < base
        gd_flat = abs(gd - base) / max(base, 1e-30) < 0.5
        lines.append(
            f"  {pde:<12} NNCG reduces loss: {improved} "
            f"(x{base / max(nncg, 1e-30):.2f}); GD ~flat: {gd_flat}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_figure1(results: List[Dict[str, Any]], out_path: str = "fig1.png") -> str:
    """Wave PDE convergence: Adam vs Adam+L-BFGS vs NNCG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wave = next((r for r in results if r["pde"] == "wave"), None)
    if wave is None:
        raise ValueError("No wave result available for Figure 1")

    conv = wave["convergence"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(conv["nncg"]["step"], conv["nncg"]["loss"], label="NNCG", lw=2)
    ax.plot(conv["gd"]["step"], conv["gd"]["loss"], label="GD", lw=2, ls="--")
    ax.axhline(wave["adam_lbfgs"]["loss"], color="k", ls=":", label="Adam+L-BFGS")
    ax.set_yscale("log")
    ax.set_xlabel("Fine-tuning iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Wave PDE: convergence after Adam+L-BFGS")
    ax.legend()
    fig.tight_layout()
    ensure_dir(os.path.dirname(out_path) or ".")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_figure4(results: List[Dict[str, Any]], out_path: str = "fig4.png") -> str:
    """Loss & gradient norm vs iterations for NNCG vs GD (all PDEs)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8), squeeze=False)
    for j, r in enumerate(results):
        conv = r["convergence"]
        ax_loss = axes[0][j]
        ax_gn = axes[1][j]
        ax_loss.plot(conv["nncg"]["step"], conv["nncg"]["loss"], label="NNCG", lw=2)
        ax_loss.plot(conv["gd"]["step"], conv["gd"]["loss"], label="GD", lw=2, ls="--")
        ax_loss.set_yscale("log")
        ax_loss.set_title(f"{r['pde']}: loss")
        ax_loss.set_xlabel("iteration")
        ax_loss.set_ylabel("loss")
        ax_loss.legend()

        ax_gn.plot(conv["nncg"]["step"], conv["nncg"]["grad_norm"], label="NNCG", lw=2)
        ax_gn.plot(conv["gd"]["step"], conv["gd"]["grad_norm"], label="GD", lw=2, ls="--")
        ax_gn.set_yscale("log")
        ax_gn.set_title(f"{r['pde']}: grad norm")
        ax_gn.set_xlabel("iteration")
        ax_gn.set_ylabel("||grad L||")
        ax_gn.legend()
    fig.tight_layout()
    ensure_dir(os.path.dirname(out_path) or ".")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_figure5(results: List[Dict[str, Any]], out_path: str = "fig5.png") -> str:
    """Absolute-error heatmaps at switch points (Adam / +L-BFGS / +NNCG)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(12, 3.5 * n), squeeze=False)
    stages = ["adam", "adam_lbfgs", "nncg"]
    for i, r in enumerate(results):
        hm = r["heatmaps"]
        for j, stage in enumerate(stages):
            d = hm[stage]
            x = np.asarray(d["x"])
            t = np.asarray(d["t"])
            e = np.asarray(d["error"])
            ax = axes[i][j]
            sc = ax.scatter(x, t, c=e, s=4, cmap="viridis")
            ax.set_title(f"{r['pde']} - {OPT_LABEL.get(stage, stage)}")
            ax.set_xlabel("x")
            ax.set_ylabel("t")
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    ensure_dir(os.path.dirname(out_path) or ".")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def _strip_heavy(result: Dict[str, Any]) -> Dict[str, Any]:
    """Remove large arrays (heatmaps) before JSON serialization."""
    out = dict(result)
    out.pop("heatmaps", None)
    return out


def save_results(results: List[Dict[str, Any]], out_dir: str) -> str:
    ensure_dir(out_dir)
    path = os.path.join(out_dir, "table2_fig145.json")
    save_json([_strip_heavy(r) for r in results], path)
    return path


def load_results(out_dir: str) -> Optional[List[Dict[str, Any]]]:
    path = os.path.join(out_dir, "table2_fig145.json")
    if os.path.isfile(path):
        return load_json(path)
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Reproduce Table 2 and Figures 1, 4, 5 (NNCG fine-tuning)."
    )
    p.add_argument("--pdes", nargs="+", default=DEFAULT_PDES)
    p.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--switch", type=int, default=DEFAULT_SWITCH)
    p.add_argument("--total-iters", type=int, default=DEFAULT_TOTAL_ITERS)
    p.add_argument("--lbfgs-memory", type=int, default=DEFAULT_LBFGS_MEMORY)
    p.add_argument("--finetune-steps", type=int, default=DEFAULT_FINETUNE_STEPS)
    p.add_argument("--mu-grid", nargs="+", type=float, default=DEFAULT_MU_GRID)
    p.add_argument("--root", type=str, default="results")
    p.add_argument("--run", action="store_true",
                   help="Run the experiments (otherwise use cached results).")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    device = get_device()
    out_dir = str(results_dir(args.root, "table2_fig145"))

    results: Optional[List[Dict[str, Any]]] = None
    if not args.run:
        results = load_results(out_dir)
        if results is None:
            logger.info("No cached results found; running experiments.")
            args.run = True

    if args.run:
        results = []
        for pde_name in args.pdes:
            logger.info("=== Running %s ===", pde_name)
            r = run_pde_experiment(
                pde_name,
                width=args.width,
                seed=args.seed,
                lr=args.lr,
                switch=args.switch,
                total_iters=args.total_iters,
                lbfgs_memory=args.lbfgs_memory,
                finetune_steps=args.finetune_steps,
                mu_grid=args.mu_grid,
                device=device,
                verbose=args.verbose,
            )
            results.append(r)
        save_results(results, out_dir)

    assert results is not None

    table = build_table2(results)
    print(format_table2(table))
    print(format_table2_comparison(table))
    print(check_table2_trends(table))

    if not args.no_plots:
        try:
            print("Figure 1 ->", plot_figure1(results, os.path.join(out_dir, "fig1.png")))
            print("Figure 4 ->", plot_figure4(results, os.path.join(out_dir, "fig4.png")))
            print("Figure 5 ->", plot_figure5(results, os.path.join(out_dir, "fig5.png")))
        except Exception as exc:  # pragma: no cover - plotting is best-effort
            logger.warning("Plotting failed: %s", exc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
