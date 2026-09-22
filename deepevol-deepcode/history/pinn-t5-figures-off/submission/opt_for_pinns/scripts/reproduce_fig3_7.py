"""Reproduce Figures 3 and 7 of the paper.

Figure 3 (convection):
    Top:    Hessian spectral density of the *full* PINN loss at the Adam+L-BFGS
            switch point (11k iterations), both raw and L-BFGS-preconditioned.
    Bottom: Per-component (residual / IC / BC) Hessian spectral densities.

Figure 7 (reaction & wave):
    Per-component Hessian spectral densities (same layout as Fig. 3 bottom).

The spectral density is estimated with Stochastic Lanczos Quadrature (SLQ),
following the PyHessian approach (see ``src/hessian/spectral_density.py``).

For the preconditioned curve we use Algorithm 2 (L-BFGS unrolling) and
Algorithm 3 (preconditioned Hessian matvec) from the paper, which require the
curvature pairs ``{s_k, y_k, rho_k}`` recorded during the Adam+L-BFGS run.

Usage
-----
    python scripts/reproduce_fig3_7.py --pde convection --out fig3.png
    python scripts/reproduce_fig3_7.py --pde reaction   --out fig7_reaction.png
    python scripts/reproduce_fig3_7.py --pde wave       --out fig7_wave.png
    python scripts/reproduce_fig3_7.py --all            # all three PDEs

By default the script trains the required models (Adam+L-BFGS to the switch
point) and then estimates the spectral densities.  Use ``--no-train`` to reuse
checkpoints saved under ``results/spectral/<pde>/``.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Make ``src`` importable when the script is run directly.
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data import build_data, make_eval_points  # noqa: E402
from src.hessian.hvp import HVPOperator  # noqa: E402
from src.hessian.lbfgs_unroll import LBFGSHistory  # noqa: E402
from src.hessian.precond_matvec import build_preconditioned_matvec  # noqa: E402
from src.hessian.spectral_density import spectral_density  # noqa: E402
from src.loss import component_loss_fns, make_loss_fn  # noqa: E402
from src.model import build_model  # noqa: E402
from src.optimizers.adam_lbfgs import train_adam_lbfgs  # noqa: E402
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

logger = get_logger("reproduce_fig3_7")

# ---------------------------------------------------------------------------
# Defaults (mirror configs/*.yaml)
# ---------------------------------------------------------------------------
DEFAULT_PDES = ["convection", "reaction", "wave"]
DEFAULT_WIDTH = 200
DEFAULT_SWITCH = 11000
DEFAULT_TOTAL_ITERS = 41000
DEFAULT_LBFGS_MEMORY = 100
DEFAULT_NUM_MATVECS = 100
DEFAULT_NUM_BINS = 200
DEFAULT_SEEDS = [345, 456, 567]

# Best learning rates per PDE (smallest final L2RE, see configs/*.yaml).
DEFAULT_BEST_LR: Dict[str, float] = {
    "convection": 1e-4,
    "reaction": 1e-3,
    "wave": 1e-3,
}

COMPONENTS = ["residual", "ic", "bc"]


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def _pde_config_path(pde_name: str, config_dir: str = "configs") -> str:
    return os.path.join(_ROOT, config_dir, f"{pde_name}.yaml")


def _load_pde_config(pde_name: str, config_dir: str = "configs") -> Dict[str, Any]:
    path = _pde_config_path(pde_name, config_dir)
    if os.path.exists(path):
        try:
            return load_config(path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to load config %s: %s", path, exc)
    return {}


def _resolve_settings(pde_name: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve width / switch / lr / seeds / SLQ settings for a PDE."""
    spectral = cfg.get("spectral", {}) or {}
    best_lr = cfg.get("best_lr", {}) or {}
    training = cfg.get("training", {}) or {}

    width = int(spectral.get("width", DEFAULT_WIDTH))
    switch = int(spectral.get("switch_iter", training.get("default_switch", DEFAULT_SWITCH)))
    lr = float(
        spectral.get(
            "lr",
            best_lr.get("adam_lbfgs", DEFAULT_BEST_LR.get(pde_name, 1e-3)),
        )
    )
    seeds = list(spectral.get("seeds", DEFAULT_SEEDS))
    num_matvecs = int(spectral.get("num_matvecs", DEFAULT_NUM_MATVECS))
    num_bins = int(spectral.get("num_bins", DEFAULT_NUM_BINS))
    lbfgs_memory = int(training.get("lbfgs_memory", DEFAULT_LBFGS_MEMORY))

    return {
        "width": width,
        "switch": switch,
        "lr": lr,
        "seeds": seeds,
        "num_matvecs": num_matvecs,
        "num_bins": num_bins,
        "lbfgs_memory": lbfgs_memory,
    }


# ---------------------------------------------------------------------------
# Problem construction / training
# ---------------------------------------------------------------------------
def _build_problem(pde_name: str, width: int, seed: int, device: torch.device):
    cfg = _load_pde_config(pde_name)
    pde_kwargs = cfg.get("pde_kwargs", {}) or {}
    pde = get_pde(pde_name, **pde_kwargs)
    set_seed(seed)
    model = build_model(width=width, depth=3, seed=seed, device=device)
    data = build_data(pde, seed=seed, device=device)
    return pde, model, data


def train_to_switch(
    pde_name: str,
    width: int,
    seed: int,
    lr: float,
    switch: int,
    lbfgs_memory: int,
    device: torch.device,
    total_iters: int = DEFAULT_TOTAL_ITERS,
    verbose: bool = False,
) -> Tuple[Any, torch.nn.Module, Any, Optional[LBFGSHistory]]:
    """Train Adam+L-BFGS up to ``switch`` iterations, recording curvature pairs."""
    pde, model, data = _build_problem(pde_name, width, seed, device)
    loss_fn = make_loss_fn(model, pde, data)

    _, history, curvature = train_adam_lbfgs(
        model,
        loss_fn,
        adam_lr=lr,
        switch_iter=switch,
        total_iters=total_iters,
        lbfgs_memory=lbfgs_memory,
        record_curvature=True,
        verbose=verbose,
        log_every=max(1, switch // 10),
    )
    return pde, model, data, curvature


# ---------------------------------------------------------------------------
# Spectral density estimation
# ---------------------------------------------------------------------------
def _density_for_loss_fn(
    loss_fn,
    model: torch.nn.Module,
    num_matvecs: int,
    num_bins: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, Any]:
    """Estimate the (raw) Hessian spectral density of a scalar loss closure."""
    op = HVPOperator(loss_fn, model)
    p = op.numel
    result = spectral_density(
        op,
        dim=p,
        num_matvecs=num_matvecs,
        num_bins=num_bins,
        device=device,
        generator=generator,
    )
    return result


def _density_preconditioned(
    loss_fn,
    model: torch.nn.Module,
    curvature: Optional[LBFGSHistory],
    num_matvecs: int,
    num_bins: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> Optional[Dict[str, Any]]:
    """Estimate the L-BFGS-preconditioned Hessian spectral density (Alg. 2 & 3)."""
    if curvature is None or len(curvature) == 0:
        logger.warning("No curvature pairs available; skipping preconditioned density.")
        return None

    op = HVPOperator(loss_fn, model)
    p = op.numel
    precond_op = build_preconditioned_matvec(op, curvature, p)
    dim = precond_op.numel
    result = spectral_density(
        precond_op,
        dim=dim,
        num_matvecs=num_matvecs,
        num_bins=num_bins,
        device=device,
        generator=generator,
    )
    return result


def compute_spectra(
    pde_name: str,
    width: int,
    seed: int,
    lr: float,
    switch: int,
    lbfgs_memory: int,
    num_matvecs: int,
    num_bins: int,
    device: torch.device,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Train to the switch point and compute full + per-component spectra."""
    pde, model, data, curvature = train_to_switch(
        pde_name,
        width,
        seed,
        lr,
        switch,
        lbfgs_memory,
        device,
        verbose=verbose,
    )

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    full_loss_fn = make_loss_fn(model, pde, data)
    logger.info("[%s] estimating full-loss spectral density ...", pde_name)
    full = _density_for_loss_fn(
        full_loss_fn, model, num_matvecs, num_bins, device, generator=gen
    )

    logger.info("[%s] estimating preconditioned spectral density ...", pde_name)
    precond = _density_preconditioned(
        full_loss_fn, model, curvature, num_matvecs, num_bins, device, generator=gen
    )

    comp_fns = component_loss_fns(model, pde, data)
    components: Dict[str, Dict[str, Any]] = {}
    for name in COMPONENTS:
        fn = comp_fns.get(name)
        if fn is None:
            continue
        logger.info("[%s] estimating '%s' spectral density ...", pde_name, name)
        components[name] = _density_for_loss_fn(
            fn, model, num_matvecs, num_bins, device, generator=gen
        )

    return {
        "pde": pde_name,
        "width": width,
        "seed": seed,
        "lr": lr,
        "switch": switch,
        "full": full,
        "preconditioned": precond,
        "components": components,
    }


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------
def _to_serializable(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def save_spectra(spectra: Dict[str, Any], out_dir: str, pde_name: str) -> str:
    ensure_dir(out_dir)
    path = os.path.join(out_dir, f"spectra_{pde_name}.json")
    save_json(_to_serializable(spectra), path)
    logger.info("Saved spectra to %s", path)
    return path


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_figure3(spectra: Dict[str, Any], out_path: str = "fig3.png") -> str:
    """Figure 3: full + preconditioned (top) and per-component (bottom)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=False)

    # --- Top: full loss, raw vs preconditioned -----------------------------
    ax = axes[0]
    full = spectra["full"]
    ax.plot(full["grid"], full["density"], label="Full loss (raw)", color="C0")
    precond = spectra.get("preconditioned")
    if precond is not None:
        ax.plot(
            precond["grid"],
            precond["density"],
            label="Full loss (L-BFGS preconditioned)",
            color="C1",
        )
    ax.set_title(f"Figure 3 (top): {spectra['pde']} full-loss Hessian spectral density")
    ax.set_xlabel("eigenvalue")
    ax.set_ylabel("spectral density")
    ax.set_yscale("log")
    ax.legend()

    # --- Bottom: per-component --------------------------------------------
    ax = axes[1]
    for name in COMPONENTS:
        comp = spectra["components"].get(name)
        if comp is None:
            continue
        ax.plot(comp["grid"], comp["density"], label=name)
    ax.set_title(f"Figure 3 (bottom): {spectra['pde']} per-component spectral density")
    ax.set_xlabel("eigenvalue")
    ax.set_ylabel("spectral density")
    ax.set_yscale("log")
    ax.legend()

    fig.tight_layout()
    ensure_dir(os.path.dirname(os.path.abspath(out_path)))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved figure to %s", out_path)
    return out_path


def plot_figure7(spectra: Dict[str, Any], out_path: str = "fig7.png") -> str:
    """Figure 7: per-component spectral density for reaction & wave."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for name in COMPONENTS:
        comp = spectra["components"].get(name)
        if comp is None:
            continue
        ax.plot(comp["grid"], comp["density"], label=name)
    ax.set_title(f"Figure 7: {spectra['pde']} per-component Hessian spectral density")
    ax.set_xlabel("eigenvalue")
    ax.set_ylabel("spectral density")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    ensure_dir(os.path.dirname(os.path.abspath(out_path)))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved figure to %s", out_path)
    return out_path


def summarize(spectra: Dict[str, Any]) -> str:
    """Human-readable summary of top eigenvalues / conditioning."""
    lines: List[str] = []
    lines.append(f"PDE: {spectra['pde']}  (width={spectra['width']}, seed={spectra['seed']})")

    def _top(d: Dict[str, Any]) -> float:
        eig = d.get("eigenvalues")
        if eig is None:
            return float("nan")
        try:
            return float(max(eig))
        except Exception:
            return float("nan")

    full = spectra["full"]
    lines.append(f"  full loss: top eigenvalue ~ {_top(full):.3e}")
    precond = spectra.get("preconditioned")
    if precond is not None:
        lines.append(f"  preconditioned: top eigenvalue ~ {_top(precond):.3e}")
    for name in COMPONENTS:
        comp = spectra["components"].get(name)
        if comp is not None:
            lines.append(f"  {name}: top eigenvalue ~ {_top(comp):.3e}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce Figures 3 and 7 (Hessian spectral density)."
    )
    parser.add_argument("--pde", type=str, default="convection", choices=DEFAULT_PDES)
    parser.add_argument("--all", action="store_true", help="Run all three PDEs.")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--switch", type=int, default=None)
    parser.add_argument("--num-matvecs", type=int, default=None)
    parser.add_argument("--num-bins", type=int, default=None)
    parser.add_argument("--total-iters", type=int, default=DEFAULT_TOTAL_ITERS)
    parser.add_argument("--out", type=str, default=None, help="Output figure path.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output dir for JSON.")
    parser.add_argument("--config-dir", type=str, default="configs")
    parser.add_argument("--cpu", action="store_true", help="Force CPU.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _run_one(pde_name: str, args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    cfg = _load_pde_config(pde_name, args.config_dir)
    settings = _resolve_settings(pde_name, cfg)

    width = args.width if args.width is not None else settings["width"]
    seed = args.seed if args.seed is not None else settings["seeds"][0]
    lr = args.lr if args.lr is not None else settings["lr"]
    switch = args.switch if args.switch is not None else settings["switch"]
    num_matvecs = (
        args.num_matvecs if args.num_matvecs is not None else settings["num_matvecs"]
    )
    num_bins = args.num_bins if args.num_bins is not None else settings["num_bins"]

    logger.info(
        "Running %s: width=%d seed=%d lr=%g switch=%d matvecs=%d",
        pde_name,
        width,
        seed,
        lr,
        switch,
        num_matvecs,
    )

    spectra = compute_spectra(
        pde_name=pde_name,
        width=width,
        seed=seed,
        lr=lr,
        switch=switch,
        lbfgs_memory=settings["lbfgs_memory"],
        num_matvecs=num_matvecs,
        num_bins=num_bins,
        device=device,
        verbose=args.verbose,
    )

    out_dir = args.out_dir or str(results_dir("spectral", pde_name))
    save_spectra(spectra, out_dir, pde_name)
    logger.info("\n%s", summarize(spectra))
    return spectra


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device("cpu") if args.cpu else get_device()

    pdes = DEFAULT_PDES if args.all else [args.pde]

    for pde_name in pdes:
        spectra = _run_one(pde_name, args, device)

        if args.out is not None:
            out_path = args.out if not args.all else f"{os.path.splitext(args.out)[0]}_{pde_name}.png"
        else:
            out_path = str(results_dir("figures", f"fig3_7_{pde_name}.png"))

        if pde_name == "convection":
            plot_figure3(spectra, out_path)
        else:
            plot_figure7(spectra, out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
