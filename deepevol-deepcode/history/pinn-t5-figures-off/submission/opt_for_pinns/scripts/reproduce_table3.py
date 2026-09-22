"""Reproduce Table 3 of "Challenges in Training PINNs: A Loss Landscape Perspective".

Table 3 reports the per-iteration wall-clock time (in seconds) of L-BFGS and
NysNewton-CG (NNCG) for each of the three benchmark PDEs, together with the
ratio ``NNCG / L-BFGS``.

Reference (paper) values
------------------------
    PDE          L-BFGS (s)   NNCG (s)   ratio
    convection   4.6e-2       2.5e-1     5.43
    reaction     3.6e-2       7.2e-1     20.00
    wave         9.0e-2       2.9e+1     322.22

The script measures these times directly by running a small number of
iterations of each optimizer on the PINN loss at the Adam+L-BFGS switch point
(11k iterations, width=200, best lr per PDE).  Because wall-clock timings are
hardware dependent, the script reports both the measured values and the
reference values, and checks that the *ordering* (L-BFGS faster than NNCG, and
NNCG/L-BFGS ratio increasing from convection -> reaction -> wave) matches.

Usage
-----
    # Measure timings (trains to switch point, then times each optimizer)
    python scripts/reproduce_table3.py --run

    # Re-print a previously saved table
    python scripts/reproduce_table3.py

    # Restrict to a subset of PDEs / tune the number of timed iterations
    python scripts/reproduce_table3.py --run --pdes convection wave --lbfgs-iters 20 --nncg-iters 5
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Make ``src`` importable when the script is run directly.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data import build_data  # noqa: E402
from src.loss import make_loss_fn  # noqa: E402
from src.model import build_model  # noqa: E402
from src.optimizers.adam_lbfgs import train_adam_lbfgs  # noqa: E402
from src.optimizers.nncg import NysNewtonCG  # noqa: E402
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_PDES: List[str] = ["convection", "reaction", "wave"]
DEFAULT_WIDTH: int = 200
DEFAULT_SEED: int = 0
DEFAULT_SWITCH: int = 11000
DEFAULT_TOTAL_ITERS: int = 41000
DEFAULT_LBFGS_MEMORY: int = 100
DEFAULT_BEST_LR: Dict[str, float] = {
    "convection": 1e-4,
    "reaction": 1e-3,
    "wave": 1e-3,
}

# Number of iterations used for the timing measurement.  Kept small so the
# script finishes quickly; per-iteration cost is what matters.
DEFAULT_LBFGS_ITERS: int = 20
DEFAULT_NNCG_ITERS: int = 5

# NNCG hyperparameters (Algorithm 4 defaults from the paper).
DEFAULT_NNCG_MU: float = 1e-2
DEFAULT_NNCG_S: int = 60
DEFAULT_NNCG_F: int = 20
DEFAULT_NNCG_M: int = 1000
DEFAULT_NNCG_EPS: float = 1e-16
DEFAULT_NNCG_ETA: float = 1.0
DEFAULT_NNCG_ALPHA: float = 0.1
DEFAULT_NNCG_BETA: float = 0.5

# Reference values from the paper (Table 3).
PAPER_TABLE3: Dict[str, Dict[str, float]] = {
    "convection": {"lbfgs": 4.6e-2, "nncg": 2.5e-1, "ratio": 5.43},
    "reaction": {"lbfgs": 3.6e-2, "nncg": 7.2e-1, "ratio": 20.0},
    "wave": {"lbfgs": 9.0e-2, "nncg": 2.9e1, "ratio": 322.22},
}

PDE_LABEL: Dict[str, str] = {
    "convection": "Convection",
    "reaction": "Reaction",
    "wave": "Wave",
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
        except Exception:  # pragma: no cover - defensive
            return {}
    return {}


def _resolve_lr(pde_name: str, cfg: Dict[str, Any], cli_lr: Optional[float]) -> float:
    """Resolve the learning rate from CLI, config, or defaults."""
    if cli_lr is not None:
        return float(cli_lr)
    best = cfg.get("best_lr", {}) if isinstance(cfg, dict) else {}
    if isinstance(best, dict) and "adam_lbfgs" in best:
        return float(best["adam_lbfgs"])
    return float(DEFAULT_BEST_LR.get(pde_name, 1e-3))


def _build_problem(
    pde_name: str,
    width: int,
    seed: int,
    device: torch.device,
    pde_kwargs: Optional[Dict[str, Any]] = None,
):
    """Instantiate PDE, model and data for a given experiment."""
    pde = get_pde(pde_name, **(pde_kwargs or {}))
    model = build_model(width=width, depth=3, seed=seed, device=device)
    data = build_data(pde, seed=seed, device=device)
    return pde, model, data


def _sync(device: torch.device) -> None:
    """Synchronise CUDA so wall-clock timings are accurate."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# ---------------------------------------------------------------------------
# Timing routines
# ---------------------------------------------------------------------------
def time_lbfgs(
    model,
    loss_fn,
    iters: int = DEFAULT_LBFGS_ITERS,
    memory: int = DEFAULT_LBFGS_MEMORY,
    lr: float = 1.0,
    device: Optional[torch.device] = None,
) -> float:
    """Measure the average per-iteration wall-clock time of L-BFGS.

    Returns the mean seconds per L-BFGS iteration.
    """
    device = device or torch.device("cpu")
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=1,
        max_eval=5,
        history_size=memory,
        line_search_fn="strong_wolfe",
    )

    # Warm-up (compilation / caching) - not timed.
    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn()
        loss.backward()
        return loss

    _sync(device)
    optimizer.step(closure)
    _sync(device)

    start = time.perf_counter()
    for _ in range(int(iters)):
        optimizer.step(closure)
    _sync(device)
    elapsed = time.perf_counter() - start
    return elapsed / max(1, int(iters))


def time_nncg(
    model,
    loss_fn,
    iters: int = DEFAULT_NNCG_ITERS,
    mu: float = DEFAULT_NNCG_MU,
    s: int = DEFAULT_NNCG_S,
    F: int = DEFAULT_NNCG_F,
    M: int = DEFAULT_NNCG_M,
    eps: float = DEFAULT_NNCG_EPS,
    eta: float = DEFAULT_NNCG_ETA,
    alpha: float = DEFAULT_NNCG_ALPHA,
    beta: float = DEFAULT_NNCG_BETA,
    device: Optional[torch.device] = None,
) -> float:
    """Measure the average per-iteration wall-clock time of NNCG.

    Returns the mean seconds per NNCG iteration.  Note that NNCG iterations are
    much more expensive than L-BFGS iterations because each one requires a
    randomized Nyström approximation (``s`` Hessian-vector products) plus up to
    ``M`` preconditioned CG iterations, each of which is itself a Hessian-vector
    product.
    """
    device = device or torch.device("cpu")
    optimizer = NysNewtonCG(
        model=model,
        loss_fn=loss_fn,
        mu=mu,
        s=s,
        F=F,
        K=int(iters),
        M=M,
        eps=eps,
        eta=eta,
        alpha=alpha,
        beta=beta,
        verbose=False,
        log_every=max(1, int(iters)),
    )

    # Warm-up iteration (not timed).
    _sync(device)
    optimizer.step()
    _sync(device)

    start = time.perf_counter()
    for _ in range(int(iters)):
        optimizer.step()
    _sync(device)
    elapsed = time.perf_counter() - start
    return elapsed / max(1, int(iters))


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------
def measure_pde(
    pde_name: str,
    width: int = DEFAULT_WIDTH,
    seed: int = DEFAULT_SEED,
    lr: Optional[float] = None,
    switch: int = DEFAULT_SWITCH,
    total_iters: int = DEFAULT_TOTAL_ITERS,
    lbfgs_memory: int = DEFAULT_LBFGS_MEMORY,
    lbfgs_iters: int = DEFAULT_LBFGS_ITERS,
    nncg_iters: int = DEFAULT_NNCG_ITERS,
    nncg_mu: float = DEFAULT_NNCG_MU,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Train to the switch point, then time L-BFGS and NNCG per iteration."""
    logger = get_logger()
    device = device or get_device()
    cfg = _load_pde_config(pde_name)
    pde_kwargs = cfg.get("pde_kwargs", {}) if isinstance(cfg, dict) else {}
    lr = _resolve_lr(pde_name, cfg, lr)

    set_seed(seed)
    pde, model, data = _build_problem(pde_name, width, seed, device, pde_kwargs)
    loss_fn = make_loss_fn(model, pde, data)

    if verbose:
        logger.info(
            "[%s] training Adam+L-BFGS to switch point (%d iters, lr=%g, width=%d)",
            pde_name,
            switch,
            lr,
            width,
        )

    # Reach the analysis point (Adam -> L-BFGS switch).
    train_adam_lbfgs(
        model,
        loss_fn,
        adam_lr=lr,
        switch_iter=switch,
        total_iters=switch,
        lbfgs_memory=lbfgs_memory,
        verbose=False,
        log_every=max(1, switch),
    )

    # --- Time L-BFGS -------------------------------------------------------
    if verbose:
        logger.info("[%s] timing L-BFGS (%d iters)", pde_name, lbfgs_iters)
    lbfgs_time = time_lbfgs(
        model,
        loss_fn,
        iters=lbfgs_iters,
        memory=lbfgs_memory,
        device=device,
    )

    # --- Time NNCG ---------------------------------------------------------
    if verbose:
        logger.info("[%s] timing NNCG (%d iters)", pde_name, nncg_iters)
    nncg_time = time_nncg(
        model,
        loss_fn,
        iters=nncg_iters,
        mu=nncg_mu,
        device=device,
    )

    ratio = nncg_time / lbfgs_time if lbfgs_time > 0 else float("inf")

    return {
        "pde": pde_name,
        "width": width,
        "seed": seed,
        "lr": lr,
        "switch": switch,
        "lbfgs_iters": int(lbfgs_iters),
        "nncg_iters": int(nncg_iters),
        "lbfgs_time": float(lbfgs_time),
        "nncg_time": float(nncg_time),
        "ratio": float(ratio),
    }


def run_table3(
    pdes: Optional[List[str]] = None,
    width: int = DEFAULT_WIDTH,
    seed: int = DEFAULT_SEED,
    lr: Optional[float] = None,
    switch: int = DEFAULT_SWITCH,
    total_iters: int = DEFAULT_TOTAL_ITERS,
    lbfgs_memory: int = DEFAULT_LBFGS_MEMORY,
    lbfgs_iters: int = DEFAULT_LBFGS_ITERS,
    nncg_iters: int = DEFAULT_NNCG_ITERS,
    nncg_mu: float = DEFAULT_NNCG_MU,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Measure per-iteration timings for each PDE and return the records."""
    pdes = pdes or list(DEFAULT_PDES)
    device = device or get_device()
    results: List[Dict[str, Any]] = []
    for pde_name in pdes:
        rec = measure_pde(
            pde_name,
            width=width,
            seed=seed,
            lr=lr,
            switch=switch,
            total_iters=total_iters,
            lbfgs_memory=lbfgs_memory,
            lbfgs_iters=lbfgs_iters,
            nncg_iters=nncg_iters,
            nncg_mu=nncg_mu,
            device=device,
            verbose=verbose,
        )
        results.append(rec)
    return results


# ---------------------------------------------------------------------------
# Formatting / comparison
# ---------------------------------------------------------------------------
def build_table(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Convert the flat list of records into ``{pde: {lbfgs, nncg, ratio}}``."""
    table: Dict[str, Dict[str, float]] = {}
    for rec in results:
        table[rec["pde"]] = {
            "lbfgs": float(rec["lbfgs_time"]),
            "nncg": float(rec["nncg_time"]),
            "ratio": float(rec["ratio"]),
        }
    return table


def format_table(table: Dict[str, Dict[str, float]]) -> str:
    """Render the measured Table 3 as text."""
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("Table 3: Per-iteration wall-clock time (seconds)")
    lines.append("=" * 72)
    header = f"{'PDE':<14}{'L-BFGS (s)':>16}{'NNCG (s)':>16}{'ratio':>16}"
    lines.append(header)
    lines.append("-" * 72)
    for pde_name in DEFAULT_PDES:
        if pde_name not in table:
            continue
        row = table[pde_name]
        lines.append(
            f"{PDE_LABEL.get(pde_name, pde_name):<14}"
            f"{row['lbfgs']:>16.4g}"
            f"{row['nncg']:>16.4g}"
            f"{row['ratio']:>16.2f}"
        )
    lines.append("=" * 72)
    return "\n".join(lines)


def format_comparison(table: Dict[str, Dict[str, float]]) -> str:
    """Compare measured timings against the paper's reference values."""
    lines: List[str] = []
    lines.append("")
    lines.append("Comparison against paper (Table 3):")
    lines.append("-" * 72)
    lines.append(
        f"{'PDE':<14}{'L-BFGS meas/paper':>24}{'NNCG meas/paper':>24}{'ratio meas/paper':>22}"
    )
    lines.append("-" * 72)
    for pde_name in DEFAULT_PDES:
        if pde_name not in table:
            continue
        meas = table[pde_name]
        ref = PAPER_TABLE3.get(pde_name, {})
        lines.append(
            f"{PDE_LABEL.get(pde_name, pde_name):<14}"
            f"{meas['lbfgs']:.3g} / {ref.get('lbfgs', float('nan')):.3g}".rjust(24)
            + f"{meas['nncg']:.3g} / {ref.get('nncg', float('nan')):.3g}".rjust(24)
            + f"{meas['ratio']:.2f} / {ref.get('ratio', float('nan')):.2f}".rjust(22)
        )
    lines.append("-" * 72)
    lines.append(
        "Note: absolute wall-clock times are hardware dependent; the paper used a "
        "single NVIDIA Titan V GPU."
    )
    return "\n".join(lines)


def check_trends(table: Dict[str, Dict[str, float]]) -> str:
    """Verify the qualitative trends expected from Table 3.

    Expected trends:
      * NNCG is slower per iteration than L-BFGS for every PDE.
      * The NNCG/L-BFGS ratio increases from convection -> reaction -> wave.
    """
    lines: List[str] = []
    lines.append("")
    lines.append("Qualitative trend check:")
    lines.append("-" * 72)

    ok_slower = True
    for pde_name in DEFAULT_PDES:
        if pde_name not in table:
            continue
        row = table[pde_name]
        slower = row["nncg"] > row["lbfgs"]
        ok_slower = ok_slower and slower
        lines.append(
            f"  {PDE_LABEL.get(pde_name, pde_name):<12} NNCG slower than L-BFGS: "
            f"{'YES' if slower else 'NO'} (ratio={row['ratio']:.2f})"
        )

    present = [p for p in DEFAULT_PDES if p in table]
    ratios = [table[p]["ratio"] for p in present]
    increasing = all(ratios[i] < ratios[i + 1] for i in range(len(ratios) - 1))
    lines.append(
        f"  Ratio increases convection -> reaction -> wave: "
        f"{'YES' if increasing else 'NO'} ({', '.join(f'{r:.2f}' for r in ratios)})"
    )
    lines.append("-" * 72)
    lines.append(
        f"  Overall: {'PASS' if (ok_slower and increasing) else 'PARTIAL'}"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_results(results: List[Dict[str, Any]], out_dir: str) -> str:
    """Persist the timing records to ``<out_dir>/table3.json``."""
    ensure_dir(out_dir)
    path = os.path.join(out_dir, "table3.json")
    save_json(results, path)
    return path


def load_results(out_dir: str) -> Optional[List[Dict[str, Any]]]:
    """Load previously saved timing records, if any."""
    path = os.path.join(out_dir, "table3.json")
    if not os.path.isfile(path):
        return None
    try:
        data = load_json(path)
    except Exception:  # pragma: no cover - defensive
        return None
    if isinstance(data, dict) and "results" in data:
        data = data["results"]
    if isinstance(data, list):
        return data
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce Table 3 (per-iteration wall-clock times) of the "
        "PINN optimizer paper.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Actually measure timings (trains to the switch point first). "
        "Without this flag, previously saved results are re-printed.",
    )
    parser.add_argument(
        "--pdes",
        nargs="+",
        default=None,
        help=f"PDEs to measure (default: {DEFAULT_PDES}).",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Adam learning rate (default: best_lr from config).",
    )
    parser.add_argument("--switch", type=int, default=DEFAULT_SWITCH)
    parser.add_argument("--total-iters", type=int, default=DEFAULT_TOTAL_ITERS)
    parser.add_argument("--lbfgs-memory", type=int, default=DEFAULT_LBFGS_MEMORY)
    parser.add_argument(
        "--lbfgs-iters",
        type=int,
        default=DEFAULT_LBFGS_ITERS,
        help="Number of L-BFGS iterations to time.",
    )
    parser.add_argument(
        "--nncg-iters",
        type=int,
        default=DEFAULT_NNCG_ITERS,
        help="Number of NNCG iterations to time.",
    )
    parser.add_argument("--nncg-mu", type=float, default=DEFAULT_NNCG_MU)
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory for saved results (default: results/table3).",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = get_logger()

    out_dir = args.out_dir or str(results_dir("table3"))
    device = torch.device("cpu") if args.cpu else get_device()

    if args.run:
        logger.info("Measuring per-iteration timings on %s", device)
        results = run_table3(
            pdes=args.pdes,
            width=args.width,
            seed=args.seed,
            lr=args.lr,
            switch=args.switch,
            total_iters=args.total_iters,
            lbfgs_memory=args.lbfgs_memory,
            lbfgs_iters=args.lbfgs_iters,
            nncg_iters=args.nncg_iters,
            nncg_mu=args.nncg_mu,
            device=device,
            verbose=args.verbose,
        )
        path = save_results(results, out_dir)
        logger.info("Saved timing results to %s", path)
    else:
        results = load_results(out_dir)
        if results is None:
            logger.warning(
                "No saved results found in %s. Run with --run to measure timings.",
                out_dir,
            )
            # Fall back to the paper's reference values so the table can still
            # be printed for inspection.
            results = [
                {
                    "pde": pde_name,
                    "lbfgs_time": PAPER_TABLE3[pde_name]["lbfgs"],
                    "nncg_time": PAPER_TABLE3[pde_name]["nncg"],
                    "ratio": PAPER_TABLE3[pde_name]["ratio"],
                }
                for pde_name in DEFAULT_PDES
            ]
            logger.info("Printing paper reference values instead.")

    table = build_table(results)
    print(format_table(table))
    print(format_comparison(table))
    print(check_trends(table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
