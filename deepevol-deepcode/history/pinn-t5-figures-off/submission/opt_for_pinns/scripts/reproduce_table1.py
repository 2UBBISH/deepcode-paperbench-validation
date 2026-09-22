"""Reproduce Table 1: best loss & L2RE per optimizer across widths.

Table 1 of "Challenges in Training PINNs: A Loss Landscape Perspective"
reports, for each PDE (convection, reaction, wave) and each optimizer
(Adam, L-BFGS, Adam+L-BFGS), the best (minimum) final loss and the
corresponding L2 relative error, aggregated over widths {50,100,200,400}
and 5 seeds.

This script consumes the artifacts produced by ``run_experiments.py``
(``results/<pde>/<optimizer>/w<width>_s<seed>/result.json`` and the
aggregated ``results/summary.json``).  If no results are present it can
optionally run the grid itself (``--run``).

Usage
-----
    python scripts/reproduce_table1.py                 # read existing results
    python scripts/reproduce_table1.py --run           # run grid then tabulate
    python scripts/reproduce_table1.py --pdes convection --widths 50 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

# Make ``src`` importable when running as a script.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils import ensure_dir, get_logger, load_json, results_dir  # noqa: E402

LOGGER = get_logger("reproduce_table1")

# Reference values from the paper (Table 1) for qualitative comparison.
PAPER_TABLE1 = {
    "convection": {
        "adam": (1.40e-4, 5.96e-2),
        "lbfgs": (1.51e-5, 8.26e-3),
        "adam_lbfgs": (5.95e-6, 4.19e-3),
    },
    "reaction": {
        "adam": (4.73e-6, 2.12e-2),
        "lbfgs": (8.93e-6, 3.83e-2),
        "adam_lbfgs": (3.26e-6, 1.92e-2),
    },
    "wave": {
        "adam": (2.03e-2, 3.49e-1),
        "lbfgs": (1.84e-2, 3.35e-1),
        "adam_lbfgs": (1.12e-3, 5.52e-2),
    },
}

OPTIMIZERS = ["adam", "lbfgs", "adam_lbfgs"]
OPT_LABEL = {"adam": "Adam", "lbfgs": "L-BFGS", "adam_lbfgs": "Adam+L-BFGS"}


# ---------------------------------------------------------------------------
# Result collection
# ---------------------------------------------------------------------------
def _iter_run_dirs(root: str, pde: str, optimizer: str) -> List[str]:
    base = os.path.join(root, pde, optimizer)
    if not os.path.isdir(base):
        return []
    out = []
    for name in sorted(os.listdir(base)):
        path = os.path.join(base, name)
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "result.json")):
            out.append(path)
    return out


def collect_results(
    root: str = "results",
    pdes: Optional[List[str]] = None,
    optimizers: Optional[List[str]] = None,
    widths: Optional[List[int]] = None,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Collect per-run results grouped by (pde, optimizer)."""
    pdes = pdes or ["convection", "reaction", "wave"]
    optimizers = optimizers or OPTIMIZERS
    collected: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for pde in pdes:
        collected[pde] = {}
        for opt in optimizers:
            runs = []
            for run_dir in _iter_run_dirs(root, pde, opt):
                try:
                    res = load_json(os.path.join(run_dir, "result.json"))
                except Exception as exc:  # pragma: no cover - defensive
                    LOGGER.warning("Failed to load %s: %s", run_dir, exc)
                    continue
                if widths is not None and res.get("width") not in widths:
                    continue
                runs.append(res)
            collected[pde][opt] = runs
    return collected


def best_per_optimizer(runs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the run with the smallest final loss (paper's selection rule)."""
    if not runs:
        return None
    return min(runs, key=lambda r: r.get("loss", float("inf")))


def build_table(
    collected: Dict[str, Dict[str, List[Dict[str, Any]]]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    table: Dict[str, Dict[str, Dict[str, float]]] = {}
    for pde, by_opt in collected.items():
        table[pde] = {}
        for opt, runs in by_opt.items():
            best = best_per_optimizer(runs)
            if best is None:
                continue
            table[pde][opt] = {
                "loss": float(best.get("loss", float("nan"))),
                "l2re": float(best.get("l2re", float("nan"))),
                "width": int(best.get("width", -1)),
                "seed": int(best.get("seed", -1)),
                "lr": float(best.get("lr", float("nan"))),
                "n_runs": len(runs),
            }
    return table


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def format_table(table: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    lines = []
    header = f"{'PDE':<12} {'Optimizer':<14} {'Loss':>12} {'L2RE':>12} {'Width':>6} {'Seed':>5} {'LR':>9}"
    lines.append(header)
    lines.append("-" * len(header))
    for pde in ["convection", "reaction", "wave"]:
        if pde not in table:
            continue
        for opt in OPTIMIZERS:
            if opt not in table[pde]:
                continue
            row = table[pde][opt]
            lines.append(
                f"{pde:<12} {OPT_LABEL.get(opt, opt):<14} "
                f"{row['loss']:>12.3e} {row['l2re']:>12.3e} "
                f"{row['width']:>6d} {row['seed']:>5d} {row['lr']:>9.1e}"
            )
        lines.append("")
    return "\n".join(lines)


def format_comparison(table: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    lines = ["Comparison against paper Table 1 (best loss, best L2RE):", ""]
    for pde in ["convection", "reaction", "wave"]:
        if pde not in table:
            continue
        lines.append(f"[{pde}]")
        for opt in OPTIMIZERS:
            if opt not in table[pde]:
                continue
            got = table[pde][opt]
            ref = PAPER_TABLE1.get(pde, {}).get(opt)
            if ref is None:
                continue
            lines.append(
                f"  {OPT_LABEL.get(opt, opt):<14} "
                f"loss {got['loss']:.3e} (paper {ref[0]:.3e}) | "
                f"L2RE {got['l2re']:.3e} (paper {ref[1]:.3e})"
            )
        lines.append("")
    return "\n".join(lines)


def check_ordering(table: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    """Verify the paper's claim that Adam+L-BFGS is best on all PDEs."""
    lines = ["Ordering check (Adam+L-BFGS should have lowest loss & L2RE):"]
    for pde in ["convection", "reaction", "wave"]:
        if pde not in table:
            continue
        losses = {o: table[pde][o]["loss"] for o in table[pde]}
        l2res = {o: table[pde][o]["l2re"] for o in table[pde]}
        best_loss_opt = min(losses, key=losses.get)
        best_l2re_opt = min(l2res, key=l2res.get)
        lines.append(
            f"  {pde:<12} best-loss={OPT_LABEL.get(best_loss_opt, best_loss_opt):<14} "
            f"best-L2RE={OPT_LABEL.get(best_l2re_opt, best_l2re_opt)}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reproduce Table 1 (best loss/L2RE per optimizer).")
    p.add_argument("--root", type=str, default="results", help="Results root directory.")
    p.add_argument("--pdes", nargs="+", default=["convection", "reaction", "wave"])
    p.add_argument("--optimizers", nargs="+", default=OPTIMIZERS)
    p.add_argument("--widths", nargs="+", type=int, default=None)
    p.add_argument("--run", action="store_true", help="Run the experiment grid first.")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--total-iters", type=int, default=41000)
    p.add_argument("--switch-iter", type=int, default=11000)
    p.add_argument("--no-tune", action="store_true", help="Skip lr tuning (use cached/default).")
    p.add_argument("--out", type=str, default=None, help="Optional JSON output path.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.run:
        from run_experiments import run_grid

        LOGGER.info("Running experiment grid ...")
        run_grid(
            pdes=args.pdes,
            optimizers=args.optimizers,
            widths=args.widths or [50, 100, 200, 400],
            seeds=args.seeds,
            root=args.root,
            total_iters=args.total_iters,
            switch_iter=args.switch_iter,
            tune=not args.no_tune,
        )

    collected = collect_results(
        root=args.root,
        pdes=args.pdes,
        optimizers=args.optimizers,
        widths=args.widths,
    )
    table = build_table(collected)

    print("=" * 78)
    print("Table 1: Best loss & L2RE per optimizer (aggregated over widths & seeds)")
    print("=" * 78)
    print(format_table(table))
    print(format_comparison(table))
    print(check_ordering(table))

    if args.out:
        ensure_dir(os.path.dirname(os.path.abspath(args.out)))
        with open(args.out, "w") as fh:
            json.dump(table, fh, indent=2)
        LOGGER.info("Wrote table to %s", args.out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
