"""Reproduce Figure 2 of the paper.

Figure 2 is a scatter plot of the final L2 relative error (L2RE) versus the
final training loss for every run in the experiment grid (PDE x optimizer x
width x seed).  The expected qualitative trend is that lower loss generally
corresponds to lower L2RE (a monotone-ish relationship), which supports the
claim that the PINN loss is a reasonable proxy for solution accuracy even
though it is ill-conditioned.

This script is read-only by default: it consumes the per-run ``result.json``
artifacts produced by ``run_experiments.py`` and renders a scatter plot.  Pass
``--run`` to first execute the experiment grid.

Usage
-----
    python scripts/reproduce_fig2.py                 # plot from existing results
    python scripts/reproduce_fig2.py --run           # run grid, then plot
    python scripts/reproduce_fig2.py --out fig2.png
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Make ``src`` and the repo root importable when run as a script.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils import ensure_dir, get_logger, load_json, results_dir  # noqa: E402

logger = get_logger("reproduce_fig2")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPTIMIZERS: List[str] = ["adam", "lbfgs", "adam_lbfgs"]

OPT_LABEL: Dict[str, str] = {
    "adam": "Adam",
    "lbfgs": "L-BFGS",
    "adam_lbfgs": "Adam+L-BFGS",
}

# Distinct marker/colour per optimizer so the scatter is readable.
OPT_STYLE: Dict[str, Dict[str, Any]] = {
    "adam": {"marker": "o", "color": "tab:blue"},
    "lbfgs": {"marker": "s", "color": "tab:orange"},
    "adam_lbfgs": {"marker": "^", "color": "tab:green"},
}

# Distinct marker edge per PDE (filled vs hollow) so PDEs are distinguishable.
PDE_MARKERS: Dict[str, str] = {
    "convection": "o",
    "reaction": "s",
    "wave": "^",
}

DEFAULT_PDES: List[str] = ["convection", "reaction", "wave"]


# ---------------------------------------------------------------------------
# Result collection
# ---------------------------------------------------------------------------


def collect_results(
    root: str = "results",
    pdes: Optional[List[str]] = None,
    optimizers: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Collect every per-run result dict found under ``root``.

    Each run lives at ``<root>/<pde>/<optimizer>/w<width>_s<seed>/result.json``
    and is expected to contain at least the keys ``loss`` and ``l2re``.

    Returns a flat list of dicts, each augmented with ``pde`` and ``optimizer``
    keys for convenience.
    """
    pdes = pdes or DEFAULT_PDES
    optimizers = optimizers or OPTIMIZERS

    runs: List[Dict[str, Any]] = []
    for pde in pdes:
        for opt in optimizers:
            opt_dir = os.path.join(root, pde, opt)
            if not os.path.isdir(opt_dir):
                continue
            for run_name in sorted(os.listdir(opt_dir)):
                run_dir = os.path.join(opt_dir, run_name)
                if not os.path.isdir(run_dir):
                    continue
                result_path = os.path.join(run_dir, "result.json")
                if not os.path.isfile(result_path):
                    continue
                try:
                    rec = load_json(result_path)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("Failed to read %s: %s", result_path, exc)
                    continue
                if "loss" not in rec or "l2re" not in rec:
                    continue
                rec.setdefault("pde", pde)
                rec.setdefault("optimizer", opt)
                rec.setdefault("run", run_name)
                runs.append(rec)

    logger.info("Collected %d runs from %s", len(runs), root)
    return runs


def _finite(rec: Dict[str, Any]) -> bool:
    """Return True if the run has finite, positive loss and L2RE values."""
    try:
        loss = float(rec["loss"])
        l2re = float(rec["l2re"])
    except (KeyError, TypeError, ValueError):
        return False
    if loss != loss or l2re != l2re:  # NaN check
        return False
    return loss > 0.0 and l2re > 0.0


def filter_runs(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only runs with finite, strictly positive loss and L2RE (log axes)."""
    kept = [r for r in runs if _finite(r)]
    dropped = len(runs) - len(kept)
    if dropped:
        logger.info("Dropped %d runs with non-positive/non-finite metrics", dropped)
    return kept


# ---------------------------------------------------------------------------
# Trend / correlation diagnostics
# ---------------------------------------------------------------------------


def spearman_correlation(xs: List[float], ys: List[float]) -> float:
    """Spearman rank correlation between two sequences.

    Implemented without scipy so the script has no hard scipy dependency.
    Returns 0.0 when the correlation is undefined (e.g. constant input).
    """
    n = len(xs)
    if n < 2:
        return 0.0

    def _ranks(vals: List[float]) -> List[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx = _ranks(xs)
    ry = _ranks(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    if dx == 0.0 or dy == 0.0:
        return 0.0
    return num / (dx * dy)


def trend_summary(runs: List[Dict[str, Any]]) -> str:
    """Human-readable summary of the loss -> L2RE trend."""
    xs = [float(r["loss"]) for r in runs]
    ys = [float(r["l2re"]) for r in runs]
    rho = spearman_correlation(xs, ys)
    lines = [
        "Figure 2 trend summary",
        "----------------------",
        f"n runs           : {len(runs)}",
        f"Spearman rho     : {rho:+.3f}  (positive => lower loss -> lower L2RE)",
    ]
    # Per-PDE breakdown.
    for pde in DEFAULT_PDES:
        sub = [r for r in runs if r.get("pde") == pde]
        if len(sub) < 2:
            continue
        sx = [float(r["loss"]) for r in sub]
        sy = [float(r["l2re"]) for r in sub]
        lines.append(f"  {pde:<11}: n={len(sub):<4} rho={spearman_correlation(sx, sy):+.3f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_figure2(
    runs: List[Dict[str, Any]],
    out_path: str = "fig2.png",
    log_axes: bool = True,
    title: str = "Figure 2: L2RE vs. final loss",
) -> str:
    """Render the L2RE-vs-loss scatter plot and save it to ``out_path``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 5.5))

    for pde in DEFAULT_PDES:
        for opt in OPTIMIZERS:
            sub = [r for r in runs if r.get("pde") == pde and r.get("optimizer") == opt]
            if not sub:
                continue
            xs = [float(r["loss"]) for r in sub]
            ys = [float(r["l2re"]) for r in sub]
            style = OPT_STYLE.get(opt, {})
            ax.scatter(
                xs,
                ys,
                marker=PDE_MARKERS.get(pde, "o"),
                color=style.get("color", None),
                alpha=0.7,
                s=28,
                edgecolors="k",
                linewidths=0.3,
                label=f"{pde} / {OPT_LABEL.get(opt, opt)}",
            )

    if log_axes:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel("Final training loss")
    ax.set_ylabel("Final L2 relative error")
    ax.set_title(title)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=7, ncol=2, loc="best")

    ensure_dir(os.path.dirname(os.path.abspath(out_path)))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved figure to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce Figure 2 (L2RE vs. loss scatter)."
    )
    parser.add_argument("--root", default="results", help="Results root directory.")
    parser.add_argument(
        "--pdes",
        nargs="+",
        default=DEFAULT_PDES,
        help="PDEs to include.",
    )
    parser.add_argument(
        "--optimizers",
        nargs="+",
        default=OPTIMIZERS,
        help="Optimizers to include.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output image path (default: results/figures/fig2.png).",
    )
    parser.add_argument(
        "--linear-axes",
        action="store_true",
        help="Use linear (instead of log) axes.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the experiment grid before plotting.",
    )
    parser.add_argument(
        "--widths",
        nargs="+",
        type=int,
        default=[50, 100, 200, 400],
        help="Widths for --run.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Seeds for --run.",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip learning-rate tuning when --run is given.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.run:
        from run_experiments import run_grid

        logger.info("Running experiment grid ...")
        run_grid(
            pdes=args.pdes,
            optimizers=args.optimizers,
            widths=args.widths,
            seeds=args.seeds,
            root=args.root,
            tune=not args.no_tune,
        )

    runs = filter_runs(collect_results(args.root, args.pdes, args.optimizers))
    if not runs:
        logger.error(
            "No runs found under %s. Run `python run_experiments.py` first "
            "(or pass --run).",
            args.root,
        )
        return 1

    print(trend_summary(runs))

    out_path = args.out
    if out_path is None:
        out_path = str(results_dir(args.root, "figures") / "fig2.png")

    plot_figure2(runs, out_path=out_path, log_axes=not args.linear_axes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
