"""Reproduce Figure 8 of the paper.

Figure 8 shows, for each optimizer (Adam, L-BFGS, Adam+L-BFGS), the
minimum / median / maximum final loss and L2 relative error (L2RE) as a
function of network width.  The expectation from the paper is that
Adam+L-BFGS achieves the best minimum (always) and best median (nearly
always), with the only exceptions being the reaction PDE at width=100
(loss) and width=200 (L2RE).

This script is read-only by default: it consumes the per-run
``result.json`` artifacts produced by ``run_experiments.py`` and renders
the figure.  Passing ``--run`` first executes the experiment grid.

Usage
-----
    python scripts/reproduce_fig8.py                 # plot from existing results
    python scripts/reproduce_fig8.py --run           # run grid then plot
    python scripts/reproduce_fig8.py --metric l2re   # plot L2RE only
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Make ``src`` and the repo root importable when run as a script.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.utils import ensure_dir, get_logger, load_json, results_dir  # noqa: E402

logger = get_logger("reproduce_fig8")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PDES = ["convection", "reaction", "wave"]
OPTIMIZERS = ["adam", "lbfgs", "adam_lbfgs"]
DEFAULT_WIDTHS = [50, 100, 200, 400]

OPT_LABEL = {
    "adam": "Adam",
    "lbfgs": "L-BFGS",
    "adam_lbfgs": "Adam+L-BFGS",
}

# Consistent styling across reproduction scripts.
OPT_STYLE = {
    "adam": {"color": "#1f77b4", "marker": "o"},
    "lbfgs": {"color": "#ff7f0e", "marker": "s"},
    "adam_lbfgs": {"color": "#2ca02c", "marker": "^"},
}

METRIC_LABEL = {
    "loss": "Loss",
    "l2re": "L2 relative error",
}


# ---------------------------------------------------------------------------
# Result collection
# ---------------------------------------------------------------------------

def collect_results(
    root: str = "results",
    pdes: Optional[Sequence[str]] = None,
    optimizers: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Walk the results tree and return a flat list of run records.

    Each record is the parsed ``result.json`` augmented with ``pde``,
    ``optimizer`` and ``run`` keys.
    """
    pdes = list(pdes) if pdes else list(DEFAULT_PDES)
    optimizers = list(optimizers) if optimizers else list(OPTIMIZERS)

    runs: List[Dict[str, Any]] = []
    for pde in pdes:
        for opt in optimizers:
            opt_dir = os.path.join(root, pde, opt)
            if not os.path.isdir(opt_dir):
                continue
            for run_name in sorted(os.listdir(opt_dir)):
                run_dir = os.path.join(opt_dir, run_name)
                result_path = os.path.join(run_dir, "result.json")
                if not os.path.isfile(result_path):
                    continue
                try:
                    rec = load_json(result_path)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("Failed to load %s: %s", result_path, exc)
                    continue
                rec.setdefault("pde", pde)
                rec.setdefault("optimizer", opt)
                rec["run"] = run_name
                runs.append(rec)
    return runs


def _finite(value: Any) -> bool:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return False
    return f == f and f not in (float("inf"), float("-inf"))


def filter_runs(runs: Sequence[Dict[str, Any]], metric: str) -> List[Dict[str, Any]]:
    """Keep runs with a finite, strictly positive value for ``metric``."""
    out = []
    for r in runs:
        v = r.get(metric)
        if _finite(v) and float(v) > 0.0:
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_by_width(
    runs: Sequence[Dict[str, Any]],
    metric: str,
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Aggregate ``metric`` into min/median/max per (optimizer, width).

    Returns ``{optimizer: {width: {"min": .., "median": .., "max": .., "n": ..}}}``.
    """
    buckets: Dict[str, Dict[int, List[float]]] = {}
    for r in runs:
        opt = r.get("optimizer")
        width = r.get("width")
        val = r.get(metric)
        if opt is None or width is None or not _finite(val):
            continue
        buckets.setdefault(opt, {}).setdefault(int(width), []).append(float(val))

    agg: Dict[str, Dict[int, Dict[str, float]]] = {}
    for opt, by_width in buckets.items():
        agg[opt] = {}
        for width, vals in by_width.items():
            vals_sorted = sorted(vals)
            n = len(vals_sorted)
            if n % 2 == 1:
                median = vals_sorted[n // 2]
            else:
                median = 0.5 * (vals_sorted[n // 2 - 1] + vals_sorted[n // 2])
            agg[opt][width] = {
                "min": vals_sorted[0],
                "median": median,
                "max": vals_sorted[-1],
                "n": float(n),
            }
    return agg


def summarize(agg: Dict[str, Dict[int, Dict[str, float]]], metric: str) -> str:
    """Human-readable summary of the aggregated statistics."""
    lines = [f"Figure 8 summary ({METRIC_LABEL.get(metric, metric)}):"]
    widths = sorted({w for by_w in agg.values() for w in by_w})
    for width in widths:
        lines.append(f"  width={width}")
        for opt in OPTIMIZERS:
            stats = agg.get(opt, {}).get(width)
            if not stats:
                continue
            lines.append(
                "    {:<12} min={:.3e}  median={:.3e}  max={:.3e}  (n={:.0f})".format(
                    OPT_LABEL.get(opt, opt),
                    stats["min"],
                    stats["median"],
                    stats["max"],
                    stats["n"],
                )
            )
    return "\n".join(lines)


def check_trends(agg: Dict[str, Dict[int, Dict[str, float]]]) -> str:
    """Qualitative check: Adam+L-BFGS should be best (min) at every width."""
    lines = ["Trend check (Adam+L-BFGS should have the lowest min):"]
    widths = sorted({w for by_w in agg.values() for w in by_w})
    for width in widths:
        best_opt = None
        best_val = float("inf")
        for opt in OPTIMIZERS:
            stats = agg.get(opt, {}).get(width)
            if stats and stats["min"] < best_val:
                best_val = stats["min"]
                best_opt = opt
        status = "OK" if best_opt == "adam_lbfgs" else "MISMATCH"
        lines.append(
            "  width={:<4} best_min={:<12} [{:.3e}]  {}".format(
                width, OPT_LABEL.get(best_opt, str(best_opt)), best_val, status
            )
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_figure8(
    agg: Dict[str, Dict[int, Dict[str, float]]],
    metric: str = "loss",
    out_path: str = "fig8.png",
    log_y: bool = True,
    title: Optional[str] = None,
) -> str:
    """Render the min/median/max-vs-width figure and save it as a PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    widths = sorted({w for by_w in agg.values() for w in by_w})
    if not widths:
        raise ValueError("No data available to plot Figure 8.")

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    for opt in OPTIMIZERS:
        by_width = agg.get(opt)
        if not by_width:
            continue
        style = OPT_STYLE.get(opt, {})
        color = style.get("color", None)
        marker = style.get("marker", "o")

        xs = [w for w in widths if w in by_width]
        mins = [by_width[w]["min"] for w in xs]
        meds = [by_width[w]["median"] for w in xs]
        maxs = [by_width[w]["max"] for w in xs]

        ax.plot(xs, meds, marker=marker, color=color, label=OPT_LABEL.get(opt, opt))
        ax.fill_between(xs, mins, maxs, color=color, alpha=0.15)

    ax.set_xlabel("Network width")
    ax.set_ylabel(METRIC_LABEL.get(metric, metric))
    ax.set_xticks(widths)
    if log_y:
        ax.set_yscale("log")
    ax.set_title(title or f"Figure 8: {METRIC_LABEL.get(metric, metric)} vs width")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ensure_dir(os.path.dirname(os.path.abspath(out_path)))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce Figure 8 (min/median/max loss & L2RE vs width)."
    )
    parser.add_argument("--root", type=str, default="results",
                        help="Root results directory.")
    parser.add_argument("--pdes", type=str, nargs="+", default=None,
                        help="PDEs to include (default: all three).")
    parser.add_argument("--optimizers", type=str, nargs="+", default=None,
                        help="Optimizers to include (default: all three).")
    parser.add_argument("--metric", type=str, default="both",
                        choices=["loss", "l2re", "both"],
                        help="Which metric to plot.")
    parser.add_argument("--out", type=str, default=None,
                        help="Output PNG path (default: results/fig8_<metric>.png).")
    parser.add_argument("--linear-axes", action="store_true",
                        help="Use linear y-axis instead of log.")
    parser.add_argument("--run", action="store_true",
                        help="Run the experiment grid before plotting.")
    parser.add_argument("--widths", type=int, nargs="+", default=None,
                        help="Widths for the grid run (default: 50 100 200 400).")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Seeds for the grid run (default: 0 1 2 3 4).")
    parser.add_argument("--no-tune", action="store_true",
                        help="Skip learning-rate tuning during the grid run.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.run:
        from run_experiments import run_grid

        widths = args.widths if args.widths else DEFAULT_WIDTHS
        seeds = args.seeds if args.seeds else [0, 1, 2, 3, 4]
        run_grid(
            pdes=args.pdes if args.pdes else DEFAULT_PDES,
            optimizers=args.optimizers if args.optimizers else OPTIMIZERS,
            widths=widths,
            seeds=seeds,
            root=args.root,
            tune=not args.no_tune,
        )

    runs = collect_results(root=args.root, pdes=args.pdes, optimizers=args.optimizers)
    if not runs:
        logger.error("No results found under %s. Run with --run first.", args.root)
        return 1

    metrics = ["loss", "l2re"] if args.metric == "both" else [args.metric]
    log_y = not args.linear_axes

    for metric in metrics:
        filtered = filter_runs(runs, metric)
        if not filtered:
            logger.warning("No finite positive values for metric '%s'.", metric)
            continue
        agg = aggregate_by_width(filtered, metric)
        print(summarize(agg, metric))
        print(check_trends(agg))
        print()

        if args.out and len(metrics) == 1:
            out_path = args.out
        else:
            out_dir = results_dir(args.root)
            out_path = str(out_dir / f"fig8_{metric}.png")
        saved = plot_figure8(agg, metric=metric, out_path=out_path, log_y=log_y)
        logger.info("Saved %s", saved)
        print(f"Saved figure to {saved}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
