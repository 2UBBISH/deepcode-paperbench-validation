"""Command line interface used to reproduce the experiments.

Examples
--------
Full reproduction of Section 6 (Table 1, Figures 2 and 8)::

    python -m pinn.cli grid

Fast smoke test of every code path (small widths, few iterations)::

    python -m pinn.cli all --quick

Spectral densities of Figures 3 and 7 for the configuration with the smallest
L2RE found by the grid search::

    python -m pinn.cli spectra --pde wave --select-best

Fine-tuning experiments of Section 7 (Figures 1, 4, 5 and Tables 2, 3)::

    python -m pinn.cli finetune --pde convection --select-best
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional, Sequence

from .experiments.analysis import figure8_stats, select_best_configuration, table1, write_csv
from .experiments.common import (
    ADAM_LRS,
    PDES,
    QUICK,
    SEEDS,
    SWITCHES,
    TOTAL_ITERS,
    WIDTHS,
    Paths,
    RunSpec,
    read_records,
)
from .experiments.finetune import MU_GRID, run_finetune_experiment
from .experiments.grid import run_grid
from .experiments.spectral import run_spectral_experiment
from .plotting import plot_bars, plot_loss_vs_l2re


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--outdir", default="runs", help="directory for results")
    parser.add_argument("--quick", action="store_true", help="tiny settings for a smoke test")
    parser.add_argument("--pdes", nargs="+", default=list(PDES))
    parser.add_argument("--widths", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--lrs", nargs="+", type=float)
    parser.add_argument("--iters", type=int)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--no-progress", action="store_true")


def _resolve(args) -> dict:
    if args.quick:
        return {
            "widths": list(QUICK["widths"]),
            "seeds": list(QUICK["seeds"]),
            "lrs": list(QUICK["lrs"]),
            "switches": list(QUICK["switches"]),
            "iters": QUICK["iters"],
        }
    return {
        "widths": args.widths or list(WIDTHS),
        "seeds": args.seeds or list(SEEDS),
        "lrs": args.lrs or list(ADAM_LRS),
        "switches": getattr(args, "switches", None) or list(SWITCHES),
        "iters": args.iters or TOTAL_ITERS,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pinn", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("grid", help="Section 6 hyper-parameter grid")
    _common(p)
    p.add_argument("--switches", nargs="+", type=int)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("analyze", help="Table 1, Figure 2 and Figure 8 from stored runs")
    p.add_argument("--outdir", default="runs")

    p = sub.add_parser("spectra", help="Figures 3 and 7")
    _common(p)
    p.add_argument("--pde", required=True)
    p.add_argument("--width", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--switch", type=int, default=11000)
    p.add_argument("--n-runs", type=int, default=1, help="number of Lanczos runs")
    p.add_argument("--n-lanczos", type=int, default=100, help="Lanczos iterations")
    p.add_argument(
        "--no-component",
        dest="component",
        action="store_false",
        help="skip the per-component spectra (bottom row of Figure 3 / Figure 7)",
    )
    p.set_defaults(component=True)
    p.add_argument("--select-best", action="store_true")

    p = sub.add_parser("finetune", help="Figures 1, 4, 5 and Tables 2, 3")
    _common(p)
    p.add_argument("--pde", required=True)
    p.add_argument("--width", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--switch", type=int, default=11000)
    p.add_argument("--nncg-iters", type=int, default=2000)
    p.add_argument("--gd-iters", type=int, default=2000)
    p.add_argument("--gd-lr", type=float, default=None)
    p.add_argument("--mu-grid", nargs="+", type=float, default=list(MU_GRID))
    p.add_argument("--sketch-size", type=int, default=60)
    p.add_argument("--cg-max-iter", type=int, default=1000)
    p.add_argument("--cg-rel-tol", type=float, default=None)
    p.add_argument("--select-best", action="store_true")

    p = sub.add_parser("all", help="run everything (grid -> analysis -> figures)")
    _common(p)
    p.add_argument("--switches", nargs="+", type=int)
    p.add_argument("--skip-grid", action="store_true")
    p.add_argument("--skip-spectra", action="store_true")
    p.add_argument("--skip-finetune", action="store_true")
    p.add_argument("--n-runs", type=int, default=1)
    p.add_argument("--n-lanczos", type=int, default=100)
    p.add_argument("--mu-grid", nargs="+", type=float, default=list(MU_GRID))
    p.add_argument("--sketch-size", type=int, default=60)
    p.add_argument("--cg-max-iter", type=int, default=1000)
    p.add_argument("--cg-rel-tol", type=float, default=None)
    p.add_argument("--select-best", action="store_true")
    return parser


def _best_spec(outdir: Paths, pde: str, switch: int, quick: bool) -> RunSpec:
    records = read_records(outdir)
    cands = [
        r
        for r in records
        if r["pde"] == pde and r["optimizer"] == "adam_lbfgs" and r["switch"] == switch
    ]
    if not cands:
        raise SystemExit(
            f"no Adam+L-BFGS({switch}) runs stored in {outdir.runs_jsonl}; "
            "run `python -m pinn.cli grid` first or pass --width/--seed/--lr"
        )
    best = select_best_configuration(cands, pde)
    return RunSpec(
        pde=pde,
        width=best["width"],
        seed=best["seed"],
        optimizer="adam_lbfgs",
        lr=best["lr"],
        switch=switch,
        iters=best["iters"],
    )


def cmd_grid(args) -> int:
    g = _resolve(args)
    specs = None
    if args.dry_run:
        from .experiments.grid import grid_specs

        specs = grid_specs(
            pdes=args.pdes,
            widths=g["widths"],
            seeds=g["seeds"],
            lrs=g["lrs"],
            switches=g["switches"],
            iters=g["iters"],
        )
        for s in specs[: args.limit] if args.limit else specs:
            print(s.run_id)
        print(f"{len(specs)} runs in total")
        return 0
    run_grid(
        Paths(args.outdir),
        pdes=args.pdes,
        widths=g["widths"],
        seeds=g["seeds"],
        lrs=g["lrs"],
        switches=g["switches"],
        iters=g["iters"],
        progress=not args.no_progress,
        limit=args.limit,
    )
    return 0


def cmd_analyze(args) -> int:
    outdir = Paths(args.outdir)
    records = read_records(outdir)
    if not records:
        print(f"no runs found in {outdir.runs_jsonl}", file=sys.stderr)
        return 1
    write_csv(table1(records), outdir.table("table1_best_optimizers.csv"))
    plot_loss_vs_l2re(records, outdir.figure("figure2_loss_vs_l2re.png"))
    stats = figure8_stats(records)
    for pde, per_opt in stats.items():
        for metric, key in (("Loss", "loss"), ("L2RE", "l2re")):
            plot_bars(
                {
                    label: {
                        "min": vals[f"min_{key}"],
                        "median": vals[f"median_{key}"],
                        "max": vals[f"max_{key}"],
                    }
                    for label, vals in per_opt.items()
                },
                metric,
                pde,
                outdir.figure(f"figure8_{pde}_{key}.png"),
                widths=next(iter(per_opt.values()))["widths"]
                if per_opt and "widths" in next(iter(per_opt.values()))
                else None,
            )
    with open(outdir.table("figure8_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"wrote tables and figures to {outdir.root}")
    return 0


def cmd_spectra(args) -> int:
    outdir = Paths(args.outdir)
    g = _resolve(args)
    spec = (
        _best_spec(outdir, args.pde, args.switch, args.quick)
        if args.select_best
        else RunSpec(
            args.pde,
            args.width or g["widths"][-1],
            args.seed if args.seed is not None else g["seeds"][0],
            "adam_lbfgs",
            args.lr or g["lrs"][-1],
            args.switch,
            g["iters"],
        )
    )
    print(f"spectral density for {spec.run_id}")
    run_spectral_experiment(
        spec.pde,
        spec.width,
        spec.seed,
        spec.lr,
        outdir,
        switch=spec.switch,
        iters=spec.iters,
        n_runs=args.n_runs,
        n_iter=args.n_lanczos,
        components=args.component,
        progress=not args.no_progress,
    )
    return 0


def cmd_finetune(args) -> int:
    outdir = Paths(args.outdir)
    g = _resolve(args)
    spec = (
        _best_spec(outdir, args.pde, args.switch, args.quick)
        if args.select_best
        else RunSpec(
            args.pde,
            args.width or g["widths"][-1],
            args.seed if args.seed is not None else g["seeds"][0],
            "adam_lbfgs",
            args.lr or g["lrs"][-1],
            args.switch,
            g["iters"],
        )
    )
    print(f"fine-tuning experiment for {spec.run_id}")
    run_finetune_experiment(
        spec.pde,
        spec.width,
        spec.seed,
        spec.lr,
        outdir,
        switch=spec.switch,
        iters=spec.iters,
        nncg_iters=args.nncg_iters,
        gd_iters=args.gd_iters,
        gd_lr=args.gd_lr,
        mu_grid=args.mu_grid,
        nncg_cfg={
            "sketch_size": args.sketch_size,
            "cg_max_iter": args.cg_max_iter,
            "cg_rel_tol": args.cg_rel_tol,
        },
        progress=not args.no_progress,
    )
    return 0


def cmd_all(args) -> int:
    outdir = Paths(args.outdir)
    g = _resolve(args)
    if not args.skip_grid:
        run_grid(
            outdir,
            pdes=args.pdes,
            widths=g["widths"],
            seeds=g["seeds"],
            lrs=g["lrs"],
            switches=g["switches"],
            iters=g["iters"],
            progress=not args.no_progress,
        )
    cmd_analyze(argparse.Namespace(outdir=args.outdir))
    if not args.skip_spectra:
        for pde in args.pdes:
            run_spectral_experiment(
                pde,
                g["widths"][-1],
                g["seeds"][0],
                g["lrs"][-1],
                outdir,
                switch=(11000 if not args.quick else g["switches"][0]),
                iters=g["iters"],
                n_runs=args.n_runs,
                n_iter=args.n_lanczos,
                components=True,
                progress=not args.no_progress,
            )
    if not args.skip_finetune:
        for pde in args.pdes:
            run_finetune_experiment(
                pde,
                g["widths"][-1],
                g["seeds"][0],
                g["lrs"][-1],
                outdir,
                switch=(11000 if not args.quick else g["switches"][0]),
                iters=g["iters"],
                nncg_iters=2000 if not args.quick else 20,
                gd_iters=2000 if not args.quick else 20,
                mu_grid=args.mu_grid,
                nncg_cfg={
                    "sketch_size": args.sketch_size,
                    "cg_max_iter": args.cg_max_iter,
                    "cg_rel_tol": args.cg_rel_tol,
                },
                progress=not args.no_progress,
            )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return {
        "grid": cmd_grid,
        "analyze": cmd_analyze,
        "spectra": cmd_spectra,
        "finetune": cmd_finetune,
        "all": cmd_all,
    }[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
