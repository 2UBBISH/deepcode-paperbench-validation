"""The hyper-parameter grid of Section 6 (Table 1, Figures 2 and 8)."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from .common import (
    ADAM_LRS,
    PDES,
    SEEDS,
    SWITCHES,
    TOTAL_ITERS,
    WIDTHS,
    Paths,
    RunSpec,
    run_training,
)


def grid_specs(
    pdes: Sequence[str] = PDES,
    widths: Sequence[int] = WIDTHS,
    seeds: Sequence[int] = SEEDS,
    lrs: Sequence[float] = ADAM_LRS,
    switches: Sequence[int] = SWITCHES,
    iters: int = TOTAL_ITERS,
) -> List[RunSpec]:
    """Every (PDE, width, seed, optimizer, learning rate) combination.

    Following Section 2.2 the Adam learning rate is tuned over five values for
    *every* optimisation strategy; L-BFGS itself uses the default learning rate
    of 1.0 (so it is run once per width/seed).
    """
    specs: List[RunSpec] = []
    for pde in pdes:
        for width in widths:
            for seed in seeds:
                for lr in lrs:
                    specs.append(RunSpec(pde, width, seed, "adam", lr, 0, iters))
                    for sw in switches:
                        specs.append(RunSpec(pde, width, seed, "adam_lbfgs", lr, sw, iters))
                specs.append(RunSpec(pde, width, seed, "lbfgs", 1.0, 0, iters))
    return specs


def run_grid(
    outdir: Paths,
    specs: Optional[Iterable[RunSpec]] = None,
    progress: bool = True,
    limit: Optional[int] = None,
    **grid_kwargs,
) -> List[dict]:
    specs = list(grid_specs(**grid_kwargs)) if specs is None else list(specs)
    if limit is not None:
        specs = specs[:limit]
    records = []
    for i, spec in enumerate(specs):
        if progress:
            print(f"[{i + 1}/{len(specs)}] {spec.run_id}", flush=True)
        records.append(run_training(spec, outdir, progress=progress))
        if progress:
            print(
                f"    loss={records[-1]['final_loss']:.3e} "
                f"l2re={records[-1]['final_l2re']:.3e} "
                f"({records[-1]['wall_clock']:.1f}s)",
                flush=True,
            )
    return records
