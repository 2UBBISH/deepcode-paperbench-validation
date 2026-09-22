"""Experiment scripts for reproducing the results of

    "Challenges in Training PINNs: A Loss Landscape Perspective"

Each module in this package maps to one or more figures/tables of the paper:

    run_optimizer_comparison.py -> Table 1, Figure 8
    run_spectral_density.py     -> Figures 3 & 7
    run_loss_vs_l2re.py         -> Figure 2
    run_nncg_finetune.py        -> Table 2, Figures 1, 4, 5
    run_wallclock.py            -> Table 3

All scripts are runnable as ``python -m opt_for_pinns.experiments.<name>`` and
write their outputs (JSON results + figures) into a user-specified output
directory (default: ``results/``).
"""

from __future__ import annotations

__all__ = [
    "run_optimizer_comparison",
    "run_spectral_density",
    "run_loss_vs_l2re",
    "run_nncg_finetune",
    "run_wallclock",
]
