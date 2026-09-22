"""Schedules used by APT.

Sparsity (Appendix A)
---------------------
"Given ``T`` pruning training steps in total, we set a pre-determined target
sparsity ``gamma_T`` ... and use cubic scheduling to control the LM parameter
size, where ``gamma_t = gamma_T + (1 - gamma_T) (1 - t/T)^3``."

Read literally this expression decreases from 1 to ``gamma_T``, i.e. ``gamma``
there denotes the *retained ratio* (the LM parameter size) rather than the
sparsity.  Equivalently, and identically to the usual cubic sparsity ramp, the
sparsity goes from 0 to the target::

    sparsity_t = gamma_T_sparsity * (1 - (1 - t/T)^3)

which is the form implemented here (``gamma_T_sparsity = 1 - gamma_T``).
The ramp grows fastest at the beginning, matching the paper's "early pruning"
description ("we prune LM parameters (increase gamma_t) during early training
when t << T").

Distillation weight mu (Eq. 7 + addendum)
-----------------------------------------
``mu`` is 0 before pruning starts and is linearly increased so that it reaches
1 at the end of the pruning stage::

    mu = min(1., (global_step - pruning_start) / (pruning_end - pruning_start))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class SparsitySchedule:
    """Cubic sparsity ramp (Appendix A)."""

    target_sparsity: float
    total_steps: int
    initial_sparsity: float = 0.0

    def density(self, step: int) -> float:
        """Fraction of the original parameters that may be retained."""
        u = self._progress(step)
        rho0 = 1.0 - self.initial_sparsity
        rhoT = 1.0 - self.target_sparsity
        # gamma_t = gamma_T + (1 - gamma_T) (1 - t/T)^3   (paper Appendix A)
        return rhoT + (rho0 - rhoT) * (1.0 - u) ** 3

    def sparsity(self, step: int) -> float:
        return 1.0 - self.density(step)

    def _progress(self, step: int) -> float:
        if self.total_steps <= 0:
            return 1.0
        return min(1.0, max(0.0, step / float(self.total_steps)))


def mu_schedule(step: int, pruning_start: int, pruning_end: int) -> float:
    """Linear ``mu`` ramp from 0 (before pruning) to 1 (end of pruning)."""
    if pruning_end <= pruning_start:
        return 1.0
    return min(1.0, max(0.0, (step - pruning_start) / float(pruning_end - pruning_start)))


def adjustment_steps(total_steps: int, interval: int) -> List[int]:
    """Steps at which APT re-selects the masks and grows the adapter ranks."""
    if interval <= 0:
        return [total_steps]
    return list(range(interval, total_steps + 1, interval))


def linear_rank(target_rank: int, initial_rank: int, step: int, total_steps: int) -> int:
    """Linear rank growth for salient adapters (Appendix A: "ranks linearly increased")."""
    if total_steps <= 0:
        return target_rank
    u = min(1.0, max(0.0, step / float(total_steps)))
    return int(round(initial_rank + (target_rank - initial_rank) * u))


def uniform_rank_preview(initial_rank: int, n_adapters: int) -> int:
    """Total tuning parameters if every adapter had ``initial_rank``."""
    return initial_rank * n_adapters


def cubic(x: float) -> float:
    return x ** 3


def exp_rank(target_rank: int, initial_rank: int, step: int, total_steps: int) -> int:
    """Alternative exponential rank ramp (kept for the ablation of Figure 5a)."""
    if total_steps <= 0:
        return target_rank
    u = min(1.0, max(0.0, step / float(total_steps)))
    return int(round(initial_rank * (target_rank / initial_rank) ** u))


__all__ = [
    "SparsitySchedule",
    "mu_schedule",
    "adjustment_steps",
    "linear_rank",
    "exp_rank",
    "cubic",
]
