"""Registry of the tasks of the paper."""

from __future__ import annotations

from typing import Dict, Type

from .base import Task
from .benchmarks import (GaussianLinearTask, GaussianMixtureTask, SLPCTask,
                         TwoMoonsTask)
from .hodgkin_huxley import HodgkinHuxleyTask
from .lotka_volterra import LotkaVolterraTask
from .sird import SIRDTask
from .tree_hmm import HMMTask, TreeTask

TASKS: Dict[str, Type[Task]] = {
    "gaussian_linear": GaussianLinearTask,
    "gaussian_mixture": GaussianMixtureTask,
    "two_moons": TwoMoonsTask,
    "slcp": SLPCTask,
    "tree": TreeTask,
    "hmm": HMMTask,
    "lotka_volterra": LotkaVolterraTask,
    "sird": SIRDTask,
    "hodgkin_huxley": HodgkinHuxleyTask,
}

# The four benchmark tasks of Lueckmann et al. (2021) that are used in Fig. 4a
BENCHMARK_TASKS = ["gaussian_linear", "gaussian_mixture", "two_moons", "slcp"]


def get_task(name: str, **kwargs) -> Task:
    """Instantiate a task by name."""
    if name not in TASKS:
        raise ValueError(f"Unknown task '{name}'. Available: {sorted(TASKS)}")
    return TASKS[name](**kwargs)


__all__ = [
    "Task", "TASKS", "BENCHMARK_TASKS", "get_task",
    "GaussianLinearTask", "GaussianMixtureTask", "TwoMoonsTask", "SLPCTask",
    "TreeTask", "HMMTask", "LotkaVolterraTask", "SIRDTask",
    "HodgkinHuxleyTask",
]
