"""Discovery helpers for run directories.

Convention: every run lives in ``<root>/<label>_<task>_seed<seed>`` where
``task`` is one of the five benchmark tasks (or the toy task) and ``label`` is
the method / ablation name, e.g. ``sapg_entropy0.005_reorientation_seed3``.
"""

from __future__ import annotations

import os
import re
from typing import Dict, Iterable, List, Sequence, Tuple

from ..envs import BENCHMARK_TASKS, TOY_TASKS

KNOWN_TASKS = tuple(BENCHMARK_TASKS) + tuple(TOY_TASKS)
SEED_PATTERN = re.compile(r"_seed(\d+)$")


def parse_run_dir(name: str, tasks: Sequence[str] = KNOWN_TASKS) -> Tuple[str, str, int]:
    seed_match = SEED_PATTERN.search(name)
    seed = int(seed_match.group(1)) if seed_match else 0
    stem = name[: seed_match.start()] if seed_match else name
    for task in sorted(tasks, key=len, reverse=True):
        if stem.endswith("_" + task):
            return stem[: -(len(task) + 1)], task, seed
    raise ValueError(f"Could not parse run directory '{name}' (expected '<label>_<task>_seed<k>')")


def collect_runs(
    root: str,
    tasks: Sequence[str] = KNOWN_TASKS,
    labels: Iterable[str] = (),
    seeds: Iterable[int] = (),
) -> Dict[str, Dict[str, List[str]]]:
    """Return ``{task: {label: [run_dir, ...]}}``."""
    labels = set(labels)
    seeds = set(seeds)
    runs: Dict[str, Dict[str, List[str]]] = {}
    if not os.path.isdir(root):
        return runs
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        try:
            label, task, seed = parse_run_dir(name, tasks)
        except ValueError:
            continue
        if labels and label not in labels:
            continue
        if seeds and seed not in seeds:
            continue
        runs.setdefault(task, {}).setdefault(label, []).append(path)
    for task in runs:
        for label in runs[task]:
            runs[task][label] = sorted(runs[task][label])
    return runs


def default_metric(task: str) -> str:
    """Successes for the hard AllegroKuka tasks, episode reward otherwise."""
    if task in ("regrasping", "throw", "reorientation"):
        return "eval0/successes"
    return "eval0/episode_reward"
