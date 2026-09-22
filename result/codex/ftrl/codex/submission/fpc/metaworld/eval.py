"""Evaluation utilities for RoboticSequence (Figure 7, Appendix F)."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from ..analysis.forward_transfer import auc, confidence_interval, forward_transfer
from .config import MetaworldConfig
from .robotic_sequence import RoboticSequence


def evaluate_stages(
    agent,
    config: MetaworldConfig,
    stages: Optional[Sequence[str]] = None,
    episodes: int = 10,
    seed: int = 0,
) -> Dict[str, float]:
    """Per-stage success rate of ``agent`` on the RoboticSequence (Figure 7).

    The agent always starts in the first stage; a stage counts as solved if the
    agent reaches the end of that stage within the time limit.
    """

    env = RoboticSequence(config, seed=seed, stages=list(stages) if stages else None)
    solved_counts = np.zeros(len(env.stages))
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        stage = 0
        done = False
        while not done:
            action = agent.select_action(obs, stage, deterministic=True)
            result = env.step(action)
            obs, stage = result.observation, result.stage
            done = result.terminated or result.truncated
        # ``stage_successes[i]`` records whether stage ``i`` was ever solved
        # during this episode (RoboticSequence only advances on success).
        solved_counts += np.asarray(env.stage_successes, dtype=np.float64)
    env.close()
    return {stage: float(count / episodes) for stage, count in zip(env.stages, solved_counts)}


def success_rate_curve(
    agent,
    config: MetaworldConfig,
    stages: Optional[Sequence[str]] = None,
    episodes: int = 10,
    seed: int = 0,
) -> float:
    """Overall RoboticSequence success rate (all stages solved in one episode)."""

    env = RoboticSequence(config, seed=seed, stages=list(stages) if stages else None)
    successes = 0
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        stage = 0
        done = False
        while not done:
            action = agent.select_action(obs, stage, deterministic=True)
            result = env.step(action)
            obs, stage = result.observation, result.stage
            done = result.terminated or result.truncated
        if sum(env.stage_successes) == len(env.stages):
            successes += 1
    env.close()
    return successes / episodes


def per_stage_curves(
    success_histories: Dict[str, Dict[str, List[float]]],
    eval_steps: Sequence[float],
) -> Dict[str, Dict[str, tuple]]:
    """Aggregate per-stage success rates across seeds with 90% confidence intervals."""

    aggregated: Dict[str, Dict[str, tuple]] = {}
    for method, per_stage in success_histories.items():
        aggregated[method] = {}
        for stage, curves in per_stage.items():
            curves = np.asarray(curves, dtype=np.float64)
            mean = curves.mean(axis=0)
            half_width = np.asarray([confidence_interval(curves[:, i])[1] for i in range(curves.shape[1])])
            aggregated[method][stage] = (mean, half_width)
    return aggregated


def forward_transfer_table(
    fine_tuned: Dict[str, List[float]],
    from_scratch: Dict[str, List[float]],
    eval_steps: Sequence[float],
    stages: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    """Forward transfer for each method and each pre-trained stage (Table 6)."""

    table: Dict[str, Dict[str, float]] = {}
    for method, per_stage in fine_tuned.items():
        table[method] = {}
        for stage in stages:
            transfers = []
            for i in range(len(per_stage[stage])):
                transfers.append(
                    forward_transfer(
                        eval_steps,
                        per_stage[stage][i],
                        eval_steps,
                        from_scratch[stage][i],
                        total_steps=float(eval_steps[-1]),
                    )
                )
            mean, half = confidence_interval(transfers)
            table[method][stage] = {"mean": mean, "half_width": half}
    return table
