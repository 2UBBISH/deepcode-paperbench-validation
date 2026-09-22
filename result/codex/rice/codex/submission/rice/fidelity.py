"""Fidelity score of an explanation method (Experiment I).

Pipeline (Section 4.1 and the addendum of the paper):

1. the explanation method produces a step level importance score for every
   step of a trajectory sampled from the pre-trained policy ``pi``;
2. a sliding window of width ``l = K * L`` selects the most critical segment
   (highest average importance);
3. the agent is fast-forwarded to the beginning of that segment and executes
   *random* actions for ``l`` steps, after which it continues with ``pi`` until
   the episode terminates;
4. with ``R`` the original episode reward and ``R'`` the perturbed one,
   ``d = |R' - R|`` and

        fidelity = log(d / d_max) - log(l / L)

A higher score means that randomising the identified steps indeed destroys the
episode reward, i.e. the explanation is faithful.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from rice.explanation.critical_states import most_critical_window
from rice.rollout import rollout_episode
from rice.utils import set_global_seeds


@dataclasses.dataclass
class FidelityConfig:
    n_trajectories: int = 500
    k_values: Sequence[float] = (0.1, 0.2, 0.3, 0.4)
    #: largest reward the environment allows in one episode ("d_max")
    d_max: float = 1.0
    seed: int = 0
    max_steps: Optional[int] = None
    deterministic_policy: bool = False
    collect_snapshots: bool = True


def random_importance(states: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """The "Random" baseline explanation: a random score per step."""
    n = len(states)
    return rng.rand(n)


def fidelity_scores_for_trajectory(
    env,
    policy,
    importance_fn: Callable[[np.ndarray], np.ndarray],
    k_values: Sequence[float],
    d_max: float,
    deterministic_policy: bool = False,
    max_steps: Optional[int] = None,
) -> Dict[float, float]:
    """Compute the fidelity score of one trajectory for every ``K``."""
    def action_fn(obs):
        return policy.act(obs, deterministic=deterministic_policy)

    episode = rollout_episode(
        env, action_fn, max_steps=max_steps, record_snapshots=True
    )
    L = episode.length
    if L == 0:
        return {k: float("nan") for k in k_values}
    original_reward = episode.total_reward
    scores = np.asarray(importance_fn(episode.obs_array()), dtype=np.float64)

    results: Dict[float, float] = {}
    for k in k_values:
        window = int(round(k * L))
        if window < 1:
            results[k] = float("nan")
            continue
        start, end = most_critical_window(scores, window)
        counter = {"t": 0}

        def perturbed_fn(obs, _window=window):
            if counter["t"] < _window:
                counter["t"] += 1
                return env.random_action()
            counter["t"] += 1
            return policy.act(obs, deterministic=deterministic_policy)

        perturbed = rollout_episode(
            env,
            perturbed_fn,
            max_steps=max_steps,
            snapshot=episode.snapshots[start],
        )
        d = abs(perturbed.total_reward - original_reward)
        d = max(d, 1e-8)
        results[k] = float(
            np.log(d / abs(d_max)) - np.log(window / float(L))
        )
    return results


def compute_fidelity(
    env,
    policy,
    importance_fn: Callable[[np.ndarray], np.ndarray],
    config: Optional[FidelityConfig] = None,
    n_seeds: int = 3,
) -> Dict[str, object]:
    """Fidelity scores over ``n_trajectories`` trajectories and ``n_seeds`` runs.

    Returns a dictionary with the per-``K`` mean/std over the seeds, plus the
    raw per-seed values (used to draw the error bars of Figure 5).
    """
    config = config or FidelityConfig()
    per_seed: Dict[float, List[float]] = {
        float(k): [] for k in config.k_values
    }
    raw: List[Dict[float, float]] = []
    for seed_offset in range(n_seeds):
        set_global_seeds(config.seed + seed_offset, env=env)
        values: List[Dict[float, float]] = []
        for _ in range(config.n_trajectories):
            values.append(
                fidelity_scores_for_trajectory(
                    env,
                    policy,
                    importance_fn,
                    config.k_values,
                    config.d_max,
                    deterministic_policy=config.deterministic_policy,
                    max_steps=config.max_steps,
                )
            )
        raw.append(values)
        for k in config.k_values:
            finite = [v[float(k)] for v in values if np.isfinite(v[float(k)])]
            per_seed[float(k)].append(float(np.mean(finite)) if finite else float("nan"))

    summary = {
        "k": [float(k) for k in config.k_values],
        "mean": [
            float(np.nanmean(per_seed[float(k)])) for k in config.k_values
        ],
        "std": [
            float(np.nanstd(per_seed[float(k)])) for k in config.k_values
        ],
        "per_seed": {str(k): v for k, v in per_seed.items()},
        "d_max": config.d_max,
        "n_trajectories": config.n_trajectories,
        "n_seeds": n_seeds,
    }
    return summary


def mask_importance_fn(mask_net):
    """Importance function of the mask network (``P(a^m = 0 | s)``)."""

    def _fn(states: np.ndarray) -> np.ndarray:
        return np.asarray(mask_net.importance(states), dtype=np.float64).reshape(-1)

    return _fn


def random_importance_fn(seed: int = 0):
    rng = np.random.RandomState(seed)

    def _fn(states: np.ndarray) -> np.ndarray:
        return random_importance(states, rng)

    return _fn
