"""Episode rollouts with full state bookkeeping.

Every rollout records, for each visited step, the simulator snapshot so that the
agent can later be *fast-forwarded* to that state (fidelity metric) or reset to
it (RICE / StateMask-R).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclasses.dataclass
class Episode:
    obs: List[np.ndarray] = dataclasses.field(default_factory=list)
    actions: List[Any] = dataclasses.field(default_factory=list)
    rewards: List[float] = dataclasses.field(default_factory=list)
    dones: List[bool] = dataclasses.field(default_factory=list)
    infos: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    #: snapshot of the environment *before* executing ``actions[t]``
    snapshots: List[Any] = dataclasses.field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.actions)

    @property
    def total_reward(self) -> float:
        return float(np.sum(self.rewards)) if self.rewards else 0.0

    def rewards_array(self) -> np.ndarray:
        return np.asarray(self.rewards, dtype=np.float64)

    def obs_array(self) -> np.ndarray:
        return np.asarray(self.obs, dtype=np.float32)


def rollout_episode(
    env,
    action_fn: Callable[[np.ndarray], Any],
    max_steps: Optional[int] = None,
    snapshot: Optional[Dict[str, Any]] = None,
    record_snapshots: bool = True,
) -> Episode:
    """Run one episode and record states/actions/rewards.

    ``snapshot`` restores a previously recorded simulator state (the
    "fast-forward" of the paper) instead of resetting from the initial state
    distribution.
    """
    episode = Episode()
    if snapshot is None:
        obs, _ = env.reset()
    else:
        env.set_state(snapshot)
        obs = env.current_obs()
    steps = 0
    while True:
        if record_snapshots:
            episode.snapshots.append(env.get_state())
        action = action_fn(obs)
        episode.obs.append(np.asarray(obs, dtype=np.float32).copy())
        episode.actions.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        episode.rewards.append(float(reward))
        done = bool(terminated or truncated)
        episode.dones.append(done)
        episode.infos.append(info if isinstance(info, dict) else {})
        steps += 1
        if done or (max_steps is not None and steps >= max_steps):
            break
    return episode


def rollout_steps(
    env,
    action_fn: Callable[[np.ndarray], Any],
    n_steps: int,
    record_snapshots: bool = True,
    initial_snapshot: Optional[Dict[str, Any]] = None,
) -> Episode:
    """Run ``n_steps`` environment steps, restarting the episode when it ends."""
    episode = Episode()
    if initial_snapshot is None:
        obs, _ = env.reset()
    else:
        env.set_state(initial_snapshot)
        obs = env.current_obs()
    steps = 0
    while steps < n_steps:
        if record_snapshots:
            episode.snapshots.append(env.get_state())
        action = action_fn(obs)
        episode.obs.append(np.asarray(obs, dtype=np.float32).copy())
        episode.actions.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        episode.rewards.append(float(reward))
        done = bool(terminated or truncated)
        episode.dones.append(done)
        episode.infos.append(info if isinstance(info, dict) else {})
        steps += 1
        if done:
            obs, _ = env.reset()
    return episode
