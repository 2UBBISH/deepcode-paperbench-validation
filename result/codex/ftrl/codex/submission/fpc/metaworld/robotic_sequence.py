"""RoboticSequence -- multi-stage Meta-World task (Appendix B.3, Algorithm 1).

The environment chains ``N`` Meta-World environments into a single episode: the
agent starts in stage 1 and only moves to stage ``i+1`` once stage ``i`` is
solved.  The episode terminates either when the last stage is solved or when the
time limit ``T`` is reached.

Modifications with respect to the original Meta-World codebase, all described in
Appendix B.3:

1. start and goal conditions are randomised,
2. the episode terminates on success *or* on the time limit, and in both cases
   the state is treated as terminal (no bootstrapping in the Q-target),
3. the normalised timestep is appended to the observation so that the MDP is
   fully observable,
4. on success the agent receives the "remaining" reward
   ``r' = beta * r * (T - t)`` with ``beta = 1.5``,
5. a stage ID is provided; the networks use a separate output head per stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import MetaworldConfig


def _make_metaworld_env(stage: str, seed: int):
    """Create a single Meta-World environment for ``stage``."""

    try:
        import gymnasium as gym
        import metaworld  # noqa: F401

        env = gym.make("Meta-World/MT1", env_name=stage, seed=seed)
        return env
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "RoboticSequence requires the Meta-World benchmark: pip install metaworld "
            "(gymnasium >= 0.28, mujoco)."
        ) from exc


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: Dict
    stage: int
    stage_solved: bool


class RoboticSequence:
    """Multi-stage Meta-World environment (Algorithm 1)."""

    def __init__(
        self,
        config: Optional[MetaworldConfig] = None,
        seed: int = 0,
        stages: Optional[List[str]] = None,
        translate_observation: float = 0.0,
    ) -> None:
        self.config = config or MetaworldConfig()
        self.stages = stages if stages is not None else list(self.config.stages)
        self.num_stages = len(self.stages)
        self.seed = seed
        self.translate_observation = translate_observation
        self._envs: List = []
        self._current_stage = 0
        self._t = 0
        self._stage_steps = 0
        self.stage_successes: List[bool] = [False] * self.num_stages

    # ------------------------------------------------------------------
    def _ensure_env(self, stage_idx: int):
        while len(self._envs) <= stage_idx:
            idx = len(self._envs)
            self._envs.append(_make_metaworld_env(self.stages[idx], seed=self.seed + idx))

    @property
    def observation_dim(self) -> int:
        self._ensure_env(0)
        base = int(np.prod(self._envs[0].observation_space.shape))
        return base + 1  # + normalised timestep

    @property
    def num_actions(self) -> int:
        self._ensure_env(0)
        return int(np.prod(self._envs[0].action_space.shape))

    # ------------------------------------------------------------------
    def _augment(self, obs: np.ndarray, stage: int) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).ravel()
        if self.translate_observation:
            obs = obs + self.translate_observation
        timestep = np.array([self._t / self.config.time_limit], dtype=np.float32)
        return np.concatenate([obs, timestep])

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self.seed = seed
        self._current_stage = 0
        self._t = 0
        self._stage_steps = 0
        self.stage_successes = [False] * self.num_stages
        self._ensure_env(0)
        obs, info = self._envs[0].reset(seed=self.seed)
        return self._augment(obs, 0), {"stage": 0, **info}

    def step(self, action: np.ndarray) -> StepResult:
        env = self._envs[self._current_stage]
        obs, reward, terminated, truncated, info = env.step(action)
        self._t += 1
        self._stage_steps += 1

        stage_solved = bool(info.get("success", 0.0) > 0.5) or bool(terminated)
        augmented_reward = float(reward)

        if stage_solved:
            # r' = beta * r * (T - t)  -- "remaining" reward on success.
            augmented_reward = self.config.success_reward_beta * float(reward) * (
                self.config.time_limit - self._t
            )
            self.stage_successes[self._current_stage] = True
            if self._current_stage + 1 < self.num_stages:
                self._current_stage += 1
                self._stage_steps = 0
                self._ensure_env(self._current_stage)
                obs, info = self._envs[self._current_stage].reset()
                terminated, truncated = False, False
            else:
                terminated = True

        time_limit_reached = self._t >= self.config.time_limit
        if time_limit_reached:
            truncated = True

        done = terminated or truncated
        augmented_obs = self._augment(obs, self._current_stage)
        return StepResult(
            observation=augmented_obs,
            reward=augmented_reward,
            terminated=terminated,
            truncated=truncated,
            info={**info, "stage": self._current_stage, "solved_stages": sum(self.stage_successes)},
            stage=self._current_stage,
            stage_solved=stage_solved,
        )

    def close(self) -> None:
        for env in self._envs:
            try:
                env.close()
            except Exception:
                pass
        self._envs = []
