"""A self-contained, CPU-runnable stand-in for ``RoboticSequence``.

The real ``RoboticSequence`` is built on Meta-World, which requires Python >= 3.10
and MuJoCo.  Neither is available in the environment where this reproduction was
prepared, so this module provides a minimal multi-stage continuous-control task
with *exactly the same structure* as RoboticSequence:

* an episode is a chain of ``N`` stages; the agent only advances to stage ``i+1``
  once stage ``i`` is solved (Algorithm 1),
* the observation is the task state, a one-hot stage id and the normalised
  timestep (Appendix B.3),
* the episode terminates on success or on the time limit, without bootstrapping,
* the agent receives the augmented "remaining" reward ``r' = beta * r * (T - t)``
  on success,
* the pre-trained policy ``pi_*`` is trained on the **last two** stages, so the
  first stages are CLOSE and the pre-trained stages are FAR -- a state coverage
  gap.

The environment is a chain of 2D point-reaching tasks: in stage ``i`` the agent
controls a point mass and has to move it to a goal location sampled uniformly in
``[-1, 1]^2``.  Each stage gives a dense ``-||x - g||`` reward and ``+1`` on
success.  The stage/goal information is part of the observation, so the task is
fully observable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


class PointReachStage:
    """A single 2D point-reaching stage with its own action transform.

    Each stage applies a different (unknown) linear transform to the action,
    ``dx = dt * A_i @ a``.  Because the transform differs between stages, the
    optimal behaviour differs as well: the stages are genuinely different
    *skills* rather than the same skill with a different goal, which is what
    makes forgetting possible (as in the Meta-World stages of the paper).
    """

    def __init__(self, rng: np.random.Generator, transform: np.ndarray, dt: float = 0.4,
                 success_radius: float = 0.25) -> None:
        self.rng = rng
        self.dt = dt
        self.transform = np.asarray(transform, dtype=np.float32)
        self.success_radius = success_radius
        self.position = np.zeros(2, dtype=np.float32)
        self.goal = np.zeros(2, dtype=np.float32)

    def reset(self) -> None:
        self.position = self.rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        self.goal = self.rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
        while np.linalg.norm(self.position - self.goal) < 0.5:
            self.goal = self.rng.uniform(-1.0, 1.0, size=2).astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[float, bool]:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.position = np.clip(self.position + self.dt * (self.transform @ action), -1.2, 1.2)
        distance = float(np.linalg.norm(self.position - self.goal))
        success = distance < self.success_radius
        reward = -distance + (1.0 if success else 0.0)
        return reward, success


def stage_transforms(num_stages: int, seed: int = 0) -> List[np.ndarray]:
    """Fixed per-stage action transforms (2x2), one for every stage."""

    rng = np.random.default_rng(1234 + seed)
    transforms = []
    for i in range(num_stages):
        angle = rng.uniform(0.0, np.pi / 3.0)
        scale = rng.uniform(0.9, 1.1, size=2)
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]], dtype=np.float32)
        transforms.append(rotation @ np.diag(scale))
    return transforms


@dataclass
class PointChainStepResult:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: Dict
    stage: int
    stage_solved: bool


class PointReachChain:
    """Multi-stage point-reaching chain (the RoboticSequence interface)."""

    def __init__(
        self,
        num_stages: int = 4,
        time_limit: int = 60,
        success_reward_beta: float = 1.5,
        seed: int = 0,
        stage_indices: Optional[List[int]] = None,
    ) -> None:
        self.num_stages = num_stages
        # ``stage_indices`` selects which of the global stages form the chain.
        # Pre-training uses the FAR stages only, e.g. ``stage_indices=[2, 3]``,
        # which is how ``pi_*`` is pre-trained in the paper.
        self.stage_indices = list(stage_indices) if stage_indices else list(range(num_stages))
        self.time_limit = time_limit
        self.success_reward_beta = success_reward_beta
        self.stages = [f"stage-{i}" for i in self.stage_indices]
        self._rng = np.random.default_rng(seed)
        self.transforms = stage_transforms(num_stages, seed)
        self._stage_envs = [
            PointReachStage(self._rng, self.transforms[i]) for i in self.stage_indices
        ]
        self._current_stage = 0
        self._t = 0
        self.stage_successes = [False] * self._chain_length

    # ------------------------------------------------------------------
    @property
    def observation_dim(self) -> int:
        return 4 + self.num_stages + 1  # pos(2) + goal(2) + one-hot stage + timestep

    @property
    def num_actions(self) -> int:
        return 2

    def _observation(self, stage: int) -> np.ndarray:
        env = self._stage_envs[stage]
        one_hot = np.zeros(self.num_stages, dtype=np.float32)
        one_hot[self.stage_indices[stage]] = 1.0
        timestep = np.array([self._t / self.time_limit], dtype=np.float32)
        return np.concatenate([env.position, env.goal, one_hot, timestep]).astype(np.float32)
    @property
    def num_chain_stages(self) -> int:
        return len(self.stage_indices)

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.transforms = stage_transforms(self.num_stages, seed)
            self._stage_envs = [
                PointReachStage(self._rng, self.transforms[i]) for i in self.stage_indices
            ]
        self._current_stage = 0
        self._t = 0
        self.stage_successes = [False] * self._chain_length
        for env in self._stage_envs:
            env.reset()
        return self._observation(0), {"stage": self.stage_indices[0]}

    @property
    def _chain_length(self) -> int:
        return len(self._stage_envs)

    def step(self, action: np.ndarray) -> PointChainStepResult:
        env = self._stage_envs[self._current_stage]
        reward, success = env.step(action)
        self._t += 1
        augmented_reward = float(reward)
        terminated = truncated = False

        if success:
            augmented_reward = self.success_reward_beta * float(reward) * (self.time_limit - self._t)
            self.stage_successes[self._current_stage] = True
            if self._current_stage + 1 < self._chain_length:
                self._current_stage += 1
            else:
                terminated = True
        if self._t >= self.time_limit:
            truncated = True

        global_stage = self.stage_indices[self._current_stage]
        return PointChainStepResult(
            observation=self._observation(self._current_stage),
            reward=augmented_reward,
            terminated=terminated,
            truncated=truncated,
            info={"stage": global_stage, "solved_stages": int(sum(self.stage_successes))},
            # ``stage`` is the GLOBAL stage index, which selects the per-stage
            # output head of the SAC networks.
            stage=global_stage,
            stage_solved=success,
        )

    def close(self) -> None:  # pragma: no cover - nothing to release
        pass


def evaluate_point_chain(
    agent,
    num_stages: int = 4,
    episodes: int = 10,
    time_limit: int = 60,
    seed: int = 0,
) -> Dict[str, float]:
    """Per-stage success rate and overall success rate of ``agent``."""

    env = PointReachChain(num_stages=num_stages, time_limit=time_limit, seed=seed)
    solved = {global_stage: 0 for global_stage in env.stage_indices}
    overall = 0
    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        stage = int(info["stage"])
        done = False
        while not done:
            action = agent.select_action(obs, stage, deterministic=True)
            result = env.step(action)
            obs, stage = result.observation, result.stage
            done = result.terminated or result.truncated
        for local_index, global_stage in enumerate(env.stage_indices):
            solved[global_stage] += int(env.stage_successes[local_index])
        overall += int(all(env.stage_successes))
    env.close()
    out = {f"stage_{i}": float(solved[i] / episodes) for i in sorted(solved)}
    out["overall"] = overall / episodes
    return out
