"""CAGE Challenge 2 environment wrapper for RICE.

This module provides a Gymnasium-compatible wrapper around the CybORG
CAGE Challenge 2 task.  The paper trains a PPO blue agent against the
red agent ``B-line`` using the *champion* scheme, and reports the
average reward across trials of length 30, 50 and 100.

Because CybORG is an optional dependency, the wrapper degrades
gracefully: if CybORG is unavailable, a lightweight mock environment
with the same observation/action shape is returned so that the rest of
the RICE pipeline can still be imported and unit-tested.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np

# ---------------------------------------------------------------------------
# Optional CybORG imports
# ---------------------------------------------------------------------------
try:
    from CybORG import CybORG
    from CybORG.Agents import B_lineAgent
    from CybORG.Agents.Wrappers import (
        ChallengeWrapper,
        FixedFlatWrapper,
        EnumActionWrapper,
        ObservationWrapper,
        BlueTableWrapper,
    )

    _CYBORG_AVAILABLE = True
except Exception as _cyborg_import_err:  # pragma: no cover
    _CYBORG_AVAILABLE = False
    CybORG = None  # type: ignore
    B_lineAgent = None  # type: ignore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_TRIAL_LENGTHS = (30, 50, 100)
DEFAULT_RED_AGENT = "B-line"
DEFAULT_SCENARIO = "Scenario2"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _flatten_obs(obs: Any) -> np.ndarray:
    """Best-effort conversion of a CybORG observation to a 1-D float vector."""
    if isinstance(obs, np.ndarray):
        return obs.astype(np.float32).ravel()
    if isinstance(obs, (list, tuple)):
        return np.asarray(obs, dtype=np.float32).ravel()
    if isinstance(obs, dict):
        parts = []
        for v in obs.values():
            parts.append(_flatten_obs(v))
        return np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)
    # Fallback: scalar or unknown
    return np.asarray([obs], dtype=np.float32).ravel()


def _make_cyborg_env(
    scenario_path: Optional[str] = None,
    red_agent: str = DEFAULT_RED_AGENT,
    trial_length: int = 30,
    seed: Optional[int] = None,
) -> gym.Env:
    """Build a raw CybORG CAGE Challenge 2 environment.

    Parameters
    ----------
    scenario_path : str, optional
        Path to the CybORG scenario file.  If ``None`` the default
        ``Scenario2`` scenario is used.
    red_agent : str
        Name of the red agent to use.  The paper uses ``B-line``.
    trial_length : int
        Number of steps per trial.
    seed : int, optional
        Random seed.

    Returns
    -------
    gym.Env
        A Gymnasium-wrapped CybORG environment.
    """
    if not _CYBORG_AVAILABLE:
        raise ImportError(
            "CybORG is required for the real CAGE Challenge 2 environment. "
            "Install it from https://github.com/cage-challenge/CybORG or "
            "use `make_cage_env(..., use_mock=True)`."
        )

    if scenario_path is None:
        # Default Scenario2 shipped with CybORG
        import inspect
        import os

        cyborg_path = inspect.getfile(CybORG)
        scenario_path = os.path.join(
            os.path.dirname(cyborg_path),
            "Shared",
            "Scenarios",
            f"{DEFAULT_SCENARIO}.yaml",
        )

    red_cls = B_lineAgent if red_agent == DEFAULT_RED_AGENT else B_lineAgent
    cyborg = CybORG(
        scenario_path,
        "sim",
        agents={"Red": red_cls()},
    )

    # Champion scheme: flatten observations and discretise actions.
    # The exact wrapper stack follows the Cardiff champion submission.
    env = ChallengeWrapper(env=cyborg, agent_name="Blue")
    env = FixedFlatWrapper(env)
    env = EnumActionWrapper(env)
    env = ObservationWrapper(env)
    env = BlueTableWrapper(env)

    if seed is not None:
        env.seed(seed)

    return env


# ---------------------------------------------------------------------------
# Mock environment (used when CybORG is not installed)
# ---------------------------------------------------------------------------
class _MockCAGEEnv(gym.Env):
    """Mock CAGE Challenge 2 environment with the same API shape.

    The observation and action dimensions are chosen to be representative
    of the champion scheme (≈ 52-D observation, 145 discrete actions).
    """

    def __init__(
        self,
        obs_dim: int = 52,
        n_actions: int = 145,
        trial_length: int = 30,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.trial_length = trial_length
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(n_actions)
        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._obs = self.observation_space.sample()

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._obs = self.observation_space.sample()
        return self._obs.astype(np.float32), {}

    def step(self, action: Any):
        self._step_count += 1
        terminated = self._step_count >= self.trial_length
        truncated = False
        # Reward is negative (penalty) with occasional large negative events.
        reward = float(self._rng.normal(-1.0, 0.5))
        self._obs = self.observation_space.sample()
        info = {"trial_length": self.trial_length, "step": self._step_count}
        return self._obs.astype(np.float32), reward, terminated, truncated, info

    def render(self):
        pass


# ---------------------------------------------------------------------------
# Main CAGE wrapper
# ---------------------------------------------------------------------------
class CAGEChallenge2Env(gym.Env):
    """Gymnasium wrapper for CAGE Challenge 2.

    A single episode consists of three trials of lengths 30, 50 and 100.
    The episode return is the average reward across the three trials,
    matching the evaluation protocol described in the paper.
    """

    def __init__(
        self,
        scenario_path: Optional[str] = None,
        red_agent: str = DEFAULT_RED_AGENT,
        trial_lengths: Tuple[int, ...] = DEFAULT_TRIAL_LENGTHS,
        use_mock: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.scenario_path = scenario_path
        self.red_agent = red_agent
        self.trial_lengths = tuple(trial_lengths)
        self.use_mock = use_mock or not _CYBORG_AVAILABLE
        self._seed = seed

        if self.use_mock:
            if not _CYBORG_AVAILABLE:
                warnings.warn(
                    "CybORG not available; using a mock CAGE Challenge 2 environment. "
                    "Install CybORG for real experiments.",
                    stacklevel=2,
                )
            self._envs = [
                _MockCAGEEnv(trial_length=L, seed=seed)
                for L in self.trial_lengths
            ]
        else:
            self._envs = [
                _make_cyborg_env(
                    scenario_path=scenario_path,
                    red_agent=red_agent,
                    trial_length=L,
                    seed=seed,
                )
                for L in self.trial_lengths
            ]

        # Use the first sub-environment to infer spaces.
        self.observation_space = self._envs[0].observation_space
        self.action_space = self._envs[0].action_space
        self._trial_idx = 0
        self._trial_rewards: List[float] = []

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self._seed = seed
            for env in self._envs:
                if hasattr(env, "seed"):
                    env.seed(seed)
        self._trial_idx = 0
        self._trial_rewards = []
        obs, info = self._reset_current_trial()
        info["trial_lengths"] = list(self.trial_lengths)
        info["current_trial_length"] = self.trial_lengths[self._trial_idx]
        return obs, info

    def _reset_current_trial(self):
        env = self._envs[self._trial_idx]
        result = env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}
        return _flatten_obs(obs), info

    def step(self, action: Any):
        env = self._envs[self._trial_idx]
        result = env.step(action)

        # Normalise both 5-tuple (Gymnasium) and 4-tuple (legacy) returns.
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            obs, reward, terminated, info = result
            truncated = False

        obs = _flatten_obs(obs)
        self._trial_rewards.append(float(reward))

        info["trial_idx"] = self._trial_idx
        info["trial_length"] = self.trial_lengths[self._trial_idx]

        if terminated or truncated:
            # Move to the next trial.
            self._trial_idx += 1
            if self._trial_idx < len(self._envs):
                obs, reset_info = self._reset_current_trial()
                info.update(reset_info)
                info["current_trial_length"] = self.trial_lengths[self._trial_idx]
                # Episode is not over yet; signal continuation.
                terminated = False
                truncated = False
            else:
                # All trials finished; report average reward.
                avg_reward = float(np.mean(self._trial_rewards)) if self._trial_rewards else 0.0
                info["trial_rewards"] = list(self._trial_rewards)
                info["average_reward"] = avg_reward
                terminated = True

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if hasattr(self._envs[0], "render"):
            self._envs[0].render()

    def close(self):
        for env in self._envs:
            if hasattr(env, "close"):
                env.close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def make_cage_env(
    scenario_path: Optional[str] = None,
    red_agent: str = DEFAULT_RED_AGENT,
    trial_lengths: Tuple[int, ...] = DEFAULT_TRIAL_LENGTHS,
    use_mock: bool = False,
    seed: Optional[int] = None,
) -> gym.Env:
    """Create a CAGE Challenge 2 environment.

    Parameters
    ----------
    scenario_path : str, optional
        Path to the CybORG scenario YAML file.
    red_agent : str, optional
        Red-agent name.  Default is ``B-line``.
    trial_lengths : tuple[int], optional
        Trial lengths to average over.  Default is ``(30, 50, 100)``.
    use_mock : bool, optional
        If ``True``, return a mock environment even if CybORG is installed.
    seed : int, optional
        Random seed.

    Returns
    -------
    gym.Env
        A Gymnasium-compatible CAGE Challenge 2 environment.
    """
    return CAGEChallenge2Env(
        scenario_path=scenario_path,
        red_agent=red_agent,
        trial_lengths=trial_lengths,
        use_mock=use_mock,
        seed=seed,
    )


__all__ = [
    "CAGEChallenge2Env",
    "make_cage_env",
    "DEFAULT_TRIAL_LENGTHS",
    "DEFAULT_RED_AGENT",
]
