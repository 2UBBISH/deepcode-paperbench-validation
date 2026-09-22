"""CAGE Challenge 2 wrapper (network defence application).

The paper evaluates a PPO blue agent defending a network against the red agent
"B-line" and reports the final reward as the sum of the average rewards of
episodes of length 30, 50 and 100 (Section C.2).

The simulator is a heavy third party dependency, therefore it is imported
lazily.  It is **not** on PyPI (the PyPI package called ``cyborg`` is an
unrelated web-scraping library), install it from the challenge repository::

    git clone https://github.com/cage-challenge/cage-challenge-2
    cd cage-challenge-2 && pip install -e .

Then point ``RICE_CAGE_PATH`` to a scenario file of the challenge (e.g.
``cage-challenge-2/.../scenario1.yaml``); see the README.  All the RICE
machinery (mask network, critical state identification, mixed initial state
distribution, RND exploration) is simulator agnostic; this module only provides
the environment plumbing and the evaluation metric of the paper ("the final
reward is the sum of the average rewards of the three episode lengths 30, 50
and 100").
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from rice.envs.adapters import StatefulEnv

try:  # pragma: no cover - optional dependency
    from CybORG import CybORG  # type: ignore

    CYBORG_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    CybORG = None  # type: ignore
    CYBORG_AVAILABLE = False


EPISODE_LENGTHS = (30, 50, 100)


class CageChallengeEnv(StatefulEnv):
    """Thin wrapper exposing the CAGE-2 blue agent task as a stateful env."""

    supports_state_restore = True

    def __init__(
        self,
        episode_length: int = 30,
        scenario_path: Optional[str] = None,
        name: str = "cage",
    ):
        if not CYBORG_AVAILABLE:  # pragma: no cover - optional dependency
            raise ImportError(
                "CybORG is not installed. Run `pip install CybORG` to enable the "
                "CAGE Challenge 2 experiments."
            )
        scenario_path = scenario_path or os.environ.get("RICE_CAGE_PATH")
        if scenario_path is None:
            raise ValueError(
                "Set RICE_CAGE_PATH to the cage-challenge-2 YAML scenario file."
            )
        self.episode_length = int(episode_length)
        self.scenario_path = scenario_path
        self._cyborg = CybORG(scenario_path, "sim", seed=0, max_episode_length=episode_length)
        self._action_space = list(self._cyborg.action_space("Blue"))
        self._action_log: list = []
        super().__init__(env=None, name=name)

    # ------------------------------------------------------------- gym API
    @property
    def observation_space(self):
        if getattr(self, "_obs_dim", None) is None:
            obs = _flatten_observation(self._cyborg.get_observation("Blue"))
            self._obs_dim = int(obs.size)
        return gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32
        )

    @property
    def action_space(self):
        return _DiscreteMask(len(self._action_space))

    @property
    def unwrapped(self):
        return self

    def random_action(self):
        return int(np.random.randint(len(self._action_space)))

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._cyborg.reset()
        self._action_log = []
        return self._obs(), {}

    def step(self, action):
        action = int(action)
        self._action_log.append(action)
        raw_action = self._action_space[action][1]
        obs = self._cyborg.step("Blue", raw_action).observation
        reward = float(obs.get("Blue", {}).get("reward", 0.0)) if isinstance(obs, dict) else 0.0
        truncated = len(self._action_log) >= self.episode_length
        return self._obs(), reward, False, truncated, {}

    def _obs(self) -> np.ndarray:
        return _flatten_observation(self._cyborg.get_observation("Blue"))

    def get_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {"actions": list(self._action_log)}
        # CybORG >= 2 exposes a native (deep-copyable) simulator state; prefer it
        getter = getattr(self._cyborg, "get_state", None)
        if callable(getter):
            try:
                state["native"] = getter()
            except Exception:  # pragma: no cover - older CybORG versions
                pass
        return state

    def set_state(self, state: Dict[str, Any]) -> None:
        setter = getattr(self._cyborg, "set_state", None)
        if "native" in state and callable(setter):
            try:
                setter(state["native"])
                self._action_log = list(state.get("actions", []))
                return
            except Exception:  # pragma: no cover - fall back to replay
                pass
        actions = state["actions"]
        self._cyborg.reset()
        self._action_log = []
        for action in actions:
            self.step(action)


def cage_episode_lengths(scenario_path: Optional[str] = None):
    """One environment per episode length used by the paper."""
    return [
        CageChallengeEnv(episode_length=length, scenario_path=scenario_path)
        for length in EPISODE_LENGTHS
    ]


def cage_final_reward(
    policy,
    scenario_path: Optional[str] = None,
    episodes_per_length: int = 3,
    verbose: bool = False,
) -> Dict[str, float]:
    """Metric of Appendix C.2: sum of average rewards over lengths 30/50/100."""
    per_length: Dict[int, float] = {}
    for length in EPISODE_LENGTHS:
        env = CageChallengeEnv(episode_length=length, scenario_path=scenario_path)
        returns = []
        for _ in range(episodes_per_length):
            obs, _ = env.reset()
            done = False
            total = 0.0
            while not done:
                action = policy.act(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                done = bool(terminated or truncated)
            returns.append(total)
        per_length[length] = float(np.mean(returns))
        if verbose:
            print("  CAGE length {}: {:.3f}".format(length, per_length[length]))
    return {
        "per_length": per_length,
        "final_reward": float(sum(per_length.values())),
    }


def _flatten_observation(obs) -> np.ndarray:
    """Flatten the nested CAGE observation dictionary into a vector."""
    values: list = []

    def _walk(obj):
        if isinstance(obj, dict):
            for key in sorted(obj):
                _walk(obj[key])
        elif isinstance(obj, (list, tuple, np.ndarray)):
            for item in obj:
                _walk(item)
        elif isinstance(obj, (int, float, bool, np.integer, np.floating)):
            values.append(float(obj))

    _walk(obs)
    return np.asarray(values, dtype=np.float32)


class _DiscreteMask:
    """Minimal stand-in for ``spaces.Discrete`` (avoids a gym dependency)."""

    def __init__(self, n: int):
        self.n = n

    def sample(self):
        return int(np.random.randint(self.n))

    def contains(self, x) -> bool:
        return isinstance(x, (int, np.integer)) and 0 <= int(x) < self.n

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return "Discrete({})".format(self.n)
