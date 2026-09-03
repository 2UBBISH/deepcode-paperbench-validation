"""Selfish-mining environment adapter for the pto-selfish-mining repository.

The paper evaluates RICE on the blockchain selfish-mining MDP from the
pto-selfish-mining repository. This module provides a thin Gymnasium-compatible
wrapper so that the rest of the RICE codebase can treat it like any other env.

Because pto-selfish-mining is an external repository, all imports are soft:
if it is not installed, the adapter classes are set to None and the factory
raises a clear ImportError only when actually called.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Soft import of the external selfish-mining repository.
try:
    from pto_selfish_mining import env as sm_env  # type: ignore
except Exception as _err:  # pragma: no cover
    sm_env = None  # type: ignore
    warnings.warn(
        "pto-selfish-mining could not be imported; selfish-mining experiments "
        f"will be unavailable. Error: {_err}"
    )


class SelfishMiningEnvAdapter(gym.Env):
    """Gymnasium adapter for the pto-selfish-mining MDP.

    The original environment is a discrete-event style simulator. We expose:
      - action space: Discrete(3) mapped to {Adopt, Reveal, Mine}
      - observation space: Box of the simulator state vector
      - reward: the block-reward signal produced by the simulator

    Parameters
    ----------
    alpha : float
        Attacker hash-rate share (default 0.35, matching the paper).
    gamma : float
        Attacker network-priority share (default 0.5).
    max_steps : int
        Episode horizon.
    whale_fee : float
        Reward for a whale transaction (default 10).
    whale_prob : float
        Probability of a whale transaction (default 0.01).
    normal_fee : float
        Reward for a normal transaction (default 1).
    """

    # Action semantics used by the paper / original repo.
    ACTION_NAMES = ["adopt", "reveal", "mine"]

    def __init__(
        self,
        alpha: float = 0.35,
        gamma: float = 0.5,
        max_steps: int = 1000,
        whale_fee: float = 10.0,
        whale_prob: float = 0.01,
        normal_fee: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if sm_env is None:
            raise ImportError(
                "pto-selfish-mining is not installed. Install it from "
                "https://github.com/AI-secure/pto-selfish-mining to use this adapter."
            )

        self.alpha = alpha
        self.gamma = gamma
        self.max_steps = max_steps
        self.whale_fee = whale_fee
        self.whale_prob = whale_prob
        self.normal_fee = normal_fee

        # Build the underlying environment. The exact constructor signature may
        # differ across versions of pto-selfish-mining, so we pass common args
        # and fall back to a minimal constructor.
        try:
            self._env = sm_env.SelfishMiningEnv(
                alpha=alpha,
                gamma=gamma,
                max_steps=max_steps,
                **kwargs,
            )
        except TypeError:
            self._env = sm_env.SelfishMiningEnv(alpha=alpha, gamma=gamma)

        self.action_space = spaces.Discrete(3)

        # Infer observation shape from a reset call.
        obs, _ = self._reset_underlying()
        obs = np.asarray(obs, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float32,
        )

        self._elapsed_steps = 0
        self._last_obs: np.ndarray = obs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _reset_underlying(self, seed: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
        """Reset the underlying env, normalising to the Gymnasium 2-tuple API."""
        if seed is not None:
            try:
                self._env.seed(seed)
            except Exception:
                pass
        result = self._env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result, {}

    def _step_underlying(self, action: int) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """Step the underlying env and normalise to Gymnasium 5-tuple API."""
        result = self._env.step(action)
        if isinstance(result, tuple):
            if len(result) == 5:
                return result  # type: ignore
            if len(result) == 4:
                obs, reward, terminated, info = result
                return obs, reward, terminated, False, info  # type: ignore
        # Some older repos return (obs, reward, done, info)
        obs, reward, done, info = result
        return obs, float(reward), bool(done), False, info

    def _sample_transaction_reward(self) -> float:
        """Sample a transaction fee reward.

        The paper mentions whale transaction fee 10 with probability 0.01 and
        normal fee 1 otherwise.
        """
        if self.np_random.random() < self.whale_prob:
            return self.whale_fee
        return self.normal_fee

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        obs, info = self._reset_underlying(seed=seed)
        self._last_obs = np.asarray(obs, dtype=np.float32)
        self._elapsed_steps = 0
        return self._last_obs, info or {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = int(np.asarray(action).item())
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action} for {self.action_space}")

        obs, reward, terminated, truncated, info = self._step_underlying(action)
        self._last_obs = np.asarray(obs, dtype=np.float32)
        self._elapsed_steps += 1

        if self._elapsed_steps >= self.max_steps:
            truncated = True

        # Augment reward with sampled transaction fee if the env did not already.
        if "transaction_reward" not in info:
            reward += self._sample_transaction_reward()
        else:
            reward += float(info["transaction_reward"])

        info["elapsed_steps"] = self._elapsed_steps
        return self._last_obs, float(reward), terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        return None

    def close(self) -> None:
        try:
            self._env.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # State capture / restore for critical-state refining
    # ------------------------------------------------------------------
    def get_simulator_state(self) -> Dict[str, Any]:
        """Return a picklable snapshot of the simulator state."""
        state: Dict[str, Any] = {
            "elapsed_steps": self._elapsed_steps,
            "last_obs": self._last_obs.copy(),
        }
        try:
            state["env_state"] = getattr(self._env, "state", None)
        except Exception:
            pass
        try:
            state["env_dict"] = self._env.__dict__.copy()
        except Exception:
            pass
        return state

    def set_simulator_state(self, state: Dict[str, Any]) -> None:
        """Restore the simulator from a snapshot produced by ``get_simulator_state``."""
        self._elapsed_steps = int(state.get("elapsed_steps", 0))
        self._last_obs = np.asarray(state.get("last_obs"), dtype=np.float32)
        env_state = state.get("env_state")
        if env_state is not None:
            try:
                self._env.state = env_state
            except Exception:
                pass
        env_dict = state.get("env_dict")
        if env_dict is not None:
            for key, value in env_dict.items():
                try:
                    setattr(self._env, key, value)
                except Exception:
                    pass


def make_selfish_mining_env(
    alpha: float = 0.35,
    gamma: float = 0.5,
    max_steps: int = 1000,
    whale_fee: float = 10.0,
    whale_prob: float = 0.01,
    normal_fee: float = 1.0,
    **kwargs: Any,
) -> SelfishMiningEnvAdapter:
    """Factory for the selfish-mining adapter."""
    return SelfishMiningEnvAdapter(
        alpha=alpha,
        gamma=gamma,
        max_steps=max_steps,
        whale_fee=whale_fee,
        whale_prob=whale_prob,
        normal_fee=normal_fee,
        **kwargs,
    )
