"""Interfaces for real-world application environments used in RICE.

These environments require external simulators that are not bundled with this
reproduction. Each class below documents the expected observation/action spaces
and reward structure so that the RICE algorithms can be applied when the
simulator is available.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


class SelfishMiningEnv(gym.Env):
    """Selfish mining environment (Bar-Zur et al., 2023).

    Action space: {Adopt(l), Reveal(l), Mine}.
    Observation space: current chain state.
    Reward: mining revenue.
    """

    def __init__(self) -> None:
        super().__init__()
        self.action_space = gym.spaces.Discrete(3)
        # Placeholder observation dimension; set to actual chain-state size when
        # the simulator is connected.
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
        )

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        raise NotImplementedError(
            "SelfishMiningEnv requires the simulator from Bar-Zur et al. (2023)."
        )


class CageChallenge2Env(gym.Env):
    """CAGE Challenge 2 autonomous cyber defense environment (CAGE, 2022).

    Action space: {Monitor, Analyze, Decoy*, Remove, Restore}.
    Observation space: network state.
    Reward: negative when the red agent maintains admin access.
    """

    def __init__(self, trial_length: int = 30) -> None:
        super().__init__()
        self.trial_length = trial_length
        self.action_space = gym.spaces.Discrete(15)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(52,), dtype=np.float32
        )

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        raise NotImplementedError(
            "CageChallenge2Env requires the CybORG simulator from CAGE Challenge 2."
        )


class MetaDriveEnv(gym.Env):
    """MetaDrive autonomous driving environment (Li et al., 2022).

    Action space: continuous [steering, acceleration/brake].
    Observation space: vector of BEV and sensor information.
    Reward: combines progress, collision penalty, and comfort terms.
    """

    def __init__(self, scenario_name: str = "Macro-v1") -> None:
        super().__init__()
        self.scenario_name = scenario_name
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(259,), dtype=np.float32
        )

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        raise NotImplementedError(
            "MetaDriveEnv requires the MetaDrive simulator."
        )
