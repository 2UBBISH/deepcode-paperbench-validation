"""Explanation methods for RICE.

Provides multiple strategies to identify critical states in trajectories.
"""
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import gymnasium as gym
import numpy as np

from rice.mask_network import MaskNetwork


class ExplanationMethod(ABC):
    """Base class for step-level explanation methods."""

    @abstractmethod
    def explain(self, trajectory: np.ndarray) -> np.ndarray:
        """Return an importance score for each state in the trajectory.

        Args:
            trajectory: Array of shape (T, obs_dim) containing states.

        Returns:
            Array of shape (T,) with importance scores (higher = more critical).
        """
        ...

    def identify_critical_steps(
        self,
        trajectory: np.ndarray,
        k: Optional[float] = None,
        n_steps: Optional[int] = None,
    ) -> np.ndarray:
        """Identify the most critical consecutive steps.

        Args:
            trajectory: Trajectory of states.
            k: Fraction of trajectory length to use as window size.
            n_steps: Exact window size (overrides k if provided).

        Returns:
            Boolean array of shape (T,) indicating selected critical steps.
        """
        scores = self.explain(trajectory)
        T = len(scores)
        if n_steps is None:
            if k is None:
                k = 0.1
            n_steps = max(1, int(T * k))
        if n_steps >= T:
            selected = np.ones(T, dtype=bool)
        else:
            best_start = 0
            best_score = -float("inf")
            for start in range(T - n_steps + 1):
                window_score = np.mean(scores[start : start + n_steps])
                if window_score > best_score:
                    best_score = window_score
                    best_start = start
            selected = np.zeros(T, dtype=bool)
            selected[best_start : best_start + n_steps] = True
        return selected


class RandomExplanation(ExplanationMethod):
    """Random baseline: assign random importance scores."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)

    def explain(self, trajectory: np.ndarray) -> np.ndarray:
        return self.rng.random(len(trajectory))


class MaskExplanation(ExplanationMethod):
    """RICE explanation using the trained mask network.

    Importance of a state is P(a^m=0 | s), i.e. the probability that the mask
    network chooses to preserve the target agent's action at that state.
    """

    def __init__(self, mask_net: MaskNetwork) -> None:
        self.mask_net = mask_net

    def explain(self, trajectory: np.ndarray) -> np.ndarray:
        return self.mask_net.importance_scores(trajectory)


class StateMaskExplanation(ExplanationMethod):
    """StateMask explanation approximation.

    StateMask (Cheng et al., 2023) trains a mask network with objective
    min |eta(pi) - eta(pi_bar)|. The RICE paper proposes a simpler alternative
    (see MaskExplanation) with equivalent fidelity. This class provides a
    compatible interface using a mask network trained with the original sign-free
    objective, implemented via direct reward-matching PPO.
    """

    def __init__(self, mask_net: MaskNetwork) -> None:
        self.mask_net = mask_net

    def explain(self, trajectory: np.ndarray) -> np.ndarray:
        return self.mask_net.importance_scores(trajectory)
