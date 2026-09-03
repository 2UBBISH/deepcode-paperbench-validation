"""
Reward Prior Distribution for Functional Reward Encodings (FRE).

Implements three families of unsupervised reward functions that are mixed
uniformly to train the FRE encoder-decoder:

1. Goal-reaching (singleton): η(s) = -1 if not at goal else 0.
   Goal state sampled uniformly from dataset states.

2. Random linear: η(s) = dot(w, s) where w ~ Uniform(-1, 1) with
   sparse mask (80% zeros) to bias towards simple functions.

3. Random MLP: 2-layer MLP with random weights (hidden dim 256, ReLU).
   Weights drawn from Kaiming uniform.

The RewardPrior class provides a unified interface to sample reward functions
from any of these families or a uniform mixture of all three.
"""

import numpy as np
from typing import Callable, Tuple, Optional, List
from abc import ABC, abstractmethod


class RewardFunction(ABC):
    """Abstract base class for a reward function η(s) -> scalar."""

    @abstractmethod
    def __call__(self, states: np.ndarray) -> np.ndarray:
        """Compute reward for a batch of states.

        Args:
            states: numpy array of shape (batch_size, state_dim)

        Returns:
            rewards: numpy array of shape (batch_size,)
        """
        pass

    @property
    @abstractmethod
    def reward_type(self) -> str:
        """Return the type of this reward function."""
        pass


class GoalReachingReward(RewardFunction):
    """Goal-reaching reward: η(s) = 0 if at goal, -1 otherwise.

    The goal state is sampled uniformly from the dataset states.
    A state is considered "at goal" if its L2 distance to the goal
    is below a threshold epsilon.
    """

    def __init__(self, goal_state: np.ndarray, epsilon: float = 0.1):
        """
        Args:
            goal_state: numpy array of shape (state_dim,)
            epsilon: distance threshold for goal achievement
        """
        self.goal_state = goal_state.copy()
        self.epsilon = epsilon

    def __call__(self, states: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(states - self.goal_state[None, :], axis=1)
        rewards = np.where(distances <= self.epsilon, 0.0, -1.0)
        return rewards

    @property
    def reward_type(self) -> str:
        return "goal_reaching"


class RandomLinearReward(RewardFunction):
    """Random linear reward: η(s) = dot(w, s) with sparse weights.

    Weights w ~ Uniform(-1, 1), then a sparse mask is applied
    (default 80% zeros) to bias towards simple functions.
    """

    def __init__(self, weight: np.ndarray):
        """
        Args:
            weight: numpy array of shape (state_dim,)
        """
        self.weight = weight.copy()

    def __call__(self, states: np.ndarray) -> np.ndarray:
        return states @ self.weight

    @property
    def reward_type(self) -> str:
        return "random_linear"


class RandomMLPReward(RewardFunction):
    """Random MLP reward: 2-layer MLP with random weights.

    Architecture: Linear(state_dim, 256) -> ReLU -> Linear(256, 1)
    Weights drawn from Kaiming uniform, biases from uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)).
    """

    def __init__(self, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray):
        """
        Args:
            W1: shape (state_dim, 256)
            b1: shape (256,)
            W2: shape (256, 1)
            b2: shape (1,)
        """
        self.W1 = W1.copy()
        self.b1 = b1.copy()
        self.W2 = W2.copy()
        self.b2 = b2.copy()

    def __call__(self, states: np.ndarray) -> np.ndarray:
        # states: (batch, state_dim)
        h = np.maximum(0, states @ self.W1 + self.b1)  # ReLU
        out = (h @ self.W2 + self.b2).squeeze(-1)  # (batch,)
        return out

    @property
    def reward_type(self) -> str:
        return "random_mlp"


class RewardPrior:
    """Generates diverse unsupervised reward functions η(s) for training FRE.

    Three families mixed uniformly:
      - Goal-reaching (singleton)
      - Random linear (sparse)
      - Random MLP

    Usage:
        prior = RewardPrior(dataset_states, state_dim)
        reward_fn, reward_type = prior.sample()
        rewards = reward_fn(states)
    """

    def __init__(
        self,
        dataset_states: np.ndarray,
        state_dim: int,
        linear_sparsity: float = 0.8,
        mlp_hidden_dim: int = 256,
        goal_epsilon: float = 0.1,
        reward_families: Optional[List[str]] = None,
    ):
        """
        Args:
            dataset_states: numpy array of shape (N, state_dim) — all states
                            from the offline dataset, used for sampling goal states.
            state_dim: dimensionality of the state space.
            linear_sparsity: fraction of weights set to zero for linear rewards.
            mlp_hidden_dim: hidden dimension for random MLP rewards.
            goal_epsilon: distance threshold for goal-reaching reward.
            reward_families: optional list of family names to use.
                             Default: ["goal_reaching", "random_linear", "random_mlp"].
                             Can be a subset for ablation experiments.
        """
        self.dataset_states = dataset_states
        self.state_dim = state_dim
        self.linear_sparsity = linear_sparsity
        self.mlp_hidden_dim = mlp_hidden_dim
        self.goal_epsilon = goal_epsilon

        if reward_families is None:
            self.reward_families = ["goal_reaching", "random_linear", "random_mlp"]
        else:
            self.reward_families = reward_families

        # Pre-compute dataset statistics for goal sampling
        self._num_dataset_states = len(dataset_states)

    def sample(self) -> Tuple[RewardFunction, str]:
        """Sample a reward function uniformly from the available families.

        Returns:
            reward_fn: a callable RewardFunction object
            reward_type: string identifying the family
        """
        family = np.random.choice(self.reward_families)

        if family == "goal_reaching":
            return self._sample_goal_reaching(), "goal_reaching"
        elif family == "random_linear":
            return self._sample_random_linear(), "random_linear"
        elif family == "random_mlp":
            return self._sample_random_mlp(), "random_mlp"
        else:
            raise ValueError(f"Unknown reward family: {family}")

    def _sample_goal_reaching(self) -> GoalReachingReward:
        """Sample a goal-reaching reward function."""
        idx = np.random.randint(0, self._num_dataset_states)
        goal_state = self.dataset_states[idx].copy()
        return GoalReachingReward(goal_state, epsilon=self.goal_epsilon)

    def _sample_random_linear(self) -> RandomLinearReward:
        """Sample a random linear reward function with sparse weights."""
        # Draw weights from Uniform(-1, 1)
        weight = np.random.uniform(-1.0, 1.0, size=self.state_dim)

        # Apply sparse mask: set random fraction to zero
        mask = np.random.rand(self.state_dim) > self.linear_sparsity
        weight = weight * mask.astype(np.float32)

        return RandomLinearReward(weight)

    def _sample_random_mlp(self) -> RandomMLPReward:
        """Sample a random MLP reward function.

        Uses Kaiming uniform initialization for weights.
        """
        # Kaiming uniform: U(-sqrt(6/fan_in), sqrt(6/fan_in))
        fan_in = self.state_dim
        bound_w1 = np.sqrt(6.0 / fan_in)
        W1 = np.random.uniform(-bound_w1, bound_w1, size=(self.state_dim, self.mlp_hidden_dim))

        fan_in_hidden = self.mlp_hidden_dim
        bound_b1 = 1.0 / np.sqrt(fan_in)
        b1 = np.random.uniform(-bound_b1, bound_b1, size=self.mlp_hidden_dim)

        bound_w2 = np.sqrt(6.0 / fan_in_hidden)
        W2 = np.random.uniform(-bound_w2, bound_w2, size=(self.mlp_hidden_dim, 1))

        bound_b2 = 1.0 / np.sqrt(fan_in_hidden)
        b2 = np.random.uniform(-bound_b2, bound_b2, size=(1,))

        return RandomMLPReward(W1, b1, W2, b2)

    def sample_batch(self, batch_size: int) -> List[Tuple[RewardFunction, str]]:
        """Sample multiple reward functions.

        Args:
            batch_size: number of reward functions to sample

        Returns:
            list of (reward_fn, reward_type) tuples
        """
        return [self.sample() for _ in range(batch_size)]


# ---------------------------------------------------------------------------
# Domain-specific reward priors (for Experiment 5: domain knowledge augmentation)
# ---------------------------------------------------------------------------

class AntMazeXYPrior(RewardPrior):
    """Reward prior that only uses XY-position for AntMaze tasks.

    In AntMaze, the state includes (x, y, vx, vy, ...). This prior
    restricts reward functions to only depend on (x, y) position,
    which is domain knowledge that improves performance.
    """

    def __init__(self, dataset_states: np.ndarray, state_dim: int, **kwargs):
        super().__init__(dataset_states, state_dim, **kwargs)
        # AntMaze state: first 2 dims are (x, y) position
        self.xy_dim = 2

    def _sample_random_linear(self) -> RandomLinearReward:
        """Linear reward only on XY dimensions."""
        weight_xy = np.random.uniform(-1.0, 1.0, size=self.xy_dim)
        mask = np.random.rand(self.xy_dim) > self.linear_sparsity
        weight_xy = weight_xy * mask.astype(np.float32)

        # Pad with zeros for remaining state dimensions
        weight = np.zeros(self.state_dim)
        weight[:self.xy_dim] = weight_xy

        return RandomLinearReward(weight)

    def _sample_random_mlp(self) -> RandomMLPReward:
        """MLP reward only on XY dimensions."""
        fan_in = self.xy_dim
        bound_w1 = np.sqrt(6.0 / fan_in)
        W1_xy = np.random.uniform(-bound_w1, bound_w1, size=(self.xy_dim, self.mlp_hidden_dim))

        bound_b1 = 1.0 / np.sqrt(fan_in)
        b1 = np.random.uniform(-bound_b1, bound_b1, size=self.mlp_hidden_dim)

        fan_in_hidden = self.mlp_hidden_dim
        bound_w2 = np.sqrt(6.0 / fan_in_hidden)
        W2 = np.random.uniform(-bound_w2, bound_w2, size=(self.mlp_hidden_dim, 1))

        bound_b2 = 1.0 / np.sqrt(fan_in_hidden)
        b2 = np.random.uniform(-bound_b2, bound_b2, size=(1,))

        # Pad W1 with zeros for non-XY dimensions
        W1 = np.zeros((self.state_dim, self.mlp_hidden_dim))
        W1[:self.xy_dim, :] = W1_xy

        return RandomMLPReward(W1, b1, W2, b2)


def create_reward_prior(
    dataset_states: np.ndarray,
    state_dim: int,
    domain: Optional[str] = None,
    reward_families: Optional[List[str]] = None,
    **kwargs,
) -> RewardPrior:
    """Factory function to create a RewardPrior, optionally domain-specific.

    Args:
        dataset_states: numpy array of all dataset states (N, state_dim)
        state_dim: state dimensionality
        domain: optional domain name ("antmaze", "walker", "cheetah", "kitchen")
        reward_families: optional list of reward families to include
        **kwargs: additional arguments passed to RewardPrior constructor

    Returns:
        RewardPrior instance
    """
    if domain == "antmaze" and kwargs.get("use_xy_prior", False):
        return AntMazeXYPrior(dataset_states, state_dim, reward_families=reward_families, **kwargs)
    else:
        return RewardPrior(dataset_states, state_dim, reward_families=reward_families, **kwargs)