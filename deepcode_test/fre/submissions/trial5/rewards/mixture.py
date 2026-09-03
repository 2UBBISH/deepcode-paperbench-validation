"""
Mixture Reward Distribution for FRE (Functional Reward Encodings).

Implements a uniform mixture over three reward function families:
  - Singleton (goal-reaching): reward = -1 until goal reached, 0 otherwise
  - Random Linear: reward(s) = dot(w, s) with sparse weight vector
  - Random MLP: reward(s) = MLP(s) with random frozen weights

Used during training to sample diverse unsupervised reward functions
for the FRE encoder and RL agent.
"""

from typing import Optional, List, Tuple
import numpy as np
import torch

from rewards.base import RewardFunction
from rewards.singleton import SingletonReward
from rewards.linear import RandomLinearReward
from rewards.mlp import RandomMLPReward


class MixtureRewardDistribution:
    """
    Uniform mixture distribution over three reward function families.

    On each call to sample(), randomly selects one of the three families
    with equal probability (1/3 each), then instantiates a fresh reward
    function from that family with random parameters.

    This implements the prior distribution p(η) described in the paper,
    which is used to generate diverse unsupervised reward functions for
    training the FRE encoder and the downstream RL agent.

    Attributes:
        state_dim: Dimensionality of the state space.
        device: Torch device for reward computation.
        singleton_threshold: L2 distance threshold for goal-reaching.
        linear_sparsity: Fraction of zero weights in linear rewards.
        mlp_hidden_dim: Hidden dimension for random MLP rewards.
    """

    def __init__(
        self,
        state_dim: int,
        device: Optional[torch.device] = None,
        singleton_threshold: float = 0.5,
        linear_sparsity: float = 0.8,
        mlp_hidden_dim: int = 256,
    ):
        """
        Initialize the mixture distribution.

        Args:
            state_dim: Dimensionality of state vectors.
            device: Torch device (CPU or CUDA).
            singleton_threshold: Distance threshold for goal-reaching reward.
            linear_sparsity: Fraction of zero entries in linear weight vectors.
            mlp_hidden_dim: Hidden layer size for random MLP rewards.
        """
        self.state_dim = state_dim
        self.device = device
        self.singleton_threshold = singleton_threshold
        self.linear_sparsity = linear_sparsity
        self.mlp_hidden_dim = mlp_hidden_dim

        # Family names for logging / introspection
        self.family_names = ["singleton", "linear", "mlp"]

    def sample(self, dataset_states: Optional[np.ndarray] = None) -> RewardFunction:
        """
        Sample a random reward function from the uniform mixture.

        With probability 1/3 each, returns a SingletonReward,
        RandomLinearReward, or RandomMLPReward. For singleton rewards,
        a goal state is sampled uniformly from the provided dataset states.

        Args:
            dataset_states: Optional numpy array of shape (N, state_dim)
                used to sample a goal for singleton rewards. Required
                if the sampled family is singleton; ignored otherwise.

        Returns:
            A RewardFunction instance ready to compute rewards on states.

        Raises:
            ValueError: If singleton is sampled but dataset_states is None.
        """
        # Uniform selection among the three families
        family_idx = np.random.randint(0, 3)

        if family_idx == 0:
            # Singleton (goal-reaching)
            if dataset_states is None:
                raise ValueError(
                    "dataset_states must be provided when sampling a singleton reward."
                )
            goal = SingletonReward.sample_goal_from_dataset(dataset_states)
            reward_fn = SingletonReward(
                state_dim=self.state_dim,
                goal=torch.from_numpy(goal).float(),
                threshold=self.singleton_threshold,
                device=self.device,
            )
        elif family_idx == 1:
            # Random Linear
            reward_fn = RandomLinearReward(
                state_dim=self.state_dim,
                sparsity=self.linear_sparsity,
                device=self.device,
            )
        else:
            # Random MLP
            reward_fn = RandomMLPReward(
                state_dim=self.state_dim,
                hidden_dim=self.mlp_hidden_dim,
                device=self.device,
            )

        return reward_fn

    def sample_batch(
        self,
        batch_size: int,
        dataset_states: Optional[np.ndarray] = None,
    ) -> List[RewardFunction]:
        """
        Sample multiple reward functions at once.

        Args:
            batch_size: Number of reward functions to sample.
            dataset_states: Optional dataset states for singleton goals.

        Returns:
            List of RewardFunction instances.
        """
        return [self.sample(dataset_states) for _ in range(batch_size)]

    def sample_with_family(
        self, dataset_states: Optional[np.ndarray] = None
    ) -> Tuple[RewardFunction, str]:
        """
        Sample a reward function and return which family it came from.

        Args:
            dataset_states: Optional dataset states for singleton goals.

        Returns:
            Tuple of (RewardFunction, family_name_string).
        """
        family_idx = np.random.randint(0, 3)
        family_name = self.family_names[family_idx]

        if family_idx == 0:
            if dataset_states is None:
                raise ValueError(
                    "dataset_states must be provided when sampling a singleton reward."
                )
            goal = SingletonReward.sample_goal_from_dataset(dataset_states)
            reward_fn = SingletonReward(
                state_dim=self.state_dim,
                goal=torch.from_numpy(goal).float(),
                threshold=self.singleton_threshold,
                device=self.device,
            )
        elif family_idx == 1:
            reward_fn = RandomLinearReward(
                state_dim=self.state_dim,
                sparsity=self.linear_sparsity,
                device=self.device,
            )
        else:
            reward_fn = RandomMLPReward(
                state_dim=self.state_dim,
                hidden_dim=self.mlp_hidden_dim,
                device=self.device,
            )

        return reward_fn, family_name

    def sample_family(self, family_name: str) -> RewardFunction:
        """
        Sample a reward function from a specific family (for controlled experiments).

        Args:
            family_name: One of "singleton", "linear", "mlp".

        Returns:
            A RewardFunction from the specified family.

        Raises:
            ValueError: If family_name is not recognized.
        """
        if family_name == "singleton":
            return SingletonReward(
                state_dim=self.state_dim,
                threshold=self.singleton_threshold,
                device=self.device,
            )
        elif family_name == "linear":
            return RandomLinearReward(
                state_dim=self.state_dim,
                sparsity=self.linear_sparsity,
                device=self.device,
            )
        elif family_name == "mlp":
            return RandomMLPReward(
                state_dim=self.state_dim,
                hidden_dim=self.mlp_hidden_dim,
                device=self.device,
            )
        else:
            raise ValueError(
                f"Unknown family '{family_name}'. "
                f"Must be one of {self.family_names}."
            )

    def to(self, device: torch.device) -> "MixtureRewardDistribution":
        """
        Move the distribution to a new device.

        Args:
            device: Target torch device.

        Returns:
            Self with updated device.
        """
        self.device = device
        return self

    def __repr__(self) -> str:
        return (
            f"MixtureRewardDistribution(state_dim={self.state_dim}, "
            f"families={self.family_names}, device={self.device})"
        )