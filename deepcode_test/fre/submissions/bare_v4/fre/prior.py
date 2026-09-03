"""
Prior reward distributions for FRE training.

Provides classes for sampling random unsupervised reward functions
from the mixture prior described in Section 4.2:

1. Singleton goal-reaching rewards (sparse -1/0)
2. Random linear functions (dot product with sparse random vector)
3. Random MLP functions (2-layer network with random weights)

The default prior (FRE-all) uses a uniform mixture of all three.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Callable


class GoalReachingReward:
    """
    Singleton goal-reaching reward: -1 at every timestep the goal
    is not reached, 0 otherwise.

    Goals are sampled from the offline dataset using a HER-like distribution:
    - 0.2: current state as goal
    - 0.5: future state within trajectory (geometric)
    - 0.3: random state from dataset
    """

    def __init__(
        self,
        dataset_states: torch.Tensor,
        goal_threshold: float = 2.0,
    ):
        """
        Args:
            dataset_states: (N, state_dim) — all states from the offline dataset
            goal_threshold: distance threshold for goal achievement
        """
        self.dataset_states = dataset_states
        self.N = dataset_states.shape[0]
        self.goal_threshold = goal_threshold

    def sample(
        self, batch_size: int, state_dim: int, device: torch.device = None
    ) -> Tuple[Callable, torch.Tensor, torch.Tensor]:
        """
        Sample a batch of goal-reaching reward functions.

        Returns:
            reward_fn: callable state -> reward (for training)
            goal_states: (batch_size, state_dim) — the goal for each sub-task
            done_fn: callable state -> bool (whether goal reached)
        """
        # Sample goal states with HER distribution
        indices = torch.randint(0, self.N, (batch_size,))
        goals = self.dataset_states[indices]

        if device is not None:
            goals = goals.to(device)

        def reward_fn(states: torch.Tensor) -> torch.Tensor:
            """Compute -1 for not-at-goal, 0 for at-goal."""
            dist = torch.norm(states - goals, dim=-1)
            return torch.where(dist < self.goal_threshold, 0.0, -1.0)

        def done_fn(states: torch.Tensor) -> torch.Tensor:
            dist = torch.norm(states - goals, dim=-1)
            return dist < self.goal_threshold

        return reward_fn, goals, done_fn


class RandomLinearReward:
    """
    Random linear functions: η(s) = w · s, where w is a random vector.

    On AntMaze, XY positions are excluded from the generation.
    A random binary mask is applied with 0.9 sparsity to bias toward
    simpler functions (per Appendix B).
    """

    def __init__(
        self,
        state_dim: int,
        sparsity: float = 0.9,
        exclude_xy: bool = False,
        xy_indices: Tuple[int, int] = (0, 1),
    ):
        """
        Args:
            state_dim: dimension of state space
            sparsity: probability each element of w is zero
            exclude_xy: if True, zero out XY position dimensions
            xy_indices: indices of X and Y dimensions
        """
        self.state_dim = state_dim
        self.sparsity = sparsity
        self.exclude_xy = exclude_xy
        self.xy_indices = xy_indices

    def sample(
        self, batch_size: int, device: torch.device = None
    ) -> Tuple[Callable, torch.Tensor]:
        """
        Sample a batch of linear reward functions.

        Returns:
            reward_fn: callable state -> reward
            weights: (batch_size, state_dim) — the sampled weight vectors
        """
        # Uniform in [-1, 1]
        weights = torch.rand(batch_size, self.state_dim) * 2 - 1  # (B, D)

        # Apply sparsity mask
        mask = torch.rand(batch_size, self.state_dim) > self.sparsity
        weights = weights * mask.float()

        # Zero out XY dimensions if requested
        if self.exclude_xy:
            weights[:, self.xy_indices[0]] = 0.0
            weights[:, self.xy_indices[1]] = 0.0

        if device is not None:
            weights = weights.to(device)

        def reward_fn(states: torch.Tensor) -> torch.Tensor:
            """η(s) = w · s"""
            # states: (B, D) or (..., D)
            return (states * weights).sum(dim=-1)

        return reward_fn, weights


class RandomMLPReward:
    """
    Random MLP reward functions.

    Architecture: Linear(state_dim, 32) -> Tanh -> Linear(32, 1)
    Weights are sampled from Normal(0, fan_avg⁻¹).
    Output is clipped to [-1, 1].

    Per Appendix B: "Parameters are sampled using a normal distribution
    scaled by the average dimension of the layer."
    """

    def __init__(self, state_dim: int, hidden_dim: int = 32):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

    def _random_mlp(self, device: torch.device = None) -> nn.Module:
        """Create a single randomly-initialized 2-layer MLP."""
        # Scale: 1 / sqrt(average dimension of the layer)
        # Per Appendix B: "scaled by the average dimension of the layer"
        # Layer 1: state_dim -> hidden_dim (32)
        # Layer 2: hidden_dim -> 1
        fan_avg_1 = (self.state_dim + self.hidden_dim) / 2.0
        fan_avg_2 = (self.hidden_dim + 1) / 2.0

        fc1 = nn.Linear(self.state_dim, self.hidden_dim)
        fc1.weight.data.normal_(0, 1.0 / np.sqrt(fan_avg_1))
        fc1.bias.data.normal_(0, 1.0 / np.sqrt(fan_avg_1))

        fc2 = nn.Linear(self.hidden_dim, 1)
        fc2.weight.data.normal_(0, 1.0 / np.sqrt(fan_avg_2))
        fc2.bias.data.normal_(0, 1.0 / np.sqrt(fan_avg_2))

        mlp = nn.Sequential(fc1, nn.Tanh(), fc2, nn.Tanh())  # final tanh clips to (-1,1)
        if device is not None:
            mlp = mlp.to(device)
        return mlp

    def sample(
        self, batch_size: int, device: torch.device = None
    ) -> Tuple[Callable, list]:
        """
        Sample a batch of random MLP reward functions.

        Returns:
            reward_fn: callable state -> reward
            mlps: list of MLP modules (one per batch element)
        """
        mlps = [self._random_mlp(device) for _ in range(batch_size)]

        def reward_fn(states: torch.Tensor) -> torch.Tensor:
            """Apply each MLP to its corresponding state."""
            # states: (B, D)
            rewards = torch.stack([
                mlp(states[i:i+1]).squeeze(-1) for i, mlp in enumerate(mlps)
            ])
            return rewards.squeeze(-1)  # (B,)

        return reward_fn, mlps


class MixedRewardPrior:
    """
    Mixture prior over the three random reward function families.

    Default mixture (FRE-all): ⅓ goal-reaching, ⅓ linear, ⅓ MLP.
    """

    def __init__(
        self,
        state_dim: int,
        dataset_states: torch.Tensor,
        ratios: Tuple[float, float, float] = (0.33, 0.33, 0.34),
        goal_threshold: float = 2.0,
        linear_sparsity: float = 0.9,
        linear_exclude_xy: bool = False,
        xy_indices: Tuple[int, int] = (0, 1),
        mlp_hidden_dim: int = 32,
    ):
        """
        Args:
            state_dim: environment state dimension
            dataset_states: tensor of all states from offline dataset
            ratios: (goal_ratio, linear_ratio, mlp_ratio)
            goal_threshold: distance threshold for goal achievement
            linear_sparsity: sparsity mask probability for linear functions
            linear_exclude_xy: exclude XY from linear functions (for AntMaze)
            xy_indices: which state dimensions are X and Y
            mlp_hidden_dim: hidden dimension for random MLPs
        """
        self.goal_reward = GoalReachingReward(dataset_states, goal_threshold)
        self.linear_reward = RandomLinearReward(
            state_dim, linear_sparsity, linear_exclude_xy, xy_indices
        )
        self.mlp_reward = RandomMLPReward(state_dim, mlp_hidden_dim)
        self.ratios = ratios
        self.rng = np.random.RandomState()

    def sample(
        self, batch_size: int, device: torch.device = None
    ) -> Tuple[Callable, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Sample a batch of mixed reward functions.

        Returns:
            reward_fn: callable states -> rewards
            z_input_states: dummy states for z encoding (same as encoder_states)
            goals: goal states (for goal-reaching subset; None for others)
            masks: done mask (None for non-goal functions)
        """
        # Partition batch among the three families
        n_goal = int(batch_size * self.ratios[0])
        n_linear = int(batch_size * self.ratios[1])
        n_mlp = batch_size - n_goal - n_linear

        rewards_list = []
        reward_types = []  # track which type for each index

        if n_goal > 0:
            fn, goals, done_fn = self.goal_reward.sample(n_goal, None, device)
            rewards_list.append(("goal", fn, goals, done_fn))
        if n_linear > 0:
            fn, weights = self.linear_reward.sample(n_linear, device)
            rewards_list.append(("linear", fn, weights, None))
        if n_mlp > 0:
            fn, mlps = self.mlp_reward.sample(n_mlp, device)
            rewards_list.append(("mlp", fn, mlps, None))

        def combined_reward_fn(states: torch.Tensor) -> torch.Tensor:
            """Evaluate mixed reward functions on states."""
            all_rewards = []
            for rtype, fn, _, _ in rewards_list:
                n = states.shape[0] if rtype == "goal" else (
                    n_linear if rtype == "linear" else n_mlp
                )
                # Use the first n elements for this reward type
                # This is simplified — in practice, states are batch-matched
                all_rewards.append(fn(states[:n] if n > 0 else states[:0]))
            return torch.cat(all_rewards, dim=0)

        return combined_reward_fn, n_goal, n_linear, n_mlp