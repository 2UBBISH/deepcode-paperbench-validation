"""
Random reward function generators for FRE.
Implements three types of reward functions:
1. Goal-reaching (singleton) rewards
2. Random linear functions
3. Random MLP functions
"""

import numpy as np
import torch
import torch.nn as nn


class GoalReachingReward:
    """Goal-reaching reward function."""

    def __init__(self, goal_state, threshold=2.0, state_indices=None):
        """
        Args:
            goal_state: Target goal state
            threshold: Distance threshold for considering goal reached
            state_indices: Which state dimensions to use for distance (None = all)
        """
        self.goal_state = goal_state
        self.threshold = threshold
        self.state_indices = state_indices

    def __call__(self, states):
        """
        Compute reward for given states.
        Returns -1 if goal not reached, 0 if reached.
        """
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).float()

        goal = torch.from_numpy(self.goal_state).float() if isinstance(self.goal_state, np.ndarray) else self.goal_state

        if self.state_indices is not None:
            states = states[..., self.state_indices]
            goal = goal[self.state_indices]

        # Compute Euclidean distance
        distances = torch.norm(states - goal, dim=-1)
        rewards = torch.where(distances < self.threshold,
                             torch.zeros_like(distances),
                             -torch.ones_like(distances))
        return rewards.numpy() if isinstance(states, torch.Tensor) else rewards


class RandomLinearReward:
    """Random linear reward function."""

    def __init__(self, state_dim, sparsity=0.9, exclude_dims=None):
        """
        Args:
            state_dim: Dimension of state space
            sparsity: Probability of zeroing each dimension
            exclude_dims: Dimensions to exclude (e.g., XY positions for AntMaze)
        """
        self.state_dim = state_dim

        # Sample random weight vector
        self.weights = np.random.uniform(-1, 1, state_dim)

        # Apply sparsity mask
        mask = np.random.binomial(1, 1 - sparsity, state_dim)
        self.weights = self.weights * mask

        # Exclude specific dimensions if specified
        if exclude_dims is not None:
            self.weights[exclude_dims] = 0

    def __call__(self, states):
        """Compute linear reward."""
        if isinstance(states, torch.Tensor):
            states = states.numpy()

        # Ensure states is 2D
        if states.ndim == 1:
            states = states.reshape(1, -1)

        rewards = np.dot(states, self.weights)
        return rewards


class RandomMLPReward(nn.Module):
    """Random MLP reward function."""

    def __init__(self, state_dim, hidden_dim=32):
        """
        Args:
            state_dim: Dimension of state space
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.state_dim = state_dim

        # Two-layer MLP with tanh activation
        self.layer1 = nn.Linear(state_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 1)

        # Initialize with scaled normal distribution
        with torch.no_grad():
            nn.init.normal_(self.layer1.weight, std=1.0 / np.sqrt(state_dim))
            nn.init.normal_(self.layer1.bias, std=1.0 / np.sqrt(state_dim))
            nn.init.normal_(self.layer2.weight, std=1.0 / np.sqrt(hidden_dim))
            nn.init.normal_(self.layer2.bias, std=1.0 / np.sqrt(hidden_dim))

    def __call__(self, states):
        """Compute MLP reward."""
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).float()

        # Ensure states is 2D
        if states.ndim == 1:
            states = states.unsqueeze(0)

        # Forward pass
        x = torch.tanh(self.layer1(states))
        rewards = self.layer2(x).squeeze(-1)

        # Clip to [-1, 1]
        rewards = torch.clamp(rewards, -1, 1)

        return rewards.detach().numpy()


class RewardFunctionSampler:
    """Samples reward functions from a mixture of types."""

    def __init__(self, dataset, state_dim,
                 goal_ratio=0.33, linear_ratio=0.33, mlp_ratio=0.34,
                 goal_threshold=2.0, exclude_dims=None):
        """
        Args:
            dataset: Offline dataset containing trajectories
            state_dim: Dimension of state space
            goal_ratio: Probability of sampling goal-reaching reward
            linear_ratio: Probability of sampling linear reward
            mlp_ratio: Probability of sampling MLP reward
            goal_threshold: Distance threshold for goal-reaching
            exclude_dims: Dimensions to exclude from linear functions
        """
        self.dataset = dataset
        self.state_dim = state_dim
        self.goal_ratio = goal_ratio
        self.linear_ratio = linear_ratio
        self.mlp_ratio = mlp_ratio
        self.goal_threshold = goal_threshold
        self.exclude_dims = exclude_dims

        # Normalize probabilities
        total = goal_ratio + linear_ratio + mlp_ratio
        self.goal_ratio /= total
        self.linear_ratio /= total
        self.mlp_ratio /= total

    def sample_goal_state(self, current_state=None, trajectory=None):
        """
        Sample goal state using hindsight relabeling.
        - 0.2: current state
        - 0.5: future state in trajectory
        - 0.3: random state from dataset
        """
        p = np.random.rand()

        if p < 0.2 and current_state is not None:
            # Use current state as goal
            return current_state
        elif p < 0.7 and trajectory is not None:
            # Sample future state from trajectory
            future_idx = np.random.randint(0, len(trajectory))
            return trajectory[future_idx]
        else:
            # Sample random state from dataset
            idx = np.random.randint(0, len(self.dataset))
            return self.dataset[idx]['observations']

    def sample(self, trajectory=None):
        """
        Sample a random reward function.

        Args:
            trajectory: Optional trajectory for hindsight relabeling

        Returns:
            Callable reward function
        """
        p = np.random.rand()

        if p < self.goal_ratio:
            # Goal-reaching reward
            goal_state = self.sample_goal_state(trajectory=trajectory)
            return GoalReachingReward(goal_state, self.goal_threshold)

        elif p < self.goal_ratio + self.linear_ratio:
            # Random linear reward
            return RandomLinearReward(self.state_dim,
                                     sparsity=0.9,
                                     exclude_dims=self.exclude_dims)
        else:
            # Random MLP reward
            return RandomMLPReward(self.state_dim)
