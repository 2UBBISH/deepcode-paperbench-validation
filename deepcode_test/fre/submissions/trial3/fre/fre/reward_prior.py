"""
Prior Reward Distribution for Functional Reward Encodings (FRE).

Implements a uniform mixture of three unsupervised reward function families:
  1. Singleton goal-reaching rewards
  2. Random linear functions
  3. Random MLP functions

These are used to generate diverse reward functions for pre-training the FRE encoder
and for conditioning the downstream RL agent.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, List


class SingletonReward:
    """
    Singleton goal-reaching reward function.
    
    Samples a goal state s_g uniformly from the dataset.
    Reward: η(s) = -1 if ||s - s_g|| > threshold, 0 otherwise.
    """
    
    def __init__(self, goal_state: np.ndarray, threshold: float = 0.5):
        """
        Args:
            goal_state: The goal state vector (d_s,).
            threshold: Distance threshold for goal reaching.
        """
        self.goal_state = goal_state.copy()
        self.threshold = threshold
    
    def __call__(self, states: np.ndarray) -> np.ndarray:
        """
        Compute rewards for a batch of states.
        
        Args:
            states: Array of shape (N, d_s) or (d_s,).
            
        Returns:
            rewards: Array of shape (N,) or scalar.
        """
        states = np.asarray(states)
        if states.ndim == 1:
            dist = np.linalg.norm(states - self.goal_state)
            return np.array(-1.0 if dist > self.threshold else 0.0)
        else:
            dists = np.linalg.norm(states - self.goal_state, axis=1)
            return np.where(dists > self.threshold, -1.0, 0.0)


class LinearReward:
    """
    Random linear reward function.
    
    Samples a random weight vector w ~ Uniform(-1, 1) and applies a sparse mask
    to bias toward simple functions.
    Reward: η(s) = w^T s.
    """
    
    def __init__(self, weights: np.ndarray):
        """
        Args:
            weights: Weight vector of shape (d_s,).
        """
        self.weights = weights.copy()
    
    def __call__(self, states: np.ndarray) -> np.ndarray:
        """
        Compute rewards for a batch of states.
        
        Args:
            states: Array of shape (N, d_s) or (d_s,).
            
        Returns:
            rewards: Array of shape (N,) or scalar.
        """
        states = np.asarray(states)
        return np.dot(states, self.weights)


class MLPReward:
    """
    Random MLP reward function.
    
    A 2-layer MLP with random weights (fixed after initialization).
    Reward: η(s) = MLP(s).
    """
    
    def __init__(self, mlp: nn.Module):
        """
        Args:
            mlp: A PyTorch MLP module with fixed random weights.
        """
        self.mlp = mlp
        # Ensure eval mode and no gradients
        self.mlp.eval()
        for param in self.mlp.parameters():
            param.requires_grad = False
    
    def __call__(self, states: np.ndarray) -> np.ndarray:
        """
        Compute rewards for a batch of states.
        
        Args:
            states: Array of shape (N, d_s) or (d_s,).
            
        Returns:
            rewards: Array of shape (N,) or scalar.
        """
        states = np.asarray(states, dtype=np.float32)
        if states.ndim == 1:
            states = states.reshape(1, -1)
        with torch.no_grad():
            states_t = torch.from_numpy(states)
            rewards_t = self.mlp(states_t)
            rewards = rewards_t.squeeze(-1).numpy()
        if rewards.ndim == 1 and rewards.shape[0] == 1:
            return rewards[0]
        return rewards


class RewardPrior:
    """
    Prior reward distribution p(η): uniform mixture of singleton, linear, and MLP
    reward functions.
    
    For each call to sample(), randomly selects one of the three families with
    equal probability (1/3 each), then samples a function from that family.
    """
    
    def __init__(
        self,
        state_dim: int,
        dataset_states: np.ndarray,
        singleton_threshold: float = 0.5,
        linear_sparsity: float = 0.5,
        mlp_hidden_dim: int = 256,
        mlp_activation: str = "relu",
        seed: Optional[int] = None,
    ):
        """
        Args:
            state_dim: Dimension of state space (d_s).
            dataset_states: Array of all states from the offline dataset (N_dataset, d_s),
                used for sampling goal states for singleton rewards.
            singleton_threshold: Distance threshold for singleton goal-reaching reward.
            linear_sparsity: Probability of zeroing out each weight element (p_mask).
            mlp_hidden_dim: Hidden dimension for random MLP.
            mlp_activation: Activation function for MLP ('relu' or 'tanh').
            seed: Random seed for reproducibility.
        """
        self.state_dim = state_dim
        self.dataset_states = np.asarray(dataset_states, dtype=np.float32)
        self.singleton_threshold = singleton_threshold
        self.linear_sparsity = linear_sparsity
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_activation = mlp_activation
        
        self.rng = np.random.RandomState(seed)
        
        # Pre-compute dataset statistics for normalization
        self.state_mean = self.dataset_states.mean(axis=0)
        self.state_std = self.dataset_states.std(axis=0) + 1e-6
    
    def _sample_singleton(self) -> SingletonReward:
        """Sample a singleton goal-reaching reward function."""
        idx = self.rng.randint(0, len(self.dataset_states))
        goal_state = self.dataset_states[idx].copy()
        return SingletonReward(goal_state, threshold=self.singleton_threshold)
    
    def _sample_linear(self) -> LinearReward:
        """Sample a random linear reward function with sparse mask."""
        # Sample weights from Uniform(-1, 1)
        weights = self.rng.uniform(-1.0, 1.0, size=self.state_dim).astype(np.float32)
        
        # Apply sparse mask: randomly zero out elements
        mask = self.rng.rand(self.state_dim) > self.linear_sparsity
        weights = weights * mask.astype(np.float32)
        
        return LinearReward(weights)
    
    def _sample_mlp(self) -> MLPReward:
        """Sample a random MLP reward function."""
        # Build a 2-layer MLP with random weights
        mlp = nn.Sequential(
            nn.Linear(self.state_dim, self.mlp_hidden_dim),
            nn.ReLU() if self.mlp_activation == "relu" else nn.Tanh(),
            nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim),
            nn.ReLU() if self.mlp_activation == "relu" else nn.Tanh(),
            nn.Linear(self.mlp_hidden_dim, 1),
        )
        
        # Initialize with random weights (default PyTorch init is already random,
        # but we use uniform for consistency with paper)
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -1.0, 1.0)
                nn.init.uniform_(m.bias, -1.0, 1.0)
        
        mlp.apply(_init_weights)
        
        return MLPReward(mlp)
    
    def sample(self) -> Tuple[str, object]:
        """
        Sample a reward function from the prior distribution.
        
        Returns:
            family: String indicating the reward family ('singleton', 'linear', 'mlp').
            reward_fn: A callable reward function object.
        """
        # Uniform mixture: choose family with equal probability
        choice = self.rng.rand()
        
        if choice < 1.0 / 3.0:
            return "singleton", self._sample_singleton()
        elif choice < 2.0 / 3.0:
            return "linear", self._sample_linear()
        else:
            return "mlp", self._sample_mlp()
    
    def compute_rewards(
        self, reward_fn: object, states: np.ndarray
    ) -> np.ndarray:
        """
        Compute rewards for a batch of states using the given reward function.
        
        Args:
            reward_fn: A reward function object (SingletonReward, LinearReward, or MLPReward).
            states: Array of shape (N, d_s).
            
        Returns:
            rewards: Array of shape (N,) with scalar rewards.
        """
        return reward_fn(states)
    
    def sample_and_compute(
        self, states: np.ndarray
    ) -> Tuple[str, np.ndarray]:
        """
        Sample a reward function and compute rewards for the given states.
        
        Convenience method combining sample() and compute_rewards().
        
        Args:
            states: Array of shape (N, d_s).
            
        Returns:
            family: String indicating the reward family.
            rewards: Array of shape (N,) with scalar rewards.
        """
        family, reward_fn = self.sample()
        rewards = self.compute_rewards(reward_fn, states)
        return family, rewards


# ============================================================
# Utility: Build random MLP for reward prior
# ============================================================

def build_random_mlp(
    input_dim: int,
    hidden_dim: int = 256,
    output_dim: int = 1,
    activation: str = "relu",
    num_layers: int = 2,
    seed: Optional[int] = None,
) -> nn.Module:
    """
    Build a random MLP with fixed weights for use as a reward function.
    
    Args:
        input_dim: Input dimension (state dimension).
        hidden_dim: Hidden layer dimension.
        output_dim: Output dimension (1 for scalar reward).
        activation: Activation function ('relu' or 'tanh').
        num_layers: Number of hidden layers.
        seed: Random seed.
        
    Returns:
        mlp: A PyTorch Sequential module with random fixed weights.
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    layers = []
    in_dim = input_dim
    for _ in range(num_layers):
        layers.append(nn.Linear(in_dim, hidden_dim))
        if activation == "relu":
            layers.append(nn.ReLU())
        elif activation == "tanh":
            layers.append(nn.Tanh())
        else:
            raise ValueError(f"Unknown activation: {activation}")
        in_dim = hidden_dim
    
    layers.append(nn.Linear(hidden_dim, output_dim))
    
    mlp = nn.Sequential(*layers)
    
    # Initialize with uniform random weights
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, -1.0, 1.0)
            nn.init.uniform_(m.bias, -1.0, 1.0)
    
    mlp.apply(_init_weights)
    
    # Freeze parameters
    for param in mlp.parameters():
        param.requires_grad = False
    
    mlp.eval()
    
    return mlp


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":
    # Quick test
    print("Testing RewardPrior...")
    
    # Create dummy dataset
    dataset_states = np.random.randn(1000, 10).astype(np.float32)
    
    prior = RewardPrior(
        state_dim=10,
        dataset_states=dataset_states,
        singleton_threshold=0.5,
        linear_sparsity=0.5,
        mlp_hidden_dim=256,
        seed=42,
    )
    
    # Test sampling
    for i in range(5):
        family, reward_fn = prior.sample()
        print(f"Sample {i}: family={family}")
        
        # Test computing rewards
        test_states = np.random.randn(32, 10).astype(np.float32)
        rewards = prior.compute_rewards(reward_fn, test_states)
        print(f"  Rewards shape: {rewards.shape}, range: [{rewards.min():.3f}, {rewards.max():.3f}]")
    
    # Test sample_and_compute
    family, rewards = prior.sample_and_compute(test_states)
    print(f"\nsample_and_compute: family={family}, rewards shape={rewards.shape}")
    
    print("All tests passed!")