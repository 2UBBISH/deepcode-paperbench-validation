"""
Replay buffer for offline RL in the FRE framework.

Stores transitions from an offline dataset and provides sampling
functionality for IQL training. Rewards are computed on-the-fly
by reward functions during training, not stored in the buffer.
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union
import torch

from data.dataset import OfflineDataset


class ReplayBuffer:
    """
    Replay buffer that wraps an OfflineDataset for RL training.
    
    Provides batch sampling of (state, action, reward, next_state, done) tuples.
    Rewards are computed externally and can be injected during sampling or
    the buffer can return raw transitions for external reward computation.
    
    Attributes:
        dataset: The underlying OfflineDataset.
        device: Torch device for tensor placement.
        state_dim: Dimension of state space.
        action_dim: Dimension of action space.
        size: Number of transitions in the buffer.
    """
    
    def __init__(
        self,
        dataset: OfflineDataset,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the replay buffer.
        
        Args:
            dataset: OfflineDataset containing states, actions, next_states, terminals.
            device: Torch device to place tensors on (default: CPU).
        """
        self.dataset = dataset
        self.device = device if device is not None else torch.device("cpu")
        
        # Cache dataset properties
        self._states = dataset.states
        self._actions = dataset.actions
        self._next_states = dataset.next_states
        self._terminals = dataset.terminals
        self._timeouts = dataset.timeouts if hasattr(dataset, 'timeouts') else None
        
        self.size = len(self._states)
        self.state_dim = self._states.shape[-1]
        self.action_dim = self._actions.shape[-1]
        
        # Pre-compute normalization stats
        self.state_mean, self.state_std = dataset.get_normalization_stats()
    
    def sample(
        self,
        batch_size: int,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size: Number of transitions to sample.
            rng: Optional random state for reproducibility.
            
        Returns:
            Dictionary with keys:
                'states': (batch_size, state_dim)
                'actions': (batch_size, action_dim)
                'next_states': (batch_size, state_dim)
                'terminals': (batch_size,)
                'timeouts': (batch_size,) if available, else None
        """
        return self.dataset.sample(batch_size, rng=rng)
    
    def sample_states(
        self,
        batch_size: int,
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        """
        Sample only states from the buffer (for FRE encoding/decoding).
        
        Args:
            batch_size: Number of states to sample.
            rng: Optional random state for reproducibility.
            
        Returns:
            Array of shape (batch_size, state_dim).
        """
        return self.dataset.sample_states(batch_size, rng=rng)
    
    def get_all_states(self) -> np.ndarray:
        """
        Get all states in the buffer.
        
        Returns:
            Array of shape (size, state_dim).
        """
        return self.dataset.get_all_states()
    
    def sample_with_rewards(
        self,
        batch_size: int,
        reward_fn,
        rng: Optional[np.random.RandomState] = None,
        return_tensors: bool = True,
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        Sample a batch and compute rewards using the given reward function.
        
        This is the primary sampling method for RL training: it samples
        transitions and computes rewards on-the-fly.
        
        Args:
            batch_size: Number of transitions to sample.
            reward_fn: Callable that maps states -> rewards (batch of states -> batch of rewards).
            rng: Optional random state for reproducibility.
            return_tensors: If True, return torch tensors on self.device.
            
        Returns:
            Dictionary with keys:
                'states': (batch_size, state_dim)
                'actions': (batch_size, action_dim)
                'rewards': (batch_size,)
                'next_states': (batch_size, state_dim)
                'terminals': (batch_size,)
                'timeouts': (batch_size,) if available
        """
        batch = self.sample(batch_size, rng=rng)
        
        # Compute rewards for current states
        states = batch['states']
        if isinstance(states, torch.Tensor):
            states_np = states.cpu().numpy()
        else:
            states_np = states
        
        # Compute rewards using the reward function
        rewards = reward_fn(states_np)
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.cpu().numpy()
        rewards = rewards.reshape(-1)
        
        batch['rewards'] = rewards
        
        if return_tensors:
            batch = self._to_tensors(batch)
        
        return batch
    
    def sample_rl_batch(
        self,
        batch_size: int,
        reward_fn,
        rng: Optional[np.random.RandomState] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch for RL training and return as tuple of tensors.
        
        Convenience method that returns (states, actions, rewards, next_states, dones).
        
        Args:
            batch_size: Number of transitions to sample.
            reward_fn: Callable that maps states -> rewards.
            rng: Optional random state.
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as torch tensors.
        """
        batch = self.sample_with_rewards(
            batch_size, reward_fn, rng=rng, return_tensors=True
        )
        
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['terminals'].float()  # Convert to float for masking
        
        return states, actions, rewards, next_states, dones
    
    def _to_tensors(self, batch: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Convert numpy arrays in batch to torch tensors on the correct device."""
        tensor_batch = {}
        for key, value in batch.items():
            if value is not None:
                if isinstance(value, np.ndarray):
                    tensor_batch[key] = torch.from_numpy(value).float().to(self.device)
                elif isinstance(value, torch.Tensor):
                    tensor_batch[key] = value.float().to(self.device)
                else:
                    tensor_batch[key] = value
            else:
                tensor_batch[key] = None
        return tensor_batch
    
    def get_normalization_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get state normalization statistics.
        
        Returns:
            Tuple of (state_mean, state_std).
        """
        return self.state_mean, self.state_std
    
    def normalize_states(self, states: np.ndarray) -> np.ndarray:
        """
        Normalize states using dataset statistics.
        
        Args:
            states: Array of shape (..., state_dim).
            
        Returns:
            Normalized states.
        """
        return (states - self.state_mean) / (self.state_std + 1e-8)
    
    def denormalize_states(self, states: np.ndarray) -> np.ndarray:
        """
        Denormalize states back to original scale.
        
        Args:
            states: Array of shape (..., state_dim).
            
        Returns:
            Denormalized states.
        """
        return states * (self.state_std + 1e-8) + self.state_mean
    
    def to(self, device: torch.device) -> "ReplayBuffer":
        """
        Move the buffer to a different device.
        
        Note: The underlying data remains in numpy; this only affects
        the default device for tensor conversion.
        
        Args:
            device: Target torch device.
            
        Returns:
            Self (for chaining).
        """
        self.device = device
        return self
    
    def __len__(self) -> int:
        return self.size
    
    def __repr__(self) -> str:
        return (
            f"ReplayBuffer(size={self.size}, state_dim={self.state_dim}, "
            f"action_dim={self.action_dim}, device={self.device})"
        )


def create_replay_buffer(
    dataset: Union[OfflineDataset, str],
    device: Optional[torch.device] = None,
    normalize_states: bool = True,
    data_path: Optional[str] = None,
) -> ReplayBuffer:
    """
    Factory function to create a ReplayBuffer from a dataset or dataset name.
    
    Args:
        dataset: Either an OfflineDataset instance or a dataset name string
                 (e.g., 'antmaze-large-diverse-v2', 'exorl-walker').
        device: Torch device for tensor placement.
        normalize_states: Whether to normalize states (only used if dataset is a string).
        data_path: Path to ExORL data (only used if dataset is a string).
        
    Returns:
        ReplayBuffer instance.
    """
    if isinstance(dataset, str):
        from data.dataset import load_dataset
        dataset = load_dataset(
            dataset, normalize_states=normalize_states, data_path=data_path
        )
    
    return ReplayBuffer(dataset, device=device)