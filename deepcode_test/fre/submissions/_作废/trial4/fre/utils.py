"""
Utility functions and replay buffer for Functional Reward Encodings (FRE).

Provides:
- ReplayBuffer: stores and samples offline RL transitions
- Dataset loading: D4RL (AntMaze, Kitchen) and ExORL (Walker, Cheetah)
- State normalization utilities
- Helper functions for training
"""

import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any, List
from collections import namedtuple
import os
import pickle

# Transition namedtuple for clarity
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """
    Replay buffer for offline RL datasets.
    
    Stores transitions (s, a, r, s', done) and supports:
    - Random sampling of batches for IQL training
    - Sampling of states only (for FRE encoder/decoder training)
    - State normalization computation
    """
    
    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        capacity: Optional[int] = None,
    ):
        """
        Initialize replay buffer from dataset arrays.
        
        Args:
            states: (N, state_dim) array of states
            actions: (N, action_dim) array of actions
            rewards: (N,) array of rewards
            next_states: (N, state_dim) array of next states
            dones: (N,) array of terminal flags
            capacity: maximum capacity (defaults to N)
        """
        self.states = states.astype(np.float32)
        self.actions = actions.astype(np.float32)
        self.rewards = rewards.astype(np.float32).reshape(-1)
        self.next_states = next_states.astype(np.float32)
        self.dones = dones.astype(np.float32).reshape(-1)
        
        self.size = len(states)
        self.capacity = capacity if capacity is not None else self.size
        
        # State statistics for normalization
        self.state_mean = np.mean(self.states, axis=0)
        self.state_std = np.std(self.states, axis=0) + 1e-6
        
        # Action statistics
        self.action_mean = np.mean(self.actions, axis=0) if len(self.actions.shape) > 1 else np.mean(self.actions)
        self.action_std = np.std(self.actions, axis=0) + 1e-6 if len(self.actions.shape) > 1 else np.std(self.actions) + 1e-6
        
        # Device for tensor conversion (set later)
        self.device = torch.device('cpu')
        
    def to(self, device: torch.device):
        """Set the device for tensor conversion."""
        self.device = device
        return self
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a random batch of transitions for IQL training.
        
        Args:
            batch_size: number of transitions to sample
            
        Returns:
            dict with keys: 'states', 'actions', 'rewards', 'next_states', 'dones'
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            'states': torch.as_tensor(self.states[indices], device=self.device),
            'actions': torch.as_tensor(self.actions[indices], device=self.device),
            'rewards': torch.as_tensor(self.rewards[indices], device=self.device),
            'next_states': torch.as_tensor(self.next_states[indices], device=self.device),
            'dones': torch.as_tensor(self.dones[indices], device=self.device),
        }
    
    def sample_states(self, batch_size: int) -> np.ndarray:
        """
        Sample random states from the dataset (for FRE encoder/decoder training).
        
        Args:
            batch_size: number of states to sample
            
        Returns:
            (batch_size, state_dim) numpy array of states
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        return self.states[indices].copy()
    
    def sample_states_torch(self, batch_size: int) -> torch.Tensor:
        """
        Sample random states as torch tensor.
        
        Args:
            batch_size: number of states to sample
            
        Returns:
            (batch_size, state_dim) torch tensor
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        return torch.as_tensor(self.states[indices], device=self.device)
    
    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state using dataset statistics."""
        return (state - self.state_mean) / self.state_std
    
    def normalize_states(self, states: np.ndarray) -> np.ndarray:
        """Normalize batch of states."""
        return (states - self.state_mean) / self.state_std
    
    def unnormalize_state(self, state: np.ndarray) -> np.ndarray:
        """Unnormalize state."""
        return state * self.state_std + self.state_mean
    
    @property
    def state_dim(self) -> int:
        return self.states.shape[1]
    
    @property
    def action_dim(self) -> int:
        if len(self.actions.shape) > 1:
            return self.actions.shape[1]
        return 1
    
    def __len__(self) -> int:
        return self.size


def load_d4rl_dataset(env_name: str) -> ReplayBuffer:
    """
    Load a D4RL dataset and create a ReplayBuffer.
    
    Supports:
    - antmaze-umaze-v2, antmaze-umaze-diverse-v2
    - antmaze-medium-play-v2, antmaze-medium-diverse-v2
    - antmaze-large-play-v2, antmaze-large-diverse-v2
    - kitchen-complete-v0, kitchen-partial-v0, kitchen-mixed-v0
    
    Args:
        env_name: D4RL environment name
        
    Returns:
        ReplayBuffer containing the dataset
    """
    try:
        import d4rl
        import gym
        
        env = gym.make(env_name)
        dataset = env.get_dataset()
        
        # D4RL datasets have different structures; handle both
        if 'observations' in dataset:
            states = dataset['observations']
        elif 'states' in dataset:
            states = dataset['states']
        else:
            raise KeyError(f"Dataset for {env_name} has no 'observations' or 'states' key")
        
        actions = dataset['actions']
        
        if 'rewards' in dataset:
            rewards = dataset['rewards']
        else:
            rewards = np.zeros(len(states))
        
        if 'next_observations' in dataset:
            next_states = dataset['next_observations']
        elif 'next_states' in dataset:
            next_states = dataset['next_states']
        else:
            # Infer next states from shifted observations
            next_states = np.roll(states, -1, axis=0)
            next_states[-1] = states[-1]
        
        if 'terminals' in dataset:
            dones = dataset['terminals']
        elif 'dones' in dataset:
            dones = dataset['dones']
        else:
            dones = np.zeros(len(states), dtype=np.float32)
            # Mark episode boundaries
            if 'timeouts' in dataset:
                dones = np.logical_or(dataset.get('terminals', np.zeros_like(dones)), 
                                      dataset['timeouts']).astype(np.float32)
        
        env.close()
        
        return ReplayBuffer(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
        )
        
    except ImportError:
        raise ImportError("D4RL is not installed. Install with: pip install d4rl")
    except Exception as e:
        raise RuntimeError(f"Failed to load D4RL dataset {env_name}: {e}")


def load_exorl_dataset(dataset_path: str, domain: str) -> ReplayBuffer:
    """
    Load an ExORL dataset from disk.
    
    ExORL datasets are typically stored as .npy or .npz files containing
    unsupervised exploration data from Walker and Cheetah environments.
    
    Expected format: directory with 'states.npy', 'actions.npy', 'rewards.npy',
    'next_states.npy', 'dones.npy' or a single .npz file.
    
    Args:
        dataset_path: path to the ExORL dataset directory or .npz file
        domain: 'walker' or 'cheetah'
        
    Returns:
        ReplayBuffer containing the dataset
    """
    if os.path.isdir(dataset_path):
        # Load from directory of .npy files
        states = np.load(os.path.join(dataset_path, 'states.npy'))
        actions = np.load(os.path.join(dataset_path, 'actions.npy'))
        rewards = np.load(os.path.join(dataset_path, 'rewards.npy'))
        next_states = np.load(os.path.join(dataset_path, 'next_states.npy'))
        dones = np.load(os.path.join(dataset_path, 'dones.npy'))
    elif dataset_path.endswith('.npz'):
        # Load from .npz file
        data = np.load(dataset_path)
        states = data['states']
        actions = data['actions']
        rewards = data.get('rewards', np.zeros(len(states)))
        next_states = data['next_states']
        dones = data.get('dones', np.zeros(len(states)))
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path}")
    
    return ReplayBuffer(
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
    )


def load_dataset(dataset_name: str, data_dir: str = "./data") -> ReplayBuffer:
    """
    Unified dataset loading function.
    
    Automatically detects whether the dataset is D4RL or ExORL based on name.
    
    Args:
        dataset_name: name of the dataset (e.g., 'antmaze-large-diverse-v2', 'exorl_walker')
        data_dir: directory where datasets are stored
        
    Returns:
        ReplayBuffer containing the dataset
    """
    if dataset_name.startswith('antmaze') or dataset_name.startswith('kitchen'):
        return load_d4rl_dataset(dataset_name)
    elif dataset_name.startswith('exorl'):
        # Parse domain from name: exorl_walker, exorl_cheetah
        parts = dataset_name.split('_', 1)
        if len(parts) == 2:
            domain = parts[1]
        else:
            domain = 'walker'
        
        dataset_path = os.path.join(data_dir, dataset_name)
        return load_exorl_dataset(dataset_path, domain)
    else:
        # Try D4RL first, then fallback to file loading
        try:
            return load_d4rl_dataset(dataset_name)
        except (ImportError, RuntimeError):
            dataset_path = os.path.join(data_dir, dataset_name)
            if os.path.exists(dataset_path):
                return load_exorl_dataset(dataset_path, 'unknown')
            raise ValueError(f"Unknown dataset: {dataset_name}")


class StateNormalizer:
    """
    Online state normalizer using running mean and std.
    Useful for normalizing states during evaluation rollouts.
    """
    
    def __init__(self, state_dim: int, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None):
        self.state_dim = state_dim
        if mean is not None and std is not None:
            self.mean = mean
            self.std = std
            self.frozen = True
        else:
            self.mean = np.zeros(state_dim, dtype=np.float32)
            self.std = np.ones(state_dim, dtype=np.float32)
            self.frozen = False
            self.count = 0
    
    def update(self, states: np.ndarray):
        """Update running statistics with new states."""
        if self.frozen:
            return
        
        batch_mean = np.mean(states, axis=0)
        batch_var = np.var(states, axis=0)
        batch_count = len(states)
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / total_count
        # Welford's online variance update
        m2_old = self.std ** 2 * self.count
        m2_new = batch_var * batch_count
        self.std = np.sqrt((m2_old + m2_new + delta**2 * self.count * batch_count / total_count) / total_count + 1e-6)
        self.count = total_count
    
    def normalize(self, state: np.ndarray) -> np.ndarray:
        """Normalize a single state."""
        return (state - self.mean) / (self.std + 1e-6)
    
    def freeze(self):
        """Freeze the normalizer statistics."""
        self.frozen = True


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_reward_statistics(replay_buffer: ReplayBuffer) -> Dict[str, float]:
    """
    Compute reward statistics from the replay buffer for reward discretization.
    
    Returns:
        dict with 'min', 'max', 'mean', 'std' of rewards
    """
    rewards = replay_buffer.rewards
    return {
        'min': float(np.min(rewards)),
        'max': float(np.max(rewards)),
        'mean': float(np.mean(rewards)),
        'std': float(np.std(rewards)),
    }


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')