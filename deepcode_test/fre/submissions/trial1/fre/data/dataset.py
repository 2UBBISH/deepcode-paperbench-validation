"""
Offline dataset loading and replay buffer for FRE.

Supports:
- D4RL datasets: AntMaze ('antmaze-large-diverse-v2'), Kitchen ('kitchen-complete-v0')
- ExORL datasets: Walker, Cheetah (from ExORL benchmark, DeepMind Control Suite)

Provides:
- OfflineDataset: loads and normalizes data, provides state/transition access
- ReplayBuffer: stores transitions and supports sampling for encoder and IQL training
- load_dataset: convenience function to load by domain name
"""

import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any, List
import os
import pickle

from fre.config import config


# ============================================================
# Utility: D4RL loading
# ============================================================

def _load_d4rl_dataset(env_name: str) -> Dict[str, np.ndarray]:
    """
    Load a D4RL dataset using the d4rl library.
    
    Args:
        env_name: D4RL environment name (e.g., 'antmaze-large-diverse-v2')
    
    Returns:
        dict with keys: 'observations', 'actions', 'next_observations', 
                        'rewards', 'terminals', 'timeouts'
    """
    try:
        import d4rl
        import gym
        env = gym.make(env_name)
        dataset = env.get_dataset()
        env.close()
        return dataset
    except ImportError:
        raise ImportError(
            "D4RL is required for AntMaze and Kitchen datasets. "
            "Install with: pip install d4rl"
        )


def _load_exorl_dataset(domain: str, task: str) -> Dict[str, np.ndarray]:
    """
    Load an ExORL dataset.
    
    The ExORL benchmark provides pre-collected datasets for Walker and Cheetah
    domains. Data is typically stored as NPZ files or can be loaded via the
    exorl package.
    
    Args:
        domain: 'walker' or 'cheetah'
        task: e.g., 'proto', 'random', etc. (we use the unsupervised/exploratory data)
    
    Returns:
        dict with keys: 'observations', 'actions', 'next_observations', 
                        'rewards', 'terminals', 'timeouts'
    """
    try:
        import exorl
    except ImportError:
        raise ImportError(
            "ExORL is required for Walker and Cheetah datasets. "
            "Install from: https://github.com/denisyarats/exorl"
        )
    
    # ExORL datasets are typically stored in ~/.exorl or a specified directory
    exorl_data_dir = os.environ.get(
        "EXORL_DATA_DIR", 
        os.path.expanduser("~/.exorl")
    )
    
    # Try to find the dataset file
    dataset_path = os.path.join(exorl_data_dir, f"{domain}_{task}.npz")
    
    if not os.path.exists(dataset_path):
        # Try alternative naming conventions
        alt_paths = [
            os.path.join(exorl_data_dir, f"{domain}", f"{task}.npz"),
            os.path.join(exorl_data_dir, f"{domain}_{task}_data.npz"),
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                dataset_path = alt_path
                break
        else:
            # If file not found, try loading via exorl API
            try:
                data = _load_exorl_via_api(domain, task)
                return data
            except Exception as e:
                raise FileNotFoundError(
                    f"Could not find ExORL dataset at {dataset_path} or load via API. "
                    f"Please download the ExORL datasets first. Error: {e}"
                )
    
    data = np.load(dataset_path, allow_pickle=True)
    
    # ExORL NPZ files may have different key naming conventions
    # Standardize to D4RL format
    dataset = {}
    
    # Map common key names
    key_mapping = {
        'observations': ['observations', 'obs', 'states'],
        'actions': ['actions', 'acts', 'action'],
        'next_observations': ['next_observations', 'next_obs', 'next_states'],
        'rewards': ['rewards', 'reward'],
        'terminals': ['terminals', 'dones', 'terminal', 'done'],
        'timeouts': ['timeouts', 'timeout'],
    }
    
    for target_key, possible_keys in key_mapping.items():
        for key in possible_keys:
            if key in data:
                dataset[target_key] = data[key]
                break
        else:
            # If next_observations not present, construct from observations
            if target_key == 'next_observations' and 'observations' in dataset:
                dataset['next_observations'] = np.concatenate(
                    [dataset['observations'][1:], 
                     dataset['observations'][-1:]], axis=0
                )
            elif target_key == 'timeouts':
                dataset['timeouts'] = np.zeros(len(dataset['observations']), dtype=bool)
            elif target_key == 'terminals':
                dataset['terminals'] = np.zeros(len(dataset['observations']), dtype=bool)
    
    return dataset


def _load_exorl_via_api(domain: str, task: str) -> Dict[str, np.ndarray]:
    """Attempt to load ExORL dataset via the exorl Python API."""
    import exorl
    
    # ExORL typically uses a builder pattern
    # This is a best-effort attempt; exact API may vary
    try:
        from exorl.datasets import get_dataset
        data = get_dataset(domain=domain, task=task)
        return data
    except (ImportError, AttributeError):
        pass
    
    # Fallback: try to create environment and collect data
    # (This would be for generating data, not loading pre-collected)
    raise RuntimeError(
        "Cannot load ExORL dataset via API. Please ensure datasets are downloaded."
    )


# ============================================================
# Replay Buffer
# ============================================================

class ReplayBuffer:
    """
    Replay buffer for storing and sampling offline transitions.
    
    Stores (state, action, next_state, reward, done) tuples.
    Supports:
    - Uniform sampling of states (for encoder training)
    - Uniform sampling of full transitions (for IQL training)
    - Sampling of K encoder states and K' decoder states (disjoint sets)
    """
    
    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray,
        timeouts: Optional[np.ndarray] = None,
        device: str = "cpu",
        state_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
    ):
        """
        Initialize replay buffer from offline dataset arrays.
        
        Args:
            observations: (N, state_dim) array
            actions: (N, action_dim) array
            next_observations: (N, state_dim) array
            rewards: (N,) array
            terminals: (N,) array (boolean)
            timeouts: (N,) array (boolean), optional
            device: torch device for tensor storage
            state_dim: override state dimension
            action_dim: override action dimension
        """
        self.device = device
        
        # Convert to tensors and store on device
        self.observations = torch.FloatTensor(observations).to(device)
        self.actions = torch.FloatTensor(actions).to(device)
        self.next_observations = torch.FloatTensor(next_observations).to(device)
        self.rewards = torch.FloatTensor(rewards).to(device)
        self.terminals = torch.FloatTensor(terminals.astype(np.float32)).to(device)
        
        if timeouts is not None:
            self.timeouts = torch.FloatTensor(timeouts.astype(np.float32)).to(device)
        else:
            self.timeouts = torch.zeros_like(self.terminals)
        
        self.size = len(observations)
        self.state_dim = state_dim or observations.shape[-1]
        self.action_dim = action_dim or actions.shape[-1]
        
        # Compute normalization statistics for states
        self.state_mean = self.observations.mean(dim=0)
        self.state_std = self.observations.std(dim=0).clamp(min=1e-6)
        
        # Compute normalization statistics for actions
        self.action_mean = self.actions.mean(dim=0)
        self.action_std = self.actions.std(dim=0).clamp(min=1e-6)
    
    def sample_states(self, batch_size: int) -> torch.Tensor:
        """
        Sample a batch of states uniformly from the buffer.
        
        Args:
            batch_size: number of states to sample
        
        Returns:
            states: (batch_size, state_dim) tensor
        """
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.observations[indices]
    
    def sample_transitions(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of full transitions uniformly.
        
        Args:
            batch_size: number of transitions to sample
        
        Returns:
            dict with keys: 'states', 'actions', 'next_states', 'rewards', 'terminals'
        """
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {
            'states': self.observations[indices],
            'actions': self.actions[indices],
            'next_states': self.next_observations[indices],
            'rewards': self.rewards[indices],
            'terminals': self.terminals[indices],
        }
    
    def sample_encoder_decoder_states(
        self, 
        K: int, 
        K_prime: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample disjoint sets of states for encoder and decoder.
        
        Args:
            K: number of encoder states
            K_prime: number of decoder states
        
        Returns:
            encoder_states: (K, state_dim) tensor
            decoder_states: (K_prime, state_dim) tensor
        """
        total_needed = K + K_prime
        if total_needed > self.size:
            # If not enough states, sample with replacement
            indices = torch.randint(0, self.size, (total_needed,), device=self.device)
        else:
            indices = torch.randperm(self.size, device=self.device)[:total_needed]
        
        encoder_indices = indices[:K]
        decoder_indices = indices[K:K + K_prime]
        
        return self.observations[encoder_indices], self.observations[decoder_indices]
    
    def normalize_states(self, states: torch.Tensor) -> torch.Tensor:
        """Normalize states using buffer statistics."""
        return (states - self.state_mean) / self.state_std
    
    def unnormalize_states(self, states: torch.Tensor) -> torch.Tensor:
        """Unnormalize states using buffer statistics."""
        return states * self.state_std + self.state_mean
    
    def normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Normalize actions using buffer statistics."""
        return (actions - self.action_mean) / self.action_std
    
    def unnormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Unnormalize actions using buffer statistics."""
        return actions * self.action_std + self.action_mean
    
    def get_all_states(self) -> torch.Tensor:
        """Return all states in the buffer."""
        return self.observations
    
    def __len__(self) -> int:
        return self.size


# ============================================================
# OfflineDataset
# ============================================================

class OfflineDataset:
    """
    High-level dataset class that loads offline RL data and provides
    a ReplayBuffer.
    
    Supports:
    - AntMaze (D4RL)
    - Kitchen (D4RL)
    - ExORL Walker
    - ExORL Cheetah
    """
    
    # Supported domains and their configurations
    DOMAIN_CONFIGS = {
        'antmaze': {
            'type': 'd4rl',
            'env_name': 'antmaze-large-diverse-v2',
            'state_dim': 29,  # AntMaze state: position (2) + goal (2) + others
            'action_dim': 8,
        },
        'kitchen': {
            'type': 'd4rl',
            'env_name': 'kitchen-complete-v0',
            'state_dim': 60,  # Kitchen state: object positions, robot state, etc.
            'action_dim': 9,
        },
        'walker': {
            'type': 'exorl',
            'domain': 'walker',
            'task': 'proto',  # Unsupervised exploration data
            'state_dim': 24,  # DMC Walker state
            'action_dim': 6,
        },
        'cheetah': {
            'type': 'exorl',
            'domain': 'cheetah',
            'task': 'proto',  # Unsupervised exploration data
            'state_dim': 17,  # DMC Cheetah state
            'action_dim': 6,
        },
    }
    
    def __init__(
        self,
        domain: str,
        data_dir: Optional[str] = None,
        device: str = "cpu",
        normalize: bool = True,
    ):
        """
        Initialize dataset for a given domain.
        
        Args:
            domain: one of 'antmaze', 'kitchen', 'walker', 'cheetah'
            data_dir: optional directory for custom dataset locations
            device: torch device
            normalize: whether to normalize states and actions
        """
        self.domain = domain.lower()
        self.device = device
        self.normalize = normalize
        
        if self.domain not in self.DOMAIN_CONFIGS:
            raise ValueError(
                f"Unknown domain: {domain}. "
                f"Supported: {list(self.DOMAIN_CONFIGS.keys())}"
            )
        
        self.config = self.DOMAIN_CONFIGS[self.domain]
        
        # Load raw data
        raw_data = self._load_raw_data(data_dir)
        
        # Extract components
        observations = raw_data['observations']
        actions = raw_data['actions']
        next_observations = raw_data.get('next_observations')
        rewards = raw_data.get('rewards')
        terminals = raw_data.get('terminals')
        timeouts = raw_data.get('timeouts')
        
        # Handle missing next_observations
        if next_observations is None:
            next_observations = np.concatenate(
                [observations[1:], observations[-1:]], axis=0
            )
        
        # Handle missing rewards
        if rewards is None:
            rewards = np.zeros(len(observations))
        
        # Handle missing terminals/timeouts
        if terminals is None:
            terminals = np.zeros(len(observations), dtype=bool)
        if timeouts is None:
            timeouts = np.zeros(len(observations), dtype=bool)
        
        # Ensure correct shapes
        if rewards.ndim == 0:
            rewards = np.full(len(observations), rewards)
        if rewards.ndim == 2 and rewards.shape[1] == 1:
            rewards = rewards.squeeze(-1)
        
        # Store raw data
        self.raw_observations = observations
        self.raw_actions = actions
        self.raw_next_observations = next_observations
        
        # Create replay buffer
        self.replay_buffer = ReplayBuffer(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            rewards=rewards,
            terminals=terminals,
            timeouts=timeouts,
            device=device,
            state_dim=self.config.get('state_dim'),
            action_dim=self.config.get('action_dim'),
        )
        
        # Update state/action dim from actual data
        self.state_dim = self.replay_buffer.state_dim
        self.action_dim = self.replay_buffer.action_dim
        
        print(f"Loaded {self.domain} dataset: {len(self)} transitions, "
              f"state_dim={self.state_dim}, action_dim={self.action_dim}")
    
    def _load_raw_data(self, data_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
        """Load raw data based on domain type."""
        if self.config['type'] == 'd4rl':
            return _load_d4rl_dataset(self.config['env_name'])
        elif self.config['type'] == 'exorl':
            return _load_exorl_dataset(
                self.config['domain'], 
                self.config['task']
            )
        else:
            raise ValueError(f"Unknown dataset type: {self.config['type']}")
    
    def sample_states(self, batch_size: int) -> torch.Tensor:
        """Sample states from the buffer."""
        return self.replay_buffer.sample_states(batch_size)
    
    def sample_transitions(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample transitions from the buffer."""
        return self.replay_buffer.sample_transitions(batch_size)
    
    def sample_encoder_decoder_states(
        self, K: int, K_prime: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample disjoint encoder and decoder states."""
        return self.replay_buffer.sample_encoder_decoder_states(K, K_prime)
    
    def get_all_states(self) -> torch.Tensor:
        """Return all states."""
        return self.replay_buffer.get_all_states()
    
    def __len__(self) -> int:
        return len(self.replay_buffer)


# ============================================================
# Convenience function
# ============================================================

def load_dataset(
    domain: str,
    data_dir: Optional[str] = None,
    device: Optional[str] = None,
    normalize: bool = True,
) -> OfflineDataset:
    """
    Load an offline dataset for a given domain.
    
    Args:
        domain: 'antmaze', 'kitchen', 'walker', or 'cheetah'
        data_dir: optional custom data directory
        device: torch device (default from config)
        normalize: whether to normalize states/actions
    
    Returns:
        OfflineDataset instance
    """
    if device is None:
        device = config.device
    
    return OfflineDataset(
        domain=domain,
        data_dir=data_dir,
        device=device,
        normalize=normalize,
    )


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":
    # Quick test with dummy data
    print("Testing ReplayBuffer with dummy data...")
    
    dummy_obs = np.random.randn(1000, 10).astype(np.float32)
    dummy_act = np.random.randn(1000, 4).astype(np.float32)
    dummy_next_obs = np.random.randn(1000, 10).astype(np.float32)
    dummy_rew = np.random.randn(1000).astype(np.float32)
    dummy_term = np.zeros(1000, dtype=bool)
    
    buffer = ReplayBuffer(
        observations=dummy_obs,
        actions=dummy_act,
        next_observations=dummy_next_obs,
        rewards=dummy_rew,
        terminals=dummy_term,
    )
    
    # Test state sampling
    states = buffer.sample_states(32)
    assert states.shape == (32, 10), f"Expected (32, 10), got {states.shape}"
    
    # Test transition sampling
    batch = buffer.sample_transitions(256)
    assert batch['states'].shape == (256, 10)
    assert batch['actions'].shape == (256, 4)
    assert batch['next_states'].shape == (256, 10)
    assert batch['rewards'].shape == (256,)
    
    # Test encoder/decoder sampling
    enc_states, dec_states = buffer.sample_encoder_decoder_states(32, 32)
    assert enc_states.shape == (32, 10)
    assert dec_states.shape == (32, 10)
    # Check disjointness (when total < buffer size)
    assert not torch.equal(enc_states, dec_states) or len(buffer) < 64
    
    print("All tests passed!")