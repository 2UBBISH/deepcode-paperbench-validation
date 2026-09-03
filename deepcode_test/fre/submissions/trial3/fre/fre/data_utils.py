"""
Data utilities for Functional Reward Encodings (FRE).

Handles:
- Loading D4RL datasets (AntMaze, Kitchen)
- Loading ExORL datasets (Walker, Cheetah) from the ExORL benchmark
- Replay buffer for offline RL training
- Sampling functions for encoder/decoder states and RL batches
"""

import numpy as np
import torch
from typing import Tuple, Dict, Optional, List, Any
from collections import namedtuple
import os
import pickle
import gym

# Try importing d4rl; handle gracefully if not installed
try:
    import d4rl
    HAS_D4RL = True
except ImportError:
    HAS_D4RL = False
    print("Warning: d4rl not installed. D4RL datasets (AntMaze, Kitchen) will not be available.")

# Try importing dm_control for ExORL
try:
    import dm_control
    HAS_DM_CONTROL = True
except ImportError:
    HAS_DM_CONTROL = False


# ============================================================
# Replay Buffer
# ============================================================

class ReplayBuffer:
    """
    Simple replay buffer for offline RL that stores the entire dataset
    and supports random sampling of states and batches.
    """
    
    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        terminals: np.ndarray,
        timeouts: Optional[np.ndarray] = None,
        capacity: Optional[int] = None,
    ):
        """
        Args:
            observations: (N, obs_dim) array of states
            actions: (N, act_dim) array of actions
            rewards: (N,) array of rewards
            next_observations: (N, obs_dim) array of next states
            terminals: (N,) array of done flags (True for terminal)
            timeouts: (N,) array of timeout flags (optional)
            capacity: Maximum capacity (defaults to dataset size)
        """
        self.observations = observations.astype(np.float32)
        self.actions = actions.astype(np.float32)
        self.rewards = rewards.astype(np.float32).reshape(-1)
        self.next_observations = next_observations.astype(np.float32)
        self.terminals = terminals.astype(np.float32).reshape(-1)
        
        if timeouts is not None:
            self.timeouts = timeouts.astype(np.float32).reshape(-1)
        else:
            self.timeouts = np.zeros_like(self.terminals)
        
        self.size = len(self.observations)
        self.capacity = capacity if capacity is not None else self.size
        
        # Compute dataset statistics
        self.obs_mean = self.observations.mean(axis=0)
        self.obs_std = self.observations.std(axis=0) + 1e-6
        self.act_mean = self.actions.mean(axis=0)
        self.act_std = self.actions.std(axis=0) + 1e-6
        
        # State dimension and action dimension
        self.state_dim = self.observations.shape[1]
        self.action_dim = self.actions.shape[1]
    
    def sample_states(self, n: int, replace: bool = True) -> np.ndarray:
        """
        Sample n states uniformly from the dataset.
        
        Args:
            n: Number of states to sample
            replace: Whether to sample with replacement
            
        Returns:
            Array of shape (n, state_dim)
        """
        indices = np.random.choice(self.size, size=n, replace=replace)
        return self.observations[indices].copy()
    
    def sample_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Sample a batch of transitions for RL training.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Dictionary with keys: 'observations', 'actions', 'rewards',
            'next_observations', 'terminals', 'timeouts'
        """
        indices = np.random.choice(self.size, size=batch_size, replace=True)
        return {
            'observations': self.observations[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_observations': self.next_observations[indices],
            'terminals': self.terminals[indices],
            'timeouts': self.timeouts[indices],
        }
    
    def sample_batch_torch(
        self, batch_size: int, device: str = 'cpu'
    ) -> Dict[str, torch.Tensor]:
        """
        Sample a batch and convert to PyTorch tensors.
        
        Args:
            batch_size: Number of transitions to sample
            device: Device to place tensors on
            
        Returns:
            Dictionary of torch tensors
        """
        batch = self.sample_batch(batch_size)
        return {
            k: torch.from_numpy(v).float().to(device)
            for k, v in batch.items()
        }
    
    def get_all_states(self) -> np.ndarray:
        """Return all states in the dataset."""
        return self.observations.copy()
    
    def get_all_actions(self) -> np.ndarray:
        """Return all actions in the dataset."""
        return self.actions.copy()
    
    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observations using dataset statistics."""
        return (obs - self.obs_mean) / self.obs_std
    
    def unnormalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Unnormalize observations."""
        return obs * self.obs_std + self.obs_mean
    
    def __len__(self) -> int:
        return self.size


# ============================================================
# D4RL Dataset Loading
# ============================================================

def load_d4rl_dataset(
    env_name: str,
    keep_trajectories: bool = False,
) -> Tuple[ReplayBuffer, gym.Env]:
    """
    Load a D4RL dataset and create a replay buffer.
    
    Args:
        env_name: D4RL environment name (e.g., 'antmaze-umaze-v0')
        keep_trajectories: If True, also return trajectory information
        
    Returns:
        replay_buffer: ReplayBuffer containing the dataset
        env: Gym environment instance
    """
    if not HAS_D4RL:
        raise ImportError(
            "d4rl is required to load D4RL datasets. "
            "Install with: pip install d4rl"
        )
    
    # Create environment
    env = gym.make(env_name)
    
    # Load dataset
    dataset = d4rl.qlearning_dataset(env)
    
    observations = dataset['observations']
    actions = dataset['actions']
    rewards = dataset['rewards']
    next_observations = dataset['next_observations']
    terminals = dataset['terminals']
    
    # Handle timeouts if available
    timeouts = dataset.get('timeouts', None)
    
    replay_buffer = ReplayBuffer(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=terminals,
        timeouts=timeouts,
    )
    
    return replay_buffer, env


def load_antmaze_dataset(
    maze_name: str = 'umaze',
    version: str = 'v0',
) -> Tuple[ReplayBuffer, gym.Env]:
    """
    Load an AntMaze dataset from D4RL.
    
    Args:
        maze_name: One of 'umaze', 'umaze-diverse', 'medium-play',
                   'medium-diverse', 'large-play', 'large-diverse'
        version: D4RL version string (usually 'v0' or 'v2')
        
    Returns:
        replay_buffer, env
    """
    env_name = f'antmaze-{maze_name}-{version}'
    return load_d4rl_dataset(env_name)


def load_kitchen_dataset(
    kitchen_type: str = 'complete',
    version: str = 'v0',
) -> Tuple[ReplayBuffer, gym.Env]:
    """
    Load a Kitchen dataset from D4RL.
    
    Args:
        kitchen_type: One of 'complete', 'partial', 'mixed'
        version: D4RL version string
        
    Returns:
        replay_buffer, env
    """
    env_name = f'kitchen-{kitchen_type}-{version}'
    return load_d4rl_dataset(env_name)


# ============================================================
# ExORL Dataset Loading
# ============================================================

def load_exorl_dataset(
    domain: str = 'walker',
    task: str = 'proto',
    data_dir: Optional[str] = None,
) -> Tuple[ReplayBuffer, Any]:
    """
    Load an ExORL dataset.
    
    The ExORL benchmark (Yarats et al., 2022) provides unsupervised
    exploration datasets for Walker and Cheetah domains.
    
    Args:
        domain: 'walker' or 'cheetah'
        task: 'proto' (prototypical) or specific task name
        data_dir: Path to ExORL data directory. If None, tries default locations.
        
    Returns:
        replay_buffer, env (or env-like object)
    """
    if data_dir is None:
        # Try common locations
        possible_dirs = [
            os.path.expanduser('~/exorl/data'),
            os.path.expanduser('~/exorl'),
            './exorl_data',
            '/tmp/exorl_data',
        ]
        for d in possible_dirs:
            if os.path.exists(d):
                data_dir = d
                break
    
    if data_dir is None or not os.path.exists(data_dir):
        raise FileNotFoundError(
            "ExORL data directory not found. Please download the ExORL datasets "
            "from https://github.com/denisyarats/exorl and set data_dir accordingly."
        )
    
    # Try loading from pickle/npz files
    file_patterns = [
        f'{domain}_{task}.pkl',
        f'{domain}_{task}.npz',
        f'{domain}/{task}.pkl',
        f'{domain}/{task}.npz',
    ]
    
    dataset = None
    for pattern in file_patterns:
        filepath = os.path.join(data_dir, pattern)
        if os.path.exists(filepath):
            dataset = _load_exorl_file(filepath)
            break
    
    if dataset is None:
        # Try loading from directory of trajectories
        traj_dir = os.path.join(data_dir, domain, task)
        if os.path.exists(traj_dir):
            dataset = _load_exorl_trajectories(traj_dir)
    
    if dataset is None:
        raise FileNotFoundError(
            f"Could not find ExORL dataset for {domain}/{task} in {data_dir}. "
            "Expected files: {domain}_{task}.pkl, {domain}_{task}.npz, or "
            "directory of trajectory files."
        )
    
    observations = dataset['observations']
    actions = dataset['actions']
    rewards = dataset.get('rewards', np.zeros(len(observations)))
    next_observations = dataset['next_observations']
    terminals = dataset.get('terminals', np.zeros(len(observations)))
    timeouts = dataset.get('timeouts', None)
    
    replay_buffer = ReplayBuffer(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=terminals,
        timeouts=timeouts,
    )
    
    # Create a simple env wrapper for ExORL
    env = _create_exorl_env(domain)
    
    return replay_buffer, env


def _load_exorl_file(filepath: str) -> Optional[Dict[str, np.ndarray]]:
    """Load ExORL data from a pickle or npz file."""
    if filepath.endswith('.pkl'):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data
    elif filepath.endswith('.npz'):
        data = np.load(filepath, allow_pickle=True)
        return dict(data)
    return None


def _load_exorl_trajectories(traj_dir: str) -> Optional[Dict[str, np.ndarray]]:
    """Load ExORL data from a directory of trajectory files."""
    all_obs = []
    all_acts = []
    all_rewards = []
    all_next_obs = []
    all_terminals = []
    
    files = sorted([f for f in os.listdir(traj_dir) if f.endswith('.npz') or f.endswith('.pkl')])
    
    for f in files:
        filepath = os.path.join(traj_dir, f)
        traj = _load_exorl_file(filepath)
        if traj is not None:
            obs = traj.get('observations', traj.get('obs'))
            acts = traj.get('actions', traj.get('act'))
            rews = traj.get('rewards', traj.get('rew'))
            next_obs = traj.get('next_observations', traj.get('next_obs'))
            terminals = traj.get('terminals', traj.get('dones'))
            
            if obs is not None:
                all_obs.append(obs)
                all_acts.append(acts)
                all_rewards.append(rews if rews is not None else np.zeros(len(obs)))
                all_next_obs.append(next_obs if next_obs is not None else obs[1:])
                all_terminals.append(terminals if terminals is not None else np.zeros(len(obs)))
    
    if len(all_obs) == 0:
        return None
    
    # Concatenate all trajectories
    return {
        'observations': np.concatenate(all_obs, axis=0),
        'actions': np.concatenate(all_acts, axis=0),
        'rewards': np.concatenate(all_rewards, axis=0),
        'next_observations': np.concatenate(all_next_obs, axis=0),
        'terminals': np.concatenate(all_terminals, axis=0),
    }


def _create_exorl_env(domain: str):
    """
    Create a simple environment wrapper for ExORL domains.
    
    This provides basic environment metadata (state_dim, action_dim, etc.)
    without requiring the full dm_control environment.
    """
    # Default dimensions for ExORL domains
    configs = {
        'walker': {
            'state_dim': 24,
            'action_dim': 6,
            'name': 'Walker (ExORL)',
        },
        'cheetah': {
            'state_dim': 17,
            'action_dim': 6,
            'name': 'Cheetah (ExORL)',
        },
    }
    
    if domain not in configs:
        raise ValueError(f"Unknown ExORL domain: {domain}. Expected 'walker' or 'cheetah'.")
    
    cfg = configs[domain]
    
    class ExORLEnvWrapper:
        def __init__(self, config):
            self.observation_space = _BoxSpace(config['state_dim'])
            self.action_space = _BoxSpace(config['action_dim'])
            self.spec = _Spec(config['name'])
            self._config = config
        
        def reset(self):
            return np.zeros(self._config['state_dim'])
        
        def step(self, action):
            return np.zeros(self._config['state_dim']), 0.0, False, {}
    
    return ExORLEnvWrapper(cfg)


class _BoxSpace:
    """Minimal Box space for environment metadata."""
    def __init__(self, dim, low=-1.0, high=1.0):
        self.shape = (dim,)
        self.low = np.full(dim, low)
        self.high = np.full(dim, high)


class _Spec:
    """Minimal spec object."""
    def __init__(self, name):
        self.name = name
        self.id = name


# ============================================================
# Dataset Loading Dispatcher
# ============================================================

def load_dataset(
    domain: str,
    task: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> Tuple[ReplayBuffer, Any]:
    """
    Unified dataset loading interface.
    
    Args:
        domain: One of 'antmaze', 'kitchen', 'walker', 'cheetah'
        task: Domain-specific task identifier
            - antmaze: 'umaze', 'umaze-diverse', 'medium-play', 
                       'medium-diverse', 'large-play', 'large-diverse'
            - kitchen: 'complete', 'partial', 'mixed'
            - walker/cheetah: 'proto' or specific ExORL task
        data_dir: Path to ExORL data (only needed for walker/cheetah)
        
    Returns:
        replay_buffer, env
    """
    if domain == 'antmaze':
        maze_name = task if task is not None else 'umaze'
        return load_antmaze_dataset(maze_name)
    
    elif domain == 'kitchen':
        kitchen_type = task if task is not None else 'complete'
        return load_kitchen_dataset(kitchen_type)
    
    elif domain in ('walker', 'cheetah'):
        exorl_task = task if task is not None else 'proto'
        return load_exorl_dataset(domain, exorl_task, data_dir)
    
    else:
        raise ValueError(
            f"Unknown domain: {domain}. "
            "Expected one of: 'antmaze', 'kitchen', 'walker', 'cheetah'."
        )


# ============================================================
# State Sampling Utilities
# ============================================================

def sample_disjoint_states(
    replay_buffer: ReplayBuffer,
    K_encoder: int = 32,
    K_decoder: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample disjoint sets of states for encoder and decoder.
    
    Args:
        replay_buffer: ReplayBuffer instance
        K_encoder: Number of encoder states
        K_decoder: Number of decoder states
        
    Returns:
        encoder_states: (K_encoder, state_dim)
        decoder_states: (K_decoder, state_dim)
    """
    total_needed = K_encoder + K_decoder
    
    if total_needed > replay_buffer.size:
        # If dataset is smaller than needed, sample with replacement
        encoder_states = replay_buffer.sample_states(K_encoder, replace=True)
        decoder_states = replay_buffer.sample_states(K_decoder, replace=True)
    else:
        # Sample without replacement to ensure disjoint sets
        indices = np.random.choice(
            replay_buffer.size, size=total_needed, replace=False
        )
        encoder_indices = indices[:K_encoder]
        decoder_indices = indices[K_encoder:]
        
        encoder_states = replay_buffer.observations[encoder_indices].copy()
        decoder_states = replay_buffer.observations[decoder_indices].copy()
    
    return encoder_states, decoder_states


def sample_encoder_states(
    replay_buffer: ReplayBuffer,
    K: int = 32,
) -> np.ndarray:
    """
    Sample K states for encoding a reward function.
    
    Args:
        replay_buffer: ReplayBuffer instance
        K: Number of states to sample
        
    Returns:
        states: (K, state_dim)
    """
    return replay_buffer.sample_states(K, replace=True)


# ============================================================
# Dataset Statistics
# ============================================================

def compute_dataset_statistics(
    replay_buffer: ReplayBuffer,
) -> Dict[str, np.ndarray]:
    """
    Compute comprehensive statistics of the dataset.
    
    Args:
        replay_buffer: ReplayBuffer instance
        
    Returns:
        Dictionary with mean, std, min, max for observations, actions, rewards
    """
    stats = {
        'obs_mean': replay_buffer.obs_mean,
        'obs_std': replay_buffer.obs_std,
        'act_mean': replay_buffer.act_mean,
        'act_std': replay_buffer.act_std,
        'reward_mean': replay_buffer.rewards.mean(),
        'reward_std': replay_buffer.rewards.std(),
        'reward_min': replay_buffer.rewards.min(),
        'reward_max': replay_buffer.rewards.max(),
        'num_transitions': replay_buffer.size,
        'state_dim': replay_buffer.state_dim,
        'action_dim': replay_buffer.action_dim,
    }
    return stats


# ============================================================
# D4RL-Specific Utilities
# ============================================================

def get_d4rl_dataset_stats(env_name: str) -> Dict[str, Any]:
    """
    Get reference statistics for a D4RL dataset.
    
    Args:
        env_name: D4RL environment name
        
    Returns:
        Dictionary with reference min/max returns for normalization
    """
    # Reference returns for D4RL datasets (from D4RL paper and common benchmarks)
    ref_stats = {
        'antmaze-umaze-v0': {'ref_min_score': 0.0, 'ref_max_score': 1.0},
        'antmaze-umaze-diverse-v0': {'ref_min_score': 0.0, 'ref_max_score': 1.0},
        'antmaze-medium-play-v0': {'ref_min_score': 0.0, 'ref_max_score': 1.0},
        'antmaze-medium-diverse-v0': {'ref_min_score': 0.0, 'ref_max_score': 1.0},
        'antmaze-large-play-v0': {'ref_min_score': 0.0, 'ref_max_score': 1.0},
        'antmaze-large-diverse-v0': {'ref_min_score': 0.0, 'ref_max_score': 1.0},
        'kitchen-complete-v0': {'ref_min_score': 0.0, 'ref_max_score': 4.0},
        'kitchen-partial-v0': {'ref_min_score': 0.0, 'ref_max_score': 4.0},
        'kitchen-mixed-v0': {'ref_min_score': 0.0, 'ref_max_score': 4.0},
    }
    
    return ref_stats.get(env_name, {'ref_min_score': 0.0, 'ref_max_score': 1.0})


# ============================================================
# ExORL-Specific Utilities
# ============================================================

def get_exorl_dataset_stats(domain: str) -> Dict[str, Any]:
    """
    Get reference statistics for an ExORL dataset.
    
    Args:
        domain: 'walker' or 'cheetah'
        
    Returns:
        Dictionary with reference statistics
    """
    ref_stats = {
        'walker': {
            'ref_min_score': 0.0,
            'ref_max_score': 1000.0,
            'state_dim': 24,
            'action_dim': 6,
        },
        'cheetah': {
            'ref_min_score': 0.0,
            'ref_max_score': 1000.0,
            'state_dim': 17,
            'action_dim': 6,
        },
    }
    return ref_stats.get(domain, {'ref_min_score': 0.0, 'ref_max_score': 1000.0})


# ============================================================
# Normalization Helpers
# ============================================================

def normalize_score(
    raw_score: float,
    ref_min: float,
    ref_max: float,
    clip: bool = True,
) -> float:
    """
    Normalize a score to [0, 100] range as done in the FRE paper.
    
    Args:
        raw_score: Raw (unnormalized) score
        ref_min: Reference minimum score
        ref_max: Reference maximum score
        clip: Whether to clip to [0, 100]
        
    Returns:
        Normalized score in [0, 100]
    """
    if ref_max == ref_min:
        return 0.0
    
    normalized = 100.0 * (raw_score - ref_min) / (ref_max - ref_min)
    
    if clip:
        normalized = np.clip(normalized, 0.0, 100.0)
    
    return normalized


# ============================================================
# Test / Sanity Check
# ============================================================

def test_replay_buffer():
    """Quick sanity check for ReplayBuffer."""
    # Create dummy data
    n = 1000
    obs = np.random.randn(n, 10)
    acts = np.random.randn(n, 3)
    rews = np.random.randn(n)
    next_obs = np.random.randn(n, 10)
    terminals = np.zeros(n)
    terminals[-10:] = 1.0
    
    buffer = ReplayBuffer(obs, acts, rews, next_obs, terminals)
    
    # Test sampling
    states = buffer.sample_states(32)
    assert states.shape == (32, 10), f"Expected (32, 10), got {states.shape}"
    
    batch = buffer.sample_batch(64)
    assert batch['observations'].shape == (64, 10)
    assert batch['actions'].shape == (64, 3)
    assert batch['rewards'].shape == (64,)
    
    # Test torch conversion
    batch_torch = buffer.sample_batch_torch(32, device='cpu')
    assert isinstance(batch_torch['observations'], torch.Tensor)
    
    # Test disjoint sampling
    enc_states, dec_states = sample_disjoint_states(buffer, K_encoder=32, K_decoder=32)
    assert enc_states.shape == (32, 10)
    assert dec_states.shape == (32, 10)
    
    print("ReplayBuffer tests passed!")
    return True


if __name__ == '__main__':
    test_replay_buffer()