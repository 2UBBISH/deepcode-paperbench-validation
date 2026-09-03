"""
Offline dataset loader for D4RL and ExORL benchmarks.

Supports:
- D4RL: AntMaze (antmaze-large-diverse-v2, etc.) and Kitchen (kitchen-complete-v0, etc.)
- ExORL: Walker and Cheetah domains from the ExORL benchmark (Yarats et al., 2022)

Provides state normalization and a unified interface for sampling batches
of (state, action, next_state, terminal) tuples. Rewards are computed
on-the-fly by sampled reward functions during training.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class OfflineDataset:
    """
    Unified offline dataset class that loads from D4RL or ExORL and provides
    normalized state access and batch sampling.

    Attributes:
        states: np.ndarray of shape (N, state_dim) - all states in dataset
        actions: np.ndarray of shape (N, action_dim)
        next_states: np.ndarray of shape (N, state_dim)
        terminals: np.ndarray of shape (N,) - boolean (1.0 = terminal)
        timeouts: np.ndarray of shape (N,) - boolean (1.0 = timeout, not true terminal)
        state_mean: np.ndarray of shape (state_dim,) - mean of states for normalization
        state_std: np.ndarray of shape (state_dim,) - std of states for normalization
        state_dim: int
        action_dim: int
        size: int - number of transitions
    """

    def __init__(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        terminals: np.ndarray,
        timeouts: Optional[np.ndarray] = None,
        normalize_states: bool = True,
        state_mean: Optional[np.ndarray] = None,
        state_std: Optional[np.ndarray] = None,
    ):
        """
        Initialize the offline dataset.

        Args:
            states: (N, state_dim) array of states
            actions: (N, action_dim) array of actions
            next_states: (N, state_dim) array of next states
            terminals: (N,) array of terminal flags (1.0 = terminal)
            timeouts: (N,) array of timeout flags (1.0 = timeout, not true terminal)
            normalize_states: Whether to normalize states to mean 0, std 1
            state_mean: Pre-computed mean for normalization (if None, computed from data)
            state_std: Pre-computed std for normalization (if None, computed from data)
        """
        self.states = np.asarray(states, dtype=np.float32)
        self.actions = np.asarray(actions, dtype=np.float32)
        self.next_states = np.asarray(next_states, dtype=np.float32)
        self.terminals = np.asarray(terminals, dtype=np.float32)

        if timeouts is not None:
            self.timeouts = np.asarray(timeouts, dtype=np.float32)
        else:
            self.timeouts = np.zeros_like(self.terminals)

        self.state_dim = self.states.shape[1]
        self.action_dim = self.actions.shape[1]
        self.size = self.states.shape[0]

        # Compute or set normalization statistics
        if normalize_states:
            if state_mean is not None and state_std is not None:
                self.state_mean = np.asarray(state_mean, dtype=np.float32)
                self.state_std = np.asarray(state_std, dtype=np.float32)
            else:
                self.state_mean = self.states.mean(axis=0, keepdims=False)
                self.state_std = self.states.std(axis=0, keepdims=False) + 1e-6

            # Normalize states in-place
            self.states = (self.states - self.state_mean) / self.state_std
            self.next_states = (self.next_states - self.state_mean) / self.state_std
        else:
            self.state_mean = np.zeros(self.state_dim, dtype=np.float32)
            self.state_std = np.ones(self.state_dim, dtype=np.float32)

        logger.info(f"Loaded dataset: {self.size} transitions, "
                     f"state_dim={self.state_dim}, action_dim={self.action_dim}")

    def sample(self, batch_size: int, rng: Optional[np.random.RandomState] = None) -> Dict[str, np.ndarray]:
        """
        Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample
            rng: Optional random state for reproducibility

        Returns:
            Dict with keys: 'states', 'actions', 'next_states', 'terminals', 'timeouts'
        """
        if rng is None:
            indices = np.random.randint(0, self.size, size=batch_size)
        else:
            indices = rng.randint(0, self.size, size=batch_size)

        return {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'next_states': self.next_states[indices],
            'terminals': self.terminals[indices],
            'timeouts': self.timeouts[indices],
        }

    def sample_states(self, batch_size: int, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """
        Sample only states (used for encoding/decoding in FRE).

        Args:
            batch_size: Number of states to sample
            rng: Optional random state

        Returns:
            np.ndarray of shape (batch_size, state_dim)
        """
        if rng is None:
            indices = np.random.randint(0, self.size, size=batch_size)
        else:
            indices = rng.randint(0, self.size, size=batch_size)
        return self.states[indices]

    def get_all_states(self) -> np.ndarray:
        """Return all states in the dataset (for goal sampling, etc.)."""
        return self.states

    def get_normalization_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (state_mean, state_std) for normalization."""
        return self.state_mean.copy(), self.state_std.copy()

    def denormalize_states(self, states: np.ndarray) -> np.ndarray:
        """Convert normalized states back to original scale."""
        return states * self.state_std + self.state_mean

    def normalize_states_custom(self, states: np.ndarray) -> np.ndarray:
        """Normalize external states using dataset statistics."""
        return (states - self.state_mean) / self.state_std

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return (f"OfflineDataset(size={self.size}, state_dim={self.state_dim}, "
                f"action_dim={self.action_dim})")


def load_d4rl_dataset(
    env_name: str,
    normalize_states: bool = True,
    clip_to_eps: bool = True,
) -> OfflineDataset:
    """
    Load a dataset from D4RL.

    Args:
        env_name: D4RL environment name (e.g., 'antmaze-large-diverse-v2',
                  'kitchen-complete-v0')
        normalize_states: Whether to normalize states
        clip_to_eps: Clip terminal flags to episode boundaries (D4RL convention)

    Returns:
        OfflineDataset instance
    """
    try:
        import d4rl
    except ImportError:
        raise ImportError(
            "D4RL is required for loading D4RL datasets. "
            "Install with: pip install d4rl"
        )

    import gym

    logger.info(f"Loading D4RL dataset: {env_name}")

    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)

    states = dataset['observations']
    actions = dataset['actions']
    next_states = dataset['next_observations']
    rewards = dataset['rewards']
    terminals = dataset['terminals']

    # D4RL uses terminals for both true termination and timeouts.
    # For AntMaze, terminals are all 0 (no early termination).
    # For Kitchen, terminals mark episode end.
    # We separate timeouts from true terminals if possible.
    timeouts = np.zeros_like(terminals)

    if clip_to_eps:
        # Clip terminals to episode boundaries: if terminal is True,
        # the next state is the start of a new episode.
        # We keep terminals as-is for D4RL since they already mark episode boundaries.
        pass

    # Close environment
    env.close()

    return OfflineDataset(
        states=states,
        actions=actions,
        next_states=next_states,
        terminals=terminals,
        timeouts=timeouts,
        normalize_states=normalize_states,
    )


def load_exorl_dataset(
    domain: str,
    data_path: Optional[str] = None,
    normalize_states: bool = True,
) -> OfflineDataset:
    """
    Load a dataset from the ExORL benchmark.

    The ExORL datasets are typically stored as .npz or .hdf5 files containing
    unsupervised exploration data from the FB paper (Touati et al., 2022).

    Args:
        domain: Domain name, one of {'walker', 'cheetah'}
        data_path: Path to the ExORL data directory. If None, will look for
                   environment variable EXORL_DATA_PATH or default './data/exorl'
        normalize_states: Whether to normalize states

    Returns:
        OfflineDataset instance
    """
    import os

    if data_path is None:
        data_path = os.environ.get('EXORL_DATA_PATH', './data/exorl')

    domain_lower = domain.lower()
    if domain_lower not in ('walker', 'cheetah'):
        raise ValueError(f"Unknown ExORL domain: {domain}. Expected 'walker' or 'cheetah'.")

    # Try to find the dataset file
    possible_files = [
        os.path.join(data_path, f'{domain_lower}_dataset.npz'),
        os.path.join(data_path, f'{domain_lower}_dataset.hdf5'),
        os.path.join(data_path, f'{domain_lower}.npz'),
        os.path.join(data_path, f'{domain_lower}.hdf5'),
        os.path.join(data_path, f'{domain_lower}_offline.npz'),
    ]

    dataset_file = None
    for f in possible_files:
        if os.path.exists(f):
            dataset_file = f
            break

    if dataset_file is None:
        # Try loading via the exorl package if available
        try:
            return _load_exorl_via_package(domain_lower, normalize_states)
        except ImportError:
            raise FileNotFoundError(
                f"Could not find ExORL dataset for domain '{domain}'. "
                f"Looked in: {possible_files}. "
                f"Please download the ExORL datasets from "
                f"https://github.com/facebookresearch/controllable_agent "
                f"and set data_path accordingly."
            )

    logger.info(f"Loading ExORL dataset from: {dataset_file}")

    if dataset_file.endswith('.npz'):
        data = np.load(dataset_file, allow_pickle=True)
        # Handle different key naming conventions
        if 'observations' in data:
            states = data['observations']
            actions = data.get('actions', np.zeros((len(states), 1)))
            next_states = data.get('next_observations', data.get('next_obs', states))
            terminals = data.get('terminals', data.get('dones', np.zeros(len(states))))
        elif 'states' in data:
            states = data['states']
            actions = data.get('actions', np.zeros((len(states), 1)))
            next_states = data.get('next_states', states)
            terminals = data.get('terminals', data.get('dones', np.zeros(len(states))))
        else:
            # Try to infer from available keys
            keys = list(data.keys())
            logger.warning(f"Unknown ExORL data format. Available keys: {keys}")
            # Assume first large array is states
            states = None
            for k in keys:
                arr = data[k]
                if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[0] > 1000:
                    states = arr
                    break
            if states is None:
                raise ValueError(f"Cannot parse ExORL dataset from {dataset_file}")
            actions = np.zeros((len(states), 1))
            next_states = states.copy()
            terminals = np.zeros(len(states))

        timeouts = np.zeros_like(terminals)

    elif dataset_file.endswith('.hdf5') or dataset_file.endswith('.h5'):
        import h5py
        with h5py.File(dataset_file, 'r') as f:
            states = f['observations'][:]
            actions = f.get('actions', np.zeros((len(states), 1)))[:]
            next_states = f.get('next_observations', f.get('next_obs', states))[:]
            terminals = f.get('terminals', f.get('dones', np.zeros(len(states))))[:]
            timeouts = np.zeros_like(terminals)
    else:
        raise ValueError(f"Unsupported file format: {dataset_file}")

    return OfflineDataset(
        states=states,
        actions=actions,
        next_states=next_states,
        terminals=terminals,
        timeouts=timeouts,
        normalize_states=normalize_states,
    )


def _load_exorl_via_package(domain: str, normalize_states: bool = True) -> OfflineDataset:
    """
    Attempt to load ExORL dataset using the exorl Python package.

    Args:
        domain: 'walker' or 'cheetah'
        normalize_states: Whether to normalize states

    Returns:
        OfflineDataset instance
    """
    try:
        import exorl
    except ImportError:
        raise ImportError(
            "ExORL package not found. Install from: "
            "https://github.com/facebookresearch/controllable_agent"
        )

    logger.info(f"Loading ExORL dataset via exorl package: {domain}")

    # The exorl package typically provides a dataset loader
    # This is a best-effort wrapper; exact API may vary
    if hasattr(exorl, 'load_dataset'):
        data = exorl.load_dataset(domain)
    elif hasattr(exorl, 'get_dataset'):
        data = exorl.get_dataset(domain)
    else:
        # Try to use the replay buffer from exorl
        from exorl.replay_buffer import ReplayBuffer
        buffer = ReplayBuffer(domain)
        data = {
            'observations': buffer.states,
            'actions': buffer.actions,
            'next_observations': buffer.next_states,
            'terminals': buffer.dones,
        }

    states = data['observations']
    actions = data.get('actions', np.zeros((len(states), 1)))
    next_states = data.get('next_observations', states)
    terminals = data.get('terminals', data.get('dones', np.zeros(len(states))))
    timeouts = np.zeros_like(terminals)

    return OfflineDataset(
        states=states,
        actions=actions,
        next_states=next_states,
        terminals=terminals,
        timeouts=timeouts,
        normalize_states=normalize_states,
    )


def load_dataset(
    dataset_name: str,
    normalize_states: bool = True,
    data_path: Optional[str] = None,
) -> OfflineDataset:
    """
    Unified dataset loading interface.

    Args:
        dataset_name: Name of the dataset. For D4RL, use the standard env name
                      (e.g., 'antmaze-large-diverse-v2', 'kitchen-complete-v0').
                      For ExORL, use 'exorl-walker' or 'exorl-cheetah'.
        normalize_states: Whether to normalize states
        data_path: Path to data directory (for ExORL)

    Returns:
        OfflineDataset instance
    """
    if dataset_name.startswith('exorl-'):
        domain = dataset_name.replace('exorl-', '')
        return load_exorl_dataset(domain, data_path=data_path,
                                  normalize_states=normalize_states)
    else:
        # Assume D4RL dataset
        return load_d4rl_dataset(dataset_name, normalize_states=normalize_states)


# Convenience function for creating a dataset from raw numpy arrays
def create_dataset_from_arrays(
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
    terminals: np.ndarray,
    timeouts: Optional[np.ndarray] = None,
    normalize_states: bool = True,
) -> OfflineDataset:
    """
    Create an OfflineDataset from raw numpy arrays.

    Args:
        states: (N, state_dim)
        actions: (N, action_dim)
        next_states: (N, state_dim)
        terminals: (N,)
        timeouts: (N,) optional
        normalize_states: Whether to normalize

    Returns:
        OfflineDataset instance
    """
    return OfflineDataset(
        states=states,
        actions=actions,
        next_states=next_states,
        terminals=terminals,
        timeouts=timeouts,
        normalize_states=normalize_states,
    )