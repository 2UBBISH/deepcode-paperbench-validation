"""
Offline dataset loading and preprocessing for FRE.

Supports:
- D4RL datasets (AntMaze, Kitchen) via the d4rl library
- ExORL datasets (walker, cheetah) via custom loading
- State normalization (mean 0, std 1) based on dataset statistics
- Sampling methods: random states, random trajectories, state-action pairs
"""

import numpy as np
import torch
from typing import Tuple, Dict, Optional, List, Any
from collections import defaultdict
import os
import pickle


# ============================================================
# Dataset Statistics & Normalization
# ============================================================

class DatasetNormalizer:
    """Compute and apply state normalization (mean 0, std 1)."""

    def __init__(self, states: np.ndarray):
        self.state_mean = np.mean(states, axis=0, keepdims=True)
        self.state_std = np.std(states, axis=0, keepdims=True)
        # Prevent division by zero
        self.state_std = np.where(self.state_std < 1e-6, 1.0, self.state_std)

    def normalize(self, state: np.ndarray) -> np.ndarray:
        return (state - self.state_mean) / self.state_std

    def denormalize(self, state: np.ndarray) -> np.ndarray:
        return state * self.state_std + self.state_mean

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {"mean": self.state_mean, "std": self.state_std}

    @classmethod
    def from_dict(cls, d: Dict[str, np.ndarray]) -> "DatasetNormalizer":
        obj = cls.__new__(cls)
        obj.state_mean = d["mean"]
        obj.state_std = d["std"]
        return obj


# ============================================================
# Offline Dataset Container
# ============================================================

class OfflineDataset:
    """
    Unified container for offline RL datasets.

    Stores:
        - observations (states): (N, state_dim)
        - actions: (N, action_dim)
        - next_observations: (N, state_dim)
        - rewards: (N,)  -- original dataset rewards (may be unused for FRE)
        - terminals: (N,)
        - timeouts: (N,)
        - episode_starts: indices where episodes begin
        - episode_lengths: lengths of each episode
    """

    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray,
        timeouts: Optional[np.ndarray] = None,
        normalizer: Optional[DatasetNormalizer] = None,
    ):
        self.observations = observations.astype(np.float32)
        self.actions = actions.astype(np.float32)
        self.next_observations = next_observations.astype(np.float32)
        self.rewards = rewards.astype(np.float32)
        self.terminals = terminals.astype(np.bool_)
        self.timeouts = (
            timeouts.astype(np.bool_)
            if timeouts is not None
            else np.zeros_like(terminals, dtype=np.bool_)
        )

        self._size = self.observations.shape[0]
        self.state_dim = self.observations.shape[1]
        self.action_dim = self.actions.shape[1]

        # Compute episode boundaries
        self._compute_episodes()

        # Normalizer
        if normalizer is None:
            self.normalizer = DatasetNormalizer(self.observations)
        else:
            self.normalizer = normalizer

        # Normalized observations (lazily cached)
        self._norm_obs: Optional[np.ndarray] = None

    def _compute_episodes(self):
        """Identify episode boundaries from terminals and timeouts."""
        done = self.terminals | self.timeouts
        # Episode starts: index 0 and indices right after a done
        self.episode_starts = [0]
        for i in range(1, self._size):
            if done[i - 1]:
                self.episode_starts.append(i)
        self.episode_starts = np.array(self.episode_starts, dtype=np.int64)

        # Episode lengths
        self.episode_lengths = []
        for i in range(len(self.episode_starts)):
            start = self.episode_starts[i]
            end = (
                self.episode_starts[i + 1]
                if i + 1 < len(self.episode_starts)
                else self._size
            )
            self.episode_lengths.append(end - start)
        self.episode_lengths = np.array(self.episode_lengths, dtype=np.int64)

        self.num_episodes = len(self.episode_starts)

    @property
    def norm_obs(self) -> np.ndarray:
        """Return normalized observations, cached."""
        if self._norm_obs is None:
            self._norm_obs = self.normalizer.normalize(self.observations)
        return self._norm_obs

    def __len__(self) -> int:
        return self._size

    # ----------------------------------------------------------
    # Sampling Methods
    # ----------------------------------------------------------

    def sample_random_states(self, n: int) -> np.ndarray:
        """Sample n random states (raw, unnormalized) uniformly from the dataset."""
        indices = np.random.randint(0, self._size, size=n)
        return self.observations[indices]

    def sample_random_norm_states(self, n: int) -> np.ndarray:
        """Sample n random normalized states."""
        indices = np.random.randint(0, self._size, size=n)
        return self.norm_obs[indices]

    def sample_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Sample a batch of transitions for IQL training.
        Returns normalized states.
        """
        indices = np.random.randint(0, self._size, size=batch_size)
        batch = {
            "observations": self.norm_obs[indices],
            "actions": self.actions[indices],
            "next_observations": self.normalizer.normalize(
                self.next_observations[indices]
            ),
            "rewards": self.rewards[indices],
            "terminals": self.terminals[indices],
            "timeouts": self.timeouts[indices],
        }
        return batch

    def sample_episode(self) -> Dict[str, np.ndarray]:
        """Sample a random full episode."""
        ep_idx = np.random.randint(0, self.num_episodes)
        start = self.episode_starts[ep_idx]
        end = (
            self.episode_starts[ep_idx + 1]
            if ep_idx + 1 < len(self.episode_starts)
            else self._size
        )
        return {
            "observations": self.observations[start:end],
            "norm_observations": self.norm_obs[start:end],
            "actions": self.actions[start:end],
            "next_observations": self.next_observations[start:end],
            "rewards": self.rewards[start:end],
            "terminals": self.terminals[start:end],
            "timeouts": self.timeouts[start:end],
        }

    def get_all_states(self) -> np.ndarray:
        """Return all raw observations."""
        return self.observations

    def get_all_norm_states(self) -> np.ndarray:
        """Return all normalized observations."""
        return self.norm_obs

    def get_state_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (min, max) of raw state dimensions."""
        return self.observations.min(axis=0), self.observations.max(axis=0)


# ============================================================
# D4RL Dataset Loading
# ============================================================

def load_d4rl_dataset(env_name: str) -> OfflineDataset:
    """
    Load a D4RL dataset.

    Args:
        env_name: D4RL environment name, e.g.:
            - 'antmaze-large-diverse-v2'
            - 'antmaze-medium-diverse-v2'
            - 'antmaze-large-play-v2'
            - 'kitchen-complete-v0'
            - 'kitchen-partial-v0'
            - 'kitchen-mixed-v0'

    Returns:
        OfflineDataset instance.
    """
    try:
        import d4rl
        import gym
    except ImportError:
        raise ImportError(
            "D4RL is required. Install with: pip install d4rl (requires MuJoCo)."
        )

    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)

    observations = dataset["observations"]
    actions = dataset["actions"]
    next_observations = dataset["next_observations"]
    rewards = dataset["rewards"]
    terminals = dataset["terminals"]

    # Some D4RL datasets include timeouts, some don't
    timeouts = dataset.get("timeouts", np.zeros_like(terminals, dtype=np.bool_))

    env.close()

    return OfflineDataset(
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        rewards=rewards,
        terminals=terminals,
        timeouts=timeouts,
    )


# ============================================================
# ExORL Dataset Loading
# ============================================================

def load_exorl_dataset(
    domain: str = "walker",
    data_path: Optional[str] = None,
) -> OfflineDataset:
    """
    Load an ExORL dataset (unsupervised exploration data).

    ExORL datasets are typically stored as .npy or .npz files.
    Expected format: dict with keys:
        'observations', 'actions', 'next_observations', 'rewards',
        'terminals', 'timeouts' (optional)

    Args:
        domain: 'walker' or 'cheetah'
        data_path: Path to the ExORL data directory. If None, looks for
                   environment variable EXORL_DATA_PATH or default './data/exorl'.

    Returns:
        OfflineDataset instance.
    """
    if data_path is None:
        data_path = os.environ.get("EXORL_DATA_PATH", "./data/exorl")

    file_path = os.path.join(data_path, f"{domain}_unsupervised.npz")

    if not os.path.exists(file_path):
        # Try alternative naming
        alt_path = os.path.join(data_path, f"{domain}.npz")
        if os.path.exists(alt_path):
            file_path = alt_path
        else:
            raise FileNotFoundError(
                f"ExORL dataset not found at {file_path} or {alt_path}. "
                f"Please download ExORL data from the official repository "
                f"and set EXORL_DATA_PATH environment variable."
            )

    data = np.load(file_path, allow_pickle=True)

    # Handle both .npz (dict-like) and .npy (array) formats
    if hasattr(data, "keys"):
        observations = data["observations"]
        actions = data["actions"]
        next_observations = data.get(
            "next_observations",
            np.roll(observations, -1, axis=0),  # fallback
        )
        rewards = data.get("rewards", np.zeros(len(observations)))
        terminals = data.get("terminals", np.zeros(len(observations), dtype=np.bool_))
        timeouts = data.get("timeouts", np.zeros(len(observations), dtype=np.bool_))
    else:
        # Assume it's a pre-processed array; try to parse
        raise ValueError(
            f"Unrecognized ExORL data format in {file_path}. "
            f"Expected .npz with dict-like structure."
        )

    return OfflineDataset(
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        rewards=rewards,
        terminals=terminals,
        timeouts=timeouts,
    )


# ============================================================
# Synthetic / Toy Dataset (for testing)
# ============================================================

def make_toy_dataset(
    state_dim: int = 4,
    action_dim: int = 2,
    num_transitions: int = 10000,
    num_episodes: int = 100,
    seed: int = 42,
) -> OfflineDataset:
    """
    Create a simple synthetic dataset for testing purposes.

    States are random walks; actions are random.
    """
    rng = np.random.RandomState(seed)

    observations = []
    actions = []
    next_observations = []
    rewards = []
    terminals = []
    timeouts = []

    trans_per_ep = num_transitions // num_episodes

    for ep in range(num_episodes):
        state = rng.randn(state_dim) * 0.1
        for t in range(trans_per_ep):
            action = rng.randn(action_dim) * 0.5
            next_state = state + action * 0.1 + rng.randn(state_dim) * 0.01

            observations.append(state.copy())
            actions.append(action.copy())
            next_observations.append(next_state.copy())
            rewards.append(0.0)  # placeholder
            terminals.append(False)
            timeouts.append(False)

            state = next_state

        # Mark last transition as terminal
        terminals[-1] = True

    return OfflineDataset(
        observations=np.array(observations),
        actions=np.array(actions),
        next_observations=np.array(next_observations),
        rewards=np.array(rewards),
        terminals=np.array(terminals),
        timeouts=np.array(timeouts),
    )


# ============================================================
# Dataset Registry
# ============================================================

DATASET_LOADERS = {
    "antmaze-large-diverse-v2": lambda: load_d4rl_dataset("antmaze-large-diverse-v2"),
    "antmaze-medium-diverse-v2": lambda: load_d4rl_dataset("antmaze-medium-diverse-v2"),
    "antmaze-large-play-v2": lambda: load_d4rl_dataset("antmaze-large-play-v2"),
    "antmaze-medium-play-v2": lambda: load_d4rl_dataset("antmaze-medium-play-v2"),
    "kitchen-complete-v0": lambda: load_d4rl_dataset("kitchen-complete-v0"),
    "kitchen-partial-v0": lambda: load_d4rl_dataset("kitchen-partial-v0"),
    "kitchen-mixed-v0": lambda: load_d4rl_dataset("kitchen-mixed-v0"),
    "exorl-walker": lambda: load_exorl_dataset("walker"),
    "exorl-cheetah": lambda: load_exorl_dataset("cheetah"),
}


def load_dataset(name: str, **kwargs) -> OfflineDataset:
    """
    Load a dataset by name.

    Args:
        name: One of the registered dataset names.
        **kwargs: Additional arguments passed to the loader.

    Returns:
        OfflineDataset instance.
    """
    if name in DATASET_LOADERS:
        return DATASET_LOADERS[name]()
    else:
        raise ValueError(
            f"Unknown dataset: {name}. Available: {list(DATASET_LOADERS.keys())}"
        )


# ============================================================
# Utility: Convert dataset to PyTorch tensors
# ============================================================

def dataset_to_tensors(
    dataset: OfflineDataset,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """Convert the full dataset to a dictionary of PyTorch tensors."""
    return {
        "observations": torch.from_numpy(dataset.norm_obs).float().to(device),
        "actions": torch.from_numpy(dataset.actions).float().to(device),
        "next_observations": torch.from_numpy(
            dataset.normalizer.normalize(dataset.next_observations)
        )
        .float()
        .to(device),
        "rewards": torch.from_numpy(dataset.rewards).float().to(device),
        "terminals": torch.from_numpy(dataset.terminals).float().to(device),
        "timeouts": torch.from_numpy(dataset.timeouts).float().to(device),
    }