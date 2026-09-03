"""
Environment interface wrappers for D4RL and ExORL benchmarks.

Supports:
- AntMaze (antmaze-large-diverse-v2 from D4RL)
- ExORL (Walker and Cheetah domains, RND dataset)
- Kitchen (D4RL)

For reproducibility, uses D4RL commit from before June 2024.
"""

import gym
import numpy as np
import torch
from typing import Tuple, Dict, Optional


def make_antmaze_env():
    """
    Create AntMaze environment (antmaze-large-diverse-v2).

    Per Appendix C.1: agent starts in center of maze for diverse behavior,
    max episode length 2000 steps.
    """
    try:
        import d4rl
    except ImportError:
        raise ImportError(
            "D4RL is required for AntMaze. Install with: pip install d4rl"
        )

    env = gym.make('antmaze-large-diverse-v2')
    env._max_episode_steps = 2000

    return env


def make_kitchen_env():
    """Create Kitchen environment from D4RL."""
    try:
        import d4rl
    except ImportError:
        raise ImportError(
            "D4RL is required for Kitchen. Install with: pip install d4rl"
        )

    env = gym.make('kitchen-complete-v0')
    return env


class OfflineDataset:
    """
    Loads and preprocesses an offline RL dataset.

    Provides:
    - states, actions, next_states, rewards, dones (for offline RL training)
    - all_states (for FRE encoder state sampling)
    - state statistics (for normalization)
    """

    def __init__(
        self,
        env_name: str,
        dataset_path: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Args:
            env_name: D4RL or ExORL environment name
            dataset_path: optional path to custom dataset
            device: torch device
        """
        self.env_name = env_name
        self.device = torch.device(device)

        if dataset_path is not None:
            self._load_custom(dataset_path)
        else:
            self._load_d4rl()

    def _load_d4rl(self):
        """Load dataset from D4RL."""
        import d4rl
        env = gym.make(self.env_name)
        dataset = env.get_dataset()

        self.states = torch.tensor(
            dataset['observations'], dtype=torch.float32, device=self.device
        )
        self.actions = torch.tensor(
            dataset['actions'], dtype=torch.float32, device=self.device
        )
        self.next_states = torch.tensor(
            dataset['next_observations'], dtype=torch.float32, device=self.device
        )
        self.rewards = torch.tensor(
            dataset['rewards'], dtype=torch.float32, device=self.device
        ).unsqueeze(-1)
        self.dones = torch.tensor(
            dataset['terminals'], dtype=torch.float32, device=self.device
        ).unsqueeze(-1)

        # All states (observations + next_observations) for encoder sampling
        self.all_states = torch.cat([self.states, self.next_states], dim=0)

        self.state_dim = self.states.shape[1]
        self.action_dim = self.actions.shape[1]

        # State statistics for normalization
        self.state_mean = self.all_states.mean(dim=0)
        self.state_std = self.all_states.std(dim=0).clamp(min=1e-6)

    def _load_custom(self, path: str):
        """Load custom dataset (e.g., ExORL RND dataset)."""
        import h5py
        with h5py.File(path, 'r') as f:
            self.states = torch.tensor(f['observations'][:], dtype=torch.float32)
            self.actions = torch.tensor(f['actions'][:], dtype=torch.float32)
            self.next_states = torch.tensor(f['next_observations'][:], dtype=torch.float32)
            self.rewards = torch.tensor(f['rewards'][:], dtype=torch.float32).unsqueeze(-1)
            self.dones = torch.tensor(f['terminals'][:], dtype=torch.float32).unsqueeze(-1)

        self.all_states = torch.cat([self.states, self.next_states], dim=0)
        self.state_dim = self.states.shape[1]
        self.action_dim = self.actions.shape[1]
        self.state_mean = self.all_states.mean(dim=0)
        self.state_std = self.all_states.std(dim=0).clamp(min=1e-6)

    def sample_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a random batch of transitions."""
        N = self.states.shape[0]
        indices = torch.randint(0, N, (batch_size,))
        return {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_states': self.next_states[indices],
            'dones': self.dones[indices],
        }

    def sample_encoder_states(self, batch_size: int, K: int = 32) -> torch.Tensor:
        """Sample K states for the FRE encoder."""
        N = self.all_states.shape[0]
        indices = torch.randint(0, N, (batch_size, K))
        return self.all_states[indices]

    def sample_decoder_states(self, batch_size: int, Kp: int = 8) -> torch.Tensor:
        """Sample K' states for the FRE decoder."""
        N = self.all_states.shape[0]
        indices = torch.randint(0, N, (batch_size, Kp))
        return self.all_states[indices]


# ---------- Auxiliary Physics Information for ExORL ----------


def get_walker_physics(physics_state) -> Dict[str, float]:
    """
    Extract auxiliary physics information for Walker domain.

    Per Appendix C.2: horizontal_velocity, torso_upright, torso_height.
    """
    return {
        'horizontal_velocity': physics_state.horizontal_velocity(),
        'torso_upright': physics_state.torso_upright(),
        'torso_height': physics_state.torso_height(),
    }


def get_cheetah_physics(physics_state) -> Dict[str, float]:
    """
    Extract auxiliary physics information for Cheetah domain.

    Per Appendix C.2: speed().
    """
    return {
        'speed': physics_state.speed(),
    }