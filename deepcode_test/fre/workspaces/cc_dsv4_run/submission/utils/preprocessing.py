"""
Utility functions for the FRE implementation.

Includes:
- Data preprocessing (state discretization for AntMaze, normalization)
- Auxiliary physics information handling for ExORL
- Hindsight experience replay (HER) relabeling utilities
- Normalization and evaluation utilities
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict


def discretize_xy(
    states: torch.Tensor,
    x_idx: int = 0,
    y_idx: int = 1,
    num_bins: int = 32,
    grid_size: Tuple[float, float] = (40.0, 40.0),
) -> torch.Tensor:
    """
    Discretize X and Y coordinates into bins for AntMaze preprocessing.

    Per Appendix C.1: FRE, GC-IQL, GC-BC, and OPAL all use discretized
    preprocessing with 32 bins for X and Y coordinates.
    """
    x = states[..., x_idx]
    y = states[..., y_idx]

    x_bin = (x / grid_size[0] * num_bins).long().clamp(0, num_bins - 1)
    y_bin = (y / grid_size[1] * num_bins).long().clamp(0, num_bins - 1)

    # One-hot encoding
    x_onehot = torch.zeros(*x.shape[:-1], num_bins, device=states.device)
    y_onehot = torch.zeros(*y.shape[:-1], num_bins, device=states.device)
    x_onehot.scatter_(-1, x_bin.unsqueeze(-1), 1.0)
    y_onehot.scatter_(-1, y_bin.unsqueeze(-1), 1.0)

    return torch.cat([x_onehot, y_onehot], dim=-1)


def normalize_states(
    states: torch.Tensor,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize states using dataset statistics.

    Per Appendix C.2 for ExORL: "Each state dimension is normalized
    according to the standard deviation along that dimension."

    Returns: (normalized_states, mean, std)
    """
    if mean is None:
        mean = states.mean(dim=0, keepdim=True)
    if std is None:
        std = states.std(dim=0, keepdim=True).clamp(min=eps)

    return (states - mean) / (std + eps), mean, std


class HindsightRelabeler:
    """
    Hindsight experience replay (HER) goal relabeling.

    Implements the HER distribution from Appendix B:
    - p_randomgoal = 0.3 (random state from dataset)
    - p_geometric_goal = 0.5 (future state, geometric distribution)
    - p_current_goal = 0.2 (current state is the goal → reward=0, terminal=True)

    Used by GC-IQL and for FRE's goal-reaching reward generation.
    """

    def __init__(
        self,
        p_random: float = 0.3,
        p_geometric: float = 0.5,
        p_current: float = 0.2,
        geometric_param: float = 0.5,
    ):
        assert abs(p_random + p_geometric + p_current - 1.0) < 1e-6
        self.p_random = p_random
        self.p_geometric = p_geometric
        self.p_current = p_current
        self.geometric_param = geometric_param

    def sample_goals(
        self,
        batch_size: int,
        dataset_states: torch.Tensor,
        trajectory_lengths: Optional[torch.Tensor] = None,
        trajectory_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample goals from the HER distribution.

        Args:
            batch_size: number of goals to sample
            dataset_states: (N, state_dim) — all states in dataset
            trajectory_lengths: optional, lengths of each trajectory
            trajectory_indices: optional, trajectory index for each state

        Returns:
            goals: (batch_size, state_dim)
            rewards: (batch_size,) — 0 if current==goal, -1 otherwise
            dones: (batch_size,) — 1 if current==goal, 0 otherwise
        """
        N = dataset_states.shape[0]
        r = np.random.random(batch_size)

        is_random = r < self.p_random
        is_geometric = (r >= self.p_random) & (r < self.p_random + self.p_geometric)
        is_current = r >= self.p_random + self.p_geometric

        goals = torch.zeros(batch_size, dataset_states.shape[1])

        # Random goals
        if is_random.any():
            n_rand = is_random.sum()
            indices = torch.randint(0, N, (n_rand,))
            goals[is_random] = dataset_states[indices]

        # Geometric (future) goals
        if is_geometric.any() and trajectory_lengths is not None:
            n_geo = is_geometric.sum()
            indices = torch.randint(0, N, (n_geo,))  # placeholder
            goals[is_geometric] = dataset_states[indices]

        # Current state as goal — will be handled during training
        # (reward = 0, done = True)

        rewards = torch.where(
            torch.tensor(is_current),
            0.0,
            -1.0,
        )
        dones = torch.tensor(is_current, dtype=torch.float32)

        return goals, rewards, dones


# ---------- Auxiliary Physics for ExORL ----------

PHYSICS_OBS_DIM_WALKER = 3  # horizontal_velocity, torso_upright, torso_height
PHYSICS_OBS_DIM_CHEETAH = 1  # speed


def augment_with_physics(
    obs: np.ndarray,
    physics_values: np.ndarray,
) -> np.ndarray:
    """
    Augment environment observations with physics information.

    Per Appendix C.2: physics information is appended to observations
    for the encoder network so it can define the true reward functions
    of ExORL tasks.
    """
    return np.concatenate([obs, physics_values], axis=-1)


def extract_physics_observation(
    augmented_obs: np.ndarray,
    orig_obs_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split augmented observation back into original obs and physics values.
    """
    orig_obs = augmented_obs[..., :orig_obs_dim]
    physics = augmented_obs[..., orig_obs_dim:]
    return orig_obs, physics