"""Goal-Conditioned IQL (GC-IQL).

From Section 5.2 / the addendum: "GC-IQL is just IQL with the additional goal
state."  Concretely:

  * goals and observations are concatenated before being fed to the networks;
  * on every update, a goal is sampled for the current state using the
    hindsight relabelling distribution

        p_random_goal   = 0.3   (a random state from the dataset)
        p_geometric_goal= 0.5   (a future state in the same trajectory, sampled
                                 with a geometric distribution)
        p_current_goal  = 0.2   (the current state itself)

    with reward ``0`` if the goal is the current state and ``-1`` otherwise,
    and a mask / terminal flag set to ``True`` whenever the goal is reached;
  * no environment rewards are used for training;
  * at evaluation time the agent is conditioned on the *ground-truth* goal of
    the downstream task.

IQL hyperparameters follow the same defaults as the FRE agent (Appendix A):
discount 0.88, expectile 0.8, AWR temperature 3.0, target update rate 0.001.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from fre.datasets import OfflineDataset

from baselines.iql_core import IQLCore, IQLCoreConfig


@dataclass
class GCIQLConfig:
    """Configuration for GC-IQL."""

    obs_dim: int
    action_dim: int
    hidden_dims: Tuple[int, ...] = (512, 512, 512)
    discount: float = 0.88
    expectile: float = 0.8
    awr_temperature: float = 3.0
    target_update_rate: float = 0.001
    learning_rate: float = 1e-4
    max_advantage_weight: float = 100.0
    # relabelling mixture
    p_random_goal: float = 0.3
    p_geometric_goal: float = 0.5
    p_current_goal: float = 0.2
    # geometric distribution parameter for future-state sampling
    geometric_p: float = 0.2
    # goal is considered reached when ||s - g|| < threshold
    goal_reached_threshold: float = 0.5

    @property
    def input_dim(self) -> int:
        return self.obs_dim * 2


class GoalRelabeler:
    """Samples ``(goal, reward, mask)`` triples with the GC-IQL distribution."""

    def __init__(self, dataset: OfflineDataset, config: GCIQLConfig, seed: int = 0) -> None:
        self.dataset = dataset
        self.config = config
        self.rng = np.random.default_rng(seed)

    def _states(self, idx: np.ndarray) -> np.ndarray:
        """GC-IQL operates on the environment's underlying observation space.

        Appendix C.2 notes that the auxiliary physics information appended for
        the FRE encoder is "necessary only for the encoder network", so the
        goal-conditioned baselines use the plain observation space.
        """
        return self.dataset.observations[idx]

    def sample(
        self, indices: np.ndarray, next_indices: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Relabel a batch of transitions.

        Args:
            indices: dataset indices of the current states ``s``.
            next_indices: dataset indices of the successor states ``s'`` (used
                to detect goal achievement).

        Returns:
            ``(goals, rewards, masks)`` where ``masks`` is 1.0 where the goal
            has been reached (the terminal flag used by IQL).
        """
        n = len(indices)
        cfg = self.config
        ds = self.dataset
        next_indices = indices if next_indices is None else next_indices

        choice = self.rng.random(n)
        goals = np.empty((n, cfg.obs_dim), dtype=np.float32)

        current_mask = choice < cfg.p_current_goal
        geom_mask = (choice >= cfg.p_current_goal) & (
            choice < cfg.p_current_goal + cfg.p_geometric_goal
        )
        random_mask = choice >= cfg.p_current_goal + cfg.p_geometric_goal

        goals[current_mask] = self._states(indices[current_mask])
        if random_mask.any():
            goals[random_mask] = self._states(
                self.rng.integers(0, len(ds), size=int(random_mask.sum()))
            )
        if geom_mask.any():
            rows = indices[geom_mask]
            tids = ds.traj_ids[rows]
            offsets = ds.traj_offsets[tids]
            lengths = ds.traj_lengths[tids]
            pos = rows - offsets
            # Geometric distribution over the remaining steps of the trajectory.
            remaining = np.maximum(lengths - pos - 1, 0)
            draws = self.rng.geometric(cfg.geometric_p, size=remaining.shape) - 1
            sampled = pos + 1 + np.minimum(draws, np.maximum(remaining - 1, 0))
            sampled = np.where(remaining > 0, sampled, pos)
            goals[geom_mask] = self._states(offsets + sampled)

        next_states = self._states(next_indices)
        dist = np.linalg.norm(next_states - goals, axis=-1)
        reached = dist < cfg.goal_reached_threshold
        rewards = np.where(reached, 0.0, -1.0).astype(np.float32)
        # Current-state goals are terminal by construction.
        reached = np.logical_or(reached, current_mask)
        masks = reached.astype(np.float32)
        return goals, rewards, masks


class GCIQLAgent:
    """IQL whose networks take ``concat(obs, goal)`` as input."""

    def __init__(self, config: GCIQLConfig, device: torch.device = torch.device("cpu")) -> None:
        self.config = config
        self.device = device
        self.core = IQLCore(
            IQLCoreConfig(
                input_dim=config.input_dim,
                action_dim=config.action_dim,
                hidden_dims=config.hidden_dims,
                discount=config.discount,
                expectile=config.expectile,
                awr_temperature=config.awr_temperature,
                target_update_rate=config.target_update_rate,
                learning_rate=config.learning_rate,
                max_advantage_weight=config.max_advantage_weight,
            ),
            device=device,
        )

    # -- updates -------------------------------------------------------------------
    def update(self, batch: Dict[str, np.ndarray], goals: np.ndarray, rewards: np.ndarray,
               masks: np.ndarray) -> Dict[str, float]:
        obs = torch.as_tensor(batch["observations"], device=self.device)
        next_obs = torch.as_tensor(batch["next_observations"], device=self.device)
        action = torch.as_tensor(batch["actions"], device=self.device)
        goal = torch.as_tensor(goals, device=self.device)
        reward = torch.as_tensor(rewards, device=self.device)
        mask = torch.as_tensor(masks, device=self.device)
        return self.core.update(obs, action, reward, next_obs, mask, goal)

    # -- inference -----------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs: np.ndarray, goal: np.ndarray, deterministic: bool = True) -> np.ndarray:
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        goal_t = torch.as_tensor(np.asarray(goal, dtype=np.float32), device=self.device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        if goal_t.dim() == 1:
            goal_t = goal_t.unsqueeze(0)
        if goal_t.shape[0] == 1 and obs_t.shape[0] > 1:
            goal_t = goal_t.expand(obs_t.shape[0], -1)
        return self.core.act(obs_t, goal_t, deterministic).cpu().numpy()

    def state_dict(self):
        return self.core.state_dict()

    def load_state_dict(self, state) -> None:
        self.core.load_state_dict(state)
