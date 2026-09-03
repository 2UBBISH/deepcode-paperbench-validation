"""Goal-Conditioned Implicit Q-Learning (GC-IQL) baseline.

This baseline trains IQL-style value, Q, and policy networks that are
conditioned on goal vectors instead of FRE latent codes. It uses hindsight
relabeling to learn goal-reaching behavior from offline trajectories:
each transition is assigned either a goal sampled from the offline state
pool or a future state from the batch, and the reward is a sparse
indicator of whether that transition reaches the assigned goal.

The network sizes follow the FRE IQL specification (256x256 MLPs), and
the evaluation protocol is identical to the other goal-conditioned
baselines: a task goal is given to ``get_task_policy``, which returns an
observation-to-action closure for rollouts.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fre.utils import to_torch
from .baseline_utils import (
    GaussianPolicy,
    TwinQNetwork,
    build_mlp,
    expectile_loss,
    hard_update,
    make_policy_fn_from_net,
    soft_update,
)

__all__ = ["GCIQL"]


class GCIQL:
    """Goal-conditioned IQL agent.

    Parameters
    ----------
    state_dim: int
        Raw state dimensionality.
    action_dim: int
        Action dimensionality.
    goal_dim: int, optional
        Dimensionality of the goal conditioning vector. Defaults to
        ``state_dim``, which is convenient when goals are full states.
    hidden_dims: Sequence[int]
        Hidden widths for the value, Q, and policy MLPs.
    gamma: float
        Discount factor.
    expectile: float
        Expectile for implicit value regression.
    awr_temperature: float
        Advantage-weighted regression temperature.
    target_tau: float
        Polyak averaging coefficient for target Q networks.
    lr: float
        Adam learning rate.
    batch_size: int
        Batch size used by ``train_step`` when inputs are not pre-batched.
    hindsight_prob: float
        Probability of relabeling a transition with a future state from
        the batch. The remaining transitions use a goal from the offline
        state pool (or the provided ``goals`` argument).
    goal_epsilon: float
        Euclidean distance threshold for sparse goal-reaching reward.
    device: Union[str, torch.device]
        Torch device.
    max_action: float
        Absolute action bound, used by the policy squashing layers.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        goal_dim: Optional[int] = None,
        hidden_dims: Sequence[int] = (256, 256),
        gamma: float = 0.99,
        expectile: float = 0.9,
        awr_temperature: float = 3.0,
        target_tau: float = 0.005,
        lr: float = 3e-4,
        batch_size: int = 256,
        hindsight_prob: float = 0.5,
        goal_epsilon: float = 1.0,
        device: Union[str, torch.device] = "cpu",
        max_action: float = 1.0,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim if goal_dim is not None else state_dim
        self.gamma = gamma
        self.expectile = expectile
        self.awr_temperature = awr_temperature
        self.target_tau = target_tau
        self.batch_size = batch_size
        self.hindsight_prob = hindsight_prob
        self.goal_epsilon = goal_epsilon
        self.device = torch.device(device)
        self.max_action = max_action

        # Q(s, a, g) critics.
        self.q1 = TwinQNetwork(
            state_dim, action_dim, self.goal_dim, hidden_dims=hidden_dims
        )
        self.q2 = TwinQNetwork(
            state_dim, action_dim, self.goal_dim, hidden_dims=hidden_dims
        )
        # Value network V(s, g).
        self.v = build_mlp(
            state_dim + self.goal_dim, hidden_dims, output_dim=1
        )
        # Gaussian policy conditioned on (s, g).
        self.policy = GaussianPolicy(
            state_dim,
            self.goal_dim,
            action_dim,
            hidden_dims=hidden_dims,
            max_action=max_action,
        )

        # Target Q networks.
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        hard_update(self.q1_target, self.q1)
        hard_update(self.q2_target, self.q2)

        self.to(self.device)

        params = (
            list(self.q1.parameters())
            + list(self.q2.parameters())
            + list(self.v.parameters())
            + list(self.policy.parameters())
        )
        self.optimizer = torch.optim.Adam(params, lr=lr)

    def to(self, device: Union[str, torch.device]) -> "GCIQL":
        self.device = torch.device(device)
        self.q1.to(self.device)
        self.q2.to(self.device)
        self.v.to(self.device)
        self.policy.to(self.device)
        self.q1_target.to(self.device)
        self.q2_target.to(self.device)
        return self

    def _t(self, x: Any) -> torch.Tensor:
        return to_torch(x, device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def _relabel_goals(
        self,
        next_states: torch.Tensor,
        goals: torch.Tensor,
        goal_pool: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return goals after hindsight relabeling.

        With probability ``hindsight_prob`` a transition uses a future
        state from the current batch. Otherwise it keeps ``goals``. If no
        explicit goals are provided, the goal is sampled from the batch's
        next states (or from ``goal_pool`` when available).
        """
        batch_size = goals.shape[0]
        if goals is None:
            goals = next_states.clone()

        hindsight_mask = (
            torch.rand(batch_size, device=self.device) < self.hindsight_prob
        )
        if hindsight_mask.any():
            # Use random future states from this batch.
            idx = torch.randint(batch_size, (int(hindsight_mask.sum().item()),), device=self.device)
            future_goals = next_states[idx]
            goals[hindsight_mask] = future_goals

        return goals

    def _compute_goal_rewards(
        self, states: torch.Tensor, goals: torch.Tensor
    ) -> torch.Tensor:
        dist = torch.norm(states - goals, dim=-1)
        reward = torch.where(
            dist < self.goal_epsilon,
            torch.zeros_like(dist),
            -torch.ones_like(dist),
        )
        return reward.unsqueeze(-1)

    def train_step(
        self,
        states: Union[np.ndarray, torch.Tensor],
        actions: Union[np.ndarray, torch.Tensor],
        next_states: Union[np.ndarray, torch.Tensor],
        dones: Optional[Union[np.ndarray, torch.Tensor]] = None,
        goals: Optional[Union[np.ndarray, torch.Tensor]] = None,
        goal_pool: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """Perform one IQL training update.

        ``goals`` can be provided explicitly; otherwise goals are taken
        from the offline state pool (``goal_pool``) if supplied, and
        finally from ``next_states``. Hindsight relabeling is applied on
        top of the selected goals.
        """
        states = self._t(states)
        actions = self._t(actions)
        next_states = self._t(next_states)

        if goals is None:
            if goal_pool is not None:
                goal_pool = self._t(goal_pool)
                idx = torch.randint(
                    goal_pool.shape[0], (states.shape[0],), device=self.device
                )
                goals = goal_pool[idx]
            else:
                goals = next_states.clone()
        else:
            goals = self._t(goals)

        goals = self._relabel_goals(next_states, goals, goal_pool)

        if dones is not None:
            dones = self._t(dones).squeeze(-1)
        else:
            dones = torch.zeros(states.shape[0], device=self.device)

        rewards = self._compute_goal_rewards(states, goals)

        q1, q2 = self.q1(states, actions, goals)
        q1_t, q2_t = self.q1_target(states, actions, goals)
        with torch.no_grad():
            v_next = self.v(next_states, goals)
            q_min_target = torch.min(q1_t, q2_t)
            q_backup = rewards + self.gamma * (1.0 - dones.unsqueeze(-1)) * v_next

        q1_loss = F.mse_loss(q1, q_backup.detach())
        q2_loss = F.mse_loss(q2, q_backup.detach())

        v_pred = self.v(states, goals)
        with torch.no_grad():
            q1_d, q2_d = self.q1(states, actions, goals)
            v_target = torch.min(q1_d, q2_d)
        v_loss = expectile_loss(v_target - v_pred, self.expectile).mean()

        action, _, _, log_prob = self.policy(states, goals)
        with torch.no_grad():
            q1_p, q2_p = self.q1(states, actions, goals)
            q_policy = torch.min(q1_p, q2_p)
            v_policy = self.v(states, goals)
            advantage = q_policy - v_policy
            advantage = torch.clamp(
                advantage,
                min=-5.0,
                max=2.0,
            )
            weight = torch.exp(advantage / self.awr_temperature)

        policy_loss = -(weight.detach() * log_prob).mean()

        total_loss = q1_loss + q2_loss + v_loss + policy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        soft_update(self.q1_target, self.q1, self.target_tau)
        soft_update(self.q2_target, self.q2, self.target_tau)

        return {
            "q1_loss": float(q1_loss.detach().cpu()),
            "q2_loss": float(q2_loss.detach().cpu()),
            "v_loss": float(v_loss.detach().cpu()),
            "policy_loss": float(policy_loss.detach().cpu()),
            "total_loss": float(total_loss.detach().cpu()),
        }

    def get_task_policy(
        self,
        goal: Union[np.ndarray, torch.Tensor],
        deterministic: bool = True,
    ):
        """Return a callable ``obs -> action`` conditioned on ``goal``."""
        goal_t = self._t(goal)
        if goal_t.dim() == 1:
            goal_t = goal_t.unsqueeze(0)

        def policy_fn(obs: np.ndarray) -> np.ndarray:
            state = self._t(obs)
            if state.dim() == 1:
                state = state.unsqueeze(0)
            return self.policy.get_action(
                state, goal_t, deterministic=deterministic, device=self.device
            )

        return policy_fn

    def state_dict(self) -> Dict[str, Any]:
        return {
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "v": self.v.state_dict(),
            "policy": self.policy.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.q1.load_state_dict(state_dict["q1"])
        self.q2.load_state_dict(state_dict["q2"])
        self.v.load_state_dict(state_dict["v"])
        self.policy.load_state_dict(state_dict["policy"])
        self.q1_target.load_state_dict(state_dict["q1_target"])
        self.q2_target.load_state_dict(state_dict["q2_target"])
        if "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and "q1" in checkpoint:
            self.load_state_dict(checkpoint)
        else:
            # Support plain model state dicts keyed by network.
            self.load_state_dict(checkpoint)
