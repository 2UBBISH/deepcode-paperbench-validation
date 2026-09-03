"""Successor Features (SF) baseline with ICM-pretrained state features.

This module implements a zero-shot offline RL baseline that first learns a
compact state-feature representation with an Intrinsic Curiosity Module
(ICM, Pathak et al. 2017), then learns successor features over those features
using temporal-difference updates.  A DDPG-style deterministic policy is
trained simultaneously by maximising :math:`psi(s,a)^T w` for randomly sampled
task vectors :math:`w`.

At evaluation time a linear reward regressor is fitted on 5120 state-reward
examples (following Touati et al. 2022), producing a task vector :math:`w`
that conditions the same universal policy without additional training.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fre.utils import to_torch

from .baseline_utils import (
    DeterministicPolicy,
    build_mlp,
    hard_update,
    make_policy_fn_from_net,
    ridge_regression,
    sample_reward_pairs,
    soft_update,
)

__all__ = ["ICMFeatureNet", "SF"]


class ICMFeatureNet(nn.Module):
    """ICM feature encoder with inverse and forward dynamics heads.

    The feature encoder maps states to a compact feature vector.  The inverse
    head predicts the action that produced a transition from the pair of
    encoded states, while the forward head predicts the next encoded state from
    the current encoded state and action.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        feature_dim: int = 256,
        hidden_dims: Sequence[int] = (256, 256),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.feature_net = build_mlp(
            state_dim,
            hidden_dims,
            feature_dim,
            activation=activation,
            output_activation=None,
        )
        self.inverse_net = build_mlp(
            2 * feature_dim,
            hidden_dims,
            action_dim,
            activation=activation,
            output_activation="tanh",
        )
        self.forward_net = build_mlp(
            feature_dim + action_dim,
            hidden_dims,
            feature_dim,
            activation=activation,
            output_activation=None,
        )

    def encode(self, states: torch.Tensor) -> torch.Tensor:
        return self.feature_net(states)

    def forward(
        self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phi = self.feature_net(states)
        phi_next = self.feature_net(next_states)

        pred_action = self.inverse_net(torch.cat([phi, phi_next], dim=-1))
        pred_phi_next = self.forward_net(torch.cat([phi, actions], dim=-1))
        return phi, phi_next, pred_action, pred_phi_next


class SF:
    """Successor Features zero-shot offline RL agent.

    Parameters
    ----------
    state_dim:
        Dimension of the environment state observations.
    action_dim:
        Dimension of the action space.
    feature_dim:
        Dimension of the learned ICM state features and successor features.
    hidden_dims:
        Hidden widths for the feature, successor-feature, and policy networks.
    icm_lr:
        Learning rate for the ICM feature/encoder optimisation.
    lr:
        Learning rate for successor-feature and policy networks.
    gamma:
        Discount factor used by the successor-feature TD target.
    tau:
        Polyak averaging coefficient for target networks.
    batch_size:
        Batch size used for random task-vector sampling in ``train_step``.
    eval_reward_samples:
        Number of state-reward samples used for evaluation reward regression.
    device:
        Torch device on which all computations are performed.
    max_action:
        Maximum absolute action value (used by the policy network).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        feature_dim: int = 256,
        hidden_dims: Sequence[int] = (256, 256),
        icm_lr: float = 1e-4,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 256,
        eval_reward_samples: int = 5120,
        device: Union[str, torch.device] = "cpu",
        max_action: float = 1.0,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.hidden_dims = tuple(hidden_dims)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.eval_reward_samples = eval_reward_samples
        self.max_action = max_action
        self.device = torch.device(device)

        self.icm = ICMFeatureNet(
            state_dim=state_dim,
            action_dim=action_dim,
            feature_dim=feature_dim,
            hidden_dims=self.hidden_dims,
            activation="relu",
        )

        self.psi = build_mlp(
            state_dim + action_dim,
            self.hidden_dims,
            feature_dim,
            activation="relu",
            output_activation=None,
        )
        self.psi_target = copy.deepcopy(self.psi)
        hard_update(self.psi_target, self.psi)

        self.policy = DeterministicPolicy(
            state_dim=state_dim,
            context_dim=feature_dim,
            action_dim=action_dim,
            hidden_dims=self.hidden_dims,
            max_action=max_action,
            activation="relu",
        )
        self.policy_target = copy.deepcopy(self.policy)
        hard_update(self.policy_target, self.policy)

        self.icm_optimizer = torch.optim.Adam(
            self.icm.parameters(), lr=icm_lr
        )
        self.sf_optimizer = torch.optim.Adam(self.psi.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.to(self.device)

    def to(self, device: Union[str, torch.device]) -> "SF":
        self.device = torch.device(device)
        self.icm.to(self.device)
        self.psi.to(self.device)
        self.psi_target.to(self.device)
        self.policy.to(self.device)
        self.policy_target.to(self.device)
        return self

    def _sample_w(self, batch_size: int) -> torch.Tensor:
        """Sample random task vectors on the unit sphere."""
        w = torch.randn(batch_size, self.feature_dim, device=self.device)
        return F.normalize(w, dim=-1)

    def _unpack_batch(
        self, batch: Any
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalise common offline-dataset batch layouts to tensors."""
        if isinstance(batch, dict):
            states = batch.get("states", batch.get("observations"))
            actions = batch.get("actions")
            next_states = batch.get("next_states", batch.get("next_observations"))
            dones = batch.get(
                "dones", batch.get("terminals", batch.get("timeouts"))
            )
        else:
            states, actions, next_states = batch[0], batch[1], batch[2]
            dones = batch[3] if len(batch) > 3 else None

        states = to_torch(states, device=self.device, dtype=torch.float32)
        actions = to_torch(actions, device=self.device, dtype=torch.float32)
        if next_states is None:
            next_states = torch.zeros_like(states)
        else:
            next_states = to_torch(
                next_states, device=self.device, dtype=torch.float32
            )
        if dones is None:
            dones = torch.zeros(states.shape[0], device=self.device)
        else:
            dones = to_torch(dones, device=self.device, dtype=torch.float32)
        if dones.ndim == 0:
            dones = dones.unsqueeze(0)
        if dones.ndim == 1:
            dones = dones.unsqueeze(-1)
        return states, actions, next_states, dones

    def train_step(self, batch: Any) -> Dict[str, float]:
        """Perform one ICM, successor-feature, and policy update.

        Parameters
        ----------
        batch:
            A transition batch from an offline dataset (dict or tuple).

        Returns
        -------
        Dictionary of scalar loss values for logging.
        """
        states, actions, next_states, dones = self._unpack_batch(batch)
        batch_size = states.shape[0]
        if batch_size == 0:
            return {
                "icm_loss": 0.0,
                "sf_loss": 0.0,
                "policy_loss": 0.0,
                "total_loss": 0.0,
            }

        # ---- ICM representation learning ---------------------------------
        phi, phi_next, pred_action, pred_phi_next = self.icm(
            states, actions, next_states
        )
        inverse_loss = F.mse_loss(pred_action, actions)
        forward_loss = F.mse_loss(pred_phi_next, phi_next.detach())
        icm_loss = inverse_loss + forward_loss

        self.icm_optimizer.zero_grad()
        icm_loss.backward()
        nn.utils.clip_grad_norm_(self.icm.parameters(), 10.0)
        self.icm_optimizer.step()

        # Use detached features for successor-feature learning.
        phi = phi.detach()

        # ---- Successor-feature TD update ---------------------------------
        w = self._sample_w(batch_size)
        with torch.no_grad():
            next_action = self.policy_target(next_states, w)
            psi_next = self.psi_target(
                torch.cat([next_states, next_action], dim=-1)
            )
            target_psi = phi + self.gamma * (1.0 - dones) * psi_next

        psi = self.psi(torch.cat([states, actions], dim=-1))
        sf_loss = F.mse_loss(psi, target_psi.detach())

        self.sf_optimizer.zero_grad()
        sf_loss.backward()
        nn.utils.clip_grad_norm_(self.psi.parameters(), 10.0)
        self.sf_optimizer.step()

        # ---- DDPG policy update -------------------------------------------
        sampled_actions = self.policy(states, w)
        q_values = (self.psi(torch.cat([states, sampled_actions], dim=-1)).detach() * w).sum(dim=-1)
        policy_loss = -q_values.mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
        self.policy_optimizer.step()

        # ---- Target updates ----------------------------------------------
        soft_update(self.psi_target, self.psi, self.tau)
        soft_update(self.policy_target, self.policy, self.tau)

        return {
            "icm_loss": float(icm_loss.detach().cpu().item()),
            "sf_loss": float(sf_loss.detach().cpu().item()),
            "policy_loss": float(policy_loss.detach().cpu().item()),
            "total_loss": float(
                (icm_loss + sf_loss + policy_loss).detach().cpu().item()
            ),
        }

    def _features_np(self, states: np.ndarray) -> np.ndarray:
        """Extract ICM features for a NumPy state array."""
        self.icm.eval()
        with torch.no_grad():
            state_t = to_torch(states, device=self.device, dtype=torch.float32)
            features = self.icm.encode(state_t)
        return features.detach().cpu().numpy()

    @torch.no_grad()
    def get_task_vector(
        self,
        reward_fn: Callable[[np.ndarray], np.ndarray],
        state_pool: np.ndarray,
        num_samples: Optional[int] = None,
        ridge: float = 1e-3,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Fit a linear reward vector :math:`w` on ICM features.

        Parameters
        ----------
        reward_fn:
            Scalar reward function accepting a NumPy state array of shape
            ``(N, state_dim)``.
        state_pool:
            Pool of offline states from which evaluation pairs are sampled.
        num_samples:
            Number of state-reward examples to regress on. Defaults to
            ``self.eval_reward_samples``.
        ridge:
            Ridge regularisation coefficient.

        Returns
        -------
        Task vector of shape ``(feature_dim,)``.
        """
        if num_samples is None:
            num_samples = self.eval_reward_samples
        states, rewards = sample_reward_pairs(
            reward_fn, state_pool, num_samples=num_samples, seed=seed
        )
        features = self._features_np(states)
        w = ridge_regression(features, rewards, ridge=ridge)
        return w

    def get_task_policy(
        self,
        reward_fn: Callable[[np.ndarray], np.ndarray],
        state_pool: np.ndarray,
        num_samples: Optional[int] = None,
        ridge: float = 1e-3,
        seed: Optional[int] = None,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Return a NumPy observation-to-action policy for a downstream reward.

        The returned closure uses exactly ``num_samples`` state-reward examples
        to infer a task vector, then conditions the pretrained deterministic
        policy on that vector.
        """
        w = self.get_task_vector(
            reward_fn,
            state_pool,
            num_samples=num_samples,
            ridge=ridge,
            seed=seed,
        )
        context = torch.as_tensor(
            w, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        return make_policy_fn_from_net(
            self.policy,
            context,
            device=self.device,
            deterministic=True,
            state_is_tensor=False,
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "icm": self.icm.state_dict(),
            "psi": self.psi.state_dict(),
            "psi_target": self.psi_target.state_dict(),
            "policy": self.policy.state_dict(),
            "policy_target": self.policy_target.state_dict(),
            "icm_optimizer": self.icm_optimizer.state_dict(),
            "sf_optimizer": self.sf_optimizer.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.icm.load_state_dict(state_dict["icm"])
        self.psi.load_state_dict(state_dict["psi"])
        self.psi_target.load_state_dict(state_dict["psi_target"])
        self.policy.load_state_dict(state_dict["policy"])
        self.policy_target.load_state_dict(state_dict["policy_target"])
        self.icm_optimizer.load_state_dict(state_dict["icm_optimizer"])
        self.sf_optimizer.load_state_dict(state_dict["sf_optimizer"])
        self.policy_optimizer.load_state_dict(state_dict["policy_optimizer"])

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location: Optional[str] = "cpu") -> None:
        checkpoint = torch.load(path, map_location=map_location)
        self.load_state_dict(checkpoint)
