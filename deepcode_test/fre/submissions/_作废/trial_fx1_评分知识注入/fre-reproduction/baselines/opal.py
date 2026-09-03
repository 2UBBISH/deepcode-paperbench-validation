"""OPAL unsupervised skill-discovery baseline.

This is a self-contained offline implementation of OPAL-style skill discovery:
a trajectory autoencoder learns a skill posterior over whole state-action
trajectories, a decoder reconstructs the trajectory from a skill code and the
initial observation, and a skill-conditioned policy is trained by behavioral
cloning so that evaluation can roll out a selected skill online.

Evaluation is *privileged*: a fixed number of skills (default 10) are sampled,
each is rolled out online, and the skill with the highest downstream task
return is selected (handled by the evaluation scripts).
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fre.utils import to_torch
from .baseline_utils import GaussianPolicy, make_policy_fn_from_net

__all__ = ["OPAL", "TrajectoryEncoder", "TrajectoryDecoder"]


class TrajectoryEncoder(nn.Module):
    """GRU trajectory encoder producing Gaussian skill posterior."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        skill_dim: int,
        hidden_size: int = 256,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.skill_dim = skill_dim
        self.gru = nn.GRU(
            input_size=state_dim + action_dim,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.mu_head = nn.Linear(hidden_size, skill_dim)
        self.logvar_head = nn.Linear(hidden_size, skill_dim)

    def forward(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, logvar) of the posterior q(z | trajectory)."""
        x = torch.cat([states, actions], dim=-1)  # (B, T, state_dim + action_dim)
        _, h = self.gru(x)
        last_hidden = h[-1]  # (B, hidden_size)
        mu = self.mu_head(last_hidden)
        logvar = self.logvar_head(last_hidden)
        return mu, logvar


class TrajectoryDecoder(nn.Module):
    """GRU cell trajectory decoder conditioned on a skill code."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        skill_dim: int,
        hidden_size: int = 256,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.skill_dim = skill_dim
        self.gru_cell = nn.GRUCell(input_size=state_dim + action_dim, hidden_size=hidden_size)
        self.z_to_hidden = nn.Linear(skill_dim, hidden_size)
        self.action_head = nn.Linear(hidden_size, action_dim)
        self.next_state_head = nn.Linear(hidden_size, state_dim)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (pred_actions, pred_next_states) with teacher forcing."""
        B, T, _ = states.shape
        h = self.z_to_hidden(z)
        pred_actions: list[torch.Tensor] = []
        pred_next_states: list[torch.Tensor] = []

        action_prev = torch.zeros(B, self.action_dim, device=states.device)
        for t in range(T):
            state_t = states[:, t, :]
            x = torch.cat([state_t, action_prev], dim=-1)
            h = self.gru_cell(x, h)
            pred_a = self.action_head(h)
            pred_ns = self.next_state_head(h)
            pred_actions.append(pred_a)
            pred_next_states.append(pred_ns)
            # Teacher forcing: use ground-truth action as next input.
            action_prev = actions[:, t, :]

        pred_actions = torch.stack(pred_actions, dim=1)  # (B, T, action_dim)
        pred_next_states = torch.stack(pred_next_states, dim=1)  # (B, T, state_dim)
        return pred_actions, pred_next_states


class OPAL:
    """Offline OPAL skill-discovery agent.

    The agent learns:
      * a trajectory encoder q(z | s_0:T, a_0:T),
      * a trajectory decoder p(s, a | z),
      * a skill-conditioned Gaussian policy pi(a | s, z) via behavioral cloning.

    Evaluation is privileged and selects the best of a small set of sampled
    skills using online rollouts (handled outside this class).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        skill_dim: int = 16,
        hidden_dims: Sequence[int] = (256, 256),
        encoder_hidden: int = 256,
        decoder_hidden: int = 256,
        lr: float = 3e-4,
        beta: float = 1.0,
        batch_size: int = 256,
        horizon: int = 16,
        device: Union[str, torch.device] = "cpu",
        max_action: float = 1.0,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.skill_dim = skill_dim
        self.hidden_dims = tuple(hidden_dims)
        self.batch_size = batch_size
        self.horizon = horizon
        self.beta = beta
        self.device = torch.device(device)
        self.max_action = max_action

        self.encoder = TrajectoryEncoder(
            state_dim=state_dim,
            action_dim=action_dim,
            skill_dim=skill_dim,
            hidden_size=encoder_hidden,
        )
        self.decoder = TrajectoryDecoder(
            state_dim=state_dim,
            action_dim=action_dim,
            skill_dim=skill_dim,
            hidden_size=decoder_hidden,
        )
        self.policy = GaussianPolicy(
            state_dim=state_dim,
            context_dim=skill_dim,
            action_dim=action_dim,
            hidden_dims=self.hidden_dims,
            max_action=max_action,
        )

        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.policy.parameters()),
            lr=lr,
        )
        self.to(self.device)

    def to(self, device: Union[str, torch.device]) -> "OPAL":
        self.device = torch.device(device)
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        self.policy.to(self.device)
        return self

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    @staticmethod
    def _kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return 0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()

    def _prepare_trajectories(
        self,
        states: Union[np.ndarray, torch.Tensor],
        actions: Union[np.ndarray, torch.Tensor],
        next_states: Optional[Union[np.ndarray, torch.Tensor]],
        horizon: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert flat transition batches to (B, T, dim) trajectory segments."""
        states = to_torch(states, self.device)
        actions = to_torch(actions, self.device)
        if next_states is None:
            next_states = states.clone()
        else:
            next_states = to_torch(next_states, self.device)

        if states.dim() == 3:
            return states, actions, next_states

        # Flat (N, dim) -> segments of length horizon.
        h = horizon or self.horizon
        total = states.shape[0]
        if total < h:
            h = total
        batch_size = total // h
        if batch_size == 0:
            batch_size = 1
            h = total
        length = batch_size * h
        states = states[:length].reshape(batch_size, h, self.state_dim)
        actions = actions[:length].reshape(batch_size, h, self.action_dim)
        next_states = next_states[:length].reshape(batch_size, h, self.state_dim)
        return states, actions, next_states

    def train_step(
        self,
        states: Union[np.ndarray, torch.Tensor],
        actions: Union[np.ndarray, torch.Tensor],
        next_states: Optional[Union[np.ndarray, torch.Tensor]] = None,
        dones: Optional[Union[np.ndarray, torch.Tensor]] = None,
        horizon: Optional[int] = None,
    ) -> Dict[str, float]:
        """One OPAL update on trajectory segments.

        Args:
            states: Flat transition states or (B, T, state_dim) trajectories.
            actions: Flat transition actions or (B, T, action_dim) trajectories.
            next_states: Optional flat next states; defaults to states when None.
            dones: Unused; kept for interface parity with other baselines.
            horizon: Segment length used when flat inputs are provided.

        Returns:
            Dictionary of scalar losses.
        """
        self.encoder.train()
        self.decoder.train()
        self.policy.train()

        states, actions, next_states = self._prepare_trajectories(
            states, actions, next_states, horizon
        )
        B, T, _ = states.shape

        # Encode posterior.
        mu, logvar = self.encoder(states, actions)
        z = self._reparameterize(mu, logvar)

        # Decode trajectory.
        pred_actions, pred_next_states = self.decoder(states, actions, z)

        action_recon = F.mse_loss(pred_actions, actions)
        # Shift states by one for next-state targets; last target is the final state.
        next_state_targets = torch.cat([states[:, 1:, :], states[:, -1:, :]], dim=1)
        state_recon = F.mse_loss(pred_next_states, next_state_targets)
        recon_loss = action_recon + state_recon

        # Skill-conditioned behavioral cloning.
        flat_states = states.reshape(-1, self.state_dim)
        flat_actions = actions.reshape(-1, self.action_dim)
        z_exp = z.unsqueeze(1).expand(-1, T, -1).reshape(-1, self.skill_dim)
        _, _, _, log_prob = self.policy(flat_states, z_exp)
        policy_loss = -log_prob.mean()

        kl_loss = self._kl_divergence(mu, logvar)
        total_loss = recon_loss + self.beta * kl_loss + policy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "total_loss": float(total_loss.detach().cpu().item()),
            "recon_loss": float(recon_loss.detach().cpu().item()),
            "kl_loss": float(kl_loss.detach().cpu().item()),
            "policy_loss": float(policy_loss.detach().cpu().item()),
        }

    def encode_skill(
        self,
        states: Union[np.ndarray, torch.Tensor],
        actions: Union[np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        """Return a sampled skill code for a trajectory."""
        self.encoder.eval()
        with torch.no_grad():
            states_t = to_torch(states, self.device)
            actions_t = to_torch(actions, self.device)
            if states_t.dim() == 2:
                states_t = states_t.unsqueeze(0)
                actions_t = actions_t.unsqueeze(0)
            mu, logvar = self.encoder(states_t, actions_t)
            z = self._reparameterize(mu, logvar)
        return z

    def sample_skills(self, num_skills: int = 10, seed: Optional[int] = None) -> torch.Tensor:
        """Sample skills from the unit-Gaussian prior."""
        if seed is not None:
            g = torch.Generator(device=self.device)
            g.manual_seed(seed)
            return torch.randn(num_skills, self.skill_dim, device=self.device, generator=g)
        return torch.randn(num_skills, self.skill_dim, device=self.device)

    def get_task_policy(
        self,
        skill_z: Union[np.ndarray, torch.Tensor],
        deterministic: bool = True,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Return an observation->action closure conditioned on `skill_z`."""
        skill_z = to_torch(skill_z, self.device)
        if skill_z.dim() == 1:
            skill_z = skill_z.unsqueeze(0)
        self.policy.eval()
        return make_policy_fn_from_net(
            self.policy,
            context=skill_z,
            device=self.device,
            deterministic=deterministic,
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.encoder.load_state_dict(state_dict["encoder"])
        self.decoder.load_state_dict(state_dict["decoder"])
        self.policy.load_state_dict(state_dict["policy"])
        if "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location=self.device))
