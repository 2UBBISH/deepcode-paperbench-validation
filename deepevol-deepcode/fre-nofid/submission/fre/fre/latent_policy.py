"""z-conditioned policy and value networks for FRE (Section 4.3).

After the FRE encoder/decoder are pretrained (Phase 1), the encoder is frozen and a
z-conditioned IQL policy is trained (Phase 2) on rewards sampled from the unsupervised
prior.  Every network consumes the latent task embedding ``z`` by *concatenating* it to
the observation:

    Q(s, a, z) -> scalar
    V(s, z)    -> scalar
    pi(a | s, z) -> Gaussian over actions

The observation inputs may be either raw states or an already-augmented observation
(e.g. ExORL appends physics quantities before feeding the network).  The latent ``z`` is
always concatenated on the *right* of the observation.

The heavy lifting (expectile regression, AWR policy extraction, target networks) lives in
``rl/iql.py``; this module only provides the z-conditioned network definitions together
with small helpers for encoding z from reward samples and for sampling actions.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "LatentMLP",
    "LatentQNetwork",
    "LatentVNetwork",
    "LatentGaussianPolicy",
    "LatentPolicyBundle",
    "DEFAULT_HIDDEN_SIZES",
    "DEFAULT_LATENT_DIM",
    "LOG_STD_MIN",
    "LOG_STD_MAX",
    "sample_latent_z",
    "concat_obs_z",
]

# IQL nets use [512, 512, 512] layers (Section 4.3 / Appendix).
DEFAULT_HIDDEN_SIZES: Tuple[int, ...] = (512, 512, 512)
DEFAULT_LATENT_DIM: int = 128

LOG_STD_MIN: float = -20.0
LOG_STD_MAX: float = 2.0

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "elu": nn.ELU,
}


def _get_activation(name: str) -> nn.Module:
    if name not in _ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{name}'. Choose from {sorted(_ACTIVATIONS)}."
        )
    return _ACTIVATIONS[name]()


def _build_mlp(
    in_dim: int,
    hidden_sizes: Sequence[int],
    out_dim: int,
    activation: str = "relu",
    output_activation: Optional[str] = None,
) -> nn.Sequential:
    """Stack Linear+activation layers, with an optional output activation."""
    layers = []
    last = in_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(last, h))
        layers.append(_get_activation(activation))
        last = h
    layers.append(nn.Linear(last, out_dim))
    if output_activation is not None:
        layers.append(_get_activation(output_activation))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Observation / latent helpers
# ---------------------------------------------------------------------------
def concat_obs_z(obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Concatenate a latent task embedding to an observation batch.

    Supports:
      * ``obs`` of shape ``(B, O)`` and ``z`` of shape ``(B, Z)`` or ``(Z,)``.
      * ``obs`` of shape ``(B, K, O)`` and ``z`` of shape ``(B, Z)`` / ``(B,1,Z)`` / ``(Z,)``.
    """
    if z.dim() == 1:
        z = z.unsqueeze(0)
    # Expand z across any extra leading dims of obs (e.g. a candidate-action dim).
    if obs.dim() > 2 and z.dim() == 2:
        target_shape = (obs.shape[0],) + (1,) * (obs.dim() - 2) + (z.shape[-1],)
        z = z.view(target_shape).expand(obs.shape[:-1] + (z.shape[-1],))
    elif z.dim() > 2:
        z = z.reshape(-1, z.shape[-1])
        if z.shape[0] == 1 and obs.shape[0] != 1:
            z = z.expand(obs.shape[0], -1)
    if z.shape[0] == 1 and obs.shape[0] != 1:
        z = z.expand(obs.shape[0], *z.shape[1:])
    return torch.cat([obs, z], dim=-1)


def sample_latent_z(
    encoder: nn.Module,
    context_states: torch.Tensor,
    context_rewards: torch.Tensor,
    use_mean: bool = False,
) -> torch.Tensor:
    """Encode a batch of reward contexts into latent task embeddings ``z``.

    Thin wrapper around an FRE encoder exposing ``encode(states, rewards, use_mean=...)``.
    Provided so the RL training loop can re-sample and re-encode ``z`` every iteration
    with a single call.
    """
    if hasattr(encoder, "encode"):
        return encoder.encode(context_states, context_rewards, use_mean=use_mean)
    mu, log_sigma = encoder(context_states, context_rewards)
    if use_mean:
        return mu
    std = torch.exp(0.5 * log_sigma.clamp(-10.0, 5.0))
    return mu + std * torch.randn_like(std)


# ---------------------------------------------------------------------------
# z-conditioned networks
# ---------------------------------------------------------------------------
class LatentMLP(nn.Module):
    """Backbone MLP mapping ``[obs ; z]`` to a hidden feature (no output activation)."""

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int = DEFAULT_LATENT_DIM,
        hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self.in_dim = self.obs_dim + self.latent_dim
        self.activation = activation
        # Build body layers manually so we can expose the last hidden dimension.
        layers = []
        last = self.in_dim
        for h in self.hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(_get_activation(activation))
            last = h
        self.body = nn.Sequential(*layers)
        self.feature_dim = last if self.hidden_sizes else self.in_dim

    def forward(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = concat_obs_z(obs, z)
        return self.body(x)


class LatentQNetwork(nn.Module):
    """Action-value network ``Q(s, a, z)`` with z concatenated to ``[s ; a]``."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = DEFAULT_LATENT_DIM,
        hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self.in_dim = self.obs_dim + self.action_dim + self.latent_dim
        self.activation = activation
        self.net = _build_mlp(self.in_dim, self.hidden_sizes, 1, activation=activation)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        if action.dim() == obs.dim() - 1:
            action = action.unsqueeze(-1)
        sa = torch.cat([obs, action], dim=-1)
        x = concat_obs_z(sa, z)
        return self.net(x).squeeze(-1)


class LatentVNetwork(nn.Module):
    """State-value network ``V(s, z)`` with z concatenated to the observation."""

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int = DEFAULT_LATENT_DIM,
        hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self.in_dim = self.obs_dim + self.latent_dim
        self.activation = activation
        self.net = _build_mlp(self.in_dim, self.hidden_sizes, 1, activation=activation)

    def forward(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = concat_obs_z(obs, z)
        return self.net(x).squeeze(-1)


class LatentGaussianPolicy(nn.Module):
    """Diagonal-Gaussian policy ``pi(a | s, z)`` with z concatenated to the observation.

    Two heads (mean, log-std) share the MLP body.  ``log_std`` is clamped to
    ``[LOG_STD_MIN, LOG_STD_MAX]`` for numerical stability.  The action is squashed with
    ``tanh`` by default (matching continuous-control convention) and the corresponding
    change-of-variables correction is exposed via :meth:`log_prob`.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = DEFAULT_LATENT_DIM,
        hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
        activation: str = "relu",
        squashed: bool = True,
        log_std_min: float = LOG_STD_MIN,
        log_std_max: float = LOG_STD_MAX,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self.in_dim = self.obs_dim + self.latent_dim
        self.activation = activation
        self.squashed = squashed
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        self.body = LatentMLP(
            obs_dim, latent_dim, hidden_sizes, activation=activation
        )
        self.mean_layer = nn.Linear(self.body.feature_dim, self.action_dim)
        self.log_std_layer = nn.Linear(self.body.feature_dim, self.action_dim)

    # -- distributions -----------------------------------------------------
    def forward(
        self, obs: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the Gaussian parameters ``(mean, log_std)``."""
        h = self.body(obs, z)
        mean = self.mean_layer(h)
        log_std = self.log_std_layer(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(
        self,
        obs: torch.Tensor,
        z: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample (or take the mean of) an action; returns ``(action, log_prob)``."""
        mean, log_std = self.forward(obs, z)
        if deterministic:
            raw_action = mean
            log_prob = torch.zeros(mean.shape[:-1], device=mean.device, dtype=mean.dtype)
        else:
            std = torch.exp(log_std)
            eps = torch.randn_like(std)
            raw_action = mean + std * eps
            log_prob = -0.5 * (
                ((raw_action - mean) / std) ** 2
                + 2.0 * log_std
                + math.log(2.0 * math.pi)
            ).sum(dim=-1)
        if self.squashed:
            action = torch.tanh(raw_action)
            if not deterministic:
                log_prob = log_prob - torch.log(
                    1.0 - action.pow(2) + 1e-6
                ).sum(dim=-1)
        else:
            action = raw_action
        return action, log_prob

    def log_prob(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        z: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Log-density of an action (used by AWR / GC-BC style objectives)."""
        if action.dim() == obs.dim() - 1:
            action = action.unsqueeze(-1)
        mean, log_std = self.forward(obs, z)
        if self.squashed:
            # Inverse tanh (with clamp for numerical safety).
            action_clamped = action.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            raw_action = 0.5 * torch.log(
                (1.0 + action_clamped) / (1.0 - action_clamped)
            )
        else:
            raw_action = action
        log_prob = -0.5 * (
            ((raw_action - mean) / torch.exp(log_std)) ** 2
            + 2.0 * log_std
            + math.log(2.0 * math.pi)
        ).sum(dim=-1)
        if self.squashed:
            log_prob = log_prob - torch.log(
                1.0 - action_clamped.pow(2) + 1e-6
            ).sum(dim=-1)
        return log_prob

    def act(
        self,
        obs: torch.Tensor,
        z: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Convenience wrapper returning only the action (for evaluation rollouts)."""
        action, _ = self.sample(obs, z, deterministic=deterministic)
        return action


class LatentPolicyBundle(nn.Module):
    """Container holding the z-conditioned ``Q``, ``V`` and ``pi`` networks.

    The IQL trainer (``rl/iql.py``) operates on this bundle.  Target networks are *not*
    created here -- the trainer owns them so it can control the EMA update rate (0.001).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = DEFAULT_LATENT_DIM,
        hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
        activation: str = "relu",
        squashed_policy: bool = True,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)

        self.q = LatentQNetwork(
            obs_dim, action_dim, latent_dim, hidden_sizes, activation
        )
        self.v = LatentVNetwork(obs_dim, latent_dim, hidden_sizes, activation)
        self.pi = LatentGaussianPolicy(
            obs_dim, action_dim, latent_dim, hidden_sizes, activation,
            squashed=squashed_policy,
        )

    # -- IQL forward passes -------------------------------------------------
    def q_value(
        self, obs: torch.Tensor, action: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        return self.q(obs, action, z)

    def value(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.v(obs, z)

    def policy(
        self, obs: torch.Tensor, z: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.pi.sample(obs, z, deterministic=deterministic)

    def policy_log_prob(
        self, obs: torch.Tensor, action: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        return self.pi.log_prob(obs, action, z)

    # -- checkpointing ------------------------------------------------------
    def policy_state_dict(self) -> dict:
        return self.pi.state_dict()

    def load_policy_state_dict(self, state_dict: dict) -> None:
        self.pi.load_state_dict(state_dict)
