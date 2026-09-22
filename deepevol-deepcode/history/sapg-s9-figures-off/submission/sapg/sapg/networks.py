"""Neural network modules for SAPG.

Implements the shared actor backbone ``B_theta`` and critic backbone ``C_psi``
described in Section 4.4 of the SAPG paper.  Both backbones are *shared* across
the leader and all follower policies and are conditioned on a per-policy latent
vector ``phi_j`` (the same ``phi_j`` is used for the actor and the critic of
policy ``j``).

Key design points (from the paper + addendum):

* Conditioning is implemented by concatenating ``phi_j`` to the observation /
  state input before the backbone MLP.
* ``theta`` and ``psi`` receive gradients from *all* objectives, while ``phi_j``
  is updated only by policy ``j``'s objective.  This is achieved by giving each
  policy its own ``nn.Parameter`` for ``phi_j`` and by masking gradients in the
  training loop (see ``algorithm.py``).
* AllegroKuka actor: ``obs -> MLP[768, 512, 256] ELU -> LSTM(1 layer, 768) ->
  Gaussian mean`` with a fixed (input independent) learnable ``sigma`` vector.
* ShadowHand actor: ``MLP[512, 512, 256, 128] ELU -> Gaussian mean``.
* AllegroHand actor: ``MLP[512, 256, 128] ELU -> Gaussian mean``.
* Critic backbones mirror the actor backbone dimensions and output a scalar
  value ``V(s)``.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SAPGConfig


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def get_activation(name: str) -> nn.Module:
    """Return an activation module by name (paper uses ELU)."""
    name = name.lower()
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation: {name}")


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    activation: str = "elu",
    output_dim: Optional[int] = None,
) -> nn.Sequential:
    """Build a plain MLP with the given hidden dims and activation."""
    layers: List[nn.Module] = []
    prev = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(get_activation(activation))
        prev = h
    if output_dim is not None:
        layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


def init_weights(module: nn.Module, gain: float = math.sqrt(2.0)) -> None:
    """Orthogonal init for linear layers (standard PPO practice)."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
# Latent conditioning
# ---------------------------------------------------------------------------
class LatentConditionedInput(nn.Module):
    """Concatenates the per-policy latent ``phi_j`` to the input features.

    The paper does not specify the conditioning mechanism in detail; we follow
    the addendum's suggestion and concatenate ``phi_j`` to the observation.
    """

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    @property
    def output_dim(self) -> int:
        return self.input_dim + self.latent_dim

    def forward(self, obs: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        if self.latent_dim == 0:
            return obs
        # latent may be [latent_dim] (shared) or [B, latent_dim] (per-sample)
        if latent.dim() == 1:
            latent = latent.unsqueeze(0).expand(obs.shape[0], -1)
        return torch.cat([obs, latent], dim=-1)


# ---------------------------------------------------------------------------
# Actor
# ---------------------------------------------------------------------------
class Actor(nn.Module):
    """Shared actor backbone ``B_theta`` conditioned on ``phi_j``.

    Supports both an LSTM variant (AllegroKuka) and a feed-forward MLP variant
    (ShadowHand / AllegroHand).  The action distribution is a diagonal Gaussian
    with a *fixed* (input independent) learnable standard deviation, as in the
    paper.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int],
        latent_dim: int = 0,
        use_lstm: bool = False,
        lstm_hidden_size: int = 768,
        lstm_num_layers: int = 1,
        activation: str = "elu",
        init_log_std: float = 0.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.use_lstm = use_lstm
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers

        self.conditioning = LatentConditionedInput(obs_dim, latent_dim)
        in_dim = self.conditioning.output_dim

        if use_lstm:
            # MLP trunk -> LSTM -> Gaussian mean
            self.trunk = build_mlp(in_dim, hidden_dims, activation)
            trunk_out = hidden_dims[-1] if len(hidden_dims) > 0 else in_dim
            self.lstm = nn.LSTM(
                input_size=trunk_out,
                hidden_size=lstm_hidden_size,
                num_layers=lstm_num_layers,
                batch_first=True,
            )
            self.mean_head = nn.Linear(lstm_hidden_size, action_dim)
        else:
            self.trunk = build_mlp(in_dim, hidden_dims, activation)
            trunk_out = hidden_dims[-1] if len(hidden_dims) > 0 else in_dim
            self.mean_head = nn.Linear(trunk_out, action_dim)

        # Fixed, input-independent learnable log-std vector.
        self.log_std = nn.Parameter(torch.full((action_dim,), float(init_log_std)))

        self.apply(init_weights)
        # Small init for the mean head so initial actions are near zero.
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)

    # -- helpers -----------------------------------------------------------
    def _encode(self, obs: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        x = self.conditioning(obs, latent)
        return self.trunk(x)

    def forward(
        self,
        obs: torch.Tensor,
        latent: torch.Tensor,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        seq_len: Optional[int] = None,
    ):
        """Compute the Gaussian mean.

        Args:
            obs: ``[B, obs_dim]`` (or ``[B, T, obs_dim]`` when ``seq_len`` given).
            latent: ``[latent_dim]`` or ``[B, latent_dim]``.
            lstm_state: optional ``(h, c)`` tuple for the LSTM variant.
            seq_len: if provided, ``obs`` is reshaped to ``[B, T, obs_dim]`` and
                processed as a sequence (used for the LSTM variant).

        Returns:
            mean: ``[B, action_dim]`` (or ``[B, T, action_dim]`` if seq_len).
            new_lstm_state: ``(h, c)`` or ``None``.
        """
        if seq_len is not None and self.use_lstm:
            B, T, D = obs.shape
            x = obs.reshape(B * T, D)
            if latent.dim() == 1:
                lat = latent.unsqueeze(0).expand(B * T, -1)
            else:
                lat = latent.unsqueeze(1).expand(B, T, -1).reshape(B * T, -1)
            x = self._encode(x, lat)
            x = x.reshape(B, T, -1)
            x, new_state = self.lstm(x, lstm_state)
            mean = self.mean_head(x)
            return mean, new_state

        x = self._encode(obs, latent)
        if self.use_lstm:
            # Single-step LSTM forward.
            x = x.unsqueeze(1)  # [B, 1, H]
            x, new_state = self.lstm(x, lstm_state)
            x = x.squeeze(1)
            mean = self.mean_head(x)
            return mean, new_state
        mean = self.mean_head(x)
        return mean, None

    def distribution(
        self,
        obs: torch.Tensor,
        latent: torch.Tensor,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        seq_len: Optional[int] = None,
    ):
        """Return a ``torch.distributions.Normal`` for the given inputs."""
        mean, new_state = self.forward(obs, latent, lstm_state, seq_len)
        std = torch.exp(self.log_std).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        return dist, new_state

    def log_prob(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        latent: torch.Tensor,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        dist, _ = self.distribution(obs, latent, lstm_state, seq_len)
        return dist.log_prob(actions).sum(dim=-1)

    def entropy(
        self,
        obs: torch.Tensor,
        latent: torch.Tensor,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        dist, _ = self.distribution(obs, latent, lstm_state, seq_len)
        return dist.entropy().sum(dim=-1)

    def act(
        self,
        obs: torch.Tensor,
        latent: torch.Tensor,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        deterministic: bool = False,
    ):
        """Sample (or take the mean of) an action for a single step."""
        dist, new_state = self.distribution(obs, latent, lstm_state)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, new_state


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------
class Critic(nn.Module):
    """Shared critic backbone ``C_psi`` conditioned on ``phi_j``.

    Mirrors the actor backbone dimensions (minus the final action head) and
    outputs a scalar state value ``V(s)``.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dims: Sequence[int],
        latent_dim: int = 0,
        use_lstm: bool = False,
        lstm_hidden_size: int = 768,
        lstm_num_layers: int = 1,
        activation: str = "elu",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.use_lstm = use_lstm
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers

        self.conditioning = LatentConditionedInput(obs_dim, latent_dim)
        in_dim = self.conditioning.output_dim

        if use_lstm:
            self.trunk = build_mlp(in_dim, hidden_dims, activation)
            trunk_out = hidden_dims[-1] if len(hidden_dims) > 0 else in_dim
            self.lstm = nn.LSTM(
                input_size=trunk_out,
                hidden_size=lstm_hidden_size,
                num_layers=lstm_num_layers,
                batch_first=True,
            )
            self.value_head = nn.Linear(lstm_hidden_size, 1)
        else:
            self.trunk = build_mlp(in_dim, hidden_dims, activation)
            trunk_out = hidden_dims[-1] if len(hidden_dims) > 0 else in_dim
            self.value_head = nn.Linear(trunk_out, 1)

        self.apply(init_weights)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def _encode(self, obs: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        x = self.conditioning(obs, latent)
        return self.trunk(x)

    def forward(
        self,
        obs: torch.Tensor,
        latent: torch.Tensor,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        seq_len: Optional[int] = None,
    ):
        """Return ``V(s)`` of shape ``[B]`` (or ``[B, T]`` when ``seq_len``)."""
        if seq_len is not None and self.use_lstm:
            B, T, D = obs.shape
            x = obs.reshape(B * T, D)
            if latent.dim() == 1:
                lat = latent.unsqueeze(0).expand(B * T, -1)
            else:
                lat = latent.unsqueeze(1).expand(B, T, -1).reshape(B * T, -1)
            x = self._encode(x, lat)
            x = x.reshape(B, T, -1)
            x, new_state = self.lstm(x, lstm_state)
            value = self.value_head(x).squeeze(-1)
            return value, new_state

        x = self._encode(obs, latent)
        if self.use_lstm:
            x = x.unsqueeze(1)
            x, new_state = self.lstm(x, lstm_state)
            x = x.squeeze(1)
            value = self.value_head(x).squeeze(-1)
            return value, new_state
        value = self.value_head(x).squeeze(-1)
        return value, None


# ---------------------------------------------------------------------------
# Container holding the shared backbones + per-policy latents
# ---------------------------------------------------------------------------
class SAPGNetworks(nn.Module):
    """Container for the shared actor/critic backbones and per-policy latents.

    ``phi_j`` are stored as a single ``nn.Parameter`` of shape
    ``[num_policies, latent_dim]``.  The training loop is responsible for
    zeroing the gradients of ``phi_j`` for all ``j != i`` when applying policy
    ``i``'s objective (see ``algorithm.py``).
    """

    def __init__(self, config: SAPGConfig, obs_dim: int, action_dim: int):
        super().__init__()
        self.config = config
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        num_policies = max(1, config.num_blocks)
        latent_dim = config.latent_dim

        self.actor = Actor(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dims=config.actor_hidden_dims,
            latent_dim=latent_dim,
            use_lstm=config.use_lstm,
            lstm_hidden_size=config.lstm_hidden_size,
            lstm_num_layers=config.lstm_num_layers,
            activation=config.activation,
        )
        self.critic = Critic(
            obs_dim=obs_dim,
            hidden_dims=config.critic_hidden_dims,
            latent_dim=latent_dim,
            use_lstm=config.use_lstm,
            lstm_hidden_size=config.lstm_hidden_size,
            lstm_num_layers=config.lstm_num_layers,
            activation=config.activation,
        )

        if latent_dim > 0:
            # Small random init for the per-policy latents.
            self.phi = nn.Parameter(torch.randn(num_policies, latent_dim) * 0.1)
        else:
            self.register_parameter("phi", None)

    # -- accessors ---------------------------------------------------------
    def latent(self, policy_idx: int) -> torch.Tensor:
        """Return ``phi_j`` for policy ``j`` (or an empty tensor if unused)."""
        if self.phi is None:
            return torch.zeros(0, device=self._device())
        return self.phi[policy_idx]

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    # -- forward helpers ---------------------------------------------------
    def actor_distribution(self, obs, policy_idx, lstm_state=None, seq_len=None):
        return self.actor.distribution(
            obs, self.latent(policy_idx), lstm_state, seq_len
        )

    def actor_log_prob(self, obs, actions, policy_idx, lstm_state=None, seq_len=None):
        return self.actor.log_prob(
            obs, actions, self.latent(policy_idx), lstm_state, seq_len
        )

    def actor_entropy(self, obs, policy_idx, lstm_state=None, seq_len=None):
        return self.actor.entropy(obs, self.latent(policy_idx), lstm_state, seq_len)

    def actor_act(self, obs, policy_idx, lstm_state=None, deterministic=False):
        return self.actor.act(
            obs, self.latent(policy_idx), lstm_state, deterministic
        )

    def critic_value(self, obs, policy_idx, lstm_state=None, seq_len=None):
        return self.critic(obs, self.latent(policy_idx), lstm_state, seq_len)

    # -- gradient masking --------------------------------------------------
    def zero_other_latent_grads(self, policy_idx: int) -> None:
        """Zero ``phi`` gradients for all policies except ``policy_idx``.

        Called before ``loss.backward()`` for policy ``i``'s objective so that
        ``phi_i`` is updated only by its own objective.
        """
        if self.phi is None or self.phi.grad is None:
            return
        mask = torch.ones_like(self.phi)
        mask[policy_idx] = 0.0
        self.phi.grad = self.phi.grad * mask


def build_networks(config: SAPGConfig, obs_dim: int, action_dim: int) -> SAPGNetworks:
    """Factory used by the training loop."""
    return SAPGNetworks(config, obs_dim, action_dim)
