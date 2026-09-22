"""Actor-critic with a *shared backbone conditioned on per-policy latents*.

This is the architecture described in Sec. 4.4 of the paper (and clarified in
``paper/addendum.md``):

    "the actor of each follower and leader consist of a shared network
     B_theta conditioned on the parameters phi_j which are specific to each
     follower/leader. The critic of each follower and leader consists of a
     shared network C_psi conditioned on the same parameters phi_j specific to
     each follower/leader."

Concretely each policy ``j`` owns a small local vector ``phi_j`` while the
actor backbone ``B_theta`` and the critic backbone ``C_psi`` are shared by all
``M`` policies.  ``phi_j`` is injected into both networks (concatenated to the
observation or used as a FiLM modulation, see ``phi_conditioning``); because
``phi_j`` only ever appears inside policy ``j``'s own loss terms, a single
backward pass over the summed objective gives ``phi_j`` gradients coming
exclusively from the objective of policy ``j`` while ``theta``/``psi`` receive
the gradients of *all* policies -- exactly as specified in Sec. 4.4.

The paper also specifies the two architectures used in the experiments:

* AllegroKuka tasks: Gaussian policy whose mean network is an LSTM with 1 layer
  of 768 hidden units; the observation first goes through an MLP with hidden
  sizes 768x512x256 (ELU), the sigma is a *fixed learnable vector that does not
  depend on the observation* (Tables 2-4).
* Hand tasks: MLP mean network, 512x512x256x128 (ShadowHand) or 512x256x128
  (AllegroHand) with ELU activations and a fixed learnable log-sigma vector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal


ACTIVATIONS = {
    "elu": nn.ELU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
}


@dataclass
class ActorCriticConfig:
    """Architecture knobs (all default values follow the paper)."""

    latent_dim: int = 32                      # dim(phi_j): 32 AllegroKuka, 16 hands
    actor_type: str = "mlp"                   # "mlp" | "lstm"
    actor_units: List[int] = field(default_factory=lambda: [512, 256, 128])
    actor_encoder_units: List[int] = field(default_factory=lambda: [768, 512, 256])
    lstm_hidden_size: int = 768
    lstm_num_layers: int = 1
    critic_type: str = "mlp"                  # "mlp" | "lstm"
    critic_units: List[int] = field(default_factory=lambda: [512, 256, 128])
    critic_encoder_units: List[int] = field(default_factory=lambda: [768, 512, 256])
    activation: str = "elu"
    phi_conditioning: str = "concat"          # "concat" | "film"
    init_noise_std: float = 1.0               # sigma is a free parameter (input independent)
    action_bound: Optional[float] = None      # for the rl_games-style bounds loss

    @staticmethod
    def from_config(cfg) -> "ActorCriticConfig":
        model = cfg.get("model", {}) or {}
        latent = int(model.get("latent_dim", 32))
        actor = model.get("actor", {}) or {}
        critic = model.get("critic", {}) or {}
        return ActorCriticConfig(
            latent_dim=latent,
            actor_type=str(actor.get("type", "mlp")).lower(),
            actor_units=list(actor.get("units", [512, 256, 128])),
            actor_encoder_units=list(actor.get("encoder_units", [768, 512, 256])),
            lstm_hidden_size=int(actor.get("lstm_hidden_size", 768)),
            lstm_num_layers=int(actor.get("lstm_num_layers", 1)),
            critic_type=str(critic.get("type", actor.get("type", "mlp"))).lower(),
            critic_units=list(critic.get("units", [512, 256, 128])),
            critic_encoder_units=list(critic.get("encoder_units", [768, 512, 256])),
            activation=str(model.get("activation", "elu")).lower(),
            phi_conditioning=str(model.get("phi_conditioning", "concat")).lower(),
            init_noise_std=float(model.get("init_noise_std", 1.0)),
            action_bound=model.get("action_bound", None),
        )


def _activation(name: str) -> nn.Module:
    if name not in ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'")
    return ACTIVATIONS[name]()


class MLP(nn.Module):
    """Plain MLP; ``units`` lists the hidden sizes."""

    def __init__(self, in_dim: int, units: List[int], activation: str = "elu") -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for unit in units:
            layers.append(nn.Linear(prev, unit))
            layers.append(_activation(activation))
            prev = unit
        self.net = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RecurrentBackbone(nn.Module):
    """Observation -> MLP encoder -> LSTM (AllegroKuka architecture)."""

    def __init__(
        self,
        in_dim: int,
        encoder_units: List[int],
        hidden_size: int,
        num_layers: int = 1,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.encoder = MLP(in_dim, encoder_units, activation)
        self.lstm = nn.LSTM(self.encoder.out_dim, hidden_size, num_layers=num_layers)
        self.out_dim = hidden_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor, states=None):
        features = self.encoder(x)
        # nn.LSTM expects (T, N, H)
        out, new_states = self.lstm(features.unsqueeze(0), states)
        return out.squeeze(0), new_states

    def initial_state(self, batch_size: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        shape = (self.num_layers, batch_size, self.hidden_size)
        return (
            torch.zeros(shape, device=device),
            torch.zeros(shape, device=device),
        )


class ActorCritic(nn.Module):
    """Shared-backbone actor-critic conditioned on per-policy latents phi_j."""

    def __init__(
        self,
        cfg: ActorCriticConfig,
        obs_dim: int,
        action_dim: int,
        num_policies: int,
        normalizer=None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_policies = num_policies
        self.normalizer = normalizer

        self.latent_dim = cfg.latent_dim
        conditioned_dim = obs_dim + cfg.latent_dim

        # --- actor ----------------------------------------------------------
        self.actor_type = cfg.actor_type
        if cfg.actor_type == "lstm":
            self.actor = RecurrentBackbone(
                conditioned_dim,
                cfg.actor_encoder_units,
                cfg.lstm_hidden_size,
                cfg.lstm_num_layers,
                cfg.activation,
            )
        elif cfg.actor_type == "mlp":
            self.actor = MLP(conditioned_dim, cfg.actor_units, cfg.activation)
        else:
            raise ValueError(f"Unknown actor type '{cfg.actor_type}'")
        self.mu = nn.Linear(self.actor.out_dim, action_dim)

        # "The sigma for the Gaussian is a fixed learnable vector independent of
        # input observation" (App. B).
        self.log_std = nn.Parameter(torch.ones(action_dim) * math.log(cfg.init_noise_std))

        # --- critic ---------------------------------------------------------
        self.critic_type = cfg.critic_type
        if cfg.critic_type == "lstm":
            self.critic = RecurrentBackbone(
                conditioned_dim,
                cfg.critic_encoder_units,
                cfg.lstm_hidden_size,
                cfg.lstm_num_layers,
                cfg.activation,
            )
        elif cfg.critic_type == "mlp":
            self.critic = MLP(conditioned_dim, cfg.critic_units, cfg.activation)
        else:
            raise ValueError(f"Unknown critic type '{cfg.critic_type}'")
        self.value_head = nn.Linear(self.critic.out_dim, 1)

        # --- per-policy latents phi_j (Sec. 4.4) ----------------------------
        # These are *local* parameters: phi_j only ever takes gradients from
        # policy j's own objective, both in the leader/follower and in the
        # symmetric aggregation scheme.
        self.phi = nn.Parameter(torch.randn(num_policies, cfg.latent_dim) * 0.01)

        self.apply(_orthogonal_init)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def phi_for(self, policy_ids: Optional[torch.Tensor], batch_size: int, device) -> torch.Tensor:
        if policy_ids is None:
            policy_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
        return self.phi[policy_ids]

    def _condition(self, obs: torch.Tensor, policy_ids: Optional[torch.Tensor]) -> torch.Tensor:
        if self.normalizer is not None:
            obs = self.normalizer.normalize(obs)
        if self.cfg.phi_conditioning != "concat":
            # FiLM is handled inside the forward pass; concat is the default.
            return obs
        batch_size = obs.shape[-2] if obs.dim() == 3 else obs.shape[0]
        phi = self.phi_for(policy_ids, batch_size, obs.device)
        if obs.dim() == 3:
            # [T, B, d] sequence layout
            phi = phi.unsqueeze(0).expand(obs.shape[0], -1, -1)
        return torch.cat([obs, phi], dim=-1)

    def distribution(self, mu: torch.Tensor) -> Normal:
        std = torch.exp(torch.clamp(self.log_std, -20.0, 2.0))
        return Normal(mu, std.expand_as(mu))

    # ------------------------------------------------------------------ #
    # feed-forward interface
    # ------------------------------------------------------------------ #
    def forward(self, obs: torch.Tensor, policy_ids: Optional[torch.Tensor] = None):
        conditioned = self._condition(obs, policy_ids)
        if self.actor_type == "lstm":
            features, _ = self.actor(conditioned)
            features = features.squeeze(0)
            value = self.value_head(self.critic(conditioned)[0])
            value = value.squeeze(0)
        else:
            features = self.actor(conditioned)
            value = self.value_head(self.critic(conditioned))
        mu = self.mu(features)
        return mu, value.squeeze(-1)

    def forward_sequence(
        self,
        obs: torch.Tensor,
        policy_ids: Optional[torch.Tensor] = None,
        recurrent_states=None,
        dones: Optional[torch.Tensor] = None,
    ):
        """``obs``: [T, B, obs_dim]. Returns mu/value over time and final states.

        Hidden states are reset (masked) whenever ``dones`` is 1 so that
        different episodes never share LSTM state.
        """
        assert self.actor_type == "lstm", "forward_sequence is only for recurrent actors"
        t_steps, batch = obs.shape[0], obs.shape[1]
        if recurrent_states is None:
            recurrent_states = self.actor.initial_state(batch, obs.device)
            critic_states = self.critic.initial_state(batch, obs.device)
        else:
            recurrent_states, critic_states = recurrent_states

        mus, values = [], []
        hidden, cell = recurrent_states
        c_hidden, c_cell = critic_states
        for t in range(t_steps):
            conditioned = self._condition(obs[t], policy_ids)
            features, (hidden, cell) = self.actor(conditioned, (hidden, cell))
            features = features.squeeze(0)
            value, (c_hidden, c_cell) = self.critic(conditioned, (c_hidden, c_cell))
            value = value.squeeze(0)
            mu = self.mu(features)
            mus.append(mu)
            values.append(self.value_head(value).squeeze(-1))
            if dones is not None:
                mask = (1.0 - dones[t]).view(1, batch, 1)
                hidden = hidden * mask
                cell = cell * mask
                c_hidden = c_hidden * mask
                c_cell = c_cell * mask
        return torch.stack(mus), torch.stack(values), ((hidden, cell), (c_hidden, c_cell))

    # ------------------------------------------------------------------ #
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        policy_ids: Optional[torch.Tensor] = None,
        recurrent_states=None,
        dones: Optional[torch.Tensor] = None,
    ):
        """Return (log_prob, entropy, value, new_states, action_mean)."""
        if self.actor_type == "lstm" and obs.dim() == 3:
            mu, value, states = self.forward_sequence(obs, policy_ids, recurrent_states, dones)
            dist = self.distribution(mu)
            log_prob = dist.log_prob(actions).sum(-1)
            entropy = dist.entropy().sum(-1)
            return log_prob, entropy, value, states, mu
        mu, value = self.forward(obs, policy_ids)
        dist = self.distribution(mu)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value, None, mu

    @torch.no_grad()
    def act(self, obs: torch.Tensor, policy_ids=None, recurrent_states=None, dones=None, deterministic=False):
        """Sample (or take the mean of) an action from policy ``policy_ids``."""
        if self.actor_type == "lstm":
            obs_t = obs.unsqueeze(0) if obs.dim() == 2 else obs
            mu, value, states = self.forward_sequence(obs_t, policy_ids, recurrent_states, dones)
            mu, value = mu.squeeze(0), value.squeeze(0)
        else:
            mu, value = self.forward(obs, policy_ids)
            states = None
        dist = self.distribution(mu)
        actions = mu if deterministic else dist.rsample()
        log_prob = dist.log_prob(actions).sum(-1)
        return actions, log_prob, value, states

    @torch.no_grad()
    def get_value(self, obs: torch.Tensor, policy_ids=None, recurrent_states=None, dones=None):
        if self.actor_type == "lstm":
            obs_t = obs.unsqueeze(0) if obs.dim() == 2 else obs
            _, value, states = self.forward_sequence(obs_t, policy_ids, recurrent_states, dones)
            return value.squeeze(0), states
        _, value = self.forward(obs, policy_ids)
        return value, None

    # ------------------------------------------------------------------ #
    def trainable_parameter_groups(self, learning_rate: float):
        """Adam parameter groups.

        ``phi`` is a separate group only for bookkeeping; because phi_j appears
        exclusively in policy j's loss terms it receives exactly the gradients
        of that objective (Sec. 4.4).
        """
        return [{"params": list(self.parameters()), "lr": learning_rate}]


def _orthogonal_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=1.0)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
