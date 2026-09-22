"""Shared actor B_theta and critic C_psi with per-policy latent conditioning phi_j.

Implements Section 4.4 of the SAPG paper: M diverse policies (1 leader + M-1
followers) share a backbone network conditioned on per-policy latent parameters
phi_j.  The actor outputs a Gaussian mean; the standard deviation is a fixed
learnable, input-independent vector.  The critic is conditioned on the *same*
phi_j and outputs V(s).

Gradient routing (critical):
    * theta (actor backbone) and psi (critic backbone) receive gradients from
      ALL objectives.
    * phi_j receives gradients ONLY from its own policy's objective.  This is
      enforced by detaching phi_j in every loss path that does not belong to
      policy j (see ``detach_phi`` helper and the ``phi_index`` argument).

Architectures (from the reproduction plan / addendum):
    * AllegroKuka : MLP(768x512x256, ELU) -> LSTM(1 layer, 768 hidden) -> mean
                    sigma = fixed learnable vector, phi_j in R^32
    * ShadowHand  : MLP(512x512x256x128, ELU) -> mean, phi_j in R^16
    * AllegroHand : MLP(512x256x128, ELU) -> mean, phi_j in R^16
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------
def get_activation(name: str) -> nn.Module:
    """Return an activation module by name (ELU used throughout the paper)."""
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


# ---------------------------------------------------------------------------
# MLP builder
# ---------------------------------------------------------------------------
def build_mlp(
    in_dim: int,
    hidden_dims: Sequence[int],
    activation: str = "elu",
    out_dim: Optional[int] = None,
    out_activation: Optional[str] = None,
) -> nn.Sequential:
    """Build a plain MLP with the given hidden dimensions."""
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(get_activation(activation))
        prev = h
    if out_dim is not None:
        layers.append(nn.Linear(prev, out_dim))
        if out_activation is not None:
            layers.append(get_activation(out_activation))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Conditioning strategies
# ---------------------------------------------------------------------------
class ConcatConditioning(nn.Module):
    """Concatenate phi_j to the input features (default strategy)."""

    def forward(self, x: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, phi], dim=-1)


class FiLMConditioning(nn.Module):
    """Feature-wise linear modulation: gamma(phi) * x + beta(phi).

    Provided as an alternative conditioning strategy (paper unspecified).
    """

    def __init__(self, feature_dim: int, phi_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.to_gamma_beta = nn.Linear(phi_dim, 2 * feature_dim)
        # Initialise to identity transform (gamma=1, beta=0).
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, x: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        gb = self.to_gamma_beta(phi)
        gamma, beta = torch.chunk(gb, 2, dim=-1)
        return (1.0 + gamma) * x + beta


# ---------------------------------------------------------------------------
# Shared actor
# ---------------------------------------------------------------------------
class SharedActor(nn.Module):
    """Shared actor backbone B_theta conditioned on per-policy phi_j.

    The network outputs the mean of a diagonal Gaussian.  The log-standard
    deviation is a fixed, learnable, input-independent parameter vector (one
    per action dimension), shared across policies (it is part of theta).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        phi_dim: int,
        hidden_dims: Sequence[int],
        activation: str = "elu",
        use_lstm: bool = False,
        lstm_hidden: int = 768,
        lstm_layers: int = 1,
        conditioning: str = "concat",
        log_std_init: float = -0.5,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.phi_dim = phi_dim
        self.use_lstm = use_lstm
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.conditioning = conditioning
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Input dimension depends on conditioning strategy.
        if conditioning == "concat":
            mlp_in = obs_dim + phi_dim
        elif conditioning == "film":
            mlp_in = obs_dim
        else:
            raise ValueError(f"Unknown conditioning: {conditioning}")

        self.mlp = build_mlp(mlp_in, hidden_dims, activation=activation)

        if conditioning == "film":
            self.film = FiLMConditioning(hidden_dims[0], phi_dim)

        feat_dim = hidden_dims[-1]

        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=feat_dim,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
            )
            head_in = lstm_hidden
        else:
            self.lstm = None
            head_in = feat_dim

        self.mean_head = nn.Linear(head_in, act_dim)

        # Fixed learnable, input-independent log-std (part of theta).
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std_init)))

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2.0))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Small gain on the mean head for stable initial policy.
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)

    # -- feature extraction -------------------------------------------------
    def extract_features(self, obs: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Run the shared backbone up to (but not including) the LSTM/head."""
        if self.conditioning == "concat":
            x = torch.cat([obs, phi], dim=-1)
            x = self.mlp(x)
        else:  # film
            # Apply MLP layer-by-layer, injecting FiLM after the first linear.
            x = obs
            first_linear_done = False
            for layer in self.mlp:
                x = layer(x)
                if isinstance(layer, nn.Linear) and not first_linear_done:
                    x = self.film(x, phi)
                    first_linear_done = True
        return x

    def forward(
        self,
        obs: torch.Tensor,
        phi: torch.Tensor,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        seq_len: Optional[int] = None,
    ):
        """Compute the Gaussian mean.

        Args:
            obs: (B, obs_dim) or (B, T, obs_dim) when using LSTM.
            phi: (B, phi_dim) or (B, T, phi_dim).
            lstm_state: optional (h, c) tuple for the LSTM.
            seq_len: sequence length T when obs is 3-D.

        Returns:
            mean: (B, act_dim) or (B, T, act_dim)
            new_lstm_state: (h, c) or None
        """
        if self.use_lstm:
            if obs.dim() == 2:
                obs = obs.unsqueeze(1)
                phi = phi.unsqueeze(1)
                squeeze = True
            else:
                squeeze = False
            B, T, _ = obs.shape
            feats = self.extract_features(
                obs.reshape(B * T, -1), phi.reshape(B * T, -1)
            )
            feats = feats.reshape(B, T, -1)
            out, new_state = self.lstm(feats, lstm_state)
            mean = self.mean_head(out)
            if squeeze:
                mean = mean.squeeze(1)
            return mean, new_state
        else:
            feats = self.extract_features(obs, phi)
            mean = self.mean_head(feats)
            return mean, None

    # -- distribution helpers ----------------------------------------------
    def std(self) -> torch.Tensor:
        return torch.exp(self.log_std.clamp(self.log_std_min, self.log_std_max))

    def distribution(
        self,
        obs: torch.Tensor,
        phi: torch.Tensor,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """Return a torch Normal distribution over actions."""
        mean, new_state = self.forward(obs, phi, lstm_state=lstm_state)
        std = self.std().expand_as(mean)
        return torch.distributions.Normal(mean, std), new_state

    def act(
        self,
        obs: torch.Tensor,
        phi: torch.Tensor,
        deterministic: bool = False,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """Sample (or take the mean of) an action; return action, logprob, state."""
        dist, new_state = self.distribution(obs, phi, lstm_state=lstm_state)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, new_state

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        phi: torch.Tensor,
        actions: torch.Tensor,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """Return log-prob and entropy of the given actions under this policy."""
        dist, new_state = self.distribution(obs, phi, lstm_state=lstm_state)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy, new_state


# ---------------------------------------------------------------------------
# Shared critic
# ---------------------------------------------------------------------------
class SharedCritic(nn.Module):
    """Shared critic C_psi conditioned on the same phi_j as the actor."""

    def __init__(
        self,
        obs_dim: int,
        phi_dim: int,
        hidden_dims: Sequence[int],
        activation: str = "elu",
        use_lstm: bool = False,
        lstm_hidden: int = 768,
        lstm_layers: int = 1,
        conditioning: str = "concat",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.phi_dim = phi_dim
        self.use_lstm = use_lstm
        self.conditioning = conditioning

        if conditioning == "concat":
            mlp_in = obs_dim + phi_dim
        elif conditioning == "film":
            mlp_in = obs_dim
        else:
            raise ValueError(f"Unknown conditioning: {conditioning}")

        self.mlp = build_mlp(mlp_in, hidden_dims, activation=activation)

        if conditioning == "film":
            self.film = FiLMConditioning(hidden_dims[0], phi_dim)

        feat_dim = hidden_dims[-1]
        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=feat_dim,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
            )
            head_in = lstm_hidden
        else:
            self.lstm = None
            head_in = feat_dim

        self.value_head = nn.Linear(head_in, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2.0))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def extract_features(self, obs: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        if self.conditioning == "concat":
            x = torch.cat([obs, phi], dim=-1)
            x = self.mlp(x)
        else:
            x = obs
            first_linear_done = False
            for layer in self.mlp:
                x = layer(x)
                if isinstance(layer, nn.Linear) and not first_linear_done:
                    x = self.film(x, phi)
                    first_linear_done = True
        return x

    def forward(
        self,
        obs: torch.Tensor,
        phi: torch.Tensor,
        lstm_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        if self.use_lstm:
            if obs.dim() == 2:
                obs = obs.unsqueeze(1)
                phi = phi.unsqueeze(1)
                squeeze = True
            else:
                squeeze = False
            B, T, _ = obs.shape
            feats = self.extract_features(
                obs.reshape(B * T, -1), phi.reshape(B * T, -1)
            )
            feats = feats.reshape(B, T, -1)
            out, new_state = self.lstm(feats, lstm_state)
            value = self.value_head(out)
            if squeeze:
                value = value.squeeze(1)
            return value.squeeze(-1), new_state
        else:
            feats = self.extract_features(obs, phi)
            value = self.value_head(feats)
            return value.squeeze(-1), None


# ---------------------------------------------------------------------------
# Multi-policy container
# ---------------------------------------------------------------------------
class MultiPolicyActorCritic(nn.Module):
    """Container holding the shared actor/critic and the M latent vectors phi_j.

    The M policies share theta (actor backbone) and psi (critic backbone) but
    each has its own latent parameter vector phi_j.  ``phi_j`` is a leaf
    parameter that only receives gradients from policy j's objective.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        num_policies: int,
        phi_dim: int,
        actor_hidden_dims: Sequence[int],
        critic_hidden_dims: Sequence[int],
        activation: str = "elu",
        use_lstm: bool = False,
        lstm_hidden: int = 768,
        lstm_layers: int = 1,
        conditioning: str = "concat",
        log_std_init: float = -0.5,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_policies = num_policies
        self.phi_dim = phi_dim

        self.actor = SharedActor(
            obs_dim=obs_dim,
            act_dim=act_dim,
            phi_dim=phi_dim,
            hidden_dims=actor_hidden_dims,
            activation=activation,
            use_lstm=use_lstm,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            conditioning=conditioning,
            log_std_init=log_std_init,
        )
        self.critic = SharedCritic(
            obs_dim=obs_dim,
            phi_dim=phi_dim,
            hidden_dims=critic_hidden_dims,
            activation=activation,
            use_lstm=use_lstm,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            conditioning=conditioning,
        )

        # Per-policy latent parameters phi_j, shape (M, phi_dim).
        self.phi = nn.Parameter(torch.zeros(num_policies, phi_dim))
        nn.init.normal_(self.phi, mean=0.0, std=0.1)

    # -- phi access ---------------------------------------------------------
    def get_phi(self, policy_index: int, batch_size: int) -> torch.Tensor:
        """Return phi_j expanded to (batch_size, phi_dim)."""
        phi = self.phi[policy_index]
        return phi.unsqueeze(0).expand(batch_size, -1)

    def get_phi_detached(self, policy_index: int, batch_size: int) -> torch.Tensor:
        """Return a detached phi_j (used in other policies' loss paths)."""
        return self.get_phi(policy_index, batch_size).detach()

    # -- convenience wrappers ----------------------------------------------
    def act(self, obs, policy_index, deterministic=False, lstm_state=None):
        phi = self.get_phi(policy_index, obs.shape[0])
        return self.actor.act(obs, phi, deterministic=deterministic, lstm_state=lstm_state)

    def evaluate_actions(self, obs, policy_index, actions, lstm_state=None):
        phi = self.get_phi(policy_index, obs.shape[0])
        return self.actor.evaluate_actions(obs, phi, actions, lstm_state=lstm_state)

    def value(self, obs, policy_index, lstm_state=None):
        phi = self.get_phi(policy_index, obs.shape[0])
        return self.critic(obs, phi, lstm_state=lstm_state)

    def actor_parameters(self):
        return self.actor.parameters()

    def critic_parameters(self):
        return self.critic.parameters()

    def phi_parameters(self):
        return [self.phi]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
# Architecture presets keyed by task name (from the reproduction plan).
ARCHITECTURE_PRESETS = {
    "allegrokuka": dict(
        actor_hidden_dims=(768, 512, 256),
        critic_hidden_dims=(768, 512, 256),
        phi_dim=32,
        use_lstm=True,
        lstm_hidden=768,
        lstm_layers=1,
    ),
    "shadowhand": dict(
        actor_hidden_dims=(512, 512, 256, 128),
        critic_hidden_dims=(512, 512, 256, 128),
        phi_dim=16,
        use_lstm=False,
    ),
    "allegrohand": dict(
        actor_hidden_dims=(512, 256, 128),
        critic_hidden_dims=(512, 256, 128),
        phi_dim=16,
        use_lstm=False,
    ),
}


def build_actor_critic(
    task: str,
    obs_dim: int,
    act_dim: int,
    num_policies: int,
    activation: str = "elu",
    conditioning: str = "concat",
    log_std_init: float = -0.5,
    **overrides,
) -> MultiPolicyActorCritic:
    """Build a MultiPolicyActorCritic using the preset for ``task``."""
    key = task.lower().replace("_", "").replace("-", "")
    if key not in ARCHITECTURE_PRESETS:
        raise ValueError(
            f"Unknown task '{task}'. Available: {list(ARCHITECTURE_PRESETS)}"
        )
    cfg = dict(ARCHITECTURE_PRESETS[key])
    cfg.update(overrides)
    return MultiPolicyActorCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        num_policies=num_policies,
        activation=activation,
        conditioning=conditioning,
        log_std_init=log_std_init,
        **cfg,
    )
