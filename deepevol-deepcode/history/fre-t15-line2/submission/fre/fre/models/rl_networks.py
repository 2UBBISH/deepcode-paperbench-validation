"""z-conditioned RL networks for the FRE agent (Algorithm 1, phase 2).

Paper: "Zero-Shot Reinforcement Learning via Functional Reward Encodings".

Relevant verbatim specifications
--------------------------------
Section 4.3 (Offline RL with FRE):

    "The latent representation z can then be used for RL training. The RL
    components (Q-function, value function, and policy) are all conditioned on
    z. [...] A standard Bellman policy improvement step using FRE looks like:

        Q(s, a, z) <- eta(s) + E_{s' ~ p(s'|s,a)} [ max_{a' in A} Q(s', a', z) ]

    [...] we use implicit Q-learning (Kostrikov et al., 2021) as the offline RL
    method to train our FRE-conditioned policy."

Addendum ("Additional Details on the FRE architecture"):

    "For conditioning the RL components (value, critic, etc.) of the FRE-agent
    with the latent embedding z, the latent embedding is simply concatenated to
    the observation state that is fed into the RL components."

Table 3 (A. Hyperparameters):

    RL Network Layers   [512, 512, 512]
    Target Update Rate  0.001
    Discount Factor     0.88
    AWR Temperature     3.0
    IQL Expectile       0.8
    Optimizer           Adam
    Learning Rate       0.0001
    Batch Size          512

Not specified in the paper (documented defaults used here)
---------------------------------------------------------
* Twin Q-functions: IQL (Kostrikov et al., 2021) uses twin critics / min-Q
  targets; the paper only writes ``Q(s, a, z)``.  We therefore parameterise a
  ``QNetwork`` with ``num_qs`` (default 2) and take the min for the TD target.
* Policy distribution: the reference IQL implementation uses a tanh-squashed
  Gaussian with a state-independent log-std head; we follow that (the paper
  only says the policy is ``pi(a | s, z)``).
* Activation: ReLU (the paper does not state one).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "QNetwork",
    "ValueNetwork",
    "PolicyNetwork",
    "RLNetworks",
    "PolicyOutput",
    "make_rl_networks",
    "build_mlp",
    "DEFAULT_RL_LAYERS",
    "DEFAULT_LATENT_DIM",
    "DEFAULT_NUM_Q_FUNCTIONS",
    "DEFAULT_DISCOUNT",
    "DEFAULT_TAU",
    "DEFAULT_EXPECTILE",
    "DEFAULT_AWR_TEMPERATURE",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_BATCH_SIZE",
]

# ----------------------------------------------------------------------------
# Hyper-parameters (Table 3).  These constants are the single source of truth
# for the FRE-conditioned RL stack; ``fre.training.iql`` and the entry-point
# scripts import them so the numbers stay in sync with the paper.
# ----------------------------------------------------------------------------
DEFAULT_RL_LAYERS: Tuple[int, int, int] = (512, 512, 512)
DEFAULT_LATENT_DIM: int = 128
DEFAULT_NUM_Q_FUNCTIONS: int = 2  # twin critics (IQL reference implementation)
DEFAULT_DISCOUNT: float = 0.88
DEFAULT_TAU: float = 0.001
DEFAULT_EXPECTILE: float = 0.8
DEFAULT_AWR_TEMPERATURE: float = 3.0
DEFAULT_LEARNING_RATE: float = 1e-4
DEFAULT_BATCH_SIZE: int = 512
DEFAULT_LOG_STD_MIN: float = -5.0
DEFAULT_LOG_STD_MAX: float = 2.0
DEFAULT_ACTIVATION: str = "relu"
DEFAULT_EPS_CLIP: float = 1e-6


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def _activation_module(name: str) -> nn.Module:
    name = (name or "relu").lower()
    if name == "relu":
        return nn.ReLU()
    if name in ("gelu",):
        return nn.GELU()
    if name in ("tanh",):
        return nn.Tanh()
    if name in ("silu", "swish"):
        return nn.SiLU()
    if name in ("elu",):
        return nn.ELU()
    raise ValueError(f"Unsupported activation: {name!r}")


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: str = DEFAULT_ACTIVATION,
    layer_norm: bool = False,
    output_activation: Optional[str] = None,
    final_layer_gain: float = 1.0,
) -> nn.Sequential:
    """Build a plain MLP.

    ``input_dim -> hidden_dims[0] -> ... -> hidden_dims[-1] -> output_dim``
    with an activation after every hidden layer (and optionally after the
    output layer).  With the Table 3 setting this is
    ``[state_dim + latent_dim] -> 512 -> 512 -> 512 -> output_dim``.
    """
    layers: List[nn.Module] = []
    prev = int(input_dim)
    for hidden in hidden_dims:
        linear = nn.Linear(prev, int(hidden))
        if final_layer_gain != 1.0:
            with torch.no_grad():
                linear.weight.mul_(final_layer_gain)
                linear.bias.mul_(final_layer_gain)
        layers.append(linear)
        if layer_norm:
            layers.append(nn.LayerNorm(int(hidden)))
        layers.append(_activation_module(activation))
        prev = int(hidden)
    out = nn.Linear(prev, int(output_dim))
    if final_layer_gain != 1.0:
        with torch.no_grad():
            out.weight.mul_(final_layer_gain)
            out.bias.mul_(final_layer_gain)
    layers.append(out)
    if output_activation is not None:
        layers.append(_activation_module(output_activation))
    return nn.Sequential(*layers)


def _as_float_tensor(x, device=None, dtype=torch.float32) -> torch.Tensor:
    """Coerce numpy / list / tensor input into a float tensor."""
    if isinstance(x, torch.Tensor):
        t = x
        if dtype is not None and t.dtype != dtype and not t.dtype.is_complex:
            t = t.to(dtype)
    else:
        t = torch.as_tensor(np.asarray(x), dtype=dtype)
    if device is not None:
        t = t.to(device)
    return t


def _as_long_tensor(x, device=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x.long()
    else:
        t = torch.as_tensor(np.asarray(x), dtype=torch.long)
    if device is not None:
        t = t.to(device)
    return t


def _expand_latent(z: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """Broadcast ``z`` so it can be concatenated with ``obs``.

    Accepts ``z`` of shape ``(B, D)`` or ``(D,)`` and ``obs`` of shape
    ``(B, obs_dim)``.  Returns ``z`` of shape ``(B, D)``.
    """
    if z.dim() == 1:
        z = z.unsqueeze(0)
    if z.shape[0] == 1 and obs.shape[0] != 1:
        z = z.expand(obs.shape[0], -1)
    if z.shape[0] != obs.shape[0]:
        raise ValueError(
            f"latent batch {z.shape[0]} does not match observation batch {obs.shape[0]}"
        )
    return z


def _concat_state_latent(obs, z) -> torch.Tensor:
    """``concat(s, z)`` -- the addendum's conditioning rule for RL components."""
    obs_t = _as_float_tensor(obs)
    if obs_t.dim() == 1:
        obs_t = obs_t.unsqueeze(0)
    z_t = _as_float_tensor(z, device=obs_t.device)
    z_t = _expand_latent(z_t, obs_t)
    return torch.cat([obs_t, z_t], dim=-1)


def _squash_correction(pre_tanh: torch.Tensor) -> torch.Tensor:
    """log(1 - tanh(x)^2) with a numerical floor (per-element)."""
    return torch.log1p(-torch.tanh(pre_tanh) ** 2 + DEFAULT_EPS_CLIP * 0.0 + 1e-6)


# ----------------------------------------------------------------------------
# Q network
# ----------------------------------------------------------------------------
class QNetwork(nn.Module):
    """Action-value function ``Q(s, a, z)``.

    Input is ``concat(s, z)`` concatenated with the action (the addendum
    specifies concatenating ``z`` to the *observation*; the action is appended
    to form the critic input, exactly as in the reference IQL implementation).

    Parameters
    ----------
    obs_dim:
        Dimensionality of the (already preprocessed) observation.
    act_dim:
        Dimensionality of the continuous action.
    latent_dim:
        128 by default (the FRE latent ``z``).
    hidden_dims:
        Table 3: ``[512, 512, 512]``.
    num_qs:
        1 for a single critic, 2 for the IQL twin-critic ensemble.  The paper
        does not specify this, so a documented default of 2 is used.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        latent_dim: int = DEFAULT_LATENT_DIM,
        hidden_dims: Sequence[int] = DEFAULT_RL_LAYERS,
        activation: str = DEFAULT_ACTIVATION,
        layer_norm: bool = False,
        num_qs: int = DEFAULT_NUM_Q_FUNCTIONS,
        state_norm: bool = False,
        state_mean=None,
        state_std=None,
        name: str = "q_network",
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.num_qs = int(num_qs)
        self.name = name
        # The observation fed here is ``s`` (not concatenated) plus ``z``; the
        # Bellman step and the policy improvement both use ``Q(s, a, z)``.
        self.input_dim = self.obs_dim + self.act_dim + self.latent_dim
        self.net = build_mlp(
            self.input_dim,
            self.hidden_dims,
            self.num_qs,
            activation=activation,
            layer_norm=layer_norm,
        )
        self.state_norm = bool(state_norm)
        self.register_buffer(
            "state_mean",
            torch.zeros(self.obs_dim) if state_mean is None else _as_float_tensor(state_mean).reshape(-1),
            persistent=False,
        )
        self.register_buffer(
            "state_std",
            torch.ones(self.obs_dim) if state_std is None else _as_float_tensor(state_std).reshape(-1),
            persistent=False,
        )

    # -- input handling -----------------------------------------------------
    def _prepare(self, obs, action, z):
        obs_t = _as_float_tensor(obs)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        act_t = _as_float_tensor(action, device=obs_t.device)
        if act_t.dim() == 1:
            act_t = act_t.unsqueeze(0)
        if self.state_norm:
            obs_t = (obs_t - self.state_mean) / (self.state_std + 1e-6)
        z_t = _as_float_tensor(z, device=obs_t.device)
        z_t = _expand_latent(z_t, obs_t)
        return torch.cat([obs_t, act_t, z_t], dim=-1)

    # -- forward ------------------------------------------------------------
    def forward(self, obs, action, z, return_all: bool = False):
        """Return ``(B,)`` Q values (min over critics unless ``return_all``).

        With ``return_all=True`` a ``(num_qs, B)`` tensor is returned.
        """
        x = self._prepare(obs, action, z)
        out = self.net(x)
        if out.dim() == 1:
            out = out.unsqueeze(-1)
        if self.num_qs == 1:
            q = out[:, 0]
            return out[:, 0] if return_all else q
        out = out.transpose(0, 1)  # (num_qs, B)
        if return_all:
            return out
        return out.min(dim=0).values

    # -- convenience --------------------------------------------------------
    def q1(self, obs, action, z) -> torch.Tensor:
        """First critic only (used for the actor's policy gradient/Q loss)."""
        out = self.forward(obs, action, z, return_all=True)
        if out.dim() == 1:
            return out
        return out[0]

    def q2(self, obs, action, z) -> torch.Tensor:
        """Second critic only (requires ``num_qs >= 2``)."""
        if self.num_qs < 2:
            raise ValueError("q2() requires num_qs >= 2")
        out = self.forward(obs, action, z, return_all=True)
        return out[1]

    def min_q(self, obs, action, z) -> torch.Tensor:
        """``min_i Q_i(s, a, z)`` -- the IQL TD-target critic."""
        return self.forward(obs, action, z)

    def q_all(self, obs, action, z) -> torch.Tensor:
        """Stacked ``(num_qs, B)`` critic outputs."""
        out = self.forward(obs, action, z, return_all=True)
        if self.num_qs == 1:
            return out.unsqueeze(0)
        return out

    def describe(self) -> Dict:
        return {
            "name": self.name,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "latent_dim": self.latent_dim,
            "hidden_dims": list(self.hidden_dims),
            "num_qs": self.num_qs,
            "input_dim": self.input_dim,
        }


# ----------------------------------------------------------------------------
# Value network
# ----------------------------------------------------------------------------
class ValueNetwork(nn.Module):
    """State-value function ``V(s, z)`` used by the IQL expectile regression.

    ``z`` is concatenated to the observation (addendum), giving an input of
    ``obs_dim + latent_dim`` through ``[512, 512, 512]`` and a scalar output.
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int = DEFAULT_LATENT_DIM,
        hidden_dims: Sequence[int] = DEFAULT_RL_LAYERS,
        activation: str = DEFAULT_ACTIVATION,
        layer_norm: bool = False,
        state_norm: bool = False,
        state_mean=None,
        state_std=None,
        name: str = "value_network",
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.name = name
        self.input_dim = self.obs_dim + self.latent_dim
        self.net = build_mlp(
            self.input_dim,
            self.hidden_dims,
            1,
            activation=activation,
            layer_norm=layer_norm,
        )
        self.state_norm = bool(state_norm)
        self.register_buffer(
            "state_mean",
            torch.zeros(self.obs_dim) if state_mean is None else _as_float_tensor(state_mean).reshape(-1),
            persistent=False,
        )
        self.register_buffer(
            "state_std",
            torch.ones(self.obs_dim) if state_std is None else _as_float_tensor(state_std).reshape(-1),
            persistent=False,
        )

    def forward(self, obs, z) -> torch.Tensor:
        obs_t = _as_float_tensor(obs)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        if self.state_norm:
            obs_t = (obs_t - self.state_mean) / (self.state_std + 1e-6)
        z_t = _as_float_tensor(z, device=obs_t.device)
        z_t = _expand_latent(z_t, obs_t)
        out = self.net(torch.cat([obs_t, z_t], dim=-1))
        return out.squeeze(-1)

    def describe(self) -> Dict:
        return {
            "name": self.name,
            "obs_dim": self.obs_dim,
            "latent_dim": self.latent_dim,
            "hidden_dims": list(self.hidden_dims),
            "input_dim": self.input_dim,
        }


# ----------------------------------------------------------------------------
# Policy network
# ----------------------------------------------------------------------------
@dataclass
class PolicyOutput:
    """Container for a policy forward pass (see :class:`PolicyNetwork`)."""

    mean: torch.Tensor            # tanh-transformed mean action (B, act_dim)
    action: torch.Tensor          # sampled (or deterministic) action (B, act_dim)
    log_prob: Optional[torch.Tensor] = None   # (B,) summed over action dims
    pre_tanh_mean: Optional[torch.Tensor] = None
    log_std: Optional[torch.Tensor] = None
    std: Optional[torch.Tensor] = None
    squashed: bool = True


class PolicyNetwork(nn.Module):
    """Gaussian policy ``pi(a | s, z)``.

    The observation is conditioned by concatenating ``z`` (addendum).  The MLP
    has the Table 3 width ``[512, 512, 512]`` and two linear heads: a mean head
    and a log-standard-deviation head.  By default the Gaussian is
    tanh-squashed onto the action support used by the reference IQL
    implementation (the paper only states ``pi(a | s, z)``).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        latent_dim: int = DEFAULT_LATENT_DIM,
        hidden_dims: Sequence[int] = DEFAULT_RL_LAYERS,
        activation: str = DEFAULT_ACTIVATION,
        layer_norm: bool = False,
        tanh_squash: bool = True,
        log_std_min: float = DEFAULT_LOG_STD_MIN,
        log_std_max: float = DEFAULT_LOG_STD_MAX,
        state_norm: bool = False,
        state_mean=None,
        state_std=None,
        name: str = "policy_network",
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.tanh_squash = bool(tanh_squash)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.name = name
        self.input_dim = self.obs_dim + self.latent_dim
        self.trunk = build_mlp(
            self.input_dim,
            self.hidden_dims,
            self.hidden_dims[-1] if self.hidden_dims else 64,
            activation=activation,
            layer_norm=layer_norm,
        )
        trunk_out = int(self.hidden_dims[-1]) if self.hidden_dims else 64
        self.mean_head = nn.Linear(trunk_out, self.act_dim)
        self.log_std_head = nn.Linear(trunk_out, self.act_dim)
        self.state_norm = bool(state_norm)
        self.register_buffer(
            "state_mean",
            torch.zeros(self.obs_dim) if state_mean is None else _as_float_tensor(state_mean).reshape(-1),
            persistent=False,
        )
        self.register_buffer(
            "state_std",
            torch.ones(self.obs_dim) if state_std is None else _as_float_tensor(state_std).reshape(-1),
            persistent=False,
        )

    # -- internals ----------------------------------------------------------
    def _features(self, obs, z) -> torch.Tensor:
        obs_t = _as_float_tensor(obs)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        if self.state_norm:
            obs_t = (obs_t - self.state_mean) / (self.state_std + 1e-6)
        z_t = _as_float_tensor(z, device=obs_t.device)
        z_t = _expand_latent(z_t, obs_t)
        return self.trunk(torch.cat([obs_t, z_t], dim=-1))

    def distribution_params(self, obs, z) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mean, std)`` of the *pre-squash* Gaussian."""
        h = self._features(obs, z)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        return mean, torch.exp(log_std)

    # -- forward ------------------------------------------------------------
    def forward(
        self,
        obs,
        z,
        sample: bool = True,
        deterministic: bool = False,
        return_log_prob: bool = True,
    ) -> PolicyOutput:
        h = self._features(obs, z)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        if deterministic or not sample:
            pre_tanh = mean
        else:
            pre_tanh = mean + std * torch.randn_like(std)

        if self.tanh_squash:
            action = torch.tanh(pre_tanh)
        else:
            action = pre_tanh

        log_prob = None
        if return_log_prob:
            dist = torch.distributions.Normal(mean, std)
            log_prob = dist.log_prob(pre_tanh)
            if self.tanh_squash:
                # Change of variables for the tanh squashing (SAC/IQL).
                log_prob = log_prob - torch.log(1.0 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1)

        mean_action = torch.tanh(mean) if self.tanh_squash else mean
        return PolicyOutput(
            mean=mean_action,
            action=action,
            log_prob=log_prob,
            pre_tanh_mean=pre_tanh,
            log_std=log_std,
            std=std,
            squashed=self.tanh_squash,
        )

    # -- convenience --------------------------------------------------------
    def act(self, obs, z, deterministic: bool = False) -> torch.Tensor:
        """Return an action tensor of shape ``(B, act_dim)``."""
        with torch.no_grad():
            return self.forward(
                obs, z, sample=not deterministic, deterministic=deterministic,
                return_log_prob=False,
            ).action

    def log_prob(self, obs, z, action) -> torch.Tensor:
        """``log pi(a | s, z)`` for given actions, summed over action dims."""
        h = self._features(obs, z)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        act_t = _as_float_tensor(action, device=mean.device)
        if act_t.dim() == 1:
            act_t = act_t.unsqueeze(0)
        dist = torch.distributions.Normal(mean, std)
        if self.tanh_squash:
            act_t = act_t.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            pre_tanh = torch.atanh(act_t)
            lp = dist.log_prob(pre_tanh) - torch.log(1.0 - act_t.pow(2) + 1e-6)
        else:
            lp = dist.log_prob(act_t)
        return lp.sum(dim=-1)

    def describe(self) -> Dict:
        return {
            "name": self.name,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "latent_dim": self.latent_dim,
            "hidden_dims": list(self.hidden_dims),
            "input_dim": self.input_dim,
            "tanh_squash": self.tanh_squash,
            "log_std_bounds": [self.log_std_min, self.log_std_max],
        }


# ----------------------------------------------------------------------------
# Container: everything the IQL trainer needs, plus polyak targets
# ----------------------------------------------------------------------------
class RLNetworks(nn.Module):
    """Bundle of the ``z``-conditioned IQL components: ``Q(s,a,z)``, ``V(s,z)``,
    ``pi(a|s,z)`` and the target value network.

    The container also optionally holds a reference to the (frozen) FRE encoder
    so that the policy-training loop can map a sampled prior reward function
    into a latent ``z`` before the RL update -- Algorithm 1, phase 2:

        Sample reward function eta ~ p(eta)
        Sample K states for encoder {s^e_k} ~ D
        Encode into latent vector z ~ p_theta({(s^e_k, eta(s^e_k))})
        Train pi(a|s,z), Q(s,a,z), V(s,z) using IQL with r = eta(s)
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        latent_dim: int = DEFAULT_LATENT_DIM,
        hidden_dims: Sequence[int] = DEFAULT_RL_LAYERS,
        activation: str = DEFAULT_ACTIVATION,
        layer_norm: bool = False,
        num_qs: int = DEFAULT_NUM_Q_FUNCTIONS,
        tanh_squash: bool = True,
        discount: float = DEFAULT_DISCOUNT,
        tau: float = DEFAULT_TAU,
        expectile: float = DEFAULT_EXPECTILE,
        awr_temperature: float = DEFAULT_AWR_TEMPERATURE,
        state_norm: bool = False,
        state_mean=None,
        state_std=None,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.discount = float(discount)                # Table 3: 0.88
        self.tau = float(tau)                          # Table 3: 0.001
        self.expectile = float(expectile)              # Table 3: 0.8
        self.awr_temperature = float(awr_temperature)  # Table 3: 3.0

        self.q = QNetwork(
            self.obs_dim,
            self.act_dim,
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims,
            activation=activation,
            layer_norm=layer_norm,
            num_qs=num_qs,
            state_norm=state_norm,
            state_mean=state_mean,
            state_std=state_std,
        )
        self.v = ValueNetwork(
            self.obs_dim,
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims,
            activation=activation,
            layer_norm=layer_norm,
            state_norm=state_norm,
            state_mean=state_mean,
            state_std=state_std,
        )
        self.policy = PolicyNetwork(
            self.obs_dim,
            self.act_dim,
            latent_dim=self.latent_dim,
            hidden_dims=self.hidden_dims,
            activation=activation,
            layer_norm=layer_norm,
            tanh_squash=tanh_squash,
            state_norm=state_norm,
            state_mean=state_mean,
            state_std=state_std,
        )

        # Target value network (target update rate 0.001, Table 3).
        self.target_v = copy.deepcopy(self.v)
        self.target_v.eval()
        for p in self.target_v.parameters():
            p.requires_grad_(False)

        # Optional frozen encoder reference (attached in phase 2 only).
        self.encoder: Optional[nn.Module] = None

    # -- encoder plumbing ---------------------------------------------------
    def attach_encoder(self, encoder: nn.Module, freeze: bool = True) -> None:
        """Attach the FRE encoder used to produce ``z`` during RL training.

        The strided scheme (Section 4.3) requires the encoder to be frozen
        before the RL phase starts, so ``freeze=True`` (the default) sets
        ``requires_grad_(False)`` on every encoder parameter.
        """
        self.encoder = encoder
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            self.encoder.eval()

    def detach_encoder(self) -> None:
        self.encoder = None

    def encode_context(
        self,
        states,
        rewards,
        sample: bool = False,
        reward_min: Optional[float] = None,
        reward_max: Optional[float] = None,
    ) -> torch.Tensor:
        """Encode ``K`` reward-labelled states into a latent ``z`` (frozen).

        ``sample=False`` (posterior mean) is the documented evaluation default;
        ``sample=True`` realises ``z ~ p_theta(z | {(s^e_k, eta(s^e_k))})`` from
        Algorithm 1's policy-training loop.
        """
        if self.encoder is None:
            raise RuntimeError("No encoder attached; call attach_encoder() first.")
        states_t = _as_float_tensor(states)
        rewards_t = _as_float_tensor(rewards, device=states_t.device)
        with torch.no_grad():
            out = self.encoder(
                states_t, rewards_t, sample=sample,
                reward_min=reward_min, reward_max=reward_max,
            )
        return out.z.detach()

    # -- value / critic queries --------------------------------------------
    def values(self, obs, z) -> torch.Tensor:
        return self.v(obs, z)

    def target_values(self, obs, z) -> torch.Tensor:
        with torch.no_grad():
            return self.target_v(obs, z)

    def q_values(self, obs, action, z, return_all: bool = False):
        return self.q(obs, action, z, return_all=return_all)

    def min_q(self, obs, action, z) -> torch.Tensor:
        return self.q(obs, action, z)

    def acts(self, obs, z, deterministic: bool = False) -> torch.Tensor:
        return self.policy.act(obs, z, deterministic=deterministic)

    def log_prob(self, obs, z, action) -> torch.Tensor:
        return self.policy.log_prob(obs, z, action)

    # -- target updates -----------------------------------------------------
    @torch.no_grad()
    def update_target(self, tau: Optional[float] = None, hard: bool = False) -> None:
        """Soft (``tau``) or hard update of the target value network.

        Table 3 "Target Update Rate" = 0.001.
        """
        tau = self.tau if tau is None else float(tau)
        for tp, sp in zip(self.target_v.parameters(), self.v.parameters()):
            if hard:
                tp.data.copy_(sp.data)
            else:
                tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)
        for tb, sb in zip(self.target_v.buffers(), self.v.buffers()):
            if tb.shape == sb.shape and tb.dtype == sb.dtype:
                tb.data.copy_(sb.data)

    def sync_target(self) -> None:
        """Hard copy of the value network into the target network."""
        self.update_target(hard=True)

    # -- parameter management ----------------------------------------------
    def nets(self) -> Dict[str, nn.Module]:
        """The three trainable RL networks (target V is not trained)."""
        return {"q": self.q, "v": self.v, "policy": self.policy}

    def trainable_parameters(self) -> List[nn.Parameter]:
        """All RL parameters that should go into the optimizer."""
        params: List[nn.Parameter] = []
        for net in (self.q, self.v, self.policy):
            params.extend([p for p in net.parameters() if p.requires_grad])
        return params

    def named_trainable_parameters(self) -> Iterable[Tuple[str, nn.Parameter]]:
        for name, net in self.nets().items():
            for pname, p in net.named_parameters():
                if p.requires_grad:
                    yield f"{name}.{pname}", p

    def set_train_mode(self) -> None:
        self.q.train()
        self.v.train()
        self.policy.train()
        self.target_v.eval()

    def set_eval_mode(self) -> None:
        self.eval()

    def assert_encoder_frozen(self) -> bool:
        """Correctness gate: no encoder parameter may require grad in phase 2."""
        if self.encoder is None:
            return True
        return not any(p.requires_grad for p in self.encoder.parameters())

    def describe(self) -> Dict:
        return {
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "latent_dim": self.latent_dim,
            "hidden_dims": list(self.hidden_dims),
            "num_qs": self.q.num_qs,
            "discount": self.discount,
            "tau": self.tau,
            "expectile": self.expectile,
            "awr_temperature": self.awr_temperature,
            "encoder_attached": self.encoder is not None,
            "encoder_frozen": self.assert_encoder_frozen(),
        }


# ----------------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------------
def make_rl_networks(
    obs_dim: int,
    act_dim: int,
    latent_dim: int = DEFAULT_LATENT_DIM,
    hidden_dims: Sequence[int] = DEFAULT_RL_LAYERS,
    num_qs: int = DEFAULT_NUM_Q_FUNCTIONS,
    encoder: Optional[nn.Module] = None,
    freeze_encoder: bool = True,
    **kwargs,
) -> RLNetworks:
    """Build the FRE-conditioned IQL networks with the paper's hyper-parameters.

    Parameters mirror Table 3's "RL Network Layers = [512, 512, 512]"; the
    latent is the 128-dimensional FRE ``z``.  Values for discount (0.88), tau
    (0.001), expectile (0.8) and AWR temperature (3.0) default to the table's
    numbers and are stored on the returned container.
    """
    nets = RLNetworks(
        obs_dim=obs_dim,
        act_dim=act_dim,
        latent_dim=latent_dim,
        hidden_dims=tuple(hidden_dims),
        num_qs=num_qs,
        **kwargs,
    )
    if encoder is not None:
        nets.attach_encoder(encoder, freeze=freeze_encoder)
    return nets
