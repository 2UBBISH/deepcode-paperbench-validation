"""Neural networks for the FRE-conditioned IQL agent.

Paper references
----------------
Section 4.3 (Offline RL with FRE):
    "The RL components (Q-function, value function, and policy) are all conditioned
    on z. ... Q(s, a, z) <- eta(s) + E_{s'~p(s'|s,a)}[max_{a' in A} Q(s', a', z)]"

Appendix A, Table 3 (Hyperparameters used for FRE):
    - RL Network Layers = [512, 512, 512]
    - Discount Factor = 0.88
    - Target Update Rate = 0.001
    - AWR Temperature = 3.0
    - IQL Expectile = 0.8

Implementation notes (paper-silent choices are flagged in comments):
    * ``z`` is *concatenated to the observation* (Plan: "z concatenated to observation"),
      i.e. the Q/V/policy input is ``concat(s, z)`` (or ``concat(s, a, z)`` for Q).
    * Hidden activations: ReLU (the FRE repo uses ReLU MLPs; the paper is silent).
    * The policy is a tanh-squashed diagonal Gaussian with state-independent learned
      log-std (standard in IQL / AWR implementations). The addendum's GC-BC baseline
      pins ``log_std`` clamp at -5.0; we reuse the same clamp for consistency.
    * Q is a twin (two independent MLPs) to reduce overestimation bias; the target is
      the min of the two target critics (paper is silent -> standard IQL default).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal

__all__ = [
    "MLP",
    "make_activation",
    "QNetwork",
    "ValueNetwork",
    "GaussianPolicy",
    "TanhNormalPolicy",
    "IQLEstimator",
    "LOG_STD_MIN",
    "LOG_STD_MAX",
]

# Addendum, "Additional Details on GC-BC": "log-std clamp -5.0".
LOG_STD_MIN = -5.0
# Upper clamp is a common IQL/AWR safeguard; the paper only specifies the -5.0 lower
# bound (for GC-BC).  We use the same upper bound as the FRE encoder's log_std head.
LOG_STD_MAX = 2.0


def make_activation(name: str) -> nn.Module:
    """Resolve an activation function by name.

    Accepts: ``relu`` (default), ``gelu``, ``silu``/``swish``, ``tanh``, ``elu``,
    ``mish``, ``leaky_relu``, or ``"none"``/``"identity"`` for a no-op.
    """
    key = str(name).lower()
    if key in ("relu",):
        return nn.ReLU()
    if key in ("gelu",):
        return nn.GELU()
    if key in ("silu", "swish"):
        return nn.SiLU()
    if key in ("tanh",):
        return nn.Tanh()
    if key in ("elu",):
        return nn.ELU()
    if key in ("mish",):
        return nn.Mish()
    if key in ("leaky_relu", "leakyrelu"):
        return nn.LeakyReLU()
    if key in ("none", "identity"):
        return nn.Identity()
    raise ValueError(f"Unknown activation: {name!r}")


class MLP(nn.Module):
    """Plain feedforward MLP.

    ``Linear -> (optional LayerNorm) -> Activation`` per hidden layer, then a final
    ``Linear`` head to ``output_dim``.  Defaults to the paper's RL network width
    ``[512, 512, 512]`` with ReLU activations.

    Args:
        input_dim: dimensionality of the network input.
        hidden_layers: sizes of the hidden layers (default ``(512, 512, 512)``).
        output_dim: dimensionality of the final linear output.
        activation: activation name (see :func:`make_activation`).
        layernorm: insert ``LayerNorm`` before each activation (paper-silent default False).
        weight_init: ``"xavier"`` (default) or ``"orthogonal"``.
        output_activation: optional activation applied to the final output.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_layers: Sequence[int] = (512, 512, 512),
        output_dim: int = 1,
        activation: str = "relu",
        layernorm: bool = False,
        weight_init: str = "xavier",
        output_activation: Optional[str] = None,
        append_input_dim: int = 0,
    ) -> None:
        super().__init__()
        hidden_layers = tuple(int(h) for h in hidden_layers)
        self.input_dim = int(input_dim)
        # ``append_input_dim`` allows the input (e.g. z) to be concatenated after the
        # hidden stack — unused by default, kept for expressive baselines.
        self.append_input_dim = int(append_input_dim)
        self.output_dim = int(output_dim)

        layers = []
        prev = self.input_dim
        for width in hidden_layers:
            layers.append(nn.Linear(prev, width))
            if layernorm:
                layers.append(nn.LayerNorm(width))
            layers.append(make_activation(activation))
            prev = width
        layers.append(nn.Linear(prev + self.append_input_dim, self.output_dim))
        self.net = nn.Sequential(*layers)
        self.output_activation = (
            make_activation(output_activation) if output_activation else None
        )
        self._hidden_layers = hidden_layers

        self._init_weights(weight_init)

    def _init_weights(self, mode: str) -> None:
        mode = str(mode).lower()
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if mode == "orthogonal":
                    nn.init.orthogonal_(module.weight, gain=1.0)
                else:  # "xavier" default
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: Tensor, extra: Optional[Tensor] = None) -> Tensor:
        """Run the MLP.

        Args:
            x: ``(..., input_dim)`` tensor.
            extra: optional additional features appended before the final linear layer.

        Returns:
            ``(..., output_dim)`` tensor.
        """
        x = x.to(self.net[0].weight.dtype) if x.is_floating_point() else x.float()
        if self.append_input_dim > 0:
            if extra is None:
                raise ValueError("`extra` is required when append_input_dim > 0")
            if layer_out := x.numel() and False:  # pragma: no cover - never taken
                pass
            # Expand extra over any leading dims of x
            while extra.dim() < x.dim():
                extra = extra.unsqueeze(0)
            extra = extra.expand(*x.shape[:-1], -1)
            hidden = x
            for layer in list(self.net)[:-1]:
                hidden = layer(hidden)
            out = self.net[-1](torch.cat([hidden, extra], dim=-1))
        else:
            out = self.net(x)
        if self.output_activation is not None:
            out = self.output_activation(out)
        return out

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, hidden={self._hidden_layers}, "
            f"output_dim={self.output_dim}"
        )


def _concat_obs_z(obs: Tensor, z: Optional[Tensor]) -> Tensor:
    """Concatenate observation and latent task vector along the last dim.

    Handles any leading batch dims by broadcasting ``z`` (e.g. a single shared ``z``
    for a whole batch, or one ``z`` per state).  This realizes the paper's statement
    that the RL components are "all conditioned on z" with z concatenated to the
    observation (Plan, Component 5).
    """
    if z is None:
        return obs
    if z.dim() == 1:
        z = z.unsqueeze(0)
    while z.dim() < obs.dim():
        z = z.unsqueeze(0)
    z = z.expand(*obs.shape[: obs.dim() - 1], z.shape[-1])
    return torch.cat([obs, z.to(obs.dtype)], dim=-1)


class QNetwork(nn.Module):
    """Twin z-conditioned Q-function ``Q(s, a, z)`` (Section 4.3).

    Input: ``concat(s, a, z)`` -> MLP ``[512, 512, 512]`` -> scalar.  Two independent
    MLPs are instantiated (``q1``, ``q2``) so IQL can use the min for the Bellman
    backup.

    Args:
        obs_dim: observation dimensionality.
        action_dim: action dimensionality.
        latent_dim: dimensionality of the task embedding ``z`` (128 for FRE).
        hidden_layers: hidden widths (Table 3: ``[512, 512, 512]``).
        activation: activation name (default ``relu``).
        layernorm: optional LayerNorm before activations.
        num_qs: number of critics (default 2).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 128,
        hidden_layers: Sequence[int] = (512, 512, 512),
        activation: str = "relu",
        layernorm: bool = False,
        num_qs: int = 2,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.num_qs = int(num_qs)
        input_dim = self.obs_dim + self.action_dim + self.latent_dim
        self.qs = nn.ModuleList(
            [
                MLP(input_dim, hidden_layers, output_dim=1, activation=activation, layernorm=layernorm)
                for _ in range(self.num_qs)
            ]
        )

    def forward(self, obs: Tensor, action: Tensor, z: Optional[Tensor] = None) -> Tensor:
        """Returns ``(B, num_qs)`` Q-values."""
        x = torch.cat([obs, action], dim=-1)
        x = _concat_obs_z(x, z)
        return torch.cat([q(x) for q in self.qs], dim=-1)

    def q_min(self, obs: Tensor, action: Tensor, z: Optional[Tensor] = None) -> Tensor:
        """Min over the twin critics -> ``(B,)`` (target backup used by IQL)."""
        return self.forward(obs, action, z).min(dim=-1).values

    @classmethod
    def from_config(cls, config, obs_dim: int, action_dim: int, **overrides) -> "QNetwork":
        kwargs = dict(
            obs_dim=obs_dim,
            action_dim=action_dim,
            latent_dim=getattr(config, "latent_dim", 128),
            hidden_layers=tuple(getattr(config, "rl_hidden_layers", (512, 512, 512))),
            activation=getattr(config, "rl_activation", "relu"),
            layernorm=getattr(config, "rl_layernorm", False),
            num_qs=getattr(config, "rl_num_qs", 2),
        )
        kwargs.update(overrides)
        return cls(**kwargs)


class ValueNetwork(nn.Module):
    """z-conditioned state value ``V(s, z)`` (Section 4.3).

    Trained by IQL's expectile regression onto ``Q(s, a, z)`` with expectile
    ``tau = 0.8`` (Appendix A, Table 3).  Input: ``concat(s, z)``.
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int = 128,
        hidden_layers: Sequence[int] = (512, 512, 512),
        activation: str = "relu",
        layernorm: bool = False,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.latent_dim = int(latent_dim)
        self.net = MLP(
            self.obs_dim + self.latent_dim,
            hidden_layers,
            output_dim=1,
            activation=activation,
            layernorm=layernorm,
        )

    def forward(self, obs: Tensor, z: Optional[Tensor] = None) -> Tensor:
        """Returns ``(B,)`` state values."""
        return self.net(_concat_obs_z(obs, z)).squeeze(-1)

    @classmethod
    def from_config(cls, config, obs_dim: int, **overrides) -> "ValueNetwork":
        kwargs = dict(
            obs_dim=obs_dim,
            latent_dim=getattr(config, "latent_dim", 128),
            hidden_layers=tuple(getattr(config, "rl_hidden_layers", (512, 512, 512))),
            activation=getattr(config, "rl_activation", "relu"),
            layernorm=getattr(config, "rl_layernorm", False),
        )
        kwargs.update(overrides)
        return cls(**kwargs)


class GaussianPolicy(nn.Module):
    """Diagonal Gaussian policy ``pi(a | s, z)`` with optional tanh squashing.

    The network outputs ``(mean, log_std)``; ``log_std`` is state-independent
    (a learned parameter) as is standard in IQL / AWR implementations, and it is
    clamped to ``[LOG_STD_MIN, LOG_STD_MAX]`` with ``LOG_STD_MIN = -5.0`` following
    the addendum's GC-BC detail.

    Args:
        obs_dim: observation dimensionality.
        action_dim: action dimensionality.
        latent_dim: task-embedding dimensionality (128 for FRE).
        hidden_layers: hidden widths (Table 3: ``[512, 512, 512]``).
        activation: activation name (default ``relu``).
        tanh_squash: squash actions through ``tanh`` (default True, matching the
            bounded continuous-control action spaces of AntMaze/ExORL/Kitchen).
        log_std_min / log_std_max: clamps applied to the predicted log-std.
        state_dependent_std: if True, ``log_std`` is a second network head.
        layernorm: optional LayerNorm before activations.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 128,
        hidden_layers: Sequence[int] = (512, 512, 512),
        activation: str = "relu",
        tanh_squash: bool = True,
        log_std_min: float = LOG_STD_MIN,
        log_std_max: float = LOG_STD_MAX,
        state_dependent_std: bool = False,
        layernorm: bool = False,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.tanh_squash = bool(tanh_squash)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.state_dependent_std = bool(state_dependent_std)

        if self.state_dependent_std:
            self.net = MLP(
                self.obs_dim + self.latent_dim,
                hidden_layers,
                output_dim=2 * self.action_dim,
                activation=activation,
                layernorm=layernorm,
            )
        else:
            self.net = MLP(
                self.obs_dim + self.latent_dim,
                hidden_layers,
                output_dim=self.action_dim,
                activation=activation,
                layernorm=layernorm,
            )
            self.log_std = nn.Parameter(torch.zeros(self.action_dim))

    # ---------------------------------------------------------------- internals
    def _heads(self, obs: Tensor, z: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        x = _concat_obs_z(obs, z)
        out = self.net(x)
        if self.state_dependent_std:
            mean, log_std = out.chunk(2, dim=-1)
        else:
            mean = out
            log_std = self.log_std.expand(mean.shape[0], -1) if mean.dim() > 1 else self.log_std
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    @staticmethod
    def _squash(mean: Tensor, log_std: Tensor) -> Tuple[Tensor, Tensor]:
        """Reparameterised tanh-squashed sample + sum-of-log-Jacobian."""
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        pre_tanh = mean + std * eps
        action = torch.tanh(pre_tanh)
        log_prob = Normal(mean, std).log_prob(pre_tanh)
        # d/dx log tanh'(x) = log(1 - tanh(x)^2); clamp for numerical stability.
        log_prob = log_prob - torch.log(1.0 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1)

    # ------------------------------------------------------------------- public
    def forward(
        self,
        obs: Tensor,
        z: Optional[Tensor] = None,
        deterministic: bool = False,
        with_log_prob: bool = False,
    ):
        """Sample an action (tanh-squashed when configured).

        Args:
            obs: ``(B, obs_dim)`` observations.
            z: ``(B, latent_dim)`` (or shared) task embedding.
            deterministic: if True, return ``tanh(mean)`` with no randomness.
            with_log_prob: return ``(action, log_prob)``, where ``log_prob`` includes
                the tanh change-of-variables term.

        Returns:
            ``action`` or ``(action, log_prob)``.
        """
        mean, log_std = self._heads(obs, z)
        if deterministic or self.tanh_squash is False:
            action = torch.tanh(mean) if self.tanh_squash else mean
            if not with_log_prob:
                return action
            std = torch.exp(log_std)
            # log prob at the deterministic action (pre-tanh mean)
            lp = Normal(mean, std).log_prob(mean)
            if self.tanh_squash:
                lp = lp - torch.log(1.0 - action.pow(2) + 1e-6)
            return action, lp.sum(dim=-1)

        if self.tanh_squash:
            action, log_prob = self._squash(mean, log_std)
        else:
            std = torch.exp(log_std)
            dist = Normal(mean, std)
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        if with_log_prob:
            return action, log_prob
        return action

    @torch.no_grad()
    def act(
        self,
        obs: Tensor,
        z: Optional[Tensor] = None,
        deterministic: bool = False,
    ) -> Tensor:
        """Inference helper (no grad): returns an action in the env action space."""
        return self.forward(obs, z, deterministic=deterministic)

    def evaluate_actions(self, obs: Tensor, action: Tensor, z: Optional[Tensor] = None) -> Tensor:
        """Log-probability of ``action`` under ``pi(.|s,z)`` (used by AWR weighting).

        Inverts the tanh squashing to obtain the pre-tanh value, then applies the
        standard log-Jacobian correction.  Returns ``(B,)``.
        """
        mean, log_std = self._heads(obs, z)
        std = torch.exp(log_std)
        if self.tanh_squash:
            # clamp avoids +/- inf at saturated actions
            action_safe = torch.clamp(action, -1.0 + 1e-6, 1.0 - 1e-6)
            pre_tanh = 0.5 * torch.log((1 + action_safe) / (1 - action_safe))
            log_prob = Normal(mean, std).log_prob(pre_tanh)
            log_prob = log_prob - torch.log(1.0 - action_safe.pow(2) + 1e-6)
        else:
            log_prob = Normal(mean, std).log_prob(action)
        return log_prob.sum(dim=-1)

    def distribution(self, obs: Tensor, z: Optional[Tensor] = None) -> Normal:
        """Returns the pre-squash :class:`~torch.distributions.Normal` (diagnostics)."""
        mean, log_std = self._heads(obs, z)
        return Normal(mean, torch.exp(log_std))

    @classmethod
    def from_config(
        cls, config, obs_dim: int, action_dim: int, **overrides
    ) -> "GaussianPolicy":
        kwargs = dict(
            obs_dim=obs_dim,
            action_dim=action_dim,
            latent_dim=getattr(config, "latent_dim", 128),
            hidden_layers=tuple(getattr(config, "rl_hidden_layers", (512, 512, 512))),
            activation=getattr(config, "rl_activation", "relu"),
            tanh_squash=getattr(config, "rl_tanh_squash", True),
            log_std_min=getattr(config, "log_std_min", LOG_STD_MIN),
            log_std_max=getattr(config, "log_std_max", LOG_STD_MAX),
            state_dependent_std=getattr(config, "rl_state_dependent_std", False),
            layernorm=getattr(config, "rl_layernorm", False),
        )
        kwargs.update(overrides)
        return cls(**kwargs)


# Tanh-squashed Gaussian policy is the default FRE/IQL policy; alias for clarity.
TanhNormalPolicy = GaussianPolicy


class IQLEstimator(nn.Module):
    """Container bundling the three z-conditioned IQL networks.

    This is a thin convenience wrapper so that ``iql.py`` (and ``main.py``) can build
    ``Q(s, a, z)``, ``V(s, z)`` and ``pi(a | s, z)`` in one call while keeping each
    network individually accessible (e.g. for separate optimizers / target copies).

    Attributes:
        critic: :class:`QNetwork` (twin critics).
        value: :class:`ValueNetwork`.
        policy: :class:`GaussianPolicy`.
        expectile: IQL expectile ``tau`` (Table 3: 0.8).
        temperature: AWR temperature ``lambda`` inverse (Table 3: 3.0).
        discount: discount factor (Table 3: 0.88).
        target_update_rate: Polyak rate for the target critic (Table 3: 0.001).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 128,
        hidden_layers: Sequence[int] = (512, 512, 512),
        activation: str = "relu",
        expectile: float = 0.8,
        temperature: float = 3.0,
        discount: float = 0.88,
        target_update_rate: float = 0.001,
        tanh_squash: bool = True,
        layernorm: bool = False,
        num_qs: int = 2,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.expectile = float(expectile)
        self.temperature = float(temperature)
        self.discount = float(discount)
        self.target_update_rate = float(target_update_rate)

        self.critic = QNetwork(
            obs_dim,
            action_dim,
            latent_dim=latent_dim,
            hidden_layers=hidden_layers,
            activation=activation,
            layernorm=layernorm,
            num_qs=num_qs,
        )
        self.value = ValueNetwork(
            obs_dim,
            latent_dim=latent_dim,
            hidden_layers=hidden_layers,
            activation=activation,
            layernorm=layernorm,
        )
        self.policy = GaussianPolicy(
            obs_dim,
            action_dim,
            latent_dim=latent_dim,
            hidden_layers=hidden_layers,
            activation=activation,
            tanh_squash=tanh_squash,
            layernorm=layernorm,
        )

    @classmethod
    def from_config(cls, config, obs_dim: int, action_dim: int, **overrides) -> "IQLEstimator":
        kwargs = dict(
            obs_dim=obs_dim,
            action_dim=action_dim,
            latent_dim=getattr(config, "latent_dim", 128),
            hidden_layers=tuple(getattr(config, "rl_hidden_layers", (512, 512, 512))),
            activation=getattr(config, "rl_activation", "relu"),
            expectile=getattr(config, "iql_expectile", 0.8),
            temperature=getattr(config, "iql_temperature", 3.0),
            discount=getattr(config, "discount", 0.88),
            target_update_rate=getattr(config, "target_update_rate", 0.001),
            tanh_squash=getattr(config, "rl_tanh_squash", True),
            layernorm=getattr(config, "rl_layernorm", False),
            num_qs=getattr(config, "rl_num_qs", 2),
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    def param_groups(self) -> Dict[str, list]:
        """Separate parameter lists for the critic / value / policy optimizers."""
        return {
            "critic": list(self.critic.parameters()),
            "value": list(self.value.parameters()),
            "policy": list(self.policy.parameters()),
        }

    def num_parameters(self) -> Dict[str, int]:
        return {
            "critic": sum(p.numel() for p in self.critic.parameters()),
            "value": sum(p.numel() for p in self.value.parameters()),
            "policy": sum(p.numel() for p in self.policy.parameters()),
        }
