"""Implicit Q-Learning (IQL) losses for the FRE-conditioned agent.

Paper: "Zero-Shot Reinforcement Learning via Functional Reward Encodings"

Relevant verbatim text (Source: §4.3 Offline RL with FRE):

    "To close the loop on the method, we must learn an FRE-conditioned policy that
    maximizes expected return for tasks within the prior reward distribution. Any
    off-the-shelf RL algorithm can be used for this purpose. The general pipeline is to
    first sample a reward function eta, encode it into z via the FRE encoder, and
    optimize pi(a | s, z)."

    Bellman policy improvement step using FRE (Equation (1) of the paper):

        Q(s, a, z) <- eta(s) + E_{s' ~ p(s'|s,a)}[ max_{a' in A} Q(s', a', z) ]

    "In our experiments, we use implicit Q-learning (Kostrikov et al., 2021) as the
    offline RL method to train our FRE-conditioned policy. This is a widely used
    offline RL algorithm that avoids querying out-of-distribution actions."

    Added by FRE (Source: Addendum - architecture clarifications):
    "For conditioning the RL components (value, critic, etc.) of the FRE-agent with the
    latent embedding z, the latent embedding is simply concatenated to the observation
    state that is fed into the RL components."

Hyper-parameters reproduced exactly (Source: §A Hyperparameters, Table 3):

    RL Network Layers   [512, 512, 512]
    Optimizer           Adam
    Learning Rate       0.0001
    Batch Size          512
    Target Update Rate  0.001
    Discount Factor     0.88
    AWR Temperature     3.0
    IQL Expectile       0.8

Explicit IQL loss forms are NOT specified in the paper (Source: not specified in the
paper) - the losses below follow the reference IQL implementation of Kostrikov et al.
(2021), which the paper explicitly names as the offline RL method being used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Model imports (with a fallback path so the module can be executed directly)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - normal package import
    from fre.models.rl_networks import (
        DEFAULT_AWR_TEMPERATURE,
        DEFAULT_DISCOUNT,
        DEFAULT_EXPECTILE,
        DEFAULT_RL_LAYERS,
        DEFAULT_TAU,
        RLNetworks,
        PolicyOutput,
        make_rl_networks,
    )
except Exception:  # pragma: no cover - direct execution fallback
    import os
    import sys

    _here = os.path.dirname(os.path.abspath(__file__))
    _root = os.path.abspath(os.path.join(_here, "..", ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from fre.models.rl_networks import (  # type: ignore
        DEFAULT_AWR_TEMPERATURE,
        DEFAULT_DISCOUNT,
        DEFAULT_EXPECTILE,
        DEFAULT_RL_LAYERS,
        DEFAULT_TAU,
        RLNetworks,
        PolicyOutput,
        make_rl_networks,
    )

__all__ = [
    "IQLConfig",
    "IQLLosses",
    "IQLLoss",
    "expectile_weights",
    "expectile_loss",
    "quantile_loss",
    "compute_advantages",
    "awr_weights",
    "soft_update",
    "hard_update",
    "polyak_update",
    "target_q_from_value",
    "IQLTrainer",
    "train_iql_step",
    "make_iql_trainer",
    "DEFAULT_IQL_CONFIG",
]

DEFAULT_IQL_CONFIG: Dict[str, float] = {
    "expectile": DEFAULT_EXPECTILE,          # 0.8 (Table 3)
    "awr_temperature": DEFAULT_AWR_TEMPERATURE,  # 3.0 (Table 3)
    "discount": DEFAULT_DISCOUNT,            # 0.88 (Table 3)
    "tau": DEFAULT_TAU,                      # 0.001 (Table 3)
    "learning_rate": 1e-4,                   # Table 3
    "batch_size": 512,                       # Table 3
    "hidden_dims": DEFAULT_RL_LAYERS,        # [512, 512, 512] (Table 3)
}

# Reference IQL implementation clips the advantage weights to avoid exploding policy
# updates; the paper does not mention this (Source: not specified in the paper).
DEFAULT_ADV_WEIGHT_CLAMP = 100.0


# ---------------------------------------------------------------------------
# Core loss primitives
# ---------------------------------------------------------------------------
def expectile_weights(error: torch.Tensor, expectile: float) -> torch.Tensor:
    """Expectile weights of the IQL value loss.

    ``|tau - 1(error < 0)|`` (Kostrikov et al., 2021, Eq. 3), the weight applied to
    the squared error for each element of ``error``. ``expectile`` tau = 0.8 per
    Table 3 of the paper.
    """
    if not 0.0 < expectile < 1.0:
        raise ValueError(f"expectile must lie in (0, 1), got {expectile}")
    return torch.where(error < 0, 1.0 - expectile, expectile)


def expectile_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    expectile: float = 0.8,
    reduction: str = "mean",
) -> torch.Tensor:
    """Asymmetric (expectile) squared loss used for the IQL value function.

        L_V = E[ |tau - 1(q_target - V)| * (q_target - V)^2 ]
    """
    error = target - prediction
    weights = expectile_weights(error, expectile)
    loss = weights * error.pow(2)
    return _reduce(loss, reduction)


def quantile_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    quantile: float = 0.8,
    reduction: str = "mean",
) -> torch.Tensor:
    """Alias of :func:`expectile_loss` (IQL's tau is an expectile, same formula)."""
    return expectile_loss(prediction, target, expectile=quantile, reduction=reduction)


def compute_advantages(
    q_values: torch.Tensor,
    values: torch.Tensor,
    detach: bool = True,
) -> torch.Tensor:
    """Advantage ``A(s, a, z) = Q(s, a, z) - V(s, z)`` for AWR policy extraction."""
    advantage = q_values - values
    return advantage.detach() if detach else advantage


def awr_weights(
    advantage: torch.Tensor,
    temperature: float = 3.0,
    clamp_max: Optional[float] = DEFAULT_ADV_WEIGHT_CLAMP,
    normalize: bool = False,
) -> torch.Tensor:
    """Advantage-weighted regression weights ``exp(temperature * A)``.

    The paper lists ``AWR Temperature = 3.0`` (Table 3). The reference IQL
    implementation multiplies the advantage by this value before exponentiating
    (i.e. it plays the role of a beta / inverse temperature); that convention is used
    here. ``normalize=True`` divides the advantage by its standard deviation (a
    known IQL variant), disabled by default since the paper does not request it.
    """
    if normalize:
        std = advantage.std()
        if float(std) > 1e-6:
            advantage = advantage / (std + 1e-6)
    # exp(advantage / temperature) if 'temperature' is interpreted as a temperature;
    # the reference implementation uses exp(beta * advantage) with beta = 3.0. Both are
    # supported: values > 0 are used as the multiplicative inverse temperature.
    weights = torch.exp(temperature * advantage)
    if clamp_max is not None:
        weights = torch.clamp(weights, max=clamp_max)
    return weights


def _reduce(loss: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"Unknown reduction '{reduction}'")


# ---------------------------------------------------------------------------
# Bellman target helpers
# ---------------------------------------------------------------------------
def target_q_from_value(
    rewards: torch.Tensor,
    values_next: torch.Tensor,
    terminals: torch.Tensor,
    discount: float = 0.88,
) -> torch.Tensor:
    """IQL Bellman target ``y = r + gamma * (1 - done) * V_target(s', z)``.

    This is the value-based form of the FRE Bellman step in §4.3,
    ``Q(s, a, z) <- eta(s) + E[max_{a'} Q(s', a', z)]``, with ``max_{a'} Q(s', a', z)``
    approximated by the target value network (IQL never queries out-of-distribution
    actions, per §4.3). ``r = eta(s)`` is the sampled prior reward function evaluated
    at the transition's state.
    """
    rewards = rewards.reshape(-1)
    values_next = values_next.reshape(-1)
    terminals = terminals.reshape(-1).float()
    return rewards + discount * (1.0 - terminals) * values_next


def soft_update(target: nn.Module, source: nn.Module, tau: float = 0.001) -> None:
    """Polyak averaging ``target <- tau * source + (1 - tau) * target`` (rate 0.001)."""
    with torch.no_grad():
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.data.mul_(1.0 - tau).add_(s_param.data, alpha=tau)
        for t_buf, s_buf in zip(target.buffers(), source.buffers()):
            t_buf.data.copy_(s_buf.data)


def polyak_update(target: nn.Module, source: nn.Module, tau: float = 0.001) -> None:
    """Alias of :func:`soft_update` (name used by several IQL codebases)."""
    soft_update(target, source, tau=tau)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    """Copy all parameters/buffers from ``source`` into ``target``."""
    with torch.no_grad():
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.data.copy_(s_param.data)
        for t_buf, s_buf in zip(target.buffers(), source.buffers()):
            t_buf.data.copy_(s_buf.data)


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------
@dataclass
class IQLConfig:
    """IQL hyper-parameters, defaulting to Table 3 of the paper."""

    expectile: float = 0.8
    awr_temperature: float = 3.0
    discount: float = 0.88
    tau: float = 0.001
    learning_rate: float = 1e-4
    batch_size: int = 512
    hidden_dims: Tuple[int, ...] = DEFAULT_RL_LAYERS
    adv_weight_clamp: Optional[float] = DEFAULT_ADV_WEIGHT_CLAMP
    normalize_advantage: bool = False
    value_loss_coef: float = 1.0
    q_loss_coef: float = 1.0
    policy_loss_coef: float = 1.0
    max_grad_norm: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "expectile": self.expectile,
            "awr_temperature": self.awr_temperature,
            "discount": self.discount,
            "tau": self.tau,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "hidden_dims": list(self.hidden_dims),
            "adv_weight_clamp": self.adv_weight_clamp,
            "normalize_advantage": self.normalize_advantage,
            "value_loss_coef": self.value_loss_coef,
            "q_loss_coef": self.q_loss_coef,
            "policy_loss_coef": self.policy_loss_coef,
            "max_grad_norm": self.max_grad_norm,
        }


@dataclass
class IQLLosses:
    """Per-update IQL loss statistics (float values, detached from the graph)."""

    value_loss: Optional[float] = None
    q_loss: Optional[float] = None
    policy_loss: Optional[float] = None
    total_loss: Optional[float] = None
    mean_q: Optional[float] = None
    mean_v: Optional[float] = None
    mean_advantage: Optional[float] = None
    mean_weight: Optional[float] = None
    mean_abs_td_error: Optional[float] = None

    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


IQLLoss = IQLLosses  # convenient alias


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class IQLTrainer:
    """IQL update engine for the z-conditioned FRE networks.

    Implements the three IQL losses (Kostrikov et al., 2021):

      * value:  ``L_V = E[ |tau - 1(Q_target - V(s,z) < 0)| (Q_target - V(s,z))^2 ]``
                with ``Q_target = min(Q1, Q2)(s, a, z)`` detached, expectile 0.8;
      * critic: ``L_Q = E[ (Q(s,a,z) - (r + gamma (1-d) V_target(s',z)))^2 ]``
                with ``r = eta(s)`` from the sampled prior reward function and
                ``gamma = 0.88``;
      * policy: ``L_pi = -E[ exp(beta A(s,a,z)) log pi(a | s, z) ]`` with the advantage
                ``A = Q(s,a,z) - V(s,z)`` detached and beta = AWR temperature 3.0.

    The class owns the optimizers (Adam, lr 1e-4 for all RL components) and the
    target value network updated at rate 0.001.
    """

    def __init__(
        self,
        networks: Optional[RLNetworks] = None,
        obs_dim: Optional[int] = None,
        act_dim: Optional[int] = None,
        latent_dim: int = 128,
        config: Optional[IQLConfig] = None,
        device: Optional[torch.device] = None,
        encoder: Optional[nn.Module] = None,
        freeze_encoder: bool = True,
        **config_overrides,
    ) -> None:
        self.config = config or IQLConfig(**config_overrides) if config is None else config
        if config is not None and config_overrides:
            for key, value in config_overrides.items():
                if not hasattr(self.config, key):
                    raise TypeError(f"Unknown IQLConfig field '{key}'")
                setattr(self.config, key, value)

        if networks is None:
            if obs_dim is None or act_dim is None:
                raise ValueError(
                    "Either pass `networks` or both `obs_dim` and `act_dim` to IQLTrainer."
                )
            networks = make_rl_networks(
                obs_dim=obs_dim,
                act_dim=act_dim,
                latent_dim=latent_dim,
                hidden_dims=tuple(self.config.hidden_dims),
                encoder=encoder,
                freeze_encoder=freeze_encoder,
            )
        elif encoder is not None:
            networks.attach_encoder(encoder, freeze=freeze_encoder)

        self.networks = networks
        self.device = device if device is not None else _infer_device(networks)
        self.networks.to(self.device)

        # Only RL parameters are optimized; the FRE encoder stays frozen (strided
        # training scheme, §4.3).
        params = list(self.networks.trainable_parameters())
        if not params:
            raise RuntimeError("No trainable RL parameters found (is the encoder frozen only?)")
        self.optimizer = torch.optim.Adam(params, lr=self.config.learning_rate)

        self.target_v = self.networks.target_v
        self.train_step_count: int = 0

    # -- convenience wrappers ------------------------------------------------
    @property
    def expectile(self) -> float:
        return self.config.expectile

    @property
    def discount(self) -> float:
        return self.config.discount

    @property
    def awr_temperature(self) -> float:
        return self.config.awr_temperature

    def encode(self, states: torch.Tensor, rewards: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode a context set of K reward-labelled states into z (frozen encoder)."""
        return self.networks.encode_context(states, rewards, **kwargs)

    # -- individual losses --------------------------------------------------
    def value_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        z: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Expectile regression of V(s, z) onto the detached min-Q(s, a, z)."""
        observations, actions = self._prepare_obs_act(observations, actions)
        with torch.no_grad():
            q_target = self.networks.min_q(observations, actions, z)
        v = self.networks.values(observations, z)
        return expectile_loss(v, q_target, expectile=self.config.expectile, reduction=reduction)

    def q_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        z: torch.Tensor,
        z_next: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        return_td_error: bool = False,
    ):
        """Bellman critic loss with the sampled prior reward ``r = eta(s)``."""
        observations, actions = self._prepare_obs_act(observations, actions)
        next_observations = self._as_tensor(next_observations, self.device)
        rewards = self._as_tensor(rewards, self.device).reshape(-1)
        terminals = self._as_tensor(terminals, self.device).reshape(-1)
        z = self._as_z(z)
        z_next = z if z_next is None else self._as_z(z_next)

        with torch.no_grad():
            v_next = self.networks.target_values(next_observations, z_next)
            targets = target_q_from_value(
                rewards, v_next, terminals, discount=self.config.discount
            )

        q = self.networks.q_values(observations, actions, z)
        td_error = q - targets.unsqueeze(0) if q.dim() == 2 else q - targets
        loss = td_error.pow(2)
        reduced = _reduce(loss, reduction)
        if return_td_error:
            return reduced, td_error.abs().mean().detach()
        return reduced

    def policy_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        z: torch.Tensor,
        reduction: str = "mean",
    ):
        """Advantage-weighted regression policy extraction (exp(beta * A) log pi)."""
        observations, actions = self._prepare_obs_act(observations, actions)
        z = self._as_z(z)
        with torch.no_grad():
            q = self.networks.min_q(observations, actions, z)
            v = self.networks.values(observations, z)
            advantage = compute_advantages(q, v, detach=True)
            weights = awr_weights(
                advantage,
                temperature=self.config.awr_temperature,
                clamp_max=self.config.adv_weight_clamp,
                normalize=self.config.normalize_advantage,
            )

        output: PolicyOutput = self.networks.policy(
            observations, z, sample=True, return_log_prob=True
        )
        log_prob = output.log_prob
        if log_prob is None:
            log_prob = self.networks.log_prob(observations, z, actions)
        loss = -(weights * log_prob)
        reduced = _reduce(loss, reduction)
        stats = {
            "advantage": advantage.mean(),
            "weight": weights.mean(),
        }
        return reduced, stats

    # -- full update --------------------------------------------------------
    def update(
        self,
        observations,
        actions,
        next_observations,
        rewards,
        terminals,
        z,
        z_next=None,
        update_target: bool = True,
        step_index: Optional[int] = None,
    ) -> IQLLosses:
        """Run one full IQL update (value -> critic -> policy) and step the optimizers.

        The three losses are optimized sequentially, exactly as in the reference IQL
        implementation (each with its own backward/step), because the policy loss
        depends on the freshly-updated value function.
        """
        observations, actions = self._prepare_obs_act(observations, actions)
        z = self._as_z(z)

        # ---- 1. value function (expectile regression) ----------------------
        v_loss = self.value_loss(observations, actions, z)
        self.optimizer.zero_grad(set_to_none=True)
        (self.config.value_loss_coef * v_loss).backward()
        self._clip_grads()
        self.optimizer.step()

        # ---- 2. critic (Bellman regression on r = eta(s)) ------------------
        q_loss, mean_abs_td = self.q_loss(
            observations,
            actions,
            next_observations,
            rewards,
            terminals,
            z,
            z_next=z_next,
            return_td_error=True,
        )
        self.optimizer.zero_grad(set_to_none=True)
        (self.config.q_loss_coef * q_loss).backward()
        self._clip_grads()
        self.optimizer.step()

        # ---- 3. policy (AWR) ----------------------------------------------
        pi_loss, pi_stats = self.policy_loss(observations, actions, z)
        self.optimizer.zero_grad(set_to_none=True)
        (self.config.policy_loss_coef * pi_loss).backward()
        self._clip_grads()
        self.optimizer.step()

        # ---- 4. target network (rate 0.001) -------------------------------
        if update_target:
            self.networks.update_target(tau=self.config.tau)

        self.train_step_count += 1
        with torch.no_grad():
            mean_q = float(self.networks.min_q(observations, actions, z).mean().item())
            mean_v = float(self.networks.values(observations, z).mean().item())

        return IQLLosses(
            value_loss=float(v_loss.detach().item()),
            q_loss=float(q_loss.detach().item()),
            policy_loss=float(pi_loss.detach().item()),
            total_loss=float(
                (self.config.value_loss_coef * v_loss
                 + self.config.q_loss_coef * q_loss
                 + self.config.policy_loss_coef * pi_loss).detach().item()
            ),
            mean_q=mean_q,
            mean_v=mean_v,
            mean_advantage=float(pi_stats["advantage"].item()),
            mean_weight=float(pi_stats["weight"].item()),
            mean_abs_td_error=float(mean_abs_td.item()),
        )

    # -- helpers ------------------------------------------------------------
    def _clip_grads(self) -> None:
        if self.config.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                list(self.networks.trainable_parameters()), self.config.max_grad_norm
            )

    def _prepare_obs_act(
        self, observations, actions
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        observations = self._as_tensor(observations, self.device)
        actions = self._as_tensor(actions, self.device)
        if observations.dim() == 3:  # (B, T, obs_dim) -> (B*T, obs_dim)
            observations = observations.reshape(-1, observations.shape[-1])
        if actions.dim() == 3:
            actions = actions.reshape(-1, actions.shape[-1])
        return observations, actions

    def _as_tensor(self, value, device: torch.device) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            if value.dtype == torch.float64:
                value = value.float()
            return value.to(device)
        return torch.as_tensor(np.asarray(value), dtype=torch.float32, device=device)

    def _as_z(self, z) -> torch.Tensor:
        z = self._as_tensor(z, self.device)
        if z.dim() == 3:  # (B, 1, latent) -> (B, latent)
            z = z.reshape(z.shape[0], -1)
        return z

    def describe(self) -> Dict[str, object]:
        info = self.config.to_dict()
        info.update(
            {
                "train_step": self.train_step_count,
                "num_trainable_params": int(
                    sum(p.numel() for p in self.networks.trainable_parameters())
                ),
                "optimizer": type(self.optimizer).__name__,
            }
        )
        return info


def _infer_device(module: nn.Module) -> torch.device:
    for param in module.parameters():
        return param.device
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Functional entry point + factory
# ---------------------------------------------------------------------------
def train_iql_step(
    trainer: IQLTrainer,
    observations,
    actions,
    next_observations,
    rewards,
    terminals,
    z,
    z_next=None,
    **kwargs,
) -> IQLLosses:
    """Convenience wrapper around :meth:`IQLTrainer.update`."""
    return trainer.update(
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        rewards=rewards,
        terminals=terminals,
        z=z,
        z_next=z_next,
        **kwargs,
    )


def make_iql_trainer(
    obs_dim: int,
    act_dim: int,
    latent_dim: int = 128,
    hidden_dims: Sequence[int] = DEFAULT_RL_LAYERS,
    encoder: Optional[nn.Module] = None,
    freeze_encoder: bool = True,
    device: Optional[torch.device] = None,
    learning_rate: float = 1e-4,
    expectile: float = 0.8,
    awr_temperature: float = 3.0,
    discount: float = 0.88,
    tau: float = 0.001,
    batch_size: int = 512,
    **kwargs,
) -> IQLTrainer:
    """Factory mirroring the other ``make_*`` helpers, with Table 3 defaults."""
    config = IQLConfig(
        expectile=expectile,
        awr_temperature=awr_temperature,
        discount=discount,
        tau=tau,
        learning_rate=learning_rate,
        batch_size=batch_size,
        hidden_dims=tuple(hidden_dims),
        **kwargs,
    )
    networks = make_rl_networks(
        obs_dim=obs_dim,
        act_dim=act_dim,
        latent_dim=latent_dim,
        hidden_dims=tuple(hidden_dims),
        encoder=encoder,
        freeze_encoder=freeze_encoder,
        discount=discount,
        tau=tau,
        expectile=expectile,
        awr_temperature=awr_temperature,
    )
    return IQLTrainer(networks=networks, config=config, device=device)
