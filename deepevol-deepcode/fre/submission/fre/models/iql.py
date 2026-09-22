"""z-conditioned Implicit Q-Learning (IQL) agent for FRE.

The FRE policy is trained with implicit Q-learning (Kostrikov et al., 2021)
conditioned on the latent task representation ``z`` produced by the frozen FRE
encoder.  All RL components are conditioned on ``z``:

    Q(s, a, z)   -- twin Q networks (min is used as the conservative estimate)
    V(s, z)      -- expectile value function
    pi(a | s, z) -- tanh-squashed Gaussian policy trained with AWR

Bellman target (paper Eq. 1, §4.3)::

    Q(s, a, z) <- eta(s) + gamma * E_{s' ~ p(s'|s,a)}[max_a' Q(s', a', z)]

where ``eta`` is the reward function sampled from the prior reward distribution
and ``r = eta(s)`` is also the reward used during training (Algorithm 1).

Two variants of the bootstrap target are supported (``bellman_target``):

* ``"max_q"``  -- exact reproduction of Eq. (1) (default).
* ``"v"``      -- standard IQL target ``eta(s) + gamma * V(s', z)`` as used in
  Kostrikov et al. (2021).  The paper notes IQL is the RL algorithm used in
  their experiments, hence both are provided; Eq. (1) is the paper's stated
  Bellman step and is therefore the default.

Losses follow Kostrikov et al. (2021) exactly:

* Expectile regression for V::

      diff = Q_target(s, a, z) - V(s, z)
      w    = |expectile - 1{diff < 0}|
      L_V  = mean(w * diff^2)

* AWR for pi (temperature ``tau_awr`` = 3.0 in the paper's Table 3)::

      adv = Q(s, a, z) - V(s, z)
      L_pi = -mean( min(exp(adv * tau_awr), 100) * log pi(a | s, z) )

Hyperparameters (Appendix A, Table 3): target update rate 0.001 (Polyak),
discount 0.88, AWR temperature 3.0, IQL expectile 0.8, Adam lr 1e-4,
RL network layers [512, 512, 512].
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # pragma: no cover - import style shim
    from .fre_decoder import build_mlp
except ImportError:  # pragma: no cover
    from fre.models.fre_decoder import build_mlp  # type: ignore


__all__ = [
    "IQL",
    "IQLAgent",
    "QNetwork",
    "ValueNetwork",
    "GaussianPolicy",
    "expectile_loss",
    "awr_loss",
    "polyak_update",
    "DEFAULT_EXPECTILE",
    "DEFAULT_AWR_TEMPERATURE",
    "DEFAULT_DISCOUNT",
    "DEFAULT_TARGET_UPDATE_RATE",
    "DEFAULT_RL_HIDDEN_DIMS",
]

# ---------------------------------------------------------------------------
# Paper hyperparameters (Appendix A, Table 3)
# ---------------------------------------------------------------------------
DEFAULT_EXPECTILE: float = 0.8
DEFAULT_AWR_TEMPERATURE: float = 3.0
DEFAULT_DISCOUNT: float = 0.88
DEFAULT_TARGET_UPDATE_RATE: float = 0.001
DEFAULT_RL_HIDDEN_DIMS: Tuple[int, ...] = (512, 512, 512)
DEFAULT_LEARNING_RATE: float = 1e-4
DEFAULT_EXP_ADV_CLAMP: float = 100.0


# ---------------------------------------------------------------------------
# Losses / helpers
# ---------------------------------------------------------------------------
def expectile_loss(diff: torch.Tensor, expectile: float = DEFAULT_EXPECTILE) -> torch.Tensor:
    """Expectile regression loss ``|expectile - 1{diff < 0}| * diff^2`` (mean)."""
    weight = torch.where(diff > 0.0, expectile, 1.0 - expectile)
    return (weight * diff.pow(2)).mean()


def awr_loss(
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    temperature: float = DEFAULT_AWR_TEMPERATURE,
    clamp: float = DEFAULT_EXP_ADV_CLAMP,
) -> torch.Tensor:
    """Advantage-weighted regression loss for the policy."""
    exp_adv = torch.exp(advantages * temperature).clamp(max=clamp)
    loss = -(exp_adv * log_probs).mean()
    return loss


@torch.no_grad()
def polyak_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak (exponential moving average) update of ``target`` from ``source``."""
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)
    for tb, sb in zip(target.buffers(), source.buffers()):
        tb.data.copy_(sb.data)


def _as_tensor(x: Any, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x), device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------
class QNetwork(nn.Module):
    """Q(s, a, z) -- state and action concatenated, then z appended."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 128,
        hidden_dims: Sequence[int] = DEFAULT_RL_HIDDEN_DIMS,
        activation: str = "relu",
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.net = build_mlp(
            self.state_dim + self.action_dim + self.latent_dim,
            tuple(hidden_dims),
            1,
            activation=activation,
            use_layer_norm=use_layer_norm,
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        states, actions, z = _align(states, actions, z)
        x = torch.cat([states, actions, z], dim=-1)
        return self.net(x).squeeze(-1)


class ValueNetwork(nn.Module):
    """V(s, z)."""

    def __init__(
        self,
        state_dim: int,
        latent_dim: int = 128,
        hidden_dims: Sequence[int] = DEFAULT_RL_HIDDEN_DIMS,
        activation: str = "relu",
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.latent_dim = int(latent_dim)
        self.net = build_mlp(
            self.state_dim + self.latent_dim,
            tuple(hidden_dims),
            1,
            activation=activation,
            use_layer_norm=use_layer_norm,
        )

    def forward(self, states: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        states, z = _align_state_latent(states, z)
        return self.net(torch.cat([states, z], dim=-1)).squeeze(-1)


class GaussianPolicy(nn.Module):
    """pi(a | s, z): tanh-squashed diagonal Gaussian."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 128,
        hidden_dims: Sequence[int] = DEFAULT_RL_HIDDEN_DIMS,
        activation: str = "relu",
        use_layer_norm: bool = False,
        action_low: Union[float, Sequence[float], None] = -1.0,
        action_high: Union[float, Sequence[float], None] = 1.0,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
        tanh_squash: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        if tanh_squash is None:
            tanh_squash = action_low is not None and action_high is not None
        self.tanh_squash = bool(tanh_squash)

        self.net = build_mlp(
            self.state_dim + self.latent_dim,
            tuple(hidden_dims),
            2 * self.action_dim,
            activation=activation,
            use_layer_norm=use_layer_norm,
        )

        if self.tanh_squash:
            low = np.broadcast_to(np.asarray(action_low, dtype=np.float32), (self.action_dim,))
            high = np.broadcast_to(np.asarray(action_high, dtype=np.float32), (self.action_dim,))
            scale = (high - low) / 2.0
            bias = (high + low) / 2.0
            self.register_buffer("action_scale", torch.as_tensor(scale.copy(), dtype=torch.float32))
            self.register_buffer("action_bias", torch.as_tensor(bias.copy(), dtype=torch.float32))
        else:
            self.register_buffer("action_scale", torch.ones(self.action_dim))
            self.register_buffer("action_bias", torch.zeros(self.action_dim))

    # -- distribution -----------------------------------------------------
    def distribution(self, states: torch.Tensor, z: torch.Tensor) -> torch.distributions.Normal:
        states, z = _align_state_latent(states, z)
        out = self.net(torch.cat([states, z], dim=-1))
        mean, log_std = torch.chunk(out, 2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return torch.distributions.Normal(mean, log_std.exp())

    def forward(self, states: torch.Tensor, z: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Return actions in the environment's action space."""
        return self.act(states, z, deterministic=deterministic)

    def act(self, states: torch.Tensor, z: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        dist = self.distribution(states, z)
        x = dist.mean if deterministic else dist.rsample()
        if self.tanh_squash:
            return torch.tanh(x) * self.action_scale + self.action_bias
        return x

    def log_prob(self, states: torch.Tensor, actions: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """log pi(a | s, z) with the tanh-squashing correction (sum over dims)."""
        dist = self.distribution(states, z)
        pre = (actions - self.action_bias) / self.action_scale
        pre = torch.clamp(pre, -1.0 + 1e-7, 1.0 - 1e-7)
        x = torch.atanh(pre)
        log_prob = dist.log_prob(x)
        if self.tanh_squash:
            log_prob = log_prob - torch.log(self.action_scale * (1.0 - torch.tanh(x).pow(2)) + 1e-6)
        return log_prob.sum(dim=-1)


def _align(states: torch.Tensor, actions: torch.Tensor, z: torch.Tensor):
    """Broadcast ``z`` to the state batch shape when needed."""
    states = torch.atleast_2d(states)
    actions = torch.atleast_2d(actions)
    z = torch.atleast_2d(z)
    if z.shape[0] == 1 and states.shape[0] > 1:
        z = z.expand(states.shape[0], -1)
    return states, actions, z


def _align_state_latent(states: torch.Tensor, z: torch.Tensor):
    states = torch.atleast_2d(states)
    z = torch.atleast_2d(z)
    if z.shape[0] == 1 and states.shape[0] > 1:
        z = z.expand(states.shape[0], -1)
    return states, z


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class IQL:
    """z-conditioned implicit Q-learning agent.

    The encoder is *not* part of this module: during Phase 2 of the strided
    schedule the FRE encoder is frozen and only its outputs ``z`` are consumed
    here (Algorithm 1, §4.3 "Practical Implementation").
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        latent_dim: int = 128,
        hidden_dims: Sequence[int] = DEFAULT_RL_HIDDEN_DIMS,
        expectile: float = DEFAULT_EXPECTILE,
        awr_temperature: float = DEFAULT_AWR_TEMPERATURE,
        discount: float = DEFAULT_DISCOUNT,
        target_update_rate: float = DEFAULT_TARGET_UPDATE_RATE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        activation: str = "relu",
        use_layer_norm: bool = False,
        action_low: Union[float, Sequence[float], None] = -1.0,
        action_high: Union[float, Sequence[float], None] = 1.0,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
        exp_adv_clamp: float = DEFAULT_EXP_ADV_CLAMP,
        bellman_target: str = "max_q",
        tau: Optional[float] = None,
        device: Union[str, torch.device, None] = None,
        name: str = "iql",
    ) -> None:
        if bellman_target not in ("max_q", "v"):
            raise ValueError("bellman_target must be 'max_q' or 'v'")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.expectile = float(expectile)
        self.awr_temperature = float(awr_temperature)
        self.discount = float(discount)
        self.target_update_rate = float(target_update_rate if tau is None else tau)
        self.learning_rate = float(learning_rate)
        self.exp_adv_clamp = float(exp_adv_clamp)
        self.bellman_target = bellman_target
        self.name = name

        self.device = torch.device(device) if device is not None else torch.device("cpu")

        common = dict(
            latent_dim=self.latent_dim,
            hidden_dims=tuple(hidden_dims),
            activation=activation,
            use_layer_norm=use_layer_norm,
        )
        self.q1 = QNetwork(self.state_dim, self.action_dim, **common)
        self.q2 = QNetwork(self.state_dim, self.action_dim, **common)
        self.q1_target = QNetwork(self.state_dim, self.action_dim, **common)
        self.q2_target = QNetwork(self.state_dim, self.action_dim, **common)
        self.value = ValueNetwork(self.state_dim, **common)
        self.policy = GaussianPolicy(
            self.state_dim,
            self.action_dim,
            action_low=action_low,
            action_high=action_high,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            **common,
        )

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        for p in self.q1_target.parameters():
            p.requires_grad_(False)
        for p in self.q2_target.parameters():
            p.requires_grad_(False)

        self.to(self.device)

        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=self.learning_rate
        )
        self.v_optimizer = torch.optim.Adam(self.value.parameters(), lr=self.learning_rate)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        self.train_steps = 0

    # -- module plumbing --------------------------------------------------
    def to(self, device: Union[str, torch.device]):
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        for module in self.modules():
            module.to(self.device)
        return self

    def modules(self):
        return [self.q1, self.q2, self.q1_target, self.q2_target, self.value, self.policy]

    def state_dict(self) -> Dict[str, Any]:
        return {
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "value": self.value.state_dict(),
            "policy": self.policy.state_dict(),
            "train_steps": self.train_steps,
            "hparams": self.hparams(),
        }

    def load_state_dict(self, state: Dict[str, Any], strict: bool = True) -> None:
        self.q1.load_state_dict(state["q1"], strict=strict)
        self.q2.load_state_dict(state["q2"], strict=strict)
        self.q1_target.load_state_dict(state.get("q1_target", state["q1"]), strict=strict)
        self.q2_target.load_state_dict(state.get("q2_target", state["q2"]), strict=strict)
        self.value.load_state_dict(state["value"], strict=strict)
        self.policy.load_state_dict(state["policy"], strict=strict)
        self.train_steps = int(state.get("train_steps", 0))

    def hparams(self) -> Dict[str, Any]:
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "latent_dim": self.latent_dim,
            "expectile": self.expectile,
            "awr_temperature": self.awr_temperature,
            "discount": self.discount,
            "target_update_rate": self.target_update_rate,
            "learning_rate": self.learning_rate,
            "bellman_target": self.bellman_target,
        }

    def train(self, mode: bool = True):
        for module in self.modules():
            module.train(mode)
        for p in self.q1_target.parameters():
            p.requires_grad_(False)
        for p in self.q2_target.parameters():
            p.requires_grad_(False)
        return self

    def eval(self):
        return self.train(False)

    def parameters(self):
        params = []
        for module in [self.q1, self.q2, self.value, self.policy]:
            params.extend(module.parameters())
        return params

    # -- individual losses -------------------------------------------------
    def q_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        rewards: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the Bellman target and the twin-Q loss."""
        q1 = self.q1(observations, actions, z)
        q2 = self.q2(observations, actions, z)

        with torch.no_grad():
            if self.bellman_target == "max_q":
                next_q1 = self.q1_target(next_observations, self.policy.act(next_observations, z), z)
                next_q2 = self.q2_target(next_observations, self.policy.act(next_observations, z), z)
                # max over the (implicit) deterministic policy action of the twin nets
                next_q = torch.maximum(next_q1, next_q2)
            else:
                next_q = self.value(next_observations, z)
            target_q = rewards + self.discount * (1.0 - dones) * next_q
            target_q = target_q.detach()

        loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        return loss, q1, target_q

    def value_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            q1 = self.q1(observations, actions, z)
            q2 = self.q2(observations, actions, z)
            q = torch.minimum(q1, q2)
        v = self.value(observations, z)
        loss = expectile_loss(q - v, expectile=self.expectile)
        return loss, v, q

    def policy_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            q = torch.minimum(self.q1(observations, actions, z), self.q2(observations, actions, z))
            v = self.value(observations, z)
            adv = q - v
        log_probs = self.policy.log_prob(observations, actions, z)
        loss = awr_loss(log_probs, adv, temperature=self.awr_temperature, clamp=self.exp_adv_clamp)
        return loss, log_probs, adv

    # -- full training step ------------------------------------------------
    @torch.no_grad()
    def evaluate_rewards(self, reward_fn: Any, observations: torch.Tensor) -> torch.Tensor:
        """Evaluate the sampled prior reward function ``eta`` on ``observations``.

        ``reward_fn`` may be a callable ``eta(states)`` or an object exposing
        ``.reward(states)`` (the interface used by ``fre.priors``).
        """
        states = observations.detach().cpu().numpy()
        if callable(reward_fn):
            out = reward_fn(states)
        elif hasattr(reward_fn, "reward"):
            out = reward_fn.reward(states)
        else:  # pragma: no cover - defensive
            raise TypeError("reward_fn must be callable or expose .reward(states)")
        out = np.asarray(out, dtype=np.float32).reshape(states.shape[0], -1)
        if out.shape[-1] != 1:
            out = out.mean(axis=-1, keepdims=True)
        return torch.as_tensor(out.reshape(-1), device=observations.device, dtype=torch.float32)

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        z: torch.Tensor,
        rewards: Optional[torch.Tensor] = None,
        reward_fn: Any = None,
        update_policy: bool = True,
    ) -> Dict[str, float]:
        """One IQL iteration on a batch of transitions conditioned on ``z``.

        Rewards are ``r = eta(s)`` (Algorithm 1).  They may be passed directly
        via ``rewards`` or computed from ``reward_fn`` (the sampled reward
        function) evaluated on ``observations``.
        """
        observations = _as_tensor(observations, self.device)
        actions = _as_tensor(actions, self.device)
        next_observations = _as_tensor(next_observations, self.device)
        dones = _as_tensor(dones, self.device).reshape(-1)
        z = _as_tensor(z, self.device)
        if z.dim() == 1:
            z = z.unsqueeze(0)

        if rewards is None:
            if reward_fn is None:
                raise ValueError("either `rewards` or `reward_fn` must be provided")
            rewards = self.evaluate_rewards(reward_fn, observations)
        rewards = _as_tensor(rewards, self.device).reshape(-1)

        metrics: Dict[str, float] = {}

        # --- Q update (Bellman with r = eta(s)) ---
        q_loss, q1, target_q = self.q_loss(
            observations, actions, next_observations, dones, rewards, z
        )
        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        # --- V update (expectile regression on min(Q1, Q2)) ---
        v_loss, v, q_min = self.value_loss(observations, actions, z)
        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()

        metrics["q_loss"] = float(q_loss.detach())
        metrics["v_loss"] = float(v_loss.detach())
        metrics["q_mean"] = float(q1.detach().mean())
        metrics["v_mean"] = float(v.detach().mean())
        metrics["target_q_mean"] = float(target_q.detach().mean())
        metrics["reward_mean"] = float(rewards.detach().mean())

        # --- policy update (AWR) ---
        if update_policy:
            policy_loss, log_probs, adv = self.policy_loss(observations, actions, z)
            self.policy_optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            self.policy_optimizer.step()
            metrics["policy_loss"] = float(policy_loss.detach())
            metrics["log_prob_mean"] = float(log_probs.detach().mean())
            metrics["advantage_mean"] = float(adv.detach().mean())
            metrics["advantage_max"] = float(adv.detach().max())

        # --- target networks (Polyak, rate 0.001) ---
        self.soft_update()

        self.train_steps += 1
        metrics["train_steps"] = float(self.train_steps)
        return metrics

    # Convenience alias matching a common naming convention in training loops.
    train_step = update

    def update_from_batch(
        self,
        batch: Any,
        z: torch.Tensor,
        rewards: Optional[torch.Tensor] = None,
        reward_fn: Any = None,
        update_policy: bool = True,
    ) -> Dict[str, float]:
        """``update`` variant accepting a ``TransitionBatch``-like object / dict."""
        if isinstance(batch, dict):
            observations = batch["observations"]
            actions = batch["actions"]
            next_observations = batch.get("next_observations", batch.get("next_obs"))
            dones = batch.get("dones", batch.get("terminals"))
            rewards = batch.get("rewards") if rewards is None else rewards
        else:
            observations = batch.observations
            actions = batch.actions
            next_observations = getattr(batch, "next_observations", None)
            dones = getattr(batch, "dones", None)
            if rewards is None:
                rewards = getattr(batch, "rewards", None)
        if next_observations is None or dones is None:
            raise ValueError("batch must contain next_observations and dones")
        if isinstance(dones, np.ndarray) and dones.dtype != np.float32:
            dones = dones.astype(np.float32)
        return self.update(
            observations,
            actions,
            next_observations,
            dones,
            z,
            rewards=rewards,
            reward_fn=reward_fn,
            update_policy=update_policy,
        )

    @torch.no_grad()
    def soft_update(self) -> None:
        polyak_update(self.q1_target, self.q1, self.target_update_rate)
        polyak_update(self.q2_target, self.q2, self.target_update_rate)

    # -- acting ------------------------------------------------------------
    @torch.no_grad()
    def select_action(
        self,
        observation: Union[np.ndarray, torch.Tensor],
        z: Union[np.ndarray, torch.Tensor],
        deterministic: bool = True,
    ) -> np.ndarray:
        """Return an action for a single observation given latent ``z``."""
        obs = _as_tensor(observation, self.device).reshape(1, -1)
        latent = _as_tensor(z, self.device).reshape(1, -1)
        action = self.policy.act(obs, latent, deterministic=deterministic)
        return action.reshape(-1).cpu().numpy()

    @torch.no_grad()
    def select_actions(
        self,
        observations: Union[np.ndarray, torch.Tensor],
        z: Union[np.ndarray, torch.Tensor],
        deterministic: bool = True,
    ) -> np.ndarray:
        """Batched action selection for vectorized / batched rollouts."""
        obs = _as_tensor(observations, self.device)
        obs = torch.atleast_2d(obs)
        latent = _as_tensor(z, self.device)
        latent = torch.atleast_2d(latent)
        if latent.shape[0] == 1 and obs.shape[0] > 1:
            latent = latent.expand(obs.shape[0], -1)
        return self.policy.act(obs, latent, deterministic=deterministic).cpu().numpy()

    def extra_repr(self) -> str:
        return (
            f"name={self.name}, state_dim={self.state_dim}, action_dim={self.action_dim}, "
            f"latent_dim={self.latent_dim}, expectile={self.expectile}, "
            f"awr_temperature={self.awr_temperature}, discount={self.discount}, "
            f"target_update_rate={self.target_update_rate}, bellman_target={self.bellman_target}"
        )


# Alias used in some scripts / configs.
IQLAgent = IQL
