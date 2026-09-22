"""Implicit Q-Learning (IQL) with Functional Reward Encoding (FRE) conditioning.

This module implements the *policy learning* half of the FRE pipeline described in
§4.3 "Offline RL with FRE" of Kuba et al., ICML 2024, together with Algorithm 1's
"Train policy" loop and the hyperparameters from Table 3 (Appendix A).

Paper equations implemented here
--------------------------------
The FRE Bellman policy improvement step (§4.3) is::

    Q(s, a, z) <- eta(s) + E_{s' ~ p(. | s, a)} [ max_{a' in A} Q(s', a', z) ]

and the practical implementation uses implicit Q-learning (Kostrikov et al., 2021)
with ``r = eta(s)`` and all three networks (``pi(a | s, z)``, ``Q(s, a, z)``,
``V(s, z)``) conditioned on the 128-dim task latent ``z`` produced by the frozen FRE
encoder (Algorithm 1: "Sample reward function eta; Sample K states for encoder;
Encode into latent vector z ~ p_theta({(s_k^e, eta(s_k^e))}); Train
pi(a|s,z), Q(s,a,z), V(s,z) using IQL with r = eta(s)").

IQL's three losses (Kostrikov et al., 2021), all expectation-maximisations
(hence the minus signs in the optimiser step below):

* value / expectile regression:
      L_V = E[ L_2^tau( Q_target(s, a, z) - V(s, z) ) ],
      L_2^tau(u) = |tau - 1{u < 0}| * u^2,   tau = 0.8  (IQL Expectile, Table 3)
* critic / TD error with the *value* bootstrap (no max over out-of-distribution
  actions):
      L_Q = E[ (r + gamma * (1 - done) * V_target(s', z) - Q(s, a, z))^2 ],
      gamma = 0.88 (Discount Factor, Table 3)
* policy extraction with advantage-weighted regression (AWR):
      L_pi = -E[ exp(beta * (Q(s, a, z) - V(s, z))) * log pi(a | s, z) ],
      beta = 3.0 (AWR Temperature, Table 3)

Target networks are updated with Polyak averaging at rate 0.001
("Target Update Rate", Table 3).

Note on the reward: the paper samples a fresh reward function ``eta ~ p(eta)``
from the prior every iteration and uses ``r = eta(s)``.  The trainer is
responsible for evaluating ``eta`` on the batch states (see
``fre/fre/prior.py``) and encoding ``z`` with the *frozen* FRE encoder; this
module consumes the already-evaluated ``rewards`` and ``z``.  Nothing here
trains the encoder -- phase 2 of Algorithm 1 must keep the eta -> z mapping
stationary.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from fre.fre.encoder import Encoder
from fre.rl.networks import (
    IQLEstimator,
    LOG_STD_MAX,
    LOG_STD_MIN,
    GaussianPolicy,
    QNetwork,
    ValueNetwork,
)

__all__ = [
    "IQL",
    "IQLLearner",
    "expectile_loss",
    "expectile",
    "awr_weights",
    "encode_latent",
    "make_iql",
    "polyak_update",
]


# --------------------------------------------------------------------------------------
# Loss helpers
# --------------------------------------------------------------------------------------
def expectile_loss(diff: torch.Tensor, expectile: float = 0.8) -> torch.Tensor:
    """Asymmetric squared loss ``L_2^tau(u) = |tau - 1{u < 0}| * u^2``.

    This is the value-function objective of IQL (Kostrikov et al., 2021); ``tau`` is
    the "IQL Expectile" of Table 3 (0.8).

    Args:
        diff: ``Q_target(s, a, z) - V(s, z)``, any shape.
        expectile: asymmetry coefficient in ``(0, 1)``.

    Returns:
        Scalar tensor (mean over all elements).
    """
    if not 0.0 < expectile < 1.0:
        raise ValueError(f"expectile must be in (0, 1), got {expectile}")
    weight = torch.where(diff > 0.0, torch.as_tensor(expectile, dtype=diff.dtype, device=diff.device),
                         torch.as_tensor(1.0 - expectile, dtype=diff.dtype, device=diff.device))
    return (weight * diff.pow(2)).mean()


def expectile_loss_elementwise(diff: torch.Tensor, expectile: float = 0.8) -> torch.Tensor:
    """Elementwise (unreduced) variant of :func:`expectile_loss` (kept for diagnostics)."""
    weight = torch.where(diff > 0.0, torch.as_tensor(expectile, dtype=diff.dtype, device=diff.device),
                         torch.as_tensor(1.0 - expectile, dtype=diff.dtype, device=diff.device))
    return weight * diff.pow(2)


# Backwards/compat alias used by some tests.
expectile = expectile_loss


def awr_weights(advantage: torch.Tensor, temperature: float = 3.0, clip: float = 100.0) -> torch.Tensor:
    """Advantage weights ``exp(beta * A(s, a, z))`` for AWR policy extraction.

    ``beta`` is the "AWR Temperature" of Table 3 (3.0).  In the reference IQL
    implementation the temperature multiplies the advantage directly (i.e. it plays
    the role of the inverse temperature in the AWR objective).  The exponential is
    clamped for numerical stability, as in the reference implementation.

    Args:
        advantage: ``Q(s, a, z) - V(s, z)`` (should be detached by the caller).
        temperature: AWR inverse temperature beta.
        clip: maximum weight.

    Returns:
        Weight tensor with the same shape as ``advantage``.
    """
    weights = torch.exp(temperature * advantage)
    return torch.clamp(weights, max=clip)


def polyak_update(target: nn.Module, source: nn.Module, rate: float) -> None:
    """Polyak (exponential moving average) target update, ``theta_t = (1-r) theta_t + r theta``."""
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must be in [0, 1], got {rate}")
    with torch.no_grad():
        for p_t, p in zip(target.parameters(), source.parameters()):
            p_t.data.mul_(1.0 - rate).add_(p.data, alpha=rate)
        for b_t, b in zip(target.buffers(), source.buffers()):
            try:
                b_t.data.copy_(b.data)
            except Exception:  # pragma: no cover - buffers that are not tensors
                pass


# --------------------------------------------------------------------------------------
# Latent encoding helper (Algorithm 1 "Train policy" loop)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def encode_latent(
    encoder: Encoder,
    states: torch.Tensor,
    rewards: torch.Tensor,
    sample: bool = False,
) -> torch.Tensor:
    """Encode a set of ``(s^e, eta(s^e))`` pairs into ``z`` with the FRE encoder.

    Args:
        encoder: frozen FRE encoder (``fre/fre/encoder.py``).
        states: ``(B, K, state_dim)`` or ``(K, state_dim)`` encoder states.
        rewards: ``(B, K)`` or ``(K,)`` reward values ``eta(s^e)``.
        sample: if ``True`` draw ``z ~ p_theta(z | L^e)``; if ``False`` use the
            posterior mean (deterministic, which is what we use for evaluation and
            for stable TD learning).

    Returns:
        ``(B, latent_dim)`` latent task vector.
    """
    encoder.eval()
    dist = encoder(states, rewards)
    z = dist.rsample() if sample else dist.loc
    return z


def _concat_obs_z(obs: torch.Tensor, z: Optional[torch.Tensor]) -> torch.Tensor:
    if z is None:
        return obs
    if z.dim() == 1:
        z = z.unsqueeze(0)
    if z.shape[0] != obs.shape[0]:
        if z.shape[0] == 1:
            z = z.expand(obs.shape[0], -1)
        else:
            raise ValueError(f"z batch size {z.shape[0]} does not match obs batch {obs.shape[0]}")
    return torch.cat([obs, z], dim=-1)


# --------------------------------------------------------------------------------------
# Main IQL learner
# --------------------------------------------------------------------------------------
class IQL:
    """Implicit Q-learning agent with reward-function-conditioned (z) networks.

    Instantiates twin ``Q(s, a, z)``, ``V(s, z)`` and ``pi(a | s, z)`` networks via
    :class:`fre.rl.networks.IQLEstimator` (hidden layers ``[512, 512, 512]``, ReLU),
    keeps target networks for ``Q`` and ``V``, and exposes one ``update`` call per
    outer training step (Algorithm 1, phase 2).

    Args:
        obs_dim: dimensionality of the observation / state (physics-augmented for ExORL).
        action_dim: dimensionality of the action space.
        latent_dim: dimensionality of the FRE task latent ``z`` (128).
        hidden_layers: RL network hidden sizes, ``[512, 512, 512]`` (Table 3).
        activation: RL network activation (ReLU; paper does not state, default ReLU).
        expectile: IQL expectile tau (0.8, Table 3).
        temperature: AWR inverse temperature beta (3.0, Table 3).
        discount: discount factor gamma (0.88, Table 3).
        target_update_rate: Polyak rate for target networks (0.001, Table 3).
        learning_rate: Adam lr (0.0001, Table 3).
        grad_clip_norm: gradient clipping (None disables; default 10.0 as a safe value).
        actor_learning_rate / critic_learning_rate: optional per-component lr overrides.
        tanh_squash: if True (default) the policy is a squashed Gaussian in ``[-1, 1]``
            (matching :class:`fre.rl.networks.GaussianPolicy`); actions passed to the
            critic are then expected to already live in ``[-1, 1]``.
        log_std_min / log_std_max: policy log-std clamp (baseline GC-BC uses -5.0).
        awr_clip: clamp on the AWR advantage weights (reference implementation: 100).
        disadvantage_normalization: divide the AWR weights by their maximum
            (``normalize``: divide by mean) -- optional trick from the reference
            implementation, disabled by default.
        device: torch device or string.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 128,
        hidden_layers: Tuple[int, ...] = (512, 512, 512),
        activation: str = "relu",
        expectile: float = 0.8,
        temperature: float = 3.0,
        discount: float = 0.88,
        target_update_rate: float = 0.001,
        learning_rate: float = 1e-4,
        grad_clip_norm: Optional[float] = 10.0,
        actor_learning_rate: Optional[float] = None,
        critic_learning_rate: Optional[float] = None,
        tanh_squash: bool = True,
        log_std_min: float = LOG_STD_MIN,
        log_std_max: float = LOG_STD_MAX,
        awr_clip: float = 100.0,
        disadvantage_normalization: Optional[str] = None,
        num_qs: int = 2,
        layernorm: bool = False,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.device = torch.device(device) if isinstance(device, str) else device
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_layers = tuple(hidden_layers)
        self.activation = activation

        self.expectile = float(expectile)
        self.temperature = float(temperature)
        self.discount = float(discount)
        self.target_update_rate = float(target_update_rate)
        self.grad_clip_norm = grad_clip_norm
        self.awr_clip = float(awr_clip)
        self.disadvantage_normalization = disadvantage_normalization
        self.tanh_squash = bool(tanh_squash)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.num_updates = 0
        self.last_metrics: Dict[str, float] = {}

        estimator = IQLEstimator(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            latent_dim=self.latent_dim,
            hidden_layers=self.hidden_layers,
            activation=self.activation,
            expectile=self.expectile,
            temperature=self.temperature,
            discount=self.discount,
            target_update_rate=self.target_update_rate,
            tanh_squash=self.tanh_squash,
            layernorm=layernorm,
            num_qs=num_qs,
        )
        self.estimator = estimator
        self.critic: QNetwork = estimator.critic
        self.value: ValueNetwork = estimator.value
        self.policy: GaussianPolicy = estimator.policy

        # Frozen copies used for the Bellman / expectile targets.
        self.target_critic: QNetwork = copy.deepcopy(self.critic)
        self.target_value: ValueNetwork = copy.deepcopy(self.value)
        for param in self.target_critic.parameters():
            param.requires_grad_(False)
        for param in self.target_value.parameters():
            param.requires_grad_(False)

        lr = float(learning_rate)
        critic_lr = float(critic_learning_rate) if critic_learning_rate is not None else lr
        actor_lr = float(actor_learning_rate) if actor_learning_rate is not None else lr
        value_params = []
        value_ids = set()
        for p in self.value.parameters():
            if p.requires_grad:
                value_params.append(p)
                value_ids.add(id(p))
        critic_params = [p for p in self.critic.parameters() if p.requires_grad]
        for p in self.critic.parameters():
            if p.requires_grad:
                value_ids.discard(id(p))
        actor_params = [p for p in self.policy.parameters() if p.requires_grad]

        self.value_optimizer = torch.optim.Adam(value_params, lr=critic_lr)
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=critic_lr)
        self.policy_optimizer = torch.optim.Adam(actor_params, lr=actor_lr)
        self.optimizer = self.critic_optimizer  # convenience alias

        self.to(self.device)

    # ----------------------------------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------------------------------
    def to(self, device: Union[str, torch.device]) -> "IQL":
        self.device = torch.device(device) if isinstance(device, str) else device
        self.critic.to(self.device)
        self.value.to(self.device)
        self.policy.to(self.device)
        self.target_critic.to(self.device)
        self.target_value.to(self.device)
        return self

    # Alias so the object can be used like an nn.Module holder in the trainer.
    def parameters(self):
        return self.estimator.param_groups()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "critic": self.critic.state_dict(),
            "value": self.value.state_dict(),
            "policy": self.policy.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "target_value": self.target_value.state_dict(),
            "num_updates": self.num_updates,
        }

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        self.critic.load_state_dict(state_dict["critic"], strict=strict)
        self.value.load_state_dict(state_dict["value"], strict=strict)
        self.policy.load_state_dict(state_dict["policy"], strict=strict)
        if "target_critic" in state_dict:
            self.target_critic.load_state_dict(state_dict["target_critic"], strict=strict)
        if "target_value" in state_dict:
            self.target_value.load_state_dict(state_dict["target_value"], strict=strict)
        self.num_updates = int(state_dict.get("num_updates", 0))

    @staticmethod
    def _as_tensor(x: Any, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype)
        import numpy as np  # local import: keep numpy optional at module import

        return torch.as_tensor(np.asarray(x), device=device, dtype=dtype)

    def _batch_tensors(
        self,
        batch: Union[Dict[str, Any], Tuple[torch.Tensor, ...]],
        rewards: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        next_z: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Normalise a replay-buffer batch (dict or tuple) into torch tensors.

        Accepted keys: ``observations``/``obs``/``states``, ``actions``,
        ``rewards`` (optional; when omitted ``rewards`` must be passed separately),
        ``next_observations``/``next_obs``, ``terminals``/``dones``, ``z``.
        """
        if isinstance(batch, dict):
            get = lambda *keys: next((batch[k] for k in keys if k in batch and batch[k] is not None), None)  # noqa: E731
            obs = get("observations", "obs", "states", "state")
            actions = get("actions", "action")
            next_obs = get("next_observations", "next_obs", "next_states", "next_state")
            terminals = get("terminals", "dones", "done")
            rew = get("rewards", "reward")
            if z is None:
                z = get("z", "latent", "task_latent")
            next_z = next_z if next_z is not None else get("next_z", "next_latent")
        else:  # tuple of tensors in (obs, action, next_obs, reward/done ...) order
            batch = tuple(batch)
            obs = batch[0]
            actions = batch[1]
            next_obs = batch[2] if len(batch) > 2 else None
            rew = batch[3] if len(batch) > 3 else None
            terminals = batch[4] if len(batch) > 4 else None

        if rewards is not None:
            rew = rewards
        if obs is None or actions is None:
            raise ValueError("batch must contain observations and actions")

        out = {
            "observations": self._as_tensor(obs, self.device),
            "actions": self._as_tensor(actions, self.device),
        }
        if out["actions"].dim() == 1:
            out["actions"] = out["actions"].unsqueeze(-1)
        out["next_observations"] = (
            self._as_tensor(next_obs, self.device) if next_obs is not None else out["observations"]
        )
        if rew is not None:
            out["rewards"] = self._as_tensor(rew, self.device)
            if out["rewards"].dim() == 2 and out["rewards"].shape[-1] == 1:
                out["rewards"] = out["rewards"].squeeze(-1)
        else:
            out["rewards"] = None
        if terminals is not None:
            t = self._as_tensor(terminals, self.device)
            if t.dim() == 2 and t.shape[-1] == 1:
                t = t.squeeze(-1)
            out["terminals"] = t.float()
        else:
            out["terminals"] = torch.zeros_like(out["rewards"]) if out["rewards"] is not None else None
        out["z"] = self._as_tensor(z, self.device) if z is not None else None
        out["next_z"] = self._as_tensor(next_z, self.device) if next_z is not None else None
        return out

    def _log_prob(self, obs: torch.Tensor, action: torch.Tensor, z: Optional[torch.Tensor]) -> torch.Tensor:
        """Log-likelihood ``log pi(a | s, z)`` of a batch of (possibly squashed) actions."""
        out = self.policy.evaluate_actions(obs, action, z)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if isinstance(out, dict):
            out = out.get("log_prob", out.get("log_probability"))
        return out.reshape(-1)

    def _policy_actions(self, obs: torch.Tensor, z: Optional[torch.Tensor], deterministic: bool = False):
        out = self.policy.act(obs, z, deterministic=deterministic)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out

    # ----------------------------------------------------------------------------------
    # Individual updates (IQL, Kostrikov et al., 2021)
    # ----------------------------------------------------------------------------------
    def update_value(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Expectile regression of ``V(s, z)`` onto ``min_j Q_j(s, a, z)`` targets."""
        with torch.no_grad():
            q_target = self.target_critic.q_min(obs, actions, z)
        v = self.value(obs, z)
        diff = q_target - v
        loss = expectile_loss(diff, self.expectile)

        self.value_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip_norm:
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.grad_clip_norm)
        self.value_optimizer.step()
        return {
            "value_loss": float(loss.item()),
            "v_mean": float(v.detach().mean().item()),
            "q_target_mean": float(q_target.mean().item()),
        }

    def update_critic(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        terminals: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        next_z: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """TD update ``|r + gamma (1-done) V_target(s', z) - Q(s, a, z)|^2``.

        ``rewards`` is ``eta(s)`` (the sampled prior reward function evaluated on the
        batch states), as prescribed by the FRE Bellman step in §4.3.
        """
        if next_z is None:
            next_z = z
        with torch.no_grad():
            next_v = self.target_value(next_obs, next_z)
            target_q = rewards + self.discount * (1.0 - terminals) * next_v
        qs = self.critic(obs, actions, z)  # (B, num_qs)
        target = target_q.unsqueeze(-1).expand_as(qs)
        loss = F.mse_loss(qs, target)

        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip_norm:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_norm)
        self.critic_optimizer.step()
        return {
            "critic_loss": float(loss.item()),
            "q_mean": float(qs.detach().mean().item()),
            "q_target_mean": float(target_q.mean().item()),
        }

    def update_policy(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """AWR-weighted maximum-likelihood policy update (``beta = 3.0``)."""
        with torch.no_grad():
            q = self.critic.q_min(obs, actions, z).detach()
            v = self.value(obs, z).detach()
            adv = q - v
            weights = awr_weights(adv, self.temperature, self.awr_clip)
            if self.disadvantage_normalization == "normalize":
                weights = weights / (weights.mean() + 1e-8)
            elif self.disadvantage_normalization == "clamp":
                weights = torch.clamp(weights, max=1.0)

        log_prob = self._log_prob(obs, actions, z)
        loss = -(weights * log_prob).mean()

        self.policy_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip_norm:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
        self.policy_optimizer.step()
        return {
            "policy_loss": float(loss.item()),
            "adv_mean": float(adv.mean().item()),
            "adv_max": float(adv.max().item()),
            "weight_mean": float(weights.mean().item()),
        }

    # ----------------------------------------------------------------------------------
    # Outer step
    # ----------------------------------------------------------------------------------
    def update(
        self,
        batch: Union[Dict[str, Any], Tuple[torch.Tensor, ...]],
        rewards: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        next_z: Optional[torch.Tensor] = None,
        target_update: bool = True,
    ) -> Dict[str, float]:
        """One IQL training iteration (value -> critic -> policy -> target update).

        Args:
            batch: dict/tuple from the replay buffer (actions already normalised to the
                policy's action space, i.e. ``[-1, 1]`` when ``tanh_squash=True``).
            rewards: ``eta(s)`` for the sampled reward function; overrides any
                ``rewards`` entry inside ``batch`` (FRE supplies rewards on the fly).
            z: ``(B, latent_dim)`` (or ``(1, latent_dim)``) task latents for the batch.
            next_z: latent used for the bootstrap term; defaults to ``z`` (single
                reward function per batch, which is how FRE is trained).
            target_update: perform the Polyak target update after this step.

        Returns:
            Dict of scalar metrics for logging.
        """
        t = self._batch_tensors(batch, rewards=rewards, z=z, next_z=next_z)
        if t["rewards"] is None:
            raise ValueError("IQL.update requires rewards: pass `rewards=eta(s)` explicitly")

        obs, act = t["observations"], t["actions"]
        rew, next_obs, term = t["rewards"], t["next_observations"], t["terminals"]
        zz, nz = t["z"], t["next_z"]

        metrics: Dict[str, float] = {}
        metrics.update(self.update_value(obs, act, zz))
        metrics.update(self.update_critic(obs, act, rew, next_obs, term, zz, nz))
        metrics.update(self.update_policy(obs, act, zz))
        if target_update:
            self.update_targets()
        self.num_updates += 1
        self.last_metrics = metrics
        return metrics

    # Alias matching common naming in the FRE repo.
    train_step = update

    def update_targets(self, rate: Optional[float] = None) -> None:
        """Polyak-average the target networks (rate 0.001, Table 3)."""
        rate = self.target_update_rate if rate is None else rate
        polyak_update(self.target_critic, self.critic, rate)
        polyak_update(self.target_value, self.value, rate)

    # ----------------------------------------------------------------------------------
    # Acting / evaluation
    # ----------------------------------------------------------------------------------
    @torch.no_grad()
    def select_action(
        self,
        obs: Any,
        z: Optional[Any] = None,
        deterministic: bool = True,
        clip: bool = True,
    ) -> torch.Tensor:
        """Return ``pi(a | s, z)`` for a single observation (or a batch of them).

        Args:
            obs: ``(obs_dim,)`` or ``(B, obs_dim)`` observation.
            z: ``(latent_dim,)`` or ``(B, latent_dim)`` task latent.
            deterministic: use the policy mean (zero-shot evaluation default).
            clip: clamp the action to ``[-1, 1]`` (only when ``tanh_squash=True``).
        """
        was_1d = False
        obs_t = self._as_tensor(obs, self.device)
        if obs_t.dim() == 1:
            was_1d = True
            obs_t = obs_t.unsqueeze(0)
        z_t = self._as_tensor(z, self.device) if z is not None else None
        if z_t is not None and z_t.dim() == 1:
            z_t = z_t.unsqueeze(0)
        action = self._policy_actions(obs_t, z_t, deterministic=deterministic)
        if clip and self.tanh_squash:
            action = torch.clamp(action, -1.0, 1.0)
        return action.squeeze(0) if was_1d else action

    act = select_action

    @torch.no_grad()
    def value_of(self, obs: Any, z: Optional[Any] = None) -> torch.Tensor:
        obs_t = self._as_tensor(obs, self.device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        z_t = self._as_tensor(z, self.device) if z is not None else None
        return self.value(obs_t, z_t)

    def train(self) -> "IQL":
        self.critic.train()
        self.value.train()
        self.policy.train()
        return self

    def eval(self) -> "IQL":
        self.critic.eval()
        self.value.eval()
        self.policy.eval()
        return self

    def extra_repr(self) -> str:
        return (
            f"obs_dim={self.obs_dim}, action_dim={self.action_dim}, latent_dim={self.latent_dim}, "
            f"expectile={self.expectile}, temperature={self.temperature}, discount={self.discount}, "
            f"target_update_rate={self.target_update_rate}"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}({self.extra_repr()})"


# Common alias: several script/trainer entry points refer to the learner as IQLLearner.
IQLLearner = IQL


# --------------------------------------------------------------------------------------
# Config-driven factory
# --------------------------------------------------------------------------------------
def make_iql(
    config: Any,
    obs_dim: int,
    action_dim: int,
    device: Optional[Union[str, torch.device]] = None,
    **overrides: Any,
) -> IQL:
    """Build an :class:`IQL` learner from a ``fre.config.default.Config``-like object.

    Reads (with paper defaults as fallback): ``latent_dim`` (128), ``rl_hidden_layers``
    ([512,512,512]), ``rl_activation``, ``iql_expectile`` (0.8), ``iql_temperature``
    (3.0), ``discount`` (0.88), ``target_update_rate`` (0.001), ``learning_rate``
    (1e-4), ``rl_num_qs``, ``rl_tanh_squash``, ``log_std_min``, ``log_std_max``,
    ``rl_layernorm``, ``device``.
    """
    kwargs: Dict[str, Any] = dict(
        obs_dim=obs_dim,
        action_dim=action_dim,
        latent_dim=int(getattr(config, "latent_dim", 128)),
        hidden_layers=tuple(getattr(config, "rl_hidden_layers", (512, 512, 512))),
        activation=getattr(config, "rl_activation", "relu"),
        expectile=float(getattr(config, "iql_expectile", 0.8)),
        temperature=float(getattr(config, "iql_temperature", 3.0)),
        discount=float(getattr(config, "discount", 0.88)),
        target_update_rate=float(getattr(config, "target_update_rate", 0.001)),
        learning_rate=float(getattr(config, "learning_rate", 1e-4)),
        grad_clip_norm=getattr(config, "grad_clip_norm", 10.0),
        tanh_squash=bool(getattr(config, "rl_tanh_squash", True)),
        log_std_min=float(getattr(config, "log_std_min", LOG_STD_MIN)),
        log_std_max=float(getattr(config, "log_std_max", LOG_STD_MAX)),
        num_qs=int(getattr(config, "rl_num_qs", 2)),
        layernorm=bool(getattr(config, "rl_layernorm", False)),
        device=device if device is not None else getattr(config, "device", "cpu"),
    )
    kwargs.update(overrides)
    return IQL(**kwargs)
