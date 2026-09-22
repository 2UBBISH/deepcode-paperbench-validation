"""Implicit Q-Learning (IQL) trainer for z-conditioned policies (FRE phase 2).

Reproduces Section 4.3 of *Zero-Shot Reinforcement Learning via Functional Reward
Encodings*:

    Q(s, a, z) <- eta(s) + gamma * E_{s'}[ max_a' Q(s', a', z) ]
    V(s, z)    <- expectile_0.8 regression of Q(s, a, z)
    pi(a|s, z) <- AWR with temperature 3.0

with ``gamma = 0.88``, expectile ``tau = 0.8``, AWR temperature ``3.0`` and a
single target network soft-updated with rate ``0.001``.  The latent task
embedding ``z`` is produced by the (frozen) FRE encoder from K=32
(state, reward) pairs; ``z`` is re-sampled / re-encoded on every iteration so
that the rewards used for the Bellman backup always match the conditioning
latent.

The networks themselves live in :mod:`fre.latent_policy` (``LatentPolicyBundle``
= Q / V / pi with z concatenated to the observations); this module owns the
optimisation logic, the target networks and the reward-prior plumbing.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------------------
# Paper hyper-parameters (Sec. 4.3 + Appendix)
# --------------------------------------------------------------------------------------
DEFAULT_DISCOUNT: float = 0.88
DEFAULT_EXPECTILE: float = 0.8
DEFAULT_AWR_TEMPERATURE: float = 3.0
DEFAULT_AWR_MAX_WEIGHT: float = 100.0
DEFAULT_TARGET_UPDATE_RATE: float = 0.001
DEFAULT_LR: float = 1e-4
DEFAULT_BATCH_SIZE: int = 512
DEFAULT_MAX_GRAD_NORM: float = 1.0
DEFAULT_NUM_CANDIDATE_ACTIONS: int = 10

__all__ = [
    "IQLConfig",
    "IQLTrainer",
    "IQL",
    "IQLAgent",
    "expectile_loss",
    "awr_weights",
    "asymmetric_l2_loss",
    "DEFAULT_DISCOUNT",
    "DEFAULT_EXPECTILE",
    "DEFAULT_AWR_TEMPERATURE",
    "DEFAULT_TARGET_UPDATE_RATE",
]


# --------------------------------------------------------------------------------------
# Loss helpers
# --------------------------------------------------------------------------------------
def expectile_loss(diff: torch.Tensor, expectile: float = DEFAULT_EXPECTILE) -> torch.Tensor:
    """Asymmetric (expectile) squared error used for the V function.

    ``diff = Q_target(s, a, z) - V(s, z)``.  Positive differences are weighted by
    ``expectile`` and negative ones by ``1 - expectile`` (tau = 0.8 puts more
    weight on the upper tail, i.e. V tracks a high expectile of Q).
    """
    weight = torch.where(diff > 0.0, float(expectile), 1.0 - float(expectile))
    return weight * diff.pow(2)


#: Explicit alias for readability at call sites.
asymmetric_l2_loss = expectile_loss


def awr_weights(
    advantage: torch.Tensor,
    temperature: float = DEFAULT_AWR_TEMPERATURE,
    max_weight: float = DEFAULT_AWR_MAX_WEIGHT,
) -> torch.Tensor:
    """``exp(temperature * advantage)`` clamped to ``max_weight`` (IQL Eq. 10)."""
    w = torch.exp(float(temperature) * advantage)
    if max_weight is not None:
        w = torch.clamp(w, max=float(max_weight))
    return w.detach()


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
@dataclass
class IQLConfig:
    """Hyper-parameters for phase-2 z-conditioned IQL training."""

    # IQL core (paper values)
    discount: float = DEFAULT_DISCOUNT
    expectile: float = DEFAULT_EXPECTILE
    awr_temperature: float = DEFAULT_AWR_TEMPERATURE
    awr_max_weight: float = DEFAULT_AWR_MAX_WEIGHT
    target_update_rate: float = DEFAULT_TARGET_UPDATE_RATE

    # optimisation
    lr: float = DEFAULT_LR
    batch_size: int = DEFAULT_BATCH_SIZE
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM

    # Bellman backup style.  The paper writes the backup with a max over
    # candidate next actions; the canonical IQL implementation bootstraps from
    # V_target(s', z).  "v" == canonical IQL (default), "max_q" == max_a' Q.
    bellman_target: str = "v"
    num_candidate_actions: int = DEFAULT_NUM_CANDIDATE_ACTIONS

    # how often (in update steps) to re-sample the reward function / re-encode z
    reward_resample_interval: int = 1

    # target networks
    use_target_network: bool = True
    soft_target_update: bool = True

    # misc
    log_interval: int = 1000
    num_encoder_samples: int = 8  # encoder set-encoding at eval time

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------------------
# Small utilities for the reward-prior interface
# --------------------------------------------------------------------------------------
def _tensor(x: Any, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def _call_reward_sampler(reward_sampler: Any, states: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Evaluate/da sample reward function on ``states``.

    Accepts, in order of preference:
      * objects exposing ``sample_reward(states)`` / ``evaluate(states)`` /
        ``reward(states)`` / ``forward(states)`` (the FRE prior families in
        ``fre/rewards``),
      * a plain callable.

    Returns a dict with at least ``rewards`` and optionally ``dones`` /
    ``context_states`` / ``context_rewards``.
    """
    out: Any = None
    for name in ("sample_reward", "evaluate", "reward", "forward", "sample"):
        fn = getattr(reward_sampler, name, None)
        if callable(fn):
            try:
                out = fn(states)
            except TypeError:
                continue
            break
    if out is None:
        if callable(reward_sampler):
            out = reward_sampler(states)
        else:  # pragma: no cover - defensive
            raise TypeError(
                "reward_sampler must be callable or expose sample_reward/evaluate/reward"
            )

    if isinstance(out, Mapping):
        result = dict(out)
    elif isinstance(out, (tuple, list)):
        result = {"rewards": out[0]}
        if len(out) > 1 and out[1] is not None:
            result["dones"] = out[1]
        if len(out) > 2 and out[2] is not None:
            result["context_states"] = out[2]
        if len(out) > 3 and out[3] is not None:
            result["context_rewards"] = out[3]
    else:
        result = {"rewards": out}

    if "dones" not in result and "done" in result:
        result["dones"] = result["done"]
    return result


# --------------------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------------------
class IQLTrainer:
    """Phase-2 trainer: frozen FRE encoder (optional) + trainable z-conditioned IQL.

    Parameters
    ----------
    policy:
        A :class:`fre.latent_policy.LatentPolicyBundle` (or any module exposing
        ``q_value`` / ``value`` / ``policy`` / ``policy_log_prob``).
    encoder:
        Frozen FRE encoder used to map a K=32 reward context to ``z``.  May be
        ``None`` when the batch already carries a latent (``batch["latents"]``)
        or an externally-supplied reward function.
    reward_sampler:
        Reward-prior object from :mod:`fre.rewards` used to sample reward
        functions eta and (optionally) their encoding contexts.
    config:
        :class:`IQLConfig`.
    """

    def __init__(
        self,
        policy: nn.Module,
        encoder: Optional[nn.Module] = None,
        reward_sampler: Optional[Any] = None,
        config: Optional[IQLConfig] = None,
        device: Optional[Union[str, torch.device]] = None,
        latent_dim: int = 128,
        lr: Optional[float] = None,
    ) -> None:
        self.config = config or IQLConfig()
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.policy = policy.to(self.device)
        self.encoder = encoder
        if self.encoder is not None:
            self.encoder = self.encoder.to(self.device)
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad_(False)

        self.reward_sampler = reward_sampler
        self.latent_dim = int(latent_dim)

        # Target networks: single target bundle for both Q and V (Sec 4.3).
        self.target_policy = copy.deepcopy(policy).to(self.device)
        self.target_policy.eval()
        for p in self.target_policy.parameters():
            p.requires_grad_(False)

        lr = float(self.config.lr if lr is None else lr)
        params = [p for p in self.policy.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(params, lr=lr)

        self._step = 0
        self._last_log: Dict[str, float] = {}
        self._rng = torch.Generator(device="cpu")

    # ------------------------------------------------------------------ helpers
    @property
    def step(self) -> int:
        return self._step

    def train(self) -> "IQLTrainer":
        self.policy.train()
        return self

    def eval(self) -> "IQLTrainer":
        self.policy.eval()
        return self

    def _as_batch(self, batch: Union[Mapping[str, Any], Sequence[Any]]) -> Dict[str, torch.Tensor]:
        """Normalise a training batch into a device/dtype-correct dict."""
        if not isinstance(batch, Mapping):
            obs, act, rew, next_obs, done = batch[:5]
            batch = {
                "observations": obs,
                "actions": act,
                "rewards": rew,
                "next_observations": next_obs,
                "dones": done,
            }
        out: Dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            if value is None or not torch.is_tensor(value):
                if value is None:
                    continue
                value = torch.as_tensor(value)
            out[key] = value.to(device=self.device, dtype=torch.float32)
        if "next_observations" not in out and "next_observations" in batch:  # pragma: no cover
            out["next_observations"] = out["next_observations"]
        # Accept common aliases.
        if "observations" not in out:
            for alias in ("obs", "states", "observation"):
                if alias in out:
                    out["observations"] = out[alias]
                    break
        if "actions" not in out:
            for alias in ("action", "acts"):
                if alias in out:
                    out["actions"] = out[alias]
                    break
        if "rewards" not in out:
            for alias in ("reward", "r"):
                if alias in out:
                    out["rewards"] = out[alias]
                    break
        if "next_observations" not in out:
            for alias in ("next_obs", "next_states", "next_observation"):
                if alias in out:
                    out["next_observations"] = out[alias]
                    break
        if "dones" not in out and "done" in out:
            out["dones"] = out["done"]
        if "rewards" in out and out["rewards"].dim() > 1:
            out["rewards"] = out["rewards"].reshape(out["rewards"].shape[0], -1).mean(-1)
        if "dones" in out:
            out["dones"] = out["dones"].reshape(out["dones"].shape[0], -1)
            if out["dones"].shape[-1] > 1:
                out["dones"] = out["dones"].amax(-1)
            else:
                out["dones"] = out["dones"].squeeze(-1)
        return out

    # ------------------------------------------------- encoding z from rewards
    @torch.no_grad()
    def encode_latent(
        self,
        context_states: torch.Tensor,
        context_rewards: torch.Tensor,
        use_mean: bool = False,
    ) -> torch.Tensor:
        """Encode a (state, reward) context into z with the frozen encoder."""
        if self.encoder is None:  # pragma: no cover - defensive
            raise RuntimeError("encode_latent requires an encoder")
        cs = _tensor(context_states, self.device)
        cr = _tensor(context_rewards, self.device)
        if hasattr(self.encoder, "encode"):
            return self.encoder.encode(cs, cr, use_mean=use_mean)
        z, _, _ = self.encoder.sample(cs, cr)  # pragma: no cover - fallback
        return z

    @torch.no_grad()
    def sample_rewards_and_latent(
        self,
        observations: torch.Tensor,
        next_observations: Optional[torch.Tensor] = None,
        batch: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """Sample eta ~ p(eta), evaluate eta(s), eta(s'), and encode z.

        Returns ``(rewards, next_rewards, latents, info)``.
        """
        obs = _tensor(observations, self.device)
        next_obs = _tensor(next_observations, self.device) if next_observations is not None else obs
        info: Dict[str, Any] = {}

        if self.reward_sampler is None:
            if batch is None:  # pragma: no cover - defensive
                raise RuntimeError("sample_rewards_and_latent needs a reward_sampler or a batch")
            rewards = batch["rewards"]
            next_rewards = batch.get("next_rewards", rewards)
            latents = batch.get("latents")
            return rewards, next_rewards, latents, info

        # Reward functions are defined on the *current* states of the batch.
        out = _call_reward_sampler(self.reward_sampler, obs)
        rewards = _tensor(out["rewards"], self.device)
        if rewards.dim() > 1:
            rewards = rewards.reshape(rewards.shape[0], -1).mean(-1)

        # eps/done mask (goal-reaching rewards terminate at the goal).
        dones = out.get("dones", out.get("done"))
        if dones is not None:
            dones = _tensor(dones, self.device)
            dones = dones.reshape(dones.shape[0], -1)
            dones = dones.amax(-1) if dones.shape[-1] > 1 else dones.squeeze(-1)
            info["dones"] = dones

        # Next-state reward: goal-reaching is evaluated on s' so the terminal
        # transition receives the reward for reaching s'; linear/MLP rewards are
        # per-state functions without termination.
        next_rewards = rewards
        eta = out.get("reward_fn", out.get("eta"))
        if eta is not None and hasattr(eta, "__call__"):
            with torch.no_grad():
                nr = eta(next_obs)
                if isinstance(nr, (tuple, list)):
                    nr = nr[0]
                nr = _tensor(nr, self.device)
            if nr.dim() > 1:
                nr = nr.reshape(nr.shape[0], -1).mean(-1)
            next_rewards = nr

        latents = out.get("latents")
        ctx_states = out.get("context_states")
        ctx_rewards = out.get("context_rewards")
        if latents is None and ctx_states is not None and ctx_rewards is not None and self.encoder is not None:
            latents = self.encode_latent(ctx_states, ctx_rewards)
        elif latents is None and "latents" in out:
            latents = out["latents"]

        if latents is not None:
            latents = _tensor(latents, self.device)
            if latents.dim() == 3 and latents.shape[1] == 1:
                latents = latents.squeeze(1)
        return rewards, next_rewards, latents, info

    # ------------------------------------------------------------------ updates
    def update(
        self,
        batch: Union[Mapping[str, torch.Tensor], Sequence[Any]],
        latents: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
        dones: Optional[torch.Tensor] = None,
        sample_reward: Optional[bool] = None,
    ) -> Dict[str, float]:
        """One IQL update step on a batch of transitions.

        The reward function is (re-)sampled and ``z`` (re-)encoded each call
        unless explicit ``latents`` / ``rewards`` are given.
        """
        data = self._as_batch(batch)
        obs = data["observations"]
        actions = data["actions"]
        next_obs = data.get("next_observations", obs)

        info: Dict[str, Any] = {}
        resample = self.config.reward_resample_interval > 0 and (
            self._step % max(1, self.config.reward_resample_interval) == 0
        )
        if sample_reward is None:
            sample_reward = self.reward_sampler is not None and resample

        if rewards is None and sample_reward:
            rewards, next_rewards, sampled_z, info = self.sample_rewards_and_latent(
                obs, next_obs, data
            )
            if latents is None:
                latents = sampled_z
        else:
            rewards = data["rewards"] if rewards is None else _tensor(rewards, self.device)
            if rewards.dim() > 1:
                rewards = rewards.reshape(rewards.shape[0], -1).mean(-1)

        if dones is None:
            dones = info.get("dones", data.get("dones"))
        if dones is None:
            dones = torch.zeros_like(rewards)
        else:
            dones = _tensor(dones, self.device).reshape(-1)

        if latents is None:
            latents = data.get("latents")
        if latents is not None:
            latents = _tensor(latents, self.device)

        metrics = self._update_from_tensors(obs, actions, rewards, next_obs, dones, latents)
        self._maybe_soft_update_targets()
        self._step += 1
        self._last_log = metrics
        return metrics

    def _update_from_tensors(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
        latents: Optional[torch.Tensor],
    ) -> Dict[str, float]:
        cfg = self.config
        z = latents
        if z is None:
            z = torch.zeros(obs.shape[0], self.latent_dim, device=self.device, dtype=obs.dtype)

        # ------------------------------------------------ V update (expectile)
        with torch.no_grad():
            q_vals = self.policy.q_value(obs, actions, z)
        v_pred = self.policy.value(obs, z).reshape(-1)
        v_loss = expectile_loss(q_vals.reshape(-1).detach() - v_pred, cfg.expectile).mean()

        self.optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        if cfg.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.policy.value_parameters()] if hasattr(self.policy, "value_parameters")
                else list(self.policy.v_network.parameters()) if hasattr(self.policy, "v_network")
                else list(self.policy.parameters()),
                cfg.max_grad_norm,
            )
        self.optimizer.step()

        # ------------------------------------------------ Q update (Bellman)
        with torch.no_grad():
            if cfg.bellman_target == "v" or not cfg.use_target_network:
                bootstrap = self.target_policy.value(next_obs, z).reshape(-1)
            else:
                cand_actions, _ = self.policy.sample(next_obs, z) if hasattr(self.policy, "sample") else (
                    None,
                    None,
                )
                if cand_actions is None:  # pragma: no cover - defensive
                    bootstrap = self.target_policy.value(next_obs, z).reshape(-1)
                else:
                    cand = torch.stack([cand_actions] * int(cfg.num_candidate_actions), dim=0)
                    nc = next_obs.unsqueeze(0).expand_as(cand[..., : next_obs.shape[-1]])
                    nz = z.unsqueeze(0).expand(cand.shape[0], -1)
                    q_cand = self.target_policy.q_value(nc, cand, nz)
                    bootstrap = q_cand.max(dim=0).values.reshape(-1)
            target_q = rewards.reshape(-1) + cfg.discount * (1.0 - dones.reshape(-1)) * bootstrap

        q_pred = self.policy.q_value(obs, actions, z).reshape(-1)
        q_loss = F.mse_loss(q_pred, target_q)

        self.optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        if cfg.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                list(self.policy.q_network.parameters())
                if hasattr(self.policy, "q_network")
                else list(self.policy.parameters()),
                cfg.max_grad_norm,
            )
        self.optimizer.step()

        # ------------------------------------------------ policy (AWR)
        with torch.no_grad():
            adv = self.policy.q_value(obs, actions, z).reshape(-1) - self.policy.value(obs, z).reshape(-1)
            weights = awr_weights(adv, cfg.awr_temperature, cfg.awr_max_weight)
            weights = weights / (weights.mean() + 1e-8)

        log_prob = self.policy.policy_log_prob(obs, actions, z).reshape(-1)
        policy_loss = -(weights * log_prob).mean()

        self.optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()
        if cfg.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                list(self.policy.policy_network.parameters())
                if hasattr(self.policy, "policy_network")
                else list(self.policy.parameters()),
                cfg.max_grad_norm,
            )
        self.optimizer.step()

        return {
            "v_loss": float(v_loss.detach().cpu()),
            "q_loss": float(q_loss.detach().cpu()),
            "policy_loss": float(policy_loss.detach().cpu()),
            "q_mean": float(q_pred.detach().mean().cpu()),
            "v_mean": float(v_pred.detach().mean().cpu()),
            "target_q_mean": float(target_q.detach().mean().cpu()),
            "reward_mean": float(rewards.detach().mean().cpu()),
            "awr_weight_mean": float(weights.detach().mean().cpu()),
            "adv_mean": float(adv.detach().mean().cpu()),
            "bellman_error": float((target_q - q_pred).detach().abs().mean().cpu()),
        }

    # ------------------------------------------------------------- target nets
    @torch.no_grad()
    def _maybe_soft_update_targets(self) -> None:
        cfg = self.config
        if not cfg.use_target_network:
            return
        rate = float(cfg.target_update_rate)
        for tp, p in zip(self.target_policy.parameters(), self.policy.parameters()):
            tp.data.mul_(1.0 - rate).add_(p.data, alpha=rate)
        # Keep buffers (e.g. running stats) in sync when present.
        for tb, b in zip(self.target_policy.buffers(), self.policy.buffers()):
            if tb.shape == b.shape:
                tb.data.copy_(b.data)

    @torch.no_grad()
    def hard_update_targets(self) -> None:
        for tp, p in zip(self.target_policy.parameters(), self.policy.parameters()):
            tp.data.copy_(p.data)

    # ---------------------------------------------------------------- rollouts
    @torch.no_grad()
    def select_action(
        self,
        obs: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Return an action for ``obs`` conditioned on latent ``z``."""
        self.policy.eval()
        o = _tensor(obs, self.device)
        if o.dim() == 1:
            o = o.unsqueeze(0)
        if z is None:
            zt = torch.zeros(o.shape[0], self.latent_dim, device=self.device)
        else:
            zt = _tensor(z, self.device)
            if zt.dim() == 1:
                zt = zt.unsqueeze(0)
            if zt.shape[0] == 1 and o.shape[0] > 1:
                zt = zt.expand(o.shape[0], -1)
        action, _ = self.policy.sample(o, zt, deterministic=deterministic)
        return action

    # ------------------------------------------------------------- checkpoint
    def state_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy.state_dict(),
            "target_policy": self.target_policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.as_dict(),
            "step": self._step,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True) -> None:
        self.policy.load_state_dict(state_dict["policy"], strict=strict)
        if "target_policy" in state_dict:
            self.target_policy.load_state_dict(state_dict["target_policy"], strict=strict)
        if "optimizer" in state_dict:
            try:
                self.optimizer.load_state_dict(state_dict["optimizer"])
            except ValueError:  # pragma: no cover - param groups changed
                pass
        self._step = int(state_dict.get("step", self._step))

    def extra_repr(self) -> str:
        c = self.config
        return (
            f"gamma={c.discount}, expectile={c.expectile}, "
            f"awr_temp={c.awr_temperature}, tau={c.target_update_rate}, lr={c.lr}"
        )


# Plan alias
IQL = IQLTrainer


# --------------------------------------------------------------------------------------
# Convenience agent: frozen encoder + trainable IQL
# --------------------------------------------------------------------------------------
class IQLAgent:
    """Bundles a frozen FRE encoder with an :class:`IQLTrainer`.

    ``update`` re-samples a reward function from ``reward_sampler`` and re-encodes
    ``z`` every iteration (freezing the encoder keeps the eta -> z map stationary
    for stable multitask TD learning, Sec. 4.3).
    """

    def __init__(
        self,
        encoder: Optional[nn.Module],
        policy: nn.Module,
        reward_sampler: Optional[Any] = None,
        config: Optional[IQLConfig] = None,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        self.encoder = encoder
        self.reward_sampler = reward_sampler
        self.trainer = IQLTrainer(
            policy=policy,
            encoder=encoder,
            reward_sampler=reward_sampler,
            config=config,
            device=device,
        )
        self.device = self.trainer.device

    # -- delegation -----------------------------------------------------------------
    @property
    def policy(self) -> nn.Module:
        return self.trainer.policy

    @property
    def step(self) -> int:
        return self.trainer.step

    def update(self, batch, **kwargs) -> Dict[str, float]:
        return self.trainer.update(batch, **kwargs)

    def select_action(self, obs, z=None, deterministic: bool = False):
        return self.trainer.select_action(obs, z, deterministic=deterministic)

    @torch.no_grad()
    def z_from_context(self, context_states, context_rewards, use_mean: bool = False):
        return self.trainer.encode_latent(context_states, context_rewards, use_mean=use_mean)

    def state_dict(self):
        return self.trainer.state_dict()

    def load_state_dict(self, sd, strict: bool = True):
        self.trainer.load_state_dict(sd, strict=strict)
