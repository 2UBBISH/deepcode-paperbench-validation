"""SAPG: Split and Aggregate Policy Gradients -- Algorithm 1 (Section 4.6).

This module implements the full SAPG training loop:

    * N massively-parallel environments are split into M contiguous blocks of
      ``N / M`` environments (Algorithm 1 / Section 4.6, Figure 3).
    * Each block ``j`` is driven by its own policy ``pi_j`` -- the *same* shared
      actor backbone ``B_theta`` and the *same* shared critic backbone
      ``C_psi``, conditioned on block-local latents ``phi_j`` (Section 4.4 and
      the Addendum clarification).
    * One policy (the *leader*, ``i = 1``) is updated with its own on-policy
      data **and** importance-sampled off-policy data from the followers
      ``X = {2, ..., M}``; the followers only use their own on-policy data
      (Section 4.3).
    * The leader's off-policy batch is sub-sampled so that equal amounts of
      on-policy and off-policy data are used per mini-batch update
      (Section 4.3), i.e. ``|D'_1| = |D_1|`` and ``lambda = 1``.
    * Followers receive an additional entropy term ``sigma * (i - 1) * H(pi)``
      while the leader has none (Section 4.5, Eq. 10).

Losses (following the paper's equations; the paper writes the off-policy
surrogate as a *reward* to be gained, Eq. 1, and adds it to the on-policy
*loss* ``L_on``, Eq. 2 -- we normalise the sign of the off-policy term so that
both are minimised losses, exactly as ``L_on`` is the negated surrogate):

    L_off(pi_i; X) = 1/|X| sum_{j in X} E_{(s,a)~pi_j}[
                        min( r_{pi_i}(s,a),
                             clip(r_{pi_i}(s,a), mu (1 - eps), mu (1 + eps)) )
                        A^{pi_i,old}(s,a) ]                       (Eq. 1)
    L(pi_i)        = L_on(pi_i) + lambda * L_off(pi_i; X)          (Eq. 2)
    V_on^target    = sum_{k=t}^{t+2} gamma^{k-t} r_k + gamma^3 V_old(s_{t+3})
                                                                   (Eq. 3)
    V_off^target   = r_t + gamma V_old(s'_{t+1})                   (Eq. 4)
    L^critic       = L_on^critic + lambda * L_off^critic           (Eqs. 5-6)

with ``r_{pi_i}(s,a) = pi_i(s,a) / pi_j(s,a)`` and the off-policy correction
``mu = pi_{i,old}(s,a) / pi_j(s,a)``.  When ``i = j`` then ``pi_j = pi_{i,old}``
and the update reduces to the usual on-policy PPO update.

Everything is deliberately decoupled from IsaacGym: the trainer only requires an
environment object exposing ``reset()``/``step(actions)`` and a policy object
exposing ``act``/``evaluate_actions``/``get_value`` (see
``sapg.models.actor.ActorCritic``).
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from ..utils.config import NUM_POLICIES, TOTAL_ENVS, SAPGConfig, ensure_dir
from ..buffers.rollout_buffer import BufferSet, RolloutBuffer
from .rollout import BlockManager, RolloutCollector, collect_data

# --------------------------------------------------------------------------- #
# Optional imports: the SAPG-specific loss modules and the PPO loss helpers.
# The trainer keeps self-contained fallbacks so that it is always runnable.
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - exercised depending on availability
    from .ppo import compute_bounds_loss, compute_ppo_loss, compute_value_loss
except Exception:  # pragma: no cover
    compute_bounds_loss = compute_ppo_loss = compute_value_loss = None  # type: ignore

try:  # pragma: no cover
    from ..losses.off_policy_loss import off_policy_loss  # type: ignore
except Exception:  # pragma: no cover
    off_policy_loss = None  # type: ignore

try:  # pragma: no cover
    from ..buffers.rollout_buffer import (  # type: ignore
        build_off_policy_batch,
        compute_n_step_targets,
        mu_from_logprobs,
    )
except Exception:  # pragma: no cover
    build_off_policy_batch = compute_n_step_targets = mu_from_logprobs = None  # type: ignore


__all__ = [
    "SAPGTrainer",
    "SAPGUpdateStats",
    "train_sapg",
    "LeaderFollowerScheme",
    "SymmetricScheme",
    "NoAggregationScheme",
    "make_scheme",
]


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #
_OBS_KEYS = ("obs", "observations", "states", "obses")
_ACTION_KEYS = ("actions", "action", "acts")
_LOGPROB_KEYS = ("logprobs", "log_prob", "log_probs", "old_logprobs", "action_log_probs")
_VALUE_KEYS = ("values", "value", "old_values", "value_old")
_ADV_KEYS = ("advantages", "advantage", "adv")
_TARGET_KEYS = ("value_targets", "returns", "return_targets", "targets", "value_target")
_MASK_KEYS = ("masks", "mask", "rnn_masks")
_REWARD_KEYS = ("rewards", "reward", "rew")
_DONE_KEYS = ("dones", "done", "terminals", "dones_mask")
_NEXT_OBS_KEYS = ("obs_next", "next_obs", "next_observations", "obs_prime", "next_states")
_HIDDEN_KEYS = ("hidden_states", "rnn_states", "hiddens")


def _first(mapping: Dict[str, Any], keys: Sequence[str]) -> Optional[Any]:
    """Return the first present value among ``keys`` (torch tensors only)."""
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        if torch.is_tensor(value):
            return value
        if isinstance(value, (list, tuple)) and value and torch.is_tensor(value[0]):
            return value[0]
    return None


def _as_dict(obj: Any) -> Dict[str, Any]:
    """Normalise a buffer/batch-like object into a plain ``dict`` of tensors."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    data = getattr(obj, "data", None)
    if isinstance(data, dict):
        return data
    return {}


def _to_flat(tensor: Optional[torch.Tensor], feature_dim: Optional[int] = None) -> Optional[torch.Tensor]:
    """Flatten the (time, env, ...) leading dims into a single sample dimension."""
    if tensor is None:
        return None
    if tensor.dim() <= 1:
        return tensor.reshape(-1)
    return tensor.reshape(-1, *tensor.shape[2:])


def _mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred.reshape(-1) - target.reshape(-1)) ** 2)


def _clipped_surrogate(ratio: torch.Tensor, advantages: torch.Tensor,
                       lower: torch.Tensor, upper: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``min(r A, clip(r, lower, upper) A)`` and the clip indicator."""
    unclipped = ratio * advantages
    clipped_ratio = torch.clamp(ratio, min=lower, max=upper)
    clipped = clipped_ratio * advantages
    return torch.min(unclipped, clipped), (clipped_ratio != ratio).float()


def _fallback_ppo_loss(ratio: torch.Tensor, advantages: torch.Tensor,
                       clip_epsilon: float) -> Dict[str, torch.Tensor]:
    surrogate, clip_mask = _clipped_surrogate(
        ratio, advantages,
        torch.full_like(ratio, 1.0 - clip_epsilon),
        torch.full_like(ratio, 1.0 + clip_epsilon),
    )
    return {
        "policy_loss": -surrogate.mean(),
        "clip_frac": clip_mask.mean(),
        "ratio_mean": ratio.mean(),
        "ratio_max": ratio.max(),
    }


def _fallback_value_loss(values: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
    loss = _mse(values, targets)
    return {"value_loss": loss, "value_loss_scaled": loss, "value_error_abs": (values - targets).abs().mean()}


def _fallback_bounds_loss(action_mean: torch.Tensor, coefficient: float) -> torch.Tensor:
    """Action-bound regularisation (keeps the pre-tanh mean inside [-1, 1])."""
    if action_mean is None or action_mean.numel() == 0:
        return torch.zeros((), device=action_mean.device if torch.is_tensor(action_mean) else None)
    excess = torch.relu(action_mean.abs() - 1.0)
    return coefficient * excess.sum(dim=-1).mean()


def _gae_td_lambda(rewards: torch.Tensor, dones: torch.Tensor, values: torch.Tensor,
                   last_values: Optional[torch.Tensor], gamma: float, tau: float,
                   n_step: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generalised advantage estimation plus n-step value targets.

    ``rewards``/``dones``/``values`` are time-major ``[T, E]`` tensors.  The
    returned advantages are GAE(lambda=tau) and the targets are the n-step
    returns of Eq. 3 (``n_step = 3`` in the paper) bootstrapped with
    ``gamma^n V_old(s_{t+n})``.
    """
    horizon = rewards.shape[0]
    device = rewards.device
    next_values = torch.zeros_like(values)
    next_values[:-1] = values[1:]
    if last_values is None:
        last_values = torch.zeros_like(values[0])
    next_values[-1] = last_values.reshape(-1)

    # --- n-step targets (Eq. 3) --------------------------------------------
    targets = torch.zeros_like(values)
    for t in range(horizon):
        acc = torch.zeros_like(rewards[t])
        discount = 1.0
        bootstrap = next_values[t]
        for k in range(n_step):
            idx = t + k
            if idx >= horizon:
                break
            acc = acc + discount * rewards[idx]
            discount = discount * gamma
            if idx + 1 < horizon:
                bootstrap = values[idx + 1]
            else:
                bootstrap = last_values.reshape(-1)
            # stop accumulating rewards after a terminal transition
            if bool(dones[idx].max().item() > 0.5) if dones.numel() else False:
                pass
        targets[t] = acc + (discount if not n_step else discount) * 0.0
        # correct bootstrap handling: use the value *after* the n-step window
        end = t + n_step
        if end < horizon:
            boot = values[end]
        else:
            boot = last_values.reshape(-1)
        # mask out steps that crossed an episode boundary
        mask = torch.ones_like(rewards[t])
        for k in range(n_step):
            if t + k < horizon:
                mask = mask * (1.0 - dones[t + k])
        targets[t] = acc * 1.0 + (gamma ** n_step) * boot * mask + acc * 0.0
        # discards rewards that occurred after an episode boundary
        acc2 = torch.zeros_like(rewards[t])
        discount = 1.0
        alive = torch.ones_like(rewards[t])
        for k in range(n_step):
            idx = t + k
            if idx >= horizon:
                break
            acc2 = acc2 + discount * rewards[idx] * alive
            alive = alive * (1.0 - dones[idx])
            discount = discount * gamma
        targets[t] = acc2 + (gamma ** n_step) * boot * alive

    # --- GAE ---------------------------------------------------------------
    advantages = torch.zeros_like(values)
    running = torch.zeros_like(values[0])
    for t in reversed(range(horizon)):
        delta = rewards[t] + gamma * next_values[t] * (1.0 - dones[t]) - values[t]
        running = delta + gamma * tau * (1.0 - dones[t]) * running
        advantages[t] = running
    return advantages, targets


class _PolicyView:
    """Binds a shared policy to one block/policy index ``j``.

    The shared actor/critic are conditioned on ``phi_j`` and (for the
    entropy-exploration variant) on a per-block sigma row, both of which are
    selected by ``policy_index=j``.  The rollout collector only knows about a
    plain ``act()`` interface, so this tiny adapter guarantees that block ``j``
    always sees its own latent.
    """

    def __init__(self, policy: Any, index: int, phi: Optional[torch.Tensor] = None) -> None:
        self._policy = policy
        self.index = index
        self.phi = phi

    # -- pass-through -------------------------------------------------------
    def __getattr__(self, name: str) -> Any:  # pragma: no cover - trivial
        return getattr(self.__dict__["_policy"], name)

    def _call(self, name: str, obs: Any, *args, **kwargs):
        fn = getattr(self._policy, name)
        kwargs.setdefault("policy_index", self.index)
        if self.phi is not None:
            kwargs.setdefault("phi", self.phi)
        try:
            return fn(obs, *args, **kwargs)
        except TypeError:
            stripped = {k: v for k, v in kwargs.items()
                        if k not in ("policy_index", "phi")}
            return fn(obs, *args, **stripped)

    # -- policy API ---------------------------------------------------------
    def act(self, obs, phi=None, hidden_state=None, masks=None, deterministic=False, **kwargs):
        kwargs.setdefault("hidden_state", hidden_state)
        kwargs.setdefault("masks", masks)
        kwargs.setdefault("deterministic", deterministic)
        return self._call("act", obs, **kwargs)

    def evaluate_actions(self, obs, actions, phi=None, hidden_state=None, masks=None, **kwargs):
        kwargs.setdefault("hidden_state", hidden_state)
        kwargs.setdefault("masks", masks)
        return self._call("evaluate_actions", obs, actions, **kwargs)

    def get_value(self, obs, phi=None, hidden_state=None, masks=None, **kwargs):
        kwargs.setdefault("hidden_state", hidden_state)
        kwargs.setdefault("masks", masks)
        return self._call("get_value", obs, **kwargs)


# --------------------------------------------------------------------------- #
# Aggregation schemes (Section 4.2 / 4.3, Figure 4)
# --------------------------------------------------------------------------- #
class _BaseScheme:
    name = "base"

    def __init__(self, num_policies: int = NUM_POLICIES, leader_index: int = 1,
                 off_policy_weight: float = 1.0, subsample: bool = True) -> None:
        self.num_policies = int(num_policies)
        self.leader_index = int(leader_index)          # 1-based, as in the paper
        self.off_policy_weight = float(off_policy_weight)
        self.subsample = bool(subsample)

    @property
    def leader(self) -> int:
        """0-based index of the leader."""
        return self.leader_index - 1

    def data_sources(self, i: int) -> List[int]:
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"{type(self).__name__}(M={self.num_policies}, leader={self.leader_index}, "
                f"lambda={self.off_policy_weight}, subsample={self.subsample})")


class LeaderFollowerScheme(_BaseScheme):
    """Section 4.3: leader ``i = 1`` uses ``X = {2, ..., M}``; followers ``X = {}``."""

    name = "leader_follower"

    def data_sources(self, i: int) -> List[int]:
        if i != self.leader:
            return []
        return [j for j in range(self.num_policies) if j != self.leader]


class SymmetricScheme(_BaseScheme):
    """Section 4.2 / Figure 4 (right): every policy uses all others."""

    name = "symmetric"

    def data_sources(self, i: int) -> List[int]:
        return [j for j in range(self.num_policies) if j != i]


class NoAggregationScheme(_BaseScheme):
    """Ablation: every policy is purely on-policy (``X = {}`` for all ``i``)."""

    name = "none"

    def data_sources(self, i: int) -> List[int]:
        return []


_SCHEMES = {
    "leader_follower": LeaderFollowerScheme,
    "leader-follower": LeaderFollowerScheme,
    "leaderfollow": LeaderFollowerScheme,
    "asymmetric": LeaderFollowerScheme,
    "symmetric": SymmetricScheme,
    "sym": SymmetricScheme,
    "none": NoAggregationScheme,
    "no_off_policy": NoAggregationScheme,
    "on_policy": NoAggregationScheme,
}


def make_scheme(name: str = "leader_follower", num_policies: int = NUM_POLICIES,
                leader_index: int = 1, off_policy_weight: float = 1.0,
                subsample: bool = True) -> _BaseScheme:
    """Factory for the aggregation variants of Section 4.2/4.3.

    Falls back on the (equivalent) implementations in
    :mod:`sapg.aggregation` when that package is importable, so that the
    ablation entry points share a single scheme definition.
    """
    try:  # pragma: no cover - depends on package availability
        from ..aggregation import make_aggregation  # type: ignore

        return make_aggregation(            # type: ignore[return-value]
            name,
            num_policies=num_policies,
            leader_index=leader_index,
            off_policy_weight=off_policy_weight,
            subsample=subsample,
        )
    except Exception:
        pass
    key = str(name).strip().lower().replace(" ", "_")
    cls = _SCHEMES.get(key)
    if cls is None:
        raise ValueError(f"unknown aggregation scheme: {name!r}")
    return cls(num_policies=num_policies, leader_index=leader_index,
               off_policy_weight=off_policy_weight, subsample=subsample)


# --------------------------------------------------------------------------- #
# Update statistics
# --------------------------------------------------------------------------- #
@dataclass
class SAPGUpdateStats:
    """Aggregated statistics for a single SAPG outer iteration."""

    policy_loss: float = 0.0
    off_policy_loss: float = 0.0
    value_loss: float = 0.0
    value_loss_off: float = 0.0
    entropy: float = 0.0
    total_loss: float = 0.0
    kl: float = 0.0
    clip_frac: float = 0.0
    off_clip_frac: float = 0.0
    ratio_mean: float = 1.0
    mu_mean: float = 1.0
    grad_norm: float = 0.0
    learning_rate: float = 0.0
    per_policy_policy_loss: List[float] = field(default_factory=list)
    per_policy_entropy: List[float] = field(default_factory=list)
    samples: int = 0
    extra: Dict[str, float] = field(default_factory=dict)

    def as_dict(self, prefix: str = "") -> Dict[str, float]:
        out: Dict[str, float] = {}
        for key in ("policy_loss", "off_policy_loss", "value_loss", "value_loss_off",
                    "entropy", "total_loss", "kl", "clip_frac", "off_clip_frac",
                    "ratio_mean", "mu_mean", "grad_norm", "learning_rate", "samples"):
            out[f"{prefix}{key}"] = float(getattr(self, key))
        for idx, value in enumerate(self.per_policy_policy_loss):
            out[f"{prefix}policy_loss/{idx}"] = float(value)
        for idx, value in enumerate(self.per_policy_entropy):
            out[f"{prefix}entropy/{idx}"] = float(value)
        for key, value in self.extra.items():
            out[f"{prefix}{key}"] = float(value)
        return out


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
class SAPGTrainer:
    """Split-and-aggregate policy-gradient trainer (Algorithm 1, Section 4.6).

    Parameters
    ----------
    config:
        :class:`sapg.utils.config.SAPGConfig` with the data-splitting,
        aggregation and optimisation hyper-parameters.
    policy:
        A single shared policy (``sapg.models.actor.ActorCritic``) built with
        ``num_policies = config.num_policies``.  One shared backbone is used for
        every block and the per-block latents ``phi_j`` select the behaviour
        (Section 4.4).
    env:
        Massively parallel environment exposing ``reset()`` and
        ``step(actions)`` (IsaacGym wrapper or a test double).
    policies:
        Optional ``list`` of ``M`` policies instead of a single shared one.
    """

    def __init__(
        self,
        config: SAPGConfig,
        policy: Any = None,
        env: Any = None,
        policies: Optional[Sequence[Any]] = None,
        optimizers: Optional[Tuple[Any, Any]] = None,
        logger: Optional[Any] = None,
        device: Optional[Any] = None,
        block_manager: Optional[BlockManager] = None,
        aggregation: Optional[Any] = None,
        critic_n_step: int = 3,
        entropy_maximize: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        self.config = config
        self.device = torch.device(device or getattr(config, "device", "cpu"))
        self.logger = logger
        self.num_policies = int(getattr(config, "num_policies", NUM_POLICIES) or NUM_POLICIES)
        self.num_envs = int(getattr(config, "num_envs", TOTAL_ENVS) or TOTAL_ENVS)
        self.horizon_length = int(getattr(config, "horizon_length", 16) or 16)
        self.gamma = float(getattr(config, "gamma", 0.99))
        self.tau = float(getattr(config, "tau", 0.95))          # GAE lambda
        self.clip_epsilon = float(getattr(config, "clip_epsilon", 0.1))
        self.critic_coefficient = float(getattr(config, "critic_coefficient", 4.0))
        self.bounds_loss_coefficient = float(getattr(config, "bounds_loss_coefficient", 1e-4))
        self.off_policy_weight = float(getattr(config, "off_policy_weight", 1.0))  # lambda
        self.entropy_coefficient = float(getattr(config, "entropy_coefficient", 0.0))  # sigma
        self.mini_epochs = int(getattr(config, "mini_epochs", 2) or 2)
        self.minibatch_size_multiplier = float(getattr(config, "minibatch_size_multiplier", 4) or 4)
        self.sequence_length = int(getattr(config, "lstm_sequence_length", 16) or 16)
        self.critic_n_step = int(critic_n_step)
        self.subsample_off_policy = bool(getattr(config, "subsample_off_policy", True))
        self.kl_threshold = float(getattr(config, "kl_threshold", 0.016))
        self.grad_norm = float(getattr(config, "grad_norm", 1.0))
        self.grad_norm_enabled = bool(getattr(config, "grad_norm", 1.0))
        self.normalize_advantages = bool(getattr(config, "normalize_advantage", True))
        self.normalize_off_policy_advantage = bool(
            getattr(config, "normalize_off_policy_advantage", True))
        self.entropy_maximize = (
            bool(getattr(config, "entropy_maximize", True))
            if entropy_maximize is None else bool(entropy_maximize)
        )
        self.recurrent = bool(getattr(config, "use_lstm", False))
        self.seed = int(getattr(config, "seed", 0))

        # ---- aggregation controller (Section 4.2/4.3) ---------------------
        if aggregation is None:
            aggregation = make_scheme(
                getattr(config, "aggregation", "leader_follower"),
                num_policies=self.num_policies,
                leader_index=int(getattr(config, "leader_index", 1)),
                off_policy_weight=self.off_policy_weight,
                subsample=self.subsample_off_policy,
            )
        self.scheme = aggregation
        self.leader_index = int(getattr(self.scheme, "leader", int(getattr(config, "leader_index", 1)) - 1))

        # ---- environment blocks / rollout collector -----------------------
        self.env = env
        self.block_manager = block_manager or BlockManager(
            num_envs=self.num_envs, num_policies=self.num_policies,
            leader_index=self.leader_index + 1,
        )
        self.block_size = int(self.block_manager.block_size)
        self.collector = RolloutCollector(self.env, config,
                                          block_manager=self.block_manager,
                                          device=self.device) if env is not None else None

        # ---- shared policy -------------------------------------------------
        if policies is not None:
            self.policies = list(policies)
            self.policy = self.policies[0]
        else:
            self.policy = policy if policy is not None else self._build_policy(config)
            self.policies = [self.policy] * self.num_policies
        if hasattr(self.policy, "to"):
            self.policy.to(self.device)

        # ---- per-block latent phi_j ---------------------------------------
        self.phis: List[Optional[torch.Tensor]] = []
        for j in range(self.num_policies):
            phi = None
            for attr in ("phi", "phi_for"):
                fn = getattr(self.policy, attr, None)
                if callable(fn):
                    try:
                        phi = fn(j)
                        break
                    except Exception:
                        phi = None
            self.phis.append(phi)
        self.views: List[_PolicyView] = [
            _PolicyView(self.policies[j], j, self.phis[j]) for j in range(self.num_policies)
        ]

        # ---- optimisation ---------------------------------------------------
        self.actor_lr = float(getattr(config, "learning_rate", 1e-4))
        self.critic_lr = float(getattr(config, "critic_learning_rate", None) or self.actor_lr)
        self.optimizer_name = str(getattr(config, "optimizer", "adam") or "adam").lower()
        self.adam_betas = tuple(getattr(config, "adam_betas", (0.9, 0.999)))
        self.adam_eps = float(getattr(config, "adam_eps", 1e-8))
        self.actor_optimizer, self.critic_optimizer = (optimizers if optimizers is not None
                                                       else self._build_optimizers())
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(self.seed)

        self.obs: Optional[torch.Tensor] = None
        self.hidden_states: Any = None
        self.update_index = 0
        self.total_samples = 0

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _build_policy(config: SAPGConfig) -> Any:
        """Instantiate the shared actor/critic policy from the configuration."""
        from ..models.actor import ActorCritic

        try:
            return ActorCritic(config=config)
        except Exception:  # pragma: no cover - explicit construction fallback
            return ActorCritic(
                obs_dim=int(getattr(config, "obs_dim", 60)),
                action_dim=int(getattr(config, "action_dim", 23)),
                phi_dim=int(getattr(config, "phi_dim", 0) or 0),
                num_policies=int(getattr(config, "num_policies", NUM_POLICIES) or NUM_POLICIES),
                mlp_units=tuple(getattr(config, "actor_mlp_units", (768, 512, 256))),
                activation=str(getattr(config, "actor_activation", "elu")),
                use_lstm=bool(getattr(config, "use_lstm", False)),
                lstm_hidden_size=int(getattr(config, "lstm_hidden_size", 768)),
                lstm_num_layers=int(getattr(config, "lstm_num_layers", 1)),
                per_block_sigma=bool(getattr(config, "per_block_sigma", False)),
                action_scale=float(getattr(config, "action_scale", 1.0)),
            )

    def _make_optimizer(self, params: Sequence[torch.nn.Parameter], lr: float):
        params = [p for p in params if p.requires_grad]
        if not params:
            return None
        if self.optimizer_name in ("adamw",):
            return torch.optim.AdamW(params, lr=lr, betas=self.adam_betas, eps=self.adam_eps)
        return torch.optim.Adam(params, lr=lr, betas=self.adam_betas, eps=self.adam_eps)

    def _build_optimizers(self) -> Tuple[Any, Any]:
        """Actor optimizer (backbone + mu + sigma + phi) and critic optimizer."""
        policy_params = list(self.policy.parameters()) if hasattr(self.policy, "parameters") else []

        def _collect(*names: str) -> List[torch.nn.Parameter]:
            out: List[torch.nn.Parameter] = []
            for name in names:
                fn = getattr(self.policy, name, None)
                if callable(fn):
                    try:
                        out.extend(list(fn()))
                    except Exception:
                        pass
            return out

        actor_params = _collect("actor_parameters")
        critic_params = _collect("critic_parameters")
        # ``phi_j`` lives on the actor and belongs to the policy objective
        # (Section 4.4: "the parameters phi_j are only updated with the
        # objective for that particular policy").  A single backward pass over
        # the summed loss guarantees that property, because L_i only ever
        # touches row i of the latent table.
        latent_params = _collect("latent_parameters", "phi_parameters")

        seen: set = set()

        def _dedup(params: Sequence[torch.nn.Parameter]) -> List[torch.nn.Parameter]:
            out = []
            for p in params:
                if p is None or id(p) in seen or not p.requires_grad:
                    continue
                seen.add(id(p))
                out.append(p)
            return out

        actor_params = _dedup(actor_params + latent_params)
        critic_params = _dedup(critic_params)
        if not actor_params and not critic_params:
            shared = _dedup(policy_params)
            optim = self._make_optimizer(shared, self.actor_lr)
            return optim, optim
        actor_opt = self._make_optimizer(actor_params or policy_params, self.actor_lr)
        critic_opt = self._make_optimizer(critic_params, self.critic_lr)
        if actor_opt is None:
            actor_opt = self._make_optimizer(list(self.policy.parameters()), self.actor_lr)
        if critic_opt is None:
            critic_opt = actor_opt
        return actor_opt, critic_opt

    # --------------------------------------------------------------- helpers
    @property
    def leader(self) -> int:
        """0-based index of the leader policy (paper: ``i = 1``)."""
        return int(self.leader_index)

    def phi(self, j: int) -> Optional[torch.Tensor]:
        """Latent ``phi_j`` of block ``j`` (``None`` when no conditioning)."""
        if 0 <= j < len(self.phis):
            return self.phis[j]
        return None

    def _entropy_coefficient_for(self, j: int) -> float:
        """``sigma * (i - 1)`` for the ``i``-th policy; the leader gets none.

        With 0-based indices ``j`` the paper's ``sigma (i - 1)`` (Eq. 10)
        becomes ``sigma * j``, which is automatically zero for the leader
        ``j = 0`` (Section 4.5: "The leader doesn't have any entropy loss").
        """
        if self.entropy_coefficient == 0.0:
            return 0.0
        return self.entropy_coefficient * float(j)

    # ------------------------------------------------------------- collection
    def collect(self, deterministic: bool = False, obs: Optional[torch.Tensor] = None):
        """Roll out each block with its own policy (Algorithm 1, line 2-3)."""
        if self.collector is None:
            raise RuntimeError("SAPGTrainer needs an environment to collect rollouts")
        if obs is None:
            obs = self.obs
        buffers, obs, hidden_states, metrics = collect_data(
            policies=self.views,
            env=self.env,
            config=self.config,
            obs=obs,
            phis=self.phis,
            hidden_states=self.hidden_states,
            block_manager=self.block_manager,
            deterministic=deterministic,
        )
        self.obs = obs
        self.hidden_states = hidden_states
        return buffers, metrics

    # ----------------------------------------------------- buffer extraction
    def _raw(self, buffer: RolloutBuffer) -> Dict[str, torch.Tensor]:
        """Fetch the time-major tensors out of a ``RolloutBuffer``."""
        out: Dict[str, torch.Tensor] = {}
        containers = [buffer, _as_dict(buffer)]
        for name in ("data", "buffers", "storage", "buffer", "tensors"):
            inner = getattr(buffer, name, None)
            if isinstance(inner, dict):
                containers.append(inner)
            elif hasattr(inner, "__dict__"):
                containers.append(vars(inner))
        for key in set(_OBS_KEYS + _ACTION_KEYS + _LOGPROB_KEYS + _VALUE_KEYS + _ADV_KEYS +
                       _TARGET_KEYS + _MASK_KEYS + _REWARD_KEYS + _DONE_KEYS + _NEXT_OBS_KEYS):
            if key in out:
                continue
            for container in containers:
                value = container.get(key) if hasattr(container, "get") else getattr(container, key, None)
                if torch.is_tensor(value):
                    out[key] = value
                    break

        # last/bootstrapped values (for the n-step target at the horizon edge)
        for name in ("last_values", "last_value", "bootstrap_values", "_last_values"):
            value = getattr(buffer, name, None)
            if torch.is_tensor(value):
                out["last_values"] = value
                break
        return out

    def _ensure_time_major(self, tensor: torch.Tensor) -> torch.Tensor:
        """Guarantee the ``(time, env, ...)`` orientation of a stored tensor."""
        if tensor.dim() < 2:
            return tensor
        horizon = self.horizon_length
        if tensor.shape[0] == horizon or tensor.shape[1] != horizon:
            return tensor
        return tensor.transpose(0, 1)

    def _prepare_buffers(self, buffers: Sequence[RolloutBuffer]) -> List[Dict[str, torch.Tensor]]:
        """Compute GAE advantages and n-step targets, return flat per-policy dicts."""
        self.raw: List[Dict[str, torch.Tensor]] = []
        self.flats: List[Dict[str, torch.Tensor]] = []

        for j, buffer in enumerate(buffers):
            raw = self._raw(buffer)
            tensormap: Dict[str, torch.Tensor] = {}
            aliases = {
                "obs": _OBS_KEYS, "actions": _ACTION_KEYS, "logprobs": _LOGPROB_KEYS,
                "values": _VALUE_KEYS, "advantages": _ADV_KEYS, "value_targets": _TARGET_KEYS,
                "masks": _MASK_KEYS, "rewards": _REWARD_KEYS, "dones": _DONE_KEYS,
                "obs_next": _NEXT_OBS_KEYS,
            }
            for canonical, keys in aliases.items():
                tensor = _first(raw, keys)
                if tensor is not None:
                    tensormap[canonical] = self._ensure_time_major(tensor)
            if "last_values" in raw:
                tensormap["last_values"] = raw["last_values"]

            rewards = tensormap.get("rewards")
            dones = tensormap.get("dones")
            values = tensormap.get("values")
            if dones is not None and dones.dim() == 2 and dones.dtype != torch.float32:
                dones = dones.float()

            advantages = tensormap.get("advantages")
            targets = tensormap.get("value_targets")
            compute_gae = getattr(buffer, "compute_gae", None)
            if advantages is None and compute_gae is not None and rewards is not None:
                for args, kwargs in ((None, {"gamma": self.gamma, "tau": self.tau}),
                                     ((self.gamma, self.tau), {}),
                                     ((self.gamma,), {"tau": self.tau})):
                    try:
                        if args is None:
                            compute_gae(**kwargs)
                        else:
                            compute_gae(*args, **kwargs)
                        break
                    except TypeError:
                        continue
                    except Exception:
                        break
                fresh = self._raw(buffer)
                advantages = _first(fresh, _ADV_KEYS)
                targets = targets if targets is not None else _first(fresh, _TARGET_KEYS)

            if advantages is None and rewards is not None and values is not None and dones is not None:
                advantages, computed_targets = _gae_td_lambda(
                    rewards, dones, values, tensormap.get("last_values"),
                    self.gamma, self.tau, self.critic_n_step)
                if targets is None:
                    targets = computed_targets

            if targets is None and advantages is not None and values is not None:
                targets = advantages + values

            flat: Dict[str, torch.Tensor] = {}
            for key, tensor in tensormap.items():
                flat[key] = _to_flat(tensor)

            if advantages is not None:
                adv_flat = _to_flat(advantages)
                if self.normalize_advantages and adv_flat.numel() > 1:
                    adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std(unbiased=False) + 1e-8)
                flat["advantages"] = adv_flat
            if targets is not None:
                flat["value_targets"] = _to_flat(targets)

            # derive the next observation when the buffer did not store it
            if "obs_next" not in flat and "obs" in tensormap and "dones" in tensormap:
                obs = tensormap["obs"]
                obs_next = torch.empty_like(obs)
                obs_next[:-1] = obs[1:]
                obs_next[-1] = obs[-1]
                flat["obs_next"] = _to_flat(obs_next)

            if "masks" not in flat or flat["masks"] is None:
                n = flat.get("obs").shape[0] if flat.get("obs") is not None else 0
                flat["masks"] = torch.ones(n, 1, device=flat["obs"].device if n else None) if n else None

            self.raw.append(tensormap)
            self.flats.append(flat)
        return self.flats

    # ------------------------------------------------- off-policy data fusion
    def _concat_flats(self, indices: Sequence[int], keys: Sequence[str]) -> Dict[str, torch.Tensor]:
        merged: Dict[str, torch.Tensor] = {}
        for key in keys:
            tensors = [self.flats[j][key] for j in indices if key in self.flats[j]]
            if not tensors:
                continue
            if tensors[0].dim() == 1:
                merged[key] = torch.cat([t.reshape(-1) for t in tensors], dim=0)
            else:
                merged[key] = torch.cat([t for t in tensors], dim=0)
        return merged

    @torch.no_grad()
    def _build_off_policy_data(self, buffers: Sequence[RolloutBuffer]) -> Dict[int, Dict[str, torch.Tensor]]:
        """Build the leader's augmented dataset ``D'_1`` (and, for the symmetric
        ablation, ``D'_i`` for every policy).

        For each policy ``i`` with sources ``X`` (Section 4.2/4.3):
          * flatten and concatenate the follower buffers,
          * sub-sample uniformly so that ``|D'_i| = |D_i|`` by default
            ("we subsample the off-policy data for the leader such that we use
            equal amounts of on-policy and off-policy data" -- Section 4.3),
          * evaluate the (old, i.e. pre-update) policy ``pi_{i,old}`` on those
            samples to obtain ``mu = pi_{i,old}(s,a) / pi_j(s,a)``  (Eq. 1) as
            well as the one-step advantage ``A^{pi_i,old}`` built from the
            critic's 1-step return (Eq. 4).
        """
        keys = ("obs", "actions", "logprobs", "rewards", "dones", "masks", "obs_next")
        out: Dict[int, Dict[str, torch.Tensor]] = {}
        for i in range(self.num_policies):
            sources = [s for s in self.scheme.data_sources(i) if 0 <= s < self.num_policies]
            if not sources:
                continue
            merged = self._concat_flats(sources, keys)
            if "obs" not in merged:
                continue
            n_total = int(merged["obs"].shape[0])
            n_target = int(self.flats[i]["obs"].shape[0])
            source_ids = torch.cat([torch.full((int(self.flats[s]["obs"].shape[0]),), s,
                                    dtype=torch.long) for s in sources if "obs" in self.flats[s]])
            if self.subsample_off_policy and n_total > n_target:
                sel = torch.randperm(n_total, generator=self._generator)[:n_target]
                merged = {k: v[sel] for k, v in merged.items()}
                source_ids = source_ids[sel]

            view = self.views[i]
            obs = merged["obs"]
            actions = merged["actions"]
            behaviour_logprob = merged.get("logprobs")
            result = self._evaluate(view, obs, actions, masks=merged.get("masks"))
            old_logprob = result.get("logprobs")
            old_values = result.get("values")

            if behaviour_logprob is None:
                behaviour_logprob = old_logprob
            if mu_from_logprobs is not None:
                try:
                    mu = mu_from_logprobs(old_logprob, behaviour_logprob)
                except Exception:
                    mu = torch.exp(old_logprob - behaviour_logprob)
            else:
                mu = torch.exp(old_logprob - behaviour_logprob)

            next_values = None
            if "obs_next" in merged and old_values is not None:
                next_result = self._evaluate(view, merged["obs_next"],
                                             torch.zeros_like(actions) if actions is not None else None,
                                             masks=merged.get("masks"))
                next_values = next_result.get("values")
            rewards = merged.get("rewards")
            dones = merged.get("dones")
            if rewards is not None and next_values is not None and old_values is not None:
                done_mask = (1.0 - dones.float()) if dones is not None else 1.0
                advantage = rewards.reshape(-1) + self.gamma * next_values.reshape(-1) * (
                    done_mask.reshape(-1) if torch.is_tensor(done_mask) else 1.0) - old_values.reshape(-1)
                value_target = rewards.reshape(-1) + self.gamma * next_values.reshape(-1) * (
                    done_mask.reshape(-1) if torch.is_tensor(done_mask) else 1.0)
            else:
                advantage = merged.get("advantages")
                value_target = merged.get("value_targets")
            if advantage is None:
                continue

            if self.normalize_off_policy_advantage and advantage.numel() > 1:
                normalized = advantage.clone()
                for source in torch.unique(source_ids):
                    mask = source_ids == source
                    if int(mask.sum()) > 1:
                        group = advantage[mask]
                        normalized[mask] = (group - group.mean()) / (group.std(unbiased=False) + 1e-8)
                advantage = normalized

            batch = dict(merged)
            batch["logprobs"] = behaviour_logprob.reshape(-1)
            batch["mu"] = mu.reshape(-1)
            batch["advantages"] = advantage.reshape(-1)
            if value_target is not None:
                batch["value_targets"] = value_target.reshape(-1)
            batch["source_policy"] = source_ids
            out[i] = batch
        return out

    # ------------------------------------------------------------ evaluation
    def _evaluate(self, view: _PolicyView, obs: torch.Tensor, actions: Optional[torch.Tensor],
                  masks: Optional[torch.Tensor] = None, hidden_state: Any = None,
                  no_grad: bool = False) -> Dict[str, torch.Tensor]:
        """Call the policy and normalise its output keys."""
        context = torch.no_grad() if no_grad else _nullcontext()
        with context:
            out: Any = None
            attempts = (
                dict(hidden_state=hidden_state, masks=masks),
                dict(masks=masks),
                dict(),
            )
            for kwargs in attempts:
                fn = getattr(view, "evaluate_actions", None)
                if fn is None:
                    break
                try:
                    out = fn(obs, actions, **kwargs)
                    break
                except TypeError:
                    continue
            if out is None:
                out = view.act(obs, hidden_state=hidden_state, masks=masks)
        mapping = _as_dict(out) if not isinstance(out, dict) else out
        if not mapping and out is not None:
            mapping = {}
        if not isinstance(out, dict):
            for attr in ("logprobs", "values", "entropy", "action_mean", "mu", "hidden_state"):
                value = getattr(out, attr, None)
                if value is not None:
                    mapping[attr] = value
        result: Dict[str, torch.Tensor] = {}
        result["logprobs"] = _first(mapping, ("logprobs", "log_prob", "log_probs", "action_log_probs"))
        result["values"] = _first(mapping, ("values", "value", "vpred"))
        result["entropy"] = _first(mapping, ("entropy", "entropies"))
        result["action_mean"] = _first(mapping, ("action_mean", "mu", "mean", "pre_tanh"))
        if result["values"] is None:
            get_value = getattr(view, "get_value", None)
            if callable(get_value):
                try:
                    value = get_value(obs, masks=masks, hidden_state=hidden_state)
                except TypeError:
                    value = get_value(obs)
                if torch.is_tensor(value):
                    result["values"] = value
                elif isinstance(value, dict):
                    result["values"] = _first(value, ("values", "value"))
        return result

    # --------------------------------------------------------------- losses
    def _on_policy_losses(self, j: int, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """On-policy PPO loss (Eq. 2) plus the entropy term of Section 4.5."""
        result = self._evaluate(self.views[j], batch["obs"], batch["actions"],
                                masks=batch.get("masks"))
        logprob = result["logprobs"]
        old_logprob = batch["logprobs"]
        ratio = torch.exp(logprob - old_logprob)
        advantages = batch["advantages"]
        if compute_ppo_loss is not None:
            try:
                info = compute_ppo_loss(ratio, advantages, self.clip_epsilon,
                                        old_logprobs=old_logprob, new_logprobs=logprob)
            except TypeError:
                info = compute_ppo_loss(ratio, advantages, self.clip_epsilon)
            info = _as_dict(info)
            policy_loss = _first(info, ("policy_loss",)) if info else None
            if policy_loss is None:
                policy_loss = _fallback_ppo_loss(ratio, advantages, self.clip_epsilon)["policy_loss"]
        else:
            info = _fallback_ppo_loss(ratio, advantages, self.clip_epsilon)
            policy_loss = info["policy_loss"]

        entropy = result.get("entropy")
        entropy_coefficient = self._entropy_coefficient_for(j)
        loss = policy_loss
        if entropy is not None and entropy_coefficient != 0.0:
            entropy_mean = entropy.mean()
            # ``L(pi_i) = L_on(pi_i) + sigma (i - 1) H`` (Eq. 10): the entropy
            # term is added with the sign that increases the entropy, i.e. the
            # follower explores more as sigma grows ("Followers with large
            # entropy losses tend to explore more actions even if they are
            # suboptimal" -- Section 4.5).
            loss = loss - entropy_coefficient * entropy_mean if self.entropy_maximize \
                else loss + entropy_coefficient * entropy_mean
        if result.get("action_mean") is not None and self.bounds_loss_coefficient:
            if compute_bounds_loss is not None:
                try:
                    loss = loss + compute_bounds_loss(result["action_mean"], self.bounds_loss_coefficient)
                except Exception:
                    loss = loss + _fallback_bounds_loss(result["action_mean"], self.bounds_loss_coefficient)
            else:
                loss = loss + _fallback_bounds_loss(result["action_mean"], self.bounds_loss_coefficient)

        values = result["values"]
        targets = batch.get("value_targets")
        value_loss = torch.zeros((), device=batch["obs"].device)
        if values is not None and targets is not None:
            if compute_value_loss is not None:
                try:
                    vinfo = _as_dict(compute_value_loss(values, targets, coefficient=1.0))
                    value_loss = _first(vinfo, ("value_loss",)) if vinfo else None
                except Exception:
                    value_loss = None
                if value_loss is None:
                    value_loss = _fallback_value_loss(values, targets)["value_loss"]
            else:
                value_loss = _fallback_value_loss(values, targets)["value_loss"]

        return {
            "loss": loss,
            "policy_loss": policy_loss.detach(),
            "value_loss": value_loss.detach() if torch.is_tensor(value_loss) else value_loss,
            "value_loss_scaled": value_loss,
            "entropy": entropy.detach().mean() if entropy is not None else torch.zeros(()),
            "ratio_mean": ratio.detach().mean(),
            "clip_frac": (ratio.detach() - 1.0).abs().gt(self.clip_epsilon).float().mean(),
            "kl": ((logprob - old_logprob).detach()).mean() ** 2 * 0.5,
        }

    def _off_policy_losses(self, i: int, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Importance-sampled off-policy loss ``L_off(pi_i; X)`` (Eq. 1)."""
        result = self._evaluate(self.views[i], batch["obs"], batch["actions"],
                                masks=batch.get("masks"))
        logprob = result["logprobs"]
        behaviour_logprob = batch["logprobs"]
        ratio = torch.exp(logprob - behaviour_logprob)          # pi_i / pi_j
        mu = batch["mu"]                                        # pi_{i,old} / pi_j
        advantages = batch["advantages"]
        lower = mu * (1.0 - self.clip_epsilon)
        upper = mu * (1.0 + self.clip_epsilon)
        surrogate, clip_mask = _clipped_surrogate(ratio, advantages, lower, upper)
        policy_loss = -surrogate.mean()

        values = result["values"]
        targets = batch.get("value_targets")
        value_loss = torch.zeros((), device=batch["obs"].device)
        if values is not None and targets is not None:
            value_loss = _fallback_value_loss(values, targets)["value_loss"]

        return {
            "loss": policy_loss,
            "policy_loss": policy_loss.detach(),
            "value_loss_scaled": value_loss,
            "value_loss": value_loss.detach(),
            "ratio_mean": ratio.detach().mean(),
            "clip_frac": clip_mask.detach().mean(),
            "mu_mean": mu.detach().mean(),
            "kl": ((logprob - behaviour_logprob).detach()).mean() ** 2 * 0.5,
        }

    # ----------------------------------------------------------- minibatching
    def _minibatch_groups(self, n: int, minibatch_size: int) -> List[torch.Tensor]:
        """Shuffled sample groups; contiguous chunks when the policy is recurrent."""
        if n <= 0:
            return []
        if self.recurrent and self.sequence_length and self.sequence_length < n:
            chunks = math.ceil(n / self.sequence_length)
            order = torch.randperm(chunks, generator=self._generator)
            per_batch = max(1, int(round(minibatch_size / float(self.sequence_length))))
            groups: List[torch.Tensor] = []
            for start in range(0, chunks, per_batch):
                pieces = []
                for c in order[start:start + per_batch].tolist():
                    pieces.append(torch.arange(c * self.sequence_length,
                                               min(n, (c + 1) * self.sequence_length)))
                groups.append(torch.cat(pieces))
            return groups
        perm = torch.randperm(n, generator=self._generator)
        return [perm[i:i + minibatch_size] for i in range(0, n, minibatch_size)]

    def _select(self, flat: Dict[str, torch.Tensor], index: torch.Tensor) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for key, value in flat.items():
            if torch.is_tensor(value) and value.dim() >= 1 and value.shape[0] == index.new_tensor(0).numel() + 0:
                pass
            if torch.is_tensor(value) and value.dim() >= 1:
                try:
                    if value.shape[0] == int(index.max()) + 1 or value.shape[0] > int(index.max()):
                        out[key] = value[index]
                        continue
                except Exception:
                    pass
            out[key] = value
        return out

    # ------------------------------------------------------------------ steps
    def train_epochs(self, buffers: Sequence[RolloutBuffer],
                     off_policy: Optional[Dict[int, Dict[str, torch.Tensor]]] = None
                     ) -> SAPGUpdateStats:
        """Mini-batch gradient descent over the summed objective (Algorithm 1)."""
        stats = SAPGUpdateStats()
        if not self.flats:
            self._prepare_buffers(buffers)
        off_policy = off_policy if off_policy is not None else self._build_off_policy_data(buffers)

        minibatch_size = max(1, int(self.block_size * self.minibatch_size_multiplier))
        samples_per_policy = int(self.flats[self.leader]["obs"].shape[0])
        num_minibatches = max(1, math.ceil(samples_per_policy / minibatch_size))

        # off-policy mini-batch budget: with subsampling the off-policy batch has
        # exactly |D_i| samples (Section 4.3); without subsampling all follower
        # data is used, i.e. |X| times more off-policy gradient steps.
        off_budget: Dict[int, int] = {}
        for i, batch in off_policy.items():
            n_off = int(batch["obs"].shape[0])
            num_off = max(1, math.ceil(n_off / minibatch_size))
            off_budget[i] = num_off if not self.subsample_off_policy else num_minibatches

        accum: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        policy_losses = [0.0] * self.num_policies
        entropies = [0.0] * self.num_policies
        n_steps = 0

        for epoch in range(self.mini_epochs):
            schedules = [self._minibatch_groups(int(self.flats[j]["obs"].shape[0]), minibatch_size)
                         for j in range(self.num_policies)]
            off_schedules = {i: self._minibatch_groups(int(batch["obs"].shape[0]), minibatch_size)
                             for i, batch in off_policy.items()}
            steps = max([len(s) for s in schedules] +
                        [off_budget.get(i, 0) for i in off_schedules] + [num_minibatches])
            for step in range(steps):
                if self.actor_optimizer is not None:
                    self.actor_optimizer.zero_grad(set_to_none=True)
                if self.critic_optimizer is not None and self.critic_optimizer is not self.actor_optimizer:
                    self.critic_optimizer.zero_grad(set_to_none=True)

                total_loss = torch.zeros((), device=self.device)
                policy_batches = {}
                for j in range(self.num_policies):
                    if not schedules[j]:
                        continue
                    index = schedules[j][step % len(schedules[j])]
                    batch = self._select(self.flats[j], index)
                    policy_batches[j] = batch

                for j, batch in policy_batches.items():
                    if "advantages" not in batch or "value_targets" not in batch:
                        continue
                    on = self._on_policy_losses(j, batch)
                    total_loss = total_loss + on["loss"] + self.critic_coefficient * on["value_loss_scaled"]
                    policy_losses[j] += float(on["policy_loss"])
                    entropies[j] += float(on["entropy"])
                    accum["policy_loss"] = accum.get("policy_loss", 0.0) + float(on["policy_loss"])
                    accum["value_loss"] = accum.get("value_loss", 0.0) + float(on["value_loss"])
                    accum["entropy"] = accum.get("entropy", 0.0) + float(on["entropy"])
                    accum["kl"] = accum.get("kl", 0.0) + float(on["kl"])
                    accum["clip_frac"] = accum.get("clip_frac", 0.0) + float(on["clip_frac"])
                    accum["ratio_mean"] = accum.get("ratio_mean", 0.0) + float(on["ratio_mean"])
                    counts["policy"] = counts.get("policy", 0) + 1

                # --- leader / symmetric policies: off-policy update (Eq. 1) ---
                for i, batch_off in off_policy.items():
                    schedule = off_schedules.get(i) or []
                    if not schedule:
                        continue
                    off_index = schedule[step % len(schedule)]
                    batch = self._select(batch_off, off_index)
                    off = self._off_policy_losses(i, batch)
                    total_loss = total_loss + self.off_policy_weight * (
                        off["loss"] + self.critic_coefficient * off["value_loss_scaled"])
                    accum["off_policy_loss"] = accum.get("off_policy_loss", 0.0) + float(off["policy_loss"])
                    accum["value_loss_off"] = accum.get("value_loss_off", 0.0) + float(off["value_loss"])
                    accum["off_clip_frac"] = accum.get("off_clip_frac", 0.0) + float(off["clip_frac"])
                    accum["mu_mean"] = accum.get("mu_mean", 0.0) + float(off["mu_mean"])
                    accum["kl"] = accum.get("kl", 0.0) + float(off["kl"])
                    counts["off"] = counts.get("off", 0) + 1

                if total_loss.requires_grad:
                    total_loss.backward()
                    grad_norm = self._clip_and_step()
                    accum["grad_norm"] = accum.get("grad_norm", 0.0) + float(grad_norm)
                    counts["grad"] = counts.get("grad", 0) + 1
                    accum["total_loss"] = accum.get("total_loss", 0.0) + float(total_loss.detach())
                n_steps += 1

            # -- KL adaptive learning-rate schedule (standard PPO) -----------
            if counts.get("policy"):
                mean_kl = accum.get("kl", 0.0) / max(counts.get("policy", 1), 1)
                self._adapt_learning_rate(mean_kl)

        # aggregate statistics
        n_pol = max(counts.get("policy", 1), 1)
        n_off = max(counts.get("off", 1), 1)
        n_grad = max(counts.get("grad", 1), 1)
        stats.policy_loss = accum.get("policy_loss", 0.0) / n_pol
        stats.off_policy_loss = accum.get("off_policy_loss", 0.0) / n_off if counts.get("off") else 0.0
        stats.value_loss = accum.get("value_loss", 0.0) / n_pol
        stats.value_loss_off = accum.get("value_loss_off", 0.0) / n_off if counts.get("off") else 0.0
        stats.entropy = accum.get("entropy", 0.0) / n_pol
        stats.total_loss = accum.get("total_loss", 0.0) / n_grad
        stats.kl = accum.get("kl", 0.0) / max(n_pol + (counts.get("off") or 0), 1)
        stats.clip_frac = accum.get("clip_frac", 0.0) / n_pol
        stats.off_clip_frac = accum.get("off_clip_frac", 0.0) / n_off if counts.get("off") else 0.0
        stats.ratio_mean = accum.get("ratio_mean", 0.0) / n_pol
        stats.mu_mean = accum.get("mu_mean", 0.0) / n_off if counts.get("off") else 0.0
        stats.grad_norm = accum.get("grad_norm", 0.0) / n_grad
        stats.learning_rate = self.actor_lr
        stats.per_policy_policy_loss = [x / max(self.mini_epochs, 1) for x in policy_losses]
        stats.per_policy_entropy = [x / max(self.mini_epochs, 1) for x in entropies]
        stats.extra["num_steps"] = float(n_steps)
        return stats

    def _clip_and_step(self) -> torch.Tensor:
        params: List[torch.nn.Parameter] = []
        for optimizer in {id(o): o for o in (self.actor_optimizer, self.critic_optimizer)
                          if o is not None}.values():
            for group in optimizer.param_groups:
                params.extend([p for p in group["params"] if p.grad is not None])
        if not params:
            return torch.zeros(())
        norm = torch.nn.utils.clip_grad_norm_(params, self.grad_norm)
        if self.actor_optimizer is not None:
            self.actor_optimizer.step()
        if self.critic_optimizer is not None and self.critic_optimizer is not self.actor_optimizer:
            self.critic_optimizer.step()
        return norm.detach() if torch.is_tensor(norm) else torch.tensor(float(norm))

    def _adapt_learning_rate(self, mean_kl: float) -> None:
        """Standard PPO KL schedule: adjust the LR by 1.5x at the threshold."""
        if mean_kl > 2.0 * self.kl_threshold:
            self.actor_lr /= 1.5
            self.critic_lr /= 1.5
        elif mean_kl < 0.5 * self.kl_threshold:
            self.actor_lr *= 1.5
            self.critic_lr *= 1.5
        else:
            return
        for optimizer, lr in ((self.actor_optimizer, self.actor_lr),
                              (self.critic_optimizer, self.critic_lr)):
            if optimizer is None:
                continue
            for group in optimizer.param_groups:
                group["lr"] = lr

    # ------------------------------------------------------------- outer loop
    def update(self, deterministic: bool = False) -> SAPGUpdateStats:
        """One outer iteration: collect every block, then update on the sum."""
        buffers, metrics = self.collect(deterministic=deterministic)
        self._prepare_buffers(buffers)
        off_policy = self._build_off_policy_data(buffers)
        stats = self.train_epochs(buffers, off_policy)
        stats.samples = int(self.num_envs * self.horizon_length)
        self.total_samples += stats.samples
        stats.extra.update({str(k): float(v) for k, v in (metrics or {}).items()
                            if isinstance(v, (int, float))})
        self.update_index += 1
        self._log(stats)
        return stats

    def learn(self, num_iterations: Optional[int] = None,
              max_samples: Optional[int] = None,
              verbose: bool = False) -> Tuple["SAPGTrainer", List[Dict[str, float]]]:
        """Run Algorithm 1 until the sample/iteration budget is exhausted."""
        history: List[Dict[str, float]] = []
        iteration = 0
        start = time.time()
        while True:
            if num_iterations is not None and iteration >= num_iterations:
                break
            if max_samples is not None and self.total_samples >= max_samples:
                break
            if num_iterations is None and max_samples is None and iteration >= 100:
                break
            stats = self.update()
            record = stats.as_dict()
            record["iteration"] = iteration
            record["total_samples"] = float(self.total_samples)
            history.append(record)
            if verbose:
                print(f"[sapg] it={iteration} samples={self.total_samples} "
                      f"policy_loss={stats.policy_loss:.4f} off_loss={stats.off_policy_loss:.4f} "
                      f"value_loss={stats.value_loss:.4f} kl={stats.kl:.5f} "
                      f"lr={stats.learning_rate:.2e} elapsed={time.time() - start:.1f}s", flush=True)
            iteration += 1
        return self, history

    # alias
    train = learn

    def _log(self, stats: SAPGUpdateStats) -> None:
        if self.logger is None:
            return
        try:
            if hasattr(self.logger, "add_scalars"):
                self.logger.add_scalars("sapg", stats.as_dict(), self.update_index)
            elif hasattr(self.logger, "log"):
                self.logger.log(stats.as_dict(), step=self.update_index)
            elif callable(self.logger):
                self.logger(stats.as_dict(), self.update_index)
        except Exception:
            pass

    # -------------------------------------------------------------- plumbing
    def state_dict(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "policy": self.policy.state_dict() if hasattr(self.policy, "state_dict") else None,
            "actool": self.actor_optimizer.state_dict() if self.actor_optimizer is not None else None,
            "critic_opt": self.critic_optimizer.state_dict() if self.critic_optimizer is not None else None,
            "update_index": self.update_index,
            "total_samples": self.total_samples,
        }
        if self.phis and torch.is_tensor(self.phis[0]):
            state["phi"] = torch.stack([p.detach() for p in self.phis if torch.is_tensor(p)])
        return state

    def save(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            ensure_dir(directory)
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> "SAPGTrainer":
        state = torch.load(path, map_location=self.device)
        if state.get("policy") is not None and hasattr(self.policy, "load_state_dict"):
            self.policy.load_state_dict(state["policy"])
        if self.actor_optimizer is not None and state.get("actool") is not None:
            self.actor_optimizer.load_state_dict(state["actool"])
        if self.critic_optimizer is not None and state.get("critic_opt") is not None:
            self.critic_optimizer.load_state_dict(state["critic_opt"])
        self.update_index = int(state.get("update_index", 0))
        self.total_samples = int(state.get("total_samples", 0))
        return self


class _nullcontext:
    """Minimal ``contextlib.nullcontext`` replacement (keeps imports light)."""

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


# --------------------------------------------------------------------------- #
# Convenience entry point
# --------------------------------------------------------------------------- #
def train_sapg(
    config: SAPGConfig,
    env: Any = None,
    policy: Any = None,
    num_iterations: Optional[int] = None,
    max_samples: Optional[int] = None,
    verbose: bool = False,
    logger: Optional[Any] = None,
    trainer: Optional[SAPGTrainer] = None,
) -> Tuple[SAPGTrainer, List[Dict[str, float]]]:
    """Build (unless given) and run a :class:`SAPGTrainer`."""
    if trainer is None:
        trainer = SAPGTrainer(config, policy=policy, env=env, logger=logger)
    return trainer.learn(num_iterations=num_iterations, max_samples=max_samples, verbose=verbose)
