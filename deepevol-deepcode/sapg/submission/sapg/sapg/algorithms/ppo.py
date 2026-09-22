"""PPO base trainer (on-policy).

This module implements the "PPO base trainer" used by SAPG as the on-policy
optimisation primitive, and directly as the ``PPO`` baseline of the paper
(Schulman et al., 2017; §3 Eq. 2, §5.2).

Paper reference
---------------
§3, Eq. 2 (clipped surrogate objective)::

    L_on(pi_theta) = E_{pi_old}[ min( r_t(pi_theta),
                                      clip(r_t(pi_theta), 1-eps, 1+eps) ) A_t^{pi_old} ]

    r_t(pi_theta) = pi_theta(a_t | s_t) / pi_old(a_t | s_t)

§5.2: "In our setting, we just increase the data throughput for PPO by
increasing the batch size proportionately to the number of environments.
In particular, we see over two orders of magnitude increase in the number of
environments (from 128 to 24576)."

Implementation notes / resolved ambiguities (see plan "implementation_strategy")
-----------------------------------------------------------------------------
* Adam is used for all training with default betas/eps (0.9, 0.999, 1e-8).
* ``tau`` is interpreted as the GAE lambda parameter.
* The total objective is ``L_policy + lambda' * L_critic`` where ``lambda'`` is
  the critic coefficient (default 4.0); the off-policy weight ``lambda`` of
  Eq. 4 is a *different* quantity and lives in :mod:`sapg.losses.off_policy_loss`.
* ``bounds_loss_coefficient = 1e-4`` is an action-bound regularisation term on
  the pre-tanh action output (rl_games / DexPBT convention):
  ``a2 = clamp(-a+1, 0) + clamp(a+1, 0)``.
* KL-adaptive learning-rate schedule with threshold ``0.016`` and grad-norm clip
  ``1.0`` (rl_games / DexPBT conventions).
* ``minibatch_size_multiplier = 4`` means the minibatch holds
  ``num_envs * multiplier`` samples, giving ``horizon_length // multiplier``
  minibatches per epoch.
"""

from __future__ import annotations

import inspect
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..buffers.rollout_buffer import BufferSet, RolloutBuffer
from ..utils.config import SAPGConfig, TOTAL_ENVS, ensure_dir
from .rollout import (
    BlockManager,
    RolloutCollector,
    collect_data,
    collect_on_policy,
)

__all__ = [
    "PPOTrainer",
    "PPOUpdateStats",
    "compute_ppo_loss",
    "compute_value_loss",
    "compute_bounds_loss",
    "train_ppo_baseline",
]


# ---------------------------------------------------------------------------
# Loss primitives (Eq. 2 of the paper)
# ---------------------------------------------------------------------------
def compute_ppo_loss(
    ratio: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float = 0.1,
    old_logprobs: Optional[torch.Tensor] = None,
    new_logprobs: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Clipped surrogate loss of Eq. 2 (returns a *loss* to minimise).

    ``L_on = -E[min(r A, clip(r, 1-eps, 1+eps) A)]``

    The negation is because the paper defines an objective to be *maximised*
    while optimisers minimise; returning ``-L_on`` is the standard convention.
    """
    if advantages.dim() > 1:
        advantages = advantages.reshape(-1)
    ratio = ratio.reshape(-1)

    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    objective = torch.min(unclipped, clipped)

    info: Dict[str, torch.Tensor] = {}
    if old_logprobs is not None and new_logprobs is not None:
        old = old_logprobs.reshape(-1)
        new = new_logprobs.reshape(-1)
        # k1 estimator of KL(pi_old || pi_theta)
        info["kl"] = (old - new).mean()
        info["clip_frac"] = ((ratio - 1.0).abs() > clip_epsilon).float().mean()
        info["ratio_mean"] = ratio.mean()
        info["ratio_max"] = ratio.max()
    info["policy_loss"] = -objective.mean()
    return info


def compute_value_loss(
    values: torch.Tensor,
    value_targets: torch.Tensor,
    coefficient: float = 4.0,
    huber_delta: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """Critic loss: (optionally clipped) squared error scaled by ``lambda'``.

    The paper composes the total objective as ``L_policy + lambda' L_critic``
    with ``lambda' = 4.0`` (Appendix B); the coefficient is applied here.
    """
    values = values.reshape(-1)
    value_targets = value_targets.reshape(-1)
    if huber_delta is not None:
        err = torch.nn.functional.huber_loss(
            values, value_targets, reduction="none", delta=huber_delta
        )
    else:
        err = (values - value_targets) ** 2
    loss = err.mean()
    return {
        "value_loss": loss,
        "value_loss_scaled": coefficient * loss,
        "value_error_abs": (values - value_targets).abs().mean(),
    }


def compute_bounds_loss(action_mean, coefficient: float = 1e-4) -> torch.Tensor:
    """Action-bound regularisation on the pre-tanh mean (DexPBT convention)."""
    if action_mean is None or coefficient <= 0.0:
        return torch.zeros((), device=action_mean.device if action_mean is not None else "cpu")
    a2 = torch.clamp(-action_mean + 1.0, min=0.0) + torch.clamp(action_mean + 1.0, min=0.0)
    return coefficient * a2.sum(dim=-1).mean()


def compute_entropy_bonus(entropy: torch.Tensor, coefficient: float) -> torch.Tensor:
    """Entropy bonus ``-coefficient * H(pi)`` (maximising entropy)."""
    if entropy is None or coefficient == 0.0:
        return torch.zeros((), device=entropy.device if entropy is not None else "cpu")
    return -coefficient * entropy.mean()


# ---------------------------------------------------------------------------
# Statistics container
# ---------------------------------------------------------------------------
@dataclass
class PPOUpdateStats:
    """Aggregated statistics for one PPO update (one outer iteration)."""

    policy_loss: float = 0.0
    value_loss: float = 0.0
    total_loss: float = 0.0
    entropy: float = 0.0
    kl: float = 0.0
    clip_frac: float = 0.0
    grad_norm: float = 0.0
    learning_rate: float = 0.0
    extra: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, float]:
        out = {
            "policy_loss": self.policy_loss,
            "value_loss": self.value_loss,
            "total_loss": self.total_loss,
            "entropy": self.entropy,
            "kl": self.kl,
            "clip_frac": self.clip_frac,
            "grad_norm": self.grad_norm,
            "learning_rate": self.learning_rate,
        }
        out.update(self.extra)
        return out


# ---------------------------------------------------------------------------
# Flexible policy adapters
# ---------------------------------------------------------------------------
_KEY_ALIASES = {
    "logprob": "logprobs",
    "log_prob": "logprobs",
    "log_probs": "logprobs",
    "value": "values",
    "vals": "values",
    "sigma": "sigmas",
    "action": "actions",
    "mu": "action_mean",
    "mean": "action_mean",
    "pre_tanh": "action_mean_pre_tanh",
    "pre_tanh_mean": "action_mean_pre_tanh",
}


def _normalise_output(out: Any) -> Dict[str, torch.Tensor]:
    """Normalise a policy output mapping to canonical keys."""
    if out is None:
        return {}
    if isinstance(out, dict):
        res: Dict[str, torch.Tensor] = {}
        for k, v in out.items():
            res[_KEY_ALIASES.get(k, k)] = v
        return res
    if isinstance(out, (tuple, list)):
        names = ["actions", "logprobs", "values", "entropy", "action_mean"]
        res = {}
        for name, v in zip(names, out):
            res[name] = v
        return res
    return {"actions": out}


def _call_evaluate(policy: nn.Module, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Call ``policy.evaluate_actions`` supporting dict-style and kwargs-style."""
    fn = getattr(policy, "evaluate_actions", None)
    if fn is None:
        fn = getattr(policy, "evaluate", None)
    if fn is None:
        raise AttributeError(
            "Policy must implement `evaluate_actions(...)` returning "
            "{'logprobs', 'values', ('entropy', 'action_mean_pre_tanh')}."
        )

    kwargs = {
        "obs": batch.get("obs"),
        "actions": batch.get("actions"),
        "masks": batch.get("masks"),
        "phi": batch.get("phi"),
        "hidden_state": batch.get("hidden_state", batch.get("states")),
        "sigma": batch.get("sigma"),
    }
    try:
        return _normalise_output(fn(**kwargs))
    except TypeError:
        pass
    # dict-style call
    try:
        return _normalise_output(fn(batch))
    except TypeError:
        pass
    # minimal call
    return _normalise_output(fn(batch.get("obs"), batch.get("actions")))


def _filter_batch_for_policy(policy: nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Drop keyword arguments the policy does not accept."""
    fn = getattr(policy, "evaluate_actions", None) or getattr(policy, "evaluate", None)
    if fn is None:
        return batch
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return batch
    params = set(sig.parameters)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return batch
    alias_inv = {v: k for k, v in _KEY_ALIASES.items()}
    accepted = set(params)
    for key, alias in alias_inv.items():
        if key in params:
            accepted.add(alias)
    filtered = {}
    for k, v in batch.items():
        if k in accepted or (k in {"obs", "actions", "masks", "phi", "hidden_state", "sigma"}):
            filtered[k] = v
    return filtered


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class PPOTrainer:
    """On-policy PPO trainer used as SAPG's inner optimiser and as a baseline.

    The trainer is deliberately agnostic to the concrete policy class; any
    module exposing ``act(...)``/``evaluate_actions(...)`` works (see
    :mod:`sapg.models.actor`, :mod:`sapg.models.critic`).

    Parameters
    ----------
    config:
        :class:`~sapg.utils.config.SAPGConfig` holding all hyperparameters
        (Appendix B, Tables 2-4).
    policy:
        The actor-critic module (shared backbone for SAPG, one policy for PPO).
    env:
        Environment exposing ``reset()`` and ``step(actions)``; must be
        vectorised over ``config.num_envs`` parallel instances.
    optimizers:
        Optional ``(actor_opt, critic_opt)``; created automatically otherwise.
    """

    def __init__(
        self,
        config: SAPGConfig,
        policy: nn.Module,
        env: Any,
        optimizers: Optional[Tuple[torch.optim.Optimizer, torch.optim.Optimizer]] = None,
        block_manager: Optional[BlockManager] = None,
        logger: Optional[Any] = None,
        device: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.policy = policy
        self.env = env
        self.logger = logger
        self.device = torch.device(device or getattr(config, "device", "cpu"))

        self.num_envs = int(config.num_envs)
        self.horizon_length = int(config.horizon_length)
        self.mini_epochs = int(config.mini_epochs)
        self.minibatch_size_multiplier = int(getattr(config, "minibatch_size_multiplier", 4) or 4)
        self.clip_epsilon = float(config.clip_epsilon)
        self.gamma = float(config.gamma)
        self.tau = float(config.tau)  # GAE lambda
        self.critic_coefficient = float(config.critic_coefficient)
        self.bounds_loss_coefficient = float(getattr(config, "bounds_loss_coefficient", 1e-4))
        self.grad_norm = float(config.grad_norm or 0.0)
        self.kl_threshold = float(config.kl_threshold or 0.0)
        self.mean_value_loss = bool(getattr(config, "mean_value_loss", False))
        self.huber_delta = getattr(config, "huber_delta", None)
        self.entropy_coefficient = float(getattr(config, "entropy_coefficient", 0.0) or 0.0)

        self.num_minibatches = max(1, self.horizon_length // max(1, self.minibatch_size_multiplier))
        self.minibatch_size = max(1, self.num_envs * self.minibatch_size_multiplier)

        self.observed_semantics = {}
        if hasattr(self.policy, "set_sigma_semantics"):
            call = getattr(self.policy, "set_sigma_semantics")
            try:
                call()
            except Exception:  # pragma: no cover - best effort
                pass

        if optimizers is not None:
            self.actor_optimizer, self.critic_optimizer = optimizers
        else:
            self.actor_optimizer, self.critic_optimizer = self._build_optimizers(policy)

        self.obs: Optional[torch.Tensor] = None
        self.hidden_state: Any = None
        self.iteration = 0
        self.sample_count = 0
        self.collector: Optional[RolloutCollector] = None
        self._history: List[Dict[str, float]] = []
        self._last_stats: Dict[str, float] = {}

    # -- optimizers ---------------------------------------------------------
    def _build_optimizers(
        self, policy: nn.Module
    ) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        betas = tuple(self.config.adam_betas)
        eps = float(self.config.adam_eps)
        actor_lr = float(self.config.learning_rate)
        critic_lr = float(self.config.critic_learning_rate or actor_lr)

        actor_params = None
        critic_params = None
        if hasattr(policy, "actor_parameters"):
            actor_params = list(policy.actor_parameters())
        if hasattr(policy, "critic_parameters"):
            critic_params = list(policy.critic_parameters())

        if actor_params is None or critic_params is None:
            params = [p for p in policy.parameters() if p.requires_grad]
            opt = torch.optim.Adam(params, lr=actor_lr, betas=betas, eps=eps)
            return opt, opt

        actor_opt = torch.optim.Adam(actor_params, lr=actor_lr, betas=betas, eps=eps)
        critic_opt = torch.optim.Adam(critic_params, lr=critic_lr, betas=betas, eps=eps)
        return actor_opt, critic_opt

    def _lr(self) -> float:
        return float(self.actor_optimizer.param_groups[0]["lr"])

    def _set_lr(self, lr: float) -> None:
        for opt in {id(self.actor_optimizer): self.actor_optimizer, id(self.critic_optimizer): self.critic_optimizer}.values():
            for group in opt.param_groups:
                group["lr"] = lr

    def _adapt_learning_rate(self, kl: float) -> None:
        """KL-adaptive LR schedule (threshold ``kl_threshold``, factor 1.5)."""
        if not self.kl_threshold:
            return
        lr = self._lr()
        if kl > 2.0 * self.kl_threshold:
            lr = max(lr / 1.5, 1e-6)
        elif kl < 0.5 * self.kl_threshold:
            lr = min(lr * 1.5, 1e-2)
        if lr != self._lr():
            self._set_lr(lr)

    # -- data collection ----------------------------------------------------
    def _init_hidden(self, num_envs: int):
        if hasattr(self.policy, "init_hidden_state"):
            try:
                return self.policy.init_hidden_state(num_envs, device=self.device)
            except TypeError:
                return self.policy.init_hidden_state(num_envs)
        return None

    def collect(self, deterministic: bool = False):
        """Collect ``horizon_length`` steps of on-policy experience."""
        if self.obs is None:
            self.obs = self._reset_env()
        if self.hidden_state is None and getattr(self.config, "recurrent", False):
            self.hidden_state = self._init_hidden(self.num_envs)

        buffer, self.obs, self.hidden_state, metrics = collect_on_policy(
            self.policy,
            self.env,
            self.config,
            obs=self.obs,
            hidden_state=self.hidden_state,
            deterministic=deterministic,
        )
        self.buffer = buffer
        self.buffer_set = BufferSet([buffer], leader_index=1)
        self._collect_metrics = metrics
        self.sample_count += buffer.num_samples if hasattr(buffer, "num_samples") else (
            self.horizon_length * self.num_envs
        )
        return buffer

    def _reset_env(self):
        if hasattr(self.env, "reset"):
            out = self.env.reset()
        else:  # pragma: no cover - unexpected interface
            raise AttributeError("Environment must implement reset().")
        if isinstance(out, tuple):
            out = out[0]
        return out

    # -- update -------------------------------------------------------------
    def _iter_minibatches(self, buffer: RolloutBuffer):
        """Yield minibatches, supporting both flat and recurrent buffers."""
        gen = None
        for attr in ("generator", "minibatches", "iterate"):
            if hasattr(buffer, attr):
                gen = getattr(buffer, attr)
                break
        if gen is None:
            raise AttributeError("RolloutBuffer exposes no minibatch iterator.")

        attempts: List[Dict[str, Any]] = [
            dict(num_minibatches=self.num_minibatches, mini_epochs=1),
            dict(num_minibatches=self.num_minibatches),
            dict(num_batches=self.num_minibatches),
            dict(batch_size=self.minibatch_size),
            dict(),
        ]
        last_err: Optional[Exception] = None
        for kwargs in attempts:
            try:
                produced = list(gen(**kwargs))
            except TypeError as err:
                last_err = err
                continue
            if produced:
                return produced
        if last_err is not None:
            raise last_err
        return []

    def train_epoch(self, buffer: Optional[RolloutBuffer] = None) -> Dict[str, float]:
        """Run one PPO update (all mini-epochs and minibatches)."""
        buffer = buffer if buffer is not None else getattr(self, "buffer", None)
        if buffer is None:
            raise RuntimeError("No rollout buffer; call collect() first.")

        if hasattr(buffer, "compute_gae"):
            try:
                buffer.compute_gae(gamma=self.gamma, tau=self.tau)
            except TypeError:
                buffer.compute_gae(self.gamma, self.tau)

        stats = PPOUpdateStats(learning_rate=self._lr())
        n_updates = 0
        for _epoch in range(self.mini_epochs):
            for batch in self._iter_minibatches(buffer):
                batch = _filter_batch_for_policy(self.policy, batch)
                if not batch:
                    continue
                out = _call_evaluate(self.policy, batch)

                logprobs = out.get("logprobs")
                if logprobs is None:
                    raise KeyError("Policy output missing 'logprobs'.")
                values = out.get("values")
                if values is None:
                    raise KeyError("Policy output missing 'values'.")

                old_logprobs = batch.get("logprobs", batch.get("old_logprobs"))
                if old_logprobs is None:
                    raise KeyError("Batch missing old log-probabilities.")

                ratio = torch.exp(logprobs - old_logprobs).reshape(-1)
                advantages = batch.get("advantages")
                if advantages is None:
                    raise KeyError("Batch missing advantages.")
                advantages = advantages.reshape(-1)
                if getattr(self.config, "normalize_advantage", True) and advantages.numel() > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                targets = batch.get("value_targets")
                if targets is None:
                    targets = batch.get("returns")
                if targets is None:
                    raise KeyError("Batch missing value targets.")

                policy_info = compute_ppo_loss(
                    ratio,
                    advantages,
                    clip_epsilon=self.clip_epsilon,
                    old_logprobs=old_logprobs,
                    new_logprobs=logprobs,
                )
                critic_info = compute_value_loss(
                    values,
                    targets,
                    coefficient=self.critic_coefficient,
                    huber_delta=self.huber_delta,
                )

                loss = policy_info["policy_loss"] + critic_info["value_loss_scaled"]

                entropy = out.get("entropy")
                if entropy is not None and self.entropy_coefficient:
                    loss = loss + compute_entropy_bonus(entropy, self.entropy_coefficient)

                bounds = compute_bounds_loss(
                    out.get("action_mean_pre_tanh", out.get("action_mean")),
                    self.bounds_loss_coefficient,
                )
                if isinstance(bounds, torch.Tensor) and bounds.numel():
                    loss = loss + bounds

                self.actor_optimizer.zero_grad(set_to_none=True)
                if self.critic_optimizer is not self.actor_optimizer:
                    self.critic_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = 0.0
                if self.grad_norm > 0:
                    params = [p for p in self.policy.parameters() if p.requires_grad and p.grad is not None]
                    if params:
                        gn = torch.nn.utils.clip_grad_norm_(params, self.grad_norm)
                        grad_norm = float(gn)
                self.actor_optimizer.step()
                if self.critic_optimizer is not self.actor_optimizer:
                    self.critic_optimizer.step()

                stats.policy_loss += float(policy_info["policy_loss"].detach())
                stats.value_loss += float(critic_info["value_loss"].detach())
                stats.total_loss += float(loss.detach())
                stats.kl += float(policy_info.get("kl", torch.zeros(())).detach())
                stats.clip_frac += float(policy_info.get("clip_frac", torch.zeros(())).detach())
                if entropy is not None:
                    stats.entropy += float(torch.as_tensor(entropy).mean().detach())
                stats.grad_norm += grad_norm
                n_updates += 1

        n = max(1, n_updates)
        stats.policy_loss /= n
        stats.value_loss /= n
        stats.total_loss /= n
        stats.kl /= n
        stats.clip_frac /= n
        stats.entropy /= n
        stats.grad_norm /= n
        self._adapt_learning_rate(stats.kl)
        stats.learning_rate = self._lr()

        self._last_stats = stats.as_dict()
        self._history.append(dict(self._last_stats))
        return self._last_stats

    # -- outer loop ---------------------------------------------------------
    def update(self, deterministic: bool = False) -> Dict[str, float]:
        """One full PPO iteration: collect then train."""
        self.iteration += 1
        self.collect(deterministic=deterministic)
        metrics = self.train_epoch()
        merged = dict(getattr(self, "_collect_metrics", {}) or {})
        merged.update(metrics)
        merged["iteration"] = float(self.iteration)
        merged["samples"] = float(self.sample_count)
        if self.logger is not None:
            log = getattr(self.logger, "log", None) or getattr(self.logger, "add_scalars", None)
            if log is not None:
                try:
                    log(merged, step=self.sample_count)
                except TypeError:
                    log(merged)
        return merged

    def learn(
        self,
        num_iterations: Optional[int] = None,
        max_samples: Optional[int] = None,
        verbose: bool = False,
    ) -> List[Dict[str, float]]:
        """Train until ``num_iterations`` or ``max_samples`` is reached."""
        samples_per_iter = self.horizon_length * self.num_envs
        if num_iterations is None:
            if max_samples is None:
                num_iterations = 1
            else:
                num_iterations = max(1, int(math.ceil(max_samples / samples_per_iter)))
        history = self._history
        for it in range(num_iterations):
            metrics = self.update()
            if verbose and (it % max(1, num_iterations // 20) == 0):
                print(
                    f"[PPO] it={it} samples={self.sample_count} "
                    f"pi_loss={metrics['policy_loss']:.4f} vf_loss={metrics['value_loss']:.4f} "
                    f"kl={metrics['kl']:.5f} lr={metrics['learning_rate']:.2e}"
                )
            if max_samples is not None and self.sample_count >= max_samples:
                break
        return history

    # -- checkpointing ------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {
            "policy": self.policy.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "iteration": self.iteration,
            "sample_count": self.sample_count,
        }

    def save(self, path: str) -> str:
        ensure_dir(os.path.dirname(path) or ".")
        torch.save(self.state_dict(), path)
        return path

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        self.iteration = ckpt.get("iteration", 0)
        self.sample_count = ckpt.get("sample_count", 0)


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------
def train_ppo_baseline(
    config: SAPGConfig,
    policy: nn.Module,
    env: Any,
    num_iterations: Optional[int] = None,
    max_samples: Optional[int] = None,
    verbose: bool = False,
    logger: Optional[Any] = None,
) -> Tuple[PPOTrainer, List[Dict[str, float]]]:
    """Construct a :class:`PPOTrainer` and train it (§5.2 PPO baseline)."""
    trainer = PPOTrainer(config, policy, env, logger=logger)
    history = trainer.learn(
        num_iterations=num_iterations, max_samples=max_samples, verbose=verbose
    )
    return trainer, history
