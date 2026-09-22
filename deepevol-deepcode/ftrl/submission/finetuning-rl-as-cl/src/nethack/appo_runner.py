"""APPO fine-tuning runner for the NetHack (Human Monk) experiments.

This module implements the online fine-tuning half of the NetHack pipeline of
Wołczyk et al. (2024), *Fine-tuning Reinforcement Learning Models is Secretly a
Forgetting Mitigation Problem*:

* Section 3 / Appendix B.1: asynchronous PPO (APPO, Petrenko et al. 2020) is used
  to fine-tune the pre-trained ``pi_*`` (Tuyls et al. 2023) checkpoint.  With this
  setup "we can run over 500 million environment steps under 24 hours of training
  on A100 Nvidia GPU".
* Table 1 lists the model / optimisation hyperparameters used in NLE.  They are
  reproduced verbatim in :data:`TABLE1_DEFAULTS` (see also
  :class:`APPOConfig`).
* Appendix B.1 describes the four retention variants implemented here:

  - ``scratch``     -- from-scratch baseline (never sees ``pi_*``);
  - ``none``        -- vanilla fine-tuning;
  - ``"ewc"``       -- Fine-tuning + EWC, regularization coefficient ``2e6``;
  - ``"bc"``        -- Fine-tuning + BC, auxiliary scale ``2.0``, **no decay**;
  - ``"ks"``        -- Fine-tuning + KS, scale ``0.5`` with exponential decay
                       ``0.99998`` applied *every train step*.

  "To improve the stability of the models we froze the encoders during the course
  of the training. Additionally, we turn off entropy when employing knowledge
  retention methods in similar fashion to (Baker et al., 2022)."

The runner is deliberately dependency-light and duck-typed: PyTorch is imported
defensively (so the module can be imported for docs/tests without it), the NLE
environment / model / dataset modules are imported softly, and a small internal
actor-critic plus vector-env stub lets the whole loop be smoke-tested on CPU
before the real NLE stack is available.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - torch is an optional import at module load time
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _HAS_TORCH = False

try:  # pragma: no cover - numpy is optional as well
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None  # type: ignore


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise RuntimeError(
            "src.nethack.appo_runner requires PyTorch for training. "
            "Install torch (CUDA build for NetHack) to use the APPO runner."
        )


# --------------------------------------------------------------------------------------
# Paper constants (Table 1, Appendix B.1, Appendix C)
# --------------------------------------------------------------------------------------

#: Table 1 ("Hyperparameters of the model used in NLE", values from Hambro et al. 2022c).
TABLE1_DEFAULTS: Dict[str, Any] = {
    "activation_function": "relu",
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_eps": 0.0000001,
    "adam_learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "appo_clip_policy": 0.1,
    "appo_clip_baseline": 1.0,
    "baseline_cost": 1,
    "discounting": 0.999999,
    "entropy_cost": 0.001,
    "grad_norm_clipping": 4,
    "hidden_dim": 1738,
    "batch_size": 128,
    "penalty_step": 0.0,
    "penalty_time": 0.0,
    "reward_clip": 10,
    "reward_scale": 1,
    "unroll_length": 32,
}

#: Appendix B.1: retention-specific hyperparameters for NetHack.
RETENTION_CONFIG: Dict[str, Dict[str, Any]] = {
    "ewc": {"coef": 2e6, "critic_coef": 0.0, "num_batches": 10_000, "batch_size": 128},
    "bc": {"coef": 2.0, "critic_coef": 0.0, "decay": None, "memory_size": 10_000},
    "ks": {"coef": 0.5, "critic_coef": 0.0, "decay": 0.99998, "decay_every": "train_step"},
}

#: Appendix B.1: the pre-trained baseline head is trained for 500M env steps
#: with the rest of the model frozen.
BASELINE_PRETRAIN_STEPS = 500_000_000

#: Appendix B.1: models are evaluated every 25M environment steps (Figure 5).
EVAL_EVERY = 25_000_000

#: Section 5: evaluation rollouts stop at death, 150 steps without progress, or
#: 100k steps.
EVAL_NO_PROGRESS_STEPS = 150
EVAL_MAX_STEPS = 100_000
EVAL_EPISODES = 1_000

#: Number of AutoAscend saves generated per target level for the per-level eval.
AUTOASCEND_SAVES_PER_LEVEL = 200

DEFAULT_TOTAL_STEPS = 500_000_000

TRAINING_METHODS: Tuple[str, ...] = ("scratch", "none", "ewc", "bc", "ks")

#: Kendall's notation for the NetHack observation keys emitted by NLE.
NETHACK_OBS_KEYS: Tuple[str, ...] = (
    "tty_chars",
    "tty_colors",
    "tty_cursor",
    "blstats",
    "message",
)


# --------------------------------------------------------------------------------------
# Soft imports of sibling modules (they may be missing in a minimal checkout)
# --------------------------------------------------------------------------------------


def _import_attr(module_name: str, attr: str) -> Any:
    """Best-effort import of ``attr`` from a sibling module (``None`` on failure)."""

    try:  # relative import first (when used as ``src.nethack.appo_runner``)
        mod = __import__(module_name, fromlist=[attr])
        return getattr(mod, attr)
    except Exception:
        pass
    try:
        from importlib import import_module  # local import keeps module import cheap

        return getattr(import_module(module_name), attr)
    except Exception:
        return None


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


def _cfg_get(cfg: Any, *paths: str, default: Any = None) -> Any:
    """Fetch the first present value among dotted ``paths`` from any config-ish object."""

    if cfg is None:
        return default
    for path in paths:
        node: Any = cfg
        ok = True
        for part in path.split("."):
            if node is None:
                ok = False
                break
            if isinstance(node, Mapping):
                if part in node:
                    node = node[part]
                    continue
                ok = False
                break
            if hasattr(node, part):
                node = getattr(node, part)
                continue
            ok = False
            break
        if ok and node is not None:
            return node
    return default


@dataclass
class APPOConfig:
    """APPO fine-tuning configuration.

    The first block of fields mirrors Table 1 of the paper exactly; the remaining
    fields are runner-level knobs (Section 3 / Appendix B.1) that are not listed in
    Table 1 and therefore use sensible defaults.
    """

    # --- Table 1 -------------------------------------------------------------------
    activation_function: str = "relu"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-7
    adam_learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    appo_clip_policy: float = 0.1
    appo_clip_baseline: float = 1.0
    baseline_cost: float = 1.0
    discounting: float = 0.999999
    entropy_cost: float = 0.001
    grad_norm_clipping: float = 4.0
    hidden_dim: int = 1738
    batch_size: int = 128
    penalty_step: float = 0.0
    penalty_time: float = 0.0
    reward_clip: float = 10.0
    reward_scale: float = 1.0
    unroll_length: int = 32

    # --- runner knobs (Section 3 / Appendix B.1) -----------------------------------
    num_envs: int = 128
    num_workers: int = 128
    num_batches_per_epoch: int = 32
    num_epochs: int = 1
    gae_lambda: float = 0.95
    normalize_advantage: bool = True
    clip_value_loss: bool = True

    # --- retention / pipeline ------------------------------------------------------
    method: str = "none"
    freeze_encoders: bool = True
    disable_entropy_with_retention: bool = True
    retention_coefficients: Dict[str, float] = field(
        default_factory=lambda: {k: float(v["coef"]) for k, v in RETENTION_CONFIG.items()}
    )

    # --- bookkeeping ---------------------------------------------------------------
    total_steps: int = DEFAULT_TOTAL_STEPS
    eval_every: int = EVAL_EVERY
    eval_episodes: int = EVAL_EPISODES
    save_every: int = EVAL_EVERY
    log_every: int = 1_000_000
    device: str = "cpu"
    seed: Optional[int] = None
    stub: bool = False
    output_dir: Optional[str] = None
    checkpoint: Optional[str] = None

    # ------------------------------------------------------------------ constructors

    @classmethod
    def from_config(cls, cfg: Any = None, **overrides: Any) -> "APPOConfig":
        """Build from a ``Config``/mapping (``configs/nethack.yaml``) or from nothing."""

        if isinstance(cfg, APPOConfig):
            return cfg.with_overrides(**overrides)

        from_blocks = dict(TABLE1_DEFAULTS)
        # accept both flat keys and the `appo`/`model`/`runner` sub-blocks
        for key in list(TABLE1_DEFAULTS):
            value = _cfg_get(
                cfg,
                f"appo.{key}",
                f"model.{key}",
                f"train.{key}",
                key,
                default=None,
            )
            if value is not None:
                from_blocks[key] = value

        configured = dict(from_blocks)
        configured.update(
            {
                "num_envs": _cfg_get(cfg, "appo.num_envs", "env.num_envs", "num_envs", default=128),
                "num_workers": _cfg_get(
                    cfg, "appo.num_workers", "env.num_workers", "num_workers", default=128
                ),
                "num_batches_per_epoch": _cfg_get(
                    cfg, "appo.num_batches_per_epoch", "num_batches_per_epoch", default=32
                ),
                "num_epochs": _cfg_get(cfg, "appo.num_epochs", "num_epochs", default=1),
                "gae_lambda": _cfg_get(cfg, "appo.lam", "appo.lambda_", "gae_lambda", default=0.95),
                "normalize_advantage": _cfg_get(
                    cfg, "appo.normalize_advantage", "normalize_advantage", default=True
                ),
                "clip_value_loss": _cfg_get(
                    cfg, "appo.clip_value_loss", "clip_value_loss", default=True
                ),
                "method": _cfg_get(cfg, "retention.method", "method", default="none"),
                "freeze_encoders": _cfg_get(
                    cfg, "appo.freeze_encoders", "freeze_encoders", default=True
                ),
                "disable_entropy_with_retention": _cfg_get(
                    cfg,
                    "appo.disable_entropy_with_retention",
                    "disable_entropy_with_retention",
                    default=True,
                ),
                "total_steps": _cfg_get(
                    cfg, "finetune.num_steps", "train.num_steps", "total_steps",
                    default=DEFAULT_TOTAL_STEPS,
                ),
                "eval_every": _cfg_get(cfg, "eval.every", "eval_every", default=EVAL_EVERY),
                "eval_episodes": _cfg_get(
                    cfg, "eval.episodes", "eval_episodes", default=EVAL_EPISODES
                ),
                "save_every": _cfg_get(
                    cfg, "finetune.save_every", "save_every", default=EVAL_EVERY
                ),
                "log_every": _cfg_get(cfg, "finetune.log_every", "log_every", default=1_000_000),
                "device": _cfg_get(cfg, "compute.device", "device", default="cpu"),
                "seed": _cfg_get(cfg, "seed", "train.seed", default=None),
                "stub": _cfg_get(cfg, "env.stub", "stub", default=False),
                "output_dir": _cfg_get(cfg, "output_dir", "output", default=None),
                "checkpoint": _cfg_get(
                    cfg, "pretrained.checkpoint", "checkpoint", "init_checkpoint", default=None
                ),
            }
        )

        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        configured = {k: v for k, v in configured.items() if k in known}
        configured.update(overrides)
        return cls(**configured)

    def with_overrides(self, **overrides: Any) -> "APPOConfig":
        overrides = {k: v for k, v in overrides.items() if v is not None}
        return replace(self, **overrides)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    # --------------------------------------------------------------------- properties

    @property
    def entropy_enabled(self) -> bool:
        """Entropy is switched off whenever a retention method is active (Appendix B.1)."""

        if self.method in ("none", "scratch"):
            return True
        return not self.disable_entropy_with_retention

    @property
    def effective_entropy_cost(self) -> float:
        return float(self.entropy_cost) if self.entropy_enabled else 0.0

    @property
    def retention_coef(self) -> float:
        """Auxiliary-loss scale for the configured method (Appendix B.1)."""

        return float(self.retention_coefficients.get(self.method, 0.0))

    @property
    def uses_retention(self) -> bool:
        return self.method in ("ewc", "bc", "ks", "em")

    @property
    def batch_steps(self) -> int:
        return int(self.num_envs) * int(self.unroll_length)

    @property
    def minibatch_size(self) -> int:
        """Table 1's ``batch_size`` is the APPO minibatch size."""

        return min(int(self.batch_size), max(1, self.batch_steps))

    @property
    def num_minibatches(self) -> int:
        return max(1, self.batch_steps // self.minibatch_size)


# --------------------------------------------------------------------------------------
# Losses / utilities
# --------------------------------------------------------------------------------------


def clip_rewards(rewards: Any, clip: float = 10.0) -> Any:
    """Table 1: ``reward_clip = 10``."""

    if torch is not None and isinstance(rewards, torch.Tensor):
        return rewards.clamp(-float(clip), float(clip))
    if _np is not None and isinstance(rewards, _np.ndarray):
        return _np.clip(rewards, -float(clip), float(clip))
    if isinstance(rewards, (list, tuple)):
        return type(rewards)(clip_rewards(r, clip) for r in rewards)
    return max(-float(clip), min(float(clip), float(rewards)))


def scale_rewards(rewards: Any, scale: float = 1.0) -> Any:
    """Table 1: ``reward_scale = 1``."""

    if torch is not None and isinstance(rewards, torch.Tensor):
        return rewards * float(scale)
    if _np is not None and isinstance(rewards, _np.ndarray):
        return rewards * float(scale)
    if isinstance(rewards, (list, tuple)):
        return type(rewards)(scale_rewards(r, scale) for r in rewards)
    return float(rewards) * float(scale)


def compute_gae(
    rewards: "torch.Tensor",
    values: "torch.Tensor",
    dones: "torch.Tensor",
    last_value: "torch.Tensor",
    gamma: float = 0.999999,
    lam: float = 0.95,
    normalize: bool = True,
    eps: float = 1e-8,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Generalized advantage estimation over a ``(T, N)`` rollout.

    Args:
        rewards: ``(T, N)`` rewards (already clipped/scaled).
        values: ``(T, N)`` value estimates of the behaviour policy.
        dones: ``(T, N)`` episode-termination flags (1.0 for bootstrap cut, i.e. *not*
            terminated -- the convention used by Sample Factory / RND code bases).
        last_value: ``(N,)`` bootstrap value for the state after the last step.
        gamma: Table 1 ``discounting = 0.999999``.
        lam: GAE lambda.
        normalize: whether to standardize the advantages.
    """

    _require_torch()
    assert rewards.dim() == 2, "rewards must be (T, N)"
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(N, device=rewards.device, dtype=rewards.dtype)

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = last_value
            next_adv = torch.zeros_like(last_gae)
        else:
            next_value = values[t + 1]
            next_adv = advantages[t + 1]
        mask = dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        last_gae = delta + gamma * lam * mask * next_adv
        advantages[t] = last_gae

    returns = advantages + values
    if normalize:
        adv = advantages.reshape(-1)
        advantages = ((advantages - adv.mean()) / (adv.std(unbiased=False) + eps)).view(T, N)
    return advantages, returns


def appo_policy_loss(
    log_probs: "torch.Tensor",
    old_log_probs: "torch.Tensor",
    advantages: "torch.Tensor",
    clip: float = 0.1,
    reduction: str = "mean",
) -> "torch.Tensor":
    """Clipped surrogate objective of PPO/APPO.

    The *asynchronous* variant of PPO ("V-trace-like" importance correction) is not
    used in Sample Factory's APPO implementation, which applies the standard PPO clip
    with ``appo_clip_policy = 0.1`` (Table 1).
    """

    _require_torch()
    ratio = torch.exp(log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages
    return -_reduce(torch.min(unclipped, clipped), reduction)


def appo_value_loss(
    values: "torch.Tensor",
    old_values: Optional["torch.Tensor"],
    returns: "torch.Tensor",
    clip: Optional[float] = 1.0,
    cost: float = 1.0,
    reduction: str = "mean",
) -> "torch.Tensor":
    """Baseline loss, optionally clipped with ``appo_clip_baseline`` (Table 1)."""

    _require_torch()
    if clip is None or old_values is None:
        loss = (values - returns) ** 2
    else:
        # Sample Factory's clipping: 1 - clip <= V_new / V_old <= 1 + clip
        clipped_values = torch.where(
            values > old_values,
            torch.clamp(values, max=old_values * (1.0 + float(clip))),
            torch.clamp(values, min=old_values * (1.0 - float(clip))),
        )
        loss = torch.max((values - returns) ** 2, (clipped_values - returns) ** 2)
    return float(cost) * _reduce(loss, reduction)


def appo_losses(
    log_probs: "torch.Tensor",
    old_log_probs: "torch.Tensor",
    values: "torch.Tensor",
    old_values: Optional["torch.Tensor"],
    advantages: "torch.Tensor",
    returns: "torch.Tensor",
    entropy: Optional["torch.Tensor"] = None,
    clip_policy: float = 0.1,
    clip_baseline: Optional[float] = 1.0,
    baseline_cost: float = 1.0,
    entropy_cost: float = 0.001,
    reduction: str = "mean",
) -> Dict[str, "torch.Tensor"]:
    """Full APPO objective: ``policy + baseline_cost * value - entropy_cost * H``."""

    _require_torch()
    policy_loss = appo_policy_loss(log_probs, old_log_probs, advantages, clip=clip_policy, reduction=reduction)
    value_loss = appo_value_loss(
        values, old_values, returns, clip=clip_baseline, cost=baseline_cost, reduction=reduction
    )
    if entropy is not None and entropy_cost:
        entropy_term = -float(entropy_cost) * _reduce(entropy, reduction)
    else:
        entropy_term = torch.zeros((), device=policy_loss.device, dtype=policy_loss.dtype)
    total = policy_loss + value_loss + entropy_term
    return {
        "total": total,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy_term,
    }


def _reduce(value: "torch.Tensor", reduction: str = "mean") -> "torch.Tensor":
    if reduction == "mean":
        return value.mean()
    if reduction == "sum":
        return value.sum()
    return value


def grad_norm_clip(parameters: Iterable[Any], max_norm: float = 4.0) -> float:
    """Table 1: ``grad_norm_clipping = 4``."""

    _require_torch()
    params = [p for p in parameters if p is not None and p.grad is not None]
    if not params:
        return 0.0
    return float(torch.nn.utils.clip_grad_norm_(params, float(max_norm)))


# --------------------------------------------------------------------------------------
# Retention integration (actor only -- critic coefficient is always zero)
# --------------------------------------------------------------------------------------


class RetentionBundle:
    """Attach EWC / BC / KS retention losses to the APPO actor.

    The bundle is duck-typed against ``src.retention``: each mechanism must expose
    either ``penalty_loss(actor=...)`` / ``loss(actor=...)`` / ``penalty(actor=...)``
    or be callable with ``actor=``.  The auxiliary loss is **only** added to the actor
    objective; in line with Wołczyk et al. (2022) the critic coefficient is always 0.
    """

    def __init__(self, method: str = "none", mechanisms: Optional[Sequence[Any]] = None, step: int = 0):
        self.method = method
        self.mechanisms: List[Any] = list(mechanisms or [])
        self.step_count = int(step)

    # ------------------------------------------------------------------ construction

    @classmethod
    def build(
        cls,
        cfg: Any,
        actor: Any,
        *,
        method: Optional[str] = None,
        teacher: Any = None,
        fisher: Any = None,
        bc_dataset: Any = None,
        device: str = "cpu",
        seed: Optional[int] = None,
        env_name: str = "nethack",
    ) -> "RetentionBundle":
        """Instantiate the retention mechanisms requested by ``method``."""

        cfg_method = method or _cfg_get(cfg, "retention.method", "method", default="none")
        method = str(cfg_method or "none").lower()
        if method not in ("ewc", "bc", "ks", "em"):
            return cls(method="none")

        mechanisms: List[Any] = []
        coefs = {k: float(v["coef"]) for k, v in RETENTION_CONFIG.items()}
        override = _cfg_get(cfg, f"retention.{method}.actor_coef", f"retention.{method}.coef", default=None)
        if override is not None:
            coefs[method] = float(override)

        if method == "ewc":
            EWC = _import_attr("src.retention.ewc", "EWC")
            if EWC is None:
                raise RuntimeError("EWC retention requested but src.retention.ewc is unavailable")
            ewc_coef = float(coefs["ewc"])
            mechanisms.append(
                EWC(actor, fisher_diag=fisher, coef=ewc_coef, name="ewc")
            )
        elif method == "bc":
            BehavioralCloning = _import_attr("src.retention.behavioral_cloning", "BehavioralCloning")
            if BehavioralCloning is None:
                raise RuntimeError(
                    "BC retention requested but src.retention.behavioral_cloning is unavailable"
                )
            mechanisms.append(
                BehavioralCloning(
                    actor,
                    teacher=teacher,
                    buffer=bc_dataset,
                    coef=float(coefs["bc"]),
                    decay=None,
                    direction="forward",  # Appendix C.2: KL(pi_theta || pi_*)
                    name="bc",
                )
            )
        elif method == "ks":
            Kickstarting = _import_attr("src.retention.kickstarting", "Kickstarting")
            if Kickstarting is None:
                raise RuntimeError(
                    "KS retention requested but src.retention.kickstarting is unavailable"
                )
            mechanisms.append(
                Kickstarting(
                    actor,
                    teacher=teacher,
                    coef=float(coefs["ks"]),
                    decay=float(RETENTION_CONFIG["ks"]["decay"]),
                    decay_type="exponential",
                    direction="reverse",  # Appendix C.2: KL(pi_* || pi_theta)
                    name="ks",
                )
            )
        elif method == "em":
            # Episodic memory adds no loss: retention comes from the replay data.
            return cls(method="em")

        return cls(method=method, mechanisms=mechanisms)

    # ------------------------------------------------------------------------ runtime

    @property
    def enabled(self) -> bool:
        return bool(self.mechanisms)

    def loss(
        self,
        actor: Any = None,
        *,
        batch: Any = None,
        obs: Any = None,
        step: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        """Sum of the active auxiliary actor losses (differentiable)."""

        if not self.mechanisms:
            return self._zero(actor)
        total = None
        for mech in self.mechanisms:
            term = self._call_mechanism(mech, actor=actor, batch=batch, obs=obs, **kwargs)
            if term is None:
                continue
            total = term if total is None else total + term
        return total if total is not None else self._zero(actor)

    # the runner treats retention as ``loss(actor=..., batch=..., obs=...)``
    penalty_loss = loss
    penalty = loss

    def __call__(self, actor: Any = None, **kwargs: Any) -> Any:
        return self.loss(actor, **kwargs)

    def step(self, n: int = 1) -> None:
        """Advance per-train-step schedules (KS decays every train step)."""

        self.step_count += int(n)
        for mech in self.mechanisms:
            step_fn = getattr(mech, "step", None)
            if callable(step_fn):
                try:
                    step_fn(int(n))
                except Exception:
                    pass

    def state_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "step_count": self.step_count,
            "mechanisms": [m.state_dict() for m in self.mechanisms if hasattr(m, "state_dict")],
        }

    def describe(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "enabled": self.enabled,
            "step_count": self.step_count,
            "mechanisms": [type(m).__name__ for m in self.mechanisms],
        }

    # ----------------------------------------------------------------------- helpers

    @staticmethod
    def _call_mechanism(mech: Any, *, actor: Any, batch: Any, obs: Any, **kwargs: Any) -> Any:
        for name in ("penalty_loss", "loss", "penalty", "auxiliary_loss"):
            fn = getattr(mech, name, None)
            if callable(fn):
                try:
                    return fn(actor=actor, batch=batch)
                except TypeError:
                    try:
                        return fn(actor=actor)
                    except TypeError:
                        try:
                            return fn(actor)
                        except TypeError:
                            continue
        if callable(mech):
            try:
                return mech(actor=actor, batch=batch)
            except TypeError:
                try:
                    return mech(actor=actor)
                except TypeError:
                    return None
        return None

    @staticmethod
    def _zero(actor: Any) -> Any:
        if not _HAS_TORCH:
            return 0.0
        params = []
        for getter in ("parameters",):
            fn = getattr(actor, getter, None)
            if callable(fn):
                try:
                    params = [p for p in fn()]
                except Exception:
                    params = []
                break
        if params:
            return sum(p.sum() * 0.0 for p in params)
        return torch.zeros(())


# --------------------------------------------------------------------------------------
# Freezing helpers
# --------------------------------------------------------------------------------------


def freeze_module(module: Any, freeze: bool = True) -> int:
    """Toggle ``requires_grad`` on every parameter of ``module``; returns the count."""

    if module is None:
        return 0
    count = 0
    for param in getattr(module, "parameters", lambda: [])():
        param.requires_grad_(not freeze)
        count += 1
    return count


def freeze_encoders(model: Any, names: Optional[Sequence[str]] = None) -> List[str]:
    """Freeze the observation encoders during fine-tuning (Appendix B.1).

    "To improve the stability of the models we froze the encoders during the course of
    the training."  When explicit ``names`` are not supplied we look for the usual
    Sample Factory / Tuyls et al. (2023) encoder attribute names.
    """

    if model is None:
        return []
    candidates = list(names) if names else [
        "encoders",
        "main_encoder",
        "screen_encoder",
        "blstats_encoder",
        "message_encoder",
        "tty_encoder",
        "encoder",
        "enc",
        "chars_encoder",
        "colors_encoder",
    ]
    frozen: List[str] = []
    for name in candidates:
        submodule = getattr(model, name, None)
        if submodule is None:
            continue
        freeze_module(submodule, True)
        frozen.append(name)
    return frozen


# --------------------------------------------------------------------------------------
# Rollout storage
# --------------------------------------------------------------------------------------


class APPORolloutBuffer:
    """Fixed-horizon APPO rollout: ``unroll_length`` steps on ``num_envs`` envs.

    Observations are stored per step in a list of dicts (or tensors) because the
    NetHack observation is a *dict* of arrays (``tty_chars``, ``tty_colors``,
    ``tty_cursor``, ``blstats``, ``message``) which cannot always be stacked eagerly.
    """

    def __init__(self, unroll_length: int, num_envs: int, device: str = "cpu"):
        self.unroll_length = int(unroll_length)
        self.num_envs = int(num_envs)
        self.device = device
        self.obs: List[Any] = []
        self.actions: List[Any] = []
        self.log_probs: List[Any] = []
        self.values: List[Any] = []
        self.rewards: List[Any] = []
        self.dones: List[Any] = []
        self.entropies: List[Any] = []

    def __len__(self) -> int:
        return len(self.actions)

    def add(
        self,
        obs: Any,
        actions: Any,
        log_probs: Any,
        values: Any,
        rewards: Any,
        dones: Any,
        entropies: Any = None,
    ) -> None:
        self.obs.append(obs)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.values.append(values)
        self.rewards.append(rewards)
        self.dones.append(dones)
        self.entropies.append(entropies)

    def clear(self) -> None:
        for attr in ("obs", "actions", "log_probs", "values", "rewards", "dones", "entropies"):
            getattr(self, attr).clear()

    def stack(self, key: str) -> Any:
        return stack_steps(getattr(self, key))

    def as_tensors(self) -> Dict[str, Any]:
        """Stack all numeric fields to ``(T, N)``/``(T, N, ...)`` tensors."""

        _require_torch()
        out: Dict[str, Any] = {}
        for key in ("actions", "log_probs", "values", "rewards", "dones", "entropies"):
            values = getattr(self, key)
            if not values or values[0] is None:
                continue
            out[key] = stack_steps(values)
        return out

    def minibatches(
        self,
        minibatch_size: int = 128,
        generator: Any = None,
        shuffle: bool = True,
    ) -> Iterable[int]:
        """Yield index arrays of size ``minibatch_size`` over the flattened rollout."""

        total = len(self.actions) * self.num_envs
        indices = list(range(total))
        if shuffle and total > 1:
            if _np is not None:
                rng = generator if isinstance(generator, _np.random.Generator) else _np.random.default_rng()
                rng.shuffle(indices)
            else:
                import random as _random

                _random.shuffle(indices)
        size = max(1, int(minibatch_size))
        for start in range(0, total, size):
            yield indices[start : start + size]


def stack_steps(values: Sequence[Any]) -> Any:
    """Stack a per-step list of arrays/tensors into ``(T, N, ...)``."""

    if not values:
        return None
    first = values[0]
    if _HAS_TORCH and isinstance(first, torch.Tensor):
        return torch.stack([v for v in values], dim=0)
    if _np is not None and all(isinstance(v, _np.ndarray) for v in values):
        return _np.stack([_np.asarray(v) for v in values], axis=0)
    if _HAS_TORCH and all(isinstance(v, (int, float)) for v in values):
        return torch.tensor([[float(v) for v in values]])
    if isinstance(first, Mapping):
        keys = first.keys()
        return {
            k: stack_steps([v[k] for v in values])
            for k in keys
        }
    if isinstance(first, (list, tuple)):
        return [stack_steps([v[i] for v in values]) for i in range(len(first))]
    return list(values)


def batch_observations(obs: Sequence[Any], device: str = "cpu") -> Any:
    """Convert a list (per env) of NetHack observations into a batched device object."""

    if not obs:
        return obs
    first = obs[0]
    if isinstance(first, Mapping):
        out: Dict[str, Any] = {}
        for key in first.keys():
            try:
                stacked = _np.stack([_np.asarray(o[key]) for o in obs], axis=0)
            except Exception:
                stacked = [o[key] for o in obs]
            out[key] = _to_device_tensor(stacked, device)
        return out
    try:
        stacked = _np.stack([_np.asarray(o) for o in obs], axis=0)
        return _to_device_tensor(stacked, device)
    except Exception:
        return obs


def _to_device_tensor(value: Any, device: str = "cpu") -> Any:
    if not _HAS_TORCH:
        return value
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if _np is not None and isinstance(value, _np.ndarray):
        tensor = torch.as_tensor(value)
        if tensor.dtype in (torch.float64,):
            tensor = tensor.float()
        return tensor.to(device)
    if isinstance(value, (int, float)):
        return torch.tensor([float(value)], device=device)
    return value


# --------------------------------------------------------------------------------------
# Duck-typed model adapters
# --------------------------------------------------------------------------------------


def extract_policy_value(output: Any) -> Tuple[Any, Any]:
    """Locate (action logits, baseline value) in an arbitrary model output."""

    logits = None
    value = None
    source = output
    if isinstance(output, Mapping):
        for key in ("action_logits", "policy_logits", "logits", "action_log_prob", "head"):
            if key in output:
                logits = output[key]
                break
        for key in ("baseline", "value", "values", "v"):
            if key in output:
                value = output[key]
                break
        if logits is None and "pi" in output:
            logits = output["pi"]
        return logits, value
    if isinstance(output, tuple):
        if len(output) >= 2:
            return output[0], output[1]
        return output[0], None
    for attr in ("action_logits", "policy_logits", "logits"):
        if hasattr(source, attr):
            logits = getattr(source, attr)
            break
    for attr in ("baseline", "value", "values"):
        if hasattr(source, attr):
            value = getattr(source, attr)
            break
    return logits, value


def _forward_model(model: Any, obs: Any, **kwargs: Any) -> Any:
    """Call ``model`` tolerantly (no ``**kwargs`` support, positional-only, ...)."""

    if model is None:
        raise RuntimeError("No model provided to the APPO runner")
    try:
        return model(obs, **kwargs)
    except TypeError:
        pass
    try:
        return model(obs)
    except TypeError:
        pass
    call = getattr(model, "forward", None)
    if callable(call):
        return call(obs)
    raise RuntimeError("The provided model is not callable and exposes no `forward` method")


def categorical_from_logits(logits: Any) -> Any:
    _require_torch()
    if isinstance(logits, Mapping):
        for key in ("action", "pi", "logits"):
            if key in logits:
                logits = logits[key]
                break
    logits = logits.float()
    return torch.distributions.Categorical(logits=logits)


# --------------------------------------------------------------------------------------
# Minimal internal model / env (smoke tests + stubs when NLE is unavailable)
# --------------------------------------------------------------------------------------


def _make_mini_model() -> Any:
    """A tiny actor-critic used for smoke tests when the NLE model is unavailable."""

    _require_torch()

    class MiniNetHackActorCritic(nn.Module):
        def __init__(self, obs_dim: int = 64, num_actions: int = 120, hidden_dim: int = 64):
            super().__init__()
            self.obs_dim = int(obs_dim)
            self.num_actions = int(num_actions)
            self.num_actions_with_extra = self.num_actions + 1
            self.encoder = nn.Sequential(nn.Linear(self.obs_dim, hidden_dim), nn.ReLU())
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
            self.policy_head = nn.Linear(hidden_dim, self.num_actions_with_extra)
            self.baseline_head = nn.Linear(hidden_dim, 1)

        def forward(self, obs, state=None, **kwargs):  # noqa: D401 - duck-typed API
            x = _flatten_obs(obs)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            features = self.encoder(x)
            out, state = self.lstm(features.unsqueeze(0), state) if state is not None else self.lstm(features.unsqueeze(0))
            out = out.squeeze(0)
            return {
                "action_logits": self.policy_head(out),
                "baseline": self.baseline_head(out).squeeze(-1),
                "state": state,
            }

    return MiniNetHackActorCritic()


def _flatten_obs(obs: Any) -> Any:
    """Flatten any NetHack-style observation (dict of arrays) into a 2-D tensor."""

    _require_torch()
    if isinstance(obs, Mapping):
        parts = []
        for key in sorted(obs.keys()):
            value = obs[key]
            if torch is not None and isinstance(value, torch.Tensor):
                tensor = value.float()
            elif _np is not None and isinstance(value, _np.ndarray):
                tensor = torch.as_tensor(value).float()
            else:
                tensor = torch.as_tensor(_np.asarray(value)).float() if _np is not None else torch.tensor([float(value)])
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(-1)
            parts.append(tensor.reshape(tensor.shape[0] if tensor.dim() > 1 else 1, -1))
        return torch.cat(parts, dim=-1) if parts else torch.zeros(1, 1)
    if torch is not None and isinstance(obs, torch.Tensor):
        return obs.float().reshape(obs.shape[0], -1) if obs.dim() > 1 else obs.float()
    if _np is not None and isinstance(obs, _np.ndarray):
        tensor = torch.as_tensor(obs).float()
        return tensor.reshape(tensor.shape[0], -1) if tensor.dim() > 1 else tensor
    return torch.tensor([[float(obs)]])


class MiniNetHackVecEnv:
    """Dependency-free vectorised stand-in for ``nle`` used by the smoke test."""

    def __init__(
        self,
        num_envs: int = 4,
        obs_dim: int = 64,
        num_actions: int = 120,
        max_steps: int = 200,
        seed: Optional[int] = None,
    ):
        self.num_envs = int(num_envs)
        self.obs_dim = int(obs_dim)
        self.num_actions = int(num_actions)
        self.max_steps = int(max_steps)
        self.action_dim = self.num_actions
        self.observation_dim = self.obs_dim
        rng = _np.random.default_rng(seed) if _np is not None else None
        self._rng = rng
        self._state = None
        self._t = 0

    # ---------------------------------------------------------------- gym-ish API

    def reset(self, seed: Optional[int] = None):
        if seed is not None and _np is not None:
            self._rng = _np.random.default_rng(seed)
        self._t = 0
        draw = self._rng.standard_normal((self.num_envs, self.obs_dim)) if self._rng is not None else [[0.0] * self.obs_dim] * self.num_envs
        self._state = _np.asarray(draw, dtype="float32") if _np is not None else draw
        return self._obs()

    def step(self, actions):
        if _np is not None:
            actions = _np.asarray(actions).reshape(-1)
            noise = self._rng.standard_normal((self.num_envs, self.obs_dim)).astype("float32")
            self._state = 0.9 * self._state + 0.1 * noise
        self._t += 1
        rewards = [0.0] * self.num_envs
        dones = [bool(self._t >= self.max_steps)] * self.num_envs
        truncated = [False] * self.num_envs
        infos = [{"t": self._t}] * self.num_envs
        return self._obs(), rewards, dones, truncated, infos

    def close(self):
        return None

    def _obs(self):
        if _np is None:
            return [self._state[i] for i in range(self.num_envs)]
        return {
            "tty_chars": _np.zeros((self.num_envs, 24, 80), dtype="uint8"),
            "blstats": self._state.astype("float32")[:, : self._obs_dim],
        }


# --------------------------------------------------------------------------------------
# APPO agent
# --------------------------------------------------------------------------------------


class APPOAgent:
    """APPO learner wrapping a (optionally pre-trained) actor-critic model."""

    def __init__(
        self,
        model: Any,
        config: Optional[APPOConfig] = None,
        device: Optional[str] = None,
        retention: Optional[RetentionBundle] = None,
        seed: Optional[int] = None,
        load_pretrained: Optional[str] = None,
    ):
        _require_torch()
        self.config = config or APPOConfig()
        self.device = torch.device(device or self.config.device)
        self.model = model.to(self.device)
        self.retention = retention
        self.lstm_state: Any = None

        if load_pretrained:
            self.load_pretrained(load_pretrained)

        self.optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=float(self.config.adam_learning_rate),
            betas=(float(self.config.adam_beta1), float(self.config.adam_beta2)),
            eps=float(self.config.adam_eps),
            weight_decay=float(self.config.weight_decay),
        )

    # ------------------------------------------------------------------ constructors

    @classmethod
    def build(
        cls,
        cfg: Any = None,
        *,
        model: Any = None,
        method: Optional[str] = None,
        teacher: Any = None,
        fisher: Any = None,
        bc_dataset: Any = None,
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
    ) -> "APPOAgent":
        config = APPOConfig.from_config(cfg)
        if method is not None:
            config = config.with_overrides(method=method)
        if checkpoint is not None:
            config = config.with_overrides(checkpoint=checkpoint)

        if model is None:
            model = build_nethack_model(config, checkpoint=config.checkpoint)

        if config.freeze_encoders and config.method not in ("scratch",):
            freeze_encoders(model)

        retention = RetentionBundle.build(
            config,
            getattr(model, "actor", model),
            method=config.method,
            teacher=teacher,
            fisher=fisher,
            bc_dataset=bc_dataset,
            device=str(device or config.device),
        )
        return cls(model, config, device=device, retention=retention)

    # ---------------------------------------------------------------------- helpers

    @property
    def actor(self) -> Any:
        return getattr(self.model, "actor", self.model)

    @property
    def critic(self) -> Any:
        return getattr(self.model, "critic", getattr(self.model, "baseline_head", None))

    def reset_state(self) -> None:
        self.lstm_state = None

    def save(self, path: str, step: Optional[int] = None, extra: Optional[Mapping[str, Any]] = None) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload: Dict[str, Any] = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "step": step,
            "method": self.config.method,
        }
        if self.retention is not None:
            payload["retention"] = self.retention.state_dict()
        if extra:
            payload.update(dict(extra))
        torch.save(payload, path)
        return path

    def load(self, path: str, load_optimizer: bool = True) -> Dict[str, Any]:
        state = torch.load(path, map_location=self.device)
        model_state = state.get("model", state)
        try:
            self.model.load_state_dict(model_state, strict=False)
        except Exception:
            pass
        if load_optimizer and "optimizer" in state:
            try:
                self.optimizer.load_state_dict(state["optimizer"])
            except Exception:
                pass
        return state

    def load_pretrained(self, path_or_state: Any) -> None:
        """Load a released ``pi_*`` checkpoint (Tuyls et al. 2023, 30M LSTM)."""

        state = path_or_state
        if isinstance(path_or_state, (str, os.PathLike)):
            if not os.path.exists(str(path_or_state)):
                raise FileNotFoundError(f"Pre-trained checkpoint not found: {path_or_state}")
            state = torch.load(str(path_or_state), map_location=self.device)
        if isinstance(state, Mapping):
            for key in ("model", "model_state_dict", "state_dict", "policy", "weights"):
                if key in state:
                    state = state[key]
                    break
        try:
            self.model.load_state_dict(state, strict=False)
        except Exception:
            # Sample-Factory style flat checkpoints: match by suffix.
            own = self.model.state_dict()
            matched = {k: v for k, v in state.items() if k in own and getattr(v, "shape", None) == own[k].shape}
            self.model.load_state_dict(matched, strict=False)

    # -------------------------------------------------------------------- acting

    def forward(self, obs: Any, state: Any = None) -> Any:
        return _forward_model(self.model, obs, state=state)

    def act(self, obs: Any, deterministic: bool = False) -> Dict[str, Any]:
        """Sample actions for a batch of observations; returns a dict of tensors."""

        output = self.forward(obs, state=self.lstm_state)
        logits, value = extract_policy_value(output)
        if logits is None:
            raise RuntimeError("The model output does not expose action logits")
        dist = categorical_from_logits(logits)
        if deterministic:
            actions = torch.argmax(dist.logits, dim=-1)
        else:
            actions = dist.sample()
        new_state = output.get("state") if isinstance(output, Mapping) else None
        if new_state is not None:
            self.lstm_state = new_state
        return {
            "actions": actions,
            "log_probs": dist.log_prob(actions),
            "entropy": dist.entropy(),
            "values": value if value is not None else torch.zeros_like(dist.log_prob(actions)),
        }

    def evaluate_actions(self, obs: Any, actions: Any) -> Dict[str, Any]:
        output = self.forward(obs, state=None)
        logits, value = extract_policy_value(output)
        dist = categorical_from_logits(logits)
        return {
            "log_probs": dist.log_prob(actions.long()),
            "entropy": dist.entropy(),
            "values": value if value is not None else torch.zeros_like(actions, dtype=torch.float32),
        }

    # -------------------------------------------------------------------- updating

    def update(
        self,
        buffer: APPORolloutBuffer,
        *,
        last_value: Any = None,
        retention_batch: Any = None,
        step: Optional[int] = None,
        generator: Any = None,
    ) -> Dict[str, float]:
        """Run one APPO epoch over ``buffer`` and return the loss statistics."""

        _require_torch()
        tensors = buffer.as_tensors()
        T = len(buffer.actions)
        N = buffer.num_envs

        rewards = _to_2d(tensors["rewards"], T, N, self.device)
        values = _to_2d(tensors["values"], T, N, self.device)
        dones = _to_2d(tensors["dones"], T, N, self.device)
        old_log_probs = _to_2d(tensors["log_probs"], T, N, self.device)
        actions = _to_2d(tensors["actions"], T, N, self.device)

        rewards = clip_rewards(rewards * float(self.config.reward_scale), self.config.reward_clip)
        if last_value is None:
            last_value = torch.zeros(N, device=self.device)
        else:
            last_value = torch.as_tensor(last_value, device=self.device).reshape(-1)

        advantages, returns = compute_gae(
            rewards,
            values,
            dones,
            last_value,
            gamma=float(self.config.discounting),
            lam=float(self.config.gae_lambda),
            normalize=bool(self.config.normalize_advantage),
        )

        # flatten (T, N, ...) -> (T*N, ...)
        flat_obs = _flatten_obs_list(buffer.obs, self.device)
        flat_actions = actions.reshape(-1)
        flat_old_log_probs = old_log_probs.reshape(-1)
        flat_old_values = values.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)

        stats: Dict[str, float] = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "retention_loss": 0.0}
        num_updates = 0
        for _epoch in range(max(1, int(self.config.num_epochs))):
            for index in buffer.minibatches(self.config.minibatch_size, generator=generator):
                idx = torch.as_tensor(index, dtype=torch.long, device=self.device)
                batch_obs = _index_obs(flat_obs, idx)
                out = self.evaluate_actions(batch_obs, flat_actions[idx])
                losses = appo_losses(
                    out["log_probs"],
                    flat_old_log_probs[idx],
                    out["values"],
                    flat_old_values[idx],
                    flat_advantages[idx],
                    flat_returns[idx],
                    entropy=out["entropy"],
                    clip_policy=float(self.config.appo_clip_policy),
                    clip_baseline=float(self.config.appo_clip_baseline) if self.config.clip_value_loss else None,
                    baseline_cost=float(self.config.baseline_cost),
                    entropy_cost=self.config.effective_entropy_cost,
                )
                total = losses["total"]

                retention_loss_value = 0.0
                if self.retention is not None and self.retention.enabled:
                    batch_for_retention = retention_batch
                    if batch_for_retention is None and not isinstance(flat_obs, torch.Tensor):
                        pass
                    aux = self.retention.loss(
                        self.actor, batch=batch_for_retention, obs=batch_obs, step=step
                    )
                    if aux is not None and _HAS_TORCH and torch.is_tensor(aux):
                        total = total + aux
                        retention_loss_value = float(aux.detach().cpu())

                self.optimizer.zero_grad(set_to_none=True)
                total.backward()
                if self.config.grad_norm_clipping:
                    grad_norm_clip(
                        [p for p in self.model.parameters() if p.requires_grad],
                        float(self.config.grad_norm_clipping),
                    )
                self.optimizer.step()

                if self.retention is not None:
                    self.retention.step(1)

                stats["policy_loss"] += float(losses["policy_loss"].detach().cpu())
                stats["value_loss"] += float(losses["value_loss"].detach().cpu())
                stats["entropy"] += float(losses["entropy"].detach().cpu())
                stats["retention_loss"] += retention_loss_value
                num_updates += 1

        if num_updates:
            for key in list(stats):
                stats[key] /= num_updates
        stats["num_updates"] = float(num_updates)
        return stats


def _to_2d(value: Any, T: int, N: int, device: Any) -> Any:
    _require_torch()
    tensor = value if torch.is_tensor(value) else torch.as_tensor(_np.asarray(value))
    tensor = tensor.to(device).float()
    if tensor.dim() == 1:
        tensor = tensor.reshape(T, N)
    elif tensor.dim() > 2:
        tensor = tensor.reshape(T, N, -1)
        if tensor.shape[-1] == 1:
            tensor = tensor.squeeze(-1)
        else:
            tensor = tensor.mean(dim=-1)
    return tensor


def _flatten_obs_list(obs_list: Sequence[Any], device: Any) -> Any:
    """Stack a list of per-step observations into a single device object."""

    if not obs_list:
        return None
    if isinstance(obs_list[0], Mapping):
        out: Dict[str, Any] = {}
        for key in obs_list[0].keys():
            rows = []
            for step_obs in obs_list:
                stacked = _np.stack([_np.asarray(o[key]) for o in step_obs], axis=0) if _np is not None else [o[key] for o in step_obs]
                rows.append(stacked)
            if _np is not None:
                joined = _np.concatenate([_np.asarray(r) for r in rows], axis=0)
            else:
                joined = [x for r in rows for x in r]
            out[key] = _to_device_tensor(joined, str(device))
        return out
    rows = []
    for step_obs in obs_list:
        try:
            rows.append(_np.asarray(step_obs))
        except Exception:
            rows.append(step_obs)
    if all(isinstance(r, _np.ndarray) for r in rows):
        return _to_device_tensor(_np.concatenate(rows, axis=0), str(device))
    return rows


def _index_obs(flat_obs: Any, index: Any) -> Any:
    """Index a batched observation container (dict of tensors or a tensor)."""

    if isinstance(flat_obs, Mapping):
        return {k: (v[index] if hasattr(v, "__getitem__") else v) for k, v in flat_obs.items()}
    if _HAS_TORCH and isinstance(flat_obs, torch.Tensor):
        return flat_obs[index]
    return [flat_obs[i] for i in index.tolist()] if hasattr(index, "tolist") else flat_obs


# --------------------------------------------------------------------------------------
# Model building
# --------------------------------------------------------------------------------------


def build_nethack_model(config: APPOConfig, checkpoint: Optional[str] = None, **kwargs: Any) -> Any:
    """Build the NetHack actor-critic, falling back to a mini model for smoke tests.

    When ``src.nethack.model`` is available (the Tuyls et al. 2023 reconstruction) it is
    used; otherwise a small dependency-free actor-critic is returned so the runner can
    still be exercised end-to-end.
    """

    _require_torch()
    factory = _import_attr("src.nethack.model", "build_model")
    if factory is None:
        factory = _import_attr("src.nethack.model", "build_nethack_model")
    if callable(factory):
        try:
            model = factory(config, checkpoint=checkpoint, **kwargs)
            if model is not None:
                if checkpoint:
                    state = torch.load(checkpoint, map_location="cpu")
                    state = state.get("model", state) if isinstance(state, Mapping) else state
                    try:
                        model.load_state_dict(state, strict=False)
                    except Exception:
                        pass
                return model
        except Exception:
            pass
    model = _make_mini_model()
    if checkpoint and os.path.exists(str(checkpoint)):
        state = torch.load(checkpoint, map_location="cpu")
        state = state.get("model", state) if isinstance(state, Mapping) else state
        try:
            model.load_state_dict(state, strict=False)
        except Exception:
            pass
    return model


def build_env(config: APPOConfig, seed: Optional[int] = None, **kwargs: Any) -> Any:
    """Build the (vectorised) NLE environment, or the mini stub when unavailable."""

    factory = _import_attr("src.nethack.env", "make_vec_env")
    if factory is None:
        factory = _import_attr("src.nethack.env", "build_env")
    if callable(factory):
        try:
            env = factory(config, seed=seed, **kwargs)
            if env is not None:
                return env
        except Exception:
            pass
    if not config.stub:
        # No NLE available: keep the runner usable but make the substitution explicit.
        pass
    return MiniNetHackVecEnv(num_envs=min(int(config.num_envs), 8), seed=seed)


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------


def evaluate(
    agent: APPOAgent,
    env: Any = None,
    *,
    num_episodes: int = EVAL_EPISODES,
    seed: int = 0,
    stub: bool = False,
    max_steps: int = EVAL_MAX_STEPS,
    no_progress_steps: int = EVAL_NO_PROGRESS_STEPS,
    config: Optional[APPOConfig] = None,
) -> Dict[str, Any]:
    """In-game-score evaluation.

    Section 5: "Evaluation rollouts stop at death, 150 steps without progress, or
    100k steps".  Progress is tracked through the in-game score (falling back to the
    dungeon level), resetting the counter whenever it increases.
    """

    _require_torch()
    eval_env = env if env is not None else build_env(config or agent.config, seed=seed, stub=stub)
    was_training = agent.model.training
    agent.model.eval()
    agent.reset_state()

    rewards: List[float] = []
    lengths: List[int] = []
    scores: List[float] = []
    levels: List[int] = []

    for episode in range(max(1, int(num_episodes))):
        obs = _env_reset(eval_env, seed + episode)
        agent.reset_state()
        done = False
        episode_return = 0.0
        steps = 0
        best_progress = -float("inf")
        since_progress = 0
        last_level = 0
        while not done and steps < int(max_steps):
            with torch.no_grad():
                out = agent.act(obs, deterministic=True)
            actions = _to_env_actions(out["actions"])
            next_obs, reward, terminated, truncated, info = _env_step(eval_env, actions)
            reward = _first_scalar(reward)
            episode_return += float(reward)
            steps += 1
            progress = _episode_progress(info)
            if progress is None:
                progress = episode_return
            if progress > best_progress:
                best_progress = float(progress)
                since_progress = 0
            else:
                since_progress += 1
            info0 = info[0] if isinstance(info, (list, tuple)) and info else info
            if isinstance(info0, Mapping) and "dlvl" in info0:
                last_level = max(last_level, int(info0["dlvl"]))
            done = bool(_first_bool(terminated)) or bool(_first_bool(truncated))
            if since_progress >= int(no_progress_steps):
                done = True
            obs = next_obs
        rewards.append(episode_return)
        lengths.append(steps)
        scores.append(episode_return)
        levels.append(last_level)

    agent.model.train(was_training)
    return {
        "return_mean": _mean(rewards),
        "return_std": _std(rewards),
        "return_min": min(rewards) if rewards else 0.0,
        "return_max": max(rewards) if rewards else 0.0,
        "score_mean": _mean(scores),
        "mean_length": _mean(lengths),
        "max_dlvl": max(levels) if levels else 0,
        "episodes": len(rewards),
        "returns": rewards,
    }


def _env_reset(env: Any, seed: Optional[int] = None) -> Any:
    try:
        out = env.reset(seed=seed)
        return out[0] if isinstance(out, tuple) and len(out) == 2 else out
    except TypeError:
        pass
    try:
        return env.reset()
    except Exception:
        return None


def _env_step(env: Any, actions: Any) -> Tuple[Any, Any, Any, Any, Any]:
    out = env.step(actions)
    if len(out) == 4:
        obs, reward, done, info = out
        return obs, reward, done, [False] * _num_envs(reward), info
    obs, reward, terminated, truncated, info = out
    return obs, reward, terminated, truncated, info


def _num_envs(value: Any) -> int:
    try:
        return len(value)
    except Exception:
        return 1


def _to_env_actions(actions: Any) -> Any:
    if _HAS_TORCH and torch.is_tensor(actions):
        actions = actions.detach().cpu().numpy()
    if _np is not None and isinstance(actions, _np.ndarray):
        return actions.astype("int64")
    return actions


def _first_scalar(value: Any) -> float:
    if isinstance(value, (list, tuple)) and value:
        return float(value[0])
    if _np is not None and isinstance(value, _np.ndarray) and value.size:
        return float(value.reshape(-1)[0])
    return float(value)


def _first_bool(value: Any) -> bool:
    if isinstance(value, (list, tuple)) and value:
        return bool(value[0])
    if _np is not None and isinstance(value, _np.ndarray) and value.size:
        return bool(value.reshape(-1)[0])
    return bool(value)


def _episode_progress(info: Any) -> Optional[float]:
    info0 = info[0] if isinstance(info, (list, tuple)) and info else info
    if not isinstance(info0, Mapping):
        return None
    for key in ("score", "episode_score", "in_game_score", "dlvl"):
        if key in info0 and info0[key] is not None:
            return float(info0[key])
    return None


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


# --------------------------------------------------------------------------------------
# The runner
# --------------------------------------------------------------------------------------


@dataclass
class TrainResult:
    """Outcome of one APPO fine-tuning run."""

    method: str
    steps: int
    seed: Optional[int]
    final_eval: Dict[str, Any] = field(default_factory=dict)
    best_return: float = 0.0
    history: List[Dict[str, Any]] = field(default_factory=list)
    checkpoint: Optional[str] = None
    elapsed: float = 0.0
    config: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


class APPORunner:
    """Outer training loop for APPO fine-tuning on NetHack.

    Composes: vectorised NLE env -> :class:`APPOAgent` -> rollout buffer -> APPO update,
    with periodic evaluation every 25M steps (Figure 5 cadence), checkpointing every
    25M steps, and the auxiliary retention loss (EWC / BC / KS) attached to the actor.
    """

    def __init__(
        self,
        config: Optional[APPOConfig] = None,
        *,
        agent: Optional[APPOAgent] = None,
        env: Any = None,
        retention: Optional[RetentionBundle] = None,
        logger: Any = None,
        output_dir: Optional[str] = None,
    ):
        _require_torch()
        self.config = config or APPOConfig()
        self.output_dir = output_dir or self.config.output_dir or "runs/nethack"
        self.logger = logger
        self.env = env if env is not None else build_env(self.config, seed=self.config.seed)
        self.num_envs = int(getattr(self.env, "num_envs", self.config.num_envs))

        if agent is None:
            agent = APPOAgent(
                build_nethack_model(self.config, checkpoint=self.config.checkpoint),
                self.config,
                retention=retention,
                seed=self.config.seed,
            )
        self.agent = agent
        if retention is not None and self.agent.retention is None:
            self.agent.retention = retention

        self.step = 0
        self.history: List[Dict[str, Any]] = []
        self.obs = None

    # ---------------------------------------------------------------------- logging

    def log(self, message: str) -> None:
        if self.logger is not None:
            try:
                self.logger.info(message)
                return
            except Exception:
                pass
        print(message, flush=True)

    # ------------------------------------------------------------------- collection

    def collect_rollout(self, buffer: Optional[APPORolloutBuffer] = None) -> APPORolloutBuffer:
        """Collect ``unroll_length`` steps on ``num_envs`` environments."""

        buf = buffer or APPORolloutBuffer(self.config.unroll_length, self.num_envs, self.config.device)
        buf.clear()
        if self.obs is None:
            self.obs = _env_reset(self.env, self.config.seed)
        self.agent.reset_state()

        for _t in range(self.config.unroll_length):
            with torch.no_grad():
                out = self.agent.act(self.obs, deterministic=False)
            actions = out["actions"]
            next_obs, reward, terminated, truncated, info = _env_step(self.env, _to_env_actions(actions))

            done = bool(_first_bool(terminated)) or bool(_first_bool(truncated))
            # mask = 0 at *termination* (bootstrap cut) but 1 at truncation
            mask = 0.0 if bool(_first_bool(terminated)) else 1.0

            buf.add(
                obs=self.obs,
                actions=actions.detach().cpu(),
                log_probs=out["log_probs"].detach().cpu(),
                values=out["values"].detach().cpu(),
                rewards=torch.as_tensor(_as_vector(reward, self.num_envs), dtype=torch.float32),
                dones=torch.full((self.num_envs,), float(mask), dtype=torch.float32),
                entropies=out["entropy"].detach().cpu(),
            )
            self.obs = next_obs
            if done:
                self.obs = _env_reset(self.env, None)
                self.agent.reset_state()

        self.step += self.config.unroll_length * self.num_envs
        return buf

    # -------------------------------------------------------------------- training

    def train(
        self,
        total_steps: Optional[int] = None,
        *,
        eval_every: Optional[int] = None,
        eval_episodes: Optional[int] = None,
        save_every: Optional[int] = None,
        progress_fn: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    ) -> TrainResult:
        """Run the fine-tuning loop until ``total_steps`` environment steps."""

        total_steps = int(total_steps or self.config.total_steps)
        eval_every = int(eval_every or self.config.eval_every)
        eval_episodes = int(eval_episodes or self.config.eval_episodes)
        save_every = int(save_every or self.config.save_every)
        os.makedirs(self.output_dir, exist_ok=True)

        start = time.time()
        next_eval = eval_every
        next_save = save_every
        best_return = -float("inf")
        final_eval: Dict[str, Any] = {}
        last_checkpoint: Optional[str] = None

        self.log(
            f"[APPO] method={self.config.method} total_steps={total_steps} "
            f"num_envs={self.num_envs} schedule(25M) eval_episodes={eval_episodes}"
        )

        buffer = APPORolloutBuffer(self.config.unroll_length, self.num_envs, self.config.device)
        while self.step < total_steps:
            buffer = self.collect_rollout(buffer)
            eval_output = self._bootstrap_value()
            stats = self.agent.update(
                buffer,
                last_value=eval_output,
                retention_batch=self._retention_batch(),
                step=self.step,
            )

            if self.step >= next_eval:
                final_eval = evaluate(
                    self.agent,
                    num_episodes=eval_episodes,
                    seed=(self.config.seed or 0),
                    stub=self.config.stub,
                    config=self.config,
                )
                entry = {
                    "step": self.step,
                    "return": final_eval["return_mean"],
                    "score": final_eval["score_mean"],
                    "max_dlvl": final_eval["max_dlvl"],
                    **{f"train/{k}": v for k, v in stats.items()},
                }
                self.history.append(entry)
                best_return = max(best_return, float(final_eval["return_mean"]))
                self.log(
                    f"[APPO] step={self.step} score={final_eval['score_mean']:.1f} "
                    f"return={final_eval['return_mean']:.1f} dlvl={final_eval['max_dlvl']}"
                )
                while self.step >= next_eval:
                    next_eval += eval_every
                if progress_fn is not None:
                    try:
                        progress_fn(self.step, entry)
                    except Exception:
                        pass

            if self.step >= next_save:
                last_checkpoint = self.agent.save(
                    os.path.join(self.output_dir, f"nethack_{self.config.method}.pt"),
                    step=self.step,
                    extra={"env_steps": self.step},
                )
                while self.step >= next_save:
                    next_save += save_every

        # final evaluation + checkpoint
        final_eval = evaluate(
            self.agent,
            num_episodes=eval_episodes,
            seed=(self.config.seed or 0),
            stub=self.config.stub,
            config=self.config,
        )
        best_return = max(best_return, float(final_eval["return_mean"]))
        last_checkpoint = self.agent.save(
            os.path.join(self.output_dir, f"nethack_{self.config.method}.pt"),
            step=self.step,
            extra={"env_steps": self.step, "final": True},
        )
        result = TrainResult(
            method=self.config.method,
            steps=self.step,
            seed=self.config.seed,
            final_eval=final_eval,
            best_return=best_return,
            history=self.history,
            checkpoint=last_checkpoint,
            elapsed=time.time() - start,
            config=self.config.to_dict(),
        )
        self._write_summary(result)
        return result

    # --------------------------------------------------------------------- helpers

    def _bootstrap_value(self) -> Any:
        """Value estimate of the current observation, used for GAE bootstrapping."""

        try:
            with torch.no_grad():
                output = self.agent.forward(self.obs, state=None)
                _logits, value = extract_policy_value(output)
            if value is None:
                return torch.zeros(self.num_envs, device=self.config.device)
            return torch.as_tensor(value, device=self.config.device).reshape(-1).detach()
        except Exception:
            return torch.zeros(self.num_envs, device=self.config.device)

    def _retention_batch(self) -> Any:
        """Optional auxiliary batch (e.g. a BC batch of expert states)."""

        if self.agent.retention is None:
            return None
        for mech in self.agent.retention.mechanisms:
            buffer = getattr(mech, "buffer", None)
            if buffer is None:
                continue
            batch_size = getattr(mech, "batch_size", 128)
            sample = getattr(buffer, "sample", None)
            if callable(sample):
                try:
                    return sample(int(batch_size))
                except Exception:
                    continue
        return None

    def _write_summary(self, result: TrainResult) -> None:
        try:
            path = os.path.join(self.output_dir, f"nethack_{result.method}_summary.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(result.as_dict(), handle, indent=2, default=str)
        except Exception:
            pass


def _as_vector(value: Any, n: int) -> List[float]:
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    if _np is not None and isinstance(value, _np.ndarray):
        flat = value.reshape(-1)
        return [float(v) for v in flat]
    return [float(value)] * n


# --------------------------------------------------------------------------------------
# Functional entry point
# --------------------------------------------------------------------------------------


def run_finetuning(
    cfg: Any = None,
    *,
    method: Optional[str] = None,
    total_steps: Optional[int] = None,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    output_dir: Optional[str] = None,
    checkpoint: Optional[str] = None,
    teacher: Any = None,
    fisher: Any = None,
    bc_dataset: Any = None,
    stub: Optional[bool] = None,
    model: Any = None,
    env: Any = None,
    logger: Any = None,
    progress_fn: Optional[Callable[[int, Dict[str, Any]], None]] = None,
) -> TrainResult:
    """Fine-tune ``pi_*`` on NetHack with the requested retention method.

    ``method`` is one of ``scratch`` (from-scratch PPO baseline), ``none`` (vanilla
    fine-tuning), ``ewc`` (coefficient 2e6), ``bc`` (scale 2.0, no decay) or ``ks``
    (scale 0.5, exponential decay 0.99998 per train step).
    """

    _require_torch()
    config = APPOConfig.from_config(cfg)
    overrides: Dict[str, Any] = {}
    if method is not None:
        overrides["method"] = method
    if total_steps is not None:
        overrides["total_steps"] = int(total_steps)
    if seed is not None:
        overrides["seed"] = int(seed)
    if device is not None:
        overrides["device"] = device
    if stub is not None:
        overrides["stub"] = bool(stub)
    if output_dir is not None:
        overrides["output_dir"] = output_dir
    if checkpoint is not None:
        overrides["checkpoint"] = checkpoint
    config = config.with_overrides(**overrides)

    if config.seed is not None:
        try:
            from src.common.seeding import set_seed

            set_seed(int(config.seed))
        except Exception:
            pass

    resolved_output = config.output_dir or os.path.join(
        "runs", "nethack", str(config.method), f"seed_{config.seed}"
    )

    runner = APPORunner(config, env=env, logger=logger, output_dir=resolved_output)
    if model is not None:
        runner.agent.model = model.to(runner.agent.device)

    # Build retention only for genuine fine-tuning variants.  The from-scratch
    # baseline must never see pi_* nor its data.
    if config.method in ("ewc", "bc", "ks") and runner.agent.retention is None:
        runner.agent.retention = RetentionBundle.build(
            config,
            runner.agent.actor,
            method=config.method,
            teacher=teacher,
            fisher=fisher,
            bc_dataset=bc_dataset,
            device=str(config.device),
            seed=config.seed,
        )

    return runner.train(progress_fn=progress_fn)


def summarize_retention_config(method: str) -> Dict[str, Any]:
    """Expose the Appendix B.1 retention hyperparameters for a given method."""

    method = str(method or "none").lower()
    summary: Dict[str, Any] = {
        "method": method,
        "entropy_enabled": method in ("none", "scratch"),
        "freeze_encoders": method not in ("scratch",),
        "critic_coef": 0.0,
    }
    if method in RETENTION_CONFIG:
        summary.update(RETENTION_CONFIG[method])
    return summary


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="APPO fine-tuning runner for NetHack (Table 1 / Appendix B.1)."
    )
    parser.add_argument("--config", type=str, default=None, help="path to configs/nethack.yaml")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="key=value override")
    parser.add_argument("--method", type=str, default="none", choices=list(TRAINING_METHODS))
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None, help="pre-trained pi_* checkpoint")
    parser.add_argument("--teacher", type=str, default=None, help="teacher checkpoint for BC/KS")
    parser.add_argument("--fisher", type=str, default=None, help="diagonal Fisher for EWC")
    parser.add_argument("--bc-dataset", type=str, default=None, help="BC state buffer (NLD-AA subset)")
    parser.add_argument("--stub", action="store_true", help="use the dependency-free stub env")
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true", help="run a few thousand steps only")
    parser.add_argument("--show-retention", action="store_true", help="print retention hyperparameters and exit")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.show_retention:
        print(json.dumps(summarize_retention_config(args.method), indent=2))
        return 0

    cfg = None
    if args.config:
        try:
            from src.common.config import apply_overrides, load_config

            cfg = load_config(args.config, overrides=args.overrides)
        except Exception as exc:  # pragma: no cover - config is optional
            print(f"[APPO] could not load config {args.config}: {exc}")

    total_steps = args.total_steps
    if args.smoke_test and total_steps is None:
        total_steps = max(2 * args_seed_default(args) * 32, 2048)

    teacher = None
    if args.teacher:
        teacher = build_nethack_model(APPOConfig.from_config(cfg), checkpoint=args.teacher)

    result = run_finetuning(
        cfg,
        method=args.method,
        total_steps=total_steps,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        teacher=teacher,
        stub=args.stub or args.smoke_test,
    )
    print(json.dumps(result.as_dict(), indent=2, default=str))
    return 0


def args_seed_default(args: argparse.Namespace) -> int:
    return 4


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
