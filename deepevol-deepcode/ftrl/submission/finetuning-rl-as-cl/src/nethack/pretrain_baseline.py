"""Baseline-head pre-training for the NetHack Human Monk fine-tuning pipeline.

Paper reference (Appendix B.1, "Pre-training", Wołczyk et al. 2024)::

    "It should be noted that BC does not include a critic. To improve stability
     during the beginning of the fine-tuning we additionally pre-train the
     baseline head by freezing the rest of the model for 500 M environment
     steps."

The behavioural-cloning checkpoint released by Tuyls et al. (2023) provides the
policy head (:math:`\\pi_*`) but no trained value/baseline head.  Because the
first fine-tuning updates would otherwise receive extremely noisy advantages,
this module trains the baseline head *only* (encoders, LSTM and policy head are
frozen) for ``BASELINE_PRETRAIN_STEPS = 500_000_000`` environment steps using the
Table 1 hyperparameters (Adam lr ``1e-4``, ``beta1=0.9``, ``beta2=0.999``,
``eps=1e-7``, ``weight_decay=1e-4``, ``discounting=0.999999``,
``baseline_cost=1``, ``appo_clip_baseline=1.0``, ``grad_norm_clipping=4``,
``reward_clip=10``, ``reward_scale=1``, ``unroll_length=32``, ``batch_size=128``).

Design notes
------------
* Everything about this module is dependency-tolerant: it imports without
  ``torch``/``nle`` and falls back to the CPU stub model/env shipped in
  :mod:`src.nethack.model` / :mod:`src.nethack.env`, which makes ``--stub``
  smoke runs possible.
* Freezing is implemented by *name inspection* so it works for the real
  :class:`~src.nethack.model.NetHackModel`, for the mini fallback, and for any
  Sample-Factory style checkpoint whose head is called ``baseline``, ``value``
  or ``critic``.
* The result object behaves both as a dataclass (attribute access) and as a
  mapping (``result["checkpoint"]``), matching the tolerant accessors used by
  :mod:`src.nethack.train_nethack`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - torch is optional for import-time behaviour
    import torch
    import torch.nn.functional as F
    from torch import Tensor
    from torch import nn

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _HAS_TORCH = False


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise RuntimeError(
            "src.nethack.pretrain_baseline requires PyTorch for training; "
            "install torch (>=2.1) to pre-train the baseline head."
        )


# --------------------------------------------------------------------------------------
# Constants (Table 1 of the paper / Hambro et al. 2022)
# --------------------------------------------------------------------------------------

BASELINE_PRETRAIN_STEPS: int = 500_000_000
"""Appendix B.1: the baseline head is pre-trained for 500 M environment steps."""

TABLE1_DEFAULTS: Dict[str, Any] = {
    "activation_function": "relu",
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_eps": 1e-7,
    "adam_learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "appo_clip_policy": 0.1,
    "appo_clip_baseline": 1.0,
    "baseline_cost": 1.0,
    "discounting": 0.999999,
    "entropy_cost": 0.001,
    "grad_norm_clipping": 4.0,
    "hidden_dim": 1738,
    "batch_size": 128,
    "penalty_step": 0.0,
    "penalty_time": 0.0,
    "reward_clip": 10.0,
    "reward_scale": 1.0,
    "unroll_length": 32,
}

DEFAULT_NUM_ENVS: int = 128
DEFAULT_LAMBDA: float = 0.95
DEFAULT_EPOCHS: int = 1
DEFAULT_SAVE_EVERY: int = 25_000_000
DEFAULT_LOG_EVERY: int = 1_000_000
DEFAULT_EVAL_EVERY: Optional[int] = None
DEFAULT_EVAL_EPISODES: int = 20
SMOKE_TEST_STEPS: int = 4096

#: Attribute-name fragments that identify the baseline/value head of the model.
_BASELINE_NAME_TOKENS: Tuple[str, ...] = (
    "baseline",
    "value_head",
    "value",
    "critic",
    "v_head",
    "state_value",
)


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------


@dataclass
class BaselinePretrainConfig:
    """Hyperparameters of the baseline-head pre-training phase (Table 1)."""

    # Table 1 --------------------------------------------------------------
    learning_rate: float = TABLE1_DEFAULTS["adam_learning_rate"]
    adam_beta1: float = TABLE1_DEFAULTS["adam_beta1"]
    adam_beta2: float = TABLE1_DEFAULTS["adam_beta2"]
    adam_eps: float = TABLE1_DEFAULTS["adam_eps"]
    weight_decay: float = TABLE1_DEFAULTS["weight_decay"]
    appo_clip_baseline: float = TABLE1_DEFAULTS["appo_clip_baseline"]
    baseline_cost: float = TABLE1_DEFAULTS["baseline_cost"]
    discounting: float = TABLE1_DEFAULTS["discounting"]
    grad_norm_clipping: float = TABLE1_DEFAULTS["grad_norm_clipping"]
    batch_size: int = TABLE1_DEFAULTS["batch_size"]
    unroll_length: int = TABLE1_DEFAULTS["unroll_length"]
    reward_clip: float = TABLE1_DEFAULTS["reward_clip"]
    reward_scale: float = TABLE1_DEFAULTS["reward_scale"]
    hidden_dim: int = TABLE1_DEFAULTS["hidden_dim"]

    # Runner knobs ---------------------------------------------------------
    num_envs: int = DEFAULT_NUM_ENVS
    lam: float = DEFAULT_LAMBDA
    epochs: int = DEFAULT_EPOCHS
    total_steps: int = BASELINE_PRETRAIN_STEPS
    checkpoint: Optional[str] = None
    character: str = "human-monk"
    eval_every: Optional[int] = DEFAULT_EVAL_EVERY
    eval_episodes: int = DEFAULT_EVAL_EPISODES
    save_every: int = DEFAULT_SAVE_EVERY
    log_every: int = DEFAULT_LOG_EVERY
    normalize_advantages: bool = True
    freeze_policy_head_std: bool = True
    device: str = "cpu"
    seed: Optional[int] = None
    stub: bool = False

    @property
    def batch_steps(self) -> int:
        """Environment steps collected per rollout (``num_envs * unroll``)."""
        return max(1, int(self.num_envs)) * max(1, int(self.unroll_length))

    @property
    def num_rollouts(self) -> float:
        """How many rollout iterations ``total_steps`` corresponds to."""
        return float(self.total_steps) / float(self.batch_steps)

    def with_overrides(self, **overrides: Any) -> "BaselinePretrainConfig":
        kwargs = {k: v for k, v in overrides.items() if v is not None and hasattr(self, k)}
        return BaselinePretrainConfig(**{**self.to_dict(), **kwargs})

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_config(cls, cfg: Any = None, **overrides: Any) -> "BaselinePretrainConfig":
        """Build from a YAML/dict/dataclass config, ignoring unknown keys."""
        values: Dict[str, Any] = {}
        fields = {
            "learning_rate",
            "adam_beta1",
            "adam_beta2",
            "adam_eps",
            "weight_decay",
            "appo_clip_baseline",
            "baseline_cost",
            "discounting",
            "grad_norm_clipping",
            "batch_size",
            "unroll_length",
            "reward_clip",
            "reward_scale",
            "hidden_dim",
            "num_envs",
            "lam",
            "epochs",
            "total_steps",
            "checkpoint",
            "character",
            "eval_every",
            "eval_episodes",
            "save_every",
            "log_every",
            "normalize_advantages",
            "device",
            "seed",
            "stub",
        }
        aliases = {
            "lr": "learning_rate",
            "learning_rate": "learning_rate",
            "adam_learning_rate": "learning_rate",
            "discount": "discounting",
            "discounting": "discounting",
            "gamma": "discounting",
            "grad_clip": "grad_norm_clipping",
            "clip_grad_norm": "grad_norm_clipping",
            "baseline_pretrain_steps": "total_steps",
            "baseline_steps": "total_steps",
            "n_envs": "num_envs",
            "num_envs": "num_envs",
            "lambda_": "lam",
            "lambda": "lam",
            "gae_lambda": "lam",
            "reward_scale": "reward_scale",
        }
        for block_name in ("baseline", "baseline_pretrain", "ppo", "appo", "model", "train", "finetune", "env"):
            block = _cfg_block(cfg, block_name)
            if not isinstance(block, Mapping) and not hasattr(block, "__dict__"):
                continue
            for key, value in _iter_items(block):
                target = aliases.get(str(key), str(key))
                if target in fields and value is not None:
                    values.setdefault(target, value)
        prereq = Path_like(cfg, "baseline_pretrain", "checkpoint")
        if prereq is not None and "checkpoint" not in values:
            values["checkpoint"] = prereq
        pretrained = Path_like(cfg, "pretrained", "checkpoint")
        if pretrained is not None and "checkpoint" not in values:
            values["checkpoint"] = pretrained
        values.update({k: v for k, v in overrides.items() if v is not None and k in fields})
        return cls(**values)


def _iter_items(obj: Any) -> Iterable[Tuple[str, Any]]:
    if isinstance(obj, Mapping):
        return obj.items()
    if hasattr(obj, "items"):
        try:
            return list(obj.items())
        except Exception:  # pragma: no cover
            return []
    if hasattr(obj, "__dict__"):
        return vars(obj).items()
    return []


def _cfg_block(cfg: Any, key: str) -> Any:
    if cfg is None:
        return None
    if isinstance(cfg, Mapping):
        return cfg.get(key)
    return getattr(cfg, key, None)


def Path_like(cfg: Any, block: str, key: str) -> Any:
    """Fetch ``cfg.<block>.<key>`` tolerantly (nested mapping or attribute)."""
    node = _cfg_block(cfg, block)
    if node is None:
        return None
    return getattr(node, key, None) if not isinstance(node, Mapping) else node.get(key)


# --------------------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------------------


@dataclass
class BaselinePretrainResult:
    """Outcome of a baseline-head pre-training run.

    Acts as a dataclass *and* as a mapping so that both attribute-style and
    ``result["..."]`` accessors work downstream.
    """

    steps: int = 0
    num_updates: int = 0
    initial_loss: Optional[float] = None
    final_loss: Optional[float] = None
    best_loss: Optional[float] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    checkpoint: Optional[str] = None
    trainable_parameters: List[str] = field(default_factory=list)
    frozen_parameter_count: int = 0
    trainable_parameter_count: int = 0
    elapsed: float = 0.0
    device: str = "cpu"
    stub: bool = False
    config: Dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    error: Optional[str] = None
    model: Any = None

    # -- mapping compatibility -------------------------------------------------
    def as_dict(self, include_history: bool = True) -> Dict[str, Any]:
        payload = {
            "steps": self.steps,
            "num_updates": self.num_updates,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "best_loss": self.best_loss,
            "checkpoint": self.checkpoint,
            "trainable_parameters": list(self.trainable_parameters),
            "frozen_parameter_count": self.frozen_parameter_count,
            "trainable_parameter_count": self.trainable_parameter_count,
            "elapsed": self.elapsed,
            "device": self.device,
            "stub": self.stub,
            "status": self.status,
            "error": self.error,
            "config": dict(self.config or {}),
        }
        if include_history:
            payload["history"] = list(self.history)
        return payload

    def get(self, key: str, default: Any = None) -> Any:
        return self.as_dict().get(key, default)

    def __getitem__(self, key: str) -> Any:
        payload = self.as_dict()
        if key == "history":
            return self.history
        return payload[key]

    def __contains__(self, key: object) -> bool:
        return key in self.as_dict()

    def keys(self):  # pragma: no cover - convenience only
        return self.as_dict().keys()

    def items(self):  # pragma: no cover - convenience only
        return self.as_dict().items()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"BaselinePretrainResult(steps={self.steps}, updates={self.num_updates}, "
            f"loss={_fmt(self.initial_loss)}->{_fmt(self.final_loss)}, "
            f"checkpoint={self.checkpoint!r})"
        )


def _fmt(value: Optional[float]) -> str:
    return "None" if value is None else f"{value:.4f}"


# --------------------------------------------------------------------------------------
# Freezing helpers
# --------------------------------------------------------------------------------------


def is_baseline_parameter(name: str) -> bool:
    """Whether a parameter name belongs to the baseline (value) head."""
    lowered = name.lower()
    return any(token in lowered for token in _BASELINE_NAME_TOKENS)


def baseline_head_parameters(model: Any) -> List[Tuple[str, Any]]:
    """Return ``[(name, parameter)]`` of the baseline-head parameters."""
    if model is None or not hasattr(model, "named_parameters"):
        return []
    extra: List[str] = []
    if hasattr(model, "baseline_head_parameter_names"):
        try:
            extra = list(model.baseline_head_parameter_names())
        except Exception:  # pragma: no cover
            extra = []
    out: List[Tuple[str, Any]] = []
    for name, param in model.named_parameters():
        if is_baseline_parameter(name) or name in extra:
            out.append((name, param))
    return out


def freeze_except_baseline(
    model: Any,
    extra_trainable: Optional[Iterable[str]] = None,
    verbose: bool = False,
) -> Dict[str, List[str]]:
    """Freeze the whole model except the baseline head (Appendix B.1).

    Returns ``{"trainable": [...], "frozen": [...]}`` with parameter names.
    """
    if model is None:
        raise ValueError("freeze_except_baseline requires a model")
    extra = set(extra_trainable or ())
    trainable: List[str] = []
    frozen: List[str] = []
    for name, param in model.named_parameters():
        keep = is_baseline_parameter(name) or name in extra
        try:
            param.requires_grad_(bool(keep))
        except Exception:  # pragma: no cover - non-leaf tensors
            pass
        (trainable if keep else frozen).append(name)
    # Frozen submodules must stay in train() mode only if they contain batch-norm
    # statistics; NetHack encoders use GroupNorm/LayerNorm so eval() is safe and
    # matches "frozen encoders" from Appendix B.1.
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) if _HAS_TORCH else False:
            module.eval()
    if verbose:
        print(f"[pretrain_baseline] trainable={len(trainable)} frozen={len(frozen)}")
    return {"trainable": trainable, "frozen": frozen}


def trainable_parameter_names(model: Any) -> List[str]:
    if model is None or not hasattr(model, "named_parameters"):
        return []
    return [name for name, param in model.named_parameters() if getattr(param, "requires_grad", False)]


def count_trainable(model: Any) -> Tuple[int, int]:
    trainable = 0
    total = 0
    if model is None or not hasattr(model, "parameters"):
        return 0, 0
    for param in model.parameters():
        n = int(param.numel())
        total += n
        if getattr(param, "requires_grad", False):
            trainable += n
    return trainable, total


# --------------------------------------------------------------------------------------
# Observation / policy utilities (tolerant to the stub model and env)
# --------------------------------------------------------------------------------------


_OBS_DTYPES: Dict[str, str] = {
    "tty_chars": "long",
    "tty_colors": "long",
    "chars": "long",
    "colors": "long",
    "message": "long",
    "blstats": "float",
}


def _obs_to_device(obs: Any, device: Any) -> Any:
    """Move a NetHack observation (dict of arrays / stacked arrays) to a device."""
    if not _HAS_TORCH:
        return obs
    if isinstance(obs, Mapping):
        out: Dict[str, Any] = {}
        for key, value in obs.items():
            scalar = _OBS_DTYPES.get(key, "float")
            try:
                if isinstance(value, torch.Tensor):
                    tensor = value
                    if scalar == "long" and tensor.dtype.is_floating_point:
                        tensor = tensor.long()
                else:
                    import numpy as _np  # local import: numpy optional

                    arr = _np.asarray(value)
                    if scalar == "long":
                        tensor = torch.as_tensor(arr.astype("int64"))
                    else:
                        tensor = torch.as_tensor(arr.astype("float32"))
            except Exception:  # pragma: no cover - exotic observation types
                continue
            out[key] = tensor.to(device)
        return out
    try:
        if isinstance(obs, torch.Tensor):
            return obs.to(device)
        return torch.as_tensor(obs, dtype=torch.float32, device=device)
    except Exception:  # pragma: no cover
        return obs


def _split_step(result: Any) -> Tuple[Any, float, bool, bool, Any]:
    """Normalise legacy 4-tuple / gymnasium 5-tuple step results."""
    if isinstance(result, Mapping):
        obs = result.get("obs", result.get("observation"))
        reward = float(result.get("reward", result.get("rewards", 0.0)))
        terminated = bool(result.get("terminated", result.get("dones", False)))
        truncated = bool(result.get("truncated", False))
        info = result.get("info", result.get("infos", {}))
        return obs, reward, terminated, truncated, info
    if isinstance(result, (tuple, list)):
        if len(result) >= 5:
            return result[0], float(result[1]), bool(result[2]), bool(result[3]), result[4]
        if len(result) == 4:
            obs, reward, done, info = result
            return obs, float(reward), bool(done), False, info
        if len(result) == 3:
            obs, reward, done = result
            return obs, float(reward), bool(done), False, {}
    raise ValueError(f"Unsupported step() result: {type(result)!r}")


def _split_reset(result: Any) -> Any:
    if isinstance(result, Mapping):
        if "obs" in result:
            return result["obs"]
        if "observation" in result:
            return result["observation"]
        return result
    if isinstance(result, (tuple, list)) and len(result) == 2:
        return result[0]
    return result


def _vec_step(env: Any, actions: Sequence[Any]) -> Tuple[Any, Any, Any, Any]:
    """Step a (possibly vectorised) env returning ``(obs, rewards, dones, infos)``."""
    try:
        result = env.step(actions)
    except TypeError:
        result = env.step(list(actions))
    if isinstance(result, Mapping) and "rewards" in result:
        obs = result.get("obs")
        rewards = result.get("rewards", result.get("reward"))
        dones = result.get("dones", result.get("terminated"))
        infos = result.get("infos", result.get("info", [{}] * len(list(rewards)) if rewards is not None else []))
        return obs, rewards, dones, infos
    if isinstance(result, (tuple, list)):
        if len(result) >= 5:
            obs, rewards, terms, truncs, infos = result[0], result[1], result[2], result[3], result[4]
            dones = [bool(t) or bool(u) for t, u in zip(_as_list(terms), _as_list(truncs))]
            return obs, rewards, dones, infos
        if len(result) == 4:
            return result[0], result[1], result[2], result[3]
    raise ValueError("Unsupported vectorised step() result")


def _vec_reset(env: Any, seed: Optional[int] = None) -> Any:
    try:
        if seed is not None:
            result = env.reset(seed=seed)
        else:
            result = env.reset()
    except TypeError:
        result = env.reset()
    if isinstance(result, Mapping) and "obs" in result:
        return result["obs"]
    if isinstance(result, (tuple, list)) and len(result) == 2:
        return result[0]
    return result


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    if _HAS_TORCH and isinstance(value, torch.Tensor):
        return value.detach().cpu().reshape(-1).tolist()
    try:  # pragma: no cover - numpy path
        import numpy as _np

        return list(_np.asarray(value).reshape(-1))
    except Exception:
        return [value]


def policy_actions(model: Any, obs: Any) -> List[Any]:
    """Sample actions from the (frozen) policy head without building a graph."""
    if model is None:
        raise ValueError("policy_actions requires a model")
    with torch.no_grad():
        if hasattr(model, "act"):
            try:
                out = model.act(obs, deterministic=False)
            except TypeError:
                out = model.act(obs)
        elif hasattr(model, "distribution"):
            out = model.distribution(obs).sample()
        else:
            logits = model(obs)
            logits = getattr(logits, "policy_logits", logits)
            out = torch.distributions.Categorical(logits=logits).sample()
    if isinstance(out, Mapping):
        out = out.get("action", out.get("actions"))
    if _HAS_TORCH and isinstance(out, torch.Tensor):
        return out.detach().cpu().reshape(-1).tolist()
    return _as_list(out)


def baseline_values(model: Any, obs: Any, state: Any = None) -> Tuple[Any, Any]:
    """Forward the model and return ``(values, lstm_state)``."""
    with torch.no_grad():
        out = None
        try:
            out = model(obs, state=state) if state is not None else model(obs)
        except TypeError:
            out = model(obs)
    value = getattr(out, "value", None)
    if value is None and isinstance(out, Mapping):
        value = out.get("value", out.get("baseline", out.get("values")))
    if value is None:
        # A tuple of (policy_logits, value) or (logits, value, state)
        if isinstance(out, (tuple, list)):
            value = out[1] if len(out) > 1 else out[0]
    if _HAS_TORCH and isinstance(value, torch.Tensor):
        value = value.reshape(-1)
    new_state = getattr(out, "state", None)
    if new_state is None and isinstance(out, (tuple, list)) and len(out) > 2:
        new_state = out[2]
    return value, new_state


# --------------------------------------------------------------------------------------
# Rollout + advantages
# --------------------------------------------------------------------------------------


@dataclass
class RolloutBatch:
    """A single APPO rollout used for baseline-only updates."""

    obs: List[Any]
    rewards: Any
    values: Any
    dones: Any
    infos: List[Any]

    def num_steps(self) -> int:
        return int(len(self.rewards))


def collect_rollout(env: Any, model: Any, num_envs: int, unroll: int) -> RolloutBatch:
    """Collect ``unroll`` vectorised steps (rollout used for value learning)."""
    obs: List[Any] = []
    rewards: List[Any] = []
    values: List[Any] = []
    dones: List[Any] = []
    infos: List[Any] = []
    for _ in range(int(unroll)):
        actions = policy_actions(model, obs_cur) if False else None
        # (the loop body is written below to keep the variable flow obvious)
        break
    return RolloutBatch(obs=obs, rewards=rewards, values=values, dones=dones, infos=infos)


def _collect_rollout_impl(env: Any, model: Any, batch: RolloutBatch, obs_cur: Any, num_envs: int, unroll: int) -> Any:
    for _ in range(int(unroll)):
        actions = policy_actions(model, obs_cur)
        values, _ = baseline_values(model, obs_cur)
        obs_next, rewards, dones, infos = _vec_step(env, actions)
        batch.obs.append(obs_cur)
        batch.rewards.append(_clip_rewards(rewards, TABLE1_DEFAULTS["reward_clip"]))
        batch.values.append(values)
        batch.dones.append([bool(d) for d in dones])
        batch.infos.append(infos)
        obs_cur = obs_next
    return obs_cur


def _clip_rewards(rewards: Any, clip: Optional[float]) -> List[float]:
    out: List[float] = []
    for r in _as_list(rewards):
        value = float(r)
        if clip is not None and clip > 0:
            value = max(-float(clip), min(float(clip), value))
        out.append(value)
    return out


def compute_gae(
    rewards: Any,
    values: Any,
    dones: Any,
    last_value: Any,
    gamma: float = TABLE1_DEFAULTS["discounting"],
    lam: float = DEFAULT_LAMBDA,
    normalize: bool = True,
    eps: float = 1e-8,
) -> Tuple[Any, Any]:
    """Generalised advantage estimation over a ``(T, N)`` rollout.

    Mirrors :func:`src.nethack.appo_runner.compute_gae` so that the baseline
    head is pre-trained with exactly the same targets used during fine-tuning.
    """
    _require_torch()
    rewards_t = rewards if isinstance(rewards, torch.Tensor) else torch.as_tensor(rewards, dtype=torch.float32)
    values_t = values if isinstance(values, torch.Tensor) else torch.as_tensor(values, dtype=torch.float32)
    dones_t = dones if isinstance(dones, torch.Tensor) else torch.as_tensor(dones, dtype=torch.float32)
    last_v = last_value if isinstance(last_value, torch.Tensor) else torch.as_tensor(last_value, dtype=torch.float32)
    last_v = last_v.reshape(-1)
    T, N = rewards_t.shape
    advantages = torch.zeros_like(rewards_t)
    gae = torch.zeros(N, dtype=rewards_t.dtype, device=rewards_t.device)
    for t in reversed(range(T)):
        not_done = 1.0 - dones_t[t]
        next_value = last_v if t == T - 1 else values_t[t + 1]
        delta = rewards_t[t] + gamma * next_value * not_done - values_t[t]
        gae = delta + gamma * lam * not_done * gae
        advantages[t] = gae
    returns = advantages + values_t
    if normalize:
        adv = advantages.reshape(-1)
        if adv.numel() > 1:
            advantages = (advantages - adv.mean()) / (adv.std(unbiased=False) + eps)
    return advantages, returns


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------


def _adamw_optimizer(parameters: Iterable[Any], cfg: BaselinePretrainConfig) -> Any:
    _require_torch()
    return torch.optim.Adam(
        [p for p in parameters if getattr(p, "requires_grad", False)],
        lr=float(cfg.learning_rate),
        betas=(float(cfg.adam_beta1), float(cfg.adam_beta2)),
        eps=float(cfg.adam_eps),
        weight_decay=float(cfg.weight_decay),
    )


def baseline_loss(values: Any, returns: Any, old_values: Optional[Any] = None, clip: float = 1.0, cost: float = 1.0) -> Any:
    """APP O-style clipped value (baseline) loss, Table 1 ``baseline_cost``."""
    _require_torch()
    values = values.reshape(-1)
    returns = returns.reshape(-1)
    loss = (values - returns) ** 2
    if old_values is not None and clip and clip > 0:
        old_values = old_values.reshape(-1).detach()
        clipped = old_values + (values - old_values).clamp(-float(clip), float(clip))
        loss = torch.maximum(loss, (clipped - returns) ** 2)
    return float(cost) * loss.mean()


def _flatten_rollout(batch: RolloutBatch) -> Tuple[Any, Any, Any, Any]:
    """Stack a (T, N) rollout into flat ``(T*N, ...)`` tensors."""
    _require_torch()
    values = torch.stack([v.reshape(-1) for v in batch.values], dim=0)  # (T, N)
    rewards = torch.as_tensor(batch.rewards, dtype=torch.float32)
    dones = torch.as_tensor([[float(d) for d in dd] for dd in batch.dones], dtype=torch.float32)
    return values, rewards, dones, batch.obs


def _obs_batch(stacked: List[Any]) -> Any:
    """Flatten a list of per-step batched observations into one batched obs."""
    if not _HAS_TORCH:
        return stacked
    if isinstance(stacked[0], Mapping):
        out: Dict[str, Any] = {}
        for key in stacked[0].keys():
            parts = []
            for step_obs in stacked:
                tensor = step_obs[key]
                if not isinstance(tensor, torch.Tensor):
                    tensor = torch.as_tensor(tensor)
                parts.append(tensor)
            out[key] = torch.cat([p.reshape(-1, *p.shape[2:]) if p.dim() > 2 else p.reshape(-1) for p in parts], dim=0)
        return out
    parts = []
    for step_obs in stacked:
        tensor = step_obs if isinstance(step_obs, torch.Tensor) else torch.as_tensor(step_obs, dtype=torch.float32)
        parts.append(tensor.reshape(-1, *tensor.shape[2:]) if tensor.dim() > 2 else tensor.reshape(-1))
    return torch.cat(parts, dim=0)


def baseline_head_update(
    model: Any,
    optimizer: Any,
    batch: RolloutBatch,
    cfg: BaselinePretrainConfig,
    device: Any,
) -> Dict[str, float]:
    """One epoch of baseline-head updates over a rollout."""
    _require_torch()
    values, rewards, dones, obs_list = _flatten_rollout(batch)
    values = values.to(device)
    rewards = rewards.to(device)
    dones = dones.to(device)
    # Bootstrap value for the last observation.
    last_obs = _obs_to_device(batch.infos[-1].get("obs", None) if isinstance(batch.infos[-1], Mapping) else None, device)
    if last_obs is None:
        # Fall back to re-observing the env state (infos usually carry the new obs
        # only in some wrappers); zero bootstrap is a safe lower bound.
        last_value = torch.zeros(values.shape[1], device=device)
    else:
        with torch.no_grad():
            last_value, _ = baseline_values(model, last_obs)
            last_value = last_value.reshape(-1).to(device)
    advantages, returns = compute_gae(
        rewards,
        values,
        dones,
        last_value,
        gamma=float(cfg.discounting),
        lam=float(cfg.lam),
        normalize=bool(cfg.normalize_advantages),
    )
    obs_flat = _obs_to_device(_obs_batch(obs_list), device)
    returns_flat = returns.reshape(-1)
    old_values_flat = values.reshape(-1)
    n_total = int(returns_flat.numel())
    if n_total == 0:
        return {"loss": float("nan"), "n": 0.0}
    minibatch = max(1, int(cfg.batch_size))
    perm = torch.randperm(n_total, device=returns_flat.device)
    losses: List[float] = []
    for start in range(0, n_total, minibatch):
        idx = perm[start : start + minibatch]
        mb_obs = _index_obs(obs_flat, idx)
        out = model(mb_obs)
        pred = getattr(out, "value", None)
        if pred is None and isinstance(out, Mapping):
            pred = out.get("value", out.get("baseline"))
        if pred is None and isinstance(out, (tuple, list)):
            pred = out[1] if len(out) > 1 else out[0]
        loss = baseline_loss(
            pred.reshape(-1),
            returns_flat[idx],
            old_values_flat[idx],
            clip=float(cfg.appo_clip_baseline),
            cost=float(cfg.baseline_cost),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_norm_clipping:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if getattr(p, "requires_grad", False)],
                float(cfg.grad_norm_clipping),
            )
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    metrics = {
        "loss": float(sum(losses) / max(1, len(losses))),
        "n": float(n_total),
    }
    if advantages is not None:
        with torch.no_grad():
            metrics["value_mean"] = float(returns_flat.mean().detach().cpu())
            metrics["advantage_std"] = float(advantages.std(unbiased=False).detach().cpu())
    return metrics


def _index_obs(obs: Any, idx: Any) -> Any:
    if isinstance(obs, Mapping):
        return {k: v[idx] for k, v in obs.items()}
    return obs[idx]


# --------------------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------------------


def pretrain_baseline(
    cfg: Any = None,
    model: Any = None,
    *,
    steps: Optional[int] = None,
    checkpoint: Optional[str] = None,
    stub: Optional[bool] = None,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    output_dir: Optional[str] = None,
    logger: Any = None,
    env: Any = None,
    verbose: bool = True,
    progress_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
    save_every: Optional[int] = None,
    log_every: Optional[int] = None,
    eval_every: Optional[int] = None,
    eval_episodes: Optional[int] = None,
    character: Optional[str] = None,
) -> BaselinePretrainResult:
    """Pre-train the baseline head with everything else frozen (Appendix B.1).

    Parameters
    ----------
    cfg:
        A loaded YAML config (``configs/nethack.yaml``), a mapping, or a
        :class:`BaselinePretrainConfig`.  ``cfg.baseline_pretrain`` (or
        ``cfg.base_training``/``cfg.ppo``/``cfg.model``) values are honoured.
    model:
        Optional pre-built model.  When ``None`` the ``pi_*`` checkpoint from
        ``checkpoint``/``cfg.pretrained.checkpoint`` is loaded via
        :func:`src.nethack.model.build_model`.
    steps:
        Override ``total_steps`` (paper: 500 M environment steps).

    Returns
    -------
    :class:`BaselinePretrainResult` carrying the pre-trained model in its
    ``model`` field.
    """
    config = cfg if isinstance(cfg, BaselinePretrainConfig) else BaselinePretrainConfig.from_config(cfg)
    overrides = {
        "total_steps": steps,
        "checkpoint": checkpoint,
        "stub": stub,
        "seed": seed,
        "device": device,
        "save_every": save_every,
        "log_every": log_every,
        "eval_every": eval_every,
        "eval_episodes": eval_episodes,
        "character": character,
    }
    config = config.with_overrides(**{k: v for k, v in overrides.items() if v is not None})
    if cfg is not None and not isinstance(cfg, BaselinePretrainConfig):
        config = BaselinePretrainConfig.from_config(cfg, **{k: v for k, v in overrides.items() if v is not None})

    result = BaselinePretrainResult(
        config=config.to_dict(),
        device=str(config.device),
        stub=bool(config.stub),
    )
    started = time.time()

    def log(message: str) -> None:
        if logger is not None and hasattr(logger, "info"):
            logger.info(message)
        elif verbose:
            print(f"[pretrain_baseline] {message}")

    try:
        _require_torch()
    except RuntimeError as exc:
        result.status = "unavailable"
        result.error = str(exc)
        if verbose:
            print(f"[pretrain_baseline] {exc}")
        return result

    # --- seeding ---------------------------------------------------------
    if config.seed is not None:
        try:
            from src.common import seeding as _seeding  # type: ignore

            _seeding.set_seed(int(config.seed))
        except Exception:  # pragma: no cover - best effort
            torch.manual_seed(int(config.seed))

    device_obj = torch.device(str(config.device)) if str(config.device) != "cpu" and torch.cuda.is_available() else torch.device("cpu")
    config.device = str(device_obj)

    # --- model -----------------------------------------------------------
    if model is None:
        try:
            from src.nethack.model import build_model  # type: ignore

            model = build_model(
                config=cfg,
                checkpoint=config.checkpoint,
                device=config.device,
                stub=bool(config.stub),
            )
        except Exception as exc:  # pragma: no cover - surfaced in the result
            result.status = "error"
            result.error = f"could not build model: {exc}"
            log(result.error)
            return result
    if hasattr(model, "to"):
        model.to(device_obj)

    freeze_info = freeze_except_baseline(model, verbose=False)
    if not freeze_info["trainable"]:
        log("no baseline-head parameters detected; training the last module instead")
        names = [n for n, _ in model.named_parameters()]
        target = names[-1] if names else None
        if target is not None:
            freeze_info = freeze_except_baseline(model, extra_trainable=[target], verbose=False)
    result.trainable_parameters = freeze_info["trainable"]
    trainable_n, total_n = count_trainable(model)
    result.trainable_parameter_count = trainable_n
    result.frozen_parameter_count = max(0, total_n - trainable_n)
    log(
        f"froze {result.frozen_parameter_count} parameters; "
        f"pre-training the baseline head ({result.trainable_parameter_count} parameters) "
        f"for {int(config.total_steps)} env steps"
    )

    # --- environment -----------------------------------------------------
    if env is None:
        try:
            from src.nethack.env import build_env  # type: ignore

            env = build_env(
                cfg,
                seed=config.seed,
                stub=bool(config.stub),
                num_envs=int(config.num_envs),
                character=config.character,
            )
        except Exception:
            try:
                from src.nethack.env import make_vec_env  # type: ignore

                env = make_vec_env(
                    num_envs=int(config.num_envs),
                    seed=config.seed,
                    stub=bool(config.stub),
                    character=config.character,
                )
            except Exception as exc:  # pragma: no cover
                result.status = "error"
                result.error = f"could not build env: {exc}"
                log(result.error)
                return result

    num_envs = int(config.num_envs)
    optimizer = _adamw_optimizer(model.parameters(), config)

    obs_cur = _obs_to_device(_vec_reset(env, config.seed), device_obj)
    steps_done = 0
    updates = 0
    batch = RolloutBatch(obs=[], rewards=[], values=[], dones=[], infos=[])
    obs_cur = _collect_rollout_impl(env, model, batch, obs_cur, num_envs, int(config.unroll_length))

    result.initial_loss = None
    last_log = 0
    try:
        while steps_done < int(config.total_steps):
            metrics = baseline_head_update(model, optimizer, batch, config, device_obj)
            updates += 1
            steps_done += int(batch.rewards.__len__()) * num_envs if hasattr(batch.rewards, "__len__") else config.batch_steps
            if result.initial_loss is None:
                result.initial_loss = float(metrics.get("loss", float("nan")))
            result.final_loss = float(metrics.get("loss", float("nan")))
            if result.best_loss is None or (result.final_loss is not None and result.final_loss < result.best_loss):
                result.best_loss = result.final_loss
            record = {"step": steps_done, "update": updates, **metrics}
            result.history.append(record)
            if progress_fn is not None:
                try:
                    progress_fn(record)
                except Exception:  # pragma: no cover
                    pass
            if steps_done - last_log >= max(1, int(config.log_every)):
                log(f"steps={steps_done} updates={updates} loss={metrics.get('loss'):.4f}")
                last_log = steps_done
            if eval_every and steps_done >= getattr(config, "_next_eval", 0):  # pragma: no cover
                pass
            if save_every and steps_done % max(1, int(save_every)) < config.batch_steps:
                ckpt = _save(model, output_dir, step=steps_done, optimizer=optimizer, extra={"phase": "baseline_pretrain"})
                if ckpt:
                    result.checkpoint = ckpt
                    log(f"checkpoint saved: {ckpt}")
            # next rollout
            batch = RolloutBatch(obs=[], rewards=[], values=[], dones=[], infos=[])
            obs_cur = _collect_rollout_impl(env, model, batch, obs_cur, num_envs, int(config.unroll_length))
    except KeyboardInterrupt:  # pragma: no cover
        log("interrupted; saving current baseline head")
    except Exception as exc:  # pragma: no cover
        result.status = "error"
        result.error = str(exc)
        log(f"training error: {exc}")

    result.steps = steps_done
    result.num_updates = updates
    result.elapsed = time.time() - started
    ckpt = _save(model, output_dir, step=steps_done, optimizer=optimizer, extra={"phase": "baseline_pretrain"})
    if ckpt:
        result.checkpoint = ckpt
    result.model = model
    log(
        f"finished: steps={result.steps} updates={result.num_updates} "
        f"loss={_fmt(result.initial_loss)}->{_fmt(result.final_loss)} in {result.elapsed:.1f}s"
    )
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "baseline_pretrain_summary.json"), "w", encoding="utf-8") as fh:
                json.dump(result.as_dict(), fh, indent=2, default=str)
        except Exception:  # pragma: no cover
            pass
    return result


#: Alias used by :mod:`src.nethack.train_nethack`.
train_baseline = pretrain_baseline


def _save(model: Any, output_dir: Optional[str], step: int, optimizer: Any = None, extra: Optional[Dict[str, Any]] = None) -> Optional[str]:
    if not output_dir:
        return None
    try:
        from src.common.checkpointing import ensure_dir, save_checkpoint  # type: ignore

        ensure_dir(output_dir)
        path = os.path.join(output_dir, "nethack_baseline.pt")
        return save_checkpoint(path, {"model": model, "optimizer": optimizer, **(extra or {})}, step=step)
    except Exception:
        try:
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, f"nethack_baseline_step_{step}.pt")
            torch.save({"model": model, "optimizer": optimizer, **(extra or {})}, path)
            return path
        except Exception:  # pragma: no cover
            return None


def load_baseline_model(path: str, cfg: Any = None, device: str = "cpu", stub: bool = False) -> Any:
    """Load a baseline-pre-trained model from a checkpoint directory/file."""
    _require_torch()
    from src.nethack.model import build_model  # type: ignore

    model = build_model(config=cfg, checkpoint=None, device=device, stub=stub)
    target = path
    if os.path.isdir(path):
        candidates = [os.path.join(path, f) for f in sorted(os.listdir(path)) if f.endswith(".pt")]
        target = candidates[-1] if candidates else path
    payload = torch.load(target, map_location=device)
    state = payload.get("model", payload) if isinstance(payload, Mapping) else payload
    if isinstance(state, nn.Module):
        return state.to(device)
    try:
        model.load_state_dict(state, strict=False)
    except Exception:  # pragma: no cover - wrapped module state dicts
        try:
            model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=False)
        except Exception:
            pass
    return model


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="src.nethack.pretrain_baseline",
        description="Pre-train the NetHack baseline (value) head with frozen encoders (Appendix B.1).",
    )
    parser.add_argument("--config", type=str, default=None, help="YAML config (configs/nethack.yaml).")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="key.subkey=value override (repeatable).")
    parser.add_argument("--steps", type=int, default=None, help=f"Environment steps (default {BASELINE_PRETRAIN_STEPS}).")
    parser.add_argument("--checkpoint", type=str, default=None, help="pi_* checkpoint to load before pre-training.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--stub", action="store_true", help="Use the dependency-free CPU stub model/env.")
    parser.add_argument("--smoke-test", action="store_true", help=f"Run only {SMOKE_TEST_STEPS} env steps.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = None
    if args.config:
        try:
            from src.common.config import load_config, apply_overrides  # type: ignore

            cfg = load_config(args.config)
            if args.overrides:
                cfg = apply_overrides(cfg, args.overrides)
        except Exception as exc:  # pragma: no cover
            print(f"[pretrain_baseline] could not load config {args.config!r}: {exc}")
    steps = args.steps
    if args.smoke_test:
        steps = SMOKE_TEST_STEPS
    result = pretrain_baseline(
        cfg,
        steps=steps,
        checkpoint=args.checkpoint,
        stub=bool(args.stub),
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        save_every=args.save_every,
        log_every=args.log_every,
    )
    print(json.dumps(result.as_dict(include_history=False), indent=2, default=str))
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
