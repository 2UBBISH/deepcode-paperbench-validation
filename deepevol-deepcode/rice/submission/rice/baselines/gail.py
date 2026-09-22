"""Generative Adversarial Imitation Learning (GAIL) baseline for RICE (Experiment IV).

Paper context
-------------
Section 4.2 (Experiment IV) — *versatility of our method*::

    "First, we obtain a pre-trained SAC agent and then use Generative Adversarial
     Imitation Learning (GAIL) (Ho & Ermon, 2016) to learn an approximated policy
     network. We compare the refining performance using our method against baseline
     methods, i.e., PPO fine-tuning, StateMask's fine-tuning from critical steps, and
     Jump-Start Reinforcement Learning (Uchendu et al., 2023). In addition, we also
     include fine-tuning the pre-trained SAC agent with the SAC algorithm as a baseline."

Section 4.1 (Baseline Refining Methods) and Appendix C.1 note that baselines use the
authors' released code when available; no GAIL code is released, therefore this module
implements an own-version GAIL:

1. **Collect expert state-action pairs** ``(s, a) ~ d^{π_E}`` by rolling the frozen
   pre-trained (SAC) expert policy ``π_E`` in the environment
   (:func:`collect_expert_demonstrations`).
2. **Behaviour-cloning warm start** (pragmatic, unsupervised-by-the-paper default that
   makes short GAIL runs stable; disabled with ``bc_pretrain_epochs=0``).
3. **Adversarial (GAIL) training** — a discriminator ``D(s, a)`` is trained with BCE to
   separate expert pairs (label 1) from policy pairs (label 0), while the policy
   ``π_θ`` is optimised with the standard PPO clipped objective on the GAIL reward

       r(s, a) = -log(1 - D(s, a))            (Ho & Ermon, 2016)

   which, for discriminator logits ``x`` and ``D = σ(x)``, equals ``softplus(x)``.
   Optional task-reward mixing is exposed via ``reward_mode`` /
   ``task_reward_coef`` (default is pure imitation, as in the paper).

The resulting *approximated policy network* ``π_G`` can then be refined by RICE
(Algorithm 2) — :meth:`GAILRefiner.refine` / :meth:`GAILRefiner.run` delegate to
``rice.refining.ppo_refine.refine_policy`` so that the "SAC → GAIL-approximation →
RICE refinement" pipeline of Experiment IV is available end-to-end.  As required by
Section 4.2, when GAIL is used *as a refining baseline*, it receives exactly the same
explanation (mask network) as RICE.

Robustness
----------
Every project import is guarded so the module stays importable without torch/SB3 and
without the rest of ``rice``; local fallbacks mirror the real helper surfaces.  All
unspecified numerics default to Stable-Baselines3 PPO defaults (γ=0.99, GAE λ=0.95,
clip 0.2, lr 3e-4, 10 epochs, batch 64).
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    # config / containers
    "GAILConfig",
    "GAILRollout",
    # networks / trainers
    "GAILDiscriminator",
    "GAILTrainer",
    "GAILRefiner",
    # factories
    "make_gail",
    "build_gail",
    "gail",
    "gail_baseline",
    "train_gail",
    "describe_gail",
    # functional helpers
    "collect_expert_demonstrations",
    "gail_reward",
    "gail_rewards",
    "behavior_clone",
    "approximate_policy_with_gail",
    # constants
    "DEFAULT_LR",
    "DEFAULT_DISC_LR",
    "DEFAULT_HIDDEN_SIZES",
    "DEFAULT_DISC_HIDDEN_SIZES",
    "DEFAULT_GAMMA",
    "DEFAULT_GAE_LAMBDA",
    "DEFAULT_CLIP_RANGE",
    "DEFAULT_N_EPOCHS",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_N_STEPS",
    "DEFAULT_TOTAL_TIMESTEPS",
    "DEFAULT_EXPERT_TIMESTEPS",
    "REWARD_MODES",
]


# --------------------------------------------------------------------------------------
# optional dependencies
# --------------------------------------------------------------------------------------
try:  # pragma: no cover - torch is expected in real runs
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    Adam = None  # type: ignore
    _HAS_TORCH = False


# ---- project utilities -----------------------------------------------------------------
try:
    from ..utils.seeding import get_rng, set_seed  # type: ignore
except Exception:  # pragma: no cover
    def get_rng(seed: Optional[int] = None) -> np.random.RandomState:  # type: ignore
        return np.random.RandomState(seed)

    def set_seed(seed: int, deterministic: bool = False) -> int:  # type: ignore
        np.random.seed(int(seed))
        if _HAS_TORCH:
            torch.manual_seed(int(seed))
        return int(seed)


try:
    from ..utils.logging import get_logger  # type: ignore
except Exception:  # pragma: no cover
    def get_logger(name: str = "rice", out_dir: Optional[str] = None, level: int = logging.INFO):  # type: ignore
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(asctime)s] %(name)s: %(message)s"))
            logger.addHandler(handler)
        logger.setLevel(level)
        if out_dir:
            try:
                os.makedirs(out_dir, exist_ok=True)
                fh = logging.FileHandler(os.path.join(out_dir, "log.txt"))
                fh.setFormatter(logging.Formatter("[%(asctime)s] %(name)s: %(message)s"))
                logger.addHandler(fh)
            except Exception:
                pass
        return logger


try:
    from ..utils.io import ensure_dir  # type: ignore
except Exception:  # pragma: no cover
    def ensure_dir(path: str) -> str:  # type: ignore
        if path:
            os.makedirs(path, exist_ok=True)
        return path


try:
    from ..models.policies import (  # type: ignore
        build_policy,
        load_policy,
        normalize_env_key,
        sample_random_action,
        save_policy,
    )

    _HAS_POLICIES = True
except Exception:  # pragma: no cover
    build_policy = None  # type: ignore
    load_policy = None  # type: ignore
    save_policy = None  # type: ignore
    sample_random_action = None  # type: ignore

    def normalize_env_key(env_id: Any) -> str:  # type: ignore
        text = str(env_id or "default").strip().lower().replace("-", "_")
        text = text.split(".")[0].split("/")[0]
        for prefix in ("sparse_",):
            if text.startswith(prefix):
                text = text[len(prefix):]
        parts = [p for p in text.split("_") if p and not (p.startswith("v") and p[1:].isdigit())]
        return "_".join(parts) or "default"

    _HAS_POLICIES = False


try:  # RICE Stage-2 refinement (only needed for ``GAILRefiner.refine``)
    from ..refining.ppo_refine import (  # type: ignore
        PPORefiner,
        RefinePPOConfig,
        evaluate_refined_policy,
        refine_policy,
        unpack_reset,
        unpack_step,
    )

    _HAS_REFINER = True
except Exception:  # pragma: no cover
    PPORefiner = None  # type: ignore
    RefinePPOConfig = None  # type: ignore
    evaluate_refined_policy = None  # type: ignore
    refine_policy = None  # type: ignore
    unpack_reset = None  # type: ignore
    unpack_step = None  # type: ignore
    _HAS_REFINER = False


_LOGGER = get_logger("rice.baselines.gail")


# --------------------------------------------------------------------------------------
# defaults (SB3 PPO defaults where the paper is silent)
# --------------------------------------------------------------------------------------
DEFAULT_LR = 3e-4
DEFAULT_DISC_LR = 3e-4
DEFAULT_HIDDEN_SIZES: Tuple[int, ...] = (64, 64)
DEFAULT_DISC_HIDDEN_SIZES: Tuple[int, ...] = (100, 100)
DEFAULT_GAMMA = 0.99
DEFAULT_GAE_LAMBDA = 0.95
DEFAULT_CLIP_RANGE = 0.2
DEFAULT_N_EPOCHS = 10
DEFAULT_BATCH_SIZE = 64
DEFAULT_VF_COEF = 0.5
DEFAULT_ENT_COEF = 0.0
DEFAULT_MAX_GRAD_NORM = 0.5
DEFAULT_N_STEPS = 2048
DEFAULT_TOTAL_TIMESTEPS = 200_000
DEFAULT_EXPERT_TIMESTEPS = 100_000
DEFAULT_BC_EPOCHS = 10
DEFAULT_BC_BATCH_SIZE = 256
DEFAULT_DISC_EPOCHS = 1
DEFAULT_DISC_BATCH_SIZE = 256
DEFAULT_HORIZON = 1000
DEFAULT_REWARD_MODE = "gail"
REWARD_MODES: Tuple[str, ...] = ("gail", "gail_task", "task", "bc")


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------
def _as_rng(rng: Any = None, seed: Optional[int] = None) -> np.random.RandomState:
    """Return a NumPy RandomState (stable across NumPy versions)."""
    if isinstance(rng, np.random.RandomState):
        return rng
    if isinstance(rng, np.random.Generator):
        return np.random.RandomState(int(rng.integers(0, 2 ** 31 - 1)))
    if rng is not None:
        try:
            return np.random.RandomState(int(rng))
        except Exception:
            pass
    return get_rng(seed)


def _flat(observation: Any) -> np.ndarray:
    """Flatten an observation (array / tuple / dict / scalar) to float32."""
    if isinstance(observation, dict):
        parts = [np.asarray(observation[k], dtype=np.float32).ravel() for k in sorted(observation)]
        if not parts:
            return np.zeros(1, dtype=np.float32)
        return np.concatenate(parts).astype(np.float32)
    if torch is not None and isinstance(observation, torch.Tensor):
        return observation.detach().cpu().numpy().astype(np.float32).ravel()
    arr = np.asarray(observation, dtype=np.float32)
    return arr.ravel()


def _to_tensor(x: Any, device: str = "cpu", dtype: Any = None):
    """Convert numpy/list/tensor to a torch tensor on ``device``."""
    if not _HAS_TORCH:
        raise ImportError("GAIL requires PyTorch for training.")
    if torch.is_tensor(x):
        return x.to(device=device)
    arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(np.float32 if dtype is torch.float32 else arr.dtype)
    tensor = torch.as_tensor(np.ascontiguousarray(arr), device=device)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _to_numpy(x: Any) -> np.ndarray:
    if torch is not None and torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _unpack_step(result: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
    """Normalise gym/gymnasium step returns to 5 items."""
    if unpack_step is not None:  # reuse the refining implementation when available
        try:
            out = unpack_step(result)
            if isinstance(out, tuple) and len(out) == 5:
                return out  # type: ignore[return-value]
        except Exception:
            pass
    if isinstance(result, tuple):
        if len(result) == 5:
            obs, rew, term, trunc, info = result
            return obs, float(rew), bool(term), bool(trunc), dict(info or {})
        if len(result) == 4:
            obs, rew, done, info = result
            return obs, float(rew), bool(done), False, dict(info or {})
    raise ValueError(f"Unsupported env.step return: {type(result)}")


def _unpack_reset(result: Any) -> Tuple[Any, Dict[str, Any]]:
    if unpack_reset is not None:
        try:
            out = unpack_reset(result)
            if isinstance(out, tuple) and len(out) == 2:
                return out  # type: ignore[return-value]
        except Exception:
            pass
    if isinstance(result, tuple) and len(result) == 2:
        obs, info = result
        return obs, dict(info or {})
    return result, {}


def _obs_scalar_space(space: Any) -> Optional[int]:
    if space is None:
        return None
    shape = getattr(space, "shape", None)
    if shape is None:
        return None
    total = 1
    for s in shape:
        total *= int(s)
    return int(total)


def _action_dim_of(env: Any, policy: Any = None) -> Tuple[int, bool]:
    """Return ``(action_dim, discrete)`` for the env / policy."""
    space = getattr(env, "action_space", None)
    if space is not None:
        if hasattr(space, "n") and not hasattr(space, "shape"):
            return int(space.n), True
        if hasattr(space, "shape"):
            dim = 1
            for s in space.shape:
                dim *= int(s)
            return int(dim), False
    if policy is not None:
        act_dim = getattr(policy, "action_dim", None)
        if act_dim is not None:
            discrete = bool(getattr(policy, "discrete", False))
            return int(act_dim), discrete
    return 1, False


def _obs_dim_of(env: Any, policy: Any = None) -> int:
    obs_space = getattr(env, "observation_space", None)
    dim = _obs_scalar_space(obs_space)
    if dim:
        return int(dim)
    if policy is not None:
        for attr in ("obs_dim", "observation_dim", "input_dim"):
            value = getattr(policy, attr, None)
            if value:
                return int(value)
    if policy is not None:
        try:
            params = list(policy.parameters())
            for p in params:
                if p.dim() == 2:
                    return int(p.shape[1])
        except Exception:
            pass
    return 1


def _policy_device(policy: Any, fallback: str = "cpu") -> str:
    try:
        for p in policy.parameters():
            return str(p.device)
    except Exception:
        pass
    return fallback


# --------------------------------------------------------------------------------------
# policy / distribution helpers (duck-typed: native ActorCritic or SB3 policy)
# --------------------------------------------------------------------------------------
def get_policy_distribution(policy: Any, obs_tensor: Any):
    """Return the policy's torch distribution for a batch of observations."""
    for attr in ("get_distribution", "distribution"):
        fn = getattr(policy, attr, None)
        if callable(fn):
            try:
                return fn(obs_tensor)
            except Exception:
                continue
    if hasattr(policy, "action_dist"):
        try:
            mean, log_std = policy.forward(obs_tensor)[:2]
            return policy.action_dist.proba_distribution(mean, log_std)
        except Exception:
            pass
    raise AttributeError("Policy exposes no distribution accessor (need get_distribution/distribution).")


def predict_values_tensor(policy: Any, obs_tensor: Any):
    """Return the policy's value estimate as a 1-D tensor (zeros when unavailable)."""
    for attr in ("predict_values", "value"):
        fn = getattr(policy, attr, None)
        if callable(fn):
            try:
                values = fn(obs_tensor)
                return values.reshape(-1, 1) if torch.is_tensor(values) else values
            except Exception:
                continue
    return torch.zeros((obs_tensor.shape[0], 1), device=obs_tensor.device)


def evaluate_policy_actions(
    policy: Any,
    observations: Any,
    actions: Any,
    discrete: bool = False,
    device: str = "cpu",
) -> Tuple[Any, Any, Any]:
    """Compute ``(log_prob, values, entropy)`` tensors for a batch of ``(s, a)``.

    Prefers the explicit ``get_distribution`` + ``predict_values`` path (no tuple-order
    ambiguity between native and SB3 policies), and falls back to
    ``evaluate_actions`` when a distribution accessor is unavailable.
    """
    obs_t = _to_tensor(observations, device, dtype=torch.float32)
    if discrete:
        act_t = _to_tensor(np.asarray(actions).astype(np.int64), device)
    else:
        act_t = _to_tensor(actions, device, dtype=torch.float32)

    try:
        dist = get_policy_distribution(policy, obs_t)
        log_prob = dist.log_prob(act_t) if not discrete else dist.log_prob(act_t.squeeze(-1) if act_t.dim() > 1 else act_t)
        values = predict_values_tensor(policy, obs_t)
        try:
            entropy = dist.entropy()
        except Exception:
            entropy = torch.zeros_like(log_prob)
        lp = log_prob.reshape(-1, 1) if log_prob.dim() == 1 else log_prob
        ent = entropy.reshape(-1, 1) if entropy.dim() == 1 else entropy
        val = values.reshape(-1, 1) if values.dim() == 1 else values
        return lp.reshape(-1), val.reshape(-1), ent.reshape(-1)
    except Exception:
        pass

    if hasattr(policy, "evaluate_actions"):
        out = policy.evaluate_actions(obs_t, act_t)
        if isinstance(out, tuple) and len(out) >= 3:
            first, second, third = out[0], out[1], out[2]
            # SB3 convention: (values, log_prob, entropy)
            values, log_prob, entropy = first, second, third
            try:
                if float(torch.as_tensor(values).abs().mean()) < 1e-8 and float(
                    torch.as_tensor(log_prob).abs().mean()
                ) > 1e-6:
                    pass
            except Exception:
                pass
            return (
                torch.as_tensor(log_prob).reshape(-1),
                torch.as_tensor(values).reshape(-1),
                torch.as_tensor(entropy).reshape(-1),
            )
    raise RuntimeError("Unable to evaluate actions for the given policy.")


def sample_policy_action(
    policy: Any,
    observation: Any,
    deterministic: bool = False,
    discrete: bool = False,
    action_space: Any = None,
    device: str = "cpu",
) -> np.ndarray:
    """Sample an action from the policy distribution (consistent with log-probs).

    ``policy(·|s)`` is sampled directly from the policy's torch distribution so that the
    action fed to the environment is exactly the action whose ``log_prob`` is later
    re-evaluated during the PPO update (no squash/clip mismatch).
    """
    if not _HAS_TORCH:
        return np.zeros(1, dtype=np.float32)
    obs_t = _to_tensor(np.asarray(_flat(observation), dtype=np.float32), device, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        try:
            dist = get_policy_distribution(policy, obs_t)
        except Exception:
            dist = None
        if dist is not None:
            if deterministic:
                if hasattr(dist, "mean") and dist.mean is not None:
                    action = dist.mean
                elif hasattr(dist, "logits"):
                    action = torch.argmax(dist.logits, dim=-1)
                else:
                    action = dist.sample()
            else:
                action = dist.sample()
        else:
            # last-resort duck-typed API
            if hasattr(policy, "predict"):
                out = policy.predict(np.asarray(_flat(observation), dtype=np.float32), deterministic=deterministic)
                action = out[0] if isinstance(out, tuple) else out
            elif hasattr(policy, "act"):
                out = policy.act(np.asarray(_flat(observation), dtype=np.float32), deterministic=deterministic)
                action = out[0] if isinstance(out, tuple) else out
            else:
                raise RuntimeError("Policy exposes neither get_distribution nor predict/act.")
    arr = _to_numpy(action)
    if discrete:
        return np.asarray(arr).reshape(-1)[:1].astype(np.int64)
    return np.asarray(arr, dtype=np.float32).ravel()


def prepare_action_for_env(action: Any, action_space: Any, discrete: bool = False) -> Any:
    """Clip / cast an action so the environment accepts it."""
    if discrete:
        arr = np.asarray(action).ravel()
        index = int(arr[0]) if arr.size else 0
        if hasattr(action_space, "n"):
            index = int(np.clip(index, 0, int(action_space.n) - 1))
        return index
    arr = np.asarray(action, dtype=np.float32).ravel()
    if action_space is not None and hasattr(action_space, "low") and hasattr(action_space, "high"):
        low = np.asarray(action_space.low, dtype=np.float32).ravel()
        high = np.asarray(action_space.high, dtype=np.float32).ravel()
        if low.shape == arr.shape:
            arr = np.clip(arr, low, high)
    return arr.astype(np.float32)


def action_features(actions: Any, discrete: bool = False, action_dim: Optional[int] = None) -> np.ndarray:
    """Feature representation of actions for the discriminator input."""
    arr = np.asarray(actions)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if discrete:
        if action_dim is None:
            action_dim = int(np.max(arr)) + 1 if arr.size else 1
        out = np.zeros((arr.shape[0], int(action_dim)), dtype=np.float32)
        idx = np.clip(arr.reshape(-1).astype(np.int64), 0, int(action_dim) - 1)
        out[np.arange(arr.shape[0]), idx] = 1.0
        return out
    return arr.astype(np.float32)


# --------------------------------------------------------------------------------------
# GAIL reward
# --------------------------------------------------------------------------------------
def gail_reward(discriminator_logits: Any) -> np.ndarray:
    """GAIL imitation reward ``-log(1 - D(s,a))`` for discriminator logits.

    Since ``D = σ(x)`` and ``1 - σ(x) = σ(-x)``::

        -log(1 - σ(x)) = -log σ(-x) = softplus(x)

    which is computed in a numerically stable way (``F.softplus``).
    """
    if _HAS_TORCH and torch.is_tensor(discriminator_logits):
        return F.softplus(discriminator_logits).detach().cpu().numpy().reshape(-1)
    x = np.asarray(discriminator_logits, dtype=np.float64).reshape(-1)
    return np.logaddexp(0.0, x)  # numerically stable softplus


def gail_rewards(discriminator: Any, observations: Any, actions: Any, device: str = "cpu") -> np.ndarray:
    """Convenience wrapper: discriminator logits -> GAIL imitation rewards."""
    if not _HAS_TORCH:
        raise ImportError("GAIL reward computation requires PyTorch.")
    with torch.no_grad():
        logits = discriminator(_to_tensor(observations, device, dtype=torch.float32),
                               _to_tensor(actions, device, dtype=torch.float32))
    return gail_reward(logits)


def compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    dones: Sequence[float],
    last_value: float = 0.0,
    gamma: float = DEFAULT_GAMMA,
    gae_lambda: float = DEFAULT_GAE_LAMBDA,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation (identical to the RICE refiner)."""
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    dones = np.asarray(dones, dtype=np.float32).reshape(-1)
    n = rewards.shape[0]
    advantages = np.zeros(n, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(n)):
        if t == n - 1:
            next_value = float(last_value)
            next_non_terminal = 1.0 - float(dones[t])
        else:
            next_value = float(values[t + 1])
            next_non_terminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


# --------------------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------------------
@dataclass
class GAILConfig:
    """Hyper-parameters of the GAIL policy-approximation baseline.

    Unspecified values follow Stable-Baselines3 PPO defaults; GAIL-specific defaults
    (discriminator hidden sizes 100x100) follow the original GAIL paper.
    """

    env_id: str = "default"
    # --- policy / imitation objective
    hidden_sizes: Tuple[int, ...] = DEFAULT_HIDDEN_SIZES
    activation: str = "tanh"
    lr: float = DEFAULT_LR
    lr_policy: Optional[float] = None
    gamma: float = DEFAULT_GAMMA
    gae_lambda: float = DEFAULT_GAE_LAMBDA
    clip_range: float = DEFAULT_CLIP_RANGE
    n_epochs: int = DEFAULT_N_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    vf_coef: float = DEFAULT_VF_COEF
    ent_coef: float = DEFAULT_ENT_COEF
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM
    target_kl: Optional[float] = None
    normalize_advantage: bool = True
    # --- discriminator
    use_discriminator: bool = True
    disc_hidden_sizes: Tuple[int, ...] = DEFAULT_DISC_HIDDEN_SIZES
    disc_activation: str = "tanh"
    lr_disc: float = DEFAULT_DISC_LR
    disc_epochs: int = DEFAULT_DISC_EPOCHS
    disc_batch_size: int = DEFAULT_DISC_BATCH_SIZE
    disc_updates: int = 1
    disc_l2: float = 0.0
    disc_label_smoothing: float = 0.0
    # --- reward shaping
    reward_mode: str = DEFAULT_REWARD_MODE
    task_reward_coef: float = 0.0
    normalize_reward: bool = True
    reward_clip: Optional[float] = 10.0
    # --- data / budget
    n_steps: Optional[int] = None
    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS
    expert_timesteps: int = DEFAULT_EXPERT_TIMESTEPS
    expert_deterministic: bool = True
    bc_pretrain_epochs: int = DEFAULT_BC_EPOCHS
    bc_batch_size: int = DEFAULT_BC_BATCH_SIZE
    bc_lr: Optional[float] = None
    # --- misc
    device: str = "cpu"
    seed: Optional[int] = None
    log_interval: int = 1
    eval_episodes: int = 10
    deterministic_eval: bool = True

    def __post_init__(self) -> None:
        self.env_id = normalize_env_key(self.env_id)
        self.hidden_sizes = tuple(int(h) for h in np.asarray(self.hidden_sizes).ravel().tolist())
        self.disc_hidden_sizes = tuple(int(h) for h in np.asarray(self.disc_hidden_sizes).ravel().tolist())
        mode = str(self.reward_mode or DEFAULT_REWARD_MODE).strip().lower()
        self.reward_mode = mode if mode in REWARD_MODES else DEFAULT_REWARD_MODE
        if self.lr_policy is not None:
            self.lr = float(self.lr_policy)
        if self.bc_lr is None:
            self.bc_lr = float(self.lr)

    # -- convenience ------------------------------------------------------------------
    @property
    def learning_rate(self) -> float:
        return float(self.lr)

    @property
    def timesteps(self) -> int:
        return int(self.total_timesteps)

    @property
    def expert_samples(self) -> int:
        return int(self.expert_timesteps)

    @classmethod
    def from_dict(cls, cfg: Any = None, **overrides: Any) -> "GAILConfig":
        """Build a config from a (possibly nested) YAML dict plus overrides."""
        data: Dict[str, Any] = {}
        if isinstance(cfg, GAILConfig):
            data = cfg.to_dict()
        elif isinstance(cfg, dict):
            merged: Dict[str, Any] = {}
            for section in ("default", "gail", "GAIL", "imitation", "imitation_learning", "refine", "baseline"):
                sub = cfg.get(section)
                if isinstance(sub, dict):
                    merged.update(sub)
            merged.update({k: v for k, v in cfg.items() if not isinstance(v, dict)})
            data = merged
        aliases = {
            "env": "env_id",
            "hidden": "hidden_sizes",
            "net_arch": "hidden_sizes",
            "policy_hidden": "hidden_sizes",
            "hidden_size": "hidden_sizes",
            "learning_rate": "lr",
            "policy_lr": "lr",
            "lr_p": "lr",
            "lam": "gae_lambda",
            "lambda_": "gae_lambda",
            "clip": "clip_range",
            "epochs": "n_epochs",
            "n_epoch": "n_epochs",
            "bs": "batch_size",
            "minibatch_size": "batch_size",
            "grad_norm": "max_grad_norm",
            "entropy": "ent_coef",
            "vf": "vf_coef",
            "disc_lr": "lr_disc",
            "discriminator_lr": "lr_disc",
            "disc_hidden": "disc_hidden_sizes",
            "discriminator_hidden": "disc_hidden_sizes",
            "use_disc": "use_discriminator",
            "timesteps": "total_timesteps",
            "total_steps": "total_timesteps",
            "expert_samples": "expert_timesteps",
            "expert_steps": "expert_timesteps",
            "rollout_length": "n_steps",
            "horizon": "n_steps",
            "bc_epochs": "bc_pretrain_epochs",
            "pretrain_epochs": "bc_pretrain_epochs",
            "bc_bs": "bc_batch_size",
            "seed_": "seed",
        }
        cleaned: Dict[str, Any] = {}
        known = set(cls.__dataclass_fields__.keys())
        for key, value in data.items():
            name = aliases.get(str(key), str(key))
            if name in known:
                cleaned[name] = value
        cleaned.update({k: v for k, v in overrides.items() if v is not None and k in known})
        return cls(**cleaned)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["hidden_sizes"] = list(self.hidden_sizes)
        data["disc_hidden_sizes"] = list(self.disc_hidden_sizes)
        data["beta"] = None  # GAIL has no mixed-init weight (kept for table-parity tooling)
        return data


# --------------------------------------------------------------------------------------
# networks
# --------------------------------------------------------------------------------------
if _HAS_TORCH:  # pragma: no cover - exercised only with torch installed

    def _activation_module(name: str):
        name = (name or "tanh").lower()
        table = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "gelu": nn.GELU,
            "leaky_relu": nn.LeakyReLU,
        }
        return table.get(name, nn.Tanh)

    def _mlp(sizes: Sequence[int], activation: str = "tanh") -> "nn.Sequential":
        layers: List[Any] = []
        act = _activation_module(activation)
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(int(sizes[i]), int(sizes[i + 1])))
            if i < len(sizes) - 2:
                layers.append(act())
        net = nn.Sequential(*layers)
        for module in net.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                nn.init.zeros_(module.bias)
        return net

    class GAILDiscriminator(nn.Module):
        """GAIL discriminator ``D(s, a)`` returning a single expert/imitation logit.

        Trained with binary cross-entropy: expert pairs get label 1, policy pairs label 0.
        """

        def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            hidden_sizes: Sequence[int] = DEFAULT_DISC_HIDDEN_SIZES,
            activation: str = "tanh",
            discrete: bool = False,
        ) -> None:
            super().__init__()
            self.obs_dim = int(obs_dim)
            self.action_dim = int(action_dim)
            self.discrete = bool(discrete)
            feature_dim = self.action_dim if self.discrete else self.action_dim
            sizes = [self.obs_dim + feature_dim] + [int(h) for h in hidden_sizes] + [1]
            self.net = _mlp(sizes, activation)

        def forward(self, observations: Any, actions: Any) -> Any:
            obs_t = observations if torch.is_tensor(observations) else _to_tensor(observations, "cpu", dtype=torch.float32)
            act_t = actions if torch.is_tensor(actions) else _to_tensor(actions, "cpu", dtype=torch.float32)
            act_t = act_t.reshape(obs_t.shape[0], -1)
            x = torch.cat([obs_t.reshape(obs_t.shape[0], -1), act_t], dim=-1)
            return self.net(x).reshape(-1)

        # -- diagnostics ---------------------------------------------------------------
        def expert_probability(self, observations: Any, actions: Any) -> np.ndarray:
            with torch.no_grad():
                return torch.sigmoid(self.forward(observations, actions)).cpu().numpy().reshape(-1)

        def imitation_reward(self, observations: Any, actions: Any) -> np.ndarray:
            with torch.no_grad():
                return gail_reward(self.forward(observations, actions))

        def discriminator_loss(self, expert_obs, expert_act, policy_obs, policy_act, l2: float = 0.0, smoothing: float = 0.0):
            expert_logits = self.forward(expert_obs, expert_act)
            policy_logits = self.forward(policy_obs, policy_act)
            one = torch.full_like(expert_logits, 1.0 - float(smoothing))
            zero = torch.full_like(policy_logits, float(smoothing))
            loss = F.binary_cross_entropy_with_logits(expert_logits, one) + F.binary_cross_entropy_with_logits(
                policy_logits, zero
            )
            if l2 > 0:
                reg = torch.tensor(0.0, device=loss.device)
                for param in self.parameters():
                    reg = reg + param.pow(2).sum()
                loss = loss + float(l2) * reg
            with torch.no_grad():
                acc = 0.5 * (
                    (torch.sigmoid(expert_logits) > 0.5).float().mean()
                    + (torch.sigmoid(policy_logits) <= 0.5).float().mean()
                )
                expert_score = torch.sigmoid(expert_logits).mean()
            return loss, float(acc.item()), float(expert_score.item())

else:  # pragma: no cover - torch missing

    class GAILDiscriminator:  # type: ignore[no-redef]
        """Placeholder discriminator when PyTorch is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("GAILDiscriminator requires PyTorch.")


# --------------------------------------------------------------------------------------
# rollout container
# --------------------------------------------------------------------------------------
@dataclass
class GAILRollout:
    """Dataset collected during one GAIL iteration."""

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    disc_rewards: np.ndarray
    task_rewards: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray
    dones: np.ndarray
    truncated: np.ndarray
    advantages: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    returns: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    episode_rewards: List[float] = field(default_factory=list)
    episode_task_returns: List[float] = field(default_factory=list)
    start_modes: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return int(np.asarray(self.rewards).shape[0])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "length": len(self),
            "mean_disc_reward": float(np.mean(self.disc_rewards)) if len(self) else 0.0,
            "mean_task_reward": float(np.mean(self.task_rewards)) if len(self) else 0.0,
            "episodes": len(self.episode_task_returns),
        }


# --------------------------------------------------------------------------------------
# expert demonstration collection
# --------------------------------------------------------------------------------------
def collect_expert_demonstrations(
    env: Any,
    expert_policy: Any,
    n_timesteps: int = DEFAULT_EXPERT_TIMESTEPS,
    seed: Optional[int] = None,
    deterministic: bool = True,
    discrete: Optional[bool] = None,
    device: str = "cpu",
    max_steps: Optional[int] = None,
    progress: bool = False,
    rng: Any = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Roll the frozen expert policy and collect ``(s, a)`` demonstrations.

    Returns ``(observations, actions, info)`` where ``info`` holds the number of
    collected transitions and the mean episode return of the expert (used for
    Experiment-IV reporting of the pre-trained SAC agent's performance).
    """
    rng = _as_rng(rng, seed)
    discrete_flag = bool(discrete) if discrete is not None else _action_dim_of(env, expert_policy)[1]
    observations: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    episode_returns: List[float] = []
    episode_return = 0.0
    steps = 0
    episode_steps = 0
    result = env.reset()
    obs, _info = _unpack_reset(result)
    while steps < int(n_timesteps):
        action = sample_policy_action(
            expert_policy, obs, deterministic=deterministic, discrete=discrete_flag,
            action_space=getattr(env, "action_space", None), device=device,
        )
        env_action = prepare_action_for_env(action, getattr(env, "action_space", None), discrete_flag)
        result = env.step(env_action)
        next_obs, reward, terminated, truncated, _info = _unpack_step(result)
        observations.append(_flat(obs))
        actions.append(np.asarray(action, dtype=np.float32).ravel() if not discrete_flag else np.asarray([int(np.asarray(action).ravel()[0])]))
        episode_return += float(reward)
        steps += 1
        episode_steps += 1
        done = bool(terminated or truncated)
        limit_reached = max_steps is not None and episode_steps >= int(max_steps)
        if done or limit_reached:
            episode_returns.append(episode_return)
            episode_return = 0.0
            episode_steps = 0
            obs, _info = _unpack_reset(env.reset())
        else:
            obs = next_obs
    info = {
        "n_transitions": int(len(observations)),
        "n_episodes": int(len(episode_returns)),
        "expert_mean_return": float(np.mean(episode_returns)) if episode_returns else float("nan"),
        "expert_std_return": float(np.std(episode_returns)) if episode_returns else float("nan"),
        "discrete": bool(discrete_flag),
    }
    if progress:
        _LOGGER.info("Collected %d expert transitions (%d episodes, mean return %.2f)",
                     info["n_transitions"], info["n_episodes"], info["mean_return"] if False else info["expert_mean_return"])
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
        info,
    )


# --------------------------------------------------------------------------------------
# behaviour cloning warm start
# --------------------------------------------------------------------------------------
def behavior_clone(
    policy: Any,
    observations: Any,
    actions: Any,
    epochs: int = DEFAULT_BC_EPOCHS,
    batch_size: int = DEFAULT_BC_BATCH_SIZE,
    lr: float = DEFAULT_LR,
    discrete: bool = False,
    device: str = "cpu",
    rng: Any = None,
    logger: Any = None,
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM,
) -> Dict[str, float]:
    """Supervised warm start of ``π_θ`` on expert ``(s, a)`` pairs.

    Pragmatic default (the paper does not specify GAIL initialisation): a short
    behaviour-cloning phase makes the adversarial loop stable for short budgets and
    leaves the GAIL objective to refine the policy afterwards.
    """
    if not _HAS_TORCH:
        return {"bc_loss": float("nan"), "epochs": 0.0}
    obs_all = np.asarray(observations, dtype=np.float32)
    act_all = np.asarray(actions, dtype=np.float32)
    if obs_all.shape[0] == 0 or epochs <= 0:
        return {"bc_loss": float("nan"), "epochs": 0.0}
    rng = _as_rng(rng)
    optimizer = Adam(policy.parameters(), lr=float(lr))
    n = obs_all.shape[0]
    last_loss = float("nan")
    for _epoch in range(int(epochs)):
        index = rng.permutation(n)
        for start in range(0, n, int(batch_size)):
            batch_idx = index[start:start + int(batch_size)]
            if batch_idx.size == 0:
                continue
            obs_t = _to_tensor(obs_all[batch_idx], device, dtype=torch.float32)
            if discrete:
                act_t = _to_tensor(act_all[batch_idx].reshape(-1).astype(np.int64), device)
            else:
                act_t = _to_tensor(act_all[batch_idx].reshape(len(batch_idx), -1), device, dtype=torch.float32)
            try:
                dist = get_policy_distribution(policy, obs_t)
            except Exception:
                break
            if discrete:
                loss = -dist.log_prob(act_t).mean()
            else:
                loss = F.mse_loss(dist.mean, act_t) if hasattr(dist, "mean") else -dist.log_prob(act_t).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), float(max_grad_norm))
            optimizer.step()
            last_loss = float(loss.detach().item())
    if logger is not None:
        try:
            logger.record(**{"gail/bc_loss": last_loss})
        except Exception:
            pass
    return {"bc_loss": last_loss, "epochs": float(epochs)}


# --------------------------------------------------------------------------------------
# GAIL trainer
# --------------------------------------------------------------------------------------
class GAILTrainer:
    """GAIL policy-approximation trainer (Experiment IV of the paper).

    The trainer learns an *approximated policy network* ``π_G`` that imitates the frozen
    pre-trained expert (a SAC agent in Experiment IV). The learned policy can be queried
    with :meth:`evaluate`, saved with :meth:`save`, and — for the full Experiment-IV
    pipeline — handed to RICE's Stage 2 via :meth:`GAILRefiner.refine`.
    """

    def __init__(
        self,
        env: Any,
        expert_policy: Any = None,
        policy: Any = None,
        config: Any = None,
        env_id: str = "default",
        device: str = "cpu",
        logger: Any = None,
        demonstrations: Optional[Tuple[Any, Any]] = None,
        observation_space: Any = None,
        action_space: Any = None,
        discrete: Optional[bool] = None,
        discriminator: Any = None,
        rng: Any = None,
        seed: Optional[int] = None,
        copy_policy: bool = True,
        **kwargs: Any,
    ) -> None:
        if not _HAS_TORCH:
            raise ImportError("GAILTrainer requires PyTorch.")

        self.config = GAILConfig.from_dict(config, **kwargs) if config is not None or kwargs else GAILConfig()
        if env_id and env_id != "default":
            self.config.env_id = normalize_env_key(env_id)
        self.env = env
        self.env_id = normalize_env_key(self.config.env_id)
        self.device = str(device or self.config.device or "cpu")
        self.logger = logger if logger is not None else _LOGGER
        self.rng = _as_rng(rng if rng is not None else self.config.seed, seed)
        self.seed = seed if seed is not None else self.config.seed
        if self.seed is not None:
            try:
                set_seed(int(self.seed))
            except Exception:
                pass

        self.observation_space = observation_space if observation_space is not None else getattr(env, "observation_space", None)
        self.action_space = action_space if action_space is not None else getattr(env, "action_space", None)
        self.discrete = bool(discrete) if discrete is not None else _action_dim_of(env)[1]
        self.action_dim, _ = _action_dim_of(env)
        self.obs_dim = _obs_dim_of(env, policy)
        self.n_steps = int(self.config.n_steps or self._resolve_horizon(env))

        # --- policies -----------------------------------------------------------------
        self.expert_policy = expert_policy
        if self.expert_policy is not None:
            try:
                self.expert_policy.eval()
            except Exception:
                pass
            for param in getattr(self.expert_policy, "parameters", lambda: [])():
                param.requires_grad_(False)

        self.policy = policy if policy is not None else self._build_policy()
        self.policy_initialized = policy is not None
        self.copy_policy = bool(copy_policy)

        # --- discriminator ------------------------------------------------------------
        self.discrete_features = bool(self.config.use_discriminator)
        self.discriminator = discriminator
        if self.discriminator is None and self.config.use_discriminator:
            in_action_dim = int(self.action_dim) if True else int(self.action_dim)
            self.discriminator = GAILDiscriminator(
                obs_dim=int(self.obs_dim),
                action_dim=in_action_dim,
                hidden_sizes=self.config.disc_hidden_sizes,
                activation=self.config.disc_activation,
                discrete=self.discrete,
            ).to(self.device)
        if self.discriminator is not None:
            self.optimizer_disc = Adam(self.discriminator.parameters(), lr=float(self.config.lr_disc))
        else:
            self.optimizer_disc = None
        self.optimizer_policy = Adam(self.policy.parameters(), lr=float(self.config.learning_rate))

        # --- expert data --------------------------------------------------------------
        self.expert_observations: Optional[np.ndarray] = None
        self.expert_actions: Optional[np.ndarray] = None
        self.expert_features: Optional[np.ndarray] = None
        self.expert_info: Dict[str, Any] = {}
        if demonstrations is not None:
            obs_d, act_d = demonstrations[0], demonstrations[1]
            self.set_expert_data(obs_d, act_d)

        # --- rollout state ------------------------------------------------------------
        self._current_obs: Optional[np.ndarray] = None
        self._episode_return = 0.0
        self._episode_task_return = 0.0
        self._episode_steps = 0

        # --- bookkeeping --------------------------------------------------------------
        self.history: List[Dict[str, float]] = []
        self.eval_history: List[Dict[str, float]] = []
        self.timers: Dict[str, float] = {}
        self._timer_ref: Dict[str, float] = {}
        self.total_samples = 0
        self.bc_stats: Dict[str, float] = {}
        reward_stats_ready = hasattr(self, "_reward_stats")
        if not reward_stats_ready:
            self._reward_stats = {"count": 0.0, "mean": 0.0, "var": 1.0}

    # -- construction helpers ---------------------------------------------------------
    def _resolve_horizon(self, env: Any) -> int:
        node = env
        seen = set()
        while node is not None and id(node) not in seen:
            seen.add(id(node))
            for attr in ("rice_max_episode_steps", "_max_episode_steps"):
                value = getattr(node, attr, None)
                if value:
                    try:
                        return int(value)
                    except Exception:
                        pass
            spec = getattr(node, "spec", None)
            if spec is not None and getattr(spec, "max_episode_steps", None):
                try:
                    return int(spec.max_episode_steps)
                except Exception:
                    pass
            node = getattr(node, "env", None)
        return DEFAULT_HORIZON

    def _build_policy(self) -> Any:
        if build_policy is None:
            raise RuntimeError("rice.models.policies.build_policy is unavailable; pass a policy explicitly.")
        try:
            return build_policy(
                env_id=self.env_id,
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                action_space=self.action_space,
                observation_space=self.observation_space,
                backend="auto",
                discrete=self.discrete,
                device=self.device,
                hidden_sizes=self.config.hidden_sizes,
                activation=self.config.activation,
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Could not build a policy for env '{self.env_id}': {exc}") from exc

    # -- expert data ------------------------------------------------------------------
    def set_expert_data(self, observations: Any, actions: Any, info: Optional[Dict[str, Any]] = None) -> None:
        """Register expert ``(s, a)`` demonstrations used by the discriminator."""
        obs_arr = np.asarray(observations, dtype=np.float32)
        act_arr = np.asarray(actions, dtype=np.float32)
        if obs_arr.ndim == 1:
            obs_arr = obs_arr.reshape(1, -1)
        if act_arr.ndim == 1:
            act_arr = act_arr.reshape(-1, 1)
        self.expert_observations = obs_arr
        self.expert_actions = act_arr
        self.expert_features = action_features(
            act_arr, discrete=self.discrete, action_dim=int(self.action_dim)
        )
        if info:
            self.expert_info = dict(info)

    def collect_expert(self, n_timesteps: Optional[int] = None, expert_policy: Any = None, progress: bool = False) -> Dict[str, Any]:
        """Collect expert demonstrations from the frozen pre-trained (SAC) agent."""
        policy = expert_policy if expert_policy is not None else self.expert_policy
        if policy is None:
            raise ValueError("collect_expert requires an expert policy.")
        self.expert_policy = policy
        n_timesteps = int(n_timesteps or self.config.expert_timesteps)
        obs, acts, info = collect_expert_demonstrations(
            self.env, policy, n_timesteps=n_timesteps, seed=self.seed,
            deterministic=bool(self.config.expert_deterministic), discrete=self.discrete,
            device=self.device, progress=progress, rng=self.rng,
        )
        self.set_expert_data(obs, acts, info)
        return info

    def _sample_expert_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.expert_observations is None or self.expert_actions is None:
            raise RuntimeError("No expert demonstrations registered (call collect_expert / set_expert_data).")
        n = self.expert_observations.shape[0]
        index = self.rng.randint(0, n, size=int(batch_size))
        return self.expert_observations[index], self.expert_actions[index]

    # -- reward -----------------------------------------------------------------------
    def discriminator_rewards(self, observations: Any, actions: Any) -> np.ndarray:
        """GAIL imitation reward ``-log(1 - D(s,a))`` for a batch of transitions."""
        if self.discriminator is None:
            return np.zeros(int(np.asarray(observations).shape[0]), dtype=np.float32)
        with torch.no_grad():
            obs_t = _to_tensor(np.asarray(observations, dtype=np.float32), self.device, dtype=torch.float32)
            act_t = _to_tensor(np.asarray(actions, dtype=np.float32).reshape(len(np.asarray(observations)), -1),
                               self.device, dtype=torch.float32)
            logits = self.discriminator(obs_t, act_t)
            rewards = gail_reward(logits)
        if self.config.normalize_reward:
            rewards = self._normalize_reward(rewards)
        if self.config.reward_clip is not None:
            rewards = np.clip(rewards, -float(self.config.reward_clip), float(self.config.reward_clip))
        return rewards.astype(np.float32)

    def _normalize_reward(self, rewards: np.ndarray) -> np.ndarray:
        """Standardise the intrinsic reward with running mean/std (default, unspecified)."""
        arr = np.asarray(rewards, dtype=np.float64).reshape(-1)
        stats = self._reward_stats
        count = stats["count"]
        for value in arr:
            count += 1.0
            delta = value - stats["mean"]
            stats["mean"] += delta / count
            stats["var"] += delta * (value - stats["mean"])
        stats["count"] = count
        std = math.sqrt(max(stats["var"] / max(count, 1.0), 1e-8))
        return (arr / std).astype(np.float32)

    def _combine_reward(self, disc_reward: float, task_reward: float) -> float:
        mode = self.config.reward_mode
        if mode == "gail":
            return float(disc_reward)
        if mode == "gail_task":
            return float(disc_reward) + float(self.config.task_reward_coef) * float(task_reward)
        if mode == "task":
            return float(task_reward)
        if mode == "bc":
            return float(task_reward)
        return float(disc_reward)

    # -- rollouts ---------------------------------------------------------------------
    def _policy_step(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Sample an action and evaluate ``(log_prob, value, entropy)`` for it."""
        device = _policy_device(self.policy, self.device)
        obs_t = _to_tensor(np.asarray(_flat(obs), dtype=np.float32), device, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dist = None
            try:
                dist = get_policy_distribution(self.policy, obs_t)
            except Exception:
                dist = None
            if dist is not None:
                action_t = dist.mean if (deterministic and getattr(dist, "mean", None) is not None) else dist.sample()
                log_prob_t = dist.log_prob(action_t.reshape(-1)[0] if self.discrete else action_t)
                if self.discrete:
                    log_prob_t = dist.log_prob(action_t.reshape(-1)[0])
                value_t = predict_values_tensor(self.policy, obs_t).reshape(-1)[0]
            else:
                raise RuntimeError("Policy exposes no distribution accessor for GAIL rollouts.")
        action = _to_numpy(action_t).ravel()
        if self.discrete:
            action = np.asarray([int(action.reshape(-1)[0])], dtype=np.int64)
        else:
            action = np.asarray(action, dtype=np.float32).ravel()
        return action, np.asarray([float(log_prob_t.item())], dtype=np.float32), float(value_t.item()), 0.0

    def collect_rollout(self, n_steps: Optional[int] = None) -> GAILRollout:
        """Collect ``n_steps`` transitions of ``π_θ`` with mixed task/GAIL rewards."""
        n_steps = int(n_steps or self.n_steps)
        observations, next_observations, actions = [], [], []
        rewards, disc_rewards, task_rewards = [], [], []
        log_probs, values, dones, truncated = [], [], [], []
        episode_rewards: List[float] = []
        episode_task_returns: List[float] = []

        if self._current_obs is None:
            obs, _info = _unpack_reset(self.env.reset())
            self._current_obs = _flat(obs)
            self._episode_return = 0.0
            self._episode_task_return = 0.0
            self._episode_steps = 0

        for _ in range(n_steps):
            obs = self._current_obs
            action_t, log_prob, value, _entropy = self._policy_step(obs, deterministic=False)
            env_action = prepare_action_for_env(action_t, self.action_space, self.discrete)
            result = self.env.step(env_action)
            next_obs_raw, task_reward, terminated, trunc, _info = _unpack_step(result)
            next_obs = _flat(next_obs_raw)

            disc = float(self.discriminator_rewards(obs.reshape(1, -1), env_action.reshape(1, -1) if not self.discrete
                                                   else action_t.reshape(1, -1))[0])
            reward = self._combine_reward(disc, task_reward)

            observations.append(obs)
            next_observations.append(next_obs)
            actions.append(action_t.astype(np.float32).ravel())
            rewards.append(reward)
            disc_rewards.append(disc)
            task_rewards.append(float(task_reward))
            log_probs.append(float(np.asarray(log_prob).reshape(-1)[0]))
            values.append(float(value))
            done = bool(terminated or trunc)
            dones.append(1.0 if terminated else 0.0)
            truncated.append(1.0 if trunc else 0.0)

            self._episode_return += reward
            self._episode_task_return += float(task_reward)
            self._episode_steps += 1
            self.total_samples += 1

            if done:
                episode_rewards.append(self._episode_return)
                episode_task_returns.append(self._episode_task_return)
                obs_reset, _info = _unpack_reset(self.env.reset())
                self._current_obs = _flat(obs_reset)
                self._episode_return = 0.0
                self._episode_task_return = 0.0
                self._episode_steps = 0
            else:
                self._current_obs = next_obs

        # bootstrap value
        with torch.no_grad():
            obs_t = _to_tensor(self._current_obs, _policy_device(self.policy, self.device), dtype=torch.float32).unsqueeze(0)
            last_value = float(predict_values_tensor(self.policy, obs_t).reshape(-1)[0].item())

        batch = GAILRollout(
            observations=np.asarray(observations, dtype=np.float32),
            next_observations=np.asarray(next_observations, dtype=np.float32),
            actions=np.asarray(actions, dtype=np.float32),
            rewards=np.asarray(rewards, dtype=np.float32),
            disc_rewards=np.asarray(disc_rewards, dtype=np.float32),
            task_rewards=np.asarray(task_rewards, dtype=np.float32),
            log_probs=np.asarray(log_probs, dtype=np.float32),
            values=np.asarray(values, dtype=np.float32),
            dones=np.asarray(dones, dtype=np.float32),
            truncated=np.asarray(truncated, dtype=np.float32),
            episode_rewards=episode_rewards,
            episode_task_returns=episode_task_returns,
        )
        batch.advantages, batch.returns = compute_gae(
            batch.rewards, batch.values, batch.dones, last_value=last_value,
            gamma=float(self.config.gamma), gae_lambda=float(self.config.gae_lambda),
        )
        return batch

    # -- updates ----------------------------------------------------------------------
    def update_discriminator(self, batch: GAILRollout) -> Dict[str, float]:
        """Train ``D`` with BCE: expert pairs (label 1) vs policy pairs (label 0)."""
        if self.discriminator is None or self.optimizer_disc is None:
            return {"disc_loss": float("nan"), "disc_accuracy": float("nan"), "expert_prob": float("nan")}
        if self.expert_observations is None:
            raise RuntimeError("Discriminator training requires expert demonstrations.")
        policy_obs = batch.observations
        policy_act = batch.actions if not self.discrete else action_features(
            batch.actions, discrete=True, action_dim=int(self.action_dim)
        )
        losses: List[float] = []
        accs: List[float] = []
        expert_probs: List[float] = []
        bs = int(self.config.disc_batch_size)
        for _ in range(int(max(1, self.config.disc_epochs * self.config.disc_updates))):
            for start in range(0, policy_obs.shape[0], bs):
                p_obs = policy_obs[start:start + bs]
                p_act = policy_act[start:start + bs]
                e_obs, e_act = self._sample_expert_batch(p_obs.shape[0])
                if self.discrete:
                    e_act = action_features(e_act, discrete=True, action_dim=int(self.action_dim))
                e_obs_t = _to_tensor(e_obs, self.device, dtype=torch.float32)
                e_act_t = _to_tensor(e_act, self.device, dtype=torch.float32)
                p_obs_t = _to_tensor(p_obs, self.device, dtype=torch.float32)
                p_act_t = _to_tensor(p_act, self.device, dtype=torch.float32)
                loss, acc, eprob = self.discriminator.discriminator_loss(
                    e_obs_t, e_act_t, p_obs_t, p_act_t,
                    l2=float(self.config.disc_l2), smoothing=float(self.config.disc_label_smoothing),
                )
                self.optimizer_disc.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), float(self.config.max_grad_norm))
                self.optimizer_disc.step()
                losses.append(float(loss.detach().item()))
                accs.append(float(acc))
                expert_probs.append(float(eprob))
        return {
            "disc_loss": float(np.mean(losses)) if losses else float("nan"),
            "disc_accuracy": float(np.mean(accs)) if accs else float("nan"),
            "expert_prob": float(np.mean(expert_probs)) if expert_probs else float("nan"),
        }

    def update_policy(self, batch: GAILRollout) -> Dict[str, float]:
        """PPO clipped update on the (GAIL) reward collected in ``batch``."""
        obs_all = batch.observations
        n = obs_all.shape[0]
        if n == 0:
            return {"policy_loss": float("nan")}
        advantages = np.asarray(batch.advantages, dtype=np.float32).copy()
        returns = np.asarray(batch.returns, dtype=np.float32).copy()
        if self.config.normalize_advantage and advantages.size > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        old_log_probs = np.asarray(batch.log_probs, dtype=np.float32)
        batch_size = int(min(self.config.batch_size, n))
        device = _policy_device(self.policy, self.device)
        stats: Dict[str, List[float]] = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_fraction": []}
        for _epoch in range(int(self.config.n_epochs)):
            index = self.rng.permutation(n)
            for start in range(0, n, batch_size):
                mb = index[start:start + batch_size]
                if mb.size == 0:
                    continue
                mb_obs = obs_all[mb]
                if self.discrete:
                    mb_act = np.asarray(batch.actions[mb]).reshape(-1).astype(np.int64)
                else:
                    mb_act = np.asarray(batch.actions[mb], dtype=np.float32).reshape(mb.size, -1)
                log_prob, values, entropy = evaluate_policy_actions(
                    self.policy, mb_obs, mb_act, discrete=self.discrete, device=device
                )
                mb_adv = _to_tensor(advantages[mb], device, dtype=torch.float32)
                mb_ret = _to_tensor(returns[mb], device, dtype=torch.float32)
                mb_old_log = _to_tensor(old_log_probs[mb], device, dtype=torch.float32)
                ratio = torch.exp(log_prob - mb_old_log)
                clip_range = float(self.config.clip_range)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, mb_ret)
                entropy_loss = -entropy.mean()
                loss = policy_loss + float(self.config.vf_coef) * value_loss + float(self.config.ent_coef) * entropy_loss
                self.optimizer_policy.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), float(self.config.max_grad_norm))
                self.optimizer_policy.step()
                with torch.no_grad():
                    approx_kl = (mb_old_log - log_prob).mean()
                    clip_frac = ((ratio - 1.0).abs() > clip_range).float().mean()
                stats["policy_loss"].append(float(policy_loss.item()))
                stats["value_loss"].append(float(value_loss.item()))
                stats["entropy"].append(float(entropy.mean().item()))
                stats["approx_kl"].append(float(approx_kl.item()))
                stats["clip_fraction"].append(float(clip_frac.item()))
                if self.config.target_kl is not None and float(approx_kl.item()) > float(self.config.target_kl):
                    break

        out = {f"gail/{k}": (float(np.mean(v)) if v else float("nan")) for k, v in stats.items()}
        out["gail/mean_reward"] = float(np.mean(batch.rewards)) if len(batch) else float("nan")
        out["gail/mean_disc_reward"] = float(np.mean(batch.disc_rewards)) if len(batch) else float("nan")
        out["gail/mean_task_reward"] = float(np.mean(batch.task_rewards)) if len(batch) else float("nan")
        return out

    # -- training loop ----------------------------------------------------------------
    def timer_start(self, name: str = "gail") -> float:
        self._timer_ref[name] = time.time()
        return self._timer_ref[name]

    def timer_end(self, name: str = "gail", accumulate: bool = True) -> float:
        start = self._timer_ref.pop(name, None)
        elapsed = 0.0 if start is None else (time.time() - start)
        if accumulate:
            self.timers[name] = self.timers.get(name, 0.0) + elapsed
        else:
            self.timers[name] = elapsed
        return elapsed

    def time_report(self) -> Dict[str, float]:
        return dict(self.timers)

    def train(
        self,
        total_timesteps: Optional[int] = None,
        total_iterations: Optional[int] = None,
        logger: Any = None,
        progress: bool = False,
        callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
    ) -> List[Dict[str, float]]:
        """Run the GAIL loop (BC warm start → adversarial training)."""
        logger = logger if logger is not None else self.logger
        if self.expert_observations is None:
            if self.expert_policy is None:
                raise RuntimeError("GAIL requires either expert demonstrations or an expert_policy.")
            self.collect_expert(progress=progress)

        # behaviour-cloning warm start (pragmatic default, see module docstring)
        if self.config.bc_pretrain_epochs > 0:
            self.timer_start("bc")
            self.bc_stats = behavior_clone(
                self.policy, self.expert_observations, self.expert_actions,
                epochs=int(self.config.bc_pretrain_epochs), batch_size=int(self.config.bc_batch_size),
                lr=float(self.config.bc_lr or self.config.learning_rate), discrete=self.discrete,
                device=self.device, rng=self.rng, logger=logger,
            )
            self.timer_end("bc")

        total_timesteps = int(total_timesteps or self.config.total_timesteps)
        n_iterations = int(total_iterations or max(1, math.ceil(total_timesteps / max(self.n_steps, 1))))

        self.timer_start("train")
        for iteration in range(1, n_iterations + 1):
            batch = self.collect_rollout()
            stats: Dict[str, float] = {}
            if self.config.use_discriminator:
                stats.update(self.update_discriminator(batch))
            stats.update(self.update_policy(batch))
            stats["gail/iteration"] = float(iteration)
            stats["gail/total_samples"] = float(self.total_samples)
            for value in batch.episode_task_returns:
                stats.setdefault("gail/episode_task_return", value)
            if batch.episode_task_returns:
                stats["gail/episode_task_return"] = float(np.mean(batch.episode_task_returns))
                stats["gail/episode_return"] = float(np.mean(batch.episode_rewards))
            self.history.append(stats)
            if logger is not None:
                try:
                    logger.record(**stats)
                except Exception:
                    pass
            if progress and (iteration % max(1, int(self.config.log_interval)) == 0):
                _LOGGER.info(
                    "[GAIL %s] iter %d/%d samples=%d mean_reward=%.3f disc_acc=%.3f",
                    self.env_id, iteration, n_iterations, self.total_samples,
                    stats.get("gail/mean_reward", float("nan")), stats.get("disc_accuracy", float("nan")),
                )
            if callback is not None:
                callback(iteration, stats)
        self.timer_end("train")
        return self.history

    fit = train

    # -- evaluation -------------------------------------------------------------------
    def evaluate(
        self,
        n_episodes: int = 10,
        deterministic: bool = True,
        policy: Any = None,
        max_steps: Optional[int] = None,
        reset_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Evaluate a policy (defaults to the current ``π_θ``) with the task reward."""
        policy = policy if policy is not None else self.policy
        device = _policy_device(policy, self.device)
        returns: List[float] = []
        lengths: List[int] = []
        horizon = int(max_steps or self._resolve_horizon(self.env))
        for _ in range(int(n_episodes)):
            result = self.env.reset(**(reset_kwargs or {}))
            obs, _info = _unpack_reset(result)
            done = False
            steps = 0
            ep_return = 0.0
            while not done and steps < horizon:
                action = sample_policy_action(
                    policy, obs, deterministic=deterministic, discrete=self.discrete,
                    action_space=self.action_space, device=device,
                )
                env_action = prepare_action_for_env(action, self.action_space, self.discrete)
                result = self.env.step(env_action)
                obs, reward, terminated, trunc, _info = _unpack_step(result)
                ep_return += float(reward)
                steps += 1
                done = bool(terminated or trunc)
            returns.append(ep_return)
            lengths.append(steps)
        result = {
            "mean_return": float(np.mean(returns)) if returns else float("nan"),
            "std_return": float(np.std(returns)) if returns else float("nan"),
            "mean_length": float(np.mean(lengths)) if lengths else float("nan"),
            "episodes": float(len(returns)),
        }
        self.eval_history.append(result)
        return result

    # -- persistence / reporting -------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        return {
            "method": "GAIL",
            "env_id": self.env_id,
            "total_samples": int(self.total_samples),
            "iterations": len(self.history),
            "n_steps": int(self.n_steps),
            "reward_mode": self.config.reward_mode,
            "bc_epochs": int(self.config.bc_pretrain_epochs),
            "bc_loss": float(self.bc_stats.get("bc_loss", float("nan"))),
            "expert_mean_return": float(self.expert_info.get("expert_mean_return", float("nan"))),
            "last_disc_accuracy": float(self.history[-1].get("disc_accuracy", float("nan"))) if self.history else float("nan"),
            "total_time": float(self.timers.get("train", 0.0)),
        }

    def save(self, path: str, extra: Optional[Dict[str, Any]] = None) -> str:
        """Persist the approximated policy (and discriminator, if present)."""
        if not _HAS_TORCH:
            raise ImportError("Saving a GAIL checkpoint requires PyTorch.")
        directory = os.path.dirname(os.path.abspath(path))
        ensure_dir(directory)
        payload: Dict[str, Any] = {
            "kind": "gail",
            "env_id": self.env_id,
            "policy_state_dict": self.policy.state_dict(),
            "obs_dim": int(self.obs_dim),
            "action_dim": int(self.action_dim),
            "discrete": bool(self.discrete),
            "hidden_sizes": list(self.config.hidden_sizes),
            "config": self.config.to_dict(),
            "summary": self.summary(),
        }
        if extra:
            payload.update(extra)
        if self.discriminator is not None:
            try:
                payload["discriminator_state_dict"] = self.discriminator.state_dict()
            except Exception:
                pass
        torch.save(payload, path)
        return path

    def save_policy(self, path: str, extra: Optional[Dict[str, Any]] = None) -> str:
        """Backward-compatible alias of :meth:`save`."""
        return self.save(path, extra=extra)


# --------------------------------------------------------------------------------------
# GAIL as a refining baseline (RICE Stage-2 delegation)
# --------------------------------------------------------------------------------------
class GAILRefiner(GAILTrainer):
    """GAIL baseline exposing the same surface as the other refining baselines.

    Pipeline of Experiment IV::

        π_SAC  --(frozen expert)-->  GAIL  -->  π_G  --(RICE refine)-->  π'

    ``approximate_policy`` learns ``π_G``; ``refine`` then runs RICE's Algorithm 2 with
    the identical explanation (mask network) used by RICE, so the comparison isolates the
    refining mechanism.
    """

    name = "GAIL"

    def approximate_policy(self, total_timesteps: Optional[int] = None, progress: bool = False, **kwargs: Any) -> Any:
        """Learn the approximated policy network and return it."""
        self.train(total_timesteps=total_timesteps, progress=progress, **kwargs)
        return self.policy

    @property
    def policy_network(self) -> Any:
        return self.policy

    @property
    def approximated_policy(self) -> Any:
        return getattr(self, "policy", None)

    def refine(
        self,
        env: Any = None,
        mask_net: Any = None,
        total_timesteps: Optional[int] = None,
        total_iterations: Optional[int] = None,
        config: Any = None,
        logger: Any = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Any]:
        """Refine ``π_G`` with RICE (Algorithm 2) using the shared explanation."""
        if not _HAS_REFINER:
            raise RuntimeError("rice.refining.ppo_refine is unavailable; cannot run RICE refinement.")
        env = env if env is not None else self.env
        refined_policy, refiner = refine_policy(  # type: ignore[misc]
            env,
            policy=self.policy,
            mask_net=mask_net,
            total_timesteps=total_timesteps,
            total_iterations=total_iterations,
            env_id=self.env_id,
            config=config,
            logger=logger if logger is not None else self.logger,
            seed=seed if seed is not None else self.seed,
            device=self.device,
            **kwargs,
        )
        self.refined_policy = refined_policy
        self.refiner = refiner
        return refined_policy, refiner

    def run(
        self,
        env: Any = None,
        expert_policy: Any = None,
        approx_timesteps: Optional[int] = None,
        refine_timesteps: Optional[int] = None,
        refine_iterations: Optional[int] = None,
        mask_net: Any = None,
        refine_config: Any = None,
        evaluate: bool = True,
        eval_episodes: int = 10,
        save_dir: Optional[str] = None,
        progress: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Full Experiment-IV GAIL pipeline (approximate then refine)."""
        env = env if env is not None else self.env
        if expert_policy is not None:
            self.expert_policy = expert_policy
        if self.expert_observations is None:
            self.collect_expert(progress=progress)
        results: Dict[str, Any] = {"method": self.name, "env_id": self.env_id}
        if evaluate:
            results["expert_eval"] = self.evaluate(n_episodes=eval_episodes, policy=self.expert_policy) if self.expert_policy is not None else {}

        self.approximate_policy(total_timesteps=approx_timesteps, progress=progress)
        results["approx_summary"] = self.summary()
        if evaluate:
            results["approx_eval"] = self.evaluate(n_episodes=eval_episodes)

        if refine_timesteps or refine_iterations:
            refined_policy, refiner = self.refine(
                env=env, mask_net=mask_net, total_timesteps=refine_timesteps,
                total_iterations=refine_iterations, config=refine_config, progress=progress,
            )
            results["refined_policy"] = refined_policy
            if evaluate:
                if evaluate_refined_policy is not None:  # type: ignore[misc]
                    results["refined_eval"] = evaluate_refined_policy(
                        env, refined_policy, env_id=self.env_id, n_episodes=eval_episodes,
                        discrete=self.discrete, device=self.device,
                    )
                else:
                    results["refined_eval"] = self.evaluate(n_episodes=eval_episodes, policy=refined_policy)

        if save_dir:
            ensure_dir(save_dir)
            results["checkpoint"] = self.save(os.path.join(save_dir, f"gail_{self.env_id}.pt"))
        return results


# --------------------------------------------------------------------------------------
# module-level API
# --------------------------------------------------------------------------------------
def train_gail(
    env: Any,
    expert_policy: Any = None,
    policy: Any = None,
    total_timesteps: Optional[int] = None,
    total_iterations: Optional[int] = None,
    expert_timesteps: Optional[int] = None,
    env_id: str = "default",
    config: Any = None,
    logger: Any = None,
    demonstrations: Optional[Tuple[Any, Any]] = None,
    save_path: Optional[str] = None,
    seed: Optional[int] = None,
    device: str = "cpu",
    progress: bool = False,
    evaluate: bool = False,
    eval_episodes: int = 10,
    **kwargs: Any,
) -> Tuple[Any, GAILTrainer]:
    """Learn an approximated policy network from a pre-trained expert via GAIL.

    Returns ``(policy, trainer)`` — the policy is the GAIL-approximated policy network
    ``π_G`` that RICE (or any refining baseline) can subsequently refine.
    """
    overrides = dict(kwargs)
    if total_timesteps is not None:
        overrides["total_timesteps"] = total_timesteps
    if expert_timesteps is not None:
        overrides["expert_timesteps"] = expert_timesteps
    if seed is not None:
        overrides["seed"] = seed
    trainer = GAILRefiner(
        env=env,
        expert_policy=expert_policy,
        policy=policy,
        config=config,
        env_id=env_id,
        device=device,
        logger=logger,
        demonstrations=demonstrations,
        rng=get_rng(seed if seed is not None else None),
        seed=seed,
        **overrides,
    )
    if trainer.expert_observations is None:
        if trainer.expert_policy is None:
            raise ValueError("train_gail requires either expert_policy or demonstrations.")
        trainer.collect_expert(progress=progress)
    trainer.train(total_timesteps=total_timesteps, total_iterations=total_iterations, progress=progress)
    if evaluate:
        trainer.evaluate(n_episodes=eval_episodes)
    if save_path:
        trainer.save(save_path)
    return trainer.policy, trainer


def approximate_policy_with_gail(env: Any, expert_policy: Any = None, **kwargs: Any) -> Any:
    """Return only the GAIL-approximated policy ``π_G`` (Experiment IV)."""
    policy, _trainer = train_gail(env, expert_policy=expert_policy, **kwargs)
    return policy


def make_gail(
    env: Any = None,
    expert_policy: Any = None,
    policy: Any = None,
    env_id: str = "default",
    config: Any = None,
    seed: Optional[int] = None,
    device: str = "cpu",
    logger: Any = None,
    **kwargs: Any,
) -> GAILRefiner:
    """Factory used by :mod:`rice.baselines` for registry-based dispatch."""
    if isinstance(config, str) and (env_id in (None, "default", "") or env_id == "default"):
        env_id = config
        config = None
    return GAILRefiner(
        env=env,
        expert_policy=expert_policy,
        policy=policy,
        config=config,
        env_id=env_id,
        device=device,
        logger=logger,
        seed=seed,
        rng=get_rng(seed if seed is not None else None),
        **kwargs,
    )


build_gail = make_gail


def gail(
    env: Any,
    expert_policy: Any = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Plain-dict GAIL pipeline helper (approximate + optional RICE refine)."""
    trainer = make_gail(env=env, expert_policy=expert_policy, **kwargs)
    run_kwargs = dict(kwargs)
    for key in (
        "expert_policy", "config", "seed", "device", "logger", "env_id", "policy",
        "expert_timesteps", "bc_pretrain_epochs", "hidden_sizes", "use_discriminator",
    ):
        run_kwargs.pop(key, None)
    return trainer.run(env=env, expert_policy=expert_policy, **run_kwargs)


gail_baseline = gail


def describe_gail(trainer: Any = None) -> str:
    """One-line human-readable description for logging/tables."""
    if trainer is None:
        return (
            "GAIL: learns an approximated policy network from a frozen pre-trained expert "
            "(Experiment IV); can then be refined by RICE."
        )
    summary = trainer.summary() if hasattr(trainer, "summary") else {}
    return (
        f"GAIL(env={summary.get('env_id', 'n/a')}, samples={summary.get('total_samples', 0)}, "
        f"reward_mode={summary.get('reward_mode', DEFAULT_REWARD_MODE)}, "
        f"bc_epochs={summary.get('bc_epochs', 0)})"
    )
