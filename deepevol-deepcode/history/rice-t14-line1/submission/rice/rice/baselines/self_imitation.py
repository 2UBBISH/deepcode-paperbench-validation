"""Self-Imitation Learning (SIL) refining baseline for RICE.

References
----------
* Oh, J., Guo, Y., Singh, S., & Lee, H. (2018). "Self-Imitation Learning."
  ICML 2018.  https://arxiv.org/abs/1806.05635
* RICE (ICML 2024, PMLR 235), Section 4.1 "Baseline Refining Methods" and
  Table 5 ("further comparison between RICE and SIL").

Where this baseline sits in the RICE paper
------------------------------------------
The paper's own refining method (Algorithm 2) is compared against three
refining baselines:

  * "PPO fine-tuning" (Schulman et al. 2017)             -> ppo_finetune.py
  * "StateMask-R" (Cheng et al. 2023)                    -> statemask_r.py
  * "Jump-Start RL" (Uchendu et al. 2023)                -> jsrl.py

Additionally the paper reports a *secondary* comparison (Table 5) between
RICE and Self-Imitation Learning (SIL) on the four MuJoCo tasks, e.g.
Hopper ``3646.46 -> 3663.91``, Walker2d ``3967.66 -> 3982.79``,
Reacher ``-2.87 -> -2.66``, HalfCheetah ``2069.80 -> 2138.89``.
The paper gives no algorithmic detail for the SIL baseline other than the
citation, therefore this module follows the *original* SIL paper:

    L^sil = - sum_t log pi_theta(a_t | s_t) * (R_t - V_theta(s_t))_+      (Eq. 4 in Oh et al.)

i.e. the policy is additionally trained by *imitating past good
experiences* -- transitions whose (Monte-Carlo) return exceeds the value
baseline.  In the original paper SIL is combined with an A2C-style
advantage actor-critic; here we plug the SIL loss into RICE's *shared*
PPO clipped surrogate so that the comparison against RICE / PPO
fine-tuning is apples-to-apples (identical optimizer, buffer, evaluation
protocol, and warm-start checkpoint).

Concretely, per refining iteration we

  1. roll out one episode-length batch with the current policy on the
     default initial-state distribution ``rho`` (SIL, like PPO
     fine-tuning, does *not* use the mixed initial state distribution nor
     the RND bonus -- those are exactly the RICE components under test),
  2. update the policy with the shared PPO loss on the on-policy buffer,
  3. push the *good* transitions (positive advantage) into a
     past-good-experience replay buffer, and
  4. take ``sil_grad_steps_per_iter`` extra gradient steps of the SIL
     loss on batches sampled from that replay buffer, using the same
     Adam optimizer as PPO.

The module returns a ``RefineResult``-compatible object so that
``rice.baselines.run_baseline`` / ``rice.evaluation.refining_eval`` can
consume it side by side with every other method.

Deviation note (documented in README): the paper does not state the SIL
coefficient, the replay-buffer size, the advantage baseline, or the
number of extra SIL gradient steps.  We use ``sil_beta = 1.0`` with the
positive-part weight clipped at ``1.0`` (the SIL paper's clipped
variation, ``c = 1.0``) and GAE advantages from the shared PPO rollout
buffer as the "is this a good experience?" criterion.
"""

from __future__ import annotations

import copy
import importlib
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - torch is expected, but keep the module import-safe
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _TORCH_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Defensive, layout-tolerant imports (repo may be launched from rice/ or rice/rice)
# --------------------------------------------------------------------------- #
def _import_first(candidates: Sequence[str]) -> Any:
    """Import the first importable dotted module from ``candidates``."""
    last_error: Optional[Exception] = None
    for name in candidates:
        try:
            return importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - defensive
            last_error = exc
    raise ImportError(f"could not import any of {list(candidates)}: {last_error}")


def _optional(candidates: Sequence[str]) -> Any:
    try:
        return _import_first(candidates)
    except Exception:  # pragma: no cover - defensive
        return None


def _get(module: Any, *names: str, default: Any = None) -> Any:
    """Return the first attribute present on ``module``."""
    if module is None:
        return default
    for name in names:
        value = getattr(module, name, None)
        if value is not None:
            return value
    return default


_ppo_mod = _optional(["rice.algorithms.ppo", "rice.rice.algorithms.ppo", "algorithms.ppo"])
_refine_mod = _optional(["rice.algorithms.refine", "rice.rice.algorithms.refine", "algorithms.refine"])
_env_mod = _optional(["rice.environments", "rice.rice.environments", "environments"])
_seed_mod = _optional(["rice.utils.seeding", "rice.rice.utils.seeding", "utils.seeding"])

PPO = _get(_ppo_mod, "PPO")
PPOConfig = _get(_ppo_mod, "PPOConfig")
ActorCritic = _get(_ppo_mod, "ActorCritic")
RolloutBuffer = _get(_ppo_mod, "RolloutBuffer")
flatten_obs = _get(_ppo_mod, "flatten_obs", default=lambda x: np.asarray(x, dtype=np.float32).reshape(-1))
make_target_policy_callable = _get(_ppo_mod, "make_target_policy_callable")

RefineConfig = _get(_refine_mod, "RefineConfig")
RefineResult = _get(_refine_mod, "RefineResult")
RefineIteration = _get(_refine_mod, "RefineIteration")
evaluate_policy = _get(_refine_mod, "evaluate_policy")
load_policy_weights = _get(_refine_mod, "load_policy_weights")

make_env = _get(_env_mod, "make_env")
default_net_arch = _get(_env_mod, "default_net_arch")
set_global_seeds = _get(_seed_mod, "set_global_seeds")


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
METHOD_NAME = "sil"
METHOD_ALIASES: Tuple[str, ...] = (
    "sil",
    "self_imitation",
    "self-imitation",
    "selfimitation",
    "self_imitation_learning",
)

DEFAULT_LR_FACTOR = 0.1          # same convention as PPO fine-tuning (lowered LR)
DEFAULT_PRETRAIN_LR = 3e-4       # SB3 PPO default
DEFAULT_N_ITERATIONS = 100
DEFAULT_EVAL_EPISODES = 5
DEFAULT_SIL_BETA = 1.0           # weight of the self-imitation loss term
DEFAULT_SIL_WEIGHT_CLIP = 1.0    # (R_t - V(s_t))_+ clipped at c (SIL paper)
DEFAULT_SIL_BUFFER_SIZE = 200_000
DEFAULT_SIL_GRAD_STEPS = 1
DEFAULT_SIL_BATCH_SIZE = 256
DEFAULT_TABLE5_REFERENCE: Dict[str, Dict[str, float]] = {
    # Table 5: SIL -> RICE (secondary in-scope comparison).
    "Hopper-v3": {"sil": 3646.46, "ours": 3663.91},
    "Walker2d-v3": {"sil": 3967.66, "ours": 3982.79},
    "Reacher-v2": {"sil": -2.87, "ours": -2.66},
    "HalfCheetah-v3": {"sil": 2069.80, "ours": 2138.89},
}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class SILConfig:
    """Hyper-parameters of the Self-Imitation Learning refining baseline.

    The task/iteration/evaluation fields intentionally mirror
    :class:`rice.baselines.ppo_finetune.PPOFineTuneConfig` (and
    :class:`rice.algorithms.refine.RefineConfig`) so that the baseline can be
    driven by the same YAML configs / comparison harness.
    """

    # --- task / accounting -------------------------------------------------- #
    task: str = "Hopper-v3"
    method: str = METHOD_NAME
    explanation: str = "none"           # SIL does not need an explanation
    weights: Any = None                 # warm-start checkpoint (path or state_dict)
    net_arch: Optional[Tuple[int, ...]] = None

    # --- refining loop (mirrors Algorithm 2's outer loop) ------------------- #
    n_iterations: int = DEFAULT_N_ITERATIONS
    steps_per_iter: Optional[int] = None       # T; defaults to env.max_episode_steps
    rollin_length: Optional[int] = None        # K; unused by SIL (no critical-state roll-in)
    total_env_steps: Optional[int] = None      # stop early once this budget is spent
    reset_on_done: bool = True

    # RICE components disabled: SIL neither mixes initial states nor uses RND.
    p: float = 0.0
    lam: float = 0.0
    alpha: float = 1e-4

    # --- Self-Imitation Learning specifics ---------------------------------- #
    sil_beta: float = DEFAULT_SIL_BETA
    sil_weight_clip: float = DEFAULT_SIL_WEIGHT_CLIP
    sil_buffer_size: int = DEFAULT_SIL_BUFFER_SIZE
    sil_grad_steps_per_iter: int = DEFAULT_SIL_GRAD_STEPS
    sil_batch_size: int = DEFAULT_SIL_BATCH_SIZE
    sil_advantage_threshold: float = 0.0       # "good experience" iff adv > threshold
    sil_ema_baseline: Optional[float] = None   # optional return baseline b (beta)
    sil_baseline_decay: float = 0.99
    sil_warmup_iterations: int = 1             # start SIL only once the buffer is non-empty
    sil_normalize_weight: bool = False

    # --- optimisation ------------------------------------------------------- #
    lr_factor: float = DEFAULT_LR_FACTOR
    learning_rate: Optional[float] = None      # overrides lr_factor * pretrain_lr
    policy_config: Optional[Any] = None        # PPOConfig
    eval_every: int = 0
    eval_episodes: int = DEFAULT_EVAL_EPISODES
    final_eval: bool = True
    deterministic_eval: bool = True
    curve_window: int = 1
    measure_baseline: bool = True

    # --- reproducibility / hw ---------------------------------------------- #
    n_seeds: int = 3
    seeds: Optional[Sequence[int]] = None
    seed: Optional[int] = None
    device: str = "auto"
    verbose: int = 1
    log_every: int = 1

    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    eval_env_kwargs: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def clone(self, **overrides: Any) -> "SILConfig":
        """Copy with ``overrides`` applied (None values are ignored)."""
        data = {k: v for k, v in self.__dict__.items() if k != "extra"}
        data["extra"] = dict(self.extra)
        data["env_kwargs"] = dict(self.env_kwargs)
        data["eval_env_kwargs"] = dict(self.eval_env_kwargs)
        for key, value in overrides.items():
            if value is not None:
                data[key] = value
        return SILConfig(**data)

    @classmethod
    def from_mapping(cls, mapping: Optional[Dict[str, Any]] = None, **overrides: Any) -> "SILConfig":
        """Build from a (YAML-parsed) mapping, tolerating common aliases."""
        cfg = cls()
        mapping = dict(mapping or {})
        aliases = {
            "beta": "p",
            "lambda": "lam",
            "lambda_": "lam",
            "coef": "lam",
            "K": "rollin_length",
            "length": "rollin_length",
            "num_iterations": "n_iterations",
            "n_episodes": "n_iterations",
            "sil_coef": "sil_beta",
            "beta_sil": "sil_beta",
            "self_imitation_beta": "sil_beta",
            "good_adv_threshold": "sil_advantage_threshold",
        }
        mapping = {aliases.get(k, k): v for k, v in mapping.items()}
        known = set(cls().__dict__.keys())
        extra = {k: v for k, v in mapping.items() if k not in known}
        clean = {k: v for k, v in mapping.items() if k in known}
        cfg = cfg.clone(**clean)
        cfg.extra.update(extra)
        if overrides:
            cfg = cfg.clone(**overrides)
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        data = dict(self.__dict__)
        data["policy_config"] = None if self.policy_config is None else "<PPOConfig>"
        return data

    # ------------------------------------------------------------------ #
    def seed_list(self) -> List[int]:
        if self.seeds:
            return [int(s) for s in self.seeds]
        seed = 0 if self.seed is None else int(self.seed)
        return [seed + i for i in range(max(1, int(self.n_seeds)))]

    def budget(self) -> Optional[float]:
        return None if self.total_env_steps is None else float(self.total_env_steps)

    def resolved_learning_rate(self, pretrain_lr: Optional[float] = None) -> float:
        if self.learning_rate is not None:
            return float(self.learning_rate)
        base = DEFAULT_PRETRAIN_LR if pretrain_lr is None else float(pretrain_lr)
        return float(base) * float(self.lr_factor)

    def policy_config_for(self, pretrain_lr: Optional[float] = None) -> Any:
        """Return a :class:`PPOConfig` with the lowered learning rate applied."""
        lr = self.resolved_learning_rate(pretrain_lr)
        if PPOConfig is None:  # pragma: no cover - ppo.py is a hard dependency
            raise ImportError("rice.algorithms.ppo.PPOConfig is required by SIL")
        if self.policy_config is None:
            return PPOConfig(learning_rate=lr)
        if hasattr(self.policy_config, "clone"):
            return self.policy_config.clone(learning_rate=lr)
        cfg = copy.deepcopy(self.policy_config)
        try:
            cfg.learning_rate = lr
        except Exception:  # pragma: no cover - defensive
            pass
        return cfg

    def to_refine_config(self, pretrain_lr: Optional[float] = None, **overrides: Any) -> Any:
        """Express the SIL baseline as a :class:`RefineConfig` (for reference)."""
        if RefineConfig is None:  # pragma: no cover - defensive
            return None
        return RefineConfig(
            p=0.0,
            lam=0.0,
            n_iterations=self.n_iterations,
            steps_per_iter=self.steps_per_iter,
            total_env_steps=self.total_env_steps,
            rollin_length=self.rollin_length,
            reset_on_done=self.reset_on_done,
            policy_config=self.policy_config_for(pretrain_lr),
            eval_every=self.eval_every,
            eval_episodes=self.eval_episodes,
            device=self.device,
            seed=self.seed,
            verbose=self.verbose,
            **overrides,
        )


# --------------------------------------------------------------------------- #
# Past-good-experience replay buffer
# --------------------------------------------------------------------------- #
class SILBuffer:
    """Replay buffer of *past good experiences* (Oh et al. 2018).

    Each entry is ``(obs, action, weight)`` where ``weight`` is the
    non-negative advantage ``(R_t - V(s_t))_+`` (optionally clipped at
    ``weight_clip``).  Only transitions passing the "good experience" test
    are stored, so sampling is uniform over good transitions.
    """

    def __init__(self, capacity: int = DEFAULT_SIL_BUFFER_SIZE, weight_clip: float = 1.0):
        self.capacity = max(1, int(capacity))
        self.weight_clip = float(weight_clip)
        self.obs: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.weights: List[float] = []
        self._cursor = 0
        self.n_added = 0

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.weights)

    def clear(self) -> None:
        self.obs, self.actions, self.weights = [], [], []
        self._cursor = 0

    def add_many(self, obs: np.ndarray, actions: np.ndarray, weights: np.ndarray) -> int:
        """Append the good transitions; returns how many were actually stored."""
        obs = np.asarray(obs, dtype=np.float32)
        actions = np.asarray(actions)
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if obs.ndim == 1:
            obs = obs[None, :]
        if actions.ndim == 1 and actions.shape[0] != obs.shape[0]:
            actions = actions[:, None]
        stored = 0
        for i in range(min(len(weights), obs.shape[0])):
            w = float(weights[i])
            if not np.isfinite(w):
                continue
            w = min(max(w, 0.0), self.weight_clip)
            if w <= 0.0:
                continue
            if len(self.obs) < self.capacity:
                self.obs.append(np.array(obs[i], dtype=np.float32, copy=True))
                self.actions.append(np.array(actions[i], copy=True))
                self.weights.append(w)
            else:  # ring buffer: overwrite the oldest entry
                idx = self._cursor % self.capacity
                self.obs[idx] = np.array(obs[i], dtype=np.float32, copy=True)
                self.actions[idx] = np.array(actions[i], copy=True)
                self.weights[idx] = w
                self._cursor += 1
            stored += 1
            self.n_added += 1
        return stored

    def sample(self, batch_size: int, rng: Optional[np.random.Generator] = None):
        """Uniformly sample ``batch_size`` good transitions."""
        n = len(self.obs)
        if n == 0:
            return None
        bs = int(min(max(1, batch_size), n))
        if rng is None:
            idx = np.random.randint(0, n, size=bs)
        else:
            idx = rng.integers(0, n, size=bs)
        idx = np.asarray(idx).reshape(-1)
        obs = np.stack([self.obs[i] for i in idx]).astype(np.float32)
        actions = np.stack([np.asarray(self.actions[i]).reshape(-1) for i in idx])
        weights = np.asarray([self.weights[i] for i in idx], dtype=np.float32)
        return obs, actions, weights

    def mean_weight(self) -> float:
        return float(np.mean(self.weights)) if self.weights else 0.0

    def state_dict(self) -> Dict[str, Any]:
        return {
            "capacity": self.capacity,
            "weight_clip": self.weight_clip,
            "obs": [o.tolist() for o in self.obs],
            "actions": [np.asarray(a).tolist() for a in self.actions],
            "weights": list(self.weights),
            "cursor": self._cursor,
            "n_added": self.n_added,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if not state:
            return
        self.capacity = int(state.get("capacity", self.capacity))
        self.weight_clip = float(state.get("weight_clip", self.weight_clip))
        self.obs = [np.asarray(o, dtype=np.float32) for o in state.get("obs", [])]
        self.actions = [np.asarray(a) for a in state.get("actions", [])]
        self.weights = [float(w) for w in state.get("weights", [])]
        self._cursor = int(state.get("cursor", 0))
        self.n_added = int(state.get("n_added", len(self.weights)))


# --------------------------------------------------------------------------- #
# Multi-seed summary (duck-types rice.algorithms.refine.RefineResult)
# --------------------------------------------------------------------------- #
@dataclass
class SILSummary:
    """Aggregate of several SIL refining runs (mean/std final reward + curves)."""

    task: str = "Hopper-v3"
    method: str = METHOD_NAME
    explanation: str = "none"
    seeds: List[int] = field(default_factory=list)
    results: List[Any] = field(default_factory=list)
    final_rewards: List[float] = field(default_factory=list)
    baseline_rewards: List[float] = field(default_factory=list)
    curves: List[np.ndarray] = field(default_factory=list)
    seconds: float = 0.0
    env_steps: int = 0
    config: Optional[Any] = None
    notes: List[str] = field(default_factory=list)
    error: Optional[str] = None

    # -- RefineResult-compatible surface ---------------------------------- #
    @property
    def final_reward(self) -> float:
        return float(np.mean(self.final_rewards)) if self.final_rewards else float("nan")

    @property
    def final_std(self) -> float:
        return float(np.std(self.final_rewards)) if self.final_rewards else 0.0

    @property
    def baseline_reward(self) -> float:
        return float(np.mean(self.baseline_rewards)) if self.baseline_rewards else float("nan")

    @property
    def baseline_std(self) -> float:
        return float(np.std(self.baseline_rewards)) if self.baseline_rewards else 0.0

    @property
    def improvement(self) -> float:
        if not self.final_rewards or not self.baseline_rewards:
            return float("nan")
        return self.final_reward - self.baseline_reward

    @property
    def iterations(self) -> int:
        return max((len(c) for c in self.curves), default=0)

    def mean_curve(self, window: int = 1) -> np.ndarray:
        window = max(1, int(window))
        if not self.curves:
            return np.zeros(0, dtype=np.float64)
        n = min(len(c) for c in self.curves)
        stack = np.stack([np.asarray(c, dtype=np.float64)[:n] for c in self.curves])
        mean = stack.mean(axis=0)
        if window > 1 and len(mean) >= window:
            kernel = np.ones(window) / float(window)
            mean = np.convolve(mean, kernel, mode="valid")
        return mean

    def curve_std(self, window: int = 1) -> np.ndarray:
        window = max(1, int(window))
        if not self.curves:
            return np.zeros(0, dtype=np.float64)
        n = min(len(c) for c in self.curves)
        stack = np.stack([np.asarray(c, dtype=np.float64)[:n] for c in self.curves])
        std = stack.std(axis=0)
        if window > 1 and len(std) >= window:
            kernel = np.ones(window) / float(window)
            std = np.convolve(std, kernel, mode="valid")
        return std

    refining_curve = mean_curve  # alias used by figure-producing code

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "method": self.method,
            "explanation": self.explanation,
            "seeds": list(self.seeds),
            "final_reward": self.final_reward,
            "final_std": self.final_std,
            "baseline_reward": self.baseline_reward,
            "baseline_std": self.baseline_std,
            "improvement": self.improvement,
            "env_steps": int(self.env_steps),
            "seconds": float(self.seconds),
            "n_seeds": len(self.results),
            "notes": list(self.notes),
            "error": self.error,
        }

    def __len__(self) -> int:
        return len(self.results)


# --------------------------------------------------------------------------- #
# Environment helpers (gym / gymnasium tolerant)
# --------------------------------------------------------------------------- #
def _env_reset(env: Any, seed: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
    try:
        out = env.reset(seed=seed) if seed is not None else env.reset()
    except TypeError:
        try:
            if seed is not None and hasattr(env, "seed"):
                env.seed(seed)
        except Exception:  # pragma: no cover
            pass
        out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], dict(out[1] or {})
    return out, {}


def _env_step(env: Any, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, float(reward), bool(terminated), bool(truncated), dict(info or {})
    if isinstance(out, tuple) and len(out) == 4:
        obs, reward, done, info = out
        return obs, float(reward), bool(done), False, dict(info or {})
    raise RuntimeError(f"unexpected env.step() return: {type(out)}")


def _env_episode_length(env: Any, default: int = 1000) -> int:
    for attr in ("max_episode_steps", "_max_episode_steps"):
        value = getattr(env, attr, None)
        if value:
            try:
                return int(value)
            except Exception:  # pragma: no cover
                pass
    spec = getattr(env, "spec", None)
    if spec is not None and getattr(spec, "max_episode_steps", None):
        return int(spec.max_episode_steps)
    inner = getattr(env, "env", None)
    depth = 0
    while inner is not None and depth < 8:
        value = getattr(inner, "_max_episode_steps", None) or getattr(inner, "max_episode_steps", None)
        if value:
            try:
                return int(value)
            except Exception:  # pragma: no cover
                pass
        inner = getattr(inner, "env", None)
        depth += 1
    return int(default)


def _obs_to_numpy(obs: Any) -> np.ndarray:
    try:
        return np.asarray(flatten_obs(obs), dtype=np.float32)
    except Exception:  # pragma: no cover - defensive
        return np.asarray(obs, dtype=np.float32).reshape(-1)


# --------------------------------------------------------------------------- #
# Main baseline runner
# --------------------------------------------------------------------------- #
class SILRefiner:
    """Self-Imitation Learning refining baseline (Oh et al. 2018).

    Parameters
    ----------
    env:
        Training environment (single, non-vectorised).  Built lazily from
        ``config.task`` when omitted.
    policy:
        Warm-start policy (:class:`~rice.algorithms.ppo.ActorCritic`, a
        Stable-Baselines3 model, or raw weights).  Built lazily when omitted.
    config:
        :class:`SILConfig`, a YAML mapping, or ``None`` for defaults.
    """

    def __init__(
        self,
        env: Any = None,
        policy: Any = None,
        config: Optional[Any] = None,
        evaluation_env: Any = None,
        state_manager: Any = None,
        rng: Any = None,
        task: Optional[str] = None,
        mask_network: Any = None,
        **kwargs: Any,
    ):
        if isinstance(config, dict):
            config = SILConfig.from_mapping(config, **kwargs)
        elif config is None:
            config = SILConfig(**kwargs) if kwargs else SILConfig()
        elif kwargs:
            config = config.clone(**kwargs)
        self.config: SILConfig = config
        if task:
            self.config.task = task

        self.env = env
        self.policy = policy
        self.mask_network = None  # SIL is explanation-free by construction
        self.evaluation_env = evaluation_env
        self.state_manager = state_manager
        self._rng = rng
        self._summary: Optional[SILSummary] = None
        self.sil_buffer = SILBuffer(
            capacity=self.config.sil_buffer_size,
            weight_clip=self.config.sil_weight_clip,
        )
        self.return_baseline: float = (
            float(self.config.sil_ema_baseline) if self.config.sil_ema_baseline is not None else 0.0
        )
        self._device = None

    # ------------------------------------------------------------------ #
    # Properties / utilities
    # ------------------------------------------------------------------ #
    @property
    def rng(self) -> np.random.Generator:
        if self._rng is None:
            seed = self.config.seed if self.config.seed is not None else self.config.seed_list()[0]
            self._rng = np.random.default_rng(seed)
        return self._rng

    @property
    def device(self):
        if self._device is None:
            if _TORCH_AVAILABLE and torch is not None and torch.cuda.is_available() and self.config.device != "cpu":
                self._device = torch.device("cuda")
            else:
                self._device = torch.device("cpu")
        return self._device

    def resolve_task(self) -> str:
        task = self.config.task or "Hopper-v3"
        if _env_mod is not None:
            resolver = getattr(_env_mod, "resolve_env_name", None)
            if callable(resolver):
                try:
                    return resolver(task)
                except Exception:  # pragma: no cover - keep raw name
                    return task
        return task

    # ------------------------------------------------------------------ #
    # Builders
    # ------------------------------------------------------------------ #
    def build_env(self, seed: Optional[int] = None):
        if self.env is not None:
            return self.env
        if make_env is None:  # pragma: no cover - defensive
            raise ImportError("rice.environments.make_env is required by the SIL baseline")
        kwargs = dict(self.config.env_kwargs)
        if seed is not None:
            kwargs.setdefault("seed", seed)
        self.env = make_env(self.resolve_task(), **kwargs)
        return self.env

    def build_eval_env(self, seed: Optional[int] = None):
        if self.evaluation_env is not None:
            return self.evaluation_env
        if make_env is None:  # pragma: no cover - defensive
            return self.build_env(seed)
        kwargs = dict(self.config.env_kwargs)
        kwargs.update(self.config.eval_env_kwargs)
        if seed is not None:
            kwargs.setdefault("seed", seed)
        self.evaluation_env = make_env(self.resolve_task(), **kwargs)
        return self.evaluation_env

    def build_policy(self, env: Any = None):
        if self.policy is not None and isinstance(self.policy, (ActorCritic,) if ActorCritic else ()):
            return self.policy
        env = env if env is not None else self.build_env()
        net_arch = tuple(self.config.net_arch) if self.config.net_arch else None
        if net_arch is None and callable(default_net_arch):
            try:
                net_arch = tuple(default_net_arch(self.resolve_task()))
            except Exception:  # pragma: no cover
                net_arch = None
        if ActorCritic is None:  # pragma: no cover - defensive
            raise ImportError("rice.algorithms.ppo.ActorCritic is required by the SIL baseline")
        policy = ActorCritic(
            observation_space=env.observation_space,
            action_space=env.action_space,
            net_arch=net_arch or (64, 64),
            device=self.config.device,
        )
        if self.policy is not None and callable(load_policy_weights):
            try:
                load_policy_weights(policy, self.policy, strict=False)
            except Exception as exc:  # pragma: no cover - defensive
                if self.config.verbose:
                    print(f"[SIL] could not load warm-start weights: {exc}")
        elif self.config.weights is not None and callable(load_policy_weights):
            try:
                load_policy_weights(policy, self.config.weights, strict=False)
            except Exception as exc:  # pragma: no cover - defensive
                if self.config.verbose:
                    print(f"[SIL] could not load weights from config: {exc}")
        self.policy = policy
        return policy

    # ------------------------------------------------------------------ #
    # SIL loss
    # ------------------------------------------------------------------ #
    def sil_loss(self, obs: np.ndarray, actions: np.ndarray, weights: np.ndarray):
        """Self-imitation loss of Oh et al. (2018), Eq. 4.

        ``L^sil = - mean_i [ (R_i - V(s_i))_+ * log pi(a_i | s_i) ]``

        The non-negative advantage ``(R_i - V(s_i))_+`` is the cached
        ``weights`` argument (already clipped at ``sil_weight_clip``); it is
        treated as a *constant* (no gradient flows through the weight), which
        is exactly how the original method defines the objective.
        """
        if not _TORCH_AVAILABLE or torch is None:  # pragma: no cover - CPU-only fallback
            return None
        policy = self.policy
        if policy is None:
            return None
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        act_t = torch.as_tensor(np.asarray(actions, dtype=np.float32), device=self.device)
        w_t = torch.as_tensor(np.asarray(weights, dtype=np.float32).reshape(-1), device=self.device)

        # Discrete action spaces store integer action indices.
        discrete = False
        try:
            import gym  # type: ignore

            discrete = isinstance(getattr(self.env, "action_space", None), gym.spaces.Discrete)
        except Exception:  # pragma: no cover
            try:
                import gymnasium as gym  # type: ignore

                discrete = isinstance(getattr(self.env, "action_space", None), gym.spaces.Discrete)
            except Exception:
                discrete = False
        if discrete:
            act_t = act_t.reshape(-1).long()

        try:
            log_prob = policy.log_prob_of(obs_t, act_t)
        except Exception:  # pragma: no cover - fallback to evaluate_actions
            _, log_prob, _ = policy.evaluate_actions(obs_t, act_t)

        loss = -(w_t.detach() * log_prob).mean()
        return loss

    def _sil_update(self, rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
        """Run ``sil_grad_steps_per_iter`` SIL gradient steps on the good-experience buffer."""
        stats = {"sil_loss": float("nan"), "sil_steps": 0.0, "sil_buffer": float(len(self.sil_buffer))}
        if not _TORCH_AVAILABLE or torch is None:  # pragma: no cover - nothing to do without torch
            return stats
        if len(self.sil_buffer) == 0:
            return stats
        ppo = getattr(self, "_ppo", None)
        if ppo is None or getattr(ppo, "optimizer", None) is None:
            return stats
        policy = self.policy
        policy.train()
        losses: List[float] = []
        for _ in range(max(1, int(self.config.sil_grad_steps_per_iter))):
            batch = self.sil_buffer.sample(
                self.config.sil_batch_size, rng=rng if rng is not None else self.rng
            )
            if batch is None:
                break
            obs, actions, weights = batch
            loss = self.sil_loss(obs, actions, weights)
            if loss is None:
                break
            ppo.optimizer.zero_grad()
            loss.backward()
            max_norm = float(getattr(getattr(ppo, "config", None), "max_grad_norm", 0.5) or 0.5)
            try:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm)
            except Exception:  # pragma: no cover
                pass
            ppo.optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        if losses:
            stats["sil_loss"] = float(np.mean(losses))
            stats["sil_steps"] = float(len(losses))
        stats["sil_buffer"] = float(len(self.sil_buffer))
        return stats

    # ------------------------------------------------------------------ #
    # Roll-out collection (Algorithm 2's T-step loop, minus RICE components)
    # ------------------------------------------------------------------ #
    def collect_iteration(self, policy, ppo, seed: Optional[int] = None, iteration: int = 1):
        """Collect one T-step batch from the *default* initial-state distribution."""
        env = self.build_env(seed)
        T = int(self.config.steps_per_iter or _env_episode_length(env, 1000))

        obs, _ = _env_reset(env, seed=None if seed is None else int(seed) + int(iteration))
        buffer = RolloutBuffer()
        steps = 0
        episode_return = 0.0
        episode_returns: List[float] = []
        episodes_finished = 0

        while steps < T:
            obs_arr = _obs_to_numpy(obs)
            action, value, log_prob = policy.act(obs_arr, deterministic=False)
            next_obs, reward, terminated, truncated, _info = _env_step(env, action)
            done = bool(terminated or truncated)
            buffer.add(
                obs=obs_arr,
                action=action,
                reward=float(reward),
                next_obs=_obs_to_numpy(next_obs),
                done=done,
                value=float(value),
                log_prob=float(log_prob),
            )
            episode_return += float(reward)
            steps += 1
            obs = next_obs
            if done:
                episode_returns.append(episode_return)
                episodes_finished += 1
                episode_return = 0.0
                if self.config.reset_on_done and steps < T:
                    obs, _ = _env_reset(env)
                else:
                    break
        if episode_return != 0.0:
            episode_returns.append(episode_return)

        info = {
            "steps": steps,
            "episode_returns": episode_returns,
            "episodes_finished": episodes_finished,
            "task_return": float(np.sum(episode_returns)) if episode_returns else float("nan"),
        }
        return buffer, info

    # ------------------------------------------------------------------ #
    # Single-seed refining run
    # ------------------------------------------------------------------ #
    def refine(self, seed: Optional[int] = None, n_iterations: Optional[int] = None, **overrides: Any):
        """Run the SIL refining loop for one seed; returns a ``RefineResult``-like object."""
        if overrides:
            self.config = self.config.clone(**overrides)
        if seed is not None:
            self.config.seed = int(seed)
        n_iterations = int(n_iterations or self.config.n_iterations)

        start = time.time()
        if callable(set_global_seeds):
            try:
                set_global_seeds(self.config.seed)
            except Exception:  # pragma: no cover
                pass

        env = self.build_env(self.config.seed)
        policy = self.build_policy(env)

        # --- baseline (pre-refining) reward --------------------------------- #
        baseline_reward = float("nan")
        if self.config.measure_baseline:
            baseline_reward = self.baseline_reward(seed=self.config.seed)

        # --- frozen reference for the "did SIL help?" audit ----------------- #
        self._initial_policy = copy.deepcopy(policy) if _TORCH_AVAILABLE else policy

        ppo = PPO(
            policy,
            config=self.config.policy_config_for(),
            device=self.config.device,
        )
        self._ppo = ppo

        self.sil_buffer = SILBuffer(
            capacity=self.config.sil_buffer_size,
            weight_clip=self.config.sil_weight_clip,
        )
        self.return_baseline = (
            float(self.config.sil_ema_baseline) if self.config.sil_ema_baseline is not None else 0.0
        )

        log: List[Dict[str, Any]] = []
        curves: List[float] = []
        total_steps = 0
        stopping_reason = "iterations"

        for iteration in range(1, n_iterations + 1):
            iter_start = time.time()
            buffer, collect_info = self.collect_iteration(policy, ppo, seed=self.config.seed, iteration=iteration)
            total_steps += int(collect_info["steps"])

            # ---- PPO update (shared implementation) ----------------------- #
            last_values = 0.0
            last_dones = None
            try:
                ppo_stats = ppo.update(buffer, last_values=last_values, last_dones=last_dones)
            except TypeError:  # pragma: no cover - older signature
                ppo_stats = ppo.update(buffer)

            # ---- harvest past good experiences --------------------------- #
            adv = np.asarray(getattr(buffer, "advantages", np.zeros(0, dtype=np.float32)), dtype=np.float64)
            obs_batch = np.asarray(buffer.obs, dtype=np.float32) if len(buffer) else np.zeros((0, 1))
            act_batch = np.asarray(buffer.actions) if len(buffer) else np.zeros((0, 1))
            if adv.size and obs_batch.shape[0] == adv.size:
                if self.config.sil_ema_baseline is not None:
                    # optional return baseline b: weight = (R_t - b)_+
                    rets = np.asarray(getattr(buffer, "returns", adv), dtype=np.float64)
                    weights = np.maximum(rets - float(self.return_baseline), 0.0)
                else:
                    weights = np.maximum(adv - float(self.config.sil_advantage_threshold), 0.0)
                if self.config.sil_normalize_weight and weights.size:
                    denom = float(np.max(weights)) or 1.0
                    weights = weights / denom
                self.sil_buffer.add_many(obs_batch, act_batch, weights)

            # ---- SIL gradient steps --------------------------------------- #
            sil_stats = {"sil_loss": float("nan"), "sil_steps": 0.0, "sil_buffer": float(len(self.sil_buffer))}
            if iteration > max(0, int(self.config.sil_warmup_iterations)) - 1:
                sil_stats = self._sil_update()

            # ---- bookkeeping ---------------------------------------------- #
            ep_returns = collect_info.get("episode_returns") or []
            if ep_returns:
                self.return_baseline = (
                    self.config.sil_baseline_decay * self.return_baseline
                    + (1.0 - self.config.sil_baseline_decay) * float(np.mean(ep_returns))
                )
                curves.append(float(np.mean(ep_returns)))
            else:
                curves.append(float(curves[-1]) if curves else float("nan"))

            record = {
                "iteration": iteration,
                "init_mode": "default",
                "rand_num": float(self.rng.uniform(0.0, 1.0)),
                "steps": int(collect_info["steps"]),
                "task_return": collect_info.get("task_return", float("nan")),
                "episode_returns": ep_returns,
                "policy_loss": float(ppo_stats.get("policy_loss", float("nan"))),
                "value_loss": float(ppo_stats.get("value_loss", float("nan"))),
                "entropy_loss": float(ppo_stats.get("entropy_loss", float("nan"))),
                "approx_kl": float(ppo_stats.get("approx_kl", float("nan"))),
                "sil_loss": sil_stats["sil_loss"],
                "sil_steps": sil_stats["sil_steps"],
                "sil_buffer": sil_stats["sil_buffer"],
                "seconds": time.time() - iter_start,
            }
            log.append(record)

            if self.config.verbose and (iteration % max(1, self.config.log_every) == 0):
                print(
                    f"[SIL:{self.resolve_task()}] iter {iteration:4d} | "
                    f"steps {record['steps']:5d} | ret {curves[-1]:10.2f} | "
                    f"pi_loss {record['policy_loss']:8.3f} | sil_loss {record['sil_loss']:8.4f} | "
                    f"sil_buf {int(record['sil_buffer']):7d}"
                )

            # ---- budget / evaluation hooks -------------------------------- #
            if self.config.total_env_steps is not None and total_steps >= float(self.config.total_env_steps):
                stopping_reason = "budget"
            if self.config.eval_every and iteration % int(self.config.eval_every) == 0:
                eval_info = self.evaluate(policy=policy, seed=self.config.seed, n_episodes=self.config.eval_episodes)
                if self.config.verbose:
                    print(f"[SIL] eval @ {iteration}: {eval_info.get('mean_return')}")
            if stopping_reason == "budget" and self.config.total_env_steps is not None:
                if total_steps >= float(self.config.total_env_steps):
                    break

        # ---- final evaluation --------------------------------------------- #
        final_eval = float("nan")
        if self.config.final_eval:
            final_eval = float(
                self.evaluate(
                    policy=policy,
                    seed=self.config.seed,
                    n_episodes=self.config.eval_episodes,
                ).get("mean_return", float("nan"))
            )

        seconds = time.time() - start
        result = self._make_result(
            task=self.resolve_task(),
            policy=policy,
            baseline_reward=baseline_reward,
            final_reward=final_eval,
            curves=curves,
            log=log,
            env_steps=total_steps,
            seconds=seconds,
            stopping_reason=stopping_reason,
        )
        self._policy_after = policy
        return result

    # Alias used by scripts / evaluation harness.
    train = refine

    # ------------------------------------------------------------------ #
    def _make_result(self, **payload: Any):
        """Build a :class:`RefineResult` when available, else a lightweight stand-in."""
        policy = payload.pop("policy", None)
        log = payload.pop("log", [])
        curves = list(payload.pop("curves", []))
        if RefineResult is not None:
            try:
                result = RefineResult(
                    policy=policy,
                    iterations=len(log),
                    env_steps=int(payload.get("env_steps", 0)),
                    seconds=float(payload.get("seconds", 0.0)),
                    episode_returns=[r.get("task_return") for r in log if r.get("task_return") is not None],
                    mean_episode_return=float(np.nanmean(curves)) if curves else float("nan"),
                    final_eval_reward=float(payload.get("final_reward", float("nan"))),
                    critical_fraction=0.0,
                    log=log,
                    stopping_reason=payload.get("stopping_reason", "iterations"),
                )
                # Attach baseline reward so comparison code can read it.
                try:
                    object.__setattr__(result, "baseline_reward", float(payload.get("baseline_reward", float("nan"))))
                except Exception:  # pragma: no cover
                    setattr(result, "baseline_reward", float(payload.get("baseline_reward", float("nan"))))
                return result
            except Exception:  # pragma: no cover - fall back below
                pass
        return _SILResult(
            task=payload.get("task", self.resolve_task()),
            curves=curves,
            final_reward=float(payload.get("final_reward", float("nan"))),
            baseline_reward=float(payload.get("baseline_reward", float("nan"))),
            env_steps=int(payload.get("env_steps", 0)),
            seconds=float(payload.get("seconds", 0.0)),
            log=log,
            stopping_reason=payload.get("stopping_reason", "iterations"),
        )

    # ------------------------------------------------------------------ #
    def evaluate(self, policy: Any = None, seed: Optional[int] = None, n_episodes: Optional[int] = None) -> Dict[str, float]:
        policy = policy or self.policy
        env = self.build_eval_env(seed)
        n = int(n_episodes or self.config.eval_episodes)
        if callable(evaluate_policy):
            try:
                return dict(
                    evaluate_policy(
                        env,
                        policy,
                        n_episodes=n,
                        seed=seed,
                        deterministic=self.config.deterministic_eval,
                    )
                )
            except TypeError:  # pragma: no cover - narrower signature
                return dict(evaluate_policy(env, policy, n_episodes=n))
        # Manual fallback (no refine.py available).
        returns: List[float] = []
        for episode in range(n):
            obs, _ = _env_reset(env, None if seed is None else int(seed) + episode)
            done = False
            ep_ret = 0.0
            while not done:
                obs_arr = _obs_to_numpy(obs)
                try:
                    action = policy.predict(obs_arr, deterministic=self.config.deterministic_eval)
                except AttributeError:
                    action = policy.act(obs_arr, deterministic=self.config.deterministic_eval)[0]
                obs, reward, terminated, truncated, _ = _env_step(env, action)
                ep_ret += float(reward)
                done = bool(terminated or truncated)
            returns.append(ep_ret)
        return {
            "mean_return": float(np.mean(returns)) if returns else float("nan"),
            "std_return": float(np.std(returns)) if returns else 0.0,
        }

    def baseline_reward(self, seed: Optional[int] = None) -> float:
        """Pre-refining (warm-start) reward measured on the evaluation env."""
        return float(self.evaluate(policy=self.policy, seed=seed, n_episodes=self.config.eval_episodes).get("mean_return", float("nan")))

    # ------------------------------------------------------------------ #
    # Multi-seed driver
    # ------------------------------------------------------------------ #
    def run(self, seeds: Optional[Iterable[int]] = None, **kwargs: Any) -> SILSummary:
        seeds = list(seeds) if seeds is not None else self.config.seed_list()
        summary = SILSummary(task=self.resolve_task(), config=self.config)
        total_seconds = 0.0
        total_steps = 0
        for seed in seeds:
            seed = int(seed)
            summary.seeds.append(seed)
            try:
                result = self.refine(seed=seed, **kwargs)
                summary.results.append(result)
                summary.final_rewards.append(float(getattr(result, "final_reward", float("nan"))))
                summary.baseline_rewards.append(float(getattr(result, "baseline_reward", float("nan"))))
                curve = getattr(result, "refining_curve", None)
                if callable(curve):
                    curve = curve(self.config.curve_window)
                summary.curves.append(np.asarray(curve, dtype=np.float64).reshape(-1))
                total_seconds += float(getattr(result, "seconds", 0.0))
                total_steps += int(getattr(result, "env_steps", 0))
            except Exception as exc:  # keep sweeps alive across seeds
                summary.notes.append(f"seed {seed} failed: {exc}")
                if summary.error is None:
                    summary.error = str(exc)
        summary.seconds = total_seconds
        summary.env_steps = total_steps
        self._summary = summary
        return summary

    run_seeds = run

    @property
    def summary(self) -> Optional[SILSummary]:
        return self._summary

    def describe(self) -> Dict[str, Any]:
        return {
            "method": METHOD_NAME,
            "task": self.resolve_task(),
            "n_iterations": self.config.n_iterations,
            "sil_beta": self.config.sil_beta,
            "sil_weight_clip": self.config.sil_weight_clip,
            "sil_buffer_size": self.config.sil_buffer_size,
            "learning_rate": self.config.resolved_learning_rate(),
            "lr_factor": self.config.lr_factor,
            "seeds": self.config.seed_list(),
        }

    def state_dict(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "config": self.config.to_dict(),
            "sil_buffer": self.sil_buffer.state_dict(),
            "return_baseline": self.return_baseline,
        }
        if self.policy is not None and hasattr(self.policy, "state_dict"):
            try:
                state["policy"] = self.policy.state_dict()
            except Exception:  # pragma: no cover
                pass
        return state

    def save(self, path: str) -> str:
        import pickle

        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        state = self.state_dict()
        if _TORCH_AVAILABLE:
            try:
                torch.save(state, path)
                return path
            except Exception:  # pragma: no cover
                pass
        with open(path, "wb") as handle:
            pickle.dump(state, handle)
        return path


@dataclass
class _SILResult:
    """Lightweight ``RefineResult`` stand-in used when ``refine.py`` is unavailable."""

    task: str = "Hopper-v3"
    method: str = METHOD_NAME
    curves: List[float] = field(default_factory=list)
    final_reward: float = float("nan")
    baseline_reward: float = float("nan")
    env_steps: int = 0
    seconds: float = 0.0
    log: List[Dict[str, Any]] = field(default_factory=list)
    stopping_reason: str = "iterations"

    @property
    def final_eval_reward(self) -> float:
        return self.final_reward

    @property
    def improvement(self) -> float:
        return self.final_reward - self.baseline_reward

    def refining_curve(self, window: int = 1) -> np.ndarray:
        curve = np.asarray(self.curves, dtype=np.float64)
        if window > 1 and curve.size >= window:
            kernel = np.ones(int(window)) / float(window)
            curve = np.convolve(curve, kernel, mode="valid")
        return curve

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "method": self.method,
            "final_reward": self.final_reward,
            "baseline_reward": self.baseline_reward,
            "improvement": self.improvement,
            "env_steps": self.env_steps,
            "seconds": self.seconds,
        }


# --------------------------------------------------------------------------- #
# Factories / functional entry points
# --------------------------------------------------------------------------- #
def make_sil_refiner(env: Any = None, policy: Any = None, config: Optional[Any] = None, **kwargs: Any) -> SILRefiner:
    """Factory mirroring :func:`rice.algorithms.refine.make_refiner`."""
    return SILRefiner(env=env, policy=policy, config=config, **kwargs)


def self_imitation_refine(
    env: Any = None,
    policy: Any = None,
    mask_network: Any = None,
    config: Optional[Any] = None,
    seeds: Optional[Iterable[int]] = None,
    n_iterations: Optional[int] = None,
    evaluation_env: Any = None,
    state_manager: Any = None,
    rng: Any = None,
    task: Optional[str] = None,
    **kwargs: Any,
):
    """Functional entry point: refine with SIL.

    Returns a single ``RefineResult`` when exactly one seed is requested and a
    :class:`SILSummary` otherwise (mirroring the other RICE baselines).
    """
    refiner = SILRefiner(
        env=env,
        policy=policy,
        config=config,
        evaluation_env=evaluation_env,
        state_manager=state_manager,
        rng=rng,
        task=task,
        **kwargs,
    )
    seed_list = [int(s) for s in seeds] if seeds is not None else None
    if seed_list is not None and len(seed_list) == 1:
        return refiner.refine(seed=seed_list[0], n_iterations=n_iterations)
    if seed_list is None and refiner.config.n_seeds == 1:
        return refiner.refine(seed=refiner.config.seed_list()[0], n_iterations=n_iterations)
    return refiner.run(seeds=seed_list, n_iterations=n_iterations)


# Aliases expected by the baseline registry / evaluation harness.
sil_baseline = self_imitation_refine
self_imitation_baseline = self_imitation_refine
make_self_imitation_refiner = make_sil_refiner
SILRunner = SILRefiner
SILEvaluator = SILRefiner
SelfImitationLearningRefiner = SILRefiner
SelfImitationLearningConfig = SILConfig


__all__ = [
    "METHOD_NAME",
    "METHOD_ALIASES",
    "DEFAULT_TABLE5_REFERENCE",
    "SILConfig",
    "SILBuffer",
    "SILRefiner",
    "SILRunner",
    "SILEvaluator",
    "SILSummary",
    "SelfImitationLearningConfig",
    "SelfImitationLearningRefiner",
    "make_sil_refiner",
    "make_self_imitation_refiner",
    "self_imitation_refine",
    "self_imitation_baseline",
    "sil_baseline",
]
