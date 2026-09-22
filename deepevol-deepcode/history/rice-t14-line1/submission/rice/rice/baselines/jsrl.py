"""Jump-Start Reinforcement Learning (JSRL) baseline for RICE.

Paper reference (RICE, ICML 2024, Section 4.1 "Baseline Refining Methods"):

    "The third baseline is Jump-Start Reinforcement Learning (referred to as
     'JSRL') (Uchendu et al., 2023). JSRL introduces a guided policy pi_g to set
     up a curriculum to train an exploration policy pi_e. Through initializing
     pi_e = pi_g, we can transform JSRL to be a refining method that can further
     improve the performance of the guided policy."

and Appendix C.1: "Regarding Jump-Start Reinforcement Learning, we use the
implementation from https://github.com/steventango/jumpstart-rl."

Faithful realization of JSRL as a *refining* baseline
-----------------------------------------------------
* ``pi_g`` is the frozen pre-trained (warm-start) policy.  This is the only
  usage consistent with the paper's sentence "through initializing
  pi_e = pi_g, we can transform JSRL to be a refining method".
* ``pi_e`` is a fresh policy whose weights are initialized to ``pi_g``.
* Curriculum: for every episode a horizon ``h`` is sampled uniformly from
  ``{0, ..., H_i}``; ``pi_g`` acts for the first ``h`` steps and ``pi_e`` for
  the remaining steps.  ``H_i`` is annealed from the full episode length down
  to 0 over the course of refining (Uchendu et al. 2023, "we reduce H over the
  course of training" -- guidance is removed progressively).
* ``pi_e`` is optimized with the *shared* clipped-surrogate PPO update from
  :mod:`rice.algorithms.ppo` (the same update used by RICE and by every other
  baseline), so the comparison is apples-to-apples.  Guided steps may
  optionally be included in the PPO batch; their actions are re-scored under
  ``pi_e`` and the PPO ratio clipping then acts as the (importance-weighted)
  imitation term of JSRL.  When ``train_on_guided_steps=False`` only the
  ``pi_e``-controlled steps enter the batch, which is the strictly on-policy
  variant.

The runner deliberately returns a :class:`rice.algorithms.refine.RefineResult`
(duck-typed) so that :mod:`rice.evaluation.refining_eval` and the
hyper-parameter sweeps can consume JSRL side by side with RICE, StateMask-R and
PPO fine-tuning.

Nothing here inspects the target agent's internals (black-box assumption): the
guide is simply an ``obs -> action`` callable produced by
``rice.algorithms.ppo.make_target_policy_callable``.
"""

from __future__ import annotations

import copy
import importlib
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Tolerant imports (the repository can be launched from several sys.path roots)
# --------------------------------------------------------------------------- #


def _import_first(candidates: Sequence[str]) -> Any:
    """Import the first importable module in ``candidates``."""
    last_error: Optional[Exception] = None
    for name in candidates:
        try:
            return importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - defensive
            last_error = exc
    raise ImportError(f"Could not import any of {list(candidates)}: {last_error}")


def _get(module: Any, *names: str, default: Any = None) -> Any:
    """Return the first existing attribute of ``module`` among ``names``."""
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
    return default


try:  # pragma: no cover - import layout dependent
    from ..algorithms import ppo as _ppo_mod
except Exception:  # pragma: no cover
    try:
        from rice.algorithms import ppo as _ppo_mod  # type: ignore
    except Exception:  # pragma: no cover
        _ppo_mod = _import_first(("rice.rice.algorithms.ppo", "algorithms.ppo"))

try:  # pragma: no cover
    from ..algorithms import refine as _refine_mod
except Exception:  # pragma: no cover
    try:
        from rice.algorithms import refine as _refine_mod  # type: ignore
    except Exception:  # pragma: no cover
        _refine_mod = _import_first(("rice.rice.algorithms.refine", "algorithms.refine"))

PPO = _get(_ppo_mod, "PPO")
PPOConfig = _get(_ppo_mod, "PPOConfig")
ActorCritic = _get(_ppo_mod, "ActorCritic")
RolloutBuffer = _get(_ppo_mod, "RolloutBuffer")
flatten_obs = _get(_ppo_mod, "flatten_obs", default=lambda x: np.asarray(x, dtype=np.float32).ravel())
make_target_policy_callable = _get(_ppo_mod, "make_target_policy_callable")

RefineConfig = _get(_refine_mod, "RefineConfig")
RefineResult = _get(_refine_mod, "RefineResult")
refine_policy = _get(_refine_mod, "refine_policy")
evaluate_policy = _get(_refine_mod, "evaluate_policy")
load_policy_weights = _get(_refine_mod, "load_policy_weights")


def _make_env(*args: Any, **kwargs: Any) -> Any:
    """Lazily build an environment through the RICE registry."""
    try:
        from ..environments import make_env, default_net_arch
    except Exception:  # pragma: no cover
        from rice.environments import make_env, default_net_arch  # type: ignore
    return make_env(*args, **kwargs), default_net_arch


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

METHOD_NAME = "jsrl"
METHOD_ALIASES: Tuple[str, ...] = (
    "jsrl",
    "jump-start-rl",
    "jumpstart",
    "jump_start_rl",
    "jump_start",
)
DEFAULT_LR_FACTOR = 1.0  # the paper does not lower pi_e's LR for JSRL
DEFAULT_PRETRAIN_LR = 3e-4  # Stable-Baselines3 PPO default (paper does not state it)
DEFAULT_N_ITERATIONS = 100
DEFAULT_EVAL_EPISODES = 5

HORIZON_MODES: Tuple[str, ...] = ("linear", "step", "exponential", "constant")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class JSRLConfig:
    """Hyper-parameters of the JSRL refining baseline.

    The paper does not specify JSRL's schedule, batch size or learning rate;
    the defaults below follow Uchendu et al. (2023) (annealed uniform horizon,
    steep linear decay of the guidance horizon) and the Stable-Baselines3 PPO
    defaults used everywhere else in this code base.  They are documented as
    deviations in the README.
    """

    task: str = "Hopper-v3"
    method: str = METHOD_NAME
    explanation: str = "none"  # JSRL does not use an explanation module

    # ---- curriculum over the guided-horizon ----
    initial_horizon: Optional[int] = None  # None -> full episode length
    final_horizon: int = 0
    horizon_mode: str = "linear"  # one of HORIZON_MODES
    horizon_decay_fraction: float = 1.0  # fraction of iterations used to anneal
    sample_horizon_uniform: bool = True  # h ~ U{0, ..., H_i} (JSRL default)
    min_guidance_iterations: int = 0  # keep H at initial for this many iterations

    # ---- optimization ----
    n_iterations: int = DEFAULT_N_ITERATIONS
    steps_per_iter: Optional[int] = None
    total_env_steps: Optional[float] = None
    train_on_guided_steps: bool = True  # include pi_g steps in the PPO batch
    lr_factor: float = DEFAULT_LR_FACTOR
    learning_rate: Optional[float] = None
    lr_schedule: str = "constant"  # "constant" | "linear"
    reset_on_done: bool = True
    handle_timeout_termination: bool = True

    # ---- evaluation ----
    eval_every: int = 0  # 0 -> only the final evaluation
    eval_episodes: int = DEFAULT_EVAL_EPISODES
    final_eval: bool = True
    deterministic_eval: bool = True
    curve_window: int = 1
    measure_baseline: bool = True

    # ---- bookkeeping ----
    n_seeds: int = 3
    seeds: Optional[Sequence[int]] = None
    seed: Optional[int] = None
    device: str = "auto"
    verbose: int = 1
    log_every: int = 1

    weights: Optional[Any] = None
    net_arch: Optional[Sequence[int]] = None
    policy_config: Optional[Any] = None
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    eval_env_kwargs: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def clone(self, **overrides: Any) -> "JSRLConfig":
        data = {k: (list(v) if isinstance(v, list) else v) for k, v in self.__dict__.items()}
        for key, value in overrides.items():
            if key not in data and key not in ("task", "method"):
                data[key] = value
                continue
            data[key] = value
        return JSRLConfig(**{k: v for k, v in data.items() if k in JSRLConfig.__dataclass_fields__})

    @classmethod
    def from_mapping(cls, mapping: Optional[Dict[str, Any]] = None, **overrides: Any) -> "JSRLConfig":
        """Build from a (YAML-derived) mapping with alias handling."""
        cfg = cls()
        if mapping:
            aliases = {
                "num_iterations": "n_iterations",
                "iterations": "n_iterations",
                "n_episodes": "n_iterations",
                "K": "rollin_length",
                "length": "rollin_length",
                "lower_lr_factor": "lr_factor",
                "lr": "learning_rate",
                "H": "initial_horizon",
                "horizon": "initial_horizon",
                "beta": "explanation",
                "lambda": "extra",
            }
            data: Dict[str, Any] = {}
            extra: Dict[str, Any] = dict(cfg.extra)
            for key, value in dict(mapping).items():
                key = str(key)
                if key in cls.__dataclass_fields__:
                    data[key] = value
                elif key in aliases:
                    target = aliases[key]
                    if target == "extra":
                        extra[key] = value
                    else:
                        data[target] = value
                else:
                    extra[key] = value
            data["extra"] = extra
            cfg = cfg.clone(**data)
        if overrides:
            cfg = cfg.clone(**overrides)
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    # ------------------------------------------------------------------ #
    def seed_list(self) -> List[Optional[int]]:
        if self.seeds:
            return list(self.seeds)
        if self.seed is not None:
            return [int(self.seed)]
        return [None] * int(self.n_seeds)

    def budget(self) -> Optional[float]:
        return self.total_env_steps

    def resolved_learning_rate(self, pretrain_lr: Optional[float] = None) -> float:
        if self.learning_rate is not None:
            return float(self.learning_rate)
        base = float(pretrain_lr if pretrain_lr is not None else DEFAULT_PRETRAIN_LR)
        return base * float(self.lr_factor)

    def policy_config_for(self, pretrain_lr: Optional[float] = None) -> Any:
        if PPOConfig is None:  # pragma: no cover
            return self.policy_config
        lr = self.resolved_learning_rate(pretrain_lr)
        base = self.policy_config if self.policy_config is not None else PPOConfig()
        try:
            base = base.clone(learning_rate=lr)
        except Exception:  # pragma: no cover - defensive
            try:
                base.learning_rate = lr
            except Exception:
                pass
        if self.net_arch:
            try:
                base = base.clone(net_arch=tuple(self.net_arch))
            except Exception:  # pragma: no cover - defensive
                pass
        return base

    def horizon_at(self, iteration: int, total_iterations: Optional[int] = None) -> int:
        """Guided horizon ``H_i`` for outer iteration ``iteration`` (0-based)."""
        total = int(total_iterations or self.n_iterations)
        initial = self.initial_horizon
        if initial is None:
            initial = 1000  # resolved to the episode length at runtime
        initial = int(initial)
        final = int(self.final_horizon)
        start = int(max(0, self.min_guidance_iterations))
        if iteration <= start:
            return initial
        span = max(1, int(round(self.horizon_decay_fraction * total)) - start)
        progress = float(iteration - start) / float(span)
        progress = min(1.0, max(0.0, progress))
        mode = (self.horizon_mode or "linear").lower()
        if mode == "linear":
            value = initial + (final - initial) * progress
        elif mode == "step":
            steps = max(1, span // 4)
            value = initial + (final - initial) * (int(progress * span) // steps) / max(1, (span // steps))
        elif mode == "exponential":
            value = final + (initial - final) * (0.5 ** (5.0 * progress))
        elif mode == "constant":
            value = initial
        else:  # pragma: no cover - unknown mode falls back to linear
            value = initial + (final - initial) * progress
        return int(max(0, min(initial, round(value))))


# --------------------------------------------------------------------------- #
# Multi-seed aggregation (duck-types RefineResult)
# --------------------------------------------------------------------------- #


@dataclass
class JSRLSummary:
    """Aggregate of several JSRL refining runs (mirrors ``RefineResult`` API)."""

    method: str = METHOD_NAME
    task: str = ""
    explanation: str = "none"
    seeds: List[Optional[int]] = field(default_factory=list)
    results: List[Any] = field(default_factory=list)
    final_rewards: List[float] = field(default_factory=list)
    baseline_rewards: List[float] = field(default_factory=list)
    curves: List[np.ndarray] = field(default_factory=list)
    seconds: float = 0.0
    env_steps: int = 0
    config: Optional[JSRLConfig] = None
    notes: List[str] = field(default_factory=list)
    error: Optional[str] = None

    # -- RefineResult-compatible surface ------------------------------------ #
    @property
    def final_reward(self) -> float:
        return float(np.mean(self.final_rewards)) if self.final_rewards else float("nan")

    @property
    def final_std(self) -> float:
        return float(np.std(self.final_rewards)) if len(self.final_rewards) > 1 else 0.0

    @property
    def baseline_reward(self) -> float:
        return float(np.mean(self.baseline_rewards)) if self.baseline_rewards else float("nan")

    @property
    def baseline_std(self) -> float:
        return float(np.std(self.baseline_rewards)) if len(self.baseline_rewards) > 1 else 0.0

    @property
    def improvement(self) -> float:
        if not self.final_rewards or not self.baseline_rewards:
            return float("nan")
        return self.final_reward - self.baseline_reward

    @property
    def iterations(self) -> int:
        total = 0
        for res in self.results:
            total += int(getattr(res, "iterations", 0) or 0)
        return total

    def mean_curve(self, window: int = 1) -> np.ndarray:
        if not self.curves:
            return np.zeros(0, dtype=np.float32)
        length = min(len(c) for c in self.curves)
        if length == 0:
            return np.zeros(0, dtype=np.float32)
        stacked = np.stack([np.asarray(c[:length], dtype=np.float32) for c in self.curves], axis=0)
        mean = stacked.mean(axis=0)
        if window and int(window) > 1:
            mean = _moving_average(mean, int(window))
        return mean

    def curve_std(self, window: int = 1) -> np.ndarray:
        if len(self.curves) < 2:
            return np.zeros_like(self.mean_curve(window))
        length = min(len(c) for c in self.curves)
        stacked = np.stack([np.asarray(c[:length], dtype=np.float32) for c in self.curves], axis=0)
        return stacked.std(axis=0)

    def refining_curve(self, window: int = 1) -> np.ndarray:
        """Alias kept for RefineResult compatibility."""
        return self.mean_curve(window)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "task": self.task,
            "explanation": self.explanation,
            "seeds": list(self.seeds),
            "final_rewards": list(self.final_rewards),
            "baseline_rewards": list(self.baseline_rewards),
            "final_reward": self.final_reward,
            "final_std": self.final_std,
            "baseline_reward": self.baseline_reward,
            "baseline_std": self.baseline_std,
            "improvement": self.improvement,
            "seconds": self.seconds,
            "env_steps": self.env_steps,
            "notes": list(self.notes),
            "error": self.error,
        }

    def __len__(self) -> int:
        return len(self.results)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if window <= 1 or values.size == 0:
        return values.astype(np.float32)
    window = int(min(window, values.size))
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="valid").astype(np.float32)


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
        except Exception:
            pass
        out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        obs, info = out
        return obs, (info if isinstance(info, dict) else {})
    return out, {}


def _env_step(env: Any, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
    out = env.step(action)
    if isinstance(out, tuple):
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            return obs, float(reward), bool(terminated), bool(truncated), (info or {})
        if len(out) == 4:
            obs, reward, done, info = out
            return obs, float(reward), bool(done), False, (info or {})
    raise ValueError(f"Unsupported env.step return: {type(out)}")


def _scalar(value: Any) -> float:
    arr = np.asarray(value, dtype=np.float64)
    return float(arr.ravel()[0]) if arr.size else 0.0


def _env_episode_length(env: Any, default: int = 1000) -> int:
    for attr in ("max_episode_steps", "_max_episode_steps", "horizon"):
        value = getattr(env, attr, None)
        if isinstance(value, (int, np.integer)) and int(value) > 0:
            return int(value)
    spec = getattr(env, "spec", None)
    value = getattr(spec, "max_episode_steps", None) if spec is not None else None
    if isinstance(value, (int, np.integer)) and int(value) > 0:
        return int(value)
    inner = getattr(env, "env", None)
    if inner is not None and inner is not env:
        return _env_episode_length(inner, default=default)
    return int(default)


# --------------------------------------------------------------------------- #
# JSRL runner
# --------------------------------------------------------------------------- #


class JSRLRefiner:
    """Jump-Start RL refining baseline (Uchendu et al. 2023) as used in §4.1.

    ``pi_g`` (guide) is the frozen pre-trained policy; ``pi_e`` (exploration
    policy) starts from the same weights and is optimized with the shared PPO
    clipped surrogate.  Episodes are segmented at a horizon ``h ~ U{0..H_i}``
    with ``H_i`` annealed from the full episode length to ``final_horizon``.
    """

    def __init__(
        self,
        env: Any = None,
        policy: Any = None,
        config: Optional[JSRLConfig] = None,
        evaluation_env: Any = None,
        state_manager: Any = None,
        rng: Any = None,
        task: Optional[str] = None,
        mask_network: Any = None,  # accepted for a uniform baseline dispatch
        **kwargs: Any,
    ) -> None:
        if config is None:
            config = JSRLConfig.from_mapping(kwargs.pop("config_dict", None), **kwargs)
        elif kwargs:
            config = config.clone(**{k: v for k, v in kwargs.items() if k in JSRLConfig.__dataclass_fields__})
        self.config = config
        self.env = env
        self.policy = policy
        self.evaluation_env = evaluation_env
        self.state_manager = state_manager
        self.task = task or config.task
        self.mask_network = mask_network
        self.rng = rng if rng is not None else np.random.default_rng(config.seed)
        self._summary: Optional[JSRLSummary] = None
        self._guide_callable: Optional[Callable[[Any], np.ndarray]] = None
        self._guide_action = 0

    # ------------------------------------------------------------------ #
    # construction helpers
    # ------------------------------------------------------------------ #
    def resolve_task(self) -> str:
        if self.task:
            return str(self.task)
        return str(self.config.task)

    def build_env(self, seed: Optional[int] = None) -> Any:
        if self.env is not None:
            return self.env
        env, _ = _make_env(self.resolve_task(), seed=seed, **dict(self.config.env_kwargs))
        self.env = env
        return env

    def build_eval_env(self, seed: Optional[int] = None) -> Any:
        if self.evaluation_env is not None:
            return self.evaluation_env
        if not self.config.final_eval:
            return self.build_env(seed=seed)
        env, _ = _make_env(self.resolve_task(), seed=seed, **dict(self.config.eval_env_kwargs))
        self.evaluation_env = env
        return env

    def build_policy(self, env: Any = None) -> Any:
        if self.policy is None:
            env = env if env is not None else self.build_env()
            obs_dim = int(np.prod(getattr(env.observation_space, "shape", (1,))))
            if ActorCritic is None:  # pragma: no cover
                raise ImportError("rice.algorithms.ppo.ActorCritic is required for JSRL.")
            arch = tuple(self.config.net_arch) if self.config.net_arch else (64, 64)
            self.policy = ActorCritic(
                observation_space=env.observation_space,
                action_space=env.action_space,
                net_arch=arch,
                obs_dim=obs_dim,
                device=self.config.device,
            )
        if self.config.weights is not None:
            try:
                self.policy = load_policy_weights(self.policy, self.config.weights)  # type: ignore[misc]
            except Exception as exc:  # pragma: no cover - defensive
                if self.config.verbose:
                    print(f"[jsrl] warm-start weight loading failed: {exc}")
        return self.policy

    def _build_guide(self) -> Callable[[Any], np.ndarray]:
        """Freeze the warm-start policy as the JSRL guided policy ``pi_g``."""
        if self._guide_callable is not None:
            return self._guide_callable
        policy = self.build_policy(self.env)
        # Deep copy -> the guide never changes while pi_e is trained.
        try:
            guide = copy.deepcopy(policy)
        except Exception:  # pragma: no cover - defensive
            guide = policy
        try:
            if hasattr(guide, "eval"):
                guide.eval()
        except Exception:
            pass
        for param in getattr(guide, "parameters", lambda: [])():
            try:
                param.requires_grad_(False)
            except Exception:
                pass
        if make_target_policy_callable is not None:
            self._guide_callable = make_target_policy_callable(guide)
        else:  # pragma: no cover - fallback
            def _call(obs: Any) -> np.ndarray:
                return np.asarray(guide.predict(obs), dtype=np.float32)

            self._guide_callable = _call
        return self._guide_callable

    # ------------------------------------------------------------------ #
    # single-seed refining
    # ------------------------------------------------------------------ #
    def refine(
        self,
        seed: Optional[int] = None,
        n_iterations: Optional[int] = None,
        **overrides: Any,
    ) -> Any:
        """Run JSRL refining.  Returns a ``RefineResult``-compatible object."""
        cfg = self.config.clone(**overrides) if overrides else self.config
        if seed is not None:
            cfg = cfg.clone(seed=seed)
        n_iterations = int(n_iterations or cfg.n_iterations)

        env = self.build_env(seed=cfg.seed)
        policy = self.build_policy(env)
        guide = self._build_guide()

        episode_length = _env_episode_length(env, default=1000)
        initial_horizon = int(cfg.initial_horizon) if cfg.initial_horizon is not None else episode_length

        steps_per_iter = cfg.steps_per_iter
        if not steps_per_iter:
            steps_per_iter = int(cfg.total_env_steps / n_iterations) if cfg.total_env_steps else episode_length

        policy_config = cfg.policy_config_for()
        optimizer = PPO(policy, config=policy_config, device=cfg.device) if PPO is not None else None
        if optimizer is None:  # pragma: no cover
            raise ImportError("rice.algorithms.ppo.PPO is required for JSRL.")

        rng = np.random.default_rng(cfg.seed if cfg.seed is not None else None)

        log: List[Dict[str, Any]] = []
        curve: List[float] = []
        episode_returns: List[float] = []
        env_steps = 0
        start_time = time.time()

        # ---- baseline (pre-refining) reward ----
        baseline_reward = float("nan")
        eval_env = self.build_eval_env(seed=cfg.seed)
        if cfg.measure_baseline and evaluate_policy is not None:
            try:
                base_stats = evaluate_policy(
                    eval_env, policy, n_episodes=cfg.eval_episodes,
                    seed=cfg.seed, deterministic=cfg.deterministic_eval,
                )
                baseline_reward = float(base_stats.get("mean_return", float("nan")))
            except Exception as exc:  # pragma: no cover - defensive
                if cfg.verbose:
                    print(f"[jsrl] baseline evaluation failed: {exc}")
        # the guide is pi_g == the warm-start policy, so its reward is the baseline

        for iteration in range(n_iterations):
            horizon = cfg.horizon_at(iteration, n_iterations)
            buffer = RolloutBuffer() if RolloutBuffer is not None else None
            if buffer is None:  # pragma: no cover
                raise ImportError("rice.algorithms.ppo.RolloutBuffer is required for JSRL.")

            iter_steps = 0
            iter_guided = 0
            iter_returns: List[float] = []

            while iter_steps < steps_per_iter:
                obs, _ = _env_reset(env, seed=None)
                t = 0
                ep_return = 0.0
                ep_guided = 0
                done = False
                truncated = False
                # one horizon draw per episode (JSRL "h ~ Uniform{0, H_i}")
                if cfg.sample_horizon_uniform and horizon > 0:
                    h = int(rng.integers(0, horizon + 1))
                else:
                    h = int(horizon)

                while not (done or truncated) and t < episode_length and iter_steps < steps_per_iter:
                    guided = t < h
                    if guided:
                        action = np.asarray(guide(obs), dtype=np.float32).ravel()
                        guided_for_rollout = True
                    else:
                        action, _value, _logprob = _policy_act(policy, obs, deterministic=False, rng=rng)
                        guided_for_rollout = False

                    next_obs, reward, done, truncated, _info = _env_step(env, action)
                    ep_return += float(reward)
                    env_steps += 1
                    iter_steps += 1
                    t += 1

                    if guided:
                        ep_guided += 1
                        iter_guided += 1
                    # Guided steps may still enter the batch: PPO re-scores the
                    # taken action under pi_e, and ratio clipping turns the
                    # surrogate into JSRL's importance-weighted imitation term.
                    if (not guided) or cfg.train_on_guided_steps:
                        value, log_prob = _policy_score(policy, obs, action)
                        buffer.add(obs, action, float(reward), next_obs,
                                   bool(done or truncated), value=value, log_prob=log_prob)

                    obs = next_obs

                if ep_return or t:
                    iter_returns.append(ep_return)
                    episode_returns.append(ep_return)

            # ---- PPO update of pi_e (shared update rule) ----
            stats: Dict[str, float] = {}
            if len(buffer) > 0:
                try:
                    stats = optimizer.update(buffer) or {}
                except Exception as exc:  # pragma: no cover - defensive
                    if cfg.verbose:
                        print(f"[jsrl] PPO update failed at iter {iteration}: {exc}")

            mean_return = float(np.mean(iter_returns)) if iter_returns else float("nan")
            curve.append(mean_return if np.isfinite(mean_return) else (curve[-1] if curve else 0.0))
            log.append(
                {
                    "iteration": iteration,
                    "horizon": int(horizon),
                    "guided_steps": int(iter_guided),
                    "steps": int(iter_steps),
                    "guided_fraction": float(iter_guided / max(1, iter_steps)),
                    "mean_episode_return": mean_return,
                    "episodes": len(iter_returns),
                    "env_steps": int(env_steps),
                    **{k: float(v) for k, v in stats.items() if isinstance(v, (int, float))},
                }
            )
            if cfg.verbose and (iteration % max(1, cfg.log_every) == 0):
                print(
                    f"[jsrl] iter {iteration + 1}/{n_iterations} H={horizon} "
                    f"guided={iter_guided}/{iter_steps} return={mean_return:.2f} "
                    f"steps={env_steps}"
                )

        seconds = time.time() - start_time

        # ---- final evaluation ----
        final_reward = float("nan")
        if cfg.final_eval and evaluate_policy is not None:
            try:
                stats = evaluate_policy(
                    eval_env, policy, n_episodes=cfg.eval_episodes,
                    seed=cfg.seed, deterministic=cfg.deterministic_eval,
                )
                final_reward = float(stats.get("mean_return", float("nan")))
            except Exception as exc:  # pragma: no cover - defensive
                if cfg.verbose:
                    print(f"[jsrl] final evaluation failed: {exc}")

        return _make_result(
            policy=policy,
            iterations=n_iterations,
            env_steps=env_steps,
            seconds=seconds,
            episode_returns=episode_returns,
            final_eval_reward=final_reward if np.isfinite(final_reward) else (curve[-1] if curve else float("nan")),
            log=log,
            curve=curve,
            baseline_reward=baseline_reward,
        )

    # aliases ------------------------------------------------------------- #
    train = refine

    def evaluate(self, seed: Optional[int] = None, n_episodes: Optional[int] = None) -> Dict[str, float]:
        env = self.build_eval_env(seed=seed)
        policy = self.build_policy(self.env)
        return evaluate_policy(  # type: ignore[misc]
            env, policy, n_episodes=int(n_episodes or self.config.eval_episodes),
            seed=seed, deterministic=self.config.deterministic_eval,
        )

    def run(self, seeds: Optional[Sequence[int]] = None, **kwargs: Any) -> Any:
        """Run over several seeds; returns a ``JSRLSummary`` (or a single result)."""
        seeds = list(seeds) if seeds is not None else self.config.seed_list()
        results: List[Any] = []
        summary = JSRLSummary(
            method=METHOD_NAME,
            task=self.resolve_task(),
            explanation="none",
            seeds=[int(s) if s is not None else None for s in seeds],
            config=self.config,
        )
        for seed in seeds:
            try:
                res = self.refine(seed=seed, **kwargs)
            except Exception as exc:  # keep sweeps alive
                summary.notes.append(f"seed {seed} failed: {exc}")
                summary.error = str(exc)
                continue
            results.append(res)
            summary.final_rewards.append(float(getattr(res, "final_reward", np.nan)))
            summary.baseline_rewards.append(float(getattr(res, "baseline_reward", np.nan)))
            summary.env_steps += int(getattr(res, "env_steps", 0) or 0)
            summary.seconds += float(getattr(res, "seconds", 0.0) or 0.0)
            try:
                summary.curves.append(np.asarray(res.refining_curve(self.config.curve_window), dtype=np.float32))
            except Exception:  # pragma: no cover - defensive
                pass
        summary.results = results
        self._summary = summary
        if len(results) == 1:
            # keep the RefineResult contract for single-seed runs
            res = results[0]
            try:
                res.summary = summary  # type: ignore[attr-defined]
            except Exception:
                pass
            return res
        return summary

    run_seeds = run

    @property
    def summary(self) -> Optional[JSRLSummary]:
        return self._summary

    def describe(self) -> Dict[str, Any]:
        return {
            "method": METHOD_NAME,
            "task": self.resolve_task(),
            "config": self.config.to_dict(),
            "source": "Uchendu et al. (2023), https://github.com/steventango/jumpstart-rl",
        }


# --------------------------------------------------------------------------- #
# policy helpers
# --------------------------------------------------------------------------- #


def _policy_act(policy: Any, obs: Any, deterministic: bool = False, rng: Any = None) -> Tuple[np.ndarray, float, float]:
    """Sample an action from the refining policy plus (value, log-prob)."""
    if hasattr(policy, "act"):
        try:
            out = policy.act(obs, deterministic=deterministic)
            if isinstance(out, tuple):
                action, value, log_prob = out[0], out[1], (out[2] if len(out) > 2 else 0.0)
                return np.asarray(action, dtype=np.float32).ravel(), _scalar(value), _scalar(log_prob)
            return np.asarray(out, dtype=np.float32).ravel(), 0.0, 0.0
        except TypeError:
            pass
    if isinstance(policy, np.ndarray):  # pragma: no cover
        return np.asarray(policy, dtype=np.float32).ravel(), 0.0, 0.0
    if callable(policy):
        return np.asarray(policy(obs), dtype=np.float32).ravel(), 0.0, 0.0
    raise TypeError(f"Cannot act with policy of type {type(policy)}")


def _policy_score(policy: Any, obs: Any, action: Any) -> Tuple[float, float]:
    """Return (value, log pi_e(a|s)) for an arbitrary already-taken action."""
    import torch  # local import: keeps module import light

    if hasattr(policy, "predict_values") and hasattr(policy, "log_prob_of"):
        with torch.no_grad():
            try:
                value = _scalar(policy.predict_values(flatten_obs(obs)))
            except Exception:
                value = 0.0
            try:
                log_prob = _scalar(policy.log_prob_of(flatten_obs(obs), np.asarray(action, dtype=np.float32).ravel()))
            except Exception:
                log_prob = 0.0
        return float(value), float(log_prob)
    if hasattr(policy, "evaluate_actions"):  # pragma: no cover - defensive
        with torch.no_grad():
            try:
                values, log_prob, _ent = policy.evaluate_actions(
                    flatten_obs(obs), np.asarray(action, dtype=np.float32).ravel()
                )
                return _scalar(values), _scalar(log_prob)
            except Exception:
                return 0.0, 0.0
    return 0.0, 0.0


def _make_result(
    policy: Any,
    iterations: int,
    env_steps: int,
    seconds: float,
    episode_returns: Sequence[float],
    final_eval_reward: float,
    log: List[Dict[str, Any]],
    curve: Sequence[float],
    baseline_reward: float = float("nan"),
) -> Any:
    """Build a ``RefineResult`` when available, else a lightweight stand-in."""
    if RefineResult is not None:
        kwargs: Dict[str, Any] = dict(
            policy=policy,
            iterations=int(iterations),
            env_steps=int(env_steps),
            seconds=float(seconds),
            episode_returns=[float(x) for x in episode_returns],
            mean_episode_return=float(np.mean(episode_returns)) if len(episode_returns) else float("nan"),
            final_eval_reward=float(final_eval_reward),
            critical_fraction=0.0,  # JSRL does not use critical states
            log=log,
            stopping_reason="iterations",
        )
        try:
            result = RefineResult(**kwargs)
        except Exception:  # pragma: no cover - tolerate signature drift
            try:
                result = RefineResult()
                for key, value in kwargs.items():
                    try:
                        setattr(result, key, value)
                    except Exception:
                        pass
            except Exception:
                result = None
        if result is not None:
            try:
                result._jsrl_curve = list(curve)  # type: ignore[attr-defined]
                result.baseline_reward = float(baseline_reward)  # type: ignore[attr-defined]
            except Exception:
                pass
            return result

    class _FallbackResult:  # pragma: no cover - only used without refine.py
        def __init__(self, **kw: Any) -> None:
            self.__dict__.update(kw)
            self._jsrl_curve = list(curve)
            self.baseline_reward = float(baseline_reward)

        @property
        def final_reward(self) -> float:
            return float(self.__dict__.get("final_eval_reward", float("nan")))

        def refining_curve(self, window: int = 1) -> np.ndarray:
            arr = np.asarray(self._jsrl_curve, dtype=np.float32)
            return _moving_average(arr, int(window)) if window and int(window) > 1 else arr

    return _FallbackResult(**kwargs)


# --------------------------------------------------------------------------- #
# Factories / functional entry points
# --------------------------------------------------------------------------- #


def make_jsrl_refiner(
    env: Any = None,
    policy: Any = None,
    config: Optional[Any] = None,
    **kwargs: Any,
) -> JSRLRefiner:
    """Factory mirroring :func:`rice.algorithms.refine.make_refiner`."""
    cfg = config
    if cfg is None:
        cfg = JSRLConfig.from_mapping(kwargs.pop("config_dict", None), **kwargs)
    elif isinstance(cfg, dict):
        cfg = JSRLConfig.from_mapping(cfg, **kwargs)
    elif isinstance(cfg, RefineConfig) if RefineConfig is not None else False:
        # tolerate a RefineConfig: keep the task/net-arch/LR information
        cfg = JSRLConfig.from_mapping(cfg.to_dict() if hasattr(cfg, "to_dict") else cfg.__dict__)
    return JSRLRefiner(env=env, policy=policy, config=cfg, **kwargs)


def jsrl_refine(
    env: Any = None,
    policy: Any = None,
    mask_network: Any = None,
    config: Optional[Any] = None,
    seeds: Optional[Sequence[int]] = None,
    n_iterations: Optional[int] = None,
    evaluation_env: Any = None,
    state_manager: Any = None,
    rng: Any = None,
    task: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Functional JSRL entry point (used by ``rice.baselines.run_baseline``)."""
    refiner = make_jsrl_refiner(
        env=env, policy=policy, config=config, evaluation_env=evaluation_env,
        state_manager=state_manager, rng=rng, task=task,
        mask_network=mask_network, **kwargs,
    )
    seeds = list(seeds) if seeds is not None else refiner.config.seed_list()
    if len(seeds) == 1:
        return refiner.refine(seed=seeds[0], n_iterations=n_iterations)
    return refiner.run(seeds=seeds, n_iterations=n_iterations)


# --------------------------------------------------------------------------- #
# Aliases (naming tolerance across the code base / external repos)
# --------------------------------------------------------------------------- #

jsrl_baseline = jsrl_refine
jumpstart_rl_refine = jsrl_refine
JumpStartRLConfig = JSRLConfig
JumpStartRLRefiner = JSRLRefiner
JSRLRunner = JSRLRefiner
JSRLEvaluator = JSRLRefiner
make_jumpstart_refiner = make_jsrl_refiner

__all__ = [
    "JSRLConfig",
    "JSRLRefiner",
    "JSRLSummary",
    "JSRLRunner",
    "JumpStartRLConfig",
    "JumpStartRLRefiner",
    "JSRLEvaluator",
    "METHOD_NAME",
    "METHOD_ALIASES",
    "HORIZON_MODES",
    "make_jsrl_refiner",
    "make_jumpstart_refiner",
    "jsrl_refine",
    "jsrl_baseline",
    "jumpstart_rl_refine",
]
