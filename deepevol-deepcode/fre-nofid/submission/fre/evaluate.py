"""Zero-shot evaluation harness for Functional Reward Encodings (FRE).

This driver implements Phase 5 of the reproduction plan:

    encode exactly K=32 (state, reward) pairs of a *target* task into a 128-d
    latent ``z`` with the frozen FRE encoder, then roll out the z-conditioned
    policy for ``num_episodes`` episodes across ``num_seeds`` seeds and report
    the normalised return/score in ``[0, 100]``.

The harness is deliberately tolerant of the different environment wrappers and
task-spec objects used across the code base:

* ``fre/envs/antmaze_wrapper.py``   -> ``AntMazeWrapper``, ``antmaze_eval_tasks``
* ``fre/envs/exorl_wrapper.py``     -> ``ExORLWrapper``, ``exorl_eval_tasks``
* ``fre/envs/kitchen_wrapper.py``   -> ``KitchenWrapper``, ``kitchen_eval_tasks``
* ``fre/rewards/eval_rewards.py``   -> ``all_eval_tasks``, ``task_groups`` ...

It can be used either as a library (``Evaluator(...).evaluate_tasks(...)``) or as
a script::

    python -m fre.evaluate --checkpoint runs/antmaze/policy.pt --domain antmaze

Notable conventions
-------------------
* ``context_size`` is K = 32 (encoder side), ``decoder_size`` is K' = 8.
* Exactly ``context_size`` reward samples are used at test time (this is the
  central zero-shot claim of the paper; baselines FB/SF use 5120).
* Episode scores are aggregated with ``score_episodes`` when a task exposes a
  ``TaskScoring`` object, otherwise a success-rate / normalised-return fallback
  is used.
* Results are reported per task group (``antmaze-goal-reaching``,
  ``exorl-walker-goals``, ...) exactly as in Table 1.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - torch is a hard dependency in practice
    import torch
except Exception as exc:  # pragma: no cover
    torch = None  # type: ignore
    _TORCH_IMPORT_ERROR = exc

from .rewards.eval_rewards import (
    ANTMAZE_GOAL_TASKS,
    ANTMAZE_MAX_STEPS,
    EXORL_MAX_STEPS,
    KITCHEN_TASKS,
    SCORE_MEAN_REWARD,
    SCORE_NORMALIZED_RETURN,
    SCORE_SUCCESS_RATE,
    TaskScoring,
    aggregate_scores,
    all_eval_tasks,
    task_groups,
    tasks_for_group,
)

# ---------------------------------------------------------------------------
# Optional model imports (kept soft so the harness can be imported without the
# networks being importable, e.g. during pure data/EvalTask testing).
# ---------------------------------------------------------------------------

try:
    from .fre.encoder import FREEncoder
except Exception:  # pragma: no cover
    try:
        from fre.encoder import FREEncoder  # type: ignore
    except Exception:
        FREEncoder = None  # type: ignore

try:
    from .fre.latent_policy import LatentPolicyBundle, sample_latent_z
except Exception:  # pragma: no cover
    try:
        from fre.latent_policy import LatentPolicyBundle, sample_latent_z  # type: ignore
    except Exception:
        LatentPolicyBundle = None  # type: ignore
        sample_latent_z = None  # type: ignore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTEXT_SIZE = 32          # K encoder (state, reward) pairs -- paper's zero-shot budget
DECODER_SIZE = 8           # K' decoder states used during Phase-1 pretraining
DEFAULT_NUM_EPISODES = 20  # evaluation episodes per task per seed
DEFAULT_NUM_SEEDS = 5      # independent seeds
DEFAULT_DOMAIN = "antmaze"
DOMAINS = ("antmaze", "exorl", "kitchen")

# Table 1 reference numbers (mean over seeds).  Used for the sanity/verification
# report only; never used to score an agent.
TABLE1_REFERENCE = {
    "antmaze": 52.8,
    "antmaze-std": 18.2,
    "exorl": 51.5,
    "exorl-std": 6.3,
    "kitchen": 66.0,
    "kitchen-std": 3.0,
    "all": 57.0,
    "all-std": 9.0,
}

_TASK_GROUP_ALIASES = {
    "ant-goal-reaching": "antmaze-goal-reaching",
    "ant-directional": "antmaze-directional",
    "ant-random-simplex": "antmaze-random-simplex",
    "ant-path-loop": "antmaze-path-loop",
    "ant-path-edges": "antmaze-path-edges",
    "ant-path-center": "antmaze-path-center",
}


# ---------------------------------------------------------------------------
# Configuration / result containers
# ---------------------------------------------------------------------------


@dataclass
class EvalConfig:
    """Configuration for a zero-shot evaluation run."""

    domain: str = DEFAULT_DOMAIN
    num_episodes: int = DEFAULT_NUM_EPISODES
    num_seeds: int = DEFAULT_NUM_SEEDS
    context_size: int = CONTEXT_SIZE
    decoder_size: int = DECODER_SIZE
    deterministic: bool = True
    context_seed: int = 0
    base_seed: int = 0
    max_steps: Optional[int] = None
    max_tasks: Optional[int] = None
    score_mode: str = SCORE_SUCCESS_RATE
    use_mean_latent: bool = False
    extra_context_noise: float = 0.0
    device: Optional[str] = None
    exorl_root: Optional[str] = None
    kitchen_dataset: Optional[str] = None
    antmaze_dataset: Optional[str] = None
    verbose: bool = True
    print_every: int = 1

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeResult:
    """Outcome of a single evaluation episode."""

    return_: float
    length: int
    success: bool = False
    terminated: bool = False
    truncated: bool = False
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def score_return(self) -> float:
        return float(self.return_)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "return": self.return_,
            "length": self.length,
            "success": self.success,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }


@dataclass
class TaskResult:
    """Evaluation outcome for a single task across all seeds."""

    name: str
    domain: str
    family: str = "unknown"
    group: Optional[str] = None
    per_seed_scores: List[float] = field(default_factory=list)
    per_seed_returns: List[float] = field(default_factory=list)
    per_seed_success: List[float] = field(default_factory=list)
    num_episodes: int = 0
    error: Optional[str] = None

    @property
    def mean(self) -> float:
        vals = [s for s in self.per_seed_scores if not math.isnan(s)]
        return float(np.mean(vals)) if vals else float("nan")

    @property
    def std(self) -> float:
        vals = [s for s in self.per_seed_scores if not math.isnan(s)]
        return float(np.std(vals)) if len(vals) > 1 else 0.0

    @property
    def mean_return(self) -> float:
        vals = [s for s in self.per_seed_returns if not math.isnan(s)]
        return float(np.mean(vals)) if vals else float("nan")

    @property
    def success_rate(self) -> float:
        vals = [s for s in self.per_seed_success if not math.isnan(s)]
        return float(np.mean(vals)) if vals else float("nan")

    def as_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out.update({"mean": self.mean, "std": self.std})
        return out


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def to_torch(x: Any, device: Optional[Any] = None, dtype: Any = None) -> "torch.Tensor":
    """Convert array-like input to a float32 torch tensor."""
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for evaluation")
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    if dtype is not None:
        t = t.to(dtype)
    elif t.dtype != torch.float32:
        t = t.float()
    if device is not None:
        t = t.to(device)
    return t


def _to_numpy(x: Any) -> np.ndarray:
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def resolve_device(device: Optional[str] = None) -> Any:
    """Pick a torch device (cuda if available, else cpu)."""
    if torch is None:  # pragma: no cover
        return None
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _extract_states(result: Any) -> np.ndarray:
    """Normalise the many shapes ``sample_context``-like helpers can return."""
    states = result
    if isinstance(result, Mapping):
        for key in ("states", "observations", "obs", "context_states", "encoder_observations"):
            if key in result:
                states = result[key]
                break
    elif isinstance(result, (tuple, list)):
        # Prefer a (N, D) array, i.e. the first 2-D element.
        for item in result:
            arr = _to_numpy(item)
            if arr.ndim >= 2:
                states = item
                break
        else:
            states = result[0]
    arr = _to_numpy(states)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _extract_rewards(result: Any) -> Optional[np.ndarray]:
    if isinstance(result, Mapping):
        for key in ("rewards", "context_rewards", "reward"):
            if key in result:
                return np.asarray(_to_numpy(result[key]), dtype=np.float32).reshape(-1)
    elif isinstance(result, (tuple, list)) and len(result) >= 2:
        second = _to_numpy(result[1])
        if second.ndim >= 1:
            return np.asarray(second, dtype=np.float32).reshape(-1)
    return None


def _task_reward_callable(task: Any) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """Best-effort accessor for a task's reward function."""
    if task is None:
        return None
    for attr in ("reward_fn", "reward_function", "fn"):
        fn = getattr(task, attr, None)
        if fn is None:
            continue
        if callable(fn):
            return fn
        inner = getattr(fn, "__call__", None)
        if inner is not None:
            return fn
    for attr in ("reward", "compute", "batch_reward"):
        fn = getattr(task, attr, None)
        if callable(fn):
            return fn
    if callable(task):
        return task
    return None


def _evaluate_reward_fn(fn: Callable[..., Any], states: np.ndarray) -> np.ndarray:
    """Apply a reward callable to ``(N, D)`` states, tolerating signatures."""
    try:
        out = fn(states)
    except TypeError:
        out = fn(to_torch(states))
    arr = _to_numpy(out)
    if arr.ndim > 1:
        arr = arr.reshape(arr.shape[0], -1).mean(axis=-1)
    return np.asarray(arr, dtype=np.float32).reshape(-1)


def _call_reset(env: Any, seed: Optional[int] = None, goal: Any = None, task: Any = None) -> Any:
    """Call ``env.reset`` with the subset of kwargs it accepts."""
    attempts = [
        dict(seed=seed, goal=goal, task=task),
        dict(seed=seed, task=task),
        dict(seed=seed, goal=goal),
        dict(seed=seed),
        dict(),
    ]
    last_exc: Optional[Exception] = None
    for kwargs in attempts:
        try:
            return env.reset(**{k: v for k, v in kwargs.items() if v is not None} or {})
        except TypeError as exc:  # signature mismatch -> try fewer kwargs
            last_exc = exc
            continue
        except Exception as exc:  # runtime error -> propagate the first time
            last_exc = exc
            break
    if last_exc is not None:
        raise last_exc
    return env.reset()


def _unpack_step(result: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
    if not isinstance(result, (tuple, list)):  # pragma: no cover - defensive
        raise ValueError(f"unexpected step() return: {type(result)}")
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
    elif len(result) == 4:
        obs, reward, done, info = result
        terminated, truncated = bool(done), False
    else:  # pragma: no cover
        raise ValueError(f"unexpected step() tuple length: {len(result)}")
    info = dict(info) if isinstance(info, Mapping) else {}
    return obs, float(reward), bool(terminated), bool(truncated), info


def _success_from_info(info: Mapping[str, Any], env: Any) -> bool:
    for key in ("success", "is_success", "goal_reached", "achieved"):
        if key in info and info[key] is not None:
            return bool(info[key])
    for attr in ("last_episode_success",):
        val = getattr(env, attr, None)
        if val is not None:
            try:
                return bool(val)
            except Exception:
                pass
    return False


# ---------------------------------------------------------------------------
# Environment construction
# ---------------------------------------------------------------------------


def build_env(
    domain: str = DEFAULT_DOMAIN,
    task: Any = None,
    *,
    config: Optional[EvalConfig] = None,
    dataset: Any = None,
    **kwargs: Any,
) -> Any:
    """Construct an environment wrapper for ``domain`` pinned to ``task``."""
    config = config or EvalConfig(domain=domain)
    domain = domain.lower()

    if domain in ("antmaze", "ant", "antmaze-large-diverse-v2"):
        from .envs.antmaze_wrapper import AntMazeWrapper

        init_kwargs: Dict[str, Any] = dict(task=task)
        if config.antmaze_dataset:
            init_kwargs["dataset_name"] = config.antmaze_dataset
        if config.max_steps:
            init_kwargs["max_episode_steps"] = config.max_steps
        init_kwargs.update(kwargs)
        if dataset is not None:
            init_kwargs["dataset"] = dataset
        try:
            return AntMazeWrapper(**init_kwargs)
        except TypeError:
            init_kwargs.pop("dataset_name", None)
            return AntMazeWrapper(**init_kwargs)

    if domain in ("exorl", "walker", "cheetah"):
        from .envs.exorl_wrapper import ExORLWrapper

        env_domain = domain if domain in ("walker", "cheetah") else kwargs.pop("exorl_domain", "walker")
        init_kwargs = dict(domain=env_domain, task=task, root=config.exorl_root)
        if config.max_steps:
            init_kwargs["max_episode_steps"] = config.max_steps
        init_kwargs.update(kwargs)
        if dataset is not None:
            init_kwargs["dataset"] = dataset
        return ExORLWrapper(**init_kwargs)

    if domain in ("kitchen", "kitchen-complete-v0"):
        from .envs.kitchen_wrapper import KitchenWrapper

        init_kwargs = dict(task=task)
        if config.kitchen_dataset:
            init_kwargs["dataset_name"] = config.kitchen_dataset
        if config.max_steps:
            init_kwargs["max_episode_steps"] = config.max_steps
        init_kwargs.update(kwargs)
        if dataset is not None:
            init_kwargs["dataset"] = dataset
        return KitchenWrapper(**init_kwargs)

    raise ValueError(f"unknown evaluation domain: {domain!r}")


def domain_of(task: Any, default: str = DEFAULT_DOMAIN) -> str:
    """Infer the env domain that owns ``task`` from its name/attributes."""
    name = ""
    if task is not None:
        name = str(getattr(task, "name", "") or getattr(task, "task", "") or "")
    dom = str(getattr(task, "domain", "") or "")
    blob = f"{dom} {name}".lower()
    for key in ("antmaze", "ant-"):
        if key in blob:
            return "antmaze"
    if "kitchen" in blob:
        return "kitchen"
    for key in ("exorl", "walker", "cheetah"):
        if key in blob:
            return "exorl"
    # Fall back on the task-object family/kind.
    family = str(getattr(task, "family", "") or "")
    if family in ("goal", "directional", "simplex", "path"):
        return "antmaze"
    return default


def suite_for_domain(domain: str, **kwargs: Any) -> Dict[str, Any]:
    """Return the zero-shot task suite (name -> task) for a domain."""
    domain = domain.lower()
    if domain in ("antmaze", "ant"):
        from .envs.antmaze_wrapper import antmaze_eval_tasks

        return _normalise_suite(antmaze_eval_tasks(**kwargs))
    if domain in ("exorl",):
        from .envs.exorl_wrapper import exorl_eval_tasks

        return _normalise_suite(exorl_eval_tasks(**kwargs))
    if domain in ("walker", "cheetah"):
        from .envs.exorl_wrapper import exorl_eval_tasks

        return _normalise_suite(exorl_eval_tasks(domain, **kwargs))
    if domain in ("kitchen",):
        from .envs.kitchen_wrapper import kitchen_eval_tasks

        return _normalise_suite(kitchen_eval_tasks(**kwargs))
    # Last resort: the pure reward-side task suite.
    return _normalise_suite(all_eval_tasks())


def _normalise_suite(suite: Any) -> Dict[str, Any]:
    """Accept either a dict or a list of task objects."""
    if suite is None:
        return {}
    if isinstance(suite, Mapping):
        return dict(suite)
    out: Dict[str, Any] = {}
    for idx, task in enumerate(suite):
        name = str(getattr(task, "name", f"task-{idx}"))
        out[name] = task
    return out


# ---------------------------------------------------------------------------
# The evaluator
# ---------------------------------------------------------------------------


class Evaluator:
    """Zero-shot evaluator: encode K reward samples -> z -> rollout."""

    def __init__(
        self,
        encoder: Any,
        policy: Any = None,
        config: Optional[EvalConfig] = None,
        device: Optional[Any] = None,
        *,
        reward_model: Any = None,
    ) -> None:
        self.encoder = encoder
        self.policy = policy
        self.reward_model = reward_model  # unused for zero-shot; kept for symmetry
        self.config = config or EvalConfig()
        self.device = device if device is not None else resolve_device(self.config.device)
        if self.encoder is not None:
            self.encoder.to(self.device)
            self.encoder.eval()
        if self.policy is not None:
            self.policy.to(self.device)
            self.policy.eval()

    # -- context construction -------------------------------------------------

    def sample_context_states(self, env: Any, num_samples: int, seed: int) -> np.ndarray:
        """Sample ``num_samples`` states that will be labelled by the target reward."""
        rng = np.random.default_rng(seed)

        for helper in ("sample_context", "sample_states", "sample_dataset_states"):
            fn = getattr(env, helper, None)
            if callable(fn):
                for kwargs in (
                    dict(num_samples=num_samples, seed=seed),
                    dict(num_samples=num_samples),
                    dict(num_states=num_samples),
                    dict(),
                ):
                    try:
                        out = fn(**kwargs)
                    except TypeError:
                        continue
                    except Exception:
                        break
                    states = _extract_states(out)
                    if states.shape[0] >= 1:
                        return _resize_states(states, num_samples, rng)
                break

        observations = self._env_observations(env)
        if observations is not None and len(observations) > 0:
            idx = rng.choice(len(observations), size=min(num_samples, len(observations)), replace=False)
            states = np.asarray(observations[idx], dtype=np.float32)
            return _resize_states(states, num_samples, rng)

        # Worst case: roll a few random-length episodes to gather states.
        states = self._rollout_states(env, num_samples, seed)
        return _resize_states(states, num_samples, rng)

    @staticmethod
    def _env_observations(env: Any) -> Optional[np.ndarray]:
        for container_name in ("dataset", "_dataset", "payload"):
            container = getattr(env, container_name, None)
            if container is None:
                continue
            for key in ("encoder_observations", "observations", "obs", "states"):
                if isinstance(container, Mapping) and key in container:
                    return np.asarray(container[key], dtype=np.float32)
            arr = getattr(container, key if False else "observations", None)
            if arr is not None:
                return np.asarray(arr, dtype=np.float32)
        direct = getattr(env, "observations", None)
        if direct is not None:
            return np.asarray(direct, dtype=np.float32)
        return None

    def _rollout_states(self, env: Any, num_samples: int, seed: int) -> np.ndarray:
        try:
            obs, _ = _unpack_reset(_call_reset(env, seed=seed))
        except Exception:
            return np.zeros((num_samples, self._infer_state_dim(env)), dtype=np.float32)
        collected = [np.asarray(obs, dtype=np.float32).reshape(-1)]
        action_dim = self._action_dim(env)
        rng = np.random.default_rng(seed)
        while sum(len(c.reshape(-1, c.size // max(1, c.shape[-1]))) for c in [np.stack(collected)]) < 0:
            break
        max_steps = num_samples * 3 + 10
        for _ in range(max_steps):
            if len(collected) >= num_samples:
                break
            try:
                out = env.step(rng.uniform(-1.0, 1.0, size=action_dim).astype(np.float32))
            except Exception:
                break
            obs, _, term, trunc, _ = _unpack_step(out)
            collected.append(np.asarray(obs, dtype=np.float32).reshape(-1))
            if term or trunc:
                try:
                    obs, _ = _unpack_reset(_call_reset(env, seed=int(rng.integers(0, 2 ** 31 - 1))))
                except Exception:
                    break
        states = np.stack(collected[:num_samples]).astype(np.float32)
        return states

    @staticmethod
    def _action_dim(env: Any) -> int:
        for attr in ("action_dim",):
            val = getattr(env, attr, None)
            if isinstance(val, int):
                return val
        space = getattr(env, "action_space", None)
        if space is not None and hasattr(space, "shape") and space.shape:
            return int(np.prod(space.shape))
        return 1

    def _infer_state_dim(self, env: Any) -> int:
        for attr in ("observation_dim", "obs_dim", "state_dim"):
            val = getattr(env, attr, None)
            if isinstance(val, int):
                return val
        if self.encoder is not None and hasattr(self.encoder, "state_dim"):
            return int(self.encoder.state_dim)
        return 1

    def build_task_context(
        self,
        task: Any,
        env: Any,
        num_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return exactly ``context_size`` ``(states, rewards)`` pairs for ``task``."""
        num_samples = int(num_samples or self.config.context_size)
        seed = self.config.context_seed if seed is None else seed

        states = self.sample_context_states(env, num_samples, seed)
        reward_fn = _task_reward_callable(task)
        if reward_fn is None:
            reward_fn = _task_reward_callable(getattr(env, "task", None))
        if reward_fn is None:
            raise ValueError(f"task {getattr(task, 'name', task)!r} exposes no reward function")

        rewards = _evaluate_reward_fn(reward_fn, states)

        # Clip into the encoder's expected [-1, 1] reward range.
        rewards = np.clip(rewards, -1.0, 1.0).astype(np.float32)
        if self.config.extra_context_noise > 0.0:
            rng = np.random.default_rng(seed + 7919)
            rewards = rewards + rng.normal(0.0, self.config.extra_context_noise, size=rewards.shape).astype(np.float32)
            rewards = np.clip(rewards, -1.0, 1.0)

        # Ensure at least one success sample when the task is goal-reaching, which
        # mirrors the paper's "guarantee at least one encoding sample contains the
        # goal" requirement for the prior AND for evaluation contexts.
        goal = getattr(task, "goal", None)
        if goal is not None and rewards.max() < 0.0:
            goal_state = self._state_for_goal(env, goal, states.shape[1])
            if goal_state is not None:
                states = np.concatenate([states[:-1], goal_state.reshape(1, -1)], axis=0)
                rewards = _evaluate_reward_fn(reward_fn, states).astype(np.float32)
                rewards = np.clip(rewards, -1.0, 1.0)

        if states.shape[0] != num_samples:
            rng = np.random.default_rng(seed)
            states = _resize_states(states, num_samples, rng)
            rewards = _resize_rewards(rewards, num_samples, states)

        return states.astype(np.float32), rewards.astype(np.float32)

    @staticmethod
    def _state_for_goal(env: Any, goal: Any, state_dim: int) -> Optional[np.ndarray]:
        try:
            goal_arr = np.asarray(goal, dtype=np.float32).reshape(-1)
        except Exception:
            return None
        state = np.zeros(state_dim, dtype=np.float32)
        if state_dim >= goal_arr.size and goal_arr.size <= 4:
            state[: goal_arr.size] = goal_arr
            return state
        return None

    def encode_task(
        self,
        task: Any,
        env: Any,
        *,
        context: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        num_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Encode a target task's K reward samples into a latent ``z`` (numpy)."""
        if context is None:
            context = self.build_task_context(task, env, num_samples=num_samples, seed=seed)
        states, rewards = context
        with torch.no_grad():
            s = to_torch(states, device=self.device).unsqueeze(0)  # (1, K, D)
            r = to_torch(rewards, device=self.device).unsqueeze(0)  # (1, K)
            z = None
            for kwargs in (
                dict(use_mean=self.config.use_mean_latent),
                dict(),
            ):
                try:
                    z = self.encoder.encode(s, r, **kwargs)
                    break
                except TypeError:
                    continue
            if z is None:  # fallback: use the posterior mean
                enc = self.encoder(s, r)
                mu = enc[0] if isinstance(enc, (tuple, list)) else enc
                z = mu
        z = _to_numpy(z)
        return np.asarray(z, dtype=np.float32).reshape(-1)

    # -- rollouts -------------------------------------------------------------

    def rollout(
        self,
        env: Any,
        z: np.ndarray,
        task: Any = None,
        *,
        seed: int = 0,
        num_episodes: Optional[int] = None,
        deterministic: Optional[bool] = None,
        max_steps: Optional[int] = None,
    ) -> List[EpisodeResult]:
        """Roll out the z-conditioned policy for ``num_episodes`` episodes."""
        num_episodes = int(num_episodes or self.config.num_episodes)
        deterministic = self.config.deterministic if deterministic is None else deterministic
        max_steps = int(max_steps or self.config.max_steps or self._default_max_steps(task, env))
        rng = np.random.default_rng(seed)
        episodes: List[EpisodeResult] = []

        for ep in range(num_episodes):
            ep_seed = int(rng.integers(0, 2 ** 31 - 1))
            try:
                obs = _unpack_reset(_call_reset(env, seed=ep_seed, goal=getattr(task, "goal", None), task=task))[0]
            except Exception as exc:  # pragma: no cover
                episodes.append(EpisodeResult(0.0, 0, False, False, False, {"error": repr(exc)}))
                continue

            obs = np.asarray(obs, dtype=np.float32).reshape(-1)
            total_reward = 0.0
            success = False
            terminated = truncated = False
            length = 0

            for step in range(max_steps):
                action = self.act(obs, z, deterministic=deterministic)
                try:
                    out = env.step(action)
                except Exception as exc:  # pragma: no cover
                    terminated = True
                    total_reward += 0.0
                    length = step + 1
                    episodes.append(EpisodeResult(total_reward, length, success, terminated, False, {"error": repr(exc)}))
                    break
                next_obs, reward, terminated, truncated, info = _unpack_step(out)
                total_reward += reward
                length = step + 1
                success = success or _success_from_info(info, env)
                obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)
                if terminated or truncated:
                    if "success" in info:
                        success = success or bool(info["success"])
                    break
            else:
                # Ran out of budget without terminal -> still record the episode.
                pass

            if length == 0:
                length = max_steps
            episodes.append(EpisodeResult(total_reward, length, success, terminated, truncated))

        return episodes

    @staticmethod
    def _default_max_steps(task: Any, env: Any) -> int:
        for src in (task, env):
            for attr in ("max_episode_steps", "max_steps", "_max_episode_steps"):
                val = getattr(src, attr, None)
                if isinstance(val, (int, float)) and val > 0:
                    return int(val)
        name = str(getattr(task, "name", ""))
        if "kitchen" in name:
            return EXORL_MAX_STEPS
        if "exorl" in name or "walker" in name or "cheetah" in name:
            return EXORL_MAX_STEPS
        return ANTMAZE_MAX_STEPS

    def act(self, obs: np.ndarray, z: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Select an action from the z-conditioned policy."""
        if self.policy is None:
            # Random policy fallback (used to sanity check envs score ~0).
            rng = np.random.default_rng(abs(hash(obs.tobytes())) % (2 ** 31))
            return rng.uniform(-1.0, 1.0, size=self._policy_action_dim()).astype(np.float32)

        with torch.no_grad():
            o = to_torch(obs, device=self.device).unsqueeze(0)          # (1, obs_dim)
            zt = to_torch(z, device=self.device).reshape(1, -1)         # (1, latent_dim)
            action = None
            for kwargs in (
                dict(deterministic=deterministic),
                dict(),
            ):
                for meth in ("act", "select_action", "sample"):
                    fn = getattr(self.policy, meth, None)
                    if not callable(fn):
                        continue
                    try:
                        out = fn(o, zt, **kwargs)
                    except TypeError:
                        try:
                            out = fn(obs, z, **kwargs)
                        except Exception:
                            continue
                    except Exception:
                        continue
                    if isinstance(out, (tuple, list)):
                        out = out[0]
                    action = out
                    break
                if action is not None:
                    break
            if action is None:  # pragma: no cover
                raise RuntimeError("policy exposes no usable act/select_action/sample method")
        arr = _to_numpy(action).reshape(-1)
        return np.clip(arr, -1.0, 1.0).astype(np.float32)

    def _policy_action_dim(self) -> int:
        for attr in ("action_dim",):
            val = getattr(self.policy, attr, None)
            if isinstance(val, int):
                return val
        return 1

    # -- scoring --------------------------------------------------------------

    def score_episodes(self, task: Any, episodes: Sequence[EpisodeResult]) -> Tuple[float, Dict[str, float]]:
        """Convert a list of episodes into a score in ``[0, 100]``."""
        returns = np.array([e.return_ for e in episodes], dtype=np.float64)
        lengths = np.array([max(1, e.length) for e in episodes], dtype=np.float64)
        successes = np.array([1.0 if e.success else 0.0 for e in episodes], dtype=np.float64)
        metrics = {
            "mean_return": float(returns.mean()) if len(returns) else float("nan"),
            "mean_length": float(lengths.mean()) if len(lengths) else float("nan"),
            "success_rate": float(successes.mean()) if len(successes) else float("nan"),
        }

        scoring = getattr(task, "scoring", None) if task is not None else None
        mode = self.config.score_mode
        if scoring is not None and getattr(scoring, "mode", None):
            mode = scoring.mode

        if mode == SCORE_SUCCESS_RATE:
            score = 100.0 * metrics["success_rate"]
        elif mode == SCORE_MEAN_REWARD:
            r_min, r_max = -1.0, 1.0
            if scoring is not None:
                r_min = getattr(scoring, "reward_min", r_min)
                r_max = getattr(scoring, "reward_max", r_max)
            mean_r = float((returns / lengths).mean()) if len(returns) else 0.0
            denom = (r_max - r_min) if (r_max - r_min) != 0 else 1.0
            score = 100.0 * (mean_r - r_min) / denom
        elif mode == SCORE_NORMALIZED_RETURN:
            v_min, v_max = 0.0, 1.0
            if scoring is not None:
                v_min = getattr(scoring, "min_return", 0.0) or 0.0
                v_max = getattr(scoring, "max_return", 1.0) or 1.0
            denom = (v_max - v_min) if (v_max - v_min) != 0 else 1.0
            score = 100.0 * (float(returns.mean()) - v_min) / denom
        else:
            score = 100.0 * metrics["success_rate"] if metrics["success_rate"] > 0 else 0.0

        if not np.isfinite(score):
            score = float("nan")
        score = float(np.clip(score, 0.0, 100.0))
        metrics["score"] = score
        return score, metrics

    # -- full evaluation ------------------------------------------------------

    def evaluate_task(
        self,
        task: Any,
        env: Any = None,
        *,
        seeds: Optional[Sequence[int]] = None,
        num_episodes: Optional[int] = None,
        context_seed: Optional[int] = None,
    ) -> TaskResult:
        """Evaluate a single task across seeds (encode once per seed)."""
        name = str(getattr(task, "name", task))
        domain = domain_of(task, self.config.domain)
        family = str(getattr(task, "family", getattr(task, "kind", "unknown")))
        result = TaskResult(name=name, domain=domain, family=family, num_episodes=int(num_episodes or self.config.num_episodes))

        if env is None:
            env = build_env(domain, task, config=self.config)

        seeds = list(seeds) if seeds is not None else list(self._seed_list())
        base_context_seed = self.config.context_seed if context_seed is None else context_seed

        errors: List[str] = []
        for i, seed in enumerate(seeds):
            try:
                env_seed = int(seed)
                if hasattr(env, "set_task"):
                    try:
                        env.set_task(task)
                    except Exception:
                        pass
                z = self.encode_task(task, env, seed=base_context_seed + i)
                episodes = self.rollout(
                    env,
                    z,
                    task,
                    seed=env_seed,
                    num_episodes=num_episodes,
                    max_steps=self.config.max_steps,
                )
                score, metrics = self.score_episodes(task, episodes)
                result.per_seed_scores.append(score)
                result.per_seed_returns.append(metrics["mean_return"])
                result.per_seed_success.append(metrics["success_rate"])
            except Exception as exc:  # record and continue with the remaining seeds
                errors.append(f"seed {seed}: {exc!r}")
                result.per_seed_scores.append(float("nan"))
                result.per_seed_returns.append(float("nan"))
                result.per_seed_success.append(float("nan"))
            if self.config.verbose and (i + 1) % max(1, self.config.print_every) == 0:
                print(
                    f"  [{name}] seed {seed}: score={result.per_seed_scores[-1]:.1f} "
                    f"(mean={result.mean:.1f})",
                    flush=True,
                )
        if errors:
            result.error = "; ".join(errors)
        return result

    def _seed_list(self) -> List[int]:
        return [self.config.base_seed + i for i in range(self.config.num_seeds)]

    def evaluate_tasks(
        self,
        tasks: Any,
        envs: Optional[Mapping[str, Any]] = None,
        *,
        seeds: Optional[Sequence[int]] = None,
        num_episodes: Optional[int] = None,
    ) -> Dict[str, TaskResult]:
        """Evaluate a suite of tasks, reusing one env per domain where possible."""
        suite = _normalise_suite(tasks) if not isinstance(tasks, Mapping) else dict(tasks)
        if self.config.max_tasks is not None:
            suite = dict(list(suite.items())[: int(self.config.max_tasks)])

        env_pool: Dict[str, Any] = dict(envs or {})
        results: Dict[str, TaskResult] = {}
        for name, task in suite.items():
            dom = domain_of(task, self.config.domain)
            env = env_pool.get(dom)
            if env is None:
                try:
                    env = build_env(dom, task, config=self.config)
                except Exception as exc:
                    results[name] = TaskResult(name=name, domain=dom, error=f"env build failed: {exc!r}")
                    continue
                env_pool[dom] = env
            if self.config.verbose:
                print(f"[evaluate] {name} (domain={dom})", flush=True)
            results[name] = self.evaluate_task(task, env, seeds=seeds, num_episodes=num_episodes)
        self._envs = env_pool
        return results

    def evaluate_domain(
        self,
        domain: str = DEFAULT_DOMAIN,
        *,
        seeds: Optional[Sequence[int]] = None,
        num_episodes: Optional[int] = None,
        suite_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, TaskResult]:
        suite = suite_for_domain(domain, **(suite_kwargs or {"root": self.config.exorl_root} if domain == "exorl" else {}))
        return self.evaluate_tasks(suite, seeds=seeds, num_episodes=num_episodes)

    def close(self) -> None:
        for env in getattr(self, "_envs", {}).values():
            try:
                env.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Context / state helpers
# ---------------------------------------------------------------------------


def _resize_states(states: np.ndarray, num_samples: int, rng: np.random.Generator) -> np.ndarray:
    n = states.shape[0]
    if n == num_samples:
        return states
    if n > num_samples:
        idx = rng.choice(n, size=num_samples, replace=False)
        return states[idx]
    idx = rng.choice(n, size=num_samples, replace=True)
    return states[idx]


def _resize_rewards(rewards: np.ndarray, num_samples: int, states: np.ndarray) -> np.ndarray:
    n = rewards.shape[0]
    if n == num_samples:
        return rewards
    if n > num_samples:
        return rewards[:num_samples]
    pad = np.full((num_samples - n,), float(rewards[-1]) if n else 0.0, dtype=rewards.dtype)
    return np.concatenate([rewards, pad], axis=0)


def _unpack_reset(result: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    if isinstance(result, (tuple, list)) and len(result) == 2:
        obs, info = result
        info = dict(info) if isinstance(info, Mapping) else {"info": info}
    elif isinstance(result, Mapping):
        obs = result.get("observation", result.get("obs"))
        info = dict(result)
    else:
        obs, info = result, {}
    return np.asarray(obs, dtype=np.float32), info


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------


@dataclass
class LoadedAgent:
    encoder: Any
    policy: Any
    meta: Dict[str, Any]


def _torch_load(path: str, device: Any) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # older torch
        return torch.load(path, map_location=device)


def _extract_state_dict(obj: Any, keys: Sequence[str]) -> Optional[Mapping[str, Any]]:
    if not isinstance(obj, Mapping):
        return None
    for key in keys:
        if key in obj and isinstance(obj[key], Mapping):
            return obj[key]
    return None


def infer_dims_from_state_dict(sd: Mapping[str, Any]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Best-effort (obs_dim, action_dim, latent_dim) from a policy state dict."""
    obs_dim = action_dim = latent_dim = None

    def _find(*patterns: str, offset: int = 0, dim: int = -1) -> Optional[int]:
        for key, value in sd.items():
            if not hasattr(value, "shape"):
                continue
            if all(p in key for p in patterns):
                try:
                    return int(value.shape[dim]) - offset
                except Exception:
                    continue
        return None

    latent_dim = _find("q_network", "0.weight", offset=0, dim=-1)
    obs_dim = _find("q_network", "0.weight", offset=None, dim=-1) if False else None
    # q_network.0.weight shape: [hidden, obs_dim + action_dim + latent_dim]

    return obs_dim, action_dim, latent_dim


def build_policy(
    obs_dim: int,
    action_dim: int,
    latent_dim: int = 128,
    *,
    hidden_sizes: Sequence[int] = (512, 512, 512),
) -> Any:
    if LatentPolicyBundle is None:  # pragma: no cover
        raise RuntimeError("LatentPolicyBundle is unavailable (torch import failed?)")
    return LatentPolicyBundle(
        obs_dim=obs_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        hidden_sizes=tuple(hidden_sizes),
    )


def build_encoder(state_dim: int, latent_dim: int = 128) -> Any:
    if FREEncoder is None:  # pragma: no cover
        raise RuntimeError("FREEncoder is unavailable (torch import failed?)")
    return FREEncoder(state_dim=state_dim, latent_dim=latent_dim)


def load_agent(
    checkpoint: str,
    *,
    state_dim: Optional[int] = None,
    action_dim: Optional[int] = None,
    latent_dim: int = 128,
    device: Optional[Any] = None,
    load_policy: bool = True,
) -> LoadedAgent:
    """Load an encoder (+ optional policy) from a training checkpoint.

    Accepts checkpoints produced by ``train_encoder.py`` / ``train_policy.py``
    which are expected to be dicts with (a subset of) keys::

        {"encoder": sd, "policy": sd, "state_dim": int, "obs_dim": int,
         "action_dim": int, "latent_dim": int, "config": {...}}
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required to load checkpoints")
    device = device if device is not None else resolve_device()
    payload = _torch_load(checkpoint, device)
    meta: Dict[str, Any] = {}

    encoder_sd = _extract_state_dict(payload, ("encoder", "encoder_state_dict", "fre_encoder"))
    policy_sd = _extract_state_dict(payload, ("policy", "policy_state_dict", "agent"))
    if encoder_sd is None and isinstance(payload, Mapping) and any(
        k.endswith("state_projection.weight") for k in payload
    ):
        encoder_sd = payload  # bare encoder state dict

    if isinstance(payload, Mapping):
        for key in ("state_dim", "obs_dim", "observation_dim", "encoder_state_dim"):
            if state_dim is None and isinstance(payload.get(key), int):
                state_dim = int(payload[key])
        if action_dim is None and isinstance(payload.get("action_dim"), int):
            action_dim = int(payload["action_dim"])
        if isinstance(payload.get("latent_dim"), int):
            latent_dim = int(payload["latent_dim"])
        cfg = payload.get("config")
        if isinstance(cfg, Mapping):
            meta["config"] = dict(cfg)
            state_dim = state_dim or cfg.get("state_dim") or cfg.get("encoder_state_dim")
            action_dim = action_dim or cfg.get("action_dim")
            latent_dim = int(cfg.get("latent_dim", latent_dim))
        meta["step"] = payload.get("step", payload.get("global_step"))

    if state_dim is None and encoder_sd is not None:
        for key, value in encoder_sd.items():
            if hasattr(value, "shape") and "state_projection" in key and key.endswith("weight"):
                state_dim = int(value.shape[1])
                break

    if state_dim is None and policy_sd is not None:
        for key, value in policy_sd.items():
            if hasattr(value, "shape") and key.endswith("0.weight") and "q_network" in key:
                if action_dim is None:
                    action_dim = None
                break

    if state_dim is None:
        raise ValueError(
            "could not infer state_dim from checkpoint; pass state_dim explicitly"
        )

    encoder = build_encoder(int(state_dim), latent_dim=latent_dim).to(device)
    if encoder_sd is not None:
        encoder.load_state_dict(encoder_sd, strict=False)
    meta["state_dim"] = int(state_dim)
    meta["latent_dim"] = int(latent_dim)

    policy = None
    if load_policy:
        if action_dim is None and policy_sd is not None:
            for key, value in policy_sd.items():
                if hasattr(value, "shape") and key.endswith("q_network.0.weight"):
                    # [hidden, obs_dim + action_dim + latent_dim]
                    total = int(value.shape[1])
                    action_dim = total - int(state_dim) - int(latent_dim)
                    break
        if action_dim is None:
            action_dim = 8 if int(state_dim) == 29 else (6 if int(state_dim) == 24 else 3)
        policy = build_policy(int(state_dim), int(action_dim), latent_dim=latent_dim).to(device)
        if policy_sd is not None:
            policy.load_state_dict(policy_sd, strict=False)
        meta["action_dim"] = int(action_dim)

    meta["checkpoint"] = checkpoint
    return LoadedAgent(encoder=encoder, policy=policy, meta=meta)


def load_eval_agent(
    checkpoint: Optional[str] = None,
    *,
    encoder: Any = None,
    policy: Any = None,
    state_dim: Optional[int] = None,
    action_dim: Optional[int] = None,
    latent_dim: int = 128,
    device: Optional[Any] = None,
) -> LoadedAgent:
    """Either load from a checkpoint or accept already-constructed modules."""
    if encoder is not None or policy is not None:
        meta = {"state_dim": state_dim, "latent_dim": latent_dim, "action_dim": action_dim}
        return LoadedAgent(encoder=encoder, policy=policy, meta=meta)
    if checkpoint is None:
        raise ValueError("either `checkpoint` or `encoder`/`policy` must be provided")
    return load_agent(
        checkpoint,
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        device=device,
    )


# ---------------------------------------------------------------------------
# Top-level convenience API
# ---------------------------------------------------------------------------


def zero_shot_evaluate(
    agent: LoadedAgent,
    domain: str = DEFAULT_DOMAIN,
    config: Optional[EvalConfig] = None,
    *,
    tasks: Optional[Mapping[str, Any]] = None,
    envs: Optional[Mapping[str, Any]] = None,
    seeds: Optional[Sequence[int]] = None,
    num_episodes: Optional[int] = None,
    suite_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, TaskResult]:
    """Run the full zero-shot evaluation for a domain and return per-task results."""
    config = config or EvalConfig(domain=domain)
    config.domain = domain
    evaluator = Evaluator(agent.encoder, agent.policy, config)
    if tasks is None:
        kwargs: Dict[str, Any] = dict(suite_kwargs or {})
        if domain == "exorl" and config.exorl_root:
            kwargs.setdefault("root", config.exorl_root)
        tasks = suite_for_domain(domain, **kwargs)
    results = evaluator.evaluate_tasks(tasks, envs=envs, seeds=seeds, num_episodes=num_episodes)
    evaluator.close()
    return results


def results_to_summary(results: Mapping[str, TaskResult]) -> Dict[str, Any]:
    """Aggregate per-task results into Table-1 style group scores."""
    per_task_scores: Dict[str, List[float]] = {}
    per_task_means: Dict[str, float] = {}
    for name, res in results.items():
        vals = [s for s in res.per_seed_scores if np.isfinite(s)]
        if not vals:
            continue
        per_task_scores[name] = vals
        per_task_means[name] = float(np.mean(vals))

    groups = aggregate_scores(per_task_scores, compute_std=True)
    summary = {
        "tasks": {
            name: {
                "mean": res.mean,
                "std": res.std,
                "mean_return": res.mean_return,
                "success_rate": res.success_rate,
                "domain": res.domain,
                "family": res.family,
                "num_episodes": res.num_episodes,
                "error": res.error,
            }
            for name, res in results.items()
        },
        "groups": groups,
        "overall": _overall_score(groups),
    }
    return summary


def _group_matches(group_name: str, task_name: str) -> bool:
    g = group_name.lower()
    t = task_name.lower()
    if g == "all":
        return True
    if g.startswith("antmaze") or g.startswith("ant-"):
        if "antmaze" not in t and "ant-" not in t:
            return False
    if g.startswith("exorl"):
        if "exorl" not in t:
            return False
        for dom in ("walker", "cheetah"):
            if dom in g and dom not in t:
                return False
    if g.startswith("kitchen"):
        if "kitchen" not in t:
            return False
    # Compare the tail descriptor (e.g. "goal-reaching", "velocity").
    for token in ("goal-reaching", "goals", "directional", "random-simplex", "simplex",
                  "path-loop", "path-edges", "path-center", "velocity"):
        if token in g and token not in t:
            return False
    return True


def _overall_score(groups: Mapping[str, Any]) -> Dict[str, float]:
    """Compute domain-level and all-task aggregate scores.

    The paper's "all" column averages the three domain column values (AntMaze,
    ExORL, Kitchen), each computed over its task set.  We reproduce that by
    averaging group means within each domain.
    """
    per_domain: Dict[str, List[float]] = {"antmaze": [], "exorl": [], "kitchen": []}
    for gname, gval in groups.items():
        mean = gval["mean"] if isinstance(gval, Mapping) else float(gval)
        if not np.isfinite(mean):
            continue
        gl = gname.lower()
        if gl.startswith("antmaze") or gl.startswith("ant"):
            per_domain["antmaze"].append(mean)
        elif gl.startswith("exorl"):
            per_domain["exorl"].append(mean)
        elif gl.startswith("kitchen"):
            per_domain["kitchen"].append(mean)

    out: Dict[str, float] = {}
    domain_means = []
    for dom, vals in per_domain.items():
        if vals:
            out[dom] = float(np.mean(vals))
            domain_means.append(float(np.mean(vals)))
        else:
            out[dom] = float("nan")
    out["all"] = float(np.mean(domain_means)) if domain_means else float("nan")
    return out


def verify_against_table1(summary: Mapping[str, Any]) -> Dict[str, Any]:
    """Compare aggregated scores with the paper's Table 1 reference values."""
    overall = summary.get("overall", {})
    report: Dict[str, Any] = {}
    for key in ("antmaze", "exorl", "kitchen", "all"):
        got = float(overall.get(key, float("nan")))
        ref = TABLE1_REFERENCE.get(key, float("nan"))
        tol = TABLE1_REFERENCE.get(f"{key}-std", 0.0) or 0.0
        report[key] = {
            "ours": got,
            "paper": ref,
            "abs_diff": abs(got - ref) if np.isfinite(got) else float("nan"),
            "within_1std": bool(np.isfinite(got) and abs(got - ref) <= tol + 1e-6),
        }
    return report


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Zero-shot evaluation for FRE")
    p.add_argument("--checkpoint", type=str, default=None, help="path to trained checkpoint")
    p.add_argument("--domain", type=str, default=DEFAULT_DOMAIN, choices=list(DOMAINS) + ["walker", "cheetah"])
    p.add_argument("--num-episodes", type=int, default=DEFAULT_NUM_EPISODES)
    p.add_argument("--num-seeds", type=int, default=DEFAULT_NUM_SEEDS)
    p.add_argument("--context-size", type=int, default=CONTEXT_SIZE)
    p.add_argument("--state-dim", type=int, default=None)
    p.add_argument("--action-dim", type=int, default=None)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--score-mode", type=str, default=SCORE_SUCCESS_RATE,
                   choices=[SCORE_SUCCESS_RATE, SCORE_MEAN_REWARD, SCORE_NORMALIZED_RETURN])
    p.add_argument("--stochastic", action="store_true", help="use stochastic policy actions")
    p.add_argument("--max-tasks", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--exorl-root", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output", type=str, default=None, help="path to write the JSON summary")
    p.add_argument("--random-policy", action="store_true", help="sanity check envs with a random policy")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    config = EvalConfig(
        domain=args.domain,
        num_episodes=args.num_episodes,
        num_seeds=args.num_seeds,
        context_size=args.context_size,
        deterministic=not args.stochastic,
        max_tasks=args.max_tasks,
        max_steps=args.max_steps,
        score_mode=args.score_mode,
        device=args.device,
        exorl_root=args.exorl_root,
        verbose=not args.quiet,
    )

    if args.random_policy or args.checkpoint is None:
        if not args.random_policy:
            print("[evaluate] no checkpoint supplied -> running random-policy sanity check")
        agent = LoadedAgent(encoder=None, policy=None, meta={"random_policy": True})
    else:
        agent = load_agent(
            args.checkpoint,
            state_dim=args.state_dim,
            action_dim=args.action_dim,
            latent_dim=args.latent_dim,
            device=resolve_device(args.device),
        )

    if agent.encoder is None:
        # Random-policy sanity check path does not need an encoder.
        class _DummyEvaluator(Evaluator):
            def encode_task(self, task, env, **kwargs):  # type: ignore[override]
                return np.zeros(self.config.latent_dim, dtype=np.float32)

        evaluator: Evaluator = _DummyEvaluator(None, None, config)
    else:
        evaluator = Evaluator(agent.encoder, agent.policy, config)

    t0 = time.time()
    results = evaluator.evaluate_domain(args.domain)
    evaluator.close()

    summary = results_to_summary(results)
    summary["config"] = config.as_dict()
    summary["wallclock_s"] = time.time() - t0
    summary["verification"] = verify_against_table1(summary)

    print("\n=== FRE zero-shot evaluation summary ===")
    for name, res in results.items():
        print(f"{name:<34} mean={res.mean:6.1f}  std={res.std:5.1f}  n_ep={res.num_episodes}")
    print("\nGroup scores:")
    for gname, gval in summary["groups"].items():
        print(f"{gname:<34} mean={gval['mean']:6.1f}  std={gval['std']:5.1f}  tasks={gval['num_tasks']}")
    print("\nDomain / overall:")
    for key in ("antmaze", "exorl", "kitchen", "all"):
        ref = TABLE1_REFERENCE.get(key)
        print(f"{key:<10} {summary['overall'].get(key, float('nan')):6.1f}   (paper: {ref})")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w") as fh:
            json.dump(_to_jsonable(summary), fh, indent=2)
        print(f"\nWrote summary to {args.output}")

    return summary


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


if __name__ == "__main__":  # pragma: no cover
    main()
