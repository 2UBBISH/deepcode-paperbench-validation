"""Zero-shot online evaluation on the D4RL Kitchen domain (FRE).

Paper references
----------------
* Appendix C.3::

    "For the Kitchen evaluation tasks, we utilize the seven standard subtasks within the D4RL
     Kitchen environment. Because each task already defines a sparse reward, we directly use those
     sparse rewards as evaluation tasks."

* Section 5: Kitchen is evaluated with 7 subtasks and the resulting row is compared against FB, SF,
  GC-IQL, GC-BC and OPAL-10 in Table 1 (FRE 66 +- 3).

Design
------
Kitchen is a multi-subtask environment: the D4RL ``kitchen-complete-v0`` task exposes a per-step
``info['task'][<subtask>]`` boolean flag and a sparse reward that counts *newly completed* subtasks.
Following the paper we define one evaluation task per standard subtask and use exactly that sparse
reward (a reward of 1.0 at the timestep where the subtask first becomes complete, 0.0 elsewhere).

The module mirrors :mod:`fre.envs.antmaze_eval` / :mod:`fre.envs.exorl_eval`:

* task classes with a uniform ``reward_from_state`` / ``reward`` interface,
* a gym wrapper that applies the task reward and tracks success,
* rollout / suite evaluation returning 0-100 normalised returns (protocol: 5 seeds x 20 episodes),
* helpers to build the K=32 ``(state, reward)`` context pairs fed to the frozen FRE encoder, and to
  wrap a z-conditioned IQL agent into a rollout-compatible action function.

A small :class:`SyntheticKitchenEnv` fallback keeps the module importable/testable without MuJoCo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "KITCHEN_SUBTASKS",
    "KITCHEN_SUBTASK_NAMES",
    "KITCHEN_MAX_EPISODE_STEPS",
    "KITCHEN_ENCODER_SAMPLES",
    "KITCHEN_OBS_DIM",
    "KITCHEN_ENV_ID",
    "KITCHEN_TABLE1_REFERENCE",
    "KITCHEN_TABLE1_FRE_PER_TASK",
    "KitchenTask",
    "SubtaskTask",
    "CombinedKitchenTask",
    "KitchenEvalWrapper",
    "KitchenEpisodeResult",
    "SyntheticKitchenEnv",
    "make_kitchen_env",
    "make_subtask_tasks",
    "make_kitchen_task_suite",
    "get_task_suite",
    "subtask_flag",
    "sample_task_encoder_pairs",
    "encode_task_latent",
    "make_kitchen_policy_fn",
    "rollout_episode",
    "evaluate_task",
    "evaluate_suite",
    "evaluate_kitchen_suite",
]


# --------------------------------------------------------------------------------------------------
# Constants (paper: Appendix C.3 / Table 1)
# --------------------------------------------------------------------------------------------------

KITCHEN_ENV_ID = "kitchen-complete-v0"
KITCHEN_MAX_EPISODE_STEPS = 1000
KITCHEN_ENCODER_SAMPLES = 32
#: Standard D4RL Kitchen observation dimension (59) -- used only as a synthetic fallback value.
KITCHEN_OBS_DIM = 59

#: The seven standard D4RL Kitchen subtasks, paired with their ``info['task']`` key.
#  Order follows the canonical D4RL subtask ordering in the full (complete) task.
KITCHEN_SUBTASKS: Tuple[Tuple[str, str], ...] = (
    ("microwave", "microwave"),
    ("kettle", "kettle"),
    ("slide-cabinet", "slide cabinet"),
    ("hinge-cabinet", "hinge cabinet"),
    ("light-switch", "light switch"),
    ("bottom-burner", "bottom burner"),
    ("top-burner", "top burner"),
)

KITCHEN_SUBTASK_NAMES: Tuple[str, ...] = tuple(name for name, _ in KITCHEN_SUBTASKS)

#: Table 1 reference values (mean +- std across 5 seeds, normalised to 0-100).
KITCHEN_TABLE1_REFERENCE: Dict[str, Dict[str, float]] = {
    "kitchen": {"FRE": 66.0, "FB": 3.0, "SF": 1.0, "GC-IQL": 59.0, "GC-BC": 35.0, "OPAL-10": 26.0},
}
KITCHEN_TABLE1_FRE_PER_TASK: Dict[str, float] = {
    # Paper reports only the aggregate Kitchen row of Table 1.
    "kitchen-all": 66.0,
}


# --------------------------------------------------------------------------------------------------
# Environment helpers
# --------------------------------------------------------------------------------------------------


class SyntheticKitchenEnv:
    """Deterministic, dependency-free stand-in for the D4RL Kitchen environment.

    Exposes the same minimal interface used by the evaluation wrappers: ``reset() -> obs``,
    ``step(action) -> (obs, reward, done, info)`` with ``info['task']`` subtask flags, ``obs_dim``
    /``action_dim`` and ``max_episode_steps``. Used when ``d4rl``/``gym`` are unavailable so that the
    evaluation plumbing can still be exercised (results are meaningless for the paper's numbers).
    """

    def __init__(
        self,
        obs_dim: int = KITCHEN_OBS_DIM,
        action_dim: int = 9,
        max_episode_steps: int = KITCHEN_MAX_EPISODE_STEPS,
        subtasks: Sequence[Tuple[str, str]] = KITCHEN_SUBTASKS,
        seed: Optional[int] = None,
        completion_prob: float = 0.01,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.max_episode_steps = int(max_episode_steps)
        self.subtasks = tuple(subtasks)
        self.completion_prob = float(completion_prob)
        self._rng = np.random.RandomState(0 if seed is None else int(seed))
        self._state: Optional[np.ndarray] = None
        self._flags: Dict[str, bool] = {}
        self._t = 0
        self.synthetic = True

    # -- spaces -----------------------------------------------------------------------------------
    class _Space:
        def __init__(self, shape, low=-1.0, high=1.0):
            self.shape = tuple(shape)
            self.low = np.full(self.shape, low, dtype=np.float32)
            self.high = np.full(self.shape, high, dtype=np.float32)

        def sample(self):
            return np.random.uniform(self.low, self.high).astype(np.float32)

    @property
    def observation_space(self):
        return self._Space((self.obs_dim,))

    @property
    def action_space(self):
        return self._Space((self.action_dim,))

    # -- gym API ----------------------------------------------------------------------------------
    def seed(self, seed: Optional[int] = None):
        self._rng = np.random.RandomState(0 if seed is None else int(seed))
        return [seed]

    def reset(self, **kwargs):
        self._t = 0
        self._flags = {key: False for _, key in self.subtasks}
        self._state = self._rng.randn(self.obs_dim).astype(np.float32)
        return self._state.copy()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        delta = np.zeros(self.obs_dim, dtype=np.float32)
        n = min(self.obs_dim, action.size)
        delta[:n] = action[:n]
        self._state = (self._state + 0.1 * delta).astype(np.float32)
        self._t += 1

        newly = 0.0
        task_info: Dict[str, float] = {}
        for idx, (_, key) in enumerate(self.subtasks):
            if not self._flags[key] and self._rng.rand() < self.completion_prob:
                self._flags[key] = True
                newly += 1.0
            task_info[key] = 1.0 if self._flags[key] else 0.0

        done = self._t >= self.max_episode_steps
        info = {"task": task_info, "success": float(newly > 0)}
        return self._state.copy(), float(newly), bool(done), info


def make_kitchen_env(
    env_id: str = KITCHEN_ENV_ID,
    seed: Optional[int] = None,
    allow_synthetic: bool = True,
    **kwargs: Any,
):
    """Create a D4RL Kitchen environment, falling back to :class:`SyntheticKitchenEnv`.

    Mirrors :func:`fre.envs.antmaze_eval.make_antmaze_env` / :func:`fre.envs.exorl_eval.make_exorl_env`.
    """
    try:  # pragma: no cover - depends on the (heavy) D4RL/MuJoCo stack
        import gym  # type: ignore

        try:
            import d4rl  # noqa: F401  (registers the kitchen-* environments)
        except Exception:
            pass

        env = gym.make(env_id)
        if seed is not None:
            try:
                env.seed(int(seed))
            except Exception:
                pass
            try:
                env.action_space.seed(int(seed))
            except Exception:
                pass
        return env
    except Exception:
        if not allow_synthetic:
            raise
        return SyntheticKitchenEnv(seed=seed, **{k: v for k, v in kwargs.items() if k in
                                                ("obs_dim", "action_dim", "max_episode_steps",
                                                 "completion_prob")})


# --------------------------------------------------------------------------------------------------
# Reward helpers
# --------------------------------------------------------------------------------------------------


def subtask_flag(info: Any, key: str) -> Optional[bool]:
    """Extract a boolean subtask-completion flag from a D4RL Kitchen ``info`` dict.

    Handles both the modern nested form ``info['task'][key]`` and a flat layout. Returns ``None``
    when the flag is absent (caller decides on a fallback).
    """
    if info is None:
        return None
    task = info.get("task", info) if isinstance(info, dict) else None
    if not isinstance(task, dict):
        return None
    if key in task:
        return bool(task[key])
    # tolerate space/underscore variants
    normalised = {str(k).replace(" ", "_").replace("-", "_").lower(): v for k, v in task.items()}
    alt = str(key).replace(" ", "_").replace("-", "_").lower()
    if alt in normalised:
        return bool(normalised[alt])
    return None


# --------------------------------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------------------------------


class KitchenTask:
    """Base class for Kitchen evaluation tasks (uniform reward interface).

    The paper's evaluation protocol normalises returns to 0-100 by averaging over 20 rollouts per
    seed and reporting the standard deviation over 5 seeds; ``min_return``/``max_return`` bound the
    raw episode return so that the normalisation is well defined.
    """

    kind = "kitchen"

    def __init__(self, name: str, max_episode_steps: int = KITCHEN_MAX_EPISODE_STEPS) -> None:
        self.name = name
        self.max_episode_steps = int(max_episode_steps)

    # -- bounds ------------------------------------------------------------------------------------
    @property
    def min_return(self) -> float:
        return 0.0

    @property
    def max_return(self) -> float:
        return 1.0

    # -- reward ------------------------------------------------------------------------------------
    def reward_from_info(self, info: Any) -> float:
        raise NotImplementedError

    def reward(self, obs, action=None, next_obs=None, info=None, **kwargs) -> float:
        """Task reward for a single transition (interface-compatible with the other eval modules)."""
        val = self.reward_from_info(info)
        return float(val)

    def reward_from_state(self, state: np.ndarray, index: Optional[int] = None,
                          flags: Optional[Dict[str, bool]] = None) -> float:
        """Reward for a dataset state (used to build encoder ``(s, eta(s))`` pairs).

        Without simulator info we approximate the sparse reward by the subtask flag recorded for the
        dataset state when available (``flags``), else 0.0.
        """
        if flags is None:
            return 0.0
        return 1.0 if bool(flags.get(self.subtask_key, False)) else 0.0

    # -- helpers -----------------------------------------------------------------------------------
    @property
    def subtask_key(self) -> str:
        raise NotImplementedError

    def success(self, info: Any) -> bool:
        flag = subtask_flag(info, self.subtask_key)
        if flag is not None:
            return bool(flag)
        if isinstance(info, dict) and "success" in info:
            return bool(info["success"])
        return False

    def normalize_return(self, total_return: float) -> float:
        lo, hi = self.min_return, self.max_return
        if hi <= lo:
            return 0.0
        return float(np.clip(100.0 * (total_return - lo) / (hi - lo), 0.0, 100.0))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "subtask": self.subtask_key,
            "max_episode_steps": self.max_episode_steps,
            "min_return": self.min_return,
            "max_return": self.max_return,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(name={self.name!r}, subtask={self.subtask_key!r})"


class SubtaskTask(KitchenTask):
    """One of the seven standard Kitchen subtasks with its native sparse reward.

    The reward is 1.0 exactly on the timestep where the subtask becomes complete (D4RL Kitchen's
    sparse "newly completed subtask" reward for a single subtask) and 0.0 otherwise. The maximum
    episode return is therefore 1.0, and the normalised return becomes ``100 * success`` (with
    partial credit if the agent completes the subtask more than once, which cannot happen).
    """

    kind = "kitchen-subtask"

    def __init__(
        self,
        subtask_name: str,
        subtask_key: Optional[str] = None,
        max_episode_steps: int = KITCHEN_MAX_EPISODE_STEPS,
        reward_on_completion: float = 1.0,
    ) -> None:
        super().__init__(name=f"kitchen-{subtask_name}", max_episode_steps=max_episode_steps)
        self._name = str(subtask_name)
        self._key = str(subtask_key if subtask_key is not None else subtask_name)
        self.reward_on_completion = float(reward_on_completion)
        self._completed = False

    @property
    def subtask_key(self) -> str:
        return self._key

    @property
    def min_return(self) -> float:
        return 0.0

    @property
    def max_return(self) -> float:
        return float(self.reward_on_completion)

    def reset(self) -> None:
        self._completed = False

    # Native sparse reward: only fire once, at the completion timestep.
    def reward_from_info(self, info: Any) -> float:
        flag = subtask_flag(info, self._key)
        if flag is None:
            if isinstance(info, dict) and "success" in info:
                flag = bool(info["success"])
            else:
                return 0.0
        if flag and not self._completed:
            self._completed = True
            return self.reward_on_completion
        return 0.0

    def reward(self, obs, action=None, next_obs=None, info=None, **kwargs) -> float:
        return float(self.reward_from_info(info))

    def reward_from_state(self, state, index: Optional[int] = None, flags=None) -> float:
        if flags is None:
            return 0.0
        return float(self.reward_on_completion) if bool(flags.get(self._key, False)) else 0.0

    def success(self, info: Any) -> bool:
        flag = subtask_flag(info, self._key)
        if flag is not None:
            return bool(flag) or self._completed
        if isinstance(info, dict) and "success" in info:
            return bool(info["success"]) or self._completed
        return self._completed


class CombinedKitchenTask(KitchenTask):
    """The full ``kitchen-complete`` task: natively sums newly completed subtasks.

    Included for completeness (the paper's Table 1 Kitchen row averages the seven subtask scores).
    """

    kind = "kitchen-complete"

    def __init__(self, max_episode_steps: int = KITCHEN_MAX_EPISODE_STEPS,
                 subtasks: Sequence[Tuple[str, str]] = KITCHEN_SUBTASKS) -> None:
        super().__init__(name="kitchen-all", max_episode_steps=max_episode_steps)
        self.subtasks = tuple(subtasks)
        self._max = float(len(self.subtasks))

    @property
    def subtask_key(self) -> str:
        return "complete"

    @property
    def max_return(self) -> float:
        return self._max

    def reset(self) -> None:
        return None

    def reward_from_info(self, info: Any) -> float:
        if not isinstance(info, dict):
            return 0.0
        task = info.get("task", info)
        if not isinstance(task, dict):
            return 0.0
        # D4RL reports the sparse reward directly in most versions; fall back to counting flags.
        if "sparse_reward" in info:
            return float(info["sparse_reward"])
        return float(sum(1.0 for key, _ in self.subtasks if subtask_flag(info, key)))

    def success(self, info: Any) -> bool:
        return self.reward_from_info(info) > 0.0


# --------------------------------------------------------------------------------------------------
# Task factories
# --------------------------------------------------------------------------------------------------


def make_subtask_tasks(
    max_episode_steps: int = KITCHEN_MAX_EPISODE_STEPS,
    subtasks: Sequence[Tuple[str, str]] = KITCHEN_SUBTASKS,
) -> List[SubtaskTask]:
    """One :class:`SubtaskTask` for each of the seven standard D4RL Kitchen subtasks (App. C.3)."""
    return [
        SubtaskTask(name, key, max_episode_steps=max_episode_steps)
        for name, key in subtasks
    ]


def make_kitchen_task_suite(
    max_episode_steps: int = KITCHEN_MAX_EPISODE_STEPS,
    include_combined: bool = False,
) -> Dict[str, KitchenTask]:
    """Build the Kitchen evaluation suite: the seven subtasks (+ optionally ``kitchen-all``)."""
    suite: Dict[str, KitchenTask] = {}
    for task in make_subtask_tasks(max_episode_steps=max_episode_steps):
        suite[task.name] = task
    if include_combined:
        suite["kitchen-all"] = CombinedKitchenTask(max_episode_steps=max_episode_steps)
    return suite


def get_task_suite(name: str = "kitchen", **kwargs) -> Dict[str, KitchenTask]:
    """Suite lookup mirroring the sibling eval modules (``"kitchen"``/``"kitchen-all"``)."""
    if name in ("kitchen", "kitchen-subtasks", "kitchen-subtask"):
        return make_kitchen_task_suite(**kwargs)
    if name in ("kitchen-complete", "kitchen-all"):
        return {"kitchen-all": CombinedKitchenTask(
            max_episode_steps=kwargs.get("max_episode_steps", KITCHEN_MAX_EPISODE_STEPS))}
    raise KeyError(f"Unknown Kitchen suite {name!r}")


# --------------------------------------------------------------------------------------------------
# Environment wrapper, rollout, evaluation
# --------------------------------------------------------------------------------------------------


class KitchenEvalWrapper:
    """Gym wrapper applying a Kitchen task's reward and tracking success.

    Supports both the legacy 4-tuple and the gymnasium 5-tuple ``step`` API.
    """

    def __init__(
        self,
        env,
        task: KitchenTask,
        max_episode_steps: Optional[int] = None,
        terminate_on_success: bool = True,
    ) -> None:
        self.env = env
        self.task = task
        self.max_episode_steps = int(max_episode_steps or task.max_episode_steps)
        self.terminate_on_success = bool(terminate_on_success)
        self._t = 0
        self._success = False
        self._success_steps = -1

        self.observation_space = getattr(env, "observation_space", None)
        self.action_space = getattr(env, "action_space", None)

    # -- helpers -----------------------------------------------------------------------------------
    def _task_reset(self) -> None:
        if hasattr(self.task, "reset"):
            try:
                self.task.reset()
            except Exception:
                pass

    def reset(self, **kwargs):
        self._task_reset()
        self._t = 0
        self._success = False
        self._success_steps = -1
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
            return np.asarray(obs, dtype=np.float32), info
        return np.asarray(out, dtype=np.float32)

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            obs, _env_reward, done, truncated, info = out
            done = bool(done or truncated)
        else:
            obs, _env_reward, done, info = out
            done = bool(done)

        reward = self.task.reward(None, action, obs, info)
        self._t += 1
        if not self._success and self.task.success(info):
            self._success = True
            self._success_steps = self._t
        if self._t >= self.max_episode_steps:
            done = True
        if self.terminate_on_success and self._success:
            done = True
        return np.asarray(obs, dtype=np.float32), float(reward), bool(done), info

    # -- pass-throughs -----------------------------------------------------------------------------
    def seed(self, seed: Optional[int] = None):
        try:
            return self.env.seed(seed)
        except Exception:
            return [seed]

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass

    def __getattr__(self, item):  # pragma: no cover - transparent delegation
        return getattr(self.__dict__["env"], item)


@dataclass
class KitchenEpisodeResult:
    task_name: str
    total_return: float
    normalized_return: float
    length: int
    success: bool
    success_steps: int = -1

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task_name": self.task_name,
            "total_return": self.total_return,
            "normalized_return": self.normalized_return,
            "length": self.length,
            "success": self.success,
            "success_steps": self.success_steps,
        }


def rollout_episode(
    env: KitchenEvalWrapper,
    act_fn: Callable[[np.ndarray], np.ndarray],
    max_episode_steps: Optional[int] = None,
    seed: Optional[int] = None,
) -> KitchenEpisodeResult:
    """Run one rollout; ``act_fn`` maps an observation to an action (may ignore the observation)."""
    if seed is not None:
        try:
            env.seed(seed)
        except Exception:
            pass
    out = env.reset(seed=seed) if seed is not None else env.reset()
    obs = out[0] if isinstance(out, tuple) else out
    obs = np.asarray(obs, dtype=np.float32)

    limit = int(max_episode_steps or env.max_episode_steps)
    total = 0.0
    steps = 0
    done = False
    while not done and steps < limit:
        action = act_fn(obs)
        obs, reward, done, _info = env.step(action)
        total += float(reward)
        steps += 1

    return KitchenEpisodeResult(
        task_name=env.task.name,
        total_return=float(total),
        normalized_return=env.task.normalize_return(total),
        length=int(steps),
        success=bool(getattr(env, "_success", False)),
        success_steps=int(getattr(env, "_success_steps", -1)),
    )


def evaluate_task(
    task: KitchenTask,
    act_fn: Callable[[np.ndarray], np.ndarray],
    env=None,
    env_id: str = KITCHEN_ENV_ID,
    num_episodes: int = 20,
    max_episode_steps: Optional[int] = None,
    seed: int = 0,
    terminate_on_success: bool = True,
    allow_synthetic: bool = True,
) -> Dict[str, float]:
    """Evaluate one Kitchen subtask (protocol: 20 episodes per seed)."""
    own_env = env is None
    if own_env:
        env = make_kitchen_env(env_id=env_id, seed=seed, allow_synthetic=allow_synthetic)
    wrapped = env if isinstance(env, KitchenEvalWrapper) else KitchenEvalWrapper(
        env, task, max_episode_steps=max_episode_steps, terminate_on_success=terminate_on_success)

    returns: List[float] = []
    scores: List[float] = []
    successes: List[float] = []
    lengths: List[int] = []
    for ep in range(int(num_episodes)):
        res = rollout_episode(wrapped, act_fn, max_episode_steps=max_episode_steps,
                              seed=None if own_env else seed + ep)
        returns.append(res.total_return)
        scores.append(res.normalized_return)
        successes.append(1.0 if res.success else 0.0)
        lengths.append(res.length)

    if own_env:
        try:
            env.close()
        except Exception:
            pass

    returns_arr = np.asarray(returns, dtype=np.float64)
    scores_arr = np.asarray(scores, dtype=np.float64)
    return {
        "task": task.name,
        "score": float(scores_arr.mean()),
        "score_std": float(scores_arr.std()),
        "return": float(returns_arr.mean()),
        "return_std": float(returns_arr.std()),
        "success_rate": float(np.mean(successes)),
        "mean_length": float(np.mean(lengths)),
        "num_episodes": float(num_episodes),
    }


def evaluate_suite(
    suite: Any,
    act_fn_factory: Callable[[KitchenTask], Callable[[np.ndarray], np.ndarray]],
    envs: Optional[Dict[str, Any]] = None,
    env_id: str = KITCHEN_ENV_ID,
    num_episodes: int = 20,
    max_episode_steps: Optional[int] = None,
    seed: int = 0,
    terminate_on_success: bool = True,
    allow_synthetic: bool = True,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Evaluate a whole Kitchen suite; ``act_fn_factory`` builds the policy for each task."""
    tasks: Iterable[KitchenTask]
    if isinstance(suite, dict):
        tasks = list(suite.values())
    else:
        tasks = list(suite)

    results: Dict[str, Any] = {}
    scores: List[float] = []
    for task in tasks:
        env = None if envs is None else envs.get(task.name)
        res = evaluate_task(
            task,
            act_fn_factory(task),
            env=env,
            env_id=env_id,
            num_episodes=num_episodes,
            max_episode_steps=max_episode_steps,
            seed=seed,
            terminate_on_success=terminate_on_success,
            allow_synthetic=allow_synthetic,
        )
        results[task.name] = res
        scores.append(res["score"])
        if verbose:
            print(f"[kitchen] {task.name:<24} score={res['score']:6.1f} "
                  f"(+-{res['score_std']:4.1f}) succ={res['success_rate']:.2f}")
    results["mean"] = float(np.mean(scores)) if scores else 0.0
    results["num_tasks"] = len(tasks)
    return results


def evaluate_kitchen_suite(
    act_fn_factory: Callable[[KitchenTask], Callable[[np.ndarray], np.ndarray]],
    suites: Optional[Dict[str, KitchenTask]] = None,
    env_id: str = KITCHEN_ENV_ID,
    num_episodes: int = 20,
    max_episode_steps: int = KITCHEN_MAX_EPISODE_STEPS,
    seed: int = 0,
    envs: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Evaluate the seven Kitchen subtasks and aggregate into the ``kitchen`` row of Table 1."""
    if suites is None:
        suites = make_kitchen_task_suite(max_episode_steps=max_episode_steps)
    results = evaluate_suite(
        suites,
        act_fn_factory,
        envs=envs,
        env_id=env_id,
        num_episodes=num_episodes,
        max_episode_steps=max_episode_steps,
        seed=seed,
        **kwargs,
    )
    row = {
        "score": results.get("mean", 0.0),
        "num_tasks": results.get("num_tasks", 0),
        "per_task": {k: v for k, v in results.items() if isinstance(v, dict)},
    }
    score_std = float(np.std([v["score"] for v in row["per_task"].values()])) if row["per_task"] else 0.0
    row["score_std"] = score_std
    out = dict(results)
    out["kitchen"] = row
    return out


# --------------------------------------------------------------------------------------------------
# Encoder-context helpers and policy wrapper
# --------------------------------------------------------------------------------------------------


def _dataset_states(dataset) -> Optional[np.ndarray]:
    if dataset is None:
        return None
    for attr in ("states", "observations", "obs"):
        arr = getattr(dataset, attr, None)
        if arr is not None:
            return np.asarray(arr)
    return None


def _dataset_subtask_flags(dataset, key: str) -> Optional[np.ndarray]:
    """Per-state subtask completion flags if the dataset exposes them, else ``None``."""
    flags = getattr(dataset, "subtask_flags", None)
    if isinstance(flags, dict) and key in flags:
        return np.asarray(flags[key], dtype=bool)
    infos = getattr(dataset, "infos", None)
    if infos is not None and len(infos):
        try:
            return np.asarray([bool(subtask_flag(i, key)) for i in infos], dtype=bool)
        except Exception:
            return None
    return None


def sample_task_encoder_pairs(
    task: KitchenTask,
    dataset=None,
    num_samples: int = KITCHEN_ENCODER_SAMPLES,
    rng: Optional[np.random.RandomState] = None,
    env_states: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample the K=32 ``(state, eta(state))`` context pairs used to encode a Kitchen task.

    Following the paper's treatment of the other domains, the context must contain at least one state
    on which the reward fires (the subgoal state); otherwise the latent ``z`` carries no information
    about the task. When dataset subtask flags are available a completed state is forced into the
    context, mirroring the goal-state guarantee of the goal-reaching prior (Appendix B).
    """
    rng = np.random.RandomState(0) if rng is None else rng
    num_samples = int(num_samples)

    states: Optional[np.ndarray] = None
    flags: Optional[np.ndarray] = None
    key = getattr(task, "subtask_key", None)

    if dataset is not None:
        states = _dataset_states(dataset)
        if states is None and hasattr(dataset, "sample_states"):
            states = np.asarray(dataset.sample_states(num_samples))
        if states is not None and key is not None:
            flags = _dataset_subtask_flags(dataset, key)
    if states is None and env_states is not None:
        states = np.asarray(env_states)
    if states is None:
        raise ValueError("sample_task_encoder_pairs requires a dataset or env_states")

    states = np.asarray(states, dtype=np.float32)
    n = states.shape[0]
    idx = rng.randint(0, n, size=num_samples) if n > 0 else np.zeros(num_samples, dtype=int)
    enc_states = states[idx].copy()

    enc_rewards = np.zeros(num_samples, dtype=np.float32)
    if flags is not None and len(flags) == n:
        enc_rewards = flags[idx].astype(np.float32)
        done = np.where(flags)[0]
        if done.size > 0:
            enc_states[0] = states[done[rng.randint(0, done.size)]]
            enc_rewards[0] = 1.0
        else:  # no completed state in the data -- mark the highest-progress state as goal
            enc_states[0] = states[-1]
            enc_rewards[0] = 0.0
    return enc_states, enc_rewards


def encode_task_latent(
    encoder,
    task: KitchenTask,
    dataset=None,
    num_samples: int = KITCHEN_ENCODER_SAMPLES,
    seed: int = 0,
    deterministic: bool = True,
    device: str = "cpu",
    env_states: Optional[np.ndarray] = None,
):
    """Encode a Kitchen task into the 128-dim latent ``z`` using the frozen FRE encoder.

    Returns the latent as a torch tensor (or ``None`` if ``torch``/the encoder is unavailable).
    """
    try:
        import torch  # local import keeps the module import-light
    except Exception:  # pragma: no cover
        return None

    rng = np.random.RandomState(seed)
    states, rewards = sample_task_encoder_pairs(
        task, dataset=dataset, num_samples=num_samples, rng=rng, env_states=env_states)

    encoder = encoder.to(device) if hasattr(encoder, "to") else encoder
    encoder.eval() if hasattr(encoder, "eval") else None
    with torch.no_grad():
        s = torch.as_tensor(states, dtype=torch.float32, device=device).unsqueeze(0)
        r = torch.as_tensor(rewards, dtype=torch.float32, device=device).unsqueeze(0)
        if hasattr(encoder, "encode"):
            try:
                z = encoder.encode(s, r, sample=not deterministic)
            except TypeError:
                z = encoder.encode(s, r)
        else:
            dist = encoder(s, r)
            z = dist.mean if deterministic else dist.rsample()
    return z


def make_kitchen_policy_fn(
    agent,
    z,
    deterministic: bool = True,
    device: str = "cpu",
) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a z-conditioned IQL agent into an ``act_fn(obs) -> action`` callable."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("make_kitchen_policy_fn requires torch") from exc

    if z is not None and hasattr(z, "to"):
        z = z.to(device)
    else:
        z = torch.as_tensor(np.asarray(z, dtype=np.float32), device=device).reshape(1, -1) \
            if z is not None else None

    def act_fn(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            o = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=device).unsqueeze(0)
            if hasattr(agent, "select_action"):
                action = agent.select_action(o, z, deterministic=deterministic)
            elif hasattr(agent, "act"):
                action = agent.act(o, z, deterministic=deterministic)
            else:  # pragma: no cover
                raise AttributeError("agent exposes neither select_action nor act")
        action = np.asarray(action.detach().cpu().numpy(), dtype=np.float32).reshape(-1)
        return np.clip(action, -1.0, 1.0)

    return act_fn
