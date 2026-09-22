"""Zero-shot evaluation baselines for FRE (Section 5.2 / Table 1, Table 2).

This package implements the comparison methods used in the paper:

* ``GC-BC``  -- goal-conditioned behavioral cloning (Addendum: 3x512 MLP, ReLU,
  LayerNorm before each activation, Gaussian action head with the log-std clamped
  at ``-5.0``, maximum-likelihood objective, goals sampled geometrically).
* ``GC-IQL`` -- goal-conditioned IQL (Addendum: goal concatenated to the
  observation, HER ratios ``p_random=0.3`` / ``p_geometric=0.5`` / ``p_current=0.2``,
  reward ``-1`` until the goal is reached where it is ``0`` and the mask is ``True``).
* ``OPAL-10`` -- privileged-execution evaluation of an OPAL agent (Addendum: no
  hand-designed rewards; the same transformer architecture as FRE for the encoder;
  at evaluation 10 skills are sampled from a unit Gaussian, each skill conditions
  the policy for an entire episode, and the best performing rollout is taken).
* ``FB`` / ``SF`` -- forward-backward / successor-feature baselines.  The paper
  trains and evaluates these with https://github.com/facebookresearch/controllable_agent,
  so only the evaluation adapter and the protocol constants (5120 context samples)
  live here.

Design notes
------------
Every baseline is exposed through a tiny *agent protocol* so the same evaluation
driver can roll all of them out:

``condition(task, context=None, rng=None) -> np.ndarray``
    Produce the per-task conditioning vector.  GC baselines return the ground-truth
    goal state supplied by the task (Table 2: "GC-BC and GC-IQL are evaluated with
    the ground-truth goal supplied"); OPAL returns a skill ``z`` sampled from a
    unit Gaussian.

``condition_many(task, num_skills, context=None, rng=None) -> np.ndarray``
    Optional; used when ``num_skills > 1`` (OPAL-10).

``act(observation, conditioning, deterministic=True) -> np.ndarray``
    Return an action for one observation.

The driver itself is deliberately duck-typed and pluggable (``env_factory`` /
``envs`` / ``rollout_fn``) so the evaluation protocol can be exercised without
MuJoCo / D4RL installed.

Everything paper-silent (rollout details, best-of-skills bookkeeping, how the
20-episode mean interacts with the 10 skills) is documented inline.
"""

from __future__ import annotations

import json
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------------------
# Protocol constants (Section 5.2, Table 1 caption, Addendum)
# --------------------------------------------------------------------------------------
NUM_EVAL_EPISODES = 20            # "a mean over twenty evaluation episodes"
NUM_TRAINING_SEEDS = 5            # "each agent is trained using five random seeds"
FRE_CONTEXT_SAMPLES = 32          # FRE uses 32 (state, reward) pairs at evaluation
FB_SF_CONTEXT_SAMPLES = 5120      # FB / SF use 5120 samples (Table 1 caption)
NORMALIZED_RETURN_MIN = 0.0
NORMALIZED_RETURN_MAX = 100.0

#: Number of skills used by the privileged OPAL-10 evaluation (Addendum).
DEFAULT_NUM_SKILLS = 10

#: Baselines implemented by the paper.
BASELINE_NAMES: Tuple[str, ...] = (
    "FRE",
    "GC-BC",
    "GC-IQL",
    "OPAL-10",
    "FB",
    "SF",
)

#: Number of (state, reward) context samples each method consumes at test time.
#: ``None`` means the method does not encode a task context from labeled states
#: (the goal-conditioned baselines receive the ground-truth goal instead).
BASELINE_CONTEXT_SAMPLES: Dict[str, Optional[int]] = {
    "FRE": FRE_CONTEXT_SAMPLES,
    "GC-BC": None,
    "GC-IQL": None,
    "OPAL-10": None,
    "FB": FB_SF_CONTEXT_SAMPLES,
    "SF": FB_SF_CONTEXT_SAMPLES,
}

#: ``method name -> (submodule, factory function)`` used by :func:`make_baseline_agent`.
BASELINE_REGISTRY: Dict[str, Tuple[str, str]] = {
    "gc-bc": ("fre.eval.baselines.gc_bc", "make_gc_bc_agent"),
    "gc_iql": ("fre.eval.baselines.gc_iql", "make_gc_iql_agent"),
    "gc-iql": ("fre.eval.baselines.gc_iql", "make_gc_iql_agent"),
    "opal": ("fre.eval.baselines.opal", "make_opal_agent"),
    "opal-10": ("fre.eval.baselines.opal", "make_opal_agent"),
}

DEFAULT_DETERMINISTIC_POLICY = True
DEFAULT_CLIP_SCORES = False

# --------------------------------------------------------------------------------------
# Helpers imported from the zero-shot harness (with light-weight fallbacks so this
# package stays importable stand-alone).
# --------------------------------------------------------------------------------------


def _identity_normalize_return(episode_return, mode="none", score_min=None, score_max=None,
                               succeeded=False, episode_length=None, clip=False):
    return float(episode_return)


_normalize_return_fn: Callable[..., float] = _identity_normalize_return
_resolve_normalization_fn: Optional[Callable[..., Tuple[str, float, float]]] = None
_task_rewards_fn: Optional[Callable[..., np.ndarray]] = None
_task_success_fn: Optional[Callable[..., np.ndarray]] = None
_task_episode_length_fn: Optional[Callable[..., int]] = None
_apply_start_state_fn: Optional[Callable[..., bool]] = None
_clip_action_fn: Optional[Callable[..., np.ndarray]] = None
_as_numpy_2d_fn: Optional[Callable[..., np.ndarray]] = None
ContextSet = Any  # type: ignore[assignment,misc]
TaskResult = None  # type: ignore[assignment]
SuiteResult = None  # type: ignore[assignment]
aggregate_seeds_fn: Optional[Callable[..., Dict[str, Any]]] = None

try:  # pragma: no cover - exercised indirectly
    from fre.eval.zero_shot_eval import (  # type: ignore
        ContextSet as _ContextSet,
        SuiteResult as _SuiteResult,
        TaskResult as _TaskResult,
        aggregate_seeds as _aggregate_seeds,
        apply_start_state as _apply_start_state,
        as_numpy_2d as _as_numpy_2d,
        clip_action as _clip_action,
        normalize_return as _normalize_return,
        resolve_normalization as _resolve_normalization,
        task_episode_length as _task_episode_length,
        task_rewards as _task_rewards,
        task_success as _task_success,
    )

    ContextSet = _ContextSet
    TaskResult = _TaskResult
    SuiteResult = _SuiteResult
    aggregate_seeds_fn = _aggregate_seeds
    _normalize_return_fn = _normalize_return
    _resolve_normalization_fn = _resolve_normalization
    _task_rewards_fn = _task_rewards
    _task_success_fn = _task_success
    _task_episode_length_fn = _task_episode_length
    _apply_start_state_fn = _apply_start_state
    _clip_action_fn = _clip_action
    _as_numpy_2d_fn = _as_numpy_2d
except Exception:  # pragma: no cover - only when the harness is unavailable
    pass


def as_2d(observations: Any) -> np.ndarray:
    """Coerce observations to a 2-D ``float64`` array."""
    if _as_numpy_2d_fn is not None:
        try:
            return _as_numpy_2d_fn(observations)
        except Exception:
            pass
    arr = np.asarray(observations, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def task_reward(task: Any, observations: Any) -> np.ndarray:
    """Evaluate a task's reward function on observations, always returning ``(N,)``."""
    obs = as_2d(observations)
    if _task_rewards_fn is not None:
        try:
            return np.asarray(_task_rewards_fn(task, obs), dtype=np.float64).reshape(-1)
        except Exception:
            pass
    fn = getattr(task, "reward_fn", None)
    if fn is None:
        fn = getattr(task, "reward", None)
    if fn is None:
        raise AttributeError("task object exposes neither 'reward_fn' nor 'reward'")
    out = np.asarray(fn(obs), dtype=np.float64)
    return out.reshape(-1)


def task_succeeded(task: Any, observations: Any, threshold: Optional[float] = None) -> np.ndarray:
    """Boolean success mask for an evaluation task."""
    obs = as_2d(observations)
    if _task_success_fn is not None:
        try:
            return np.asarray(_task_success_fn(task, obs, threshold), dtype=bool).reshape(-1)
        except Exception:
            pass
    fn = getattr(task, "success", None)
    if callable(fn):
        try:
            return np.asarray(fn(obs), dtype=bool).reshape(-1)
        except TypeError:
            return np.asarray(fn(obs, threshold), dtype=bool).reshape(-1)
    reward = task_reward(task, obs)
    reward_max = getattr(task, "reward_max", None)
    if reward_max is None:
        return np.zeros(reward.shape[0], dtype=bool)
    return reward >= float(reward_max) - 1e-8


def episode_length_for(task: Any, default: int = 1000) -> int:
    """Episode length for a task (2000 for AntMaze, 1000 for ExORL / Kitchen)."""
    if _task_episode_length_fn is not None:
        try:
            return int(_task_episode_length_fn(task, default))
        except Exception:
            pass
    return int(getattr(task, "eval_episode_length", default) or default)


def normalize_score(task: Any, episode_return: float, succeeded: bool = False,
                    episode_length: Optional[int] = None, clip: bool = DEFAULT_CLIP_SCORES) -> float:
    """Map an episode return onto the paper's 0-100 scale."""
    if _normalize_return_fn is not None and _resolve_normalization_fn is not None:
        try:
            mode, score_min, score_max = _resolve_normalization_fn(task, None, episode_length)
            return float(_normalize_return_fn(episode_return, mode, score_min, score_max,
                                             succeeded=succeeded,
                                             episode_length=episode_length, clip=clip))
        except Exception:
            pass
    if succeeded:
        return float(NORMALIZED_RETURN_MAX)
    lo = getattr(task, "reward_min", None)
    hi = getattr(task, "reward_max", None)
    if lo is None or hi is None or float(hi) <= float(lo) or not episode_length:
        return float(episode_return)
    raw = (episode_return - float(lo) * episode_length) / ((float(hi) - float(lo)) * episode_length)
    return float(np.clip(raw, 0.0, 1.0) * 100.0)


def clip_action(env: Any, action: Any) -> np.ndarray:
    """Clip an action into the environment's action space."""
    act = np.asarray(action, dtype=np.float64).reshape(-1)
    if _clip_action_fn is not None:
        try:
            return np.asarray(_clip_action_fn(env, act), dtype=np.float64).reshape(-1)
        except Exception:
            pass
    space = getattr(env, "action_space", None)
    if space is not None and hasattr(space, "low") and hasattr(space, "high"):
        return np.clip(act, np.asarray(space.low).reshape(-1), np.asarray(space.high).reshape(-1))
    return act


# --------------------------------------------------------------------------------------
# Agent protocol
# --------------------------------------------------------------------------------------


class BaselineAgent(ABC):
    """Minimal interface every baseline must implement for evaluation."""

    name: str = "baseline"

    #: Number of conditionings evaluated per episode (``10`` for OPAL-10, ``1`` otherwise).
    num_skills: int = 1

    @abstractmethod
    def condition(self, task: Any, context: Any = None, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Return the conditioning vector used by :meth:`act`."""

    @abstractmethod
    def act(self, observation: Any, conditioning: Any, deterministic: bool = True) -> np.ndarray:
        """Return an action for a single observation under ``conditioning``."""

    # -- optional hooks -----------------------------------------------------------------
    def condition_many(self, task: Any, num_skills: int, context: Any = None,
                       rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Return ``num_skills`` conditionings (mean implementation: repeat :meth:`condition`)."""
        single = np.asarray(self.condition(task, context=context, rng=rng), dtype=np.float64).reshape(1, -1)
        return np.repeat(single, int(num_skills), axis=0)

    def context_for(self, task: Any, replay_buffer: Any = None, num_samples: int = FRE_CONTEXT_SAMPLES,
                    rng: Optional[np.random.Generator] = None, use_encoder_inputs: Optional[bool] = None) -> Any:
        """Labeled context set for methods that consume one (FB/SF: 5120 samples)."""
        n = BASELINE_CONTEXT_SAMPLES.get(self.name, None)
        if n is None:
            n = int(num_samples)
        return build_context(task, replay_buffer, num_samples=int(n), rng=rng,
                             use_encoder_inputs=use_encoder_inputs)

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "num_skills": int(self.num_skills)}


class CallableAgent(BaselineAgent):
    """Adapter wrapping plain functions in the :class:`BaselineAgent` protocol.

    Useful for tests and for methods that live outside this codebase (e.g. FB/SF
    checkpoints trained with ``facebookresearch/controllable_agent``).
    """

    def __init__(self, act_fn: Callable[[Any, Any, bool], np.ndarray],
                 condition_fn: Optional[Callable[..., np.ndarray]] = None,
                 name: str = "callable", num_skills: int = 1,
                 condition_many_fn: Optional[Callable[..., np.ndarray]] = None,
                 context_samples: Optional[int] = None):
        self._act_fn = act_fn
        self._condition_fn = condition_fn
        self._condition_many_fn = condition_many_fn
        self.name = name
        self.num_skills = int(num_skills)
        self._context_samples = context_samples

    def condition(self, task, context=None, rng=None):
        if self._condition_fn is None:
            return np.zeros(1, dtype=np.float64)
        return np.asarray(self._condition_fn(task, context, rng), dtype=np.float64).reshape(-1)

    def condition_many(self, task, num_skills, context=None, rng=None):
        if self._condition_many_fn is not None:
            return np.asarray(self._condition_many_fn(task, num_skills, context, rng), dtype=np.float64)
        return super().condition_many(task, num_skills, context=context, rng=rng)

    def act(self, observation, conditioning, deterministic=True):
        return np.asarray(self._act_fn(observation, conditioning, deterministic), dtype=np.float64)

    def context_for(self, task, replay_buffer=None, num_samples=FRE_CONTEXT_SAMPLES, rng=None,
                    use_encoder_inputs=None):
        if self._context_samples is None:
            return super().context_for(task, replay_buffer, num_samples=num_samples, rng=rng,
                                       use_encoder_inputs=use_encoder_inputs)
        return build_context(task, replay_buffer, num_samples=int(self._context_samples), rng=rng,
                             use_encoder_inputs=use_encoder_inputs)


def build_context(task: Any, replay_buffer: Any, num_samples: int = FRE_CONTEXT_SAMPLES,
                  rng: Optional[np.random.Generator] = None,
                  use_encoder_inputs: Optional[bool] = None) -> Any:
    """Sample ``num_samples`` states from the offline dataset labeled by ``task``.

    Section 5.2 / Table 1: FRE uses 32 pairs, FB/SF use 5120.  The states are drawn
    uniformly from the offline dataset, exactly as during FRE encoder training.
    """
    if replay_buffer is None:
        return None
    try:
        from fre.eval.zero_shot_eval import sample_task_context  # type: ignore

        return sample_task_context(replay_buffer, task, num_samples=int(num_samples), rng=rng,
                                  use_encoder_inputs=use_encoder_inputs, source="offline_dataset")
    except Exception:
        pass
    if rng is None:
        rng = np.random.default_rng(0)
    kw: Dict[str, Any] = {"rng": rng}
    if use_encoder_inputs is not None:
        kw["encoder_input"] = use_encoder_inputs
    try:
        states = replay_buffer.sample_states(int(num_samples), **kw)
    except TypeError:
        states = replay_buffer.sample_states(int(num_samples))
    states = as_2d(states)
    rewards = task_reward(task, states)
    return {"states": states, "rewards": rewards, "task": getattr(task, "name", "task"),
            "num_samples": int(num_samples)}


# --------------------------------------------------------------------------------------
# Rollouts
# --------------------------------------------------------------------------------------


@dataclass
class EpisodeRollout:
    """Outcome of a single evaluation rollout."""

    episode_return: float
    length: int
    success: bool
    score: float
    skill_index: int = 0
    final_observation: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_return": float(self.episode_return),
            "length": int(self.length),
            "success": bool(self.success),
            "score": float(self.score),
            "skill_index": int(self.skill_index),
        }


def _reset_env(env: Any, seed: Optional[int] = None, state: Any = None,
               allow_state_reset: bool = True) -> np.ndarray:
    """Reset ``env`` (gym / gymnasium API) optionally to a fixed start state."""
    if state is not None and allow_state_reset:
        if _apply_start_state_fn is not None:
            try:
                if _apply_start_state_fn(env, as_2d(state)[0]):
                    obs = getattr(env, "_last_observation", None)
                    if obs is not None:
                        return as_2d(obs)[0]
            except Exception:
                pass
        for meth in ("reset_to_state", "set_state", "reset_to_observation", "set_observation"):
            fn = getattr(env, meth, None)
            if callable(fn):
                try:
                    out = fn(as_2d(state)[0])
                    if out is not None:
                        return as_2d(out)[0]
                    obs = getattr(env, "_last_observation", None)
                    if obs is not None:
                        return as_2d(obs)[0]
                    break
                except Exception:
                    continue
    try:
        if seed is not None:
            out = env.reset(seed=int(seed))
        else:
            out = env.reset()
    except TypeError:
        out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        out = out[0]
    return as_2d(out)[0]


def _unpack_step(result: Any) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
    """Handle both 4-tuple (gym) and 5-tuple (gymnasium) ``step`` returns."""
    if not (isinstance(result, tuple) and len(result) >= 4):
        raise ValueError("environment.step must return a tuple of length 4 or 5")
    obs = result[0]
    if len(result) == 5:
        reward, terminated, truncated, info = result[1], result[2], result[3], result[4]
        done = bool(terminated) or bool(truncated)
        info = info if isinstance(info, dict) else {}
        info.setdefault("terminated", bool(terminated))
        info.setdefault("truncated", bool(truncated))
    else:
        reward, done, info = result[1], result[2], result[3]
        info = info if isinstance(info, dict) else {}
    return as_2d(obs)[0], float(np.asarray(reward).reshape(-1)[0]), bool(done), info


def _info_success(info: Mapping[str, Any]) -> bool:
    for key in ("success", "is_success", "goal_reached", "task_success"):
        if key in info and np.asarray(info[key]).size:
            if bool(np.asarray(info[key]).reshape(-1)[0]):
                return True
    return False


def rollout_episode(env: Any, task: Any, conditioning: Any,
                    act_fn: Callable[[np.ndarray, Any], np.ndarray],
                    episode_length: Optional[int] = None, seed: Optional[int] = None,
                    initial_state: Any = None, deterministic: bool = True,
                    skill_index: int = 0, allow_state_reset: bool = True,
                    terminate_on_success: Optional[bool] = None) -> EpisodeRollout:
    """Roll one episode of a conditioned baseline policy.

    The return is accumulated from the *evaluation task's* reward function
    ``eta(s)`` (the same reward functions used to label the FRE context), and
    success is read from the task (or the env ``info`` dict for goal tasks).
    """
    length = int(episode_length or episode_length_for(task, 1000))
    obs = _reset_env(env, seed=seed, state=initial_state, allow_state_reset=allow_state_reset)
    total_return = 0.0
    success = False
    steps = 0
    if terminate_on_success is None:
        terminate_on_success = bool(getattr(task, "is_goal_task", False))
    for t in range(length):
        action = act_fn(obs, conditioning)
        action = clip_action(env, action)
        stepped = env.step(action)
        next_obs, _env_reward, done, info = _unpack_step(stepped)
        try:
            step_reward = float(task_reward(task, next_obs)[0])
        except Exception:
            step_reward = 0.0
        total_return += step_reward
        steps = t + 1
        try:
            if bool(task_succeeded(task, next_obs)[0]):
                success = True
        except Exception:
            pass
        if _info_success(info):
            success = True
        obs = next_obs
        if done or (success and terminate_on_success):
            break
    score = normalize_score(task, total_return, succeeded=success, episode_length=length)
    return EpisodeRollout(episode_return=total_return, length=steps, success=success,
                          score=score, skill_index=int(skill_index), final_observation=obs)


def default_rollout_fn(agent: BaselineAgent, task: Any, conditionings: np.ndarray,
                       env: Any, num_episodes: int, episode_length: int, seed: int,
                       deterministic: bool = True, initial_state: Any = None,
                       best_of_skills: bool = True) -> List[EpisodeRollout]:
    """Run ``num_episodes`` rollouts, taking the best skill per episode when >1 skill.

    For OPAL-10 the addendum states: "10 random skills are sampled from a unit
    Gaussian, for each skill ``z`` the policy is conditioned on it and evaluated
    for the entire episode, and the best performing rollout is taken."
    With the paper's 20-episode protocol the natural reading (used here) is that
    each of the 20 evaluation episodes is scored by the best of the sampled skills.
    """
    conds = np.atleast_2d(np.asarray(conditionings, dtype=np.float64))
    results: List[EpisodeRollout] = []
    rng = np.random.default_rng(seed)

    def act_fn(observation, conditioning):
        return agent.act(observation, conditioning, deterministic=deterministic)

    for ep in range(int(num_episodes)):
        ep_seed = int(rng.integers(0, 2 ** 31 - 1))
        best: Optional[EpisodeRollout] = None
        for k in range(conds.shape[0]):
            roll = rollout_episode(env, task, conds[k], act_fn, episode_length=episode_length,
                                   seed=ep_seed, initial_state=initial_state,
                                   deterministic=deterministic, skill_index=k)
            if best is None or (roll.score, roll.episode_return) > (best.score, best.episode_return):
                best = roll
            if not best_of_skills:
                results.append(roll)
        if best_of_skills and best is not None:
            results.append(best)
    return results


# --------------------------------------------------------------------------------------
# Result containers
# --------------------------------------------------------------------------------------


@dataclass
class BaselineTaskResult:
    """Aggregated result for one (method, task, seed)."""

    method: str
    task_name: str
    domain: str = "generic"
    task_group: str = "generic"
    score: float = 0.0
    score_std: float = 0.0
    mean_return: float = 0.0
    return_std: float = 0.0
    success_rate: float = 0.0
    per_episode_scores: List[float] = field(default_factory=list)
    per_episode_returns: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    num_episodes: int = 0
    num_skills: int = 1
    normalization: str = "return_bounds"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "task": self.task_name,
            "task_name": self.task_name,
            "domain": self.domain,
            "task_group": self.task_group,
            "score": float(self.score),
            "score_std": float(self.score_std),
            "mean_return": float(self.mean_return),
            "return_std": float(self.return_std),
            "success_rate": float(self.success_rate),
            "num_episodes": int(self.num_episodes),
            "num_skills": int(self.num_skills),
            "normalization": self.normalization,
            "per_episode_scores": [float(x) for x in self.per_episode_scores],
        }


@dataclass
class BaselineSuiteResult:
    """Results for all tasks of one suite for one seed."""

    suite_name: str
    method: str
    domain: str = "generic"
    task_results: List[BaselineTaskResult] = field(default_factory=list)
    seed: int = 0
    num_episodes: int = NUM_EVAL_EPISODES
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def scores(self) -> List[float]:
        return [float(r.score) for r in self.task_results]

    @property
    def mean_score(self) -> float:
        return float(np.mean(self.scores)) if self.task_results else 0.0

    @property
    def std_score(self) -> float:
        return float(np.std(self.scores)) if self.task_results else 0.0

    def get(self, name: str) -> Optional[BaselineTaskResult]:
        for res in self.task_results:
            if res.task_name == name:
                return res
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite": self.suite_name,
            "method": self.method,
            "domain": self.domain,
            "seed": int(self.seed),
            "mean_score": self.mean_score,
            "std_score": self.std_score,
            "num_episodes": int(self.num_episodes),
            "tasks": {r.task_name: r.to_dict() for r in self.task_results},
        }


@dataclass
class BaselineResult:
    """Aggregate report for a baseline across suites and seeds.

    Provides the same reporting surface as ``EvalReport`` so that
    :func:`fre.utils.logging.aggregate_seeds` can consume ``to_dict()``.
    """

    method: str = "baseline"
    suite_results: List[BaselineSuiteResult] = field(default_factory=list)
    seed: int = 0
    num_episodes: int = NUM_EVAL_EPISODES
    num_context_samples: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- accumulation -------------------------------------------------------------------
    def add(self, result: BaselineSuiteResult) -> "BaselineResult":
        self.suite_results.append(result)
        return self

    def extend(self, results: Iterable[BaselineSuiteResult]) -> "BaselineResult":
        for res in results:
            self.add(res)
        return self

    # -- views --------------------------------------------------------------------------
    def scores(self) -> Dict[str, float]:
        return {r.suite_name: r.mean_score for r in self.suite_results}

    def task_scores(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for suite in self.suite_results:
            for res in suite.task_results:
                out[res.task_name] = float(res.score)
        return out

    @property
    def mean_score(self) -> float:
        vals = [r.mean_score for r in self.suite_results]
        return float(np.mean(vals)) if vals else 0.0

    @property
    def std_score(self) -> float:
        vals = [r.mean_score for r in self.suite_results]
        return float(np.std(vals)) if vals else 0.0

    def summary(self) -> Dict[str, Any]:
        """Flat ``{suite: score}`` / ``{task: score}`` summary."""
        summary: Dict[str, Any] = {"method": self.method,
                                   "mean_score": self.mean_score,
                                   "std_score": self.std_score}
        for suite in self.suite_results:
            summary[suite.suite_name] = suite.mean_score
            for res in suite.task_results:
                summary[f"{suite.suite_name}/{res.task_name}"] = float(res.score)
        return summary

    def format_table(self, digits: int = 1) -> str:
        lines = [f"{self.method} (seed {self.seed})"]
        for suite in self.suite_results:
            lines.append(f"  {suite.suite_name}: {suite.mean_score:.{digits}f} "
                         f"+- {suite.std_score:.{digits}f}")
            for res in suite.task_results:
                lines.append(f"    {res.task_name}: {res.score:.{digits}f}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "seed": int(self.seed),
            "num_episodes": int(self.num_episodes),
            "num_context_samples": self.num_context_samples,
            "mean_score": self.mean_score,
            "std_score": self.std_score,
            "suites": {r.suite_name: r.to_dict() for r in self.suite_results},
            "metadata": dict(self.metadata),
        }

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w") as handle:
            json.dump(self.to_dict(), handle, indent=2, default=float)
        return path


# --------------------------------------------------------------------------------------
# Evaluation driver
# --------------------------------------------------------------------------------------


def _episode_initial_state(task: Any, suite: Any, replay_buffer: Any, rng: Any) -> Any:
    """Start state for an episode (AntMaze: maze center, per Appendix C.1)."""
    for source in (getattr(task, "start_state", None), getattr(suite, "start_state", None)):
        if callable(source):
            try:
                return np.asarray(source(rng=rng), dtype=np.float64).reshape(-1)
            except TypeError:
                try:
                    return np.asarray(source(), dtype=np.float64).reshape(-1)
                except Exception:
                    pass
            except Exception:
                pass
    meta = dict(getattr(task, "metadata", {}) or {})
    if "start_state" in meta:
        return np.asarray(meta["start_state"], dtype=np.float64).reshape(-1)
    return None


def evaluate_baseline_suite(agent: BaselineAgent, suite: Any, seed: int = 0,
                            env_factory: Optional[Callable[..., Any]] = None,
                            envs: Optional[Mapping[str, Any]] = None,
                            rollout_fn: Optional[Callable[..., List[EpisodeRollout]]] = None,
                            replay_buffer: Any = None,
                            num_episodes: int = NUM_EVAL_EPISODES,
                            deterministic: bool = DEFAULT_DETERMINISTIC_POLICY,
                            use_encoder_inputs: Optional[bool] = None,
                            best_of_skills: bool = True) -> BaselineSuiteResult:
    """Evaluate one baseline on every task of a suite for a single seed."""
    rng = np.random.default_rng(seed)
    suite_result = BaselineSuiteResult(
        suite_name=getattr(suite, "name", "suite"),
        method=agent.name,
        domain=getattr(suite, "domain", "generic"),
        seed=int(seed),
        num_episodes=int(num_episodes),
    )
    tasks = list(getattr(suite, "tasks", suite) or [])
    for task in tasks:
        task_name = getattr(task, "name", str(task))
        # Context (only used by methods that consume labeled states: FB / SF / OPAL encoder)
        context = None
        if BASELINE_CONTEXT_SAMPLES.get(agent.name, 0):
            context = agent.context_for(task, replay_buffer=replay_buffer, rng=rng,
                                        use_encoder_inputs=use_encoder_inputs)

        n_skills = int(getattr(agent, "num_skills", 1) or 1)
        if n_skills > 1:
            conditionings = np.atleast_2d(agent.condition_many(task, n_skills, context=context, rng=rng))
        else:
            conditionings = np.asarray(agent.condition(task, context=context, rng=rng),
                                       dtype=np.float64).reshape(1, -1)

        if rollout_fn is not None:
            rolls = list(rollout_fn(agent=agent, task=task, suite=suite, conditionings=conditionings,
                                    seed=int(seed), num_episodes=int(num_episodes),
                                    replay_buffer=replay_buffer, context=context))
        else:
            env = None
            if envs is not None:
                env = envs.get(task_name) or envs.get(getattr(suite, "name", "suite"))
            if env is None and env_factory is not None:
                try:
                    env = env_factory(task, seed=int(seed))
                except TypeError:
                    env = env_factory(task=task, seed=int(seed))
            if env is None:
                # No simulator available: report an empty (NaN) result rather than crashing,
                # so the harness can be exercised in environments without MuJoCo/D4RL.
                suite_result.task_results.append(BaselineTaskResult(
                    method=agent.name, task_name=task_name,
                    domain=getattr(task, "domain", "generic"),
                    task_group=getattr(task, "task_group", "generic"),
                    num_episodes=0, num_skills=n_skills,
                    metadata={"error": "no_env_available"},
                ))
                continue
            length = episode_length_for(task, getattr(suite, "eval_episode_length", 1000))
            rolls = default_rollout_fn(agent, task, conditionings, env,
                                       num_episodes=int(num_episodes), episode_length=length,
                                       seed=int(seed), deterministic=deterministic,
                                       best_of_skills=bool(best_of_skills and n_skills > 1))

        scores = [float(r.score) for r in rolls]
        returns = [float(r.episode_return) for r in rolls]
        successes = [bool(r.success) for r in rolls]
        suite_result.task_results.append(BaselineTaskResult(
            method=agent.name,
            task_name=task_name,
            domain=getattr(task, "domain", "generic"),
            task_group=getattr(task, "task_group", "generic"),
            score=float(np.mean(scores)) if scores else 0.0,
            score_std=float(np.std(scores)) if scores else 0.0,
            mean_return=float(np.mean(returns)) if returns else 0.0,
            return_std=float(np.std(returns)) if returns else 0.0,
            success_rate=float(np.mean(successes)) if successes else 0.0,
            per_episode_scores=scores,
            per_episode_returns=returns,
            episode_lengths=[int(r.length) for r in rolls],
            num_episodes=len(rolls),
            num_skills=n_skills,
        ))
    return suite_result


def evaluate_baseline(agent: BaselineAgent, suites: Sequence[Any], seeds: Sequence[int] = (0,),
                      env_factory: Optional[Callable[..., Any]] = None,
                      envs: Optional[Mapping[str, Any]] = None,
                      rollout_fn: Optional[Callable[..., List[EpisodeRollout]]] = None,
                      replay_buffer: Any = None,
                      num_episodes: int = NUM_EVAL_EPISODES,
                      deterministic: bool = DEFAULT_DETERMINISTIC_POLICY,
                      use_encoder_inputs: Optional[bool] = None,
                      best_of_skills: bool = True,
                      method: Optional[str] = None,
                      save_path: Optional[str] = None) -> List[BaselineResult]:
    """Run the full zero-shot protocol (5 seeds x 20 episodes) for one baseline.

    Returns one :class:`BaselineResult` per seed (mirroring the FRE harness) so
    that :func:`aggregate_baseline_seeds` can produce Table 1 style mean +- std rows.
    """
    reports: List[BaselineResult] = []
    suite_list = list(suites or [])
    for seed in seeds:
        report = BaselineResult(method=method or agent.name, seed=int(seed),
                                num_episodes=int(num_episodes),
                                num_context_samples=BASELINE_CONTEXT_SAMPLES.get(agent.name))
        for suite in suite_list:
            report.add(evaluate_baseline_suite(
                agent, suite, seed=int(seed), env_factory=env_factory, envs=envs,
                rollout_fn=rollout_fn, replay_buffer=replay_buffer, num_episodes=int(num_episodes),
                deterministic=deterministic, use_encoder_inputs=use_encoder_inputs,
                best_of_skills=best_of_skills,
            ))
        reports.append(report)
        if save_path:
            report.save(save_path)
    return reports


def aggregate_baseline_seeds(reports: Sequence[BaselineResult]) -> Dict[str, Dict[str, float]]:
    """Aggregate per-seed baseline reports into ``mean``/``std`` (Table 1 / Table 4 rows)."""
    if not reports:
        return {}
    suite_names: List[str] = []
    task_names: List[str] = []
    per_suite: Dict[str, List[float]] = {}
    per_task: Dict[str, List[float]] = {}
    for report in reports:
        for suite in report.suite_results:
            if suite.suite_name not in suite_names:
                suite_names.append(suite.suite_name)
            per_suite.setdefault(suite.suite_name, []).append(float(suite.mean_score))
            for res in suite.task_results:
                key = f"{suite.suite_name}/{res.task_name}"
                if key not in task_names:
                    task_names.append(key)
                per_task.setdefault(key, []).append(float(res.score))
    out: Dict[str, Dict[str, float]] = {}
    for name in suite_names:
        vals = np.asarray(per_suite[name], dtype=np.float64)
        out[name] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                     "num_seeds": int(vals.size)}
    for name in task_names:
        vals = np.asarray(per_task[name], dtype=np.float64)
        out[name] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                     "num_seeds": int(vals.size)}
    all_vals = np.asarray([r.mean_score for r in reports], dtype=np.float64)
    out["all"] = {"mean": float(np.mean(all_vals)), "std": float(np.std(all_vals)),
                  "num_seeds": int(all_vals.size)}
    return out


def format_baseline_summary(summary: Mapping[str, Mapping[str, float]], digits: int = 1) -> str:
    """Paper-style ``name: mean +- std`` rendering of an aggregated summary."""
    lines = []
    for key, stats in summary.items():
        mean = stats.get("mean", float("nan")) if isinstance(stats, Mapping) else float("nan")
        std = stats.get("std", float("nan")) if isinstance(stats, Mapping) else float("nan")
        lines.append(f"{key}: {mean:.{digits}f} +- {std:.{digits}f}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Lazy access to the concrete baseline implementations
# --------------------------------------------------------------------------------------

_LAZY_ATTRS: Dict[str, str] = {
    "GCBCAgent": "gc_bc",
    "make_gc_bc_agent": "gc_bc",
    "GC_BC_DEFAULTS": "gc_bc",
    "GCIQLAgent": "gc_iql",
    "make_gc_iql_agent": "gc_iql",
    "GC_IQL_DEFAULTS": "gc_iql",
    "OpalAgent": "opal",
    "make_opal_agent": "opal",
    "OPAL_DEFAULTS": "opal",
}


def __getattr__(name: str) -> Any:  # PEP 562
    if name in _LAZY_ATTRS:
        import importlib

        module = importlib.import_module(f"{__name__}.{_LAZY_ATTRS[name]}")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(list(globals().keys()) + list(_LAZY_ATTRS.keys()) + list(__all__)))


def make_baseline_agent(name: str, **kwargs: Any) -> BaselineAgent:
    """Factory resolving a baseline name to its agent implementation."""
    key = str(name).strip().lower()
    if key not in BASELINE_REGISTRY:
        raise ValueError(f"unknown baseline {name!r}; available: {sorted(BASELINE_REGISTRY)}")
    import importlib

    submodule, factory = BASELINE_REGISTRY[key]
    module = importlib.import_module(submodule)
    return getattr(module, factory)(**kwargs)


__all__ = [
    # protocol / containers
    "BaselineAgent",
    "CallableAgent",
    "BaselineResult",
    "BaselineSuiteResult",
    "BaselineTaskResult",
    "EpisodeRollout",
    # driver
    "evaluate_baseline",
    "evaluate_baseline_suite",
    "default_rollout_fn",
    "rollout_episode",
    "aggregate_baseline_seeds",
    "format_baseline_summary",
    "build_context",
    "make_baseline_agent",
    "task_reward",
    "task_succeeded",
    "episode_length_for",
    "normalize_score",
    "clip_action",
    "as_2d",
    # constants
    "BASELINE_NAMES",
    "BASELINE_REGISTRY",
    "BASELINE_CONTEXT_SAMPLES",
    "NUM_EVAL_EPISODES",
    "NUM_TRAINING_SEEDS",
    "FRE_CONTEXT_SAMPLES",
    "FB_SF_CONTEXT_SAMPLES",
    "NORMALIZED_RETURN_MIN",
    "NORMALIZED_RETURN_MAX",
    "DEFAULT_NUM_SKILLS",
    "DEFAULT_DETERMINISTIC_POLICY",
]
