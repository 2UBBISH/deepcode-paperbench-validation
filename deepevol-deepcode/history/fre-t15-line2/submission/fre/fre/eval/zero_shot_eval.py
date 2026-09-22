"""Zero-shot evaluation harness for Functional Reward Encodings (FRE).

Paper: "Zero-Shot Reinforcement Learning via Functional Reward Encodings".

Protocol (Source: Section 5.2 "How does FRE perform on zero-shot offline RL
benchmarks, compared to prior methods?"):

    "All methods are evaluated using a mean over twenty evaluation episodes, and
     each agent is trained using five random seeds, with the standard deviation
     across seeds shown."

and (same section, part 2):

    "... we give these methods 5120 reward samples during evaluation time (in
     comparison to only 32 for FRE)."

So for FRE: the frozen encoder consumes exactly ``K = 32`` reward-labelled states
``(s_k, eta(s_k))`` sampled uniformly from the *unlabeled* offline dataset and
labelled with the *true* reward function ``eta`` of the novel task, produces a
latent ``z`` (the posterior mean is used at evaluation time, since the paper does
not specify whether to sample -- see the inline note), and the frozen policy
``pi(a | s, z)`` is rolled out for the episode without any further training.

The encoder itself is the permutation-invariant transformer described in
§4.1 "Practical Implementation": "K encoder states are sampled uniformly from the
offline dataset, then labeled with a scalar reward according to the given reward
function eta. The resulting reward is discretized according to magnitude into a
learned embedding token space." -- see ``fre/models/encoder.py``; this module only
calls it.

Environment interaction is deliberately duck-typed / pluggable because the paper
does not specify the environment-wrapper code:

* ``env``            -- gym-like object with ``reset()`` / ``step(a)``; the reward
                        used for scoring comes from the *evaluation task* reward
                        function (task reward functions are defined on
                        observations, see ``fre/envs/*_tasks.py``), unless
                        ``reward_source="env"``.
* ``env_factory``    -- callable ``(task, episode_index) -> env`` returning a fresh
                        environment per episode.
* ``rollout_fn``     -- full escape hatch: ``rollout_fn(task=..., z=..., policy=...,
                        episode_index=..., max_steps=..., rng=..., evaluator=...)``
                        for custom simulators (keeps this harness unit-testable
                        without MuJoCo/D4RL).

Defaults where the paper is silent are documented inline with
"Source: not specified in the paper".
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports (tolerant, following the pattern of the rest of the package)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised implicitly
    from fre.envs import EvalTask, TaskSuite  # type: ignore
except Exception:  # pragma: no cover - fallback for direct execution
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _PKG_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
    if _PKG_ROOT not in sys.path:
        sys.path.insert(0, _PKG_ROOT)
    try:
        from fre.envs import EvalTask, TaskSuite  # type: ignore
    except Exception:  # last resort: minimal stand-ins

        @dataclass
        class EvalTask:  # type: ignore
            name: str
            reward_fn: Callable[[Any], np.ndarray]
            domain: str = "generic"
            task_group: str = "generic"
            is_goal_task: bool = False
            goal: Any = None
            threshold: Optional[float] = None
            eval_episode_length: int = 1000
            reward_min: Optional[float] = None
            reward_max: Optional[float] = None
            metadata: Dict[str, Any] = field(default_factory=dict)

            def reward(self, observations):
                return np.asarray(self.reward_fn(observations), dtype=np.float64).reshape(-1)

            def __call__(self, observations):
                return self.reward(observations)

        @dataclass
        class TaskSuite:  # type: ignore
            name: str
            tasks: List[EvalTask] = field(default_factory=list)
            eval_episode_length: int = 1000
            aggregate: Optional[str] = None
            domain: str = "generic"
            metadata: Dict[str, Any] = field(default_factory=dict)

            def __len__(self):
                return len(self.tasks)

            def __iter__(self):
                return iter(self.tasks)

            def __getitem__(self, index):
                return self.tasks[index]

            @property
            def names(self):
                return [t.name for t in self.tasks]

            def get(self, name):
                for t in self.tasks:
                    if t.name == name:
                        return t
                raise KeyError(name)


try:  # pragma: no cover
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


# ---------------------------------------------------------------------------
# Protocol constants (Source: Section 5.2, Table 1 caption)
# ---------------------------------------------------------------------------
NUM_EVAL_EPISODES: int = 20
"""Mean over twenty evaluation episodes (Source: Section 5.2)."""

NUM_TRAINING_SEEDS: int = 5
"""Each agent is trained using five random seeds (Source: Section 5.2)."""

FRE_CONTEXT_SAMPLES: int = 32
"""FRE encodes 32 (state, reward) pairs at evaluation (Source: Section 5.2, Table 1)."""

FB_SF_CONTEXT_SAMPLES: int = 5120
"""FB/SF are given 5120 reward samples at evaluation (Source: Section 5.2)."""

NORMALIZED_RETURN_MIN: float = 0.0
NORMALIZED_RETURN_MAX: float = 100.0
"""Results are normalized between 0 and 100 (Source: Table 1 caption)."""

DEFAULT_EVAL_EPISODE_LENGTH: int = 1000
"""Fallback episode length; AntMaze uses 2000 and ExORL uses 1000 (Source: C.1/C.2)."""

DEFAULT_DETERMINISTIC_POLICY: bool = True
"""Evaluation uses the mean action (Source: not specified in the paper)."""

DEFAULT_USE_POSTERIOR_MEAN: bool = True
"""Encode z as the posterior mean at eval time (Source: not specified in the paper)."""

DEFAULT_REWARD_RANGE_MODE: str = "task"
"""How the eval-time reward discretization range is chosen.

``"task"``     -- use the evaluated task's own reward bounds (``reward_min`` /
                  ``reward_max``, or the min/max observed over the 32 context
                  rewards) so all 32 reward bins are used, mirroring the training
                  priors which bind a per-``eta`` range (``RewardDiscretizer.
                  fit_from_function``).
``"encoder"``  -- use the range the encoder was trained/configured with.
Source: not specified in the paper.
"""

NORMALIZATION_MODES = ("return_bounds", "success", "none", "per_step_bounds")


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------
@dataclass
class ContextSet:
    """The ``K`` reward-labelled states handed to the encoder for one task."""

    states: np.ndarray                 # (K, state_dim) -- encoder input representation
    rewards: np.ndarray                # (K,)
    task_name: str = ""
    reward_min: Optional[float] = None
    reward_max: Optional[float] = None
    source: str = "offline_dataset"
    num_samples: int = FRE_CONTEXT_SAMPLES

    def __post_init__(self) -> None:
        self.states = np.asarray(self.states, dtype=np.float32)
        self.rewards = np.asarray(self.rewards, dtype=np.float32).reshape(-1)
        if self.states.ndim == 1:
            self.states = self.states[None, :]
        self.num_samples = int(self.states.shape[0])

    def reward_range(self) -> Tuple[float, float]:
        lo, hi = self.reward_min, self.reward_max
        if lo is None or hi is None or not np.isfinite([lo, hi]).all():
            lo = float(np.min(self.rewards))
            hi = float(np.max(self.rewards))
        if hi <= lo:  # degenerate context (e.g. every context reward == -1)
            lo, hi = float(lo) - 1.0, float(hi) + 1.0
        return float(lo), float(hi)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_name": self.task_name,
            "num_samples": int(self.num_samples),
            "state_dim": int(self.states.shape[-1]),
            "reward_min": float(self.reward_range()[0]),
            "reward_max": float(self.reward_range()[1]),
            "mean_reward": float(np.mean(self.rewards)) if self.rewards.size else 0.0,
            "source": self.source,
        }


@dataclass
class EpisodeResult:
    """Outcome of a single evaluation episode."""

    episode_return: float
    length: int
    success: bool = False
    score: Optional[float] = None
    final_observation: Optional[np.ndarray] = None
    task_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_return": float(self.episode_return),
            "length": int(self.length),
            "success": bool(self.success),
            "score": None if self.score is None else float(self.score),
            "task_name": self.task_name,
        }


@dataclass
class TaskResult:
    """Aggregated zero-shot result for one evaluation task (one seed)."""

    task_name: str
    task_group: str = ""
    domain: str = ""
    score: float = 0.0
    score_std: float = 0.0
    mean_return: float = 0.0
    return_std: float = 0.0
    per_episode_scores: List[float] = field(default_factory=list)
    per_episode_returns: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    num_episodes: int = 0
    normalization: str = "return_bounds"
    score_bounds: Tuple[float, float] = (0.0, 1.0)
    reward_min: Optional[float] = None
    reward_max: Optional[float] = None
    context: Optional[Dict[str, Any]] = None
    latent_norm: Optional[float] = None
    decoder_mse: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        if not self.per_episode_scores:
            return 0.0
        return float(np.mean(self.per_episode_scores)) / NORMALIZED_RETURN_MAX

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["score_bounds"] = list(self.score_bounds)
        return out


@dataclass
class SuiteResult:
    """All task results for one suite (one seed)."""

    suite_name: str
    domain: str = ""
    task_results: List[TaskResult] = field(default_factory=list)
    mean_score: float = 0.0
    std_score: float = 0.0
    num_context_samples: int = FRE_CONTEXT_SAMPLES
    num_episodes: int = NUM_EVAL_EPISODES
    seed: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def scores(self) -> Dict[str, float]:
        return {r.task_name: float(r.score) for r in self.task_results}

    def get(self, name: str) -> TaskResult:
        for r in self.task_results:
            if r.task_name == name:
                return r
        raise KeyError(name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "domain": self.domain,
            "mean_score": float(self.mean_score),
            "std_score": float(self.std_score),
            "scores": self.scores,
            "num_context_samples": int(self.num_context_samples),
            "num_episodes": int(self.num_episodes),
            "seed": int(self.seed),
            "tasks": [r.to_dict() for r in self.task_results],
            "metadata": _jsonable(self.metadata),
        }


@dataclass
class EvalReport:
    """Full evaluation report across suites / tasks (optionally several seeds)."""

    suite_results: Dict[str, SuiteResult] = field(default_factory=dict)
    seed: Optional[int] = None
    method: str = "FRE"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add(self, result: SuiteResult) -> SuiteResult:
        self.suite_results[result.suite_name] = result
        return result

    def scores(self) -> Dict[str, float]:
        """Flat ``{suite/task: score}`` mapping."""
        out: Dict[str, float] = {}
        for suite_name, suite in self.suite_results.items():
            for task_name, score in suite.scores.items():
                out[f"{suite_name}/{task_name}"] = float(score)
        return out

    def summary(self) -> Dict[str, float]:
        """Suite-level (and per-task) summary, in the style of the paper's tables."""
        summary: Dict[str, float] = {}
        for suite_name, suite in self.suite_results.items():
            summary[suite_name] = float(suite.mean_score)
            for task_name, score in suite.scores.items():
                summary[f"{suite_name}/{task_name}"] = float(score)
        return summary

    def format_table(self, digits: int = 1) -> str:
        lines = []
        for suite_name, suite in self.suite_results.items():
            lines.append(f"{suite_name}: {suite.mean_score:.{digits}f} "
                         f"+/- {suite.std_score:.{digits}f}")
            for task in suite.task_results:
                lines.append(
                    f"    {task.task_name:<32s} {task.score:>{digits + 3}.{digits}f} "
                    f"+/- {task.score_std:.{digits}f}   "
                    f"(return {task.mean_return:.2f}, n={task.num_episodes}, "
                    f"norm={task.normalization})"
                )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "seed": self.seed,
            "summary": {k: float(v) for k, v in self.summary().items()},
            "suites": {k: v.to_dict() for k, v in self.suite_results.items()},
            "metadata": _jsonable(self.metadata),
        }

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return path


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def as_numpy_2d(observations: Any) -> np.ndarray:
    """Coerce observations/tensors to a 2-D float array (rows = states)."""
    if torch is not None and isinstance(observations, torch.Tensor):
        observations = observations.detach().cpu().numpy()
    arr = np.asarray(observations, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def task_rewards(task: Any, observations: Any) -> np.ndarray:
    """Evaluate a task's reward function on observations, always returning ``(N,)``."""
    fn = getattr(task, "reward", None) or getattr(task, "reward_fn", None)
    if fn is None:  # pragma: no cover - defensive
        raise TypeError(f"task {task!r} exposes neither 'reward' nor 'reward_fn'")
    out = np.asarray(fn(observations), dtype=np.float64)
    return out.reshape(-1)


def task_success(task: Any, observations: Any, threshold: Optional[float] = None) -> np.ndarray:
    """Boolean success mask for a task, if the task exposes a success predicate."""
    fn = getattr(task, "success", None)
    if callable(fn):
        try:
            out = np.asarray(fn(observations) if threshold is None else fn(observations, threshold))
            return out.reshape(-1).astype(bool)
        except TypeError:  # pragma: no cover - signature mismatch
            pass
    # Fallback: a goal-style task "succeeds" where the reward is at its maximum.
    rewards = task_rewards(task, observations)
    rmax = getattr(task, "reward_max", None)
    if rmax is None:
        return rewards > 0.0
    return rewards >= float(rmax) - 1e-8


def task_episode_length(task: Any, default: Optional[int] = None) -> int:
    length = getattr(task, "eval_episode_length", None)
    if length is None:
        length = default if default is not None else DEFAULT_EVAL_EPISODE_LENGTH
    return int(length)


# ---------------------------------------------------------------------------
# Context construction: 32 uniformly sampled (state, eta(state)) pairs
# ---------------------------------------------------------------------------
def sample_task_context(
    replay_buffer: Any,
    task: Any,
    num_samples: int = FRE_CONTEXT_SAMPLES,
    rng: Optional[np.random.Generator] = None,
    use_encoder_inputs: Optional[bool] = None,
    reward_fn: Optional[Callable[[Any], np.ndarray]] = None,
    source: str = "offline_dataset",
) -> ContextSet:
    """Sample ``num_samples`` states uniformly from ``D`` and label them with ``eta``.

    The paper samples the ``K`` encoder states uniformly from the offline dataset
    and labels them with the scalar reward of the given reward function
    (Source: Section 4.1 "Practical Implementation": "K encoder states are sampled
    uniformly from the offline dataset, then labeled with a scalar reward according
    to the given reward function eta"), and FRE uses exactly 32 such pairs at
    evaluation (Source: Section 5.2).

    For ExORL the encoder consumes physics-augmented observations while the value
    functions/policy do not (Source: C.2), so ``use_encoder_inputs`` defaults to
    ``True`` whenever the buffer carries a distinct encoder observation space.
    """
    rng = np.random.default_rng() if rng is None else rng
    if use_encoder_inputs is None:
        use_encoder_inputs = int(getattr(replay_buffer, "encoder_obs_dim",
                                         getattr(replay_buffer, "obs_dim", 0))) != int(
            getattr(replay_buffer, "obs_dim", 0))

    states = replay_buffer.sample_states(int(num_samples), rng=rng,
                                         encoder_input=bool(use_encoder_inputs))
    states = as_numpy_2d(states)

    labeller = reward_fn if reward_fn is not None else (lambda obs: task_rewards(task, obs))
    rewards = np.asarray(labeller(states), dtype=np.float32).reshape(-1)

    reward_min = getattr(task, "reward_min", None)
    reward_max = getattr(task, "reward_max", None)
    try:  # tasks may publish bounds in metadata (e.g. AntMaze score_bounds)
        bounds = (getattr(task, "metadata", {}) or {}).get("reward_bounds")
        if bounds is not None and len(bounds) == 2:
            reward_min, reward_max = float(bounds[0]), float(bounds[1])
    except Exception:  # pragma: no cover
        pass

    return ContextSet(
        states=states,
        rewards=rewards,
        task_name=getattr(task, "name", ""),
        reward_min=None if reward_min is None else float(reward_min),
        reward_max=None if reward_max is None else float(reward_max),
        source=source,
    )


def encode_context(
    encoder: Any,
    context: ContextSet,
    sample: bool = not DEFAULT_USE_POSTERIOR_MEAN,
    device: Any = None,
    reward_range_mode: str = DEFAULT_REWARD_RANGE_MODE,
) -> Any:
    """Encode ``(s_k, eta(s_k))`` pairs into the latent ``z`` with the frozen encoder.

    Returns the mean (default) or a sample of ``p_theta(z | .)``.  The posterior
    mean is used at evaluation because the paper does not state which is used
    (Source: not specified in the paper).
    """
    if torch is None:  # pragma: no cover - torch is a hard requirement in practice
        raise ImportError("torch is required for FRE evaluation")

    device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    states = torch.as_tensor(as_numpy_2d(context.states)[None, ...], dtype=torch.float32, device=device)
    rewards = torch.as_tensor(np.asarray(context.rewards, dtype=np.float32)[None, ...],
                              dtype=torch.float32, device=device)

    kwargs: Dict[str, Any] = {"sample": bool(sample)}
    if reward_range_mode in ("task", "per_step_bounds", "auto"):
        lo, hi = context.reward_range()
        kwargs["reward_min"] = lo
        kwargs["reward_max"] = hi
    # else: leave the encoder's own configured range in place ("encoder" mode)

    if hasattr(encoder, "eval"):
        encoder.eval()
    with torch.no_grad():
        try:
            out = encoder(states, rewards, **kwargs)
        except TypeError:  # pragma: no cover - signature fallback
            out = encoder(states, rewards)
    z = getattr(out, "z", out)
    if isinstance(z, (tuple, list)):
        z = z[0]
    return z


def context_reconstruction_error(decoder: Any, context: ContextSet, z: Any,
                                 device: Any = None) -> Optional[float]:
    """Diagnostic: MSE of ``q_theta(eta(s)|s,z)`` on the encoding context (optional)."""
    if decoder is None or torch is None:
        return None
    try:
        device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        states = torch.as_tensor(as_numpy_2d(context.states)[None, ...], dtype=torch.float32,
                                 device=device)
        targets = torch.as_tensor(np.asarray(context.rewards, dtype=np.float32)[None, ...],
                                  dtype=torch.float32, device=device)
        if isinstance(z, torch.Tensor):
            z_t = z.to(device)
        else:  # pragma: no cover
            z_t = torch.as_tensor(np.asarray(z), dtype=torch.float32, device=device)
        with torch.no_grad():
            out = decoder(states, z_t, return_output=True)
        return float(out.mse(targets, reduction="mean").detach().cpu())
    except Exception:  # pragma: no cover - diagnostics must never break evaluation
        return None


# ---------------------------------------------------------------------------
# Score normalization to [0, 100]
# ---------------------------------------------------------------------------
def resolve_normalization(task: Any, suite: Optional[Any] = None,
                          episode_length: Optional[int] = None) -> Tuple[str, float, float]:
    """Return ``(mode, score_min, score_max)`` for a task's episode return.

    The paper only states that results are normalized between 0 and 100
    (Source: Table 1 caption); the exact normalization per domain is not given.
    Defaults implemented here:

    * ``return_bounds`` -- ``score = 100 * (R - R_min) / (R_max - R_min)`` with
      ``R_min/R_max`` from the task/suite metadata (``return_min``/``return_max``
      or a ``score_bounds`` pair) or, failing that,
      ``[reward_min * L, reward_max * L]`` (e.g. AntMaze goal-reaching with
      ``-1/0`` rewards over ``L = 2000`` steps gives ``[-2000, 0]``).
    * ``success`` -- ``100`` if the task was achieved during the episode else
      ``0`` (used for non-negative sparse rewards such as the Kitchen subtasks,
      whose paper metric is a percentage of subtasks solved).
    * ``none`` -- the raw episode return is the score.

    Source: not specified in the paper (documented default).
    """
    length = int(episode_length or task_episode_length(task, suite_eval_length(suite)))

    meta: Dict[str, Any] = {}
    for source in (getattr(suite, "metadata", None), getattr(task, "metadata", None)):
        if isinstance(source, dict):
            meta.update({k: v for k, v in source.items() if k not in meta})

    # 1) explicit mode
    mode = None
    for key in ("normalization", "score_normalization", "score_mode"):
        if meta.get(key) is not None:
            mode = str(meta[key])
            break
    aliases = {
        "return": "return_bounds", "returns": "return_bounds",
        "return_bounds": "return_bounds", "bounds": "return_bounds",
        "success": "success", "binary": "success", "success_rate": "success",
        "none": "none", "raw": "none",
        "per_step_bounds": "per_step_bounds",
    }
    if mode is not None and mode.lower() in aliases:
        mode = aliases[mode.lower()]
    else:
        mode = None

    # 2) explicit bounds
    lo: Optional[float] = None
    hi: Optional[float] = None
    for key_lo, key_hi in (("return_min", "return_max"), ("min_total", "max_total"),
                           ("score_min", "score_max")):
        if meta.get(key_lo) is not None and meta.get(key_hi) is not None:
            lo, hi = float(meta[key_lo]), float(meta[key_hi])
            break
    if lo is None:
        bounds = meta.get("score_bounds")
        if isinstance(bounds, dict):
            for key_lo, key_hi in (("return_min", "return_max"), ("min", "max"),
                                   ("min_total", "max_total")):
                if bounds.get(key_lo) is not None and bounds.get(key_hi) is not None:
                    lo, hi = float(bounds[key_lo]), float(bounds[key_hi])
                    if key_lo == "min" and abs(lo) <= 1.5 and abs(hi) <= 1.5:
                        lo, hi = lo * length, hi * length  # per-step bounds
                    break
        elif isinstance(bounds, (tuple, list)) and len(bounds) == 2:
            lo, hi = float(bounds[0]), float(bounds[1])
    if lo is None:
        r_min = getattr(task, "reward_min", None)
        r_max = getattr(task, "reward_max", None)
        if r_min is not None and r_max is not None:
            if mode == "per_step_bounds":
                lo, hi = float(r_min), float(r_max)
            else:
                lo, hi = float(r_min) * length, float(r_max) * length

    if mode is None:
        if lo is not None and hi is not None and hi > lo:
            mode = "return_bounds"
        elif getattr(task, "is_goal_task", False):
            mode = "return_bounds"
        else:
            r_min = getattr(task, "reward_min", None)
            r_max = getattr(task, "reward_max", None)
            if r_min is not None and r_max is not None and float(r_min) >= 0.0 and float(r_max) <= 1.0:
                mode = "success"
            else:
                mode = "return_bounds"

    if mode in ("return_bounds", "per_step_bounds"):
        if lo is None or hi is None or not np.isfinite([lo, hi]).all() or hi <= lo:
            r_min = getattr(task, "reward_min", None)
            r_max = getattr(task, "reward_max", None)
            r_min = -1.0 if r_min is None else float(r_min)
            r_max = 0.0 if r_max is None else float(r_max)
            if mode == "per_step_bounds":
                lo, hi = r_min, r_max
            else:
                lo, hi = r_min * length, r_max * length
        if hi <= lo:  # pragma: no cover - degenerate
            lo, hi = 0.0, 1.0

    if lo is None:
        lo, hi = 0.0, 1.0
    return str(mode), float(lo), float(hi)


def suite_eval_length(suite: Optional[Any]) -> Optional[int]:
    if suite is None:
        return None
    return getattr(suite, "eval_episode_length", None)


def normalize_return(episode_return: float, mode: str, score_min: float, score_max: float,
                     succeeded: bool = False, episode_length: Optional[int] = None,
                     clip: bool = False) -> float:
    """Map an episode return (or success flag) to the paper's 0-100 scale."""
    if mode == "success":
        score = NORMALIZED_RETURN_MAX if succeeded else NORMALIZED_RETURN_MIN
    elif mode == "none":
        score = float(episode_return)
    elif mode == "per_step_bounds":
        length = int(episode_length or 1)
        denom = (score_max - score_min)
        per_step = episode_return / max(length, 1)
        score = NORMALIZED_RETURN_MAX * (per_step - score_min) / denom if denom > 1e-12 else 0.0
    else:  # return_bounds
        denom = float(score_max - score_min)
        score = NORMALIZED_RETURN_MAX * (float(episode_return) - score_min) / denom if denom > 1e-12 else 0.0
    if clip:
        score = float(np.clip(score, NORMALIZED_RETURN_MIN, NORMALIZED_RETURN_MAX))
    return float(score)


# ---------------------------------------------------------------------------
# Rollout machinery
# ---------------------------------------------------------------------------
def _extract_step(step_out: Any) -> Tuple[Any, float, bool, Dict[str, Any]]:
    """Normalize gym/gymnasium step returns into ``(obs, reward, done, info)``."""
    info: Dict[str, Any] = {}
    if isinstance(step_out, tuple):
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            return obs, float(reward), bool(terminated or truncated), dict(info or {})
        if len(step_out) == 4:
            obs, reward, done, info = step_out
            return obs, float(reward), bool(done), dict(info or {})
        if len(step_out) == 3:
            obs, reward, done = step_out
            return obs, float(reward), bool(done), {}
        if len(step_out) == 2:  # pragma: no cover - custom envs
            obs, reward = step_out
            return obs, float(reward), False, {}
    return step_out, 0.0, False, info


def _extract_reset(reset_out: Any) -> Tuple[Any, Dict[str, Any]]:
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        return reset_out[0], dict(reset_out[1] or {})
    return reset_out, {}


def apply_start_state(env: Any, state: Any) -> bool:
    """Best-effort initialization of the environment to a given observation.

    Used for the paper's AntMaze protocol where "the ant robot is placed in the
    center of the maze" (Source: C.1) -- the actual simulator call is
    environment-specific, so every plausible hook is tried in turn.
    """
    if state is None:
        return False
    candidates = (
        ("reset_to_state", (state,)),
        ("set_state", (state,)),
        ("reset_to_observation", (state,)),
        ("set_observation", (state,)),
    )
    for name, args in candidates:
        fn = getattr(env, name, None)
        if callable(fn):
            try:
                fn(*args)
                return True
            except Exception:  # pragma: no cover - keep trying hooks
                continue
    try:
        out = env.reset(state=state)  # gymnasium-style
        _extract_reset(out)
        return True
    except Exception:
        pass
    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is not None and unwrapped is not env:  # pragma: no cover
        set_state = getattr(unwrapped, "set_state", None)
        if callable(set_state):
            try:
                set_state(state)
                return True
            except Exception:
                return False
    return False


def policy_action(policy: Any, observation: np.ndarray, z: Any,
                  deterministic: bool = True, device: Any = None) -> np.ndarray:
    """Query the z-conditioned policy for a single action (env action vector)."""
    if torch is None:  # pragma: no cover
        raise ImportError("torch is required for FRE evaluation")
    if device is None:
        device = z.device if isinstance(z, torch.Tensor) else "cpu"
    obs_t = torch.as_tensor(np.asarray(observation, dtype=np.float32)[None, :],
                            dtype=torch.float32, device=device)
    z_t = z if isinstance(z, torch.Tensor) else torch.as_tensor(
        np.asarray(z), dtype=torch.float32, device=device)
    if z_t.ndim == 1:
        z_t = z_t[None, :]

    action = None
    with torch.no_grad():
        for name in ("acts", "act", "action", "select_action"):
            fn = getattr(policy, name, None)
            if callable(fn):
                try:
                    out = fn(obs_t, z_t, deterministic=deterministic)
                except TypeError:
                    try:
                        out = fn(obs_t, z_t)
                    except TypeError:
                        continue
                action = getattr(out, "action", out)
                break
        if action is None:
            # RLNetworks exposes the policy as `.policy`
            inner = getattr(policy, "policy", None)
            if inner is not None and inner is not policy:
                return policy_action(inner, observation, z, deterministic=deterministic,
                                     device=device)
            if callable(policy):
                out = policy(obs_t, z_t)
                action = getattr(out, "action", out)
            else:  # pragma: no cover
                raise TypeError(f"policy {policy!r} exposes no action interface")

    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()
    return np.asarray(action, dtype=np.float32).reshape(-1)


def clip_action(env: Any, action: np.ndarray) -> np.ndarray:
    space = getattr(env, "action_space", None)
    if space is None:
        return action
    low = getattr(space, "low", None)
    high = getattr(space, "high", None)
    if low is None or high is None:
        return action
    low = np.asarray(low, dtype=np.float32).reshape(-1)
    high = np.asarray(high, dtype=np.float32).reshape(-1)
    if low.size == action.size:
        return np.clip(action, low, high)
    return action


class ZeroShotEvaluator:
    """Encodes 32 reward-labelled states into ``z`` and rolls out ``pi(a|s,z)``.

    Parameters
    ----------
    encoder : frozen FRE encoder (``fre.models.encoder.FREEncoder``).
    policy  : ``RLNetworks`` (or a bare ``PolicyNetwork``) conditioned on ``z``.
    replay_buffer : unlabeled offline buffer used to sample the 32 context states.
    decoder : optional reward decoder, used only for a reconstruction diagnostic.
    device : torch device (defaults to CUDA when available).
    num_context_samples : ``K`` at evaluation; 32 for FRE (Source: Section 5.2).
    num_episodes : 20 (Source: Section 5.2).
    seeds : the five training seeds the paper averages over (Source: Section 5.2).
    """

    def __init__(
        self,
        encoder: Any,
        policy: Any,
        replay_buffer: Any,
        decoder: Any = None,
        device: Any = None,
        num_context_samples: int = FRE_CONTEXT_SAMPLES,
        num_episodes: int = NUM_EVAL_EPISODES,
        seeds: Sequence[int] = (0, 1, 2, 3, 4),
        deterministic_policy: bool = DEFAULT_DETERMINISTIC_POLICY,
        use_encoder_inputs: Optional[bool] = None,
        obs_transform: Optional[Callable[[Any], Any]] = None,
        reward_range_mode: str = DEFAULT_REWARD_RANGE_MODE,
        clip_scores: bool = False,
        sample_latent: bool = not DEFAULT_USE_POSTERIOR_MEAN,
        record_observations: bool = False,
        name: str = "FRE",
    ) -> None:
        self.encoder = encoder
        self.policy = policy
        self.replay_buffer = replay_buffer
        self.decoder = decoder
        self.device = device if device is not None else (
            "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu")
        self.num_context_samples = int(num_context_samples)
        self.num_episodes = int(num_episodes)
        self.seeds = tuple(int(s) for s in seeds)
        self.deterministic_policy = bool(deterministic_policy)
        self.use_encoder_inputs = use_encoder_inputs
        self.obs_transform = obs_transform
        self.reward_range_mode = reward_range_mode
        self.clip_scores = bool(clip_scores)
        self.sample_latent = bool(sample_latent)
        self.record_observations = bool(record_observations)
        self.name = name

    # -- context / latent ----------------------------------------------------
    def task_context(self, task: Any, num_samples: Optional[int] = None,
                     rng: Optional[np.random.Generator] = None,
                     reward_fn: Optional[Callable[[Any], np.ndarray]] = None) -> ContextSet:
        return sample_task_context(
            self.replay_buffer, task,
            num_samples=self.num_context_samples if num_samples is None else int(num_samples),
            rng=rng,
            use_encoder_inputs=self.use_encoder_inputs,
            reward_fn=reward_fn,
        )

    def encode_task(self, task: Any, rng: Optional[np.random.Generator] = None,
                    context: Optional[ContextSet] = None,
                    num_samples: Optional[int] = None) -> Tuple[Any, ContextSet]:
        if context is None:
            context = self.task_context(task, num_samples=num_samples, rng=rng)
        z = encode_context(self.encoder, context, sample=self.sample_latent, device=self.device,
                           reward_range_mode=self.reward_range_mode)
        return z, context

    # -- episode -------------------------------------------------------------
    def run_episode(
        self,
        env: Any,
        task: Any,
        z: Any,
        episode_index: int = 0,
        max_steps: Optional[int] = None,
        start_state: Any = None,
        rng: Optional[np.random.Generator] = None,
        reward_fn: Optional[Callable[[Any], np.ndarray]] = None,
        reward_source: str = "task",
    ) -> EpisodeResult:
        """Roll out the z-conditioned policy for a single episode.

        Rewards are taken from the evaluation task's reward function by default,
        because the paper's tasks (goal-reaching, directional, simplex, velocity,
        Kitchen subtasks) are defined on observations (Source: C.1/C.2/C.3).
        """
        length = int(max_steps or task_episode_length(task))
        if reward_fn is None:
            reward_fn = lambda obs: task_rewards(task, obs)  # noqa: E731

        reset_out = env.reset()
        obs, _ = _extract_reset(reset_out)
        if start_state is not None:
            apply_start_state(env, start_state)

        total = 0.0
        success = False
        observations: List[np.ndarray] = []
        step = 0
        for step in range(length):
            obs_used = self._transform_obs(obs)
            if self.record_observations:
                observations.append(np.asarray(obs_used, dtype=np.float32).copy())
            action = policy_action(self.policy, as_numpy_2d(obs_used)[0], z,
                                   deterministic=self.deterministic_policy, device=self.device)
            action = clip_action(env, action)
            step_out = env.step(action)
            next_obs, env_reward, done, info = _extract_step(step_out)
            obs_for_reward = self._transform_obs(next_obs)
            if reward_source == "env":
                step_reward = float(env_reward)
            else:
                step_reward = float(np.sum(reward_fn(as_numpy_2d(obs_for_reward))))

            done_task = self._task_done(task, obs_for_reward, info)
            if self._task_succeeded(task, obs_for_reward, info):
                success = True
            total += step_reward
            obs = next_obs
            if done or done_task:
                break

        return EpisodeResult(
            episode_return=float(total),
            length=int(step + 1) if length > 0 else 0,
            success=bool(success),
            final_observation=(as_numpy_2d(obs)[0] if (self.record_observations and length > 0)
                               else None),
            task_name=getattr(task, "name", ""),
        )

    # -- helpers -------------------------------------------------------------
    def _transform_obs(self, obs: Any) -> Any:
        if self.obs_transform is None:
            return obs
        return self.obs_transform(obs)

    @staticmethod
    def _task_succeeded(task: Any, observation: Any, info: Dict[str, Any]) -> bool:
        if info:
            for key in ("success", "is_success", "goal_reached"):
                if info.get(key) is not None:
                    return bool(np.any(np.asarray(info[key])))
        try:
            return bool(np.any(task_success(task, as_numpy_2d(observation))))
        except Exception:  # pragma: no cover
            return False

    @staticmethod
    def _task_done(task: Any, observation: Any, info: Dict[str, Any]) -> bool:
        fn = getattr(task, "done", None)
        if callable(fn):
            try:
                return bool(np.any(np.asarray(fn(as_numpy_2d(observation)))))
            except TypeError:
                try:
                    return bool(np.any(np.asarray(fn(observation))))
                except Exception:  # pragma: no cover
                    pass
            except Exception:  # pragma: no cover
                pass
        return False

    # -- task / suite evaluation --------------------------------------------
    def evaluate_task(
        self,
        task: Any,
        suite: Optional[Any] = None,
        env: Any = None,
        env_factory: Optional[Callable[..., Any]] = None,
        rollout_fn: Optional[Callable[..., EpisodeResult]] = None,
        num_episodes: Optional[int] = None,
        max_steps: Optional[int] = None,
        seed: int = 0,
        start_states: Any = None,
        reward_fn: Optional[Callable[[Any], np.ndarray]] = None,
        reward_source: str = "task",
        task_index: int = 0,
        context: Optional[ContextSet] = None,
    ) -> TaskResult:
        """Zero-shot evaluation of one task: encode 32 pairs, then roll out 20 episodes."""
        num_episodes = int(self.num_episodes if num_episodes is None else num_episodes)
        rng = np.random.default_rng(int(seed) * 10_000 + int(task_index))
        z, context = self.encode_task(task, rng=rng, context=context)
        length = int(max_steps or task_episode_length(task, suite_eval_length(suite)))
        mode, score_min, score_max = resolve_normalization(task, suite, episode_length=length)

        episodes: List[EpisodeResult] = []
        for ep in range(num_episodes):
            start_state = _pick_start_state(start_states, ep)
            if rollout_fn is not None:
                episode = rollout_fn(
                    task=task, z=z, policy=self.policy, episode_index=ep,
                    max_steps=length,
                    rng=np.random.default_rng(
                        int(seed) * 10_000 + int(task_index) * 100 + ep),
                    evaluator=self,
                )
            else:
                episode_env = env
                if episode_env is None and env_factory is not None:
                    episode_env = env_factory(task, ep)
                if episode_env is None:  # pragma: no cover - configuration error
                    raise ValueError(
                        "evaluate_task requires one of 'env', 'env_factory' or 'rollout_fn'")
                episode = self.run_episode(
                    episode_env, task, z, episode_index=ep,
                    max_steps=length, start_state=start_state, rng=rng,
                    reward_fn=reward_fn, reward_source=reward_source,
                )
            episode.score = normalize_return(
                episode.episode_return, mode, score_min, score_max,
                succeeded=episode.success,
                episode_length=episode.length, clip=self.clip_scores,
            )
            episodes.append(episode)

        returns = np.asarray([e.episode_return for e in episodes], dtype=np.float64)
        scores = np.asarray([float(e.score) for e in episodes], dtype=np.float64)
        lengths = [int(e.length) for e in episodes]

        return TaskResult(
            task_name=getattr(task, "name", f"task_{task_index}"),
            task_group=getattr(task, "task_group", ""),
            domain=getattr(task, "domain", ""),
            score=float(np.mean(scores)) if scores.size else 0.0,
            score_std=float(np.std(scores, ddof=0)) if scores.size else 0.0,
            mean_return=float(np.mean(returns)) if returns.size else 0.0,
            return_std=float(np.std(returns, ddof=0)) if returns.size else 0.0,
            per_episode_scores=[float(s) for s in scores],
            per_episode_returns=[float(r) for r in returns],
            episode_lengths=lengths,
            num_episodes=int(len(episodes)),
            normalization=mode,
            score_bounds=(float(score_min), float(score_max)),
            reward_min=getattr(task, "reward_min", None),
            reward_max=getattr(task, "reward_max", None),
            context=context.to_dict(),
            latent_norm=_latent_norm(z),
            decoder_mse=context_reconstruction_error(self.decoder, context, z, device=self.device),
            metadata={"success_rate": float(np.mean([1.0 if e.success else 0.0 for e in episodes]))
                      if episodes else 0.0},
        )

    def evaluate_suite(
        self,
        suite: Any,
        env: Any = None,
        env_factory: Optional[Callable[..., Any]] = None,
        rollout_fn: Optional[Callable[..., EpisodeResult]] = None,
        num_episodes: Optional[int] = None,
        max_steps: Optional[int] = None,
        seed: int = 0,
        start_states: Any = None,
        reward_fn_factory: Optional[Callable[[Any], Callable[[Any], np.ndarray]]] = None,
        reward_source: str = "task",
        tasks: Optional[Sequence[str]] = None,
    ) -> SuiteResult:
        """Evaluate every task in a suite and aggregate to a mean score."""
        task_list = list(suite)
        if tasks is not None:
            wanted = set(tasks)
            task_list = [t for t in task_list if getattr(t, "name", None) in wanted]

        results: List[TaskResult] = []
        for idx, task in enumerate(task_list):
            task_env = env
            task_rollout = rollout_fn
            task_starts = start_states
            if isinstance(env, dict):  # per-task env mapping
                task_env = env.get(getattr(task, "name", ""), None)
            if isinstance(rollout_fn, dict):
                task_rollout = rollout_fn.get(getattr(task, "name", ""), None)
            if isinstance(start_states, dict):
                task_starts = start_states.get(getattr(task, "name", ""), None)
            reward_fn = None if reward_fn_factory is None else reward_fn_factory(task)
            results.append(self.evaluate_task(
                task, suite=suite, env=task_env, env_factory=env_factory, rollout_fn=task_rollout,
                num_episodes=num_episodes, max_steps=max_steps, seed=seed,
                start_states=task_starts, reward_fn=reward_fn, reward_source=reward_source,
                task_index=idx,
            ))

        scores = np.asarray([r.score for r in results], dtype=np.float64)
        return SuiteResult(
            suite_name=getattr(suite, "name", "suite"),
            domain=getattr(suite, "domain", ""),
            task_results=results,
            mean_score=float(np.mean(scores)) if scores.size else 0.0,
            std_score=float(np.std(scores, ddof=0)) if scores.size else 0.0,
            num_context_samples=self.num_context_samples,
            num_episodes=int(num_episodes or self.num_episodes),
            seed=int(seed),
            metadata={"aggregate": getattr(suite, "aggregate", None)},
        )

    def evaluate(
        self,
        suites: Dict[str, Any],
        seed: int = 0,
        **kwargs: Any,
    ) -> EvalReport:
        """Evaluate several suites and return a combined :class:`EvalReport`."""
        report = EvalReport(seed=int(seed), method=self.name,
                            metadata={"num_context_samples": self.num_context_samples,
                                      "num_episodes": int(self.num_episodes)})
        for _name, suite in suites.items():
            report.add(self.evaluate_suite(suite, seed=seed, **kwargs))
        return report


def _latent_norm(z: Any) -> Optional[float]:
    if z is None:
        return None
    if torch is not None and isinstance(z, torch.Tensor):
        z = z.detach().cpu().numpy()
    try:
        return float(np.linalg.norm(np.asarray(z, dtype=np.float64)))
    except Exception:  # pragma: no cover
        return None


def _pick_start_state(start_states: Any, episode_index: int) -> Any:
    if start_states is None:
        return None
    if callable(start_states):
        try:
            return start_states(episode_index)
        except TypeError:  # pragma: no cover
            return start_states()
    arr = start_states
    if isinstance(arr, (list, tuple)) and len(arr) > 0 and not isinstance(arr[0], (int, float)):
        return arr[episode_index % len(arr)]
    try:
        arr = np.asarray(arr)
    except Exception:  # pragma: no cover
        return start_states
    if arr.ndim == 2:
        return arr[episode_index % arr.shape[0]]
    return start_states


# ---------------------------------------------------------------------------
# Multi-seed aggregation (Section 5.2: five random seeds)
# ---------------------------------------------------------------------------
def aggregate_seeds(reports: Sequence[EvalReport]) -> Dict[str, Dict[str, float]]:
    """Mean/std across training seeds, per suite and per task (Source: Section 5.2)."""
    per_suite: Dict[str, List[float]] = {}
    per_task: Dict[str, List[float]] = {}
    for report in reports:
        for suite_name, suite in report.suite_results.items():
            per_suite.setdefault(suite_name, []).append(float(suite.mean_score))
            for task_name, score in suite.scores.items():
                per_task.setdefault(f"{suite_name}/{task_name}", []).append(float(score))

    out: Dict[str, Dict[str, float]] = {}
    for key, values in per_suite.items():
        arr = np.asarray(values, dtype=np.float64)
        out[key] = {"mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=0)),
                    "num_seeds": int(arr.size)}
    for key, values in per_task.items():
        arr = np.asarray(values, dtype=np.float64)
        out.setdefault(key, {})
        out[key].update({"task_mean": float(np.mean(arr)), "task_std": float(np.std(arr, ddof=0)),
                         "task_num_seeds": int(arr.size)})
    return out


def format_seed_summary(summary: Dict[str, Dict[str, float]], digits: int = 1) -> str:
    """Render ``aggregate_seeds`` output in the paper's ``mean +/- std`` style."""
    lines = []
    for key, stats in summary.items():
        if "mean" in stats:
            lines.append(f"{key:<40s} {stats['mean']:.{digits}f} +/- {stats['std']:.{digits}f} "
                         f"({stats['num_seeds']} seeds)")
        else:
            lines.append(f"    {key:<36s} {stats['task_mean']:.{digits}f} "
                         f"+/- {stats['task_std']:.{digits}f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience entry points
# ---------------------------------------------------------------------------
def evaluate_zero_shot(
    encoder: Any,
    policy: Any,
    replay_buffer: Any,
    suites: Dict[str, Any],
    decoder: Any = None,
    seeds: Sequence[int] = (0,),
    device: Any = None,
    num_context_samples: int = FRE_CONTEXT_SAMPLES,
    num_episodes: int = NUM_EVAL_EPISODES,
    obs_transform: Optional[Callable[[Any], Any]] = None,
    use_encoder_inputs: Optional[bool] = None,
    rollout_fn: Optional[Callable[..., EpisodeResult]] = None,
    envs: Optional[Any] = None,
    start_states: Any = None,
    clip_scores: bool = False,
    method: str = "FRE",
) -> Tuple[List[EvalReport], Dict[str, Dict[str, float]]]:
    """Run the paper's zero-shot protocol for one or more seeds.

    Returns ``(reports, per_seed_aggregate)`` where each report holds the 20-episode
    mean per task and the aggregate holds the mean/std across seeds
    (Source: Section 5.2).
    """
    evaluator = ZeroShotEvaluator(
        encoder=encoder, policy=policy, replay_buffer=replay_buffer, decoder=decoder,
        device=device, num_context_samples=num_context_samples, num_episodes=num_episodes,
        obs_transform=obs_transform, use_encoder_inputs=use_encoder_inputs,
        clip_scores=clip_scores, name=method,
    )
    reports: List[EvalReport] = []
    for seed in seeds:
        report = evaluator.evaluate(suites, seed=int(seed), rollout_fn=rollout_fn,
                                    env=envs, start_states=start_states)
        reports.append(report)
    return reports, aggregate_seeds(reports)


def evaluate_registered_suites(
    encoder: Any,
    policy: Any,
    replay_buffer: Any,
    suite_groups: Dict[str, str],
    task_suite_factories: Dict[str, Callable[..., Any]],
    **kwargs: Any,
) -> Tuple[List[EvalReport], Dict[str, Dict[str, float]]]:
    """Build suites via their factories (per domain) and evaluate them.

    ``suite_groups`` maps an output name (e.g. ``"ant-goal-reaching"``) to a factory
    key (e.g. ``"antmaze"``), and ``task_suite_factories[s]`` is called with
    ``group=<output name>`` -- mirroring ``make_antmaze_task_suite`` /
    ``make_exorl_task_suite`` / ``make_kitchen_task_suite`` in ``fre.envs``.
    """
    suites: Dict[str, Any] = {}
    for name, factory_key in suite_groups.items():
        factory = task_suite_factories[factory_key]
        try:
            suites[name] = factory(group=name)
        except TypeError:  # pragma: no cover - factories accepting positional group
            suites[name] = factory(name)
    return evaluate_zero_shot(encoder, policy, replay_buffer, suites, **kwargs)


__all__ = [
    # protocol constants
    "NUM_EVAL_EPISODES",
    "NUM_TRAINING_SEEDS",
    "FRE_CONTEXT_SAMPLES",
    "FB_SF_CONTEXT_SAMPLES",
    "NORMALIZED_RETURN_MIN",
    "NORMALIZED_RETURN_MAX",
    "DEFAULT_EVAL_EPISODE_LENGTH",
    "NORMALIZATION_MODES",
    # containers
    "ContextSet",
    "EpisodeResult",
    "TaskResult",
    "SuiteResult",
    "EvalReport",
    # harness
    "ZeroShotEvaluator",
    "evaluate_zero_shot",
    "evaluate_registered_suites",
    "aggregate_seeds",
    "format_seed_summary",
    # helpers
    "sample_task_context",
    "encode_context",
    "context_reconstruction_error",
    "resolve_normalization",
    "normalize_return",
    "task_rewards",
    "task_success",
    "task_episode_length",
    "policy_action",
    "clip_action",
    "apply_start_state",
    "as_numpy_2d",
]
