"""Zero-shot evaluation harness for Functional Reward Encodings (FRE).

This module implements the evaluation protocol described in Section 5.2 and
Appendix C of the paper:

* A **new task** is specified by a reward function ``eta(s)`` (AntMaze goals /
  directional / opensimplex / path tasks, ExORL velocity and goal-reaching
  tasks, Kitchen sparse subtasks).
* ``K = 32`` reward-annotated samples ``(s, eta(s))`` are drawn from the
  offline dataset (for goal tasks, at least one sample is guaranteed to be the
  goal, cf. Appendix B) and encoded into a single latent ``z`` with the FRE
  encoder (the posterior **mean** is used at evaluation time).
* The ``z``-conditioned policy is then rolled out **without any training**:
  ``20`` episodes per seed and ``5`` seeds, with a maximum trajectory length of
  ``2000`` steps on AntMaze and ``1000`` steps on ExORL (Kitchen: 280).

Returns are reported as ``[0, 100]``-normalized episode returns, averaged over
the 20 episodes and with the standard deviation taken across the 5 seeds
(Section 5.2, "Miscellaneous details").  When comparing several methods on the
same task set, the reference return range can be shared through the
``calibration`` argument so that the normalization is identical across methods.
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from fre.evaluation.metrics import (
    NUM_EVAL_EPISODES,
    NUM_SEEDS,
    NORMALIZED_MAX,
    NORMALIZED_MIN,
    EpisodeResult,
    TaskResult,
    aggregate_episodes,
    aggregate_seeds,
    aggregate_task_sets,
    aggregate_tasks,
    build_score_table,
    format_metric,
    format_table,
    normalize_return,
    relative_normalize,
    rewards_to_returns,
)

try:  # torch is a hard requirement for actually evaluating a FRE model
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch is expected to be installed
    torch = None  # type: ignore
    _HAS_TORCH = False

__all__ = [
    "EvalConfig",
    "resolve_domain",
    "build_evaluation_tasks",
    "task_names_for",
    "dataset_states_array",
    "encoder_states_for_domain",
    "task_reward_values",
    "select_encoding_samples",
    "encoding_samples_from_dataset",
    "encode_task_latent",
    "RolloutResult",
    "make_action_fn",
    "rollout_episode",
    "evaluate_task",
    "evaluate_fre",
    "evaluate_policy",
    "resolve_return_range",
    "normalize_episode_return",
    "summarize_results",
    "format_result_table",
    "save_results",
    "relative_normalized_scores",
    "FRE_NUM_ENCODING_SAMPLES",
    "FRE_NUM_EVAL_EPISODES",
    "FRE_NUM_EVAL_SEEDS",
    "FRE_MAX_EPISODE_STEPS",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Number of reward-annotated ``(s, eta(s))`` samples used to encode a task at
#: evaluation time (Section 5.2: "only 32 for FRE").
FRE_NUM_ENCODING_SAMPLES = 32

#: Evaluation episodes per seed (Section 5.2).
FRE_NUM_EVAL_EPISODES = NUM_EVAL_EPISODES  # 20

#: Number of evaluation seeds (Section 5.2 / "Miscellaneous details").
FRE_NUM_EVAL_SEEDS = NUM_SEEDS  # 5

#: Maximum trajectory length per domain (Appendix C / Addendum).
FRE_MAX_EPISODE_STEPS: Dict[str, int] = {
    "antmaze": 2000,
    "exorl": 1000,
    "exorl:walker": 1000,
    "exorl:cheetah": 1000,
    "kitchen": 280,
}

#: Range of the (per-reward-function normalized) reward fed to the encoder.
ENCODER_REWARD_MIN = -1.0
ENCODER_REWARD_MAX = 1.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EvalConfig:
    """Configuration of one zero-shot evaluation run.

    Attributes
    ----------
    domain:
        ``"antmaze"``, ``"exorl:walker"``, ``"exorl:cheetah"``, ``"exorl"`` or
        ``"kitchen"``.
    task_set:
        Aggregate task set to evaluate (``"goal-reaching"``, ``"directional"``,
        ``"random-simplex"``, ``"path"``, ``"all"``).
    num_episodes:
        Episodes per seed (paper: 20).
    seeds:
        Evaluation seeds (paper: 5 training seeds).
    num_encoding_samples:
        ``K`` samples used to encode the test reward function (paper: 32).
    deterministic:
        Use the mean action of the policy.
    max_episode_steps:
        Optional override of the per-domain trajectory length.
    discretize_xy:
        AntMaze X/Y discretization into 32 bins (Appendix C.1), applied to the
        **encoder** inputs only.
    encoder_normalizes_rewards:
        Let :class:`~fre.models.reward_embedding.RewardEmbedding` perform the
        per-reward-function min/max normalization (default) instead of
        normalizing the samples here.
    policy_uses_encoder_observation:
        Whether the policy receives the encoder observation (ExORL: physics
        augmented) or the raw environment observation.  Appendix C.2 notes that
        performance is not greatly affected by this choice; ``True`` is the
        paper-faithful default because the RL networks are trained on the same
        observation space as the encoder.
    normalize_returns:
        Map raw episode returns to ``[0, 100]``.
    return_range:
        Explicit reference ``(min, max)`` return range overriding the metadata /
        family heuristic.
    goal_slot:
        Encoder sample slot overwritten by the goal (goal-reaching tasks).  Set
        to ``None`` to disable goal injection.
    save_json:
        Optional path where the full result dictionary is dumped.
    """

    domain: str = "antmaze"
    task_set: str = "all"
    num_episodes: int = FRE_NUM_EVAL_EPISODES
    seeds: Sequence[int] = field(default_factory=lambda: tuple(range(FRE_NUM_EVAL_SEEDS)))
    num_encoding_samples: int = FRE_NUM_ENCODING_SAMPLES
    deterministic: bool = True
    max_episode_steps: Optional[int] = None
    discretize_xy: bool = True
    encoder_normalizes_rewards: bool = True
    policy_uses_encoder_observation: bool = True
    normalize_returns: bool = True
    return_range: Optional[Tuple[float, float]] = None
    goal_slot: Optional[int] = -1
    save_json: Optional[str] = None
    verbose: bool = True

    def steps_for_domain(self) -> int:
        """Maximum trajectory length for this domain (2000 AntMaze / 1000 ExORL)."""
        if self.max_episode_steps is not None:
            return int(self.max_episode_steps)
        return int(FRE_MAX_EPISODE_STEPS.get(self.domain, 1000))


# ---------------------------------------------------------------------------
# Domain / task helpers
# ---------------------------------------------------------------------------


def resolve_domain(domain: Optional[str] = None, task_name: Optional[str] = None) -> str:
    """Resolve a domain string from an explicit domain or a task-name prefix."""
    if domain:
        return str(domain)
    if task_name:
        from fre.envs import domain_from_task_name

        return domain_from_task_name(str(task_name))
    return "antmaze"


def build_evaluation_tasks(domain: str, task_set: str = "all", **kwargs: Any) -> List[Any]:
    """Instantiate the list of evaluation tasks for ``domain``/``task_set``."""
    from fre.envs import build_tasks

    return list(build_tasks(domain, task_set=task_set, **kwargs))


def task_names_for(domain: str, task_set: str = "all") -> Tuple[str, ...]:
    """Names of the tasks contained in ``domain``/``task_set``."""
    return tuple(task.name for task in build_evaluation_tasks(domain, task_set=task_set))


def dataset_states_array(dataset: Union[Any, np.ndarray, None]) -> np.ndarray:
    """Extract a ``(N, state_dim)`` array of states from a dataset-like object."""
    if dataset is None:
        raise ValueError("A dataset (or array of states) is required for evaluation encoding.")
    if isinstance(dataset, np.ndarray):
        return np.asarray(dataset, dtype=np.float32)
    for attr in ("observations", "states"):
        if hasattr(dataset, attr):
            return np.asarray(getattr(dataset, attr), dtype=np.float32)
    raise TypeError(f"Cannot extract states from object of type {type(dataset)!r}")


# ---------------------------------------------------------------------------
# Encoder input construction
# ---------------------------------------------------------------------------


def encoder_states_for_domain(
    domain: str,
    states: np.ndarray,
    discretize_xy: bool = True,
    num_bins: Optional[int] = None,
) -> np.ndarray:
    """Preprocess states into the representation the FRE encoder consumes.

    AntMaze: X/Y are discretized into 32 bins (Appendix C.1).  ExORL: the
    physics information appended during training (Appendix C.2) is expected to
    be present already.  Kitchen: the raw 59-d observation is used.
    """
    states = np.asarray(states, dtype=np.float32)
    if states.ndim == 1:
        states = states[None]
    if domain.startswith("antmaze"):
        if not discretize_xy:
            return states
        from fre.data.preprocessing import NUM_XY_BINS, discretize_antmaze_xy

        return np.asarray(
            discretize_antmaze_xy(states, num_bins=int(num_bins or NUM_XY_BINS)),
            dtype=np.float32,
        )
    return states


def task_reward_values(task: Any, states: np.ndarray) -> np.ndarray:
    """Evaluate the task reward function ``eta(s)`` on ``states``."""
    states = np.asarray(states, dtype=np.float32)
    if hasattr(task, "encoder_reward"):
        rewards = task.encoder_reward(states)
    elif hasattr(task, "reward_fn") and hasattr(task.reward_fn, "reward"):
        rewards = task.reward_fn.reward(states)
    elif hasattr(task, "reward"):
        rewards = task.reward(states)
    else:  # pragma: no cover - defensive
        raise TypeError(f"Task {task!r} does not expose a reward function.")
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    if rewards.shape[0] != states.shape[0]:  # pragma: no cover - broadcasted output
        rewards = np.broadcast_to(rewards, (states.shape[0],)).astype(np.float32)
    return rewards


def select_encoding_samples(
    task: Any,
    dataset_states: np.ndarray,
    num_samples: int = FRE_NUM_ENCODING_SAMPLES,
    rng: Optional[np.random.Generator] = None,
    domain: str = "antmaze",
    replace_last_with_goal: bool = True,
    discretize_xy: bool = True,
    normalize_rewards: bool = False,
) -> Dict[str, Any]:
    """Build the ``(s, eta(s))`` encoding samples for one test task.

    Returns a dictionary with keys ``states`` (``(K, state_dim)`` encoder
    inputs), ``rewards`` (``(K,)``), ``raw_states`` (``(K, state_dim)``, before
    the encoder-specific discretization) and ``goal_included``.
    """
    states_all = np.asarray(dataset_states, dtype=np.float32)
    rng = rng or np.random.default_rng(0)
    k = int(num_samples)

    idx = rng.integers(0, states_all.shape[0], size=k)
    raw = states_all[idx].copy()

    goal = getattr(task, "goal", None)
    goal_included = False
    if replace_last_with_goal and goal is not None:
        goal = np.asarray(goal, dtype=np.float32).reshape(-1)
        if goal.shape[0] == states_all.shape[1]:
            # pick a dataset state that reaches the goal, to stay on-manifold
            try:
                success = np.asarray(task.is_success(states_all)).reshape(-1).astype(bool)
            except Exception:  # pragma: no cover - defensive
                success = np.zeros(states_all.shape[0], dtype=bool)
            if success.any():
                raw[-1] = states_all[np.flatnonzero(success)[0]]
                goal_included = True
            else:
                warnings.warn(
                    f"No dataset state satisfies task {getattr(task, 'name', task)!r}; "
                    "the goal sample will be omitted from the encoding set.",
                    RuntimeWarning,
                )

    rewards = task_reward_values(task, raw)
    if normalize_rewards:
        r_min = float(np.min(rewards))
        r_max = float(np.max(rewards))
        if r_max - r_min > 1e-6:
            rewards = (rewards - r_min) / (r_max - r_min)
    rewards = np.clip(rewards, ENCODER_REWARD_MIN, ENCODER_REWARD_MAX)

    enc_states = encoder_states_for_domain(domain, raw, discretize_xy=discretize_xy)
    out: Dict[str, Any] = {
        "states": enc_states.astype(np.float32),
        "rewards": rewards.astype(np.float32),
        "raw_states": raw.astype(np.float32),
        "goal_included": bool(goal_included),
        "task": str(getattr(task, "name", task)),
    }
    if getattr(task, "max_episode_steps", None) is not None:
        out["max_episode_steps"] = int(task.max_episode_steps)
    return out


def encoding_samples_from_dataset(
    task: Any,
    dataset_states: Union[Any, np.ndarray],
    cfg: Optional[EvalConfig] = None,
    seed: int = 0,
    domain: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience wrapper: sample encoding pairs for ``task`` from a dataset."""
    cfg = cfg or EvalConfig()
    domain = resolve_domain(domain or cfg.domain, getattr(task, "name", None))
    rng = np.random.default_rng(int(seed))
    return select_encoding_samples(
        task,
        dataset_states_array(dataset_states),
        num_samples=cfg.num_encoding_samples,
        rng=rng,
        domain=domain,
        replace_last_with_goal=cfg.goal_slot is not None,
        discretize_xy=cfg.discretize_xy,
        normalize_rewards=not cfg.encoder_normalizes_rewards,
    )


def encode_task_latent(
    model: Any,
    states: Union[np.ndarray, "torch.Tensor"],
    rewards: Union[np.ndarray, "torch.Tensor"],
    device: Optional[Any] = None,
    sample: bool = False,
    already_normalized: bool = False,
) -> np.ndarray:
    """Encode a batch of ``(s, eta(s))`` samples into the task latent ``z``.

    The posterior **mean** is used (``sample=False``); the returned array has
    shape ``(batch, latent_dim)``.
    """
    if not _HAS_TORCH:  # pragma: no cover - torch is expected
        raise RuntimeError("PyTorch is required to encode FRE task latents.")

    s = torch.as_tensor(np.asarray(states, dtype=np.float32))
    r = torch.as_tensor(np.asarray(rewards, dtype=np.float32))
    if s.dim() == 2:
        s = s.unsqueeze(0)
    if r.dim() == 1:
        r = r.unsqueeze(0)
    if device is not None:
        s = s.to(device)
        r = r.to(device)

    if hasattr(model, "encode"):
        try:
            z = model.encode(s, r, sample=sample, already_normalized=already_normalized)
        except TypeError:  # pragma: no cover - encoder with a smaller signature
            z = model.encode(s, r, sample=sample)
    elif hasattr(model, "encoder"):  # pragma: no cover - fallback
        z = model.encoder(s, r, sample=sample, already_normalized=already_normalized)
    else:  # pragma: no cover - defensive
        raise TypeError(f"Model {type(model)!r} does not expose an `encode` method.")

    if isinstance(z, (tuple, list)):  # (z, mean, log_std)
        z = z[0]
    return z.detach().cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------


@dataclass
class RolloutResult:
    """Result of a single zero-shot evaluation episode."""

    return_: float
    length: int
    success: bool = False
    success_step: Optional[int] = None
    normalized_return: Optional[float] = None
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def reward(self) -> float:
        return float(self.return_)

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "return": float(self.return_),
            "length": int(self.length),
            "success": bool(self.success),
            "success_step": self.success_step,
        }
        if self.normalized_return is not None:
            out["normalized_return"] = float(self.normalized_return)
        return out

    def to_episode_result(self) -> EpisodeResult:
        return EpisodeResult(
            return_=float(self.return_),
            length=int(self.length),
            success=bool(self.success),
            success_steps=self.success_step,
        )


def _to_numpy(x: Any) -> np.ndarray:
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def make_action_fn(
    policy: Any,
    latent: Union[np.ndarray, "torch.Tensor"],
    deterministic: bool = True,
    action_dim: Optional[int] = None,
    seed: int = 0,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a ``state -> action`` callable conditioned on the task latent ``z``.

    ``policy`` may be an :class:`~fre.models.iql.IQL` agent (exposing
    ``select_action``), an ``nn.Module`` policy, or a plain callable
    ``policy(state, z)``.
    """
    z = np.asarray(_to_numpy(latent), dtype=np.float32).reshape(-1)
    rng = np.random.default_rng(int(seed))

    def action_fn(state: np.ndarray) -> np.ndarray:
        s = np.asarray(state, dtype=np.float32).reshape(-1)
        if policy is None:  # random policy fallback
            return rng.uniform(-1.0, 1.0, size=int(action_dim or 1)).astype(np.float32)
        if hasattr(policy, "select_action"):
            try:
                a = policy.select_action(s, z, deterministic=deterministic)
            except TypeError:  # pragma: no cover - signature without keyword
                a = policy.select_action(s, z)
            return _to_numpy(a).reshape(-1).astype(np.float32)
        if _HAS_TORCH and isinstance(policy, torch.nn.Module):
            with torch.no_grad():
                st = torch.as_tensor(s, dtype=torch.float32).unsqueeze(0)
                zt = torch.as_tensor(z, dtype=torch.float32).unsqueeze(0)
                if hasattr(policy, "act"):
                    a = policy.act(st, zt, deterministic=deterministic)
                else:
                    dist = policy.distribution(st, zt)
                    a = dist.mean if deterministic else dist.sample()
            return _to_numpy(a).reshape(-1).astype(np.float32)
        if callable(policy):
            a = policy(s, z)
            return _to_numpy(a).reshape(-1).astype(np.float32)
        raise TypeError(f"Cannot use policy of type {type(policy)!r} for rollouts.")  # pragma: no cover

    return action_fn


def _obs_to_state(obs: Any, observation_fn: Optional[Callable[[Any], np.ndarray]] = None) -> np.ndarray:
    """Flatten an environment observation into the state vector ``s``."""
    if observation_fn is not None:
        return np.asarray(observation_fn(obs), dtype=np.float32).reshape(-1)
    if isinstance(obs, dict):  # gymnasium Dict obs (dm_control style)
        for key in ("observation", "obs", "state"):
            if key in obs:
                return np.asarray(obs[key], dtype=np.float32).reshape(-1)
        return np.concatenate(
            [np.asarray(v, dtype=np.float32).reshape(-1) for v in obs.values()]
        ).astype(np.float32)
    if isinstance(obs, (tuple, list)):  # (state, physics) tuples
        return np.concatenate(
            [np.asarray(v, dtype=np.float32).reshape(-1) for v in obs]
        ).astype(np.float32)
    return np.asarray(obs, dtype=np.float32).reshape(-1)


def rollout_episode(
    env: Any,
    action_fn: Callable[[np.ndarray], np.ndarray],
    task: Any,
    max_episode_steps: Optional[int] = None,
    seed: Optional[int] = None,
    observation_fn: Optional[Callable[[Any], np.ndarray]] = None,
    discount: float = 1.0,
) -> RolloutResult:
    """Roll out ``action_fn`` in ``env`` and report the task return.

    The environment is expected to already emit the *task* reward ``eta(s)``
    (see :func:`fre.envs.wrap_env` / ``make_*_task_env``); the return is the
    (undiscounted by default) sum of those rewards.
    """
    from fre.envs.reward_wrappers import call_env_reset, call_env_step

    if max_episode_steps is None:
        max_episode_steps = int(getattr(task, "max_episode_steps", 1000) or 1000)

    reset_kwargs: Dict[str, Any] = {}
    if seed is not None:
        reset_kwargs["seed"] = int(seed)
    obs, _info = call_env_reset(env, **reset_kwargs)

    rewards: List[float] = []
    success = False
    success_step: Optional[int] = None

    for t in range(int(max_episode_steps)):
        state = _obs_to_state(obs, observation_fn)
        action = action_fn(state)
        obs, reward, done, info = call_env_step(env, action)
        rewards.append(float(reward))

        try:
            reached = bool(np.asarray(task.is_success(_obs_to_state(obs, observation_fn))).reshape(-1)[0])
        except Exception:  # pragma: no cover - task without success predicate
            reached = False
        if reached and not success:
            success = True
            success_step = t
        if bool(done):
            break

    return RolloutResult(
        return_=float(rewards_to_returns(rewards, discount=discount)),
        length=len(rewards),
        success=success,
        success_step=success_step,
        info={"seed": seed},
    )


# ---------------------------------------------------------------------------
# Return normalization
# ---------------------------------------------------------------------------


def _heuristic_return_range(task: Any, cfg: Optional[EvalConfig] = None) -> Tuple[float, float]:
    """Best-effort reference return range for a task (paper-silent default)."""
    steps = int(
        (cfg.max_episode_steps if cfg is not None and cfg.max_episode_steps else None)
        or getattr(task, "max_episode_steps", None)
        or 1000
    )
    name = str(getattr(task, "name", "")).lower()
    family = str(getattr(task, "family", "")).lower()
    tag = f"{name} {family}"

    if "goal" in tag or "path" in tag or "simplex" in tag:
        # reward is -1 per step until the goal is reached, then 0
        return (-float(steps), 0.0)
    if "velocity" in tag or "directional" in tag:
        # reward in [0, 1] per step (family-dependent)
        return (0.0, float(steps))
    if "kitchen" in tag:
        return (0.0, float(steps))
    return (float(-steps), float(steps))


def resolve_return_range(
    task: Any,
    cfg: Optional[EvalConfig] = None,
    calibration: Optional[Mapping[str, Tuple[float, float]]] = None,
) -> Tuple[float, float]:
    """Return the ``(ref_min, ref_max)`` range used to normalize task returns.

    Resolution order:

    1. ``cfg.return_range`` (explicit override),
    2. ``calibration`` mapping keyed by task name / task set / element,
    3. task metadata (``return_range`` / ``ref_min`` / ``ref_max``),
    4. a family heuristic based on the per-step reward range.
    """
    if cfg is not None and cfg.return_range is not None:
        return (float(cfg.return_range[0]), float(cfg.return_range[1]))

    name = str(getattr(task, "name", ""))
    if calibration:
        for key in (name, getattr(task, "task_set", None), getattr(task, "element", None)):
            if key is not None and key in calibration:
                lo, hi = calibration[key]
                return (float(lo), float(hi))

    meta = getattr(task, "metadata", None) or {}
    if "return_range" in meta:
        lo, hi = meta["return_range"]
        return (float(lo), float(hi))
    if "ref_min" in meta and "ref_max" in meta:
        return (float(meta["ref_min"]), float(meta["ref_max"]))
    return _heuristic_return_range(task, cfg)


def normalize_episode_return(
    raw_return: float,
    ref_min: float,
    ref_max: float,
    clip: bool = True,
) -> float:
    """Map a raw episode return into ``[0, 100]`` (Table 1 convention)."""
    value = normalize_return(
        raw_return,
        ref_min=ref_min,
        ref_max=ref_max,
        out_min=NORMALIZED_MIN,
        out_max=NORMALIZED_MAX,
    )
    value = float(np.asarray(value).reshape(-1)[0])
    if clip:
        value = float(np.clip(value, NORMALIZED_MIN, NORMALIZED_MAX))
    return value


# ---------------------------------------------------------------------------
# Task / agent evaluation
# ---------------------------------------------------------------------------


def _make_task_env(
    task: Any,
    domain: str,
    seed: Optional[int] = None,
    max_episode_steps: Optional[int] = None,
    env: Any = None,
    env_factory: Optional[Callable[..., Any]] = None,
) -> Any:
    """Create (or reuse) an environment that emits the task reward ``eta(s)``."""
    if env is not None:
        return env
    if env_factory is not None:
        return env_factory(task, seed)

    if domain.startswith("antmaze"):
        from fre.envs.antmaze_tasks import make_antmaze_task_env

        return make_antmaze_task_env(task, seed=seed, max_episode_steps=max_episode_steps)
    if domain.startswith("exorl"):
        from fre.envs.exorl_tasks import make_exorl_task_env

        sub = domain.split(":", 1)[1] if ":" in domain else None
        return make_exorl_task_env(
            task, seed=seed, max_episode_steps=max_episode_steps, domain=sub
        )
    if domain.startswith("kitchen"):
        from fre.envs.kitchen_tasks import make_kitchen_task_env

        return make_kitchen_task_env(task, seed=seed, max_episode_steps=max_episode_steps)
    raise ValueError(f"Unknown domain {domain!r}.")  # pragma: no cover


def _observation_fn_for(
    cfg: EvalConfig, domain: str, policy_uses_encoder_obs: bool
) -> Optional[Callable[[Any], np.ndarray]]:
    """Select the observation presented to the policy during rollouts."""
    if policy_uses_encoder_obs or not domain.startswith("exorl"):
        return None  # environment observation == encoder observation

    def _base_obs(obs: Any) -> np.ndarray:
        arr = _obs_to_state(obs, None)
        from fre.data.preprocessing import exorl_goal_state_dims

        sl = exorl_goal_state_dims(cfg.domain, state_dim=arr.shape[0])
        return arr[sl]

    return _base_obs


def evaluate_task(
    model: Any,
    policy: Any,
    task: Any,
    dataset_states: Union[Any, np.ndarray],
    cfg: Optional[EvalConfig] = None,
    domain: Optional[str] = None,
    seeds: Optional[Sequence[int]] = None,
    env: Any = None,
    env_factory: Optional[Callable[..., Any]] = None,
    calibration: Optional[Mapping[str, Tuple[float, float]]] = None,
    device: Optional[Any] = None,
) -> TaskResult:
    """Zero-shot evaluation of a single test task.

    For every seed: sample ``K`` reward-annotated samples, encode ``z``
    (posterior mean), roll out ``cfg.num_episodes`` episodes, and average the
    episode returns.  The reported task score is the mean over seeds with the
    standard deviation taken across seeds (Section 5.2).
    """
    cfg = cfg or EvalConfig()
    domain = resolve_domain(domain or cfg.domain, getattr(task, "name", None))
    seeds = tuple(int(s) for s in (seeds if seeds is not None else cfg.seeds))
    states_all = dataset_states_array(dataset_states)
    max_steps = (
        int(cfg.max_episode_steps)
        if cfg.max_episode_steps is not None
        else int(getattr(task, "max_episode_steps", None) or cfg.steps_for_domain())
    )
    ref_min, ref_max = resolve_return_range(task, cfg, calibration)
    obs_fn = _observation_fn_for(cfg, domain, cfg.policy_uses_encoder_observation)

    per_seed: List[Dict[str, Any]] = []
    for seed in seeds:
        samples = select_encoding_samples(
            task,
            states_all,
            num_samples=cfg.num_encoding_samples,
            rng=np.random.default_rng(seed),
            domain=domain,
            replace_last_with_goal=cfg.goal_slot is not None,
            discretize_xy=cfg.discretize_xy,
            normalize_rewards=not cfg.encoder_normalizes_rewards,
        )
        z = encode_task_latent(
            model,
            samples["states"],
            samples["rewards"],
            device=device,
            sample=False,
            already_normalized=not cfg.encoder_normalizes_rewards,
        )

        env_used = _make_task_env(
            task,
            domain,
            seed=seed,
            max_episode_steps=max_steps,
            env=env,
            env_factory=env_factory,
        )
        action_fn = make_action_fn(policy, z, deterministic=cfg.deterministic, seed=seed)

        rollouts: List[RolloutResult] = []
        for ep in range(int(cfg.num_episodes)):
            rollout = rollout_episode(
                env_used,
                action_fn,
                task,
                max_episode_steps=max_steps,
                seed=seed * 1000 + ep,
                observation_fn=obs_fn,
            )
            if cfg.normalize_returns:
                rollout.normalized_return = normalize_episode_return(
                    rollout.return_, ref_min, ref_max
                )
            rollouts.append(rollout)

        episode_metrics = aggregate_episodes([r.to_episode_result() for r in rollouts])
        raw_returns = [r.return_ for r in rollouts]
        norm_returns = (
            [r.normalized_return for r in rollouts] if cfg.normalize_returns else raw_returns
        )
        per_seed.append(
            {
                "seed": seed,
                "mean_return": float(np.mean(raw_returns)),
                "std_return": float(np.std(raw_returns)),
                "normalized_mean": float(np.mean(norm_returns)),
                "normalized_std": float(np.std(norm_returns)),
                "success_rate": float(episode_metrics.get("success_rate", 0.0)),
                "num_episodes": int(cfg.num_episodes),
                "goal_included": bool(samples["goal_included"]),
            }
        )
        if env is None and hasattr(env_used, "close"):
            try:  # fresh env per seed: release MuJoCo resources
                env_used.close()
            except Exception:  # pragma: no cover - best effort
                pass

    seed_metrics = aggregate_seeds(per_seed, normalize=cfg.normalize_returns)
    return TaskResult(
        name=str(getattr(task, "name", "task")),
        mean=float(seed_metrics["mean"]),
        std=float(seed_metrics["std"]),
        normalized=bool(cfg.normalize_returns),
        task_set=str(getattr(task, "task_set", cfg.task_set)),
        success_rate=float(seed_metrics.get("success_rate", 0.0)),
        num_seeds=len(seeds),
        metadata={
            "per_seed": per_seed,
            "normalized_mean": float(seed_metrics.get("normalized_mean", seed_metrics["mean"])),
            "normalized_std": float(seed_metrics.get("normalized_std", seed_metrics["std"])),
            "return_range": (ref_min, ref_max),
            "max_episode_steps": max_steps,
            "num_encoding_samples": int(cfg.num_encoding_samples),
        },
    )


def evaluate_fre(
    model: Any,
    policy: Any,
    domain: str = "antmaze",
    task_set: str = "all",
    dataset: Union[Any, np.ndarray] = None,
    cfg: Optional[EvalConfig] = None,
    tasks: Optional[Sequence[Any]] = None,
    seeds: Optional[Sequence[int]] = None,
    env: Any = None,
    env_factory: Optional[Callable[..., Any]] = None,
    calibration: Optional[Mapping[str, Tuple[float, float]]] = None,
    device: Optional[Any] = None,
    task_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Zero-shot evaluation of FRE over an entire task set.

    Returns a dictionary with per-task :class:`~fre.evaluation.metrics.TaskResult`
    objects, aggregate task-set scores, and a ``[0, 100]`` score table.
    """
    cfg = cfg or EvalConfig(domain=domain, task_set=task_set)
    cfg.domain = domain
    cfg.task_set = task_set
    if seeds is not None:
        cfg.seeds = tuple(int(s) for s in seeds)

    tasks = (
        list(tasks)
        if tasks is not None
        else build_evaluation_tasks(domain, task_set, **(task_kwargs or {}))
    )
    if cfg.verbose:
        print(
            f"[eval] domain={domain} task_set={task_set} tasks={len(tasks)} "
            f"seeds={len(cfg.seeds)} episodes={cfg.num_episodes} "
            f"K={cfg.num_encoding_samples} max_steps={cfg.steps_for_domain()}"
        )

    results: Dict[str, TaskResult] = {}
    for task in tasks:
        t0 = time.time()
        result = evaluate_task(
            model,
            policy,
            task,
            dataset,
            cfg=cfg,
            domain=domain,
            seeds=seeds,
            env=env,
            env_factory=env_factory,
            calibration=calibration,
            device=device,
        )
        results[result.name] = result
        if cfg.verbose:
            print(
                f"[eval] {result.name:<28} "
                f"{format_metric(result.mean, result.std)}  "
                f"(norm {format_metric(result.metadata.get('normalized_mean'), result.metadata.get('normalized_std'))})"
                f"  [{time.time() - t0:.1f}s]"
            )

    summary = summarize_results(results, domain=domain, task_set=task_set, cfg=cfg)
    if cfg.save_json:
        save_results(cfg.save_json, summary)
    return summary


def evaluate_policy(
    policy: Any,
    z: Union[np.ndarray, "torch.Tensor"],
    task: Any,
    domain: Optional[str] = None,
    env_factory: Optional[Callable[..., Any]] = None,
    env: Any = None,
    num_episodes: int = FRE_NUM_EVAL_EPISODES,
    seed: int = 0,
    max_episode_steps: Optional[int] = None,
    deterministic: bool = True,
) -> Dict[str, float]:
    """Roll out a policy for a *given* latent ``z`` (used by OPAL / GC baselines)."""
    domain = resolve_domain(domain, getattr(task, "name", None))
    max_steps = int(
        max_episode_steps
        or getattr(task, "max_episode_steps", None)
        or FRE_MAX_EPISODE_STEPS.get(domain, 1000)
    )
    env_used = _make_task_env(
        task,
        domain,
        seed=seed,
        max_episode_steps=max_steps,
        env=env,
        env_factory=env_factory,
    )
    action_fn = make_action_fn(policy, z, deterministic=deterministic, seed=seed)
    rollouts = [
        rollout_episode(
            env_used, action_fn, task, max_episode_steps=max_steps, seed=seed * 1000 + ep
        )
        for ep in range(int(num_episodes))
    ]
    metrics = aggregate_episodes([r.to_episode_result() for r in rollouts])
    if env is None and hasattr(env_used, "close"):
        try:
            env_used.close()
        except Exception:  # pragma: no cover
            pass
    return metrics


# ---------------------------------------------------------------------------
# Aggregation / reporting
# ---------------------------------------------------------------------------


def summarize_results(
    results: Mapping[str, TaskResult],
    domain: str = "antmaze",
    task_set: str = "all",
    cfg: Optional[EvalConfig] = None,
) -> Dict[str, Any]:
    """Aggregate per-task results into task-set / domain level scores."""
    cfg = cfg or EvalConfig(domain=domain, task_set=task_set)
    task_results = dict(results)
    task_to_set = {
        name: str(getattr(res, "task_set", task_set) or task_set)
        for name, res in task_results.items()
    }
    grouped = aggregate_task_sets(task_results, task_to_set=task_to_set)
    overall = aggregate_tasks(task_results.values())
    scores = {
        str(name): float(res.metadata.get("normalized_mean", res.mean))
        for name, res in task_results.items()
    }
    return {
        "domain": domain,
        "task_set": task_set,
        "config": {
            "num_episodes": int(cfg.num_episodes),
            "seeds": list(cfg.seeds),
            "num_encoding_samples": int(cfg.num_encoding_samples),
            "max_episode_steps": int(cfg.steps_for_domain()),
            "normalize_returns": bool(cfg.normalize_returns),
        },
        "tasks": {name: res.to_dict() for name, res in task_results.items()},
        "task_results": task_results,
        "task_sets": grouped,
        "overall": overall,
        "scores": scores,
        "table": build_score_table({domain: scores}),
    }


def format_result_table(summary: Mapping[str, Any], title: Optional[str] = None) -> str:
    """Render a summary dictionary as a human-readable score table."""
    task_results = summary.get("task_results") or {}
    table: Dict[str, Dict[str, str]] = {"normalized": {}}
    for name, res in sorted(task_results.items()):
        if not isinstance(res, TaskResult):
            continue
        table["normalized"][name] = format_metric(
            res.metadata.get("normalized_mean", res.mean),
            res.metadata.get("normalized_std", res.std),
        )
    title = title or (
        f"FRE zero-shot evaluation: {summary.get('domain')}/{summary.get('task_set')}"
    )
    text = format_table(table, title=title)
    overall = summary.get("overall") or {}
    if overall:
        text += "\n" + f"overall: {format_metric(overall.get('mean'), overall.get('std'))}"
    return text


def save_results(path: str, summary: Mapping[str, Any]) -> str:
    """Dump a summary to JSON (TaskResult objects are converted to dicts)."""
    import json

    payload = {
        "domain": summary.get("domain"),
        "task_set": summary.get("task_set"),
        "config": summary.get("config"),
        "tasks": summary.get("tasks"),
        "task_sets": summary.get("task_sets"),
        "overall": summary.get("overall"),
        "scores": summary.get("scores"),
    }
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, default=float)
    return path


def relative_normalized_scores(
    method_results: Mapping[str, Mapping[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Normalize a method x task-set score table so the best method scores 1.0.

    Used by the scaling study (Figure 5 / Table 4), where "there are four
    columns that have a normalized return of 1 (one for each task set)".
    """
    per_set: Dict[str, Dict[str, float]] = {}
    for method, scores in method_results.items():
        for key, value in scores.items():
            per_set.setdefault(str(key), {})[method] = float(value)
    normalized: Dict[str, Dict[str, float]] = {m: {} for m in method_results}
    for key, scores in per_set.items():
        scaled = relative_normalize(scores)
        for method, value in scaled.items():
            normalized.setdefault(method, {})[key] = float(value)
    return normalized
