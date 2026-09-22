"""Kitchen evaluation task suite for FRE (Appendix C.3).

The paper states (Appendix C.3):

    "For the Kitchen evaluation tasks, we utilize the seven standard subtasks
     within the D4RL Kitchen environment. Because each task already defines a
     sparse reward, we directly use those sparse rewards as evaluation tasks."

Accordingly this module exposes the seven standard D4RL Kitchen subtasks
(``microwave``, ``kettle``, ``light switch``, ``slide cabinet``,
``hinge cabinet``, ``top burner``, ``bottom burner``) as ``eta(s)`` reward
functions evaluated directly on the raw (59-d) D4RL Kitchen observation, using
the canonical ``OBS_ELEMENT_INDICES`` / ``OBS_ELEMENT_GOALS`` / ``BONUS_THRESH``
constants from ``d4rl.kitchen.kitchen_envs``.  A subtask is considered complete
when every observation entry belonging to the relevant kitchen element is within
``BONUS_THRESH`` of its goal value; the sparse reward is ``1`` then and ``0``
otherwise (matching the D4RL "element complete" indicator).

Everything here is simulator-independent: rewards are pure functions of states
so that the FRE encoder can be conditioned on `(s, eta(s))` pairs sampled from
the offline dataset as well as from live rollouts.  ``d4rl``/``gym``/``torch``
are imported lazily/optionally so that the module remains importable in a
minimal environment.
"""

from __future__ import annotations

import copy as _copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from fre.priors.goal_functions import RewardFunction

__all__ = [
    "KitchenSubtaskReward",
    "KitchenAllSubtasksReward",
    "TaskSpec",
    "KITCHEN_ENV_NAME",
    "KITCHEN_MAX_EPISODE_STEPS",
    "KITCHEN_OBS_DIM",
    "KITCHEN_BONUS_THRESH",
    "OBS_ELEMENT_INDICES",
    "OBS_ELEMENT_GOALS",
    "KITCHEN_SUBTASKS",
    "KITCHEN_ELEMENT_GOALS",
    "KITCHEN_TASK_SETS",
    "kitchen_element_achieved",
    "kitchen_subtask_success",
    "kitchen_completed_elements",
    "kitchen_completion_fraction",
    "kitchen_encoder_states",
    "make_subtask",
    "make_all_subtasks_task",
    "list_task_sets",
    "task_names",
    "get_task",
    "build_task_set",
    "build_tasks",
    "make_kitchen_env",
    "reset_kitchen_env",
    "make_kitchen_task_env",
    "encoding_samples_for_task",
    "encoding_samples_from_env",
]


# --------------------------------------------------------------------------------------
# Constants (canonical D4RL Kitchen constants)
# --------------------------------------------------------------------------------------

KITCHEN_ENV_NAME = "kitchen-mixed-v0"
KITCHEN_MAX_EPISODE_STEPS = 280
KITCHEN_OBS_DIM = 59

#: Tolerance used by D4RL to declare a kitchen element "complete".
KITCHEN_BONUS_THRESH = 0.3

#: Observation indices (into the raw 59-d D4RL Kitchen observation) of each element.
OBS_ELEMENT_INDICES: Dict[str, np.ndarray] = {
    "bottom burner": np.array([11, 12]),
    "top burner": np.array([15, 16]),
    "light switch": np.array([17, 18]),
    "slide cabinet": np.array([19]),
    "hinge cabinet": np.array([20, 21]),
    "microwave": np.array([22]),
    "kettle": np.array([23, 24, 25, 26, 27, 28, 29]),
}

#: Goal values for the observation entries above (D4RL Kitchen).
OBS_ELEMENT_GOALS: Dict[str, np.ndarray] = {
    "bottom burner": np.array([-0.88, -0.01]),
    "top burner": np.array([-0.92, -0.01]),
    "light switch": np.array([-0.69, -0.05]),
    "slide cabinet": np.array([0.37]),
    "hinge cabinet": np.array([0.0, 1.45]),
    "microwave": np.array([-0.75]),
    "kettle": np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]),
}

#: The seven standard Kitchen subtasks used for evaluation.
KITCHEN_SUBTASKS: Tuple[str, ...] = (
    "microwave",
    "kettle",
    "light switch",
    "slide cabinet",
    "hinge cabinet",
    "top burner",
    "bottom burner",
)

#: Convenience mapping ``element -> (indices, goals)``.
KITCHEN_ELEMENT_GOALS: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
    element: (OBS_ELEMENT_INDICES[element], OBS_ELEMENT_GOALS[element])
    for element in KITCHEN_SUBTASKS
}

#: Aggregate task-set names.  ``all`` == ``subtasks`` (the 7 standard subtasks).
KITCHEN_TASK_SETS: Dict[str, Tuple[str, ...]] = {
    "subtasks": KITCHEN_SUBTASKS,
    "kitchen": KITCHEN_SUBTASKS,
    "standard": KITCHEN_SUBTASKS,
    "all": KITCHEN_SUBTASKS,
}


# --------------------------------------------------------------------------------------
# Small tensor/numpy helpers (torch is optional)
# --------------------------------------------------------------------------------------

try:  # pragma: no cover - optional dependency
    import torch  # type: ignore

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False


def _is_tensor(value: Any) -> bool:
    return _HAS_TORCH and isinstance(value, torch.Tensor)


def _as_float_array(states: Any) -> np.ndarray:
    """Convert numpy/torch/list input into a float64 numpy array."""
    if _is_tensor(states):
        return states.detach().cpu().numpy().astype(np.float64)
    return np.asarray(states, dtype=np.float64)


def _restore(values: np.ndarray, template: Any) -> Any:
    """Cast numpy result back to the type/dtype/device of ``template``."""
    if _is_tensor(template):
        out = torch.as_tensor(values, dtype=template.dtype, device=template.device)
        return out
    if isinstance(template, np.ndarray):
        return values.astype(template.dtype, copy=False)
    return values


# --------------------------------------------------------------------------------------
# Element / subtask success helpers
# --------------------------------------------------------------------------------------


def _element_values(states: np.ndarray, element: str, offset: int = 0) -> np.ndarray:
    """Slice the observation entries belonging to ``element``.

    ``states`` has shape ``(..., obs_dim)``; the returned array has shape
    ``(..., len(indices))``.
    """
    if element not in OBS_ELEMENT_INDICES:
        raise KeyError(f"Unknown kitchen element {element!r}; valid: {sorted(OBS_ELEMENT_INDICES)}")
    idx = np.asarray(OBS_ELEMENT_INDICES[element], dtype=np.int64) + int(offset)
    if states.shape[-1] <= int(idx.max()):
        raise ValueError(
            f"Observation of shape {states.shape} is too small to index element "
            f"{element!r} (needs at least {int(idx.max()) + 1} dims)."
        )
    return states[..., idx]


def kitchen_element_achieved(
    states: Any,
    element: str,
    threshold: float = KITCHEN_BONUS_THRESH,
    offset: int = 0,
) -> Any:
    """Boolean mask (``states.shape[:-1]``) marking where ``element`` is complete."""
    arr = _as_float_array(states)
    flat = arr.reshape(-1, arr.shape[-1])
    values = _element_values(flat, element, offset=offset)
    goals = np.asarray(OBS_ELEMENT_GOALS[element], dtype=np.float64).reshape(1, -1)
    achieved = np.all(np.abs(values - goals) < float(threshold), axis=-1)
    achieved = achieved.reshape(arr.shape[:-1])
    return _restore(achieved, states) if isinstance(states, (np.ndarray,)) or _is_tensor(states) else achieved


def kitchen_subtask_success(
    states: Any,
    element: str,
    threshold: float = KITCHEN_BONUS_THRESH,
    offset: int = 0,
) -> Any:
    """Alias of :func:`kitchen_element_achieved` (float 0/1 mask)."""
    achieved = np.asarray(kitchen_element_achieved(states, element, threshold=threshold, offset=offset))
    out = achieved.astype(np.float64)
    if _is_tensor(states):
        return torch.as_tensor(out, dtype=states.dtype, device=states.device)
    if isinstance(states, np.ndarray):
        return out.astype(np.float32)
    return out


def kitchen_completed_elements(
    states: Any,
    threshold: float = KITCHEN_BONUS_THRESH,
    offset: int = 0,
    elements: Sequence[str] = KITCHEN_SUBTASKS,
) -> np.ndarray:
    """Number of completed elements for each state -> array of shape ``states.shape[:-1]``."""
    arr = _as_float_array(states)
    flat = arr.reshape(-1, arr.shape[-1])
    counts = np.zeros(flat.shape[0], dtype=np.float64)
    for element in elements:
        values = _element_values(flat, element, offset=offset)
        goals = np.asarray(OBS_ELEMENT_GOALS[element], dtype=np.float64).reshape(1, -1)
        counts += np.all(np.abs(values - goals) < float(threshold), axis=-1).astype(np.float64)
    return counts.reshape(arr.shape[:-1])


def kitchen_completion_fraction(
    states: Any,
    threshold: float = KITCHEN_BONUS_THRESH,
    offset: int = 0,
    elements: Sequence[str] = KITCHEN_SUBTASKS,
) -> np.ndarray:
    """Fraction of completed elements in ``[0, 1]`` (D4RL sparse score, normalised)."""
    num = max(len(tuple(elements)), 1)
    return kitchen_completed_elements(states, threshold=threshold, offset=offset, elements=elements) / float(num)


# --------------------------------------------------------------------------------------
# Reward functions
# --------------------------------------------------------------------------------------


class KitchenSubtaskReward(RewardFunction):
    """Sparse reward for a single standard Kitchen subtask.

    ``eta(s) = 1`` if the element's observation entries are within
    ``threshold`` of their goal values (D4RL sparse subtask reward), else ``0``.
    """

    def __init__(
        self,
        element: str,
        threshold: float = KITCHEN_BONUS_THRESH,
        offset: int = 0,
        reward_achieved: float = 1.0,
        reward_unachieved: float = 0.0,
        state_dim: Optional[int] = None,
        normalise: bool = False,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if element not in OBS_ELEMENT_INDICES:
            raise KeyError(
                f"Unknown kitchen element {element!r}; valid: {sorted(OBS_ELEMENT_INDICES)}"
            )
        self.element = element
        self.threshold = float(threshold)
        self.offset = int(offset)
        self.reward_achieved = float(reward_achieved)
        self.reward_unachieved = float(reward_unachieved)
        self._state_dim = state_dim
        # D4RL sparse subtask rewards are already 0/1; `normalise` is kept as a flag for
        # config compatibility but is a no-op for a binary reward.
        self.normalise = bool(normalise)
        self.name = name or f"kitchen-{element}"
        self.indices = np.asarray(OBS_ELEMENT_INDICES[element], dtype=np.int64)
        self.goal_values = np.asarray(OBS_ELEMENT_GOALS[element], dtype=np.float64)

    # -- RewardFunction API -------------------------------------------------------------
    @property
    def state_dim(self) -> Optional[int]:  # type: ignore[override]
        return self._state_dim

    @property
    def family(self) -> str:
        return "kitchen-subtask"

    def distance(self, states: Any) -> Any:
        """Max absolute element-goal deviation (used for progress diagnostics)."""
        arr = _as_float_array(states)
        flat = arr.reshape(-1, arr.shape[-1])
        values = _element_values(flat, self.element, offset=self.offset)
        dev = np.abs(values - self.goal_values.reshape(1, -1)).max(axis=-1)
        dev = dev.reshape(arr.shape[:-1])
        return _restore(dev, states) if _is_tensor(states) else dev

    def achieved(self, states: Any) -> Any:
        return kitchen_element_achieved(states, self.element, threshold=self.threshold, offset=self.offset)

    def reward(self, states: Any) -> Any:
        achieved = np.asarray(
            kitchen_element_achieved(states, self.element, threshold=self.threshold, offset=self.offset)
        ).astype(np.float64)
        out = achieved * self.reward_achieved + (1.0 - achieved) * self.reward_unachieved
        if _is_tensor(states):
            return torch.as_tensor(out, dtype=states.dtype, device=states.device)
        if isinstance(states, np.ndarray):
            return out.astype(np.float32)
        return out

    def __call__(self, states: Any) -> Any:  # pragma: no cover - trivial
        return self.reward(states)

    def extra_repr(self) -> str:
        return f"element={self.element}, threshold={self.threshold}, offset={self.offset}"


class KitchenAllSubtasksReward(RewardFunction):
    """D4RL Kitchen sparse reward over all seven standard subtasks.

    ``mode="count"``   -> integer number of completed subtasks (D4RL behaviour),
    ``mode="fraction"``-> ``count / 7`` in ``[0, 1]``   (default),
    ``mode="dense"``   -> per-element indicator vector of shape ``states.shape[:-1] + (7,)``.
    """

    def __init__(
        self,
        threshold: float = KITCHEN_BONUS_THRESH,
        offset: int = 0,
        elements: Sequence[str] = KITCHEN_SUBTASKS,
        mode: str = "fraction",
        state_dim: Optional[int] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        mode = str(mode).lower()
        if mode not in ("count", "fraction", "dense"):
            raise ValueError(f"Unknown mode {mode!r}; expected 'count', 'fraction' or 'dense'.")
        self.threshold = float(threshold)
        self.offset = int(offset)
        self.elements = tuple(elements)
        self.mode = mode
        self._state_dim = state_dim
        self.name = name or "kitchen-all-subtasks"

    @property
    def state_dim(self) -> Optional[int]:  # type: ignore[override]
        return self._state_dim

    @property
    def family(self) -> str:
        return "kitchen-all"

    def completed(self, states: Any) -> np.ndarray:
        return kitchen_completed_elements(
            states, threshold=self.threshold, offset=self.offset, elements=self.elements
        )

    def reward(self, states: Any) -> Any:
        if self.mode == "dense":
            arr = _as_float_array(states)
            flat = arr.reshape(-1, arr.shape[-1])
            stacked = np.stack(
                [
                    np.all(
                        np.abs(_element_values(flat, e, offset=self.offset)
                               - np.asarray(OBS_ELEMENT_GOALS[e], dtype=np.float64).reshape(1, -1))
                        < self.threshold,
                        axis=-1,
                    ).astype(np.float64)
                    for e in self.elements
                ],
                axis=-1,
            )
            stacked = stacked.reshape(arr.shape[:-1] + (len(self.elements),))
            if _is_tensor(states):
                return torch.as_tensor(stacked, dtype=states.dtype, device=states.device)
            return stacked.astype(np.float32) if isinstance(states, np.ndarray) else stacked

        counts = kitchen_completed_elements(
            states, threshold=self.threshold, offset=self.offset, elements=self.elements
        )
        if self.mode == "fraction":
            counts = counts / float(max(len(self.elements), 1))
        if _is_tensor(states):
            return torch.as_tensor(counts, dtype=states.dtype, device=states.device)
        return counts.astype(np.float32) if isinstance(states, np.ndarray) else counts

    def __call__(self, states: Any) -> Any:  # pragma: no cover - trivial
        return self.reward(states)

    def extra_repr(self) -> str:
        return f"mode={self.mode}, elements={self.elements}, threshold={self.threshold}"


# --------------------------------------------------------------------------------------
# Task specification
# --------------------------------------------------------------------------------------


@dataclass
class TaskSpec:
    """A single Kitchen evaluation task."""

    name: str
    reward_fn: Any
    task_set: str = "kitchen"
    element: Optional[str] = None
    goal: Optional[np.ndarray] = None
    success_fn: Optional[Callable[[Any], Any]] = None
    done_fn: Optional[Callable[[Any], Any]] = None
    max_episode_steps: int = KITCHEN_MAX_EPISODE_STEPS
    threshold: float = KITCHEN_BONUS_THRESH
    env_name: str = KITCHEN_ENV_NAME
    state_dim: Optional[int] = None
    offset: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- introspection ------------------------------------------------------------------
    @property
    def family(self) -> str:
        return getattr(self.reward_fn, "family", "kitchen")

    @property
    def is_subtask(self) -> bool:
        return self.element is not None

    # -- reward evaluation --------------------------------------------------------------
    def reward(self, states: Any) -> Any:
        fn = self.reward_fn
        if hasattr(fn, "reward") and not isinstance(fn, type):
            return fn.reward(states)
        return fn(states)

    def encoder_reward(self, states: Any) -> np.ndarray:
        """Reward used for the FRE encoder tokens (always a 1-D float vector)."""
        out = self.reward(states)
        if _is_tensor(out):
            out = out.detach().cpu().numpy()
        out = np.asarray(out, dtype=np.float64)
        if out.ndim > 1:
            out = out.reshape(out.shape[0], -1).mean(axis=-1)
        return out.reshape(-1)

    def is_success(self, states: Any) -> Any:
        if self.success_fn is not None:
            return self.success_fn(states)
        if self.element is not None:
            return kitchen_element_achieved(
                states, self.element, threshold=self.threshold, offset=self.offset
            )
        return np.asarray(self.reward(states)) > 0.0

    def is_done(self, states: Any) -> Any:
        if self.done_fn is not None:
            return self.done_fn(states)
        return np.asarray(self.is_success(states)).astype(bool)

    def success_fn_for_wrapper(self) -> Callable[[Any], Any]:
        return self.is_success

    def copy(self, **overrides: Any) -> "TaskSpec":
        new = _copy.copy(self)
        for key, value in overrides.items():
            setattr(new, key, value)
        return new

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "task_set": self.task_set,
            "element": self.element,
            "family": self.family,
            "threshold": self.threshold,
            "max_episode_steps": self.max_episode_steps,
            "env_name": self.env_name,
            "metadata": dict(self.metadata),
        }


# --------------------------------------------------------------------------------------
# Task construction
# --------------------------------------------------------------------------------------


def make_subtask(
    element: str,
    threshold: float = KITCHEN_BONUS_THRESH,
    offset: int = 0,
    name: Optional[str] = None,
    max_episode_steps: int = KITCHEN_MAX_EPISODE_STEPS,
    env_name: str = KITCHEN_ENV_NAME,
    task_set: str = "kitchen",
    **kwargs: Any,
) -> TaskSpec:
    """Build the sparse single-subtask task for ``element``."""
    reward_fn = KitchenSubtaskReward(
        element=element,
        threshold=threshold,
        offset=offset,
        name=name or f"kitchen-{element}",
        **kwargs,
    )
    return TaskSpec(
        name=name or f"kitchen-{element}",
        reward_fn=reward_fn,
        task_set=task_set,
        element=element,
        threshold=threshold,
        max_episode_steps=max_episode_steps,
        env_name=env_name,
        offset=offset,
        metadata={"element": element, "goal": np.asarray(OBS_ELEMENT_GOALS[element]).tolist()},
    )


def make_all_subtasks_task(
    mode: str = "fraction",
    threshold: float = KITCHEN_BONUS_THRESH,
    offset: int = 0,
    name: str = "kitchen-all-subtasks",
    max_episode_steps: int = KITCHEN_MAX_EPISODE_STEPS,
    env_name: str = KITCHEN_ENV_NAME,
    task_set: str = "kitchen",
    **kwargs: Any,
) -> TaskSpec:
    """Task rewarding the (normalised) number of completed standard subtasks."""
    reward_fn = KitchenAllSubtasksReward(
        threshold=threshold, offset=offset, mode=mode, name=name, **kwargs
    )
    return TaskSpec(
        name=name,
        reward_fn=reward_fn,
        task_set=task_set,
        element=None,
        threshold=threshold,
        max_episode_steps=max_episode_steps,
        env_name=env_name,
        offset=offset,
        metadata={"mode": mode, "elements": list(KITCHEN_SUBTASKS)},
    )


def list_task_sets() -> Tuple[str, ...]:
    """Aggregate Kitchen task-set names."""
    return tuple(KITCHEN_TASK_SETS.keys())


def task_names(task_set: Optional[str] = None) -> Tuple[str, ...]:
    """Individual task names belonging to ``task_set``."""
    if task_set is None or task_set in ("all", "kitchen", "standard", "subtasks"):
        return tuple(f"kitchen-{e}" for e in KITCHEN_SUBTASKS)
    if task_set == "aggregate":
        return ("kitchen-all-subtasks",)
    if task_set.startswith("kitchen-"):
        return (task_set,)
    if task_set in KITCHEN_SUBTASKS:
        return (f"kitchen-{task_set}",)
    raise KeyError(f"Unknown Kitchen task set {task_set!r}; valid: {list_task_sets()}")


def get_task(name: str, **kwargs: Any) -> TaskSpec:
    """Instantiate one Kitchen task by name.

    Accepts both ``"kitchen-microwave"`` and the bare element ``"microwave"``.
    """
    if name == "kitchen-all-subtasks":
        return make_all_subtasks_task(**kwargs)
    element = name[len("kitchen-"):] if name.startswith("kitchen-") else name
    if element not in OBS_ELEMENT_INDICES:
        raise KeyError(
            f"Unknown Kitchen task {name!r}; valid subtasks: "
            f"{['kitchen-' + e for e in KITCHEN_SUBTASKS]}"
        )
    return make_subtask(element, name=name if name.startswith("kitchen-") else f"kitchen-{element}", **kwargs)


def build_task_set(
    task_set: str = "all",
    threshold: float = KITCHEN_BONUS_THRESH,
    offset: int = 0,
    max_episode_steps: int = KITCHEN_MAX_EPISODE_STEPS,
    env_name: str = KITCHEN_ENV_NAME,
    include_aggregate: bool = False,
    subtasks: Sequence[str] = KITCHEN_SUBTASKS,
    **kwargs: Any,
) -> List[TaskSpec]:
    """Build the list of Kitchen evaluation tasks for ``task_set``."""
    if task_set in ("all", "kitchen", "standard", "subtasks", None):
        tasks = [
            make_subtask(
                e,
                threshold=threshold,
                offset=offset,
                max_episode_steps=max_episode_steps,
                env_name=env_name,
                task_set="kitchen",
                **kwargs,
            )
            for e in subtasks
        ]
        if include_aggregate:
            tasks.append(
                make_all_subtasks_task(
                    threshold=threshold,
                    offset=offset,
                    max_episode_steps=max_episode_steps,
                    env_name=env_name,
                    task_set="kitchen-aggregate",
                    **kwargs,
                )
            )
        return tasks
    if task_set == "aggregate":
        return [
            make_all_subtasks_task(
                threshold=threshold,
                offset=offset,
                max_episode_steps=max_episode_steps,
                env_name=env_name,
                **kwargs,
            )
        ]
    if task_set in KITCHEN_SUBTASKS or task_set.startswith("kitchen-"):
        return [
            get_task(
                task_set,
                threshold=threshold,
                offset=offset,
                max_episode_steps=max_episode_steps,
                env_name=env_name,
                **kwargs,
            )
        ]
    raise KeyError(f"Unknown Kitchen task set {task_set!r}; valid: {list_task_sets()}")


def build_tasks(task_set: str = "all", **kwargs: Any) -> List[TaskSpec]:
    """Alias for :func:`build_task_set`."""
    return build_task_set(task_set=task_set, **kwargs)


# --------------------------------------------------------------------------------------
# Environment creation / wrapping
# --------------------------------------------------------------------------------------


def make_kitchen_env(
    env_name: str = KITCHEN_ENV_NAME,
    seed: Optional[int] = None,
    max_episode_steps: int = KITCHEN_MAX_EPISODE_STEPS,
    env: Any = None,
    terminate_on_success: bool = True,
    **kwargs: Any,
) -> Any:
    """Create (or pass through) the D4RL Kitchen environment."""
    if env is None:
        import gym  # noqa: F401  (imported for registration side effects)
        import d4rl  # noqa: F401,E402

        env = gym.make(env_name)
    if seed is not None:
        try:
            env.seed(seed)
        except Exception:
            pass
        try:
            env.action_space.seed(seed)
        except Exception:
            pass
    try:
        from fre.envs.reward_wrappers import TimeLimitWrapper

        if max_episode_steps is not None:
            env = TimeLimitWrapper(env, max_episode_steps=int(max_episode_steps))
    except Exception:  # pragma: no cover - gym-free fallback
        pass
    return env


def reset_kitchen_env(env: Any, seed: Optional[int] = None, **kwargs: Any) -> Tuple[Any, Dict[str, Any]]:
    """Reset a Kitchen env, normalising gym/gymnasium return signatures."""
    from fre.envs.reward_wrappers import call_env_reset

    return call_env_reset(env, seed=seed, **kwargs)


def make_kitchen_task_env(
    task: Union[TaskSpec, str],
    env_name: Optional[str] = None,
    seed: Optional[int] = None,
    max_episode_steps: Optional[int] = None,
    env: Any = None,
    count_success_as_done: bool = True,
    name: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Wrap a Kitchen environment so that it emits ``task``'s sparse reward ``eta(s)``."""
    if isinstance(task, str):
        task = get_task(task)
    env_name = env_name or task.env_name
    max_episode_steps = (
        task.max_episode_steps if max_episode_steps is None else max_episode_steps
    )
    env = make_kitchen_env(
        env_name=env_name,
        seed=seed,
        max_episode_steps=max_episode_steps,
        env=env,
        **kwargs,
    )
    from fre.envs.reward_wrappers import wrap_env

    return wrap_env(
        env,
        reward_fn=task.reward_fn,
        success_fn=task.success_fn_for_wrapper(),
        count_success_as_done=count_success_as_done,
        max_episode_steps=max_episode_steps,
        domain="kitchen",
        name=name or task.name,
    )


def kitchen_encoder_states(states: Any, **kwargs: Any) -> Any:
    """Prepare Kitchen states for the FRE encoder.

    Kitchen states are used raw (no discretisation/normalisation is specified by
    the paper for this domain); this helper simply guarantees a float tensor/array
    and an ``(..., obs_dim)`` layout.
    """
    if _is_tensor(states):
        return states
    arr = np.asarray(states, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


# --------------------------------------------------------------------------------------
# Zero-shot encoding samples:  K pairs  (s, eta(s))
# --------------------------------------------------------------------------------------


def _normalise_rewards(rewards: np.ndarray) -> np.ndarray:
    """Per-reward-function min/max rescale to ``[0, 1]`` (matches RewardEmbedding)."""
    rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
    if rewards.size == 0:
        return rewards.astype(np.float32)
    lo = float(np.min(rewards))
    hi = float(np.max(rewards))
    if hi - lo < 1e-6:
        return np.zeros_like(rewards, dtype=np.float32)
    return ((rewards - lo) / (hi - lo)).astype(np.float32)


def encoding_samples_for_task(
    task: Union[TaskSpec, str],
    dataset_states: Any,
    num_samples: int = 32,
    rng: Union[int, np.random.Generator, None] = None,
    replace_last_with_goal: bool = False,
    normalize_rewards: bool = True,
    device: Any = None,
) -> Dict[str, Any]:
    """Sample ``num_samples`` ``(s, eta(s))`` pairs for zero-shot encoding of ``task``."""
    if isinstance(task, str):
        task = get_task(task)
    if isinstance(rng, np.random.Generator):
        gen = rng
    else:
        gen = np.random.default_rng(rng)

    states = _as_float_array(dataset_states)
    if states.ndim == 1:
        states = states[None, :]
    num = int(min(int(num_samples), states.shape[0])) if states.shape[0] > 0 else 0
    indices = gen.integers(0, states.shape[0], size=max(num, 1))
    sampled = states[indices]

    rewards = task.encoder_reward(sampled)
    rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)

    if normalize_rewards:
        rewards = _normalise_rewards(rewards)

    out: Dict[str, Any] = {
        "states": sampled.astype(np.float32),
        "rewards": rewards.astype(np.float32),
        "indices": indices,
        "task": task.name,
        "element": task.element,
    }
    if device is not None and _HAS_TORCH:
        out["states"] = torch.as_tensor(out["states"], device=device)
        out["rewards"] = torch.as_tensor(out["rewards"], device=device)
    return out


def encoding_samples_from_env(
    task: Union[TaskSpec, str],
    env: Any,
    num_samples: int = 32,
    policy: Optional[Callable[[Any], Any]] = None,
    seed: Optional[int] = None,
    normalize_rewards: bool = True,
    max_steps: Optional[int] = None,
    device: Any = None,
) -> Dict[str, Any]:
    """Collect ``(s, eta(s))`` pairs for ``task`` by rolling out ``env``."""
    if isinstance(task, str):
        task = get_task(task)
    from fre.envs.reward_wrappers import call_env_reset, call_env_step

    obs, _ = call_env_reset(env, seed=seed)
    collected_states: List[np.ndarray] = []
    steps = 0
    limit = int(max_steps) if max_steps is not None else int(task.max_episode_steps)
    while len(collected_states) < int(num_samples) and steps < limit:
        collected_states.append(np.asarray(obs, dtype=np.float64).reshape(-1))
        if policy is None:
            action = env.action_space.sample()
        else:
            action = policy(obs)
        obs, _, done, _ = call_env_step(env, action)
        steps += 1
        if done:
            obs, _ = call_env_reset(env)

    states = np.stack(collected_states, axis=0) if collected_states else np.zeros((0, KITCHEN_OBS_DIM))
    rewards = np.asarray(task.encoder_reward(states), dtype=np.float64).reshape(-1)
    if normalize_rewards and rewards.size:
        rewards = _normalise_rewards(rewards)

    out: Dict[str, Any] = {
        "states": states.astype(np.float32),
        "rewards": rewards.astype(np.float32),
        "task": task.name,
        "element": task.element,
    }
    if device is not None and _HAS_TORCH:
        out["states"] = torch.as_tensor(out["states"], device=device)
        out["rewards"] = torch.as_tensor(out["rewards"], device=device)
    return out
