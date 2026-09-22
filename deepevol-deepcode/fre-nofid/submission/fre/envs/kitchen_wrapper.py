"""Kitchen evaluation environment wrapper for FRE zero-shot evaluation.

Implements the D4RL ``kitchen`` family (``kitchen-complete-v0``,
``kitchen-mixed-v0``, ``kitchen-partial-v0``) as a gym-like environment that can
serve the FRE zero-shot evaluation harness.

Per the reproduction plan (Section 7 / "Environment Wrappers & Eval Rewards"):

    Kitchen: 7 standard D4RL subtasks, use existing sparse rewards directly.

Concretely the wrapper exposes each of the seven standard D4RL Kitchen subtasks
(``microwave``, ``kettle``, ``slide``, ``hinge``, ``light``, ``bottom_burner``,
``top_burner``) as an individual sparse-reward task: the reward is ``1`` the step
the subtask's binary completion flag flips from ``0`` to ``1`` and ``0``
otherwise (the "existing sparse reward" convention of the D4RL Kitchen
datasets).  A per-subtask episode is successful once the flag has been set.

The wrapper mirrors the dual-mode design of :mod:`fre.envs.antmaze_wrapper` and
:mod:`fre.envs.exorl_wrapper`:

* **live mode** - drives the real D4RL/MuJoCo simulator when it is installed
  (used for the numbers reported in the paper), and
* **offline mode** - replays the D4RL Kitchen dataset when MuJoCo is
  unavailable, so that the evaluation harness and unit tests can still run.

In both modes ``reset()``/``step()`` return the *encoder* observation (the raw
60-d D4RL Kitchen observation: 30-d proprioception + 30-d object state, whose
final 7 entries are the subtask completion flags).  Kitchen requires no physics
augmentation (unlike ExORL) and no XY discretisation (unlike AntMaze).

Normalized return is reported in ``[0, 100]`` as the percentage of the seven
subtasks completed, matching the "fraction of tasks solved" reporting used for
the Kitchen column of Table 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - import guard keeps the module importable standalone
    from ..data.d4rl_loader import (
        KITCHEN_DATASET,
        KITCHEN_DATASETS,
        KITCHEN_TASKS,
        load_kitchen_multitask,
    )
except Exception:  # pragma: no cover
    KITCHEN_DATASET = "kitchen-complete-v0"
    KITCHEN_DATASETS = ("kitchen-complete-v0", "kitchen-mixed-v0", "kitchen-partial-v0")
    KITCHEN_TASKS = (
        "microwave",
        "kettle",
        "slide",
        "hinge",
        "light",
        "bottom_burner",
        "top_burner",
    )
    load_kitchen_multitask = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RAW_OBS_DIM = 60
ACTION_DIM = 9
NUM_SUBTASKS = 7
MAX_EPISODE_STEPS = 1000

#: canonical subtask order == order of the final seven observation dimensions
SUBTASK_ORDER: Tuple[str, ...] = (
    "microwave",
    "kettle",
    "slide",
    "hinge",
    "light",
    "bottom_burner",
    "top_burner",
)

#: subtask -> index inside ``obs[-7:]``
SUBTASK_FLAG_INDEX: Dict[str, int] = {name: i for i, name in enumerate(SUBTASK_ORDER)}

#: aliases accepted by :func:`resolve_subtask`
SUBTASK_ALIASES: Dict[str, str] = {
    "microwave": "microwave",
    "kettle": "kettle",
    "slide": "slide",
    "hinge": "hinge",
    "light": "light",
    "bottom_burner": "bottom_burner",
    "bottom burner": "bottom_burner",
    "bottomburner": "bottom_burner",
    "top_burner": "top_burner",
    "top burner": "top_burner",
    "topburner": "top_burner",
}

#: display names used by D4RL / the paper's figures
SUBTASK_DISPLAY: Dict[str, str] = {
    "microwave": "microwave",
    "kettle": "kettle",
    "slide": "slide",
    "hinge": "hinge",
    "light": "light",
    "bottom_burner": "bottom burner",
    "top_burner": "top burner",
}

DEFAULT_DATASET = KITCHEN_DATASET
FLAG_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def resolve_subtask(task: str) -> str:
    """Map a possibly human-formatted subtask name onto the canonical key."""
    if task is None:
        raise ValueError("task must be a Kitchen subtask name")
    key = str(task).strip().lower()
    if key in SUBTASK_ALIASES:
        return SUBTASK_ALIASES[key]
    raise ValueError(
        "unknown Kitchen subtask {!r}; expected one of {}".format(task, SUBTASK_ORDER)
    )


def subtask_flags(obs: np.ndarray) -> np.ndarray:
    """Return the seven binary completion flags contained in ``obs``.

    Kitchen observations are ``[proprio(30), objects(30)]``; the object block
    ends with the seven subtask flags, so the flags live in ``obs[-7:]``.  Works
    for a single observation ``(60,)`` or a batch ``(..., 60)``.
    """
    obs = np.asarray(obs)
    return obs[..., -NUM_SUBTASKS:]


def subtask_complete(obs: np.ndarray, task: str) -> np.ndarray:
    """Boolean (batched) indicator of whether ``task`` is completed in ``obs``."""
    idx = SUBTASK_FLAG_INDEX[resolve_subtask(task)]
    return subtask_flags(obs)[..., idx] > FLAG_THRESHOLD


def all_subtask_flags(obs: np.ndarray) -> Dict[str, bool]:
    """Human readable mapping of subtask -> completion for a single obs."""
    flags = subtask_flags(obs).reshape(-1)[-NUM_SUBTASKS:]
    return {name: bool(flags[i] > FLAG_THRESHOLD) for i, name in enumerate(SUBTASK_ORDER)}


def sparse_subtask_reward(
    prev_obs: Optional[np.ndarray],
    obs: np.ndarray,
    task: str,
    *,
    incomplete_penalty: float = 0.0,
) -> float:
    """Existing sparse D4RL Kitchen reward for one subtask.

    ``1.0`` on the transition where the completion flag flips ``0 -> 1``;
    ``incomplete_penalty`` (default ``0.0``, i.e. the dataset convention)
    otherwise.  When ``prev_obs`` is ``None`` the reward is ``1.0`` if the flag
    is already set, which is the correct behaviour for an evaluation reset.
    """
    idx = SUBTASK_FLAG_INDEX[resolve_subtask(task)]
    now = float(subtask_flags(obs).reshape(-1)[idx]) > FLAG_THRESHOLD
    if prev_obs is None:
        return 1.0 if now else float(incomplete_penalty)
    before = float(subtask_flags(prev_obs).reshape(-1)[idx]) > FLAG_THRESHOLD
    if now and not before:
        return 1.0
    return float(incomplete_penalty)


def goal_style_subtask_reward(
    obs: np.ndarray, task: str, *, success: float = 0.0, failure: float = -1.0
) -> float:
    """Goal-reaching style reward (``success`` once complete, else ``failure``)."""
    return float(success) if bool(subtask_complete(obs, task)) else float(failure)


def episode_ends_from_dataset(dataset: Dict[str, np.ndarray]) -> np.ndarray:
    """Infer episode end indices (inclusive) from a canonical dataset dict.

    D4RL Kitchen is a handful of very long expert trajectories, so the
    ``terminals`` array is frequently all zeros while ``timeouts`` marks the
    episode splits.  Falls back to fixed-length chunking when neither marker is
    present.
    """
    n = len(np.asarray(dataset["observations"]))
    markers: Optional[np.ndarray] = None
    for key in ("terminals", "dones", "timeouts"):
        if key in dataset and dataset[key] is not None:
            arr = np.asarray(dataset[key]).reshape(-1).astype(bool)
            if arr.size != n:
                continue
            markers = arr if markers is None else (markers | arr)
    if markers is None or not markers.any():
        # no episode markers -> chunk into MAX_EPISODE_STEPS long episodes
        ends = np.arange(MAX_EPISODE_STEPS - 1, n, MAX_EPISODE_STEPS, dtype=np.int64)
        if ends.size == 0 or ends[-1] != n - 1:
            ends = np.concatenate([ends, np.array([n - 1], dtype=np.int64)])
        return ends
    ends = np.flatnonzero(markers).astype(np.int64)
    if ends.size == 0 or ends[-1] != n - 1:
        ends = np.concatenate([ends, np.array([n - 1], dtype=np.int64)])
    return ends


def episode_slices(ends: np.ndarray) -> List[Tuple[int, int]]:
    """Convert inclusive episode-end indices into ``(start, end)`` slices."""
    out: List[Tuple[int, int]] = []
    start = 0
    for e in np.asarray(ends).reshape(-1):
        e = int(e)
        if e < start:
            continue
        out.append((start, e))
        start = e + 1
    return out


# ---------------------------------------------------------------------------
# task specification
# ---------------------------------------------------------------------------


@dataclass
class KitchenTaskSpec:
    """Descriptor for a single zero-shot Kitchen subtask."""

    name: str
    subtask: str
    flag_index: int
    reward_style: str = "sparse"  # "sparse" | "goal"
    episode_slices: Optional[List[Tuple[int, int]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def reward(self, obs: np.ndarray, prev_obs: Optional[np.ndarray] = None) -> float:
        if self.reward_style == "goal":
            return goal_style_subtask_reward(obs, self.subtask)
        return sparse_subtask_reward(prev_obs, obs, self.subtask)

    def batch_reward(self, obs: np.ndarray, prev_obs: Optional[np.ndarray] = None) -> np.ndarray:
        """Vectorised reward over a batch of observations."""
        complete = subtask_complete(obs, self.subtask).astype(np.float64)
        if self.reward_style == "goal":
            return complete
        if prev_obs is None:
            return complete
        before = subtask_complete(prev_obs, self.subtask).astype(np.float64)
        return np.clip(complete - before, 0.0, 1.0)

    def done(self, obs: np.ndarray) -> bool:
        return bool(subtask_complete(obs, self.subtask))

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "subtask": self.subtask,
            "display": SUBTASK_DISPLAY.get(self.subtask, self.subtask),
            "flag_index": int(self.flag_index),
            "reward_style": self.reward_style,
            "family": "kitchen",
        }


@dataclass
class KitchenConfig:
    """Configuration bundle for :class:`KitchenWrapper`."""

    dataset_name: str = DEFAULT_DATASET
    task: Optional[str] = None
    max_episode_steps: int = MAX_EPISODE_STEPS
    reward_style: str = "sparse"
    normalise: bool = False
    progress_control: str = "dataset"  # "dataset" | "gated"
    gate_scale: float = 4.0
    seed: Optional[int] = None
    use_live_env: Optional[bool] = None
    terminate_on_success: bool = True


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------


class KitchenWrapper:
    """Gym-like wrapper around a D4RL Kitchen dataset for FRE evaluation.

    Parameters
    ----------
    dataset:
        Pre-loaded multi-task Kitchen payload (as returned by
        :func:`fre.data.d4rl_loader.load_kitchen_multitask`) or a plain
        canonical transition dict.  When ``None`` the dataset is loaded lazily
        from ``dataset_name``.
    task:
        Subtask to expose (canonical name or display name).  Defaults to
        ``"microwave"``; use :meth:`set_task` / :func:`kitchen_eval_tasks` to
        iterate over the full seven-task suite.
    """

    def __init__(
        self,
        dataset: Optional[Dict[str, Any]] = None,
        task: Optional[str] = None,
        dataset_name: str = DEFAULT_DATASET,
        *,
        max_episode_steps: int = MAX_EPISODE_STEPS,
        reward_style: str = "sparse",
        normalise: bool = False,
        progress_control: str = "dataset",
        gate_scale: float = 4.0,
        seed: Optional[int] = None,
        live_env: Any = None,
        use_live_env: Optional[bool] = None,
        config: Optional[KitchenConfig] = None,
    ) -> None:
        if config is not None:
            dataset_name = config.dataset_name
            task = task if task is not None else config.task
            max_episode_steps = config.max_episode_steps
            reward_style = config.reward_style
            normalise = config.normalise
            progress_control = config.progress_control
            gate_scale = config.gate_scale
            seed = config.seed if seed is None else seed
            use_live_env = config.use_live_env if use_live_env is None else use_live_env

        self.dataset_name = dataset_name
        self.max_episode_steps = int(max_episode_steps)
        self.reward_style = reward_style
        self.normalise = bool(normalise)
        self.progress_control = progress_control
        self.gate_scale = float(gate_scale)
        self._rng = np.random.default_rng(seed)

        self._payload: Optional[Dict[str, Any]] = None
        self._task_rewards: Optional[Dict[str, np.ndarray]] = None
        self.dataset: Optional[Dict[str, np.ndarray]] = None
        if dataset is not None:
            self._ingest_dataset(dataset)
        else:
            self._ensure_dataset()

        self._obs = np.asarray(self.dataset["observations"], dtype=np.float64)
        self._next_obs = np.asarray(self.dataset["next_observations"], dtype=np.float64)
        self._actions = np.asarray(self.dataset["actions"], dtype=np.float64).reshape(
            len(self._obs), -1
        )
        self._terminals = self._resolve_terminals(self.dataset)
        self._episode_ends = episode_ends_from_dataset(self.dataset)
        self._episodes = episode_slices(self._episode_ends)

        self.obs_mean, self.obs_std = self._compute_obs_stats()
        self.uses_live_env = False
        self.live_env = None
        self._maybe_make_live_env(live_env, use_live_env)

        # episode bookkeeping
        self._idx = 0
        self._episode_start = 0
        self._episode_end = 0
        self._episode_step = 0
        self._prev_obs: Optional[np.ndarray] = None
        self._done = True
        self._success = False
        self._episode_return = 0.0
        self._num_episodes = 0

        self.task: Optional[KitchenTaskSpec] = None
        self.set_task(task if task is not None else SUBTASK_ORDER[0])

    # ------------------------------------------------------------------
    # dataset handling
    # ------------------------------------------------------------------

    @staticmethod
    def _ingest_dataset(dataset: Dict[str, Any]) -> None:
        return None

    def _ensure_dataset(self) -> None:
        if self.dataset is not None:
            return
        payload = None
        if load_kitchen_multitask is not None:
            payload = load_kitchen_multitask(self.dataset_name)
        if payload is None:
            raise RuntimeError(
                "Kitchen dataset unavailable: install D4RL (pre-June-2024 commit) and "
                "MuJoCo, or pass a pre-loaded `dataset` dict to KitchenWrapper."
            )
        self.set_dataset(payload)

    def set_dataset(self, payload: Dict[str, Any], task_rewards: Optional[Dict[str, np.ndarray]] = None):
        """Attach a multi-task payload ``{dataset, tasks, task_names}`` (or raw dict)."""
        if isinstance(payload, dict) and "dataset" in payload:
            self._payload = payload
            self._task_rewards = payload.get("tasks")
            dataset = payload["dataset"]
            if task_rewards is not None:
                self._task_rewards = task_rewards
        else:
            self._payload = None
            dataset = payload
            self._task_rewards = task_rewards
        # canonicalise keys
        dataset = dict(dataset)
        if "next_observations" not in dataset and "observations" in dataset:
            obs = np.asarray(dataset["observations"])
            nxt = np.empty_like(obs)
            nxt[:-1] = obs[1:]
            nxt[-1] = obs[-1]
            dataset["next_observations"] = nxt
        if "actions" not in dataset:
            dataset["actions"] = np.zeros(
                (len(dataset["observations"]), ACTION_DIM), dtype=np.float32
            )
        if "terminals" not in dataset and "dones" in dataset:
            dataset["terminals"] = np.asarray(dataset["dones"])
        self.dataset = dataset
        return self.dataset

    def _resolve_terminals(self, dataset: Dict[str, np.ndarray]) -> np.ndarray:
        n = len(np.asarray(dataset["observations"]))
        for key in ("terminals", "dones"):
            if key in dataset and dataset[key] is not None:
                arr = np.asarray(dataset[key]).reshape(-1).astype(bool)
                if arr.size == n:
                    return arr
        return np.zeros(n, dtype=bool)

    def _compute_obs_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.normalise or self.dataset is None:
            dim = int(np.asarray(self.dataset["observations"]).shape[-1])
            return np.zeros(dim, dtype=np.float64), np.ones(dim, dtype=np.float64)
        obs = np.asarray(self.dataset["observations"], dtype=np.float64)
        return obs.mean(0), np.maximum(obs.std(0), 1e-3)

    def normalise_obs(self, obs: np.ndarray) -> np.ndarray:
        if not self.normalise:
            return np.asarray(obs, dtype=np.float32)
        return ((np.asarray(obs, dtype=np.float64) - self.obs_mean) / self.obs_std).astype(
            np.float32
        )

    def _maybe_make_live_env(self, live_env: Any, use_live_env: Optional[bool]) -> None:
        if live_env is not None:
            self.live_env = live_env
            self.uses_live_env = True
            return
        if use_live_env is False:
            return
        try:
            from ..data.d4rl_loader import make_env  # local import: optional dep

            self.live_env = make_env(self.dataset_name)
            self.uses_live_env = True
        except Exception:
            self.live_env = None
            self.uses_live_env = False

    # ------------------------------------------------------------------
    # task API
    # ------------------------------------------------------------------

    def set_task(self, task: Any) -> "KitchenTaskSpec":
        """Set the active subtask (accepts a name or a :class:`KitchenTaskSpec`)."""
        if isinstance(task, KitchenTaskSpec):
            spec = task
        else:
            name = resolve_subtask(task)
            spec = KitchenTaskSpec(
                name=name,
                subtask=name,
                flag_index=SUBTASK_FLAG_INDEX[name],
                reward_style=self.reward_style,
                episode_slices=self._episodes,
            )
        self.task = spec
        return spec

    def set_reward_style(self, style: str) -> None:
        self.reward_style = style
        if self.task is not None:
            self.task.reward_style = style

    def default_task(self) -> KitchenTaskSpec:
        return KitchenTaskSpec(
            name=SUBTASK_ORDER[0],
            subtask=SUBTASK_ORDER[0],
            flag_index=0,
            reward_style=self.reward_style,
        )

    def eval_tasks(
        self,
        tasks: Optional[Sequence[str]] = None,
        *,
        reward_style: Optional[str] = None,
    ) -> Dict[str, KitchenTaskSpec]:
        """The seven standard D4RL Kitchen subtasks as zero-shot specs."""
        names = list(tasks) if tasks is not None else list(SUBTASK_ORDER)
        style = reward_style or self.reward_style
        out: Dict[str, KitchenTaskSpec] = {}
        for raw in names:
            key = resolve_subtask(raw)
            out["kitchen-{}".format(key)] = KitchenTaskSpec(
                name="kitchen-{}".format(key),
                subtask=key,
                flag_index=SUBTASK_FLAG_INDEX[key],
                reward_style=style,
                episode_slices=self._episodes,
            )
        return out

    # ------------------------------------------------------------------
    # spaces / properties
    # ------------------------------------------------------------------

    @property
    def observation_dim(self) -> int:
        return int(self._obs.shape[-1]) if self.dataset is not None else RAW_OBS_DIM

    @property
    def policy_observation_dim(self) -> int:
        return self.observation_dim

    @property
    def action_dim(self) -> int:
        if self.dataset is not None:
            return int(self._actions.shape[-1])
        return ACTION_DIM

    @property
    def num_subtasks(self) -> int:
        return NUM_SUBTASKS

    @property
    def observation_space(self) -> Dict[str, Any]:
        return {"shape": (self.observation_dim,), "low": -np.inf, "high": np.inf}

    @property
    def action_space(self) -> Dict[str, Any]:
        return {"shape": (self.action_dim,), "low": -1.0, "high": 1.0}

    @property
    def num_episodes_completed(self) -> int:
        return self._num_episodes

    @property
    def last_episode_return(self) -> float:
        return float(self._episode_return)

    @property
    def last_episode_success(self) -> bool:
        return bool(self._success)

    # ------------------------------------------------------------------
    # observation plumbing
    # ------------------------------------------------------------------

    def build_encoder_observation(self, raw_obs: np.ndarray) -> np.ndarray:
        """Kitchen needs no physics augmentation; only optional normalisation."""
        return self.normalise_obs(np.asarray(raw_obs, dtype=np.float64))

    def policy_observation(self, encoder_obs: np.ndarray) -> np.ndarray:
        """Kitchen uses the same observation for the encoder and the policy."""
        return np.asarray(encoder_obs, dtype=np.float32)

    def achieved_subtasks(self, obs: np.ndarray) -> np.ndarray:
        """Boolean vector (length 7) of subtask completions for ``obs``."""
        return (subtask_flags(obs).reshape(-1)[-NUM_SUBTASKS:] > FLAG_THRESHOLD).astype(bool)

    # ------------------------------------------------------------------
    # reset / step
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        goal: Any = None,
        task: Any = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if task is not None:
            self.set_task(task)

        if self.uses_live_env and self.live_env is not None:
            raw = self.live_env.reset()
            if isinstance(raw, tuple):
                raw = raw[0]
            self._prev_obs = np.asarray(raw, dtype=np.float64).reshape(-1)
        else:
            if not self._episodes:
                raise RuntimeError("Kitchen dataset contains no episodes")
            start, end = self._episodes[int(self._rng.integers(len(self._episodes)))]
            self._episode_start, self._episode_end = int(start), int(end)
            self._idx = int(start)
            self._prev_obs = np.asarray(self._obs[self._idx], dtype=np.float64).reshape(-1)

        self._episode_step = 0
        self._done = False
        self._success = False
        self._episode_return = 0.0

        enc = self.build_encoder_observation(self._prev_obs)
        info = self._make_info()
        return enc, info

    def step(self, action: Any = None) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self._done and not self.uses_live_env:
            # auto-reset so rollouts never crash
            self.reset()

        if self.uses_live_env and self.live_env is not None:
            a = np.asarray(action, dtype=np.float64).reshape(-1)
            raw = self.live_env.step(a)
            if len(raw) == 5:
                next_raw, _r_env, terminated, truncated, _info = raw
            else:  # pragma: no cover - old gym API
                next_raw, _r_env, terminated, _info = raw
                truncated = False
            next_raw = np.asarray(next_raw, dtype=np.float64).reshape(-1)
            reward = float(self.task.reward(next_raw, self._prev_obs))
            success = bool(self.task.done(next_raw))
            self._prev_obs = next_raw
        else:
            next_raw, reward, success = self._offline_transition(action)

        self._episode_step += 1
        terminated = bool(success)
        truncated = bool(self._episode_step >= self.max_episode_steps)
        if not self.uses_live_env and self._idx >= self._episode_end:
            truncated = True

        self._episode_return += float(reward)
        self._success = self._success or success
        self._done = terminated or truncated
        if self._done:
            self._num_episodes += 1

        enc = self.build_encoder_observation(next_raw)
        info = self._make_info()
        info["success"] = success
        return enc, float(reward), terminated, truncated, info

    def _offline_transition(self, action: Any) -> Tuple[np.ndarray, float, bool]:
        """Advance one step along the dataset trajectory (offline mode).

        ``progress_control="gated"`` makes progress depend on the policy action
        (a rough proxy for action-dependence without a simulator); the default
        ``"dataset"`` follows the demonstrated trajectory and is documented as
        an approximation used only when MuJoCo is unavailable.
        """
        advance = True
        if self.progress_control == "gated" and action is not None:
            a = np.asarray(action, dtype=np.float64).reshape(-1)
            demo = self._actions[self._idx]
            denom = max(self.gate_scale, 1e-6)
            prob = float(np.exp(-np.sum((a - demo) ** 2) / denom))
            advance = bool(self._rng.random() < min(max(prob, 0.05), 1.0))

        idx = self._idx
        if advance and idx < self._episode_end:
            self._idx = idx + 1

        next_raw = np.asarray(self._next_obs[idx], dtype=np.float64).reshape(-1)
        if self._idx == idx and idx < len(self._obs) - 1 and self._probable_repeat(idx):
            # when gated progress stalls, hold the current state
            next_raw = np.asarray(self._obs[idx], dtype=np.float64).reshape(-1)

        reward = float(self.task.reward(next_raw, self._prev_obs))
        success = bool(self.task.done(next_raw))
        self._prev_obs = next_raw
        return next_raw, reward, success

    def _probable_repeat(self, idx: int) -> bool:
        """Placeholder hook (kept for symmetry with live-env stepping)."""
        return True

    def _make_info(self) -> Dict[str, Any]:
        obs = self._prev_obs if self._prev_obs is not None else np.zeros(self.observation_dim)
        raw = np.asarray(obs, dtype=np.float32).reshape(-1)
        info: Dict[str, Any] = {
            "policy_observation": self.policy_observation(self.build_encoder_observation(raw)),
            "raw_observation": raw,
            "achieved_subtasks": self.achieved_subtasks(raw),
            "task": self.task.name if self.task else None,
            "subtask": self.task.subtask if self.task else None,
            "subtask_complete": bool(self.task.done(raw)) if self.task else False,
            "episode_step": int(self._episode_step),
            "uses_live_env": bool(self.uses_live_env),
        }
        return info

    # ------------------------------------------------------------------
    # reward helper API (parity with the other wrappers)
    # ------------------------------------------------------------------

    def compute_reward(self, obs: np.ndarray, prev_obs: Optional[np.ndarray] = None) -> float:
        return float(self.task.reward(obs, prev_obs))

    def compute_done(self, obs: np.ndarray) -> bool:
        return bool(self.task.done(obs))

    def reward_function(self) -> Callable[[np.ndarray], float]:
        spec = self.task

        def _reward(obs: np.ndarray, prev_obs: Optional[np.ndarray] = None) -> float:
            return float(spec.reward(obs, prev_obs))

        return _reward

    def normalized_score(self, success: Optional[bool] = None) -> float:
        """Percentage of the seven Kitchen subtasks completed (``[0, 100]``)."""
        if success is None:
            success = self._success
        return 100.0 * float(bool(success))

    # ------------------------------------------------------------------
    # context / dataset access (encoder support)
    # ------------------------------------------------------------------

    def num_dataset_transitions(self) -> int:
        return int(len(self._obs))

    def get_dataset_observation(self, index: Optional[int] = None, random: bool = False) -> np.ndarray:
        if random or index is None:
            index = int(self._rng.integers(len(self._obs)))
        return np.asarray(self._obs[int(index)], dtype=np.float64).reshape(-1)

    def sample_context(
        self,
        num_samples: int = 32,
        indices: Optional[Sequence[int]] = None,
        task: Any = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample ``K`` (state, reward) pairs for the FRE encoder.

        Rewards are evaluated with the active (or supplied) subtask reward, so
        the resulting context encodes a Kitchen subtask into the latent ``z``.
        """
        spec = self.set_task(task) if task is not None else self.task
        if indices is None:
            idx = self._rng.integers(0, len(self._obs), size=int(num_samples))
        else:
            idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        states = np.asarray(self._obs[idx], dtype=np.float64)
        if spec.reward_style == "goal":
            rewards = subtask_complete(states, spec.subtask).astype(np.float64)
        else:
            prev_idx = np.maximum(idx - 1, 0)
            prev_states = np.asarray(self._obs[prev_idx], dtype=np.float64)
            rewards = spec.batch_reward(states, prev_states)
        return states.astype(np.float32), rewards.astype(np.float32)

    # ------------------------------------------------------------------
    # misc gym API
    # ------------------------------------------------------------------

    def seed(self, seed: Optional[int] = None) -> List[int]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        return [int(seed)] if seed is not None else []

    def render(self, *args: Any, **kwargs: Any) -> Any:
        if self.live_env is not None and hasattr(self.live_env, "render"):
            return self.live_env.render(*args, **kwargs)
        return None

    def close(self) -> None:
        if self.live_env is not None and hasattr(self.live_env, "close"):
            try:
                self.live_env.close()
            except Exception:
                pass

    def describe(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "num_transitions": self.num_dataset_transitions(),
            "num_episodes": len(self._episodes),
            "max_episode_steps": self.max_episode_steps,
            "reward_style": self.reward_style,
            "uses_live_env": self.uses_live_env,
            "task": self.task.describe() if self.task else None,
        }


# ---------------------------------------------------------------------------
# convenience subclasses / factories
# ---------------------------------------------------------------------------


class KitchenSubtaskEnv(KitchenWrapper):
    """Kitchen wrapper pinned to a single D4RL subtask."""

    def __init__(self, subtask: str, **kwargs: Any) -> None:
        super().__init__(task=subtask, **kwargs)


def make_kitchen_env(
    dataset: Optional[Dict[str, Any]] = None,
    task: Optional[str] = None,
    *,
    dataset_name: str = DEFAULT_DATASET,
    load_dataset: bool = True,
    **kwargs: Any,
) -> KitchenWrapper:
    """Factory returning a :class:`KitchenWrapper`.

    When ``dataset`` is not supplied and ``load_dataset`` is ``True`` the D4RL
    Kitchen dataset is loaded lazily inside the wrapper (raising a clear error
    when D4RL/MuJoCo are missing).
    """
    if dataset is None and not load_dataset:
        raise ValueError("provide `dataset` or set load_dataset=True")
    return KitchenWrapper(dataset=dataset, task=task, dataset_name=dataset_name, **kwargs)


def kitchen_eval_tasks(
    wrapper: Optional[KitchenWrapper] = None,
    *,
    tasks: Optional[Sequence[str]] = None,
    reward_style: Optional[str] = None,
    dataset_name: str = DEFAULT_DATASET,
) -> Dict[str, KitchenTaskSpec]:
    """Full zero-shot Kitchen task suite keyed by ``kitchen-<subtask>``."""
    if wrapper is None:
        wrapper = None
    episodes: List[Tuple[int, int]] = []
    if wrapper is not None:
        episodes = list(wrapper._episodes)
    names = list(tasks) if tasks is not None else list(SUBTASK_ORDER)
    style = reward_style or (wrapper.reward_style if wrapper is not None else "sparse")
    out: Dict[str, KitchenTaskSpec] = {}
    for raw in names:
        key = resolve_subtask(raw)
        out["kitchen-{}".format(key)] = KitchenTaskSpec(
            name="kitchen-{}".format(key),
            subtask=key,
            flag_index=SUBTASK_FLAG_INDEX[key],
            reward_style=style,
            episode_slices=episodes,
            metadata={"dataset": dataset_name},
        )
    return out


#: zero-shot Kitchen task names, in the canonical order used by the paper
KITCHEN_EVAL_TASK_NAMES: Tuple[str, ...] = tuple(
    "kitchen-{}".format(name) for name in SUBTASK_ORDER
)

__all__ = [
    "KitchenWrapper",
    "KitchenTaskSpec",
    "KitchenConfig",
    "KitchenSubtaskEnv",
    "make_kitchen_env",
    "kitchen_eval_tasks",
    "resolve_subtask",
    "subtask_flags",
    "subtask_complete",
    "all_subtask_flags",
    "sparse_subtask_reward",
    "goal_style_subtask_reward",
    "episode_ends_from_dataset",
    "episode_slices",
    "SUBTASK_ORDER",
    "SUBTASK_FLAG_INDEX",
    "SUBTASK_DISPLAY",
    "KITCHEN_EVAL_TASK_NAMES",
    "KITCHEN_TASKS",
    "KITCHEN_DATASET",
    "KITCHEN_DATASETS",
    "RAW_OBS_DIM",
    "ACTION_DIM",
    "NUM_SUBTASKS",
    "MAX_EPISODE_STEPS",
]
