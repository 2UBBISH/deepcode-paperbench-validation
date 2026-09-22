"""NetHack environment construction and wrappers.

This module builds the NetHack Learning Environment (NLE) used for the Human
Monk fine-tuning experiments of Wołczyk et al. (2024), Appendix B.1 / Section 5.

It provides:

* :class:`NetHackEnv` -- a single (possibly vectorized) NetHack environment with
  the observation keys expected by :mod:`src.nethack.model`
  (``tty_chars``, ``tty_colors``, ``blstats``, ``message``), score/dungeon-level
  bookkeeping and the paper's rollout-termination rules.
* :class:`NetHackVecEnv` -- a minimal synchronous vectorized wrapper over
  ``num_envs`` copies, yielding observations as dicts of stacked arrays.
* :class:`MiniNetHackVecEnv` -- a dependency-free CPU stand-in used for smoke
  tests and for `--stub` runs when ``nle`` is not installed.
* :func:`make_env` / :func:`make_vec_env` / :func:`build_env` factories.

The paper's evaluation protocol (Section 5, Appendix B.1) stops a rollout when:

1. the agent dies, or
2. 150 steps pass without progress (progress = in-game score / dungeon depth
   increase), or
3. 100k environment steps are reached.

Those constants are exposed as :data:`NO_PROGRESS_STEPS`,
:data:`EVAL_MAX_STEPS` and implemented by :class:`NoProgressTracker`.

Nothing here imports torch/nle at module import time: both are resolved lazily so
that the module can be imported (and unit-tested) in a minimal environment.
"""

from __future__ import annotations

import math
import os
import random
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

#: Observation keys consumed by the model (Appendix B.1 architecture).
OBSERVATION_KEYS: Tuple[str, ...] = ("tty_chars", "tty_colors", "blstats", "message")

#: Shape helpers (24x80 main dungeon screen, 25 blstats features).
MAIN_SCREEN_SHAPE: Tuple[int, int] = (24, 80)
BLSTATS_DIM: int = 25
MESSAGE_LENGTH: int = 256
NUM_MESSAGE_CHARS: int = 128

#: NetHack action space size.
NUM_ACTIONS: int = 120

#: Problem-specific defaults.
ENV_NAME: str = "NetHackChallenge-v0"
NLE_RAW_ENV: str = "NetHackChallenge-v0"
HUMAN_MONK_CHARACTER: str = "human-monk"

#: Rollout termination rules from Section 5 / Appendix B.1.
MAX_EPISODE_STEPS: int = 100_000
NO_PROGRESS_STEPS: int = 150
EVAL_MAX_STEPS: int = 100_000
EVAL_EPISODES: int = 1_000
EVAL_EVERY: int = 25_000_000

#: Progress is measured by the in-game score (falling back to dungeon depth).
PROGRESS_KEYS: Tuple[str, ...] = ("score", "dlvl", "xplvl")

#: Default number of parallel environments for APPO fine-tuning (Table 1: 128).
DEFAULT_NUM_ENVS: int = 128

__all__ = [
    "OBSERVATION_KEYS",
    "MAIN_SCREEN_SHAPE",
    "BLSTATS_DIM",
    "MESSAGE_LENGTH",
    "NUM_MESSAGE_CHARS",
    "NUM_ACTIONS",
    "ENV_NAME",
    "NLE_RAW_ENV",
    "HUMAN_MONK_CHARACTER",
    "MAX_EPISODE_STEPS",
    "NO_PROGRESS_STEPS",
    "EVAL_MAX_STEPS",
    "EVAL_EPISODES",
    "EVAL_EVERY",
    "DEFAULT_NUM_ENVS",
    "EpisodeStats",
    "NoProgressTracker",
    "NetHackEnv",
    "NetHackVecEnv",
    "MiniNetHackEnv",
    "MiniNetHackVecEnv",
    "nle_available",
    "character_from_name",
    "make_env",
    "make_vec_env",
    "build_env",
    "stack_observations",
    "observation_shapes",
    "evaluate_episode",
    "evaluate_policy",
    "main",
]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def nle_available() -> bool:
    """Return ``True`` when the ``nle`` package can be imported."""

    try:  # pragma: no cover - depends on the environment
        import nle  # noqa: F401

        return True
    except Exception:  # pragma: no cover
        return False


def character_from_name(name: str = HUMAN_MONK_CHARACTER) -> str:
    """Normalise a NetHack role string (defaults to the Human Monk)."""

    if not name:
        return HUMAN_MONK_CHARACTER
    return str(name).strip().lower().replace("_", "-").replace(" ", "-")


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Best-effort attribute / mapping lookup."""

    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_array(x: Any) -> Any:
    """Convert a tensor/array/list into a numpy array when possible."""

    try:
        import numpy as np
    except Exception:  # pragma: no cover
        return x
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    detach = getattr(x, "detach", None)
    if callable(detach):
        try:
            x = detach().cpu().numpy()
        except Exception:  # pragma: no cover
            return x
    return np.asarray(x)


# ---------------------------------------------------------------------------
# episode statistics & progress tracking
# ---------------------------------------------------------------------------


@dataclass
class EpisodeStats:
    """Per-episode bookkeeping used for logging and aggregation."""

    episode_return: float = 0.0
    length: int = 0
    max_score: float = 0.0
    score: float = 0.0
    dlvl: float = 1.0
    max_dlvl: float = 1.0
    xplvl: float = 0.0
    turns: float = 0.0
    gold: float = 0.0
    deaths: int = 0
    eaten: float = 0.0

    def update(self, info: Mapping[str, Any]) -> None:
        """Update the statistics from an NLE ``info`` dictionary."""

        def num(key: str, default: float = 0.0) -> float:
            try:
                return float(info.get(key, default))
            except Exception:  # pragma: no cover
                return default

        self.score = num("score", self.score)
        self.max_score = max(self.max_score, self.score)
        self.dlvl = num("dlvl", self.dlvl) or self.dlvl
        self.max_dlvl = max(self.max_dlvl, self.dlvl)
        self.xplvl = num("xplvl", self.xplvl)
        self.turns = num("turns", self.turns)
        self.gold = num("gold", self.gold)
        self.deaths = int(num("deaths", self.deaths))
        self.eaten = num("eaten", self.eaten)

    @property
    def progress(self) -> float:
        """Progress signal used by :class:`NoProgressTracker`."""

        return float(self.score) + 0.01 * float(self.max_dlvl)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "episode_return": float(self.episode_return),
            "length": int(self.length),
            "score": float(self.score),
            "max_score": float(self.max_score),
            "dlvl": float(self.dlvl),
            "max_dlvl": float(self.max_dlvl),
            "xplvl": float(self.xplvl),
            "turns": float(self.turns),
            "gold": float(self.gold),
            "deaths": int(self.deaths),
            "eaten": float(self.eaten),
        }


class NoProgressTracker:
    """Tracks "steps without progress" for the evaluation stopping rule.

    The paper's rollouts stop at death, after 150 steps without progress, or at
    100k steps (Section 5). Progress is defined as an increase of the in-game
    score (or dungeon depth) -- the counter resets whenever it increases.
    """

    def __init__(
        self,
        no_progress_steps: int = NO_PROGRESS_STEPS,
        max_steps: int = EVAL_MAX_STEPS,
        num_envs: int = 1,
    ) -> None:
        self.no_progress_steps = int(no_progress_steps)
        self.max_steps = int(max_steps)
        self.num_envs = int(num_envs)
        self._best = [float("-inf")] * self.num_envs
        self._since_progress = [0] * self.num_envs
        self._steps = [0] * self.num_envs

    def reset(self, index: Optional[int] = None) -> None:
        idxs = range(self.num_envs) if index is None else [int(index)]
        for i in idxs:
            self._best[i] = float("-inf")
            self._since_progress[i] = 0
            self._steps[i] = 0

    def update(self, progress: float, index: int = 0) -> bool:
        """Register one step; returns ``True`` if the episode must stop."""

        i = int(index)
        value = float(progress)
        self._steps[i] += 1
        if value > self._best[i]:
            self._best[i] = value
            self._since_progress[i] = 0
        else:
            self._since_progress[i] += 1
        return self.should_stop(i)

    def should_stop(self, index: int = 0) -> bool:
        i = int(index)
        if self._since_progress[i] >= self.no_progress_steps:
            return True
        if self.max_steps > 0 and self._steps[i] >= self.max_steps:
            return True
        return False

    @property
    def steps(self) -> List[int]:
        return list(self._steps)

    @property
    def since_progress(self) -> List[int]:
        return list(self._since_progress)


# ---------------------------------------------------------------------------
# single environment
# ---------------------------------------------------------------------------


class NetHackEnv:
    """Thin wrapper around the NetHack Learning Environment.

    Parameters
    ----------
    character:
        NetHack role string (default ``"human-monk"`` for the paper's setting).
    max_episode_steps:
        Hard cap on the number of steps per episode (100k for evaluation).
    no_progress_steps:
        Stop the episode after this many steps without progress (150).
    seed:
        Optional environment seed.
    stub:
        When ``True`` (or when ``nle``/``gym`` are unavailable) a deterministic
        CPU stand-in is used instead of the real environment.
    """

    def __init__(
        self,
        character: str = HUMAN_MONK_CHARACTER,
        max_episode_steps: int = MAX_EPISODE_STEPS,
        no_progress_steps: int = NO_PROGRESS_STEPS,
        seed: Optional[int] = None,
        stub: bool = False,
        env_id: str = NLE_RAW_ENV,
        observation_keys: Sequence[str] = OBSERVATION_KEYS,
        **env_kwargs: Any,
    ) -> None:
        self.character = character_from_name(character)
        self.max_episode_steps = int(max_episode_steps or 0)
        self.no_progress_steps = int(no_progress_steps or 0)
        self.env_id = env_id
        self.observation_keys = tuple(observation_keys)
        self.env_kwargs = dict(env_kwargs)
        self.seed_value = seed
        self.is_stub = bool(stub)

        self._env: Any = None
        self._rng = random.Random(seed if seed is not None else 0)
        if not self.is_stub:
            self._env = self._try_make()
        if self._env is None:
            self.is_stub = True
            self._env = MiniNetHackEnv(seed=seed)

        self.stats = EpisodeStats()
        self.tracker = NoProgressTracker(
            no_progress_steps=self.no_progress_steps or 10 ** 9,
            max_steps=self.max_episode_steps or 0,
        )
        self._episode_step = 0

    # -- construction ------------------------------------------------------
    def _try_make(self) -> Any:
        try:  # pragma: no cover - requires nle
            import gym  # type: ignore
        except Exception:
            try:
                import gymnasium as gym  # type: ignore
            except Exception:
                return None
        kwargs: Dict[str, Any] = {"observation_keys": self.observation_keys}
        kwargs.update(self.env_kwargs)
        try:
            env = gym.make(self.env_id, **kwargs)
        except Exception:
            return None
        return env

    # -- gym-style API -----------------------------------------------------
    @property
    def observation_keys_used(self) -> Tuple[str, ...]:
        return self.observation_keys

    @property
    def action_dim(self) -> int:
        space = getattr(self._env, "action_space", None)
        try:
            return int(getattr(space, "n"))
        except Exception:
            return NUM_ACTIONS

    @property
    def observation_dim(self) -> int:
        """Flat observation dimensionality (for stubs / sanity checks)."""

        try:
            return int(sum(int(v) for v in observation_shapes().values() if v))
        except Exception:  # pragma: no cover
            return 0

    def seed(self, seed: Optional[int] = None) -> Optional[int]:
        self.seed_value = seed
        if seed is not None:
            self._rng = random.Random(seed)
        seeder = getattr(self._env, "seed", None)
        if callable(seeder):
            try:
                return seeder(seed)
            except Exception:
                return seed
        return seed

    def reset(self, seed: Optional[int] = None, **kwargs: Any) -> Dict[str, Any]:
        self.stats = EpisodeStats()
        self.tracker.reset()
        self._episode_step = 0
        if seed is not None:
            self.seed(seed)
        out = self._env.reset(**kwargs)
        obs, info = self._split_reset(out, kwargs.get("seed", seed))
        self.stats.update(info)
        self.tracker.update(self.stats.progress)
        return {"obs": obs, "info": info}

    def step(self, action: Any) -> Dict[str, Any]:
        out = self._env.step(int(action))
        obs, reward, terminated, truncated, info = self._split_step(out)
        self._episode_step += 1
        self.stats.episode_return += float(reward)
        self.stats.length += 1
        self.stats.update(info)

        stop = bool(terminated) or bool(truncated)
        stop = stop or self.tracker.update(self.stats.progress)
        if self.max_episode_steps > 0 and self._episode_step >= self.max_episode_steps:
            stop = True

        info = dict(info or {})
        info.setdefault("score", self.stats.score)
        info.setdefault("dlvl", self.stats.dlvl)
        info.setdefault("episode_step", self._episode_step)
        info["steps_without_progress"] = self.tracker.since_progress[0]
        info["episode_return"] = self.stats.episode_return
        info["no_progress_stop"] = bool(
            self.no_progress_steps > 0
            and self.tracker.since_progress[0] >= self.no_progress_steps
        )
        return {
            "obs": obs,
            "reward": float(reward),
            "terminated": bool(terminated) or stop,
            "truncated": bool(truncated),
            "info": info,
        }

    # -- gym API compatibility --------------------------------------------
    def _split_reset(self, out: Any, seed: Optional[int]) -> Tuple[Any, Dict[str, Any]]:
        if isinstance(out, tuple) and len(out) == 2:
            return out[0], dict(out[1] or {})
        return out, {}

    def _split_step(self, out: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        if isinstance(out, tuple):
            if len(out) == 5:
                obs, reward, term, trunc, info = out
                return obs, float(reward), bool(term), bool(trunc), dict(info or {})
            if len(out) == 4:
                obs, reward, done, info = out
                return obs, float(reward), bool(done), False, dict(info or {})
        raise ValueError(f"unexpected step() output: {type(out)!r}")

    # -- evaluation --------------------------------------------------------
    def evaluate_episode(self, policy: Any, deterministic: bool = True) -> Dict[str, Any]:
        return evaluate_episode(self, policy, deterministic=deterministic)

    def close(self) -> None:
        closer = getattr(self._env, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:  # pragma: no cover
                pass

    def render(self, *args: Any, **kwargs: Any) -> Any:
        renderer = getattr(self._env, "render", None)
        if callable(renderer):
            try:
                return renderer(*args, **kwargs)
            except Exception:  # pragma: no cover
                return None
        return None


# ---------------------------------------------------------------------------
# dependency-free stub environment
# ---------------------------------------------------------------------------


class MiniNetHackEnv:
    """Deterministic CPU stand-in for the NetHack environment.

    It exposes the same ``reset``/``step`` protocol and the four observation
    keys, producing an episode that describes a plausible score/depth
    progression. Used for smoke tests and ``--stub`` runs.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        max_episode_steps: int = MAX_EPISODE_STEPS,
        num_actions: int = NUM_ACTIONS,
        **_: Any,
    ) -> None:
        self.num_actions = int(num_actions)
        self.max_episode_steps = int(max_episode_steps or 10 ** 6)
        self.seed_value = seed
        self._rng = random.Random(seed if seed is not None else 0)
        self._step = 0
        self._score = 0.0

    def _obs(self) -> Dict[str, Any]:
        try:
            import numpy as np
        except Exception:  # pragma: no cover
            chars = [[0] * MAIN_SCREEN_SHAPE[1] for _ in range(MAIN_SCREEN_SHAPE[0])]
            colors = [[0] * MAIN_SCREEN_SHAPE[1] for _ in range(MAIN_SCREEN_SHAPE[0])]
            blstats = [0.0] * BLSTATS_DIM
            message = [[0] * NUM_MESSAGE_CHARS for _ in range(MESSAGE_LENGTH)]
        else:
            chars = np.zeros(MAIN_SCREEN_SHAPE, dtype=np.int64)
            colors = np.zeros(MAIN_SCREEN_SHAPE, dtype=np.int64)
            blstats = np.zeros((BLSTATS_DIM,), dtype=np.float32)
            message = np.zeros((MESSAGE_LENGTH, NUM_MESSAGE_CHARS), dtype=np.uint8)
            blstats[0] = float(self._step)
            chars[0, 0] = 64  # '@'
        return {
            "tty_chars": chars,
            "tty_colors": colors,
            "blstats": blstats,
            "message": message,
        }

    def _info(self) -> Dict[str, Any]:
        return {
            "score": self._score,
            "dlvl": 1 + float(self._step) // 500.0,
            "xplvl": float(self._step) // 200.0,
            "turns": float(self._step),
            "gold": 0.0,
            "deaths": 0.0,
            "health": 12.0,
            "time": float(self._step),
            "max_episode_steps": self.max_episode_steps,
        }

    def seed(self, seed: Optional[int] = None) -> Optional[int]:
        self.seed_value = seed
        if seed is not None:
            self._rng = random.Random(seed)
        return seed

    def reset(self, seed: Optional[int] = None, **_: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if seed is not None:
            self.seed(seed)
        self._step = 0
        self._score = 0.0
        return self._obs(), self._info()

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        self._step += 1
        # Small, sparse reward with occasional discoveries (coin/score bump).
        reward = 0.0
        if self._step % 37 == 0:
            reward = 1.0
            self._score += 5.0
        if self._step % 211 == 0:
            reward += 9.0
            self._score += 44.0
        done = self._step >= self.max_episode_steps
        return self._obs(), reward, bool(done), False, self._info()

    def close(self) -> None:
        return None

    def render(self, *_: Any, **__: Any) -> None:
        return None


# ---------------------------------------------------------------------------
# vectorized environments
# ---------------------------------------------------------------------------


class NetHackVecEnv:
    """Minimal synchronous vectorized NetHack environment.

    ``reset`` returns ``{"obs": {key: array[n, ...]}, "info": [info, ...]}`` and
    ``step`` accepts a sequence of actions, returning dict observations together
    with arrays of rewards / dones / infos.
    """

    def __init__(
        self,
        num_envs: int = DEFAULT_NUM_ENVS,
        character: str = HUMAN_MONK_CHARACTER,
        base_seed: Optional[int] = None,
        stub: bool = False,
        max_episode_steps: int = MAX_EPISODE_STEPS,
        no_progress_steps: int = NO_PROGRESS_STEPS,
        env_id: str = NLE_RAW_ENV,
        **env_kwargs: Any,
    ) -> None:
        self.num_envs = int(num_envs)
        self.base_seed = base_seed
        self.stub = bool(stub)
        self.observation_keys = tuple(env_kwargs.pop("observation_keys", OBSERVATION_KEYS))
        env_cls = MiniNetHackEnv if self.stub else NetHackEnv
        self.envs: List[Any] = []
        for i in range(self.num_envs):
            seed = None if base_seed is None else int(base_seed) + i
            if env_cls is MiniNetHackEnv:
                self.envs.append(
                    MiniNetHackEnv(
                        seed=seed,
                        max_episode_steps=max_episode_steps,
                        **{
                            k: v
                            for k, v in env_kwargs.items()
                            if k in ("num_actions",)
                        },
                    )
                )
            else:
                self.envs.append(
                    env_cls(
                        character=character,
                        max_episode_steps=max_episode_steps,
                        no_progress_steps=no_progress_steps,
                        seed=seed,
                        stub=stub,
                        env_id=env_id,
                        observation_keys=self.observation_keys,
                        **env_kwargs,
                    )
                )
        self._last_obs: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]

    # -- helpers -----------------------------------------------------------
    @property
    def action_dim(self) -> int:
        try:
            return int(self.envs[0].action_dim)
        except Exception:
            return NUM_ACTIONS

    @property
    def observation_shape(self) -> Dict[str, Tuple[int, ...]]:
        shapes = observation_shapes()
        return {k: (self.num_envs,) + tuple(v) for k, v in shapes.items()}

    def reset(self, seed: Optional[int] = None, **kwargs: Any) -> Dict[str, Any]:
        obs_list: List[Dict[str, Any]] = []
        infos: List[Dict[str, Any]] = []
        for i, env in enumerate(self.envs):
            s = None if seed is None else int(seed) + i
            out = env.reset(seed=s, **kwargs)
            if isinstance(out, Mapping):
                obs, info = out.get("obs"), out.get("info", {})
            else:
                obs, info = out
            obs_list.append(obs)
            infos.append(dict(info or {}))
        self._last_obs = obs_list
        return {"obs": stack_observations(obs_list, self.observation_keys), "info": infos}

    def step(self, actions: Sequence[Any]) -> Dict[str, Any]:
        obs_list: List[Dict[str, Any]] = []
        rewards: List[float] = []
        dones: List[bool] = []
        infos: List[Dict[str, Any]] = []
        for i, env in enumerate(self.envs):
            action = actions[i] if i < len(actions) else 0
            out = env.step(action)
            if isinstance(out, Mapping):
                obs, reward = out.get("obs"), out.get("reward", 0.0)
                done = bool(out.get("terminated", False)) or bool(out.get("truncated", False))
                info = dict(out.get("info", {}) or {})
            else:
                obs, reward, done, info = out
                done = bool(done)
                info = dict(info or {})
            if done:
                reset_out = env.reset()
                obs = reset_out["obs"] if isinstance(reset_out, Mapping) else reset_out[0]
                info["episode_end"] = True
            obs_list.append(obs)
            rewards.append(float(reward))
            dones.append(bool(done))
            infos.append(info)
        self._last_obs = obs_list
        return {
            "obs": stack_observations(obs_list, self.observation_keys),
            "rewards": rewards,
            "dones": dones,
            "infos": infos,
        }

    def close(self) -> None:
        for env in self.envs:
            try:
                env.close()
            except Exception:  # pragma: no cover
                pass


class MiniNetHackVecEnv(NetHackVecEnv):
    """Vectorized stub environment (identical to ``NetHackVecEnv(stub=True)``)."""

    def __init__(self, num_envs: int = 8, base_seed: Optional[int] = None, **kwargs: Any) -> None:
        kwargs.pop("stub", None)
        super().__init__(num_envs=num_envs, base_seed=base_seed, stub=True, **kwargs)


# ---------------------------------------------------------------------------
# observation utilities
# ---------------------------------------------------------------------------


def _stack_arrays(items: Sequence[Any]) -> Any:
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        return list(items)
    arrays = [_as_array(x) for x in items]
    try:
        return np.stack(arrays, axis=0)
    except Exception:
        return arrays


def stack_observations(
    observations: Sequence[Mapping[str, Any]],
    keys: Sequence[str] = OBSERVATION_KEYS,
) -> Dict[str, Any]:
    """Stack a list of per-env observation dicts into batched arrays."""

    keys = tuple(keys) if keys else OBSERVATION_KEYS
    out: Dict[str, Any] = {}
    for key in keys:
        values = []
        for obs in observations:
            if obs is None:
                values.append(None)
            elif isinstance(obs, Mapping):
                values.append(obs.get(key))
            else:
                values.append(None)
        if any(v is None for v in values):
            continue
        out[key] = _stack_arrays(values)
    return out


def observation_shapes() -> Dict[str, Tuple[int, ...]]:
    """Shapes of the canonical NetHack observation keys."""

    return {
        "tty_chars": tuple(MAIN_SCREEN_SHAPE),
        "tty_colors": tuple(MAIN_SCREEN_SHAPE),
        "blstats": (BLSTATS_DIM,),
        "message": (MESSAGE_LENGTH, NUM_MESSAGE_CHARS),
    }


# ---------------------------------------------------------------------------
# rollout / evaluation helpers
# ---------------------------------------------------------------------------


def _select_action(policy: Any, obs: Any, deterministic: bool, rng: random.Random) -> int:
    """Duck-typed action selection for arbitrary policies."""

    if policy is None:
        return rng.randrange(NUM_ACTIONS)
    act = getattr(policy, "act", None)
    if callable(act):
        for kwargs in (
            {"deterministic": deterministic},
            {},
        ):
            try:
                out = act(obs, **kwargs)
            except TypeError:
                continue
            except Exception:
                continue
            if isinstance(out, Mapping):
                out = out.get("action", out.get("actions"))
            if isinstance(out, (list, tuple)):
                out = out[0]
            if hasattr(out, "item"):
                try:
                    out = out.item()
                except Exception:
                    pass
            return int(out)
    if callable(policy):
        try:
            out = policy(obs)
        except TypeError:
            out = policy()
        if isinstance(out, (list, tuple)):
            out = out[0]
        return int(out)
    return rng.randrange(NUM_ACTIONS)


def evaluate_episode(
    env: Any,
    policy: Any,
    deterministic: bool = True,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Run a single episode until the paper's stopping criteria trigger."""

    rng = random.Random(seed if seed is not None else 0)
    out = env.reset(seed=seed) if seed is not None else env.reset()
    obs = out["obs"] if isinstance(out, Mapping) else out[0]
    done = False
    stats = EpisodeStats()
    while not done:
        action = _select_action(policy, obs, deterministic, rng)
        out = env.step(action)
        if isinstance(out, Mapping):
            obs = out["obs"]
            stats.episode_return += float(out.get("reward", 0.0))
            done = bool(out.get("terminated", False)) or bool(out.get("truncated", False))
            info = dict(out.get("info", {}) or {})
        else:  # pragma: no cover - tuple style
            obs, reward, term, trunc, info = out
            stats.episode_return += float(reward)
            done = bool(term) or bool(trunc)
            info = dict(info or {})
        stats.length += 1
        stats.update(info)
    stats.update(info)
    return stats.as_dict()


def evaluate_policy(
    policy: Any,
    env: Any = None,
    num_episodes: int = EVAL_EPISODES,
    seed: int = 0,
    stub: bool = False,
    deterministic: bool = True,
    max_steps: int = EVAL_MAX_STEPS,
    no_progress_steps: int = NO_PROGRESS_STEPS,
    **env_kwargs: Any,
) -> Dict[str, Any]:
    """Evaluate ``policy`` with the Section 5 stopping rules.

    Returns mean/median score plus trajectory statistics (turns, steps, dlvl,
    xplvl, eating, gold, ...) matching the additional NetHack metrics of the
    paper's Table 4.
    """

    created = False
    if env is None:
        env = NetHackEnv(
            stub=stub,
            max_episode_steps=max_steps,
            no_progress_steps=no_progress_steps,
            **env_kwargs,
        )
        created = True

    scores: List[float] = []
    turns: List[float] = []
    dlvls: List[float] = []
    xplvls: List[float] = []
    golds: List[float] = []
    eaten: List[float] = []
    lengths: List[int] = []
    successes: List[float] = []

    try:
        for i in range(int(num_episodes)):
            result = evaluate_episode(env, policy, deterministic, seed=seed + i)
            scores.append(float(result["score"]))
            turns.append(float(result["turns"]))
            dlvls.append(float(result["dlvl"]))
            xplvls.append(float(result["xplvl"]))
            golds.append(float(result["gold"]))
            eaten.append(float(result["eaten"]))
            lengths.append(int(result["length"]))
            successes.append(1.0 if result["score"] > 0 else 0.0)
    finally:
        if created:
            close = getattr(env, "close", None)
            if callable(close):
                close()

    def _mean(values: Sequence[float]) -> float:
        return float(sum(values) / len(values)) if values else float("nan")

    def _median(values: Sequence[float]) -> float:
        if not values:
            return float("nan")
        ordered = sorted(values)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return float(ordered[mid])
        return float(0.5 * (ordered[mid - 1] + ordered[mid]))

    return {
        "score": _mean(scores),
        "mean_score": _mean(scores),
        "median_score": _median(scores),
        "max_score": max(scores) if scores else float("nan"),
        "episode_return": _mean(scores),
        "return_mean": _mean(scores),
        "turns": _mean(turns),
        "steps": _mean(lengths),
        "dlvl": _mean(dlvls),
        "xplvl": _mean(xplvls),
        "gold": _mean(golds),
        "eating": _mean(eaten),
        "success_rate": _mean(successes),
        "episodes": int(len(scores)),
        "stub": bool(env is not None and getattr(env, "is_stub", False)),
    }


# ---------------------------------------------------------------------------
# factories
# ---------------------------------------------------------------------------


def make_env(
    seed: Optional[int] = None,
    stub: bool = False,
    character: str = HUMAN_MONK_CHARACTER,
    max_episode_steps: int = MAX_EPISODE_STEPS,
    no_progress_steps: int = NO_PROGRESS_STEPS,
    **kwargs: Any,
) -> Any:
    """Build a single NetHack environment (real NLE, or the CPU stub)."""

    if stub or not nle_available():
        try:
            stub_env = MiniNetHackEnv(seed=seed, max_episode_steps=max_episode_steps)
            validated = _validate_stub(stub_env)
            return validated
        except Exception:  # pragma: no cover - never expected
            return MiniNetHackEnv(seed=seed, max_episode_steps=max_episode_steps)

    env = NetHackEnv(
        character=character,
        max_episode_steps=max_episode_steps,
        no_progress_steps=no_progress_steps,
        seed=seed,
        stub=False,
        **kwargs,
    )
    if env.is_stub:
        # nle/gym missing at runtime -> fall back to the stub interface
        return MiniNetHackEnv(seed=seed, max_episode_steps=max_episode_steps)
    return env


def _validate_stub(env: Any) -> Any:
    """Sanity-check the stub API once (cheap) so downstream code is safe."""

    out = env.reset()
    obs = out["obs"] if isinstance(out, Mapping) else out[0]
    if not isinstance(obs, Mapping) or "tty_chars" not in obs:
        raise RuntimeError("stub environment did not produce the expected observation keys")
    return env


def make_vec_env(
    num_envs: int = DEFAULT_NUM_ENVS,
    seed: Optional[int] = None,
    stub: bool = False,
    character: str = HUMAN_MONK_CHARACTER,
    max_episode_steps: int = MAX_EPISODE_STEPS,
    no_progress_steps: int = NO_PROGRESS_STEPS,
    **kwargs: Any,
) -> Any:
    """Build a vectorized NetHack environment of ``num_envs`` copies."""

    use_stub = bool(stub) or not nle_available()
    if use_stub:
        return MiniNetHackVecEnv(num_envs=num_envs, base_seed=seed, **kwargs)
    return NetHackVecEnv(
        num_envs=num_envs,
        character=character,
        base_seed=seed,
        stub=False,
        max_episode_steps=max_episode_steps,
        no_progress_steps=no_progress_steps,
        **kwargs,
    )


#: Alias used by :mod:`src.nethack.appo_runner` (``build_env``).
def build_env(
    config: Any = None,
    seed: Optional[int] = None,
    stub: Optional[bool] = None,
    num_envs: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """Build the training environment from a config object.

    Reads ``env``/``num_envs``/``stub``/``character`` keys from ``config`` when
    available, then falls back to the paper's Table 1 defaults (128 envs).
    """

    def cfg_get(key: str, default: Any = None) -> Any:
        if config is None:
            return default
        if isinstance(config, Mapping):
            for candidate in (key, f"num_{key}s"):
                if candidate in config:
                    return config[candidate]
            env_block = config.get("env", None)
            if isinstance(env_block, Mapping) and key in env_block:
                return env_block[key]
            env_block = config.get("ppo", None)
            if isinstance(env_block, Mapping) and key in env_block:
                return env_block[key]
            return default
        for candidate in (key, f"num_{key}s"):
            if hasattr(config, candidate):
                return getattr(config, candidate)
        for block_name in ("env", "ppo"):
            block = getattr(config, block_name, None)
            if block is not None:
                if isinstance(block, Mapping) and key in block:
                    return block[key]
                if hasattr(block, key):
                    return getattr(block, key)
        return default

    if num_envs is None:
        num_envs = cfg_get("num_envs", cfg_get("num_env", DEFAULT_NUM_ENVS))
    if stub is None:
        stub = bool(cfg_get("stub", False))
    character = kwargs.pop("character", None) or cfg_get("character", HUMAN_MONK_CHARACTER)
    max_episode_steps = kwargs.pop(
        "max_episode_steps",
        cfg_get("max_episode_steps", cfg_get("max_steps_per_episode", MAX_EPISODE_STEPS)),
    )
    no_progress_steps = kwargs.pop("no_progress_steps", cfg_get("no_progress_steps", NO_PROGRESS_STEPS))
    observation_keys = kwargs.pop("observation_keys", cfg_get("observation_keys", OBSERVATION_KEYS))

    return make_vec_env(
        num_envs=int(num_envs or 1),
        seed=seed,
        stub=bool(stub),
        character=str(character),
        max_episode_steps=int(max_episode_steps or 0),
        no_progress_steps=int(no_progress_steps or 0),
        observation_keys=tuple(observation_keys),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# CLI (smoke test)
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    import argparse
    import json

    parser = argparse.ArgumentParser(description="NetHack environment smoke test")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stub", action="store_true", help="force the CPU stub environment")
    parser.add_argument("--evaluate", type=int, default=0, help="run N stub evaluation episodes")
    args = parser.parse_args(list(argv) if argv is not None else None)

    env = make_vec_env(num_envs=args.num_envs, seed=args.seed, stub=args.stub)
    report: Dict[str, Any] = {
        "nle_available": nle_available(),
        "stub": bool(getattr(env, "stub", True)),
        "num_envs": getattr(env, "num_envs", args.num_envs),
        "action_dim": getattr(env, "action_dim", NUM_ACTIONS),
        "observation_shape": getattr(env, "observation_shape", observation_shapes()),
    }
    out = env.reset()
    rewards: List[float] = []
    for _ in range(args.steps):
        actions = [random.randrange(env.action_dim) for _ in range(env.num_envs)]
        out = env.step(actions)
        rewards.extend(float(r) for r in out["rewards"])
    report["steps"] = args.steps
    report["mean_reward"] = sum(rewards) / len(rewards) if rewards else 0.0
    if args.evaluate:
        report["evaluation"] = evaluate_policy(None, num_episodes=args.evaluate, seed=args.seed, stub=True)
    env.close()
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
