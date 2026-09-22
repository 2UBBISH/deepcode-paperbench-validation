"""Montezuma's Revenge environment pipeline (Appendix B.2, Section 3).

This module builds the Atari ``MontezumaRevengeNoFrameskip-v4`` environment
exactly as used in the paper:

* ``MaxStepPerEpisode = 4500``                (Table 2)
* ``StateStackSize = 4``                      (Table 2)
* ``PreProcHeight = ProProcWidth = 84``       (Table 2)
* ``StickyAction = True``, ``ActionProb = 0.25`` (Table 2)
* ``LifeDone = False``                        (Table 2)

and implements the *room* bookkeeping used for the Montezuma analyses:

* rooms are enumerated following the official level-1 progression
  (Figure 12); the player starts in Room 1;
* pre-training of ``pi_*`` (called M1 then M2 in the plan) covers rooms from
  Room 7 onward, so Room 7 and beyond are the **FAR** states while Rooms 1-6
  are **CLOSE** states;
* a room is considered *successfully completed* when the agent earns a coin
  (score increase), acquires a new item, or leaves through a different passage
  than the one it entered (Appendix B.2).

The Atari dependency (``gym`` + ``ale-py``) is optional; a
:class:`DummyMontezumaEnv` reproduces the same interface for CPU smoke tests
whose RAM byte encodes a scripted room progression.
"""

from __future__ import annotations

import argparse
import math
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # optional -- the module must stay importable without gym/ale-py
    import gym  # type: ignore
except Exception:  # pragma: no cover
    try:
        import gymnasium as gym  # type: ignore
    except Exception:
        gym = None  # type: ignore

from src.common.seeding import set_seed

# ---------------------------------------------------------------------------
# Constants (Table 2 / Appendix B.2)
# ---------------------------------------------------------------------------
ENV_ID = "MontezumaRevengeNoFrameskip-v4"
MAX_STEPS_PER_EPISODE = 4500
FRAME_STACK = 4
FRAME_HEIGHT = 84
FRAME_WIDTH = 84
ACTION_PROB = 0.25  # sticky-action probability
NUM_ACTIONS = 18  # full Atari action set used by the RND reference code
OBS_NORM_STEP = 50
OBS_SHAPE = (FRAME_STACK, FRAME_HEIGHT, FRAME_WIDTH)

START_ROOM = 1
ROOM7 = 7  # pre-training starts here (main text, Section 3)
MAX_ROOM = 24  # last room of the first level of Montezuma's Revenge

#: Atari 2600 RAM byte holding the current room number in Montezuma's Revenge.
ROOM_RAM_ADDRESS = 0x83

#: Simplified (coarse) room graph of level 1 following Figure 12: an ordered
#: list of (room, has_coin, has_item, exits) used to decide whether a room was
#: completed in the dummy environment.
MONTEZUMA_ROOMS: Tuple[int, ...] = tuple(range(START_ROOM, MAX_ROOM + 1))

# Keys reported in ``info`` for the room bookkeeping.
ROOM_INFO_KEYS = ("room", "room_index", "room_completed", "room_step", "progress")


# ---------------------------------------------------------------------------
# Frame preprocessing
# ---------------------------------------------------------------------------
def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Convert an RGB frame to an 84x84 uint8 grayscale frame.

    A dependency-light re-implementation of the standard Atari preprocessing
    (only the *resolution* part matters for the reproduction; the ALE wrapper
    already returns grayscale frames when ``grayscale_obs=True``).
    """
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = arr[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    if arr.ndim == 2 and arr.shape != (FRAME_HEIGHT, FRAME_WIDTH):
        arr = _resize_nearest(arr, (FRAME_HEIGHT, FRAME_WIDTH))
    return np.asarray(arr, dtype=np.uint8)


def _resize_nearest(arr: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    h, w = arr.shape[:2]
    th, tw = size
    ys = (np.arange(th) * (h / th)).astype(np.int64).clip(0, h - 1)
    xs = (np.arange(tw) * (w / tw)).astype(np.int64).clip(0, w - 1)
    return arr[ys][:, xs]


class FrameStack:
    """Channel-first stack of the last ``k`` frames (Table 2: ``k = 4``)."""

    def __init__(self, k: int = FRAME_STACK, shape: Tuple[int, int] = (FRAME_HEIGHT, FRAME_WIDTH)):
        self.k = int(k)
        self.shape = shape
        self._frames: deque = deque(maxlen=self.k)

    def reset(self, frame: np.ndarray) -> np.ndarray:
        f = preprocess_frame(frame)
        self._frames.clear()
        for _ in range(self.k):
            self._frames.append(f)
        return self.observation()

    def push(self, frame: np.ndarray) -> np.ndarray:
        f = preprocess_frame(frame)
        if not self._frames:
            return self.reset(f)
        self._frames.append(f)
        return self.observation()

    def observation(self) -> np.ndarray:
        if not self._frames:
            return np.zeros((self.k,) + self.shape, dtype=np.uint8)
        return np.stack(tuple(self._frames), axis=0).astype(np.uint8)

    @property
    def observation_shape(self) -> Tuple[int, int, int]:
        return (self.k,) + self.shape


# ---------------------------------------------------------------------------
# Room bookkeeping
# ---------------------------------------------------------------------------
@dataclass
class RoomEvent:
    """A room-completion event observed during a rollout."""

    step: int
    from_room: int
    to_room: int
    reason: str  # "coin" | "item" | "exit" | "reset"
    score_before: float = 0.0
    score_after: float = 0.0
    items_before: int = 0
    items_after: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "from_room": self.from_room,
            "to_room": self.to_room,
            "reason": self.reason,
            "score_before": self.score_before,
            "score_after": self.score_after,
            "items_before": self.items_before,
            "items_after": self.items_after,
        }


class RoomTracker:
    """Tracks room visitation and room-completion events (Appendix B.2).

    A room counts as completed when at least one of the following happens:

    * the extrinsic score increases (a coin was earned),
    * the number of collected items increases (a new item was acquired),
    * the room index changes (the agent exited through a different passage).
    """

    def __init__(self, start_room: int = START_ROOM, far_room: int = ROOM7):
        self.start_room = int(start_room)
        self.far_room = int(far_room)
        self.reset()

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> None:
        self.step = 0
        self.room = self.start_room
        self.entry_room = self.start_room
        self.room_step = 0
        self.score = 0.0
        self.items = 0
        self.visits: Dict[int, int] = {self.start_room: 1}
        self.room_steps: Dict[int, int] = {self.start_room: 0}
        self.events: List[RoomEvent] = []
        self.completed_rooms: List[int] = []
        self.room_order: List[int] = [self.start_room]
        self.entered_far = self.room >= self.far_room

    # -- observation -------------------------------------------------------
    def update(
        self,
        room: Optional[int] = None,
        score: Optional[float] = None,
        items: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Feed one environment step and return the room info dict."""
        self.step += 1
        self.room_step += 1
        prev_room, prev_score, prev_items = self.room, self.score, self.items
        if room is not None:
            self.room = int(room)
        if score is not None:
            self.score = float(score)
        if items is not None:
            self.items = int(items)

        completed_reason: Optional[str] = None
        if self.score > prev_score:
            completed_reason = "coin"
        elif self.items > prev_items:
            completed_reason = "item"
        if completed_reason is None and self.room != prev_room:
            completed_reason = "exit"

        if completed_reason is not None:
            event = RoomEvent(
                step=self.step,
                from_room=prev_room,
                to_room=self.room if self.room != prev_room else prev_room,
                reason=completed_reason,
                score_before=prev_score,
                score_after=self.score,
                items_before=prev_items,
                items_after=self.items,
            )
            self.events.append(event)
            if prev_room not in self.completed_rooms:
                self.completed_rooms.append(prev_room)

        if self.room != prev_room:
            self.visits[self.room] = self.visits.get(self.room, 0) + 1
            self.room_steps.setdefault(self.room, 0)
            self.room_order.append(self.room)
            self.room_step = 0
            self.entry_room = self.room

        self.room_steps[self.room] = self.room_steps.get(self.room, 0) + 1
        self.entered_far = self.entered_far or (self.room >= self.far_room)
        return self.info()

    def info(self) -> Dict[str, Any]:
        return {
            "room": self.room,
            "room_index": self.room - self.start_room + 1,
            "room_step": self.room_step,
            "room_completed": bool(self.events and self.events[-1].step == self.step),
            "is_far": self.room >= self.far_room,
            "max_room": max(self.visits) if self.visits else self.start_room,
        }

    # -- aggregates --------------------------------------------------------
    @property
    def reached_room7(self) -> bool:
        return self.entered_far

    def room7_success(self, far_room: Optional[int] = None) -> bool:
        """True if the agent completed the designated pre-training room."""
        target = self.far_room if far_room is None else int(far_room)
        return target in self.completed_rooms or any(
            e.from_room == target or e.to_room > target for e in self.events
        )

    def visitation(self, num_rooms: Optional[int] = None) -> np.ndarray:
        n = int(num_rooms if num_rooms is not None else MAX_ROOM)
        out = np.zeros(n, dtype=np.float64)
        for room, count in self.visits.items():
            if 1 <= room <= n:
                out[room - 1] = count
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "room": self.room,
            "visits": dict(self.visits),
            "room_steps": dict(self.room_steps),
            "events": [e.as_dict() for e in self.events],
            "completed_rooms": list(self.completed_rooms),
            "room_order": list(self.room_order),
            "reached_room7": self.reached_room7,
        }


def room_from_ram(ram: Optional[Sequence[int]]) -> Optional[int]:
    """Extract the current room number from an Atari RAM vector.

    Montezuma's Revenge stores the room number at RAM address ``0x83``.
    """
    if ram is None:
        return None
    try:
        value = int(np.asarray(ram).reshape(-1)[ROOM_RAM_ADDRESS])
    except Exception:
        return None
    # The byte counts rooms of the whole game; level 1 rooms are 1..24.
    if value <= 0:
        return None
    return value


def items_from_ram(ram: Optional[Sequence[int]]) -> int:
    """Number of inventory items (used for the 'acquired item' criterion)."""
    if ram is None:
        return 0
    arr = np.asarray(ram).reshape(-1)
    if arr.size <= 0x8A:
        return 0
    try:
        return int(np.count_nonzero(arr[0x86:0x8B]))
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Sticky actions wrapper
# ---------------------------------------------------------------------------
class StickyActionWrapper:
    """Repeat the previous action with probability ``p`` (``ActionProb``)."""

    def __init__(self, env: Any, action_prob: float = ACTION_PROB, seed: Optional[int] = None):
        self.env = env
        self.action_prob = float(action_prob)
        self.rng = np.random.default_rng(seed)
        self.last_action = 0

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        self.last_action = 0
        return out

    def step(self, action: int):
        if self.rng.random() < self.action_prob:
            action = self.last_action
        self.last_action = int(action)
        return self.env.step(action)

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - delegation
        return getattr(self.env, item)


# ---------------------------------------------------------------------------
# Real environment
# ---------------------------------------------------------------------------
class MontezumaEnv:
    """Montezuma's Revenge with the paper's preprocessing and room tracking.

    Parameters
    ----------
    seed:
        Seed used for ``set_seed`` plus the sticky-action RNG.
    max_steps:
        ``MaxStepPerEpisode`` from Table 2 (4500).
    frame_stack:
        ``StateStackSize`` from Table 2 (4).
    action_prob:
        ``ActionProb`` from Table 2 (0.25).
    life_done:
        ``LifeDone`` from Table 2 (False); episode does not end on life loss.
    far_room:
        Room from which pre-training starts (Room 7 in the main text).
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        max_steps: int = MAX_STEPS_PER_EPISODE,
        frame_stack: int = FRAME_STACK,
        action_prob: float = ACTION_PROB,
        life_done: bool = False,
        grayscale: bool = True,
        far_room: int = ROOM7,
        env_id: str = ENV_ID,
        frameskip: int = 4,
        sticky: bool = True,
    ):
        if gym is None:  # pragma: no cover - depends on the environment
            raise RuntimeError(
                "gym/gymnasium is required for MontezumaEnv; "
                "install gym + ale-py or use DummyMontezumaEnv for smoke tests"
            )
        self.seed = seed
        self.env_id = env_id
        self.max_steps = int(max_steps)
        self.far_room = int(far_room)
        self.life_done = bool(life_done)
        self.action_prob = float(action_prob) if sticky else 0.0

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.env = gym.make(env_id, frameskip=1)
            except TypeError:
                self.env = gym.make(env_id)
        self.env = _apply_atari_wrappers(
            self.env,
            frame_stack=frame_stack,
            action_prob=self.action_prob,
            seed=seed,
            grayscale=grayscale,
            frameskip=frameskip,
            sticky=sticky,
        )
        self.stack = FrameStack(frame_stack)
        self.tracker = RoomTracker(far_room=far_room)
        self.steps = 0
        self.episode_return = 0.0
        self.lives = 0
        self._last_obs: Optional[np.ndarray] = None

    # -- spaces ------------------------------------------------------------
    @property
    def observation_shape(self) -> Tuple[int, int, int]:
        return OBS_SHAPE

    @property
    def observation_dim(self) -> int:
        return int(np.prod(OBS_SHAPE))

    @property
    def action_dim(self) -> int:
        try:
            return int(self.env.action_space.n)
        except Exception:
            return NUM_ACTIONS

    # -- api ---------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            set_seed(int(seed))
        out = self.env.reset(**({"seed": seed} if seed is not None else {}) or kwargs)
        obs_raw, info = _split_reset(out)
        self.stack = FrameStack(self.stack.k)
        obs = self.stack.reset(obs_raw)
        self.tracker.reset()
        self.steps = 0
        self.episode_return = 0.0
        self.lives = int(info.get("lives", 0)) if isinstance(info, dict) else 0
        self._last_obs = obs
        info = dict(info) if isinstance(info, dict) else {}
        info.update(self.tracker.info())
        info["lives"] = self.lives
        info["score"] = float(info.get("score", 0.0))
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        out = self.env.step(int(action))
        obs_raw, reward, done, truncated, info = _split_step(out)
        obs = self.stack.push(obs_raw)
        self.steps += 1
        self.episode_return += float(reward)

        ram = _ram_of(self.env)
        info = dict(info) if isinstance(info, dict) else {}
        room = info.get("room") or room_from_ram(ram)
        score = info.get("score", None)
        if score is None:
            score = _score_of(self.env, info)
        items = items_from_ram(ram)
        lives = int(info.get("lives", self.lives))
        life_lost = lives < self.lives
        self.lives = lives

        room_info = self.tracker.update(room=room, score=score, items=items)
        info.update(room_info)
        info["lives"] = lives
        info["life_lost"] = bool(life_lost)
        info["episode_step"] = self.steps
        info["episode_return"] = self.episode_return
        if ram is not None:
            info["ram"] = np.asarray(ram).astype(np.uint8)

        timeout = self.steps >= self.max_steps
        terminated = bool(done) or timeout
        if self.life_done and life_lost:
            terminated = True
        self._last_obs = obs
        return obs, float(reward), terminated, bool(truncated), info

    def close(self) -> None:  # pragma: no cover - resource cleanup
        try:
            self.env.close()
        except Exception:
            pass

    def render(self, *args, **kwargs):  # pragma: no cover
        return self.env.render(*args, **kwargs)


def _apply_atari_wrappers(
    env: Any,
    frame_stack: int,
    action_prob: float,
    seed: Optional[int],
    grayscale: bool = True,
    frameskip: int = 4,
    sticky: bool = True,
) -> Any:
    """Apply max-and-skip / grayscale / resize / sticky-actions wrappers."""
    wrapped = env
    for ctor, kwargs in (
        ("MaxAndSkipEnv", {"skip": frameskip}),
        ("EpisodicLifeEnv", {}),
        ("WarpFrame", {"width": FRAME_WIDTH, "height": FRAME_HEIGHT, "grayscale": grayscale}),
        ("ClipRewardEnv", {}),
    ):
        if ctor == "EpisodicLifeEnv":
            continue  # LifeDone = False in Table 2
        factory = _get_atari_wrapper(ctor)
        if factory is None:
            continue
        try:
            wrapped = factory(wrapped, **kwargs)
        except Exception:
            continue
    if sticky and action_prob > 0:
        wrapped = StickyActionWrapper(wrapped, action_prob=action_prob, seed=seed)
    return wrapped


def _get_atari_wrapper(name: str) -> Optional[Callable[..., Any]]:
    for module_name in ("baselines.common.atari_wrappers", "stable_baselines3.common.atari_wrappers"):
        try:
            module = __import__(module_name, fromlist=[name])
            return getattr(module, name)
        except Exception:
            continue
    return None


def _ram_of(env: Any) -> Optional[np.ndarray]:
    cur = env
    for _ in range(6):
        for attr in ("ale", "env", "unwrapped", "_env"):
            cur = getattr(cur, attr, None) or cur
            ale = getattr(cur, "ale", None)
            if ale is not None and hasattr(ale, "getRAM"):
                try:
                    return np.asarray(ale.getRAM())
                except Exception:
                    return None
    return None


def _score_of(env: Any, info: Dict[str, Any]) -> float:
    if "score" in info and info["score"] is not None:
        return float(info["score"])
    cur = env
    for _ in range(6):
        ale = getattr(cur, "ale", None)
        if ale is not None and hasattr(ale, "getScore"):
            try:
                return float(ale.getScore())
            except Exception:
                break
        cur = getattr(cur, "env", None) or getattr(cur, "unwrapped", cur)
    return float(info.get("episode_return", 0.0))


def _split_reset(out: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    if isinstance(out, tuple) and len(out) == 2:
        obs, info = out
        return np.asarray(obs), dict(info or {})
    return np.asarray(out), {}


def _split_step(out: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
    if not isinstance(out, tuple):  # pragma: no cover
        raise TypeError("environment step must return a tuple")
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return np.asarray(obs), float(reward), bool(terminated), bool(truncated), dict(info or {})
    obs, reward, done, info = out  # legacy gym API
    return np.asarray(obs), float(reward), bool(done), False, dict(info or {})


# ---------------------------------------------------------------------------
# Dummy (dependency-free) environment for smoke tests
# ---------------------------------------------------------------------------
class DummyMontezumaEnv:
    """Scripted stand-in for Montezuma's Revenge used by CPU smoke tests.

    The environment implements the same interface as :class:`MontezumaEnv`
    (84x84x4 observations, sticky actions, room progression reported through
    ``info["room"]``, a room-completion signal driven by coins/items/exits and
    a terminal state after ``max_steps``).
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        max_steps: int = MAX_STEPS_PER_EPISODE,
        frame_stack: int = FRAME_STACK,
        action_prob: float = ACTION_PROB,
        far_room: int = ROOM7,
        room_steps: int = 40,
        reward_per_room: float = 100.0,
        episodes_to_solve: int = 1,
        **_: Any,
    ):
        self.rng = np.random.default_rng(seed if seed is not None else 0)
        self.max_steps = int(max_steps)
        self.action_prob = float(action_prob)
        self.far_room = int(far_room)
        self.room_steps = int(room_steps)
        self.reward_per_room = float(reward_per_room)
        self.episodes_to_solve = int(episodes_to_solve)
        self.stack = FrameStack(frame_stack)
        self.action_dim = NUM_ACTIONS
        self.episode = -1
        self._reset_state()

    # -- internals ---------------------------------------------------------
    def _reset_state(self) -> None:
        self.steps = 0
        self.room = START_ROOM
        self.room_t = 0
        self.score = 0.0
        self.items = 0
        self.episode_return = 0.0
        self.lives = 3
        self.last_action = 0
        self.tracker = RoomTracker(far_room=self.far_room)
        self.stack = FrameStack(self.stack.k)

    def _frame(self) -> np.ndarray:
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
        frame[:, :] = int((self.room * 7) % 255)
        frame[: max(1, self.room), :] = 255
        return frame

    # -- api ---------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, **kwargs):
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        self.episode += 1
        self._reset_state()
        obs = self.stack.reset(self._frame())
        info = {"room": self.room, "score": self.score, "lives": self.lives, "ram": self._ram()}
        info.update(self.tracker.info())
        return obs, info

    def _ram(self) -> np.ndarray:
        ram = np.zeros(128, dtype=np.uint8)
        ram[ROOM_RAM_ADDRESS] = self.room
        ram[0x86 : 0x86 + min(self.items, 5)] = 1
        return ram

    def step(self, action: int):
        if self.rng.random() < self.action_prob:
            action = self.last_action
        self.last_action = int(action)
        self.steps += 1
        self.room_t += 1
        reward = 0.0
        completed = False
        if self.room_t >= self.room_steps:
            # Finish the current room: coin + exit.
            self.score += self.reward_per_room
            if self.room % 3 == 0:
                self.items += 1
            reward = self.reward_per_room
            completed = True
            if self.room < MAX_ROOM:
                self.room += 1
                self.room_t = 0
        timeout = self.steps >= self.max_steps
        done = timeout or self.room >= MAX_ROOM + 1
        obs = self.stack.push(self._frame())
        self.episode_return += reward
        info = {
            "room": self.room,
            "score": self.score,
            "lives": self.lives,
            "ram": self._ram(),
            "episode_step": self.steps,
            "episode_return": self.episode_return,
        }
        info.update(self.tracker.update(room=self.room, score=self.score, items=self.items))
        info["room_completed"] = completed
        return obs, float(reward), bool(done), bool(timeout), info

    def close(self) -> None:  # pragma: no cover
        return None

    def seed(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))


# ---------------------------------------------------------------------------
# Factories / vectorisation
# ---------------------------------------------------------------------------
def make_env(
    seed: Optional[int] = None,
    stub: bool = False,
    far_room: int = ROOM7,
    **kwargs: Any,
) -> Any:
    """Create a Montezuma environment (real Atari or the dummy stand-in)."""
    if stub:
        kwargs.setdefault("far_room", far_room)
        return DummyMontezumaEnv(seed=seed, **kwargs)
    kwargs.pop("room_steps", None)
    kwargs.pop("reward_per_room", None)
    kwargs.pop("episodes_to_solve", None)
    return MontezumaEnv(seed=seed, far_room=far_room, **kwargs)


class MontezumaVecEnv:
    """Minimal synchronous vectorised environment (``NumEnv = 128``)."""

    def __init__(
        self,
        num_envs: int = 1,
        base_seed: Optional[int] = None,
        stub: bool = False,
        far_room: int = ROOM7,
        **env_kwargs: Any,
    ):
        self.num_envs = int(num_envs)
        self.envs = [
            make_env(
                seed=(None if base_seed is None else int(base_seed) + i),
                stub=stub,
                far_room=far_room,
                **env_kwargs,
            )
            for i in range(self.num_envs)
        ]

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs_list, info_list = [], []
        for env in self.envs:
            obs, info = env.reset()
            obs_list.append(obs)
            info_list.append(info)
        return np.stack(obs_list), _stack_infos(info_list)

    def step(self, actions: Sequence[int]):
        obs_list, rews, terms, truncs, info_list = [], [], [], [], []
        for env, act in zip(self.envs, actions):
            obs, rew, term, trunc, info = env.step(int(act))
            obs_list.append(obs)
            rews.append(rew)
            terms.append(term)
            truncs.append(trunc)
            info_list.append(info)
        return (
            np.stack(obs_list),
            np.asarray(rews, dtype=np.float32),
            np.asarray(terms, dtype=bool),
            np.asarray(truncs, dtype=bool),
            _stack_infos(info_list),
        )

    # Auto-reset handled by the trainer (keeps the room bookkeeping explicit).
    def close(self) -> None:
        for env in self.envs:
            env.close()

    @property
    def observation_shape(self) -> Tuple[int, int, int]:
        return OBS_SHAPE

    @property
    def observation_dim(self) -> int:
        return int(np.prod(OBS_SHAPE))

    @property
    def action_dim(self) -> int:
        return int(self.envs[0].action_dim)


def _stack_infos(infos: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    keys = set()
    for info in infos:
        keys.update(info.keys())
    out: Dict[str, Any] = {}
    for key in keys:
        values = [info.get(key) for info in infos]
        if all(isinstance(v, (int, float, bool, np.number)) or v is None for v in values):
            out[key] = np.asarray([0 if v is None else v for v in values])
        elif key == "ram":
            out[key] = np.stack([np.asarray(v) for v in values if v is not None])
        else:
            out[key] = values
    return out


# ---------------------------------------------------------------------------
# Rollouts / evaluation
# ---------------------------------------------------------------------------
def select_action(policy: Any, obs: np.ndarray, deterministic: bool = False, rng=None) -> int:
    """Duck-typed action selection supporting nn.Modules and plain callables."""
    if callable(policy) and not hasattr(policy, "act"):
        out = policy(obs)
        return int(np.asarray(_to_numpy(out)).reshape(-1)[0])
    for name in ("act", "select_action", "sample_action", "get_action"):
        fn = getattr(policy, name, None)
        if callable(fn):
            out = _call_with_supported_kwargs(fn, obs, deterministic=deterministic)
            if isinstance(out, tuple):
                out = out[0]
            if isinstance(out, dict):
                out = out.get("action", out)
            return int(np.asarray(_to_numpy(out)).reshape(-1)[0])
    if callable(policy):
        out = policy(obs)
        return int(np.asarray(_to_numpy(out)).reshape(-1)[0])
    raise TypeError(f"cannot select an action with policy of type {type(policy)!r}")


def _call_with_supported_kwargs(fn: Callable, *args, **kwargs):
    import inspect

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(*args)
    accepted = {
        k: v
        for k, v in kwargs.items()
        if k in sig.parameters or any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    }
    return fn(*args, **accepted)


def _to_numpy(value: Any) -> Any:
    for attr in ("detach", "cpu", "numpy"):
        fn = getattr(value, attr, None)
        if callable(fn) and attr in ("detach", "cpu", "numpy"):
            try:
                if attr == "numpy" and hasattr(value, "detach"):
                    continue
                value = fn()
            except Exception:
                continue
    return value


def collect_trajectories(
    policy: Any,
    env: Any = None,
    num_trajectories: int = 500,
    min_room: int = ROOM7,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    deterministic: bool = False,
    stub: bool = False,
    record_all: bool = False,
    max_episodes: Optional[int] = None,
    progress_fn: Optional[Callable[[int, int], None]] = None,
) -> List[List[Dict[str, Any]]]:
    """Collect trajectories, optionally keeping only the FAR part.

    This reproduces Appendix B.2: "we collected more than 500 trajectories
    sampled from a pre-trained PPO agent with RND that achieved an episode
    cumulative reward of around 7000".  When ``min_room`` is given (Room 7 in
    the main text) only the portion of each trajectory from that room onward
    is returned, which is the dataset used by behavioral cloning (M2).
    """
    if env is None:
        env = make_env(seed=seed, stub=stub)
    if seed is not None:
        set_seed(int(seed))
    rng = np.random.default_rng(seed if seed is not None else 0)
    trajectories: List[List[Dict[str, Any]]] = []
    episodes = 0
    limit = int(max_episodes) if max_episodes is not None else int(num_trajectories) * 20
    while len(trajectories) < num_trajectories and episodes < limit:
        episodes += 1
        obs, info = env.reset()
        traj: List[Dict[str, Any]] = []
        step = 0
        done = False
        while not done and (max_steps is None or step < max_steps):
            action = select_action(policy, obs, deterministic=deterministic, rng=rng)
            next_obs, reward, done, trunc, next_info = env.step(action)
            room = int(next_info.get("room", info.get("room", START_ROOM)))
            traj.append(
                {
                    "obs": np.asarray(obs).copy(),
                    "action": int(action),
                    "reward": float(reward),
                    "next_obs": np.asarray(next_obs).copy(),
                    "done": bool(done or trunc),
                    "room": room,
                    "step": step,
                }
            )
            obs, info = next_obs, next_info
            step += 1
        if not traj:
            continue
        if record_all:
            trajectories.append(traj)
        else:
            first_far = next((i for i, tr in enumerate(traj) if tr["room"] >= min_room), None)
            if first_far is None:
                continue
            trajectories.append(traj[first_far:])
        if progress_fn is not None:
            progress_fn(len(trajectories), num_trajectories)
    return trajectories


def evaluate_policy(
    policy: Any,
    env: Any = None,
    num_episodes: int = 100,
    seed: int = 0,
    deterministic: bool = True,
    stub: bool = False,
    far_room: int = ROOM7,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate a policy and report return plus the Room-7 success rate."""
    if env is None:
        env = make_env(seed=seed, stub=stub)
    rng = np.random.default_rng(seed)
    returns: List[float] = []
    lengths: List[int] = []
    rooms: List[int] = []
    room7_successes = 0
    traj_returns: List[float] = []
    for ep in range(int(num_episodes)):
        obs, info = env.reset()
        done = False
        ep_ret = 0.0
        steps = 0
        while not done and (max_steps is None or steps < max_steps):
            action = select_action(policy, obs, deterministic=deterministic, rng=rng)
            obs, reward, done, trunc, info = env.step(action)
            ep_ret += float(reward)
            steps += 1
        tracker = getattr(env, "tracker", None)
        if tracker is not None and tracker.room7_success(far_room):
            room7_successes += 1
        returns.append(ep_ret)
        lengths.append(steps)
        rooms.append(int(info.get("room", START_ROOM)))
        traj_returns.append(ep_ret)
    n = max(1, len(returns))
    return {
        "return_mean": float(np.mean(returns)) if returns else 0.0,
        "return_std": float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0,
        "return_min": float(np.min(returns)) if returns else 0.0,
        "return_max": float(np.max(returns)) if returns else 0.0,
        "room7_success_rate": float(room7_successes) / n,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "max_room": float(np.max(rooms)) if rooms else 0.0,
        "episodes": float(len(returns)),
    }


def room7_success_rate(
    policy: Any,
    env: Any = None,
    num_episodes: int = 100,
    seed: int = 0,
    stub: bool = False,
    far_room: int = ROOM7,
) -> float:
    """Room-7 success rate (Figure 6, computed every 5M steps)."""
    stats = evaluate_policy(
        policy, env=env, num_episodes=num_episodes, seed=seed, stub=stub, far_room=far_room
    )
    return float(stats["room7_success_rate"])


def room_visitation(policy: Any, env: Any = None, num_episodes: int = 1, seed: int = 0, stub: bool = False):
    """Aggregate room-visitation counts (Appendix E figures)."""
    if env is None:
        env = make_env(seed=seed, stub=stub)
    total = np.zeros(MAX_ROOM, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for _ in range(int(num_episodes)):
        obs, info = env.reset()
        done = False
        while not done:
            action = select_action(policy, obs, deterministic=False, rng=rng)
            obs, reward, done, trunc, info = env.step(action)
        tracker = getattr(env, "tracker", None)
        if tracker is not None:
            total += tracker.visitation(MAX_ROOM)
    return total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Montezuma's Revenge environment utilities")
    parser.add_argument("--stub", action="store_true", help="use the dependency-free dummy env")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--far-room", type=int, default=ROOM7)
    parser.add_argument("--random-policy", action="store_true", help="roll out a random policy")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    env = make_env(seed=args.seed, stub=args.stub, far_room=args.far_room)
    if args.random_policy:
        policy = lambda obs: np.random.randint(env.action_dim)  # noqa: E731
    else:
        policy = lambda obs: 0  # noqa: E731
    stats = evaluate_policy(
        policy,
        env=env,
        num_episodes=args.episodes,
        seed=args.seed,
        max_steps=args.max_steps,
        far_room=args.far_room,
    )
    for key, value in stats.items():
        print(f"{key}: {value}")
    env.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
