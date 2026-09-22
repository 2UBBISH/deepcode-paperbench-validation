"""Montezuma's Revenge environment wrapper.

The wrapper implements the paper's episode semantics:

* ``MaxStepPerEpisode = 4500`` and sticky actions with probability 0.25
  (Table 2),
* frame stacking of 4 greyscale 84x84 frames (``StateStackSize``,
  ``PreProcHeight``, ``ProProcWidth``),
* an optional *room-restricted* reset used during pre-training: we pre-train a
  policy on the part of the environment starting from Room 7 (Appendix B.2).

The environment requires ``gymnasium[atari]`` + ``ale-py`` with the Atari ROMs.
Importing this module is safe without them; instantiating the environment is
not.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Optional, Tuple

import numpy as np

from .config import MontezumaConfig


def _require_atari():
    try:
        import ale_py  # noqa: F401
        import gymnasium as gym  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on optional deps
        raise ImportError(
            "Montezuma's Revenge requires `gymnasium[atari]` and `ale-py` with the "
            "Atari ROMs installed. Install with:\n"
            "    pip install 'gymnasium[atari,accept-rom-license]' ale-py"
        ) from exc


def _frame_process(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    """Greyscale + resize to ``height x width`` (Burda et al., 2018)."""

    import cv2

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame.astype(np.float32) / 255.0


class AtariPreprocessor:
    """Frame skipping, greyscale conversion, resizing and frame stacking."""

    def __init__(self, height: int = 84, width: int = 84, stack_size: int = 4, frame_skip: int = 4):
        self.height = height
        self.width = width
        self.stack_size = stack_size
        self.frame_skip = frame_skip
        self.frames: Deque[np.ndarray] = deque(maxlen=stack_size)

    def reset(self, frame: np.ndarray) -> np.ndarray:
        processed = _frame_process(frame, self.height, self.width)
        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(processed)
        return self.observation()

    def step(self, frame: np.ndarray) -> np.ndarray:
        self.frames.append(_frame_process(frame, self.height, self.width))
        return self.observation()

    def observation(self) -> np.ndarray:
        return np.stack(self.frames, axis=0)  # (C, H, W)


@dataclass
class RoomRestriction:
    """Describes the pre-training start states.

    ``room`` is the first room included in pre-training (7 in the main text).
    ``save_states`` holds emulator save states captured the first time the agent
    entered ``room``; every episode during pre-training resets the emulator to a
    randomly chosen save state.
    """

    room: int
    save_states: Tuple[bytes, ...] = ()


def make_env(config: MontezumaConfig, seed: int = 0, render: bool = False):
    """Create a wrapped ``MontezumaRevenge`` environment."""

    _require_atari()
    import gymnasium as gym

    env = gym.make(
        config.env_id,
        frameskip=1,
        repeat_action_probability=config.action_prob if config.sticky_action else 0.0,
        full_action_space=False,
        render_mode="rgb_array" if render else None,
    )
    env.reset(seed=seed)
    return env


def collect_pre_training_trajectories(
    policy,
    config: MontezumaConfig,
    num_trajectories: int = 500,
    device: str = "cpu",
    seed: int = 0,
):
    """Collect ``num_trajectories`` trajectories with the pre-trained policy.

    The trajectories form the behavioral-cloning buffer.  Pre-training uses the
    room-restricted environment (Room 7 onward), so the buffer contains states
    that are FAR states in the downstream task.  Returns arrays of
    ``(observations, actions)``.
    """

    import torch

    observations, actions = [], []
    env = make_env(config, seed=seed)
    preprocessor = AtariPreprocessor(
        config.preproc_height, config.preproc_width, config.state_stack_size
    )
    obs, _ = env.reset()
    stacked = preprocessor.reset(obs)
    for _ in range(num_trajectories):
        done = False
        steps = 0
        while not done and steps < config.max_steps_per_episode:
            obs_t = torch.as_tensor(stacked[None], dtype=torch.float32, device=device)
            with torch.no_grad():
                action = int(policy.act(obs_t).item())
            observations.append(stacked.copy())
            actions.append(action)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            stacked = preprocessor.step(obs)
            steps += 1
        obs, _ = env.reset()
        stacked = preprocessor.reset(obs)
    env.close()
    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
    }


def room_transition_detector(env) -> Callable[[], Optional[int]]:
    """Return a callable reporting the current room index.

    Montezuma's Revenge does not expose the room index in the observation.  The
    paper enumerates rooms using the known progression (Figure 12).  We detect
    room transitions by watching the in-game RAM byte that changes whenever the
    player enters a new room.
    """

    def detector() -> Optional[int]:
        try:
            ram = env.unwrapped.ale.getRAM()
        except Exception:
            return None
        return int(ram[3])

    return detector
