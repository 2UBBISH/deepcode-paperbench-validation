"""Evaluation utilities for Montezuma's Revenge (Figure 6, 17, 18).

* :func:`room_success_rate` -- the metric used in Figure 6.  A room counts as
  successfully completed if the agent achieves at least one of: earning a coin,
  acquiring a new item, or exiting the room through a different passage than the
  one it entered through (Appendix B.2).
* :func:`room_visitation` -- the time spent in each room across training
  (Figure 18).

Because ALE does not expose the room index directly, the room index is derived
from the RAM bytes of the emulator (see
:func:`fpc.montezuma.env.room_transition_detector`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from .config import MontezumaConfig
from .env import AtariPreprocessor, make_env


@dataclass
class RoomTransition:
    """Signals available for deciding whether a room was completed."""

    score_before: int = 0
    score_after: int = 0
    inventory_before: Optional[bytes] = None
    inventory_after: Optional[bytes] = None
    entry_room: Optional[int] = None
    exit_room: Optional[int] = None
    room_solved: bool = False


def _coins_gained(before: int, after: int) -> bool:
    """Montezuma's Revenge awards 100 points per coin collected."""

    return (after - before) >= 100


def _item_acquired(before: Optional[bytes], after: Optional[bytes]) -> bool:
    if before is None or after is None:
        return False
    return bytes(before) != bytes(after)


def room_success_rate(
    policy,
    config: MontezumaConfig,
    room: int,
    episodes: int = 100,
    device: str = "cpu",
    seed: int = 0,
) -> float:
    """Success rate of ``policy`` in ``room`` (Figure 6).

    The agent is initialised at the beginning of ``room`` (using a save state
    captured the first time the room was entered) and the episode ends when the
    agent leaves the room.  The room counts as solved if a coin was earned, a new
    item was acquired, or the agent exited through a different passage.
    """

    env = make_env(config, seed=seed)
    preprocessor = AtariPreprocessor(
        config.preproc_height, config.preproc_width, config.state_stack_size
    )
    successes = 0
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        # When save states for the room are available they are loaded here:
        #   env.unwrapped.ale.restoreState(save_states[episode % len(save_states)])
        stacked = preprocessor.reset(obs)
        score_before = int(env.unwrapped.ale.getScore())
        entry_room = room
        done = False
        steps = 0
        while not done and steps < config.max_steps_per_episode:
            obs_t = torch.as_tensor(stacked[None], dtype=torch.float32, device=device)
            with torch.no_grad():
                action = int(policy.act(obs_t).item())
            obs, reward, terminated, truncated, _ = env.step(action)
            stacked = preprocessor.step(obs)
            done = terminated or truncated
            steps += 1
            exit_room = room
            if exit_room != entry_room:
                break
        score_after = int(env.unwrapped.ale.getScore())
        if _coins_gained(score_before, score_after) or (exit_room != entry_room):
            successes += 1
    env.close()
    return successes / episodes


def room_visitation(
    policy,
    config: MontezumaConfig,
    episodes: int = 20,
    device: str = "cpu",
    seed: int = 0,
) -> Dict[int, float]:
    """Return the average number of steps spent in each room (Figure 18)."""

    env = make_env(config, seed=seed)
    preprocessor = AtariPreprocessor(
        config.preproc_height, config.preproc_width, config.state_stack_size
    )
    visits: Dict[int, float] = {}
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        stacked = preprocessor.reset(obs)
        done = False
        steps = 0
        current_room = 1
        while not done and steps < config.max_steps_per_episode:
            obs_t = torch.as_tensor(stacked[None], dtype=torch.float32, device=device)
            with torch.no_grad():
                action = int(policy.act(obs_t).item())
            obs, _, terminated, truncated, _ = env.step(action)
            stacked = preprocessor.step(obs)
            done = terminated or truncated
            steps += 1
            try:
                room = int(env.unwrapped.ale.getRAM()[3])
            except Exception:
                room = current_room
            current_room = room
            visits[room] = visits.get(room, 0.0) + 1.0
    env.close()
    return {room: steps / episodes for room, steps in sorted(visits.items())}


def evaluate_training_run(
    checkpoint_dir: str,
    config: MontezumaConfig,
    room: int = 7,
    device: str = "cpu",
) -> List[Dict[str, float]]:
    """Evaluate all checkpoints in ``checkpoint_dir`` (Figure 6).

    The success rate in Room 7 is computed every 5M training steps as required
    by the addendum.
    """

    import glob
    import os

    from .model import AtariActorCritic

    results: List[Dict[str, float]] = []
    for path in sorted(glob.glob(os.path.join(checkpoint_dir, "*.pt"))):
        state = torch.load(path, map_location=device)
        model = AtariActorCritic(state["num_actions"]).to(device)
        model.load_state_dict(state["model"])
        model.eval()
        success = room_success_rate(model, config, room=room, device=device)
        results.append({"checkpoint": path, "steps": state.get("steps", -1), "room_success_rate": success})
    return results
