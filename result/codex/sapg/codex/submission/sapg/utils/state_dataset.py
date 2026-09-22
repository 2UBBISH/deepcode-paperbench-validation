"""State-visitation recorder used by the diversity analyses of Sec. 6.4.

Figure 7 (PCA) and Figure 8 (MLP) compare the *states visited during training*
by SAPG, PPO and a randomly initialised policy.  Trainers therefore dump the
on-policy state batch every ``record_interval`` iterations; this module also
contains the helper that rolls out a randomly initialised policy.
"""

from __future__ import annotations

import glob
import os
from typing import List, Optional

import numpy as np
import torch


def record_states(logdir: str, iteration: int, states: np.ndarray, tag: str = "train") -> str:
    out_dir = os.path.join(logdir, "states")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{tag}_{iteration:07d}.npy")
    np.save(path, states.astype(np.float32))
    return path


def load_state_dataset(path_or_dir: str, tag: Optional[str] = None, max_batches: Optional[int] = None):
    if os.path.isdir(path_or_dir):
        # accept either a directory of .npy batches or a run directory that
        # contains the `states/` sub-directory written by the trainers
        states_dir = os.path.join(path_or_dir, "states")
        root = states_dir if os.path.isdir(states_dir) else path_or_dir
        pattern = os.path.join(root, f"{tag}_*.npy" if tag else "*.npy")
        files = sorted(glob.glob(pattern))
    else:
        files = [path_or_dir]
    if max_batches is not None:
        files = files[:max_batches]
    if not files:
        raise FileNotFoundError(f"No state files found under {path_or_dir}")
    return np.concatenate([np.load(f) for f in files], axis=0)


def collect_random_policy_states(
    env,
    num_steps: int,
    seed: int = 0,
    record_every: int = 1,
) -> List[np.ndarray]:
    """Roll out a uniformly random policy and return the visited states."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs = env.reset()
    states: List[np.ndarray] = []
    for step in range(num_steps):
        actions = torch.rand((env.num_envs, env.action_dim), device=env.device) * 2.0 - 1.0
        step_result = env.step(actions)
        obs = step_result.obs
        if step % record_every == 0:
            states.append(obs.detach().cpu().numpy().astype(np.float32))
    return states
