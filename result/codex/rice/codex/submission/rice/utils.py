"""Small helpers shared across the package."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_global_seeds(seed: int, env=None) -> None:
    """Seed python, numpy, torch and (optionally) an environment.

    Reproducibility matters twice in a reproduction: it makes the reported
    numbers repeatable, and it makes the *random* baseline of the fidelity
    metric (and the random actions used to blind the agent in Algorithm 1)
    depend only on the seed of the run.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if env is not None:
        try:
            env.reset(seed=seed)
        except TypeError:  # pragma: no cover - env without seeding support
            env.reset()
        action_space = getattr(env, "action_space", None)
        if action_space is not None and hasattr(action_space, "seed"):
            try:
                action_space.seed(seed + 1)
            except Exception:  # pragma: no cover - defensive
                pass
