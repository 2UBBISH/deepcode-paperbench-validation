"""Uniform policy interface used across the reproduction.

Everything that acts in an environment implements ``act(obs, deterministic)``,
so the mask network training (Algorithm 1), the critical state selection, the
refining algorithms and the fidelity metric can be used interchangeably with
our own networks or with Stable-Baselines3 models.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch

from rice.networks import ActorCritic


class TorchPolicy:
    """Wraps an :class:`ActorCritic` into the common policy interface."""

    def __init__(self, net: ActorCritic, device: str = "cpu"):
        self.net = net.to(device)
        self.device = device

    def act(self, obs, deterministic: bool = False) -> np.ndarray:
        action, _, _ = self.net.act(np.asarray(obs, dtype=np.float32), deterministic)
        return action

    def __call__(self, obs) -> np.ndarray:  # pragma: no cover - convenience
        return self.act(obs, deterministic=True)


class CallablePolicy:
    def __init__(self, fn: Callable[[np.ndarray], np.ndarray]):
        self.fn = fn

    def act(self, obs, deterministic: bool = True) -> np.ndarray:
        return np.asarray(self.fn(np.asarray(obs, dtype=np.float32)))


class SB3Policy:
    """Adapter for a Stable-Baselines3 model (``model.predict``)."""

    def __init__(self, model):
        self.model = model

    def act(self, obs, deterministic: bool = False) -> np.ndarray:
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return np.asarray(action)


def load_policy(
    path: str,
    obs_dim: Optional[int] = None,
    act_dim: Optional[int] = None,
    discrete: Optional[bool] = None,
    hidden=(64, 64),
    device: str = "cpu",
) -> TorchPolicy:
    """Load a checkpoint produced by :mod:`rice.training`.

    The checkpoint stores the architecture, hence ``obs_dim``/``act_dim`` are
    only needed for legacy checkpoints (raw ``state_dict``).
    """
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        net = ActorCritic(
            ckpt["obs_dim"],
            ckpt["act_dim"],
            hidden=ckpt.get("hidden", hidden),
            discrete=ckpt.get("discrete", True),
        )
        net.load_state_dict(ckpt["state_dict"])
    else:  # raw state dict
        if obs_dim is None or act_dim is None or discrete is None:
            raise ValueError("architecture arguments required for raw state dicts")
        net = ActorCritic(obs_dim, act_dim, hidden=hidden, discrete=discrete)
        net.load_state_dict(ckpt)
    net.eval()
    return TorchPolicy(net, device=device)
