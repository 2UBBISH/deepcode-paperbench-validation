"""Prior reward-function mixture used by FRE.

The paper samples reward functions uniformly from three unsupervised families:

1. Singleton goal-reaching rewards
       r(s) = 0 if ||s - g|| < threshold else -1
2. Random linear rewards
       r(s) = w^T s, with w sampled uniformly and a sparse mask applied to bias
       the sampler towards simple functions.
3. Random MLP rewards
       A two-layer MLP with random initialization, hidden size 64, ReLU and a
       scalar output.

All reward functions expose a ``__call__(states) -> torch.Tensor`` interface so
they can be used interchangeably by the FRE VAE encoder/decoder and the IQL
trainer.
"""

from __future__ import annotations

import abc
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from fre.config import RewardSamplerConfig


# ---------------------------------------------------------------------------
# Reward function objects
# ---------------------------------------------------------------------------
class RewardFunction(abc.ABC):
    """Base class for all sampled reward functions."""

    name: str = "reward"

    @abc.abstractmethod
    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        """Return a scalar reward for every state in ``states``.

        Parameters
        ----------
        states:
            A tensor of shape ``(..., state_dim)``.

        Returns
        -------
        rewards:
            A tensor of shape ``(...,)`` (or ``(..., 1)``; the caller should
            squeeze when necessary).
        """

    def to(self, device: torch.device) -> "RewardFunction":
        """Move any internal parameters to ``device``. Default is a no-op."""
        return self


class SingletonRewardFunction(RewardFunction):
    """Sparse goal-reaching reward: 0 within a threshold, -1 otherwise."""

    name = "singleton"

    def __init__(self, goal: torch.Tensor, threshold: float = 1.0):
        self.goal = goal
        self.threshold = float(threshold)

    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        goal = self.goal.to(device=states.device, dtype=states.dtype)
        dist = torch.norm(states - goal, dim=-1)
        return torch.where(dist < self.threshold, torch.zeros_like(dist), -torch.ones_like(dist))

    def to(self, device: torch.device) -> "SingletonRewardFunction":
        self.goal = self.goal.to(device)
        return self

    def __repr__(self) -> str:
        return f"SingletonRewardFunction(threshold={self.threshold})"


class LinearRewardFunction(RewardFunction):
    """Linear reward ``w^T s`` with an optional sparse mask."""

    name = "linear"

    def __init__(self, weights: torch.Tensor, mask: Optional[torch.Tensor] = None):
        self.weights = weights
        self.mask = mask

    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        w = self.weights.to(device=states.device, dtype=states.dtype)
        if self.mask is not None:
            mask = self.mask.to(device=states.device, dtype=states.dtype)
            w = w * mask
        return states @ w

    def to(self, device: torch.device) -> "LinearRewardFunction":
        self.weights = self.weights.to(device)
        if self.mask is not None:
            self.mask = self.mask.to(device)
        return self

    def __repr__(self) -> str:
        active = int(self.mask.sum().item()) if self.mask is not None else self.weights.numel()
        return f"LinearRewardFunction(active_dims={active}/{self.weights.numel()})"


class MLPRewardFunction(RewardFunction):
    """Random two-layer MLP reward function."""

    name = "mlp"

    def __init__(self, mlp: nn.Module):
        self.mlp = mlp

    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        out = self.mlp(states)
        return out.squeeze(-1)

    def to(self, device: torch.device) -> "MLPRewardFunction":
        self.mlp = self.mlp.to(device)
        return self

    def __repr__(self) -> str:
        return "MLPRewardFunction(hidden_size=64)"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def _cfg_float(cfg: RewardSamplerConfig, name: str, default: float) -> float:
    return float(getattr(cfg, name, default))


def _cfg_int(cfg: RewardSamplerConfig, name: str, default: int) -> int:
    return int(getattr(cfg, name, default))


def _cfg_bool(cfg: RewardSamplerConfig, name: str, default: bool) -> bool:
    return bool(getattr(cfg, name, default))


def _active_families(cfg: RewardSamplerConfig) -> list[str]:
    """Return the enabled reward-family names according to the config."""
    families: list[str] = []
    # Accept both *_enabled and bare family-name booleans to stay robust to
    # minor naming differences in fre/config.py.
    if _cfg_bool(cfg, "singleton_enabled", True) and _cfg_bool(cfg, "singleton", True):
        families.append("singleton")
    if _cfg_bool(cfg, "linear_enabled", True) and _cfg_bool(cfg, "linear", True):
        families.append("linear")
    if _cfg_bool(cfg, "mlp_enabled", True) and _cfg_bool(cfg, "mlp", True):
        families.append("mlp")
    return families


def _family_weights(cfg: RewardSamplerConfig, families: Sequence[str]) -> np.ndarray:
    """Return a probability vector over ``families``."""
    raw = getattr(cfg, "family_weights", None)
    if raw is None:
        raw = getattr(cfg, "weights", None)
    if raw is None:
        return np.ones(len(families), dtype=np.float64) / len(families)

    if isinstance(raw, dict):
        vals = np.array([float(raw.get(name, 1.0)) for name in families], dtype=np.float64)
    else:
        vals = np.array([float(x) for x in raw], dtype=np.float64)
        if vals.size == 3 and len(families) < 3:
            # Keep only the entries for enabled families.  The config uses
            # [singleton, linear, mlp] ordering by convention.
            mapping = {"singleton": 0, "linear": 1, "mlp": 2}
            vals = np.array([vals[mapping[name]] for name in families], dtype=np.float64)

    total = vals.sum()
    if total <= 0:
        return np.ones(len(families), dtype=np.float64) / len(families)
    return vals / total


def _choose_family(cfg: RewardSamplerConfig, rng: np.random.Generator) -> str:
    families = _active_families(cfg)
    if not families:
        # Fall back to singleton if everything is disabled; never return an
        # invalid reward function.
        families = ["singleton"]
    probs = _family_weights(cfg, families)
    return families[int(rng.choice(len(families), p=probs))]


def _make_singleton(states: torch.Tensor, cfg: RewardSamplerConfig, rng: np.random.Generator) -> RewardFunction:
    idx = int(rng.integers(0, states.shape[0]))
    goal = states[idx].clone()
    threshold = _cfg_float(cfg, "singleton_threshold", 1.0)
    return SingletonRewardFunction(goal, threshold)


def _make_linear(states: torch.Tensor, cfg: RewardSamplerConfig, rng: np.random.Generator) -> RewardFunction:
    state_dim = int(states.shape[-1])
    # Uniformly sampled direction; normalize so reward magnitudes are bounded
    # and the discretization layer sees a consistent range.
    w = torch.from_numpy(rng.uniform(-1.0, 1.0, size=(state_dim,)).astype(np.float32))
    norm = w.norm()
    if norm > 1e-8:
        w = w / norm

    active_fraction = _cfg_float(cfg, "linear_active_fraction", 0.3)
    active_fraction = float(np.clip(active_fraction, 0.0, 1.0))
    if active_fraction < 1.0:
        mask_vals = rng.random(state_dim) < active_fraction
        if not mask_vals.any():
            mask_vals[int(rng.integers(0, state_dim))] = True
        mask = torch.from_numpy(mask_vals.astype(np.float32))
    else:
        mask = torch.ones(state_dim, dtype=torch.float32)

    scale = _cfg_float(cfg, "linear_scale", 1.0)
    w = w * scale
    return LinearRewardFunction(w, mask)


def _make_mlp(states: torch.Tensor, cfg: RewardSamplerConfig, rng: np.random.Generator) -> RewardFunction:
    state_dim = int(states.shape[-1])
    hidden_size = _cfg_int(cfg, "mlp_hidden_size", 64)
    net = nn.Sequential(
        nn.Linear(state_dim, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 1),
    )

    # Random initialization.  We use the standard PyTorch default but scale the
    # final layer slightly so rewards have a bounded dynamic range.
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    net.apply(_init)
    output_scale = _cfg_float(cfg, "mlp_output_scale", 1.0)
    if output_scale != 1.0:
        with torch.no_grad():
            net[-1].weight.mul_(output_scale)
    return MLPRewardFunction(net)


# ---------------------------------------------------------------------------
# Main sampling API
# ---------------------------------------------------------------------------
def sample_reward(
    states: torch.Tensor,
    cfg: RewardSamplerConfig,
    rng: Optional[np.random.Generator] = None,
) -> RewardFunction:
    """Sample a reward function from the prior mixture.

    Parameters
    ----------
    states:
        A representative set of dataset states used for sampling goal states.
    cfg:
        Reward-sampler configuration (``fre.config.RewardSamplerConfig``).
    rng:
        Optional NumPy generator for reproducibility.

    Returns
    -------
    A callable :class:`RewardFunction`.
    """
    if rng is None:
        rng = np.random.default_rng()

    if not isinstance(states, torch.Tensor):
        states = torch.as_tensor(states, dtype=torch.float32)
    if states.ndim == 1:
        states = states.unsqueeze(0)
    if states.shape[0] == 0:
        raise ValueError("sample_reward requires at least one state.")

    family = _choose_family(cfg, rng)
    if family == "singleton":
        reward_fn = _make_singleton(states, cfg, rng)
    elif family == "linear":
        reward_fn = _make_linear(states, cfg, rng)
    elif family == "mlp":
        reward_fn = _make_mlp(states, cfg, rng)
    else:  # pragma: no cover - defensive fallback
        reward_fn = _make_singleton(states, cfg, rng)
    return reward_fn


def sample_rewards_batch(
    states: torch.Tensor,
    cfg: RewardSamplerConfig,
    num_rewards: int,
    rng: Optional[np.random.Generator] = None,
) -> list[RewardFunction]:
    """Sample ``num_rewards`` reward functions, mostly useful for ablations."""
    if rng is None:
        rng = np.random.default_rng()
    return [sample_reward(states, cfg, rng) for _ in range(num_rewards)]


__all__ = [
    "RewardFunction",
    "SingletonRewardFunction",
    "LinearRewardFunction",
    "MLPRewardFunction",
    "sample_reward",
    "sample_rewards_batch",
]
