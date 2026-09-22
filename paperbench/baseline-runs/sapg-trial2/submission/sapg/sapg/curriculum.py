"""Success-tolerance curriculum for hard manipulation tasks.

The SAPG paper uses a curriculum on the success tolerance ``delta`` for the
hard AllegroKuka tasks (Regrasping / Throw / Reorientation).  The tolerance
starts at 7.5cm and is annealed down to 1cm.  Whenever the average number of
successes per episode exceeds a threshold (3 in the paper), the tolerance is
reduced by 10%.

This module provides a small, self-contained scheduler that can be driven by
the training loop and applied to any environment exposing an
``update_curriculum`` method (see ``sapg.envs``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class CurriculumConfig:
    """Configuration for the success-tolerance curriculum.

    Attributes:
        initial_tolerance: Starting success tolerance (metres).  Paper: 0.075.
        min_tolerance: Lower bound on the tolerance (metres).  Paper: 0.01.
        decay: Multiplicative decay applied per curriculum step.  Paper: 0.9.
        success_threshold: Average successes per episode above which the
            tolerance is annealed.  Paper: 3.
        warmup_episodes: Number of episodes to observe before allowing the
            first curriculum step (avoids annealing on noisy early estimates).
        enabled: Whether the curriculum is active.
    """

    initial_tolerance: float = 0.075
    min_tolerance: float = 0.01
    decay: float = 0.9
    success_threshold: float = 3.0
    warmup_episodes: int = 0
    enabled: bool = True


class SuccessToleranceCurriculum:
    """Anneals the success tolerance of an environment based on success rate.

    The scheduler tracks the average number of successes per episode (as
    reported by the environment's ``info["success"]`` signal) and, once it
    exceeds ``success_threshold``, multiplies the current tolerance by
    ``decay`` (clamped at ``min_tolerance``).

    Example:
        >>> curriculum = SuccessToleranceCurriculum()
        >>> tol = curriculum.update(avg_successes_per_episode=4.0)
        >>> tol < 0.075
        True
    """

    def __init__(self, config: Optional[CurriculumConfig] = None, **kwargs: Any) -> None:
        if config is None:
            config = CurriculumConfig(**kwargs)
        self.config = config
        self.tolerance: float = float(config.initial_tolerance)
        self.num_steps: int = 0
        self.episodes_seen: int = 0
        self.history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def update(self, avg_successes_per_episode: float) -> float:
        """Advance the curriculum by one evaluation interval.

        Args:
            avg_successes_per_episode: Mean number of successes achieved per
                episode over the most recent evaluation window.

        Returns:
            The (possibly updated) current success tolerance.
        """
        self.episodes_seen += 1
        if not self.config.enabled:
            return self.tolerance

        if self.episodes_seen <= self.config.warmup_episodes:
            return self.tolerance

        if avg_successes_per_episode > self.config.success_threshold:
            new_tol = max(
                self.config.min_tolerance,
                self.tolerance * self.config.decay,
            )
            if new_tol < self.tolerance:
                self.tolerance = new_tol
                self.num_steps += 1

        self.history.append(
            {
                "tolerance": self.tolerance,
                "avg_successes": float(avg_successes_per_episode),
                "num_steps": float(self.num_steps),
            }
        )
        return self.tolerance

    def step(self, avg_successes_per_episode: float) -> float:
        """Alias for :meth:`update`."""
        return self.update(avg_successes_per_episode)

    def apply(self, env: Any) -> float:
        """Push the current tolerance into an environment.

        Works with any env exposing either an ``update_curriculum`` method or
        a ``config.success_tolerance`` attribute (the SAPG envs expose both).

        Args:
            env: Environment instance.

        Returns:
            The tolerance that was applied.
        """
        if hasattr(env, "update_curriculum"):
            # Envs implement their own annealing; sync our value first.
            try:
                env.config.success_tolerance = self.tolerance
            except AttributeError:
                pass
            return float(self.tolerance)
        if hasattr(env, "config") and hasattr(env.config, "success_tolerance"):
            env.config.success_tolerance = self.tolerance
        return float(self.tolerance)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def done(self) -> bool:
        """Whether the tolerance has reached its minimum."""
        return self.tolerance <= self.config.min_tolerance + 1e-12

    def state_dict(self) -> Dict[str, Any]:
        return {
            "tolerance": self.tolerance,
            "num_steps": self.num_steps,
            "episodes_seen": self.episodes_seen,
            "config": vars(self.config),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.tolerance = float(state.get("tolerance", self.config.initial_tolerance))
        self.num_steps = int(state.get("num_steps", 0))
        self.episodes_seen = int(state.get("episodes_seen", 0))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"SuccessToleranceCurriculum(tolerance={self.tolerance:.4f}, "
            f"steps={self.num_steps}, done={self.done})"
        )


def compute_avg_successes_per_episode(
    success_flags: torch.Tensor,
    episode_lengths: torch.Tensor,
) -> float:
    """Utility to convert per-step success flags into successes per episode.

    Args:
        success_flags: ``(num_envs, horizon)`` boolean/float tensor of success
            indicators recorded during a rollout.
        episode_lengths: ``(num_envs,)`` number of steps each env contributed.

    Returns:
        Mean number of successes per episode (float).
    """
    if success_flags.numel() == 0:
        return 0.0
    flags = success_flags.float()
    per_env = flags.sum(dim=-1)
    lengths = episode_lengths.float().clamp(min=1.0)
    # Normalise by the fraction of an episode observed, then average.
    per_episode = per_env / lengths
    return float(per_episode.mean().item())


__all__ = [
    "CurriculumConfig",
    "SuccessToleranceCurriculum",
    "compute_avg_successes_per_episode",
]
