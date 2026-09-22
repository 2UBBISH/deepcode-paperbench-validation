"""Success-tolerance curriculum for SAPG.

The paper uses a simple success-based curriculum on the AllegroKuka tasks
(Regrasping, Throw, Reorientation):

    * The task tolerance ``delta`` starts at 7.5 cm.
    * Whenever the average number of successes per episode exceeds 3, the
      tolerance is decreased by 10% (multiplicatively).
    * The tolerance is floored at 1 cm.
    * After every success the target / object is re-randomised.

This module implements that logic in a task-agnostic way so it can be used by
all environments (AllegroKuka, Shadow Hand, Allegro Hand).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CurriculumConfig:
    """Configuration for :class:`SuccessToleranceCurriculum`.

    Attributes:
        initial_delta: Starting tolerance (metres). Paper: 0.075 (7.5 cm).
        min_delta: Lower bound on the tolerance (metres). Paper: 0.01 (1 cm).
        decrease_factor: Multiplicative decrease applied when the success
            threshold is crossed. Paper: 0.9 (-10%).
        success_threshold: Average successes per episode above which the
            tolerance is decreased. Paper: 3.0.
        warmup_episodes: Number of episodes to observe before allowing the
            first decrease (avoids triggering on noisy early statistics).
        ema_alpha: Exponential moving average coefficient used to smooth the
            per-episode success signal. ``1.0`` disables smoothing.
        enabled: Whether the curriculum is active.
    """

    initial_delta: float = 0.075
    min_delta: float = 0.01
    decrease_factor: float = 0.9
    success_threshold: float = 3.0
    warmup_episodes: int = 0
    ema_alpha: float = 1.0
    enabled: bool = True


class SuccessToleranceCurriculum:
    """Success-based tolerance curriculum.

    The curriculum tracks the (optionally smoothed) average number of
    successes per episode.  Once this average exceeds ``success_threshold``
    the tolerance ``delta`` is multiplied by ``decrease_factor`` (clamped to
    ``min_delta``).

    Example:
        >>> cur = SuccessToleranceCurriculum()
        >>> cur.delta
        0.075
        >>> for _ in range(10):
        ...     cur.update(4.0)  # 4 successes per episode on average
        >>> cur.delta < 0.075
        True
    """

    def __init__(self, config: Optional[CurriculumConfig] = None, **kwargs):
        if config is None:
            config = CurriculumConfig(**kwargs)
        self.config = config

        self.delta: float = float(config.initial_delta)
        self.episodes: int = 0
        self.total_successes: float = 0.0
        self.avg_successes: float = 0.0
        self.num_decreases: int = 0
        self.history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def update(self, successes: float, num_episodes: int = 1) -> bool:
        """Register the outcome of one (or more) finished episodes.

        Args:
            successes: Number of successes achieved in the episode(s).
            num_episodes: How many episodes this observation covers.

        Returns:
            ``True`` if the tolerance was decreased by this call.
        """
        if num_episodes <= 0:
            return False

        self.episodes += int(num_episodes)
        self.total_successes += float(successes)

        per_episode = float(successes) / float(num_episodes)

        alpha = float(self.config.ema_alpha)
        if alpha >= 1.0 or self.avg_successes == 0.0:
            self.avg_successes = per_episode
        else:
            self.avg_successes = (1.0 - alpha) * self.avg_successes + alpha * per_episode

        decreased = False
        if (
            self.config.enabled
            and self.episodes >= self.config.warmup_episodes
            and self.avg_successes > self.config.success_threshold
            and self.delta > self.config.min_delta
        ):
            new_delta = max(self.config.min_delta, self.delta * self.config.decrease_factor)
            if new_delta < self.delta:
                self.delta = new_delta
                self.num_decreases += 1
                decreased = True

        self.history.append(
            {
                "episode": float(self.episodes),
                "successes": float(successes),
                "avg_successes": float(self.avg_successes),
                "delta": float(self.delta),
                "decreased": float(decreased),
            }
        )
        return decreased

    def update_batch(self, successes_per_env: List[float]) -> bool:
        """Convenience wrapper for a batch of finished episodes.

        Args:
            successes_per_env: List with the number of successes of each
                finished episode in this iteration.

        Returns:
            ``True`` if the tolerance was decreased.
        """
        if not successes_per_env:
            return False
        return self.update(sum(successes_per_env), num_episodes=len(successes_per_env))

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    @property
    def normalized_delta(self) -> float:
        """Delta expressed in ``[0, 1]`` between ``min_delta`` and ``initial_delta``."""
        lo, hi = self.config.min_delta, self.config.initial_delta
        if hi <= lo:
            return 0.0
        return float((self.delta - lo) / (hi - lo))

    @property
    def at_minimum(self) -> bool:
        return self.delta <= self.config.min_delta + 1e-12

    def state_dict(self) -> Dict[str, float]:
        return {
            "delta": float(self.delta),
            "episodes": float(self.episodes),
            "total_successes": float(self.total_successes),
            "avg_successes": float(self.avg_successes),
            "num_decreases": float(self.num_decreases),
        }

    def load_state_dict(self, state: Dict[str, float]) -> None:
        self.delta = float(state.get("delta", self.delta))
        self.episodes = int(state.get("episodes", self.episodes))
        self.total_successes = float(state.get("total_successes", self.total_successes))
        self.avg_successes = float(state.get("avg_successes", self.avg_successes))
        self.num_decreases = int(state.get("num_decreases", self.num_decreases))

    def reset(self) -> None:
        self.delta = float(self.config.initial_delta)
        self.episodes = 0
        self.total_successes = 0.0
        self.avg_successes = 0.0
        self.num_decreases = 0
        self.history = []

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"SuccessToleranceCurriculum(delta={self.delta:.4f}, "
            f"avg_successes={self.avg_successes:.3f}, "
            f"episodes={self.episodes}, decreases={self.num_decreases})"
        )


__all__ = ["SuccessToleranceCurriculum", "CurriculumConfig"]
