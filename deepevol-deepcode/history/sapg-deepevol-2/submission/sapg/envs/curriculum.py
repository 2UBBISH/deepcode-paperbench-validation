"""Success-tolerance curriculum for SAPG manipulation tasks.

The curriculum follows the paper's description for the AllegroKuka Regrasping task:
the success tolerance ``delta`` starts at 7.5cm and is decayed by 10% whenever the
average number of successes per episode exceeds 3.  The tolerance is clamped to a
minimum of 1cm.

The same schedule is reused (with task-specific initial/minimum tolerances) for the
Throw and Reorientation tasks and for the ShadowHand / AllegroHand environments.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class SuccessToleranceCurriculum:
    """Success-tolerance curriculum.

    Parameters
    ----------
    initial_tolerance:
        Starting tolerance ``delta`` (in meters for position tasks).
    min_tolerance:
        Lower bound for the tolerance.
    decay:
        Multiplicative decay applied when the success threshold is exceeded.
    success_threshold:
        Average successes per episode above which the tolerance is decayed.
    """

    def __init__(
        self,
        initial_tolerance: float = 0.075,
        min_tolerance: float = 0.01,
        decay: float = 0.9,
        success_threshold: float = 3.0,
    ) -> None:
        self.initial_tolerance = float(initial_tolerance)
        self.min_tolerance = float(min_tolerance)
        self.decay = float(decay)
        self.success_threshold = float(success_threshold)

        self.tolerance = float(initial_tolerance)
        self.num_updates = 0

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #
    def update(self, mean_successes: float) -> float:
        """Update the tolerance given the mean successes per episode.

        Returns the (possibly updated) tolerance.
        """
        if mean_successes > self.success_threshold:
            self.tolerance = max(self.min_tolerance, self.tolerance * self.decay)
            self.num_updates += 1
        return self.tolerance

    def get_tolerance(self) -> float:
        return self.tolerance

    def reset(self) -> None:
        self.tolerance = float(self.initial_tolerance)
        self.num_updates = 0

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def state_dict(self) -> dict:
        return {
            "tolerance": self.tolerance,
            "num_updates": self.num_updates,
            "initial_tolerance": self.initial_tolerance,
            "min_tolerance": self.min_tolerance,
            "decay": self.decay,
            "success_threshold": self.success_threshold,
        }

    def load_state_dict(self, state: dict) -> None:
        self.tolerance = state.get("tolerance", self.initial_tolerance)
        self.num_updates = state.get("num_updates", 0)
        self.initial_tolerance = state.get("initial_tolerance", self.initial_tolerance)
        self.min_tolerance = state.get("min_tolerance", self.min_tolerance)
        self.decay = state.get("decay", self.decay)
        self.success_threshold = state.get("success_threshold", self.success_threshold)


def build_curriculum(cfg) -> Optional[SuccessToleranceCurriculum]:
    """Build a curriculum from a config dict/object.

    Returns ``None`` if the curriculum is disabled.
    """
    if cfg is None:
        return None

    def _get(key, default=None):
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    if not _get("use_curriculum", True):
        return None

    return SuccessToleranceCurriculum(
        initial_tolerance=_get("curriculum_initial_tolerance", 0.075),
        min_tolerance=_get("curriculum_min_tolerance", 0.01),
        decay=_get("curriculum_decay", 0.9),
        success_threshold=_get("curriculum_success_threshold", 3.0),
    )
