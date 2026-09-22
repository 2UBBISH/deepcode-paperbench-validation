"""Coefficient schedules for the retention losses.

The NetHack kickstarting loss uses ``coefficient = 0.5`` scaled by an
exponential decay of ``0.99998`` applied every training step (Appendix B.1).
"""

from __future__ import annotations

import abc


class CoefficientSchedule(abc.ABC):
    def __init__(self, initial: float) -> None:
        self.initial = float(initial)

    @abc.abstractmethod
    def value(self, step: int) -> float:
        ...

    def __call__(self, step: int) -> float:
        return self.value(step)


class ConstantSchedule(CoefficientSchedule):
    """``value(step) == initial`` for every step (used by BC in NetHack)."""

    def value(self, step: int) -> float:  # noqa: D102
        return self.initial


class ExponentialDecaySchedule(CoefficientSchedule):
    """``value(step) = initial * decay ** step``.

    NetHack kickstarting uses ``decay=0.99998`` decayed once per training step.
    """

    def __init__(self, initial: float, decay: float) -> None:
        super().__init__(initial)
        self.decay = float(decay)

    def value(self, step: int) -> float:  # noqa: D102
        if self.decay >= 1.0:
            return self.initial
        return self.initial * (self.decay ** step)


def make_schedule(initial: float, decay: float = 1.0) -> CoefficientSchedule:
    if decay >= 1.0:
        return ConstantSchedule(initial)
    return ExponentialDecaySchedule(initial, decay)
