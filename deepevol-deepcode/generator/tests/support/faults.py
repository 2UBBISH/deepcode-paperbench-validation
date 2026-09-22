"""Small, explicit fault controls for component and boundary tests."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass


FaultFactory = Callable[[], Exception]
FaultStep = Exception | FaultFactory | None


@dataclass(frozen=True, slots=True)
class _FaultScript:
    steps: tuple[FaultFactory | None, ...]
    repeat_last: bool


class FaultInjector:
    """Keep deterministic failures at a named test boundary.

    Faults remain installed until ``clear`` is called.  Keeping the lifetime
    explicit prevents a test from silently losing its failure condition after
    the first request, while the factory form gives each request a fresh
    exception instance when needed.
    """

    def __init__(self) -> None:
        self._factories: dict[str, FaultFactory] = {}
        self._scripts: dict[str, _FaultScript] = {}
        self._script_positions: dict[str, int] = {}
        self._calls: dict[str, int] = {}

    @staticmethod
    def _validate_operation(operation: str) -> None:
        if not operation or operation.strip() != operation:
            raise ValueError("fault operation must be a non-empty trimmed name")

    @staticmethod
    def _factory(step: FaultStep) -> FaultFactory | None:
        if step is None:
            return None
        if isinstance(step, Exception):
            return lambda step=step: step
        if callable(step):
            return step
        raise TypeError("fault step must be None, an exception, or an exception factory")

    def inject(
        self,
        operation: str,
        error: Exception | FaultFactory,
    ) -> None:
        """Install a persistent fault until ``clear`` is called."""

        self._validate_operation(operation)
        factory = self._factory(error)
        assert factory is not None
        self._scripts.pop(operation, None)
        self._script_positions.pop(operation, None)
        self._factories[operation] = factory

    def inject_sequence(
        self,
        operation: str,
        steps: Iterable[FaultStep],
        *,
        repeat_last: bool = False,
    ) -> None:
        """Install an ordered, deterministic fault script.

        ``None`` means that the boundary call is allowed to proceed.  A
        script is consumed once per ``raise_if_injected`` call; by default it
        becomes inactive after its final step.  ``repeat_last`` is useful for
        a sustained outage without hiding the exact transition in a test.
        """

        self._validate_operation(operation)
        factories = tuple(self._factory(step) for step in steps)
        if not factories:
            raise ValueError("fault sequence must contain at least one step")
        self._factories.pop(operation, None)
        self._scripts[operation] = _FaultScript(factories, repeat_last)
        self._script_positions[operation] = 0

    def raise_if_injected(self, operation: str) -> None:
        self._validate_operation(operation)
        self._calls[operation] = self._calls.get(operation, 0) + 1
        script = self._scripts.get(operation)
        if script is not None:
            position = self._script_positions[operation]
            if position < len(script.steps):
                factory = script.steps[position]
                next_position = position + 1
                if next_position >= len(script.steps) and script.repeat_last:
                    next_position = len(script.steps) - 1
                self._script_positions[operation] = next_position
            else:  # pragma: no cover - defensive state guard
                factory = None
            if factory is not None:
                error = factory()
                if not isinstance(error, Exception):
                    raise TypeError("fault factory must return an exception")
                raise error
            return
        factory = self._factories.get(operation)
        if factory is not None:
            error = factory()
            if not isinstance(error, Exception):
                raise TypeError("fault factory must return an exception")
            raise error

    def calls(self, operation: str) -> int:
        """Return the number of boundary invocations observed for an operation."""

        self._validate_operation(operation)
        return self._calls.get(operation, 0)

    def clear(self, operation: str | None = None) -> None:
        if operation is None:
            self._factories.clear()
            self._scripts.clear()
            self._script_positions.clear()
        else:
            self._validate_operation(operation)
            self._factories.pop(operation, None)
            self._scripts.pop(operation, None)
            self._script_positions.pop(operation, None)


__all__ = ["FaultFactory", "FaultInjector", "FaultStep"]
