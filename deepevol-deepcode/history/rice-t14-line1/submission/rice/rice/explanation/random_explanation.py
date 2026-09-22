"""Random explanation baseline for RICE (paper Sec. 4.1).

The paper says:

    "Additionally, we introduce "Random" as a baseline explanation method.
     "Random" identifies critical steps by randomly selecting a visited state as
     the critical state."

    -- RICE, Sec. 4.1 (Baseline Explanation Methods)

So the Random explanation has *no* knowledge of the target agent or of the mask
network: every visited state is assigned an i.i.d. uniform importance score, and
the "critical" state of a trajectory is a uniformly random visited state.  This
is exactly the behaviour of ``rice/evaluation/fidelity_score.py::
random_window_index`` (uniformly random sliding-window start), so the fidelity
score computed with this explanation is an unbiased random-window baseline.

Interface contract (shared by every module in ``rice.explanation``)
-------------------------------------------------------------------
All explanation objects are duck-typed and expose

* ``importance(states) -> np.ndarray`` (n,) -- higher = more important; the
  per-state importance score used by the sliding-window fidelity metric and by
  ``rice.algorithms.critical_state`` (``argmax`` selection).
* ``score(states)`` / ``__call__(states)`` -- aliases of ``importance``.
* ``select_index(states, rng=None) -> int`` and ``critical_state(states, ...)``
  -- the trajectory-level critical state.
* ``mask_prob_zero(states)`` -- alias kept for API parity with StateMask / the
  RICE mask network (importance == P(mask = 0)).
* ``reset()``, ``describe()`` / ``as_dict()``, ``state_dict()`` /
  ``load_state_dict()``.

The objects never inspect the target agent's internal architecture, in line with
the black-box assumption of the paper (Appendix / addendum).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "RandomExplanationConfig",
    "RandomExplanation",
    "make_random_explanation",
    "build_random_explanation",
    "make_explanation",
    "NAME",
    "ALIASES",
]

#: canonical name of the method (matches ``rice.evaluation.EXPLANATION_METHODS``)
NAME = "random"

#: names accepted by :func:`make_explanation` / registry lookups
ALIASES: Tuple[str, ...] = (
    "random",
    "rand",
    "random_explanation",
    "random-explanation",
    "baseline_random",
)


@dataclass
class RandomExplanationConfig:
    """Configuration of the Random explanation baseline.

    Parameters
    ----------
    seed:
        Seed of the local RNG (reproducibility of the 3-seed runs).
    mode:
        ``"uniform"`` assigns an i.i.d. uniform score in ``[low, high]`` to each
        visited state (this is what makes the sliding-window selection random).
        ``"one_hot"`` marks a single uniformly chosen state per trajectory with
        score 1 and the others with 0 (equivalent to picking the critical state
        at random, and also used for the fidelity window selection).
    low, high:
        Bounds of the uniform importance scores (``mode == "uniform"``).
    deterministic:
        If ``True`` every state gets the *same* score ``high``.  Not used by the
        paper; only kept so that experiments can disable the randomness while
        keeping the same code path.
    """

    seed: Optional[int] = None
    mode: str = "uniform"
    low: float = 0.0
    high: float = 1.0
    deterministic: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return dict(
            seed=self.seed,
            mode=self.mode,
            low=self.low,
            high=self.high,
            deterministic=self.deterministic,
        )

    def clone(self, **overrides: Any) -> "RandomExplanationConfig":
        data = self.to_dict()
        data.update(overrides)
        return RandomExplanationConfig(**data)

    @classmethod
    def from_mapping(
        cls, mapping: Optional[Dict[str, Any]] = None, **overrides: Any
    ) -> "RandomExplanationConfig":
        data: Dict[str, Any] = {}
        if mapping:
            data.update({k: v for k, v in dict(mapping).items() if k in cls.__dataclass_fields__})
        data.update(overrides)
        return cls(**data)


class RandomExplanation:
    """Uniform-random explanation of a target agent (paper Sec. 4.1 baseline).

    Parameters
    ----------
    config:
        :class:`RandomExplanationConfig` instance (or ``None`` for defaults).
    seed:
        Convenience override of ``config.seed``.
    rng:
        Optional pre-built RNG (any object exposing ``uniform``/``choice``, e.g.
        :class:`rice.utils.seeding.RNG` or ``numpy.random.Generator``).
    """

    name = NAME
    aliases = ALIASES

    def __init__(
        self,
        config: Optional[RandomExplanationConfig] = None,
        seed: Optional[int] = None,
        rng: Any = None,
        **kwargs: Any,
    ) -> None:
        if config is None:
            config = RandomExplanationConfig.from_mapping(kwargs.pop("config", None))
        if kwargs:
            config = config.clone(**{k: v for k, v in kwargs.items() if k in config.__dataclass_fields__})
        self.config = config
        if seed is not None:
            self.config.seed = seed
        if self.config.mode not in ("uniform", "one_hot"):
            raise ValueError(
                f"unknown RandomExplanation mode {self.config.mode!r}; "
                "expected 'uniform' or 'one_hot'"
            )
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng(self.config.seed)
        self._uniform = getattr(self.rng, "uniform", None)
        self._choice = getattr(self.rng, "choice", None)
        self.reset()

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _as_sequence(states: Any) -> Sequence[Any]:
        """Return ``states`` as a sequence (list / array), preserving order."""
        if states is None:
            return []
        if isinstance(states, np.ndarray) and states.ndim == 1:
            # a single state vector -> one-element trajectory
            return [states]
        if isinstance(states, (list, tuple)):
            return states
        try:
            return list(states)
        except TypeError:
            return [states]

    def _draw_uniform(self, n: int) -> np.ndarray:
        if self.config.deterministic:
            return np.full(n, float(self.config.high), dtype=np.float64)
        if callable(self._uniform):
            return np.asarray(self._uniform(self.config.low, self.config.high, n), dtype=np.float64).reshape(-1)[:n]
        # numpy Generator.uniform supports `size`
        return np.asarray(
            self.rng.uniform(self.config.low, self.config.high, size=n), dtype=np.float64
        ).reshape(-1)[:n]

    # ------------------------------------------------------------- interface
    def importance(self, states: Any) -> np.ndarray:
        """Per-state random importance score, shape ``(n,)``."""
        seq = self._as_sequence(states)
        n = len(seq)
        if n == 0:
            return np.zeros(0, dtype=np.float64)
        if self.config.mode == "one_hot":
            scores = np.zeros(n, dtype=np.float64)
            scores[self.select_index(seq)] = float(self.config.high)
            return scores
        return np.clip(self._draw_uniform(n), self.config.low, self.config.high)

    # aliases used by different call sites (fidelity / critical-state / mask)
    def score(self, states: Any) -> np.ndarray:
        return self.importance(states)

    def mask_prob_zero(self, states: Any) -> np.ndarray:
        """API parity with the mask network: importance == P(mask = 0)."""
        return self.importance(states)

    def __call__(self, states: Any) -> np.ndarray:
        return self.importance(states)

    def select_index(self, states: Any, rng: Any = None) -> int:
        """Uniformly random index of a visited state (the critical state)."""
        seq = self._as_sequence(states)
        n = len(seq)
        if n == 0:
            return 0
        gen = rng if rng is not None else self.rng
        if callable(getattr(gen, "integers", None)):
            return int(np.asarray(gen.integers(0, n)).reshape(-1)[0])
        if callable(getattr(gen, "choice", None)):
            try:
                return int(gen.choice(n))
            except TypeError:  # pragma: no cover - exotic RNG
                pass
        return int(np.random.default_rng().integers(0, n))

    def best_index(self, states: Any, rng: Any = None) -> int:
        """Alias of :meth:`select_index` (random baseline == random argmax)."""
        return self.select_index(states, rng=rng)

    def critical_state(self, states: Any, rng: Any = None, return_index: bool = False):
        seq = self._as_sequence(states)
        index = self.select_index(seq, rng=rng)
        if len(seq) == 0:
            state = None
        else:
            state = seq[index]
        return (state, index) if return_index else state

    def critical_states_batch(self, trajectories: Sequence[Sequence[Any]], rng: Any = None):
        """One random critical state per trajectory (utilities/testing)."""
        return [self.critical_state(traj, rng=rng) for traj in trajectories]

    def select_top_k(self, states: Any, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Random top-k indices (``k >= n`` returns all indices)."""
        seq = self._as_sequence(states)
        n = len(seq)
        k = int(min(max(k, 0), n))
        if k == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
        idx = np.asarray(self._sample_without_replacement(n, k), dtype=np.int64)
        return idx, np.ones(k, dtype=np.float64)

    def _sample_without_replacement(self, n: int, k: int) -> np.ndarray:
        if callable(getattr(self.rng, "choice", None)):
            try:
                return np.asarray(self.rng.choice(n, size=k, replace=False), dtype=np.int64)
            except TypeError:
                pass
        return np.random.default_rng(self.config.seed).choice(n, size=k, replace=False)

    # ------------------------------------------------------------ bookkeeping
    def reset(self, seed: Optional[int] = None) -> None:
        """Re-seed the local RNG (called at the beginning of an episode)."""
        if seed is not None:
            self.config.seed = seed
            self.rng = np.random.default_rng(seed)
            self._uniform = getattr(self.rng, "uniform", None)
            self._choice = getattr(self.rng, "choice", None)
        self.calls = 0
        self.history: list = []

    def update(self, *args: Any, **kwargs: Any) -> None:
        """No-op: the Random explanation is non-parametric."""
        return None

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "RandomExplanation",
            "black_box": True,
            "trainable": False,
            "config": self.config.to_dict(),
        }

    def as_dict(self) -> Dict[str, Any]:
        return self.describe()

    def state_dict(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {"config": self.config.to_dict(), "rng": None}
        try:
            state["rng"] = getattr(self.rng, "bit_generator", None).state
        except Exception:  # pragma: no cover - RNG without exposed state
            state["rng"] = None
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> "RandomExplanation":
        if not state:
            return self
        if "config" in state and state["config"]:
            self.config = RandomExplanationConfig.from_mapping(state["config"])
        rng_state = state.get("rng")
        if rng_state is not None:
            try:
                self.rng.bit_generator.state = rng_state
                self._uniform = getattr(self.rng, "uniform", None)
                self._choice = getattr(self.rng, "choice", None)
            except Exception:  # pragma: no cover
                pass
        return self

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"RandomExplanation(mode={self.config.mode!r}, seed={self.config.seed})"


# --------------------------------------------------------------- factories
def make_random_explanation(
    config: Optional[Any] = None, seed: Optional[int] = None, **kwargs: Any
) -> RandomExplanation:
    """Build the Random explanation baseline.

    Accepts a :class:`RandomExplanationConfig`, a plain mapping (parsed YAML) or
    keyword arguments; unknown keys are ignored.
    """
    if isinstance(config, dict):
        config = RandomExplanationConfig.from_mapping(config, **kwargs)
        kwargs = {}
    elif config is None:
        config = RandomExplanationConfig.from_mapping(kwargs.pop("config", None) if "config" in kwargs else None)
    return RandomExplanation(config=config, seed=seed, **kwargs)


def build_random_explanation(
    config: Optional[Any] = None, seed: Optional[int] = None, **kwargs: Any
) -> RandomExplanation:
    """Alias of :func:`make_random_explanation` (used by the evaluation layer)."""
    return make_random_explanation(config=config, seed=seed, **kwargs)


# ``make_explanation`` is provided by ``rice.explanation`` but a local alias is
# handy when this module is imported directly.
make_explanation = make_random_explanation
