"""Aggregation schemes for SAPG (Section 4.2 / 4.3 of the paper).

This package exposes the two aggregation topologies studied in the paper:

* :mod:`sapg.aggregation.leader_follower` -- the *leader-follower* scheme of
  Section 4.3 (also Algorithm 1): a single leader policy ``i = 1`` is updated
  with importance-sampled data collected by every other policy
  ``X = {2, ..., M}`` while the followers remain purely on-policy.
* :mod:`sapg.aggregation.symmetric` -- the *symmetric* aggregation baseline of
  Section 4.2 where every policy ``i`` aggregates data from all other policies
  ``X = {j : j != i}``.

Both modules share the same public surface (``make_aggregation`` style
factories, ``data_sources(i)`` topology queries and Eq. (3)/(4) loss builders)
so that a caller can swap them transparently, e.g.::

    from sapg.aggregation import make_aggregation
    agg = make_aggregation("leader_follower", config=config)

The factory :func:`make_aggregation` dispatches on the scheme *name* and falls
back gracefully when one of the optional backends is missing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

__all__ = [
    "LEADER_FOLLOWER",
    "SYMMETRIC",
    "NO_AGGREGATION",
    "LeaderFollowerAggregator",
    "SymmetricAggregator",
    "AggregationStats",
    "OffPolicySubsampler",
    "make_aggregation",
    "make_leader_follower_aggregator",
    "make_symmetric_aggregator",
    "data_sources",
]

# ---------------------------------------------------------------------------
# Scheme tags (shared string constants)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - trivial import bookkeeping
    from .leader_follower import (  # noqa: F401
        LEADER_FOLLOWER,
        NO_AGGREGATION,
        SYMMETRIC,
        AggregationStats,
        LeaderFollowerAggregator,
        OffPolicySubsampler,
        make_aggregation as _lf_make_aggregation,
    )

    _HAS_LEADER_FOLLOWER = True
except Exception:  # pragma: no cover - defensive fallback
    LEADER_FOLLOWER = "leader_follower"
    SYMMETRIC = "symmetric"
    NO_AGGREGATION = "none"
    AggregationStats = None  # type: ignore[assignment]
    LeaderFollowerAggregator = None  # type: ignore[assignment]
    OffPolicySubsampler = None  # type: ignore[assignment]
    _lf_make_aggregation = None
    _HAS_LEADER_FOLLOWER = False


try:  # pragma: no cover - trivial import bookkeeping
    from .symmetric import SymmetricAggregator  # noqa: F401

    _HAS_SYMMETRIC = True
except Exception:  # pragma: no cover - defensive fallback
    SymmetricAggregator = None  # type: ignore[assignment]
    _HAS_SYMMETRIC = False


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
def make_leader_follower_aggregator(config: Optional[Any] = None, **kwargs: Any):
    """Build a :class:`LeaderFollowerAggregator` (Section 4.3)."""
    if not _HAS_LEADER_FOLLOWER:  # pragma: no cover - defensive
        raise ImportError(
            "sapg.aggregation.leader_follower is unavailable; cannot build the "
            "leader-follower aggregator."
        )
    from .leader_follower import LeaderFollowerAggregator as _LFA

    return _LFA(config=config, **kwargs)


def make_symmetric_aggregator(config: Optional[Any] = None, **kwargs: Any):
    """Build a :class:`SymmetricAggregator` (Section 4.2)."""
    if not _HAS_SYMMETRIC:  # pragma: no cover - defensive
        raise ImportError(
            "sapg.aggregation.symmetric is unavailable; cannot build the "
            "symmetric aggregator."
        )
    from .symmetric import SymmetricAggregator as _SA

    return _SA(config=config, **kwargs)


def make_aggregation(name: str = LEADER_FOLLOWER, config: Optional[Any] = None, **kwargs: Any):
    """Factory dispatching on the aggregation scheme name.

    Parameters
    ----------
    name:
        One of ``"leader_follower"`` (Section 4.3, default), ``"symmetric"``
        (Section 4.2 ablation), ``"none"`` (no off-policy data at all).
    config:
        Optional :class:`~sapg.utils.config.SAPGConfig`.
    **kwargs:
        Forwarded to the aggregator constructor.
    """
    key = str(name).lower().replace("-", "_")
    if key in ("symmetric", "sym"):
        return make_symmetric_aggregator(config=config, **kwargs)
    if _HAS_LEADER_FOLLOWER:  # delegate to the rich leader-follower factory
        return _lf_make_aggregation(name=name, config=config, **kwargs)  # type: ignore[misc]
    # Last-resort minimal controller so callers can still query the topology.
    return _FallbackAggregator(config=config, name=key, **kwargs)


class _FallbackAggregator:
    """Minimal topology-only aggregation controller (Section 4.3 default)."""

    name = LEADER_FOLLOWER

    def __init__(
        self,
        config: Optional[Any] = None,
        name: str = LEADER_FOLLOWER,
        num_policies: Optional[int] = None,
        leader_index: Optional[int] = None,
        **_: Any,
    ) -> None:
        from ..utils.config import NUM_POLICIES

        self.name = name
        self.num_policies = int(
            num_policies
            if num_policies is not None
            else getattr(config, "num_policies", NUM_POLICIES)
        )
        self.leader_index = int(
            leader_index
            if leader_index is not None
            else getattr(config, "leader_index", 1)
        )
        if name == NO_AGGREGATION:
            self.leader_index = 0

    # -- topology ---------------------------------------------------------
    def is_leader(self, i: int) -> bool:
        return self.name == LEADER_FOLLOWER and int(i) == self.leader_index

    def data_sources(self, i: int) -> List[int]:
        i = int(i)
        if self.name == NO_AGGREGATION:
            return []
        if self.name == LEADER_FOLLOWER:
            if i == self.leader_index:
                return [j for j in range(1, self.num_policies + 1) if j != i]
            return []
        # symmetric
        return [j for j in range(1, self.num_policies + 1) if j != i]

    def __len__(self) -> int:  # pragma: no cover - convenience
        return self.num_policies

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"{type(self).__name__}(name={self.name!r}, "
            f"M={self.num_policies}, leader={self.leader_index})"
        )


def data_sources(
    scheme: Any, i: int, num_policies: Optional[int] = None, leader_index: int = 1
) -> List[int]:
    """Return the set ``X`` of source policies used to update policy ``i``.

    Works with any object exposing ``data_sources`` (the aggregator classes) or
    with a plain scheme name string (``"leader_follower"`` / ``"symmetric"`` /
    ``"none"``).  Policy indices are 1-based, matching the paper: the leader is
    policy ``1`` and followers are ``2..M``.
    """
    if hasattr(scheme, "data_sources"):
        return list(scheme.data_sources(i))
    fallback = _FallbackAggregator(
        name=str(scheme).lower().replace("-", "_"),
        num_policies=num_policies,
        leader_index=leader_index,
    )
    return fallback.data_sources(i)
