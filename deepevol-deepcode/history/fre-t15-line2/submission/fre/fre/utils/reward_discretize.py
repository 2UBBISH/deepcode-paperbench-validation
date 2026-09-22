"""Reward discretization for the FRE encoder's reward-token embedding table.

Paper references
----------------
Appendix "Additional Details on the FRE architecture":

    - the scalar reward is discretized into 32 bins by rescaling the reward to
      [0, 1] and then multiplying by 32 and flooring to the nearest integer
    - The discretized reward is mapped to a continuous vector representation
      using a learned embedding table.

Appendix A (Table 3): "Number of Reward Embeddings & 32".

Implementation note (Source: not specified in the paper)
--------------------------------------------------------
Rescaling to ``[0, 1]`` and multiplying by 32 can produce the value ``32`` when
the reward equals ``reward_max`` exactly (and negative values when the reward
falls outside the range used for rescaling).  The embedding table only holds 32
entries, so the floored bin index is clipped to ``[0, 31]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "NUM_REWARD_EMBEDDINGS",
    "NUM_REWARD_BINS",
    "MIN_REWARD_INDEX",
    "MAX_REWARD_INDEX",
    "DEFAULT_REWARD_MIN",
    "DEFAULT_REWARD_MAX",
    "rescale_reward",
    "discretize_reward",
    "discretize_rewards",
    "RewardDiscretizer",
    "make_reward_discretizer",
]

#: Number of learned reward embeddings (Table 3: "Number of Reward Embeddings").
NUM_REWARD_EMBEDDINGS: int = 32
#: Alias kept for readability in places that think in terms of bins.
NUM_REWARD_BINS: int = NUM_REWARD_EMBEDDINGS
#: Lowest valid embedding index.
MIN_REWARD_INDEX: int = 0
#: Highest valid embedding index (``NUM_REWARD_EMBEDDINGS - 1``).
MAX_REWARD_INDEX: int = NUM_REWARD_EMBEDDINGS - 1

#: Default reward range used when no information about the reward function's
#: range is available.  Both the goal-reaching and MLP priors live in ``[-1, 1]``
#: (the goal-reaching reward is ``-1``/``0`` and the MLP prior is clipped to
#: ``[-1, 1]``), and the linear prior is normalized by its own bounds.
DEFAULT_REWARD_MIN: float = -1.0
DEFAULT_REWARD_MAX: float = 1.0

_EPS = 1e-8

ArrayLike = Union[float, int, np.ndarray, Sequence[float]]


def _is_torch(x: Any) -> bool:
    """Return True if ``x`` is a ``torch.Tensor`` (torch imported lazily)."""
    module = type(x).__module__ or ""
    return module.split(".")[0] == "torch"


def _as_array(x: ArrayLike) -> Tuple[np.ndarray, bool]:
    """Coerce input to a float64 numpy array, reporting whether it was a tensor."""
    if _is_torch(x):
        with _torch_no_grad():
            arr = x.detach().cpu().numpy().astype(np.float64)
        return arr, True
    if isinstance(x, np.ndarray):
        return x.astype(np.float64, copy=False), False
    return np.asarray(x, dtype=np.float64), False


def _torch_no_grad():
    import torch  # local import so the module works without torch for numpy use

    return torch.no_grad()


def _to_torch_like(values: np.ndarray, reference: Any, dtype: Any) -> Any:
    """Re-wrap ``values`` into a tensor matching ``reference``'s device/dtype."""
    import torch  # local import

    return torch.as_tensor(values, device=reference.device).to(dtype)


def rescale_reward(
    reward: ArrayLike,
    reward_min: float = DEFAULT_REWARD_MIN,
    reward_max: float = DEFAULT_REWARD_MAX,
) -> ArrayLike:
    """Rescale ``reward`` linearly from ``[reward_min, reward_max]`` to ``[0, 1]``.

    This is the first half of the paper's discretization recipe.  The returned
    value is *not* clipped: rewards outside the range produce values outside
    ``[0, 1]``, which the subsequent flooring/clipping step in
    :func:`discretize_reward` handles.

    Args:
        reward: scalar or array of rewards.
        reward_min: reward value mapped to ``0``.
        reward_max: reward value mapped to ``1``.

    Returns:
        Rescaled reward(s) with the same container type as the input (torch in ->
        torch out, numpy/float in -> numpy out).
    """
    span = float(reward_max) - float(reward_min)
    if abs(span) < _EPS:
        span = _EPS
    arr, was_tensor = _as_array(reward)
    scaled = (arr - float(reward_min)) / span
    if was_tensor:
        return _to_torch_like(scaled, reward, reward.dtype)
    if isinstance(reward, (float, int, np.floating, np.integer)):
        return float(scaled)
    return scaled.astype(np.float32, copy=False) if scaled.ndim else float(scaled)


def discretize_reward(
    reward: ArrayLike,
    reward_min: float = DEFAULT_REWARD_MIN,
    reward_max: float = DEFAULT_REWARD_MAX,
    num_embeddings: int = NUM_REWARD_EMBEDDINGS,
    clip: bool = True,
) -> ArrayLike:
    """Map reward value(s) to integer embedding-table indices.

    Follows the paper exactly: rescale to ``[0, 1]``, multiply by the number of
    reward embeddings (32), then floor.  The floored index is clipped into
    ``[0, num_embeddings - 1]`` because ``floor(1 * 32) == 32`` is out of range
    for a 32-entry table (Source: not specified in the paper).

    Args:
        reward: scalar or array of rewards.
        reward_min: reward value mapped to bin ``0``.
        reward_max: reward value mapped to the top of the bin range.
        num_embeddings: size of the learned embedding table (32 per Table 3).
        clip: whether to clip indices into ``[0, num_embeddings - 1]``.

    Returns:
        Integer indices in ``[0, num_embeddings - 1]``; torch input yields a
        ``torch.long`` tensor, otherwise an ``int`` / ``np.ndarray``.
    """
    scaled = rescale_reward(reward, reward_min=reward_min, reward_max=reward_max)
    # "multiplying by 32 and flooring to the nearest integer"
    if _is_torch(scaled):
        import torch

        as_float = scaled.to(torch.float64) * float(num_embeddings)
        idx = torch.floor(as_float)
        if clip:
            idx = torch.clamp(idx, MIN_REWARD_INDEX, float(num_embeddings - 1))
        return idx.to(torch.long)
    arr = np.asarray(scaled, dtype=np.float64) * float(num_embeddings)
    idx = np.floor(arr)
    if clip:
        idx = np.clip(idx, MIN_REWARD_INDEX, num_embeddings - 1)
    idx = idx.astype(np.int64)
    if idx.ndim == 0:
        return int(idx)
    return idx


def discretize_rewards(
    rewards: ArrayLike,
    reward_min: float = DEFAULT_REWARD_MIN,
    reward_max: float = DEFAULT_REWARD_MAX,
    num_embeddings: int = NUM_REWARD_EMBEDDINGS,
    clip: bool = True,
) -> ArrayLike:
    """Alias of :func:`discretize_reward` for batch-shaped inputs."""
    return discretize_reward(
        rewards,
        reward_min=reward_min,
        reward_max=reward_max,
        num_embeddings=num_embeddings,
        clip=clip,
    )


@dataclass
class RewardDiscretizer:
    """Callable reward -> embedding-index converter bound to a reward range.

    Each sampled prior reward function ``eta`` has a known range (e.g. ``[-1, 0]``
    for goal-reaching, ``[-1, 1]`` for the clipped MLP prior, or the exact
    ``w^T s`` bounds for the linear prior).  Binding that range once means the
    encoder's token construction is a single call::

        indices = discretizer(rewards)          # -> long tensor / int array

    Attributes:
        reward_min: lower end of the reward range mapped to bin ``0``.
        reward_max: upper end of the reward range.
        num_embeddings: embedding-table size (32 per Table 3).
        clip: clip indices to ``[0, num_embeddings - 1]``.
        name: optional descriptive name (used for logging/diagnostics).
    """

    reward_min: float = DEFAULT_REWARD_MIN
    reward_max: float = DEFAULT_REWARD_MAX
    num_embeddings: int = NUM_REWARD_EMBEDDINGS
    clip: bool = True
    name: str = "reward_discretizer"

    def __post_init__(self) -> None:
        if self.num_embeddings <= 0:
            raise ValueError("num_embeddings must be positive")
        if float(self.reward_max) <= float(self.reward_min):
            # A degenerate/decreasing range cannot be rescaled meaningfully; fall
            # back to the documented default range so training stays well defined.
            self.reward_min = DEFAULT_REWARD_MIN
            self.reward_max = DEFAULT_REWARD_MAX

    # ------------------------------------------------------------------ core
    def rescale(self, reward: ArrayLike) -> ArrayLike:
        """Rescale reward(s) to ``[0, 1]`` (see :func:`rescale_reward`)."""
        return rescale_reward(reward, self.reward_min, self.reward_max)

    def discretize(self, reward: ArrayLike) -> ArrayLike:
        """Map reward(s) to embedding indices (see :func:`discretize_reward`)."""
        return discretize_reward(
            reward,
            reward_min=self.reward_min,
            reward_max=self.reward_max,
            num_embeddings=self.num_embeddings,
            clip=self.clip,
        )

    def __call__(self, reward: ArrayLike) -> ArrayLike:
        return self.discretize(reward)

    # --------------------------------------------------------------- helpers
    def set_range(self, reward_min: float, reward_max: float) -> "RewardDiscretizer":
        """Update the reward range in place (e.g. a newly sampled ``eta``)."""
        self.reward_min = float(reward_min)
        self.reward_max = float(reward_max)
        if self.reward_max <= self.reward_min:
            self.reward_min = DEFAULT_REWARD_MIN
            self.reward_max = DEFAULT_REWARD_MAX
        return self

    def fit_from_function(
        self,
        reward_function: Callable[[ArrayLike], ArrayLike],
        states: Optional[Any] = None,
    ) -> "RewardDiscretizer":
        """Bind the range from a prior reward function object.

        Prefers an explicit ``reward_bounds(states)`` / ``reward_min`` /
        ``reward_max`` interface (as implemented by the prior reward classes in
        :mod:`fre.reward_priors`) and otherwise estimates the range from the
        rewards of the provided ``states``.
        """
        rmin = getattr(reward_function, "reward_min", None)
        rmax = getattr(reward_function, "reward_max", None)
        bounds_fn = getattr(reward_function, "reward_bounds", None)
        if callable(bounds_fn):
            try:
                bounds = bounds_fn(states) if states is not None else bounds_fn()
            except TypeError:
                bounds = bounds_fn()
            if bounds is not None and len(bounds) == 2:
                rmin, rmax = float(bounds[0]), float(bounds[1])
        if (rmin is None or rmax is None) and states is not None:
            values = np.asarray(reward_function(states), dtype=np.float64)
            if values.size:
                rmin = float(values.min()) if rmin is None else rmin
                rmax = float(values.max()) if rmax is None else rmax
        if rmin is None:
            rmin = DEFAULT_REWARD_MIN
        if rmax is None:
            rmax = DEFAULT_REWARD_MAX
        return self.set_range(rmin, rmax)

    def describe(self) -> dict:
        """Return a small diagnostic dict (logging friendly)."""
        return {
            "name": self.name,
            "reward_min": float(self.reward_min),
            "reward_max": float(self.reward_max),
            "num_embeddings": int(self.num_embeddings),
            "clip": bool(self.clip),
        }


def make_reward_discretizer(
    reward_min: float = DEFAULT_REWARD_MIN,
    reward_max: float = DEFAULT_REWARD_MAX,
    num_embeddings: int = NUM_REWARD_EMBEDDINGS,
    clip: bool = True,
    name: str = "reward_discretizer",
    reward_function: Optional[Callable[[ArrayLike], ArrayLike]] = None,
    states: Optional[Any] = None,
) -> RewardDiscretizer:
    """Factory building a :class:`RewardDiscretizer`, optionally fitted to ``eta``."""
    disc = RewardDiscretizer(
        reward_min=reward_min,
        reward_max=reward_max,
        num_embeddings=num_embeddings,
        clip=clip,
        name=name,
    )
    if reward_function is not None:
        disc.fit_from_function(reward_function, states=states)
    return disc
