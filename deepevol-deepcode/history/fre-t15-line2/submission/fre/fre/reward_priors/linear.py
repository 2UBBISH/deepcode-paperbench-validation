"""Random linear reward functions (the "low frequency" component of the FRE prior).

Paper reference (FRE, Appendix B -- Training Details):

    "Random Linear functions are generated according to a uniform vector within -1 and 1.
    On AntMaze, we remove the XY positions from this generation as the scale of the
    dimensions led to instability. A random binary mask is applied with a 0.9 chance to
    zero the vector at that dimension, to encourage sparsity and bias towards simpler
    functions."

and (Section 4.2, mixture of random unsupervised functions):

    "The particular mixture we use consists of random singleton functions (corresponding to
    "goal reaching" rewards), random neural networks (MLPs with two linear layers), and
    random linear functions (corresponding to "MLPs" with one linear layer). ... A uniform
    mixture of the three function classes are used during training."

A linear reward function is therefore

    eta(s) = w^T s ,        w_i ~ U[-1, 1]  (with per-dimension masking)

with a per-dimension binary mask that zeros dimension i with probability 0.9 (and keeps it
with probability 0.1).  On AntMaze the XY position dimensions are excluded from the
generation, i.e. their weight is forced to zero (they are *removed from the generation*, so
they never receive a non-zero weight and also do not count towards the sparsity mask's notion
of "the vector").

Everything in this module subclasses the shared `RewardFunction` / `RewardPrior`
abstractions defined in :mod:`fre.reward_priors.goal_reaching`, so that the mixture prior
(:mod:`fre.reward_priors.mixture`) can treat every function class uniformly.

Details the paper does not specify (documented defaults):
  * `reward_min` / `reward_max`: needed by the 32-bin reward discretizer (which rescales the
    reward to [0, 1] before binning).  We compute the exact range of  w^T s  over the
    per-dimension box [state_min, state_max] of the offline dataset.  If no box is supplied
    the range is inferred lazily from the states that are labelled (see `reward_bounds`).
  * If the 0.9 sparsity mask zeroes *every* dimension we force one random (non-excluded)
    dimension to stay active -- an all-zero reward function is uninformative and would make
    the encoder/decoder target constant.
  * No clipping of the linear output is applied (the paper only clips the *MLP* output to
    [-1, 1]); an optional `clip` argument is provided for convenience.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - the shared ABCs live in goal_reaching.py
    from fre.reward_priors.goal_reaching import (
        RewardFunction,
        RewardPrior,
        as_numpy_2d,
    )
except ImportError:  # pragma: no cover - direct module execution fallback
    from goal_reaching import RewardFunction, RewardPrior, as_numpy_2d  # type: ignore

__all__ = [
    "ANTMAZE_POSITION_DIMS",
    "LINEAR_WEIGHT_RANGE",
    "LINEAR_ZERO_PROB",
    "LinearRewardFunction",
    "LinearRewardPrior",
    "make_linear_prior",
]

# ---------------------------------------------------------------------------
# Constants from the paper (Appendix B / Appendix A)
# ---------------------------------------------------------------------------

#: Uniform range of each linear weight: "a uniform vector within -1 and 1".
LINEAR_WEIGHT_RANGE = 1.0

#: Probability that the random binary mask zeroes a given dimension ("a 0.9 chance").
LINEAR_ZERO_PROB = 0.9

#: AntMaze observation layout (D4RL antmaze-large-diverse-v2 state is 29-d: qpos(15)+qvel(14)).
#: The XY positions are qpos[0], qpos[1]; these are the dimensions removed for AntMaze.
ANTMAZE_POSITION_DIMS: Tuple[int, ...] = (0, 1)


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------


@dataclass
class LinearRewardFunction(RewardFunction):
    """A single random linear reward function ``eta(s) = w^T s``.

    Parameters
    ----------
    weight:
        Float vector of shape ``(state_dim,)``.  Already masked (excluded / zeroed
        dimensions have weight exactly 0).
    exclude_dims:
        Dimensions that were excluded from the generation (kept for bookkeeping / the
        description string).
    state_min, state_max:
        Optional per-dimension box used to compute the exact reward range; also used for a
        linear rescaling to [-1, 1].  Both may be ``None``.
    clip:
        Optional symmetric clipping range for the output (default: no clipping, as the paper
        clips only the MLP prior's output).
    """

    weight: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float32))
    exclude_dims: Tuple[int, ...] = ()
    state_min: Optional[np.ndarray] = None
    state_max: Optional[np.ndarray] = None
    clip: Optional[float] = None
    name: str = "linear"

    # -- construction -------------------------------------------------------
    def __post_init__(self) -> None:
        self.weight = np.asarray(self.weight, dtype=np.float32).reshape(-1)
        if self.state_min is not None:
            self.state_min = np.asarray(self.state_min, dtype=np.float32).reshape(-1)
        if self.state_max is not None:
            self.state_max = np.asarray(self.state_max, dtype=np.float32).reshape(-1)
        self.exclude_dims = tuple(int(d) for d in self.exclude_dims)
        rmin, rmax = self._range_from_box()
        if rmin is None or rmax is None:
            # Without a state box the range is unknown a priori: fall back to the range of
            # the weight vector itself (data-independent, conservative default).
            abs_w = float(np.abs(self.weight).sum())
            rmin, rmax = -abs_w, abs_w
        self.reward_min = float(rmin)
        self.reward_max = float(rmax)
        # Attributes required by the shared (mutable) RewardFunction base class.
        if not hasattr(self, "reward_unreached"):
            self.reward_unreached = self.reward_min
        if not hasattr(self, "reward_reached"):
            self.reward_reached = self.reward_max

    def _range_from_box(self) -> Tuple[Optional[float], Optional[float]]:
        """Exact min/max of ``w^T s`` over the box [state_min, state_max]."""
        if self.state_min is None or self.state_max is None:
            return None, None
        lo = self.state_min[: self.weight.shape[0]]
        hi = self.state_max[: self.weight.shape[0]]
        w = self.weight[: lo.shape[0]]
        per_dim_max = np.maximum(w * lo, w * hi)
        per_dim_min = np.minimum(w * lo, w * hi)
        return float(per_dim_min.sum()), float(per_dim_max.sum())

    # -- RewardFunction API -------------------------------------------------
    def _rewards_and_dones(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        states = as_numpy_2d(states)
        w = self.weight[: states.shape[-1]]
        rewards = states[:, : w.shape[0]] @ w
        if self.clip is not None:
            rewards = np.clip(rewards, -float(self.clip), float(self.clip))
        rewards = rewards.astype(np.float32)
        dones = np.zeros((rewards.shape[0],), dtype=bool)
        return rewards, dones

    def __call__(self, states: np.ndarray) -> np.ndarray:
        return self._rewards_and_dones(states)[0]

    def reward_bounds(self, states: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """Bounds used by the reward discretizer.

        With a state box available the bounds are exact (computed at construction).  Otherwise
        they are estimated from the provided states.
        """
        if self.state_min is not None and self.state_max is not None:
            return self.reward_min, self.reward_max
        if states is None:
            return self.reward_min, self.reward_max
        r = self._rewards_and_dones(states)[0]
        return float(np.min(r)), float(np.max(r))

    def describe(self) -> Dict[str, Any]:
        active = np.flatnonzero(np.abs(self.weight) > 0.0)
        return {
            "name": self.name,
            "num_active_dims": int(active.size),
            "active_dims": active.tolist(),
            "exclude_dims": list(self.exclude_dims),
            "reward_min": self.reward_min,
            "reward_max": self.reward_max,
            "weight": self.weight.copy(),
        }


# ---------------------------------------------------------------------------
# Prior distribution over linear reward functions
# ---------------------------------------------------------------------------


class LinearRewardPrior(RewardPrior):
    """Distribution over random linear reward functions (Appendix B).

    Sampling procedure:
      1. draw ``w ~ U[-1, 1]^d``,
      2. force the excluded dimensions (AntMaze XY positions) to zero,
      3. apply a random binary mask that zeroes each remaining dimension with probability
         ``zero_prob = 0.9``,
      4. (documented default) if every dimension got zeroed, keep a single random dimension.
    """

    def __init__(
        self,
        state_dim: int,
        exclude_dims: Sequence[int] = (),
        state_min: Optional[np.ndarray] = None,
        state_max: Optional[np.ndarray] = None,
        weight_range: float = LINEAR_WEIGHT_RANGE,
        zero_prob: float = LINEAR_ZERO_PROB,
        ensure_nonzero: bool = True,
        clip: Optional[float] = None,
        seed: int = 0,
        name: str = "linear",
    ) -> None:
        self.state_dim = int(state_dim)
        self.exclude_dims = tuple(int(d) for d in np.atleast_1d(exclude_dims).tolist())
        self.state_min = None if state_min is None else np.asarray(state_min, dtype=np.float32).reshape(-1)
        self.state_max = None if state_max is None else np.asarray(state_max, dtype=np.float32).reshape(-1)
        self.weight_range = float(weight_range)
        self.zero_prob = float(zero_prob)
        self.ensure_nonzero = bool(ensure_nonzero)
        self.clip = clip
        self.seed = int(seed)
        self.name = name
        self._rng = np.random.default_rng(self.seed)
        self._active_dims = np.array(
            [d for d in range(self.state_dim) if d not in set(self.exclude_dims)], dtype=np.int64
        )
        if self._active_dims.size == 0:  # everything excluded -> nothing to sample
            self._active_dims = np.arange(self.state_dim, dtype=np.int64)

    # -- RewardPrior API ----------------------------------------------------
    def sample(self, rng: Optional[np.random.Generator] = None) -> LinearRewardFunction:
        rng = rng if rng is not None else self._rng
        w = np.zeros((self.state_dim,), dtype=np.float32)
        # 1) uniform vector within -1 and 1 (only over the non-excluded dimensions)
        w[self._active_dims] = rng.uniform(
            -self.weight_range, self.weight_range, size=self._active_dims.size
        ).astype(np.float32)
        # 2) random binary mask: zero a dimension with probability `zero_prob`
        keep = rng.random(self._active_dims.size) > self.zero_prob
        if not keep.any() and self.ensure_nonzero:
            # Documented default: never emit an all-zero (constant) reward function.
            keep[rng.integers(self._active_dims.size)] = True
        masked = np.zeros_like(w)
        masked[self._active_dims[keep]] = w[self._active_dims[keep]]
        return LinearRewardFunction(
            weight=masked,
            exclude_dims=self.exclude_dims,
            state_min=self.state_min,
            state_max=self.state_max,
            clip=self.clip,
            name=self.name,
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state_dim": self.state_dim,
            "exclude_dims": list(self.exclude_dims),
            "weight_range": self.weight_range,
            "zero_prob": self.zero_prob,
            "expected_active_dims": float(
                self._active_dims.size * (1.0 - self.zero_prob) if self.ensure_nonzero else 0.0
            ),
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_linear_prior(
    replay_buffer: Any = None,
    state_dim: Optional[int] = None,
    exclude_dims: Sequence[int] = (),
    seed: int = 0,
    name: str = "linear",
    **kwargs: Any,
) -> LinearRewardPrior:
    """Build a :class:`LinearRewardPrior`, pulling defaults from a replay buffer if given.

    The per-dimension state box (used for the exact reward range / normalization) is taken
    from the replay buffer's state statistics when available.
    """
    state_min = kwargs.pop("state_min", None)
    state_max = kwargs.pop("state_max", None)
    if replay_buffer is not None:
        if state_dim is None:
            state_dim = getattr(replay_buffer, "obs_dim", None)
        if state_min is None or state_max is None:
            try:
                stats = getattr(replay_buffer, "state_box", None)
                if callable(stats):
                    box = stats()
                    state_min, state_max = box
                else:
                    # fall back to (mean -+ 3 std) as a data-driven box
                    mean, std = replay_buffer.state_statistics()  # type: ignore[attr-defined]
                    state_min = np.asarray(mean, dtype=np.float32) - 3.0 * np.asarray(std, dtype=np.float32)
                    state_max = np.asarray(mean, dtype=np.float32) + 3.0 * np.asarray(std, dtype=np.float32)
            except Exception:  # pragma: no cover - defensive
                pass
    if state_dim is None:
        raise ValueError("make_linear_prior requires `state_dim` or a `replay_buffer` with `obs_dim`.")
    return LinearRewardPrior(
        state_dim=int(state_dim),
        exclude_dims=exclude_dims,
        state_min=state_min,
        state_max=state_max,
        seed=seed,
        name=name,
        **kwargs,
    )
