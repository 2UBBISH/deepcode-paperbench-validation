"""Goal-reaching reward functions and the HER-style goal-reaching prior.

Implements the goal-reaching component of the unsupervised reward prior
``p(eta)`` used to pretrain the FRE encoder/decoder (paper Sec. 4.2 and App. B).

Reward family
-------------
A *singleton* goal-reaching reward function is parameterised by a goal state
``g`` sampled from the offline dataset:

.. math::
    \\eta_g(s) = \\begin{cases}
        r_{succ} & \\|s - g\\| \\le \\tau \\\\
        r_{fail} & \\text{otherwise}
    \\end{cases}

with the FRE defaults ``r_succ = 0``, ``r_fail = -1`` (matching the paper's
"reward ``-1`` until the goal is reached, ``0`` thereafter") and ``tau`` the
goal threshold.  The induced *done mask* is ``1`` whenever the goal is
achieved, i.e. ``done(s) = [\\|s - g\\| <= tau]``.

Goal sampling (hindsight / HER)
-------------------------------
Goals are drawn from the dataset with the mixture used by the paper's prior::

    p(current state)              = 0.2
    p(future state in same traj)  = 0.5   (geometric offset by default)
    p(random state in dataset)    = 0.3

Additionally, :func:`sample_reward_context` guarantees that **at least one** of
the K = 32 encoder context samples achieves the goal (reward ``r_succ``), which
is what allows the VAE posterior to identify the goal from the reward context.

All distances/rewards are computed in numpy so the prior can be used without a
GPU, while the base class handles torch/numpy interop transparently.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # pragma: no cover - import shim for both package and script usage
    from .base import (
        RewardFunction,
        RewardFunctionPrior,
        episode_boundaries,
        get_observations,
        get_rng,
        get_terminals,
        to_numpy,
    )
except ImportError:  # pragma: no cover
    from fre.rewards.base import (  # type: ignore
        RewardFunction,
        RewardFunctionPrior,
        episode_boundaries,
        get_observations,
        get_rng,
        get_terminals,
        to_numpy,
    )


__all__ = [
    # constants
    "GOAL_REACHING_FAMILY",
    "DEFAULT_P_CURRENT",
    "DEFAULT_P_FUTURE",
    "DEFAULT_P_RANDOM",
    "DEFAULT_GOAL_THRESHOLD",
    "DEFAULT_GOAL_REWARD",
    "DEFAULT_FAILURE_REWARD",
    "CONTEXT_SIZE",
    "DECODER_SIZE",
    # rewards
    "GoalReachingReward",
    "GoalReachingPrior",
    "GoalReachingRewardPrior",
    "make_goal_reaching_reward",
    "make_goal_reaching_prior",
    # helpers
    "goal_distance",
    "goal_success",
    "episode_index_of",
    "future_indices",
    "her_goal_indices",
    "sample_reward_context",
    "sample_context_and_decoder_pairs",
]


# ---------------------------------------------------------------------------
# Constants (paper defaults)
# ---------------------------------------------------------------------------

GOAL_REACHING_FAMILY = "goal_reaching"

#: Hindsight goal sampling mixture (Sec. 4.2): p(current) / p(future) / p(random).
DEFAULT_P_CURRENT = 0.2
DEFAULT_P_FUTURE = 0.5
DEFAULT_P_RANDOM = 0.3

#: Success radius (AntMaze uses ``dist <= 2``; ExORL uses ``dist < 0.1``).
DEFAULT_GOAL_THRESHOLD = 2.0

#: Reward values: ``-1`` until the goal is reached, ``0`` thereafter.
DEFAULT_GOAL_REWARD = 0.0
DEFAULT_FAILURE_REWARD = -1.0

#: Number of encoder context samples (K) and decoder samples (K') per function.
CONTEXT_SIZE = 32
DECODER_SIZE = 8

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Distance helpers
# ---------------------------------------------------------------------------


def _as_features(states: Any, dims: Optional[Any] = None) -> np.ndarray:
    """Return ``states`` as a float64 array, optionally restricted to ``dims``."""
    arr = to_numpy(states, dtype=np.float64)
    if dims is None:
        return arr
    if isinstance(dims, slice):
        return arr[..., dims]
    if isinstance(dims, (int, np.integer)):
        return arr[..., int(dims) : int(dims) + 1]
    idx = list(dims)
    return arr[..., idx]


def goal_distance(
    states: Any,
    goal: Any,
    metric: str = "euclidean",
    dims: Optional[Any] = None,
    weights: Optional[Any] = None,
) -> np.ndarray:
    """Distance between ``states`` and ``goal``.

    Args:
        states: ``(..., D)`` array-like of states.
        goal: ``(D,)`` (or broadcastable) goal state.
        metric: one of ``"euclidean"``, ``"squared"``, ``"manhattan"``,
            ``"chebyshev"``, ``"cosine"``.
        dims: optional slice/index list restricting which state dimensions take
            part in the distance (e.g. the two XY dims of AntMaze).
        weights: optional per-dimension scaling applied before the norm.

    Returns:
        ``(...,)`` array of distances.
    """
    s = _as_features(states, dims)
    g = np.asarray(goal, dtype=np.float64)
    if dims is not None:
        g = _as_features(g, dims)
    diff = s - g
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        diff = diff * w

    metric = (metric or "euclidean").lower()
    if metric in ("euclidean", "l2", "norm"):
        return np.sqrt(np.sum(diff * diff, axis=-1) + _EPS)
    if metric in ("squared", "sqeuclidean", "l2_squared"):
        return np.sum(diff * diff, axis=-1)
    if metric in ("manhattan", "l1", "cityblock"):
        return np.sum(np.abs(diff), axis=-1)
    if metric in ("chebyshev", "linf", "max"):
        return np.max(np.abs(diff), axis=-1)
    if metric == "cosine":
        num = np.sum(s * g, axis=-1)
        den = (np.linalg.norm(s, axis=-1) * np.linalg.norm(g) + _EPS)
        return 1.0 - num / den
    raise ValueError(f"Unknown distance metric: {metric!r}")


def goal_success(distance: Any, threshold: float = DEFAULT_GOAL_THRESHOLD) -> np.ndarray:
    """Boolean ``distance <= threshold`` success mask."""
    return np.asarray(distance, dtype=np.float64) <= float(threshold)


# ---------------------------------------------------------------------------
# Goal-reaching reward function
# ---------------------------------------------------------------------------


class GoalReachingReward(RewardFunction):
    """Singleton sparse (or dense) goal-reaching reward ``eta_g(s)``.

    Args:
        goal: ``(D,)`` goal state.
        state_dim: dimensionality of the state space.
        threshold: success radius ``tau``.
        reward_success: reward returned at (or inside) the goal (default ``0``).
        reward_failure: reward returned away from the goal (default ``-1``).
        reward_type: ``"sparse"`` (paper default), ``"dense"`` (linear in
            distance) or ``"exp"`` (exponential shaping).
        metric: distance metric, see :func:`goal_distance`.
        dims: optional subset of state dimensions used for the distance.
        weights: optional per-dimension weights.
        done_on_success: if ``True`` the done mask fires at the goal.
        dense_scale: normalising constant for dense/exp shaping (defaults to the
            success threshold, so the shaped reward is ``-1`` one threshold away).
        name: optional human readable name.
        metadata: optional metadata dict (goal index, sampling strategy, ...).
    """

    family = GOAL_REACHING_FAMILY

    def __init__(
        self,
        goal: Any,
        state_dim: Optional[int] = None,
        threshold: float = DEFAULT_GOAL_THRESHOLD,
        reward_success: float = DEFAULT_GOAL_REWARD,
        reward_failure: float = DEFAULT_FAILURE_REWARD,
        reward_type: str = "sparse",
        metric: str = "euclidean",
        dims: Optional[Any] = None,
        weights: Optional[Any] = None,
        done_on_success: bool = True,
        dense_scale: Optional[float] = None,
        goal_index: Optional[int] = None,
        strategy: Optional[str] = None,
        name: Optional[str] = None,
        clip: float = 1.0,
        mean: Optional[Any] = None,
        std: Optional[Any] = None,
        normalise: bool = False,
        device: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        goal_arr = np.asarray(goal, dtype=np.float64).reshape(-1)
        if state_dim is None:
            state_dim = int(goal_arr.shape[0])

        meta: Dict[str, Any] = dict(metadata or {})
        meta.setdefault("goal", goal_arr.tolist())
        if goal_index is not None:
            meta.setdefault("goal_index", int(goal_index))
        if strategy is not None:
            meta.setdefault("strategy", strategy)

        super().__init__(
            state_dim=int(state_dim),
            name=name or "goal_reaching",
            clip=clip,
            mean=mean,
            std=std,
            normalise=normalise,
            device=device,
            metadata=meta,
        )

        self.goal = goal_arr.astype(np.float64)
        self.threshold = float(threshold)
        self.reward_success = float(reward_success)
        self.reward_failure = float(reward_failure)
        self.reward_type = str(reward_type).lower()
        self.metric = metric
        self.dims = dims
        self.weights = None if weights is None else np.asarray(weights, dtype=np.float64)
        self.done_on_success = bool(done_on_success)
        self.dense_scale = float(dense_scale) if dense_scale else max(self.threshold, _EPS)
        self.goal_index = None if goal_index is None else int(goal_index)
        self.strategy = strategy

    # -- core interface ----------------------------------------------------

    def distance(self, states: Any) -> np.ndarray:
        """Distance of ``states`` to the goal (shape preserving)."""
        return goal_distance(
            states, self.goal, metric=self.metric, dims=self.dims, weights=self.weights
        )

    def is_success(self, states: Any) -> np.ndarray:
        """Boolean success mask (goal reached within ``threshold``)."""
        return goal_success(self.distance(states), self.threshold)

    def _compute(self, states: Any) -> np.ndarray:
        dist = self.distance(states)
        if self.reward_type == "sparse":
            r = np.where(
                dist <= self.threshold,
                self.reward_success,
                self.reward_failure,
            )
        elif self.reward_type in ("dense", "linear"):
            # 0 at the goal, -1 one ``dense_scale`` away, clipped at -1 further out.
            r = self.reward_success - np.clip(dist / self.dense_scale, 0.0, 1.0) * (
                self.reward_success - self.reward_failure
            )
        elif self.reward_type in ("exp", "exponential"):
            scale = self.dense_scale
            r = self.reward_failure + (self.reward_success - self.reward_failure) * np.exp(
                -dist / max(scale, _EPS)
            )
        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type!r}")

        r = np.asarray(r, dtype=np.float32)
        if self.clip is not None:
            r = np.clip(r, -float(self.clip), float(self.clip))
        return r

    def done_numpy(self, states: Any) -> np.ndarray:  # noqa: D102 - base hook
        if not self.done_on_success:
            return np.zeros(np.asarray(self.distance(states)).shape, dtype=bool)
        return np.asarray(self.is_success(states), dtype=bool)

    # -- introspection -----------------------------------------------------

    def describe(self) -> Dict[str, Any]:  # noqa: D102 - base hook
        info = {
            "family": self.family,
            "name": self.name,
            "goal": self.goal.tolist(),
            "threshold": self.threshold,
            "reward_type": self.reward_type,
            "metric": self.metric,
            "reward_success": self.reward_success,
            "reward_failure": self.reward_failure,
        }
        if self.goal_index is not None:
            info["goal_index"] = self.goal_index
        if self.strategy is not None:
            info["strategy"] = self.strategy
        return info

    def extra_repr(self) -> str:  # noqa: D102 - base hook
        g = np.array2string(self.goal, precision=2, threshold=8)
        return (
            f"goal={g}, threshold={self.threshold}, reward_type={self.reward_type}, "
            f"metric={self.metric}"
        )


# ---------------------------------------------------------------------------
# Hindsight (HER) goal index sampling
# ---------------------------------------------------------------------------


def episode_index_of(
    indices: Any, episode_ends: Any, episode_starts: Optional[Any] = None
) -> np.ndarray:
    """Map flat transition indices to their episode index.

    Args:
        indices: ``(N,)`` flat transition indices.
        episode_ends: inclusive end index of each episode (sorted).
        episode_starts: optional inclusive start index of each episode.

    Returns:
        ``(N,)`` episode index per transition.
    """
    ends = np.asarray(episode_ends, dtype=np.int64).reshape(-1)
    if ends.size == 0:
        raise ValueError("episode_ends must be non-empty")
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    return np.searchsorted(ends, idx, side="left")


def future_indices(
    indices: Any,
    episode_ends: Any,
    rng: np.random.Generator,
    mode: str = "geometric",
    geom_p: float = 0.5,
    uniform_span: Optional[int] = None,
) -> np.ndarray:
    """Sample a future index inside the same episode as each input index.

    ``mode="geometric"`` samples an offset ``~ Geometric(geom_p)`` (>= 1) and
    clamps it to the episode end; ``mode="uniform"`` samples uniformly from the
    remaining steps of the episode.  Indices that are already the last step of
    their episode fall back to themselves.
    """
    ends = np.asarray(episode_ends, dtype=np.int64).reshape(-1)
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    ep = episode_index_of(idx, ends)
    ep_end = ends[ep]  # inclusive end of the containing episode

    remaining = ep_end - idx
    mode = (mode or "geometric").lower()
    if mode in ("geometric", "geom"):
        offsets = rng.geometric(float(geom_p), size=idx.shape).astype(np.int64)
    elif mode in ("uniform",):
        # Uniform over (idx, ep_end]; ``remaining`` may be 0 for last steps.
        u = rng.random(idx.shape)
        span = np.maximum(remaining, 1)
        limits = span if uniform_span is None else np.minimum(span, int(uniform_span))
        offsets = 1 + np.floor(u * limits).astype(np.int64)
    else:
        raise ValueError(f"Unknown future sampling mode: {mode!r}")

    offsets = np.maximum(offsets, 1)
    goal_idx = np.minimum(idx + offsets, ep_end)
    goal_idx = np.maximum(goal_idx, idx)  # never go backwards
    return goal_idx


def her_goal_indices(
    indices: Any,
    episode_ends: Any,
    rng: Optional[np.random.Generator] = None,
    num_states: Optional[int] = None,
    p_current: float = DEFAULT_P_CURRENT,
    p_future: float = DEFAULT_P_FUTURE,
    p_random: float = DEFAULT_P_RANDOM,
    future_mode: str = "geometric",
    geom_p: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample hindsight goals for each transition index.

    Strategy per index: ``p_current`` -> the state itself, ``p_future`` -> a
    later state of the *same* trajectory (this is the standard HER "future"
    strategy), ``p_random`` -> a uniformly random dataset state.

    Returns:
        ``(goal_indices, strategy_ids)`` where ``strategy_ids`` is 0/1/2 for
        current/future/random (useful for logging and for the FRE-hint priors).
    """
    rng = get_rng(rng)
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    n = idx.shape[0]
    if num_states is None:
        ends = np.asarray(episode_ends, dtype=np.int64).reshape(-1)
        num_states = int(ends[-1]) + 1

    total = float(p_current) + float(p_future) + float(p_random)
    if total <= 0:
        raise ValueError("Goal sampling probabilities must sum to a positive value")
    p_cur = float(p_current) / total
    p_fut = float(p_future) / total

    u = rng.random(n)
    cur_mask = u < p_cur
    fut_mask = (u >= p_cur) & (u < p_cur + p_fut)
    rnd_mask = ~(cur_mask | fut_mask)

    goals = np.empty(n, dtype=np.int64)
    strategy = np.empty(n, dtype=np.int64)
    goals[cur_mask] = idx[cur_mask]
    strategy[cur_mask] = 0

    if fut_mask.any():
        goals[fut_mask] = future_indices(
            idx[fut_mask], episode_ends, rng, mode=future_mode, geom_p=geom_p
        )
        strategy[fut_mask] = 1

    if rnd_mask.any():
        goals[rnd_mask] = rng.integers(0, int(num_states), size=int(rnd_mask.sum()))
        strategy[rnd_mask] = 2

    return goals, strategy


# ---------------------------------------------------------------------------
# Goal-reaching prior
# ---------------------------------------------------------------------------


class GoalReachingPrior(RewardFunctionPrior):
    """Samples goal-reaching reward functions from the offline dataset.

    Args:
        state_dim: state dimensionality.
        source: offline dataset (dict, replay buffer, or ``(N, D)`` array) used
            to sample goal states.  Can also be set later via :meth:`set_source`.
        rng / seed: numpy RNG or seed.
        p_current / p_future / p_random: HER mixture (paper: 0.2 / 0.5 / 0.3).
        future_mode: ``"geometric"`` (default) or ``"uniform"`` future sampling.
        geom_p: geometric offset parameter.
        goal_threshold: success radius ``tau``.
        reward_success / reward_failure: reward values away/in the goal.
        reward_type: ``"sparse"`` (paper default), ``"dense"`` or ``"exp"``.
        metric / dims / weights: distance configuration.
        done_on_success: whether the done mask fires on success.
        goal_pool: optional explicit pool of goal states ``(M, D)``; when given
            goals are sampled from this pool instead of ``source`` observations.
        goal_indices: optional indices into the source used as the goal pool
            (used to build evaluation-style priors with a fixed goal set).
    """

    family = GOAL_REACHING_FAMILY

    def __init__(
        self,
        state_dim: Optional[int] = None,
        source: Optional[Any] = None,
        rng: Optional[Any] = None,
        seed: Optional[int] = None,
        p_current: float = DEFAULT_P_CURRENT,
        p_future: float = DEFAULT_P_FUTURE,
        p_random: float = DEFAULT_P_RANDOM,
        future_mode: str = "geometric",
        geom_p: float = 0.5,
        goal_threshold: float = DEFAULT_GOAL_THRESHOLD,
        reward_success: float = DEFAULT_GOAL_REWARD,
        reward_failure: float = DEFAULT_FAILURE_REWARD,
        reward_type: str = "sparse",
        metric: str = "euclidean",
        dims: Optional[Any] = None,
        weights: Optional[Any] = None,
        done_on_success: bool = True,
        goal_pool: Optional[Any] = None,
        goal_indices: Optional[Any] = None,
        clip: float = 1.0,
        name: str = "goal_reaching_prior",
        **kwargs: Any,
    ) -> None:
        super().__init__(state_dim=state_dim, source=source, rng=rng, seed=seed, **kwargs)

        self.p_current = float(p_current)
        self.p_future = float(p_future)
        self.p_random = float(p_random)
        self.future_mode = future_mode
        self.geom_p = float(geom_p)
        self.goal_threshold = float(goal_threshold)
        self.reward_success = float(reward_success)
        self.reward_failure = float(reward_failure)
        self.reward_type = str(reward_type).lower()
        self.metric = metric
        self.dims = dims
        self.weights = weights
        self.done_on_success = bool(done_on_success)
        self.clip = float(clip)
        self.name = name

        self._rng = get_rng(rng, seed)
        self._goal_pool: Optional[np.ndarray] = (
            None if goal_pool is None else np.asarray(goal_pool, dtype=np.float64)
        )
        self._goal_pool_indices: Optional[np.ndarray] = (
            None if goal_indices is None else np.asarray(goal_indices, dtype=np.int64)
        )
        self._episode_ends: Optional[np.ndarray] = None
        self._episode_starts: Optional[np.ndarray] = None

        if self._goal_pool is not None and state_dim is None:
            self.state_dim = int(self._goal_pool.shape[-1])

    # -- source / episode bookkeeping --------------------------------------

    def set_source(self, source: Any) -> None:  # noqa: D102 - base hook
        super().set_source(source)
        self._episode_ends = None
        self._episode_starts = None
        self._goal_pool = None

    def set_goal_pool(
        self, pool: Optional[Any] = None, indices: Optional[Any] = None
    ) -> None:
        """Fix the set of states goals may be drawn from (e.g. eval tasks)."""
        self._goal_pool = None if pool is None else np.asarray(pool, dtype=np.float64)
        self._goal_pool_indices = (
            None if indices is None else np.asarray(indices, dtype=np.int64)
        )

    def _resolve_source(self, source: Optional[Any] = None) -> Any:
        src = source if source is not None else getattr(self, "source", None)
        if src is None:
            raise ValueError(
                "GoalReachingPrior requires a dataset source. Pass `source=` or call "
                "`set_source(dataset)` before sampling reward functions."
            )
        return src

    def observations(self, source: Optional[Any] = None) -> np.ndarray:
        """Observation array of the (goal) pool used for sampling."""
        if self._goal_pool is not None:
            return self._goal_pool
        obs = np.asarray(get_observations(self._resolve_source(source)), dtype=np.float64)
        if self._goal_pool_indices is not None:
            return obs[self._goal_pool_indices]
        return obs

    @property
    def episode_ends(self) -> np.ndarray:
        """Inclusive end index of each episode (cached)."""
        if self._episode_ends is None:
            src = self._resolve_source()
            terminals = get_terminals(src)
            obs = get_observations(src)
            if terminals is None:
                ends = np.array([len(obs) - 1], dtype=np.int64)
            else:
                ends = np.asarray(np.nonzero(np.asarray(terminals).reshape(-1))[0], dtype=np.int64)
                if ends.size == 0 or ends[-1] != len(obs) - 1:
                    ends = np.concatenate([ends, [len(obs) - 1]]).astype(np.int64)
            self._episode_ends = ends
        return self._episode_ends

    @property
    def episode_starts(self) -> np.ndarray:
        """Inclusive start index of each episode (cached)."""
        if self._episode_starts is None:
            ends = self.episode_ends
            starts = np.concatenate([[0], ends[:-1] + 1]).astype(np.int64)
            self._episode_starts = starts
        return self._episode_starts

    def episode_bounds(self) -> List[Tuple[int, int]]:
        """``(start, end_exclusive)`` bounds of every episode."""
        return list(zip(self.episode_starts.tolist(), (self.episode_ends + 1).tolist()))

    # -- goal sampling ------------------------------------------------------

    def sample_goal_indices(
        self,
        indices: Optional[Any] = None,
        rng: Optional[Any] = None,
        num_goals: Optional[int] = None,
        source: Optional[Any] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample dataset indices to use as goals (HER mixture)."""
        rng = get_rng(rng) if rng is not None else self._rng
        obs = self.observations(source)
        n_states = obs.shape[0]
        if indices is None:
            k = int(num_goals if num_goals is not None else 1)
            indices = rng.integers(0, n_states, size=k)

        if self._goal_pool is not None:
            # Goals are drawn from a fixed pool: current/future are meaningless
            # so we simply sample uniformly from the pool.
            idx = np.asarray(indices, dtype=np.int64).reshape(-1)
            goals = rng.integers(0, n_states, size=idx.shape[0])
            return goals, np.full(idx.shape[0], 2, dtype=np.int64)

        return her_goal_indices(
            np.asarray(indices, dtype=np.int64),
            self.episode_ends,
            rng,
            num_states=n_states,
            p_current=self.p_current,
            p_future=self.p_future,
            p_random=self.p_random,
            future_mode=self.future_mode,
            geom_p=self.geom_p,
        )

    def sample_goals(
        self,
        num_goals: int = 1,
        rng: Optional[Any] = None,
        indices: Optional[Any] = None,
        source: Optional[Any] = None,
    ) -> np.ndarray:
        """Return ``(num_goals, D)`` goal states sampled from the source."""
        rng = get_rng(rng) if rng is not None else self._rng
        goal_idx, _ = self.sample_goal_indices(indices=indices, rng=rng, num_goals=num_goals)
        obs = self.observations(source)
        return obs[goal_idx]

    def sample_goal_indices_and_goals(
        self, num_goals: int = 1, rng: Optional[Any] = None, indices: Optional[Any] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(goal_indices, goals, strategy_ids)`` for logging/reuse."""
        rng = get_rng(rng) if rng is not None else self._rng
        goal_idx, strategy = self.sample_goal_indices(
            indices=indices, rng=rng, num_goals=num_goals
        )
        obs = self.observations()
        return goal_idx, obs[goal_idx], strategy

    def her_pairs(
        self,
        indices: Any,
        rng: Optional[Any] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Hindsight ``(state_indices, goal_indices, strategy_ids)`` pairs.

        Exposed for the GC-IQL / GC-BC baselines, which need the same
        current/future/random goal relabelling.
        """
        rng = get_rng(rng) if rng is not None else self._rng
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        goals, strategy = self.sample_goal_indices(indices=idx, rng=rng)
        return idx, goals, strategy

    # -- reward function construction --------------------------------------

    def build_function(
        self,
        goal: Any,
        goal_index: Optional[int] = None,
        strategy: Optional[str] = None,
        **overrides: Any,
    ) -> GoalReachingReward:
        """Wrap a goal state into a :class:`GoalReachingReward`."""
        kwargs: Dict[str, Any] = dict(
            threshold=self.goal_threshold,
            reward_success=self.reward_success,
            reward_failure=self.reward_failure,
            reward_type=self.reward_type,
            metric=self.metric,
            dims=self.dims,
            weights=self.weights,
            done_on_success=self.done_on_success,
            clip=self.clip,
            goal_index=goal_index,
            strategy=strategy,
        )
        kwargs.update(overrides)
        return GoalReachingReward(
            goal=goal, state_dim=self.state_dim, **kwargs
        )

    def sample_functions(  # noqa: D102 - base hook
        self,
        num_functions: int = 1,
        source: Optional[Any] = None,
        rng: Optional[Any] = None,
        indices: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[GoalReachingReward]:
        rng = get_rng(rng) if rng is not None else self._rng
        goal_idx, goals, strategy = self.sample_goal_indices_and_goals(
            num_goals=int(num_functions), rng=rng, indices=indices
        )
        strategy_names = {0: "current", 1: "future", 2: "random"}
        return [
            self.build_function(
                goals[k],
                goal_index=int(goal_idx[k]),
                strategy=strategy_names.get(int(strategy[k]), "random"),
                **kwargs,
            )
            for k in range(int(num_functions))
        ]

    # -- encoder context ---------------------------------------------------

    def sample_context(
        self,
        num_samples: int = CONTEXT_SIZE,
        source: Optional[Any] = None,
        rng: Optional[Any] = None,
        reward_fn: Optional[RewardFunction] = None,
        include_success: bool = True,
        num_success_samples: int = 1,
        **fn_kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray, GoalReachingReward]:
        """Sample a K-sample reward context for the FRE encoder.

        Guarantees that at least ``num_success_samples`` of the returned states
        achieve the goal, so the posterior can identify the task.

        Returns:
            ``(states, rewards, reward_fn)`` with ``states`` ``(num_samples, D)``
            and ``rewards`` ``(num_samples,)``.
        """
        rng = get_rng(rng) if rng is not None else self._rng
        if reward_fn is None:
            reward_fn = self.sample_functions(1, source=source, rng=rng, **fn_kwargs)[0]
        pool = self.observations(source)
        states, rewards = sample_reward_context(
            reward_fn,
            pool,
            num_samples=num_samples,
            rng=rng,
            include_success=include_success,
            num_success_samples=num_success_samples,
        )
        return states, rewards, reward_fn

    def sample_context_and_decoder(
        self,
        num_context: int = CONTEXT_SIZE,
        num_decoder: int = DECODER_SIZE,
        source: Optional[Any] = None,
        rng: Optional[Any] = None,
        reward_fn: Optional[RewardFunction] = None,
        include_success: bool = True,
        **fn_kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, GoalReachingReward]:
        """Context (K) + disjoint decoder (K') state/reward pairs.

        Returns ``(context_states, context_rewards, decoder_states,
        decoder_rewards, reward_fn)`` where the decoder states are sampled
        independently of the encoder context states.
        """
        rng = get_rng(rng) if rng is not None else self._rng
        if reward_fn is None:
            reward_fn = self.sample_functions(1, source=source, rng=rng, **fn_kwargs)[0]
        pool = self.observations(source)
        return sample_context_and_decoder_pairs(
            reward_fn,
            pool,
            num_context=num_context,
            num_decoder=num_decoder,
            rng=rng,
            include_success=include_success,
        )

    def extra_repr(self) -> str:  # noqa: D102 - base hook
        return (
            f"p_current={self.p_current}, p_future={self.p_future}, p_random={self.p_random}, "
            f"future_mode={self.future_mode}, threshold={self.goal_threshold}, "
            f"reward_type={self.reward_type}"
        )


#: Plan-facing alias (the plan refers to this prior as the "goal-reaching" prior).
GoalReachingRewardPrior = GoalReachingPrior


# ---------------------------------------------------------------------------
# Context builders (guarantee the goal is represented in the encoder context)
# ---------------------------------------------------------------------------


def sample_reward_context(
    reward_fn: RewardFunction,
    state_pool: Any,
    num_samples: int = CONTEXT_SIZE,
    rng: Optional[np.random.Generator] = None,
    include_success: bool = True,
    num_success_samples: int = 1,
    replace: bool = True,
    shuffle: bool = True,
    success_states: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample ``num_samples`` ``(state, reward)`` pairs for the encoder.

    Per App. B, "we guarantee that at least one of the encoding samples contains
    the goal".  For goal-reaching functions this is implemented by reserving
    ``num_success_samples`` slots for states that achieve the goal (the goal
    itself when the reward function exposes one, otherwise sampled from
    ``success_states``).
    """
    rng = get_rng(rng)
    pool = np.asarray(to_numpy(state_pool, dtype=np.float64))
    if pool.ndim == 1:
        pool = pool.reshape(-1, 1)
    n_pool = pool.shape[0]
    if n_pool == 0:
        raise ValueError("state_pool is empty")

    num_samples = int(num_samples)
    num_success_samples = int(max(0, min(num_success_samples, num_samples))) if include_success else 0
    num_random = num_samples - num_success_samples

    chosen: List[np.ndarray] = []
    if num_success_samples > 0:
        candidates = None
        if success_states is not None:
            candidates = np.asarray(to_numpy(success_states, dtype=np.float64)).reshape(-1, pool.shape[-1])
        elif getattr(reward_fn, "goal", None) is not None:
            candidates = np.asarray(reward_fn.goal, dtype=np.float64).reshape(1, -1)
        if candidates is None or candidates.shape[0] == 0:
            # No explicit success state available: draw the success slots from
            # states that the reward function actually considers successful.
            mask = np.asarray(reward_fn.done(pool), dtype=bool).reshape(-1)
            if mask.any():
                candidates = pool[np.nonzero(mask)[0]]
            else:
                candidates = pool[rng.integers(0, n_pool, size=num_success_samples)]
        sel = rng.integers(0, candidates.shape[0], size=num_success_samples)
        chosen.append(np.asarray(candidates[sel]).reshape(num_success_samples, pool.shape[-1]))

    if num_random > 0:
        idx = rng.integers(0, n_pool, size=num_random) if replace else rng.choice(
            n_pool, size=num_random, replace=False
        )
        chosen.append(pool[idx])

    states = np.concatenate(chosen, axis=0) if chosen else np.zeros((0, pool.shape[-1]), dtype=np.float64)
    if shuffle:
        perm = rng.permutation(states.shape[0])
        states = states[perm]

    rewards = np.asarray(reward_fn.compute_numpy(states), dtype=np.float32).reshape(-1)
    return states.astype(np.float32), rewards


def sample_context_and_decoder_pairs(
    reward_fn: RewardFunction,
    state_pool: Any,
    num_context: int = CONTEXT_SIZE,
    num_decoder: int = DECODER_SIZE,
    rng: Optional[np.random.Generator] = None,
    include_success: bool = True,
    num_success_samples: int = 1,
    decoder_states: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, RewardFunction]:
    """Build (context, decoder) pairs for the Phase-1 FRE objective.

    The K encoder context samples come from :func:`sample_reward_context` (goal
    guaranteed present); the K' decoder states are drawn independently of the
    context states, as specified in the paper ("K' = 8 decoder states, disjoint
    from the K = 32 encoder states").
    """
    rng = get_rng(rng)
    context_states, context_rewards = sample_reward_context(
        reward_fn,
        state_pool,
        num_samples=num_context,
        rng=rng,
        include_success=include_success,
        num_success_samples=num_success_samples,
    )

    pool = np.asarray(to_numpy(state_pool, dtype=np.float64))
    if pool.ndim == 1:
        pool = pool.reshape(-1, 1)

    if decoder_states is None:
        idx = rng.integers(0, pool.shape[0], size=int(num_decoder))
        decoder_states = pool[idx]
    else:
        decoder_states = np.asarray(to_numpy(decoder_states, dtype=np.float64)).reshape(
            int(num_decoder), pool.shape[-1]
        )

    decoder_rewards = np.asarray(
        reward_fn.compute_numpy(decoder_states), dtype=np.float32
    ).reshape(-1)
    return (
        context_states,
        context_rewards,
        decoder_states.astype(np.float32),
        decoder_rewards,
        reward_fn,
    )


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def make_goal_reaching_reward(goal: Any, **kwargs: Any) -> GoalReachingReward:
    """Convenience factory for a single goal-reaching reward function."""
    return GoalReachingReward(goal=goal, **kwargs)


def make_goal_reaching_prior(
    source: Optional[Any] = None,
    state_dim: Optional[int] = None,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> GoalReachingPrior:
    """Convenience factory for the HER goal-reaching prior."""
    return GoalReachingPrior(state_dim=state_dim, source=source, seed=seed, **kwargs)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(0)
    n, d = 200, 4
    obs = rng.normal(size=(n, d)).astype(np.float32)
    terminals = np.zeros(n, dtype=np.float32)
    terminals[[49, 99, 149, 199]] = 1.0
    source = {"observations": obs, "terminals": terminals}

    prior = GoalReachingPrior(state_dim=d, source=source, seed=0, goal_threshold=0.5)
    assert np.isclose(prior.p_current + prior.p_future + prior.p_random, 1.0)

    goal_idx, goals, strategy = prior.sample_goal_indices_and_goals(num_goals=1000)
    frac = np.bincount(strategy, minlength=3) / 1000.0
    print("strategy fractions (current/future/random):", np.round(frac, 3))
    assert abs(frac[0] - 0.2) < 0.06 and abs(frac[1] - 0.5) < 0.06 and abs(frac[2] - 0.3) < 0.06

    fns = prior.sample_functions(8)
    assert all(isinstance(f, GoalReachingReward) for f in fns)
    f = fns[0]
    assert np.isclose(f.compute_numpy(f.goal).reshape(-1)[0], 0.0)
    far = f.goal + 10.0
    assert np.isclose(f.compute_numpy(far).reshape(-1)[0], -1.0)
    assert bool(f.done(f.goal).reshape(-1)[0]) is True
    assert bool(f.done(far).reshape(-1)[0]) is False

    states, rewards = sample_reward_context(f, obs, num_samples=32, rng=rng)
    assert states.shape == (32, d) and rewards.shape == (32,)
    assert (rewards >= -1.0 - 1e-6).all() and (rewards <= 1.0 + 1e-6).all()
    assert (rewards == 0.0).any(), "context must contain at least one goal sample"

    cs, cr, ds, dr, fn = sample_context_and_decoder_pairs(f, obs, rng=rng)
    assert cs.shape == (32, d) and ds.shape == (8, d)
    print("goal_reaching self-test passed")
