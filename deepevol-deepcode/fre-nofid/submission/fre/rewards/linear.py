"""Linear reward functions for the FRE unsupervised reward prior.

This module implements the *linear* component of the FRE reward-function prior
``p(eta)`` described in Section 4.2 / Appendix B of

    "Zero-Shot Reinforcement Learning via Functional Reward Encodings"

A linear reward function is a Markovian map ``eta: S -> R`` of the form

    eta(s) = sum_i  w_i * m_i * s_i          (inner product with a random vector)

where

  * ``w ~ Uniform(-1, 1)^d`` is a random weight vector, and
  * ``m ~ Bernoulli(1 - 0.9)^d`` is a *sparsity* mask which independently zeroes
    each input dimension with probability ``0.9`` (i.e. keeps a dimension with
    probability ``0.1``).

The resulting scalar reward is finally clipped to ``[-1, 1]`` so that it can be
discretised by the FRE encoder (rescale -> *32 -> floor -> 32 bins).

Special case (AntMaze): the raw AntMaze observation contains the XY position of
the ant in its first two dims.  Because the paper explicitly removes XY
positions from the generation of linear reward functions for AntMaze, the prior
accepts an ``exclude_dims`` argument; when it is supplied the excluded dims are
forced to zero (masked out) so that position-only linear functions cannot be
sampled.  Convenience constructors ``antmaze_linear_prior`` are provided.

The module intentionally mirrors the interface of
``fre/rewards/goal_reaching.py``: a :class:`LinearReward` (subclass of
:class:`~fre.rewards.base.RewardFunction`) for a *single* reward function, and a
:class:`LinearPrior` (subclass of
:class:`~fre.rewards.base.RewardFunctionPrior`) which samples a family of such
functions.  The latter exposes the same ``sample_context`` /
``sample_context_and_decoder`` helpers as the goal-reaching prior so the mixture
prior in ``fre/rewards/prior.py`` can drive all three families identically.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Robust import of the shared reward base classes.  Supports both
# ``import fre.rewards.linear`` and a direct file import / script execution.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import plumbing
    from .base import (
        RewardFunction,
        RewardFunctionPrior,
        episode_boundaries,
        get_observations,
        get_rng,
        get_terminals,
        to_numpy,
        to_torch,
    )
except ImportError:  # pragma: no cover - import plumbing
    try:
        from base import (  # type: ignore
            RewardFunction,
            RewardFunctionPrior,
            episode_boundaries,
            get_observations,
            get_rng,
            get_terminals,
            to_numpy,
            to_torch,
        )
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from fre.rewards.base import (  # type: ignore
            RewardFunction,
            RewardFunctionPrior,
            episode_boundaries,
            get_observations,
            get_rng,
            get_terminals,
            to_numpy,
            to_torch,
        )


# ---------------------------------------------------------------------------
# Constants (Appendix B)
# ---------------------------------------------------------------------------
LINEAR_FAMILY = "linear"

#: Probability that any individual input dimension is zeroed by the sparsity
#: mask (the paper masks each dim with probability 0.9).
DEFAULT_MASK_PROB = 0.9

#: Probability that a dimension survives the sparsity mask.
DEFAULT_KEEP_PROB = 1.0 - DEFAULT_MASK_PROB

#: Uniform weight range for the random weight vector.
DEFAULT_WEIGHT_RANGE = 1.0

#: Encoder-context / decoder-context sizes used by FRE (K=32, K'=8).
CONTEXT_SIZE = 32
DECODER_SIZE = 8

#: AntMaze XY observation dims that must be excluded from linear reward
#: generation (Section 4.2: "on AntMaze remove XY positions").
ANTMAZE_XY_DIMS = (0, 1)

#: Observation dims for the environments handled by FRE.
ANTMAZE_OBS_DIM = 29
ANTMAZE_ACTION_DIM = 8

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Single linear reward function
# ---------------------------------------------------------------------------
class LinearReward(RewardFunction):
    """A single linear reward function ``eta(s) = w . (m * s)``.

    Parameters
    ----------
    weights:
        ``(state_dim,)`` array (or scalar/list) of real-valued weights ``w``.
    mask:
        Optional ``(state_dim,)`` binary mask ``m``.  ``True`` keeps a dim,
        ``False`` zeroes it.  When omitted a mask is derived from
        ``exclude_dims`` (excluded dims are zeroed, all others kept).
    state_dim:
        Optional dimensionality; inferred from ``weights`` when possible.
    exclude_dims:
        Iterable of dims forced to zero (e.g. ``(0, 1)`` for AntMaze XY).
    normalise:
        When ``True`` the (masked) state is standardised with ``mean``/``std``
        before the inner product (the base class handles the transform).
    clip:
        Output clipping magnitude (default ``1.0`` matching the prior range).
    """

    family = LINEAR_FAMILY

    def __init__(
        self,
        weights: Any,
        mask: Optional[Any] = None,
        state_dim: Optional[int] = None,
        exclude_dims: Optional[Sequence[int]] = None,
        clip: float = 1.0,
        name: Optional[str] = None,
        mean: Optional[Any] = None,
        std: Optional[Any] = None,
        normalise: bool = False,
        device: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if state_dim is None:
            state_dim = int(w.shape[0])
        # Pad / truncate weights to the declared state dim.
        if w.shape[0] < state_dim:
            w = np.concatenate([w, np.zeros(state_dim - w.shape[0], dtype=np.float64)])
        elif w.shape[0] > state_dim:
            w = w[:state_dim]

        if mask is None:
            m = np.ones(state_dim, dtype=np.float64)
            if exclude_dims is not None:
                for d in exclude_dims:
                    d = int(d)
                    if 0 <= d < state_dim:
                        m[d] = 0.0
        else:
            m = np.asarray(mask, dtype=np.float64).reshape(-1)
            m = (m > 0.5).astype(np.float64)
            if m.shape[0] < state_dim:
                m = np.concatenate([m, np.zeros(state_dim - m.shape[0], dtype=np.float64)])
            elif m.shape[0] > state_dim:
                m = m[:state_dim]

        self.weights = (w * m).astype(np.float32)
        self.mask = m.astype(np.float32)
        self.raw_weights = w.astype(np.float32)
        self.state_dim = int(state_dim)
        self.exclude_dims = tuple(int(d) for d in (exclude_dims or ()))
        self.num_active = int(np.sum(m > 0.5))

        meta = dict(metadata or {})
        meta.setdefault("num_active_dims", self.num_active)
        meta.setdefault("exclude_dims", list(self.exclude_dims))

        super().__init__(
            state_dim=state_dim,
            name=name or "linear_reward",
            clip=clip,
            mean=mean,
            std=std,
            normalise=normalise,
            device=device,
            metadata=meta,
            **kwargs,
        )

    # -- core computation ---------------------------------------------------
    def _compute(self, states: np.ndarray) -> np.ndarray:
        """Inner product of the (masked) state with the weight vector."""
        s = np.asarray(states, dtype=np.float64)
        active = self.mask > 0.5
        if not np.any(active):
            return np.zeros(s.shape[:-1], dtype=np.float64)
        w = self.weights.astype(np.float64)
        return np.sum(s[..., active] * w[active], axis=-1)

    # -- introspection ------------------------------------------------------
    def active_dims(self) -> np.ndarray:
        """Indices of the dimensions kept by the sparsity mask."""
        return np.flatnonzero(self.mask > 0.5)

    def describe(self) -> Dict[str, Any]:
        info = super().describe()
        info.update(
            {
                "num_active_dims": self.num_active,
                "active_dims": self.active_dims().tolist()[:64],
                "weight_norm": float(np.linalg.norm(self.weights)),
                "exclude_dims": list(self.exclude_dims),
            }
        )
        return info

    def extra_repr(self) -> str:
        return f"state_dim={self.state_dim}, active={self.num_active}, exclude={self.exclude_dims}"


# ---------------------------------------------------------------------------
# Linear reward-function prior
# ---------------------------------------------------------------------------
class LinearPrior(RewardFunctionPrior):
    """Samples linear reward functions ``eta(s) = w . (m * s)``.

    Mirrors :class:`~fre.rewards.goal_reaching.GoalReachingPrior`: it can be
    given the offline dataset (``source``) so the state dimensionality is
    inferred, and it exposes ``sample_context`` / ``sample_context_and_decoder``
    so the mixture prior can drive every family through the same call site.

    Parameters
    ----------
    state_dim:
        Observation dimensionality.  Inferred from ``source`` when ``None``.
    exclude_dims:
        Dims forced to zero when generating reward functions.  For AntMaze this
        should be ``(0, 1)`` (the XY position).
    mask_prob:
        Per-dim probability of being zeroed (paper: 0.9).
    weight_range:
        Weights are sampled ``Uniform(-weight_range, weight_range)``.
    source:
        Dataset / replay buffer used only to infer ``state_dim`` and to draw
        random states for encoder context sampling.
    """

    family = LINEAR_FAMILY

    def __init__(
        self,
        state_dim: Optional[int] = None,
        source: Any = None,
        rng: Optional[Any] = None,
        seed: Optional[int] = None,
        mask_prob: float = DEFAULT_MASK_PROB,
        keep_prob: Optional[float] = None,
        weight_range: float = DEFAULT_WEIGHT_RANGE,
        exclude_dims: Optional[Sequence[int]] = None,
        clip: float = 1.0,
        name: str = "linear_prior",
        **kwargs: Any,
    ) -> None:
        if keep_prob is not None:
            mask_prob = 1.0 - float(keep_prob)
        self.mask_prob = float(np.clip(mask_prob, 0.0, 1.0))
        self.weight_range = float(weight_range)
        self.exclude_dims = tuple(int(d) for d in (exclude_dims or ()))
        self._source = source

        if state_dim is None and source is not None:
            state_dim = _infer_state_dim(source)
        if state_dim is None:
            state_dim = ANTMAZE_OBS_DIM

        super().__init__(
            state_dim=int(state_dim),
            source=source,
            rng=rng,
            seed=seed,
            clip=clip,
            name=name,
            **kwargs,
        )
        # Attach dataset-derived generators (cached).
        self._observations: Optional[np.ndarray] = None
        self._terminals: Optional[np.ndarray] = None

    # -- dataset wiring -----------------------------------------------------
    def set_source(self, source: Any) -> None:
        """Attach the offline dataset used for context-state sampling."""
        self._source = source
        self._observations = None
        self._terminals = None
        if source is not None and self.state_dim is None:
            self.state_dim = _infer_state_dim(source)

    @property
    def observations_cached(self) -> np.ndarray:
        """Cached ``(N, D)`` observation array from the source dataset."""
        if self._observations is None:
            if self._source is None:
                raise ValueError(
                    "LinearPrior has no source dataset; call set_source(...) "
                    "or pass source= to the constructor before sampling contexts."
                )
            obs = np.asarray(get_observations(self._source), dtype=np.float32)
            if obs.ndim == 1:
                obs = obs[None, :]
            self._observations = obs
            self.state_dim = int(obs.shape[-1])
        elif self.state_dim is None:
            self.state_dim = int(self._observations.shape[-1])
        return self._observations

    @property
    def terminals_cached(self) -> Optional[np.ndarray]:
        if self._terminals is None and self._source is not None:
            term = get_terminals(self._source)
            self._terminals = None if term is None else np.asarray(term).reshape(-1)
        return self._terminals

    # -- sampling -----------------------------------------------------------
    def sample_mask(self, rng: Optional[Any] = None, state_dim: Optional[int] = None) -> np.ndarray:
        """Sample a sparsity mask; dims in ``exclude_dims`` are always zeroed."""
        rng = get_rng(rng, None)
        d = int(state_dim if state_dim is not None else self.state_dim)
        keep = rng.random(d) >= self.mask_prob
        mask = keep.astype(np.float64)
        for dim in self.exclude_dims:
            if 0 <= dim < d:
                mask[dim] = 0.0
        # Guarantee that at least one dimension survives, otherwise the reward
        # function is identically zero (degenerate and uninformative).
        if not np.any(mask > 0.5) and d > 0:
            free = [i for i in range(d) if i not in set(self.exclude_dims)]
            pool = free if free else list(range(d))
            mask[int(rng.choice(pool))] = 1.0
        return mask

    def sample_weights(self, rng: Optional[Any] = None, state_dim: Optional[int] = None) -> np.ndarray:
        """Sample ``Uniform(-weight_range, weight_range)`` weights."""
        rng = get_rng(rng, None)
        d = int(state_dim if state_dim is not None else self.state_dim)
        return rng.uniform(-self.weight_range, self.weight_range, size=d)

    def sample_function(
        self,
        rng: Optional[Any] = None,
        state_dim: Optional[int] = None,
        mask: Optional[Any] = None,
        weights: Optional[Any] = None,
        **fn_kwargs: Any,
    ) -> LinearReward:
        """Sample a single :class:`LinearReward`."""
        rng = get_rng(rng, None)
        d = int(state_dim if state_dim is not None else self.state_dim)
        w = np.asarray(weights, dtype=np.float64) if weights is not None else self.sample_weights(rng, d)
        m = np.asarray(mask, dtype=np.float64) if mask is not None else self.sample_mask(rng, d)
        meta = {"family": LINEAR_FAMILY, "source": "linear_prior"}
        return LinearReward(
            weights=w,
            mask=m,
            state_dim=d,
            exclude_dims=self.exclude_dims,
            clip=float(fn_kwargs.pop("clip", self.clip if hasattr(self, "clip") else 1.0)),
            **fn_kwargs,
        ) if False else self._build(w, m, d, fn_kwargs, meta)

    def _build(
        self,
        weights: np.ndarray,
        mask: np.ndarray,
        state_dim: int,
        fn_kwargs: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LinearReward:
        return LinearReward(
            weights=weights,
            mask=mask,
            state_dim=state_dim,
            exclude_dims=self.exclude_dims,
            clip=float(getattr(self, "clip", 1.0)),
            metadata=metadata,
            **fn_kwargs,
        )

    def sample_functions(
        self,
        num_functions: int = 1,
        source: Any = None,
        rng: Optional[Any] = None,
        state_dim: Optional[int] = None,
        **kwargs: Any,
    ) -> List[LinearReward]:
        """Sample ``num_functions`` independent linear reward functions."""
        if source is not None:
            self.set_source(source)
        rng = get_rng(rng if rng is not None else self.rng, self.seed)
        d = int(state_dim if state_dim is not None else self.state_dim)
        functions: List[LinearReward] = []
        for _ in range(int(num_functions)):
            functions.append(self.sample_function(rng=rng, state_dim=d, **kwargs))
        return functions

    def __call__(self, num_functions: int = 1, **kwargs: Any):
        functions = self.sample_functions(num_functions, **kwargs)
        return functions[0] if int(num_functions) == 1 else functions

    # -- encoder / decoder context builders ---------------------------------
    def sample_states(
        self,
        num_states: int = CONTEXT_SIZE,
        rng: Optional[Any] = None,
        source: Any = None,
        indices: Optional[Any] = None,
        replace: bool = True,
    ) -> np.ndarray:
        """Draw ``num_states`` states (from the dataset when available)."""
        rng = get_rng(rng if rng is not None else self.rng, self.seed)
        if source is not None:
            self.set_source(source)
        if indices is not None:
            idx = np.asarray(indices).reshape(-1)
            obs = self.observations_cached if self._source is not None else None
            if obs is not None and idx.size:
                return obs[np.clip(idx, 0, obs.shape[0] - 1)].astype(np.float32)
        if self._source is not None:
            obs = self.observations_cached
            n = obs.shape[0]
            sel = rng.integers(0, n, size=int(num_states)) if replace else _choice_no_replace(rng, n, num_states)
            return obs[sel].astype(np.float32)
        # Fallback: standard-normal states of the declared dimensionality.
        return rng.normal(size=(int(num_states), int(self.state_dim))).astype(np.float32)

    def sample_context(
        self,
        num_samples: int = CONTEXT_SIZE,
        source: Any = None,
        rng: Optional[Any] = None,
        reward_fn: Optional[RewardFunction] = None,
        indices: Optional[Any] = None,
        shuffle: bool = True,
        **fn_kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(states, rewards)`` for the K encoder context pairs.

        Shapes are ``(K, state_dim)`` and ``(K,)``; rewards come from
        ``reward_fn`` (sampled fresh when not supplied).
        """
        rng = get_rng(rng if rng is not None else self.rng, self.seed)
        if reward_fn is None:
            reward_fn = self.sample_function(rng=rng, **fn_kwargs)
        states = self.sample_states(int(num_samples), rng=rng, source=source, indices=indices)
        rewards = reward_fn.compute_numpy(states)
        if shuffle and states.shape[0] > 1:
            perm = rng.permutation(states.shape[0])
            states, rewards = states[perm], rewards[perm]
        return states.astype(np.float32), rewards.astype(np.float32)

    def sample_context_and_decoder(
        self,
        num_context: int = CONTEXT_SIZE,
        num_decoder: int = DECODER_SIZE,
        source: Any = None,
        rng: Optional[Any] = None,
        reward_fn: Optional[RewardFunction] = None,
        indices: Optional[Any] = None,
        disjoint: bool = True,
        **fn_kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, RewardFunction]:
        """Return disjoint (context, decoder) `(states, rewards)` pairs.

        Mirrors
        :func:`fre.rewards.goal_reaching.sample_context_and_decoder_pairs`;
        returns ``(ctx_states, ctx_rewards, dec_states, dec_rewards, reward_fn)``.
        """
        rng = get_rng(rng if rng is not None else self.rng, self.seed)
        if reward_fn is None:
            reward_fn = self.sample_function(rng=rng, **fn_kwargs)
        n_total = int(num_context) + int(num_decoder)
        states = self.sample_states(n_total, rng=rng, source=source, indices=indices)
        rewards = reward_fn.compute_numpy(states).astype(np.float32)
        if disjoint and self._source is not None:
            # Guarantee that context / decoder states are different rows.
            perm = rng.permutation(n_total)
            states, rewards = states[perm], rewards[perm]
        ctx_states = states[: int(num_context)].astype(np.float32)
        ctx_rewards = rewards[: int(num_context)].astype(np.float32)
        dec_states = states[int(num_context):].astype(np.float32)
        dec_rewards = rewards[int(num_context):].astype(np.float32)
        return ctx_states, ctx_rewards, dec_states, dec_rewards, reward_fn

    # -- introspection ------------------------------------------------------
    def describe(self) -> Dict[str, Any]:
        return {
            "family": LINEAR_FAMILY,
            "state_dim": self.state_dim,
            "mask_prob": self.mask_prob,
            "weight_range": self.weight_range,
            "exclude_dims": list(self.exclude_dims),
        }

    def extra_repr(self) -> str:
        return (
            f"state_dim={self.state_dim}, mask_prob={self.mask_prob}, "
            f"weight_range={self.weight_range}, exclude={self.exclude_dims}"
        )


# Plan-facing alias.
LinearRewardPrior = LinearPrior


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _infer_state_dim(source: Any) -> Optional[int]:
    """Best-effort inference of the observation dimensionality of ``source``."""
    if source is None:
        return None
    try:
        obs = get_observations(source)
        obs = np.asarray(obs)
        if obs.ndim == 1:
            return int(obs.shape[0])
        return int(obs.shape[-1])
    except Exception:
        for attr in ("obs_dim", "observation_dim", "state_dim"):
            val = getattr(source, attr, None)
            if isinstance(val, int):
                return int(val)
        return None


def _choice_no_replace(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
    k = int(min(k, n))
    return rng.choice(n, size=k, replace=False)


def make_linear_reward(weights: Any, **kwargs: Any) -> LinearReward:
    """Convenience constructor for a single :class:`LinearReward`."""
    return LinearReward(weights, **kwargs)


def make_linear_prior(
    source: Any = None,
    state_dim: Optional[int] = None,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> LinearPrior:
    """Convenience constructor for a :class:`LinearPrior`."""
    return LinearPrior(state_dim=state_dim, source=source, seed=seed, **kwargs)


def antmaze_linear_prior(
    source: Any = None,
    state_dim: int = ANTMAZE_OBS_DIM,
    exclude_dims: Sequence[int] = ANTMAZE_XY_DIMS,
    **kwargs: Any,
) -> LinearPrior:
    """Linear prior configured for AntMaze (XY position dims removed)."""
    return LinearPrior(
        state_dim=state_dim,
        source=source,
        exclude_dims=exclude_dims,
        **kwargs,
    )


def sample_linear_context(
    reward_fn: LinearReward,
    state_pool: Any,
    num_samples: int = CONTEXT_SIZE,
    rng: Optional[Any] = None,
    replace: bool = True,
    shuffle: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample ``(states (K,D), rewards (K,))`` for a given linear reward fn.

    Mirrors :func:`fre.rewards.goal_reaching.sample_reward_context` so the
    mixture prior can build encoder contexts for the linear family.
    """
    rng = get_rng(rng, None)
    states = np.asarray(state_pool, dtype=np.float32)
    if states.ndim == 1:
        states = states[None, :]
    if states.shape[0] == 0:
        raise ValueError("state_pool must contain at least one state")
    n = states.shape[0]
    sel = rng.integers(0, n, size=int(num_samples)) if replace else _choice_no_replace(rng, n, num_samples)
    sel_states = states[sel].astype(np.float32)
    rewards = reward_fn.compute_numpy(sel_states).astype(np.float32)
    if shuffle and sel_states.shape[0] > 1:
        perm = rng.permutation(sel_states.shape[0])
        sel_states, rewards = sel_states[perm], rewards[perm]
    return sel_states, rewards


# Entries marked ``family="linear"`` are discoverable by the mixture prior.
__all__ = [
    "LINEAR_FAMILY",
    "DEFAULT_MASK_PROB",
    "DEFAULT_KEEP_PROB",
    "DEFAULT_WEIGHT_RANGE",
    "ANTMAZE_XY_DIMS",
    "CONTEXT_SIZE",
    "DECODER_SIZE",
    "LinearReward",
    "LinearPrior",
    "LinearRewardPrior",
    "make_linear_reward",
    "make_linear_prior",
    "antmaze_linear_prior",
    "sample_linear_context",
]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    rng = get_rng(None, 0)
    d = 29

    # 1) Sparsity statistics over many sampled functions.
    prior = LinearPrior(state_dim=d, seed=0)
    masks = np.stack([prior.sample_mask(rng) for _ in range(500)])
    keep_rate = float(masks.mean())
    print(f"[test] empirical keep-rate = {keep_rate:.3f} (expected ~{DEFAULT_KEEP_PROB})")
    assert abs(keep_rate - DEFAULT_KEEP_PROB) < 0.05, keep_rate

    # 2) Weight range.
    w = prior.sample_weights(rng)
    assert np.all(np.abs(w) <= DEFAULT_WEIGHT_RANGE + 1e-6), np.abs(w).max()

    # 3) Reward values and clipping.
    fn = prior.sample_function(rng)
    states = rng.normal(size=(64, d)).astype(np.float32) * 3.0
    r = fn.compute_numpy(states)
    assert r.shape == (64,), r.shape
    assert np.all(r >= -1.0 - 1e-6) and np.all(r <= 1.0 + 1e-6), (r.min(), r.max())
    print(f"[test] reward range = [{r.min():.3f}, {r.max():.3f}]")

    # 4) AntMaze XY exclusion: weights on dims 0,1 must be zero.
    am = antmaze_linear_prior(state_dim=d, seed=1)
    fn_am = am.sample_function(rng)
    assert abs(fn_am.weights[0]) < 1e-9 and abs(fn_am.weights[1]) < 1e-9, fn_am.weights[:2]
    print("[test] AntMaze XY dims excluded OK")

    # 5) Batch of functions via sample_functions.
    fns = prior.sample_functions(5, rng=rng)
    assert len(fns) == 5
    assert all(isinstance(f, LinearReward) for f in fns)

    # 6) Context / decoder builders with a synthetic dataset.
    class _Src:
        def __init__(self):
            self.observations = rng.normal(size=(200, d)).astype(np.float32)

    src = _Src()
    prior2 = LinearPrior(source=src, seed=2)
    assert prior2.state_dim == d, prior2.state_dim
    cs, cr = prior2.sample_context(32, rng=rng)
    assert cs.shape == (32, d) and cr.shape == (32,), (cs.shape, cr.shape)
    cs2, cr2, ds, dr, rf = prior2.sample_context_and_decoder(32, 8, rng=rng)
    assert cs2.shape == (32, d) and ds.shape == (8, d)
    assert cr2.shape == (32,) and dr.shape == (8,)
    print("[test] context/decoder builders OK")

    # 7) Torch interop via base class.
    try:
        import torch

        t = fn(states)
        assert tuple(t.shape) == (64,), tuple(t.shape)
        print("[test] torch interop OK")
    except ImportError:
        pass

    # 8) Sanity: a pure-random policy's linear reward is near zero in mean.
    print(f"[test] mean reward over random states = {r.mean():.4f}")
    print("All linear reward self-tests passed.")
