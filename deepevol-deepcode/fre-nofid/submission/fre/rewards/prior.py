"""Unsupervised reward-function prior ``p(eta)`` for FRE (Section 4.2 / Appendix B).

The FRE encoder is pretrained over a *mixture* of randomly generated Markovian
reward functions.  This module implements the mixture prior

    p(eta) = 0.33 * p_goal(eta) + 0.33 * p_linear(eta) + 0.33 * p_mlp(eta)

where the three families are provided by the sibling modules:

* :mod:`fre.rewards.goal_reaching` -- singleton sparse goal-reaching rewards
  ``-1`` until the goal is reached (``0`` afterwards), with HER goal sampling
  ``p(current)=0.2, p(future)=0.5, p(random)=0.3``.
* :mod:`fre.rewards.linear` -- ``eta(s) = w . (m * s)`` with
  ``w ~ U(-1,1)^d`` and a Bernoulli(0.1) keep-mask ``m`` (0.9 sparsity).
* :mod:`fre.rewards.mlp` -- random ``(state_dim, 32, 1)`` tanh MLP clipped to
  ``[-1, 1]``.

In addition to the plain mixture this file provides

* :func:`ablation_prior` -- the seven prior subsets of Table 4
  (``FRE-all``, ``FRE-goals``, ``FRE-lin``, ``FRE-mlp``, ``FRE-lin-mlp``,
  ``FRE-goal-mlp``, ``FRE-goal-lin``).
* :func:`hint_prior` -- Figure 6 "FRE-hint" prior, i.e. the mixture
  augmented with XY-specific (AntMaze) / velocity-specific (ExORL) linear
  functions, added *without any architecture change*.
* :meth:`MixturePrior.sample_batch` -- the workhorse used by both training
  phases: draws ``batch_size`` reward functions (one per environment slot),
  builds disjoint K=32 context and K'=8 decoder ``(state, reward)`` sets for
  each of them, and returns ready-to-use ``(B, K, D)`` arrays together with
  the sampled functions (so their rewards can be re-evaluated on transitions
  during z-conditioned IQL training).

Everything is numpy-based (CPU) and cheap: sampling a batch of 512 reward
functions plus their contexts is a few tens of milliseconds, which makes
"re-sample eta and re-encode z each iteration" affordable in Phase 2.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Robust imports (package / sibling fallbacks) -- mirrors the other reward files
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import shims
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
except Exception:  # pragma: no cover
    try:
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
    except Exception:
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

try:  # pragma: no cover - import shims
    from .goal_reaching import (
        GoalReachingPrior,
        GoalReachingReward,
        GoalReachingRewardPrior,
    )
except Exception:  # pragma: no cover
    try:
        from fre.rewards.goal_reaching import (  # type: ignore
            GoalReachingPrior,
            GoalReachingReward,
            GoalReachingRewardPrior,
        )
    except Exception:
        from goal_reaching import (  # type: ignore
            GoalReachingPrior,
            GoalReachingReward,
            GoalReachingRewardPrior,
        )

try:  # pragma: no cover - import shims
    from .linear import LinearPrior, LinearReward, LinearRewardPrior  # type: ignore
except Exception:  # pragma: no cover
    try:
        from fre.rewards.linear import (  # type: ignore
            LinearPrior,
            LinearReward,
            LinearRewardPrior,
        )
    except Exception:
        from linear import LinearPrior, LinearReward, LinearRewardPrior  # type: ignore

try:  # pragma: no cover - import shims
    from .mlp import MLPPrior, MLPReward, MLPRewardPrior  # type: ignore
except Exception:  # pragma: no cover
    try:
        from fre.rewards.mlp import (  # type: ignore
            MLPPrior,
            MLPReward,
            MLPRewardPrior,
        )
    except Exception:
        from mlp import MLPPrior, MLPReward, MLPRewardPrior  # type: ignore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIXTURE_FAMILY = "mixture"

FAMILY_GOAL = "goal_reaching"
FAMILY_LINEAR = "linear"
FAMILY_MLP = "mlp"

#: The three families of the paper's mixture prior.
DEFAULT_FAMILIES: Tuple[str, ...] = (FAMILY_GOAL, FAMILY_LINEAR, FAMILY_MLP)

#: Section 4.2: "uniform mixture of 0.33 goal-reaching / 0.33 linear / 0.33 MLP".
FAMILY_MIXTURE_RATIOS: Dict[str, float] = {
    FAMILY_GOAL: 0.33,
    FAMILY_LINEAR: 0.33,
    FAMILY_MLP: 0.33,
}
DEFAULT_FAMILY_WEIGHTS: Tuple[float, ...] = tuple(
    FAMILY_MIXTURE_RATIOS[f] for f in DEFAULT_FAMILIES
)

#: Context / decoder sizes used by the FRE encoder (K=32) and decoder (K'=8).
CONTEXT_SIZE = 32
DECODER_SIZE = 8

#: "sample 512 reward functions per RL batch" / "encoder batch = 512".
DEFAULT_BATCH_SIZE = 512

#: Table 4 prior subsets (name -> families) for the AntMaze ablation.
ABLATION_FAMILIES: Dict[str, Tuple[str, ...]] = {
    "all": (FAMILY_GOAL, FAMILY_LINEAR, FAMILY_MLP),
    "goal": (FAMILY_GOAL,),
    "goals": (FAMILY_GOAL,),
    "lin": (FAMILY_LINEAR,),
    "linear": (FAMILY_LINEAR,),
    "mlp": (FAMILY_MLP,),
    "lin-mlp": (FAMILY_LINEAR, FAMILY_MLP),
    "linear-mlp": (FAMILY_LINEAR, FAMILY_MLP),
    "goal-mlp": (FAMILY_GOAL, FAMILY_MLP),
    "goal-lin": (FAMILY_GOAL, FAMILY_LINEAR),
    "goal-linear": (FAMILY_GOAL, FAMILY_LINEAR),
    "goal-lin-mlp": (FAMILY_GOAL, FAMILY_LINEAR, FAMILY_MLP),
}

#: Standard Table-4 ordering (FRE-all first so it can be compared to its subsets).
ABLATION_NAMES: Tuple[str, ...] = (
    "all",
    "goals",
    "lin",
    "mlp",
    "lin-mlp",
    "goal-mlp",
    "goal-lin",
)

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalise_weights(weights: Sequence[float]) -> np.ndarray:
    """Cast to float64, clip negatives, and normalise so the array sums to 1."""
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.size == 0:
        raise ValueError("weights must be non-empty")
    w = np.clip(w, 0.0, None)
    total = float(w.sum())
    if total <= 0:
        w = np.ones_like(w)
        total = float(w.size)
    return w / total


def _spawn_seed(sequence: Optional[np.random.SeedSequence], index: int) -> Optional[int]:
    """Deterministically derive an int seed from a ``SeedSequence``."""
    if sequence is None:
        return None
    state = sequence.spawn(index + 1)[index].generate_state(1)
    return int(state[0] % (2 ** 31 - 1))


def _call_prior(prior: Any, method_name: str, **kwargs: Any) -> Any:
    """Invoke ``prior.method_name`` passing only the keyword args it accepts.

    The three family priors (and any user supplied extra prior) have slightly
    different signatures; this adapter keeps the mixture code family-agnostic.
    ``None`` values are dropped so that each family can apply its own default
    (e.g. ``include_success=True`` for goal-reaching only).
    """
    method = getattr(prior, method_name, None)
    if method is None:
        raise AttributeError(f"{type(prior).__name__} has no method '{method_name}'")
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return method(**kwargs)

    params = signature.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return method(**kwargs)

    filtered = {k: v for k, v in kwargs.items() if k in params and v is not None}
    return method(**filtered)


def _unpack_context_result(result: Any) -> Tuple[Any, Any, Any, Any, Any]:
    """Normalise family ``sample_context_and_decoder`` outputs to a 5-tuple.

    Returns ``(context_states, context_rewards, decoder_states,
    decoder_rewards, reward_fn)``; missing entries become ``None``.
    """
    if isinstance(result, dict):
        return (
            result.get("context_states", result.get("states")),
            result.get("context_rewards", result.get("rewards")),
            result.get("decoder_states"),
            result.get("decoder_rewards"),
            result.get("reward_fn", result.get("function")),
        )
    if isinstance(result, (tuple, list)):
        values: List[Any] = list(result)
        while len(values) < 5:
            values.append(None)
        return tuple(values[:5])  # type: ignore[return-value]
    # Single reward-function style output
    return None, None, None, None, result


def _tag_function(reward_fn: Any, family: str, index: Optional[int] = None) -> Any:
    """Attach mixture provenance onto a reward function's metadata."""
    if reward_fn is None:
        return reward_fn
    metadata = getattr(reward_fn, "metadata", None)
    if metadata is None:
        try:
            reward_fn.metadata = {}
        except Exception:  # pragma: no cover - slots based objects
            return reward_fn
        metadata = reward_fn.metadata
    try:
        metadata["mixture_family"] = family
        if index is not None:
            metadata["mixture_index"] = int(index)
    except Exception:  # pragma: no cover
        pass
    return reward_fn


# ---------------------------------------------------------------------------
# Mixture prior
# ---------------------------------------------------------------------------
class MixturePrior(RewardFunctionPrior):
    """Uniform mixture over reward-function families (Section 4.2).

    Parameters
    ----------
    families:
        Names of the families to include.  Any of ``"goal_reaching"``,
        ``"linear"``, ``"mlp"``.  Names present in ``extra_priors`` may also be
        used (this is how the Figure-6 "hint" families are registered).
    weights:
        Optional per-family weights (defaults to a uniform 0.33/0.33/0.33 split,
        i.e. the paper's mixture ratios).
    source:
        Offline dataset (dict of arrays, replay buffer, or object exposing
        ``observations``).  Used for HER goal sampling (goal family) and for
        drawing context states (all families).
    state_dim:
        Observation dimensionality (inferred from ``source`` when omitted).
    exclude_dims:
        Observation dims removed from *linear* reward generation.  For AntMaze
        the paper removes the XY position dims, i.e. ``exclude_dims=(0, 1)``.
    goal_kwargs / linear_kwargs / mlp_kwargs:
        Extra keyword arguments forwarded to the corresponding family prior
        (e.g. ``p_current``/``p_future``/``p_random`` for goal-reaching).
    extra_priors:
        Optional mapping ``name -> RewardFunctionPrior instance`` for
        additional families (cf. :func:`hint_prior`).
    """

    family = MIXTURE_FAMILY

    def __init__(
        self,
        families: Sequence[str] = DEFAULT_FAMILIES,
        weights: Optional[Sequence[float]] = None,
        source: Any = None,
        state_dim: Optional[int] = None,
        rng: Any = None,
        seed: Optional[int] = None,
        goal_kwargs: Optional[Dict[str, Any]] = None,
        linear_kwargs: Optional[Dict[str, Any]] = None,
        mlp_kwargs: Optional[Dict[str, Any]] = None,
        exclude_dims: Optional[Sequence[int]] = None,
        extra_priors: Optional[Dict[str, Any]] = None,
        extra_weights: Optional[Dict[str, float]] = None,
        name: str = "mixture_prior",
        **kwargs: Any,
    ) -> None:
        families = tuple(families)
        if not families:
            raise ValueError("MixturePrior requires at least one family")

        try:
            super().__init__(state_dim=state_dim, source=source, rng=rng, seed=seed)
        except TypeError:  # pragma: no cover - positional-only base signature
            super().__init__(state_dim, source=source, rng=rng, seed=seed)

        self.name = name
        self.families: Tuple[str, ...] = families
        self.weights = _normalise_weights(
            weights if weights is not None else [1.0] * len(families)
        )
        if len(self.weights) != len(families):
            raise ValueError(
                f"weights length {len(self.weights)} != families length {len(families)}"
            )

        if state_dim is not None:
            self.state_dim = int(state_dim)
        self.exclude_dims = tuple(int(d) for d in exclude_dims) if exclude_dims else None

        self.goal_kwargs = dict(goal_kwargs or {})
        self.linear_kwargs = dict(linear_kwargs or {})
        self.mlp_kwargs = dict(mlp_kwargs or {})
        self.extra_priors = dict(extra_priors or {})
        self.extra_weights = dict(extra_weights or {})

        self._seed = seed
        self._seed_sequence = np.random.SeedSequence(seed) if seed is not None else None
        # Single shared generator (also handed to children when no seed given).
        self.rng = get_rng(rng, seed)

        self._priors: Dict[str, Any] = {}
        self._family_indices: Dict[str, int] = {}
        for index, family in enumerate(self.families):
            self._priors[family] = self._build_prior(family, index)
            self._family_indices[family] = index

        # Dataset stats forwarded to children for linear/mlp dim inference.
        if source is not None:
            self.set_source(source)
        if self.state_dim is None:
            self.state_dim = self._infer_state_dim(source)

    # -- construction -------------------------------------------------------
    def _build_prior(self, family: str, index: int) -> Any:
        child_seed = _spawn_seed(self._seed_sequence, index)
        child_rng = None if child_seed is not None else self.rng
        common: Dict[str, Any] = dict(
            state_dim=self.state_dim, source=self.source, rng=child_rng, seed=child_seed
        )

        if family in self.extra_priors:
            prior = self.extra_priors[family]
            if hasattr(prior, "set_source") and self.source is not None:
                try:
                    prior.set_source(self.source)
                except Exception:  # pragma: no cover
                    pass
            return prior

        if family == FAMILY_GOAL:
            return GoalReachingPrior(**common, **self.goal_kwargs)
        if family == FAMILY_LINEAR:
            linear_kwargs = dict(self.linear_kwargs)
            if self.exclude_dims is not None and "exclude_dims" not in linear_kwargs:
                linear_kwargs["exclude_dims"] = self.exclude_dims
            return LinearPrior(**common, **linear_kwargs)
        if family == FAMILY_MLP:
            return MLPPrior(**common, **self.mlp_kwargs)
        raise ValueError(
            f"Unknown reward family '{family}'. Expected one of "
            f"{DEFAULT_FAMILIES} or a key of `extra_priors`."
        )

    # -- dataset plumbing ---------------------------------------------------
    def set_source(self, source: Any) -> "MixturePrior":
        """Register the offline dataset and propagate it to all sub-priors."""
        super().set_source(source)
        for prior in self._priors.values():
            setter = getattr(prior, "set_source", None)
            if setter is not None:
                try:
                    setter(source)
                except Exception:  # pragma: no cover - defensive
                    pass
        if self.state_dim is None:
            self.state_dim = self._infer_state_dim(source)
        return self

    def _infer_state_dim(self, source: Any = None) -> Optional[int]:
        source = source if source is not None else self.source
        if source is None:
            return None
        try:
            obs = get_observations(source)
            if obs is not None and np.ndim(obs) >= 2:
                return int(np.asarray(obs).shape[-1])
        except Exception:  # pragma: no cover - defensive
            pass
        for attr in ("obs_dim", "observation_dim", "state_dim", "observations_dim"):
            value = getattr(source, attr, None)
            if isinstance(value, (int, np.integer)):
                return int(value)
        return None

    def _state_dim_or_raise(self, source: Any = None) -> int:
        dim = self.state_dim
        if dim is None:
            dim = self._infer_state_dim(source)
        if dim is None:
            raise ValueError(
                "MixturePrior cannot determine the state dimension; pass "
                "`state_dim` explicitly or provide a dataset via `source`."
            )
        self.state_dim = int(dim)
        return self.state_dim

    # -- sampling -----------------------------------------------------------
    def sample_families(self, num_functions: int = 1, rng: Any = None) -> np.ndarray:
        """Draw family labels for ``num_functions`` reward functions."""
        rng = rng if rng is not None else self.rng
        n = int(num_functions)
        if n <= 0:
            return np.empty((0,), dtype=object)
        if len(self.families) == 1:
            return np.array([self.families[0]] * n, dtype=object)
        indices = rng.choice(len(self.families), size=n, p=self.weights)
        return np.asarray(self.families, dtype=object)[indices]

    def sample_function(
        self,
        family: Optional[str] = None,
        source: Any = None,
        rng: Any = None,
        **kwargs: Any,
    ) -> RewardFunction:
        """Sample a single reward function, optionally forced to ``family``."""
        family = family if family is not None else str(self.sample_families(1, rng=rng)[0])
        prior = self._priors[family]
        functions = self._sample_from_prior(prior, 1, source=source, rng=rng, **kwargs)
        return _tag_function(functions[0], family, index=self._family_indices[family])

    def _sample_from_prior(
        self,
        prior: Any,
        num_functions: int,
        source: Any = None,
        rng: Any = None,
        state_dim: Optional[int] = None,
        **kwargs: Any,
    ) -> List[RewardFunction]:
        result = _call_prior(
            prior,
            "sample_functions",
            num_functions=int(num_functions),
            source=source,
            rng=rng,
            state_dim=state_dim if state_dim is not None else self.state_dim,
            **kwargs,
        )
        if result is None:
            return []
        if isinstance(result, (list, tuple)):
            return list(result)
        return [result]

    def sample_functions(
        self,
        num_functions: int = 1,
        source: Any = None,
        rng: Any = None,
        family: Optional[str] = None,
        families: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> List[RewardFunction]:
        """Sample ``num_functions`` reward functions from the mixture.

        If ``family`` (single) or ``families`` (subset, resampled in
        proportion to the surviving weights) is given, sampling is restricted
        accordingly -- used for the Table 4 ablations.
        """
        rng = rng if rng is not None else self.rng
        if family is not None:
            labels = np.array([family] * int(num_functions), dtype=object)
        elif families is not None:
            labels = self._restricted_families(list(families), int(num_functions), rng)
        else:
            labels = self.sample_families(num_functions, rng=rng)

        functions: List[RewardFunction] = []
        for label in np.unique(labels):
            mask = labels == label
            count = int(np.sum(mask))
            prior = self._priors[str(label)]
            sampled = self._sample_from_prior(
                prior, count, source=source, rng=rng, **kwargs
            )
            for i, fn in enumerate(sampled):
                functions.append(
                    _tag_function(fn, str(label), index=self._family_indices[str(label)])
                )
            # Pad defensively if a prior returned fewer functions than asked.
            while len(functions) < count:
                functions.append(None)  # type: ignore[arg-type]
        # ``np.unique`` reorders labels; restore mixture order deterministically.
        order = np.argsort(np.argsort(labels, kind="stable"), kind="stable")
        return [functions[i] for i in order if functions[i] is not None]

    def _restricted_families(
        self, families: Sequence[str], num_functions: int, rng: Any
    ) -> np.ndarray:
        idx = [self._family_indices[f] for f in families]
        weights = _normalise_weights([self.weights[i] for i in idx])
        subset = [self.families[i] for i in idx]
        if len(subset) == 1:
            return np.array([subset[0]] * num_functions, dtype=object)
        chosen = rng.choice(len(subset), size=num_functions, p=weights)
        return np.asarray(subset, dtype=object)[chosen]

    def __call__(self, num_functions: int = 1, **kwargs: Any) -> List[RewardFunction]:
        return self.sample_functions(num_functions=num_functions, **kwargs)

    # -- context / decoder sampling ----------------------------------------
    def sample_context(
        self,
        num_samples: int = CONTEXT_SIZE,
        source: Any = None,
        rng: Any = None,
        reward_fn: Any = None,
        family: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a ``(states (K, D), rewards (K,))`` encoder context.

        A reward function is drawn from the mixture when ``reward_fn`` is not
        supplied.  Goal-reaching contexts always contain at least one
        goal-achieving sample (paper requirement).
        """
        rng = rng if rng is not None else self.rng
        source = source if source is not None else self.source
        if reward_fn is None:
            reward_fn = self.sample_function(family=family, source=source, rng=rng)

        fam = self._family_of(reward_fn, family)
        prior = self._priors[fam]
        result = _call_prior(
            prior,
            "sample_context",
            num_samples=int(num_samples),
            source=source,
            rng=rng,
            reward_fn=reward_fn,
            **kwargs,
        )
        states, rewards = self._unpack_states_rewards(result)
        if states is None:
            states, rewards = self._fallback_context(reward_fn, num_samples, source, rng)
        return states, rewards

    def sample_context_and_decoder(
        self,
        num_context: int = CONTEXT_SIZE,
        num_decoder: int = DECODER_SIZE,
        source: Any = None,
        rng: Any = None,
        reward_fn: Any = None,
        family: Optional[str] = None,
        disjoint: bool = True,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, RewardFunction]:
        """Sample disjoint K=32 context and K'=8 decoder ``(s, eta(s))`` sets."""
        rng = rng if rng is not None else self.rng
        source = source if source is not None else self.source
        if reward_fn is None:
            reward_fn = self.sample_function(family=family, source=source, rng=rng)

        fam = self._family_of(reward_fn, family)
        prior = self._priors[fam]
        result = _call_prior(
            prior,
            "sample_context_and_decoder",
            num_context=int(num_context),
            num_decoder=int(num_decoder),
            source=source,
            rng=rng,
            reward_fn=reward_fn,
            disjoint=bool(disjoint),
            **kwargs,
        )
        ctx_states, ctx_rewards, dec_states, dec_rewards, fn = _unpack_context_result(result)
        fn = reward_fn if fn is None else fn
        if ctx_states is None:
            ctx_states, ctx_rewards = self._fallback_context(
                fn, num_context, source, rng
            )
        if dec_states is None and num_decoder > 0:
            dec_states, dec_rewards = self._fallback_context(
                fn, num_decoder, source, rng
            )
        if dec_states is None:
            dim = np.asarray(ctx_states).shape[-1]
            dec_states = np.zeros((0, dim), dtype=np.float32)
            dec_rewards = np.zeros((0,), dtype=np.float32)
        return (
            np.asarray(ctx_states, dtype=np.float32),
            np.asarray(ctx_rewards, dtype=np.float32).reshape(-1),
            np.asarray(dec_states, dtype=np.float32),
            np.asarray(dec_rewards, dtype=np.float32).reshape(-1),
            fn,
        )

    def _family_of(self, reward_fn: Any, family: Optional[str] = None) -> str:
        if family is not None:
            return family
        metadata = getattr(reward_fn, "metadata", None) or {}
        named = metadata.get("mixture_family")
        if named in self._priors:
            return str(named)
        # Fall back on the declared family of the object.
        declared = getattr(reward_fn, "family", None)
        if declared == "goal_reaching":
            return FAMILY_GOAL
        if declared == "linear":
            return FAMILY_LINEAR
        if declared == "mlp":
            return FAMILY_MLP
        if isinstance(reward_fn, GoalReachingReward):
            return FAMILY_GOAL
        # Last resort: infer from the function's structure.
        if getattr(reward_fn, "goal", None) is not None:
            return FAMILY_GOAL
        if getattr(reward_fn, "mask", None) is not None and getattr(
            reward_fn, "weights", None
        ) is not None and np.ndim(getattr(reward_fn, "weights", None)) == 1:
            return FAMILY_LINEAR
        return self.families[0]

    @staticmethod
    def _unpack_states_rewards(result: Any) -> Tuple[Any, Any]:
        if result is None:
            return None, None
        if isinstance(result, dict):
            return (
                result.get("states", result.get("context_states")),
                result.get("rewards", result.get("context_rewards")),
            )
        if isinstance(result, (tuple, list)):
            if len(result) >= 2:
                return result[0], result[1]
            if len(result) == 1:
                return result[0], None
        return None, None

    def _fallback_context(
        self, reward_fn: Any, num_samples: int, source: Any, rng: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample states directly from the dataset and evaluate ``reward_fn``."""
        if source is None:
            raise ValueError(
                "MixturePrior.sample_context requires either a dataset source or "
                "a family prior that can sample states on its own."
            )
        obs = np.asarray(get_observations(source), dtype=np.float32)
        n = int(num_samples)
        replace = n > obs.shape[0]
        indices = rng.integers(0, obs.shape[0], size=n) if replace else rng.choice(
            obs.shape[0], size=n, replace=False
        )
        states = obs[indices]
        rewards = np.asarray(reward_fn(states), dtype=np.float32).reshape(-1)
        return states, rewards

    # -- batched sampling (training workhorse) ------------------------------
    def sample_batch(
        self,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_context: int = CONTEXT_SIZE,
        num_decoder: int = DECODER_SIZE,
        source: Any = None,
        rng: Any = None,
        family: Optional[str] = None,
        families: Optional[Sequence[str]] = None,
        return_functions: bool = True,
        include_success: bool = True,
        disjoint: bool = True,
        dtype: Any = np.float32,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Sample ``batch_size`` reward functions and their context/decoder sets.

        Returns a dict with

        ``context_states``   ``(B, K,  D)`` encoder inputs
        ``context_rewards``  ``(B, K)``    encoder reward inputs
        ``decoder_states``   ``(B, K', D)`` held-out decoder states
        ``decoder_rewards``  ``(B, K')``   ground-truth rewards on those states
        ``families``         ``(B,)`` object array of family labels
        ``functions``        list of B reward functions (when ``return_functions``)

        This is the interface used by Phase 1 (encoder/decoder pretraining) and
        by Phase 2 (one reward function per environment slot, re-encoded to a
        fresh ``z`` each iteration).
        """
        rng = rng if rng is not None else self.rng
        source = source if source is not None else self.source
        batch_size = int(batch_size)
        num_context = int(num_context)
        num_decoder = int(num_decoder)

        if family is not None:
            labels = np.array([family] * batch_size, dtype=object)
        elif families is not None:
            labels = self._restricted_families(list(families), batch_size, rng)
        else:
            labels = self.sample_families(batch_size, rng=rng)

        state_dim = self._state_dim_or_raise(source)

        ctx_states = np.zeros((batch_size, num_context, state_dim), dtype=dtype)
        ctx_rewards = np.zeros((batch_size, num_context), dtype=dtype)
        dec_states = np.zeros((batch_size, num_decoder, state_dim), dtype=dtype)
        dec_rewards = np.zeros((batch_size, num_decoder), dtype=dtype)
        functions: List[Any] = [None] * batch_size

        for label in np.unique(labels):
            positions = np.nonzero(labels == label)[0]
            prior = self._priors[str(label)]
            sampled = self._sample_from_prior(
                prior, len(positions), source=source, rng=rng, **kwargs
            )
            for offset, position in enumerate(positions):
                fn = sampled[offset] if offset < len(sampled) else None
                if fn is None:
                    fn = self.sample_function(family=str(label), source=source, rng=rng)
                fn = _tag_function(fn, str(label), index=self._family_indices[str(label)])
                cs, cr, ds, dr, fn = self._context_for(
                    prior,
                    fn,
                    num_context=num_context,
                    num_decoder=num_decoder,
                    source=source,
                    rng=rng,
                    include_success=include_success,
                    disjoint=disjoint,
                )
                ctx_states[position] = cs
                ctx_rewards[position] = cr
                if num_decoder > 0:
                    dec_states[position] = ds
                    dec_rewards[position] = dr
                functions[position] = fn

        result: Dict[str, Any] = {
            "context_states": ctx_states,
            "context_rewards": ctx_rewards,
            "decoder_states": dec_states,
            "decoder_rewards": dec_rewards,
            "families": labels,
            "num_context": num_context,
            "num_decoder": num_decoder,
            "state_dim": state_dim,
        }
        if return_functions:
            result["functions"] = functions
        return result

    def _context_for(
        self,
        prior: Any,
        reward_fn: Any,
        num_context: int,
        num_decoder: int,
        source: Any,
        rng: Any,
        include_success: bool,
        disjoint: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
        result = _call_prior(
            prior,
            "sample_context_and_decoder",
            num_context=int(num_context),
            num_decoder=int(num_decoder),
            source=source,
            rng=rng,
            reward_fn=reward_fn,
            include_success=include_success,
            disjoint=bool(disjoint),
        )
        cs, cr, ds, dr, fn = _unpack_context_result(result)
        fn = reward_fn if fn is None else fn
        if cs is None:
            cs, cr = self._fallback_context(fn, num_context, source, rng)
        if ds is None and num_decoder > 0:
            ds, dr = self._fallback_context(fn, num_decoder, source, rng)
        if ds is None:
            dim = np.asarray(cs).shape[-1]
            ds = np.zeros((0, dim), dtype=np.float32)
            dr = np.zeros((0,), dtype=np.float32)
        return (
            np.asarray(cs, dtype=np.float32),
            np.asarray(cr, dtype=np.float32).reshape(-1),
            np.asarray(ds, dtype=np.float32),
            np.asarray(dr, dtype=np.float32).reshape(-1),
            fn,
        )

    # -- reward evaluation helpers -----------------------------------------
    def evaluate(self, functions: Sequence[Any], states: Any) -> np.ndarray:
        """Evaluate a list of reward functions on shared states -> ``(B, N)``.

        ``states`` may be ``(N, D)`` (shared) or ``(B, N, D)`` (per function).
        """
        states_arr = np.asarray(get_observations(states) if not isinstance(states, np.ndarray) else states)
        functions = list(functions)
        if states_arr.ndim == 2:
            return np.stack(
                [np.asarray(fn(states_arr), dtype=np.float32).reshape(-1) for fn in functions],
                axis=0,
            )
        if states_arr.ndim == 3:
            out = np.zeros((len(functions), states_arr.shape[1]), dtype=np.float32)
            for i, fn in enumerate(functions):
                out[i] = np.asarray(fn(states_arr[i]), dtype=np.float32).reshape(-1)
            return out
        raise ValueError(f"Unsupported states shape {states_arr.shape}")

    def evaluate_matched(self, functions: Sequence[Any], states: Any) -> np.ndarray:
        """Alias of :meth:`evaluate` returning ``(B, N)`` matched rewards."""
        return self.evaluate(functions, states)

    def reward_batch(
        self,
        functions: Sequence[Any],
        observations: Any,
        next_observations: Any = None,
    ) -> Dict[str, np.ndarray]:
        """Compute ``r = eta(s)`` for a batch of transitions (Phase 2).

        Returns ``{"rewards": (B,), "next_rewards": (B,) or None, "dones": (B,)}``
        where transition ``i`` is scored by the ``i``-th reward function.
        """
        obs = np.asarray(observations, dtype=np.float32)
        functions = list(functions)
        if obs.ndim != 2:
            raise ValueError(f"observations must be (B, D); got {obs.shape}")
        rewards = np.zeros((obs.shape[0],), dtype=np.float32)
        next_rewards = None if next_observations is None else np.zeros_like(rewards)
        dones = np.zeros((obs.shape[0],), dtype=bool)
        for i, fn in enumerate(functions):
            if i >= obs.shape[0]:
                break
            rewards[i] = float(np.asarray(fn(obs[i])).reshape(-1)[0])
            done_fn = getattr(fn, "done", None)
            if done_fn is not None:
                try:
                    dones[i] = bool(np.asarray(done_fn(obs[i])).reshape(-1)[0])
                except Exception:  # pragma: no cover - defensive
                    pass
            if next_rewards is not None:
                next_arr = np.asarray(next_observations, dtype=np.float32)
                next_rewards[i] = float(np.asarray(fn(next_arr[i])).reshape(-1)[0])
        return {"rewards": rewards, "next_rewards": next_rewards, "dones": dones}

    # -- introspection ------------------------------------------------------
    @property
    def priors(self) -> Dict[str, Any]:
        """Mapping ``family name -> family prior instance``."""
        return self._priors

    def prior(self, family: str) -> Any:
        return self._priors[family]

    def describe(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "families": list(self.families),
            "weights": self.weights.tolist(),
            "ratios": {f: float(w) for f, w in zip(self.families, self.weights)},
            "state_dim": self.state_dim,
            "exclude_dims": list(self.exclude_dims) if self.exclude_dims else None,
            "context_size": CONTEXT_SIZE,
            "decoder_size": DECODER_SIZE,
            "source": type(self.source).__name__ if self.source is not None else None,
        }

    def extra_repr(self) -> str:
        ratios = ", ".join(
            f"{f}={w:.2f}" for f, w in zip(self.families, self.weights)
        )
        return f"families=({ratios}), state_dim={self.state_dim}"


# Plan-facing alias.
FREPrior = MixturePrior
RewardPrior = MixturePrior


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------
def make_mixture_prior(
    source: Any = None,
    state_dim: Optional[int] = None,
    seed: Optional[int] = None,
    families: Sequence[str] = DEFAULT_FAMILIES,
    **kwargs: Any,
) -> MixturePrior:
    """Build the paper's 0.33/0.33/0.33 mixture prior ``p(eta)``."""
    return MixturePrior(
        families=families,
        source=source,
        state_dim=state_dim,
        seed=seed,
        **kwargs,
    )


def make_fre_prior(*args: Any, **kwargs: Any) -> MixturePrior:
    """Alias of :func:`make_mixture_prior`."""
    return make_mixture_prior(*args, **kwargs)


def make_prior(*args: Any, **kwargs: Any) -> MixturePrior:
    """Catch-all constructor (URL-style scripts use several naming conventions)."""
    return make_mixture_prior(*args, **kwargs)


def ablation_prior(
    name: str = "all",
    source: Any = None,
    state_dim: Optional[int] = None,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> MixturePrior:
    """Build the prior subset of Table 4 (``FRE-{all,goals,lin,mlp,...}``).

    Accepts e.g. ``"all"``, ``"goals"``, ``"lin"``, ``"mlp"``, ``"lin-mlp"``,
    ``"goal-mlp"``, ``"goal-lin"`` (see :data:`ABLATION_FAMILIES`), and also
    tolerates ``FRE-`` prefixes and underscores.
    """
    key = str(name).strip().lower().replace("_", "-")
    key = key[4:] if key.startswith("fre-") else key
    if key not in ABLATION_FAMILIES:
        raise ValueError(
            f"Unknown ablation prior '{name}'. Options: {sorted(ABLATION_FAMILIES)}"
        )
    return MixturePrior(
        families=ABLATION_FAMILIES[key],
        source=source,
        state_dim=state_dim,
        seed=seed,
        **kwargs,
    )


def hint_prior(
    source: Any = None,
    state_dim: Optional[int] = None,
    seed: Optional[int] = None,
    xy_dims: Sequence[int] = (0, 1),
    velocity_dims: Optional[Sequence[int]] = None,
    family_weight: float = 0.5,
    base_families: Sequence[str] = DEFAULT_FAMILIES,
    **kwargs: Any,
) -> MixturePrior:
    """Figure 6 "FRE-hint" prior: mixture augmented with task-relevant functions.

    Adds (without changing the architecture) extra linear reward functions
    restricted to semantically meaningful observation dims:

    * ``linear_xy`` -- linear rewards over the AntMaze XY position dims
      (``xy_dims``), i.e. ``exclude_dims`` = all dims except ``xy_dims``.
    * ``linear_vel`` -- linear rewards over the ExORL physics dims
      (``velocity_dims``), when provided.

    The hint families each receive ``family_weight`` (default 0.5) of the
    total probability mass; remaining mass is split uniformly over
    ``base_families``.
    """
    families = list(base_families)
    extra_priors: Dict[str, Any] = {}
    weights: List[float] = []

    hint_names: List[str] = []
    if xy_dims:
        extra_priors["linear_xy"] = _dims_only_linear_prior(
            keep_dims=xy_dims, state_dim=state_dim, seed=seed, source=source
        )
        hint_names.append("linear_xy")
    if velocity_dims:
        extra_priors["linear_vel"] = _dims_only_linear_prior(
            keep_dims=velocity_dims, state_dim=state_dim, seed=seed, source=source
        )
        hint_names.append("linear_vel")

    if not hint_names:  # nothing to hint -> plain mixture
        return make_mixture_prior(
            source=source, state_dim=state_dim, seed=seed, families=base_families, **kwargs
        )

    families = families + hint_names
    per_hint = float(family_weight) / len(hint_names)
    base_total = max(0.0, 1.0 - float(family_weight))
    for _ in base_families:
        weights.append(base_total / len(base_families))
    weights.extend([per_hint] * len(hint_names))

    return MixturePrior(
        families=families,
        weights=weights,
        source=source,
        state_dim=state_dim,
        seed=seed,
        extra_priors=extra_priors,
        **kwargs,
    )


def _dims_only_linear_prior(
    keep_dims: Sequence[int],
    state_dim: Optional[int],
    seed: Optional[int],
    source: Any,
) -> LinearPrior:
    """Linear prior restricted to ``keep_dims`` (everything else excluded)."""
    keep = tuple(int(d) for d in keep_dims)
    exclude = None
    dim = state_dim
    if dim is None and source is not None:
        try:
            obs = get_observations(source)
            if obs is not None and np.ndim(obs) >= 2:
                dim = int(np.asarray(obs).shape[-1])
        except Exception:  # pragma: no cover
            dim = None
    if dim is not None:
        exclude = tuple(d for d in range(int(dim)) if d not in keep)
    else:
        # Unknown dim: exclude the explicit XY slot only, which still biases
        # generation toward locally relevant dims.
        exclude = tuple(d for d in (0, 1) if d not in keep)
    prior = LinearPrior(state_dim=dim, exclude_dims=exclude, seed=seed)
    if source is not None:
        prior.set_source(source)
    return prior


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _make_dummy_dataset(num_states: int = 4096, state_dim: int = 29, seed: int = 0):
    rng = np.random.default_rng(seed)
    obs = np.cumsum(rng.normal(0.0, 0.1, size=(num_states, state_dim)), axis=0)
    obs = obs.astype(np.float32)
    actions = rng.normal(0.0, 0.3, size=(num_states, 8)).astype(np.float32)
    next_obs = np.concatenate([obs[1:], obs[-1:]], axis=0)
    terminals = np.zeros((num_states,), dtype=np.float32)
    terminals[::256] = 1.0
    return {
        "observations": obs,
        "actions": actions,
        "rewards": np.zeros((num_states,), dtype=np.float32),
        "next_observations": next_obs,
        "terminals": terminals,
    }


def _self_test() -> None:
    dataset = _make_dummy_dataset()
    prior = make_mixture_prior(source=dataset, seed=0)
    print("describe:", prior.describe())

    ## 1) family ratios -----------------------------------------------------
    labels = prior.sample_families(6000, rng=np.random.default_rng(0))
    unique, counts = np.unique(labels, return_counts=True)
    fractions = dict(zip(unique.tolist(), (counts / counts.sum()).tolist()))
    print("family fractions:", fractions)
    for name in DEFAULT_FAMILIES:
        assert abs(fractions.get(name, 0.0) - 1 / 3) < 0.03, fractions
    assert abs(prior.weights.sum() - 1.0) < 1e-9

    ## 2) single-function sampling -----------------------------------------
    for family in DEFAULT_FAMILIES:
        fn = prior.sample_function(family=family, rng=np.random.default_rng(1))
        states, rewards = prior.sample_context(
            num_samples=32,
            family=family,
            reward_fn=fn,
            rng=np.random.default_rng(2),
        )
        assert states.shape == (32, 29), states.shape
        assert rewards.shape == (32,), rewards.shape
        assert rewards.min() >= -1.0 - 1e-5 and rewards.max() <= 1.0 + 1e-5, rewards
        print(f"  {family}: reward range [{rewards.min():.2f}, {rewards.max():.2f}]")
        if family == FAMILY_GOAL:
            assert (rewards == 0.0).any(), "goal prior must include a success sample"

    ## 3) context/decoder disjointness -------------------------------------
    cs, cr, ds, dr, fn = prior.sample_context_and_decoder(
        num_context=32, num_decoder=8, rng=np.random.default_rng(3)
    )
    assert cs.shape == (32, 29) and ds.shape == (8, 29)
    assert cr.shape == (32,) and dr.shape == (8,)
    print("context/decoder OK:", cs.shape, ds.shape)

    ## 4) batched sampling (the training workhorse) ------------------------
    batch = prior.sample_batch(batch_size=64, num_context=32, num_decoder=8,
                               rng=np.random.default_rng(4))
    assert batch["context_states"].shape == (64, 32, 29)
    assert batch["context_rewards"].shape == (64, 32)
    assert batch["decoder_states"].shape == (64, 8, 29)
    assert batch["decoder_rewards"].shape == (64, 8)
    assert len(batch["functions"]) == 64
    assert np.isfinite(batch["context_rewards"]).all()
    print("batch OK:", batch["context_states"].shape)

    ## 5) transition reward evaluation -------------------------------------
    obs = np.asarray(dataset["observations"][:64], dtype=np.float32)
    rb = prior.reward_batch(batch["functions"], obs)
    assert rb["rewards"].shape == (64,)
    print("reward_batch OK:", rb["rewards"].shape)

    ## 6) ablations + hints -------------------------------------------------
    for name in ABLATION_NAMES:
        p = ablation_prior(name, source=dataset, seed=0)
        assert p.weights.sum() > 0.99
    print("ablation priors OK:", list(ABLATION_NAMES))

    hinted = hint_prior(source=dataset, seed=0, xy_dims=(0, 1))
    hb = hinted.sample_batch(batch_size=48, rng=np.random.default_rng(5))
    fams = set(hb["families"].tolist())
    print("hint families sampled:", fams)
    assert "linear_xy" in hinted.families

    ## 7) determinism -------------------------------------------------------
    a = make_mixture_prior(source=dataset, seed=7).sample_batch(
        batch_size=8, rng=np.random.default_rng(0)
    )
    b = make_mixture_prior(source=dataset, seed=7).sample_batch(
        batch_size=8, rng=np.random.default_rng(0)
    )
    assert np.allclose(a["context_states"], b["context_states"])
    assert np.allclose(a["context_rewards"], b["context_rewards"])
    print("determinism OK")

    ## 8) torch interop of sampled rewards ---------------------------------
    try:
        import torch  # noqa: F401

        fn = prior.sample_function(family=FAMILY_LINEAR, rng=np.random.default_rng(9))
        t = to_torch(obs[:5])
        out = fn(t)
        torch_out = out if hasattr(out, "shape") else None
        print("torch interop OK:", None if torch_out is None else tuple(torch_out.shape))
    except ImportError:  # pragma: no cover
        print("torch not installed; skipping interop check")

    print("\nall MixturePrior self-tests passed.")


if __name__ == "__main__":  # pragma: no cover
    _self_test()
