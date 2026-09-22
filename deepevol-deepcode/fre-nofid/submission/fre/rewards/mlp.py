"""Random MLP reward-function family for the FRE unsupervised reward prior p(eta).

This module implements the third component of the FRE reward-function prior
(Section 4.2 / Appendix B of "Zero-Shot Reinforcement Learning via Functional
Reward Encodings"): randomly initialised multi-layer perceptrons

    eta(s) = clip( W_L tanh( ... tanh( W_1 s + b_1 ) ... ) + b_L , -1, 1 )

where every weight matrix ``W_l`` is drawn i.i.d. from ``N(0, scale_l^2)`` and
the hidden activations are ``tanh``.  The paper specifies a ``(state_dim, 32, 1)``
architecture with hidden width 32, and the initialisation scale is divided by
the average layer dimension so that activations do not blow up or vanish.

The public interface deliberately mirrors :mod:`fre.rewards.linear` and
:mod:`fre.rewards.goal_reaching` (``sample_context`` /
``sample_context_and_decoder`` returning ``(ctx_states, ctx_rewards,
dec_states, dec_rewards, reward_fn)``) so that the 0.33/0.33/0.33 mixture prior
in :mod:`fre.rewards.prior` can drive every family uniformly.

Only numpy is required for sampling reward functions (they are cheap closed-form
functions of the state); torch is only touched in the self-test interop check.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# --- Robust base-class import (works as package member or standalone script) ---
try:  # pragma: no cover - import shim
    from .base import (  # type: ignore
        RewardFunction,
        RewardFunctionPrior,
        episode_boundaries,
        get_observations,
        get_rng,
        get_terminals,
        to_numpy,
        to_torch,
    )
except Exception:  # pragma: no cover - fallback for direct execution
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


__all__ = [
    "MLP_FAMILY",
    "DEFAULT_HIDDEN_SIZES",
    "DEFAULT_INIT_SCALE",
    "DEFAULT_ACTIVATION",
    "CONTEXT_SIZE",
    "DECODER_SIZE",
    "MLPReward",
    "MLPPrior",
    "MLPRewardPrior",
    "make_mlp_reward",
    "make_mlp_prior",
    "sample_mlp_context",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Family identifier used by the mixture prior.
MLP_FAMILY = "mlp"

#: Default hidden layer widths.  Appendix B specifies a single hidden layer of
#: width 32, i.e. an architecture of ``(state_dim -> 32 -> 1)``.
DEFAULT_HIDDEN_SIZES: Tuple[int, ...] = (32,)

#: Default standard deviation for the weight initialisation.  The actual scale
#: per layer is ``init_scale / sqrt(avg_layer_dim)`` (see :func:`layer_init_scale`).
DEFAULT_INIT_SCALE: float = 1.0

#: Hidden activation non-linearity (paper: tanh).
DEFAULT_ACTIVATION: str = "tanh"

#: FRE encoder context size (K = 32 (state, reward) pairs).
CONTEXT_SIZE = 32

#: FRE decoder size (K' = 8 held-out states, disjoint from the context).
DECODER_SIZE = 8

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------


def _tanh(x: np.ndarray) -> np.ndarray:
    """Numerically stable tanh (numpy already clamps large values safely)."""
    return np.tanh(x)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _gelu(x: np.ndarray) -> np.ndarray:
    # tanh approximation of the Gaussian error linear unit.
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))


def _identity(x: np.ndarray) -> np.ndarray:
    return x


_ACTIVATIONS = {
    "tanh": _tanh,
    "relu": _relu,
    "gelu": _gelu,
    "identity": _identity,
    "linear": _identity,
    "none": _identity,
}


def get_activation(name: str):
    """Return the numpy activation callable for ``name`` (case-insensitive)."""
    key = str(name).lower()
    if key not in _ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{name}'. Available: {sorted(_ACTIVATIONS)}"
        )
    return _ACTIVATIONS[key]


# ---------------------------------------------------------------------------
# Weight initialisation
# ---------------------------------------------------------------------------


def layer_init_scale(fan_in: int, fan_out: int, init_scale: float = DEFAULT_INIT_SCALE) -> float:
    """Return the per-layer weight std ``init_scale / sqrt(avg_layer_dim)``.

    Appendix B scales the standard deviation of the Gaussian weight init by the
    average layer dimension so that the reward field stays roughly unit-scaled
    regardless of the input dimensionality or network width.
    """
    avg_dim = 0.5 * float(fan_in + fan_out)
    if avg_dim <= 0.0:
        return float(init_scale)
    return float(init_scale) / np.sqrt(avg_dim)


def init_mlp_weights(
    state_dim: int,
    hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
    rng: Optional[np.random.Generator] = None,
    init_scale: float = DEFAULT_INIT_SCALE,
    output_scale: Optional[float] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Sample ``(weights, biases)`` for an MLP ``state_dim -> *hidden -> 1``.

    Every layer uses ``N(0, (init_scale / sqrt(avg_layer_dim))^2)`` weights and a
    zero bias, matching the paper's random-MLP reward family.
    """
    rng = get_rng(rng)
    dims = [int(state_dim)] + [int(h) for h in hidden_sizes] + [1]
    weights: List[np.ndarray] = []
    biases: List[np.ndarray] = []
    for i in range(len(dims) - 1):
        fan_in, fan_out = dims[i], dims[i + 1]
        # The scalar output head is scaled down so the pre-clip output range is
        # comparable to [-1, 1] rather than a sum of many hidden units.
        if i == len(dims) - 2 and output_scale is not None:
            scale = float(output_scale) / np.sqrt(max(fan_in, 1))
        else:
            scale = layer_init_scale(fan_in, fan_out, init_scale)
        w = rng.normal(loc=0.0, scale=scale, size=(fan_in, fan_out)).astype(np.float64)
        b = np.zeros((fan_out,), dtype=np.float64)
        weights.append(w)
        biases.append(b)
    return weights, biases


def forward_mlp(
    states: np.ndarray,
    weights: Sequence[np.ndarray],
    biases: Sequence[np.ndarray],
    activation: str = DEFAULT_ACTIVATION,
) -> np.ndarray:
    """Forward pass of the random MLP (pre-clip) over ``states`` shaped ``(..., D)``."""
    act = get_activation(activation)
    x = np.asarray(states, dtype=np.float64)
    last = len(weights) - 1
    for i, (w, b) in enumerate(zip(weights, biases)):
        x = x @ w + b
        if i < last:
            x = act(x)
    return x


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------


class MLPReward(RewardFunction):
    """A single random-MLP reward function ``eta(s) = clip(MLP(s), -1, 1)``.

    Parameters
    ----------
    weights, biases:
        List of weight matrices / bias vectors, applied in order.  The final
        layer must output a single scalar.
    state_dim:
        Observation dimensionality (inferred from the first weight matrix when
        omitted).
    hidden_sizes:
        Stored for bookkeeping / description only.
    activation:
        Hidden activation name (default ``"tanh"``).
    clip:
        Output clipping magnitude; the reward lies in ``[-clip, clip]`` with
        ``clip=1.0`` by default, matching the encoder's discretisation range.
    """

    family = MLP_FAMILY

    def __init__(
        self,
        weights: Sequence[np.ndarray],
        biases: Optional[Sequence[np.ndarray]] = None,
        state_dim: Optional[int] = None,
        hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
        activation: str = DEFAULT_ACTIVATION,
        init_scale: float = DEFAULT_INIT_SCALE,
        seed: Optional[int] = None,
        clip: float = 1.0,
        name: Optional[str] = None,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        normalise: bool = False,
        device: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if len(weights) == 0:
            raise ValueError("MLPReward requires at least one weight matrix.")
        self.weights = [np.asarray(w, dtype=np.float64) for w in weights]
        if biases is None:
            biases = [np.zeros((w.shape[1],), dtype=np.float64) for w in self.weights]
        self.biases = [np.asarray(b, dtype=np.float64) for b in biases]

        inferred = int(self.weights[0].shape[0])
        self.state_dim = int(state_dim) if state_dim is not None else inferred
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self.activation = str(activation)
        self.init_scale = float(init_scale)
        self.seed = seed

        meta = dict(metadata or {})
        meta.setdefault("family", MLP_FAMILY)
        meta.setdefault("hidden_sizes", self.hidden_sizes)
        meta.setdefault("activation", self.activation)

        if name is None:
            name = "mlp_reward"

        super().__init__(
            state_dim=self.state_dim,
            name=name,
            clip=clip,
            mean=mean,
            std=std,
            normalise=normalise,
            device=device,
            metadata=meta,
        )

    # -- core computation -------------------------------------------------
    def _compute(self, states: np.ndarray) -> np.ndarray:
        """Pre-clip forward pass; base class handles the ``[-clip, clip]`` clip."""
        out = forward_mlp(states, self.weights, self.biases, self.activation)
        return np.asarray(out, dtype=np.float64).reshape(-1)

    def raw_output(self, states: np.ndarray) -> np.ndarray:
        """Return the un-clipped MLP output (useful for diagnostics)."""
        return np.asarray(
            forward_mlp(states, self.weights, self.biases, self.activation),
            dtype=np.float64,
        ).reshape(-1)

    # -- introspection ----------------------------------------------------
    def describe(self) -> Dict[str, Any]:
        info = super().describe()
        info.update(
            {
                "family": MLP_FAMILY,
                "hidden_sizes": list(self.hidden_sizes),
                "activation": self.activation,
                "init_scale": self.init_scale,
                "num_layers": len(self.weights),
            }
        )
        return info

    def extra_repr(self) -> str:
        return (
            f"state_dim={self.state_dim}, hidden_sizes={list(self.hidden_sizes)}, "
            f"activation={self.activation}, clip={self.clip}"
        )


# ---------------------------------------------------------------------------
# Reward-function prior
# ---------------------------------------------------------------------------


class MLPPrior(RewardFunctionPrior):
    """Prior over random-MLP reward functions ``p(eta)``.

    Each sampled function is an independent MLP with ``tanh`` hidden activations
    and a ``(state_dim, *hidden_sizes, 1)`` weight tensor initialised from a
    scaled Gaussian.  Mirrors :class:`~fre.rewards.linear.LinearPrior` so the
    mixture prior can call it uniformly.
    """

    family = MLP_FAMILY

    def __init__(
        self,
        state_dim: Optional[int] = None,
        source: Optional[Any] = None,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
        hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
        activation: str = DEFAULT_ACTIVATION,
        init_scale: float = DEFAULT_INIT_SCALE,
        output_scale: Optional[float] = 1.0,
        clip: float = 1.0,
        name: str = "mlp_prior",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            state_dim=state_dim,
            source=source,
            rng=rng,
            seed=seed,
            clip=clip,
            name=name,
            **kwargs,
        )
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self.activation = str(activation)
        self.init_scale = float(init_scale)
        self.output_scale = None if output_scale is None else float(output_scale)
        self._generator = rng if rng is not None else get_rng(seed=seed)
        self._observations_cache: Optional[np.ndarray] = None
        self._terminals_cache: Optional[np.ndarray] = None

    # -- dataset plumbing -------------------------------------------------
    def set_source(self, source: Any) -> "MLPPrior":
        super().set_source(source)
        self._observations_cache = None
        self._terminals_cache = None
        if self.state_dim is None:
            self.state_dim = _infer_state_dim(source)
        return self

    @property
    def observations_cached(self) -> Optional[np.ndarray]:
        if self._observations_cache is None and self.source is not None:
            try:
                self._observations_cache = get_observations(self.source)
            except Exception:
                self._observations_cache = None
        return self._observations_cache

    @property
    def terminals_cached(self) -> Optional[np.ndarray]:
        if self._terminals_cache is None and self.source is not None:
            try:
                self._terminals_cache = get_terminals(self.source)
            except Exception:
                self._terminals_cache = None
        return self._terminals_cache

    def _resolve_state_dim(self, state_dim: Optional[int] = None) -> int:
        dim = state_dim or self.state_dim
        if dim is None:
            obs = self.observations_cached
            if obs is not None:
                dim = int(obs.shape[-1])
        if dim is None:
            raise ValueError(
                "MLPPrior could not determine state_dim; pass it explicitly or "
                "attach a dataset via set_source()."
            )
        self.state_dim = int(dim)
        return int(dim)

    # -- sampling ---------------------------------------------------------
    def sample_weights(
        self,
        rng: Optional[np.random.Generator] = None,
        state_dim: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Sample one MLP's weights/biases."""
        rng = get_rng(rng if rng is not None else self._generator)
        dim = self._resolve_state_dim(state_dim)
        return init_mlp_weights(
            dim,
            hidden_sizes=self.hidden_sizes,
            rng=rng,
            init_scale=self.init_scale,
            output_scale=self.output_scale,
        )

    def sample_function(
        self,
        rng: Optional[np.random.Generator] = None,
        state_dim: Optional[int] = None,
        weights: Optional[Sequence[np.ndarray]] = None,
        biases: Optional[Sequence[np.ndarray]] = None,
        **fn_kwargs: Any,
    ) -> MLPReward:
        """Build a single :class:`MLPReward` (sampling weights when not given)."""
        rng = get_rng(rng if rng is not None else self._generator)
        dim = self._resolve_state_dim(state_dim)
        if weights is None:
            weights, biases = self.sample_weights(rng=rng, state_dim=dim)
        return MLPReward(
            weights=weights,
            biases=biases,
            state_dim=dim,
            hidden_sizes=self.hidden_sizes,
            activation=self.activation,
            init_scale=self.init_scale,
            clip=fn_kwargs.pop("clip", self.clip),
            **fn_kwargs,
        )

    def sample_functions(
        self,
        num_functions: int = 1,
        source: Optional[Any] = None,
        rng: Optional[np.random.Generator] = None,
        state_dim: Optional[int] = None,
        **kwargs: Any,
    ) -> List[MLPReward]:
        """Sample ``num_functions`` independent random-MLP reward functions."""
        if source is not None:
            self.set_source(source)
        rng = get_rng(rng if rng is not None else self._generator)
        dim = self._resolve_state_dim(state_dim)
        return [
            self.sample_function(rng=rng, state_dim=dim, **kwargs)
            for _ in range(int(num_functions))
        ]

    def __call__(self, num_functions: int = 1, **kwargs: Any) -> List[MLPReward]:
        return self.sample_functions(num_functions=num_functions, **kwargs)

    # -- state / context sampling ----------------------------------------
    def sample_states(
        self,
        num_states: int,
        rng: Optional[np.random.Generator] = None,
        source: Optional[Any] = None,
        indices: Optional[Sequence[int]] = None,
        replace: bool = True,
    ) -> np.ndarray:
        """Sample ``(num_states, D)`` states from the dataset (or use indices)."""
        rng = get_rng(rng if rng is not None else self._generator)
        if source is not None:
            self.set_source(source)
        obs = self.observations_cached
        if obs is None:
            raise ValueError("MLPPrior.sample_states requires a source dataset.")
        n = len(obs)
        num_states = int(num_states)
        if indices is not None:
            idx = np.asarray(indices, dtype=np.int64).reshape(-1)
            if idx.size == 0:
                raise ValueError("Received an empty index array.")
            if idx.size < num_states:
                if replace:
                    extra = rng.choice(n, size=num_states - idx.size, replace=True)
                    idx = np.concatenate([idx, extra])
                else:
                    num_states = idx.size
            return obs[idx[:num_states]].astype(np.float32)
        if replace:
            idx = rng.choice(n, size=num_states, replace=True)
        else:
            num_states = min(num_states, n)
            idx = rng.choice(n, size=num_states, replace=False)
        return obs[idx].astype(np.float32)

    def sample_context(
        self,
        num_samples: int = CONTEXT_SIZE,
        source: Optional[Any] = None,
        rng: Optional[np.random.Generator] = None,
        reward_fn: Optional[MLPReward] = None,
        indices: Optional[Sequence[int]] = None,
        shuffle: bool = True,
        **fn_kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray, MLPReward]:
        """Return ``(states (K,D), rewards (K,), reward_fn)`` for one MLP task."""
        rng = get_rng(rng if rng is not None else self._generator)
        if source is not None:
            self.set_source(source)
        if reward_fn is None:
            reward_fn = self.sample_function(rng=rng, **fn_kwargs)
        states = self.sample_states(num_samples, rng=rng, indices=indices)
        rewards = reward_fn.compute_numpy(states).astype(np.float32)
        if shuffle:
            perm = rng.permutation(len(states))
            states = states[perm]
            rewards = rewards[perm]
        return states, rewards, reward_fn

    def sample_context_and_decoder(
        self,
        num_context: int = CONTEXT_SIZE,
        num_decoder: int = DECODER_SIZE,
        source: Optional[Any] = None,
        rng: Optional[np.random.Generator] = None,
        reward_fn: Optional[MLPReward] = None,
        indices: Optional[Sequence[int]] = None,
        disjoint: bool = True,
        **fn_kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MLPReward]:
        """Return context (K) and disjoint decoder (K') (state, reward) sets.

        Mirrors the linear / goal-reaching families: ``(ctx_states, ctx_rewards,
        dec_states, dec_rewards, reward_fn)``.
        """
        rng = get_rng(rng if rng is not None else self._generator)
        if source is not None:
            self.set_source(source)
        if reward_fn is None:
            reward_fn = self.sample_function(rng=rng, **fn_kwargs)

        total = int(num_context) + int(num_decoder)
        obs = self.observations_cached
        if disjoint and obs is not None:
            n = len(obs)
            if indices is None:
                take = min(total, n)
                chosen = rng.choice(n, size=take, replace=False)
                while chosen.size < total and n > 0:
                    extra = rng.choice(n, size=total - chosen.size, replace=False)
                    chosen = np.concatenate([chosen, extra])
            else:
                base = np.asarray(indices, dtype=np.int64).reshape(-1)
                chosen = base
                if chosen.size < total:
                    extra = rng.choice(n, size=total - chosen.size, replace=False)
                    chosen = np.concatenate([chosen, extra])
            ctx_idx = chosen[:num_context]
            dec_idx = chosen[num_context:total]
            ctx_states = obs[ctx_idx].astype(np.float32)
            dec_states = obs[dec_idx].astype(np.float32)
        else:
            ctx_states = self.sample_states(num_context, rng=rng, indices=indices)
            dec_states = self.sample_states(num_decoder, rng=rng, indices=None)

        ctx_rewards = reward_fn.compute_numpy(ctx_states).astype(np.float32)
        dec_rewards = reward_fn.compute_numpy(dec_states).astype(np.float32)
        return ctx_states, ctx_rewards, dec_states, dec_rewards, reward_fn

    # -- introspection ----------------------------------------------------
    def describe(self) -> Dict[str, Any]:
        info = super().describe()
        info.update(
            {
                "family": MLP_FAMILY,
                "hidden_sizes": list(self.hidden_sizes),
                "activation": self.activation,
                "init_scale": self.init_scale,
            }
        )
        return info

    def extra_repr(self) -> str:
        return (
            f"state_dim={self.state_dim}, hidden_sizes={list(self.hidden_sizes)}, "
            f"activation={self.activation}, init_scale={self.init_scale}"
        )


#: Plan-facing alias.
MLPRewardPrior = MLPPrior


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def make_mlp_reward(
    state_dim: Optional[int] = None,
    weights: Optional[Sequence[np.ndarray]] = None,
    biases: Optional[Sequence[np.ndarray]] = None,
    hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
    activation: str = DEFAULT_ACTIVATION,
    init_scale: float = DEFAULT_INIT_SCALE,
    output_scale: Optional[float] = 1.0,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> MLPReward:
    """Construct a :class:`MLPReward`, sampling weights when none are supplied."""
    if weights is None:
        if state_dim is None:
            raise ValueError("make_mlp_reward needs either weights or state_dim.")
        rng = get_rng(seed=seed)
        weights, biases = init_mlp_weights(
            state_dim,
            hidden_sizes=hidden_sizes,
            rng=rng,
            init_scale=init_scale,
            output_scale=output_scale,
        )
    return MLPReward(
        weights=weights,
        biases=biases,
        state_dim=state_dim,
        hidden_sizes=hidden_sizes,
        activation=activation,
        init_scale=init_scale,
        seed=seed,
        **kwargs,
    )


def make_mlp_prior(
    source: Optional[Any] = None,
    state_dim: Optional[int] = None,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> MLPPrior:
    """Convenience factory for :class:`MLPPrior`."""
    return MLPPrior(state_dim=state_dim, source=source, seed=seed, **kwargs)


def sample_mlp_context(
    reward_fn: MLPReward,
    state_pool: Any,
    num_samples: int = CONTEXT_SIZE,
    rng: Optional[np.random.Generator] = None,
    replace: bool = True,
    shuffle: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(states (K,D), rewards (K,))`` for ``reward_fn`` on ``state_pool``.

    Mirrors :func:`fre.rewards.linear.sample_linear_context`.  ``state_pool`` may
    be an array or a replay buffer / dataset dict consumed by ``get_observations``.
    """
    rng = get_rng(rng)
    obs = get_observations(state_pool)
    n = len(obs)
    k = int(num_samples)
    if k <= 0:
        return np.zeros((0, obs.shape[-1]), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    if replace or k >= n:
        idx = rng.choice(n, size=k, replace=True)
    else:
        idx = rng.choice(n, size=k, replace=False)
    states = obs[idx].astype(np.float32)
    rewards = reward_fn.compute_numpy(states).astype(np.float32)
    if shuffle:
        perm = rng.permutation(len(states))
        states = states[perm]
        rewards = rewards[perm]
    return states, rewards


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _infer_state_dim(source: Any) -> Optional[int]:
    """Best-effort observation-dim inference from a dataset / buffer."""
    if source is None:
        return None
    try:
        obs = get_observations(source)
        return int(np.asarray(obs).shape[-1])
    except Exception:
        pass
    for attr in ("obs_dim", "observation_dim", "state_dim"):
        val = getattr(source, attr, None)
        if val is not None:
            try:
                return int(val)
            except Exception:
                continue
    return None


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(0)
    state_dim = 29
    obs = rng.normal(size=(1000, state_dim)).astype(np.float32)

    prior = make_mlp_prior(source={"observations": obs}, state_dim=state_dim, seed=0)

    # 1) A sampled reward function produces scalar outputs in [-1, 1].
    fn = prior.sample_function()
    r = fn.compute_numpy(obs[:200])
    assert r.shape == (200,), r.shape
    assert np.all(r >= -1.0 - 1e-6) and np.all(r <= 1.0 + 1e-6), (r.min(), r.max())
    print(f"[ok] reward range: [{r.min():+.3f}, {r.max():+.3f}]")

    # 2) Repeated calls return independent functions (different weights).
    fn2 = prior.sample_function()
    diff = np.abs(fn.weights[0] - fn2.weights[0]).mean()
    assert diff > 0, "sampled functions should differ"
    print(f"[ok] sampled functions differ (mean |dW| = {diff:.4f})")

    # 3) Context + disjoint decoder pairs.
    cs, cr, ds, dr, fn3 = prior.sample_context_and_decoder(32, 8)
    assert cs.shape == (32, state_dim) and cr.shape == (32,), (cs.shape, cr.shape)
    assert ds.shape == (8, state_dim) and dr.shape == (8,), (ds.shape, dr.shape)
    assert np.all(cr >= -1.0 - 1e-6) and np.all(cr <= 1.0 + 1e-6)
    print(f"[ok] context {cs.shape}, decoder {ds.shape}")

    # 4) torch interop through the base RewardFunction __call__.
    try:
        import torch

        t = torch.as_tensor(obs[:16], dtype=torch.float32)
        out = fn(t)
        assert out.shape == (16,), out.shape
        print(f"[ok] torch interop: {tuple(out.shape)}")
    except Exception as exc:  # pragma: no cover
        print(f"[skip] torch interop ({exc})")

    # 5) Mixture-style factory sweep across several functions.
    fns = prior.sample_functions(5)
    assert len(fns) == 5
    print("[ok] sampled 5 independent MLP reward functions")
    print("MLP prior self-test passed.")
