"""Random MLP reward functions for the FRE prior reward distribution.

Paper references
----------------
Section 4.2 ("Random Functions as a Prior Reward Distribution"):

    "The particular mixture we use consists of random singleton functions
    (corresponding to "goal reaching" rewards), random neural networks (MLPs with two
    linear layers), and random linear functions (corresponding to "MLPs" with one
    linear layer). This provides both a degree of structure and a mixture of high
    frequency (singletons) and low frequency (linear) functions, with the MLPs serving
    as an intermediate function complexity. A uniform mixture of the three function
    classes are used during training."

Appendix B ("Training Details"):

    "Random MLP functions are generated using a neural network of size
    (state_dim, 32, 1). Parameters are sampled using a normal distribution scaled by the
    average dimension of the layer. A tanh activation is used between the two layers.
    The final output of the neural network is clipped between -1 and 1."

So a sampled reward function is

    eta(s) = clip( W2 . tanh(W1 s + b1) + b2 , -1, 1 )

with ``W1`` of shape ``(state_dim, 32)``, ``b1`` of shape ``(32,)``, ``W2`` of shape
``(32, 1)`` and ``b2`` of shape ``(1,)``.  Weights are drawn from a zero-mean Gaussian
whose standard deviation is ``1 / sqrt(average dimension of the layer)`` (the paper's
"normal distribution scaled by the average dimension of the layer").

The clipping to ``[-1, 1]`` makes the reward range of this function class exactly
``[-1, 1]``, which is what the reward discretizer (``fre.utils.reward_discretize``)
needs in order to rescale rewards into ``[0, 1]`` before binning into 32 embeddings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - normal package import path
    from fre.reward_priors.goal_reaching import (
        RewardFunction,
        RewardPrior,
        as_numpy_2d,
    )
except ImportError:  # pragma: no cover - allows running this module standalone
    from goal_reaching import (  # type: ignore
        RewardFunction,
        RewardPrior,
        as_numpy_2d,
    )


# --------------------------------------------------------------------------------------
# Constants (Appendix B / Table 3)
# --------------------------------------------------------------------------------------
#: Hidden width of the random MLP prior (paper: network of size (state_dim, 32, 1)).
MLP_HIDDEN_DIM: int = 32
#: The final output of the network is clipped between -1 and 1 (Appendix B).
MLP_OUTPUT_CLIP: float = 1.0


# --------------------------------------------------------------------------------------
# Parameter container
# --------------------------------------------------------------------------------------
class MLPParameters:
    """Container for the weights of a two-layer random MLP ``(state_dim, 32, 1)``.

    Kept as a plain class (rather than a dataclass field on the reward function) so that
    the reward-function dataclass can be constructed / copied without numpy-array
    equality or default-value pitfalls.
    """

    __slots__ = ("w1", "b1", "w2", "b2", "hidden_dim", "state_dim")

    def __init__(
        self,
        w1: np.ndarray,
        b1: np.ndarray,
        w2: np.ndarray,
        b2: np.ndarray,
    ) -> None:
        self.w1 = np.asarray(w1, dtype=np.float32)
        self.b1 = np.asarray(b1, dtype=np.float32).reshape(-1)
        self.w2 = np.asarray(w2, dtype=np.float32)
        self.b2 = np.asarray(b2, dtype=np.float32).reshape(-1)
        self.state_dim = int(self.w1.shape[0])
        self.hidden_dim = int(self.w1.shape[1])

    def forward(self, states: np.ndarray) -> np.ndarray:
        """Raw (un-clipped) network output of shape ``(N,)``."""
        h = np.tanh(states @ self.w1 + self.b1)
        return (h @ self.w2 + self.b2).reshape(-1)

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.w1, self.b1, self.w2, self.b2

    def copy(self) -> "MLPParameters":
        return MLPParameters(self.w1.copy(), self.b1.copy(), self.w2.copy(), self.b2.copy())

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"MLPParameters(state_dim={self.state_dim}, hidden_dim={self.hidden_dim}, "
            f"layers=[{self.state_dim},{self.hidden_dim},1])"
        )


def sample_mlp_parameters(
    state_dim: int,
    hidden_dim: int = MLP_HIDDEN_DIM,
    rng: Optional[np.random.Generator] = None,
    scale_mode: str = "average",
) -> MLPParameters:
    """Sample weights for a random ``(state_dim, hidden_dim, 1)`` MLP.

    "Parameters are sampled using a normal distribution scaled by the average dimension of
    the layer" (Appendix B).  For a layer with ``fan_in`` inputs and ``fan_out`` outputs
    we therefore use

        std = 1 / sqrt((fan_in + fan_out) / 2)

    i.e. the standard deviation is the inverse square root of the *average* of the
    layer's two dimensions.  ``scale_mode="average"`` implements exactly that; the
    alternatives (``"fan_in"`` / ``"fan_out"``) are provided only for ablations.
    """
    if state_dim <= 0:
        raise ValueError(f"state_dim must be positive, got {state_dim}")
    if hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")

    rng = np.random.default_rng() if rng is None else rng

    def layer_scale(fan_in: int, fan_out: int) -> float:
        if scale_mode == "average":
            denom = 0.5 * (fan_in + fan_out)
        elif scale_mode == "fan_in":
            denom = float(fan_in)
        elif scale_mode == "fan_out":
            denom = float(fan_out)
        else:
            raise ValueError(f"unknown scale_mode: {scale_mode!r}")
        return float(1.0 / math.sqrt(max(denom, 1e-8)))

    hidden_shape = (int(state_dim), int(hidden_dim))
    out_shape = (int(hidden_dim), 1)
    w1 = rng.normal(0.0, layer_scale(*hidden_shape), size=hidden_shape)
    b1 = rng.normal(0.0, layer_scale(*hidden_shape), size=(int(hidden_dim),))
    w2 = rng.normal(0.0, layer_scale(*out_shape), size=out_shape)
    b2 = rng.normal(0.0, layer_scale(*out_shape), size=(1,))
    return MLPParameters(w1, b1, w2, b2)


# --------------------------------------------------------------------------------------
# Reward function: one sampled random MLP
# --------------------------------------------------------------------------------------
@dataclass
class RandomMLPReward(RewardFunction):
    """A single random MLP reward function ``eta(s) = clip(MLP(s), -1, 1)``.

    The network is a two-layer MLP ``(state_dim, 32, 1)`` with a ``tanh`` nonlinearity
    between the two linear layers (Appendix B).
    """

    params: Any = None
    hidden_dim: int = MLP_HIDDEN_DIM
    output_clip: float = MLP_OUTPUT_CLIP
    state_mean: Optional[np.ndarray] = None
    state_std: Optional[np.ndarray] = None
    normalize_states: bool = False
    name: str = "mlp"

    # ---- construction -----------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.params is None:
            raise ValueError("RandomMLPReward requires `params` (see sample_mlp_parameters)")
        if not isinstance(self.params, MLPParameters):
            # Accept a tuple/list/dict of arrays for convenience.
            if isinstance(self.params, dict):
                self.params = MLPParameters(
                    self.params["w1"], self.params["b1"], self.params["w2"], self.params["b2"]
                )
            else:
                w1, b1, w2, b2 = self.params
                self.params = MLPParameters(w1, b1, w2, b2)

        self.hidden_dim = int(self.params.hidden_dim)
        if self.state_mean is not None:
            self.state_mean = np.asarray(self.state_mean, dtype=np.float32).reshape(-1)
        if self.state_std is not None:
            self.state_std = np.asarray(self.state_std, dtype=np.float32).reshape(-1)

        # Output is clipped, so the reward range of this function class is exactly [-1, 1].
        self.reward_min = float(-abs(self.output_clip))
        self.reward_max = float(abs(self.output_clip))

    # ---- helpers ----------------------------------------------------------------------
    @property
    def state_dim(self) -> int:
        return int(self.params.state_dim)

    def _prepare_states(self, states: Any) -> np.ndarray:
        arr = as_numpy_2d(states)
        if self.normalize_states and self.state_std is not None:
            mean = self.state_mean if self.state_mean is not None else 0.0
            arr = (arr - mean) / np.maximum(self.state_std, 1e-6)
        return arr

    # ---- RewardFunction API -----------------------------------------------------------
    def _rewards_and_dones(self, states: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(rewards, dones)`` for the given states.

        Random MLP rewards are *not* terminal: the ``done`` mask is always ``False``
        (the paper only defines a done mask for goal-reaching / singleton functions).
        """
        arr = self._prepare_states(states)
        raw = self.params.forward(arr)
        rewards = np.clip(raw, -abs(self.output_clip), abs(self.output_clip)).astype(np.float32)
        dones = np.zeros(rewards.shape, dtype=bool)
        return rewards, dones

    def __call__(self, states: Any) -> np.ndarray:
        return self._rewards_and_dones(states)[0]

    def rewards(self, states: Any) -> np.ndarray:
        """Alias of ``__call__`` (explicit name for readability at call sites)."""
        return self._rewards_and_dones(states)[0]

    def label(
        self,
        states: Any,
        ensure_goal: bool = False,
        rng: Optional[np.random.Generator] = None,
        in_place: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Label states with this reward function; returns ``(rewards, dones)``.

        ``ensure_goal`` is a no-op for non-singleton reward functions: only goal-reaching
        priors need to guarantee that the goal state appears among the encoding samples
        (Appendix B).
        """
        arr = states
        if ensure_goal:
            arr = self.ensure_goal_in_set(states, rng=rng, in_place=False)
        return self._rewards_and_dones(arr)

    def ensure_goal_in_set(
        self,
        states: Any,
        rng: Optional[np.random.Generator] = None,
        in_place: bool = True,
    ) -> Any:
        """No-op: a random MLP has no distinguished "goal" state."""
        return states

    def reward_bounds(self, states: Any = None) -> Tuple[float, float]:
        """Exact (and state independent) bounds of this function class: ``[-1, 1]``."""
        return self.reward_min, self.reward_max

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "random_mlp",
            "name": self.name,
            "state_dim": self.state_dim,
            "hidden_dim": self.hidden_dim,
            "layers": [self.state_dim, self.hidden_dim, 1],
            "activation": "tanh",
            "output_clip": float(abs(self.output_clip)),
            "reward_min": self.reward_min,
            "reward_max": self.reward_max,
            "normalize_states": bool(self.normalize_states),
        }


# --------------------------------------------------------------------------------------
# Prior distribution over random MLPs
# --------------------------------------------------------------------------------------
@dataclass
class RandomMLPPrior(RewardPrior):
    """``p(eta)`` over random two-layer MLP reward functions.

    One third of the FRE prior mixture (Section 4.2, Table 3: "Ratio of Randomm MLP
    Rewards = 0.33").
    """

    state_dim: int = 0
    hidden_dim: int = MLP_HIDDEN_DIM
    output_clip: float = MLP_OUTPUT_CLIP
    state_mean: Optional[np.ndarray] = None
    state_std: Optional[np.ndarray] = None
    normalize_states: bool = False
    scale_mode: str = "average"
    seed: int = 0
    name: str = "mlp"

    def __post_init__(self) -> None:
        if self.state_dim is None or int(self.state_dim) <= 0:
            raise ValueError(
                "RandomMLPPrior requires a positive `state_dim` (infer it from the "
                "replay buffer's obs_dim, or pass it explicitly)."
            )
        self.state_dim = int(self.state_dim)
        self.hidden_dim = int(self.hidden_dim)
        self._rng = np.random.default_rng(self.seed)
        if self.state_mean is not None:
            self.state_mean = np.asarray(self.state_mean, dtype=np.float32).reshape(-1)
        if self.state_std is not None:
            self.state_std = np.asarray(self.state_std, dtype=np.float32).reshape(-1)

    def sample(self, rng: Optional[np.random.Generator] = None) -> RandomMLPReward:
        """Draw a fresh random MLP reward function."""
        rng = self._rng if rng is None else rng
        params = sample_mlp_parameters(
            state_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
            rng=rng,
            scale_mode=self.scale_mode,
        )
        return RandomMLPReward(
            params=params,
            hidden_dim=self.hidden_dim,
            output_clip=self.output_clip,
            state_mean=self.state_mean,
            state_std=self.state_std,
            normalize_states=self.normalize_states,
            name=self.name,
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "type": "random_mlp_prior",
            "name": self.name,
            "state_dim": self.state_dim,
            "hidden_dim": self.hidden_dim,
            "layers": [self.state_dim, self.hidden_dim, 1],
            "activation": "tanh",
            "output_clip": float(abs(self.output_clip)),
            "param_init": "normal(0, 1/sqrt(mean(fan_in, fan_out)))",
            "scale_mode": self.scale_mode,
            "seed": int(self.seed),
        }


# --------------------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------------------
def make_mlp_prior(
    replay_buffer: Any = None,
    state_dim: Optional[int] = None,
    hidden_dim: int = MLP_HIDDEN_DIM,
    seed: int = 0,
    name: str = "mlp",
    **kwargs: Any,
) -> RandomMLPPrior:
    """Build a :class:`RandomMLPPrior`, inferring ``state_dim`` from a replay buffer.

    Mirrors ``make_linear_prior``: if ``state_dim`` is not given we take
    ``replay_buffer.obs_dim``.  Optional state statistics from the buffer are used only
    to optionally center/scale inputs when ``normalize_states=True`` (a convenience
    default; the paper does not specify input normalization for this prior class).
    """
    state_mean = kwargs.pop("state_mean", None)
    state_std = kwargs.pop("state_std", None)

    if state_dim is None and replay_buffer is not None:
        state_dim = getattr(replay_buffer, "obs_dim", None)
    if state_dim is None:
        raise ValueError(
            "make_mlp_prior could not resolve `state_dim`: pass it explicitly or provide "
            "a replay buffer exposing `obs_dim`."
        )

    if replay_buffer is not None and (state_mean is None or state_std is None):
        stats_fn = getattr(replay_buffer, "state_statistics", None)
        if callable(stats_fn):
            try:
                mean, std = stats_fn()
                state_mean = np.asarray(mean, dtype=np.float32) if state_mean is None else state_mean
                state_std = np.asarray(std, dtype=np.float32) if state_std is None else state_std
            except Exception:  # pragma: no cover - statistics are best effort
                pass

    return RandomMLPPrior(
        state_dim=int(state_dim),
        hidden_dim=int(hidden_dim),
        state_mean=state_mean,
        state_std=state_std,
        seed=int(seed),
        name=name,
        **kwargs,
    )


__all__ = [
    "MLP_HIDDEN_DIM",
    "MLP_OUTPUT_CLIP",
    "MLPParameters",
    "sample_mlp_parameters",
    "RandomMLPReward",
    "RandomMLPPrior",
    "make_mlp_prior",
]
