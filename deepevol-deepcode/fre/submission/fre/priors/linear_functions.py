"""Random linear reward functions used as part of the FRE prior reward distribution.

Appendix B, "Training Details":

    Random Linear functions are generated according to a uniform vector within -1 and 1.
    On AntMaze, we remove the XY positions from this generation as the scale of the
    dimensions led to instability. A random binary mask is applied with a 0.9 chance to
    zero the vector at that dimension, to encourage sparsity and bias towards simpler
    functions.

Concretely, for a state ``s`` the reward is

    eta(s) = <w, s> [+ b]      with   w_d = u_d * m_d,  u_d ~ U(-1, 1),  m_d ~ Bern(1 - p)

and ``p = 0.9`` (i.e. each dimension is zeroed with probability 0.9).  ``b`` (an optional
uniform bias) is disabled by default because the paper describes the linear class as
"'MLPs' with one linear layer" (Section 4.2); a bias-free linear layer keeps the family
centred at zero, consistent with the random-MLP family (tanh output in [-1, 1]).

AntMaze note: the XY positions live at indices ``(0, 1)`` of the observation (see
:mod:`fre.data.preprocessing`); those dimensions are excluded from the *generation* of
the weight vector, so ``eta`` never depends on XY (Appendix B).
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # pragma: no cover - torch is a hard dependency in practice
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False

from fre.priors.goal_functions import RewardFunction

__all__ = [
    "LinearRewardFunction",
    "sample_linear_weights",
    "sample_linear_reward_functions",
    "antmaze_exclude_dims",
    "DEFAULT_MASK_PROB",
    "DEFAULT_WEIGHT_RANGE",
    "ANTMAZE_XY_INDICES",
]

#: Probability that an individual weight dimension is zeroed (Appendix B: "0.9 chance").
DEFAULT_MASK_PROB: float = 0.9

#: Uniform distribution support for the raw weights (Appendix B: "within -1 and 1").
DEFAULT_WEIGHT_RANGE: Tuple[float, float] = (-1.0, 1.0)

#: AntMaze XY observation indices removed from the prior generation (Appendix B).
ANTMAZE_XY_INDICES: Tuple[int, int] = (0, 1)


def _as_rng(rng: Optional[Union[np.random.Generator, int]]) -> np.random.Generator:
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, np.random.RandomState):  # pragma: no cover - legacy
        return np.random.default_rng(int(rng.randint(0, 2 ** 31 - 1)))
    return np.random.default_rng(int(rng))


class LinearRewardFunction(RewardFunction):
    """Random linear reward function ``eta(s) = <w, s> (+ b)``.

    Parameters
    ----------
    weights:
        Either a 1-D array ``(state_dim,)`` describing a single reward function, or a
        2-D array ``(num_functions, state_dim)`` describing a batch of reward functions.
    bias:
        Optional scalar / ``(num_functions,)`` additive offset.  ``None`` (default)
        disables the bias, matching the paper's "one linear layer" description.
    state_dim:
        Number of observation dimensions.  Inferred from ``weights`` when omitted.
    mask:
        Optional binary mask (same shape as ``weights``) recording which dimensions
        survived sampling.  Kept for introspection / logging.
    exclude_dims:
        Dimensions excluded from generation (e.g. AntMaze XY).  Rewards are insensitive
        to these dimensions by construction.
    """

    family: str = "linear"

    def __init__(
        self,
        weights: Union[np.ndarray, Sequence[float]],
        bias: Optional[Union[float, np.ndarray, Sequence[float]]] = None,
        state_dim: Optional[int] = None,
        mask: Optional[np.ndarray] = None,
        exclude_dims: Optional[Iterable[int]] = None,
        weight_range: Tuple[float, float] = DEFAULT_WEIGHT_RANGE,
        mask_prob: float = DEFAULT_MASK_PROB,
        name: Optional[str] = None,
    ) -> None:
        w = np.asarray(weights, dtype=np.float32)
        if w.ndim == 0:
            w = w.reshape(1)
        if state_dim is None:
            state_dim = int(w.shape[-1])
        self.weights = w
        self.state_dim = int(state_dim)
        self.mask = None if mask is None else np.asarray(mask, dtype=np.float32)
        self.exclude_dims = tuple(int(d) for d in (exclude_dims or ()))
        self.weight_range = (float(weight_range[0]), float(weight_range[1]))
        self.mask_prob = float(mask_prob)
        if bias is None:
            self.bias = None
        else:
            b = np.asarray(bias, dtype=np.float32)
            self.bias = float(b) if b.ndim == 0 else b
        self.name = name

    # ------------------------------------------------------------------ helpers
    @property
    def num_functions(self) -> int:
        return 1 if self.weights.ndim == 1 else int(self.weights.shape[0])

    def at(self, index: int) -> "LinearRewardFunction":
        """Return a single-function view of a batched reward function."""
        if self.weights.ndim == 1:
            if index != 0:
                raise IndexError("single linear reward function has only index 0")
            return self
        bias = None
        if self.bias is not None:
            bias = float(np.asarray(self.bias).reshape(-1)[index])
        mask = None if self.mask is None else self.mask[index]
        return LinearRewardFunction(
            self.weights[index],
            bias=bias,
            state_dim=self.state_dim,
            mask=mask,
            exclude_dims=self.exclude_dims,
            weight_range=self.weight_range,
            mask_prob=self.mask_prob,
            name=None if self.name is None else f"{self.name}[{index}]",
        )

    # ------------------------------------------------------------------- reward
    def reward(self, states: Any) -> Any:
        """Evaluate ``eta(s)`` for ``states`` of shape ``(..., state_dim)``."""
        return self.__call__(states)

    def __call__(self, states: Any) -> Any:
        if _HAS_TORCH and isinstance(states, torch.Tensor):  # type: ignore[arg-type]
            w = torch.as_tensor(
                np.atleast_2d(self.weights), dtype=states.dtype, device=states.device
            )
            flat_states = states.reshape(-1, states.shape[-1])
            out = flat_states @ w.t()  # (M, F)
            if self.bias is not None:
                b = torch.as_tensor(
                    np.broadcast_to(np.atleast_1d(np.asarray(self.bias, dtype=np.float32)), (w.shape[0],)),
                    dtype=states.dtype,
                    device=states.device,
                )
                out = out + b
            out = out.reshape(tuple(states.shape[:-1]) + (w.shape[0],))
            if self.weights.ndim == 1:
                out = out[..., 0]
            return out

        s = np.asarray(states, dtype=np.float32)
        w = np.atleast_2d(self.weights)
        flat_states = s.reshape(-1, s.shape[-1])
        out = flat_states @ w.T  # (M, F)
        if self.bias is not None:
            out = out + np.atleast_1d(np.asarray(self.bias, dtype=np.float32))
        out = out.reshape(tuple(s.shape[:-1]) + (w.shape[0],))
        if self.weights.ndim == 1:
            out = out[..., 0]
        return out.astype(np.float32)

    # ----------------------------------------------------------------- sampling
    @classmethod
    def sample(
        cls,
        num_functions: int,
        state_dim: int,
        rng: Optional[Union[np.random.Generator, int]] = None,
        mask_prob: float = DEFAULT_MASK_PROB,
        weight_range: Tuple[float, float] = DEFAULT_WEIGHT_RANGE,
        exclude_dims: Optional[Iterable[int]] = None,
        return_mask: bool = False,
        bias: Optional[Any] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Union["LinearRewardFunction", Tuple["LinearRewardFunction", np.ndarray]]:
        """Sample ``num_functions`` random linear reward functions (Appendix B)."""
        rng = _as_rng(rng)
        weights, mask = sample_linear_weights(
            num_functions,
            state_dim,
            rng=rng,
            mask_prob=mask_prob,
            weight_range=weight_range,
            exclude_dims=exclude_dims,
            return_mask=True,
        )
        obj = cls(
            weights[0] if int(num_functions) == 1 else weights,
            bias=bias,
            state_dim=state_dim,
            mask=mask[0] if int(num_functions) == 1 else mask,
            exclude_dims=exclude_dims,
            mask_prob=mask_prob,
            weight_range=weight_range,
            name=name,
        )
        if return_mask:
            return obj, mask
        return obj

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"state_dim={self.state_dim}, num_functions={self.num_functions}, "
            f"mask_prob={self.mask_prob}, exclude_dims={self.exclude_dims}"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}({self.extra_repr()})"


def sample_linear_weights(
    num_functions: int,
    state_dim: int,
    rng: Optional[Union[np.random.Generator, int]] = None,
    mask_prob: float = DEFAULT_MASK_PROB,
    weight_range: Tuple[float, float] = DEFAULT_WEIGHT_RANGE,
    exclude_dims: Optional[Iterable[int]] = None,
    return_mask: bool = False,
    dtype=np.float32,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Sample ``(num_functions, state_dim)`` random linear weights (Appendix B).

    1. draw each dimension uniformly from ``weight_range`` (default ``[-1, 1]``),
    2. apply an independent Bernoulli ``mask_prob`` (=0.9) zeroing mask,
    3. force ``exclude_dims`` (AntMaze XY) to zero so the function is independent of them
       "as the scale of the dimensions led to instability".
    """
    rng = _as_rng(rng)
    num_functions = max(int(num_functions), 1)
    state_dim = int(state_dim)
    low, high = float(weight_range[0]), float(weight_range[1])
    weights = rng.uniform(low, high, size=(num_functions, state_dim)).astype(dtype)
    mask = (rng.random((num_functions, state_dim)) >= float(mask_prob)).astype(dtype)
    weights = (weights * mask).astype(dtype)
    for dim in tuple(int(d) for d in (exclude_dims or ())):
        if 0 <= dim < state_dim:
            weights[:, dim] = 0.0
            mask[:, dim] = 0.0
    if return_mask:
        return weights, mask
    return weights


def sample_linear_reward_functions(
    num_functions: int,
    state_dim: int,
    rng: Optional[Union[np.random.Generator, int]] = None,
    mask_prob: float = DEFAULT_MASK_PROB,
    weight_range: Tuple[float, float] = DEFAULT_WEIGHT_RANGE,
    exclude_dims: Optional[Iterable[int]] = None,
    as_list: bool = False,
    bias: Optional[Any] = None,
    name: Optional[str] = None,
    **kwargs: Any,
) -> Union[LinearRewardFunction, List[LinearRewardFunction]]:
    """Sample random linear reward functions (Appendix B).

    Returns one batched :class:`LinearRewardFunction` (evaluable over all sampled
    functions simultaneously) by default; ``as_list=True`` returns one object per
    function, the form consumed by :mod:`fre.priors.reward_prior`.
    """
    obj, _mask = LinearRewardFunction.sample(  # type: ignore[misc]
        num_functions,
        state_dim,
        rng=rng,
        mask_prob=mask_prob,
        weight_range=weight_range,
        exclude_dims=exclude_dims,
        return_mask=True,
        bias=bias,
        name=name,
    )
    if not as_list:
        return obj
    return [obj.at(i) for i in range(obj.num_functions)]


def antmaze_exclude_dims(
    xy_indices: Iterable[int] = ANTMAZE_XY_INDICES,
    state_dim: Optional[int] = None,
) -> Tuple[int, ...]:
    """Return the AntMaze dimensions excluded from linear generation (XY)."""
    dims = tuple(int(d) for d in xy_indices)
    if state_dim is not None:
        dims = tuple(d for d in dims if 0 <= d < int(state_dim))
    return dims
