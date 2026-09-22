"""Random MLP reward functions (the third family of the FRE prior distribution).

Paper specification
-------------------
Section 4.2:

    "... a reasonable yet powerful prior distribution can be constructed from a
    mixture of random unsupervised functions. The particular mixture we use
    consists of random singleton functions (corresponding to "goal reaching"
    rewards), random neural networks (MLPs with two linear layers), and random
    linear functions ... A uniform mixture of the three function classes are
    used during training."

Appendix B:

    "Random MLP functions are generated using a neural network of size
    (state_dim, 32, 1). Parameters are sampled using a normal distribution
    scaled by the average dimension of the layer. A tanh activation is used
    between the two layers. The final output of the neural network is clipped
    between -1 and 1."

So a sampled reward function is::

    eta(s) = clip( W2 @ tanh(W1 @ s + b1) + b2, -1, 1 )

with ``W1`` of shape ``(32, state_dim)`` and ``W2`` of shape ``(1, 32)``.  Each
parameter tensor is drawn entrywise from a zero-mean normal whose standard
deviation derives from the *average dimension of the layer* (the plan's
default resolution: ``1 / sqrt((fan_in + fan_out) / 2)``).

Config flags (the paper is silent on some of these; the plan requires them to
be swappable):

* ``scaling``: ``"avg_dim_sqrt"`` (paper-faithful default), ``"fan_in"``,
  ``"fan_avg"`` or ``"none"``.
* ``bias``: sample layer biases (default ``True``); ``False`` gives a purely
  homogeneous function, mirroring the linear family.
* ``clip_value``: half-width of the symmetric output clip (default ``1.0``,
  ``None`` disables it).
* ``zero_dims``: optional input dimensions to force-ignore (all-zero incoming
  weights).  Default ``None`` keeps all dimensions, matching Appendix B.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # optional: allow torch tensors as states
    import torch  # type: ignore

    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch is a hard dependency of FRE
    _HAS_TORCH = False

from .goal_functions import RewardFunction


__all__ = [
    "MLPRewardFunction",
    "sample_mlp_parameters",
    "sample_mlp_reward_functions",
    "DEFAULT_HIDDEN_DIM",
    "DEFAULT_CLIP_VALUE",
    "DEFAULT_HIDDEN_ACTIVATION",
]

# ---------------------------------------------------------------------------
# Defaults (Appendix B)
# ---------------------------------------------------------------------------

#: Hidden layer width of the random MLP: ``(state_dim, 32, 1)``.
DEFAULT_HIDDEN_DIM: int = 32

#: The final output of the network is clipped between -1 and 1.
DEFAULT_CLIP_VALUE: float = 1.0

#: Activation used between the two linear layers.
DEFAULT_HIDDEN_ACTIVATION: str = "tanh"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_rng(rng: Union[None, int, np.random.Generator, np.random.RandomState]) -> np.random.Generator:
    """Normalise ``None | int | Generator | RandomState`` into a Generator."""
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, np.random.RandomState):
        return np.random.default_rng(int(rng.randint(0, 2**31 - 1)))
    return np.random.default_rng(int(rng))


def _layer_scale(fan_in: int, fan_out: int, scaling: str) -> float:
    """Standard deviation used to sample the weights of one layer.

    The paper's rule is "sampled using a normal distribution scaled by the
    average dimension of the layer", i.e. ``1 / sqrt((fan_in + fan_out) / 2)``.
    """
    scaling = (scaling or "avg_dim_sqrt").lower()
    if scaling in ("avg_dim_sqrt", "avg", "average", "avg_dim"):
        avg_dim = 0.5 * (float(fan_in) + float(fan_out))
        return 1.0 / np.sqrt(max(avg_dim, 1.0))
    if scaling in ("fan_in_sqrt", "fan_in"):
        return 1.0 / np.sqrt(max(float(fan_in), 1.0))
    if scaling in ("fan_avg",):
        return float(np.sqrt(2.0 / max(float(fan_in) + float(fan_out), 1.0)))
    if scaling in ("none", "identity", "1", "unit"):
        return 1.0
    raise ValueError(f"Unknown MLP weight scaling '{scaling}'.")


def _apply_activation(x: Any, activation: str, use_torch: bool) -> Any:
    """Apply the hidden activation to ``x`` (numpy or torch tensor)."""
    act = (activation or DEFAULT_HIDDEN_ACTIVATION).lower()
    if use_torch:
        if act == "tanh":
            return torch.tanh(x)
        if act == "relu":
            return torch.relu(x)
        if act == "gelu":
            return torch.nn.functional.gelu(x)
        if act in ("silu", "swish"):
            return torch.nn.functional.silu(x)
        if act in ("identity", "linear", "none"):
            return x
        raise ValueError(f"Unknown MLP prior activation '{activation}'.")

    if act == "tanh":
        return np.tanh(x)
    if act == "relu":
        return np.maximum(x, 0.0)
    if act == "gelu":
        return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
    if act in ("silu", "swish"):
        return x / (1.0 + np.exp(-x))
    if act in ("identity", "linear", "none"):
        return x
    raise ValueError(f"Unknown MLP prior activation '{activation}'.")


# ---------------------------------------------------------------------------
# Batched random MLP reward function
# ---------------------------------------------------------------------------


class MLPRewardFunction(RewardFunction):
    """A (batch of) random MLP reward function(s) ``eta(s)``.

    Output shape convention matches :class:`LinearRewardFunction`: for a single
    function the result has shape ``states.shape[:-1]``; for ``N`` batched
    functions it has shape ``states.shape[:-1] + (N,)``.
    """

    family = "mlp"

    def __init__(
        self,
        weights: Union[Sequence[Any], Sequence[Sequence[Any]]],
        state_dim: Optional[int] = None,
        clip_value: Optional[float] = DEFAULT_CLIP_VALUE,
        hidden_activation: str = DEFAULT_HIDDEN_ACTIVATION,
        zero_dims: Optional[Iterable[int]] = None,
        name: Optional[str] = None,
    ) -> None:
        params = list(weights)
        if len(params) == 2 and np.asarray(params[0][0]).ndim == 2:
            # A single function ``[(w1, b1), (w2, b2)]`` -> wrap as a batch of one.
            params = [params]
        if not params:
            raise ValueError("MLPRewardFunction requires at least one parameter set.")

        parsed: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for entry in params:
            w1, b1, w2, b2 = entry
            w1 = np.asarray(w1, dtype=np.float32)
            b1 = np.asarray(b1, dtype=np.float32).reshape(-1)
            w2 = np.asarray(w2, dtype=np.float32).reshape(-1)
            b2 = np.asarray(b2, dtype=np.float32).reshape(-1)
            if w2.shape[0] != w1.shape[0]:
                raise ValueError("Inconsistent hidden size between W1 and W2.")
            if b1.shape[0] != w1.shape[0]:
                raise ValueError("Hidden bias b1 does not match the hidden width.")
            parsed.append((w1, b1, w2, b2))

        self.state_dim = int(state_dim) if state_dim is not None else int(parsed[0][0].shape[1])
        self.clip_value = None if clip_value is None else float(clip_value)
        self.hidden_activation = hidden_activation
        self.zero_dims = tuple(sorted({int(d) for d in zero_dims})) if zero_dims is not None else ()
        self.name = name

        if self.zero_dims:
            for i, (w1, b1, w2, b2) in enumerate(parsed):
                w1 = w1.copy()
                for dim in self.zero_dims:
                    if 0 <= dim < w1.shape[1]:
                        w1[:, dim] = 0.0
                parsed[i] = (w1, b1, w2, b2)

        self._params = parsed
        # Stacked arrays for batched evaluation.
        self._W1 = np.stack([p[0] for p in parsed], axis=0)  # (N, H, D)
        self._B1 = np.stack([p[1] for p in parsed], axis=0)  # (N, H)
        self._W2 = np.stack([p[2] for p in parsed], axis=0)  # (N, H)
        self._B2 = np.stack([p[3] for p in parsed], axis=0)  # (N,)

    # -- introspection ----------------------------------------------------

    @property
    def num_functions(self) -> int:
        return len(self._params)

    @property
    def hidden_dim(self) -> int:
        return int(self._W1.shape[1])

    @property
    def weights(self) -> np.ndarray:
        """Stacked first-layer weights, shape ``(N, hidden, state_dim)``."""
        return self._W1

    def at(self, index: int) -> "MLPRewardFunction":
        """Return the ``index``-th function as a standalone object."""
        w1, b1, w2, b2 = self._params[index]
        return MLPRewardFunction(
            (w1, b1, w2, b2),
            state_dim=self.state_dim,
            clip_value=self.clip_value,
            hidden_activation=self.hidden_activation,
            name=self.name,
        )

    # -- evaluation -------------------------------------------------------

    def reward(self, states: Any) -> Any:
        """Evaluate ``eta(s)`` (clipped to ``[-1, 1]`` by default)."""
        if _HAS_TORCH and isinstance(states, torch.Tensor):
            return self._reward_torch(states)

        states = np.asarray(states)
        flat = states.reshape(-1, states.shape[-1]).astype(np.float32, copy=False)

        if self.num_functions == 1:
            w1, b1, w2, b2 = self._params[0]
            hidden = _apply_activation(flat @ w1.T + b1[None, :], self.hidden_activation, False)
            out = hidden @ w2 + b2[0]
        else:
            hidden = np.einsum("mhd,nd->nmh", self._W1, flat) + self._B1[:, None, :]
            hidden = _apply_activation(hidden, self.hidden_activation, False)
            out = np.einsum("nmh,mh->nm", hidden, self._W2) + self._B2[None, :]

        if self.clip_value is not None:
            out = np.clip(out, -self.clip_value, self.clip_value)

        if self.num_functions > 1:
            return out.reshape(states.shape[:-1] + (self.num_functions,))
        return out.reshape(states.shape[:-1])

    def _reward_torch(self, states: "torch.Tensor") -> "torch.Tensor":  # noqa: F821
        device, dtype = states.device, states.dtype
        w1 = torch.as_tensor(self._W1, device=device, dtype=dtype)
        b1 = torch.as_tensor(self._B1, device=device, dtype=dtype)
        w2 = torch.as_tensor(self._W2, device=device, dtype=dtype)
        b2 = torch.as_tensor(self._B2, device=device, dtype=dtype)

        flat = states.reshape(-1, states.shape[-1])
        if self.num_functions == 1:
            hidden = _apply_activation(flat @ w1[0].t() + b1[0][None, :], self.hidden_activation, True)
            out = hidden @ w2[0] + b2[0]
        else:
            hidden = torch.einsum("mhd,nd->nmh", w1, flat) + b1[None, :, :]
            hidden = _apply_activation(hidden, self.hidden_activation, True)
            out = torch.einsum("nmh,mh->nm", hidden, w2) + b2[None, :]

        if self.clip_value is not None:
            out = torch.clamp(out, -self.clip_value, self.clip_value)

        if self.num_functions > 1:
            return out.reshape(states.shape[:-1] + (self.num_functions,))
        return out.reshape(states.shape[:-1])

    def extra_repr(self) -> str:
        return (
            f"state_dim={self.state_dim}, hidden_dim={self.hidden_dim}, "
            f"num_functions={self.num_functions}, activation={self.hidden_activation}, "
            f"clip={self.clip_value}"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}({self.extra_repr()})"


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_mlp_parameters(
    num_functions: int,
    state_dim: int,
    rng: Union[None, int, np.random.Generator, np.random.RandomState] = None,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    scaling: str = "avg_dim_sqrt",
    bias: bool = True,
    zero_dims: Optional[Iterable[int]] = None,
    dtype: Any = np.float32,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Sample parameters for ``num_functions`` random MLPs of size ``(D, H, 1)``.

    Returns a list of ``(w1, b1, w2, b2)`` tuples with shapes ``(H, D)``,
    ``(H,)``, ``(H,)`` and ``()``.
    """
    n = int(num_functions)
    d = int(state_dim)
    h = int(hidden_dim)
    if n <= 0 or d <= 0 or h <= 0:
        raise ValueError("num_functions, state_dim and hidden_dim must all be positive.")

    rng = _as_rng(rng)
    scale1 = _layer_scale(d, h, scaling)
    scale2 = _layer_scale(h, 1, scaling)

    w1 = (rng.standard_normal((n, h, d)) * scale1).astype(dtype)
    w2 = (rng.standard_normal((n, 1, h)) * scale2).astype(dtype)
    if bias:
        b1 = (rng.standard_normal((n, h)) * scale1).astype(dtype)
        b2 = (rng.standard_normal((n, 1)) * scale2).astype(dtype)
    else:
        b1 = np.zeros((n, h), dtype=dtype)
        b2 = np.zeros((n, 1), dtype=dtype)

    if zero_dims is not None:
        for dim in zero_dims:
            dim = int(dim)
            if 0 <= dim < d:
                w1[:, :, dim] = 0.0

    return [(w1[i], b1[i], w2[i, 0], b2[i, 0]) for i in range(n)]


def sample_mlp_reward_functions(
    num_functions: int,
    state_dim: int,
    rng: Union[None, int, np.random.Generator, np.random.RandomState] = None,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    scaling: str = "avg_dim_sqrt",
    clip_value: Optional[float] = DEFAULT_CLIP_VALUE,
    hidden_activation: str = DEFAULT_HIDDEN_ACTIVATION,
    bias: bool = True,
    zero_dims: Optional[Iterable[int]] = None,
    as_list: bool = False,
    name: Optional[str] = None,
    **kwargs,
) -> Union[MLPRewardFunction, List[MLPRewardFunction]]:
    """Sample random MLP reward functions.

    Returns one batched :class:`MLPRewardFunction` by default, or a list of
    single-function objects when ``as_list=True`` (the per-function form used by
    the reward-prior mixture dispatcher).
    """
    params = sample_mlp_parameters(
        num_functions=num_functions,
        state_dim=state_dim,
        rng=rng,
        hidden_dim=hidden_dim,
        scaling=scaling,
        bias=bias,
        zero_dims=zero_dims,
    )
    fn = MLPRewardFunction(
        params,
        state_dim=state_dim,
        clip_value=clip_value,
        hidden_activation=hidden_activation,
        name=name,
    )
    if not as_list:
        return fn
    return [fn.at(i) for i in range(fn.num_functions)]
