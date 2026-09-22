"""z-conditioned RL network builders for FRE.

The paper (Appendix A, Table 3) specifies a plain MLP with hidden layers
``[512, 512, 512]`` for every RL component (Q-function, value function and
policy) and states that the latent task representation ``z`` is *simply
concatenated to the observation* that is fed into each RL component
(Section 4.3) rather than being injected through any special conditioning
mechanism.

This module centralises those builders so that:

* :class:`fre.models.iql.IQL` (the main trainable agent) and the in-house
  baselines (GC-IQL, GC-BC and OPAL) all share the exact same network shapes;
* the layer sizes / activations / layer-norm switches are config driven and
  therefore trivially overridable from ``fre/configs/*.yaml``.

Nothing in here depends on the encoder, so the module stays importable during
Phase 1 of the strided schedule (encoder/decoder only training).
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn


__all__ = [
    "DEFAULT_RL_HIDDEN_DIMS",
    "DEFAULT_RL_ACTIVATION",
    "DEFAULT_LATENT_DIM",
    "get_activation",
    "build_z_conditioned_mlp",
    "ZConditionedMLP",
    "make_q_network",
    "make_value_network",
    "make_policy_network",
    "make_iql_networks",
    "build_gc_mlp",
    "initialize_weights",
    "count_parameters",
]


# ---------------------------------------------------------------------------
# Defaults -- taken from Appendix A, Table 3
# ---------------------------------------------------------------------------
DEFAULT_RL_HIDDEN_DIMS: Tuple[int, ...] = (512, 512, 512)
DEFAULT_RL_ACTIVATION: str = "relu"
DEFAULT_LATENT_DIM: int = 128


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def get_activation(name: Optional[Union[str, Callable[[], nn.Module]]]) -> nn.Module:
    """Resolve an activation name (or callable) into an ``nn.Module``.

    The paper is silent about the RL activation function; we default to ReLU
    (the standard choice for the reference IQL implementation).  ``None``
    returns ``nn.Identity`` so a pure linear stack can be built.
    """
    if name is None:
        return nn.Identity()
    if not isinstance(name, str):
        return name() if isinstance(name, type) else name
    key = name.strip().lower()
    mapping = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "elu": nn.ELU,
        "mish": nn.Mish,
        "tanh": nn.Tanh,
        "identity": nn.Identity,
        "linear": nn.Identity,
        "none": nn.Identity,
    }
    if key not in mapping:
        raise ValueError(f"Unknown activation '{name}'. Options: {sorted(mapping)}")
    return mapping[key]()


def initialize_weights(module: nn.Module, gain: float = math.sqrt(2.0)) -> nn.Module:
    """Xavier/Glorot init for ``nn.Linear`` blocks with zeroed biases.

    ``gain=sqrt(2)`` is the standard ReLU gain.  LayerNorm weights are reset to
    one/zero.  Other module types are left untouched.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    return module


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """Number of parameters, optionally restricted to those requiring grad."""
    return sum(
        p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only)
    )


# ---------------------------------------------------------------------------
# Generic z-conditioned MLP
# ---------------------------------------------------------------------------
def build_z_conditioned_mlp(
    state_dim: int,
    latent_dim: int = DEFAULT_LATENT_DIM,
    action_dim: int = 0,
    hidden_dims: Sequence[int] = DEFAULT_RL_HIDDEN_DIMS,
    output_dim: int = 1,
    activation: Union[str, Callable] = DEFAULT_RL_ACTIVATION,
    output_activation: Optional[Union[str, Callable]] = None,
    use_layer_norm: bool = False,
    layernorm_before_activation: bool = True,
    init_gain: float = math.sqrt(2.0),
    init: bool = True,
) -> nn.Sequential:
    """Build an MLP consuming ``concat([state, action?, z])``.

    Parameters
    ----------
    state_dim:
        Dimensionality of the (possibly augmented) observation.
    latent_dim:
        Dimensionality of the FRE latent task representation ``z`` (128).
    action_dim:
        Number of action dimensions to concatenate between the state and ``z``.
        Use ``0`` for the value function / policy backbone, ``>0`` for the
        Q-function.
    hidden_dims:
        Hidden layer widths; the paper uses ``[512, 512, 512]`` (Table 3).
    output_dim:
        Size of the final linear head (1 for Q/V, ``2 * action_dim`` for a
        Gaussian policy head).
    use_layer_norm:
        If ``True``, insert ``nn.LayerNorm`` in each hidden block.  The paper's
        RL networks are plain MLPs, so the default is ``False``; GC-BC
        (Addendum) uses LayerNorm and enables this flag.
    """
    input_dim = int(state_dim) + int(action_dim) + int(latent_dim)
    layers = []
    prev_dim = input_dim
    for hidden in hidden_dims:
        hidden = int(hidden)
        layers.append(nn.Linear(prev_dim, hidden))
        if use_layer_norm and layernorm_before_activation:
            layers.append(nn.LayerNorm(hidden))
        layers.append(get_activation(activation))
        if use_layer_norm and not layernorm_before_activation:
            layers.append(nn.LayerNorm(hidden))
        prev_dim = hidden
    if output_dim is not None and output_dim > 0:
        layers.append(nn.Linear(prev_dim, int(output_dim)))
        if output_activation is not None:
            layers.append(get_activation(output_activation))
    net = nn.Sequential(*layers)
    if init:
        net.apply(lambda m: initialize_weights(m, gain=init_gain))
    return net


class ZConditionedMLP(nn.Module):
    """Thin wrapper around a z-conditioned MLP.

    Keeps track of the shapes so it can be re-used for Q (state + action),
    V (state) and policy (state) heads while always appending ``z`` last::

        forward(states, z=z)                        # value / policy backbone
        forward(states, actions, z=z)               # Q-function

    ``z`` may be a single ``(latent_dim,)`` vector (broadcast over the batch),
    a ``(batch, latent_dim)`` tensor, or a ``(batch, 1, latent_dim)`` tensor.
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int = DEFAULT_LATENT_DIM,
        action_dim: int = 0,
        hidden_dims: Sequence[int] = DEFAULT_RL_HIDDEN_DIMS,
        output_dim: int = 1,
        activation: Union[str, Callable] = DEFAULT_RL_ACTIVATION,
        output_activation: Optional[Union[str, Callable]] = None,
        use_layer_norm: bool = False,
        init_gain: float = math.sqrt(2.0),
        init: bool = True,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.output_dim = int(output_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        # Raw (possibly un-augmented) observation width, without z.
        self.raw_dim = self.state_dim + self.action_dim
        self.net = build_z_conditioned_mlp(
            state_dim=self.state_dim,
            latent_dim=self.latent_dim,
            action_dim=self.action_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            activation=activation,
            output_activation=output_activation,
            use_layer_norm=use_layer_norm,
            init_gain=init_gain,
            init=init,
        )
        # Input width including z -- kept for state_dict / sanity checks.
        self.input_dim = self.raw_dim + self.latent_dim

    @staticmethod
    def _flatten_latent(z: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Collapse any extra leading dims of ``z`` to ``(batch_size, latent_dim)``."""
        if z.dim() == 1:
            return z.unsqueeze(0).expand(batch_size, -1)
        latent_dim = z.shape[-1]
        return z.reshape(-1, latent_dim)[:batch_size] if z.numel() != batch_size * latent_dim else z.reshape(batch_size, latent_dim)

    def forward(
        self,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if z is None:
            z = latent
        if z is None:
            raise ValueError("ZConditionedMLP.forward requires `z` (or `latent`).")
        if actions is not None and self.action_dim > 0:
            piece = torch.cat([states, actions], dim=-1)
        else:
            piece = states
        z = self._flatten_latent(z, piece.shape[0])
        x = torch.cat([piece, z], dim=-1)
        return self.net(x)

    def extra_repr(self) -> str:
        return (
            f"state_dim={self.state_dim}, action_dim={self.action_dim}, "
            f"latent_dim={self.latent_dim}, hidden_dims={list(self.hidden_dims)}, "
            f"output_dim={self.output_dim}"
        )


# ---------------------------------------------------------------------------
# Factory helpers -- these reuse the concrete networks from fre.models.iql so
# that the agent and any ad-hoc experiment share identical definitions.
# ---------------------------------------------------------------------------
def _iql_module():
    """Import :mod:`fre.models.iql` (no circular dependency: iql does not import us)."""
    try:
        from fre.models import iql as _iql  # type: ignore
    except Exception:  # pragma: no cover - fallback for direct file execution
        from . import iql as _iql  # type: ignore
    return _iql


def make_q_network(
    state_dim: int,
    action_dim: int,
    latent_dim: int = DEFAULT_LATENT_DIM,
    hidden_dims: Sequence[int] = DEFAULT_RL_HIDDEN_DIMS,
    activation: str = DEFAULT_RL_ACTIVATION,
    use_layer_norm: bool = False,
    **kwargs,
) -> nn.Module:
    """Q(s, a, z) network with the paper's ``[512, 512, 512]`` MLP."""
    iql = _iql_module()
    return iql.QNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        hidden_dims=tuple(hidden_dims),
        activation=activation,
        use_layer_norm=use_layer_norm,
        **kwargs,
    )


def make_value_network(
    state_dim: int,
    latent_dim: int = DEFAULT_LATENT_DIM,
    hidden_dims: Sequence[int] = DEFAULT_RL_HIDDEN_DIMS,
    activation: str = DEFAULT_RL_ACTIVATION,
    use_layer_norm: bool = False,
    **kwargs,
) -> nn.Module:
    """V(s, z) network with the paper's ``[512, 512, 512]`` MLP."""
    iql = _iql_module()
    return iql.ValueNetwork(
        state_dim=state_dim,
        latent_dim=latent_dim,
        hidden_dims=tuple(hidden_dims),
        activation=activation,
        use_layer_norm=use_layer_norm,
        **kwargs,
    )


def make_policy_network(
    state_dim: int,
    action_dim: int,
    latent_dim: int = DEFAULT_LATENT_DIM,
    hidden_dims: Sequence[int] = DEFAULT_RL_HIDDEN_DIMS,
    activation: str = DEFAULT_RL_ACTIVATION,
    use_layer_norm: bool = False,
    **kwargs,
) -> nn.Module:
    """pi(a | s, z) tanh-squashed Gaussian policy with ``[512, 512, 512]`` MLP."""
    iql = _iql_module()
    return iql.GaussianPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=latent_dim,
        hidden_dims=tuple(hidden_dims),
        activation=activation,
        use_layer_norm=use_layer_norm,
        **kwargs,
    )


def make_iql_networks(
    state_dim: int,
    action_dim: int,
    latent_dim: int = DEFAULT_LATENT_DIM,
    hidden_dims: Sequence[int] = DEFAULT_RL_HIDDEN_DIMS,
    activation: str = DEFAULT_RL_ACTIVATION,
    use_layer_norm: bool = False,
    action_low: float = -1.0,
    action_high: float = 1.0,
    **kwargs,
) -> Dict[str, nn.Module]:
    """Convenience builder returning all z-conditioned RL modules.

    Returns a dict with keys ``q1``, ``q2``, ``q_target1``, ``q_target2``, ``v``
    and ``policy`` -- the same modules :class:`fre.models.iql.IQL` owns.
    """
    q1 = make_q_network(state_dim, action_dim, latent_dim, hidden_dims, activation, use_layer_norm, **kwargs)
    q2 = make_q_network(state_dim, action_dim, latent_dim, hidden_dims, activation, use_layer_norm, **kwargs)
    q_target1 = make_q_network(state_dim, action_dim, latent_dim, hidden_dims, activation, use_layer_norm, **kwargs)
    q_target2 = make_q_network(state_dim, action_dim, latent_dim, hidden_dims, activation, use_layer_norm, **kwargs)
    q_target1.load_state_dict(q1.state_dict())
    q_target2.load_state_dict(q2.state_dict())
    for net in (q_target1, q_target2):
        net.requires_grad_(False)
    v = make_value_network(state_dim, latent_dim, hidden_dims, activation, use_layer_norm, **kwargs)
    policy = make_policy_network(
        state_dim,
        action_dim,
        latent_dim,
        hidden_dims,
        activation,
        use_layer_norm,
        action_low=action_low,
        action_high=action_high,
    )
    return {
        "q1": q1,
        "q2": q2,
        "q_target1": q_target1,
        "q_target2": q_target2,
        "v": v,
        "policy": policy,
    }


# ---------------------------------------------------------------------------
# Goal-conditioned baseline builder (GC-IQL / GC-BC)
# ---------------------------------------------------------------------------
def build_gc_mlp(
    input_dim: int,
    hidden_dims: Sequence[int] = DEFAULT_RL_HIDDEN_DIMS,
    output_dim: int = 1,
    activation: str = DEFAULT_RL_ACTIVATION,
    output_activation: Optional[Union[str, Callable]] = None,
    use_layer_norm: bool = False,
    layernorm_before_activation: bool = True,
    init_gain: float = math.sqrt(2.0),
) -> nn.Sequential:
    """MLP builder for the goal-conditioned baselines.

    GC-IQL concatenates the goal to the observation and uses the same
    ``[512, 512, 512]`` ReLU MLP as FRE.  GC-BC (Addendum) instead uses *three
    hidden layers of 512 units with ReLU and a LayerNorm applied before each
    activation* -- that is ``use_layer_norm=True`` with
    ``layernorm_before_activation=True``.
    """
    layers = []
    prev_dim = int(input_dim)
    for hidden in hidden_dims:
        hidden = int(hidden)
        layers.append(nn.Linear(prev_dim, hidden))
        if use_layer_norm and layernorm_before_activation:
            layers.append(nn.LayerNorm(hidden))
        layers.append(get_activation(activation))
        if use_layer_norm and not layernorm_before_activation:
            layers.append(nn.LayerNorm(hidden))
        prev_dim = hidden
    if output_dim is not None and output_dim > 0:
        layers.append(nn.Linear(prev_dim, int(output_dim)))
        if output_activation is not None:
            layers.append(get_activation(output_activation))
    net = nn.Sequential(*layers)
    net.apply(lambda m: initialize_weights(m, gain=init_gain))
    return net
