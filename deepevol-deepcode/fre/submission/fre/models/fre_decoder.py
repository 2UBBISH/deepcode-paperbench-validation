"""FRE reward decoder.

Implements the decoder half of the Functional Reward Encoding (FRE) model:

    q_theta(eta(s^d) | s^d, z)

The paper (Section 4.1, "Practical Implementation") specifies:

    "The decoder q_theta(eta(s) | s, z) is implemented as a feedforward neural
    network. Crucially, the states sampled for decoding are different than those
    used for encoding. [...] the decoder independently predicts the reward for
    each state, given the shared latent encoding z. We train both the encoder
    and decoder networks jointly, minimizing mean-squared error between the
    predicted and true rewards under the decoding states."

Appendix A (Table 3) lists the decoder network as ``[512, 512, 512]`` and the
number of decoder states (reward pairs to decode) as ``K' = 8``.

The Addendum ("Additional Details on the FRE architecture") clarifies the input
layout:

    "There is no embedding step for the observation state passed to the decoder.
    The raw state and the z-vector are concatenated directly."

so the decoder input is ``[s^d, z]`` with the raw state (no learned projection)
concatenated to the 128-d latent ``z``.  Each decoder state is decoded
*independently* given the *shared* latent ``z``.

The decoder parametrizes a unit-variance Gaussian ``q_theta(eta(s^d) | s^d, z)``
(as used in the variational lower bound of Equation 6); with a fixed unit
variance, maximizing the log-likelihood is equivalent to minimizing the
mean-squared error between predicted and true rewards (plus a constant).
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Union

import torch
import torch.nn as nn


__all__ = ["FREDecoder", "build_mlp"]


def _get_activation(name: Union[str, Callable[[], nn.Module]]) -> nn.Module:
    """Resolve an activation name (or factory) into an activation module."""
    if callable(name):
        module = name()
        if isinstance(module, nn.Module):
            return module
        raise ValueError("Activation factory must return an nn.Module instance.")

    lowered = str(name).lower()
    if lowered == "relu":
        return nn.ReLU()
    if lowered in ("gelu",):
        return nn.GELU()
    if lowered in ("silu", "swish"):
        return nn.SiLU()
    if lowered == "elu":
        return nn.ELU()
    if lowered == "tanh":
        return nn.Tanh()
    if lowered == "mish":
        return nn.Mish()
    raise ValueError(f"Unsupported activation: {name}")


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: Union[str, Callable[[], nn.Module]] = "relu",
    output_activation: Optional[Union[str, Callable[[], nn.Module]]] = None,
    use_layer_norm: bool = False,
    init_gain: float = 1.0,
) -> nn.Sequential:
    """Build a plain feed-forward MLP.

    Layer ordering follows the paper's decoder description: linear -> (optional
    layer norm) -> activation for every hidden block, and a final linear head
    (optionally followed by ``output_activation``).
    """
    layers: List[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(_get_activation(activation))
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    if output_activation is not None:
        layers.append(_get_activation(output_activation))

    mlp = nn.Sequential(*layers)

    # Xavier/Glorot-style initialization with an optional gain.
    for module in mlp.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=init_gain)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    return mlp


class FREDecoder(nn.Module):
    """Feed-forward decoder predicting ``eta(s^d)`` from ``(s^d, z)``.

    Args:
        state_dim: Dimensionality of the raw (decoder) state.
        latent_dim: Dimensionality of the latent task encoding ``z`` (128 in FRE).
        hidden_dims: Hidden layer widths.  The paper uses ``[512, 512, 512]``.
        activation: Hidden activation name (default ``relu``).
        output_activation: Optional activation applied to the scalar output.
            ``None`` (default) gives an unbounded reward prediction, which is
            what the MSE objective of Equation 6 requires.
        use_layer_norm: Apply ``LayerNorm`` before hidden activations.
        output_dim: Number of outputs per state (1 for scalar rewards).

    Notes:
        The decoder is evaluated on *decoder states* that are disjoint from the
        encoder states.  Every decoding state is fed through the same MLP with
        the same shared ``z``; there is no interaction between decoder states.
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int = 128,
        hidden_dims: Sequence[int] = (512, 512, 512),
        activation: Union[str, Callable[[], nn.Module]] = "relu",
        output_activation: Optional[Union[str, Callable[[], nn.Module]]] = None,
        use_layer_norm: bool = False,
        output_dim: int = 1,
        init_gain: float = 1.0,
    ) -> None:
        super().__init__()

        self.state_dim = int(state_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.output_dim = int(output_dim)
        self.activation_name = activation

        self.net = build_mlp(
            input_dim=self.state_dim + self.latent_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            activation=activation,
            output_activation=output_activation,
            use_layer_norm=use_layer_norm,
            init_gain=init_gain,
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _broadcast_latent(z: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """Broadcast ``z`` so that it aligns with the trailing state axes.

        Accepts ``states`` of shape ``(..., K, state_dim)`` (or
        ``(..., state_dim)``) and ``z`` of shape ``(..., latent_dim)``.  The
        latent is expanded to the state's leading shape so the shared ``z`` is
        used for every decoder state.
        """
        if z.dim() == states.dim() - 1:
            z = z.unsqueeze(-2)
        if z.dim() != states.dim():
            raise ValueError(
                f"Latent tensor of shape {tuple(z.shape)} cannot be broadcast "
                f"against states of shape {tuple(states.shape)}."
            )
        target_shape = states.shape[:-1] + (z.shape[-1],)
        if tuple(z.shape) != tuple(target_shape):
            z = z.expand(*target_shape)
        return z

    def predict(
        self,
        states: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the reward for ``states`` given latent ``z``.

        Args:
            states: ``(..., K, state_dim)`` or ``(..., state_dim)`` raw states.
            z: ``(..., latent_dim)`` latent task encoding (broadcast over ``K``).

        Returns:
            ``(..., K)`` (or ``(...)`` when ``output_dim == 1``) predicted
            rewards.  With ``output_dim > 1`` the trailing dimension is kept.
        """
        single_state = states.dim() == z.dim()

        if single_state:
            states = states.unsqueeze(-2)  # (..., 1, state_dim)
        z_expanded = self._broadcast_latent(z, states)

        # Raw state concatenated directly with z (no embedding step).
        inputs = torch.cat([states, z_expanded], dim=-1)
        outputs = self.net(inputs)  # (..., K, output_dim)

        if self.output_dim == 1:
            outputs = outputs.squeeze(-1)
        if single_state:
            outputs = outputs.squeeze(-2)
        return outputs

    # ------------------------------------------------------------------
    # nn.Module API
    # ------------------------------------------------------------------
    def forward(
        self,
        states: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Alias for :meth:`predict`."""
        return self.predict(states, z)

    # ------------------------------------------------------------------
    # losses
    # ------------------------------------------------------------------
    def mse_loss(
        self,
        states: torch.Tensor,
        z: torch.Tensor,
        target_rewards: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Mean-squared-error decoder loss over decoding states.

        Corresponds to the ``log q_theta(eta(s^d) | s^d, z)`` term of Equation 6
        (up to a constant) with a unit-variance Gaussian decoder.

        Args:
            states: ``(..., K, state_dim)`` decoder states.
            z: ``(..., latent_dim)`` shared latent code.
            target_rewards: ``(..., K)`` true reward values ``eta(s^d)``.
            mask: Optional ``(..., K)`` boolean mask; ``True`` marks valid
                decoder states (invalid ones are excluded from both numerator
                and denominator).

        Returns:
            Scalar loss (mean over valid elements).
        """
        preds = self.predict(states, z)
        squared = (preds - target_rewards) ** 2
        if mask is None:
            return squared.mean()
        mask = mask.to(squared.dtype)
        denom = mask.sum().clamp(min=1.0)
        return (squared * mask).sum() / denom

    def log_likelihood(
        self,
        states: torch.Tensor,
        z: torch.Tensor,
        target_rewards: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Unit-variance Gaussian log-likelihood ``log q_theta(eta | s, z)``.

        Returns the log-likelihood summed over decoding states, which is the
        quantity appearing inside the expectation of Equation 6 (the paper
        writes ``sum_{k=1}^{K'} log q_theta(eta(s_k^d) | s_k^d, z)``).
        """
        preds = self.predict(states, z)
        squared = (preds - target_rewards) ** 2
        # log N(x; mu, 1) = -0.5 * (x - mu)^2 - 0.5 * log(2 * pi)
        log_prob = -0.5 * squared - 0.5 * math.log(2.0 * math.pi)
        if mask is not None:
            log_prob = log_prob * mask.to(log_prob.dtype)
        return log_prob.sum()

    # ------------------------------------------------------------------
    def extra_repr(self) -> str:
        return (
            f"state_dim={self.state_dim}, latent_dim={self.latent_dim}, "
            f"hidden_dims={self.hidden_dims}, output_dim={self.output_dim}"
        )
