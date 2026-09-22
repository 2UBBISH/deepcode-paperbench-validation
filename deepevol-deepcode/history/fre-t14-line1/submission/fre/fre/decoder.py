"""FRE decoder q_theta(eta(s) | s, z).

Paper (Section 4.1, "Practical Implementation"):
    "The decoder q_theta(eta(s) | s, z) is implemented as a feedforward neural
    network. Crucially, the states sampled for decoding are different than those
    used for encoding. The encoding network makes use of the entire set of
    (s_{1..K}, eta(s_{1..K})) pairs, whereas the decoder independently predicts
    the reward for each state, given the shared latent encoding z. We train both
    the encoder and decoder networks jointly, minimizing mean-squared error
    between the predicted and true rewards under the decoding states."

Appendix A (Table 3): "Decoder Network Layers [512, 512, 512]".

Addendum ("Additional Details on the FRE architecture"):
    - "There is no embedding step for the observation state passed to the
      decoder. The raw state and the z-vector are concatenated directly."
    - K' (Reward Pairs to Decode) = 8.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal

__all__ = ["Decoder", "decoder_mse_loss"]


def _make_activation(name: str) -> nn.Module:
    """Resolve an activation function by name.

    The paper does not state the decoder activation; Gelu is used by default
    (a sensible default that matches the encoder's chosen activation).
    """
    name = (name or "gelu").lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu" or name == "swish":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "elu":
        return nn.ELU()
    if name == "mish":
        return nn.Mish()
    raise ValueError(f"Unsupported activation: {name!r}")


class Decoder(nn.Module):
    """Feedforward MLP decoder q_theta(eta(s) | s, z).

    The decoder consumes the *raw* state (no embedding) concatenated with the
    128-dimensional latent task vector ``z``, and outputs a scalar reward
    prediction for each provided state.  The same ``z`` is shared across all
    K' decoding states; states are decoded independently (pointwise MLP).

    Parameters
    ----------
    state_dim:
        Dimensionality of the raw observation.
    latent_dim:
        Dimensionality of the latent task vector ``z`` (128 in the paper).
    hidden_layers:
        MLP hidden widths; paper uses ``(512, 512, 512)``.
    activation:
        Hidden activation name (paper silent -> "gelu" default).
    output_dim:
        Number of outputs per state.  1 for scalar reward prediction (MSE).
    predict_log_std:
        If True a second output head predicts a log standard deviation,
        enabling an (optional) Gaussian likelihood q_theta(eta(s) | s, z);
        the paper's objective is MSE so this defaults to False.
    log_std_min / log_std_max:
        Clamping range for the optional log-std head.
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int = 128,
        hidden_layers: Sequence[int] = (512, 512, 512),
        activation: str = "gelu",
        output_dim: int = 1,
        predict_log_std: bool = False,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        layernorm: bool = False,
    ) -> None:
        super().__init__()
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")

        self.state_dim = int(state_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_layers = tuple(int(h) for h in hidden_layers)
        self.activation_name = activation
        self.output_dim = int(output_dim)
        self.predict_log_std = bool(predict_log_std)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        # Input = raw state concatenated with z (no observation embedding).
        input_dim = self.state_dim + self.latent_dim

        layers = []
        prev = input_dim
        for hidden in self.hidden_layers:
            layers.append(nn.Linear(prev, hidden))
            if layernorm:
                layers.append(nn.LayerNorm(hidden))
            layers.append(_make_activation(self.activation_name))
            prev = hidden
        layers.append(nn.Linear(prev, self.output_dim))
        self.net = nn.Sequential(*layers)

        if self.predict_log_std:
            log_std_head = nn.Linear(prev, self.output_dim)
            nn.init.constant_(log_std_head.bias, 0.0)
            nn.init.normal_(log_std_head.weight, std=0.01)
            self.log_std_head: Optional[nn.Module] = log_std_head
        else:
            self.log_std_head = None

        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _prepare_inputs(self, states: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Concatenate raw states with z, broadcasting z over the state axis.

        Accepts states of shape ``(B, K', state_dim)`` or ``(K', state_dim)``
        and z of shape ``(B, latent_dim)`` or ``(latent_dim,)``.
        """
        if states.dim() == 2:
            states = states.unsqueeze(0)
        if states.dim() != 3:
            raise ValueError(
                f"states must have shape (B, K', state_dim) or (K', state_dim); "
                f"got {tuple(states.shape)}"
            )
        if states.shape[-1] != self.state_dim:
            raise ValueError(
                f"expected final state dim {self.state_dim}, got {states.shape[-1]}"
            )

        if z.dim() == 1:
            z = z.unsqueeze(0)
        if z.dim() != 2 or z.shape[-1] != self.latent_dim:
            raise ValueError(
                f"z must have shape (B, {self.latent_dim}); got {tuple(z.shape)}"
            )

        batch, num_states = states.shape[0], states.shape[1]
        if z.shape[0] == 1 and batch != 1:
            # Broadcast a single shared z across a batch of state sets.
            z = z.expand(batch, self.latent_dim)
        elif z.shape[0] != batch:
            raise ValueError(
                f"batch mismatch between states ({batch}) and z ({z.shape[0]})"
            )

        z_expanded = z.unsqueeze(1).expand(batch, num_states, self.latent_dim)
        return torch.cat([states, z_expanded], dim=-1)

    def forward(self, states: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Predict rewards for each decoding state.

        Returns a tensor of shape ``(B, K')`` (squeezing the scalar output dim).
        """
        x = self._prepare_inputs(states, z)
        out = self.net(x)
        return out.squeeze(-1)

    def forward_dist(self, states: torch.Tensor, z: torch.Tensor) -> Normal:
        """Gaussian decoder q_theta(eta(s) | s, z) (optional, non-paper default).

        Used only when ``predict_log_std=True``; the paper minimizes MSE so the
        point-prediction ``forward`` is the canonical interface.
        """
        x = self._prepare_inputs(states, z)
        mean = self.net(x)
        if self.log_std_head is None:
            raise RuntimeError(
                "forward_dist requires predict_log_std=True at construction"
            )
        log_std = self.log_std_head(x).clamp(self.log_std_min, self.log_std_max)
        return Normal(mean, torch.exp(log_std))

    def loss(
        self,
        states: torch.Tensor,
        z: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Mean-squared error between predicted and true rewards.

        Implements the reconstruction term of the information-bottleneck
        objective (Eq. 6): ``sum_k log q_theta(eta(s_k^d) | s_k^d, z)`` with a
        Gaussian unit-variance likelihood, i.e. MSE up to a constant.
        """
        pred = self.forward(states, z)
        return decoder_mse_loss(pred, targets, reduction=reduction)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        config,
        state_dim: int,
        **overrides,
    ) -> "Decoder":
        """Build a decoder from a ``fre.config.default.Config``-like object."""
        kwargs = dict(
            state_dim=state_dim,
            latent_dim=getattr(config, "latent_dim", 128),
            hidden_layers=getattr(config, "decoder_layers", (512, 512, 512)),
            activation=getattr(config, "decoder_activation", "gelu"),
            output_dim=getattr(config, "decoder_output_dim", 1),
            predict_log_std=getattr(config, "decoder_predict_log_std", False),
            log_std_min=getattr(config, "log_std_min", -5.0),
            log_std_max=getattr(config, "log_std_max", 2.0),
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def num_decoder_samples(self) -> int:
        """K': number of decoding states per reward function (8 in the paper)."""
        return int(getattr(self, "_num_decoder_samples", 8))

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extra_repr(self) -> str:
        return (
            f"state_dim={self.state_dim}, latent_dim={self.latent_dim}, "
            f"hidden_layers={self.hidden_layers}, activation={self.activation_name}, "
            f"output_dim={self.output_dim}, predict_log_std={self.predict_log_std}"
        )


def decoder_mse_loss(
    predicted: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """MSE between predicted and true rewards (Eq. 6 reconstruction term).

    ``predicted`` and ``targets`` broadcast to a common shape; typically both
    are ``(batch, K')``.
    """
    targets = targets.reshape(predicted.shape)
    return nn.functional.mse_loss(predicted, targets, reduction=reduction)
