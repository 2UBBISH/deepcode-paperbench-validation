"""FRE decoder: q_theta(eta(s) | s, z).

From the paper (Sec 4.1):

    "The decoder takes as input the concatenation of the raw environment state and
     the latent task embedding z, and predicts the reward of that state. We use an
     MLP with hidden sizes [512, 512, 512]."

Notes / implementation decisions (matching the reproduction plan + addendum):
  * There is **no observation-embedding step** in the decoder (unlike the encoder,
    which learns a 64-d projection). The *raw* (already normalized / preprocessed)
    state vector is concatenated directly with z.
  * The decoder predicts each decoder state *independently* given the shared z.
    In practice this means a single f(s, z) MLP applied to a set of K'=8 decoder
    states, all conditioned on the SAME z (the encoder context uses K=32 states,
    disjoint from the decoder states).
  * The reward targets produced by the reward prior live in [-1, 1]; we optionally
    squash the output with tanh to keep the predicted reward field in the same range.
    This can be disabled (``output_activation=None``) for an unconstrained scalar head.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn


def _build_mlp(
    in_dim: int,
    hidden_sizes: Sequence[int],
    out_dim: int,
    activation: str = "relu",
    output_activation: Optional[str] = "tanh",
) -> nn.Sequential:
    """Stack of Linear -> activation layers followed by a final Linear (+ optional act)."""
    act_cls = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
        "mish": nn.Mish,
    }
    if activation not in act_cls:
        raise ValueError(f"Unsupported activation '{activation}'")

    layers = []
    prev = in_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(act_cls[activation]())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    if output_activation is not None:
        if output_activation not in act_cls:
            raise ValueError(f"Unsupported output activation '{output_activation}'")
        layers.append(act_cls[output_activation]())
    return nn.Sequential(*layers)


class FREDecoder(nn.Module):
    """Predicts reward eta(s) for a batch of states conditioned on latent z.

    Args:
        state_dim: dimensionality of the *raw* environment state.
        latent_dim: dimensionality of the task embedding z.
        hidden_sizes: MLP hidden sizes. Paper uses [512, 512, 512].
        activation: hidden activation (paper/reference uses ReLU).
        output_activation: applied to the scalar reward output. Default "tanh"
            keeps predictions within the [-1, 1] range of the reward prior.
            Use None for an unconstrained head.
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int = 128,
        hidden_sizes: Sequence[int] = (512, 512, 512),
        activation: str = "relu",
        output_activation: Optional[str] = "tanh",
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self.input_dim = self.state_dim + self.latent_dim

        self.net = _build_mlp(
            in_dim=self.input_dim,
            hidden_sizes=self.hidden_sizes,
            out_dim=1,
            activation=activation,
            output_activation=output_activation,
        )

    # ------------------------------------------------------------------ #
    # Core forward
    # ------------------------------------------------------------------ #
    def forward(self, states: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Predict rewards.

        Supported shapes (all handled by ``torch.cat`` broadcasting semantics):

        * states: (..., state_dim), z: (latent_dim,) or (..., latent_dim)
          -> returns (..., 1) / (...,)

        The common training/eval cases are:
          * ``states`` (K', state_dim), ``z`` (latent_dim,)  -> (K', 1,)
          * ``states`` (B, K', state_dim), ``z`` (B, latent_dim) -> (B, K', 1)
          * ``states`` (B, K', state_dim), ``z`` (B, 1, latent_dim) -> (B, K', 1)
        """
        return self.predict(states, z)

    def predict(self, states: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Semantically-identical alias of ``forward`` (explicit reward naming)."""
        states = torch.as_tensor(states)
        z = torch.as_tensor(z)
        dtype = self.net[0].weight.dtype
        device = self.net[0].weight.device
        states = states.to(device=device, dtype=dtype)
        z = z.to(device=device, dtype=dtype)

        # Broadcast z up to the state batch shape if needed.
        z = _align_latent(z, states, self.latent_dim)

        x = torch.cat([states, z], dim=-1)
        out = self.net(x)
        return out.squeeze(-1)

    def predict_for_context(
        self,
        decoder_states: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the shared z to a set of decoder states (K').

        Args:
            decoder_states: (B, K', state_dim) or (K', state_dim).
            z: (B, latent_dim) or (latent_dim,).
        Returns:
            (B, K') or (K',) predicted rewards.
        """
        if decoder_states.dim() == 2 and z.dim() == 2:
            # (K', S) with (B, Z) -> treat states as shared across batch.
            decoder_states = decoder_states.unsqueeze(0)  # (1, K', S)
            out = self.predict(decoder_states, z.unsqueeze(1))  # (B, K')
            return out
        if decoder_states.dim() == 2 and z.dim() == 1:
            return self.predict(decoder_states, z)
        if decoder_states.dim() == 3 and z.dim() == 2:
            return self.predict(decoder_states, z.unsqueeze(1))
        if decoder_states.dim() == 3 and z.dim() == 3:
            return self.predict(decoder_states, z)
        raise ValueError(
            f"Unsupported shapes: decoder_states {tuple(decoder_states.shape)}, "
            f"z {tuple(z.shape)}"
        )

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"state_dim={self.state_dim}, latent_dim={self.latent_dim}, "
            f"hidden_sizes={self.hidden_sizes}"
        )


def _align_latent(z: torch.Tensor, states: torch.Tensor, latent_dim: int) -> torch.Tensor:
    """Expand/broadcast z so that ``cat([states, z], dim=-1)`` is well formed."""
    # Scalar-batch latents: (latent_dim,) -> (1, ..., latent_dim)
    if z.dim() == 1:
        # (Z,) -> (1, 1, ..., Z) with as many singleton dims as states has leading dims.
        shape = (1,) * (states.dim() - 1) + (latent_dim,)
        z = z.view(*shape)
    elif z.dim() == states.dim() - 1:
        # e.g. states (B, K, S) with z (B, Z) -> (B, 1, Z)
        z = z.unsqueeze(1)
    return z.expand(*states.shape[:-1], latent_dim)


class FREModel(nn.Module):
    """Convenience container bundling the FRE encoder + decoder.

    Kept deliberately thin so that ``vae_loss.py`` / training drivers can treat the
    pair as a single module for checkpointing while still accessing the individual
    components (``model.encoder`` / ``model.decoder``).
    """

    def __init__(self, encoder: nn.Module, decoder: FREDecoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    @property
    def latent_dim(self) -> int:
        return self.decoder.latent_dim

    def forward(
        self,
        context_states: torch.Tensor,
        context_rewards: torch.Tensor,
        decoder_states: torch.Tensor,
    ):
        """Returns (pred_rewards, mu, log_sigma, z)."""
        z, mu, log_sigma = self.encoder.sample(context_states, context_rewards)
        pred = self.decoder.predict_for_context(decoder_states, z)
        return pred, mu, log_sigma, z


__all__ = ["FREDecoder", "FREModel"]
