"""Unit tests for FRE reward-decoder reconstruction and VAE training dynamics.

These tests do not require MuJoCo, D4RL, or ExORL. They build a small
:class:`FREVAE` on CPU, train it on a smooth linear reward function, and
verify that:

1. The MLP decoder broadcasts latent codes across leading state dimensions.
2. Reconstruction MSE decreases during VAE training.
3. The KL-divergence term stays finite and bounded.
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

# Allow running tests directly from the repository root or via pytest.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from fre.decoder import RewardDecoder  # noqa: E402
from fre.fre_vae import FREVAE  # noqa: E402


def _to_float(value) -> float:
    """Convert a torch scalar, numpy scalar, or python number to float."""
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _get_metric(metrics: dict, names) -> float | None:
    """Look up a metric from a training-step result dict using flexible keys."""
    if not isinstance(metrics, dict):
        return None
    for key in names:
        if key in metrics:
            return _to_float(metrics[key])
    # Some FREVAE versions nest losses under a "losses" sub-dict.
    nested = metrics.get("losses")
    if isinstance(nested, dict):
        for key in names:
            if key in nested:
                return _to_float(nested[key])
    return None


def _smooth_linear_reward(weights: torch.Tensor):
    """Return a deterministic, smooth (unclipped) linear reward callable.

    The weights are chosen to keep outputs well inside [-1, 1] so the decoder
    learns a clean linear mapping without clipping artifacts.
    """

    def reward_fn(states: torch.Tensor) -> torch.Tensor:
        if not isinstance(states, torch.Tensor):
            states = torch.as_tensor(states, dtype=torch.float32)
        return torch.clamp(states.to(weights.dtype) @ weights, -1.0, 1.0)

    return reward_fn


def _make_small_vae(state_dim: int) -> FREVAE:
    """Build a small, fast CPU FREVAE suitable for smoke tests."""
    return FREVAE(
        state_dim=state_dim,
        latent_dim=16,
        d_model=64,
        nhead=4,
        num_layers=2,
        reward_bins=16,
        embedding_dim=16,
        decoder_hidden=(64, 64),
        beta=1.0,
        device="cpu",
    )


def test_decoder_broadcasts_latent_over_states() -> None:
    """The decoder must accept z with fewer leading dims than states."""
    torch.manual_seed(0)
    state_dim = 6
    latent_dim = 16
    decoder = RewardDecoder(
        state_dim=state_dim,
        latent_dim=latent_dim,
        hidden_dims=(32, 32),
        activation="relu",
    )

    # states: (batch, num_states, state_dim); z: (batch, latent_dim)
    states = torch.randn(4, 7, state_dim)
    z = torch.randn(4, latent_dim)
    out = decoder(states, z)
    assert out.shape == (4, 7, 1)

    # A single shared latent should also broadcast across a 2-D state batch.
    states_2d = torch.randn(10, state_dim)
    z_single = torch.randn(latent_dim)
    out_2d = decoder(states_2d, z_single)
    assert out_2d.shape == (10, 1)


def test_fre_vae_reconstruction_decreases_on_linear_reward() -> None:
    """Overfitting a fixed linear reward should lower decoder MSE."""
    torch.manual_seed(0)
    state_dim = 6
    vae = _make_small_vae(state_dim)

    generator = torch.Generator().manual_seed(1234)
    encoder_states = torch.rand(64, state_dim, generator=generator) * 0.8 - 0.4
    decoder_states = torch.rand(128, state_dim, generator=generator) * 0.8 - 0.4

    weights = torch.linspace(-0.05, 0.05, state_dim)
    reward_fn = _smooth_linear_reward(weights)
    encoder_rewards = reward_fn(encoder_states)
    decoder_rewards = reward_fn(decoder_states)

    optimizer = vae.configure_optimizer(lr=3e-3)

    initial_recon = None
    last_recon = None
    last_kl = None

    for _step in range(500):
        metrics = vae.training_step(
            encoder_states,
            encoder_rewards,
            decoder_states,
            decoder_rewards,
            optimizer,
        )
        recon = _get_metric(
            metrics,
            ["recon_loss", "reconstruction_loss", "reconstruction", "recon", "mse"],
        )
        kl = _get_metric(metrics, ["kl_loss", "kl_divergence", "kl"])

        if recon is not None:
            recon_value = _to_float(recon)
            assert math.isfinite(recon_value)
            if initial_recon is None:
                initial_recon = recon_value
            last_recon = recon_value

        if kl is not None:
            last_kl = _to_float(kl)
            assert math.isfinite(last_kl)
            assert last_kl < 100.0, f"KL exploded to {last_kl}"

    assert initial_recon is not None, "training_step did not report reconstruction loss"
    assert last_recon is not None, "training_step did not report final reconstruction loss"
    assert last_recon < initial_recon, (
        f"Reconstruction did not decrease: {initial_recon:.6f} -> {last_recon:.6f}"
    )

    # After training, decoding a newly encoded latent should approximately
    # recover the true linear rewards.
    with torch.no_grad():
        mu, logvar, z = vae.encode(decoder_states, decoder_rewards)
        predicted = vae.decode_reward(decoder_states, z).squeeze(-1)
        final_mse = float(F.mse_loss(predicted, decoder_rewards).item())
        assert final_mse < initial_recon + 1e-4, (
            f"Decoded rewards diverged: initial={initial_recon:.6f}, "
            f"final_decode={final_mse:.6f}"
        )
        assert last_kl is None or math.isfinite(last_kl)


def test_fre_vae_kl_remains_bounded_during_training() -> None:
    """KL should remain finite/bounded while the VAE fits a reward family."""
    torch.manual_seed(1)
    state_dim = 4
    vae = _make_small_vae(state_dim)

    generator = torch.Generator().manual_seed(42)
    encoder_states = torch.rand(32, state_dim, generator=generator) * 0.8 - 0.4
    decoder_states = torch.rand(64, state_dim, generator=generator) * 0.8 - 0.4

    weights = torch.linspace(-0.08, 0.08, state_dim)
    reward_fn = _smooth_linear_reward(weights)
    encoder_rewards = reward_fn(encoder_states)
    decoder_rewards = reward_fn(decoder_states)

    optimizer = vae.configure_optimizer(lr=1e-3)
    observed_kl = []

    for _step in range(300):
        metrics = vae.training_step(
            encoder_states,
            encoder_rewards,
            decoder_states,
            decoder_rewards,
            optimizer,
        )
        kl = _get_metric(metrics, ["kl_loss", "kl_divergence", "kl"])
        if kl is not None:
            value = _to_float(kl)
            assert math.isfinite(value)
            observed_kl.append(value)

    assert observed_kl, "training_step did not report KL loss"
    assert max(observed_kl) < 100.0
    assert min(observed_kl) >= 0.0
