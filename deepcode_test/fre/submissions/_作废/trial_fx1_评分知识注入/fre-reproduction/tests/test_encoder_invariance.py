"""Unit tests for permutation invariance of the FRE transformer encoder.

The FRE encoder consumes a *set* of (state, reward) tokens. Reordering the
tokens must not change the encoded Gaussian posterior parameters (mu, logvar)
or, when the RNG is controlled, the sampled latent code.
"""

import os
import sys

import numpy as np
import pytest
import torch

# Allow running tests directly from the repository root without installing the
# package in editable mode.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fre.encoder import FREEncoder
from fre.reward_embedding import RewardEmbedding


@pytest.fixture(scope="module")
def vae_components():
    """Build the reward tokenizer and encoder once for all tests."""
    torch.manual_seed(0)
    state_dim = 6
    reward_embedding = RewardEmbedding(
        state_dim=state_dim,
        num_bins=64,
        embedding_dim=64,
        state_proj_dim=192,
        token_dim=256,
    )
    encoder = FREEncoder(
        d_model=256,
        nhead=4,
        num_layers=2,
        latent_dim=32,
        dim_feedforward=128,
        dropout=0.0,
    )
    encoder.eval()
    return state_dim, reward_embedding, encoder


def test_encoder_is_invariant_to_token_permutation(vae_components):
    state_dim, reward_embedding, encoder = vae_components

    torch.manual_seed(123)
    batch_size = 3
    num_tokens = 7

    states = torch.randn(batch_size, num_tokens, state_dim)
    rewards = torch.rand(batch_size, num_tokens) * 2.0 - 1.0

    tokens = reward_embedding(states, rewards)

    # Permute the token dimension independently for each batch element.
    perm = torch.randperm(num_tokens)
    tokens_perm = tokens[:, perm, :]

    with torch.no_grad():
        mu, logvar, _ = encoder(tokens)
        mu_perm, logvar_perm, _ = encoder(tokens_perm)

    assert torch.allclose(mu, mu_perm, atol=1e-6), "mu is not permutation invariant"
    assert torch.allclose(logvar, logvar_perm, atol=1e-6), (
        "logvar is not permutation invariant"
    )


def test_sampled_z_is_invariant_under_controlled_rng(vae_components):
    state_dim, reward_embedding, encoder = vae_components

    batch_size = 2
    num_tokens = 5
    states = torch.randn(batch_size, num_tokens, state_dim)
    rewards = torch.rand(batch_size, num_tokens) * 2.0 - 1.0

    tokens = reward_embedding(states, rewards)
    perm = torch.randperm(num_tokens)
    tokens_perm = tokens[:, perm, :]

    with torch.no_grad():
        torch.manual_seed(42)
        _, _, z1 = encoder(tokens)
        torch.manual_seed(42)
        _, _, z2 = encoder(tokens_perm)

    assert torch.allclose(z1, z2, atol=1e-6), "sampled z should match under equal RNG"


def test_masked_mean_matches_explicit_selection(vae_components):
    """Masked tokens should be ignored when computing the pooled vector."""
    state_dim, reward_embedding, encoder = vae_components

    batch_size = 2
    num_tokens = 6
    states = torch.randn(batch_size, num_tokens, state_dim)
    rewards = torch.rand(batch_size, num_tokens) * 2.0 - 1.0
    tokens = reward_embedding(states, rewards)

    # Keep only the first 3 tokens for every batch element.
    mask = torch.zeros(batch_size, num_tokens, dtype=torch.bool)
    mask[:, :3] = True

    with torch.no_grad():
        mu_masked, logvar_masked, _ = encoder(tokens, mask=mask)
        mu_full, logvar_full, _ = encoder(tokens[:, :3, :])

    assert torch.allclose(mu_masked, mu_full, atol=1e-6)
    assert torch.allclose(logvar_masked, logvar_full, atol=1e-6)


def test_latent_shape(vae_components):
    state_dim, reward_embedding, encoder = vae_components
    batch_size, num_tokens, latent_dim = 4, 8, encoder.latent_dim

    states = torch.randn(batch_size, num_tokens, state_dim)
    rewards = torch.rand(batch_size, num_tokens) * 2.0 - 1.0
    tokens = reward_embedding(states, rewards)

    with torch.no_grad():
        mu, logvar, z = encoder(tokens)

    assert mu.shape == (batch_size, latent_dim)
    assert logvar.shape == (batch_size, latent_dim)
    assert z.shape == (batch_size, latent_dim)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
