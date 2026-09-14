"""Unit tests for score-network architectures in ``npse/src/networks.py``.

These tests validate output shapes, standardization behaviour, time-embedding
formulas, and helper utilities against the specifications in the NPSE paper:

- Theta/x embeddings: 3-layer MLPs with 256 hidden units and SiLU activations.
- Time embedding: 64 dimensions, sine for the first 32 terms and cosine for the
  last 32 terms, using ``10000^((i-1)/31)``.
- Score MLP: concatenates ``[theta_emb, x_emb, t_emb]`` and outputs ``theta_dim``
  values.
"""

from __future__ import annotations

import math

import pytest
import torch

from npse.src.networks import (
    EmbeddingMLP,
    MLP,
    PriorScoreNetwork,
    ScoreNetwork,
    SinusoidalTimeEmbedding,
    Standardizer,
    _hidden_size,
    compute_standardization_stats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_score_network(theta_dim: int = 4, x_dim: int = 6) -> ScoreNetwork:
    return ScoreNetwork(theta_dim=theta_dim, x_dim=x_dim, hidden_dim=64)


def _make_prior_score_network(theta_dim: int = 5) -> PriorScoreNetwork:
    return PriorScoreNetwork(theta_dim=theta_dim, hidden_dim=64)


# ---------------------------------------------------------------------------
# _hidden_size
# ---------------------------------------------------------------------------
def test_hidden_size_minimum_and_scaling() -> None:
    assert _hidden_size(0) == 30
    assert _hidden_size(1) == 30
    assert _hidden_size(2) == 30
    assert _hidden_size(7) == 30  # 4 * 7 = 28 -> 30
    assert _hidden_size(8) == 32  # 4 * 8 = 32 -> 32
    assert _hidden_size(10) == 40
    assert _hidden_size(31) == 124


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------
def test_mlp_output_shape() -> None:
    mlp = MLP(in_dim=5, out_dim=3, hidden_dim=32, num_layers=3)
    x = torch.randn(17, 5)
    y = mlp(x)
    assert y.shape == (17, 3)


def test_mlp_single_sample() -> None:
    mlp = MLP(in_dim=2, out_dim=1, hidden_dim=16, num_layers=3)
    y = mlp(torch.randn(2))
    assert y.shape == (1,)


def test_mlp_deterministic_eval_mode() -> None:
    mlp = MLP(in_dim=3, out_dim=2)
    mlp.eval()
    x = torch.randn(4, 3)
    with torch.no_grad():
        y1 = mlp(x)
        y2 = mlp(x)
    assert torch.allclose(y1, y2)


# ---------------------------------------------------------------------------
# EmbeddingMLP
# ---------------------------------------------------------------------------
def test_embedding_mlp_output_shape() -> None:
    emb = EmbeddingMLP(in_dim=7, out_dim=12, hidden_dim=32)
    x = torch.randn(9, 7)
    y = emb(x)
    assert y.shape == (9, 12)


def test_embedding_mlp_single_sample() -> None:
    emb = EmbeddingMLP(in_dim=3, out_dim=5, hidden_dim=32)
    y = emb(torch.randn(3))
    assert y.shape == (5,)


# ---------------------------------------------------------------------------
# SinusoidalTimeEmbedding
# ---------------------------------------------------------------------------
def test_time_embedding_shape_batched() -> None:
    emb = SinusoidalTimeEmbedding(embedding_dim=64)
    t = torch.rand(11)
    out = emb(t)
    assert out.shape == (11, 64)


def test_time_embedding_shape_scalar_tensor() -> None:
    emb = SinusoidalTimeEmbedding(embedding_dim=64)
    out = emb(torch.tensor(0.3))
    assert out.numel() == 64


def test_time_embedding_formula() -> None:
    emb = SinusoidalTimeEmbedding(embedding_dim=64)
    t = torch.tensor([0.25, 1.0])
    out = emb(t)
    assert out.shape == (2, 64)

    expected = torch.empty(2, 64)
    # First 32 entries are sine terms.
    for j in range(32):
        expected[:, j] = torch.sin(t / (10000.0 ** (j / 31.0)))
    # Last 32 entries are cosine terms.
    for j in range(32):
        expected[:, 32 + j] = torch.cos(t / (10000.0 ** (j / 31.0)))

    assert torch.allclose(out, expected, atol=1e-6)


def test_time_embedding_finite_and_dtype() -> None:
    emb = SinusoidalTimeEmbedding(embedding_dim=64)
    t = torch.rand(5, dtype=torch.float64)
    out = emb(t)
    assert out.dtype == torch.float64
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Standardizer
# ---------------------------------------------------------------------------
def test_standardizer_affine_transform() -> None:
    stdizer = Standardizer()
    stdizer.set(mean=torch.tensor([1.0, 2.0]), std=torch.tensor([2.0, 4.0]))
    x = torch.tensor([[1.0, 2.0], [3.0, 6.0]])
    y = stdizer(x)
    expected = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    assert torch.allclose(y, expected, atol=1e-6)


def test_standardizer_zero_std_is_clamped() -> None:
    stdizer = Standardizer()
    stdizer.set(mean=torch.zeros(3), std=torch.zeros(3))
    x = torch.randn(6, 3)
    y = stdizer(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_standardizer_shape_preserved() -> None:
    stdizer = Standardizer()
    stdizer.set(mean=torch.zeros(2), std=torch.ones(2))
    x = torch.randn(8, 2)
    y = stdizer(x)
    assert y.shape == x.shape


# ---------------------------------------------------------------------------
# compute_standardization_stats
# ---------------------------------------------------------------------------
def test_compute_standardization_stats_mean() -> None:
    samples = torch.tensor([[0.0, 2.0], [2.0, 4.0], [4.0, 6.0]])
    mean, std = compute_standardization_stats(samples)
    assert mean.shape == (2,)
    assert std.shape == (2,)
    assert torch.allclose(mean, torch.tensor([2.0, 4.0]))


def test_compute_standardization_stats_constant_columns() -> None:
    samples = torch.ones(10, 4)
    mean, std = compute_standardization_stats(samples)
    assert torch.allclose(mean, torch.ones(4))
    assert torch.allclose(std, torch.zeros(4), atol=1e-7)


def test_compute_standardization_stats_nonnegative_std() -> None:
    samples = torch.randn(50, 5)
    mean, std = compute_standardization_stats(samples)
    assert torch.all(std >= 0.0)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(std).all()


# ---------------------------------------------------------------------------
# ScoreNetwork
# ---------------------------------------------------------------------------
def test_score_network_output_shape() -> None:
    net = _make_score_network(theta_dim=4, x_dim=6)
    theta = torch.randn(8, 4)
    x = torch.randn(8, 6)
    t = torch.rand(8)
    score = net(theta, x, t)
    assert score.shape == (8, 4)


def test_score_network_single_sample() -> None:
    net = _make_score_network(theta_dim=2, x_dim=3)
    score = net(torch.randn(2), torch.randn(3), torch.tensor(0.5))
    assert score.shape == (2,)


def test_score_network_set_standardization() -> None:
    net = _make_score_network(theta_dim=4, x_dim=6)
    net.set_standardization(
        theta_mean=torch.zeros(4),
        theta_std=torch.ones(4),
        x_mean=torch.zeros(6),
        x_std=torch.ones(6),
    )
    theta = torch.randn(8, 4)
    x = torch.randn(8, 6)
    t = torch.rand(8)
    score = net(theta, x, t)
    assert score.shape == (8, 4)
    assert torch.isfinite(score).all()


def test_score_network_deterministic_eval_mode() -> None:
    net = _make_score_network(theta_dim=3, x_dim=4)
    net.eval()
    theta = torch.randn(5, 3)
    x = torch.randn(5, 4)
    t = torch.rand(5)
    with torch.no_grad():
        s1 = net(theta, x, t)
        s2 = net(theta, x, t)
    assert torch.allclose(s1, s2)


# ---------------------------------------------------------------------------
# PriorScoreNetwork
# ---------------------------------------------------------------------------
def test_prior_score_network_output_shape() -> None:
    net = _make_prior_score_network(theta_dim=5)
    theta = torch.randn(9, 5)
    t = torch.rand(9)
    score = net(theta, t)
    assert score.shape == (9, 5)


def test_prior_score_network_single_sample() -> None:
    net = _make_prior_score_network(theta_dim=3)
    score = net(torch.randn(3), torch.tensor(0.7))
    assert score.shape == (3,)


def test_prior_score_network_set_standardization() -> None:
    net = _make_prior_score_network(theta_dim=4)
    net.set_standardization(theta_mean=torch.zeros(4), theta_std=torch.ones(4))
    theta = torch.randn(7, 4)
    t = torch.rand(7)
    score = net(theta, t)
    assert score.shape == (7, 4)
    assert torch.isfinite(score).all()


# ---------------------------------------------------------------------------
# Manual runner (allows execution without pytest)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        test_hidden_size_minimum_and_scaling,
        test_mlp_output_shape,
        test_mlp_single_sample,
        test_mlp_deterministic_eval_mode,
        test_embedding_mlp_output_shape,
        test_embedding_mlp_single_sample,
        test_time_embedding_shape_batched,
        test_time_embedding_shape_scalar_tensor,
        test_time_embedding_formula,
        test_time_embedding_finite_and_dtype,
        test_standardizer_affine_transform,
        test_standardizer_zero_std_is_clamped,
        test_standardizer_shape_preserved,
        test_compute_standardization_stats_mean,
        test_compute_standardization_stats_constant_columns,
        test_compute_standardization_stats_nonnegative_std,
        test_score_network_output_shape,
        test_score_network_single_sample,
        test_score_network_set_standardization,
        test_score_network_deterministic_eval_mode,
        test_prior_score_network_output_shape,
        test_prior_score_network_single_sample,
        test_prior_score_network_set_standardization,
    ]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {test.__name__}: {exc}")
    if failed:
        raise SystemExit(f"{failed} test(s) failed")
    print("All network tests passed.")
