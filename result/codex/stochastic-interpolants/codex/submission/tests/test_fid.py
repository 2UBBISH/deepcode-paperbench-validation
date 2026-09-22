"""FID utilities (Tables 2 and 3) - checked without downloading Inception."""

import torch

from si_couplings.fid import compute_statistics, frechet_distance


def test_frechet_distance_analytic_gaussians():
    torch.manual_seed(0)
    d = 8
    mu = torch.zeros(d, dtype=torch.float64)
    A = torch.randn(d, d, dtype=torch.float64)
    s1 = A @ A.T / d + torch.eye(d, dtype=torch.float64)
    B = torch.randn(d, d, dtype=torch.float64)
    s2 = B @ B.T / d + torch.eye(d, dtype=torch.float64)
    assert frechet_distance({"mu": mu, "sigma": s1}, {"mu": mu, "sigma": s1}) < 1e-8
    far = frechet_distance({"mu": mu, "sigma": s1}, {"mu": mu + 1.0, "sigma": s2})
    assert far > 0.5


def test_compute_statistics_recovers_moments():
    torch.manual_seed(0)
    d, n = 4, 20_000
    feats = torch.randn(n, d)

    def feature_fn(x):  # the "extractor" is the identity here
        return x

    stats = compute_statistics(feature_fn, [feats], n_samples=n)
    assert stats["n"] == n
    assert torch.allclose(stats["mu"].float(), feats.mean(0), atol=1e-5)
    assert torch.allclose(stats["sigma"].float(), torch.cov(feats.T), atol=1e-3)


def test_compute_statistics_respects_n_samples():
    feats = torch.randn(100, 3)
    stats = compute_statistics(lambda x: x, [feats], n_samples=10)
    assert stats["n"] == 10
