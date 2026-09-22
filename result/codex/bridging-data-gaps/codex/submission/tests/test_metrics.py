"""Checks of the evaluation metrics of Section 5.2."""

from __future__ import annotations

import numpy as np
import torch

from dpms_ant.metrics import frechet_distance, intra_lpips


class DummyLPIPS:
    """Deterministic pseudo-perceptual distance (used to test the clustering)."""

    def __init__(self):
        self.using_official = False

    def __call__(self, a, b):
        return (a - b).reshape(a.shape[0], -1).abs().mean(dim=1)

    def pairwise(self, a, b):
        rows = []
        for index in range(a.shape[0]):
            rows.append(self(a[index : index + 1].expand(b.shape[0], *a.shape[1:]), b))
        return torch.stack(rows, 0)


def test_intra_lpips_is_zero_for_exact_copies():
    train = torch.randn(10, 3, 8, 8)
    generated = train.clone()
    score = intra_lpips(generated, train, metric=DummyLPIPS(), device="cpu")
    assert abs(score) < 1e-6


def test_intra_lpips_grows_with_diversity():
    train = torch.zeros(10, 3, 8, 8)
    identical = train.clone()
    diverse = train + 0.5 * torch.randn_like(train)
    low = intra_lpips(identical, train, metric=DummyLPIPS(), device="cpu")
    high = intra_lpips(diverse, train, metric=DummyLPIPS(), device="cpu")
    assert high > low


def test_frechet_distance_of_identical_features_is_zero():
    features = np.random.RandomState(0).randn(200, 16)
    assert frechet_distance(features, features) < 1e-6


def test_frechet_distance_increases_with_shift():
    rng = np.random.RandomState(1)
    a = rng.randn(200, 8)
    b = a + 1.0
    c = a + 5.0
    assert frechet_distance(a, b) < frechet_distance(a, c)

