"""Tests for the latent hierarchies, the soft-label loss and the baselines."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lca_on_the_line.baselines import (  # noqa: E402
    aline,
    average_confidence,
    fit_temperature,
    softmax_np,
)
from lca_on_the_line.latent import (  # noqa: E402
    latent_lca_distance_matrix,
    latent_lca_height_matrix,
    process_lca_matrix,
)
from lca_on_the_line.soft_labels import (  # noqa: E402
    LinearProbe,
    build_soft_labels,
    interpolate_weights,
    lca_alignment_loss,
    soft_label_targets,
)


# --------------------------------------------------------------------------- #
# latent hierarchies
# --------------------------------------------------------------------------- #
def _toy_class_features(n_classes=60, dim=16, seed=0):
    rng = np.random.RandomState(seed)
    centroids = rng.randn(6, dim)
    feats = []
    for i in range(n_classes):
        group = i // (n_classes // 6)
        feats.append(centroids[min(group, 5)] + 0.1 * rng.randn(dim))
    return np.array(feats)


def test_height_matrix_diagonal_is_max():
    feats = _toy_class_features()
    height = latent_lca_height_matrix(feats, n_levels=5, seed=0)
    assert np.allclose(np.diag(height), 5)
    assert np.allclose(height, height.T)
    # close-by classes share a cluster at a deeper level than distant ones
    assert height[0, 1] >= height[0, -1]


def test_distance_matrix_has_zero_diagonal():
    feats = _toy_class_features()
    dist = latent_lca_distance_matrix(feats, n_levels=5, seed=0)
    assert np.allclose(np.diag(dist), 0.0)
    assert np.all(dist >= 0)
    assert np.allclose(dist, dist.T)


def test_process_lca_matrix_latent_is_inverted_and_scaled():
    feats = _toy_class_features()
    height = latent_lca_height_matrix(feats, n_levels=5, seed=0)
    processed = process_lca_matrix(height, tree_prefix="latent", temperature=1.0)
    processed = processed.numpy()
    assert processed.min() == pytest.approx(0.0)
    assert processed.max() == pytest.approx(1.0)
    assert np.allclose(np.diag(processed), 0.0)
    # the reversed matrix (used as soft targets) has a unit diagonal
    assert np.allclose(np.diag(1.0 - processed), 1.0)


def test_process_lca_matrix_wordnet_keeps_distance_order():
    raw = np.array([[0.0, 1.0, 4.0], [1.0, 0.0, 3.0], [4.0, 3.0, 0.0]])
    processed = process_lca_matrix(raw, tree_prefix="WordNet", temperature=1.0).numpy()
    assert np.allclose(np.diag(processed), 0.0)
    # a distance of 0 maps to 0, the maximum distance maps to a larger value
    assert processed[0, 1] < processed[0, 2]


def test_wordnet_hierarchy_with_temperature():
    from lca_on_the_line.hierarchy import load_wordnet_hierarchy

    hierarchy = load_wordnet_hierarchy()
    matrix = hierarchy.lca_distance_matrix("depth")
    processed = process_lca_matrix(matrix, "WordNet", temperature=25.0).numpy()
    assert np.allclose(np.diag(processed), 0.0)
    assert processed.max() == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# soft labels
# --------------------------------------------------------------------------- #
def test_lca_alignment_loss_matches_algorithm_one():
    logits = torch.randn(4, 5, requires_grad=False)
    targets = torch.tensor([0, 1, 2, 3])
    matrix = torch.zeros(5, 5)
    matrix[0, 1] = matrix[1, 0] = 0.5
    loss = lca_alignment_loss(logits, targets, "CE", matrix, lambda_weight=0.03)
    # manual computation
    probs = torch.softmax(logits, dim=1)
    one_hot = torch.nn.functional.one_hot(targets, 5).float()
    standard = -(one_hot * torch.log(probs + 1e-12)).sum(dim=1)
    reverse = 1.0 - matrix
    soft = -(reverse[targets] * torch.log(probs + 1e-12)).mean(dim=1)
    expected = (0.03 * standard + soft).mean()
    assert loss.item() == pytest.approx(expected.item(), rel=1e-5)


def test_lca_alignment_loss_is_lower_for_aligned_predictions():
    matrix = torch.zeros(4, 4)
    matrix[0, 1] = matrix[1, 0] = 1.0  # class 1 is close to class 0
    good = torch.tensor([[5.0, 2.0, 0.0, 0.0]])   # predicts the close class
    bad = torch.tensor([[0.0, 0.0, 0.0, 5.0]])    # predicts a distant class
    targets = torch.tensor([0])
    good_loss = lca_alignment_loss(good, targets, "CE", matrix).item()
    bad_loss = lca_alignment_loss(bad, targets, "CE", matrix).item()
    assert good_loss < bad_loss


def test_soft_targets_have_one_on_the_ground_truth():
    raw = np.array([[0.0, 2.0, 4.0], [2.0, 0.0, 6.0], [4.0, 6.0, 0.0]])
    processed = build_soft_labels(raw, temperature=1.0, tree_prefix="WordNet")
    targets = soft_label_targets(processed, [0, 1])
    assert targets[0, 0].item() == pytest.approx(1.0)
    assert targets[1, 1].item() == pytest.approx(1.0)
    assert targets[0, 2] < targets[0, 1]


def test_weight_interpolation():
    a = LinearProbe(4, 3)
    b = LinearProbe(4, 3)
    with torch.no_grad():
        a.fc.weight.fill_(0.0)
        a.fc.bias.fill_(0.0)
        b.fc.weight.fill_(2.0)
        b.fc.bias.fill_(4.0)
    mid = interpolate_weights(a, b, alpha=0.5)
    assert torch.allclose(mid.fc.weight, torch.full_like(mid.fc.weight, 1.0))
    assert torch.allclose(mid.fc.bias, torch.full_like(mid.fc.bias, 2.0))
    only_ce = interpolate_weights(a, b, alpha=1.0)
    assert torch.allclose(only_ce.fc.bias, torch.zeros_like(only_ce.fc.bias))


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #
def test_average_confidence():
    probs = np.array([[0.7, 0.3], [0.9, 0.1]])
    assert average_confidence(probs) == pytest.approx(0.8)


def test_fit_temperature_is_positive_and_calibrates():
    rng = np.random.RandomState(0)
    logits = rng.randn(200, 10) * 5
    targets = rng.randint(0, 10, size=200)
    temperature = fit_temperature(logits, targets)
    assert temperature > 0
    # calibrated probabilities should be closer to the true accuracy than T=1
    preds = logits.argmax(axis=1)
    acc = float((preds == targets).mean())
    conf = average_confidence(softmax_np(logits, temperature))
    base_conf = average_confidence(softmax_np(logits, 1.0))
    assert abs(conf - acc) <= abs(base_conf - acc) + 1e-6


def test_aline_recovers_known_accuracies():
    """If OOD accuracy equals ID accuracy, both predictors should be close."""
    rng = np.random.RandomState(0)
    n_models, n_samples, n_classes = 12, 4000, 20
    accs = np.linspace(0.3, 0.9, n_models)
    test_preds, shift_preds = [], []
    for acc in accs:
        correct = rng.rand(n_samples) < acc
        labels = rng.randint(0, n_classes, size=n_samples)
        preds = labels.copy()
        flip = ~correct
        preds[flip] = (labels[flip] + 1) % n_classes
        test_preds.append(preds)
        # the OOD predictions are the same model with a different random flip
        correct_ood = rng.rand(n_samples) < acc
        preds_ood = labels.copy()
        flip_ood = ~correct_ood
        preds_ood[flip_ood] = (labels[flip_ood] + 2) % n_classes
        shift_preds.append(preds_ood)
    (pred_s, pred_d), _, _ = aline(test_preds, accs, shift_preds)
    assert np.corrcoef(pred_s, accs)[0, 1] > 0.9
    assert np.corrcoef(pred_d, accs)[0, 1] > 0.5
