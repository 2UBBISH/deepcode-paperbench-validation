"""Metric + LCA-distance unit tests."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lca_on_the_line.hierarchy import load_wordnet_hierarchy  # noqa: E402
from lca_on_the_line.lca import (  # noqa: E402
    dataset_elca,
    dataset_lca,
    dataset_lca_from_matrix,
    elca_from_logits,
    topk_accuracy,
)
from lca_on_the_line.metrics import (  # noqa: E402
    error_prediction_mae,
    ken,
    minmax_scale,
    pea,
    r2,
    spearman,
)


@pytest.fixture(scope="module")
def hierarchy():
    return load_wordnet_hierarchy()


def test_minmax_scale_range():
    x = np.array([3.0, 1.0, 2.0])
    scaled = minmax_scale(x)
    assert scaled.min() == pytest.approx(0.0)
    assert scaled.max() == pytest.approx(1.0)


def test_r2_is_pea_squared():
    rng = np.random.RandomState(0)
    x = rng.rand(50)
    y = 2 * x + rng.rand(50) * 0.1
    assert r2(x, y) == pytest.approx(pea(x, y) ** 2)
    assert pea(x, y) == pytest.approx(abs(np.corrcoef(x, y)[0, 1]), abs=1e-9)


def test_rank_correlations_match_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.RandomState(1)
    x, y = rng.rand(40), rng.rand(40)
    assert ken(x, y) == pytest.approx(abs(scipy_stats.kendalltau(x, y).correlation))
    assert spearman(x, y) == pytest.approx(abs(scipy_stats.spearmanr(x, y).correlation))


def test_error_prediction_mae_is_zero_for_perfect_linear_relation():
    x = np.linspace(0, 1, 20)
    y = 0.5 * x + 0.25
    assert error_prediction_mae(x, y) < 1e-9


def test_dataset_lca_only_counts_mistakes(hierarchy):
    # class 0 (tench) predicted as class 1 (goldfish): distance = log2(2) = 1
    preds = np.array([1, 5, 7])
    targets = np.array([0, 5, 7])
    value, details = dataset_lca(
        hierarchy=hierarchy,
        predictions=preds,
        targets=targets,
        return_details=True,
    )
    assert value == pytest.approx(1.0)
    assert details["wrong"].tolist() == [True, False, False]
    assert details["top1"] == pytest.approx(2 / 3)


def test_dataset_lca_matches_matrix_variant(hierarchy):
    matrix = hierarchy.lca_distance_matrix("information")
    preds = np.array([1, 2, 999, 5])
    targets = np.array([0, 3, 4, 5])
    a = dataset_lca(hierarchy=hierarchy, predictions=preds, targets=targets)
    b = dataset_lca_from_matrix(matrix, preds, targets)
    assert a == pytest.approx(b)


def test_all_correct_gives_zero_lca(hierarchy):
    preds = np.array([0, 1, 2])
    targets = np.array([0, 1, 2])
    assert dataset_lca(hierarchy=hierarchy, predictions=preds, targets=targets) == 0.0


def test_topk_accuracy():
    logits = np.zeros((2, 5))
    logits[0, 3] = 10.0
    logits[1, 1] = 5.0
    logits[1, 4] = 6.0
    targets = np.array([3, 4])
    assert topk_accuracy(logits, targets, 1) == pytest.approx(1.0)
    targets = np.array([2, 4])
    assert topk_accuracy(logits, targets, 1) == pytest.approx(0.5)
    # class 2 is not a top-2 for sample 0, class 4 is the top-1 for sample 1
    assert topk_accuracy(logits, targets, 2) == pytest.approx(0.5)


def test_elca_is_zero_for_a_perfect_one_hot(hierarchy):
    probs = np.zeros((3, 1000))
    targets = np.array([4, 5, 6])
    probs[np.arange(3), targets] = 1.0
    assert dataset_elca(probs, targets, hierarchy=hierarchy) == pytest.approx(0.0)


def test_elca_from_logits_matches_the_probability_version(hierarchy):
    rng = np.random.RandomState(0)
    logits = rng.randn(40, 1000)
    targets = rng.randint(0, 1000, size=40)
    from lca_on_the_line.lca import softmax

    direct = dataset_elca(softmax(logits), targets, hierarchy=hierarchy)
    chunked = elca_from_logits(logits, targets, hierarchy=hierarchy, chunk_size=7)
    assert chunked == pytest.approx(direct, rel=1e-9)


def test_elca_is_monotone_in_mistake_severity(hierarchy):
    """A close mistake must have a lower ELCA than a distant one."""
    close = np.full((1, 1000), -20.0)
    close[0, 1] = 0.0      # predicts goldfish for a tench
    far = np.full((1, 1000), -20.0)
    far[0, 999] = 0.0      # predicts toilet tissue for a tench
    targets = np.array([0])
    assert elca_from_logits(close, targets, hierarchy=hierarchy) < elca_from_logits(
        far, targets, hierarchy=hierarchy
    )
