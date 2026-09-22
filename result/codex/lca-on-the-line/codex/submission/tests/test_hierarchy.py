"""Sanity checks for the WordNet hierarchy and LCA distances.

The paper addendum explicitly asks for sanity checks on the LCA matrix:
it stores a *distance* (zero diagonal, non-negative), so the inverted matrix
used for the soft labels must have a diagonal of ones.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lca_on_the_line.hierarchy import (  # noqa: E402
    load_imagenet_class_index,
    load_wordnet_hierarchy,
)


@pytest.fixture(scope="module")
def hierarchy():
    return load_wordnet_hierarchy()


def test_leaf_synsets_match_imagenet_class_index(hierarchy):
    synsets, names = load_imagenet_class_index()
    assert len(synsets) == 1000
    assert len(hierarchy.class_synsets) == 1000
    assert hierarchy.class_synsets == synsets
    assert hierarchy.total_leaves == 1000
    assert len(hierarchy.roots) == 1
    assert len(names) == 1000


def test_lca_of_siblings_is_the_parent(hierarchy):
    # tench and goldfish are both cyprinids (WordNet n01439121)
    tench = hierarchy.class_synsets[0]
    goldfish = hierarchy.class_synsets[1]
    lca = hierarchy.lca(tench, goldfish)
    assert hierarchy.leaves_below(lca) == 2
    assert hierarchy.is_ancestor(lca, tench)
    assert hierarchy.is_ancestor(lca, goldfish)


def test_information_content_matches_definition(hierarchy):
    # I(y) = log2|L| - log2|L(y)|, uniform over the 1000 leaves
    for synset in hierarchy.class_synsets[:50]:
        assert hierarchy.information(synset) == pytest.approx(np.log2(1000))
    root = hierarchy.roots[0]
    assert hierarchy.information(root) == pytest.approx(0.0)


def test_lca_distance_matrix_is_a_distance(hierarchy):
    matrix = hierarchy.lca_distance_matrix("information")
    assert matrix.shape == (1000, 1000)
    # symmetric, non-negative, zero diagonal
    assert np.allclose(matrix, matrix.T)
    assert np.all(matrix >= 0)
    assert np.allclose(np.diag(matrix), 0.0)
    # distances are bounded by log2(1000)
    assert matrix.max() <= np.log2(1000) + 1e-9
    # siblings (tench/goldfish) are the closest possible pairs
    assert matrix[0, 1] == pytest.approx(1.0)
    # far apart classes hit the root
    assert matrix[0, 999] == pytest.approx(np.log2(1000))


def test_depth_variant_is_a_distance(hierarchy):
    matrix = hierarchy.lca_distance_matrix("depth")
    assert np.allclose(np.diag(matrix), 0.0)
    assert np.all(matrix >= 0)
    assert np.allclose(matrix, matrix.T)
    assert matrix.max() > 0
    # the depth variant is the path length through the LCA
    assert matrix[0, 1] == pytest.approx(2.0)


def test_inverted_matrix_has_unit_diagonal(hierarchy):
    matrix = hierarchy.lca_distance_matrix("information")
    inverted = matrix.max() - matrix
    assert np.allclose(np.diag(inverted), matrix.max())
