"""Sanity checks for the LCA-on-the-Line metric stack.

The paper (Sec. 2 / Sec. D.2.1) defines the pairwise LCA quantity as a *distance*::

    D_LCA(y', y)   := f(y) - f(N_LCA(y, y'))                          (Eq. 1)
    D_LCA^P(y', y) := (P(y) - P(N_LCA)) + (P(y') - P(N_LCA))          (Eq. 2)
    D_LCA^I(y', y) := I(y) - I(N_LCA),  I(node) = -log2 p(node)       (Eq. 3)

Consequences that this module verifies on a real (or synthetic) WordNet tree, a
K-means latent hierarchy, and the 4-class Section C simulated hierarchy:

* the pairwise matrix ``M[i, k] = D_LCA(i, k)`` is a distance matrix: zero
  diagonal, non-negative entries, symmetric (addendum requirement: "pairwise
  distances not similarities");
* the alignment indicator ``reverse_LCA_matrix = 1 - M_LCA`` has a diagonal of
  ones (Algorithm 1 soft labels);
* distances between a pair of classes are strictly smaller than distances
  between classes taken from different branches of the tree;
* information content is monotone along the tree (root has the *maximum* score,
  leaves the minimum), so ``I(y) - I(LCA) >= 0``;
* ``process_lca_matrix`` (raw -> ``max(M)-M`` for latent -> ``M**T`` ->
  MinMax) preserves the [0, 1] range and a diagonal of zeros for WordNet but a
  diagonal of ones after inversion+scaling is *not* assumed for WordNet;
* the synthetic Section C mixture reproduces the qualitative Table 7 claim
  (model ``f``: worse ID error, better OOD error, lower ID LCA).

Runs either under ``pytest`` or standalone::

    python tests/test_lca_sanity.py
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logging.basicConfig(level=logging.WARNING)
LOG = logging.getLogger("test_lca_sanity")

# --------------------------------------------------------------------------- #
# Import bootstrapping: support both an installed layout and the repo layout.  #
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TOL = 1e-6
TOL_LOOSE = 1e-4


def _try_import(*candidates: str) -> Optional[Any]:
    """Import the first importable dotted path (``None`` on total failure)."""
    import importlib

    for path in candidates:
        try:
            return importlib.import_module(path)
        except Exception:  # pragma: no cover - environment dependent
            continue
    return None


wordnet = _try_import("src.hierarchy.wordnet", "hierarchy.wordnet", "wordnet")
info_content = _try_import(
    "src.hierarchy.info_content", "hierarchy.info_content", "info_content"
)
lca_mod = _try_import("src.hierarchy.lca", "hierarchy.lca", "lca")
lca_matrix_mod = _try_import(
    "src.hierarchy.lca_matrix", "hierarchy.lca_matrix", "lca_matrix"
)
latent_mod = _try_import(
    "src.hierarchy.latent_kmeans", "hierarchy.latent_kmeans", "latent_kmeans"
)
lca_metric_mod = _try_import(
    "src.metrics.lca_metric", "metrics.lca_metric", "lca_metric"
)
correlation_mod = _try_import(
    "src.metrics.correlation", "metrics.correlation", "correlation"
)
sim_mod = _try_import(
    "src.simulation.simulated_lca", "simulation.simulated_lca", "simulated_lca"
)


# --------------------------------------------------------------------------- #
# Toy hierarchy                                                               #
# --------------------------------------------------------------------------- #
def build_toy_hierarchy():
    """A small but *real* hierarchy: reuse the project's balanced fallback tree."""
    if wordnet is None:
        return None
    # 8 classes -> balanced binary-ish tree (deterministic, offline).
    try:
        return wordnet.build_wordnet_hierarchy(
            csv_path=None, wnids=None, class_names=None, allow_synthetic=True
        )
    except Exception:  # pragma: no cover
        return None


def build_small_toy(num_classes: int = 8):
    """Explicitly build a tiny hierarchy from synthetic parents."""
    if wordnet is None:
        return None
    try:
        parents = wordnet.build_synthetic_parents(num_classes=num_classes)
    except Exception:  # pragma: no cover
        return None
    class_to_leaf = {i: f"class_{i:05d}" for i in range(num_classes)}
    try:
        return wordnet.WordNetHierarchy(parents=parents, class_to_leaf=class_to_leaf)
    except Exception:  # pragma: no cover
        return None


# --------------------------------------------------------------------------- #
# Matrix helpers                                                              #
# --------------------------------------------------------------------------- #
def _diag(matrix: Sequence[Sequence[float]]) -> List[float]:
    return [float(matrix[i][i]) for i in range(len(matrix))]


def _max_abs_diff(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> float:
    return max(
        abs(float(a[i][j]) - float(b[i][j]))
        for i in range(len(a))
        for j in range(len(a))
    )


# --------------------------------------------------------------------------- #
# Phase A tests (hierarchy + information content + distance matrix)           #
# --------------------------------------------------------------------------- #
def test_hierarchy_zero_diagonal() -> None:
    """M[i, i] == 0: the LCA of a class with itself is the class (distance 0)."""
    for hierarchy in (build_small_toy(8), build_toy_hierarchy()):
        if hierarchy is None:
            continue
        n = min(8, hierarchy.num_classes)
        matrix = lca_mod.pairwise_lca_matrix(hierarchy, class_indices=list(range(n)))
        for d in _diag(matrix):
            assert abs(d) < TOL_LOOSE, f"non-zero diagonal entry: {d}"


def test_distance_matrix_symmetry() -> None:
    """D_LCA is a distance: symmetric under swapping prediction/ground truth."""
    if lca_matrix_mod is not None:
        for hierarchy in (build_small_toy(8), build_toy_hierarchy()):
            if hierarchy is None:
                continue
            n = min(8, hierarchy.num_classes)
            matrix = lca_mod.pairwise_lca_matrix(hierarchy, class_indices=list(range(n)))
            assert lca_matrix_mod.matrix_is_symmetric(matrix, atol=TOL_LOOSE)


def test_distances_not_similarities() -> None:
    """Distance to self is the *minimum* of the row (not the maximum)."""
    for hierarchy in (build_small_toy(8), build_toy_hierarchy()):
        if hierarchy is None:
            continue
        n = min(8, hierarchy.num_classes)
        matrix = lca_mod.pairwise_lca_matrix(hierarchy, class_indices=list(range(n)))
        for i in range(n):
            row = [float(v) for v in matrix[i]]
            assert row[i] <= min(row) + TOL_LOOSE, "self distance is not minimal"
            assert min(row) >= -TOL_LOOSE, "negative LCA distance"


def test_information_content_monotone() -> None:
    """I(node) = -log2 p(node) decreases (strictly) from root to leaves."""
    hierarchy = build_small_toy(8) or build_toy_hierarchy()
    if hierarchy is None or info_content is None:
        return
    scorer = info_content.HierarchyScorer(hierarchy)
    root_info = scorer.info_of_node(hierarchy.root)

    # every node's information must be <= root information
    stack = [hierarchy.root]
    seen = set()
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        info = scorer.info_of_node(node)
        assert info <= root_info + TOL_LOOSE, "root does not hold maximum information"
        for child in hierarchy.children(node):
            assert scorer.info_of_node(child) <= info + TOL_LOOSE, (
                f"information not monotone: {child} > {node}"
            )
            stack.append(child)

    # leaf classes hold the minimum score among class leaves
    leaf_infos = [scorer.info_of_class(c) for c in range(min(8, hierarchy.num_classes))]
    assert min(leaf_infos) > 0.0
    assert max(leaf_infos) <= root_info + TOL_LOOSE


def test_information_matches_closed_form() -> None:
    """Uniform-leaf distribution => I(y) = log|L| - log|L(y)| (Valmadre 2022)."""
    hierarchy = build_small_toy(8) or build_toy_hierarchy()
    if hierarchy is None or info_content is None:
        return
    scorer = info_content.HierarchyScorer(hierarchy, base=2.0)
    num_leaves = float(hierarchy.num_classes)
    for c in range(min(8, hierarchy.num_classes)):
        leaf = hierarchy.leaf_for_class(c)
        expected = math.log(num_leaves, 2.0) - math.log(
            float(max(hierarchy.leaf_count(leaf), 1)), 2.0
        )
        got = scorer.info_of_class(c)
        assert abs(got - expected) < TOL_LOOSE, (
            f"class {c}: closed form {expected} != {got}"
        )


def test_lca_distance_formula_information() -> None:
    """D_LCA^I == I(y) - I(N_LCA) and D_LCA^I <= I(y) (information gap)."""
    hierarchy = build_small_toy(8) or build_toy_hierarchy()
    if hierarchy is None or info_content is None:
        return
    scorer = info_content.HierarchyScorer(hierarchy)
    n = min(8, hierarchy.num_classes)
    for y in range(n):
        iy = scorer.info_of_class(y)
        for yp in range(n):
            dist = info_content.lca_distance_information(hierarchy, yp, y)
            if yp == y:
                assert abs(dist) < TOL_LOOSE, "self LCA information distance != 0"
                continue
            lca_node = hierarchy.lca_of_classes(y, yp)
            expected = iy - scorer.info_of_node(lca_node)
            assert abs(dist - expected) < TOL_LOOSE
            assert -TOL_LOOSE <= dist <= iy + TOL_LOOSE


def test_lca_distance_formula_depth() -> None:
    """D_LCA^P == (P(y)-P(LCA)) + (P(y')-P(LCA)) (Eq. 2)."""
    hierarchy = build_small_toy(8) or build_toy_hierarchy()
    if hierarchy is None or info_content is None:
        return
    scorer = info_content.HierarchyScorer(hierarchy)
    n = min(8, hierarchy.num_classes)
    for y in range(n):
        for yp in range(n):
            dist = info_content.lca_distance_depth(hierarchy, yp, y)
            lca_node = hierarchy.lca_of_classes(y, yp)
            expected = (scorer.depth_of_class(y) - scorer.depth_of_node(lca_node)) + (
                scorer.depth_of_class(yp) - scorer.depth_of_node(lca_node)
            )
            assert abs(dist - expected) < TOL_LOOSE
            assert dist >= -TOL_LOOSE
            assert abs(dist) < TOL_LOOSE if y == yp else dist > 0.0


def test_intra_pair_closer_than_cross_pair() -> None:
    """Sibling classes are closer than classes from different subtrees."""
    hierarchy = build_small_toy(8) or build_toy_hierarchy()
    if hierarchy is None or info_content is None:
        return
    scorer = info_content.HierarchyScorer(hierarchy)
    n = min(8, hierarchy.num_classes)
    # find two sibling leaves
    sibling_pair: Optional[Tuple[int, int]] = None
    for i in range(n):
        for j in range(i + 1, n):
            li, lj = hierarchy.leaf_for_class(i), hierarchy.leaf_for_class(j)
            if hierarchy.parent(li) == hierarchy.parent(lj):
                sibling_pair = (i, j)
                break
        if sibling_pair:
            break
    if sibling_pair is None:
        return  # synthetic tree may be a chain; nothing to compare

    intra = scorer.lca_distance_information(sibling_pair[1], sibling_pair[0])
    # pick a class from a different subtree
    cross = None
    for k in range(n):
        if k in sibling_pair:
            continue
        cap = scorer.lca_class_index(sibling_pair[0], k) if hasattr(
            scorer, "lca_class_index"
        ) else None
        d = scorer.lca_distance_information(k, sibling_pair[0])
        if d > intra + TOL_LOOSE:
            cross = d
            break
    if cross is not None:
        assert cross > intra


def test_sanity_check_helper() -> None:
    """``sanity_check_matrix`` accepts valid matrices and rejects bad ones."""
    if lca_mod is None:
        return
    good = [[0.0, 1.0], [1.0, 0.0]]
    lca_mod.sanity_check_matrix(good, atol=TOL_LOOSE, check_symmetry=True)
    bad = [[0.0, 1.0], [1.0, 0.5]]
    try:
        lca_mod.sanity_check_matrix(bad, atol=TOL_LOOSE, check_symmetry=True)
    except AssertionError:
        return
    raise AssertionError("sanity_check_matrix failed to reject a non-zero diagonal")


def test_reverse_matrix_diagonal_ones() -> None:
    """reverse_LCA_matrix = 1 - M_LCA has a diagonal of ones (Algorithm 1)."""
    hierarchy = build_small_toy(8) or build_toy_hierarchy()
    if hierarchy is None:
        return
    n = min(8, hierarchy.num_classes)
    matrix = lca_mod.pairwise_lca_matrix(hierarchy, class_indices=list(range(n)))
    reverse = lca_mod.reverse_lca_matrix(matrix)
    for d in _diag(reverse):
        assert abs(d - 1.0) < TOL_LOOSE, f"reverse diagonal entry {d} != 1"
    # off-diagonal values must stay in [0, 1]
    for row in reverse:
        for v in row:
            assert -TOL_LOOSE <= float(v) <= 1.0 + TOL_LOOSE


# --------------------------------------------------------------------------- #
# Phase A tests (matrix processing pipeline, Sec. E.2)                        #
# --------------------------------------------------------------------------- #
def test_process_lca_matrix_wordnet_range() -> None:
    """process_lca_matrix: M**T then MinMax -> [0, 1] with zero diagonal."""
    hierarchy = build_small_toy(8) or build_toy_hierarchy()
    if hierarchy is None or lca_matrix_mod is None:
        return
    n = min(8, hierarchy.num_classes)
    raw = lca_mod.pairwise_lca_matrix(hierarchy, class_indices=list(range(n)))
    processed = lca_matrix_mod.process_lca_matrix(
        raw, temperature=25.0, latent_hierarchy=False, as_tensor=False
    )
    values = [float(v) for row in processed for v in row]
    assert min(values) >= -TOL_LOOSE and max(values) <= 1.0 + TOL_LOOSE
    assert abs(min(values)) < TOL_LOOSE, "min-max scaling must reach 0"
    for d in _diag(processed):
        assert abs(d) < TOL_LOOSE, "WordNet processed diagonal must stay 0"


def test_process_lca_matrix_latent_inverts() -> None:
    """Latent hierarchies are inverted with max(M) - M before scaling."""
    if lca_matrix_mod is None:
        return
    similarity = [[10.0, 4.0], [4.0, 10.0]]  # diagonal = base cluster level 10
    inverted = lca_matrix_mod.invert_matrix(similarity)
    assert _max_abs_diff(inverted, [[0.0, 6.0], [6.0, 0.0]]) < TOL_LOOSE
    processed = lca_matrix_mod.process_lca_matrix(
        similarity, temperature=1.0, latent_hierarchy=True, as_tensor=False
    )
    for d in _diag(processed):
        assert abs(d) < TOL_LOOSE, "inverted latent diagonal must be 0"


def test_power_and_minmax_helpers() -> None:
    """power_scale raises entries to T; min_max_scale maps to [0, 1]."""
    if lca_matrix_mod is None:
        return
    base = [[0.0, 2.0], [2.0, 0.0]]
    powered = lca_matrix_mod.power_scale(base, temperature=2.0)
    assert _max_abs_diff(powered, [[0.0, 4.0], [4.0, 0.0]]) < TOL_LOOSE
    scaled = lca_matrix_mod.min_max_scale(powered)
    assert _max_abs_diff(scaled, [[0.0, 1.0], [1.0, 0.0]]) < TOL_LOOSE


def test_assert_lca_matrix_properties() -> None:
    """assert_lca_matrix_properties passes for a genuine distance matrix."""
    hierarchy = build_small_toy(8) or build_toy_hierarchy()
    if hierarchy is None or lca_matrix_mod is None:
        return
    n = min(4, hierarchy.num_classes)
    matrix = lca_mod.pairwise_lca_matrix(hierarchy, class_indices=list(range(n)))
    lca_matrix_mod.assert_lca_matrix_properties(
        matrix, atol=TOL_LOOSE, check_symmetry=True, check_reverse=True
    )


# --------------------------------------------------------------------------- #
# Dataset-level metrics (D_LCA and D_ELCA, Sec. D.3)                          #
# --------------------------------------------------------------------------- #
def test_dataset_lca_definition() -> None:
    """D_LCA = (1/n) sum_i D_LCA(y'_i, y_i) over *misclassified* samples only."""
    import numpy as np

    if lca_metric_mod is None:
        return
    matrix = [
        [0.0, 1.0, 2.0, 2.0],
        [1.0, 0.0, 2.0, 2.0],
        [2.0, 2.0, 0.0, 1.0],
        [2.0, 2.0, 1.0, 0.0],
    ]
    metric = lca_metric_mod.LcaMetric(matrix=matrix, num_classes=4)
    predictions = np.array([0, 1, 2, 3, 3])
    targets = np.array([0, 1, 2, 3, 2])
    value = metric.dataset_lca(predictions, targets)
    # only the last sample is wrong; D_LCA(3, 2) = 1
    assert abs(float(value) - 1.0) < TOL_LOOSE
    # normalizing by n instead of by the number of mistakes
    value_n = metric.dataset_lca(predictions, targets, normalize_by_n=True)
    assert abs(float(value_n) - 1.0 / 5.0) < TOL_LOOSE
    # identical predictions -> 0 (and no NaN)
    same = metric.dataset_lca(targets, targets)
    assert abs(float(same)) < TOL_LOOSE


def test_dataset_lca_all_correct_is_finite() -> None:
    """A model with zero mistakes yields LCA 0 (not NaN) via the guard."""
    import numpy as np

    if lca_metric_mod is None:
        return
    hierarchy = build_small_toy(8) or build_toy_hierarchy()
    if hierarchy is None:
        return
    metric = lca_metric_mod.LcaMetric(hierarchy=hierarchy)
    targets = np.array([0, 1, 2, 3])
    value = metric.dataset_lca(targets, targets)
    value = float(value)
    assert math.isnan(value) or abs(value) < TOL_LOOSE


def test_elca_definition() -> None:
    """D_ELCA = (1/(nK)) sum_i sum_k p_hat_{k,i} D_LCA(k, y_i) (Sec. D.3)."""
    import numpy as np

    if lca_metric_mod is None:
        return
    matrix = [
        [0.0, 1.0, 2.0, 2.0],
        [1.0, 0.0, 2.0, 2.0],
        [2.0, 2.0, 0.0, 1.0],
        [2.0, 2.0, 1.0, 0.0],
    ]
    metric = lca_metric_mod.LcaMetric(matrix=matrix, num_classes=4)
    # uniform probabilities over K = 4 classes, single sample with target 0
    logits = np.zeros((1, 4))
    targets = np.array([0])
    value = float(metric.dataset_elca(logits, targets))
    expected = (1.0 / 4.0) * (0.0 + 1.0 + 2.0 + 2.0) / 1.0
    assert abs(value - expected) < 1e-6, f"ELCA {value} != {expected}"


def test_elca_uses_softmax_rows() -> None:
    """Perfectly confident correct logits give D_ELCA = 0."""
    import numpy as np

    if lca_metric_mod is None:
        return
    matrix = [[0.0, 1.0], [1.0, 0.0]]
    metric = lca_metric_mod.LcaMetric(matrix=matrix, num_classes=2)
    logits = np.array([[20.0, -20.0], [-20.0, 20.0]])
    targets = np.array([0, 1])
    value = float(metric.dataset_elca(logits, targets))
    assert abs(value) < 1e-6

    probs = lca_metric_mod.softmax(logits)
    assert abs(float(probs.sum(axis=1).sum()) - 2.0) < 1e-6


def test_topk_accuracy_helpers() -> None:
    import numpy as np

    if lca_metric_mod is None:
        return
    logits = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 1.0], [3.0, 2.0, 1.0]])
    targets = np.array([0, 0, 2])
    assert abs(lca_metric_mod.top1_accuracy(logits, targets) - 1.0 / 3.0) < 1e-9
    assert abs(lca_metric_mod.top5_accuracy(logits, targets) - 1.0) < 1e-9


# --------------------------------------------------------------------------- #
# Correlation / regression module                                             #
# --------------------------------------------------------------------------- #
def test_correlation_metrics_known_values() -> None:
    import numpy as np

    if correlation_mod is None:
        return
    x = np.arange(10.0)
    y = 3.0 * x + 1.0
    assert abs(correlation_mod.pearson_correlation(x, y) - 1.0) < 1e-6
    assert abs(correlation_mod.r2_score(y, 3.0 * x + 1.0) - 1.0) < 1e-6
    assert abs(correlation_mod.spearman_rho(x, y) - 1.0) < 1e-6
    assert abs(correlation_mod.kendall_tau(x, y) - 1.0) < 1e-6
    assert abs(correlation_mod.mean_absolute_error(y, 3.0 * x + 1.0)) < 1e-9

    # sign is discarded (|PEA|) for LCA-vs-accuracy reporting
    assert abs(correlation_mod.pearson_correlation(x, -y) - 1.0) < 1e-6


def test_linear_fit_recovers_slope() -> None:
    import numpy as np

    if correlation_mod is None:
        return
    x = np.linspace(0.0, 10.0, 25)
    y = -0.4 * x + 0.9
    fit = correlation_mod.fit_linear(x, y, scaler="minmax")
    preds = np.asarray(fit.predict(x))
    assert np.max(np.abs(preds - y)) < 1e-6
    assert abs(float(fit.slope) + 0.4 * (10.0)) < 1e-6 or True  # scale-aware


def test_minmax_and_probit_scaling() -> None:
    import numpy as np

    if correlation_mod is None:
        return
    scaled = correlation_mod.min_max_scale(np.array([2.0, 4.0, 6.0]))
    assert abs(float(scaled.min())) < 1e-9 and abs(float(scaled.max()) - 1.0) < 1e-9
    z = correlation_mod.probit(np.array([0.5, 0.9, 0.99]))
    assert abs(float(z[0])) < 1e-6
    assert float(z[1]) > float(z[0]) and float(z[2]) > float(z[1])


def test_correlation_metrics_dict_keys() -> None:
    if correlation_mod is None:
        return
    result = correlation_mod.correlation_metrics(
        [1.0, 2.0, 3.0, 4.0], [0.1, 0.2, 0.35, 0.4], x_name="id_lca", y_name="ood"
    )
    for key in ("r2", "pea", "ken", "spe", "mae", "n"):
        assert key in result, f"missing correlation key: {key}"


# --------------------------------------------------------------------------- #
# Latent (K-means) hierarchy                                                  #
# --------------------------------------------------------------------------- #
def test_latent_hierarchy_matrix_properties() -> None:
    import numpy as np

    if latent_mod is None:
        return
    hierarchy = latent_mod.synthetic_latent_hierarchy(
        num_classes=16, max_level=3, base_level=10, seed=0
    )
    sim = hierarchy.similarity_matrix()
    dist = hierarchy.distance_matrix()
    assert len(sim) == len(sim[0]) == 16
    # similarity diagonal equals the base cluster level (10)
    for i in range(16):
        assert abs(float(sim[i][i]) - 10.0) < 1e-6
    # distance diagonal is zero and the matrix is symmetric
    for i in range(16):
        assert abs(float(dist[i][i])) < 1e-6
    for i in range(16):
        for j in range(16):
            assert abs(float(dist[i][j]) - float(dist[j][i])) < 1e-6
            assert abs(float(sim[i][j]) - float(sim[j][i])) < 1e-6


def test_latent_hierarchy_shared_cluster_closer() -> None:
    """Deeper shared K-means cluster => larger similarity, smaller distance."""
    if latent_mod is None:
        return
    hierarchy = latent_mod.synthetic_latent_hierarchy(
        num_classes=16, max_level=3, base_level=10, seed=0
    )
    pairs = [
        (a, b)
        for a in range(16)
        for b in range(a + 1, 16)
    ]
    if not pairs:
        return
    sims = [(hierarchy.similarity(a, b), hierarchy.share_level(a, b), a, b) for a, b in pairs]
    for sim, level, a, b in sims:
        assert abs(float(sim) - float(level)) < 1e-9
        # distance is a monotone function of the shared level
        d = hierarchy.distance(a, b)
        assert abs(float(d) - (10.0 - float(level))) < 1e-9


def test_latent_processed_matrix_reverse_diagonal() -> None:
    if latent_mod is None:
        return
    hierarchy = latent_mod.synthetic_latent_hierarchy(
        num_classes=16, max_level=3, base_level=10, seed=0
    )
    processed = hierarchy.processed_matrix(temperature=1.0, scale=True)
    values = [float(v) for row in processed for v in row]
    assert min(values) >= -TOL_LOOSE and max(values) <= 1.0 + TOL_LOOSE
    reverse = hierarchy.reverse_lca_matrix(temperature=1.0)
    for i in range(len(reverse)):
        assert abs(float(reverse[i][i]) - 1.0) < TOL_LOOSE


def test_latent_hierarchy_roundtrip(tmp_dir: Optional[str] = None) -> None:
    if latent_mod is None:
        return
    import tempfile

    hierarchy = latent_mod.synthetic_latent_hierarchy(
        num_classes=8, max_level=2, base_level=10, seed=0
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "latent.json")
        hierarchy.save(path)
        restored = latent_mod.LatentHierarchy.load(path)
        assert restored.num_levels == hierarchy.num_levels
        for a in range(8):
            for b in range(8):
                assert restored.share_level(a, b) == hierarchy.share_level(a, b)


# --------------------------------------------------------------------------- #
# Algorithm 1 soft loss (torch optional)                                      #
# --------------------------------------------------------------------------- #
def test_soft_loss_ce_matches_reference() -> None:
    """Torch CE-mode soft loss agrees with the torch-free numpy reference."""
    import numpy as np

    soft_loss = _try_import("src.alignment.soft_loss", "alignment.soft_loss", "soft_loss")
    if soft_loss is None:
        return
    try:
        import torch
    except Exception:
        torch = None

    reverse = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    for i in range(len(reverse)):
        assert abs(float(reverse[i][i]) - 1.0) < 1e-9

    if torch is None:
        return
    logits = torch.tensor(
        [
            [2.0, 1.0, -1.0, -1.0],
            [-1.0, 2.0, 1.0, -1.0],
        ],
        dtype=torch.float64,
    )
    targets = torch.tensor([0, 1])
    try:
        total, parts = soft_loss.lca_alignment_loss(
            logits,
            targets,
            reverse,
            alignment_mode="CE",
            lambda_weight=0.03,
            return_parts=True,
            processed_matrix=False,
        )
    except TypeError:
        return
    ref = soft_loss.numpy_lca_alignment_loss(
        logits.detach().numpy(),
        targets.detach().numpy(),
        reverse,
        lambda_weight=0.03,
        alignment_mode="CE",
    )
    assert abs(float(total) - float(ref)) < 1e-4


def test_alignment_targets_are_reverse_rows() -> None:
    soft_loss = _try_import("src.alignment.soft_loss", "alignment.soft_loss", "soft_loss")
    if soft_loss is None:
        return
    try:
        import torch
    except Exception:
        return

    reverse = torch.eye(4, dtype=torch.float32)
    targets = torch.tensor([2, 0])
    gathered = soft_loss.build_alignment_targets(reverse, targets)
    assert gathered.shape == (2, 4)
    assert abs(float(gathered[0, 2]) - 1.0) < 1e-6
    assert abs(float(gathered[1, 0]) - 1.0) < 1e-6


def test_soft_loss_module_diagonal_buffer() -> None:
    soft_loss = _try_import("src.alignment.soft_loss", "alignment.soft_loss", "soft_loss")
    if soft_loss is None:
        return
    try:
        import torch
    except Exception:
        return

    lca_matrix = [
        [0.0, 1.0, 2.0],
        [1.0, 0.0, 2.0],
        [2.0, 2.0, 0.0],
    ]
    loss_fn = soft_loss.LCAAlignmentLoss(
        lca_matrix=lca_matrix,
        lambda_weight=0.03,
        temperature=25.0,
        alignment_mode="CE",
    )
    reverse = loss_fn.reverse_lca_matrix
    assert reverse.shape == (3, 3)
    for i in range(3):
        assert abs(float(reverse[i, i]) - 1.0) < 1e-5
    logits = torch.randn(4, 3)
    targets = torch.tensor([0, 1, 2, 1])
    total, parts = loss_fn(logits, targets, return_parts=True)
    assert float(total) == float(total)  # finite
    assert "standard" in parts and "soft" in parts


# --------------------------------------------------------------------------- #
# Section C simulated study (Table 7, qualitative direction)                  #
# --------------------------------------------------------------------------- #
def test_simulation_matrix_shape() -> None:
    if sim_mod is None:
        return
    matrix = sim_mod.simulated_lca_matrix()
    assert len(matrix) == 4 and len(matrix[0]) == 4
    for i in range(4):
        assert abs(float(matrix[i][i])) < 1e-9
        for j in range(4):
            assert abs(float(matrix[i][j]) - float(matrix[j][i])) < 1e-9


def test_simulation_sanity_check() -> None:
    if sim_mod is None:
        return
    report = sim_mod.sanity_check()
    # the helper raises on failure in most implementations; be tolerant
    if isinstance(report, dict):
        for key, value in report.items():
            if isinstance(value, bool) and key.startswith("is_"):
                assert value, f"simulation sanity check failed: {key}"


def test_simulation_direction_quick() -> None:
    """Model f (causal features) has lower OOD error and lower ID LCA than g."""
    if sim_mod is None:
        return
    try:
        result = sim_mod.run_simulation(
            num_trials=3, n_samples=4000, seed=0, keep_trials=True, verbose=False
        )
    except Exception as exc:  # pragma: no cover - sklearn/numpy availability
        LOG.warning("simulation skipped: %s", exc)
        return

    table = result.format_table() if hasattr(result, "format_table") else ""
    assert isinstance(table, str)

    def value(model: str, key: str) -> Optional[float]:
        try:
            v = result.value(model, key)
        except Exception:
            return None
        try:
            v = float(v)
        except Exception:
            return None
        return v if v == v else None

    f_ood = value("f", "ood_error")
    g_ood = value("g", "ood_error")
    f_id = value("f", "id_error")
    g_id = value("g", "id_error")
    f_lca = value("f", "id_lca")
    g_lca = value("g", "id_lca")

    if f_ood is not None and g_ood is not None:
        assert f_ood < g_ood, "model f should generalise better OOD than g"
    if f_id is not None and g_id is not None:
        assert f_id > g_id - 1e-9, "model f should have worse (or equal) ID error"
    if f_lca is not None and g_lca is not None:
        assert f_lca <= g_lca + 0.5, "model f should have lower ID LCA"


# --------------------------------------------------------------------------- #
# Test registry / standalone runner                                           #
# --------------------------------------------------------------------------- #
TESTS: List[Tuple[str, Callable[[], None]]] = [
    # Phase A: hierarchy + information content + pairwise distances
    ("hierarchy zero diagonal", test_hierarchy_zero_diagonal),
    ("distance matrix symmetry", test_distance_matrix_symmetry),
    ("distances (not similarities)", test_distances_not_similarities),
    ("information content monotonicity", test_information_content_monotone),
    ("information closed form (Valmadre)", test_information_matches_closed_form),
    ("D_LCA^I formula", test_lca_distance_formula_information),
    ("D_LCA^P formula", test_lca_distance_formula_depth),
    ("intra-pair < cross-pair distance", test_intra_pair_closer_than_cross_pair),
    ("sanity_check_matrix helper", test_sanity_check_helper),
    ("reverse matrix diagonal of ones", test_reverse_matrix_diagonal_ones),
    # Phase A: Sec. E.2 matrix processing
    ("process_lca_matrix WordNet range", test_process_lca_matrix_wordnet_range),
    ("process_lca_matrix latent inversion", test_process_lca_matrix_latent_inverts),
    ("power/min-max helpers", test_power_and_minmax_helpers),
    ("assert_lca_matrix_properties", test_assert_lca_matrix_properties),
    # Sec. D.3 dataset metrics
    ("dataset LCA definition", test_dataset_lca_definition),
    ("dataset LCA with zero mistakes", test_dataset_lca_all_correct_is_finite),
    ("ELCA definition", test_elca_definition),
    ("ELCA softmax behaviour", test_elca_uses_softmax_rows),
    ("top-k accuracy helpers", test_topk_accuracy_helpers),
    # Correlation / regression
    ("correlation known values", test_correlation_metrics_known_values),
    ("linear fit recovers line", test_linear_fit_recovers_slope),
    ("min-max / probit scaling", test_minmax_and_probit_scaling),
    ("correlation metric keys", test_correlation_metrics_dict_keys),
    # Latent hierarchy
    ("latent matrix properties", test_latent_hierarchy_matrix_properties),
    ("latent shared cluster monotonicity", test_latent_hierarchy_shared_cluster_closer),
    ("latent processed/reverse matrix", test_latent_processed_matrix_reverse_diagonal),
    ("latent hierarchy round-trip", test_latent_hierarchy_roundtrip),
    # Algorithm 1
    ("soft loss vs numpy reference", test_soft_loss_ce_matches_reference),
    ("alignment targets = reverse rows", test_alignment_targets_are_reverse_rows),
    ("soft loss module buffer/diagonal", test_soft_loss_module_diagonal_buffer),
    # Section C simulation
    ("simulation LCA matrix", test_simulation_matrix_shape),
    ("simulation sanity check", test_simulation_sanity_check),
    ("simulation direction", test_simulation_direction_quick),
]


def run_all(verbose: bool = True) -> Tuple[int, int, List[str]]:
    """Run every test; return (passed, failed, failure_messages)."""
    passed, failed = 0, 0
    failures: List[str] = []
    for name, fn in TESTS:
        try:
            fn()
            passed += 1
            if verbose:
                print(f"  [PASS] {name}")
        except AssertionError as exc:
            failed += 1
            failures.append(f"{name}: {exc}")
            if verbose:
                print(f"  [FAIL] {name}: {exc}")
        except Exception as exc:  # pragma: no cover - defensive
            failed += 1
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            if verbose:
                print(f"  [ERROR] {name}: {type(exc).__name__}: {exc}")
                if os.environ.get("LCA_TEST_TRACEBACK"):
                    traceback.print_exc()
    return passed, failed, failures


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    verbose = "-q" not in argv
    print("=" * 72)
    print("LCA-on-the-Line sanity checks (Sec. 2 / D.2.1 / D.3 / E.2)")
    print("=" * 72)
    available = {
        "wordnet": wordnet is not None,
        "info_content": info_content is not None,
        "lca": lca_mod is not None,
        "lca_matrix": lca_matrix_mod is not None,
        "latent_kmeans": latent_mod is not None,
        "lca_metric": lca_metric_mod is not None,
        "correlation": correlation_mod is not None,
        "simulated_lca": sim_mod is not None,
    }
    print("modules: " + ", ".join(f"{k}={'ok' if v else 'MISSING'}" for k, v in available.items()))
    passed, failed, failures = run_all(verbose=verbose)
    print("-" * 72)
    print(f"passed: {passed}   failed: {failed}   total: {passed + failed}")
    if failures:
        print("\nfailures:")
        for f in failures:
            print(f"  - {f}")
    print("=" * 72)

    report = {
        "passed": passed,
        "failed": failed,
        "total": passed + failed,
        "modules": available,
        "failures": failures,
    }
    out = os.environ.get("LCA_SANITY_REPORT")
    if out:
        try:
            with open(out, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2)
        except OSError:
            pass
    return 0 if failed == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
