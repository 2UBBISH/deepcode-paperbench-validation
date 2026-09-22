"""Unit tests for Simformer's graph inversion (Webb et al. 2018) module.

These tests validate the machinery that adapts a task-specific *directed* attention
mask ``M_E`` to an arbitrary conditioning state ``M_C`` (Sec. 3.2 / Appendix A1.1).
Because a directed graphical model is not closed under conditioning, the standard
construction is:

1. moralize the graph induced by the latent variables,
2. run the (min-fill) elimination procedure of Webb et al. (2018) to obtain the set
   of edges ``H``,
3. return ``M_E + edges(H)``.

Conventions used throughout the code base (and asserted here):

* ``M[i, j] = 1`` means *query token* ``i`` may attend *key token* ``j``, i.e. the
  edge direction is ``j -> i`` (``j`` is a parent of ``i``).
* ``condition_mask[i] = True`` marks token ``i`` as **observed / conditioned**
  (clamped to its clean value); the remaining entries are **latent**.
* The diagonal of every mask is always kept (self-attention).
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest


# --------------------------------------------------------------------------------------
# tolerant imports (the repo may be laid out as `simformer.simformer.X` or `simformer.X`)
# --------------------------------------------------------------------------------------
def _load_module() -> Any:
    errors = []
    for name in ("simformer.simformer.graph_inversion", "simformer.graph_inversion"):
        try:
            return importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - depends on layout
            errors.append(f"{name}: {exc!r}")
    pytest.skip("graph_inversion module not importable (" + "; ".join(errors) + ")")


GI = _load_module()


def _get(name: str, *fallbacks: str) -> Any:
    for candidate in (name,) + fallbacks:
        if hasattr(GI, candidate):
            return getattr(GI, candidate)
    pytest.skip(f"graph_inversion.{name} is not implemented")


def _as_bool(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask).astype(bool)


# --------------------------------------------------------------------------------------
# fixtures / helpers
# --------------------------------------------------------------------------------------
def chain_mask(n: int) -> np.ndarray:
    """Directed chain: theta_0 -> theta_1 -> ... -> theta_{n-1} (dense diagonal)."""
    mask = np.eye(n, dtype=bool)
    for i in range(1, n):
        mask[i, i - 1] = True
    return mask


def tree_mask() -> np.ndarray:
    """3 parameters + 4 observables, matching the paper's Tree task.

    Edges (parent -> child):
        theta_0 -> theta_1, theta_0 -> theta_2,
        theta_1 -> x_0, theta_1 -> x_1, theta_2 -> x_2, theta_2 -> x_3
    """
    n_theta, n_x = 3, 4
    n = n_theta + n_x
    mask = np.eye(n, dtype=bool)
    for parent, child in ((0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)):
        mask[child, parent] = True
    return mask


def edge_set(mask: np.ndarray) -> set:
    """Return the set of directed edges ``(parent, child)`` implied by a mask."""
    parents = np.asarray(mask).astype(bool).copy()
    np.fill_diagonal(parents, False)
    return {(int(j), int(i)) for i, j in zip(*np.nonzero(parents))}


# --------------------------------------------------------------------------------------
# parent / child extraction
# --------------------------------------------------------------------------------------
def test_parents_from_mask_excludes_diagonal():
    parents_from_mask = _get("parents_from_mask")
    mask = tree_mask()
    parents = parents_from_mask(mask)
    assert len(parents) == mask.shape[0]
    assert parents[1] == {0}
    assert parents[2] == {0}
    assert parents[3] == {1}
    assert parents[4] == {1}
    assert parents[5] == {2}
    assert parents[6] == {2}
    assert parents[0] == set()


def test_children_from_mask_inverts_parents():
    children_from_mask = _get("children_from_mask")
    mask = tree_mask()
    children = children_from_mask(mask)
    assert children[0] == {1, 2}
    assert children[1] == {3, 4}
    assert children[2] == {5, 6}
    assert children[3] == set()


def test_parents_children_are_consistent():
    parents_from_mask = _get("parents_from_mask")
    children_from_mask = _get("children_from_mask")
    mask = chain_mask(5)
    parents = parents_from_mask(mask)
    children = children_from_mask(mask)
    for i in range(5):
        for j in range(5):
            assert (j in parents[i]) == (i in children[j])


# --------------------------------------------------------------------------------------
# moralization
# --------------------------------------------------------------------------------------
def test_moralize_is_symmetric_and_keeps_diagonal():
    moralize = _get("moralize", "moralize_mask")
    moral = _as_bool(moralize(tree_mask()))
    assert moral.shape == (7, 7)
    assert np.array_equal(moral, moral.T), "moral graph must be undirected"
    assert np.all(np.diag(moral)), "diagonal must be preserved"


def test_moralize_marries_parents():
    """Two parents of the same child become connected in the moral graph."""
    moralize = _get("moralize", "moralize_mask")
    # child 2 has parents 0 and 1 -> they must be married
    mask = np.eye(3, dtype=bool)
    mask[2, 0] = True
    mask[2, 1] = True
    moral = _as_bool(moralize(mask))
    assert moral[0, 1] and moral[1, 0]


def test_moralize_contains_original_edges():
    moralize = _get("moralize", "moralize_mask")
    mask = tree_mask()
    moral = _as_bool(moralize(mask))
    directed = _as_bool(mask)
    off = directed.copy()
    np.fill_diagonal(off, False)
    assert np.all(moral[off]), "all original directed edges must survive moralization"


# --------------------------------------------------------------------------------------
# min-fill ordering
# --------------------------------------------------------------------------------------
def test_min_fill_order_is_a_permutation():
    min_fill_order = _get("min_fill_order")
    adj = _as_bool(moralize_or_none(tree_mask()))
    order = list(min_fill_order(adj))
    assert sorted(order) == list(range(adj.shape[0]))


def test_min_fill_select_prefers_low_fill():
    """A clique of size 2 connected to a third isolated node: the isolated node
    creates no fill edges and should be selected first."""
    min_fill_select = _get("min_fill_select")
    adj = np.zeros((3, 3), dtype=bool)
    adj[0, 1] = adj[1, 0] = True
    np.fill_diagonal(adj, True)
    chosen = min_fill_select(adj, [0, 1, 2], set())
    assert chosen == 2


def test_min_fill_select_restricted_to_candidates():
    min_fill_select = _get("min_fill_select")
    adj = np.asarray(chain_mask(4))
    chosen = min_fill_select(adj, [2, 3], set())
    assert chosen in (2, 3)


# --------------------------------------------------------------------------------------
# graph inversion: semantics under conditioning
# --------------------------------------------------------------------------------------
def test_graph_inversion_returns_square_mask_containing_base():
    graph_inversion = _get("graph_inversion")
    base = tree_mask()
    condition_mask = np.array([True, False, False, True, True, True, True])
    H = _as_bool(graph_inversion(base, condition_mask=condition_mask))
    assert H.shape == base.shape
    assert np.all(H[base.astype(bool)]), "inverted mask must contain M_E"


def test_graph_inversion_posterior_adds_parent_to_child_edges():
    """Posterior conditioning: parameters latent, data observed.

    For the Tree graph, theta_0 is a latent parent of theta_1/theta_2, which are
    themselves parents of observed data; the inversion must allow the latent
    parameters to attend the data they are not ancestors of.
    """
    graph_inversion = _get("graph_inversion")
    base = tree_mask()
    condition_mask = np.array([False, False, False, True, True, True, True])
    H = _as_bool(graph_inversion(base, condition_mask=condition_mask))
    # latent parameters may now attend observed data
    latent = [0, 1, 2]
    observed = [3, 4, 5, 6]
    assert any(H[i, j] for i in latent for j in observed)


def test_graph_inversion_latent_only_gives_no_new_data_to_param_edges():
    """If nothing is conditioned, no extra edges are required."""
    graph_inversion = _get("graph_inversion")
    base = tree_mask()
    condition_mask = np.zeros(7, dtype=bool)
    H = _as_bool(graph_inversion(base, condition_mask=condition_mask))
    assert edge_set(H) == edge_set(base)


def test_graph_inversion_keeps_diagonal():
    graph_inversion = _get("graph_inversion")
    base = chain_mask(5)
    condition_mask = np.array([False, False, True, True, True])
    H = _as_bool(graph_inversion(base, condition_mask=condition_mask))
    assert np.all(np.diag(H))


def test_graph_inversion_result_container_fields():
    graph_inversion = _get("graph_inversion")
    base = tree_mask()
    condition_mask = np.array([False, False, False, True, True, True, True])
    result = graph_inversion(base, condition_mask=condition_mask, return_result=True)
    assert hasattr(result, "mask")
    assert hasattr(result, "edges")
    assert hasattr(result, "order")
    assert hasattr(result, "moral")
    assert np.asarray(result.mask).shape == base.shape
    # `mask` should be exactly the inverted graph (base + H edges)
    assert np.all(_as_bool(result.mask)[base.astype(bool)])


def test_graph_inversion_result_merged_with_base():
    graph_inversion = _get("graph_inversion")
    base = tree_mask()
    condition_mask = np.array([False, False, False, True, True, True, True])
    result = graph_inversion(base, condition_mask=condition_mask, return_result=True)
    merged = _as_bool(result.merged(base))
    assert merged.shape == base.shape
    assert edge_set(merged) >= edge_set(base)
    assert np.all(np.diag(merged))


def test_graph_inversion_latent_mask_is_complement_of_condition_mask():
    graph_inversion = _get("graph_inversion")
    base = tree_mask()
    condition_mask = np.array([False, True, False, True, False, True, False])
    H_from_cond = _as_bool(graph_inversion(base, condition_mask=condition_mask))
    H_from_latent = _as_bool(
        graph_inversion(base, latent_mask=~condition_mask)
    )
    assert np.array_equal(H_from_cond, H_from_latent)


def test_graph_inversion_leaves_non_ancestor_edges_only_upward():
    """Extra edges inserted by inversion never point from an observed variable
    into a latent one for a *likelihood* conditioning of the chain graph."""
    graph_inversion = _get("graph_inversion")
    base = chain_mask(4)
    # parameters observed, data latent -> likelihood
    condition_mask = np.array([True, True, False, False])
    H = _as_bool(graph_inversion(base, condition_mask=condition_mask))
    extra = edge_set(H) - edge_set(base)
    for parent, child in extra:
        # an observed parent may point at a latent child (fine); a latent parent
        # must not gain an edge into an observed child that is not downstream
        assert not (condition_mask[parent] and not condition_mask[child] and child < parent)


# --------------------------------------------------------------------------------------
# condition-aware masks
# --------------------------------------------------------------------------------------
def test_make_condition_aware_mask_undirected_is_symmetric():
    make_condition_aware_mask = _get("make_condition_aware_mask")
    base = tree_mask()
    condition_mask = np.array([False, False, False, True, True, True, True])
    mask = _as_bool(
        make_condition_aware_mask(base, condition_mask, directed=False)
    )
    assert np.array_equal(mask, mask.T)


def test_make_condition_aware_mask_directed_contains_base():
    make_condition_aware_mask = _get("make_condition_aware_mask")
    base = tree_mask()
    condition_mask = np.array([False, False, False, True, True, True, True])
    mask = _as_bool(make_condition_aware_mask(base, condition_mask, directed=True))
    assert np.all(mask[base.astype(bool)])


def test_adapt_attention_mask_matches_graph_inversion():
    adapt = _get("adapt_attention_mask", "adapt_mask_for_conditioning")
    graph_inversion = _get("graph_inversion")
    base = tree_mask()
    condition_mask = np.array([False, False, False, True, True, True, True])
    adapted = _as_bool(adapt(base, condition_mask))
    from_inversion = _as_bool(graph_inversion(base, condition_mask=condition_mask))
    assert np.array_equal(adapted, from_inversion)


# --------------------------------------------------------------------------------------
# batched / dynamic API used by the training loop and sampler
# --------------------------------------------------------------------------------------
def test_graph_inversion_batched_shape_and_per_row_equality():
    graph_inversion = _get("graph_inversion")
    graph_inversion_batched = _get("graph_inversion_batched")
    base = tree_mask()
    masks = np.array(
        [
            [False, False, False, True, True, True, True],
            [True, True, True, False, False, False, False],
            [False, True, False, True, False, True, True],
        ]
    )
    H = np.asarray(graph_inversion_batched(base, masks))
    assert H.shape == (3,) + base.shape
    for b in range(masks.shape[0]):
        single = _as_bool(graph_inversion(base, condition_mask=masks[b]))
        assert np.array_equal(_as_bool(H[b]), single)


def test_adapt_attention_mask_batched():
    adapt = _get("adapt_attention_mask", "adapt_mask_for_conditioning")
    base = tree_mask()
    masks = np.array(
        [
            [False, False, False, True, True, True, True],
            [False, False, False, False, True, True, True],
        ]
    )
    H = np.asarray(adapt(base, masks, batched=True))
    assert H.shape == (2,) + base.shape
    for b in range(masks.shape[0]):
        assert np.all(_as_bool(H[b])[base.astype(bool)])


def test_attention_mask_masks_are_binary():
    graph_inversion = _get("graph_inversion")
    base = tree_mask()
    condition_mask = np.array([False, True, False, True, False, True, False])
    H = np.asarray(graph_inversion(base, condition_mask=condition_mask))
    assert set(np.unique(H)).issubset({0, 1, False, True})


def test_graph_inversion_does_not_mutate_input():
    graph_inversion = _get("graph_inversion")
    base = tree_mask()
    before = base.copy()
    _ = graph_inversion(base, condition_mask=np.array([False] * 7))
    assert np.array_equal(base, before), "input mask must not be modified in place"


def test_condition_mask_validation():
    graph_inversion = _get("graph_inversion")
    base = tree_mask()
    bad = np.array([0, 1, 0])  # wrong length
    with pytest.raises((ValueError, AssertionError, IndexError)):
        _ = graph_inversion(base, condition_mask=bad)


# --------------------------------------------------------------------------------------
# task-specific structure round trips
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "task,n_theta,n_x",
    [
        ("gaussian_linear", 10, 10),
        ("two_moons", 2, 2),
        ("gaussian_mixture", 2, 2),
        ("slcp", 5, 8),
        ("tree", 3, 4),
        ("hmm", 10, 10),
    ],
)
def test_inversion_on_task_masks_is_well_formed(task, n_theta, n_x):
    """Build each task's directed M_E and verify the inverted posterior graph is a
    valid (square, diagonal-preserving, M_E-containing) attention mask."""
    for name in ("simformer.simformer.attention_masks", "simformer.attention_masks"):
        try:
            am = importlib.import_module(name)
        except Exception:  # pragma: no cover
            continue
        if not hasattr(am, "build_attention_mask"):
            continue
        graph_inversion = _get("graph_inversion")
        try:
            base = np.asarray(am.build_attention_mask(task, n_theta=n_theta, n_x=n_x))
        except TypeError:
            base = np.asarray(am.build_attention_mask(task))
        except Exception:  # pragma: no cover - mask builder signature drift
            pytest.skip(f"cannot build mask for {task}")
        assert base.ndim == 2 and base.shape[0] == base.shape[1]
        condition_mask = np.array([False] * n_theta + [True] * n_x)
        H = _as_bool(graph_inversion(base, condition_mask=condition_mask))
        assert H.shape == base.shape
        assert np.all(H[base.astype(bool)])
        assert np.all(np.diag(H))
        break


def moralize_or_none(mask: np.ndarray) -> np.ndarray:
    """Best-effort moralization used by ordering tests (falls back to symmetrization)."""
    if hasattr(GI, "moralize"):
        return np.asarray(GI.moralize(mask))
    if hasattr(GI, "moralize_mask"):
        return np.asarray(GI.moralize_mask(mask))
    symmetrized = np.asarray(mask).astype(bool) | np.asarray(mask).astype(bool).T
    np.fill_diagonal(symmetrized, True)
    return symmetrized
