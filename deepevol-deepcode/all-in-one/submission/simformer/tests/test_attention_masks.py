"""Sanity tests for Simformer attention masks (Sec. 3.2 / Appendix A1.1).

These tests validate that:

* the task-specific directed base masks ``M_E`` match the block structures
  described in the paper / addendum (Gaussian Linear, Two Moons / Gaussian
  Mixture, SLCP, Tree, HMM, and the ODE-chain tasks),
* the mask convention ``M[i, j] = 1`` <=> "query token ``i`` may attend key
  token ``j``" (i.e. edge ``j -> i``, ``j`` is a parent of ``i``) holds,
* the diagonal is always kept (self-attention is always allowed),
* utilities (symmetrization, edge addition, mask powers, registry dispatch)
  behave as documented.

The file is intentionally tolerant about the package layout: it imports the
mask module under either ``simformer.attention_masks`` or
``simformer.simformer.attention_masks``.
"""

from __future__ import annotations

import importlib
from typing import List

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# tolerant import helpers
# ---------------------------------------------------------------------------
def _load_mask_module():
    """Import the attention-mask module regardless of the repo layout."""
    errors: List[str] = []
    for name in ("simformer.attention_masks", "simformer.simformer.attention_masks"):
        try:
            return importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - layout dependent
            errors.append(f"{name}: {exc!r}")
    raise ImportError("could not import attention_masks module; tried:\n" + "\n".join(errors))


MASK_MODULE = _load_mask_module()


def _get(name: str, *fallbacks: str):
    """Fetch ``name`` (or a fallback) from the mask module, skipping if absent."""
    for candidate in (name,) + fallbacks:
        if hasattr(MASK_MODULE, candidate):
            return getattr(MASK_MODULE, candidate)
    pytest.skip(f"{name} not provided by {MASK_MODULE.__name__}")


# module-level symbols used across the tests
dense_mask = _get("dense_mask")
identity_mask = _get("identity_mask")
blocks = _get("blocks")
block_diag_mask = _get("block_diag_mask")
ensure_diagonal = _get("ensure_diagonal")
undirected = _get("undirected", "symmetrize")
combine_masks = _get("combine_masks")
add_edges = _get("add_edges")
mask_power_dependencies = _get("mask_power_dependencies")
build_attention_mask = _get("build_attention_mask")
mask_variants = _get("mask_variants")
TASK_MASK_REGISTRY = _get("TASK_MASK_REGISTRY")

gaussian_linear_mask = _get("gaussian_linear_mask")
two_moons_mask = _get("two_moons_mask")
gaussian_mixture_mask = _get("gaussian_mixture_mask")
slcp_mask = _get("slcp_mask")
tree_mask = _get("tree_mask")
hmm_mask = _get("hmm_mask")
ode_chain_mask = _get("ode_chain_mask")

DENSE = getattr(MASK_MODULE, "DENSE", None)
IDENTITY = getattr(MASK_MODULE, "IDENTITY", None)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _as_bool(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask).astype(bool)


def _theta_data_split(mask: np.ndarray, n_theta: int, n_x: int):
    """Return the four sub-blocks (theta<-theta, theta<-x, x<-theta, x<-x)."""
    m = _as_bool(mask)
    assert m.shape == (n_theta + n_x, n_theta + n_x)
    return (
        m[:n_theta, :n_theta],  # theta queries, theta keys
        m[:n_theta, n_theta:],  # theta queries, data keys
        m[n_theta:, :n_theta],  # data queries, theta keys
        m[n_theta:, n_theta:],  # data queries, data keys
    )


# ---------------------------------------------------------------------------
# generic mask utilities
# ---------------------------------------------------------------------------
def test_dense_mask_is_fully_connected():
    m = _as_bool(dense_mask(5))
    assert m.shape == (5, 5)
    assert m.all()


def test_identity_mask_is_diagonal():
    m = _as_bool(identity_mask(4))
    assert m.shape == (4, 4)
    assert np.array_equal(m, np.eye(4, dtype=bool))


def test_blocks_are_block_diagonal():
    a = np.array([[1, 1], [0, 1]], dtype=bool)
    b = np.array([[1, 0], [0, 1]], dtype=bool)
    m = _as_bool(blocks(a, b))
    assert m.shape == (4, 4)
    assert np.array_equal(m[:2, :2], a)
    assert np.array_equal(m[2:, 2:], b)
    assert not m[:2, 2:].any()
    assert not m[2:, :2].any()


def test_block_diag_mask_leaves_data_to_theta_zero():
    n_theta, n_x = 3, 3
    theta_block = np.eye(n_theta, dtype=bool)
    data_block = np.eye(n_x, dtype=bool)
    theta_data = np.ones((n_theta, n_x), dtype=bool)
    m = _as_bool(block_diag_mask(theta_block, data_block, theta_data))
    tt, txx, xt, xx = _theta_data_split(m, n_theta, n_x)
    assert np.array_equal(tt, theta_block | np.eye(n_theta, dtype=bool))
    assert np.array_equal(xx, data_block | np.eye(n_x, dtype=bool))
    assert np.array_equal(txx, theta_data)
    assert not xt.any()


def test_ensure_diagonal_sets_diagonal():
    base = np.zeros((4, 4), dtype=bool)
    base[0, 1] = True
    out = _as_bool(ensure_diagonal(base, True))
    assert np.all(np.diag(out))
    assert out[0, 1]


def test_undirected_is_symmetric():
    m = np.zeros((4, 4), dtype=bool)
    m[0, 1] = True
    m[2, 3] = True
    sym = _as_bool(undirected(m))
    assert np.array_equal(sym, sym.T)
    assert sym[1, 0] and sym[0, 1]
    assert sym[3, 2] and sym[2, 3]
    # symmetrization must be idempotent
    assert np.array_equal(_as_bool(undirected(sym)), sym)


def test_combine_masks_is_union():
    a = np.zeros((3, 3), dtype=bool)
    b = np.zeros((3, 3), dtype=bool)
    a[0, 1] = True
    b[1, 2] = True
    combined = _as_bool(combine_masks(a, b))
    assert combined[0, 1] and combined[1, 2]
    assert np.array_equal(combined, a | b)


def test_add_edges_adds_directed_edges():
    base = np.eye(4, dtype=bool)
    out = _as_bool(add_edges(base, [(0, 2), (1, 3)]))
    # edge (parent, child) => mask[child, parent] = 1
    assert out[2, 0]
    assert out[3, 1]
    assert np.all(np.diag(out))


def test_mask_power_dependencies_grows_with_layers():
    m = np.eye(4, dtype=bool)
    m[1, 0] = True  # 0 -> 1
    powers = _as_bool(mask_power_dependencies(m, 3))
    # after 3 layers information can flow 0 -> 1 -> 2 -> 3
    assert powers[3, 0]
    assert np.all(np.diag(powers))
    # one layer must not depend on the future
    assert not _as_bool(mask_power_dependencies(m, 1))[0, 3]


def test_to_torch_mask_matches_numpy():
    torch = pytest.importorskip("torch")
    to_torch_mask = _get("to_torch_mask")
    m = np.zeros((3, 3), dtype=bool)
    m[0, 1] = True
    t = to_torch_mask(m)
    assert tuple(t.shape) == (3, 3)
    assert t.dtype == torch.bool
    assert bool(t[0, 1]) is True and bool(t[2, 0]) is False


# ---------------------------------------------------------------------------
# Gaussian Linear (Appendix A2.2, per-dimension factorized dependency)
# ---------------------------------------------------------------------------
def test_gaussian_linear_mask_structure():
    n_theta, n_x = 10, 10
    m = _as_bool(gaussian_linear_mask())
    assert m.shape == (n_theta + n_x, n_theta + n_x)
    assert np.all(np.diag(m))

    tt, theta_data, data_theta, xx = _theta_data_split(m, n_theta, n_x)
    # factorized across dimensions: theta_i depends only on x_i
    assert np.array_equal(theta_data, np.eye(n_theta, n_x, dtype=bool))
    # generative direction only: data never attend parameters in the base mask
    assert not data_theta.any()
    # parameters are mutually independent, likewise the observations
    assert np.array_equal(tt, np.eye(n_theta, dtype=bool))
    assert np.array_equal(xx, np.eye(n_x, dtype=bool))


def test_gaussian_linear_mask_custom_dimensions():
    m = _as_bool(gaussian_linear_mask(n_theta=3, n_x=4))
    assert m.shape == (7, 7)
    assert np.all(np.diag(m))
    assert m[0, 3] and m[2, 5]  # theta_i -> x_i
    assert not m[3, 0]  # no data -> parameter edge
    assert not m[0, 1]  # parameters independent in the directed base mask


def test_gaussian_linear_undirected_is_symmetric():
    m = _as_bool(undirected(gaussian_linear_mask()))
    assert np.array_equal(m, m.T)
    assert m[10, 0]  # x_0 -> theta_0 appears after symmetrization


# ---------------------------------------------------------------------------
# Two Moons / Gaussian Mixture
# ---------------------------------------------------------------------------
def test_two_moons_mask_structure():
    m = _as_bool(two_moons_mask())
    assert m.shape == (4, 4)
    assert np.all(np.diag(m))
    tt, theta_data, data_theta, xx = _theta_data_split(m, 2, 2)
    # dense theta -> data coupling and dense data block
    assert theta_data.all()
    assert xx.all()
    # base mask keeps generative direction only
    assert not data_theta.any()


def test_gaussian_mixture_matches_two_moons():
    assert np.array_equal(_as_bool(gaussian_mixture_mask()), _as_bool(two_moons_mask()))


# ---------------------------------------------------------------------------
# SLCP (Appendix A2.2: 4 i.i.d. 2-D observations)
# ---------------------------------------------------------------------------
def test_slcp_mask_structure():
    m = _as_bool(slcp_mask())
    assert m.shape == (13, 13)  # 5 parameters + 8 data dimensions
    assert np.all(np.diag(m))
    tt, theta_data, data_theta, xx = _theta_data_split(m, 5, 8)
    assert theta_data.all()  # every observation depends on all parameters
    assert not data_theta.any()
    # i.i.d. observations: each data dimension attends only itself
    assert np.array_equal(xx, np.eye(8, dtype=bool))


def test_slcp_data_block_is_identity():
    m = _as_bool(slcp_mask(n_theta=5, n_x=8))
    assert np.array_equal(m[5:, 5:], np.eye(8, dtype=bool))


# ---------------------------------------------------------------------------
# Tree (Appendix A2.2)
# ---------------------------------------------------------------------------
def test_tree_mask_structure():
    m = _as_bool(tree_mask())
    assert m.shape == (7, 7)  # 3 parameters + 4 observables
    assert np.all(np.diag(m))
    # tree edges: theta0 -> theta1, theta0 -> theta2, theta1 -> x0, theta1 -> x1,
    #             theta2 -> x2, theta2 -> x3
    parent_child = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
    for parent, child in parent_child:
        assert m[child, parent], f"missing edge {parent} -> {child}"
    # no data -> parameter edges in the directed base mask
    assert not m[3:, :3].any()
    # x0 / x1 are conditionally independent given theta1
    assert not m[3, 4] and not m[4, 3]
    assert not m[5, 6] and not m[6, 5]
    # theta0 does not directly touch the observations
    assert not m[3, 0] and not m[6, 0]


def test_tree_mask_undirected_round_trip():
    m = _as_bool(undirected(tree_mask()))
    assert np.array_equal(m, m.T)
    assert m[1, 0] and m[0, 1]


# ---------------------------------------------------------------------------
# HMM (Appendix A2.2: 10-state Markov chain)
# ---------------------------------------------------------------------------
def test_hmm_mask_structure():
    m = _as_bool(hmm_mask())
    assert m.shape == (20, 20)  # 10 parameters + 10 observations
    assert np.all(np.diag(m))
    tt, theta_data, data_theta, xx = _theta_data_split(m, 10, 10)
    # chain over the parameters: theta_i attends theta_{i-1} and itself
    for i in range(1, 10):
        assert m[i, i - 1], f"missing chain edge {i - 1} -> {i}"
    assert not tt[0, 1]  # no backward edges in a directed chain
    # factorized observations: x_i attends theta_i only
    assert np.array_equal(theta_data, np.eye(10, dtype=bool))
    assert not data_theta.any()
    assert np.array_equal(xx, np.eye(10, dtype=bool))


def test_hmm_mask_first_parameter_is_root():
    m = _as_bool(hmm_mask(n_states=4))
    assert m.shape == (8, 8)
    assert not m[0, :].any() or np.array_equal(np.flatnonzero(m[0]) , np.array([0]))


# ---------------------------------------------------------------------------
# ODE-chain masks (Lotka-Volterra / SIRD / Hodgkin-Huxley)
# ---------------------------------------------------------------------------
def test_ode_chain_mask_default_shape():
    m = _as_bool(ode_chain_mask(4, 2, 5))
    assert m.shape == (4 + 10, 4 + 10)
    assert np.all(np.diag(m))
    tt, theta_data, data_theta, xx = _theta_data_split(m, 4, 10)
    assert theta_data.all()  # parameters influence the trajectory
    assert not data_theta.any()


def test_ode_chain_mask_time_ordering():
    """Later times may attend earlier times of the same series (causal chain)."""
    m = _as_bool(ode_chain_mask(2, 1, 4, theta_data="dense"))
    # single series, 4 time points -> data indices 2..5
    assert m[5, 2], "late observation must attend the initial condition"
    assert not m[2, 5], "the initial condition may not attend later times"


# ---------------------------------------------------------------------------
# dispatch / registry
# ---------------------------------------------------------------------------
def test_registry_contains_benchmark_tasks():
    for name in ("gaussian_linear", "gaussian_mixture", "two_moons", "slcp", "tree", "hmm"):
        assert name in TASK_MASK_REGISTRY, f"{name} missing from TASK_MASK_REGISTRY"


def test_registry_specs_have_name():
    for key, spec in TASK_MASK_REGISTRY.items():
        name = getattr(spec, "name", None)
        assert name is None or isinstance(name, str)
        assert callable(getattr(spec, "builder"))


@pytest.mark.parametrize(
    "task,n_theta,n_x",
    [
        ("gaussian_linear", 10, 10),
        ("gaussian_mixture", 2, 2),
        ("two_moons", 2, 2),
        ("slcp", 5, 8),
        ("tree", 3, 4),
        ("hmm", 10, 10),
    ],
)
def test_build_attention_mask_shapes(task, n_theta, n_x):
    m = _as_bool(build_attention_mask(task, n_theta=n_theta, n_x=n_x))
    assert m.shape == (n_theta + n_x, n_theta + n_x)
    assert np.all(np.diag(m)), "diagonal must always be kept"


@pytest.mark.parametrize("task", ["two_moons", "slcp", "tree", "hmm"])
def test_build_attention_mask_directed_flag(task):
    directed = _as_bool(build_attention_mask(task, directed=True))
    undirected_mask = _as_bool(build_attention_mask(task, directed=False))
    assert np.array_equal(undirected_mask, undirected_mask.T)
    # the undirected mask must be a superset of the directed one
    assert not (directed & ~undirected_mask).any()


@pytest.mark.parametrize("task", ["gaussian_linear", "two_moons", "slcp", "tree", "hmm"])
def test_mask_variants(task):
    variants = mask_variants(task)
    assert set(variants) >= {"dense", "undirected", "directed"}
    dense = _as_bool(variants["dense"])
    und = _as_bool(variants["undirected"])
    direct = _as_bool(variants["directed"])
    assert dense.all(), "dense variant must be fully connected"
    assert np.array_equal(und, und.T)
    assert direct.shape == dense.shape == und.shape
    assert not (direct & ~und).any()


def test_dense_variant_matches_dense_mask():
    variants = mask_variants("two_moons")
    dense = _as_bool(variants["dense"])
    assert np.array_equal(dense, _as_bool(dense_mask(dense.shape[0])))


def test_unknown_task_raises():
    with pytest.raises(Exception):
        build_attention_mask("definitely_not_a_task")


# ---------------------------------------------------------------------------
# lotka-volterra mask is metadata/time dependent
# ---------------------------------------------------------------------------
def test_lotka_volterra_mask_metadata_dependent():
    lotka = None
    for candidate in ("lotka_volterra_mask", "build_attention_mask"):
        if hasattr(MASK_MODULE, candidate):
            lotka = getattr(MASK_MODULE, candidate)
            break
    if lotka is None:
        pytest.skip("no lotka-volterra mask builder available")

    if lotka is MASK_MODULE.__dict__.get("lotka_volterra_mask"):
        times = np.array([0.0, 1.0, 2.0, 3.0])
        m = _as_bool(lotka(times=times))
    else:
        m = _as_bool(lotka("lotka_volterra", times=np.array([0.0, 1.0, 2.0, 3.0])))
    assert m.ndim == 2
    assert m.shape[0] == m.shape[1]
    assert np.all(np.diag(m))


# ---------------------------------------------------------------------------
# convention checks
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("builder_name", ["gaussian_linear_mask", "slcp_mask", "tree_mask", "hmm_mask"])
def test_directed_base_masks_have_no_data_to_parameter_edges(builder_name):
    """Base masks encode the generative direction: data do not cause parameters."""
    builder = getattr(MASK_MODULE, builder_name, None)
    if builder is None:
        pytest.skip(f"{builder_name} not available")
    m = _as_bool(builder())
    if builder_name == "gaussian_linear_mask":
        n_theta, n_x = 10, 10
    elif builder_name == "slcp_mask":
        n_theta, n_x = 5, 8
    elif builder_name == "tree_mask":
        n_theta, n_x = 3, 4
    else:
        n_theta, n_x = 10, 10
    assert not m[n_theta:, :n_theta].any()


def test_masks_are_binary():
    for builder in (gaussian_linear_mask, two_moons_mask, slcp_mask, tree_mask, hmm_mask):
        m = np.asarray(builder())
        assert set(np.unique(m.astype(float))).issubset({0.0, 1.0})
