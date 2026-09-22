"""Attention masks encoding dependency structures (Paper Sec. 3.2, App. A1.1, Addendum).

The Simformer represents knowledge about conditional dependency structures between
parameters ``theta`` and data ``x`` in the attention mask :math:`M_E` of the transformer
(paper Sec. 3.2).  Two flavours are distinguished:

* **undirected** masks are symmetric (obtained by making the directed mask undirected),
* **directed** masks are non-symmetric and encode causal relations.  Directed masks have
  to be *dynamically adapted* when dependencies change due to conditioning (Webb et al.,
  2018); this is implemented in :mod:`simformer.graph_inversion`.

Convention
----------
All masks are boolean/float arrays ``M`` of shape ``(n_tokens, n_tokens)``.  Entry
``M[i, j] = 1`` (True) means that token ``i`` (query) *may attend* token ``j`` (key),
i.e. a directed edge ``j -> i`` in the graphical model (information flows from ``j`` to
``i``).  The attention mask passed to the transformer follows the PyTorch convention of
boolean masks where ``True`` means "keep this key".  The diagonal is always ``True``
(self-attention is never blocked) -- the Addendum states "Diagonal is always true".

Variable ordering (matches :mod:`simformer.tokenizer`)::

    [ theta scalars | x scalars | function-valued tokens ]

Tasks (Addendum "Task Dependencies"):
    * Gaussian Linear: data depends on parameters but is factorized across dimensions,
    * Two Moons / Gaussian Mixture: each data variable depends on all parameters and the
      other data variables,
    * SLCP: dense parameter-data dependence (i.i.d. observations),
    * Tree: diagonal always true, follows tree dependencies,
    * HMM: Markov chain for parameters and factorized data,
    * Lotka-Volterra: mask dynamically adapts to the input times.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # scipy is only needed for block_diag construction (Addendum snippet)
    from scipy.linalg import block_diag

    _HAS_SCIPY = True
except Exception:  # pragma: no cover - scipy is a core dependency
    _HAS_SCIPY = False

    def block_diag(*arrs):  # type: ignore[misc]
        raise RuntimeError("scipy is required for block_diag")


__all__ = [
    "DENSE",
    "IDENTITY",
    "dense_mask",
    "identity_mask",
    "blocks",
    "block_diag_mask",
    "ensure_diagonal",
    "undirected",
    "symmetrize",
    "combine_masks",
    "mask_power_dependencies",
    "gaussian_linear_mask",
    "two_moons_mask",
    "gaussian_mixture_mask",
    "slcp_mask",
    "tree_mask",
    "hmm_mask",
    "lotka_volterra_mask",
    "sird_mask",
    "hodgkin_huxley_mask",
    "ode_chain_mask",
    "build_attention_mask",
    "mask_variants",
    "add_edges",
    "to_torch_mask",
    "AttentionMaskSpec",
    "TASK_MASK_REGISTRY",
]


# --------------------------------------------------------------------------------------
# basic building blocks
# --------------------------------------------------------------------------------------
def dense_mask(n: int) -> np.ndarray:
    """Dense (fully connected) mask of size ``n`` -- the "dense" Simformer variant."""
    return np.ones((int(n), int(n)), dtype=bool)


def identity_mask(n: int) -> np.ndarray:
    """Identity mask -- only self-attention, used for marginal estimation (App. A1.4)."""
    return np.eye(int(n), dtype=bool)


def blocks(*mats: np.ndarray) -> np.ndarray:
    """Concatenate quadratic blocks along the diagonal (undirected composition)."""
    mats = [np.asarray(m, dtype=bool) for m in mats]
    return block_diag(*mats).astype(bool)


def block_diag_mask(theta_block: np.ndarray, data_block: np.ndarray,
                    theta_data_block: np.ndarray) -> np.ndarray:
    """Assemble a joint mask from the four sub-blocks.

    Parameters
    ----------
    theta_block : (n_theta, n_theta) - dependencies among parameters,
    data_block : (n_x, n_x) - dependencies among data variables,
    theta_data_block : (n_x, n_theta) - dependencies of data on parameters.

    The ``data -> theta`` block is left as zeros: parameters never depend on data in the
    *generative* directed graphical model (this is exactly the block that graph inversion
    has to fill in to represent the posterior).
    """
    theta_block = np.asarray(theta_block, dtype=bool)
    data_block = np.asarray(data_block, dtype=bool)
    theta_data_block = np.asarray(theta_data_block, dtype=bool)
    n_theta, n_x = theta_block.shape[0], data_block.shape[0]
    if theta_data_block.shape != (n_x, n_theta):
        raise ValueError(
            f"theta_data_block must have shape {(n_x, n_theta)}, got {theta_data_block.shape}"
        )
    M = np.zeros((n_theta + n_x, n_theta + n_x), dtype=bool)
    M[:n_theta, :n_theta] = theta_block
    M[n_theta:, :n_theta] = theta_data_block
    M[n_theta:, n_theta:] = data_block
    return ensure_diagonal(M)


def ensure_diagonal(mask: np.ndarray, value: bool = True) -> np.ndarray:
    """Force the diagonal of a mask to ``value`` (Addendum: "Diagonal is always true")."""
    mask = np.asarray(mask, dtype=bool).copy()
    if value:
        np.fill_diagonal(mask, True)
    else:
        np.fill_diagonal(mask, False)
    return mask


def undirected(mask: np.ndarray) -> np.ndarray:
    """Make a directed mask undirected (Addendum "Task Dependencies")."""
    mask = np.asarray(mask, dtype=bool)
    return ensure_diagonal(mask | mask.T)


# alias used in App. A1 / figure captions
symmetrize = undirected


def combine_masks(*masks: np.ndarray) -> np.ndarray:
    """Union of several masks (used to add edges of ``H`` to ``M_E``)."""
    if len(masks) == 0:
        raise ValueError("combine_masks requires at least one mask")
    out = np.asarray(masks[0], dtype=bool).copy()
    for m in masks[1:]:
        out = out | np.asarray(m, dtype=bool)
    return out


def add_edges(mask: np.ndarray, edges: Iterable[Tuple[int, int]],
              symmetric: bool = False) -> np.ndarray:
    """Add directed edges ``(i, j)`` (meaning token ``i`` attends token ``j``) to a mask."""
    out = np.asarray(mask, dtype=bool).copy()
    for i, j in edges:
        out[int(i), int(j)] = True
        if symmetric:
            out[int(j), int(i)] = True
    return out


def mask_power_dependencies(mask: np.ndarray, n_layers: int) -> np.ndarray:
    """Return ``D = I(M_E^l > 0)`` for an ``l``-layer transformer (App. A1.1).

    The ``n``-th power of the adjacency matrix counts walks of length ``n``, so
    ``M_E^l > 0`` marks all pairs of nodes that are connected through up to ``l`` layers.
    """
    mask = np.asarray(mask, dtype=float)
    if n_layers < 1:
        raise ValueError("n_layers must be >= 1")
    acc = np.linalg.matrix_power(mask, int(n_layers))
    return acc > 0


def to_torch_mask(mask: np.ndarray, device=None, dtype=None):
    """Convert a numpy mask to a boolean torch tensor (transformer convention)."""
    import torch

    t = torch.as_tensor(np.asarray(mask, dtype=bool))
    if device is not None:
        t = t.to(device)
    if dtype is not None:
        t = t.to(dtype)
    return t


# --------------------------------------------------------------------------------------
# task-specific directed base masks (Addendum "Task Dependencies", App. A2.2)
# --------------------------------------------------------------------------------------
def gaussian_linear_mask(n_theta: int = 10, n_x: int = 10) -> np.ndarray:
    """Gaussian Linear task: data depends on parameters, factorized across dimensions.

    ``x_i`` depends only on ``theta_i`` (prior ``N(0, 0.1 I)``, likelihood
    ``N(theta, 0.1 I)`` in Lueckmann et al. (2021)), i.e. the joint is fully factorized
    across the dimension index.  Constructed with ``block_diag`` of ``[[1, 1], [0, 1]]``
    blocks, as in the Addendum snippet.
    """
    n = min(int(n_theta), int(n_x))
    per_dim = np.array([[True, True], [False, True]], dtype=bool)
    M = block_diag(*[per_dim for _ in range(n)]).astype(bool)
    if n_theta != n_x:  # fall back to explicit construction for unequal dims
        M = np.zeros((n_theta + n_x, n_theta + n_x), dtype=bool)
        for i in range(n):
            M[i, i] = True
            M[n_theta + i, i] = True
            M[n_theta + i, n_theta + i] = True
    return ensure_diagonal(M)


def two_moons_mask(n_theta: int = 2, n_x: int = 2) -> np.ndarray:
    """Two Moons: each data variable depends on all parameters and the other data vars."""
    theta_block = identity_mask(n_theta)
    data_block = dense_mask(n_x)
    theta_data = dense_mask(n_x)[:, :n_theta] if n_theta else np.zeros((n_x, 0), bool)
    theta_data = np.ones((n_x, max(n_theta, 1)), dtype=bool)[:, :n_theta] \
        if n_theta else np.zeros((n_x, 0), dtype=bool)
    return block_diag_mask(theta_block, data_block, theta_data)


def gaussian_mixture_mask(n_theta: int = 2, n_x: int = 2) -> np.ndarray:
    """Gaussian Mixture: same dependency structure as Two Moons."""
    return two_moons_mask(n_theta=n_theta, n_x=n_x)


def slcp_mask(n_theta: int = 5, n_x: int = 8) -> np.ndarray:
    """SLCP: dense parameter-data dependence with i.i.d. observations.

    Each of the (4 two-dimensional) observations is conditionally independent given
    ``theta``, hence the data block is the identity.
    """
    theta_block = identity_mask(n_theta)
    data_block = identity_mask(n_x)
    theta_data = np.ones((n_x, n_theta), dtype=bool)
    return block_diag_mask(theta_block, data_block, theta_data)


def tree_mask() -> np.ndarray:
    """Tree task (App. A2.2): ``theta_0 -> theta_1, theta_2`` and
    ``theta_1 -> x_0, x_1``, ``theta_2 -> x_2, x_3``."""
    n_theta, n_x = 3, 4
    M = np.zeros((n_theta + n_x, n_theta + n_x), dtype=bool)

    # theta_1 <- theta_0, theta_2 <- theta_0
    M[1, 0] = True
    M[2, 0] = True
    # x_0, x_1 <- theta_1
    M[n_theta + 0, 1] = True
    M[n_theta + 1, 1] = True
    # x_2, x_3 <- theta_2
    M[n_theta + 2, 2] = True
    M[n_theta + 3, 2] = True
    return ensure_diagonal(M)


def hmm_mask(n_states: int = 10) -> np.ndarray:
    """HMM task (App. A2.2): Markov chain over parameters with factorized observations.

    ``theta_{i+1} <- theta_i`` and ``x_i <- theta_i``.
    """
    n = int(n_states)
    M = np.zeros((2 * n, 2 * n), dtype=bool)
    for i in range(n - 1):
        M[i + 1, i] = True
    for i in range(n):
        M[n + i, i] = True
    return ensure_diagonal(M)


def ode_chain_mask(n_theta: int, n_series: int, n_times: int,
                   series_major: bool = True,
                   theta_block: str = "identity",
                   data_data: str = "chain",
                   connect_series: bool = True,
                   theta_data: str = "dense") -> np.ndarray:
    """Mask for ODE-based simulators with function-valued observations.

    Data tokens are arranged either time-major (``series_major=False``, i.e. for each time
    step all series, matching ``odeint``-style outputs) or series-major
    (``series_major=True``, i.e. full trajectory of series 1, then series 2, ...).

    ``data_data`` controls data-data dependencies:

    * ``"chain"``: an observation at time index ``k`` attends observations of the previous
      time index ``k-1`` (both directions in the state vector if ``connect_series``);
      this mirrors the sequential dependency induced by a numerical ODE integrator,
    * ``"dense"``: all data variables attend each other,
    * ``"identity"``: only self-attention.

    ``theta_data`` controls the parameter -> data block (``"dense"`` or ``"identity"``).
    """
    n_theta, n_series, n_times = int(n_theta), int(n_series), int(n_times)
    n_data = n_series * n_times
    n = n_theta + n_data
    M = np.zeros((n, n), dtype=bool)

    if theta_block == "dense":
        M[:n_theta, :n_theta] = True
    else:
        M[:n_theta, :n_theta] = np.eye(n_theta, dtype=bool)

    if theta_data == "dense":
        M[n_theta:, :n_theta] = True
    elif theta_data == "identity":
        M[n_theta:, :n_theta] = False
    else:
        raise ValueError(f"unknown theta_data: {theta_data}")

    def idx(series: int, time: int) -> int:
        if series_major:
            return n_theta + series * n_times + time
        return n_theta + time * n_series + series

    if data_data == "dense":
        M[n_theta:, n_theta:] = True
    elif data_data == "identity":
        pass
    elif data_data == "chain":
        for t in range(1, n_times):
            for s in range(n_series):
                if connect_series:
                    for s2 in range(n_series):
                        M[idx(s, t), idx(s2, t - 1)] = True
                else:
                    M[idx(s, t), idx(s, t - 1)] = True
    else:
        raise ValueError(f"unknown data_data: {data_data}")

    return ensure_diagonal(M)


def lotka_volterra_mask(times: Optional[Sequence[float]] = None,
                        n_theta: int = 4, n_series: int = 2,
                        n_times: Optional[int] = None,
                        theta_data: str = "dense",
                        theta_block: str = "identity",
                        n_fixed_steps: Optional[int] = None) -> np.ndarray:
    """Lotka-Volterra mask; *dynamically adapts to the input times* (Fig. A4).

    The prey/predator trajectories are generated by integrating the ODE system
    ``dx/dt = alpha x - beta x y``, ``dy/dt = delta x y - gamma y``, possibly with a
    fixed number of internal solver steps.  The attention mask reflects the resulting
    sequential dependency: an observation at time ``t_k`` can attend observations of the
    previous time index (and all four parameters).

    Parameters
    ----------
    times : observation times (used to determine the number of time points).  If the
        number of solver steps ``n_fixed_steps`` is given, internal states are also
        included in the chain.
    """
    if n_times is None:
        if times is not None:
            n_times = len(times)
        elif n_fixed_steps is not None:
            n_times = int(n_fixed_steps) + 1
        else:
            n_times = 15
    n_times = int(n_times)
    if n_fixed_steps is not None:
        n_times = max(n_times, int(n_fixed_steps) + 1)
    return ode_chain_mask(
        n_theta=n_theta,
        n_series=n_series,
        n_times=n_times,
        series_major=False,
        theta_block=theta_block,
        theta_data=theta_data,
        data_data="chain",
        connect_series=True,
    )


def sird_mask(n_theta: int = 3, n_series: int = 4, n_times: int = 20,
              n_index_points: int = 0) -> np.ndarray:
    """SIRD mask: global rates, the function-valued contact rate ``beta(t)`` and the four
    compartment trajectories ``S, I, R, D``.

    ``n_theta`` counts the scalar global variables; the function-valued contact rate is
    represented by ``n_index_points`` additional parameter tokens (one per index point),
    which are treated as parameters and therefore attend all other parameters and all
    data variables.
    """
    n_theta_total = int(n_theta) + int(n_index_points)
    return ode_chain_mask(
        n_theta=n_theta_total,
        n_series=n_series,
        n_times=n_times,
        series_major=False,
        theta_block="dense",
        theta_data="dense",
        data_data="chain",
        connect_series=True,
    )


def hodgkin_huxley_mask(n_theta: int = 4, n_series: int = 1, n_times: int = 100,
                        n_stats: int = 0) -> np.ndarray:
    """Hodgkin-Huxley mask: gating/conductance parameters -> voltage trace.

    The voltage trace is one function-valued data variable whose values are chained in
    time by the numerical integrator.  Optional summary-statistic tokens (``n_stats``)
    attend all data and parameter tokens.
    """
    M = ode_chain_mask(
        n_theta=n_theta,
        n_series=n_series,
        n_times=n_times,
        series_major=True,
        theta_block="identity",
        theta_data="dense",
        data_data="chain",
        connect_series=True,
    )
    if n_stats > 0:
        n = M.shape[0] + int(n_stats)
        M2 = np.ones((n, n), dtype=bool)
        M2[: M.shape[0], : M.shape[0]] = M
        M = ensure_diagonal(M2)
    return M


# --------------------------------------------------------------------------------------
# task registry / dispatch
# --------------------------------------------------------------------------------------
@dataclass
class AttentionMaskSpec:
    """Description of the dependency structure of a task."""

    name: str
    builder: Callable[..., np.ndarray]
    kwargs: Dict = field(default_factory=dict)
    supports_directed: bool = True
    metadata_dependent: bool = False


TASK_MASK_REGISTRY: Dict[str, AttentionMaskSpec] = {
    "gaussian_linear": AttentionMaskSpec("gaussian_linear", gaussian_linear_mask),
    "linear_gaussian": AttentionMaskSpec("gaussian_linear", gaussian_linear_mask),
    "gaussian_mixture": AttentionMaskSpec("gaussian_mixture", gaussian_mixture_mask),
    "two_moons": AttentionMaskSpec("two_moons", two_moons_mask),
    "slcp": AttentionMaskSpec("slcp", slcp_mask),
    "tree": AttentionMaskSpec("tree", tree_mask),
    "hmm": AttentionMaskSpec("hmm", hmm_mask),
    "lotka_volterra": AttentionMaskSpec(
        "lotka_volterra", lotka_volterra_mask, metadata_dependent=True
    ),
    "sird": AttentionMaskSpec("sird", sird_mask),
    "hodgkin_huxley": AttentionMaskSpec("hodgkin_huxley", hodgkin_huxley_mask),
}


def build_attention_mask(task: str, n_theta: Optional[int] = None,
                         n_x: Optional[int] = None,
                         directed: bool = True,
                         **kwargs) -> np.ndarray:
    """Build the base attention mask ``M_E`` for a named task.

    Parameters
    ----------
    task : registered task name (see :data:`TASK_MASK_REGISTRY`).
    n_theta, n_x : sizes of the parameter / data parts (defaults per task).
    directed : if ``False`` the undirected version of the mask is returned
        (Addendum: "The undirected mask is obtained by making it undirected").
    """
    key = str(task).lower()
    if key not in TASK_MASK_REGISTRY:
        raise KeyError(
            f"unknown task '{task}'. Available: {sorted(set(TASK_MASK_REGISTRY))}"
        )
    spec = TASK_MASK_REGISTRY[key]
    kwargs = dict(kwargs)
    defaults = dict(spec.kwargs)
    defaults.update(kwargs)

    sig_kwargs = dict(defaults)
    if "gaussian_linear" in (key, spec.name):
        sig_kwargs.setdefault("n_theta", 10 if n_theta is None else n_theta)
        sig_kwargs.setdefault("n_x", 10 if n_x is None else n_x)
    elif key in ("two_moons", "gaussian_mixture"):
        sig_kwargs.setdefault("n_theta", 2 if n_theta is None else n_theta)
        sig_kwargs.setdefault("n_x", 2 if n_x is None else n_x)
    elif key == "slcp":
        sig_kwargs.setdefault("n_theta", 5 if n_theta is None else n_theta)
        sig_kwargs.setdefault("n_x", 8 if n_x is None else n_x)
    elif key == "hmm":
        if n_theta is not None:
            sig_kwargs.setdefault("n_states", n_theta)

    mask = spec.builder(**sig_kwargs)
    mask = ensure_diagonal(mask)
    return mask if directed else undirected(mask)


def mask_variants(task: str, n_theta: Optional[int] = None,
                  n_x: Optional[int] = None, **kwargs) -> Dict[str, np.ndarray]:
    """Return the three Simformer mask variants compared in Fig. 4.

    * ``"dense"``: fully connected mask,
    * ``"undirected"``: undirected task mask,
    * ``"directed"``: directed task mask (to be adapted by graph inversion when
      conditioning on data).
    """
    base = build_attention_mask(task, n_theta=n_theta, n_x=n_x, directed=True, **kwargs)
    return {
        "dense": dense_mask(base.shape[0]),
        "undirected": undirected(base),
        "directed": base,
    }
