"""Attention masks ``M_E`` and the graph inversion of Webb et al. (2018).

Sec. 3.2 of the paper: the Simformer can incorporate known dependency structure
of the simulator through the *attention mask* of the transformer.  A mask entry
``M_E[i, j] = 1`` means that variable ``i`` may attend to (and hence depend on)
variable ``j``, i.e. ``j`` is a parent of ``i``.

If the dependency structure is *directed*, the mask has to be adapted whenever
the set of variables that are conditioned on changes: to sample the posterior we
need edges pointing from the data to the parameters, which are not present in
the generative (likelihood) direction.  The paper uses *graph inversion*
(Algorithm 1 of the addendum, following Webb et al., 2018):

```
Input: joint Bayesian net structure G as mask M_E, latent variables Z
1  J <- MORALIZE(G)                       # undirected + connect parents
2  set all vertices of J unmarked
3  H <- {VARIABLES(G), empty edge set}
4  S <- all latent variables without latent parent in G
5  while S != empty:
6      select v in S according to min-fill criterion
7      add edges in J between unmarked neighbours of v
8      make unmarked neighbours of v in J, v's parents in H
9      mark v and remove from S
10     for unmarked child latents u of v in G:
11         add u to S if all its parent latents in G are marked
12 return H
```

The edges of ``H`` are added to the base attention mask ``M_E`` to obtain the
final attention mask.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
#  Graph utilities
# --------------------------------------------------------------------------- #
def parents(mask: np.ndarray, v: int) -> np.ndarray:
    """Parents of ``v`` given an attention-style adjacency mask ``mask[i, j]``."""
    p = np.flatnonzero(mask[v] > 0)
    return p[p != v]


def children(mask: np.ndarray, v: int) -> np.ndarray:
    """Children of ``v`` (nodes that attend to ``v``)."""
    c = np.flatnonzero(mask[:, v] > 0)
    return c[c != v]


def moralize(mask: np.ndarray) -> np.ndarray:
    """Make the graph undirected and connect all parents of a common child.

    The diagonal is set to ``True``: a variable is always allowed to attend to
    itself.
    """
    n = mask.shape[0]
    adjacency = np.asarray(mask) > 0
    J = adjacency | adjacency.T
    for v in range(n):                       # connect co-parents
        p = parents(adjacency, v)
        if p.size > 1:
            J[np.ix_(p, p)] = True
    np.fill_diagonal(J, True)
    return J


def _neighbours(J: np.ndarray, v: int, unmarked: np.ndarray) -> np.ndarray:
    nb = np.flatnonzero(J[v] > 0)
    nb = nb[(nb != v) & unmarked[nb]]
    return nb


def graph_inversion(base_mask: np.ndarray,
                    condition_state: np.ndarray) -> np.ndarray:
    """Graph inversion (Webb et al., 2018) as described in the addendum.

    Parameters
    ----------
    base_mask:
        ``(n, n)`` directed attention mask ``M_E`` of the generative model.
    condition_state:
        ``(n,)`` in ``{0, 1}``.  ``1`` marks a conditioned (observed) variable,
        ``0`` a latent variable.

    Returns
    -------
    ``(n, n)`` boolean matrix with the edges of ``H`` (``H[i, j] = 1``: ``i``
    attends to ``j``), to be added to ``base_mask``.
    """
    base_mask = np.asarray(base_mask) > 0
    condition_state = np.asarray(condition_state).astype(bool)
    n = base_mask.shape[0]
    latent = ~condition_state

    J = moralize(base_mask)
    unmarked = np.ones(n, dtype=bool)
    H = np.zeros((n, n), dtype=bool)

    # S <- all latent variables without latent parent in G
    S = [v for v in range(n)
         if latent[v] and not np.any(latent[parents(base_mask, v)])]

    while S:
        # --- min-fill criterion: pick the node that adds the fewest edges
        best_v, best_fill = None, np.inf
        for v in S:
            nb = _neighbours(J, v, unmarked)
            if nb.size <= 1:
                fill = 0
            else:
                sub = J[np.ix_(nb, nb)]
                fill = int(np.sum(~sub) // 2)
            if fill < best_fill:
                best_v, best_fill = v, fill
        v = best_v

        # --- add edges in J between unmarked neighbours of v (fill-in)
        nb = _neighbours(J, v, unmarked)
        if nb.size > 1:
            J[np.ix_(nb, nb)] = True

        # --- unmarked neighbours of v in J become v's parents in H
        nb = _neighbours(J, v, unmarked)
        for u in nb:
            H[v, u] = True

        unmarked[v] = False
        S.remove(v)

        # --- once all latent parents of a latent child are eliminated it can be
        #     added to the elimination order.
        for u in children(base_mask, v):
            if not latent[u] or u in S or not unmarked[u]:
                continue
            p = parents(base_mask, u)
            latent_parents = p[latent[p]]
            if all(not unmarked[pp] for pp in latent_parents):
                S.append(u)
    return H


def inversion_attention_mask(base_mask: np.ndarray,
                             condition_state: np.ndarray) -> np.ndarray:
    """Final attention mask: base mask with the inverted edges added."""
    base_mask = np.asarray(base_mask) > 0
    H = graph_inversion(base_mask, condition_state)
    out = base_mask | H
    np.fill_diagonal(out, True)
    return out


def moral_neighbour_mask(base_mask: np.ndarray,
                         condition_state_batch: np.ndarray) -> np.ndarray:
    """Vectorised approximation of :func:`inversion_attention_mask`.

    The conditional independencies encoded by the attention mask require that a
    latent variable can attend to the variables it is *not* independent of after
    conditioning, i.e. to its neighbours in the moralised graph whose dependency
    is re-introduced by conditioning.  This is exactly the (first-order part of
    the) set of edges added by graph inversion:

        ``M[i, j] = M_E[i, j] or (latent_i and moralised_adjacency[i, j])``

    The full algorithm additionally adds triangulation ("fill-in") edges during
    the min-fill elimination; this vectorised variant is used for the randomly
    sampled condition masks of a training batch (where graph inversion would have
    to be run for every batch element separately), see
    :meth:`simformer.model.Simformer.build_attention_mask`.
    """
    base_mask = np.asarray(base_mask) > 0
    J = moralize(base_mask)
    np.fill_diagonal(J, True)
    condition_state_batch = np.asarray(condition_state_batch)
    latent = (condition_state_batch <= 0.5)             # (B, n)
    # (B, n, 1) & (1, n, n) -> (B, n, n)
    dynamic = latent[:, :, None] & J[None, :, :]
    out = base_mask[None, :, :] | dynamic
    diag = np.eye(base_mask.shape[0], dtype=bool)[None, :, :]
    return out | diag


# --------------------------------------------------------------------------- #
#  Base attention masks of the tasks of the paper (addendum, "Task Dependencies")
# --------------------------------------------------------------------------- #
def _block_mask(*blocks: np.ndarray) -> np.ndarray:
    """Concatenate blocks of a block matrix given as already joined row blocks."""
    return np.concatenate(blocks, axis=0)


def gaussian_linear_mask(n_params: int = 10, n_data: int = 10) -> np.ndarray:
    """Data depends on parameters but is factorised across dimensions."""
    assert n_params == n_data
    M_tt = np.eye(n_params)
    M_xx = np.eye(n_data)
    zeros = np.zeros((n_params, n_data))
    M_tx = np.eye(n_data, n_params)
    return _block_mask(np.hstack([M_tt, zeros]), np.hstack([M_tx, M_xx]))


def dense_data_mask(n_params: int = 2, n_data: int = 2) -> np.ndarray:
    """Two Moons / Gaussian Mixture: each data variable depends on all parameters
    and on the previous data variables."""
    M_tt = np.eye(n_params)
    M_xx = np.tril(np.ones((n_data, n_data)))
    zeros = np.zeros((n_params, n_data))
    M_tx = np.ones((n_data, n_params))
    return _block_mask(np.hstack([M_tt, zeros]), np.hstack([M_tx, M_xx]))


def slcp_mask(n_params: int = 5, n_data: int = 8, dim_per_obs: int = 2) -> np.ndarray:
    """SLCP: dense parameter-data dependence, i.i.d. observations."""
    M_tt = np.eye(n_params)
    blocks = [np.tril(np.ones((dim_per_obs, dim_per_obs)))
              for _ in range(n_data // dim_per_obs)]
    M_xx = np.zeros((n_data, n_data))
    offset = 0
    for b in blocks:
        k = b.shape[0]
        M_xx[offset:offset + k, offset:offset + k] = b
        offset += k
    zeros = np.zeros((n_params, n_data))
    M_tx = np.ones((n_data, n_params))
    return _block_mask(np.hstack([M_tt, zeros]), np.hstack([M_tx, M_xx]))


def tree_mask(n_params: int = 3, n_data: int = 4,
              param_parents: Optional[dict] = None,
              data_parents: Optional[dict] = None) -> np.ndarray:
    """Tree structure: diagonal is always true, edges follow the tree.

    ``theta_1, theta_2`` depend on ``theta_0``; ``x_1, x_2`` depend on
    ``theta_1`` and ``x_3, x_4`` on ``theta_2`` (addendum).  The addendum lists a
    ``10 x 10`` template; we build the same structure for the actual number of
    variables of the task.
    """
    if param_parents is None:
        param_parents = {0: [], 1: [0], 2: [0]}
    if data_parents is None:
        data_parents = {0: [1], 1: [1], 2: [2], 3: [2]}
    n = n_params + n_data
    M = np.eye(n)
    for child, ps in param_parents.items():
        for p in ps:
            M[child, p] = 1.0
    for child, ps in data_parents.items():
        for p in ps:
            M[n_params + child, p] = 1.0
            M[n_params + child, n_params + child] = 1.0
    return M


def hmm_mask(n_params: int = 10, n_data: int = 10) -> np.ndarray:
    """Hidden Markov model: Markov chain in the parameters, factorised data."""
    M_tt = np.eye(n_params) + np.diag(np.ones(n_params - 1), k=-1)
    M_xx = np.eye(n_data)
    zeros = np.zeros((n_params, n_data))
    M_tx = np.eye(n_params, n_data)
    return _block_mask(np.hstack([M_tt, zeros]), np.hstack([M_tx, M_xx]))


def lotka_volterra_mask(prey_times: np.ndarray, predator_times: np.ndarray,
                        prey_param_indices: Tuple[int, ...] = (0, 1),
                        predator_param_indices: Tuple[int, ...] = (2, 3),
                        ) -> np.ndarray:
    """Metadata dependent mask of the Lotka-Volterra task.

    The dynamics are Markovian, so ``M_x1x1 = M_x2x2 = I + subdiag`` among the
    *selected* time points.  Data variables of a species depend on the
    corresponding parameters, and the cross-species dependence is causal: a
    variable only depends on the (temporally) previous variables of the other
    species.
    """
    prey_times = np.asarray(prey_times, dtype=float)
    predator_times = np.asarray(predator_times, dtype=float)
    T1, T2 = len(prey_times), len(predator_times)
    n_params = 4
    n_data = T1 + T2
    n = n_params + n_data
    M = np.eye(n)

    # parameter -> data
    for p in prey_param_indices:
        M[n_params:n_params + T1, p] = 1.0
    for p in predator_param_indices:
        M[n_params + T1:n_params + T1 + T2, p] = 1.0

    # Markovian within each species (only among selected time points)
    prey_order = np.argsort(prey_times, kind="stable")
    for a in range(len(prey_order)):
        for b in range(a):
            M[n_params + prey_order[a], n_params + prey_order[b]] = 1.0
    pred_order = np.argsort(predator_times, kind="stable")
    for a in range(len(pred_order)):
        for b in range(a):
            M[n_params + T1 + pred_order[a],
              n_params + T1 + pred_order[b]] = 1.0

    # causal cross-species dependence
    for i, ti in enumerate(prey_times):
        for j, tj in enumerate(predator_times):
            if tj < ti:
                M[n_params + i, n_params + T1 + j] = 1.0
            if ti < tj:
                M[n_params + T1 + j, n_params + i] = 1.0
    return M


def sird_mask(beta_times: np.ndarray, data_times: np.ndarray,
              n_global_params: int = 2) -> np.ndarray:
    """Mask of the time dependent SIRD task.

    * ``beta(t)`` follows a Gaussian process prior, i.e. all entries of ``beta``
      are (a priori) dependent on each other,
    * the global parameters (recovery / death rate) are independent of each other
      and of ``beta``,
    * an observation at time ``t`` depends on ``beta(t')`` for all ``t' <= t``
      (the contact rate at later times cannot influence the past) as well as on
      the previous observations.
    """
    beta_times = np.asarray(beta_times, dtype=float).reshape(-1)
    data_times = np.asarray(data_times, dtype=float).reshape(-1)
    n_beta = beta_times.size
    n_data = data_times.size
    n = n_global_params + n_beta + n_data
    M = np.eye(n)

    # GP prior: dense dependence between the beta variables
    beta_slice = slice(n_global_params, n_global_params + n_beta)
    M[beta_slice, beta_slice] = 1.0

    # data -> previous data (Markov chain) and data -> beta(t' <= t), global params
    data_slice = slice(n_global_params + n_beta, n)
    for i, ti in enumerate(data_times):
        M[n_global_params + n_beta + i, :n_global_params] = 1.0
        for j, tj in enumerate(data_times):
            if tj < ti:
                M[n_global_params + n_beta + i,
                  n_global_params + n_beta + j] = 1.0
        for j, tj in enumerate(beta_times):
            if tj <= ti:
                M[n_global_params + n_beta + i, n_global_params + j] = 1.0
    return M


def dense_mask(n_variables: int) -> np.ndarray:
    """Fully connected attention mask (the "dense" Simformer variant)."""
    return np.ones((n_variables, n_variables))


def undirected(mask: np.ndarray) -> np.ndarray:
    """Symmetrise a directed mask (the "undirected graph" Simformer variant)."""
    mask = np.asarray(mask) > 0
    out = mask | mask.T
    np.fill_diagonal(out, True)
    return out
