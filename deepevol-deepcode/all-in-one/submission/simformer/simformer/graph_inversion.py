"""Graph inversion (Webb et al., 2018) for the Simformer attention mask.

The Simformer can enforce directed dependency structures via a non-symmetric
attention mask ``M_E`` (Section 3.2).  A directed graphical model, however, is
*not* closed under conditioning/marginalization: if we condition on some of the
variables (the "given"/observed variables), the remaining dependencies between
the latent variables may no longer be faithfully represented by the original
directed mask.  With :math:`M_C` we therefore dynamically adapt the mask at
train/inference time using the graph-inversion algorithm of Webb et al. (2018)
(Appendix Sec. A1.1, Addendum "Graph Inversion").

The algorithm, in the paper's own notation:

1. Input: Joint Bayesian net structure :math:`G` as mask :math:`M_E`, latent
   variables :math:`Z` as given by :math:`M_C`.
2. :math:`J \\leftarrow \\mathrm{MORALIZE}(G)`  (make undirected + connect parents).
3. Set all vertices of :math:`J` to be unmarked.
4. :math:`H \\leftarrow \\{\\mathrm{VARIABLES}(G), \\emptyset\\}`, i.e. an unconnected graph.
5. :math:`S \\leftarrow` all latent variables without latent parent in :math:`G`.
6. while :math:`S \\neq \\emptyset` do
7.     Select :math:`v \\in S` according to the min-fill criterion.
8.     Add edges in :math:`J` between unmarked neighbours of :math:`v`.
9.     Make unmarked neighbours of :math:`v` in :math:`J`, :math:`v`'s parents in :math:`H`.
10.    Mark :math:`v` and remove from :math:`S`.
11.    for unmarked child latents :math:`u` of :math:`v` in :math:`G` do
12.        Add :math:`u` to :math:`S` if all its parent latents in :math:`G` are marked.
13.    end for
14. end while
15. return :math:`H`.

To produce the final attention mask, the edges in :math:`H` are added to the
base attention mask :math:`M_E` (i.e. ``M_final = combine_masks(M_E, H)``).

Mask convention (matches :mod:`simformer.attention_masks` and the tokenizer):
``M[i, j] = 1`` means *query token ``i`` may attend key token ``j``*, i.e. there
is a directed edge ``j -> i`` in the dependency graph.  Equivalently, ``j`` is a
*parent* of ``i``.  Consequently, "make unmarked neighbours of ``v`` in ``J``,
``v``'s parents in ``H``" is written as ``H[v, u] = 1`` for every unmarked
neighbour ``u`` of ``v`` in ``J``.

Sanity checks against the paper (Appendix Sec. A1.1):
  * For the **likelihood** (data latent, parameters given) no additional edges
    have to be introduced.
  * For the **posterior** (parameters latent, data given) additional edges are
    inserted into the upper right corner, i.e. the parameters attend the data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from .attention_masks import (
    add_edges,
    combine_masks,
    ensure_diagonal,
    undirected,
)

__all__ = [
    "GraphInversionResult",
    "parents_from_mask",
    "children_from_mask",
    "moralize",
    "moralize_mask",
    "min_fill_order",
    "min_fill_select",
    "graph_inversion",
    "graph_inversion_batched",
    "adapt_attention_mask",
    "adapt_mask_for_conditioning",
    "make_condition_aware_mask",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _as_bool_mask(mask: np.ndarray) -> np.ndarray:
    """Cast any mask-like array to a boolean ``(n, n)`` matrix."""
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError(f"mask must be square and 2-D, got shape {mask.shape}")
    return mask.astype(bool)


def parents_from_mask(mask: np.ndarray) -> List[Set[int]]:
    """Parents of every node, ``parents[i] = {j : M[i, j] = 1, j != i}``.

    ``M[i, j] = 1`` denotes the directed edge ``j -> i`` (``i`` attends ``j``).
    Self-attention (the diagonal) is not an edge.
    """
    m = _as_bool_mask(mask)
    n = m.shape[0]
    return [set(int(j) for j in np.nonzero(m[i])[0] if j != i) for i in range(n)]


def children_from_mask(mask: np.ndarray) -> List[Set[int]]:
    """Children of every node, ``children[i] = {j : M[j, i] = 1, j != i}``."""
    parents = parents_from_mask(mask)
    n = len(parents)
    children: List[Set[int]] = [set() for _ in range(n)]
    for i, ps in enumerate(parents):
        for p in ps:
            children[p].add(i)
    return children


# ---------------------------------------------------------------------------
# step 2: moralization
# ---------------------------------------------------------------------------
def moralize(mask: np.ndarray) -> np.ndarray:
    """MORALIZE(G): make ``G`` undirected and connect all parents of a node.

    Returns a *symmetric* boolean mask (undirected graph ``J``) containing
    (a) the skeleton of ``G`` and (b) a clique between the parents of every
    node (marrying parents).
    """
    m = _as_bool_mask(mask)
    n = m.shape[0]
    parents = parents_from_mask(m)

    j = m | m.T          # make undirected
    for i in range(n):
        ps = sorted(parents[i])
        for a in ps:
            for b in ps:
                if a != b:
                    j[a, b] = True
    np.fill_diagonal(j, False)
    return j


#: Alias kept for API symmetry with the addendum algorithm listing.
moralize_mask = moralize


# ---------------------------------------------------------------------------
# step 7: min-fill criterion
# ---------------------------------------------------------------------------
def _fill_in_edges(adjacency: np.ndarray, node: int, allowed: Optional[Set[int]] = None) -> List[Tuple[int, int]]:
    """Edges that would have to be added among the neighbours of ``node``."""
    if allowed is None:
        neigh = [int(k) for k in np.nonzero(adjacency[node])[0] if k != node]
    else:
        neigh = [k for k in range(adjacency.shape[0]) if adjacency[node, k] and k in allowed and k != node]
    edges: List[Tuple[int, int]] = []
    for a_idx, a in enumerate(neigh):
        for b in neigh[a_idx + 1:]:
            if not adjacency[a, b]:
                edges.append((a, b))
    return edges


def min_fill_select(
    adjacency: np.ndarray,
    candidates: Iterable[int],
    marked: Set[int],
) -> int:
    """Select ``v in S`` according to the min-fill criterion.

    The min-fill criterion chooses the node whose elimination adds the fewest
    edges to ``J`` ("Node that adds fewest edges below").  Ties are broken by
    the node index for determinism.
    """
    candidates = [int(c) for c in candidates if int(c) not in marked]
    if not candidates:
        raise ValueError("min_fill_select called with an empty candidate set")

    best_node = candidates[0]
    best_cost = None
    unmarked = set(range(adjacency.shape[0])) - marked
    for v in candidates:
        cost = len(_fill_in_edges(adjacency, v, allowed=unmarked))
        if best_cost is None or cost < best_cost or (cost == best_cost and v < best_node):
            best_cost = cost
            best_node = v
    return best_node


def min_fill_order(adjacency: np.ndarray, candidates: Optional[Iterable[int]] = None) -> List[int]:
    """Full elimination order under the min-fill criterion (for testing/debug)."""
    adj = np.array(adjacency, dtype=bool, copy=True)
    remaining = set(int(c) for c in (candidates if candidates is not None else range(adj.shape[0])))
    order: List[int] = []
    marked: Set[int] = set()
    while remaining:
        v = min_fill_select(adj, remaining, marked)
        for a, b in _fill_in_edges(adj, v, allowed=remaining):
            adj[a, b] = True
            adj[b, a] = True
        marked.add(v)
        remaining.discard(v)
        order.append(v)
    return order


# ---------------------------------------------------------------------------
# steps 3-15: the inversion algorithm
# ---------------------------------------------------------------------------
@dataclass
class GraphInversionResult:
    """Container for the outcome of :func:`graph_inversion`.

    Attributes
    ----------
    mask : np.ndarray
        ``H`` as a directed attention mask (edges added to ``M_E``).
    edges : List[Tuple[int, int]]
        The edges of ``H`` as ``(parent, child)`` index pairs.
    order : List[int]
        Order in which the latent variables were processed.
    moral : np.ndarray
        The moralized graph ``J`` (after the intermediate edge additions).
    """

    mask: np.ndarray
    edges: List[Tuple[int, int]]
    order: List[int]
    moral: np.ndarray

    def merged(self, base_mask: np.ndarray, symmetrize: bool = False) -> np.ndarray:
        """Add the edges of ``H`` to the base attention mask ``M_E``."""
        final = combine_masks(base_mask, self.mask)
        if symmetrize:
            final = undirected(final)
        return ensure_diagonal(final, True)


def graph_inversion(
    mask: np.ndarray,
    latent_mask: Optional[np.ndarray] = None,
    condition_mask: Optional[np.ndarray] = None,
    *,
    return_result: bool = False,
) -> Union[np.ndarray, GraphInversionResult]:
    """Run the graph-inversion algorithm of Webb et al. (2018).

    Parameters
    ----------
    mask :
        The directed base attention mask :math:`M_E` (``M[i, j] = 1`` means
        query ``i`` attends key ``j``, i.e. edge ``j -> i``).
    latent_mask / condition_mask :
        One of the two must be given.  ``condition_mask[i] = True`` marks a
        *conditioned* (observed/given) variable, so the latent variables are
        ``Z = ~condition_mask`` (this matches the Simformer convention where
        ``M_C = True`` means "clamped to the clean value").  ``latent_mask``
        directly specifies :math:`Z`.
    return_result :
        If ``True`` return a :class:`GraphInversionResult`, otherwise only the
        mask ``H`` (as ``float`` 0/1 matrix).

    Returns
    -------
    ``H`` (or a :class:`GraphInversionResult`).  The final attention mask is
    obtained by adding the edges of ``H`` to ``M_E`` (see
    :meth:`GraphInversionResult.merged` / :func:`adapt_attention_mask`).
    """
    m = _as_bool_mask(mask)
    n = m.shape[0]

    if latent_mask is None:
        if condition_mask is None:
            raise ValueError("either `latent_mask` or `condition_mask` must be provided")
        condition = np.asarray(condition_mask).astype(bool).reshape(-1)
        if condition.shape[0] != n:
            raise ValueError(f"condition_mask has length {condition.shape[0]}, expected {n}")
        latent = ~condition
    else:
        latent = np.asarray(latent_mask).astype(bool).reshape(-1)
        if latent.shape[0] != n:
            raise ValueError(f"latent_mask has length {latent.shape[0]}, expected {n}")

    # ---- 2. J <- MORALIZE(G)
    j = moralize(m)
    # ---- 3. all vertices unmarked
    marked: Set[int] = set()
    # ---- 4. H <- unconnected graph on the same variables
    h = np.zeros((n, n), dtype=bool)
    edges: List[Tuple[int, int]] = []
    order: List[int] = []

    parents = parents_from_mask(m)
    children = children_from_mask(m)
    latent_set = set(int(i) for i in np.nonzero(latent)[0])

    # ---- 5. S <- all latent variables without latent parent in G
    s: List[int] = [
        v for v in sorted(latent_set)
        if not any(p in latent_set for p in parents[v])
    ]

    # ---- 6. while S != empty
    while s:
        # ---- 7. select v in S by min-fill criterion
        v = min_fill_select(j, s, marked)

        # ---- 8. add edges in J between unmarked neighbours of v
        unmarked_neighbours = [
            u for u in range(n) if j[v, u] and u not in marked and u != v
        ]
        for a_idx, a in enumerate(unmarked_neighbours):
            for b in unmarked_neighbours[a_idx + 1:]:
                j[a, b] = True
                j[b, a] = True

        # ---- 9. make unmarked neighbours of v in J, v's parents in H
        for u in unmarked_neighbours:
            if not h[v, u]:
                h[v, u] = True
                edges.append((u, v))

        # ---- 10. mark v and remove it from S
        marked.add(v)
        if v in s:
            s.remove(v)
        order.append(v)

        # ---- 11./12. children of v become eligible once all their latent
        #              parents are marked
        for u in sorted(children[v]):
            if u in latent_set and u not in marked and u not in s:
                if all(p in marked for p in parents[u]):
                    s.append(u)

    if return_result:
        return GraphInversionResult(mask=h.astype(float), edges=edges, order=order, moral=j)
    return h.astype(float)


def graph_inversion_batched(
    mask: np.ndarray,
    condition_masks: np.ndarray,
    *,
    return_results: bool = False,
) -> Union[np.ndarray, List[GraphInversionResult]]:
    """Run the inversion for every sample of a batch of condition masks.

    ``condition_masks`` has shape ``(B, n)`` (``True`` = conditioned).  Returns
    a stack ``(B, n, n)`` of ``H`` masks, or the list of results.
    """
    condition_masks = np.asarray(condition_masks)
    if condition_masks.ndim == 1:
        condition_masks = condition_masks[None, :]

    results: List[GraphInversionResult] = []
    for b in range(condition_masks.shape[0]):
        res = graph_inversion(mask, condition_mask=condition_masks[b], return_result=True)
        results.append(res)

    if return_results:
        return results
    return np.stack([r.mask for r in results], axis=0)


# ---------------------------------------------------------------------------
# applying the inversion to the base attention mask
# ---------------------------------------------------------------------------
def adapt_attention_mask(
    mask: np.ndarray,
    condition_mask: np.ndarray,
    *,
    symmetrize: bool = False,
    batched: bool = False,
) -> np.ndarray:
    """Adapt a directed attention mask to the given condition state(s).

    ``mask`` is the base directed mask :math:`M_E` (shape ``(n, n)``).
    ``condition_mask`` is either a single ``(n,)`` array or a batch ``(B, n)``
    with ``True`` denoting conditioned/observed variables.  The returned mask is
    ``combine_masks(M_E, H)`` where ``H`` is produced by
    :func:`graph_inversion`; optionally the result is symmetrized (undirected
    masks do not require the inversion in the first place, but this is provided
    for completeness).

    If ``batched=True`` (or ``condition_mask`` is 2-D) the output has shape
    ``(B, n, n)``, i.e. one mask per batch element.
    """
    condition_mask = np.asarray(condition_mask)
    if condition_mask.ndim == 2:
        batched = True

    if not batched:
        res = graph_inversion(mask, condition_mask=condition_mask, return_result=True)
        return res.merged(mask, symmetrize=symmetrize)

    condition_masks = condition_mask if condition_mask.ndim == 2 else condition_mask[None, :]
    out = []
    for b in range(condition_masks.shape[0]):
        res = graph_inversion(mask, condition_mask=condition_masks[b], return_result=True)
        out.append(res.merged(mask, symmetrize=symmetrize))
    return np.stack(out, axis=0)


#: Backwards-compatible alias ("dynamic masks" for conditioning).
adapt_mask_for_conditioning = adapt_attention_mask


def make_condition_aware_mask(
    base_mask: np.ndarray,
    condition_mask: np.ndarray,
    *,
    directed: bool = True,
    batched: bool = False,
) -> np.ndarray:
    """Convenience wrapper producing the final (condition-aware) attention mask.

    For ``directed=False`` the base mask is simply symmetrized and returned (the
    graph inversion is only needed for *directed* masks, cf. Appendix A1.1).
    """
    if not directed:
        m = undirected(base_mask)
        if np.asarray(condition_mask).ndim == 2 and batched:
            b = np.asarray(condition_mask).shape[0]
            return np.repeat(m[None, :, :], b, axis=0)
        return m
    return adapt_attention_mask(base_mask, condition_mask, batched=batched)
