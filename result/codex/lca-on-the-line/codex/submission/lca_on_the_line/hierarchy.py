"""WordNet class hierarchy for ImageNet and the LCA distance matrix.

The hierarchy is the ``imagenet_fiveai`` taxonomy
(https://github.com/jvlmdr/hiercls/blob/main/resources/hierarchy/imagenet_fiveai.csv),
which is the ImageNet hierarchy used by Bertinetto et al. (2020) -- the same
taxonomy adopted by the paper.

The CSV stores ``parent_synset,child_synset`` edges.  Its 1000 leaves are
exactly the 1000 ImageNet classes and correspond one-to-one with the synsets of
the standard ``imagenet_class_index.json``.

Two node scores are implemented, following Appendix D.2.1:

* ``information`` -- ``I(y) = -log2 p(y) = log2 |L| - log2 |L(y)|`` where
  ``p`` is uniform over the 1000 leaf nodes and ``L(y)`` are the leaves below
  ``y`` (this is the definition requested by the paper addendum).  It yields
  ``D_LCA^I(y', y) = I(y) - I(LCA(y, y'))``.
* ``depth`` -- ``D_LCA^P(y', y) = (P(y) - P(LCA)) + (P(y') - P(LCA))`` where
  ``P`` is the depth of a node measured from the root.  The extra
  ``(P(y') - P(LCA))`` term counts the descent to the prediction node and
  counters tree imbalance.  The paper uses this variant for the linear-probing
  experiments (Appendix D.2.1).

For leaf-to-leaf pairs the information variant reduces to the number of leaves
under the lowest common ancestor: ``D = log2(#leaves below LCA)``.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)

DEFAULT_HIERARCHY_CSV = os.path.join(
    _REPO_ROOT, "data", "hierarchy", "imagenet_fiveai.csv"
)
DEFAULT_CLASS_INDEX_JSON = os.path.join(
    _REPO_ROOT, "data", "hierarchy", "imagenet_class_index.json"
)


def load_imagenet_class_index(path: str = DEFAULT_CLASS_INDEX_JSON
                              ) -> Tuple[List[str], List[str]]:
    """Return ``(synsets, names)`` ordered by ImageNet class index 0..999."""
    with open(path, "r") as fh:
        raw = json.load(fh)
    if isinstance(raw, list):  # some copies are a flat list of [synset, name]
        pairs = raw
    else:
        pairs = [raw[str(i)] for i in range(len(raw))]
    synsets = [p[0] for p in pairs]
    names = [p[1] for p in pairs]
    if len(synsets) != 1000:
        raise ValueError(
            "expected 1000 ImageNet classes, found %d in %s" % (len(synsets), path)
        )
    return synsets, names


def _read_edges(path: str) -> List[Tuple[str, str]]:
    edges: List[Tuple[str, str]] = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            edges.append((parts[0].strip(), parts[1].strip()))
    return edges


def _orient_edges(edges: Sequence[Tuple[str, str]]
                  ) -> List[Tuple[str, str]]:
    """Figure out whether the file stores (parent, child) or (child, parent).

    A well formed taxonomy has exactly one root (a node that never appears as a
    child).  We pick the orientation that satisfies that property; the
    ``imagenet_fiveai`` file is ``parent,child``.
    """

    def num_roots(pairs):
        children = {c for _, c in pairs}
        nodes = {n for p in pairs for n in p}
        return len([n for n in nodes if n not in children])

    forward = list(edges)
    backward = [(c, p) for p, c in edges]
    n_fwd, n_bwd = num_roots(forward), num_roots(backward)
    if n_fwd == 1:
        return forward
    if n_bwd == 1:
        return backward
    # Fall back to the documented convention.
    return forward


class WordNetHierarchy:
    """Static taxonomy over the 1000 ImageNet classes."""

    def __init__(
        self,
        edges: Iterable[Tuple[str, str]],
        class_synsets: Sequence[str],
        synset_names: Optional[Dict[str, str]] = None,
        class_names: Optional[Sequence[str]] = None,
    ):
        self.edges: List[Tuple[str, str]] = [(p, c) for p, c in edges]
        self.parent: Dict[str, Optional[str]] = {}
        self.children: Dict[str, List[str]] = {}
        for parent, child in self.edges:
            # A node may appear in only one of the two roles.
            self.parent.setdefault(parent, None)
            self.parent[child] = parent
            self.children.setdefault(parent, [])
            self.children.setdefault(child, [])
            self.children[parent].append(child)

        self.node_names: Dict[str, str] = dict(synset_names or {})
        self.class_synsets: List[str] = list(class_synsets)
        self.class_names: List[str] = list(class_names or class_synsets)
        self.index_of: Dict[str, int] = {
            s: i for i, s in enumerate(self.class_synsets)
        }

        self._build_caches()

    # ------------------------------------------------------------------ #
    # construction helpers
    # ------------------------------------------------------------------ #
    def _build_caches(self) -> None:
        self.roots: List[str] = [
            n for n, p in self.parent.items() if p is None
        ]
        self.height: Dict[str, int] = {}
        self.depth: Dict[str, int] = {}
        self._leaves_below: Dict[str, int] = {}
        self.path_to_root: Dict[str, List[str]] = {}

        leaf_set = set(self.class_synsets)
        order = self._topological_order()
        for node in order:
            kids = self.children.get(node, [])
            if not kids:
                self.height[node] = 0
                self._leaves_below[node] = 1 if node in leaf_set else 0
            else:
                self.height[node] = 1 + max(self.height[k] for k in kids)
                self._leaves_below[node] = sum(
                    self._leaves_below[k] for k in kids
                )

        for node in self.parent:
            path, cur = [], node
            while cur is not None:
                path.append(cur)
                cur = self.parent[cur]
            self.path_to_root[node] = path
            # depth measured from the root (root -> node edge count)
            self.depth[node] = len(path) - 1
        self._path_sets = {n: set(p) for n, p in self.path_to_root.items()}

        self.num_classes = len(self.class_synsets)
        self.total_leaves = int(sum(self._leaves_below[c] for c in self.class_synsets))

    def _topological_order(self) -> List[str]:
        order: List[str] = []
        stack: List[Tuple[str, bool]] = [(r, False) for r in self.roots]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            stack.append((node, True))
            for kid in self.children.get(node, []):
                stack.append((kid, False))
        return order

    # ------------------------------------------------------------------ #
    # node queries
    # ------------------------------------------------------------------ #
    def node_depth(self, node: str) -> int:
        """``P(x)``: number of edges from the root down to ``x``."""
        return self.depth[node]

    def node_height(self, node: str) -> int:
        """Number of edges from ``x`` down to its deepest leaf."""
        return self.height[node]

    def leaves_below(self, node: str) -> int:
        """``|L(x)|``: number of ImageNet classes below ``node``."""
        return self._leaves_below[node]

    def information(self, node: str) -> float:
        """``I(node) = log2 |L| - log2 |L(node)|`` (uniform over leaves)."""
        below = self._leaves_below[node]
        if below <= 0:
            return float(np.log2(self.total_leaves))
        return float(np.log2(self.total_leaves) - np.log2(below))

    def lca(self, a: str, b: str) -> str:
        """Lowest common ancestor of two synsets."""
        if a == b:
            return a
        set_b = self._path_sets[b]
        for node in self.path_to_root[a]:
            if node in set_b:
                return node
        raise ValueError("nodes %r and %r are not connected" % (a, b))

    def is_ancestor(self, ancestor: str, node: str) -> bool:
        return ancestor in self._path_sets[node]

    # ------------------------------------------------------------------ #
    # distances
    # ------------------------------------------------------------------ #
    def lca_node_matrix(self) -> np.ndarray:
        """(n, n) integer matrix of LCA *node ids* (into ``self.nodes``)."""
        nodes = list(self.parent.keys())
        node_id = {n: i for i, n in enumerate(nodes)}
        n = len(self.class_synsets)
        out = np.zeros((n, n), dtype=np.int32)
        for i, a in enumerate(self.class_synsets):
            for j in range(i, n):
                b = self.class_synsets[j]
                out[i, j] = out[j, i] = node_id[self.lca(a, b)]
        self._nodes = nodes
        return out

    def lca_distance_matrix(self, score: str = "information") -> np.ndarray:
        """(n, n) matrix of pairwise LCA distances.

        The matrix stores a *distance*: the diagonal is exactly zero, and the
        off-diagonal entries are non-negative.  ``score`` selects between the
        information-content variant (``D_LCA^I``, used for the benchmark
        measurements) and the tree-depth path variant (``D_LCA^P``, used for the
        linear-probing soft labels).
        """
        score = score.lower()
        n = len(self.class_synsets)
        out = np.zeros((n, n), dtype=np.float64)
        for i, a in enumerate(self.class_synsets):
            for j in range(i + 1, n):
                b = self.class_synsets[j]
                anc = self.lca(a, b)
                if score in ("information", "info", "i"):
                    d = self.information(a) - self.information(anc)
                elif score in ("depth", "path", "p"):
                    d = (self.depth[a] - self.depth[anc]) + (
                        self.depth[b] - self.depth[anc]
                    )
                else:
                    raise ValueError("unknown score %r" % score)
                out[i, j] = out[j, i] = d
        return out

    # ------------------------------------------------------------------ #
    # convenience
    # ------------------------------------------------------------------ #
    def taxonomy_path(self, synset: str, max_depth: int = 3) -> List[str]:
        """Human-readable is-a path from ``synset`` up towards the root."""
        path = []
        cur: Optional[str] = synset
        while cur is not None and len(path) < max_depth + 1:
            path.append(self.node_names.get(cur, cur))
            cur = self.parent[cur]
        return path

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            "WordNetHierarchy(nodes=%d, classes=%d, roots=%s)"
            % (len(self.parent), self.num_classes, self.roots)
        )


def load_wordnet_hierarchy(
    hierarchy_csv: str = DEFAULT_HIERARCHY_CSV,
    class_index_json: str = DEFAULT_CLASS_INDEX_JSON,
    names_json: Optional[str] = None,
) -> WordNetHierarchy:
    """Build the ImageNet WordNet hierarchy from the bundled resources."""
    edges = _orient_edges(_read_edges(hierarchy_csv))
    synsets, names = load_imagenet_class_index(class_index_json)

    node_names: Dict[str, str] = {}
    if names_json and os.path.exists(names_json):
        with open(names_json, "r") as fh:
            node_names = json.load(fh)
    else:
        default_names = os.path.join(
            os.path.dirname(hierarchy_csv), "imagenet_fiveai.json"
        )
        if os.path.exists(default_names):
            with open(default_names, "r") as fh:
                node_names = json.load(fh)

    return WordNetHierarchy(
        edges=edges,
        class_synsets=synsets,
        synset_names=node_names,
        class_names=names,
    )
