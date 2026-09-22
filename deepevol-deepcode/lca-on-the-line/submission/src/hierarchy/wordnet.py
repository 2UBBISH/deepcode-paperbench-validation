"""WordNet class hierarchy for ImageNet-1k.

Paper reference
---------------
* Sec. 2 / Sec. D.2.1 -- LCA distance uses WordNet (Miller et al., 1990) to encode
  the class taxonomy of ImageNet.
* Addendum ("WordNet dataset") -- the taxonomy is the ``imagenet_fiveai.csv`` file
  distributed with https://github.com/jvlmdr/hiercls.
* Addendum ("Sanity checking the LCA distance matrix") -- the resulting matrix must
  store *distances* (zero diagonal), not similarities.

The module builds a rooted tree over the 1000 ImageNet classes:

* ``class_to_leaf``   : ImageNet class index -> hierarchy node (the class node).
* ``parents``         : child node -> parent node (root maps to ``None``).
* ``depth``           : node -> number of edges from the root (root depth 0).
* ``leaf_count``      : node -> number of *class* nodes in its subtree.

The information content of a node (Valmadre, 2022; used in this paper) is

    I(y) = log |L| - log |L(y)|                       (uniform leaf distribution)

where ``|L|`` is the number of leaves (1000 classes) and ``|L(y)|`` the number of
class nodes in the subtree of ``y``.  Because every *class* node has ``|L(y)| = 1``
(unless one ImageNet class is an ancestor of another, which does not happen in
ImageNet-1k), all class nodes share the same information content.  As a
consequence the information-content LCA distance is symmetric:

    D_LCA^I(y', y) = I(y) - I(LCA(y, y')) = log |L| - I(LCA) = log |L(LCA)|,

which equals ``log2`` of the number of leaves under the LCA node.
"""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SYNSET_RE = re.compile(r"^n\d{8}$")
IMAGENET_NUM_CLASSES = 1000

# ---------------------------------------------------------------------------
# WordNet helpers
# ---------------------------------------------------------------------------


def is_synset(token: str) -> bool:
    """Return True if ``token`` looks like a WordNet synset offset id (``n01234567``)."""
    return bool(SYNSET_RE.match(str(token).strip()))


def _wnid_to_synset(wnid: str):
    """Convert an ImageNet synset id (``n01440764``) to an ``nltk`` synset."""
    from nltk.corpus import wordnet as wn  # imported lazily (heavy import)

    return wn.synset_from_pos_and_offset(wnid[0], int(wnid[1:]))


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------


def parse_hierarchy_csv(path: str):
    """Parse a hierarchy CSV into ``(edges, class_to_synset, synset_to_class)``.

    The parser is deliberately tolerant of the different column layouts used by
    public ImageNet hierarchies (notably ``imagenet_fiveai.csv`` from
    ``jvlmdr/hiercls``):

    * rows of the form ``child_synset, parent_synset`` -> tree edges;
    * rows mixing a synset id and an integer class index (``0..999``) ->
      class index <-> synset mapping;
    * a header row (non synset / non integer first row) is skipped.

    The parent/child direction of two-synset rows is inferred: the edge endpoint
    that never occurs in the other column is the root, so the column containing
    the (single) root is the *parent* column.
    """
    rows: List[List[str]] = []
    with open(path, "r", newline="") as handle:
        for raw in csv.reader(handle):
            cells = [c.strip() for c in raw if c is not None and c.strip() != ""]
            if not cells:
                continue
            rows.append(cells)

    if not rows:
        return [], {}, {}

    # Drop a header row.
    first = rows[0]
    has_synset = any(is_synset(c) for c in first)
    has_int = any(c.lstrip("-").isdigit() for c in first)
    if not has_synset and not has_int:
        rows = rows[1:]

    edges: List[Tuple[str, str]] = []
    class_to_synset: Dict[int, str] = {}
    synset_to_class: Dict[str, int] = {}

    for cells in rows:
        synsets = [c for c in cells if is_synset(c)]
        ints = [c for c in cells if c.lstrip("-").isdigit()]
        ints = [int(c) for c in ints if 0 <= int(c) < IMAGENET_NUM_CLASSES]

        if len(synsets) >= 2:
            # Child/parent edge -- direction resolved later.
            edges.append((synsets[0], synsets[1]))
            # A third (integer) column may still carry the class index.
            if len(synsets) >= 3:
                for extra in synsets[2:]:
                    edges.append((synsets[1], extra))
            if ints:
                class_to_synset.setdefault(ints[0], synsets[-1])
                synset_to_class.setdefault(synsets[-1], ints[0])
        elif len(synsets) == 1 and ints:
            class_to_synset.setdefault(ints[0], synsets[0])
            synset_to_class.setdefault(synsets[0], ints[0])
        elif len(synsets) == 1:
            # single-column listing: row order == class index
            idx = len(synset_to_class)
            class_to_synset.setdefault(idx, synsets[0])
            synset_to_class.setdefault(synsets[0], idx)

    if edges:
        edges = _orient_edges(edges)

    return edges, class_to_synset, synset_to_class


def _orient_edges(edges: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Ensure edges are ``(child, parent)`` by locating the unique root."""
    col0 = {a for a, _ in edges}
    col1 = {b for _, b in edges}
    only0 = col0 - col1          # candidates for "never a parent" -> leaves of col0
    only1 = col1 - col0
    # The parent column contains the root, i.e. the column whose values are
    # (almost) never present in the other column has *fewer* such unique values.
    if len(only1) <= len(only0):
        # col1 is the parent column: rows are already (child, parent)
        return list(edges)
    return [(b, a) for a, b in edges]


# ---------------------------------------------------------------------------
# nltk based construction
# ---------------------------------------------------------------------------


def build_parents_from_wnids(wnids: Sequence[str]) -> Dict[str, Optional[str]]:
    """Build a ``child -> parent`` map from a list of ImageNet synset ids.

    Uses the first hypernym path returned by ``nltk``'s WordNet, which keeps the
    induced sub-tree small (only the nodes on the paths to the 1000 classes).
    """
    parents: Dict[str, Optional[str]] = {}
    for wnid in wnids:
        try:
            synset = _wnid_to_synset(wnid)
        except Exception:  # pragma: no cover - depends on nltk data
            logger.warning("Could not resolve synset %s with nltk", wnid)
            continue
        paths = synset.hypernym_paths()
        if not paths:
            parents.setdefault(wnid, None)
            continue
        path = min(paths, key=len)  # root ... leaf
        for child, parent in zip(path[1:], path[:-1]):
            child_id = _synset_id(child)
            parent_id = _synset_id(parent)
            parents.setdefault(child_id, parent_id)
            parents.setdefault(parent_id, None)
        parents.setdefault(wnid, None)
        # make sure the ImageNet id itself is the node we use for the class
        if _synset_id(synset) != wnid:
            parents[wnid] = _synset_id(synset)
    return parents


def _synset_id(synset) -> str:
    """Stable string id for an ``nltk`` synset (its WordNet offset)."""
    return "%s%08d" % (synset.pos(), synset.offset())


def get_imagenet_wnids() -> List[str]:
    """Return the 1000 ImageNet-1k synset ids in canonical (label) order.

    Tries, in order: HuggingFace ``datasets`` metadata, a local text file, and
    finally the ``nltk`` WordNet (not available in a guaranteed order).
    """
    try:  # pragma: no cover - requires the HF dataset metadata
        from datasets import load_dataset

        dataset = load_dataset("imagenet-1k", split="validation", trust_remote_code=True)
        names = dataset.features["label"].names
        if len(names) == IMAGENET_NUM_CLASSES and all(is_synset(n) for n in names):
            return list(names)
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not obtain ImageNet wnids from HuggingFace: %s", exc)
    return []


# ---------------------------------------------------------------------------
# Synthetic fallback hierarchy
# ---------------------------------------------------------------------------


def build_synthetic_parents(
    num_classes: int = IMAGENET_NUM_CLASSES, branching: int = 2, seed: int = 0
) -> Dict[str, Optional[str]]:
    """Deterministic balanced tree over ``num_classes`` leaves.

    Used only when no WordNet data is available (e.g. offline unit tests).  It is
    a valid rooted tree with well defined depths and leaf counts, so all LCA
    property tests (zero diagonal, symmetry, monotonic information content) hold.
    """
    rng = np.random.RandomState(seed)
    parents: Dict[str, Optional[str]] = {}
    # Random balanced assignment of leaves to a complete branching tree.
    order = rng.permutation(num_classes)
    nodes: Dict[str, Optional[str]] = {}
    next_id = [0]

    def new_node(parent: Optional[str]) -> str:
        node = "s%05d" % next_id[0]
        next_id[0] += 1
        nodes[node] = parent
        return node

    def build(leaves: List[int], parent: Optional[str]) -> Optional[str]:
        node = new_node(parent)
        if len(leaves) == 1:
            parents["n%08d" % leaves[0]] = node
            return node
        chunks = [leaves[i::branching] for i in range(branching)]
        for chunk in chunks:
            if chunk:
                build(chunk, node)
        return node

    build(list(order), None)
    parents.update(nodes)
    return parents


# ---------------------------------------------------------------------------
# The hierarchy object
# ---------------------------------------------------------------------------


@dataclass
class WordNetHierarchy:
    """Rooted tree over the ImageNet classes with LCA / information-content support."""

    parents: Dict[str, Optional[str]]
    class_to_leaf: Dict[int, str]
    class_names: Optional[List[str]] = None
    _children: Dict[str, List[str]] = field(default_factory=dict, repr=False)
    _depth: Dict[str, int] = field(default_factory=dict, repr=False)
    _leaf_count: Dict[str, int] = field(default_factory=dict, repr=False)
    _root: Optional[str] = field(default=None, repr=False)

    # -- construction -------------------------------------------------------
    def __post_init__(self) -> None:
        self._finalize()

    def _finalize(self) -> None:
        self._children = {}
        roots: List[str] = []
        for node, parent in self.parents.items():
            self._children.setdefault(node, [])
            if parent is None:
                roots.append(node)
            else:
                self._children.setdefault(parent, []).append(node)

        if not roots:
            raise ValueError("Hierarchy has no root node.")
        # If several roots exist (forest), attach them under a virtual root.
        if len(roots) > 1:
            virtual = "__root__"
            self._children.setdefault(virtual, [])
            for r in roots:
                self.parents[r] = virtual
                self._children[virtual].append(r)
            self.parents[virtual] = None
            roots = [virtual]
        self._root = roots[0]

        # depths (BFS)
        self._depth = {self._root: 0}
        stack = [self._root]
        while stack:
            node = stack.pop()
            for child in self._children.get(node, ()):
                self._depth[child] = self._depth[node] + 1
                stack.append(child)

        # leaf counts = number of *class* nodes in the subtree
        class_nodes = set(self.class_to_leaf.values())
        self._leaf_count = {}
        # iterative post-order
        stack = [(self._root, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                count = 1 if node in class_nodes else 0
                for child in self._children.get(node, ()):
                    count += self._leaf_count.get(child, 0)
                self._leaf_count[node] = count
            else:
                stack.append((node, True))
                for child in self._children.get(node, ()):
                    stack.append((child, False))

    # -- accessors ----------------------------------------------------------
    @property
    def root(self) -> str:
        return self._root

    @property
    def num_classes(self) -> int:
        return len(self.class_to_leaf)

    def depth(self, node: str) -> int:
        """Number of edges from the root (``P(x)`` in the paper)."""
        return self._depth[node]

    def leaf_count(self, node: str) -> int:
        """Number of class nodes in the subtree of ``node`` (``|L(node)|``)."""
        return self._leaf_count.get(node, 0)

    def children(self, node: str) -> List[str]:
        return self._children.get(node, [])

    def parent(self, node: str) -> Optional[str]:
        return self.parents.get(node)

    def ancestor_path(self, node: str) -> List[str]:
        """Path from ``node`` up to and including the root."""
        path = []
        cur: Optional[str] = node
        while cur is not None:
            path.append(cur)
            cur = self.parents.get(cur)
        return path

    def leaf_for_class(self, class_index: int) -> str:
        """Node of an ImageNet class index (leaf nodes / argmax y_i in the paper)."""
        if class_index not in self.class_to_leaf:
            raise KeyError("Unknown class index %r" % (class_index,))
        return self.class_to_leaf[class_index]

    def lca_node(self, node_a: str, node_b: str) -> str:
        """Lowest common ancestor node ``N_LCA``."""
        ancestors_b = set(self.ancestor_path(node_b))
        for node in self.ancestor_path(node_a):
            if node in ancestors_b:
                return node
        return self._root

    def lca_of_classes(self, class_a: int, class_b: int) -> str:
        return self.lca_node(self.leaf_for_class(class_a), self.leaf_for_class(class_b))

    def class_name(self, class_index: int) -> str:
        if self.class_names and 0 <= class_index < len(self.class_names):
            return self.class_names[class_index]
        return "class_%d" % class_index


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_wordnet_hierarchy(
    csv_path: Optional[str] = None,
    wnids: Optional[Sequence[str]] = None,
    class_names: Optional[List[str]] = None,
    allow_synthetic: bool = True,
) -> WordNetHierarchy:
    """Load the ImageNet WordNet hierarchy.

    Order of preference:

    1. ``csv_path`` -- parse ``imagenet_fiveai.csv`` (Sec. 2 / Addendum).  Edges
       are used directly when present, otherwise the synset ids are expanded with
       ``nltk``.
    2. ``wnids`` / HuggingFace ImageNet metadata expanded with ``nltk``.
    3. A deterministic synthetic tree (only for offline tests).
    """
    parents: Dict[str, Optional[str]] = {}
    class_to_leaf: Dict[int, str] = {}
    names = list(class_names) if class_names else None

    if csv_path:
        try:
            edges, csv_class_map, csv_synset_map = parse_hierarchy_csv(csv_path)
        except FileNotFoundError:
            edges, csv_class_map, csv_synset_map = [], {}, {}

        if csv_class_map:
            class_to_leaf = dict(csv_class_map)
        if edges:
            for child, parent in edges:
                parents[child] = parent
            for node in list(parents):
                parents.setdefault(parents[node], None) if parents[node] else None
            for node in list(parents):
                if parents[node] not in parents and parents[node] is not None:
                    parents[parents[node]] = None
            # ensure every class index maps to a node present in the tree
            if not class_to_leaf and csv_synset_map:
                class_to_leaf = {i: s for s, i in csv_synset_map.items()}
            for idx, node in class_to_leaf.items():
                parents.setdefault(node, None)
            return WordNetHierarchy(parents, class_to_leaf, names)

        if csv_class_map:
            # csv only provided the class <-> synset mapping: expand with nltk
            nodes = [csv_class_map[i] for i in sorted(csv_class_map)]
            parents = build_parents_from_wnids(nodes)
            if parents:
                class_to_leaf = dict(csv_class_map)
            else:
                class_to_leaf = {}

    if not parents:
        ids = list(wnids) if wnids else get_imagenet_wnids()
        if not ids:
            ids = []
        if ids:
            parents = build_parents_from_wnids(ids)
            if parents:
                class_to_leaf = {i: wn for i, wn in enumerate(ids)}

    if not parents:
        if not allow_synthetic:
            raise RuntimeError("No WordNet hierarchy available.")
        logger.warning(
            "Falling back to a synthetic hierarchy (no WordNet data available)."
        )
        parents = build_synthetic_parents(IMAGENET_NUM_CLASSES)
        class_to_leaf = {i: "n%08d" % i for i in range(IMAGENET_NUM_CLASSES)}

    return WordNetHierarchy(parents, class_to_leaf, names)


# ---------------------------------------------------------------------------
# Simulated small hierarchy (§C, Table 7)
# ---------------------------------------------------------------------------


def build_two_pair_hierarchy() -> WordNetHierarchy:
    """The 4-class hierarchy of Sec. C: root -> {(1,2), (3,4)}.

    Used by the simulated-data study and by the unit tests (small, exact).
    """
    parents = {
        "root": None,
        "group_a": "root",
        "group_b": "root",
        "c1": "group_a",
        "c2": "group_a",
        "c3": "group_b",
        "c4": "group_b",
    }
    class_to_leaf = {0: "c1", 1: "c2", 2: "c3", 3: "c4"}
    return WordNetHierarchy(parents, class_to_leaf)
