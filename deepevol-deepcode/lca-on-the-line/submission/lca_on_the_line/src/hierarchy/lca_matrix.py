"""LCA distance matrix construction and processing.

Implements the ``n x n`` pairwise LCA distance matrix described in the paper
(Section 2 / Section D.2.1) together with the matrix processing pipeline of
Section E.2 used by the taxonomy-alignment soft loss (Algorithm 1).

Paper references
----------------
* Section 2 / D.2.1: ``M[i, k] = D_LCA(i, k)`` with

  ``D_LCA^I(y', y) = I(y) - I(N_LCA(y, y'))``   (information content, used for
  the main LCA measurements)

  ``D_LCA^P(y', y) = (P(y) - P(N_LCA)) + (P(y') - P(N_LCA))``  (tree depth,
  used for the linear-probing experiments).

* Section E.2: given a dataset with ``n`` classes, we first establish an
  ``n x n`` LCA distance matrix ``M`` where ``M[i, k]`` indicates the pairwise
  LCA distance ``D_LCA(i, k)``, computed using either the WordNet hierarchy or a
  latent hierarchy derived from K-means clustering. Next, we scale ``M`` by
  applying a temperature term ``T`` and finally apply MinMax scaling to
  normalize values between 0 and 1::

        M_LCA = MinMax(M ** T)

* Algorithm 1 / Addendum: for latent (K-means) hierarchies the raw pairwise
  matrix stores *distances*, and the latent hierarchy is inverted via
  ``max(M) - M`` before being raised to the temperature power, since latent
  cluster co-membership is a similarity. The reverse matrix used as the
  alignment indicator is ``reverse_LCA_matrix = 1 - M_LCA``.

All functions accept either nested Python lists or numpy arrays and return the
same "nested list" representation unless explicitly asked for a tensor, so that
this module stays usable without PyTorch installed (torch is imported lazily).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple, Union

from .lca import (
    DEFAULT_DISTANCE_MODE,
    LcaDistance,
    pairwise_lca_matrix,
    reverse_lca_matrix as _reverse_lca_matrix,
)
from .wordnet import WordNetHierarchy

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_LATENT_BASE_LEVEL",
    "min_max_scale",
    "power_scale",
    "invert_matrix",
    "process_lca_matrix",
    "reverse_lca_matrix",
    "lca_matrix_from_hierarchy",
    "build_lca_matrix",
    "to_torch_tensor",
    "from_torch_tensor",
    "matrix_diagonal_is_zero",
    "matrix_is_symmetric",
    "assert_lca_matrix_properties",
    "LcaMatrixProcessor",
]

#: First cluster level (1-indexed) used by the latent K-means hierarchies; the
#: level *below* this one is the 1000 ImageNet class leaves (see Section E.1).
DEFAULT_LATENT_BASE_LEVEL: int = 10

Matrix = Union[Sequence[Sequence[float]], "object"]  # nested list or ndarray


# --------------------------------------------------------------------------- #
# Low level scaling helpers
# --------------------------------------------------------------------------- #
def _as_nested(matrix: Matrix) -> List[List[float]]:
    """Convert numpy arrays / nested sequences to nested Python lists of floats."""
    if hasattr(matrix, "tolist"):  # numpy array (or torch tensor)
        matrix = matrix.tolist()
    return [[float(value) for value in row] for row in matrix]


def _matrix_min_max(matrix: List[List[float]]) -> Tuple[float, float]:
    values = [v for row in matrix for v in row]
    if not values:
        raise ValueError("Cannot scale an empty LCA matrix.")
    return min(values), max(values)


def min_max_scale(matrix: Matrix, feature_range: Tuple[float, float] = (0.0, 1.0)) -> List[List[float]]:
    """MinMax scale a matrix into ``feature_range`` (default ``[0, 1]``).

    Mirrors scikit-learn's ``MinMaxScaler`` applied to the whole matrix.  A
    degenerate (constant) matrix maps to the low end of the range.
    """
    nested = _as_nested(matrix)
    lo, hi = _matrix_min_max(nested)
    target_lo, target_hi = feature_range
    if hi - lo <= 0.0:
        return [[target_lo for _ in row] for row in nested]
    scale = (target_hi - target_lo) / (hi - lo)
    return [[target_lo + (value - lo) * scale for value in row] for row in nested]


def power_scale(matrix: Matrix, temperature: float = 1.0) -> List[List[float]]:
    """Raise every entry of the matrix to the temperature power (``M ** T``).

    Section E.2 applies ``M ** T`` before MinMax scaling.  A large temperature
    (``T = 25`` in the paper) sharpens the alignment indicator so that
    semantically closer classes receive a larger likelihood.
    """
    nested = _as_nested(matrix)
    if temperature == 1.0:
        return nested
    if temperature < 0:
        raise ValueError(f"temperature must be non-negative, got {temperature}.")
    # Negative entries cannot survive a fractional power; clamp for safety.
    return [[max(value, 0.0) ** temperature for value in row] for row in nested]


def invert_matrix(matrix: Matrix) -> List[List[float]]:
    """Return ``max(M) - M`` (distance -> similarity inversion).

    Used for latent (K-means) hierarchies, where the raw pairwise quantity is a
    distance but the taxonomy signal is a similarity: two classes that already
    share a cluster at a low level receive a *high* alignment score.
    """
    nested = _as_nested(matrix)
    _, hi = _matrix_min_max(nested)
    return [[hi - value for value in row] for row in nested]


# --------------------------------------------------------------------------- #
# Main processing pipeline
# --------------------------------------------------------------------------- #
def process_lca_matrix(
    lca_matrix: Matrix,
    temperature: float = 1.0,
    latent_hierarchy: bool = False,
    invert: Optional[bool] = None,
    scale: bool = True,
    feature_range: Tuple[float, float] = (0.0, 1.0),
    as_tensor: bool = False,
    device: Optional[str] = None,
    dtype: Optional[object] = None,
) -> Union[List[List[float]], "object"]:
    """Process a pairwise LCA distance matrix following Section E.2.

    Pipeline (``M_LCA = MinMax(M ** T)``)::

        M  ->  (invert: max(M) - M, latent hierarchies only)
           ->  M ** temperature
           ->  MinMax scale to [0, 1]

    Parameters
    ----------
    lca_matrix:
        The raw ``n x n`` pairwise LCA *distance* matrix ``M``.
    temperature:
        Temperature term ``T`` from Section E.2 (``T = 25`` for the linear
        probing experiments).
    latent_hierarchy:
        Whether the matrix comes from a K-means latent hierarchy.  If ``True``
        the matrix is inverted (``max(M) - M``) before the temperature power, as
        the latent hierarchy encodes cluster *similarity*.
    invert:
        Explicit override of the inversion decision.  ``None`` (default) means
        "invert iff ``latent_hierarchy``".
    scale:
        Whether to apply MinMax scaling at the end.
    feature_range:
        Range for the MinMax scaling (default ``[0, 1]``).
    as_tensor:
        Return a ``torch.Tensor`` instead of a nested list.
    device, dtype:
        Optional torch device / dtype used when ``as_tensor`` is ``True``.

    Returns
    -------
    The processed matrix (nested list by default, tensor when requested).
    """
    nested = _as_nested(lca_matrix)
    if invert is None:
        invert = latent_hierarchy

    if invert:
        nested = invert_matrix(nested)
    nested = power_scale(nested, temperature)
    if scale:
        nested = min_max_scale(nested, feature_range=feature_range)

    if as_tensor:
        return to_torch_tensor(nested, dtype=dtype, device=device)
    return nested


def reverse_lca_matrix(lca_matrix: Matrix) -> List[List[float]]:
    """Return ``reverse_LCA_matrix = 1 - M_LCA`` (Algorithm 1).

    The row for the ground-truth class contains its single largest value 1.0,
    which acts as the alignment indicator for the soft LCA loss.
    """
    nested = _as_nested(lca_matrix)
    return [[1.0 - value for value in row] for row in nested]


# --------------------------------------------------------------------------- #
# Building the matrix from a hierarchy
# --------------------------------------------------------------------------- #
def lca_matrix_from_hierarchy(
    hierarchy: WordNetHierarchy,
    mode: str = DEFAULT_DISTANCE_MODE,
    class_indices: Optional[Sequence[int]] = None,
    temperature: Optional[float] = None,
    latent_hierarchy: bool = False,
    as_tensor: bool = False,
    device: Optional[str] = None,
) -> Union[List[List[float]], "object"]:
    """Build (and optionally process) the ``n x n`` LCA matrix ``M``.

    ``M[i, k] = D_LCA(i, k)`` computed on ``hierarchy`` using information content
    (``mode="information"``, the default used for LCA measurements) or tree depth
    (``mode="depth"``, used for linear probing).
    """
    raw = pairwise_lca_matrix(hierarchy, class_indices=class_indices, mode=mode)
    if temperature is None:
        if as_tensor:
            return to_torch_tensor(raw, device=device)
        return raw
    return process_lca_matrix(
        raw,
        temperature=temperature,
        latent_hierarchy=latent_hierarchy,
        as_tensor=as_tensor,
        device=device,
    )


def build_lca_matrix(
    hierarchy: WordNetHierarchy,
    mode: str = DEFAULT_DISTANCE_MODE,
    class_indices: Optional[Sequence[int]] = None,
    temperature: Optional[float] = None,
    latent_hierarchy: bool = False,
    as_tensor: bool = False,
    device: Optional[str] = None,
) -> Union[List[List[float]], "object"]:
    """Alias of :func:`lca_matrix_from_hierarchy` kept for readability."""
    return lca_matrix_from_hierarchy(
        hierarchy,
        mode=mode,
        class_indices=class_indices,
        temperature=temperature,
        latent_hierarchy=latent_hierarchy,
        as_tensor=as_tensor,
        device=device,
    )


# --------------------------------------------------------------------------- #
# torch helpers (lazy import so the module works without torch)
# --------------------------------------------------------------------------- #
def to_torch_tensor(matrix: Matrix, dtype: Optional[object] = None, device: Optional[str] = None):
    """Convert a nested list / ndarray matrix to a ``torch.Tensor``."""
    import torch  # lazy

    tensor = torch.as_tensor(_as_nested(matrix))
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    else:
        tensor = tensor.float()
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def from_torch_tensor(tensor) -> List[List[float]]:
    """Convert a ``torch.Tensor`` back to a nested list of floats."""
    return _as_nested(tensor.detach().cpu().tolist())


# --------------------------------------------------------------------------- #
# Sanity checks
# --------------------------------------------------------------------------- #
def matrix_diagonal_is_zero(matrix: Matrix, atol: float = 1e-5) -> bool:
    """True when the diagonal of the (raw) LCA distance matrix is all zeros."""
    nested = _as_nested(matrix)
    n = len(nested)
    return all(
        abs(nested[i][j]) <= atol
        for i in range(n)
        for j in range(n)
        if i == j
    )


def matrix_is_symmetric(matrix: Matrix, atol: float = 1e-5) -> bool:
    """True when ``M[i, k] == M[k, i]`` within ``atol``."""
    nested = _as_nested(matrix)
    n = len(nested)
    return all(
        abs(nested[i][j] - nested[j][i]) <= atol
        for i in range(n)
        for j in range(i + 1, n)
    )


def assert_lca_matrix_properties(
    matrix: Matrix,
    atol: float = 1e-5,
    check_symmetry: bool = True,
    check_reverse: bool = True,
) -> None:
    """Validate the distance-matrix invariants used throughout the paper.

    * zero diagonal (``D_LCA(y, y) = 0``),
    * symmetry (``D_LCA(y, y') = D_LCA(y', y)`` for the information/depth
      scores defined in Section D.2.1),
    * ``reverse_LCA_matrix = 1 - M`` has a diagonal of ones.
    """
    nested = _as_nested(matrix)
    n = len(nested)
    if any(len(row) != n for row in nested):
        raise AssertionError("LCA matrix must be square.")

    for i in range(n):
        if abs(nested[i][i]) > atol:
            raise AssertionError(
                f"LCA distance of class {i} to itself should be 0, got {nested[i][i]}."
            )
    if check_symmetry and not matrix_is_symmetric(nested, atol=atol):
        max_asym = max(
            abs(nested[i][j] - nested[j][i]) for i in range(n) for j in range(n)
        )
        raise AssertionError(f"LCA matrix is not symmetric (max asymmetry {max_asym}).")
    if check_reverse:
        reverse = reverse_lca_matrix(nested)
        for i in range(n):
            if abs(reverse[i][i] - 1.0) > atol:
                raise AssertionError(
                    f"reverse_LCA_matrix diagonal at {i} should be 1, got {reverse[i][i]}."
                )


# --------------------------------------------------------------------------- #
# Convenience processor object
# --------------------------------------------------------------------------- #
class LcaMatrixProcessor:
    """Stateful helper bundling matrix construction and §E.2 processing.

    Parameters
    ----------
    hierarchy:
        The class hierarchy used for WordNet-based distances.
    mode:
        ``"information"`` (default, LCA measurements) or ``"depth"``
        (linear probing).
    temperature:
        Temperature term ``T`` from Section E.2.
    latent_hierarchy:
        Whether the matrix should be inverted before the temperature power
        (K-means latent hierarchies).
    """

    def __init__(
        self,
        hierarchy: WordNetHierarchy,
        mode: str = DEFAULT_DISTANCE_MODE,
        temperature: float = 1.0,
        latent_hierarchy: bool = False,
    ) -> None:
        self.hierarchy = hierarchy
        self.mode = mode
        self.temperature = temperature
        self.latent_hierarchy = latent_hierarchy
        self._distance = LcaDistance(hierarchy, mode=mode)

    # -- raw matrix -------------------------------------------------------- #
    def raw_matrix(self, class_indices: Optional[Sequence[int]] = None) -> List[List[float]]:
        """Unprocessed pairwise LCA distance matrix ``M``."""
        return self._distance.matrix(class_indices=class_indices)

    # -- processed matrix -------------------------------------------------- #
    def processed_matrix(
        self,
        class_indices: Optional[Sequence[int]] = None,
        as_tensor: bool = False,
        device: Optional[str] = None,
    ) -> Union[List[List[float]], "object"]:
        """``M_LCA = MinMax(M ** T)`` (inverted first for latent hierarchies)."""
        return self.process(self.raw_matrix(class_indices), as_tensor=as_tensor, device=device)

    def process(
        self,
        lca_matrix: Matrix,
        temperature: Optional[float] = None,
        as_tensor: bool = False,
        device: Optional[str] = None,
    ) -> Union[List[List[float]], "object"]:
        """Apply the Section E.2 processing pipeline to a given matrix."""
        return process_lca_matrix(
            lca_matrix,
            temperature=self.temperature if temperature is None else temperature,
            latent_hierarchy=self.latent_hierarchy,
            as_tensor=as_tensor,
            device=device,
        )

    # -- reverse matrix ---------------------------------------------------- #
    def reverse_matrix(
        self,
        class_indices: Optional[Sequence[int]] = None,
        as_tensor: bool = False,
        device: Optional[str] = None,
    ) -> Union[List[List[float]], "object"]:
        """``1 - MinMax(M ** T)`` used as the alignment indicator (Algorithm 1)."""
        processed = self.processed_matrix(class_indices)
        reverse = reverse_lca_matrix(processed)
        if as_tensor:
            return to_torch_tensor(reverse, device=device)
        return reverse
