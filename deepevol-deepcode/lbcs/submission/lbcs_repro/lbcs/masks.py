"""Mask representation, initialization, and grouping for Refined Coreset Selection.

Paper references
----------------
* §2 "Objective formulations": the ``0-1`` masks are introduced with
  ``m in {0, 1}^n`` and ``m_i = 1`` indicating that the data point
  ``(x_i, y_i)`` is selected into the coreset (otherwise excluded).  The
  secondary objective is ``f_2(m) := ||m||_0``.
* Algorithm 1, line 2: "Initialize masks ``m`` randomly with ``||m||_0 = k``".
* §3.2 "Algorithm flow and tricks for acceleration": "the mask search space can
  be narrowed by treating several examples as a group.  The examples in the same
  group share the same mask in coreset selection."

Two levels of representation are supported:

1. **Example level** (canonical, exactly as in the paper): a mask is a vector of
   length ``n`` whose entries live in ``{0, 1}``.
2. **Group level** (acceleration trick of §3.2): a mask of length ``G = n /
   group_size`` where every example belonging to the same group reads the same
   entry.  ``Grouping.expand`` lifts a group-level mask back to the example
   level so that the objectives ``f1``/``f2`` remain defined over ``n`` examples.

Masks used during the outer search are *relaxed* (continuous) vectors whose
entries live in ``[-1, 1]``; the projection rules themselves live in
``lbcs/discretize.py`` (Appendix A), but initialization here is kept consistent
with them: selected entries start at ``+1`` and unselected entries at ``-1``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Union

import numpy as np

try:  # torch is optional for pure unit tests of the mask algebra
    import torch
except Exception:  # pragma: no cover - torch is a hard requirement in practice
    torch = None  # type: ignore

__all__ = [
    "Grouping",
    "init_binary_mask",
    "init_continuous_mask",
    "init_grouped_binary_mask",
    "continuous_to_binary",
    "l0_norm",
    "num_selected",
    "selected_indices",
    "expand_mask",
    "reduce_mask",
    "masked_weights",
    "validation_mask_init",
]

ArrayLike = Union[np.ndarray, "torch.Tensor", Sequence[float]]


# ---------------------------------------------------------------------------
# small conversion helpers
# ---------------------------------------------------------------------------
def _as_numpy(mask: ArrayLike) -> np.ndarray:
    """Return ``mask`` as a float numpy array (works with torch tensors)."""
    if torch is not None and isinstance(mask, torch.Tensor):
        return mask.detach().cpu().numpy()
    return np.asarray(mask, dtype=np.float64)


def _make_generator(seed: Optional[int] = None, generator=None) -> np.random.Generator:
    """Return a ``numpy`` random Generator from a seed or an existing generator."""
    if generator is not None:
        return generator
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# objective f_2(m) = ||m||_0
# ---------------------------------------------------------------------------
def l0_norm(mask: ArrayLike) -> float:
    """``||m||_0``: number of non-zero entries of ``mask`` (Eq. (2)).

    Following the paper, ``f_2(m) := ||m||_0`` counts the *selected* entries.  For
    a binary mask this is the number of selected examples; for a relaxed mask the
    number of non-zero entries is returned, which coincides with the number of
    selected examples once the mask has been discretized.
    """
    arr = _as_numpy(mask)
    return float(np.count_nonzero(arr))


def num_selected(mask: ArrayLike) -> int:
    """Number of examples selected by ``mask`` (integer version of :func:`l0_norm`)."""
    return int(np.count_nonzero(_as_numpy(mask)))


def selected_indices(mask: ArrayLike) -> np.ndarray:
    """Indices ``i`` with ``m_i != 0`` (i.e. the coreset indices)."""
    return np.nonzero(_as_numpy(mask))[0]


# ---------------------------------------------------------------------------
# initialization
# ---------------------------------------------------------------------------
def init_binary_mask(
    n: int,
    k: Optional[int] = None,
    seed: Optional[int] = None,
    generator: Optional[np.random.Generator] = None,
    dtype=np.float32,
    return_torch: bool = False,
) -> np.ndarray:
    """Initialize a binary mask with exactly ``||m||_0 = k`` (Algorithm 1, line 2).

    ``k`` indices are sampled without replacement out of ``n``; the mask entries
    of the sampled indices are set to ``1`` and all remaining entries to ``0``.
    When ``k`` is ``None`` a uniform random fraction of the dataset is used
    (half of the examples), which is the behaviour needed by the Figure 1
    experiments that rely on "an arbitrarily random subset".

    Parameters
    ----------
    n : int
        Number of examples in the dataset ``D``.
    k : int, optional
        Predefined size (the paper's ``k``).  Must satisfy ``0 <= k <= n``.
    seed / generator : optional
        Deterministic sampling controls.
    dtype : numpy dtype
        Output dtype (float32 by default so relaxed updates are lossless).
    return_torch : bool
        If True and torch is available, return a ``torch.Tensor`` instead.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    if k is None:
        k = max(1, n // 2)
    k = int(k)
    if k < 0 or k > n:
        raise ValueError(f"k={k} out of range [0, {n}]")

    rng = _make_generator(seed, generator)
    mask = np.zeros(int(n), dtype=dtype)
    if k > 0:
        idx = rng.choice(int(n), size=k, replace=False)
        mask[idx] = 1.0
    if return_torch and torch is not None:
        return torch.as_tensor(mask)  # type: ignore[return-value]
    return mask


def init_continuous_mask(
    n: int,
    k: Optional[int] = None,
    seed: Optional[int] = None,
    generator: Optional[np.random.Generator] = None,
    noise: float = 0.0,
    dtype=np.float32,
    return_torch: bool = False,
) -> np.ndarray:
    """Initialize a *relaxed* mask for the outer search.

    The relaxed domain used by LexiFlow (Appendix A) is ``[-1, 1]`` with the
    discretization "values in ``[-1, 0)`` are mapped to ``0`` and values in
    ``[0, 1]`` to ``1``".  Consequently a relaxed mask is initialized consistently
    from the binary initialization: selected entries are ``+1`` and unselected
    entries are ``-1``.  Optional ``noise`` perturbs the entries (still clamped to
    ``[-1, 1]``), which breaks ties between groups of identical examples.
    """
    mask = init_binary_mask(n, k=k, seed=seed, generator=generator, dtype=np.float64)
    cont = np.where(mask > 0.0, 1.0, -1.0)
    if noise and noise > 0.0:
        rng = _make_generator(seed, generator)
        cont = cont + rng.normal(0.0, float(noise), size=cont.shape)
        cont = np.clip(cont, -1.0, 1.0)
    cont = cont.astype(dtype, copy=False)
    if return_torch and torch is not None:
        return torch.as_tensor(cont)  # type: ignore[return-value]
    return cont


def continuous_to_binary(mask: ArrayLike, threshold: float = 0.0, dtype=np.float32) -> np.ndarray:
    """Threshold a relaxed mask at ``threshold`` (default ``0``) to ``{0, 1}``.

    This mirrors the Appendix A rule ``[-1, 0) -> 0`` and ``[0, 1] -> 1``.
    """
    arr = _as_numpy(mask)
    out = (arr >= float(threshold)).astype(dtype)
    if torch is not None and isinstance(mask, torch.Tensor):
        return torch.as_tensor(out, device=mask.device)  # type: ignore[return-value]
    return out


# ---------------------------------------------------------------------------
# grouping (§3.2 acceleration trick)
# ---------------------------------------------------------------------------
@dataclass
class Grouping:
    """Assign the ``n`` examples of ``D`` to ``G`` groups sharing one mask entry.

    §3.2: "the mask search space can be narrowed by treating several examples as a
    group.  The examples in the same group share the same mask in coreset
    selection."

    With ``group_size = 1`` the grouping degenerates to the canonical example-level
    representation ``m in {0, 1}^n``.  With ``group_size = g > 1`` the search
    operates on ``G = ceil(n / g)`` entries and :meth:`expand` lifts a group-level
    mask to the example level, so that ``f_1`` and ``f_2`` stay defined over the
    ``n`` examples of the dataset.

    Attributes
    ----------
    n : int
        Number of examples.
    group_size : int
        Number of examples per group (``g``).
    assignment : numpy.ndarray
        ``assignment[i]`` is the group index of example ``i`` (length ``n``).
    group_sizes : numpy.ndarray
        Number of examples in each group (length ``G``); the last group may be
        smaller when ``g`` does not divide ``n``.
    """

    n: int
    group_size: int = 1
    seed: Optional[int] = None
    shuffle: bool = True
    assignment: np.ndarray = field(init=False, repr=False)
    group_sizes: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.n = int(self.n)
        self.group_size = max(1, int(self.group_size))
        if self.group_size == 1:
            self.assignment = np.arange(self.n, dtype=np.int64)
        else:
            order = np.arange(self.n, dtype=np.int64)
            if self.shuffle:
                rng = np.random.default_rng(self.seed)
                order = rng.permutation(order)
            self.assignment = np.empty(self.n, dtype=np.int64)
            self.assignment[order] = np.arange(self.n, dtype=np.int64) // self.group_size
        self.num_groups = int(self.assignment.max()) + 1 if self.n > 0 else 0
        self.group_sizes = np.bincount(self.assignment, minlength=self.num_groups).astype(np.int64)

    # -- sizes -------------------------------------------------------------
    @property
    def G(self) -> int:
        """Number of mask entries (search-space dimension)."""
        return self.num_groups

    @property
    def compression(self) -> float:
        """Ratio between the search-space dimension and the number of examples."""
        return float(self.num_groups) / float(self.n) if self.n else 1.0

    # -- lifting / projection ---------------------------------------------
    def expand(self, group_mask: ArrayLike) -> np.ndarray:
        """Lift a group-level mask (length ``G``) to the example level (length ``n``)."""
        arr = _as_numpy(group_mask).reshape(-1)
        if arr.size != self.num_groups:
            raise ValueError(
                f"group mask has {arr.size} entries but the grouping has {self.num_groups} groups"
            )
        return arr[self.assignment]

    def reduce(self, example_mask: ArrayLike, mode: str = "mean") -> np.ndarray:
        """Aggregate an example-level mask (length ``n``) onto groups (length ``G``).

        ``mode='mean'`` gives the fraction of selected examples per group;
        ``mode='any'`` marks a group as selected if at least one example is;
        ``mode='all'`` requires every example of the group to be selected.
        """
        arr = _as_numpy(example_mask).reshape(-1)
        if arr.size != self.n:
            raise ValueError(f"example mask has {arr.size} entries but n={self.n}")
        if mode == "mean":
            return np.bincount(self.assignment, weights=arr, minlength=self.num_groups) / np.maximum(
                self.group_sizes, 1
            )
        if mode == "any":
            return np.bincount(
                self.assignment, weights=(arr > 0).astype(np.float64), minlength=self.num_groups
            ) > 0
        if mode == "all":
            sel = np.bincount(
                self.assignment, weights=(arr > 0).astype(np.float64), minlength=self.num_groups
            )
            return sel >= self.group_sizes
        raise ValueError(f"unknown reduce mode {mode!r}")

    def replicate_weights(self, group_mask: ArrayLike) -> np.ndarray:
        """Per-example indicator of a group-level mask (alias of :meth:`expand`)."""
        return self.expand(group_mask)

    def group_of(self, index: int) -> int:
        """Group index of example ``index``."""
        return int(self.assignment[int(index)])

    def groups(self) -> Iterable[np.ndarray]:
        """Iterate over the example indices of every group."""
        return (np.nonzero(self.assignment == g)[0] for g in range(self.num_groups))

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "group_size": self.group_size,
            "num_groups": self.num_groups,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "compression": self.compression,
        }


def expand_mask(group_mask: ArrayLike, grouping: Optional[Grouping]) -> np.ndarray:
    """Expand to the example level; identity when ``grouping`` is ``None``."""
    if grouping is None:
        return _as_numpy(group_mask).reshape(-1).astype(np.float64)
    return grouping.expand(group_mask)


def reduce_mask(example_mask: ArrayLike, grouping: Optional[Grouping], mode: str = "mean") -> np.ndarray:
    """Reduce to the group level; identity when ``grouping`` is ``None``."""
    if grouping is None:
        return _as_numpy(example_mask).reshape(-1).astype(np.float64)
    return grouping.reduce(example_mask, mode=mode)


def init_grouped_binary_mask(
    grouping: Grouping,
    k: Optional[int] = None,
    seed: Optional[int] = None,
    generator: Optional[np.random.Generator] = None,
    dtype=np.float32,
    return_torch: bool = False,
) -> np.ndarray:
    """Random grouped initialization of the coreset mask.

    Groups are drawn uniformly at random (without replacement) and *wholly*
    selected until adding the next group would exceed the predefined size ``k``.
    If the running total can not reach ``k`` exactly, the last drawn group is
    partially selected so that the achieved count is as close as possible to
    ``k`` from below -- the paper guarantees ``||m||_0 = k`` for the example-level
    initialization, while grouping necessarily quantizes the achievable sizes.
    Returns an example-level binary mask of length ``n``.
    """
    rng = _make_generator(seed, generator)
    n = grouping.n
    if k is None:
        k = max(1, n // 2)
    k = int(min(max(k, 0), n))

    perm = rng.permutation(grouping.num_groups)
    chosen = np.zeros(n, dtype=bool)
    total = 0
    for g in perm:
        idx = np.nonzero(grouping.assignment == g)[0]
        if total + idx.size <= k:
            chosen[idx] = True
            total += idx.size
        else:
            need = k - total
            if need > 0:
                pick = rng.choice(idx, size=need, replace=False)
                chosen[pick] = True
                total += need
            break
        if total == k:
            break

    mask = chosen.astype(dtype)
    if return_torch and torch is not None:
        return torch.as_tensor(mask)  # type: ignore[return-value]
    return mask


# ---------------------------------------------------------------------------
# convenience helpers used by the inner loop / objectives
# ---------------------------------------------------------------------------
def masked_weights(mask: ArrayLike, normalize: bool = True) -> np.ndarray:
    """Return the per-example weights used by ``L(m, theta)``.

    ``L(m, theta) = (1 / ||m||_0) * sum_i m_i * l(h(x_i; theta), y_i)`` (Eq. (1)),
    so the weights of the selected examples are ``1 / ||m||_0`` and the weights of
    the others are ``0``.  With ``normalize=False`` the raw indicator is returned.
    """
    arr = _as_numpy(mask).reshape(-1)
    if not normalize:
        return arr
    total = np.count_nonzero(arr)
    if total == 0:
        return np.zeros_like(arr)
    return arr / float(total)


def validation_mask_init(n: int, k: Optional[int] = None, seed: int = 0) -> np.ndarray:
    """Small self-check used by the validation harness.

    Returns a ``(mask, ok)`` style tuple?  No -- kept as a plain mask so that the
    helper is side-effect free; the assertions live in ``tests``/experiment code.
    """
    return init_binary_mask(n, k=k, seed=seed)
