"""Discretisation utilities for FRE.

The FRE encoder consumes a set of ``K`` ``(state, reward)`` pairs.  Each scalar
reward is turned into a *discrete* bin index which indexes into a learned
embedding table (``fre/fre/reward_embeddings.py``).  The paper (Sec 4.1 and the
corrected addendum) specifies the exact recipe::

    reward in [-1, 1]
        -> rescale to [0, 1]      ( + 1 ) / 2
        -> multiply by 32
        -> floor
        -> clamp to {0, ..., 31}  (32 bins)

This module centralises that recipe -- plus the related helpers used by the
environment wrappers/eval rewards (XY -> 32x32 grid, bin centres, bin distance,
ordinal/normalised reward encodings) -- so every component of the code base
performs *identical* arithmetic.

Everything here is intentionally dependency-light: only ``numpy`` is required and
``torch`` support is optional/deferred (mirroring ``fre/utils/normalization.py``
and ``fre/utils/logging.py``).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

try:  # pragma: no cover - torch is optional for pure-numpy usage
    import torch

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False


__all__ = [
    # constants
    "NUM_REWARD_BINS",
    "REWARD_RESCALE",
    "REWARD_MIN",
    "REWARD_MAX",
    "STATE_EMB_DIM",
    "REWARD_EMB_DIM",
    "TOKEN_DIM",
    "DEFAULT_XY_BINS",
    "DEFAULT_XY_EXTENT",
    # scalar discretisation
    "rescale_reward",
    "discretize_reward",
    "undiscretize_reward",
    "one_hot_reward",
    "normalized_reward",
    "bin_index_to_center",
    "bins_to_centers",
    "discretize_state",
    "discretize_value",
    # XY / grid discretisation
    "discretize_xy",
    "bin_to_xy",
    "xy_distance",
    "bin_distance",
    "xy_grid_indices",
    "grid_size",
    # misc helpers
    "one_hot",
    "to_numpy",
    "to_torch_like",
    "is_torch_tensor",
]


# ---------------------------------------------------------------------------
# Constants (kept in sync with fre/fre/reward_embeddings.py and the paper)
# ---------------------------------------------------------------------------
NUM_REWARD_BINS: int = 32
"""Number of discrete reward bins (paper: rescale -> *32 -> floor)."""

REWARD_RESCALE: float = float(NUM_REWARD_BINS)
"""Multiplicative factor applied after rescaling to ``[0, 1]``."""

REWARD_MIN: float = -1.0
REWARD_MAX: float = 1.0

STATE_EMB_DIM: int = 64
REWARD_EMB_DIM: int = 64
TOKEN_DIM: int = STATE_EMB_DIM + REWARD_EMB_DIM  # 128

DEFAULT_XY_BINS: int = 32
DEFAULT_XY_EXTENT: float = 36.0


# ---------------------------------------------------------------------------
# numpy / torch interop helpers
# ---------------------------------------------------------------------------
def is_torch_tensor(x: Any) -> bool:
    """Return ``True`` if ``x`` is a ``torch.Tensor`` (and torch is available)."""
    return bool(_TORCH_AVAILABLE and isinstance(x, torch.Tensor))


def to_numpy(x: Any, dtype: Optional[Any] = None) -> np.ndarray:
    """Convert tensor/array/scalar/sequence to a ``numpy`` array.

    Args:
        x: input value.
        dtype: optional numpy dtype for the result (``float32`` for floats by
            default when the input is a float tensor/array).
    """
    if is_torch_tensor(x):
        arr = x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        arr = x
    else:
        arr = np.asarray(x)

    if dtype is None:
        if np.issubdtype(arr.dtype, np.floating):
            dtype = np.float32
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def to_torch_like(original: Any, array: Any):
    """Convert ``array`` back to a torch tensor if ``original`` was one."""
    if is_torch_tensor(original):
        tensor = original.detach()
        out = tensor.new_tensor(np.asarray(array))
        return out
    return array


# ---------------------------------------------------------------------------
# Scalar reward discretisation
# ---------------------------------------------------------------------------
def rescale_reward(reward: Any) -> np.ndarray:
    """Rescale rewards from ``[-1, 1]`` to ``[0, 1]``.

    ``r -> (r - (-1)) / (1 - (-1)) = (r + 1) / 2``.
    """
    arr = to_numpy(reward, dtype=np.float32)
    return (arr - REWARD_MIN) / (REWARD_MAX - REWARD_MIN)


def normalized_reward(reward: Any) -> np.ndarray:
    """Clip rewards to ``[REWARD_MIN, REWARD_MAX]`` then rescale to ``[0, 1]``."""
    arr = to_numpy(reward, dtype=np.float32)
    return rescale_reward(np.clip(arr, REWARD_MIN, REWARD_MAX))


def discretize_reward(reward: Any) -> np.ndarray:
    """Discretise scalar rewards into ``32`` bins.

    Recipe (paper Sec 4.1 / corrected addendum)::

        r in [-1, 1] -> (r + 1) / 2 -> * 32 -> floor -> clamp to [0, 31]

    The input is first clipped to ``[-1, 1]`` so out-of-range rewards (or the
    floating point edge case ``r == 1.0``) still map to a valid bin index.
    Returns an integer array with the same shape as the input.  If the input was
    a ``torch.Tensor`` the returned array is passed back through
    ``to_torch_like`` by callers that need tensors; the primary API returns a
    numpy array for portability.
    """
    arr = np.clip(to_numpy(reward, dtype=np.float32), REWARD_MIN, REWARD_MAX)
    scaled = (arr - REWARD_MIN) / (REWARD_MAX - REWARD_MIN) * REWARD_RESCALE
    bins = np.floor(scaled)
    bins = np.clip(bins, 0, NUM_REWARD_BINS - 1)
    return bins.astype(np.int64)


def undiscretize_reward(bins: Any, center: bool = True) -> np.ndarray:
    """Map bin indices back to rewards in ``[-1, 1]``.

    With ``center=True`` the bin *centre* is returned, i.e. the inverse of
    ``(bin + 0.5) / 32 * 2 - 1``.  With ``center=False`` the lower edge of the
    bin is used (``bin / 32 * 2 - 1``), matching the encoding ``floor`` step.
    """
    arr = to_numpy(bins, dtype=np.float32)
    offset = 0.5 if center else 0.0
    norm = (arr + offset) / float(NUM_REWARD_BINS)
    return norm * (REWARD_MAX - REWARD_MIN) + REWARD_MIN


def one_hot_reward(reward: Any, num_bins: int = NUM_REWARD_BINS) -> np.ndarray:
    """One-hot encode scalar rewards using the same discretisation recipe."""
    bins = discretize_reward(reward).reshape(-1)
    if num_bins != NUM_REWARD_BINS:
        bins = np.clip(bins, 0, num_bins - 1)
    out = np.zeros((bins.shape[0], num_bins), dtype=np.float32)
    out[np.arange(bins.shape[0]), bins] = 1.0
    return out


def bin_index_to_center(bin_index: Any, low: float = REWARD_MIN, high: float = REWARD_MAX,
                        num_bins: int = NUM_REWARD_BINS) -> np.ndarray:
    """Return the value at the centre of a uniformly spaced bin."""
    arr = to_numpy(bin_index, dtype=np.float32)
    width = (high - low) / float(num_bins)
    return low + (arr + 0.5) * width


def bins_to_centers(edges: Sequence[float]) -> np.ndarray:
    """Given bin *edges*, return their centres."""
    edges_arr = to_numpy(edges, dtype=np.float32)
    return 0.5 * (edges_arr[:-1] + edges_arr[1:])


# ---------------------------------------------------------------------------
# Generic value / state discretisation into uniform bins
# ---------------------------------------------------------------------------
def discretize_value(values: Any, low: float, high: float, num_bins: int,
                     clip: bool = True) -> np.ndarray:
    """Discretise arbitrary values into ``num_bins`` uniform bins over ``[low, high]``.

    ``idx = floor((v - low) / (high - low) * num_bins)`` clamped to
    ``[0, num_bins - 1]``.
    """
    arr = to_numpy(values, dtype=np.float32)
    if clip:
        arr = np.clip(arr, low, high)
    if high <= low:
        raise ValueError(f"high ({high}) must be greater than low ({low})")
    scaled = (arr - low) / (high - low) * float(num_bins)
    bins = np.floor(scaled)
    return np.clip(bins, 0, num_bins - 1).astype(np.int64)


def discretize_state(states: Any, low: Any, high: Any, num_bins: Any,
                     clip: bool = True) -> np.ndarray:
    """Per-dimension discretisation of states into uniform bins.

    ``low``, ``high`` and ``num_bins`` may be scalars or broadcastable arrays of
    length ``state_dim`` (the last axis of ``states``).
    """
    arr = to_numpy(states, dtype=np.float32)
    low_arr = np.asarray(low, dtype=np.float32)
    high_arr = np.asarray(high, dtype=np.float32)
    bins_arr = np.asarray(num_bins, dtype=np.float32)

    if clip:
        arr = np.clip(arr, low_arr, high_arr)
    scaled = (arr - low_arr) / np.maximum(high_arr - low_arr, 1e-12) * bins_arr
    idx = np.floor(scaled)
    upper = np.maximum(bins_arr - 1.0, 0.0)
    return np.clip(idx, 0, upper).astype(np.int64)


# ---------------------------------------------------------------------------
# XY grid discretisation (AntMaze: full maze -> 32 x 32 grid)
# ---------------------------------------------------------------------------
def discretize_xy(xy: Any, extent: float = DEFAULT_XY_EXTENT,
                  num_bins: int = DEFAULT_XY_BINS) -> np.ndarray:
    """Discretise continuous XY positions into ``num_bins`` bins per axis.

    ``bin = floor(xy / extent * num_bins)`` clamped to ``[0, num_bins - 1]``,
    matching ``fre/envs/antmaze_wrapper.py`` and
    ``fre/rewards/eval_rewards.py``.
    """
    arr = to_numpy(xy, dtype=np.float32)
    scaled = arr / float(extent) * float(num_bins)
    idx = np.floor(scaled)
    return np.clip(idx, 0, num_bins - 1).astype(np.int64)


def bin_to_xy(bins: Any, extent: float = DEFAULT_XY_EXTENT,
              num_bins: int = DEFAULT_XY_BINS) -> np.ndarray:
    """Inverse of ``discretize_xy``: return the centre of each XY bin."""
    arr = to_numpy(bins, dtype=np.float32)
    cell = float(extent) / float(num_bins)
    return (arr + 0.5) * cell


def xy_distance(a: Any, b: Any) -> np.ndarray:
    """Euclidean distance between XY positions (continuous)."""
    a_arr = to_numpy(a, dtype=np.float64)
    b_arr = to_numpy(b, dtype=np.float64)
    return np.linalg.norm(a_arr - b_arr, axis=-1)


def bin_distance(a: Any, b: Any, extent: float = DEFAULT_XY_EXTENT,
                 num_bins: int = DEFAULT_XY_BINS) -> np.ndarray:
    """Euclidean distance in *bin* units between two XY bin coordinates.

    The paper's AntMaze goal criterion is "within 2 bins" of the goal, so this
    helper is used to implement the ``dist <= 2`` success check.
    """
    a_arr = to_numpy(a, dtype=np.float64)
    b_arr = to_numpy(b, dtype=np.float64)
    return np.linalg.norm(a_arr - b_arr, axis=-1)


def xy_grid_indices(xy: Any, extent: float = DEFAULT_XY_EXTENT,
                    num_bins: int = DEFAULT_XY_BINS) -> np.ndarray:  # pragma: no cover - placeholder removed below
    raise NotImplementedError


def grid_size(num_bins: int = DEFAULT_XY_BINS) -> int:
    """Number of cells in a square ``num_bins x num_bins`` grid."""
    return int(num_bins) * int(num_bins)


# ---------------------------------------------------------------------------
# Small generic one-hot helper
# ---------------------------------------------------------------------------
def one_hot(indices: Any, num_classes: int, dtype: Any = np.float32) -> np.ndarray:
    """One-hot encode integer indices along a new trailing axis."""
    idx = to_numpy(indices)
    idx = idx.astype(np.int64)
    out = np.zeros(idx.shape + (int(num_classes),), dtype=dtype)
    valid = (idx >= 0) & (idx < int(num_classes))
    np.put_along_axis(
        out.reshape(-1, int(num_classes)),
        idx.reshape(-1, 1).clip(0, int(num_classes) - 1),
        1.0,
        axis=1,
    )
    out[(~valid)] = 0.0
    return out


def reward_discretization_info() -> Dict[str, Any]:
    """Return a small descriptive dict of the reward discretisation scheme."""
    return {
        "num_bins": NUM_REWARD_BINS,
        "rescale": REWARD_RESCALE,
        "reward_min": REWARD_MIN,
        "reward_max": REWARD_MAX,
        "state_emb_dim": STATE_EMB_DIM,
        "reward_emb_dim": REWARD_EMB_DIM,
        "token_dim": TOKEN_DIM,
        "recipe": "clip(r,[-1,1]) -> (r+1)/2 -> *32 -> floor -> clamp[0,31]",
    }


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def _self_test() -> None:  # pragma: no cover - exercised manually
    # reward bins
    assert discretize_reward(-1.0) == 0
    assert discretize_reward(0.0) == 16
    assert discretize_reward(1.0) == NUM_REWARD_BINS - 1
    assert int(discretize_reward(2.0)) == NUM_REWARD_BINS - 1  # clipped

    r = np.linspace(-1.0, 1.0, 257, dtype=np.float32)
    b = discretize_reward(r)
    assert b.min() >= 0 and b.max() <= NUM_REWARD_BINS - 1
    assert np.all(np.diff(b) >= 0)

    # centre round-trip
    centres = undiscretize_reward(b, center=True)
    assert centres.min() >= REWARD_MIN and centres.max() <= REWARD_MAX

    # one hot
    oh = one_hot_reward(np.array([-1.0, 0.0, 1.0], dtype=np.float32))
    assert oh.shape == (3, NUM_REWARD_BINS)
    assert np.allclose(oh.sum(axis=1), 1.0)

    # XY bins
    xy = np.array([[0.0, 0.0], [35.9, 35.9], [18.0, 18.0]], dtype=np.float32)
    bins = discretize_xy(xy)
    assert bins[0].tolist() == [0, 0]
    assert bins[1].tolist() == [DEFAULT_XY_BINS - 1, DEFAULT_XY_BINS - 1]
    back = bin_to_xy(bins)
    assert back.shape == xy.shape
    assert bin_distance(np.array([0, 0]), np.array([2, 0])) == 2.0

    # generic
    vals = discretize_value(np.array([-2.0, 0.0, 2.0]), -1.0, 1.0, 4)
    assert vals.tolist() == [0, 2, 3]

    # torch round trip
    if _TORCH_AVAILABLE:
        t = torch.tensor([-1.0, 0.0, 1.0])
        out = discretize_reward(t)
        assert isinstance(out, np.ndarray)

    print("discretize.py self-test passed")
    print(reward_discretization_info())


if __name__ == "__main__":  # pragma: no cover
    _self_test()
