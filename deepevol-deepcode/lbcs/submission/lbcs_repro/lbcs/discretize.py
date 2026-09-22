"""Mask discretization and projection for Lexicographic Bilevel Coreset Selection.

The paper (Appendix A, "Technical details") states verbatim:

    "Note that in experiments, when updating as did in Algorithm 2, the value of
     m less than -1 becomes -1 and the value greater than 1 becomes 1. Then during
     discretization, m in [-1, 0) will be projected to 0, and m in [0, 1] will be
     projected to 1."

So there are two distinct operations:

1. ``clamp_mask``  -- applied while updating the relaxed mask inside the LexiFlow
   outer loop (Algorithm 2): values ``< -1`` become ``-1``, values ``> 1`` become
   ``1``.  This keeps the sampled points in the box ``[-1, 1]^n``.

2. ``discretize_mask`` / ``project_to_binary`` -- applied when the final coreset
   must be built: ``m_i in [-1, 0)  ->  0`` and ``m_i in [0, 1]  ->  1``.
   (Equivalently: threshold the relaxed mask at 0, but we implement the interval
   rule directly so the half-open convention is explicit and testable.)

All functions accept either ``numpy`` arrays or ``torch`` tensors.  ``torch`` is a
soft dependency: it is imported lazily so that pure-numpy unit tests of the mask
algebra run without PyTorch installed.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import numpy as np

try:  # soft dependency
    import torch

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch is optional for mask-only math
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False


__all__ = [
    "clamp_mask",
    "clamp_",
    "discretize_mask",
    "project_to_binary",
    "is_binary",
    "relaxed_from_binary",
    "binary_from_relaxed",
    "project_inplace",
    "DEFAULT_LOWER",
    "DEFAULT_UPPER",
]

# The paper's interval endpoints (Appendix A).
DEFAULT_LOWER = -1.0
DEFAULT_UPPER = 1.0
#: Threshold used by the discretization rule: ``m_i < 0 -> 0``, ``m_i >= 0 -> 1``.
DISCRETIZATION_THRESHOLD = 0.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _is_torch(x: Any) -> bool:
    return _TORCH_AVAILABLE and isinstance(x, torch.Tensor)


def _is_array(x: Any) -> bool:
    return isinstance(x, np.ndarray) or (not _is_torch(x) and hasattr(x, "__array__"))


def _as_numpy(x: Any) -> np.ndarray:
    """Convert ``x`` (tensor / array / sequence) to a numpy array."""
    if _is_torch(x):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _like(reference: Any, values: Any, dtype: Optional[Any] = None):
    """Return ``values`` in the same container family as ``reference``."""
    if _is_torch(reference):
        t = torch.as_tensor(values)
        if dtype is not None:
            t = t.to(dtype=dtype)
        else:
            t = t.to(dtype=reference.dtype)
        return t.to(device=reference.device)
    arr = np.asarray(values)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _out_dtype(mask: Any, default=np.float64) -> Any:
    if _is_torch(mask):
        return mask.dtype
    arr = _as_numpy(mask)
    return arr.dtype if arr.dtype.kind == "f" else default


# ---------------------------------------------------------------------------
# 1) clamping during updates  (Algorithm 2, Appendix A "Technical details")
# ---------------------------------------------------------------------------
def clamp_mask(mask: Any, lower: float = DEFAULT_LOWER, upper: float = DEFAULT_UPPER):
    """Clamp relaxed mask values to ``[lower, upper]`` (default ``[-1, 1]``).

    Implements the paper's rule: "the value of m less than -1 becomes -1 and the
    value greater than 1 becomes 1".

    Parameters
    ----------
    mask:
        Relaxed mask (array-like or tensor) of shape ``(n,)`` (or any shape).
    lower, upper:
        Interval bounds; defaults are the paper's ``-1`` and ``1``.

    Returns
    -------
    Same container type as ``mask`` with every entry inside ``[lower, upper]``.
    """
    if _is_torch(mask):
        return torch.clamp(mask, min=float(lower), max=float(upper))
    arr = _as_numpy(mask)
    return np.clip(arr, float(lower), float(upper))


def clamp_(
    mask: Any, lower: float = DEFAULT_LOWER, upper: float = DEFAULT_UPPER
) -> Any:
    """In-place clamp for tensors (falls back to :func:`clamp_mask` for arrays).

    Used inside the LexiFlow update step where ``m_{t+1} = m_t +- delta * u`` is
    immediately clamped back into the feasible box.
    """
    if _is_torch(mask):
        with torch.no_grad():
            mask.clamp_(min=float(lower), max=float(upper))
        return mask
    if isinstance(mask, np.ndarray):
        np.clip(mask, float(lower), float(upper), out=mask)
        return mask
    return clamp_mask(mask, lower=lower, upper=upper)


# ---------------------------------------------------------------------------
# 2) discretization / final projection
# ---------------------------------------------------------------------------
def discretize_mask(
    mask: Any,
    threshold: float = DISCRETIZATION_THRESHOLD,
    *,
    lower: float = DEFAULT_LOWER,
    upper: float = DEFAULT_UPPER,
    clamp_first: bool = True,
):
    """Project a relaxed mask in ``[-1, 1]`` onto ``{0, 1}^n``.

    Paper rule (Appendix A): entries in ``[-1, 0)`` are projected to ``0`` and
    entries in ``[0, 1]`` are projected to ``1`` -- i.e. the threshold is exactly
    ``0`` with the *inclusive* upper half.

    Parameters
    ----------
    mask:
        Relaxed mask (any container).
    threshold:
        Boundary between the two intervals; must be ``0.0`` to match the paper.
        Exposed only for unit tests of alternative conventions.
    lower, upper:
        Assumed physical range of the relaxed mask.  When ``clamp_first`` is True
        the input is first mapped through :func:`clamp_mask` so that out-of-range
        values cannot silently violate the paper's intervals.
    clamp_first:
        Clamp to ``[lower, upper]`` before thresholding.

    Returns
    -------
    Binary mask in ``{0., 1.}`` with the same container type as the input.
    """
    if _is_torch(mask):
        x = clamp_mask(mask, lower, upper) if clamp_first else mask
        binary = (x >= float(threshold)).to(x.dtype)
        return binary
    arr = _as_numpy(mask).astype(np.float64, copy=True)
    if clamp_first:
        np.clip(arr, float(lower), float(upper), out=arr)
    binary = (arr >= float(threshold)).astype(_pick_float_dtype(mask))
    return _like(mask, binary) if _is_torch(mask) else binary


def project_to_binary(mask: Any, **kwargs):
    """Alias of :func:`discretize_mask` used for the final coreset mask.

    The name mirrors the plan's vocabulary ("project values in [-1,0) to 0 and
    values in [0,1] to 1").
    """
    return discretize_mask(mask, **kwargs)


def project_inplace(mask: Any, **kwargs) -> Any:
    """In-place discretization for numpy arrays, container-preserving otherwise."""
    if isinstance(mask, np.ndarray):
        out = discretize_mask(mask, **kwargs)
        mask[...] = out
        return mask
    return discretize_mask(mask, **kwargs)


def _pick_float_dtype(mask: Any) -> Any:
    if _is_array(mask):
        arr = _as_numpy(mask)
        if getattr(arr, "dtype", None) is not None and arr.dtype.kind == "f":
            return arr.dtype
    return np.float64


# ---------------------------------------------------------------------------
# 3) convenience conversions
# ---------------------------------------------------------------------------
def is_binary(mask: Any, atol: float = 1e-9) -> bool:
    """True iff every entry of ``mask`` is (numerically) in ``{0, 1}``."""
    arr = _as_numpy(mask)
    if arr.size == 0:
        return True
    return bool(np.all(np.isclose(arr, 0.0, atol=atol) | np.isclose(arr, 1.0, atol=atol)))


def relaxed_from_binary(mask: Any, *, positive: float = 1.0, negative: float = -1.0):
    """Lift a binary mask ``{0,1}`` to the relaxed ``{-1, +1}`` representation.

    Consistent with the appendix intervals: a selected example (``1``) becomes
    ``+1`` (inside ``[0, 1]``) and an unselected example (``0``) becomes ``-1``
    (inside ``[-1, 0)``), so a round trip through
    :func:`discretize_mask` recovers the original mask exactly.
    """
    arr = _as_numpy(mask).astype(np.float64)
    out = np.where(arr > 0.5, positive, negative)
    return _like(mask, out) if _is_torch(mask) else out


def binary_from_relaxed(mask: Any, **kwargs):
    """Alias of :func:`discretize_mask` for readability at call sites."""
    return discretize_mask(mask, **kwargs)


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def _selftest() -> None:  # pragma: no cover - exercised via unit tests
    x = np.array([-2.0, -1.0, -0.5, -1e-12, 0.0, 1e-12, 0.5, 1.0, 2.0])
    clamped = clamp_mask(x)
    assert np.allclose(clamped, [-1.0, -1.0, -0.5, -1e-12, 0.0, 1e-12, 0.5, 1.0, 1.0])
    bin_ = discretize_mask(x)
    assert np.array_equal(bin_, [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    # round trip from a binary mask
    b = np.array([0.0, 1.0, 1.0, 0.0])
    assert np.array_equal(discretize_mask(relaxed_from_binary(b)), b)
    assert is_binary(b) and not is_binary(np.array([0.5]))


if __name__ == "__main__":  # pragma: no cover
    _selftest()
    print("discretize.py self-test passed")
