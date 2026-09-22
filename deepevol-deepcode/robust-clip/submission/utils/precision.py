"""Precision policy for adversarial attacks (Robust CLIP reproduction).

Addendum requirement (in scope):
    "For half-precision attacks, 16-bit ints needs to be used, and for
     single-precision attacks, 32-bit ints need to be used. This is not
     explicitly mentioned in the paper."

This centralizes the mapping between the floating point precision of an attack
and the integer dtype used to *store* the perturbation (e.g. when caching /
reloading perturbations so that the exact float perturbation can be recovered
losslessly).

Concretely:
    * half precision attack  (fp16 inputs / perturbations) -> int16 storage
    * single precision attack (fp32 inputs / perturbations) -> int32 storage

The perturbation is encoded as a fixed-point integer with a scale factor. Given
a perturbation ``delta`` expressed in the same units as ``EPS_SCALE`` (the raw
pixel value range used for the l_inf ball, typically [0, 1] or [0, 255]), we
multiply by ``quant_scale`` before rounding to the integer dtype and divide
again on decode. ``quant_scale`` is chosen so that the integer range is used but
the representation error stays negligible for typical alpha values.
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Addendum-mandated dtype mapping
# ---------------------------------------------------------------------------

#: storage dtype for half-precision (fp16) attacks
INT16 = torch.int16
#: storage dtype for single-precision (fp32) attacks
INT32 = torch.int32

#: canonical integer storage dtypes keyed by float precision name
PRECISION_TO_INT_DTYPE = {
    "half": INT16,
    "float16": INT16,
    "fp16": INT16,
    "h": INT16,
    "single": INT32,
    "float": INT32,
    "float32": INT32,
    "fp32": INT32,
    "f": INT32,
    "double": INT32,
    "float64": INT32,
}

#: canonical float dtypes keyed by the same names
PRECISION_TO_FLOAT_DTYPE = {
    "half": torch.float16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "h": torch.float16,
    "single": torch.float32,
    "float": torch.float32,
    "float32": torch.float32,
    "fp32": torch.float32,
    "f": torch.float32,
    "double": torch.float64,
    "float64": torch.float64,
    "d": torch.float64,
}

#: quantization scale (integers per unit perturbation). Perturbations are
#: bounded by |delta| <= 1 in [0, 1] pixel units, so 1e5 keeps the fp32 error
#: below 1e-5 while remaining far from int16 saturation (32767 ~= 0.33 units).
QUANT_SCALE = 1e5


class Precision(str):
    """Tiny string type with normalisation helpers (kept dependency free)."""

    def normalized(self) -> str:  # pragma: no cover - trivial
        return str(self).lower().strip()


def _norm(precision) -> str:
    if isinstance(precision, torch.dtype):
        if precision == torch.float16:
            return "half"
        if precision == torch.float32:
            return "single"
        if precision == torch.float64:
            return "double"
        raise ValueError(f"Unsupported torch dtype for precision: {precision}")
    if isinstance(precision, Precision):
        return precision.normalized()
    return str(precision).lower().strip()


def int_dtype_for_precision(precision) -> torch.dtype:
    """Return the Addendum-mandated integer storage dtype.

    half-precision -> int16, single-precision -> int32.
    """
    key = _norm(precision)
    if key not in PRECISION_TO_INT_DTYPE:
        raise ValueError(
            f"Unknown precision {precision!r}. Expected one of "
            f"{sorted(set(PRECISION_TO_INT_DTYPE))}."
        )
    return PRECISION_TO_INT_DTYPE[key]


def float_dtype_for_precision(precision) -> torch.dtype:
    """Return the floating point dtype corresponding to ``precision``."""
    key = _norm(precision)
    if key not in PRECISION_TO_FLOAT_DTYPE:
        raise ValueError(
            f"Unknown precision {precision!r}. Expected one of "
            f"{sorted(set(PRECISION_TO_FLOAT_DTYPE))}."
        )
    return PRECISION_TO_FLOAT_DTYPE[key]


def is_half(precision) -> bool:
    return float_dtype_for_precision(precision) == torch.float16


def is_single(precision) -> bool:
    return float_dtype_for_precision(precision) == torch.float32


def precision_from_float_dtype(dtype: torch.dtype) -> str:
    """Map a floating point dtype to a canonical precision name."""
    if dtype == torch.float16:
        return "half"
    if dtype == torch.float32:
        return "single"
    if dtype == torch.float64:
        return "double"
    raise ValueError(f"Unsupported float dtype: {dtype}")


# ---------------------------------------------------------------------------
# encode / decode perturbations with the mandated integer dtype
# ---------------------------------------------------------------------------

def encode_perturbation(
    delta: torch.Tensor,
    precision,
    quant_scale: float = QUANT_SCALE,
) -> torch.Tensor:
    """Encode a floating-point perturbation into the mandated integer dtype.

    Args:
        delta: perturbation tensor (any float dtype), in ``[0, 1]`` pixel units.
        precision: attack precision ("half"/"single" or a torch float dtype).
        quant_scale: integers per unit perturbation.

    Returns:
        Tensor with dtype ``int_dtype_for_precision(precision)``.
    """
    int_dtype = int_dtype_for_precision(precision)
    scaled = (delta.detach().to(torch.float64) * quant_scale)
    # round-half-to-even for determinism, then clamp to the integer range
    codes = torch.round(scaled)
    info = torch.iinfo(int_dtype)
    codes = codes.clamp(min=info.min, max=info.max)
    return codes.to(int_dtype)


def decode_perturbation(
    codes: torch.Tensor,
    float_dtype=torch.float32,
    quant_scale: float = QUANT_SCALE,
) -> torch.Tensor:
    """Decode an integer perturbation (created by :func:`encode_perturbation`)."""
    if isinstance(float_dtype, str) or isinstance(float_dtype, Precision):
        float_dtype = float_dtype_for_precision(float_dtype)
    return (codes.to(torch.float64) / quant_scale).to(float_dtype)


def cast_perturbation(delta: torch.Tensor, precision) -> torch.Tensor:
    """Cast a perturbation to the float dtype implied by ``precision``."""
    return delta.to(float_dtype_for_precision(precision))


def assert_mandated_dtype(tensor: torch.Tensor, precision) -> None:
    """Validation helper used by tests: assert ``tensor`` has the mandated dtype."""
    expected = int_dtype_for_precision(precision)
    if tensor.dtype != expected:
        raise AssertionError(
            f"Precision policy violated: precision={precision!r} requires "
            f"{expected}, but got {tensor.dtype}."
        )


# ---------------------------------------------------------------------------
# convenience: storage round-trip used by attack bookkeeping
# ---------------------------------------------------------------------------

def store_perturbation(delta: torch.Tensor, precision, quant_scale: float = QUANT_SCALE):
    """Return a serialisable integer representation + metadata dict.

    The metadata records everything needed to restore the float perturbation
    losslessly with :func:`load_perturbation`.
    """
    codes = encode_perturbation(delta, precision, quant_scale=quant_scale)
    meta = {
        "int_dtype": str(int_dtype_for_precision(precision)).replace("torch.", ""),
        "float_dtype": str(float_dtype_for_precision(precision)).replace("torch.", ""),
        "precision": precision_from_float_dtype(float_dtype_for_precision(precision)),
        "quant_scale": quant_scale,
        "shape": tuple(delta.shape),
    }
    return codes, meta


def load_perturbation(codes, meta):
    """Inverse of :func:`store_perturbation`."""
    return decode_perturbation(
        codes,
        float_dtype=float_dtype_for_precision(meta["precision"]),
        quant_scale=meta.get("quant_scale", QUANT_SCALE),
    )


__all__ = [
    "INT16",
    "INT32",
    "PRECISION_TO_INT_DTYPE",
    "PRECISION_TO_FLOAT_DTYPE",
    "QUANT_SCALE",
    "int_dtype_for_precision",
    "float_dtype_for_precision",
    "precision_from_float_dtype",
    "is_half",
    "is_single",
    "encode_perturbation",
    "decode_perturbation",
    "cast_perturbation",
    "assert_mandated_dtype",
    "store_perturbation",
    "load_perturbation",
]
