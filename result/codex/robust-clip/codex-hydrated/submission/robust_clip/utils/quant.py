"""Integer-grid quantisation of adversarial perturbations.

The addendum to the paper specifies that

    "For half-precision attacks, 16-bit ints need to be used, and for
     single-precision attacks, 32-bit ints need to be used."

Images are handled on the integer grid ``{0, ..., 255}`` (uint8 image data) and
the perturbation that is added to them is *rounded to an integer* before being
added back.  Which integer dtype is used depends on the precision of the model
the attack is run against: the fp16 model is attacked with perturbations
represented as ``int16`` and the fp32 model with ``int32``.  This makes the
half-precision attack cheaper (and slightly weaker) than the single-precision
one, which is exactly why the ensemble pipeline of Sec. 4.1 first runs a cheap
half-precision attack and only then a single-precision one.
"""
from __future__ import annotations

import torch


INT_DTYPES = {
    16: torch.int16,
    32: torch.int32,
    64: torch.int64,
}


def int_dtype_for_precision(bits: int) -> torch.dtype:
    if bits not in INT_DTYPES:
        raise ValueError(f"unsupported integer width {bits}; expected 16, 32 or 64")
    return INT_DTYPES[bits]


def quantize_pixels(images: torch.Tensor, bits: int = 32, scale: float = 255.0) -> torch.Tensor:
    """Round ``images`` (assumed to live in ``[0, 1]``) onto the integer grid.

    ``bits`` selects the integer width used for the rounding (16 for
    half-precision attacks, 32 for single-precision attacks).  The returned
    tensor keeps the float dtype of ``images`` and stays in ``[0, 1]``.
    """
    if bits is None:
        return images
    dtype = int_dtype_for_precision(bits)
    rounded = torch.round(images * scale).to(dtype)
    return rounded.to(images.dtype) / scale


def quantize_delta(delta: torch.Tensor, bits: int = 32, scale: float = 255.0) -> torch.Tensor:
    """Round a pixel-space perturbation onto the integer grid."""
    if bits is None:
        return delta
    dtype = int_dtype_for_precision(bits)
    return torch.round(delta * scale).to(dtype).to(delta.dtype) / scale
