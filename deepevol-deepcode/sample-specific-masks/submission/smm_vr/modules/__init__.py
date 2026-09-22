"""SMM core module package.

This package aggregates the two non-learnable/learnable building blocks of the
Sample-specific Multi-channel Masks (SMM) method:

* :mod:`smm_vr.modules.patch_interp` -- the non-parametric patch-wise
  (block-replication) interpolation that upsamples the low-resolution mask
  produced by ``f_mask`` back to the pre-trained model's input resolution
  (paper Sec. 3.3, Appendix A.3, Table 5).
* :mod:`smm_vr.modules.reprogram` -- the reprogramming function
  ``f_in(x_i) = r(x_i) + delta * mask_i`` (paper Eq. 4) together with the
  learnable shared pattern ``delta`` (zero-initialized, Algorithm 1) and the
  shared-mask baselines' wrapper (Pad/Narrow/Medium/Full).

All re-exports are wrapped in ``try/except ImportError`` so that
``import smm_vr.modules`` never fails while the project is being built
incrementally.
"""

from __future__ import annotations

from typing import List

__all__: List[str] = []


def _extend(names) -> None:
    """Append ``names`` to ``__all__`` without duplicates."""
    for name in names:
        if name not in __all__:
            __all__append(name)


def __all__append(name: str) -> None:
    __all__.append(name)


# ---------------------------------------------------------------------------
# Patch-wise interpolation (Sec. 3.3, Appendix A.3)
# ---------------------------------------------------------------------------
_PATCH_INTERP_AVAILABLE = False
try:  # pragma: no cover - guarded import for partial builds
    from .patch_interp import (  # noqa: F401
        DEFAULT_PATCH_SIZE,
        PatchWiseInterpolation,
        patch_wise_interpolate,
    )

    _PATCH_INTERP_AVAILABLE = True
    _extend(
        [
            "PatchWiseInterpolation",
            "patch_wise_interpolate",
            "DEFAULT_PATCH_SIZE",
        ]
    )
except ImportError:  # pragma: no cover
    DEFAULT_PATCH_SIZE = 8


# ---------------------------------------------------------------------------
# Reprogramming function f_in (Eq. 4, Algorithm 1) + shared-mask baselines
# ---------------------------------------------------------------------------
_REPROGRAM_AVAILABLE = False
try:  # pragma: no cover - guarded import for partial builds
    from .reprogram import (  # noqa: F401
        SMMReprogram,
        SMMReprogramming,
        SharedMaskReprogram,
        build_smm_reprogram,
        init_zero_pattern,
    )

    _REPROGRAM_AVAILABLE = True
    _extend(
        [
            "SMMReprogram",
            "SMMReprogramming",
            "SharedMaskReprogram",
            "build_smm_reprogram",
            "init_zero_pattern",
        ]
    )
except ImportError:  # pragma: no cover
    pass


def build_reprogram(backbone: str = "resnet18", **kwargs):
    """Convenience accessor for the SMM ``f_in`` wrapper.

    Delegates to :func:`smm_vr.modules.reprogram.build_smm_reprogram` so that
    callers can obtain a reprogramming module from a single import surface.
    Raises ``ImportError`` if the reprogramming module could not be imported.
    """
    if not _REPROGRAM_AVAILABLE:
        raise ImportError(
            "smm_vr.modules.reprogram is unavailable; cannot build SMM f_in."
        )
    return build_smm_reprogram(backbone=backbone, **kwargs)


_extend(["build_reprogram"])
