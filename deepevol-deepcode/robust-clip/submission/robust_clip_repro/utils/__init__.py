"""Utility subpackage for the Robust CLIP reproduction.

Exposes the small, dependency-light helpers that every other part of the
package relies on:

* :mod:`robust_clip_repro.utils.precision` -- the Addendum-mandated precision
  policy (``int16`` for half-precision attacks, ``int32`` for single-precision
  attacks) plus fixed-point encode/decode and storage helpers for adversarial
  perturbations.
* :mod:`robust_clip_repro.utils.normalization` -- pixel-space vs. model-space
  bookkeeping so that every l_inf / l_2 ball is computed around *non-normalized*
  ``[0, 1]`` pixels, exactly as the Addendum requires.
* :mod:`robust_clip_repro.utils.logging` -- logging, JSON export and
  hyper-parameter provenance bookkeeping (so values the Addendum never states
  are logged as externally supplied instead of being invented).

Nothing in this subpackage contains paper-specific formulas; it is pure glue.
Importing :mod:`robust_clip_repro.utils` is intentionally side-effect free and
must not import torch eagerly (the individual modules lazily import their heavy
dependencies).
"""

from __future__ import annotations

from typing import List

from . import logging as _logging
from . import normalization as _normalization
from . import precision as _precision

__all__: List[str] = [
    "precision",
    "normalization",
    "logging",
    # precision policy (Addendum)
    "Precision",
    "INT16",
    "INT32",
    "QUANT_SCALE",
    "int_dtype_for_precision",
    "float_dtype_for_precision",
    "precision_from_float_dtype",
    "encode_perturbation",
    "decode_perturbation",
    "cast_perturbation",
    "assert_mandated_dtype",
    "store_perturbation",
    "load_perturbation",
    "is_half",
    "is_single",
    # raw vs normalized pixel handling (Addendum)
    "NormalizationSpec",
    "get_normalization",
    "normalize_pixels",
    "denormalize_pixels",
    "ensure_raw_pixels",
    "clamp_pixels",
    "adversarial_pixels",
    "project_linf",
    "project_l2",
    "project_ball",
    "perturbation_norm",
    "ball_contains",
    "pixel_delta_from_model_delta",
    "model_delta_from_pixel_delta",
    "normalization_metadata",
    "CLIP_MEAN",
    "CLIP_STD",
    "PIXEL_MIN",
    "PIXEL_MAX",
    # logging / provenance
    "get_logger",
    "setup_logging",
    "configure_from_args",
    "log_config",
    "log_provenance",
    "provenance_metadata",
    "save_json",
    "load_json",
    "to_json",
    "log_metric",
    "log_table",
    "Timer",
    "log_pixel_space",
    "UNSPECIFIED",
    "EXTERNAL_DEFAULT",
]


def __getattr__(name: str):  # pragma: no cover - thin re-export shim
    """Lazily resolve re-exported names to keep imports cheap and optional.

    ``utils.precision`` has no third-party dependency, but
    ``utils.normalization`` and the logging module may touch torch/yaml at call
    time.  Resolving attribute access through this shim means
    ``from robust_clip_repro.utils import project_linf`` still works without the
    package eagerly importing every submodule.
    """

    for module in (_precision, _normalization, _logging):
        attribute = getattr(module, name, None)
        if attribute is not None:
            return attribute
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():  # pragma: no cover - introspection helper
    return sorted(set(globals()) | set(__all__))
