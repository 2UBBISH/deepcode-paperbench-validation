"""Stochastic-interpolant core: coefficients and the interpolant process.

This package implements the mathematical objects of *Stochastic Interpolants
with Data-Dependent Couplings*:

* :mod:`si.interpolants.coefficients` -- the time-dependent coefficient
  functions ``alpha_t``, ``beta_t``, ``gamma_t`` (and their analytic
  derivatives) satisfying the boundary conditions of Definition 3.1
  (Eq. 1 / Eq. 20):

      alpha_0 = beta_1 = 1,   alpha_1 = beta_0 = gamma_0 = gamma_1 = 0,
      alpha_t^2 + beta_t^2 + gamma_t^2 > 0.

  Presets:

  - ``"linear"``      : ``alpha_t = 1 - t``, ``beta_t = t``,
                        ``gamma_t = sqrt(2 t (1 - t))``
  - ``"gamma0"``      : ``alpha_t = 1 - t``, ``beta_t = t``, ``gamma_t = 0``
                        (the deterministic preset used in the reported runs)
  - ``"inpainting"``  : ``alpha_t = t``, ``beta_t = 1 - t``, ``gamma_t = 0``
                        (Section 4.1 orientation, so ``I_0 = x_0`` corrupt,
                        ``I_1 = x_1`` clean)
  - ``"superres"``    : alias of ``"gamma0"``

* :mod:`si.interpolants.interpolant` -- assembles the process
  ``I_t = alpha_t x_0 + beta_t x_1 + gamma_t z`` together with its time
  derivative ``I_dot_t = alpha_dot_t x_0 + beta_dot_t x_1 + gamma_dot_t z``
  (Definition 3.1, Eq. 1 / Eq. 20), including batch-time broadcasting.

Symbols are resolved lazily (PEP 562) so that ``import si.interpolants`` does
not eagerly pull in heavy sub-modules; the two underlying modules are
nevertheless dependency-light (``torch`` only).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

__all__: List[str] = [
    # coefficients
    "Coefficients",
    "get_coefficients",
    "COEFFICIENT_PRESETS",
    "linear",
    "gamma0",
    "inpainting",
    "superres",
    # interpolant process
    "Interpolant",
    "broadcast_t",
]

_COEFFICIENTS_NAMES = frozenset(
    {
        "Coefficients",
        "get_coefficients",
        "COEFFICIENT_PRESETS",
        "linear",
        "gamma0",
        "inpainting",
        "superres",
    }
)

_INTERPOLANT_NAMES = frozenset({"Interpolant", "broadcast_t"})


def __getattr__(name: str) -> Any:
    """Lazily resolve a public symbol from the coefficients/interpolant modules."""
    if name in _COEFFICIENTS_NAMES:
        from . import coefficients as _coefficients

        try:
            return getattr(_coefficients, name)
        except AttributeError as exc:  # pragma: no cover - defensive
            raise AttributeError(
                f"module 'si.interpolants.coefficients' has no attribute {name!r}"
            ) from exc

    if name in _INTERPOLANT_NAMES:
        from . import interpolant as _interpolant

        try:
            return getattr(_interpolant, name)
        except AttributeError as exc:  # pragma: no cover - defensive
            raise AttributeError(
                f"module 'si.interpolants.interpolant' has no attribute {name!r}"
            ) from exc

    raise AttributeError(f"module 'si.interpolants' has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(__all__))


def describe() -> Dict[str, Optional[str]]:  # pragma: no cover - introspection
    """Short human-readable descriptions of the sub-modules."""
    return {
        "coefficients": "alpha_t, beta_t, gamma_t (+ derivatives) presets (Def. 3.1, Eq. 1/20)",
        "interpolant": "I_t = alpha_t x0 + beta_t x1 + gamma_t z and its derivative",
    }
