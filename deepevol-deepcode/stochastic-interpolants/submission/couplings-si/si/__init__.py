"""Stochastic Interpolants with Data-Dependent Couplings.

Reference implementation of

    "Stochastic Interpolants with Data-Dependent Couplings"

The package is organised as a small set of orthogonal pieces:

``si.interpolants``
    The interpolant process ``I_t = alpha_t x0 + beta_t x1 + gamma_t z`` and its
    analytic time derivative (Section 3, Def. 3.1, Eq. 1).

``si.couplings``
    Data-dependent couplings ``rho(x0, x1) = rho1(x1) rho0(x0 | x1)`` with
    ``x0 = m(x1) + sigma zeta`` (Section 3.2, Eqs. 16-19).  Concrete couplings:
    in-painting (Section 4.1) and super-resolution (Section 4.2).

``si.models``
    The velocity network ``b_hat_t(x, xi)`` (Appendix B) and the optional score
    network ``g_hat_t(x, xi)`` (Section 3.1).

``si.losses``
    The simulation-free regression objectives: velocity loss ``L_b`` (Eq. 22)
    and the optional score loss ``L_g`` (Eq. 7).

``si.samplers``
    Probability-flow ODE integration (Algorithm 2) and the optional forward /
    backward SDEs (Eqs. 11 and 13).

``si.data``
    ImageNet-1k loading and the ``[-1, 1]`` image conventions.

``si.utils``
    Config loading, distributed (Lightning Fabric) helpers and optional EMA.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "__version__",
]


def __dir__():  # pragma: no cover - introspection helper
    return sorted(list(globals().keys()) + list(__all__))
