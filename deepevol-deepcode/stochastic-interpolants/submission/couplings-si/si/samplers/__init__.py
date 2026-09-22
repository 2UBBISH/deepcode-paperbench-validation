"""Samplers for stochastic interpolants with data-dependent couplings.

This package exposes the integration routines used to draw samples from the
coupled stochastic-interpolant model:

* the probability-flow ODE ``X_dot = b_hat_t(X_t, xi)`` (paper Section 3.4,
  Algorithm 2), solved either with an adaptive Dopri solver from
  :mod:`torchdiffeq` or with a fixed-step forward-Euler / RK4 integrator, and
* the optional forward/backward SDEs ``dX^F`` / ``dX^R`` (paper Corollary 3.1,
  Eqs. 11 and 13), which additionally require the score network ``g_hat``.

The heavy lifting lives in :mod:`si.samplers.ode` (deterministic paths, used in
all reported experiments) and :mod:`si.samplers.sde` (stochastic paths, only
needed when ``gamma_t != 0``).  This module is a thin aggregator: it re-exports
both behind a guarded import so that ODE-only usage keeps working even when the
optional SDE module is unavailable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .ode import (
    DOPRI_METHODS,
    FIXED_METHODS,
    TORCHDIFFEQ_AVAILABLE,
    ODESampler,
    dopri_sample,
    euler_sample,
    initial_condition,
    make_drift_fn,
    make_project_fn,
    ode_sample,
    probability_flow_ode,
    rk4_sample,
    sample_from_coupling,
    solve_ode,
)

try:  # pragma: no cover - exercised only when the optional module exists
    from .sde import (
        SDESampler,
        forward_sde_sample,
        reverse_sde_sample,
        sde_sample,
    )

    SDE_AVAILABLE = True
except Exception:  # pragma: no cover
    SDESampler = None  # type: ignore[assignment]
    forward_sde_sample = None  # type: ignore[assignment]
    reverse_sde_sample = None  # type: ignore[assignment]
    sde_sample = None  # type: ignore[assignment]
    SDE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Registry / factory
# ---------------------------------------------------------------------------
SAMPLERS: Dict[str, Any] = {
    "dopri": dopri_sample,
    "dopri5": dopri_sample,
    "euler": euler_sample,
    "rk4": rk4_sample,
    "ode": probability_flow_ode,
    "probability_flow": probability_flow_ode,
}
if SDE_AVAILABLE:
    SAMPLERS.update(
        {
            "sde": sde_sample,
            "sde_forward": forward_sde_sample,
            "sde_reverse": reverse_sde_sample,
        }
    )


def get_sampler(name: str, **kwargs: Any):
    """Return a callable sampler selected by ``name``.

    Parameters
    ----------
    name:
        One of ``"dopri"``/``"dopri5"``, ``"euler"``, ``"rk4"``,
        ``"ode"``/``"probability_flow"`` (and ``"sde"``,
        ``"sde_forward"``/``"sde_reverse"`` when the optional SDE module is
        present).
    **kwargs:
        Forwarded to :class:`~si.samplers.ode.ODESampler` when a class-based
        sampler is requested via ``name in {"ode_sampler", "sde_sampler"}``;
        otherwise ignored in favour of returning the plain function.
    """
    key = str(name).lower()

    if key in {"ode_sampler", "odesampler"}:
        return ODESampler(**kwargs)
    if key in {"sde_sampler", "sdesampler"}:
        if not SDE_AVAILABLE:
            raise ImportError(
                "si.samplers.sde is unavailable; cannot build an SDESampler."
            )
        return SDESampler(**kwargs)

    if key not in SAMPLERS:
        # Fall back to a method-name lookup for the generic ODE solver.
        if key in DOPRI_METHODS or key in FIXED_METHODS:
            return lambda model, x0, **kw: probability_flow_ode(
                model, x0, method=key, **kw
            )
        valid = sorted(set(SAMPLERS) | {"ode_sampler", "sde_sampler"})
        raise ValueError(
            f"Unknown sampler '{name}'. Valid names: {valid}"
        )
    return SAMPLERS[key]


def list_samplers() -> List[str]:
    """Return the sorted list of sampler names known to the registry."""
    return sorted(SAMPLERS)


__all__: List[str] = [
    # ODE samplers / solvers
    "probability_flow_ode",
    "ode_sample",
    "euler_sample",
    "dopri_sample",
    "rk4_sample",
    "solve_ode",
    "sample_from_coupling",
    "initial_condition",
    "make_drift_fn",
    "make_project_fn",
    "ODESampler",
    # SDE samplers (optional)
    "sde_sample",
    "forward_sde_sample",
    "reverse_sde_sample",
    "SDESampler",
    "SDE_AVAILABLE",
    # constants
    "TORCHDIFFEQ_AVAILABLE",
    "DOPRI_METHODS",
    "FIXED_METHODS",
    # registry / factory
    "SAMPLERS",
    "get_sampler",
    "list_samplers",
]
