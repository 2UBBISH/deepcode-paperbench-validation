"""Interpolant coefficients alpha_t, beta_t, gamma_t and their time derivatives.

The stochastic interpolant (Definition 3.1 of the paper) is

    I_t = alpha_t x_0 + beta_t x_1 + gamma_t z ,      t in [0, 1]

where alpha_t, beta_t and gamma_t^2 are differentiable functions of time
satisfying the boundary conditions

    alpha_0 = beta_1 = 1,   alpha_1 = beta_0 = gamma_0 = gamma_1 = 0,
    alpha_t^2 + beta_t^2 + gamma_t^2 > 0  for all t in [0, 1].

This module exposes a small set of *presets* that satisfy those conditions and
provide, for each of them, the coefficients as well as their analytic time
derivatives so that the interpolant process and the losses can depend on them
uniformly.

Presets
-------
``linear``      alpha_t = 1 - t, beta_t = t, gamma_t = sqrt(2 t (1 - t))
                (the "simple instance" of Eq. (1) used throughout the theory;
                this is the standard stochastic-interpolant choice).

``gamma0``      alpha_t = 1 - t, beta_t = t, gamma_t = 0
                (the deterministic choice used when the base density is a
                "noisy version of the target", cf. Section 3.2 / Eq. 20; the
                z term is unused and the score is not available from gamma_t).

``inpainting``  alpha_t = t, beta_t = 1 - t, gamma_t = 0
                (Section 4.1: "In the interpolant (20), we set alpha_t = t and
                beta_t = 1 - t").  Note this reverses the usual orientation so
                that I_0 = x_0 (the corrupted image) and I_1 = x_1 (the clean
                image).

``superres``    alias of ``gamma0`` (the plan's default for super-resolution:
                the gamma_t = 0 coupling style with alpha_t = 1 - t, beta_t = t).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import torch

__all__ = [
    "Coefficients",
    "linear",
    "gamma0",
    "inpainting",
    "superres",
    "get_coefficients",
    "COEFFICIENT_PRESETS",
]


class Coefficients:
    """Container of the coefficient callables for a given interpolant preset.

    Each callable takes a tensor of times ``t`` (any shape, typically ``(B,)``)
    and returns a tensor broadcastable against a batch of samples of shape
    ``(B, ...)``.  The returned value has the same leading dimension as ``t`` and
    is unsqueezed to ``(B, 1, 1, ...)`` by :meth:`as_coeff` — but here we simply
    return shape ``t.shape`` and rely on the caller to unsqueeze.
    """

    def __init__(
        self,
        name: str,
        alpha: Callable[[torch.Tensor], torch.Tensor],
        beta: Callable[[torch.Tensor], torch.Tensor],
        gamma: Callable[[torch.Tensor], torch.Tensor],
        alpha_dot: Callable[[torch.Tensor], torch.Tensor],
        beta_dot: Callable[[torch.Tensor], torch.Tensor],
        gamma_dot: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        self.name = name
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.alpha_dot = alpha_dot
        self.beta_dot = beta_dot
        self.gamma_dot = gamma_dot

    # -- evaluation helpers -------------------------------------------------
    def evaluate(
        self, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(alpha_t, beta_t, gamma_t)`` evaluated at ``t``."""
        return self.alpha(t), self.beta(t), self.gamma(t)

    def evaluate_dot(
        self, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(alpha_dot_t, beta_dot_t, gamma_dot_t)`` evaluated at ``t``."""
        return self.alpha_dot(t), self.beta_dot(t), self.gamma_dot(t)

    def uses_noise(self) -> bool:
        """Whether ``gamma_t`` is (structurally) non-zero for this preset."""
        # Probe a generic interior point; all presets here are non-zero identically
        # or identically zero, so a single probe suffices.
        return bool(torch.any(self.gamma(torch.tensor([0.37])) != 0))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Coefficients(name={self.name!r})"


# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------
def _linear() -> Coefficients:
    """alpha=1-t, beta=t, gamma=sqrt(2 t (1 - t))."""

    def alpha(t: torch.Tensor) -> torch.Tensor:
        return 1.0 - t

    def beta(t: torch.Tensor) -> torch.Tensor:
        return t

    def gamma(t: torch.Tensor) -> torch.Tensor:
        # Clamp to avoid NaNs from tiny negative values due to floating point.
        return torch.sqrt(torch.clamp(2.0 * t * (1.0 - t), min=0.0))

    def alpha_dot(t: torch.Tensor) -> torch.Tensor:
        return -torch.ones_like(t)

    def beta_dot(t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)

    def gamma_dot(t: torch.Tensor) -> torch.Tensor:
        # d/dt sqrt(2 t (1-t)) = (1 - 2 t) / sqrt(2 t (1 - t))
        inner = torch.clamp(2.0 * t * (1.0 - t), min=0.0)
        sqrt = torch.sqrt(inner)
        num = 1.0 - 2.0 * t
        # Guard the singular endpoints (t=0, t=1) with a small epsilon; the value
        # is only ever multiplied by z there and gamma_0 = gamma_1 = 0.
        eps = 1e-12
        return num / torch.clamp(sqrt, min=eps)

    return Coefficients(
        "linear", alpha, beta, gamma, alpha_dot, beta_dot, gamma_dot
    )


def _gamma0() -> Coefficients:
    """alpha=1-t, beta=t, gamma=0 (the deterministic choice used in experiments)."""

    def alpha(t: torch.Tensor) -> torch.Tensor:
        return 1.0 - t

    def beta(t: torch.Tensor) -> torch.Tensor:
        return t

    def gamma(t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(t)

    def alpha_dot(t: torch.Tensor) -> torch.Tensor:
        return -torch.ones_like(t)

    def beta_dot(t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)

    def gamma_dot(t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(t)

    return Coefficients(
        "gamma0", alpha, beta, gamma, alpha_dot, beta_dot, gamma_dot
    )


def _inpainting() -> Coefficients:
    """alpha=t, beta=1-t, gamma=0 (Section 4.1)."""

    def alpha(t: torch.Tensor) -> torch.Tensor:
        return t

    def beta(t: torch.Tensor) -> torch.Tensor:
        return 1.0 - t

    def gamma(t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(t)

    def alpha_dot(t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)

    def beta_dot(t: torch.Tensor) -> torch.Tensor:
        return -torch.ones_like(t)

    def gamma_dot(t: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(t)

    return Coefficients(
        "inpainting", alpha, beta, gamma, alpha_dot, beta_dot, gamma_dot
    )


# Registry of the available presets.
COEFFICIENT_PRESETS: Dict[str, Callable[[], Coefficients]] = {
    "linear": _linear,
    "gamma0": _gamma0,
    "inpainting": _inpainting,
    "superres": _gamma0,  # alias for the super-resolution default
}


def get_coefficients(preset: str = "linear") -> Coefficients:
    """Return a fresh :class:`Coefficients` instance for ``preset``.

    Parameters
    ----------
    preset:
        One of ``"linear"``, ``"gamma0"``, ``"inpainting"``, ``"superres"``.
    """
    if preset not in COEFFICIENT_PRESETS:
        raise ValueError(
            f"Unknown coefficient preset {preset!r}. "
            f"Available: {sorted(COEFFICIENT_PRESETS)}"
        )
    return COEFFICIENT_PRESETS[preset]()


# Convenience module-level singletons (matching the plan's naming).
linear = get_coefficients("linear")
gamma0 = get_coefficients("gamma0")
inpainting = get_coefficients("inpainting")
superres = get_coefficients("superres")


# ---------------------------------------------------------------------------
# Self-test / boundary conditions (used by unit tests and phase-1 validation)
# ---------------------------------------------------------------------------
def _check_boundary_conditions(coeffs: Coefficients, atol: float = 1e-6) -> None:
    t0 = torch.tensor([0.0])
    t1 = torch.tensor([1.0])

    a0, b0, g0 = coeffs.evaluate(t0)
    a1, b1, g1 = coeffs.evaluate(t1)

    assert math.isclose(float(a0), 1.0, abs_tol=atol), (coeffs.name, "alpha_0")
    assert math.isclose(float(b1), 1.0, abs_tol=atol), (coeffs.name, "beta_1")
    assert math.isclose(float(a1), 0.0, abs_tol=atol), (coeffs.name, "alpha_1")
    assert math.isclose(float(b0), 0.0, abs_tol=atol), (coeffs.name, "beta_0")
    assert math.isclose(float(g0), 0.0, abs_tol=atol), (coeffs.name, "gamma_0")
    assert math.isclose(float(g1), 0.0, abs_tol=atol), (coeffs.name, "gamma_1")

    ts = torch.linspace(0, 1, 64)
    a, b, g = coeffs.evaluate(ts)
    assert torch.all(a**2 + b**2 + g**2 > 0), (coeffs.name, "positivity")


def _check_derivatives(coeffs: Coefficients, atol: float = 1e-4) -> None:
    """Finite-difference check of the analytic derivatives."""
    ts = torch.linspace(0.05, 0.95, 32)
    h = 1e-5
    for fn, dfn in (
        (coeffs.alpha, coeffs.alpha_dot),
        (coeffs.beta, coeffs.beta_dot),
        (coeffs.gamma, coeffs.gamma_dot),
    ):
        fd = (fn(ts + h) - fn(ts - h)) / (2 * h)
        assert torch.allclose(fd, dfn(ts), atol=atol), coeffs.name


if __name__ == "__main__":  # pragma: no cover
    for _name in COEFFICIENT_PRESETS:
        _c = get_coefficients(_name)
        _check_boundary_conditions(_c)
        _check_derivatives(_c)
        print(f"ok: {_name}")
