"""Stochastic interpolant process (Definition 3.1, Eq. 1).

The interpolant is the process

    I_t = alpha_t x_0 + beta_t x_1 + gamma_t z ,      t in [0, 1]        (Eq. 1)

with z ~ N(0, Id) independent of the pair (x_0, x_1).  In the data-dependent
coupling setting of Section 3.2 the base sample is itself built from the target,
x_0 = m(x_1) + sigma zeta (Eq. 18), in which case gamma_t may be set to zero and
the interpolant reduces to

    I_t = alpha_t (m(x_1) + sigma zeta) + beta_t x_1 .                  (Eq. 20)

This module is agnostic to how (x_0, x_1) were coupled: it simply evaluates I_t
and its time derivative

    I_dot_t = alpha_dot_t x_0 + beta_dot_t x_1 + gamma_dot_t z

from the coefficients provided by :mod:`si.interpolants.coefficients`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from .coefficients import Coefficients, get_coefficients

__all__ = ["Interpolant", "broadcast_t"]


def broadcast_t(t: torch.Tensor, ndim: int) -> torch.Tensor:
    """Reshape a batch of times to broadcast against data of ``ndim`` dims.

    ``t`` may be a scalar tensor or of shape ``(B,)`` (per-item times).  The
    result has shape ``(B, 1, ..., 1)`` with ``ndim - 1`` trailing singleton
    axes so that it broadcasts against ``x`` of shape ``(B, ...)``.
    """
    if not torch.is_tensor(t):
        t = torch.as_tensor(t)
    if t.ndim == 0:
        return t
    if t.ndim == 1:
        return t.reshape(-1, *([1] * (ndim - 1)))
    return t


class Interpolant:
    """Evaluates ``I_t`` and ``I_dot_t`` for a given coefficient preset.

    Parameters
    ----------
    coefficients:
        A :class:`~si.interpolants.coefficients.Coefficients` instance, or any
        object exposing ``evaluate(t)`` / ``evaluate_dot(t)`` returning the
        triples ``(alpha_t, beta_t, gamma_t)`` and
        ``(alpha_dot_t, beta_dot_t, gamma_dot_t)``.
    """

    def __init__(self, coefficients: Optional[Coefficients] = None) -> None:
        if coefficients is None:
            coefficients = get_coefficients("linear")
        self.coefficients = coefficients

    # ------------------------------------------------------------------ #
    # coefficient accessors
    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        return getattr(self.coefficients, "name", "custom")

    def uses_noise(self) -> bool:
        """Whether the preset has a structurally non-zero ``gamma_t``."""
        uses = getattr(self.coefficients, "uses_noise", None)
        return bool(uses()) if callable(uses) else True

    def alpha_beta_gamma(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.coefficients.evaluate(t)

    def alpha_beta_gamma_dot(
        self, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.coefficients.evaluate_dot(t)

    # ------------------------------------------------------------------ #
    # interpolant evaluation
    # ------------------------------------------------------------------ #
    def I_t(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluate ``I_t = alpha_t x_0 + beta_t x_1 + gamma_t z`` (Eq. 1).

        ``t`` may be a scalar or of shape ``(B,)``; it is broadcast over the
        batch and all data dimensions.  When ``gamma_t`` is identically zero the
        noise ``z`` is unused and may be omitted.
        """
        alpha, beta, gamma = self.alpha_beta_gamma(t)
        tb = broadcast_t(t, x0.ndim)
        alpha = broadcast_t(alpha, x0.ndim) if alpha.ndim else alpha
        beta = broadcast_t(beta, x0.ndim) if beta.ndim else beta
        gamma = broadcast_t(gamma, x0.ndim) if gamma.ndim else gamma

        out = alpha * x0 + beta * x1
        if z is not None and bool(torch.any(gamma != 0)):
            out = out + gamma * z
        _ = tb  # kept for symmetry / debugging
        return out

    def I_dot(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluate ``I_dot_t = alpha_dot_t x_0 + beta_dot_t x_1 + gamma_dot_t z``."""
        alpha_dot, beta_dot, gamma_dot = self.alpha_beta_gamma_dot(t)
        alpha_dot = broadcast_t(alpha_dot, x0.ndim) if alpha_dot.ndim else alpha_dot
        beta_dot = broadcast_t(beta_dot, x0.ndim) if beta_dot.ndim else beta_dot
        gamma_dot = broadcast_t(gamma_dot, x0.ndim) if gamma_dot.ndim else gamma_dot

        out = alpha_dot * x0 + beta_dot * x1
        if z is not None and bool(torch.any(gamma_dot != 0)):
            out = out + gamma_dot * z
        return out

    # ------------------------------------------------------------------ #
    # convenience
    # ------------------------------------------------------------------ #
    def __call__(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        t: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        return_derivative: bool = False,
    ):
        """Return ``I_t`` (and optionally ``I_dot_t``)."""
        it = self.I_t(x0, x1, t, z)
        if not return_derivative:
            return it
        return it, self.I_dot(x0, x1, t, z)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Interpolant(preset={self.name!r})"


# -------------------------------------------------------------------------- #
# self-test: boundary conditions and analytic-vs-FD derivative agreement
# -------------------------------------------------------------------------- #
def _self_test() -> None:  # pragma: no cover
    torch.manual_seed(0)
    for preset in ("linear", "gamma0", "inpainting", "superres"):
        interp = Interpolant(get_coefficients(preset))
        B, d = 8, 5
        x0 = torch.randn(B, d)
        x1 = torch.randn(B, d)
        z = torch.randn(B, d)
        t = torch.rand(B)

        # t=0 must give x0 (alpha_0 = 1, beta_0 = gamma_0 = 0)
        i0 = interp.I_t(x0, x1, torch.zeros(()), z)
        assert torch.allclose(i0, x0, atol=1e-6), (preset, i0 - x0)
        # t=1 must give x1 (beta_1 = 1, alpha_1 = gamma_1 = 0)
        i1 = interp.I_t(x0, x1, torch.ones(()), z)
        assert torch.allclose(i1, x1, atol=1e-6), (preset, i1 - x1)

        # derivative check against finite differences
        eps = 1e-4
        fd = (interp.I_t(x0, x1, t + eps, z) - interp.I_t(x0, x1, t - eps, z)) / (2 * eps)
        ad = interp.I_dot(x0, x1, t, z)
        assert torch.allclose(fd, ad, atol=1e-3), (preset, (fd - ad).abs().max())
        print(f"[interpolant] {preset:10s} OK  gamma_t non-zero: {interp.uses_noise()}")


if __name__ == "__main__":  # pragma: no cover
    _self_test()
