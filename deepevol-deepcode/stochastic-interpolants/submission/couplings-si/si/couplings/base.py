"""Abstract interface for data-dependent couplings.

The paper (Section 3.2) replaces the independent base/target pairing of the standard
stochastic interpolant by a data-dependent coupling

    rho(x0, x1) = rho1(x1) rho0(x0 | x1),      int rho0(x0|x1) rho1(x1) dx1 = rho0(x0)

with a conditional base density of the generic form

    x0 = m(x1) + sigma * zeta,      zeta ~ N(0, Id),      zeta _||_ m(x1)

If ``m(x1)`` is deterministic given ``x1`` (possibly given additional conditional
information ``xi``) then

    rho0(x0 | x1) = N(x0 ; m(x1), C),      C = sigma sigma^T

In this case the interpolant can be used with ``gamma_t = 0`` and reduces to

    I_t = alpha_t (m(x1) + sigma zeta) + beta_t x1

The score associated to this Gaussian base is still available because of the factor
``sigma zeta``, so long as ``sigma`` is invertible.

A concrete coupling therefore only has to specify

  * ``m(x1, xi)``  -- the corrupted observation (masked pixels, low-res image, ...),
  * ``sigma``      -- the (scalar or matrix) standard deviation of the added noise,
  * the extra conditional information ``xi`` that must be handed to the model.

The :class:`Coupling` base class stores the interpolant coefficient preset and the
``sigma`` convention and exposes the two functions used by the training / sampling
loops:

  * :meth:`build_x0`       -- draw ``x0 = m(x1) + sigma zeta`` and return the
                              conditioning tensor ``xi`` to feed to the model.
  * :meth:`conditioning`   -- return only ``xi`` (used at sampling time when the
                              target ``x1`` is known through the observed data).
"""

from __future__ import annotations

import abc
from typing import Any, Optional, Tuple

import torch


class Coupling(abc.ABC):
    """Abstract data-dependent coupling ``rho(x0, x1) = rho1(x1) rho0(x0 | x1)``.

    Parameters
    ----------
    sigma:
        Scale of the injected Gaussian noise, ``sigma`` in ``x0 = m(x1) + sigma zeta``.
        ``sigma > 0`` smooths the base density off the low-dimensional manifold on
        which ``m(x1)`` lives (this is what keeps the score available so long as
        ``sigma`` is invertible).  ``sigma = 0`` is allowed for in-painting where the
        base is already supported on the full space (it equals a masked image plus
        full-dimensional noise).
    coefficients:
        Name of the interpolant coefficient preset (see
        :mod:`si.interpolants.coefficients`).  For the coupled construction the paper
        uses ``gamma_t = 0``, i.e. the ``"gamma0"`` preset (the in-painting task uses
        the reversed orientation ``alpha_t = t, beta_t = 1 - t`` -> preset
        ``"inpainting"``).
    """

    #: task name, overridden by subclasses (used for logging / config round-trips)
    name: str = "coupling"

    def __init__(
        self,
        sigma: float = 1.0,
        coefficients: str = "gamma0",
        requires_conditioning: bool = True,
    ) -> None:
        if sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {sigma}")
        self.sigma = float(sigma)
        self.coefficients_name = str(coefficients)
        self.requires_conditioning = bool(requires_conditioning)

    # ------------------------------------------------------------------ #
    # abstract API
    # ------------------------------------------------------------------ #
    @abc.abstractmethod
    def m(self, x1: torch.Tensor, xi: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Corrupted observation ``m(x1)`` (Eq. 18), shape broadcastable to ``x1``."""
        raise NotImplementedError

    @abc.abstractmethod
    def conditioning(self, x1: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Conditional information ``xi`` handed to the model together with ``x_t``."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # concrete API
    # ------------------------------------------------------------------ #
    def sample_xi(self, x1: torch.Tensor, **kwargs: Any) -> Optional[torch.Tensor]:
        """Optional *random* part of the conditional information ``xi``.

        Defaults to ``None`` (``xi`` deterministic given ``x1``).  Random couplings
        (e.g. the random in-painting mask) override this.
        """
        return None

    def build_x0(
        self,
        x1: torch.Tensor,
        xi: Optional[torch.Tensor] = None,
        zeta: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        return_xi: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Draw the coupled base sample ``x0 = m(x1) + sigma zeta``.

        Parameters
        ----------
        x1:
            Target batch, ``(B, C, H, W)`` (or any shape).
        xi:
            Conditional information.  If ``None`` it is obtained from
            :meth:`conditioning` (possibly drawing the random part via
            :meth:`sample_xi`).
        zeta:
            Pre-drawn standard normal noise of shape ``x1.shape``.  Drawn if ``None``.
        generator:
            Optional ``torch.Generator`` used for both ``xi`` (if random) and ``zeta``.
        return_xi:
            If ``False`` only ``x0`` is returned (as the first element of the tuple,
            the second element being ``None``).

        Returns
        -------
        (x0, xi_for_model):
            ``x0`` is the base sample; ``xi_for_model`` is the conditioning tensor that
            must be appended to the model input at every timestep (``None`` when the
            coupling needs no conditioning).
        """
        if xi is None and self.requires_conditioning:
            xi = self.sample_xi(x1, generator=generator)
            if xi is None:
                xi = self.conditioning(x1)
        if zeta is None:
            zeta = torch.randn(
                x1.shape, dtype=x1.dtype, device=x1.device, generator=generator
            )
        x0 = self.m(x1, xi) + self.sigma * zeta
        return x0, (xi if return_xi else None)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def add_noise(
        mean: torch.Tensor,
        sigma: float,
        zeta: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """``mean + sigma * zeta`` with ``zeta ~ N(0, Id)`` (Eq. 18)."""
        if sigma == 0.0:
            return mean
        if zeta is None:
            zeta = torch.randn(
                mean.shape, dtype=mean.dtype, device=mean.device, generator=generator
            )
        return mean + sigma * zeta

    def score_available(self) -> bool:
        """Whether the Gaussian score ``N(x0; m(x1), C)`` is well defined.

        Per Section 3.2 this holds whenever ``sigma`` is invertible.
        """
        return self.sigma > 0.0

    def covariance(self) -> float:
        """``C = sigma sigma^T`` (isotropic convention, ``C = sigma^2 I``)."""
        return self.sigma**2

    def extra_repr(self) -> str:
        return (
            f"sigma={self.sigma}, coefficients='{self.coefficients_name}', "
            f"score_available={self.score_available()}"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}({self.extra_repr()})"


__all__ = ["Coupling"]
