r"""Super-resolution data-dependent coupling (Section 4.2 of the paper).

The paper defines image down/up-sampling operators

.. math::

    \mathcal{D}: \mathbb{R}^{C \times W \times H}
        \to \mathbb{R}^{C \times W_{\mathrm{low}} \times H_{\mathrm{low}}},
    \qquad
    \mathcal{U}: \mathbb{R}^{C \times W_{\mathrm{low}} \times H_{\mathrm{low}}}
        \to \mathbb{R}^{C \times W \times H},

and builds the *base* sample of the stochastic interpolant as

.. math::

    x_0 = \mathcal{U}(\mathcal{D}(x_1)) + \sigma \zeta,
    \qquad \zeta \sim \mathrm{N}(0, Id), \quad \sigma > 0.

Notice that with :math:`\sigma = 0` the base density would be concentrated on a
lower-dimensional manifold; a strictly positive :math:`\sigma` smooths it so that
it is well defined over the whole ambient space.

The model additionally receives the low-resolution image at *all* times, i.e. the
problem is the coupled one

.. math::

    \rho(x_0, x_1 \mid \xi) = \rho_1(x_1) \, \rho_0(x_0 \mid x_1, \xi),
    \qquad \xi = \mathcal{U}(\mathcal{D}(x_1)).

The returned conditioning tensor :math:`\xi` (the upsampled low-resolution image)
is appended to the channel dimension of the velocity-network input.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from .base import Coupling
from .resize import ResizePair, default_superres_pair, downsample, upsample

__all__ = ["SuperresCoupling", "superres_coupling"]


class SuperresCoupling(Coupling):
    """Super-resolution coupling: ``x0 = U(D(x1)) + sigma * zeta``.

    Parameters
    ----------
    low_res:
        Spatial size of the low-resolution image (an int such as ``64`` applies to
        both width and height, a sequence gives ``(W_low, H_low)``).
    sigma:
        Standard deviation of the Gaussian noise added to the upsampled
        low-resolution image.  The paper requires ``sigma > 0``; it does not
        specify the value, so a small positive default is used.
    down_mode, up_mode, antialias:
        Interpolation modes used by the :math:`\\mathcal{D}` / :math:`\\mathcal{U}`
        operators (see :mod:`si.couplings.resize`).  ``D`` uses area interpolation
        by default, ``U`` bilinear interpolation.
    coefficients:
        Interpolant coefficient preset (paper's coupled-base experiments use the
        ``gamma_t = 0`` preset).
    requires_conditioning:
        Whether the model is conditioned on :math:`\\xi` (always true for the
        paper's super-resolution task, where the low-resolution image is appended
        to the network input at every time).
    """

    name = "superres"

    def __init__(
        self,
        low_res: Optional[int] = 64,
        sigma: float = 0.05,
        down_mode: str = "area",
        up_mode: str = "bilinear",
        antialias: bool = True,
        coefficients: str = "gamma0",
        requires_conditioning: bool = True,
        resize_pair: Optional[ResizePair] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            sigma=sigma,
            coefficients=coefficients,
            requires_conditioning=requires_conditioning,
        )
        if low_res is None and resize_pair is None:
            raise ValueError(
                "SuperresCoupling needs either `low_res` or an explicit `resize_pair`."
            )
        if resize_pair is None:
            resize_pair = default_superres_pair(
                low_res,
                down_mode=down_mode,
                up_mode=up_mode,
                antialias=antialias,
            )
        self.resize_pair = resize_pair
        self.low_res = resize_pair.low_res
        self.down_mode = getattr(resize_pair, "down_mode", down_mode)
        self.up_mode = getattr(resize_pair, "up_mode", up_mode)

    # ------------------------------------------------------------------
    # Resize operators
    # ------------------------------------------------------------------
    def D(self, x1: torch.Tensor, low_res: Optional[int] = None) -> torch.Tensor:
        """Downsample a high-resolution image: :math:`\\mathcal{D}(x_1)`."""
        return self.resize_pair.D(x1, low_res=low_res)

    def U(self, low: torch.Tensor, high_res: Optional[int] = None) -> torch.Tensor:
        """Upsample a low-resolution image: :math:`\\mathcal{U}(x_{low})`."""
        return self.resize_pair.U(low, high_res=high_res)

    def UD(self, x1: torch.Tensor, low_res: Optional[int] = None) -> torch.Tensor:
        """The low-resolution content embedded in high-resolution space."""
        return self.resize_pair.UD(x1, low_res=low_res)

    # ------------------------------------------------------------------
    # Coupling interface
    # ------------------------------------------------------------------
    def m(self, x1: torch.Tensor, xi: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Deterministic base mean :math:`m(x_1) = \mathcal{U}(\mathcal{D}(x_1))`.

        ``xi`` is accepted for interface compatibility with the other couplings; for
        super-resolution the conditioning *is* the mean, so it is simply returned
        when supplied (it must already be the upsampled low-resolution image).
        """
        if xi is not None:
            return xi
        return self.UD(x1)

    def conditioning(self, x1: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        r"""Model conditioning :math:`\xi = \mathcal{U}(\mathcal{D}(x_1))`."""
        low_res = kwargs.get("low_res", None)
        return self.UD(x1, low_res=low_res)

    def sample_xi(self, x1: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """For super-resolution the conditioning is deterministic given ``x1``."""
        return self.conditioning(x1, **kwargs)

    def build_x0(
        self,
        x1: torch.Tensor,
        xi: Optional[torch.Tensor] = None,
        zeta: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        return_xi: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build ``x0 = U(D(x1)) + sigma * zeta`` and its conditioning ``xi``."""
        if xi is None:
            xi = self.conditioning(x1)
        if x1.shape != xi.shape:
            # ``xi`` may be given at the low resolution; upsample it to x1's shape.
            xi = self.U(xi, high_res=x1.shape[-2:])
        return super().build_x0(
            x1,
            xi=xi,
            zeta=zeta,
            generator=generator,
            return_xi=return_xi,
        )

    def observed(self, x1: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Conditioning available *before* any noise is added (same as ``xi``)."""
        return self.conditioning(x1, **kwargs)

    # ------------------------------------------------------------------
    # Model-side helpers
    # ------------------------------------------------------------------
    def model_input(self, x_t: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """Append the conditioning ``xi`` to the channels of ``x_t``.

        The paper appends the upsampled low-resolution image to the channel
        dimension of the velocity model input at *all* times.
        """
        if xi.shape[-2:] != x_t.shape[-2:]:
            xi = self.U(xi, high_res=x_t.shape[-2:])
        return torch.cat([x_t, xi], dim=1)

    def in_channels(self, channels: int) -> int:
        """Number of input channels of the velocity net for a ``channels`` image."""
        return channels + channels

    def extra_repr(self) -> str:
        return (
            f"low_res={tuple(self.low_res)}, down_mode={self.down_mode!r}, "
            f"up_mode={self.up_mode!r}"
        )


def superres_coupling(
    low_res: int = 64,
    sigma: float = 0.05,
    **kwargs: Any,
) -> SuperresCoupling:
    """Factory for the default paper super-resolution coupling (64 -> 256/512)."""
    return SuperresCoupling(low_res=low_res, sigma=sigma, **kwargs)


def _self_test() -> None:
    torch.manual_seed(0)

    for high_res, low_res in ((256, 64), (512, 256), (256, 64)):
        coupling = SuperresCoupling(low_res=low_res, sigma=0.05)
        x1 = torch.randn(2, 3, high_res, high_res)

        # D / U shapes
        low = coupling.D(x1)
        assert low.shape == (2, 3, low_res, low_res), low.shape
        xi = coupling.conditioning(x1)
        assert xi.shape == x1.shape, xi.shape

        # Base sample: x0 = U(D(x1)) + sigma zeta
        x0, xi_out = coupling.build_x0(x1)
        assert x0.shape == x1.shape
        assert torch.allclose(xi_out, xi)
        residual = x0 - xi
        # Noise should be small but non-zero (sigma > 0), and deterministic part exact.
        assert residual.abs().mean() > 0
        assert residual.std().item() == pytest_approx(0.05, 0.02)

        # With sigma = 0 the base sample equals the upsampled low-res image.
        det = SuperresCoupling(low_res=low_res, sigma=0.0)
        x0_det, _ = det.build_x0(x1)
        assert torch.allclose(x0_det, xi)

        # Conditioning appended to the model input on a separate channel block.
        x_t = torch.randn_like(x1)
        net_in = coupling.model_input(x_t, xi)
        assert net_in.shape == (2, 6, high_res, high_res)
        assert torch.allclose(net_in[:, :3], x_t)
        assert torch.allclose(net_in[:, 3:], xi)

    print("superres coupling self-test passed")


def pytest_approx(value: float, tol: float) -> bool:
    """Tiny helper so the self-test needs no pytest dependency."""
    class _Approx:
        def __eq__(self, other: float) -> bool:  # type: ignore[override]
            return abs(float(other) - value) <= tol

    return _Approx()  # type: ignore[return-value]


if __name__ == "__main__":
    _self_test()
