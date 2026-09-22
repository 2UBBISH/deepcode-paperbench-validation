"""Data-dependent coupling for the ImageNet in-painting task (paper Section 4.1).

Recall the general coupling of the paper

    rho(x0, x1) = rho1(x1) rho0(x0 | x1),      x0 = m(x1) + sigma zeta,

which here corresponds to ``rho(x0, x1 | xi) = rho1(x1) rho0(x0 | x1, xi)`` with the
deterministic part

    m(x1, xi) = xi o x1

and the (masked) noise

    x0 = xi o x1 + (1 - xi) o zeta,     zeta ~ N(0, Id)  (independent noise per channel).

``xi in {0,1}^{C x W x H}`` is the conditioning variable: it takes the same value at every
channel of a given spatial location (see ``si.couplings.mask``).  ``xi = 1`` means the pixel is
*observed* (kept from ``x1``) and ``xi = 0`` means it is *missing / in the mask* (initialized with
noise).  During training the mask is drawn by tiling the image into ``num_tiles = 64`` tiles, each
tile entering the mask with probability ``p = 0.3``.

The interpolant used for in-painting is Eq. (20) of the paper with ``alpha_t = t`` and
``beta_t = 1 - t`` (and ``gamma_t = 0``), so that ``I_0 = x0`` is the corrupted image and
``I_1 = x1`` the clean one.  Because ``xi o I_t = xi o x1`` for every ``t``, the velocity field
satisfies ``b_t(x, xi) = 0`` outside of the masked region; ``mask_velocity`` builds that structural
property into the network output (no inference-time correction such as replacement or MCMC is
needed).

The model sees the mask (appended as extra image channels); the partial image is *not* given as
extra conditioning because it is already present, uncorrupted, inside ``x_t`` for every ``t``.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from .base import Coupling
from .mask import RandomTileMask, default_inpainting_mask

__all__ = ["InpaintingCoupling", "inpainting_coupling"]


class InpaintingCoupling(Coupling):
    """Coupling ``x0 = xi o x1 + (1 - xi) o zeta`` for in-painting (Section 4.1).

    Parameters
    ----------
    sigma:
        Noise scale.  The paper uses unit-variance noise inside the masked region, i.e.
        ``sigma = 1`` with the noise restricted to the missing pixels.
    num_tiles:
        Number of tiles the image is split into when drawing a random training mask (64).
    missing_prob:
        Probability that a tile enters the mask (i.e. becomes missing), ``p = 0.3``.
    expand_channels:
        If ``True`` the mask is materialized with ``C`` channels (as the model input
        conditioning requires), otherwise it has a single channel that broadcasts over channels.
    generator:
        Optional ``torch.Generator`` used for all draws (masks and noise).
    coefficients:
        Name of the interpolant coefficient preset used with this coupling.  For in-painting the
        paper sets ``alpha_t = t`` and ``beta_t = 1 - t`` (Section 4.1), which is the
        ``"inpainting"`` preset.
    """

    name = "coupling"

    def __init__(
        self,
        sigma: float = 1.0,
        num_tiles: int = 64,
        missing_prob: float = 0.3,
        expand_channels: bool = False,
        generator: Optional[torch.Generator] = None,
        coefficients: str = "inpainting",
        requires_conditioning: bool = True,
        mask_sampler: Optional[RandomTileMask] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            sigma=sigma,
            coefficients=coefficients,
            requires_conditioning=requires_conditioning,
        )
        self.num_tiles = int(num_tiles)
        self.missing_prob = float(missing_prob)
        self.expand_channels = bool(expand_channels)
        self.generator = generator
        self.mask_sampler = mask_sampler or default_inpainting_mask(
            num_tiles=self.num_tiles,
            missing_prob=self.missing_prob,
            expand_channels=self.expand_channels,
        )
        if self.sigma == 0.0:
            # Without noise the base sample is the observed part of x1 only, which is a
            # degenerate conditional; allowed but flagged.
            self.requires_conditioning = True

    # ------------------------------------------------------------------
    # conditioning / mask
    # ------------------------------------------------------------------
    def sample_xi(self, x1: torch.Tensor, generator: Optional[torch.Generator] = None, **kwargs: Any) -> torch.Tensor:
        """Draw the random tiled missingness mask ``xi`` for ``x1`` (Section 4.1).

        Returns a binary tensor with ``xi = 1`` on observed pixels and ``xi = 0`` on missing ones.
        """
        gen = generator if generator is not None else self.generator
        return self.mask_sampler.sample(x1.shape, generator=gen, device=x1.device)

    def mask(self, x1: torch.Tensor, generator: Optional[torch.Generator] = None, **kwargs: Any) -> torch.Tensor:
        """Alias of :meth:`sample_xi`."""
        return self.sample_xi(x1, generator=generator, **kwargs)

    def conditioning(
        self,
        x1: torch.Tensor,
        xi: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Image-shaped conditioning given to the model: the mask ``xi``.

        The mask is appended as extra channels of the image ``x_t``.  It is expanded over the
        channel dimension when needed so that it can be concatenated directly.
        """
        if xi is None:
            xi = self.sample_xi(x1, generator=generator)
        return self.expand_xi(xi, x1)

    @staticmethod
    def expand_xi(xi: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Broadcast the mask to ``x1``'s number of channels (``xi`` is channel-shared)."""
        if xi.dim() == x1.dim() - 1:  # (B, H, W) -> (B, 1, H, W)
            xi = xi.unsqueeze(1)
        if xi.shape[1] == 1 and x1.shape[1] > 1:
            xi = xi.expand(-1, x1.shape[1], *xi.shape[2:])
        return xi

    # ------------------------------------------------------------------
    # base sample
    # ------------------------------------------------------------------
    def m(self, x1: torch.Tensor, xi: Optional[torch.Tensor] = None, **kwargs: Any) -> torch.Tensor:
        """Deterministic part of the base sample ``m(x1, xi) = xi o x1``."""
        if xi is None:
            raise ValueError("InpaintingCoupling.m requires the mask xi")
        return xi.to(x1.dtype) * x1

    def build_x0(
        self,
        x1: torch.Tensor,
        xi: Optional[torch.Tensor] = None,
        zeta: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        return_xi: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Build the coupled base sample ``x0 = xi o x1 + (1 - xi) o zeta`` (Section 4.1).

        ``zeta ~ N(0, I)`` is drawn with independent noise for each channel of the masked region.
        Returns ``(x0, xi_for_model)`` (the second element is ``None`` when ``return_xi=False``).
        """
        gen = generator if generator is not None else self.generator
        if xi is None:
            xi = self.sample_xi(x1, generator=gen)
        xi_ = self.expand_xi(xi, x1).to(x1.dtype)
        if zeta is None:
            zeta = torch.randn(x1.shape, device=x1.device, dtype=x1.dtype, generator=gen)
        x0 = xi_ * x1 + (1.0 - xi_) * zeta
        return x0, (xi_ if return_xi else None)

    def observed(self, x1: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """The observed (unmasked) part of the image, ``xi o x1``."""
        return self.m(x1, xi)

    def add_noise(
        self,
        mean: torch.Tensor,
        sigma: Optional[float] = None,
        zeta: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Add isotropic Gaussian noise to ``mean`` (SR-style ``m + sigma zeta`` convention)."""
        s = self.sigma if sigma is None else float(sigma)
        if s == 0.0:
            return mean
        if zeta is None:
            zeta = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
        return mean + s * zeta

    # ------------------------------------------------------------------
    # structural masking of the velocity field
    # ------------------------------------------------------------------
    @staticmethod
    def mask_velocity(v: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """Zero the predicted velocity on the *observed* pixels.

        Since ``xi o I_t = xi o x1`` for every ``t``, ``I_dot`` (and hence ``b_t``) vanishes where
        ``xi = 1``: the unmasked pixels stay fixed and only the masked region is regenerated.
        """
        if xi.dim() == v.dim() - 1:
            xi = xi.unsqueeze(1)
        keep = 1.0 - xi.to(v.dtype)
        return v * keep

    @staticmethod
    def masked_region(xi: torch.Tensor) -> torch.Tensor:
        """Indicator of the masked (missing) region, ``1 - xi``."""
        return 1.0 - xi

    # ------------------------------------------------------------------
    # misc
    # ------------------------------------------------------------------
    def score_available(self) -> bool:
        """The score is available only when the noise covariance is invertible."""
        return self.sigma > 0.0

    def extra_repr(self) -> str:
        return (
            f"num_tiles={self.num_tiles}, missing_prob={self.missing_prob}, "
            f"expand_channels={self.expand_channels}"
        )


def inpainting_coupling(
    sigma: float = 1.0,
    num_tiles: int = 64,
    missing_prob: float = 0.3,
    **kwargs: Any,
) -> InpaintingCoupling:
    """Factory for the paper's in-painting coupling (64 tiles, ``p = 0.3``)."""
    return InpaintingCoupling(
        sigma=sigma,
        num_tiles=num_tiles,
        missing_prob=missing_prob,
        **kwargs,
    )


# ----------------------------------------------------------------------
# self test
# ----------------------------------------------------------------------
def _self_test() -> None:
    torch.manual_seed(0)
    coupling = InpaintingCoupling()
    x1 = torch.randn(4, 3, 256, 256)

    x0, xi = coupling.build_x0(x1)
    assert x0.shape == x1.shape, (x0.shape, x1.shape)
    assert xi.shape in {(4, 1, 256, 256), (4, 3, 256, 256)}, xi.shape

    # observed pixels are copied exactly, masked pixels carry the noise
    observed = xi.bool().expand_as(x1)
    assert torch.allclose(x0[observed], x1[observed]), "observed pixels must be preserved"
    assert not torch.allclose(x0[~observed], x1[~observed]), "masked pixels must be re-initialized"

    # the velocity vanishes on observed pixels and is untouched in the mask
    v = torch.randn_like(x1)
    v_masked = coupling.mask_velocity(v, xi)
    assert torch.allclose(v_masked[observed], torch.zeros_like(v_masked[observed]))
    assert torch.allclose(v_masked[~observed], v[~observed])

    # conditioning is the mask, expanded to the image channels
    cond = coupling.conditioning(x1, xi)
    assert cond.shape == x1.shape, cond.shape

    # default interpolant preset follows Section 4.1: alpha_t = t, beta_t = 1 - t, gamma_t = 0
    from ..interpolants.coefficients import get_coefficients, inpainting as inpainting_coeffs

    coeffs = get_coefficients(coupling.coefficients_name)
    assert coeffs.name == inpainting_coeffs.name == "inpainting"
    t = torch.tensor([0.0, 0.5, 1.0])
    a, b, g = coeffs.evaluate(t)
    assert torch.allclose(a, t)
    assert torch.allclose(b, 1.0 - t)
    assert torch.allclose(g, torch.zeros_like(t))

    # empirical missing fraction should be close to p = 0.3 (64 tiles, p = 0.3)
    big = torch.randn(16, 3, 256, 256)
    missing = 1.0 - coupling.sample_xi(big).float()
    frac = missing.mean().item()
    assert abs(frac - 0.3) < 0.08, frac
    print(f"[inpainting] self-test passed (missing fraction = {frac:.3f})")


if __name__ == "__main__":
    _self_test()
