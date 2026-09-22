"""Precision aware perturbation handling for the adversarial evaluations.

The paper evaluates every attack *twice*: first an inexpensive half-precision
attack, then a single-precision attack on the samples that survived the first
stage (Sec. 4.1, App. B.6).  A perturbation of size ``eps = k / 255`` is only
meaningful together with the numeric precision in which the perturbed image is
fed to the model:

* half precision  -> the image (and hence the perturbation) is stored as 16-bit
  values, so ``eps`` is a *16-bit integer* number of grid steps,
* single precision -> the same perturbation is represented with 32-bit values.

The addendum of the paper states this explicitly: *"For half-precision attacks,
16-bit ints need to be used, and for single-precision attacks, 32-bit ints need
to be used."*

:class:`PerturbationGrid` implements exactly this book keeping.  Perturbations
are generated as integers in ``[-eps_int, eps_int]``, scaled by ``1/255`` and
snapped onto the grid of the target dtype (``torch.float16`` or
``torch.float32``), and every forward pass of an attack is performed on the
snapped image.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

#: eps and the attack step size are always expressed as multiples of 1/255.
EPS_DENOMINATOR = 255


def eps_to_int(eps: Union[str, float, int]) -> int:
    """``2/255`` -> ``2``.

    Accepts floats as well (``2 / 255`` -> ``2``).
    """
    if isinstance(eps, str):
        if "/" in eps:
            num, denom = eps.split("/")
            return int(round(float(num)))
        eps = float(eps)
    return int(round(float(eps) * EPS_DENOMINATOR))


def int_to_eps(eps_int: int) -> float:
    """``2`` -> ``2/255`` (returned as a float)."""
    return float(eps_int) / EPS_DENOMINATOR


def snap_to_dtype(x: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Snap ``x`` onto the grid of ``dtype`` and return it as a float32 tensor.

    This is the operation that turns "the attack runs at half precision" into an
    actual (integer valued) perturbation: casting a float32 image in ``[0, 1]``
    to ``torch.float16`` quantizes it to the 16-bit grid, casting it back to
    float32 keeps the quantized values.  All attacks call ``snap_to_dtype`` after
    every update so that the perturbation never leaves the grid of the precision
    the attack is run in.
    """
    if dtype == torch.float32:
        return x
    return x.to(dtype).to(torch.float32)


class PerturbationGrid:
    """Book keeping for an ``l_inf`` ball of radius ``eps_int / 255`` in a given precision.

    Parameters
    ----------
    eps_int:
        radius expressed in units of ``1/255`` (e.g. ``2`` for ``2/255``).
    dtype:
        precision of the attack (``torch.float16`` -> 16-bit ints,
        ``torch.float32`` -> 32-bit ints).
    alpha_int:
        step size in units of ``1/255`` (defaults to ``eps_int``, which is what
        App. B.6 uses for APGD).
    x_min, x_max:
        valid range of the *non-normalized* input.  The l_inf ball is always
        computed around the non-normalized image, normalization (if any) is
        applied afterwards by the model wrapper.
    """

    def __init__(
        self,
        eps_int: int,
        dtype: torch.dtype = torch.float32,
        alpha_int: Optional[int] = None,
        x_min: float = 0.0,
        x_max: float = 1.0,
    ):
        self.eps_int = int(eps_int)
        self.dtype = dtype
        self.alpha_int = int(alpha_int) if alpha_int is not None else int(eps_int)
        self.x_min = float(x_min)
        self.x_max = float(x_max)

    # -- properties -----------------------------------------------------
    @property
    def eps(self) -> float:
        return int_to_eps(self.eps_int)

    @property
    def alpha(self) -> float:
        return int_to_eps(self.alpha_int)

    # -- perturbations --------------------------------------------------
    def snap(self, x: torch.Tensor) -> torch.Tensor:
        """Snap an (adversarial) image onto the grid of ``self.dtype``."""
        return snap_to_dtype(x, self.dtype)

    def random_start(self, x: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Uniformly sampled perturbation inside the ball.

        The perturbation is sampled on the integer grid ``[-eps_int, eps_int]``,
        i.e. exactly the initialization used in the paper ("initialization with
        uniform random perturbation", see addendum / App. B.6).
        """
        shape = x.shape
        device = x.device
        ints = torch.randint(
            low=-self.eps_int,
            high=self.eps_int + 1,
            size=shape,
            device=device,
            generator=generator,
            dtype=torch.int32,
        )
        delta = ints.to(torch.float32) / EPS_DENOMINATOR
        x_adv = self.clean(x) + delta
        return self.project(x_adv, x)

    def project(self, x_adv: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Project ``x_adv`` onto ``{z : ||z - snap(x)||_inf <= eps} ∩ [x_min, x_max]``.

        The reference point of the ball is the *clean image in the precision of the
        attack*: for a half-precision attack the clean image is quantized to
        16-bit values as well, so the perturbation remains a 16-bit integer
        multiple of ``1/255`` (see the addendum).

        Note that the result is snapped to the grid of the precision, therefore in
        *half precision* the projection is exact up to half an ulp of the 16-bit
        grid (``2**-12`` near one), which is the granularity at which a
        half-precision attack can move the pixels at all.
        """
        x_ref = self.clean(x)
        x_adv = torch.min(torch.max(x_adv, x_ref - self.eps), x_ref + self.eps)
        x_adv = torch.clamp(x_adv, self.x_min, self.x_max)
        return self.snap(x_adv)

    def clean(self, x: torch.Tensor) -> torch.Tensor:
        """The clean image as seen by an attack of this precision."""
        return self.snap(x).clamp(self.x_min, self.x_max)

    def quantize_delta(self, delta: torch.Tensor) -> torch.Tensor:
        """Snap a raw perturbation onto the integer grid of the precision."""
        delta = delta.to(self.dtype).to(torch.float32)
        return torch.clamp(delta, -self.eps, self.eps)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"PerturbationGrid(eps={self.eps_int}/255, alpha={self.alpha_int}/255, "
            f"dtype={self.dtype}, range=[{self.x_min}, {self.x_max}])"
        )
