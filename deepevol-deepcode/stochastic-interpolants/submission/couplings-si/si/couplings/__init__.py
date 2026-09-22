"""Data-dependent couplings for stochastic interpolants.

This package implements the coupling ``rho(x0, x1) = rho1(x1) rho0(x0 | x1)``
introduced in *Stochastic Interpolants with Data-Dependent Couplings*.  The base
sample is built conditionally from the target as

    x0 = m(x1) + sigma * zeta,      zeta ~ N(0, I),

and the conditioning ``xi`` (the mask for in-painting, the up-sampled low
resolution image for super-resolution) is handed to the velocity network.

Concrete couplings
------------------
``InpaintingCoupling``  -- Section 4.1:  ``m(x1) = xi * x1`` with a random tiled
                           missingness mask ``xi``.
``SuperresCoupling``    -- Section 4.2:  ``m(x1) = U(D(x1))`` and
                           ``xi = U(D(x1))``.

Utilities
---------
``RandomTileMask`` / ``random_tile_mask`` -- 64-tile missingness masks (p=0.3).
``ResizePair`` / ``downsample`` / ``upsample`` -- the ``D`` and ``U`` operators.
"""

from __future__ import annotations

from .base import Coupling
from .inpainting import InpaintingCoupling, inpainting_coupling
from .mask import (
    RandomTileMask,
    count_missing,
    default_inpainting_mask,
    missing_fraction,
    random_tile_mask,
    tile_grid,
)
from .resize import (
    Downsample,
    ResizePair,
    Upsample,
    default_superres_pair,
    downsample,
    low_resolution,
    normalize_size,
    upsample,
)
from .superres import SuperresCoupling, superres_coupling

__all__ = [
    # base interface
    "Coupling",
    # in-painting
    "InpaintingCoupling",
    "inpainting_coupling",
    # super-resolution
    "SuperresCoupling",
    "superres_coupling",
    # mask utilities
    "RandomTileMask",
    "random_tile_mask",
    "default_inpainting_mask",
    "tile_grid",
    "count_missing",
    "missing_fraction",
    # resize utilities
    "ResizePair",
    "Downsample",
    "Upsample",
    "downsample",
    "upsample",
    "low_resolution",
    "normalize_size",
    "default_superres_pair",
]

# Convenient registry mapping a task name to the class that implements its
# coupling.  Used by ``train.py`` / ``sample.py`` to build couplings from a
# config without importing each module explicitly.
COUPLINGS = {
    "inpainting": InpaintingCoupling,
    "superres": SuperresCoupling,
    "superresolution": SuperresCoupling,
}


def get_coupling(name: str, **kwargs) -> Coupling:
    """Build a coupling by task name.

    Parameters
    ----------
    name : str
        One of ``"inpainting"``, ``"superres"`` (alias ``"superresolution"``).
    **kwargs
        Forwarded to the coupling constructor (``sigma``, ``num_tiles``,
        ``missing_prob``, ``low_res``, ...).

    Returns
    -------
    Coupling
    """
    key = str(name).lower()
    if key not in COUPLINGS:
        raise ValueError(
            f"Unknown coupling '{name}'. Available: {sorted(COUPLINGS)}"
        )
    return COUPLINGS[key](**kwargs)
