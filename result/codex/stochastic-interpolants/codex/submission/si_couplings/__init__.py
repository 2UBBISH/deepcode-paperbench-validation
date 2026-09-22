"""Stochastic interpolants with data-dependent couplings.

Reference implementation of

    M. S. Albergo*, M. Goldstein*, N. M. Boffi, R. Ranganath, E. Vanden-Eijnden,
    "Stochastic Interpolants with Data-Dependent Couplings", ICML 2024.

The package is organised along the paper:

``interpolants``  Definition 3.1 / A.1: I_t = alpha_t x_0 + beta_t x_1 + gamma_t z
``couplings``     Section 3.2 / 3.3 and the task couplings of Section 4
``losses``        Theorem 3.1 / A.1: the objectives L_b and L_g
``solvers``       Corollary 3.1: probability-flow ODE and the forward/backward SDEs
``models``        Appendix B: DDPM U-Net velocity model, 2-D toy networks
``train``         Algorithm 1
``sample``        Algorithm 2
``fid``           Tables 2 and 3: FID-50k
"""

from .couplings import (
    CoupledBatch,
    Coupling,
    DataDecorruptionCoupling,
    GaussianAdaptedCoupling,
    IndependentCoupling,
    InpaintingCoupling,
    SuperResolutionCoupling,
    build_coupling,
    tile_mask,
)
from .interpolants import (
    Coeffs,
    GaussianCoupling,
    InterpolantSchedule,
    LinearInterpolant,
    VPInterpolant,
    build_interpolant,
)
from .losses import (
    score_loss,
    transport_cost_upper_bound,
    velocity_loss,
)
from .solvers import (
    backward_sde_sample,
    forward_sde_sample,
    odeint,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "Coeffs",
    "GaussianCoupling",
    "InterpolantSchedule",
    "LinearInterpolant",
    "VPInterpolant",
    "build_interpolant",
    "CoupledBatch",
    "Coupling",
    "DataDecorruptionCoupling",
    "GaussianAdaptedCoupling",
    "IndependentCoupling",
    "InpaintingCoupling",
    "SuperResolutionCoupling",
    "build_coupling",
    "tile_mask",
    "velocity_loss",
    "score_loss",
    "transport_cost_upper_bound",
    "odeint",
    "forward_sde_sample",
    "backward_sde_sample",
]
