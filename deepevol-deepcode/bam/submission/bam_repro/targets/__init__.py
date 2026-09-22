"""Target distributions for the BaM reproduction (``bam_repro.targets``).

This package aggregates the four families of target distributions used in the
paper *Batch and Match: Black-Box Variational Inference with a Score-Based
Divergence*:

* :mod:`~bam_repro.targets.gaussian_target` -- synthetic full-covariance
  Gaussian targets ``p = N(mu*, Sigma*)`` with ``Sigma* = A A^T``
  (Section 5.1 / Appendix E.3, D = 4, 16, 64, 256).  Analytic score and
  closed-form forward/reverse KL.
* :mod:`~bam_repro.targets.sinh_arcsinh` -- the non-Gaussian sinh-arcsinh
  normal target (Section 5.1 / Appendix E.4, D = 10) with skew ``s`` and
  tail-weight ``tau``.
* :mod:`~bam_repro.targets.posteriordb_target` -- three hierarchical
  posteriors from posteriorDB (``ark`` D=7, ``gp-pois-regr`` D=13,
  ``eight-schools-centered`` D=10) backed by BridgeStan when available
  (Section 5.2 / Appendix E.5).
* :mod:`~bam_repro.targets.vae_target` -- the CIFAR-10 deep generative model
  target ``p(z' | x')`` with a pre-trained convolutional decoder
  (Section 5.3 / Appendix E.6).

Every target exposes a common minimal interface consumed by the BaM optimizer
(:mod:`bam_repro.bam`) and the baselines (:mod:`bam_repro.baselines`)::

    z = target.sample(n)          # draws from the target
    g = target.score(z)           # analytic gradient of log p at z
    lp = target.log_prob(z)       # log density up to a constant

The optional heavy dependencies (BridgeStan / posteriorDB cache, CIFAR-10
data) are imported in a guarded fashion so that ``import bam_repro.targets``
never fails on a machine that only needs the synthetic experiments.
"""

from __future__ import annotations

from typing import List

__version__ = "0.1.0"

__all__: List[str] = []


# ---------------------------------------------------------------------------
# Gaussian synthetic target (Section 5.1 / E.3) -- always available.
# ---------------------------------------------------------------------------
from .gaussian_target import (  # noqa: E402
    GaussianTarget,
    gaussian_forward_kl,
    gaussian_reverse_kl,
    gaussian_score_divergence,
    one_step_recovery,
    random_gaussian_A,
    random_gaussian_target,
)

__all__ += [
    "GaussianTarget",
    "random_gaussian_target",
    "random_gaussian_A",
    "gaussian_forward_kl",
    "gaussian_reverse_kl",
    "gaussian_score_divergence",
    "one_step_recovery",
]

try:  # pragma: no cover - name tolerances across module revisions
    from .gaussian_target import gaussian_kl as _gaussian_kl_impl  # noqa: F401
except ImportError:  # pragma: no cover
    _gaussian_kl_impl = None


# ---------------------------------------------------------------------------
# sinh-arcsinh non-Gaussian target (Section 5.1 / E.4) -- always available.
# ---------------------------------------------------------------------------
from .sinh_arcsinh import (  # noqa: E402
    PAPER_DIM as SINH_ARCSINH_DIM,
    PAPER_SKEW_VALUES,
    PAPER_TAIL_VALUES,
    SinhArcsinhTarget,
    paper_sweep_targets,
    random_sinh_arcsinh_target,
    sinh_arcsinh_forward,
    sinh_arcsinh_inverse,
    sinh_arcsinh_score,
    sinh_arcsinh_target,
)

__all__ += [
    "SinhArcsinhTarget",
    "sinh_arcsinh_target",
    "random_sinh_arcsinh_target",
    "paper_sweep_targets",
    "sinh_arcsinh_forward",
    "sinh_arcsinh_inverse",
    "sinh_arcsinh_score",
    "PAPER_SKEW_VALUES",
    "PAPER_TAIL_VALUES",
    "SINH_ARCSINH_DIM",
]


# ---------------------------------------------------------------------------
# posteriorDB targets (Section 5.2 / E.5) -- optional BridgeStan backend.
# ---------------------------------------------------------------------------
try:
    from .posteriordb_target import (  # noqa: E402
        BRIDGESTAN_AVAILABLE,
        PAPER_INIT_MEAN_SCALE,
        PAPER_POSTERIORDB_BATCH_SIZES,
        PAPER_POSTERIORDB_MODELS,
        PAPER_POSTERIORDB_NAMES,
        PAPER_POSTERIORDB_N_RUNS,
        POSTERIORDB_AVAILABLE,
        PosteriorDBTarget,
        StanTarget,
        SurrogateGaussianTarget,
        bridgestan_available,
        get_target,
        load_posteriordb_target,
        make_target,
        posteriordb_available,
        posteriordb_targets,
    )

    __all__ += [
        "PosteriorDBTarget",
        "StanTarget",
        "SurrogateGaussianTarget",
        "load_posteriordb_target",
        "get_target",
        "make_target",
        "posteriordb_targets",
        "bridgestan_available",
        "posteriordb_available",
        "BRIDGESTAN_AVAILABLE",
        "POSTERIORDB_AVAILABLE",
        "PAPER_POSTERIORDB_MODELS",
        "PAPER_POSTERIORDB_NAMES",
        "PAPER_POSTERIORDB_BATCH_SIZES",
        "PAPER_POSTERIORDB_N_RUNS",
        "PAPER_INIT_MEAN_SCALE",
    ]
except Exception as _exc:  # pragma: no cover - optional backend
    import warnings as _warnings

    _warnings.warn(
        f"bam_repro.targets: posteriorDB targets unavailable ({_exc!r}); "
        "the synthetic experiments remain fully functional.",
        RuntimeWarning,
        stacklevel=2,
    )


# ---------------------------------------------------------------------------
# VAE / CIFAR-10 target (Section 5.3 / E.6) -- optional heavy module.
# ---------------------------------------------------------------------------
try:
    from .vae_target import (  # noqa: E402
        PAPER_VAE_ADVI_LR,
        PAPER_VAE_BAM_LAMBDAS,
        PAPER_VAE_BATCH_SIZES,
        PAPER_VAE_GRAD_BUDGET,
        PAPER_VAE_PILOT_T,
        PAPER_VAE_T,
        PAPER_VAE_WALLCLOCK_TARGET_BATCH,
        VAE,
        VAE_IMAGE_DIM,
        VAE_IMAGE_SHAPE,
        VAE_LATENT_DIM,
        VAE_SIGMA2,
        Decoder,
        ImageData,
        VAEPosteriorTarget,
        VAETarget,
        amortized_reconstruction_mse,
        load_cifar10,
        load_vae,
        save_vae,
        score_function,
        synthetic_image_data,
        train_vae,
        vae_target_from_image,
    )

    __all__ += [
        "VAETarget",
        "VAEPosteriorTarget",
        "VAE",
        "Decoder",
        "ImageData",
        "vae_target_from_image",
        "score_function",
        "train_vae",
        "save_vae",
        "load_vae",
        "load_cifar10",
        "synthetic_image_data",
        "amortized_reconstruction_mse",
        "VAE_LATENT_DIM",
        "VAE_IMAGE_DIM",
        "VAE_IMAGE_SHAPE",
        "VAE_SIGMA2",
        "PAPER_VAE_PILOT_T",
        "PAPER_VAE_T",
        "PAPER_VAE_BATCH_SIZES",
        "PAPER_VAE_ADVI_LR",
        "PAPER_VAE_BAM_LAMBDAS",
        "PAPER_VAE_GRAD_BUDGET",
        "PAPER_VAE_WALLCLOCK_TARGET_BATCH",
    ]
except Exception as _exc:  # pragma: no cover - optional heavy module
    import warnings as _warnings

    _warnings.warn(
        f"bam_repro.targets: VAE target unavailable ({_exc!r}); "
        "the synthetic and posteriorDB experiments remain functional.",
        RuntimeWarning,
        stacklevel=2,
    )


def available_targets() -> List[str]:
    """Return the names of the target families importable in this environment."""
    names = ["gaussian", "sinh-arcsinh"]
    if "PosteriorDBTarget" in globals():
        names.append("posteriordb")
    if "VAETarget" in globals():
        names.append("vae")
    return names
