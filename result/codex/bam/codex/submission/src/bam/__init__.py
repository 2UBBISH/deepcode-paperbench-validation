"""BaM: batch and match black-box variational inference with a score-based divergence.

Reference implementation accompanying the reproduction of

    Cai, Modi, Pillaud-Vivien, Margossian, Gower, Blei & Saul,
    "Batch and match: black-box variational inference with a score-based
    divergence", ICML 2024.

The package is organized as follows::

    bam.linalg      quadratic matrix equations / PSD helpers (Appendix B)
    bam.divergence  score-based divergence and KL estimators (Section 2, Appendix A)
    bam.targets     target distributions p (Gaussian, sinh-arcsinh, posteriors, VAE)
    bam.bam         the BaM algorithm (Algorithm 1)
    bam.baselines   ADVI / Score / Fisher (Algorithm 2) and GSM (Algorithm 3)
    bam.metrics     KL divergences, relative mean/SD errors, reconstruction MSE
    bam.adam        a small dependency-free Adam optimizer used by the baselines
"""

import jax

# the experiments need double precision (BaM's covariance updates are ill
# conditioned when run in float32)
jax.config.update("jax_enable_x64", True)

__all__ = ["linalg", "divergence", "targets", "bam", "baselines", "metrics", "adam"]
__version__ = "1.0.0"
