"""DPMs-ANT: Adapting Pretrained Diffusion Models for Few-Shot Image Generation.

This package reproduces the paper "Adapting Pretrained Diffusion Models for
Few-Shot Image Generation" (DPMs-ANT), which adapts a *frozen* pretrained
diffusion model (DDPM 256x256 or LDM 64x64) to a new target domain using only
~10 training images by inserting zero-initialized *adaptors* into the U-Net and
training them with an adversarial-noise similarity-guided objective
(Algorithm 1, Eq. 8).

Top level convenience re-exports keep the import surface small.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
