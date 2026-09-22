"""Latent diffusion (LDM) backend.

Section 5.2: "We evaluate our method not only on the DDPM framework but also in
LDM.  For this, we employ a pre-trained DDPM similar to DDPM-PA and use
pre-trained LDMs as provided in (Rombach et al., 2022).  We restrict the
fine-tuning to the shift module of the U-Net, maintaining the pre-trained DPMs
and autoencoders in LDMs as they are."

Everything in :mod:`dpms_ant` (adaptors, similarity guidance, adversarial noise
selection, Algorithm 1) is written against an abstract ``eps_theta``; this
module provides the LDM instantiation of that interface:

* :class:`LDMBackend` -- a diffusers ``UNet2DConditionModel``/``UNet2DModel``
  together with the KL/VQ autoencoder, exposing ``encode``/``decode``/forward
  in *latent* space (the adaptors are attached to the latent U-Net).
* :class:`PixelSpaceClassifier` -- makes the *image* domain classifier
  applicable to noised latents, by decoding the latent and differentiating
  through the (frozen) decoder.  The paper's classifier is trained on noised
  *images* (Section 5.5), so this keeps the similarity measure in pixel space.

The LDM schedule is the "scaled linear" one used by Rombach et al.
(``beta`` linear in ``sqrt(beta)``, ``beta_start=0.0015``, ``beta_end=0.0195``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .schedules import DiffusionSchedule


@dataclass
class LDMConfig:
    """Configuration of the LDM backbone."""

    model_id: str = "CompVis/ldm-celebahq-256"  # any diffusers-format LDM
    vae_id: Optional[str] = None                # defaults to the model's VAE
    image_size: int = 256
    latent_channels: int = 4
    scaling_factor: Optional[float] = None      # taken from the VAE config
    num_timesteps: int = 1000
    beta_start: float = 0.0015
    beta_end: float = 0.0195
    schedule: str = "scaled_linear"
    sample_mean: bool = True                    # deterministic encoding


class LDMBackend(nn.Module):
    """``eps_theta`` in latent space plus its autoencoder."""

    def __init__(
        self,
        unet: nn.Module,
        vae: Optional[nn.Module] = None,
        scaling_factor: float = 0.18215,
        latent_channels: int = 4,
        sample_mean: bool = True,
    ):
        super().__init__()
        self.unet = unet
        self.vae = vae
        self.scaling_factor = float(scaling_factor)
        self.latent_channels = latent_channels
        self.sample_mean = sample_mean
        if self.vae is not None:
            for parameter in self.vae.parameters():
                parameter.requires_grad_(False)
            self.vae.eval()

    # ------------------------------------------------------------------ #
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Images in ``[-1, 1]`` -> scaled latents."""
        assert self.vae is not None, "the backend has no autoencoder"
        posterior = self.vae.encode(images)
        latent = posterior.latent_dist.mean if self.sample_mean else posterior.latent_dist.sample()
        return latent * self.scaling_factor

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        assert self.vae is not None, "the backend has no autoencoder"
        return self.vae.decode(latents / self.scaling_factor).sample

    def forward(self, latents: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Epsilon prediction in latent space (the U-Net's own interface)."""
        output = self.unet(latents, timesteps)
        return output.sample if hasattr(output, "sample") else output


class PixelSpaceClassifier(nn.Module):
    """Applies an *image* classifier to noised latents.

    ``p_phi(y | x_t)`` of the paper is defined on noised images.  For LDMs the
    diffusion variable is a latent, so the latent is decoded first and the
    gradient of the classifier log-probability is taken through the decoder
    (both the decoder and the classifier stay frozen).
    """

    def __init__(self, classifier: nn.Module, backend: LDMBackend):
        super().__init__()
        self.classifier = classifier
        self.backend = backend

    def forward(self, latents: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            images = self.backend.decode(latents)
        return self.classifier(images, timesteps)


def build_ldm(
    config: Optional[LDMConfig] = None,
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[LDMBackend, DiffusionSchedule]:
    """Load a pre-trained LDM (diffusers format) and its diffusion schedule."""
    from diffusers import AutoencoderKL, UNet2DConditionModel, VQModel

    config = config or LDMConfig()
    unet = None
    try:
        unet = UNet2DConditionModel.from_pretrained(config.model_id, subfolder="unet")
    except Exception:
        try:
            from diffusers import UNet2DModel

            unet = UNet2DModel.from_pretrained(config.model_id, subfolder="unet")
        except Exception as error:  # pragma: no cover
            raise RuntimeError(
                f"could not load an LDM U-Net from {config.model_id!r}.  Pass a "
                "diffusers-format directory (see scripts/convert_ldm_checkpoint.py) "
                "or a Hugging Face hub id of an unconditional LDM."
            ) from error

    vae_id = config.vae_id or config.model_id
    vae = None
    for cls in (AutoencoderKL, VQModel):
        try:
            vae = cls.from_pretrained(vae_id, subfolder="vae")
            break
        except Exception:
            vae = None
    scaling_factor = config.scaling_factor
    if scaling_factor is None:
        scaling_factor = float(getattr(getattr(vae, "config", None), "scaling_factor", 0.18215) or 0.18215)

    backend = LDMBackend(
        unet=unet,
        vae=vae,
        scaling_factor=scaling_factor,
        latent_channels=config.latent_channels,
        sample_mean=config.sample_mean,
    ).to(device)
    schedule = DiffusionSchedule(
        num_timesteps=config.num_timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        schedule=config.schedule,
    ).to(device)
    if verbose:
        print(
            f"[ldm] loaded {config.model_id} (vae={'yes' if vae is not None else 'no'}, "
            f"scaling_factor={scaling_factor}, schedule={config.schedule})",
            flush=True,
        )
    return backend, schedule
