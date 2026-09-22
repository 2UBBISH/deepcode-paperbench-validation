"""Checks of the LDM (latent diffusion) backend (Section 5.2)."""

from __future__ import annotations

import copy

import pytest
import torch

diffusers = pytest.importorskip("diffusers")

from diffusers import AutoencoderKL, UNet2DModel  # noqa: E402

from dpms_ant.adaptor import AdaptorConfig, add_adaptors  # noqa: E402
from dpms_ant.ant_trainer import eps_predictor  # noqa: E402
from dpms_ant.backbones import Classifier256Config, build_classifier_unet  # noqa: E402
from dpms_ant.guidance import classifier_guidance  # noqa: E402
from dpms_ant.ldm_backend import LDMBackend, PixelSpaceClassifier  # noqa: E402
from dpms_ant.schedules import DiffusionSchedule  # noqa: E402
from dpms_ant.utils import parameter_rate  # noqa: E402


def tiny_backend() -> LDMBackend:
    unet = UNet2DModel(
        sample_size=32,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(32, 64),
        down_block_types=("DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
        norm_num_groups=32,
    )
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        latent_channels=4,
        block_out_channels=(32,),
        layers_per_block=1,
        norm_num_groups=32,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        sample_size=32,
    )
    return LDMBackend(unet, vae, scaling_factor=0.18215)


def test_encode_decode_round_trip_shapes():
    backend = tiny_backend()
    images = torch.randn(2, 3, 32, 32)
    latents = backend.encode(images)
    assert latents.shape == (2, 4, 32, 32)
    assert backend.decode(latents).shape == (2, 3, 32, 32)


def test_adaptors_can_be_attached_to_a_diffusers_unet():
    backend = tiny_backend()
    reference = copy.deepcopy(backend.unet)
    latents = torch.randn(2, 4, 32, 32)
    timesteps = torch.zeros(2, dtype=torch.long)
    add_adaptors(
        backend.unet,
        AdaptorConfig(bottleneck_factor=2, bottleneck_dim=8),
        example_input=latents[:1],
        forward_kwargs={"timestep": timesteps[:1]},
    )
    with torch.no_grad():
        adapted = backend.unet(latents, timesteps).sample
        original = reference(latents, timesteps).sample
    assert torch.allclose(adapted, original, atol=1e-5)
    assert 0.0 < parameter_rate(backend.unet) < 1.0


def test_epsilon_prediction_in_latent_space():
    backend = tiny_backend()
    latents = torch.randn(2, 4, 32, 32)
    predictor = eps_predictor(backend, in_channels=4)
    with torch.no_grad():
        eps = predictor(latents, torch.zeros(2, dtype=torch.long))
    assert eps.shape == latents.shape


def test_pixel_space_classifier_guidance_is_taken_with_respect_to_latents():
    backend = tiny_backend()
    classifier = build_classifier_unet(2, Classifier256Config(image_size=32))
    wrapped = PixelSpaceClassifier(classifier, backend)
    latents = torch.randn(2, 4, 32, 32)
    schedule = DiffusionSchedule(
        num_timesteps=100, schedule="scaled_linear", beta_start=0.0015, beta_end=0.0195
    )
    timesteps = torch.full((2,), 30, dtype=torch.long)
    guidance = classifier_guidance(wrapped, schedule, latents, timesteps, gamma=1.0)
    assert guidance.shape == latents.shape
    assert torch.isfinite(guidance).all()
