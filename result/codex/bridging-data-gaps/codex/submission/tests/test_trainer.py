"""Checks of Algorithm 1 (`ANTTrainer`)."""

from __future__ import annotations

import torch
import torch.nn as nn

from dpms_ant.adaptor import AdaptorConfig, adaptor_parameters, add_adaptors
from dpms_ant.ant_trainer import ANTConfig, ANTTrainer, eps_predictor
from dpms_ant.backbones import DDPM256Config, build_ddpm_unet
from dpms_ant.schedules import DiffusionSchedule


class TinyImageClassifier(nn.Module):
    """Minimal ``p_phi(y | x_t)`` for image-shaped inputs."""

    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, num_classes),
        )

    def forward(self, x, timesteps=None):
        return self.net(x)


def build_small_denoiser(image_size: int = 32):
    model = build_ddpm_unet(
        DDPM256Config(
            image_size=image_size,
            num_channels=32,
            num_res_blocks=1,
            channel_mult="1,2",
            attention_resolutions="8,4",
            num_head_channels=16,
        )
    )
    with torch.no_grad():
        model.out[-1].weight.normal_(0.0, 0.02)
        model.out[-1].bias.zero_()
    return model


def test_trainer_reduces_the_loss_without_adversarial_noise():
    """With a stationary objective the adaptor-only training must make progress."""
    torch.manual_seed(0)
    schedule = DiffusionSchedule(num_timesteps=100)
    model = build_small_denoiser()
    x = torch.randn(1, 3, 32, 32)
    t = torch.zeros(1, dtype=torch.long)
    add_adaptors(
        model,
        AdaptorConfig(bottleneck_factor=4, bottleneck_dim=8),
        example_input=x,
        forward_kwargs={"timesteps": t},
    )
    classifier = TinyImageClassifier()
    config = ANTConfig(
        iterations=20,
        batch_size=4,
        lr=1e-3,
        use_similarity_guidance=False,
        use_adversarial_noise=False,
        log_every=10,
    )
    target = torch.randn(4, 3, 32, 32)
    trainer = ANTTrainer(model, classifier, schedule, config, in_channels=3)
    history = trainer.train(iter(lambda: target, None))
    assert history["loss"][-1] < history["loss"][0]
    assert not any(
        p.requires_grad for name, p in model.named_parameters() if ".adaptor." not in name
    )
    assert len(adaptor_parameters(model)) > 0


def test_trainer_runs_with_adversarial_noise_and_similarity_guidance():
    torch.manual_seed(0)
    schedule = DiffusionSchedule(num_timesteps=50)
    model = build_small_denoiser()
    x = torch.randn(1, 3, 32, 32)
    t = torch.zeros(1, dtype=torch.long)
    add_adaptors(
        model,
        AdaptorConfig(bottleneck_factor=4, bottleneck_dim=8),
        example_input=x,
        forward_kwargs={"timesteps": t},
    )
    before = [p.detach().clone() for p in adaptor_parameters(model)]
    classifier = TinyImageClassifier()
    config = ANTConfig(
        iterations=3,
        batch_size=4,
        lr=1e-4,
        use_similarity_guidance=True,
        use_adversarial_noise=True,
        log_every=1,
    )
    config.adversarial.num_steps = 2
    target = torch.randn(4, 3, 32, 32)
    trainer = ANTTrainer(model, classifier, schedule, config, in_channels=3)
    history = trainer.train(iter(lambda: target, None))
    assert len(history["loss"]) == 3
    after = adaptor_parameters(model)
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))


def test_eps_predictor_splits_learned_variance():
    model = build_small_denoiser()
    predictor = eps_predictor(model, in_channels=3)
    x = torch.randn(2, 3, 32, 32)
    t = torch.tensor([1, 2])
    with torch.no_grad():
        eps = predictor(x, t)
    assert eps.shape == (2, 3, 32, 32)
