"""Checks of the adaptor module (Section 4.3) and of its injection."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from dpms_ant.adaptor import (
    Adaptor,
    AdaptorConfig,
    AdaptorWrapper,
    add_adaptors,
    adaptor_parameters,
)
from dpms_ant.backbones import DDPM256Config, build_ddpm_unet
from dpms_ant.utils import parameter_rate


def tiny_unet():
    model = build_ddpm_unet(
        DDPM256Config(
            image_size=32,
            num_channels=32,
            num_res_blocks=1,
            channel_mult="1,2",
            attention_resolutions="8,4",
            num_head_channels=16,
        )
    )
    # guided-diffusion zero-initialises the output convolution; a freshly built
    # (i.e. untrained) model therefore outputs exactly zero and no gradient can
    # flow backwards.  Give it small random weights, like a trained checkpoint.
    with torch.no_grad():
        model.out[-1].weight.normal_(0.0, 0.02)
        model.out[-1].bias.zero_()
    return model


def test_adaptor_is_identity_at_initialisation():
    adaptor = Adaptor(16, 32, AdaptorConfig(bottleneck_factor=4, bottleneck_dim=8))
    x = torch.randn(2, 16, 16, 16)
    out = adaptor(x, out_shape=(2, 32, 8, 8))
    assert out.shape == (2, 32, 8, 8)
    assert out.abs().sum().item() == 0.0  # zero-initialised head


def test_adaptor_handles_odd_spatial_sizes():
    adaptor = Adaptor(8, 8, AdaptorConfig(bottleneck_factor=4, bottleneck_dim=8))
    x = torch.randn(1, 8, 7, 5)
    out = adaptor(x, out_shape=(1, 8, 7, 5))
    assert out.shape == (1, 8, 7, 5)


def test_injection_preserves_the_forward_pass():
    model = tiny_unet()
    reference = copy.deepcopy(model)
    x = torch.randn(1, 3, 32, 32)
    t = torch.tensor([5])
    add_adaptors(
        model,
        AdaptorConfig(bottleneck_factor=4, bottleneck_dim=8),
        example_input=x,
        forward_kwargs={"timesteps": t},
    )
    with torch.no_grad():
        assert torch.allclose(model(x, t), reference(x, t), atol=1e-5)


def test_only_adaptors_are_trainable_and_receive_gradients():
    model = tiny_unet()
    x = torch.randn(1, 3, 32, 32)
    t = torch.tensor([7])
    add_adaptors(
        model,
        AdaptorConfig(bottleneck_factor=4, bottleneck_dim=8),
        example_input=x,
        forward_kwargs={"timesteps": t},
    )
    parameters = adaptor_parameters(model)
    assert parameters and all(p.requires_grad for p in parameters)
    frozen = [
        name for name, p in model.named_parameters() if p.requires_grad and ".adaptor." not in name
    ]
    assert frozen == []

    out = model(x, t)
    target = torch.randn_like(out)  # the U-Net predicts 6 channels (learn_sigma)
    (out - target).pow(2).mean().backward()
    received = [
        name for name, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0
    ]
    assert received and all(".adaptor." in name for name in received)


def test_parameter_rate_is_small():
    model = tiny_unet()
    x = torch.randn(1, 3, 32, 32)
    t = torch.tensor([1])
    add_adaptors(
        model,
        AdaptorConfig(bottleneck_factor=4, bottleneck_dim=8),
        example_input=x,
        forward_kwargs={"timesteps": t},
    )
    assert 0.0 < parameter_rate(model) < 0.5


def test_wrapper_forwards_extra_arguments():
    base = nn.Conv2d(4, 4, kernel_size=1)
    wrapper = AdaptorWrapper(
        base, Adaptor(4, 4, AdaptorConfig(bottleneck_factor=2, bottleneck_dim=4))
    )
    x = torch.randn(2, 4, 8, 6)
    assert torch.allclose(wrapper(x), base(x), atol=1e-6)
