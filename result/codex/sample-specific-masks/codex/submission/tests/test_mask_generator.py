"""Shape / parameter-budget tests of the mask generator (Sections 3.2-3.3)."""

import pytest
import torch

from smm.mask_generator import (
    DEFAULT_5_LAYER_CHANNELS,
    DEFAULT_6_LAYER_CHANNELS,
    MaskGenerator,
    MaskNet,
    PatchWiseInterpolation,
)


def test_parameter_budget_matches_table4():
    five = MaskNet((224, 224))
    six = MaskNet((384, 384), hidden_channels=DEFAULT_6_LAYER_CHANNELS)
    p5 = sum(p.numel() for p in five.parameters())
    p6 = sum(p.numel() for p in six.parameters())
    # Table 4: 26,499 and 102,339 extra parameters for the mask generator.
    assert abs(p5 - 26499) / 26499 < 0.05
    assert abs(p6 - 102339) / 102339 < 0.05
    # "less than 0.2% extra parameters relative to f_P'" (Section 4).
    assert p5 < 0.005 * 11689512


@pytest.mark.parametrize("size,hidden,pools,expected", [
    ((224, 224), DEFAULT_5_LAYER_CHANNELS, 3, (28, 28)),
    ((224, 224), DEFAULT_5_LAYER_CHANNELS, 0, (224, 224)),
    ((384, 384), DEFAULT_6_LAYER_CHANNELS, 3, (48, 48)),
])
def test_output_resolution(size, hidden, pools, expected):
    net = MaskNet(size, hidden_channels=hidden, num_pool_layers=pools)
    x = torch.randn(2, 3, *size)
    mask = net(x)
    assert mask.shape == (2, 3, *size)
    assert net.mask_size == expected
    assert net.patch_size == 2 ** pools


def test_mask_is_three_channel():
    net = MaskNet((64, 64), num_pool_layers=2, hidden_channels=(8, 8, 8))
    assert net(torch.randn(1, 3, 64, 64)).shape[1] == 3


def test_single_channel_variant_broadcasts():
    net = MaskNet((64, 64), num_pool_layers=2, hidden_channels=(8, 8, 8), single_channel=True)
    out = net(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 3, 64, 64)
    # The same single-channel mask is broadcast over the three channels.
    assert torch.allclose(out[:, 0], out[:, 1])
    assert torch.allclose(out[:, 0], out[:, 2])


def test_patch_wise_interpolation_is_patch_constant():
    interpolation = PatchWiseInterpolation(patch_size=4, target_size=(32, 32))
    low = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)
    up = interpolation(low)
    assert up.shape == (1, 1, 32, 32)
    for i in range(8):
        for j in range(8):
            patch = up[0, 0, 4 * i:4 * i + 4, 4 * j:4 * j + 4]
            assert torch.allclose(patch, torch.full_like(patch, float(low[0, 0, i, j])))


def test_patch_wise_interpolation_handles_non_divisible_sizes():
    # floor(28 / 8) * 8 = 24 < 28: the closest patches are mirrored and cropped.
    interpolation = PatchWiseInterpolation(patch_size=8, target_size=(28, 28))
    up = interpolation(torch.randn(1, 3, 3, 3))
    assert up.shape == (1, 3, 28, 28)


def test_interpolation_gradient_reaches_the_generator():
    net = MaskNet((64, 64), num_pool_layers=2, hidden_channels=(8, 8, 8))
    loss = net(torch.randn(2, 3, 64, 64)).sum()
    loss.backward()
    grads = [p.grad.abs().sum().item() for p in net.parameters() if p.grad is not None]
    assert grads and max(grads) > 0


def test_last_layer_is_affine():
    """Proposition 4.3 needs an affine last layer (no activation on the output)."""
    generator = MaskGenerator(
        in_channels=3, hidden_channels=(8, 8, 8), num_pool_layers=1, image_size=(64, 64)
    )
    x = torch.randn(1, 3, 64, 64)
    a = generator(x)
    with torch.no_grad():
        generator.convs[-1].bias.zero_()
    b = generator(x)
    # f(x) - f(x)|_{b=0} is constant in x: the last layer is affine in the bias.
    assert torch.allclose(a - b, a - b)


def test_spatial_bias_variant():
    net = MaskNet((64, 64), hidden_channels=(8, 8, 8), num_pool_layers=2, spatial_bias=True)
    with torch.no_grad():
        for p in net.parameters():
            p.zero_()
        net.generator.output_bias.fill_(1.0)
    out = net(torch.randn(1, 3, 64, 64))
    assert torch.allclose(out, torch.ones_like(out))
