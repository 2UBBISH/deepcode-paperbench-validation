"""Tests of ``f_in``: SMM, the shared-mask baselines and the ablations."""

import torch

from smm.reprogram import (
    InputReprogramming,
    InputReprogrammingConfig,
    border_mask,
    build_input_reprogramming,
)


def test_border_mask_widths():
    mask = border_mask((224, 224), 28)[0, 0]
    assert mask[:28].all() and mask[-28:].all()
    assert mask[:, :28].all() and mask[:, -28:].all()
    assert mask[28:-28, 28:-28].sum() == 0


def test_shared_mask_variants():
    x = torch.randn(2, 3, 224, 224)
    full = build_input_reprogramming((224, 224), "full")
    narrow = build_input_reprogramming((224, 224), "narrow")
    medium = build_input_reprogramming((224, 224), "medium")
    pad = build_input_reprogramming((224, 224), "pad")
    assert torch.allclose(full.make_mask(x), torch.ones_like(x))       # full watermark
    assert narrow.make_mask(x).sum() < full.make_mask(x).sum()
    assert narrow.make_mask(x).sum() < medium.make_mask(x).sum()
    assert pad.make_mask(x).sum() == narrow.make_mask(x).sum()          # same border width


def test_pad_variant_resizes_the_image():
    cfg = InputReprogrammingConfig(image_size=(224, 224), variant="shared", mask_kind="pad",
                                   pad_width=28)
    module = InputReprogramming(cfg)
    x = torch.randn(2, 3, 224, 224)
    r_x = module.resized_input(x)
    assert r_x.shape == (2, 3, 224, 224)
    assert torch.allclose(r_x[:, :, :28, :], torch.zeros_like(r_x[:, :, :28, :]))
    assert not torch.allclose(r_x[:, :, 28:-28, 28:-28], torch.zeros_like(r_x[:, :, 28:-28, 28:-28]))


def test_smm_forward_and_parameters():
    module = build_input_reprogramming((224, 224), "smm", num_layers=5, num_pool_layers=3)
    x = torch.randn(2, 3, 224, 224)
    out, mask = module(x, return_mask=True)
    assert out.shape == x.shape and mask.shape == x.shape
    assert module.delta.shape == (1, 3, 224, 224)
    assert torch.allclose(module.delta, torch.zeros_like(module.delta))  # delta starts at zero
    assert len(module.mask_parameters()) > 0


def test_only_mask_variant_has_no_delta():
    module = build_input_reprogramming((224, 224), "only_mask")
    assert module.delta is None
    out = module(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 3, 224, 224)


def test_single_channel_variant():
    module = build_input_reprogramming((224, 224), "single_channel")
    assert module.mask_net.single_channel
    mask = module.make_mask(torch.randn(1, 3, 224, 224))
    assert torch.allclose(mask[:, 0], mask[:, 1])


def test_vit_mask_net_six_layers():
    module = build_input_reprogramming((384, 384), "smm", num_layers=6, num_pool_layers=3)
    assert len(module.mask_net.generator.convs) == 6
    out = module(torch.randn(1, 3, 384, 384))
    assert out.shape == (1, 3, 384, 384)
    assert module.mask_net.mask_size == (48, 48)


def test_patch_size_sweep_shapes():
    for l in range(5):
        module = build_input_reprogramming((224, 224), "smm", num_pool_layers=l)
        assert module.patch_size == 2 ** l
        assert module(torch.randn(1, 3, 224, 224)).shape == (1, 3, 224, 224)


def test_gradient_flows_to_delta_and_mask_generator():
    module = build_input_reprogramming((64, 64), "smm", num_pool_layers=2)
    # delta is initialised to zero (Section 3.4), so the mask generator only
    # starts receiving gradients once the pattern is non-zero.
    with torch.no_grad():
        module.delta.fill_(1.0)
    out = module(torch.randn(2, 3, 64, 64))
    out.sum().backward()
    assert module.delta.grad is not None and module.delta.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.mask_parameters())


def test_zero_initialised_delta_blocks_mask_gradients_at_the_first_step():
    module = build_input_reprogramming((64, 64), "smm", num_pool_layers=2)
    assert torch.allclose(module.delta, torch.zeros_like(module.delta))
    out = module(torch.randn(2, 3, 64, 64))
    out.sum().backward()
    assert module.delta.grad is not None and module.delta.grad.abs().sum() > 0
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in module.mask_parameters())
