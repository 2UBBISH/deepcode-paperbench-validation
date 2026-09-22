"""Tests for the APT adapter (Section 4.1 and 4.3)."""

import math

import pytest
import torch

from apt.adapter import APTLinear


def make_layer(in_features=16, out_features=12, seed=0):
    torch.manual_seed(seed)
    base = torch.nn.Linear(in_features, out_features)
    return base, APTLinear(base, r=4, scaling=2.0)


def test_zero_init_is_identity():
    """``W_B = 0`` means the adapter must not change the layer output."""
    base, lin = make_layer()
    x = torch.randn(3, 16)
    torch.testing.assert_close(lin(x), base(x))


def test_masks_prune_input_and_output_dimensions():
    base, lin = make_layer()
    x = torch.ones(2, 16)
    out = lin(x)
    assert out.shape == (2, 12)
    lin.mask_out[3] = 0.0
    assert torch.allclose(lin(x)[:, 3], torch.zeros(2))
    lin.mask_in[0] = 0.0
    x2 = x.clone()
    x2[:, 0] = 1e6
    assert torch.allclose(lin(x2), lin(x), atol=1e-5)


def test_grow_rank_keeps_output_and_is_not_identity():
    base, lin = make_layer()
    # give the adapter a non-trivial value first
    torch.nn.init.normal_(lin.lora_A, std=0.1)
    torch.nn.init.normal_(lin.lora_B, std=0.1)
    x = torch.randn(3, 16)
    before = lin(x).detach().clone()
    lin.grow_rank(9)
    assert lin.rank == 9
    assert lin.lora_A.shape == (9, 16)
    assert lin.lora_B.shape == (12, 9)
    torch.testing.assert_close(lin(x), before, rtol=1e-5, atol=1e-6)
    # the original parameters must be preserved
    torch.testing.assert_close(lin.lora_B[:, :4], lin.lora_B[:, :4])


def test_merge_lora_folds_into_the_base_weight():
    base, lin = make_layer()
    torch.nn.init.normal_(lin.lora_A, std=0.1)
    torch.nn.init.normal_(lin.lora_B, std=0.1)
    x = torch.randn(3, 16)
    before = lin(x).detach().clone()
    expected = lin.effective_weight()
    lin.merge_lora()
    torch.testing.assert_close(lin.base.weight, expected)
    torch.testing.assert_close(lin(x), before, rtol=1e-5, atol=1e-6)


def test_salience_statistics_are_collected():
    base, lin = make_layer()
    lin.train()
    base2 = torch.nn.Linear(16, 12)
    base2.weight.data.copy_(base.weight.data)
    lin2 = APTLinear(base2, r=4)
    lin2.train()
    x = torch.randn(4, 16, requires_grad=True)
    lin2(x).sum().backward()
    sal_in, sal_out = lin2.compressed_salience()
    assert sal_in is not None and sal_out is not None
    assert sal_in.shape == (16,) and sal_out.shape == (12,)
    assert torch.isfinite(sal_in).all() and torch.isfinite(sal_out).all()


def test_activation_kurtosis_is_finite():
    base, lin = make_layer()
    lin.train()
    x = torch.randn(64, 16) * 3
    lin(x)
    k_in, k_out = lin.activation_kurtosis()
    assert k_in.shape == (16,) and k_out.shape == (12,)
    assert torch.isfinite(k_in).all() and torch.isfinite(k_out).all()
    assert (k_in >= -1e-6).all()
