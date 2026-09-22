"""Couplings of Sections 3.2, 3.3 and 4."""

import torch

from si_couplings.couplings import (
    DataDecorruptionCoupling,
    IndependentCoupling,
    InpaintingCoupling,
    SuperResolutionCoupling,
    SuperResolutionBaselineCoupling,
    tile_mask,
)
from si_couplings.interpolants import build_interpolant


def test_tile_mask_structure():
    """64 tiles, each missing with probability p = 0.3, constant over channels."""
    torch.manual_seed(0)
    xi = tile_mask(64, 3, 64, 64, n_tiles=8, p_missing=0.3)
    assert xi.shape == (64, 1, 64, 64)
    assert set(xi.unique().tolist()) <= {0.0, 1.0}
    # tiles are 8x8 blocks
    block = xi[0, 0, :8, :8]
    assert block.min() == block.max()
    # the observed fraction is roughly 1 - 0.3
    assert abs(xi.mean().item() - 0.7) < 0.05


def test_inpainting_base_is_observed_plus_noise():
    torch.manual_seed(0)
    coupling = InpaintingCoupling(n_tiles=8, p_missing=0.3)
    x1 = torch.randn(4, 3, 32, 32)
    batch = coupling.sample(x1)
    xi = batch.cond
    assert torch.allclose(xi * batch.x0, xi * x1)
    # the base is not the target: the missing region has been re-drawn
    assert not torch.allclose(batch.x0, x1)
    # the mask returned to the model is the complement of xi
    assert torch.allclose(batch.mask, 1 - xi)


def test_inpainting_interpolant_fixes_observed_pixels():
    """xi o I_t = xi o x_1 for every t, hence b_t = 0 on observed pixels."""
    torch.manual_seed(0)
    coupling = InpaintingCoupling()
    x1 = torch.randn(2, 3, 16, 16)
    batch = coupling.sample(x1)
    sch = build_interpolant("linear_zero_gamma")
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        tb = torch.full((2,), t)
        i_t = sch.interpolate(tb, batch.x0, batch.x1)
        i_dot = sch.interpolate_velocity(tb, batch.x0, batch.x1)
        assert torch.allclose(batch.cond * i_t, batch.cond * x1, atol=1e-5)
        assert (batch.cond * i_dot).abs().max() < 1e-6


def test_data_decorruption_coupling_statistics():
    """E|x_0 - x_1|^2 = d sigma^2 (Sec. 3.3)."""
    torch.manual_seed(0)
    d, sigma, n = 8, 0.5, 40_000
    coupling = DataDecorruptionCoupling(sigma=sigma)
    x1 = torch.randn(n, d)
    batch = coupling.sample(x1)
    mse = ((batch.x0 - batch.x1) ** 2).sum(-1).mean()
    assert abs(mse.item() - d * sigma**2) < 0.2


def test_super_resolution_coupling():
    torch.manual_seed(0)
    coupling = SuperResolutionCoupling(scale=4, sigma=0.5)
    x1 = torch.randn(2, 3, 64, 64)
    batch = coupling.sample(x1)
    assert batch.cond.shape == x1.shape  # up-sampled low-resolution image
    assert batch.x0.shape == x1.shape
    # the conditioning is exactly xi = U(D(x_1))
    xi = coupling.upsample(coupling.downsample(x1), size=x1.shape[-2:])
    assert torch.allclose(batch.cond, xi, atol=1e-6)
    # and the base is xi plus sigma zeta, so it stays close to the low-res image
    assert (batch.x0 - batch.cond).abs().mean() < 0.6
    # the low-resolution content is preserved up to the interpolation error
    # (bicubic up-sampling followed by area down-sampling is not an exact
    # inverse, so we only require the two low-resolution images to agree
    # to within a small relative tolerance)
    low = coupling.downsample(x1)
    assert (coupling.downsample(batch.cond) - low).abs().mean() < 0.1


def test_independent_coupling_is_uncorrelated():
    torch.manual_seed(0)
    coupling = IndependentCoupling()
    x1 = torch.randn(20_000, 4)
    batch = coupling.sample(x1)
    corr = torch.corrcoef(torch.stack([batch.x0.flatten(), batch.x1.flatten()]))[0, 1]
    assert corr.abs() < 0.03


def test_super_resolution_baseline_is_uncoupled_but_conditioned():
    """The uncoupled super-resolution baseline still sees the low-res image."""
    torch.manual_seed(0)
    coupling = SuperResolutionBaselineCoupling(scale=2, base_scale=1.0)
    x1 = torch.randn(4, 3, 16, 16)
    batch = coupling.sample(x1)
    xi = coupling.upsample(coupling.downsample(x1), size=x1.shape[-2:])
    assert torch.allclose(batch.cond, xi, atol=1e-6)
    assert batch.x0.shape == x1.shape
    corr = torch.corrcoef(torch.stack([batch.x0.flatten(), batch.x1.flatten()]))[0, 1]
    assert corr.abs() < 0.05
