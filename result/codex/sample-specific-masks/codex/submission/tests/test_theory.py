"""Tests of the theoretical claims (Theorem 4.2, Propositions 4.3 and B.1)."""

import torch

from smm.theory import (
    empirical_approximation_error_experiment,
    encode_shared_mask,
    snap_mask_to_patches,
    verify_sample_specific_inclusion,
    verify_shared_mask_inclusion,
    verify_watermark_inclusion,
)
from smm.mask_generator import MaskNet


def test_snap_mask_to_patches_is_patch_constant():
    mask = torch.zeros(16, 16)
    mask[:5, :] = 1.0
    snapped = snap_mask_to_patches(mask, patch_size=4)
    assert torch.allclose(snapped[:8], torch.ones(8, 16))    # rows 0-7 are covered
    assert snapped[8:].sum() == 0


def test_proposition_4_3_shared_mask_inclusion():
    result = verify_shared_mask_inclusion(num_pool_layers=2)
    assert result["included"] == 1.0
    assert result["max_abs_diff"] < 1e-6


def test_proposition_4_3_watermark_special_case():
    result = verify_watermark_inclusion()
    assert result["included"] == 1.0
    assert result["mask_error"] < 1e-6


def test_proposition_B_1_sample_specific_inclusion():
    assert verify_sample_specific_inclusion()["included"] == 1.0


def test_encode_shared_mask_rejects_unrepresentable_masks():
    mask_net = MaskNet((16, 16), hidden_channels=(4, 4, 4), num_pool_layers=2)
    mask = torch.zeros(16, 16)
    mask[3, 3] = 1.0  # not constant over the 4x4 patches
    try:
        encode_shared_mask(mask, mask_net, snap_to_patches=False)
    except ValueError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("expected a ValueError for a non-representable mask")


def test_empirical_approximation_error_ordering():
    result = empirical_approximation_error_experiment(
        num_samples=512, steps=120, restarts=1, seed=0
    )
    errors = result["approximation_error"]
    # Theorem 4.2 + Proposition 4.3: SMM is not worse than the shared mask or a
    # sample-specific pattern without the shared delta.
    assert errors["smm"] <= errors["shr"] + 1e-6
    assert errors["smm"] <= errors["sp"] + 1e-6
