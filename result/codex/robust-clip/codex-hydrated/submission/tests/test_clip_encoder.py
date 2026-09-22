"""Tests of the CLIP encoder wrapper that do not need pretrained weights.

Two properties that are easy to break and expensive to notice:

1. the pixel-space convention -- ``encode_image`` must apply exactly the same
   normalisation as the official ``open_clip`` transform, and
2. the positional-embedding interpolation for resolutions other than the one
   the checkpoint was trained at (App. B.10) must not compound when the
   resolution changes back and forth.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

open_clip = pytest.importorskip("open_clip")

from robust_clip.models import CLIPImageEncoder, build_pixel_transforms  # noqa: E402


def _random_encoder(feature: str = "projected_class_token") -> CLIPImageEncoder:
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=None, force_quick_gelu=True, jit=False
    )
    model.eval()
    return CLIPImageEncoder(model, feature=feature)


def test_pixel_space_normalisation_matches_open_clip():
    """``encode_image`` on [0,1] images == ``open_clip`` on normalised images."""
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    encoder = _random_encoder()
    _, pixel_transform = build_pixel_transforms(224, "ViT-B-32")
    reference_transform = transforms.Compose(
        [
            pixel_transform,
            transforms.Normalize(
                mean=tuple(encoder.image_mean.tolist()), std=tuple(encoder.image_std.tolist())
            ),
        ]
    )
    from PIL import Image

    pil_images = [Image.fromarray((torch.rand(256, 256, 3) * 255).byte().numpy()) for _ in range(3)]
    normalised = torch.stack([reference_transform(image) for image in pil_images])
    images = torch.stack([pixel_transform(image) for image in pil_images])

    with torch.no_grad():
        ours = encoder.encode_image(images)
        reference = encoder.model.encode_image(normalised)
    assert torch.allclose(ours, reference, atol=1e-5)


def test_positional_embedding_interpolation_is_reversible():
    encoder = _random_encoder()
    x224 = torch.rand(1, 3, 224, 224)
    x32 = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        first = encoder.encode_image(x224)
        encoder.encode_image(x32)          # triggers the interpolation
        second = encoder.encode_image(x224)
        encoder.encode_image(x32)          # and back again
        third = encoder.encode_image(x224)
    assert torch.allclose(first, second, atol=1e-6)
    assert torch.allclose(first, third, atol=1e-6)

    with torch.no_grad():
        out = encoder.encode_image(x32)
    assert out.shape == (1, encoder.embed_dim)


def test_patch_tokens_have_the_expected_grid():
    encoder = _random_encoder()
    # ViT-B/32 has a patch size of 32
    for size, expected in [(224, 49), (192, 36), (64, 4)]:
        tokens = encoder.patch_tokens(torch.rand(1, 3, size, size))
        assert tokens.shape[1] == expected, f"{size}px -> {tokens.shape[1]} patches"
        assert tokens.shape == (1, expected, encoder.width)


def test_pixel_transforms_stop_before_normalisation():
    train_tf, eval_tf = build_pixel_transforms(224, "ViT-B-32")
    from PIL import Image

    image = Image.fromarray((torch.rand(256, 256, 3) * 255).byte().numpy())
    tensor = eval_tf(image)
    assert tensor.shape == (3, 224, 224)
    assert float(tensor.min()) >= 0.0 and float(tensor.max()) <= 1.0


if __name__ == "__main__":  # pragma: no cover
    test_pixel_space_normalisation_matches_open_clip()
    test_positional_embedding_interpolation_is_reversible()
    test_patch_tokens_have_the_expected_grid()
    test_pixel_transforms_stop_before_normalisation()
    print("clip encoder tests OK")
