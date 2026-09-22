import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from foa.models.prompt_vit import PromptViT  # noqa: E402


@pytest.fixture(scope="session")
def tiny_model():
    """A randomly initialised ViT-tiny wrapped for prompt adaptation (CPU, no download)."""
    import timm

    torch.manual_seed(0)
    vit = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10)
    vit.eval()
    for p in vit.parameters():
        p.requires_grad_(False)
    return PromptViT(vit, num_prompts=3)


@pytest.fixture(scope="session")
def tiny_images():
    torch.manual_seed(0)
    return torch.randn(4, 3, 224, 224)
