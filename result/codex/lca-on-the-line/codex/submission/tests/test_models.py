"""Model-zoo tests (no checkpoint downloads: weights=None is used)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lca_on_the_line.models import (  # noqa: E402
    ModelSpec,
    TorchvisionClassifier,
    all_model_specs,
    clip_specs,
    lavis_specs,
    openclip_specs,
    templates_key,
    vision_model_specs,
)


def test_registry_counts_match_appendix_a():
    assert len(vision_model_specs()) == 36
    assert len(clip_specs()) == 7
    assert len(openclip_specs()) == 30
    assert len(lavis_specs()) == 2
    specs = all_model_specs()
    assert len(specs) == 75
    assert sum(s.family == "VM" for s in specs) == 36
    assert sum(s.family == "VLM" for s in specs) == 39
    assert len({s.name for s in specs}) == 75


def test_model_names_are_unique_and_readable():
    for spec in all_model_specs():
        assert spec.name
        assert spec.source in ("torchvision", "clip", "open_clip", "lavis")
        assert spec.family in ("VM", "VLM")


def test_templates_key_is_content_aware():
    a = ["a photo of a {}."]
    b = ["a photo of a {}."]
    c = ["a cropped photo of a {}."]
    d = ["a photo of a {}.", "a {}."]
    assert templates_key(a) == templates_key(b)
    assert templates_key(a) != templates_key(c)
    assert templates_key(a) != templates_key(d)


@pytest.mark.parametrize(
    "arch,expected_dim",
    [
        ("resnet18", 512),
        ("vgg11", 4096),
        ("densenet121", 1024),
        ("googlenet", 1024),      # must be the *main* head, not the auxiliary
        ("inception_v3", 2048),   # ditto
        ("squeezenet1_1", 512 * 13 * 13),  # conv head, no nn.Linear
    ],
)
def test_feature_dimensions(arch, expected_dim):
    spec = ModelSpec(name=arch, family="VM", source="torchvision", arch=arch,
                     pretrained=None)
    classifier = TorchvisionClassifier(spec, device="cpu", batch_size=1)
    size = 299 if arch == "inception_v3" else 224
    image = [np.zeros((size, size, 3), dtype=np.uint8)]
    from PIL import Image

    features = classifier.features([Image.fromarray(image[0])])
    assert features.shape == (1, expected_dim)


def test_feature_module_is_the_main_classifier():
    """GoogLeNet/Inception keep auxiliary classifiers; we must not pick them."""
    for arch, expected in (("googlenet", "fc"), ("inception_v3", "fc"),
                           ("resnet18", "fc"), ("swin_b", "head")):
        spec = ModelSpec(name=arch, family="VM", source="torchvision", arch=arch,
                         pretrained=None)
        classifier = TorchvisionClassifier(spec, device="cpu")
        module = classifier._feature_module()
        assert module is getattr(classifier.model, expected)
