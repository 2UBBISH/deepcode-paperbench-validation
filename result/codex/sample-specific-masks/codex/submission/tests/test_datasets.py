"""Tests of the transforms (addendum) and of the deterministic split builder."""

import pytest
import torch

from smm.datasets import (
    DATASET_SPECS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_transforms,
    deterministic_split,
)


def test_table6_specification():
    expected = {
        "cifar10": (50000, 10000, 10),
        "cifar100": (50000, 10000, 100),
        "svhn": (73257, 26032, 10),
        "gtsrb": (39209, 12630, 43),
        "flowers102": (4093, 2463, 102),
        "dtd": (2820, 1692, 47),
        "ucf101": (7639, 3783, 101),
        "food101": (50500, 30300, 101),
        "sun397": (15888, 19850, 397),
        "eurosat": (13500, 8100, 10),
        "oxfordpets": (2944, 3669, 37),
    }
    for name, (n_train, n_test, n_classes) in expected.items():
        spec = DATASET_SPECS[name]
        assert (spec.train_size, spec.test_size, spec.num_classes) == (n_train, n_test, n_classes)


@pytest.mark.parametrize("image_size", [224, 384])
@pytest.mark.parametrize("train", [True, False])
def test_transforms(image_size, train):
    from PIL import Image

    transform = build_transforms(image_size, train)
    img = Image.fromarray((torch.rand(40, 50, 3).numpy() * 255).astype("uint8"))
    out = transform(img)
    assert out.shape == (3, image_size, image_size)
    assert out.dtype == torch.float32


def test_train_transform_includes_random_crop_and_flip():
    transform = build_transforms(224, train=True)
    names = [type(t).__name__ for t in transform.transforms]
    assert names[:3] == ["Resize", "RandomCrop", "RandomHorizontalFlip"]
    assert transform.transforms[-1].mean == IMAGENET_MEAN
    assert transform.transforms[-1].std == IMAGENET_STD


def test_deterministic_split_matches_requested_sizes():
    targets = [i % 10 for i in range(1000)]
    train_idx, test_idx = deterministic_split(targets, 600, 250, seed=0)
    assert len(train_idx) == 600
    assert len(test_idx) == 250
    assert set(train_idx).isdisjoint(test_idx)
    # roughly class balanced
    counts = [targets[i] for i in train_idx]
    assert max(counts.count(c) for c in range(10)) - min(counts.count(c) for c in range(10)) <= 1


def test_deterministic_split_is_reproducible():
    targets = [i % 7 for i in range(700)]
    a = deterministic_split(targets, 400, 200, seed=0)
    b = deterministic_split(targets, 400, 200, seed=0)
    assert a == b
    c = deterministic_split(targets, 400, 200, seed=1)
    assert a != c
