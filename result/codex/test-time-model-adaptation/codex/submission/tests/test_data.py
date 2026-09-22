"""Tests of the dataset layout / label mapping and of the stream builders."""
from __future__ import annotations

import os

import pytest
import torch
from PIL import Image

from foa.data import datasets as D


def _write_images(folder, n=2):
    os.makedirs(folder, exist_ok=True)
    for i in range(n):
        Image.new("RGB", (16, 16), color=(i * 37 % 255, 10, 20)).save(
            os.path.join(folder, f"{i}.JPEG")
        )


@pytest.fixture()
def fake_imagenet(tmp_path):
    wnids = list(D.imagenet_class_index().keys())[:4]
    root = tmp_path
    for wnid in wnids:
        _write_images(os.path.join(root, "val", wnid))
        _write_images(os.path.join(root, "imagenet-c", "gaussian_noise", "5", wnid))
        _write_images(os.path.join(root, "imagenet-r", wnid))
        _write_images(os.path.join(root, "sketch", wnid))
    # ImageNet-V2 names its folders with the (0-based) class index
    for idx in range(3):
        _write_images(os.path.join(root, "imagenet-v2", "imagenetv2-matched-frequency-format-val", str(idx)))
    return root, wnids


def test_imagenet_val_and_imageNet_c_records(fake_imagenet):
    root, wnids = fake_imagenet
    index = D.imagenet_class_index()
    val = D.imagenet_val_records(str(root))
    assert len(val) == 2 * len(wnids)
    assert all(r.label == index[os.path.basename(os.path.dirname(r.path))] for r in val)

    c = D.imagenet_c_records(str(root), "gaussian_noise", 5)
    assert len(c) == 2 * len(wnids)
    assert {r.label for r in c} == {index[w] for w in wnids}


def test_imagenet_r_v2_sketch_records(fake_imagenet):
    root, _ = fake_imagenet
    assert len(D.imagenet_r_records(str(root))) == 8
    assert len(D.imagenet_sketch_records(str(root))) == 8
    v2 = D.imagenet_v2_records(str(root))
    assert len(v2) == 6
    assert {r.label for r in v2} == {0, 1, 2}


def test_corruption_and_stream_helpers(fake_imagenet):
    root, _ = fake_imagenet
    records = D.imagenet_c_records(str(root), "gaussian_noise", 5)
    transform = D.build_eval_transform(None, img_size=32)
    ds = D.ImageRecordDataset(records, transform)
    image, label = ds[0]
    assert image.shape == (3, 32, 32)
    assert isinstance(label, int)

    cfg = D.StreamConfig(batch_size=2, shuffle=False, num_workers=0, order="class_order")
    loader = D.build_stream(records, transform, cfg)
    labels = torch.cat([t for _, t in loader])
    assert torch.equal(labels, labels.sort().values)


def test_synthetic_stream():
    stream = D.SyntheticStream(num_samples=12, num_classes=5, img_size=32, batch_size=4)
    batches = list(stream)
    assert len(batches) == 3
    images, targets = batches[0]
    assert images.shape == (4, 3, 32, 32)
    assert targets.min() >= 0 and targets.max() < 5
