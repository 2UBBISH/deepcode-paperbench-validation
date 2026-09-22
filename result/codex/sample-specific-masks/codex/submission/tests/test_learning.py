"""Does the training loop actually learn?  Tiny over-fitting checks.

These tests optimise a handful of synthetic images (no downloads, random
backbone) and assert that the training loss of the reprogramming module goes
down -- i.e. that gradients reach both ``delta`` and the mask generator and that
the label-mapping machinery is wired correctly.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from smm.train import TrainConfig, train


def _tiny_loader(num_samples: int = 8, image_size: int = 64, num_classes: int = 4,
                 seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(num_samples, 3, image_size, image_size, generator=g)
    y = torch.arange(num_samples) % num_classes
    return DataLoader(TensorDataset(x, y), batch_size=num_samples, shuffle=False)


def _run(method: str, label_mapping: str = "rlm", epochs: int = 6):
    cfg = TrainConfig(
        dataset="cifar10", backbone="resnet18", image_size=64, method=method,
        num_pool_layers=2, epochs=epochs, batch_size=8, seed=0,
        label_mapping=label_mapping, device="cpu", pretrained=False,
        eval_every=0, num_workers=0, lr_delta=0.05, lr_mask=0.05,
        gamma_delta=1.0, gamma_mask=1.0,
    )
    loader = _tiny_loader()
    return train(cfg, loader, loader)


def test_smm_reduces_the_training_loss():
    result = _run("smm")
    losses = [h["train_loss"] for h in result["history"]]
    assert losses[-1] < losses[0], losses


def test_shared_mask_baseline_reduces_the_training_loss():
    result = _run("full")
    losses = [h["train_loss"] for h in result["history"]]
    assert losses[-1] < losses[0], losses


def test_flm_and_ilm_mappings_are_usable():
    for mapping in ("flm", "ilm"):
        result = _run("smm", label_mapping=mapping, epochs=2)
        assert 0.0 <= result["test_accuracy"] <= 100.0
        assert len(result["label_mapping_table"]["source"]) == result["num_classes"]
