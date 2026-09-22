"""End-to-end smoke test of the training loop (Algorithm 1) on synthetic data.

No dataset download and no pre-trained weights are needed: a small randomly
initialised backbone is used, which keeps the test fast (a few seconds on CPU)
while exercising the same code path as the full experiments.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from smm.main import build_parser, config_from_args
from smm.train import ReprogramModel, TrainConfig, train


def _synthetic_loader(num_samples: int = 24, image_size: int = 64, num_classes: int = 5,
                      batch_size: int = 8, seed: int = 0, train: bool = True):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(num_samples, 3, image_size, image_size, generator=g)
    y = torch.randint(0, num_classes, (num_samples,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=train)


def test_reprogram_model_shapes_all_methods():
    for method in ["smm", "full", "narrow", "medium", "pad", "only_mask", "single_channel"]:
        model = ReprogramModel("resnet18", "cifar10", method=method, image_size=64,
                               pretrained=False)
        out = model.input_transform(torch.randn(2, 3, 64, 64))
        assert out.shape == (2, 3, 64, 64), method


def test_vit_backbone_accepts_384_input():
    model = ReprogramModel("vit_b32", "cifar10", method="smm", image_size=384, num_layers=6,
                           pretrained=False)
    logits = model.pretrained_logits(torch.randn(1, 3, 384, 384))
    assert logits.shape == (1, 1000)


def test_training_loop_runs_and_reports_accuracy():
    cfg = TrainConfig(
        dataset="cifar10", backbone="resnet18", image_size=64, method="smm",
        num_layers=5, num_pool_layers=2, epochs=2, batch_size=8, seed=0,
        label_mapping="ilm", device="cpu", pretrained=False, eval_every=1, num_workers=0,
    )
    train_loader = _synthetic_loader()
    test_loader = _synthetic_loader(seed=1, train=False)
    result = train(cfg, train_loader, test_loader)
    assert 0.0 <= result["test_accuracy"] <= 100.0
    assert len(result["history"]) == 2
    assert result["num_mask_parameters"] > 0
    assert result["num_delta_parameters"] == 3 * 64 * 64
    assert len(result["label_mapping_table"]["source"]) == 10


def test_baseline_training_loop_runs():
    cfg = TrainConfig(
        dataset="cifar10", backbone="resnet18", image_size=64, method="full",
        epochs=1, batch_size=8, seed=0, label_mapping="rlm", device="cpu",
        pretrained=False, eval_every=1, num_workers=0,
    )
    result = train(cfg, _synthetic_loader(), _synthetic_loader(seed=1, train=False))
    assert result["num_mask_parameters"] == 0
    assert result["num_delta_parameters"] == 3 * 64 * 64


def test_cli_config_resolution():
    args = build_parser().parse_args(
        ["--config", "configs/vitb32.yaml", "--dataset", "flowers102", "--method", "smm"]
    )
    cfg = config_from_args(args)
    assert cfg.backbone == "vit_b32"
    assert cfg.resolved_image_size() == 384
    assert cfg.num_layers == 6
