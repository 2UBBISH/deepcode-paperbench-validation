"""End-to-end smoke test of Algorithm 1 and Algorithm 2 on synthetic images."""

import torch
from torch.utils.data import DataLoader

from si_couplings.data.imagenet import SyntheticImages
from si_couplings.sample import sample_from_base, sample_inpainting
from si_couplings.train import build_experiment, train


def _config(task: str, coupling: str) -> dict:
    return {
        "task": task,
        "seed": 0,
        "interpolant": "linear_zero_gamma" if coupling != "independent" else "linear",
        "coupling": {"name": coupling, "n_tiles": 4, "p_missing": 0.3, "scale": 2, "sigma": 0.5},
        "data": {"kind": "synthetic", "image_size": 16, "channels": 3, "num_classes": 4, "n": 256},
        "model": {
            "kind": "unet",
            "dim": 8,
            "dim_mults": [1, 2],
            "attn_dim_head": 8,
            "attn_heads": 2,
            "image_size": 16,
            "num_classes": 4,
        },
        "optim": {"batch_size": 8, "lr": 1e-3, "max_steps": 3, "log_every": 1, "save_every": 3},
    }


def test_training_and_sampling_inpainting(tmp_path):
    cfg = _config("inpainting", "inpainting")
    ckpt = train(cfg, out_dir=tmp_path, max_steps=3)
    assert ckpt.exists()

    parts = build_experiment(cfg)
    model, coupling = parts["model"], parts["coupling"]
    x1 = torch.randn(2, 3, 16, 16)
    out = sample_inpainting(model, coupling, x1, steps=3, method="euler")
    assert out["sample"].shape == x1.shape
    # the observed pixels are untouched by construction
    xi = out["xi"]
    assert torch.allclose(xi * out["sample"], xi * x1, atol=1e-4)


def test_training_baseline_independent_coupling(tmp_path):
    cfg = _config("inpainting", "independent")
    train(cfg, out_dir=tmp_path, max_steps=2)
    parts = build_experiment(cfg)
    sample = sample_from_base(parts["model"], torch.randn(2, 3, 16, 16), steps=2, method="euler")
    assert sample.shape == (2, 3, 16, 16)


def test_super_resolution_smoke(tmp_path):
    cfg = _config("super_resolution", "super_resolution")
    cfg["data"]["image_size"] = 16
    parts = build_experiment(cfg)
    model, coupling = parts["model"], parts["coupling"]
    from si_couplings.data.superres import downsample
    from si_couplings.sample import sample_super_resolution

    high = torch.randn(2, 3, 16, 16)
    low = downsample(high, coupling.scale)
    out = sample_super_resolution(model, coupling, low, steps=3, method="euler", high_res_truth=high)
    assert out["sample"].shape == high.shape
