"""Smoke tests for the baselines, the quantisation and the evaluation runner."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from foa.core.statistics import compute_source_statistics
from foa.data.datasets import IMAGENET_C_CORRUPTIONS, SyntheticStream
from foa.evaluation.runner import run_stream
from foa.methods import BNAdapt, CoTTA, FOAMethod, LAME, NoAdapt, SAR, T3A, TENT
from foa.methods.variants import CMANormAdapt, SGDAdapt
from foa.config import FOAConfig
from foa.quantization import (
    calibrate,
    dequantize_vit,
    quantize_vit,
    set_quantization_enabled,
)


def _stream(n=3, bs=4, num_classes=10):
    return SyntheticStream(
        num_samples=n * bs, num_classes=num_classes, img_size=224, batch_size=bs, num_batches=n
    )


@pytest.mark.parametrize(
    "build",
    [
        lambda m: NoAdapt(m),
        lambda m: TENT(m, lr=1e-3),
        lambda m: SAR(m, lr=1e-3, num_classes=10),
        lambda m: CoTTA(m, lr=0.05, num_augmentations=2),
        lambda m: LAME(m, k=2),
        lambda m: T3A(m, num_classes=10),
        lambda m: BNAdapt(m),
    ],
)
def test_baselines_run(tiny_model, build):
    method = build(tiny_model)
    result = run_stream(method, _stream())
    assert 0.0 <= result.acc <= 100.0
    assert 0.0 <= result.ece <= 100.0
    assert result.num_samples == 12


def test_foamethod_runs(tiny_model, tiny_images):
    stats = compute_source_statistics(tiny_model, [tiny_images])
    method = FOAMethod(tiny_model, stats, cfg=FOAConfig(popsize=3, batch_size=4), device=torch.device("cpu"))
    result = run_stream(method, _stream())
    assert result.num_samples == 12
    assert "fitness_min" in result.extra


def test_foa_interval_method_runs(tiny_model, tiny_images):
    stats = compute_source_statistics(tiny_model, [tiny_images])
    cfg = FOAConfig(popsize=3, batch_size=1, interval=3, interval_store="image")
    method = FOAMethod(tiny_model, stats, cfg=cfg, device=torch.device("cpu"))
    result = run_stream(method, _stream(n=2, bs=3))
    assert result.num_samples == 6  # all samples are eventually predicted


def test_runner_defers_predictions():
    class Deferred:
        def __init__(self):
            self.last_extra = {}
            self.buffer = []

        def reset(self):
            self.buffer.clear()

        def step(self, images):
            self.buffer.append(images)
            if sum(b.shape[0] for b in self.buffer) < 4:
                return None
            logits = torch.zeros(4, 5)
            logits[:, 1] = 10.0
            self.buffer.clear()
            return logits

        def flush(self):
            return None

    stream = [(torch.zeros(2, 3, 8, 8), torch.ones(2, dtype=torch.long)) for _ in range(2)]
    result = run_stream(Deferred(), stream)
    assert result.num_samples == 4
    assert result.acc == pytest.approx(100.0)


# --------------------------------------------------------------------------------------
# PTQ4ViT
# --------------------------------------------------------------------------------------
def test_quantization_disabled_matches_float(tiny_model, tiny_images):
    import copy
    import timm

    vit = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10)
    vit.load_state_dict(tiny_model.vit.state_dict())
    vit.eval()
    from foa.models.prompt_vit import PromptViT

    qvit = copy.deepcopy(vit)
    quantize_vit(qvit, bits=8)
    set_quantization_enabled(qvit, False)
    with torch.no_grad():
        a = vit(tiny_images)
        b = qvit(tiny_images)
    assert torch.allclose(a, b, atol=1e-5)


def test_quantization_error_grows_with_lower_bits(tiny_model, tiny_images):
    import copy

    calib = [torch.randn(4, 3, 224, 224)]
    errors = {}
    for bits in (8, 6):
        vit = copy.deepcopy(tiny_model.vit)
        quantize_vit(vit, bits=bits)
        calibrate(vit, calib, search=True)
        with torch.no_grad():
            errors[bits] = float((vit(tiny_images) - tiny_model.vit(tiny_images)).abs().mean())
        dequantize_vit(vit)
        with torch.no_grad():
            assert torch.allclose(vit(tiny_images), tiny_model.vit(tiny_images), atol=1e-5)
    assert errors[8] < errors[6]


def test_foa_runs_on_quantized_model(tiny_model, tiny_images):
    import copy

    from foa.models.prompt_vit import PromptViT

    vit = copy.deepcopy(tiny_model.vit)
    quantize_vit(vit, bits=8)
    calibrate(vit, [torch.randn(4, 3, 224, 224)], search=True)
    model = PromptViT(vit, num_prompts=3)
    stats = compute_source_statistics(model, [tiny_images])
    method = FOAMethod(model, stats, cfg=FOAConfig(popsize=2, batch_size=4), device=torch.device("cpu"))
    result = run_stream(method, _stream(n=1, bs=4))
    assert result.num_samples == 4


def test_corruption_list_matches_the_paper():
    assert len(IMAGENET_C_CORRUPTIONS) == 15
    assert IMAGENET_C_CORRUPTIONS[0] == "gaussian_noise"
    assert IMAGENET_C_CORRUPTIONS[-1] == "jpeg_compression"
    assert "elastic_transform" in IMAGENET_C_CORRUPTIONS


# --------------------------------------------------------------------------------------
# Table 9 design-choice variants
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "build",
    [
        lambda m, s: SGDAdapt(m, s, "prompts", "entropy", lr=0.01),
        lambda m, s: SGDAdapt(m, s, "norm", "eqn5"),
        lambda m, s: SGDAdapt(m, s, "prompts", "eqn5"),
        lambda m, s: CMANormAdapt(m, s, "eqn5", popsize=3),
        lambda m, s: CMANormAdapt(m, s, "entropy", popsize=3),
    ],
)
def test_table9_variants_run(tiny_model, tiny_images, build):
    stats = compute_source_statistics(tiny_model, [tiny_images])
    result = run_stream(build(tiny_model, stats), _stream(n=1, bs=4))
    assert result.num_samples == 4
