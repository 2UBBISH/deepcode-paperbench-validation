"""Integration test for the OpenFlamingo + OpenCLIP vision-encoder swap.

Builds a *tiny* Flamingo (random OpenCLIP ViT-B/32 + `tiny-random-gpt2`)
through the real ``open_flamingo`` package and checks the three things that the
paper's Table 1 depends on:

* the swapped-in encoder satisfies OpenFlamingo's contract (it returns
  ``(pooled, tokens)`` and element 1 is the token sequence the perceiver
  consumes),
* the visual features keep their gradient after
  :func:`patch_flamingo_for_attacks` -- without that patch the white-box
  attacks on `vision_x` are impossible,
* the runner's ``nll`` and ``generate`` work on pixel-space images.

The test is skipped when ``open_flamingo`` (or its tiny model) is unavailable.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

open_flamingo = pytest.importorskip("open_flamingo")
open_clip = pytest.importorskip("open_clip")

from robust_clip.lvlm.openflamingo import (  # noqa: E402
    OpenClipFlamingoVisionEncoder,
    OpenFlamingoRunner,
    patch_flamingo_for_attacks,
    replace_openflamingo_vision_encoder,
)
from robust_clip.models import CLIPImageEncoder  # noqa: E402


TINY_LM = "hf-internal-testing/tiny-random-gpt2"


def _tiny_flamingo():
    from open_flamingo import create_model_and_transforms

    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-B-32",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path=TINY_LM,
        tokenizer_path=TINY_LM,
        cross_attn_every_n_layers=1,
        decoder_layers_attr_name="transformer.h",
    )
    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=None, force_quick_gelu=True, jit=False
    )
    encoder = CLIPImageEncoder(clip_model)
    replace_openflamingo_vision_encoder(model, encoder)
    patch_flamingo_for_attacks(model)
    model.eval()
    return model, image_processor, tokenizer, encoder


def test_vision_encoder_contract_and_gradients():
    model, _, _, encoder = _tiny_flamingo()
    assert isinstance(model.vision_encoder, OpenClipFlamingoVisionEncoder)
    assert getattr(model, "attacks_use_gradients", False)

    images = torch.rand(2, 3, 224, 224, requires_grad=True)
    pooled, tokens = model.vision_encoder(images)
    assert pooled.shape == (2, encoder.embed_dim)
    assert tokens.shape == (2, 49, encoder.width)

    # the perceiver resampler consumes exactly this token sequence
    resampled = model.perceiver(tokens.unsqueeze(1).unsqueeze(1))
    assert resampled.shape[0] == 2 and resampled.shape[-1] == model.vis_dim

    # ... and gradients flow back to ``vision_x``, which is what the attacks need
    # (upstream's ``_encode_vision_x`` discards them with ``torch.no_grad()``)
    tokens.sum().backward()
    assert images.grad is not None
    assert float(images.grad.abs().sum()) > 0.0


def test_runner_nll_and_generate():
    model, image_processor, tokenizer, encoder = _tiny_flamingo()
    runner = OpenFlamingoRunner(model, image_processor, tokenizer, encoder, image_size=224, device="cpu")

    prompt = "<image>Question: is there a cat? Short answer:"
    images = torch.rand(2, 3, 224, 224, requires_grad=True)
    try:
        loss = runner.nll(images, [prompt, prompt], ["yes", "no"])
    except TypeError as error:  # pragma: no cover - environment-dependent
        # ``open_flamingo`` 2.0.1 was released against transformers ~4.30; with
        # a much newer transformers the GPT-2 decoder API changed and the
        # language-model forward itself fails.  That is unrelated to the
        # vision-encoder swap tested here.
        if "positional arguments" in str(error) or "positional" in str(error):
            pytest.skip(f"open_flamingo/transformers version mismatch: {error}")
        raise
    assert loss.shape == (2,)
    grad, = torch.autograd.grad(loss.sum(), images)
    assert float(grad.abs().sum()) > 0.0

    outputs = runner.generate(images.detach(), [prompt, prompt], max_new_tokens=2)
    assert isinstance(outputs, list) and len(outputs) == 2


if __name__ == "__main__":  # pragma: no cover
    test_vision_encoder_contract_and_gradients()
    test_runner_nll_and_generate()
    print("openflamingo integration OK")
