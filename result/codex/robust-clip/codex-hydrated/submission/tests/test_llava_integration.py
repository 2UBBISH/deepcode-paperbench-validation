"""Integration test for the LLaVA + OpenCLIP vision tower plumbing.

It builds a *tiny* LLaVA with the HuggingFace classes (no downloads: the
language model is randomly initialised and the vision encoder is a randomly
initialised OpenCLIP ViT-B/32), swaps in :class:`OpenClipVisionTower` and checks
that

* the tower exposes ``hidden_states`` in the layout that
  ``LlavaModel.get_image_features`` expects (so LLaVA's own feature selection is
  reused, with ``vision_feature_layer = -2``),
* the number of image tokens produced by the tower matches the number of
  ``<image>`` placeholders that :meth:`LlavaOpenClip.tokenize_prompts` inserts,
* the negative log-likelihood used by the attacks is differentiable w.r.t. the
  input image (which is what the whole attack pipeline relies on), and
* generation runs.
"""
from __future__ import annotations

import os
import re
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

open_clip = pytest.importorskip("open_clip")
transformers = pytest.importorskip("transformers")

from robust_clip.models import CLIPImageEncoder  # noqa: E402
from robust_clip.lvlm.llava import (  # noqa: E402
    LlavaOpenClip,
    OpenClipVisionTower,
    build_llava_prompt,
    inner_vision_tower,
    replace_vision_encoder,
)


VOCAB_SIZE = 33000        # must be > image_token_index (32000)
IMAGE_TOKEN_INDEX = 32000


class DummyTokenizer:
    """Deterministic word-level tokenizer with an ``<image>`` token."""

    def __init__(self):
        self.pad_token_id = 0
        self.image_token_id = IMAGE_TOKEN_INDEX

    def _encode(self, text: str):
        ids = [1]  # bos
        # ``<image>`` is a *special* token: a real tokenizer splits it out even
        # when it is repeated without separators (as LLaVA does).
        for chunk in re.split(r"(<image>)", text):
            if chunk == "<image>":
                ids.append(IMAGE_TOKEN_INDEX)
            elif chunk:
                for word in chunk.split(" "):
                    if word:
                        ids.append(abs(hash(word)) % (VOCAB_SIZE - 10) + 10)
        return ids

    def __call__(self, texts, return_tensors="pt", padding=True, add_special_tokens=True):
        if isinstance(texts, str):
            texts = [texts]
        sequences = [self._encode(t) for t in texts]
        max_len = max(len(s) for s in sequences)
        input_ids = torch.zeros(len(sequences), max_len, dtype=torch.long)
        attention = torch.zeros_like(input_ids)
        for i, sequence in enumerate(sequences):
            input_ids[i, :len(sequence)] = torch.tensor(sequence)
            attention[i, :len(sequence)] = 1
        return {"input_ids": input_ids, "attention_mask": attention}

    def batch_decode(self, sequences, skip_special_tokens=True):
        return ["answer"] * len(sequences)


def build_tiny_lvlm():
    from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig, LlavaForConditionalGeneration

    vision_config = CLIPVisionConfig(
        hidden_size=768,
        image_size=224,
        patch_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=1024,
        projection_dim=512,
    )
    text_config = LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=2048,
    )
    config = LlavaConfig(
        vision_config=vision_config,
        text_config=text_config,
        image_token_index=IMAGE_TOKEN_INDEX,
        vision_feature_layer=-2,
        vision_feature_select_strategy="default",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = LlavaForConditionalGeneration(config)
    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=None, force_quick_gelu=True, jit=False
    )
    encoder = CLIPImageEncoder(clip_model)
    replace_vision_encoder(model, encoder)
    tokenizer = DummyTokenizer()
    return LlavaOpenClip(model, tokenizer, encoder, image_size=224)


def test_tower_is_the_openclip_one_and_exposes_hf_layout():
    lvlm = build_tiny_lvlm()
    tower = inner_vision_tower(lvlm.model)
    assert isinstance(tower, OpenClipVisionTower)

    images = torch.rand(2, 3, 224, 224)
    out = lvlm.model.model.vision_tower(images, output_hidden_states=True)
    # 12 OpenCLIP blocks -> 13 HF hidden states (embedding + one per block)
    assert len(out.hidden_states) == 13
    # hidden_states[-2] is the output of the second-to-last block, with the CLS
    # token first (LLaVA drops it with [:, 1:])
    assert out.hidden_states[-2].shape == (2, 1 + 49, 768)

    features = lvlm.model.model.get_image_features(
        pixel_values=images,
        vision_feature_layer=-2,
        vision_feature_select_strategy="default",
        image_sizes=None,
    )
    assert len(features) == 2 and features[0].shape == (49, 64)


def test_nll_is_differentiable_and_generation_runs():
    lvlm = build_tiny_lvlm()
    prompt = build_llava_prompt("Describe the image concisely.", task="coco_caption")
    assert lvlm.num_image_tokens == 49
    tokens = lvlm.tokenize_prompts([prompt])
    assert int((tokens["input_ids"] == IMAGE_TOKEN_INDEX).sum()) == 49

    images = torch.rand(1, 3, 224, 224, requires_grad=True)
    loss = lvlm.nll(images, [prompt], ["a cat sits on a mat"])
    assert loss.shape == (1,)
    grad, = torch.autograd.grad(loss.sum(), images)
    assert grad.shape == images.shape
    assert float(grad.abs().sum()) > 0.0

    outputs = lvlm.generate(images.detach(), [prompt], max_new_tokens=3)
    assert len(outputs) == 1


if __name__ == "__main__":  # pragma: no cover
    test_tower_is_the_openclip_one_and_exposes_hf_layout()
    test_nll_is_differentiable_and_generation_runs()
    print("llava integration OK")
