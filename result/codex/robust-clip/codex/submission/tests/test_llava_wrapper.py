"""Integration test of the LLaVA wrapper with a tiny randomly initialized model.

The vision tower is an OpenCLIP ViT-B/32 (random weights, no download) and the
language model is a tiny randomly initialized ``LlavaForConditionalGeneration``;
this validates the multimodal input construction (``<image>`` token replacement),
the gradient flow into the image and the interface used by the attacks.
"""

import torch

from robust_clip.models.clip_encoder import load_clip
from robust_clip.models.lvlm.llava_openclip import (
    IMAGE_TOKEN_INDEX,
    LlavaOpenCLIP,
    OpenCLIPVisionTower,
    tokenizer_image_token,
)


class DummyTokenizer:
    """Minimal stand-in for the LLaVA tokenizer (deterministic ids, no download)."""

    bos_token_id = 1
    unk_token_id = 0
    pad_token_id = 0
    eos_token_id = 2
    vocab_size = 128

    def __call__(self, text, add_special_tokens=True, return_tensors=None, padding=False, **kwargs):
        single = isinstance(text, str)
        texts = [text] if single else list(text)
        encoded: list = []
        for value in texts:
            ids = [self.bos_token_id] if add_special_tokens else []
            ids += [3 + (ord(char) % 90) for char in value]
            encoded.append(ids)
        if return_tensors is None:
            # a real HF tokenizer returns plain python lists here
            return SimpleBatch(encoded[0] if single else encoded)
        encoded = [torch.tensor(ids, dtype=torch.long) for ids in encoded]
        if padding:
            length = max(ids.shape[0] for ids in encoded)
            padded = torch.full((len(encoded), length), self.pad_token_id, dtype=torch.long)
            for i, ids in enumerate(encoded):
                padded[i, : ids.shape[0]] = ids
            return SimpleBatch(padded)
        return SimpleBatch(torch.stack(encoded))

    def batch_decode(self, sequences, skip_special_tokens=True):
        return ["decoded" for _ in sequences]


class SimpleBatch(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as error:  # pragma: no cover
            raise AttributeError(item) from error

    def __init__(self, input_ids):
        super().__init__(input_ids=input_ids)
        self.input_ids = input_ids


def build_tiny_llava():
    from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig, LlavaForConditionalGeneration

    config = LlavaConfig(
        text_config=LlamaConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=2048,
        ),
        vision_config=CLIPVisionConfig(
            hidden_size=768,
            image_size=224,
            patch_size=32,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
        ),
        mm_hidden_size=768,
        mm_projector_type="mlp2x_gelu",
        image_token_index=IMAGE_TOKEN_INDEX,
        vision_feature_layer=-2,
        vision_feature_select_strategy="default",
    )
    return LlavaForConditionalGeneration(config)


def build_wrapper():
    clip = load_clip(arch="ViT-B-32", pretrained=None, device="cpu")
    wrapper = LlavaOpenCLIP(
        hf_model=build_tiny_llava(),
        tokenizer=DummyTokenizer(),
        vision_tower=OpenCLIPVisionTower(clip),
        device="cpu",
        dtype=torch.float32,
    )
    return wrapper


def test_tokenizer_image_token_inserts_the_image_index():
    tokenizer = DummyTokenizer()
    ids = tokenizer_image_token("USER: <image>\nhello ASSISTANT:", tokenizer)
    assert IMAGE_TOKEN_INDEX in ids
    assert ids.count(IMAGE_TOKEN_INDEX) == 1


def test_image_tokens_are_replaced_by_projected_features():
    wrapper = build_wrapper()
    images = torch.rand(2, 3, 224, 224)
    prompts = [wrapper.build_prompt("Describe this image in detail.") for _ in range(2)]
    input_ids = wrapper.build_input_ids(prompts)
    embeds, attention_mask, labels = wrapper.prepare_inputs_labels_for_multimodal(input_ids, images)
    num_image_tokens = wrapper.num_image_tokens
    assert num_image_tokens == 49
    # 2 samples, image replaced by 49 tokens, embeddings of the LM dimension
    assert embeds.shape[0] == 2 and embeds.shape[2] == 32
    assert embeds.shape[1] > num_image_tokens
    assert attention_mask.shape == embeds.shape[:2]
    assert labels is None


def test_target_loss_is_differentiable_wrt_the_image():
    wrapper = build_wrapper()
    images = torch.rand(2, 3, 224, 224, requires_grad=True)
    prompts = [wrapper.build_prompt("What is in the image?") for _ in range(2)]
    targets = ["a cat", "a dog"]
    loss = wrapper.target_loss(images, prompts, targets)
    assert loss.shape == (2,)
    loss.sum().backward()
    assert images.grad is not None and images.grad.abs().sum() > 0


def test_generation_returns_strings():
    wrapper = build_wrapper()
    images = torch.rand(1, 3, 224, 224)
    prompts = [wrapper.build_prompt("Describe this image in detail.")]
    out = wrapper.generate(images, prompts, max_new_tokens=2)
    assert len(out) == 1 and isinstance(out[0], str)


def test_ensemble_attack_runs_on_a_real_llava_wrapper():
    """The half- and single-precision stages of the attack ensemble on LLaVA."""
    from robust_clip.attacks.lvlm_attack import EnsembleAttackConfig, LVLMEnsembleAttack

    wrapper = build_wrapper()
    images = torch.rand(1, 3, 224, 224)
    prompts = [wrapper.build_prompt("What is in the image?")]
    references = [["a cat", "a dog", "a bird", "a car", "a tree"]]
    config = EnsembleAttackConfig(
        eps="2/255",
        half_precision_iters=2,
        single_precision_iters=2,
        num_ground_truths=2,
        score_threshold=0.0,
        targeted_vqa=False,
        max_new_tokens=2,
    )
    metric = lambda generations, refs: [float(len(gen)) for gen, ref in zip(generations, refs)]
    attack = LVLMEnsembleAttack(wrapper, config, metric_fn=metric)
    result = attack.run(images, prompts, references, is_vqa=False)
    assert result.x_adv.shape == images.shape
    assert (result.x_adv - images).abs().max() <= 2 / 255 + 2 ** -11
    # the model must be restored to single precision after the half precision stage
    assert next(wrapper.parameters()).dtype == torch.float32
