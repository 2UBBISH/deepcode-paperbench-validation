"""LLaVA-1.5 with an OpenCLIP vision tower.

LLaVA-1.5 7B consists of the frozen CLIP ViT-L/14 vision encoder, a two layer
MLP ("multi-modal projector") and Vicuna-1.5 (Llama-2) 7B.  The paper replaces
the vision encoder with a robust one *without any retraining of the LVLM*
(Sec. 3 / Sec. 4), which is exactly what this module implements: the language
model and the projector are frozen, only the vision tower changes.

Two details from the addendum are implemented explicitly:

1. LLaVA-1.5 7B is set up to use the **OpenAI CLIP ViT-L/14@224** vision encoder
   (instead of the default ViT-L/14@336),
2. LLaVA is run with the **OpenCLIP** implementation of CLIP instead of the
   Hugging Face one, hence :class:`OpenCLIPVisionTower`.

LLaVA consumes the *patch tokens of the second-to-last layer*
(``vision_feature_layer = -2``, ``vision_feature_select_strategy = 'default'``),
which is what :class:`~robust_clip.models.clip_encoder.CLIPEncoder` returns via
``vision_forward``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..clip_encoder import CLIPEncoder, load_clip
from ...utils.transforms import normalize_images
from .base import LVLM

LOGGER = logging.getLogger(__name__)

#: token index LLaVA uses as a placeholder for the image (see the LLaVA repository)
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"


@dataclass
class VisionTowerOutput:
    """Minimal stand-in for ``CLIPVisionModelOutput`` (hidden states + last hidden state)."""

    last_hidden_state: torch.Tensor
    hidden_states: Optional[tuple] = None
    pooler_output: Optional[torch.Tensor] = None


class OpenCLIPVisionTower(nn.Module):
    """Drop-in replacement of LLaVA's ``CLIPVisionTower`` backed by OpenCLIP.

    ``forward`` returns the *last* hidden state (Hugging Face convention, used by
    OpenFlamingo) and, if requested, the tuple of all block outputs -- exactly
    like the ``hidden_states`` of ``CLIPVisionModel``, so that LLaVA's
    ``vision_feature_layer=-2`` / ``vision_feature_select_strategy='default'``
    logic can be applied unchanged.
    """

    def __init__(
        self,
        clip: CLIPEncoder,
        select_layer: int = -2,
        select_feature: str = "patch",
    ):
        super().__init__()
        self.clip = clip
        self.select_layer = select_layer
        self.select_feature = select_feature

    # ------------------------------------------------------------------
    # attributes expected by LLaVA / OpenFlamingo
    # ------------------------------------------------------------------
    @property
    def image_size(self) -> int:
        return self.clip.image_size

    @property
    def image_mean(self):
        return self.clip.image_mean

    @property
    def image_std(self):
        return self.clip.image_std

    @property
    def config(self):
        return self.clip.model.visual

    def forward(
        self,
        images: torch.Tensor,
        output_hidden_states: bool = True,
        normalize: bool = False,
        **kwargs,
    ) -> VisionTowerOutput:
        """Run the tower.

        ``normalize=False`` (default) expects ``[0, 1]`` images and applies the
        CLIP normalization here; pass ``normalize=True`` for images that are
        already normalized (e.g. produced by an HF image processor).
        """
        if normalize:
            images = normalize_images(images, self.image_mean, self.image_std)
        # the attacker keeps the perturbation in float32 with values snapped onto
        # the grid of the attack precision; the network itself is cast to that
        # precision (see `robust_clip.attacks.lvlm_attack.model_precision`)
        param_dtype = next(self.parameters()).dtype
        if param_dtype != torch.float32:
            images = images.to(param_dtype)

        n_layers = self.clip.num_layers
        layers = list(range(0, n_layers))  # all block outputs, HF `hidden_states` convention
        out = self.clip.vision_forward(images, layers=layers, input_is_normalized=True)
        hidden_states = tuple(out["hidden_states"])
        last = hidden_states[-1]
        last_hidden_state = self.clip.model.visual.ln_post(last)
        return VisionTowerOutput(
            last_hidden_state=last_hidden_state,
            hidden_states=hidden_states if output_hidden_states else None,
            pooler_output=last_hidden_state[:, 0],
        )

    def feature_select(self, image_forward_outs: VisionTowerOutput) -> torch.Tensor:
        """LLaVA's feature selection (``select_layer`` + ``select_feature``)."""
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            return image_features[:, 1:]
        if self.select_feature == "cls_patch":
            return image_features
        raise ValueError(f"unexpected select_feature {self.select_feature!r}")

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """LLaVA style encoding: selected tokens of the ``select_layer``."""
        if self.select_layer == -1:
            return self.feature_select(
                VisionTowerOutput(last_hidden_state=self.clip.model.visual(images), hidden_states=None)
            )
        return self.feature_select(self.forward(images, output_hidden_states=True))


def tokenizer_image_token(prompt: str, tokenizer, image_token_index: int = IMAGE_TOKEN_INDEX) -> List[int]:
    """Tokenize a prompt that contains ``<image>`` (LLaVA repository convention).

    The ``<image>`` placeholder is replaced by ``image_token_index = -200``; all
    other tokens are the ones of the LLaVA tokenizer.
    """
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)]
    input_ids: List[int] = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])
    for chunk in prompt_chunks:
        input_ids.extend(chunk[offset:])
        offset = 0
        input_ids.append(image_token_index)
    return input_ids[:-1] if len(prompt_chunks) > 1 else input_ids


class LlavaOpenCLIP(LVLM):
    """LLaVA-1.5 with a (robust) OpenCLIP vision encoder.

    Parameters
    ----------
    hf_model:
        ``transformers.LlavaForConditionalGeneration``; its vision tower is
        *replaced* by the OpenCLIP one from ``vision_tower``.
    tokenizer:
        the LLaVA tokenizer (``AutoTokenizer`` of the LLaVA checkpoint).
    vision_tower:
        :class:`OpenCLIPVisionTower`, i.e. the robust (FARE / TeCoA) encoder.
    """

    def __init__(
        self,
        hf_model,
        tokenizer,
        vision_tower: OpenCLIPVisionTower,
        select_layer: int = -2,
        select_feature: str = "patch",
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.float16,
        system_prompt: Optional[str] = None,
    ):
        super().__init__()
        from .prompts import LLAVA_SYSTEM_PROMPT

        self.device = torch.device(device)
        self.dtype = dtype
        self.system_prompt = system_prompt or LLAVA_SYSTEM_PROMPT

        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.vision_tower = vision_tower
        self.select_layer = select_layer
        self.select_feature = select_feature

        # the projector of the frozen LLaVA checkpoint is kept
        self.projector = hf_model.model.multi_modal_projector
        self.language_model = hf_model.model.language_model
        self.embed_tokens = self.language_model.get_input_embeddings()

        for param in self.parameters():
            param.requires_grad_(False)
        self.to(self.device)
        self.eval()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @property
    def num_image_tokens(self) -> int:
        size = self.vision_tower.image_size
        patch = self.vision_tower.clip.model.visual.patch_size
        patch = patch[0] if isinstance(patch, (tuple, list)) else patch
        return (size // patch) ** 2

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """``[B, 3, H, W]`` in ``[0, 1]`` -> projected image tokens ``[B, N, D_llm]``."""
        selected = self.vision_tower.encode(images)  # [B, N, width]
        selected = selected.to(self.projector.linear_1.weight.dtype if hasattr(self.projector, "linear_1") else self.projector.weight.dtype)
        return self.projector(selected)

    def build_input_ids(self, prompts: Sequence[str]) -> torch.Tensor:
        """Tokenize prompts containing ``<image>`` into ``[B, T]`` ids (LLaVA convention)."""
        batch = []
        for prompt in prompts:
            ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX)
            batch.append(torch.tensor(ids, dtype=torch.long))
        length = max(len(ids) for ids in batch)
        pad_id = self.pad_token_id
        padded = torch.full((len(batch), length), pad_id, dtype=torch.long)
        for i, ids in enumerate(batch):
            padded[i, : len(ids)] = ids
        return padded

    @property
    def pad_token_id(self) -> int:
        """LLaVA uses the ``<unk>`` token as padding token."""
        return self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

    def _strip_padding(self, ids: torch.Tensor) -> torch.Tensor:
        """Drop the right padding of a single sequence."""
        keep = (ids != self.pad_token_id).nonzero(as_tuple=True)[0]
        return ids if keep.numel() == 0 else ids[: keep[-1] + 1]

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids: torch.Tensor,
        images: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
    ):
        """Merge image tokens and text tokens (LLaVA's ``prepare_inputs_labels_for_multimodal``).

        Returns ``(inputs_embeds, attention_mask, labels)`` where image tokens are
        replaced by the projected vision features (differentiable with respect to
        ``images``, which is what the attacks exploit).  The image occupies a
        single token in ``input_ids``: the language model sees exactly the
        ``<image>`` position replaced by the ``N`` projected patch tokens.
        """
        image_features = self.encode_images(images)  # [B, N, D]
        batch_size, num_image_tokens, _ = image_features.shape
        embed_dtype = self.embed_tokens.weight.dtype
        embeds, new_labels = [], []
        for i in range(batch_size):
            prompt_ids = self._strip_padding(input_ids[i])
            ids = prompt_ids
            labels = None
            if target_ids is not None:
                targets_i = self._strip_padding(target_ids[i])
                ids = torch.cat([prompt_ids, targets_i], dim=0)
                labels = torch.full_like(ids, -100)
                labels[prompt_ids.shape[0] :] = targets_i

            image_positions = (ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
            chunks = []
            label_chunks = []
            start = 0
            for position in image_positions.tolist():
                chunks.append(self.embed_tokens(ids[start:position]))
                if labels is not None:
                    label_chunks.append(labels[start:position])
                chunks.append(image_features[i].to(embed_dtype))
                if labels is not None:
                    label_chunks.append(torch.full((num_image_tokens,), -100, device=ids.device, dtype=torch.long))
                start = position + 1
            tail_ids = ids[start:]
            chunks.append(self.embed_tokens(tail_ids))
            if labels is not None:
                label_chunks.append(labels[start:])

            embeds.append(torch.cat(chunks, dim=0))
            if labels is not None:
                new_labels.append(torch.cat(label_chunks, dim=0))

        max_length = max(e.shape[0] for e in embeds)
        hidden = embeds[0].shape[-1]
        padded_embeds = torch.zeros((batch_size, max_length, hidden), device=embeds[0].device, dtype=embeds[0].dtype)
        attention_mask = torch.zeros((batch_size, max_length), device=embeds[0].device, dtype=torch.long)
        padded_labels = None
        if new_labels:
            padded_labels = torch.full((batch_size, max_length), -100, device=embeds[0].device, dtype=torch.long)
        for i, embed in enumerate(embeds):
            length = embed.shape[0]
            padded_embeds[i, :length] = embed
            attention_mask[i, :length] = 1
            if padded_labels is not None:
                padded_labels[i, :length] = new_labels[i]
        return padded_embeds, attention_mask, padded_labels

    def build_prompt(self, user_message: str) -> str:
        from .prompts import build_llava_prompt

        return build_llava_prompt(user_message, system_prompt=self.system_prompt)

    # ------------------------------------------------------------------
    # LVLM interface
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        prompts: Optional[Sequence[str]] = None,
        max_new_tokens: int = 32,
        num_beams: int = 1,
        do_sample: bool = False,
    ) -> List[str]:
        """Greedy generation (LLaVA's evaluation scripts use temperature 0)."""
        if prompts is None:
            prompts = [self.build_prompt(self.caption_prompt)] * images.shape[0]
        input_ids = self.build_input_ids(prompts).to(self.device)
        inputs_embeds, attention_mask, _ = self.prepare_inputs_labels_for_multimodal(input_ids, images)
        inputs_embeds = inputs_embeds.to(self.dtype)

        outputs = self.hf_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
            pad_token_id=self.pad_token_id,
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def forward_logits(
        self,
        images: torch.Tensor,
        prompts: Sequence[str],
        targets: Optional[Sequence[str]] = None,
    ):
        """Run the language model and return ``(logits, labels, attention_mask)``."""
        input_ids = self.build_input_ids(prompts).to(self.device)
        target_ids = None
        if targets is not None:
            target_ids = self.tokenizer(
                list(targets), add_special_tokens=False, return_tensors="pt", padding=True
            ).input_ids.to(self.device)
        inputs_embeds, attention_mask, labels = self.prepare_inputs_labels_for_multimodal(
            input_ids, images, target_ids=target_ids
        )
        outputs = self.hf_model(
            inputs_embeds=inputs_embeds.to(self.dtype),
            attention_mask=attention_mask,
        )
        return outputs.logits.float(), labels, attention_mask

    def target_loss(
        self,
        images: torch.Tensor,
        prompts: Sequence[str],
        targets: Sequence[str],
        reduction: str = "none",
    ) -> torch.Tensor:
        """Per-sample teacher-forced cross entropy of ``targets`` (differentiable)."""
        logits, labels, _ = self.forward_logits(images, prompts, targets)
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shift_labels.shape)
        counts = (shift_labels != -100).sum(dim=1).clamp_min(1)
        per_sample = losses.sum(dim=1) / counts
        if reduction == "mean":
            return per_sample.mean()
        if reduction == "sum":
            return per_sample.sum()
        return per_sample

    @torch.no_grad()
    def score(
        self,
        images: torch.Tensor,
        prompts: Sequence[str],
        candidates: Sequence[Sequence[str]],
        max_new_tokens: int = 32,
    ) -> List[float]:
        """Captioning-style scoring: returns the generated strings' CIDEr if available."""
        generations = self.generate(images, prompts, max_new_tokens=max_new_tokens)
        try:
            from ...eval.cider import Cider

            scorer = Cider()
            return [scorer.score([gen], list(cands)) for gen, cands in zip(generations, candidates)]
        except Exception:  # pragma: no cover - CIDEr needs reference captions
            return [float(gen in cands) for gen, cands in zip(generations, candidates)]


def load_llava_openclip(
    model_path: str = "liuhaotian/llava-v1.5-7b",
    arch: str = "ViT-L-14",
    pretrained: str = "openai",
    checkpoint: Optional[str] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    select_layer: int = -2,
    select_feature: str = "patch",
    system_prompt: Optional[str] = None,
) -> LlavaOpenCLIP:
    """Load LLaVA-1.5 and plug in the (robust) OpenCLIP vision encoder."""
    from transformers import AutoTokenizer, LlavaForConditionalGeneration

    LOGGER.info("loading LLaVA from %s", model_path)
    hf_model = LlavaForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    # LLaVA uses the <unk> token for padding (see the LLaVA repository)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"

    clip = load_clip(arch=arch, pretrained=pretrained, checkpoint=checkpoint, device=device)
    vision_tower = OpenCLIPVisionTower(clip, select_layer=select_layer, select_feature=select_feature)
    return LlavaOpenCLIP(
        hf_model=hf_model,
        tokenizer=tokenizer,
        vision_tower=vision_tower,
        select_layer=select_layer,
        select_feature=select_feature,
        device=device,
        dtype=dtype,
        system_prompt=system_prompt,
    )
