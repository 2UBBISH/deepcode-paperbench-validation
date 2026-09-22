"""OpenFlamingo-9B with a (robust) OpenCLIP vision encoder.

The paper evaluates OpenFlamingo-9B in the zero-shot setting (*"the model is
prompted with some context text but without context images"*, Sec. 4.1) and
exchanges its frozen CLIP vision encoder for the robust FARE / TeCoA one, again
without retraining the LVLM.

The class wraps the ``open_flamingo`` package (see the addendum: *"The
OpenFlamingo model used by the paper is taken from this repository"*): the
PerceiverResampler and the language model (MPT-7B) of the released checkpoint are
kept frozen, only ``model.vision_encoder`` is replaced by
:class:`OpenFlamingoVisionAdapter`, a thin adapter around
:class:`~robust_clip.models.clip_encoder.CLIPEncoder`.

Install the optional dependency with::

    pip install git+https://github.com/mlfoundations/open_flamingo.git
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..clip_encoder import CLIPEncoder, load_clip
from ...utils.transforms import normalize_images
from .base import LVLM
from . import prompts as P

LOGGER = logging.getLogger(__name__)


class VisionEncoderOutput(dict):
    """Dict with attribute access -- mimics Hugging Face's ``ModelOutput``."""

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as error:  # pragma: no cover - defensive
            raise AttributeError(item) from error


class OpenFlamingoVisionAdapter(nn.Module):
    """Vision encoder interface expected by OpenFlamingo, backed by OpenCLIP."""

    def __init__(self, clip: CLIPEncoder, image_size: int = 224):
        super().__init__()
        self.clip = clip
        self.image_size = image_size

    @property
    def config(self):  # pragma: no cover - only used by some OpenFlamingo versions
        return self.clip.model.visual

    @property
    def dtype(self) -> torch.dtype:
        return next(self.clip.model.visual.parameters()).dtype

    def forward(self, images: torch.Tensor, *args, **kwargs) -> VisionEncoderOutput:
        """``images`` are already normalized by the OpenFlamingo image processor."""
        out = self.clip.vision_forward(images, layers=(-1,), input_is_normalized=True)
        last = out["hidden_states"][-1]
        return VisionEncoderOutput(
            last_hidden_state=last,
            hidden_states=(last,),
            pooler_output=last[:, 0],
        )


class OpenFlamingoWrapper(LVLM):
    """OpenFlamingo-9B with a swappable OpenCLIP vision tower."""

    def __init__(
        self,
        model,
        image_processor,
        tokenizer,
        vision_encoder: OpenFlamingoVisionAdapter,
        device: Union[str, torch.device] = "cuda",
        image_size: int = 224,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.model = model
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.vision_encoder = vision_encoder

        # exchange the frozen CLIP encoder of the checkpoint
        self.model.vision_encoder = vision_encoder
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.image_size = image_size
        self.image_mean = tuple(getattr(image_processor, "image_mean", (0.48145466, 0.4578275, 0.40821073)))
        self.image_std = tuple(getattr(image_processor, "image_std", (0.26862954, 0.26130258, 0.27577711)))
        self.media_token_id = getattr(self.model, "media_token_id", None)

    # ------------------------------------------------------------------
    # input preparation
    # ------------------------------------------------------------------
    def prepare_images(self, images: torch.Tensor) -> torch.Tensor:
        """``[B, 3, H, W]`` in ``[0, 1]`` -> normalized, resized OpenFlamingo input."""
        images = images.to(self.device)
        if images.shape[-1] != self.image_size or images.shape[-2] != self.image_size:
            images = F.interpolate(images, size=(self.image_size, self.image_size), mode="bicubic", align_corners=False)
        return normalize_images(images, self.image_mean, self.image_std)

    def _tokenize_prompt(self, prompt: str) -> torch.Tensor:
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        if self.media_token_id is not None:
            # OpenFlamingo uses <image> as the placeholder of the media token
            image_token_ids = self.tokenizer("<image>", add_special_tokens=False, return_tensors="pt").input_ids
            for value in image_token_ids.flatten().tolist():
                ids = torch.where(ids == value, torch.full_like(ids, self.media_token_id), ids)
        return ids

    def _build_lang_inputs(self, prompts: Sequence[str]):
        encodings = [self._tokenize_prompt(prompt).flatten() for prompt in prompts]
        length = max(ids.shape[0] for ids in encodings)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        lang_x = torch.full((len(encodings), length), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(encodings), length), dtype=torch.long)
        for i, ids in enumerate(encodings):
            lang_x[i, : ids.shape[0]] = ids
            attention_mask[i, : ids.shape[0]] = 1
        return lang_x.to(self.device), attention_mask.to(self.device)

    def _vision_x(self, images: torch.Tensor) -> torch.Tensor:
        """OpenFlamingo expects ``(B, T_img, F, C, H, W)``; we use one frame per image."""
        prepared = self.prepare_images(images)
        return prepared[:, None, None].to(next(self.model.parameters()).dtype)

    # ------------------------------------------------------------------
    # LVLM interface
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        prompts: Optional[Sequence[str]] = None,
        max_new_tokens: int = 32,
        num_beams: int = 3,
        do_sample: bool = False,
        **kwargs,
    ) -> List[str]:
        """Generate captions / short answers (OpenFlamingo uses beam search)."""
        if prompts is None:
            prompts = [P.OF_CAPTION_PROMPT] * images.shape[0]
        lang_x, attention_mask = self._build_lang_inputs(prompts)
        vision_x = self._vision_x(images)
        vision_x_mask = torch.ones((images.shape[0], 1), dtype=torch.long, device=self.device)
        outputs = self.model.generate(
            vision_x=vision_x,
            vision_x_mask=vision_x_mask,
            lang_x=lang_x,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            **kwargs,
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def target_loss(
        self,
        images: torch.Tensor,
        prompts: Sequence[str],
        targets: Sequence[str],
        reduction: str = "none",
    ) -> torch.Tensor:
        """Per-sample NLL of ``targets`` given image + prompt (differentiable in ``images``)."""
        lang_x, attention_mask = self._build_lang_inputs(prompts)
        target_ids = self.tokenizer(list(targets), add_special_tokens=False, return_tensors="pt", padding=True).input_ids
        target_ids = target_ids.to(self.device)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        full_ids = torch.cat([lang_x, target_ids], dim=1)
        labels = full_ids.clone()
        labels[:, : lang_x.shape[1]] = -100
        labels[full_ids == pad_id] = -100
        full_mask = torch.cat([attention_mask, (target_ids != pad_id).long()], dim=1)

        vision_x = self._vision_x(images)
        vision_x_mask = torch.ones((images.shape[0], 1), dtype=torch.long, device=self.device)
        outputs = self.model(
            vision_x=vision_x,
            vision_x_mask=vision_x_mask,
            lang_x=full_ids,
            attention_mask=full_mask,
            labels=labels,
        )
        logits = outputs.logits.float()
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
    def score(self, images, prompts, candidates, max_new_tokens: int = 32) -> List[float]:
        """CIDEr for captioning candidates (see :class:`robust_clip.eval.cider.Cider`)."""
        generations = self.generate(images, prompts, max_new_tokens=max_new_tokens)
        from ...eval.cider import Cider

        scorer = Cider()
        return [scorer.score([gen], list(cands)) for gen, cands in zip(generations, candidates)]


def load_open_flamingo(
    model_name: str = "openflamingo/OpenFlamingo-9B-vitl-mpt7b",
    lang_encoder_path: str = "anas-awadalla/mpt-7b",
    cross_attn_every_n_layers: int = 4,
    clip_arch: str = "ViT-L-14",
    clip_pretrained: str = "openai",
    checkpoint: Optional[str] = None,
    device: str = "cuda",
    image_size: int = 224,
) -> OpenFlamingoWrapper:
    """Load OpenFlamingo-9B and replace the vision encoder with the (robust) OpenCLIP one."""
    try:
        from huggingface_hub import hf_hub_download
        from open_flamingo import create_model_and_transforms
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "OpenFlamingo is required for the OpenFlamingo rows of Table 1/2:\n"
            "    pip install git+https://github.com/mlfoundations/open_flamingo.git"
        ) from error

    LOGGER.info("loading OpenFlamingo from %s", model_name)
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path=clip_arch,
        clip_vision_encoder_pretrained=clip_pretrained,
        lang_encoder_path=lang_encoder_path,
        tokenizer_path=lang_encoder_path,
        cross_attn_every_n_layers=cross_attn_every_n_layers,
    )
    checkpoint_path = hf_hub_download(model_name, "checkpoint.pt")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=False)

    clip = load_clip(arch=clip_arch, pretrained=clip_pretrained, checkpoint=checkpoint, device=device)
    vision_encoder = OpenFlamingoVisionAdapter(clip, image_size=image_size)
    return OpenFlamingoWrapper(
        model=model,
        image_processor=image_processor,
        tokenizer=tokenizer,
        vision_encoder=vision_encoder,
        device=device,
        image_size=image_size,
    )
