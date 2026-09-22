"""LLaVA-1.5 with an OpenCLIP vision encoder.

The paper replaces the frozen CLIP vision encoder inside LLaVA-1.5 7B by the
(robust) FARE / TeCoA encoder; no part of LLaVA is retrained.  The addendum
specifies how the original LLaVA-1.5 setup differs from the HuggingFace default:

* the vision encoder is the **OpenAI CLIP ViT-L/14@224** model
  (the HF default is ViT-L/14@336);
* the code was modified so that LLaVA uses the **OpenCLIP** implementation of
  CLIP instead of the HuggingFace one;
* LLaVA consumes the **second-to-last layer** patch tokens
  (``vision_feature_layer = -2``, ``vision_feature_select_strategy = 'default'``).

This module provides

* :class:`OpenClipVisionTower` -- a drop-in replacement for LLaVA's
  ``CLIPVisionTower`` that returns ``B x N x width`` patch tokens produced by an
  arbitrary :class:`~robust_clip.models.CLIPImageEncoder` (so a FARE encoder can
  be plugged in by simply constructing it from a checkpoint), and
* :class:`LlavaOpenClip` -- ``LlavaForConditionalGeneration`` with that tower,
  plus the pixel-space preprocessing / prompting / NLL utilities needed by the
  attacks.
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models import CLIPImageEncoder, load_clip_encoder
from .base import LVLM


# --------------------------------------------------------------------------- #
#                        LLaVA-1.5 prompt templates                            #
# --------------------------------------------------------------------------- #
DEFAULT_SYSTEM_PROMPT = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the human's questions."
)

#: Task specific questions, following the LLaVA-1.5 evaluation scripts.
TASK_QUESTIONS = {
    "coco_caption": "Describe the image concisely.",
    "flickr_caption": "Describe the image concisely.",
    "vqav2": "{question}\nAnswer the question using a single word or phrase.",
    "textvqa": "{question}\nAnswer the question using a single word or phrase.",
    "pope": "{question}\nAnswer the question using a single word or phrase.",
    "scienceqa": "{question}\nAnswer with the option's letter from the given choices directly.",
}


def build_llava_prompt(
    question: str,
    task: str = "vqav2",
    system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT,
) -> str:
    """The ``USER: <image> ... ASSISTANT:`` template of LLaVA-1.5."""
    template = TASK_QUESTIONS.get(task, "{question}")
    body = template.format(question=question)
    prefix = f"{system_prompt} " if system_prompt else ""
    return f"{prefix}USER: <image>\n{body}\nASSISTANT:"


# --------------------------------------------------------------------------- #
#                              vision tower                                    #
# --------------------------------------------------------------------------- #
class OpenClipVisionTower(nn.Module):
    """LLaVA vision tower backed by :mod:`open_clip`.

    ``forward`` emulates the HuggingFace ``CLIPVisionModel``: it returns a
    ``BaseModelOutputWithPooling`` whose ``hidden_states`` follow HF's
    convention (``hidden_states[0]`` is the embedding output, ``hidden_states[i]``
    the output of block ``i - 1``).  ``LlavaForConditionalGeneration`` then picks
    ``hidden_states[vision_feature_layer]`` with ``vision_feature_layer = -2``
    and drops the class token -- i.e. it consumes exactly the
    :meth:`~robust_clip.models.CLIPImageEncoder.patch_tokens` of the
    second-to-last block, computed by OpenCLIP.

    Reproducing HF's interface (rather than re-implementing LLaVA's feature
    selection) keeps this module compatible with every ``transformers`` version
    that ships a native LLaVA implementation.

    The tower expects **pixel-space** images in ``[0, 1]`` and applies the CLIP
    normalisation internally.  Keeping the normalisation inside the tower is
    what allows the attacks to be computed with respect to the non-normalised
    inputs (see the addendum).
    """

    def __init__(
        self,
        encoder: CLIPImageEncoder,
        select_layer: int = -2,
        select_feature: str = "patch",
        image_size: Optional[int] = None,
        pixel_space_inputs: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.select_layer = select_layer
        self.select_feature = select_feature
        self.is_loaded = True
        self.hidden_size = encoder.width
        self.patch_size = encoder.patch_size
        raw_size = image_size if image_size is not None else getattr(encoder.visual, "image_size", 224)
        if isinstance(raw_size, (tuple, list)):
            raw_size = raw_size[-1]
        self.image_size = int(raw_size)
        self.num_layers = len(encoder.visual.transformer.resblocks)
        #: ``True`` (default): the tower receives *pixel-space* images in
        #: ``[0, 1]`` and applies the CLIP normalisation itself, which is what
        #: lets the attacks perturb the non-normalised inputs.  Set to ``False``
        #: if the caller feeds images that LLaVA's own processor already
        #: normalised.
        self.pixel_space_inputs = pixel_space_inputs

    # LLaVA reads these attributes from the tower.
    @property
    def config(self):
        from types import SimpleNamespace

        return SimpleNamespace(
            hidden_size=self.hidden_size,
            image_size=self.image_size,
            patch_size=self.patch_size,
            model_type="openclip",
        )

    def feature_select(self, tokens: torch.Tensor, cls_token: Optional[torch.Tensor] = None):
        if self.select_feature == "patch":
            return tokens
        if self.select_feature == "cls_patch":
            return torch.cat([cls_token, tokens], dim=1) if cls_token is not None else tokens
        raise ValueError(f"unknown select_feature '{self.select_feature}'")

    @property
    def num_image_tokens(self) -> int:
        size = self.image_size[-1] if isinstance(self.image_size, (tuple, list)) else self.image_size
        patch = self.patch_size[-1] if isinstance(self.patch_size, (tuple, list)) else self.patch_size
        return (int(size) // int(patch)) ** 2

    def forward(self, images: torch.Tensor, output_hidden_states: bool = True, **kwargs):
        """Return ``hidden_states`` in the HuggingFace ``CLIPVisionModel`` layout."""
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        param_dtype = next(self.parameters()).dtype
        x = images.to(dtype=param_dtype) if images.dtype != param_dtype else images
        if self.pixel_space_inputs:
            x = self.encoder.normalize(x)
        self.encoder.interpolate_pos_embed_if_needed(x)
        out = self.encoder.visual.forward_intermediates(
            x,
            output_fmt="NCHW",
            output_extra_tokens=True,
            intermediates_only=True,
        )
        spatial = out["image_intermediates"]        # list of B x C x H x W
        prefix = out["image_intermediates_prefix"]  # list of B x 1 x C
        block_states = []
        for cls_token, feature_map in zip(prefix, spatial):
            b, c, h, w = feature_map.shape
            patches = feature_map.reshape(b, c, h * w).permute(0, 2, 1)
            block_states.append(torch.cat([cls_token.to(patches.dtype), patches], dim=1))
        # ``hidden_states[0]`` is the embedding output; the block outputs follow.
        hidden_states = (None,) + tuple(block_states)
        last_hidden_state = hidden_states[self.select_layer]
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=None,
            hidden_states=hidden_states if output_hidden_states else None,
        )

    def patch_tokens(self, images: torch.Tensor, include_cls: bool = False) -> torch.Tensor:
        """Direct access to the tokens used by LLaVA (``B x N x width``)."""
        param_dtype = next(self.parameters()).dtype
        x = images.to(dtype=param_dtype) if images.dtype != param_dtype else images
        return self.encoder.patch_tokens(
            x, layer=self.select_layer, include_cls=include_cls
        )

    # ------------------------------------------------------------------ misc
    def load_model(self, checkpoint: str) -> None:
        self.encoder.load_finetuned(checkpoint)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


def replace_vision_encoder(model: nn.Module, encoder: CLIPImageEncoder) -> nn.Module:
    """Swap LLaVA's vision tower for ``encoder`` and keep everything else frozen."""
    tower = OpenClipVisionTower(encoder)
    # ``LlavaForConditionalGeneration`` keeps the actual module inside
    # ``model.model`` (a ``LlavaModel``) -- that is the object whose ``forward``
    # calls ``self.vision_tower``, so this is the attribute that must change.
    candidates = []
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "vision_tower"):
        candidates.append(inner)
    if hasattr(model, "vision_tower"):
        candidates.append(model)
    if not candidates:
        raise AttributeError("could not find a `vision_tower` attribute on the model")
    for owner in candidates:
        try:
            owner.vision_tower = tower
        except AttributeError:  # read-only property on some versions
            continue
    if hasattr(model, "config") and hasattr(model.config, "vision_feature_layer"):
        model.config.vision_feature_layer = -2
        model.config.vision_feature_select_strategy = "default"
    if not isinstance(inner_vision_tower(model), OpenClipVisionTower):
        raise RuntimeError("failed to install the OpenCLIP vision tower")
    return model


def inner_vision_tower(model) -> Optional[nn.Module]:
    """The module that ``LlavaModel.forward`` will actually call."""
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "vision_tower"):
        return inner.vision_tower
    return getattr(model, "vision_tower", None)


# --------------------------------------------------------------------------- #
#                                 the model                                    #
# --------------------------------------------------------------------------- #
class LlavaOpenClip(LVLM):
    """LLaVA-1.5 (`LlavaForConditionalGeneration`) with an OpenCLIP tower.

    Only the language model and the multimodal projector are taken from the
    released LLaVA-1.5 checkpoint; the vision encoder is instantiated from
    ``open_clip`` so that FARE / TeCoA checkpoints can be loaded into it (which
    is the core intervention of the paper).
    """

    def __init__(self, model, processor, encoder: CLIPImageEncoder, image_size: int = 224):
        self.model = model
        self.processor = processor
        self.encoder = encoder
        self.image_size = image_size
        self.name = "LLaVA-1.5-7B"
        self._dtype = next(encoder.parameters()).dtype

    # ------------------------------------------------------------- factories
    @classmethod
    def from_pretrained(
        cls,
        llava_path: str = "llava-hf/llava-1.5-7b-hf",
        clip_arch: str = "ViT-L-14",
        clip_pretrained: str = "openai",
        clip_checkpoint: Optional[str] = None,
        clip_checkpoint_key: Optional[str] = None,
        image_size: int = 224,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        low_cpu_mem_usage: bool = True,
        local_files_only: bool = False,
    ) -> "LlavaOpenClip":
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        model = LlavaForConditionalGeneration.from_pretrained(
            llava_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
            local_files_only=local_files_only,
        )
        processor = AutoProcessor.from_pretrained(
            llava_path, local_files_only=local_files_only
        )
        encoder = load_clip_encoder(
            arch=clip_arch,
            pretrained=clip_pretrained,
            image_size=image_size,
            checkpoint=clip_checkpoint,
            checkpoint_key=clip_checkpoint_key,
            feature="projected_class_token",
            device=device,
            dtype=dtype,
            freeze_text=True,
        )
        model = replace_vision_encoder(model, encoder)
        model = model.to(device=device)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        return cls(model, processor, encoder, image_size=image_size)

    # -------------------------------------------------------------- plumbing
    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def tokenizer(self):
        # ``processor`` is a ``LlavaProcessor`` (which owns a tokenizer) when the
        # model is loaded from the hub; a bare tokenizer works as well.
        return getattr(self.processor, "tokenizer", self.processor)

    @property
    def image_token_id(self) -> int:
        return int(self.model.config.image_token_index)

    def set_precision(self, dtype: torch.dtype) -> None:
        """Switch between the half-precision and single-precision attack stages."""
        self._dtype = dtype
        self.model.to(dtype=dtype)
        self.encoder.to(dtype=dtype)
        self.encoder.visual.to(dtype=dtype)

    def to(self, device):
        self.model = self.model.to(device)
        self.encoder.to(device)
        return self

    # ---------------------------------------------------------- preprocessing
    def preprocess(self, images: Sequence) -> torch.Tensor:
        """Resize/crop to ``image_size`` and return a pixel-space tensor.

        ``images`` may be PIL images or CHW float tensors in ``[0, 1]``.
        No normalisation is applied here: the OpenCLIP tower does that.
        """
        from PIL import Image
        from torchvision.transforms.functional import center_crop, pil_to_tensor, resize

        tensors: List[torch.Tensor] = []
        for image in images:
            if isinstance(image, torch.Tensor):
                tensor = image.detach().float().cpu()
            else:
                if image.mode != "RGB":
                    image = image.convert("RGB")
                tensor = pil_to_tensor(image).float() / 255.0
            _, h, w = tensor.shape
            scale = self.image_size / min(h, w)
            if scale < 1:
                tensor = resize(
                    tensor, [int(round(h * scale)), int(round(w * scale))], antialias=True
                )
            tensor = center_crop(tensor, [self.image_size, self.image_size])
            tensors.append(tensor)
        return torch.stack(tensors, dim=0)

    def tokenize_prompts(self, prompts: Sequence[str], device=None) -> dict:
        """Tokenise LLaVA prompts, expanding ``<image>`` to one token per patch.

        LLaVA's processor replaces the single ``<image>`` placeholder by
        ``num_image_tokens`` copies of the image token, which the model then
        overwrites with the projected vision features.  We do the same here so
        that the number of placeholders always matches the number of patch
        tokens produced by the (possibly swapped-in) vision encoder.
        """
        expanded = [
            prompt.replace("<image>", "<image>" * self.num_image_tokens, 1) for prompt in prompts
        ]
        enc = self.tokenizer(
            expanded,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        )
        if device is not None:
            enc = {k: v.to(device) for k, v in enc.items()}
        return enc

    @property
    def num_image_tokens(self) -> int:
        return int(self.vision_tower.num_image_tokens)

    @property
    def vision_tower(self) -> OpenClipVisionTower:
        """Locate the (replaced) vision tower, which HF keeps in different spots."""
        for owner in (getattr(self.model, "model", None), self.model):
            if owner is not None and hasattr(owner, "vision_tower"):
                tower = getattr(owner, "vision_tower")
                if isinstance(tower, OpenClipVisionTower):
                    return tower
        raise AttributeError("no OpenCLIP vision tower found on the model")

    @property
    def image_token(self) -> str:
        return "<image>"

    # ------------------------------------------------------------------- NLL
    def nll(
        self,
        images: torch.Tensor,
        prompts: Sequence[str],
        continuations: Sequence[str],
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Per-sample negative log-likelihood of the continuation tokens.

        This is the quantity maximised by the untargeted attacks and minimised
        by the targeted ones (Sec. 4.1 / App. B.6).
        """
        device = self.device
        images = images.to(device=device, dtype=self._dtype)
        full_texts = [p + c for p, c in zip(prompts, continuations)]
        # NOTE: the prompts contain the ``<image>`` placeholder, which has to be
        # expanded to one token per patch before they are fed to the model.
        full = self.tokenize_prompts(full_texts, device=device)
        prompt_enc = self.tokenize_prompts(list(prompts), device=device)
        input_ids = full["input_ids"].to(device)
        attention_mask = full["attention_mask"].to(device)
        labels = input_ids.clone()

        prompt_lengths = prompt_enc["attention_mask"].sum(dim=1).to(device)
        positions = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
        labels[positions < prompt_lengths.unsqueeze(1)] = -100
        labels[attention_mask == 0] = -100

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=images,
        )
        logits = outputs.logits.float()
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        token_losses = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shift_labels.shape)
        valid = (shift_labels != -100).float()
        per_sample = (token_losses * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        if reduction == "none":
            return per_sample
        if reduction == "sum":
            return (token_losses * valid).sum(dim=1)
        return per_sample

    def full_text_nll(
        self,
        images: torch.Tensor,
        texts: Sequence[str],
        context_length: int = 0,
        reduction: str = "mean",
    ):
        """NLL of ``texts`` (optionally only the tokens after ``context_length``)."""
        device = self.device
        images = images.to(device=device, dtype=self._dtype)
        enc = self.tokenizer(list(texts), return_tensors="pt", padding=True, add_special_tokens=True)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        labels = input_ids.clone()
        if context_length > 0:
            positions = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
            labels[positions < context_length] = -100
        labels[attention_mask == 0] = -100
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=images)
        logits = outputs.logits.float()
        token_losses = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
            reduction="none",
        )
        valid = (labels[:, 1:] != -100).reshape(-1).float()
        per_sample = (token_losses * valid).view(labels.shape[0], -1).sum(dim=1)
        per_sample = per_sample / valid.view(labels.shape[0], -1).sum(dim=1).clamp(min=1.0)
        if reduction == "none":
            return per_sample
        return per_sample.mean()

    # -------------------------------------------------------------- generate
    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        prompts: Sequence[str],
        max_new_tokens: int = 64,
        num_beams: int = 1,
        do_sample: bool = False,
        **generation_kwargs,
    ) -> List[str]:
        device = self.device
        images = images.to(device=device, dtype=self._dtype)
        enc = self.tokenize_prompts(prompts, device=device)
        # Wrap the text prompts with the ``<image>`` placeholder expected by LLaVA
        # when the caller has not already included it.
        input_ids = enc["input_ids"]
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=enc["attention_mask"],
            pixel_values=images,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            **generation_kwargs,
        )
        generated = outputs[:, input_ids.shape[1]:]
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)


def load_llava_1p5(
    clip_checkpoint: Optional[str] = None,
    clip_arch: str = "ViT-L-14",
    image_size: int = 224,
    llava_path: str = "llava-hf/llava-1.5-7b-hf",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    **kwargs,
) -> LlavaOpenClip:
    """Convenience loader used by the evaluation scripts.

    ``clip_checkpoint`` is a FARE / TeCoA checkpoint (``None`` keeps the
    original, non-robust CLIP encoder).
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return LlavaOpenClip.from_pretrained(
        llava_path=llava_path,
        clip_arch=clip_arch,
        clip_pretrained="openai",
        clip_checkpoint=clip_checkpoint,
        image_size=image_size,
        device=device,
        dtype=dtype,
        **kwargs,
    )
