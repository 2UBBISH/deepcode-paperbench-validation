"""OpenFlamingo-9B with a swappable OpenCLIP vision encoder.

The paper uses OpenFlamingo-9B (``ViT-L/14`` + MPT-7B) in the zero-shot setting,
i.e. the model is prompted with (context) text but **without** context images
(Sec. 4.1).  As for LLaVA, the frozen CLIP vision encoder is replaced by the
(robust) FARE / TeCoA encoder; the perceiver resampler and the language model
are left untouched.

The vision encoder in OpenFlamingo consumes the **output of all tokens** of the
CLIP vision transformer (App. B.1), so the wrapper returns the full token
sequence ``B x (1 + N) x width`` of the ViT.

Notes
-----
``open_flamingo`` has changed its internal APIs several times; the loader below
targets the released ``openflamingo/OpenFlamingo-9B-vitl-mpt7b`` checkpoint and
the ``open_flamingo`` package that ships with it, and falls back gracefully if
the vision encoder cannot be swapped automatically.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models import CLIPImageEncoder, load_clip_encoder
from ..utils.common import LOGGER
from .base import LVLM


#: Zero-shot prompts of the OpenFlamingo evaluation protocol.
OPENFLAMINGO_PROMPTS = {
    "coco_caption": "<image>",
    "flickr_caption": "<image>",
    "vqav2": "<image>Question: {question} Short answer:",
    "textvqa": "<image>Question: {question} Short answer:",
    "pope": "<image>Question: {question} Short answer:",
}


def build_openflamingo_prompt(question: Optional[str] = None, task: str = "vqav2") -> str:
    template = OPENFLAMINGO_PROMPTS.get(task, "<image>")
    return template.format(question=question) if question is not None else template


class OpenClipFlamingoVisionEncoder(nn.Module):
    """Drop-in replacement for OpenFlamingo's frozen CLIP vision encoder.

    OpenFlamingo computes the visual features with
    ``self.vision_encoder(vision_x)[1]``, i.e. it takes the *second* element of
    the vision encoder's output -- exactly the ``(pooled, tokens)`` tuple that
    ``open_clip``'s ``VisionTransformer`` returns when ``output_tokens=True``.
    This wrapper reproduces that contract: element 1 is the ``B x N x width``
    token sequence of the last block (no class token), which the perceiver
    resampler consumes.

    Note that :meth:`robust_clip.lvlm.openflamingo.patch_flamingo_for_attacks`
    must be called before the model is used for *gradient-based* attacks:
    upstream's ``_encode_vision_x`` computes the visual features inside
    ``torch.no_grad()``.
    """

    def __init__(self, encoder: CLIPImageEncoder, layer: Optional[int] = None):
        super().__init__()
        self.encoder = encoder
        self.layer = layer  # ``None`` -> last block
        self.output_tokens = True  # API flag used by open_flamingo
        # The runner feeds pixel-space images (see :class:`LVLM`), so the tower
        # applies the CLIP normalisation itself.
        self.pixel_space_inputs = True

    def forward(self, images: torch.Tensor, **kwargs) -> torch.Tensor:
        param_dtype = next(self.parameters()).dtype
        x = images.to(dtype=param_dtype) if images.dtype != param_dtype else images
        if self.pixel_space_inputs:
            x = self.encoder.normalize(x)
        self.encoder.interpolate_pos_embed_if_needed(x)
        out = self.encoder.visual.forward_intermediates(
            x, output_fmt="NCHW", output_extra_tokens=True, intermediates_only=True,
        )
        # ``None`` -> the last block, as in open_clip's ``output_tokens=True``
        layer = -1 if self.layer is None else self.layer
        spatial = out["image_intermediates"][layer]
        b, c, h, w = spatial.shape
        tokens = spatial.reshape(b, c, h * w).permute(0, 2, 1)
        with torch.no_grad():
            pooled = self.encoder.encode_image(x, already_normalized=True)
        # ``(pooled, tokens)``: upstream OpenFlamingo uses element 1
        return pooled, tokens

    @property
    def patch_grid(self) -> tuple:
        visual = self.encoder.visual
        size = visual.image_size if hasattr(visual, "image_size") else 224
        if isinstance(size, (tuple, list)):
            size = size[-1]
        return size // self.encoder.patch_size, size // self.encoder.patch_size

    @property
    def hidden_size(self) -> int:
        return self.encoder.width


def replace_openflamingo_vision_encoder(model, encoder: CLIPImageEncoder, layer: Optional[int] = None):
    """Swap ``model.vision_encoder`` for the OpenCLIP implementation."""
    new_encoder = OpenClipFlamingoVisionEncoder(encoder, layer=layer)
    if hasattr(model, "vision_encoder"):
        # ``Flamingo`` stores the vision width as ``vis_dim`` (the perceiver
        # resampler is built with it); older forks call it ``vision_encoder_dim``.
        old_dim = getattr(model, "vis_dim", None) or getattr(model, "vision_encoder_dim", None)
        model.vision_encoder = new_encoder
        if old_dim is not None and old_dim != encoder.width:
            LOGGER.warning(
                "vision encoder width changed (%s -> %s); check the perceiver resampler",
                old_dim,
                encoder.width,
            )
        return model
    raise AttributeError("could not find `vision_encoder` on the OpenFlamingo model")


def patch_flamingo_for_attacks(model):
    """Compute the visual features *with* gradients.

    ``open_flamingo``'s ``Flamingo._encode_vision_x`` wraps the vision encoder
    in ``torch.no_grad()``, which is fine for generation but makes white-box
    attacks on the image impossible.  The paper's LVLM evaluations are exactly
    such attacks (Sec. 4.1), so we install an equivalent implementation without
    the ``no_grad`` context.  Everything else (rearranging, the perceiver
    resampler and conditioning the language-model layers) is unchanged.
    """
    from einops import rearrange  # dependency of open_flamingo

    def encode_vision_x(self, vision_x):
        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, t, f = vision_x.shape[:3]
        assert f == 1, "Only single frame supported"
        flat = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        tokens = self.vision_encoder(flat)[1]
        tokens = rearrange(tokens, "(b T F) v d -> b T F v d", b=b, T=t, F=f)
        tokens = self.perceiver(tokens)
        for layer_module in self.lang_encoder._get_decoder_layers():
            layer_module.condition_vis_x(tokens)

    import types

    model._encode_vision_x = types.MethodType(encode_vision_x, model)
    model.attacks_use_gradients = True
    LOGGER.info("patched OpenFlamingo so that vision features keep their gradient")
    return model


class OpenFlamingoRunner(LVLM):
    """Minimal evaluation/attack interface around an OpenFlamingo model."""

    def __init__(
        self,
        model,
        image_processor,
        tokenizer,
        encoder: CLIPImageEncoder,
        image_size: int = 224,
        device: str = "cuda",
    ):
        self.model = model
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.image_size = image_size
        self.name = "OpenFlamingo-9B"
        self._dtype = next(encoder.parameters()).dtype
        self.device_str = device

    # ------------------------------------------------------------- factories
    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str = "openflamingo/OpenFlamingo-9B-vitl-mpt7b",
        clip_arch: str = "ViT-L-14",
        clip_pretrained: str = "openai",
        clip_checkpoint: Optional[str] = None,
        clip_checkpoint_key: Optional[str] = None,
        image_size: int = 224,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        lang_encoder_path: str = "anas-awadalla/mpt-7b",
        cross_attn_every_n_layers: int = 4,
        local_checkpoint_path: Optional[str] = None,
    ) -> "OpenFlamingoRunner":
        from open_flamingo import create_model_and_transforms

        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path=clip_arch,
            clip_vision_encoder_pretrained=clip_pretrained,
            lang_encoder_path=lang_encoder_path,
            tokenizer_path=lang_encoder_path,
            cross_attn_every_n_layers=cross_attn_every_n_layers,
        )
        if local_checkpoint_path is None:
            from huggingface_hub import hf_hub_download

            local_checkpoint_path = hf_hub_download(checkpoint, "checkpoint.pt")
        state_dict = torch.load(local_checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

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
        model = replace_openflamingo_vision_encoder(model, encoder)
        model = patch_flamingo_for_attacks(model)
        model = model.to(device=device)
        if dtype != torch.float32:
            # the language model stays in its own precision, the vision tower is
            # what the attacks need at reduced precision
            encoder.visual.to(dtype=dtype)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        return cls(model, image_processor, tokenizer, encoder, image_size=image_size, device=device)

    # -------------------------------------------------------------- plumbing
    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def set_precision(self, dtype: torch.dtype) -> None:
        self._dtype = dtype
        self.encoder.to(dtype=dtype)
        self.encoder.visual.to(dtype=dtype)
        for param in self.model.parameters():
            param.data = param.data.to(dtype) if param.is_floating_point() else param.data

    def to(self, device):
        self.model = self.model.to(device)
        self.encoder.to(device)
        return self

    def preprocess(self, images: Sequence) -> torch.Tensor:
        """Return pixel-space ``[0, 1]`` images of size ``image_size``."""
        from PIL import Image
        from torchvision.transforms.functional import center_crop, pil_to_tensor, resize

        tensors = []
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
                tensor = resize(tensor, [int(round(h * scale)), int(round(w * scale))], antialias=True)
            tensors.append(center_crop(tensor, [self.image_size, self.image_size]))
        return torch.stack(tensors, dim=0)

    def _tokenize(self, texts: Sequence[str], device=None, max_length: int = 128) -> dict:
        enc = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_length,
        )
        if device is not None:
            enc = {k: v.to(device) for k, v in enc.items()}
        return enc

    def _forward_logits(self, images: torch.Tensor, enc: dict) -> torch.Tensor:
        images = images.to(device=self.device, dtype=self._dtype)
        out = self.model(
            vision_x=images.unsqueeze(1).unsqueeze(1) if images.dim() == 4 else images,
            lang_x=enc["input_ids"],
            attention_mask=enc.get("attention_mask"),
        )
        logits = getattr(out, "logits", None)
        if logits is None:
            logits = out[0] if isinstance(out, (tuple, list)) else out
        return logits.float()

    # ------------------------------------------------------------------- NLL
    def nll(
        self,
        images: torch.Tensor,
        prompts: Sequence[str],
        continuations: Sequence[str],
        reduction: str = "mean",
    ) -> torch.Tensor:
        device = self.device
        full_texts = [p + c for p, c in zip(prompts, continuations)]
        full = self._tokenize(full_texts, device=device)
        prompt_enc = self._tokenize(list(prompts), device=device)
        input_ids = full["input_ids"]
        labels = input_ids.clone()
        prompt_lengths = prompt_enc["attention_mask"].sum(dim=1).to(device)
        positions = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
        labels[positions < prompt_lengths.unsqueeze(1)] = -100
        if "attention_mask" in full:
            labels[full["attention_mask"] == 0] = -100

        logits = self._forward_logits(images, full)
        token_losses = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
            reduction="none",
        )
        valid = (labels[:, 1:] != -100).reshape(-1).float()
        per_sample = (token_losses * valid).view(labels.shape[0], -1).sum(dim=1)
        per_sample = per_sample / valid.view(labels.shape[0], -1).sum(dim=1).clamp(min=1.0)
        return per_sample if reduction == "none" else per_sample.mean()

    # -------------------------------------------------------------- generate
    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        prompts: Sequence[str],
        max_new_tokens: int = 32,
        **generation_kwargs,
    ) -> List[str]:
        device = self.device
        enc = self._tokenize(list(prompts), device=device)
        images = images.to(device=device, dtype=self._dtype)
        vision_x = images.unsqueeze(1).unsqueeze(1) if images.dim() == 4 else images
        generated = self.model.generate(
            vision_x=vision_x,
            lang_x=enc["input_ids"],
            attention_mask=enc.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            **generation_kwargs,
        )
        if isinstance(generated, (tuple, list)):
            generated = generated[0]
        generated = generated[:, enc["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)
