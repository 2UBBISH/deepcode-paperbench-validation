"""OpenCLIP based CLIP encoders.

The paper uses the OpenCLIP implementation of CLIP for *all* experiments and
explicitly notes that LLaVA was adapted to work with the OpenCLIP implementation
instead of the Hugging Face one (see the addendum).  This module therefore
provides a thin, dependency-light wrapper around :mod:`open_clip` that

* exposes the projected image / text embeddings used for zero-shot
  classification (Sec. 4.3, ``phi(x)`` and ``psi(t)`` of Sec. 3),
* exposes the *intermediate* vision tokens that the LVLMs consume: LLaVA operates
  on the patch tokens of the second-to-last layer (App. B.1: "LLaVA operates on
  second-last layer outputs"), OpenFlamingo on the full last layer output,
* can load the checkpoints produced by ``robust_clip.training`` (i.e. the FARE /
  TeCoA fine-tuned vision encoders).

Vision encoder usage in the paper:

* LLaVA-1.5 7B uses the **OpenAI CLIP ViT-L/14@224** encoder (the addendum
  points out that LLaVA is switched from the default ViT-L/14@336 to the 224
  model), i.e. ``arch='ViT-L-14', pretrained='openai'``.
* OpenFlamingo-9B uses the **LAION-2B CLIP ViT-L/14** encoder of the original
  OpenFlamingo release, i.e. ``arch='ViT-L-14', pretrained='laion2b_s32b_b82k'``.
* The ablations of App. B.3-B.5 use ViT-B/32 (``arch='ViT-B-32'``).
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn

import open_clip

from ..utils.transforms import normalize_images

LOGGER = logging.getLogger(__name__)

#: (arch, pretrained) pairs used in the paper.
OPENAI_ARCHS = {"ViT-L-14": "openai", "ViT-L-14-336": "openai", "ViT-B-32": "openai"}
LAION_ARCHS = {"ViT-L-14": "laion2b_s32b_b82k", "ViT-B-32": "laion2b_s34b_b79k"}

#: number of transformer blocks of the supported architectures
VIT_LAYERS = {
    "ViT-B-32": 12,
    "ViT-B-16": 12,
    "ViT-L-14": 24,
    "ViT-L-14-336": 24,
    "ViT-H-14": 32,
}


def _strip_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove the wrappers (``module.``, ``model.``, ``_orig_mod.``) of saved checkpoints."""
    out = {}
    for key, value in state_dict.items():
        for prefix in ("module.", "_orig_mod."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        if key.startswith("model."):
            key = key[len("model."):]
        out[key] = value
    return out


class CLIPEncoder(nn.Module):
    """Wrapper around an :class:`open_clip.model.CLIP` model.

    Parameters
    ----------
    arch, pretrained:
        passed to :func:`open_clip.create_model_and_transforms`.
    checkpoint:
        optional path to a fine-tuned checkpoint (the output of
        ``robust_clip.training.train``); only the CLIP weights are restored.
    device:
        device to move the model to.
    """

    def __init__(
        self,
        arch: str = "ViT-L-14",
        pretrained: Optional[str] = "openai",
        checkpoint: Optional[str] = None,
        device: Union[str, torch.device] = "cpu",
        precision: str = "fp32",
    ):
        super().__init__()
        self.arch = arch
        self.pretrained = pretrained
        self.device = torch.device(device)

        model, _, preprocess_val = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        self.model = model
        self.preprocess = preprocess_val
        self.tokenizer = open_clip.get_tokenizer(arch)

        if checkpoint is not None:
            self.load_checkpoint(checkpoint)

        self.to(self.device)
        self.eval()
        for param in self.parameters():
            param.requires_grad_(False)

        self._precision = precision
        self.dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]

    # ------------------------------------------------------------------
    # loading / saving
    # ------------------------------------------------------------------
    def load_checkpoint(self, checkpoint: str) -> "CLIPEncoder":
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f"checkpoint {checkpoint} not found")
        state = torch.load(checkpoint, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = _strip_prefixes(state)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            LOGGER.warning("missing keys when loading %s: %d", checkpoint, len(missing))
        if unexpected:
            LOGGER.warning("unexpected keys when loading %s: %d", checkpoint, len(unexpected))
        return self

    # ------------------------------------------------------------------
    # properties used by the LVLM wrappers
    # ------------------------------------------------------------------
    @property
    def visual(self) -> nn.Module:
        return self.model.visual

    @property
    def image_size(self) -> int:
        size = self.model.visual.image_size
        return int(size) if isinstance(size, (int,)) else int(size[0])

    @property
    def image_mean(self):
        return tuple(self.model.visual.image_mean)

    @property
    def image_std(self):
        return tuple(self.model.visual.image_std)

    @property
    def embed_dim(self) -> int:
        return int(self.model.visual.output_dim)

    @property
    def width(self) -> int:
        """Hidden width of the vision transformer (input dim of the MM projector)."""
        return int(self.model.visual.width)

    @property
    def num_layers(self) -> int:
        return VIT_LAYERS.get(self.arch, len(self.model.visual.transformer.resblocks))

    # ------------------------------------------------------------------
    # forward passes
    # ------------------------------------------------------------------
    def prepare_inputs(self, images: torch.Tensor, normalized: bool = False) -> torch.Tensor:
        """Move images to the device and apply the CLIP normalization.

        The convention of this repository is that data pipelines supply images in
        ``[0, 1]`` (so that the l_inf ball of the attacks is defined on the
        *non-normalized* inputs, as in the addendum of the paper) and that the
        model wrapper performs the normalization.
        """
        images = images.to(self.device)
        if not normalized:
            images = normalize_images(images, self.image_mean, self.image_std)
        return images

    def image_embedding(
        self,
        images: torch.Tensor,
        normalize: bool = True,
        input_is_normalized: bool = False,
        autocast_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Differentiable projected image embedding ``phi(x)`` (used during training)."""
        dtype = autocast_dtype or self.dtype
        images = self.prepare_inputs(images, normalized=input_is_normalized)
        if dtype != torch.float32:
            features = self.model.encode_image(images.to(dtype)).float()
        else:
            features = self.model.encode_image(images)
        return features / features.norm(dim=-1, keepdim=True) if normalize else features

    def encode_image(
        self,
        images: torch.Tensor,
        normalize: bool = True,
        autocast_dtype: Optional[torch.dtype] = None,
        input_is_normalized: bool = False,
    ) -> torch.Tensor:
        """Projected image embedding ``phi(x)`` (unnormalized, as in CLIP).

        ``normalize=True`` returns the unit-norm embedding, which is what
        zero-shot classification uses (cosine similarity).
        """
        dtype = autocast_dtype or self.dtype
        images = self.prepare_inputs(images, normalized=input_is_normalized)
        with torch.no_grad():
            if dtype != torch.float32:
                features = self.model.encode_image(images.to(dtype)).float()
            else:
                features = self.model.encode_image(images)
        return features / features.norm(dim=-1, keepdim=True) if normalize else features

    @torch.no_grad()
    def encode_text(self, texts: Union[Sequence[str], torch.Tensor], normalize: bool = True) -> torch.Tensor:
        """Projected text embedding ``psi(t)``."""
        if isinstance(texts, (list, tuple)):
            tokens = self.tokenizer(list(texts)).to(self.device)
        else:
            tokens = texts.to(self.device)
        features = self.model.encode_text(tokens)
        return features / features.norm(dim=-1, keepdim=True) if normalize else features

    def vision_forward(
        self,
        images: torch.Tensor,
        layers: Sequence[int] = (-1, -2),
        use_ln_post: bool = False,
        input_is_normalized: bool = False,
    ) -> Dict[str, object]:
        """Run the vision tower once and return the requested intermediate layers.

        Returns a dict with

        * ``"hidden_states"``: list of the outputs of the requested blocks, each
          ``[B, 1 + N_patches, width]`` (class token first, exactly like the
          ``hidden_states`` of the Hugging Face implementation, so that LLaVA's
          ``vision_feature_layer=-2`` / ``vision_feature_select_strategy='default'``
          logic maps one-to-one onto it),
        * ``"class_token"``: the class token of the **last** block (with
          ``ln_post`` applied if ``use_ln_post``),
        * ``"patch_tokens"``: the patch tokens of the second-to-last block,
        * ``"projected"``: the final projected CLIP embedding ``phi(x)``.

        LLaVA consumes ``class_token``-stripped patch tokens of layer ``-2`` and
        OpenFlamingo the full last-layer output.
        """
        images = self.prepare_inputs(images, normalized=input_is_normalized)
        visual = self.model.visual
        requested = list(layers)
        # Hugging Face convention: hidden_states[0] is the embedding output and
        # hidden_states[k] the output of block k-1, i.e. for positive indices the
        # block index is ``layer - 1`` while negative indices are identical.
        block_indices = [layer if layer < 0 else layer - 1 for layer in requested if layer != 0]

        try:
            out = visual.forward_intermediates(
                images,
                indices=block_indices,
                output_fmt="NLC",
                output_extra_tokens=True,
                normalize_intermediates=use_ln_post,
            )
            intermediates = out["image_intermediates"]
            prefix = out.get("image_intermediates_prefix")
            hidden = []
            k = 0
            for layer in requested:
                if layer == 0:  # embedding output, not provided by open_clip
                    emb = visual._embeds(images)
                    hidden.append(emb)
                    continue
                patch = intermediates[k]
                cls = prefix[k] if prefix is not None else patch[:, :1]
                hidden.append(torch.cat([cls, patch], dim=1))
                k += 1
            projected = out.get("image_features")
        except (AttributeError, TypeError, IndexError):  # pragma: no cover - fallback for old open_clip
            hidden = self._vision_forward_with_hooks(images, requested)
            projected = None

        last = hidden[-1]
        class_token = visual.ln_post(last[:, 0]) if use_ln_post else last[:, 0]
        patch_tokens = last[:, 1:]
        if projected is None:
            projected = self.model.visual(images)
        patch_tokens_layer_minus_2 = None
        for layer, state in zip(requested, hidden):
            if layer % (self.num_layers + 1) == (self.num_layers - 1) % (self.num_layers + 1):
                patch_tokens_layer_minus_2 = state[:, 1:]
        return {
            "hidden_states": hidden,
            "class_token": class_token,
            "patch_tokens": patch_tokens,
            "patch_tokens_layer_minus_2": patch_tokens_layer_minus_2,
            "projected": projected,
        }

    def _vision_forward_with_hooks(self, images: torch.Tensor, layers: Sequence[int]) -> List[torch.Tensor]:
        """Fallback that captures block outputs with forward hooks (old open_clip)."""
        visual = self.model.visual
        blocks = list(visual.transformer.resblocks)
        store: Dict[int, torch.Tensor] = {}
        handles = []

        def make_hook(idx):
            def hook(module, inputs, output):
                store[idx] = output
            return hook

        for idx, block in enumerate(blocks):
            handles.append(block.register_forward_hook(make_hook(idx)))
        try:
            with torch.no_grad():
                embeddings = visual._embeds(images) if hasattr(visual, "_embeds") else None
                visual(images)
        finally:
            for handle in handles:
                handle.remove()

        hidden = []
        for layer in layers:
            if layer == 0:
                if embeddings is None:  # pragma: no cover - defensive
                    raise RuntimeError("cannot return the embedding layer for this open_clip version")
                hidden.append(embeddings)
            else:
                idx = layer if layer < 0 else layer - 1
                hidden.append(store[idx])
        return hidden

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.encode_image(images)


def load_clip(
    arch: str = "ViT-L-14",
    pretrained: Optional[str] = "openai",
    checkpoint: Optional[str] = None,
    device: Union[str, torch.device] = "cpu",
    precision: str = "fp32",
) -> CLIPEncoder:
    """Convenience wrapper around :class:`CLIPEncoder`."""
    return CLIPEncoder(
        arch=arch,
        pretrained=pretrained,
        checkpoint=checkpoint,
        device=device,
        precision=precision,
    )


#: alias kept for symmetry with the LVLM wrappers
CLIPImageEncoder = CLIPEncoder


def build_zero_shot_classifier(
    clip: CLIPEncoder,
    classnames: Sequence[str],
    templates: Sequence[str],
    batch_size: int = 128,
) -> torch.Tensor:
    """Text classifier of the zero-shot evaluation (Sec. 4.3).

    Mirrors ``CLIP_benchmark`` / OpenCLIP: all templates are encoded, then
    averaged per class and normalized.  The returned tensor ``[num_classes, d]``
    is used together with ``logit_scale`` of the CLIP model.
    """
    prompts = [template.format(c.replace("_", " ")) for c in classnames for template in templates]
    with torch.no_grad():
        embeddings = []
        for i in range(0, len(prompts), batch_size):
            tokens = clip.tokenizer(prompts[i : i + batch_size]).to(clip.device)
            embeddings.append(clip.model.encode_text(tokens))
        embeddings = torch.cat(embeddings, dim=0)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        embeddings = embeddings.reshape(len(classnames), len(templates), -1).mean(dim=1)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    return embeddings
