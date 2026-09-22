"""CLIP loading / wrapping utilities.

The paper replaces the (frozen) vision encoder of LLaVA and OpenFlamingo by an
adversarially fine-tuned CLIP vision encoder.  Two different "views" of that
encoder are relevant:

* the **projected class token** (:meth:`CLIPImageEncoder.encode_image`) -- this is
  the embedding :math:`\\phi(x)` that enters the zero-shot classifier of
  Sec. 3.1 and therefore also the FARE / TeCoA fine-tuning losses;
* the **penultimate-layer patch tokens** (:meth:`CLIPImageEncoder.patch_tokens`)
  -- this is what LLaVA-1.5 (``select_layer = -2``, ``select_feature = 'patch'``)
  and OpenFlamingo feed to their language model.

Both are produced by the *same* set of ViT weights, which is why fine-tuning
with a class-token loss is enough to make the LVLMs robust (App. B.1).
"""
from __future__ import annotations

import dataclasses
import math
import os
from typing import Optional, Sequence

import torch
import torch.nn as nn

import open_clip

from .utils.common import LOGGER


#: Architectures used in the paper.  ``ViT-L-14`` is the encoder of LLaVA-1.5 and
#: OpenFlamingo-9B; ``ViT-B-32`` is used for the (appendix) ablations of Sec. B.3.
CLIP_ARCHITECTURES = {
    "ViT-L-14": dict(model_name="ViT-L-14", pretrained="openai", image_size=224, width=1024),
    "ViT-L-14-336": dict(
        model_name="ViT-L-14-336", pretrained="openai", image_size=336, width=1024
    ),
    "ViT-B-32": dict(model_name="ViT-B-32", pretrained="openai", image_size=224, width=768),
    "ViT-B-16": dict(model_name="ViT-B-16", pretrained="openai", image_size=224, width=768),
}


@dataclasses.dataclass
class CLIPSpec:
    """Everything needed to instantiate a CLIP backbone."""

    arch: str = "ViT-L-14"
    pretrained: str = "openai"
    image_size: int = 224
    #: optional checkpoint with fine-tuned vision weights (FARE / TeCoA)
    checkpoint: Optional[str] = None
    #: name of the key inside the checkpoint that holds the state dict
    checkpoint_key: Optional[str] = None

    @classmethod
    def from_arch(cls, arch: str, **kwargs) -> "CLIPSpec":
        base = dict(CLIP_ARCHITECTURES.get(arch, {"model_name": arch, "image_size": 224}))
        model_name = base.pop("model_name", arch)
        base.pop("width", None)
        base.update(kwargs)
        base.setdefault("pretrained", "openai")
        return cls(arch=model_name, **base)


def build_clip(spec: CLIPSpec):
    """Create an ``open_clip`` model + the matching train/val transforms."""
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        spec.arch,
        pretrained=spec.pretrained,
        # NOTE: the keyword is ``force_image_size``; ``image_size`` would be
        # forwarded to the model constructor and raise a TypeError.
        force_image_size=spec.image_size,
        force_quick_gelu=True,
        jit=False,
    )
    tokenizer = open_clip.get_tokenizer(spec.arch)
    return model, preprocess_train, preprocess_val, tokenizer


def build_pixel_transforms(image_size: Optional[int] = None, arch: str = "ViT-L-14"):
    """Transforms that stop *before* the CLIP normalisation.

    Every attack in the paper is carried out on non-normalised images, so the
    data pipeline has to hand pixel-space tensors (``[0, 1]``) to the models and
    leave the normalisation to :meth:`CLIPImageEncoder.normalize`.
    """
    from open_clip.transform import image_transform

    if image_size is None:
        image_size = CLIP_ARCHITECTURES.get(arch, {}).get("image_size", 224)
    # ``image_transform`` always appends a ``Normalize``; using the identity
    # normalisation gives us the plain pixel-space transform we need.
    identity = dict(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
    train_tf = image_transform(image_size, is_train=True, **identity)
    eval_tf = image_transform(image_size, is_train=False, **identity)
    return train_tf, eval_tf


class CLIPImageEncoder(nn.Module):
    """Wraps an ``open_clip`` model and exposes the views used in the paper.

    Parameters
    ----------
    model:
        the ``open_clip`` model (vision tower + text tower).
    feature:
        ``"projected_class_token"`` (default, matches :math:`\\phi` of Eq. (1)),
        ``"class_token"`` (the raw output of the ViT, before the projection) or
        ``"cls_patch"`` (all tokens of the penultimate layer, LLaVA style).
    freeze_text:
        the adversarial fine-tuning only updates the vision encoder; the text
        encoder stays frozen, exactly as in the paper.
    """

    def __init__(self, model, feature: str = "projected_class_token", freeze_text: bool = True):
        super().__init__()
        self.model = model
        self.feature = feature
        self.freeze_text = freeze_text
        #: pristine positional embedding, cached for resolution changes
        self._orig_pos_embed = None
        if freeze_text:
            self.freeze_text_tower()

    def freeze_text_tower(self) -> None:
        """Only the vision encoder (and the logit scale) stays trainable."""
        for name, param in self.model.named_parameters():
            if name.startswith("visual."):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

    # ------------------------------------------------------------------ config
    @property
    def visual(self) -> nn.Module:
        return self.model.visual

    @property
    def embed_dim(self) -> int:
        """Dimensionality of the :math:`\\phi(x)` used in the training loss."""
        return int(self.model.visual.output_dim)

    @property
    def width(self) -> int:
        """Width of the ViT patch tokens (input dim of the LVLM projector)."""
        visual = self.model.visual
        if hasattr(visual, "width"):
            return int(visual.width)
        return int(visual.transformer.width)

    @property
    def patch_size(self) -> int:
        return int(self.model.visual.patch_size[0])

    @property
    def image_mean(self) -> torch.Tensor:
        mean = self.model.visual.image_mean
        return torch.as_tensor(mean, dtype=torch.float32)

    @property
    def image_std(self) -> torch.Tensor:
        std = self.model.visual.image_std
        return torch.as_tensor(std, dtype=torch.float32)

    @property
    def logit_scale(self) -> torch.Tensor:
        return self.model.logit_scale

    # ------------------------------------------------------------- preprocessing
    def normalize(self, images: torch.Tensor) -> torch.Tensor:
        """Map images from ``[0, 1]`` (pixel space) to the CLIP input space.

        All attacks in the paper operate on the *non-normalised* inputs, i.e. the
        :math:`\\ell_\\infty` ball is computed in pixel space and the
        normalisation is applied afterwards (see the addendum).
        """
        mean = self.image_mean.to(images.device, images.dtype)
        std = self.image_std.to(images.device, images.dtype)
        return (images - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)

    def denormalize(self, images: torch.Tensor) -> torch.Tensor:
        mean = self.image_mean.to(images.device, images.dtype)
        std = self.image_std.to(images.device, images.dtype)
        return images * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

    def interpolate_pos_embed_if_needed(self, x: torch.Tensor) -> None:
        """Bicubically interpolate the positional embedding when the resolution changes.

        The paper evaluates CIFAR10/CIFAR100 and STL-10 "at their respective
        original resolution" (App. B.10), which is not the 224x224 grid the CLIP
        ViT was trained on.  As in the original CLIP code, the positional
        embedding is interpolated to the grid implied by the input size; the
        model weights themselves are unchanged.

        The interpolation is always computed from the *pristine* positional
        embedding (cached on first use), so switching resolutions back and forth
        (e.g. evaluating CIFAR at 32x32 and ImageNet at 224x224) does not
        compound the resampling.
        """
        visual = self.model.visual
        pos_embed = getattr(visual, "positional_embedding", None)
        if pos_embed is None or pos_embed.dim() != 2:
            return
        if self._orig_pos_embed is None:
            # cache the checkpoint's positional embedding on first use
            self._orig_pos_embed = pos_embed.detach().clone()
        height, width = x.shape[-2:]
        patch = self.patch_size
        grid_h, grid_w = height // patch, width // patch
        base_pos = self._orig_pos_embed
        num_positions = base_pos.shape[0]
        if num_positions < 2:
            return
        if grid_h < 1 or grid_w < 1:
            return
        if num_positions - 1 == grid_h * grid_w:
            # the current resolution matches the checkpoint's grid: restore the
            # pristine embedding if a previous call changed it
            if pos_embed.shape != base_pos.shape:
                visual.positional_embedding = nn.Parameter(
                    base_pos.to(device=pos_embed.device, dtype=pos_embed.dtype).clone(),
                    requires_grad=pos_embed.requires_grad,
                )
            return
        num_prefix = num_positions - 1
        side = int(round(math.sqrt(num_prefix)))
        if side * side != num_prefix:
            return
        prefix, grid = base_pos[:1], base_pos[1:]
        grid = grid.reshape(1, side, side, -1).permute(0, 3, 1, 2)
        grid = torch.nn.functional.interpolate(
            grid, size=(grid_h, grid_w), mode="bicubic", align_corners=False
        )
        grid = grid.permute(0, 2, 3, 1).reshape(grid_h * grid_w, -1)
        new_pos = torch.cat([prefix, grid], dim=0).to(dtype=pos_embed.dtype, device=pos_embed.device)
        visual.positional_embedding = nn.Parameter(new_pos, requires_grad=pos_embed.requires_grad)

    # --------------------------------------------------------------- embeddings
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """``phi(x)`` -- makes the encoder usable as a plain ``nn.Module``.

        This matters for ``DistributedDataParallel``: the training losses call
        the encoder through ``__call__`` (i.e. through ``DDP.forward``) so that
        gradients are synchronised across processes.
        """
        return self.encode_image(images)

    def encode_image(
        self, images: torch.Tensor, already_normalized: bool = False
    ) -> torch.Tensor:
        """Image embedding :math:`\\phi(x)` (``B x D``).

        ``images`` are expected in ``[0, 1]`` pixel space (the convention of the
        whole repository: all attacks perturb the *non-normalised* inputs); the
        CLIP normalisation is applied here.  Pass ``already_normalized=True`` if
        the caller has already applied it.
        """
        x = images if already_normalized else self.normalize(images)
        self.interpolate_pos_embed_if_needed(x)
        if self.feature == "projected_class_token":
            return self.model.encode_image(x)
        if self.feature == "class_token":
            out = self.model.visual.forward_intermediates(
                x, output_fmt="NCHW", output_extra_tokens=True, intermediates_only=True
            )
            return out["image_intermediates_prefix"][-1][:, 0]
        if self.feature == "cls_patch":
            return self.patch_tokens(x, already_normalized=True, include_cls=True)
        raise ValueError(f"unknown feature '{self.feature}'")

    def patch_tokens(
        self,
        images: torch.Tensor,
        layer: int = -2,
        include_cls: bool = False,
        already_normalized: bool = False,
    ) -> torch.Tensor:
        """Penultimate-layer tokens (``B x N x width``); LLaVA/OF consume these.

        ``layer=-2`` follows LLaVA-1.5's ``select_layer`` /
        BLIP-2's ``vision_feature_layer``: the output of the second-to-last
        transformer block.  ``visual.forward_intermediates`` returns the output
        of every block in order, so ``intermediates[-2]`` is exactly the
        ``hidden_states[-2]`` of the HuggingFace ``CLIPVisionModel`` (for a model
        with ``L`` blocks, HF's ``hidden_states`` has ``L + 1`` entries whose
        first one is the embedding output, hence its index ``-2`` is the output
        of block ``L - 2``).
        """
        x = images if already_normalized else self.normalize(images)
        self.interpolate_pos_embed_if_needed(x)
        out = self.model.visual.forward_intermediates(
            x, output_fmt="NCHW", output_extra_tokens=True, intermediates_only=True
        )
        spatial = out["image_intermediates"][layer]  # B x C x H x W
        b, c, h, w = spatial.shape
        tokens = spatial.reshape(b, c, h * w).permute(0, 2, 1).contiguous()
        if include_cls:
            cls = out["image_intermediates_prefix"][layer]
            tokens = torch.cat([cls, tokens], dim=1)
        return tokens

    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        return self.model.encode_text(text_tokens)

    # ----------------------------------------------------------------- training
    def trainable_parameters(self):
        return [p for p in self.visual.parameters() if p.requires_grad]

    def train(self, mode: bool = True):  # noqa: D102 - keeps text tower frozen
        super().train(mode)
        if self.freeze_text:
            self.model.transformer.eval()
            self.model.ln_final.eval()
        return self

    # -------------------------------------------------------------- checkpoint
    def load_finetuned(self, checkpoint_path: str, key: Optional[str] = None, strict: bool = True):
        """Load a FARE / TeCoA checkpoint into the *vision* tower."""
        state = torch.load(checkpoint_path, map_location="cpu")
        if key is not None:
            state = state[key]
        elif isinstance(state, dict) and "state_dict" in state and not self._looks_like_state_dict(state):
            state = state["state_dict"]
        if isinstance(state, dict) and "model" in state and not self._looks_like_state_dict(state):
            state = state["model"]

        visual_state = {}
        for name, tensor in state.items():
            for prefix in ("model.visual.", "visual.", "module.model.visual.", "module.visual."):
                if name.startswith(prefix):
                    visual_state[name[len(prefix):]] = tensor
                    break
        if not visual_state:
            raise RuntimeError(f"no vision-encoder weights found in {checkpoint_path}")
        # Drop entries whose shape does not match, e.g. a positional embedding
        # that was interpolated by a previous evaluation at another resolution.
        current = self.visual.state_dict()
        for name in list(visual_state):
            if name in current and current[name].shape != visual_state[name].shape:
                LOGGER.warning(
                    "skipping '%s' (checkpoint %s vs model %s)",
                    name, tuple(visual_state[name].shape), tuple(current[name].shape),
                )
                visual_state.pop(name)
        missing, unexpected = self.visual.load_state_dict(visual_state, strict=False)
        missing = [m for m in missing if "position_ids" not in m]
        if strict and missing:
            LOGGER.warning("missing keys while loading vision encoder: %s", missing[:5])
        if unexpected:
            LOGGER.warning("unexpected keys while loading vision encoder: %s", unexpected[:5])
        # the positional embedding may have changed: rebuild the cache lazily
        self._orig_pos_embed = None
        LOGGER.info("loaded fine-tuned vision encoder from %s", checkpoint_path)
        return self

    @staticmethod
    def _looks_like_state_dict(obj) -> bool:
        if not isinstance(obj, dict):
            return False
        return all(isinstance(v, torch.Tensor) for v in obj.values()) and len(obj) > 0


def load_clip_encoder(
    arch: str = "ViT-L-14",
    pretrained: str = "openai",
    image_size: int = 224,
    checkpoint: Optional[str] = None,
    checkpoint_key: Optional[str] = None,
    feature: str = "projected_class_token",
    device="cpu",
    dtype: torch.dtype = torch.float32,
    freeze_text: bool = True,
) -> CLIPImageEncoder:
    """Convenience constructor used by the attack / evaluation scripts."""
    spec = CLIPSpec.from_arch(
        arch,
        image_size=image_size,
        checkpoint=checkpoint,
        checkpoint_key=checkpoint_key,
    )
    spec.pretrained = pretrained
    model, _, _, _ = build_clip(spec)
    encoder = CLIPImageEncoder(model, feature=feature, freeze_text=freeze_text)
    if checkpoint:
        encoder.load_finetuned(checkpoint, key=checkpoint_key)
    encoder = encoder.to(device=device)
    if dtype != torch.float32:
        # The vision tower is the only part we ever need at reduced precision.
        encoder.visual.to(dtype=dtype)
    encoder.eval()
    return encoder


def load_clip_image_transforms(arch: str = "ViT-L-14", image_size: int = 224, pretrained="openai"):
    """Return ``(preprocess_train, preprocess_val)`` for a given architecture."""
    _, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        arch, pretrained=pretrained, image_size=image_size, force_quick_gelu=True, jit=False
    )
    return preprocess_train, preprocess_val


def class_token_and_projection(encoder: CLIPImageEncoder):
    """Return ``(proj_weight, proj_bias)`` if the visual tower has a projection."""
    proj = getattr(encoder.visual, "proj", None)
    if proj is None:
        return None, None
    return proj.weight, proj.bias


def named_visual_parameters(encoder: CLIPImageEncoder, prefixes: Sequence[str] = ("transformer",)):
    """Helpful for the appendix ablations (e.g. only fine-tuning the last block)."""
    out = []
    for name, param in encoder.visual.named_parameters():
        if any(name.startswith(p) for p in prefixes):
            out.append((name, param))
    return out


def resolve_checkpoint_dir(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))
