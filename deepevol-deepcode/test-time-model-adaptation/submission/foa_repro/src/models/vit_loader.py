"""ViT-Base loader with all-layer CLS feature extraction.

This module implements the backbone used by FOA (Forward-Optimization Adaptation).

Paper specification (Section 2, Eqn. 1):
    E_i   = L_i(E_{i-1}),  i = 1..N
    y_hat = Head(e_N^0)

where ``e_i^0`` is the [CLS] token of layer ``L_i``.  We therefore expose *all*
the CLS tokens ``{e_n^0}_{n=1..N}`` (plus the layer-0 CLS embedding ``e_0^0``,
needed because the source statistics bank stores ``{mu_i^S, sigma_i^S}_{i=0}^{N}``)
together with the logits produced by ``Head(e_N^0)``.

Key properties required by the paper:
  * All model parameters are **frozen**; the model is never updated
    (Section 3: "keeping all other model parameters frozen").
  * No gradient is ever computed (Section 2 / 3 / 4: FOA is
    backpropagation-free).  Every forward runs under ``torch.no_grad()`` and the
    model is put in ``eval()`` mode.
  * Prompt injection is optional: source ID statistics are computed **without**
    the newly inserted prompt (Appendix B.2: "The source in-distribution
    statistics {mu_i^S, sigma_i^S}_{i=0}^N are calculated without using the newly
    inserted prompt.").
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

try:  # timm is a hard requirement but keep the import error informative
    import timm
except Exception as exc:  # pragma: no cover - environment dependent
    timm = None
    _TIMM_IMPORT_ERROR = exc


# ---------------------------------------------------------------------------
# Checkpoint handling
# ---------------------------------------------------------------------------

#: The exact checkpoint used in the paper (Appendix B.2 footnote 1).
FOA_CHECKPOINT_URL = (
    "https://storage.googleapis.com/vit_models/augreg/"
    "B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0-"
    "imagenet2012-steps_20k-lr_0.01-res_224.npz"
)

#: timm tag of the same weights (augreg ViT-B/16 pretrained on ImageNet-21k and
#: fine-tuned on ImageNet-2012 at 224px).  Used when ``pretrained=True`` and no
#: explicit checkpoint is given.
FOA_TIMM_TAG = "vit_base_patch16_224.augreg_in21k_ft_in1k"

DEFAULT_MODEL_NAME = "vit_base_patch16_224"


def _npz_to_state_dict(path: str) -> Dict[str, torch.Tensor]:
    """Convert the augreg ``.npz`` checkpoint to a timm state dict.

    The augreg checkpoints store raw numpy arrays keyed by the JAX/pytorch
    module names used by ``vit_models``.  They are almost identical to timm's
    naming; we only need to strip the optional ``pre_logits`` representation
    layer (timm's ViT-Base with ``num_classes=1000`` has no such layer).
    """
    with np.load(path, allow_pickle=False) as data:
        state = {k: torch.from_numpy(np.asarray(v)) for k, v in data.items()}

    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("pre_logits."):
            # timm's plain ViT has no representation layer before the head.
            continue
        if key == "pos_embed":
            value = value.reshape(1, *value.shape[-2:]) if value.ndim == 2 else value
        if key == "cls_token":
            value = value.reshape(1, 1, -1) if value.ndim == 2 else value
        cleaned[key] = value
    return cleaned


def _load_npz_into_model(model: nn.Module, path: str, strict: bool = True) -> None:
    state = _npz_to_state_dict(path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if strict and missing:
        # tolerate a missing ``head`` when the caller replaced it (num_classes)
        critical_missing = [k for k in missing if not k.startswith("head.")]
        if critical_missing:
            raise RuntimeError(
                f"Missing keys while loading {path}: {critical_missing[:8]}"
            )
    if strict and unexpected:
        critical_unexpected = [
            k for k in unexpected if not k.startswith("pre_logits.")
        ]
        if critical_unexpected:
            raise RuntimeError(
                f"Unexpected keys while loading {path}: {critical_unexpected[:8]}"
            )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class ViTWithCLSFeatures(nn.Module):
    """A frozen ViT-Base that returns all-layer [CLS] tokens and logits.

    Parameters
    ----------
    model_name:
        timm model name.  Defaults to ``vit_base_patch16_224``.
    checkpoint:
        Optional path to a local ``.npz`` (augreg) checkpoint.  If ``None`` and
        ``pretrained`` is True the timm weights of :data:`FOA_TIMM_TAG` are used.
    pretrained:
        Whether to initialise from (cached) pretrained weights.
    num_classes:
        Number of classes of the classification head (1000 for ImageNet).
    device:
        Torch device string.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        checkpoint: Optional[str] = None,
        pretrained: bool = True,
        num_classes: int = 1000,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        if timm is None:  # pragma: no cover
            raise ImportError(f"timm is required for the ViT backbone: {_TIMM_IMPORT_ERROR}")

        self.model_name = model_name
        self.num_classes = num_classes
        self.device_name = device

        pretrained_cfg = None
        if pretrained and checkpoint is None:
            pretrained_cfg = FOA_TIMM_TAG
            try:
                self.model = timm.create_model(
                    model_name, pretrained=True, num_classes=num_classes,
                    pretrained_cfg=pretrained_cfg,
                )
            except Exception:
                self.model = timm.create_model(
                    model_name, pretrained=True, num_classes=num_classes
                )
        else:
            self.model = timm.create_model(
                model_name, pretrained=False, num_classes=num_classes
            )
            if checkpoint is not None:
                if not os.path.exists(checkpoint):
                    raise FileNotFoundError(f"ViT checkpoint not found: {checkpoint}")
                if checkpoint.endswith(".npz"):
                    _load_npz_into_model(self.model, checkpoint)
                else:
                    state = torch.load(checkpoint, map_location="cpu")
                    state = state.get("model", state.get("state_dict", state))
                    self.model.load_state_dict(state, strict=False)

        # --------------------------------------------------------------
        # Freeze everything: FOA never updates the model weights.
        # --------------------------------------------------------------
        for param in self.model.parameters():
            param.requires_grad_(False)

        self.model.eval()
        self.to(device)
        self.eval()

        self.num_layers: int = self._infer_num_layers()
        self.embed_dim: int = int(self.model.embed_dim)
        self.num_patches: int = int(self.model.patch_embed.num_patches)
        self.patch_size: int = int(self.model.patch_embed.patch_size[0])
        self.img_size: int = int(self.model.patch_embed.img_size[0])
        self.num_prefix_tokens: int = int(getattr(self.model, "num_prefix_tokens", 1))

        # Hook output of every transformer block (-> e_1^0 ... e_N^0).
        self._block_outputs: Dict[int, torch.Tensor] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _infer_num_layers(self) -> int:
        for attr in ("blocks", "layers", "encoder"):
            mod = getattr(self.model, attr, None)
            if mod is None:
                continue
            if hasattr(mod, "__len__"):
                return len(mod)
            if hasattr(mod, "blocks"):
                return len(mod.blocks)
        raise RuntimeError("Could not infer the number of transformer layers.")

    def _register_hooks(self) -> None:
        blocks = getattr(self.model, "blocks", None)
        if blocks is None:
            raise RuntimeError("Expected a timm ViT with a `.blocks` attribute.")

        def make_hook(idx: int):
            def hook(_module, _inputs, output):
                tensor = output[0] if isinstance(output, (tuple, list)) else output
                self._block_outputs[idx] = tensor[:, 0]
            return hook

        for idx, block in enumerate(blocks):
            self._handles.append(block.register_forward_hook(make_hook(idx)))

    def remove_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _embed(self, pixel_values: torch.Tensor, prompt: Optional[torch.Tensor]):
        """Build the input sequence for the first transformer block.

        Sequence order (Addendum): ``[CLS token, learnable prompts, patch embeddings]``.

        Returns ``(x, cls_embedding)`` where ``x`` is the (optionally prompt
        augmented) token sequence and ``cls_embedding`` is ``e_0^0``.
        """
        x = self.model.patch_embed(pixel_values)
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
        pos_embed = self.model.pos_embed

        if prompt is not None and prompt.numel() > 0:
            prompt = prompt.to(dtype=x.dtype, device=x.device)
            if prompt.dim() == 2:
                prompt = prompt.unsqueeze(0)
            n_prompt = prompt.shape[1]
            x = torch.cat([cls_token, prompt.expand(x.shape[0], -1, -1), x], dim=1)
            # [CLS, prompts, patches] positional layout: prompts receive no
            # positional embedding (default; documented in README).
            prompt_pos = torch.zeros(
                1, n_prompt, pos_embed.shape[-1],
                dtype=pos_embed.dtype, device=pos_embed.device,
            )
            pos = torch.cat([pos_embed[:, :1], prompt_pos, pos_embed[:, 1:]], dim=1)
        else:
            x = torch.cat([cls_token, x], dim=1)
            pos = pos_embed

        x = x + pos
        if getattr(self.model, "pos_drop", None) is not None:
            x = self.model.pos_drop(x)
        cls_embedding = x[:, 0]
        return x, cls_embedding

    def forward_features(
        self,
        pixel_values: torch.Tensor,
        prompt: Optional[torch.Tensor] = None,
    ):
        """Run the frozen backbone and return ``(cls_features, logits)``.

        ``cls_features`` is a list of length ``N + 1``:
            index 0  -> ``e_0^0`` (input to the first block)
            index n  -> ``e_n^0`` for n = 1..N (output of block n)
        ``logits`` are ``Head(e_N^0)``.
        """
        self._block_outputs.clear()
        with torch.no_grad():
            x, cls_0 = self._embed(pixel_values, prompt)
            for block in self.model.blocks:
                x = block(x)
            x = self.model.norm(x)
            cls_features: List[torch.Tensor] = [cls_0]
            for idx in range(self.num_layers):
                out = self._block_outputs.get(idx)
                if out is None:
                    raise RuntimeError(f"Missing hook output for block {idx}")
                cls_features.append(out)
            final_cls = x[:, 0]
            logits = self.model.head(final_cls) if self.model.head is not None else final_cls
            return cls_features, logits, final_cls

    def forward_with_features(
        self,
        pixel_values: torch.Tensor,
        prompt: Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        """Dictionary-returning convenience wrapper used by FOA / source stats."""
        cls_features, logits, final_cls = self.forward_features(pixel_values, prompt)
        return {
            "cls_features": cls_features,   # list of N+1 tensors, [B, d]
            "logits": logits,               # [B, num_classes]
            "final_cls": final_cls,         # e_N^0 before the head, [B, d]
        }

    def forward(
        self,
        pixel_values: torch.Tensor,
        prompt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return logits (``Head(e_N^0)``) for plain usage."""
        _, logits, _ = self.forward_features(pixel_values, prompt)
        return logits

    # ------------------------------------------------------------------
    @property
    def prompt_dim(self) -> int:
        """Flattened dimension of the CMA search space for one prompt token."""
        return self.embed_dim

    def head(self, cls_feature: torch.Tensor) -> torch.Tensor:
        """Apply only the classification head (used after activation shifting)."""
        with torch.no_grad():
            return self.model.head(cls_feature)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


def build_vit(
    model_name: str = DEFAULT_MODEL_NAME,
    checkpoint: Optional[str] = None,
    pretrained: bool = True,
    num_classes: int = 1000,
    device: str = "cuda",
) -> ViTWithCLSFeatures:
    """Factory mirroring the config interface."""
    return ViTWithCLSFeatures(
        model_name=model_name,
        checkpoint=checkpoint,
        pretrained=pretrained,
        num_classes=num_classes,
        device=device,
    )


__all__ = [
    "ViTWithCLSFeatures",
    "build_vit",
    "FOA_CHECKPOINT_URL",
    "FOA_TIMM_TAG",
    "DEFAULT_MODEL_NAME",
]
