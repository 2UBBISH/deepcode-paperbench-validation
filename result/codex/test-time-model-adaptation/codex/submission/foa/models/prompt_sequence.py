"""Token-sequence backbones (VisionMamba) adapted with input prompts (Table 10).

"For VisionMamba, we concatenate learnable input prompts with the patch embeddings."

This module implements a generic wrapper for any *token sequence* classifier that exposes

* ``patch_embed``  - ``[B, 3, H, W] -> [B, m, d]``,
* ``blocks``       - a list of callables mapping ``[B, N, d] -> [B, N, d]``,
* ``norm_f`` / ``norm`` - the final normalisation,
* ``head``         - the classifier.

That covers the official VisionMamba (``vim``) implementation as well as any timm model
with the same layout (e.g. ``mambaout_*``), which is what the smoke tests use because the
``mamba-ssm`` CUDA kernels required by Vim are not installable on a CPU-only machine.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AdaptableModel


class PromptSequenceModel(AdaptableModel):
    """Frozen token-sequence classifier + ``N_p`` learnable input prompts."""

    def __init__(
        self,
        backbone: nn.Module,
        num_prompts: int = 3,
        prompt_init: str = "uniform",
        prompt_init_bound: float = 1.0,
        add_pos_embed: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_prompts = int(num_prompts)
        self.embed_dim = int(getattr(backbone, "embed_dim", getattr(backbone, "num_features", 0)))
        self.num_layers = len(self._blocks())
        head = getattr(backbone, "head", None)
        self.num_classes = int(getattr(backbone, "num_classes", getattr(head, "out_features", 1000)))
        self.add_pos_embed = add_pos_embed
        self.pos_embed = getattr(backbone, "pos_embed", None)
        init = torch.empty(self.num_prompts, self.embed_dim)
        if prompt_init == "uniform":
            nn.init.uniform_(init, -prompt_init_bound, prompt_init_bound)
        elif prompt_init == "normal":
            nn.init.normal_(init, std=0.02)
        else:
            init.zero_()
        self.register_buffer("prompt_init_values", init, persistent=False)

    def _blocks(self) -> List[nn.Module]:
        """The sequence of token-mixing stages, whatever they are called."""
        for attr in ("blocks", "stages", "layers"):
            mods = getattr(self.backbone, attr, None)
            if isinstance(mods, nn.Sequential):
                return list(mods)
            if isinstance(mods, (list, nn.ModuleList)):
                return list(mods)
        raise AttributeError("the backbone exposes none of blocks/stages/layers")

    # ----------------------------------------------------------------------------------
    def initial_prompt_vector(self, device=None, dtype=None) -> torch.Tensor:
        v = self.prompt_init_values.reshape(-1)
        return v.to(device=device or v.device, dtype=dtype or v.dtype)

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        patch_embed = getattr(self.backbone, "patch_embed", None)
        if patch_embed is None:                      # e.g. timm's MambaOut ConvNet stem
            stem = getattr(self.backbone, "stem", None)
            if stem is None:
                raise AttributeError("the backbone exposes no patch_embed or stem")
            x = stem(images)
            if x.dim() == 4:                         # NCHW -> tokens
                x = x.flatten(2).transpose(1, 2)
            return x
        x = patch_embed(images)
        if self.add_pos_embed and self.pos_embed is not None:
            pos = self.pos_embed
            if pos.dim() == 4:                      # HWC layout used by Vim
                pos = pos.reshape(1, -1, pos.shape[-1])
            if pos.shape[1] == x.shape[1]:
                x = x + pos
            elif pos.shape[1] == x.shape[1] + 1:    # class token exists but is unused here
                x = x + pos[:, 1:]
        return x

    def forward_tokens(
        self,
        tokens: torch.Tensor,
        prompt: Optional[torch.Tensor] = None,
        shift: Optional[torch.Tensor] = None,
        return_layers: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
        x = tokens
        n_patches = x.shape[1]
        if prompt is not None and self.num_prompts > 0:
            p = prompt.reshape(1, self.num_prompts, self.embed_dim).to(x.dtype)
            x = torch.cat([p.expand(x.shape[0], -1, -1), x], dim=1)
        layer_feats: Optional[List[torch.Tensor]] = None
        if return_layers:
            layer_feats = [self._pool_tokens(x, n_patches)]
        for blk in self._blocks():
            x = blk(x)
            if isinstance(x, (tuple, list)):
                x = x[0]
            if layer_feats is not None:
                layer_feats.append(self._pool_tokens(x, n_patches))
        e_n = self._pool_tokens(x, n_patches, apply_norm=True)
        if shift is not None:
            e_n = e_n + shift.reshape(1, -1).to(e_n.dtype)
        logits = self.backbone.head(e_n)
        if layer_feats is not None:
            layer_feats[-1] = e_n
        return logits, e_n, layer_feats

    @staticmethod
    def _pool(x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)

    def _pool_tokens(self, x: torch.Tensor, n_patches: int, apply_norm: bool = False) -> torch.Tensor:
        """Pool the *patch* tokens (the prompts must not leak into the classifier)."""
        if apply_norm:
            norm = getattr(self.backbone, "norm_f", None) or getattr(self.backbone, "norm", None)
            if norm is not None:
                x = norm(x)
        if x.shape[1] > n_patches:
            x = x[:, x.shape[1] - n_patches :]
        return self._pool(x)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.forward_tokens(self.preprocess(images))
        return logits


def build_prompt_visionmamba(
    checkpoint: str = "vim_tiny_patch16_224",
    num_prompts: int = 3,
    pretrained: bool = True,
):
    """Load the official VisionMamba checkpoint and wrap it.

    ``pip install vision-mamba`` (or the official repository) is required; if it is not
    available we fall back to a timm token-sequence model so that the code path stays
    runnable, in which case the numbers of Table 10 obviously do not apply.
    """
    try:  # pragma: no cover - depends on an optional dependency
        from vim.models_mamba import create_model as vim_create_model  # type: ignore

        backbone = vim_create_model(checkpoint, pretrained=pretrained)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad_(False)
        return PromptSequenceModel(backbone, num_prompts=num_prompts)
    except Exception as exc:
        print(f"[visionmamba] official package unavailable ({exc}); falling back to timm")
        import timm

        backbone = timm.create_model("vit_tiny_patch16_224", pretrained=pretrained)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad_(False)
        return PromptSequenceModel(backbone, num_prompts=num_prompts, add_pos_embed=True)
