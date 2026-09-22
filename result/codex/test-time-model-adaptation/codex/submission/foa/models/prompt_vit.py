"""Prompt-injected (timm) Vision Transformer used as FOA's backbone.

Reference checkpoint from the paper (Appendix B.2, footnote 1):
``B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0-imagenet2012-steps_20k-lr_0.01-res_224.npz``
which is exactly timm's ``vit_base_patch16_224.augreg_in21k_ft_in1k``.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AdaptableModel

PAPER_VIT_CHECKPOINT = "vit_base_patch16_224.augreg_in21k_ft_in1k"


class PromptViT(AdaptableModel):
    """Vision Transformer with a learnable prompt inserted after the CLS token.

    The extra prompt tokens are injected between the CLS token and the patch embeddings
    (see the addendum of the paper: "*The arrangement of input sequence elements is
    [CLS token, learnable prompts, patch embeddings] in that specific order*").
    """

    def __init__(
        self,
        vit: nn.Module,
        num_prompts: int = 3,
        prompt_pos: str = "zero",
        prompt_init: str = "uniform",
        prompt_init_bound: float = 1.0,
        prompt_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert getattr(vit, "global_pool", "token") == "token", (
            "PromptViT expects a ViT whose classifier consumes the CLS token"
        )
        self.vit = vit
        self.num_prompts = int(num_prompts)
        self.embed_dim = int(vit.embed_dim)
        self.num_layers = len(vit.blocks)
        self.num_classes = int(vit.num_classes)
        self.num_prefix_tokens = int(vit.num_prefix_tokens)
        self._no_embed_class = bool(getattr(vit, "no_embed_class", False))
        self.prompt_pos_mode = prompt_pos
        self.prompt_dropout = nn.Dropout(prompt_dropout) if prompt_dropout > 0 else nn.Identity()

        if self.num_prompts > 0:
            pos = self._make_prompt_pos(prompt_pos)          # [1, Np, D]
            self.register_buffer("prompt_pos", pos, persistent=False)
            init = torch.empty(self.num_prompts, self.embed_dim)
            if prompt_init == "uniform":
                nn.init.uniform_(init, -prompt_init_bound, prompt_init_bound)
            elif prompt_init == "normal":
                nn.init.normal_(init, std=0.02)
            elif prompt_init == "zeros":
                init.zero_()
            else:
                raise ValueError(f"unknown prompt_init: {prompt_init}")
            self.register_buffer("prompt_init_values", init, persistent=False)
        else:
            self.register_buffer("prompt_pos", torch.zeros(1, 0, self.embed_dim), persistent=False)
            self.register_buffer(
                "prompt_init_values", torch.zeros(0, self.embed_dim), persistent=False
            )

    # ----------------------------------------------------------------------------------
    def _make_prompt_pos(self, mode: str) -> torch.Tensor:
        vit = self.vit
        pos_embed = vit.pos_embed
        if pos_embed is None:
            return torch.zeros(1, self.num_prompts, self.embed_dim)
        if mode == "cls":
            # repeat the positional embedding of the CLS token
            return pos_embed[:, :1].repeat(1, self.num_prompts, 1)
        if mode == "zero":
            return torch.zeros(1, self.num_prompts, self.embed_dim)
        if mode == "learnable":
            return torch.zeros(1, self.num_prompts, self.embed_dim)
        if mode == "interpolate":
            total = 1 + self.num_prompts + vit.patch_embed.num_patches
            out = F.interpolate(
                pos_embed.permute(0, 2, 1), size=total, mode="linear", align_corners=False
            ).permute(0, 2, 1)
            return out[:, 1 : 1 + self.num_prompts]
        raise ValueError(f"unknown prompt_pos mode: {mode}")

    # ----------------------------------------------------------------------------------
    @property
    def prompt_dim(self) -> int:
        return self.num_prompts * self.embed_dim

    def initial_prompt_vector(self, device=None, dtype=None) -> torch.Tensor:
        """Flattened ``m^(0)``; the paper uses *uniform initialization*."""
        v = self.prompt_init_values.reshape(-1)
        return v.to(device=device or v.device, dtype=dtype or v.dtype)

    # ----------------------------------------------------------------------------------
    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """``[B, 3, H, W] -> [B, 1 + m, d]`` with positional embeddings added.

        The result is the *prompt-free* token sequence ``[CLS, patch embeddings]``.  The
        learnable prompts are inserted between the CLS token and the patches by
        :meth:`forward_tokens`.
        """
        vit = self.vit
        x = vit.patch_embed(images)
        B = x.shape[0]
        if not torch.jit.is_scripting() and getattr(vit, "dynamic_img_size", False):
            raise NotImplementedError("dynamic_img_size is not supported by PromptViT")
        pos_embed = vit.pos_embed
        if pos_embed is None:
            return torch.cat([vit.cls_token.expand(B, -1, -1), x], dim=1)
        if self._no_embed_class:
            patch_pos = pos_embed
            cls_pos = torch.zeros_like(pos_embed[:, :1])
        else:
            patch_pos = pos_embed[:, self.num_prefix_tokens :]
            cls_pos = pos_embed[:, : self.num_prefix_tokens]
        cls = vit.cls_token.expand(B, -1, -1) + cls_pos
        return torch.cat([cls, x + patch_pos], dim=1)

    def forward_tokens(
        self,
        tokens: torch.Tensor,
        prompt: Optional[torch.Tensor] = None,
        shift: Optional[torch.Tensor] = None,
        return_layers: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
        vit = self.vit
        prefix = self.num_prefix_tokens
        cls = tokens[:, :prefix]
        patch_tokens = tokens[:, prefix:]
        B = tokens.shape[0]

        if prompt is not None and self.num_prompts > 0:
            p = prompt.reshape(1, self.num_prompts, self.embed_dim).to(tokens.dtype)
            p = self.prompt_dropout(p) + self.prompt_pos.to(tokens.dtype)
            tokens = torch.cat([cls, p.expand(B, -1, -1), patch_tokens], dim=1)

        tokens = vit.patch_drop(tokens)
        tokens = vit.norm_pre(tokens)

        layer_cls: Optional[List[torch.Tensor]] = None
        if return_layers:
            layer_cls = [tokens[:, 0]]
        for blk in vit.blocks:
            tokens = blk(tokens)
            if layer_cls is not None:
                layer_cls.append(tokens[:, 0])
        tokens = vit.norm(tokens)
        e_n = tokens[:, 0]

        if shift is not None:
            # Eqn. (7): e_N^0 <- e_N^0 + gamma * d
            e_n = e_n + shift.reshape(1, -1).to(e_n.dtype)

        if layer_cls is not None:
            # e_N^0 is the (post-norm) activation that is fed to the task head; it equals
            # the returned activation, i.e. the shift is reflected in the last entry too.
            layer_cls[-1] = e_n

        logits = self.vit.head(e_n)
        return logits, e_n, layer_cls

    # ----------------------------------------------------------------------------------
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.forward_with_prompt(images)
        return logits


def build_prompt_vit(
    checkpoint: str = PAPER_VIT_CHECKPOINT,
    num_prompts: int = 3,
    pretrained: bool = True,
    prompt_pos: str = "zero",
    prompt_init: str = "uniform",
    prompt_init_bound: float = 1.0,
    **timm_kwargs,
) -> PromptViT:
    """Create the ViT-Base/16 backbone used everywhere in the paper."""
    import timm

    vit = timm.create_model(checkpoint, pretrained=pretrained, **timm_kwargs)
    vit.eval()
    for p in vit.parameters():
        p.requires_grad_(False)
    return PromptViT(
        vit,
        num_prompts=num_prompts,
        prompt_pos=prompt_pos,
        prompt_init=prompt_init,
        prompt_init_bound=prompt_init_bound,
    )
