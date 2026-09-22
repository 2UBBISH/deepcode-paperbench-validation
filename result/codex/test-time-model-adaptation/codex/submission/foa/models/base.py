"""Common interface shared by every backbone that FOA can adapt.

FOA only ever needs three things from a backbone:

* the CLS token produced by *every* layer, ``{e_i^0}_{i=0..N}`` (Eqn. (2) / Eqn. (5)),
* the final CLS activation ``e_N^0``, which is the input of the task head and the
  quantity that the back-to-source activation shifting edits (Eqn. (7)),
* the logits ``Head(e_N^0)``.

The optional ``prompt`` argument injects the learnable prompt ``p`` (Section 3.1) and
``shift`` implements Eqn. (7), i.e. ``e_N^0 <- e_N^0 + gamma * d``.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class AdaptableModel(nn.Module):
    """Backbone wrapper exposing the forward-only adaptation hooks used by FOA."""

    #: number of transformer/convolutional stages, i.e. ``N``
    num_layers: int
    #: dimension ``d`` of a single prompt embedding
    embed_dim: int
    #: number of classes of the task head
    num_classes: int

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Turn a batch of images into the token sequence the backbone consumes.

        For a ViT this is ``[CLS, patch_1, ..., patch_m]``; for a ConvNet it is the image
        itself and for VisionMamba it is the patch sequence.
        """
        raise NotImplementedError

    def forward_tokens(
        self,
        tokens: torch.Tensor,
        prompt: Optional[torch.Tensor] = None,
        shift: Optional[torch.Tensor] = None,
        return_layers: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
        """Run the backbone on an already pre-processed token sequence.

        Args:
            prompt: ``[N_p, d]`` (or flattened ``[N_p * d]``) prompt embeddings that are
                concatenated right after the CLS token, i.e. the input sequence is
                ``[CLS, prompts, patch embeddings]``.
            shift: ``[d]`` vector ``gamma * d_t`` added to the final CLS activation
                (Eqn. (7)).  ``None`` disables the shift.
            return_layers: also return ``[e_0^0, ..., e_N^0]``.

        Returns:
            ``(logits, e_N_0, layer_cls_tokens)`` where ``layer_cls_tokens`` is ``None``
            unless ``return_layers`` is set.
        """
        raise NotImplementedError

    def forward_with_prompt(
        self,
        images: torch.Tensor,
        prompt: Optional[torch.Tensor] = None,
        shift: Optional[torch.Tensor] = None,
        return_layers: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
        """Convenience wrapper: ``preprocess`` + ``forward_tokens``."""
        return self.forward_tokens(
            self.preprocess(images), prompt=prompt, shift=shift, return_layers=return_layers
        )

    def input_token_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """Cacheable input tokens (used by the ``FOA-I V1`` interval strategy)."""
        return self.preprocess(images)

    # -- convenience -----------------------------------------------------------------
    @torch.no_grad()
    def cls_statistics(
        self,
        images: torch.Tensor,
        prompt: Optional[torch.Tensor] = None,
        precomputed_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Per-layer mean / std of the CLS tokens of a batch.

        Implements the "Statistics calculation" paragraph of Section 3.1, which is used
        both for the source in-distribution statistics ``{mu_i^S, sigma_i^S}`` and for
        the target statistics ``{mu_i(X_t), sigma_i(X_t)}`` of the fitness function
        (Eqn. (5)).
        """
        if precomputed_tokens is None:
            _, _, feats = self.forward_with_prompt(images, prompt=prompt, return_layers=True)
        else:
            _, _, feats = self.forward_tokens(
                precomputed_tokens, prompt=prompt, return_layers=True
            )
        assert feats is not None
        means = [f.mean(dim=0) for f in feats]
        stds = [f.std(dim=0, unbiased=False) for f in feats]
        return means, stds

    def prompt_parameter_count(self, num_prompts: int) -> int:
        return num_prompts * self.embed_dim

    @property
    def prompt_dim(self) -> int:
        """Dimension of the flattened prompt vector optimised by CMA-ES."""
        return int(getattr(self, "num_prompts", 1)) * int(self.embed_dim)
