"""Common interface that the LVLM attack pipeline talks to.

Both :class:`robust_clip.lvlm.llava.LlavaOpenClip` and
:class:`robust_clip.lvlm.openflamingo.OpenFlamingoRunner` implement

* ``nll(images, prompts, continuations)`` -- the (differentiable) negative
  log-likelihood of ``continuations`` conditioned on
  ``(prompt, images)``.  This is the quantity the attacks optimise; and
* ``generate(images, prompts, **kwargs)`` -- greedy / beam decoding, used to
  obtain the actual caption or answer that is scored.

Everything is kept in *pixel space* (``[0, 1]`` tensors) so that the attacks
can be formulated as perturbations of the non-normalised inputs, as the paper
does.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import torch


class LVLM:
    """Abstract base class for the LVLMs used in the paper."""

    name: str = "lvlm"

    # ------------------------------------------------------------------ utils
    def set_precision(self, dtype: torch.dtype) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def to(self, device):  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def dtype(self) -> torch.dtype:  # pragma: no cover - abstract
        raise NotImplementedError

    # ------------------------------------------------------------------- core
    def nll(
        self,
        images: torch.Tensor,
        prompts: Sequence[str],
        continuations: Sequence[str],
    ) -> torch.Tensor:  # pragma: no cover - abstract
        """Per-sample mean NLL of ``continuation`` given ``prompt`` + ``image``."""
        raise NotImplementedError

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        prompts: Sequence[str],
        max_new_tokens: int = 32,
        **kwargs,
    ) -> List[str]:  # pragma: no cover - abstract
        raise NotImplementedError

    # ------------------------------------------------------------- convenience
    @torch.no_grad()
    def caption_batch(self, images: torch.Tensor, prompts: Sequence[str], **kwargs) -> List[str]:
        return self.generate(images, prompts, **kwargs)

    def describe(self) -> Dict[str, object]:
        return {"name": self.name, "dtype": str(self.dtype)}
