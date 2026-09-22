"""Common interface of the LVLM wrappers."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn


class LVLM(nn.Module):
    """Interface used by the evaluation and attack code.

    All wrappers take images in ``[0, 1]`` (the perturbation budget of the
    attacks is defined on the non-normalized pixels) and strings as prompts.
    """

    #: text of the prompt used for the captioning tasks
    caption_prompt: str = "Describe this image in detail."

    def generate(
        self,
        images: torch.Tensor,
        prompts: Optional[Sequence[str]] = None,
        max_new_tokens: int = 32,
        num_beams: int = 1,
        do_sample: bool = False,
    ) -> List[str]:
        """Greedy / beam-search generation for a batch of ``[0, 1]`` images."""
        raise NotImplementedError

    def target_loss(
        self,
        images: torch.Tensor,
        prompts: Sequence[str],
        targets: Sequence[str],
        reduction: str = "none",
    ) -> torch.Tensor:
        """Differentiable per-sample cross entropy of ``targets`` given image + prompt.

        This is the objective of the attacks of Sec. 4.1/4.2 and of the
        jailbreaking attack of Qi et al. (2023) used in Sec. 4.4: the untargeted
        attacks maximize it with respect to the *ground-truth* caption / answer,
        the targeted attacks minimize it with respect to the *target* string.
        """
        raise NotImplementedError

    @torch.no_grad()
    def score(
        self,
        images: torch.Tensor,
        prompts: Sequence[str],
        candidates: Sequence[Sequence[str]],
    ) -> List[float]:
        """Score generated answers, e.g. CIDEr for captioning or accuracy for VQA.

        ``candidates[i]`` holds all valid ground-truth answers of sample ``i``.
        """
        raise NotImplementedError
