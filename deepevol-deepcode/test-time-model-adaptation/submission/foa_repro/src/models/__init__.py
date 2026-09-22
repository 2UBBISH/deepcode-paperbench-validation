"""Model components for the FOA reproduction.

Two pieces live here:

* :mod:`src.models.vit_loader` -- the frozen timm ViT-Base backbone that exposes
  all-layer ``[CLS]`` token features through forward hooks, plus the
  classification head.  Nothing ever enables gradients on it.
* :mod:`src.models.prompt_injection` -- the small learnable input prompt that is
  prepended to the patch sequence as ``[CLS, prompts, patches]``.  It is the
  only mutable object in the whole pipeline and its flat vector of length
  ``d * N_p`` is the CMA-ES search space.

Importing this package must stay cheap and side-effect free: the heavy
``timm``/``torch`` imports happen inside the submodules themselves.
"""

from __future__ import annotations

__all__ = ["vit_loader", "prompt_injection"]

__version__ = "0.1.0"
