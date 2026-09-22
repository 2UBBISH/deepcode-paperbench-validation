"""Learnable input prompt for FOA (Forward-Optimization Adaptation).

Paper references
----------------
§3.1 "CMA-Based Prompt Adaptation":  "we add new prompt embeddings at the beginning of the
model input (i.e., before the first transformer layer) for test-time updating, while keeping
all other model parameters frozen" ... "p in R^{d x N_p} consists of N_p prompt embeddings,
each of dimension d."

§4 "Implementation Details": "We set the number of prompt embeddings N_p to 3 and initialize
prompts with uniform initialization."

Addendum: "The arrangement of input sequence elements is [CLS token, learnable prompts, patch
embeddings] in that specific order."

The prompt is the ONLY learnable object of FOA.  It is never updated by backpropagation --
CMA-ES samples candidates p_k^(t) ~ m^(t) + tau^(t) N(0, Sigma^(t)) (Eqn. 6) and the winning
candidate is simply injected into the frozen backbone for the next batch.  This module therefore
stores the prompt as a *buffer-like* tensor (``requires_grad_(False)``) and exposes a flat
vector interface (``get_prompt``/``set_prompt``) so that the CMA wrapper can ask for / inject
candidates of dimension d * N_p (= 2304 for ViT-Base with N_p = 3).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class PromptInjection(nn.Module):
    """Holds the (d, N_p) input prompt prepended to the ViT patch sequence.

    Parameters
    ----------
    embed_dim : int
        Hidden dimension ``d`` of the ViT (768 for ViT-Base).
    num_prompts : int
        Number of prompt embeddings ``N_p`` (paper default 3).  ``N_p = 0`` is allowed and means
        "no prompt" (used when collecting the source in-distribution statistics, since the paper
        states "The source in-distribution statistics {mu_i^S, sigma_i^S}_{i=0..N} are calculated
        without using the newly inserted prompt").
    init : str
        "uniform" (paper default) or "zeros" / "normal".
    init_range : float
        Half-width of the uniform initialisation ``U(-init_range, +init_range)``.  The paper only
        says "uniform initialization" without giving the range; we default to 0.01 and document it.
    seed : Optional[int]
        Seed used for the (deterministic) uniform initialisation.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_prompts: int = 3,
        init: str = "uniform",
        init_range: float = 0.01,
        seed: Optional[int] = 0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.num_prompts = int(num_prompts)
        self.init = init
        self.init_range = float(init_range)

        if self.num_prompts > 0:
            prompt = self._initialise(seed=seed, dtype=dtype)
        else:  # no prompt: keep an empty placeholder so state_dicts stay well formed
            prompt = torch.zeros(0, self.embed_dim, dtype=dtype)
        # ``requires_grad_(False)`` : FOA never backpropagates.
        self.prompt = nn.Parameter(prompt, requires_grad=False)
        if device is not None:
            self.to(device)

    # ------------------------------------------------------------------ helpers
    def _initialise(self, seed: Optional[int], dtype: torch.dtype) -> torch.Tensor:
        g = None
        if seed is not None:
            g = torch.Generator(device="cpu").manual_seed(int(seed))
        if self.init == "uniform":
            p = (torch.rand(self.num_prompts, self.embed_dim, generator=g, dtype=dtype) * 2.0 - 1.0)
            p = p * self.init_range
        elif self.init == "normal":
            p = torch.randn(self.num_prompts, self.embed_dim, generator=g, dtype=dtype) * self.init_range
        elif self.init == "zeros":
            p = torch.zeros(self.num_prompts, self.embed_dim, dtype=dtype)
        else:
            raise ValueError(f"Unknown prompt init: {self.init!r}")
        return p

    # ---------------------------------------------------------------- interface
    @property
    def prompt_dim(self) -> int:
        """Dimension of the flattened CMA search space: d * N_p."""
        return self.embed_dim * self.num_prompts

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.num_prompts, self.embed_dim)

    def as_tensor(self, batch_prompt: bool = True) -> torch.Tensor:
        """Return the prompt shaped for the backbone's ``_embed``.

        Returns ``[1, N_p, d]`` when ``batch_prompt=True`` (a single prompt shared by the whole
        batch, as in FOA where only the input prompt is adapted), else ``[N_p, d]``.
        """
        p = self.prompt
        if self.num_prompts == 0:
            return p.reshape(0, self.embed_dim).unsqueeze(0) if batch_prompt else p.reshape(0, self.embed_dim)
        return p.unsqueeze(0) if batch_prompt else p

    def get_prompt(self) -> torch.Tensor:
        """Flat prompt vector of length ``d * N_p`` (row-major over the N_p embeddings)."""
        return self.prompt.detach().reshape(-1).clone()

    def set_prompt(self, vector: torch.Tensor) -> None:
        """Inject a flat candidate vector (length ``d * N_p``) produced by CMA.

        Accepts either a flat tensor of length ``d*N_p`` or a ``[N_p, d]`` tensor.
        """
        v = torch.as_tensor(vector, dtype=self.prompt.dtype)
        if self.num_prompts == 0:
            return
        if v.dim() == 1:
            if v.numel() != self.prompt_dim:
                raise ValueError(
                    f"prompt vector has {v.numel()} elements, expected {self.prompt_dim}"
                )
            v = v.reshape(self.num_prompts, self.embed_dim)
        elif v.dim() == 2:
            if tuple(v.shape) != (self.num_prompts, self.embed_dim):
                raise ValueError(
                    f"prompt tensor has shape {tuple(v.shape)}, expected "
                    f"{(self.num_prompts, self.embed_dim)}"
                )
        else:
            raise ValueError(f"unsupported prompt tensor ndim={v.dim()}")
        with torch.no_grad():
            self.prompt.copy_(v.reshape(self.num_prompts, self.embed_dim).to(self.prompt.device))

    def forward(self, batch_size: Optional[int] = None) -> torch.Tensor:
        """Return the prompt as ``[1, N_p, d]`` (batch_size is accepted for API symmetry)."""
        return self.as_tensor(batch_prompt=True)

    # ------------------------------------------------------------------- repr
    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, num_prompts={self.num_prompts}, "
            f"init={self.init}, init_range={self.init_range}"
        )


def build_prompt(embed_dim: int = 768, num_prompts: int = 3, **kwargs) -> PromptInjection:
    """Factory matching the configuration dictionary of the plan
    (``cfg.prompt.num_prompts``, ``cfg.prompt.init``, ``cfg.prompt.init_range``)."""
    return PromptInjection(embed_dim=embed_dim, num_prompts=num_prompts, **kwargs)
