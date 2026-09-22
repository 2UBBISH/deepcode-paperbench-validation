"""CFG sampler: turn a pair of (conditional, unconditional) next-token logit vectors
into a single sampled token.

Pipeline order (paper Sec. 2.2, Eq. 7 + Addendum harness defaults):

    1. raw CFG logit combination        guided = uncond + gamma * (cond - uncond)
    2. temperature scaling             guided / temperature
    3. top-p (nucleus) filtering       keep the smallest set with cumulative p >= top_p
    4. softmax                         p = softmax(guided / T)
    5. multinomial sample              (or argmax when temperature == 0 / do_sample=False)

Notes
-----
* The CFG combination is applied to *raw pre-softmax logits* (Sec. 2.2: "sample the
  next i-th token w_i in the logits space").
* ``gamma == 1.0`` recovers the plain conditional model and ``gamma == 0.0`` the plain
  unconditional model (Eq. 7 collapses to log P(w_i | w_<i)).
* "Unless otherwise specified, models are sampled with the default hyperparameters in
  Eleuther's LLM evaluation harness" (Addendum).  The harness defaults used here are
  ``temperature = 0.0`` (greedy decoding) and ``top_p = 1.0`` (no truncation).
* HumanEval uses ``temperature in {0.2, 0.6, 0.8}`` (Sec. 3.3.1).
* CFG strengths swept in the paper: ``gamma in {1.0, 1.1, 1.25, 1.5, 1.75, 2.0}``
  (Sec. 3.3.1 fn. 3).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

try:  # pragma: no cover - import guard so the module is usable standalone
    from .logits import cfg_combine, softmax, top_p_filter
except ImportError:  # pragma: no cover
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from cfg.logits import cfg_combine, softmax, top_p_filter  # type: ignore

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Paper-level constants
# --------------------------------------------------------------------------------------

#: CFG strengths used across the paper (Sec. 3.3.1, fn. 3).
CFG_GAMMAS: Tuple[float, ...] = (1.0, 1.1, 1.25, 1.5, 1.75, 2.0)

#: Temperatures used for the HumanEval program-synthesis sweep (Sec. 3.3.1).
HUMANEVAL_TEMPERATURES: Tuple[float, ...] = (0.2, 0.6, 0.8)

#: Default harness decoding hyperparameters (Addendum: Eleuther lm-evaluation-harness).
HARNESS_DEFAULT_TEMPERATURE: float = 0.0
HARNESS_DEFAULT_TOP_P: float = 1.0

#: Entropy / overlap analysis uses gamma = 1.5 as the "CFG" operating point (Sec. 5.1).
ANALYSIS_GAMMA: float = 1.5


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


@dataclass
class SamplingConfig:
    """Decoding hyper-parameters.

    Attributes
    ----------
    gamma:
        CFG guidance strength. ``1.0`` -> vanilla conditional, ``0.0`` -> unconditional.
    temperature:
        Softmax temperature applied *after* the CFG combination. ``0.0`` (or
        ``do_sample=False``) means greedy decoding (the harness default).
    top_p:
        Nucleus probability mass. ``1.0`` disables nucleus filtering.
    top_k:
        0 disables top-k filtering. Kept for completeness; the paper does not use it.
    do_sample:
        If False, take the argmax of the filtered distribution (greedy).
    seed:
        Optional RNG seed for the multinomial draw (reproducible pass@k).
    """

    gamma: float = 1.0
    temperature: float = HARNESS_DEFAULT_TEMPERATURE
    top_p: float = HARNESS_DEFAULT_TOP_P
    top_k: int = 0
    do_sample: bool = True
    seed: Optional[int] = None
    extra: dict = field(default_factory=dict)

    def replace(self, **kwargs: Any) -> "SamplingConfig":
        """Return a copy of this config with ``kwargs`` overridden."""
        return replace(self, **kwargs)

    @property
    def greedy(self) -> bool:
        return (not self.do_sample) or self.temperature == 0.0 or self.temperature is None

    def as_dict(self) -> dict:
        return {
            "gamma": self.gamma,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "do_sample": self.do_sample,
            "seed": self.seed,
        }


# --------------------------------------------------------------------------------------
# Individual pipeline stages
# --------------------------------------------------------------------------------------


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Stage 2: divide logits by the temperature.

    ``temperature <= 0`` or ``None`` is a no-op (greedy decoding uses raw logits).
    """
    if temperature is None or temperature <= 0:
        return logits
    return logits / float(temperature)


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Keep only the ``top_k`` highest logits (set the rest to ``-inf``)."""
    if not top_k or top_k <= 0:
        return logits
    k = min(int(top_k), logits.shape[-1])
    kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Stage 3: nucleus (top-p) filtering on logits, delegated to ``logits.top_p_filter``."""
    if top_p is None or top_p >= 1.0:
        return logits
    return top_p_filter(logits, top_p)


def guided_logits_from_pair(
    logits_cond: torch.Tensor,
    logits_uncond: Optional[torch.Tensor] = None,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Stage 1: raw CFG combination (Eq. 7)."""
    if logits_uncond is None or gamma == 1.0:
        return logits_cond
    return cfg_combine(logits_uncond, logits_cond, gamma)


# --------------------------------------------------------------------------------------
# The sampler
# --------------------------------------------------------------------------------------


class CFGSampler:
    """Stateless-by-default sampler implementing the five-stage CFG pipeline."""

    def __init__(self, config: Optional[SamplingConfig] = None, generator: Optional[torch.Generator] = None):
        self.config = config or SamplingConfig()
        self.generator = generator
        if self.generator is None and self.config.seed is not None:
            self.generator = torch.Generator().manual_seed(int(self.config.seed))

    # -- configuration helpers ---------------------------------------------------------

    def set_gamma(self, gamma: float) -> "CFGSampler":
        self.config = self.config.replace(gamma=float(gamma))
        return self

    def set_temperature(self, temperature: float) -> "CFGSampler":
        self.config = self.config.replace(temperature=float(temperature))
        return self

    def with_config(self, **kwargs: Any) -> "CFGSampler":
        return CFGSampler(self.config.replace(**kwargs), generator=self.generator)

    def reseed(self, seed: Optional[int]) -> "CFGSampler":
        self.config = self.config.replace(seed=seed)
        self.generator = None if seed is None else torch.Generator().manual_seed(int(seed))
        return self

    # -- distributions -----------------------------------------------------------------

    def distribution(
        self,
        logits_cond: torch.Tensor,
        logits_uncond: Optional[torch.Tensor] = None,
        config: Optional[SamplingConfig] = None,
    ) -> torch.Tensor:
        """Return the post-CFG, post-temperature, post-top-p softmax distribution."""
        cfg = config or self.config
        guided = guided_logits_from_pair(logits_cond, logits_uncond, cfg.gamma)
        guided = apply_temperature(guided, cfg.temperature)
        # greedy path keeps raw (temperature-free) preferences for its argmax
        if cfg.greedy:
            guided = guided_logits_from_pair(logits_cond, logits_uncond, cfg.gamma)
        guided = apply_top_k(guided, cfg.top_k)
        guided = apply_top_p(guided, cfg.top_p)
        return softmax(guided, axis=-1)

    # -- token selection ---------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        logits_cond: torch.Tensor,
        logits_uncond: Optional[torch.Tensor] = None,
        config: Optional[SamplingConfig] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample the next token(s).

        Parameters
        ----------
        logits_cond:
            ``[..., vocab]`` conditional next-token logits (prompt ``c`` present).
        logits_uncond:
            ``[..., vocab]`` unconditional logits (prefix dropped or negative prompt
            ``c_bar``). ``None`` means vanilla conditional decoding.
        config / generator:
            Per-call overrides of the sampler defaults.

        Returns
        -------
        (tokens, probs):
            ``tokens`` has shape ``logits_cond.shape[:-1]`` and ``probs`` the full
            post-CFG distribution ``[..., vocab]``.
        """
        cfg = config or self.config
        gen = generator or self.generator

        guided = guided_logits_from_pair(logits_cond, logits_uncond, cfg.gamma)

        greedy = cfg.greedy
        if not greedy:
            guided = apply_temperature(guided, cfg.temperature)
        guided = apply_top_k(guided, cfg.top_k)
        guided = apply_top_p(guided, cfg.top_p)

        probs = softmax(guided, axis=-1)

        if greedy:
            tokens = torch.argmax(probs, dim=-1)
            return tokens, probs

        flat = probs.reshape(-1, probs.shape[-1])
        if gen is not None:
            idx = torch.multinomial(flat, num_samples=1, generator=gen).squeeze(-1)
        else:
            idx = torch.multinomial(flat, num_samples=1).squeeze(-1)
        tokens = idx.reshape(probs.shape[:-1])
        return tokens, probs

    # -- convenience -------------------------------------------------------------------

    def next_token(
        self,
        model_wrapper: Any,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        prompt_length: Optional[int] = None,
        negative_input_ids: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        config: Optional[SamplingConfig] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience wrapper around :class:`~src.cfg.model_wrapper.CFGModelWrapper`.

        Runs the dual forward pass and samples one token per batch row.
        """
        cfg = config or self.config
        dual = model_wrapper.dual_logits(
            input_ids,
            attention_mask=attention_mask,
            prompt_length=prompt_length,
            only_last=True,
            negative_input_ids=negative_input_ids,
            negative_attention_mask=negative_attention_mask,
        )
        return self.sample(dual.cond, dual.uncond, config=cfg)


# --------------------------------------------------------------------------------------
# Functional API (used by the generation loop and scripts)
# --------------------------------------------------------------------------------------


@torch.no_grad()
def cfg_sample(
    logits_cond: torch.Tensor,
    logits_uncond: Optional[torch.Tensor] = None,
    gamma: float = 1.0,
    temperature: float = HARNESS_DEFAULT_TEMPERATURE,
    top_p: float = HARNESS_DEFAULT_TOP_P,
    top_k: int = 0,
    do_sample: bool = True,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One-shot CFG sampling following the exact stage order of the paper."""
    sampler = CFGSampler(
        SamplingConfig(
            gamma=gamma,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
        ),
        generator=generator,
    )
    return sampler.sample(logits_cond, logits_uncond)


def greedy_token(
    logits_cond: torch.Tensor,
    logits_uncond: Optional[torch.Tensor] = None,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Argmax of the CFG-combined raw logits (temperature 0)."""
    guided = guided_logits_from_pair(logits_cond, logits_uncond, gamma)
    return torch.argmax(guided, dim=-1)


def cfG_overhead_factor(config: Optional[SamplingConfig] = None) -> float:
    """Inference FLOPs multiplier: 2 forward passes per token unless gamma == 1."""
    gamma = (config or SamplingConfig()).gamma
    return 1.0 if gamma == 1.0 else 2.0


def get_generator(seed: Optional[int], device: Union[str, torch.device] = "cpu") -> Optional[torch.Generator]:
    """Deterministic ``torch.Generator`` (``None`` seed -> global RNG)."""
    if seed is None:
        return None
    gen = torch.Generator(device="cpu" if str(device) == "auto" else device)
    gen.manual_seed(int(seed))
    return gen


def grid_configs(
    gammas: Tuple[float, ...] = CFG_GAMMAS,
    temperatures: Tuple[float, ...] = (HARNESS_DEFAULT_TEMPERATURE,),
    top_p: float = HARNESS_DEFAULT_TOP_P,
    seed: Optional[int] = 0,
):
    """Yield ``SamplingConfig`` for every (gamma, temperature) grid point."""
    for temp in temperatures:
        for gamma in gammas:
            yield SamplingConfig(
                gamma=float(gamma),
                temperature=float(temp),
                top_p=float(top_p),
                do_sample=(temp > 0),
                seed=seed,
            )


def numpy_distribution(probs: torch.Tensor) -> np.ndarray:
    """Detached float64 numpy view of a probability tensor (for analysis modules)."""
    return probs.detach().to(torch.float64).cpu().numpy()


__all__ = [
    "SamplingConfig",
    "CFGSampler",
    "cfg_sample",
    "greedy_token",
    "guided_logits_from_pair",
    "apply_temperature",
    "apply_top_p",
    "apply_top_k",
    "grid_configs",
    "get_generator",
    "cfG_overhead_factor",
    "numpy_distribution",
    "CFG_GAMMAS",
    "HUMANEVAL_TEMPERATURES",
    "HARNESS_DEFAULT_TEMPERATURE",
    "HARNESS_DEFAULT_TOP_P",
    "ANALYSIS_GAMMA",
]
