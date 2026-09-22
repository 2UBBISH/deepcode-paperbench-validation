"""Sampling subpackage for DPMs-ANT.

Exposes the adapted reverse-process samplers (DDPM / DDIM) used to generate
images from a pretrained diffusion model whose zero-initialized adaptors have
been fine-tuned by DPMs-ANT.

The public API is resolved lazily (PEP 562) so that importing
``dpm_ant.sampling`` does not eagerly pull in torch or the model stack.

Paper references
----------------
* Eq. (2): DDPM reverse step / training objective.
* Eq. (3): general (DDIM-style) reverse process with stochasticity ``eta``.
* Eq. (4): classifier-guided reverse process.
* Section 5.2 Configurations: DDIM ``eta = 0`` with 100 steps for evaluation,
  DDPM ``eta = 1`` with the full ``T = 1000`` steps optional. For the LDM
  backbone, sampling happens in the 64x64 latent space and the result is
  decoded with the frozen autoencoder.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List

__all__: List[str] = [
    # config
    "SamplingConfig",
    # samplers
    "Sampler",
    "DDIMSampler",
    "build_sampler",
    "sample_images",
    # functional wrappers
    "ddim_sample",
    "ddpm_sample",
]

_EXPORTS: Dict[str, str] = {
    "SamplingConfig": "sampler",
    "Sampler": "sampler",
    "DDIMSampler": "sampler",
    "build_sampler": "sampler",
    "sample_images": "sampler",
    "ddim_sample": "sampler",
    "ddpm_sample": "sampler",
}


def __getattr__(name: str) -> Any:  # pragma: no cover - trivial dispatch
    """Lazily resolve a public sampling symbol from its submodule."""
    if name in _EXPORTS:
        module = importlib.import_module(f".{_EXPORTS[name]}", __name__)
        try:
            value = getattr(module, name)
        except AttributeError as exc:  # pragma: no cover - defensive
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r} "
                f"(missing from {module.__name__!r})"
            ) from exc
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(__all__))
