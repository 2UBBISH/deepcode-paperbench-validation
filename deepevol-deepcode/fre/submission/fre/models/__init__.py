"""FRE model package.

Aggregates the components of the Functional Reward Encoding (FRE) model:

* :class:`~fre.models.reward_embedding.RewardEmbedding` -- scalar reward
  discretization (32 bins) + learned embedding table (64-d).
* :class:`~fre.models.fre_encoder.FREEncoder` -- permutation-invariant
  transformer VAE encoder ``p_theta(z | L^e)`` producing the 128-d task latent.
* :class:`~fre.models.fre_decoder.FREDecoder` -- feed-forward reward decoder
  ``q_theta(eta(s^d) | s^d, z)``.
* :class:`~fre.models.fre_model.FREModel` -- encoder + decoder + Eq. (6)
  variational information-bottleneck objective.

The z-conditioned IQL agent (``fre.models.iql``) and the RL network builders
(``fre.models.rl_networks``) are exposed lazily through the module level
``__getattr__`` so that this package remains importable while those modules are
being developed, and so that heavy optional dependencies are only imported when
actually used.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Tuple

# --- Core FRE components -----------------------------------------------------
from fre.models.reward_embedding import RewardEmbedding
from fre.models.fre_encoder import (
    FREEncoder,
    TransformerBlock,
    kl_divergence_to_unit_gaussian,
)
from fre.models.fre_decoder import FREDecoder, build_mlp
from fre.models.fre_model import (
    FREModel,
    fre_elbo_loss,
    DEFAULT_BETA,
    DEFAULT_NUM_ENCODER_PAIRS,
    DEFAULT_NUM_DECODER_PAIRS,
)

# --- Optional / later-stage components --------------------------------------
# Submodules that are imported lazily (name -> submodule attribute path).
_LAZY_SUBMODULES: Dict[str, str] = {
    "rl_networks": "fre.models.rl_networks",
    "iql": "fre.models.iql",
}


def __getattr__(name: str) -> Any:
    """Lazily import optional FRE submodules (``rl_networks``, ``iql``).

    This keeps ``import fre.models`` working before the RL-side modules exist
    and avoids eagerly importing the agent (which pulls in torch optimisers and
    dataset utilities) when only the encoder/decoder are needed.
    """
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(_LAZY_SUBMODULES[name])
        globals()[name] = module
        return module

    # Convenience: allow `from fre.models import IQLAgent` style access once the
    # module exists, without hard-coding names at import time.
    for submodule_name in _LAZY_SUBMODULES:
        try:
            module = importlib.import_module(_LAZY_SUBMODULES[submodule_name])
        except Exception:  # pragma: no cover - module not available yet
            continue
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> Tuple[str, ...]:
    return tuple(sorted(set(list(globals().keys()) + list(_LAZY_SUBMODULES.keys()))))


__all__ = [
    # reward embedding
    "RewardEmbedding",
    # encoder
    "FREEncoder",
    "TransformerBlock",
    "kl_divergence_to_unit_gaussian",
    # decoder
    "FREDecoder",
    "build_mlp",
    # full model
    "FREModel",
    "fre_elbo_loss",
    "DEFAULT_BETA",
    "DEFAULT_NUM_ENCODER_PAIRS",
    "DEFAULT_NUM_DECODER_PAIRS",
    # lazily exposed
    "rl_networks",
    "iql",
]
