"""Neural network components for Functional Reward Encodings (FRE).

This subpackage contains the three learned components of FRE:
    * :mod:`fre.models.encoder`     -- permutation-invariant transformer VIB encoder
      ``p_theta(z | (s^e, eta(s^e))_{1..K})`` producing a 128-d Gaussian latent ``z``.
    * :mod:`fre.models.decoder`     -- feed-forward reward decoder
      ``q_theta(eta(s^d) | s^d, z)`` with layers ``[512, 512, 512]``.
    * :mod:`fre.models.rl_networks` -- ``z``-conditioned ``Q(s, a, z)``, ``V(s, z)`` and
      ``pi(a | s, z)`` MLPs with layers ``[512, 512, 512]`` used by the IQL agent.

The public surface is exposed lazily (PEP 562) so that importing :mod:`fre.models`
never fails while individual modules are still being built out, and so that heavy
optional dependencies (``torch``) are only pulled in when actually needed.

Paper: "Zero-Shot Reinforcement Learning via Functional Reward Encodings"
(see also https://github.com/kvfrans/fre).
"""

from __future__ import annotations

from typing import Any, Dict, List

__all__: List[str] = [
    # encoder.py
    "FREEncoder",
    "EncoderOutput",
    "make_fre_encoder",
    # decoder.py
    "RewardDecoder",
    "make_reward_decoder",
    # rl_networks.py
    "QNetwork",
    "ValueNetwork",
    "PolicyNetwork",
    "RLNetworks",
    "make_rl_networks",
]

# Map public name -> submodule that defines it. Kept as a plain dict so the
# module can be imported without touching torch until an attribute is requested.
_LAZY_ATTRS: Dict[str, str] = {
    "FREEncoder": "encoder",
    "EncoderOutput": "encoder",
    "make_fre_encoder": "encoder",
    "RewardDecoder": "decoder",
    "make_reward_decoder": "decoder",
    "QNetwork": "rl_networks",
    "ValueNetwork": "rl_networks",
    "PolicyNetwork": "rl_networks",
    "RLNetworks": "rl_networks",
    "make_rl_networks": "rl_networks",
}


def __getattr__(name: str) -> Any:
    """Lazily import a public symbol from its defining submodule (PEP 562)."""
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f"{__name__}.{module_name}")
    try:
        return getattr(module, name)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise AttributeError(
            f"{module.__name__!r} does not define {name!r}"
        ) from exc


def __dir__() -> List[str]:
    return sorted(list(globals().keys()) + __all__)
