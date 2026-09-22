"""Functional Reward Encoding (FRE) core package.

This package contains the components of the FRE method from
"Zero-Shot Reinforcement Learning via Functional Reward Encodings" (ICML 2024):

* :mod:`fre.fre.reward_embedding` -- reward discretization into 32 bins plus a learned
  embedding table (Section 4.1 "Practical Implementation").
* :mod:`fre.fre.encoder` -- permutation-invariant transformer encoder
  ``p_theta(z | s^e_1, eta(s^e_1), ..., s^e_K, eta(s^e_K))`` producing a 128-dim Gaussian
  latent ``z`` (Section 4.1).
* :mod:`fre.fre.decoder` -- MLP decoder ``q_theta(eta(s) | s, z)`` with layers ``[512, 512, 512]``
  (Section 4.1, Appendix A).
* :mod:`fre.fre.fre_model` -- the joint encoder/decoder module together with the variational
  Information-Bottleneck objective of Equation (6): MSE + ``beta * KL(q(z|context) || N(0, I))``.
* :mod:`fre.fre.prior` -- the mixture prior over random unsupervised reward functions,
  a uniform mixture of singleton goal-reaching, random linear and random MLP functions
  (Section 4.2, Appendix B).
* :mod:`fre.fre.trainer` -- the strided training controller implementing Algorithm 1
  (encoder-only phase, then frozen-encoder policy phase).

The submodules are imported lazily (PEP 562 module ``__getattr__``) so that importing
``fre.fre`` does not eagerly pull in ``torch`` or any other heavy dependency, which keeps
configuration/inspection code import-light.
"""

from typing import TYPE_CHECKING

__all__ = [
    "RewardEmbedding",
    "Encoder",
    "TransformerEncoder",
    "Decoder",
    "FREModel",
    "PriorSampler",
    "RewardPrior",
    "FRETrainer",
]

# Mapping of public symbol -> (submodule, attribute). Resolved on first access.
_LAZY_ATTRS = {
    "RewardEmbedding": ("fre.fre.reward_embedding", "RewardEmbedding"),
    "Encoder": ("fre.fre.encoder", "Encoder"),
    "TransformerEncoder": ("fre.fre.encoder", "TransformerEncoder"),
    "Decoder": ("fre.fre.decoder", "Decoder"),
    "FREModel": ("fre.fre.fre_model", "FREModel"),
    "PriorSampler": ("fre.fre.prior", "PriorSampler"),
    "RewardPrior": ("fre.fre.prior", "RewardPrior"),
    "FRETrainer": ("fre.fre.trainer", "FRETrainer"),
}

if TYPE_CHECKING:  # pragma: no cover - typing aid only
    from fre.fre.decoder import Decoder  # noqa: F401
    from fre.fre.encoder import Encoder, TransformerEncoder  # noqa: F401
    from fre.fre.fre_model import FREModel  # noqa: F401
    from fre.fre.prior import PriorSampler, RewardPrior  # noqa: F401
    from fre.fre.reward_embedding import RewardEmbedding  # noqa: F401
    from fre.fre.trainer import FRETrainer  # noqa: F401


def __getattr__(name):  # pragma: no cover - exercised implicitly
    """Resolve lazily exported symbols (PEP 562)."""
    if name in _LAZY_ATTRS:
        import importlib

        module_name, attr_name = _LAZY_ATTRS[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():  # pragma: no cover - convenience only
    return sorted(list(globals().keys()) + __all__)
