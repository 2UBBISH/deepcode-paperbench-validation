"""Test-time adaptation methods: FOA (ours) and the baselines of the paper."""
from .base import TTAMethod, features_and_logits, get_backbone  # noqa: F401
from .bn_adapt import BNAdapt  # noqa: F401
from .cotta import CoTTA  # noqa: F401
from .foa_method import FOAMethod  # noqa: F401
from .lame import LAME  # noqa: F401
from .no_adapt import NoAdapt  # noqa: F401
from .sar import SAR  # noqa: F401
from .t3a import T3A  # noqa: F401
from .tent import TENT  # noqa: F401

__all__ = [
    "TTAMethod",
    "features_and_logits",
    "get_backbone",
    "NoAdapt",
    "TENT",
    "SAR",
    "CoTTA",
    "LAME",
    "T3A",
    "BNAdapt",
    "FOAMethod",
]
