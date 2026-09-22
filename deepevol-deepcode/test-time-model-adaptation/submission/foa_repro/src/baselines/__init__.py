"""Baseline adapters for the FOA reproduction.

This sub-package contains thin, self-contained adapters for every method that FOA
is compared against on ImageNet-C / ImageNet-R / ImageNet-V2(MF) / ImageNet-Sketch
(paper Section 4 "Compared Methods", Appendix B.2 hyper-parameters):

======================  ==========================  ===================================
method                  module                      key config block
======================  ==========================  ===================================
LAME  (post-hoc)        :mod:`src.baselines.lame`   ``baselines.lame``  (kNN k=5, BS=64)
T3A   (post-hoc)        :mod:`src.baselines.t3a`    ``baselines.t3a``   (M=20, BS=64)
TENT  (gradient)        :mod:`src.baselines.tent`   ``baselines.tent``  (SGD, mom .9, lr 1e-3)
SAR   (gradient)        :mod:`src.baselines.sar`    ``baselines.sar``   (blocks 1-8, thresh .4 ln C)
CoTTA (gradient)        :mod:`src.baselines.cotta`  ``baselines.cotta`` (lr .05, p .01, EMA .999)
MEMO  (gradient)        :mod:`src.baselines.memo`   ``baselines.memo``  (32 views, per-sample)
======================  ==========================  ===================================

Every adapter exposes the same duck-typed online test-time-adaptation protocol so
that ``scripts/run_baselines.py`` can drive them interchangeably::

    obj = build_baseline(model, cfg)     # or the method-specific factory
    logits = obj.step(images, targets)   # adapt-then-predict for one batch
    obj.reset()                          # episodic reset between streams

All adapters are imported lazily / defensively so that a missing optional
dependency of one baseline never prevents the others from being used.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__version__ = "0.1.0"

#: Canonical baseline names in the order used for the paper's comparison tables.
BASELINE_NAMES: List[str] = ["noadapt", "lame", "t3a", "tent", "sar", "cotta", "memo"]

#: Baselines that never touch model weights (frozen backbone, post-hoc refinement).
POSTHOC_BASELINES: List[str] = ["noadapt", "lame", "t3a"]

#: Baselines that perform gradient updates at test time (used only for comparison).
GRADIENT_BASELINES: List[str] = ["tent", "sar", "cotta", "memo"]

#: module name -> factory symbol produced by that module (kept for reference only;
#: ``get_baseline`` resolves the symbol dynamically to tolerate naming variants).
_FACTORY_CANDIDATES: Dict[str, tuple] = {
    "lame": ("build_lame", "build_baseline", "build"),
    "t3a": ("build_t3a", "build_baseline", "build"),
    "tent": ("build_tent", "build_baseline", "build"),
    "sar": ("build_sar", "build_baseline", "build"),
    "cotta": ("build_cotta", "build_baseline", "build"),
    "memo": ("build_memo", "build_baseline", "build"),
}

#: module name -> class symbol, also resolved dynamically.
_CLASS_CANDIDATES: Dict[str, tuple] = {
    "lame": ("LAME",),
    "t3a": ("T3A", "T3ABaseline"),
    "tent": ("TENT",),
    "sar": ("SAR",),
    "cotta": ("CoTTA", "CoTTABaseline"),
    "memo": ("MEMO", "MEMOBaseline"),
}


def _import_module(name: str):
    """Import ``src.baselines.<name>`` tolerating both package and script layouts."""
    if name not in _FACTORY_CANDIDATES:
        raise KeyError(
            f"Unknown baseline '{name}'. Available: {', '.join(BASELINE_NAMES)}"
        )
    last_err: Optional[BaseException] = None
    for mod_name in (f"src.baselines.{name}", f"baselines.{name}", name):
        try:
            return importlib.import_module(mod_name)
        except ImportError as err:  # pragma: no cover - depends on sys.path
            last_err = err
    logger.warning("Baseline '%s' is unavailable (%s)", name, last_err)
    return None


def _resolve(module: Any, symbols: tuple):
    for symbol in symbols:
        obj = getattr(module, symbol, None)
        if obj is not None:
            return obj
    return None


def available_baselines() -> List[str]:
    """Return the subset of baseline names whose modules can be imported."""
    out: List[str] = []
    for name in BASELINE_NAMES:
        if name == "noadapt":
            out.append(name)  # implemented inline by the runner
            continue
        if _import_module(name) is not None:
            out.append(name)
    return out


def get_baseline_class(name: str):
    """Return the adapter class for ``name`` (or ``None`` when unavailable)."""
    module = _import_module(name)
    if module is None:
        return None
    return _resolve(module, _CLASS_CANDIDATES[name])


def get_baseline_factory(name: str):
    """Return the ``build_*`` factory for ``name`` (or ``None`` when unavailable)."""
    module = _import_module(name)
    if module is None:
        return None
    return _resolve(module, _FACTORY_CANDIDATES[name])


def build_baseline(name: str, model=None, cfg=None, device=None, **overrides):
    """Instantiate a baseline adapter by name.

    Mirrors the dispatch performed inside ``scripts/run_baselines.py``:
    prefer the module-level factory, fall back to constructing the class.
    """
    factory = get_baseline_factory(name)
    if factory is not None:
        try:
            return factory(model=model, cfg=cfg, device=device, **overrides)
        except TypeError:
            return factory(model, cfg, **overrides)
    cls = get_baseline_class(name)
    if cls is None:
        return None
    return cls(model=model, config=cfg, device=device, **overrides)


__all__ = [
    "__version__",
    "BASELINE_NAMES",
    "POSTHOC_BASELINES",
    "GRADIENT_BASELINES",
    "available_baselines",
    "get_baseline_class",
    "get_baseline_factory",
    "build_baseline",
]
