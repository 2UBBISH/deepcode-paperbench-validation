"""Attack engines for the Robust CLIP reproduction (package glue).

This package aggregates the attack implementations required by the paper /
Addendum:

* :mod:`robust_clip_repro.attacks.pgd` -- the Addendum-exact ``l_inf`` PGD
  (uniform random initialisation inside the ball, momentum ``0.9``, gradient
  normalised then element-wise signed, ``l_inf`` ball computed around
  **non-normalised** pixels, ``int16``/``int32`` perturbation storage).
* :mod:`robust_clip_repro.attacks.apgd` -- wrapper around the upstream
  ``fra31/robust-finetuning`` APGD implementation (upstream defaults preserved).
* :mod:`robust_clip_repro.attacks.jailbreak` -- the universal targeted visual
  jailbreak attacker (Qi et al., 2023) adapted from MiniGPT / LLaVA-LLaMA-2 13B
  to LLaVA-1.5 7B (5000 iterations, ``alpha = 1/255``, no momentum, a single
  ``clean.jpeg`` source image).
* :mod:`robust_clip_repro.attacks.vqa_schedule` -- the precision-graded VQA
  attack scheduler (top-5 low-precision attacks -> arg-min ground truth ->
  high-precision attack -> targeted ``"maybe"`` with a clean perturbation
  initialisation -> targeted ``"Word"`` with a second, separate clean
  initialisation, skipped on TextVQA).
* :mod:`robust_clip_repro.attacks.captioning` -- the staged captioning attack
  suite where CIDEr is recomputed immediately after *every* attack and only the
  per-sample worst case is retained.

Everything is resolved lazily through :pep:`562` module ``__getattr__`` so that
``import robust_clip_repro.attacks`` stays cheap and does not import
``torch``/``open_clip`` until an attribute is actually requested.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional, Tuple

__all__: List[str] = [
    # --- attacks.pgd (Addendum-exact PGD) ---
    "PGDLinfAttack",
    "PGDLinfConfig",
    "pgd_linf_attack",
    "MOMENTUM",
    # --- attacks.apgd (fra31/robust-finetuning wrapper) ---
    "APGDAttack",
    "APGDConfig",
    "apgd_attack",
    "apgd_logit_loss",
    "make_apgd_loss_fn",
    "UPSTREAM_DEFAULTS",
    "SUPPORTED_LOSSES",
    # --- attacks.jailbreak ---
    "JailbreakAttack",
    "JailbreakConfig",
    "perturb_jailbreak_image",
    "load_target_strings",
    "tokenize_target_ids",
    "build_targeted_loss_fn",
    "JAILBREAK_ITERATIONS",
    "JAILBREAK_ALPHA",
    "JAILBREAK_MOMENTUM",
    "SUPPORTED_JAILBREAK_LOSSES",
    # --- attacks.vqa_schedule ---
    "VQAAttackScheduler",
    "VQAAttackConfig",
    "AttackRequest",
    "VQAScheduleResult",
    "most_frequent_ground_truths",
    "most_frequent_ground_truth",
    "is_textvqa",
    "make_vqa_pgd_attack_fn",
    "TOP_K_GROUND_TRUTHS",
    "VQA_TARGET_MAYBE",
    "VQA_TARGET_WORD",
    "LOW_PRECISION",
    "HIGH_PRECISION",
    "STAGE_ORDER",
    "CALL_ORDER",
    # --- attacks.captioning ---
    "CaptioningAttackSuite",
    "CaptioningAttackConfig",
    "CaptioningAttackRequest",
    "CaptioningAttackRecord",
    "CaptioningAttackResult",
    "CaptioningSampleState",
    "make_apgd_attack_fn",
    "make_captioning_pgd_attack_fn",
    "make_cider_fn",
    "build_caption_loss_fn",
    "run_captioning_attack",
    "default_threshold",
    "NUM_GROUND_TRUTHS",
    "HALF_PRECISION",
    "SINGLE_PRECISION",
    "STAGE_HALF_PRECISION",
    "STAGE_SINGLE_PRECISION",
]

#: Submodules that make up this package (importable as
#: ``robust_clip_repro.attacks.<name>``).
_SUBMODULES: Tuple[str, ...] = ("pgd", "apgd", "jailbreak", "vqa_schedule", "captioning")

#: ``public name -> (submodule, attribute name or None)`` mapping used by the
#: lazy :func:`__getattr__`.  ``None`` means "same name as the public name".
#:
#: Ambiguous helpers that exist in more than one submodule (for instance
#: ``make_pgd_attack_fn`` in both ``vqa_schedule`` and ``captioning``) are
#: deliberately *not* re-exported under their bare name; they are exposed via
#: unambiguous aliases instead, and remain reachable through their own module
#: (``robust_clip_repro.attacks.captioning.make_pgd_attack_fn``).
_EXPORTS: Dict[str, Tuple[str, Optional[str]]] = {
    # --- attacks.pgd ---
    "PGDLinfAttack": ("pgd", None),
    "PGDLinfConfig": ("pgd", None),
    "pgd_linf_attack": ("pgd", None),
    "MOMENTUM": ("pgd", None),
    # --- attacks.apgd ---
    "APGDAttack": ("apgd", None),
    "APGDConfig": ("apgd", None),
    "apgd_attack": ("apgd", None),
    "apgd_logit_loss": ("apgd", None),
    "make_apgd_loss_fn": ("apgd", None),
    "UPSTREAM_DEFAULTS": ("apgd", None),
    "SUPPORTED_LOSSES": ("apgd", None),
    # --- attacks.jailbreak ---
    "JailbreakAttack": ("jailbreak", None),
    "JailbreakConfig": ("jailbreak", None),
    "perturb_jailbreak_image": ("jailbreak", None),
    "load_target_strings": ("jailbreak", None),
    "tokenize_target_ids": ("jailbreak", None),
    "build_targeted_loss_fn": ("jailbreak", None),
    "JAILBREAK_ITERATIONS": ("jailbreak", None),
    "JAILBREAK_ALPHA": ("jailbreak", None),
    "JAILBREAK_MOMENTUM": ("jailbreak", None),
    "SUPPORTED_JAILBREAK_LOSSES": ("jailbreak", "SUPPORTED_LOSSES"),
    # --- attacks.vqa_schedule ---
    "VQAAttackScheduler": ("vqa_schedule", None),
    "VQAAttackConfig": ("vqa_schedule", None),
    "AttackRequest": ("vqa_schedule", None),
    "VQAScheduleResult": ("vqa_schedule", None),
    "most_frequent_ground_truths": ("vqa_schedule", None),
    "most_frequent_ground_truth": ("vqa_schedule", None),
    "is_textvqa": ("vqa_schedule", None),
    "make_vqa_pgd_attack_fn": ("vqa_schedule", "make_pgd_attack_fn"),
    "TOP_K_GROUND_TRUTHS": ("vqa_schedule", None),
    "VQA_TARGET_MAYBE": ("vqa_schedule", None),
    "VQA_TARGET_WORD": ("vqa_schedule", None),
    "LOW_PRECISION": ("vqa_schedule", None),
    "HIGH_PRECISION": ("vqa_schedule", None),
    "STAGE_ORDER": ("vqa_schedule", None),
    "CALL_ORDER": ("vqa_schedule", None),
    # --- attacks.captioning ---
    "CaptioningAttackSuite": ("captioning", None),
    "CaptioningAttackConfig": ("captioning", None),
    "CaptioningAttackRequest": ("captioning", None),
    "CaptioningAttackRecord": ("captioning", None),
    "CaptioningAttackResult": ("captioning", None),
    "CaptioningSampleState": ("captioning", None),
    "make_apgd_attack_fn": ("captioning", None),
    "make_captioning_pgd_attack_fn": ("captioning", "make_pgd_attack_fn"),
    "make_cider_fn": ("captioning", None),
    "build_caption_loss_fn": ("captioning", None),
    "run_captioning_attack": ("captioning", None),
    "default_threshold": ("captioning", None),
    "NUM_GROUND_TRUTHS": ("captioning", None),
    "HALF_PRECISION": ("captioning", None),
    "SINGLE_PRECISION": ("captioning", None),
    "STAGE_HALF_PRECISION": ("captioning", None),
    "STAGE_SINGLE_PRECISION": ("captioning", None),
}


def _import_submodule(name: str) -> Any:
    """Import and cache one of this package's submodules."""
    module = importlib.import_module("{}.{}".format(__name__, name))
    globals()[name] = module
    return module


def _resolve(name: str) -> Any:
    """Resolve ``name`` through the lazy export table."""
    entry = _EXPORTS.get(name)
    if entry is None:
        raise AttributeError(
            "module {!r} has no attribute {!r}. Available lazy exports: {}".format(
                __name__, name, ", ".join(sorted(__all__))
            )
        )
    submodule_name, attribute = entry
    module = _import_submodule(submodule_name)
    try:
        value = getattr(module, attribute or name)
    except AttributeError as exc:  # pragma: no cover - guard for drift
        raise AttributeError(
            "module {!r} does not define {!r}: {}".format(
                "{}.{}".format(__name__, submodule_name), attribute or name, exc
            )
        ) from exc
    globals()[name] = value
    return value


def __getattr__(name: str) -> Any:  # PEP 562
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
    if name in _SUBMODULES:
        return _import_submodule(name)
    return _resolve(name)


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(__all__) | set(_SUBMODULES))


def available_attacks() -> List[str]:
    """Return the list of attack submodule names bundled with this package."""
    return list(_SUBMODULES)


def list_exports() -> Dict[str, str]:
    """Return ``public name -> "module.attribute"`` for every lazy export."""
    return {
        name: "{}.{}".format(submodule, attribute or name)
        for name, (submodule, attribute) in sorted(_EXPORTS.items())
    }


def preload(modules: Optional[Any] = None) -> Dict[str, bool]:
    """Eagerly import the named submodules (all by default).

    Returns a mapping ``submodule -> imported successfully``.  Failures are
    reported rather than raised so callers (e.g. smoke tests) can detect a
    missing optional dependency such as ``torch`` or ``open_clip``.
    """
    names = list(modules) if modules else list(_SUBMODULES)
    status: Dict[str, bool] = {}
    for name in names:
        try:
            _import_submodule(name)
            status[name] = True
        except Exception:  # pragma: no cover - depends on the environment
            status[name] = False
    return status
