"""Output label-mapping package for SMM (Sample-specific Multi-channel Masks).

This package implements the three non-parametric mappings from the target label
space ``Y^T`` onto a subset of the ImageNet (pre-trained) label space ``Y^P``
described in Sec. 2.3 / Appendix A.4 of the SMM paper:

* :mod:`smm_vr.label_mapping.rlm` -- :math:`f_{\\mathrm{out}}^{\\mathrm{Rlm}}`
  (Algorithm 2 of the paper's Appendix A.4 numbering is the frequency matrix;
  Rlm itself is a fixed random injective mapping chosen before training).
* :mod:`smm_vr.label_mapping.flm` -- :math:`f_{\\mathrm{out}}^{\\mathrm{Flm}}`
  (Algorithm 3): greedy injective selection from the frequency matrix, computed
  once before training and kept fixed.
* :mod:`smm_vr.label_mapping.ilm` -- :math:`f_{\\mathrm{out}}^{\\mathrm{Ilm}}`
  (Algorithm 4): the frequency matrix and injective mapping are recomputed
  before every epoch using the current reprogramming function ``f_in``. This is
  the default mapping for the main Tables 1 and 2.

The shared building block is :mod:`smm_vr.label_mapping.frequency`
(Algorithm 2): the integer count matrix
``d in Z^{|Y^P| x |Y^T|}`` of argmax predictions of ``f_P(f_in(x_i))``.

All mappings are stored as integer index buffers (no learnable parameters) so
they add zero trainable weights to the SMM pipeline. They all expose a uniform
interface consumed by the training/evaluation loops::

    logits_target = mapping(logits_imagenet)   # forward()
    mapping.update(...)                        # no-op for Rlm/Flm, refresh for Ilm
    mapping.recomputes_each_epoch              # bool flag for the epoch loop

This module re-exports the public API of the four submodules using guarded
imports so that ``import smm_vr.label_mapping`` never fails during incremental
construction.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Frequency matrix -- Algorithm 2
# ---------------------------------------------------------------------------
try:  # pragma: no cover - guarded for incremental builds
    from .frequency import (  # noqa: F401
        collect_predictions,
        compute_frequency_matrix,
        frequency_matrix_from_predictions,
        zero_frequency_matrix,
    )
except ImportError:  # pragma: no cover
    pass

# ---------------------------------------------------------------------------
# Flm -- Algorithm 3 (fixed greedy injective mapping)
# ---------------------------------------------------------------------------
try:  # pragma: no cover
    from .flm import (  # noqa: F401
        IGNORE_INDEX,
        FlmLabelMapping,
        FlmMapping,
        apply_label_mapping,
        build_flm_mapping,
        compute_flm_mapping,
        flm_mapping_from_frequency_matrix,
        flm_mapping_from_predictions,
    )
except ImportError:  # pragma: no cover
    pass

# ---------------------------------------------------------------------------
# Ilm -- Algorithm 4 (per-epoch recomputation; default for main tables)
# ---------------------------------------------------------------------------
try:  # pragma: no cover
    from .ilm import (  # noqa: F401
        IlmLabelMapping,
        IlmMapping,
        build_ilm_mapping,
        compute_ilm_mapping,
        ilm_mapping_from_frequency_matrix,
        ilm_mapping_from_predictions,
        update_ilm_mapping,
    )
except ImportError:  # pragma: no cover
    pass

# ---------------------------------------------------------------------------
# Rlm -- fixed random injective mapping chosen before training
# ---------------------------------------------------------------------------
try:  # pragma: no cover
    from .rlm import (  # noqa: F401
        RlmLabelMapping,
        RlmMapping,
        build_rlm_mapping,
        random_injective_mapping,
        random_subset,
        rlm_mapping_from_seed,
    )
except ImportError:  # pragma: no cover
    pass

# ``IGNORE_INDEX`` / ``apply_label_mapping`` are canonically defined in ``flm``
# but conceptually shared by every mapping; fall back to a local definition if
# the import above failed for any reason.
try:  # pragma: no cover
    IGNORE_INDEX  # type: ignore[used-before-def]
except NameError:  # pragma: no cover
    IGNORE_INDEX = -1  # type: ignore[assignment]

try:  # pragma: no cover
    apply_label_mapping  # type: ignore[used-before-def]
except NameError:  # pragma: no cover
    import torch as _torch

    def apply_label_mapping(logits, target_to_pretrained):  # type: ignore[misc]
        """Index-select matched ImageNet logits -> target-space logits."""
        index = _torch.as_tensor(target_to_pretrained, device=logits.device).long()
        return logits.index_select(-1, index)


# ---------------------------------------------------------------------------
# Uniform dispatch helpers
# ---------------------------------------------------------------------------
LABEL_MAPPINGS = ("rlm", "flm", "ilm")
DEFAULT_LABEL_MAPPING = "ilm"


def build_label_mapping(
    name: str = DEFAULT_LABEL_MAPPING,
    *,
    classifier=None,
    model=None,
    data_loader=None,
    num_target_classes: int,
    num_pretrained_classes: int = 1000,
    f_in=None,
    device=None,
    seed: int = 0,
    **kwargs,
):
    """Build an output label mapping by (case-insensitive) name.

    Parameters
    ----------
    name:
        One of ``"rlm"``, ``"flm"``, ``"ilm"`` (``"ilm"`` is the default used for
        the paper's main Tables 1 and 2).
    classifier, model:
        The frozen pre-trained classifier ``f_P`` (alias, either accepted).
    data_loader:
        Target-task training loader used to compute the frequency matrix
        (``flm``/``ilm`` only; ``rlm`` is random and ignores it).
    num_target_classes:
        Size of the target label space ``|Y^T|``.
    num_pretrained_classes:
        Size of the ImageNet label space ``|Y^P|`` (1000 for ImageNet-1K).
    f_in:
        Current reprogramming function; ``None`` means the identity used at
        initialisation for Flm.
    seed:
        Seed for the random subset drawn by Rlm.

    Returns
    -------
    A mapping module exposing ``forward``, ``update`` and
    ``recomputes_each_epoch``.
    """
    key = str(name).strip().lower()
    if key in ("rlm", "random"):
        return build_rlm_mapping(
            num_target_classes=num_target_classes,
            num_pretrained_classes=num_pretrained_classes,
            seed=seed,
            device=device,
            **kwargs,
        )
    if key in ("flm", "frequent", "frequency"):
        return build_flm_mapping(
            classifier if classifier is not None else model,
            data_loader,
            num_target_classes,
            num_pretrained_classes=num_pretrained_classes,
            f_in=f_in,
            device=device,
            **kwargs,
        )
    if key in ("ilm", "iterative", "iterated"):
        return build_ilm_mapping(
            classifier if classifier is not None else model,
            data_loader,
            num_target_classes,
            num_pretrained_classes=num_pretrained_classes,
            f_in=f_in,
            device=device,
            **kwargs,
        )
    raise ValueError(
        f"Unknown label mapping {name!r}; expected one of {LABEL_MAPPINGS}."
    )


def list_label_mappings():
    """Return the supported label-mapping names in paper order."""
    return list(LABEL_MAPPINGS)


__all__ = [
    # frequency (Algorithm 2)
    "compute_frequency_matrix",
    "frequency_matrix_from_predictions",
    "collect_predictions",
    "zero_frequency_matrix",
    # flm (Algorithm 3)
    "FlmMapping",
    "FlmLabelMapping",
    "build_flm_mapping",
    "compute_flm_mapping",
    "flm_mapping_from_frequency_matrix",
    "flm_mapping_from_predictions",
    # ilm (Algorithm 4)
    "IlmMapping",
    "IlmLabelMapping",
    "build_ilm_mapping",
    "compute_ilm_mapping",
    "ilm_mapping_from_frequency_matrix",
    "ilm_mapping_from_predictions",
    "update_ilm_mapping",
    # rlm
    "RlmMapping",
    "RlmLabelMapping",
    "build_rlm_mapping",
    "rlm_mapping_from_seed",
    "random_subset",
    "random_injective_mapping",
    # shared
    "apply_label_mapping",
    "IGNORE_INDEX",
    # dispatch
    "build_label_mapping",
    "list_label_mappings",
    "LABEL_MAPPINGS",
    "DEFAULT_LABEL_MAPPING",
]
