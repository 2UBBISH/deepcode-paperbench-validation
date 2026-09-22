"""SMM experiment runners (ICML 2024: *Sample-specific Multi-channel Masks for
Visual Reprogramming*).

This package collects the high-level drivers that reproduce the paper's tables
and figures under the addendum scope:

===========================  ==================================================
Runner                       Reproduces
===========================  ==================================================
``run_main``                 Table 1 (ResNet-18/ResNet-50), Table 2 (ViT-B32)
                             with ``f_out = Ilm`` on the 11 target tasks.
``run_ablations``            Table 3 (masking strategies) and Figure 4
                             (patch-size study, ``l in {0,1,2,3,4}``).
``run_label_mappings``       Table 10 (``Rlm`` / ``Flm`` / ``Ilm`` study).
``run_scaling``              Table 11 (EuroSAT + ResNet-18 ``f_mask`` width
                             scaling).
``run_finetuning``           Table 13 (LoRA vs SMM) and Table 14
                             (Finetuning-FC with/without SMM).
``run_stanfordcars``         Table 12 (fine-grained failure case).
===========================  ==================================================

All runners share the paper's protocol: 200 epochs, milestones ``(100, 145)``,
batch size 256 (64 for DTD and OxfordPets), ``alpha_delta = 0.01`` with
``gamma_delta = 0.1``, and three seeds ``{0, 1, 2}`` aggregated as mean +/- std.

The imports below are guarded so that ``import smm_vr.experiments`` never fails
while the individual runners are being built incrementally.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

__all__: List[str] = []


def _extend(names: List[str]) -> None:
    """Append ``names`` to ``__all__`` without creating duplicates."""
    for name in names:
        if name not in __all__:
            __all__.append(name)


# ---------------------------------------------------------------------------
# Registry of experiment names -> (module, callable, short description)
# ---------------------------------------------------------------------------
EXPERIMENTS: Dict[str, Dict[str, str]] = {
    "main": {
        "module": "run_main",
        "entry": "run_main_experiment",
        "description": "Tables 1 and 2: ResNet-18/ResNet-50/ViT-B32 with Ilm.",
    },
    "ablations": {
        "module": "run_ablations",
        "entry": "run_ablations_experiment",
        "description": "Table 3 masking strategies and Figure 4 patch-size sweep.",
    },
    "label_mappings": {
        "module": "run_label_mappings",
        "entry": "run_label_mappings_experiment",
        "description": "Table 10: Rlm / Flm / Ilm comparison with and without SMM.",
    },
    "scaling": {
        "module": "run_scaling",
        "entry": "run_scaling_experiment",
        "description": "Table 11: EuroSAT + ResNet-18 f_mask channel scaling.",
    },
    "finetuning": {
        "module": "run_finetuning",
        "entry": "run_finetuning_experiment",
        "description": "Tables 13/14: LoRA vs SMM and Finetuning-FC (+SMM).",
    },
    "stanfordcars": {
        "module": "run_stanfordcars",
        "entry": "run_stanfordcars_experiment",
        "description": "Table 12: fine-grained failure case (all methods < 10%).",
    },
}

EXPERIMENT_NAMES: Tuple[str, ...] = tuple(EXPERIMENTS.keys())

EXPERIMENT_ALIASES: Dict[str, str] = {
    "main": "main",
    "table1": "main",
    "table2": "main",
    "tables12": "main",
    "ablation": "ablations",
    "ablations": "ablations",
    "table3": "ablations",
    "figure4": "ablations",
    "patch_size": "ablations",
    "label_mapping": "label_mappings",
    "label_mappings": "label_mappings",
    "table10": "label_mappings",
    "scaling": "scaling",
    "table11": "scaling",
    "finetune": "finetuning",
    "finetuning": "finetuning",
    "table13": "finetuning",
    "table14": "finetuning",
    "stanfordcars": "stanfordcars",
    "cars": "stanfordcars",
    "table12": "stanfordcars",
}


def canonical_experiment_name(name: str) -> str:
    """Normalise an experiment spelling to a key of :data:`EXPERIMENTS`."""
    if not name:
        return "main"
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    if key in EXPERIMENTS:
        return key
    if key in EXPERIMENT_ALIASES:
        return EXPERIMENT_ALIASES[key]
    squashed = key.replace("_", "")
    for candidate in EXPERIMENTS:
        if candidate.replace("_", "") == squashed:
            return candidate
    raise ValueError(
        "Unknown experiment {!r}; expected one of {}.".format(
            name, ", ".join(EXPERIMENT_NAMES)
        )
    )


def list_experiments() -> List[str]:
    """Return the canonical experiment names."""
    return list(EXPERIMENT_NAMES)


def describe_experiments() -> List[Dict[str, str]]:
    """Return metadata rows describing every experiment runner."""
    return [dict(name=name, **spec) for name, spec in EXPERIMENTS.items()]


def get_experiment(name: str):
    """Import and return the entry callable of an experiment runner.

    Parameters
    ----------
    name:
        Experiment name or alias (see :data:`EXPERIMENT_ALIASES`).

    Returns
    -------
    Callable
        The runner's ``run_<name>_experiment`` function.

    Raises
    ------
    ValueError
        If ``name`` is not a known experiment.
    ImportError
        If the runner module or its entry point is not available yet.
    """
    import importlib

    key = canonical_experiment_name(name)
    spec = EXPERIMENTS[key]
    module_name = "smm_vr.experiments." + spec["module"]
    module = importlib.import_module(module_name)
    entry = spec["entry"]
    if not hasattr(module, entry):
        for fallback in ("run", "main", "main_experiment"):
            if hasattr(module, fallback):
                return getattr(module, fallback)
        raise ImportError(
            "{} does not expose {!r}.".format(module_name, entry)
        )
    return getattr(module, entry)


def run_experiment(name: str, *args, **kwargs):
    """Dispatch to an experiment runner by name."""
    return get_experiment(name)(*args, **kwargs)


# ---------------------------------------------------------------------------
# Guarded re-exports (populated only when the runner modules import cleanly)
# ---------------------------------------------------------------------------
_MAIN_AVAILABLE = False
_ABLATIONS_AVAILABLE = False
_LABEL_MAPPINGS_AVAILABLE = False
_SCALING_AVAILABLE = False
_FINETUNING_AVAILABLE = False
_STANFORDCARS_AVAILABLE = False

try:  # pragma: no cover - depends on incremental build order
    from .run_main import (  # noqa: F401
        MAIN_DATASETS,
        TABLE1_AVERAGES,
        TABLE2_AVERAGES,
        build_all_methods,
        run_main_experiment,
        run_resnet18_experiment,
        run_resnet50_experiment,
        run_single_dataset,
        run_table1,
        run_table2,
        run_vit_b32_experiment,
    )

    _MAIN_AVAILABLE = True
    _extend(
        [
            "MAIN_DATASETS",
            "TABLE1_AVERAGES",
            "TABLE2_AVERAGES",
            "build_all_methods",
            "run_main_experiment",
            "run_resnet18_experiment",
            "run_resnet50_experiment",
            "run_single_dataset",
            "run_table1",
            "run_table2",
            "run_vit_b32_experiment",
        ]
    )
except ImportError:  # pragma: no cover
    pass

try:  # pragma: no cover - depends on incremental build order
    from .run_ablations import (  # noqa: F401
        ABLATION_RESULTS,
        MASKING_VARIANTS,
        PATCH_SIZES,
        PATCH_STUDY_L,
        run_ablation_experiment,
        run_ablations_experiment,
        run_masking_ablation,
        run_patch_size_study,
    )

    _ABLATIONS_AVAILABLE = True
    _extend(
        [
            "ABLATION_RESULTS",
            "MASKING_VARIANTS",
            "PATCH_SIZES",
            "PATCH_STUDY_L",
            "run_ablation_experiment",
            "run_ablations_experiment",
            "run_masking_ablation",
            "run_patch_size_study",
        ]
    )
except ImportError:  # pragma: no cover
    pass

try:  # pragma: no cover - depends on incremental build order
    from .run_label_mappings import (  # noqa: F401
        LABEL_MAPPING_AVERAGES,
        LABEL_MAPPINGS,
        run_label_mapping_experiment,
        run_label_mappings_experiment,
        run_single_mapping,
    )

    _LABEL_MAPPINGS_AVAILABLE = True
    _extend(
        [
            "LABEL_MAPPING_AVERAGES",
            "LABEL_MAPPINGS",
            "run_label_mapping_experiment",
            "run_label_mappings_experiment",
            "run_single_mapping",
        ]
    )
except ImportError:  # pragma: no cover
    pass

try:  # pragma: no cover - depends on incremental build order
    from .run_scaling import (  # noqa: F401
        SCALING_DATASET,
        SCALING_LEVELS,
        TABLE11_REFERENCE,
        build_scaled_mask_generator,
        run_scaling_experiment,
        run_scaling_study,
    )

    _SCALING_AVAILABLE = True
    _extend(
        [
            "SCALING_DATASET",
            "SCALING_LEVELS",
            "TABLE11_REFERENCE",
            "build_scaled_mask_generator",
            "run_scaling_experiment",
            "run_scaling_study",
        ]
    )
except ImportError:  # pragma: no cover
    pass

try:  # pragma: no cover - depends on incremental build order
    from .run_finetuning import (  # noqa: F401
        FINETUNE_METHODS,
        run_finetuning_experiment,
        run_table13,
        run_table14,
    )

    _FINETUNING_AVAILABLE = True
    _extend(
        [
            "FINETUNE_METHODS",
            "run_finetuning_experiment",
            "run_table13",
            "run_table14",
        ]
    )
except ImportError:  # pragma: no cover
    pass

try:  # pragma: no cover - depends on incremental build order
    from .run_stanfordcars import (  # noqa: F401
        STANFORDCARS_DATASET,
        TABLE12_REFERENCE,
        run_stanfordcars_experiment,
        run_stanfordcars_failure_case,
    )

    _STANFORDCARS_AVAILABLE = True
    _extend(
        [
            "STANFORDCARS_DATASET",
            "TABLE12_REFERENCE",
            "run_stanfordcars_experiment",
            "run_stanfordcars_failure_case",
        ]
    )
except ImportError:  # pragma: no cover
    pass


_extend(
    [
        "EXPERIMENTS",
        "EXPERIMENT_NAMES",
        "EXPERIMENT_ALIASES",
        "canonical_experiment_name",
        "list_experiments",
        "describe_experiments",
        "get_experiment",
        "run_experiment",
    ]
)
