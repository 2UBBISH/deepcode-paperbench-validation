"""Knowledge-retention mechanisms for fine-tuning RL models.

This package implements the four retention techniques studied in
Wołczyk et al. (2024), *Fine-tuning Reinforcement Learning Models is Secretly a
Forgetting Mitigation Problem* (Appendix C), together with the diagonal Fisher
estimator required by :class:`~src.retention.ewc.EWC`:

``ewc``
    ``L_aux(θ) = Σ_i F^i (θ_pre^i − θ^i)²`` -- Elastic Weight Consolidation
    (Section 2 "Knowledge retention", Appendix C.1).  Coefficient for NetHack:
    ``2e6``; for the Meta-World actor: ``100``.

``behavioral_cloning``
    ``L_BC(θ) = E_{s ~ B_BC}[ D_KL^s(π_* ‖ π_θ) ]`` -- distillation on a buffer
    of pre-training states (Section 2 "Behavioral cloning", Appendix C.2).
    NetHack scale ``2.0`` with no decay.

``kickstarting``
    ``L_KS(θ) = E_{s ~ B_θ}[ D_KL^s(π_*(·|s) ‖ π_θ(·|s)) ]`` -- reverse-KL
    distillation on online policy data (Section 2 "Kickstarting", Appendix
    C.2).  NetHack scale ``0.5`` with exponential decay ``0.99998`` per train
    step.

``episodic_memory``
    No auxiliary loss: pre-training transitions gathered with ``π_*`` are
    inserted into the off-policy replay buffer and a fixed fraction (10 % of a
    100k buffer) is protected from being overwritten (Appendix C.3).

``fisher``
    Diagonal Fisher Information Matrix of the policy at ``θ_*``, accumulated
    from the log-likelihood of expert actions (Appendix C.1 / Appendix B.1).
    NetHack uses 10000 NLD-AA batches of size 128.

All retention terms are applied to the **actor only** -- the critic coefficient
is always ``0``.

Every sub-module is imported lazily (PEP 562) so this package can be imported
on CPU-only machines that lack PyTorch; ``torch`` is only required when a
retention object is actually constructed.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple

__all__: List[str] = [
    # sub-modules
    "ewc",
    "fisher",
    "behavioral_cloning",
    "kickstarting",
    "episodic_memory",
    # helpers
    "load",
    "require",
    "available_modules",
    "missing_modules",
    "describe",
    "build_retention",
    "retention_config",
    "main",
    # re-exported symbols
    "EWC",
    "ewc_loss",
    "diagonal_fisher_penalty",
    "ewc_coef_for",
    "DEFAULT_EWC_COEF",
    "FisherEstimator",
    "compute_fisher_diagonal",
    "fisher_dot",
    "DEFAULT_FISHER_BATCHES",
    "DEFAULT_BATCH_SIZE",
    "BehavioralCloning",
    "BCBuffer",
    "CoefficientSchedule",
    "build_bc_buffer",
    "bc_loss",
    "bc_coef_for",
    "DEFAULT_BC_COEF",
    "DEFAULT_BC_MEMORY",
    "Kickstarting",
    "KickstartingLoss",
    "RolloutBuffer",
    "OnlineBuffer",
    "PerStepDecay",
    "kickstarting_loss",
    "ks_loss",
    "ks_coef_for",
    "ks_decay_for",
    "ks_config_for",
    "DEFAULT_KS_COEF",
    "DEFAULT_KS_DECAY",
    "EpisodicMemory",
    "EM",
    "EpisodicMemoryBuffer",
    "EMReplayBuffer",
    "PriorTaskBuffer",
    "MixedBatchSampler",
    "collect_trajectories",
    "em_fraction_for",
    "em_capacity_for",
    "DEFAULT_EM_FRACTION",
    "DEFAULT_EM_CAPACITY",
]


# ---------------------------------------------------------------------------
# module discovery / lazy loading
# ---------------------------------------------------------------------------

_MODULE_PATHS: Dict[str, str] = {
    "ewc": "src.retention.ewc",
    "fisher": "src.retention.fisher",
    "behavioral_cloning": "src.retention.behavioral_cloning",
    "bc": "src.retention.behavioral_cloning",
    "kickstarting": "src.retention.kickstarting",
    "ks": "src.retention.kickstarting",
    "episodic_memory": "src.retention.episodic_memory",
    "em": "src.retention.episodic_memory",
}

#: Cross-module aliases used across the code base (``BC = BehavioralCloning``,
#: ``KS = Kickstarting`` and ``EM = EpisodicMemory``).
_MODULE_ALIASES: Dict[str, str] = {
    "bc": "behavioral_cloning",
    "ks": "kickstarting",
    "em": "episodic_memory",
}

_REEXPORTS: Dict[str, Tuple[str, str]] = {
    # EWC (Appendix C.1)
    "EWC": ("ewc", "EWC"),
    "ewc_loss": ("ewc", "ewc_loss"),
    "diagonal_fisher_penalty": ("ewc", "diagonal_fisher_penalty"),
    "normalize_param_name": ("ewc", "normalize_param_name"),
    "ewc_coef_for": ("ewc", "ewc_coef_for"),
    "DEFAULT_EWC_COEF": ("ewc", "DEFAULT_EWC_COEF"),
    # Fisher (Appendix C.1 / Appendix B.1)
    "FisherEstimator": ("fisher", "FisherEstimator"),
    "compute_fisher_diagonal": ("fisher", "compute_fisher_diagonal"),
    "fisher_dot": ("fisher", "fisher_dot"),
    "DEFAULT_FISHER_BATCHES": ("fisher", "DEFAULT_FISHER_BATCHES"),
    "DEFAULT_BATCH_SIZE": ("fisher", "DEFAULT_BATCH_SIZE"),
    # Behavioral cloning (Appendix C.2)
    "BehavioralCloning": ("behavioral_cloning", "BehavioralCloning"),
    "BC": ("behavioral_cloning", "BehavioralCloning"),
    "BCBuffer": ("behavioral_cloning", "BCBuffer"),
    "CoefficientSchedule": ("behavioral_cloning", "CoefficientSchedule"),
    "build_bc_buffer": ("behavioral_cloning", "build_bc_buffer"),
    "bc_loss": ("behavioral_cloning", "bc_loss"),
    "kl_s_divergence": ("behavioral_cloning", "kl_s_divergence"),
    "sampled_kl": ("behavioral_cloning", "sampled_kl"),
    "freeze_teacher": ("behavioral_cloning", "freeze_teacher"),
    "bc_coef_for": ("behavioral_cloning", "bc_coef_for"),
    "DEFAULT_BC_COEF": ("behavioral_cloning", "DEFAULT_BC_COEF"),
    "DEFAULT_BC_MEMORY": ("behavioral_cloning", "DEFAULT_BC_MEMORY"),
    # Kickstarting (Appendix C.2)
    "Kickstarting": ("kickstarting", "Kickstarting"),
    "KickstartingLoss": ("kickstarting", "KickstartingLoss"),
    "KS": ("kickstarting", "Kickstarting"),
    "RolloutBuffer": ("kickstarting", "RolloutBuffer"),
    "OnlineBuffer": ("kickstarting", "OnlineBuffer"),
    "PerStepDecay": ("kickstarting", "PerStepDecay"),
    "kickstarting_loss": ("kickstarting", "kickstarting_loss"),
    "ks_loss": ("kickstarting", "ks_loss"),
    "ks_coef_for": ("kickstarting", "ks_coef_for"),
    "ks_decay_for": ("kickstarting", "ks_decay_for"),
    "ks_config_for": ("kickstarting", "ks_config_for"),
    "DEFAULT_KS_COEF": ("kickstarting", "DEFAULT_KS_COEF"),
    "DEFAULT_KS_DECAY": ("kickstarting", "DEFAULT_KS_DECAY"),
    # Episodic memory (Appendix C.3)
    "EpisodicMemory": ("episodic_memory", "EpisodicMemory"),
    "EM": ("episodic_memory", "EpisodicMemory"),
    "EpisodicMemoryBuffer": ("episodic_memory", "EpisodicMemoryBuffer"),
    "EMReplayBuffer": ("episodic_memory", "EMReplayBuffer"),
    "PriorTaskBuffer": ("episodic_memory", "PriorTaskBuffer"),
    "MixedBatchSampler": ("episodic_memory", "MixedBatchSampler"),
    "collect_trajectories": ("episodic_memory", "collect_trajectories"),
    "stack_batch": ("episodic_memory", "stack_batch"),
    "em_fraction_for": ("episodic_memory", "em_fraction_for"),
    "em_capacity_for": ("episodic_memory", "em_capacity_for"),
    "DEFAULT_EM_FRACTION": ("episodic_memory", "DEFAULT_EM_FRACTION"),
    "DEFAULT_EM_CAPACITY": ("episodic_memory", "DEFAULT_EM_CAPACITY"),
}

_CACHE: Dict[str, ModuleType] = {}
_FAILED: Dict[str, str] = {}

#: Retention method names accepted by :func:`build_retention`.
RETENTION_METHODS: Tuple[str, ...] = ("none", "ewc", "bc", "ks", "em")


#: Per-environment appendix-C hyperparameters (Appendix B.1 / B.3, Table 3).
_PAPER_RETENTION_CONFIG: Dict[str, Dict[str, Dict[str, Any]]] = {
    "nethack": {
        "ewc": {"coef": 2.0e6, "critic_coef": 0.0, "num_batches": 10000, "batch_size": 128},
        "bc": {"coef": 2.0, "critic_coef": 0.0, "decay": None, "memory_size": 10000},
        "ks": {"coef": 0.5, "critic_coef": 0.0, "decay": 0.99998, "decay_type": "exponential"},
        "em": {"coef": 0.0, "critic_coef": 0.0, "fraction": 0.1, "capacity": 100000},
    },
    "robotic_sequence": {
        "ewc": {"coef": 100.0, "critic_coef": 0.0, "num_batches": 1000, "batch_size": 128},
        "bc": {"coef": 1.0, "critic_coef": 0.0, "decay": None, "memory_size": 10000},
        "ks": {"coef": 1.0, "critic_coef": 0.0, "decay": None},
        "em": {"coef": 0.0, "critic_coef": 0.0, "fraction": 0.1, "capacity": 100000},
    },
    "metaworld": {
        "ewc": {"coef": 100.0, "critic_coef": 0.0, "num_batches": 1000, "batch_size": 128},
        "bc": {"coef": 1.0, "critic_coef": 0.0, "decay": None, "memory_size": 10000},
        "ks": {"coef": 1.0, "critic_coef": 0.0, "decay": None},
        "em": {"coef": 0.0, "critic_coef": 0.0, "fraction": 0.1, "capacity": 100000},
    },
    "montezuma": {
        "ewc": {"coef": 1.0, "critic_coef": 0.0, "num_batches": 1000, "batch_size": 128},
        "bc": {"coef": 1.0, "critic_coef": 0.0, "decay": None, "memory_size": 5000},
        "ks": {"coef": 1.0, "critic_coef": 0.0, "decay": None},
        "em": {"coef": 0.0, "critic_coef": 0.0, "fraction": 0.1, "capacity": 100000},
    },
}

#: Which paper deliverables each retention mechanism feeds (for reporting).
PAPER_FIGURES: Dict[str, Tuple[str, ...]] = {
    "figure_2": ("ewc", "behavioral_cloning", "kickstarting", "episodic_memory"),
    "figure_3a": ("ewc", "behavioral_cloning", "kickstarting"),
    "figure_3b": ("ewc", "behavioral_cloning"),
    "figure_3c": ("ewc", "behavioral_cloning", "episodic_memory"),
    "table_1": ("ewc", "behavioral_cloning", "kickstarting"),
    "table_3": ("ewc", "behavioral_cloning", "episodic_memory"),
    "table_4": ("ewc", "behavioral_cloning", "kickstarting"),
}


def _resolve(name: str) -> List[str]:
    """Return candidate dotted paths for ``name`` (de-duplicated, ordered)."""
    key = str(name).strip()
    candidates: List[str] = []
    mapped = _MODULE_PATHS.get(key)
    if mapped:
        candidates.append(mapped)
    if "." in key:
        candidates.append(key)
    else:
        candidates.extend((f"src.retention.{key}", f"retention.{key}", key))
    seen: set = set()
    ordered: List[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    return ordered


def load(name: str, required: bool = False) -> Optional[ModuleType]:
    """Lazily import a retention sub-module by short name or dotted path.

    ``ewc``, ``fisher``, ``behavioral_cloning`` (alias ``bc``), ``kickstarting``
    (alias ``ks``) and ``episodic_memory`` (alias ``em``) are understood.
    Returns ``None`` when the module cannot be imported, unless
    ``required=True``, in which case an :class:`ImportError` is raised.
    """
    key = str(name).strip()
    lookup = _MODULE_ALIASES.get(key, key)
    if lookup in _CACHE:
        return _CACHE[lookup]
    errors: List[str] = []
    for candidate in _resolve(key):
        try:
            module = import_module(candidate)
        except Exception as exc:  # pragma: no cover - environment dependent
            errors.append(f"{candidate}: {exc!r}")
            continue
        _CACHE[lookup] = module
        _FAILED.pop(lookup, None)
        return module
    message = "; ".join(errors) or f"unknown retention module {name!r}"
    _FAILED[lookup] = message
    if required:
        raise ImportError(f"could not import retention module {name!r}: {message}")
    return None


def require(name: str) -> ModuleType:
    """Strict variant of :func:`load` that always raises on failure."""
    module = load(name, required=True)
    assert module is not None  # for type checkers
    return module


def available_modules() -> List[str]:
    """Sorted short names of retention sub-modules that import successfully."""
    names = ("ewc", "fisher", "behavioral_cloning", "kickstarting", "episodic_memory")
    return sorted(name for name in names if load(name) is not None)


def missing_modules() -> List[str]:
    """Sorted short names of retention sub-modules that failed to import."""
    names = ("ewc", "fisher", "behavioral_cloning", "kickstarting", "episodic_memory")
    return sorted(name for name in names if load(name) is None)


def _cached(short: str, attribute: str) -> Any:
    lookup = _MODULE_ALIASES.get(short, short)
    module = load(lookup, required=True)
    try:
        return getattr(module, attribute)
    except AttributeError as exc:  # pragma: no cover - API drift
        raise AttributeError(
            f"module {lookup!r} has no attribute {attribute!r}"
        ) from exc


def retention_config(env_name: str = "nethack", method: str = "ewc") -> Dict[str, Any]:
    """Return the paper's appendix-C hyperparameters for an env/method pair."""
    env = str(env_name or "").strip().lower().replace("-", "_")
    if env in ("robotic", "roboticsequence", "robotic_sequence"):
        env = "robotic_sequence"
    if env in ("meta_world", "metaworld_v2"):
        env = "metaworld"
    if env in ("montezumas_revenge", "montezuma_revenge", "atari"):
        env = "montezuma"
    table = _PAPER_RETENTION_CONFIG.get(env, _PAPER_RETENTION_CONFIG["nethack"])
    key = _MODULE_ALIASES.get(str(method or "").strip().lower(), str(method or "").strip().lower())
    return dict(table.get(key, {}))


def build_retention(
    method: str,
    actor: Any,
    *,
    env_name: str = "nethack",
    teacher: Any = None,
    fisher: Any = None,
    bc_dataset: Any = None,
    buffer: Any = None,
    config: Any = None,
    device: Any = None,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """Instantiate one of the four actor-only retention mechanisms.

    ``method`` accepts ``"none"`` (returns ``None``), ``"ewc"``,
    ``"bc"``/``"behavioral_cloning"``, ``"ks"``/``"kickstarting"`` and
    ``"em"``/``"episodic_memory"``.  Coefficients default to the values
    reported in the paper for ``env_name`` (Appendix B.1 / B.3, Table 3) and can
    be overridden through ``kwargs`` (e.g. ``coef=...``).

    All mechanisms are applied to the **actor only**; the critic coefficient is
    always ``0``.
    """
    key = _MODULE_ALIASES.get(
        str(method or "none").strip().lower(), str(method or "none").strip().lower()
    )
    key = {
        "behavioral_cloning": "bc",
        "kickstarting": "ks",
        "episodic_memory": "em",
        "vanilla": "none",
        "scratch": "none",
    }.get(key, key)

    if key in ("none", "", "null", "no", "off"):
        return None
    if key not in ("ewc", "bc", "ks", "em"):
        raise ValueError(f"unknown retention method {method!r}")

    params = retention_config(env_name, key)
    params.update({k: v for k, v in kwargs.items() if v is not None})

    if key == "ewc":
        EWC = _cached("ewc", "EWC")
        return EWC(
            actor,
            fisher_diag=fisher,
            coef=params.get("coef", 1.0),
            normalize=params.get("normalize", False),
            **{
                k: v
                for k, v in params.items()
                if k not in ("coef", "normalize", "critic_coef", "num_batches", "batch_size", "mode")
            },
        )
    if key == "bc":
        BehavioralCloning = _cached("behavioral_cloning", "BehavioralCloning")
        return BehavioralCloning(
            actor,
            teacher=teacher,
            buffer=bc_dataset if bc_dataset is not None else buffer,
            coef=params.get("coef", 1.0),
            decay=params.get("decay"),
            device=device,
            **{
                k: v
                for k, v in params.items()
                if k not in ("coef", "decay", "critic_coef")
            },
        )
    if key == "ks":
        Kickstarting = _cached("kickstarting", "Kickstarting")
        return Kickstarting(
            actor,
            teacher=teacher,
            buffer=buffer,
            coef=params.get("coef", 1.0),
            decay=params.get("decay"),
            decay_type=params.get("decay_type", "exponential"),
            device=device,
            **{
                k: v
                for k, v in params.items()
                if k not in ("coef", "decay", "decay_type", "critic_coef")
            },
        )
    EpisodicMemory = _cached("episodic_memory", "EpisodicMemory")
    return EpisodicMemory(
        actor=actor,
        teacher=teacher,
        buffer=buffer,
        env_name=env_name,
        device=device,
        seed=seed,
        **{
            k: v
            for k, v in params.items()
            if k not in ("coef", "critic_coef")
        },
    )


def describe() -> Dict[str, Any]:
    """Return an availability/parameter report for the retention package."""
    modules = ("ewc", "fisher", "behavioral_cloning", "kickstarting", "episodic_memory")
    available: List[str] = []
    missing: Dict[str, str] = {}
    for name in modules:
        if load(name) is not None:
            available.append(name)
        else:
            missing[name] = _FAILED.get(name, "unavailable")
    return {
        "package": "src.retention",
        "modules": list(modules),
        "available": sorted(available),
        "missing": missing,
        "methods": list(RETENTION_METHODS),
        "figures": dict(PAPER_FIGURES),
        "config": _PAPER_RETENTION_CONFIG,
    }


def __getattr__(name: str) -> Any:
    """PEP 562 hook: resolve sub-modules and re-exported symbols lazily."""
    if name in ("_CACHE", "_FAILED", "_MODULE_PATHS", "_REEXPORTS", "_MODULE_ALIASES"):
        raise AttributeError(name)

    short = _MODULE_ALIASES.get(name, name)
    if short in _MODULE_PATHS:
        module = load(short, required=True)
        globals()[name] = module
        return module

    entry = _REEXPORTS.get(name)
    if entry is not None:
        value = _cached(entry[0], entry[1])
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(__all__))


def main(argv: Optional[List[str]] = None) -> int:
    """Small CLI reporting which retention mechanisms are importable."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="python -m src.retention",
        description="Knowledge-retention mechanisms for RL fine-tuning "
        "(Wołczyk et al., 2024).",
    )
    parser.add_argument(
        "--require",
        nargs="*",
        default=None,
        metavar="NAME",
        help="retention modules that must be importable (e.g. ewc bc ks em)",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument(
        "--config",
        nargs=2,
        default=None,
        metavar=("ENV", "METHOD"),
        help="print the paper hyperparameters for an env/method pair",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.config:
        payload = retention_config(args.config[0], args.config[1])
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.require:
        for name in args.require:
            require(name)

    report = describe()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("retention modules:")
        for name in report["modules"]:
            status = "ok" if name in report["available"] else "unavailable"
            print(f"  - {name}: {status}")
        print("methods: " + ", ".join(report["methods"]))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
