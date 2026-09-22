"""NetHack Human Monk fine-tuning track (Sections 3-5, Table 1, Appendix B.1).

This package implements the NetHack half of the reproduction of *Fine-tuning
Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem*
(Wolczyk et al., 2024):

* :mod:`src.nethack.encoders` - main-screen / blstats / message encoders
  (character + colour embedding lookup -> ResNet, two 2-layer MLPs).
* :mod:`src.nethack.model` - joint actor/critic with a shared LSTM
  (``hidden_dim = 1738``), policy head over 120 actions and baseline head,
  plus loading of the released 30M LSTM checkpoint (Tuyls et al., 2023).
* :mod:`src.nethack.dataset` - NLD-AA pipeline: 16 shards, local ``nld-aa-v0``
  sqlite database, ``TtyrecDataset`` batches used for the BC state buffer, the
  diagonal Fisher and expert log-likelihoods (~8000 Human Monk games).
* :mod:`src.nethack.env` - NLE wrappers with the paper's evaluation/rollout
  termination rules (death, 150 steps without progress, 100k steps).
* :mod:`src.nethack.appo_runner` - APPO fine-tuning with Table 1
  hyperparameters and the actor-only retention bundle (EWC / BC / KS).
* :mod:`src.nethack.pretrain_baseline` - value-head pre-training for 500M env
  steps with everything else frozen (Appendix B.1).
* :mod:`src.nethack.per_level_eval` - per-level (level 4, Sokoban) evaluation
  using 200 AutoAscend saves per level, every 25M environment steps.
* :mod:`src.nethack.train_nethack` - pipeline orchestration across the five
  variants (from-scratch, vanilla fine-tuning, +EWC, +BC, +KS) and seeds.

Everything is imported lazily (PEP 562) so that importing this package never
fails on machines without ``nle`` / ``torch`` installed with a GPU build.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "encoders",
    "model",
    "dataset",
    "env",
    "appo_runner",
    "pretrain_baseline",
    "per_level_eval",
    "train_nethack",
    # helpers
    "load",
    "require",
    "available_modules",
    "missing_modules",
    "describe",
    "main",
    # re-exports
    "NetHackModelConfig",
    "NetHackModel",
    "MiniNetHackModel",
    "ModelOutput",
    "NetHackActor",
    "build_model",
    "build_nethack_model",
    "build_actor",
    "load_lstm_checkpoint",
    "freeze_encoders",
    "encoder_output_dim",
    "split_observation",
    "OBSERVATION_KEYS",
    "observation_shapes",
    "make_env",
    "make_vec_env",
    "build_env",
    "evaluate_policy",
    "evaluate_episode",
    "TtyrecDataset",
    "open_dataset",
    "iterate_batches",
    "fisher_batches",
    "load_states",
    "sample_states",
    "build_dataset",
    "download_shards",
    "APPOConfig",
    "APPOAgent",
    "APPORunner",
    "RetentionBundle",
    "run_finetuning",
    "evaluate",
    "summarize_retention_config",
    "pretrain_baseline",
    "train_baseline",
    "load_baseline_model",
    "PerLevelEvaluator",
    "evaluate_per_level",
    "run_per_level_eval",
    "generate_saves",
    "NetHackPipelineConfig",
    "MethodRun",
    "PipelineResult",
    "run_pipeline",
    "TABLE1_DEFAULTS",
    "RETENTION_CONFIG",
    "TRAINING_METHODS",
    "BASELINE_PRETRAIN_STEPS",
    "PER_LEVEL_TARGETS",
]

#: short name -> dotted module path
_MODULE_PATHS: Dict[str, str] = {
    "encoders": "src.nethack.encoders",
    "model": "src.nethack.model",
    "dataset": "src.nethack.dataset",
    "env": "src.nethack.env",
    "appo_runner": "src.nethack.appo_runner",
    "pretrain_baseline": "src.nethack.pretrain_baseline",
    "per_level_eval": "src.nethack.per_level_eval",
    "train_nethack": "src.nethack.train_nethack",
    # convenient aliases
    "appo": "src.nethack.appo_runner",
    "pipeline": "src.nethack.train_nethack",
    "baseline": "src.nethack.pretrain_baseline",
}

#: re-exported symbol -> (short module, attribute)
_REEXPORTS: Dict[str, Tuple[str, str]] = {
    # model
    "NetHackModelConfig": ("model", "NetHackModelConfig"),
    "NetHackModel": ("model", "NetHackModel"),
    "MiniNetHackModel": ("model", "MiniNetHackModel"),
    "ModelOutput": ("model", "ModelOutput"),
    "NetHackActor": ("model", "NetHackActor"),
    "build_model": ("model", "build_model"),
    "build_nethack_model": ("model", "build_nethack_model"),
    "build_actor": ("model", "build_actor"),
    "load_lstm_checkpoint": ("model", "load_lstm_checkpoint"),
    "freeze_encoders": ("model", "freeze_encoders"),
    # encoders
    "NetHackEncoders": ("encoders", "NetHackEncoders"),
    "EncoderConfig": ("encoders", "EncoderConfig"),
    "build_encoders": ("encoders", "build_encoders"),
    "encoder_output_dim": ("encoders", "encoder_output_dim"),
    "split_observation": ("encoders", "split_observation"),
    # env
    "OBSERVATION_KEYS": ("env", "OBSERVATION_KEYS"),
    "observation_shapes": ("env", "observation_shapes"),
    "make_env": ("env", "make_env"),
    "make_vec_env": ("env", "make_vec_env"),
    "build_env": ("env", "build_env"),
    "evaluate_policy": ("env", "evaluate_policy"),
    "evaluate_episode": ("env", "evaluate_episode"),
    "NetHackEnv": ("env", "NetHackEnv"),
    "NetHackVecEnv": ("env", "NetHackVecEnv"),
    # dataset
    "TtyrecDataset": ("dataset", "TtyrecDataset"),
    "open_dataset": ("dataset", "open_dataset"),
    "iterate_batches": ("dataset", "iterate_batches"),
    "fisher_batches": ("dataset", "fisher_batches"),
    "load_states": ("dataset", "load_states"),
    "sample_states": ("dataset", "sample_states"),
    "build_dataset": ("dataset", "build_dataset"),
    "download_shards": ("dataset", "download_shards"),
    # appo
    "APPOConfig": ("appo_runner", "APPOConfig"),
    "APPOAgent": ("appo_runner", "APPOAgent"),
    "APPORunner": ("appo_runner", "APPORunner"),
    "RetentionBundle": ("appo_runner", "RetentionBundle"),
    "run_finetuning": ("appo_runner", "run_finetuning"),
    "evaluate": ("appo_runner", "evaluate"),
    "summarize_retention_config": ("appo_runner", "summarize_retention_config"),
    "TABLE1_DEFAULTS": ("appo_runner", "TABLE1_DEFAULTS"),
    "RETENTION_CONFIG": ("appo_runner", "RETENTION_CONFIG"),
    "TRAINING_METHODS": ("appo_runner", "TRAINING_METHODS"),
    # baseline pre-training
    "pretrain_baseline": ("pretrain_baseline", "pretrain_baseline"),
    "train_baseline": ("pretrain_baseline", "train_baseline"),
    "load_baseline_model": ("pretrain_baseline", "load_baseline_model"),
    "BASELINE_PRETRAIN_STEPS": ("pretrain_baseline", "BASELINE_PRETRAIN_STEPS"),
    # per-level evaluation
    "PerLevelEvaluator": ("per_level_eval", "PerLevelEvaluator"),
    "evaluate_per_level": ("per_level_eval", "evaluate_per_level"),
    "run_per_level_eval": ("per_level_eval", "run_per_level_eval"),
    "generate_saves": ("per_level_eval", "generate_saves"),
    "PER_LEVEL_TARGETS": ("per_level_eval", "PER_LEVEL_TARGETS"),
    # pipeline
    "NetHackPipelineConfig": ("train_nethack", "NetHackPipelineConfig"),
    "MethodRun": ("train_nethack", "MethodRun"),
    "PipelineResult": ("train_nethack", "PipelineResult"),
    "run_pipeline": ("train_nethack", "run_pipeline"),
}

#: paper deliverables produced by this sub-package
PAPER_FIGURES: Dict[str, Tuple[str, ...]] = {
    "figure_3a": ("train_nethack", "appo_runner"),
    "figure_5": ("per_level_eval", "dataset"),
    "table_1": ("appo_runner", "model", "env"),
    "table_4": ("env", "appo_runner"),
    "table_5": ("per_level_eval", "train_nethack"),
}

_CACHE: Dict[str, ModuleType] = {}
_FAILED: Dict[str, str] = {}


def _resolve(name: str) -> List[str]:
    """Return candidate dotted paths for ``name`` (deduplicated)."""
    if "." in name:
        candidates = [name]
    else:
        mapped = _MODULE_PATHS.get(name)
        candidates = [mapped] if mapped else []
        candidates += [
            "src.nethack.{}".format(name),
            "nethack.{}".format(name),
            name,
        ]
    out: List[str] = []
    for candidate in candidates:
        if candidate and candidate not in out:
            out.append(candidate)
    return out


def load(name: str, required: bool = False) -> Optional[ModuleType]:
    """Import a NetHack sub-module by short name or dotted path.

    Returns ``None`` when the module (or one of its heavy dependencies such as
    ``nle``) is unavailable, unless ``required`` is true.
    """
    if name in _CACHE:
        return _CACHE[name]
    if name in _FAILED and not required:
        return None

    last_error: Optional[BaseException] = None
    for candidate in _resolve(name):
        if candidate in _CACHE:
            return _CACHE[candidate]
        try:
            module = importlib.import_module(candidate)
        except Exception as exc:  # pragma: no cover - environment dependent
            last_error = exc
            continue
        _CACHE[name] = module
        _CACHE[candidate] = module
        return module

    message = "could not import '{}': {!r}".format(name, last_error)
    _FAILED[name] = message
    if required:
        raise ImportError(message)
    return None


def require(name: str) -> ModuleType:
    """Strict :func:`load` that always raises :class:`ImportError`."""
    module = load(name, required=True)
    assert module is not None  # for type checkers
    return module


def available_modules() -> List[str]:
    """Sorted short names of NetHack sub-modules that import successfully."""
    return sorted(name for name in _MODULE_PATHS if load(name) is not None)


def missing_modules() -> List[str]:
    """Sorted short names of NetHack sub-modules whose import failed."""
    return sorted(name for name in _MODULE_PATHS if load(name) is None)


def describe() -> Dict[str, Any]:
    """Availability report for the NetHack track."""
    available = available_modules()
    missing = missing_modules()
    return {
        "package": "src.nethack",
        "modules": dict(_MODULE_PATHS),
        "available": available,
        "missing": missing,
        "figures": dict(PAPER_FIGURES),
        "nle": load("env") is not None,
        "torch": _torch_available(),
    }


def _torch_available() -> bool:
    try:  # pragma: no cover - environment dependent
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def _cached(short: str, attribute: str) -> Any:
    module = load(short, required=True)
    try:
        return getattr(module, attribute)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise AttributeError(
            "module '{}' has no attribute '{}'".format(_MODULE_PATHS.get(short, short), attribute)
        ) from exc


def main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover - CLI helper
    """Report which NetHack sub-modules are importable in this environment."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="NetHack track status")
    parser.add_argument("--require", nargs="*", default=None,
                        help="modules that must be available (exit 1 otherwise)")
    parser.add_argument("--json", action="store_true", help="machine readable output")
    args = parser.parse_args(argv)

    if args.require:
        for name in args.require:
            require(name)

    report = describe()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("NetHack track: {} available, {} missing".format(
            len(report["available"]), len(report["missing"])))
        for name in report["available"]:
            print("  ok      {}".format(name))
        for name in report["missing"]:
            print("  missing {}".format(name))
    return 0


def __getattr__(name: str) -> Any:  # PEP 562
    if name in _MODULE_PATHS:
        module = load(name)
        if module is None:
            raise AttributeError(
                "NetHack sub-module '{}' is unavailable: {}".format(
                    name, _FAILED.get(name, "import failed"))
            )
        return module
    if name in _REEXPORTS:
        short, attribute = _REEXPORTS[name]
        value = _cached(short, attribute)
        globals()[name] = value
        return value
    raise AttributeError("module '{}' has no attribute '{}'".format(__name__, name))


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(__all__))
