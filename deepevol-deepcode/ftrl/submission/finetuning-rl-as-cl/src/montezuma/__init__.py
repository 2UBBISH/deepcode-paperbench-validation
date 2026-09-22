"""Montezuma's Revenge pipeline for the reproduction of

    "Fine-tuning Reinforcement Learning Models is Secretly a
     Forgetting Mitigation Problem"  (Wolczyk et al., 2024)

Sub-modules
-----------
``env``             Atari ``MontezumaRevengeNoFrameskip-v4`` wrapper with
                    sticky actions / frame stacking, room tracking
                    (Room 7 = FAR boundary) and rollout helpers.
``model``           Nature-CNN actor-critic, RND target/predictor pair and
                    intrinsic-reward & observation normalisers (Table 2).
``ppo_rnd``         PPO + Random Network Distillation agent, rollout storage,
                    GAE and the shared ``train_ppo_rnd`` entry point.
``m1_train``        Trains the exploration agent ``M1`` from scratch until the
                    episode return reaches ~7000 (Section 3 / Appendix B.2).
``m2_bc``           Collects Room-7-onward ``M1`` trajectories, behavioural-
                    clones ``M2`` (``pi_*``) and fine-tunes it on the whole game
                    with optional actor-only BC / EWC retention.
``train_montezuma`` Orchestration driver: M1 -> M2 -> fine-tuning variants,
                    multi-seed aggregation and the Figure-13 KL-weight sweep.

Everything here is lazy: importing :mod:`src.montezuma` must never fail when
``torch``, ``ale-py``/``gym`` or ``numpy`` are unavailable, because the toy,
robotic-sequence and retention tests are run on bare CPU containers.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any, Dict, List, Optional, Tuple

__all__: List[str] = [
    # sub-modules
    "env",
    "model",
    "ppo_rnd",
    "m1_train",
    "m2_bc",
    "train_montezuma",
    # helpers
    "load",
    "require",
    "available_modules",
    "missing_modules",
    "describe",
    # re-exported names
    "ENV_ID",
    "ROOM7",
    "OBS_SHAPE",
    "NUM_ACTIONS",
    "MAX_STEPS_PER_EPISODE",
    "MontezumaEnv",
    "DummyMontezumaEnv",
    "MontezumaVecEnv",
    "make_env",
    "collect_trajectories",
    "evaluate_policy",
    "room7_success_rate",
    "RoomTracker",
    "MontezumaModelConfig",
    "PolicyNetwork",
    "RNDModel",
    "build_policy",
    "build_rnd",
    "build_models",
    "TABLE2_DEFAULTS",
    "PPORNDConfig",
    "PPORNDAgent",
    "PPORNDTrainer",
    "RolloutStorage",
    "compute_gae",
    "train_ppo_rnd",
    "M1Result",
    "M1Config",
    "train_m1",
    "load_m1_agent",
    "save_m1_checkpoint",
    "evaluate_m1",
    "M2BCConfig",
    "BCDataset",
    "pretrain_m2",
    "finetune_m2",
    "train_from_scratch",
    "build_bc_dataset",
    "sweep_kl_weight",
    "MontezumaPipelineConfig",
    "PipelineResult",
    "MethodRun",
    "run_pipeline",
    "kl_weight_sweep",
]

_MODULE_PATHS: Dict[str, str] = {
    "env": "src.montezuma.env",
    "model": "src.montezuma.model",
    "ppo_rnd": "src.montezuma.ppo_rnd",
    "m1_train": "src.montezuma.m1_train",
    "m2_bc": "src.montezuma.m2_bc",
    "train_montezuma": "src.montezuma.train_montezuma",
    # convenience aliases used by the CLI dispatchers / trainers
    "ppo": "src.montezuma.ppo_rnd",
    "pipeline": "src.montezuma.train_montezuma",
    "m1": "src.montezuma.m1_train",
    "m2": "src.montezuma.m2_bc",
}

#: ``symbol -> (short module name, attribute)``.  Mirrors the public surface of
#: the Montezuma sub-modules so callers can do ``from src.montezuma import X``.
_REEXPORTS: Dict[str, Tuple[str, str]] = {
    # ---- env.py -----------------------------------------------------------
    "ENV_ID": ("env", "ENV_ID"),
    "ROOM7": ("env", "ROOM7"),
    "START_ROOM": ("env", "START_ROOM"),
    "MAX_ROOM": ("env", "MAX_ROOM"),
    "OBS_SHAPE": ("env", "OBS_SHAPE"),
    "NUM_ACTIONS": ("env", "NUM_ACTIONS"),
    "FRAME_STACK": ("env", "FRAME_STACK"),
    "ACTION_PROB": ("env", "ACTION_PROB"),
    "MAX_STEPS_PER_EPISODE": ("env", "MAX_STEPS_PER_EPISODE"),
    "MontezumaEnv": ("env", "MontezumaEnv"),
    "DummyMontezumaEnv": ("env", "DummyMontezumaEnv"),
    "MontezumaVecEnv": ("env", "MontezumaVecEnv"),
    "FrameStack": ("env", "FrameStack"),
    "RoomTracker": ("env", "RoomTracker"),
    "RoomEvent": ("env", "RoomEvent"),
    "make_env": ("env", "make_env"),
    "collect_trajectories": ("env", "collect_trajectories"),
    "evaluate_policy": ("env", "evaluate_policy"),
    "room7_success_rate": ("env", "room7_success_rate"),
    "room_visitation": ("env", "room_visitation"),
    "select_action": ("env", "select_action"),
    # ---- model.py ---------------------------------------------------------
    "MontezumaModelConfig": ("model", "MontezumaModelConfig"),
    "PolicyNetwork": ("model", "PolicyNetwork"),
    "RNDModel": ("model", "RNDModel"),
    "NatureCNN": ("model", "NatureCNN"),
    "RunningMeanStd": ("model", "RunningMeanStd"),
    "ObsNormalizer": ("model", "ObsNormalizer"),
    "IntrinsicRewardNormalizer": ("model", "IntrinsicRewardNormalizer"),
    "build_policy": ("model", "build_policy"),
    "build_rnd": ("model", "build_rnd"),
    "build_models": ("model", "build_models"),
    "save_model_state": ("model", "save_model_state"),
    "load_model_state": ("model", "load_model_state"),
    "TABLE2_DEFAULTS": ("model", "TABLE2_DEFAULTS"),
    # ---- ppo_rnd.py -------------------------------------------------------
    "PPORNDConfig": ("ppo_rnd", "PPORNDConfig"),
    "PPORNDAgent": ("ppo_rnd", "PPORNDAgent"),
    "PPORNDTrainer": ("ppo_rnd", "PPORNDTrainer"),
    "RolloutStorage": ("ppo_rnd", "RolloutStorage"),
    "compute_gae": ("ppo_rnd", "compute_gae"),
    "train_ppo_rnd": ("ppo_rnd", "train_ppo_rnd"),
    # ---- m1_train.py ------------------------------------------------------
    "M1Result": ("m1_train", "M1Result"),
    "M1Config": ("m1_train", "M1Config"),
    "train_m1": ("m1_train", "train_m1"),
    "load_m1_agent": ("m1_train", "load_m1_agent"),
    "save_m1_checkpoint": ("m1_train", "save_m1_checkpoint"),
    "evaluate_m1": ("m1_train", "evaluate_m1"),
    # ---- m2_bc.py ---------------------------------------------------------
    "M2BCConfig": ("m2_bc", "M2BCConfig"),
    "BCDataset": ("m2_bc", "BCDataset"),
    "M2PretrainResult": ("m2_bc", "M2PretrainResult"),
    "M2FinetuneResult": ("m2_bc", "M2FinetuneResult"),
    "pretrain_m2": ("m2_bc", "pretrain_m2"),
    "finetune_m2": ("m2_bc", "finetune_m2"),
    "train_from_scratch": ("m2_bc", "train_from_scratch"),
    "build_bc_dataset": ("m2_bc", "build_bc_dataset"),
    "build_bc_dataset_from_m1": ("m2_bc", "build_bc_dataset_from_m1"),
    "make_bc_aux_loss": ("m2_bc", "make_bc_aux_loss"),
    "sweep_kl_weight": ("m2_bc", "sweep_kl_weight"),
    "evaluate_room7_success_rate": ("m2_bc", "evaluate_room7_success_rate"),
    # ---- train_montezuma.py ----------------------------------------------
    "MontezumaPipelineConfig": ("train_montezuma", "MontezumaPipelineConfig"),
    "PipelineResult": ("train_montezuma", "PipelineResult"),
    "MethodRun": ("train_montezuma", "MethodRun"),
    "run_pipeline": ("train_montezuma", "run_pipeline"),
    "kl_weight_sweep": ("train_montezuma", "kl_weight_sweep"),
}

#: paper deliverables produced by this sub-package (used by ``describe``).
PAPER_FIGURES: Dict[str, Tuple[str, ...]] = {
    "figure_3b": ("train_montezuma", "m2_bc"),
    "figure_6": ("train_montezuma", "m2_bc", "env"),
    "figure_13": ("m2_bc", "train_montezuma"),
    "figure_17": ("train_montezuma",),
    "figure_18": ("train_montezuma",),
    "figure_19": ("train_montezuma",),
    "table_2": ("model", "ppo_rnd"),
}

_CACHE: Dict[str, ModuleType] = {}
_FAILED: Dict[str, str] = {}


def _resolve(name: str) -> List[str]:
    """Return candidate dotted paths for a short module name."""
    if name in _MODULE_PATHS:
        candidates = [_MODULE_PATHS[name]]
    elif "." in name:
        candidates = [name]
    else:
        candidates = []
    for candidate in (
        name,
        f"src.montezuma.{name}",
        f"montezuma.{name}",
        "src." + name if not name.startswith("src.") else name,
    ):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def load(name: str, required: bool = False) -> Optional[ModuleType]:
    """Import a Montezuma sub-module by short name or dotted path.

    Returns ``None`` when the module (or one of its optional dependencies)
    cannot be imported, unless ``required`` is true.
    """
    if name in _CACHE:
        return _CACHE[name]
    errors: List[str] = []
    for candidate in _resolve(name):
        try:
            module = importlib.import_module(candidate)
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"{candidate}: {exc!r}")
            continue
        _CACHE[name] = module
        _CACHE[candidate] = module
        _FAILED.pop(name, None)
        return module
    message = errors[-1] if errors else f"module {name!r} not found"
    _FAILED[name] = message
    if required:
        raise ImportError(
            f"could not import Montezuma sub-module {name!r}: {message}"
        )
    return None


def require(name: str) -> ModuleType:
    """Like :func:`load` but always raises when the import fails."""
    module = load(name, required=True)
    assert module is not None  # for type checkers
    return module


def available_modules() -> List[str]:
    """Short names of the Montezuma sub-modules that import successfully."""
    names = ("env", "model", "ppo_rnd", "m1_train", "m2_bc", "train_montezuma")
    return sorted(name for name in names if load(name) is not None)


def missing_modules() -> List[str]:
    """Short names of the Montezuma sub-modules whose import failed."""
    names = ("env", "model", "ppo_rnd", "m1_train", "m2_bc", "train_montezuma")
    return sorted(name for name in names if load(name) is None)


def describe() -> Dict[str, Any]:
    """Report the availability of the Montezuma pipeline and its figures."""
    available = available_modules()
    return {
        "package": "src.montezuma",
        "modules": list(_MODULE_PATHS),
        "available": available,
        "missing": missing_modules(),
        "figures": dict(PAPER_FIGURES),
        "torch": load("model").__name__ if load("model") is not None else None,
    }


def _cached(short: str, attribute: str) -> Any:
    module = _CACHE.get(short) or load(short)
    if module is None:
        raise AttributeError(
            f"module 'src.montezuma' has no attribute {attribute!r}: "
            f"sub-module {short!r} is unavailable ({_FAILED.get(short, 'unknown')})"
        )
    return getattr(module, attribute)


def __getattr__(name: str) -> Any:
    """PEP 562 hook: lazily resolve sub-modules and re-exported symbols."""
    if name in _CACHE:
        return _CACHE[name]
    if name in _MODULE_PATHS and name in ("env", "model", "ppo_rnd", "m1_train",
                                          "m2_bc", "train_montezuma"):
        module = load(name)
        if module is not None:
            return module
    if name in _REEXPORTS:
        short, attribute = _REEXPORTS[name]
        value = _cached(short, attribute)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'src.montezuma' has no attribute {name!r}")


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(__all__))


def main(argv: Optional[List[str]] = None) -> int:
    """Tiny CLI reporting the availability of the Montezuma pipeline."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="python -m src.montezuma",
        description="Montezuma's Revenge (M1 / M2) reproduction status",
    )
    parser.add_argument("--require", nargs="*", default=None,
                        help="sub-module names that must be importable")
    parser.add_argument("--json", action="store_true", help="machine readable")
    args = parser.parse_args(argv)

    info = describe()
    if args.require:
        for name in args.require:
            require(name)
    if args.json:
        print(json.dumps(info, indent=2, default=str))
    else:
        print("Montezuma's Revenge pipeline (src.montezuma)")
        print(f"  available : {', '.join(info['available']) or '-'}")
        print(f"  missing   : {', '.join(info['missing']) or '-'}")
        for figure, modules in sorted(info["figures"].items()):
            print(f"  {figure:10s}: {', '.join(modules)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
