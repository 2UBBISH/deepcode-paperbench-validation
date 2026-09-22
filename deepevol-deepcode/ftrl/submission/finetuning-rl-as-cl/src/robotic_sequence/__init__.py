"""RoboticSequence (Meta-World) track of the paper reproduction.

Wołczyk et al. (2024), *Fine-tuning Reinforcement Learning Models is Secretly a
Forgetting Mitigation Problem* -- Appendix B.3 / Table 3 + Figures 3c, 7, 8, 20,
22, 26, 27 and Table 6.

This package implements

* ``env`` -- Algorithm 1: the sequential multi-stage Meta-World wrapper with the
  augmented success reward ``r'_t = beta * r_t * (T - t)`` and the normalised
  timestep ``t / T`` appended to the observation (``T = 200``, ``beta = 1.5``).
  The main sequence is ``hammer -> push -> peg-unplug-side -> push-wall`` where
  the first two stages are CLOSE and the last two are FAR (forgotten) stages.
* ``heads`` -- per-stage output-head primitives (one policy / Q head per stage,
  routed by the stage-ID one-hot).
* ``sac`` -- Soft Actor-Critic learner: 4x256 MLP with Leaky-ReLU and LayerNorm
  after the first layer, automatic entropy tuning, Adam (lr ``1e-3``), batch
  ``128``; plus the replay buffer with a protected prefix used by Episodic
  Memory.
* ``model`` -- architecture/assembly helpers (actor, twin-Q, target networks,
  optimizers, actor-only parameter lists for the retention losses, and feature
  hooks used by the CKA / PCA analyses).
* ``train_robotic`` -- the pipeline driver: pre-train ``pi_*`` on the FAR stages
  (last two), fine-tune the whole sequence with one of the four actor-only
  retention settings (``none`` / ``ewc`` / ``bc`` / ``em``), evaluate per-stage
  success rates, expert-action log-likelihood and forward transfer, and
  aggregate over >= 20 seeds with 90% confidence intervals.

Importing this package never requires ``torch``, ``metaworld`` or ``numpy``:
sub-modules and their public symbols are resolved lazily through PEP 562 so that
CPU-only / dependency-light environments can still import the package (for
example to inspect availability or to run the pure-python env stubs).
"""

from __future__ import annotations

import importlib
import types
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    # sub-modules
    "env",
    "heads",
    "sac",
    "model",
    "train_robotic",
    # helpers
    "load",
    "require",
    "available_modules",
    "missing_modules",
    "describe",
    "build_env",
    "build_agent",
    "main",
    # re-exports (resolved lazily)
    "RoboticSequenceEnv",
    "RoboticSequenceVecEnv",
    "RoboticSequenceTask",
    "DummyStageEnv",
    "make_stage_env",
    "tasks_for",
    "prefix_tasks",
    "per_stage_success_rate",
    "forward_transfer_metric",
    "augmented_reward",
    "normalized_timestep",
    "TIME_LIMIT",
    "BETA",
    "CLOSE_TASKS",
    "FAR_TASKS",
    "ROBOTIC_SEQUENCE_TASKS",
    "ALTERNATIVE_ORDERINGS",
    "SACAgent",
    "SACConfig",
    "SACPolicy",
    "SACTwinQ",
    "ReplayBuffer",
    "SACBatch",
    "SquashedNormal",
    "PerStageHeads",
    "MLPTrunk",
    "build_sac_agent",
    "build_actor",
    "build_critic",
    "build_models",
    "build_agent_from_config",
    "ActorCritic",
    "TargetNetwork",
    "soft_update",
    "actor_parameters",
    "actor_head_parameters",
    "policy_features",
    "layer_names",
    "save_model",
    "load_model",
    "RoboticModelConfig",
    "PolicyHead",
    "QHead",
    "PerStageHeadBank",
    "make_policy_heads",
    "make_q_heads",
    "stage_index",
    "one_hot_stage",
    "head_parameters",
    "pretrain_pistar",
    "finetune",
    "train_from_scratch",
    "evaluate_agent",
    "expert_log_likelihood",
    "forward_transfer_table",
    "run_experiment",
    "aggregate_seeds",
    "build_retention",
]

# --------------------------------------------------------------------------- #
# Lazy module registry
# --------------------------------------------------------------------------- #

_MODULE_PATHS: Dict[str, str] = {
    "env": "src.robotic_sequence.env",
    "heads": "src.robotic_sequence.heads",
    "sac": "src.robotic_sequence.sac",
    "model": "src.robotic_sequence.model",
    "train_robotic": "src.robotic_sequence.train_robotic",
    # aliases
    "trainer": "src.robotic_sequence.train_robotic",
    "training": "src.robotic_sequence.train_robotic",
    "environment": "src.robotic_sequence.env",
    "networks": "src.robotic_sequence.model",
}

# symbol -> (short module name, attribute name)
_REEXPORTS: Dict[str, Tuple[str, str]] = {
    # ---------------- env (Algorithm 1) ---------------- #
    "RoboticSequenceEnv": ("env", "RoboticSequenceEnv"),
    "RoboticSequenceVecEnv": ("env", "RoboticSequenceVecEnv"),
    "RoboticSequenceTask": ("env", "RoboticSequenceTask"),
    "DummyStageEnv": ("env", "DummyStageEnv"),
    "make_stage_env": ("env", "make_stage_env"),
    "tasks_for": ("env", "tasks_for"),
    "prefix_tasks": ("env", "prefix_tasks"),
    "strip_version": ("env", "strip_version"),
    "register_robotic_sequence_envs": ("env", "register_robotic_sequence_envs"),
    "per_stage_success_rate": ("env", "per_stage_success_rate"),
    "forward_transfer_metric": ("env", "forward_transfer_metric"),
    "augmented_reward": ("env", "augmented_reward"),
    "normalized_timestep": ("env", "normalized_timestep"),
    "TIME_LIMIT": ("env", "TIME_LIMIT"),
    "BETA": ("env", "BETA"),
    "CLOSE_TASKS": ("env", "CLOSE_TASKS"),
    "FAR_TASKS": ("env", "FAR_TASKS"),
    "ROBOTIC_SEQUENCE_TASKS": ("env", "ROBOTIC_SEQUENCE_TASKS"),
    "CONTINUAL_WORLD_TASK_ORDER": ("env", "CONTINUAL_WORLD_TASK_ORDER"),
    "ALTERNATIVE_ORDERINGS": ("env", "ALTERNATIVE_ORDERINGS"),
    # ---------------- heads ---------------- #
    "PolicyHead": ("heads", "PolicyHead"),
    "QHead": ("heads", "QHead"),
    "PerStageHeadBank": ("heads", "PerStageHeadBank"),
    "make_policy_heads": ("heads", "make_policy_heads"),
    "make_q_heads": ("heads", "make_q_heads"),
    "stage_index": ("heads", "stage_index"),
    "one_hot_stage": ("heads", "one_hot_stage"),
    "head_parameters": ("heads", "head_parameters"),
    # ---------------- sac ---------------- #
    "SACAgent": ("sac", "SACAgent"),
    "SACConfig": ("sac", "SACConfig"),
    "SACPolicy": ("sac", "SACPolicy"),
    "SACTwinQ": ("sac", "SACTwinQ"),
    "ReplayBuffer": ("sac", "ReplayBuffer"),
    "SACBatch": ("sac", "SACBatch"),
    "SquashedNormal": ("sac", "SquashedNormal"),
    "PerStageHeads": ("sac", "PerStageHeads"),
    "MLPTrunk": ("sac", "MLPTrunk"),
    "build_sac_agent": ("sac", "build_sac_agent"),
    "LOG_STD_MIN": ("sac", "LOG_STD_MIN"),
    "LOG_STD_MAX": ("sac", "LOG_STD_MAX"),
    # ---------------- model ---------------- #
    "build_actor": ("model", "build_actor"),
    "build_critic": ("model", "build_critic"),
    "build_models": ("model", "build_models"),
    "ActorCritic": ("model", "ActorCritic"),
    "TargetNetwork": ("model", "TargetNetwork"),
    "soft_update": ("model", "soft_update"),
    "make_optimizers": ("model", "make_optimizers"),
    "actor_parameters": ("model", "actor_parameters"),
    "actor_head_parameters": ("model", "actor_head_parameters"),
    "policy_features": ("model", "policy_features"),
    "layer_names": ("model", "layer_names"),
    "analyzer_layers": ("model", "analyzer_layers"),
    "save_model": ("model", "save_model"),
    "load_model": ("model", "load_model"),
    "RoboticModelConfig": ("model", "RoboticModelConfig"),
    # ---------------- train_robotic ---------------- #
    "pretrain_pistar": ("train_robotic", "pretrain_pistar"),
    "finetune": ("train_robotic", "finetune"),
    "train_from_scratch": ("train_robotic", "train_from_scratch"),
    "evaluate_agent": ("train_robotic", "evaluate_agent"),
    "expert_log_likelihood": ("train_robotic", "expert_log_likelihood"),
    "forward_transfer_table": ("train_robotic", "forward_transfer_table"),
    "run_experiment": ("train_robotic", "run_experiment"),
    "aggregate_seeds": ("train_robotic", "aggregate_seeds"),
    "build_retention": ("train_robotic", "build_retention"),
    "PretrainResult": ("train_robotic", "PretrainResult"),
    "FinetuneResult": ("train_robotic", "FinetuneResult"),
}

# Paper deliverables implemented by this package.
PAPER_FIGURES: Dict[str, Tuple[str, ...]] = {
    "figure_3c": ("train_robotic", "env"),
    "figure_7": ("train_robotic", "env"),
    "figure_8": ("train_robotic", "model"),
    "figure_20": ("train_robotic", "env"),
    "figure_22": ("train_robotic",),
    "figure_26": ("model", "train_robotic"),
    "figure_27": ("model", "train_robotic"),
    "table_3": ("sac", "model", "train_robotic"),
    "table_6": ("train_robotic",),
}

_CACHE: Dict[str, types.ModuleType] = {}
_FAILED: Dict[str, str] = {}


# --------------------------------------------------------------------------- #
# Lazy loading helpers
# --------------------------------------------------------------------------- #

def _resolve(name: str) -> List[str]:
    """Return candidate dotted paths for ``name`` (short name or dotted path)."""
    if not isinstance(name, str):
        raise TypeError("module name must be a string, got %r" % (type(name),))
    mapped = _MODULE_PATHS.get(name)
    candidates: List[str] = []
    if mapped:
        candidates.append(mapped)
    if "." in name:
        candidates.append(name)
    candidates.extend(
        [
            "src.robotic_sequence.%s" % name,
            "robotic_sequence.%s" % name,
            name,
        ]
    )
    seen = set()
    unique: List[str] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def load(name: str, required: bool = False) -> Optional[types.ModuleType]:
    """Lazily import a RoboticSequence sub-module.

    ``name`` may be a short name listed in :data:`_MODULE_PATHS` or a full dotted
    path.  Returns ``None`` when the module cannot be imported (for example when
    ``metaworld`` or ``torch`` is missing), unless ``required=True`` in which case
    the original :class:`ImportError` is re-raised.
    """
    if name in _CACHE:
        return _CACHE[name]

    last_error: Optional[BaseException] = None
    for candidate in _resolve(name):
        try:
            module = importlib.import_module(candidate)
        except Exception as exc:  # pragma: no cover - env dependent
            last_error = exc
            continue
        _CACHE[name] = module
        return module

    message = "could not import %r (%s)" % (name, last_error)
    _FAILED[name] = message
    if required:
        raise ImportError(message)
    return None


def require(name: str) -> types.ModuleType:
    """Import a sub-module or raise :class:`ImportError`."""
    module = load(name, required=True)
    assert module is not None  # for type checkers
    return module


def available_modules() -> List[str]:
    """Short names of the sub-modules that can currently be imported."""
    return sorted(
        short for short in _MODULE_PATHS if load(short) is not None
    )


def missing_modules() -> List[str]:
    """Short names of the sub-modules whose import failed."""
    return sorted(
        short for short in _MODULE_PATHS if load(short) is None
    )


def _torch_available() -> bool:
    try:  # pragma: no cover - trivial
        import importlib.util

        return importlib.util.find_spec("torch") is not None
    except Exception:  # pragma: no cover
        return False


def _metaworld_available() -> bool:
    try:  # pragma: no cover - trivial
        import importlib.util

        return importlib.util.find_spec("metaworld") is not None
    except Exception:  # pragma: no cover
        return False


def describe() -> Dict[str, Any]:
    """Availability report for the RoboticSequence sub-package."""
    available = available_modules()
    missing = missing_modules()
    return {
        "package": "src.robotic_sequence",
        "modules": sorted(_MODULE_PATHS),
        "available": available,
        "missing": missing,
        "figures": {k: list(v) for k, v in PAPER_FIGURES.items()},
        "torch": _torch_available(),
        "metaworld": _metaworld_available(),
        "paper": "Fine-tuning RL Models is Secretly a Forgetting Mitigation Problem",
    }


# --------------------------------------------------------------------------- #
# Convenience builders
# --------------------------------------------------------------------------- #

def build_env(task_order: str = "main", **kwargs: Any) -> Any:
    """Build a :class:`RoboticSequenceEnv` (see :mod:`src.robotic_sequence.env`)."""
    return require("env").RoboticSequenceEnv(task_order=task_order, **kwargs)


def build_agent(
    cfg: Any = None,
    obs_dim: Optional[int] = None,
    action_dim: Optional[int] = None,
    n_stages: int = 1,
    retention: Any = None,
    device: Optional[str] = None,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """Build a :class:`SACAgent` for RoboticSequence (Appendix B.3)."""
    sac = require("sac")
    return sac.build_sac_agent(
        cfg,
        obs_dim,
        action_dim,
        n_stages=n_stages,
        retention=retention,
        device=device,
        seed=seed,
        **kwargs,
    )


# ``build_agent_from_config`` mirrors :func:`build_agent` but reads the geometry
# from a loaded YAML config when the dimensions are not supplied explicitly.
def build_agent_from_config(
    cfg: Any,
    retention: Any = None,
    device: Optional[str] = None,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """Build a SAC agent using dimensions discovered from the environment."""
    env = build_env(task_order=getattr(cfg, "task_order", "main") if cfg else "main", stub=True)
    obs_dim = int(getattr(env, "observation_dim", 0) or 0)
    action_dim = int(getattr(env, "action_dim", 0) or 0)
    n_stages = int(getattr(env, "n_stages", 1) or 1)
    return build_agent(
        cfg,
        obs_dim,
        action_dim,
        n_stages=n_stages,
        retention=retention,
        device=device,
        seed=seed,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# PEP 562 lazy attribute access
# --------------------------------------------------------------------------- #

def __getattr__(name: str) -> Any:
    """Resolve sub-modules and re-exported symbols on first access."""
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)

    if name in _MODULE_PATHS:
        module = load(name, required=True)
        globals()[name] = module
        return module

    target = _REEXPORTS.get(name)
    if target is None:
        raise AttributeError(
            "module %r has no attribute %r" % (__name__, name)
        )

    short, attribute = target
    module = load(short, required=True)
    try:
        value = getattr(module, attribute)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise AttributeError(
            "module %r does not define %r (needed for %r)"
            % (short, attribute, name)
        ) from exc
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    return sorted(set(globals()) | set(__all__))


# --------------------------------------------------------------------------- #
# CLI: report package availability
# --------------------------------------------------------------------------- #

def main(argv: Optional[List[str]] = None) -> int:
    """Tiny CLI reporting which RoboticSequence sub-modules are importable."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="python -m src.robotic_sequence",
        description="Report availability of the RoboticSequence sub-package.",
    )
    parser.add_argument(
        "--require",
        nargs="*",
        default=None,
        metavar="MODULE",
        help="Fail if any of these sub-modules cannot be imported.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    args = parser.parse_args(argv)

    status = describe()
    for name in args.require or []:
        require(name)

    if args.json:
        print(json.dumps(status, indent=2, sort_keys=True))
    else:
        print("RoboticSequence package: %s" % __name__)
        print("  available: %s" % (", ".join(status["available"]) or "-"))
        print("  missing  : %s" % (", ".join(status["missing"]) or "-"))
        print("  torch=%s metaworld=%s" % (status["torch"], status["metaworld"]))
    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
