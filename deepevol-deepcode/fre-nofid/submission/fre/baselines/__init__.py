"""Baseline agents for the FRE reproduction.

This package aggregates the in-repo baseline comparators used in Table 1 of
"Zero-Shot Reinforcement Learning via Functional Reward Encodings":

* :mod:`fre.baselines.gc_iql` -- Goal-Conditioned IQL (IQL + goal concatenation)
* :mod:`fre.baselines.gc_bc`  -- Goal-Conditioned Behaviour Cloning
* :mod:`fre.baselines.opal`   -- OPAL-style variational skill model + z-conditioned IQL

The FB (Forward-Backward) and SF (Successor Features) baselines from
``facebookresearch/controllable_agent`` are **not** re-implemented here; only
their identifiers are recorded so that pipeline drivers (``fre/main.py``) can
report them as external results.

Imports are best-effort so that a partial install (e.g. a numpy-only CPU machine
without ``torch``) still exposes whichever baseline modules are importable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

__all__: List[str] = []


def _try_import(module: str, names: List[str]) -> None:
    """Best-effort re-export of ``names`` from ``module``.

    Failures (missing optional dependency, partially written module, ...) are
    swallowed so that the package stays importable.
    """
    try:
        mod = __import__(module, fromlist=["*"])
    except Exception:  # pragma: no cover - defensive
        return
    for name in names:
        try:
            value = getattr(mod, name)
        except AttributeError:
            continue
        globals()[name] = value
        if name not in __all__:
            __all__.append(name)


# --------------------------------------------------------------------------- #
# Requested baseline identifiers (from the plan's ``baselines/`` layout)
# --------------------------------------------------------------------------- #
IN_REPO_BASELINES = ("gc_iql", "gc_bc", "opal")
"""Baselines implemented inside this repository."""

EXTERNAL_BASELINES = ("fb", "sf")
"""Baselines obtained from https://github.com/facebookresearch/controllable_agent.

The paper's reproduction plan explicitly requires running these externally
(DDPG + ICM features for SF) rather than re-implementing them here.
"""

ALL_BASELINES = IN_REPO_BASELINES + EXTERNAL_BASELINES

BASELINE_DESCRIPTIONS: Dict[str, str] = {
    "gc_iql": "Goal-conditioned IQL; goal concatenated to observations; "
    "goal sampling 0.2 current / 0.5 geometric-future / 0.3 random; "
    "sparse reward 0 at goal else -1; eval conditioned on ground-truth goal.",
    "gc_bc": "Goal-conditioned behaviour cloning; 3x512 MLP + ReLU + LayerNorm; "
    "diagonal Gaussian head (log-std clamped at -5); MLE objective; "
    "geometric future-state goal sampling only.",
    "opal": "Unsupervised skill discovery (variational skill VAE) reusing the FRE "
    "transformer encoder; privileged eval samples 10 Gaussian skills and "
    "reports the best rollout (OPAL-10).",
    "fb": "Forward-Backward representations (external, controllable_agent).",
    "sf": "Successor Features with ICM features (external, controllable_agent).",
}


# --------------------------------------------------------------------------- #
# GC-IQL
# --------------------------------------------------------------------------- #
_try_import(
    __name__ + ".gc_iql",
    [
        # config
        "GCIConfig",
        # networks
        "GoalQNetwork",
        "GoalVNetwork",
        "GoalGaussianPolicy",
        # agent
        "GCIAgent",
        "GCIQL",
        "GCIQLAgent",
        # factories / loops
        "build_gc_iql",
        "train_gc_iql",
        # goal relabelling helpers
        "episode_boundaries",
        "sample_goal_indices",
        "relabel_goals",
        "sample_goal_batch",
        "sparse_goal_reward",
        # losses
        "expectile_loss",
        "awr_weights",
        # constants
        "DEFAULT_HIDDEN_SIZES",
        "DEFAULT_LR",
        "DEFAULT_BATCH_SIZE",
        "DEFAULT_DISCOUNT",
        "DEFAULT_EXPECTILE",
        "DEFAULT_AWR_TEMPERATURE",
        "DEFAULT_TARGET_UPDATE_RATE",
        "DEFAULT_P_CURRENT",
        "DEFAULT_P_FUTURE",
        "DEFAULT_P_RANDOM",
        "DEFAULT_GEOM_P",
        "DEFAULT_GOAL_TOLERANCE",
        # CLI
        "parse_args_gc_iql",
        "main_gc_iql",
    ],
)

# The GC-IQL module also exposes ``parse_args`` / ``main``; keep them under
# unambiguous aliases so they do not clobber the sibling baselines' entries.
try:  # pragma: no cover - defensive
    from . import gc_iql as _gc_iql_module

    if not hasattr(gc_iql_module := _gc_iql_module, "parse_args_gc_iql"):  # type: ignore[truthy-function]
        pass
    if hasattr(_gc_iql_module, "parse_args") and "parse_args_gc_iql" not in __all__:
        globals()["parse_args_gc_iql"] = _gc_iql_module.parse_args
        __all__.append("parse_args_gc_iql")
    if hasattr(_gc_iql_module, "main") and "main_gc_iql" not in __all__:
        globals()["main_gc_iql"] = _gc_iql_module.main
        __all__.append("main_gc_iql")
except Exception:  # pragma: no cover - defensive
    pass


# --------------------------------------------------------------------------- #
# GC-BC
# --------------------------------------------------------------------------- #
_try_import(
    __name__ + ".gc_bc",
    [
        # config
        "GCBConfig",
        # networks / agent
        "GaussianPolicy",
        "GCBCAgent",
        "GCBC",
        # factories / loops
        "build_gc_bc",
        "train_gc_bc",
        # goal relabelling helpers
        "geometric_future_goal_indices",
        "relabel_geometric_goals",
        "sample_goal_batch",
        # constants
        "DEFAULT_LOG_STD_CLAMP",
        "DEFAULT_MAX_GRAD_NORM",
        # CLI
        "parse_args_gc_bc",
        "main_gc_bc",
    ],
)

try:  # pragma: no cover - defensive
    from . import gc_bc as _gc_bc_module

    if hasattr(_gc_bc_module, "parse_args") and "parse_args_gc_bc" not in __all__:
        globals()["parse_args_gc_bc"] = _gc_bc_module.parse_args
        __all__.append("parse_args_gc_bc")
    if hasattr(_gc_bc_module, "main") and "main_gc_bc" not in __all__:
        globals()["main_gc_bc"] = _gc_bc_module.main
        __all__.append("main_gc_bc")
except Exception:  # pragma: no cover - defensive
    pass


# --------------------------------------------------------------------------- #
# OPAL
# --------------------------------------------------------------------------- #
_try_import(
    __name__ + ".opal",
    [
        # config
        "OPALConfig",
        # components
        "OPALSkillEncoder",
        "SkillDecoder",
        "ActionPrior",
        # agent
        "OPALAgent",
        "OPAL",
        "OPALAgentBaseline",
        # factories / loops
        "build_opal",
        "train_opal",
        # constants
        "DEFAULT_NUM_SKILLS",
        "DEFAULT_NUM_EVAL_SKILLS",
        "DEFAULT_DECODER_HIDDEN",
        "DEFAULT_STEPS",
        "DEFAULT_SKILL_BETA",
        "DEFAULT_ACTION_PRIOR_COEFF",
        "DEFAULT_CONTEXT_SIZE",
        "DEFAULT_LATENT_DIM",
        "DEFAULT_NUM_BLOCKS",
        "DEFAULT_NUM_HEADS",
        "DEFAULT_MLP_DIM",
        # CLI
        "parse_args_opal",
        "main_opal",
    ],
)

try:  # pragma: no cover - defensive
    from . import opal as _opal_module

    if hasattr(_opal_module, "parse_args") and "parse_args_opal" not in __all__:
        globals()["parse_args_opal"] = _opal_module.parse_args
        __all__.append("parse_args_opal")
    if hasattr(_opal_module, "main") and "main_opal" not in __all__:
        globals()["main_opal"] = _opal_module.main
        __all__.append("main_opal")
except Exception:  # pragma: no cover - defensive
    pass


# --------------------------------------------------------------------------- #
# Uniform baseline factory (used by fre/main.py)
# --------------------------------------------------------------------------- #
def available_baselines() -> Dict[str, bool]:
    """Return which in-repo baselines were successfully imported."""
    return {
        "gc_iql": "GCIAgent" in globals() or "build_gc_iql" in globals(),
        "gc_bc": "GCBCAgent" in globals() or "build_gc_bc" in globals(),
        "opal": "OPALAgent" in globals() or "build_opal" in globals(),
    }


def build_baseline(
    name: str,
    obs_dim: int,
    action_dim: int,
    config: Optional[Any] = None,
    device: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """Construct an in-repo baseline agent by name.

    Args:
        name: one of ``"gc_iql"``, ``"gc_bc"``, ``"opal"`` (case/hyphen tolerant).
        obs_dim: observation dimensionality.
        action_dim: action dimensionality.
        config: optional baseline-specific config dataclass.
        device: optional torch device.
        **kwargs: forwarded to the baseline-specific ``build_*`` factory.

    Raises:
        ValueError: if ``name`` is unknown.
        ImportError: if the requested baseline module could not be imported.
    """
    key = str(name).strip().lower().replace("-", "_")
    factories = {
        "gc_iql": ("build_gc_iql", globals().get("build_gc_iql")),
        "gc_bc": ("build_gc_bc", globals().get("build_gc_bc")),
        "opal": ("build_opal", globals().get("build_opal")),
    }
    if key in ("fb", "sf"):
        raise NotImplementedError(
            f"The '{key}' baseline is external; run "
            "facebookresearch/controllable_agent instead of importing it here."
        )
    if key not in factories:
        raise ValueError(
            f"Unknown baseline '{name}'. Available: {sorted(factories)} "
            f"(external: {list(EXTERNAL_BASELINES)})"
        )
    factory_name, factory = factories[key]
    if factory is None:
        raise ImportError(
            f"Baseline '{key}' is unavailable (module failed to import). "
            f"Available: {[k for k, v in available_baselines().items() if v]}"
        )
    return factory(obs_dim, action_dim, config=config, device=device, **kwargs)


def train_baseline(
    name: str,
    buffer: Any,
    obs_dim: int,
    action_dim: int,
    config: Optional[Any] = None,
    device: Optional[Any] = None,
    **kwargs: Any,
) -> Any:
    """Train an in-repo baseline on an offline replay buffer.

    Mirrors :func:`build_baseline` but dispatches to the matching ``train_*``
    driver (``train_gc_iql`` / ``train_gc_bc`` / ``train_opal``).
    """
    key = str(name).strip().lower().replace("-", "_")
    trainers = {
        "gc_iql": globals().get("train_gc_iql"),
        "gc_bc": globals().get("train_gc_bc"),
        "opal": globals().get("train_opal"),
    }
    if key not in trainers:
        raise ValueError(
            f"Unknown baseline '{name}'. Available: {sorted(trainers)} "
            f"(external: {list(EXTERNAL_BASELINES)})"
        )
    trainer = trainers[key]
    if trainer is None:
        raise ImportError(
            f"Baseline '{key}' is unavailable (module failed to import). "
            f"Available: {[k for k, v in available_baselines().items() if v]}"
        )
    return trainer(
        buffer, obs_dim, action_dim, config=config, device=device, **kwargs
    )


def __getattr__(name: str) -> Any:
    available = sorted(
        n for n in __all__ if n and not n.startswith("_")
    )
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}. "
        f"Available baseline symbols: {available}"
    )


__all__.extend(
    [
        "IN_REPO_BASELINES",
        "EXTERNAL_BASELINES",
        "ALL_BASELINES",
        "BASELINE_DESCRIPTIONS",
        "available_baselines",
        "build_baseline",
        "train_baseline",
    ]
)


if __name__ == "__main__":  # pragma: no cover - manual sanity check
    status = available_baselines()
    print("baselines package status:")
    for key in IN_REPO_BASELINES:
        print(f"  {key:8s} importable={status[key]}")
    print(f"  external: {list(EXTERNAL_BASELINES)}")
    print(f"exported symbols: {len(__all__)}")
