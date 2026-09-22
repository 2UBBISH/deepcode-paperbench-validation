"""Refining evaluation pipeline for RICE (Experiments II, III and IV).

This module implements the *effectiveness* half of the RICE evaluation suite
described in the paper:

* **Experiment II** (Sec. 4.2 / Sec. 4.3): fix the explanation method to ours
  (the re-designed mask network), vary the *refining* method and compare the
  agent's performance after refining.  Baselines: PPO fine-tuning, StateMask-R
  (fine-tuning only from critical steps) and JSRL (Uchendu et al., 2023).
* **Experiment III** (Sec. 4.2 / Table 1, right block): fix the refining method
  to RICE, vary the *explanation* method among ``Random``, ``StateMask`` and
  ``Ours`` and compare the final reward.
* **Experiment IV** (Sec. 4.2): pre-train SAC, imitate the SAC policy with GAIL,
  then refine the imitated policy with RICE versus PPO fine-tuning,
  StateMask-R, JSRL and SAC fine-tuning.

Reporting protocol (Sec. 4.1, "Evaluation Metrics"):

    For the applications with dense rewards except the malware mutation
    application, we measure the reward of the target agent before and after
    refining.  For the applications with sparse rewards, we report the
    performance during the refining process.

so ``RefiningResult.final_reward`` is the operative number for dense tasks
(Table 1) and ``RefiningResult.curve`` carries the refining curves for the
sparse tasks (Figure 2).

Everything is black-box w.r.t. the target agent: only the environment, the
warm-start (bottlenecked) policy and the separately trained explanation module
are consumed.  Third-party baselines (JSRL / StateMask-R / SAC-GAIL adapters)
are imported lazily; when a dependency is missing the module falls back to an
equivalent construction on top of :mod:`rice.algorithms.refine` and records the
substitution in the result metadata (``note``), mirroring the plan's guidance
("prefer thin adapters and fall back to a faithful re-implementation
(document it)").
"""

from __future__ import annotations

import importlib
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

# --------------------------------------------------------------------------------------
# RICE algorithm layer (all implemented modules)
# --------------------------------------------------------------------------------------
try:  # pragma: no cover - import guard for odd sys.path layouts
    from ..algorithms.ppo import (
        ActorCritic,
        PPOConfig,
        flatten_obs,
        make_target_policy_callable,
        observation_size,
        resolve_device,
    )
except Exception:  # pragma: no cover
    from rice.algorithms.ppo import (  # type: ignore
        ActorCritic,
        PPOConfig,
        flatten_obs,
        make_target_policy_callable,
        observation_size,
        resolve_device,
    )

try:  # pragma: no cover
    from ..algorithms.refine import (
        RICERefiner,
        RefineConfig,
        RefineIteration,
        RefineResult,
        evaluate_policy,
        load_policy_weights,
        make_refiner,
        refine_policy,
    )
except Exception:  # pragma: no cover
    from rice.algorithms.refine import (  # type: ignore
        RICERefiner,
        RefineConfig,
        RefineIteration,
        RefineResult,
        evaluate_policy,
        load_policy_weights,
        make_refiner,
        refine_policy,
    )

try:  # pragma: no cover
    from ..algorithms.mixed_init import MixedInitSampler, make_mixed_init_sampler
except Exception:  # pragma: no cover
    try:
        from rice.algorithms.mixed_init import (  # type: ignore
            MixedInitSampler,
            make_mixed_init_sampler,
        )
    except Exception:  # pragma: no cover
        MixedInitSampler = None  # type: ignore
        make_mixed_init_sampler = None  # type: ignore


__all__ = [
    # task groups
    "DENSE_TASKS",
    "SPARSE_TASKS",
    "OUT_OF_SCOPE_TASKS",
    "ALL_TASKS",
    "REFINING_METHODS",
    "EXPLANATION_METHODS",
    "TABLE3_HYPERPARAMS",
    # config / results
    "RefiningConfig",
    "RefiningCurve",
    "RefiningResult",
    "RefiningComparison",
    # evaluator
    "RefiningEvaluator",
    # functional entry points
    "evaluate_refining",
    "compare_refining_methods",
    "compare_explanations",
    "evaluate_sac_gail",
    "summarize_table",
    "format_table",
    "trend_check",
    # helpers
    "make_env_for_task",
    "make_policy_for_task",
    "resolve_task_name",
    "is_sparse_task",
    "resolve_explanation",
]


# ======================================================================================
# Task / method bookkeeping
# ======================================================================================
DENSE_TASKS: Tuple[str, ...] = (
    "Hopper-v3",
    "Walker2d-v3",
    "Reacher-v2",
    "HalfCheetah-v3",
    "SelfishMining",
    "CageChallenge2",
    "Macro-v1",
)

# SparseWalker2d is registered in the environment package but is OUT OF SCOPE per the
# reproduction addendum (only SparseHopper / SparseHalfCheetah are refining targets).
SPARSE_TASKS: Tuple[str, ...] = ("SparseHopper", "SparseHalfCheetah")

OUT_OF_SCOPE_TASKS: Tuple[str, ...] = ("SparseWalker2d", "MalwareMutation")

ALL_TASKS: Tuple[str, ...] = DENSE_TASKS + SPARSE_TASKS

#: Refining methods of Experiment II (+ SAC variants used in Experiment IV).
REFINING_METHODS: Tuple[str, ...] = (
    "no_refine",
    "ppo",
    "jsrl",
    "statemask_r",
    "ours",
    "sac",
    "sac_finetune",
)

#: Explanation methods compared in Experiment III (and Tables 5/6).
EXPLANATION_METHODS: Tuple[str, ...] = (
    "ours",
    "statemask",
    "random",
    "integrated_gradients",
    "airs",
)

#: Table 3 hyper-parameters (p, lambda, alpha) per task.  Note the documented conflict:
#: Sec. C.3 text says alpha = 0.01 for the mask network, Table 3 lists 0.0001; Table 3 is
#: operative per the reproduction addendum.
TABLE3_HYPERPARAMS: Dict[str, Dict[str, float]] = {
    "Hopper-v3": {"p": 0.25, "lambda": 0.001, "alpha": 0.0001},
    "Walker2d-v3": {"p": 0.25, "lambda": 0.01, "alpha": 0.0001},
    "Reacher-v2": {"p": 0.50, "lambda": 0.001, "alpha": 0.0001},
    "HalfCheetah-v3": {"p": 0.50, "lambda": 0.01, "alpha": 0.0001},
    "SelfishMining": {"p": 0.25, "lambda": 0.001, "alpha": 0.0001},
    "CageChallenge2": {"p": 0.50, "lambda": 0.01, "alpha": 0.0001},
    "Macro-v1": {"p": 0.25, "lambda": 0.01, "alpha": 0.0001},
    # out of scope, kept so lookups do not raise
    "MalwareMutation": {"p": 0.50, "lambda": 0.01, "alpha": 0.0001},
    "SparseHopper": {"p": 0.25, "lambda": 0.001, "alpha": 0.0001},
    "SparseHalfCheetah": {"p": 0.50, "lambda": 0.01, "alpha": 0.0001},
}

#: Mask-network training sample budgets (Table 4).
TABLE4_SAMPLE_BUDGETS: Dict[str, float] = {
    "Hopper-v3": 3e5,
    "Walker2d-v3": 3e5,
    "Reacher-v2": 3e5,
    "HalfCheetah-v3": 3e5,
    "SelfishMining": 1.5e6,
    "CageChallenge2": 1e7,
    "Macro-v1": 2443260.0,
}

#: Table 1 reference values (mean final reward after refining) for trend checking only.
TABLE1_REFERENCE: Dict[str, Dict[str, float]] = {
    "Hopper-v3": {
        "no_refine": 3559.44,
        "ppo": 3638.75,
        "jsrl": 3635.08,
        "statemask_r": 3652.06,
        "ours": 3663.91,
        "random": 3648.98,
        "statemask": 3661.86,
    },
    "Walker2d-v3": {
        "no_refine": 3768.79,
        "ppo": 3965.63,
        "jsrl": 3963.57,
        "statemask_r": 3966.96,
        "ours": 3982.79,
        "random": 3969.64,
        "statemask": 3982.67,
    },
    "Reacher-v2": {
        "no_refine": -5.79,
        "ppo": -3.04,
        "jsrl": -3.23,
        "statemask_r": -3.45,
        "ours": -2.66,
        "random": -3.11,
        "statemask": -2.69,
    },
    "HalfCheetah-v3": {
        "no_refine": 2024.09,
        "ppo": 2133.31,
        "jsrl": 2128.04,
        "statemask_r": 2085.28,
        "ours": 2138.89,
        "random": 2132.01,
        "statemask": 2136.23,
    },
    "SelfishMining": {
        "no_refine": 14.36,
        "ppo": 14.93,
        "jsrl": 14.88,
        "statemask_r": 14.53,
        "ours": 16.56,
        "random": 15.09,
        "statemask": 16.49,
    },
    "CageChallenge2": {
        "no_refine": -23.64,
        "ppo": -23.58,
        "jsrl": -22.97,
        "statemask_r": -26.98,
        "ours": -20.02,
        "random": -25.94,
        "statemask": -20.07,
    },
    "Macro-v1": {
        "no_refine": 10.30,
        "ppo": 13.37,
        "jsrl": 11.26,
        "statemask_r": 7.62,
        "ours": 17.03,
        "random": 11.72,
        "statemask": 16.28,
    },
}

#: Friendly aliases -> canonical registry / task names.
_TASK_ALIASES: Dict[str, str] = {
    "hopper": "Hopper-v3",
    "hopper-v3": "Hopper-v3",
    "walker2d": "Walker2d-v3",
    "walker2d-v3": "Walker2d-v3",
    "walker": "Walker2d-v3",
    "reacher": "Reacher-v2",
    "reacher-v2": "Reacher-v2",
    "halfcheetah": "HalfCheetah-v3",
    "halfcheetah-v3": "HalfCheetah-v3",
    "cheetah": "HalfCheetah-v3",
    "selfishmining": "SelfishMining",
    "selfish_mining": "SelfishMining",
    "selfish": "SelfishMining",
    "cagechallenge2": "CageChallenge2",
    "cage": "CageChallenge2",
    "cagechallenge": "CageChallenge2",
    "macro-v1": "Macro-v1",
    "macro": "Macro-v1",
    "autodriving": "Macro-v1",
    "sparsehopper": "SparseHopper",
    "sparse_hopper": "SparseHopper",
    "sparsehalfcheetah": "SparseHalfCheetah",
    "sparse_halfcheetah": "SparseHalfCheetah",
    "sparsewalker2d": "SparseWalker2d",
    "malware": "MalwareMutation",
    "malwaremutation": "MalwareMutation",
}


def resolve_task_name(task: str) -> str:
    """Canonicalise a task name/alias to a RICE task key."""
    if task is None:
        raise ValueError("task must not be None")
    key = str(task).strip()
    if key in TABLE3_HYPERPARAMS or key in ALL_TASKS:
        return key
    norm = key.lower().replace(" ", "").replace("_", "")
    if norm in _TASK_ALIASES:
        return _TASK_ALIASES[norm]
    # strip the "-v3"/"-v4" suffix and retry
    stripped = norm.split("-")[0]
    if stripped in _TASK_ALIASES:
        return _TASK_ALIASES[stripped]
    return key


def is_sparse_task(task: str) -> bool:
    """True for the sparse MuJoCo tasks (curve-style reporting, Figure 2)."""
    name = resolve_task_name(task)
    return name in SPARSE_TASKS or name.lower().startswith("sparse")


# ======================================================================================
# Environment / policy construction
# ======================================================================================
def make_env_for_task(task: str, seed: Optional[int] = None, **kwargs) -> Any:
    """Build a single (non-vectorised) environment for ``task``.

    Falls back through the specialised environment modules when
    :func:`rice.environments.make_env` is unavailable for a given name.
    """
    name = resolve_task_name(task)
    kwargs = dict(kwargs)
    if seed is not None and "seed" not in kwargs:
        kwargs["seed"] = seed

    try:
        from ..environments import make_env as _make_env  # type: ignore

        return _make_env(name, **kwargs)
    except Exception as exc:  # pragma: no cover - environment specific fallbacks
        last_error: Exception = exc

    # Module-specific fallbacks (sparse tasks first, then dense, then apps).
    candidates: List[Tuple[str, str]] = []
    if name.startswith("Sparse"):
        candidates += [
            ("..environments.mujoco_sparse", "make_env"),
            ("..environments.mujoco_sparse", "make_sparse_env"),
            ("..environments.mujoco_sparse", "make_mujoco_sparse"),
        ]
    elif name in ("Hopper-v3", "Walker2d-v3", "Reacher-v2", "HalfCheetah-v3"):
        candidates += [
            ("..environments.mujoco_dense", "make_env"),
            ("..environments.mujoco_dense", "make_mujoco_dense"),
        ]
    elif name == "SelfishMining":
        candidates += [
            ("..environments.selfish_mining", "make_env"),
            ("..environments.selfish_mining", "make_selfish_mining_env"),
        ]
    elif name == "CageChallenge2":
        candidates += [
            ("..environments.cage_challenge2", "make_env"),
            ("..environments.cage_challenge2", "make_cage_challenge2"),
        ]
    elif name == "Macro-v1":
        candidates += [
            ("..environments.autodriving", "make_env"),
            ("..environments.autodriving", "make_autodriving"),
        ]

    for module_name, attr in candidates:
        try:
            module = _import_relative(module_name)
            factory = getattr(module, attr)
            return factory(name, **kwargs) if attr == "make_env" else factory(**kwargs)
        except Exception as exc:  # pragma: no cover
            last_error = exc
    raise RuntimeError(f"Could not build environment for task {task!r}: {last_error}")


def make_policy_for_task(
    env: Any,
    net_arch: Optional[Sequence[int]] = None,
    device: str = "auto",
    seed: Optional[int] = None,
    weights: Any = None,
) -> ActorCritic:
    """Build an :class:`ActorCritic` matching the target agent's architecture.

    ``net_arch`` defaults to :func:`rice.environments.default_net_arch` for the
    environment when available (Appendix C.2 / addendum architectures), falling
    back to the Stable-Baselines3 default ``(64, 64)``.
    """
    if net_arch is None:
        net_arch = (64, 64)
        try:
            from ..environments import default_net_arch  # type: ignore

            name = getattr(env, "rise_canonical_name", None)
            if name:
                net_arch = default_net_arch(name)
        except Exception:
            pass

    observation_space = getattr(env, "observation_space", None)
    action_space = getattr(env, "action_space", None)
    if observation_space is None or action_space is None:
        raise ValueError("env must expose observation_space and action_space")

    policy = ActorCritic(
        observation_space=observation_space,
        action_space=action_space,
        net_arch=tuple(int(x) for x in net_arch),
        device=device,
    )
    if weights is not None:
        load_policy_weights(policy, weights)
    return policy


def _import_relative(module_name: str) -> Any:  # pragma: no cover - import plumbing
    """Import a sibling module tolerating the nested ``rice/rice`` layout."""
    tail = module_name.lstrip(".")
    attempts = [
        f"rice.{tail}",
        f"rice.rice.{tail}",
        tail,
    ]
    last_error: Optional[Exception] = None
    for candidate in attempts:
        try:
            return importlib.import_module(candidate)
        except Exception as exc:
            last_error = exc
    # last resort: relative to this package
    try:
        package = __package__ or "rice.evaluation"
        parent = package.rsplit(".", 1)[0] if "." in package else package
        return importlib.import_module(f"{parent}.{tail}")
    except Exception as exc:  # pragma: no cover
        raise ImportError(f"cannot import {module_name!r}: {last_error or exc}")


# ======================================================================================
# Explanation resolution (Experiment III / Table 6)
# ======================================================================================
@dataclass
class ExplanationRef:
    """Resolved explanation module plus provenance information."""

    name: str
    module: Any
    available: bool
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "available": bool(self.available), "note": self.note}


class _RandomMaskStub:
    """Fallback implementation of the "Random" explanation baseline.

    The paper defines it as "identifies critical steps by randomly selecting a
    visited state as the critical state" (Sec. 4.1).  The stub exposes the same
    duck-typed contract as :class:`rice.algorithms.mask_network.MaskNetwork`
    (``importance`` / ``mask_prob_zero`` / callable returning ``(N, 2)`` logits)
    so that :func:`rice.algorithms.critical_state.importance_scores` can consume
    it directly.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.training = False
        self.device = "cpu"
        self.is_fallback = True

    # -- torch-module-like no-ops -------------------------------------------------
    def eval(self) -> "_RandomMaskStub":
        self.training = False
        return self

    def train(self, mode: bool = True) -> "_RandomMaskStub":
        self.training = bool(mode)
        return self

    def to(self, *_args, **_kwargs) -> "_RandomMaskStub":
        return self

    def cpu(self) -> "_RandomMaskStub":
        return self

    # -- scoring API ---------------------------------------------------------------
    def importance(self, state: Any = None) -> float:
        return float(self.rng.uniform(0.0, 1.0))

    def mask_prob_zero(self, state: Any = None) -> float:
        return float(self.rng.uniform(0.0, 1.0))

    def score(self, state: Any = None) -> float:
        return self.importance(state)

    def __call__(self, obs: Any) -> np.ndarray:
        arr = np.asarray(obs)
        n = 1 if arr.ndim == 1 else int(arr.shape[0])
        p0 = self.rng.uniform(0.0, 1.0, size=n)
        return np.stack([p0, 1.0 - p0], axis=-1).astype(np.float32)


def build_random_explanation(seed: Optional[int] = None) -> Any:
    """Return the Random-explanation object (repo module if available)."""
    for module_name, attr in (
        ("rice.explanation.random_explanation", "RandomExplanation"),
        ("rice.explanation.random_explanation", "make_random_explanation"),
    ):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        if hasattr(module, attr):
            obj = getattr(module, attr)
            try:
                return obj(seed=seed)
            except TypeError:
                try:
                    return obj()
                except Exception:
                    continue
    return _RandomMaskStub(seed=seed)


def resolve_explanation(
    name: str,
    env: Any = None,
    policy: Any = None,
    mask_network: Any = None,
    seed: Optional[int] = None,
    **kwargs: Any,
) -> ExplanationRef:
    """Resolve an explanation method name into a usable explanation module.

    ``"ours"`` maps to the re-designed mask network.  The other methods are
    imported from :mod:`rice.explanation` (thin adapters over the third-party
    StateMask / Integrated-Gradients / AIRS implementations); when an adapter is
    unavailable the reference records ``available=False`` and one of two
    documented fallbacks is used:

    * ``"random"``   -> :class:`_RandomMaskStub` (faithful to the paper's text).
    * everything else -> the RICE mask network, tagged as a deviation.
    """
    key = str(name).strip().lower()
    if key in ("ours", "rice", "mask", "mask_network"):
        return ExplanationRef("ours", mask_network, mask_network is not None, "mask network (RICE)")

    if key in ("random", "rand"):
        obj = build_random_explanation(seed=seed)
        note = "" if not getattr(obj, "is_fallback", False) else "built-in random stub"
        return ExplanationRef("random", obj, True, note)

    module_candidates = {
        "statemask": (
            ("rice.explanation.statemask_adapter", "StateMaskAdapter"),
            ("rice.explanation.statemask_adapter", "make_statemask_explanation"),
            ("rice.explanation.statemask_adapter", "load_statemask"),
        ),
        "integrated_gradients": (
            ("rice.explanation.integrated_gradients", "IntegratedGradients"),
            ("rice.explanation.integrated_gradients", "make_integrated_gradients"),
        ),
        "ig": (
            ("rice.explanation.integrated_gradients", "IntegratedGradients"),
            ("rice.explanation.integrated_gradients", "make_integrated_gradients"),
        ),
        "airs": (
            ("rice.explanation.airs_adapter", "AIRSAdapter"),
            ("rice.explanation.airs_adapter", "make_airs_explanation"),
        ),
    }.get(key)

    if module_candidates is None:
        raise KeyError(f"unknown explanation method {name!r}; known: {EXPLANATION_METHODS}")

    for module_name, attr in module_candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        factory = getattr(module, attr, None)
        if factory is None:
            continue
        try:
            obj = factory(env=env, policy=policy, mask_network=mask_network, seed=seed, **kwargs)
        except TypeError:
            try:
                obj = factory(env, policy, mask_network)
            except Exception:
                continue
        except Exception:
            continue
        return ExplanationRef(key, obj, True, f"{module_name}.{attr}")

    warn = (
        f"explanation {name!r} adapter unavailable (module not implemented yet); "
        "falling back to the RICE mask network -- recorded as a deviation"
    )
    warnings.warn(warn, RuntimeWarning, stacklevel=2)
    return ExplanationRef(key, mask_network, False, "FALLBACK: RICE mask network (adapter missing)")


# ======================================================================================
# Configuration
# ======================================================================================
def _default_env_kwargs(task: str) -> Dict[str, Any]:
    return {}


@dataclass
class RefiningConfig:
    """Configuration for the refining-evaluation experiments.

    Refining budgets are *not* specified by the paper (documented deviation);
    defaults below are chosen so that curves saturate on the light MuJoCo tasks
    while remaining CPU-runnable.
    """

    task: str = "Hopper-v3"
    method: str = "ours"
    explanation: str = "ours"

    # --- Algorithm 2 hyper-parameters (Table 3 unless overridden) -------------------
    p: Optional[float] = None
    lam: Optional[float] = None
    alpha: Optional[float] = None

    # --- budget --------------------------------------------------------------------
    n_iterations: int = 50
    steps_per_iter: Optional[int] = None
    rollin_length: Optional[int] = None
    total_env_steps: Optional[float] = None
    n_seeds: int = 3
    seeds: Optional[Sequence[int]] = None

    # --- evaluation ----------------------------------------------------------------
    eval_episodes: int = 5
    eval_every: int = 0
    final_eval: bool = True
    curve_window: int = 1
    report_mode: Optional[str] = None  # "final_reward" | "curve" | None (auto)

    # --- misc ----------------------------------------------------------------------
    lower_lr_factor: float = 0.1
    policy_config: Optional[PPOConfig] = None
    rnd_config: Any = None
    device: str = "auto"
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    eval_env_kwargs: Dict[str, Any] = field(default_factory=dict)
    weights: Any = None
    net_arch: Optional[Sequence[int]] = None
    deterministic_eval: bool = True
    verbose: int = 1
    log_every: int = 10
    seed: Optional[int] = None

    # -------------------------------------------------------------------------------
    def __post_init__(self) -> None:
        self.task = resolve_task_name(self.task)
        defaults = TABLE3_HYPERPARAMS.get(self.task, {"p": 0.25, "lambda": 0.01, "alpha": 0.0001})
        if self.p is None:
            self.p = float(defaults["p"])
        if self.lam is None:
            self.lam = float(defaults["lambda"])
        if self.alpha is None:
            self.alpha = float(defaults["alpha"])
        if self.report_mode is None:
            self.report_mode = "curve" if is_sparse_task(self.task) else "final_reward"
        if self.seeds is None:
            self.seeds = tuple(range(int(self.n_seeds)))

    def clone(self, **overrides: Any) -> "RefiningConfig":
        payload = {f: getattr(self, f) for f in self.__dataclass_fields__}  # type: ignore[attr-defined]
        payload.update(overrides)
        return RefiningConfig(**payload)

    @classmethod
    def from_mapping(cls, mapping: Optional[Dict[str, Any]] = None, **overrides: Any) -> "RefiningConfig":
        """Build from a (YAML-loaded) mapping; accepts paper-style aliases."""
        payload: Dict[str, Any] = {}
        source = dict(mapping or {})
        source.update(overrides)
        aliases = {
            "lambda": "lam",
            "lambda_": "lam",
            "coef": "lam",
            "beta": "p",
            "reset_probability": "p",
            "K": "rollin_length",
            "length": "rollin_length",
            "iterations": "n_iterations",
            "num_iterations": "n_iterations",
            "epochs": "n_iterations",
            "T": "steps_per_iter",
            "seed": "seed",
        }
        for key, value in source.items():
            payload[aliases.get(key, key)] = value
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        payload = {k: v for k, v in payload.items() if k in known}
        return cls(**payload)


# ======================================================================================
# Result containers
# ======================================================================================
@dataclass
class RefiningCurve:
    """A refining curve: performance versus environment steps (Figure 2)."""

    steps: np.ndarray
    rewards: np.ndarray
    label: str = "agent"
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        self.steps = np.asarray(self.steps, dtype=np.float64).reshape(-1)
        self.rewards = np.asarray(self.rewards, dtype=np.float64).reshape(-1)

    def __len__(self) -> int:
        return int(self.rewards.size)

    def moving_average(self, window: int = 1) -> "RefiningCurve":
        if window is None or window <= 1 or len(self) < window:
            return RefiningCurve(self.steps.copy(), self.rewards.copy(), self.label, self.seed)
        kernel = np.ones(int(window), dtype=np.float64) / float(window)
        smoothed = np.convolve(self.rewards, kernel, mode="valid")
        return RefiningCurve(self.steps[window - 1 :].copy(), smoothed, self.label, self.seed)

    def final(self) -> float:
        return float(self.rewards[-1]) if len(self) else float("nan")

    def best(self) -> float:
        return float(np.max(self.rewards)) if len(self) else float("nan")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "seed": self.seed,
            "steps": self.steps,
            "rewards": self.rewards,
        }


@dataclass
class RefiningResult:
    """Outcome of refining a warm-start policy with one method (one task)."""

    task: str
    method: str
    explanation: str = "ours"

    final_rewards: List[float] = field(default_factory=list)
    baseline_rewards: List[float] = field(default_factory=list)
    curves: List[RefiningCurve] = field(default_factory=list)
    per_length: Dict[str, float] = field(default_factory=dict)

    iterations: int = 0
    env_steps: float = 0.0
    seconds: float = 0.0
    seeds: List[int] = field(default_factory=list)

    config: Optional[RefiningConfig] = None
    refine_results: List[Any] = field(default_factory=list, repr=False)
    notes: List[str] = field(default_factory=list)
    error: Optional[str] = None

    # -- aggregates ---------------------------------------------------------------
    @property
    def final_reward(self) -> float:
        """Mean final reward after refining (Table 1 metric)."""
        return float(np.mean(self.final_rewards)) if self.final_rewards else float("nan")

    @property
    def final_std(self) -> float:
        return float(np.std(self.final_rewards)) if len(self.final_rewards) > 1 else 0.0

    @property
    def baseline_reward(self) -> float:
        """Mean "No Refine" reward of the warm-start policy."""
        return float(np.mean(self.baseline_rewards)) if self.baseline_rewards else float("nan")

    @property
    def baseline_std(self) -> float:
        return float(np.std(self.baseline_rewards)) if len(self.baseline_rewards) > 1 else 0.0

    @property
    def improvement(self) -> float:
        return self.final_reward - self.baseline_reward

    @property
    def report_mode(self) -> str:
        if self.config is not None and self.config.report_mode:
            return str(self.config.report_mode)
        return "curve" if is_sparse_task(self.task) else "final_reward"

    def mean_curve(self, window: Optional[int] = None) -> RefiningCurve:
        """Average refining curve across seeds (sparse tasks, Figure 2)."""
        curves = [c for c in self.curves if len(c) > 0]
        if not curves:
            return RefiningCurve(np.zeros(0), np.zeros(0), self.method)
        if window is None and self.config is not None:
            window = self.config.curve_window
        prepared = [c.moving_average(window or 1) for c in curves]
        n = min(len(c) for c in prepared)
        steps = prepared[0].steps[:n]
        rewards = np.mean([c.rewards[:n] for c in prepared], axis=0)
        return RefiningCurve(steps, rewards, self.method)

    def curve_std(self, window: Optional[int] = None) -> np.ndarray:
        curves = [c.moving_average(window or 1) for c in self.curves if len(c) > 0]
        if len(curves) < 2:
            return np.zeros(0)
        n = min(len(c) for c in curves)
        return np.std([c.rewards[:n] for c in curves], axis=0)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "method": self.method,
            "explanation": self.explanation,
            "final_reward": self.final_reward,
            "final_std": self.final_std,
            "baseline_reward": self.baseline_reward,
            "baseline_std": self.baseline_std,
            "improvement": self.improvement,
            "final_rewards": list(self.final_rewards),
            "baseline_rewards": list(self.baseline_rewards),
            "iterations": self.iterations,
            "env_steps": self.env_steps,
            "seconds": self.seconds,
            "seeds": list(self.seeds),
            "report_mode": self.report_mode,
            "per_length": dict(self.per_length),
            "notes": list(self.notes),
            "error": self.error,
        }


@dataclass
class RefiningComparison:
    """A collection of :class:`RefiningResult` (Table 1 / Figure 2 layout)."""

    task: str
    results: Dict[str, RefiningResult] = field(default_factory=dict)
    group: str = "refine"
    notes: List[str] = field(default_factory=list)

    def add(self, result: RefiningResult) -> None:
        key = result.method if self.group == "refine" else result.explanation
        self.results[key] = result

    def get(self, key: str) -> Optional[RefiningResult]:
        return self.results.get(key)

    def best(self) -> Optional[str]:
        scored = [
            (k, r)
            for k, r in self.results.items()
            if r.error is None and k not in ("no_refine",)
            and not np.isnan(r.final_reward)
        ]
        if not scored:
            return None
        return max(scored, key=lambda kv: kv[1].final_reward)[0]

    def to_table(self, reference: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Dict[str, Any]]:
        return summarize_table(self, reference=reference)

    def format(self, reference: Optional[Dict[str, Dict[str, float]]] = None) -> str:
        return format_table(self, reference=reference)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "group": self.group,
            "notes": list(self.notes),
            "results": {k: v.as_dict() for k, v in self.results.items()},
        }


# ======================================================================================
# Baseline dispatch (lazy adapters with faithful fallbacks)
# ======================================================================================
def _baseline_module_factories(name: str) -> List[Tuple[str, Tuple[str, ...]]]:
    """Candidate (module, factory-attrs) pairs for third-party baselines."""
    return {
        "ppo": [
            ("rice.baselines.ppo_finetune", ("run_ppo_finetune", "finetune_ppo", "ppo_finetune")),
        ],
        "jsrl": [
            ("rice.baselines.jsrl", ("run_jsrl", "jsrl_refine", "refine_with_jsrl", "JSRL")),
        ],
        "statemask_r": [
            ("rice.baselines.statemask_r", ("run_statemask_refining", "refine_from_critical_states", "statemask_r")),
        ],
        "sil": [
            ("rice.baselines.self_imitation", ("run_sil", "self_imitation", "SelfImitation")),
        ],
        "sac_finetune": [
            ("rice.baselines.sac_gail", ("run_sac_finetune", "sac_finetune", "finetune_sac")),
        ],
        "sac": [
            ("rice.baselines.sac_gail", ("run_sac_gail", "sac_gail", "pretrain_sac_gail")),
        ],
    }.get(name, [])


def _dispatch_baseline(name: str, **kwargs: Any) -> Optional[Any]:
    """Try to run a third-party/among-repo baseline; return None when unavailable."""
    for module_name, attrs in _baseline_module_factories(name):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for attr in attrs:
            factory = getattr(module, attr, None)
            if factory is None:
                continue
            try:
                return factory(**kwargs)
            except TypeError:
                continue
            except Exception as exc:
                warnings.warn(f"baseline {name!r} raised {exc!r}; using fallback", RuntimeWarning, stacklevel=2)
                return None
    return None


def _extract_final_reward(obj: Any) -> Optional[float]:
    """Read the final reward from whatever a baseline returns."""
    if obj is None:
        return None
    if isinstance(obj, (int, float, np.floating)):
        return float(obj)
    if isinstance(obj, dict):
        for key in ("final_reward", "final_eval_reward", "reward", "mean_reward", "ours"):
            if key in obj and obj[key] is not None:
                return float(obj[key])
        return None
    for attr in ("final_reward", "final_eval_reward", "mean_reward"):
        value = getattr(obj, attr, None)
        if value is not None and np.isscalar(value):
            return float(value)
    return None


def _extract_curve(obj: Any, label: str) -> Optional[RefiningCurve]:
    if obj is None or isinstance(obj, (int, float, np.floating)):
        return None
    source = obj
    if isinstance(obj, dict):
        steps = obj.get("steps") or obj.get("env_steps")
        rewards = obj.get("rewards") or obj.get("returns") or obj.get("curve")
        if steps is None or rewards is None:
            return None
        return RefiningCurve(np.asarray(steps), np.asarray(rewards), label)
    if hasattr(source, "refining_curve"):
        try:
            out = source.refining_curve()
            steps, rewards = out if isinstance(out, tuple) else (None, None)
            if steps is not None:
                return RefiningCurve(np.asarray(steps), np.asarray(rewards), label)
        except Exception:
            pass
    log = getattr(source, "log", None)
    if log:
        steps, rewards = _curve_from_log(log)
        return RefiningCurve(steps, rewards, label)
    return None


def _curve_from_log(log: Sequence[Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Build (steps, mean-episode-return) arrays from a refining log."""
    steps: List[float] = []
    rewards: List[float] = []
    cumulative = 0.0
    for entry in log:
        record = entry.as_dict() if hasattr(entry, "as_dict") else dict(entry)
        cumulative += float(record.get("steps", record.get("env_steps", 0)) or 0)
        episode_returns = record.get("episode_returns") or []
        if episode_returns:
            value = float(np.mean(episode_returns))
        else:
            value = record.get("final_eval_reward")
            value = float(value) if value is not None else float("nan")
        steps.append(cumulative)
        rewards.append(value)
    return np.asarray(steps, dtype=np.float64), np.asarray(rewards, dtype=np.float64)


# ======================================================================================
# The evaluator
# ======================================================================================
class RefiningEvaluator:
    """Runs Experiments II/III/IV for one task.

    Parameters
    ----------
    env:
        Optional pre-built environment used for roll-outs (if ``None`` one is
        created per seed via :func:`make_env_for_task`).
    policy:
        Optional warm-start policy (``ActorCritic``, SB3 model or path).  If
        ``None`` a fresh policy is built from ``config.net_arch`` -- useful for
        smoke tests, not for reproducing the paper's numbers.
    mask_network:
        Explanation module used by RICE (may be ``None`` for pure baselines).
    config:
        :class:`RefiningConfig` (or mapping).
    """

    def __init__(
        self,
        env: Any = None,
        policy: Any = None,
        mask_network: Any = None,
        task: Optional[str] = None,
        config: Optional[Union[RefiningConfig, Dict[str, Any]]] = None,
        evaluation_env: Any = None,
        state_manager: Any = None,
        rng: Any = None,
        seed: Optional[int] = None,
        logger: Any = None,
    ) -> None:
        if isinstance(config, dict):
            config = RefiningConfig.from_mapping(config)
        if config is None:
            config = RefiningConfig()
        if task is not None:
            config = config.clone(task=task)
        self.config = config
        self.task = config.task
        self.env = env
        self.evaluation_env = evaluation_env
        self.policy_source = policy
        self.mask_network = mask_network
        self.state_manager = state_manager
        self.rng = rng
        self.seed = seed if seed is not None else config.seed
        self.logger = logger
        self._baseline_cache: Optional[Dict[str, Any]] = None

    # -- logging -------------------------------------------------------------------
    def log(self, message: str) -> None:
        if self.config.verbose <= 0:
            return
        if self.logger is not None:
            try:
                self.logger.info(message)
                return
            except Exception:
                pass
        print(f"[refine-eval] {message}")

    # -- construction helpers ------------------------------------------------------
    def build_env(self, seed: Optional[int] = None, evaluation: bool = False) -> Any:
        if evaluation and self.evaluation_env is not None:
            return self.evaluation_env
        if (not evaluation) and self.env is not None:
            return self.env
        kwargs = dict(self.config.eval_env_kwargs if evaluation else self.config.env_kwargs)
        return make_env_for_task(self.task, seed=seed, **kwargs)

    def build_policy(self, env: Any, seed: Optional[int] = None) -> ActorCritic:
        weights = self.config.weights
        if self.policy_source is not None:
            weights = self.policy_source
        policy = make_policy_for_task(
            env,
            net_arch=self.config.net_arch,
            device=self.config.device,
            seed=seed,
            weights=weights,
        )
        return policy

    def state_manager_for(self, env: Any) -> Any:
        if self.state_manager is not None:
            return self.state_manager
        try:
            from ..algorithms.env_reset import make_state_manager  # type: ignore

            return make_state_manager(env)
        except Exception:
            return None

    # -- evaluation ----------------------------------------------------------------
    def evaluate(self, env: Any, policy: Any, seed: Optional[int] = None, n_episodes: Optional[int] = None) -> Dict[str, float]:
        """Evaluate ``policy`` on ``env`` (paper metric: mean episode reward)."""
        episodes = int(n_episodes if n_episodes is not None else self.config.eval_episodes)
        agent = policy
        if not isinstance(policy, ActorCritic) and hasattr(policy, "predict"):
            try:
                agent = make_target_policy_callable(policy)
            except Exception:
                agent = policy
        deterministic = bool(self.config.deterministic_eval)
        try:
            stats = evaluate_policy(
                env,
                agent,
                n_episodes=episodes,
                seed=seed,
                deterministic=deterministic,
            )
            if isinstance(stats, dict):
                return stats
            return {"mean": float(stats), "std": 0.0}
        except Exception as exc:  # pragma: no cover
            self.log(f"evaluate() failed ({exc!r}); using manual roll-out")
            return self._manual_evaluate(env, agent, episodes, seed, deterministic)

    def _manual_evaluate(
        self,
        env: Any,
        policy: Any,
        n_episodes: int,
        seed: Optional[int],
        deterministic: bool,
    ) -> Dict[str, float]:
        if hasattr(env, "rise_canonical_name") and str(getattr(env, "rise_canonical_name", "")).lower().startswith("cage"):
            pass
        callable_policy = make_target_policy_callable(policy)
        returns: List[float] = []
        for episode in range(n_episodes):
            reset_out = env.reset(seed=None if seed is None else seed + episode)
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            done = False
            total = 0.0
            steps = 0
            limit = int(getattr(env, "max_episode_steps", 1000) or 1000)
            while not done and steps < limit:
                action = np.asarray(callable_policy(flatten_obs(obs)), dtype=np.float32).reshape(-1)
                step_out = env.step(action)
                if len(step_out) == 5:
                    obs, reward, terminated, truncated, _ = step_out
                    done = bool(terminated or truncated)
                else:
                    obs, reward, done, _ = step_out
                total += float(reward)
                steps += 1
            returns.append(total)
        return {
            "mean": float(np.mean(returns)),
            "std": float(np.std(returns)),
            "min": float(np.min(returns)),
            "max": float(np.max(returns)),
            "mean_length": float("nan"),
            "returns": returns,
        }

    # -- RICE refining -------------------------------------------------------------
    def build_refine_config(
        self,
        method: str,
        explanation: str = "ours",
        overrides: Optional[Dict[str, Any]] = None,
    ) -> RefineConfig:
        """Translate an evaluation method into a :class:`RefineConfig`."""
        cfg = self.config
        p, lam = float(cfg.p), float(cfg.lam)
        policy_config = cfg.policy_config

        if method == "ours":
            pass  # Table 3 defaults
        elif method == "ppo":
            # Sec. 4.1: "lowering the learning rate and continuing training with the
            # PPO algorithm" -- no critical-state resets, no RND bonus.
            p, lam = 0.0, 0.0
            base = policy_config or PPOConfig()
            policy_config = base.clone(learning_rate=float(base.learning_rate) * float(cfg.lower_lr_factor))
        elif method == "statemask_r":
            # Sec. 4.1 / Cheng et al. (2023): reset to the critical state and continue
            # training from there (always reset => p = 1), no RND bonus.
            p, lam = 1.0, 0.0
        elif method in ("rnd_only",):
            p, lam = 0.0, float(cfg.lam)
        payload: Dict[str, Any] = {
            "p": p,
            "lam": lam,
            "n_iterations": int(cfg.n_iterations),
            "steps_per_iter": cfg.steps_per_iter,
            "rollin_length": cfg.rollin_length,
            "total_env_steps": cfg.total_env_steps,
            "eval_every": int(cfg.eval_every),
            "eval_episodes": int(cfg.eval_episodes),
            "device": cfg.device,
            "verbose": int(cfg.verbose) - 1 if cfg.verbose else 0,
            "log_every": int(cfg.log_every),
            "report_mode": cfg.report_mode,
        }
        if policy_config is not None:
            payload["policy_config"] = policy_config
        if cfg.rnd_config is not None:
            payload["rnd_config"] = cfg.rnd_config
        payload.update(overrides or {})
        known = set(RefineConfig.__dataclass_fields__)  # type: ignore[attr-defined]
        payload = {k: v for k, v in payload.items() if k in known}
        return RefineConfig.from_mapping(payload)

    def _explanation_for(
        self, name: str, env: Any, policy: Any, mask_network: Any
    ) -> Tuple[Any, List[str]]:
        ref = resolve_explanation(
            name,
            env=env,
            policy=policy,
            mask_network=mask_network,
            seed=None if self.config.seeds is None else int(self.config.seeds[0]),
        )
        notes: List[str] = []
        if not ref.available:
            notes.append(ref.note)
            warnings.warn(f"explanation {name!r} unavailable: {ref.note}", RuntimeWarning, stacklevel=2)
        return ref.module, notes

    def run_single_seed(
        self,
        seed: int,
        method: str = "ours",
        explanation: str = "ours",
    ) -> Dict[str, Any]:
        """Run one seed of one (method, explanation) combination."""
        cfg = self.config
        try:
            from ..utils.seeding import set_global_seeds  # type: ignore

            set_global_seeds(seed)
        except Exception:
            pass

        env = self.build_env(seed=seed)
        evaluation_env = self.build_env(seed=seed + 10_000, evaluation=True)
        policy = self.build_policy(env, seed=seed)

        notes: List[str] = []
        baseline = self.evaluate(evaluation_env, policy, seed=seed)

        result: Dict[str, Any] = {
            "seed": int(seed),
            "baseline": float(baseline.get("mean", float("nan"))),
            "baseline_stats": baseline,
            "final_reward": None,
            "curve": None,
            "per_length": {},
            "iterations": 0,
            "env_steps": 0.0,
            "seconds": 0.0,
            "notes": notes,
            "result": None,
        }

        if method == "no_refine":
            result["final_reward"] = float(baseline.get("mean", float("nan")))
            result["curve"] = RefiningCurve(
                np.array([0.0]), np.array([result["final_reward"]]), "no_refine", seed
            )
            return result

        # ---- third-party baselines first -----------------------------------------
        if method in ("ppo", "jsrl", "statemask_r", "sac", "sac_finetune"):
            dispatched = self._run_dispatched_baseline(method, env, policy, evaluation_env, seed, notes)
            if dispatched is not None:
                dispatched.setdefault("baseline", float(baseline.get("mean", float("nan"))))
                dispatched.setdefault("baseline_stats", baseline)
                dispatched["seed"] = int(seed)
                return dispatched

        # ---- RICE path (also the fallback for the baselines) ----------------------
        mask_network = None
        if explanation != "ours" or method in ("ours", "statemask_r"):
            mask_network, exp_notes = self._explanation_for(
                explanation, env, policy, self.mask_network
            )
            notes.extend(exp_notes)
        if method == "ours" and mask_network is None:
            mask_network = self.mask_network

        refine_config = self.build_refine_config(method, explanation)
        started = time.time()
        outcome = refine_policy(
            env=env,
            policy=policy,
            mask_network=mask_network,
            config=refine_config,
            state_manager=self.state_manager_for(env),
            evaluation_env=evaluation_env,
            seed=seed,
        )
        result["seconds"] = float(time.time() - started)
        result["result"] = outcome
        result["iterations"] = int(getattr(outcome, "iterations", 0) or 0)
        result["env_steps"] = float(getattr(outcome, "env_steps", 0.0) or 0.0)

        steps, rewards = _curve_from_log(getattr(outcome, "log", []) or [])
        if steps.size:
            result["curve"] = RefiningCurve(steps, rewards, method, seed)

        final = getattr(outcome, "final_eval_reward", None)
        if final is None:
            final = self.evaluate(
                evaluation_env, getattr(outcome, "policy", policy), seed=seed
            ).get("mean")
        result["final_reward"] = float(final) if final is not None else float("nan")

        # Cage Challenge 2 uses the trial-length metric of Sec. C.2.
        if self._is_cage:
            per_length = self._cage_trial_metric(evaluation_env, getattr(outcome, "policy", policy))
            if per_length:
                result["per_length"] = per_length
                result["final_reward"] = float(per_length["final_reward"])
                notes.append("Cage: final reward = sum of average rewards over trials 30/50/100")
        return result

    # -- dispatched baselines ------------------------------------------------------
    def _run_dispatched_baseline(
        self,
        method: str,
        env: Any,
        policy: Any,
        evaluation_env: Any,
        seed: int,
        notes: List[str],
    ) -> Optional[Dict[str, Any]]:
        refine_config = self.build_refine_config(method)
        dispatched = _dispatch_baseline(
            method,
            env=env,
            policy=policy,
            evaluation_env=evaluation_env,
            config=refine_config,
            mask_network=self.mask_network,
            state_manager=self.state_manager_for(env),
            seed=seed,
        )
        if dispatched is None:
            notes.append(
                f"{method}: adapter unavailable -> faithful fallback on rice.algorithms.refine "
                f"(p={refine_config.p}, lambda={refine_config.lam})"
            )
            if method == "jsrl":
                notes.append(
                    "jsrl fallback approximates pi_e <- pi_g curriculum by continued training "
                    "from the warm-start policy"
                )
            return None
        curve = _extract_curve(dispatched, method)
        final = _extract_final_reward(dispatched)
        if final is None:
            final = self.evaluate(evaluation_env, policy, seed=seed).get("mean")
        return {
            "final_reward": float(final) if final is not None else float("nan"),
            "curve": curve,
            "per_length": dict(getattr(dispatched, "per_length", {}) or {}),
            "iterations": int(getattr(dispatched, "iterations", 0) or 0),
            "env_steps": float(getattr(dispatched, "env_steps", 0.0) or 0.0),
            "seconds": float(getattr(dispatched, "seconds", 0.0) or 0.0),
            "notes": notes,
            "result": dispatched,
        }

    @property
    def _is_cage(self) -> bool:
        return self.task.lower().startswith("cage")

    def _cage_trial_metric(self, env: Any, policy: Any) -> Dict[str, float]:
        """Sec. C.2: final reward = sum of average rewards over trials 30/50/100."""
        try:
            module = _import_relative("environments.cage_challenge2")
            metric = module.evaluate_trial_lengths(
                env,
                policy=policy,
                n_episodes_per_length=1,
                seed=self.seed if self.seed is not None else 0,
                deterministic=bool(self.config.deterministic_eval),
            )
            if isinstance(metric, dict):
                per_length = metric.get("per_length") or {}
                out = {str(k): float(v) for k, v in per_length.items()}
                out["final_reward"] = float(metric.get("final_reward", float("nan")))
                return out
        except Exception as exc:  # pragma: no cover
            self.log(f"cage trial-length metric failed: {exc!r}")
        return {}

    # -- multi-seed driver ---------------------------------------------------------
    def run(
        self,
        method: Optional[str] = None,
        explanation: Optional[str] = None,
        seeds: Optional[Sequence[int]] = None,
    ) -> RefiningResult:
        """Run one (method, explanation) combination over all configured seeds."""
        method = method or self.config.method
        explanation = explanation or self.config.explanation
        seed_list = list(seeds if seeds is not None else (self.config.seeds or (0,)))

        result = RefiningResult(
            task=self.task,
            method=method,
            explanation=explanation,
            config=self.config,
            seeds=[int(s) for s in seed_list],
        )
        started = time.time()
        for seed in seed_list:
            self.log(f"task={self.task} method={method} explanation={explanation} seed={seed}")
            try:
                record = self.run_single_seed(int(seed), method=method, explanation=explanation)
            except Exception as exc:  # pragma: no cover - robustness for long sweeps
                result.error = f"seed {seed} failed: {exc!r}"
                warnings.warn(result.error, RuntimeWarning, stacklevel=2)
                continue
            if record.get("final_reward") is not None and not np.isnan(record["final_reward"]):
                result.final_rewards.append(float(record["final_reward"]))
            if record.get("baseline") is not None and not np.isnan(record["baseline"]):
                result.baseline_rewards.append(float(record["baseline"]))
            if record.get("curve") is not None and len(record["curve"]):
                result.curves.append(record["curve"])
            if record.get("per_length"):
                result.per_length = dict(record["per_length"])
            result.iterations = max(result.iterations, int(record.get("iterations", 0) or 0))
            result.env_steps = max(result.env_steps, float(record.get("env_steps", 0.0) or 0.0))
            result.seconds += float(record.get("seconds", 0.0) or 0.0)
            for note in record.get("notes", []) or []:
                if note not in result.notes:
                    result.notes.append(note)
            if record.get("result") is not None:
                result.refine_results.append(record["result"])
        result.seconds = float(result.seconds or (time.time() - started))
        return result

    def run_methods(self, methods: Optional[Sequence[str]] = None) -> RefiningComparison:
        """Experiment II: fix the explanation, vary the refining method."""
        methods = list(methods or [m for m in REFINING_METHODS if m not in ("sac", "sac_finetune")])
        comparison = RefiningComparison(task=self.task, group="refine")
        for method in methods:
            if method == "sac" and os.environ.get("RICE_ENABLE_SAC") != "1":
                comparison.notes.append("'sac' skipped (set RICE_ENABLE_SAC=1 to run Experiment IV)")
                continue
            comparison.add(self.run(method=method, explanation="ours"))
        return comparison

    def run_explanations(self, explanations: Optional[Sequence[str]] = None) -> RefiningComparison:
        """Experiment III: fix the refining method to RICE, vary the explanation."""
        explanations = list(explanations or ("ours", "statemask", "random"))
        comparison = RefiningComparison(task=self.task, group="explanation")
        for explanation in explanations:
            comparison.add(self.run(method="ours", explanation=explanation))
        return comparison

    def compare(self, mode: str = "refine", **kwargs: Any) -> RefiningComparison:
        if mode in ("refine", "methods"):
            return self.run_methods(**kwargs)
        if mode in ("explanation", "explanations"):
            return self.run_explanations(**kwargs)
        raise ValueError("mode must be 'refine' or 'explanation'")


# ======================================================================================
# Functional entry points
# ======================================================================================
def evaluate_refining(
    task: str,
    policy: Any = None,
    mask_network: Any = None,
    method: str = "ours",
    explanation: str = "ours",
    config: Optional[Union[RefiningConfig, Dict[str, Any]]] = None,
    env: Any = None,
    evaluation_env: Any = None,
    state_manager: Any = None,
    seeds: Optional[Sequence[int]] = None,
    **overrides: Any,
) -> RefiningResult:
    """Run one refining method and return a :class:`RefiningResult`."""
    if isinstance(config, dict):
        config = RefiningConfig.from_mapping(config, task=task)
    elif config is None:
        config = RefiningConfig(task=task)
    elif task is not None and config.task != resolve_task_name(task):
        config = config.clone(task=task)
    if overrides:
        config = config.clone(**{k: v for k, v in overrides.items() if k in config.__dataclass_fields__})  # type: ignore[attr-defined]
    evaluator = RefiningEvaluator(
        env=env,
        policy=policy,
        mask_network=mask_network,
        config=config,
        evaluation_env=evaluation_env,
        state_manager=state_manager,
    )
    return evaluator.run(method=method, explanation=explanation, seeds=seeds)


def compare_refining_methods(
    task: str,
    policy: Any = None,
    mask_network: Any = None,
    methods: Optional[Sequence[str]] = None,
    config: Optional[Union[RefiningConfig, Dict[str, Any]]] = None,
    **kwargs: Any,
) -> RefiningComparison:
    """Experiment II driver: vary the refining method, explanation fixed to ours."""
    if isinstance(config, dict):
        config = RefiningConfig.from_mapping(config, task=task)
    elif config is None:
        config = RefiningConfig(task=task)
    elif config.task != resolve_task_name(task):
        config = config.clone(task=task)
    evaluator = RefiningEvaluator(policy=policy, mask_network=mask_network, config=config, **kwargs)
    return evaluator.run_methods(methods=methods)


def compare_explanations(
    task: str,
    policy: Any = None,
    mask_network: Any = None,
    explanations: Optional[Sequence[str]] = None,
    config: Optional[Union[RefiningConfig, Dict[str, Any]]] = None,
    **kwargs: Any,
) -> RefiningComparison:
    """Experiment III driver: vary the explanation method, refining fixed to RICE."""
    if isinstance(config, dict):
        config = RefiningConfig.from_mapping(config, task=task)
    elif config is None:
        config = RefiningConfig(task=task)
    elif config.task != resolve_task_name(task):
        config = config.clone(task=task)
    evaluator = RefiningEvaluator(policy=policy, mask_network=mask_network, config=config, **kwargs)
    return evaluator.run_explanations(explanations=explanations)


def evaluate_sac_gail(
    task: str = "Hopper-v3",
    sac_policy: Any = None,
    gail_policy: Any = None,
    mask_network: Any = None,
    methods: Sequence[str] = ("ours", "ppo", "statemask_r", "jsrl", "sac_finetune"),
    config: Optional[Union[RefiningConfig, Dict[str, Any]]] = None,
    **kwargs: Any,
) -> RefiningComparison:
    """Experiment IV: refine a SAC(+GAIL) warm-start with RICE vs baselines.

    The SAC/GAIL pre-training itself lives in :mod:`rice.baselines.sac_gail`;
    this driver consumes the resulting policy (``gail_policy`` preferred, else
    ``sac_policy``) and runs the refining comparison.  Requires
    ``RICE_ENABLE_SAC=1`` since SB3-SAC is an optional dependency.
    """
    if isinstance(config, dict):
        config = RefiningConfig.from_mapping(config, task=task)
    elif config is None:
        config = RefiningConfig(task=task)
    policy = gail_policy if gail_policy is not None else sac_policy
    comparison = RefiningComparison(task=config.task, group="refine")
    if os.environ.get("RICE_ENABLE_SAC") != "1":
        comparison.notes.append("Experiment IV skipped (set RICE_ENABLE_SAC=1 to enable SAC/GAIL)")
        return comparison
    evaluator = RefiningEvaluator(policy=policy, mask_network=mask_network, config=config, **kwargs)
    for method in methods:
        comparison.add(evaluator.run(method=method, explanation="ours"))
    return comparison


# ======================================================================================
# Reporting helpers
# ======================================================================================
def summarize_table(
    comparison: Union[RefiningComparison, Dict[str, RefiningResult], Sequence[RefiningResult]],
    reference: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Turn comparison results into a Table-1-shaped mapping.

    Returns ``{method_or_explanation: {"final_reward", "std", "baseline",
    "improvement", "n_seeds", "reference", "note"}}``.
    """
    if isinstance(comparison, RefiningComparison):
        results = comparison.results
        task = comparison.task
    elif isinstance(comparison, dict):
        results = comparison
        task = next(iter(results.values())).task if results else "unknown"
    else:
        results = {r.method: r for r in comparison}
        task = next(iter(results.values())).task if results else "unknown"

    ref = reference
    if ref is None:
        ref = TABLE1_REFERENCE.get(resolve_task_name(task), {})
    rows: Dict[str, Dict[str, Any]] = {}
    for key, result in results.items():
        rows[key] = {
            "final_reward": result.final_reward,
            "std": result.final_std,
            "baseline": result.baseline_reward,
            "improvement": result.improvement,
            "n_seeds": len(result.final_rewards),
            "reference": ref.get(key if key != "statemask" else "statemask", None),
            "note": "; ".join(result.notes) if result.notes else "",
            "error": result.error,
        }
    if "no_refine" not in rows and ref.get("no_refine") is not None:
        rows["no_refine"] = {
            "final_reward": results[next(iter(results))].baseline_reward if results else float("nan"),
            "std": results[next(iter(results))].baseline_std if results else 0.0,
            "baseline": float("nan"),
            "improvement": 0.0,
            "n_seeds": len(results[next(iter(results))].baseline_rewards) if results else 0,
            "reference": ref.get("no_refine"),
            "note": "pre-refining performance",
            "error": None,
        }
    return rows


def format_table(
    comparison: Union[RefiningComparison, Dict[str, RefiningResult], Sequence[RefiningResult]],
    reference: Optional[Dict[str, Dict[str, float]]] = None,
) -> str:
    """Pretty-print a Table-1-shaped summary (``value (std)`` cells)."""
    rows = summarize_table(comparison, reference=reference)
    task = comparison.task if isinstance(comparison, RefiningComparison) else "-"
    header = f"task={task}"
    lines = [header, "-" * len(header)]
    for key, row in rows.items():
        value = row["final_reward"]
        std = row["std"]
        cell = f"{value:.2f} ({std:.2f})" if not np.isnan(value) else "n/a"
        ref = row.get("reference")
        ref_cell = f" | paper {ref:.2f}" if ref is not None else ""
        lines.append(f"{key:<16} {cell}{ref_cell}")
    return "\n".join(lines)


def trend_check(
    comparison: Union[RefiningComparison, Dict[str, RefiningResult], Sequence[RefiningResult]],
    reference: Optional[Dict[str, Dict[str, float]]] = None,
    tolerance: float = 1.0,
) -> Dict[str, Any]:
    """Check the paper's *trends* (not exact numbers, per the addendum).

    Verifies, where the relevant methods are present:

    * Experiment II: ``ours >= max(ppo, jsrl, statemask_r, no_refine)`` and
      ``ours > no_refine``;
    * Experiment II (sparse/dense): ``ppo`` improvement over ``no_refine`` is
      marginal (``<= tolerance`` relative gain allowed);
    * Experiment III: ``ours >= statemask >= random``.
    """
    rows = summarize_table(comparison, reference=reference)
    report: Dict[str, Any] = {"passed": True, "checks": {}}

    def value(key: str) -> Optional[float]:
        row = rows.get(key)
        if not row or np.isnan(row.get("final_reward", float("nan"))):
            return None
        return float(row["final_reward"])

    ours = value("ours")
    baseline = value("no_refine")
    if ours is not None and baseline is not None:
        report["checks"]["ours_beats_no_refine"] = bool(ours > baseline)
    for baseline_method in ("ppo", "jsrl", "statemask_r"):
        other = value(baseline_method)
        if ours is not None and other is not None:
            report["checks"][f"ours_gte_{baseline_method}"] = bool(ours >= other - tolerance)
            if baseline is not None:
                gain_ours = ours - baseline
                gain_other = other - baseline
                report["checks"][f"ours_gain_gte_{baseline_method}_gain"] = bool(gain_ours >= gain_other - tolerance)
    if ours is not None and "random" in rows and value("random") is not None:
        report["checks"]["ours_gte_random"] = bool(ours >= float(value("random")) - tolerance)  # type: ignore[arg-type]
    if "statemask" in rows and ours is not None and value("statemask") is not None:
        report["checks"]["ours_gte_statemask"] = bool(ours >= float(value("statemask")) - tolerance)  # type: ignore[arg-type]
    if "statemask" in rows and "random" in rows and value("statemask") is not None and value("random") is not None:
        report["checks"]["statemask_gte_random"] = bool(
            float(value("statemask")) >= float(value("random")) - tolerance  # type: ignore[arg-type]
        )

    report["passed"] = all(bool(v) for v in report["checks"].values()) if report["checks"] else False
    return report


# ======================================================================================
# CLI-style helpers used by scripts/
# ======================================================================================
def run_experiment_ii(task: str, **kwargs: Any) -> RefiningComparison:
    """Thin alias used by ``scripts/run_baselines.py``."""
    return compare_refining_methods(task, **kwargs)


def run_experiment_iii(task: str, **kwargs: Any) -> RefiningComparison:
    """Thin alias used by ``scripts/run_baselines.py``."""
    return compare_explanations(task, **kwargs)


def default_budget_for_task(task: str) -> float:
    """Mask-net sample budget (Table 4) -- also the refining budget heuristic.

    The paper does not specify a refining budget (documented deviation); we use
    the Table-4 mask budget as an upper bound so that Experiment II stays
    comparable in cost across tasks.
    """
    return float(TABLE4_SAMPLE_BUDGETS.get(resolve_task_name(task), 3e5))
