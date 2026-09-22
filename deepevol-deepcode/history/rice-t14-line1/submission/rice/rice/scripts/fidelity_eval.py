"""Experiment I driver: fidelity of the explanation + mask-training efficiency.

Implements the two halves of Experiment I of the RICE paper (Proc. 41st ICML 2024):

1. **Fidelity** (Sec. 4.1 "Evaluation Metrics", Sec. 4.2 "Experiment I", Fig. 5).
   For each explanation method we

   * roll out the (pre-trained) target policy and score every visited state with
     the explanation method's importance;
   * slide a window of width ``l = L * K`` (``L`` = trajectory length) and take the
     window with the highest average importance;
   * fast-forward to the critical step, force the target agent to take random
     actions for ``l`` steps (masking), then follow the policy again;
   * measure the reward change ``d = |R' - R|`` and compute

     ``Fidelity Score = log(d / d_max) - log(l / L)``

     exactly as reported in Sec. 4.1.

   Settings from Sec. 4.2: 500 trajectories, ``K = 10%, 20%, 30%, 40%``, repeated
   3 times with different random seeds, reporting mean and standard deviation.

2. **Efficiency** (Table 4). Train the mask network with a *fixed number of
   samples* (Table 4 budgets) using our method (Algorithm 1, vanilla PPO) and
   compare the wall-clock seconds against the published StateMask numbers; the
   paper reports ``16.8%`` less time on average.

The heavy lifting lives in :mod:`rice.evaluation.fidelity_score` and
:mod:`rice.algorithms.mask_network`; this file is the CLI/orchestration layer
plus artifact writing + trend checking (the addendum asks for trends, not exact
numbers).

Unspecified in the paper (documented deviations, also printed in the notes):
``d_max`` is estimated per-environment from warm-up episodes unless given, and
the mask-network training budget for the *refining* experiments is not stated
(only the mask-training budget is, via Table 4).
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------------------
# Path bootstrap: allow running as `python scripts/fidelity_eval.py` from the repo root,
# from inside `rice/`, or as an installed package.
# --------------------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (
    os.path.dirname(_HERE),                        # rice/rice
    os.path.dirname(os.path.dirname(_HERE)),       # repo root
    os.path.dirname(os.path.dirname(os.path.dirname(_HERE))),
):
    if _cand and _cand not in sys.path:
        sys.path.insert(0, _cand)


# --------------------------------------------------------------------------------------
# Reference constants (verbatim from the paper)
# --------------------------------------------------------------------------------------
#: Table 4 -- number of samples used to train the mask network (fixed budget).
TABLE4_SAMPLE_BUDGETS: Dict[str, float] = {
    "Hopper-v3": 3e5,
    "Walker2d-v3": 3e5,
    "Reacher-v2": 3e5,
    "HalfCheetah-v3": 3e5,
    "SelfishMining": 1.5e6,
    "CageChallenge2": 1e7,
    "Macro-v1": 2443260,
    # out of scope, kept so lookups do not KeyError
    "MalwareMutation": 32349,
}

#: Table 4 -- seconds to train the mask network with StateMask / with our method.
TABLE4_STATEMASK_SECONDS: Dict[str, float] = {
    "Hopper-v3": 15393,
    "Walker2d-v3": 2240,
    "Reacher-v2": 8571,
    "HalfCheetah-v3": 1579,
    "SelfishMining": 9520,
    "CageChallenge2": 79382,
    "Macro-v1": 109802,
    "MalwareMutation": 50775,
}
TABLE4_OURS_SECONDS: Dict[str, float] = {
    "Hopper-v3": 12426,
    "Walker2d-v3": 1899,
    "Reacher-v2": 7033,
    "HalfCheetah-v3": 1317,
    "SelfishMining": 8360,
    "CageChallenge2": 65400,
    "Macro-v1": 88761,
    "MalwareMutation": 41340,
}
#: Sec. C.3 "Efficiency Comparison": our method is 16.8% faster on average.
EFFICIENCY_DROP_REFERENCE = 0.168

#: Sec. 4.2 Experiment I settings.
DEFAULT_KS: Tuple[float, ...] = (0.10, 0.20, 0.30, 0.40)
DEFAULT_N_TRAJECTORIES = 500
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2)

#: Sec. 4.1 -- explanation methods compared in Fig. 5 (+ Table 6 extras).
EXPLANATIONS: Tuple[str, ...] = (
    "ours",
    "statemask",
    "random",
    "integrated_gradients",
    "airs",
)

#: Table 3 -- alpha used when *training* a mask network for the fidelity study.
DEFAULT_ALPHA = 1e-4
ALPHA_SWEEP_VALUES: Tuple[float, ...] = (0.01, 0.001, 0.0001)

DENSE_TASKS: Tuple[str, ...] = (
    "Hopper-v3",
    "Walker2d-v3",
    "Reacher-v2",
    "HalfCheetah-v3",
    "SelfishMining",
    "CageChallenge2",
    "Macro-v1",
)
SPARSE_TASKS: Tuple[str, ...] = ("SparseHopper", "SparseHalfCheetah")
ALL_TASKS: Tuple[str, ...] = DENSE_TASKS + SPARSE_TASKS
OUT_OF_SCOPE_TASKS: Tuple[str, ...] = ("SparseWalker2d", "MalwareMutation")

TASK_ALIASES: Dict[str, str] = {
    "hopper": "Hopper-v3",
    "hopper-v3": "Hopper-v3",
    "walk": "Walker2d-v3",
    "walker": "Walker2d-v3",
    "walker2d": "Walker2d-v3",
    "walker2d-v3": "Walker2d-v3",
    "reacher": "Reacher-v2",
    "reacher-v2": "Reacher-v2",
    "halfcheetah": "HalfCheetah-v3",
    "half-cheetah": "HalfCheetah-v3",
    "halfcheetah-v3": "HalfCheetah-v3",
    "selfish": "SelfishMining",
    "selfishmining": "SelfishMining",
    "selfish_mining": "SelfishMining",
    "cage": "CageChallenge2",
    "cage2": "CageChallenge2",
    "cagechallenge2": "CageChallenge2",
    "cage_challenge2": "CageChallenge2",
    "auto": "Macro-v1",
    "autodriving": "Macro-v1",
    "macro": "Macro-v1",
    "macro-v1": "Macro-v1",
    "sparsehopper": "SparseHopper",
    "sparse_hopper": "SparseHopper",
    "sparsehalfcheetah": "SparseHalfCheetah",
    "sparse_halfcheetah": "SparseHalfCheetah",
}

EXPLANATION_ALIASES: Dict[str, str] = {
    "our": "ours",
    "rice": "ours",
    "mask": "ours",
    "masknetwork": "ours",
    "state_mask": "statemask",
    "state-mask": "statemask",
    "statemask_adapter": "statemask",
    "rand": "random",
    "random_explanation": "random",
    "ig": "integrated_gradients",
    "integratedgradients": "integrated_gradients",
    "int_grad": "integrated_gradients",
    "attention": "airs",
    "yu2023": "airs",
}

#: Expected trend of Fig. 5 (addendum: ours ~ StateMask > Random).
EXPECTED_TREND: Dict[str, str] = {
    "ours": "high",
    "statemask": "high",
    "random": "low",
}


# --------------------------------------------------------------------------------------
# Small shared helpers (mirroring the style of the other scripts in this repo)
# --------------------------------------------------------------------------------------
def canonical_task(name: str) -> str:
    """Map a task spelling/alias to the canonical RICE task name."""
    if name is None:
        return "Hopper-v3"
    key = str(name).strip()
    if key in ALL_TASKS or key in OUT_OF_SCOPE_TASKS:
        return key
    lowered = key.lower().replace(" ", "").replace("_", "").replace("-", "")
    for alias, target in TASK_ALIASES.items():
        if alias.replace("_", "").replace("-", "") == lowered:
            return target
    # tolerate "Hopper" / "HalfCheetah" etc.
    for task in ALL_TASKS:
        if task.lower().replace("-", "").replace("_", "") == lowered:
            return task
    return key


def canonical_explanation(name: str) -> str:
    """Map an explanation spelling/alias to a canonical explanation name."""
    if name is None:
        return "ours"
    key = str(name).strip().lower()
    if key in EXPLANATIONS:
        return key
    if key in EXPLANATION_ALIASES:
        return EXPLANATION_ALIASES[key]
    normalized = key.replace("-", "_").replace(" ", "_")
    if normalized in EXPLANATIONS:
        return normalized
    raise KeyError(f"Unknown explanation method {name!r}. Known: {EXPLANATIONS}")


def is_sparse(task: str) -> bool:
    """Whether the task is a sparse-reward variant."""
    task = canonical_task(task)
    return task in SPARSE_TASKS or task.lower().startswith("sparse")


def resolve_device(device: str = "auto") -> str:
    """Resolve ``auto`` to ``cuda``/``cpu``."""
    if device and device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def import_first(module_names: Sequence[str]) -> Optional[Any]:
    """Import the first importable dotted module (layout tolerance)."""
    for name in module_names:
        try:
            return importlib.import_module(name)
        except Exception:
            continue
    return None


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    """Fetch the first existing attribute from ``obj``."""
    for name in names:
        if obj is not None and hasattr(obj, name):
            return getattr(obj, name)
    return default


def ensure_dir(path: str) -> str:
    if path:
        os.makedirs(path, exist_ok=True)
    return path


def json_default(obj: Any) -> Any:
    """JSON fallback for numpy / torch objects."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    try:  # torch tensors
        return obj.detach().cpu().numpy().tolist()
    except Exception:
        pass
    return str(obj)


def task_budget(task: str, override: Optional[float] = None) -> float:
    """Table 4 mask-training sample budget for ``task``."""
    if override is not None:
        return float(override)
    return float(TABLE4_SAMPLE_BUDGETS.get(canonical_task(task), 3e5))


def task_alpha(override: Optional[float] = None) -> float:
    """Table 3 alpha (operative value, per the addendum: 1e-4)."""
    return float(DEFAULT_ALPHA if override is None else override)


# --------------------------------------------------------------------------------------
# Config of one fidelity-evaluation run
# --------------------------------------------------------------------------------------
@dataclass
class FidelityRunConfig:
    """Settings of an Experiment I fidelity evaluation."""

    task: str = "Hopper-v3"
    ks: Tuple[float, ...] = DEFAULT_KS
    n_trajectories: int = DEFAULT_N_TRAJECTORIES
    seeds: Tuple[int, ...] = DEFAULT_SEEDS
    d_max: Optional[float] = None
    estimate_d_max: bool = True
    d_max_warmup_episodes: int = 5
    max_steps: Optional[int] = None
    deterministic_actions: bool = False
    random_window: bool = False
    capture_mode: str = "replay"
    device: str = "auto"
    out_dir: str = "results/fidelity"
    weights: Optional[str] = None
    mask_weights: Optional[str] = None
    alpha: float = DEFAULT_ALPHA
    net_arch: Optional[Tuple[int, ...]] = None
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    verbose: int = 1
    log_every: int = 100

    def clone(self, **overrides: Any) -> "FidelityRunConfig":
        data = dict(self.__dict__)
        data.update(overrides)
        return FidelityRunConfig(**data)

    @classmethod
    def from_mapping(cls, mapping: Optional[Dict[str, Any]] = None, **overrides: Any) -> "FidelityRunConfig":
        data: Dict[str, Any] = dict(mapping or {})
        data.update(overrides)
        if "K" in data and "ks" not in data:
            data["ks"] = data.pop("K")
        if "num_trajectories" in data:
            data["n_trajectories"] = data.pop("num_trajectories")
        if "tasks" in data:
            task = data.pop("tasks")
            if isinstance(task, (list, tuple)):
                task = task[0] if task else "Hopper-v3"
            data["task"] = task
        known = set(cls.__dataclass_fields__.keys())  # type: ignore[attr-defined]
        unknown = {k: v for k, v in data.items() if k not in known}
        data = {k: v for k, v in data.items() if k in known}
        cfg = cls(**data)
        if unknown:
            cfg.env_kwargs.setdefault("_config_extra", unknown)
        return cfg

    def seed_list(self) -> List[int]:
        seeds = list(self.seeds) if self.seeds is not None else [0, 1, 2]
        return [int(s) for s in seeds]

    def ks_list(self) -> List[float]:
        return [float(k) for k in self.ks]


# --------------------------------------------------------------------------------------
# Mask-training efficiency (Table 4)
# --------------------------------------------------------------------------------------
@dataclass
class MaskTimingResult:
    """Wall-clock result of training the mask network with a fixed sample budget."""

    task: str
    samples: float
    seconds: float
    mean_mask_rate: float
    alpha: float
    reference_statemask_seconds: Optional[float] = None
    reference_ours_seconds: Optional[float] = None
    relative_to_statemask: Optional[float] = None
    reference_drop: Optional[float] = None
    collapsed: bool = False
    error: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "samples": float(self.samples),
            "seconds": float(self.seconds),
            "mean_mask_rate": float(self.mean_mask_rate),
            "alpha": float(self.alpha),
            "reference_statemask_seconds": self.reference_statemask_seconds,
            "reference_ours_seconds": self.reference_ours_seconds,
            "relative_to_statemask": self.relative_to_statemask,
            "reference_drop": self.reference_drop,
            "collapsed": bool(self.collapsed),
            "error": self.error,
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------
class FidelityRunner:
    """Orchestrates Experiment I (fidelity + efficiency) for one or more tasks."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = resolve_device(getattr(args, "device", "auto"))
        self.notes: List[str] = []
        self.fidelity_module = import_first(
            ("rice.evaluation.fidelity_score", "rice.rice.evaluation.fidelity_score")
        )
        self.mask_module = import_first(
            ("rice.algorithms.mask_network", "rice.rice.algorithms.mask_network")
        )
        self.explanation_module = import_first(
            ("rice.explanation", "rice.rice.explanation")
        )
        self.env_module = import_first(
            ("rice.environments", "rice.rice.environments")
        )
        self.refine_module = import_first(
            ("rice.algorithms.refine", "rice.rice.algorithms.refine")
        )
        if self.fidelity_module is None:
            self.notes.append(
                "rice.evaluation.fidelity_score not importable; fidelity numbers unavailable."
            )

    # ---------------------------------------------------------------- environment
    def build_env(self, task: str, seed: int = 0):
        """Construct a single (non-vectorised) environment."""
        task = canonical_task(task)
        env_kwargs = dict(getattr(self.args, "env_kwargs", {}) or {})
        # thin out private config extras
        env_kwargs = {k: v for k, v in env_kwargs.items() if not k.startswith("_")}
        env_kwargs.pop("seed", None)

        if self.env_module is not None and hasattr(self.env_module, "make_env"):
            try:
                return self.env_module.make_env(task, seed=seed, **env_kwargs)
            except Exception as exc:  # pragma: no cover - defensive
                self.notes.append(f"make_env({task}) failed ({exc}); trying task module.")

        for module_name, factory_names in (
            ("rice.environments.mujoco_sparse", ("make_env", "make_sparse_env", "make_mujoco_sparse")),
            ("rice.environments.mujoco_dense", ("make_env", "make_dense_env", "make_mujoco_dense")),
            ("rice.environments.selfish_mining", ("make_env", "make_selfish_mining_env")),
            ("rice.environments.cage_challenge2", ("make_env", "make_cage_env")),
            ("rice.environments.autodriving", ("make_env", "make_autodriving_env")),
        ):
            module = import_first((module_name, module_name.replace("rice.environments", "rice.rice.environments")))
            if module is None:
                continue
            for fname in factory_names:
                factory = getattr(module, fname, None)
                if factory is None:
                    continue
                try:
                    return factory(task, seed=seed, **env_kwargs)
                except TypeError:
                    try:
                        return factory(name=task, seed=seed, **env_kwargs)
                    except Exception:
                        continue
                except Exception:
                    continue
        raise RuntimeError(f"Unable to construct environment {task!r}")

    def net_arch(self, task: str) -> Tuple[int, ...]:
        if getattr(self.args, "net_arch", None):
            return tuple(self.args.net_arch)
        arch = None
        if self.env_module is not None and hasattr(self.env_module, "default_net_arch"):
            try:
                arch = self.env_module.default_net_arch(canonical_task(task))
            except Exception:
                arch = None
        return tuple(arch) if arch else (64, 64)

    # ---------------------------------------------------------------- policy
    def build_policy(self, env: Any, task: str, seed: int = 0):
        """Construct the warm-start target policy and load weights when given."""
        ppo_module = import_first(("rice.algorithms.ppo", "rice.rice.algorithms.ppo"))
        if ppo_module is None or not hasattr(ppo_module, "ActorCritic"):
            return None
        obs_space = getattr(env, "observation_space", None)
        act_space = getattr(env, "action_space", None)
        policy = ppo_module.ActorCritic(
            observation_space=obs_space,
            action_space=act_space,
            net_arch=self.net_arch(task),
            device=self.device,
        )
        weights = self._resolve_weights(task, seed)
        if weights and self.refine_module is not None and hasattr(self.refine_module, "load_policy_weights"):
            try:
                self.refine_module.load_policy_weights(policy, weights, strict=False)
            except Exception as exc:  # pragma: no cover
                self.notes.append(f"could not load weights {weights!r} ({exc}); using random policy.")
        elif weights:
            try:
                import torch

                state = torch.load(weights, map_location=self.device)
                if isinstance(state, dict) and "policy" in state:
                    state = state["policy"]
                policy.load_state_dict(state, strict=False)
            except Exception as exc:  # pragma: no cover
                self.notes.append(f"could not load weights {weights!r} ({exc}); using random policy.")
        else:
            self.notes.append(
                f"no pre-trained weights for {task} (seed {seed}); fidelity measured on an untrained policy."
            )
        try:
            policy.eval()
        except Exception:
            pass
        return policy

    def _resolve_weights(self, task: str, seed: int) -> Optional[str]:
        explicit = getattr(self.args, "weights", None)
        if explicit:
            cand = explicit.format(task=task, seed=seed) if "{" in explicit else explicit
            if os.path.exists(cand):
                return cand
            if os.path.isdir(cand):
                for name in (
                    f"{task}_seed{seed}.pt",
                    f"{task.lower()}_seed{seed}.pt",
                    f"{task}.pt",
                ):
                    p = os.path.join(cand, name)
                    if os.path.exists(p):
                        return p
        out_dir = getattr(self.args, "pretrain_dir", None) or "results/pretrain"
        for cand in (
            os.path.join(out_dir, "weights", f"{task}_seed{seed}.pt"),
            os.path.join(out_dir, f"{task}_seed{seed}.pt"),
        ):
            if os.path.exists(cand):
                return cand
        return None

    # ---------------------------------------------------------------- explanation
    def build_explanation(self, name: str, env: Any, policy: Any, task: str, seed: int = 0):
        """Build an explanation provider exposing ``importance(states)``.

        Priority: RICE explanation registry -> provider module -> mask network
        checkpoint -> freshly trained mask network (Algorithm 1) -> uniform scores.
        """
        name = canonical_explanation(name)

        # 1) registry dispatch (knows about all providers + aliases)
        if self.explanation_module is not None and hasattr(self.explanation_module, "make_explanation"):
            try:
                kwargs: Dict[str, Any] = {}
                if name == "random":
                    kwargs["seed"] = seed
                else:
                    kwargs.update(env=env, policy=policy, task=task, seed=seed, device=self.device)
                expl = self.explanation_module.make_explanation(name, **kwargs)
                if expl is not None:
                    return expl
            except Exception as exc:
                self.notes.append(f"explanation registry failed for {name!r} ({exc}); using fallback.")

        # 2) direct provider construction
        if name == "random":
            module = import_first(
                ("rice.explanation.random_explanation", "rice.rice.explanation.random_explanation")
            )
            factory = _get(module, "make_random_explanation", "build_random_explanation", "make_explanation")
            if factory is not None:
                try:
                    return factory(seed=seed)
                except Exception:
                    pass
        if name in ("ours", "statemask"):
            module = import_first(
                ("rice.explanation.statemask_adapter", "rice.rice.explanation.statemask_adapter")
            )
            factory = _get(module, "make_statemask_explanation", "build_statemask", "make_explanation")
            if factory is not None:
                try:
                    mask_net = self.build_mask_network(env, policy, task, seed, train=False)
                    return factory(env=env, policy=policy, mask_network=mask_net, seed=seed, task=task)
                except Exception:
                    pass
        if name == "integrated_gradients":
            module = import_first(
                ("rice.explanation.integrated_gradients", "rice.rice.explanation.integrated_gradients")
            )
            factory = _get(module, "make_integrated_gradients", "build_integrated_gradients", "make_explanation")
            if factory is not None:
                try:
                    return factory(env=env, policy=policy, seed=seed, device=self.device)
                except Exception:
                    pass
        if name == "airs":
            module = import_first(("rice.explanation.airs_adapter", "rice.rice.explanation.airs_adapter"))
            factory = _get(module, "make_airs_explanation", "build_airs", "make_explanation")
            if factory is not None:
                try:
                    return factory(env=env, policy=policy, seed=seed, device=self.device)
                except Exception:
                    pass

        # 3) mask-network fallback (a plain callable importance function)
        mask_net = self.build_mask_network(env, policy, task, seed, train=False)
        if mask_net is not None:
            self.notes.append(f"explanation {name!r} resolved to a bare mask network.")
            return mask_net
        self.notes.append(
            f"explanation {name!r} unavailable; falling back to uniform importance scores."
        )
        return None

    def build_mask_network(self, env: Any, policy: Any, task: str, seed: int = 0, train: bool = False):
        """Build (and optionally train) the mask network used as the explanation."""
        if self.mask_module is None or not hasattr(self.mask_module, "MaskNetwork"):
            return None
        mask_net = None
        try:
            mask_net = self.mask_module.MaskNetwork(
                observation_space=getattr(env, "observation_space", None),
                net_arch=self.net_arch(task),
                device=self.device,
            )
        except Exception as exc:
            self.notes.append(f"MaskNetwork construction failed ({exc}).")
            return None
        weights = self._resolve_mask_weights(task, seed)
        if weights:
            try:
                import torch

                state = torch.load(weights, map_location=self.device)
                if isinstance(state, dict) and "mask_network" in state:
                    state = state["mask_network"]
                if hasattr(mask_net, "load_policy_state_dict"):
                    mask_net.load_policy_state_dict(state)
                else:
                    mask_net.load_state_dict(state, strict=False)
            except Exception as exc:
                self.notes.append(f"could not load mask weights {weights!r} ({exc}).")
        elif train:
            try:
                trainer_cls = getattr(self.mask_module, "MaskNetworkTrainer", None)
                cfg_cls = getattr(self.mask_module, "MaskNetworkConfig", None)
                if trainer_cls is None or cfg_cls is None:
                    return mask_net
                cfg = cfg_cls(alpha=task_alpha(getattr(self.args, "alpha", None)), device=self.device)
                trainer = trainer_cls(env=env, target_policy=policy, mask_network=mask_net, config=cfg)
                trainer.train()
            except Exception as exc:
                self.notes.append(f"mask training inside explanation failed ({exc}).")
        return mask_net

    def _resolve_mask_weights(self, task: str, seed: int) -> Optional[str]:
        explicit = getattr(self.args, "mask_weights", None)
        if explicit:
            cand = explicit.format(task=task, seed=seed) if "{" in explicit else explicit
            if os.path.exists(cand):
                return cand
            if os.path.isdir(cand):
                for name in (f"{task}_seed{seed}.pt", f"{task}.pt"):
                    p = os.path.join(cand, name)
                    if os.path.exists(p):
                        return p
        mask_dir = getattr(self.args, "mask_dir", None) or "results/mask"
        for cand in (
            os.path.join(mask_dir, "weights", f"{task}_seed{seed}.pt"),
            os.path.join(mask_dir, f"{task}_seed{seed}.pt"),
        ):
            if os.path.exists(cand):
                return cand
        return None

    # ---------------------------------------------------------------- fidelity
    def _fidelity_config(self, d_max: Optional[float]) -> Any:
        if self.fidelity_module is None or not hasattr(self.fidelity_module, "FidelityConfig"):
            return None
        cfg = self.fidelity_module.FidelityConfig(
            ks=tuple(self.args.ks),
            num_trajectories=int(self.args.trajectories),
            seeds=tuple(self.args.seeds),
            d_max=d_max,
            max_steps=getattr(self.args, "max_steps", None),
            deterministic_actions=bool(getattr(self.args, "deterministic_actions", False)),
            random_window=bool(getattr(self.args, "random_window", False)),
            capture_mode=getattr(self.args, "capture_mode", "replay"),
            seed=int(self.args.seed),
            device=self.device,
            verbose=int(getattr(self.args, "verbose", 1)),
        )
        return cfg

    def estimate_d_max(self, env: Any, policy: Any) -> Optional[float]:
        """Estimate per-environment ``d_max`` (unspecified in the paper)."""
        if getattr(self.args, "d_max", None) is not None:
            return float(self.args.d_max)
        if not getattr(self.args, "estimate_d_max", True):
            return None
        if self.fidelity_module is None or not hasattr(self.fidelity_module, "estimated_d_max"):
            return None
        n_episodes = int(getattr(self.args, "d_max_warmup_episodes", 5) or 5)
        returns: List[float] = []
        for ep in range(n_episodes):
            try:
                ret = self._rollout_return(env, policy, seed=1000 + ep)
                returns.append(float(ret))
            except Exception as exc:
                self.notes.append(f"d_max warm-up episode failed ({exc}).")
                break
        if not returns:
            return None
        return self.fidelity_module.estimated_d_max(np.asarray(returns))

    def _rollout_return(self, env: Any, policy: Any, seed: int = 0, max_steps: Optional[int] = None) -> float:
        """One episode of the target policy, returning the total reward."""
        reset = getattr(env, "reset", None)
        step = getattr(env, "step", None)
        try:
            out = reset(seed=seed)
        except TypeError:
            out = reset()
        obs = out[0] if isinstance(out, tuple) else out
        total = 0.0
        limit = max_steps or getattr(self.args, "max_steps", None)
        for t in range(int(limit) if limit else 100000):
            action = self._policy_action(policy, obs)
            out = step(action)
            if len(out) == 5:
                obs, reward, terminated, truncated, _info = out
                done = bool(terminated) or bool(truncated)
            else:
                obs, reward, done, _info = out
                done = bool(done)
            total += float(reward)
            if done:
                break
        return total

    def _policy_action(self, policy: Any, obs: Any) -> np.ndarray:
        if policy is None:
            return np.zeros(1, dtype=np.float32)
        for name in ("predict", "act"):
            fn = getattr(policy, name, None)
            if callable(fn):
                try:
                    out = fn(obs, deterministic=False)
                    if isinstance(out, tuple):
                        out = out[0]
                    return np.asarray(out)
                except Exception:
                    continue
        if callable(policy):
            try:
                return np.asarray(policy(obs))
            except Exception:
                pass
        return np.zeros(1, dtype=np.float32)

    # ---------------------------------------------------------------- efficiency
    def time_mask_training(self, task: str, seed: int = 0, samples: Optional[float] = None,
                           alpha: Optional[float] = None) -> MaskTimingResult:
        """Train the mask network with a fixed sample budget and time it (Table 4)."""
        task = canonical_task(task)
        budget = task_budget(task, samples)
        alpha_value = task_alpha(alpha if alpha is not None else getattr(self.args, "alpha", None))
        result = MaskTimingResult(
            task=task,
            samples=budget,
            seconds=float("nan"),
            mean_mask_rate=float("nan"),
            alpha=alpha_value,
            reference_statemask_seconds=TABLE4_STATEMASK_SECONDS.get(task),
            reference_ours_seconds=TABLE4_OURS_SECONDS.get(task),
        )
        if self.mask_module is None or not hasattr(self.mask_module, "MaskNetworkTrainer"):
            result.error = "rice.algorithms.mask_network.MaskNetworkTrainer not importable"
            return result
        try:
            env = self.build_env(task, seed=seed)
            policy = self.build_policy(env, task, seed=seed)
            mask_net = self.mask_module.MaskNetwork(
                observation_space=getattr(env, "observation_space", None),
                net_arch=self.net_arch(task),
                device=self.device,
            )
            cfg_cls = getattr(self.mask_module, "MaskNetworkConfig")
            cfg = cfg_cls(
                alpha=alpha_value,
                total_samples=budget,
                device=self.device,
                seed=seed,
            )
            trainer = self.mask_module.MaskNetworkTrainer(
                env=env, target_policy=policy, mask_network=mask_net, config=cfg
            )
            t0 = time.time()
            info = trainer.train()
            seconds = float(time.time() - t0)
            result.seconds = seconds
            result.mean_mask_rate = float((info or {}).get("mean_mask_rate", float("nan")))
            result.collapsed = bool(
                not np.isnan(result.mean_mask_rate) and result.mean_mask_rate <= 1e-6
            )
            if result.reference_statemask_seconds:
                result.relative_to_statemask = seconds / float(result.reference_statemask_seconds)
                result.reference_drop = 1.0 - result.relative_to_statemask
            if result.collapsed:
                result.notes.append(
                    "mask network collapsed to always-KEEP (mean mask rate ~ 0): "
                    "increase alpha / training budget."
                )
        except Exception as exc:  # pragma: no cover - defensive
            result.error = f"{type(exc).__name__}: {exc}"
            result.notes.append(traceback.format_exc().splitlines()[-1])
        return result

    # ---------------------------------------------------------------- experiment
    def run_fidelity(self, tasks: Sequence[str], explanations: Sequence[str]) -> Dict[str, Any]:
        """Run the fidelity comparison (Fig. 5) + timing (Table 4) for each task."""
        payload: Dict[str, Any] = {
            "config": {
                "ks": [float(k) for k in self.args.ks],
                "trajectories": int(self.args.trajectories),
                "seeds": [int(s) for s in self.args.seeds],
                "explanations": [canonical_explanation(e) for e in explanations],
                "device": self.device,
            },
            "tasks": {},
        }

        for task in tasks:
            task = canonical_task(task)
            if task in OUT_OF_SCOPE_TASKS:
                self.notes.append(f"{task} is out of scope per the addendum; skipped.")
                continue

            entry: Dict[str, Any] = {"task": task, "fidelity": {}, "timing": None, "d_max": None}
            env = None
            policy = None
            try:
                env = self.build_env(task, seed=int(self.args.seed))
                policy = self.build_policy(env, task, seed=int(self.args.seed))
            except Exception as exc:
                entry["error"] = f"environment/policy construction failed: {exc}"
                payload["tasks"][task] = entry
                continue

            d_max = self.estimate_d_max(env, policy) if env is not None else None
            entry["d_max"] = None if d_max is None else float(d_max)
            if d_max is None:
                self.notes.append(
                    f"{task}: d_max not provided/estimated; fidelity scores rely on the evaluator default."
                )

            if self.fidelity_module is not None and hasattr(self.fidelity_module, "evaluate_methods"):
                method_map: Dict[str, Any] = {}
                for name in explanations:
                    name = canonical_explanation(name)
                    try:
                        method_map[name] = self.build_explanation(name, env, policy, task, seed=int(self.args.seed))
                    except Exception as exc:
                        self.notes.append(f"{task}/{name}: explanation construction failed ({exc}).")
                cfg = self._fidelity_config(d_max)
                try:
                    results = self.fidelity_module.evaluate_methods(
                        env, policy, method_map, config=cfg
                    )
                    for name, result in (results or {}).items():
                        entry["fidelity"][name] = self._result_to_dict(result)
                except Exception as exc:
                    entry["error"] = f"fidelity evaluation failed: {exc}"
                    entry["traceback"] = traceback.format_exc()
                    self.notes.append(f"{task}: fidelity evaluation failed ({exc}).")
            else:
                entry["error"] = "rice.evaluation.fidelity_score.evaluate_methods unavailable"

            # ---- Table 4 timing (only when requested; training is expensive)
            if getattr(self.args, "time_mask", False):
                timings = []
                for seed in self.args.seeds:
                    timings.append(
                        self.time_mask_training(task, seed=int(seed), samples=getattr(self.args, "samples", None))
                    )
                entry["timing"] = {
                    "per_seed": [t.as_dict() for t in timings],
                    "mean_seconds": float(np.nanmean([t.seconds for t in timings])),
                    "samples": float(timings[0].samples) if timings else None,
                    "alpha": float(timings[0].alpha) if timings else None,
                    "mean_mask_rate": float(np.nanmean([t.mean_mask_rate for t in timings])),
                }
                ref_sm = TABLE4_STATEMASK_SECONDS.get(task)
                if ref_sm:
                    entry["timing"]["reference_statemask_seconds"] = float(ref_sm)
                    entry["timing"]["relative_to_statemask"] = (
                        entry["timing"]["mean_seconds"] / float(ref_sm)
                    )
                    entry["timing"]["reference_drop"] = 1.0 - entry["timing"]["relative_to_statemask"]

            payload["tasks"][task] = entry
            try:
                _close(env)
            except Exception:
                pass

        payload["notes"] = list(self.notes)
        return payload

    @staticmethod
    def _result_to_dict(result: Any) -> Dict[str, Any]:
        if result is None:
            return {}
        if hasattr(result, "as_dict"):
            try:
                return result.as_dict()
            except Exception:
                pass
        if isinstance(result, dict):
            return result
        return {"repr": repr(result)}

    # ---------------------------------------------------------------- reporting
    def trend_check(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Verify the Fig. 5 trend: ours ~ StateMask > Random."""
        verdict: Dict[str, Any] = {"trend": EXPECTED_TREND, "tasks": {}, "summary": {}}
        n_ok = 0
        n_total = 0
        for task, entry in (payload.get("tasks") or {}).items():
            fidelity = entry.get("fidelity") or {}
            if not fidelity:
                continue
            means = {}
            for name, data in fidelity.items():
                means_all: List[float] = []
                scores = data.get("scores") if isinstance(data, dict) else None
                if isinstance(scores, (list, tuple)):
                    for row in scores:
                        if isinstance(row, dict) and "mean" in row:
                            means_all.append(float(row["mean"]))
                        elif isinstance(row, (list, tuple)) and row:
                            means_all.append(float(np.mean(row)))
                    if means_all:
                        means[name] = float(np.mean(means_all))
                if name not in means and isinstance(data, dict) and "means" in data:
                    arr = np.asarray(data["means"], dtype=float)
                    means[name] = float(np.nanmean(arr)) if arr.size else None
            task_verdict: Dict[str, Any] = {"means": means}
            ok = True
            if "ours" in means and "random" in means and means["random"] is not None:
                ok = ok and means["ours"] > means["random"]
                task_verdict["ours_gt_random"] = bool(means["ours"] > means["random"])
            if "statemask" in means and "random" in means and means["random"] is not None:
                ok = ok and means["statemask"] > means["random"]
                task_verdict["statemask_gt_random"] = bool(means["statemask"] > means["random"])
            if "ours" in means and "statemask" in means and means["statemask"] is not None:
                # equivalence (addendum: strict superiority is ignored)
                delta = abs(means["ours"] - means["statemask"])
                task_verdict["ours_statemask_abs_diff"] = float(delta)
            task_verdict["ok"] = bool(ok)
            verdict["tasks"][task] = task_verdict
            n_total += 1
            n_ok += int(ok)
        verdict["summary"] = {
            "tasks_ok": n_ok,
            "tasks_total": n_total,
            "ok": bool(n_total > 0 and n_ok == n_total),
        }
        return verdict

    def efficiency_check(self, payload: Dict[str, Any], tolerance: float = 0.05) -> Dict[str, Any]:
        """Check the Table 4 trend: our mask training is faster than StateMask."""
        rows: List[Dict[str, Any]] = []
        for task, entry in (payload.get("tasks") or {}).items():
            timing = entry.get("timing")
            if not timing:
                continue
            ref = timing.get("reference_statemask_seconds")
            measured = timing.get("mean_seconds")
            rows.append(
                {
                    "task": task,
                    "samples": timing.get("samples"),
                    "seconds": measured,
                    "reference_statemask_seconds": ref,
                    "reference_ours_seconds": TABLE4_OURS_SECONDS.get(task),
                    "relative_to_reference": timing.get("relative_to_statemask"),
                    "drop": timing.get("reference_drop"),
                }
            )
        drops = [r["drop"] for r in rows if r.get("drop") is not None]
        verdict = {
            "rows": rows,
            "mean_drop": float(np.mean(drops)) if drops else None,
            "reference_mean_drop": EFFICIENCY_DROP_REFERENCE,
            "tolerance": tolerance,
        }
        if drops:
            verdict["ok"] = bool(abs(float(np.mean(drops)) - EFFICIENCY_DROP_REFERENCE) <= tolerance)
        else:
            verdict["ok"] = None
        return verdict

    # ---------------------------------------------------------------- artifacts
    def save(self, payload: Dict[str, Any], verdict: Dict[str, Any], efficiency: Dict[str, Any]) -> None:
        out_dir = ensure_dir(getattr(self.args, "out_dir", "results/fidelity"))
        with open(os.path.join(out_dir, "fidelity.json"), "w") as fh:
            json.dump(payload, fh, indent=2, default=json_default)
        with open(os.path.join(out_dir, "fidelity_trend.json"), "w") as fh:
            json.dump({"trend": verdict, "efficiency": efficiency}, fh, indent=2, default=json_default)

        rows: List[Dict[str, Any]] = []
        for task, entry in (payload.get("tasks") or {}).items():
            fidelity = entry.get("fidelity") or {}
            for name, data in fidelity.items():
                if isinstance(data, dict) and isinstance(data.get("scores"), (list, tuple)):
                    for row in data["scores"]:
                        if isinstance(row, dict):
                            rows.append({"task": task, "method": name, **row})
        if rows:
            keys = sorted({k for row in rows for k in row})
            with open(os.path.join(out_dir, "fidelity.csv"), "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=keys)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)

        timing_rows = efficiency.get("rows") or []
        if timing_rows:
            keys = sorted({k for row in timing_rows for k in row})
            with open(os.path.join(out_dir, "table4_timing.csv"), "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=keys)
                writer.writeheader()
                for row in timing_rows:
                    writer.writerow(row)

        notes = payload.get("notes") or []
        if notes:
            with open(os.path.join(out_dir, "notes.txt"), "w") as fh:
                fh.write("\n".join(str(n) for n in notes) + "\n")

    def plot(self, payload: Dict[str, Any]) -> Optional[str]:
        """Figure-5-style bar plot of fidelity scores over K for each method."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            self.notes.append("matplotlib unavailable; skipping fidelity plot.")
            return None

        tasks = [t for t, e in (payload.get("tasks") or {}).items() if e.get("fidelity")]
        if not tasks:
            return None
        ks = [float(k) for k in self.args.ks]
        fig, axes = plt.subplots(1, len(tasks), figsize=(4 * len(tasks), 3.2), squeeze=False)
        for ax, task in zip(axes[0], tasks):
            fidelity = payload["tasks"][task]["fidelity"]
            for name, data in fidelity.items():
                if not isinstance(data, dict) or not isinstance(data.get("scores"), (list, tuple)):
                    continue
                means, stds = [], []
                for row in data["scores"]:
                    if isinstance(row, dict):
                        means.append(row.get("mean", np.nan))
                        stds.append(row.get("std", 0.0) or 0.0)
                    elif isinstance(row, (list, tuple)) and row:
                        means.append(float(np.mean(row)))
                        stds.append(float(np.std(row)))
                if len(means) == len(ks):
                    ax.errorbar(ks, means, yerr=stds, marker="o", capsize=3, label=name)
            ax.set_title(task, fontsize=9)
            ax.set_xlabel("K")
            ax.set_ylabel("fidelity score")
            ax.set_xticks(ks)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        fig.tight_layout()
        path = os.path.join(ensure_dir(getattr(self.args, "out_dir", "results/fidelity")), "fidelity_scores.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path


def _close(env: Any) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fidelity_eval",
        description="RICE Experiment I: fidelity scores of explanation methods (+ Table 4 timing).",
    )
    parser.add_argument("--task", default=None, help="single task name/alias")
    parser.add_argument("--tasks", nargs="*", default=None, help="one or more task names/aliases")
    parser.add_argument("--all", action="store_true", help="run all in-scope tasks")
    parser.add_argument("--sparse", action="store_true", help="include the sparse MuJoCo tasks")
    parser.add_argument(
        "--explanations",
        nargs="*",
        default=["ours", "statemask", "random"],
        help=f"explanation methods (known: {', '.join(EXPLANATIONS)})",
    )
    parser.add_argument("--ks", nargs="*", type=float, default=list(DEFAULT_KS),
                        help="window fractions K (Sec. 4.2: 0.1 0.2 0.3 0.4)")
    parser.add_argument("--trajectories", type=int, default=DEFAULT_N_TRAJECTORIES,
                        help="number of trajectories per (K, seed) cell (paper: 500)")
    parser.add_argument("--seeds", nargs="*", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--seed", type=int, default=0, help="seed used for env/policy/explanation construction")
    parser.add_argument("--d-max", dest="d_max", type=float, default=None,
                        help="maximum possible reward change per episode (paper: unspecified)")
    parser.add_argument("--no-estimate-d-max", dest="estimate_d_max", action="store_false", default=True,
                        help="disable the data-driven d_max estimate")
    parser.add_argument("--d-max-warmup-episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--deterministic-actions", action="store_true")
    parser.add_argument("--random-window", action="store_true",
                        help="choose a uniformly random window instead of the argmax-importance one")
    parser.add_argument("--capture-mode", default="replay", choices=("replay", "step"))
    parser.add_argument("--time-mask", action="store_true",
                        help="also train/tim the mask network (Table 4)")
    parser.add_argument("--samples", type=float, default=None,
                        help="override the Table 4 mask-training sample budget")
    parser.add_argument("--alpha", type=float, default=None,
                        help="alpha of the blinding bonus (Table 3: 0.0001)")
    parser.add_argument("--weights", default=None, help="pre-trained target policy weights (path or dir)")
    parser.add_argument("--mask-weights", dest="mask_weights", default=None, help="mask network weights")
    parser.add_argument("--pretrain-dir", dest="pretrain_dir", default="results/pretrain")
    parser.add_argument("--mask-dir", dest="mask_dir", default="results/mask")
    parser.add_argument("--net-arch", dest="net_arch", nargs="*", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", dest="out_dir", default="results/fidelity")
    parser.add_argument("--plot", action="store_true", help="produce Figure-5-style plot")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--verbose", type=int, default=1)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.verbose = 0 if args.quiet else args.verbose
    args.env_kwargs = {}

    # ---- resolve tasks
    tasks: List[str] = []
    if args.tasks:
        tasks.extend(canonical_task(t) for t in args.tasks)
    if args.task:
        tasks.append(canonical_task(args.task))
    if args.all or args.sparse:
        tasks.extend(DENSE_TASKS)
        if args.sparse:
            tasks.extend(SPARSE_TASKS)
    tasks = list(dict.fromkeys(tasks))
    if not tasks:
        tasks = ["Hopper-v3"]

    explanations = [canonical_explanation(e) for e in args.explanations]

    # ---- seeding
    try:
        seeding = import_first(("rice.utils.seeding", "rice.rice.utils.seeding"))
        if seeding is not None and hasattr(seeding, "set_global_seeds"):
            seeding.set_global_seeds(args.seed)
    except Exception:
        pass

    runner = FidelityRunner(args)
    payload = runner.run_fidelity(tasks, explanations)
    verdict = runner.trend_check(payload)
    efficiency = runner.efficiency_check(payload) if args.time_mask else {"rows": [], "ok": None}
    runner.save(payload, verdict, efficiency)

    if args.plot:
        runner.plot(payload)

    if not args.quiet:
        print("=" * 78)
        print("RICE Experiment I -- fidelity of explanation methods")
        print("=" * 78)
        print(f"tasks        : {', '.join(tasks)}")
        print(f"explanations : {', '.join(explanations)}")
        print(f"K values     : {list(args.ks)}   trajectories: {args.trajectories}   seeds: {list(args.seeds)}")
        print(f"device       : {runner.device}")
        for task, entry in (payload.get("tasks") or {}).items():
            print(f"\n[{task}] d_max={entry.get('d_max')}")
            if entry.get("error"):
                print(f"  error: {entry['error']}")
            for name, data in (entry.get("fidelity") or {}).items():
                scores = data.get("scores") if isinstance(data, dict) else None
                if scores:
                    pretty = ", ".join(
                        f"K={row.get('k', '?')}: {row.get('mean', float('nan')):.3f}±{row.get('std', 0.0) or 0.0:.3f}"
                        for row in scores
                        if isinstance(row, dict)
                    )
                    print(f"  {name:<22} {pretty}")
            timing = entry.get("timing")
            if timing:
                print(
                    f"  mask timing: {timing.get('mean_seconds'):.1f}s for {timing.get('samples')} samples "
                    f"(StateMask reference {timing.get('reference_statemask_seconds')}s, "
                    f"drop {timing.get('reference_drop')})"
                )
        print("\nTrend check (ours ~ statemask > random):")
        for task, tv in verdict["tasks"].items():
            print(f"  {task:<16} ok={tv.get('ok')} means={ {k: round(v, 3) for k, v in tv.get('means', {}).items() if v is not None} }")
        print(f"  summary: {verdict['summary']}")
        if args.time_mask:
            print(f"\nEfficiency check (Table 4, mean drop target {EFFICIENCY_DROP_REFERENCE:.3f}):")
            print(f"  measured mean drop = {efficiency.get('mean_drop')} ok={efficiency.get('ok')}")
        notes = payload.get("notes") or []
        if notes:
            print("\nNotes / documented deviations:")
            for n in notes:
                print(f"  - {n}")
        print(f"\nArtifacts written to: {os.path.abspath(args.out_dir)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
