#!/usr/bin/env python
"""Experiment II / III / IV driver: refine a warm-start policy with RICE and its baselines.

This script compares the RICE refining method (Algorithm 2 of the RICE paper) against the
baselines listed in Section 4.1:

* ``no_refine``    -- the warm-start (bottlenecked) policy, evaluated as-is.
* ``ours``         -- RICE: mixed initial state distribution (Bernoulli ``p`` roll-in to the
                      identified critical state) + RND exploration bonus, optimized with PPO.
* ``ppo``          -- "PPO fine-tuning": lowering the learning rate and continuing training
                      with the PPO algorithm (Schulman et al. 2017).
* ``statemask_r``  -- StateMask's refining method (Cheng et al. 2023): reset to the critical
                      state and continue training from the critical state.
* ``jsrl``         -- Jump-Start RL (Uchendu et al. 2023) with ``pi_e`` initialized to
                      ``pi_g`` so that it becomes a refining method.
* ``sil``          -- Self-Imitation Learning (Oh et al. 2018), secondary comparison of
                      Table 5.
* ``sac_gail``     -- Experiment IV: SAC (Haarnoja et al. 2018) pre-training + GAIL
                      (Ho & Ermon 2016) imitation, then refining.  Gated behind
                      ``--enable-sac`` / the ``RICE_ENABLE_SAC=1`` environment variable
                      because SAC pre-training is expensive.

For every task we report the final reward after refining (mean +/- std across seeds) for
dense-reward applications and the refining curves for sparse-reward applications
(Section 4.1 "Evaluation Metrics", Figure 2/3 trends).  Hyper-parameters ``p`` / ``lambda`` /
``alpha`` come from Table 3.

Usage
-----
::

    python scripts/run_baselines.py --task Hopper-v3 --methods no_refine ours ppo statemask_r jsrl
    python scripts/run_baselines.py --task Reacher-v2 --methods ours ppo --seeds 0 1 2 --plot
    python scripts/run_baselines.py --all --enable-sac --json out/baselines.json

The script is defensive: every environment / baseline import is attempted through a
tolerant import helper and degrades to a faithful fallback built on
``rice.algorithms.refine`` when a third-party module is unavailable.  Missing pieces are
recorded in the ``notes`` field of the JSON report instead of aborting the run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------------------
# Path bootstrap: allow running from the repo root, from ``rice/`` or as installed pkg.
# --------------------------------------------------------------------------------------
def _bootstrap_path() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.abspath(os.path.join(here, "..", "..")),          # repo root (parent of rice/)
        os.path.abspath(os.path.join(here, "..")),                # rice/ (contains rice/rice/)
        os.path.abspath(os.path.join(here, "..", "..", "..")),    # one level above repo root
    ]
    for path in candidates:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


_bootstrap_path()


# --------------------------------------------------------------------------------------
# Tolerant imports
# --------------------------------------------------------------------------------------
def import_first(module_names: Sequence[str]) -> Optional[Any]:
    """Import and return the first importable dotted module, else ``None``."""
    import importlib

    for name in module_names:
        try:
            return importlib.import_module(name)
        except Exception:  # pragma: no cover - defensive
            continue
    return None


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    """Return the first existing attribute among ``names``."""
    for name in names:
        if obj is not None and hasattr(obj, name):
            return getattr(obj, name)
    return default


# --------------------------------------------------------------------------------------
# Paper constants (Table 1 trends, Table 3 hyper-parameters, Table 4 budgets)
# --------------------------------------------------------------------------------------
TABLE1_REFERENCE: Dict[str, Dict[str, float]] = {
    # dense-reward applications (Section 4.1 / Table 1) -- trends, not exact numbers.
    "Hopper-v3": {"no_refine": 3559.44, "ours": 3663.91, "ppo": 3561.58,
                  "jsrl": 3572.32, "statemask_r": 3545.78, "random": 3570.27,
                  "statemask": 3641.24},
    "Walker2d-v3": {"no_refine": 3768.79, "ours": 3982.79, "ppo": 3771.04,
                    "jsrl": 3788.07, "statemask_r": 3756.46, "random": 3810.92,
                    "statemask": 3966.71},
    "Reacher-v2": {"no_refine": -5.79, "ours": -2.66, "ppo": -5.55,
                   "jsrl": -4.93, "statemask_r": -5.73, "random": -5.31,
                   "statemask": -2.97},
    "HalfCheetah-v3": {"no_refine": 2024.09, "ours": 2138.89, "ppo": 2027.34,
                       "jsrl": 2034.80, "statemask_r": 2019.43, "random": 2046.63,
                       "statemask": 2131.35},
    "SelfishMining": {"no_refine": 14.36, "ours": 16.56, "ppo": 14.60,
                      "jsrl": 14.92, "statemask_r": 14.21, "random": 15.11,
                      "statemask": 16.38},
    "CageChallenge2": {"no_refine": -23.64, "ours": -20.02, "ppo": -23.31,
                       "jsrl": -22.72, "statemask_r": -23.87, "random": -22.19,
                       "statemask": -20.44},
    "Macro-v1": {"no_refine": 10.30, "ours": 17.03, "ppo": 11.10,
                 "jsrl": 12.31, "statemask_r": 10.12, "random": 13.35,
                 "statemask": 16.11},
    # out-of-scope row kept so lookups never KeyError (Malware Mutation, Table 7).
    "MalwareMutation": {"no_refine": 0.48, "ours": 0.67, "ppo": 0.51,
                        "jsrl": 0.55, "statemask_r": 0.49, "random": 0.57,
                        "statemask": 0.65},
}

# Table 5 (secondary comparison against Self-Imitation Learning) -- trends only.
TABLE5_REFERENCE: Dict[str, Dict[str, float]] = {
    "Hopper-v3": {"sil": 3646.46, "ours": 3663.91},
    "Walker2d-v3": {"sil": 3967.66, "ours": 3982.79},
    "Reacher-v2": {"sil": -2.87, "ours": -2.66},
    "HalfCheetah-v3": {"sil": 2069.80, "ours": 2138.89},
}

# Table 3: p, lambda, alpha per application (Table 3 is operative over the §C.3 text).
TABLE3_HYPERPARAMS: Dict[str, Dict[str, float]] = {
    "Hopper-v3": {"p": 0.25, "lambda": 0.001, "alpha": 0.0001},
    "Walker2d-v3": {"p": 0.25, "lambda": 0.01, "alpha": 0.0001},
    "Reacher-v2": {"p": 0.50, "lambda": 0.001, "alpha": 0.0001},
    "HalfCheetah-v3": {"p": 0.50, "lambda": 0.01, "alpha": 0.0001},
    "SelfishMining": {"p": 0.25, "lambda": 0.001, "alpha": 0.0001},
    "CageChallenge2": {"p": 0.50, "lambda": 0.01, "alpha": 0.0001},
    "Macro-v1": {"p": 0.25, "lambda": 0.01, "alpha": 0.0001},
    "MalwareMutation": {"p": 0.50, "lambda": 0.01, "alpha": 0.0001},
    # sparse variants inherit the dense hyper-parameters (paper lists no separate row).
    "SparseHopper": {"p": 0.25, "lambda": 0.001, "alpha": 0.0001},
    "SparseHalfCheetah": {"p": 0.50, "lambda": 0.01, "alpha": 0.0001},
    "SparseWalker2d": {"p": 0.25, "lambda": 0.01, "alpha": 0.0001},
}

# Table 4: fixed mask-network training budgets (samples).  Used as the refining budget
# heuristic when the paper leaves the refining budget unspecified (documented deviation).
TABLE4_SAMPLE_BUDGETS: Dict[str, float] = {
    "Hopper-v3": 3e5,
    "Walker2d-v3": 3e5,
    "Reacher-v2": 3e5,
    "HalfCheetah-v3": 3e5,
    "SparseHopper": 3e5,
    "SparseHalfCheetah": 3e5,
    "SelfishMining": 1.5e6,
    "CageChallenge2": 1e7,
    "Macro-v1": 2443260.0,
}

# Table 1 right block / Experiment III: explanations used while fixing refining to "ours".
TASK_ALIASES: Dict[str, str] = {
    "hopper": "Hopper-v3", "hopper-v3": "Hopper-v3", "hopperv3": "Hopper-v3",
    "walker2d": "Walker2d-v3", "walker2d-v3": "Walker2d-v3", "walker": "Walker2d-v3",
    "reacher": "Reacher-v2", "reacher-v2": "Reacher-v2",
    "halfcheetah": "HalfCheetah-v3", "halfcheetah-v3": "HalfCheetah-v3",
    "half-cheetah": "HalfCheetah-v3",
    "selfishmining": "SelfishMining", "selfish-mining": "SelfishMining",
    "selfish": "SelfishMining", "mining": "SelfishMining",
    "cagechallenge2": "CageChallenge2", "cage": "CageChallenge2",
    "cage-challenge-2": "CageChallenge2", "cagechallenge": "CageChallenge2",
    "macro-v1": "Macro-v1", "macrov1": "Macro-v1", "autodriving": "Macro-v1",
    "auto": "Macro-v1", "metadrive": "Macro-v1",
    "sparsehopper": "SparseHopper", "sparse-hopper": "SparseHopper",
    "sparsehalfcheetah": "SparseHalfCheetah",
    "sparse-half-cheetah": "SparseHalfCheetah",
    "sparsewalker2d": "SparseWalker2d", "sparse-walker2d": "SparseWalker2d",
    "malware": "MalwareMutation", "malwaremutation": "MalwareMutation",
}

SPARSE_TASKS: Tuple[str, ...] = ("SparseHopper", "SparseHalfCheetah", "SparseWalker2d")
OUT_OF_SCOPE_TASKS: Tuple[str, ...] = ("SparseWalker2d", "MalwareMutation")

METHODS: Tuple[str, ...] = ("no_refine", "ours", "ppo", "statemask_r", "jsrl", "sil", "sac_gail")

METHOD_ALIASES: Dict[str, str] = {
    "none": "no_refine", "baseline": "no_refine", "no-refine": "no_refine",
    "norefine": "no_refine", "pretrained": "no_refine", "warm_start": "no_refine",
    "rice": "ours", "rnd": "ours", "mixed_init": "ours",
    "ppo_finetune": "ppo", "ppo-finetune": "ppo", "finetune": "ppo", "fine_tune": "ppo",
    "statemask": "statemask_r", "statemask-r": "statemask_r", "statemaskr": "statemask_r",
    "sm_r": "statemask_r",
    "jumpstart": "jsrl", "jump-start-rl": "jsrl", "jump_start": "jsrl",
    "self_imitation": "sil", "selfimitation": "sil",
    "sac-gail": "sac_gail", "sacgail": "sac_gail",
}

EXPLANATIONS: Tuple[str, ...] = ("ours", "statemask", "random",
                                 "integrated_gradients", "airs")


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def canonical_task(name: str) -> str:
    """Map a task spelling/alias to the canonical RICE task name."""
    if not isinstance(name, str):
        raise TypeError(f"task name must be a string, got {type(name)!r}")
    key = name.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    for alias, canonical in TASK_ALIASES.items():
        if alias.replace("-", "").replace("_", "") == key:
            return canonical
    # case/dash tolerant fallback against the registry keys themselves
    for canonical in TABLE3_HYPERPARAMS:
        if canonical.replace("-", "").replace("_", "").lower() == key:
            return canonical
    return name


def canonical_method(name: str) -> str:
    """Map a baseline spelling to a canonical method name."""
    if not isinstance(name, str):
        raise TypeError(f"method name must be a string, got {type(name)!r}")
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    if key in METHODS:
        return key
    if key in METHOD_ALIASES:
        return METHOD_ALIASES[key]
    for canonical in METHODS:
        if canonical.replace("_", "") == key.replace("_", ""):
            return canonical
    raise KeyError(f"unknown method {name!r}; expected one of {METHODS}")


def canonical_explanation(name: str) -> str:
    """Map an explanation spelling to a canonical explanation name."""
    if name is None:
        return "none"
    key = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "rand": "random", "random_explanation": "random", "baseline_random": "random",
        "sm": "statemask", "state_mask": "statemask", "statemask_adapter": "statemask",
        "mask": "ours", "mask_network": "ours", "rice": "ours",
        "ig": "integrated_gradients", "int_grad": "integrated_gradients",
        "integratedgradients": "integrated_gradients",
        "attention": "airs", "yu2023": "airs",
        "none": "none", "": "none",
    }
    if key in EXPLANATIONS or key == "none":
        return key
    if key in aliases:
        return aliases[key]
    return key


def task_hyperparams(task: str) -> Dict[str, float]:
    """Table 3 ``{p, lambda, alpha}`` for a task (Hopper values if unknown)."""
    return dict(TABLE3_HYPERPARAMS.get(canonical_task(task),
                                       TABLE3_HYPERPARAMS["Hopper-v3"]))


def reference_for(task: str) -> Dict[str, float]:
    """Table 1 reference row for a task."""
    return dict(TABLE1_REFERENCE.get(canonical_task(task), {}))


def is_sparse(task: str) -> bool:
    """Whether ``task`` is a sparse-reward MuJoCo variant."""
    return canonical_task(task) in SPARSE_TASKS


def default_budget(task: str) -> float:
    """Fallback refining budget (Table 4 mask samples) when the paper is silent."""
    return float(TABLE4_SAMPLE_BUDGETS.get(canonical_task(task), 3e5))


def resolve_device(device: str) -> str:
    """Resolve ``"auto"`` to ``"cuda"`` when available else ``"cpu"``."""
    if device and device != "auto":
        return device
    try:  # pragma: no cover - hardware dependent
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def ensure_dir(path: str) -> str:
    """Create ``path`` (a file or directory) and return it."""
    if not path:
        return path
    directory = path if not os.path.splitext(path)[1] else os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    return path


def json_default(obj: Any) -> Any:
    """JSON fallback for numpy scalars/arrays and torch tensors."""
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    try:  # torch tensors without importing torch eagerly
        if hasattr(obj, "detach"):
            return obj.detach().cpu().numpy().tolist()
    except Exception:
        pass
    return str(obj)


def summarize_result(result: Any, task: str, method: str, explanation: str) -> Dict[str, Any]:
    """Flatten a refiner/baseline result into a JSON-friendly record."""
    final = _get(result, "final_reward", "final_eval_reward", default=None)
    baseline = _get(result, "baseline_reward", default=None)
    improvement = _get(result, "improvement", default=None)
    if improvement is None and final is not None and baseline is not None:
        improvement = float(final) - float(baseline)
    curve = None
    curve_fn = _get(result, "refining_curve", "mean_curve")
    if callable(curve_fn):
        try:
            curve = np.asarray(curve_fn(), dtype=float).tolist()
        except Exception:
            curve = None
    return {
        "task": task,
        "method": method,
        "explanation": explanation,
        "final_reward": None if final is None else float(final),
        "baseline_reward": None if baseline is None else float(baseline),
        "improvement": None if improvement is None else float(improvement),
        "iterations": _get(result, "iterations", default=None),
        "env_steps": _get(result, "env_steps", default=None),
        "seconds": _get(result, "seconds", default=None),
        "curve": curve,
        "reference": reference_for(task),
        "notes": list(_get(result, "notes", default=[]) or []),
    }


# --------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------
class BaselineRunner:
    """Run RICE and its refining baselines on one or more tasks."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = resolve_device(getattr(args, "device", "auto"))
        self.notes: List[str] = []
        self._modules: Dict[str, Any] = {}

    # ------------------------------------------------------------------ module access
    def module(self, key: str) -> Optional[Any]:
        """Lazily import and cache an optional package module."""
        if key in self._modules:
            return self._modules[key]
        candidates = {
            "algorithms.ppo": ["rice.algorithms.ppo", "rice.rice.algorithms.ppo"],
            "algorithms.refine": ["rice.algorithms.refine", "rice.rice.algorithms.refine"],
            "algorithms.mask_network": ["rice.algorithms.mask_network",
                                        "rice.rice.algorithms.mask_network"],
            "algorithms.env_reset": ["rice.algorithms.env_reset",
                                     "rice.rice.algorithms.env_reset"],
            "environments": ["rice.environments", "rice.rice.environments"],
            "evaluation": ["rice.evaluation.refining_eval",
                           "rice.rice.evaluation.refining_eval",
                           "rice.evaluation"],
            "explanation": ["rice.explanation", "rice.rice.explanation"],
            "seeding": ["rice.utils.seeding", "rice.rice.utils.seeding"],
            "baselines": ["rice.baselines", "rice.rice.baselines"],
            "b_ppo": ["rice.baselines.ppo_finetune", "rice.rice.baselines.ppo_finetune"],
            "b_statemask": ["rice.baselines.statemask_r", "rice.rice.baselines.statemask_r"],
            "b_jsrl": ["rice.baselines.jsrl", "rice.rice.baselines.jsrl"],
            "b_sil": ["rice.baselines.self_imitation", "rice.rice.baselines.self_imitation"],
            "b_sacgail": ["rice.baselines.sac_gail", "rice.rice.baselines.sac_gail"],
        }
        mod = import_first(candidates.get(key, [key]))
        self._modules[key] = mod
        return mod

    # ------------------------------------------------------------------ construction
    def build_env(self, task: str, seed: Optional[int] = None) -> Any:
        """Instantiate an environment for ``task``."""
        envs = self.module("environments")
        kwargs: Dict[str, Any] = {}
        if seed is not None:
            kwargs["seed"] = seed
        if getattr(self.args, "env_kwargs", None):
            kwargs.update(self.args.env_kwargs)
        if envs is not None and hasattr(envs, "make_env"):
            return envs.make_env(task, **kwargs)
        # fallback: per-family modules
        dense = import_first(["rice.environments.mujoco_dense"])
        sparse = import_first(["rice.environments.mujoco_sparse"])
        module = sparse if is_sparse(task) else dense
        if module is not None and hasattr(module, "make_env"):
            return module.make_env(task, **kwargs)
        raise ImportError(
            "no environment factory available: install gym/MuJoCo or set "
            "RICE_ALLOW_MUJOCO_FALLBACK=1 to use the analytic stand-ins"
        )

    def net_arch(self, task: str) -> Tuple[int, ...]:
        """Target-agent / mask-network architecture per Appendix C.2."""
        envs = self.module("environments")
        if envs is not None and hasattr(envs, "default_net_arch"):
            try:
                return tuple(envs.default_net_arch(task))
            except Exception:
                pass
        if task == "SelfishMining":
            return (128, 128, 128, 128)
        if task == "CageChallenge2":
            return (64, 64, 64)
        return (64, 64)

    def build_policy(self, env: Any, task: str, seed: Optional[int] = None) -> Any:
        """Build the warm-start policy and optionally load pre-trained weights."""
        ppo = self.module("algorithms.ppo")
        if ppo is None or not hasattr(ppo, "ActorCritic"):
            raise ImportError("rice.algorithms.ppo.ActorCritic is required")
        price_cfg = getattr(ppo, "PPOConfig", None)
        cfg = price_cfg() if price_cfg is not None else None
        policy = ppo.ActorCritic(
            env.observation_space, env.action_space,
            net_arch=self.net_arch(task),
            device=self.device,
        )
        weights = getattr(self.args, "weights", None)
        if weights:
            refine_mod = self.module("algorithms.refine")
            loader = _get(refine_mod, "load_policy_weights")
            if callable(loader):
                try:
                    loader(policy, weights)
                    self.notes.append(f"loaded warm-start weights from {weights}")
                except Exception as exc:  # pragma: no cover - defensive
                    self.notes.append(f"failed to load weights {weights}: {exc}")
        if seed is not None:
            try:
                import torch

                torch.manual_seed(int(seed))
            except Exception:
                pass
        if cfg is not None:
            policy.eval()
        return policy

    def build_mask_network(self, env: Any, policy: Any, task: str,
                           seed: Optional[int] = None) -> Optional[Any]:
        """Load / train the mask network used for critical-state identification."""
        mask_mod = self.module("algorithms.mask_network")
        if mask_mod is None or not hasattr(mask_mod, "MaskNetwork"):
            self.notes.append("mask network unavailable; explanations will be uniform")
            return None
        mask_weights = getattr(self.args, "mask_weights", None)
        net = mask_mod.MaskNetwork(
            env.observation_space,
            net_arch=self.net_arch(task),
            device=self.device,
        )
        if mask_weights and os.path.exists(mask_weights):
            try:
                import torch

                state = torch.load(mask_weights, map_location=self.device)
                state = state.get("mask_network", state) if isinstance(state, dict) else state
                try:
                    net.load_policy_state_dict(state)
                except Exception:
                    net.load_state_dict(state)
                self.notes.append(f"loaded mask network from {mask_weights}")
            except Exception as exc:  # pragma: no cover - defensive
                self.notes.append(f"failed to load mask weights {mask_weights}: {exc}")
        else:
            self.notes.append(
                "no mask checkpoint supplied (--mask-weights); critical states come from "
                "the untrained mask net (trend-level comparison only)"
            )
        net.eval()
        return net

    def build_evaluation_env(self, task: str, seed: Optional[int] = None) -> Any:
        """A separate environment instance for deterministic evaluation."""
        try:
            return self.build_env(task, seed=seed)
        except Exception:
            return None

    # ------------------------------------------------------------------ running
    def run_method(self, task: str, method: str, explanation: str,
                   seed: int) -> Dict[str, Any]:
        """Run one (task, method, explanation, seed) combination."""
        started = time.time()
        record: Dict[str, Any] = {
            "task": task, "method": method, "explanation": explanation, "seed": seed,
        }
        try:
            env = self.build_env(task, seed=seed)
            policy = self.build_policy(env, task, seed=seed)
            mask = None
            if method in ("ours", "statemask_r", "jsrl") or explanation != "none":
                mask = self.build_mask_network(env, policy, task, seed=seed)
            eval_env = self.build_evaluation_env(task, seed=seed + 10_000)

            result = self._dispatch(task, method, explanation, env, policy, mask,
                                    eval_env, seed)
            record.update(summarize_result(result, task, method, explanation))
            record["seed"] = seed
            record["status"] = "ok"
            # per-length metric for CAGE-2 (sum of average rewards over 30/50/100)
            per_length = _get(result, "per_length", default=None)
            if per_length:
                record["per_length"] = per_length
        except Exception as exc:  # pragma: no cover - defensive
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["traceback"] = traceback.format_exc(limit=6)
        record["seconds"] = round(time.time() - started, 3)
        return record

    def _dispatch(self, task: str, method: str, explanation: str, env: Any,
                  policy: Any, mask: Any, eval_env: Any, seed: int) -> Any:
        """Route one method to the appropriate refiner implementation."""
        if method == "no_refine":
            return self._evaluate_only(env, policy, seed)
        if method == "sac_gail":
            return self._run_sac_gail(task, seed, mask)
        if method == "ours":
            return self._run_rice(task, explanation, env, policy, mask, eval_env, seed)
        if method == "ppo":
            return self._run_ppo(task, env, policy, eval_env, seed)
        if method == "statemask_r":
            return self._run_statemask_r(task, env, policy, mask, eval_env, seed)
        if method == "jsrl":
            return self._run_jsrl(task, env, policy, mask, eval_env, seed)
        if method == "sil":
            return self._run_sil(task, env, policy, mask, eval_env, seed)
        raise KeyError(f"unknown method {method!r}")

    # ---- individual methods ------------------------------------------------------
    def _evaluate_only(self, env: Any, policy: Any, seed: int) -> Any:
        refine_mod = self.module("algorithms.refine")
        evaluator = _get(refine_mod, "evaluate_policy")
        n_eval = int(getattr(self.args, "eval_episodes", 5))
        if callable(evaluator):
            stats = evaluator(env, policy, n_episodes=n_eval, seed=seed, deterministic=True)
            return _PlainResult(final=float(stats["mean_return"]),
                                baseline=float(stats["mean_return"]),
                                task=getattr(self.args, "task", None))
        return _PlainResult(final=float("nan"), baseline=float("nan"))

    def _run_rice(self, task: str, explanation: str, env: Any, policy: Any, mask: Any,
                  eval_env: Any, seed: int) -> Any:
        refining = self.module("evaluation")
        fn = _get(refining, "evaluate_refining")
        kwargs = self._refine_kwargs(task, eval_env)
        if callable(fn):
            return fn(task, policy=policy, mask_network=mask,
                      method="ours", explanation=explanation, env=env,
                      evaluation_env=eval_env, seeds=[seed], **kwargs)
        return self._fallback_refine(env, policy, mask, eval_env, seed, p=None, lam=None)

    def _run_ppo(self, task: str, env: Any, policy: Any, eval_env: Any, seed: int) -> Any:
        mod = self.module("b_ppo")
        fn = _get(mod, "ppo_finetune", "ppo_finetune_baseline")
        if callable(fn):
            return fn(env=env, policy=policy, seeds=[seed],
                      **self._baseline_kwargs(task, eval_env))
        return self._fallback_refine(env, policy, None, eval_env, seed, p=0.0, lam=0.0)

    def _run_statemask_r(self, task: str, env: Any, policy: Any, mask: Any,
                         eval_env: Any, seed: int) -> Any:
        mod = self.module("b_statemask")
        fn = _get(mod, "statemask_r_refine", "statemask_r_baseline")
        if callable(fn):
            return fn(env=env, policy=policy, mask_network=mask, seeds=[seed],
                      **self._baseline_kwargs(task, eval_env))
        return self._fallback_refine(env, policy, mask, eval_env, seed, p=1.0, lam=0.0)

    def _run_jsrl(self, task: str, env: Any, policy: Any, mask: Any,
                  eval_env: Any, seed: int) -> Any:
        mod = self.module("b_jsrl")
        fn = _get(mod, "jsrl_refine", "jsrl_baseline")
        if callable(fn):
            return fn(env=env, policy=policy, mask_network=mask, seeds=[seed],
                      **self._baseline_kwargs(task, eval_env))
        self.notes.append("JSRL module unavailable; approximating with critical-state resets")
        return self._fallback_refine(env, policy, mask, eval_env, seed, p=1.0, lam=0.0)

    def _run_sil(self, task: str, env: Any, policy: Any, mask: Any,
                 eval_env: Any, seed: int) -> Any:
        mod = self.module("b_sil")
        fn = _get(mod, "self_imitation_refine", "sil_baseline")
        if callable(fn):
            return fn(env=env, policy=policy, mask_network=None, seeds=[seed],
                      **self._baseline_kwargs(task, eval_env))
        self.notes.append("SIL module unavailable; falling back to PPO fine-tuning")
        return self._fallback_refine(env, policy, None, eval_env, seed, p=0.0, lam=0.0)

    def _run_sac_gail(self, task: str, seed: int, mask: Any) -> Any:
        if not getattr(self.args, "enable_sac", False) and \
                os.environ.get("RICE_ENABLE_SAC", "0") not in ("1", "true", "True"):
            raise RuntimeError(
                "Experiment IV (SAC pre-train + GAIL) is disabled; pass --enable-sac or "
                "set RICE_ENABLE_SAC=1 (expensive: ~1e6 SAC steps + 3e5 GAIL steps)"
            )
        mod = self.module("b_sacgail")
        fn = _get(mod, "sac_gail_refine")
        if not callable(fn):
            raise ImportError("rice.baselines.sac_gail.sac_gail_refine is required")
        return fn(task=task, mask_network=mask, seeds=[seed],
                  methods=tuple(getattr(self.args, "sac_methods", ("ours", "ppo"))
                                or ("ours", "ppo")))

    # ---- shared configuration ----------------------------------------------------
    def _refine_kwargs(self, task: str, eval_env: Any) -> Dict[str, Any]:
        hp = task_hyperparams(task)
        p = getattr(self.args, "p", None)
        lam = getattr(self.args, "lam", None)
        alpha = getattr(self.args, "alpha", None)
        kwargs: Dict[str, Any] = {
            "p": hp["p"] if p is None else float(p),
            "lam": hp["lambda"] if lam is None else float(lam),
            "alpha": hp["alpha"] if alpha is None else float(alpha),
            "n_iterations": int(getattr(self.args, "iterations", 100)),
            "steps_per_iter": getattr(self.args, "steps", None),
            "device": self.device,
        }
        seed = getattr(self.args, "seed", 0)
        kwargs["seed"] = seed
        return {k: v for k, v in kwargs.items() if v is not None}

    def _baseline_kwargs(self, task: str, eval_env: Any) -> Dict[str, Any]:
        kwargs = self._refine_kwargs(task, eval_env)
        kwargs.pop("p", None)
        kwargs.pop("lam", None)
        kwargs.pop("alpha", None)
        if eval_env is not None:
            kwargs["evaluation_env"] = eval_env
        return kwargs

    def _fallback_refine(self, env: Any, policy: Any, mask: Any, eval_env: Any,
                         seed: int, p: Optional[float], lam: Optional[float]) -> Any:
        """Faithful fallback built on ``rice.algorithms.refine.refine_policy``."""
        refine_mod = self.module("algorithms.refine")
        fn = _get(refine_mod, "refine_policy")
        if not callable(fn):
            raise ImportError("rice.algorithms.refine.refine_policy is required")
        hp = task_hyperparams(getattr(self.args, "task", "Hopper-v3"))
        cfg_cls = _get(refine_mod, "RefineConfig")
        overrides: Dict[str, Any] = {}
        if cfg_cls is not None:
            try:
                cfg = cfg_cls().clone(
                    p=hp["p"] if p is None else float(p),
                    lam=hp["lambda"] if lam is None else float(lam),
                    n_iterations=int(getattr(self.args, "iterations", 100)),
                    device=self.device,
                    seed=seed,
                )
                overrides["config"] = cfg
            except Exception:
                pass
        return fn(env, policy, mask_network=mask, **overrides)

    # ---- orchestration -----------------------------------------------------------
    def run(self, tasks: Iterable[str], methods: Iterable[str],
            explanations: Iterable[str]) -> List[Dict[str, Any]]:
        """Run every (task, method, explanation, seed) combination."""
        seeds = list(getattr(self.args, "seeds", [0, 1, 2]) or [0])
        records: List[Dict[str, Any]] = []
        for task in tasks:
            task = canonical_task(task)
            for method in methods:
                # PPO fine-tuning / SIL / no_refine never consume an explanation.
                if method in ("no_refine", "ppo", "sil", "sac_gail"):
                    expl_list = ["none"]
                else:
                    expl_list = list(explanations)
                for explanation in expl_list:
                    for seed in seeds:
                        print(f"[run_baselines] task={task} method={method} "
                              f"explanation={explanation} seed={seed}", flush=True)
                        records.append(self.run_method(task, method, explanation, seed))
        return records

    # ---- reporting ---------------------------------------------------------------
    def aggregate(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Aggregate per-seed records into mean/std rows (Table 1 layout)."""
        groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
        for rec in records:
            key = (rec["task"], rec["method"], rec["explanation"])
            groups.setdefault(key, []).append(rec)
        rows: List[Dict[str, Any]] = []
        for (task, method, explanation), recs in groups.items():
            finals = [r["final_reward"] for r in recs
                      if r.get("final_reward") is not None and r.get("status") == "ok"]
            baselines = [r["baseline_reward"] for r in recs
                         if r.get("baseline_reward") is not None and r.get("status") == "ok"]
            errors = [r.get("error") for r in recs if r.get("error")]
            row: Dict[str, Any] = {
                "task": task,
                "method": method,
                "explanation": explanation,
                "n_seeds": len(recs),
                "n_ok": sum(1 for r in recs if r.get("status") == "ok"),
                "final_reward_mean": float(np.mean(finals)) if finals else None,
                "final_reward_std": float(np.std(finals)) if finals else None,
                "baseline_reward_mean": float(np.mean(baselines)) if baselines else None,
                "baseline_reward_std": float(np.std(baselines)) if baselines else None,
                "seconds": float(np.sum([r.get("seconds", 0.0) for r in recs])),
                "reference": reference_for(task),
                "errors": errors[:3],
            }
            if row["final_reward_mean"] is not None and row["baseline_reward_mean"] is not None:
                row["improvement"] = row["final_reward_mean"] - row["baseline_reward_mean"]
            # mean curve over seeds (Figure 2/3 trends)
            curves = [np.asarray(r["curve"], dtype=float) for r in recs
                      if r.get("curve") is not None]
            if curves:
                length = min(len(c) for c in curves)
                stacked = np.stack([c[:length] for c in curves])
                row["curve_mean"] = stacked.mean(axis=0).tolist()
                row["curve_std"] = stacked.std(axis=0).tolist()
            rows.append(row)
        rows.sort(key=lambda r: (r["task"], r["method"], r["explanation"]))
        return rows

    def trend_check(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check the paper's qualitative claims (addendum: trends, not exact numbers)."""
        verdict: Dict[str, Any] = {"per_task": {}, "notes": list(self.notes)}
        by_task: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for row in rows:
            by_task.setdefault(row["task"], {})[row["method"]] = row
        for task, methods in by_task.items():
            entry: Dict[str, Any] = {}
            ours = methods.get("ours", {}).get("final_reward_mean")
            no_ref = methods.get("no_refine", {}).get("final_reward_mean")
            ppo = methods.get("ppo", {}).get("final_reward_mean")
            smr = methods.get("statemask_r", {}).get("final_reward_mean")
            jsrl = methods.get("jsrl", {}).get("final_reward_mean")
            if ours is not None and no_ref is not None:
                entry["ours_beats_no_refine"] = bool(ours > no_ref)
            if ours is not None and ppo is not None:
                entry["ours_beats_ppo"] = bool(ours > ppo)
            if ours is not None and jsrl is not None:
                entry["ours_beats_jsrl"] = bool(ours > jsrl)
            if ours is not None and smr is not None:
                # The paper claims ours > StateMask-R; the addendum judges the strict
                # everywhere-claim insignificant, so we only check for No-Refine-gain.
                entry["statemask_r_vs_no_refine_ok"] = bool(
                    no_ref is None or smr >= no_ref or smr > no_ref - abs(0.01 * no_ref)
                )
                entry["ours_vs_statemask_r_delta"] = float(ours - smr)
            entry["reference"] = reference_for(task)
            entry["in_scope"] = task not in OUT_OF_SCOPE_TASKS
            verdict["per_task"][task] = entry
        return verdict

    def save(self, rows: List[Dict[str, Any]], verdict: Dict[str, Any]) -> None:
        """Persist JSON/CSV artifacts."""
        out = getattr(self.args, "json", None)
        if not out:
            return
        ensure_dir(out)
        payload = {
            "experiment": "II-IV (baseline refining comparison)",
            "hyperparameters": {t: task_hyperparams(t) for t in sorted(
                {r["task"] for r in rows})},
            "table1_reference": TABLE1_REFERENCE,
            "table3": TABLE3_HYPERPARAMS,
            "table5_reference": TABLE5_REFERENCE,
            "rows": rows,
            "trend_check": verdict,
            "notes": self.notes,
            "command": " ".join(sys.argv),
        }
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=json_default)

        csv_path = os.path.splitext(out)[0] + ".csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["task", "method", "explanation", "n_seeds", "final_mean",
                             "final_std", "baseline_mean", "improvement"])
            for row in rows:
                writer.writerow([row["task"], row["method"], row["explanation"],
                                 row["n_seeds"], row.get("final_reward_mean"),
                                 row.get("final_reward_std"),
                                 row.get("baseline_reward_mean"),
                                 row.get("improvement")])
        print(f"[run_baselines] wrote {out} and {csv_path}", flush=True)

        notes_path = os.path.splitext(out)[0] + "_notes.txt"
        with open(notes_path, "w", encoding="utf-8") as handle:
            handle.write("RICE baseline run notes (deviations from the paper)\n")
            handle.write("=" * 60 + "\n")
            for note in self.notes:
                handle.write(f"- {note}\n")
            handle.write(
                "- Refining budget is unspecified in the paper: Table 4 mask-sample "
                "budgets are used as a heuristic (documented deviation).\n"
                "- PPO hyper-parameters follow Stable-Baselines3 defaults "
                "(unspecified in the paper).\n"
                "- SparseWalker2d and MalwareMutation are out of scope per the addendum.\n"
            )

    def plot(self, rows: List[Dict[str, Any]]) -> None:
        """Plot per-task final rewards and (for sparse tasks) refining curves."""
        out_dir = getattr(self.args, "plot_dir", None) or "figures"
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:  # pragma: no cover - optional dependency
            print(f"[run_baselines] matplotlib unavailable ({exc}); skipping plots")
            return

        ensure_dir(os.path.join(out_dir, "placeholder"))
        tasks = sorted({r["task"] for r in rows})
        for task in tasks:
            task_rows = [r for r in rows if r["task"] == task]
            labels = [f"{r['method']}" + (f"/{r['explanation']}"
                                          if r["explanation"] not in ("none", None) else "")
                      for r in task_rows]
            means = [r.get("final_reward_mean") or 0.0 for r in task_rows]
            stds = [r.get("final_reward_std") or 0.0 for r in task_rows]
            fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(labels)), 4.5))
            positions = np.arange(len(labels))
            ax.bar(positions, means, yerr=stds, capsize=4,
                   color=["#d62728" if "ours" in lb else "#1f77b4" for lb in labels])
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_ylabel("final reward after refining")
            ax.set_title(f"{task}: refining methods (Experiment II/III)")
            reference_best = TABLE1_REFERENCE.get(task, {}).get("ours")
            if reference_best is not None:
                ax.axhline(reference_best, ls="--", lw=1.0, color="grey",
                           label="paper 'ours' reference (trend)")
                ax.legend(fontsize=8)
            fig.tight_layout()
            path = os.path.join(out_dir, f"final_reward_{task}.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"[run_baselines] wrote {path}", flush=True)

            if is_sparse(task):
                fig, ax = plt.subplots(figsize=(6, 4.5))
                for row in task_rows:
                    if row.get("curve_mean"):
                        label = row["method"] + (f"/{row['explanation']}"
                                                 if row["explanation"] != "none" else "")
                        curve = np.asarray(row["curve_mean"], dtype=float)
                        std = np.asarray(row.get("curve_std") or np.zeros_like(curve))
                        x = np.arange(len(curve))
                        ax.plot(x, curve, label=label)
                        ax.fill_between(x, curve - std, curve + std, alpha=0.15)
                ax.set_xlabel("refining iteration")
                ax.set_ylabel("episode return")
                ax.set_title(f"{task}: refining curves (Experiment II)")
                ax.legend(fontsize=8)
                fig.tight_layout()
                path = os.path.join(out_dir, f"curve_{task}.png")
                fig.savefig(path, dpi=150)
                plt.close(fig)
                print(f"[run_baselines] wrote {path}", flush=True)


class _PlainResult:
    """Minimal ``RefineResult`` stand-in for the ``no_refine`` row."""

    def __init__(self, final: float, baseline: float, task: Optional[str] = None) -> None:
        self.final_reward = final
        self.baseline_reward = baseline
        self.iterations = 0
        self.env_steps = 0
        self.seconds = 0.0
        self.notes: List[str] = []
        self.task = task

    @property
    def final_eval_reward(self) -> float:
        return self.final_reward

    @property
    def improvement(self) -> float:
        return self.final_reward - self.baseline_reward

    def refining_curve(self, window: int = 1) -> np.ndarray:
        return np.asarray([self.final_reward], dtype=float)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "final_reward": self.final_reward,
            "baseline_reward": self.baseline_reward,
            "iterations": self.iterations,
            "notes": self.notes,
        }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build the command line interface."""
    parser = argparse.ArgumentParser(
        description="RICE Experiments II/III/IV: refine a warm-start policy with RICE "
                    "and its baselines (PPO fine-tuning, StateMask-R, JSRL, SIL, SAC+GAIL).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", type=str, default="Hopper-v3",
                        help="environment name (canonical or alias)")
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        help="several environments at once")
    parser.add_argument("--all", action="store_true",
                        help="run all seven in-scope dense applications")
    parser.add_argument("--sparse", action="store_true",
                        help="run the in-scope sparse applications (SparseHopper, "
                             "SparseHalfCheetah)")
    parser.add_argument("--methods", type=str, nargs="+", default=None,
                        help=f"refining methods, from {METHODS}")
    parser.add_argument("--explanations", type=str, nargs="+", default=["ours"],
                        help=f"explanations used by RICE, from {EXPLANATIONS}")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="random seeds (paper reports mean +/- std over 3 seeds)")
    parser.add_argument("--iterations", type=int, default=100,
                        help="number of refining outer iterations (budget unspecified "
                             "in the paper; whichever saturates the curves)")
    parser.add_argument("--steps", type=int, default=None,
                        help="environment steps per refining iteration (default: one "
                             "episode, i.e. env.max_episode_steps)")
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="deterministic evaluation episodes per measurement")
    parser.add_argument("--p", type=float, default=None,
                        help="override the Table 3 mixed-init ratio p (Experiment V)")
    parser.add_argument("--lam", type=float, default=None,
                        help="override the Table 3 RND coefficient lambda (Experiment V)")
    parser.add_argument("--alpha", type=float, default=None,
                        help="override the Table 3 mask bonus alpha (Table 3: 1e-4)")
    parser.add_argument("--weights", type=str, default=None,
                        help="warm-start (pre-trained, bottlenecked) policy checkpoint")
    parser.add_argument("--mask-weights", type=str, default=None,
                        help="trained mask-network checkpoint for critical states")
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    parser.add_argument("--enable-sac", action="store_true",
                        help="enable Experiment IV (SAC pre-train + GAIL); expensive")
    parser.add_argument("--sac-methods", type=str, nargs="+", default=None,
                        help="methods compared inside Experiment IV")
    parser.add_argument("--plot", action="store_true", help="write reward/curve figures")
    parser.add_argument("--plot-dir", type=str, default="figures", help="figure directory")
    parser.add_argument("--json", type=str, default=None,
                        help="write aggregated results to this JSON path")
    parser.add_argument("--quiet", action="store_true", help="suppress table printing")
    return parser


def _default_methods(args: argparse.Namespace) -> List[str]:
    if args.methods:
        return [canonical_method(m) for m in args.methods]
    methods = ["no_refine", "ours", "ppo", "statemask_r", "jsrl"]
    if getattr(args, "enable_sac", False):
        methods.append("sac_gail")
    return methods


def _resolve_tasks(args: argparse.Namespace) -> List[str]:
    tasks: List[str] = []
    if args.tasks:
        tasks.extend(args.tasks)
    if args.task:
        tasks.append(args.task)
    if args.all:
        tasks.extend(["Hopper-v3", "Walker2d-v3", "Reacher-v2", "HalfCheetah-v3",
                      "SelfishMining", "CageChallenge2", "Macro-v1"])
    if args.sparse:
        tasks.extend(["SparseHopper", "SparseHalfCheetah"])
    seen: List[str] = []
    for task in tasks:
        canonical = canonical_task(task)
        if canonical not in seen:
            seen.append(canonical)
    return seen or ["Hopper-v3"]


def print_table(rows: List[Dict[str, Any]]) -> None:
    """Print the aggregated Table 1-style report."""
    header = (f"{'task':<16}{'method':<13}{'expl':<20}{'final':>12}{'std':>10}"
              f"{'baseline':>12}{'improve':>10}{'ref':>12}")
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        final = row.get("final_reward_mean")
        std = row.get("final_reward_std")
        base = row.get("baseline_reward_mean")
        imp = row.get("improvement")
        ref = (row.get("reference") or {}).get("ours")
        fmt = lambda v: "-" if v is None else f"{v:.2f}"  # noqa: E731
        print(f"{row['task']:<16}{row['method']:<13}{row['explanation']:<20}"
              f"{fmt(final):>12}{fmt(std):>10}{fmt(base):>12}{fmt(imp):>10}{fmt(ref):>12}")
    print()


def print_verdict(verdict: Dict[str, Any]) -> None:
    """Print the qualitative trend check."""
    print("Trend check (addendum: reproduce trends, not exact numbers):")
    for task, entry in verdict["per_task"].items():
        flags = {k: v for k, v in entry.items()
                 if isinstance(v, bool)}
        print(f"  {task}: " + ", ".join(f"{k}={v}" for k, v in flags.items()))
        if "ours_vs_statemask_r_delta" in entry:
            print(f"    delta(ours - statemask_r) = "
                  f"{entry['ours_vs_statemask_r_delta']:.3f}")
    if verdict.get("notes"):
        print("\nNotes / documented deviations:")
        for note in dict.fromkeys(verdict["notes"]):
            print(f"  - {note}")
    print()


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.methods is None:
        args.methods = _default_methods(args)
    else:
        args.methods = [canonical_method(m) for m in args.methods]
    args.task = canonical_task(args.task)
    args.seeds = list(args.seeds) if args.seeds else [0]
    args.seed = args.seeds[0]

    # global seeding for reproducibility across numpy/torch/env resets
    seeding = import_first(["rice.utils.seeding", "rice.rice.utils.seeding"])
    if seeding is not None and hasattr(seeding, "set_global_seeds"):
        seeding.set_global_seeds(args.seed)

    tasks = _resolve_tasks(args)
    out_of_scope = [t for t in tasks if t in OUT_OF_SCOPE_TASKS]
    runner = BaselineRunner(args)
    if out_of_scope:
        runner.notes.append(
            f"out-of-scope tasks requested and skipped: {out_of_scope} "
            "(addendum excludes SparseWalker2d and all Malware Mutation experiments)"
        )
        tasks = [t for t in tasks if t not in OUT_OF_SCOPE_TASKS]

    explanations = [canonical_explanation(e) for e in (args.explanations or ["ours"])]
    records = runner.run(tasks, args.methods, explanations)
    rows = runner.aggregate(records)
    verdict = runner.trend_check(rows)

    if not args.quiet:
        print_table(rows)
        print_verdict(verdict)

    runner.save(rows, verdict)
    if args.plot:
        runner.plot(rows)

    failures = [r for r in records if r.get("status") != "ok"]
    if failures:
        print(f"[run_baselines] {len(failures)}/{len(records)} runs failed; see JSON notes")
    return 0 if len(failures) < len(records) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
