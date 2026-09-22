#!/usr/bin/env python
"""RICE ``run_refine`` driver -- Algorithm 2 for Experiments II / III / IV.

This script refines a warm-start (bottlenecked) DRL policy with RICE's
Algorithm 2 (mixed initial state distribution + RND exploration bonus, PPO
update) and/or the baseline refining methods, then reports the final reward
(dense tasks, Table 1) and the refining curves (sparse tasks, Figure 2).

Paper references
----------------
* Algorithm 2 -- "Refining the DRL Agent" (Sec. 3.3).
* Sec. 4.1 Experiment Setup -- baselines (PPO fine-tuning, StateMask-R, JSRL).
* Sec. 4.2 Experiment II (refining effectiveness) / III (different
  explanations) / IV (non-PPO pre-trained agent, SAC + GAIL).
* Table 3 -- per-environment ``p`` / ``lambda`` / ``alpha``.

Usage (examples)
----------------
::

    # RICE on Hopper (Table 3 defaults, 3 seeds)
    python scripts/run_refine.py --task Hopper-v3 --method ours --seeds 0 1 2

    # Experiment II -- compare RICE against all baselines on Reacher
    python scripts/run_refine.py --task Reacher-v2 --methods ours ppo statemask_r jsrl

    # Experiment III -- fix the refining method, vary the explanation
    python scripts/run_refine.py --task HalfCheetah-v3 --explanations ours statemask random

    # Experiment V -- hyper-parameter overrides
    python scripts/run_refine.py --task Hopper-v3 --p 0.5 --lambda 0.01

    # Sparse task (refining curves, Figure 2)
    python scripts/run_refine.py --task SparseHopper --curves

The script is intentionally tolerant: it degrades gracefully when a task's
simulator (MuJoCo / CAGE-2 / MetaDrive) or a third-party baseline is missing,
falling back to the closest faithful re-implementation shipped in this repo.
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

# --------------------------------------------------------------------------
# Path bootstrap: allow running from the repository root *or* from rice/rice.
# --------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
for _candidate in (_ROOT, os.path.join(_ROOT, "rice"), os.path.join(_ROOT, "..")):
    _candidate = os.path.abspath(_candidate)
    if os.path.isdir(os.path.join(_candidate, "rice")) or os.path.isdir(
        os.path.join(_candidate, "algorithms")
    ):
        if _candidate not in sys.path:
            sys.path.insert(0, _candidate)


# --------------------------------------------------------------------------
# Reference numbers (trend checks only -- see README "reproduce trends").
# Source: Table 1.  Keys are canonical task names used by
# ``rice.evaluation.refining_eval``.
# --------------------------------------------------------------------------
TABLE1_REFERENCE: Dict[str, Dict[str, float]] = {
    "Hopper-v3": {"no_refine": 3559.44, "ppo": 3576.46, "jsrl": 3580.92,
                  "statemask_r": 3496.26, "ours": 3663.91},
    "Walker2d-v3": {"no_refine": 3768.79, "ppo": 3789.62, "jsrl": 3795.16,
                    "statemask_r": 3746.06, "ours": 3982.79},
    "Reacher-v2": {"no_refine": -5.79, "ppo": -5.62, "jsrl": -5.14,
                   "statemask_r": -6.26, "ours": -2.66},
    "HalfCheetah-v3": {"no_refine": 2024.09, "ppo": 2038.89, "jsrl": 2056.66,
                       "statemask_r": 1955.68, "ours": 2138.89},
    "SelfishMining": {"no_refine": 14.36, "ppo": 14.78, "jsrl": 14.97,
                      "statemask_r": 13.71, "ours": 16.56},
    "CageChallenge2": {"no_refine": -23.64, "ppo": -23.33, "jsrl": -22.96,
                       "statemask_r": -24.01, "ours": -20.02},
    "Macro-v1": {"no_refine": 10.30, "ppo": 10.92, "jsrl": 11.48,
                 "statemask_r": 9.85, "ours": 17.03},
}

# Table 3 -- per-environment hyper-parameters (operative per the addendum).
TABLE3_HYPERPARAMS: Dict[str, Dict[str, float]] = {
    "Hopper-v3": {"p": 0.25, "lambda": 0.001, "alpha": 0.0001},
    "Walker2d-v3": {"p": 0.25, "lambda": 0.01, "alpha": 0.0001},
    "Reacher-v2": {"p": 0.50, "lambda": 0.001, "alpha": 0.0001},
    "HalfCheetah-v3": {"p": 0.50, "lambda": 0.01, "alpha": 0.0001},
    "SelfishMining": {"p": 0.25, "lambda": 0.001, "alpha": 0.0001},
    "CageChallenge2": {"p": 0.50, "lambda": 0.01, "alpha": 0.0001},
    "Macro-v1": {"p": 0.25, "lambda": 0.01, "alpha": 0.0001},
    # Out of scope (Malware Mutation) kept only so lookups do not KeyError.
    "MalwareMutation": {"p": 0.50, "lambda": 0.01, "alpha": 0.0001},
}

# Alias table: task spellings -> canonical name.
TASK_ALIASES: Dict[str, str] = {
    "hopper": "Hopper-v3", "hopper-v3": "Hopper-v3", "hopperv3": "Hopper-v3",
    "walker": "Walker2d-v3", "walker2d": "Walker2d-v3",
    "walker2d-v3": "Walker2d-v3", "walker2dv3": "Walker2d-v3",
    "reacher": "Reacher-v2", "reacher-v2": "Reacher-v2", "reacherv2": "Reacher-v2",
    "halfcheetah": "HalfCheetah-v3", "half_cheetah": "HalfCheetah-v3",
    "halfcheetah-v3": "HalfCheetah-v3", "halfcheetahv3": "HalfCheetah-v3",
    "selfish": "SelfishMining", "selfishmining": "SelfishMining",
    "selfish_mining": "SelfishMining", "blockchain": "SelfishMining",
    "cage": "CageChallenge2", "cagechallenge2": "CageChallenge2",
    "cage-2": "CageChallenge2", "cage2": "CageChallenge2",
    "networkdefense": "CageChallenge2", "network_defense": "CageChallenge2",
    "auto": "Macro-v1", "autodriving": "Macro-v1", "autonomousdriving": "Macro-v1",
    "macro": "Macro-v1", "macro-v1": "Macro-v1", "metadrive": "Macro-v1",
    "sparsehopper": "SparseHopper", "sparse_hopper": "SparseHopper",
    "sparsehalfcheetah": "SparseHalfCheetah", "sparse_halfcheetah": "SparseHalfCheetah",
}

SPARSE_TASKS: Tuple[str, ...] = ("SparseHopper", "SparseHalfCheetah")

# Refining methods understood by --method/--methods.
METHODS: Tuple[str, ...] = ("no_refine", "ours", "ppo", "statemask_r", "jsrl", "sil", "sac_gail")
METHOD_ALIASES: Dict[str, str] = {
    "rice": "ours", "rnd": "ours", "refine": "ours", "algorithm2": "ours",
    "none": "no_refine", "norefine": "no_refine", "baseline": "no_refine",
    "no_refine": "no_refine",
    "ppo_finetune": "ppo", "ppo-ft": "ppo", "ppo_ft": "ppo", "finetune": "ppo",
    "statemask": "statemask_r", "statemask-r": "statemask_r",
    "statemaskr": "statemask_r", "smr": "statemask_r",
    "jumpstart": "jsrl", "jump-start-rl": "jsrl", "jump_start": "jsrl",
    "self_imitation": "sil", "selfimitation": "sil",
    "sacgail": "sac_gail", "sac-gail": "sac_gail",
}

EXPLANATIONS: Tuple[str, ...] = ("ours", "statemask", "random",
                                 "integrated_gradients", "airs")


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def canonical_task(name: str) -> str:
    """Map a task spelling to a canonical RICE task name."""
    if name is None:
        return "Hopper-v3"
    key = str(name).strip()
    if key in TABLE1_REFERENCE or key in TABLE3_HYPERPARAMS:
        return key
    key_norm = key.lower().replace(" ", "").replace("_", "").replace("-", "")
    for alias, canon in TASK_ALIASES.items():
        if alias.replace("-", "").replace("_", "") == key_norm:
            return canon
    # Fall back to a case-insensitive direct match.
    for canon in list(TABLE1_REFERENCE) + list(TABLE3_HYPERPARAMS):
        if canon.lower() == key.lower():
            return canon
    return key


def canonical_method(name: str) -> str:
    """Map a method spelling to a canonical refining-method name."""
    key = str(name).strip().lower().replace(" ", "_")
    if key in METHODS:
        return key
    return METHOD_ALIASES.get(key, key)


def canonical_explanation(name: str) -> str:
    """Map an explanation spelling to a canonical name."""
    key = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "ours": "ours", "rice": "ours", "mask": "ours", "masknetwork": "ours",
        "statemask": "statemask", "state_mask": "statemask", "sm": "statemask",
        "random": "random", "rand": "random",
        "integrated_gradients": "integrated_gradients", "ig": "integrated_gradients",
        "int_grad": "integrated_gradients",
        "airs": "airs", "attention": "airs",
    }
    return mapping.get(key, key)


def task_hyperparams(task: str) -> Dict[str, float]:
    """Table 3 hyper-parameters for ``task`` (Hopper defaults if unknown)."""
    canon = canonical_task(task)
    for key, values in TABLE3_HYPERPARAMS.items():
        if key.lower() == str(canon).lower():
            return dict(values)
    if str(canon).startswith("Sparse"):
        base = canon.replace("Sparse", "")
        for key, values in TABLE3_HYPERPARAMS.items():
            if key.lower().startswith(base.lower()):
                return dict(values)
    return dict(TABLE3_HYPERPARAMS["Hopper-v3"])


def reference_for(task: str) -> Dict[str, float]:
    """Table 1 reference row for ``task`` (empty dict when unavailable)."""
    canon = canonical_task(task)
    for key, row in TABLE1_REFERENCE.items():
        if key.lower() == str(canon).lower():
            return dict(row)
    if str(canon).startswith("Sparse"):
        return {"no_refine": float("nan"), "ours": float("nan")}
    return {}


def is_sparse(task: str) -> bool:
    canon = canonical_task(task)
    return any(s.lower() == str(canon).lower() for s in SPARSE_TASKS) or str(canon).lower().startswith(
        "sparse"
    )


def resolve_device(device: str) -> str:
    if device and device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover - torch optional
        return "cpu"


def import_first(module_names: Sequence[str]) -> Optional[Any]:
    """Import the first importable dotted module name."""
    import importlib

    for name in module_names:
        try:
            return importlib.import_module(name)
        except Exception:
            continue
    return None


def ensure_dir(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    os.makedirs(path, exist_ok=True)
    return path


def json_default(obj: Any) -> Any:
    """JSON serializer for numpy / torch scalars."""
    try:
        import numpy as np

        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    return str(obj)


# --------------------------------------------------------------------------
# Structured metric extraction (works with RefineResult & friends)
# --------------------------------------------------------------------------
def _get(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if obj is None:
            continue
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            try:
                value = getattr(obj, name)
            except Exception:
                continue
            if callable(value):
                try:
                    value = value()
                except TypeError:
                    continue
                except Exception:
                    continue
            if value is not None:
                return value
    return default


def _scalar(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return default
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            if value.size == 0:
                return default
            return float(np.asarray(value).reshape(-1)[-1])
        if isinstance(value, (list, tuple)):
            return _scalar(value[-1] if len(value) else None, default)
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return default


def _curve(result: Any, window: int = 1) -> Tuple[List[float], List[float]]:
    """Return (steps, rewards) for a refining result, if available."""
    curve = _get(result, "refining_curve", "mean_curve")
    if curve is None:
        return [], []
    if callable(curve):
        try:
            curve = curve(window) if window and window > 1 else curve()
        except Exception:
            try:
                curve = curve()
            except Exception:
                return [], []
    rewards = _get(curve, "rewards", "values", "rewards_list")
    steps = _get(curve, "steps", "steps_list")
    if rewards is None and isinstance(curve, (list, tuple)):
        rewards = list(curve)
    if rewards is None:
        return [], []
    rewards = [float(x) for x in list(rewards)]
    if steps is None:
        steps = list(range(1, len(rewards) + 1))
    return [int(s) for s in list(steps)], rewards


def summarize_result(result: Any, task: str, method: str, explanation: str) -> Dict[str, Any]:
    """Flatten a refining result into a JSON-friendly record."""
    reference = reference_for(task)
    final_reward = _scalar(_get(
        result, "final_reward", "final_eval_reward", "mean_episode_return"
    ))
    baseline_reward = _scalar(_get(
        result, "baseline_reward", "baseline_final_reward", "baseline_env_reward"
    ))
    env_steps = _get(result, "env_steps", "total_env_steps", default=None)
    seconds = _scalar(_get(result, "seconds", "wall_clock", default=float("nan")))
    iterations = _get(result, "iterations", default=None)
    if isinstance(iterations, (list, tuple)):
        iterations = len(iterations)
    steps, rewards = _curve(result)
    improvement = final_reward - baseline_reward
    record = {
        "task": canonical_task(task),
        "method": method,
        "explanation": explanation,
        "final_reward": final_reward,
        "baseline_reward": baseline_reward,
        "improvement": improvement,
        "env_steps": env_steps,
        "iterations": iterations,
        "seconds": seconds,
        "curve_steps": steps,
        "curve_rewards": rewards,
        "reference": reference,
        "reference_improvement": (
            reference.get("ours", float("nan")) - reference.get("no_refine", float("nan"))
            if reference else float("nan")
        ),
    }
    return record


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------
class RefineRunner:
    """Drives RICE refining (and baselines) for a single task."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.task = canonical_task(args.task)
        self.device = resolve_device(args.device)
        self.hyper = task_hyperparams(self.task)
        self.hyper.update({k: v for k, v in
                           (("p", args.p), ("lambda", args.lam), ("alpha", args.alpha))
                           if v is not None})
        self.output_dir = ensure_dir(args.output_dir)
        self.seeds: List[int] = list(args.seeds) if args.seeds else [0]
        self.results: List[Dict[str, Any]] = []
        self.notes: List[str] = []

    # -- setup ------------------------------------------------------------
    def build_env(self, seed: Optional[int] = None):
        """Instantiate the task environment through the RICE registry."""
        env = None
        if self.args.env_kwargs:
            try:
                extra = json.loads(self.args.env_kwargs)
            except Exception:
                extra = {}
        else:
            extra = {}

        # 1) environment registry
        envmod = import_first(("rice.environments", "rice.rice.environments", "environments"))
        if envmod is not None:
            try:
                env = envmod.make_env(self.task, seed=seed, **extra)
            except Exception as exc:
                self.notes.append(f"registry make_env failed for {self.task}: {exc}")
        # 2) direct module fallbacks
        if env is None:
            module_name = None
            if is_sparse(self.task):
                module_name = "mujoco_sparse"
            elif self.task in ("Hopper-v3", "Walker2d-v3", "Reacher-v2", "HalfCheetah-v3"):
                module_name = "mujoco_dense"
            elif self.task == "SelfishMining":
                module_name = "selfish_mining"
            elif self.task == "CageChallenge2":
                module_name = "cage_challenge2"
            elif self.task == "Macro-v1":
                module_name = "autodriving"
            if module_name:
                mod = import_first((
                    f"rice.environments.{module_name}",
                    f"rice.rice.environments.{module_name}",
                    f"environments.{module_name}",
                ))
                if mod is not None:
                    try:
                        env = mod.make_env(self.task, seed=seed, **extra)
                    except Exception as exc:
                        self.notes.append(f"{module_name}.make_env failed: {exc}")
        if env is None:
            raise RuntimeError(
                f"could not build environment for task={self.task!r}; "
                "install the simulator or pass --env-kwargs"
            )
        return env

    def build_policy(self, env):
        """Build the target (warm-start) policy matching the env architecture."""
        from rice.algorithms.ppo import ActorCritic  # type: ignore

        net_arch = tuple(self.args.net_arch) if self.args.net_arch else None
        if net_arch is None:
            envmod = import_first(("rice.environments", "rice.rice.environments", "environments"))
            try:
                net_arch = envmod.default_net_arch(self.task) if envmod else (64, 64)
            except Exception:
                net_arch = (64, 64)
        try:
            policy = ActorCritic(env.observation_space, env.action_space,
                                 net_arch=net_arch, device=self.device)
        except TypeError:
            policy = ActorCritic(env.observation_space, env.action_space, net_arch=net_arch)
        # Warm start from a checkpoint when provided.
        weights = self.args.policy_weights or self.args.weights
        if weights:
            try:
                from rice.algorithms.refine import load_policy_weights  # type: ignore

                load_policy_weights(policy, weights, strict=False)
                self.notes.append(f"warm-started policy from {weights}")
            except Exception as exc:
                self.notes.append(f"failed to load policy weights {weights}: {exc}")
        return policy

    def build_mask_network(self, env, policy):
        """Load / build the mask network (explanation) used by RICE."""
        from rice.algorithms.mask_network import MaskNetwork  # type: ignore

        net_arch = tuple(self.args.mask_net_arch) if self.args.mask_net_arch else None
        try:
            mask = MaskNetwork(env.observation_space, net_arch=net_arch, device=self.device)
        except TypeError:
            mask = MaskNetwork(env.observation_space)
        if self.args.mask_weights:
            try:
                import torch  # type: ignore

                state = torch.load(self.args.mask_weights, map_location=self.device)
                if isinstance(state, dict) and "model_state_dict" in state:
                    state = state["model_state_dict"]
                missing, unexpected = mask.load_state_dict(state, strict=False)
                self.notes.append(
                    f"loaded mask weights from {self.args.mask_weights} "
                    f"(missing={len(missing)}, unexpected={len(unexpected)})"
                )
            except Exception as exc:
                self.notes.append(f"failed to load mask weights {self.args.mask_weights}: {exc}")
        return mask

    def build_evaluation_env(self, seed: Optional[int] = None):
        if self.args.no_eval_env:
            return None
        try:
            return self.build_env(seed=None if seed is None else seed + 10_000)
        except Exception as exc:
            self.notes.append(f"no evaluation env: {exc}")
            return None

    # -- running ----------------------------------------------------------
    def run_method(self, method: str, explanation: str, seed: int) -> Dict[str, Any]:
        """Run one (method, explanation, seed) configuration."""
        method = canonical_method(method)
        explanation = canonical_explanation(explanation)
        env = self.build_env(seed=seed)
        eval_env = self.build_evaluation_env(seed=seed)
        policy = self.build_policy(env)
        mask = self.build_mask_network(env, policy)

        base = _get(policy, "state_dict", default=None)
        if callable(base):
            try:
                base = base()
            except Exception:
                base = None

        kwargs: Dict[str, Any] = {
            "task": self.task,
            "method": method,
            "explanation": explanation,
            "p": self.hyper.get("p"),
            "lam": self.hyper.get("lambda"),
            "alpha": self.hyper.get("alpha"),
            "n_iterations": self.args.iterations,
            "steps_per_iter": self.args.steps_per_iter,
            "total_env_steps": self.args.env_steps,
            "n_seeds": 1,
            "seeds": [seed],
            "eval_episodes": self.args.eval_episodes,
            "eval_every": self.args.eval_every,
            "device": self.device,
            "verbose": self.args.verbose,
            "log_every": self.args.log_every,
        }
        if self.args.rollin_length is not None:
            kwargs["rollin_length"] = self.args.rollin_length

        started = time.time()
        result = None

        # "no_refine" is a pure evaluation of the warm-start policy.
        if method == "no_refine":
            from rice.algorithms.refine import evaluate_policy  # type: ignore

            stats = evaluate_policy(eval_env or env, policy,
                                    n_episodes=self.args.eval_episodes, seed=seed)
            result = {
                "final_reward": stats.get("mean_return", float("nan")),
                "baseline_reward": stats.get("mean_return", float("nan")),
                "seconds": time.time() - started,
                "env_steps": int(stats.get("mean_length", 0) or 0) * self.args.eval_episodes,
            }
            return summarize_result(result, self.task, method, explanation)

        # Baselines that ship as dedicated modules.
        if method in ("ppo", "statemask_r", "jsrl", "sil", "sac_gail"):
            if method == "sac_gail" and not self.args.enable_sac:
                raise RuntimeError(
                    "SAC+GAIL (Experiment IV) is expensive: pass --enable-sac "
                    "(and set RICE_ENABLE_SAC=1) to run it"
                )
            if method == "sac_gail":
                os.environ.setdefault("RICE_ENABLE_SAC", "1")
            baseline_mod = import_first((
                f"rice.baselines.{method}",
                f"rice.rice.baselines.{method}",
            ))
            if baseline_mod is not None:
                factory = getattr(baseline_mod, f"make_{method}_refiner", None) or getattr(
                    baseline_mod, f"make_{method}_refiner", None
                )
                try:
                    classes = {
                        "ppo": "PPOFineTuner",
                        "statemask_r": "StateMaskRRefiner",
                        "jsrl": "JSRLRefiner",
                        "sil": "SILRefiner",
                        "sac_gail": "SACGAILPipeline",
                    }
                    cls = getattr(baseline_mod, classes[method], None)
                    if cls is None:
                        raise AttributeError(f"{classes[method]} not found in {baseline_mod}")
                    runner = cls(env=env, policy=policy, mask_network=mask,
                                 evaluation_env=eval_env, task=self.task, **kwargs)
                    if hasattr(runner, "refine"):
                        result = runner.refine(seed=seed, n_iterations=self.args.iterations)
                    elif hasattr(runner, "run"):
                        result = runner.run(seeds=[seed])
                    else:  # pragma: no cover
                        raise AttributeError("baseline runner has no refine/run method")
                except Exception as exc:
                    self.notes.append(f"{method} baseline failed ({exc}); using fallback")
                    result = None

            if result is None:
                # Fallback: emulate the baseline with the shared refining loop.
                from rice.algorithms.refine import refine_policy  # type: ignore

                if method == "ppo":
                    kwargs.update({"p": 0.0, "lam": 0.0})
                elif method == "statemask_r":
                    kwargs.update({"p": 1.0, "lam": 0.0})
                elif method == "jsrl":
                    # JSRL curriculum approximated by always-critical roll-in.
                    kwargs.update({"p": 1.0, "lam": 0.0})
                elif method == "sil":
                    kwargs.update({"p": 0.0, "lam": 0.0})
                result = refine_policy(env=env, policy=policy, mask_network=mask,
                                       **kwargs)

            record = summarize_result(result, self.task, method, explanation)
            record["seconds"] = (record.get("seconds") or 0.0) or (time.time() - started)
            return record

        # Default: RICE (Algorithm 2).
        evaluator_mod = import_first((
            "rice.evaluation.refining_eval",
            "rice.rice.evaluation.refining_eval",
        ))
        result = None
        if evaluator_mod is not None and hasattr(evaluator_mod, "evaluate_refining"):
            try:
                result = evaluator_mod.evaluate_refining(
                    task=self.task, policy=policy, mask_network=mask,
                    method=method, explanation=explanation,
                    env=env, evaluation_env=eval_env, seeds=[seed], **kwargs,
                )
            except Exception as exc:
                self.notes.append(f"evaluate_refining failed ({exc}); direct fallback")
                result = None
        if result is None:
            from rice.algorithms.refine import refine_policy  # type: ignore

            result = refine_policy(env=env, policy=policy, mask_network=mask, **kwargs)

        record = summarize_result(result, self.task, method, explanation)
        record["seconds"] = (record.get("seconds") or 0.0) or (time.time() - started)
        return record

    def run(self, methods: Sequence[str], explanations: Sequence[str]) -> List[Dict[str, Any]]:
        """Run the cross-product of methods x explanations x seeds."""
        from rice.utils.seeding import set_global_seeds  # type: ignore

        total = len(methods) * len(explanations) * len(self.seeds)
        index = 0
        for method in methods:
            for explanation in explanations:
                # RICE honours --explanation; baselines ignore it (fixed explanation).
                for seed in self.seeds:
                    index += 1
                    set_global_seeds(seed)
                    label = f"[{index}/{total}] task={self.task} method={method} " \
                            f"explanation={explanation} seed={seed}"
                    if self.args.verbose:
                        print(label, flush=True)
                    try:
                        record = self.run_method(method, explanation, seed)
                    except Exception as exc:
                        record = {
                            "task": self.task, "method": canonical_method(method),
                            "explanation": canonical_explanation(explanation),
                            "seed": seed, "final_reward": float("nan"),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        self.notes.append(f"{label} failed: {traceback.format_exc(limit=3)}")
                    record["seed"] = seed
                    self.results.append(record)
                    if self.args.verbose:
                        print(
                            f"    -> final_reward={record.get('final_reward')}, "
                            f"baseline={record.get('baseline_reward')}, "
                            f"improvement={record.get('improvement')}",
                            flush=True,
                        )
        return self.results

    # -- reporting --------------------------------------------------------
    def aggregate(self) -> List[Dict[str, Any]]:
        """Aggregate per-seed records into mean/std per (method, explanation)."""
        try:
            import numpy as np
        except Exception:  # pragma: no cover
            np = None

        groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for record in self.results:
            key = (record.get("method"), record.get("explanation"))
            groups.setdefault(key, []).append(record)

        rows: List[Dict[str, Any]] = []
        for (method, explanation), records in groups.items():
            finals = [r.get("final_reward") for r in records
                      if r.get("final_reward") is not None]
            bases = [r.get("baseline_reward") for r in records
                     if r.get("baseline_reward") is not None]
            improvements = [r.get("improvement") for r in records
                            if r.get("improvement") is not None]
            if np is not None:
                def _ms(values):
                    arr = np.asarray([v for v in values if v == v], dtype=float)
                    if arr.size == 0:
                        return float("nan"), float("nan")
                    return float(arr.mean()), float(arr.std())

                fm, fs = _ms(finals)
                bm, bs = _ms(bases)
                im, istd = _ms(improvements)
            else:  # pragma: no cover
                fm = fs = bm = bs = im = istd = float("nan")
            reference = reference_for(self.task)
            rows.append({
                "task": self.task,
                "method": method,
                "explanation": explanation,
                "n_seeds": len(records),
                "final_reward_mean": fm,
                "final_reward_std": fs,
                "baseline_reward_mean": bm,
                "baseline_reward_std": bs,
                "improvement_mean": im,
                "improvement_std": istd,
                "reference_ours": reference.get("ours"),
                "reference_no_refine": reference.get("no_refine"),
                "task_reference": reference,
            })
        return rows

    def trend_check(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """Qualitative success criteria from the reproduction plan."""
        reference = reference_for(self.task)
        by_method = {r["method"]: r for r in rows}
        verdict: Dict[str, Any] = {"task": self.task, "reference": reference,
                                   "checks": {}, "passed": True, "notes": []}

        def check(name: str, ok: Optional[bool], detail: str = "") -> None:
            verdict["checks"][name] = bool(ok) if ok is not None else None
            if ok is False:
                verdict["notes"].append(detail or name)
            if ok is None:
                verdict["notes"].append(f"{name}: insufficient data")

        ours = by_method.get("ours")
        no_refine = by_method.get("no_refine")
        if ours and no_refine and ours["final_reward_mean"] == ours["final_reward_mean"]:
            check(
                "ours_improves_over_no_refine",
                ours["final_reward_mean"] > no_refine["final_reward_mean"],
                f"ours={ours['final_reward_mean']:.3f} <= "
                f"no_refine={no_refine['final_reward_mean']:.3f}",
            )
        if ours:
            for other in ("ppo", "statemask_r", "jsrl", "sil"):
                row = by_method.get(other)
                if row is None or row["final_reward_mean"] != row["final_reward_mean"]:
                    continue
                check(
                    f"ours_better_than_{other}",
                    ours["final_reward_mean"] >= row["final_reward_mean"],
                    f"ours={ours['final_reward_mean']:.3f} < {other}="
                    f"{row['final_reward_mean']:.3f}",
                )
        if ours and "improvement_mean" in ours and ours["improvement_mean"] == ours["improvement_mean"]:
            ref_impr = (reference.get("ours", float("nan"))
                        - reference.get("no_refine", float("nan")))
            if ref_impr == ref_impr:
                check(
                    "improvement_direction_matches_paper",
                    (ours["improvement_mean"] > 0) == (ref_impr > 0),
                    f"measured improvement={ours['improvement_mean']:.3f}, "
                    f"paper={ref_impr:.3f}",
                )
        verdict["passed"] = all(v for v in verdict["checks"].values() if v is not None) \
            if any(v is not None for v in verdict["checks"].values()) else None
        return verdict

    def save(self, rows: Sequence[Dict[str, Any]], verdict: Optional[Dict[str, Any]] = None) -> None:
        if not self.output_dir:
            return
        tag = canonical_task(self.task)
        with open(os.path.join(self.output_dir, f"refine_{tag}.json"), "w") as handle:
            json.dump(
                {
                    "task": tag,
                    "config": vars(self.args) if hasattr(self.args, "__dict__") else {},
                    "hyperparameters": self.hyper,
                    "records": list(self.results),
                    "aggregate": list(rows),
                    "trend_check": verdict or {},
                    "notes": self.notes,
                },
                handle, indent=2, default=json_default,
            )
        csv_path = os.path.join(self.output_dir, f"refine_{tag}.csv")
        fields = ["task", "method", "explanation", "n_seeds", "final_reward_mean",
                  "final_reward_std", "baseline_reward_mean", "baseline_reward_std",
                  "improvement_mean", "improvement_std", "reference_ours",
                  "reference_no_refine"]
        with open(csv_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k) for k in fields})
        # Refining curves (Figure 2 / Figure 3) for sparse tasks.
        curve_path = os.path.join(self.output_dir, f"curves_{tag}.json")
        with open(curve_path, "w") as handle:
            json.dump(
                {
                    r.get("method"): {
                        "seed": r.get("seed"),
                        "steps": r.get("curve_steps", []),
                        "rewards": r.get("curve_rewards", []),
                    }
                    for r in self.results if r.get("curve_rewards")
                },
                handle, indent=2, default=json_default,
            )
        if self.notes:
            with open(os.path.join(self.output_dir, f"notes_{tag}.txt"), "w") as handle:
                handle.write("\n".join(self.notes))

    def plot(self, rows: Sequence[Dict[str, Any]]) -> None:
        if not self.args.plot or not self.output_dir:
            return
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except Exception as exc:
            self.notes.append(f"matplotlib unavailable ({exc}); skipping plots")
            return

        # Refining curves per method.
        groups: Dict[str, List[Tuple[List[int], List[float]]]] = {}
        for record in self.results:
            if record.get("curve_rewards"):
                groups.setdefault(str(record.get("method")), []).append(
                    (record.get("curve_steps", []), record["curve_rewards"])
                )
        if groups:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for method, curves in sorted(groups.items()):
                max_len = max(len(c[1]) for c in curves)
                stacked = np.full((len(curves), max_len), np.nan)
                for i, (_, rewards) in enumerate(curves):
                    stacked[i, :len(rewards)] = rewards
                mean = np.nanmean(stacked, axis=0)
                std = np.nanstd(stacked, axis=0)
                x = np.arange(1, max_len + 1)
                ax.plot(x, mean, label=method)
                ax.fill_between(x, mean - std, mean + std, alpha=0.2)
            ax.set_xlabel("outer iteration")
            ax.set_ylabel("episode return")
            ax.set_title(f"Refining curves -- {self.task}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(self.output_dir, f"curves_{self.task}.png"), dpi=150)
            plt.close(fig)

        # Final reward bar chart.
        if rows:
            fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(rows)), 4.0))
            labels = [f"{r['method']}\n({r['explanation']})" for r in rows]
            means = [r.get("final_reward_mean", float("nan")) for r in rows]
            stds = [r.get("final_reward_std", 0.0) or 0.0 for r in rows]
            ax.bar(range(len(rows)), means, yerr=stds, capsize=4)
            ax.set_xticks(range(len(rows)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("final reward")
            ax.set_title(f"{self.task} -- final reward after refining")
            ref = reference_for(self.task).get("no_refine")
            if ref is not None and ref == ref:
                ax.axhline(ref, color="grey", ls="--", lw=1,
                           label="Table 1 no_refine (trend)")
                ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(self.output_dir, f"final_reward_{self.task}.png"), dpi=150)
            plt.close(fig)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RICE Algorithm 2 -- refining a pre-trained DRL agent "
                    "(Experiments II/III/IV).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", "--env", dest="task", default="Hopper-v3",
                        help="environment name (Hopper-v3, Walker2d-v3, Reacher-v2, "
                             "HalfCheetah-v3, SelfishMining, CageChallenge2, Macro-v1, "
                             "SparseHopper, SparseHalfCheetah)")
    parser.add_argument("--method", "--methods", dest="methods", nargs="+",
                        default=["ours"], help=f"one or more of {METHODS}")
    parser.add_argument("--explanation", "--explanations", dest="explanations",
                        nargs="+", default=["ours"],
                        help=f"one or more of {EXPLANATIONS}")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0],
                        help="random seeds (paper reports mean/std over 3 seeds)")
    parser.add_argument("--iterations", type=int, default=100,
                        help="number of outer refining iterations")
    parser.add_argument("--steps-per-iter", "--steps_per_iter", dest="steps_per_iter",
                        type=int, default=None,
                        help="roll-out steps T per outer iteration (default: episode length)")
    parser.add_argument("--env-steps", "--total-env-steps", dest="env_steps", type=int,
                        default=None, help="optional total environment-step budget")
    parser.add_argument("--rollin-length", "--rollin_length", dest="rollin_length",
                        type=int, default=None,
                        help="K, the roll-in trajectory length (default: one full episode)")
    parser.add_argument("--p", type=float, default=None,
                        help="mixed-init probability (Table 3 default per task)")
    parser.add_argument("--lambda", "--lam", dest="lam", type=float, default=None,
                        help="RND intrinsic-reward coefficient (Table 3 default per task)")
    parser.add_argument("--alpha", type=float, default=None,
                        help="mask-network blinding bonus (Table 3 default per task)")
    parser.add_argument("--eval-episodes", "--eval_episodes", dest="eval_episodes",
                        type=int, default=5)
    parser.add_argument("--eval-every", "--eval_every", dest="eval_every", type=int,
                        default=0, help="0 disables intermediate evaluation")
    parser.add_argument("--policy-weights", "--weights", dest="weights", default=None,
                        help="warm-start checkpoint (SB3 model .zip, state dict .pt/.pth)")
    parser.add_argument("--mask-weights", dest="mask_weights", default=None,
                        help="trained mask-network checkpoint")
    parser.add_argument("--net-arch", "--net_arch", dest="net_arch", type=int, nargs="+",
                        default=None, help="target policy MLP width (overrides per-task default)")
    parser.add_argument("--mask-net-arch", "--mask_net_arch", dest="mask_net_arch",
                        type=int, nargs="+", default=None)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0")
    parser.add_argument("--output-dir", "--out", dest="output_dir", default="results/refine")
    parser.add_argument("--plot", action="store_true", help="write refining/final-reward plots")
    parser.add_argument("--curves", action="store_true",
                        help="record refining curves (default on for sparse tasks)")
    parser.add_argument("--no-eval-env", "--no_eval_env", dest="no_eval_env",
                        action="store_true", help="reuse the training env for evaluation")
    parser.add_argument("--enable-sac", "--enable_sac", dest="enable_sac", action="store_true",
                        help="allow the expensive SAC+GAIL Experiment IV pipeline")
    parser.add_argument("--env-kwargs", "--env_kwargs", dest="env_kwargs", default=None,
                        help="JSON dict of keyword arguments forwarded to make_env")
    parser.add_argument("--verbose", "-v", action="count", default=0)
    parser.add_argument("--log-every", "--log_every", dest="log_every", type=int, default=1)
    parser.add_argument("--json", action="store_true", help="print the aggregate JSON to stdout")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.curves:
        pass  # curves are always recorded when the refiner exposes them
    if is_sparse(args.task) and not args.plot:
        # Figure 2/3 ask for curves on sparse tasks.
        args.plot = True

    runner = RefineRunner(args)
    print(
        f"RICE refining | task={runner.task} | device={runner.device} | "
        f"p={runner.hyper.get('p')} lambda={runner.hyper.get('lambda')} "
        f"alpha={runner.hyper.get('alpha')} | seeds={runner.seeds}",
        flush=True,
    )

    methods = [canonical_method(m) for m in args.methods]
    explanations = [canonical_explanation(e) for e in args.explanations]
    # Baselines do not consume an explanation: collapse to a single run.
    if methods and all(m != "ours" for m in methods):
        explanations = explanations[:1]

    runner.run(methods, explanations)
    rows = runner.aggregate()
    verdict = runner.trend_check(rows)

    # Console report
    header = f"{'method':<14}{'expl':<22}{'seeds':>6}{'final':>12}{'baseline':>12}{'improve':>12}"
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{str(row['method']):<14}{str(row['explanation']):<22}"
            f"{row['n_seeds']:>6}"
            f"{row.get('final_reward_mean', float('nan')):>12.3f}"
            f"{row.get('baseline_reward_mean', float('nan')):>12.3f}"
            f"{row.get('improvement_mean', float('nan')):>12.3f}"
        )
    reference = reference_for(runner.task)
    if reference:
        print(f"\nTable 1 reference (trend check): {reference}")
    print(f"\nTrend check: passed={verdict.get('passed')}")
    for name, ok in verdict.get("checks", {}).items():
        print(f"  - {name}: {ok}")
    for note in verdict.get("notes", []) + runner.notes:
        print(f"  ! {note}")

    runner.save(rows, verdict)
    runner.plot(rows)
    if args.output_dir:
        print(f"\nWrote results to {os.path.abspath(args.output_dir)}")
    if args.json:
        print(json.dumps({"aggregate": rows, "trend_check": verdict},
                         indent=2, default=json_default))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
