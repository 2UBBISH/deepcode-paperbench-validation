#!/usr/bin/env python
"""RICE: A Refining Scheme for Reinforcement Learning with Explanation (ICML 2024).

Top-level command-line dispatcher for the whole reproduction.

The package implements the two-stage RICE pipeline:

* **Stage 1 -- Explanation** (Algorithm 1): a mask network (a simplified
  StateMask) is trained with *vanilla* PPO plus a blinding bonus ``alpha *
  a_t^m`` so that it assigns a step-level importance score
  ``I(s_t) = P(a_t^m = 0 | s_t)`` (probability of "keep") to each visited state.
* **Stage 2 -- Refining** (Algorithm 2): the pre-trained (bottlenecked) policy
  ``pi`` is refined with PPO on the mixed initial state distribution
  ``mu(s) = beta * d_rho^pihat(s) + (1 - beta) * rho(s)`` (realised by resetting
  to a mask-identified critical state with probability ``p``) and with the
  Random Network Distillation bonus ``lambda * ||f(s_{t+1}) - fhat(s_{t+1})||^2``
  added to the task reward.

This module only *dispatches*: every algorithm lives in ``rice/`` and every
experiment driver lives in ``experiments/``.  See ``README.md`` for the full
reproduction recipe and ``configs/*.yaml`` for hyper-parameters.

Examples
--------
::

    python main.py list-envs
    python main.py show-config --config hopper
    python main.py train-target --env hopper --timesteps 1000000
    python main.py train-mask   --env hopper --config hopper
    python main.py fidelity     --env hopper --K 0.1 0.2 0.3 0.4
    python main.py refine       --env hopper --method ours
    python main.py experiment   --name exp1 --env hopper
    python main.py experiment   --name exp2 --envs hopper halfcheetah
    python main.py experiment   --name exp3 --env walker2d
    python main.py experiment   --name exp4 --env hopper
    python main.py experiment   --name exp5 --env hopper --sweeps p lambda alpha
    python main.py ablation     --ablation all
    python main.py plot
    python main.py run-all      --envs hopper walker2d  # smoke pipeline
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__version__ = "0.1.0"

# --------------------------------------------------------------------------- #
# Paper constants (Tables 2-4, Sections 4.1-4.2, Appendix C.2)
# --------------------------------------------------------------------------- #

#: Applications in scope for the reproduction (Malware Mutation excluded).
APPLICATIONS: Tuple[str, ...] = (
    "hopper",
    "walker2d",
    "reacher",
    "halfcheetah",
    "selfish_mining",
    "cage2",
    "autodriving",
)

DENSE_ENVS: Tuple[str, ...] = ("hopper", "walker2d", "reacher", "halfcheetah")
MUJOCO_ENVS: Tuple[str, ...] = DENSE_ENVS
SPARSE_ENVS: Tuple[str, ...] = ("sparse_hopper", "sparse_halfcheetah")
APP_ENVS: Tuple[str, ...] = ("selfish_mining", "cage2", "autodriving")

#: Table 3 -- hyper-parameter choices used in Experiments I-V.
#: p controls the mixed initial state distribution, lambda the exploration
#: bonus and alpha the mask ratio (blinding bonus).  alpha is insensitive and
#: is pinned to the Table-3 value ``0.0001``.
TABLE3: Dict[str, Dict[str, float]] = {
    "hopper": {"p": 0.25, "lambda": 0.001, "alpha": 0.0001},
    "walker2d": {"p": 0.25, "lambda": 0.01, "alpha": 0.0001},
    "reacher": {"p": 0.50, "lambda": 0.001, "alpha": 0.0001},
    "halfcheetah": {"p": 0.50, "lambda": 0.01, "alpha": 0.0001},
    "selfish_mining": {"p": 0.25, "lambda": 0.001, "alpha": 0.0001},
    "cage2": {"p": 0.50, "lambda": 0.01, "alpha": 0.0001},
    "autodriving": {"p": 0.25, "lambda": 0.01, "alpha": 0.0001},
    # Malware Mutation is out of scope but kept for completeness of the table.
    "malware_mutation": {"p": 0.50, "lambda": 0.01, "alpha": 0.0001},
}

#: Table 4 -- fixed mask-network training sample budgets (Experiment I).
TABLE4_SAMPLES: Dict[str, int] = {
    "hopper": 300_000,
    "walker2d": 300_000,
    "reacher": 300_000,
    "halfcheetah": 300_000,
    "sparse_hopper": 300_000,
    "sparse_halfcheetah": 300_000,
    "selfish_mining": 1_500_000,
    "cage2": 10_000_000,
    "autodriving": 2_443_260,
    "malware_mutation": 32_349,
}

#: Table 4 -- reference mask-training wall-clock times (seconds), RICE vs StateMask.
TABLE4_TIMES: Dict[str, Dict[str, float]] = {
    "hopper": {"ours": 12426.0, "statemask": 15393.0},
    "halfcheetah": {"ours": 1317.0, "statemask": 1579.0},
    "cage2": {"ours": 65400.0, "statemask": 79382.0},
}

#: Paper-reported relative time reduction of RICE's mask trainer over StateMask.
PAPER_TIME_REDUCTION = 0.168  # 16.8 %

#: Experiment V sweeps (Section 4.2).
P_VALUES: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
LAMBDA_VALUES: Tuple[float, ...] = (0.0, 0.1, 0.01, 0.001)
ALPHA_VALUES: Tuple[float, ...] = (0.01, 0.001, 0.0001)
K_VALUES: Tuple[float, ...] = (0.10, 0.20, 0.30, 0.40)

DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2)

#: Explanation / refining method names used by the experiment drivers.
EXPLANATION_METHODS: Tuple[str, ...] = (
    "random",
    "statemask",
    "ours",
    "integrated_gradients",
    "airs",
)
CORE_EXPLANATION_METHODS: Tuple[str, ...] = ("random", "statemask", "ours")
REFINING_METHODS: Tuple[str, ...] = (
    "ours",
    "ppo_finetune",
    "statemask_r",
    "jsrl",
    "sac_finetune",
)

#: Experiment IV is Hopper-only (pre-train a SAC agent, GAIL-approximate it).
EXPERIMENT4_ENVS: Tuple[str, ...] = ("hopper",)

#: Malware Mutation is explicitly out of scope (see the reproduction plan).
OUT_OF_SCOPE: Tuple[str, ...] = ("malware_mutation",)

DEFAULT_OUT_DIR = "results"
DEFAULT_CONFIG = "default"
DEFAULT_DEVICE = "cpu"


# --------------------------------------------------------------------------- #
# Lazy / defensive imports
# --------------------------------------------------------------------------- #
def _try_import(module: str) -> Optional[Any]:
    """Import ``module`` returning ``None`` instead of raising."""
    try:
        return importlib.import_module(module)
    except Exception:  # pragma: no cover - defensive
        return None


def _lazy(module: str, *attrs: str) -> Tuple[Optional[Any], Dict[str, Any]]:
    """Best-effort attribute fetch from a module path.

    Returns ``(module, {attr: value})`` where missing attributes map to
    ``None``.  Used so that the CLI stays usable even when optional
    dependencies (torch, SB3, MuJoCo, DI-drive, ...) are absent.
    """
    mod = _try_import(module)
    found: Dict[str, Any] = {}
    for attr in attrs:
        found[attr] = getattr(mod, attr, None) if mod is not None else None
    return mod, found


# --------------------------------------------------------------------------- #
# Environment / config helpers
# --------------------------------------------------------------------------- #
def normalize_env_id(env_id: str) -> str:
    """Canonicalise an environment identifier to a RICE application key."""
    if not env_id:
        return "default"
    name = str(env_id).strip().lower()
    if name.endswith(".yaml"):
        name = name[:-5]
    name = name.replace("-v0", "").replace("-v1", "").replace("-v2", "")
    name = name.replace("-v3", "").replace("-v4", "").replace("-v5", "")
    name = name.replace("sparse-", "sparse_").replace("-", "_").replace(" ", "_")
    aliases = {
        "selfish": "selfish_mining",
        "selfishmining": "selfish_mining",
        "cage": "cage2",
        "cage_2": "cage2",
        "cage_challenge_2": "cage2",
        "auto": "autodriving",
        "autodrive": "autodriving",
        "auto_driving": "autodriving",
        "metadrive": "autodriving",
        "meta_drive": "autodriving",
        "sparsehopper": "sparse_hopper",
        "sparsehalfcheetah": "sparse_halfcheetah",
        "half_cheetah": "halfcheetah",
        "walker": "walker2d",
        "walker_2d": "walker2d",
    }
    return aliases.get(name, name)


def load_config(name: str = DEFAULT_CONFIG, override: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Load a YAML config (merged over ``configs/default.yaml``).

    ``override`` accepts a list of ``key=value`` strings for quick CLI tweaks
    (values are parsed as YAML scalars when possible).
    """
    cfg: Dict[str, Any] = {}
    try:
        from rice.utils.io import get_config  # type: ignore
    except Exception:
        get_config = None  # type: ignore

    if get_config is not None:
        try:
            cfg = dict(get_config(name) or {})
        except Exception:
            cfg = {}
    if not cfg:
        cfg = _read_yaml_fallback(name)
    if override:
        cfg = _apply_overrides(cfg, override)
    return cfg


def _read_yaml_fallback(name: str) -> Dict[str, Any]:
    """Minimal YAML reader used when ``rice.utils.io`` is unavailable."""
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    root = os.path.dirname(os.path.abspath(__file__))
    path = name
    if not os.path.isabs(path):
        for stem in (name, f"{name}.yaml", f"{name}.yml"):
            candidate = os.path.join(root, "configs", stem)
            if os.path.exists(candidate):
                path = candidate
                break
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return dict(yaml.safe_load(handle) or {})
    except Exception:
        return {}


def _coerce_value(raw: str) -> Any:
    """Parse a CLI override value into a Python scalar/collection."""
    try:
        import yaml  # type: ignore

        return yaml.safe_load(raw)
    except Exception:
        lowered = raw.lower()
        if lowered in ("true", "false"):
            return lowered == "true"
        if lowered in ("none", "null"):
            return None
        try:
            return int(raw)
        except ValueError:
            pass
        try:
            return float(raw)
        except ValueError:
            return raw


def _apply_overrides(cfg: Dict[str, Any], overrides: Sequence[str]) -> Dict[str, Any]:
    """Apply dotted ``key=value`` overrides onto a nested config dict."""
    for item in overrides:
        if "=" not in item:
            continue
        key, raw = item.split("=", 1)
        parts = [p for p in key.strip().split(".") if p]
        if not parts:
            continue
        node = cfg
        for part in parts[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                node[part] = nxt
            node = nxt
        node[parts[-1]] = _coerce_value(raw.strip())
    return cfg


def apply_table3(cfg: Dict[str, Any], env_id: str) -> Dict[str, Any]:
    """Fill ``p``/``lambda``/``alpha`` from Table 3 unless already specified."""
    key = normalize_env_id(env_id)
    row = TABLE3.get(key)
    if not row:
        return cfg
    refine = cfg.setdefault("refine", {})
    explanation = cfg.setdefault("explanation", {})
    rnd = cfg.setdefault("rnd", {})
    refine.setdefault("p", row["p"])
    refine.setdefault("lam", row["lambda"])
    explanation.setdefault("alpha", row["alpha"])
    rnd.setdefault("lam", row["lambda"])
    return cfg


def table3_for(env_id: str) -> Dict[str, float]:
    """Return the Table-3 hyper-parameters for an application."""
    return dict(TABLE3.get(normalize_env_id(env_id), TABLE3.get("hopper", {})))


def mask_samples_for(env_id: str, cfg: Optional[Dict[str, Any]] = None) -> int:
    """Table-4 mask-net sample budget for an application."""
    key = normalize_env_id(env_id)
    if cfg:
        exp = cfg.get("explanation") or {}
        value = exp.get("total_timesteps")
        if value:
            return int(value)
    return int(TABLE4_SAMPLES.get(key, 300_000))


# --------------------------------------------------------------------------- #
# Dispatcher machinery
# --------------------------------------------------------------------------- #
class Command:
    """Description of one CLI sub-command and how to execute it."""

    __slots__ = ("name", "help", "target", "fn")

    def __init__(self, name: str, help: str, target: str, fn: Optional[Callable] = None):
        self.name = name
        self.help = help
        self.target = target  # "module:attr"
        self.fn = fn

    def resolve(self) -> Optional[Callable]:
        """Resolve the callable, importing lazily."""
        if self.fn is not None:
            return self.fn
        module_path, _, attr = self.target.partition(":")
        mod = _try_import(module_path)
        if mod is None:
            return None
        return getattr(mod, attr, None)


def _command_table() -> Dict[str, Command]:
    """Registry of available CLI sub-commands (lazily resolved)."""
    return {
        "train-target": Command(
            "train-target",
            "Pre-train the frozen target policy pi (scripts/train_target.py).",
            "scripts.train_target:main",
        ),
        "train-targets": Command(
            "train-targets",
            "Pre-train the frozen target policy for several applications.",
            "scripts.train_target:main",
        ),
        "train-mask": Command(
            "train-mask",
            "Train the Stage-1 mask network with Algorithm 1 (scripts/train_mask.py).",
            "scripts.train_mask:main",
        ),
        "fidelity": Command(
            "fidelity",
            "Evaluate the fidelity score (Experiment I metric, scripts/run_fidelity.py).",
            "scripts.run_fidelity:main",
        ),
        "refine": Command(
            "refine",
            "Refine the pre-trained policy with Algorithm 2 (scripts/run_refine.py).",
            "scripts.run_refine:main",
        ),
        "ablation": Command(
            "ablation",
            "Run the ablation studies (scripts/run_ablation.py).",
            "scripts.run_ablation:main",
        ),
        "plot": Command(
            "plot",
            "Render tables/figures from results/*.json (scripts/plot_results.py).",
            "scripts.plot_results:main",
        ),
    }


# --------------------------------------------------------------------------- #
# Built-in commands (no external deps)
# --------------------------------------------------------------------------- #
def list_envs(as_json: bool = False) -> Dict[str, Any]:
    """List the in-scope environments and their metadata."""
    info: Dict[str, Any] = {}
    make_env_mod, _ = _lazy("rice.envs.make_env", "summarize_envs", "available_envs")
    summary = None
    if make_env_mod is not None:
        try:
            summary = make_env_mod.summarize_envs()
        except Exception:
            summary = None
    for key in tuple(APPLICATIONS) + tuple(SPARSE_ENVS):
        entry = dict(summary.get(key, {})) if isinstance(summary, dict) else {}
        entry.setdefault("key", key)
        entry.setdefault("table3", table3_for(key))
        entry.setdefault("mask_samples", mask_samples_for(key))
        info[key] = entry
    if as_json:
        print(json.dumps(info, indent=2, default=str))
    else:
        print("RICE environments (in scope):")
        for key, entry in info.items():
            gym_id = entry.get("gym_id") or entry.get("env_id") or "-"
            backend = entry.get("backend") or entry.get("supported", "-")
            print(f"  {key:<20} gym_id={gym_id:<24} backend={backend}")
        print("\nOut of scope (excluded per the reproduction plan):")
        for key in OUT_OF_SCOPE:
            print(f"  {key}")
    return info


def show_config(config: str = DEFAULT_CONFIG, as_json: bool = False) -> Dict[str, Any]:
    """Print a (merged) configuration."""
    cfg = load_config(config)
    if as_json:
        print(json.dumps(cfg, indent=2, default=str))
    else:
        print(f"# configs/{config}.yaml (merged over configs/default.yaml)")
        print(json.dumps(cfg, indent=2, default=str))
    return cfg


def show_table3(env: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    """Print Table 3 (hyper-parameter choices for Experiments I-V)."""
    if env:
        row = table3_for(env)
        print(f"Table 3 -- {env}: p={row.get('p')} lambda={row.get('lambda')} alpha={row.get('alpha')}")
        return {normalize_env_id(env): row}
    header = f"{'env':<18}{'p':>8}{'lambda':>10}{'alpha':>10}"
    print("Table 3 -- hyper-parameter choices (Experiments I-V)")
    print(header)
    print("-" * len(header))
    for key, row in TABLE3.items():
        print(f"{key:<18}{row['p']:>8}{row['lambda']:>10}{row['alpha']:>10}")
    return TABLE3


def show_sweeps() -> Dict[str, Sequence[float]]:
    """Print the Experiment-V sweep grids."""
    sweeps = {"p": P_VALUES, "lambda": LAMBDA_VALUES, "alpha": ALPHA_VALUES}
    print("Experiment V sweeps (Section 4.2)")
    print(f"  p      in {list(P_VALUES)}")
    print(f"  lambda in {list(LAMBDA_VALUES)}")
    print(f"  alpha  in {list(ALPHA_VALUES)}")
    return sweeps


def show_experiments() -> Dict[str, str]:
    """Describe Experiment I-V and map them to their drivers."""
    table = {
        "exp1": "Fidelity + mask-training efficiency (script: experiments/exp1_fidelity_efficiency.py)",
        "exp2": "Refining effectiveness, dense + sparse (script: experiments/exp2_refine_effectiveness.py)",
        "exp3": "Explanation quality effect (script: experiments/exp3_explanation_quality.py)",
        "exp4": "SAC agent + GAIL approximation, Hopper (script: experiments/exp4_sac_agent.py)",
        "exp5": "Hyper-parameter sensitivity p/lambda/alpha (script: experiments/exp5_hyperparams.py)",
    }
    for name, desc in table.items():
        print(f"  {name}: {desc}")
    return table


def check_setup() -> Dict[str, Any]:
    """Report which optional backends / modules are importable."""
    modules = {
        "torch": "torch",
        "stable_baselines3": "stable_baselines3",
        "gym": "gym",
        "numpy": "numpy",
        "matplotlib": "matplotlib",
        "yaml": "yaml",
        "rice.envs.make_env": "rice.envs.make_env",
        "rice.models.policies": "rice.models.policies",
        "rice.explanation.mask_trainer": "rice.explanation.mask_trainer",
        "rice.explanation.fidelity": "rice.explanation.fidelity",
        "rice.refining.ppo_refine": "rice.refining.ppo_refine",
        "rice.baselines.statemask_r": "rice.baselines.statemask_r",
    }
    report: Dict[str, bool] = {}
    print("RICE setup check:")
    for label, module in modules.items():
        ok = _try_import(module) is not None
        report[label] = ok
        print(f"  [{'ok ' if ok else 'MISSING'}] {label}")
    print("\nNote: missing MuJoCo/DI-drive still allows smoke tests via fallback envs.")
    return report


# --------------------------------------------------------------------------- #
# Training / experiment commands
# --------------------------------------------------------------------------- #
def _env_cfg(env: str, config: Optional[str] = None) -> Dict[str, Any]:
    """Load config for an application: explicit config wins, else the env id."""
    cfg = load_config(config or env)
    if not cfg or (config is None and not cfg.get("env")):
        cfg = load_config(env)
    return apply_table3(cfg, env)


def cmd_train_target(args: argparse.Namespace) -> int:
    """Dispatch to ``scripts/train_target.py``."""
    command = _command_table()["train-target"]
    fn = command.resolve()
    if fn is None:
        return _missing("scripts/train_target.py --env %s" % args.env)
    argv = [
        "--env",
        args.env,
        "--timesteps",
        str(args.timesteps),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]
    if args.config:
        argv += ["--config", args.config]
    if getattr(args, "out_dir", None):
        argv += ["--out-dir", args.out_dir]
    return int(fn(argv) or 0)


def _missing(what: str) -> int:
    print(f"[main] requested component is not available: {what}")
    print("[main] install the optional dependencies (see requirements.txt) and retry.")
    return 1


def cmd_train_mask(args: argparse.Namespace) -> int:
    """Dispatch to ``scripts/train_mask.py`` (Algorithm 1)."""
    fn = _command_table()["train-mask"].resolve()
    if fn is None:
        return _missing("scripts/train_mask.py --env %s" % args.env)
    argv = [
        "--env",
        args.env,
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--method",
        args.method,
    ]
    if args.config:
        argv += ["--config", args.config]
    if args.timesteps is not None:
        argv += ["--timesteps", str(args.timesteps)]
    if args.checkpoint:
        argv += ["--checkpoint", args.checkpoint]
    if getattr(args, "out_dir", None):
        argv += ["--out-dir", args.out_dir]
    return int(fn(argv) or 0)


def cmd_fidelity(args: argparse.Namespace) -> int:
    """Dispatch to ``scripts/run_fidelity.py`` (Experiment I metric)."""
    fn = _command_table()["fidelity"].resolve()
    if fn is None:
        return _missing("scripts/run_fidelity.py --env %s" % args.env)
    argv = [
        "--env",
        args.env,
        "--device",
        args.device,
        "--n-trajectories",
        str(args.n_trajectories),
    ]
    if args.config:
        argv += ["--config", args.config]
    if args.K:
        argv += ["--K", *[str(k) for k in args.K]]
    if args.method:
        argv += ["--method", args.method]
    if args.seeds:
        argv += ["--seeds", *[str(s) for s in args.seeds]]
    if getattr(args, "out_dir", None):
        argv += ["--out-dir", args.out_dir]
    return int(fn(argv) or 0)


def cmd_refine(args: argparse.Namespace) -> int:
    """Dispatch to ``scripts/run_refine.py`` (Algorithm 2)."""
    fn = _command_table()["refine"].resolve()
    if fn is None:
        return _missing("scripts/run_refine.py --env %s" % args.env)
    argv = [
        "--env",
        args.env,
        "--method",
        args.method,
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]
    if args.config:
        argv += ["--config", args.config]
    if args.p is not None:
        argv += ["--p", str(args.p)]
    if args.lam is not None:
        argv += ["--lam", str(args.lam)]
    if args.timesteps is not None:
        argv += ["--timesteps", str(args.timesteps)]
    if args.checkpoint:
        argv += ["--checkpoint", args.checkpoint]
    if getattr(args, "out_dir", None):
        argv += ["--out-dir", args.out_dir]
    return int(fn(argv) or 0)


def cmd_experiment(args: argparse.Namespace) -> int:
    """Run one of the five paper experiments via its driver module."""
    mapping = {
        "exp1": ("experiments.exp1_fidelity_efficiency", "run_experiment1"),
        "fidelity": ("experiments.exp1_fidelity_efficiency", "run_experiment1"),
        "exp2": ("experiments.exp2_refine_effectiveness", "run_experiment2_multi"),
        "refining": ("experiments.exp2_refine_effectiveness", "run_experiment2_multi"),
        "exp3": ("experiments.exp3_explanation_quality", "run_experiment3_multi"),
        "explanation": ("experiments.exp3_explanation_quality", "run_experiment3_multi"),
        "exp4": ("experiments.exp4_sac_agent", "run_experiment4"),
        "sac": ("experiments.exp4_sac_agent", "run_experiment4"),
        "exp5": ("experiments.exp5_hyperparams", "run_experiment5_multi"),
        "hyperparams": ("experiments.exp5_hyperparams", "run_experiment5_multi"),
    }
    name = str(args.name).lower()
    if name not in mapping:
        print(f"[main] unknown experiment {args.name!r}; choose from {sorted(set(mapping))}")
        return 2
    module_path, fn_name = mapping[name]
    mod = _try_import(module_path)
    runner = getattr(mod, fn_name, None) if mod is not None else None
    if runner is None:
        return _missing(f"{module_path}:{fn_name}")

    envs = [normalize_env_id(e) for e in (args.envs or [args.env or "hopper"])]
    out_dir = args.out_dir or os.path.join(DEFAULT_OUT_DIR, name)
    kwargs: Dict[str, Any] = {"device": args.device, "out_dir": out_dir, "progress": args.progress}
    if args.seeds:
        kwargs["seeds"] = tuple(args.seeds)
    if args.timesteps is not None:
        kwargs["refine_timesteps"] = args.timesteps

    print(f"[main] running {name} on {envs} (device={args.device}, out={out_dir})")
    started = time.time()
    report: Any = None
    try:
        if name in ("exp1", "fidelity"):
            reports = {}
            for env in envs:
                cfg = _env_cfg(env, args.config)
                reports[env] = runner(env, cfg=cfg, device=args.device, out_dir=out_dir, progress=args.progress)
            report = reports
        elif name in ("exp2", "refining"):
            cfg = _env_cfg(envs[0] if len(envs) == 1 else "default", args.config)
            report = runner(envs, cfg=cfg, device=args.device, out_dir=out_dir, progress=args.progress, **{
                k: v for k, v in kwargs.items() if k not in ("device", "out_dir", "progress")
            })
        else:
            cfg = _env_cfg(envs[0], args.config)
            kwargs["cfg"] = cfg
            report = runner(envs[0], **kwargs) if name in ("exp4", "sac") else runner(envs, **kwargs)
    except TypeError:
        # Drivers have slightly different signatures; retry positionally.
        try:
            report = runner(envs[0], **kwargs)  # type: ignore[arg-type]
        except Exception:
            traceback.print_exc()
            return 1
    except Exception:
        traceback.print_exc()
        return 1

    elapsed = time.time() - started
    print(f"[main] {name} finished in {elapsed:.1f}s")
    _dump_report(report, out_dir, f"{name}_main")
    return 0


def cmd_ablation(args: argparse.Namespace) -> int:
    """Dispatch to ``scripts/run_ablation.py``."""
    fn = _command_table()["ablation"].resolve()
    if fn is None:
        return _missing("scripts/run_ablation.py")
    argv = ["--ablation", args.ablation, "--device", args.device]
    if args.config:
        argv += ["--config", args.config]
    if args.envs:
        argv += ["--envs", *args.envs]
    if args.sweeps:
        argv += ["--sweeps", *args.sweeps]
    if getattr(args, "out_dir", None):
        argv += ["--out-dir", args.out_dir]
    if args.progress:
        argv += ["--verbose"]
    return int(fn(argv) or 0)


def cmd_plot(args: argparse.Namespace) -> int:
    """Dispatch to ``scripts/plot_results.py``."""
    fn = _command_table()["plot"].resolve()
    if fn is None:
        return _missing("scripts/plot_results.py")
    argv: List[str] = []
    if args.results_dir:
        argv += ["--results-dir", args.results_dir]
    if args.out_dir:
        argv += ["--out-dir", args.out_dir]
    if args.formats:
        argv += ["--formats", *args.formats]
    if args.no_plots:
        argv += ["--no-plots"]
    return int(fn(argv) or 0)


def cmd_run_all(args: argparse.Namespace) -> int:
    """Run a lightweight end-to-end pipeline for one or more applications.

    For each env: (optionally) pre-train the target policy, train the Stage-1
    mask explanation (Algorithm 1), evaluate fidelity, and refine with
    Algorithm 2.  Intended as a smoke test / sanity pipeline.
    """
    envs = [normalize_env_id(e) for e in (args.envs or list(APPLICATIONS))]
    out_dir = args.out_dir or os.path.join(DEFAULT_OUT_DIR, "run_all")
    print(f"[main] run-all for {envs}; backend check:")
    check_setup()
    rc = 0
    for env in envs:
        cfg = _env_cfg(env, args.config)
        steps = mask_samples_for(env, cfg)
        print(f"\n[main] === {env} (mask budget {steps}) ===")
        ns = argparse.Namespace(
            env=env, config=args.config, timesteps=args.timesteps, seed=args.seed,
            device=args.device, method="ours", checkpoint=None,
            K=list(K_VALUES), n_trajectories=args.n_trajectories, seeds=list(DEFAULT_SEEDS),
            p=None, lam=None, out_dir=out_dir,
        )
        if not args.skip_mask:
            rc |= cmd_train_mask(ns)
        if not args.skip_fidelity:
            rc |= cmd_fidelity(ns)
        if not args.skip_refine:
            rc |= cmd_refine(ns)
    print("\n[main] run-all complete")
    return rc


def _dump_report(report: Any, out_dir: Optional[str], stem: str) -> Optional[str]:
    """Persist an experiment report as JSON when possible."""
    if out_dir is None:
        return None
    try:
        from rice.utils.io import ensure_dir, save_json  # type: ignore
    except Exception:
        try:
            os.makedirs(out_dir, exist_ok=True)

            def save_json(obj: Any, path: str, indent: int = 2) -> str:  # type: ignore
                with open(path, "w", encoding="utf-8") as handle:
                    json.dump(obj, handle, indent=indent, default=str)
                return path

            ensure_dir = lambda p: (os.makedirs(p, exist_ok=True) or p)  # type: ignore
        except Exception:
            return None
    try:
        path = os.path.join(str(out_dir), f"{stem}.json")
        save_json(report, path)
        print(f"[main] wrote {path}")
        return path
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    """Build the top-level RICE CLI parser."""
    parser = argparse.ArgumentParser(
        prog="rice",
        description="RICE: A Refining Scheme for Reinforcement Learning with Explanation (ICML 2024).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"RICE {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # --- informational commands ------------------------------------------- #
    p = sub.add_parser("list-envs", help="list in-scope environments")
    p.add_argument("--json", action="store_true", help="emit JSON")

    p = sub.add_parser("show-config", help="print a merged configuration")
    p.add_argument("--config", default=DEFAULT_CONFIG, help="config name/stem/path")
    p.add_argument("--json", action="store_true", help="emit JSON")

    p = sub.add_parser("show-table3", help="print Table 3 hyper-parameters")
    p.add_argument("--env", default=None, help="single environment (optional)")

    sub.add_parser("show-sweeps", help="print Experiment V sweep grids")
    sub.add_parser("experiments", help="describe Experiments I-V")
    sub.add_parser("check-setup", help="report importable optional backends")

    # --- training / evaluation ------------------------------------------- #
    common = dict(env="--env", config="--config", seed="--seed", device="--device",
                  timesteps="--timesteps", out_dir="--out-dir", checkpoint="--checkpoint")

    p = sub.add_parser("train-target", help="pre-train the frozen target policy pi")
    p.add_argument("--env", default="hopper")
    p.add_argument("--config", default=None)
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=DEFAULT_DEVICE)
    p.add_argument("--out-dir", default=None)

    p = sub.add_parser("train-mask", help="train the Stage-1 mask network (Algorithm 1)")
    p.add_argument("--env", default="hopper")
    p.add_argument("--config", default=None)
    p.add_argument("--method", default="ours", choices=list(EXPLANATION_METHODS))
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=DEFAULT_DEVICE)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--out-dir", default=None)

    p = sub.add_parser("fidelity", help="evaluate the fidelity score (Experiment I)")
    p.add_argument("--env", default="hopper")
    p.add_argument("--config", default=None)
    p.add_argument("--method", default="ours", choices=list(EXPLANATION_METHODS))
    p.add_argument("--K", type=float, nargs="+", default=list(K_VALUES))
    p.add_argument("--n-trajectories", type=int, default=500)
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--device", default=DEFAULT_DEVICE)
    p.add_argument("--out-dir", default=None)

    p = sub.add_parser("refine", help="refine with Algorithm 2 (mixed init + RND)")
    p.add_argument("--env", default="hopper")
    p.add_argument("--config", default=None)
    p.add_argument("--method", default="ours", choices=list(REFINING_METHODS))
    p.add_argument("--p", type=float, default=None, help="mixed-init probability (else Table 3)")
    p.add_argument("--lam", type=float, default=None, help="RND coefficient (else Table 3)")
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=DEFAULT_DEVICE)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--out-dir", default=None)

    # --- experiments / ablations / plots --------------------------------- #
    p = sub.add_parser("experiment", help="run one of the five paper experiments")
    p.add_argument("--name", default="exp1",
                   choices=["exp1", "exp2", "exp3", "exp4", "exp5", "fidelity",
                            "refining", "explanation", "sac", "hyperparams"])
    p.add_argument("--env", default="hopper")
    p.add_argument("--envs", nargs="+", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--device", default=DEFAULT_DEVICE)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--progress", action="store_true")

    p = sub.add_parser("ablation", help="run the RICE ablations")
    p.add_argument("--ablation", default="all",
                   choices=["explanation", "refining", "hyperparams", "sil", "all"])
    p.add_argument("--envs", nargs="+", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--sweeps", nargs="+", default=["p", "lambda", "alpha"])
    p.add_argument("--device", default=DEFAULT_DEVICE)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--progress", action="store_true")

    p = sub.add_parser("plot", help="render tables/figures from results")
    p.add_argument("--results-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--formats", nargs="+", default=["csv", "md"])
    p.add_argument("--no-plots", action="store_true")

    p = sub.add_parser("run-all", help="lightweight end-to-end pipeline per application")
    p.add_argument("--envs", nargs="+", default=list(APPLICATIONS))
    p.add_argument("--config", default=None)
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--n-trajectories", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=DEFAULT_DEVICE)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--skip-mask", action="store_true")
    p.add_argument("--skip-fidelity", action="store_true")
    p.add_argument("--skip-refine", action="store_true")

    return parser


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    """RICE CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.command:
        parser.print_help()
        print("\nTypical workflow:")
        print("  1) python main.py check-setup")
        print("  2) python main.py train-target --env hopper")
        print("  3) python main.py train-mask   --env hopper")
        print("  4) python main.py fidelity     --env hopper")
        print("  5) python main.py refine       --env hopper")
        print("  6) python main.py experiment   --name exp2 --envs hopper halfcheetah")
        print("  7) python main.py plot")
        return 0

    command = args.command
    try:
        if command == "list-envs":
            list_envs(as_json=args.json)
            return 0
        if command == "show-config":
            show_config(config=args.config, as_json=args.json)
            return 0
        if command == "show-table3":
            show_table3(env=args.env)
            return 0
        if command == "show-sweeps":
            show_sweeps()
            return 0
        if command == "experiments":
            show_experiments()
            return 0
        if command == "check-setup":
            check_setup()
            return 0
        if command == "train-target":
            return cmd_train_target(args)
        if command == "train-mask":
            return cmd_train_mask(args)
        if command == "fidelity":
            return cmd_fidelity(args)
        if command == "refine":
            return cmd_refine(args)
        if command == "experiment":
            return cmd_experiment(args)
        if command == "ablation":
            return cmd_ablation(args)
        if command == "plot":
            return cmd_plot(args)
        if command == "run-all":
            return cmd_run_all(args)
    except KeyboardInterrupt:  # pragma: no cover
        print("\n[main] interrupted")
        return 130
    except Exception:  # pragma: no cover - defensive
        traceback.print_exc()
        return 1

    parser.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
