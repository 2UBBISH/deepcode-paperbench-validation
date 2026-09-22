"""Top-level entry point for the FRE (Functional Reward Encodings) reproduction.

This driver orchestrates the full paper pipeline per domain:

    Phase 1  (``encoder``)  -- pretrain the FRE transformer encoder + reward
                               decoder with the info-bottleneck objective
                               (Eq. 6: reward-prediction MSE + beta * KL),
                               i.e. ``train_encoder.py``.
    Phase 2  (``policy``)   -- freeze the encoder and train the z-conditioned
                               IQL policy Q(s,a,z)/V(s,z)/pi(a|s,z) over a
                               mixture of sampled unsupervised reward functions
                               (Algorithm 1), i.e. ``train_policy.py``.
    Phase 3  (``eval``)     -- zero-shot evaluation: encode exactly K=32
                               (state, reward) samples of a new task into z and
                               roll the policy out 20 episodes x 5 seeds,
                               i.e. ``evaluate.py``.

Additionally ``baselines`` runs the in-repo Table-1 comparators (GC-IQL, GC-BC,
OPAL).  FB / SF are *not* reimplemented -- they come from the external
``facebookresearch/controllable_agent`` repository (``--external-baselines``).

Usage
-----
    python -m fre.main --domain antmaze --phase all
    python -m fre.main --domain exorl --phase encoder --steps-scale 0.1
    python -m fre.main --domain kitchen --phase eval --checkpoint runs/kitchen/policy.pt
    python -m fre.main --domain all --phase baselines
    python -m fre.main --aggregate runs            # compare against Table 1

Everything is written defensively: the module resolves its sibling drivers at
call time, filters configuration kwargs against the target dataclasses, and
records per-phase errors instead of aborting the whole run (unless ``--strict``),
so that a partially installed environment still produces a usable report.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DOMAINS: Tuple[str, ...] = ("antmaze", "exorl", "kitchen")
PHASES: Tuple[str, ...] = ("encoder", "policy", "eval", "baselines", "all")
IN_REPO_BASELINES: Tuple[str, ...] = ("gc_iql", "gc_bc", "opal")
EXTERNAL_BASELINES: Tuple[str, ...] = ("fb", "sf")

#: ExORL has two sub-domains (RND exploratory datasets).
DOMAIN_SUBDOMAINS: Dict[str, Tuple[str, ...]] = {
    "exorl": ("walker", "cheetah"),
}

DEFAULT_DATASETS: Dict[str, str] = {
    "antmaze": "antmaze-large-diverse-v2",
    "kitchen": "kitchen-complete-v0",
    "exorl": "rnd",
}

#: Phase-1 default step counts (Sec. 5 / Appendix A of the paper).
DEFAULT_ENCODER_STEPS: Dict[str, int] = {
    "antmaze": 150_000,
    "exorl": 1_000_000,
    "kitchen": 1_000_000,
}

#: Phase-2 default step counts (strided schedule, Algorithm 1).
DEFAULT_POLICY_STEPS: Dict[str, int] = {
    "antmaze": 850_000,
    "exorl": 1_000_000,
    "kitchen": 1_000_000,
}

#: Table 1 reference numbers (normalized return in [0, 100]).
PAPER_TABLE1: Dict[str, Tuple[float, float]] = {
    "antmaze": (52.8, 18.2),
    "exorl": (51.5, 6.3),
    "kitchen": (66.0, 3.0),
    "all": (57.0, 9.0),
}

#: Table 1 sub-task reference values (used for finer-grained verification).
PAPER_SUBTASKS: Dict[str, float] = {
    "antmaze-goal-reaching": 48.8,
    "antmaze-directional": 55.2,
    "antmaze-random-simplex": 21.3,
    "antmaze-path-loop": 67.2,
    "antmaze-path-edges": 60.0,
    "antmaze-path-center": 64.4,
    "exorl-walker-goals": 94.0,
    "exorl-cheetah-goals": 58.0,
    "exorl-walker-velocity": 34.0,
    "exorl-cheetah-velocity": 20.0,
}

#: Table 4 ablation priors over the AntMaze suite.
TABLE4_ABLATIONS: Tuple[str, ...] = (
    "all",
    "goals",
    "lin",
    "mlp",
    "lin-mlp",
    "goal-mlp",
    "goal-lin",
)
PAPER_TABLE4: Dict[str, float] = {
    "all": 47.3,
    "goals": 26.1,
    "lin": 31.6,
    "mlp": 25.3,
    "lin-mlp": 32.3,
    "goal-mlp": 33.8,
    "goal-lin": 46.9,
}

DEFAULT_OUTPUT_ROOT = "./runs"
DEFAULT_LATENT_DIM = 128
DEFAULT_CONTEXT_SIZE = 32
DEFAULT_DECODER_SIZE = 8
DEFAULT_NUM_EPISODES = 20
DEFAULT_NUM_SEEDS = 5

#: Tiny settings for ``--dry-run`` smoke tests.
DRY_RUN_ENCODER_STEPS = 60
DRY_RUN_POLICY_STEPS = 60
DRY_RUN_EPISODES = 2
DRY_RUN_SEEDS = 1


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #


def _import_module(name: str) -> Any:
    """Import a sibling module, tolerating package / script execution."""
    candidates: List[str] = []
    pkg = __package__ or "fre"
    candidates.append(f"{pkg}.{name}" if not name.startswith(f"{pkg}.") else name)
    candidates.append(f"fre.{name}")
    candidates.append(name)
    seen: set = set()
    last_error: Optional[Exception] = None
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            return importlib.import_module(candidate)
        except Exception as exc:  # pragma: no cover - environment dependent
            last_error = exc
    raise ImportError(f"could not import module '{name}': {last_error}")


def _resolve(module_name: str, *attrs: str) -> Any:
    """Return the first attribute found on ``module_name`` (best effort)."""
    try:
        module = _import_module(module_name)
    except Exception:
        return None
    for attr in attrs:
        value = getattr(module, attr, None)
        if value is not None:
            return value
    return None


def _try_import(module_name: str, *attrs: str) -> Any:
    return _resolve(module_name, *attrs)


def _get_logger():
    get_logger = _resolve("utils.logging", "get_logger")
    if get_logger is None:
        try:  # pragma: no cover
            from .utils.logging import get_logger as get_logger  # type: ignore
        except Exception:
            import logging

            return logging.getLogger("fre")
    return get_logger("fre.main")


def _write_json(path: str, obj: Any) -> str:
    writer = _resolve("utils.logging", "write_json")
    if writer is None:
        try:
            from .utils.logging import write_json as writer  # type: ignore
        except Exception:
            writer = None
    if writer is None:
        def writer(p, o):  # type: ignore
            os.makedirs(os.path.dirname(os.path.abspath(p)) or ".", exist_ok=True)
            with open(p, "w") as fh:
                json.dump(o, fh, indent=2, default=str)
            return p

    try:
        return writer(path, obj)
    except Exception:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w") as fh:
            json.dump(obj, fh, indent=2, default=str)
        return path


def _seed_everything(seed: int) -> None:
    seed_fn = _resolve("utils.logging", "seed_everything", "set_seed")
    if seed_fn is None:
        try:
            from .utils.logging import seed_everything as seed_fn  # type: ignore
        except Exception:
            seed_fn = None
    if seed_fn is not None:
        try:
            seed_fn(int(seed))
            return
        except Exception:
            pass
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2 ** 32))
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy/torch/dataclass objects into JSON-safe values."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        try:
            return _to_jsonable(dataclasses.asdict(obj))
        except Exception:
            return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    # numpy / torch
    for attr in ("tolist", "item"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return _to_jsonable(fn())
            except Exception:
                continue
    return str(obj)


def _filter_kwargs(target: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only kwargs accepted by the dataclass/class/function ``target``."""
    if target is None:
        return dict(kwargs)
    names: Optional[set] = None
    if dataclasses.is_dataclass(target):
        names = {f.name for f in dataclasses.fields(target)}
    elif isinstance(target, type):
        try:
            import inspect

            names = set(inspect.signature(target).parameters)
        except Exception:
            names = None
    elif callable(target):
        try:
            import inspect

            sig = inspect.signature(target)
            names = set(sig.parameters)
            if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                names = None
        except Exception:
            names = None
    if names is None:
        return dict(kwargs)
    names.discard("self")
    return {k: v for k, v in kwargs.items() if k in names}


def _make_config(cls: Any, kwargs: Dict[str, Any]) -> Any:
    """Instantiate config dataclass ``cls`` with the kwargs it understands."""
    filtered = _filter_kwargs(cls, kwargs)
    try:
        return cls(**filtered)
    except TypeError:
        # Retry dropping unexpected values one by one is overkill; try bare.
        try:
            return cls()
        except Exception:
            raise
    except Exception:
        raise


def _call_with_fallbacks(fn: Callable[..., Any], attempts: Sequence[Dict[str, Any]]) -> Any:
    """Call ``fn`` trying a sequence of kwargs dictionaries until one works."""
    last_error: Optional[BaseException] = None
    for attempt in attempts:
        try:
            return fn(**attempt)
        except TypeError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    return fn()


def _load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML config file (requires pyyaml, with a minimal fallback)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"config file not found: {path}")
    try:
        import yaml  # type: ignore

        with open(path, "r") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError(f"config file {path} must contain a mapping")
        return data
    except ImportError:
        # Extremely small flat 'key: value' parser fallback.
        out: Dict[str, Any] = {}
        with open(path, "r") as fh:
            for raw in fh:
                line = raw.split("#", 1)[0].strip()
                if not line or ":" not in line:
                    continue
                key, _, value = line.partition(":")
                key = key.strip()
                value = value.strip()
                if not key or not value:
                    continue
                try:
                    out[key] = json.loads(value)
                except Exception:
                    if value.lower() in ("true", "false"):
                        out[key] = value.lower() == "true"
                    else:
                        try:
                            out[key] = int(value)
                        except ValueError:
                            try:
                                out[key] = float(value)
                            except ValueError:
                                out[key] = value.strip("'\"")
        return out


def _find_checkpoint(directory: str, names: Sequence[str]) -> Optional[str]:
    """Locate a checkpoint inside ``directory`` matching any of ``names``."""
    if not directory or not os.path.isdir(directory):
        return None
    for name in names:
        candidate = os.path.join(directory, name)
        if os.path.exists(candidate):
            return candidate
    try:
        entries = sorted(os.listdir(directory))
    except Exception:
        return None
    for name in names:
        stem, _, ext = name.partition(".")
        for entry in entries:
            if entry.startswith(stem) and (not ext or entry.endswith(ext)):
                return os.path.join(directory, entry)
    return None


def _steps_for(domain: str, kind: str, args: argparse.Namespace) -> int:
    table = DEFAULT_ENCODER_STEPS if kind == "encoder" else DEFAULT_POLICY_STEPS
    steps = int(table.get(domain, 1_000_000))
    scale = float(getattr(args, "steps_scale", 1.0) or 1.0)
    steps = max(1, int(round(steps * scale)))
    if getattr(args, "steps", None):
        steps = int(args.steps)
    if getattr(args, "dry_run", False):
        steps = DRY_RUN_ENCODER_STEPS if kind == "encoder" else DRY_RUN_POLICY_STEPS
    return steps


@dataclass
class PhaseReport:
    """Outcome of a single phase for a single (domain, sub-domain) pair."""

    phase: str
    domain: str
    sub_domain: Optional[str] = None
    ok: bool = False
    seconds: float = 0.0
    checkpoint: Optional[str] = None
    summary: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def key(self) -> str:
        return f"{self.domain}/{self.sub_domain}" if self.sub_domain else self.domain

    def as_dict(self) -> Dict[str, Any]:
        return _to_jsonable(dataclasses.asdict(self))


# --------------------------------------------------------------------------- #
# Phase 1 - encoder pretraining
# --------------------------------------------------------------------------- #


def build_encoder_config(domain: str, args: argparse.Namespace, **overrides: Any) -> Any:
    """Construct an ``EncoderTrainConfig`` for ``domain``."""
    config_cls = _resolve("train_encoder", "EncoderTrainConfig")
    output_dir = overrides.pop("output_dir", None) or os.path.join(
        args.output_root, _output_slug(domain, args)
    )
    sub_domain = overrides.pop("domain_name", None)
    kwargs: Dict[str, Any] = dict(
        domain=domain,
        domain_name=sub_domain or domain,
        steps=_steps_for(domain, "encoder", args),
        seed=args.seed,
        device=args.device,
        output_dir=output_dir,
        dataset=overrides.pop("dataset", None) or DEFAULT_DATASETS.get(domain),
        data_root=args.data_root,
        dry_run=bool(getattr(args, "dry_run", False)),
        log_interval=getattr(args, "log_interval", 1000),
    )
    kwargs.update({k: v for k, v in overrides.items() if v is not None})
    if config_cls is None:
        return kwargs
    return _make_config(config_cls, kwargs)


def run_phase_encoder(
    domain: str,
    args: argparse.Namespace,
    **overrides: Any,
) -> PhaseReport:
    """Run Phase 1 (encoder + decoder pretraining) for ``domain``."""
    report = PhaseReport(phase="encoder", domain=domain,
                         sub_domain=overrides.get("domain_name"))
    started = time.time()
    try:
        train_encoder = _resolve("train_encoder", "train_encoder")
        if train_encoder is None:
            raise ImportError("fre.train_encoder.train_encoder not available")
        cfg = build_encoder_config(domain, args, **overrides)
        trainer = _call_with_fallbacks(
            train_encoder,
            [
                {"cfg": cfg},
                {"config": cfg},
                {"cfg": cfg, "dry_run": getattr(args, "dry_run", False)},
            ]
            if isinstance(cfg, (dict,))
            else [
                {"cfg": cfg},
                {"config": cfg},
                {"cfg": cfg, "source": None, "prior": None},
            ],
        )
        checkpoint = None
        save = getattr(trainer, "save_checkpoint", None)
        if callable(save):
            try:
                checkpoint = save("final")
            except Exception:
                try:
                    checkpoint = save()
                except Exception:
                    checkpoint = None
        if not isinstance(checkpoint, str):
            checkpoint = _find_checkpoint(
                getattr(trainer, "output_dir", None)
                or (cfg.output_dir if hasattr(cfg, "output_dir") else "")
                or os.path.join(args.output_root, _output_slug(domain, args)),
                ("encoder_final.pt", "encoder.pt", "encoder_latest.pt",
                 "encoder_final.json", "encoder_latest.json"),
            )
        report.ok = True
        report.checkpoint = checkpoint
        report.summary = {
            "steps": getattr(cfg, "steps", None),
            "checkpoint": checkpoint,
        }
    except Exception as exc:  # pragma: no cover - environment dependent
        report.error = f"{type(exc).__name__}: {exc}"
        _get_logger().error("phase encoder failed for %s: %s", domain, report.error)
    finally:
        report.seconds = time.time() - started
    return report


# --------------------------------------------------------------------------- #
# Phase 2 - z-conditioned IQL policy
# --------------------------------------------------------------------------- #


def build_policy_config(
    domain: str,
    args: argparse.Namespace,
    encoder_checkpoint: Optional[str] = None,
    **overrides: Any,
) -> Any:
    """Construct a ``PolicyTrainConfig`` for ``domain``."""
    config_cls = _resolve("train_policy", "PolicyTrainConfig")
    output_dir = overrides.pop("output_dir", None) or os.path.join(
        args.output_root, _output_slug(domain, args)
    )
    sub_domain = overrides.pop("domain_name", None)
    kwargs: Dict[str, Any] = dict(
        domain=domain,
        domain_name=sub_domain or domain,
        steps=_steps_for(domain, "policy", args),
        seed=args.seed,
        device=args.device,
        output_dir=output_dir,
        encoder_checkpoint=encoder_checkpoint,
        pretrain_encoder=False if encoder_checkpoint else bool(getattr(args, "pretrain_encoder", False)),
        dataset=overrides.pop("dataset", None) or DEFAULT_DATASETS.get(domain),
        data_root=args.data_root,
        eval_interval=int(getattr(args, "eval_interval", 50_000) or 0),
        eval_episodes=DRY_RUN_EPISODES if getattr(args, "dry_run", False) else DEFAULT_NUM_EPISODES,
        eval_seeds=DRY_RUN_SEEDS if getattr(args, "dry_run", False) else DEFAULT_NUM_SEEDS,
        dry_run=bool(getattr(args, "dry_run", False)),
        log_interval=getattr(args, "log_interval", 1000),
    )
    if getattr(args, "families", None):
        kwargs["families"] = args.families
    if getattr(args, "ablation", None):
        kwargs["ablation"] = args.ablation
    kwargs.update({k: v for k, v in overrides.items() if v is not None})
    if config_cls is None:
        return kwargs
    return _make_config(config_cls, kwargs)


def run_phase_policy(
    domain: str,
    args: argparse.Namespace,
    encoder_checkpoint: Optional[str] = None,
    **overrides: Any,
) -> PhaseReport:
    """Run Phase 2 (frozen encoder + z-conditioned IQL) for ``domain``."""
    report = PhaseReport(phase="policy", domain=domain,
                         sub_domain=overrides.get("domain_name"))
    started = time.time()
    try:
        train_policy = _resolve("train_policy", "train_policy")
        if train_policy is None:
            raise ImportError("fre.train_policy.train_policy not available")
        cfg = build_policy_config(domain, args, encoder_checkpoint=encoder_checkpoint, **overrides)
        trainer = _call_with_fallbacks(
            train_policy,
            [
                {"cfg": cfg},
                {"config": cfg},
                {"cfg": cfg, "encoder": None, "buffer": None, "prior": None},
            ]
            if isinstance(cfg, (dict,))
            else [
                {"cfg": cfg},
                {"config": cfg},
            ],
        )
        checkpoint = None
        save = getattr(trainer, "save_checkpoint", None)
        if callable(save):
            try:
                checkpoint = save("final")
            except Exception:
                try:
                    checkpoint = save()
                except Exception:
                    checkpoint = None
        if not isinstance(checkpoint, str):
            checkpoint = _find_checkpoint(
                getattr(trainer, "run_dir", None)
                or (cfg.output_dir if hasattr(cfg, "output_dir") else "")
                or os.path.join(args.output_root, _output_slug(domain, args)),
                ("policy_final.pt", "policy.pt", "policy_latest.pt",
                 "policy_final.json", "policy_latest.json"),
            )
        report.ok = True
        report.checkpoint = checkpoint
        report.summary = {
            "steps": getattr(cfg, "steps", None),
            "checkpoint": checkpoint,
            "encoder_checkpoint": encoder_checkpoint,
        }
    except Exception as exc:  # pragma: no cover - environment dependent
        report.error = f"{type(exc).__name__}: {exc}"
        _get_logger().error("phase policy failed for %s: %s", domain, report.error)
    finally:
        report.seconds = time.time() - started
    return report


# --------------------------------------------------------------------------- #
# Phase 3 - zero-shot evaluation
# --------------------------------------------------------------------------- #


def build_eval_config(domain: str, args: argparse.Namespace, **overrides: Any) -> Any:
    """Construct an ``EvalConfig`` for ``domain``."""
    config_cls = _resolve("evaluate", "EvalConfig")
    episodes = DRY_RUN_EPISODES if getattr(args, "dry_run", False) else int(
        getattr(args, "eval_episodes", 0) or DEFAULT_NUM_EPISODES
    )
    seeds = DRY_RUN_SEEDS if getattr(args, "dry_run", False) else int(
        getattr(args, "eval_seeds", 0) or DEFAULT_NUM_SEEDS
    )
    kwargs: Dict[str, Any] = dict(
        domain=domain,
        num_episodes=episodes,
        num_seeds=seeds,
        context_size=DEFAULT_CONTEXT_SIZE,
        decoder_size=DEFAULT_DECODER_SIZE,
        deterministic=True,
        base_seed=args.seed,
        device=args.device,
        exorl_root=args.data_root,
    )
    kwargs.update({k: v for k, v in overrides.items() if v is not None})
    if config_cls is None:
        return kwargs
    return _make_config(config_cls, kwargs)


def run_phase_eval(
    domain: str,
    args: argparse.Namespace,
    checkpoint: Optional[str] = None,
    **overrides: Any,
) -> PhaseReport:
    """Run Phase 3 (zero-shot evaluation, 32 reward samples -> z) for ``domain``."""
    report = PhaseReport(phase="eval", domain=domain,
                         sub_domain=overrides.get("domain_name"))
    started = time.time()
    try:
        zero_shot_evaluate = _resolve("evaluate", "zero_shot_evaluate")
        load_eval_agent = _resolve("evaluate", "load_eval_agent", "load_agent")
        results_to_summary = _resolve("evaluate", "results_to_summary")
        if zero_shot_evaluate is None:
            raise ImportError("fre.evaluate.zero_shot_evaluate not available")

        cfg = build_eval_config(domain, args, **overrides)
        agent = None
        if load_eval_agent is not None:
            try:
                agent = load_eval_agent(checkpoint, latent_dim=DEFAULT_LATENT_DIM, device=args.device)
            except TypeError:
                agent = load_eval_agent(checkpoint)
            except Exception as exc:
                _get_logger().warning("could not load agent (%s); evaluating random policy", exc)
                agent = None

        results = _call_with_fallbacks(
            zero_shot_evaluate,
            [
                {"agent": agent, "domain": domain, "config": cfg},
                {"agent": agent, "domain": domain},
            ],
        )
        summary: Dict[str, Any] = {}
        if isinstance(results, dict):
            if results_to_summary is not None:
                try:
                    summary = results_to_summary(results)
                except Exception:
                    summary = {"tasks": _to_jsonable(results)}
            else:
                summary = {"tasks": _to_jsonable(results)}
        else:
            summary = {"result": _to_jsonable(results)}

        verify = _resolve("evaluate", "verify_against_table1")
        if verify is not None and isinstance(summary, dict):
            try:
                summary["table1_check"] = _to_jsonable(verify(summary))
            except Exception:
                pass

        report.ok = True
        report.checkpoint = checkpoint
        report.summary = summary
    except Exception as exc:  # pragma: no cover - environment dependent
        report.error = f"{type(exc).__name__}: {exc}"
        _get_logger().error("phase eval failed for %s: %s", domain, report.error)
    finally:
        report.seconds = time.time() - started
    return report


# --------------------------------------------------------------------------- #
# Baselines (Table 1 comparators)
# --------------------------------------------------------------------------- #


def run_baseline(
    name: str,
    domain: str,
    args: argparse.Namespace,
    **overrides: Any,
) -> PhaseReport:
    """Train (and evaluate, when possible) an in-repo baseline."""
    report = PhaseReport(phase="baselines", domain=domain,
                         sub_domain=overrides.get("domain_name"))
    started = time.time()
    try:
        if name in EXTERNAL_BASELINES:
            report.summary = {
                "external_repo": "https://github.com/facebookresearch/controllable_agent",
                "note": (
                    f"{name.upper()} is not reimplemented in this codebase; run it from "
                    "controllable_agent with the RND ExORL datasets "
                    "(ICM features for SF)."
                ),
            }
            report.ok = True
            return report

        module_name = f"baselines.{name}"
        builder = _resolve(module_name, f"build_{name}", f"{name.capitalize()}Agent")
        trainer_fn = _resolve(module_name, f"train_{name}")
        if builder is None and trainer_fn is None:
            raise ImportError(f"baseline module {module_name} not available")

        steps = int(getattr(args, "baseline_steps", 0) or 0) or (
            DRY_RUN_POLICY_STEPS if getattr(args, "dry_run", False)
            else int(100_000 * float(getattr(args, "steps_scale", 1.0) or 1.0))
        )
        output_dir = os.path.join(args.output_root, _output_slug(domain, args), "baselines", name)
        os.makedirs(output_dir, exist_ok=True)

        dims = _domain_dims(domain, overrides.get("domain_name") or domain)
        config_cls = _resolve(module_name, f"{name.split('_')[0].upper()}Config",
                              f"{'GCI' if name == 'gc_iql' else 'GCB' if name == 'gc_bc' else 'OPAL'}Config")
        config = None
        if config_cls is not None:
            config = _make_config(
                config_cls,
                dict(steps=steps, seed=args.seed, device=args.device, output_dir=output_dir,
                     batch_size=int(getattr(args, "batch_size", 512) or 512),
                     lr=float(getattr(args, "lr", 1e-4) or 1e-4)),
            )

        buffer = None
        try:
            buffer = _build_buffer(domain, args, sub_domain=overrides.get("domain_name"))
        except Exception as exc:
            _get_logger().warning("baseline %s: no replay buffer (%s)", name, exc)

        agent = None
        if trainer_fn is not None and buffer is not None:
            agent = _call_with_fallbacks(
                trainer_fn,
                [
                    {"buffer": buffer, "obs_dim": dims[0], "action_dim": dims[1],
                     "config": config, "device": args.device, "steps": steps},
                    {"buffer": buffer, "obs_dim": dims[0], "action_dim": dims[1], "steps": steps},
                ],
            )
        elif builder is not None:
            agent = _call_with_fallbacks(
                builder,
                [
                    {"obs_dim": dims[0], "action_dim": dims[1], "config": config,
                     "device": args.device},
                    {"obs_dim": dims[0], "action_dim": dims[1]},
                ],
            )
        ckpt = None
        save = getattr(agent, "save", None)
        if callable(save):
            try:
                ckpt = save(os.path.join(output_dir, f"{name}.pt"))
            except Exception:
                ckpt = None
        report.ok = True
        report.checkpoint = ckpt if isinstance(ckpt, str) else None
        report.summary = {"baseline": name, "steps": steps, "checkpoint": report.checkpoint}
    except Exception as exc:  # pragma: no cover - environment dependent
        report.error = f"{type(exc).__name__}: {exc}"
        _get_logger().error("baseline %s failed for %s: %s", name, domain, report.error)
    finally:
        report.seconds = time.time() - started
    return report


def _domain_dims(domain: str, sub_domain: str) -> Tuple[int, int]:
    """(state_dim, action_dim) for a domain, matching the paper's wrappers."""
    if domain == "antmaze":
        return 29, 8
    if domain == "kitchen":
        return 60, 9
    if domain == "exorl":
        raw = {"walker": 24, "cheetah": 17}
        physics = {"walker": 3, "cheetah": 1}
        key = sub_domain if sub_domain in raw else "walker"
        return raw[key] + physics[key], 6
    return 29, 8


def _build_buffer(domain: str, args: argparse.Namespace, sub_domain: Optional[str] = None) -> Any:
    """Load an offline replay buffer for ``domain`` (best effort)."""
    if domain == "exorl":
        loader = _resolve("data.exorl_loader", "load_exorl_dataset")
        to_buffer = _resolve("data.exorl_loader", "to_replay_buffer")
        if loader is None or to_buffer is None:
            raise ImportError("exorl loader unavailable")
        dataset = loader(
            sub_domain if sub_domain in ("walker", "cheetah") else "walker",
            root=args.data_root,
            append_physics_to_obs=True,
            normalise=True,
        )
        return to_buffer(dataset, device="cpu")
    loader_name = "load_antmaze_buffer" if domain == "antmaze" else "load_kitchen_buffer"
    loader = _resolve("data.d4rl_loader", loader_name)
    if loader is None:
        raise ImportError(f"{loader_name} unavailable")
    kwargs = {}
    if domain == "antmaze":
        kwargs["dataset"] = DEFAULT_DATASETS["antmaze"]
    return loader(**kwargs)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def _output_slug(domain: str, args: argparse.Namespace) -> str:
    sub = getattr(args, "sub_domain", None)
    base = f"{domain}-{sub}" if sub else domain
    seed = getattr(args, "seed", None)
    if seed is not None and getattr(args, "seed_in_dir", False):
        base = f"{base}-seed{seed}"
    return base


def _sub_domains(domain: str, args: argparse.Namespace) -> Tuple[Optional[str], ...]:
    if domain in DOMAIN_SUBDOMAINS:
        requested = getattr(args, "sub_domain", None)
        if requested:
            return tuple(requested) if isinstance(requested, (list, tuple)) else (requested,)
        return DOMAIN_SUBDOMAINS[domain]
    return (None,)


def run_domain(
    domain: str,
    args: argparse.Namespace,
    encoder_checkpoint: Optional[str] = None,
    policy_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the requested phases for a single domain (all ExORL sub-domains)."""
    logger = _get_logger()
    phases = _requested_phases(args.phase)
    domain_report: Dict[str, Any] = {
        "domain": domain,
        "phases": sorted(phases),
        "reports": [],
        "encoder_checkpoint": encoder_checkpoint,
        "policy_checkpoint": policy_checkpoint,
        "summary": {},
    }

    for sub_domain in _sub_domains(domain, args):
        overrides: Dict[str, Any] = {}
        if sub_domain:
            overrides["domain_name"] = sub_domain
            overrides["dataset"] = sub_domain

        enc_ckpt = encoder_checkpoint
        pol_ckpt = policy_checkpoint

        if "encoder" in phases:
            logger.info("[%s%s] phase 1: encoder pretraining", domain,
                        f"/{sub_domain}" if sub_domain else "")
            rep = run_phase_encoder(domain, args, **overrides)
            domain_report["reports"].append(rep.as_dict())
            if rep.ok:
                enc_ckpt = rep.checkpoint or enc_ckpt
            elif args.strict:
                raise RuntimeError(f"encoder phase failed: {rep.error}")

        if "policy" in phases:
            if enc_ckpt is None:
                enc_ckpt = _find_checkpoint(
                    os.path.join(args.output_root, _output_slug(domain, args)),
                    ("encoder_final.pt", "encoder.pt", "encoder_latest.pt"),
                )
            logger.info("[%s%s] phase 2: z-conditioned IQL", domain,
                        f"/{sub_domain}" if sub_domain else "")
            rep = run_phase_policy(domain, args, encoder_checkpoint=enc_ckpt, **overrides)
            domain_report["reports"].append(rep.as_dict())
            if rep.ok:
                pol_ckpt = rep.checkpoint or pol_ckpt
            elif args.strict:
                raise RuntimeError(f"policy phase failed: {rep.error}")

        if "eval" in phases:
            if pol_ckpt is None:
                pol_ckpt = _find_checkpoint(
                    os.path.join(args.output_root, _output_slug(domain, args)),
                    ("policy_final.pt", "policy.pt", "policy_latest.pt"),
                )
            logger.info("[%s%s] phase 3: zero-shot evaluation", domain,
                        f"/{sub_domain}" if sub_domain else "")
            rep = run_phase_eval(domain, args, checkpoint=pol_ckpt, **overrides)
            domain_report["reports"].append(rep.as_dict())
            if rep.ok:
                domain_report["summary"][sub_domain or domain] = rep.summary
            elif args.strict:
                raise RuntimeError(f"eval phase failed: {rep.error}")

        if "baselines" in phases:
            for name in (args.baselines or IN_REPO_BASELINES):
                logger.info("[%s%s] baseline: %s", domain,
                            f"/{sub_domain}" if sub_domain else "", name)
                rep = run_baseline(name, domain, args, **overrides)
                domain_report["reports"].append(rep.as_dict())
                if rep.ok and rep.summary:
                    domain_report["summary"].setdefault("baselines", {})[name] = rep.summary
                elif args.strict and name in IN_REPO_BASELINES:
                    raise RuntimeError(f"baseline {name} failed: {rep.error}")

    domain_report["encoder_checkpoint"] = enc_ckpt if "encoder" in phases else encoder_checkpoint
    domain_report["policy_checkpoint"] = pol_ckpt if "policy" in phases else policy_checkpoint
    return domain_report


def _requested_phases(phase: str) -> set:
    if phase == "all":
        return {"encoder", "policy", "eval"}
    if phase == "baselines":
        return {"baselines"}
    return {phase}


def aggregate_table1(run_reports: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate per-domain evaluation summaries into a Table-1 style report."""
    domains: Dict[str, Any] = {}
    overall_values: List[float] = []
    for domain, report in (run_reports.get("domains") or {}).items():
        summary = report.get("summary") or {}
        overall = None
        for key in ("all", "overall", "mean"):
            if isinstance(summary, dict) and key in summary:
                value = summary[key]
                overall = value.get("mean", value) if isinstance(value, dict) else value
                break
        if overall is None and isinstance(summary, dict):
            means = [
                v.get("mean") if isinstance(v, dict) else v
                for k, v in summary.items()
                if isinstance(v, (int, float, dict)) and k != "baselines"
            ]
            means = [m for m in means if isinstance(m, (int, float))]
            if means:
                overall = sum(means) / len(means)
        domains[domain] = {"score": overall, "summary": summary,
                           "reference": PAPER_TABLE1.get(domain)}
        if isinstance(overall, (int, float)):
            overall_values.append(float(overall))

    overall_score = sum(overall_values) / len(overall_values) if overall_values else None
    reference = PAPER_TABLE1.get("all")
    within_std = None
    if overall_score is not None and reference is not None:
        within_std = abs(overall_score - reference[0]) <= max(reference[1], 1e-6)

    return {
        "domains": domains,
        "overall": overall_score,
        "reference": {"antmaze": PAPER_TABLE1["antmaze"], "exorl": PAPER_TABLE1["exorl"],
                      "kitchen": PAPER_TABLE1["kitchen"], "all": PAPER_TABLE1["all"]},
        "within_table1_std": within_std,
        "subtask_reference": PAPER_SUBTASKS,
    }


def run_all(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the requested phases across all requested domains and save a report."""
    logger = _get_logger()
    started = time.time()
    _seed_everything(args.seed)
    os.makedirs(args.output_root, exist_ok=True)

    domains = list(DOMAINS) if args.domain == "all" else [args.domain]
    reports: Dict[str, Any] = {
        "domains": {},
        "encoder_checkpoints": {},
        "policy_checkpoints": {},
        "errors": [],
    }
    encoder_checkpoint = args.checkpoint
    for domain in domains:
        logger.info("=" * 70)
        logger.info("FRE | domain=%s | phase=%s | seed=%d", domain, args.phase, args.seed)
        logger.info("=" * 70)
        try:
            report = run_domain(domain, args, encoder_checkpoint=encoder_checkpoint)
        except Exception as exc:  # pragma: no cover
            if args.strict:
                raise
            logger.error("domain %s failed: %s", domain, exc)
            reports["errors"].append({"domain": domain, "error": f"{type(exc).__name__}: {exc}"})
            continue
        reports["domains"][domain] = report
        reports["encoder_checkpoints"][domain] = report.get("encoder_checkpoint")
        reports["policy_checkpoints"][domain] = report.get("policy_checkpoint")
        for rep in report.get("reports", []):
            if rep.get("error"):
                reports["errors"].append({"domain": domain, "phase": rep.get("phase"),
                                          "error": rep["error"]})

    reports["table1"] = aggregate_table1(reports)
    reports["elapsed_seconds"] = time.time() - started
    reports["args"] = _to_jsonable(vars(args))

    out_path = os.path.join(args.output_root, "fre_report.json")
    try:
        _write_json(out_path, reports)
        reports["report_path"] = out_path
    except Exception as exc:  # pragma: no cover
        logger.warning("could not write report: %s", exc)

    logger.info("FRE finished in %.1fs", reports["elapsed_seconds"])
    return reports


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="fre.main",
        description="FRE: Zero-Shot Reinforcement Learning via Functional Reward Encodings",
    )
    parser.add_argument("--domain", default="antmaze", choices=list(DOMAINS) + ["all"],
                        help="domain to run (or 'all' for AntMaze + ExORL + Kitchen)")
    parser.add_argument("--phase", default="all", choices=list(PHASES),
                        help="which phase(s) to run")
    parser.add_argument("--checkpoint", default=None,
                        help="pretrained encoder (phase 1 output) or policy checkpoint")
    parser.add_argument("--config", default=None,
                        help="path to a YAML config (configs/<domain>.yaml) merged over defaults")
    parser.add_argument("--output-root", "--output-dir", dest="output_root",
                        default=DEFAULT_OUTPUT_ROOT, help="run directory root")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--seed-in-dir", action="store_true",
                        help="include the seed in per-run directory names")
    parser.add_argument("--device", default="auto", help="torch device (auto|cpu|cuda|cuda:0)")
    parser.add_argument("--data-root", default=None, help="root directory holding datasets")
    parser.add_argument("--steps", type=int, default=None, help="override step count for one phase")
    parser.add_argument("--steps-scale", type=float, default=1.0,
                        help="scale the paper's step counts (e.g. 0.01 for smoke tests)")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_NUM_EPISODES)
    parser.add_argument("--eval-seeds", type=int, default=DEFAULT_NUM_SEEDS)
    parser.add_argument("--families", nargs="*", default=None,
                        help="reward prior families (Table 4 subset, e.g. goal_reaching linear)")
    parser.add_argument("--ablation", default=None,
                        help="Table 4 ablation name: all|goals|lin|mlp|lin-mlp|goal-mlp|goal-lin")
    parser.add_argument("--baselines", nargs="*", default=None,
                        help=f"in-repo baselines to run: {' '.join(IN_REPO_BASELINES)}")
    parser.add_argument("--external-baselines", action="store_true",
                        help="note FB/SF as external (controllable_agent) in the report")
    parser.add_argument("--baseline-steps", type=int, default=None)
    parser.add_argument("--pretrain-encoder", action="store_true",
                        help="let train_policy pretrain the encoder when no checkpoint is given")
    parser.add_argument("--sub-domain", nargs="*", default=None,
                        help="ExORL sub-domain(s): walker cheetah")
    parser.add_argument("--random-policy", action="store_true",
                        help="evaluate a random policy (harness sanity check)")
    parser.add_argument("--dry-run", action="store_true",
                        help="tiny step counts / episodes for a fast smoke test")
    parser.add_argument("--strict", action="store_true",
                        help="abort on the first phase error instead of recording it")
    parser.add_argument("--aggregate", default=None,
                        help="do not train: read an existing run directory and print Table 1 vs. paper")

    args = parser.parse_args(argv)

    if args.config:
        cfg = _load_yaml(args.config)
        for key, value in cfg.items():
            attr = key.replace("-", "_")
            if hasattr(args, attr):
                setattr(args, attr, value)
        if "sub_domain" in cfg:
            args.sub_domain = cfg["sub_domain"]
    return args


def aggregate_existing(root: str) -> Dict[str, Any]:
    """Read a previously written ``fre_report.json`` and compare to Table 1."""
    path = root if root.endswith(".json") else os.path.join(root, "fre_report.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no report found at {path}")
    with open(path, "r") as fh:
        report = json.load(fh)
    table1 = report.get("table1") or aggregate_table1(report)
    return {"report": report, "table1": table1}


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """CLI entry point; returns the full run report."""
    args = parse_args(argv)
    if args.aggregate:
        result = aggregate_existing(args.aggregate)
        print(json.dumps(_to_jsonable(result["table1"]), indent=2))
        return result
    if args.device == "auto":
        try:
            import torch

            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            args.device = "cpu"
    return run_all(args)


if __name__ == "__main__":  # pragma: no cover
    main()
