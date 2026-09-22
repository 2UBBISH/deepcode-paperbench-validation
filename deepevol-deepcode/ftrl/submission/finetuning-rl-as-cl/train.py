"""Environment-agnostic training entry point for the reproduction of

    "Fine-tuning Reinforcement Learning Models is Secretly a Forgetting
    Mitigation Problem" (Wołczyk et al., 2024)

The module is intentionally a *thin dispatcher*: it resolves the requested
environment to the corresponding trainer already implemented in ``src/`` and
forwards the run configuration.  All heavy third-party imports (torch, NLE,
Meta-World, gym/ALE) are performed lazily inside the corresponding trainer, so
this file (and ``--help``) works on a dependency-free CPU-only machine.

Usage
-----
    python train.py --env robotic_sequence
    python train.py --env nethack --method ks --seed 0 --total-steps 500000000
    python train.py --env montezuma --method bc --output-dir results/mz
    python train.py --env toy --set apple_retrieval.M=50
    python train.py --list

The canonical programmatic interface is :func:`run_env`, which is what
``main.py`` (``cmd_train``) invokes.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import sys
import traceback
from typing import Any, Dict, Iterable, List, Optional, Sequence

__all__ = ["run_env", "train", "ENV_TRAINERS", "METHOD_ALIASES", "build_parser", "main"]

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:  # allow `python train.py` from any cwd
    sys.path.insert(0, _HERE)

# ---------------------------------------------------------------------------
# Environment -> (module, callable) routing table
# ---------------------------------------------------------------------------
ENV_TRAINERS: Dict[str, str] = {
    # environment alias             : "module:function"
    "robotic_sequence": "src.robotic_sequence.train_robotic:run_experiment",
    "robotic-sequence": "src.robotic_sequence.train_robotic:run_experiment",
    "robotic": "src.robotic_sequence.train_robotic:run_experiment",
    "metaworld": "src.robotic_sequence.train_robotic:run_experiment",
    "meta-world": "src.robotic_sequence.train_robotic:run_experiment",
    "montezuma": "src.montezuma.train_montezuma:run_pipeline",
    "montezumas": "src.montezuma.train_montezuma:run_pipeline",
    "nethack": "src.nethack.train_nethack:run_pipeline",
    "net_hack": "src.nethack.train_nethack:run_pipeline",
}

TOY_ENVS = ("toy", "two_state_mdp", "apple_retrieval", "toy_mdp")

# Aliases used across the paper / configs -> canonical method names
METHOD_ALIASES: Dict[str, str] = {
    "": "none",
    "vanilla": "none",
    "vanilla_finetuning": "none",
    "finetune": "none",
    "fine_tuning": "none",
    "none": "none",
    "ewc": "ewc",
    "elastic": "ewc",
    "bc": "bc",
    "behavioral_cloning": "bc",
    "replay": "bc",
    "distillation": "bc",
    "ks": "ks",
    "kickstarting": "ks",
    "kickstart": "ks",
    "em": "em",
    "episodic_memory": "em",
    "episodic": "em",
    "scratch": "scratch",
    "from_scratch": "scratch",
    "baseline": "scratch",
}

# Default method sets per environment (matches the paper's experiment grids)
DEFAULT_METHODS: Dict[str, Sequence[str]] = {
    "robotic_sequence": ("none", "ewc", "bc", "em"),
    "montezuma": ("none", "bc", "ewc", "scratch"),
    "nethack": ("none", "ewc", "bc", "ks", "scratch"),
    "toy": ("none",),
}


def normalize_method(method: Optional[str]) -> Optional[str]:
    """Map a method alias (``vanilla``/``kickstarting``/...) to its canonical name."""
    if method is None:
        return None
    key = str(method).strip().lower().replace("-", "_").replace(" ", "_")
    return METHOD_ALIASES.get(key, key)


def normalize_env(env: Optional[str], cfg: Any = None) -> str:
    """Resolve the environment name from an explicit argument, config or default."""
    candidates: List[Any] = [env, os.environ.get("FTRL_ENV")]
    for attr in ("env_name", "name", "env"):
        value = _cfg_get(cfg, attr)
        if isinstance(value, str) and value:
            candidates.append(value)
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip().lower()
    return "robotic_sequence"


# ---------------------------------------------------------------------------
# Small tolerant helpers
# ---------------------------------------------------------------------------
def _import(module: str, attr: Optional[str] = None) -> Any:
    """Import ``module`` (trying ``module`` then ``src.module``), optionally ``attr``."""
    last_error: Optional[BaseException] = None
    for name in (module, f"src.{module}"):
        try:
            mod = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - tolerate missing heavy deps
            last_error = exc
            continue
        if attr is None:
            return mod
        if hasattr(mod, attr):
            return getattr(mod, attr)
        # try a relative-style import such as "src.x.y"
    if last_error is not None:
        raise ImportError(f"could not import '{module}' (attr={attr}): {last_error!r}")
    raise ImportError(f"could not import '{module}' (attr={attr})")


def _cfg_get(cfg: Any, key: str, default: Optional[Any] = None) -> Any:
    """Dotted-path lookup tolerant of dicts, Config objects and dataclasses."""
    if cfg is None:
        return default
    if "." in key:
        head, _, tail = key.partition(".")
        return _cfg_get(_cfg_get(cfg, head, None), tail, default)
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    try:
        value = getattr(cfg, key)
    except Exception:  # noqa: BLE001
        try:
            value = cfg[key]  # type: ignore[index]
        except Exception:  # noqa: BLE001
            return default
    return default if value is None else value


def _forward(fn: Any, **kwargs: Any) -> Any:
    """Call ``fn`` passing only the keyword arguments accepted by its signature."""
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(**kwargs)
    accepts_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    )
    if accepts_kwargs:
        return fn(**kwargs)
    allowed = {k: v for k, v in kwargs.items() if k in signature.parameters}
    return fn(**allowed)


def _jsonable(value: Any, _depth: int = 0) -> Any:
    """Best-effort conversion of result objects into JSON-serialisable data."""
    if _depth > 4:
        return repr(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v, _depth + 1) for v in value]
    for attr in ("as_dict", "to_dict"):
        method = getattr(value, attr, None)
        if callable(method):
            try:
                return _jsonable(method(), _depth + 1)
            except Exception:  # noqa: BLE001
                pass
    return repr(value)


def _module_path(spec: str) -> Any:
    module, _, attr = spec.partition(":")
    return _import(module, attr or None)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _canonical_methods(method: Optional[str], env: str) -> Sequence[str]:
    if method is None or method in ("", "all"):
        return DEFAULT_METHODS.get(env, ("none",))
    normalized = normalize_method(method)
    return (normalized or "none",)


def _canonical_seeds(seeds: Optional[Iterable[int]], seed: Optional[int],
                     num_seeds: Optional[int]) -> List[int]:
    if seeds is not None:
        return [int(s) for s in seeds]
    if num_seeds is not None and int(num_seeds) > 0:
        base = 0 if seed is None else int(seed)
        return list(range(base, base + int(num_seeds)))
    if seed is not None:
        return [int(seed)]
    return [0]


def run_env(env: Optional[str] = None, cfg: Any = None, *,
            method: Optional[str] = None,
            seed: Optional[int] = None,
            seeds: Optional[Iterable[int]] = None,
            num_seeds: Optional[int] = None,
            total_steps: Optional[int] = None,
            output_dir: Optional[str] = None,
            checkpoint: Optional[str] = None,
            stub: Optional[bool] = None,
            device: Optional[str] = None,
            pretrain_only: bool = False,
            sweep: bool = False,
            verbose: bool = True,
            logger: Any = None,
            progress_fn: Any = None,
            **kwargs: Any) -> Dict[str, Any]:
    """Train ``env`` with the paper's protocol and return a JSON-able summary.

    Parameters
    ----------
    env:
        Environment identifier (``robotic_sequence``/``metaworld``, ``montezuma``,
        ``nethack`` or ``toy``).
    cfg:
        Loaded configuration object (``src.common.config.Config``/mapping) or
        ``None`` to let the trainer build its own defaults.
    method:
        Retention method (``none``/``ewc``/``bc``/``ks``/``em``/``scratch``).
        Aliases such as ``vanilla`` and ``kickstarting`` are accepted.
    total_steps:
        Environment-step budget for the fine-tuning stage.
    output_dir:
        Directory where checkpoints/summaries are written.
    checkpoint:
        Pre-trained ``pi_*`` checkpoint to start fine-tuning from.
    stub:
        Use the dependency-free CPU stand-in environments (smoke tests).

    Returns
    -------
    dict
        ``{"env", "method", "seeds", "status", "result"}`` where ``result`` is the
        trainer's (serialised) output, or an ``"error"``/``"traceback"`` pair when
        the environment is unavailable.
    """
    name = normalize_env(env, cfg)
    payload: Dict[str, Any] = {
        "env": name,
        "method": normalize_method(method),
        "seeds": _canonical_seeds(seeds, seed, num_seeds),
        "output_dir": output_dir,
        "total_steps": total_steps,
    }

    if name in TOY_ENVS:
        return _run_toy(payload, cfg=cfg, output_dir=output_dir, sweep=sweep,
                        verbose=verbose, name=name)

    spec = ENV_TRAINERS.get(name)
    if spec is None:
        payload.update(status="unknown-env",
                       error=f"unknown environment '{name}'",
                       known=sorted(ENV_TRAINERS) + list(TOY_ENVS))
        return payload

    try:
        trainer = _module_path(spec)
    except Exception as exc:  # noqa: BLE001
        payload.update(status="unavailable", error=f"{type(exc).__name__}: {exc}")
        if verbose:
            traceback.print_exc()
        return payload

    methods = _canonical_methods(method, name)
    try:
        if name in ("robotic_sequence", "robotic-sequence", "robotic",
                    "metaworld", "meta-world"):
            result = _forward(
                trainer,
                cfg=cfg,
                stub=stub,
                seed=payload["seeds"][0],
                device=device,
                output_dir=output_dir,
                num_steps=total_steps,
                pretrain_only=pretrain_only,
                logger=logger,
            )
        elif name.startswith("montezuma"):
            result = _forward(
                trainer,
                cfg=cfg,
                methods=methods,
                seeds=payload["seeds"],
                m1_checkpoint=checkpoint,
                total_steps=total_steps,
                output_dir=output_dir,
                stub=stub,
                verbose=verbose,
                logger=logger,
                progress_fn=progress_fn,
            )
            payload["methods"] = list(methods)
        else:  # nethack
            result = _forward(
                trainer,
                cfg=cfg,
                methods=methods,
                seeds=payload["seeds"],
                total_steps=total_steps,
                checkpoint=checkpoint,
                output_dir=output_dir,
                stub=stub,
                verbose=verbose,
                logger=logger,
                progress_fn=progress_fn,
            )
            payload["methods"] = list(methods)
    except Exception as exc:  # noqa: BLE001 - report, never crash the dispatcher
        payload.update(status="error", error=f"{type(exc).__name__}: {exc}",
                       traceback=traceback.format_exc())
        if verbose:
            traceback.print_exc()
        return payload

    payload.update(status="ok", result=_jsonable(result))
    if verbose:
        print(f"[train] {name}: finished ({len(payload['seeds'])} seed(s))")
    return payload


# Alias kept for convenience / backwards compatibility with callers
train = run_env


def _run_toy(payload: Dict[str, Any], *, cfg: Any, output_dir: Optional[str],
             sweep: bool, verbose: bool, name: str) -> Dict[str, Any]:
    """Run the Appendix-A toy sanity checks (two-state MDP / AppleRetrieval)."""
    try:
        runner = _import("toy", "run_toy_examples")
    except Exception as exc:  # noqa: BLE001
        payload.update(status="unavailable", error=f"{type(exc).__name__}: {exc}")
        if verbose:
            traceback.print_exc()
        return payload
    try:
        result = _forward(runner, output_dir=output_dir, sweep=sweep, verbose=verbose)
    except Exception as exc:  # noqa: BLE001
        payload.update(status="error", error=f"{type(exc).__name__}: {exc}",
                       traceback=traceback.format_exc())
        if verbose:
            traceback.print_exc()
        return payload
    payload.update(status="ok", result=_jsonable(result))
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train.py",
        description=("Environment-agnostic training entry point for the "
                     "'fine-tuning RL as a forgetting-mitigation problem' reproduction."),
    )
    parser.add_argument("--env", "--environment", dest="env", default=None,
                        help="toy | robotic_sequence (metaworld) | montezuma | nethack")
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_argument("--set", dest="overrides", nargs="*", default=None,
                        help="config overrides of the form a.b=value")
    parser.add_argument("--method", default=None,
                        help="none|ewc|bc|ks|em|scratch (aliases accepted)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="explicit list of seeds (overrides --seed/--num-seeds)")
    parser.add_argument("--num-seeds", type=int, default=None,
                        help="run seeds seed..seed+num_seeds-1")
    parser.add_argument("--total-steps", type=int, default=None,
                        help="fine-tuning environment-step budget")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint", default=None,
                        help="pre-trained pi_* checkpoint to fine-tune from")
    parser.add_argument("--device", default=None, help="cpu|cuda")
    parser.add_argument("--pretrain-only", action="store_true",
                        help="only pre-train pi_* (robotic_sequence)")
    parser.add_argument("--sweep", action="store_true",
                        help="run the toy sweeps (toy environment only)")
    parser.add_argument("--stub", action="store_true",
                        help="use dependency-free CPU stand-in environments")
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--list", action="store_true",
                        help="list the environments handled by this entry point")
    return parser


def _load_config(config: Optional[str], env: Optional[str],
                 overrides: Optional[Iterable[str]] = None) -> Any:
    """Best-effort config loading (absolute / cwd-relative / configs/<env>.yaml)."""
    if not config:
        if not env:
            return None
        candidate = os.path.join(_HERE, "configs", f"{env}.yaml")
        config = candidate if os.path.isfile(candidate) else None
    if not config:
        return None
    paths = [config, os.path.join(os.getcwd(), config),
             os.path.join(_HERE, "configs", config)]
    for path in paths:
        if os.path.isfile(path):
            try:
                loader = _import("common.config", "load_config")
                cfg = loader(path)
                if overrides:
                    apply = _import("common.config", "apply_overrides")
                    cfg = _forward(apply, cfg=cfg, overrides=list(overrides))
                return cfg
            except Exception:  # noqa: BLE001
                if not os.environ.get("FTRL_QUIET"):
                    traceback.print_exc()
                return None
    if not os.environ.get("FTRL_QUIET"):
        print(f"[train] warning: config not found: {config}", file=sys.stderr)
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        print("Environments:")
        for name in sorted(set(ENV_TRAINERS) | set(TOY_ENVS)):
            print(f"  - {name}")
        print("Default methods per environment:")
        for name, methods in DEFAULT_METHODS.items():
            print(f"  {name}: {', '.join(methods)}")
        return 0

    cfg = _load_config(args.config, args.env or "robotic_sequence", args.overrides)
    result = run_env(
        env=args.env,
        cfg=cfg,
        method=args.method,
        seed=args.seed,
        seeds=args.seeds,
        num_seeds=args.num_seeds,
        total_steps=args.total_steps,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        stub=(True if args.stub else None),
        device=args.device,
        pretrain_only=args.pretrain_only,
        sweep=args.sweep,
        verbose=not args.quiet,
    )

    if args.json or not args.quiet:
        try:
            print(json.dumps(result, indent=2, default=str))
        except Exception:  # noqa: BLE001
            print(result)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        summary_path = os.path.join(args.output_dir, "train_summary.json")
        try:
            with open(summary_path, "w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2, default=str)
        except Exception:  # noqa: BLE001
            pass

    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
