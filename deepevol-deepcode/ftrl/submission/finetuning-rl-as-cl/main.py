#!/usr/bin/env python
"""Unified command-line dispatcher for the reproduction of

    "Fine-tuning Reinforcement Learning Models is Secretly a Forgetting
     Mitigation Problem"  (Wolczyk et al., 2024)

The repository is organised as one package per environment plus a shared
``src.retention`` module implementing the four knowledge-retention mechanisms
(EWC, behavioral cloning, kickstarting, episodic memory).  This file is a thin
dispatcher: it parses the common CLI flags, resolves the YAML config through
``src.common.config`` and forwards the call to the environment specific
trainer/evaluator or to one of the analysis modules.

Sub-commands
------------
``train``    -- environment agnostic training entry point (``train.py``)
``evaluate`` -- environment agnostic evaluation entry point (``evaluate.py``)
``toy``      -- Appendix A sanity checks (two-state MDP + AppleRetrieval)
``analyze``  -- post-hoc analysis figures / tables (CKA, forward transfer, ...)
``test``     -- run the dependency-light smoke test-suite shipped in ``tests``

Examples
--------
    python main.py train  --config configs/robotic_sequence.yaml --method bc
    python main.py train  --env nethack --method ks --total-steps 500000000
    python main.py evaluate --env nethack --checkpoint results/nethack/ks/seed_0
    python main.py toy
    python main.py analyze --what forward_transfer --results-dir results/robotic_sequence
    python main.py test

All environment specific dependencies (NLE, Meta-World, ALE, torch) are imported
lazily so that ``python main.py --help`` and the toy/test sub-commands work on a
bare CPU-only machine.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

__version__ = "1.0.0"

# Environments handled by the repository (name -> default config file name).
ENVS: Dict[str, str] = {
    "toy": "toy.yaml",
    "robotic_sequence": "robotic_sequence.yaml",
    "metaworld": "robotic_sequence.yaml",
    "montezuma": "montezuma.yaml",
    "nethack": "nethack.yaml",
}

# Retention methods reproduced in the paper.
METHODS = ("none", "vanilla", "ewc", "bc", "ks", "kickstarting", "em", "scratch")

# Analysis deliverables (name -> short module inside src.analysis).
ANALYSIS_TARGETS: Dict[str, str] = {
    "forward_transfer": "forward_transfer",
    "ft": "forward_transfer",
    "table6": "forward_transfer",
    "cka": "cka",
    "loglikelihood": "loglikelihood",
    "log_likelihood": "loglikelihood",
    "ll": "loglikelihood",
    "pca": "pca_viz",
    "pca_viz": "pca_viz",
    "density": "density_plots",
    "density_plots": "density_plots",
    "returns": "return_distribution",
    "return_distribution": "return_distribution",
    "plotting": "plotting",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _log(message: str) -> None:
    print(message, flush=True)


def _import(module: str, attr: Optional[str] = None) -> Any:
    """Import ``module`` (``src.x.y`` or ``x.y``) optionally fetching ``attr``."""
    candidates = [module]
    if not module.startswith("src."):
        candidates.append("src." + module)
    last_error: Optional[BaseException] = None
    for candidate in candidates:
        try:
            mod = importlib.import_module(candidate)
        except Exception as exc:  # pragma: no cover - environment dependent
            last_error = exc
            continue
        if attr is None:
            return mod
        if hasattr(mod, attr):
            return getattr(mod, attr)
        last_error = AttributeError(f"{candidate} has no attribute {attr!r}")
    raise ImportError(f"could not import {module!r}: {last_error}")


def _resolve_config_path(config: Optional[str], env: Optional[str]) -> Optional[str]:
    """Best-effort resolution of a config path from ``--config`` / ``--env``."""

    if config:
        if os.path.isabs(config) or os.path.exists(config):
            return config
        # allow passing just the file name or the environment name
        candidate = os.path.join(_HERE, "configs", config)
        if os.path.exists(candidate):
            return candidate
        if not config.endswith(".yaml"):
            candidate = os.path.join(_HERE, "configs", config + ".yaml")
            if os.path.exists(candidate):
                return candidate
        return config
    if env:
        name = ENVS.get(env.lower(), f"{env.lower()}.yaml")
        candidate = os.path.join(_HERE, "configs", name)
        if os.path.exists(candidate):
            return candidate
    return None


def _load_config(config: Optional[str], env: Optional[str], overrides: Optional[Sequence[str]]) -> Any:
    """Load and merge the YAML config, tolerating a missing config file."""

    path = _resolve_config_path(config, env)
    if path is None:
        return None
    try:
        load_config, apply_overrides = _import("src.common.config", "load_config"), None
        apply_overrides = _import("src.common.config", "apply_overrides")
    except ImportError:  # pragma: no cover - fallback
        load_config, apply_overrides = None, None  # type: ignore[assignment]
    if load_config is None:  # pragma: no cover - fallback
        return None
    try:
        cfg = load_config(path)
    except Exception as exc:
        _log(f"[main] warning: could not load config {path!r}: {exc}")
        return None
    if overrides and apply_overrides is not None:
        try:
            cfg = apply_overrides(cfg, list(overrides))
        except Exception as exc:  # pragma: no cover
            _log(f"[main] warning: could not apply overrides: {exc}")
    return cfg


def _forward(fn: Any, **kwargs: Any) -> Any:
    """Call ``fn`` with the subset of kwargs it accepts (tolerant signature)."""

    import inspect

    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return fn(**kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return fn(**kwargs)
    accepted = {k: v for k, v in kwargs.items() if k in signature.parameters}
    return fn(**accepted)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    for attr in ("as_dict", "to_dict"):
        method = getattr(value, attr, None)
        if callable(method):
            try:
                return _jsonable(method())
            except Exception:  # pragma: no cover - defensive
                pass
    return repr(value)


# ---------------------------------------------------------------------------
# sub-commands
# ---------------------------------------------------------------------------
def cmd_train(args: argparse.Namespace) -> int:
    env = (args.env or "").lower()
    if env == "toy":
        return cmd_toy(args)
    cfg = _load_config(args.config, env or None, args.set)
    if cfg is not None and not env:
        env = str(getattr(cfg, "env_name", "") or getattr(cfg, "name", "") or "")
    try:
        run_env = _import("train", "run_env")
    except ImportError as exc:
        _log(f"[main] error: could not import the training entry point: {exc}")
        return 2
    result = _forward(
        run_env,
        cfg=cfg,
        config=cfg,
        env=env or None,
        env_name=env or None,
        method=args.method,
        seed=args.seed,
        seeds=args.seeds,
        total_steps=args.total_steps,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        stub=args.stub,
        device=args.device,
        num_seeds=args.num_seeds,
        make_plots=not args.no_plot,
        verbose=not args.quiet,
    )
    if result is not None:
        _emit(result, args)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    env = (args.env or "").lower()
    cfg = _load_config(args.config, env or None, args.set)
    if cfg is not None and not env:
        env = str(getattr(cfg, "env_name", "") or getattr(cfg, "name", "") or "")
    try:
        run_eval = _import("evaluate", "evaluate")
    except ImportError as exc:
        _log(f"[main] error: could not import the evaluation entry point: {exc}")
        return 2
    result = _forward(
        run_eval,
        cfg=cfg,
        config=cfg,
        env=env or None,
        env_name=env or None,
        method=args.method,
        seed=args.seed,
        checkpoint=args.checkpoint,
        num_episodes=args.num_episodes,
        output_dir=args.output_dir,
        stub=args.stub,
        device=args.device,
        per_level=args.per_level,
    )
    if result is not None:
        _emit(result, args)
    return 0


def cmd_toy(args: argparse.Namespace) -> int:
    try:
        run_toy_examples = _import("src.toy", "run_toy_examples")
    except ImportError:
        try:
            run_toy_examples = _import("src.toy.two_state_mdp", "main")
            return int(run_toy_examples(list(args.set or [])) or 0)
        except ImportError as exc:
            _log(f"[main] error: toy modules unavailable: {exc}")
            return 2
    output_dir = args.output_dir or os.path.join("results", "toy")
    summary = _forward(
        run_toy_examples,
        output_dir=output_dir,
        mdp_scenarios=None,
        sweep=args.sweep,
        verbose=not args.quiet,
    )
    _emit(summary, args)
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    what = (args.what or "forward_transfer").lower()
    if what == "all":
        status = 0
        for name in ("forward_transfer", "cka", "loglikelihood", "pca", "density", "returns", "plotting"):
            sub = argparse.Namespace(**vars(args))
            sub.what = name
            status = max(status, cmd_analyze(sub))
        return status
    module_name = ANALYSIS_TARGETS.get(what)
    if module_name is None:
        _log(f"[main] error: unknown analysis target {what!r}; "
             f"choose from {sorted(set(ANALYSIS_TARGETS))} or 'all'")
        return 2
    try:
        module = _import(f"src.analysis.{module_name}")
    except ImportError as exc:
        _log(f"[main] error: analysis module {module_name!r} unavailable: {exc}")
        return 2
    entry = getattr(module, "main", None)
    if not callable(entry):
        _log(f"[main] error: {module_name!r} exposes no main()")
        return 2
    argv: List[str] = []
    if args.results_dir:
        argv += ["--results-dir", args.results_dir]
    if args.output_dir:
        argv += ["--output-dir", args.output_dir]
    if args.methods:
        argv += ["--methods", args.methods]
    if args.prefix_lengths:
        argv += ["--prefix-lengths", args.prefix_lengths]
    if args.confidence is not None:
        argv += ["--confidence", str(args.confidence)]
    if getattr(args, "step", None) is not None:
        argv += ["--step", str(args.step)]
    if getattr(args, "plot", False):
        argv += ["--plot"]
    return int(entry(argv) or 0)


def cmd_test(args: argparse.Namespace) -> int:
    try:
        main = _import("tests", "main")
    except ImportError as exc:  # pragma: no cover
        _log(f"[main] error: test package unavailable: {exc}")
        return 2
    argv: List[str] = []
    if args.quiet:
        argv.append("--quiet")
    return int(main(argv) or 0)


def cmd_analyze_all(args: argparse.Namespace) -> int:  # pragma: no cover - convenience
    sub = argparse.Namespace(**vars(args))
    sub.what = "all"
    return cmd_analyze(sub)


def _emit(result: Any, args: argparse.Namespace) -> None:
    if args.quiet:
        return
    payload = _jsonable(result)
    text = json.dumps(payload, indent=2, default=str)
    if args.json:
        print(text, flush=True)
    else:
        _log(f"[main] result:\n{text[:4000]}")


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Fine-tuning RL models is secretly a forgetting mitigation problem",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="command")

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", type=str, default=None,
                       help="path to a YAML config (or just its file/env name)")
        p.add_argument("--env", type=str, default=None, choices=sorted(ENVS),
                       help="environment to run")
        p.add_argument("--set", nargs="*", default=None, metavar="KEY=VALUE",
                       help="config overrides, e.g. sac.lr=0.0005")
        p.add_argument("--output-dir", type=str, default=None)
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--device", type=str, default=None)
        p.add_argument("--stub", action="store_true",
                       help="use the dependency-free CPU stand-in environments")
        p.add_argument("--quiet", "-q", action="store_true")
        p.add_argument("--json", action="store_true", help="print machine readable JSON")

    p_train = sub.add_parser("train", help="train / fine-tune an agent")
    add_common(p_train)
    p_train.add_argument("--method", type=str, default=None, choices=list(METHODS))
    p_train.add_argument("--seeds", nargs="*", type=int, default=None)
    p_train.add_argument("--num-seeds", type=int, default=None)
    p_train.add_argument("--total-steps", type=float, default=None)
    p_train.add_argument("--checkpoint", type=str, default=None)
    p_train.add_argument("--no-plot", action="store_true")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="evaluate a trained agent")
    add_common(p_eval)
    p_eval.add_argument("--method", type=str, default=None)
    p_eval.add_argument("--checkpoint", type=str, default=None)
    p_eval.add_argument("--num-episodes", type=int, default=None)
    p_eval.add_argument("--per-level", action="store_true",
                        help="NetHack per-level (level 4 / Sokoban) evaluation")
    p_eval.set_defaults(func=cmd_evaluate)

    p_toy = sub.add_parser("toy", help="Appendix A toy sanity checks")
    add_common(p_toy)
    p_toy.add_argument("--sweep", action="store_true", help="also sweep M and c")
    p_toy.set_defaults(func=cmd_toy)

    p_an = sub.add_parser("analyze", help="post-hoc analyses (figures/tables)")
    add_common(p_an)
    p_an.add_argument("--what", type=str, default="forward_transfer",
                      help="analysis target: " + ", ".join(sorted(set(ANALYSIS_TARGETS))) + ", all")
    p_an.add_argument("--results-dir", type=str, default=None)
    p_an.add_argument("--methods", type=str, default=None, help="comma separated method list")
    p_an.add_argument("--prefix-lengths", type=str, default=None, help="comma separated ints")
    p_an.add_argument("--confidence", type=float, default=None)
    p_an.add_argument("--step", type=int, default=None)
    p_an.add_argument("--plot", action="store_true")
    p_an.set_defaults(func=cmd_analyze)

    p_test = sub.add_parser("test", help="run the bundled smoke tests")
    p_test.add_argument("--quiet", "-q", action="store_true")
    p_test.set_defaults(func=cmd_test)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if getattr(args, "func", None) is None:
        parser.print_help()
        return 0
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
