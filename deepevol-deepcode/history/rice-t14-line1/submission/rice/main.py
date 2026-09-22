#!/usr/bin/env python3
"""RICE: single command-line entry point.

RICE (A Refining scheme for ReInForcement learning with Explanation, ICML 2024,
PMLR 235) refines a pre-trained (bottlenecked) DRL policy by

  * identifying *critical states* (exploration frontiers) with a re-designed
    StateMask mask network                       -- Algorithm 1 (CORE COMPONENT 1)
  * building a mixed initial-state distribution
    ``mu(s) = beta * d_rho^pi_hat(s) + (1 - beta) * rho(s)``,
    realised as a Bernoulli(p) roll-in           -- CORE COMPONENT 2
  * adding a normalized Random Network Distillation exploration bonus
    ``R' = R + lambda * ||f(s_{t+1}) - f_hat(s_{t+1})||^2``   -- CORE COMPONENT 3
  * continuing PPO training on the resulting augmented objective
                                                 -- Algorithm 2 (CORE COMPONENT 4)

This module is a *thin dispatcher*: every sub-command forwards the remaining
argv to the script / evaluation module that owns the real implementation.  It
never imports torch/gym at module import time, so ``python main.py list`` works
on a bare CPU-only install.

Usage
-----
    python main.py list                      # show envs / tasks / methods
    python main.py pretrain   --task Hopper-v3 --seeds 0 1 2
    python main.py train-mask --task Hopper-v3 --seeds 0 1 2
    python main.py fidelity   --task Hopper-v3 --explanations ours statemask random
    python main.py refine     --task Hopper-v3 --methods ours ppo statemask_r
    python main.py baselines  --tasks all --methods ours ppo jsrl statemask_r
    python main.py sac-gail   --task Hopper-v3 --enable-sac      # Experiment IV
    python main.py sweep      --param p --task Hopper-v3 --values 0 0.25 0.5 0.75 1
    python main.py all        -- --seeds 0 1 2                   # run everything
    python main.py test                                          # pytest suite

Each sub-command accepts ``-h`` and delegates to the owning script's own
argparse parser, e.g. ``python main.py train-mask --help``.
"""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Path bootstrap: support the three repository layouts
#   (a) installed package              -> `import rice`
#   (b) repo root on sys.path          -> main.py at <repo>/main.py
#   (c) nested layout                  -> main.py at <repo>/rice/main.py
# ---------------------------------------------------------------------------
_HERE: Path = Path(__file__).resolve()
_REPO_ROOT: Path = _HERE.parent
_PACKAGE_ROOT: Path = _REPO_ROOT / "rice"  # inner import root (rice/rice/...)
_CANDIDATE_ROOTS: Tuple[Path, ...] = (_REPO_ROOT, _PACKAGE_ROOT, _REPO_ROOT.parent)


def _bootstrap_paths() -> List[str]:
    """Append discoverable import roots to ``sys.path`` (returns what was added)."""
    added: List[str] = []
    for root in _CANDIDATE_ROOTS:
        try:
            if root.is_dir():
                path = str(root)
                if path not in sys.path:
                    sys.path.append(path)
                    added.append(path)
        except OSError:  # pragma: no cover - defensive
            continue
    return added


_BOOTSTRAPPED: List[str] = _bootstrap_paths()


# ---------------------------------------------------------------------------
# Sub-command table.  ``None`` module => handled natively in this file.
# ---------------------------------------------------------------------------
SUBCOMMANDS: Tuple[Tuple[str, Optional[str], str], ...] = (
    (
        "pretrain",
        "rice.scripts.pretrain_agent",
        "Pre-train the warm-start (bottlenecked) policies pi for each task",
    ),
    (
        "train-mask",
        "rice.scripts.train_mask",
        "Algorithm 1: train the mask network (Table 3 alpha, Table 4 budget)",
    ),
    (
        "fidelity",
        "rice.scripts.fidelity_eval",
        "Experiment I: explanation fidelity scores (Fig. 5) + mask timing (Table 4)",
    ),
    (
        "refine",
        "rice.scripts.run_refine",
        "Algorithm 2: refine a pre-trained agent (Experiments II / III)",
    ),
    (
        "baselines",
        "rice.scripts.run_baselines",
        "Experiments II-IV: RICE vs PPO-FT / StateMask-R / JSRL / SIL / SAC",
    ),
    (
        "sac-gail",
        "rice.scripts.run_sac_gail",
        "Experiment IV: SAC pre-train + GAIL imitation, then refine (ours vs baselines)",
    ),
    ("sweep", None, "Experiment V: sweep p / lambda / alpha sensitivity"),
    ("all", None, "Run every experiment (3 seeds) via scripts/run_all_experiments.sh"),
    ("list", None, "List environments, tasks, refining methods and explanations"),
    ("test", None, "Run the unit / behavioural test-suite with pytest"),
)

# Candidate dotted-module paths per script sub-command (layout tolerance:
# `rice.scripts.X`, nested `rice.rice.scripts.X`, and bare `scripts.X`).
_SCRIPT_MODULE_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "pretrain": (
        "rice.scripts.pretrain_agent",
        "rice.rice.scripts.pretrain_agent",
        "scripts.pretrain_agent",
    ),
    "train-mask": (
        "rice.scripts.train_mask",
        "rice.rice.scripts.train_mask",
        "scripts.train_mask",
    ),
    "fidelity": (
        "rice.scripts.fidelity_eval",
        "rice.rice.scripts.fidelity_eval",
        "scripts.fidelity_eval",
    ),
    "refine": (
        "rice.scripts.run_refine",
        "rice.rice.scripts.run_refine",
        "scripts.run_refine",
    ),
    "baselines": (
        "rice.scripts.run_baselines",
        "rice.rice.scripts.run_baselines",
        "scripts.run_baselines",
    ),
    "sac-gail": (
        "rice.scripts.run_sac_gail",
        "rice.rice.scripts.run_sac_gail",
        "scripts.run_sac_gail",
    ),
}

# Friendly aliases -> canonical sub-command name.
_ALIASES: Dict[str, str] = {
    "pre-train": "pretrain",
    "pre_train": "pretrain",
    "warmstart": "pretrain",
    "warm-start": "pretrain",
    "mask": "train-mask",
    "train-mask-net": "train-mask",
    "train_mask": "train-mask",
    "mask-train": "train-mask",
    "fidelity-eval": "fidelity",
    "fidelity_eval": "fidelity",
    "exp1": "fidelity",
    "exp-i": "fidelity",
    "experiment-i": "fidelity",
    "refine-agent": "refine",
    "run-refine": "refine",
    "exp2": "refine",
    "exp3": "refine",
    "exp-ii": "refine",
    "exp-iii": "refine",
    "baseline": "baselines",
    "run-baselines": "baselines",
    "exp4": "baselines",
    "exp-iv": "baselines",
    "sac": "sac-gail",
    "sac_gail": "sac-gail",
    "gail": "sac-gail",
    "sensitivity": "sweep",
    "hyperparam-sweep": "sweep",
    "exp5": "sweep",
    "exp-v": "sweep",
    "experiment-v": "sweep",
    "reproduce": "all",
    "run-all": "all",
    "experiments": "all",
    "envs": "list",
    "tasks": "list",
    "tests": "test",
    "pytest": "test",
}

_CANONICAL: Tuple[str, ...] = tuple(name for name, _, _ in SUBCOMMANDS)


def canonical_command(name: str) -> str:
    """Normalise a user-supplied sub-command spelling to a canonical name."""
    key = (name or "").strip().lower().replace("_", "-").replace(" ", "-")
    key = _ALIASES.get(key, key)
    if key not in _CANONICAL:
        raise KeyError(
            "unknown command %r; choose one of: %s"
            % (name, ", ".join(_CANONICAL))
        )
    return key


def _import_first(candidates: Sequence[str]) -> Optional[object]:
    """Import the first importable module from ``candidates`` (else ``None``)."""
    for name in candidates:
        try:
            return importlib.import_module(name)
        except Exception:  # ImportError / SyntaxError / missing deps
            continue
    return None


def _import_optional(candidates: Sequence[str]) -> Optional[object]:
    return _import_first(candidates)


# ---------------------------------------------------------------------------
# Native sub-commands
# ---------------------------------------------------------------------------
def _cmd_list(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="main.py list",
        description="List RICE environments, tasks, refining methods and explanations.",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args(list(argv))

    payload: Dict[str, List[str]] = {
        "environments": [],
        "dense_tasks": [],
        "sparse_tasks": [],
        "out_of_scope": [],
        "refining_methods": [],
        "explanations": [],
        "commands": list(_CANONICAL),
    }

    envs = _import_optional(
        (
            "rice.environments",
            "rice.rice.environments",
            "environments",
        )
    )
    if envs is not None:
        for attr in ("available_envs",):
            fn = getattr(envs, attr, None)
            if callable(fn):
                try:
                    payload["environments"] = list(fn())
                except Exception:
                    pass
        for attr in ("DENSE_ENVS", "SPARSE_ENVS", "ALL_ENVS"):
            value = getattr(envs, attr, None)
            if value:
                if attr == "DENSE_ENVS":
                    payload["dense_tasks"] = list(value)
                elif attr == "SPARSE_ENVS":
                    payload["sparse_tasks"] = list(value)

    refining = _import_optional(
        (
            "rice.evaluation.refining_eval",
            "rice.rice.evaluation.refining_eval",
            "evaluation.refining_eval",
        )
    )
    if refining is not None:
        for attr, bucket in (
            ("DENSE_TASKS", "dense_tasks"),
            ("SPARSE_TASKS", "sparse_tasks"),
            ("OUT_OF_SCOPE_TASKS", "out_of_scope"),
            ("REFINING_METHODS", "refining_methods"),
            ("EXPLANATION_METHODS", "explanations"),
        ):
            value = getattr(refining, attr, None)
            if value and not payload.get(bucket):
                payload[bucket] = list(value)

    explained = _import_optional(
        (
            "rice.explanation",
            "rice.rice.explanation",
            "explanation",
        )
    )
    if explained is not None and not payload["explanations"]:
        value = getattr(explained, "EXPLANATION_NAMES", None)
        if value:
            payload["explanations"] = list(value)

    if args.json:
        import json

        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    title = "RICE v%s  (ICML 2024, PMLR 235)" % __version__
    print(title)
    print("=" * len(title))
    print("\nCommand-line sub-commands:")
    for name, _, description in SUBCOMMANDS:
        print("  %-11s %s" % (name, description))
    for label, values in (
        ("Environments", payload["environments"]),
        ("Dense tasks (Experiment II)", payload["dense_tasks"]),
        ("Sparse tasks (Experiment II, curves)", payload["sparse_tasks"]),
        ("Out of scope", payload["out_of_scope"]),
        ("Refining methods", payload["refining_methods"]),
        ("Explanations", payload["explanations"]),
    ):
        if values:
            print("\n%s:" % label)
            for value in values:
                print("  - %s" % value)
    print("\nRun `python main.py <command> --help` for a sub-command's own options.")
    return 0


def _cmd_sweep(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="main.py sweep",
        description=(
            "Experiment V: hyper-parameter sensitivity.  The paper sweeps "
            "p in {0,0.25,0.5,0.75,1}, lambda in {0,0.1,0.01,0.001} and "
            "alpha in {0.01,0.001,0.0001}."
        ),
    )
    parser.add_argument("--param", default="p", help="p | lambda | lambda_ | alpha | beta")
    parser.add_argument("--task", default="Hopper-v3", help="task name (canonical or alias)")
    parser.add_argument(
        "--values",
        nargs="*",
        type=float,
        default=None,
        help="explicit sweep values; omitted -> the paper's grid for the parameter",
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="random seeds")
    parser.add_argument(
        "--mode",
        default="auto",
        choices=("auto", "refine", "fidelity"),
        help="measurement backend (alpha defaults to fidelity, p/lambda to refine)",
    )
    parser.add_argument("--iterations", type=int, default=None, help="refining outer iterations per value")
    parser.add_argument("--eval-episodes", type=int, default=None, help="evaluation episodes")
    parser.add_argument("--explanation", default="ours", help="explanation used for refining")
    parser.add_argument("--out-dir", default=os.path.join("runs", "sweep"), help="artifact directory")
    parser.add_argument("--plot", action="store_true", help="write Figures 7-9 style PNGs")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    args, unknown = parser.parse_known_args(list(argv))
    if unknown:
        print("[main.py sweep] ignoring unrecognised arguments: %s" % " ".join(unknown))

    sweep_module = _import_optional(
        (
            "rice.evaluation.hyperparam_sweep",
            "rice.rice.evaluation.hyperparam_sweep",
            "evaluation.hyperparam_sweep",
        )
    )
    if sweep_module is None:
        print(
            "[main.py sweep] rice.evaluation.hyperparam_sweep is unavailable "
            "(check RICE_BOOTSTRAPPED paths: %s)" % _BOOTSTRAPPED,
            file=sys.stderr,
        )
        return 1

    config_cls = getattr(sweep_module, "SweepConfig", None)
    run_sweep = getattr(sweep_module, "run_sweep", None)
    canonical_param = getattr(sweep_module, "canonical_param", None)
    if run_sweep is None:
        print("[main.py sweep] run_sweep() not found in hyperparam_sweep", file=sys.stderr)
        return 1

    param = canonical_param(args.param) if callable(canonical_param) else args.param
    kwargs: Dict[str, object] = {}
    if config_cls is not None:
        try:
            kwargs["config"] = config_cls(
                task=args.task,
                param=param,
                mode=args.mode,
                explanation=args.explanation,
                save_dir=args.out_dir,
                plot=bool(args.plot),
                verbose=0 if args.quiet else 1,
                **(
                    {"n_iterations": args.iterations}
                    if args.iterations is not None
                    else {}
                ),
                **(
                    {"eval_episodes": args.eval_episodes}
                    if args.eval_episodes is not None
                    else {}
                ),
                **({"seeds": tuple(args.seeds)} if args.seeds else {}),
            )
        except Exception as exc:  # keep going with kwargs-only configuration
            print("[main.py sweep] could not build SweepConfig (%s); using defaults" % exc)

    result = run_sweep(
        param=param,
        task=args.task,
        values=tuple(args.values) if args.values else None,
        **kwargs,
    )

    if args.json:
        import json

        payload = result.as_dict() if hasattr(result, "as_dict") else dict(result)
        print(json.dumps(payload, indent=2, default=str, sort_keys=True))
    else:
        formatter = getattr(result, "format", None) or getattr(result, "table", None)
        if callable(formatter):
            try:
                print(formatter())
            except TypeError:
                print(formatter(10))
        else:
            print(result)

    saver = getattr(result, "as_dict", None)
    if callable(saver):
        try:
            os.makedirs(args.out_dir, exist_ok=True)
            import json

            out = os.path.join(args.out_dir, "sweep_%s_%s.json" % (param, args.task))
            with open(out, "w", encoding="utf-8") as handle:
                json.dump(result.as_dict(), handle, indent=2, default=str)
            print("[main.py sweep] wrote %s" % out)
        except Exception as exc:  # pragma: no cover - artifact writing is best effort
            print("[main.py sweep] could not write artifacts: %s" % exc, file=sys.stderr)

    trend = getattr(result, "trend_check", None)
    if callable(trend):
        try:
            verdict = trend()
            if isinstance(verdict, dict) and verdict:
                print("[main.py sweep] trend check: %s" % verdict.get("status", verdict))
        except Exception:
            pass
    return 0


def _cmd_all(argv: Sequence[str]) -> int:
    """Run every experiment through the orchestrator shell script."""
    script_candidates = (
        _REPO_ROOT / "scripts" / "run_all_experiments.sh",
        _PACKAGE_ROOT / "scripts" / "run_all_experiments.sh",
        _REPO_ROOT.parent / "scripts" / "run_all_experiments.sh",
    )
    script = next((p for p in script_candidates if p.is_file()), None)
    if script is None:
        print(
            "[main.py all] run_all_experiments.sh not found; looked in:\n  %s"
            % "\n  ".join(str(p) for p in script_candidates),
            file=sys.stderr,
        )
        return 1

    extra = list(argv)
    if extra and extra[0] == "--":
        extra = extra[1:]
    command = ["bash", str(script)] + extra
    print("[main.py all] $ %s" % " ".join(command))
    try:
        return int(subprocess.call(command, cwd=str(script.parent.parent)))
    except OSError as exc:
        print("[main.py all] failed to execute: %s" % exc, file=sys.stderr)
        return 1


def _cmd_test(argv: Sequence[str]) -> int:
    """Run the pytest suite (dependency-tolerant: skips when torch/gym missing)."""
    tests_dirs = (
        _PACKAGE_ROOT / "tests",
        _REPO_ROOT / "tests",
        _REPO_ROOT.parent / "tests",
    )
    tests_dir = next((p for p in tests_dirs if p.is_dir()), None)
    extra = list(argv)
    if extra and extra[0] == "--":
        extra = extra[1:]
    if tests_dir is None:
        print("[main.py test] tests/ directory not found; looked in:\n  %s"
              % "\n  ".join(str(p) for p in tests_dirs), file=sys.stderr)
        return 1
    command = [sys.executable, "-m", "pytest", str(tests_dir), "-q"] + extra
    print("[main.py test] $ %s" % " ".join(command))
    try:
        return int(subprocess.call(command, cwd=str(_REPO_ROOT)))
    except OSError as exc:
        print("[main.py test] failed to execute pytest: %s" % exc, file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# Dispatch of the script-backed sub-commands
# ---------------------------------------------------------------------------
def _run_script(command: str, argv: Sequence[str]) -> int:
    candidates = _SCRIPT_MODULE_CANDIDATES.get(command, ())
    module = _import_first(candidates) if candidates else None
    if module is None:
        print(
            "[main.py %s] could not import %s\n  sys.path roots added: %s"
            % (command, " / ".join(candidates), _BOOTSTRAPPED),
            file=sys.stderr,
        )
        return 1
    main_fn = getattr(module, "main", None)
    if not callable(main_fn):
        print("[main.py %s] %s exposes no main()" % (command, module.__name__), file=sys.stderr)
        return 1
    try:
        result = main_fn(list(argv))
    except SystemExit as exc:  # argparse -h / bad arguments
        code = exc.code
        if code is None:
            return 0
        return int(code) if isinstance(code, int) else 1
    if isinstance(result, bool):
        return int(result)
    if isinstance(result, int):
        return result
    return 0


def _print_overview() -> None:
    print("RICE v%s -- Refining scheme for ReInforcement learning with Explanation" % __version__)
    print("ICML 2024 (PMLR 235).  Reference: https://github.com/chengzelei/RICE\n")
    print("Sub-commands:")
    for name, module, description in SUBCOMMANDS:
        target = module or "(native)"
        print("  %-11s %-40s %s" % (name, target, description))
    print("\nTypical reproduction order (each forwards to its own script):")
    print("  1. python main.py pretrain   --task all            # warm-start policies pi")
    print("  2. python main.py train-mask --task all            # Algorithm 1 (mask net)")
    print("  3. python main.py fidelity   --task all            # Experiment I (+ Table 4 timing)")
    print("  4. python main.py baselines  --tasks all            # Experiments II / III")
    print("  5. python main.py sac-gail   --task Hopper-v3 --enable-sac   # Experiment IV")
    print("  6. python main.py sweep      --param p --task Hopper-v3      # Experiment V")
    print("Or simply: python main.py all -- --seeds 0 1 2")
    print("\nOut of scope (never a success criterion): Malware Mutation, SparseWalker2d")
    print("refining, sparse hyper-parameter sweeps, autonomous-driving qualitative")
    print("analysis, and the Section 3.4 / Appendix B theory code.")
    print("\nRun `python main.py <command> --help` for the options of a sub-command.")


def build_parser() -> argparse.ArgumentParser:
    """Top-level parser (a sub-command name plus forwarded raw argv)."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", nargs="?", help="sub-command to run (see below)")
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="arguments forwarded verbatim to the sub-command",
    )
    parser.add_argument("--version", action="version", version="RICE %s" % __version__)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.  Returns a process exit status."""
    raw: List[str] = list(sys.argv[1:] if argv is None else argv)

    if not raw or raw[0] in ("-h", "--help"):
        _print_overview()
        return 0
    if raw[0] in ("-v", "--version"):
        print("RICE %s" % __version__)
        return 0

    command_token, rest = raw[0], raw[1:]
    try:
        command = canonical_command(command_token)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        _print_overview()
        return 2

    if command == "list":
        return _cmd_list(rest)
    if command == "sweep":
        return _cmd_sweep(rest)
    if command == "all":
        return _cmd_all(rest)
    if command == "test":
        return _cmd_test(rest)
    return _run_script(command, rest)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
