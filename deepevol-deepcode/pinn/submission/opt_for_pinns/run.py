#!/usr/bin/env python3
"""Main entry point for the *Challenges in Training PINNs* reproduction.

Dispatches the three experiment suites described in the reproduction plan:

  compare   -> Figures 2/8 + Table 1      (optimizer comparison: Adam / L-BFGS / Adam+L-BFGS)
  spectral  -> Figures 3/7                (spectral density of H_L and of the L-BFGS-preconditioned Hessian)
  nncg      -> Figures 1/4/5 + Tables 2/3 (NysNewton-CG fine-tuning after Adam+L-BFGS stalls)
  all       -> runs the three suites in order

Usage
-----
    python run.py compare                 # full optimizer-comparison sweep
    python run.py spectral --quick        # fast smoke test of the spectral pipeline
    python run.py nncg --device cuda
    python run.py all --config configs/default.yaml
    python run.py compare --widths 50 100 --lrs 1e-3 --seeds 345
    python run.py --list                  # show the resolved experiment plan (no training)

The module only performs *dispatch and argument plumbing*: every algorithm lives
in ``src/`` and every experiment driver lives in ``experiments/``.  Optional
dependencies (spectral sub-package, plotting, pyyaml) are probed at runtime and
missing pieces cause a clear error message rather than a stack trace.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# --------------------------------------------------------------------------------------
# Path bootstrap: make ``src`` / ``experiments`` importable no matter where we are called
# --------------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

# Sub-command registry: name -> (module path, human description)
COMMANDS: Dict[str, Dict[str, str]] = {
    "compare": {
        "module": "experiments.run_optimizer_comparison",
        "description": "Optimizer comparison (Adam / L-BFGS / Adam+L-BFGS) -> Fig 2, Fig 8, Table 1",
    },
    "spectral": {
        "module": "experiments.run_spectral_density",
        "description": "Hessian spectral-density study (H_L and preconditioned) -> Fig 3, Fig 7",
    },
    "nncg": {
        "module": "experiments.run_nncg_finetune",
        "description": "NysNewton-CG fine-tuning after Adam+L-BFGS -> Fig 1, Fig 4, Fig 5, Table 2, Table 3",
    },
    "all": {
        "module": "",
        "description": "Run compare, then spectral, then nncg",
    },
}

# ---------------------------------------------------------------------------
# Optional dependency probes (kept cheap and side-effect free)
# ---------------------------------------------------------------------------


def _module_available(name: str) -> bool:
    """Return True if ``name`` can be imported."""
    try:
        importlib.import_module(name)
        return True
    except Exception:  # pragma: no cover - depends on environment
        return False


def environment_report() -> Dict[str, Any]:
    """Collect a JSON-serializable snapshot of the runtime environment.

    Used for ``--list`` and written to ``results/environment.json`` by ``main``
    so every reproduction is traceable.
    """
    report: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "project_root": str(PROJECT_ROOT),
        "packages": {},
        "src_modules": {},
    }
    for pkg in ("torch", "numpy", "scipy", "matplotlib", "yaml", "pyhessian", "tqdm", "pandas"):
        report["packages"][pkg] = _module_available(pkg)
    try:
        import torch  # noqa: F401

        torch_version = getattr(torch, "__version__", "unknown")
        report["packages"]["torch"] = True
        report["torch_version"] = torch_version
        report["cuda_available"] = bool(torch.cuda.is_available())
        report["cuda_device"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        )
    except Exception as exc:  # pragma: no cover
        report["torch_version"] = None
        report["torch_error"] = str(exc)
    for mod in (
        "src.pinns.model",
        "src.pinns.problems",
        "src.pinns.sampling",
        "src.pinns.loss",
        "src.pinns.metrics",
        "src.optimizers.first_order",
        "src.optimizers.lbfgs_wrapper",
        "src.optimizers.combined",
        "src.optimizers.armijo",
        "src.optimizers.nystrom",
        "src.optimizers.nncg",
        "src.spectral.hvp",
        "src.spectral.lbfgs_unroll",
        "src.spectral.preconditioned_mvp",
        "src.spectral.spectral_density",
        "src.utils.seeding",
        "src.utils.plotting",
    ):
        report["src_modules"][mod] = _module_available(mod)
    return report


# ---------------------------------------------------------------------------
# Config handling
# ---------------------------------------------------------------------------


def _fallback_config() -> Dict[str, Any]:
    """Minimal config used when PyYAML / configs/default.yaml are unavailable.

    Mirrors the important keys of ``configs/default.yaml`` so the runners can
    still operate (they all merge their own defaults underneath anyway).
    """
    return {
        "pdes": ["convection", "reaction", "wave"],
        "problems": {
            "convection": {"beta": 40.0, "x_min": 0.0, "x_max": 6.283185307179586,
                           "t_min": 0.0, "t_max": 1.0},
            "reaction": {"rho": 5.0, "x_min": 0.0, "x_max": 6.283185307179586,
                         "t_min": 0.0, "t_max": 1.0},
            "wave": {"beta": 5.0, "c2": 4.0, "x_min": 0.0, "x_max": 1.0,
                     "t_min": 0.0, "t_max": 1.0},
        },
        "network": {"depth": 3, "width": 200, "widths": [50, 100, 200, 400],
                    "activation": "tanh", "in_dim": 2, "out_dim": 1},
        "sampling": {"n_residual": 10000, "n_ic": 257, "n_bc": 101,
                     "n_grid_x": 255, "n_grid_t": 100, "replace": True},
        "optimizer": {
            "adam": {"lrs": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]},
            "lbfgs": {"lr": 1.0, "history_size": 100, "line_search_fn": "strong_wolfe"},
            "combined": {"switch_points": [1000, 11000, 31000], "switch_iteration": 11000,
                         "total_iterations": 41000},
            "gd": {"lr": 1e-4},
        },
        "nncg": {"eta": 1.0, "K": 2000, "s": 60, "F": 20, "mu": 1e-2,
                 "mus": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1], "epsilon": 1e-16, "M": 1000,
                 "alpha": 0.1, "beta": 0.5},
        "spectral": {"n_iter": 100, "n_vec": 1, "n_grid": 200, "backend": "native",
                     "top_k": 10, "with_components": True, "with_preconditioned": True},
        "experiment": {"seeds": [345, 456, 567, 678, 789], "widths": [50, 100, 200, 400],
                       "adam_lrs": [1e-4, 1e-3, 1e-2], "switch_iteration": 11000,
                       "total_iterations": 41000, "finetune_steps": 2000,
                       "eval_every": 500, "selection": "l2re",
                       "optimizers": ["adam", "lbfgs", "adam+lbfgs"]},
        "runtime": {"dtype": "float64", "device": "cpu", "verbose": True,
                    "make_plots": True, "save_checkpoints": True},
        "paths": {"outdir": "results", "optimizer_comparison": "results/optimizer_comparison",
                  "spectral_density": "results/spectral_density",
                  "nncg_finetune": "results/nncg_finetune", "figures": "results/figures"},
    }


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = dict(base)
    for key, value in (override or {}).items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load ``configs/default.yaml`` (or ``path``), falling back to built-ins."""
    cfg = _fallback_config()
    candidate = Path(path) if path else DEFAULT_CONFIG_PATH
    if candidate.exists():
        try:
            import yaml  # optional

            with open(candidate, "r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
            cfg = _deep_merge(cfg, loaded)
        except Exception as exc:  # pragma: no cover - env dependent
            print(f"[run.py] warning: could not parse {candidate} ({exc}); using defaults")
    elif path:
        print(f"[run.py] warning: config {candidate} not found; using built-in defaults")
    return cfg


def _quick_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Apply the config's ``quick`` block (plan: ``--quick`` smoke-test budgets)."""
    quick = cfg.get("quick") or {}
    out = _deep_merge(cfg, quick)
    # Normalise the legacy key typo found in the shipped YAML.
    opt = out.get("optimizer", {})
    combined = opt.get("combined", {}) if isinstance(opt, dict) else {}
    if isinstance(combined, dict) and "swich_points" in combined and "switch_points" not in combined:
        combined["switch_points"] = combined.pop("swich_points")
    return out


# ---------------------------------------------------------------------------
# Argument plumbing into the experiment drivers
# ---------------------------------------------------------------------------


def _parse_csv(values: Optional[Sequence[str]]) -> Optional[List[str]]:
    if not values:
        return None
    out: List[str] = []
    for value in values:
        out.extend(part for part in str(value).replace(",", " ").split() if part)
    return out or None


def _parse_floats(values: Optional[Sequence[str]]) -> Optional[List[float]]:
    parsed = _parse_csv(values)
    return [float(v) for v in parsed] if parsed else None


def _parse_ints(values: Optional[Sequence[str]]) -> Optional[List[int]]:
    parsed = _parse_csv(values)
    return [int(v) for v in parsed] if parsed else None


def _run_driver(module_name: str, argv: Sequence[str], description: str) -> int:
    """Import ``module_name`` and call its ``main(argv)`` with ``argv``.

    Returns an exit code.  Failure to import (missing optional dependency or an
    unimplemented sub-package) is reported clearly instead of propagating a raw
    traceback.
    """
    print(f"\n=== {description} ===")
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        print(f"[run.py] ERROR: cannot import '{module_name}': {type(exc).__name__}: {exc}")
        print("[run.py] hint: implement the missing module / install the missing dependency.")
        return 2
    main_fn = getattr(module, "main", None)
    if main_fn is None:
        print(f"[run.py] ERROR: '{module_name}' exposes no main(argv) entry point.")
        return 2
    started = time.time()
    try:
        main_fn(list(argv))
    except SystemExit as exc:  # argparse inside the driver
        code = exc.code if isinstance(exc.code, int) else 0
        print(f"[run.py] {description} exited with code {code}")
        return code
    except TypeError:
        # Some drivers accept keyword arguments only; fall back to a no-arg call.
        try:
            main_fn()
        except Exception as exc2:
            print(f"[run.py] ERROR in '{module_name}': {type(exc2).__name__}: {exc2}")
            return 1
    except Exception as exc:
        print(f"[run.py] ERROR in '{module_name}': {type(exc).__name__}: {exc}")
        return 1
    print(f"[run.py] {description} finished in {time.time() - started:.1f}s")
    return 0


def _driver_argv(args: argparse.Namespace, command: str) -> List[str]:
    """Translate the shared CLI options into the driver-specific argv list.

    The three drivers share the same option names (``--quick``, ``--pdes``,
    ``--widths``, ``--lrs``, ``--seeds``, ``--device``, ``--outdir``, ``--quiet``,
    ``--switch``, ``--total-iterations``, ``--no-plots``, ``--config``), so a single
    translation table is enough.
    """
    argv: List[str] = []
    if args.quick:
        argv.append("--quick")
    if args.config:
        argv += ["--config", str(args.config)]

    # ---- sweep axes (compare / spectral / nncg) ----
    if args.pdes:
        argv += ["--pdes", *args.pdes]
    if args.widths:
        argv += ["--widths", *args.widths]
    if args.lrs:
        argv += ["--lrs", *args.lrs]
    if args.seeds:
        argv += ["--seeds", *args.seeds]

    # ---- budgets ----
    if args.switch is not None and command in ("spectral", "nncg"):
        argv += ["--switch", str(args.switch)]
    if args.total_iterations is not None:
        argv += ["--total-iterations", str(args.total_iterations)]
    if args.finetune_steps is not None and command == "nncg":
        argv += ["--finetune-steps", str(args.finetune_steps)]

    # ---- optimizers (compare only) ----
    if args.optimizers and command == "compare":
        argv += ["--optimizers", *args.optimizers]

    # ---- runtime ----
    if args.device:
        argv += ["--device", str(args.device)]
    if args.outdir:
        argv += ["--outdir", str(args.outdir)]
    if args.quiet:
        argv.append("--quiet")
    if args.no_plots:
        argv.append("--no-plots")

    # Options accepted only by some drivers are filtered below.
    accepted = _driver_accepts(command)
    filtered: List[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token.startswith("--") and token[2:].replace("-", "_") not in accepted:
            # Drop this flag (and its value, if it has one that is not a flag).
            i += 1
            if i < len(argv) and not argv[i].startswith("--") and token not in _FLAGS_WITHOUT_VALUE:
                i += 1
            continue
        filtered.append(token)
        i += 1
    return filtered


_FLAGS_WITHOUT_VALUE = {"--quick", "--quiet", "--no-plots", "--no-checkpoints", "--list"}

# Options accepted by each experiment driver's argparse parser.
_SHARED_OPTIONS = {
    "quick", "config", "pdes", "widths", "lrs", "seeds", "switch",
    "total_iterations", "device", "outdir", "quiet", "no_plots",
}
_DRIVER_OPTIONS: Dict[str, set] = {
    "compare": _SHARED_OPTIONS | {"optimizers", "no_checkpoints"},
    "spectral": _SHARED_OPTIONS,
    "nncg": _SHARED_OPTIONS | {"finetune_steps", "no_checkpoints"},
}


def _driver_accepts(command: str) -> set:
    return _DRIVER_OPTIONS.get(command, _SHARED_OPTIONS)


# ---------------------------------------------------------------------------
# Plan report (``--list``)
# ---------------------------------------------------------------------------


def describe_plan(cfg: Dict[str, Any], command: str) -> str:
    """Human-readable summary of what ``command`` would run."""
    exp = cfg.get("experiment", {})
    opt = cfg.get("optimizer", {})
    combined = opt.get("combined", {}) if isinstance(opt, dict) else {}
    lines = [
        f"command            : {command}",
        f"PDEs               : {cfg.get('pdes', ['convection', 'reaction', 'wave'])}",
        f"widths             : {exp.get('widths')}",
        f"seeds              : {exp.get('seeds')}",
        f"Adam LR grid       : {(opt.get('adam', {}) or {}).get('lrs')}",
        f"Adam sweep LRs     : {exp.get('adam_lrs')}",
        f"switch points      : {combined.get('switch_points')} (default {combined.get('switch_iteration')})",
        f"total iterations   : {combined.get('total_iterations')}",
        f"fine-tune steps    : {exp.get('finetune_steps')}",
        f"selection rule     : {exp.get('selection')} over {exp.get('selection_keys')}",
        f"device / dtype     : {(cfg.get('runtime') or {}).get('device')} / {(cfg.get('runtime') or {}).get('dtype')}",
        f"output directory   : {(cfg.get('paths') or {}).get('outdir')}",
    ]
    if command in ("compare", "all"):
        lines.append(f"[compare]  optimizers = {exp.get('optimizers')}")
    if command in ("spectral", "all"):
        spec = cfg.get("spectral", {})
        lines.append(
            f"[spectral] n_iter={spec.get('n_iter')} n_vec={spec.get('n_vec')} "
            f"n_grid={spec.get('n_grid')} backend={spec.get('backend')} "
            f"components={spec.get('components', ['residual', 'initial', 'boundary'])}"
        )
    if command in ("nncg", "all"):
        ncfg = cfg.get("nncg", {})
        lines.append(
            f"[nncg]     K={ncfg.get('K')} s={ncfg.get('s')} F={ncfg.get('F')} "
            f"mu_grid={ncfg.get('mus')} M={ncfg.get('M')} "
            f"alpha={ncfg.get('alpha')} beta={ncfg.get('beta')}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description=(
            "Reproduction driver for 'Challenges in Training PINNs: A Loss Landscape Perspective'. "
            "Dispatches the optimizer-comparison, spectral-density and NysNewton-CG experiment suites."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=sorted(COMMANDS.keys()),
        help="Which experiment suite to run.",
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to the YAML config (default: configs/default.yaml).")
    parser.add_argument("--quick", action="store_true",
                        help="Run reduced budgets (smoke test) using the config's 'quick' block.")
    parser.add_argument("--pdes", nargs="+", default=None,
                        help="Subset of PDEs to run, e.g. --pdes convection wave.")
    parser.add_argument("--widths", nargs="+", default=None,
                        help="Network widths to sweep, e.g. --widths 50 100.")
    parser.add_argument("--lrs", nargs="+", default=None,
                        help="Adam learning rates to sweep, e.g. --lrs 1e-3.")
    parser.add_argument("--seeds", nargs="+", default=None,
                        help="Random seeds to sweep, e.g. --seeds 345 456.")
    parser.add_argument("--optimizers", nargs="+", default=None,
                        help="(compare) optimizers: adam lbfgs adam+lbfgs.")
    parser.add_argument("--switch", type=int, default=None,
                        help="(spectral/nncg) Adam->L-BFGS switch iteration (default 11000).")
    parser.add_argument("--total-iterations", type=int, default=None, dest="total_iterations",
                        help="Total optimisation iterations per run (default 41000).")
    parser.add_argument("--finetune-steps", type=int, default=None, dest="finetune_steps",
                        help="(nncg) Number of NNCG / GD fine-tuning steps (default 2000).")
    parser.add_argument("--device", type=str, default=None,
                        help="Torch device: 'cpu' or 'cuda' (default from config).")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Root output directory for results/figures (default from config).")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip figure generation (matplotlib may be unavailable).")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging verbosity.")
    parser.add_argument("--list", action="store_true",
                        help="Print the resolved experiment plan and environment report, then exit.")
    parser.add_argument("--json-summary", type=str, default=None,
                        help="Optional path; write the environment/plan report as JSON here.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.pdes = _parse_csv(args.pdes)
    args.widths = _parse_ints(args.widths)
    args.lrs = _parse_csv(args.lrs)
    args.seeds = _parse_ints(args.seeds)
    args.optimizers = _parse_csv(args.optimizers)

    cfg = load_config(args.config)
    if args.quick:
        cfg = _quick_overrides(cfg)

    # CLI overrides take precedence over the config file.
    if args.device:
        cfg.setdefault("runtime", {})["device"] = args.device
    if args.outdir:
        cfg.setdefault("paths", {})["outdir"] = args.outdir
    if args.total_iterations is not None:
        cfg.setdefault("optimizer", {}).setdefault("combined", {})[
            "total_iterations"
        ] = args.total_iterations

    if not args.quiet:
        print("=" * 78)
        print("PINN loss-landscape reproduction :: 'Challenges in Training PINNs'")
        print("=" * 78)
        print(describe_plan(cfg, args.command))

    env = environment_report()

    if args.list:
        print("\n--- environment ---")
        for key in ("python", "torch_version", "cuda_available", "cuda_device"):
            if key in env:
                print(f"{key:20s}: {env[key]}")
        print("packages           :", {k: v for k, v in env["packages"].items()})
        missing = [m for m, ok in env["src_modules"].items() if not ok]
        print("missing src modules:", missing if missing else "none")
        if args.json_summary:
            payload = {"plan": describe_plan(cfg, args.command), "environment": env}
            _write_json(Path(args.json_summary), payload)
        return 0

    # Persist the environment report next to the other results (best effort).
    outdir = Path((cfg.get("paths") or {}).get("outdir", "results"))
    try:
        _write_json(outdir / "environment.json", env)
    except Exception:
        pass

    if args.json_summary:
        try:
            _write_json(Path(args.json_summary), {"plan": describe_plan(cfg, args.command),
                                                  "environment": env})
        except Exception:
            pass

    commands: List[str] = ["compare", "spectral", "nncg"] if args.command == "all" else [args.command]
    driver_argv = _driver_argv(args, args.command) if args.command != "all" else None

    exit_code = 0
    for command in commands:
        info = COMMANDS[command]
        argv_for_driver = driver_argv if command == args.command and driver_argv is not None \
            else _driver_argv(args, command)
        code = _run_driver(info["module"], argv_for_driver, info["description"])
        exit_code = exit_code or code

    if not args.quiet:
        print("\n" + "=" * 78)
        print(f"All requested suites finished (exit code {exit_code}).")
        print(f"Results written under: {outdir.resolve()}")
        print("  optimizer comparison : Table 1, Fig 2, Fig 8")
        print("  spectral density     : Fig 3, Fig 7")
        print("  NNCG fine-tuning     : Fig 1, Fig 4, Fig 5, Table 2, Table 3")
        print("=" * 78)
    return exit_code


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


if __name__ == "__main__":
    raise SystemExit(main())
