"""Command line entry point for the Robust CLIP reproduction package.

This module only wires the individual harnesses together.  It contains no
paper-specific formula: all algorithm details live in the modules it dispatches
to and all hyperparameters are supplied through the YAML configs (or CLI
overrides).  Whenever the paper/Addendum is silent about a value the harness
logs it as an externally supplied default instead of inventing a paper value.

Usage
-----
    python -m robust_clip_repro.main imagenet   --config configs/pgd_eval.yaml --eps 4/255
    python -m robust_clip_repro.main vqa        --config configs/vqa_attack.yaml --dataset TextVQA
    python -m robust_clip_repro.main captioning --config configs/captioning.yaml --eps 2/255
    python -m robust_clip_repro.main jailbreak  --config configs/jailbreak.yaml
    python -m robust_clip_repro.main train      --config configs/robust_clip.yaml
    python -m robust_clip_repro.main smoke      # model-free end-to-end checks
    python -m robust_clip_repro.main assets     # prefetch jailbreak assets
    python -m robust_clip_repro.main summary    # print provenance information
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "MODE_IMAGENET",
    "MODE_VQA",
    "MODE_CAPTIONING",
    "MODE_JAILBREAK",
    "MODE_TRAIN",
    "MODES",
    "DEFAULT_CONFIGS",
    "build_arg_parser",
    "main",
    "run_mode",
]

LOGGER = logging.getLogger("robust_clip_repro.main")

PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = PACKAGE_DIR / "configs"

MODE_IMAGENET = "imagenet"
MODE_VQA = "vqa"
MODE_CAPTIONING = "captioning"
MODE_JAILBREAK = "jailbreak"
MODE_TRAIN = "train"
MODE_SMOKE = "smoke"
MODE_ASSETS = "assets"
MODE_SUMMARY = "summary"

MODES = (
    MODE_IMAGENET,
    MODE_VQA,
    MODE_CAPTIONING,
    MODE_JAILBREAK,
    MODE_TRAIN,
    MODE_SMOKE,
    MODE_ASSETS,
    MODE_SUMMARY,
)

#: Default config file per mode.  All attack budgets (eps/alpha/iterations) are
#: deliberately left unset inside these files when the Addendum does not state
#: them, and must be supplied on the command line.
DEFAULT_CONFIGS: Dict[str, Optional[str]] = {
    MODE_IMAGENET: "pgd_eval.yaml",
    MODE_VQA: "vqa_attack.yaml",
    MODE_CAPTIONING: "captioning.yaml",
    MODE_JAILBREAK: "jailbreak.yaml",
    MODE_TRAIN: "robust_clip.yaml",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _setup_logging(verbose: bool = True) -> None:
    level = logging.INFO if verbose else logging.WARNING
    root = logging.getLogger("robust_clip_repro")
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
        )
        root.addHandler(handler)
    root.setLevel(level)


def resolve_config_path(mode: str, config: Optional[str]) -> Optional[str]:
    """Resolve a user supplied config path, falling back to the packaged one."""
    if config:
        path = Path(config)
        if path.exists():
            return str(path)
        candidate = CONFIG_DIR / config
        if candidate.exists():
            return str(candidate)
        LOGGER.warning("config %s not found; proceeding without it", config)
        return str(path)
    default = DEFAULT_CONFIGS.get(mode)
    if default:
        candidate = CONFIG_DIR / default
        if candidate.exists():
            return str(candidate)
        LOGGER.info("no default config for mode %s (looked for %s)", mode, candidate)
    return None


def parse_fraction(value: Any) -> Optional[float]:
    """Parse ``"4/255"`` / ``"0.0156"`` style epsilon specifications."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        if "/" in text:
            num, den = text.split("/", 1)
            return float(num) / float(den)
        return float(text)
    except (TypeError, ValueError):
        LOGGER.warning("could not parse numeric value %r; ignoring", value)
        return None


def load_config_dict(path: Optional[str]) -> Dict[str, Any]:
    """Best-effort YAML load; returns ``{}`` when unavailable."""
    if not path:
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:  # pragma: no cover - PyYAML always available in the env
        LOGGER.warning("PyYAML not installed; config %s ignored", path)
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except FileNotFoundError:
        LOGGER.warning("config file %s does not exist", path)
        return {}
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("failed to parse config %s: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _section(cfg: Dict[str, Any], *names: str) -> Dict[str, Any]:
    """Return the first nested section present, merged with ``cfg`` itself."""
    merged: Dict[str, Any] = {}
    for name in names:
        block = cfg.get(name)
        if isinstance(block, dict):
            merged.update(block)
    for key, value in cfg.items():
        if key not in names and not isinstance(value, dict):
            merged.setdefault(key, value)
    return merged


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "as_dict"):
        try:
            return _jsonable(obj.as_dict())
        except Exception:  # pragma: no cover - defensive
            return repr(obj)
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:  # pragma: no cover - defensive
            return repr(obj)
    return repr(obj)


def print_json(payload: Any) -> None:
    print(json.dumps(_jsonable(payload), indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# mode handlers
# ---------------------------------------------------------------------------
def run_imagenet(args: argparse.Namespace) -> int:
    from . import eval_imagenet

    argv: List[str] = []
    if args.config:
        argv += ["--config", args.config]
    if args.model:
        argv += ["--model", args.model]
    if args.attack:
        argv += ["--attack", args.attack]
    if args.norm:
        argv += ["--norm", args.norm]
    eps = parse_fraction(args.eps)
    if eps is not None:
        argv += ["--eps", repr(eps)]
    alpha = parse_fraction(args.alpha)
    if alpha is not None:
        argv += ["--alpha", repr(alpha)]
    if args.iterations is not None:
        argv += ["--iterations", str(args.iterations)]
    if args.num_samples is not None:
        argv += ["--num-samples", str(args.num_samples)]
    if args.output:
        argv += ["--output", args.output]
    if args.device:
        argv += ["--device", args.device]
    if args.extra:
        argv += list(args.extra)
    return eval_imagenet.main(argv)


def run_vqa(args: argparse.Namespace) -> int:
    from . import eval_vqa

    argv: List[str] = []
    if args.config:
        argv += ["--config", args.config]
    if args.dataset:
        argv += ["--dataset", args.dataset]
    if args.model:
        argv += ["--model", args.model]
    for flag, value in (
        ("--eps", parse_fraction(args.eps)),
        ("--alpha", parse_fraction(args.alpha)),
        ("--low-eps", parse_fraction(args.low_eps)),
        ("--high-eps", parse_fraction(args.high_eps)),
        ("--targeted-eps", parse_fraction(args.targeted_eps)),
    ):
        if value is not None:
            argv += [flag, repr(value)]
    if args.iterations is not None:
        argv += ["--iterations", str(args.iterations)]
    if args.num_samples is not None:
        argv += ["--num-samples", str(args.num_samples)]
    if args.output:
        argv += ["--output", args.output]
    if args.device:
        argv += ["--device", args.device]
    if args.extra:
        argv += list(args.extra)
    return eval_vqa.main(argv)


def run_captioning(args: argparse.Namespace) -> int:
    from . import eval_captioning

    argv: List[str] = []
    if args.config:
        argv += ["--config", args.config]
    if args.dataset:
        argv += ["--dataset", args.dataset]
    if args.model:
        argv += ["--model", args.model]
    eps = parse_fraction(args.eps)
    if eps is not None:
        argv += ["--eps", repr(eps)]
    if args.num_samples is not None:
        argv += ["--num-samples", str(args.num_samples)]
    if args.output:
        argv += ["--output", args.output]
    if args.device:
        argv += ["--device", args.device]
    if args.extra:
        argv += list(args.extra)
    return eval_captioning.main(argv)


def run_jailbreak(args: argparse.Namespace) -> int:
    from . import eval_jailbreak

    argv: List[str] = []
    if args.config:
        argv += ["--config", args.config]
    if args.model:
        argv += ["--model", args.model]
    if args.num_samples is not None:
        argv += ["--num-samples", str(args.num_samples)]
    if args.output:
        argv += ["--output", args.output]
    if args.device:
        argv += ["--device", args.device]
    if args.extra:
        argv += list(args.extra)
    return eval_jailbreak.main(argv)


def run_train(args: argparse.Namespace) -> int:
    from . import train_robust_clip

    argv: List[str] = []
    if args.config:
        argv += ["--config", args.config]
    if args.output:
        argv += ["--output", args.output]
    if args.device:
        argv += ["--device", args.device]
    if args.extra:
        argv += list(args.extra)
    return train_robust_clip.main(argv)


def run_smoke(args: argparse.Namespace) -> int:
    """Model-free checks of every Addendum invariant implemented in the repo."""
    results: Dict[str, Any] = {}
    failures: List[str] = []

    packages = [
        ("precision", "utils.precision", "_self_test"),
        ("pgd", "attacks.pgd", "_self_test"),
        ("apgd", "attacks.apgd", "_self_test"),
        ("jailbreak_attack", "attacks.jailbreak", "_self_test"),
        ("vqa_schedule", "attacks.vqa_schedule", "_self_test"),
        ("captioning_attack", "attacks.captioning", "_self_test"),
        ("cider", "metrics.cider", "_self_test"),
        ("classification", "metrics.classification", "_self_test"),
        ("vqa_metrics", "metrics.vqa", "_self_test"),
        ("jailbreak_metrics", "metrics.jailbreak", "_self_test"),
        ("imagenet_data", "data.imagenet", None),
        ("captions_data", "data.coco_captioning", "_self_test"),
        ("benchmarks_data", "data.benchmarks", "_self_test"),
    ]

    for name, module_path, fn_name in packages:
        try:
            module = __import__("robust_clip_repro." + module_path, fromlist=["_self_test"])
        except Exception as exc:  # pragma: no cover - defensive
            results[name] = f"IMPORT FAILED: {exc}"
            failures.append(f"{name}: import error {exc}")
            continue
        fn = getattr(module, fn_name, None) if fn_name else None
        if fn is None:
            results[name] = "imported (no self-test)"
            continue
        try:
            outcome = fn(verbose=False) if "verbose" in getattr(fn, "__code__", type("", (), {"co_varnames": ()})).co_varnames else fn()
            results[name] = "ok" if outcome is not False else "FAILED"
            if outcome is False:
                failures.append(f"{name}: self-test returned False")
        except TypeError:
            try:
                outcome = fn()
                results[name] = "ok" if outcome is not False else "FAILED"
                if outcome is False:
                    failures.append(f"{name}: self-test returned False")
            except Exception as exc:
                results[name] = f"FAILED: {exc}"
                failures.append(f"{name}: {exc}")
        except Exception as exc:
            results[name] = f"FAILED: {exc}"
            failures.append(f"{name}: {exc}")

    for name, runner in (
        ("eval_imagenet", "eval_imagenet.run_smoke_test"),
        ("eval_vqa", "eval_vqa.run_smoke_test"),
        ("eval_captioning", "eval_captioning.run_smoke_test"),
        ("eval_jailbreak", "eval_jailbreak.run_smoke_test"),
    ):
        try:
            module_name, attr = runner.rsplit(".", 1)
            module = __import__("robust_clip_repro." + module_name, fromlist=[attr])
            fn = getattr(module, attr)
            try:
                fn(verbose=False)
            except TypeError:
                fn()
            results[name] = "ok"
        except Exception as exc:
            results[name] = f"FAILED: {exc}"
            failures.append(f"{name}: {exc}")

    print_json({"results": results, "failures": failures})
    if failures:
        LOGGER.error("%d smoke-test failure(s)", len(failures))
        return 1
    LOGGER.info("all smoke tests passed")
    return 0


def run_assets(args: argparse.Namespace) -> int:
    try:
        from .data import jailbreak_assets
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.error("cannot import jailbreak assets module: %s", exc)
        return 1
    try:
        paths = jailbreak_assets.ensure_all_assets(
            assets_dir=args.assets_dir, allow_download=not args.no_download
        )
    except Exception as exc:
        LOGGER.error("asset fetch failed: %s", exc)
        return 1
    print_json({k: str(v) for k, v in paths.items()})
    return 0


def run_summary(args: argparse.Namespace) -> int:
    """Print the provenance of every config: paper-stated vs. external."""
    payload: Dict[str, Any] = {"configs": {}, "repository": "https://github.com/haotian-liu/LLaVA"}
    for mode, filename in DEFAULT_CONFIGS.items():
        if not filename:
            continue
        cfg = load_config_dict(str(CONFIG_DIR / filename))
        block = _section(cfg, mode, "eval", "attack", "training", "model")
        payload["configs"][mode] = {
            "path": str(CONFIG_DIR / filename),
            "provenance": block.get("provenance", cfg.get("provenance", {})),
        }
    print_json(payload)
    return 0


_HANDLERS = {
    MODE_IMAGENET: run_imagenet,
    MODE_VQA: run_vqa,
    MODE_CAPTIONING: run_captioning,
    MODE_JAILBREAK: run_jailbreak,
    MODE_TRAIN: run_train,
    MODE_SMOKE: run_smoke,
    MODE_ASSETS: run_assets,
    MODE_SUMMARY: run_summary,
}


def run_mode(args: argparse.Namespace) -> int:
    handler = _HANDLERS.get(args.mode)
    if handler is None:
        LOGGER.error("unknown mode %s (choose from %s)", args.mode, ", ".join(MODES))
        return 2
    return handler(args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robust_clip_repro",
        description=(
            "Robust CLIP reproduction: unsupervised adversarial fine-tuning of vision "
            "embeddings for robust large vision-language models."
        ),
    )
    parser.add_argument("mode", choices=MODES, help="which component to run")
    parser.add_argument("--config", default=None, help="path to a YAML config file")
    parser.add_argument("--model", default=None, help="victim model name/path")
    parser.add_argument("--dataset", default=None, help="dataset name (VQA / captioning mode)")
    parser.add_argument("--attack", default=None, help="attacker: pgd | apgd")
    parser.add_argument("--norm", default=None, help="perturbation norm: linf | l2")
    parser.add_argument(
        "--eps",
        default=None,
        help="perturbation budget, e.g. 4/255 (UNSPECIFIED by the Addendum for general PGD)",
    )
    parser.add_argument("--alpha", default=None, help="step size (defaults to eps where applicable)")
    parser.add_argument(
        "--low-eps", default=None, help="VQA low/half-precision stage budget (external default)"
    )
    parser.add_argument(
        "--high-eps", default=None, help="VQA high/single-precision stage budget (external default)"
    )
    parser.add_argument(
        "--targeted-eps", default=None, help="VQA targeted stage budget (external default)"
    )
    parser.add_argument("--iterations", type=int, default=None, help="attack iterations")
    parser.add_argument("--num-samples", type=int, default=None, help="limit the number of samples")
    parser.add_argument("--output", default=None, help="output file for the JSON result")
    parser.add_argument("--device", default=None, help="torch device, e.g. cuda:0")
    parser.add_argument("--assets-dir", default=None, help="jailbreak asset cache directory")
    parser.add_argument(
        "--no-download", action="store_true", help="never download assets (cache only)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    parser.add_argument(
        "extra", nargs="*", help="extra arguments forwarded verbatim to the sub-command"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    _setup_logging(verbose=args.verbose or os.environ.get("ROBUST_CLIP_VERBOSE") == "1")
    args.config = resolve_config_path(args.mode, args.config)
    if args.config:
        LOGGER.info("using config %s", args.config)
    try:
        return run_mode(args)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        LOGGER.warning("interrupted")
        return 130


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
