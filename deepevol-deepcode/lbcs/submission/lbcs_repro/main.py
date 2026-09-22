"""Primary executable entry point for the LBCS reproduction.

This module is the *glue layer* of the reproduction (plan item 19): it parses YAML
configurations (with ``inherit`` / deep-merge semantics), applies CLI overrides,
sets seeds, configures logging, dispatches to the experiment drivers registered in
:mod:`lbcs_repro.experiments`, and persists the resolved configuration plus a run
summary.  It contains no paper-specific formula: every algorithm lives in
:mod:`lbcs_repro.lbcs`, :mod:`lbcs_repro.baselines`, :mod:`lbcs_repro.models`,
:mod:`lbcs_repro.data` and :mod:`lbcs_repro.experiments`.

Scope (plan "Out of scope"): no ImageNet-1k (Section 5.4), no continual learning
(Appendix E.5), no streaming (Appendix E.6).

Usage examples
--------------
    python -m lbcs_repro.main --list
    python -m lbcs_repro.main --experiment figure1 --config lbcs_repro/configs/figure1.yaml
    python -m lbcs_repro.main --experiment table1 --smoke
    python -m lbcs_repro.main --experiment all --smoke
    python -m lbcs_repro.main --selftest
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger("lbcs_repro.main")

# ---------------------------------------------------------------------------
# paths / defaults
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(_HERE, "configs")
DEFAULT_CONFIG_NAME = "default.yaml"
DEFAULT_OUTPUT_ROOT = "results"

#: experiment id -> config file shipped with the reproduction
EXPERIMENT_CONFIGS: Dict[str, str] = {
    "figure1": "figure1.yaml",
    "table1": "section5_1.yaml",
    "table2": "section5_2.yaml",
    "table3": "section5_2.yaml",
    "figure2": "section5_3.yaml",
    "table8": "section5_3.yaml",
    "table9": "section6.yaml",
    "table5": "section6.yaml",
    "table6": "section6.yaml",
    "appendix_c3": "figure1.yaml",
}

#: keys that must never be inherited from a parent config
_NO_MERGE_KEYS = ("inherit", "inherits")


# ---------------------------------------------------------------------------
# YAML / deep-merge helpers (glue only)
# ---------------------------------------------------------------------------


def _yaml():
    """Return the ``yaml`` module or raise a helpful error."""
    try:
        import yaml  # type: ignore

        return yaml
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "PyYAML is required to read configuration files "
            "(pip install pyyaml). Error: {0}".format(exc)
        )


def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``overlay`` into ``base`` (overlay wins).

    Lists are replaced, not concatenated, so a section config can fully control a
    sweep such as ``ks: [...]``.  ``base`` is never mutated.
    """
    if not isinstance(base, dict):
        base = {}
    if not isinstance(overlay, dict):
        return copy.deepcopy(base)

    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in _NO_MERGE_KEYS:
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def parse_overrides(pairs: Optional[Iterable[str]]) -> Dict[str, Any]:
    """Parse ``key.sub=value`` CLI overrides into a nested dict.

    Values are parsed as YAML scalars when possible (so ``T=500`` is an int and
    ``warm_start=false`` is a bool), falling back to raw strings.
    """
    result: Dict[str, Any] = {}
    if not pairs:
        return result

    yaml = None
    try:
        yaml = _yaml()
    except ImportError:  # pragma: no cover
        yaml = None

    for item in pairs:
        if "=" not in item:
            LOGGER.warning("Ignoring malformed override %r (expected key=value)", item)
            continue
        dotted, raw = item.split("=", 1)
        dotted = dotted.strip()
        if not dotted:
            continue
        if yaml is not None:
            try:
                value: Any = yaml.safe_load(raw)
            except Exception:
                value = raw
        else:  # pragma: no cover - minimal fallback
            value = raw
            lowered = raw.strip().lower()
            if lowered in ("true", "false"):
                value = lowered == "true"
            elif lowered in ("none", "null", "~"):
                value = None
            else:
                try:
                    value = int(raw)
                except ValueError:
                    try:
                        value = float(raw)
                    except ValueError:
                        value = raw

        node = result
        parts = [p for p in dotted.split(".") if p]
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return result


def resolve_config_path(name_or_path: Optional[str]) -> Optional[str]:
    """Resolve a config name/path against the bundled ``configs/`` directory."""
    if not name_or_path:
        return None
    if os.path.isfile(name_or_path):
        return os.path.abspath(name_or_path)
    candidate = os.path.join(CONFIG_DIR, name_or_path)
    if os.path.isfile(candidate):
        return os.path.abspath(candidate)
    if not name_or_path.endswith((".yaml", ".yml")):
        candidate = os.path.join(CONFIG_DIR, name_or_path + ".yaml")
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return None


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file into a dict (empty file -> ``{}``)."""
    yaml = _yaml()
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


def _load_with_inheritance(path: str, seen: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Load ``path`` resolving its ``inherit`` / ``inherits`` parents (deep-merge)."""
    seen = list(seen or [])
    abspath = os.path.abspath(path)
    if abspath in seen:
        LOGGER.warning("Circular config inheritance detected at %s; ignoring", abspath)
        return {}
    seen.append(abspath)

    data = load_yaml(abspath)
    parents: List[str] = []
    for key in _NO_MERGE_KEYS:
        raw = data.get(key)
        if not raw:
            continue
        if isinstance(raw, str):
            raw = [raw]
        parents.extend(str(item) for item in raw)
    if not parents:
        return data

    merged: Dict[str, Any] = {}
    for parent in parents:
        parent_path = resolve_config_path(parent)
        if parent_path is None:
            parent_path = os.path.join(os.path.dirname(abspath), parent)
        if not os.path.isfile(parent_path):
            LOGGER.warning("Inherited config %r not found; skipping", parent)
            continue
        merged = deep_merge(merged, _load_with_inheritance(parent_path, seen))
    merged = deep_merge(merged, data)
    for key in _NO_MERGE_KEYS:
        merged.pop(key, None)
    return merged


def load_config(
    name_or_path: Optional[str] = None,
    *,
    overrides: Optional[Dict[str, Any]] = None,
    experiment: Optional[str] = None,
    use_default: bool = True,
) -> Dict[str, Any]:
    """Load and merge a configuration.

    Resolution order (later wins):
      1. ``configs/default.yaml`` (unless ``use_default=False``);
      2. the experiment's own section config (if ``experiment`` given and no
         explicit ``name_or_path``);
      3. ``name_or_path``;
      4. ``overrides``.
    """
    config: Dict[str, Any] = {}

    if use_default:
        default_path = os.path.join(CONFIG_DIR, DEFAULT_CONFIG_NAME)
        if os.path.isfile(default_path):
            config = deep_merge(config, _load_with_inheritance(default_path))
        else:  # pragma: no cover - packaging dependent
            LOGGER.warning("default config not found at %s", default_path)

    path = resolve_config_path(name_or_path)
    if path is None and name_or_path:
        LOGGER.warning("Config %r not found; continuing with defaults", name_or_path)

    if path is None and experiment:
        mapped = EXPERIMENT_CONFIGS.get(experiment)
        if mapped:
            path = resolve_config_path(mapped)

    if path is not None:
        config = deep_merge(config, _load_with_inheritance(path))

    if overrides:
        config = deep_merge(config, overrides)
    return config


# ---------------------------------------------------------------------------
# runtime helpers
# ---------------------------------------------------------------------------


def resolve_output_root(config: Optional[Dict[str, Any]] = None, override: Optional[str] = None) -> str:
    """Determine the results directory (CLI > config > default)."""
    if override:
        return override
    if config:
        for key in ("output_root", "output_dir", "results_dir"):
            value = config.get(key)
            if value:
                return str(value)
    return DEFAULT_OUTPUT_ROOT


def apply_runtime_config(config: Dict[str, Any], *, seed: Optional[int] = None, device: Optional[str] = None) -> Dict[str, Any]:
    """Fold CLI-level ``seed``/``device`` into the config dict."""
    if seed is not None:
        config["seed"] = int(seed)
    if device is not None:
        config["device"] = device
    return config


def setup_logging_from_config(config: Dict[str, Any], *, verbose: bool = False) -> logging.Logger:
    """Configure logging from the config ``logging`` block (best effort)."""
    try:
        from lbcs_repro.utils.logging import LoggingConfig, setup_logging
    except Exception:  # pragma: no cover - torch-free / partial installs
        try:
            from utils.logging import LoggingConfig, setup_logging  # type: ignore
        except Exception:
            level = "DEBUG" if verbose else str(config.get("logging", {}).get("level", "INFO"))
            logging.basicConfig(level=getattr(logging, level, logging.INFO))
            return logging.getLogger("lbcs_repro")

    block = config.get("logging") or {}
    log_config = LoggingConfig.from_dict(block)
    if verbose:
        log_config = log_config.with_overrides(level="DEBUG")
    if config.get("output_root"):
        log_config = log_config.with_overrides(output_dir=str(config.get("output_root")))
    return setup_logging(config=log_config)


def set_seed_everything(seed: Optional[int]) -> Optional[int]:
    """Seed all RNGs through the shared utility layer (best effort)."""
    if seed is None:
        seed = 0
    try:
        from lbcs_repro.utils.seed import set_seed
    except Exception:  # pragma: no cover
        try:
            from utils.seed import set_seed  # type: ignore
        except Exception:
            import random

            import numpy as np

            random.seed(int(seed))
            np.random.seed(int(seed) % (2**32))
            try:
                import torch

                torch.manual_seed(int(seed))
                if torch.cuda.is_available():  # pragma: no cover
                    torch.cuda.manual_seed_all(int(seed))
            except Exception:
                pass
            return int(seed)
    try:
        return set_seed(int(seed))
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("set_seed failed (%s); continuing", exc)
        return int(seed)


def _experiments_module():
    """Import the experiment registry package (with a path fallback)."""
    try:
        from lbcs_repro import experiments as experiments_pkg

        return experiments_pkg
    except Exception:  # pragma: no cover - direct script execution
        sys.path.insert(0, os.path.dirname(_HERE))
        from lbcs_repro import experiments as experiments_pkg  # type: ignore

        return experiments_pkg


def available_experiments(include_aliases: bool = False) -> List[str]:
    """Canonical experiment ids known to the registry."""
    try:
        return list(_experiments_module().available_experiments(include_aliases=include_aliases))
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Experiment registry unavailable (%s); using fallback list", exc)
        return sorted(EXPERIMENT_CONFIGS)


def out_of_scope() -> Dict[str, str]:
    """Sections explicitly excluded from the reproduction."""
    try:
        return dict(_experiments_module().OUT_OF_SCOPE)
    except Exception:  # pragma: no cover
        return {
            "section5.4": "ImageNet-1k experiments (out of scope)",
            "appendix_e.5": "Continual learning (out of scope)",
            "appendix_e.6": "Streaming coreset selection (out of scope)",
        }


def repeats_for(section: str) -> int:
    """Paper repeat protocol for a section key (e.g. ``section5.1``)."""
    try:
        return int(_experiments_module().REPEATS.get(section, 1))
    except Exception:  # pragma: no cover
        table = {"section5.1": 20, "section5.2": 10, "section5.3": 10, "section6": 10}
        return int(table.get(section, 1))


def _expand_kwargs(config: Dict[str, Any], experiment: str) -> Dict[str, Any]:
    """Build the kwargs forwarded to a driver entrypoint.

    Drivers document their own keyword surface; we forward the resolved config,
    the experiment id, and the nested blocks most drivers accept so that a config
    file can drive a driver without code changes.
    """
    kwargs: Dict[str, Any] = {"config": config, "experiment": experiment}
    for key in (
        "seed",
        "device",
        "num_workers",
        "data_root",
        "output_dir",
        "output_root",
        "save_artifacts",
        "verbose",
        "smoke",
    ):
        if key in config:
            kwargs[key] = config[key]

    section_key = {
        "figure1": "figure1",
        "table1": "table1",
        "table2": "table2",
        "table3": "table2",
        "table2_table3": "table2",
        "figure2": "robustness",
        "table8": "robustness",
        "table9": "table9",
        "table5": "table5",
        "table6": "table6",
        "appendix_c3": "figure1",
    }.get(experiment)
    if section_key and isinstance(config.get(section_key), dict):
        kwargs["section"] = config[section_key]
    return kwargs


def run_experiment(
    experiment: str,
    config: Optional[Dict[str, Any]] = None,
    *,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Dispatch a single experiment id through the registry."""
    logger = logger or LOGGER
    config = config or {}
    pkg = _experiments_module()
    spec = pkg.experiment_spec(experiment)
    logger.info("[main] running experiment '%s' -> %s.%s", spec["id"], spec["module"], spec["entrypoint"])
    started = time.time()
    kwargs = _expand_kwargs(config, spec["id"])
    result = pkg.run_experiment(spec["id"], **kwargs)
    elapsed = time.time() - started
    logger.info("[main] experiment '%s' finished in %.1fs", spec["id"], elapsed)
    return {"experiment": spec["id"], "module": spec["module"], "entrypoint": spec["entrypoint"], "wall_time": elapsed, "result": result}


def _json_default(obj: Any) -> Any:
    """Best-effort JSON encoder used for the run summary."""
    for attr in ("to_dict", "tolist", "item"):
        if hasattr(obj, attr):
            try:
                value = getattr(obj, attr)
                return value() if callable(value) else value
            except Exception:
                continue
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    return str(obj)


def save_run_summary(summary: Dict[str, Any], output_dir: str, name: str = "main_run") -> Optional[str]:
    """Persist the run summary (resolved config + dispatch record) as JSON."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, name + ".json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, default=_json_default)
        return path
    except Exception as exc:  # pragma: no cover - filesystem dependent
        LOGGER.warning("Could not write run summary: %s", exc)
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lbcs_repro.main",
        description=(
            "Refined Coreset Selection / LBCS reproduction entry point "
            "(Figure 1, Table 1, Tables 2-3, Figure 2, Tables 5/6/8/9, Appendix C.3)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--experiment", "-e", default=None, help="Experiment id (or 'all'); see --list.")
    parser.add_argument("--config", "-c", default=None, help="Config file path or name under configs/.")
    parser.add_argument("--output-dir", "--output-root", dest="output_dir", default=None, help="Results directory.")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed.")
    parser.add_argument("--device", default=None, help="Torch device string (e.g. cuda, cuda:0, cpu).")
    parser.add_argument("--repeats", type=int, default=None, help="Override the number of repeats.")
    parser.add_argument("--smoke", action="store_true", help="Use fast smoke-test settings.")
    parser.add_argument("--paper", action="store_true", help="Use the full paper protocol (explicit).")
    parser.add_argument("--set", dest="overrides", action="append", default=None, metavar="KEY=VALUE", help="Override a config key (dotted path), repeatable.")
    parser.add_argument("--list", action="store_true", help="List available experiments and exit.")
    parser.add_argument("--no-save", action="store_true", help="Do not write artifacts/run summary.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose (DEBUG) logging.")
    parser.add_argument("--selftest", action="store_true", help="Run the offline self-test and exit.")
    return parser


def _apply_smoke(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply the ``smoke`` block of a config over the base values."""
    smoke = config.get("smoke")
    if isinstance(smoke, dict):
        return deep_merge(config, smoke)
    return config


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    if args.list:
        _print_experiment_list()
        return 0

    if args.selftest:
        report = _selftest(verbose=True)
        return 0 if report.get("ok") else 1

    if not args.experiment:
        build_argparser().print_help()
        return 2

    overrides = parse_overrides(args.overrides)
    if args.repeats is not None:
        overrides = deep_merge(
            overrides,
            {"repeats": int(args.repeats), "table1": {"repeats": int(args.repeats)}, "table2": {"repeats": int(args.repeats)}},
        )
    if args.smoke:
        overrides = deep_merge(overrides, {"smoke": True})
    if args.paper:
        overrides = deep_merge(overrides, {"smoke": False, "paper": True})

    config = load_config(args.config, overrides=overrides, experiment=args.experiment)
    if args.smoke:
        config = _apply_smoke(config)
    config = apply_runtime_config(config, seed=args.seed, device=args.device)
    config.setdefault("seed", 0)

    logger = setup_logging_from_config(config, verbose=args.verbose)
    set_seed_everything(config.get("seed"))

    output_root = resolve_output_root(config, args.output_dir)
    config["output_root"] = output_root

    for key, value in out_of_scope().items():
        logger.info("[main] out of scope: %s (%s)", key, value)

    logger.info("=" * 78)
    logger.info("LBCS reproduction: experiment=%s seed=%s device=%s", args.experiment, config.get("seed"), config.get("device"))
    logger.info("=" * 78)

    if str(args.experiment).lower() in ("all", "*", "everything"):
        records = run_all_experiments(config, output_root=output_root, logger=logger)
    else:
        records = [run_experiment(args.experiment, config, logger=logger)]

    summary = {
        "experiments": records,
        "config": config,
        "seed": config.get("seed"),
        "device": config.get("device"),
        "output_root": output_root,
        "out_of_scope": out_of_scope(),
    }
    if not args.no_save and (config.get("save_artifacts", True) is not False):
        path = save_run_summary(summary, output_root)
        if path:
            logger.info("[main] wrote %s", path)
    return 0


def run_all_experiments(
    config: Optional[Dict[str, Any]] = None,
    *,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """Run every canonical in-scope experiment with the same base config.

    Failures are recorded, not raised, so a long reproduction run can proceed to
    the remaining experiments.
    """
    logger = logger or LOGGER
    records: List[Dict[str, Any]] = []
    for name in available_experiments():
        try:
            records.append(run_experiment(name, config, logger=logger))
        except Exception as exc:  # pragma: no cover - long-running driver dependent
            logger.exception("[main] experiment '%s' failed: %s", name, exc)
            records.append({"experiment": name, "failed": True, "error": str(exc)})
    if not (config or {}).get("save_artifacts", True) is False:
        save_run_summary({"experiments": records, "config": config or {}}, output_root, name="run_all_main")
    return records


def _print_experiment_list() -> None:
    lines = ["Available experiments (canonical ids and aliases):", ""]
    try:
        pkg = _experiments_module()
        for canonical in pkg.CANONICAL_EXPERIMENTS:
            spec = pkg.experiment_spec(canonical)
            lines.append("  {0:<12} {1}".format(canonical, spec["description"]))
        lines.append("")
        lines.append("Aliases: " + ", ".join(sorted(pkg.available_experiments(include_aliases=True))))
        lines.append("")
        lines.append("Repeat protocol: " + ", ".join("{0}={1}".format(k, v) for k, v in pkg.REPEATS.items()))
    except Exception:
        for canonical in sorted(EXPERIMENT_CONFIGS):
            lines.append("  {0:<12} (config: {1})".format(canonical, EXPERIMENT_CONFIGS[canonical]))
    lines.append("")
    lines.append("Out of scope (not executed):")
    for key, note in out_of_scope().items():
        lines.append("  {0:<14} {1}".format(key, note))
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# self-test (offline, no data / no GPU required)
# ---------------------------------------------------------------------------


def _selftest(verbose: bool = True) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}

    merged = deep_merge({"a": {"b": 1, "c": [1, 2]}, "d": 2}, {"a": {"c": [3]}})
    checks["deep_merge_nested"] = merged == {"a": {"b": 1, "c": [3]}, "d": 2}
    checks["deep_merge_no_mutation"] = deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}

    parsed = parse_overrides(["T=500", "lbcs.epsilon=0.3", "smoke=false"])
    checks["parse_overrides"] = parsed["T"] == 500 and parsed["lbcs"]["epsilon"] == 0.3 and parsed["smoke"] is False

    default_path = os.path.join(CONFIG_DIR, DEFAULT_CONFIG_NAME)
    checks["default_config_exists"] = os.path.isfile(default_path)

    try:
        config = load_config(None, experiment="figure1")
        checks["figure1_config_loads"] = isinstance(config, dict) and bool(config)
        checks["figure1_has_lbcs_block"] = isinstance(config.get("lbcs"), dict)
    except Exception as exc:
        checks["figure1_config_loads"] = False
        checks["figure1_config_error"] = str(exc)

    try:
        names = available_experiments()
        checks["registry_available"] = len(names) > 0
        checks["scope_excludes_imagenet"] = all("imagenet" not in n.lower() for n in names)
    except Exception as exc:
        checks["registry_available"] = False
        checks["registry_error"] = str(exc)

    checks["repeats_section5_1"] = repeats_for("section5.1") == 20
    checks["out_of_scope_recorded"] = set(out_of_scope()) >= {"section5.4", "appendix_e.5", "appendix_e.6"}

    ok = all(bool(v) for k, v in checks.items() if not k.endswith("_error"))
    report = {"checks": checks, "ok": ok}
    if verbose:
        print("[main] selftest")
        for key, value in sorted(checks.items()):
            print("  {0:<28} {1}".format(key, value))
        print("  overall:", "OK" if ok else "FAILED")
    return report


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
