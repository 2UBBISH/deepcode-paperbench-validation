#!/usr/bin/env python
"""Orchestrator for every in-scope LBCS reproduction experiment.

This is the top-level glue layer (plan item 19, "scripts/run_all.py"):
it resolves the requested experiments + YAML configuration, applies
CLI overrides, then dispatches each experiment to its driver through the
``lbcs_repro.experiments`` registry, collecting a single reproducible run
report (JSON + plain text) under ``<output_root>/``.

In-scope experiments (paper artefacts)
--------------------------------------
* ``figure1``       -- Figure 1 / Appendix C.3 trivial-solution study (Eq. (3) vs Eq. (4))
* ``table1``        -- Section 5.1 Table 1 preliminary superiority (MNIST-S)
* ``table2``        -- Section 5.2 Table 2 competitor comparison
* ``table3``        -- Section 5.2 Table 3 matched-size comparison
* ``figure2``       -- Section 5.3 Figure 2 robustness against imperfect supervision
* ``table8``        -- Appendix E.3 Table 8 optimized coreset sizes under imperfect supervision
* ``table9``        -- Section 6 / Appendix E.4 Table 9 T-sweep
* ``table5``        -- Section 6 Table 5 mask initialization (LBCS + Moderate)
* ``table6``        -- Section 6 Table 6 cross-architecture (ViT-small / WideResNet on SVHN)
* ``appendix_c3``   -- Appendix C.3 settings + probabilistic gradient-norm details

Explicitly OUT OF SCOPE (never dispatched here)
-----------------------------------------------
* Section 5.4      -- ImageNet-1k
* Appendix E.5     -- continual learning
* Appendix E.6     -- streaming

Usage
-----
::

    python -m lbcs_repro.scripts.run_all --list
    python -m lbcs_repro.scripts.run_all --experiments table1 --smoke
    python -m lbcs_repro.scripts.run_all --groups section5.2 section5.3
    python -m lbcs_repro.scripts.run_all --all
    python -m lbcs_repro.scripts.run_all --selftest

Nothing in this file contains paper-specific formulas; all algorithm code
lives in ``lbcs_repro/lbcs``, ``lbcs_repro/baselines``, ``lbcs_repro/models``
and the individual ``experiments`` drivers.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import inspect
import json
import logging
import os
import sys
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Make ``lbcs_repro`` importable when this file is executed directly
# (``python lbcs_repro/scripts/run_all.py``) instead of as a module.
# --------------------------------------------------------------------------
if __package__ in (None, ""):  # pragma: no cover - bootstrap only
    _HERE = os.path.dirname(os.path.abspath(__file__))
    _ROOT = os.path.dirname(os.path.dirname(_HERE))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

LOGGER = logging.getLogger("lbcs_repro.scripts.run_all")

# --------------------------------------------------------------------------
# Experiment bookkeeping
# --------------------------------------------------------------------------

#: Canonical run order for the orchestrator.  Mirrors the paper's flow
#: (Figure 1 -> Section 5.1 -> Section 5.2 -> Section 5.3 -> Section 6).
ALL_EXPERIMENTS: Tuple[str, ...] = (
    "figure1",
    "table1",
    "table2",
    "table3",
    "figure2",
    "table8",
    "table9",
    "table5",
    "table6",
    "appendix_c3",
)

#: Named groups so users can request a whole paper section at once.
GROUPS: Dict[str, Tuple[str, ...]] = {
    "figure1": ("figure1",),
    "c3": ("appendix_c3", "figure1"),
    "appendix_c3": ("appendix_c3",),
    "section5.1": ("table1",),
    "section5_1": ("table1",),
    "section5.2": ("table2", "table3"),
    "section5_2": ("table2", "table3"),
    "section5.3": ("figure2", "table8"),
    "section5_3": ("figure2", "table8"),
    "section6": ("table5", "table6", "table9"),
    "core": ("figure1", "table1", "appendix_c3"),
    "all": ALL_EXPERIMENTS,
}

#: Sections deliberately not reproduced (paper-external to this plan).
OUT_OF_SCOPE: Dict[str, str] = {
    "section5.4": "ImageNet-1k experiments (not reproduced)",
    "appendix_e.5": "continual learning experiments (not reproduced)",
    "appendix_e.6": "streaming coreset selection (not reproduced)",
}

#: Map experiment id -> config section key holding its protocol block.
EXPERIMENT_SECTIONS: Dict[str, str] = {
    "figure1": "figure1",
    "table1": "table1",
    "table2": "table2",
    "table3": "table2",
    "figure2": "robustness",
    "table8": "robustness",
    "table9": "table9",
    "table5": "table5",
    "table6": "table6",
    "appendix_c3": "appendix_c3",
}

#: Driver module + entrypoint fallbacks if the registry is unavailable.
EXPERIMENT_ENTRYPOINTS: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "figure1": ("lbcs_repro.experiments.figure1_trivial", ("run_figure1", "run")),
    "table1": ("lbcs_repro.experiments.table1_prelim", ("run_table1", "run")),
    "table2": ("lbcs_repro.experiments.table2_table3_compare", ("run_table2", "run")),
    "table3": ("lbcs_repro.experiments.table2_table3_compare", ("run_table3", "run")),
    "figure2": ("lbcs_repro.experiments.figure2_robustness", ("run_figure2", "run")),
    "table8": ("lbcs_repro.experiments.table8_sizes", ("run_table8", "run")),
    "table9": ("lbcs_repro.experiments.table9_search_times", ("run_table9", "run")),
    "table5": ("lbcs_repro.experiments.table5_init", ("run_table5", "run")),
    "table6": ("lbcs_repro.experiments.table6_cross_arch", ("run_table6", "run")),
    "appendix_c3": ("lbcs_repro.experiments.appendix_c3", ("run_appendix_c3", "run")),
}

#: Map experiment id -> config filename used for its protocol block.
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

#: Paper statistical repeat protocol (also mirrors utils.seed.PAPER_REPEATS).
PAPER_REPEATS: Dict[str, int] = {
    "section5.1": 20,
    "section5.2": 10,
    "section5.3": 10,
    "section6": 10,
}

DEFAULT_OUTPUT_ROOT = "results"
DEFAULT_CONFIG_NAME = "default.yaml"
_NO_MERGE_KEYS = ("inherit", "inherits")

# --------------------------------------------------------------------------
# Small helpers (torch/driver import fallbacks)
# --------------------------------------------------------------------------


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def config_dir() -> str:
    """Directory containing the bundled YAML configs."""
    return os.path.join(_repo_root(), "configs")


def _import_first(candidates: Sequence[Tuple[str, str]]) -> Optional[Any]:
    """Import the first ``(module, attribute)`` pair that resolves."""
    for module_name, attr in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception:  # pragma: no cover - optional import
            continue
        obj = getattr(module, attr, None)
        if obj is not None:
            return obj
    return None


def out_of_scope() -> Dict[str, str]:
    """Return the mapping of sections explicitly excluded from this reproduction."""
    return dict(OUT_OF_SCOPE)


def repeats_for(section: str) -> int:
    """Paper repeat count for a section key (``section5.1`` etc.)."""
    return int(PAPER_REPEATS.get(section, 1))


def available_experiments(include_aliases: bool = False) -> List[str]:
    """Experiment ids the orchestrator knows about (canonical order first)."""
    try:  # prefer the package registry so aliases stay authoritative
        registry = importlib.import_module("lbcs_repro.experiments")
        names = registry.available_experiments(include_aliases=include_aliases)
        if names:
            return list(names)
    except Exception:  # pragma: no cover - registry optional
        pass
    if include_aliases:
        return sorted(set(ALL_EXPERIMENTS) | set(GROUPS))
    return list(ALL_EXPERIMENTS)


# --------------------------------------------------------------------------
# Config plumbing
# --------------------------------------------------------------------------


def deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``overlay`` into ``base`` (overlay wins).

    ``inherit``/``inherits`` keys are dropped from the result.  Lists are
    replaced wholesale (paper protocols are explicit sequences).
    """
    if not isinstance(base, dict):
        base = {}
    result: Dict[str, Any] = {}
    for key, value in base.items():
        if key in _NO_MERGE_KEYS:
            continue
        result[key] = copy.deepcopy(value)
    if not isinstance(overlay, dict):
        return result
    for key, value in overlay.items():
        if key in _NO_MERGE_KEYS:
            continue
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def parse_overrides(pairs: Optional[Iterable[str]]) -> Dict[str, Any]:
    """Parse ``key.sub=value`` CLI overrides into a nested dict."""
    if not pairs:
        return {}
    try:  # reuse main.py when available (identical semantics)
        parser = _import_first([("lbcs_repro.main", "parse_overrides")])
        if callable(parser):
            return parser(pairs)
    except Exception:  # pragma: no cover
        pass

    out: Dict[str, Any] = {}
    for raw in pairs:
        if "=" not in raw:
            LOGGER.warning("Ignoring malformed override %r (expected key=value)", raw)
            continue
        dotted, _, value = raw.partition("=")
        node = out
        parts = [p for p in dotted.strip().split(".") if p]
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1] if parts else dotted] = _coerce_scalar(value)
    return out


def _coerce_scalar(value: str) -> Any:
    """YAML-coerce a CLI value, falling back to the raw string."""
    text = value.strip()
    if yaml is not None:
        try:
            return yaml.safe_load(text)
        except Exception:
            pass
    lowered = text.lower()
    if lowered in ("true", "yes"):
        return True
    if lowered in ("false", "no"):
        return False
    if lowered in ("none", "null", "~"):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return value


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file into a dict (empty/unsupported -> ``{}``)."""
    if yaml is None:
        raise RuntimeError("PyYAML is required to load configuration files")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}


def resolve_config_path(name_or_path: str) -> Optional[str]:
    """Resolve a config name/path against the bundled ``configs/`` directory."""
    if not name_or_path:
        return None
    if os.path.exists(name_or_path):
        return os.path.abspath(name_or_path)
    stem = name_or_path if name_or_path.endswith((".yaml", ".yml")) else name_or_path + ".yaml"
    candidate = os.path.join(config_dir(), stem)
    return candidate if os.path.exists(candidate) else None


def load_config(
    experiment: Optional[str] = None,
    config: Optional[str] = None,
    *,
    overrides: Optional[Dict[str, Any]] = None,
    use_default: bool = True,
    smoke: bool = False,
) -> Dict[str, Any]:
    """Merge ``default.yaml`` + experiment config + explicit path + overrides.

    Delegates to :func:`lbcs_repro.main.load_config` when importable so the
    CLI and this orchestrator share one precedence rule; otherwise performs
    the equivalent deep-merge locally.
    """
    try:
        loader = _import_first([("lbcs_repro.main", "load_config")])
        if callable(loader) and config is None and overrides is None:
            merged = loader(experiment=experiment, use_default=use_default)
            if isinstance(merged, dict):
                if smoke:
                    merged = _apply_smoke(merged)
                return merged
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.debug("main.load_config unavailable (%s); using local merge", exc)

    merged: Dict[str, Any] = {}
    if use_default:
        default_path = os.path.join(config_dir(), DEFAULT_CONFIG_NAME)
        if os.path.exists(default_path):
            try:
                merged = deep_merge(merged, load_yaml(default_path))
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Could not load %s: %s", default_path, exc)

    candidates: List[str] = []
    if experiment:
        name = EXPERIMENT_CONFIGS.get(experiment)
        if name:
            candidates.append(name)
    if config:
        candidates.append(config)

    for candidate in candidates:
        path = resolve_config_path(candidate)
        if path is None:
            LOGGER.warning("Config %r not found; skipping", candidate)
            continue
        try:
            merged = deep_merge(merged, load_yaml(path))
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Could not load %s: %s", path, exc)

    if overrides:
        merged = deep_merge(merged, overrides)
    if smoke:
        merged = _apply_smoke(merged)
    return merged


def _apply_smoke(config: Dict[str, Any]) -> Dict[str, Any]:
    """Fold the config's ``smoke`` block into the top level (CPU fast path)."""
    smoke_block = config.get("smoke")
    if isinstance(smoke_block, dict):
        return deep_merge(config, smoke_block)
    return config


def section_config(config: Dict[str, Any], experiment: str) -> Dict[str, Any]:
    """Extract the protocol block a driver expects.

    Drivers accept either a full config dict or their own section dict; we
    hand over the whole (merged) config plus a convenience section alias so
    both call styles work.
    """
    section_key = EXPERIMENT_SECTIONS.get(experiment, experiment)
    out: Dict[str, Any] = dict(config)
    block = config.get(section_key)
    if isinstance(block, dict):
        out.setdefault("section", section_key)
        out[section_key] = block
    return out


# --------------------------------------------------------------------------
# Driver dispatch
# --------------------------------------------------------------------------


def _call_with_supported_kwargs(fn: Callable[..., Any], kwargs: Dict[str, Any]) -> Any:
    """Call ``fn`` with the subset of ``kwargs`` it accepts.

    Handles drivers with explicit signatures (filtering unknown keys) and
    drivers declaring ``**kwargs`` (pass everything).
    """
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return fn(**kwargs)
    params = signature.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(**kwargs)
    accepted = {
        name: value
        for name, value in kwargs.items()
        if name in params
        and params[name].kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    return fn(**accepted)


def resolve_entrypoint(experiment: str) -> Callable[..., Any]:
    """Locate the driver callable for an experiment id."""
    try:
        registry = importlib.import_module("lbcs_repro.experiments")
        resolver = getattr(registry, "get_experiment", None)
        if callable(resolver):
            return resolver(experiment)
    except Exception as exc:  # pragma: no cover - registry optional
        LOGGER.debug("Registry lookup for %s failed (%s)", experiment, exc)

    spec = EXPERIMENT_ENTRYPOINTS.get(experiment)
    if spec is None:
        raise KeyError(f"Unknown experiment {experiment!r}")
    module_name, entrypoints = spec
    module = importlib.import_module(module_name)
    for attr in entrypoints:
        fn = getattr(module, attr, None)
        if callable(fn):
            return fn
    for attr in ("run", "main"):
        fn = getattr(module, attr, None)
        if callable(fn):
            return fn
    raise AttributeError(f"No callable entrypoint found in {module_name}")


def run_one(
    experiment: str,
    config: Optional[Dict[str, Any]] = None,
    *,
    logger: Optional[logging.Logger] = None,
    dry_run: bool = False,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run a single experiment driver and return a summary record."""
    log = logger or LOGGER
    started = time.time()
    record: Dict[str, Any] = {
        "experiment": experiment,
        "section": EXPERIMENT_SECTIONS.get(experiment, experiment),
        "status": "pending",
        "wall_time": 0.0,
        "artifacts": {},
        "result_keys": [],
        "error": None,
    }
    if experiment not in ALL_EXPERIMENTS:
        record["status"] = "unknown"
        record["error"] = f"{experiment!r} is not an in-scope experiment"
        log.error(record["error"])
        return record

    try:
        entrypoint = resolve_entrypoint(experiment)
    except Exception as exc:
        record["status"] = "import_error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        log.error("Could not resolve driver for %s: %s", experiment, exc)
        record["wall_time"] = time.time() - started
        return record

    if dry_run:
        record["status"] = "dry_run"
        record["wall_time"] = time.time() - started
        return record

    payload = section_config(config or {}, experiment)
    payload["experiment"] = experiment
    if overrides:
        payload = deep_merge(payload, overrides)

    log.info("=" * 78)
    log.info("Running experiment %s (%s)", experiment, record["section"])
    log.info("=" * 78)
    try:
        result = _call_with_supported_kwargs(entrypoint, {"config": payload, **payload})
    except TypeError as exc:
        # Older/leaner drivers only accept keyword overrides.
        log.debug("Falling back to plain kwargs call for %s (%s)", experiment, exc)
        try:
            result = _call_with_supported_kwargs(entrypoint, dict(payload))
        except Exception as exc2:  # pragma: no cover - surfaced in the report
            record["status"] = "failed"
            record["error"] = f"{type(exc2).__name__}: {exc2}"
            record["wall_time"] = time.time() - started
            log.error("Experiment %s failed: %s", experiment, exc2)
            return record
    except Exception as exc:  # pragma: no cover - surfaced in the report
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["wall_time"] = time.time() - started
        log.error("Experiment %s failed: %s", experiment, exc)
        return record

    record["status"] = "ok"
    record["wall_time"] = time.time() - started
    if isinstance(result, dict):
        record["result_keys"] = sorted(str(k) for k in result.keys())
        artifacts = result.get("artifacts")
        if isinstance(artifacts, dict):
            record["artifacts"] = {str(k): str(v) for k, v in artifacts.items()}
        record["result"] = result
    else:
        record["result"] = result
    log.info(
        "Experiment %s finished in %.1fs (%d result keys)",
        experiment,
        record["wall_time"],
        len(record["result_keys"]),
    )
    return record


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def resolve_experiments(
    names: Optional[Sequence[str]] = None,
    groups: Optional[Sequence[str]] = None,
    *,
    include_all_on_empty: bool = True,
) -> List[str]:
    """Expand ids/groups into a deduplicated, canonically-ordered list."""
    requested: List[str] = []

    def _expand(token: str) -> None:
        token = (token or "").strip()
        if not token:
            return
        normalised = token.replace(" ", "").replace("-", "_")
        lowered = token.strip().lower()
        if lowered in GROUPS:
            requested.extend(GROUPS[lowered])
            return
        if normalised.lower() in GROUPS:
            requested.extend(GROUPS[normalised.lower()])
            return
        key = lowered.replace(" ", "").replace("_", "")
        for group_name, members in GROUPS.items():
            if group_name.replace(".", "").replace("_", "") == key:
                requested.extend(members)
                return
        # Fall back to the package registry's alias resolution.
        try:
            registry = importlib.import_module("lbcs_repro.experiments")
            resolver = getattr(registry, "experiment_id", None)
            if callable(resolver):
                requested.append(resolver(token))
                return
        except Exception:  # pragma: no cover
            pass
        alias = token.strip().lower().replace(" ", "")
        for alias_name, canonical in (
            ("figure1_trivial", "figure1"),
            ("table1_prelim", "table1"),
            ("table2_table3", "table2"),
            ("table2_table3_compare", "table2"),
            ("figure2_robustness", "figure2"),
            ("table8_sizes", "table8"),
            ("table9_search_times", "table9"),
            ("table5_init", "table5"),
            ("table6_cross_arch", "table6"),
            ("appendixc3", "appendix_c3"),
        ):
            if alias.replace("_", "") == alias_name.replace("_", ""):
                requested.append(canonical)
                return
        if lowered in ALL_EXPERIMENTS:
            requested.append(lowered)
            return
        raise KeyError(
            f"Unknown experiment/group {token!r}. "
            f"Available: {', '.join(ALL_EXPERIMENTS)} | groups: {', '.join(sorted(GROUPS))}"
        )

    for token in list(names or []) + list(groups or []):
        _expand(token)

    if not requested and include_all_on_empty:
        requested = list(ALL_EXPERIMENTS)

    ordered = [name for name in ALL_EXPERIMENTS if name in set(requested)]
    # Preserve any extra registry ids that are not in our canonical tuple.
    for name in requested:
        if name not in ordered:
            ordered.append(name)
    return ordered


def run_all(
    experiments: Optional[Sequence[str]] = None,
    *,
    groups: Optional[Sequence[str]] = None,
    config: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    smoke: bool = False,
    dry_run: bool = False,
    continue_on_error: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Run the requested experiments and persist one aggregate report."""
    log = logger or LOGGER
    selected = resolve_experiments(experiments, groups)
    started = time.time()

    log.info("Orchestrating %d experiment(s): %s", len(selected), ", ".join(selected))
    log.info("Out of scope: %s", "; ".join(f"{k} -> {v}" for k, v in OUT_OF_SCOPE.items()))

    records: List[Dict[str, Any]] = []
    for experiment in selected:
        merged = load_config(experiment, config, overrides=overrides, smoke=smoke)
        if seed is not None:
            merged["seed"] = int(seed)
        if device is not None:
            merged["device"] = device
        if output_root:
            merged["output_root"] = output_root
            merged.setdefault("output_dir", os.path.join(output_root, experiment))

        record = run_one(
            experiment,
            merged,
            logger=log,
            dry_run=dry_run,
            overrides=overrides,
        )
        records.append(record)
        if record.get("status") == "failed" and not continue_on_error:
            log.error("Stopping after failure of %s (continue_on_error=False)", experiment)
            break

    report = build_report(records, output_root=output_root, seed=seed, device=device,
                          smoke=smoke, dry_run=dry_run, wall_time=time.time() - started)
    if not dry_run:
        save_report(report, output_root)
    log.info(
        "Run report: %d ok / %d attempted in %.1fs",
        report["num_ok"],
        report["num_attempted"],
        report["wall_time"],
    )
    return report


def build_report(
    records: Sequence[Dict[str, Any]],
    *,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    seed: Optional[int] = None,
    device: Optional[str] = None,
    smoke: bool = False,
    dry_run: bool = False,
    wall_time: float = 0.0,
) -> Dict[str, Any]:
    """Assemble the aggregate run report (JSON-serializable)."""
    summarised: List[Dict[str, Any]] = []
    for record in records:
        entry = {
            "experiment": record.get("experiment"),
            "section": record.get("section"),
            "status": record.get("status"),
            "wall_time": round(float(record.get("wall_time") or 0.0), 3),
            "artifacts": record.get("artifacts", {}),
            "result_keys": record.get("result_keys", []),
            "error": record.get("error"),
        }
        summarised.append(entry)

    num_ok = sum(1 for r in summarised if r["status"] == "ok")
    return {
        "num_attempted": len(summarised),
        "num_ok": num_ok,
        "num_failed": sum(1 for r in summarised if r["status"] in ("failed", "import_error")),
        "num_dry_run": sum(1 for r in summarised if r["status"] == "dry_run"),
        "wall_time": round(float(wall_time), 3),
        "seed": seed,
        "device": device,
        "smoke": bool(smoke),
        "dry_run": bool(dry_run),
        "output_root": output_root,
        "repeats": dict(PAPER_REPEATS),
        "out_of_scope": dict(OUT_OF_SCOPE),
        "experiments": summarised,
    }


def format_report(report: Dict[str, Any]) -> str:
    """Render the run report as plain text."""
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("LBCS reproduction -- run_all summary")
    lines.append("=" * 78)
    lines.append(
        f"attempted={report.get('num_attempted', 0)} ok={report.get('num_ok', 0)} "
        f"failed={report.get('num_failed', 0)} wall_time={report.get('wall_time', 0.0):.1f}s"
    )
    lines.append(f"seed={report.get('seed')} device={report.get('device')} smoke={report.get('smoke')}")
    lines.append("")
    header = f"{'experiment':<14} {'section':<12} {'status':<12} {'wall_time':>9}  artifacts"
    lines.append(header)
    lines.append("-" * len(header))
    for entry in report.get("experiments", []):
        artifacts = ", ".join(sorted(entry.get("artifacts", {}).keys()))
        lines.append(
            f"{str(entry.get('experiment')):<14} {str(entry.get('section')):<12} "
            f"{str(entry.get('status')):<12} {float(entry.get('wall_time') or 0.0):>8.1f}s  {artifacts}"
        )
        if entry.get("error"):
            lines.append(f"    error: {entry['error']}")
    lines.append("")
    lines.append("Repeat protocol (mean +- std over repeats):")
    for section, repeats in report.get("repeats", {}).items():
        lines.append(f"  {section:<12} {repeats} repeats")
    lines.append("")
    lines.append("Out of scope (not reproduced):")
    for key, description in report.get("out_of_scope", {}).items():
        lines.append(f"  {key}: {description}")
    lines.append("=" * 78)
    return "\n".join(lines)


def save_report(report: Dict[str, Any], output_root: str = DEFAULT_OUTPUT_ROOT) -> Dict[str, str]:
    """Persist the run report as JSON + text under ``output_root``."""
    os.makedirs(output_root, exist_ok=True)
    paths: Dict[str, str] = {}

    json_path = os.path.join(output_root, "run_all_report.json")
    try:
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=_json_default)
        paths["json"] = json_path
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Could not write %s: %s", json_path, exc)

    text_path = os.path.join(output_root, "run_all_report.txt")
    try:
        with open(text_path, "w", encoding="utf-8") as handle:
            handle.write(format_report(report))
        paths["txt"] = text_path
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Could not write %s: %s", text_path, exc)

    # Also drop the per-experiment records for downstream aggregation.
    records_path = os.path.join(output_root, "run_all_records.jsonl")
    try:
        with open(records_path, "w", encoding="utf-8") as handle:
            for entry in report.get("experiments", []):
                handle.write(json.dumps(entry, default=_json_default) + "\n")
        paths["jsonl"] = records_path
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Could not write %s: %s", records_path, exc)

    report["artifacts"] = paths
    return paths


def _json_default(obj: Any) -> Any:
    """JSON fallback for numpy scalars / arrays / dataclasses."""
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:  # pragma: no cover
            pass
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:  # pragma: no cover
            pass
    try:
        import numpy as np  # local import: numpy always available in practice

        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:  # pragma: no cover
        pass
    return str(obj)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    """Build the orchestrator CLI parser."""
    parser = argparse.ArgumentParser(
        prog="run_all",
        description=(
            "Run the in-scope LBCS reproduction experiments "
            "(Figure 1, Tables 1/2/3/5/6/8/9, Figure 2, Appendix C.3). "
            "ImageNet-1k, continual learning and streaming are out of scope."
        ),
    )
    parser.add_argument(
        "--experiments", "-e", nargs="+", default=None,
        help="Experiment ids (or aliases) to run; defaults to every in-scope experiment.",
    )
    parser.add_argument(
        "--groups", "-g", nargs="+", default=None,
        help=f"Named groups: {', '.join(sorted(GROUPS))}",
    )
    parser.add_argument("--all", action="store_true", help="Run every in-scope experiment.")
    parser.add_argument("--config", "-c", default=None, help="Extra YAML config path/name to merge.")
    parser.add_argument(
        "--set", dest="overrides", nargs="*", default=None,
        help="Config overrides as key.sub=value pairs.",
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Results directory.")
    parser.add_argument("--seed", type=int, default=None, help="Base seed for all repeats.")
    parser.add_argument("--device", default=None, help="Torch device (e.g. cuda, cpu).")
    parser.add_argument(
        "--smoke", action="store_true",
        help="Apply each config's smoke block (tiny CPU run) before executing.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Resolve and report without running.")
    parser.add_argument("--continue-on-error", action="store_true", default=True,
                        help="Keep going when an experiment fails (default).")
    parser.add_argument("--stop-on-error", dest="continue_on_error", action="store_false",
                        help="Abort the sweep at the first failing experiment.")
    parser.add_argument("--list", action="store_true", help="List experiments and groups, then exit.")
    parser.add_argument("--selftest", action="store_true", help="Run offline self-tests, then exit.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging.")
    return parser


def list_experiments(as_text: bool = True) -> Any:
    """List in-scope experiments, groups and out-of-scope sections."""
    catalogue = {
        "experiments": available_experiments(include_aliases=True),
        "canonical": list(ALL_EXPERIMENTS),
        "groups": {k: list(v) for k, v in sorted(GROUPS.items())},
        "out_of_scope": dict(OUT_OF_SCOPE),
        "repeats": dict(PAPER_REPEATS),
    }
    if not as_text:
        return catalogue
    lines = ["In-scope experiments:"]
    for name in ALL_EXPERIMENTS:
        config_name = EXPERIMENT_CONFIGS.get(name, "-")
        lines.append(f"  {name:<12} config={config_name}")
    lines.append("")
    lines.append("Groups:")
    for group, members in sorted(GROUPS.items()):
        lines.append(f"  {group:<12} -> {', '.join(members)}")
    lines.append("")
    lines.append("Out of scope:")
    for key, description in OUT_OF_SCOPE.items():
        lines.append(f"  {key}: {description}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    parser = build_argparser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    if args.selftest:
        result = _selftest(verbose=True)
        return 0 if result.get("ok") else 1

    if args.list:
        print(list_experiments())
        return 0

    experiments: List[str] = []
    if args.all:
        experiments = list(ALL_EXPERIMENTS)
    elif args.experiments:
        experiments = list(args.experiments)
    groups = list(args.groups or [])

    if not experiments and not groups:
        LOGGER.info("No experiments requested; defaulting to the full in-scope suite.")

    overrides = parse_overrides(args.overrides)
    if args.seed is not None:
        overrides = deep_merge(overrides, {"seed": int(args.seed)})
    if args.device is not None:
        overrides = deep_merge(overrides, {"device": args.device})

    report = run_all(
        experiments,
        groups=groups,
        config=args.config,
        overrides=overrides or None,
        seed=args.seed,
        device=args.device,
        output_root=args.output_root,
        smoke=args.smoke,
        dry_run=args.dry_run,
        continue_on_error=bool(args.continue_on_error),
    )

    print(format_report(report))
    if report.get("num_failed"):
        return 1
    return 0


# --------------------------------------------------------------------------
# Offline self-test
# --------------------------------------------------------------------------


def _selftest(verbose: bool = True) -> Dict[str, Any]:
    """Offline checks of orchestration logic (no torch, GPU or datasets)."""
    checks: Dict[str, Any] = {}

    # 1) deep_merge semantics
    merged = deep_merge(
        {"a": 1, "nested": {"x": 1, "y": 2}, "lst": [1, 2], "inherit": ["default.yaml"]},
        {"nested": {"y": 3, "z": 4}, "lst": [9]},
    )
    checks["deep_merge"] = (
        merged["a"] == 1
        and merged["nested"] == {"x": 1, "y": 3, "z": 4}
        and merged["lst"] == [9]
        and "inherit" not in merged
    )

    # 2) override parsing with nested keys + scalar coercion
    ov = parse_overrides(["lbcs.epsilon=0.3", "table1.ks=[200,400]", "smoke=true"])
    checks["parse_overrides"] = (
        ov.get("lbcs", {}).get("epsilon") == 0.3
        and ov.get("table1", {}).get("ks") == [200, 400]
        and ov.get("smoke") is True
    )

    # 3) group / alias resolution
    resolved = resolve_experiments(["section5.2"], None)
    checks["resolve_groups"] = resolved == ["table2", "table3"]
    resolved_all = resolve_experiments(["figure1_trivial"], None)
    checks["resolve_alias"] = resolved_all == ["figure1"]
    # canonical ordering restored even when ids are given out of order
    checks["resolve_order"] = resolve_experiments(["table9", "figure1"], None) == ["figure1", "table9"]
    # unknown tokens raise
    try:
        resolve_experiments(["does_not_exist"], None)
        checks["resolve_unknown_raises"] = False
    except KeyError:
        checks["resolve_unknown_raises"] = True

    # 4) report assembly + rendering
    fake_records = [
        {"experiment": "figure1", "section": "figure1", "status": "ok", "wall_time": 1.5,
         "artifacts": {"figure1.png": "results/figure1/figure1.png"}, "result_keys": ["table"], "error": None},
        {"experiment": "table1", "section": "table1", "status": "failed", "wall_time": 0.2,
         "artifacts": {}, "result_keys": [], "error": "RuntimeError: boom"},
    ]
    report = build_report(fake_records, output_root="results", wall_time=1.7)
    text = format_report(report)
    checks["build_report"] = (
        report["num_attempted"] == 2 and report["num_ok"] == 1 and report["num_failed"] == 1
    )
    checks["format_report"] = "figure1" in text and "Out of scope" in text

    # 5) scope discipline: no out-of-scope experiment can be dispatched
    checks["scope_clean"] = all(
        token not in name
        for name in ALL_EXPERIMENTS
        for token in ("imagenet", "continual", "stream")
    )
    checks["out_of_scope_recorded"] = set(OUT_OF_SCOPE) == {"section5.4", "appendix_e.5", "appendix_e.6"}

    # 6) repeat protocol matches the paper
    checks["repeats"] = (
        PAPER_REPEATS.get("section5.1") == 20
        and PAPER_REPEATS.get("section5.2") == 10
        and PAPER_REPEATS.get("section5.3") == 10
    )

    # 7) every experiment maps to a driver entrypoint + config file
    checks["entrypoints_complete"] = all(name in EXPERIMENT_ENTRYPOINTS for name in ALL_EXPERIMENTS)
    checks["configs_complete"] = all(name in EXPERIMENT_CONFIGS for name in ALL_EXPERIMENTS)

    # 8) dry-run dispatch produces a report without touching data/torch
    dry = run_all(["figure1"], output_root=os.path.join("results", "_selftest"),
                  dry_run=True, logger=logging.getLogger("lbcs_repro.selftest"))
    checks["dry_run"] = dry["num_dry_run"] == 1 and dry["num_ok"] == 0

    # 9) resolve entrypoint only for in-scope experiments (import may fail offline)
    checks["unknown_experiment_rejected"] = (
        run_one("section5.4_imagenet", {}, logger=logging.getLogger("lbcs_repro.selftest"))["status"]
        == "unknown"
    )

    checks["ok"] = all(bool(v) for v in checks.values())

    if verbose:
        print("run_all self-test")
        print("-" * 60)
        for name, value in checks.items():
            print(f"  {str(name):<28} {'PASS' if value else 'FAIL'}")
        print("-" * 60)
        print(f"  overall: {'PASS' if checks['ok'] else 'FAIL'}")
    return checks


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
