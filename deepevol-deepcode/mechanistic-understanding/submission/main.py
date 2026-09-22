#!/usr/bin/env python
"""End-to-end orchestration for the GPT2-medium DPO / toxicity reproduction.

This is the top-level entry point described in the reproduction plan
("main.py runs the configured pipeline phases").  It wires together the phase
scripts that implement the paper *A Mechanistic Understanding of Alignment
Algorithms: A Case Study on DPO and Toxicity* (GPT2-medium only; Llama2/GLU
paths are out of scope).

Pipeline phases (in dependency order)::

    probe          -> scripts/train_probe.py          # W_Toxic  (Jigsaw, ~94% acc)
    vectors        -> scripts/extract_toxic_vectors.py # MLP.v_Toxic / SVD.U_Toxic
    interventions  -> scripts/run_interventions.py     # Table 2 (residual subtraction)
    pairs          -> scripts/generate_pairs.py        # 24,576 PPLM/greedy pairs
    dpo            -> scripts/train_dpo.py             # GPT2_DPO (Table 8 hyperparams)
    eval           -> scripts/eval_model.py            # toxicity / PPL / F1
    analyze        -> scripts/analyze_dpo.py           # Figures 1-5, Tables 2-4 claims
    unalign        -> scripts/unalign_gpt2.py          # Table 4 (key-vector scaling)

Usage examples::

    python main.py --list
    python main.py --phase probe
    python main.py --phases probe,vectors,interventions
    python main.py --all --quick                 # smoke test of everything
    python main.py --all --dry-run               # show commands only
    python main.py --status                      # artifact report

Every phase can also be launched directly with its own script; this module only
adds configuration merging, artifact checks, ordering and a summary report.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Repository bootstrap (allow `python main.py` from any working directory)
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

ROOT = _THIS_DIR
SCRIPTS_DIR = os.path.join(ROOT, "scripts")

# ---------------------------------------------------------------------------
# Constants -- paper / plan defaults (GPT2-medium, in scope)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = os.path.join("configs", "default.yaml")
DEFAULT_MODEL = "openai-community/gpt2-medium"
DEFAULT_DPO_DIR = os.path.join("artifacts", "models", "gpt2_dpo")
DEFAULT_PROBE_PATH = os.path.join("artifacts", "probe", "w_toxic.pt")
DEFAULT_VECTORS_PATH = os.path.join("artifacts", "vectors", "toxic_vectors.pt")
DEFAULT_PAIRS_PATH = os.path.join("artifacts", "data", "pairs.jsonl")
DEFAULT_INTERVENTIONS_PATH = os.path.join("artifacts", "interventions", "interventions.json")
DEFAULT_ANALYSIS_PATH = os.path.join("artifacts", "analysis", "analyze_dpo_results.json")
DEFAULT_EVAL_PATH = os.path.join("artifacts", "eval", "eval_summary.json")
DEFAULT_UNALIGN_PATH = os.path.join("artifacts", "unalign", "unalign_results.json")

N_PAIRS = 24_576
N_CHALLENGE_PROMPTS = 1_199
N_SHIT_PROMPTS = 295
N_F1_SENTENCES = 2_000
CONVERGENCE_EXAMPLES = 6_700

# Paper reference values (Tables 2 / 4) -- used for the final report only.
REFERENCE = {
    "gpt2": {"toxicity": 0.453, "perplexity": 21.70, "f1": 0.193},
    "w_toxic": {"toxicity": 0.245, "perplexity": 23.56, "f1": 0.193},
    "mlp_v_770_19": {"toxicity": 0.305, "perplexity": 23.30, "f1": 0.192},
    "svd_u_toxic_0": {"toxicity": 0.268, "perplexity": 23.48, "f1": 0.193},
    "gpt2_dpo": {"toxicity": 0.208, "perplexity": 23.34, "f1": 0.195},
    "unaligned": {"toxicity": 0.458, "perplexity": 23.30, "f1": 0.195},
}


class Phase:
    """One pipeline phase: a wrapper around a `scripts/*.py` entry point."""

    def __init__(
        self,
        name: str,
        script: str,
        description: str,
        produces: Sequence[str] = (),
        requires: Sequence[str] = (),
        paper: str = "",
        heavy: bool = True,
    ) -> None:
        self.name = name
        self.script = script
        self.description = description
        self.produces = list(produces)
        self.requires = list(requires)
        self.paper = paper
        self.heavy = heavy

    @property
    def path(self) -> str:
        return os.path.join(SCRIPTS_DIR, self.script)

    def exists(self) -> bool:
        return os.path.isfile(self.path)

    def outputs_present(self) -> bool:
        """True when at least one declared artifact already exists."""
        return any(os.path.exists(_p(p)) for p in self.produces)

    def missing_inputs(self) -> List[str]:
        return [p for p in self.requires if not os.path.exists(_p(p))]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "script": os.path.relpath(self.path, ROOT),
            "description": self.description,
            "produces": self.produces,
            "requires": self.requires,
            "paper": self.paper,
            "available": self.exists(),
            "completed": self.outputs_present(),
        }


def _p(path: str) -> str:
    """Resolve a possibly-relative artifact path against the repo root."""
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


# Phase registry, in dependency order.
PHASES: List[Phase] = [
    Phase(
        "probe",
        "train_probe.py",
        "Train the linear toxicity probe W_Toxic on Jigsaw (target ~94% valid acc).",
        produces=[DEFAULT_PROBE_PATH],
        paper="S3.1, S4.2",
    ),
    Phase(
        "vectors",
        "extract_toxic_vectors.py",
        "Rank MLP value vectors by cosine sim to W_Toxic[:,1]; keep top 128; SVD.",
        produces=[
            DEFAULT_VECTORS_PATH,
            os.path.join("artifacts", "vectors", "table1_projections.json"),
        ],
        requires=[DEFAULT_PROBE_PATH],
        paper="S3.1, S3.2, Table 1",
    ),
    Phase(
        "interventions",
        "run_interventions.py",
        "Residual-stream subtraction with W_Toxic / MLP.v_Toxic / SVD.U_Toxic.",
        produces=[DEFAULT_INTERVENTIONS_PATH],
        requires=[DEFAULT_PROBE_PATH, DEFAULT_VECTORS_PATH],
        paper="S3.3, Table 2",
    ),
    Phase(
        "pairs",
        "generate_pairs.py",
        "Generate 24,576 PPLM-toxic / greedy-non-toxic Wikitext-2 pairs.",
        produces=[DEFAULT_PAIRS_PATH],
        requires=[DEFAULT_PROBE_PATH],
        paper="S4.2, Appendix E Table 9",
    ),
    Phase(
        "dpo",
        "train_dpo.py",
        "DPO-align GPT2-medium on the pair dataset (beta=0.1, RMSProp, lr=1e-6).",
        produces=[os.path.join(DEFAULT_DPO_DIR, "config.json")],
        requires=[DEFAULT_PAIRS_PATH],
        paper="S4.1, S4.2, Appendix E Table 8",
    ),
    Phase(
        "eval",
        "eval_model.py",
        "Toxicity / Wikitext-2 PPL / F1 for GPT2 and GPT2_DPO.",
        produces=[DEFAULT_EVAL_PATH],
        requires=[DEFAULT_DPO_DIR],
        paper="S3.3",
    ),
    Phase(
        "analyze",
        "analyze_dpo.py",
        "Mechanistic analyses: logit lens, activations, parameter diff, delta_x.",
        produces=[DEFAULT_ANALYSIS_PATH],
        requires=[DEFAULT_DPO_DIR, DEFAULT_VECTORS_PATH],
        paper="S4.2, S5.1-5.2, Figures 1-5",
    ),
    Phase(
        "unalign",
        "unalign_gpt2.py",
        "Scale top-7 toxic MLP key vectors 10x and re-evaluate (un-alignment).",
        produces=[DEFAULT_UNALIGN_PATH],
        requires=[DEFAULT_DPO_DIR, DEFAULT_VECTORS_PATH],
        paper="S6, Table 4",
    ),
]

PHASE_NAMES = [p.name for p in PHASES]
PHASE_MAP: Dict[str, Phase] = {p.name: p for p in PHASES}

# Phase-specific extra CLI flags passed through to the underlying scripts.
PHASE_FLAGS: Dict[str, List[str]] = {
    "probe": [],
    "vectors": [],
    "interventions": [],
    "pairs": [],
    "dpo": [],
    "eval": [],
    "analyze": [],
    "unalign": [],
}


# ---------------------------------------------------------------------------
# Configuration handling
# ---------------------------------------------------------------------------


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Best-effort YAML/JSON config loader; returns ``{}`` on any failure."""
    if not path:
        return {}
    full = _p(path)
    if not os.path.isfile(full):
        return {}
    try:
        if full.endswith((".yaml", ".yml")):
            import yaml  # noqa: WPS433 (lazy import)

            with open(full, "r", encoding="utf-8") as handle:
                return yaml.safe_load(handle) or {}
        with open(full, "r", encoding="utf-8") as handle:
            return json.load(handle) or {}
    except Exception:  # pragma: no cover - config is advisory
        return {}


def dig(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Nested dict lookup tolerant of missing keys."""
    node: Any = cfg
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def resolve_settings(args: argparse.Namespace) -> Dict[str, Any]:
    """Merge CLI args > YAML config > plan defaults into a flat settings dict."""
    cfg = load_config(args.config)
    quick = bool(getattr(args, "quick", False))

    seed = args.seed if args.seed is not None else dig(cfg, "seed", default=0)
    device = args.device if args.device is not None else dig(cfg, "device", default=None)
    model = args.model or dig(cfg, "model", "name", default=DEFAULT_MODEL) or DEFAULT_MODEL

    settings: Dict[str, Any] = {
        "config": args.config,
        "config_values": cfg,
        "model": model,
        "seed": 0 if seed is None else int(seed),
        "device": device,
        "quick": quick,
        "dry_run": bool(getattr(args, "dry_run", False)),
        "dpo_dir": args.dpo_dir or dig(cfg, "paths", "dpo_dir", default=DEFAULT_DPO_DIR) or DEFAULT_DPO_DIR,
        "probe_path": args.probe_path
        or dig(cfg, "paths", "probe", default=DEFAULT_PROBE_PATH)
        or DEFAULT_PROBE_PATH,
        "vectors_path": args.vectors_path
        or dig(cfg, "paths", "vectors", default=DEFAULT_VECTORS_PATH)
        or DEFAULT_VECTORS_PATH,
        "pairs_path": args.pairs_path
        or dig(cfg, "paths", "data_dir", default=None)
        or DEFAULT_PAIRS_PATH,
        "n_pairs": int(dig(cfg, "data", "pairwise", "n_pairs", default=N_PAIRS) or N_PAIRS),
        "valid_ratio": float(dig(cfg, "data", "pairwise", "valid_ratio", default=0.1) or 0.1),
        "n_challenge_prompts": int(
            dig(cfg, "evaluation", "n_challenge_prompts", default=N_CHALLENGE_PROMPTS)
            or N_CHALLENGE_PROMPTS
        ),
        "n_f1_sentences": int(
            dig(cfg, "evaluation", "n_f1_sentences", default=N_F1_SENTENCES) or N_F1_SENTENCES
        ),
        "max_new_tokens": int(
            dig(cfg, "evaluation", "max_new_tokens", default=20) or 20
        ),
        "layer": int(dig(cfg, "analysis", "layer", default=19) or 19),
        "mlp_idx": int(dig(cfg, "analysis", "mlp_idx", default=770) or 770),
        "n_vectors": int(dig(cfg, "vectors", "unalign", "n_vectors", default=7) or 7),
        "unalign_scale": float(dig(cfg, "vectors", "unalign", "scale", default=10.0) or 10.0),
        "target_ppl": float(dig(cfg, "reference", "gpt2_dpo_ppl", default=23.34) or 23.34),
        "in_scope": dig(cfg, "in_scope", default={"model": "gpt2-medium", "llama2": False, "glu": False}),
    }

    if quick:
        settings["n_challenge_prompts"] = min(settings["n_challenge_prompts"], 32)
        settings["n_f1_sentences"] = min(settings["n_f1_sentences"], 32)
        settings["n_pairs"] = min(settings["n_pairs"], 32)
        settings["max_new_tokens"] = min(settings["max_new_tokens"], 8)

    return settings


# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------


def build_command(phase: Phase, args: argparse.Namespace, settings: Dict[str, Any]) -> List[str]:
    """Build the argv list used to launch one phase script."""
    cmd: List[str] = [sys.executable, os.path.relpath(phase.path, ROOT)]

    if args.config:
        cmd += ["--config", args.config]
    if args.device:
        cmd += ["--device", args.device]
    if getattr(args, "seed", None) is not None:
        cmd += ["--seed", str(args.seed)]
    if getattr(args, "quiet", False):
        cmd += ["--quiet"]
    if getattr(args, "quick", False):
        cmd += ["--quick"]

    name = phase.name

    if name == "probe":
        if args.model:
            cmd += ["--model", args.model]
        cmd += ["--probe-path", settings["probe_path"]]

    elif name == "vectors":
        if args.model:
            cmd += ["--model", args.model]
        cmd += ["--probe-path", settings["probe_path"]]
        cmd += ["--out-dir", os.path.dirname(_p(settings["vectors_path"]))]

    elif name == "interventions":
        if args.model:
            cmd += ["--model", args.model]
        cmd += ["--probe-path", settings["probe_path"]]
        cmd += ["--vectors-path", settings["vectors_path"]]
        cmd += ["--target-ppl", str(settings["target_ppl"])]
        if getattr(args, "no_alpha_search", False):
            cmd += ["--no-alpha-search"]

    elif name == "pairs":
        if args.model:
            cmd += ["--model", args.model]
        cmd += ["--probe-path", settings["probe_path"]]
        cmd += ["--n-pairs", str(settings["n_pairs"])]
        cmd += ["--pairs-path", settings["pairs_path"]]
        if getattr(args, "force", False):
            cmd += ["--force"]
        else:
            cmd += ["--resume"]

    elif name == "dpo":
        if args.model:
            cmd += ["--model", args.model]
        cmd += ["--pairs", settings["pairs_path"]]
        cmd += ["--output-dir", settings["dpo_dir"]]
        cmd += ["--valid-ratio", str(settings["valid_ratio"])]
        if getattr(args, "max_pairs", None):
            cmd += ["--max-pairs", str(args.max_pairs)]

    elif name == "eval":
        cmd += ["--dpo-dir", settings["dpo_dir"]]
        cmd += ["--n-prompts", str(settings["n_challenge_prompts"])]
        cmd += ["--max-new-tokens", str(settings["max_new_tokens"])]
        cmd += ["--all"]
        if args.model:
            cmd += ["--model", args.model]

    elif name == "analyze":
        if args.model:
            cmd += ["--model", args.model]
        cmd += ["--dpo-model", settings["dpo_dir"]]
        cmd += ["--probe", settings["probe_path"]]
        cmd += ["--vectors", settings["vectors_path"]]
        cmd += ["--layer", str(settings["layer"])]
        cmd += ["--mlp-idx", str(settings["mlp_idx"])]
        cmd += ["--phase", "all"]

    elif name == "unalign":
        cmd += ["--dpo-dir", settings["dpo_dir"]]
        cmd += ["--probe-path", settings["probe_path"]]
        cmd += ["--vectors-path", settings["vectors_path"]]
        cmd += ["--n-vectors", str(settings["n_vectors"])]
        cmd += ["--scale", str(settings["unalign_scale"])]
        if args.model:
            cmd += ["--model", args.model]

    cmd += [flag for flag in PHASE_FLAGS.get(name, []) if flag]
    return cmd


# ---------------------------------------------------------------------------
# Phase execution
# ---------------------------------------------------------------------------


def run_phase(
    phase: Phase,
    args: argparse.Namespace,
    settings: Dict[str, Any],
    stop_on_error: bool = True,
) -> Dict[str, Any]:
    """Run one phase as a subprocess; returns a result record."""
    record: Dict[str, Any] = {
        "phase": phase.name,
        "script": os.path.relpath(phase.path, ROOT),
        "command": "",
        "status": "pending",
        "returncode": None,
        "elapsed": 0.0,
        "error": None,
    }

    if not phase.exists():
        record["status"] = "missing-script"
        record["error"] = f"{phase.script} not found under scripts/"
        print(f"[main] SKIP {phase.name}: {record['error']}")
        return record

    missing = phase.missing_inputs()
    if missing:
        record["status"] = "missing-inputs"
        record["error"] = "missing: " + ", ".join(os.path.relpath(m, ROOT) for m in missing)
        print(f"[main] SKIP {phase.name}: {record['error']}")
        print(f"        (run the earlier phases first: {', '.join(PHASE_NAMES)})")
        return record

    if phase.outputs_present() and not getattr(args, "force", False):
        record["status"] = "cached"
        print(f"[main] CACHED {phase.name}: outputs already exist (use --force to rerun)")
        return record

    cmd = build_command(phase, args, settings)
    record["command"] = " ".join(cmd)
    print(f"[main] RUN  {phase.name}: {record['command']}")

    if settings.get("dry_run"):
        record["status"] = "dry-run"
        return record

    start = time.time()
    try:
        completed = subprocess.run(cmd, cwd=ROOT, check=False)
        record["returncode"] = int(completed.returncode)
        record["status"] = "ok" if completed.returncode == 0 else "failed"
    except KeyboardInterrupt:
        record["status"] = "interrupted"
        record["error"] = "KeyboardInterrupt"
        raise
    except Exception as exc:  # pragma: no cover - defensive
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        record["elapsed"] = round(time.time() - start, 2)

    if record["status"] != "ok":
        print(f"[main] {phase.name} -> {record['status']} (rc={record['returncode']})")
        if stop_on_error:
            print("[main] stopping after failure (use --keep-going to continue)")

    return record


def run_in_process(phase: Phase, args: argparse.Namespace, settings: Dict[str, Any]) -> Dict[str, Any]:
    """Alternative to subprocess: import the script and call its ``main()``.

    Useful for debuggers/profilers; memory from one phase is *not* released, so
    the subprocess path is the default.
    """
    record: Dict[str, Any] = {
        "phase": phase.name,
        "script": os.path.relpath(phase.path, ROOT),
        "command": "(in-process)",
        "status": "pending",
        "returncode": None,
        "elapsed": 0.0,
        "error": None,
    }
    if not phase.exists():
        record["status"] = "missing-script"
        return record
    missing = phase.missing_inputs()
    if missing:
        record["status"] = "missing-inputs"
        record["error"] = ", ".join(missing)
        return record

    cmd = build_command(phase, args, settings)
    argv = cmd[2:]  # drop interpreter + script path
    start = time.time()
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(f"_phase_{phase.name}", phase.path)
        module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        assert spec and spec.loader
        spec.loader.exec_module(module)
        rc = module.main(argv) if hasattr(module, "main") else 0
        record["returncode"] = int(rc or 0)
        record["status"] = "ok" if not rc else "failed"
    except Exception as exc:  # pragma: no cover - defensive
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        record["elapsed"] = round(time.time() - start, 2)
    return record


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def artifact_status(settings: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Check which declared artifacts exist on disk."""
    rows: List[Dict[str, Any]] = []
    for phase in PHASES:
        for path in phase.produces:
            full = _p(path)
            rows.append(
                {
                    "phase": phase.name,
                    "artifact": os.path.relpath(full, ROOT),
                    "exists": os.path.exists(full),
                    "size": os.path.getsize(full) if os.path.exists(full) else 0,
                }
            )
    return rows


def load_json_safe(path: str) -> Dict[str, Any]:
    full = _p(path)
    if not os.path.isfile(full):
        return {}
    try:
        with open(full, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {"data": data}
    except Exception:
        return {}


def extract_metrics(summary: Dict[str, Any]) -> Dict[str, float]:
    """Pull a (toxicity, perplexity, f1) triple out of a loosely-typed summary."""
    metrics: Dict[str, float] = {}
    if not summary:
        return metrics

    def _num(value: Any) -> Optional[float]:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, dict):
            for key in ("mean", "mean_toxicity", "ppl", "perplexity", "mean_f1", "f1", "value"):
                if key in value:
                    got = _num(value[key])
                    if got is not None:
                        return got
        return None

    for key, aliases in (
        ("toxicity", ("toxicity", "mean_toxicity")),
        ("perplexity", ("perplexity", "ppl")),
        ("f1", ("f1", "mean_f1")),
    ):
        for alias in aliases:
            if alias in summary:
                got = _num(summary[alias])
                if got is not None:
                    metrics[key] = got
                    break
    # Nested per-model sections (e.g. {"sections": {"toxicity": {...}}}).
    for section_key in ("sections", "metrics", "results", "summary"):
        node = summary.get(section_key)
        if isinstance(node, dict):
            nested = extract_metrics(node)
            for key, value in nested.items():
                metrics.setdefault(key, value)
    return metrics


def print_table(rows: List[Tuple[str, ...]], headers: Sequence[str]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


def print_status(settings: Dict[str, Any]) -> None:
    print("=" * 78)
    print("Artifact status")
    print("=" * 78)
    rows = [(r["phase"], r["artifact"], "yes" if r["exists"] else "-", r["size"]) for r in artifact_status(settings)]
    print_table(rows, ("phase", "artifact", "exists", "bytes"))


def print_plan() -> None:
    print("=" * 78)
    print("Reproduction phases (GPT2-medium; Llama2/GLU out of scope)")
    print("=" * 78)
    rows = []
    for phase in PHASES:
        rows.append(
            (
                phase.name,
                os.path.relpath(phase.path, ROOT),
                "yes" if phase.exists() else "MISSING",
                "done" if phase.outputs_present() else "-",
                phase.paper,
            )
        )
    print_table(rows, ("phase", "script", "script?", "artifacts?", "paper"))
    print()
    print("Run e.g.  python main.py --all --quick   (smoke test)")
    print("          python main.py --phases probe,vectors,interventions")


def print_reference() -> None:
    print("=" * 78)
    print("Paper reference values (Tables 2 & 4, GPT2-medium)")
    print("=" * 78)
    rows = [
        (label, f"{v['toxicity']:.3f}", f"{v['perplexity']:.2f}", f"{v['f1']:.3f}")
        for label, v in REFERENCE.items()
    ]
    print_table(rows, ("model / intervention", "toxicity", "perplexity", "F1"))


def collect_eval_metrics(settings: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Best-effort collection of measured metrics from evaluation artifacts."""
    candidates = {
        "eval": DEFAULT_EVAL_PATH,
        "analyze": DEFAULT_ANALYSIS_PATH,
        "interventions": DEFAULT_INTERVENTIONS_PATH,
        "unalign": DEFAULT_UNALIGN_PATH,
    }
    found: Dict[str, Dict[str, float]] = {}
    for label, path in candidates.items():
        data = load_json_safe(path)
        if data:
            metrics = extract_metrics(data)
            if metrics:
                found[label] = metrics
    # Per-model evaluation JSONs written by scripts/eval_model.py
    eval_dir = _p(os.path.join("artifacts", "eval"))
    if os.path.isdir(eval_dir):
        for name in sorted(os.listdir(eval_dir)):
            if name.endswith(".json"):
                data = load_json_safe(os.path.join("artifacts", "eval", name))
                metrics = extract_metrics(data)
                if metrics:
                    found.setdefault(name[:-5], metrics)
    return found


def print_report(settings: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
    print()
    print("=" * 78)
    print("Pipeline summary")
    print("=" * 78)
    rows = [
        (
            r["phase"],
            r["status"],
            "-" if r["returncode"] is None else r["returncode"],
            f"{r['elapsed']:.1f}s",
            (r["error"] or "")[:40],
        )
        for r in results
    ]
    print_table(rows, ("phase", "status", "rc", "time", "note"))

    measured = collect_eval_metrics(settings)
    if measured:
        print()
        print("Measured metrics (from artifacts)")
        print("-" * 78)
        mrows = []
        for label, metrics in sorted(measured.items()):
            mrows.append(
                (
                    label,
                    f"{metrics.get('toxicity', float('nan')):.3f}" if "toxicity" in metrics else "-",
                    f"{metrics.get('perplexity', float('nan')):.2f}" if "perplexity" in metrics else "-",
                    f"{metrics.get('f1', float('nan')):.3f}" if "f1" in metrics else "-",
                )
            )
        print_table(mrows, ("artifact", "toxicity", "perplexity", "F1"))

    print()
    print_reference()


def save_run_summary(
    settings: Dict[str, Any],
    results: List[Dict[str, Any]],
    path: str = os.path.join("artifacts", "main_run_summary.json"),
) -> str:
    payload = {
        "model": settings.get("model"),
        "seed": settings.get("seed"),
        "quick": settings.get("quick"),
        "dry_run": settings.get("dry_run"),
        "config": settings.get("config"),
        "phases": results,
        "artifacts": artifact_status(settings),
        "measured": collect_eval_metrics(settings),
        "reference": REFERENCE,
        "in_scope": settings.get("in_scope"),
    }
    full = _p(path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return full


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------


def parse_phases(raw: Optional[str], all_flag: bool) -> List[str]:
    if all_flag:
        return list(PHASE_NAMES)
    if not raw:
        return []
    names = [part.strip() for part in raw.replace(" ", ",").split(",") if part.strip()]
    unknown = [n for n in names if n not in PHASE_MAP]
    if unknown:
        raise SystemExit(
            f"unknown phase(s): {', '.join(unknown)}\nknown phases: {', '.join(PHASE_NAMES)}"
        )
    return names


def order_phases(names: Sequence[str]) -> List[Phase]:
    """Return phases in canonical dependency order, restricted to ``names``."""
    wanted = set(names)
    return [p for p in PHASES if p.name in wanted]


def check_environment(verbose: bool = True) -> Dict[str, Any]:
    """Report on interpreter/packages/GPU; never fatal."""
    info: Dict[str, Any] = {"python": sys.version.split()[0], "executable": sys.executable}
    for mod in ("torch", "transformers", "datasets", "numpy", "yaml", "matplotlib", "sklearn"):
        try:
            module = __import__(mod)
            info[mod] = getattr(module, "__version__", "present")
        except Exception:
            info[mod] = None
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_devices"] = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        info["cuda_available"] = False
        info["cuda_devices"] = 0
    if verbose:
        print("=" * 78)
        print("Environment")
        print("=" * 78)
        for key, value in info.items():
            print(f"  {key}: {value}")
        if not info.get("cuda_available"):
            print("  note: no CUDA GPU detected -- full reproduction (DPO + PPLM) needs one.")
    return info


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "End-to-end orchestration for the GPT2-medium DPO/toxicity mechanistic "
            "reproduction (probe -> vectors -> interventions/pairs -> DPO -> eval/analysis/unalign)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mode = parser.add_argument_group("pipeline selection")
    mode.add_argument("--phase", "--only", dest="phase", default=None,
                      help="run a single phase by name")
    mode.add_argument("--phases", default=None,
                      help="comma-separated list of phases (dependency order is enforced)")
    mode.add_argument("--all", action="store_true", help="run every phase in order")
    mode.add_argument("--list", action="store_true", help="list available phases and exit")
    mode.add_argument("--status", action="store_true", help="report artifact status and exit")
    mode.add_argument("--env", action="store_true", help="report environment info and exit")
    mode.add_argument("--skip", default=None, help="comma-separated phases to skip")

    cfg = parser.add_argument_group("configuration")
    cfg.add_argument("--config", default=DEFAULT_CONFIG, help="YAML/JSON config path")
    cfg.add_argument("--model", default=None, help=f"base model name/path (default {DEFAULT_MODEL})")
    cfg.add_argument("--dpo-dir", default=None, help="GPT2_DPO checkpoint directory")
    cfg.add_argument("--probe-path", default=None, help="W_Toxic artifact path")
    cfg.add_argument("--vectors-path", default=None, help="toxic-vector artifact path")
    cfg.add_argument("--pairs-path", default=None, help="preference-pair JSONL path")
    cfg.add_argument("--seed", type=int, default=None, help="global random seed")
    cfg.add_argument("--device", default=None, help="torch device (cpu/cuda); default auto")

    run = parser.add_argument_group("execution control")
    run.add_argument("--quick", action="store_true", help="small prompt/pair counts for smoke tests")
    run.add_argument("--dry-run", action="store_true", help="print commands without executing")
    run.add_argument("--force", action="store_true", help="rerun phases even if artifacts exist")
    run.add_argument("--keep-going", action="store_true", help="continue after a failing phase")
    run.add_argument("--in-process", action="store_true",
                     help="call phase main() in-process instead of spawning subprocesses")
    run.add_argument("--max-pairs", type=int, default=None, help="cap pairs for DPO training")
    run.add_argument("--no-alpha-search", action="store_true",
                     help="interventions: use alpha=1 instead of matching post-DPO PPL")
    run.add_argument("--quiet", action="store_true", help="pass --quiet to phase scripts")
    run.add_argument("--summary", default=os.path.join("artifacts", "main_run_summary.json"),
                     help="where to write the run summary JSON")
    run.add_argument("--no-summary", action="store_true", help="do not write the run summary JSON")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list:
        print_plan()
        return 0

    settings = resolve_settings(args)

    if args.env:
        check_environment(verbose=True)
        return 0

    if args.status:
        print_status(settings)
        print()
        print_reference()
        return 0

    # Which phases?
    names: List[str] = []
    if args.all:
        names = list(PHASE_NAMES)
    elif args.phases:
        names = parse_phases(args.phases, all_flag=False)
    elif args.phase:
        names = parse_phases(args.phase, all_flag=False)
    else:
        parser.print_help()
        print()
        print_plan()
        return 0

    skip = set(parse_phases(args.skip, all_flag=False)) if args.skip else set()
    names = [n for n in names if n not in skip]
    phases = order_phases(names)
    if not phases:
        print("[main] nothing to do (all selected phases were skipped)")
        return 0

    print("=" * 78)
    print("GPT2-medium DPO / toxicity reproduction")
    print("=" * 78)
    print(f"  model        : {settings['model']}")
    print(f"  config       : {args.config}")
    print(f"  seed         : {settings['seed']}")
    print(f"  device       : {settings['device'] or 'auto'}")
    print(f"  phases       : {', '.join(p.name for p in phases)}")
    if settings["quick"]:
        print("  mode         : QUICK (reduced counts; smoke test only)")
    if settings["dry_run"]:
        print("  mode         : DRY RUN")
    scope = settings.get("in_scope") or {}
    if isinstance(scope, dict):
        print(f"  in scope     : {scope.get('model', 'gpt2-medium')} "
              f"(llama2={scope.get('llama2', False)}, glu={scope.get('glu', False)})")

    results: List[Dict[str, Any]] = []
    interrupted = False
    try:
        for phase in phases:
            if args.in_process:
                record = run_in_process(phase, args, settings)
            else:
                record = run_phase(phase, args, settings, stop_on_error=not args.keep_going)
            results.append(record)
            if record["status"] in ("failed", "error", "missing-inputs", "missing-script"):
                if not args.keep_going:
                    break
    except KeyboardInterrupt:
        interrupted = True
        print("\n[main] interrupted by user")

    print_report(settings, results)

    if not args.no_summary:
        try:
            path = save_run_summary(settings, results, args.summary)
            print(f"\n[main] wrote run summary -> {os.path.relpath(path, ROOT)}")
        except Exception as exc:  # pragma: no cover - reporting is best effort
            print(f"[main] could not write run summary: {exc}")

    if interrupted:
        return 130
    failures = [r for r in results if r["status"] in ("failed", "error")]
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# Pipeline helpers used elsewhere / interactive
# ---------------------------------------------------------------------------


def run_full_pipeline(quick: bool = False, keep_going: bool = False, **overrides: Any) -> Dict[str, Any]:
    """Programmatic entry point: run all phases and return the run summary.

    >>> run_full_pipeline(quick=True, dry_run=True)   # doctest: +SKIP
    """
    argv: List[str] = ["--all"]
    if quick:
        argv.append("--quick")
    if keep_going:
        argv.append("--keep-going")
    for key, value in overrides.items():
        if value is None or value is False:
            continue
        flag = "--" + key.replace("_", "-")
        if value is True:
            argv.append(flag)
        else:
            argv += [flag, str(value)]
    rc = main(argv)
    return {"returncode": rc, "summary": load_json_safe(overrides.get("summary", "artifacts/main_run_summary.json"))}


def check_prerequisites(settings: Optional[Dict[str, Any]] = None) -> Dict[str, List[str]]:
    """Report which phase input artifacts are missing (ordering sanity check)."""
    report: Dict[str, List[str]] = {}
    for phase in PHASES:
        missing = phase.missing_inputs()
        if missing:
            report[phase.name] = [os.path.relpath(m, ROOT) for m in missing]
    return report


def _print_environment_warning() -> None:
    info = check_environment(verbose=False)
    if not info.get("torch"):
        print("[main] WARNING: torch is not importable; only --list/--status/--dry-run will work.")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:  # pragma: no cover - top-level guard
        traceback.print_exc()
        sys.exit(1)
