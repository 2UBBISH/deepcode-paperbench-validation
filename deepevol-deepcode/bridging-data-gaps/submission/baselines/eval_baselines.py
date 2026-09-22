#!/usr/bin/env python
"""Baseline evaluation scaffolding for the DPMs-ANT reproduction.

The paper (Section 5.2 "Baselines", Tables 1-2, Appendix B.2 Table 4,
Appendix B.4 Table 9) compares DPMs-ANT against:

    * DDPM-PA   (Zhu et al. 2022)  -- 10-shot adaptor-free pipeline built on
                                      the same DDPM/LDM backbones.
    * TGAN      (Wang et al. 2018)
    * TGAN+ADA  (Wang et al. 2018 + Karras et al. 2020)
    * EWC       (Li et al. 2020; Elich et al. 2021 style)
    * CDC       (Ojha et al. 2021)
    * DCL       (Zhao et al. 2022)

This module does not reimplement those methods (they are external codebases).
Instead it provides, in the spirit of the reproduction plan:

    1. A registry describing each baseline: where its code lives, the command
       template used to train it, the command template used to dump generated
       images, the required environment variables and the paper-reported
       numbers for the tasks of interest.
    2. A ``run_baseline`` driver that renders those command templates, executes
       them (or prints them with ``--dry-run``), and collects the generated
       image directories using the *exact same directory layout* the ANT
       pipeline produces, so evaluation is apples-to-apples.
    3. Evaluation wrappers that call the shared DPMs-ANT metrics
       (``dpm_ant.evaluation.evaluate.evaluate_from_dirs`` /
       ``dpm_ant.evaluation.intra_lpips`` / ``dpm_ant.evaluation.fid``) so
       Intra-LPIPS and FID are computed identically for ANT and baselines.
    4. Paper reference tables + a ``compare_to_paper`` helper so a reproduction
       run can be diffed against Tables 1-2, 4 and 9 automatically.
    5. A small CLI
       (``--list``, ``--method``, ``--task``, ``--eval-only``, ``--report``,
        ``--dry-run``, ...).

Usage examples
--------------
    # what is available?
    python baselines/eval_baselines.py --list

    # dry-run the DDPM-PA recipe for FFHQ -> Sunglasses (DDPM backbone)
    python baselines/eval_baselines.py --method ddpm_pa --task ffhq_sunglasses --dry-run

    # score already generated baseline images with the ANT metrics
    python baselines/eval_baselines.py --eval-only \
        --generated-dir outputs/baselines/ddpm_pa/ffhq_sunglasses_ddpm/samples \
        --task ffhq_sunglasses --report outputs/baselines/ddpm_pa_ffhq_sunglasses.json

Everything degrades gracefully: missing codebases, missing datasets and
missing optional metric packages produce warnings (or dry-run command prints)
rather than hard failures, matching the defensive style of
``scripts/run_ablation.py`` and ``scripts/run_eval.sh``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger("dpm_ant.baselines")

# ---------------------------------------------------------------------------
# Repository layout / defaults
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

DEFAULT_CONFIG = os.path.join(_ROOT, "configs", "default.yaml")
PER_TASK_CONFIG = os.path.join(_ROOT, "configs", "per_task.yaml")
CLASSIFIER_CONFIG = os.path.join(_ROOT, "configs", "classifier.yaml")

DEFAULT_OUT_ROOT = os.path.join("outputs", "baselines")
DEFAULT_EXTERNAL_ROOT = os.path.join("external")


# ---------------------------------------------------------------------------
# Task registry (mirrors main.py / configs/per_task.yaml naming)
# ---------------------------------------------------------------------------
TASKS: Dict[str, Dict[str, str]] = {
    # task key -> source / target / default backbone
    "ffhq_babies": {"source": "ffhq", "target": "babies", "backbone": "ddpm"},
    "ffhq_sunglasses": {"source": "ffhq", "target": "sunglasses", "backbone": "ddpm"},
    "ffhq_raphael": {"source": "ffhq", "target": "raphael", "backbone": "ddpm"},
    "ffhq_sketches": {"source": "ffhq", "target": "sketches", "backbone": "ddpm"},
    "ffhq_amedeo": {"source": "ffhq", "target": "amedeo", "backbone": "ddpm"},
    "church_haunted_houses": {
        "source": "lsun_church",
        "target": "haunted_houses",
        "backbone": "ddpm",
    },
    "church_landscape_drawings": {
        "source": "lsun_church",
        "target": "landscape_drawings",
        "backbone": "ddpm",
    },
}

# Alias -> canonical task key.
TASK_ALIASES: Dict[str, str] = {
    "babies": "ffhq_babies",
    "sunglasses": "ffhq_sunglasses",
    "raphael": "ffhq_raphael",
    "raphael_paintings": "ffhq_raphael",
    "sketches": "ffhq_sketches",
    "amedeo": "ffhq_amedeo",
    "amedeo_modigliani": "ffhq_amedeo",
    "haunted_houses": "church_haunted_houses",
    "landscape_drawings": "church_landscape_drawings",
}


def resolve_task(task: str) -> str:
    """Canonicalise a task name.

    ``ffhq_sunglasses_ldm`` -> ``ffhq_sunglasses``; ``Sunglasses`` ->
    ``ffhq_sunglasses``.
    """
    if not task:
        return "ffhq_sunglasses"
    name = str(task).strip().lower().replace("-", "_").replace(" ", "_")
    for suffix in ("_ldm", "_ddpm"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    if name in TASKS:
        return name
    for key, entry in TASKS.items():
        if entry["target"] in name or key in name:
            return key
    return TASK_ALIASES.get(name, name)


def task_backbone(task: str, default: Optional[str] = None) -> str:
    """Recover the backbone encoded in the task name (``*_ldm`` suffix)."""
    if default:
        return str(default).lower()
    raw = str(task or "").lower()
    if raw.endswith("_ldm"):
        return "ldm"
    if raw.endswith("_ddpm"):
        return "ddpm"
    return TASKS.get(resolve_task(task), {}).get("backbone", "ddpm")


# ---------------------------------------------------------------------------
# Paper reference numbers (Tables 1-2, 4, 9)
# ---------------------------------------------------------------------------
# Intra-LPIPS (higher is better). Numbers transcribed from the paper's
# reported tables; `None` means the paper does not report that cell.
PAPER_INTRA_LPIPS: Dict[str, Dict[str, Optional[float]]] = {
    "ffhq_babies": {
        "tgan": 0.548,
        "tgan_ada": 0.567,
        "ewc": 0.541,
        "cdc": 0.585,
        "dcl": 0.601,
        "ddpm_pa": 0.652,
        "ddpm_ant": 0.692,
        "ldm_ant": 0.709,
    },
    "ffhq_sunglasses": {
        "tgan": 0.526,
        "tgan_ada": 0.550,
        "ewc": 0.516,
        "cdc": 0.553,
        "dcl": 0.570,
        "ddpm_pa": 0.584,
        "ddpm_ant": 0.613,
        "ldm_ant": 0.632,
    },
    "ffhq_raphael": {
        "tgan": 0.511,
        "tgan_ada": 0.530,
        "ewc": 0.508,
        "cdc": 0.533,
        "dcl": 0.549,
        "ddpm_pa": 0.556,
        "ddpm_ant": 0.565,
        "ldm_ant": 0.584,
    },
    "church_haunted_houses": {
        "tgan": 0.501,
        "tgan_ada": 0.528,
        "ewc": 0.505,
        "cdc": 0.533,
        "dcl": 0.548,
        "ddpm_pa": 0.594,
        "ddpm_ant": 0.628,
        "ldm_ant": 0.646,
    },
    "church_landscape_drawings": {
        "tgan": 0.562,
        "tgan_ada": 0.579,
        "ewc": 0.574,
        "cdc": 0.613,
        "dcl": 0.629,
        "ddpm_pa": 0.706,
        "ddpm_ant": 0.723,
        "ldm_ant": 0.738,
    },
    # Appendix B.2, Table 4 (ANT only; baselines not tabulated there)
    "ffhq_sketches": {"ddpm_ant": 0.544},
    "ffhq_amedeo": {"ddpm_ant": 0.620},
}

# Intra-LPIPS standard deviations reported in Table 4.
PAPER_INTRA_LPIPS_STD: Dict[str, Dict[str, float]] = {
    "ffhq_sketches": {"ddpm_ant": 0.025},
    "ffhq_amedeo": {"ddpm_ant": 0.021},
}

# FID (lower is better) for the two FID target sets (Section 5.2 / Table 2).
PAPER_FID: Dict[str, Dict[str, Optional[float]]] = {
    "ffhq_babies": {
        "tgan": 63.14,
        "tgan_ada": 57.91,
        "ewc": 60.74,
        "cdc": 53.67,
        "dcl": 51.20,
        "ddpm_pa": 49.83,
        "ddpm_ant": 46.70,
    },
    "ffhq_sunglasses": {
        "tgan": 36.42,
        "tgan_ada": 32.55,
        "ewc": 34.11,
        "cdc": 30.02,
        "dcl": 28.74,
        "ddpm_pa": 24.31,
        "ddpm_ant": 20.06,
    },
}

# Appendix B.4, Table 9 -- user study (average preference %, ANT vs DDPM-PA).
PAPER_USER_STUDY: Dict[str, float] = {
    "ant_preference": 73.35,
    "ddpm_pa_preference": 26.65,
    "num_participants": 60,
}

BASELINE_ORDER: Tuple[str, ...] = ("tgan", "tgan_ada", "ewc", "cdc", "dcl", "ddpm_pa")


# ---------------------------------------------------------------------------
# Baseline registry
# ---------------------------------------------------------------------------
@dataclass
class BaselineSpec:
    """Description of an external baseline method.

    Attributes
    ----------
    name : canonical identifier (e.g. ``ddpm_pa``).
    title : human readable name used in plots/READMEs.
    repo : expected local checkout (relative to the workspace root).
    url : upstream repository URL (documentation only).
    kind : ``"diffusion"`` for DDPM-PA, ``"gan"`` for the StyleGAN2-based
        baselines; used to pick sensible defaults.
    shots : number of target images the method is run with (paper: 10).
    train_cmd : command template executed to train the baseline. Placeholders
        available: ``{repo} {source_dir} {target_dir} {out_dir} {shots}
        {seed} {image_size} {gpu} {python}``.
    sample_cmd : command template that writes generated images into
        ``{samples_dir}``. Empty string means "infer from train_cmd output".
    env : extra environment variables required by the codebase.
    notes : free-form notes surfaced by ``--list``.
    """

    name: str
    title: str
    repo: str
    url: str
    kind: str = "gan"
    shots: int = 10
    train_cmd: str = ""
    sample_cmd: str = ""
    env: Dict[str, str] = field(default_factory=dict)
    notes: str = ""
    paper_intra_lpips: Dict[str, Optional[float]] = field(default_factory=dict)
    paper_fid: Dict[str, Optional[float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_PY = "{python}"
_GPU = "{gpu}"
_IMG = "{image_size}"

BASELINES: Dict[str, BaselineSpec] = {
    # ------------------------------------------------------------------
    # DDPM-PA (Zhu et al. 2022) -- the closest diffusion baseline: it swaps
    # the diffusion U-Net for a target-specific parameterization module while
    # keeping the source model frozen. The paper's *own* pipeline is the
    # reference for Tables 1-2 (0.706 vs 0.723 / 0.738 on Landscape drawings).
    # ------------------------------------------------------------------
    "ddpm_pa": BaselineSpec(
        name="ddpm_pa",
        title="DDPM-PA (Zhu et al. 2022)",
        repo=os.path.join(DEFAULT_EXTERNAL_ROOT, "DDPM-PA"),
        url="https://github.com/zhujiapeng/DDPM-PA",
        kind="diffusion",
        shots=10,
        # Step 1: learn the parameterization module on the 10-shot target set.
        train_cmd=(
            _PY + " -m ddpm_pa.train"
            " --config {repo}/configs/ffhq256.yaml"
            " --source_dir {source_dir}"
            " --target_dir {target_dir}"
            " --shots {shots}"
            " --output_dir {out_dir}/train"
            " --seed {seed}"
            " --gpu {gpu}"
        ),
        # Step 2: dump 1,000 samples (used for Intra-LPIPS) and the larger FID
        # set. Kept identical to scripts/sample.py's output layout.
        sample_cmd=(
            _PY + " -m ddpm_pa.sample"
            " --model_path {out_dir}/train/model.pt"
            " --num_samples {num_samples}"
            " --batch_size {batch_size}"
            " --image_size " + _IMG +
            " --output_dir {samples_dir}"
            " --seed {seed}"
            " --gpu {gpu}"
        ),
        env={"CUDA_VISIBLE_DEVICES": "{gpu}"},
        notes=(
            "Wraps the official DDPM-PA repository. Table 1 reference: "
            "Intra-LPIPS 0.706 on LSUN Church -> Landscape drawings (DDPM-ANT "
            "0.723, LDM-ANT 0.738)."
        ),
        paper_intra_lpips={
            t: PAPER_INTRA_LPIPS[t].get("ddpm_pa", None) for t in TASKS
        },
        paper_fid={t: PAPER_FID.get(t, {}).get("ddpm_pa", None) for t in TASKS},
    ),
    # ------------------------------------------------------------------
    # GAN baselines: StyleGAN2 backbone with the method-specific adaptation.
    # ------------------------------------------------------------------
    "tgan": BaselineSpec(
        name="tgan",
        title="TGAN (Wang et al. 2018)",
        repo=os.path.join(DEFAULT_EXTERNAL_ROOT, "TGAN"),
        url="https://github.com/akanimax/TGAN",
        kind="gan",
        train_cmd=(
            _PY + " train.py"
            " --data_dir {target_dir}"
            " --source_ckpt {out_dir}/source.pth"
            " --shots {shots}"
            " --output_dir {out_dir}/train"
            " --seed {seed}"
            " --gpu {gpu}"
        ),
        sample_cmd=(
            _PY + " generate.py"
            " --checkpoint {out_dir}/train/gen.pth"
            " --num_samples {num_samples}"
            " --output_dir {samples_dir}"
            " --gpu {gpu}"
        ),
        paper_intra_lpips={t: PAPER_INTRA_LPIPS[t].get("tgan", None) for t in TASKS},
        paper_fid={t: PAPER_FID.get(t, {}).get("tgan", None) for t in TASKS},
    ),
    "tgan_ada": BaselineSpec(
        name="tgan_ada",
        title="TGAN + ADA (Wang et al. 2018; Karras et al. 2020)",
        repo=os.path.join(DEFAULT_EXTERNAL_ROOT, "TGAN-ADA"),
        url="https://github.com/akanimax/TGAN",
        kind="gan",
        train_cmd=(
            _PY + " train.py"
            " --data_dir {target_dir}"
            " --source_ckpt {out_dir}/source.pth"
            " --shots {shots}"
            " --ada"
            " --output_dir {out_dir}/train"
            " --seed {seed}"
            " --gpu {gpu}"
        ),
        sample_cmd=(
            _PY + " generate.py"
            " --checkpoint {out_dir}/train/gen.pth"
            " --num_samples {num_samples}"
            " --output_dir {samples_dir}"
            " --gpu {gpu}"
        ),
        paper_intra_lpips={t: PAPER_INTRA_LPIPS[t].get("tgan_ada", None) for t in TASKS},
        paper_fid={t: PAPER_FID.get(t, {}).get("tgan_ada", None) for t in TASKS},
    ),
    "ewc": BaselineSpec(
        name="ewc",
        title="EWC (Elastic Weight Consolidation)",
        repo=os.path.join(DEFAULT_EXTERNAL_ROOT, "EWC"),
        url="https://github.com/joansj/hat",
        kind="gan",
        train_cmd=(
            _PY + " train_ewc.py"
            " --source_ckpt {out_dir}/source.pth"
            " --target_dir {target_dir}"
            " --shots {shots}"
            " --output_dir {out_dir}/train"
            " --seed {seed}"
            " --gpu {gpu}"
        ),
        sample_cmd=(
            _PY + " generate.py"
            " --checkpoint {out_dir}/train/gen.pth"
            " --num_samples {num_samples}"
            " --output_dir {samples_dir}"
            " --gpu {gpu}"
        ),
        paper_intra_lpips={t: PAPER_INTRA_LPIPS[t].get("ewc", None) for t in TASKS},
        paper_fid={t: PAPER_FID.get(t, {}).get("ewc", None) for t in TASKS},
    ),
    "cdc": BaselineSpec(
        name="cdc",
        title="CDC (Ojha et al. 2021)",
        repo=os.path.join(DEFAULT_EXTERNAL_ROOT, "CDC"),
        url="https://github.com/utkarshojha/few-shot-gan-adaptation",
        kind="gan",
        train_cmd=(
            _PY + " train_cdc.py"
            " --source_ckpt {out_dir}/source.pth"
            " --target_dir {target_dir}"
            " --shots {shots}"
            " --output_dir {out_dir}/train"
            " --seed {seed}"
            " --gpu {gpu}"
        ),
        sample_cmd=(
            _PY + " generate.py"
            " --checkpoint {out_dir}/train/gen.pth"
            " --num_samples {num_samples}"
            " --output_dir {samples_dir}"
            " --gpu {gpu}"
        ),
        paper_intra_lpips={t: PAPER_INTRA_LPIPS[t].get("cdc", None) for t in TASKS},
        paper_fid={t: PAPER_FID.get(t, {}).get("cdc", None) for t in TASKS},
    ),
    "dcl": BaselineSpec(
        name="dcl",
        title="DCL (Zhao et al. 2022)",
        repo=os.path.join(DEFAULT_EXTERNAL_ROOT, "DCL"),
        url="https://github.com/sjtuytc/DCL",
        kind="gan",
        train_cmd=(
            _PY + " train.py"
            " --source_ckpt {out_dir}/source.pth"
            " --target_dir {target_dir}"
            " --shots {shots}"
            " --output_dir {out_dir}/train"
            " --seed {seed}"
            " --gpu {gpu}"
        ),
        sample_cmd=(
            _PY + " generate.py"
            " --checkpoint {out_dir}/train/gen.pth"
            " --num_samples {num_samples}"
            " --output_dir {samples_dir}"
            " --gpu {gpu}"
        ),
        paper_intra_lpips={t: PAPER_INTRA_LPIPS[t].get("dcl", None) for t in TASKS},
        paper_fid={t: PAPER_FID.get(t, {}).get("dcl", None) for t in TASKS},
    ),
}


def get_baseline(name: str) -> BaselineSpec:
    """Look up a baseline spec, raising a helpful error for unknown names."""
    key = str(name or "").strip().lower().replace("-", "_")
    if key in ("tgan+ada", "tgan_ada"):
        key = "tgan_ada"
    if key not in BASELINES:
        raise KeyError(
            f"unknown baseline {name!r}; available: {sorted(BASELINES)}"
        )
    return BASELINES[key]


def list_baselines() -> List[Dict[str, Any]]:
    """Return the registry as plain dicts (for ``--list`` / docs)."""
    out = []
    for name in BASELINE_ORDER:
        spec = BASELINES[name]
        out.append(
            {
                "name": spec.name,
                "title": spec.title,
                "kind": spec.kind,
                "repo": spec.repo,
                "url": spec.url,
                "available": os.path.isdir(os.path.join(_ROOT, spec.repo)),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def _load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    try:
        import yaml  # type: ignore
    except Exception:  # pragma: no cover - PyYAML optional
        LOGGER.warning("PyYAML unavailable; skipping config %s", path)
        return {}
    if not os.path.isfile(path):
        LOGGER.warning("config file not found: %s", path)
        return {}
    try:
        with open(path, "r") as fh:
            data = yaml.safe_load(fh) or {}
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # pragma: no cover - malformed yaml
        LOGGER.warning("failed to parse %s: %s", path, exc)
        return {}


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(
    config_path: Optional[str] = None,
    per_task_path: Optional[str] = None,
    classifier_path: Optional[str] = None,
    task: Optional[str] = None,
) -> Dict[str, Any]:
    """Merge default + per_task (+ classifier) configs like the main scripts."""
    cfg = _load_yaml(config_path or DEFAULT_CONFIG)
    per_task = _load_yaml(per_task_path or PER_TASK_CONFIG)
    if isinstance(per_task.get("defaults"), dict):
        cfg = _deep_update(cfg, per_task["defaults"])
    cfg = _deep_update(cfg, {k: v for k, v in per_task.items() if k != "defaults"})
    cfg = _deep_update(cfg, _load_yaml(classifier_path or CLASSIFIER_CONFIG))
    if task:
        key = resolve_task(task)
        entry = (cfg.get("tasks") or {}).get(key)
        if isinstance(entry, dict):
            cfg["task_cfg"] = entry
        cfg["task"] = key
    return cfg


def _resolve_dir(cfg: Dict[str, Any], kind: str, name: str) -> Optional[str]:
    """Resolve a dataset directory from the (nested) config."""
    data = cfg.get("data") or {}
    candidates = [
        (data.get("source_dirs") or {}).get(name) if kind == "source" else None,
        (data.get("target_dirs") or {}).get(name) if kind == "target" else None,
        (data.get("fid_target_dirs") or {}).get(name) if kind == "target" else None,
        (data.get("dirs") or {}).get(name),
        (data.get("targets") or {}).get(name) if kind == "target" else None,
        data.get(name),
    ]
    for cand in candidates:
        if cand and os.path.isdir(str(cand)):
            return str(cand)
    for cand in candidates:  # fall back to first non-empty even if missing
        if cand:
            return str(cand)
    return None


# ---------------------------------------------------------------------------
# Command rendering / execution
# ---------------------------------------------------------------------------
def build_command(
    template: str,
    spec: BaselineSpec,
    *,
    source_dir: Optional[str] = None,
    target_dir: Optional[str] = None,
    out_dir: str = ".",
    samples_dir: Optional[str] = None,
    shots: Optional[int] = None,
    seed: int = 0,
    image_size: int = 256,
    num_samples: int = 1000,
    batch_size: int = 16,
    gpu: str = "0",
    python: Optional[str] = None,
) -> List[str]:
    """Render a command template into an argv list.

    Templates are tokenised with :mod:`shlex` so quoted paths survive.
    """
    if not template:
        return []
    repo = os.path.join(_ROOT, spec.repo)
    mapping = {
        "repo": repo,
        "source_dir": source_dir or "",
        "target_dir": target_dir or "",
        "out_dir": out_dir,
        "samples_dir": samples_dir or os.path.join(out_dir, "samples"),
        "shots": int(shots if shots is not None else spec.shots),
        "seed": int(seed),
        "image_size": int(image_size),
        "num_samples": int(num_samples),
        "batch_size": int(batch_size),
        "gpu": str(gpu),
        "python": python or sys.executable or "python",
    }
    rendered = template.format(**mapping)
    argv = shlex.split(rendered)
    # Commands are expected to run from inside the baseline checkout.
    return argv


def run_command(
    argv: Sequence[str],
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    dry_run: bool = False,
    log_file: Optional[str] = None,
    check: bool = False,
) -> Dict[str, Any]:
    """Execute (or just print) a command, capturing status/time/output tail."""
    argv = [str(a) for a in argv]
    record: Dict[str, Any] = {
        "command": " ".join(shlex.quote(a) for a in argv),
        "cwd": cwd,
        "dry_run": bool(dry_run),
        "returncode": None,
        "elapsed_sec": 0.0,
    }
    if not argv:
        record["skipped"] = "empty command"
        return record
    if dry_run:
        print("[dry-run] " + record["command"])
        record["returncode"] = 0
        return record

    run_env = dict(os.environ)
    for key, value in (env or {}).items():
        run_env[key] = str(value).format(gpu=env.get("gpu", "0"))
    if env and "CUDA_VISIBLE_DEVICES" in env:
        run_env["CUDA_VISIBLE_DEVICES"] = str(env["CUDA_VISIBLE_DEVICES"])

    start = time.time()
    try:
        proc = subprocess.run(
            argv,
            cwd=cwd or None,
            env=run_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        record["returncode"] = proc.returncode
        tail = (proc.stdout or "").strip().splitlines()[-20:]
        record["output_tail"] = tail
        if log_file:
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
            with open(log_file, "a") as fh:
                fh.write("$ " + record["command"] + "\n")
                fh.write(proc.stdout or "")
                fh.write("\n")
    except FileNotFoundError as exc:
        record["returncode"] = 127
        record["error"] = str(exc)
        LOGGER.warning("command not found: %s", argv[0] if argv else "?")
    except Exception as exc:  # pragma: no cover - unexpected failure
        record["returncode"] = 1
        record["error"] = str(exc)
        LOGGER.warning("command failed: %s", exc)
    record["elapsed_sec"] = round(time.time() - start, 3)

    if check and record["returncode"] not in (0, None):
        raise RuntimeError(
            f"command failed ({record['returncode']}): {record['command']}"
        )
    return record


# ---------------------------------------------------------------------------
# Baseline run driver
# ---------------------------------------------------------------------------
@dataclass
class BaselineRunConfig:
    """Settings for one baseline run/evaluation."""

    method: str = "ddpm_pa"
    task: str = "ffhq_sunglasses"
    backbone: str = "ddpm"
    out_root: str = DEFAULT_OUT_ROOT
    source_dir: Optional[str] = None
    target_dir: Optional[str] = None
    fid_target_dir: Optional[str] = None
    shots: int = 10
    seed: int = 0
    gpu: str = "0"
    image_size: int = 256
    num_samples: int = 1000
    fid_num_samples: Optional[int] = None
    batch_size: int = 16
    python: Optional[str] = None
    method_root: Optional[str] = None  # override for the baseline checkout
    dry_run: bool = False
    skip_train: bool = False
    skip_sample: bool = False
    verbose: bool = True

    @property
    def run_dir(self) -> str:
        return os.path.join(
            self.out_root, self.method, f"{resolve_task(self.task)}_{self.backbone}"
        )

    @property
    def samples_dir(self) -> str:
        return os.path.join(self.run_dir, "samples")

    @property
    def fid_samples_dir(self) -> str:
        return os.path.join(self.run_dir, "samples_fid")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_baseline(
    method: str,
    task: str,
    cfg: Optional[Dict[str, Any]] = None,
    *,
    backbone: Optional[str] = None,
    out_root: Optional[str] = None,
    source_dir: Optional[str] = None,
    target_dir: Optional[str] = None,
    shots: Optional[int] = None,
    seed: int = 0,
    gpu: str = "0",
    num_samples: int = 1000,
    batch_size: int = 16,
    dry_run: bool = False,
    skip_train: bool = False,
    skip_sample: bool = False,
    check_repo: bool = True,
    **overrides: Any,
) -> Dict[str, Any]:
    """Train + sample a baseline method with its official codebase.

    Returns a JSON-serialisable record describing the rendered commands, their
    exit status, elapsed time and the produced sample directory. Nothing here
    performs metric computation -- see :func:`evaluate_baseline`.
    """
    cfg = cfg or {}
    spec = get_baseline(method)
    task_key = resolve_task(task)
    bb = backbone or task_backbone(task)

    run_cfg = BaselineRunConfig(
        method=spec.name,
        task=task_key,
        backbone=bb,
        out_root=out_root or cfg.get("out_root") or DEFAULT_OUT_ROOT,
        source_dir=source_dir or _resolve_dir(cfg, "source", TASKS.get(task_key, {}).get("source", "")),
        target_dir=target_dir or _resolve_dir(cfg, "target", TASKS.get(task_key, {}).get("target", "")),
        shots=int(shots if shots is not None else spec.shots),
        seed=int(seed),
        gpu=str(gpu),
        num_samples=int(num_samples),
        batch_size=int(batch_size),
        dry_run=bool(dry_run),
        skip_train=bool(skip_train),
        skip_sample=bool(skip_sample),
        **{k: v for k, v in overrides.items() if hasattr(BaselineRunConfig, k)},
    )

    repo_dir = os.path.join(_ROOT, spec.repo)
    os.makedirs(run_cfg.run_dir, exist_ok=True)
    log_file = os.path.join(run_cfg.run_dir, "run.log")

    record: Dict[str, Any] = {
        "method": spec.name,
        "title": spec.title,
        "task": task_key,
        "backbone": bb,
        "shots": run_cfg.shots,
        "seed": run_cfg.seed,
        "run_dir": run_cfg.run_dir,
        "samples_dir": run_cfg.samples_dir,
        "repo": spec.repo,
        "url": spec.url,
        "config": run_cfg.to_dict(),
        "commands": [],
    }

    if check_repo and not os.path.isdir(repo_dir) and not dry_run:
        record["status"] = "missing_repo"
        record["message"] = (
            f"baseline checkout not found at {repo_dir}; clone {spec.url} into "
            f"{spec.repo} (see baselines/README.md) or pass --dry-run to "
            f"inspect the command template."
        )
        LOGGER.warning(record["message"])
        return record

    # Command templates: allow per-method overrides through the config.
    method_cfg = ((cfg.get("baselines") or {}).get(spec.name) or {})
    train_tpl = method_cfg.get("train_cmd") or spec.train_cmd
    sample_tpl = method_cfg.get("sample_cmd") or spec.sample_cmd

    env = dict(spec.env)
    env["gpu"] = run_cfg.gpu
    cwd = repo_dir if os.path.isdir(repo_dir) else _ROOT

    # --- train -----------------------------------------------------------
    if run_cfg.skip_train:
        record["commands"].append({"command": "", "skipped": "skip_train"})
    else:
        argv = build_command(
            train_tpl,
            spec,
            source_dir=run_cfg.source_dir,
            target_dir=run_cfg.target_dir,
            out_dir=run_cfg.run_dir,
            samples_dir=run_cfg.samples_dir,
            shots=run_cfg.shots,
            seed=run_cfg.seed,
            image_size=run_cfg.image_size,
            num_samples=run_cfg.num_samples,
            batch_size=run_cfg.batch_size,
            gpu=run_cfg.gpu,
            python=run_cfg.python,
        )
        record["commands"].append(
            run_command(argv, cwd=cwd, env=env, dry_run=run_cfg.dry_run, log_file=log_file)
        )

    # --- sample (1,000 images for Intra-LPIPS) ---------------------------
    if run_cfg.skip_sample:
        record["commands"].append({"command": "", "skipped": "skip_sample"})
    else:
        argv = build_command(
            sample_tpl,
            spec,
            source_dir=run_cfg.source_dir,
            target_dir=run_cfg.target_dir,
            out_dir=run_cfg.run_dir,
            samples_dir=run_cfg.samples_dir,
            shots=run_cfg.shots,
            seed=run_cfg.seed,
            image_size=run_cfg.image_size,
            num_samples=run_cfg.num_samples,
            batch_size=run_cfg.batch_size,
            gpu=run_cfg.gpu,
            python=run_cfg.python,
        )
        record["commands"].append(
            run_command(argv, cwd=cwd, env=env, dry_run=run_cfg.dry_run, log_file=log_file)
        )

    # --- sample (larger FID set, only when a target set is available) ----
    fid_size = None
    if run_cfg.fid_target_dir or cfg.get("evaluation", {}).get("fid"):
        fid_size = int(
            (cfg.get("evaluation", {}).get("fid") or {}).get(
                "num_samples", 2500
            )
        )
    if fid_size and not run_cfg.skip_sample:
        argv = build_command(
            sample_tpl,
            spec,
            source_dir=run_cfg.source_dir,
            target_dir=run_cfg.target_dir,
            out_dir=run_cfg.run_dir,
            samples_dir=run_cfg.fid_samples_dir,
            shots=run_cfg.shots,
            seed=run_cfg.seed,
            image_size=run_cfg.image_size,
            num_samples=fid_size,
            batch_size=run_cfg.batch_size,
            gpu=run_cfg.gpu,
            python=run_cfg.python,
        )
        record["commands"].append(
            run_command(argv, cwd=cwd, env=env, dry_run=run_cfg.dry_run, log_file=log_file)
        )
        record["fid_samples_dir"] = run_cfg.fid_samples_dir
        record["fid_num_samples"] = fid_size

    codes = [
        c.get("returncode") for c in record["commands"] if c.get("command")
    ]
    if not codes:
        record["status"] = "no_command"
    elif all(code == 0 for code in codes):
        record["status"] = "ok"
    else:
        record["status"] = "failed"

    record["elapsed_sec"] = round(
        sum(float(c.get("elapsed_sec") or 0.0) for c in record["commands"]), 3
    )
    return record


# ---------------------------------------------------------------------------
# Evaluation with the shared DPMs-ANT metrics
# ---------------------------------------------------------------------------
def _list_images(root: Optional[str]) -> int:
    if not root or not os.path.isdir(root):
        return 0
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".npy", ".pt")
    count = 0
    for _, _, files in os.walk(root):
        count += sum(1 for f in files if f.lower().endswith(exts))
    return count


def evaluate_baseline(
    generated_dir: str,
    task: str,
    cfg: Optional[Dict[str, Any]] = None,
    *,
    reference_dir: Optional[str] = None,
    fid_reference_dir: Optional[str] = None,
    backbone: Optional[str] = None,
    num_samples: Optional[int] = None,
    compute_intra_lpips: bool = True,
    compute_fid: bool = True,
    fid_backend: Optional[str] = None,
    device: Optional[str] = None,
    return_details: bool = False,
) -> Dict[str, Any]:
    """Score a directory of baseline images with Intra-LPIPS and/or FID.

    Uses ``dpm_ant.evaluation.evaluate.evaluate_from_dirs`` when available so
    the numbers are directly comparable with the ANT pipeline; falls back to
    the individual metric modules.
    """
    cfg = cfg or {}
    task_key = resolve_task(task)
    bb = backbone or task_backbone(task)
    report: Dict[str, Any] = {
        "method": "baseline",
        "task": task_key,
        "backbone": bb,
        "generated_dir": generated_dir,
        "num_generated": _list_images(generated_dir),
    }

    if not os.path.isdir(generated_dir):
        report["status"] = "missing_generated_dir"
        report["message"] = f"no generated images at {generated_dir}"
        LOGGER.warning(report["message"])
        return report

    eval_cfg = dict(cfg)
    eval_cfg.setdefault("device", device)
    if fid_backend:
        eval_cfg.setdefault("evaluation", {})
        if isinstance(eval_cfg["evaluation"], dict):
            eval_cfg["evaluation"].setdefault("fid", {})
            if isinstance(eval_cfg["evaluation"]["fid"], dict):
                eval_cfg["evaluation"]["fid"]["backend"] = fid_backend

    # Preferred path: the shared end-to-end directory evaluator.
    try:
        from dpm_ant.evaluation.evaluate import evaluate_from_dirs  # type: ignore

        fid_dirs = None
        if fid_reference_dir:
            fid_dirs = fid_reference_dir
            if isinstance(fid_dirs, str):
                fid_dirs = {"fid": fid_dirs}
        out = evaluate_from_dirs(
            generated_dir=generated_dir,
            reference_dir=reference_dir or fid_reference_dir,
            fid_reference_dirs=fid_dirs,
            cfg=eval_cfg,
            backbone=bb,
            task=task_key,
            num_samples=num_samples,
            device=device,
            compute_intra_lpips=compute_intra_lpips,
            compute_fid=compute_fid,
            return_details=return_details,
        )
        if isinstance(out, dict):
            report.update(out)
            report["metric_source"] = "evaluate_from_dirs"
            report["status"] = "ok"
            return report
    except Exception as exc:  # pragma: no cover - optional dependency
        LOGGER.warning("evaluate_from_dirs unavailable (%s); using metrics directly", exc)

    # Fallback: call the individual metric modules.
    status_ok = True
    if compute_intra_lpips:
        try:
            from dpm_ant.evaluation.intra_lpips import (  # type: ignore
                compute_intra_lpips as _il,
                load_reference_images,
            )

            n = int(num_samples or (cfg.get("evaluation", {}).get("intra_lpips", {}) or {}).get("num_samples", 1000))
            gen = load_reference_images(generated_dir, size=256, limit=n)
            ref = None
            if reference_dir and os.path.isdir(reference_dir):
                ref = load_reference_images(reference_dir, size=256)
            res = _il(gen, train_images=ref, train_dir=None if ref is not None else reference_dir,
                      num_samples=n, device=device or "cpu",
                      return_details=return_details)
            report["intra_lpips"] = float(res.get("intra_lpips", float("nan")))
            if return_details:
                report["intra_lpips_details"] = {
                    k: v for k, v in res.items() if k not in ("intra_lpips",)
                }
        except Exception as exc:
            LOGGER.warning("Intra-LPIPS failed: %s", exc)
            status_ok = False
            report["intra_lpips"] = None

    if compute_fid:
        try:
            from dpm_ant.evaluation.fid import compute_fid_from_dirs  # type: ignore

            n_fid = int(
                num_samples
                or (cfg.get("evaluation", {}).get("fid", {}) or {}).get("num_samples", 2500)
            )
            res = compute_fid_from_dirs(
                generated_dir,
                fid_reference_dir or reference_dir,
                cfg=eval_cfg,
                device=device,
                backend=fid_backend,
                target=task_key,
                num_samples=n_fid,
                return_details=return_details,
            )
            report["fid"] = float(res.get("fid", float("nan")))
            if return_details:
                report["fid_details"] = {
                    k: v for k, v in res.items() if k not in ("fid",)
                }
        except Exception as exc:
            LOGGER.warning("FID failed: %s", exc)
            status_ok = False
            report["fid"] = None

    report["metric_source"] = "direct"
    report["status"] = "ok" if status_ok else "partial"
    return report


# ---------------------------------------------------------------------------
# Aggregation + paper comparison
# ---------------------------------------------------------------------------
def compare_to_paper(
    method: str,
    task: str,
    intra_lpips: Optional[float] = None,
    fid: Optional[float] = None,
    intra_lpips_std: Optional[float] = None,
    rel_tol: float = 0.05,
    abs_tol_intra: float = 0.02,
    abs_tol_fid: float = 2.0,
) -> Dict[str, Any]:
    """Diff measured baseline numbers against the paper's reported tables."""
    key = resolve_task(task)
    name = get_baseline(method).name
    out: Dict[str, Any] = {"task": key, "method": name, "notes": []}

    ref_il = PAPER_INTRA_LPIPS.get(key, {}).get(name)
    out["paper_intra_lpips"] = ref_il
    if intra_lpips is not None and ref_il is not None:
        out["intra_lpips"] = float(intra_lpips)
        diff = float(intra_lpips) - float(ref_il)
        out["intra_lpips_diff"] = diff
        out["intra_lpips_within_tolerance"] = bool(
            abs(diff) <= max(abs_tol_intra, rel_tol * abs(float(ref_il)))
        )
        if not out["intra_lpips_within_tolerance"]:
            out["notes"].append(
                "Intra-LPIPS differs from the paper; keep in mind CDC/DDPM-PA "
                "numbers are sensitive to the LPIPS backbone (AlexNet)."
            )
    if intra_lpips_std is not None:
        out["intra_lpips_std"] = float(intra_lpips_std)

    ref_fid = PAPER_FID.get(key, {}).get(name)
    out["paper_fid"] = ref_fid
    if fid is not None and ref_fid is not None:
        out["fid"] = float(fid)
        diff = float(fid) - float(ref_fid)
        out["fid_diff"] = diff
        out["fid_within_tolerance"] = bool(
            abs(diff) <= max(abs_tol_fid, rel_tol * abs(float(ref_fid)))
        )
        if not out["fid_within_tolerance"]:
            out["notes"].append(
                "FID differs from the paper; verify the reference set size "
                "(Sunglasses 2.5k / Babies 2.7k) and the clean-fid backend."
            )
    return out


def summarize_matrix(
    results: Sequence[Dict[str, Any]], metric: str = "intra_lpips"
) -> Dict[str, Dict[str, Optional[float]]]:
    """Group baseline records into a ``{task: {method: value}}`` table."""
    table: Dict[str, Dict[str, Optional[float]]] = {}
    for rec in results or []:
        if not isinstance(rec, dict):
            continue
        task = resolve_task(rec.get("task", ""))
        method = rec.get("method") or "unknown"
        if method in ("baseline",):
            method = rec.get("baseline_method") or method
        value = rec.get(metric)
        table.setdefault(task, {})[method] = (
            float(value) if value is not None else None
        )
    return table


def format_report(report: Dict[str, Any]) -> str:
    """Human-readable rendering of a baseline evaluation report."""
    lines: List[str] = []
    lines.append("=" * 72)
    lines.append("Baseline evaluation report")
    lines.append("=" * 72)
    lines.append(f"task           : {report.get('task')}")
    lines.append(f"backbone       : {report.get('backbone')}")
    lines.append(f"generated dir  : {report.get('generated_dir')}")
    lines.append(f"num generated  : {report.get('num_generated')}")
    for metric in ("intra_lpips", "fid"):
        if report.get(metric) is not None:
            value = report[metric]
            paper = report.get(f"paper_{metric}")
            extra = f"   (paper: {paper})" if paper is not None else ""
            lines.append(f"{metric:15s}: {value:.4f}{extra}")
    for row in report.get("table", []) or []:
        lines.append(
            f"  {row.get('task'):28s} {row.get('method'):10s} "
            f"intra_lpips={row.get('intra_lpips')} fid={row.get('fid')}"
        )
    if report.get("notes"):
        lines.append("notes:")
        for note in report["notes"]:
            lines.append(f"  - {note}")
    lines.append("=" * 72)
    return "\n".join(lines)


def save_report(report: Dict[str, Any], path: str, indent: int = 2) -> str:
    """Persist a report to JSON, creating parent directories."""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def _default(obj: Any) -> Any:
        try:
            import numpy as np  # type: ignore

            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except Exception:
            pass
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if hasattr(obj, "item"):
            try:
                return obj.item()
            except Exception:
                pass
        return str(obj)

    with open(path, "w") as fh:
        json.dump(report, fh, indent=indent, default=_default)
    return path


def print_paper_tables() -> str:
    """Render the paper's baseline tables (for ``--paper``)."""
    lines: List[str] = []
    lines.append("Intra-LPIPS (higher is better) -- Tables 1 & 4")
    header = f"{'task':30s}" + "".join(f"{m:>10s}" for m in BASELINE_ORDER + ("ddpm_ant", "ldm_ant"))
    lines.append(header)
    for task, row in PAPER_INTRA_LPIPS.items():
        cells = "".join(
            f"{(row[m] if row.get(m) is not None else '-')!s:>10s}"
            for m in BASELINE_ORDER + ("ddpm_ant", "ldm_ant")
        )
        lines.append(f"{task:30s}{cells}")
    lines.append("")
    lines.append("FID (lower is better) -- Table 2")
    header = f"{'task':30s}" + "".join(f"{m:>10s}" for m in BASELINE_ORDER + ("ddpm_ant",))
    lines.append(header)
    for task, row in PAPER_FID.items():
        cells = "".join(
            f"{(row[m] if row.get(m) is not None else '-')!s:>10s}"
            for m in BASELINE_ORDER + ("ddpm_ant",)
        )
        lines.append(f"{task:30s}{cells}")
    lines.append("")
    lines.append("User study -- Appendix B.4, Table 9")
    for key, value in PAPER_USER_STUDY.items():
        lines.append(f"  {key}: {value}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="baselines/eval_baselines.py",
        description=(
            "Run and/or evaluate the DPMs-ANT baselines (DDPM-PA, TGAN, "
            "TGAN+ADA, EWC, CDC, DCL) with the same Intra-LPIPS / FID metrics."
        ),
    )
    parser.add_argument("--list", action="store_true", help="list registered baselines")
    parser.add_argument("--paper", action="store_true", help="print the paper's baseline tables")
    parser.add_argument("--method", default="ddpm_pa", help="baseline name (e.g. ddpm_pa, tgan, dcl)")
    parser.add_argument("--methods", nargs="*", default=None, help="several baselines at once")
    parser.add_argument("--tasks", nargs="*", default=None, help="several tasks at once")
    parser.add_argument("--task", default="ffhq_sunglasses", help="task key (e.g. ffhq_sunglasses)")
    parser.add_argument("--backbone", default=None, choices=[None, "ddpm", "ldm"])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--per-task-config", default=PER_TASK_CONFIG)
    parser.add_argument("--classifier-config", default=CLASSIFIER_CONFIG)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--generated-dir", default=None, help="score this directory instead of generating")
    parser.add_argument("--reference-dir", default=None, help="10-shot reference images for Intra-LPIPS")
    parser.add_argument("--fid-reference-dir", default=None, help="larger target set for FID")
    parser.add_argument("--source-dir", default=None)
    parser.add_argument("--target-dir", default=None)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--fid-backend", default=None, choices=[None, "clean-fid", "pytorch-fid", "torch"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--eval-only", action="store_true", help="skip training/sampling, only evaluate --generated-dir")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-sample", action="store_true")
    parser.add_argument("--no-intra-lpips", action="store_true")
    parser.add_argument("--no-fid", action="store_true")
    parser.add_argument("--report", default=None, help="path to write the JSON report")
    parser.add_argument("--dry-run", action="store_true", help="print commands without executing")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.list:
        for row in list_baselines():
            status = "checkout found" if row["available"] else "checkout MISSING"
            print(f"{row['name']:10s} {row['title']:45s} [{status}] {row['url']}")
        print("\nPaper reference tables:\n")
        print(print_paper_tables())
        return 0

    if args.paper:
        print(print_paper_tables())
        return 0

    # multi-task / multi-method drivers
    tasks = args.tasks or [args.task]
    methods = args.methods or [args.method]

    results: List[Dict[str, Any]] = []
    for task in tasks:
        cfg = load_config(
            args.config, args.per_task_config, args.classifier_config, task=task
        )
        for method in methods:
            spec = get_baseline(method)

            if args.eval_only or args.generated_dir:
                gen_dir = args.generated_dir or os.path.join(
                    args.out_root,
                    spec.name,
                    f"{resolve_task(task)}_{args.backbone or task_backbone(task)}",
                    "samples",
                )
                report = evaluate_baseline(
                    gen_dir,
                    task,
                    cfg,
                    reference_dir=args.reference_dir,
                    fid_reference_dir=args.fid_reference_dir,
                    backbone=args.backbone,
                    num_samples=args.num_samples,
                    compute_intra_lpips=not args.no_intra_lpips,
                    compute_fid=not args.no_fid,
                    fid_backend=args.fid_backend,
                    device=args.device,
                )
                report["baseline_method"] = spec.name
                report.update(
                    compare_to_paper(
                        spec.name,
                        task,
                        intra_lpips=report.get("intra_lpips"),
                        fid=report.get("fid"),
                        intra_lpips_std=report.get("intra_lpips_std"),
                    )
                )
                results.append(report)
                print(format_report(report))
                continue

            run_rec = run_baseline(
                spec.name,
                task,
                cfg,
                backbone=args.backbone,
                out_root=args.out_root,
                source_dir=args.source_dir,
                target_dir=args.target_dir,
                shots=args.shots,
                seed=args.seed,
                gpu=args.gpu,
                num_samples=args.num_samples,
                batch_size=args.batch_size,
                dry_run=args.dry_run,
                skip_train=args.skip_train,
                skip_sample=args.skip_sample,
            )
            print(f"[{spec.name}/{resolve_task(task)}] status: {run_rec.get('status')}")
            if args.dry_run:
                for cmd in run_rec.get("commands", []):
                    if cmd.get("command"):
                        print("  " + cmd["command"])
            results.append(run_rec)

            # Evaluate right away when sampling produced images.
            samples_dir = run_rec.get("samples_dir")
            if (
                not args.dry_run
                and samples_dir
                and os.path.isdir(samples_dir)
                and _list_images(samples_dir) > 0
            ):
                report = evaluate_baseline(
                    samples_dir,
                    task,
                    cfg,
                    reference_dir=args.reference_dir,
                    fid_reference_dir=args.fid_reference_dir,
                    backbone=args.backbone,
                    num_samples=args.num_samples,
                    compute_intra_lpips=not args.no_intra_lpips,
                    compute_fid=not args.no_fid,
                    fid_backend=args.fid_backend,
                    device=args.device,
                )
                report["baseline_method"] = spec.name
                report.update(
                    compare_to_paper(
                        spec.name,
                        task,
                        intra_lpips=report.get("intra_lpips"),
                        fid=report.get("fid"),
                    )
                )
                results.append(report)
                print(format_report(report))

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "config": args.config,
            "per_task_config": args.per_task_config,
            "classifier_config": args.classifier_config,
        },
        "tasks": tasks,
        "methods": methods,
        "paper_intra_lpips": PAPER_INTRA_LPIPS,
        "paper_fid": PAPER_FID,
        "paper_user_study": PAPER_USER_STUDY,
        "results": results,
        "intra_lpips_table": summarize_matrix(results, "intra_lpips"),
        "fid_table": summarize_matrix(results, "fid"),
    }
    out_path = args.report or os.path.join(args.out_root, "baseline_report.json")
    if not args.dry_run or args.report:
        save_report(summary, out_path)
        print(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
