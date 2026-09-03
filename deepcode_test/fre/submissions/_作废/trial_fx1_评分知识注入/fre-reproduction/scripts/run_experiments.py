#!/usr/bin/env python
"""Full benchmark sweep orchestrator for the FRE reproduction codebase.

This script runs the complete strided training and evaluation pipeline for
Functional Reward Encodings (FRE) and the five comparison baselines across the
three benchmark domains used in the paper:

* AntMaze-large-diverse-v2
* ExORL walker/cheetah exploratory datasets
* D4RL Franka Kitchen

The default launch mode uses ``subprocess`` so that every phase is executed
through its normal command-line entry point, which keeps checkpoint formats
and logging identical to a manual run.  Alternatively, ``--dry-run`` prints
the exact commands without executing them.

Examples
--------
    # FRE only, AntMaze, 5 seeds
    python scripts/run_experiments.py --domains antmaze --methods fre

    # FRE + all baselines, all domains, short smoke run
    python scripts/run_experiments.py --domains antmaze exorl kitchen \\
        --methods fre fb sf gc_iql gc_bc opal --seeds 0 1 2 3 4

    # Single seed, no baselines, custom output root
    python scripts/run_experiments.py --domains kitchen --methods fre \\
        --seeds 0 --output_root experiments
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
FRE_METHODS = ("fre",)
BASELINE_METHODS = ("fb", "sf", "gc_iql", "gc_bc", "opal")
ALL_METHODS = FRE_METHODS + BASELINE_METHODS

DOMAIN_DATASETS = {
    "antmaze": "antmaze-large-diverse-v2",
    "kitchen": "kitchen-complete-v0",
    "walker": "walker",
    "cheetah": "cheetah",
    "exorl": "walker",
}

# The four ExORL tasks are handled by passing ``--domain walker`` or
# ``--domain cheetah`` to the evaluation script.  For the sweep we train two
# separate agents, one per DMC domain, as in the paper.
DOMAIN_ALIASES = {
    "antmaze": "antmaze",
    "kitchen": "kitchen",
    "walker": "walker",
    "cheetah": "cheetah",
    "exorl-walker": "walker",
    "exorl-cheetah": "cheetah",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the FRE benchmark reproduction sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["antmaze", "walker", "cheetah", "kitchen"],
        help="Domains to run. ExORL is split into walker/cheetah.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(ALL_METHODS),
        help="Methods to run: fre, fb, sf, gc_iql, gc_bc, opal.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Random training seeds.",
    )
    parser.add_argument("--output_root", type=str, default="experiments")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--vae_steps", type=int, default=None)
    parser.add_argument("--rl_steps", type=int, default=None)
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--num_examples", type=int, default=32)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--baseline_steps", type=int, default=None)
    parser.add_argument("--opal_num_skills", type=int, default=10)
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip phases whose final checkpoint/result file already exists.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands instead of executing them.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print stdout/stderr live instead of capturing it.",
    )
    return parser.parse_args(argv)


def resolve_domain(domain: str) -> str:
    return DOMAIN_ALIASES.get(domain.lower(), domain.lower())


def method_dir(root: Path, domain: str, method: str, seed: int) -> Path:
    return root / method / domain / f"seed{seed}"


def checkpoint_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def run_cmd(
    cmd: List[str],
    dry_run: bool = False,
    verbose: bool = False,
    env: Optional[Dict[str, str]] = None,
) -> int:
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"\n[run_experiments] $ {cmd_str}", flush=True)
    if dry_run:
        return 0

    full_env = os.environ.copy()
    full_env.setdefault("PYTHONPATH", str(ROOT))
    if env:
        full_env.update(env)

    if verbose:
        return subprocess.call(cmd, cwd=str(ROOT), env=full_env)

    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=full_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.stdout:
        print(proc.stdout[-4000:], flush=True)
    return proc.returncode


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def base_flags(domain: str, seed: int, out_dir: Path, args: argparse.Namespace) -> List[str]:
    dataset_name = args.dataset_path or DOMAIN_DATASETS.get(domain, domain)
    flags = [
        "--domain",
        domain,
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--output_dir",
        str(out_dir),
    ]
    if domain not in ("synthetic",):
        flags += ["--dataset_name", dataset_name]
    if args.dataset_path:
        flags += ["--dataset_path", str(args.dataset_path)]
    return flags


def run_fre_domain(
    domain: str,
    seed: int,
    root: Path,
    args: argparse.Namespace,
) -> int:
    domain = resolve_domain(domain)
    out = method_dir(root, domain, "fre", seed)
    encoder_out = out / "encoder"
    rl_out = out / "rl"
    eval_out = out / "eval"
    ensure_dir(encoder_out)
    ensure_dir(rl_out)
    ensure_dir(eval_out)

    encoder_final = encoder_out / "encoder_final.pt"
    agent_final = rl_out / "agent_final.pt"
    results_file = eval_out / "results.json"

    # Phase 1: FRE encoder/decoder VAE training.
    encoder_cmd = [
        sys.executable,
        "-u",
        str(SCRIPTS / "train_fre_encoder.py"),
        *base_flags(domain, seed, encoder_out, args),
    ]
    if args.vae_steps is not None:
        encoder_cmd += ["--steps", str(args.vae_steps)]
    if args.skip_existing and checkpoint_exists(encoder_final):
        print(f"[skip] FRE encoder already exists: {encoder_final}")
    else:
        code = run_cmd(encoder_cmd, args.dry_run, args.verbose)
        if code != 0:
            return code

    # Phase 2: FRE-conditioned IQL training with the frozen encoder.
    rl_cmd = [
        sys.executable,
        "-u",
        str(SCRIPTS / "train_rl.py"),
        *base_flags(domain, seed, rl_out, args),
        "--vae_checkpoint",
        str(encoder_final),
    ]
    if args.rl_steps is not None:
        rl_cmd += ["--steps", str(args.rl_steps)]
    if args.skip_existing and checkpoint_exists(agent_final):
        print(f"[skip] FRE RL agent already exists: {agent_final}")
    else:
        code = run_cmd(rl_cmd, args.dry_run, args.verbose)
        if code != 0:
            return code

    # Phase 3: zero-shot evaluation with 32 reward examples.
    eval_cmd = [
        sys.executable,
        "-u",
        str(SCRIPTS / "eval_zero_shot.py"),
        "--domain",
        domain,
        "--checkpoint",
        str(agent_final),
        "--seed",
        str(seed),
        "--output_dir",
        str(eval_out),
        "--device",
        args.device,
        "--num_examples",
        str(args.num_examples),
        "--num_episodes",
        str(args.eval_episodes),
    ]
    if args.skip_existing and results_file.exists():
        print(f"[skip] FRE eval results already exist: {results_file}")
    else:
        code = run_cmd(eval_cmd, args.dry_run, args.verbose)
        if code != 0:
            return code
    return 0


def run_baseline_domain(
    domain: str,
    baseline: str,
    seed: int,
    root: Path,
    args: argparse.Namespace,
) -> int:
    domain = resolve_domain(domain)
    out = method_dir(root, domain, baseline, seed)
    train_out = out / "train"
    eval_out = out / "eval"
    ensure_dir(train_out)
    ensure_dir(eval_out)

    checkpoint = train_out / f"{baseline}_{domain}_seed{seed}_final.pt"
    results_file = eval_out / "results.json"

    train_cmd = [
        sys.executable,
        "-u",
        str(SCRIPTS / "train_baselines.py"),
        "--baseline",
        baseline,
        *base_flags(domain, seed, train_out, args),
    ]
    if args.baseline_steps is not None:
        train_cmd += ["--steps", str(args.baseline_steps)]
    if args.skip_existing and checkpoint_exists(checkpoint):
        print(f"[skip] {baseline} checkpoint already exists: {checkpoint}")
    else:
        code = run_cmd(train_cmd, args.dry_run, args.verbose)
        if code != 0:
            return code

    eval_cmd = [
        sys.executable,
        "-u",
        str(SCRIPTS / "eval_baselines.py"),
        "--baseline",
        baseline,
        "--domain",
        domain,
        "--checkpoint",
        str(checkpoint),
        "--seed",
        str(seed),
        "--output_dir",
        str(eval_out),
        "--device",
        args.device,
        "--num_episodes",
        str(args.eval_episodes),
    ]
    if baseline == "opal":
        eval_cmd += ["--num_skills", str(args.opal_num_skills)]
    if args.skip_existing and results_file.exists():
        print(f"[skip] {baseline} eval results already exist: {results_file}")
    else:
        code = run_cmd(eval_cmd, args.dry_run, args.verbose)
        if code != 0:
            return code
    return 0


def run_sweep(args: argparse.Namespace) -> int:
    methods = [m.lower() for m in args.methods]
    unknown = [m for m in methods if m not in ALL_METHODS]
    if unknown:
        print(f"Unknown methods: {unknown}")
        return 2

    fre_requested = "fre" in methods
    baseline_methods = [m for m in methods if m in BASELINE_METHODS]

    root = Path(args.output_root)
    ensure_dir(root)

    summary: Dict[str, List[Dict[str, object]]] = {"domains": [], "results": []}

    domains: List[str] = []
    for raw in args.domains:
        d = resolve_domain(raw)
        if d not in domains:
            domains.append(d)

    summary["domains"] = domains

    for domain in domains:
        for seed in args.seeds:
            if fre_requested:
                print(f"\n=== FRE | domain={domain} | seed={seed} ===", flush=True)
                code = run_fre_domain(domain, seed, root, args)
                if code != 0:
                    print(f"FRE failed for domain={domain}, seed={seed} (exit {code})")
                    if not args.dry_run:
                        return code
                summary["results"].append(
                    {
                        "method": "fre",
                        "domain": domain,
                        "seed": seed,
                        "status": "ok" if code == 0 else f"failed:{code}",
                    }
                )

            for baseline in baseline_methods:
                print(
                    f"\n=== {baseline} | domain={domain} | seed={seed} ===",
                    flush=True,
                )
                code = run_baseline_domain(domain, baseline, seed, root, args)
                if code != 0:
                    print(
                        f"{baseline} failed for domain={domain}, seed={seed} "
                        f"(exit {code})"
                    )
                    if not args.dry_run:
                        return code
                summary["results"].append(
                    {
                        "method": baseline,
                        "domain": domain,
                        "seed": seed,
                        "status": "ok" if code == 0 else f"failed:{code}",
                    }
                )

    summary_path = root / "sweep_summary.json"
    if not args.dry_run:
        ensure_dir(root)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print(f"\nSweep summary written to {summary_path}")
    else:
        print(f"\nDry run complete. Would write summary to {summary_path}")

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    return run_sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())
