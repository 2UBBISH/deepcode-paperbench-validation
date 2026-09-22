#!/usr/bin/env python
"""Generate the pairwise toxic dataset for DPO training (Section 4.2).

Reproduces the PPLM-based preference-pair construction described in the paper:

    "To generate pairwise preference data, we use sentences from Wikitext-2 as
     prompts. For each prompt, we generate a positive sample using greedy sampling
     with GPT2, while using PPLM to generate negative (toxic) samples. We use our
     toxic probe W_Toxic as our attribute classifier to guide towards toxic
     outputs. We create 24,576 pairs of toxic and nontoxic continuations."

The preferred (``chosen``) continuation is the greedy GPT2 continuation and the
non-preferred (``rejected``) continuation is the PPLM toxic continuation, with

    p(y | a) ∝ p(y) p(a | y)                                        (Eq. 1)

PPLM hyperparameters come from Appendix E, Table 9:

    STEP SIZE 0.4 | TEMPERATURE 1 | TOP K 10 | NUM ITERATIONS 50 |
    WINDOW LENGTH 0 | HORIZON LENGTH 1 | DECAY FALSE | GAMMA 1 |
    GM SCALE 0.95 | KL SCALE 0.1

Generation is sharded and resumable so the (expensive) 24,576-pair run can be
interrupted and restarted deterministically.

Usage
-----
    python scripts/generate_pairs.py --n-pairs 24576
    python scripts/generate_pairs.py --quick            # smoke test
    python scripts/generate_pairs.py --resume           # continue a shard run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Make the repository root importable when executed as a plain script.
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

DEFAULT_MODEL = "openai-community/gpt2-medium"
DEFAULT_CONFIG = os.path.join("configs", "pplm.yaml")
DEFAULT_PROBE_PATH = os.path.join("artifacts", "probe", "w_toxic.pt")
DEFAULT_SHARD_DIR = os.path.join("artifacts", "data", "pairs_shards")
DEFAULT_PAIRS_PATH = os.path.join("artifacts", "data", "pairs.jsonl")
DEFAULT_OUT_DIR = os.path.join("artifacts", "data")

N_PAIRS = 24_576

# Appendix E, Table 9 defaults (kept here so `--help`/dry-run works offline).
TABLE9_DEFAULTS: Dict[str, Any] = {
    "step_size": 0.4,
    "temperature": 1.0,
    "top_k": 10,
    "num_iterations": 50,
    "window_length": 0,
    "horizon_length": 1,
    "decay": False,
    "gamma": 1.0,
    "gm_scale": 0.95,
    "kl_scale": 0.1,
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Best-effort YAML config loader; returns ``{}`` when unavailable."""
    if not path:
        return {}
    if not os.path.isfile(path):
        return {}
    try:
        import yaml  # noqa: WPS433 (lazy import)

        with open(path, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def _dig(cfg: Dict[str, Any], *keys, default=None):
    """Nested dict lookup tolerant of missing intermediate keys."""
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur if cur is not None else default


def resolve_settings(args: argparse.Namespace) -> Dict[str, Any]:
    """Merge CLI args, ``configs/pplm.yaml`` and Table 9 defaults."""
    cfg = load_config(getattr(args, "config", None))

    def pick(cli_name: str, *cfg_keys, default=None):
        value = getattr(args, cli_name, None)
        if value is not None:
            return value
        for key in cfg_keys:
            found = _dig(cfg, *key.split("."), default=None)
            if found is not None:
                return found
        return default

    n_pairs = pick("n_pairs", "n_pairs", "dataset.n_pairs", default=N_PAIRS)
    max_new_tokens = pick(
        "max_new_tokens", "max_new_tokens", "generation.max_new_tokens", default=20
    )

    settings: Dict[str, Any] = {
        "model_name": pick("model", "model_name", "model.name", default=DEFAULT_MODEL),
        "probe_path": pick("probe_path", "probe_path", default=DEFAULT_PROBE_PATH),
        "shard_dir": pick("shard_dir", "shard_dir", default=DEFAULT_SHARD_DIR),
        "pairs_path": pick("pairs_path", "pairs_path", default=DEFAULT_PAIRS_PATH),
        "n_pairs": int(n_pairs),
        "max_new_tokens": int(max_new_tokens),
        "shard_size": int(pick("shard_size", "shard_size", default=512)),
        "max_prompt_length": int(
            pick("max_prompt_length", "max_prompt_length", default=96)
        ),
        "batch_prompts": int(pick("batch_prompts", "batch_prompts", default=1)),
        "seed": int(pick("seed", "seed", default=0)),
        "device": pick("device", "device", default=None),
        "resume": bool(getattr(args, "resume", False)),
        "force": bool(getattr(args, "force", False)),
        "quick": bool(getattr(args, "quick", False)),
        "verbose": not bool(getattr(args, "quiet", False)),
        "write_json": not bool(getattr(args, "no_json", False)),
        "split": pick("split", "split", "data.split", default="train"),
        "min_chars": int(pick("min_chars", "min_chars", default=40)),
        "min_words": int(pick("min_words", "min_words", default=8)),
    }

    # PPLM hyperparameters (CLI > yaml > Table 9).
    pplm: Dict[str, Any] = {}
    for key, default in TABLE9_DEFAULTS.items():
        pplm[key] = pick(key, f"pplm.{key}", key, default=default)
    # `num_iterations` CLI override, Table 9 says 50.
    settings["pplm"] = pplm
    return settings


# ---------------------------------------------------------------------------
# Model / prompt helpers
# ---------------------------------------------------------------------------
def load_model_safe(name_or_path: str, device: Optional[str] = None):
    """Load GPT2 (or a local checkpoint) through ``src.model_utils``."""
    from src.model_utils import load_model

    return load_model(name_or_path, device=device)


def load_prompts(settings: Dict[str, Any]) -> List[str]:
    """Draw Wikitext-2 prompt sentences for pair construction."""
    from data.wikitext import prompt_pool

    prompts = prompt_pool(
        n=settings["n_pairs"],
        split=settings["split"],
        seed=settings["seed"],
        min_chars=settings["min_chars"],
        min_words=settings["min_words"],
    )
    if not prompts:
        raise RuntimeError("Wikitext-2 prompt pool is empty; check the dataset cache.")
    return list(prompts)


def build_generator(settings: Dict[str, Any], model, tokenizer):
    """Instantiate a configured :class:`~src.pplm_generate.PPLMGenerator`."""
    from src.pplm_generate import PPLMConfig, PPLMGenerator

    config = PPLMConfig.from_dict(settings["pplm"])
    config.max_new_tokens = settings["max_new_tokens"]
    config.max_prompt_length = settings["max_prompt_length"]
    config.device = settings["device"]
    return PPLMGenerator(
        model,
        tokenizer,
        config=config,
        device=settings["device"],
        probe_path=settings["probe_path"],
    )


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------
def run_generation(args: argparse.Namespace) -> int:
    """Generate (or resume) the pairwise toxic dataset."""
    settings = resolve_settings(args)
    started = time.time()

    if settings["quick"]:
        settings["n_pairs"] = min(settings["n_pairs"], 8)
        settings["max_new_tokens"] = min(settings["max_new_tokens"], 8)
        settings["shard_size"] = min(settings["shard_size"], 8)
        settings["pplm"]["num_iterations"] = min(
            int(settings["pplm"]["num_iterations"]), 3
        )

    if settings["verbose"]:
        print("=" * 72)
        print("PPLM pairwise toxic dataset generation (Section 4.2)")
        print("=" * 72)
        for key in (
            "model_name",
            "probe_path",
            "n_pairs",
            "max_new_tokens",
            "shard_dir",
            "pairs_path",
            "seed",
        ):
            print(f"  {key:<16}: {settings[key]}")
        print(f"  pplm            : {settings['pplm']}")

    if args.dry_run:
        print("\n[dry-run] configuration resolved; nothing generated.")
        return 0

    from data.pairwise import count_existing_pairs

    done = count_existing_pairs(settings["shard_dir"])
    if done and not settings["resume"] and not settings["force"]:
        if done >= settings["n_pairs"]:
            print(
                f"\n[info] {done} pairs already exist in {settings['shard_dir']} "
                "(>= requested). Use --force to regenerate."
            )
            return 0
    if settings["force"]:
        done = 0
    if settings["verbose"] and done:
        print(f"\n[resume] {done} pairs already generated; continuing from there.")

    prompts = load_prompts(settings)
    if settings["verbose"]:
        print(f"[data] {len(prompts)} Wikitext-2 prompts available for pair generation.")

    remaining = max(0, settings["n_pairs"] - done)
    if remaining == 0:
        print(f"[done] nothing to do; {done} pairs already present.")
        return 0
    # Deterministic resume: skip the prompts already consumed.
    prompts = prompts[done : done + remaining]
    if settings["verbose"]:
        print(f"[plan] generating {len(prompts)} new pairs (target {settings['n_pairs']}).")

    model, tokenizer = load_model_safe(settings["model_name"], device=settings["device"])
    generator = build_generator(settings, model, tokenizer)
    if settings["verbose"]:
        print("[pplm] generator ready (attribute classifier = W_Toxic).")

    from src.pplm_generate import generate_and_save

    summary = generate_and_save(
        prompts,
        model,
        tokenizer,
        generator=generator,
        shard_dir=settings["shard_dir"],
        out_path=settings["pairs_path"],
        shard_size=settings["shard_size"],
        max_new_tokens=settings["max_new_tokens"],
        seed=settings["seed"] + done,
        num_iterations=settings["pplm"]["num_iterations"],
        device=settings["device"],
        verbose=settings["verbose"],
    )

    summary["elapsed_sec"] = round(time.time() - started, 2)
    summary["requested_pairs"] = settings["n_pairs"]
    summary["resumed_from"] = done
    summary["table9"] = dict(settings["pplm"])

    if settings["verbose"]:
        print("\n" + "-" * 72)
        print(
            f"[done] {summary.get('n_pairs', '?')} pairs "
            f"({summary.get('n_shards', '?')} shards) in {summary['elapsed_sec']}s"
        )
        print(f"       shards : {summary.get('shard_dir')}")
        print(f"       merged : {summary.get('pairs_path')}")

    if settings["write_json"]:
        out = os.path.join(DEFAULT_OUT_DIR, "generate_pairs_summary.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(_json_safe(summary), handle, indent=2)
        if settings["verbose"]:
            print(f"       summary: {out}")

    return 0


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------
def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy/torch scalars and paths to JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    for attr in ("item", "tolist"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return _json_safe(fn())
            except Exception:
                pass
    return str(obj)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the PPLM toxic / greedy non-toxic preference pairs used for "
            "DPO training (Section 4.2; PPLM hyperparameters in Appendix E Table 9)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config path.")
    parser.add_argument("--model", default=None, help="Model name or local path.")
    parser.add_argument(
        "--probe-path", default=None, help="Trained W_Toxic artifact (.pt)."
    )
    parser.add_argument("--n-pairs", type=int, default=None, help="Number of pairs.")
    parser.add_argument(
        "--shard-dir", default=None, help="Directory for resumable pair shards."
    )
    parser.add_argument("--pairs-path", default=None, help="Merged JSONL output path.")
    parser.add_argument("--shard-size", type=int, default=None, help="Pairs per shard.")
    parser.add_argument(
        "--max-new-tokens", type=int, default=None, help="Continuation length."
    )
    parser.add_argument("--max-prompt-length", type=int, default=None)
    parser.add_argument("--split", default=None, help="Wikitext-2 split for prompts.")
    parser.add_argument("--min-chars", type=int, default=None)
    parser.add_argument("--min-words", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="'cuda', 'cpu', or None=auto.")

    # PPLM / Table 9 overrides.
    parser.add_argument("--step-size", type=float, default=None, dest="step_size")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None, dest="top_k")
    parser.add_argument("--num-iterations", type=int, default=None, dest="num_iterations")
    parser.add_argument("--window-length", type=int, default=None, dest="window_length")
    parser.add_argument(
        "--horizon-length", type=int, default=None, dest="horizon_length"
    )
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--gm-scale", type=float, default=None, dest="gm_scale")
    parser.add_argument("--kl-scale", type=float, default=None, dest="kl_scale")

    parser.add_argument(
        "--resume", action="store_true", help="Continue an interrupted shard run."
    )
    parser.add_argument(
        "--force", action="store_true", help="Ignore existing shards and regenerate."
    )
    parser.add_argument("--quick", action="store_true", help="Tiny smoke-test run.")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument("--no-json", action="store_true", dest="no_json")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_generation(args)
    except KeyboardInterrupt:
        print("\n[interrupted] shards are saved; rerun with --resume.", file=sys.stderr)
        return 130
    except Exception as exc:  # pragma: no cover - surfaced for debugging
        print(f"[error] {exc}", file=sys.stderr)
        if not getattr(args, "quiet", False):
            traceback.print_exc()
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
