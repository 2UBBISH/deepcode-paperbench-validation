"""Shared CLI helpers (argparse defaults + sys.path bootstrap)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def add_common_args(parser: argparse.ArgumentParser, default_model: str = "gpt2-medium") -> None:
    parser.add_argument("--model", default=default_model, help="HF model id of the base LM")
    parser.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--cache-dir", default=None, help="HF datasets cache dir")


def add_artifacts_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--probe-path", default="artifacts/probe/toxic_probe.pt")
    parser.add_argument("--vectors-path", default="artifacts/toxic_vectors/toxic_vectors.pt")


def load_model_and_tokenizer(args, path: str | None = None, dtype: str = "float32"):
    from dpo_toxic.utils import load_lm

    return load_lm(path or args.model, dtype=dtype, device=args.device)


def load_probe_from_args(args, device=None):
    from dpo_toxic.probe import load_probe

    return load_probe(args.probe_path, device=device or args.device)
