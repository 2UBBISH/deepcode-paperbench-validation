"""Shared CLI helpers for the experiment scripts."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bbox_adapter.config import RunConfig, load_run_config  # noqa: E402


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, default=None,
                        help="YAML/JSON run configuration (see configs/).")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["strategyqa", "gsm8k", "truthfulqa", "scienceqa"])
    parser.add_argument("--adapter-size", type=str, default=None, choices=["0.1B", "0.3B"])
    parser.add_argument("--adapter-model", type=str, default=None,
                        help="Override the adapter backbone, e.g. microsoft/deberta-v3-large.")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory with local dataset copies (JSON/JSONL).")
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--mock-llm", action="store_true",
                        help="Use the deterministic offline LLM (smoke tests).")
    parser.add_argument("--llm-provider", type=str, default=None,
                        choices=["azure", "openai", "huggingface", "mock"])
    parser.add_argument("--llm-model", type=str, default=None,
                        help="e.g. gpt-3.5-turbo, davinci-002, mistralai/Mixtral-8x7B-v0.1")
    parser.add_argument("--seed", type=int, default=None)
    return parser


def build_config(args) -> RunConfig:
    config = load_run_config(args.config) if args.config else RunConfig()
    if args.dataset:
        config.dataset = args.dataset
    if args.seed is not None:
        config.seed = args.seed
        config.adapter.seed = args.seed
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.adapter_size:
        config.adapter.adapter_size = args.adapter_size
    if args.adapter_model:
        config.adapter.model_name = args.adapter_model
    if args.llm_model:
        config.llm.name = args.llm_model
    if args.llm_provider:
        config.llm.provider = args.llm_provider
    if args.mock_llm:
        config.llm.provider = "mock"
        config.llm.name = "mock-llm"
    return config


def write_json(path: str, payload) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return path
