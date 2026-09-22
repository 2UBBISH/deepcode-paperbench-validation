#!/usr/bin/env python
"""Build the dataset artifacts required by the forgetting-forecasting pipeline.

This script materialises the artifacts described in §2 and §4.1 of
"What Will My Model Forget? Forecasting Forgotten Examples in Language Model
Refinement":

* ``d_pt.jsonl``      -- the 36 P3 *train* tasks, 100 examples/task (3600 total).
* ``d_pt_hat.jsonl``  -- the subset of ``D_PT`` that ``f0`` answers correctly.
* ``d_r.jsonl``       -- the mispredicted examples forming ``D_R``
                          (BART0: P3 test tasks, FLAN-T5: MMLU validation).
* ``d_r_train.jsonl`` / ``d_r_test.jsonl`` -- the 60/40 split of ``D_R``.
* ``id.jsonl`` / ``ood.jsonl``             -- the BART0 ID/OOD partition.
* ``manifest.json``   -- bookkeeping (sizes, filtering flags, EM stats).

Example
-------
    python scripts/build_datasets.py --model-key BART0_L
    python scripts/build_datasets.py --model-key FLAN-T5_L --output-dir artifacts/FLAN-T5_L

When no base LM is available (e.g. a CPU smoke test), pass ``--passthrough``
so that the D_PT_hat / D_R filtering steps are skipped and the resulting
manifests record ``filtered: false``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Make ``src`` importable when the script is executed directly.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data import dataset_builders as db  # noqa: E402

logger = logging.getLogger("build_datasets")

DEFAULT_CONFIG_PATH = os.path.join(_ROOT, "config", "config.yaml")
DEFAULT_TASKS_PATH = os.path.join(_ROOT, "config", "tasks.yaml")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load ``config/config.yaml`` (gracefully returning ``{}`` if absent)."""
    path = path or DEFAULT_CONFIG_PATH
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
        logger.info("Loaded config from %s", path)
        return cfg
    except FileNotFoundError:
        logger.warning("Config file %s not found; falling back to defaults.", path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not parse %s (%s); falling back to defaults.", path, exc)
    return {}


def cfg_get(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Nested lookup: ``cfg_get(cfg, "data", "examples_per_task", default=100)``."""
    node: Any = cfg
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


# ---------------------------------------------------------------------------
# Optional predictor (base LM) construction
# ---------------------------------------------------------------------------
def build_predictor(model_key: str, cfg: Dict[str, Any]):
    """Try to construct a base-LM predictor for filtering.

    The predictor contract expected by ``src.data.dataset_builders`` is a
    callable ``predict(examples) -> List[str]`` (or an object exposing
    ``predict`` / ``predict_batch`` / ``generate``).  If the modelling stack
    (``src.modeling.base_lm``) or its weights are unavailable we return
    ``None`` so that the caller can either pass ``--passthrough`` or raise.
    """
    try:
        from src.modeling.base_lm import load_base_lm  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional deps
        logger.warning("Base LM stack unavailable (%s).", exc)
        return None

    try:
        model = load_base_lm(
            model_key,
            device=cfg_get(cfg, "device", default="cuda"),
            dtype=cfg_get(cfg, "dtype", default="float32"),
            cache_dir=cfg_get(cfg, "cache_dir", default=None),
            max_input_len=cfg_get(cfg, "data", "max_input_len", default=512),
            max_output_len=cfg_get(cfg, "data", "max_output_len", default=64),
        )
        logger.info("Loaded predictor for %s", model_key)
        return model
    except Exception as exc:  # pragma: no cover - weights may not be present
        logger.warning("Could not load base LM '%s' (%s).", model_key, exc)
        return None


# ---------------------------------------------------------------------------
# Summary / reporting
# ---------------------------------------------------------------------------
def summarize(result: Dict[str, Any]) -> None:
    """Log a compact, human-readable summary of the built artifacts."""
    manifest = result.get("manifest", result)
    logger.info("=" * 68)
    logger.info("Dataset build summary")
    logger.info("=" * 68)
    for key in (
        "model_key",
        "n_d_pt",
        "n_d_pt_hat",
        "n_d_r",
        "n_d_r_train",
        "n_d_r_test",
        "n_id",
        "n_ood",
        "base_em_percent",
        "filtered",
        "output_dir",
    ):
        if key in manifest:
            logger.info("  %-18s %s", key, manifest[key])

    n_pt_hat = manifest.get("n_d_pt_hat")
    n_d_r = manifest.get("n_d_r")
    if n_pt_hat and n_d_r:
        logger.info("  #pairs available    %d", n_pt_hat * n_d_r)
        logger.info(
            "  positive prevalence  ~%.2f%% (must land in the 1%%-10%% range)",
            100.0 / max(n_pt_hat, 1),
        )
    logger.info("=" * 68)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model-key",
        default="BART0_L",
        choices=["BART0_L", "FLAN-T5_L", "FLAN-T5_3B", "FLAN-T5_small"],
        help="Which base LM defines D_PT_hat / D_R (default: BART0_L).",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to config.yaml.")
    parser.add_argument("--tasks-yaml", default=DEFAULT_TASKS_PATH, help="Path to tasks.yaml.")
    parser.add_argument("--output-dir", default=None, help="Where artifacts are written.")
    parser.add_argument("--examples-per-task", type=int, default=None, help="100 by default.")
    parser.add_argument("--r-train-ratio", type=float, default=None, help="0.6 by default.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (42 by default).")
    parser.add_argument("--cache-dir", default=None, help="HuggingFace cache directory.")
    parser.add_argument("--json-dir", default=None, help="Local ReCross JSON directory for BART0 tasks.")
    parser.add_argument("--mmlu-data-dir", default=None, help="Local MMLU (Berkeley release) directory.")
    parser.add_argument("--batch-size", type=int, default=8, help="Predictor batch size.")
    parser.add_argument(
        "--passthrough",
        action="store_true",
        help="Skip EM filtering (no base LM); manifests record filtered=false.",
    )
    parser.add_argument("--quiet", action="store_true", help="Only log warnings/errors.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    cfg = load_config(args.config)

    output_dir = args.output_dir or os.path.join(
        cfg_get(cfg, "output_dir", default="artifacts"),
        args.model_key,
    )
    examples_per_task = (
        args.examples_per_task
        if args.examples_per_task is not None
        else cfg_get(cfg, "data", "examples_per_task", default=100)
    )
    r_train_ratio = (
        args.r_train_ratio
        if args.r_train_ratio is not None
        else cfg_get(cfg, "data", "r_train_ratio", default=0.6)
    )
    seed = args.seed if args.seed is not None else cfg_get(cfg, "seed", default=42)
    cache_dir = args.cache_dir or cfg_get(cfg, "cache_dir", default=None)
    json_dir = args.json_dir or cfg_get(cfg, "data", "recal_json_dir", default=None)
    mmlu_data_dir = args.mmlu_data_dir or cfg_get(cfg, "data", "mmlu_data_dir", default=None)

    predictor = None
    if not args.passthrough:
        predictor = build_predictor(args.model_key, cfg)
        if predictor is None:
            logger.error(
                "No predictor could be constructed for %s. Re-run with --passthrough "
                "for a smoke test, or install the modelling stack/weights.",
                args.model_key,
            )
            return 2

    result = db.build_all(
        out_dir=output_dir,
        model_key=args.model_key,
        predictor=predictor,
        r_predictor=None,  # the same f0 is used for D_PT_hat and D_R filtering
        examples_per_task=examples_per_task,
        r_train_ratio=r_train_ratio,
        seed=seed,
        tasks_yaml=args.tasks_yaml,
        cache_dir=cache_dir,
        json_dir=json_dir,
        mmlu_data_dir=mmlu_data_dir,
        allow_passthrough=bool(args.passthrough),
    )

    summarize(result)
    manifest_path = os.path.join(output_dir, "manifest.json")
    if os.path.exists(manifest_path):
        logger.info("Manifest written to %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
