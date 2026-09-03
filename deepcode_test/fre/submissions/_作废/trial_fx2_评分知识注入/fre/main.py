"""Central command-line entry point for the Functional Reward Encodings (FRE) project.

This module intentionally stays lightweight: heavy dependencies (PyTorch, D4RL,
MuJoCo, etc.) are imported lazily inside each command dispatcher.  It accepts a
subcommand and forwards all remaining arguments to the corresponding pipeline
entry point implemented elsewhere in the repository.

Supported commands
------------------
    pretrain      Pretrain the FRE variational autoencoder on random reward priors.
    train         Freeze the pretrained FRE encoder and train FRE-conditioned IQL.
    eval          Evaluate a trained FRE agent on zero-shot downstream tasks.
    baselines     Train/evaluate comparison baselines (GC-IQL, GC-BC, OPAL, FB, SF).
    visualize     Generate reward/value/policy visualizations.

Examples
--------
    python -m fre.main pretrain --config antmaze --seed 0
    python -m fre.main train    --config antmaze --seed 0 --model-path checkpoints/fre.pt
    python -m fre.main eval     --config antmaze --seed 0 --model-path checkpoints/fre.pt
                                --agent-path checkpoints/iql.pt
    python -m fre.main baselines --config antmaze --baseline gc_iql --seed 0
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

__all__ = ["build_parser", "main"]

LOGGER = logging.getLogger("fre.main")


def _import_main(module_path: str):
    """Import and return the ``main`` function from a pipeline module."""
    import importlib

    module = importlib.import_module(module_path)
    main_fn = getattr(module, "main", None)
    if main_fn is None:
        raise AttributeError(f"{module_path} does not expose a 'main' function")
    return main_fn


def _dispatch_pipeline(command: str, argv: Optional[List[str]] = None) -> int:
    """Route a command to its matching pipeline entry point.

    Parameters
    ----------
    command:
        One of ``pretrain``, ``train``, ``eval``, ``baselines``, or
        ``visualize``.
    argv:
        Remaining command-line arguments forwarded to the pipeline main.

    Returns
    -------
    Exit code returned by the pipeline entry point (``0`` on success).
    """
    pipeline_modules = {
        "pretrain": "fre.pipeline.pretrain_encoder",
        "train": "fre.pipeline.train_agent",
        "eval": "fre.pipeline.evaluate",
        "baselines": "fre.pipeline.evaluate_baselines",
        "visualize": "fre.pipeline.visualize",
    }

    if command not in pipeline_modules:
        raise ValueError(f"Unknown command: {command}")

    module_path = pipeline_modules[command]
    try:
        main_fn = _import_main(module_path)
    except ImportError as exc:
        if command == "visualize":
            LOGGER.error(
                "The visualize pipeline is not available yet or is missing an "
                "optional dependency: %s",
                exc,
            )
            return 1
        LOGGER.error("Failed to import %s: %s", module_path, exc)
        return 1
    except Exception as exc:  # noqa: BLE001 - report a friendly error
        LOGGER.error("Could not load entry point for '%s': %s", command, exc)
        return 1

    # Pipeline entry points accept ``argv=None`` to use sys.argv, but we pass
    # the forwarded list explicitly so that programmatic callers work too.
    try:
        result = main_fn(argv if argv is not None else [])
    except SystemExit as exc:  # argparse in submodules may call sys.exit
        return int(exc.code) if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted by user")
        return 130
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Pipeline '%s' failed", command)
        return 1

    # Most pipeline mains return None; treat that as success.
    return result if isinstance(result, int) else 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level command dispatcher parser."""
    parser = argparse.ArgumentParser(
        prog="fre",
        description=(
            "Functional Reward Encodings for Zero-Shot Offline Reinforcement Learning "
            "— run pretraining, FRE-conditioned IQL training, evaluation, baselines, "
            "or visualization."
        ),
    )
    parser.add_argument(
        "command",
        choices=["pretrain", "train", "eval", "baselines", "visualize"],
        help="Pipeline command to execute.",
    )
    parser.add_argument(
        "cmd_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the selected pipeline entry point.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
        help="Show version and exit.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Execute the FRE command-line dispatcher.

    Parameters
    ----------
    argv:
        Optional argument list. When ``None``, ``sys.argv[1:]`` is used.

    Returns
    -------
    Integer exit code.
    """
    parser = build_parser()

    # argparse.REMAINDER can swallow '--help' in some configurations; handle the
    # explicit help cases before full parsing to ensure a helpful message.
    if argv is None:
        argv = sys.argv[1:]

    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return 0 if not argv else 0

    if argv[0] in {"-v", "--version"}:
        parser.print_version()
        return 0

    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else (0 if exc.code is None else 1)

    return _dispatch_pipeline(args.command, args.cmd_args)


if __name__ == "__main__":
    raise SystemExit(main())
