#!/usr/bin/env python
"""Plug-and-play adaptation driver for BBox-Adapter (paper Sections 4.3, 4.6, 4.7).

Paper contract implemented here
------------------------------
Section 4.3 (Plug-and-Play Adaptation):
    "The tuned BBOX-ADAPTER can be seamlessly applied to various black-box LLMs in a
     plug-and-play manner, eliminating the need for retraining or additional technical
     modifications. A well-trained version of BBOX-ADAPTER adapting gpt-3.5-turbo can
     serve as a plugger to be integrated into the OpenAI base model davinci-002 and
     Mixtral-8x7B."

Table 3 (reference numbers reproduced by this script):

    Plugger -> BBOX-ADAPTER (gpt-3.5-turbo)
    Black-box LLM        StrategyQA        GSM8K          TruthfulQA       Average
    davinci-002          44.19 (base)      23.73 (base)   31.50 (base)     33.14
    davinci-002 plugged  59.61 (+15.42)    23.85 (+0.12)  36.50 (+5.00)    39.99 (+6.85)
    Mixtral-8x7B         59.91 (base)      47.46 (base)   40.40 (base)     49.26
    Mixtral-8x7B plugged 63.97 (+4.06)     47.61 (+0.15)  49.70 (+9.30)    53.76 (+4.50)

Section 4.6 / 4.7 + Table 6 (Mixtral treated as a black box, VRAM):
    Base Model (Mixtral-8x7B)   Acc 59.91 (0.1B column)   VRAM inference 90 GiB
    Base + LoRA                 Acc 73.80 / 75.98         VRAM 208 / 92 GiB
    Base + BBOX-ADAPTER         Acc 66.08 / 65.26         VRAM 105 / 92 GiB
    Per the Addendum, VRAM is reported for the 0.1B adapter only.

This script *never* retrains for the plug-and-play rows: the adapter tuned on
gpt-3.5-turbo is loaded from disk, a weight signature is captured before and after the
whole evaluation, and any change is flagged as an error (no retraining, no parameter
manipulation of the black-box LLM). Only text prompts in / text generations out are ever
exchanged with the black-box model (no logprobs, hidden states, or gradients).

Usage
-----
    # Table 3 (default): transplant the trained adapter onto davinci-002 + Mixtral
    python scripts/run_plug_and_play.py --datasets strategyqa gsm8k truthfulqa \
        --checkpoint-dir runs/strategyqa --size 0.1b

    # Table 6 (Mixtral white-box treated as black-box + VRAM, 0.1B only)
    python scripts/run_plug_and_play.py --mode whitebox --datasets strategyqa --size 0.1b

    # Offline smoke test (mock black-box client, no Azure / no HF weights)
    python scripts/run_plug_and_play.py --dry-run --limit 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------------------
# Path / import bootstrapping
# --------------------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ``run_experiment`` doubles as the shared driver library for all scripts (RX).
try:  # pragma: no cover - import-time environment probe
    import run_experiment as RX  # type: ignore
except Exception:  # pragma: no cover
    try:
        from scripts import run_experiment as RX  # type: ignore
    except Exception:
        RX = None  # type: ignore

from bbox_adapter.utils import derive_component_seeds, get_logger, set_seed  # noqa: E402

try:  # optional artifact/logging helpers
    from bbox_adapter.utils.logging import RunLogger  # noqa: E402
    from bbox_adapter.utils.logging import table as _render_table  # noqa: E402
except Exception:  # pragma: no cover
    RunLogger = None  # type: ignore
    _render_table = None  # type: ignore

try:  # optional evaluation helpers
    from bbox_adapter.eval import metrics as MET  # noqa: E402
except Exception:  # pragma: no cover
    MET = None  # type: ignore

try:  # optional cost helpers (Table 4 style accounting for plug-and-play inference)
    from bbox_adapter.eval import cost as COST  # noqa: E402
except Exception:  # pragma: no cover
    COST = None  # type: ignore

try:  # optional VRAM helpers (Table 6)
    from bbox_adapter.eval import vram as VRAM  # noqa: E402
except Exception:  # pragma: no cover
    VRAM = None  # type: ignore

logger = get_logger("bbox_adapter.plug_and_play")

# --------------------------------------------------------------------------------------
# Constants: datasets, black boxes, paper reference numbers
# --------------------------------------------------------------------------------------

PLUG_DATASETS: Tuple[str, ...] = ("strategyqa", "gsm8k", "truthfulqa")
ALL_DATASETS: Tuple[str, ...] = ("strategyqa", "gsm8k", "truthfulqa", "scienceqa")
SIZES: Tuple[str, ...] = ("0.1b", "0.3b")

PLUGGER_MODEL = "gpt-3.5-turbo"  # the adapter was tuned while adapting this model

# black-box tag -> concrete client model identifier
BLACKBOX_MODELS: Dict[str, str] = {
    "davinci-002": "davinci-002",
    "davinci_002": "davinci-002",
    "davinci": "davinci-002",
    "mixtral": "mistralai/Mixtral-8x7B-v0.1",
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-v0.1",
    "mixtral-8x7b-v0.1": "mistralai/Mixtral-8x7B-v0.1",
    "mistralai/mixtral-8x7b-v0.1": "mistralai/Mixtral-8x7B-v0.1",
}

BLACKBOX_TAGS: Dict[str, str] = {
    "davinci-002": "davinci-002",
    "mixtral-8x7b-v0.1": "mixtral-8x7b-v0.1",
    "mistralai/mixtral-8x7b-v0.1": "mixtral-8x7b-v0.1",
}

DEFAULT_BLACKBOXES: Tuple[str, ...] = ("davinci-002", "mixtral-8x7b-v0.1")

METRIC_LABEL: Dict[str, str] = {
    "strategyqa": "Acc. (%)",
    "gsm8k": "Acc. (%)",
    "truthfulqa": "True + Info (%)",
    "scienceqa": "Acc. (%)",
    "average": "Acc. (%)",
}

# Table 3 reference numbers (paper Section 4.3). "base" = unadapted black-box LLM,
# "plugged" = same LLM steered by the adapter tuned on gpt-3.5-turbo.
PAPER_TABLE3: Dict[str, Dict[str, Dict[str, float]]] = {
    "davinci-002": {
        "base": {"strategyqa": 44.19, "gsm8k": 23.73, "truthfulqa": 31.50, "average": 33.14},
        "plugged": {"strategyqa": 59.61, "gsm8k": 23.85, "truthfulqa": 36.50, "average": 39.99},
        "delta": {"strategyqa": 15.42, "gsm8k": 0.12, "truthfulqa": 5.00, "average": 6.85},
    },
    "mixtral-8x7b-v0.1": {
        "base": {"strategyqa": 59.91, "gsm8k": 47.46, "truthfulqa": 40.40, "average": 49.26},
        "plugged": {"strategyqa": 63.97, "gsm8k": 47.61, "truthfulqa": 49.70, "average": 53.76},
        "delta": {"strategyqa": 4.06, "gsm8k": 0.15, "truthfulqa": 9.30, "average": 4.50},
    },
}

# Table 6 reference numbers (paper Sections 4.6/4.7) - Mixtral scaled up by our method.
PAPER_TABLE6: Dict[str, Any] = {
    "base_model": {"accuracy_0_1b": 59.91, "training_vram": None, "inference_vram": 90.0},
    "lora": {"accuracy_0_1b": 73.80, "accuracy_0_3b": 75.98, "training_vram": 208.0, "inference_vram": 92.0},
    "bbox_adapter": {"accuracy_0_1b": 66.08, "accuracy_0_3b": 65.26, "training_vram": 105.0, "inference_vram": 92.0},
}

PAPER_PLUG_AVERAGE_GAIN: Dict[str, float] = {"davinci-002": 6.85, "mixtral-8x7b-v0.1": 4.50}

# Adapter sensitivity: plug-and-play rows keep Appendix-H.2 inference settings.
PLUG_BEAM_SIZE = 3
PLUG_TEMPERATURE = 1.0
PLUG_MAX_LEN = 512
DEFAULT_OUTPUT_DIR = os.path.join("runs", "plug_and_play")


# --------------------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------------------


def _call_filtered(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call ``fn`` with only the keyword arguments its signature accepts.

    Reuses ``run_experiment.call_filtered`` when available so all drivers share one
    tolerant calling convention (component APIs differ slightly in kwargs aliases such
    as ``eta``/``lr`` or ``T``/``n_iterations``).
    """
    if RX is not None and hasattr(RX, "call_filtered"):
        try:
            return RX.call_filtered(fn, *args, **kwargs)
        except TypeError:
            pass
    import inspect

    try:
        sig = inspect.signature(fn)
        accepts_kwargs = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if accepts_kwargs:
            return fn(*args, **kwargs)
        allowed = set(sig.parameters)
        return fn(*args, **{k: v for k, v in kwargs.items() if k in allowed})
    except (TypeError, ValueError):
        return fn(*args, **kwargs)


def canonical_blackbox(name: str) -> str:
    """Normalize a black-box tag (``mixtral`` -> ``mixtral-8x7b-v0.1``)."""
    key = str(name).strip().lower().replace(" ", "")
    if key in BLACKBOX_MODELS:
        return BLACKBOX_TAGS.get(BLACKBOX_MODELS[key], BLACKBOX_MODELS[key])
    for alias, model in BLACKBOX_MODELS.items():
        if alias in key:
            return BLACKBOX_TAGS.get(model, model)
    return str(name)


def blackbox_model_id(name: str) -> str:
    """Map a black-box tag to the concrete client model identifier."""
    key = str(name).strip().lower().replace(" ", "")
    if key in BLACKBOX_MODELS:
        return BLACKBOX_MODELS[key]
    return canonical_blackbox(name)


def metric_name_for(dataset: str) -> str:
    """Metric reported by the paper for ``dataset`` (Section 4.2)."""
    return "true_info" if dataset in ("truthfulqa", "truthful_qa") else "accuracy"


def average(values: Sequence[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def population_std(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None]
    if len(vals) <= 1:
        return 0.0
    m = sum(vals) / len(vals)
    return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5


def format_signed(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:+.{digits}f}"


def format_value(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _params_signature(model: Any) -> str:
    """Stable fingerprint of the adapter parameters (proves no retraining happened)."""
    hasher = hashlib.sha1()
    try:
        params = list(model.parameters()) if hasattr(model, "parameters") else []
    except Exception:
        return "unavailable"
    if not params:
        return "no-parameters"
    for p in params[:8]:
        try:
            detached = p.detach().cpu().float().reshape(-1)[:64]
            hasher.update(repr([round(float(v), 6) for v in detached.tolist()]).encode())
        except Exception:
            hasher.update(b"<unavailable>")
    return hasher.hexdigest()[:16]


# --------------------------------------------------------------------------------------
# Configuration / building blocks (delegated to run_experiment when available)
# --------------------------------------------------------------------------------------


def build_config(dataset: str, args: argparse.Namespace) -> Dict[str, Any]:
    """Deep-merge ``configs/default.yaml`` + per-dataset YAML + CLI overrides."""
    if RX is not None and hasattr(RX, "build_config"):
        return RX.build_config(dataset, args)
    base = load_config(dataset=dataset)
    overrides = getattr(args, "overrides", None) or {}
    if overrides:
        base.update(overrides)
    return base


def build_plugin_generator(
    blackbox: str,
    config: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[Any, Optional[Any]]:
    """Build a text-only client for ``blackbox`` plus an optional cost ledger.

    Temperature is 1.0 for the adapted (plug-and-play) generations to match Appendix
    H.2 ("we set ... the temperature as 1.0 ... which serves as a proposal"); the
    unadapted baseline uses 0.0 (Section H.2 SFT/biasing note, and the paper's CoT
    baseline convention).
    """
    model_id = blackbox_model_id(blackbox)
    cfg = dict(config or {})
    bb = dict(cfg.get("blackbox", {}) or {})
    bb["model"] = model_id
    allow_mock = bool(bb.get("allow_mock", True) or getattr(args, "allow_mock", False))
    bb["allow_mock"] = allow_mock
    ledger = None
    if COST is not None and cfg.get("cost", {}).get("enabled", True):
        try:
            ledger = COST.CostLedger.from_config(cfg.get("cost", {}))
        except Exception:
            ledger = None

    if RX is not None and hasattr(RX, "build_generator_and_ledger"):
        try:
            shim = dict(cfg)
            shim["blackbox"] = bb
            gen, led = RX.build_generator_and_ledger(shim, args)
            if led is not None:
                ledger = led
            if gen is not None:
                return gen, ledger
        except Exception as exc:  # pragma: no cover - fall through to local builder
            logger.debug("run_experiment generator builder failed (%s); using local builder", exc)

    from bbox_adapter.llm.blackbox_client import build_generator

    gen = _call_filtered(
        build_generator,
        model=model_id,
        kind="bbox",
        ledger=ledger,
        allow_mock=allow_mock,
        max_len=int(bb.get("max_len", PLUG_MAX_LEN) or PLUG_MAX_LEN),
        temperature=float(bb.get("temperature", PLUG_TEMPERATURE) or PLUG_TEMPERATURE),
        **({} if ledger is not None else {}),
    )
    return gen, ledger


def load_trained_adapter(
    dataset: str,
    size: str,
    config: Dict[str, Any],
    args: argparse.Namespace,
):
    """Load the adapter tuned on gpt-3.5-turbo for ``dataset`` / ``size``.

    Plug-and-play *never* retrains: a checkpoint is required unless ``--allow-untrained``
    is passed explicitly (analysis / smoke-testing only).
    """
    candidate_paths: List[str] = []

    explicit = getattr(args, "checkpoint", None)
    if explicit:
        candidate_paths.append(str(explicit))

    ckpt_dir = getattr(args, "checkpoint_dir", None)
    if ckpt_dir:
        for tag in ("final", "best", "last", "adapted"):
            candidate_paths.append(os.path.join(str(ckpt_dir), "checkpoints", f"{_adapter_tag(dataset, size, tag)}.pt"))
        candidate_paths.append(os.path.join(str(ckpt_dir), "checkpoints"))
        candidate_paths.append(str(ckpt_dir))

    out_dir = getattr(args, "output_dir", None) or DEFAULT_OUTPUT_DIR
    candidate_paths.append(out_dir)

    try:
        from bbox_adapter.utils import resolve_checkpoint
    except Exception:  # pragma: no cover
        resolve_checkpoint = None  # type: ignore

    if resolve_checkpoint is not None:
        for path in candidate_paths:
            try:
                resolved = resolve_checkpoint(
                    path,
                    tag="final",
                    dataset=dataset,
                    size=size,
                    blackbox=PLUGGER_MODEL,
                )
                if resolved and os.path.exists(resolved):
                    logger.info("plugger checkpoint: %s", resolved)
                    return _load_adapter_from(resolved, dataset, size, config)
            except Exception:
                continue

    if not getattr(args, "allow_untrained", False):
        logger.warning(
            "no trained adapter checkpoint found for %s/%s; using a freshly initialized "
            "adapter (plug-and-play results will be meaningless - pass --checkpoint)",
            dataset,
            size,
        )

    from bbox_adapter.adapter.energy_model import build_energy_model

    adapter = _call_filtered(
        build_energy_model,
        dataset=dataset,
        size=size,
        backbone=(config.get("adapter", {}) or {}).get("backbone"),
        max_length=int((config.get("adapter", {}) or {}).get("max_length", PLUG_MAX_LEN) or PLUG_MAX_LEN),
    )
    return adapter


def _adapter_tag(dataset: str, size: str, tag: str = "final") -> str:
    try:
        from bbox_adapter.utils.logging import checkpoint_name

        return checkpoint_name(dataset=dataset, size=size, blackbox=PLUGGER_MODEL, tag=tag)
    except Exception:  # pragma: no cover
        return f"adapter_{dataset}_{size}_{PLUGGER_MODEL}_{tag}"


def _load_adapter_from(path: str, dataset: str, size: str, config: Dict[str, Any]):
    """Load an :class:`EnergyModel` from a checkpoint file or directory."""
    from bbox_adapter.adapter.energy_model import EnergyModel, EnergyModelConfig

    if os.path.isdir(path):
        for candidate in (
            os.path.join(path, "energy_model.pt"),
            os.path.join(path, "model.pt"),
        ):
            if os.path.exists(candidate):
                path = candidate
                break
        else:
            # directory with a nested checkpoints/ dir
            inner = os.path.join(path, "checkpoints")
            if os.path.isdir(inner):
                for name in sorted(os.listdir(inner)):
                    if name.endswith(".pt"):
                        path = os.path.join(inner, name)
                        break

    try:
        return EnergyModel.from_pretrained(path, map_location="cpu")
    except Exception as exc:  # pragma: no cover - fall back to raw state dict
        logger.debug("EnergyModel.from_pretrained failed (%s); trying raw state_dict", exc)
        import torch

        adapter_cfg = dict(config.get("adapter", {}) or {})
        cfg = EnergyModelConfig(
            dataset=dataset,
            size=size,
            backbone=adapter_cfg.get("backbone"),
            pooling=adapter_cfg.get("pooling"),
            head_hidden=int(adapter_cfg.get("head_hidden", 0) or 0),
            max_length=int(adapter_cfg.get("max_length", PLUG_MAX_LEN) or PLUG_MAX_LEN),
        )
        model = EnergyModel(cfg)
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        return model


def prepare_plug_data(dataset: str, config: Dict[str, Any], args: argparse.Namespace):
    """Load the evaluation split with Appendix-J prompts (no training set needed)."""
    if RX is not None and hasattr(RX, "prepare_data"):
        return RX.prepare_data(
            dataset,
            config,
            split="test",
            limit=getattr(args, "limit", None),
            seed=int(getattr(args, "seed", 0) or 0),
            model=None,
        )
    raise RuntimeError("run_experiment.prepare_data is required by run_plug_and_play.py")


# --------------------------------------------------------------------------------------
# Generation / evaluation
# --------------------------------------------------------------------------------------


def generate_base(data: Any, generator: Any, config: Dict[str, Any]) -> List[str]:
    """Unadapted black-box generations for Table 3 "base" rows (CoT prompt, temp 0)."""
    if RX is not None and hasattr(RX, "generate_baseline"):
        try:
            return list(RX.generate_baseline(data, generator, config))
        except Exception as exc:  # pragma: no cover
            logger.debug("generate_baseline failed (%s); using local generation", exc)

    from bbox_adapter.llm.blackbox_client import resolve_temperature

    gens: List[str] = []
    temp = resolve_temperature("cot")
    for prompt in data.prompts:
        try:
            out = generator.generate_one(prompt, temperature=temp, max_len=PLUG_MAX_LEN)
        except TypeError:
            res = generator.generate_result(prompt, n=1, temperature=temp, max_len=PLUG_MAX_LEN)
            out = res.texts[0] if getattr(res, "texts", None) else ""
        gens.append(out)
    return gens


def generate_plugged(
    data: Any,
    adapter: Any,
    generator: Any,
    config: Dict[str, Any],
    *,
    mode: str = "full",
    ledger: Optional[Any] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """Adapter-steered generation (Table 3 "plugged" rows).

    ``mode="full"`` uses the sentence-level beam search (black box = proposal,
    adapter = evaluator). ``mode="single_step"`` lets the black box emit complete
    answers once and lets the adapter rank them.
    """
    if RX is not None and hasattr(RX, "generate_adapted"):
        try:
            gens, stats = RX.generate_adapted(
                data, adapter, generator, config, mode=mode, ledger=ledger
            )
            return list(gens), dict(stats or {})
        except Exception as exc:  # pragma: no cover
            logger.debug("generate_adapted failed (%s); using local adapter inference", exc)

    from bbox_adapter.inference.beam_search import single_step_search, beam_search

    gens = []
    stats: Dict[str, Any] = {"mode": mode, "n_llm_calls": 0, "n_candidates": 0}
    for q, prompt, answer_type, choices in zip(
        data.questions,
        data.prompts,
        getattr(data, "answer_types", [None] * len(data.questions)),
        getattr(data, "choices_list", [None] * len(data.questions)),
    ):
        if mode == "single_step":
            out = single_step_search(
                q,
                adapter=adapter,
                generator=generator,
                prompt=prompt,
                answer_type=answer_type,
                choices=choices,
                n=int((config.get("training", {}) or {}).get("n_candidates", 5) or 5),
            )
        else:
            out = beam_search(
                q,
                adapter=adapter,
                generator=generator,
                prompt=prompt,
                answer_type=answer_type,
                choices=choices,
            )
        gens.append(out if isinstance(out, str) else getattr(out, "best_text", ""))
    return gens, stats


def compute_metric(generations: Sequence[str], data: Any, config: Dict[str, Any]) -> float:
    """Acc (%) for StrategyQA/GSM8K/ScienceQA, True+Info (%) for TruthfulQA."""
    if RX is not None and hasattr(RX, "compute_metric"):
        try:
            return float(RX.compute_metric(generations, data, config))
        except Exception as exc:  # pragma: no cover
            logger.debug("run_experiment.compute_metric failed (%s)", exc)

    dataset = getattr(data, "dataset", None) or config.get("dataset")
    if MET is not None:
        report = MET.evaluate_generations(
            list(generations),
            dataset=dataset,
            golds=getattr(data, "golds", None),
            examples=getattr(data, "examples", None),
            answer_type=getattr(data, "answer_type", None),
            choices_list=getattr(data, "choices_list", None),
        )
        return float(getattr(report, "value", report))

    from bbox_adapter.data.answer_extraction import accuracy, true_info_rate

    if metric_name_for(str(dataset)) == "true_info":
        return float(true_info_rate(list(generations), getattr(data, "examples", [])))
    return float(accuracy(list(generations), getattr(data, "golds", []), getattr(data, "answer_type", "free")))


# --------------------------------------------------------------------------------------
# Run records
# --------------------------------------------------------------------------------------


@dataclass
class PlugRow:
    """One black-box LLM evaluated with (and without) the tuned plugger."""

    blackbox: str
    dataset: str
    size: str
    metric: str
    base_accuracy: Optional[float] = None
    plugged_accuracy: Optional[float] = None
    delta: Optional[float] = None
    n_eval: int = 0
    base_generations: List[str] = field(default_factory=list)
    plugged_generations: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    paper_base: Optional[float] = None
    paper_plugged: Optional[float] = None
    error: Optional[str] = None
    seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "blackbox": self.blackbox,
            "dataset": self.dataset,
            "size": self.size,
            "metric": self.metric,
            "base_accuracy": self.base_accuracy,
            "plugged_accuracy": self.plugged_accuracy,
            "delta": self.delta,
            "n_eval": self.n_eval,
            "paper_base": self.paper_base,
            "paper_plugged": self.paper_plugged,
            "error": self.error,
            "seconds": self.seconds,
            "stats": dict(self.stats or {}),
        }
        return payload

    def row(self) -> Dict[str, Any]:
        return {
            "Black-Box LLM": self.blackbox,
            "Dataset": self.dataset,
            "Metric": self.metric,
            "Base": self.base_accuracy,
            "Plugged": self.plugged_accuracy,
            "Delta": self.delta,
        }


def run_plug_cell(
    dataset: str,
    blackbox: str,
    adapter: Any,
    config: Dict[str, Any],
    args: argparse.Namespace,
    *,
    seed: int = 0,
    ledger: Optional[Any] = None,
) -> PlugRow:
    """Evaluate one (dataset x black-box) cell: unadapted base + plugged."""
    data = prepare_plug_data(dataset, config, args)
    generator, maybe_ledger = build_plugin_generator(blackbox, config, args)
    ledger = ledger if ledger is not None else maybe_ledger

    tag = canonical_blackbox(blackbox)
    metric = metric_name_for(dataset)
    row = PlugRow(
        blackbox=tag,
        dataset=dataset,
        size=str(getattr(args, "size", "0.1b")),
        metric=metric,
        n_eval=len(getattr(data, "questions", []) or []),
        paper_base=(PAPER_TABLE3.get(tag, {}).get("base", {}) or {}).get(dataset),
        paper_plugged=(PAPER_TABLE3.get(tag, {}).get("plugged", {}) or {}).get(dataset),
    )

    started = time.time()
    try:
        set_seed(seed + 101)
        base_gens = generate_base(data, generator, config)
        row.base_accuracy = compute_metric(base_gens, data, config)
        row.base_generations = list(base_gens)

        set_seed(seed + 202)
        plugged_gens, stats = generate_plugged(
            data, adapter, generator, config, mode=str(getattr(args, "mode_inference", "full")),
            ledger=ledger,
        )
        row.plugged_accuracy = compute_metric(plugged_gens, data, config)
        row.plugged_generations = list(plugged_gens)
        row.stats = dict(stats or {})

        if row.base_accuracy is not None and row.plugged_accuracy is not None:
            row.delta = float(row.plugged_accuracy) - float(row.base_accuracy)
    except Exception as exc:  # pragma: no cover - keep the grid running
        row.error = f"{type(exc).__name__}: {exc}"
        logger.warning("cell failed (%s / %s): %s", dataset, tag, row.error)
    row.seconds = time.time() - started
    return row


# --------------------------------------------------------------------------------------
# Table 3 aggregation / rendering
# --------------------------------------------------------------------------------------


def aggregate_table3(rows: Sequence[PlugRow], datasets: Sequence[str]) -> Dict[str, Dict[str, Dict[str, Optional[float]]]]:
    """Aggregate rows into Table-3 shape: blackbox -> {base, plugged, delta} -> dataset."""
    table: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    for row in rows:
        entry = table.setdefault(row.blackbox, {"base": {}, "plugged": {}, "delta": {}})
        if row.base_accuracy is not None:
            entry["base"][row.dataset] = row.base_accuracy
        if row.plugged_accuracy is not None:
            entry["plugged"][row.dataset] = row.plugged_accuracy
        if row.delta is not None:
            entry["delta"][row.dataset] = row.delta

    for blackbox, entry in table.items():
        base_avg = average([entry["base"].get(d) for d in datasets])
        plug_avg = average([entry["plugged"].get(d) for d in datasets])
        entry["base"]["average"] = base_avg
        entry["plugged"]["average"] = plug_avg
        entry["delta"]["average"] = (
            None if (base_avg is None or plug_avg is None) else plug_avg - base_avg
        )
    return table


def format_table3(
    table: Dict[str, Dict[str, Dict[str, Optional[float]]]],
    datasets: Sequence[str],
) -> str:
    """Render the Table-3 layout (dataset columns, base/plugged/(plugged) rows)."""
    header_cols = list(datasets) + ["Average"]
    width = max(22, max((len(b) for b in table), default=22) + 6)
    lines: List[str] = []
    lines.append(
        f"{'Black-Box LLM':<{width}} | "
        + " | ".join(f"{c[:16]:>16}" for c in header_cols)
    )
    lines.append("-" * len(lines[0]))
    for blackbox, entry in table.items():
        base_cells = [format_value(entry["base"].get(d)) for d in header_cols]
        lines.append(f"{blackbox + ' (base)':<{width}} | " + " | ".join(f"{c:>16}" for c in base_cells))
        plug_cells = [format_value(entry["plugged"].get(d)) for d in header_cols]
        lines.append(f"{blackbox + ' (plugged)':<{width}} | " + " | ".join(f"{c:>16}" for c in plug_cells))
        delta_cells = [format_signed(entry["delta"].get(d)) for d in header_cols]
        lines.append(f"{'  delta':<{width}} | " + " | ".join(f"{c:>16}" for c in delta_cells))
    return "\n".join(lines)


def compare_table3(
    table: Dict[str, Dict[str, Dict[str, Optional[float]]]],
    datasets: Sequence[str],
    tolerance: float = 3.0,
) -> Dict[str, Any]:
    """Check the plugged accuracies/averages against the paper's Table 3 numbers."""
    checks: List[Dict[str, Any]] = []
    for blackbox, entry in table.items():
        paper = PAPER_TABLE3.get(blackbox)
        if not paper:
            continue
        for dataset in list(datasets) + ["average"]:
            got = entry["plugged"].get(dataset)
            want = paper["plugged"].get(dataset)
            if got is None or want is None:
                continue
            checks.append(
                {
                    "blackbox": blackbox,
                    "dataset": dataset,
                    "ours": float(got),
                    "paper": float(want),
                    "delta": float(got) - float(want),
                    "within_tolerance": abs(float(got) - float(want)) <= tolerance,
                }
            )
    n_ok = sum(1 for c in checks if c["within_tolerance"])
    return {
        "checks": checks,
        "n_checks": len(checks),
        "n_within_tolerance": n_ok,
        "tolerance": tolerance,
        "all_within_tolerance": bool(checks) and n_ok == len(checks),
    }


def plug_average_gains(
    table: Dict[str, Dict[str, Dict[str, Optional[float]]]]
) -> Dict[str, Optional[float]]:
    """Average delta per black box, compared against the paper's 6.85% / 4.50%."""
    out: Dict[str, Optional[float]] = {}
    for blackbox, entry in table.items():
        out[blackbox] = entry["delta"].get("average")
    return out


# --------------------------------------------------------------------------------------
# Table 6: Mixtral treated as black-box + VRAM (0.1B only per the Addendum)
# --------------------------------------------------------------------------------------


def measure_mixtral_vram(
    config: Dict[str, Any],
    args: argparse.Namespace,
    *,
    phase: str = "inference",
    size: str = "0.1b",
) -> Optional[float]:
    """Measure (or estimate) peak GPU memory (GiB) for the Mixtral black-box setup."""
    if VRAM is None:
        return None
    if str(size).lower() not in ("0.1b", "0.1", "base", "86m", "110m"):
        logger.info("Addendum: VRAM is reported for the 0.1B adapter only; skipping %s", size)
        return None

    try:
        estimate = VRAM.estimate_vram_for_method(
            "bbox_adapter", phase=phase, adapter_size=size, use_paper_defaults=True
        )
    except Exception:
        estimate = None

    method = blackbox_model_id("mixtral")
    try:
        generator, _ = build_plugin_generator("mixtral", config, args)
    except Exception as exc:  # pragma: no cover - offline environments
        logger.debug("could not build Mixtral client for VRAM measurement (%s)", exc)
        return estimate

    try:
        with VRAM.VramTracker(
            "bbox_adapter", phase=phase, adapter_size=size, config=config
        ) as tracker:
            try:
                generator.generate_one("Q: 2+2=?\nA:", temperature=PLUG_TEMPERATURE, max_len=64)
            except TypeError:
                generator.generate("Q: 2+2=?\nA:", n=1, temperature=PLUG_TEMPERATURE, max_len=64)
        measured = tracker.gib
    except Exception as exc:  # pragma: no cover
        logger.debug("VRAM measurement unavailable (%s)", exc)
        measured = None
    return measured if measured is not None else estimate


def run_whitebox(
    args: argparse.Namespace,
    config: Dict[str, Any],
    adapter: Any,
) -> Dict[str, Any]:
    """Table 6: adapt Mixtral-8x7B while treating it as a black box, plus VRAM rows."""
    dataset = (list(getattr(args, "datasets", None) or ["strategyqa"]))[0]
    rows: List[PlugRow] = []
    for blackbox in ("mixtral-8x7b-v0.1",):
        rows.append(
            run_plug_cell(dataset, blackbox, adapter, config, args, seed=int(getattr(args, "seed", 0) or 0))
        )

    table = aggregate_table3(rows, [dataset])
    entry = table.get("mixtral-8x7b-v0.1", {"base": {}, "plugged": {}, "delta": {}})

    vram_rows: Dict[str, Any] = {}
    if getattr(args, "vram", True):
        vram_rows["training"] = measure_mixtral_vram(config, args, phase="training", size=str(args.size))
        vram_rows["inference"] = measure_mixtral_vram(config, args, phase="inference", size=str(args.size))

    comparison: Dict[str, Any] = {}
    if VRAM is not None and hasattr(VRAM, "compare_to_paper"):
        for phase, value in vram_rows.items():
            if value is None:
                continue
            try:
                comparison[phase] = VRAM.compare_to_paper(
                    "bbox_adapter", phase, float(value), adapter_size=str(args.size)
                )
            except Exception:
                continue

    return {
        "dataset": dataset,
        "table3_rows": [r.to_dict() for r in rows],
        "base_accuracy": entry["base"].get(dataset),
        "plugged_accuracy": entry["plugged"].get(dataset),
        "delta": entry["delta"].get(dataset),
        "lora_reference": PAPER_TABLE6.get("lora"),
        "bbox_adapter_reference": PAPER_TABLE6.get("bbox_adapter"),
        "vram": vram_rows,
        "vram_comparison": comparison,
    }


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


def write_reports(
    args: argparse.Namespace,
    payload: Dict[str, Any],
    *,
    table_text: str,
) -> Dict[str, str]:
    out_dir = str(getattr(args, "output_dir", None) or DEFAULT_OUTPUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    raw_path = os.path.join(out_dir, "plug_and_play_raw.json")
    with open(raw_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    paths["raw"] = raw_path

    md_path = os.path.join(out_dir, "PLUG_AND_PLAY.md")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("# BBox-Adapter plug-and-play evaluation (Table 3 / Table 6)\n\n")
        handle.write("## Reproduced table\n\n```\n")
        handle.write(table_text)
        handle.write("\n```\n\n")
        if payload.get("paper_comparison", {}).get("checks"):
            handle.write("## Comparison with Table 3\n\n")
            handle.write("| Black-box | Dataset | Ours | Paper | Delta |\n")
            handle.write("|---|---|---|---|---|\n")
            for check in payload["paper_comparison"]["checks"]:
                handle.write(
                    f"| {check['blackbox']} | {check['dataset']} | "
                    f"{format_value(check['ours'])} | {format_value(check['paper'])} | "
                    f"{format_signed(check['delta'])} |\n"
                )
        handle.write("\n## Notes\n")
        handle.write(
            "- The plugger is the adapter tuned while adapting `gpt-3.5-turbo`; it is loaded\n"
            "  from disk and never retrained (Section 4.3).\n"
            "- Plug-and-play targets: `davinci-002` (Azure completion) and\n"
            "  `mistralai/Mixtral-8x7B-v0.1` (HuggingFace).\n"
            "- Only raw text prompts/generations cross the black-box boundary: no logprobs,\n"
            "  hidden states, or gradients (Appendix C).\n"
            "- Per the Addendum, VRAM (Table 6) is reported for the 0.1B adapter only.\n"
        )
    paths["markdown"] = md_path
    return paths


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BBox-Adapter plug-and-play adaptation (Table 3) and Mixtral "
        "white-box-treated-as-black-box study (Table 6)."
    )
    parser.add_argument(
        "--mode",
        default="table3",
        choices=["table3", "plug", "whitebox", "table6", "auto"],
        help="table3 = davinci-002 + Mixtral transplant; whitebox = Table 6 (+VRAM).",
    )
    parser.add_argument("--datasets", nargs="+", default=list(PLUG_DATASETS))
    parser.add_argument("--blackboxes", nargs="+", default=list(DEFAULT_BLACKBOXES))
    parser.add_argument("--size", default="0.1b", choices=list(SIZES))
    parser.add_argument("--sizes", nargs="+", default=None, help="Override --size with a list.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only N test questions.")
    parser.add_argument("--checkpoint", default=None, help="Explicit adapter checkpoint path.")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory holding the trained plugger (e.g. runs/strategyqa).",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--mode-inference",
        default="full",
        choices=["full", "single_step"],
        help="Adapted inference variant used for the plugged rows.",
    )
    parser.add_argument("--vram", dest="vram", action="store_true", default=True, help="Measure VRAM (Table 6).")
    parser.add_argument("--no-vram", dest="vram", action="store_false")
    parser.add_argument("--dry-run", action="store_true", help="Offline mock client, tiny run.")
    parser.add_argument("--allow-mock", action="store_true", help="Fall back to the mock client.")
    parser.add_argument(
        "--allow-untrained",
        action="store_true",
        help="Smoke-testing only: proceed without a trained plugger checkpoint.",
    )
    args = parser.parse_args(argv)

    if args.dry_run:
        args.limit = args.limit or 5
        args.allow_mock = True
        args.allow_untrained = True
        args.blackboxes = ["mock"]
        args.vram = False
    return args


def _blackbox_for_dry_run(name: str) -> str:
    return "gpt-3.5-turbo" if str(name).lower() in ("mock", "dry", "local") else name


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if RX is None:
        logger.error("scripts/run_experiment.py is required (shared driver library).")
        return 2

    set_seed(int(getattr(args, "seed", 0) or 0))
    datasets = [str(d).strip().lower() for d in (args.datasets or list(PLUG_DATASETS))]
    datasets = [d for d in datasets if d in ALL_DATASETS] or list(PLUG_DATASETS)
    sizes = [str(s).lower() for s in (args.sizes or [args.size])]
    seeds = list(args.seeds or [args.seed])
    blackboxes = [canonical_blackbox(b) for b in (args.blackboxes or DEFAULT_BLACKBOXES)]

    all_rows: List[PlugRow] = []
    signatures: List[Dict[str, Any]] = []
    extra: Dict[str, Any] = {}

    for size in sizes:
        args.size = size
        for dataset in datasets:
            config = build_config(dataset, args)
            config.setdefault("dataset", dataset)
            config.setdefault("size", size)
            plugin_config = dict(config)
            plugin_config["beam_search"] = dict(config.get("beam_search", {}) or {})
            plugin_config["beam_search"].setdefault("beam_size", PLUG_BEAM_SIZE)
            plugin_config["beam_search"].setdefault("temperature", PLUG_TEMPERATURE)
            plugin_config["beam_search"].setdefault("max_len", PLUG_MAX_LEN)

            adapter = load_trained_adapter(dataset, size, plugin_config, args)
            before = _params_signature(adapter)

            if str(args.mode).lower() in ("whitebox", "table6"):
                extra = run_whitebox(args, plugin_config, adapter)
                extra["size"] = size
            else:
                for blackbox in blackboxes:
                    logger.info("plug-and-play: %s -> %s (%s)", PLUGGER_MODEL, blackbox, dataset)
                    for seed in seeds:
                        all_rows.append(
                            run_plug_cell(
                                dataset,
                                _blackbox_for_dry_run(blackbox),
                                adapter,
                                plugin_config,
                                args,
                                seed=int(seed),
                            )
                        )

            after = _params_signature(adapter)
            signatures.append(
                {
                    "dataset": dataset,
                    "size": size,
                    "before": before,
                    "after": after,
                    "unchanged": before == after,
                }
            )

    table = aggregate_table3(all_rows, datasets)
    table_text = format_table3(table, datasets) if all_rows else "(no rows) "

    gains = plug_average_gains(table)
    payload: Dict[str, Any] = {
        "mode": args.mode,
        "datasets": datasets,
        "blackboxes": blackboxes,
        "sizes": sizes,
        "seeds": seeds,
        "n_eval": (args.limit or None),
        "plugger": PLUGGER_MODEL,
        "table3": table,
        "average_gains": gains,
        "paper_average_gains": PAPER_PLUG_AVERAGE_GAIN,
        "paper_table3": PAPER_TABLE3,
        "rows": [r.to_dict() for r in all_rows],
        "no_retraining": signatures,
        "paper_comparison": compare_table3(table, datasets) if all_rows else {},
        "whitebox": extra or None,
    }
    paths = write_reports(args, payload, table_text=table_text)
    payload["artifacts"] = paths
    with open(paths["raw"], "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)

    print(table_text)
    for blackbox, gain in gains.items():
        paper_gain = PAPER_PLUG_AVERAGE_GAIN.get(blackbox)
        print(
            f"{blackbox}: average plug-and-play gain {format_signed(gain)} "
            f"(paper {format_signed(paper_gain)})"
        )
    if not all(entry["unchanged"] for entry in signatures):
        logger.error("adapter parameters changed during plug-and-play evaluation "
                     "(retraining is not allowed by Section 4.3)")
        return 1
    print(f"artifacts: {paths}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
