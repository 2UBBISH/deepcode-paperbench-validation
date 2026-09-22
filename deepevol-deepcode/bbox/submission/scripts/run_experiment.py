#!/usr/bin/env python
"""Main BBox-Adapter experiment driver (reproduces Table 2 / Table 4 / Fig. 3(b)).

Paper: "Lightweight Adapting for Black-Box Large Language Models" (BBox-Adapter).

This entry point drives the online adaptation of a frozen black-box LLM
(gpt-3.5-turbo / davinci-002 / Mixtral-8x7B) via a small pretrained energy
adapter ``g_theta`` learned with the ranking-based NCE objective of Eq. (2)-(3),
evaluated with the sentence-level beam search of Eq. (4) (Section 3.3).

Supported runs
--------------
* ``--setting ground_truth``  -> Table 2 row "BBox-Adapter (Ground-Truth)"
* ``--setting ai_feedback``   -> Table 2 row "BBox-Adapter (AI Feedback)"
* ``--setting combined``      -> Table 2 row "BBox-Adapter (Combined)"
* ``--mode single``           -> Table 4 "single-step" variant (base model emits
                                 complete answers, the adapter only ranks them)
* ``--mode full``             -> Table 4 "full-step" variant (sentence-level beam
                                 search, beam size 3 by default)
* ``--baseline cot``          -> CoT prompting of the *unadapted* black-box LLM
* ``--sweep-beams``           -> Figure 3(a) beam-size sweep (k = 1, 3, 5)
* ``--sweep-iterations``      -> Figure 3(b) iteration sweep (T = 0..4)

Black-box contract (Appendix C): only raw text is ever requested.  The client
asserts that no ``logprobs``/``echo``/``logit_bits`` keys are present in any
request payload, and the adapter gradient flows *only* through ``g_theta``.

Usage
-----
    python scripts/run_experiment.py --dataset strategyqa --setting combined
    python scripts/run_experiment.py --dataset gsm8k --size 0.3b --mode single
    python scripts/run_experiment.py --all-settings --limit-test 100 --dry-run

Environment: ``AZURE_OPENAI_API_KEY`` / ``AZURE_OPENAI_ENDPOINT`` for Azure, or
``HF_TOKEN`` for Mixtral.  Without credentials the black-box client degrades to
a deterministic offline mock (``--allow-mock``, the default) so the whole
pipeline can be smoke-tested end to end.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Make the repository importable when executed as a plain script.
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bbox_adapter.data import dataset_specs  # noqa: E402  (paper split sizes)
from bbox_adapter.data.answer_extraction import (  # noqa: E402
    ANSWER_TYPE_MCQ,
    ANSWER_TYPE_NUMERIC,
    ANSWER_TYPE_TOXIC,
    ANSWER_TYPE_TRUTHFULQA,
    ANSWER_TYPE_YESNO,
    extract_final_answer,
    is_correct,
)
from bbox_adapter.utils import (  # noqa: E402
    derive_component_seeds,
    get_logger,
    load_config,
    set_seed,
)

LOGGER = get_logger("bbox_adapter.scripts.run_experiment")

# ---------------------------------------------------------------------------
# Paper constants (Table 2 / Table 4 / Appendix H.2).
# ---------------------------------------------------------------------------
SETTINGS = ("ground_truth", "ai_feedback", "combined")
DATASETS = ("strategyqa", "gsm8k", "truthfulqa", "scienceqa")
ALL_DATASETS = DATASETS + ("toxigen",)

SETTING_ALIASES = {
    "gt": "ground_truth",
    "groundtruth": "ground_truth",
    "ground-truth": "ground_truth",
    "ai": "ai_feedback",
    "aifeedback": "ai_feedback",
    "ai-feedback": "ai_feedback",
    "gpt4": "ai_feedback",
    "combine": "combined",
    "combined": "combined",
}

DATASET_ALIASES = {
    "strategy-qa": "strategyqa",
    "strategy_qa": "strategyqa",
    "sq": "strategyqa",
    "gsm": "gsm8k",
    "gsm-8k": "gsm8k",
    "math": "gsm8k",
    "truthful": "truthfulqa",
    "truthful_qa": "truthfulqa",
    "tqa": "truthfulqa",
    "sci": "scienceqa",
    "science-qa": "scienceqa",
    "sciq": "scienceqa",
    "toxicity": "toxigen",
}

# Table 2: base gpt-3.5-turbo reference accuracies (CoT prompting).
PAPER_BASE = {"strategyqa": 66.59, "gsm8k": 67.51, "truthfulqa": 77.00, "scienceqa": 72.90}

# Table 2: BBox-Adapter reference numbers per setting.
PAPER_TABLE2 = {
    "ground_truth": {"strategyqa": 71.62, "gsm8k": 73.86, "truthfulqa": 79.70, "scienceqa": 78.53},
    "ai_feedback": {"strategyqa": 69.85, "gsm8k": 73.50, "truthfulqa": 82.10, "scienceqa": 78.30},
    "combined": {"strategyqa": 72.27, "gsm8k": 74.28, "truthfulqa": 83.60, "scienceqa": 79.40},
}

# Table 4: (accuracy, training $, inference $/1k Q) for the base model and variants.
PAPER_TABLE4 = {
    "strategyqa": {
        "gpt-3.5-turbo": (66.59, None, 0.41),
        "azure_sft": (76.86, 153.00, 7.50),
        "bbox_single_step": (69.87, 2.77, 2.20),
        "bbox_full_step": (71.62, 3.48, 5.37),
    },
    "gsm8k": {
        "gpt-3.5-turbo": (67.51, None, 1.22),
        "azure_sft": (69.94, 216.50, 28.30),
        "bbox_single_step": (71.13, 7.54, 3.10),
        "bbox_full_step": (74.28, 11.58, 12.46),
    },
}

# Appendix H.2 defaults (also mirrored in configs/default.yaml).
H2 = {
    "lr": 5e-6,
    "batch_size": 64,
    "max_train_steps": 6000,
    "weight_decay": 0.01,
    "beam_size": 3,
    "max_len": 512,
    "temperature": 1.0,
}

DATASET_METRIC = {
    "strategyqa": "accuracy",
    "gsm8k": "accuracy",
    "truthfulqa": "true_info",
    "scienceqa": "accuracy",
    "toxigen": "toxicity",
}


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def _normalize(value: Optional[str], table: Dict[str, str]) -> Optional[str]:
    if value is None:
        return None
    key = str(value).strip().lower().replace(" ", "_")
    key = table.get(key, key)
    return key


def normalize_setting(value: Optional[str]) -> Optional[str]:
    return _normalize(value, SETTING_ALIASES)


def normalize_dataset(value: Optional[str]) -> Optional[str]:
    return _normalize(value, DATASET_ALIASES)


def call_filtered(fn: Callable, *args: Any, **kwargs: Any) -> Any:
    """Call ``fn`` with only the keyword arguments its signature accepts.

    The reproduction plan fixes the *semantics* of each component but several
    helper signatures admit aliases (``eta`` vs ``lr``, ``T`` vs ``n_iterations``),
    so entry points pass the union and let this adapter filter it down.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):  # builtins / C callables
        return fn(*args, **kwargs)
    params = sig.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(*args, **kwargs)
    allowed = set(params)
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        LOGGER.debug("call_filtered: dropped unsupported kwargs for %s: %s", getattr(fn, "__name__", fn), dropped)
    return fn(*args, **filtered)


def deep_get(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    node: Any = config
    for part in path.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return default
    return node if node is not None else default


def section(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = config.get(name) if isinstance(config, dict) else None
    return dict(value) if isinstance(value, dict) else {}


def safe_div(a: float, b: float) -> Optional[float]:
    return (a / b) if b else None


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def delta_percent(value: Optional[float], base: Optional[float]) -> Optional[float]:
    """Absolute percentage-point difference (paper's Delta(%) convention)."""
    if value is None or base is None:
        return None
    return value - base


@dataclass
class RunData:
    """Prompt-level view of one dataset split, ready for adaptation/eval."""

    dataset: str
    answer_type: str
    metric: str
    questions: List[str] = field(default_factory=list)
    prompts: List[str] = field(default_factory=list)
    golds: List[Any] = field(default_factory=list)
    answer_types: List[Any] = field(default_factory=list)
    choices_list: List[Any] = field(default_factory=list)
    uids: List[str] = field(default_factory=list)
    examples: List[Any] = field(default_factory=list)
    n_shot: int = 0

    def __len__(self) -> int:
        return len(self.questions)

    def subset(self, n: int, *, seed: int = 0) -> "RunData":
        if n is None or n <= 0 or n >= len(self):
            return self
        import random

        idx = list(range(len(self)))
        random.Random(seed).shuffle(idx)
        idx = sorted(idx[:n])
        return RunData(
            dataset=self.dataset,
            answer_type=self.answer_type,
            metric=self.metric,
            questions=[self.questions[i] for i in idx],
            prompts=[self.prompts[i] for i in idx],
            golds=[self.golds[i] for i in idx],
            answer_types=[self.answer_types[i] for i in idx],
            choices_list=[self.choices_list[i] for i in idx],
            uids=[self.uids[i] for i in idx],
            examples=[self.examples[i] for i in idx],
            n_shot=self.n_shot,
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def build_config(dataset: str, args: argparse.Namespace) -> Dict[str, Any]:
    """Load ``configs/default.yaml`` deep-merged with the per-dataset YAML."""
    overrides: Dict[str, Any] = {}
    if args.size:
        overrides.setdefault("adapter", {})["size"] = args.size
        overrides["size"] = args.size
    if args.model:
        overrides.setdefault("blackbox", {})["model"] = args.model
    if args.allow_mock is not None:
        overrides.setdefault("blackbox", {})["allow_mock"] = bool(args.allow_mock)
    if args.loss:
        overrides.setdefault("loss", {})["name"] = args.loss
    if args.alpha is not None:
        overrides.setdefault("loss", {})["alpha"] = float(args.alpha)
    if args.lr is not None:
        overrides.setdefault("training", {})["lr"] = float(args.lr)
    if args.beam_size is not None:
        overrides.setdefault("beam_search", {})["beam_size"] = int(args.beam_size)
    if args.iterations is not None:
        overrides.setdefault("training", {})["n_iterations"] = int(args.iterations)
    if args.train_steps is not None:
        overrides.setdefault("training", {})["max_train_steps"] = int(args.train_steps)

    config = load_config(dataset=dataset, overrides=overrides or None, with_defaults=True)
    # The paper's Appendix H.2 defaults must survive any YAML gap.
    training = config.setdefault("training", {})
    for key, value in H2.items():
        if key in ("beam_size",):
            config.setdefault("beam_search", {}).setdefault(key, value)
        else:
            training.setdefault(key, value)
    config.setdefault("dataset", dataset)
    return config


def training_kwargs(config: Dict[str, Any], dataset: str, setting: str, size: str) -> Dict[str, Any]:
    """Flatten the YAML into OnlineAdaptationConfig keyword arguments."""
    tr = section(config, "training")
    loss = section(config, "loss")
    ad = section(config, "adapter")
    bs = section(config, "beam_search")
    buf = section(config, "buffer")
    return {
        "dataset": dataset,
        "size": size,
        "n_iterations": tr.get("n_iterations", 4),
        "n_candidates": tr.get("n_candidates", 5),
        "k_init": tr.get("k_init", 5),
        "beam_size": bs.get("beam_size", 3),
        "samples_per_beam": bs.get("n_samples", tr.get("samples_per_beam", 2)),
        "max_steps": bs.get("max_steps", 6),
        "alpha": loss.get("alpha", 1e-2),
        "lr": tr.get("lr", 5e-6),
        "batch_size": tr.get("batch_size", 64),
        "max_train_steps": tr.get("max_train_steps", 6000),
        "weight_decay": tr.get("weight_decay", 0.01),
        "betas": tuple(tr.get("betas", (0.9, 0.999))),
        "warmup_steps": tr.get("warmup_steps"),
        "max_grad_norm": tr.get("max_grad_norm", 1.0),
        "schedule": tr.get("schedule", "constant"),
        "sel_mode": setting,
        "loss_name": loss.get("name", "nce"),
        "temperature": bs.get("temperature", H2["temperature"]),
        "max_len": bs.get("max_len", H2["max_len"]),
        "max_length": ad.get("max_length", H2["max_len"]),
        "outcome_supervision": tr.get("outcome_supervision", buf.get("outcome_supervision", True)),
        "use_beam_search": tr.get("use_beam_search", True),
        "seed": tr.get("seed", config.get("seed", 0)),
    }


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def prepare_data(
    dataset: str,
    config: Dict[str, Any],
    *,
    split: str = "test",
    limit: Optional[int] = None,
    seed: int = 0,
    model: Optional[str] = None,
) -> RunData:
    """Load a split and build the Appendix-J prompt for every example."""
    from bbox_adapter.data.loaders import load_split
    from bbox_adapter.llm.prompts import build_prompt, resolve_prompt_spec

    spec = dataset_specs.get_spec(dataset)
    examples = load_split(dataset, split=split, seed=seed)
    if limit:
        examples = examples[:limit]

    data = RunData(dataset=dataset, answer_type=spec.answer_type, metric=spec.metric)
    prompt_spec = resolve_prompt_spec(dataset)
    data.n_shot = (prompt_spec.n_shot if prompt_spec is not None else int(spec.extra.get("shot", 0)))

    for ex in examples:
        choices = getattr(ex, "choices", None)
        try:
            prompt = build_prompt(
                ex.question,
                dataset=dataset,
                choices=choices,
                context=(ex.meta or {}).get("context") if getattr(ex, "meta", None) else None,
                model=model,
            )
        except Exception as exc:  # pragma: no cover - prompt construction is best-effort
            LOGGER.warning("prompt build failed (%s); using raw question", exc)
            prompt = ex.question
        data.questions.append(ex.question)
        data.prompts.append(prompt)
        data.golds.append(ex.answer)
        data.answer_types.append(spec.answer_type)
        data.choices_list.append(choices)
        data.uids.append(ex.uid)
        data.examples.append(ex)
    LOGGER.info("loaded %s/%s: %d examples (metric=%s, shots=%d)", dataset, split, len(data), spec.metric, data.n_shot)
    return data


# ---------------------------------------------------------------------------
# Black-box client & adapter construction
# ---------------------------------------------------------------------------
def build_generator_and_ledger(config: Dict[str, Any], args: argparse.Namespace):
    """Instantiate the text-only black-box client (plus the cost ledger)."""
    from bbox_adapter.llm.blackbox_client import build_generator

    bb = section(config, "blackbox")
    model = bb.get("model", "gpt-3.5-turbo")
    kind = "cot" if args.baseline == "cot" else bb.get("kind", "bbox")

    ledger = None
    try:
        from bbox_adapter.eval.cost import CostLedger

        ledger = call_filtered(CostLedger.from_config, config) if hasattr(CostLedger, "from_config") else None
    except Exception as exc:  # pragma: no cover - cost accounting is optional
        LOGGER.debug("cost ledger unavailable: %s", exc)

    generator = call_filtered(
        build_generator,
        model,
        kind=kind,
        ledger=ledger,
        allow_mock=bool(bb.get("allow_mock", True)),
        max_len=int(bb.get("max_len", H2["max_len"])),
    )
    return generator, ledger


def build_adapter(config: Dict[str, Any], dataset: str, size: str, seed: int = 0):
    """Instantiate the scalar energy adapter g_theta (Section 3.1)."""
    from bbox_adapter.adapter.energy_model import build_energy_model

    ad = section(config, "adapter")
    set_seed(seed)
    adapter = call_filtered(
        build_energy_model,
        dataset=dataset,
        size=size,
        backbone=ad.get("backbone") or config.get("backbone"),
        pooling=ad.get("pooling"),
        head_hidden=ad.get("head_hidden", 0),
        dropout=ad.get("dropout", 0.0),
        max_length=ad.get("max_length", H2["max_len"]),
        output_scale=ad.get("output_scale", 1.0),
        freeze_backbone=ad.get("freeze_backbone", False),
    )
    try:
        summary = adapter.parameter_summary()
        LOGGER.info("adapter backbone=%s params=%s", summary.get("backbone"), summary.get("total_billions"))
    except Exception:  # pragma: no cover
        pass
    return adapter


def build_loss(config: Dict[str, Any], loss_name: str):
    """Ranking-based NCE (Eq. 2/3) or the MLM ablation."""
    from bbox_adapter.losses import get_loss

    loss_cfg = section(config, "loss")
    return call_filtered(
        get_loss,
        loss_name,
        alpha=loss_cfg.get("alpha", 1e-2),
        reduction=loss_cfg.get("reduction", "mean"),
        temperature=loss_cfg.get("temperature", 1.0),
        label_smoothing=loss_cfg.get("label_smoothing", 0.0),
        energy_clamp=loss_cfg.get("energy_clamp"),
        reg_mode=loss_cfg.get("reg_mode", "split"),
        use_regularizer=loss_cfg.get("use_regularizer", True),
    )


# ---------------------------------------------------------------------------
# Adapted inference (Section 3.3, Eq. 4)
# ---------------------------------------------------------------------------
def build_beam_search(adapter, generator, config: Dict[str, Any], ledger=None):
    from bbox_adapter.inference.beam_search import BeamSearchConfig, SentenceBeamSearch

    bs = section(config, "beam_search")
    bs_cfg = call_filtered(
        BeamSearchConfig,
        beam_size=bs.get("beam_size", 3),
        n_samples=bs.get("n_samples", 2),
        max_steps=bs.get("max_steps", 6),
        temperature=bs.get("temperature", H2["temperature"]),
        max_len=bs.get("max_len", H2["max_len"]),
        normalization=bs.get("normalization", "none"),
        length_penalty=bs.get("length_penalty", 0.0),
        deduplicate=bs.get("deduplicate", True),
        stop_when_all_complete=bs.get("stop_when_all_complete", True),
        include_prompt_in_energy=bs.get("include_prompt_in_energy", True),
        seed=config.get("seed", 0),
    )
    return SentenceBeamSearch(adapter, generator, config=bs_cfg, token_counter=ledger)


def generate_adapted(
    data: RunData,
    adapter,
    generator,
    config: Dict[str, Any],
    *,
    mode: str = "full",
    ledger=None,
) -> Tuple[List[str], Dict[str, Any]]:
    """Run adapted inference over a split, returning raw generations + stats."""
    mode = (mode or "full").strip().lower()
    generations: List[str] = []
    stats = {"mode": mode, "n_llm_calls": 0, "n_candidates": 0, "steps_used": []}

    if mode in ("single", "single_step", "single-step", "rank"):
        from bbox_adapter.inference.beam_search import single_step_search

        n = section(config, "beam_search").get("n_candidates", 5) or 5
        for q, prompt, at, ch in zip(data.questions, data.prompts, data.answer_types, data.choices_list):
            result = call_filtered(
                single_step_search,
                q,
                adapter=adapter,
                generator=generator,
                prompt=prompt,
                n=n,
                answer_type=at,
                choices=ch,
                return_result=True,
            )
            generations.append(getattr(result, "best_text", str(result)))
            stats["n_llm_calls"] += getattr(result, "n_llm_calls", 0)
            stats["n_candidates"] += getattr(result, "n_candidates", 0)
        return generations, stats

    searcher = build_beam_search(adapter, generator, config, ledger=ledger)
    for i, (q, prompt, at, ch) in enumerate(
        zip(data.questions, data.prompts, data.answer_types, data.choices_list)
    ):
        result = call_filtered(
            searcher.run, q, prompt=prompt, answer_type=at, choices=ch, return_result=True
        )
        generations.append(getattr(result, "best_text", str(result)))
        stats["n_llm_calls"] += getattr(result, "n_llm_calls", 0)
        stats["n_candidates"] += getattr(result, "n_candidates", 0)
        stats["steps_used"].append(getattr(result, "steps_used", 0))
        if (i + 1) % 25 == 0:
            LOGGER.info("adapted inference %d/%d", i + 1, len(data))
    return generations, stats


def generate_baseline(data: RunData, generator, config: Dict[str, Any]) -> List[str]:
    """CoT prompting of the unadapted black-box LLM (temperature 0.0)."""
    bb = section(config, "blackbox")
    max_len = int(bb.get("max_len", H2["max_len"]))
    texts: List[str] = []
    for i, prompt in enumerate(data.prompts):
        try:
            out = generator.generate(prompt, n=1, temperature=0.0, max_len=max_len)
        except TypeError:
            out = generator.generate(prompt, 1, 0.0, max_len)
        texts.append(out[0] if isinstance(out, (list, tuple)) and out else str(out))
        if (i + 1) % 25 == 0:
            LOGGER.info("baseline generation %d/%d", i + 1, len(data))
    return texts


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metric(generations: Sequence[str], data: RunData, config: Dict[str, Any]) -> float:
    """Dispatch to the paper's metric (Acc / True+Info / Toxic%)."""
    metric = DATASET_METRIC.get(data.dataset, data.metric or "accuracy")
    try:
        from bbox_adapter.eval.metrics import MetricsConfig, evaluate_generations

        cfg = call_filtered(
            MetricsConfig,
            dataset=data.dataset,
            metric=metric,
            answer_type=data.answer_type,
            toxicity_threshold=deep_get(config, "eval.toxicity_threshold", 0.5),
            gpt_judge=deep_get(config, "eval.gpt_judge", False),
        )
        report = call_filtered(
            evaluate_generations,
            list(generations),
            dataset=data.dataset,
            metric=metric,
            golds=data.golds,
            examples=data.examples,
            answer_type=data.answer_type,
            choices_list=data.choices_list,
            config=cfg,
        )
        return float(getattr(report, "value", report))
    except Exception as exc:
        LOGGER.warning("eval.metrics unavailable (%s); using answer_extraction fallback", exc)

    from bbox_adapter.data.answer_extraction import accuracy as _accuracy

    if metric == "true_info":
        from bbox_adapter.data.answer_extraction import true_info_rate

        return float(true_info_rate(list(generations), data.examples))
    return float(
        _accuracy(
            list(generations),
            data.golds,
            data.answer_type,
            choices_list=data.choices_list,
        )
    )


# ---------------------------------------------------------------------------
# Online adaptation (Algorithm 1)
# ---------------------------------------------------------------------------
def run_adaptation(
    adapter,
    generator,
    train: RunData,
    config: Dict[str, Any],
    *,
    setting: str,
    size: str,
    eval_fn: Optional[Callable] = None,
    ledger=None,
) -> Tuple[Any, Dict[str, Any]]:
    """Drive Algorithm 1 and return ``(adapter, history)``.

    The outer loop ``t = 0..T-1`` wraps the inner pass over the training set;
    ``theta`` is updated with AdamW (Eq. 7) on mini-batches of contrastive sets
    (one positive + its negatives), using the ranking-NCE objective of Eq. (2).
    """
    kwargs = training_kwargs(config, train.dataset, setting, size)
    kwargs["device"] = section(config, "training").get("device", "auto")

    loss_name = kwargs.pop("loss_name", section(config, "loss").get("name", "nce"))
    loss = build_loss(config, loss_name)

    try:
        from bbox_adapter.training.online_adaptation import OnlineAdaptationConfig

        cfg = call_filtered(OnlineAdaptationConfig.from_dict, dict(kwargs))
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("OnlineAdaptationConfig unavailable (%s); passing raw kwargs", exc)
        cfg = kwargs

    try:
        from bbox_adapter.training.online_adaptation import run_online_adaptation

        history = call_filtered(
            run_online_adaptation,
            adapter,
            generator,
            train.questions,
            prompts=train.prompts,
            golds=train.golds,
            answer_types=train.answer_types,
            choices_list=train.choices_list,
            uids=train.uids,
            config=cfg,
            eval_fn=eval_fn,
        )
        LOGGER.info("online adaptation finished (%s)", type(history).__name__)
        return adapter, _as_history(history)
    except TypeError as exc:
        LOGGER.debug("run_online_adaptation signature mismatch (%s); driving OnlineAdapter", exc)
    except Exception as exc:
        LOGGER.warning("run_online_adaptation failed (%s); driving OnlineAdapter", exc)

    from bbox_adapter.training.online_adaptation import OnlineAdapter

    runner = call_filtered(OnlineAdapter, adapter, generator, config=cfg, loss=loss)
    history: Dict[str, Any] = {"mode": "online_adapter", "iterations": []}

    try:
        runner.initialize(
            train.questions,
            train.prompts,
            train.golds,
            train.answer_types,
            train.choices_list,
            train.uids,
            kwargs.get("k_init", 5),
        )
    except TypeError:
        runner.initialize(train.questions, prompts=train.prompts, golds=train.golds, uids=train.uids)
    except Exception as exc:  # pragma: no cover
        LOGGER.error("buffer initialization failed: %s", exc)
        raise

    n_iter = int(kwargs.get("n_iterations", 4))
    for t in range(n_iter):
        LOGGER.info("=== outer iteration %d/%d (T=%d) ===", t + 1, n_iter, t)
        try:
            out = call_filtered(
                runner.run,
                train.questions,
                prompts=train.prompts,
                golds=train.golds,
                answer_types=train.answer_types,
                choices_list=train.choices_list,
                uids=train.uids,
                eval_fn=eval_fn,
                t=t,
                m=kwargs.get("n_candidates", 5),
                k=kwargs.get("k_init", 5),
            )
        except TypeError:
            out = runner.run(train.questions) if not eval_fn else None
        history["iterations"].append(out if isinstance(out, dict) else {"iteration": t, "result": str(out)})
    return adapter, history


def _as_history(history: Any) -> Dict[str, Any]:
    if history is None:
        return {}
    if isinstance(history, dict):
        return history
    if hasattr(history, "to_dict"):
        try:
            return history.to_dict()
        except Exception:  # pragma: no cover
            pass
    if hasattr(history, "__dict__"):
        return {k: v for k, v in vars(history).items() if not k.startswith("_")}
    return {"history": str(history)}


# ---------------------------------------------------------------------------
# One full experiment (one dataset x one setting x one seed)
# ---------------------------------------------------------------------------
def run_single(
    dataset: str,
    setting: str,
    args: argparse.Namespace,
    config: Dict[str, Any],
    *,
    seed: int = 0,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "dataset": dataset,
        "setting": setting,
        "mode": args.mode,
        "seed": seed,
        "size": args.size,
        "model": args.model,
        "metric": DATASET_METRIC.get(dataset, "accuracy"),
    }
    set_seed(seed)

    train = prepare_data(
        dataset,
        config,
        split="train",
        limit=args.limit_train,
        seed=seed,
        model=args.model,
    )
    test = prepare_data(
        dataset,
        config,
        split="test",
        limit=args.limit_test,
        seed=seed,
        model=args.model,
    )
    row["n_train"], row["n_test"] = len(train), len(test)
    if not test:
        raise RuntimeError(f"no test examples for {dataset}")

    generator, ledger = build_generator_and_ledger(config, args)

    # --- CoT baseline of the *unadapted* black box (Table 2 first row) ----
    if args.baseline == "cot" or args.eval_base:
        base_gens = generate_baseline(test, generator, config)
        row["base_accuracy"] = compute_metric(base_gens, test, config)
        if args.baseline == "cot":
            row["accuracy"] = row["base_accuracy"]
            row["delta"] = 0.0
            return row
    else:
        row["base_accuracy"] = PAPER_BASE.get(dataset)

    # --- Adapter + Algorithm 1 -------------------------------------------
    adapter = build_adapter(config, dataset, args.size, seed=seed)

    def eval_fn(adapter_, t):  # Figure 3(b) score curve
        if not args.eval_every_iteration:
            return None
        dev = test.subset(args.curve_eval_limit or 32, seed=seed)
        gens, _ = generate_adapted(dev, adapter_, generator, config, mode=args.mode, ledger=ledger)
        return compute_metric(gens, dev, config)

    t0 = time.time()
    phase = None
    if ledger is not None and hasattr(ledger, "phase"):
        try:
            phase = ledger.phase("training")
            phase.__enter__()
        except Exception:  # pragma: no cover
            phase = None
    adapter, history = run_adaptation(
        adapter,
        generator,
        train,
        config,
        setting=setting,
        size=args.size,
        eval_fn=eval_fn if args.eval_every_iteration else None,
        ledger=ledger,
    )
    if phase is not None:
        try:
            phase.__exit__(None, None, None)
        except Exception:  # pragma: no cover
            pass
    row["train_seconds"] = time.time() - t0

    # --- Adapted inference + metric ---------------------------------------
    t1 = time.time()
    phase = None
    if ledger is not None and hasattr(ledger, "phase"):
        try:
            phase = ledger.phase("inference")
            phase.__enter__()
        except Exception:  # pragma: no cover
            phase = None
    generations, gen_stats = generate_adapted(
        test, adapter, generator, config, mode=args.mode, ledger=ledger
    )
    if phase is not None:
        try:
            phase.__exit__(None, None, None)
        except Exception:  # pragma: no cover
            pass
    row["inference_seconds"] = time.time() - t1
    row["accuracy"] = compute_metric(generations, test, config)
    row["delta"] = delta_percent(row["accuracy"], row.get("base_accuracy"))
    row["generation_stats"] = {
        k: (v if not isinstance(v, list) else [len(v), sum(v) / len(v) if v else 0.0])
        for k, v in gen_stats.items()
    }
    row["score_curve"] = _extract_score_curve(history)
    row["history"] = history if args.keep_history else None
    row["paper"] = PAPER_TABLE2.get(setting, {}).get(dataset)
    row["paper_base"] = PAPER_BASE.get(dataset)

    # --- Cost accounting (Table 4) ----------------------------------------
    if ledger is not None:
        row.update(collect_cost(ledger, len(test), len(train)))
    if args.save_adapter:
        path = os.path.join(args.output_dir, f"adapter_{dataset}_{args.size}_{setting}.pt")
        try:
            adapter.save_pretrained(path)
            row["adapter_path"] = path
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("could not save adapter: %s", exc)
    return row


def _extract_score_curve(history: Dict[str, Any]) -> Optional[List[Any]]:
    if not isinstance(history, dict):
        return None
    for key in ("score_curve", "eval_scores", "scores", "accuracy_curve"):
        if key in history:
            return history[key]
    return None


def collect_cost(ledger, n_test: int, n_train: int) -> Dict[str, Any]:
    """Convert the token ledger into the paper's $/1k-question figures."""
    out: Dict[str, Any] = {}
    try:
        out["training_cost"] = float(ledger.training_cost)
        out["inference_cost"] = float(ledger.inference_cost)
        out["total_cost"] = float(ledger.total_cost)
    except Exception as exc:  # pragma: no cover
        LOGGER.debug("cost fields unavailable: %s", exc)
    try:
        out["inference_cost_per_1k"] = float(
            call_filtered(ledger.cost_per_1k_questions, "inference", n_test)
            if hasattr(ledger, "cost_per_1k_questions")
            else 0.0
        )
    except Exception as exc:  # pragma: no cover
        LOGGER.debug("cost_per_1k_questions unavailable: %s", exc)
    try:
        usage = ledger.usage("total")
        out["prompt_tokens"] = int(usage.get("prompt_tokens", 0))
        out["completion_tokens"] = int(usage.get("completion_tokens", 0))
        out["total_tokens"] = int(usage.get("total_tokens", 0))
    except Exception:  # pragma: no cover
        pass
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-seed rows into Table 2 / Table 4 shaped records."""
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in results:
        grouped.setdefault((row["dataset"], row["setting"]), []).append(row)

    table2: Dict[str, Dict[str, Any]] = {}
    for (dataset, setting), rows in sorted(grouped.items()):
        accs = [r["accuracy"] for r in rows if r.get("accuracy") is not None]
        if not accs:
            continue
        mean = sum(accs) / len(accs)
        std = population_std(accs)
        bases = [r["base_accuracy"] for r in rows if r.get("base_accuracy") is not None]
        base = sum(bases) / len(bases) if bases else PAPER_BASE.get(dataset)
        entry = table2.setdefault(dataset, {})
        entry[setting] = {
            "accuracy": mean,
            "std": std,
            "delta": delta_percent(mean, base),
            "base_accuracy": base,
            "n_seeds": len(accs),
            "paper": PAPER_TABLE2.get(setting, {}).get(dataset),
            "paper_delta": delta_percent(
                PAPER_TABLE2.get(setting, {}).get(dataset), PAPER_BASE.get(dataset)
            ),
        }
    return table2


def population_std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


def format_main_table(table2: Dict[str, Dict[str, Any]]) -> str:
    datasets = [d for d in DATASETS if d in table2]
    header = ["Adapter \\ Dataset"] + [f"{d} (Acc/True+Info %)" for d in datasets] + ["avg delta"]
    lines = ["\t".join(header)]
    settings = ["ground_truth", "ai_feedback", "combined"]
    for setting in settings:
        row = [f"BBox-Adapter ({setting})"]
        deltas = []
        for d in datasets:
            entry = table2.get(d, {}).get(setting)
            if not entry:
                row.append("-")
                continue
            row.append(f"{entry['accuracy']:.2f} +- {entry['std']:.2f}")
            if entry.get("delta") is not None:
                deltas.append(entry["delta"])
        row.append(f"{sum(deltas) / len(deltas):.2f}" if deltas else "-")
        lines.append("\t".join(row))
    paper_row = ["paper delta"] + [
        fmt(
            delta_percent(PAPER_TABLE2["combined"].get(d), PAPER_BASE.get(d)),
            2,
        )
        for d in datasets
    ]
    lines.append("\t".join(paper_row))
    return "\n".join(lines)


def compare_to_paper(table2: Dict[str, Dict[str, Any]], tolerance: float = 3.0) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    for dataset, settings in table2.items():
        for setting, entry in settings.items():
            paper = entry.get("paper")
            if paper is None or entry.get("accuracy") is None:
                continue
            diff = entry["accuracy"] - paper
            checks[f"{dataset}/{setting}"] = {
                "reproduced": round(entry["accuracy"], 2),
                "paper": paper,
                "diff": round(diff, 2),
                "within_tolerance": abs(diff) <= tolerance,
            }
    n_ok = sum(1 for v in checks.values() if v["within_tolerance"])
    return {
        "checks": checks,
        "n_within_tolerance": n_ok,
        "n_checked": len(checks),
        "tolerance": tolerance,
    }


def format_cost_table(results: List[Dict[str, Any]]) -> str:
    lines = ["dataset\tvariant\tacc(train $ / inference $ per 1k Q)\tpaper"]
    for row in results:
        if row.get("mode") not in ("single", "full"):
            continue
        variant = "bbox_single_step" if row["mode"] == "single" else "bbox_full_step"
        paper = PAPER_TABLE4.get(row["dataset"], {}).get(variant)
        lines.append(
            "\t".join(
                [
                    row["dataset"],
                    variant,
                    "{:.2f} / {} / {}".format(
                        row.get("accuracy") or float("nan"),
                        fmt(row.get("training_cost"), 2),
                        fmt(row.get("inference_cost_per_1k"), 2),
                    ),
                    str(paper),
                ]
            )
        )
    return "\n".join(lines)


def write_reports(results: List[Dict[str, Any]], table2: Dict[str, Any], args) -> Dict[str, str]:
    os.makedirs(args.output_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    raw_path = os.path.join(args.output_dir, "results_raw.json")
    with open(raw_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, default=str)
    paths["raw"] = raw_path

    summary = {
        "table2": table2,
        "paper_table2": PAPER_TABLE2,
        "paper_base": PAPER_BASE,
        "comparison": compare_to_paper(table2, tolerance=args.tolerance),
    }
    sum_path = os.path.join(args.output_dir, "summary.json")
    with open(sum_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    paths["summary"] = sum_path

    md_path = os.path.join(args.output_dir, "REPORT.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("# BBox-Adapter main results\n\n")
        fh.write("## Table 2 reproduction (gpt-3.5-turbo, CoT prompts)\n\n```\n")
        fh.write(format_main_table(table2))
        fh.write("\n```\n\n## Table 4 cost snapshot\n\n```\n")
        fh.write(format_cost_table(results))
        fh.write("\n```\n\n## Comparison to paper\n\n```\n")
        fh.write(json.dumps(summary["comparison"], indent=2))
        fh.write("\n```\n")
    paths["markdown"] = md_path

    for key in ("training_curves", "score_curves"):
        curves = {r["dataset"]: r.get("score_curve") for r in results if r.get("score_curve")}
        if curves:
            curve_path = os.path.join(args.output_dir, f"{key}.json")
            with open(curve_path, "w", encoding="utf-8") as fh:
                json.dump(curves, fh, indent=2, default=str)
            paths[key] = curve_path
    return paths


# ---------------------------------------------------------------------------
# Sweeps (Figure 3)
# ---------------------------------------------------------------------------
def sweep_beams(args: argparse.Namespace, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Figure 3(a): beam sizes k = 1, 3, 5 (average gain +2.41%)."""
    rows: List[Dict[str, Any]] = []
    for k in [int(x) for x in str(args.sweep_beams).split(",") if x.strip()]:
        cfg = json.loads(json.dumps(config))
        cfg.setdefault("beam_search", {})["beam_size"] = k
        LOGGER.info("--- beam sweep k=%d ---", k)
        row = run_single(args.dataset, args.setting, args, cfg, seed=args.seed)
        row["beam_size"] = k
        rows.append(row)
    accs = [r["accuracy"] for r in rows if r.get("accuracy") is not None]
    spread = (max(accs) - min(accs)) if accs else None
    LOGGER.info("beam sweep spread over k=%s: %s", args.sweep_beams, fmt(spread, 2))
    return rows


def sweep_iterations(args: argparse.Namespace, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Figure 3(b): un-adapted (T=0) is *worse* than the base model."""
    rows: List[Dict[str, Any]] = []
    # T = 0: the randomly-initialized adapter must not help (Section 4.6).
    zero_cfg = json.loads(json.dumps(config))
    zero_cfg.setdefault("training", {})["n_iterations"] = 0
    zero_cfg.setdefault("training", {})["max_train_steps"] = 0
    LOGGER.info("--- iteration sweep T=0 (untrained adapter) ---")
    row0 = run_single(args.dataset, args.setting, args, zero_cfg, seed=args.seed)
    row0["T"] = 0
    rows.append(row0)
    for t in range(1, int(args.iteration_max) + 1):
        cfg = json.loads(json.dumps(config))
        cfg.setdefault("training", {})["n_iterations"] = t
        LOGGER.info("--- iteration sweep T=%d ---", t)
        row = run_single(args.dataset, args.setting, args, cfg, seed=args.seed)
        row["T"] = t
        rows.append(row)
    base = row0.get("base_accuracy")
    if base is not None and row0.get("accuracy") is not None:
        LOGGER.info(
            "T=0 check: adapted=%.2f base=%.2f (must be lower) -> %s",
            row0["accuracy"],
            base,
            "OK" if row0["accuracy"] <= base else "UNEXPECTED",
        )
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BBox-Adapter main experiment driver (Table 2 / Table 4 / Figure 3).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", default="strategyqa", help="strategyqa|gsm8k|truthfulqa|scienceqa|toxigen")
    p.add_argument("--datasets", default=None, help="comma separated list of datasets (overrides --dataset)")
    p.add_argument("--setting", default="ground_truth", choices=list(SETTINGS) + ["gt", "ai", "groundtruth"])
    p.add_argument("--all-settings", action="store_true", help="run ground_truth, ai_feedback and combined")
    p.add_argument("--size", default="0.1b", help="adapter size: 0.1b or 0.3b (Section 4.1)")
    p.add_argument("--mode", default="full", choices=["full", "single"], help="Table 4 inference variant")
    p.add_argument("--model", default=None, help="black-box model (default: gpt-3.5-turbo)")
    p.add_argument("--baseline", default=None, choices=[None, "cot", "sft", "azure_sft", "sft_lora"], help="run only a baseline")
    p.add_argument("--loss", default=None, choices=[None, "nce", "mlm"], help="NCE (default) or the MLM ablation")
    p.add_argument("--alpha", type=float, default=None, help="Eq.(3) regularizer coefficient (default 1e-2)")
    p.add_argument("--lr", type=float, default=None, help="default 5e-6 (Appendix H.2)")
    p.add_argument("--beam-size", type=int, default=None, help="default 3")
    p.add_argument("--iterations", type=int, default=None, help="outer loop T (default 4)")
    p.add_argument("--train-steps", type=int, default=None, help="total mini-batch steps (default 6000)")
    p.add_argument("--limit-train", type=int, default=None, help="debug: cap training questions")
    p.add_argument("--limit-test", type=int, default=None, help="debug: cap test questions")
    p.add_argument("--num-seeds", type=int, default=1, help="repeat runs for Table 10 standard deviations")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default=None, help="default runs/<dataset>")
    p.add_argument("--allow-mock", dest="allow_mock", action="store_true", default=None)
    p.add_argument("--no-allow-mock", dest="allow_mock", action="store_false")
    p.add_argument("--eval-base", action="store_true", default=True, help="evaluate the CoT base model too")
    p.add_argument("--no-eval-base", dest="eval_base", action="store_false")
    p.add_argument("--eval-every-iteration", action="store_true", help="record the Fig. 3(b) score curve")
    p.add_argument("--curve-eval-limit", type=int, default=32)
    p.add_argument("--sweep-beams", default=None, help="e.g. '1,3,5' -> Figure 3(a)")
    p.add_argument("--sweep-iterations", action="store_true", help="T=0..N -> Figure 3(b)")
    p.add_argument("--iteration-max", type=int, default=4)
    p.add_argument("--keep-history", action="store_true")
    p.add_argument("--save-adapter", action="store_true")
    p.add_argument("--tolerance", type=float, default=3.0, help="accuracy tolerance vs the paper (percentage points)")
    p.add_argument("--dry-run", action="store_true", help="tiny offline smoke run (mock client, small caps)")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.dry_run:
        args.limit_train = args.limit_train or 8
        args.limit_test = args.limit_test or 8
        args.train_steps = args.train_steps or 4
        args.iterations = args.iterations if args.iterations is not None else 1
        if args.allow_mock is None:
            args.allow_mock = True

    datasets = (
        [d.strip() for d in str(args.datasets).split(",") if d.strip()]
        if args.datasets
        else [args.dataset]
    )
    datasets = [normalize_dataset(d) or d for d in datasets]
    settings = list(SETTINGS) if args.all_settings else [normalize_setting(args.setting) or args.setting]

    all_results: List[Dict[str, Any]] = []
    for dataset in datasets:
        if args.output_dir:
            out_dir = args.output_dir
        else:
            out_dir = os.path.join("runs", f"{dataset}_{args.size}_{args.mode}")
        if len(datasets) > 1:
            out_dir = os.path.join(out_dir, dataset)
        args.output_dir = out_dir

        components = derive_component_seeds(args.seed)
        LOGGER.info(
            "dataset=%s size=%s mode=%s settings=%s seeds=%d component seeds=%s",
            dataset,
            args.size,
            args.mode,
            settings,
            args.num_seeds,
            components,
        )

        config = build_config(dataset, args)

        if args.sweep_beams:
            all_results.extend(sweep_beams(args, config))
            continue
        if args.sweep_iterations:
            all_results.extend(sweep_iterations(args, config))
            continue

        for setting in settings:
            for k in range(max(1, args.num_seeds)):
                seed = args.seed + k
                LOGGER.info("### run dataset=%s setting=%s seed=%d ###", dataset, setting, seed)
                try:
                    row = run_single(dataset, setting, args, config, seed=seed)
                except Exception as exc:  # keep going: partial tables are still useful
                    LOGGER.error("run failed (dataset=%s setting=%s seed=%d): %s", dataset, setting, seed, exc)
                    if args.dry_run:
                        raise
                    continue
                row["setting"] = setting
                all_results.append(row)
                LOGGER.info(
                    "%s/%s/seed%d: acc=%.2f delta=%s (paper %s)",
                    dataset,
                    setting,
                    seed,
                    row.get("accuracy") or float("nan"),
                    fmt(row.get("delta"), 2),
                    row.get("paper"),
                )

    if not all_results:
        LOGGER.error("no results produced")
        return 1

    table2 = summarize(all_results)
    paths = write_reports(all_results, table2, args)

    print("\n=== Table 2 reproduction ===")
    print(format_main_table(table2))
    print("\n=== Table 4 cost snapshot ===")
    print(format_cost_table(all_results))
    print("\n=== comparison to paper (tolerance %.1f pp) ===" % args.tolerance)
    print(json.dumps(compare_to_paper(table2, tolerance=args.tolerance), indent=2))
    print("\nartifacts:")
    for key, path in paths.items():
        print(f"  {key}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
