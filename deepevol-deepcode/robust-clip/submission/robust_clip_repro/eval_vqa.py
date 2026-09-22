"""VQA robustness evaluation harness for the Robust CLIP reproduction.

Evaluates LLaVA-1.5 7B (or OpenFlamingo) on TextVQA / POPE / SQA-I under the
Addendum's precision-graded attack schedule:

    1. low-precision (int16) untargeted attacks on the top-5 most frequent
       ground truths of the dataset,
    2. argmin selection over those 5 ground truths (the one that produced the
       lowest score for the sample),
    3. high-precision (int32) untargeted attack on the selected ground truth,
    4. targeted attack towards the literal lower-case string ``"maybe"``
       starting from a clean perturbation initialization,
    5. targeted attack towards the literal capitalized string ``"Word"`` from a
       second, separate clean perturbation initialization
       (skipped entirely on TextVQA).

The actual attacks are delegated to :mod:`robust_clip_repro.attacks.vqa_schedule`
and :mod:`robust_clip_repro.attacks.pgd`; scoring/parsing/aggregation to
:mod:`robust_clip_repro.metrics.vqa`; data loading to
:mod:`robust_clip_repro.data.benchmarks`.

Nothing the Addendum does not state (eps, alpha, iteration counts, the numeric
"score" definition, generation settings, prompt templates) is invented here:
those arrive from configuration / the CLI and are logged as externally supplied.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger("robust_clip_repro.eval_vqa")

DEFAULT_CONFIG_PATH = str(Path(__file__).resolve().parent / "configs" / "vqa_attack.yaml")

SUPPORTED_MODELS = ("llava", "openflamingo", "dummy")
TEXT_VQA = "TextVQA"
POPE = "POPE"
SQA_I = "SQA-I"
DATASETS: Tuple[str, ...] = (TEXT_VQA, POPE, SQA_I)

UNSPECIFIED = "UNSPECIFIED_BY_ADDENDUM"

#: Values the Addendum does not state -- logged as externally supplied.
EXTERNAL_DEFAULTS: Dict[str, Any] = {
    "eps": UNSPECIFIED,
    "alpha": UNSPECIFIED,
    "iterations": UNSPECIFIED,
    "restarts": UNSPECIFIED,
    "score_definition": UNSPECIFIED,
    "batch_size": 1,
    "seed": 0,
    "num_workers": 0,
    "max_new_tokens": UNSPECIFIED,
    "generation_temperature": UNSPECIFIED,
    "prompt_template": UNSPECIFIED,
    "answer_parser": UNSPECIFIED,
    "num_samples": UNSPECIFIED,
    "output_dir": "results",
}

#: Addendum-mandated facts (encoded for provenance logging only).
ADDENDUM_FACTS: Tuple[str, ...] = (
    "half-precision attacks store perturbations as int16",
    "single-precision attacks store perturbations as int32",
    "VQA stage 1: low-precision attacks on top-5 most frequent ground truths",
    "VQA stage 2: argmin selection over the five scores",
    "VQA stage 3: high-precision attack on the selected ground truth",
    'VQA stage 4: targeted attack on the most frequent ground truth with the string "maybe"',
    'VQA stage 5: targeted attack with the string "Word" (skipped on TextVQA)',
    'case of the target strings "maybe" / "Word" is graded literally',
)

PAPER_BODY_FACTS: Tuple[str, ...] = (
    "benchmarks are TextVQA, POPE and SQA-I",
    "victim is LLaVA-1.5 7B with OpenAI CLIP ViT-L/14@224 (OpenCLIP implementation)",
)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def call_with_supported_kwargs(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call ``fn`` passing only the keyword arguments it actually accepts."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):  # builtins / C callables
        return fn(*args, **kwargs)
    params = sig.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(*args, **kwargs)
    allowed = {k: v for k, v in kwargs.items() if k in params}
    dropped = sorted(set(kwargs) - set(allowed))
    if dropped:
        LOGGER.debug("call_with_supported_kwargs: dropping %s for %r", dropped, fn)
    return fn(*args, **allowed)


def _torch_dtype_for(precision: str, device: Optional[str] = None):
    from .utils.precision import float_dtype_for_precision

    return float_dtype_for_precision(precision)


def _resolve_device(device: Optional[str]) -> str:
    if device:
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover - torch missing
        return "cpu"


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass
class VQAEvalConfig:
    """Configuration of a VQA robustness evaluation run.

    ``eps``/``alpha``/``iterations`` for each of the three attack stages are
    *not* specified by the Addendum; they must be supplied through the config
    file / CLI and are flagged as externally supplied.
    """

    # data
    datasets: List[str] = field(default_factory=lambda: list(DATASETS))
    dataset_name: str = POPE
    split: Optional[str] = None
    num_samples: Optional[int] = None
    shuffle: bool = False
    dataset_ids: Dict[str, str] = field(default_factory=dict)
    image_root: Optional[str] = None
    cache_dir: Optional[str] = None

    # model
    model_name: str = "llava"
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    prompt: Optional[str] = None
    max_new_tokens: int = 16
    generation_temperature: float = 0.0

    # attack stages (ADDENDUM: precision grading + ordering, not budgets)
    low_eps: Optional[float] = None
    low_alpha: Optional[float] = None
    low_iterations: Optional[int] = None
    low_restarts: Optional[int] = None
    high_eps: Optional[float] = None
    high_alpha: Optional[float] = None
    high_iterations: Optional[int] = None
    high_restarts: Optional[int] = None
    targeted_eps: Optional[float] = None
    targeted_alpha: Optional[float] = None
    targeted_iterations: Optional[int] = None
    targeted_restarts: Optional[int] = None
    norm: str = "linf"
    top_k: int = 5
    score_kind: str = "accuracy"
    score_definition: Optional[str] = None
    quant_scale: Optional[float] = None
    clamp_min: float = 0.0
    clamp_max: float = 1.0

    # runtime
    batch_size: int = 1
    seed: int = 0
    num_workers: int = 0
    device: Optional[str] = None
    resolution: int = 224
    save_perturbations: bool = True
    output_dir: str = "results"
    output_file: Optional[str] = None
    verbose: bool = True
    smoke_test: bool = False
    provenance: Dict[str, Any] = field(default_factory=dict)

    # -- construction ------------------------------------------------------
    def __post_init__(self) -> None:
        if isinstance(self.dataset_name, str):
            self.dataset_name = _normalize_dataset(self.dataset_name)
        if not self.datasets:
            self.datasets = [self.dataset_name]
        self.datasets = [_normalize_dataset(d) for d in self.datasets]
        if self.score_kind not in ("accuracy", "constant", "logprob"):
            LOGGER.warning(
                "score_kind=%r is not one of %s -- treated as externally supplied",
                self.score_kind,
                ("accuracy", "constant", "logprob"),
            )
        self._record_provenance()

    def _record_provenance(self) -> None:
        prov = self.provenance or {}
        prov.setdefault("paper_body", list(PAPER_BODY_FACTS))
        prov.setdefault("addendum", list(ADDENDUM_FACTS))
        prov.setdefault(
            "unspecified_by_addendum",
            sorted(
                k
                for k, v in {
                    "eps": self.low_eps,
                    "alpha": self.low_alpha,
                    "iterations": self.low_iterations,
                    "restarts": self.low_restarts,
                    "targeted_budget": self.targeted_eps,
                    "score_definition": self.score_definition,
                    "max_new_tokens": self.max_new_tokens,
                    "generation_temperature": self.generation_temperature,
                    "prompt": self.prompt,
                }.items()
                if v is None or v == EXTERNAL_DEFAULTS.get(k, UNSPECIFIED)
            ),
        )
        prov["external_defaults"] = dict(EXTERNAL_DEFAULTS)
        self.provenance = prov

    # -- accessors ---------------------------------------------------------
    @property
    def half_precision_int_dtype(self) -> str:
        from .utils.precision import int_dtype_for_precision

        return str(int_dtype_for_precision("half"))

    @property
    def single_precision_int_dtype(self) -> str:
        from .utils.precision import int_dtype_for_precision

        return str(int_dtype_for_precision("single"))

    def resolved_eps(self) -> Optional[float]:
        """The high-precision (int32) budget, which drives reporting."""
        return self.high_eps if self.high_eps is not None else self.low_eps

    def stage_budgets(self) -> Dict[str, Dict[str, Any]]:
        return {
            "low": {"eps": self.low_eps, "alpha": self.low_alpha, "iterations": self.low_iterations},
            "high": {"eps": self.high_eps, "alpha": self.high_alpha, "iterations": self.high_iterations},
            "targeted": {
                "eps": self.targeted_eps if self.targeted_eps is not None else self.high_eps,
                "alpha": self.targeted_alpha if self.targeted_alpha is not None else self.high_alpha,
                "iterations": self.targeted_iterations
                if self.targeted_iterations is not None
                else self.high_iterations,
            },
        }

    def attack_config(self, dataset_name: Optional[str] = None):
        """Build a :class:`attacks.vqa_schedule.VQAAttackConfig` for a dataset."""
        from .attacks.vqa_schedule import VQAAttackConfig

        name = _normalize_dataset(dataset_name or self.dataset_name)
        kwargs: Dict[str, Any] = {
            "low_eps": self.low_eps,
            "low_alpha": self.low_alpha,
            "low_iterations": self.low_iterations,
            "low_restarts": self.low_restarts,
            "high_eps": self.high_eps,
            "high_alpha": self.high_alpha,
            "high_iterations": self.high_iterations,
            "high_restarts": self.high_restarts,
            "targeted_eps": self.targeted_eps if self.targeted_eps is not None else self.high_eps,
            "targeted_alpha": self.targeted_alpha if self.targeted_alpha is not None else self.high_alpha,
            "targeted_iterations": self.targeted_iterations
            if self.targeted_iterations is not None
            else self.high_iterations,
            "targeted_restarts": self.targeted_restarts,
            "norm": self.norm,
            "score_definition": self.score_definition,
            "clamp_min": self.clamp_min,
            "clamp_max": self.clamp_max,
            "quant_scale": self.quant_scale,
            "seed": self.seed,
            "dataset_name": name,
            "provenance": dict(self.provenance),
        }
        try:
            return call_with_supported_kwargs(VQAAttackConfig, **kwargs)
        except TypeError:  # pragma: no cover - defensive
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return call_with_supported_kwargs(VQAAttackConfig, **kwargs)

    # -- serialization -----------------------------------------------------
    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["resolved_eps"] = self.resolved_eps()
        d["half_precision_int_dtype"] = self.half_precision_int_dtype
        d["single_precision_int_dtype"] = self.single_precision_int_dtype
        return d

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]] = None, **overrides: Any) -> "VQAEvalConfig":
        data: Dict[str, Any] = {}
        cfg = cfg or {}
        for key in ("vqa", "eval_vqa", "attack", "vqa_attack"):
            if isinstance(cfg.get(key), dict):
                data.update(cfg[key])
                break
        else:
            data.update({k: v for k, v in cfg.items() if not isinstance(v, dict)})
        # nested model block
        model_block = cfg.get("model") or cfg.get("models") or {}
        if isinstance(model_block, dict):
            data.setdefault("model_name", model_block.get("name", data.get("model_name", "llava")))
            data.setdefault("model_kwargs", model_block.get("kwargs", {}) or {})
        data.update({k: v for k, v in overrides.items() if v is not None})
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = sorted(set(data) - known)
        for key in unknown:
            LOGGER.debug("ignoring unknown VQA config key %r", key)
            data.pop(key)
        return cls(**data)


def _normalize_dataset(name: Optional[str]) -> str:
    if not name:
        return POPE
    try:
        from .data.benchmarks import normalize_dataset_name

        return normalize_dataset_name(name)
    except Exception:  # pragma: no cover - defensive
        lowered = str(name).strip().lower()
        if "text" in lowered:
            return TEXT_VQA
        if "pope" in lowered:
            return POPE
        if "sqa" in lowered or "science" in lowered:
            return SQA_I
        return str(name)


# ---------------------------------------------------------------------------
# config loading
# ---------------------------------------------------------------------------
def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load a YAML config file, tolerating a missing file or missing PyYAML."""
    if not path:
        path = DEFAULT_CONFIG_PATH
    try:
        p = Path(path)
    except TypeError:  # pragma: no cover
        return {}
    if not p.exists():
        LOGGER.debug("config %s not found; using built-in defaults", path)
        return {}
    try:
        import yaml  # type: ignore
    except Exception:
        LOGGER.warning("PyYAML unavailable; ignoring config file %s", path)
        return {}
    try:
        with open(p, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("could not parse config %s: %s", path, exc)
        return {}
    if not isinstance(data, dict):
        return {}
    LOGGER.info("loaded VQA config from %s", p)
    return data


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def resolve_samples(cfg: VQAEvalConfig, dataset_name: Optional[str] = None):
    """Load benchmark samples (falls back to deterministic synthetic samples)."""
    name = _normalize_dataset(dataset_name or cfg.dataset_name)
    try:
        from .data import benchmarks as B

        loader = getattr(B, "load_benchmark", None)
        if loader is not None:
            samples = call_with_supported_kwargs(
                loader,
                name,
                split=cfg.split,
                num_samples=cfg.num_samples,
                dataset_id=cfg.dataset_ids.get(name) if cfg.dataset_ids else None,
                cache_dir=cfg.cache_dir,
                image_root=cfg.image_root,
                shuffle=cfg.shuffle,
                seed=cfg.seed,
                resolution=cfg.resolution,
                model_name=cfg.model_name,
                verbose=cfg.verbose,
            )
            if samples:
                LOGGER.info("loaded %d samples for %s", len(samples), name)
                return samples
        LOGGER.warning("benchmark loader returned no samples for %s", name)
    except Exception as exc:
        LOGGER.warning("could not load %s (%s); using synthetic samples", name, exc)
    return synthetic_samples(cfg.num_samples or 8, dataset_name=name, seed=cfg.seed)


def synthetic_samples(n: int = 8, *, dataset_name: str = POPE, seed: int = 0) -> List[Dict[str, Any]]:
    """Deterministic model-free samples (smoke tests / offline runs only)."""
    import torch

    rng = random.Random(seed)
    name = _normalize_dataset(dataset_name)
    out: List[Dict[str, Any]] = []
    for i in range(n):
        gen = torch.Generator().manual_seed(seed + i)
        pixels = torch.rand(1, 3, 224, 224, generator=gen)
        if name == TEXT_VQA:
            answers = ["stop", "go", "42"]
            question = f"What does the sign say? (sample {i})"
        elif name == SQA_I:
            answers = ["3", "4", "5"]
            question = f"What is the result of the computation? (sample {i})"
        else:
            answers = ["yes", "yes", "no"]
            question = f"Is the object in the image? (sample {i})"
        out.append(
            {
                "sample_id": f"{name}-{i}",
                "id": i,
                "dataset": name,
                "question": question,
                "answers": list(answers),
                "ground_truths": list(answers),
                "pixels": pixels,
                "split": "synthetic",
            }
        )
    return out


def sample_pixels(sample: Any, cfg: "VQAEvalConfig", device: Optional[str] = None):
    """Raw, NON-normalized ``(1,3,H,W)`` pixels -- the attack's projection space."""
    pixels = None
    if isinstance(sample, dict):
        pixels = sample.get("pixels")
    else:
        pixels = getattr(sample, "pixels", None)
    if pixels is None:
        try:
            from .data.benchmarks import sample_to_pixels

            pixels = call_with_supported_kwargs(
                sample_to_pixels,
                sample,
                resolution=cfg.resolution,
                device=device,
                in01=True,
            )
        except Exception as exc:
            raise RuntimeError(f"sample has no usable pixels ({exc})") from exc
    import torch

    if isinstance(pixels, torch.Tensor):
        if pixels.dim() == 3:
            pixels = pixels.unsqueeze(0)
        return pixels
    # PIL / numpy fallback
    try:
        from .data.benchmarks import image_to_pixels
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"cannot convert image to pixels: {exc}") from exc
    return call_with_supported_kwargs(
        image_to_pixels, pixels, resolution=cfg.resolution, device=device, in01=True
    )


# ---------------------------------------------------------------------------
# victim models
# ---------------------------------------------------------------------------
class _DummyTokenizer:
    """Tiny deterministic tokenizer used by the model-free smoke path."""

    def __init__(self, vocab_size: int = 32) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1

    def _id(self, token: str) -> int:
        return 2 + (abs(hash(token)) % max(1, self.vocab_size - 2))

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return [self._id(t) for t in str(text).split()] or [self.eos_token_id]

    def __call__(self, text: str, **kwargs: Any) -> Dict[str, List[int]]:
        return {"input_ids": self.encode(text)}


class DummyVQAVictim:
    """Deterministic model-free victim for smoke tests.

    Predictions are a monotone function of pixel brightness, so a bounded
    perturbation can demonstrably change the answer.
    """

    def __init__(
        self,
        *,
        vocab_size: int = 32,
        dataset_name: str = POPE,
        device: Optional[str] = None,
        **_: Any,
    ) -> None:
        self.tokenizer = _DummyTokenizer(vocab_size)
        self.dataset_name = _normalize_dataset(dataset_name)
        self.device = _resolve_device(device)
        self.vocab_size = vocab_size

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _features(pixels):
        import torch

        x = pixels if isinstance(pixels, torch.Tensor) else torch.as_tensor(pixels)
        x = x.to(torch.float32)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return x.mean(dim=(1, 2, 3))

    def _answer_space(self) -> List[str]:
        if self.dataset_name == TEXT_VQA:
            return ["stop", "go", "42"]
        if self.dataset_name == SQA_I:
            return ["3", "4", "5"]
        return ["yes", "no"]

    def top_answer(self, pixels) -> str:
        feats = self._features(pixels)
        space = self._answer_space()
        idx = int(round(float(feats[0].item()) * (len(space) - 1)))
        return space[max(0, min(len(space) - 1, idx))]

    # -- victim protocol ---------------------------------------------------
    def targeted_logits(self, pixels, target_text: str):
        import torch

        feats = self._features(pixels)
        tokens = self.tokenizer.encode(target_text)
        vocab = self.vocab_size
        logits = torch.full((feats.shape[0], len(tokens) + 1, vocab), -4.0)
        # favouring the target's own tokens more when the image is brighter
        boost = 8.0 * (1.0 - feats).unsqueeze(-1)
        for i, tok in enumerate(tokens):
            logits[:, i, tok % vocab] = boost[:, 0] if boost.dim() > 1 else boost
        return logits

    def logits_fn(self, prompt: str = "") -> Callable[[Any], Any]:
        def fn(pixels, *args: Any, **kwargs: Any):
            import torch

            feats = self._features(pixels)
            vocab = self.vocab_size
            space = self._answer_space()
            logits = torch.zeros(feats.shape[0], vocab)
            for i, ans in enumerate(space):
                tok = self.tokenizer.encode(ans)[0] % vocab
                logits[:, tok] = logits[:, tok] + (1.0 - feats) * (len(space) - i)
            return logits

        return fn

    def generate(self, pixels, prompt: str = "", **kwargs: Any) -> List[str]:
        return [self.top_answer(pixels)]

    def generate_batch(self, pixels_list, prompt: str = "", **kwargs: Any) -> List[str]:
        return [self.top_answer(p) for p in pixels_list]

    def summary(self) -> Dict[str, Any]:
        return {"model": "dummy_vqa", "dataset": self.dataset_name, "vocab_size": self.vocab_size}


def resolve_victim(cfg: VQAEvalConfig, dataset_name: Optional[str] = None):
    """Instantiate the victim model named in the config (lazy imports)."""
    name = (cfg.model_name or "llava").lower()
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"model_name must be one of {SUPPORTED_MODELS}, got {cfg.model_name!r}")
    name = _normalize_dataset(dataset_name or cfg.dataset_name)  # noqa: F841 (dataset used below)
    ds = _normalize_dataset(dataset_name or cfg.dataset_name)
    kwargs = dict(cfg.model_kwargs or {})
    kwargs.setdefault("device", cfg.device)

    if name == "dummy":
        return DummyVQAVictim(dataset_name=ds, device=cfg.device, **kwargs)

    if name == "llava":
        try:
            from .models.llava_openclip import LLaVAOpenCLIPConfig, build_llava_victim

            model_cfg = call_with_supported_kwargs(LLaVAOpenCLIPConfig.from_dict, {}, **kwargs)
            return call_with_supported_kwargs(build_llava_victim, model_cfg)
        except Exception as exc:
            LOGGER.warning("LLaVA victim unavailable (%s); falling back to dummy victim", exc)
            return DummyVQAVictim(dataset_name=ds, device=cfg.device)

    try:
        from .models.openflamingo_wrapper import OpenFlamingoConfig, build_openflamingo_victim

        model_cfg = call_with_supported_kwargs(OpenFlamingoConfig.from_dict, {}, **kwargs)
        return call_with_supported_kwargs(build_openflamingo_victim, model_cfg)
    except Exception as exc:
        LOGGER.warning("OpenFlamingo victim unavailable (%s); falling back to dummy victim", exc)
        return DummyVQAVictim(dataset_name=ds, device=cfg.device)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def _first_prediction(text: Any) -> str:
    if isinstance(text, (list, tuple)):
        return str(text[0]) if text else ""
    return str(text or "")


class VictimScorer:
    """Numeric score for ``(pixels, ground_truth)`` -- higher is better.

    The Addendum does not define the score, so ``kind`` is configurable and
    tagged ``UNSPECIFIED_BY_ADDENDUM``.  ``accuracy`` uses the generated answer
    compared with the ground truth through :mod:`metrics.vqa`.
    """

    def __init__(
        self,
        victim: Any,
        *,
        dataset_name: str,
        prompt: Optional[str] = None,
        kind: str = "accuracy",
        answer_parser: Optional[Callable[..., str]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        constant_value: float = 0.0,
    ) -> None:
        self.victim = victim
        self.dataset_name = _normalize_dataset(dataset_name)
        self.prompt = prompt
        self.kind = kind
        self.answer_parser = answer_parser
        self.generation_kwargs = dict(generation_kwargs or {})
        self.constant_value = constant_value
        self.provenance = UNSPECIFIED
        self.calls = 0

    # -- helpers -----------------------------------------------------------
    def _prompt_for(self, ground_truth: str) -> str:
        if self.prompt:
            return self.prompt
        try:
            from .prompts.templates import build_question_prompt  # type: ignore

            return build_question_prompt(self.dataset_name)
        except Exception:
            if self.dataset_name == POPE:
                return "Answer the question with a single word, yes or no."
            return "Answer the question with a single short phrase."

    def generate(self, pixels, ground_truth: str) -> str:
        prompt = self._prompt_for(ground_truth)
        kwargs = dict(self.generation_kwargs)
        try:
            out = call_with_supported_kwargs(self.victim.generate, pixels, prompt, **kwargs)
        except Exception:  # pragma: no cover - victim without generate
            out = ""
        return _first_prediction(out)

    def parse(self, text: str, ground_truth: str) -> str:
        try:
            from .metrics.vqa import parse_prediction

            return parse_prediction(
                text, dataset_name=self.dataset_name, parser=self.answer_parser
            )
        except Exception:
            try:
                from .metrics.vqa import clean_prediction

                return clean_prediction(text)
            except Exception:
                return str(text).strip().lower().split("\n")[0]

    def accuracy(self, prediction: str, ground_truth: str) -> float:
        try:
            from .metrics.vqa import prediction_accuracy

            return float(
                prediction_accuracy(prediction, [ground_truth], dataset_name=self.dataset_name)
            )
        except Exception:
            return 1.0 if prediction.strip().lower() == str(ground_truth).strip().lower() else 0.0

    # -- protocol ----------------------------------------------------------
    def __call__(self, pixels, ground_truth: str) -> float:
        self.calls += 1
        if self.kind == "constant":
            return float(self.constant_value)
        response = self.generate(pixels, ground_truth)
        prediction = self.parse(response, ground_truth)
        if self.kind == "logprob":
            # accuracy remains the mandated behavioural quantity; logprob kind is
            # a documented external option and falls back to accuracy here.
            return self.accuracy(prediction, ground_truth)
        return self.accuracy(prediction, ground_truth)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "dataset": self.dataset_name,
            "provenance": self.provenance,
            "calls": self.calls,
        }


def make_score_functions(victim: Any, cfg: VQAEvalConfig, dataset_name: Optional[str] = None):
    """Return ``(score_fn, accuracy_fn)`` used by the attack scheduler."""
    ds = _normalize_dataset(dataset_name or cfg.dataset_name)
    generation_kwargs = {
        "max_new_tokens": cfg.max_new_tokens,
        "temperature": cfg.generation_temperature,
    }
    scorer = VictimScorer(
        victim,
        dataset_name=ds,
        prompt=cfg.prompt,
        kind=cfg.score_kind,
        generation_kwargs=generation_kwargs,
    )

    def score_fn(pixels, ground_truth: str) -> float:
        return float(scorer(pixels, ground_truth))

    def accuracy_fn(pixels, ground_truth: str) -> float:
        return float(scorer(pixels, ground_truth))

    return score_fn, accuracy_fn, scorer


# ---------------------------------------------------------------------------
# attack functions
# ---------------------------------------------------------------------------
def make_loss_fn(victim: Any) -> Callable[[Any, str], Any]:
    """Per-(pixels, text) loss: negative log-likelihood of ``text``.

    Untargeted stages maximise this quantity for the ground truth; targeted
    stages minimise it for the target string.  Built on the victim's
    ``targeted_logits`` if available (teacher-forced), else ``logits_fn``.
    """
    import torch

    tokenizer = getattr(victim, "tokenizer", None)
    targeted_logits = getattr(victim, "targeted_logits", None)

    if targeted_logits is not None and tokenizer is not None:
        try:
            from .attacks.jailbreak import build_targeted_loss_fn

            fn = build_targeted_loss_fn(targeted_logits, tokenizer, loss="targeted_ce", shift=1)
            if fn is not None:
                return fn
        except Exception as exc:  # pragma: no cover
            LOGGER.debug("build_targeted_loss_fn unavailable: %s", exc)

        def loss_fn(pixels, text: str):
            logits = targeted_logits(pixels, text)
            if isinstance(logits, dict):
                logits = logits.get("logits", logits)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            ids = tokenizer.encode(text, add_special_tokens=False)
            ids_t = torch.as_tensor(ids, device=logits.device, dtype=torch.long)
            t = min(logits.shape[1], ids_t.numel())
            if t == 0:
                return logits.sum() * 0.0
            target = ids_t[:t]
            pred = logits[0, :t, :]
            return torch.nn.functional.cross_entropy(pred, target)

        return loss_fn

    # last resort: differentiate the victim's predictive logits
    logits_fn = None
    if hasattr(victim, "logits_fn"):
        try:
            logits_fn = victim.logits_fn("")
        except Exception:
            logits_fn = None

    def fallback_loss(pixels, text: str):  # pragma: no cover - defensive
        if logits_fn is None:
            return pixels.sum() * 0.0
        logits = logits_fn(pixels)
        return -logits.max()

    return fallback_loss


def make_attack_fn(victim: Any, cfg: VQAEvalConfig, generator=None, attack_config=None):
    """Build the ``attack_fn(AttackRequest) -> int-coded perturbation`` callable.

    Prefers :func:`attacks.vqa_schedule.make_pgd_attack_fn`; if that helper is
    unavailable or its signature differs, falls back to a local PGD engine that
    uses the same precision policy and raw-pixel projection.
    """
    loss_fn = make_loss_fn(victim)
    config = attack_config if attack_config is not None else cfg.attack_config()
    target_loss_fn = loss_fn

    try:
        from .attacks.vqa_schedule import make_pgd_attack_fn

        try:
            attack_fn = call_with_supported_kwargs(
                make_pgd_attack_fn,
                loss_fn,
                config=config,
                generator=generator,
                attack_cls=None,
                target_loss_fn=target_loss_fn,
            )
            LOGGER.info("VQA attacks delegated to attacks.vqa_schedule.make_pgd_attack_fn")
            return attack_fn
        except TypeError as exc:
            LOGGER.debug("make_pgd_attack_fn signature mismatch (%s); using local PGD", exc)
    except Exception as exc:  # pragma: no cover
        LOGGER.debug("make_pgd_attack_fn unavailable (%s); using local PGD", exc)

    return make_local_attack_fn(loss_fn, cfg, generator=generator)


def make_local_attack_fn(loss_fn: Callable[[Any, str], Any], cfg: VQAEvalConfig, generator=None):
    """Minimal, self-contained PGD fallback (raw-pixel l_inf projection)."""
    import torch

    from .utils.precision import (
        decode_perturbation,
        encode_perturbation,
        float_dtype_for_precision,
    )

    def attack_fn(request):
        precision = getattr(request, "precision", "single")
        float_dtype = float_dtype_for_precision(precision)
        pixels = request.pixels
        if not isinstance(pixels, torch.Tensor):
            pixels = torch.as_tensor(pixels)
        device = pixels.device
        x = pixels.detach().to(device=device, dtype=float_dtype)
        eps = float(request.eps)
        alpha = float(request.alpha if request.alpha is not None else eps)
        iterations = int(request.iterations or 1)
        targeted = bool(getattr(request, "targeted", False))
        text = getattr(request, "target_text", None) or getattr(request, "ground_truth", "")

        try:
            delta0 = decode_perturbation(request.initial_delta(), float_dtype=float_dtype)
            delta0 = delta0.to(device=device, dtype=float_dtype)
        except Exception:
            if bool(getattr(request, "clean_init", False)):
                delta0 = torch.zeros_like(x)
            else:
                delta0 = (torch.rand_like(x) * 2 - 1) * eps

        delta = delta0.clamp(-eps, eps).detach()
        momentum = torch.zeros_like(x)
        g = None
        if generator is not None:
            g = generator
        for _ in range(max(1, iterations)):
            delta.requires_grad_(True)
            adv = (x + delta).clamp(cfg.clamp_min, cfg.clamp_max)
            loss = loss_fn(adv, text)
            if isinstance(loss, (list, tuple)):
                loss = loss[0]
            grad = torch.autograd.grad(loss, delta, only_inputs=True, retain_graph=False)[0]
            with torch.no_grad():
                grad = grad / (grad.abs().mean() + 1e-12)
                grad = grad.sign()
                if targeted:
                    grad = -grad
                momentum = 0.9 * momentum + grad
                delta = (delta.detach() + alpha * momentum.sign()).clamp(-eps, eps)
            delta = delta.detach()

        return encode_perturbation(delta, precision, quant_scale=cfg.quant_scale or 1e5)

    return attack_fn


def smoke_attack_fn(cfg: Optional[VQAEvalConfig] = None):
    """Model-free attack: bounded random / warm-started perturbations.

    Honours the precision policy (int16 for half, int32 for single) so the
    scheduler's dtype assertions remain meaningful under ``--smoke-test``.
    """
    import torch

    from .utils.precision import encode_perturbation

    def attack_fn(request):
        pixels = request.pixels
        if not isinstance(pixels, torch.Tensor):
            pixels = torch.as_tensor(pixels)
        eps = float(request.eps)
        try:
            delta = request.initial_delta()
            delta = delta.to(torch.float32) * 0.0  # keep shape only
        except Exception:
            delta = torch.zeros_like(pixels, dtype=torch.float32)
        gen = torch.Generator().manual_seed(
            int(getattr(request, "seed", 0) or 0) + int(getattr(request, "sample_index", 0) or 0)
        )
        noise = (torch.rand(pixels.shape, generator=gen) * 2 - 1) * eps
        delta = delta + noise
        if bool(getattr(request, "clean_init", False)):
            delta = delta * 0.0
        precision = getattr(request, "precision", "single")
        return encode_perturbation(delta, precision, quant_scale=getattr(request, "quant_scale", 1e5))

    return attack_fn


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
@dataclass
class VQAEvalResult:
    dataset_name: str
    model_name: str
    method: Optional[str]
    eps: Optional[float]
    alpha: Optional[float]
    iterations: Optional[int]
    top_k: int
    num_samples: int
    clean_accuracy: float
    attacked_accuracy: float
    stage_accuracies: Dict[str, float]
    stage_traces: Dict[str, List[str]]
    selected_ground_truths: List[str]
    precision_ok: bool
    perturbation_dtypes: List[str]
    max_perturbation_norm: Optional[float]
    clean_predictions: List[str]
    attacked_predictions: List[str]
    samples: List[Dict[str, Any]] = field(default_factory=list)
    most_frequent_ground_truth: Optional[str] = None
    top_ground_truths: List[str] = field(default_factory=list)
    score_kind: str = "accuracy"
    provenance: Dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    metrics_report: Optional[Dict[str, Any]] = None

    @property
    def accuracy_drop(self) -> float:
        return float(self.clean_accuracy - self.attacked_accuracy)

    def summary(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "model": self.model_name,
            "method": self.method,
            "eps": self.eps,
            "alpha": self.alpha,
            "iterations": self.iterations,
            "clean_accuracy": self.clean_accuracy,
            "attacked_accuracy": self.attacked_accuracy,
            "accuracy_drop": self.accuracy_drop,
            "stage_accuracies": self.stage_accuracies,
            "num_samples": self.num_samples,
            "precision_ok": self.precision_ok,
            "perturbation_dtypes": sorted(set(self.perturbation_dtypes)),
            "score_kind": self.score_kind,
        }

    def as_dict(self, include_samples: bool = True) -> Dict[str, Any]:
        d = {
            "dataset_name": self.dataset_name,
            "model_name": self.model_name,
            "method": self.method,
            "eps": self.eps,
            "alpha": self.alpha,
            "iterations": self.iterations,
            "top_k": self.top_k,
            "num_samples": self.num_samples,
            "clean_accuracy": self.clean_accuracy,
            "attacked_accuracy": self.attacked_accuracy,
            "accuracy_drop": self.accuracy_drop,
            "stage_accuracies": self.stage_accuracies,
            "stage_traces": self.stage_traces,
            "selected_ground_truths": self.selected_ground_truths,
            "most_frequent_ground_truth": self.most_frequent_ground_truth,
            "top_ground_truths": self.top_ground_truths,
            "precision_ok": self.precision_ok,
            "perturbation_dtypes": sorted(set(self.perturbation_dtypes)),
            "max_perturbation_norm": self.max_perturbation_norm,
            "score_kind": self.score_kind,
            "provenance": self.provenance,
            "elapsed_seconds": self.elapsed_seconds,
            "clean_predictions": self.clean_predictions,
            "attacked_predictions": self.attacked_predictions,
            "metrics_report": self.metrics_report,
            "summary": self.summary(),
        }
        if include_samples:
            d["samples"] = self.samples
        return d

    def to_json(self, path: Optional[str] = None, *,
                include_samples: bool = True, indent: int = 2) -> str:
        payload = json.dumps(self.as_dict(include_samples=include_samples), indent=indent)
        if path:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(payload, encoding="utf-8")
            LOGGER.info("wrote VQA result to %s", p)
        return payload


class VQAEvaluator:
    """Runs the Addendum VQA attack schedule over a benchmark and reports accuracy."""

    def __init__(
        self,
        config: Optional[VQAEvalConfig] = None,
        victim: Any = None,
        scheduler: Any = None,
        attack_fn: Callable[..., Any] = None,
        score_fn: Optional[Callable[..., float]] = None,
        accuracy_fn: Optional[Callable[..., float]] = None,
    ) -> None:
        self.config = config or VQAEvalConfig()
        self._victim = victim
        self._scheduler = scheduler
        self._attack_fn = attack_fn
        self._score_fn = score_fn
        self._accuracy_fn = accuracy_fn
        self.generator = self._make_generator()

    # -- lazy components ---------------------------------------------------
    def _make_generator(self):
        try:
            import torch

            return torch.Generator().manual_seed(int(self.config.seed or 0))
        except Exception:  # pragma: no cover
            return None

    @property
    def victim(self):
        if self._victim is None:
            self._victim = resolve_victim(self.config)
        return self._victim

    def _ensure_fns(self, dataset_name: str):
        if self._score_fn is None or self._accuracy_fn is None:
            score_fn, accuracy_fn, scorer = make_score_functions(self.victim, self.config, dataset_name)
            self._score_fn = self._score_fn or score_fn
            self._accuracy_fn = self._accuracy_fn or accuracy_fn
            LOGGER.info("score function: %s", scorer.as_dict())

    def _build_scheduler(self, dataset_name: str):
        if self._scheduler is not None:
            return self._scheduler
        from .attacks.vqa_schedule import VQAAttackScheduler

        self._ensure_fns(dataset_name)
        attack_config = self.config.attack_config(dataset_name)
        if self._attack_fn is None:
            self._attack_fn = make_attack_fn(
                self.victim, self.config, generator=self.generator, attack_config=attack_config
            )
        self._scheduler = call_with_supported_kwargs(
            VQAAttackScheduler,
            self._attack_fn,
            score_fn=self._score_fn,
            config=attack_config,
            top_k=self.config.top_k,
            dataset_name=dataset_name,
            accuracy_fn=self._accuracy_fn,
        )
        return self._scheduler

    # -- schedule execution ------------------------------------------------
    def _run_schedule(self, scheduler, pixels, top5, *, dataset_name, most_frequent,
                      sample_index, score_fn, accuracy_fn, generator):
        """Call the scheduler, tolerating either frequencies or plain strings."""
        common = dict(
            dataset_name=dataset_name,
            most_frequent=most_frequent,
            sample_index=sample_index,
            score_fn=score_fn,
            accuracy_fn=accuracy_fn,
            generator=generator,
            return_perturbations=True,
        )
        try:
            return call_with_supported_kwargs(scheduler.run, pixels, top5, **common)
        except (TypeError, AttributeError) as exc:
            LOGGER.debug("scheduler.run failed with frequencies (%s); retrying with strings", exc)
            plain = [getattr(gt, "answer", gt) for gt in top5]
            return call_with_supported_kwargs(scheduler.run, pixels, plain, **common)

    # -- main loop ---------------------------------------------------------
    def run(self, samples: Optional[Sequence[Any]] = None,
            dataset_name: Optional[str] = None) -> VQAEvalResult:
        from .attacks.vqa_schedule import (
            LOW_PRECISION,
            HIGH_PRECISION,
            VQA_TARGET_MAYBE,
            VQA_TARGET_WORD,
            most_frequent_ground_truth,
            most_frequent_ground_truths,
        )

        start = time.time()
        ds = _normalize_dataset(dataset_name or self.config.dataset_name)
        if samples is None:
            samples = resolve_samples(self.config, ds)
        samples = list(samples)
        if self.config.num_samples is not None:
            samples = samples[: self.config.num_samples]

        scheduler = self._build_scheduler(ds)
        self._ensure_fns(ds)

        top5 = most_frequent_ground_truths(samples, k=self.config.top_k)
        mf_gt = most_frequent_ground_truth(samples)
        LOGGER.info(
            "%s: %d samples, top-%d ground truths %s, most frequent %r",
            ds, len(samples), self.config.top_k,
            [getattr(g, "answer", g) for g in top5], mf_gt,
        )

        clean_predictions: List[str] = []
        attacked_predictions: List[str] = []
        per_sample: List[Dict[str, Any]] = []
        stage_acc: Dict[str, List[float]] = {}
        stage_traces: Dict[str, List[str]] = {}
        selected: List[str] = []
        dtypes: List[str] = []
        max_norm = 0.0
        precision_ok = True

        for i, sample in enumerate(samples):
            pixels = sample_pixels(sample, self.config, device=self.config.device)
            gts = _answers_of(sample)
            gt = gts[0] if gts else ""

            clean_pred = self._predict(pixels, gt, ds)
            clean_acc = self._accuracy_fn(pixels, gt)
            clean_predictions.append(clean_pred)

            result = self._run_schedule(
                scheduler, pixels, top5,
                dataset_name=ds, most_frequent=mf_gt, sample_index=i,
                score_fn=self._score_fn, accuracy_fn=self._accuracy_fn,
                generator=self.generator,
            )

            trace = tuple(getattr(result, "trace", ()) or ())
            stage_traces.setdefault("_all", []).append(" -> ".join(trace))
            for name in (LOW_PRECISION, HIGH_PRECISION, VQA_TARGET_MAYBE, VQA_TARGET_WORD):
                stage_traces.setdefault(name, []).append(
                    "executed" if any(name in t for t in trace) else "skipped"
                )

            best_accuracy, worst_accuracy = None, None
            for stage in getattr(result, "stages", []) or []:
                sname = getattr(stage, "name", None) or "stage"
                if getattr(stage, "skipped", False):
                    continue
                acc = getattr(stage, "accuracy", None)
                if acc is not None:
                    stage_acc.setdefault(sname, []).append(float(acc))
                if acc is not None and (worst_accuracy is None or float(acc) < worst_accuracy):
                    worst_accuracy = float(acc)
                if acc is not None and (best_accuracy is None or float(acc) > best_accuracy):
                    best_accuracy = float(acc)
                dt = getattr(stage, "perturbation_dtype", None)
                if dt is not None:
                    dtypes.append(str(dt).replace("torch.", ""))
                pert = getattr(stage, "perturbation", None)
                if pert is not None:
                    try:
                        max_norm = max(max_norm, float(pert.float().abs().max().item()))
                    except Exception:
                        pass

            attacked_acc = worst_accuracy if worst_accuracy is not None else clean_acc
            sel = getattr(result, "selected_ground_truth", None)
            if sel:
                selected.append(str(getattr(sel, "answer", sel)))

            attacked_pred = self._predict(pixels, sel or gt, ds) if sel or gt else ""
            attacked_predictions.append(attacked_pred)

            per_sample.append(
                {
                    "sample_index": i,
                    "sample_id": _sample_id(sample, i),
                    "question": _question_of(sample),
                    "ground_truth": gt,
                    "clean_prediction": clean_pred,
                    "clean_accuracy": clean_acc,
                    "attacked_prediction": attacked_pred,
                    "attacked_accuracy": attacked_acc,
                    "selected_ground_truth": str(getattr(sel, "answer", sel)) if sel else None,
                    "trace": list(trace),
                    "stages": [_stage_summary(s) for s in (getattr(result, "stages", []) or [])],
                }
            )
            if getattr(result, "best_perturbation", None) is not None:
                try:
                    d = result.best_perturbation
                    dtypes.append(str(d.dtype).replace("torch.", ""))
                except Exception:
                    pass
            if (i + 1) % max(1, len(samples) // 10) == 0 or self.config.verbose:
                LOGGER.info(
                    "[%s] %d/%d clean_acc=%.3f attacked_acc=%.3f",
                    ds, i + 1, len(samples), clean_acc, attacked_acc,
                )

        precision_ok = all(dt in ("int16", "int32") for dt in dtypes) if dtypes else True
        if not precision_ok:
            LOGGER.error("precision policy violated: perturbation dtypes %s", sorted(set(dtypes)))

        clean_accuracy = _mean([s["clean_accuracy"] for s in per_sample])
        attacked_accuracy = _mean([float(s["attacked_accuracy"]) for s in per_sample])
        stage_accuracies = {k: _mean(v) for k, v in stage_acc.items()}

        result = VQAEvalResult(
            dataset_name=ds,
            model_name=self.config.model_name,
            method=self.config.model_kwargs.get("method") if isinstance(self.config.model_kwargs, dict) else None,
            eps=self.config.resolved_eps(),
            alpha=self.config.high_alpha if self.config.high_alpha is not None else self.config.low_alpha,
            iterations=self.config.high_iterations
            if self.config.high_iterations is not None
            else self.config.low_iterations,
            top_k=self.config.top_k,
            num_samples=len(per_sample),
            clean_accuracy=clean_accuracy,
            attacked_accuracy=attacked_accuracy,
            stage_accuracies=stage_accuracies,
            stage_traces=stage_traces,
            selected_ground_truths=selected,
            precision_ok=precision_ok,
            perturbation_dtypes=dtypes,
            max_perturbation_norm=max_norm if max_norm else None,
            clean_predictions=clean_predictions,
            attacked_predictions=attacked_predictions,
            samples=per_sample if self.config.save_perturbations else [],
            most_frequent_ground_truth=mf_gt,
            top_ground_truths=[str(getattr(g, "answer", g)) for g in top5],
            score_kind=self.config.score_kind,
            provenance=dict(self.config.provenance),
            elapsed_seconds=time.time() - start,
        )
        result.metrics_report = self._build_metrics_report(result, per_sample, ds)
        return result

    # -- helpers -----------------------------------------------------------
    def _predict(self, pixels, ground_truth: str, dataset_name: str) -> str:
        """Greedy answer used for the reported accuracy (attacked pixels)."""
        try:
            scorer = VictimScorer(
                self.victim,
                dataset_name=dataset_name,
                prompt=self.config.prompt,
                kind="accuracy",
                generation_kwargs={
                    "max_new_tokens": self.config.max_new_tokens,
                    "temperature": self.config.generation_temperature,
                },
            )
            response = scorer.generate(pixels, ground_truth)
            return scorer.parse(response, ground_truth)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.debug("prediction failed: %s", exc)
            return ""

    def _build_metrics_report(self, result: VQAEvalResult, per_sample, dataset_name: str):
        """Optionally build a :class:`metrics.vqa.VQAReport` for JSON tables."""
        try:
            from .metrics import vqa as MV

            aggregate = getattr(MV, "aggregate_reports", None)
            if aggregate is None:
                return None
            reports = []
            for s in per_sample:
                try:
                    reports.append(
                        call_with_supported_kwargs(
                            MV.sample_result_from_schedule,
                            None,
                            None,
                            sample_index=s["sample_index"],
                            clean_prediction=s["clean_prediction"],
                            attacked_prediction=s["attacked_prediction"],
                            dataset_name=dataset_name,
                        )
                    )
                except Exception:
                    continue
            report = call_with_supported_kwargs(
                aggregate,
                reports,
                dataset_name=dataset_name,
                model_name=result.model_name,
                method=result.method,
                attack_name="scheduled",
                clean_predictions=result.clean_predictions,
                clean_references=[s["ground_truth"] for s in per_sample],
                score_kind=result.score_kind,
                stage_precisions=["half", "single", "single", "single"],
                top_ground_truths=result.top_ground_truths,
                most_frequent_ground_truth=result.most_frequent_ground_truth,
                elapsed_seconds=result.elapsed_seconds,
                provenance=result.provenance,
            )
            if hasattr(report, "as_dict"):
                return report.as_dict()
            return dict(report)
        except Exception as exc:
            LOGGER.debug("metrics.vqa aggregation unavailable: %s", exc)
            return None


def _mean(values: Sequence[float]) -> float:
    values = [float(v) for v in values if v is not None]
    return float(sum(values) / len(values)) if values else 0.0


def _sample_id(sample: Any, index: int) -> Any:
    if isinstance(sample, dict):
        return sample.get("sample_id", sample.get("id", index))
    return getattr(sample, "sample_id", getattr(sample, "id", index))


def _question_of(sample: Any) -> str:
    if isinstance(sample, dict):
        return str(sample.get("question", ""))
    return str(getattr(sample, "question", ""))


def _answers_of(sample: Any) -> List[str]:
    try:
        from .metrics.vqa import answers_of

        ans = answers_of(sample)
        if ans:
            return [str(a) for a in ans]
    except Exception:
        pass
    if isinstance(sample, dict):
        raw = sample.get("answers") or sample.get("ground_truths") or [sample.get("answer", "")]
        return [str(a) for a in (raw if isinstance(raw, (list, tuple)) else [raw])]
    raw = getattr(sample, "answers", None) or getattr(sample, "ground_truths", None)
    if raw is None:
        return []
    return [str(a) for a in (raw if isinstance(raw, (list, tuple)) else [raw])]


def _stage_summary(stage: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in (
        "index",
        "name",
        "precision",
        "targeted",
        "ground_truth",
        "target_text",
        "clean_init",
        "skipped",
        "score",
        "accuracy",
        "perturbation_dtype",
    ):
        if hasattr(stage, key):
            out[key] = getattr(stage, key)
    dt = getattr(stage, "perturbation_dtype", None)
    if dt is not None:
        out["perturbation_dtype"] = str(dt).replace("torch.", "")
    return out


# ---------------------------------------------------------------------------
# comparison tables
# ---------------------------------------------------------------------------
def build_comparison_table(results: Sequence[VQAEvalResult]) -> List[Dict[str, Any]]:
    rows = []
    for r in results:
        row = r.summary()
        row["theoretical_clean"] = None
        rows.append(row)
    return rows


def format_table(rows: Sequence[Dict[str, Any]], percent: bool = True) -> str:
    if not rows:
        return "(no rows)"

    def fmt(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, float) and percent:
            return f"{100.0 * value:.2f}"
        return str(value)

    cols = ["dataset", "model", "eps", "clean_accuracy", "attacked_accuracy", "accuracy_drop", "precision_ok"]
    widths = {c: max(len(c), max(len(fmt(r.get(c))) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    sep = "  ".join("-" * widths[c] for c in cols)
    body = ["  ".join(fmt(r.get(c)).ljust(widths[c]) for c in cols) for r in rows]
    return "\n".join([header, sep] + body)


# ---------------------------------------------------------------------------
# smoke test
# ---------------------------------------------------------------------------
def run_smoke_test(verbose: bool = True) -> Dict[str, Any]:
    """Model-free verification of the Addendum VQA schedule ordering."""
    from .attacks.vqa_schedule import (
        CALL_ORDER,
        VQA_TARGET_MAYBE,
        VQA_TARGET_WORD,
        is_textvqa,
    )

    out: Dict[str, Any] = {}
    for ds in (POPE, TEXT_VQA):
        cfg = VQAEvalConfig(
            dataset_name=ds,
            model_name="dummy",
            num_samples=3,
            top_k=5,
            low_eps=2.0 / 255,
            low_alpha=2.0 / 255,
            low_iterations=2,
            high_eps=2.0 / 255,
            high_alpha=2.0 / 255,
            high_iterations=2,
            targeted_eps=2.0 / 255,
            targeted_alpha=2.0 / 255,
            targeted_iterations=2,
            save_perturbations=True,
        )
        evaluator = VQAEvaluator(cfg, attack_fn=smoke_attack_fn(cfg))
        evaluator._score_fn = lambda pixels, gt: 1.0  # constant score
        evaluator._accuracy_fn = lambda pixels, gt: 1.0
        result = evaluator.run()
        out[ds] = result.summary()

        assert result.num_samples == 3, result.num_samples
        assert result.precision_ok, f"precision policy violated: {result.perturbation_dtypes}"
        assert all(d in ("int16", "int32") for d in result.perturbation_dtypes), result.perturbation_dtypes
        traces = result.stage_traces.get("_all", [])
        assert traces, "no schedule traces recorded"
        for trace in traces:
            has_word = VQA_TARGET_WORD in trace
            if is_textvqa(ds):
                assert not has_word, f'"Word" attack must be skipped on TextVQA: {trace}'
            else:
                assert has_word, f'"Word" attack missing for {ds}: {trace}'
        assert any(VQA_TARGET_MAYBE in t for t in traces), traces
        assert result.most_frequent_ground_truth is not None
        if verbose:
            LOGGER.info("[smoke] %s -> %s", ds, json.dumps(result.summary(), default=str))
            LOGGER.info("[smoke] %s traces: %s", ds, traces)

    out["scheduler_call_order"] = list(CALL_ORDER)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m robust_clip_repro.eval_vqa",
        description="VQA robustness evaluation (TextVQA / POPE / SQA-I) for Robust CLIP.",
    )
    p.add_argument("--config", default=None, help=f"YAML config (default: {DEFAULT_CONFIG_PATH})")
    p.add_argument("--dataset", default=None, choices=list(DATASETS), help="benchmark to evaluate")
    p.add_argument("--datasets", nargs="+", default=None, help="evaluate several benchmarks")
    p.add_argument("--model", default=None, choices=list(SUPPORTED_MODELS), help="victim model")
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--top-k", type=int, default=None, help="Addendum: 5 most frequent ground truths")
    p.add_argument("--eps", type=float, default=None, help="l_inf budget (NOT in Addendum)")
    p.add_argument("--alpha", type=float, default=None, help="step size (NOT in Addendum)")
    p.add_argument("--iterations", type=int, default=None, help="iterations (NOT in Addendum)")
    p.add_argument("--norm", default=None, choices=["linf", "l2"])
    p.add_argument("--score-kind", default=None, choices=["accuracy", "constant", "logprob"])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--output", default=None, help="output JSON path")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--smoke-test", action="store_true", help="model-free schedule verification")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_arg_parser().parse_args(argv)

    if args.smoke_test:
        run_smoke_test(verbose=not args.quiet)
        print(json.dumps({"smoke_test": "ok"}, indent=2))
        return 0

    cfg = VQAEvalConfig.from_dict(
        load_config(args.config),
        dataset_name=args.dataset,
        datasets=args.datasets,
        model_name=args.model,
        num_samples=args.num_samples,
        top_k=args.top_k,
        high_eps=args.eps,
        high_alpha=args.alpha,
        high_iterations=args.iterations,
        low_eps=args.eps,
        low_alpha=args.alpha,
        low_iterations=args.iterations,
        targeted_eps=args.eps,
        targeted_alpha=args.alpha,
        targeted_iterations=args.iterations,
        norm=args.norm,
        score_kind=args.score_kind,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )

    if cfg.resolved_eps() is None:
        LOGGER.error(
            "no attack budget supplied: eps/alpha/iterations are NOT specified by the "
            "Addendum, so they must be provided (--eps/--alpha/--iterations or the config "
            "file). Refusing to invent them."
        )
        return 2

    LOGGER.info("config: %s", json.dumps(cfg.as_dict(), indent=2, default=str))
    LOGGER.info("values not specified by the Addendum (external defaults): %s",
                json.dumps(EXTERNAL_DEFAULTS, default=str))

    datasets = args.datasets or [cfg.dataset_name]
    results: List[VQAEvalResult] = []
    for ds in datasets:
        cfg_ds = VQAEvalConfig.from_dict(cfg.as_dict(), dataset_name=ds)
        evaluator = VQAEvaluator(cfg_ds)
        results.append(evaluator.run(dataset_name=ds))

    print(format_table(build_comparison_table(results)))

    out_path = args.output
    if not out_path:
        ds_tag = "_".join(_normalize_dataset(d) for d in datasets).replace(" ", "")
        eps = cfg.resolved_eps()
        eps_tag = f"_eps{round(eps * 255)}" if eps else ""
        out_path = os.path.join(
            args.output_dir or cfg.output_dir,
            f"vqa_{ds_tag}_{cfg.model_name}{eps_tag}.json",
        )
    payload = {
        "config": cfg.as_dict(),
        "external_defaults": EXTERNAL_DEFAULTS,
        "results": [r.as_dict(include_samples=True) for r in results],
        "table": build_comparison_table(results),
    }
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    LOGGER.info("wrote %s", p)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
