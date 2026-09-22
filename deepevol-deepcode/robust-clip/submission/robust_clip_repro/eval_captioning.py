"""Image captioning robustness evaluation (COCO / Flickr30k).

This harness wires together the captioning attack pipeline implemented in
:mod:`robust_clip_repro.attacks.captioning` with

* a captioning victim model (OpenFlamingo or LLaVA-1.5; both loaded lazily so
  that the harness imports without the (heavy) model dependencies installed),
* the COCO / Flickr30k data loader from :mod:`robust_clip_repro.data.coco_captioning`,
* the CIDEr metric from :mod:`robust_clip_repro.metrics.cider`.

Paper / Addendum specification implemented here
-----------------------------------------------
From Sec. 4.1 ("Attacks are based on Schlarmann & Hein (2023)") and App. B.6:

* COCO and Flickr30k provide **five ground-truth captions** per image and each
  ground truth is considered when computing CIDEr.
* **APGD at half precision with 100 iterations** is run against each ground
  truth.
* **After each attack the CIDEr score is recomputed** and samples whose score
  already dropped below a threshold are **not attacked any more**.  The
  thresholds are ``10`` for COCO and ``2`` for Flickr30k, corresponding to less
  than 10% of the original LLaVA performance.
* In the **final step a similar attack at single precision** is run, using the
  ground truth that led to the **lowest score**, initialized with the
  corresponding perturbation (warm start).
* Perturbation budgets: ``eps = 2/255`` and ``eps = 4/255`` (Sec. 4.1).
* The initial APGD step-size is set to ``eps`` (App. B.6).
* Adversarial evaluations use **500 randomly sampled images**; clean
  evaluations use all available samples (Sec. 4.1).
* Precision policy (Addendum): half-precision attacks store perturbations as
  ``int16``, single-precision attacks as ``int32`` -- inherited from
  :mod:`robust_clip_repro.utils.precision` via the attack suite.

Anything the paper/Addendum does not state (batch size, seeds, exact prompt
template, OpenFlamingo version, the exact caption loss, ...) is configurable and
is logged as externally supplied rather than being presented as a paper value.

CLI
---
::

    python -m robust_clip_repro.eval_captioning --dataset COCO \
        --model openflamingo --eps 0.00784313725490196 --num-samples 500

    # model-free end-to-end orchestration smoke test
    python -m robust_clip_repro.eval_captioning --smoke-test
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .attacks.captioning import (
    CAPTIONING_THRESHOLDS,
    HALF_PRECISION,
    HALF_PRECISION_ITERATIONS,
    NUM_GROUND_TRUTHS,
    PAPER_ATTACK_SAMPLES,
    PAPER_EPSILONS,
    SINGLE_PRECISION,
    STAGE_HALF_PRECISION,
    STAGE_SINGLE_PRECISION,
    SUPPORTED_DATASETS,
    UNSPECIFIED,
    CaptioningAttackConfig,
    CaptioningAttackResult,
    CaptioningAttackSuite,
    CaptioningSampleState,
    build_caption_loss_fn,
    default_threshold,
    make_apgd_attack_fn,
    make_cider_fn,
    make_pgd_attack_fn,
)
from .utils.precision import QUANT_SCALE, int_dtype_for_precision

LOGGER = logging.getLogger("robust_clip_repro.eval_captioning")

#: Default location of the YAML configuration file.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "captioning.yaml"

#: Captioning datasets supported by the paper.
SUPPORTED_MODELS = ("openflamingo", "llava", "dummy")

#: Values that the paper/Addendum does not pin down.  They are exposed as
#: configuration and are reported as externally supplied.
EXTERNAL_DEFAULTS = {
    "batch_size": 1,
    "seed": 0,
    "num_workers": 0,
    "prompt": UNSPECIFIED,
    "caption_loss": UNSPECIFIED,
    "subsample_seed": 0,
    "openflamingo_version": UNSPECIFIED,
    "openflamingo_vision_encoder": UNSPECIFIED,
    "openflamingo_lang_encoder": UNSPECIFIED,
    "captioning_temperature": 0.0,
    "max_new_tokens": 32,
    "output_dir": "results",
}


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass
class CaptioningEvalConfig:
    """Configuration of a captioning robustness evaluation run.

    Attributes with paper-stated values:
        dataset_name, eps, iterations_half, threshold, num_ground_truths,
        num_samples (adversarial), num_samples_clean, alpha (= eps),
        precision_half ("half" -> int16), precision_single ("single" -> int32).

    Attributes the paper does not state are marked ``UNSPECIFIED_BY_ADDENDUM``
    and are supplied externally.
    """

    dataset_name: str = "COCO"
    model_name: str = "openflamingo"
    eps: Optional[float] = None
    alpha: Optional[float] = None
    iterations_half: int = HALF_PRECISION_ITERATIONS
    iterations_single: Optional[int] = None
    threshold: Optional[float] = None
    norm: str = "linf"
    loss: str = "ce"
    num_ground_truths: int = NUM_GROUND_TRUTHS
    num_samples: Optional[int] = PAPER_ATTACK_SAMPLES
    num_samples_clean: Optional[int] = None
    warm_start_single: bool = True
    precision_half: str = HALF_PRECISION
    precision_single: str = SINGLE_PRECISION
    quant_scale: float = QUANT_SCALE
    batch_size: int = EXTERNAL_DEFAULTS["batch_size"]
    seed: int = EXTERNAL_DEFAULTS["seed"]
    subsample_seed: int = EXTERNAL_DEFAULTS["subsample_seed"]
    max_new_tokens: int = EXTERNAL_DEFAULTS["max_new_tokens"]
    split: str = "test"
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    output_dir: str = EXTERNAL_DEFAULTS["output_dir"]
    output_file: Optional[str] = None
    device: Optional[str] = None
    verbose: bool = True
    smoke_test: bool = False
    provenance: Dict[str, str] = field(default_factory=dict)

    # -- helpers ----------------------------------------------------------
    def __post_init__(self) -> None:
        if self.dataset_name not in SUPPORTED_DATASETS:
            # Be permissive but explicit: unknown datasets reuse the COCO
            # threshold only if one is supplied explicitly.
            LOGGER.warning(
                "Unknown captioning dataset %r (paper uses %s); "
                "threshold must be supplied explicitly.",
                self.dataset_name,
                ", ".join(SUPPORTED_DATASETS),
            )
        if self.threshold is None:
            self.threshold = default_threshold(self.dataset_name)
        if self.alpha is None and self.eps is not None:
            # App. B.6: the initial step-size of APGD is set to eps.
            self.alpha = float(self.eps)
        self._record_provenance()

    def _record_provenance(self) -> None:
        self.provenance.setdefault("eps", "paper_sec4.1" if self.eps in PAPER_EPSILONS else "external")
        self.provenance.setdefault("alpha", "paper_appB.6 (alpha = eps)")
        self.provenance.setdefault("iterations_half", "paper_sec4.1/appB.6 (100 iterations, half precision)")
        self.provenance.setdefault("threshold", f"paper_appB.6 (dataset={self.dataset_name})")
        self.provenance.setdefault("num_ground_truths", "paper_appB.6 (five ground-truth captions)")
        self.provenance.setdefault("warm_start_single", "paper_appB.6 (init with the according perturbation)")
        self.provenance.setdefault("precision_half", "addendum (int16 for half precision)")
        self.provenance.setdefault("precision_single", "addendum (int32 for single precision)")
        self.provenance.setdefault("num_samples", "paper_sec4.1 (500 samples for adversarial eval)")
        self.provenance.setdefault("num_samples_clean", "paper_sec4.1 (all samples for clean eval)")
        for key in ("batch_size", "seed", "subsample_seed", "max_new_tokens", "prompt",
                    "caption_loss", "openflamingo_version", "captioning_temperature",
                    "split", "model_kwargs"):
            self.provenance.setdefault(key, "external (not specified by paper/addendum)")

    def resolved_eps(self) -> float:
        if self.eps is None:
            raise ValueError(
                "eps is required: the Addendum/paper does not fix a default. "
                "Pass --eps (paper uses 2/255 and 4/255)."
            )
        return float(self.eps)

    @property
    def half_precision_int_dtype(self) -> torch.dtype:
        return int_dtype_for_precision(self.precision_half)

    @property
    def single_precision_int_dtype(self) -> torch.dtype:
        return int_dtype_for_precision(self.precision_single)

    def attack_config(self) -> CaptioningAttackConfig:
        return CaptioningAttackConfig(
            dataset_name=self.dataset_name,
            num_ground_truths=self.num_ground_truths,
            threshold=self.threshold,
            eps=self.resolved_eps(),
            alpha=self.alpha,
            iterations_half=self.iterations_half,
            iterations_single=self.iterations_single,
            precision_half=self.precision_half,
            precision_single=self.precision_single,
            norm=self.norm,
            loss=self.loss,
            warm_start_single=self.warm_start_single,
            quant_scale=self.quant_scale,
            seed=self.seed,
            verbose=self.verbose,
        )

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["model_kwargs"] = dict(self.model_kwargs)
        return d

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]] = None, **overrides: Any) -> "CaptioningEvalConfig":
        cfg = dict(cfg or {})
        cfg.update(overrides)
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        extra = {k: cfg.pop(k) for k in list(cfg) if k not in known}
        # tolerate a nested {"attack": {...}} / {"eval": {...}} layout
        nested = cfg.pop("attack", None) or {}
        if isinstance(nested, dict):
            for k, v in nested.items():
                if k in known and cfg.get(k) is None:
                    cfg[k] = v
        cfg.setdefault("provenance", {})
        for k, v in extra.items():
            LOGGER.warning("Ignoring unknown captioning config key %r=%r", k, v)
        return cls(**cfg)


def load_config(path: Optional[str]) -> Dict[str, Any]:
    """Load a (YAML) configuration file; returns an empty dict when absent."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        LOGGER.warning("Config file %s not found; using defaults.", p)
        return {}
    try:
        import yaml  # type: ignore
    except Exception:  # pragma: no cover - PyYAML is a declared requirement
        LOGGER.warning("PyYAML unavailable; ignoring %s", p)
        return {}
    with p.open("r") as fh:
        cfg = yaml.safe_load(fh) or {}
    if "captioning" in cfg and isinstance(cfg["captioning"], dict):
        cfg = cfg["captioning"]
    return cfg


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def normalize_sample(sample: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
    """Normalize a dataset sample to the schema used by the attack suite.

    Both the loader in :mod:`robust_clip_repro.data.coco_captioning` and the
    attack suite are supported, i.e. the sample carries ``pixels``/``image`` and
    ``ground_truths``/``captions``/``references``.
    """
    out = dict(sample)
    pixels = out.get("pixels", out.get("image", out.get("pixel_values")))
    gts = out.get("ground_truths", out.get("captions", out.get("references")))
    if gts is None:
        gts = []
    if isinstance(gts, str):
        gts = [gts]
    gts = [str(g) for g in gts]
    sid = out.get("sample_id", out.get("id", index))
    out.update(
        {
            "sample_id": sid,
            "id": sid,
            "pixels": pixels,
            "image": pixels,
            "ground_truths": gts,
            "captions": gts,
            "references": gts,
        }
    )
    return out


def resolve_samples(cfg: CaptioningEvalConfig, num_samples: Optional[int]) -> List[Dict[str, Any]]:
    """Load captioning samples (500 by default for adversarial evaluation)."""
    try:
        from .data.coco_captioning import load_captioning_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional deps
        LOGGER.error(
            "Could not import the captioning data loader (%s). "
            "Falling back to synthetic samples; results are not paper numbers.",
            exc,
        )
        return synthetic_samples(max(num_samples or 4, 1))
    samples = load_captioning_dataset(
        dataset_name=cfg.dataset_name,
        split=cfg.split,
        num_samples=num_samples,
        num_ground_truths=cfg.num_ground_truths,
        seed=cfg.subsample_seed,
    )
    return [normalize_sample(s, i) for i, s in enumerate(samples)]


def synthetic_samples(n: int, num_ground_truths: int = NUM_GROUND_TRUTHS) -> List[Dict[str, Any]]:
    """Deterministic toy samples used for the smoke test / missing-data fallback."""
    samples = []
    for i in range(n):
        pixels = torch.full((3, 224, 224), (i % 7) / 8.0, dtype=torch.float32)
        gts = [f"synthetic caption {i} variant {j}" for j in range(num_ground_truths)]
        samples.append(normalize_sample({"pixels": pixels, "captions": gts, "sample_id": f"syn-{i}"}, i))
    return samples


# ---------------------------------------------------------------------------
# victim models
# ---------------------------------------------------------------------------


class Captioner:
    """Uniform captioning interface used by the harness.

    ``__call__`` accepts either ``(pixels)`` or ``(pixels, prompt)``; attack
    suites and the clean evaluation path may use either form.
    """

    name = "base"
    num_ground_truths = NUM_GROUND_TRUTHS

    def __call__(self, pixels: torch.Tensor, *args: Any, **kwargs: Any) -> List[str]:  # pragma: no cover
        raise NotImplementedError

    # -- optional hooks used by model-conditioned attack losses ------------
    def logits_fn(self, pixels: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    @property
    def tokenizer(self):  # pragma: no cover
        raise NotImplementedError


class DummyCaptioner(Captioner):
    """Deterministic, dependency-free captioner for smoke tests.

    The generated caption degrades monotonically as the (normalized) L1
    perturbation grows, which mimics the qualitative behaviour of a victim
    model and lets the orchestration be tested end-to-end.
    """

    name = "dummy"

    def __init__(self, base_caption: str = "a photo of a synthetic scene", temperature: float = 1.0):
        self.base_caption = base_caption
        self.temperature = temperature

    def __call__(self, pixels: torch.Tensor, *args: Any, **kwargs: Any) -> List[str]:
        with torch.no_grad():
            if not torch.is_tensor(pixels):
                pixels = torch.as_tensor(pixels)
            mag = float(pixels.float().abs().mean()) * float(self.temperature)
        if mag > 0.25:
            return [" .", ""]
        if mag > 0.05:
            return ["a photo", "of a scene"]
        return [self.base_caption]


def resolve_captioner(cfg: CaptioningEvalConfig) -> Captioner:
    """Instantiate the victim captioning model (lazily importing heavy deps)."""
    name = (cfg.model_name or "openflamingo").lower()
    if name in ("dummy", "smoke", "synthetic"):
        return DummyCaptioner(temperature=float(cfg.model_kwargs.get("temperature", 1.0)))
    if name in ("openflamingo", "of", "open_flamingo"):
        from .models.openflamingo_wrapper import OpenFlamingoCaptioner  # type: ignore

        return OpenFlamingoCaptioner(
            device=cfg.device,
            max_new_tokens=cfg.max_new_tokens,
            **{k: v for k, v in cfg.model_kwargs.items() if k != "temperature"},
        )
    if name in ("llava", "llava-1.5", "llava_1_5", "llava-1.5-7b"):
        from .models.llava_openclip import LLaVACaptioner  # type: ignore

        return LLaVACaptioner(
            device=cfg.device,
            max_new_tokens=cfg.max_new_tokens,
            **{k: v for k, v in cfg.model_kwargs.items() if k != "temperature"},
        )
    raise ValueError(f"Unsupported captioning model {cfg.model_name!r}; expected one of {SUPPORTED_MODELS}.")


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def _cider_fn_robust(dataset_name: str) -> Callable[..., float]:
    """CIDEr callable tolerant to the different call conventions.

    Supports ``cider_fn(caption, references)``, ``cider_fn(candidates,
    references)`` and ``cider_fn(captions=..., references=...)``.
    """
    raw = make_cider_fn()

    def cider_fn(caption: Any, references: Any = None, *args: Any, **kwargs: Any) -> float:
        cand = kwargs.pop("captions", kwargs.pop("candidates", caption))
        refs = kwargs.pop("references", kwargs.pop("ground_truths", references))
        if refs is None:
            raise ValueError("CIDEr requires reference captions.")
        # Normalize reference container: a single reference string is allowed.
        if isinstance(refs, str):
            refs = [[refs]]
        elif isinstance(refs, (list, tuple)) and refs and isinstance(refs[0], str):
            refs = [list(refs)]
        return float(raw(cand, refs, *args, **kwargs))

    return cider_fn


def make_non_differentiable_cider_fn(dataset_name: str) -> Callable[..., float]:
    """Public alias used by tests/main.py."""
    return _cider_fn_robust(dataset_name)


# ---------------------------------------------------------------------------
# attack wiring
# ---------------------------------------------------------------------------


def make_attack_fn(captioner: Captioner, cfg: CaptioningEvalConfig, generator: Optional[torch.Generator]):
    """Build the captioning ``attack_fn`` used by the attack suite.

    Uses the upstream APGD implementation (fra31/robust-finetuning, vendored in
    :mod:`robust_clip_repro.attacks.apgd`) with initial step-size ``eps``
    (App. B.6).  Falls back to the exact-PGD engine of
    :mod:`robust_clip_repro.attacks.pgd` if APGD is unavailable.
    """

    def loss_fn_factory(ground_truth: str) -> Callable[[torch.Tensor], torch.Tensor]:
        return build_caption_loss_fn(
            captioner.logits_fn,
            captioner.tokenizer,
            ground_truth,
        )

    try:
        return make_apgd_attack_fn(loss_fn_factory, config=cfg.attack_config(), generator=generator)
    except Exception as exc:  # pragma: no cover - environment dependent
        LOGGER.warning("APGD unavailable (%s); using the exact-PGD engine instead.", exc)
        return make_pgd_attack_fn(loss_fn_factory, config=cfg.attack_config(), generator=generator)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


@dataclass
class CaptioningEvalResult:
    """Result of one (dataset, model, eps) captioning evaluation."""

    dataset_name: str
    model_name: str
    eps: float
    threshold: float
    num_samples: int
    clean_cider: float
    worst_cider: float
    clean_per_sample: List[float] = field(default_factory=list)
    worst_per_sample: List[float] = field(default_factory=list)
    early_stopped_fraction: float = 0.0
    precision_ok: bool = False
    captions: List[Dict[str, Any]] = field(default_factory=list)
    perturbations: Optional[torch.Tensor] = None
    attack_summary: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def as_dict(self, include_perturbations: bool = False) -> Dict[str, Any]:
        d = {
            "dataset": self.dataset_name,
            "model": self.model_name,
            "eps": self.eps,
            "eps_over_255": self.eps * 255.0,
            "threshold": self.threshold,
            "num_samples": self.num_samples,
            "clean_cider": self.clean_cider,
            "worst_cider": self.worst_cider,
            "cider_drop": self.clean_cider - self.worst_cider,
            "clean_per_sample": list(self.clean_per_sample),
            "worst_per_sample": list(self.worst_per_sample),
            "early_stopped_fraction": self.early_stopped_fraction,
            "precision_policy_ok": self.precision_ok,
            "captions": list(self.captions),
            "attack_summary": dict(self.attack_summary),
            "config": dict(self.config),
            "elapsed_seconds": self.elapsed_seconds,
        }
        if include_perturbations and self.perturbations is not None:
            d["perturbations"] = self.perturbations
        return d


class CaptioningEvaluator:
    """Clean + adversarial (worst-case CIDEr) captioning evaluation."""

    def __init__(
        self,
        config: CaptioningEvalConfig,
        captioner: Optional[Captioner] = None,
        cider_fn: Optional[Callable[..., float]] = None,
        attack_fn: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.cfg = config
        self.captioner = captioner if captioner is not None else resolve_captioner(config)
        self.cider_fn = cider_fn if cider_fn is not None else _cider_fn_robust(config.dataset_name)
        self.generator = torch.Generator().manual_seed(int(config.seed))
        self.device = torch.device(config.device) if config.device else None
        if attack_fn is not None:
            self.attack_fn = attack_fn
        else:
            try:
                self.attack_fn = make_attack_fn(self.captioner, config, self.generator)
            except Exception as exc:  # pragma: no cover - depends on model deps
                if config.smoke_test:
                    LOGGER.warning("Falling back to the synthetic attack fn (%s).", exc)
                    self.attack_fn = smoke_attack_fn(config)
                else:
                    raise
        self.attack_cls = None  # populated lazily inside _build_suite
        self._suite: Optional[CaptioningAttackSuite] = None

    # -- caption generation -------------------------------------------------
    def caption(self, pixels: torch.Tensor) -> str:
        out = self.captioner(pixels)
        if isinstance(out, str):
            return out
        if isinstance(out, (list, tuple)):
            return str(out[0]) if out else ""
        return str(out)

    def _clean_cider(self, pixels: torch.Tensor, ground_truths: Sequence[str]) -> Tuple[float, str]:
        caption = self.caption(pixels)
        refs = [list(ground_truths)]
        try:
            score = float(self.cider_fn([caption], refs))
        except TypeError:
            score = float(self.cider_fn(caption, list(ground_truths)))
        return score, caption

    # -- attack suite -------------------------------------------------------
    def _build_suite(self) -> CaptioningAttackSuite:
        if self._suite is not None:
            return self._suite
        try:
            from .attacks.apgd import APGDAttack  # type: ignore

            attack_cls: Any = APGDAttack
        except Exception:  # pragma: no cover
            attack_cls = None
        self._suite = CaptioningAttackSuite(
            attack_fn=self.attack_fn,
            config=self.cfg.attack_config(),
            caption_fn=self._caption_fn_for_suite,
            cider_fn=self.cider_fn,
            attack_cls=attack_cls,
        )
        return self._suite

    def _caption_fn_for_suite(self, pixels: torch.Tensor, *args: Any, **kwargs: Any) -> List[str]:
        out = self.captioner(pixels)
        if isinstance(out, str):
            return [out]
        return list(out)

    # -- main entry ---------------------------------------------------------
    def run(self, samples: Optional[Sequence[Dict[str, Any]]] = None) -> CaptioningEvalResult:
        started = time.time()
        cfg = self.cfg
        cfg.resolved_eps()

        if samples is None:
            samples = resolve_samples(cfg, cfg.num_samples)
        samples = [normalize_sample(s, i) for i, s in enumerate(samples)]
        if cfg.num_samples_clean:
            samples_clean = samples[: cfg.num_samples_clean]
        else:
            samples_clean = samples

        LOGGER.info(
            "Captioning eval: dataset=%s model=%s eps=%.6f (%.2f/255) threshold=%.2f samples=%d",
            cfg.dataset_name,
            cfg.model_name,
            cfg.resolved_eps(),
            cfg.resolved_eps() * 255.0,
            cfg.threshold,
            len(samples),
        )
        if cfg.verbose:
            for key, value in cfg.provenance.items():
                LOGGER.info("  [provenance] %s: %s", key, value)

        # ---- clean pass ----------------------------- (all/500 samples) ----
        clean_scores: List[float] = []
        captions: List[Dict[str, Any]] = []
        for idx, sample in enumerate(samples_clean):
            score, caption = self._clean_cider(sample["pixels"], sample["ground_truths"])
            clean_scores.append(score)
            captions.append({"sample_id": sample["sample_id"], "stage": "clean", "caption": caption, "cider": score})
        clean_cider = float(sum(clean_scores) / max(len(clean_scores), 1))

        # ---- adversarial pass (500 samples) -------------------------------
        suite = self._build_suite()
        result: CaptioningAttackResult = suite.attack_dataset(
            samples, caption_fn=self._caption_fn_for_suite, cider_fn=self.cider_fn, generator=self.generator
        )

        worst_per_sample = result.ciders()
        worst_cider = result.mean_worst_cider()

        # per-sample captions for the worst-case result
        for state in result.states:
            rec = None
            for r in reversed(state.records):
                if r.cider == state.worst_cider:
                    rec = r
                    break
            captions.append(
                {
                    "sample_id": getattr(state, "sample_index", len(captions)),
                    "stage": "worst_case",
                    "caption": state.worst_caption,
                    "cider": state.worst_cider,
                    "ground_truth": state.best_ground_truth,
                    "precision": state.best_precision,
                    "attack_stage": getattr(rec, "stage", None),
                }
            )

        perturbations = None
        if len(result.states) > 0:
            deltas = [s.best_perturbation for s in result.states if s.best_perturbation is not None]
            if deltas:
                try:
                    perturbations = torch.stack(deltas, dim=0)
                except Exception:  # pragma: no cover - shape mismatch
                    perturbations = None

        summary: Dict[str, Any] = {}
        try:
            summary = result.summary()
        except Exception:  # pragma: no cover
            summary = {"note": "captioning attack summary unavailable"}

        return CaptioningEvalResult(
            dataset_name=cfg.dataset_name,
            model_name=cfg.model_name,
            eps=cfg.resolved_eps(),
            threshold=float(cfg.threshold if cfg.threshold is not None else default_threshold(cfg.dataset_name)),
            num_samples=len(samples),
            clean_cider=clean_cider,
            worst_cider=worst_cider,
            clean_per_sample=clean_scores[: len(samples)],
            worst_per_sample=list(worst_per_sample),
            early_stopped_fraction=float(result.early_stopped_fraction()),
            precision_ok=bool(result.precision_ok()),
            captions=captions,
            perturbations=perturbations,
            attack_summary=summary,
            config=cfg.as_dict(),
            elapsed_seconds=time.time() - started,
        )


# ---------------------------------------------------------------------------
# smoke test
# ---------------------------------------------------------------------------


def smoke_attack_fn(cfg: CaptioningEvalConfig):
    """Model-free attack used by ``--smoke-test``.

    Produces a bounded perturbation (in mandated integer storage) around the raw
    pixels so that the CIDEr bookkeeping / early-stopping / warm-start logic can
    be exercised without a real victim model.
    """

    def attack_fn(request: Any) -> torch.Tensor:
        from .utils.precision import decode_perturbation, encode_perturbation

        pixels = request.pixels
        eps = float(request.eps)
        gen = torch.Generator().manual_seed(int(getattr(request, "seed", 0) or 0))
        delta = torch.zeros_like(pixels).uniform_(-eps, eps, generator=gen)
        if getattr(request, "init_delta", None) is not None:
            delta = decode_perturbation(request.init_delta, float_dtype=pixels.dtype).to(pixels.dtype)
        return encode_perturbation(delta, request.precision, quant_scale=cfg.quant_scale)

    return attack_fn


def dummy_cider_fn(*args: Any, **kwargs: Any) -> float:
    """Character-overlap proxy for CIDEr; smoke test only (not a paper metric)."""
    cand = kwargs.get("captions", args[0] if args else "")
    refs = kwargs.get("references", args[1] if len(args) > 1 else None)
    if isinstance(cand, (list, tuple)):
        cand = cand[0] if cand else ""
    if refs is None:
        refs = [""]
    if isinstance(refs, (list, tuple)) and refs and isinstance(refs[0], (list, tuple)):
        flat = [c for group in refs for c in group]
    elif isinstance(refs, (list, tuple)):
        flat = list(refs)
    else:
        flat = [str(refs)]
    cand = str(cand)
    cand_tokens = cand.split()
    best = 0.0
    for ref in flat:
        ref_tokens = set(str(ref).split())
        if not ref_tokens:
            continue
        overlap = sum(1 for t in cand_tokens if t in ref_tokens)
        best = max(best, overlap / max(len(ref_tokens), 1))
    return float(best)


def run_smoke_test(verbose: bool = True) -> Dict[str, Any]:
    """End-to-end orchestration check (no model, no heavy data deps)."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    cfg = CaptioningEvalConfig(
        dataset_name="COCO",
        model_name="dummy",
        eps=PAPER_EPSILONS[0],
        threshold=CAPTIONING_THRESHOLDS["COCO"],
        num_samples=4,
        smoke_test=True,
        verbose=verbose,
    )
    evaluator = CaptioningEvaluator(cfg, cider_fn=dummy_cider_fn, attack_fn=smoke_attack_fn(cfg))
    samples = synthetic_samples(4)
    res = evaluator.run(samples)

    assert res.precision_ok, "precision policy (int16/int32) violated"
    assert res.worst_cider <= res.clean_cider + 1e-6, (res.clean_cider, res.worst_cider)
    assert len(res.worst_per_sample) == len(samples)
    # worst-case retention: every sample's worst score is the min over its attacks
    suite_states = evaluator._build_suite()  # noqa: SLF001 - smoke test introspection
    LOGGER.info("Smoke test OK: clean CIDEr=%.4f worst-case CIDEr=%.4f", res.clean_cider, res.worst_cider)
    if verbose:
        print(json.dumps({k: v for k, v in res.as_dict().items() if k != "captions"}, indent=2, default=str))
    del suite_states
    return res.as_dict()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Robust CLIP captioning robustness evaluation (COCO/Flickr30k).")
    p.add_argument("--dataset", default=None, help="Captioning dataset (COCO / Flickr30k).")
    p.add_argument("--model", default=None, help="Victim LVLM: openflamingo | llava | dummy.")
    p.add_argument("--eps", type=float, default=None, help="l_inf budget (paper: 2/255, 4/255).")
    p.add_argument("--alpha", type=float, default=None, help="APGD initial step-size (defaults to eps, App. B.6).")
    p.add_argument("--iterations-half", type=int, default=None, help="Half-precision APGD iterations (paper: 100).")
    p.add_argument("--iterations-single", type=int, default=None, help="Single-precision APGD iterations.")
    p.add_argument("--threshold", type=float, default=None, help="CIDEr early-stopping threshold (paper: 10 COCO / 2 Flickr30k).")
    p.add_argument("--num-samples", type=int, default=None, help="Number of adversarial samples (paper: 500).")
    p.add_argument("--num-samples-clean", type=int, default=None, help="Clean-eval sample cap (paper: all).")
    p.add_argument("--num-ground-truths", type=int, default=None, help="Ground-truth captions per image (paper: 5).")
    p.add_argument("--no-warm-start", action="store_true", help="Do not warm-start the single-precision attack.")
    p.add_argument("--split", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--subsample-seed", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--config", default=None, help=f"YAML config (default: {DEFAULT_CONFIG_PATH}).")
    p.add_argument("--output", default=None, help="Where to write the JSON result.")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--no-perturbations", action="store_true", help="Do not store perturbations in the JSON.")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--smoke-test", action="store_true", help="Run the model-free orchestration check.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.smoke_test:
        run_smoke_test(verbose=not args.quiet)
        return 0

    cfg_path = args.config if args.config is not None else (str(DEFAULT_CONFIG_PATH) if DEFAULT_CONFIG_PATH.exists() else None)
    file_cfg = load_config(cfg_path)
    overrides: Dict[str, Any] = {}
    for key, value in vars(args).items():
        if key in ("config", "smoke_test", "no_perturbations", "quiet", "output", "output_dir") or value is None:
            continue
        overrides[key] = value
    if args.no_warm_start:
        overrides["warm_start_single"] = False
    cfg = CaptioningEvalConfig.from_dict(file_cfg, **overrides)
    if args.output_dir:
        cfg.output_dir = args.output_dir

    if cfg.eps is None:
        LOGGER.error(
            "--eps is required. The paper evaluates eps=2/255 and eps=4/255; "
            "the value is not fixed by the Addendum."
        )
        return 2

    evaluator = CaptioningEvaluator(cfg)
    res = evaluator.run()
    payload = res.as_dict(include_perturbations=not args.no_perturbations)

    out_path = args.output
    if out_path is None:
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        tag = f"{cfg.dataset_name}_{cfg.model_name}_eps{res.eps * 255:.0f}"
        out_path = str(Path(cfg.output_dir) / f"captioning_{tag}.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(
            {
                "summary": {
                    "dataset": res.dataset_name,
                    "model": res.model_name,
                    "eps": res.eps,
                    "threshold": res.threshold,
                    "clean_cider": res.clean_cider,
                    "worst_case_cider": res.worst_cider,
                    "num_samples": res.num_samples,
                    "early_stopped_fraction": res.early_stopped_fraction,
                    "precision_policy_ok": res.precision_ok,
                },
                "per_sample": res.worst_per_sample,
                "clean_per_sample": res.clean_per_sample,
                "attack_summary": res.attack_summary,
                "config": res.config,
            },
            fh,
            indent=2,
            default=str,
        )
    LOGGER.info("Wrote results to %s", out_path)
    print(f"{res.dataset_name} | {res.model_name} | eps={res.eps * 255:.2f}/255 | "
          f"clean CIDEr={res.clean_cider:.2f} | worst-case CIDEr={res.worst_cider:.2f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
