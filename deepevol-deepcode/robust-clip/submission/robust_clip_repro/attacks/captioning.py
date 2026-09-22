"""Captioning attack suite with per-attack CIDEr bookkeeping (Robust CLIP).

Specification sources
---------------------
* Paper body, Sec. 4.1 ("Attack setup", "Datasets and metrics"):

  "We first conduct APGD attacks ... at half precision with 100 iterations,
   using several groundtruth captions/answers as labels. After each attack, we
   do not attack samples whose score is already below a threshold anymore. In
   the final step we employ a similar attack at single precision."

  "We use a variety of image captioning (COCO, Flickr30k) ... tasks ... For
   all these tasks, we use 500 randomly sampled images for the adversarial
   evaluations ... We report the CIDEr score for captioning."

* Paper body, App. B.6 (Untargeted Attack Details):

  "For the captioning tasks COCO and Flickr30k there are five ground truth
   captions available for each image and each is considered for computation of
   the CIDEr score. We conduct APGD attacks at half precision with 100
   iterations against each ground-truth. After each attack we compute the
   CIDEr scores and do not attack the samples anymore that already have a
   score below 10 or 2 for COCO and Flickr30k respectively. These thresholds
   correspond to less than 10% of the original LLaVA performance. ... In the
   final step we employ a similar attack at single precision, using the
   ground-truth that led to the lowest score and initialize it with the
   according perturbation. ... Following Schlarmann & Hein (2023), we set the
   initial step-size of APGD to epsilon."

* Addendum (in scope, benchmark authors' clarifications):

  "For computation of the CIDEr scores, they compute the CIDEr scores after
   every attack, so that they can take the worst case score for each sample,
   and remember the best ground-truth and perturbation for the single-precision
   attack."

  "For half-precision attacks, 16-bit ints needs to be used, and for
   single-precision attacks, 32-bit ints need to be used."

  "The APGD algorithm is taken from https://github.com/fra31/robust-finetuning."

  "For ... perturbation strengths of eps = 2/255 and eps = 4/255."

Everything the paper does not state (attack norm beyond l_inf, the exact
captioning prompts, how the caption loss is formed) is exposed as
configuration with an explicit "unspecified" marker rather than being
invented.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from ..utils.precision import (
    QUANT_SCALE,
    assert_mandated_dtype,
    decode_perturbation,
    encode_perturbation,
    float_dtype_for_precision,
    int_dtype_for_precision,
)

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paper-stated constants
# ---------------------------------------------------------------------------

#: Number of ground-truth captions per image (App. B.6).
NUM_GROUND_TRUTHS: int = 5

#: Half-precision APGD iteration budget (App. B.6, Sec. 4.1).
HALF_PRECISION_ITERATIONS: int = 100

#: CIDEr early-stopping thresholds (App. B.6). 10 for COCO, 2 for Flickr30k.
CAPTIONING_THRESHOLDS: Dict[str, float] = {
    "COCO": 10.0,
    "Flickr30k": 2.0,
}

#: Perturbation strengths evaluated in Sec. 4.1.
PAPER_EPSILONS: Tuple[float, ...] = (2.0 / 255.0, 4.0 / 255.0)

#: Precision names for the two stages (Addendum: int16 for half, int32 for
#: single).
HALF_PRECISION: str = "half"
SINGLE_PRECISION: str = "single"

#: Marker used for every quantity the paper/Addendum does not state.
UNSPECIFIED = "UNSPECIFIED_BY_PAPER"

#: Stage / record names.
STAGE_HALF_PRECISION = "half_precision_gt"
STAGE_SINGLE_PRECISION = "single_precision_best_gt"

#: Number of adversarial samples used in Sec. 4.1 (500 randomly sampled images).
PAPER_ATTACK_SAMPLES: int = 500

SUPPORTED_DATASETS: Tuple[str, ...] = ("COCO", "Flickr30k")

AttackFn = Callable[["CaptioningAttackRequest"], torch.Tensor]
CaptionFn = Callable[[torch.Tensor], str]
CiderFn = Callable[[str, Sequence[str]], float]
CaptionLossFn = Callable[[torch.Tensor], torch.Tensor]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CaptioningAttackConfig:
    """Configuration of the captioning attack pipeline.

    Values with a source comment come verbatim from the paper/Addendum; every
    other field is defaulted and marked as externally supplied.
    """

    # --- dataset / task -----------------------------------------------------
    dataset_name: str = "COCO"
    num_ground_truths: int = NUM_GROUND_TRUTHS          # App. B.6
    #: CIDEr threshold below which a sample is no longer attacked (App. B.6).
    threshold: Optional[float] = None

    # --- perturbation budget ------------------------------------------------
    #: Sec. 4.1 evaluates eps in {2/255, 4/255}; no single value is mandated.
    eps: Optional[float] = None
    #: App. B.6 / Schlarmann & Hein (2023): initial step-size == eps.
    alpha: Optional[float] = None

    # --- stage budgets ------------------------------------------------------
    iterations_half: int = HALF_PRECISION_ITERATIONS    # App. B.6: 100
    iterations_single: int = HALF_PRECISION_ITERATIONS
    restarts_half: int = 1
    restarts_single: int = 1
    precision_half: str = HALF_PRECISION
    precision_single: str = SINGLE_PRECISION

    # --- attack details -----------------------------------------------------
    norm: str = "linf"
    loss: str = "ce"
    random_start: bool = True
    #: Warm-start the single-precision stage with the best perturbation found
    #: during the half-precision stage (App. B.6).
    warm_start_single: bool = True
    score_for_selection: str = "cider_min"              # worst-case selection
    higher_is_better: bool = True                       # CIDEr is a maximised metric

    # --- precision policy (Addendum) ---------------------------------------
    quant_scale: float = QUANT_SCALE

    # --- bookkeeping --------------------------------------------------------
    seed: Optional[int] = None
    device: Optional[str] = None
    clamp_min: float = 0.0
    clamp_max: float = 1.0
    provenance: Dict[str, str] = field(default_factory=dict)
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.threshold is None:
            key = _canonical_dataset(self.dataset_name)
            if key in CAPTIONING_THRESHOLDS:
                self.threshold = CAPTIONING_THRESHOLDS[key]
            else:
                raise ValueError(
                    f"Unknown dataset {self.dataset_name!r}: the CIDEr threshold "
                    f"is not stated by the paper. Known: "
                    f"{sorted(CAPTIONING_THRESHOLDS)}; pass threshold explicitly."
                )
        if self.alpha is None:
            if self.eps is None:
                raise ValueError(
                    "eps is not stated by the paper for arbitrary budgets; "
                    "supply eps (Sec. 4.1 evaluates 2/255 and 4/255)."
                )
            # App. B.6: "we set the initial step-size of APGD to eps".
            self.alpha = float(self.eps)
        if self.eps is None:
            raise ValueError("eps must be supplied (paper does not fix one value).")
        if self.norm not in ("linf", "l2"):
            raise ValueError(f"Unsupported norm {self.norm!r}.")
        if self.num_ground_truths < 1:
            raise ValueError("num_ground_truths must be >= 1.")
        if self.precision_half != HALF_PRECISION:
            raise ValueError(
                "Addendum mandates half-precision attacks (int16); "
                f"got precision_half={self.precision_half!r}."
            )
        if self.precision_single != SINGLE_PRECISION:
            raise ValueError(
                "Addendum mandates single-precision attacks (int32); "
                f"got precision_single={self.precision_single!r}."
            )
        self.provenance.setdefault("iterations_half", "App. B.6 / Sec. 4.1 (100)")
        self.provenance.setdefault("num_ground_truths", "App. B.6 (5)")
        self.provenance.setdefault("threshold", "App. B.6 (COCO 10, Flickr30k 2)")
        self.provenance.setdefault("alpha", "App. B.6 (alpha = eps)")
        self.provenance.setdefault("warm_start_single", "App. B.6")
        self.provenance.setdefault("precision_policy", "Addendum (int16 half / int32 single)")
        self.provenance.setdefault("cidEr_after_every_attack", "Addendum")
        self.provenance.setdefault(
            "eps", "Sec. 4.1 evaluates 2/255 and 4/255 (no single mandated value)"
        )
        self.provenance.setdefault(
            "caption_loss", UNSPECIFIED + " (caption/answer loss not stated)"
        )
        self.provenance.setdefault("restarts", UNSPECIFIED + " (defaults to 1)")

    # -- convenience --------------------------------------------------------
    @property
    def int_dtype_half(self) -> torch.dtype:
        return int_dtype_for_precision(self.precision_half)

    @property
    def int_dtype_single(self) -> torch.dtype:
        return int_dtype_for_precision(self.precision_single)

    def budget(self, precision: str) -> Tuple[float, float, int, int]:
        """Return ``(eps, alpha, iterations, restarts)`` for a stage."""
        if precision == self.precision_half:
            return self.eps, self.alpha, self.iterations_half, self.restarts_half
        return self.eps, self.alpha, self.iterations_single, self.restarts_single

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, object]] = None, **overrides) -> "CaptioningAttackConfig":
        cfg = dict(cfg or {})
        cfg.update(overrides)
        unknown = {k: cfg.pop(k) for k in list(cfg) if k not in cls.__dataclass_fields__}
        if unknown:
            LOGGER.warning("Ignoring unknown captioning config keys: %s", sorted(unknown))
        return cls(**cfg)  # type: ignore[arg-type]

    def as_dict(self) -> Dict[str, object]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _canonical_dataset(name: str) -> str:
    key = str(name).strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    if key in ("coco", "mscoco", "coco2014", "cocokarpathy"):
        return "COCO"
    if key in ("flickr30k", "flickr30kentities", "flickr"):
        return "Flickr30k"
    return str(name)


# ---------------------------------------------------------------------------
# Request / bookkeeping dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CaptioningAttackRequest:
    """One attack stage handed to an injected ``attack_fn``."""

    pixels: torch.Tensor                 # raw, non-normalized pixels [1, C, H, W]
    ground_truth: str                    # caption used as the attack label
    ground_truths: Sequence[str] = field(default_factory=tuple)
    precision: str = SINGLE_PRECISION
    eps: float = 0.0
    alpha: float = 0.0
    iterations: int = 100
    restarts: int = 1
    stage: str = STAGE_HALF_PRECISION
    sample_index: int = 0
    #: Warm-start perturbation in *integer codes* or ``None`` for a fresh init.
    init_delta: Optional[torch.Tensor] = None
    clean_init: bool = False
    targeted: bool = False
    target_text: Optional[str] = None
    norm: str = "linf"
    random_start: bool = True
    quant_scale: float = QUANT_SCALE
    seed: Optional[int] = None

    @property
    def int_dtype(self) -> torch.dtype:
        return int_dtype_for_precision(self.precision)

    @property
    def float_dtype(self) -> torch.dtype:
        return float_dtype_for_precision(self.precision)

    def initial_delta(self, *, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Return the int-coded starting perturbation.

        Warm start (``init_delta``) takes precedence, then a clean (zero)
        initialization, then uniform random inside the epsilon ball.
        """
        if self.init_delta is not None:
            return self.init_delta.to(self.int_dtype)
        if self.clean_init or not self.random_start:
            delta = torch.zeros_like(self.pixels, dtype=torch.float32)
        else:
            delta = (torch.rand(self.pixels.shape, generator=generator,
                                device=self.pixels.device, dtype=torch.float32) * 2 - 1) * self.eps
        return encode_perturbation(delta, self.precision, quant_scale=self.quant_scale)


@dataclass
class CaptioningAttackRecord:
    """Outcome of a single attack + CIDEr computation."""

    stage: str
    ground_truth: str
    precision: str
    iteration_index: int
    cider: float
    caption: str
    perturbation_dtype: Optional[torch.dtype] = None
    warm_started: bool = False
    iteration_budget: Optional[int] = None
    log: Dict[str, object] = field(default_factory=dict)

    @property
    def int_dtype_ok(self) -> bool:
        if self.perturbation_dtype is None:
            return True
        return self.perturbation_dtype in (torch.int16, torch.int32)


@dataclass
class CaptioningSampleState:
    """Per-sample worst-case bookkeeping (Addendum requirement)."""

    sample_index: int
    ground_truths: List[str]
    worst_cider: float = float("inf")
    worst_caption: str = ""
    best_ground_truth: Optional[str] = None
    #: int-coded perturbation that achieved the worst (lowest) CIDEr so far.
    best_perturbation: Optional[torch.Tensor] = None
    best_precision: Optional[str] = None
    stopped_early: bool = False
    records: List[CaptioningAttackRecord] = field(default_factory=list)
    single_precision_state: Optional[Dict[str, object]] = None
    quant_scale: float = QUANT_SCALE

    # -- updates ------------------------------------------------------------
    def update(self, cider: float, *, caption: str, ground_truth: str,
               perturbation: torch.Tensor, precision: str,
               iteration_index: int = 0, stage: str = STAGE_HALF_PRECISION,
               warm_started: bool = False,
               iteration_budget: Optional[int] = None) -> bool:
        """Record one CIDEr result; keeps the worst case per sample.

        Returns ``True`` when this attack improved (lowered) the worst CIDEr.
        """
        record = CaptioningAttackRecord(
            stage=stage,
            ground_truth=ground_truth,
            precision=precision,
            iteration_index=iteration_index,
            cider=float(cider),
            caption=caption,
            perturbation_dtype=perturbation.dtype,
            warm_started=warm_started,
            iteration_budget=iteration_budget,
        )
        self.records.append(record)
        improved = float(cider) < self.worst_cider
        if improved:
            self.worst_cider = float(cider)
            self.worst_caption = caption
            self.best_ground_truth = ground_truth
            self.best_perturbation = perturbation.detach().clone()
            self.best_precision = precision
        if stage == STAGE_SINGLE_PRECISION:
            self.single_precision_state = {
                "cider": float(cider),
                "ground_truth": ground_truth,
                "perturbation": perturbation.detach().clone(),
                "caption": caption,
            }
        return improved

    def below_threshold(self, threshold: Optional[float]) -> bool:
        if threshold is None:
            return False
        if self.worst_cider < threshold:
            self.stopped_early = True
        return self.stopped_early

    def adversarial_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        if self.best_perturbation is None:
            return pixels
        delta = decode_perturbation(
            self.best_perturbation,
            float_dtype=torch.float32,
            quant_scale=self.quant_scale,
        )
        return (pixels + delta).clamp(0.0, 1.0)

    def as_dict(self) -> Dict[str, object]:
        return {
            "sample_index": self.sample_index,
            "ground_truths": list(self.ground_truths),
            "worst_cider": self.worst_cider,
            "best_ground_truth": self.best_ground_truth,
            "best_precision": self.best_precision,
            "stopped_early": self.stopped_early,
            "num_attacks": len(self.records),
            "per_attack_cider": [r.cider for r in self.records],
        }


@dataclass
class CaptioningAttackResult:
    """Aggregate result of a captioning attack run."""

    states: List[CaptioningSampleState]
    config: CaptioningAttackConfig
    dataset_name: str = "COCO"

    # -- reporting ----------------------------------------------------------
    def mean_worst_cider(self) -> float:
        if not self.states:
            return float("nan")
        return sum(s.worst_cider for s in self.states) / len(self.states)

    def ciders(self) -> List[float]:
        return [s.worst_cider for s in self.states]

    def early_stopped_fraction(self) -> float:
        if not self.states:
            return float("nan")
        return sum(1.0 for s in self.states if s.stopped_early) / len(self.states)

    def precision_ok(self) -> bool:
        return all(r.int_dtype_ok for s in self.states for r in s.records)

    def summary(self) -> Dict[str, object]:
        return {
            "dataset": self.dataset_name,
            "eps": self.config.eps,
            "alpha": self.config.alpha,
            "threshold": self.config.threshold,
            "num_samples": len(self.states),
            "mean_worst_cider": self.mean_worst_cider(),
            "early_stopped_fraction": self.early_stopped_fraction(),
            "precision_policy_ok": self.precision_ok(),
            "provenance": dict(self.config.provenance),
        }


# ---------------------------------------------------------------------------
# The attack suite
# ---------------------------------------------------------------------------


class CaptioningAttackSuite:
    """Runs the paper's captioning attack pipeline with per-attack CIDEr.

    The suite is model agnostic: captioning is injected via ``caption_fn`` and
    scoring via ``cider_fn``, while the actual perturbation is produced by
    ``attack_fn`` (see :func:`make_apgd_attack_fn` / :func:`make_pgd_attack_fn`).
    CIDEr is recomputed immediately after **every** attack and only the worst
    per-sample value is retained (Addendum).
    """

    def __init__(
        self,
        attack_fn: AttackFn,
        config: Optional[CaptioningAttackConfig] = None,
        *,
        caption_fn: Optional[CaptionFn] = None,
        cider_fn: Optional[CiderFn] = None,
        attack_cls=None,
    ) -> None:
        self.config = config or CaptioningAttackConfig()
        self.attack_fn = attack_fn
        self.caption_fn = caption_fn
        self.cider_fn = cider_fn
        self.attack_cls = attack_cls

    # -- planning -----------------------------------------------------------
    def plan(self, num_ground_truths: Optional[int] = None) -> List[Tuple[str, str, Optional[str]]]:
        """Return the ordered stage plan ``(stage, precision, target_text)``."""
        n = self.config.num_ground_truths if num_ground_truths is None else num_ground_truths
        stages: List[Tuple[str, str, Optional[str]]] = [
            (STAGE_HALF_PRECISION, self.config.precision_half, None) for _ in range(n)
        ]
        stages.append((STAGE_SINGLE_PRECISION, self.config.precision_single, None))
        return stages

    # -- single sample ------------------------------------------------------
    def attack_sample(
        self,
        pixels: torch.Tensor,
        ground_truths: Sequence[str],
        *,
        sample_index: int = 0,
        caption_fn: Optional[CaptionFn] = None,
        cider_fn: Optional[CiderFn] = None,
        generator: Optional[torch.Generator] = None,
        return_state: bool = False,
    ):
        """Run the half-precision stage then the warm-started single-precision stage.

        For every ground truth the sample is attacked at half precision, a
        caption is generated, CIDEr is computed immediately, and the worst case
        is retained. Attacking stops as soon as the CIDEr falls below the
        dataset threshold. Finally a single-precision attack reuses the
        ground truth (and perturbation) that gave the lowest CIDEr.
        """
        cfg = self.config
        caption_fn = caption_fn or self.caption_fn
        cider_fn = cider_fn or self.cider_fn
        if caption_fn is None or cider_fn is None:
            raise ValueError("caption_fn and cider_fn must be provided.")

        gts = list(ground_truths)[: cfg.num_ground_truths] or list(ground_truths)
        state = CaptioningSampleState(
            sample_index=sample_index,
            ground_truths=gts,
            quant_scale=cfg.quant_scale,
        )

        # ---- stage 1: half-precision (int16), untargeted, one GT at a time
        eps, alpha, iters, restarts = cfg.budget(cfg.precision_half)
        for k, gt in enumerate(gts):
            request = CaptioningAttackRequest(
                pixels=pixels,
                ground_truth=gt,
                ground_truths=gts,
                precision=cfg.precision_half,
                eps=eps,
                alpha=alpha,
                iterations=iters,
                restarts=restarts,
                stage=STAGE_HALF_PRECISION,
                sample_index=sample_index,
                clean_init=False,
                random_start=cfg.random_start,
                norm=cfg.norm,
                quant_scale=cfg.quant_scale,
                seed=cfg.seed,
            )
            delta = self._call_attack(request, generator=generator)
            self._assert_policy(delta, cfg.precision_half)
            caption = caption_fn(self._adv(pixels, delta))
            cider = float(cider_fn(caption, gts))
            state.update(
                cider,
                caption=caption,
                ground_truth=gt,
                perturbation=delta,
                precision=cfg.precision_half,
                iteration_index=k,
                stage=STAGE_HALF_PRECISION,
                iteration_budget=iters,
            )
            if state.below_threshold(cfg.threshold):
                break

        # ---- stage 2: single precision (int32), warm start from best GT ----
        if state.best_ground_truth is not None:
            eps, alpha, iters, restarts = cfg.budget(cfg.precision_single)
            init_delta = state.best_perturbation if cfg.warm_start_single else None
            request = CaptioningAttackRequest(
                pixels=pixels,
                ground_truth=state.best_ground_truth,
                ground_truths=gts,
                precision=cfg.precision_single,
                eps=eps,
                alpha=alpha,
                iterations=iters,
                restarts=restarts,
                stage=STAGE_SINGLE_PRECISION,
                sample_index=sample_index,
                init_delta=init_delta,
                clean_init=init_delta is None,
                random_start=cfg.random_start,
                norm=cfg.norm,
                quant_scale=cfg.quant_scale,
                seed=cfg.seed,
            )
            delta = self._call_attack(request, generator=generator)
            self._assert_policy(delta, cfg.precision_single)
            caption = caption_fn(self._adv(pixels, delta))
            cider = float(cider_fn(caption, gts))
            state.update(
                cider,
                caption=caption,
                ground_truth=state.best_ground_truth,
                perturbation=delta,
                precision=cfg.precision_single,
                iteration_index=len(gts),
                stage=STAGE_SINGLE_PRECISION,
                warm_started=init_delta is not None,
                iteration_budget=iters,
            )

        if cfg.verbose:
            LOGGER.info(
                "captioning sample %d: worst CIDEr %.3f (gt=%r, %d attacks%s)",
                sample_index,
                state.worst_cider,
                state.best_ground_truth,
                len(state.records),
                ", early stop" if state.stopped_early else "",
            )
        return state if return_state else state.adversarial_pixels(pixels)

    # -- dataset ------------------------------------------------------------
    def attack_dataset(
        self,
        samples: Iterable[Dict[str, object]],
        *,
        caption_fn: Optional[CaptionFn] = None,
        cider_fn: Optional[CiderFn] = None,
        generator: Optional[torch.Generator] = None,
    ) -> CaptioningAttackResult:
        """Attack a full captioning dataset, returning worst-case CIDEr bookkeeping.

        ``samples`` yields dicts with keys ``image`` (raw pixels [1,C,H,W] or
        [C,H,W]), ``captions`` (sequence of references) and optional ``id``.
        """
        cfg = self.config
        key = _canonical_dataset(cfg.dataset_name)
        states: List[CaptioningSampleState] = []
        for i, sample in enumerate(samples):
            pixels = self._as_pixels(sample["image"])
            gts = [str(c) for c in sample.get("captions") or sample.get("ground_truths") or []]
            state = self.attack_sample(
                pixels,
                gts,
                sample_index=int(sample.get("id", i) or i),
                caption_fn=caption_fn,
                cider_fn=cider_fn,
                generator=generator,
                return_state=True,
            )
            states.append(state)  # type: ignore[arg-type]
        return CaptioningAttackResult(states=states, config=cfg, dataset_name=key)

    # -- helpers ------------------------------------------------------------
    def _call_attack(self, request: CaptioningAttackRequest,
                     *, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        try:
            delta = self.attack_fn(request)  # type: ignore[call-arg]
        except TypeError:
            delta = self.attack_fn(request, generator=generator)  # type: ignore[call-arg]
        if not isinstance(delta, torch.Tensor):
            raise TypeError("attack_fn must return an integer-coded perturbation tensor")
        return delta

    @staticmethod
    def _adv(pixels: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        adv = pixels + decode_perturbation(delta, float_dtype=torch.float32)
        return adv.clamp(0.0, 1.0)

    @staticmethod
    def _as_pixels(image) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            pixels = image
        else:  # PIL / numpy
            import numpy as np

            arr = np.asarray(image, dtype="float32")
            if arr.max() > 1.5:
                arr = arr / 255.0
            pixels = torch.from_numpy(arr)
            if pixels.ndim == 3 and pixels.shape[-1] in (1, 3):
                pixels = pixels.permute(2, 0, 1)
        pixels = pixels.float()
        if pixels.ndim == 3:
            pixels = pixels.unsqueeze(0)
        return pixels

    @staticmethod
    def _assert_policy(delta: torch.Tensor, precision: str) -> None:
        expected = int_dtype_for_precision(precision)
        if delta.dtype != expected:
            raise AssertionError(
                f"Addendum precision policy violated: {precision} attack produced "
                f"{delta.dtype}, expected {expected}."
            )
        assert_mandated_dtype(delta, precision)


# ---------------------------------------------------------------------------
# Cider bootstrap
# ---------------------------------------------------------------------------


def make_cider_fn(references_key: str = "captions"):
    """Return a CIDEr function backed by :mod:`metrics.cider` (lazy import)."""
    def _cider(hypothesis: str, references: Sequence[str]) -> float:
        from ..metrics.cider import cider_score

        return float(cider_score(hypothesis, references))

    return _cider


def default_threshold(dataset_name: str) -> float:
    key = _canonical_dataset(dataset_name)
    if key not in CAPTIONING_THRESHOLDS:
        raise ValueError(
            f"No paper-stated CIDEr threshold for {dataset_name!r}; "
            f"known datasets: {sorted(CAPTIONING_THRESHOLDS)}."
        )
    return CAPTIONING_THRESHOLDS[key]


# ---------------------------------------------------------------------------
# Attack-function builders
# ---------------------------------------------------------------------------


def make_apgd_attack_fn(
    loss_fn_factory: Callable[[str], CaptionLossFn],
    *,
    config: Optional[CaptioningAttackConfig] = None,
    generator: Optional[torch.Generator] = None,
    attack_cls=None,
) -> AttackFn:
    """Build the default ``attack_fn`` on top of the upstream APGD attack.

    ``loss_fn_factory(ground_truth) -> loss_fn`` must return a callable mapping
    raw pixels (with model normalization applied internally) to a scalar loss
    that is **maximised** for the untargeted captioning attack.
    """
    cfg = config or CaptioningAttackConfig()

    def attack_fn(request: CaptioningAttackRequest, generator=None) -> torch.Tensor:
        if attack_cls is not None:
            attacker_cls = attack_cls
        else:
            from .apgd import APGDAttack

            attacker_cls = APGDAttack
        eps, alpha, iters, restarts = (
            request.eps,
            request.alpha,
            request.iterations,
            request.restarts,
        )
        attacker = attacker_cls(
            eps=eps,
            alpha=alpha,
            iterations=iters,
            restarts=restarts,
            precision=request.precision,
            norm=request.norm,
            loss=cfg.loss,
            random_start=request.random_start,
            quant_scale=request.quant_scale,
            clamp=(cfg.clamp_min, cfg.clamp_max),
            seed=request.seed,
        )
        delta0 = request.initial_delta(generator=generator)
        loss_fn = loss_fn_factory(request.ground_truth)
        out = attacker.perturb(
            request.pixels,
            loss_fn,
            delta0=delta0,
            return_delta=True,
            generator=generator,
        )
        return out

    return attack_fn


def make_pgd_attack_fn(
    loss_fn_factory: Callable[[str], CaptionLossFn],
    *,
    config: Optional[CaptioningAttackConfig] = None,
    generator: Optional[torch.Generator] = None,
    attack_cls=None,
) -> AttackFn:
    """Fallback ``attack_fn`` built on the Addendum-exact PGD implementation."""
    cfg = config or CaptioningAttackConfig()

    def attack_fn(request: CaptioningAttackRequest, generator=None) -> torch.Tensor:
        if attack_cls is not None:
            attacker_cls = attack_cls
        else:
            from .pgd import PGDLinfAttack

            attacker_cls = PGDLinfAttack
        attacker = attacker_cls(
            eps=request.eps,
            alpha=request.alpha,
            iterations=request.iterations,
            precision=request.precision,
            random_start=request.random_start,
            quant_scale=request.quant_scale,
            clamp=(cfg.clamp_min, cfg.clamp_max),
        )
        loss_fn = loss_fn_factory(request.ground_truth)
        out = attacker.perturb(
            request.pixels,
            loss_fn,
            delta0=request.initial_delta(generator=generator),
            return_delta=True,
            generator=generator,
        )
        return out

    return attack_fn


def build_caption_loss_fn(
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
    tokenizer,
    ground_truth: str,
    *,
    reduction: str = "mean",
    ignore_index: int = -100,
) -> CaptionLossFn:
    """Untargeted teacher-forced caption loss over raw pixels.

    ``logits_fn(pixels) -> logits`` of shape ``[B, T, V]`` (T >= len(tokens)).
    The returned loss is the cross-entropy of the ground-truth caption tokens;
    attacks maximise it.

    Note: the exact captioning prompt/loss is not specified by the paper, so
    this follows the standard teacher-forced objective used for captioning
    attacks and is logged as externally supplied.
    """
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    ids = tok(ground_truth, return_tensors="pt", add_special_tokens=False).input_ids
    ids = ids.to(dtype=torch.long)

    def loss_fn(pixels: torch.Tensor) -> torch.Tensor:
        logits = logits_fn(pixels)
        if logits.ndim == 2:  # single-step logits
            logits = logits.unsqueeze(1)
        t = min(logits.shape[1], ids.shape[1])
        import torch.nn.functional as F

        target = ids[:, :t].to(logits.device).expand(logits.shape[0], -1)
        return F.cross_entropy(
            logits[:, :t].transpose(1, 2),
            target,
            reduction=reduction,
            ignore_index=ignore_index,
        )

    return loss_fn


# ---------------------------------------------------------------------------
# Functional convenience
# ---------------------------------------------------------------------------


def run_captioning_attack(
    samples: Iterable[Dict[str, object]],
    attack_fn: AttackFn,
    *,
    config: Optional[CaptioningAttackConfig] = None,
    caption_fn: Optional[CaptionFn] = None,
    cider_fn: Optional[CiderFn] = None,
    dataset_name: Optional[str] = None,
    eps: Optional[float] = None,
) -> CaptioningAttackResult:
    """Run the full captioning attack pipeline over ``samples``."""
    if config is None:
        config = CaptioningAttackConfig(
            dataset_name=dataset_name or "COCO",
            eps=eps,
        )
    elif dataset_name is not None or eps is not None:
        config = replace(
            config,
            dataset_name=dataset_name or config.dataset_name,
            eps=eps if eps is not None else config.eps,
        )
        config.__post_init__()
    suite = CaptioningAttackSuite(attack_fn, config, caption_fn=caption_fn, cider_fn=cider_fn)
    return suite.attack_dataset(samples, caption_fn=caption_fn, cider_fn=cider_fn)


# ---------------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------------


def _self_test() -> None:
    """Smoke test: CIDEr after every attack, worst case retained, warm start."""
    logging.basicConfig(level=logging.WARNING)
    cfg = CaptioningAttackConfig(dataset_name="COCO", eps=2.0 / 255.0)
    assert cfg.threshold == 10.0
    assert abs(cfg.alpha - 2.0 / 255.0) < 1e-12
    assert cfg.int_dtype_half == torch.int16
    assert cfg.int_dtype_single == torch.int32

    calls: List[Tuple[str, str]] = []
    cider_calls: List[str] = []

    def attack_fn(request: CaptioningAttackRequest) -> torch.Tensor:
        calls.append((request.stage, request.precision))
        delta = request.initial_delta()
        assert delta.dtype == request.int_dtype
        # Deterministic pseudo-improvement so the worst case is well defined.
        return delta * 0

    ciders = {"gt0": 60.0, "gt1": 40.0, "gt2": 5.0}
    state_box: Dict[str, int] = {"k": 0}
    seq = [65.0, 45.0, 4.0, 3.0]

    def caption_fn(pixels: torch.Tensor) -> str:
        i = state_box["k"]
        state_box["k"] += 1
        return f"caption{i}"

    def cider_fn(caption: str, refs) -> float:
        cider_calls.append(caption)
        return seq[min(len(cider_calls) - 1, len(seq) - 1)]

    suite = CaptioningAttackSuite(attack_fn, cfg, caption_fn=caption_fn, cider_fn=cider_fn)
    state = suite.attack_sample(
        torch.zeros(1, 3, 8, 8),
        ["a cat", "two dogs", "on grass", "outside", "green"],
        sample_index=0,
        return_state=True,
    )
    # Early stopping: CIDEr 4.0 < 10 stops after the third half-precision attack.
    assert len(calls) >= 4, calls
    half_calls = [c for c in calls if c[0] == STAGE_HALF_PRECISION]
    assert len(half_calls) == 3, half_calls
    assert calls[-1][0] == STAGE_SINGLE_PRECISION
    assert calls[-1][1] == "single"
    # CIDEr computed after every attack.
    assert len(cider_calls) == len(state.records)
    # Worst case retained, not the last value.
    assert state.worst_cider == min(seq[: len(state.records)])
    assert state.best_ground_truth == "on grass"
    assert state.best_perturbation is not None
    assert state.best_perturbation.dtype == torch.int16
    assert state.stopped_early
    assert state.single_precision_state is not None
    assert state.single_precision_state["perturbation"].dtype == torch.int32
    print("captioning self-test OK:", state.as_dict())


if __name__ == "__main__":  # pragma: no cover
    _self_test()
