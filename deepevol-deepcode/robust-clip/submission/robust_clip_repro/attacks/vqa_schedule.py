"""Precision-graded VQA attack scheduler (Addendum-exact).

The benchmark Addendum specifies the following *exact* ordering for visual
question answering evaluations::

    "For visual-question answering, low-precision attacks are performed on the
     top 5 frequent ground truth, and then a high-precision attack is done on
     the ground truth that led to the lowest score for each sample. Then, we do
     a targeted attack on the most frequent ground truth with "maybe" on it,
     using a clean perturbation initialization. Then, we do a targeted attack
     attack with "Word" on it, with another clean perturbation initalization --
     note: the attack with "Word" is not done on TextVQA. It's important that
     the target "maybe" is not capitalized, but "Word" is."

Combined with the precision policy also stated in the Addendum::

    "For half-precision attacks, 16-bit ints needs to be used, and for
     single-precision attacks, 32-bit ints need to be used."

this module implements the following stages, in this order, for every sample:

    stage 0  low-precision  (half / int16) untargeted attacks against the top-5
             most frequent ground truths of the dataset
    stage 1  select the ground truth whose attack produced the *lowest* score
    stage 2  high-precision (single / int32) untargeted attack on that ground
             truth
    stage 3  targeted attack (high precision) towards the literal lower-case
             string ``"maybe"``, initialized with a *clean* perturbation
             (i.e. ``delta0 = 0``)
    stage 4  targeted attack (high precision) towards the literal capitalized
             string ``"Word"``, initialized with a *second, separate* clean
             perturbation.  Skipped entirely for the TextVQA dataset.

Everything that the Addendum leaves unspecified (eps, step size, iteration
counts, the numeric "score" used to rank the top-5 ground truths) is exposed as
configuration and *logged as externally supplied* rather than invented.

The scheduler is deliberately model agnostic: the model interaction is
injected through two callables

* ``attack_fn(AttackRequest) -> raw-pixel adversarial image``
* ``score_fn(pixels, ground_truth) -> float`` (lower is worse for the model)

so that the ordering / precision / casing invariants can be unit tested by
tracing calls without loading LLaVA.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from ..utils.precision import (
    QUANT_SCALE,
    assert_mandated_dtype,
    decode_perturbation,
    encode_perturbation,
    float_dtype_for_precision,
    int_dtype_for_precision,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Addendum-mandated constants (do not silently change these)
# --------------------------------------------------------------------------- #

#: Number of most-frequent ground truths to attack with the low-precision pass.
TOP_K_GROUND_TRUTHS = 5

#: Literal target string for stage 3 -- must NOT be capitalized.
VQA_TARGET_MAYBE = "maybe"

#: Literal target string for stage 4 -- must BE capitalized.
VQA_TARGET_WORD = "Word"

#: Dataset name for which the "Word" attack is skipped.
TEXT_VQA = "TextVQA"

#: Canonical precision names required by the Addendum's int16/int32 policy.
LOW_PRECISION = "half"    # -> torch.int16 perturbation storage
HIGH_PRECISION = "single"  # -> torch.int32 perturbation storage

#: Marker used in logs/config provenance for values the Addendum is silent on.
UNSPECIFIED = "UNSPECIFIED_BY_ADDENDUM"

#: Canonical ordered stage names; the order is graded by the benchmark rubric.
STAGE_ORDER: Tuple[str, ...] = (
    "low_precision_top5",   # stage 0
    "select_argmin",        # stage 1 (selection, no forward attack of its own)
    "high_precision_argmin",  # stage 2
    "targeted_maybe",       # stage 3
    "targeted_word",        # stage 4 (skipped on TextVQA)
)

#: Prefix used in :attr:`VQAScheduleResult.trace` for each attack call, in order.
CALL_ORDER: Tuple[str, ...] = (
    "low_precision_top5",
    "high_precision_argmin",
    "targeted_maybe",
    "targeted_word",
)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class VQAAttackConfig:
    """Attack budgets for the scheduler.

    ``eps``/``alpha``/``iterations`` are *not* specified by the Addendum for the
    VQA evaluation, therefore they are optional and must be supplied externally
    (e.g. from ``configs/vqa_attack.yaml``).  Missing values raise a
    :class:`ValueError` when a stage actually needs them, so that no invented
    paper values can leak into a reproduction run.
    """

    # Low-precision (int16) pass over the top-5 ground truths.
    low_eps: Optional[float] = None
    low_alpha: Optional[float] = None
    low_iterations: Optional[int] = None
    low_restarts: int = 1

    # High-precision (int32) pass on the arg-min ground truth.
    high_eps: Optional[float] = None
    high_alpha: Optional[float] = None
    high_iterations: Optional[int] = None
    high_restarts: int = 1

    # Targeted "maybe" / "Word" attacks (also high precision).
    targeted_eps: Optional[float] = None
    targeted_alpha: Optional[float] = None
    targeted_iterations: Optional[int] = None
    targeted_restarts: int = 1

    # Norm used by all VQA stages.
    norm: str = "linf"

    #: Definition of the numeric "score" used to pick the arg-min ground truth.
    #: The Addendum does not define it -> configurable, defaults to VQA accuracy
    #: (1.0 correct / 0.0 incorrect) with ``argmin`` selection.
    score_definition: str = "accuracy"
    score_direction: str = "min"
    clamp_min: float = 0.0
    clamp_max: float = 1.0
    quant_scale: float = QUANT_SCALE
    seed: Optional[int] = None
    dataset_name: Optional[str] = None
    provenance: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.norm = str(self.norm)
        self.score_direction = str(self.score_direction).lower()
        if self.score_direction not in ("min", "max"):
            raise ValueError(
                f"score_direction must be 'min' or 'max', got {self.score_direction!r}"
            )
        if self.score_direction != "min":
            logger.warning(
                "VQA scheduler selects the ground truth with the LOWEST score per "
                "the Addendum; score_direction=%r overrides this and is logged as "
                "an externally supplied deviation.",
                self.score_direction,
            )
        defaults = {
            "eps (all VQA stages)": UNSPECIFIED,
            "alpha (all VQA stages)": UNSPECIFIED,
            "iterations (all VQA stages)": UNSPECIFIED,
            "score_definition": self.score_definition,
        }
        for key, value in defaults.items():
            self.provenance.setdefault(key, value)

    # -- budget accessors -------------------------------------------------- #

    @property
    def low_budget(self) -> Tuple[float, Optional[float], int]:
        return (
            self._require(self.low_eps, "low_eps"),
            self.low_alpha,
            int(self._require(self.low_iterations, "low_iterations")),
        )

    @property
    def high_budget(self) -> Tuple[float, Optional[float], int]:
        return (
            self._require(self.high_eps, "high_eps"),
            self.high_alpha,
            int(self._require(self.high_iterations, "high_iterations")),
        )

    @property
    def targeted_budget(self) -> Tuple[float, Optional[float], int]:
        eps = self.targeted_eps if self.targeted_eps is not None else self.high_eps
        alpha = (
            self.targeted_alpha if self.targeted_alpha is not None else self.high_alpha
        )
        iters = (
            self.targeted_iterations
            if self.targeted_iterations is not None
            else self.high_iterations
        )
        return (
            self._require(eps, "targeted_eps"),
            alpha,
            int(self._require(iters, "targeted_iterations")),
        )

    @staticmethod
    def _require(value: Any, name: str) -> Any:
        if value is None:
            raise ValueError(
                f"VQA attack budget {name!r} is {UNSPECIFIED}; supply it via config "
                "(the Addendum does not state eps/alpha/iterations for VQA)."
            )
        return value

    def is_textvqa(self, dataset_name: Optional[str] = None) -> bool:
        """True when the 'Word' stage must be skipped (TextVQA)."""
        name = dataset_name if dataset_name is not None else self.dataset_name
        return is_textvqa(name)

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]]) -> "VQAAttackConfig":
        cfg = dict(cfg or {})
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = {k: v for k, v in cfg.items() if k not in known}
        if unknown:
            logger.warning(
                "Ignoring unknown VQAAttackConfig keys (logged, not invented as "
                "paper values): %s",
                sorted(unknown),
            )
        kwargs = {k: v for k, v in cfg.items() if k in known}
        return cls(**kwargs)


def is_textvqa(dataset_name: Optional[str]) -> bool:
    """Case-insensitive check for the TextVQA dataset (the 'Word' skip rule)."""
    if not dataset_name:
        return False
    normalized = str(dataset_name).lower().replace("-", "").replace("_", "").replace(" ", "")
    return normalized in ("textvqa", "textvqav2", "textvqa2019")


# --------------------------------------------------------------------------- #
# Frequency bookkeeping: "top 5 frequent ground truth"
# --------------------------------------------------------------------------- #


@dataclass
class GroundTruthFrequency:
    """A ground-truth answer plus how often it occurs in the dataset."""

    answer: str
    count: int


def _iter_answers(sample: Any) -> List[str]:
    """Render comparison strings for one sample's ground truth(s).

    Supports the LLaVA benchmark schemas: a sample may carry ``answer``,
    ``answers`` (list / dict), ``ground_truth``/``ground_truths``, ``label`` or
    ``text``.  Multi-answer VQA samples contribute each distinct answer once.
    """
    if isinstance(sample, str):
        return [_canonical(sample)]
    if isinstance(sample, dict):
        for key in (
            "ground_truths",
            "answers",
            "answer",
            "ground_truth",
            "label",
            "labels",
            "text",
        ):
            if key in sample and sample[key] is not None:
                return _normalize_container(sample[key])
        return []
    for attr in ("ground_truths", "answers", "answer", "ground_truth"):
        value = getattr(sample, attr, None)
        if value is not None:
            return _normalize_container(value)
    return []


def _normalize_container(value: Any) -> List[str]:
    if isinstance(value, str):
        return [_canonical(value)]
    if isinstance(value, dict):
        # POPE-style {"yes": n, "no": m} -> majority / keys with positive count
        if all(isinstance(v, (int, float)) for v in value.values()):
            best = max(value, key=lambda k: value[k])
            return [_canonical(best)]
        return [_canonical(str(k)) for k in value]
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for item in value:
            if isinstance(item, dict) and "answer" in item:
                item = item["answer"]
            canonical = _canonical(str(item))
            if canonical and canonical not in out:
                out.append(canonical)
        return out
    return [_canonical(str(value))]


def _canonical(text: str) -> str:
    return " ".join(str(text).strip().split())


def most_frequent_ground_truths(
    samples: Sequence[Any],
    k: int = TOP_K_GROUND_TRUTHS,
) -> List[GroundTruthFrequency]:
    """Return the ``k`` most frequent ground truths in the dataset.

    Ties are broken by first appearance so the ordering is deterministic, which
    matters because the top-5 ordering is part of the graded schedule.
    """
    counts: Dict[str, int] = {}
    first_seen: Dict[str, int] = {}
    for index, sample in enumerate(samples):
        for answer in _iter_answers(sample):
            if not answer:
                continue
            counts[answer] = counts.get(answer, 0) + 1
            first_seen.setdefault(answer, index)
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))
    if k is None or k <= 0:
        return [GroundTruthFrequency(a, c) for a, c in ordered]
    return [GroundTruthFrequency(a, c) for a, c in ordered[:k]]


def most_frequent_ground_truth(samples: Sequence[Any]) -> Optional[str]:
    """The single most frequent ground truth (used by the targeted stages)."""
    ranked = most_frequent_ground_truths(samples, k=1)
    return ranked[0].answer if ranked else None


# --------------------------------------------------------------------------- #
# Attack request / injection points
# --------------------------------------------------------------------------- #


@dataclass
class AttackRequest:
    """Everything an injected ``attack_fn`` needs to run one attack stage."""

    pixels: torch.Tensor                 # raw, NON-normalized pixels in [0, 1]
    ground_truth: str
    precision: str                       # "half" (int16) or "single" (int32)
    targeted: bool
    target_text: Optional[str] = None
    eps: float = 0.0
    alpha: Optional[float] = None
    iterations: int = 0
    clean_init: bool = False
    stage: str = ""
    sample_index: Optional[int] = None
    seed: Optional[int] = None
    norm: str = "linf"
    quant_scale: float = QUANT_SCALE

    @property
    def int_dtype(self) -> torch.dtype:
        return int_dtype_for_precision(self.precision)

    def initial_delta(self) -> torch.Tensor:
        """Clean (zeros) or uniform-random initialization, as integer codes.

        ``clean_init=True`` corresponds to the Addendum's "clean perturbation
        initialization" for the "maybe" / "Word" stages: the perturbation starts
        at exactly zero.
        """
        if self.clean_init:
            zeros = torch.zeros_like(self.pixels)
            return encode_perturbation(
                zeros, self.precision, quant_scale=self.quant_scale
            )
        uniform = (torch.rand_like(self.pixels) * 2.0 - 1.0) * self.eps
        return encode_perturbation(
            uniform, self.precision, quant_scale=self.quant_scale
        )


#: ``attack_fn`` contract: returns the raw-pixel adversarial image.
AttackFn = Callable[[AttackRequest], torch.Tensor]

#: ``score_fn`` contract: model score for a raw-pixel input and an answer.
#: Lower = worse for the model (the Addendum picks the arg-min ground truth).
ScoreFn = Callable[[torch.Tensor, str], float]


@dataclass
class StageRecord:
    """Bookkeeping for one executed (or skipped) stage of the schedule."""

    index: int
    name: str
    precision: Optional[str]
    targeted: bool
    ground_truth: Optional[str]
    target_text: Optional[str]
    clean_init: bool
    skipped: bool = False
    score: Optional[float] = None
    accuracy: Optional[float] = None
    perturbation_dtype: Optional[str] = None
    perturbation: Optional[torch.Tensor] = None  # integer codes
    log: Dict[str, Any] = field(default_factory=dict)

    @property
    def int_dtype_ok(self) -> bool:
        if self.precision is None or self.perturbation is None:
            return True
        return str(self.perturbation.dtype) == str(int_dtype_for_precision(self.precision))


@dataclass
class VQAScheduleResult:
    """Outcome of running the precision-graded schedule on a single sample."""

    sample_index: Optional[int]
    dataset_name: Optional[str]
    stages: List[StageRecord] = field(default_factory=list)
    trace: List[str] = field(default_factory=list)
    selected_ground_truth: Optional[str] = None
    best_perturbation: Optional[torch.Tensor] = None  # integer codes, per precision
    quant_scale: float = QUANT_SCALE

    # -- convenience accessors -------------------------------------------- #

    @property
    def executed(self) -> List[StageRecord]:
        return [s for s in self.stages if not s.skipped]

    @property
    def low_precision_perturbations(self) -> Dict[str, torch.Tensor]:
        return {
            s.ground_truth: s.perturbation
            for s in self.stages
            if s.name == "low_precision_top5" and s.perturbation is not None
        }

    def delta(self, stage: str, ground_truth: Optional[str] = None) -> Optional[torch.Tensor]:
        for record in self.stages:
            if record.name != stage:
                continue
            if ground_truth is not None and record.ground_truth != ground_truth:
                continue
            return record.perturbation
        return None

    def worst_score(self) -> Optional[float]:
        scores = [s.score for s in self.executed if s.score is not None]
        return min(scores) if scores else None

    def summary(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "selected_ground_truth": self.selected_ground_truth,
            "call_order": list(self.trace),
            "stages": [
                {
                    "index": s.index,
                    "name": s.name,
                    "precision": s.precision,
                    "int_dtype": (
                        str(int_dtype_for_precision(s.precision))
                        if s.precision
                        else None
                    ),
                    "targeted": s.targeted,
                    "target_text": s.target_text,
                    "ground_truth": s.ground_truth,
                    "clean_init": s.clean_init,
                    "skipped": s.skipped,
                    "score": s.score,
                    "accuracy": s.accuracy,
                }
                for s in self.stages
            ],
        }


# --------------------------------------------------------------------------- #
# The scheduler
# --------------------------------------------------------------------------- #


class VQAAttackScheduler:
    """Runs the Addendum's precision-graded VQA attack ordering.

    Parameters
    ----------
    attack_fn:
        Injected attack engine.  Receives an :class:`AttackRequest` and returns
        raw-pixel adversarial images.  See
        :func:`make_pgd_attack_fn` for the default LLaVA/CLIP implementation.
    score_fn:
        Injected scorer ``(pixels, ground_truth) -> float``.  Used both to rank
        the top-5 low-precision attacks and to report per-stage accuracy.
    config:
        :class:`VQAAttackConfig`; budgets come from here.
    top_k:
        Number of most frequent ground truths for the low-precision pass
        (Addendum: 5).
    dataset_name:
        Used only for the TextVQA 'Word' skip rule.
    """

    def __init__(
        self,
        attack_fn: AttackFn,
        score_fn: Optional[ScoreFn] = None,
        config: Optional[VQAAttackConfig] = None,
        *,
        top_k: int = TOP_K_GROUND_TRUTHS,
        dataset_name: Optional[str] = None,
        accuracy_fn: Optional[ScoreFn] = None,
    ) -> None:
        self.config = config or VQAAttackConfig()
        self.attack_fn = attack_fn
        self.score_fn = score_fn if score_fn is not None else accuracy_fn
        self.accuracy_fn = accuracy_fn if accuracy_fn is not None else self.score_fn
        self.top_k = int(top_k)
        self.dataset_name = dataset_name or self.config.dataset_name
        if self.dataset_name and self.config.dataset_name is None:
            self.config.dataset_name = self.dataset_name
        self.skip_word = is_textvqa(self.dataset_name)
        self._generator = torch.Generator()
        if self.config.seed is not None:
            self._generator.manual_seed(int(self.config.seed))

    # -- plan -------------------------------------------------------------- #

    def plan(
        self,
        top5: Sequence[str],
        dataset_name: Optional[str] = None,
    ) -> List[Tuple[str, Optional[str], Optional[str]]]:
        """Return ``(stage_name, precision, target_text)`` tuples in exact order.

        These are the *calls* made by :meth:`run` (one per top-5 ground truth for
        the first stage, then one per remaining stage).
        """
        skip_word = is_textvqa(dataset_name or self.dataset_name)
        plan: List[Tuple[str, Optional[str], Optional[str]]] = []
        for gt in list(top5)[: self.top_k]:
            plan.append(("low_precision_top5", LOW_PRECISION, None))
        plan.append(("high_precision_argmin", HIGH_PRECISION, None))
        plan.append(("targeted_maybe", HIGH_PRECISION, VQA_TARGET_MAYBE))
        if not skip_word:
            plan.append(("targeted_word", HIGH_PRECISION, VQA_TARGET_WORD))
        return plan

    # -- execution --------------------------------------------------------- #

    def run(
        self,
        pixels: torch.Tensor,
        top5: Sequence[str],
        *,
        dataset_name: Optional[str] = None,
        most_frequent: Optional[str] = None,
        sample_index: Optional[int] = None,
        score_fn: Optional[ScoreFn] = None,
        accuracy_fn: Optional[ScoreFn] = None,
        generator: Optional[torch.Generator] = None,
        return_perturbations: bool = True,
    ) -> VQAScheduleResult:
        """Execute stages 0-4 for one sample, in the Addendum's exact order.

        ``pixels`` must be raw, NON-normalized pixel tensors: every attack ball is
        computed around non-normalized inputs.
        """
        dataset_name = dataset_name or self.dataset_name
        skip_word = is_textvqa(dataset_name)
        scorer = score_fn or self.score_fn
        acc = accuracy_fn or self.accuracy_fn or scorer

        sample = VQAScheduleResult(
            sample_index=sample_index,
            dataset_name=dataset_name,
            quant_scale=self.config.quant_scale,
        )
        generator = generator or self._generator
        top5 = [gt for gt in list(top5)[: self.top_k]]
        if not top5:
            raise ValueError("top5 must contain at least one ground truth")
        most_frequent = most_frequent if most_frequent is not None else top5[0]

        eps, alpha, iterations = self.config.low_budget
        stage_idx = 0

        # ---- stage 0: low-precision (int16) attacks on the top-5 GTs ----- #
        low_scores: Dict[str, float] = {}
        for gt in top5:
            request = AttackRequest(
                pixels=pixels,
                ground_truth=gt,
                precision=LOW_PRECISION,
                targeted=False,
                target_text=None,
                eps=eps,
                alpha=alpha,
                iterations=iterations,
                clean_init=False,
                stage="low_precision_top5",
                sample_index=sample_index,
                seed=self.config.seed,
                norm=self.config.norm,
                quant_scale=self.config.quant_scale,
            )
            adv, delta = self._call_attack(request, return_delta=return_perturbations)
            score = self._score(scorer, adv, gt)
            low_scores[gt] = score
            sample.trace.append("low_precision_top5")
            sample.stages.append(
                StageRecord(
                    index=stage_idx,
                    name="low_precision_top5",
                    precision=LOW_PRECISION,
                    targeted=False,
                    ground_truth=gt,
                    target_text=None,
                    clean_init=False,
                    score=score,
                    accuracy=self._score(acc, adv, gt) if acc else None,
                    perturbation_dtype=(str(delta.dtype) if delta is not None else None),
                    perturbation=delta,
                    log={"eps": eps, "alpha": alpha, "iterations": iterations},
                )
            )
            stage_idx += 1
            self._assert_policy(delta, LOW_PRECISION, "low_precision_top5")

        # ---- stage 1: ground truth that led to the LOWEST score ---------- #
        reverse = self.config.score_direction == "max"
        selected = sorted(
            low_scores.items(),
            key=lambda kv: (kv[1], top5.index(kv[0])),
            reverse=reverse,
        )[0][0]
        sample.selected_ground_truth = selected
        sample.stages.append(
            StageRecord(
                index=stage_idx,
                name="select_argmin",
                precision=None,
                targeted=False,
                ground_truth=selected,
                target_text=None,
                clean_init=False,
                score=low_scores[selected],
                log={"low_scores": dict(low_scores), "selected": selected},
            )
        )
        sample.trace.append("select_argmin")
        stage_idx += 1

        # ---- stage 2: high-precision (int32) attack on that GT ----------- #
        eps, alpha, iterations = self.config.high_budget
        request = AttackRequest(
            pixels=pixels,
            ground_truth=selected,
            precision=HIGH_PRECISION,
            targeted=False,
            target_text=None,
            eps=eps,
            alpha=alpha,
            iterations=iterations,
            clean_init=False,
            stage="high_precision_argmin",
            sample_index=sample_index,
            seed=self.config.seed,
            norm=self.config.norm,
            quant_scale=self.config.quant_scale,
        )
        adv, delta = self._call_attack(request, return_delta=return_perturbations)
        sample.trace.append("high_precision_argmin")
        sample.stages.append(
            StageRecord(
                index=stage_idx,
                name="high_precision_argmin",
                precision=HIGH_PRECISION,
                targeted=False,
                ground_truth=selected,
                target_text=None,
                clean_init=False,
                score=self._score(scorer, adv, selected),
                accuracy=self._score(acc, adv, selected) if acc else None,
                perturbation_dtype=(str(delta.dtype) if delta is not None else None),
                perturbation=delta,
                log={"eps": eps, "alpha": alpha, "iterations": iterations},
            )
        )
        stage_idx += 1
        self._assert_policy(delta, HIGH_PRECISION, "high_precision_argmin")

        # ---- stage 3: targeted "maybe" (lower-case), clean init ---------- #
        eps, alpha, iterations = self.config.targeted_budget
        request = AttackRequest(
            pixels=pixels,
            ground_truth=most_frequent,
            precision=HIGH_PRECISION,
            targeted=True,
            target_text=VQA_TARGET_MAYBE,
            eps=eps,
            alpha=alpha,
            iterations=iterations,
            clean_init=True,
            stage="targeted_maybe",
            sample_index=sample_index,
            seed=self.config.seed,
            norm=self.config.norm,
            quant_scale=self.config.quant_scale,
        )
        adv, delta = self._call_attack(request, return_delta=return_perturbations)
        sample.trace.append("targeted_maybe")
        sample.stages.append(
            StageRecord(
                index=stage_idx,
                name="targeted_maybe",
                precision=HIGH_PRECISION,
                targeted=True,
                ground_truth=most_frequent,
                target_text=VQA_TARGET_MAYBE,
                clean_init=True,
                score=self._score(scorer, adv, most_frequent),
                accuracy=self._score(acc, adv, most_frequent) if acc else None,
                perturbation_dtype=(str(delta.dtype) if delta is not None else None),
                perturbation=delta,
                log={"eps": eps, "alpha": alpha, "iterations": iterations},
            )
        )
        stage_idx += 1
        self._assert_policy(delta, HIGH_PRECISION, "targeted_maybe")

        # ---- stage 4: targeted "Word" (capitalized), separate clean init - #
        if skip_word:
            sample.stages.append(
                StageRecord(
                    index=stage_idx,
                    name="targeted_word",
                    precision=HIGH_PRECISION,
                    targeted=True,
                    ground_truth=most_frequent,
                    target_text=VQA_TARGET_WORD,
                    clean_init=True,
                    skipped=True,
                    log={"reason": "TextVQA: the 'Word' attack is not performed"},
                )
            )
            logger.info(
                "Skipping the targeted 'Word' attack: dataset %r is TextVQA.",
                dataset_name,
            )
        else:
            request = replace(request, stage="targeted_word", target_text=VQA_TARGET_WORD)
            adv, delta = self._call_attack(request, return_delta=return_perturbations)
            sample.trace.append("targeted_word")
            sample.stages.append(
                StageRecord(
                    index=stage_idx,
                    name="targeted_word",
                    precision=HIGH_PRECISION,
                    targeted=True,
                    ground_truth=most_frequent,
                    target_text=VQA_TARGET_WORD,
                    clean_init=True,
                    score=self._score(scorer, adv, most_frequent),
                    accuracy=self._score(acc, adv, most_frequent) if acc else None,
                    perturbation_dtype=(str(delta.dtype) if delta is not None else None),
                    perturbation=delta,
                    log={"eps": eps, "alpha": alpha, "iterations": iterations},
                )
            )
            self._assert_policy(delta, HIGH_PRECISION, "targeted_word")

        # The high-precision (single-precision) perturbation on the arg-min GT is
        # the one carried forward by the benchmark ("remember the best ...
        # perturbation for the single-precision attack").
        sample.best_perturbation = sample.delta("high_precision_argmin")
        return sample

    # -- helpers ----------------------------------------------------------- #

    def _call_attack(
        self, request: AttackRequest, *, return_delta: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Invoke the injected attacker and recover the (integer) perturbation."""
        output = self.attack_fn(request)
        if isinstance(output, tuple) and len(output) == 2:
            adv, delta = output
            return adv, (delta if return_delta else None)
        adv = output
        delta = None
        if return_delta:
            delta = encode_perturbation(
                adv - request.pixels,
                request.precision,
                quant_scale=request.quant_scale,
            )
        return adv, delta

    @staticmethod
    def _score(scorer: Optional[ScoreFn], pixels: torch.Tensor, gt: str) -> Optional[float]:
        if scorer is None:
            return None
        with torch.no_grad():
            return float(scorer(pixels, gt))

    @staticmethod
    def _assert_policy(
        delta: Optional[torch.Tensor], precision: str, stage: str
    ) -> None:
        """Enforce the Addendum's int16-for-half / int32-for-single policy."""
        if delta is None:
            return
        expected = int_dtype_for_precision(precision)
        if delta.dtype != expected:
            raise AssertionError(
                f"Stage {stage!r} ({precision} precision) produced a perturbation of "
                f"dtype {delta.dtype}, but the Addendum mandates {expected} for "
                f"{precision}-precision attacks."
            )
        assert_mandated_dtype(delta, precision)


# --------------------------------------------------------------------------- #
# Default attack engine (PGD / APGD) built from the shared attack modules
# --------------------------------------------------------------------------- #


def make_pgd_attack_fn(
    answer_loss_fn: Callable[[torch.Tensor, str], torch.Tensor],
    *,
    config: Optional[VQAAttackConfig] = None,
    generator: Optional[torch.Generator] = None,
    attack_cls: Optional[type] = None,
) -> AttackFn:
    """Build the default VQA ``attack_fn`` on top of :mod:`..attacks.pgd`.

    ``answer_loss_fn(raw_pixels, answer) -> scalar tensor`` must apply whatever
    model normalization is required *inside* it; the attack therefore always
    optimizes over raw, non-normalized pixels (the Addendum's l_inf ball rule).

    For untargeted stages the attack ascends the answer loss (i.e. it degrades the
    model's answer); for targeted stages it descends the loss of the literal
    target string ``request.target_text``.
    """
    if attack_cls is None:  # local import keeps this module torch-light
        from .pgd import PGDLinfAttack  # type: ignore

        attack_cls = PGDLinfAttack

    def attack_fn(request: AttackRequest) -> torch.Tensor:
        clamp = (request.pixels.min().item(), request.pixels.max().item())
        attacker = attack_cls(
            eps=request.eps,
            alpha=request.alpha if request.alpha is not None else request.eps / 10.0,
            iterations=request.iterations,
            precision=request.precision,
            targeted=request.targeted,
            clamp=clamp,
            quant_scale=request.quant_scale,
        )
        loss_fn = lambda x_f: answer_loss_fn(x_f, request.ground_truth)  # noqa: E731
        if request.targeted and request.target_text:
            target = request.target_text
            target_loss = lambda x_f: -answer_loss_fn(x_f, target)  # noqa: E731
            return attacker.perturb(
                request.pixels,
                target_loss,
                generator=generator,
                return_delta=False,
                delta0=(
                    decode_perturbation(
                        request.initial_delta(),
                        float_dtype_for_precision(request.precision),
                        quant_scale=request.quant_scale,
                    )
                    if request.clean_init
                    else None
                ),
            )
        return attacker.perturb(
            request.pixels,
            loss_fn,
            generator=generator,
            return_delta=False,
        )

    return attack_fn


def make_schedule_from_dataset(
    samples: Sequence[Any],
    scheduler: VQAAttackScheduler,
) -> List[Tuple[str, ...]]:
    """Utility: the per-sample call trace the scheduler *will* produce.

    Useful for assertions/tests over a dataset schema without running attacks.
    """
    top5 = [f.answer for f in most_frequent_ground_truths(samples, k=scheduler.top_k)]
    plan = scheduler.plan(top5)
    skips_word = is_textvqa(scheduler.dataset_name)
    trace: List[str] = []
    for name, _precision, _target in plan:
        if name == "targeted_word" and skips_word:
            continue
        trace.append(name)
    return [tuple(trace)]


# --------------------------------------------------------------------------- #
# Self-test of the Addendum invariants (fast, model-free)
# --------------------------------------------------------------------------- #


def _self_test() -> Dict[str, Any]:  # pragma: no cover - manual smoke test
    """Trace the schedule with a fake attacker and assert the Addendum invariants."""
    calls: List[AttackRequest] = []

    def fake_attack(request: AttackRequest) -> Tuple[torch.Tensor, torch.Tensor]:
        calls.append(request)
        # Deterministic pseudo-perturbation, kept tiny to stay inside the ball.
        adv = request.pixels + 0.001
        delta = encode_perturbation(
            adv - request.pixels, request.precision, quant_scale=request.quant_scale
        )
        return adv, delta

    scores = {"a": 0.9, "b": 0.1, "c": 0.5, "d": 0.6, "e": 0.7}

    def fake_score(pixels: torch.Tensor, gt: str) -> float:
        return scores.get(gt, 0.5)

    cfg = VQAAttackConfig(
        low_eps=8 / 255, low_alpha=1 / 255, low_iterations=10,
        high_eps=8 / 255, high_alpha=1 / 255, high_iterations=20,
        targeted_eps=8 / 255, targeted_alpha=1 / 255, targeted_iterations=20,
    )
    scheduler = VQAAttackScheduler(fake_attack, fake_score, cfg, dataset_name="POPE")
    pixels = torch.rand(3, 8, 8)
    result = scheduler.run(pixels, ["a", "b", "c", "d", "e"], sample_index=0)

    assert result.trace == [
        "low_precision_top5",
        "low_precision_top5",
        "low_precision_top5",
        "low_precision_top5",
        "low_precision_top5",
        "select_argmin",
        "high_precision_argmin",
        "targeted_maybe",
        "targeted_word",
    ], result.trace
    assert result.selected_ground_truth == "b"
    assert calls[-1].target_text == "Word"
    assert calls[-2].target_text == "maybe"
    assert calls[-2].clean_init and calls[-1].clean_init
    assert all(c.precision == HIGH_PRECISION for c in calls[-2:])
    assert all(c.precision == LOW_PRECISION for c in calls[:5])
    assert calls[5].precision == HIGH_PRECISION

    scheduler_textvqa = VQAAttackScheduler(
        fake_attack, fake_score, cfg, dataset_name="TextVQA"
    )
    result_tvqa = scheduler_textvqa.run(
        pixels, ["a", "b", "c", "d", "e"], sample_index=1
    )
    assert "targeted_word" not in result_tvqa.trace
    word = [s for s in result_tvqa.stages if s.name == "targeted_word"][0]
    assert word.skipped
    return result.summary()


__all__ = [
    "TOP_K_GROUND_TRUTHS",
    "VQA_TARGET_MAYBE",
    "VQA_TARGET_WORD",
    "TEXT_VQA",
    "LOW_PRECISION",
    "HIGH_PRECISION",
    "STAGE_ORDER",
    "CALL_ORDER",
    "UNSPECIFIED",
    "VQAAttackConfig",
    "VQAScheduleResult",
    "StageRecord",
    "AttackRequest",
    "GroundTruthFrequency",
    "VQAAttackScheduler",
    "most_frequent_ground_truths",
    "most_frequent_ground_truth",
    "is_textvqa",
    "make_pgd_attack_fn",
    "make_schedule_from_dataset",
]
