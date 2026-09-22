"""The attack ensemble used for the LVLM robustness evaluation (Sec. 4.1, App. B.6).

Quoting App. B.6 of the paper:

* captioning (COCO, Flickr30k): *"For the captioning tasks COCO and Flickr30k
  there are five ground truth captions available for each image and each is
  considered for computation of the CIDEr score. We conduct APGD attacks at half
  precision with 100 iterations against each ground-truth. After each attack we
  compute the CIDEr scores and do not attack the samples anymore that already
  have a score below 10 or 2 for COCO and Flickr30k respectively. [...] In the
  final step we employ a similar attack at single precision, using the
  ground-truth that led to the lowest score and initialize it with the according
  perturbation."*
* VQA (VQAv2, TextVQA): *"we use a similar scheme, however the score-threshold
  is set to 0 and we use the five most frequent ground-truths among the ten
  available ones. Additionally, we employ targeted attacks at single precision
  with target strings "Maybe" and "Word". For TextVQA it was observed that the
  second targeted attack is not necessary, thus we apply only the first one."*
* *"Following Schlarmann & Hein (2023), we set the initial step-size of APGD to
  eps."*

The two stages differ in the *numeric precision of the perturbation*: the first
stage runs at half precision (16-bit integers), the second one at single
precision (32-bit integers), see the addendum and
:mod:`robust_clip.utils.precision`.

Because the evaluation only needs the *worst* score of every sample, the CIDEr /
accuracy score is computed after every attack and the best (for the attacker)
perturbation is kept together with the ground truth that produced it.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from ..models.lvlm.base import LVLM
from ..utils.precision import eps_to_int
from .apgd import APGDAttack

LOGGER = logging.getLogger(__name__)

__all__ = ["EnsembleAttackConfig", "LVLMEnsembleAttack", "AttackResult", "most_frequent"]


@contextmanager
def model_precision(model, dtype: torch.dtype):
    """Run the model in ``dtype`` (the attack precision) and restore it afterwards.

    The two stages of the ensemble attack differ in the precision in which the
    *model* is evaluated: the first stage runs the LVLM in half precision
    (16-bit integer perturbations), the second one in single precision (32-bit
    integer perturbations).  Casting the (frozen) model is exactly what the
    paper's evaluation does, since the networks themselves are unchanged.
    """
    try:
        original = next(model.parameters()).dtype
    except StopIteration:  # pragma: no cover - defensive
        original = dtype
    if dtype != original:
        model.to(dtype)
        if hasattr(model, "dtype"):
            model.dtype = dtype
    try:
        yield model
    finally:
        if dtype != original:
            model.to(original)
            if hasattr(model, "dtype"):
                model.dtype = original


def most_frequent(values: Sequence[str], k: int) -> List[str]:
    """The ``k`` most frequent strings (ties broken by first occurrence)."""
    order: List[str] = []
    counts: Dict[str, int] = {}
    for value in values:
        if value not in counts:
            order.append(value)
            counts[value] = 0
        counts[value] += 1
    ranked = sorted(order, key=lambda v: (-counts[v], order.index(v)))
    return ranked[:k]


@dataclass
class EnsembleAttackConfig:
    """Configuration of the ensemble attack (App. B.6)."""

    eps: str = "2/255"
    half_precision_iters: int = 100
    single_precision_iters: int = 100
    num_ground_truths: int = 5  # 5 captions (COCO/Flickr30k) / 5 most frequent answers (VQA)
    score_threshold: float = 10.0  # 10 (COCO), 2 (Flickr30k), 0 (VQA)
    single_precision_on_best_gt: bool = True
    targeted_vqa: bool = True
    #: target strings of the targeted VQA attacks: "maybe" (not capitalized) and
    #: "Word" (capitalized), App. B.6 / addendum
    vqa_targets: Tuple[str, ...] = ("maybe", "Word")
    targeted_use_word: bool = True  # "the attack with 'Word' is not done on TextVQA"
    targeted_clean_init: bool = True  # targeted attacks use a clean perturbation initialization
    random_start: bool = True
    max_new_tokens: int = 32
    num_beams: int = 1


@dataclass
class AttackResult:
    """Result of the ensemble attack on one batch."""

    x_adv: torch.Tensor  # worst case perturbation per sample
    scores: List[float]  # worst case score per sample (CIDEr / accuracy)
    clean_scores: List[float]
    best_gt_index: List[int]  # index of the ground truth that led to the worst score
    stage: List[str]  # which stage produced the worst case ('half'/'single'/'targeted')
    meta: Dict[str, float] = field(default_factory=dict)


class LVLMEnsembleAttack:
    """Precision aware ensemble attack (half precision -> single precision -> targeted)."""

    def __init__(
        self,
        model: LVLM,
        config: Optional[EnsembleAttackConfig] = None,
        metric_fn: Optional[Callable[[Sequence[str], Sequence[Sequence[str]]], Sequence[float]]] = None,
    ):
        self.model = model
        self.config = config or EnsembleAttackConfig()
        #: ``metric_fn(generations, references) -> per-sample score`` (CIDEr or accuracy)
        self.metric_fn = metric_fn or (lambda generations, references: [0.0] * len(generations))
        try:
            self._device_type = next(model.parameters()).device.type
        except StopIteration:  # pragma: no cover - defensive
            self._device_type = "cpu"

    def _effective_dtype(self, dtype: torch.dtype) -> torch.dtype:
        """Half precision requires a GPU (several fp16 kernels are missing on CPU).

        On CPU the half precision stage is therefore executed in single precision
        (with a warning); the evaluations of the paper run on GPUs, where the
        first stage of the ensemble is a genuine 16-bit attack.
        """
        if dtype == torch.float16 and self._device_type == "cpu":
            if not getattr(self, "_half_precision_warned", False):
                LOGGER.warning(
                    "half precision kernels are not available on CPU - running the first stage in fp32"
                )
                self._half_precision_warned = True
            return torch.float32
        return dtype

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _score(self, images: torch.Tensor, prompts: Sequence[str], references: Sequence[Sequence[str]]) -> List[float]:
        generations = self.model.generate(
            images,
            prompts,
            max_new_tokens=self.config.max_new_tokens,
            num_beams=self.config.num_beams,
        )
        return list(self.metric_fn(generations, references))

    def _run_stage(
        self,
        images: torch.Tensor,
        prompts: Sequence[str],
        targets: Sequence[str],
        dtype: torch.dtype,
        n_iter: int,
        x_init: Optional[torch.Tensor] = None,
        targeted: bool = False,
    ) -> torch.Tensor:
        """One APGD run on the token-level loss of ``targets``.

        ``forward_fn`` is the identity because :meth:`LVLM.target_loss` already
        re-encodes the image; the loss is negated for the targeted attacks since
        APGD always maximizes the returned loss.
        """
        dtype = self._effective_dtype(dtype)
        attacker = APGDAttack(
            eps=self.config.eps,
            n_iter=n_iter,
            n_restarts=1,
            loss="custom",
            dtype=dtype,
            alpha_init=self.config.eps,
        )

        def forward_fn(x_adv):
            return x_adv

        def loss_fn(_, x_adv):
            nll = self.model.target_loss(x_adv, prompts, targets, reduction="none")
            return -nll if targeted else nll

        with model_precision(self.model, dtype):
            return attacker.attack(images, forward_fn, custom_loss=loss_fn, x_init=x_init)

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------
    def run(
        self,
        images: torch.Tensor,
        prompts: Sequence[str],
        references: Sequence[Sequence[str]],
        is_vqa: bool = False,
    ) -> AttackResult:
        """Attack a batch of ``[0, 1]`` images.

        Parameters
        ----------
        references:
            for captioning the five ground truth captions, for VQA the ten
            ground truth answers of each sample.
        """
        cfg = self.config
        batch_size = images.shape[0]
        device = images.device

        clean_scores = self._score(images, prompts, references)
        best_x = images.detach().clone()
        best_scores = list(clean_scores)
        best_gt = [-1] * batch_size
        best_stage = ["clean"] * batch_size

        # ---- stage 1: half precision, one APGD run per ground truth ----------
        targets_per_sample: List[List[str]] = []
        for refs in references:
            if is_vqa:
                candidates = most_frequent(refs, cfg.num_ground_truths)
            else:
                candidates = list(refs)[: cfg.num_ground_truths]
            while len(candidates) < cfg.num_ground_truths:  # pad if fewer references
                candidates.append(candidates[-1] if candidates else "")
            targets_per_sample.append(candidates)

        active = torch.ones(batch_size, dtype=torch.bool, device=device)
        half_gt_x = images.detach().clone()
        half_gt_index = [0] * batch_size
        half_gt_score = list(clean_scores)

        for gt_index in range(cfg.num_ground_truths):
            targets = [t[gt_index] for t in targets_per_sample]
            # each ground truth gets its own attack (random start, no warm start)
            x_candidate = self._run_stage(
                images,
                prompts,
                targets,
                dtype=torch.float16,
                n_iter=cfg.half_precision_iters,
                x_init=None if cfg.random_start else images,
            )
            scores = self._score(x_candidate, prompts, references)
            improved = torch.tensor(
                [s < b for s, b in zip(scores, best_scores)], dtype=torch.bool, device=device
            ) & active
            for i in range(batch_size):
                if bool(improved[i]):
                    best_scores[i] = scores[i]
                    best_x[i] = x_candidate[i]
                    best_gt[i] = gt_index
                    best_stage[i] = "half"
                if scores[i] < half_gt_score[i]:
                    half_gt_score[i] = scores[i]
                    half_gt_x[i] = x_candidate[i]
                    half_gt_index[i] = gt_index
            active = active & torch.tensor(
                [s >= cfg.score_threshold for s in best_scores], dtype=torch.bool, device=device
            )
            if not bool(active.any()):
                break

        # ---- stage 2: single precision, warm started on the best ground truth
        if cfg.single_precision_on_best_gt:
            targets = [targets_per_sample[i][half_gt_index[i]] for i in range(batch_size)]
            x_candidate = self._run_stage(
                images,
                prompts,
                targets,
                dtype=torch.float32,
                n_iter=cfg.single_precision_iters,
                x_init=half_gt_x,
            )
            scores = self._score(x_candidate, prompts, references)
            for i in range(batch_size):
                if scores[i] < best_scores[i]:
                    best_scores[i] = scores[i]
                    best_x[i] = x_candidate[i]
                    best_gt[i] = half_gt_index[i]
                    best_stage[i] = "single"

        # ---- stage 3: targeted attacks on the VQA tasks ---------------------
        if is_vqa and cfg.targeted_vqa:
            targets_to_run = list(cfg.vqa_targets)
            if not cfg.targeted_use_word and "Word" in targets_to_run:
                targets_to_run.remove("Word")
            for target_string in targets_to_run:
                targets = [target_string] * batch_size
                x_candidate = self._run_stage(
                    images,
                    prompts,
                    targets,
                    dtype=torch.float32,
                    n_iter=cfg.single_precision_iters,
                    x_init=images if cfg.targeted_clean_init else best_x,
                    targeted=True,
                )
                scores = self._score(x_candidate, prompts, references)
                for i in range(batch_size):
                    if scores[i] < best_scores[i]:
                        best_scores[i] = scores[i]
                        best_x[i] = x_candidate[i]
                        best_gt[i] = -1
                        best_stage[i] = f"targeted:{target_string}"

        return AttackResult(
            x_adv=best_x,
            scores=best_scores,
            clean_scores=list(clean_scores),
            best_gt_index=best_gt,
            stage=best_stage,
            meta={"eps_int": eps_to_int(cfg.eps), "is_vqa": float(is_vqa)},
        )
