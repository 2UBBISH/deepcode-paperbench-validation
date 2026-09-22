"""The attack pipeline used against the LVLMs (Sec. 4.1, 4.2 and App. B.6-B.9).

Untargeted evaluation (App. B.6)
--------------------------------
* **Captioning** (COCO, Flickr30k): there are five ground-truth captions per
  image.  APGD is run **at half precision** for 100 iterations against each of
  them.  After every attack the CIDEr scores are computed and samples whose
  score has dropped below a threshold (10 for COCO, 2 for Flickr30k, i.e. below
  ~10% of the original LLaVA score) are not attacked any further.  The worst
  CIDEr score and the corresponding ground truth / perturbation are
  remembered.  Finally a single-precision attack with 100 iterations is run on
  the ground truth that led to the lowest score, initialised with the
  perturbation from the half-precision stage.
* **VQA** (VQAv2, TextVQA): the same scheme with the five most frequent ground
  truth answers, a score threshold of 0 and, in addition, **targeted** attacks
  at single precision with the target strings ``"maybe"`` (not capitalised) and
  ``"Word"`` (capitalised).  The ``"Word"`` attack is not run on TextVQA.

The initial step size of APGD is set to :math:`\\varepsilon` following
Schlarmann & Hein (2023), the momentum is 0.9 and the gradient is normalised
with the elementwise sign (addendum).  Half-precision attacks use 16-bit
integer perturbations, single-precision attacks 32-bit ones.

Stealthy targeted attacks (Sec. 4.2)
------------------------------------
APGD with **10 000 iterations** optimising the exact target caption; the attack
counts as successful if the target string appears verbatim in the output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from ..lvlm.base import LVLM
from ..utils.common import LOGGER
from .apgd import APGDAttack


@dataclass
class UntargetedResult:
    images: torch.Tensor                      # worst-case adversarial images
    scores: torch.Tensor                      # worst-case score per sample
    outputs: List[str]                        # model outputs for those images
    perturbations: List[Dict[str, object]] = field(default_factory=list)


class LVLMAttackPipeline:
    """The ensemble (half precision -> single precision) attack of Sec. 4.1."""

    def __init__(
        self,
        lvlm: LVLM,
        eps: float,
        n_iter: int = 100,
        alpha: Optional[float] = None,
        momentum: float = 0.9,
        grad_normalization: str = "elementwise_sign",
        step_schedule: bool = True,
        random_init: bool = True,
        half_dtype: torch.dtype = torch.float16,
        single_dtype: torch.dtype = torch.float32,
        half_bits: int = 16,
        single_bits: int = 32,
        max_new_tokens: int = 32,
        use_half_precision_stage: bool = True,
        use_single_precision_stage: bool = True,
    ):
        self.lvlm = lvlm
        self.eps = float(eps)
        self.n_iter = int(n_iter)
        self.alpha = float(alpha) if alpha is not None else float(eps)
        self.momentum = momentum
        self.grad_normalization = grad_normalization
        self.step_schedule = step_schedule
        self.random_init = random_init
        self.half_dtype = half_dtype
        self.single_dtype = single_dtype
        device = getattr(lvlm, "device", None)
        if half_dtype == torch.float16 and device is not None and device.type == "cpu":
            # fp16 is not supported on CPU: run the "half-precision" stage in
            # fp32 but keep the 16-bit integer perturbation grid, which is what
            # makes the first stage cheaper/weaker than the second one.
            LOGGER.info("CPU detected: running the half-precision stage in fp32")
            self.half_dtype = torch.float32
        self.half_bits = half_bits
        self.single_bits = single_bits
        self.max_new_tokens = max_new_tokens
        self.use_half_precision_stage = use_half_precision_stage
        self.use_single_precision_stage = use_single_precision_stage

    # ------------------------------------------------------------------ utils
    def _make_attack(self, dtype: torch.dtype, bits: int, n_iter: Optional[int] = None,
                     alpha: Optional[float] = None, random_init: Optional[bool] = None) -> APGDAttack:
        return APGDAttack(
            eps=self.eps,
            n_iter=n_iter or self.n_iter,
            alpha=alpha if alpha is not None else self.alpha,
            momentum=self.momentum,
            grad_normalization=self.grad_normalization,
            step_schedule=self.step_schedule,
            random_init=self.random_init if random_init is None else random_init,
            quantize_bits=bits,
        )

    def _attack_subset(
        self,
        x: torch.Tensor,
        active: torch.Tensor,
        prompts: Sequence[str],
        continuations: Sequence[str],
        dtype: torch.dtype,
        bits: int,
        x_init: Optional[torch.Tensor] = None,
        maximize: bool = True,
        random_init: Optional[bool] = None,
        n_iter: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Attack the samples flagged by ``active``; others are returned unchanged."""
        idx = torch.nonzero(active, as_tuple=False).squeeze(1)
        out = x.detach().clone()
        losses = torch.zeros(x.shape[0], device=x.device)
        if idx.numel() == 0:
            return out, losses

        self.lvlm.set_precision(dtype)
        sub_prompts = [prompts[i] for i in idx.tolist()]
        sub_cont = [continuations[i] for i in idx.tolist()]
        sub_x = x[idx].detach()
        sub_start = x_init[idx].detach() if x_init is not None else None

        def loss_fn(z: torch.Tensor) -> torch.Tensor:
            # The attack differentiates through `nll`, so the model call must be
            # inside the graph even though the weights are frozen.
            return self.lvlm.nll(z, sub_prompts, sub_cont, reduction="none")

        attack = self._make_attack(dtype, bits, n_iter=n_iter, alpha=alpha, random_init=random_init)
        x_adv, sub_losses = attack.perturb(
            loss_fn, sub_x, x_start=sub_start, maximize=maximize
        )
        out[idx] = x_adv.to(out.dtype)
        losses[idx] = sub_losses.to(losses.dtype)
        return out, losses

    @torch.no_grad()
    def _generate(
        self, x: torch.Tensor, prompts: Sequence[str], dtype: Optional[torch.dtype] = None
    ) -> List[str]:
        if dtype is not None:
            self.lvlm.set_precision(dtype)
        return self.lvlm.generate(x, prompts, max_new_tokens=self.max_new_tokens)

    # ------------------------------------------------------------ captioning
    def attack_captioning(
        self,
        images: torch.Tensor,
        prompts: Sequence[str],
        references: Sequence[Sequence[str]],
        score_fn: Callable[[Sequence[str], Sequence[Sequence[str]]], List[float]],
        threshold: float = 10.0,
        active: Optional[torch.Tensor] = None,
    ) -> UntargetedResult:
        """See the module docstring; ``score_fn`` computes the CIDEr score."""
        n = images.shape[0]
        n_refs = max(len(r) for r in references)
        device = images.device
        active = torch.ones(n, dtype=torch.bool, device=device) if active is None else active.clone()
        worst = torch.full((n,), float("inf"), device=device)
        x_worst = images.clone()
        best_ref = torch.zeros(n, dtype=torch.long, device=device)
        x_per_ref: List[torch.Tensor] = []

        # ---- half-precision stage: one attack per ground-truth caption -------
        for j in range(n_refs):
            was_active = active.clone()
            continuations = [refs[j] if j < len(refs) else refs[0] for refs in references]
            if self.use_half_precision_stage:
                x_j, _ = self._attack_subset(
                    images,
                    active,
                    prompts,
                    continuations,
                    dtype=self.half_dtype,
                    bits=self.half_bits,
                    random_init=True,
                )
            else:
                x_j = images.clone()
            x_per_ref.append(x_j)
            # Samples that are already broken keep their current worst-case image.
            x_eval = torch.where(was_active.view(-1, 1, 1, 1), x_j, x_worst)
            outputs = self._generate(x_eval, prompts, dtype=self.half_dtype)
            scores = torch.tensor(score_fn(outputs, references), device=device, dtype=torch.float32)
            better = was_active & (scores < worst)
            worst[better] = scores[better]
            x_worst[better] = x_eval[better]
            best_ref[better] = j
            active = active & (worst >= threshold)
            LOGGER.info(
                "captioning attack (half precision, gt %d/%d): mean score %.2f, active %d/%d",
                j + 1, n_refs, float(scores.mean()), int(active.sum()), n,
            )

        # ---- single-precision stage on the worst ground truth ----------------
        if self.use_single_precision_stage:
            continuations = [
                references[i][int(best_ref[i])] if int(best_ref[i]) < len(references[i]) else references[i][0]
                for i in range(n)
            ]
            x_init = torch.stack([x_per_ref[int(best_ref[i])][i] for i in range(n)]) if x_per_ref else images
            x_final, _ = self._attack_subset(
                images,
                torch.ones(n, dtype=torch.bool, device=device),
                prompts,
                continuations,
                dtype=self.single_dtype,
                bits=self.single_bits,
                x_init=x_init,
                random_init=False,
            )
            outputs = self._generate(x_final, prompts, dtype=self.single_dtype)
            scores = torch.tensor(score_fn(outputs, references), device=device, dtype=torch.float32)
            better = scores < worst
            worst[better] = scores[better]
            x_worst[better] = x_final[better]
            LOGGER.info("captioning attack (single precision): mean score %.2f", float(scores.mean()))

        outputs = self._generate(x_worst, prompts, dtype=self.single_dtype)
        return UntargetedResult(images=x_worst, scores=worst, outputs=outputs)

    # ------------------------------------------------------------------- VQA
    def attack_vqa(
        self,
        images: torch.Tensor,
        prompts: Sequence[str],
        answers: Sequence[Sequence[str]],
        score_fn: Callable[[Sequence[str], Sequence[Sequence[str]]], List[float]],
        threshold: float = 0.0,
        max_answers: int = 5,
        targeted_strings: Sequence[str] = ("maybe",),
        active: Optional[torch.Tensor] = None,
    ) -> UntargetedResult:
        """VQA attack: untargeted (half + single precision) plus targeted strings."""
        n = images.shape[0]
        device = images.device
        active = torch.ones(n, dtype=torch.bool, device=device) if active is None else active.clone()
        worst = torch.full((n,), float("inf"), device=device)
        x_worst = images.clone()
        best_idx = torch.zeros(n, dtype=torch.long, device=device)
        x_per_ans: List[torch.Tensor] = []

        for j in range(max_answers):
            was_active = active.clone()
            continuations = [ans[j] if j < len(ans) else ans[0] for ans in answers]
            if self.use_half_precision_stage:
                x_j, _ = self._attack_subset(
                    images,
                    active,
                    prompts,
                    continuations,
                    dtype=self.half_dtype,
                    bits=self.half_bits,
                    random_init=True,
                )
            else:
                x_j = images.clone()
            x_per_ans.append(x_j)
            x_eval = torch.where(was_active.view(-1, 1, 1, 1), x_j, x_worst)
            outputs = self._generate(x_eval, prompts, dtype=self.half_dtype)
            scores = torch.tensor(score_fn(outputs, answers), device=device, dtype=torch.float32)
            better = was_active & (scores < worst)
            worst[better] = scores[better]
            x_worst[better] = x_eval[better]
            best_idx[better] = j
            active = active & (worst > threshold)
            LOGGER.info(
                "vqa attack (half precision, answer %d/%d): mean score %.3f, active %d/%d",
                j + 1, max_answers, float(scores.mean()), int(active.sum()), n,
            )

        if self.use_single_precision_stage:
            continuations = [
                answers[i][int(best_idx[i])] if int(best_idx[i]) < len(answers[i]) else answers[i][0]
                for i in range(n)
            ]
            x_init = torch.stack([x_per_ans[int(best_idx[i])][i] for i in range(n)]) if x_per_ans else images
            x_final, _ = self._attack_subset(
                images,
                torch.ones(n, dtype=torch.bool, device=device),
                prompts,
                continuations,
                dtype=self.single_dtype,
                bits=self.single_bits,
                x_init=x_init,
                random_init=False,
            )
            outputs = self._generate(x_final, prompts, dtype=self.single_dtype)
            scores = torch.tensor(score_fn(outputs, answers), device=device, dtype=torch.float32)
            better = scores < worst
            worst[better] = scores[better]
            x_worst[better] = x_final[better]
            LOGGER.info("vqa attack (single precision): mean score %.3f", float(scores.mean()))

            # ---- targeted attacks with "maybe" (and "Word") -------------------
            for target in targeted_strings:
                x_t, _ = self._attack_subset(
                    images,
                    torch.ones(n, dtype=torch.bool, device=device),
                    prompts,
                    [target] * n,
                    dtype=self.single_dtype,
                    bits=self.single_bits,
                    x_init=None,               # clean perturbation initialisation
                    maximize=False,
                    random_init=False,
                )
                outputs = self._generate(x_t, prompts, dtype=self.single_dtype)
                scores = torch.tensor(score_fn(outputs, answers), device=device, dtype=torch.float32)
                better = scores < worst
                worst[better] = scores[better]
                x_worst[better] = x_t[better]
                LOGGER.info("vqa target '%s': mean score %.3f", target, float(scores.mean()))

        outputs = self._generate(x_worst, prompts, dtype=self.single_dtype)
        return UntargetedResult(images=x_worst, scores=worst, outputs=outputs)


# --------------------------------------------------------------------------- #
#                       stealthy targeted attacks (Sec. 4.2)                   #
# --------------------------------------------------------------------------- #
class TargetedStringAttack:
    """Force an LVLM to output an exact target string (Schlarmann & Hein, 2023).

    10 000 iterations of APGD, :math:`\\ell_\\infty` radii of ``2/255`` and
    ``4/255``; an attack counts as successful if the target string appears
    verbatim in the generated output.
    """

    def __init__(
        self,
        lvlm: LVLM,
        eps: float = 4 / 255,
        n_iter: int = 10000,
        alpha: Optional[float] = None,
        momentum: float = 0.9,
        grad_normalization: str = "elementwise_sign",
        step_schedule: bool = True,
        random_init: bool = True,
        bits: int = 32,
        dtype: torch.dtype = torch.float32,
        max_new_tokens: int = 32,
    ):
        self.lvlm = lvlm
        self.eps = float(eps)
        self.n_iter = int(n_iter)
        self.alpha = float(alpha) if alpha is not None else float(eps)
        self.momentum = momentum
        self.grad_normalization = grad_normalization
        self.step_schedule = step_schedule
        self.random_init = random_init
        self.bits = bits
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens

    def attack(
        self,
        images: torch.Tensor,
        prompts: Sequence[str],
        targets: Sequence[str],
    ) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        self.lvlm.set_precision(self.dtype)

        def loss_fn(z: torch.Tensor) -> torch.Tensor:
            return self.lvlm.nll(z, list(prompts), list(targets), reduction="none")

        attack = APGDAttack(
            eps=self.eps,
            n_iter=self.n_iter,
            alpha=self.alpha,
            momentum=self.momentum,
            grad_normalization=self.grad_normalization,
            step_schedule=self.step_schedule,
            random_init=self.random_init,
            quantize_bits=self.bits,
        )
        x_adv, losses = attack.perturb(loss_fn, images, maximize=False)
        with torch.no_grad():
            outputs = self.lvlm.generate(x_adv, list(prompts), max_new_tokens=self.max_new_tokens)
        success = torch.tensor(
            [t.strip() in o for t, o in zip(targets, outputs)], dtype=torch.bool
        )
        return x_adv, outputs, success


def summarise_targeted_results(outputs: Sequence[str], targets: Sequence[str]) -> Dict[str, float]:
    """Mean success rate of a targeted attack (Table 3 reports ``k/25``)."""
    hits = sum(1 for o, t in zip(outputs, targets) if t.strip() in o)
    return {"success": hits, "total": len(outputs), "rate": hits / max(1, len(outputs))}
