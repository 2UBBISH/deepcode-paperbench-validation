"""APGD -- the AutoAttack attack used throughout the paper.

Croce & Hein (2020), *Reliable evaluation of adversarial robustness with an
ensemble of diverse parameter-free attacks*.  The paper uses

* APGD-CE and APGD-DLR with 100 iterations each for the zero-shot
  classification evaluation (Sec. 4.3).  Note that -- unlike Mao et al. (2023)
  -- the **targeted** DLR loss is used, following AutoAttack;
* APGD with 100 iterations for the LVLM evaluation (Sec. 4.1), where the
  initial step size is set to :math:`\\varepsilon` (Schlarmann & Hein, 2023);
* APGD with 10 000 iterations for the stealthy targeted attacks (Sec. 4.2);
* the same machinery with 10 iterations for the inner maximisation during
  fine-tuning (App. B.1); see :mod:`robust_clip.training.pgd` for that variant.

The implementation follows ``robust-finetuning`` (fra31) / AutoAttack: Nesterov
momentum, a step-size schedule that halves the step when the loss does not
improve and doubles it when it does, and a best-iterate tracking.  Gradient
normalisation defaults to *elementwise sign* in the :math:`\\ell_\\infty` threat
model, as specified in the addendum to the paper.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..training.pgd import normalize_gradient
from ..utils.quant import quantize_delta


LossFn = Callable[[torch.Tensor], torch.Tensor]


# --------------------------------------------------------------------------- #
#                                 loss functions                               #
# --------------------------------------------------------------------------- #
def ce_loss(logits: torch.Tensor, targets: torch.Tensor, reduction: str = "none") -> torch.Tensor:
    return F.cross_entropy(logits.float(), targets, reduction=reduction)


def dlr_loss(logits: torch.Tensor, targets: torch.Tensor, reduction: str = "none") -> torch.Tensor:
    """Difference of logits ratio (Croce & Hein, 2020), untargeted."""
    logits = logits.float()
    targets = targets.long()
    num_classes = logits.size(1)
    if num_classes < 3:
        return ce_loss(logits, targets)
    sorted_logits, _ = logits.sort(dim=1)
    idx = torch.arange(logits.size(0), device=logits.device)
    z_y = logits[idx, targets]
    # ``ind`` is 1 when the ground-truth class is the most likely one; in that
    # case the runner-up is used in the numerator (AutoAttack, eq. (7)).
    ind = (sorted_logits[:, -1] == z_y).float()
    numerator = z_y - sorted_logits[:, -2] * ind - sorted_logits[:, -1] * (1.0 - ind)
    denominator = sorted_logits[:, -1] - sorted_logits[:, -3] + 1e-12
    out = -numerator / denominator
    return out if reduction == "none" else out.mean()


def dlr_loss_targeted(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_classes: torch.Tensor,
    reduction: str = "none",
) -> torch.Tensor:
    """Targeted DLR loss (AutoAttack); ``target_classes`` is the desired class."""
    logits = logits.float()
    idx = torch.arange(logits.size(0), device=logits.device)
    z_y = logits[idx, targets.long()]
    z_t = logits[idx, target_classes.long()]
    sorted_logits, _ = logits.sort(dim=1)
    denominator = sorted_logits[:, -1] - 0.5 * (sorted_logits[:, -3] + sorted_logits[:, -4]) + 1e-12
    out = -(z_y - z_t) / denominator
    return out if reduction == "none" else out.mean()


def runner_up_classes(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Class with the highest logit among the non-ground-truth classes."""
    logits = logits.detach().float().clone()
    idx = torch.arange(logits.size(0), device=logits.device)
    logits[idx, targets.long()] = -float("inf")
    return logits.argmax(dim=1)


def make_loss(
    kind: str,
    targets: Optional[torch.Tensor] = None,
    target_classes: Optional[torch.Tensor] = None,
    logits_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    num_classes: Optional[int] = None,
) -> LossFn:
    """Build the per-sample loss minimised by :class:`APGDAttack`.

    ``kind`` is one of ``"ce"``, ``"dlr"``, ``"targeted_ce"`` and
    ``"targeted_dlr"``.  ``logits_fn`` maps a batch of pixel-space images to
    logits.
    """
    if logits_fn is None:
        raise ValueError("logits_fn is required")
    kind = kind.lower()

    def loss_fn(x: torch.Tensor) -> torch.Tensor:
        logits = logits_fn(x)
        if kind == "ce":
            return ce_loss(logits, targets)
        if kind == "dlr":
            return dlr_loss(logits, targets)
        if kind == "targeted_ce":
            return ce_loss(logits, target_classes)
        if kind == "targeted_dlr":
            return dlr_loss_targeted(logits, targets, target_classes)
        raise ValueError(f"unknown loss '{kind}'")

    return loss_fn


# --------------------------------------------------------------------------- #
#                                    APGD                                      #
# --------------------------------------------------------------------------- #
class APGDAttack:
    """APGD for the :math:`\\ell_\\infty` threat model.

    Parameters
    ----------
    eps:
        perturbation radius (in units of the ``[0, 1]`` pixel range, e.g.
        ``2 / 255``).
    n_iter:
        100 for the evaluation attacks of the paper, 10 000 for the stealthy
        targeted attacks.
    alpha:
        initial step size.  ``None`` uses ``2 * eps`` (AutoAttack); the LVLM
        pipeline of the paper instead uses ``alpha = eps``.
    momentum:
        Nesterov momentum coefficient (0.75 in AutoAttack, 0.9 in the paper's
        PGD/LVLM attacks).
    grad_normalization:
        ``"elementwise_sign"`` (paper / addendum), ``"mean_abs"`` (AutoAttack)
        or ``"none"``.
    quantize_bits:
        16 for the half-precision stage of the LVLM pipeline, 32 for the
        single-precision stage, ``None`` to keep continuous perturbations.
    step_schedule:
        whether to use the APGD step-size schedule (halve on stagnation, double
        on improvement).
    """

    def __init__(
        self,
        eps: float,
        n_iter: int = 100,
        alpha: Optional[float] = None,
        momentum: float = 0.75,
        grad_normalization: str = "elementwise_sign",
        n_restarts: int = 1,
        random_init: bool = True,
        step_schedule: bool = True,
        quantize_bits: Optional[int] = None,
        clamp_min: float = 0.0,
        clamp_max: float = 1.0,
    eot: int = 1,
    ):
        self.eps = float(eps)
        self.n_iter = int(n_iter)
        self.alpha = float(alpha) if alpha is not None else 2 * float(eps)
        self.alpha0 = self.alpha
        self.momentum = float(momentum)
        self.grad_normalization = grad_normalization
        self.n_restarts = int(n_restarts)
        self.random_init = random_init
        self.step_schedule = step_schedule
        self.quantize_bits = quantize_bits
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.eot = max(1, int(eot))

    # ------------------------------------------------------------------ utils
    def _project(self, x: torch.Tensor, x_orig: torch.Tensor) -> torch.Tensor:
        x = torch.max(torch.min(x, x_orig + self.eps), x_orig - self.eps)
        return x.clamp(min=self.clamp_min, max=self.clamp_max)

    # ----------------------------------------------------------------- attack
    def perturb(
        self,
        loss_fn: LossFn,
        x: torch.Tensor,
        x_start: Optional[torch.Tensor] = None,
        x_best: Optional[torch.Tensor] = None,
        loss_best: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        maximize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Optimise ``loss_fn`` (per-sample) inside the :math:`\\ell_\\infty` ball.

        ``x`` is the *clean* batch and defines the centre of the
        :math:`\\ell_\\infty` ball; ``x_start`` (optional) is the point the
        optimisation is initialised from -- the paper's ensemble pipeline uses
        the perturbation found by the half-precision stage there.

        ``maximize`` follows the ``AutoAttack`` convention: if true (default)
        the returned point maximises ``loss_fn`` (this is what the zero-shot
        classification attacks and the untargeted LVLM attacks do -- the latter
        maximise the negative log-likelihood of the ground-truth answer).  For
        the *targeted* LVLM attacks ``maximize=False`` is used together with the
        negative log-likelihood of the target string, so that the loss is
        minimised.

        Returns the best adversarial batch found and the corresponding
        per-sample losses.
        """
        direction = 1.0 if maximize else -1.0
        x_orig = x.detach()
        best = x_best.detach().clone() if x_best is not None else x_orig.clone()
        if loss_best is not None:
            best_loss = loss_best.detach().clone().float()
        else:
            fill = -float("inf") if maximize else float("inf")
            best_loss = torch.full((x.shape[0],), fill, device=x.device, dtype=torch.float32)

        checkpoints = {int(self.n_iter * (i + 1) / (self.n_restarts + 1)) for i in range(self.n_restarts)}
        if self.step_schedule:
            checkpoints = {self.n_iter // 10, self.n_iter // 10 * 3, self.n_iter // 10 * 7} | checkpoints
        checkpoints = {c for c in checkpoints if 0 < c < self.n_iter}

        n_runs = self.n_restarts + 1
        for run in range(n_runs):
            alpha = torch.full((x.shape[0],), self.alpha0, device=x.device, dtype=torch.float32)
            if x_start is not None and run == 0:
                x_adv = self._project(x_start.detach().clone(), x_orig)
            else:
                x_adv = x_orig.clone()
            if self.random_init and run > 0:
                delta = torch.empty_like(x_adv).uniform_(-self.eps, self.eps, generator=generator)
                if self.quantize_bits is not None:
                    delta = quantize_delta(delta, self.quantize_bits)
                x_adv = self._project(x_adv + delta, x_orig)

            grad_prev = torch.zeros_like(x_adv)
            x_prev = x_adv.clone()

            for i in range(self.n_iter):
                x_adv = x_adv.detach().requires_grad_(True)
                loss = torch.zeros(x.shape[0], device=x.device)
                grad = torch.zeros_like(x_adv)
                for _ in range(self.eot):
                    step_loss = loss_fn(x_adv)
                    step_grad, = torch.autograd.grad(step_loss.sum(), x_adv, only_inputs=True)
                    loss = loss + step_loss.detach()
                    grad = grad + step_grad.detach()
                loss = loss / self.eot
                grad = grad / self.eot

                grad = normalize_gradient(grad, self.grad_normalization)
                grad = grad + self.momentum * grad_prev
                x_adv = x_adv.detach() + direction * alpha.view(-1, 1, 1, 1) * grad.sign()
                if self.quantize_bits is not None:
                    x_adv = x_orig + quantize_delta(x_adv - x_orig, self.quantize_bits)
                x_adv = self._project(x_adv, x_orig)
                grad_prev = grad

                with torch.no_grad():
                    better = (direction * loss) > (direction * best_loss)
                    if better.any():
                        best[better] = x_adv[better].clone()
                        best_loss[better] = loss[better]

                # APGD step-size schedule: double the step size if the best
                # point has improved since the last checkpoint, otherwise halve
                # it and restart the momentum.
                if self.step_schedule and i in checkpoints:
                    changed = (best != x_prev).flatten(1).any(dim=1)
                    alpha = torch.where(
                        changed,
                        torch.clamp(alpha * 2.0, max=self.eps * 4),
                        torch.clamp(alpha / 2.0, min=1e-10),
                    )
                    grad_prev = torch.zeros_like(grad_prev)
                    x_prev = best.clone()

        return best.detach(), best_loss.detach()


def apgd_linf(
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    eps: float = 2 / 255,
    n_iter: int = 100,
    loss: str = "ce",
    target_classes: Optional[torch.Tensor] = None,
    alpha: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convenience wrapper: build the loss and run :class:`APGDAttack`."""
    loss_fn = make_loss(
        loss,
        targets=y,
        target_classes=target_classes,
        logits_fn=logits_fn,
    )
    attack = APGDAttack(eps=eps, n_iter=n_iter, alpha=alpha, **kwargs)
    return attack.perturb(loss_fn, x)
