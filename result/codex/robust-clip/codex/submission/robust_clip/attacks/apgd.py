"""APGD -- the parameter-free adversarial attack of Croce & Hein (2020).

The paper uses APGD for

* the zero-shot classification evaluation (Sec. 4.3): *"We employ the first two
  attacks of AutoAttack, namely APGD with cross-entropy loss and APGD with
  targeted DLR loss (100 iterations each)"* (the standard AutoAttack uses 9
  candidate target classes for the targeted DLR attack),
* the LVLM evaluation pipeline (Sec. 4.1 / App. B.6): APGD with 100 iterations,
  run first at half precision and then at single precision on the samples that
  survived the first stage,
* the embedding-loss analysis of App. C.4 (100-step APGD with ``eps = 4/255``).

The implementation mirrors AutoAttack:

* normalized gradient steps with the momentum term
  ``x_adv + a * step + (1 - a) * (x_adv - x_adv_old)``, ``a = 0.75``,
* the adaptive, per-sample step size of the "non-targeted" rule (reduction by a
  factor 2 when the loss oscillates or stops improving),
* the DLR loss for the (targeted) attacks:
  ``-(x_y - max_{j != y} x_j) / (x_max_1 - x_third_max)``,
* restarts, keeping per-sample the iterate with the largest loss (equivalent to
  AutoAttack's ``best_loss=True`` bookkeeping).

Differences with respect to the default AutoAttack configuration are the ones
used in the paper: the initial step size is ``eps`` (*"Following Schlarmann &
Hein (2023), we set the initial step-size of APGD to eps"*, App. B.6), the
number of iterations is 100, and the perturbation is snapped to the grid of the
precision the attack runs in (16-bit ints for the half-precision stage, 32-bit
ints for the single-precision stage).

The loss convention is "the attack always *maximizes* the returned per-sample
loss", which covers the untargeted losses (`ce`, `dlr`) as well as the targeted
ones (the DLR-targeted loss and the token-level losses used for the targeted
LVLM attacks are written with a minus sign).
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F

from ..utils.precision import PerturbationGrid, eps_to_int
from ..utils.misc import parse_eps

__all__ = [
    "APGDAttack",
    "apgd_attack",
    "ce_loss",
    "dlr_loss",
    "dlr_loss_targeted",
    "normalize_linf",
]


def normalize_linf(t: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Scale the perturbation ``t`` so that its largest absolute value is 1."""
    scale = t.abs().flatten(1).max(dim=1)[0].clamp_min(eps)
    return t / scale.view(-1, *([1] * (t.dim() - 1)))


def normalize_l2(t: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    norm = t.flatten(1).norm(p=2, dim=1).clamp_min(eps)
    return t / norm.view(-1, *([1] * (t.dim() - 1)))


def ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Untargeted cross-entropy loss (APGD-CE)."""
    return F.cross_entropy(logits, labels, reduction="none")


def dlr_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Difference of Logits Ratio loss (Croce & Hein, 2020)."""
    sorted_logits, _ = logits.sort(dim=1)
    batch = torch.arange(logits.shape[0], device=logits.device)
    is_correct = (sorted_logits[:, -1] == logits[batch, labels]).float()
    return -(
        logits[batch, labels]
        - sorted_logits[:, -2] * is_correct
        - sorted_logits[:, -1] * (1.0 - is_correct)
    ) / (sorted_logits[:, -1] - sorted_logits[:, -3] + 1e-12)


def dlr_loss_targeted(logits: torch.Tensor, labels: torch.Tensor, target_labels: torch.Tensor) -> torch.Tensor:
    """Targeted DLR loss: maximizing it pushes ``target_labels`` above the true class."""
    sorted_logits, _ = logits.sort(dim=1)
    batch = torch.arange(logits.shape[0], device=logits.device)
    return -(logits[batch, labels] - logits[batch, target_labels]) / (
        sorted_logits[:, -1] - 0.5 * (sorted_logits[:, -3] + sorted_logits[:, -4]) + 1e-12
    )


class APGDAttack:
    """AutoAttack-style APGD for the losses `ce`, `dlr`, `dlr_targeted` or a custom loss.

    Parameters
    ----------
    eps:
        l_inf radius, as float or as ``'k/255'`` string.
    n_iter:
        100 in all experiments of the paper.
    n_restarts:
        1 restart for AutoAttack's standard version (Sec. 4.3) and for the LVLM
        pipeline.
    dtype:
        ``torch.float16`` (half precision stage) or ``torch.float32``.
    alpha_init:
        initial step size; defaults to ``eps`` (App. B.6).
    rho:
        threshold of the oscillation check (0.75, as in AutoAttack).
    eot_iter:
        expectation over transformations iterations (1 everywhere in the paper).
    """

    def __init__(
        self,
        eps: Union[str, float] = "2/255",
        n_iter: int = 100,
        n_restarts: int = 1,
        loss: str = "ce",
        dtype: torch.dtype = torch.float32,
        alpha_init: Optional[Union[str, float]] = None,
        rho: float = 0.75,
        eot_iter: int = 1,
        seed: int = 0,
        track_best_every: int = 1,
    ):
        self.eps = eps
        self.eps_int = eps_to_int(eps)
        self.n_iter = int(n_iter)
        self.n_restarts = int(n_restarts)
        self.loss = loss
        self.dtype = dtype
        self.rho = rho
        self.eot_iter = eot_iter
        self.seed = seed
        self.track_best_every = max(1, int(track_best_every))
        # the paper initializes the APGD step size with eps (App. B.6)
        self.alpha_init = parse_eps(eps) if alpha_init is None else parse_eps(alpha_init)

        # checkpoints of the step size controller (AutoAttack defaults)
        self.n_iter_2 = max(int(0.22 * self.n_iter), 1)
        self.n_iter_min = max(int(0.06 * self.n_iter), 1)
        self.size_decr = max(int(0.03 * self.n_iter), 1)

    # ------------------------------------------------------------------
    # loss helpers
    # ------------------------------------------------------------------
    def _make_loss(self, labels, target_labels, custom_loss, forward_fn):
        if custom_loss is not None:
            return custom_loss
        if self.loss == "ce":
            return lambda outputs, _: ce_loss(outputs, labels)
        if self.loss == "dlr":
            return lambda outputs, _: dlr_loss(outputs, labels)
        if self.loss == "dlr_targeted":
            if target_labels is None:
                raise ValueError("dlr_targeted requires target_labels")
            return lambda outputs, _: dlr_loss_targeted(outputs, labels, target_labels)
        raise ValueError(f"unknown loss {self.loss!r}")

    def _loss_and_grad(self, x_adv, forward_fn, loss_fn):
        x_adv = x_adv.detach().requires_grad_(True)
        grad = torch.zeros_like(x_adv)
        loss_value = None
        for _ in range(self.eot_iter):
            with torch.enable_grad():
                outputs = forward_fn(x_adv)
                loss_value = loss_fn(outputs, x_adv)
                if loss_value.dim() == 0:
                    loss_value = loss_value.expand(x_adv.shape[0])
                grad = grad + torch.autograd.grad(loss_value.sum(), x_adv, retain_graph=False)[0]
        return loss_value.detach(), grad.detach() / float(self.eot_iter)

    # ------------------------------------------------------------------
    # the attack
    # ------------------------------------------------------------------
    def _single_run(
        self,
        x: torch.Tensor,
        forward_fn: Callable,
        loss_fn: Callable,
        grid: PerturbationGrid,
        x_init: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        eps = grid.eps
        batch_size = x.shape[0]
        dims = [1] * (x.dim() - 1)

        if x_init is not None:
            x_adv = grid.project(x_init.detach().to(torch.float32), x)
        else:
            # AutoAttack's initialization: random direction, scaled to the eps-ball
            t = 2 * torch.rand(x.shape, device=x.device, dtype=torch.float32, generator=generator) - 1
            x_adv = grid.project(x + eps * normalize_linf(t), x)

        loss_best, grad = self._loss_and_grad(x_adv, forward_fn, loss_fn)
        x_best = x_adv.detach().clone()
        grad_best = grad.clone()
        loss_best = loss_best.clone()

        step_size = self.alpha_init * torch.ones([batch_size, *dims], device=x.device, dtype=torch.float32)
        x_adv_old = x_adv.detach().clone()

        losses = torch.zeros([self.n_iter, batch_size], device=x.device)
        loss_best_last_check = loss_best.clone()
        reduced_last_check = torch.ones_like(loss_best)
        counter3 = 0
        k = self.n_iter_2

        for i in range(self.n_iter):
            # ---- gradient step (with momentum) ------------------------------
            with torch.no_grad():
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.detach().clone()
                a = 0.75 if i > 0 else 1.0

                x_adv_1 = x_adv + step_size * grad.sign()
                x_adv_1 = grid.project(x_adv_1, x)
                x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                x_adv = grid.project(x_adv_1, x)

            # ---- new gradient ----------------------------------------------
            loss_indiv, grad = self._loss_and_grad(x_adv, forward_fn, loss_fn)

            # ---- bookkeeping of the best points -----------------------------
            with torch.no_grad():
                losses[i] = loss_indiv
                improved = loss_indiv > loss_best
                x_best[improved] = x_adv.detach()[improved]
                grad_best[improved] = grad[improved]
                loss_best[improved] = loss_indiv[improved]

                # ---- adaptive step size ------------------------------------
                counter3 += 1
                if counter3 == k:
                    if i + 1 >= k:
                        recent = losses[i - k + 1 : i + 1]
                        increasing = (recent[1:] > recent[:-1]).float().sum(dim=0)
                        oscillation = (increasing <= k * self.rho * torch.ones_like(increasing)).float()
                        no_improvement = (1.0 - reduced_last_check) * (
                            loss_best_last_check >= loss_best
                        ).float()
                        oscillation = torch.max(oscillation, no_improvement)
                        reduced_last_check = oscillation.clone()
                        loss_best_last_check = loss_best.clone()

                        if oscillation.sum() > 0:
                            ind = (oscillation > 0).nonzero().squeeze(1)
                            step_size[ind] = step_size[ind] / 2.0
                            x_adv[ind] = x_best[ind].clone()
                            grad[ind] = grad_best[ind].clone()
                        k = max(k - self.size_decr, self.n_iter_min)
                    counter3 = 0

        return x_best

    @torch.no_grad()
    def _evaluate(self, x_adv, forward_fn, loss_fn) -> torch.Tensor:
        return loss_fn(forward_fn(x_adv), x_adv).detach()

    def attack(
        self,
        images: torch.Tensor,
        forward_fn: Callable[[torch.Tensor], object],
        labels: Optional[torch.Tensor] = None,
        custom_loss: Optional[Callable[[object, torch.Tensor], torch.Tensor]] = None,
        x_init: Optional[torch.Tensor] = None,
        x_min: float = 0.0,
        x_max: float = 1.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Run APGD (with ``n_restarts`` restarts) and return the adversarial images.

        ``forward_fn(x_adv)`` returns whatever ``loss_fn`` consumes; ``labels`` is
        only required for the ``ce`` / ``dlr`` / ``dlr_targeted`` losses.  For the
        LVLM attacks a ``custom_loss`` returning the (negated) token-level cross
        entropy of the ground-truth / target string is used instead.
        """
        x = images.detach().to(torch.float32).clamp(x_min, x_max)
        grid = PerturbationGrid(self.eps_int, dtype=self.dtype, alpha_int=self.eps_int, x_min=x_min, x_max=x_max)
        loss_fn = self._make_loss(labels, None, custom_loss, forward_fn)

        best_x = x.clone()
        best_loss = torch.full((x.shape[0],), -float("inf"), device=x.device)
        for restart in range(self.n_restarts):
            candidate = self._single_run(
                x,
                forward_fn,
                loss_fn,
                grid,
                x_init=x_init if restart == 0 else None,
                generator=generator,
            )
            candidate_loss = self._evaluate(candidate, forward_fn, loss_fn)
            better = candidate_loss > best_loss
            best_loss = torch.where(better, candidate_loss, best_loss)
            best_x[better] = candidate[better]
        return best_x.detach()

    def attack_targeted(
        self,
        images: torch.Tensor,
        forward_fn: Callable[[torch.Tensor], torch.Tensor],
        labels: torch.Tensor,
        n_target_classes: int = 9,
        x_init: Optional[torch.Tensor] = None,
        x_min: float = 0.0,
        x_max: float = 1.0,
    ) -> torch.Tensor:
        """APGD with the targeted DLR loss over the ``n_target_classes`` closest classes.

        This is AutoAttack's "APGD-T", the second attack of the standard version
        used for the zero-shot evaluation of Sec. 4.3 (100 iterations).
        """
        x = images.detach().to(torch.float32).clamp(x_min, x_max)
        grid = PerturbationGrid(self.eps_int, dtype=self.dtype, alpha_int=self.eps_int, x_min=x_min, x_max=x_max)

        with torch.no_grad():
            logits = forward_fn(x)
        order = logits.sort(dim=1)[1]

        best_x = x.clone()
        best_loss = torch.full((x.shape[0],), -float("inf"), device=x.device)
        for target_class in range(2, n_target_classes + 2):
            target_labels = order[:, -target_class]
            loss_fn = lambda outputs, _: dlr_loss_targeted(outputs, labels, target_labels)
            candidate = self._single_run(x, forward_fn, loss_fn, grid, x_init=x_init)
            candidate_loss = self._evaluate(candidate, forward_fn, loss_fn)
            better = candidate_loss > best_loss
            best_loss = torch.where(better, candidate_loss, best_loss)
            best_x[better] = candidate[better]
        return best_x.detach()

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"APGDAttack(loss={self.loss}, eps={self.eps_int}/255, n_iter={self.n_iter}, "
            f"n_restarts={self.n_restarts}, dtype={self.dtype})"
        )


def apgd_attack(
    images: torch.Tensor,
    forward_fn: Callable,
    labels: Optional[torch.Tensor] = None,
    eps: Union[str, float] = "2/255",
    n_iter: int = 100,
    loss: str = "ce",
    dtype: torch.dtype = torch.float32,
    n_restarts: int = 1,
    custom_loss: Optional[Callable] = None,
    x_init: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """Functional wrapper around :class:`APGDAttack`."""
    attacker = APGDAttack(eps=eps, n_iter=n_iter, n_restarts=n_restarts, loss=loss, dtype=dtype, **kwargs)
    return attacker.attack(images, forward_fn, labels=labels, custom_loss=custom_loss, x_init=x_init)
