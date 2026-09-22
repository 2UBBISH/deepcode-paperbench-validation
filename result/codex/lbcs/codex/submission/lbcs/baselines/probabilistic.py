"""Probabilistic bilevel coreset selection (Zhou et al., ICML 2022).

This is the method this paper builds upon; it is used as (a) the "trivial
solution" baseline in Section 2.1 / Figure 1 and (b) one of the compared
baselines of Section 5.2.

Method (Appendix C.1).  Each mask entry is reparameterised as a Bernoulli
random variable, ``m_i ~ Bern(s_i)``, and the coreset size is controlled by
``E ||m||_0 = sum_i s_i = 1^T s``.  Without the size objective (equation (3)):

    min_s  E_{p(m|s)} f_1(m)  s.t. theta(m) in arg min_theta L(m, theta)

and with the trade-off (equation (4)):

    min_s  (1 - lambda) E_{p(m|s)} f_1(m) + lambda E_{p(m|s)} f_2(m).

The outer loop is solved with the unbiased policy-gradient estimator derived
in Appendix C.2 (equation (29)):

    grad_s [ E f_1 + E f_2 ] = E_{p(m|s)}[ f_1(m) (m - s) / (s(1 - s)) ] + 1

in which the first term is estimated with samples of ``m`` and the second term
is available in closed form.  ``C`` samples of the policy gradient estimator
are used per outer iteration, which is the ``C`` of the time-complexity
comparison in Section 6.

For Figure 1 the settings of Appendix C.3 are used: a subset of MNIST, the
ConvNet of Zhou et al. (2022), an inner loop of 100 epochs of SGD with
learning rate 0.1 and momentum 0.9, and an outer loop of Adam with learning
rate 2.5 and a cosine scheduler.  The addendum additionally states that
``lambda = 0.5`` is used for equation (4) and that ``T = 1000`` outer
iterations are run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

from ..data import DatasetBundle
from ..inner_loop import InnerLoopConfig, InnerLoopTrainer
from ..objectives import BilevelObjective, ObjectiveConfig
from ..utils import discretize, set_seed


@dataclass
class ProbabilisticConfig:
    k: int = 200
    T: int = 1000
    C: int = 1                     # sampling times of the policy gradient
    lam: Optional[float] = None    # None -> equation (3); 0.5 -> equation (4)
    s_lr: float = 2.5              # Adam learning rate on the probabilities
    s_scheduler: Optional[str] = "cosine"
    clamp_eps: float = 1e-4
    seed: int = 0
    inner: InnerLoopConfig = field(default_factory=InnerLoopConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    init_mode: str = "uniform"     # "uniform" | "random"
    verbose: bool = False


class ProbabilisticCoreset:
    """Probabilistic (continuous-relaxation) bilevel coreset selection."""

    def __init__(self, bundle: DatasetBundle,
                 model_factory: Callable[[], nn.Module],
                 device: torch.device,
                 config: Optional[ProbabilisticConfig] = None):
        self.bundle = bundle
        self.model_factory = model_factory
        self.device = device
        self.cfg = config or ProbabilisticConfig()
        self.trainer = InnerLoopTrainer(self.model_factory, self.cfg.inner,
                                        self.device)
        self.objective = BilevelObjective(self.bundle, self.trainer,
                                          self.device, self.cfg.objective)

    # ------------------------------------------------------------------
    def _init_probabilities(self, generator: torch.Generator) -> torch.Tensor:
        n = self.bundle.n
        if self.cfg.init_mode == "random":
            s = torch.rand(n, generator=generator)
            s = s * (2.0 * self.cfg.k / n)
            return s.clamp(self.cfg.clamp_eps, 1 - self.cfg.clamp_eps)
        return torch.full((n,), float(self.cfg.k) / n)

    def _sample_mask(self, s: torch.Tensor,
                     generator: torch.Generator) -> torch.Tensor:
        u = torch.rand(s.numel(), generator=generator)
        return (u < s).to(torch.float32)

    # ------------------------------------------------------------------
    def run(self) -> Dict[str, object]:
        cfg = self.cfg
        set_seed(cfg.seed)
        gen = torch.Generator().manual_seed(cfg.seed)

        n = self.bundle.n
        s = self._init_probabilities(gen).to(self.device)

        # logits are kept only for reporting / checkpointing
        optimizer = torch.optim.Adam([s], lr=cfg.s_lr)
        scheduler = None
        if cfg.s_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(cfg.T, 1))

        trace: List[Dict[str, float]] = []
        lam = 0.0 if cfg.lam is None else float(cfg.lam)
        for t in range(cfg.T):
            grad = torch.zeros_like(s)
            f1_accum = 0.0
            size_accum = 0.0
            for _ in range(cfg.C):
                m = self._sample_mask(s, gen).to(self.device)
                f = self.objective.evaluate(2.0 * m - 1.0)
                f1 = float(f[0])
                # equation (29): f_1(m) * (m - s) / (s (1 - s))  (+ lambda * 1)
                score = (m - s) / (s * (1.0 - s))
                grad += (1.0 - lam) * f1 * score
                grad += lam * torch.ones_like(s)
                f1_accum += f1
                size_accum += float(m.sum().item())
            grad /= cfg.C
            optimizer.zero_grad(set_to_none=True)
            s.grad = grad
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            with torch.no_grad():
                s.clamp_(cfg.clamp_eps, 1 - cfg.clamp_eps)

            trace.append({
                "iter": t,
                "f1": f1_accum / cfg.C,
                "f2": size_accum / cfg.C,
                "f2_expected": float(s.sum().item()),
            })
            if cfg.verbose and (t + 1) % max(cfg.T // 20, 1) == 0:
                print(f"[Probabilistic] t={t + 1} f1={trace[-1]['f1']:.4f} "
                      f"f2={trace[-1]['f2']:.1f} "
                      f"E|m|={trace[-1]['f2_expected']:.1f}")

        final_mask = discretize(2.0 * (s >= 0.5).float() - 1.0)
        return {
            "s": s.detach().cpu(),
            "mask": final_mask.cpu(),
            "coreset_size": int(final_mask.sum().item()),
            "trace": trace,
            "lambda": cfg.lam,
            "k": cfg.k,
        }


def probabilistic_select(bundle: DatasetBundle, k: int,
                         model_factory: Callable[[], nn.Module],
                         device: torch.device,
                         cfg: Optional[ProbabilisticConfig] = None,
                         **_) -> torch.Tensor:
    """Baseline entry point: return the selected indices."""
    if cfg is None:
        cfg = ProbabilisticConfig()
    cfg.k = k
    result = ProbabilisticCoreset(bundle, model_factory, device, cfg).run()
    return torch.nonzero(result["mask"] > 0.5, as_tuple=False).flatten()
