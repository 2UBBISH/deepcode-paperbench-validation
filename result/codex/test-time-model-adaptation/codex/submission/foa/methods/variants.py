"""Design-choice variants of Table 9 (learnable parameters x optimiser x loss).

The paper compares

* learnable parameters: the affine parameters of the normalisation layers (TENT-style)
  versus the newly inserted prompts,
* optimiser: SGD versus CMA-ES,
* loss: prediction entropy versus Eqn. (5).

``exp1``-``exp6`` of Table 9 are exactly the six combinations, and ``Ours`` is
"prompts + CMA + Eqn. (5)" - the FOA method itself.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from ..core.cma import make_cma
from ..core.fitness import activation_discrepancy, foa_fitness, prediction_entropy
from ..core.statistics import FeatureStatistics
from ..models.base import AdaptableModel
from .base import TTAMethod, get_backbone
from .tent import collect_norm_parameters


class SGDAdapt(TTAMethod):
    """Adapt either *prompts* or *norm-layer affine parameters* with SGD.

    Args:
        params: ``"prompts"`` or ``"norm"``.
        loss: ``"entropy"`` or ``"eqn5"`` (the fitness function of the paper).
    """

    def __init__(
        self,
        model: AdaptableModel,
        source_stats: Optional[FeatureStatistics] = None,
        params: str = "prompts",
        loss: str = "entropy",
        lr: float = 1e-3,
        momentum: float = 0.9,
        lam: float = 30.0,
        num_prompts: int = 3,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(model, device=device)
        self.last_extra = {}
        self.source_stats = source_stats
        self.params_kind = params
        self.loss_kind = loss
        self.lam = lam
        self.embed_dim = self.model.embed_dim
        self.num_prompts = num_prompts
        trainable = []
        if params == "prompts":
            assert isinstance(self.model, AdaptableModel)
            init = self.model.initial_prompt_vector(device=self.device, dtype=torch.float32)
            self.prompt = nn.Parameter(init.reshape(num_prompts, self.embed_dim).clone())
            trainable = [self.prompt]
        elif params == "norm":
            trainable = collect_norm_parameters(self.model)
        else:
            raise ValueError(params)
        assert trainable, f"SGDAdapt({params}) found no trainable parameters"
        self.name = f"SGD/{params}/{loss}"
        self.optimizer = torch.optim.SGD(trainable, lr=lr, momentum=momentum)

    def _loss(self, images: torch.Tensor) -> torch.Tensor:
        prompt = getattr(self, "prompt", None)
        logits, _, feats = self.model.forward_with_prompt(
            images, prompt=prompt, return_layers=True
        )
        if self.loss_kind == "entropy":
            return prediction_entropy(logits, reduction="mean")
        assert self.source_stats is not None and feats is not None
        disc = activation_discrepancy(feats, self.source_stats, layers=range(1, len(feats)))
        return prediction_entropy(logits, reduction="mean") + self.lam * disc

    def step(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        self.optimizer.zero_grad()
        loss = self._loss(images)
        loss.backward()
        self.optimizer.step()
        self.last_extra = {"loss": float(loss.item())}
        with torch.no_grad():
            prompt = getattr(self, "prompt", None)
            logits, _, _ = self.model.forward_with_prompt(images, prompt=prompt)
        return logits


class CMANormAdapt(TTAMethod):
    """``exp4``/``exp5``: CMA-ES over the affine parameters of the normalisation layers.

    Optimising ~38k parameters with a full covariance matrix is intractable, which is
    exactly the point the paper makes ("CMA fails to handle ultra-high-dimensional
    optimization"): the search collapses and the accuracy drops to chance level.  We use
    the separable variant of CMA-ES so that the experiment is actually runnable, and the
    same degeneracy is observed.
    """

    def __init__(
        self,
        model: AdaptableModel,
        source_stats: Optional[FeatureStatistics] = None,
        loss: str = "eqn5",
        lam: float = 30.0,
        sigma0: float = 0.1,
        popsize: int = 28,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(model, device=device)
        self.last_extra = {}
        self.source_stats = source_stats
        self.loss_kind = loss
        self.lam = lam
        self.sigma0 = sigma0
        self.popsize = popsize
        self.seed = seed
        self.name = f"CMA/norm/{loss}"
        self.reset()

    # ----------------------------------------------------------------------------------
    def _norm_params(self) -> Sequence[nn.Parameter]:
        backbone = get_backbone(self.model)
        params = []
        for m in backbone.modules():
            if isinstance(m, nn.LayerNorm):
                params.extend(list(m.parameters()))
        return params

    def reset(self) -> None:
        self.params = self._norm_params()
        with torch.no_grad():
            self.origin = torch.cat([p.detach().reshape(-1).clone() for p in self.params])
        try:  # separable CMA keeps the memory footprint tractable (see the class docstring)
            import cmaes

            self.es = cmaes.SepCMA(
                mean=self.origin.cpu().numpy().astype(np.float64),
                sigma=self.sigma0,
                population_size=self.popsize,
                seed=self.seed,
            )
            self._backend = "cmaes"
        except ImportError:  # pragma: no cover
            self.es = None
            self._backend = "none"

    def _write_params(self, flat: np.ndarray) -> None:
        t = torch.as_tensor(flat, dtype=torch.float32, device=self.device)
        offset = 0
        with torch.no_grad():
            for p in self.params:
                n = p.numel()
                p.copy_(t[offset : offset + n].reshape(p.shape))
                offset += n

    def _fitness(self, images: torch.Tensor) -> float:
        logits, _, feats = self.model.forward_with_prompt(images, return_layers=True)
        value = foa_fitness(
            logits,
            feats,
            self.source_stats,
            lam=self.lam,
            layers=range(1, len(feats)),
            use_entropy=self.loss_kind in ("entropy", "eqn5"),
            use_activation_discrepancy=self.loss_kind == "eqn5",
            normalize_entropy=True,
        )
        return float(value.item())

    @torch.no_grad()
    def step(self, images: torch.Tensor) -> torch.Tensor:
        if self.es is None:  # pragma: no cover
            logits, _, _ = self.model.forward_with_prompt(images.to(self.device))
            return logits
        images = images.to(self.device)
        solutions, values = [], []
        best_logits = None
        best = float("inf")
        for _ in range(self.popsize):
            sol = self.es.ask()
            self._write_params(sol)
            value = self._fitness(images)
            solutions.append(sol)
            values.append(value)
            if value < best:
                best = value
                best_logits = self.model.forward_with_prompt(images)[0]
        # ``cmaes`` uses an ask-and-tell interface with one solution per call
        self.es.tell(list(zip(solutions, values)))
        self.last_extra = {"fitness_min": best}
        if best_logits is None:  # pragma: no cover
            best_logits = self.model.forward_with_prompt(images)[0]
        return best_logits
