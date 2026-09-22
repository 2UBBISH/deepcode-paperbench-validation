"""FOA: test-time **F**orward-**O**ptimization **A**daptation (Section 3, Algorithm 1).

For every incoming batch of test samples ``X_t`` the algorithm

1. runs one forward pass with the current CMA mean prompt to obtain the *unshifted* test
   statistics ``mu_N(X_t)`` and updates the exponential moving average of Eqn. (9),
2. samples ``K`` candidate prompts from the CMA distribution (Eqn. (6)),
3. for every candidate: forwards ``[CLS, p_k, patches]``, applies the back-to-source
   activation shift of Eqn. (7) to ``e_N^0``, predicts, and evaluates the unsupervised
   fitness of Eqn. (5),
4. updates the CMA distribution from the ranking of the fitness values and returns the
   prediction of the best candidate.

No gradient is ever computed and no model weight is ever modified.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..config import FOAConfig, default_popsize
from ..models.base import AdaptableModel
from .activation_shift import BackToSourceShifting
from .cma import CMAOptimizer, make_cma
from .fitness import activation_discrepancy, foa_fitness, prediction_entropy
from .statistics import FeatureStatistics


@dataclass
class FOAStepOutput:
    logits: torch.Tensor                 # [B, C] final prediction of the batch
    prompt: torch.Tensor                 # [N_p, d] best prompt of this iteration
    fitness: np.ndarray = field(default_factory=lambda: np.zeros(0))  # [K]
    shift: Optional[torch.Tensor] = None  # gamma * d_t
    extra: Dict[str, float] = field(default_factory=dict)


class FOA:
    """Online forward-only adaptation for a single model."""

    def __init__(
        self,
        model: AdaptableModel,
        source_stats: FeatureStatistics,
        cfg: Optional[FOAConfig] = None,
        device: Optional[torch.device] = None,
        criterion_layers: Optional[Sequence[int]] = None,
    ) -> None:
        self.cfg = cfg or FOAConfig()
        self.device = torch.device(device or self.cfg.device)
        self.model = model.to(self.device).eval()
        self.source_stats = source_stats.to(self.device)
        self.num_prompts = self.cfg.num_prompts
        # the prompt of a ConvNet backbone (Table 10) is a small convolution kernel rather
        # than a set of token embeddings, hence the model defines its own prompt dimension
        self.prompt_dim = int(
            getattr(self.model, "prompt_dim", self.num_prompts * self.model.embed_dim)
        )
        self.criterion_layers = (
            list(criterion_layers) if criterion_layers is not None
            else list(range(1, self.source_stats.num_layers))
        )
        self.reset()

    # ----------------------------------------------------------------------------------
    def reset(self) -> None:
        """Re-initialise the CMA distribution (``m=0, Sigma=I, tau=1``)."""
        if self.cfg.cma_mean_from_prompt_init and hasattr(self.model, "initial_prompt_vector"):
            mean0 = self.model.initial_prompt_vector().detach().cpu().numpy().astype(np.float64)
        else:
            mean0 = np.zeros(self.prompt_dim, dtype=np.float64)
        self.es: CMAOptimizer = make_cma(
            mean0,
            sigma0=self.cfg.cma_sigma0,
            popsize=self.cfg.resolved_popsize(self.prompt_dim),
            seed=self.cfg.cma_seed,
        )
        self.shifting = BackToSourceShifting(
            self.source_stats.means[-1].detach().clone(),
            alpha=self.cfg.shift_ema_alpha,
            gamma=self.cfg.shift_gamma,
            enabled=self.cfg.use_activation_shifting,
        )
        self.iteration = 0

    # ----------------------------------------------------------------------------------
    def _prompt_tensor(self, flat: np.ndarray) -> torch.Tensor:
        return (
            torch.as_tensor(flat, dtype=torch.float32, device=self.device)
            .reshape(-1)
        )

    @property
    def mean_prompt(self) -> torch.Tensor:
        return self._prompt_tensor(self.es.mean)

    @torch.no_grad()
    def _batch_statistics(
        self, batch: torch.Tensor, prompt: Optional[torch.Tensor], is_tokens: bool = False
    ):
        means, stds = self.model.cls_statistics(
            batch if not is_tokens else None,  # type: ignore[arg-type]
            prompt=prompt,
            precomputed_tokens=batch if is_tokens else None,
        )
        return means, stds

    def _forward(
        self,
        batch: torch.Tensor,
        prompt: Optional[torch.Tensor],
        shift: Optional[torch.Tensor],
        return_layers: bool = False,
        is_tokens: bool = False,
    ):
        if is_tokens:
            return self.model.forward_tokens(
                batch, prompt=prompt, shift=shift, return_layers=return_layers
            )
        return self.model.forward_with_prompt(
            batch, prompt=prompt, shift=shift, return_layers=return_layers
        )

    # ----------------------------------------------------------------------------------
    @torch.no_grad()
    def step(self, images: torch.Tensor, is_tokens: bool = False) -> FOAStepOutput:
        """Process one test batch and return its predictions."""
        cfg = self.cfg
        lam = self._lambda_for(images.shape[0])
        batch = images.to(self.device, non_blocking=True)

        # (1) update the shifting direction from the *unshifted* features of the current
        #     state of the model (Eqn. (8)-(9)).
        shift = None
        if cfg.use_activation_shifting:
            means, _ = self._batch_statistics(batch, self.mean_prompt, is_tokens=is_tokens)
            shift = self.shifting.update(means[-1])

        # (2) sample K candidate prompts (Eqn. (6)).
        solutions = self.es.ask()
        values = np.empty(len(solutions), dtype=np.float64)
        best_logits: Optional[torch.Tensor] = None
        best_fitness = float("inf")
        entropy_all = np.empty(len(solutions), dtype=np.float64)
        disc_all = np.empty(len(solutions), dtype=np.float64)

        # (3) evaluate every candidate with forward passes only.
        for k, sol in enumerate(solutions):
            prompt = self._prompt_tensor(sol)
            logits, e_n, feats = self._forward(
                batch, prompt=prompt, shift=shift, return_layers=True, is_tokens=is_tokens
            )
            assert feats is not None
            value = foa_fitness(
                logits,
                feats,
                self.source_stats,
                lam,
                layers=self.criterion_layers,
                use_entropy=cfg.use_entropy,
                use_activation_discrepancy=cfg.use_activation_discrepancy,
            )
            values[k] = float(value.item())
            if cfg.use_entropy:
                entropy_all[k] = float(prediction_entropy(logits).item())
            if cfg.use_activation_discrepancy:
                disc_all[k] = float(
                    activation_discrepancy(feats, self.source_stats, layers=self.criterion_layers).item()
                )
            if values[k] < best_fitness:
                best_fitness = values[k]
                best_logits = logits

        # (4) update the search distribution and return the best candidate.
        self.es.tell(solutions, values)
        self.iteration += 1
        assert best_logits is not None
        best_idx = int(np.argmin(values))
        prompt = self._prompt_tensor(solutions[best_idx])
        extra = {
            "fitness_mean": float(values.mean()),
            "fitness_min": float(values.min()),
            "fitness_max": float(values.max()),
            "entropy_mean": float(entropy_all.mean()) if cfg.use_entropy else float("nan"),
            "discrepancy_mean": float(disc_all.mean()) if cfg.use_activation_discrepancy else float("nan"),
            "sigma": float(self.es.sigma),
            "lam": float(lam),
        }
        return FOAStepOutput(
            logits=best_logits, prompt=prompt, fitness=values, shift=shift, extra=extra
        )

    # ----------------------------------------------------------------------------------
    def _lambda_for(self, batch_size: int) -> float:
        return self.cfg.resolved_lambda(batch_size)

    @torch.no_grad()
    def predict(
        self, images: torch.Tensor, prompt: Optional[torch.Tensor] = None, shift=None
    ) -> torch.Tensor:
        images = images.to(self.device)
        logits, _, _ = self.model.forward_with_prompt(images, prompt=prompt, shift=shift)
        return logits

    @torch.no_grad()
    def flush(self) -> Optional[FOAStepOutput]:  # pragma: no cover - nothing to flush
        """Plain FOA predicts every batch immediately, so there is nothing to flush."""
        return None


class FOAInterval:
    """FOA-I: interval update strategy for single-sample adaptation (Section 4.4).

    Instead of requiring a whole batch, FOA-I collects ``I`` consecutive test samples
    before performing one CMA update.  If ``store="feature"`` (``FOA-I V1``) the input
    token embeddings are cached, with ``store="image"`` (``FOA-I V2``) the raw images are
    cached; both produce the same predictions and only differ in memory footprint.
    """

    def __init__(
        self,
        model: AdaptableModel,
        source_stats: FeatureStatistics,
        cfg: Optional[FOAConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.cfg = cfg or FOAConfig()
        self.interval = max(1, int(self.cfg.interval))
        self.foa = FOA(model, source_stats, cfg=self.cfg, device=device)
        self.model = self.foa.model
        self.device = self.foa.device
        self.store = self.cfg.interval_store
        self._buffer_images: List[torch.Tensor] = []
        self._buffer_embed: List[torch.Tensor] = []

    def reset(self) -> None:
        self.foa.reset()
        self._buffer_images.clear()
        self._buffer_embed.clear()

    @torch.no_grad()
    def step(self, images: torch.Tensor) -> Optional[FOAStepOutput]:
        """Feed a batch of samples.

        Returns ``None`` while the interval is still being filled (the caller can use
        :meth:`predict_pending` to obtain provisional predictions), and the
        :class:`FOAStepOutput` of the whole interval - i.e. the predictions obtained with
        the freshly updated prompt - once ``I`` samples have been collected.
        """
        images = images.to(self.device)
        if self.interval == 1:
            return self.foa.step(images)
        if self.store == "feature":
            self._buffer_embed.extend(list(self.model.input_token_embeddings(images).unbind(0)))
        else:
            self._buffer_images.extend(list(images.unbind(0)))
        if len(self._current_buffer_size()) < self.interval:
            return None
        if self.store == "feature":
            batch = torch.stack(self._buffer_embed, dim=0)
            self._buffer_embed.clear()
            return self.foa.step(batch, is_tokens=True)
        batch = torch.stack(self._buffer_images, dim=0)
        self._buffer_images.clear()
        return self.foa.step(batch)

    @torch.no_grad()
    def predict_pending(self, images: torch.Tensor) -> torch.Tensor:
        """Provisional prediction for the samples that are still inside the interval."""
        return self.foa.predict(
            images.to(self.device), prompt=self.foa.mean_prompt, shift=self._current_shift()
        )

    @property
    def pending(self) -> int:
        return len(self._current_buffer_size())

    @torch.no_grad()
    def flush(self) -> Optional[FOAStepOutput]:
        """Adapt on the (possibly incomplete) interval at the end of the stream."""
        if self.pending == 0:
            return None
        if self.store == "feature":
            batch = torch.stack(self._buffer_embed, dim=0)
            self._buffer_embed.clear()
            return self.foa.step(batch, is_tokens=True)
        batch = torch.stack(self._buffer_images, dim=0)
        self._buffer_images.clear()
        return self.foa.step(batch)

    # -- helpers ---------------------------------------------------------------------
    def _current_buffer_size(self):
        return self._buffer_embed if self.store == "feature" else self._buffer_images

    def _current_shift(self):
        if not self.cfg.use_activation_shifting:
            return None
        direction = self.foa.shifting.direction
        return None if direction is None else self.cfg.shift_gamma * direction
