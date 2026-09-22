"""Forward-Optimization Adaptation (FOA) -- Algorithm 1 of the paper.

This module implements the *core* online test-time adaptation loop of
"Test-Time Model Adaptation with Only Forward Passes":

    Algorithm 1  Forward-Optimization Adaptation (FOA)
    --------------------------------------------------------------
    Input:  batches of test samples {X_t}_{t=1..T}, model f_Theta(.) = Head(L_i(.)),
            ID statistics {mu_i^S, sigma_i^S}_{i=0..N}, pop. size K.
            Initialize m^(0)=0, Sigma^(0)=I, tau^(0)=1 in Eqn. (6).
      for t = 1..T do
          Sampling K prompt solutions {p_k^t}_{k=1..K} by Eqn. (6).
          for k = 1..K do
              Calculate all layers' CLS features {e_n^0}_{n=1..N} using Eqn. (1)
              with input [p_k^t ; X_t].
              Adjust e_N^0 to source domain by Eqn. (7).
              Predict Y_hat_t^k by Head(e_N^0).
              Calculate fitness value v_k per Eqn. (5).
          end for
          Update m^(t), Sigma^(t), tau^(t) according to {v_k} using CMA-ES.
          Select final Y_hat_t from {Y_hat_t^k} with best v_k.
      end for
    Output: predictions {Y_hat_t}_{t=1..T}
    --------------------------------------------------------------

Paper hyper-parameters (Section 4 "Implementation Details" and Appendix B.2):

* ``N_p = 3`` prompt embeddings, uniform initialization;
* batch size ``BS = 64``;
* population size ``K = 28 = 4 + 3*log(prompt_dim)`` (Hansen, 2016);
* ``lambda`` in Eqn. (5) is ``0.4 * BS/64`` on ImageNet-C/V2/Sketch and
  ``0.2 * BS/64`` on ImageNet-R;
* moving average factor ``alpha = 0.1`` in Eqn. (9);
* step size ``gamma = 1.0`` in Eqn. (7);
* the source ID statistics are computed **without** the inserted prompt.

There is **no backward pass anywhere in this file**: the model stays in
``eval()`` mode with ``requires_grad_(False)`` on every parameter and all
forward passes run under ``torch.no_grad()``.  The only object that changes
over time is (a) the CMA-ES search distribution over the prompt and (b) the
EMA estimate ``mu_N(t)`` of the test feature center used by the
back-to-source activation shifting.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

__all__ = [
    "FOA",
    "FOARunner",
    "FOAState",
    "build_foa",
    "unpack_batch",
    "CANONICAL_BS",
]


CANONICAL_BS = 64

#: dataset names for which lambda_base = 0.4 * BS/64 (ImageNet-R uses 0.2)
_LAMBDA_04_DATASETS = (
    "imagenet-c",
    "imagenet_v2",
    "imagenet-v2",
    "imagenetv2",
    "imagenet-sketch",
    "imagenet_sketch",
    "imagenetsketch",
    "imagenet-1k",
    "imagenet",
)


# --------------------------------------------------------------------------- #
# small config / batch helpers
# --------------------------------------------------------------------------- #
def _cfg_get(cfg: Any, *keys: str, default: Any = None) -> Any:
    """Fetch a dotted key from a dict-like or attribute-like config."""
    if cfg is None:
        return default
    for key in keys:
        cur = cfg
        ok = True
        for part in key.split("."):
            if cur is None:
                ok = False
                break
            if isinstance(cur, dict):
                if part in cur:
                    cur = cur[part]
                else:
                    ok = False
                    break
            else:
                if hasattr(cur, part):
                    cur = getattr(cur, part)
                else:
                    ok = False
                    break
        if ok and cur is not None:
            return cur
    return default


def unpack_batch(batch: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Normalize a stream batch into ``(images, targets)``.

    Accepts dicts with keys ``image/images/x/pixel_values`` and
    ``label/labels/y/target``, plain ``(images, labels)`` tuples/lists, or a
    bare tensor of images.
    """
    targets: Optional[torch.Tensor] = None
    if isinstance(batch, dict):
        images = None
        for key in ("image", "images", "x", "pixel_values", "inputs"):
            if key in batch and batch[key] is not None:
                images = batch[key]
                break
        for key in ("label", "labels", "y", "target", "targets"):
            if key in batch and batch[key] is not None:
                targets = batch[key]
                break
        if images is None:
            raise KeyError(f"Cannot find images in batch with keys {list(batch)}")
    elif isinstance(batch, (list, tuple)):
        images = batch[0]
        targets = batch[1] if len(batch) > 1 else None
    else:
        images = batch

    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)
    if not isinstance(images, torch.Tensor):
        images = torch.as_tensor(images)
    if images.dim() == 3:
        images = images.unsqueeze(0)

    if targets is not None:
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets)
        elif not isinstance(targets, torch.Tensor):
            targets = torch.as_tensor(targets)
        if targets.dim() == 0:
            targets = targets.unsqueeze(0)
    return images, targets


class FOAState:
    """Mutable online state of the FOA loop (prompt + CMA + shifting EMA)."""

    __slots__ = ("batch_index", "mu_test", "best_fitness_history", "fitness_history")

    def __init__(self) -> None:
        self.batch_index: int = 0
        self.mu_test: Optional[torch.Tensor] = None  # mu_N(t) of Eqn. (9)
        self.best_fitness_history: List[float] = []
        self.fitness_history: List[List[float]] = []


# --------------------------------------------------------------------------- #
# main class
# --------------------------------------------------------------------------- #
class FOA:
    """Forward-only, weight-frozen prompt adaptation (Algorithm 1).

    Parameters
    ----------
    model:
        Frozen backbone exposing ``forward_with_features(images, prompt)`` ->
        dict with ``cls_features`` (list, index 0 = patch-embedding stage,
        index n = block n output), ``final_cls`` and ``logits``, plus
        ``head(cls_feature)`` and ``embed_dim``.  Typically
        :class:`src.models.vit_loader.ViTWithCLSFeatures`.
    prompt:
        :class:`src.models.prompt_injection.PromptInjection` instance.  Built
        from ``cfg`` when omitted.
    optimizer:
        CMA-ES wrapper (:class:`src.method.cma_wrapper.CMAOptimizer`).  Built
        from ``cfg`` when omitted.
    fitness:
        :class:`src.method.fitness.FitnessFunction` implementing Eqn. (5).
        Built from ``cfg`` + ``source_stats`` when omitted.
    shifter:
        :class:`src.method.activation_shifting.ActivationShifting`.  Built from
        ``cfg`` + ``source_stats`` when omitted; set to ``None`` (or pass
        ``shifting_enabled=False``) to reproduce the Table 5 ablation without
        activation shifting.
    source_stats:
        Source ID statistics bank ``{mu_i^S, sigma_i^S}_{i=0..N}`` (Eqn. 5/7).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        prompt: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        fitness: Optional[Any] = None,
        shifter: Optional[Any] = None,
        source_stats: Optional[Any] = None,
        cfg: Optional[Any] = None,
        device: Optional[Any] = None,
        *,
        population_size: Optional[int] = None,
        num_prompts: Optional[int] = None,
        lambda_value: Optional[float] = None,
        lambda_base: Optional[float] = None,
        gamma: Optional[float] = None,
        alpha: Optional[float] = None,
        shifting_enabled: Optional[bool] = None,
        seed: Optional[int] = None,
        dataset: Optional[str] = None,
        shift_features_in_discrepancy: bool = False,
    ) -> None:
        self.cfg = cfg
        self.device = self._resolve_device(model, device)
        self.model = model
        self.model.to(self.device)
        self.model.eval()
        self._freeze(self.model)

        # ---------------- dataset / lambda (Eqn. 5 trade-off) --------------- #
        self.dataset = (dataset or _cfg_get(cfg, "data.dataset", default="imagenet-c") or "imagenet-c")
        bs = int(_cfg_get(cfg, "data.batch_size", default=CANONICAL_BS) or CANONICAL_BS)
        self.batch_size = bs
        base = lambda_base
        if base is None:
            base = _cfg_get(cfg, "fitness.lambda_base", "fitness.lambda", default=None)
        if base is None:
            base = 0.2 if "imagenet-r" in str(self.dataset).lower() else 0.4
        scale = bool(_cfg_get(cfg, "fitness.lambda_scale_with_bs", default=True))
        resolved = float(base) * (bs / CANONICAL_BS) if scale else float(base)
        if lambda_value is None:
            lambda_value = _cfg_get(cfg, "fitness.lambda_value", default=None)
        self.lambda_base = float(base)
        self.lambda_value = float(lambda_value) if lambda_value is not None else resolved

        # ---------------------------- prompt -------------------------------- #
        if prompt is None:
            from ..models.prompt_injection import build_prompt

            prompt = build_prompt(
                embed_dim=int(getattr(model, "embed_dim", 768)),
                num_prompts=int(num_prompts if num_prompts is not None
                                else _cfg_get(cfg, "prompt.num_prompts", default=3)),
                init=_cfg_get(cfg, "prompt.init", default="uniform"),
                init_range=float(_cfg_get(cfg, "prompt.init_range", default=0.01)),
                seed=int(seed if seed is not None else _cfg_get(cfg, "seed", "prompt.seed", default=0) or 0),
                device=self.device,
            )
        self.prompt = prompt
        self.prompt_dim = int(getattr(self.prompt, "prompt_dim", 0))
        self.num_prompts = int(getattr(self.prompt, "num_prompts", 0))

        # ---------------------------- CMA-ES -------------------------------- #
        if optimizer is None:
            from .cma_wrapper import build_cma_optimizer

            optimizer = build_cma_optimizer(cfg=cfg, dim=self.prompt_dim)
        self.optimizer = optimizer
        if population_size is not None:
            try:
                object.__setattr__(optimizer, "population_size", int(population_size))
            except Exception:  # pragma: no cover - defensive
                pass
        self.population_size = int(
            population_size
            if population_size is not None
            else getattr(self.optimizer, "population_size", 28)
        )

        # ---------------------- source statistics --------------------------- #
        self.source_stats = source_stats

        # --------------------------- fitness -------------------------------- #
        if fitness is None and source_stats is not None:
            from .fitness import build_fitness

            fitness = build_fitness(
                source_stats,
                cfg,
                dataset=self.dataset,
                lambda_value=self.lambda_value,
            )
        self.fitness = fitness

        # ----------------------- activation shifting ------------------------- #
        if shifting_enabled is None:
            shifting_enabled = bool(_cfg_get(cfg, "shifting.enabled", default=True))
        if gamma is None:
            gamma = _cfg_get(cfg, "shifting.gamma", default=1.0)
        if alpha is None:
            alpha = _cfg_get(cfg, "shifting.alpha", default=0.1)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        if shifter is None and shifting_enabled and source_stats is not None:
            from .activation_shifting import build_activation_shifter

            shifter = build_activation_shifter(
                source_stats, cfg, enabled=True, alpha=self.alpha, gamma=self.gamma,
                device=self.device,
            )
        self.shifter = shifter if shifting_enabled else None
        self.shifting_enabled = self.shifter is not None and self.gamma != 0.0
        self.shift_features_in_discrepancy = bool(shift_features_in_discrepancy)

        # mu_N^S (Eqn. 7 / Eqn. 8)
        self.source_mean = self._resolve_source_mean()
        if self.shifting_enabled and self.source_mean is None:
            logger.warning(
                "Activation shifting requested but no source mean mu_N^S could be "
                "resolved; disabling shifting."
            )
            self.shifting_enabled = False
        if self.source_mean is not None:
            self.source_mean = self.source_mean.to(self.device).float()

        # ------------------------------ state ------------------------------- #
        self.state = FOAState()
        self.ece_bins = int(_cfg_get(cfg, "eval.ece_bins", default=15) or 15)
        self._shift_snapshot: Optional[torch.Tensor] = None
        self.wall_clock_s = 0.0

    # ------------------------------------------------------------------ #
    # construction helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _freeze(module: torch.nn.Module) -> None:
        for p in module.parameters():
            p.requires_grad_(False)
        for p in getattr(module, "buffers", lambda: [])():
            pass

    def _resolve_device(self, model: torch.nn.Module, device: Any) -> torch.device:
        if device is None:
            try:
                return next(model.parameters()).device
            except StopIteration:  # pragma: no cover
                return torch.device("cpu")
        dev = torch.device(device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; falling back to CPU.")
            return torch.device("cpu")
        return dev

    def _resolve_source_mean(self) -> Optional[torch.Tensor]:
        """Return mu_N^S from the shifter or from the statistics bank."""
        if self.shifter is not None:
            for attr in ("source_mean", "mu_N", "mu_S", "mean_source"):
                val = getattr(self.shifter, attr, None)
                if isinstance(val, torch.Tensor):
                    return val.detach().clone()
        stats = self.source_stats
        if stats is None:
            return None
        for attr in ("mu_final", "mu_N"):
            val = getattr(stats, attr, None)
            if isinstance(val, torch.Tensor):
                return val.detach().clone()
        mu = getattr(stats, "mu", None)
        if isinstance(mu, (list, tuple)) and len(mu) > 0:
            return torch.as_tensor(mu[-1]).detach().clone()
        return None

    # ------------------------------------------------------------------ #
    # activation shifting (Eqn. 7, 8, 9)
    # ------------------------------------------------------------------ #
    def _batch_mean(self, final_cls: torch.Tensor) -> torch.Tensor:
        """mu_N(X_t): mean of the N-th layer's CLS feature over batch X_t."""
        if self.shifter is not None and hasattr(self.shifter, "batch_mean"):
            try:
                return self.shifter.batch_mean(final_cls)
            except Exception:  # pragma: no cover - defensive
                pass
        return final_cls.mean(dim=0)

    def _current_test_mean(self, final_cls: torch.Tensor) -> torch.Tensor:
        """mu_N(t-1): EMA estimate of the test feature center (Eqn. 9).

        For the very first batch there is no history yet, so mu_N(0) is
        initialized with the first batch's own mean, i.e. mu_N(X_1)
        (documented default; the per-batch order is always
        ``initialize -> shift -> EMA update``).
        """
        if self.state.mu_test is None:
            return self._batch_mean(final_cls)
        return self.state.mu_test

    def shift_features(self, final_cls: torch.Tensor) -> torch.Tensor:
        """Apply Eqn. (7): ``e_N^0 <- e_N^0 + gamma * d`` with

        ``d_t = mu_N^S - mu_N(t-1)``   (Eqn. 8).
        """
        if not self.shifting_enabled or self.source_mean is None:
            return final_cls
        mean_prev = self._current_test_mean(final_cls)
        direction = self.source_mean.to(final_cls.dtype) - mean_prev.to(final_cls.dtype)
        return final_cls + self.gamma * direction

    def update_test_mean(self, final_cls: torch.Tensor) -> None:
        """Eqn. (9): ``mu_N(t) = alpha * mu_N(X_t) + (1 - alpha) * mu_N(t-1)``.

        Uses the **un-shifted** batch statistics of the selected candidate.
        """
        if not self.shifting_enabled:
            return
        batch_mean = self._batch_mean(final_cls).detach()
        if self.state.mu_test is None:
            self.state.mu_test = batch_mean.clone()
        else:
            self.state.mu_test = (
                self.alpha * batch_mean + (1.0 - self.alpha) * self.state.mu_test
            ).detach()
        if self.shifter is not None and hasattr(self.shifter, "update"):
            try:  # keep the reference shifter object in sync (optional)
                self.shifter.update(final_cls)
            except Exception:  # pragma: no cover - defensive
                pass

    def _snapshot_shift_state(self) -> Optional[torch.Tensor]:
        """Snapshot mu_N(t-1) so every candidate is scored against the same d_t."""
        if not self.shifting_enabled:
            return None
        if self.state.mu_test is None:
            return None
        return self.state.mu_test.detach().clone()

    def _restore_shift_state(self, snapshot: Optional[torch.Tensor]) -> None:
        if not self.shifting_enabled:
            return
        self.state.mu_test = None if snapshot is None else snapshot.detach().clone()

    # ------------------------------------------------------------------ #
    # candidate evaluation (inner loop of Algorithm 1)
    # ------------------------------------------------------------------ #
    def evaluate_candidate(
        self, prompt_vector: Any, images: torch.Tensor
    ) -> Dict[str, Any]:
        """Forward one candidate prompt and compute its fitness value v_k."""
        images = images.to(self.device, non_blocking=True)
        with torch.no_grad():
            if self.prompt is not None:
                self.prompt.set_prompt(np.asarray(prompt_vector, dtype=np.float32))
                prompt_tensor = self.prompt.as_tensor()
            else:
                prompt_tensor = None

            out = self.model.forward_with_features(images, prompt=prompt_tensor)
            cls_features = out["cls_features"] if isinstance(out, dict) else out[0]
            final_cls = (
                out["final_cls"]
                if isinstance(out, dict) and out.get("final_cls", None) is not None
                else cls_features[-1]
            )

            # Eqn. (7): adjust e_N^0 to the source domain before the head.
            shifted_final = self.shift_features(final_cls)
            logits = self.model.head(shifted_final)

            # Eqn. (5): entropy on the shifted prediction + activation discrepancy
            # between the current test-batch CLS statistics and the source ones.
            discrepancy_features = cls_features
            if self.shift_features_in_discrepancy:
                discrepancy_features = list(cls_features[:-1]) + [shifted_final]
            terms = self._fitness_terms(logits, discrepancy_features)

            result: Dict[str, Any] = {
                "fitness": float(terms["total"]),
                "logits": logits.detach(),
                "final_cls": final_cls.detach(),
                "shifted_final": shifted_final.detach(),
                "cls_features": [f.detach() for f in cls_features],
                "terms": terms.get("terms"),
                "entropy": float(terms.get("entropy", 0.0)),
                "discrepancy": float(terms.get("discrepancy", 0.0)),
            }
        return result

    def _fitness_terms(self, logits: torch.Tensor, cls_features: Sequence[torch.Tensor]) -> Dict[str, Any]:
        """Robustly invoke the Eqn. (5) fitness function."""
        fit = self.fitness
        if fit is None:
            raise RuntimeError(
                "FOA requires a fitness function implementing Eqn. (5) "
                "(src/method/fitness.py) or a source statistics bank."
            )
        if hasattr(fit, "evaluate"):
            terms = fit.evaluate(logits, cls_features)
            if hasattr(terms, "total"):
                return {
                    "total": float(terms.total),
                    "entropy": float(getattr(terms, "entropy", 0.0)),
                    "discrepancy": float(getattr(terms, "discrepancy", 0.0)),
                    "terms": terms,
                }
            if isinstance(terms, dict):
                return {
                    "total": float(terms.get("total", terms.get("fitness", 0.0))),
                    "entropy": float(terms.get("entropy", 0.0)),
                    "discrepancy": float(terms.get("discrepancy", 0.0)),
                    "terms": terms,
                }
            return {"total": float(terms), "entropy": 0.0, "discrepancy": 0.0, "terms": terms}
        # callable fallback: fitness(logits, cls_features) -> scalar (or terms)
        value = fit(logits, cls_features)
        if hasattr(value, "total"):
            return {
                "total": float(value.total),
                "entropy": float(getattr(value, "entropy", 0.0)),
                "discrepancy": float(getattr(value, "discrepancy", 0.0)),
                "terms": value,
            }
        return {"total": float(value), "entropy": 0.0, "discrepancy": 0.0, "terms": value}

    # ------------------------------------------------------------------ #
    # outer loop of Algorithm 1 (one test batch)
    # ------------------------------------------------------------------ #
    def adapt_batch(
        self, images: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Run one full CMA iteration on ``X_t`` and return the batch prediction."""
        t0 = time.time()
        images = images.to(self.device, non_blocking=True)

        # d_t must be identical for every candidate -> freeze mu_N(t-1).
        snapshot = self._snapshot_shift_state()

        if self.optimizer is not None:
            solutions = self.optimizer.ask(self.population_size)
        else:  # pragma: no cover - prompt-only inference
            solutions = np.zeros((1, max(self.prompt_dim, 1)), dtype=np.float32)
        solutions = np.atleast_2d(np.asarray(solutions, dtype=np.float64))

        results: List[Dict[str, Any]] = []
        for k in range(solutions.shape[0]):
            self._restore_shift_state(snapshot)
            results.append(self.evaluate_candidate(solutions[k], images))

        values = np.asarray([r["fitness"] for r in results], dtype=np.float64)
        best_index = self._best_index(values)

        # Update the CMA-ES distribution parameters (m, tau, Sigma) with {v_k}.
        if self.optimizer is not None and hasattr(self.optimizer, "tell"):
            try:
                self.optimizer.tell(values.tolist(), [np.asarray(s, dtype=np.float64) for s in solutions])
            except TypeError:
                self.optimizer.tell(values.tolist())

        # Final prediction = the candidate with the best (lowest) fitness.
        best = results[best_index]

        # Eqn. (9): EMA update of the test center with the *un-shifted* statistics
        # of the selected candidate (order: initialize -> shift -> update).
        self._restore_shift_state(snapshot)
        self.update_test_mean(best["final_cls"])

        self.state.batch_index += 1
        self.state.best_fitness_history.append(float(best["fitness"]))
        self.state.fitness_history.append(values.tolist())

        logits = best["logits"]
        out: Dict[str, Any] = {
            "logits": logits,
            "predictions": logits.argmax(dim=-1) if logits.dim() > 1 else logits.argmax().reshape(1),
            "fitness": float(best["fitness"]),
            "fitness_values": values,
            "best_index": int(best_index),
            "entropy": float(best.get("entropy", 0.0)),
            "discrepancy": float(best.get("discrepancy", 0.0)),
            "terms": best.get("terms"),
            "prompt": np.asarray(solutions[best_index], dtype=np.float32),
            "cls_features": best["cls_features"],
            "final_cls": best["final_cls"],
            "targets": targets,
            "batch_index": self.state.batch_index - 1,
            "elapsed_s": time.time() - t0,
        }
        return out

    def _best_index(self, values: np.ndarray) -> int:
        if self.optimizer is not None and hasattr(self.optimizer, "best_index"):
            try:
                return int(self.optimizer.best_index(values))
            except Exception:  # pragma: no cover - defensive
                pass
        finite = np.where(np.isfinite(values), values, np.inf)
        return int(np.argmin(finite))

    # ------------------------------------------------------------------ #
    # episode container
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Reset the online state (new test domain / new CMA episode)."""
        self.state = FOAState()
        if self.optimizer is not None and hasattr(self.optimizer, "reset"):
            self.optimizer.reset()
        if self.shifter is not None and hasattr(self.shifter, "reset"):
            try:
                self.shifter.reset()
            except Exception:  # pragma: no cover
                pass

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Frozen-model (no adaptation) prediction -- the NoAdapt reference."""
        images = images.to(self.device, non_blocking=True)
        with torch.no_grad():
            out = self.model.forward_with_features(images, prompt=None)
            logits = out["logits"] if isinstance(out, dict) else out[1]
        return logits

    # ------------------------------------------------------------------ #
    # streaming evaluation
    # ------------------------------------------------------------------ #
    def run(
        self,
        stream: Iterable[Any],
        accumulator: Optional[Any] = None,
        verbose: bool = False,
        log_every: int = 20,
        max_batches: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Evaluate Algorithm 1 over an ordered, single-pass test stream."""
        if accumulator is None:
            accumulator = _make_accumulator(self.ece_bins)

        t0 = time.time()
        num_samples = 0
        per_batch: List[Dict[str, Any]] = []
        for i, batch in enumerate(stream):
            if max_batches is not None and i >= max_batches:
                break
            images, targets = unpack_batch(batch)
            out = self.adapt_batch(images, targets=targets)
            logits = out["logits"]
            num_samples += int(logits.shape[0])
            if targets is not None and accumulator is not None:
                accumulator.update(logits, targets)
            per_batch.append(
                {
                    "batch_index": out["batch_index"],
                    "fitness": out["fitness"],
                    "best_index": out["best_index"],
                    "elapsed_s": out["elapsed_s"],
                }
            )
            if verbose and log_every and (i + 1) % log_every == 0:
                logger.info(
                    "[FOA] batch %d | best fitness %.4f | elapsed %.1fs",
                    i + 1, out["fitness"], time.time() - t0,
                )

        self.wall_clock_s = time.time() - t0
        summary: Dict[str, Any] = {
            "accuracy": None,
            "ece": None,
            "num_samples": num_samples,
            "num_batches": self.state.batch_index,
            "population_size": self.population_size,
            "num_prompts": self.num_prompts,
            "prompt_dim": self.prompt_dim,
            "lambda": self.lambda_value,
            "shifting_enabled": bool(self.shifting_enabled),
            "gamma": self.gamma,
            "alpha": self.alpha,
            "wall_clock_s": self.wall_clock_s,
            "per_batch": per_batch,
        }
        if accumulator is not None:
            metrics = accumulator.compute()
            summary["accuracy"] = float(metrics.get("accuracy", float("nan")))
            summary["ece"] = float(metrics.get("ece", float("nan")))
            summary["num_samples"] = int(metrics.get("num_samples", num_samples))
        return summary

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def state_dict(self) -> Dict[str, Any]:
        sd: Dict[str, Any] = {
            "batch_index": self.state.batch_index,
            "mu_test": None if self.state.mu_test is None else self.state.mu_test.detach().cpu(),
            "best_fitness_history": list(self.state.best_fitness_history),
            "prompt": self.prompt.get_prompt() if self.prompt is not None else None,
        }
        if self.optimizer is not None and hasattr(self.optimizer, "state_dict"):
            try:
                sd["cma"] = self.optimizer.state_dict()
            except Exception:  # pragma: no cover
                pass
        return sd

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.state.batch_index = int(state.get("batch_index", 0))
        mu = state.get("mu_test", None)
        self.state.mu_test = None if mu is None else torch.as_tensor(mu).to(self.device)
        self.state.best_fitness_history = list(state.get("best_fitness_history", []))
        if state.get("prompt") is not None and self.prompt is not None:
            self.prompt.set_prompt(np.asarray(state["prompt"], dtype=np.float32))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"FOA(dataset={self.dataset!r}, K={self.population_size}, "
            f"N_p={self.num_prompts}, prompt_dim={self.prompt_dim}, "
            f"lambda={self.lambda_value:.4f}, gamma={self.gamma}, alpha={self.alpha}, "
            f"shifting={self.shifting_enabled})"
        )


#: convenience alias used by some runner scripts
FOARunner = FOA


# --------------------------------------------------------------------------- #
# metrics helper (thin wrapper so foa.py has no hard dependency on eval/)
# --------------------------------------------------------------------------- #
def _make_accumulator(n_bins: int = 15) -> Optional[Any]:
    try:
        from ..eval.metrics import MetricAccumulator

        return MetricAccumulator(n_bins=n_bins)
    except Exception:  # pragma: no cover - defensive
        pass
    try:
        from src.eval.metrics import MetricAccumulator  # type: ignore

        return MetricAccumulator(n_bins=n_bins)
    except Exception:  # pragma: no cover - defensive
        return None


# --------------------------------------------------------------------------- #
# factories
# --------------------------------------------------------------------------- #
def build_foa(
    cfg: Optional[Any] = None,
    source_stats: Optional[Any] = None,
    model: Optional[torch.nn.Module] = None,
    device: Optional[Any] = None,
    *,
    population_size: Optional[int] = None,
    num_prompts: Optional[int] = None,
    lambda_value: Optional[float] = None,
    lambda_base: Optional[float] = None,
    gamma: Optional[float] = None,
    alpha: Optional[float] = None,
    shifting_enabled: Optional[bool] = None,
    use_entropy: Optional[bool] = None,
    use_discrepancy: Optional[bool] = None,
    prompt: Optional[Any] = None,
    optimizer: Optional[Any] = None,
    fitness: Optional[Any] = None,
    shifter: Optional[Any] = None,
    **kwargs: Any,
) -> FOA:
    """Config-driven factory assembling the full Algorithm 1 pipeline."""
    cfg = cfg if cfg is not None else {}
    dataset = kwargs.pop("dataset", None) or _cfg_get(cfg, "data.dataset", default="imagenet-c")

    # ------------------------------ backbone ------------------------------- #
    if model is None:
        from ..models.vit_loader import build_vit

        model = build_vit(
            model_name=_cfg_get(cfg, "model.name", default="vit_base_patch16_224"),
            checkpoint=_cfg_get(cfg, "model.checkpoint", default=None),
            pretrained=bool(_cfg_get(cfg, "model.pretrained", default=True)),
            num_classes=int(_cfg_get(cfg, "model.num_classes", default=1000)),
            device=device if device is not None else _cfg_get(cfg, "model.device", default="cuda"),
        )
    dev = device if device is not None else getattr(model, "device", None)

    # --------------------------- source statistics -------------------------- #
    if source_stats is None:
        path = _cfg_get(cfg, "source_stats.path", default=None)
        if path:
            try:
                from .source_stats import load_source_stats

                source_stats = load_source_stats(path, device="cpu")
            except Exception as exc:  # pragma: no cover - user guidance
                logger.warning(
                    "Could not load source statistics from %s (%s). "
                    "Run scripts/compute_source_stats.py first.", path, exc,
                )

    # -------------------------------- fitness ------------------------------- #
    if fitness is None and source_stats is not None:
        from .fitness import build_fitness

        overrides: Dict[str, Any] = {}
        if use_entropy is not None:
            overrides["use_entropy"] = use_entropy
        if use_discrepancy is not None:
            overrides["use_discrepancy"] = use_discrepancy
        fitness = build_fitness(
            source_stats, cfg, dataset=dataset, lambda_value=lambda_value, **overrides
        )

    return FOA(
        model=model,
        prompt=prompt,
        optimizer=optimizer,
        fitness=fitness,
        shifter=shifter,
        source_stats=source_stats,
        cfg=cfg,
        device=dev,
        population_size=population_size,
        num_prompts=num_prompts,
        lambda_value=lambda_value,
        lambda_base=lambda_base,
        gamma=gamma,
        alpha=alpha,
        shifting_enabled=shifting_enabled,
        dataset=dataset,
        **kwargs,
    )


def run_algorithm1(stream: Iterable[Any], **kwargs: Any) -> Dict[str, Any]:
    """Convenience helper: ``build_foa(**kwargs).run(stream)``."""
    foa = build_foa(**kwargs)
    return foa.run(stream)
