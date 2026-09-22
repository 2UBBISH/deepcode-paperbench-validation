"""Unsupervised fitness function for FOA's CMA-based prompt adaptation.

This module implements Eqn. (5) of the paper:

    L(f_Theta(p ; X_t)) = sum_{x in X_t} sum_{c in C} -yhat_c log yhat_c
                          + lambda * sum_{i=1}^{N} ( || mu_i(X_t) - mu_i^S ||_2
                                                     + || sigma_i(X_t) - sigma_i^S ||_2 )

i.e. the per-sample prediction entropy of the *current* batch plus a
lambda-weighted, per-layer L2 discrepancy between the test-batch CLS activation
statistics ``{mu_i(X_t), sigma_i(X_t)}`` and the source in-distribution
statistics ``{mu_i^S, sigma_i^S}`` collected offline
(see :mod:`src.method.source_stats`).

Key paper specifications (Sec. 3.1, Sec. 4.3, App. B.2, App. C):

* The discrepancy term sums BOTH mean and standard-deviation terms over layers
  ``i = 1 .. N`` (layer 0 is stored but not used by Eqn. (5)).
* Test-batch statistics are computed directly from the current batch, i.e. no
  EMA on the fitness statistics: ``beta = 1.0`` (Table 15 -- "using
  mu_i(X_t) and sigma_i(X_t) (i.e., beta = 1.0) for Eqn. (5) achieves
  remarkable performance on both ImageNet-C and ImageNet-R, without requiring
  additional hyperparameter").  The optional ``beta < 1.0`` EMA variant of
  Table 15 is supported for completeness but NOT used by default.
* ``lambda = 0.4 * BS / 64`` on ImageNet-C / V2 / Sketch and
  ``0.2 * BS / 64`` on ImageNet-R (App. B.2).
* For the SGD experiments of Table 9 the entropy term is divided by the batch
  size of 64 and ``lambda = 30``.

Component ablations of Table 5 (entropy only / discrepancy only / both, with or
without activation shifting) are obtained through the ``use_entropy`` /
``use_discrepancy`` flags rather than through separate code paths, so that the
reduction order -- and therefore the CMA ranking -- is bit-identical across
variants.

Nothing in this file ever enables gradients or touches model parameters: FOA is
backpropagation-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .source_stats import (
    SourceStats,
    activation_discrepancy,
    batch_statistics,
)

__all__ = [
    "FitnessConfig",
    "FitnessFunction",
    "FitnessTerms",
    "entropy_term",
    "sample_entropy",
    "discrepancy_terms",
    "build_fitness",
    "resolve_lambda",
    "LAMBDA_BASE_IMAGENETC",
    "LAMBDA_BASE_IMAGENETR",
]

# ---------------------------------------------------------------------------
# Paper constants (App. B.2)
# ---------------------------------------------------------------------------

#: lambda base for ImageNet-C / ImageNet-V2 / ImageNet-Sketch.
LAMBDA_BASE_IMAGENETC = 0.4
#: lambda base for ImageNet-R (App. B.2: "0.2 x BS/64 on ImageNet-R").
LAMBDA_BASE_IMAGENETR = 0.2
#: Canonical FOA batch size used to normalise lambda.
CANONICAL_BS = 64
#: Datasets that use lambda_base = 0.4.
_LAMBDA_04_DATASETS = ("imagenet-c", "c", "imagenet-v2", "imagenet-v2-mf", "v2", "sketch", "imagenet-sketch", "imagenet-1k", "in")
#: Datasets that use lambda_base = 0.2.
_LAMBDA_02_DATASETS = ("imagenet-r", "r")


def resolve_lambda(
    batch_size: int = CANONICAL_BS,
    lambda_base: float = LAMBDA_BASE_IMAGENETC,
    dataset: Optional[str] = None,
    lambda_scale_with_bs: bool = True,
    lambda_value: Optional[float] = None,
) -> float:
    """Resolve the Eqn. (5) trade-off parameter ``lambda``.

    App. B.2: "The lambda in Eqn. (5) is set to ``0.4 x BS/64`` on
    ImageNet-C/V2/Sketch, and ``0.2 x BS/64`` on ImageNet-R to balance the
    magnitude of two losses."

    Parameters
    ----------
    batch_size:
        Current test batch size ``BS``.
    lambda_base:
        Base value, 0.4 for C/V2/Sketch, 0.2 for R.  Overridden by ``dataset``
        when the dataset name is recognised.
    dataset:
        Optional dataset name; used to pick the base value automatically.
    lambda_scale_with_bs:
        If True (default) multiply by ``BS / 64``; if False use the base as-is.
    lambda_value:
        Explicit override (wins over everything).  Used by the sensitivity
        sweep of Table 13.
    """

    if lambda_value is not None:
        return float(lambda_value)

    if dataset is not None:
        key = str(dataset).lower().replace("_", "-")
        if key in _LAMBDA_02_DATASETS:
            lambda_base = LAMBDA_BASE_IMAGENETR
        elif key in _LAMBDA_04_DATASETS:
            lambda_base = LAMBDA_BASE_IMAGENETC

    lam = float(lambda_base)
    if lambda_scale_with_bs:
        lam *= float(batch_size) / float(CANONICAL_BS)
    return lam


# ---------------------------------------------------------------------------
# Term-wise API (used by the ablation script)
# ---------------------------------------------------------------------------


def sample_entropy(logits: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Per-sample prediction entropy of a logit tensor.

    ``logits`` has shape ``[B, C]``; returns ``[B]`` with
    ``H = -sum_c yhat_c log yhat_c`` where ``yhat = softmax(logits)`` -- the
    left-hand term of Eqn. (5) where ``yhat_c`` is "the c-th element of yhat in
    Eqn. (2) w.r.t. sample x".  The sum over classes is computed in float64 for
    a candidate-independent, deterministic reduction.
    """

    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    # Guard against log(0) * 0 -> nan: use the standard convention 0 * log 0 = 0.
    ent = -(probs * log_probs.clamp_min(math.log(eps)))
    return ent.sum(dim=-1, dtype=torch.float64)


def entropy_term(
    logits: torch.Tensor,
    reduction: str = "sum",
    eps: float = 1e-12,
) -> torch.Tensor:
    """Sum of per-sample prediction entropies over the batch (Eqn. 5, term 1).

    ``reduction="sum"`` follows Eqn. (5) exactly (``sum_{x in X_t}``).
    ``reduction="mean"`` is provided for the Table-9 SGD variant where the
    entropy loss "is divided by the batch size of 64".
    """

    ent = sample_entropy(logits, eps=eps)
    if reduction == "sum":
        return ent.sum()
    if reduction == "mean":
        return ent.mean()
    raise ValueError(f"Unknown entropy reduction '{reduction}'")


def discrepancy_terms(
    cls_features: Sequence[torch.Tensor],
    source_stats: SourceStats,
    layer_start: int = 1,
    layer_end: Optional[int] = None,
    unbiased: bool = False,
    mu_test: Optional[Sequence[torch.Tensor]] = None,
    sigma_test: Optional[Sequence[torch.Tensor]] = None,
) -> Dict[str, Any]:
    """Per-layer L2 activation discrepancies between test batch and source ID.

    Returns a dict with

    * ``"mu"``: list of ``|| mu_i(X_t) - mu_i^S ||_2`` for ``i = layer_start..layer_end``
    * ``"sigma"``: list of ``|| sigma_i(X_t) - sigma_i^S ||_2`` for the same range
    * ``"mu_sum"``, ``"sigma_sum"``: their (unweighted) sums
    * ``"total"``: ``mu_sum + sigma_sum`` i.e. the UNWEIGHTED bracket of Eqn. (5)

    The ``lambda`` weighting is applied by :class:`FitnessFunction`, so the
    ablation script can inspect individual layers.
    """

    if mu_test is None or sigma_test is None:
        mu_test, sigma_test = batch_statistics(cls_features, unbiased=unbiased)

    n_layers = min(len(mu_test), int(source_stats.num_layers))
    end = n_layers if layer_end is None else min(int(layer_end), n_layers)

    mu_terms: List[torch.Tensor] = []
    sigma_terms: List[torch.Tensor] = []
    for i in range(int(layer_start), end):
        diff_mu = (mu_test[i].double() - source_stats.mu_i(i, device=mu_test[i].device).double())
        diff_sigma = (sigma_test[i].double() - source_stats.sigma_i(i, device=sigma_test[i].device).double())
        # Exact L2 norm (not squared), as written in Eqn. (5).
        mu_terms.append(torch.linalg.vector_norm(diff_mu, ord=2))
        sigma_terms.append(torch.linalg.vector_norm(diff_sigma, ord=2))

    mu_sum = torch.stack(mu_terms).sum() if mu_terms else torch.zeros((), dtype=torch.float64)
    sigma_sum = torch.stack(sigma_terms).sum() if sigma_terms else torch.zeros((), dtype=torch.float64)

    return {
        "mu": mu_terms,
        "sigma": sigma_terms,
        "mu_sum": mu_sum,
        "sigma_sum": sigma_sum,
        "total": mu_sum + sigma_sum,
        "layers": list(range(int(layer_start), end)),
    }


# ---------------------------------------------------------------------------
# Configuration / result containers
# ---------------------------------------------------------------------------


@dataclass
class FitnessConfig:
    """Hyper-parameters of Eqn. (5)."""

    #: Base lambda; dataset-aware resolution happens in ``resolve_lambda``.
    lambda_base: float = LAMBDA_BASE_IMAGENETC
    #: Explicit lambda override (Table 13 sensitivity sweep).
    lambda_value: Optional[float] = None
    #: Multiply lambda by BS/64 (App. B.2).
    lambda_scale_with_bs: bool = True
    #: Dataset name used to choose the lambda base.
    dataset: Optional[str] = None
    #: EMA balance for the fitness statistics; 1.0 == batch statistics (App. C).
    beta: float = 1.0
    #: Eqn. (5) left term.
    use_entropy: bool = True
    #: Eqn. (5) right term.
    use_discrepancy: bool = True
    #: "sum" (Eqn. 5) or "mean" (Table 9 SGD variant divides entropy by BS=64).
    entropy_reduction: str = "sum"
    #: Must stay "sum" to match Eqn. (5); kept explicit for clarity.
    discrepancy_reduction: str = "sum"
    #: Sum over layers i = 1..N (layer 0 stored but unused by Eqn. 5).
    layer_start: int = 1
    #: Inclusive-exclusive upper bound; None -> N (all layers after layer_start).
    layer_end: Optional[int] = None
    #: Numerics.
    eps: float = 1e-6
    #: Use Bessel-corrected std for the batch statistics (paper: population std).
    unbiased: bool = False
    #: Table 9 variant: divide entropy by 64 and use lambda = 30.
    sgd_variant: bool = False

    def resolved_lambda(self, batch_size: int = CANONICAL_BS) -> float:
        if self.sgd_variant:
            # App. B.2: "For updating prompts and normalization layers with SGD
            # optimizer in Table 9 with Eqn. (5), the entropy loss is divided by
            # the batch size of 64 and the lambda is set to 30."
            base = 30.0 if self.lambda_value is None else float(self.lambda_value)
            return base
        return resolve_lambda(
            batch_size=batch_size,
            lambda_base=self.lambda_base,
            dataset=self.dataset,
            lambda_scale_with_bs=self.lambda_scale_with_bs,
            lambda_value=self.lambda_value,
        )

    def effective_entropy_reduction(self) -> str:
        if self.sgd_variant and self.entropy_reduction == "sum":
            # App. B.2 for the Table-9 SGD variant: entropy loss / BS (=64).
            return "mean"
        return self.entropy_reduction

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FitnessTerms:
    """Term-wise breakdown of Eqn. (5) for one candidate prompt."""

    total: torch.Tensor
    entropy: torch.Tensor
    discrepancy: torch.Tensor
    mu_sum: torch.Tensor
    sigma_sum: torch.Tensor
    lambda_value: float
    per_layer_mu: List[torch.Tensor] = field(default_factory=list)
    per_layer_sigma: List[torch.Tensor] = field(default_factory=list)
    batch_size: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "fitness": float(self.total.detach().cpu()),
            "entropy": float(self.entropy.detach().cpu()),
            "discrepancy": float(self.discrepancy.detach().cpu()),
            "mu_discrepancy": float(self.mu_sum.detach().cpu()),
            "sigma_discrepancy": float(self.sigma_sum.detach().cpu()),
            "lambda": self.lambda_value,
            "batch_size": self.batch_size,
            "per_layer_mu": [float(t.detach().cpu()) for t in self.per_layer_mu],
            "per_layer_sigma": [float(t.detach().cpu()) for t in self.per_layer_sigma],
        }


# ---------------------------------------------------------------------------
# Main fitness
# ---------------------------------------------------------------------------


class FitnessFunction:
    """Eqn. (5): entropy + lambda-weighted activation discrepancy.

    The object is *stateless* with respect to the test stream when
    ``beta == 1.0`` (the paper's configuration).  When ``beta < 1.0`` (the
    Table 15 EMA variant) an internal running estimate of the test statistics is
    maintained and must be reset between streams via :meth:`reset`.

    Usage::

        fit = FitnessFunction(source_stats, FitnessConfig(dataset="imagenet-c"))
        terms = fit(logits, cls_features)     # -> FitnessTerms
        score = terms.total                   # lower is better for CMA-ES
    """

    def __init__(
        self,
        source_stats: SourceStats,
        config: Optional[FitnessConfig] = None,
        **kwargs: Any,
    ):
        self.source_stats = source_stats
        if config is None:
            config = FitnessConfig(**kwargs)
        elif kwargs:
            merged = config.to_dict()
            merged.update(kwargs)
            config = FitnessConfig(**merged)
        self.config = config

        # Optional EMA state for the Table 15 (beta < 1) variant.  Layers are
        # indexed 0..N; the running statistics are stored per layer.
        self._ema_mu: Optional[List[torch.Tensor]] = None
        self._ema_sigma: Optional[List[torch.Tensor]] = None
        self._num_steps: int = 0

    # -- properties ---------------------------------------------------------

    @property
    def beta(self) -> float:
        return float(self.config.beta)

    def reset(self) -> None:
        """Clear the EMA state (call between test streams / datasets)."""
        self._ema_mu = None
        self._ema_sigma = None
        self._num_steps = 0

    # -- statistics used by the discrepancy term ---------------------------

    def _test_statistics(
        self,
        cls_features: Sequence[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Statistics entering Eqn. (5) for the current batch.

        With ``beta = 1.0`` (paper default) these are exactly the batch
        statistics ``mu_i(X_t), sigma_i(X_t)``.  With ``beta < 1`` the running
        EMA ``beta * batch + (1 - beta) * previous`` of Table 15 is used (the
        first batch initialises the EMA to the batch statistics).
        """

        mu_batch, sigma_batch = batch_statistics(
            cls_features, unbiased=bool(self.config.unbiased)
        )

        if self.beta >= 1.0:
            return mu_batch, sigma_batch

        if self._ema_mu is None:
            self._ema_mu = [m.detach().clone() for m in mu_batch]
            self._ema_sigma = [s.detach().clone() for s in sigma_batch]
        else:
            n = min(len(self._ema_mu), len(mu_batch))
            for i in range(n):
                self._ema_mu[i] = (
                    self.beta * mu_batch[i] + (1.0 - self.beta) * self._ema_mu[i]
                )
                self._ema_sigma[i] = (
                    self.beta * sigma_batch[i] + (1.0 - self.beta) * self._ema_sigma[i]
                )
        self._num_steps += 1
        return list(self._ema_mu), list(self._ema_sigma)

    # -- main entry points --------------------------------------------------

    def evaluate(
        self,
        logits: torch.Tensor,
        cls_features: Sequence[torch.Tensor],
    ) -> FitnessTerms:
        """Compute Eqn. (5) term by term for one candidate prompt.

        Parameters
        ----------
        logits:
            ``[B, C]`` predictions *after* activation shifting.
        cls_features:
            All-layer CLS features ``{e_n^0}_{n=1..N}`` (plus layer 0) produced
            by ``ViTWithCLSFeatures.forward_features`` with the candidate prompt
            injected.  These are the *un-shifted* features: Eqn. (5) regularises
            the activation statistics of the test batch against the source
            statistics, independently of the shifting of Eqn. (7).
        """

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        batch_size = int(logits.shape[0])
        lam = self.config.resolved_lambda(batch_size)
        device = logits.device

        # ---- left term: prediction entropy (Eqn. 5, term 1) ----
        if self.config.use_entropy:
            ent = entropy_term(
                logits,
                reduction=self.config.effective_entropy_reduction(),
                eps=1e-12,
            )
            if self.config.effective_entropy_reduction() == "mean" and self.config.entropy_reduction == "sum":
                # Table-9 SGD variant: entropy/BS instead of sum, keeping the
                # same overall scale as the sum form used by CMA.
                ent = ent * float(batch_size)
        else:
            ent = torch.zeros((), dtype=torch.float64, device=device)

        # ---- right term: activation discrepancy (Eqn. 5, term 2) ----
        if self.config.use_discrepancy:
            mu_test, sigma_test = self._test_statistics(cls_features)
            terms = discrepancy_terms(
                cls_features,
                self.source_stats,
                layer_start=int(self.config.layer_start),
                layer_end=self.config.layer_end,
                unbiased=bool(self.config.unbiased),
                mu_test=mu_test,
                sigma_test=sigma_test,
            )
            disc = terms["total"]
            per_layer_mu = terms["mu"]
            per_layer_sigma = terms["sigma"]
            mu_sum = terms["mu_sum"]
            sigma_sum = terms["sigma_sum"]
        else:
            disc = torch.zeros((), dtype=torch.float64, device=device)
            per_layer_mu, per_layer_sigma = [], []
            mu_sum = torch.zeros((), dtype=torch.float64, device=device)
            sigma_sum = torch.zeros((), dtype=torch.float64, device=device)

        if self.config.discrepancy_reduction == "mean" and batch_size > 0:
            disc = disc / float(batch_size)

        weighted_disc = lam * disc
        total = ent + weighted_disc

        return FitnessTerms(
            total=total,
            entropy=torch.as_tensor(ent),
            discrepancy=torch.as_tensor(weighted_disc),
            mu_sum=torch.as_tensor(mu_sum),
            sigma_sum=torch.as_tensor(sigma_sum),
            lambda_value=float(lam),
            per_layer_mu=list(per_layer_mu),
            per_layer_sigma=list(per_layer_sigma),
            batch_size=batch_size,
        )

    # ``fit(logits, cls_features)`` -> scalar fitness (lower is better).
    def __call__(
        self,
        logits: torch.Tensor,
        cls_features: Sequence[torch.Tensor],
        return_terms: bool = False,
    ):
        terms = self.evaluate(logits, cls_features)
        if return_terms:
            return terms
        return terms.total

    def scalar(
        self,
        logits: torch.Tensor,
        cls_features: Sequence[torch.Tensor],
    ) -> float:
        """Convenience: plain Python float fitness for CMA-ES ``tell``."""
        return float(self.evaluate(logits, cls_features).total.detach().cpu())

    # -- helpers used by the runners ---------------------------------------

    def score_batch(
        self,
        logits: torch.Tensor,
        cls_features: Sequence[torch.Tensor],
    ) -> Dict[str, Any]:
        return self.evaluate(logits, cls_features).as_dict()

    def extra_repr(self) -> str:
        return (
            f"lambda_base={self.config.lambda_base}, beta={self.config.beta}, "
            f"entropy={self.config.use_entropy}, discrepancy={self.config.use_discrepancy}, "
            f"layers={self.config.layer_start}..{self.config.layer_end}"
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"{self.__class__.__name__}({self.extra_repr()})"


# ---------------------------------------------------------------------------
# Factories (config-driven, also cover the Table 5 ablations)
# ---------------------------------------------------------------------------


def build_fitness(
    source_stats: SourceStats,
    cfg: Optional[Dict[str, Any]] = None,
    dataset: Optional[str] = None,
    **overrides: Any,
) -> FitnessFunction:
    """Build a :class:`FitnessFunction` from a config mapping.

    Recognised config keys (all optional): ``lambda`` / ``lambda_base`` /
    ``lambda_value``, ``beta``, ``use_entropy``, ``use_discrepancy``,
    ``entropy_reduction``, ``discrepancy_reduction``, ``layer_start``,
    ``layer_end``, ``unbiased``, ``sgd_variant``.
    """

    cfg = dict(cfg or {})
    # Accept both ``fitness:`` sub-dicts and flat dicts.
    if "fitness" in cfg and isinstance(cfg["fitness"], dict):
        cfg = dict(cfg["fitness"])

    known = {f for f in FitnessConfig.__dataclass_fields__.keys()}
    kwargs: Dict[str, Any] = {}
    for k, v in cfg.items():
        if k in known and v is not None:
            kwargs[k] = v
    # ``lambda`` is the paper's symbol for the trade-off parameter.
    if "lambda" in cfg and cfg["lambda"] is not None and "lambda_base" not in kwargs:
        kwargs["lambda_base"] = cfg["lambda"]
    if dataset is not None:
        kwargs.setdefault("dataset", dataset)
    kwargs.update({k: v for k, v in overrides.items() if v is not None})
    return FitnessFunction(source_stats, FitnessConfig(**kwargs))


def ablation_config(variant: str, **kwargs: Any) -> FitnessConfig:
    """Return the :class:`FitnessConfig` of a Table 5 / Table 9 variant.

    ``variant`` is one of

    * ``"no_adapt"``                 -- no fitness at all (NoAdapt baseline)
    * ``"entropy"``                  -- CMA + entropy only (Table 5 row 2)
    * ``"discrepancy"``              -- CMA + activation discrepancy only (row 3)
    * ``"full"``                     -- entropy + discrepancy (row 6)
    * ``"sgd_eqn5"``                 -- Table 9: SGD + Eqn. (5), lambda = 30
    * ``"sgd_entropy"``              -- Table 9: SGD + entropy (lr 0.01)
    """

    key = str(variant).lower()
    if key in ("no_adapt", "noadapt", "none"):
        return FitnessConfig(use_entropy=False, use_discrepancy=False, **kwargs)
    if key in ("entropy", "entropy_only", "ent"):
        return FitnessConfig(use_entropy=True, use_discrepancy=False, **kwargs)
    if key in ("discrepancy", "act_discrepancy", "disc", "disc_only"):
        return FitnessConfig(use_entropy=False, use_discrepancy=True, **kwargs)
    if key in ("full", "entropy+discrepancy", "eqn5", "eqn_5"):
        return FitnessConfig(use_entropy=True, use_discrepancy=True, **kwargs)
    if key in ("sgd_eqn5", "sgd_eqn_5", "sgd"):
        return FitnessConfig(
            use_entropy=True,
            use_discrepancy=True,
            sgd_variant=True,
            lambda_value=30.0,
            **kwargs,
        )
    if key in ("sgd_entropy", "sgd_ent"):
        return FitnessConfig(
            use_entropy=True, use_discrepancy=False, sgd_variant=True, **kwargs
        )
    raise ValueError(f"Unknown fitness ablation variant '{variant}'")
