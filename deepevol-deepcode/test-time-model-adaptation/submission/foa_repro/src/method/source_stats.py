"""Source in-distribution activation statistics bank for FOA.

Paper reference
---------------
Section 3.1 ("Statistics calculation"):

    Before TTA, we first collect a small set of source in-distribution samples
    D_S = {x_q}_{q=1}^Q and feed them into the model to obtain the corresponding
    CLS tokens {e_i^0}_{i=1}^N.  Then, we calculate the mean and standard
    deviations of CLS tokens {e_i^0}_{i=1}^N over all samples in D_S to obtain
    source in-distribution statistics {mu_i^S, sigma_i^S}_{i=0}^N.  Note that we
    only need a small number of in-distribution samples without labels for
    calculation, e.g., 32 samples are sufficient for the ImageNet dataset.

Appendix B.2:

    ... The source in-distribution statistics {mu_i^S, sigma_i^S}_{i=0}^N are
    calculated without using the newly inserted prompt.

Appendix C / Figure 2 (c): the number of source samples Q is swept over
{16, 32, 64, 100, 200, 400, 800, 1600}; FOA is stable for Q > 32.

Implementation notes / documented defaults
------------------------------------------
*  All N+1 layer statistics are stored (i = 0..N), because (a) Equation (5)
   sums over i = 1..N and (b) the final-layer mean ``mu_N^S`` is reused by the
   back-to-source activation shifting of Equation (7).  Layer 0 is the patch
   embedding stage *after* the CLS token has been combined with its positional
   embedding, matching ``ViTWithCLSFeatures.forward_features`` (index 0 of the
   returned ``cls_features`` list).
*  The paper does not state whether the population (biased) or the sample
   (unbiased) standard deviation is used.  We default to the population
   standard deviation (``unbiased=False``), the usual convention in feature
   statistics matching, and record the choice in the checkpoint metadata.
*  Statistics are accumulated in float64 to keep the running sums accurate for
   large Q, then cast back to the requested dtype.
*  No gradient is ever enabled here: the frozen backbone is evaluated in
   ``torch.no_grad()`` / ``eval()`` mode.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch

try:  # importable both as a package (`src.method...`) and script-style
    from ..models.vit_loader import ViTWithCLSFeatures
except Exception:  # pragma: no cover - defensive import for script-style usage
    ViTWithCLSFeatures = Any  # type: ignore


__all__ = [
    "SourceStats",
    "SourceStatsAccumulator",
    "compute_source_statistics",
    "compute_source_statistics_from_loader",
    "batch_statistics",
    "activation_discrepancy",
    "save_source_stats",
    "load_source_stats",
    "DEFAULT_NUM_SOURCE_SAMPLES",
]


#: Q = 32 unlabeled ID images are sufficient (Section 3.1 / Appendix B.2).
DEFAULT_NUM_SOURCE_SAMPLES: int = 32


# ---------------------------------------------------------------------------
# container
# ---------------------------------------------------------------------------
@dataclass
class SourceStats:
    """Per-layer source (in-distribution) CLS statistics ``{mu_i^S, sigma_i^S}``.

    Attributes
    ----------
    mu, sigma:
        Lists of length ``N + 1``; entry ``i`` is a 1-D tensor of shape ``[d]``
        holding the source mean / standard deviation of the CLS token at layer
        ``i`` (``i = 0`` is the embedding stage, ``i = N`` the final block).
    num_samples:
        Number ``Q`` of source images used to compute the statistics.
    model_name, dataset, seed:
        Provenance metadata written to the checkpoint.
    unbiased:
        Whether the sample (unbiased) standard deviation was used.
    """

    mu: List[torch.Tensor]
    sigma: List[torch.Tensor]
    num_samples: int
    model_name: str = ""
    dataset: str = ""
    seed: int = 0
    unbiased: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)

    # -- basic properties ---------------------------------------------------
    @property
    def num_layers(self) -> int:
        """Number of transformer blocks ``N`` (statistics stored for 0..N)."""
        return len(self.mu) - 1 if len(self.mu) > 0 else 0

    @property
    def dim(self) -> int:
        """Feature width ``d`` of the CLS tokens."""
        return int(self.mu[0].numel()) if self.mu else 0

    @property
    def mu_final(self) -> torch.Tensor:
        """``mu_N^S`` -- final-layer source mean reused by Eqn. (7)."""
        return self.mu[-1]

    @property
    def sigma_final(self) -> torch.Tensor:
        """``sigma_N^S`` -- final-layer source standard deviation."""
        return self.sigma[-1]

    def mu_i(self, i: int) -> torch.Tensor:
        return self.mu[i]

    def sigma_i(self, i: int) -> torch.Tensor:
        return self.sigma[i]

    # -- device / dtype handling -------------------------------------------
    def to(
        self, device: Union[str, torch.device], dtype: Optional[torch.dtype] = None
    ) -> "SourceStats":
        """Move all statistics to ``device`` (optionally casting the dtype)."""
        self.mu = [
            t.to(device=device, dtype=dtype if dtype is not None else t.dtype)
            for t in self.mu
        ]
        self.sigma = [
            t.to(device=device, dtype=dtype if dtype is not None else t.dtype)
            for t in self.sigma
        ]
        return self

    # -- serialisation ------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {
            "mu": [t.detach().cpu() for t in self.mu],
            "sigma": [t.detach().cpu() for t in self.sigma],
            "num_samples": int(self.num_samples),
            "num_layers": int(self.num_layers),
            "dim": int(self.dim),
            "model_name": self.model_name,
            "dataset": self.dataset,
            "seed": int(self.seed),
            "unbiased": bool(self.unbiased),
            "meta": dict(self.meta),
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "SourceStats":
        return cls(
            mu=[torch.as_tensor(t).clone() for t in state["mu"]],
            sigma=[torch.as_tensor(t).clone() for t in state["sigma"]],
            num_samples=int(state.get("num_samples", 0)),
            model_name=state.get("model_name", ""),
            dataset=state.get("dataset", ""),
            seed=int(state.get("seed", 0)),
            unbiased=bool(state.get("unbiased", False)),
            meta=dict(state.get("meta", {})),
        )


# ---------------------------------------------------------------------------
# streaming accumulator
# ---------------------------------------------------------------------------
class SourceStatsAccumulator:
    """Accumulates per-layer CLS statistics in a streaming (O(d)) fashion.

    Uses float64 running sums of ``x`` and ``x^2`` so the result stays
    numerically stable even for Q = 1600 samples, without retaining features.
    """

    def __init__(self, num_layers: int, dim: int, unbiased: bool = False) -> None:
        self.num_layers = int(num_layers)
        self.dim = int(dim)
        self.unbiased = bool(unbiased)
        self.count = 0
        self._sum = [
            torch.zeros(self.dim, dtype=torch.float64) for _ in range(self.num_layers + 1)
        ]
        self._sumsq = [
            torch.zeros(self.dim, dtype=torch.float64) for _ in range(self.num_layers + 1)
        ]

    def update(self, cls_features: Sequence[torch.Tensor]) -> None:
        """Fold one batch of all-layer CLS features into the accumulator.

        ``cls_features`` is the list returned by
        ``ViTWithCLSFeatures.forward_features`` (length ``N + 1``); each entry
        has shape ``[B, d]``.
        """
        if len(cls_features) != self.num_layers + 1:
            raise ValueError(
                f"expected {self.num_layers + 1} layer feature maps, "
                f"got {len(cls_features)}"
            )
        with torch.no_grad():
            for i, feat in enumerate(cls_features):
                x = feat.detach().to(device="cpu", dtype=torch.float64)
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                self._sum[i] += x.sum(dim=0)
                self._sumsq[i] += (x * x).sum(dim=0)
            self.count += int(cls_features[0].shape[0])

    def update_single(self, cls_features: Sequence[torch.Tensor]) -> None:
        """Fold a single sample (``[d]``-shaped features) into the accumulator."""
        with torch.no_grad():
            self.update([f.unsqueeze(0) if f.dim() == 1 else f for f in cls_features])

    # -- result -------------------------------------------------------------
    def compute(
        self,
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = "cpu",
    ) -> SourceStats:
        if self.count == 0:
            raise RuntimeError("no samples were accumulated")
        n = float(self.count)
        mu: List[torch.Tensor] = []
        sigma: List[torch.Tensor] = []
        for i in range(self.num_layers + 1):
            mean = self._sum[i] / n
            var = self._sumsq[i] / n - mean * mean
            var = torch.clamp(var, min=0.0)
            if self.unbiased and self.count > 1:
                var = var * (n / (n - 1.0))
            mu.append(mean.to(device=device, dtype=dtype))
            sigma.append(torch.sqrt(var).to(device=device, dtype=dtype))
        return SourceStats(
            mu=mu,
            sigma=sigma,
            num_samples=int(self.count),
            unbiased=self.unbiased,
        )


# ---------------------------------------------------------------------------
# computation from a frozen backbone
# ---------------------------------------------------------------------------
def _as_batch(images: torch.Tensor) -> torch.Tensor:
    if images.dim() == 3:  # single CHW image
        images = images.unsqueeze(0)
    return images


def compute_source_statistics(
    model: "ViTWithCLSFeatures",
    image_iter: Iterable[Union[torch.Tensor, Tuple[torch.Tensor, Any]]],
    num_samples: int = DEFAULT_NUM_SOURCE_SAMPLES,
    device: Union[str, torch.device] = "cuda",
    unbiased: bool = False,
    dtype: torch.dtype = torch.float32,
    max_batches: Optional[int] = None,
    progress: bool = False,
) -> SourceStats:
    """Compute ``{mu_i^S, sigma_i^S}_{i=0}^N`` from ``num_samples`` ID images.

    Parameters
    ----------
    model:
        Frozen ViT with CLS hooks.  It is used **without** the inserted prompt,
        exactly as required by Appendix B.2.
    image_iter:
        Iterable yielding either ``images`` (``[B, 3, H, W]``) or
        ``(images, labels)`` tuples.  Labels are ignored (unlabeled samples).
    num_samples:
        ``Q`` -- number of source images to consume (32 by default).
    device, dtype:
        Where/how to evaluate and store the statistics.
    max_batches:
        Optional hard cap on the number of batches consumed (useful in tests).
    progress:
        Print progress with ``tqdm`` when available.

    Returns
    -------
    SourceStats
    """
    device = device if isinstance(device, torch.device) else torch.device(device)
    model = model.to(device)
    model.eval()
    num_layers = int(getattr(model, "num_layers"))
    dim = int(getattr(model, "embed_dim"))
    acc = SourceStatsAccumulator(num_layers=num_layers, dim=dim, unbiased=unbiased)

    it: Iterable = image_iter
    if progress:
        try:
            from tqdm import tqdm  # local import: optional dependency

            it = tqdm(image_iter, desc="source stats")
        except Exception:  # pragma: no cover
            it = image_iter

    n_consumed, n_batches = 0, 0
    with torch.no_grad():
        for batch in it:
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            images = _as_batch(images).to(device)
            if images.numel() == 0:
                continue
            # NOTE: no prompt is injected here (Appendix B.2)
            cls_features, _logits, _final = model.forward_features(images, prompt=None)
            remaining = num_samples - n_consumed
            if images.shape[0] > remaining:
                cls_features = [f[:remaining] for f in cls_features]
                images = images[:remaining]
            acc.update(cls_features)
            n_consumed += int(images.shape[0])
            n_batches += 1
            if n_consumed >= num_samples:
                break
            if max_batches is not None and n_batches >= max_batches:
                break

    stats = acc.compute(dtype=dtype, device=device)
    stats.model_name = str(getattr(model, "model_name", "") or "")
    stats.meta.update({"num_batches": n_batches, "layers": list(range(num_layers + 1))})
    return stats


def compute_source_statistics_from_loader(
    model: "ViTWithCLSFeatures",
    dataloader: Iterable,
    num_samples: int = DEFAULT_NUM_SOURCE_SAMPLES,
    device: Union[str, torch.device] = "cuda",
    dataset: str = "imagenet-1k",
    seed: int = 0,
    unbiased: bool = False,
    dtype: torch.dtype = torch.float32,
    progress: bool = False,
) -> SourceStats:
    """Convenience wrapper adding dataset/seed provenance metadata."""
    stats = compute_source_statistics(
        model,
        dataloader,
        num_samples=num_samples,
        device=device,
        unbiased=unbiased,
        dtype=dtype,
        progress=progress,
    )
    stats.dataset = dataset
    stats.seed = int(seed)
    return stats


# ---------------------------------------------------------------------------
# test-batch statistics (inputs of Equation (5) / Equation (7))
# ---------------------------------------------------------------------------
def batch_statistics(
    cls_features: Sequence[torch.Tensor],
    unbiased: bool = False,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Per-layer mean/std of the current test batch ``X_t``.

    Implements the second half of Section 3.1 ("Similarly, we calculate the
    target testing statistics {mu_i(X_t), sigma_i(X_t)}_{i=0}^N over the current
    batch of testing samples X_t.").

    Returns lists ``(mu, sigma)`` of length ``N + 1``; entry ``i`` has shape
    ``[d]``.  Batch statistics (i.e. ``beta = 1.0`` in Appendix C Table 15) are
    used, matching the main-paper configuration.
    """
    mu: List[torch.Tensor] = []
    sigma: List[torch.Tensor] = []
    for feat in cls_features:
        x = feat.detach()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.float()
        mu.append(x.mean(dim=0))
        if x.shape[0] > 1:
            sigma.append(x.std(dim=0, unbiased=unbiased))
        else:
            sigma.append(torch.zeros_like(mu[-1]))
    return mu, sigma


def activation_discrepancy(
    mu_test: Sequence[torch.Tensor],
    sigma_test: Sequence[torch.Tensor],
    source_stats: SourceStats,
    layer_range: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Unweighted per-layer activation discrepancy of Equation (5).

    ``sum_{i in layer_range} [ ||mu_i(X_t) - mu_i^S||_2
                              + ||sigma_i(X_t) - sigma_i^S||_2 ]``

    The trade-off parameter ``lambda`` is applied by the fitness module, not
    here.  ``layer_range`` defaults to ``(1, N)``, i.e. the paper sums the
    discrepancy over the transformer layers but not the layer-0 embedding.
    """
    n = source_stats.num_layers
    lo, hi = (1, n) if layer_range is None else (int(layer_range[0]), int(layer_range[1]))
    total: Optional[torch.Tensor] = None
    for i in range(max(lo, 0), min(hi, n) + 1):
        device = mu_test[i].device
        mu_s = source_stats.mu[i].to(device=device, dtype=mu_test[i].dtype)
        sg_s = source_stats.sigma[i].to(device=device, dtype=sigma_test[i].dtype)
        term = (mu_test[i] - mu_s).norm(p=2) + (sigma_test[i] - sg_s).norm(p=2)
        total = term if total is None else total + term
    if total is None:
        raise ValueError("empty layer range for the activation discrepancy")
    return total


# ---------------------------------------------------------------------------
# checkpoint I/O
# ---------------------------------------------------------------------------
def save_source_stats(stats: SourceStats, path: str) -> str:
    """Persist the statistics bank (default: ``./checkpoints/source_stats_*.pt``)."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(stats.state_dict(), path)
    return path


def load_source_stats(
    path: str,
    device: Union[str, torch.device] = "cpu",
    dtype: Optional[torch.dtype] = None,
    map_location: Optional[str] = "cpu",
) -> SourceStats:
    """Load a previously computed statistics bank and place it on ``device``."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"source statistics not found at '{path}'. Run "
            f"`python scripts/compute_source_stats.py` first (or point "
            f"--source-stats at an existing checkpoint)."
        )
    state = torch.load(path, map_location=map_location or "cpu")
    stats = state if isinstance(state, SourceStats) else SourceStats.from_state_dict(state)
    return stats.to(device=device, dtype=dtype)


if __name__ == "__main__":  # tiny self-test on random features
    feats = [torch.randn(8, 16) + 3.0 for _ in range(4)]
    acc = SourceStatsAccumulator(num_layers=3, dim=16)
    acc.update(feats)
    st = acc.compute()
    print("num_layers", st.num_layers, "dim", st.dim, "Q", st.num_samples)
    print("mu_0[:3]", st.mu[0][:3].tolist())
    m, s = batch_statistics(feats)
    print("discrepancy", float(activation_discrepancy(m, s, st)))
