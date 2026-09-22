"""Utility helpers shared across SNPSE/NPSE/TSNPSE modules.

This module is intentionally dependency-light: it only relies on ``torch`` and
``numpy`` (imported lazily) so that every other module can import it safely.

Contents
--------
* Reproducibility helpers (:func:`set_seed`, :func:`seed_worker`).
* Standardisation transforms (:class:`Standardiser`) used to whiten the
  parameters ``theta`` and the observations ``x`` before score matching, and to
  un-standardise posterior samples afterwards.  As noted in the reproduction
  plan, :math:`\\theta_t` and :math:`x` are standardised in the NPSE/TSNPSE
  training pipeline.
* Batch sampling helpers (:func:`sample_batch_indices`, :func:`make_batches`,
  :func:`random_choice`).
* Small tensor helpers used by the samplers/HPR code (``flatten_inputs``,
  ``ensure_2d``, ``as_column``).
* Optional device selection (:func:`get_device`) and a light progress bar.

Nothing in this file is paper-specific; the paper is silent on these details so
sensible defaults are used (see ``configs/default.yaml``).
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import torch

__all__ = [
    "set_seed",
    "seed_worker",
    "get_device",
    "Standardiser",
    "running_standardiser",
    "fit_standardiser",
    "sample_batch_indices",
    "make_batches",
    "random_choice",
    "flatten_inputs",
    "ensure_2d",
    "as_column",
    "to_tensor",
    "normalise_weights",
    "safe_log",
    "quantile",
    "expanding_mean_std",
    "ProgressBar",
    "Timer",
    "infinite_batches",
    "count_parameters",
]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: Optional[int], deterministic: bool = False) -> Optional[torch.Generator]:
    """Seed python/numpy/torch RNGs and return a torch generator.

    Parameters
    ----------
    seed:
        Seed to use.  ``None`` means "do nothing" (and ``None`` is returned).
    deterministic:
        If ``True``, also request deterministic cudnn behaviour.

    Returns
    -------
    Optional[torch.Generator]
        A generator seeded with ``seed`` (or ``None`` when ``seed is None``).
    """
    if seed is None:
        return None

    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # numpy is optional
        import numpy as np  # noqa: WPS433 (local import on purpose)

        np.random.seed(seed)
    except Exception:  # pragma: no cover - numpy always present in practice
        pass

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # pragma: no cover - older torch
            pass
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:  # pragma: no cover
            pass

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:  # pragma: no cover - DataLoader hook
    """DataLoader ``worker_init_fn`` that keeps workers reproducible."""
    worker_seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    random.seed(worker_seed)
    try:
        import numpy as np

        np.random.seed(worker_seed)
    except Exception:
        pass


def get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Return a torch device, defaulting to CUDA when available."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Standardisation
# ---------------------------------------------------------------------------
@dataclass
class Standardiser:
    """Affine standardisation transform.

    ``theta_std = (theta - shift) / scale`` with ``scale`` a per-dimension
    standard deviation (``std``) or range-based scale.  The inverse transform is
    ``theta = theta_std * scale + shift``.  ``log_abs_det`` returns the log
    absolute determinant of the forward Jacobian, which is a constant
    ``-sum(log(scale))`` and is needed when converting densities back to the
    original parameter space.
    """

    shift: torch.Tensor
    scale: torch.Tensor
    eps: float = 1e-6

    # -- construction -------------------------------------------------
    @classmethod
    def from_data(cls, data: torch.Tensor, eps: float = 1e-6, mode: str = "std") -> "Standardiser":
        data = ensure_2d(data)
        if mode == "std":
            shift = data.mean(dim=0)
            scale = data.std(dim=0, unbiased=False)
        elif mode == "range":
            shift = data.min(dim=0).values
            scale = (data.max(dim=0).values - data.min(dim=0).values) / 2.0
        elif mode == "minmax":
            lo = data.min(dim=0).values
            hi = data.max(dim=0).values
            shift = lo
            scale = (hi - lo)
        else:
            raise ValueError(f"unknown standardisation mode: {mode!r}")
        scale = torch.clamp(scale, min=eps)
        return cls(shift=shift.detach().clone(), scale=scale.detach().clone(), eps=eps)

    @classmethod
    def identity(cls, dim: int, dtype: torch.dtype = torch.float32) -> "Standardiser":
        return cls(
            shift=torch.zeros(dim, dtype=dtype),
            scale=torch.ones(dim, dtype=dtype),
        )

    # -- transforms ---------------------------------------------------
    def to_std(self, value: torch.Tensor) -> torch.Tensor:
        shift, scale = self._broadcast(value)
        return (value - shift) / scale

    def from_std(self, value: torch.Tensor) -> torch.Tensor:
        shift, scale = self._broadcast(value)
        return value * scale + shift

    #: aliases mirroring the sampler's ``x_shift``/``x_scale`` interface
    def standardise(self, value: torch.Tensor) -> torch.Tensor:
        return self.to_std(value)

    def unstandardise(self, value: torch.Tensor) -> torch.Tensor:
        return self.from_std(value)

    def to(self, device: Union[str, torch.device], dtype: Optional[torch.dtype] = None) -> "Standardiser":
        self.shift = self.shift.to(device=device, dtype=dtype)
        self.scale = self.scale.to(device=device, dtype=dtype)
        return self

    # -- properties ---------------------------------------------------
    @property
    def dim(self) -> int:
        return int(self.shift.numel())

    @property
    def log_abs_det(self) -> torch.Tensor:
        """log|det(d theta_std / d theta)| (a scalar)."""
        return -torch.log(self.scale).sum()

    def log_det(self) -> torch.Tensor:
        return self.log_abs_det

    def log_det_to_std(self) -> torch.Tensor:
        """Alias returning the log determinant in the *forward* direction."""
        return self.log_abs_det

    def as_tuple(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.shift, self.scale

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {"shift": self.shift.clone(), "scale": self.scale.clone()}

    def copy(self) -> "Standardiser":
        return Standardiser(self.shift.clone(), self.scale.clone(), eps=self.eps)

    def _broadcast(self, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shift, scale = self.shift, self.scale
        if shift.device != value.device or shift.dtype != value.dtype:
            shift = shift.to(device=value.device, dtype=value.dtype)
            scale = scale.to(device=value.device, dtype=value.dtype)
        return shift, scale


def fit_standardiser(data: torch.Tensor, **kwargs: Any) -> Standardiser:
    """Convenience wrapper around :meth:`Standardiser.from_data`."""
    return Standardiser.from_data(data, **kwargs)


def running_standardiser(
    data: torch.Tensor,
    previous: Optional[Standardiser] = None,
    weight: float = 1.0,
    mode: str = "std",
) -> Standardiser:
    """Incrementally merged standardiser (useful across sequential rounds).

    ``weight`` is the relative weight of ``data`` compared with ``previous``
    (e.g. the ratio of newly drawn samples to previously seen samples).
    """
    current = Standardiser.from_data(data, mode=mode)
    if previous is None:
        return current
    total = weight + 1.0
    shift = (current.shift * weight + previous.shift) / total
    scale = (current.scale * weight + previous.scale) / total
    return Standardiser(shift=shift, scale=torch.clamp(scale, min=1e-6))


# ---------------------------------------------------------------------------
# Batching helpers
# ---------------------------------------------------------------------------
def sample_batch_indices(
    n: int,
    batch_size: int,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Uniformly sample ``batch_size`` indices in ``[0, n)`` with replacement."""
    if n <= 0:
        raise ValueError("sample_batch_indices requires n > 0")
    idx = torch.randint(0, n, (int(batch_size),), generator=generator)
    if device is not None:
        idx = idx.to(device)
    return idx


def make_batches(
    n: int,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False,
    generator: Optional[torch.Generator] = None,
) -> List[torch.Tensor]:
    """Split ``[0, n)`` into a list of index tensors."""
    if shuffle:
        perm = torch.randperm(n, generator=generator)
    else:
        perm = torch.arange(n)
    batches = [perm[i : i + batch_size] for i in range(0, n, batch_size)]
    if drop_last and batches and batches[-1].numel() < batch_size:
        batches = batches[:-1]
    return batches


def infinite_batches(
    n: int,
    batch_size: int,
    generator: Optional[torch.Generator] = None,
    shuffle: bool = True,
    drop_last: bool = False,
) -> Iterator[torch.Tensor]:
    """Endlessly yield index batches (used by the stochastic training loop)."""
    while True:
        for batch in make_batches(
            n, batch_size, shuffle=shuffle, drop_last=drop_last, generator=generator
        ):
            yield batch


def random_choice(
    n: int,
    size: int,
    generator: Optional[torch.Generator] = None,
    replace: bool = True,
) -> torch.Tensor:
    """Draw ``size`` indices from ``range(n)``."""
    size = int(size)
    if replace:
        return torch.randint(0, n, (size,), generator=generator)
    if size > n:
        raise ValueError("cannot sample without replacement more indices than available")
    return torch.randperm(n, generator=generator)[:size]


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------
def to_tensor(value: Any, dtype: Optional[torch.dtype] = None, device: Any = None) -> torch.Tensor:
    """Convert numpy/list/scalar input into a torch tensor."""
    if isinstance(value, torch.Tensor):
        tensor = value
    else:
        tensor = torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor


def ensure_2d(value: torch.Tensor, name: str = "value") -> torch.Tensor:
    """Return ``value`` as a 2-D ``(n, d)`` tensor (1-D is treated as one row)."""
    if value.dim() == 1:
        return value.unsqueeze(0)
    if value.dim() == 2:
        return value
    raise ValueError(f"{name} must be 1-D or 2-D, got shape {tuple(value.shape)}")


def flatten_inputs(value: torch.Tensor, name: str = "value") -> torch.Tensor:
    """Flatten all leading dimensions into a single batch dimension."""
    if value.dim() <= 2:
        return value
    return value.reshape(-1, value.shape[-1])


def as_column(value: torch.Tensor) -> torch.Tensor:
    """Ensure a ``(n, 1)`` shape."""
    if value.dim() == 1:
        return value.unsqueeze(-1)
    if value.dim() == 2 and value.shape[-1] == 1:
        return value
    return value.reshape(-1, 1)


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------
def normalise_weights(
    weights: torch.Tensor,
    clip: Optional[float] = None,
    eps: float = 1e-12,
    detach: bool = False,
) -> torch.Tensor:
    """Normalise weights to have mean 1 (optionally clipping the max ratio)."""
    if detach:
        weights = weights.detach()
    weights = weights.clamp_min(eps)
    if clip is not None:
        weights = weights.clamp(max=float(clip))
    mean = weights.mean()
    if torch.isfinite(mean) and mean > 0:
        weights = weights / mean
    return weights


def safe_log(value: torch.Tensor, eps: float = 1e-30) -> torch.Tensor:
    """Numerically safe natural logarithm."""
    return torch.log(value.clamp_min(eps))


def quantile(value: torch.Tensor, q: float) -> torch.Tensor:
    """Scalar quantile of a flattened tensor (``q`` in ``[0, 1]``)."""
    flat = value.reshape(-1)
    if flat.numel() == 0:
        raise ValueError("quantile of an empty tensor is undefined")
    return torch.quantile(flat, float(q))


def expanding_mean_std(values: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute mean/std over a sequence of (already computed) element tensors."""
    if len(values) == 0:
        raise ValueError("expanding_mean_std requires at least one tensor")
    stacked = torch.stack([v.reshape(-1) for v in values], dim=0)
    return stacked.mean(), stacked.std(unbiased=False)


# ---------------------------------------------------------------------------
# Tiny progress bar / timer (keeps the dependency list short)
# ---------------------------------------------------------------------------
class ProgressBar:  # pragma: no cover - cosmetic
    """Minimal progress printer (``tqdm`` is used when available)."""

    def __init__(self, total: Optional[int], desc: str = "", enabled: bool = True):
        self.total = total
        self.desc = desc
        self.enabled = enabled
        self._n = 0
        self._tqdm = None
        if enabled:
            try:
                from tqdm.auto import tqdm  # type: ignore

                self._tqdm = tqdm(total=total, desc=desc)
            except Exception:
                self._tqdm = None

    def update(self, n: int = 1) -> None:
        self._n += n
        if self._tqdm is not None:
            self._tqdm.update(n)
        elif self.enabled and self.total:
            # simple percentage print (kept quiet above 100 calls)
            pass

    def set_postfix(self, **kwargs: Any) -> None:
        if self._tqdm is not None:
            self._tqdm.set_postfix(**kwargs)

    def close(self) -> None:
        if self._tqdm is not None:
            self._tqdm.close()

    def __enter__(self) -> "ProgressBar":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class Timer:  # pragma: no cover - cosmetic
    """Simple wall-clock timer."""

    def __init__(self) -> None:
        import time

        self._time = time
        self._start: Optional[float] = None
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = self._time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._start is not None:
            self.elapsed = self._time.perf_counter() - self._start


def count_parameters(module: torch.nn.Module) -> int:
    """Number of trainable parameters (mirrors ``score_network.count_parameters``)."""
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _selftest() -> None:  # pragma: no cover - manual smoke test
    torch.manual_seed(0)
    data = torch.randn(1000, 3) * torch.tensor([2.0, 0.5, 1.0]) + torch.tensor([1.0, -1.0, 0.0])
    std = Standardiser.from_data(data)
    back = std.from_std(std.to_std(data))
    assert torch.allclose(back, data, atol=1e-5), "roundtrip failed"
    assert abs(float(std.to_std(data).mean())) < 1e-5, "mean not removed"
    assert abs(float(std.to_std(data).std(unbiased=False).mean()) - 1.0) < 1e-4, "std not removed"

    gen = torch.Generator().manual_seed(1)
    idx = sample_batch_indices(10, 5, generator=gen)
    assert idx.shape == (5,) and int(idx.max()) < 10

    batches = make_batches(10, 4)
    assert sum(b.numel() for b in batches) == 10

    w = normalise_weights(torch.tensor([1.0, 2.0, 3.0]))
    assert abs(float(w.mean()) - 1.0) < 1e-6

    print("utils._selftest: OK")


if __name__ == "__main__":  # pragma: no cover
    _selftest()
