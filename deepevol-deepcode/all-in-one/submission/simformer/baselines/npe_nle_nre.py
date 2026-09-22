"""Neural Posterior / Likelihood / Ratio Estimation baselines for Simformer.

Paper reference (Appendix A2.1 -- "Training and model configurations"):

    "For implementing Neural Posterior Estimation (NPE), Neural Ratio Estimation
    (NRE), and Neural Likelihood Estimation (NLE), we utilize the sbi library
    (Tejero-Cantero et al., 2020), adopting default parameters but opting for a
    more expressive neural spline flow for NPE and NLE.  Each method was trained
    using the provided training loop with a batch size of 1000 and an Adam
    optimizer.  Training ceased upon convergence, as indicated by early stopping
    based on validation loss."

So the *primary* implementation is a thin wrapper around ``sbi`` with a neural
spline flow density estimator, batch size 1000, Adam and early stopping.  Since
``sbi`` is an optional dependency of this repository, a self-contained
PyTorch fallback (conditional masked-autoregressive flow for NPE/NLE and an MLP
ratio classifier for NRE, both trained with the same protocol) is provided so
that the benchmark can still be reproduced without the external library.

All three baselines expose a uniform interface::

    baseline.sample(n_samples, x_obs) -> np.ndarray (n_samples, n_parameters)

which allows :mod:`simformer.scripts.run_experiments` (and the evaluation
modules) to treat Simformer and every baseline identically.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# optional dependencies
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised only when torch is installed
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _HAS_TORCH = False

    class _ModuleStub:  # minimal placeholder so that class definitions work
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "PyTorch is required for the NPE/NLE/NRE baselines. Install torch."
            )

    nn = type("nn", (), {"Module": _ModuleStub})  # type: ignore

try:  # pragma: no cover - exercised only when sbi is installed
    import sbi  # noqa: F401
    from sbi.inference import NLE, NPE, NRE  # type: ignore
    from sbi.utils import BoxUniform  # type: ignore

    _HAS_SBI = True
    _SBI_ERROR: Optional[str] = None
except Exception as _exc:  # pragma: no cover
    _HAS_SBI = False
    _SBI_ERROR = repr(_exc)
    NPE = NLE = NRE = None  # type: ignore
    BoxUniform = None  # type: ignore


# ---------------------------------------------------------------------------
# constants (paper Appendix A2.1)
# ---------------------------------------------------------------------------
DEFAULT_BATCH_SIZE = 1000
DEFAULT_LR = 5e-4
DEFAULT_VAL_FRACTION = 0.1
DEFAULT_PATIENCE = 20
DEFAULT_MAX_EPOCHS = 300
DEFAULT_HIDDEN_DIMS = (128, 128)
DEFAULT_N_TRANSFORMS = 5
DEFAULT_DENSITY_ESTIMATOR = "nsf"
DEFAULT_MCMC_STEPS = 2000
DEFAULT_MCMC_STEP_SIZE = 0.1
DEFAULT_N_REFERENCE = 1000
DEFAULT_N_TARGETS = 10
DEFAULT_N_EVAL_SAMPLES = 1000
METHOD_ALIASES = {
    "npe": "npe",
    "snpe": "npe",
    "npe_nsf": "npe",
    "nle": "nle",
    "snle": "nle",
    "nre": "nre",
    "snre": "nre",
    "nre_a": "nre",
    "nre_b": "nre",
}
IMPLEMENTED_METHODS = ("npe", "nle", "nre")

ArrayLike = Union[np.ndarray, Sequence[float]]


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass
class BaselineConfig:
    """Training / inference configuration for the sbi baselines.

    Defaults follow Appendix A2.1 (batch size 1000, Adam, early stopping on the
    validation loss, neural spline flow for NPE and NLE).
    """

    method: str = "npe"
    batch_size: int = DEFAULT_BATCH_SIZE
    lr: float = DEFAULT_LR
    max_epochs: int = DEFAULT_MAX_EPOCHS
    validation_fraction: float = DEFAULT_VAL_FRACTION
    patience: int = DEFAULT_PATIENCE
    density_estimator: str = DEFAULT_DENSITY_ESTIMATOR
    hidden_dims: Tuple[int, ...] = DEFAULT_HIDDEN_DIMS
    n_transforms: int = DEFAULT_N_TRANSFORMS
    standardize: bool = True
    device: str = "cpu"
    seed: int = 0
    n_simulations: int = 10_000
    use_sbi: Optional[bool] = None
    mcmc_steps: int = DEFAULT_MCMC_STEPS
    mcmc_step_size: float = DEFAULT_MCMC_STEP_SIZE
    mcmc_warmup: int = 0
    verbose: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.method = normalize_method(self.method)

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["hidden_dims"] = list(self.hidden_dims)
        return d

    @classmethod
    def from_dict(cls, cfg: Optional[Union[Dict[str, Any], "BaselineConfig"]] = None,
                  **kwargs: Any) -> "BaselineConfig":
        if isinstance(cfg, BaselineConfig):
            base = cfg.to_dict()
        elif isinstance(cfg, dict):
            base = dict(cfg)
        elif cfg is None:
            base = {}
        else:  # pragma: no cover - defensive
            base = {}
        base.update({k: v for k, v in kwargs.items() if v is not None})
        known = {f for f in cls.__dataclass_fields__}
        base = {k: v for k, v in base.items() if k in known}
        if "hidden_dims" in base and base["hidden_dims"] is not None:
            base["hidden_dims"] = tuple(int(h) for h in base["hidden_dims"])
        return cls(**base)


def normalize_method(method: str) -> str:
    """Map method aliases (``snpe`` etc.) to canonical names."""
    key = str(method).strip().lower()
    if key not in METHOD_ALIASES:
        raise ValueError(
            f"unknown baseline method {method!r}; expected one of "
            f"{sorted(METHOD_ALIASES)}"
        )
    return METHOD_ALIASES[key]


def available_methods() -> Tuple[str, ...]:
    """Canonical baseline method names implemented in this module."""
    return IMPLEMENTED_METHODS


def sbi_available() -> bool:
    """Whether the external ``sbi`` library could be imported."""
    return bool(_HAS_SBI)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
class Standardizer:
    """Per-dimension z-scoring (offsets are constants and drop out of ratios)."""

    def __init__(self, mean: Optional[np.ndarray] = None,
                 std: Optional[np.ndarray] = None) -> None:
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float64)
        self.std = None if std is None else np.asarray(std, dtype=np.float64)

    def fit(self, x: np.ndarray, eps: float = 1e-6) -> "Standardizer":
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        self.mean = x.mean(axis=0)
        self.std = np.maximum(x.std(axis=0), eps)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None:
            return np.asarray(x, dtype=np.float64)
        return (np.asarray(x, dtype=np.float64) - self.mean) / self.std

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        if self.mean is None:
            return np.asarray(z, dtype=np.float64)
        return np.asarray(z, dtype=np.float64) * self.std + self.mean


def _as_2d(x: ArrayLike) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def _as_rng(rng: Optional[Union[np.random.Generator, int]] = None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    if rng is None:
        return np.random.default_rng(0)
    return np.random.default_rng(int(rng))


def _call_with_seed_kwargs(fn: Callable, *args: Any, seed: int = 0,
                           **kwargs: Any) -> Any:
    """Call ``fn`` tolerating the two common dataset-generation signatures."""
    attempts = [
        dict(seed=seed, **kwargs),
        dict(rng=np.random.default_rng(seed), **kwargs),
        dict(**kwargs),
    ]
    last: Optional[Exception] = None
    for kw in attempts:
        try:
            return fn(*args, **kw)
        except TypeError as exc:  # signature mismatch
            last = exc
            continue
    if last is not None:
        raise last
    return fn(*args, **kwargs)  # pragma: no cover


def prior_bounds(task: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Best-effort extraction of uniform prior bounds ``(low, high)`` from a task."""
    lo = None
    hi = None
    for attr_lo, attr_hi, n_attr in (
        ("prior_low", "prior_high", "n_parameters"),
        ("mass_min", "mass_max", "n_parameters"),
        ("low", "high", "n_parameters"),
    ):
        if hasattr(task, attr_lo) and hasattr(task, attr_hi):
            n = int(getattr(task, n_attr, 1) or 1)
            lo = np.full(n, float(getattr(task, attr_lo)))
            hi = np.full(n, float(getattr(task, attr_hi)))
            break
    if lo is None and hasattr(task, "prior_mean") and hasattr(task, "prior_std"):
        n = int(getattr(task, "n_parameters", 1) or 1)
        mean = np.full(n, float(getattr(task, "prior_mean")))
        std = np.full(n, float(getattr(task, "prior_std")))
        lo, hi = mean - 4.0 * std, mean + 4.0 * std
    if lo is None:
        return None
    return lo, hi


def simulate_dataset(task: Any, n_simulations: int, seed: int = 0,
                     verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Draw ``(theta, x)`` training pairs from a task simulator."""
    n_simulations = int(n_simulations)
    rng = np.random.default_rng(seed)

    make = getattr(task, "make_dataset", None)
    if callable(make):
        try:
            out = _call_with_seed_kwargs(make, n_simulations, seed=seed,
                                         verbose=verbose)
            if isinstance(out, (tuple, list)) and len(out) == 2:
                return _as_2d(out[0]), _as_2d(out[1])
            if isinstance(out, np.ndarray) and out.ndim == 2:
                n_par = int(getattr(task, "n_parameters", 1) or 1)
                return out[:, :n_par], out[:, n_par:]
        except Exception:
            if verbose:
                print("[baseline] make_dataset failed, falling back to prior+simulate")

    theta = np.asarray(task.prior_sample(n_simulations, rng=rng), dtype=np.float64)
    theta = _as_2d(theta)
    try:
        x = task.simulate(theta, rng=rng)
    except TypeError:
        x = task.simulate(theta, rng)
    x = _as_2d(x)
    if x.shape[0] != theta.shape[0] and x.shape[0] == 1:
        x = np.repeat(x, theta.shape[0], axis=0)
    return theta, x


def task_prior_log_prob(task: Any) -> Callable[[np.ndarray], np.ndarray]:
    """Return a vectorised ``log p(theta)`` built from the task."""

    def log_prior(theta: np.ndarray) -> np.ndarray:
        theta = _as_2d(theta)
        try:
            out = np.asarray(task.log_prior(theta), dtype=np.float64)
        except Exception:
            out = np.array([float(task.log_prior(t)) for t in theta])
        out = np.atleast_1d(np.squeeze(out))
        if out.ndim == 0:
            out = np.full(theta.shape[0], float(out))
        return out

    return log_prior


def _posterior_condition_mask(task: Any) -> np.ndarray:
    n_par = int(getattr(task, "n_parameters", 1) or 1)
    n_dat = int(getattr(task, "n_data", 1) or 1)
    return np.concatenate([np.zeros(n_par), np.ones(n_dat)])


def _joint_from(task: Any, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    fn = getattr(task, "to_joint", None)
    if callable(fn):
        try:
            return np.asarray(fn(theta, x), dtype=np.float64)
        except Exception:
            pass
    return np.concatenate([_as_2d(theta), _as_2d(x)], axis=-1)


def reference_posterior_samples(task: Any, x_obs: np.ndarray, n_samples: int = 1000,
                                seed: int = 0) -> np.ndarray:
    """Ground-truth posterior samples for one observation (task-native or MCMC)."""
    x_obs = np.asarray(x_obs, dtype=np.float64).reshape(1, -1)
    rng = np.random.default_rng(seed)
    for name in ("reference_posterior_sample", "ground_truth_posterior"):
        fn = getattr(task, name, None)
        if callable(fn):
            try:
                out = fn(x_obs, n_samples=int(n_samples), rng=rng)
                return _as_2d(out)
            except TypeError:
                try:
                    return _as_2d(fn(x_obs, int(n_samples), rng))
                except Exception:
                    continue
            except Exception:
                continue
    # fall back to the NumPy reference MCMC implementation
    try:
        from ..reference.mcmc import sample_reference  # type: ignore
    except Exception:  # pragma: no cover
        try:
            from simformer.reference.mcmc import sample_reference  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"no reference posterior sampler available: {exc!r}")
    joint = _joint_from(task, np.zeros((1, int(getattr(task, "n_parameters", 1) or 1))), x_obs)
    mask = _posterior_condition_mask(task)
    return _as_2d(sample_reference(task, mask, joint, n_samples=int(n_samples), seed=seed))


# ---------------------------------------------------------------------------
# self-contained neural spline-free fallback: conditional MAF
# ---------------------------------------------------------------------------
class _MaskedLinear(nn.Module):  # type: ignore[misc]
    """Linear layer whose weights are masked according to MADE degrees."""

    def __init__(self, in_features: int, out_features: int,
                 in_degrees: np.ndarray, out_degrees: np.ndarray,
                 bias: bool = True) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        mask = (out_degrees[:, None] >= in_degrees[None, :]).astype(np.float32)
        self.register_buffer("mask", torch.tensor(mask))
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.05)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        w = self.weight * self.mask
        return F.linear(x, w, self.bias)


class _MADE(nn.Module):  # type: ignore[misc]
    """Masked autoencoder density estimator producing (mu, log_sigma)."""

    def __init__(self, n_inputs: int, context_dim: int, target_dim: int,
                 hidden_dims: Sequence[int] = DEFAULT_HIDDEN_DIMS,
                 activation: str = "relu") -> None:
        super().__init__()
        self.target_dim = int(target_dim)
        max_degree = max(self.target_dim - 1, 0)
        in_degrees = np.concatenate([
            np.full(int(context_dim), max_degree, dtype=np.float64),
            np.arange(self.target_dim, dtype=np.float64),
        ])
        hidden_dims = [int(h) for h in hidden_dims] or [128]
        dims = [int(n_inputs)] + hidden_dims
        self.layers = nn.ModuleList()
        rng = np.random.default_rng(0)
        for i in range(len(dims) - 1):
            if i < len(dims) - 2:
                out_deg = rng.integers(0, max_degree + 1, size=dims[i + 1]).astype(np.float64)
            else:
                out_deg = np.tile(np.arange(self.target_dim, dtype=np.float64), 2)
            self.layers.append(_MaskedLinear(dims[i], dims[i + 1], in_degrees, out_deg))
            in_degrees = out_deg
        self.activation = _get_activation(activation)

    def forward(self, z: "torch.Tensor", context: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        h = torch.cat([context, z], dim=-1)
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = self.activation(h)
        mu, log_sigma = h[:, : self.target_dim], h[:, self.target_dim:]
        log_sigma = torch.clamp(log_sigma, -6.0, 6.0)
        return mu, log_sigma


def _get_activation(name: str) -> Callable[["torch.Tensor"], "torch.Tensor"]:
    name = str(name).lower()
    if name in ("relu",):
        return F.relu
    if name in ("gelu",):
        return F.gelu
    if name in ("silu", "swish"):
        return F.silu
    if name in ("tanh",):
        return torch.tanh
    return F.relu


class ConditionalMAF(nn.Module):  # type: ignore[misc]
    """Stack of affine autoregressive transforms (MAF) conditioned on a context.

    This is the self-contained stand-in for ``sbi``'s neural spline flow.  It is
    trained with the identical protocol (batch size 1000, Adam, early stopping
    on validation loss) and supports both density evaluation and sampling.
    """

    def __init__(self, target_dim: int, context_dim: int,
                 n_transforms: int = DEFAULT_N_TRANSFORMS,
                 hidden_dims: Sequence[int] = DEFAULT_HIDDEN_DIMS,
                 activation: str = "relu") -> None:
        super().__init__()
        self.target_dim = int(target_dim)
        self.context_dim = int(context_dim)
        self.n_transforms = int(n_transforms)
        self.transforms = nn.ModuleList([
            _MADE(self.target_dim + self.context_dim, self.context_dim,
                  self.target_dim, hidden_dims=hidden_dims, activation=activation)
            for _ in range(self.n_transforms)
        ])

    # -- density -----------------------------------------------------------
    def forward(self, target: "torch.Tensor",
                context: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        z = target
        log_det = torch.zeros(target.shape[0], device=target.device, dtype=target.dtype)
        for made in self.transforms:
            mu, log_sigma = made(z, context)
            z = (z - mu) * torch.exp(-log_sigma)
            log_det = log_det - log_sigma.sum(dim=-1)
        return z, log_det

    def log_prob(self, target: "torch.Tensor", context: "torch.Tensor") -> "torch.Tensor":
        z, log_det = self.forward(target, context)
        base = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * self.target_dim * math.log(2.0 * math.pi)
        return base + log_det

    # -- sampling ----------------------------------------------------------
    def inverse(self, u: "torch.Tensor", context: "torch.Tensor") -> "torch.Tensor":
        x = u.clone()
        for made in reversed(self.transforms):
            for i in range(self.target_dim):
                mu, log_sigma = made(x, context)
                x = torch.cat([
                    x[:, :i],
                    (u[:, i:i + 1] * torch.exp(log_sigma[:, i:i + 1]) + mu[:, i:i + 1]),
                    x[:, i + 1:],
                ], dim=-1) if i + 1 < self.target_dim or i > 0 else x
                if i == 0 and self.target_dim == 1:
                    x = u[:, 0:1] * torch.exp(log_sigma[:, 0:1]) + mu[:, 0:1]
            # end dimension loop
        return x

    def sample(self, context: "torch.Tensor", n_samples: int = 1,
               generator: Optional[Any] = None) -> "torch.Tensor":
        u = torch.randn(context.shape[0], self.target_dim, device=context.device,
                        dtype=context.dtype, generator=generator)
        return self.inverse(u, context)


class _ConditionalDensityNetwork:
    """Train/eval wrapper around :class:`ConditionalMAF` (NumPy in/out)."""

    def __init__(self, target_dim: int, context_dim: int,
                 config: Optional[BaselineConfig] = None) -> None:
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required for the baseline fallback flows")
        cfg = config or BaselineConfig()
        self.config = cfg
        self.target_dim = int(target_dim)
        self.context_dim = int(context_dim)
        torch.manual_seed(cfg.seed)
        self.net = ConditionalMAF(
            self.target_dim, self.context_dim,
            n_transforms=cfg.n_transforms,
            hidden_dims=cfg.hidden_dims,
        ).to(cfg.device)
        self.target_scaler = Standardizer()
        self.context_scaler = Standardizer()
        self.history: Dict[str, List[float]] = {"train": [], "val": []}
        self.best_val = float("inf")

    # -- data preparation --------------------------------------------------
    def prepare(self, target: np.ndarray, context: np.ndarray,
                fit: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        target = _as_2d(target)
        context = _as_2d(context)
        if fit:
            if self.config.standardize:
                self.target_scaler.fit(target)
                self.context_scaler.fit(context)
        return self.target_scaler.transform(target), self.context_scaler.transform(context)

    # -- training ----------------------------------------------------------
    def fit(self, target: np.ndarray, context: np.ndarray,
            config: Optional[BaselineConfig] = None) -> Dict[str, Any]:
        cfg = config or self.config
        self.config = cfg
        t_scaled, c_scaled = self.prepare(target, context, fit=True)
        n = t_scaled.shape[0]
        n_val = max(int(round(cfg.validation_fraction * n)), 1) if n > 10 else 0
        n_train = n - n_val
        rng = np.random.default_rng(cfg.seed)
        perm = rng.permutation(n)
        train_idx, val_idx = perm[:n_train], perm[n_train:]

        t_all = torch.tensor(t_scaled, dtype=torch.float32)
        c_all = torch.tensor(c_scaled, dtype=torch.float32)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=float(cfg.lr))
        best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
        best_val = float("inf")
        patience = 0
        n_seen = 0
        epochs = max(int(cfg.max_epochs), 1)
        batch_size = max(int(cfg.batch_size), 1)

        for epoch in range(epochs):
            self.net.train()
            order = rng.permutation(n_train)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_train, batch_size):
                idx = order[start:start + batch_size]
                if idx.size == 0:
                    continue
                tb = t_all[train_idx[idx]]
                cb = c_all[train_idx[idx]]
                loss = -self.net.log_prob(tb, cb).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                optimizer.step()
                epoch_loss += float(loss.detach())
                n_batches += 1
                n_seen += int(idx.size)
            epoch_loss /= max(n_batches, 1)
            self.history["train"].append(epoch_loss)

            if n_val > 0:
                self.net.eval()
                with torch.no_grad():
                    vl = -self.net.log_prob(t_all[val_idx], c_all[val_idx]).mean()
                val_loss = float(vl)
                self.history["val"].append(val_loss)
                if val_loss < best_val - 1e-4:
                    best_val = val_loss
                    patience = 0
                    best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                else:
                    patience += 1
                if patience >= int(cfg.patience):
                    if cfg.verbose:
                        print(f"[baseline] early stopping at epoch {epoch} "
                              f"(best val {best_val:.4f})")
                    break
            else:
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                epoch_loss_prev = self.history["train"][-2] if len(self.history["train"]) > 1 else None
                if epoch_loss_prev is not None and epoch_loss > epoch_loss_prev:
                    patience += 1
                    if patience >= int(cfg.patience):
                        break
        self.net.load_state_dict(best_state)
        self.best_val = best_val
        return {"train_loss": self.history["train"], "val_loss": self.history["val"],
                "best_val": best_val, "n_train": n_train, "n_val": n_val,
                "epochs": len(self.history["train"])}

    # -- inference ---------------------------------------------------------
    def _context_tensor(self, context: np.ndarray) -> "torch.Tensor":
        c = self.context_scaler.transform(_as_2d(context))
        return torch.tensor(c, dtype=torch.float32, device=self.config.device)

    def log_prob(self, target: np.ndarray, context: np.ndarray) -> np.ndarray:
        self.net.eval()
        t = self.target_scaler.transform(_as_2d(target))
        with torch.no_grad():
            lp = self.net.log_prob(
                torch.tensor(t, dtype=torch.float32, device=self.config.device),
                self._context_tensor(context),
            )
        return lp.cpu().numpy().reshape(-1)

    def sample(self, context: np.ndarray, n_samples: int = 1,
               rng: Optional[np.random.Generator] = None) -> np.ndarray:
        self.net.eval()
        ctx = np.repeat(_as_2d(context), int(n_samples), axis=0)
        with torch.no_grad():
            z = self.net.sample(self._context_tensor(ctx), int(n_samples))
        return self.target_scaler.inverse_transform(z.cpu().numpy())


class _RatioClassifier(nn.Module):  # type: ignore[misc]
    """MLP classifier used by the NRE fallback."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = DEFAULT_HIDDEN_DIMS,
                 activation: str = "relu") -> None:
        super().__init__()
        dims = [int(input_dim)] + [int(h) for h in hidden_dims] + [1]
        layers: List[Any] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
                layers.append(_get_activation(activation)())
        self.net = nn.Sequential(*layers)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# MCMC posterior sampling for likelihood-/ratio-based baselines
# ---------------------------------------------------------------------------
def _mcmc_posterior_samples(log_prob: Callable[[np.ndarray], np.ndarray],
                            init: np.ndarray, n_steps: int = DEFAULT_MCMC_STEPS,
                            step_size: float = DEFAULT_MCMC_STEP_SIZE,
                            rng: Optional[np.random.Generator] = None,
                            target_accept: float = 0.3) -> np.ndarray:
    """Vectorised random-walk Metropolis-Hastings; keeps the last sample/chain.

    Mirrors the reference protocol of Appendix A2.2 (chains initialised from the
    joint, last sample per chain retained).
    """
    rng = _as_rng(rng)
    x = np.array(_as_2d(init), dtype=np.float64, copy=True)
    n_chains, dim = x.shape
    lp = np.asarray(log_prob(x), dtype=np.float64).reshape(-1)
    log_step = math.log(max(step_size, 1e-6))
    n_accept = 0
    for step in range(int(n_steps)):
        proposal = x + math.exp(log_step) * rng.standard_normal((n_chains, dim))
        lp_new = np.asarray(log_prob(proposal), dtype=np.float64).reshape(-1)
        with np.errstate(invalid="ignore"):
            accept = np.log(rng.uniform(size=n_chains)) < (lp_new - lp)
        accept &= np.isfinite(lp_new)
        x[accept] = proposal[accept]
        lp[accept] = lp_new[accept]
        n_accept += int(accept.sum())
        if step == 100:  # short adaptation phase
            rate = n_accept / (n_chains * 101)
            log_step += 0.5 * (math.log(max(rate, 1e-3)) - math.log(target_accept))
    return x


# ---------------------------------------------------------------------------
# baseline objects
# ---------------------------------------------------------------------------
class BaselinePosterior:
    """Common interface for posterior/likelihood/ratio baselines."""

    method = "baseline"

    def __init__(self, task: Any, config: Optional[BaselineConfig] = None) -> None:
        self.task = task
        self.config = config or BaselineConfig()
        self.name = f"{self.method}"
        self.train_info: Dict[str, Any] = {}
        self.use_sbi = bool(_HAS_SBI and (self.config.use_sbi is not False))
        self.sbi_error: Optional[str] = None
        self.dim = int(getattr(task, "n_parameters", 1) or 1)

    # -- API ---------------------------------------------------------------
    def posterior_samples(self, x_obs: ArrayLike, n_samples: int = 1000,
                          seed: int = 0) -> np.ndarray:
        raise NotImplementedError

    def sample(self, n_samples: int, x_obs: ArrayLike, seed: int = 0,
               rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is not None and not isinstance(rng, (int, type(None))):
            seed = int(np.random.default_rng().integers(2 ** 31 - 1)) if isinstance(rng, int) else seed
        return self.posterior_samples(x_obs, n_samples=n_samples, seed=seed)

    def __call__(self, n_samples: int, x_obs: ArrayLike, seed: int = 0) -> np.ndarray:
        return self.sample(n_samples, x_obs, seed=seed)

    def log_likelihood(self, x: ArrayLike, theta: ArrayLike) -> np.ndarray:
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "method": self.method,
                "config": self.config.to_dict(), "train_info": self._jsonable(self.train_info),
                "used_sbi": self.sbi_error is None and self.use_sbi}

    @staticmethod
    def _jsonable(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: BaselinePosterior._jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [BaselinePosterior._jsonable(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        return str(obj)


class NPEBaseline(BaselinePosterior):
    """Neural Posterior Estimation: amortised ``q(theta | x)``.

    Uses ``sbi`` with a neural spline flow when available (Appendix A2.1),
    otherwise a conditional MAF trained with the same protocol.
    """

    method = "npe"

    def __init__(self, task: Any, config: Optional[BaselineConfig] = None) -> None:
        super().__init__(task, config)
        self.posterior = None       # sbi posterior object
        self.density = None         # fallback flow
        self.prior = None

    def fit(self, theta: np.ndarray, x: np.ndarray,
            config: Optional[BaselineConfig] = None) -> Dict[str, Any]:
        cfg = config or self.config
        self.config = cfg
        self.dim = _as_2d(theta).shape[1]
        if self.use_sbi:
            try:
                return self._fit_sbi(theta, x, cfg)
            except Exception as exc:  # pragma: no cover - depends on install
                self.sbi_error = repr(exc)
                if cfg.verbose:
                    print(f"[baseline] sbi NPE failed ({exc!r}); using fallback flow")
        return self._fit_fallback(theta, x, cfg)

    def _fit_sbi(self, theta: np.ndarray, x: np.ndarray, cfg: BaselineConfig) -> Dict[str, Any]:
        self.prior = _make_box_prior(self.task)
        inference = NPE(prior=self.prior, density_estimator=cfg.density_estimator,
                        device=cfg.device, show_progress_bars=False)
        inference.append_simulations(
            torch.tensor(_as_2d(theta), dtype=torch.float32),
            torch.tensor(_as_2d(x), dtype=torch.float32),
        )
        density_estimator = inference.train(
            training_batch_size=int(cfg.batch_size),
            learning_rate=float(cfg.lr),
            max_num_epochs=int(cfg.max_epochs),
            stop_after_epochs=int(cfg.patience),
            validation_fraction=float(cfg.validation_fraction),
            show_train_summary=bool(cfg.verbose),
        )
        self.posterior = inference.build_posterior(density_estimator)
        self.train_info = {"backend": "sbi", "density_estimator": cfg.density_estimator,
                           "n_simulations": int(_as_2d(theta).shape[0])}
        return self.train_info

    def _fit_fallback(self, theta: np.ndarray, x: np.ndarray, cfg: BaselineConfig) -> Dict[str, Any]:
        theta, x = _as_2d(theta), _as_2d(x)
        self.density = _ConditionalDensityNetwork(theta.shape[1], x.shape[1], cfg)
        info = self.density.fit(theta, x, cfg)
        self.train_info = {"backend": "fallback_maf", **info}
        return self.train_info

    def posterior_samples(self, x_obs: ArrayLike, n_samples: int = 1000,
                          seed: int = 0) -> np.ndarray:
        x_obs = _as_2d(x_obs)
        if self.posterior is not None:
            x_t = torch.tensor(x_obs, dtype=torch.float32)
            samples = self.posterior.sample((int(n_samples),), x=x_t,
                                            show_progress_bars=False)
            return samples.detach().cpu().numpy().reshape(int(n_samples), -1)
        if self.density is None:
            raise RuntimeError("NPE baseline has not been trained")
        return self.density.sample(x_obs, int(n_samples),
                                   rng=np.random.default_rng(seed))

    def log_prob(self, theta: ArrayLike, x: ArrayLike) -> np.ndarray:
        if self.posterior is not None:
            raise NotImplementedError("sbi posteriors do not expose log_prob directly")
        return self.density.log_prob(theta, x)


class NLEBaseline(BaselinePosterior):
    """Neural Likelihood Estimation: ``q(x | theta)`` + MCMC for the posterior."""

    method = "nle"

    def __init__(self, task: Any, config: Optional[BaselineConfig] = None) -> None:
        super().__init__(task, config)
        self.posterior = None
        self.density = None
        self.prior = None
        self.log_prior = task_prior_log_prob(task)

    def fit(self, theta: np.ndarray, x: np.ndarray,
            config: Optional[BaselineConfig] = None) -> Dict[str, Any]:
        cfg = config or self.config
        self.config = cfg
        self.dim = _as_2d(theta).shape[1]
        if self.use_sbi:
            try:
                return self._fit_sbi(theta, x, cfg)
            except Exception as exc:  # pragma: no cover
                self.sbi_error = repr(exc)
                if cfg.verbose:
                    print(f"[baseline] sbi NLE failed ({exc!r}); using fallback flow")
        return self._fit_fallback(theta, x, cfg)

    def _fit_sbi(self, theta: np.ndarray, x: np.ndarray, cfg: BaselineConfig) -> Dict[str, Any]:
        self.prior = _make_box_prior(self.task)
        inference = NLE(prior=self.prior, density_estimator=cfg.density_estimator,
                        device=cfg.device, show_progress_bars=False)
        inference.append_simulations(
            torch.tensor(_as_2d(theta), dtype=torch.float32),
            torch.tensor(_as_2d(x), dtype=torch.float32),
        )
        density_estimator = inference.train(
            training_batch_size=int(cfg.batch_size),
            learning_rate=float(cfg.lr),
            max_num_epochs=int(cfg.max_epochs),
            stop_after_epochs=int(cfg.patience),
            validation_fraction=float(cfg.validation_fraction),
            show_train_summary=bool(cfg.verbose),
        )
        self.posterior = inference.build_posterior(density_estimator, prior=self.prior)
        self.train_info = {"backend": "sbi", "density_estimator": cfg.density_estimator}
        return self.train_info

    def _fit_fallback(self, theta: np.ndarray, x: np.ndarray, cfg: BaselineConfig) -> Dict[str, Any]:
        theta, x = _as_2d(theta), _as_2d(x)
        self.n_data = x.shape[1]
        self.density = _ConditionalDensityNetwork(x.shape[1], theta.shape[1], cfg)
        info = self.density.fit(x, theta, cfg)
        self.train_info = {"backend": "fallback_maf", **info}
        return self.train_info

    # -- likelihood --------------------------------------------------------
    def log_likelihood(self, x: ArrayLike, theta: ArrayLike) -> np.ndarray:
        x, theta = _as_2d(x), _as_2d(theta)
        if x.shape[0] == 1 and theta.shape[0] > 1:
            x = np.repeat(x, theta.shape[0], axis=0)
        if theta.shape[0] == 1 and x.shape[0] > 1:
            theta = np.repeat(theta, x.shape[0], axis=0)
        if self.density is None:
            raise RuntimeError("NLE baseline has not been trained")
        return self.density.log_prob(x, theta)

    def posterior_samples(self, x_obs: ArrayLike, n_samples: int = 1000,
                          seed: int = 0) -> np.ndarray:
        x_obs = _as_2d(x_obs)
        if self.posterior is not None:
            x_t = torch.tensor(x_obs, dtype=torch.float32)
            samples = self.posterior.sample((int(n_samples),), x=x_t,
                                            show_progress_bars=False)
            return samples.detach().cpu().numpy().reshape(int(n_samples), -1)
        if self.density is None:
            raise RuntimeError("NLE baseline has not been trained")
        rng = np.random.default_rng(seed)

        def log_prob(theta: np.ndarray) -> np.ndarray:
            return self.log_prior(theta) + self.log_likelihood(x_obs, theta)

        init = self._init_chains(int(n_samples), rng)
        return _mcmc_posterior_samples(log_prob, init, n_steps=self.config.mcmc_steps,
                                       step_size=self.config.mcmc_step_size, rng=rng)

    def _init_chains(self, n_chains: int, rng: np.random.Generator) -> np.ndarray:
        try:
            init = _as_2d(self.task.prior_sample(n_chains, rng=rng))
        except Exception:
            init = rng.standard_normal((n_chains, self.dim))
        if init.shape[0] != n_chains:
            init = np.repeat(init[:1], n_chains, axis=0) + 0.1 * rng.standard_normal((n_chains, init.shape[1]))
        return init


class NREBaseline(BaselinePosterior):
    """Neural Ratio Estimation: classifier ratio ``r(theta, x)`` + MCMC."""

    method = "nre"

    def __init__(self, task: Any, config: Optional[BaselineConfig] = None) -> None:
        super().__init__(task, config)
        self.classifier = None
        self.log_prior = task_prior_log_prob(task)
        self.log_ratio = None

    def fit(self, theta: np.ndarray, x: np.ndarray,
            config: Optional[BaselineConfig] = None) -> Dict[str, Any]:
        cfg = config or self.config
        self.config = cfg
        theta, x = _as_2d(theta), _as_2d(x)
        self.dim = theta.shape[1]
        if self.use_sbi:
            try:
                return self._fit_sbi(theta, x, cfg)
            except Exception as exc:  # pragma: no cover
                self.sbi_error = repr(exc)
                if cfg.verbose:
                    print(f"[baseline] sbi NRE failed ({exc!r}); using fallback classifier")
        return self._fit_fallback(theta, x, cfg)

    def _fit_sbi(self, theta: np.ndarray, x: np.ndarray, cfg: BaselineConfig) -> Dict[str, Any]:
        self.prior = _make_box_prior(self.task)
        inference = NRE(prior=self.prior, classifier="resnet", device=cfg.device,
                        show_progress_bars=False)
        inference.append_simulations(
            torch.tensor(theta, dtype=torch.float32),
            torch.tensor(x, dtype=torch.float32),
        )
        classifier = inference.train(
            training_batch_size=int(cfg.batch_size),
            learning_rate=float(cfg.lr),
            max_num_epochs=int(cfg.max_epochs),
            stop_after_epochs=int(cfg.patience),
            validation_fraction=float(cfg.validation_fraction),
            show_train_summary=bool(cfg.verbose),
        )
        self.posterior = inference.build_posterior(classifier, prior=self.prior)
        self.train_info = {"backend": "sbi", "classifier": "resnet"}
        return self.train_info

    def _fit_fallback(self, theta: np.ndarray, x: np.ndarray, cfg: BaselineConfig) -> Dict[str, Any]:
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required for the NRE fallback classifier")
        torch.manual_seed(cfg.seed)
        n = theta.shape[0]
        rng = np.random.default_rng(cfg.seed)
        theta_scaler = Standardizer().fit(theta)
        x_scaler = Standardizer().fit(x)
        t_scaled = theta_scaler.transform(theta)
        x_scaled = x_scaler.transform(x)

        input_dim = theta.shape[1] + x.shape[1]
        clf = _RatioClassifier(input_dim, cfg.hidden_dims).to(cfg.device)
        optimizer = torch.optim.Adam(clf.parameters(), lr=float(cfg.lr))

        n_val = max(int(round(cfg.validation_fraction * n)), 1) if n > 10 else 0
        idx = rng.permutation(n)
        train_idx, val_idx = idx[: n - n_val], idx[n - n_val:]

        best_state = {k: v.detach().clone() for k, v in clf.state_dict().items()}
        best_val, patience = float("inf"), 0
        batch_size = max(int(cfg.batch_size), 1)
        train_losses: List[float] = []
        val_losses: List[float] = []

        for epoch in range(max(int(cfg.max_epochs), 1)):
            clf.train()
            order = rng.permutation(train_idx.size)
            epoch_loss, n_batches = 0.0, 0
            for start in range(0, train_idx.size, batch_size):
                sel = train_idx[order[start:start + batch_size]]
                perm_x = x_scaled[rng.permutation(sel)]
                pos = np.concatenate([t_scaled[sel], x_scaled[sel]], axis=-1)
                neg = np.concatenate([t_scaled[sel], perm_x], axis=-1)
                X = torch.tensor(np.concatenate([pos, neg], axis=0), dtype=torch.float32,
                                 device=cfg.device)
                y = torch.cat([torch.ones(pos.shape[0]), torch.zeros(neg.shape[0])]).to(cfg.device)
                logits = clf(X)
                loss = F.binary_cross_entropy_with_logits(logits, y)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(clf.parameters(), 5.0)
                optimizer.step()
                epoch_loss += float(loss.detach())
                n_batches += 1
            train_losses.append(epoch_loss / max(n_batches, 1))
            if val_idx.size > 0:
                clf.eval()
                with torch.no_grad():
                    perm_x = x_scaled[rng.permutation(val_idx)]
                    pos = np.concatenate([t_scaled[val_idx], x_scaled[val_idx]], axis=-1)
                    neg = np.concatenate([t_scaled[val_idx], perm_x], axis=-1)
                    X = torch.tensor(np.concatenate([pos, neg], axis=0), dtype=torch.float32,
                                     device=cfg.device)
                    y = torch.cat([torch.ones(pos.shape[0]), torch.zeros(neg.shape[0])]).to(cfg.device)
                    vl = float(F.binary_cross_entropy_with_logits(clf(X), y))
                val_losses.append(vl)
                if vl < best_val - 1e-4:
                    best_val = vl
                    patience = 0
                    best_state = {k: v.detach().clone() for k, v in clf.state_dict().items()}
                else:
                    patience += 1
                    if patience >= int(cfg.patience):
                        break
        clf.load_state_dict(best_state)
        self.classifier = clf
        self.theta_scaler = theta_scaler
        self.x_scaler = x_scaler
        self.train_info = {"backend": "fallback_classifier", "train_loss": train_losses,
                           "val_loss": val_losses, "best_val": best_val}
        return self.train_info

    # -- ratio -------------------------------------------------------------
    def _log_ratio_numpy(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        if self.classifier is None:
            raise RuntimeError("NRE baseline has not been trained")
        t = self.theta_scaler.transform(_as_2d(theta))
        xs = self.x_scaler.transform(_as_2d(x))
        if t.shape[0] == 1 and xs.shape[0] > 1:
            t = np.repeat(t, xs.shape[0], axis=0)
        if xs.shape[0] == 1 and t.shape[0] > 1:
            xs = np.repeat(xs, t.shape[0], axis=0)
        X = torch.tensor(np.concatenate([t, xs], axis=-1), dtype=torch.float32,
                         device=self.config.device)
        self.classifier.eval()
        with torch.no_grad():
            logits = self.classifier(X)
        return logits.cpu().numpy().reshape(-1)

    def posterior_samples(self, x_obs: ArrayLike, n_samples: int = 1000,
                          seed: int = 0) -> np.ndarray:
        x_obs = _as_2d(x_obs)
        if self.posterior is not None:
            x_t = torch.tensor(x_obs, dtype=torch.float32)
            samples = self.posterior.sample((int(n_samples),), x=x_t,
                                            show_progress_bars=False)
            return samples.detach().cpu().numpy().reshape(int(n_samples), -1)
        rng = np.random.default_rng(seed)

        def log_prob(theta: np.ndarray) -> np.ndarray:
            return self.log_prior(theta) + self._log_ratio_numpy(theta, x_obs)

        try:
            init = _as_2d(self.task.prior_sample(int(n_samples), rng=rng))
        except Exception:
            init = rng.standard_normal((int(n_samples), self.dim))
        return _mcmc_posterior_samples(log_prob, init, n_steps=self.config.mcmc_steps,
                                       step_size=self.config.mcmc_step_size, rng=rng)


def _make_box_prior(task: Any) -> Any:
    """Build an ``sbi`` BoxUniform prior from the task's prior bounds (or None)."""
    bounds = prior_bounds(task)
    if bounds is None or BoxUniform is None:
        return None
    lo, hi = bounds
    return BoxUniform(low=torch.tensor(lo, dtype=torch.float32),
                      high=torch.tensor(hi, dtype=torch.float32))


# ---------------------------------------------------------------------------
# training entry points
# ---------------------------------------------------------------------------
def train_baseline(method: str, task: Any, n_simulations: int = 10_000,
                   config: Optional[BaselineConfig] = None, seed: int = 0,
                   verbose: bool = False, **kwargs: Any) -> BaselinePosterior:
    """Train one baseline on a task using ``n_simulations`` simulations."""
    method = normalize_method(method)
    cfg = BaselineConfig.from_dict(config, method=method, n_simulations=n_simulations,
                                   seed=seed, verbose=verbose, **kwargs)
    theta, x = simulate_dataset(task, n_simulations, seed=seed, verbose=verbose)
    cls = {"npe": NPEBaseline, "nle": NLEBaseline, "nre": NREBaseline}[method]
    baseline = cls(task, cfg)
    t0 = time.time()
    info = baseline.fit(theta, x, cfg)
    info["elapsed_seconds"] = time.time() - t0
    info["n_simulations"] = int(theta.shape[0])
    baseline.train_info = info
    if verbose:
        print(f"[baseline] {method}: trained on {theta.shape[0]} simulations "
              f"in {info['elapsed_seconds']:.1f}s (backend "
              f"{info.get('backend', 'sbi')})")
    return baseline


def train_npe(task: Any, n_simulations: int = 10_000, **kwargs: Any) -> NPEBaseline:
    """Train NPE (neural spline flow / MAF fallback) on a task."""
    return train_baseline("npe", task, n_simulations=n_simulations, **kwargs)  # type: ignore[return-value]


def train_nle(task: Any, n_simulations: int = 10_000, **kwargs: Any) -> NLEBaseline:
    """Train NLE on a task."""
    return train_baseline("nle", task, n_simulations=n_simulations, **kwargs)  # type: ignore[return-value]


def train_nre(task: Any, n_simulations: int = 10_000, **kwargs: Any) -> NREBaseline:
    """Train NRE on a task."""
    return train_baseline("nre", task, n_simulations=n_simulations, **kwargs)  # type: ignore[return-value]


def build_baseline(method: str, task: Any, n_simulations: int = 10_000,
                   config: Optional[BaselineConfig] = None, seed: int = 0,
                   verbose: bool = False, **kwargs: Any) -> BaselinePosterior:
    """Factory used by the experiment scripts (``method`` in npe/nle/nre)."""
    return train_baseline(method, task, n_simulations=n_simulations, config=config,
                          seed=seed, verbose=verbose, **kwargs)


def build_npe(task: Any, *args: Any, **kwargs: Any) -> NPEBaseline:
    """Alias for :func:`train_npe` (``*args`` may hold ``n_simulations``)."""
    if args and "n_simulations" not in kwargs:
        kwargs["n_simulations"] = args[0]
    return train_npe(task, **kwargs)


def build_nle(task: Any, *args: Any, **kwargs: Any) -> NLEBaseline:
    """Alias for :func:`train_nle`."""
    if args and "n_simulations" not in kwargs:
        kwargs["n_simulations"] = args[0]
    return train_nle(task, **kwargs)


def build_nre(task: Any, *args: Any, **kwargs: Any) -> NREBaseline:
    """Alias for :func:`train_nre`."""
    if args and "n_simulations" not in kwargs:
        kwargs["n_simulations"] = args[0]
    return train_nre(task, **kwargs)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def evaluate_baseline_c2st(baseline: BaselinePosterior, task: Any, *,
                           n_targets: int = DEFAULT_N_TARGETS,
                           n_samples: int = DEFAULT_N_EVAL_SAMPLES,
                           n_reference: int = DEFAULT_N_REFERENCE,
                           seed: int = 0, verbose: bool = False,
                           return_result: bool = False, **kwargs: Any) -> Dict[str, Any]:
    """C2ST accuracy of a baseline posterior against ground-truth references.

    Follows the protocol of Sec. 4.1: several observations are drawn from the
    joint distribution, ground-truth posterior samples are obtained with MCMC
    (or the task-native reference sampler) and the C2ST accuracy (0.5 = perfect)
    is averaged over observations.  Returns a dict with the mean/std accuracy.
    """
    try:
        from ..eval.c2st import c2st_accuracy  # type: ignore
    except Exception:  # pragma: no cover
        try:
            from simformer.eval.c2st import c2st_accuracy  # type: ignore
        except Exception:
            c2st_accuracy = None  # type: ignore

    rng = np.random.default_rng(seed)
    theta, x = simulate_dataset(task, max(int(n_targets), 1), seed=seed)
    accuracies: List[float] = []
    errors: List[str] = []
    for i in range(int(n_targets)):
        x_obs = x[i]
        try:
            approx = _as_2d(baseline.sample(int(n_samples), x_obs, seed=seed + i))
        except Exception as exc:  # pragma: no cover
            errors.append(repr(exc))
            continue
        try:
            ref = reference_posterior_samples(task, x_obs, n_samples=int(n_reference),
                                              seed=seed + i)
        except Exception as exc:  # pragma: no cover
            errors.append(repr(exc))
            continue
        if c2st_accuracy is None:
            errors.append("c2st module unavailable")
            continue
        try:
            acc = c2st_accuracy(approx, ref, n_trees=100, seed=seed + i,
                                return_result=return_result)
        except TypeError:
            acc = c2st_accuracy(approx, ref, n_trees=100, seed=seed + i)
        if hasattr(acc, "accuracy"):
            accuracies.append(float(acc.accuracy))
        else:
            accuracies.append(float(acc))

    n = len(accuracies)
    if n == 0:
        return {"method": baseline.method, "mean": float("nan"), "std": float("nan"),
                "n": 0, "errors": errors}
    mean = float(np.mean(accuracies))
    std = float(np.std(accuracies, ddof=1)) if n > 1 else 0.0
    result = {"method": baseline.method, "mean": mean, "std": std, "n": n,
              "accuracies": [float(a) for a in accuracies],
              "stderr": std / math.sqrt(n) if n > 0 else float("nan"),
              "errors": errors}
    if verbose:
        print(f"[baseline] {baseline.method} C2ST = {mean:.3f} +- {std:.3f} (n={n})")
    return result


# ---------------------------------------------------------------------------
# module level registry
# ---------------------------------------------------------------------------
BASELINE_BUILDERS: Dict[str, Callable[..., BaselinePosterior]] = {
    "npe": build_npe,
    "nle": build_nle,
    "nre": build_nre,
}

__all__ = [
    "BaselineConfig",
    "BaselinePosterior",
    "NPEBaseline",
    "NLEBaseline",
    "NREBaseline",
    "ConditionalMAF",
    "Standardizer",
    "BASELINE_BUILDERS",
    "build_baseline",
    "build_npe",
    "build_nle",
    "build_nre",
    "train_baseline",
    "train_npe",
    "train_nle",
    "train_nre",
    "evaluate_baseline_c2st",
    "available_methods",
    "normalize_method",
    "sbi_available",
    "simulate_dataset",
    "reference_posterior_samples",
    "prior_bounds",
    "task_prior_log_prob",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_LR",
    "DEFAULT_PATIENCE",
    "DEFAULT_DENSITY_ESTIMATOR",
]
