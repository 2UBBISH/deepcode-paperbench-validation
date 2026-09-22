"""PCA-based state-space diversity metric (SAPG Sec. 6.4, Figure 7).

The paper defines the metric as follows:

    "PCA - We compute the reconstruction error of a batch of states using ``k``
     most significant components of PCA and plot this error as a function of
     ``k``. In general, a set that has variation along fewer dimensions of space
     can be compressed with fewer principal vectors and will have lower
     reconstruction error. This metric therefore measures the extent to which
     the policy explores different dimensions of state space. Figure-7 contains
     the plots for this metric. We find that the rate of decrease in
     reconstruction error with an increase in components is the slowest for our
     method."                                                    --- Sec. 6.4

Implementation notes
--------------------
* The metric only needs a batch of visited environment states
  ``X in R^{num_samples x state_dim}``.  We compute the (economy-size) SVD of
  the mean-centred matrix ``X = U S V^T`` and project onto the ``k`` leading
  right-singular vectors, i.e. ``X_k = X V_k V_k^T``.
* Three error conventions are supported:
    - ``"mse"``      : ``mean_{samples, dims} (x - x_hat)^2`` (default)
    - ``"sum"``      : ``mean_{samples} ||x - x_hat||^2``
    - ``"relative"`` : ``sum ||x - x_hat||^2 / sum ||x - mean(x)||^2`` -- the
      fraction of (centred) variance *not* captured by the ``k`` components.
  Results are comparable across methods only for a fixed convention; the paper
  does not specify one, so ``"mse"`` is used by default.
* Standardisation (z-scoring each state dimension before the SVD) is optional
  and enabled by default; it prevents state dimensions with a naturally larger
  numerical scale from dominating the principal directions.
* ``scikit-learn`` is used when available (``sklearn.decomposition.PCA``), with
  a dependency-free ``numpy``/``torch`` SVD fallback so the metric works in a
  minimal environment.  The plan lists scikit-learn as a dependency, but the
  fallback keeps unit tests runnable without it.

The module is deliberately policy/trainer agnostic: it consumes plain state
batches.  Use :func:`states_from_buffer` / :func:`collect_states` to obtain them
from a :class:`~sapg.buffers.rollout_buffer.RolloutBuffer` or a live
environment.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

try:  # pragma: no cover - optional dependency
    import numpy as _np
except Exception:  # pragma: no cover
    _np = None

try:  # pragma: no cover - optional dependency
    import torch

    HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None
    HAS_TORCH = False

try:  # pragma: no cover - optional dependency
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:  # pragma: no cover
    plt = None
    HAS_MATPLOTLIB = False


__all__ = [
    "METHOD_SAPG",
    "METHOD_PPO",
    "METHOD_RANDOM",
    "DEFAULT_K_VALUES",
    "PCAConfig",
    "PCADiversity",
    "PCAResult",
    "pca_reconstruction_error",
    "pca_reconstruction_curve",
    "pca_components",
    "explained_variance_ratio",
    "reconstruction_error_per_dimension",
    "states_from_buffer",
    "collect_states",
    "compare_pca_diversity",
    "plot_pca_diversity",
    "saturation_point",
    "curve_decrease_rate",
    "main",
]

METHOD_SAPG = "sapg"
METHOD_PPO = "ppo"
METHOD_RANDOM = "random"

#: Default number of principal components evaluated (Figure 7 x-axis).
DEFAULT_K_VALUES: Tuple[int, ...] = tuple(range(1, 33))

_LOG_EPS = 1e-12


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _as_matrix(states: Any) -> Any:
    """Return ``states`` as a 2-D float array (numpy if available, else torch)."""
    if _np is not None:
        if HAS_TORCH and isinstance(states, torch.Tensor):
            arr = states.detach().to("cpu", dtype=torch.float64).numpy()
        elif isinstance(states, _np.ndarray):
            arr = states
        else:
            arr = _np.asarray(states)
        arr = _np.asarray(arr, dtype=_np.float64)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if arr.ndim > 2:
            arr = arr.reshape(-1, arr.shape[-1])
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    if not HAS_TORCH:  # pragma: no cover - defensive
        raise RuntimeError("Either numpy or torch is required for the PCA diversity metric")
    if not isinstance(states, torch.Tensor):
        states = torch.as_tensor(states)
    out = states.detach().to("cpu", dtype=torch.float64)
    if out.dim() == 3 and out.shape[-1] == 1:
        out = out.squeeze(-1)
    if out.dim() > 2:
        out = out.reshape(-1, out.shape[-1])
    if out.dim() == 1:
        out = out.unsqueeze(-1)
    return out


def _numpy_available() -> bool:
    return _np is not None


def _to_list(values: Any) -> List[float]:
    if HAS_TORCH and isinstance(values, torch.Tensor):
        return [float(v) for v in values.detach().to("cpu", dtype=torch.float64).reshape(-1)]
    return [float(v) for v in values]


def _get(obj: Any, keys: Sequence[str], default: Any = None) -> Any:
    """Duck-typed lookup across dicts, buffers and plain objects."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        for key in keys:
            if key in obj:
                return obj[key]
        return default
    getter = getattr(obj, "get", None)
    if callable(getter):
        for key in keys:
            try:
                value = getter(key)
            except Exception:
                value = None
            if value is not None:
                return value
    for key in keys:
        if hasattr(obj, key):
            return getattr(obj, key)
    for container in ("data", "storage", "tensors", "fields", "buffers"):
        inner = getattr(obj, container, None)
        if inner is None or inner is obj:
            continue
        found = _get(inner, keys, None)
        if found is not None:
            return found
    return default


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass
class PCAConfig:
    """Configuration of the PCA diversity evaluation (Sec. 6.4)."""

    #: Number of leading components evaluated (Figure 7 x-axis).
    k_values: Sequence[int] = DEFAULT_K_VALUES
    #: Error convention: ``"mse" | "sum" | "relative"``.
    error_type: str = "mse"
    #: z-score each state dimension before the SVD.
    standardize: bool = True
    #: Mean-centre the states before the SVD (PCA requirement).
    center: bool = True
    #: Maximum number of samples used (subsampled uniformly when exceeded).
    max_samples: Optional[int] = 400_000
    #: Optional random seed for subsampling reproducibility.
    seed: int = 0
    #: Backend: ``"auto" | "sklearn" | "numpy" | "torch"``.
    backend: str = "auto"

    def to_dict(self) -> Dict[str, Any]:
        data = dict(self.__dict__)
        data["k_values"] = list(self.k_values)
        return data

    @classmethod
    def from_any(cls, config: Any = None, **overrides: Any) -> "PCAConfig":
        values: Dict[str, Any] = {}
        if isinstance(config, PCAConfig):
            values = config.to_dict()
        elif isinstance(config, dict):
            values = dict(config)
        elif config is not None:
            for name in ("k_values", "error_type", "standardize", "center", "max_samples", "seed", "backend"):
                if hasattr(config, name):
                    values[name] = getattr(config, name)
        values.update({k: v for k, v in overrides.items() if v is not None})
        allowed = set(cls.__dataclass_fields__)
        values = {k: v for k, v in values.items() if k in allowed}
        return cls(**values)


@dataclass
class PCAResult:
    """Reconstruction-error curve of one method (one Figure-7 line)."""

    name: str
    k_values: List[int] = field(default_factory=list)
    errors: List[float] = field(default_factory=list)
    error_type: str = "mse"
    state_dim: int = 0
    num_samples: int = 0
    explained_variance: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- convenience -------------------------------------------------------
    def error_at(self, k: int) -> float:
        try:
            index = self.k_values.index(int(k))
        except ValueError:
            raise KeyError(f"k={k} was not evaluated (available: {self.k_values})")
        return self.errors[index]

    @property
    def normalized_errors(self) -> List[float]:
        """Errors scaled so that the largest evaluated ``k`` maps to 1.0."""
        if not self.errors:
            return []
        reference = self.errors[-1]
        if abs(reference) < _LOG_EPS:
            return list(self.errors)
        return [e / reference for e in self.errors]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "k_values": list(self.k_values),
            "errors": list(self.errors),
            "error_type": self.error_type,
            "state_dim": int(self.state_dim),
            "num_samples": int(self.num_samples),
            "explained_variance": list(self.explained_variance),
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# core metric
# ---------------------------------------------------------------------------
class PCADiversity:
    """Reconstruction error of a state batch under a ``k``-component PCA.

    Parameters
    ----------
    states:
        ``[num_samples, state_dim]`` batch of visited environment states
        (torch tensor, numpy array or nested sequence).
    config:
        :class:`PCAConfig` or dict of overrides.
    """

    def __init__(self, states: Any, config: Optional[Any] = None, **overrides: Any):
        self.config = PCAConfig.from_any(config, **overrides)
        matrix = _as_matrix(states)
        self.backend = self._resolve_backend(matrix)
        self._sklearn_pca = None

        if self.config.standardize:
            self.mean = self._mean(matrix)
            self.std = self._std(matrix)
            matrix = (matrix - self.mean) / (self.std + _LOG_EPS)
        else:
            self.mean = None
            self.std = None

        if self.config.center and not self.config.standardize:
            self.mean = self._mean(matrix)
            matrix = matrix - self.mean

        self.states = self._subsample(matrix)
        self.num_samples = int(self.states.shape[0])
        self.state_dim = int(self.states.shape[1])

        if self.backend == "sklearn":
            from sklearn.decomposition import PCA  # type: ignore

            self._sklearn_pca = PCA(
                n_components=min(self.state_dim, self.num_samples),
                svd_solver="full" if self.num_samples <= 5000 else "randomized",
                random_state=self.config.seed,
            )
            self._sklearn_pca.fit(
                self.states.numpy() if HAS_TORCH and isinstance(self.states, torch.Tensor) else self.states
            )
            self.singular_values = _to_list(self._sklearn_pca.singular_values_)
            self.explained_variance_ratio_ = _to_list(self._sklearn_pca.explained_variance_ratio_)
            self.components_ = self._sklearn_pca.components_
        else:
            self._decompose()

    # -- construction helpers ---------------------------------------------
    def _resolve_backend(self, matrix: Any) -> str:
        requested = str(self.config.backend).lower()
        if requested in ("sklearn", "numpy", "torch"):
            return requested
        # auto
        if _numpy_available():
            if self._sklearn_available():
                return "sklearn"
            return "numpy"
        return "torch"

    @staticmethod
    def _sklearn_available() -> bool:
        try:  # pragma: no cover - environment dependent
            import sklearn.decomposition  # noqa: F401

            return True
        except Exception:
            return False

    def _subsample(self, matrix: Any) -> Any:
        limit = self.config.max_samples
        if limit is None:
            return matrix
        n = int(matrix.shape[0])
        if n <= int(limit):
            return matrix
        if HAS_TORCH and isinstance(matrix, torch.Tensor):
            generator = torch.Generator().manual_seed(int(self.config.seed))
            idx = torch.randperm(n, generator=generator)[: int(limit)]
            return matrix[idx]
        if _numpy_available():
            rng = _np.random.RandomState(int(self.config.seed))
            idx = rng.permutation(n)[: int(limit)]
            return matrix[idx]
        return matrix  # pragma: no cover

    # -- small helpers -----------------------------------------------------
    def _mean(self, matrix: Any) -> Any:
        if HAS_TORCH and isinstance(matrix, torch.Tensor):
            return matrix.mean(dim=0, keepdim=True)
        return matrix.mean(axis=0, keepdims=True)

    def _std(self, matrix: Any) -> Any:
        if HAS_TORCH and isinstance(matrix, torch.Tensor):
            return matrix.std(dim=0, keepdim=True, unbiased=False)
        return matrix.std(axis=0, keepdims=True)

    def _decompose(self) -> None:
        if HAS_TORCH and isinstance(self.states, torch.Tensor):
            # economy SVD; the matrix is already centred above.
            _u, s, vh = torch.linalg.svd(self.states, full_matrices=False)
            self.components_ = vh
            self.singular_values = _to_list(s)
            total = float((s ** 2).sum().clamp_min(_LOG_EPS))
            self.explained_variance_ratio_ = _to_list((s ** 2) / total)
        else:
            _u, s, vh = _np.linalg.svd(self.states, full_matrices=False)
            self.components_ = vh
            self.singular_values = _to_list(s)
            total = float(_np.sum(s ** 2)) or _LOG_EPS
            self.explained_variance_ratio_ = _to_list((s ** 2) / total)

    # -- metric ------------------------------------------------------------
    def curve(self, k_values: Optional[Sequence[int]] = None) -> PCAResult:
        """Reconstruction error as a function of the number of components."""
        ks = list(k_values if k_values is not None else self.config.k_values)
        ks = sorted({int(k) for k in ks if int(k) >= 0})
        errors = [self.reconstruction_error(k) for k in ks]
        return PCAResult(
            name="pca",
            k_values=ks,
            errors=errors,
            error_type=self.config.error_type,
            state_dim=self.state_dim,
            num_samples=self.num_samples,
            explained_variance=list(self.explained_variance_ratio_),
        )

    def reconstruction_error(self, k: int) -> float:
        """Reconstruction error using the ``k`` most significant components."""
        k = int(k)
        if k < 0:
            raise ValueError("k must be non-negative")
        if k == 0:
            return self._residual_error(self.states)
        k = min(k, self.state_dim)
        components = self._top_components(k)
        recon = self._project(self.states, components)
        return self._residual_error(self.states, recon)

    def sequence(self, k_values: Optional[Sequence[int]] = None) -> Tuple[List[int], List[float]]:
        result = self.curve(k_values)
        return result.k_values, result.errors

    # -- projection --------------------------------------------------------
    def _top_components(self, k: int) -> Any:
        if self._sklearn_pca is not None:
            return self._sklearn_pca.components_[:k]
        return self.components_[:k]

    def _project(self, matrix: Any, components: Any) -> Any:
        if HAS_TORCH and isinstance(matrix, torch.Tensor):
            comp = components
            if not isinstance(comp, torch.Tensor):
                comp = torch.as_tensor(comp, dtype=matrix.dtype)
            return (matrix @ comp.transpose(-1, -2)) @ comp
        return (matrix @ components.T) @ components

    def _residual_error(self, matrix: Any, recon: Optional[Any] = None) -> float:
        residual = matrix if recon is None else matrix - recon
        etype = str(self.config.error_type).lower()
        if HAS_TORCH and isinstance(residual, torch.Tensor):
            sq = residual ** 2
            if etype in ("mse", "mean", "mean_squared_error"):
                return float(sq.mean())
            if etype in ("sum", "sse", "mse_per_sample", "per_sample"):
                return float(sq.sum(dim=-1).mean())
            if etype in ("relative", "fraction", "variance"):
                base = float((matrix - matrix.mean(dim=0, keepdim=True)).pow(2).sum())
                return float(sq.sum()) / (base + _LOG_EPS)
            raise ValueError(f"unknown error_type '{self.config.error_type}'")
        sq = _np.asarray(residual, dtype=_np.float64) ** 2
        if etype in ("mse", "mean", "mean_squared_error"):
            return float(sq.mean())
        if etype in ("sum", "sse", "mse_per_sample", "per_sample"):
            return float(sq.sum(axis=-1).mean())
        if etype in ("relative", "fraction", "variance"):
            mean = _np.asarray(matrix, dtype=_np.float64).mean(axis=0, keepdims=True)
            base = float(((_np.asarray(matrix, dtype=_np.float64) - mean) ** 2).sum())
            return float(sq.sum()) / (base + _LOG_EPS)
        raise ValueError(f"unknown error_type '{self.config.error_type}'")

    # -- diagnostics -------------------------------------------------------
    @property
    def explained_variance_ratio(self) -> List[float]:
        return list(self.explained_variance_ratio_)

    def saturation_point(self, tolerance: float = 1e-3) -> int:
        """Smallest ``k`` whose residual error drops below ``tolerance``."""
        for k in range(1, self.state_dim + 1):
            if self.reconstruction_error(k) < tolerance:
                return k
        return self.state_dim

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"PCADiversity(state_dim={self.state_dim}, num_samples={self.num_samples}, "
            f"backend={self.backend}, error_type={self.config.error_type})"
        )


# ---------------------------------------------------------------------------
# functional API
# ---------------------------------------------------------------------------
def pca_components(states: Any, k: int, config: Optional[Any] = None, **overrides: Any) -> Any:
    """Return the ``k`` leading principal directions of ``states``."""
    return PCADiversity(states, config, **overrides)._top_components(int(k))


def explained_variance_ratio(states: Any, config: Optional[Any] = None, **overrides: Any) -> List[float]:
    """Explained-variance ratio of each principal component (descending)."""
    return PCADiversity(states, config, **overrides).explained_variance_ratio


def pca_reconstruction_error(states: Any, k: int, config: Optional[Any] = None, **overrides: Any) -> float:
    """Reconstruction error of ``states`` using ``k`` principal components."""
    return PCADiversity(states, config, **overrides).reconstruction_error(k)


def pca_reconstruction_curve(
    states: Any,
    k_values: Optional[Sequence[int]] = None,
    config: Optional[Any] = None,
    name: str = "policy",
    **overrides: Any,
) -> PCAResult:
    """Full Figure-7 style curve: error vs number of principal components."""
    metric = PCADiversity(states, config, **overrides)
    result = metric.curve(k_values)
    result.name = name
    return result


def reconstruction_error_per_dimension(
    states: Any, config: Optional[Any] = None, **overrides: Any
) -> List[float]:
    """Error curve evaluated for every ``k = 0 .. state_dim`` (diagnostic)."""
    metric = PCADiversity(states, config, **overrides)
    return [metric.reconstruction_error(k) for k in range(0, metric.state_dim + 1)]


# ---------------------------------------------------------------------------
# state collection helpers
# ---------------------------------------------------------------------------
def states_from_buffer(
    buffer: Any,
    key: str = "obs",
    max_samples: Optional[int] = None,
    seed: int = 0,
) -> Any:
    """Extract a flat ``[num_samples, obs_dim]`` state batch from a buffer.

    Accepts :class:`~sapg.buffers.rollout_buffer.RolloutBuffer` instances,
    dict-like batches, or objects exposing ``.data`` / ``.storage``.  Handles
    both fixed-horizon (``[T, N, obs_dim]``) and flat (``[num_samples, obs_dim]``)
    layouts.
    """
    keys = [key]
    if key == "obs":
        keys += ["observations", "observation", "states", "s"]
    elif key == "next_obs":
        keys += ["obs_next", "next_observations"]
    states = _get(buffer, keys, None)
    if states is None:
        raise KeyError(f"could not find observations under any of {keys} in {type(buffer).__name__}")
    matrix = _as_matrix(states)
    if matrix.shape[0] == 0:
        return matrix
    limit = max_samples
    if limit is not None and matrix.shape[0] > int(limit):
        if _numpy_available():
            rng = _np.random.RandomState(int(seed))
            idx = rng.permutation(int(matrix.shape[0]))[: int(limit)]
            matrix = matrix[idx]
        elif HAS_TORCH and isinstance(matrix, torch.Tensor):
            generator = torch.Generator().manual_seed(int(seed))
            idx = torch.randperm(int(matrix.shape[0]), generator=generator)[: int(limit)]
            matrix = matrix[idx]
    return matrix


def collect_states(
    policy: Any,
    env: Any,
    num_steps: int = 16,
    obs: Any = None,
    deterministic: bool = False,
    max_samples: Optional[int] = None,
    phi: Any = None,
    policy_index: Optional[int] = None,
) -> Any:
    """Roll ``policy`` in ``env`` for ``num_steps`` and return the visited states.

    Mirrors :func:`sapg.algorithms.rollout.collect_on_policy` but keeps only the
    observations, which is all the diversity metrics need.
    """
    frames: List[Any] = []
    hidden_state = None
    masks = None
    if obs is None:
        obs = env.reset()
        if hasattr(policy, "init_hidden"):
            batch = int(obs.shape[0]) if hasattr(obs, "shape") and len(getattr(obs, "shape", ())) > 1 else 1
            try:
                hidden_state = policy.init_hidden(batch, obs.device if hasattr(obs, "device") else None)
            except Exception:  # pragma: no cover - optional recurrent policies
                hidden_state = None

    for _ in range(int(num_steps)):
        frames.append(obs)
        kwargs: Dict[str, Any] = {"deterministic": deterministic}
        if phi is not None:
            kwargs["phi"] = phi
        if policy_index is not None:
            kwargs["policy_index"] = policy_index
        if hidden_state is not None:
            kwargs["hidden_state"] = hidden_state
        if masks is not None:
            kwargs["masks"] = masks
        try:
            out = policy.act(obs, **kwargs)
        except TypeError:
            out = policy.act(obs)
        actions = out["actions"] if isinstance(out, dict) else out
        step_out = env.step(actions)
        if isinstance(step_out, tuple) and len(step_out) >= 4:
            obs, _reward, dones, _info = step_out[:4]
        else:  # pragma: no cover - defensive
            obs = step_out[0]
            dones = None
        if isinstance(out, dict) and out.get("hidden_state") is not None:
            hidden_state = out["hidden_state"]
        masks = dones

    first = frames[0]
    if HAS_TORCH and isinstance(first, torch.Tensor):
        return torch.cat([f.reshape(-1, f.shape[-1]) for f in frames], dim=0)
    if _numpy_available():
        return _np.concatenate([_as_matrix(f) for f in frames], axis=0)
    return first  # pragma: no cover


# ---------------------------------------------------------------------------
# multi-method comparison + Figure 7
# ---------------------------------------------------------------------------
def compare_pca_diversity(
    state_batches: Dict[str, Any],
    k_values: Optional[Sequence[int]] = None,
    config: Optional[Any] = None,
    **overrides: Any,
) -> Dict[str, PCAResult]:
    """Compute the PCA error curve for each method (SAPG / PPO / random ...)."""
    results: Dict[str, PCAResult] = {}
    for name, states in state_batches.items():
        result = pca_reconstruction_curve(states, k_values=k_values, config=config, name=name, **overrides)
        result.metadata.setdefault("method", name)
        results[name] = result
    return results


def curve_decrease_rate(result: PCAResult) -> float:
    """Slope of the (log) error curve: smaller magnitude == slower decrease.

    The paper's claim is that SAPG shows the *slowest* decrease of
    reconstruction error with increasing ``k`` (Sec. 6.4), so this scalar is a
    compact summary of Figure 7 (more negative => faster compression).
    """
    if len(result.errors) < 2:
        return 0.0
    xs = [float(k) for k in result.k_values]
    ys = [math.log(max(e, _LOG_EPS)) for e in result.errors]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom <= _LOG_EPS:
        return 0.0
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom


def saturation_point(result: PCAResult, tolerance: float = 1e-3) -> int:
    """Smallest ``k`` in ``result`` whose error is below ``tolerance``."""
    for k, err in zip(result.k_values, result.errors):
        if err < tolerance:
            return int(k)
    return int(result.k_values[-1]) if result.k_values else 0


def plot_pca_diversity(
    results: Union[Dict[str, PCAResult], Iterable[PCAResult]],
    path: Optional[str] = None,
    title: str = "PCA reconstruction error vs number of components",
    log_y: bool = True,
    normalize: bool = False,
    show: bool = False,
) -> Any:
    """Render Figure 7 (error vs ``k``, one line per method).

    Returns the matplotlib figure (or ``None`` when matplotlib is unavailable).
    """
    items = list(results.items()) if isinstance(results, dict) else [(r.name, r) for r in results]
    if not HAS_MATPLOTLIB:  # pragma: no cover - environment dependent
        return None

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    for name, result in items:
        ys = result.normalized_errors if normalize else result.errors
        ax.plot(result.k_values, ys, marker="o", markersize=3, label=str(name))
    ax.set_xlabel("number of principal components k")
    ax.set_ylabel("reconstruction error")
    ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        fig.savefig(path, dpi=150)
    if show:  # pragma: no cover - interactive only
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - convenience
    """Standalone smoke run on synthetic state batches."""
    import argparse

    parser = argparse.ArgumentParser(description="PCA state-space diversity metric (Fig. 7)")
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--state-dim", type=int, default=63)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not HAS_TORCH:
        print("torch is required for the smoke run")
        return 1
    generator = torch.Generator().manual_seed(0)
    sapg_states = torch.randn(args.samples, args.state_dim, generator=generator)
    ppo_states = torch.randn(args.samples, args.state_dim, generator=generator) * 0.5
    ppo_states[:, 8:] = 0.0  # artificially low-dimensional (PPO-like)
    results = compare_pca_diversity({"sapg": sapg_states, "ppo": ppo_states})
    for name, result in results.items():
        print(f"{name}: log-error slope = {curve_decrease_rate(result):.4f}")
    if args.out:
        plot_pca_diversity(results, args.out)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
