"""PosteriorDB targets for the hierarchical Bayesian model experiments (Section 5.2).

The paper considers three targets from posteriordb (Magnusson et al., 2022), a
database of Stan models with HMC reference samples:

    * ``ark``                  (D = 7,  nearly Gaussian)
    * ``gp-pois-regr``         (D = 13, non-Gaussian)
    * ``eight-schools-centered``(D = 10, non-Gaussian hierarchical model)

The posterior density is

    p(z | {x_n}) ∝ p(z) p({x_n} | z)                      (eq. 5.2.1)

and the algorithms only need the score  ∇_z log p(z | {x_n}).

Gradients are obtained through the Python interface of BridgeStan
(https://github.com/roualdes/bridgestan).  Both ``bridgestan`` and the
``posteriordb`` reference data are *optional* dependencies: when they are not
available this module transparently falls back to

    1. a locally cached Stan model + JSON data / HMC samples, if present in
       ``data/posteriordb`` next to the repository, and
    2. a Gaussian surrogate target with the paper's dimension and HMC-style
       reference moments (so that the experiment plumbing, metrics and curves
       remain fully exercisable, runnable and self-consistent on any machine).

The surrogate is clearly flagged (``is_surrogate``) and named
``"<model>-surrogate"`` so downstream reporting never confuses the two.

Reference moments (posterior mean and SD) are always available through
:meth:`PosteriorDBTarget.reference_moments`, which is what the relative mean /
SD error metrics of Section 5.2 / Appendix E.5 consume:

    relative mean error = ||(μ - μ̂) / σ||_2,
    relative SD   error = ||(σ - σ̂) / σ||_2.

Source: §5.2, §E.5 (addendum: BridgeStan).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from ..bam.matrix_equations import array_namespace, ensure_spd, inverse_spd, symmetrize
from ..bam.vi_base import standard_normal

__all__ = [
    "PAPER_POSTERIORDB_MODELS",
    "PAPER_POSTERIORDB_NAMES",
    "PosteriorDBTarget",
    "StanTarget",
    "SurrogateGaussianTarget",
    "get_target",
    "load_target",
    "make_target",
    "posteriordb_targets",
    "posterior_moments",
    "reference_moments",
    "bridgestan_available",
    "posteriordb_available",
    "BRIDGESTAN_AVAILABLE",
    "POSTERIORDB_AVAILABLE",
    "PAPER_POSTERIORDB_BATCH_SIZES",
    "PAPER_POSTERIORDB_N_RUNS",
]

# ---------------------------------------------------------------------------
# Paper constants
# ---------------------------------------------------------------------------

#: Model name -> dimension, in the order they appear in the paper.
PAPER_POSTERIORDB_MODELS: Dict[str, int] = {
    "ark": 7,
    "gp-pois-regr": 13,
    "eight-schools-centered": 10,
}

#: The three model names in paper order.
PAPER_POSTERIORDB_NAMES: Tuple[str, ...] = tuple(PAPER_POSTERIORDB_MODELS)

#: Batch sizes used in Figure 5.3 / E.6.
PAPER_POSTERIORDB_BATCH_SIZES: Tuple[int, ...] = (8, 32)

#: Number of independent runs (Section E.5 protocol).
PAPER_POSTERIORDB_N_RUNS: int = 5

#: Initialisation used for every algorithm: μ_0 ~ Uniform[0, 0.1], Σ_0 = I.
PAPER_INIT_MEAN_SCALE: float = 0.1

#: Directory where locally cached Stan models / reference samples are looked
#: for.  Relative paths are resolved from this file (repo root) and from CWD.
_CACHE_ENV = "BAM_POSTERIORDB_CACHE"
_DEFAULT_CACHE_DIRS = (
    os.path.join("bam_repro", "data", "posteriordb"),
    os.path.join("data", "posteriordb"),
)


# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------


def bridgestan_available() -> bool:
    """Return ``True`` when the ``bridgestan`` Python package can be imported."""
    try:  # pragma: no cover - depends on environment
        import bridgestan  # noqa: F401

        return True
    except Exception:  # pragma: no cover
        return False


def posteriordb_available() -> bool:
    """Return ``True`` for the legacy R/python ``posteriordb`` bindings."""
    try:  # pragma: no cover - depends on environment
        import posteriordb  # noqa: F401

        return True
    except Exception:  # pragma: no cover
        return False


BRIDGESTAN_AVAILABLE = bridgestan_available()
POSTERIORDB_AVAILABLE = posteriordb_available()


def _cache_dirs() -> List[str]:
    dirs: List[str] = []
    env = os.environ.get(_CACHE_ENV)
    if env:
        dirs.extend(part for part in env.split(os.pathsep) if part)
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))  # repo root (contains bam_repro/)
    for rel in _DEFAULT_CACHE_DIRS:
        dirs.append(os.path.join(root, rel))
        dirs.append(os.path.join(here, os.path.basename(rel)))
        dirs.append(rel)
    return dirs


def _find_cache_file(name: str) -> Optional[str]:
    for directory in _cache_dirs():
        path = os.path.join(directory, name)
        if os.path.exists(path):
            return path
    return None


# ---------------------------------------------------------------------------
# Surrogate Gaussian target (fallback)
# ---------------------------------------------------------------------------


@dataclass
class SurrogateGaussianTarget:
    """Full-covariance Gaussian surrogate used when Stan is unavailable.

    The interface mirrors the other targets in :mod:`bam_repro.targets`
    (``mean``, ``cov``, ``score``/``grad_log_prob``, ``log_prob``, ``sample``,
    ``dim``) so that BaM and every baseline can consume it unchanged.
    """

    mean: np.ndarray
    cov: np.ndarray
    name: str = "gaussian"
    is_surrogate: bool = True
    _precision: Optional[np.ndarray] = field(default=None, repr=False)
    _cholesky: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64).reshape(-1)
        cov = np.asarray(self.cov, dtype=np.float64)
        cov = ensure_spd(symmetrize(cov), jitter=1e-10, min_eig=1e-10)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "cov", cov)

    # -- basic properties -------------------------------------------------
    @property
    def dim(self) -> int:
        return int(self.mean.shape[0])

    @property
    def xp(self):
        return np

    @property
    def precision(self) -> np.ndarray:
        if self._precision is None:
            object.__setattr__(self, "_precision", inverse_spd(self.cov, jitter=1e-12))
        return self._precision

    @property
    def cholesky(self) -> np.ndarray:
        if self._cholesky is None:
            cov = ensure_spd(self.cov, jitter=1e-12, min_eig=1e-12)
            object.__setattr__(self, "_cholesky", np.linalg.cholesky(cov))
        return self._cholesky

    # -- density / score --------------------------------------------------
    def log_prob(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        single = z.ndim == 1
        z = np.atleast_2d(z)
        L = self.cholesky
        d = z - self.mean
        sol = np.linalg.solve(L, d.T).T
        quad = np.sum(sol**2, axis=1)
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        logp = -0.5 * (quad + log_det + self.dim * np.log(2.0 * np.pi))
        return logp[0] if single else logp

    def score(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        single = z.ndim == 1
        z = np.atleast_2d(z)
        g = -(z - self.mean) @ self.precision.T
        return g[0] if single else g

    #: alias matching the target API used elsewhere in the repo
    grad_log_prob = score

    def sample(
        self,
        n: int,
        rng: Optional[np.random.Generator] = None,
        key: Any = None,
        dtype: Any = None,
    ) -> np.ndarray:
        eps = standard_normal((int(n), self.dim), xp=np, rng=rng, key=key)
        return self.mean[None, :] + eps @ self.cholesky.T

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dim": self.dim,
            "mean": self.mean.copy(),
            "cov": self.cov.copy(),
            "is_surrogate": True,
        }


def _surrogate_from_moments(
    model: str,
    mean: np.ndarray,
    sd: np.ndarray,
    corr: Optional[np.ndarray] = None,
) -> SurrogateGaussianTarget:
    """Build a Gaussian surrogate matching given reference moments."""
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    sd = np.asarray(sd, dtype=np.float64).reshape(-1)
    dim = mean.shape[0]
    if corr is None:
        corr = np.eye(dim)
    corr = np.asarray(corr, dtype=np.float64)
    cov = corr * np.outer(sd, sd)
    return SurrogateGaussianTarget(mean=mean, cov=cov, name=f"{model}-surrogate")


# ---------------------------------------------------------------------------
# BridgeStan target
# ---------------------------------------------------------------------------


class StanTarget:
    """Score of a Stan model posterior via BridgeStan.

    Parameters
    ----------
    stan_file, data_file:
        Paths handed to :class:`bridgestan.StanModel`.
    name:
        Model label, e.g. ``"ark"``.
    reference_samples:
        Optional ``(N, D)`` array of HMC reference draws.  When absent, draws
        are generated by sampling from a Laplace/Gaussian surrogate fitted to
        the returned moments (clearly reported through ``is_surrogate``).
    """

    is_surrogate: bool = False

    def __init__(
        self,
        stan_file: str,
        data_file: Optional[str] = None,
        name: str = "stan",
        model: Any = None,
        reference_samples: Optional[np.ndarray] = None,
        dim: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        self.name = name
        self.stan_file = stan_file
        self.data_file = data_file
        self.seed = seed
        self._model = model
        if self._model is None:  # pragma: no cover - requires bridgestan
            import bridgestan

            self._model = bridgestan.StanModel(stan_file, data_file)

        # Dimension: prefer the explicitly given value, then the model API.
        if dim is None:
            dim = getattr(self._model, "param_num", None)
            if dim is None:
                dim = len(self._initial_point())
        self._dim = int(dim)

        self._reference_samples = (
            None
            if reference_samples is None
            else np.asarray(reference_samples, dtype=np.float64).reshape(-1, self._dim)
        )
        self._reference_moments: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._initial: Optional[np.ndarray] = None

    # -- BridgeStan plumbing ---------------------------------------------
    def _initial_point(self) -> np.ndarray:
        if self._initial is None:
            try:
                q0 = np.asarray(self._model.param_unc_init(), dtype=np.float64)
            except Exception:  # pragma: no cover - API differences
                q0 = np.zeros(self._dim, dtype=np.float64)
            self._initial = q0.reshape(-1)
        return self._initial

    def log_prob(self, z: np.ndarray) -> np.ndarray:
        """Unconstrained ``log p(z)`` (up to an additive constant)."""
        z = np.atleast_2d(np.asarray(z, dtype=np.float64))
        out = np.empty(z.shape[0], dtype=np.float64)
        for i, z_i in enumerate(z):
            try:  # pragma: no cover - requires bridgestan
                lp, _ = self._model.log_density_gradient(
                    np.ascontiguousarray(z_i), propto=False, jacobian=True
                )
                out[i] = float(lp)
            except Exception:  # pragma: no cover
                out[i] = float(self._model.log_density(np.ascontiguousarray(z_i), False))
        return out if np.asarray(z).ndim > 1 else out[0]

    def score(self, z: np.ndarray) -> np.ndarray:
        """Score ∇_z log p(z | {x_n}) from BridgeStan."""
        z = np.asarray(z, dtype=np.float64)
        single = z.ndim == 1
        z2 = np.atleast_2d(z)
        out = np.empty_like(z2, dtype=np.float64)
        for i, z_i in enumerate(z2):
            try:  # pragma: no cover - requires bridgestan
                _, grad = self._model.log_density_gradient(
                    np.ascontiguousarray(z_i), propto=False, jacobian=True
                )
                out[i] = np.asarray(grad, dtype=np.float64).reshape(-1)
            except Exception:  # pragma: no cover
                out[i] = np.asarray(
                    self._model.log_density_gradient(np.ascontiguousarray(z_i))[1],
                    dtype=np.float64,
                ).reshape(-1)
        return out[0] if single else out

    grad_log_prob = score

    # -- properties ------------------------------------------------------
    @property
    def dim(self) -> int:
        return self._dim

    @property
    def xp(self):
        return np

    @property
    def is_surrogate(self) -> bool:  # type: ignore[override]
        return self._reference_samples is None

    @property
    def reference_samples(self) -> np.ndarray:
        if self._reference_samples is None:
            self._reference_samples = self._make_laplace_reference()
        return self._reference_samples

    @property
    def reference_moments(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._reference_moments is None:
            self._reference_moments = posterior_moments(self.reference_samples)
        return self._reference_moments

    def sample(
        self,
        n: int,
        rng: Optional[np.random.Generator] = None,
        key: Any = None,
        dtype: Any = None,
    ) -> np.ndarray:
        """Draw from the Laplace approximation at the posterior mode.

        Used only to generate surrogate reference samples when real HMC draws
        are unavailable; the variational algorithms never call this method.
        """
        mu, Sigma = self._laplace_approximation()
        eps = standard_normal((int(n), self.dim), xp=np, rng=rng, key=key)
        L = np.linalg.cholesky(ensure_spd(Sigma, jitter=1e-10, min_eig=1e-10))
        return mu[None, :] + eps @ L.T

    def _laplace_approximation(self) -> Tuple[np.ndarray, np.ndarray]:
        """Find the mode by gradient ascent (no Hessian required)."""
        z = self._initial_point().copy()
        lr = 1e-2
        for _ in range(2000):
            g = np.asarray(self.score(z), dtype=np.float64).reshape(-1)
            if not np.all(np.isfinite(g)):
                break
            prev = np.linalg.norm(g)
            z = z + lr * g
            new = np.asarray(self.score(z), dtype=np.float64).reshape(-1)
            if not np.all(np.isfinite(new)):
                break
            if np.linalg.norm(new) > prev:
                lr *= 0.5
            if np.linalg.norm(new) < 1e-8:
                break
        # Gaussian/identity fallback covariance: negative inverse of the local
        # finite-difference Hessian approximation of the score's Jacobian.
        H = _finite_difference_jacobian(self.score, z)
        H = -0.5 * (H + H.T)  # negative Hessian is SPD at the mode
        H = ensure_spd(symmetrize(H), jitter=1e-8, min_eig=1e-8)
        return z, inverse_spd(H, jitter=1e-10)

    def _make_laplace_reference(self) -> np.ndarray:  # pragma: no cover - fallback
        rng = np.random.default_rng(self.seed)
        return self.sample(4000, rng=rng)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dim": self.dim,
            "stan_file": self.stan_file,
            "data_file": self.data_file,
            "is_surrogate": self.is_surrogate,
        }


def _finite_difference_jacobian(fn: Callable[[np.ndarray], np.ndarray], z: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Central-difference Jacobian of ``fn`` at ``z`` (D x D)."""
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    d = z.shape[0]
    J = np.empty((d, d), dtype=np.float64)
    for j in range(d):
        step = np.zeros(d)
        step[j] = eps
        fp = np.asarray(fn(z + step), dtype=np.float64).reshape(-1)
        fm = np.asarray(fn(z - step), dtype=np.float64).reshape(-1)
        J[:, j] = (fp - fm) / (2.0 * eps)
    return J


# ---------------------------------------------------------------------------
# Unified posterior-db target
# ---------------------------------------------------------------------------


@dataclass
class PosteriorDBTarget:
    """A Section 5.2 posterior target with score, samples and moments.

    Wraps a *backend* that is either a :class:`StanTarget` (BridgeStan-powered)
    or a :class:`SurrogateGaussianTarget` fallback.  The object exposes the
    target interface used throughout the code base (``score``,
    ``grad_log_prob``, ``log_prob``, ``sample``, ``dim``) plus the HMC reference
    pieces required by the error metrics.
    """

    name: str
    dim: int
    backend: Any
    reference_samples: Optional[np.ndarray] = None
    is_surrogate: bool = False

    def __post_init__(self) -> None:
        if self.dim is None:
            object.__setattr__(self, "dim", int(self.backend.dim))
        self.dim = int(self.dim)
        if self.reference_samples is not None:
            self.reference_samples = np.asarray(self.reference_samples, dtype=np.float64).reshape(
                -1, self.dim
            )
        self._moments_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None

    # -- target interface -------------------------------------------------
    @property
    def xp(self):
        return np

    @property
    def mean(self) -> np.ndarray:  # target-style attribute (surrogate only)
        return getattr(self.backend, "mean")

    @property
    def cov(self) -> np.ndarray:  # target-style attribute (surrogate only)
        return getattr(self.backend, "cov")

    def score(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(self.backend.score(z), dtype=np.float64)

    grad_log_prob = score

    def log_prob(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(self.backend.log_prob(z), dtype=np.float64)

    def sample(
        self,
        n: int,
        rng: Optional[np.random.Generator] = None,
        key: Any = None,
        dtype: Any = None,
    ) -> np.ndarray:
        if self.reference_samples is not None:
            rng = np.random.default_rng(0) if rng is None else rng
            idx = rng.integers(0, self.reference_samples.shape[0], size=int(n))
            return self.reference_samples[idx]
        return np.asarray(self.backend.sample(n, rng=rng, key=key))

    # -- reference quantities (Section 5.2 / E.5) -------------------------
    def reference_moments(self, n_draws: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Posterior mean μ and SD σ estimated from HMC reference samples."""
        samples = self.reference_samples
        if samples is None:
            samples = getattr(self.backend, "reference_samples", None)
        if samples is None:  # pure surrogate built from given moments
            return self.mean.copy(), np.sqrt(np.diag(self.cov))
        samples = np.asarray(samples, dtype=np.float64).reshape(-1, self.dim)
        if n_draws is not None and n_draws < samples.shape[0]:
            samples = samples[: int(n_draws)]
        return posterior_moments(samples)

    def validation_log_prob(self, z: np.ndarray) -> np.ndarray:
        """Alias used by some baselines for the (unnormalised) log density."""
        return self.log_prob(z)

    # -- convenience -------------------------------------------------------
    def initial_state(
        self,
        n: int = 1,
        rng: Optional[np.random.Generator] = None,
        mean_scale: float = PAPER_INIT_MEAN_SCALE,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Paper initialisation μ_0 ~ Uniform[0, 0.1], Σ_0 = I (Section E.5)."""
        if rng is None:
            rng = np.random.default_rng(0)
        mus = rng.uniform(0.0, mean_scale, size=(int(n), self.dim))
        Sigmas = np.tile(np.eye(self.dim), (int(n), 1, 1))
        return mus, Sigmas, np.arange(int(n))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dim": self.dim,
            "is_surrogate": bool(self.is_surrogate),
            "n_reference": (
                0 if self.reference_samples is None else int(self.reference_samples.shape[0])
            ),
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"PosteriorDBTarget(name={self.name!r}, dim={self.dim}, "
            f"surrogate={self.is_surrogate})"
        )


# ---------------------------------------------------------------------------
# Reference moment utilities (also used by metrics/posteriordb_metrics.py)
# ---------------------------------------------------------------------------


def posterior_moments(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Posterior mean and SD from ``(N, D)`` (or ``(N,)``) samples."""
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim == 1:
        samples = samples[:, None]
    mean = np.mean(samples, axis=0)
    sd = np.std(samples, axis=0, ddof=1 if samples.shape[0] > 1 else 0)
    return mean, sd


def reference_moments(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Alias of :func:`posterior_moments` for readability at call sites."""
    return posterior_moments(samples)


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------


def _surrogate_defaults(model: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Documented default reference moments for the surrogate fallback.

    These are *not* the paper's numbers (posteriordb HMC draws are required for
    those); they are fixed, deterministic, reproducible stand-in moments with
    the correct dimension so the full experiment pipeline runs.  The means are
    sampled from Uniform[0, 0.1] and the SDs from Uniform[0.2, 1.5], both with a
    fixed seed, and a mildly correlated covariance is used.
    """
    dim = PAPER_POSTERIORDB_MODELS[model]
    rng = np.random.default_rng(abs(hash(model)) % (2**32))
    mean = rng.uniform(0.0, PAPER_INIT_MEAN_SCALE, size=dim)
    sd = rng.uniform(0.2, 1.5, size=dim)
    A = rng.normal(size=(dim, dim))
    corr = A @ A.T
    corr = corr / np.sqrt(np.outer(np.diag(corr), np.diag(corr)))
    return mean, sd, corr


def _load_reference_samples(name: str) -> Optional[np.ndarray]:
    """Load locally cached HMC (or MCMC) reference draws if available."""
    candidates = [
        f"{name}.reference_samples.npy",
        f"{name}.samples.npy",
        os.path.join(name, "reference_samples.npy"),
        os.path.join(name, "samples.npy"),
        f"{name}.json",
    ]
    for cname in candidates:
        path = _find_cache_file(cname)
        if path is None:
            continue
        try:
            if path.endswith(".npy"):
                arr = np.load(path)
            else:
                with open(path, "r") as fh:
                    blob = json.load(fh)
                arr = np.asarray(
                    blob.get("samples", blob.get("reference_samples", blob)), dtype=np.float64
                )
            arr = np.asarray(arr, dtype=np.float64)
            return arr.reshape(-1, arr.shape[-1])
        except Exception:
            continue
    return None


def _load_stan_paths(name: str) -> Optional[Tuple[str, Optional[str]]]:
    """Locate a cached ``<name>.stan`` + ``<name>.data.json`` pair."""
    stan = _find_cache_file(f"{name}.stan")
    if stan is None:
        stan = _find_cache_file(os.path.join(name, f"{name}.stan"))
    if stan is None:
        return None
    data = _find_cache_file(f"{name}.data.json")
    if data is None:
        data = _find_cache_file(os.path.join(name, f"{name}.data.json"))
    return stan, data


def load_posteriordb_target(
    name: str,
    reference_samples: Optional[np.ndarray] = None,
    dim: Optional[int] = None,
    seed: int = 0,
    allow_surrogate: bool = True,
) -> PosteriorDBTarget:
    """Load a Section 5.2 target by name.

    Resolution order: cached Stan model + BridgeStan → surrogate Gaussian with
    cached/fixed reference moments.  Passing ``allow_surrogate=False`` raises
    instead of falling back.
    """
    if dim is None:
        dim = PAPER_POSTERIORDB_MODELS.get(name)
    if reference_samples is not None:
        reference_samples = np.asarray(reference_samples, dtype=np.float64)
        reference_samples = reference_samples.reshape(-1, reference_samples.shape[-1])
        if dim is None:
            dim = reference_samples.shape[1]

    # 1) BridgeStan path.
    if BRIDGESTAN_AVAILABLE:
        paths = _load_stan_paths(name)
        if paths is not None:
            stan_file, data_file = paths
            try:  # pragma: no cover - requires bridgestan + stan files
                if reference_samples is None:
                    reference_samples = _load_reference_samples(name)
                backend = StanTarget(
                    stan_file=stan_file,
                    data_file=data_file,
                    name=name,
                    reference_samples=reference_samples,
                    dim=dim,
                    seed=seed,
                )
                return PosteriorDBTarget(
                    name=name,
                    dim=int(backend.dim),
                    backend=backend,
                    reference_samples=reference_samples,
                    is_surrogate=reference_samples is None,
                )
            except Exception:
                pass

    # 2) Cached reference samples without BridgeStan: use a Gaussian surrogate
    #    with those moments (score is exact for the surrogate).
    if reference_samples is None:
        reference_samples = _load_reference_samples(name)

    if reference_samples is not None:
        mu, sd = posterior_moments(reference_samples)
        backend = _surrogate_from_moments(name, mu, sd)
        return PosteriorDBTarget(
            name=name,
            dim=int(backend.dim),
            backend=backend,
            reference_samples=reference_samples,
            is_surrogate=True,
        )

    if not allow_surrogate:
        raise RuntimeError(
            f"Could not load posteriordb target {name!r}: BridgeStan and cached "
            f"Stan models / reference samples are unavailable."
        )

    # 3) Fully synthetic surrogate.
    mean, sd, corr = _surrogate_defaults(name)
    backend = _surrogate_from_moments(name, mean, sd, corr)
    rng = np.random.default_rng(seed)
    samples = backend.sample(4000, rng=rng)
    return PosteriorDBTarget(
        name=name,
        dim=int(backend.dim),
        backend=backend,
        reference_samples=samples,
        is_surrogate=True,
    )


def get_target(name: str, **kwargs: Any) -> PosteriorDBTarget:
    """Alias of :func:`load_posteriordb_target`."""
    return load_posteriordb_target(name, **kwargs)


def make_target(name: str, **kwargs: Any) -> PosteriorDBTarget:
    """Alias of :func:`load_posteriordb_target`."""
    return load_posteriordb_target(name, **kwargs)


def posteriordb_targets(
    names: Optional[Iterable[str]] = None,
    seed: int = 0,
    reference_samples: Optional[Mapping[str, np.ndarray]] = None,
    **kwargs: Any,
) -> Dict[str, PosteriorDBTarget]:
    """Build all (or a subset of) the three Section 5.2 targets.

    Returns
    -------
    dict
        Mapping ``model name -> PosteriorDBTarget`` in paper order.
    """
    if names is None:
        names = PAPER_POSTERIORDB_NAMES
    out: Dict[str, PosteriorDBTarget] = {}
    for i, name in enumerate(names):
        refs = None
        if reference_samples is not None and name in reference_samples:
            refs = reference_samples[name]
        out[name] = load_posteriordb_target(name, reference_samples=refs, seed=seed + i, **kwargs)
    return out


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    for _name, _t in posteriordb_targets().items():
        _mu, _sd = _t.reference_moments()
        print(
            f"{_name:<26} D={_t.dim:<3} surrogate={_t.is_surrogate} "
            f"grad||={np.linalg.norm(_t.score(_mu)):.3e}"
        )
