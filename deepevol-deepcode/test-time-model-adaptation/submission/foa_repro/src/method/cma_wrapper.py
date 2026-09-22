"""CMA-ES wrapper used by FOA (Algorithm 1, Eqn. (6)).

The paper optimizes a *learnable input prompt* of shape ``d x N_p`` (``d`` = ViT
hidden size, ``N_p`` = number of prompt embeddings) with a covariance matrix
adaptation evolution strategy (Hansen & Ostermeier, 2001; Hansen et al., 2003;
Hansen, 2016).  The prompt is flattened into a vector of dimension
``d * N_p`` (2304 for ViT-Base with ``N_p = 3``) and, at every test batch ``t``,
CMA samples a population of ``K`` candidate prompts from

    p_k^{(t)} ~ m^{(t)} + tau^{(t)} * N(0, Sigma^{(t)})                (Eqn. 6)

with the *initialization prescribed by Algorithm 1*:

    m^{(0)} = 0,  Sigma^{(0)} = I,  tau^{(0)} = 1.

After the ``K`` candidates are scored with the unsupervised fitness of
Eqn. (5) -- which is minimized -- the distribution parameters
``m^{(t)}, tau^{(t)}, Sigma^{(t)}`` are updated by maximizing the likelihood of
the previously successful candidates (the ``tell`` step).  No gradient is ever
computed and the model weights are never touched: this wrapper only manipulates
the search distribution.

Backend
-------
The addendum states that the paper used the Python library at
https://github.com/CyberAgentAILab/cmaes, which is the default backend here
(``backend="cmaes"``).  ``pycma`` (the ``cma`` package) is supported as a
fallback with identical semantics, and ``backend="auto"`` picks whichever is
installed.  Both libraries *minimize* the fitness, exactly as FOA requires.

Population size
---------------
``K = 28 = 4 + 3 * log(prompt dim)`` for ViT-Base / ``N_p = 3`` (Section 4,
Implementation Details; Hansen 2016).  Note ``4 + floor(3 ln 2304) = 27`` while
the paper reports 28, so the paper's rounding is ``ceil``; both are exposed and
``ceil`` is the default so that the default configuration reproduces ``K = 28``.

Ambiguities resolved with documented defaults
---------------------------------------------
* ``tau^{0} = 1.0`` is taken literally from Algorithm 1 (``sigma0`` is
  configurable through ``cfg.cma.sigma0``).
* A single global seed is used for the CMA sampler (``cfg.cma.seed``) so runs
  are reproducible.
* ``tell`` is called once per test batch with all ``K`` fitness values.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "CMAOptimizer",
    "build_cma_optimizer",
    "default_population_size",
    "CMA_AVAILABLE_BACKENDS",
    "DEFAULT_MEAN0",
    "DEFAULT_SIGMA0",
]

# ---------------------------------------------------------------------------
# Paper constants
# ---------------------------------------------------------------------------
DEFAULT_MEAN0 = 0.0      # m^{(0)} = 0              (Algorithm 1)
DEFAULT_SIGMA0 = 1.0     # tau^{(0)} = 1            (Algorithm 1)
DEFAULT_COV0 = 1.0       # Sigma^{(0)} = I          (Algorithm 1)
CANONICAL_POP_SIZE = 28  # K = 4 + 3 * log(prompt dim) for dim = 2304

# Backends
BACKEND_CMAEAS = "cmaes"      # CyberAgentAILab/cmaes (addendum-preferred)
BACKEND_PYCMA = "pycma"       # pycma / `cma`
BACKEND_AUTO = "auto"
CMA_AVAILABLE_BACKENDS = (BACKEND_CMAEAS, BACKEND_PYCMA, BACKEND_AUTO)

_CLIP = 1e12  # guard against inf/nan fitness values


def _has(modname: str) -> bool:
    """Return ``True`` if ``modname`` can be imported."""
    try:
        __import__(modname)
        return True
    except Exception:  # pragma: no cover - environment dependent
        return False


def default_population_size(dim: int, rule: str = "ceil") -> int:
    """``K = 4 + 3 * log(dim)`` (Hansen, 2016) as used in Section 4.

    Parameters
    ----------
    dim:
        Dimension of the CMA search space, i.e. ``d * N_p`` (2304 for ViT-Base
        with ``N_p = 3``).
    rule:
        ``"ceil"`` reproduces the paper's ``K = 28`` for ``dim = 2304``;
        ``"floor"`` gives Hansen's textbook ``27``; ``"round"`` rounds to
        nearest.  ``"exact"`` returns the un-rounded float value.
    """
    dim = int(dim)
    if dim <= 1:
        value = 4.0
    else:
        value = 4.0 + 3.0 * math.log(dim)
    rule = (rule or "ceil").lower()
    if rule == "ceil":
        return int(math.ceil(value))
    if rule == "floor":
        return int(math.floor(value))
    if rule == "round":
        return int(round(value))
    if rule == "exact":
        return value  # type: ignore[return-value]
    raise ValueError(
        f"Unknown population-size rule {rule!r}; expected ceil/floor/round/exact."
    )


class CMAOptimizer:
    """Thin ask/tell wrapper around a CMA-ES implementation.

    The optimizer lives in the flattened prompt space ``R^{d * N_p}`` and
    minimizes the FOA fitness of Eqn. (5).

    Parameters
    ----------
    dim:
        Flattened prompt dimension ``d * N_p``.
    population_size:
        ``K``.  ``None`` -> ``default_population_size(dim)`` (28 for ViT-Base,
        ``N_p = 3``).
    sigma0:
        Initial overall step size ``tau^{(0)}`` (Algorithm 1: 1.0).
    mean0:
        Initial mean ``m^{(0)}`` (Algorithm 1: 0).  A scalar value is broadcast
        to all ``dim`` coordinates; a vector of length ``dim`` is used as is.
    cov0:
        Initial covariance ``Sigma^{(0)}`` (Algorithm 1: identity).  A scalar
        is interpreted as ``cov0 * I``; a matrix of shape ``(dim, dim)`` is used
        directly.
    seed:
        Seed for the sampler (fixes reproducibility of Eqn. (6)).
    backend:
        ``"cmaes"`` (default), ``"pycma"`` or ``"auto"``.
    bounds:
        Optional ``(lower, upper)`` box constraints forwarded to the backend.
    pop_size_rule:
        Rounding rule for the default population size (see
        :func:`default_population_size`).
    """

    def __init__(
        self,
        dim: int,
        population_size: Optional[int] = None,
        sigma0: float = DEFAULT_SIGMA0,
        mean0: Any = DEFAULT_MEAN0,
        cov0: Any = DEFAULT_COV0,
        seed: Optional[int] = 0,
        backend: str = BACKEND_AUTO,
        bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
        pop_size_rule: str = "ceil",
        tolfun: float = 1e-12,
        tolx: float = 1e-12,
        **backend_kwargs: Any,
    ) -> None:
        self.dim = int(dim)
        if self.dim <= 0:
            raise ValueError(f"CMA search dimension must be positive, got {dim!r}.")
        self.pop_size_rule = pop_size_rule
        self.population_size = int(
            population_size
            if population_size
            else default_population_size(self.dim, pop_size_rule)
        )
        if self.population_size < 2:
            raise ValueError("CMA population size K must be >= 2.")
        self.sigma0 = float(sigma0)
        self.mean0_value = mean0
        self.cov0 = cov0
        self.seed = seed
        self.bounds = bounds
        self.tolfun = float(tolfun)
        self.tolx = float(tolx)
        self._backend_kwargs = dict(backend_kwargs)

        self._mean0 = self._make_mean(mean0)
        self.backend = self._resolve_backend(backend)

        # Bookkeeping
        self.generation = 0
        self.num_evaluations = 0
        self.pending_solutions: Optional[np.ndarray] = None
        self.last_values: Optional[np.ndarray] = None
        self.last_best_index: Optional[int] = None

        self._init_backend()

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    def _make_mean(self, mean0: Any) -> np.ndarray:
        if np.isscalar(mean0):
            return np.full(self.dim, float(mean0), dtype=np.float64)
        arr = np.asarray(mean0, dtype=np.float64).reshape(-1)
        if arr.size != self.dim:
            raise ValueError(
                f"mean0 must be a scalar or of length {self.dim}, got {arr.size}."
            )
        return arr.copy()

    def _make_cov(self) -> Optional[np.ndarray]:
        cov0 = self.cov0
        if cov0 is None:
            return None
        if np.isscalar(cov0):
            return float(cov0) * np.eye(self.dim, dtype=np.float64)
        arr = np.asarray(cov0, dtype=np.float64)
        if arr.ndim == 1:
            arr = np.diag(arr)
        if arr.shape != (self.dim, self.dim):
            raise ValueError(
                f"cov0 must be a scalar or shape {(self.dim, self.dim)}, got {arr.shape}."
            )
        return arr.copy()

    @staticmethod
    def _resolve_backend(backend: str) -> str:
        backend = (backend or BACKEND_AUTO).lower()
        if backend == BACKEND_CMAEAS:
            if not _has("cmaes"):
                raise ImportError(
                    "backend='cmaes' requested but the `cmaes` package is not "
                    "installed.  Install it with `pip install cmaes` (the "
                    "library used by the paper, per the addendum)."
                )
            return BACKEND_CMAEAS
        if backend == BACKEND_PYCMA:
            if not _has("cma"):
                raise ImportError(
                    "backend='pycma' requested but the `cma` package is not "
                    "installed.  Install it with `pip install cma`."
                )
            return BACKEND_PYCMA
        if backend == BACKEND_AUTO:
            if _has("cmaes"):
                return BACKEND_CMAEAS
            if _has("cma"):
                return BACKEND_PYCMA
            raise ImportError(
                "No CMA-ES backend found.  Install `cmaes` (preferred, "
                "https://github.com/CyberAgentAILab/cmaes) or `cma` (pycma)."
            )
        raise ValueError(
            f"Unknown backend {backend!r}; expected one of {CMA_AVAILABLE_BACKENDS}."
        )

    def _init_backend(self) -> None:
        if self.backend == BACKEND_CMAEAS:
            self._init_cmaes()
        else:
            self._init_pycma()

    def _init_cmaes(self) -> None:
        import cmaes  # local import keeps the module importable without the dep

        cov = self._make_cov()
        kwargs: Dict[str, Any] = {
            "mean": self._mean0.copy(),
            "sigma": self.sigma0,
            "population_size": self.population_size,
            "seed": self.seed,
        }
        if self.bounds is not None:
            kwargs["bounds"] = (
                np.asarray(self.bounds[0], dtype=np.float64),
                np.asarray(self.bounds[1], dtype=np.float64),
            )
        if cov is not None and abs(float(np.trace(cov)) / self.dim - 1.0) > 1e-12:
            # Only pass an explicit covariance when it is not the identity: the
            # Algorithm-1 default Sigma^{(0)} = I is already the library default.
            for key in ("cov", "covariance_matrix", "C"):
                try:
                    self._optimizer = cmaes.CMA(**kwargs, **{key: cov})
                    return
                except TypeError:
                    continue
        try:
            self._optimizer = cmaes.CMA(**kwargs, **self._backend_kwargs)
        except TypeError:
            kwargs.pop("bounds", None)
            self._optimizer = cmaes.CMA(**kwargs)

    def _init_pycma(self) -> None:
        import cma  # pycma

        options: Dict[str, Any] = {
            "popsize": self.population_size,
            "seed": self.seed,
            "verbose": -9,          # silence pycma's stdout chatter
            "tolfun": self.tolfun,
            "tolx": self.tolx,
            "maxiter": 10 ** 9,     # FOA runs one CMA iteration per test batch
            "CMA_diagonal": 0,
        }
        if self.bounds is not None:
            options["bounds"] = [
                np.asarray(self.bounds[0], dtype=np.float64).tolist(),
                np.asarray(self.bounds[1], dtype=np.float64).tolist(),
            ]
        options.update(self._backend_kwargs)
        self._optimizer = cma.CMAEvolutionStrategy(
            self._mean0.copy(), self.sigma0, options
        )

    # ------------------------------------------------------------------
    # ask / tell -- Eqn. (6) and the CMA distribution update
    # ------------------------------------------------------------------
    def ask(self, population_size: Optional[int] = None) -> np.ndarray:
        """Sample ``K`` candidate prompts ``{p_k}`` from Eqn. (6).

        Returns an array of shape ``(K, dim)`` (``float64``).  The sampled
        solutions are cached and consumed by the matching :meth:`tell`.
        """
        k = int(population_size) if population_size else self.population_size
        if self.backend == BACKEND_CMAEAS:
            samples = np.stack(
                [np.asarray(self._optimizer.ask(), dtype=np.float64) for _ in range(k)],
                axis=0,
            )
        else:
            samples = np.asarray(self._optimizer.ask(), dtype=np.float64)
            if samples.ndim == 1:  # defensive: single-solution backend
                samples = samples[None, :]
        if samples.shape != (k, self.dim):
            samples = samples.reshape(k, self.dim)
        self.pending_solutions = samples
        return samples

    ask_population = ask  # alias emphasising the population contract

    def tell(
        self,
        values: Optional[Sequence[float]] = None,
        solutions: Optional[np.ndarray] = None,
    ) -> None:
        """Update ``m, tau, Sigma`` from the ranked fitness values.

        Parameters
        ----------
        values:
            The ``K`` fitness values ``{v_k}`` of Eqn. (5) (lower is better).
            NaNs/infs are replaced by a large finite penalty so the CMA ranking
            stays well defined.
        solutions:
            Optional explicit candidates; defaults to those from the last
            :meth:`ask`.
        """
        if solutions is None:
            solutions = self.pending_solutions
        if solutions is None:
            raise RuntimeError("tell() called before ask(): no candidate population.")
        solutions = np.asarray(solutions, dtype=np.float64).reshape(-1, self.dim)
        if values is None:
            raise ValueError("tell() requires the K fitness values.")
        vals = np.asarray(values, dtype=np.float64).reshape(-1)
        if vals.size != solutions.shape[0]:
            raise ValueError(
                f"Got {vals.size} fitness values for {solutions.shape[0]} candidates."
            )
        vals = np.where(np.isfinite(vals), vals, _CLIP)

        if self.backend == BACKEND_CMAEAS:
            self._optimizer.tell(
                [(solutions[i].copy(), float(vals[i])) for i in range(vals.size)]
            )
        else:
            try:
                self._optimizer.tell(list(solutions), [float(v) for v in vals])
            except Exception:
                # pycma's ask/tell interface is stateful; fall back to the
                # "ask one at a time" contract if the batch form is rejected.
                for i in range(vals.size):
                    self._optimizer.tell(
                        [solutions[i].tolist()], [float(vals[i])]
                    )

        self.generation += 1
        self.num_evaluations += int(vals.size)
        self.last_values = vals
        self.last_best_index = int(np.argmin(vals))
        self.pending_solutions = None

    # ------------------------------------------------------------------
    # convenience
    # ------------------------------------------------------------------
    def optimize_step(
        self,
        evaluate: Callable[[np.ndarray], float],
        candidates: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """One full FOA CMA iteration: sample -> score -> update.

        Parameters
        ----------
        evaluate:
            Callable mapping a candidate vector ``[dim]`` to a scalar fitness
            (lower is better).  It must not build a computation graph -- FOA is
            backpropagation-free.
        candidates:
            Optional pre-sampled population (e.g. for replay/tests).

        Returns
        -------
        dict with keys ``candidates`` ``(K, dim)``, ``values`` ``(K,)``,
        ``best_index``, ``best_candidate`` ``(dim,)``, ``best_value``,
        ``mean``, ``sigma``, ``generation``.
        """
        if candidates is None:
            candidates = self.ask()
        else:
            candidates = np.asarray(candidates, dtype=np.float64).reshape(-1, self.dim)
            self.pending_solutions = candidates
        values = np.asarray(
            [float(evaluate(np.asarray(p, dtype=np.float64))) for p in candidates],
            dtype=np.float64,
        )
        best_index = int(np.argmin(np.where(np.isfinite(values), values, _CLIP)))
        self.tell(values, candidates)
        return {
            "candidates": candidates,
            "values": values,
            "best_index": best_index,
            "best_candidate": candidates[best_index].copy(),
            "best_value": float(values[best_index]),
            "mean": self.mean.copy(),
            "sigma": self.sigma,
            "generation": self.generation,
        }

    @staticmethod
    def best_index(values: np.ndarray) -> int:
        """Index of the best (lowest) fitness, matching Algorithm 1's final pick."""
        vals = np.asarray(values, dtype=np.float64).reshape(-1)
        return int(np.argmin(np.where(np.isfinite(vals), vals, _CLIP)))

    # ------------------------------------------------------------------
    # state access
    # ------------------------------------------------------------------
    @property
    def mean(self) -> np.ndarray:
        """Current distribution mean ``m^{(t)}``."""
        opt = self._optimizer
        for attr in ("mean", "_mean"):
            if hasattr(opt, attr):
                return np.asarray(getattr(opt, attr), dtype=np.float64).reshape(-1)
        return np.asarray(opt.xmean, dtype=np.float64).reshape(-1)

    @property
    def sigma(self) -> float:
        """Current overall step size ``tau^{(t)}``."""
        opt = self._optimizer
        for attr in ("sigma", "_sigma"):
            if hasattr(opt, attr):
                try:
                    return float(getattr(opt, attr))
                except Exception:  # pragma: no cover
                    continue
        try:
            return float(opt.sigma0)
        except Exception:  # pragma: no cover
            return float("nan")

    @property
    def covariance(self) -> np.ndarray:
        """Current covariance matrix ``Sigma^{(t)}``."""
        opt = self._optimizer
        for attr in ("_C", "C", "covariance_matrix"):
            if hasattr(opt, attr):
                try:
                    return np.asarray(getattr(opt, attr), dtype=np.float64)
                except Exception:  # pragma: no cover
                    continue
        return np.eye(self.dim, dtype=np.float64)

    def state_dict(self) -> Dict[str, Any]:
        """Serializable optimizer state (distribution mean/step size/covariance)."""
        return {
            "dim": self.dim,
            "population_size": self.population_size,
            "sigma0": self.sigma0,
            "seed": self.seed,
            "backend": self.backend,
            "generation": int(self.generation),
            "num_evaluations": int(self.num_evaluations),
            "mean": self.mean.tolist(),
            "sigma": self.sigma,
            "covariance": self.covariance.tolist(),
        }

    def reset(self) -> None:
        """Reset the search distribution to the Algorithm-1 initialization."""
        self._init_backend()
        self.generation = 0
        self.num_evaluations = 0
        self.pending_solutions = None
        self.last_values = None
        self.last_best_index = None

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, K={self.population_size}, tau0={self.sigma0}, "
            f"backend={self.backend}, seed={self.seed}"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}({self.extra_repr()})"


def build_cma_optimizer(
    cfg: Any = None,
    dim: Optional[int] = None,
    prompt: Any = None,
    **overrides: Any,
) -> CMAOptimizer:
    """Factory mirroring the config interface (``cfg.cma.*``).

    ``dim`` defaults to ``prompt.prompt_dim`` (``d * N_p``) when a
    :class:`~src.models.prompt_injection.PromptInjection` instance is supplied,
    or to ``cfg.prompt.num_prompts * cfg.model.embed_dim`` when only a config is
    available.
    """
    params: Dict[str, Any] = {}

    def _cfg_get(cfg_obj: Any, key: str, default: Any = None) -> Any:
        try:
            return cfg_obj[key]
        except Exception:
            return getattr(cfg_obj, key, default)

    if cfg is not None:
        cma_cfg = _cfg_get(cfg, "cma", {}) or {}
        for src_key, dst_key in (
            ("population_size", "population_size"),
            ("sigma0", "sigma0"),
            ("mean0", "mean0"),
            ("cov0", "cov0"),
            ("seed", "seed"),
            ("backend", "backend"),
            ("pop_size_rule", "pop_size_rule"),
        ):
            value = _cfg_get(cma_cfg, src_key, None)
            if value is not None:
                params[dst_key] = value
        if dim is None:
            if prompt is not None and hasattr(prompt, "prompt_dim"):
                dim = int(prompt.prompt_dim)
            else:
                model_cfg = _cfg_get(cfg, "model", {}) or {}
                prompt_cfg = _cfg_get(cfg, "prompt", {}) or {}
                embed_dim = int(_cfg_get(model_cfg, "embed_dim", 768) or 768)
                num_prompts = int(_cfg_get(prompt_cfg, "num_prompts", 3) or 3)
                dim = embed_dim * num_prompts

    if prompt is not None and hasattr(prompt, "prompt_dim"):
        dim = int(prompt.prompt_dim)
    if dim is None:
        raise ValueError(
            "build_cma_optimizer requires `dim` (or `prompt`/`cfg` to infer it)."
        )

    params.update(overrides)
    params.setdefault("seed", 0)
    return CMAOptimizer(dim=int(dim), **params)
