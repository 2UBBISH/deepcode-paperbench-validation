"""CMA-ES optimiser used by FOA (Eqn. (6) of the paper).

The paper's Appendix B.2 states that the CMA-ES implementation comes from the
`cmaes <https://github.com/CyberAgentAILab/cmaes>`_ Python library.  We therefore use
that library whenever it is installed and fall back to a dependency-free NumPy
implementation of the very same algorithm (Hansen, 2016) so that the repository stays
self-contained.  Both expose the same tiny interface::

    opt = make_cma(mean, sigma0, popsize, seed=...)
    solutions = opt.ask()                    # [K, dim]
    opt.tell(solutions, values)              # values: [K]; lower is better
    opt.mean, opt.sigma, opt.covariance
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

__all__ = ["CMAOptimizer", "NumpyCMA", "make_cma"]


class CMAOptimizer:
    """Abstract CMA-ES interface used by :class:`foa.core.foa.FOA`."""

    dim: int
    popsize: int

    @property
    def mean(self) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def sigma(self) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def covariance(self) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def ask(self) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def tell(self, solutions: np.ndarray, values: Sequence[float]) -> None:
        raise NotImplementedError


class CmaesLibraryOptimizer(CMAOptimizer):
    """Thin adapter around :class:`cmaes.CMA` (the library named in the paper)."""

    def __init__(self, mean, sigma0: float, popsize: Optional[int] = None, seed=None, **kwargs):
        import cmaes  # imported lazily so that the fallback works without the package

        params = {
            "mean": np.asarray(mean, dtype=np.float64),
            "sigma": float(sigma0),
            "seed": seed,
        }
        if popsize is not None:
            # the argument is called ``population_size`` in cmaes >= 0.10 and ``popsize``
            # in older releases - support both.
            try:
                self._cma = cmaes.CMA(population_size=popsize, **params, **kwargs)
            except TypeError:
                self._cma = cmaes.CMA(popsize=popsize, **params, **kwargs)
        else:
            self._cma = cmaes.CMA(**params, **kwargs)
        self.dim = int(self._cma.dim)
        popsize_attr = getattr(self._cma, "population_size", None)
        if popsize_attr is None:
            popsize_attr = getattr(self._cma, "popsize", None)
        self.popsize = int(popsize_attr or 0)

    @property
    def mean(self) -> np.ndarray:
        # ``cmaes`` exposes ``mean`` in recent releases and ``_mean`` in older ones
        if hasattr(self._cma, "mean"):
            return np.asarray(self._cma.mean)
        return np.asarray(self._cma._mean)

    @property
    def sigma(self) -> float:
        return float(getattr(self._cma, "_sigma", getattr(self._cma, "sigma", 0.0)))

    @property
    def covariance(self) -> np.ndarray:
        return self._cma._C

    def ask(self) -> np.ndarray:
        # ``cmaes``' ask-and-tell interface samples *one* solution per call, so the whole
        # population is drawn in a loop (see the README of CyberAgentAILab/cmaes).
        return np.stack([np.asarray(self._cma.ask(), dtype=np.float64) for _ in range(self.popsize)])

    def tell(self, solutions: np.ndarray, values: Sequence[float]) -> None:
        self._cma.tell(list(zip(np.asarray(solutions), list(values))))


class NumpyCMA(CMAOptimizer):
    """Self-contained CMA-ES (Hansen, 2016) - used when ``cmaes`` is unavailable.

    It implements the standard rank-one + rank-mu covariance update with cumulative
    step-size adaptation, i.e. the exact algorithm that the paper relies on.  The
    default population size follows the paper, ``K = 28``.
    """

    def __init__(
        self,
        mean: np.ndarray,
        sigma0: float = 1.0,
        popsize: Optional[int] = None,
        seed: Optional[int] = None,
        cov: Optional[np.ndarray] = None,
    ) -> None:
        n = int(np.asarray(mean).size)
        self.dim = n
        self.popsize = int(popsize) if popsize else int(4 + 3 * np.log(n))
        self._rng = np.random.RandomState(seed)
        self._mean = np.asarray(mean, dtype=np.float64).reshape(-1).copy()
        self._sigma = float(sigma0)
        self._C = (
            np.eye(n) if cov is None else np.asarray(cov, dtype=np.float64).reshape(n, n).copy()
        )
        self._B = np.eye(n)
        self._D = np.ones(n)
        self._inv_sqrt_C = np.eye(n)
        self._eig_eval = 0

        # --- strategy parameters (Hansen, 2016) --------------------------------------
        mu = self.popsize // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        self._mu = mu
        self._weights = weights / weights.sum()
        self._mu_eff = 1.0 / np.sum(self._weights**2)

        self._cc = (4 + self._mu_eff / n) / (n + 4 + 2 * self._mu_eff / n)
        self._cs = (self._mu_eff + 2) / (n + self._mu_eff + 5)
        self._c1 = 2 / ((n + 1.3) ** 2 + self._mu_eff)
        self._cmu = min(
            1 - self._c1,
            2 * (self._mu_eff - 2 + 1 / self._mu_eff) / ((n + 2) ** 2 + self._mu_eff),
        )
        self._damps = (
            1
            + 2 * max(0, np.sqrt((self._mu_eff - 1) / (n + 1)) - 1)
            + self._cs
        )
        self._chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))

        self._pc = np.zeros(n)
        self._ps = np.zeros(n)
        self._update_eigensystem(force=True)

    # -- helpers ---------------------------------------------------------------------
    def _update_eigensystem(self, force: bool = False) -> None:
        if not force and self._eig_eval < 1.0 / (10 * self.dim * (self._c1 + self._cmu)):
            return
        self._C = np.triu(self._C) + np.triu(self._C, 1).T  # symmetrise
        d, b = np.linalg.eigh(self._C)
        d = np.maximum(d, 1e-30)
        self._D = np.sqrt(d)
        self._B = b
        self._inv_sqrt_C = b @ np.diag(1.0 / self._D) @ b.T
        self._eig_eval = 0

    @property
    def mean(self) -> np.ndarray:
        return self._mean

    @property
    def sigma(self) -> float:
        return self._sigma

    @property
    def covariance(self) -> np.ndarray:
        return self._C

    # -- interface -------------------------------------------------------------------
    def ask(self) -> np.ndarray:
        z = self._rng.randn(self.popsize, self.dim)
        y = z @ (self._B * self._D).T
        return self._mean + self._sigma * y

    def tell(self, solutions: np.ndarray, values: Sequence[float]) -> None:
        n = self.dim
        solutions = np.asarray(solutions, dtype=np.float64).reshape(-1, n)
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        order = np.argsort(values)
        x_sel = solutions[order[: self._mu]]
        y_sel = (x_sel - self._mean) / self._sigma
        y_w = self._weights @ y_sel

        # cumulation of the step size (CSA)
        self._ps = (1 - self._cs) * self._ps + np.sqrt(
            self._cs * (2 - self._cs) * self._mu_eff
        ) * (self._inv_sqrt_C @ y_w)
        hsig = (
            np.linalg.norm(self._ps)
            / np.sqrt(1 - (1 - self._cs) ** (2 * (self._eig_eval + 1)))
            / self._chi_n
            < 1.4 + 2 / (n + 1)
        )
        self._pc = (1 - self._cc) * self._pc + hsig * np.sqrt(
            self._cc * (2 - self._cc) * self._mu_eff
        ) * y_w

        # covariance matrix adaptation
        artmp = y_sel
        rank_mu = sum(
            w * np.outer(a, a) for w, a in zip(self._weights, artmp)
        )
        delta_hsig = (1 - hsig) * self._cc * (2 - self._cc)
        self._C = (
            (1 - self._c1 - self._cmu) * self._C
            + self._c1 * (np.outer(self._pc, self._pc) + delta_hsig * self._C)
            + self._cmu * rank_mu
        )

        # step-size control
        self._sigma *= np.exp((self._cs / self._damps) * (np.linalg.norm(self._ps) / self._chi_n - 1))

        # mean update
        self._mean = self._mean + self._sigma * y_w

        self._eig_eval += 1
        self._update_eigensystem()


def make_cma(
    mean,
    sigma0: float = 1.0,
    popsize: Optional[int] = None,
    seed: Optional[int] = None,
    prefer_library: bool = True,
    **kwargs,
) -> CMAOptimizer:
    """Instantiate a CMA-ES optimiser, preferring the ``cmaes`` library."""
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    if prefer_library:
        try:
            return CmaesLibraryOptimizer(mean, sigma0, popsize=popsize, seed=seed, **kwargs)
        except ImportError:
            pass
    return NumpyCMA(mean, sigma0=sigma0, popsize=popsize, seed=seed, **kwargs)
