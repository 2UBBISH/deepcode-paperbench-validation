"""Uniform sampling baseline (Appendix D.1, Section 5.2).

Paper text (Appendix D.1):

    "Uniform sampling. For this baseline, we randomly select partial data from
    full data to construct a coreset."

This is the simplest competitor in Table 2 / Table 3 of the paper: given a
*predefined* coreset size ``k`` (the baselines "construct the coreset with a
predetermined coreset size, where the size is not further minimized by
optimization"), draw ``k`` examples uniformly at random from the full training
set (without replacement) and set ``m_i = 1`` for the drawn indices.

The class follows the shared :class:`lbcs_repro.baselines.base.BaselineSelector`
interface so that it can be swapped with EL2N / GraNd / Influential / Moderate /
CCS inside the experiment drivers.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .base import (
    BaselineSelector,
    indices_to_mask,
    resolve_seed,
)

LOGGER = logging.getLogger(__name__)

__all__ = ["UniformSelector", "UniformSampling", "uniform_mask", "uniform_indices"]


class UniformSelector(BaselineSelector):
    """Random subset selection (Uniform) as described in Appendix D.1.

    Parameters
    ----------
    seed:
        Base seed for the deterministic random draw.  Per-repeat seeds are
        derived with :func:`resolve_seed` (the paper repeats every §5.2
        experiment ten times).
    device:
        Accepted for interface compatibility; unused (no model is required).
    stratified:
        If ``True``, draw ``k / C`` examples from each class.  The paper's
        Uniform baseline is *not* class-aware (it "randomly selects partial
        data from full data"); ``stratified=False`` is therefore the default
        and reproduces Table 2.  The option only exists to support the
        class-imbalanced analysis of §5.3.
    num_classes:
        Number of classes, required only when ``stratified=True``.
    """

    name = "Uniform"
    abbreviation = "Uniform"
    requires_model = False
    higher_is_better = False  # scores are random / unused

    def __init__(
        self,
        seed: Optional[int] = None,
        device: Any = None,
        stratified: bool = False,
        num_classes: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(seed=seed, device=device, **kwargs)
        self.stratified = bool(stratified)
        self.num_classes = num_classes

    # ------------------------------------------------------------------ #
    # Core interface
    # ------------------------------------------------------------------ #
    def compute_scores(
        self,
        dataset: Any = None,
        targets: Optional[Sequence[int]] = None,
        n: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Return random "scores" used purely to rank examples.

        Uniform sampling ignores any notion of example importance; we expose a
        uniform random score vector so that the usual top-``k`` machinery of
        :class:`ScoreBaseline` remains valid.  Ranking a random score vector is
        exactly a random subset of size ``k``.
        """
        n = self._resolve_n(dataset=dataset, targets=targets, n=n)
        rng = np.random.default_rng(self._seed_for(seed))
        return rng.random(n)

    def select_indices(
        self,
        n: Optional[int] = None,
        k: Optional[int] = None,
        dataset: Any = None,
        targets: Optional[Sequence[int]] = None,
        num_classes: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Draw ``k`` example indices uniformly at random without replacement."""
        n = self._resolve_n(dataset=dataset, targets=targets, n=n)
        k = self._resolve_k(k=k, n=n)
        rng = np.random.default_rng(self._seed_for(seed))

        if self.stratified:
            targets = self._resolve_targets(dataset=dataset, targets=targets)
            c = num_classes or self.num_classes or (
                int(np.max(targets)) + 1 if targets is not None else None
            )
            if targets is None or c is None:
                raise ValueError(
                    "UniformSelector(stratified=True) requires `targets` and "
                    "`num_classes`."
                )
            return self._stratified_draw(np.asarray(targets), k, int(c), rng)

        # np.random.Generator.choice without replacement == uniform random subset.
        return np.sort(rng.choice(n, size=k, replace=False)).astype(np.int64)

    def select_mask(
        self,
        n: Optional[int] = None,
        k: Optional[int] = None,
        dataset: Any = None,
        targets: Optional[Sequence[int]] = None,
        num_classes: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Return the binary mask ``m in {0,1}^n`` with exactly ``||m||_0 = k``."""
        n = self._resolve_n(dataset=dataset, targets=targets, n=n)
        idx = self.select_indices(
            n=n,
            k=k,
            dataset=dataset,
            targets=targets,
            num_classes=num_classes,
            seed=seed,
            **kwargs,
        )
        return indices_to_mask(idx, n)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _seed_for(self, seed: Optional[int]) -> int:
        return int(self.seed if seed is None else seed)

    def _resolve_n(
        self,
        dataset: Any = None,
        targets: Optional[Sequence[int]] = None,
        n: Optional[int] = None,
    ) -> int:
        if n is not None:
            return int(n)
        if targets is not None:
            return int(len(targets))
        if dataset is not None:
            for attr in ("targets", "labels", "__len__"):
                try:
                    if attr == "__len__":
                        return int(len(dataset))
                    value = getattr(dataset, attr)
                    if value is not None:
                        return int(len(value))
                except Exception:  # pragma: no cover - defensive
                    continue
        raise ValueError("UniformSelector needs `n`, `targets`, or `dataset`.")

    def _resolve_k(self, k: Optional[int], n: int) -> int:
        if k is None:
            raise ValueError("UniformSelector requires the predefined size `k`.")
        k = int(k)
        if k < 0 or k > n:
            raise ValueError(f"k={k} outside the valid range [0, {n}].")
        return k

    def _resolve_targets(
        self, dataset: Any = None, targets: Optional[Sequence[int]] = None
    ) -> Optional[np.ndarray]:
        if targets is not None:
            return np.asarray(targets, dtype=np.int64)
        if dataset is not None:
            for attr in ("targets", "labels", "clean_targets"):
                value = getattr(dataset, attr, None)
                if value is not None:
                    try:
                        return np.asarray(value, dtype=np.int64).reshape(-1)
                    except Exception:
                        if hasattr(value, "numpy"):
                            return value.numpy().astype(np.int64).reshape(-1)
        return None

    @staticmethod
    def _stratified_draw(
        targets: np.ndarray, k: int, num_classes: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Draw ``k`` indices with (approximately) equal per-class counts."""
        picked = []
        classes = np.arange(num_classes)
        quota = k // num_classes
        remainder = k - quota * num_classes

        for c in classes:
            pool = np.flatnonzero(targets == c)
            take = min(quota, pool.size)
            if take > 0:
                picked.append(rng.choice(pool, size=take, replace=False))

        # Distribute the remainder over classes that still have spare examples.
        if remainder > 0:
            spare_classes = []
            for c in classes:
                pool = np.flatnonzero(targets == c)
                already = sum(1 for i in picked[-1] if False)  # placeholder
                del already
                if pool.size > quota:
                    spare_classes.append(c)
            if spare_classes:
                picked_from = []
                for c in rng.permutation(spare_classes)[:remainder]:
                    pool = np.flatnonzero(targets == c)
                    available = np.setdiff1d(pool, picked_from, assume_unique=False)
                    if available.size:
                        picked_from.append(int(rng.choice(available)))
                picked.append(np.asarray(picked_from, dtype=np.int64))

        if not picked:
            raise ValueError("Stratified uniform draw produced no indices.")
        idx = np.unique(np.concatenate([np.asarray(p, dtype=np.int64) for p in picked]))
        # Top up if rounding/shortage left us below k.
        if idx.size < k:
            spare = np.setdiff1d(np.arange(targets.size), idx, assume_unique=False)
            extra = rng.choice(spare, size=min(k - idx.size, spare.size), replace=False)
            idx = np.unique(np.concatenate([idx, extra.astype(np.int64)]))
        return np.sort(idx[:k]).astype(np.int64)


# ---------------------------------------------------------------------- #
# Functional API (used by the experiment drivers / Table 2-3 harness)
# ---------------------------------------------------------------------- #
def uniform_indices(
    n: int,
    k: int,
    seed: Optional[int] = 0,
    repeat: int = 0,
    **kwargs: Any,
) -> np.ndarray:
    """Convenience wrapper: ``k`` random indices out of ``n`` (without replacement)."""
    return UniformSelector(seed=seed, **kwargs).select_indices(
        n=n, k=k, seed=resolve_seed(seed, repeat)
    )


def uniform_mask(
    n: int,
    k: int,
    seed: Optional[int] = 0,
    repeat: int = 0,
    **kwargs: Any,
) -> np.ndarray:
    """Convenience wrapper: binary coreset mask of size ``k`` (Appendix D.1)."""
    return UniformSelector(seed=seed, **kwargs).select_mask(
        n=n, k=k, seed=resolve_seed(seed, repeat)
    )


# Aliases matching other naming styles in the code base.
UniformSampling = UniformSelector


def _selftest(verbose: bool = True) -> Dict[str, Any]:
    """Offline checks: exact size, no duplicates, determinism, stratified mode."""
    report: Dict[str, Any] = {}
    n, k = 500, 100

    m = uniform_mask(n, k, seed=0)
    report["mask_size"] = int(np.count_nonzero(m))
    assert m.shape == (n,)
    assert set(np.unique(m)).issubset({0.0, 1.0})
    assert int(np.count_nonzero(m)) == k, "uniform mask must select exactly k"

    idx = uniform_mask(n, k, seed=0)
    assert np.array_equal(idx, m), "same seed must give the same subset"
    assert not np.array_equal(uniform_mask(n, k, seed=1), m), "seed must matter"

    for r in range(3):
        mr = uniform_mask(n, k, seed=7, repeat=r)
        assert int(np.count_nonzero(mr)) == k

    sel = UniformSelector(seed=3, stratified=True, num_classes=10)
    tgt = np.repeat(np.arange(10), 50)
    ms = sel.select_mask(n=tgt.size, k=100, targets=tgt, num_classes=10)
    assert int(np.count_nonzero(ms)) == 100
    counts = np.bincount(tgt[ms.astype(bool)], minlength=10)
    report["stratified_counts"] = counts.tolist()
    assert counts.min() >= 9, "stratified draw should be class balanced"

    report["ok"] = True
    if verbose:
        LOGGER.info("uniform selftest: %s", report)
    return report


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    print(_selftest())
