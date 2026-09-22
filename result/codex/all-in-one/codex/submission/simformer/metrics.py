"""Evaluation metrics of the paper.

* Classifier two-sample test (C2ST) accuracy: the paper trains a **random forest
  with 100 trees** to discriminate samples from the Simformer and reference
  samples of the ground truth distribution (addendum, "Training").  ``0.5``
  means the two sets of samples are indistinguishable (perfect alignment),
  ``1.0`` means they can be perfectly separated.
* Expected coverage (Hermans et al., 2022): used for the calibration analysis of
  the SIR task in the main text (Fig. 6 / Appendix Fig. A13).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


def c2st_accuracy(samples_a: np.ndarray, samples_b: np.ndarray,
                  n_trees: int = 100, n_folds: int = 5,
                  seed: int = 0, standardize: bool = True,
                  max_samples: Optional[int] = None) -> float:
    """C2ST accuracy between two sets of samples (random forest, 100 trees).

    The classifier is evaluated with ``n_folds``-fold cross validation on the
    balanced data set, which yields a low variance estimate of the accuracy
    (a value close to ``0.5`` means that the two distributions match).
    """
    samples_a = np.atleast_2d(np.asarray(samples_a, dtype=float))
    samples_b = np.atleast_2d(np.asarray(samples_b, dtype=float))
    if samples_a.shape[1] != samples_b.shape[1]:
        raise ValueError("both sample sets need to have the same dimension")
    n = min(len(samples_a), len(samples_b))
    if max_samples is not None:
        n = min(n, max_samples)
    a = samples_a[:n]
    b = samples_b[:n]
    features = np.concatenate([a, b], axis=0)
    labels = np.concatenate([np.zeros(n), np.ones(n)])
    if standardize:
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True) + 1e-12
        features = (features - mean) / std
    accuracy = []
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True,
                               random_state=seed)
    for train_idx, test_idx in splitter.split(features, labels):
        classifier = RandomForestClassifier(n_estimators=n_trees,
                                            random_state=seed, n_jobs=-1)
        classifier.fit(features[train_idx], labels[train_idx])
        accuracy.append(float(np.mean(
            classifier.predict(features[test_idx]) == labels[test_idx])))
    return float(np.mean(accuracy))


def expected_coverage(samples: np.ndarray, true_values: np.ndarray,
                      alphas: Optional[Sequence[float]] = None,
                      distance: str = "euclidean") -> np.ndarray:
    """Expected coverage of posterior samples (Hermans et al., 2022).

    For every credibility level ``alpha`` the fraction of test cases is computed
    for which the ground truth parameter lies in the ``alpha`` highest density
    region of the approximate posterior (estimated by the fraction of posterior
    samples that are at least as close to the posterior mode as the ground
    truth).
    """
    samples = np.atleast_3d(np.asarray(samples, dtype=float))
    true_values = np.atleast_2d(np.asarray(true_values, dtype=float))
    if samples.ndim == 3:
        samples = samples[:, :, 0] if samples.shape[2] == 1 else samples
    if samples.ndim == 2:
        samples = samples[None]
    if alphas is None:
        alphas = np.linspace(0.05, 0.95, 10)
    coverage = np.zeros((len(alphas), samples.shape[0]))
    for i in range(samples.shape[0]):
        posterior = samples[i]
        center = np.median(posterior, axis=0)
        d_samples = np.linalg.norm(posterior - center, axis=-1)
        d_true = np.linalg.norm(true_values[i] - center, axis=-1)
        rank = np.mean(d_samples <= d_true)
        coverage[:, i] = (rank <= alphas).astype(float)
    return np.mean(coverage, axis=-1)


def sample_mmd(samples_a: np.ndarray, samples_b: np.ndarray,
               bandwidth: Optional[float] = None) -> float:
    """Maximum mean discrepancy between two sample sets (diagnostic)."""
    from scipy.spatial.distance import cdist

    a = np.asarray(samples_a, dtype=float)
    b = np.asarray(samples_b, dtype=float)
    if bandwidth is None:
        combined = np.concatenate([a, b], axis=0)
        pairwise = cdist(combined[:200], combined[:200])
        bandwidth = np.median(pairwise[pairwise > 0]) + 1e-12

    def kernel(x, y):
        return np.exp(-cdist(x, y) ** 2 / (2 * bandwidth ** 2))

    kaa = kernel(a, a).mean()
    kbb = kernel(b, b).mean()
    kab = kernel(a, b).mean()
    return float(kaa + kbb - 2 * kab)
