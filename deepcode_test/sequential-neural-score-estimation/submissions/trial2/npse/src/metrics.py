"""Evaluation metrics for Neural Posterior Score Estimation.

This module implements the quantitative metrics used to assess the quality of
approximate posteriors throughout the NPSE paper:

* Classifier 2-Sample Test (C2ST)
* Maximum Mean Discrepancy (MMD)
* Simulation-Based Calibration (coverage)
* Posterior predictive checks

All functions operate on ``torch.Tensor`` inputs (CPU or GPU) and return Python
scalars or tensors that are straightforward to log.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import torch

__all__ = [
    "c2st_score",
    "c2st_nn_score",
    "mmd",
    "rbf_mmd",
    "median_heuristic_bandwidth",
    "simulation_based_calibration",
    "expected_calibration_error",
    "posterior_predictive_check",
]


# ---------------------------------------------------------------------------
# Classifier 2-Sample Test (C2ST)
# ---------------------------------------------------------------------------
def _to_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert a tensor/array to a float64 CPU numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().to(torch.float64).numpy()
    return np.asarray(x, dtype=np.float64)


def _torch_mlp_classifier(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    hidden_sizes: Sequence[int] = (128, 128),
    epochs: int = 300,
    batch_size: int = 128,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    seed: int = 42,
) -> float:
    """Train a small torch MLP classifier as a scikit-learn-free C2ST fallback."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_dim = X_train.shape[1]
    layers: list[torch.nn.Module] = []
    prev = in_dim
    for h in hidden_sizes:
        layers.append(torch.nn.Linear(prev, h))
        layers.append(torch.nn.ReLU())
        prev = h
    layers.append(torch.nn.Linear(prev, 2))
    net = torch.nn.Sequential(*layers).to(device)

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    n = len(X_train)
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            out = net(X_train[idx])
            loss = loss_fn(out, y_train[idx].long())
            opt.zero_grad()
            loss.backward()
            opt.step()

    net.eval()
    with torch.no_grad():
        pred = net(X_test).argmax(dim=1)
        acc = (pred == y_test.long()).float().mean().item()
    return float(acc)


def c2st_nn_score(
    samples_a: Union[torch.Tensor, np.ndarray],
    samples_b: Union[torch.Tensor, np.ndarray],
    hidden_sizes: Sequence[int] = (128, 128),
    epochs: int = 300,
    batch_size: int = 128,
    lr: float = 1e-3,
    val_fraction: float = 0.5,
    device: Optional[torch.device] = None,
    seed: int = 42,
) -> float:
    """Classifier 2-Sample Test using a pure-torch MLP.

    Combines the two sample sets, labels them 0/1, splits into train/test, and
    returns the held-out classification accuracy. A value close to 0.5 indicates
    the samples are indistinguishable.
    """
    a = _to_numpy(samples_a)
    b = _to_numpy(samples_b)
    n_a = len(a)
    n_b = len(b)
    # Balance the classes.
    n = min(n_a, n_b)
    rng = np.random.RandomState(seed)
    a = a[rng.choice(n_a, size=n, replace=False)]
    b = b[rng.choice(n_b, size=n, replace=False)]

    X = torch.tensor(np.concatenate([a, b], axis=0), dtype=torch.float32)
    y = torch.tensor(
        np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)]),
        dtype=torch.float32,
    )

    n_train = int((1.0 - val_fraction) * len(X))
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed))
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    return _torch_mlp_classifier(
        X[train_idx],
        y[train_idx],
        X[test_idx],
        y[test_idx],
        hidden_sizes=hidden_sizes,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        seed=seed,
    )


def c2st_score(
    samples_a: Union[torch.Tensor, np.ndarray],
    samples_b: Union[torch.Tensor, np.ndarray],
    hidden_sizes: Sequence[int] = (128, 128),
    n_folds: int = 5,
    max_iter: int = 500,
    random_state: int = 42,
    use_sklearn: bool = True,
) -> float:
    """Classifier 2-Sample Test (C2ST).

    Trains a classifier to distinguish samples from two distributions and
    reports its accuracy using stratified cross-validation. Accuracy near 0.5
    means the two sample sets are statistically indistinguishable, while higher
    values indicate a mismatch.

    Parameters
    ----------
    samples_a, samples_b:
        Sample matrices of shape ``(n_samples, dim)``.
    hidden_sizes:
        Hidden-layer sizes for the MLP classifier.
    n_folds:
        Number of cross-validation folds (used when scikit-learn is available).
    max_iter:
        Maximum number of iterations for the scikit-learn MLP classifier.
    random_state:
        Random seed for reproducible fold assignment and classifier init.
    use_sklearn:
        If ``True``, prefer ``sklearn``'s MLP classifier. Otherwise use the
        built-in torch classifier.

    Returns
    -------
    float
        Mean held-out classification accuracy.
    """
    if use_sklearn:
        try:
            from sklearn.model_selection import StratifiedKFold, cross_val_score
            from sklearn.neural_network import MLPClassifier

            a = _to_numpy(samples_a)
            b = _to_numpy(samples_b)
            n = min(len(a), len(b))
            rng = np.random.RandomState(random_state)
            a = a[rng.choice(len(a), size=n, replace=False)]
            b = b[rng.choice(len(b), size=n, replace=False)]

            X = np.concatenate([a, b], axis=0)
            y = np.concatenate([np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)])

            clf = MLPClassifier(
                hidden_layer_sizes=tuple(hidden_sizes),
                max_iter=max_iter,
                random_state=random_state,
                early_stopping=False,
            )
            # Use leave-one-out-like behavior for very small sample counts.
            n_splits = min(n_folds, n)
            if n_splits < 2:
                # Fall back to a single stratified train/test split.
                from sklearn.model_selection import train_test_split

                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=0.5, stratify=y, random_state=random_state
                )
                clf.fit(X_tr, y_tr)
                return float(clf.score(X_te, y_te))

            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
            return float(np.mean(scores))
        except ImportError:  # pragma: no cover - environment dependent
            pass

    return c2st_nn_score(
        samples_a,
        samples_b,
        hidden_sizes=hidden_sizes,
        seed=random_state,
    )


# ---------------------------------------------------------------------------
# Maximum Mean Discrepancy (MMD)
# ---------------------------------------------------------------------------
def median_heuristic_bandwidth(
    samples_a: Union[torch.Tensor, np.ndarray],
    samples_b: Union[torch.Tensor, np.ndarray],
) -> float:
    """Median heuristic for the RBF kernel bandwidth.

    Uses the median of pairwise squared distances between samples from the two
    sets (a robust scale estimate).
    """
    a = _to_numpy(samples_a)
    b = _to_numpy(samples_b)
    # Subsample to keep the distance computation tractable.
    max_points = 1000
    if len(a) > max_points:
        a = a[np.random.RandomState(0).choice(len(a), size=max_points, replace=False)]
    if len(b) > max_points:
        b = b[np.random.RandomState(0).choice(len(b), size=max_points, replace=False)]

    aa = np.sum(a * a, axis=1, keepdims=True)
    bb = np.sum(b * b, axis=1, keepdims=True)
    sq = aa + bb.T - 2.0 * (a @ b.T)
    med = np.median(sq)
    return float(np.sqrt(max(med, 1e-12)))


def rbf_mmd(
    samples_a: Union[torch.Tensor, np.ndarray],
    samples_b: Union[torch.Tensor, np.ndarray],
    bandwidth: Optional[float] = None,
) -> float:
    """Biased RBF-kernel Maximum Mean Discrepancy.

    ``MMD² = E[k(a,a')] + E[k(b,b')] - 2 E[k(a,b)]`` computed on finite samples.
    """
    a = _to_numpy(samples_a)
    b = _to_numpy(samples_b)
    if bandwidth is None:
        bandwidth = median_heuristic_bandwidth(a, b)
    sigma = float(bandwidth)

    def kernel(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        xx = np.sum(X * X, axis=1, keepdims=True)
        yy = np.sum(Y * Y, axis=1, keepdims=True)
        sq = xx + yy.T - 2.0 * (X @ Y.T)
        return np.exp(-sq / (2.0 * sigma * sigma))

    k_aa = kernel(a, a)
    k_bb = kernel(b, b)
    k_ab = kernel(a, b)

    # Exclude the diagonal for unbiased within-set estimates.
    n_a, n_b = len(a), len(b)
    if n_a > 1:
        np.fill_diagonal(k_aa, 0.0)
        k_aa_mean = np.sum(k_aa) / (n_a * (n_a - 1))
    else:
        k_aa_mean = np.mean(k_aa)
    if n_b > 1:
        np.fill_diagonal(k_bb, 0.0)
        k_bb_mean = np.sum(k_bb) / (n_b * (n_b - 1))
    else:
        k_bb_mean = np.mean(k_bb)

    k_ab_mean = np.mean(k_ab)
    mmd2 = k_aa_mean + k_bb_mean - 2.0 * k_ab_mean
    # Clamp small negative values that can arise from finite samples.
    return float(max(mmd2, 0.0))


def mmd(
    samples_a: Union[torch.Tensor, np.ndarray],
    samples_b: Union[torch.Tensor, np.ndarray],
    bandwidth: Optional[float] = None,
) -> float:
    """Maximum Mean Discrepancy with RBF kernel (alias of :func:`rbf_mmd`).

    Returns the squared MMD; lower values indicate closer distributions.
    """
    return rbf_mmd(samples_a, samples_b, bandwidth=bandwidth)


# ---------------------------------------------------------------------------
# Simulation-Based Calibration
# ---------------------------------------------------------------------------
def simulation_based_calibration(
    prior_sampler: Callable[[int], Union[torch.Tensor, np.ndarray]],
    simulator: Callable[[Union[torch.Tensor, np.ndarray]], Union[torch.Tensor, np.ndarray]],
    posterior_sampler: Callable[[Union[torch.Tensor, np.ndarray], int], Union[torch.Tensor, np.ndarray]],
    n_trials: int,
    n_posterior_samples: int,
    theta_dim: int,
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulation-Based Calibration (SBC) rank computation.

    For each trial, sample a true parameter from the prior, simulate an
    observation, then sample from the approximate posterior. The rank of the
    true parameter among the posterior samples (per dimension) should be
    uniformly distributed for a calibrated posterior.

    Parameters
    ----------
    prior_sampler:
        Callable returning ``(n, theta_dim)`` prior samples.
    simulator:
        Callable mapping ``(n, theta_dim)`` parameters to ``(n, x_dim)`` data.
    posterior_sampler:
        Callable ``(x_obs, n_samples) -> (n_samples, theta_dim)`` posterior draws.
    n_trials:
        Number of calibration trials.
    n_posterior_samples:
        Number of posterior draws per trial.
    theta_dim:
        Dimensionality of the parameter vector.

    Returns
    -------
    ranks:
        Array of shape ``(n_trials, theta_dim)`` with integer ranks in
        ``[0, n_posterior_samples]`` (inclusive).
    coverage:
        Array of shape ``(n_trials, theta_dim)`` with rank / n_posterior_samples,
        i.e. the fraction of posterior samples below the true parameter.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    ranks = np.zeros((n_trials, theta_dim), dtype=np.int64)
    coverage = np.zeros((n_trials, theta_dim), dtype=np.float64)

    for i in range(n_trials):
        theta_true = _to_numpy(prior_sampler(1))
        if theta_true.ndim == 1:
            theta_true = theta_true[None, :]
        x_obs = _to_numpy(simulator(theta_true))
        if x_obs.ndim == 1:
            x_obs = x_obs[None, :]
        # posterior_sampler expects a single observation (theta_dim,) or (1, dim).
        x_obs_single = x_obs[0]
        post = _to_numpy(posterior_sampler(x_obs_single, n_posterior_samples))
        if post.ndim == 1:
            post = post[:, None]
        for d in range(theta_dim):
            rank = int(np.sum(post[:, d] < theta_true[0, d]))
            ranks[i, d] = rank
            coverage[i, d] = rank / float(n_posterior_samples)

    return ranks, coverage


def expected_calibration_error(
    coverage: Union[torch.Tensor, np.ndarray],
    n_bins: int = 10,
) -> float:
    """Expected calibration error from SBC coverage values.

    Bins coverage values into ``n_bins`` intervals and measures the average
    absolute deviation of the empirical cumulative frequency from the diagonal.
    Lower values indicate better calibration.
    """
    cov = np.asarray(_to_numpy(coverage) if not isinstance(coverage, np.ndarray) else coverage).ravel()
    cov = np.clip(cov, 0.0, 1.0 - 1e-12)

    errors = []
    for b in range(n_bins):
        lo = b / float(n_bins)
        hi = (b + 1) / float(n_bins)
        mask = (cov >= lo) & (cov < hi)
        if mask.sum() == 0:
            continue
        empirical = mask.sum() / float(len(cov))
        expected = hi - lo
        errors.append(abs(empirical - expected))
    return float(np.mean(errors)) if errors else float("inf")


# ---------------------------------------------------------------------------
# Posterior Predictive Check
# ---------------------------------------------------------------------------
def posterior_predictive_check(
    simulator: Callable[[Union[torch.Tensor, np.ndarray]], Union[torch.Tensor, np.ndarray]],
    posterior_samples: Union[torch.Tensor, np.ndarray],
    x_obs: Union[torch.Tensor, np.ndarray],
    statistic: Optional[Callable[[Union[torch.Tensor, np.ndarray]], Union[torch.Tensor, np.ndarray]]] = None,
    n_predictive_samples: Optional[int] = None,
) -> dict:
    """Posterior predictive check.

    Simulates data from posterior parameter draws and compares the resulting
    (optionally summarized) predictive samples with the observed data. Returns
    the mean and standard deviation of the predictive summary statistics, the
    standardized distance to the observed summary, and the fraction of
    predictive summaries exceeding that distance.

    Parameters
    ----------
    simulator:
        Maps ``(n, theta_dim)`` parameters to ``(n, x_dim)`` data.
    posterior_samples:
        Approximate posterior draws of shape ``(n, theta_dim)``.
    x_obs:
        Observed data vector of shape ``(x_dim,)``.
    statistic:
        Optional summary-statistic function applied to both predictive and
        observed data.
    n_predictive_samples:
        If provided, subsample the posterior before simulating.

    Returns
    -------
    dict
        With keys ``predictive_mean``, ``predictive_std``, ``observed_summary``,
        ``distance``, and ``p_value``.
    """
    post = _to_numpy(posterior_samples)
    if post.ndim == 1:
        post = post[:, None]
    if n_predictive_samples is not None and len(post) > n_predictive_samples:
        idx = np.random.RandomState(0).choice(len(post), size=n_predictive_samples, replace=False)
        post = post[idx]

    x_pred = _to_numpy(simulator(post))
    if statistic is not None:
        x_pred = _to_numpy(statistic(x_pred))
        x_obs_s = _to_numpy(statistic(np.asarray(_to_numpy(x_obs)).reshape(1, -1)))
    else:
        x_obs_s = np.asarray(_to_numpy(x_obs)).reshape(1, -1)

    if x_pred.ndim == 1:
        x_pred = x_pred[:, None]

    mean = np.mean(x_pred, axis=0)
    std = np.std(x_pred, axis=0) + 1e-12
    z = (x_obs_s - mean) / std
    distance = float(np.sqrt(np.mean(z * z)))

    # Fraction of predictive samples whose standardized norm exceeds the observed.
    pred_z = (x_pred - mean) / std
    pred_norm = np.sqrt(np.sum(pred_z * pred_z, axis=1))
    obs_norm = float(np.sqrt(np.sum(z * z)))
    p_value = float(np.mean(pred_norm >= obs_norm))

    return {
        "predictive_mean": mean,
        "predictive_std": std,
        "observed_summary": x_obs_s,
        "distance": distance,
        "p_value": p_value,
    }
