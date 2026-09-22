"""MLP-based state-space diversity metric for SAPG (Section 6.4, Figure 8).

The paper's second diversity metric trains small feedforward *reconstruction*
networks on batches of environment states collected by each algorithm:

    "We train feedforward networks with small hidden layers on the task of input
     reconstruction on batches of environment states visited by our algorithm and
     PPO during training.  The idea behind this is that if a batch of states has a
     more diverse data distribution then it should be harder to reconstruct the
     distribution using small hidden layers because high diversity implies that the
     distribution is less compressible.  Thus, high training error on a batch of
     states is a strong indicator of diversity in the batch.  As can be observed
     from the plots in Figure-8, we find that training error is consistently higher
     for our method compared to PPO across different hidden layer sizes."

Addendum (authoritative hyperparameters for Figure 8):

    "For figure 8, the neural network was a two layer of the same size (the size is
     shown in the x-axis of the plot).  The activation function used was ReLU,
     trained with Adam optimizer using default hyperparameters from pytorch.
     Each method was trained on 400k state-transitions on an L2 reconstruction
     loss."

Therefore the reconstructed network is ``obs_dim -> w -> w -> obs_dim`` (two
hidden layers of identical width ``w``, the x-axis of Figure 8), ReLU
activations, Adam with PyTorch defaults (lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
weight_decay=0) and the squared-L2 reconstruction loss (MSE) over the batch.

The metric reported per width is the **training error** on the states visited by
the method (higher error => more diverse / less compressible data).  A held-out
split of the *same* state batch is also reported for reference (``test_error``),
but the paper's Figure 8 quantity is the training error.

Usage
-----
>>> from sapg.analysis.diversity_mlp import compare_mlp_diversity
>>> curves = compare_mlp_diversity({"sapg": sapg_states, "ppo": ppo_states})
>>> for width, res_sapg, res_ppo in zip(curves["widths"], curves["sapg"], curves["ppo"]):
...     print(width, res_sapg.train_error, res_ppo.train_error)

The module is importable without PyTorch (a NumPy random-feature ridge-regression
approximation is used as a fallback) and without matplotlib (plotting degrades to
a no-op returning the figure data).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------------------
# Optional dependencies
# --------------------------------------------------------------------------------------

try:  # pragma: no cover - exercised only when torch is installed
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    HAS_TORCH = False

try:  # pragma: no cover
    import numpy as np

    HAS_NUMPY = True
except Exception:  # pragma: no cover
    np = None  # type: ignore
    HAS_NUMPY = False


# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

METHOD_SAPG = "sapg"
METHOD_PPO = "ppo"
METHOD_RANDOM = "random"

#: Default hidden widths swept on the x-axis of Figure 8 (powers of two).
DEFAULT_WIDTHS: Tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128, 256)

#: Paper protocol: 400k state-transitions per method (Addendum).
DEFAULT_MAX_SAMPLES = 400_000

#: Two hidden layers of *equal* size (Addendum).
DEFAULT_NUM_LAYERS = 2

#: Adam defaults from PyTorch (Addendum: "default hyperparameters from pytorch").
DEFAULT_LR = 1e-3
DEFAULT_BETAS = (0.9, 0.999)
DEFAULT_EPS = 1e-8
DEFAULT_WEIGHT_DECAY = 0.0

_LOSS_EPS = 1e-12


# --------------------------------------------------------------------------------------
# Configuration / result containers
# --------------------------------------------------------------------------------------


def _to_tuple(value: Any, default: Optional[Sequence[int]] = None) -> Tuple[int, ...]:
    """Coerce ``value`` into a tuple of ints (falling back to ``default``)."""
    if value is None:
        return tuple(default) if default is not None else tuple(DEFAULT_WIDTHS)
    if isinstance(value, (int, float)):
        return (int(value),)
    out: List[int] = []
    for v in value:
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            continue
    return tuple(out) if out else tuple(default or DEFAULT_WIDTHS)


@dataclass
class MLPConfig:
    """Configuration of the Figure-8 reconstruction-network diversity metric.

    Attributes mirror the Addendum protocol; the only deviations are the training
    budget knobs (epochs / batch size / patience) which the paper leaves implicit.
    """

    widths: Sequence[int] = field(default_factory=lambda: tuple(DEFAULT_WIDTHS))
    num_layers: int = DEFAULT_NUM_LAYERS
    activation: str = "relu"
    learning_rate: float = DEFAULT_LR
    betas: Tuple[float, float] = DEFAULT_BETAS
    eps: float = DEFAULT_EPS
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    loss: str = "mse"  # L2 reconstruction loss
    batch_size: int = 4096
    num_epochs: int = 20
    max_samples: int = DEFAULT_MAX_SAMPLES
    train_ratio: float = 0.9
    standardize: bool = True
    center: bool = True
    shuffle: bool = True
    seed: int = 0
    device: str = "cpu"
    backend: str = "auto"  # "auto" | "torch" | "numpy"
    log_every: int = 0
    early_stop_patience: int = 0
    min_delta: float = 1e-6
    eval_every: int = 1

    # ------------------------------------------------------------------ helpers
    @property
    def width_list(self) -> Tuple[int, ...]:
        return _to_tuple(self.widths)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["widths"] = list(self.width_list)
        d["betas"] = list(self.betas)
        return d

    @classmethod
    def from_any(cls, config: Any = None, **overrides: Any) -> "MLPConfig":
        """Build a config from another config object / dict / explicit overrides."""
        cfg = cls()
        if config is not None:
            if isinstance(config, MLPConfig):
                cfg = cls(**config.to_dict())
            elif isinstance(config, dict):
                cfg = cls.from_dict(config)
            else:
                data: Dict[str, Any] = {}
                for f in cls.__dataclass_fields__:  # type: ignore[attr-defined]
                    if hasattr(config, f):
                        data[f] = getattr(config, f)
                # common aliases
                for alias, target in (
                    ("mlp_widths", "widths"),
                    ("hidden_sizes", "widths"),
                    ("diversity_widths", "widths"),
                    ("lr", "learning_rate"),
                    ("num_seeds", "seed"),
                    ("device_type", "device"),
                ):
                    if hasattr(config, alias) and target not in data:
                        data[target] = getattr(config, alias)
                cfg = cls.from_dict(data)
        if overrides:
            cfg = cfg.replace(**overrides)
        return cfg

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "MLPConfig":
        cfg = cls()
        if not data:
            return cfg
        unknown = set(data) - set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        clean = {k: v for k, v in data.items() if k not in unknown}
        if "widths" in clean:
            clean["widths"] = _to_tuple(clean["widths"])
        if "betas" in clean and clean["betas"] is not None:
            clean["betas"] = tuple(float(b) for b in clean["betas"])
        return cfg.replace(**clean)

    def replace(self, **overrides: Any) -> "MLPConfig":
        data = self.to_dict()
        data["widths"] = tuple(self.width_list)
        data["betas"] = tuple(self.betas)
        for key, value in overrides.items():
            if key not in self.__dataclass_fields__:  # type: ignore[attr-defined]
                continue
            if key == "widths":
                value = _to_tuple(value)
            elif key == "betas" and value is not None:
                value = tuple(float(b) for b in value)
            data[key] = value
        return MLPConfig(
            widths=tuple(data["widths"]),
            num_layers=int(data["num_layers"]),
            activation=data["activation"],
            learning_rate=float(data["learning_rate"]),
            betas=tuple(data["betas"]),
            eps=float(data["eps"]),
            weight_decay=float(data["weight_decay"]),
            loss=data["loss"],
            batch_size=int(data["batch_size"]),
            num_epochs=int(data["num_epochs"]),
            max_samples=int(data["max_samples"]),
            train_ratio=float(data["train_ratio"]),
            standardize=bool(data["standardize"]),
            center=bool(data["center"]),
            shuffle=bool(data["shuffle"]),
            seed=int(data["seed"]),
            device=data["device"],
            backend=data["backend"],
            log_every=int(data["log_every"]),
            early_stop_patience=int(data["early_stop_patience"]),
            min_delta=float(data["min_delta"]),
            eval_every=int(data["eval_every"]),
        )


@dataclass
class MLPResult:
    """Result of training one reconstruction network of a given hidden width."""

    name: str
    width: int
    train_error: float
    test_error: float = float("nan")
    state_dim: int = 0
    num_train_samples: int = 0
    num_test_samples: int = 0
    epochs: int = 0
    history: List[float] = field(default_factory=list)
    num_layers: int = DEFAULT_NUM_LAYERS
    activation: str = "relu"
    optimizer: str = "adam"
    learning_rate: float = DEFAULT_LR
    batch_size: int = 4096
    backend: str = "torch"
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ helpers
    def error_at(self, width: int) -> float:
        raise NotImplementedError("error_at is only defined for curves, not single widths")

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["final_train_error"] = self.train_error
        d["final_test_error"] = self.test_error
        d.pop("history", None)
        return d

    def __float__(self) -> float:  # pragma: no cover - convenience
        return float(self.train_error)


# --------------------------------------------------------------------------------------
# Torch reconstruction network
# --------------------------------------------------------------------------------------

if HAS_TORCH:  # pragma: no cover - requires torch

    class MLPReconstructor(nn.Module):
        """Two hidden layers of identical width, ReLU, reconstructing the input.

        ``obs_dim -> w -> w -> obs_dim`` per the Addendum description of Figure 8.
        """

        def __init__(
            self,
            input_dim: int,
            width: int,
            num_layers: int = DEFAULT_NUM_LAYERS,
            output_dim: Optional[int] = None,
            activation: str = "relu",
        ) -> None:
            super().__init__()
            self.input_dim = int(input_dim)
            self.width = int(width)
            self.num_layers = max(int(num_layers), 1)
            self.output_dim = int(output_dim if output_dim is not None else input_dim)
            layers: List[nn.Module] = []
            in_dim = self.input_dim
            for _ in range(self.num_layers):
                layers.append(nn.Linear(in_dim, self.width))
                if activation is None or str(activation).lower() == "identity":
                    pass
                elif str(activation).lower() == "relu":
                    layers.append(nn.ReLU())
                elif str(activation).lower() == "elu":
                    layers.append(nn.ELU())
                elif str(activation).lower() == "tanh":
                    layers.append(nn.Tanh())
                elif str(activation).lower() in ("gelu",):
                    layers.append(nn.GELU())
                else:
                    layers.append(nn.ReLU())
                in_dim = self.width
            layers.append(nn.Linear(in_dim, self.output_dim))
            self.net = nn.Sequential(*layers)
            self._init_weights()

        def _init_weights(self) -> None:
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def forward(self, x):  # noqa: D401 - simple pass-through
            return self.net(x)

else:

    class MLPReconstructor:  # type: ignore[no-redef]
        """Placeholder used when torch is unavailable (numpy backend is used instead)."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover
            raise RuntimeError(
                "MLPReconstructor requires PyTorch; install torch or run with backend='numpy'."
            )


# --------------------------------------------------------------------------------------
# Parameter-free reference: tiny hand-rolled MLP (numpy) used as a fallback
# --------------------------------------------------------------------------------------


class _NumpyRandomFeatureReconstructor:  # pragma: no cover - fallback path
    """Random-feature + ridge-regression stand-in for the tiny reconstruction net.

    Used only when PyTorch is unavailable.  Two random ReLU layers of width ``w``
    are followed by a closed-form ridge fit of the output layer.  Training error
    still increases with data diversity (the random features span a fixed, small
    subspace), so the qualitative Figure-8 conclusion is preserved.
    """

    def __init__(self, input_dim: int, width: int, num_layers: int = 2, seed: int = 0, ridge: float = 1e-6):
        self.input_dim = int(input_dim)
        self.width = int(width)
        self.num_layers = max(int(num_layers), 1)
        rng = np.random.RandomState(seed)
        self.W1 = rng.normal(0.0, 1.0 / math.sqrt(max(self.input_dim, 1)), size=(self.input_dim, self.width))
        self.b1 = np.zeros(self.width, dtype=np.float64)
        self.W2 = rng.normal(0.0, 1.0 / math.sqrt(max(self.width, 1)), size=(self.width, self.width))
        self.b2 = np.zeros(self.width, dtype=np.float64)
        self.Wout = np.zeros((self.width, self.input_dim), dtype=np.float64)
        self.bout = np.zeros(self.input_dim, dtype=np.float64)
        self.ridge = float(ridge)

    def _features(self, X: np.ndarray) -> np.ndarray:
        H = np.maximum(X @ self.W1 + self.b1, 0.0)
        for _ in range(self.num_layers - 1):
            H = np.maximum(H @ self.W2 + self.b2, 0.0)
        return H

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        y = X if y is None else y
        H = self._features(X)
        A = H.T @ H + self.ridge * np.eye(H.shape[1], dtype=np.float64)
        B = H.T @ y
        W = np.linalg.solve(A, B)
        self.Wout = W
        self.bout = (y - H @ W).mean(axis=0) if y.shape[0] else self.bout

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._features(X) @ self.Wout + self.bout

    def mse(self, X: np.ndarray) -> float:
        pred = self.predict(X)
        return float(np.mean((pred - X) ** 2))


# --------------------------------------------------------------------------------------
# Core metric
# --------------------------------------------------------------------------------------


def _as_matrix(states: Any) -> Any:
    """Convert a state batch (tensor / ndarray / dict / buffer) into a 2-D matrix."""
    # dict / buffer style containers
    if isinstance(states, dict):
        for key in ("obs", "observations", "states", "state", "data"):
            if key in states:
                return _as_matrix(states[key])
        return np.asarray(list(states.values())[0]) if states else np.zeros((0, 0))

    # RolloutBuffer-like object
    for key in ("obs", "observations", "states"):
        if hasattr(states, key):
            value = getattr(states, key)
            if value is not None:
                return _as_matrix(value)
    if hasattr(states, "data") and isinstance(getattr(states, "data"), dict):
        return _as_matrix(getattr(states, "data"))
    if hasattr(states, "get") and callable(getattr(states, "get")):
        try:
            value = states.get("obs", None)
        except Exception:
            value = None
        if value is not None:
            return _as_matrix(value)

    if HAS_TORCH and isinstance(states, torch.Tensor):
        mat = states.reshape(-1, states.shape[-1]).detach().cpu().float().numpy()
        return mat
    if HAS_NUMPY and isinstance(states, np.ndarray):
        arr = np.asarray(states, dtype=np.float64)
        return arr.reshape(-1, arr.shape[-1]) if arr.ndim > 2 else arr
    if isinstance(states, (list, tuple)):
        if HAS_NUMPY:
            arr = np.asarray(states, dtype=np.float64)
            return arr.reshape(-1, arr.shape[-1]) if arr.ndim > 2 else arr
        return states
    if HAS_NUMPY:
        arr = np.asarray(states, dtype=np.float64)
        return arr.reshape(-1, arr.shape[-1]) if arr.ndim > 2 else arr
    return states  # pragma: no cover


def _to_numpy(states: Any) -> "np.ndarray":
    mat = _as_matrix(states)
    if HAS_NUMPY and isinstance(mat, np.ndarray):
        return np.asarray(mat, dtype=np.float64)
    if HAS_TORCH and torch is not None and isinstance(mat, torch.Tensor):
        return mat.detach().cpu().float().numpy().astype("float64")
    if HAS_NUMPY:
        return np.asarray(mat, dtype=np.float64)
    raise RuntimeError("NumPy is required to evaluate the MLP diversity metric.")


def _subsample(X: "np.ndarray", max_samples: Optional[int], seed: int) -> "np.ndarray":
    if max_samples is None or max_samples <= 0 or X.shape[0] <= max_samples:
        return X
    rng = np.random.RandomState(seed)
    idx = rng.choice(X.shape[0], size=int(max_samples), replace=False)
    idx.sort()
    return X[idx]


def _standardize(
    X_train: "np.ndarray",
    X_test: "np.ndarray",
    center: bool = True,
    scale: bool = True,
) -> Tuple["np.ndarray", "np.ndarray", Dict[str, Any]]:
    stats: Dict[str, Any] = {}
    if not center:
        return X_train, X_test, stats
    mean = X_train.mean(axis=0)
    stats["mean"] = mean
    Xtr = X_train - mean
    Xte = X_test - mean
    if scale:
        std = X_train.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        stats["std"] = std
        Xtr = Xtr / std
        Xte = Xte / std
    return Xtr, Xte, stats


def _split(
    X: "np.ndarray", train_ratio: float, seed: int
) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray", "np.ndarray"]:
    n = X.shape[0]
    n_train = int(round(max(min(train_ratio, 1.0), 0.0) * n))
    n_train = max(min(n_train, n), 1)
    rng = np.random.RandomState(seed + 12345)
    perm = rng.permutation(n)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:] if n_train < n else perm[:0]
    return X[train_idx], X[test_idx], train_idx, test_idx


def _resolve_backend(config: MLPConfig) -> str:
    backend = str(config.backend).lower()
    if backend in ("torch", "pytorch") and HAS_TORCH:
        return "torch"
    if backend in ("np", "numpy") and HAS_NUMPY:
        return "numpy"
    if backend in ("torch", "pytorch", "np", "numpy"):
        # explicitly requested backend unavailable -> best available
        return "torch" if HAS_TORCH else ("numpy" if HAS_NUMPY else "numpy")
    if HAS_TORCH:
        return "torch"
    return "numpy"


def train_reconstruction_mlp(
    states: Any,
    width: int = 16,
    config: Any = None,
    name: str = "policy",
    test_states: Any = None,
    **overrides: Any,
) -> MLPResult:
    """Train one reconstruction network of hidden ``width`` on ``states``.

    Implements the Figure-8 protocol: two hidden layers of equal width, ReLU
    activation, Adam with PyTorch defaults, L2 (MSE) reconstruction loss, at most
    400k state-transitions.

    Args:
        states: state batch (tensor / ndarray / dict / RolloutBuffer / buffer-like).
        width: hidden width ``w`` (x-axis of Figure 8).
        config: optional :class:`MLPConfig` (or duck-typed config).
        name: label for the batch's method (e.g. ``"sapg"`` / ``"ppo"``).
        test_states: optional separate held-out batch; otherwise a split of
            ``states`` is used.
        **overrides: config overrides.

    Returns:
        :class:`MLPResult` with ``train_error`` (the paper's Figure-8 quantity).
    """
    cfg = MLPConfig.from_any(config, **overrides)
    width = int(width)
    backend = _resolve_backend(cfg)

    X_all = _to_numpy(states)
    X_all = _subsample(X_all, cfg.max_samples, cfg.seed)
    state_dim = int(X_all.shape[1]) if X_all.ndim == 2 and X_all.shape[1] else 1
    if X_all.ndim != 2 or X_all.shape[0] == 0:
        return MLPResult(
            name=str(name),
            width=width,
            train_error=float("nan"),
            state_dim=state_dim,
            backend=backend,
        )

    if test_states is not None:
        X_test = _subsample(_to_numpy(test_states), cfg.max_samples, cfg.seed + 1)
        X_train = X_all
    else:
        X_train, X_test, _, _ = _split(X_all, cfg.train_ratio, cfg.seed)

    if X_test is None or len(X_test) == 0:
        X_test = X_train

    Xtr, Xte, stats = _standardize(X_train, X_test, cfg.center, cfg.standardize)

    history: List[float] = []
    epochs = 0

    if backend == "torch":
        torch.manual_seed(cfg.seed)
        device = cfg.device if (torch is not None and hasattr(torch, "device")) else "cpu"
        try:
            device_obj = torch.device(device)
        except Exception:  # pragma: no cover
            device_obj = torch.device("cpu")
        Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32, device=device_obj)
        Xte_t = torch.as_tensor(Xte, dtype=torch.float32, device=device_obj)

        model = MLPReconstructor(
            input_dim=state_dim,
            width=width,
            num_layers=cfg.num_layers,
            output_dim=state_dim,
            activation=cfg.activation,
        ).to(device_obj)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(cfg.learning_rate),
            betas=(float(cfg.betas[0]), float(cfg.betas[1])),
            eps=float(cfg.eps),
            weight_decay=float(cfg.weight_decay),
        )
        loss_fn = torch.nn.MSELoss()

        n = int(Xtr_t.shape[0])
        batch_size = max(min(int(cfg.batch_size), n), 1)
        best = float("inf")
        best_state = None
        bad_epochs = 0

        for epoch in range(max(int(cfg.num_epochs), 1)):
            model.train()
            if cfg.shuffle:
                perm = torch.randperm(n, device=device_obj)
            else:
                perm = torch.arange(n, device=device_obj)
            total = 0.0
            count = 0
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                xb = Xtr_t[idx]
                pred = model(xb)
                loss = loss_fn(pred, xb)  # L2 reconstruction loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.detach().item()) * int(xb.shape[0])
                count += int(xb.shape[0])
            epoch_loss = total / max(count, 1)
            history.append(epoch_loss)
            epochs = epoch + 1

            if cfg.early_stop_patience > 0:
                if epoch_loss < best - cfg.min_delta:
                    best = epoch_loss
                    bad_epochs = 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    bad_epochs += 1
                    if bad_epochs >= int(cfg.early_stop_patience):
                        break
            if cfg.log_every and (epoch + 1) % int(cfg.log_every) == 0:
                print(f"[mlp-diversity:{name}] width={width} epoch={epoch + 1} loss={epoch_loss:.6f}")

        if best_state is not None and cfg.early_stop_patience > 0:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            train_error = float(loss_fn(model(Xtr_t), Xtr_t).item())
            test_error = float(loss_fn(model(Xte_t), Xte_t).item())

        metadata: Dict[str, Any] = {"standardized": bool(cfg.standardize), "center": bool(cfg.center)}
        metadata.update({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in stats.items()})

        return MLPResult(
            name=str(name),
            width=int(width),
            train_error=train_error,
            test_error=test_error,
            state_dim=state_dim,
            num_train_samples=int(Xtr.shape[0]),
            num_test_samples=int(Xte.shape[0]),
            epochs=epochs,
            history=history,
            num_layers=int(cfg.num_layers),
            activation=str(cfg.activation),
            optimizer="adam",
            learning_rate=float(cfg.learning_rate),
            batch_size=int(cfg.batch_size),
            backend="torch",
            metadata=metadata,
        )

    # ------------------------------------------------------------------ numpy path
    model_np = _NumpyRandomFeatureReconstructor(
        input_dim=state_dim, width=width, num_layers=cfg.num_layers, seed=cfg.seed
    )
    model_np.fit(Xtr)
    train_error = model_np.mse(Xtr)
    test_error = model_np.mse(Xte)
    history.append(train_error)
    metadata = {
        "standardized": bool(cfg.standardize),
        "center": bool(cfg.center),
        "backend_note": "numpy random-feature ridge approximation (torch unavailable)",
    }
    return MLPResult(
        name=str(name),
        width=int(width),
        train_error=train_error,
        test_error=test_error,
        state_dim=state_dim,
        num_train_samples=int(Xtr.shape[0]),
        num_test_samples=int(Xte.shape[0]),
        epochs=1,
        history=history,
        num_layers=int(cfg.num_layers),
        activation=str(cfg.activation),
        optimizer="ridge",
        learning_rate=float(cfg.learning_rate),
        batch_size=int(cfg.batch_size),
        backend="numpy",
        metadata=metadata,
    )


def mlp_reconstruction_error(
    states: Any,
    width: int = 16,
    config: Any = None,
    name: str = "policy",
    **overrides: Any,
) -> float:
    """Convenience: training error of a width-``w`` reconstruction network."""
    return float(
        train_reconstruction_mlp(states, width=width, config=config, name=name, **overrides).train_error
    )


def mlp_reconstruction_curve(
    states: Any,
    widths: Optional[Sequence[int]] = None,
    config: Any = None,
    name: str = "policy",
    test_states: Any = None,
    **overrides: Any,
) -> List[MLPResult]:
    """Full Figure-8 style curve: one :class:`MLPResult` per hidden width."""
    cfg = MLPConfig.from_any(config, **overrides)
    width_list = _to_tuple(widths, default=cfg.width_list)
    return [
        train_reconstruction_mlp(
            states,
            width=int(w),
            config=cfg,
            name=name,
            test_states=test_states,
        )
        for w in width_list
    ]


class MLPDiversity:
    """Diversity metric object wrapping a batch of visited states (Figure 8).

    Attributes:
        states: the (possibly subsampled / standardized) state matrix used.
        state_dim: observation dimensionality.
        num_samples: number of state transitions retained.
        backend: ``"torch"`` or ``"numpy"``.
    """

    def __init__(self, states: Any, config: Any = None, name: str = "policy", **overrides: Any) -> None:
        self.config = MLPConfig.from_any(config, **overrides)
        self.name = str(name)
        X = _to_numpy(states)
        X = _subsample(X, self.config.max_samples, self.config.seed)
        self.states = X
        self.state_dim = int(X.shape[1]) if X.ndim == 2 and X.shape[1] else 1
        self.num_samples = int(X.shape[0]) if X.ndim == 2 else 0
        self.backend = _resolve_backend(self.config)
        self.results: List[MLPResult] = []

    # ------------------------------------------------------------------ API
    def train(self, width: int) -> MLPResult:
        """Train a reconstruction network of hidden ``width`` and return its result."""
        result = train_reconstruction_mlp(self.states, width=width, config=self.config, name=self.name)
        self.results = [r for r in self.results if int(r.width) != int(width)]
        self.results.append(result)
        self.results.sort(key=lambda r: int(r.width))
        return result

    def training_error(self, width: int) -> float:
        return float(self.train(width).train_error)

    def curve(self, widths: Optional[Sequence[int]] = None) -> List[MLPResult]:
        """Training-error curve over hidden widths (Figure 8)."""
        width_list = _to_tuple(widths, default=self.config.width_list)
        self.results = [
            train_reconstruction_mlp(self.states, width=int(w), config=self.config, name=self.name)
            for w in width_list
        ]
        return self.results

    # ------------------------------------------------------------------ summaries
    @property
    def widths(self) -> List[int]:
        return [int(r.width) for r in self.results]

    @property
    def errors(self) -> List[float]:
        return [float(r.train_error) for r in self.results]

    @property
    def test_errors(self) -> List[float]:
        return [float(r.test_error) for r in self.results]

    def mean_error(self) -> float:
        errs = [e for e in self.errors if not math.isnan(e)]
        return float(sum(errs) / len(errs)) if errs else float("nan")

    def summary(self) -> Dict[str, Any]:
        widths = self.widths
        errors = self.errors
        return {
            "name": self.name,
            "state_dim": self.state_dim,
            "num_samples": self.num_samples,
            "backend": self.backend,
            "widths": widths,
            "errors": errors,
            "mean_error": self.mean_error(),
            "max_error": max(errors) if errors else float("nan"),
            "min_error": min(errors) if errors else float("nan"),
        }


# --------------------------------------------------------------------------------------
# Multi-method comparison
# --------------------------------------------------------------------------------------


def compare_mlp_diversity(
    state_batches: Dict[str, Any],
    widths: Optional[Sequence[int]] = None,
    config: Any = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Train Figure-8 curves for several methods (e.g. SAPG vs PPO vs random).

    Returns a dict with key ``"widths"`` plus one list of :class:`MLPResult` per
    method name, and a ``"summary"`` entry holding mean training errors.
    """
    cfg = MLPConfig.from_any(config, **overrides)
    width_list = _to_tuple(widths, default=cfg.width_list)
    out: Dict[str, Any] = {"widths": list(width_list)}
    summaries: Dict[str, Any] = {}
    for name, states in state_batches.items():
        curves = mlp_reconstruction_curve(states, widths=width_list, config=cfg, name=name)
        out[str(name)] = curves
        errs = [float(r.train_error) for r in curves]
        summaries[str(name)] = {
            "mean_error": float(sum(errs) / len(errs)) if errs else float("nan"),
            "errors": errs,
            "max_error": max(errs) if errs else float("nan"),
        }
    out["summary"] = summaries
    return out


def diversity_ranking(curves: Dict[str, Any], metric: str = "mean_error") -> List[Tuple[str, float]]:
    """Rank methods by diversity (higher reconstruction training error == more diverse)."""
    summary = curves.get("summary", {}) if isinstance(curves, dict) else {}
    rows: List[Tuple[str, float]] = []
    for name, data in summary.items():
        value = data.get(metric, float("nan")) if isinstance(data, dict) else float(data)
        rows.append((str(name), float(value)))
    rows.sort(key=lambda item: item[1], reverse=True)
    return rows


def error_increase_ratio(curves: Dict[str, Any], reference: str = METHOD_PPO, method: str = METHOD_SAPG) -> float:
    """Mean error ratio ``method/reference`` (>1 means the method is more diverse)."""
    summary = curves.get("summary", {}) if isinstance(curves, dict) else {}
    ref = summary.get(reference, {}).get("mean_error", float("nan")) if summary else float("nan")
    cur = summary.get(method, {}).get("mean_error", float("nan")) if summary else float("nan")
    if not ref or math.isnan(ref) or math.isnan(cur):
        return float("nan")
    return float(cur / ref)


def curve_ordering(curves: List[MLPResult]) -> Dict[int, float]:
    """Map width -> training error (for plotting / monotonicity checks)."""
    return {int(r.width): float(r.train_error) for r in curves}


# --------------------------------------------------------------------------------------
# State collection helpers (shared protocol with diversity_pca)
# --------------------------------------------------------------------------------------


def states_from_buffer(buffer: Any, key: str = "obs", max_samples: Optional[int] = None, seed: int = 0) -> Any:
    """Extract a flat state batch from a RolloutBuffer / dict-like container."""
    value = None
    if isinstance(buffer, dict):
        for candidate in (key, "obs", "observations", "states"):
            if candidate in buffer:
                value = buffer[candidate]
                break
    else:
        for candidate in (key, "obs", "observations", "states"):
            if hasattr(buffer, candidate):
                value = getattr(buffer, candidate)
                break
        if value is None and hasattr(buffer, "data") and isinstance(getattr(buffer, "data"), dict):
            for candidate in (key, "obs", "observations", "states"):
                if candidate in getattr(buffer, "data"):
                    value = getattr(buffer, "data")[candidate]
                    break
        if value is None and hasattr(buffer, "get"):
            try:
                value = buffer.get(key, None)
            except Exception:
                value = None
    if value is None:
        return _to_numpy(buffer)
    mat = _to_numpy(value)
    return _subsample(mat, max_samples, seed)


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
    """Roll ``policy`` in ``env`` for ``num_steps`` and return visited states.

    Mirrors :func:`sapg.algorithms.rollout.collect_on_policy` but only records
    observations; policy API is duck-typed (``act`` with optional ``phi`` /
    ``policy_index`` / ``deterministic`` / ``hidden_state`` kwargs).
    """
    states: List[Any] = []
    hidden_state = None
    if obs is None:
        obs = env.reset()
    for _ in range(int(num_steps)):
        if HAS_TORCH and torch is not None and isinstance(obs, torch.Tensor):
            states.append(obs.detach().cpu().clone())
        elif HAS_NUMPY and isinstance(obs, np.ndarray):
            states.append(np.array(obs, copy=True))
        else:  # pragma: no cover
            states.append(obs)

        kwargs: Dict[str, Any] = {"deterministic": deterministic}
        if phi is not None:
            kwargs["phi"] = phi
        if policy_index is not None:
            kwargs["policy_index"] = policy_index
        if hidden_state is not None:
            kwargs["hidden_state"] = hidden_state
        try:
            out = policy.act(obs, **kwargs)
        except TypeError:
            out = policy.act(obs)
        if isinstance(out, dict):
            hidden_state = out.get("hidden_state", hidden_state)
            actions = out.get("actions", out.get("action"))
        else:  # pragma: no cover
            actions = out
        step_out = env.step(actions)
        obs = step_out[0] if isinstance(step_out, (tuple, list)) else step_out

    if not states:  # pragma: no cover
        return _to_numpy(obs)
    if HAS_TORCH and torch is not None and isinstance(states[0], torch.Tensor):
        stacked = torch.cat([s.reshape(-1, s.shape[-1]) for s in states], dim=0)
        mat = stacked.detach().cpu().float().numpy()
    else:  # pragma: no cover
        mat = np.concatenate([np.asarray(s).reshape(-1, np.asarray(s).shape[-1]) for s in states], axis=0)
    return _subsample(mat, max_samples, 0)


# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------


def plot_mlp_diversity(
    curves: Dict[str, Any],
    path: Optional[str] = None,
    title: str = "State diversity (MLP reconstruction training error)",
    log_x: bool = True,
    log_y: bool = False,
    show: bool = False,
) -> Any:
    """Render a Figure-8 style plot: training error vs hidden width, per method."""
    try:  # pragma: no cover - optional dependency
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    width_list = curves.get("widths") if isinstance(curves, dict) else None
    for name, value in (curves.items() if isinstance(curves, dict) else []):
        if name in ("widths", "summary"):
            continue
        widths = [int(r.width) for r in value]
        errors = [float(r.train_error) for r in value]
        ax.plot(widths or width_list, errors, marker="o", label=str(name))
    if log_x:
        ax.set_xscale("log", base=2)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("hidden layer width")
    ax.set_ylabel("reconstruction training error (MSE)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig.savefig(path, dpi=150)
    if show:  # pragma: no cover
        plt.show()
    plt.close(fig)
    return fig


# --------------------------------------------------------------------------------------
# CLI smoke test
# --------------------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover
    """Smoke run: compare a low-dimensional and a high-dimensional state batch."""
    import argparse

    parser = argparse.ArgumentParser(description="MLP state-diversity metric (Figure 8)")
    parser.add_argument("--num-samples", type=int, default=20000)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--widths", type=int, nargs="*", default=[2, 4, 8, 16])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(list(argv) if argv is not None else None)

    rng = np.random.RandomState(args.seed)
    low = rng.normal(size=(args.num_samples, args.state_dim)) @ rng.normal(size=(2, args.state_dim))
    high = rng.normal(size=(args.num_samples, args.state_dim))

    curves = compare_mlp_diversity(
        {"structured": low, "diverse": high},
        widths=args.widths,
        config=MLPConfig(num_epochs=args.epochs, max_samples=args.num_samples, seed=args.seed),
    )
    print("width  structured  diverse")
    for w, a, b in zip(curves["widths"], curves["structured"], curves["diverse"]):
        print(f"{w:5d}  {a.train_error:10.5f}  {b.train_error:10.5f}")
    print("ranking (higher error = more diverse):", diversity_ranking(curves))
    return 0


__all__ = [
    "MLPConfig",
    "MLPResult",
    "MLPDiversity",
    "MLPReconstructor",
    "train_reconstruction_mlp",
    "mlp_reconstruction_error",
    "mlp_reconstruction_curve",
    "compare_mlp_diversity",
    "diversity_ranking",
    "error_increase_ratio",
    "curve_ordering",
    "states_from_buffer",
    "collect_states",
    "plot_mlp_diversity",
    "main",
    "METHOD_SAPG",
    "METHOD_PPO",
    "METHOD_RANDOM",
    "DEFAULT_WIDTHS",
    "DEFAULT_MAX_SAMPLES",
    "DEFAULT_NUM_LAYERS",
    "HAS_TORCH",
    "HAS_NUMPY",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
