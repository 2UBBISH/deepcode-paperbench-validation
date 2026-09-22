"""Toxicity probe ``W_Toxic`` for GPT2.

Reproduction of Section 3.1 of *"A Mechanistic Understanding of Alignment
Algorithms: A Case Study on DPO and Toxicity"*.

The paper trains a linear probe on a binary toxicity classification task
(Jigsaw, 561,808 comments, 90:10 split) using the residual stream of the **last
layer, averaged across all timesteps** :math:`\\overline{\\mathbf{x}}^{L-1}`::

    P(Toxic | x) = softmax(W_Toxic @ x),        W_Toxic in R^{d x 2}

Our probe reaches ~94% accuracy on the validation split.  The column
``W_Toxic[:, 1]`` is the *toxic direction* and is used everywhere in the rest of
the reproduction (cosine-similarity search for ``MLP.v_Toxic``, residual-stream
subtraction, PPLM attribute classifier, un-alignment).

Missing paper details (documented defaults, see plan):
  * optimizer: AdamW, lr = 1e-3, batch size 256, up to 20 epochs with early
    stopping on validation accuracy.

The module is import-cheap: ``torch``/``transformers`` are imported lazily
through :mod:`src.model_utils`.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

PROBE_PATH = "artifacts/probe/w_toxic.pt"
PROBE_JSON_PATH = "artifacts/probe/w_toxic.json"

N_TOXIC_CLASSES = 2
TOXIC_INDEX = 1
NON_TOXIC_INDEX = 0

DEFAULT_LAYER = -1  # last transformer layer => x^{L-1}
DEFAULT_POSITION = "block_out"  # residual stream after the last MLP block
DEFAULT_BATCH_SIZE = 256
DEFAULT_MAX_LENGTH = 128
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_EPOCHS = 20
DEFAULT_PATIENCE = 3
DEFAULT_TRAIN_BATCH_SIZE = 256
TARGET_VALID_ACCURACY = 0.94


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _to_tensor(x):
    import torch

    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(np.asarray(x))


def _as_numpy(x) -> np.ndarray:
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def _pool_states(states, attention_mask=None, pooling: str = "mean"):
    """Pool ``[batch, seq, d]`` hidden states to ``[batch, d]``.

    ``pooling="mean"`` implements the paper's "averaged across all timesteps"
    (mask aware); ``"last"`` keeps the final position.
    """

    if states.dim() == 2:
        return states
    if pooling == "last":
        if attention_mask is None:
            return states[:, -1, :]
        lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
        idx = lengths.view(-1, 1, 1).expand(-1, 1, states.size(-1))
        return states.gather(1, idx).squeeze(1)
    if attention_mask is None:
        return states.mean(dim=1)
    mask = attention_mask.unsqueeze(-1).to(states.dtype)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return (states * mask).sum(dim=1) / denom


def _state_at(capture, layer: int, kind: str = DEFAULT_POSITION):
    """Fetch ``[batch, seq, d]`` states for ``layer`` from a capture object."""

    getter_name = "get_block_out" if kind in ("block_out", "out") else "get_mid"
    getter = getattr(capture, getter_name, None)
    if callable(getter):
        try:
            return getter(layer)
        except Exception:
            pass
    container = getattr(capture, "block_out" if kind in ("block_out", "out") else "mid", None)
    if container is None:
        raise AttributeError(f"residual capture has no states for kind={kind!r}")
    if isinstance(container, dict):
        return container[layer]
    return container[layer]


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #


@dataclass
class ProbeTrainResult:
    """Everything produced by :func:`train_probe`."""

    probe: "ToxicityProbe"
    train_accuracy: float = float("nan")
    valid_accuracy: float = float("nan")
    valid_balanced_accuracy: float = float("nan")
    valid_f1: float = float("nan")
    valid_auc: float = float("nan")
    valid_loss: float = float("nan")
    train_loss: float = float("nan")
    epochs_run: int = 0
    n_train: int = 0
    n_valid: int = 0
    history: List[Dict[str, float]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    # -- convenience ------------------------------------------------------
    @property
    def accuracy(self) -> float:
        return self.valid_accuracy

    @property
    def W(self):
        return self.probe.W

    @property
    def toxic_direction(self):
        return self.probe.toxic_direction

    def summary(self) -> Dict[str, Any]:
        return {
            "model_name": self.probe.model_name,
            "layer": self.probe.layer,
            "d_model": self.probe.d_model,
            "train_accuracy": float(self.train_accuracy),
            "valid_accuracy": float(self.valid_accuracy),
            "valid_balanced_accuracy": float(self.valid_balanced_accuracy),
            "valid_f1": float(self.valid_f1),
            "valid_auc": float(self.valid_auc),
            "valid_loss": float(self.valid_loss),
            "train_loss": float(self.train_loss),
            "epochs_run": int(self.epochs_run),
            "n_train": int(self.n_train),
            "n_valid": int(self.n_valid),
            "target_valid_accuracy": TARGET_VALID_ACCURACY,
            "meta": dict(self.meta),
        }

    def to_dict(self) -> Dict[str, Any]:
        return self.summary()


# --------------------------------------------------------------------------- #
# Probe
# --------------------------------------------------------------------------- #


class ToxicityProbe:
    """Linear toxicity probe ``softmax(W_Toxic @ x)``.

    ``W`` has shape ``[d_model, 2]``: column 0 is the non-toxic class and
    column 1 the toxic class, exactly as in the paper.  All cosine-similarity
    computations in the reproduction use :attr:`toxic_direction`
    (``W[:, 1]``).
    """

    def __init__(
        self,
        W,
        bias=None,
        model_name: str = "gpt2",
        layer: int = DEFAULT_LAYER,
        n_layers: int = 24,
        position: str = DEFAULT_POSITION,
        pooling: str = "mean",
        valid_accuracy: float = float("nan"),
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        import torch

        self.W = torch.as_tensor(W).float()
        if self.W.dim() != 2:
            raise ValueError(f"W_Toxic must be 2-D [d_model, 2], got shape {tuple(self.W.shape)}")
        if self.W.shape[1] != N_TOXIC_CLASSES:
            if self.W.shape[0] == N_TOXIC_CLASSES:
                # tolerate a [2, d_model] transpose provided by external code
                self.W = self.W.t().contiguous()
            else:
                raise ValueError(
                    f"W_Toxic must be [d_model, 2] (or [2, d_model]); got {tuple(self.W.shape)}"
                )
        self.bias = None
        if bias is not None:
            self.bias = torch.as_tensor(bias).float().reshape(-1)
        self.model_name = model_name
        self.layer = int(layer)
        self.n_layers = int(n_layers)
        self.position = position
        self.pooling = pooling
        self.valid_accuracy = float(valid_accuracy)
        self.meta: Dict[str, Any] = dict(meta or {})

    # -- shapes -----------------------------------------------------------
    @property
    def d_model(self) -> int:
        return int(self.W.shape[0])

    @property
    def toxic_direction(self):
        """``W_Toxic[:, 1]`` -- the toxic direction used everywhere."""
        return self.W[:, TOXIC_INDEX]

    @property
    def non_toxic_direction(self):
        return self.W[:, NON_TOXIC_INDEX]

    def unit_toxic_direction(self):
        v = self.toxic_direction
        return v / v.norm().clamp(min=1e-12)

    # -- inference --------------------------------------------------------
    def forward(self, x):
        """Logits ``W^T x`` for features ``x`` of shape ``[..., d_model]``."""
        x = _to_tensor(x).float()
        if self.bias is not None:
            return x @ self.W + self.bias
        return x @ self.W

    __call__ = forward

    def probabilities(self, x):
        import torch

        return torch.softmax(self.forward(x), dim=-1)

    def toxic_probability(self, x):
        """``P(Toxic | x)`` -- used by PPLM as the attribute classifier.

        Differentiable w.r.t. ``x``, which is what PPLM needs.
        """
        return self.probabilities(x)[..., TOXIC_INDEX]

    def toxicity_margin(self, x):
        """Difference of class logits (toxic minus non-toxic)."""
        logits = self.forward(x)
        return logits[..., TOXIC_INDEX] - logits[..., NON_TOXIC_INDEX]

    def predict(self, x) -> np.ndarray:
        import torch

        with torch.no_grad():
            return self.forward(x).argmax(dim=-1).cpu().numpy()

    def score_numpy(self, features: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            return self.toxic_probability(_to_tensor(features)).cpu().numpy()

    # -- persistence ------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "layer": self.layer,
            "n_layers": self.n_layers,
            "position": self.position,
            "pooling": self.pooling,
            "valid_accuracy": float(self.valid_accuracy),
            "d_model": self.d_model,
            "meta": self.meta,
        }

    def state_dict(self) -> Dict[str, Any]:
        d = {"W": self.W.detach().cpu()}
        if self.bias is not None:
            d["bias"] = self.bias.detach().cpu()
        return d

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any], **kwargs) -> "ToxicityProbe":
        return cls(state["W"], bias=state.get("bias"), **kwargs)


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #


def collect_probe_features(
    model,
    tokenizer,
    texts: Sequence[str],
    layer: int = DEFAULT_LAYER,
    position: str = DEFAULT_POSITION,
    pooling: str = "mean",
    batch_size: int = 8,
    max_length: int = DEFAULT_MAX_LENGTH,
    device: Optional[str] = None,
    max_texts: Optional[int] = None,
    verbose: bool = False,
) -> np.ndarray:
    """Residual-stream features :math:`\\overline{\\mathbf{x}}^{L-1}` for ``texts``.

    Returns an array of shape ``[n_texts, d_model]``.
    """

    import torch

    from .model_utils import capture_residual_streams, model_info, resolve_device

    if max_texts is not None:
        texts = list(texts)[: int(max_texts)]
    texts = list(texts)
    if not texts:
        raise ValueError("collect_probe_features received no texts")

    dev = resolve_device(device) if device is not None else resolve_device()
    info = model_info(model)
    n_layers = info.n_layers
    resolved_layer = n_layers - 1 if layer < 0 else int(layer)

    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"

    feats: List[np.ndarray] = []
    n_batches = (len(texts) + batch_size - 1) // batch_size
    iterator = range(0, len(texts), batch_size)
    if verbose:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, total=n_batches, desc="probe features")
        except Exception:
            pass

    for start in iterator:
        chunk = texts[start : start + batch_size]
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(dev)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(dev)
        with torch.no_grad():
            with capture_residual_streams(model) as capture:
                model(input_ids=input_ids, attention_mask=attention_mask)
            states = _state_at(capture, resolved_layer, position)
            pooled = _pool_states(states, attention_mask, pooling)
        feats.append(_as_numpy(pooled))
    return np.concatenate(feats, axis=0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def _class_weights(y, class_weight: Any, num_classes: int = N_TOXIC_CLASSES):
    import torch

    if class_weight is None or class_weight is False:
        return None
    y = np.asarray(y).astype(np.int64)
    if isinstance(class_weight, str) and class_weight.lower() == "balanced":
        counts = np.bincount(y, minlength=num_classes).astype(np.float64)
        counts[counts == 0] = 1.0
        w = counts.sum() / (num_classes * counts)
        return torch.tensor(w, dtype=torch.float32)
    if isinstance(class_weight, (int, float)):
        # interpret as a weight applied to the toxic class
        return torch.tensor([1.0, float(class_weight)], dtype=torch.float32)
    return torch.tensor(np.asarray(class_weight, dtype=np.float32))


def _metrics(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    preds = logits.argmax(axis=-1)
    labels = np.asarray(labels).astype(np.int64)
    acc = float((preds == labels).mean()) if len(labels) else float("nan")
    tp = float(((preds == 1) & (labels == 1)).sum())
    fp = float(((preds == 1) & (labels == 0)).sum())
    fn = float(((preds == 0) & (labels == 1)).sum())
    tn = float(((preds == 0) & (labels == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    tpr = rec
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced = 0.5 * (tpr + tnr)
    # AUC from the toxic-class score (rank statistic, no sklearn needed)
    scores = logits[..., TOXIC_INDEX] if logits.ndim > 1 else logits
    auc = _auc(scores, labels)
    return {
        "accuracy": acc,
        "balanced_accuracy": float(balanced),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "auc": float(auc),
    }


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # average ranks for ties
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def train_probe(
    features: np.ndarray,
    labels: np.ndarray,
    valid_features: Optional[np.ndarray] = None,
    valid_labels: Optional[np.ndarray] = None,
    model_name: str = "gpt2",
    layer: int = DEFAULT_LAYER,
    n_layers: int = 24,
    position: str = DEFAULT_POSITION,
    pooling: str = "mean",
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    class_weight: Any = None,
    seed: int = 0,
    device: Optional[str] = None,
    valid_ratio: float = 0.1,
    verbose: bool = False,
    meta: Optional[Dict[str, Any]] = None,
) -> ProbeTrainResult:
    """Train the linear probe on pre-computed features.

    ``features`` must have shape ``[n, d_model]`` (see
    :func:`collect_probe_features`).  When no validation arrays are given, a
    deterministic ``valid_ratio`` split of the training data is used.
    """

    import torch
    from torch import nn

    from .model_utils import resolve_device, set_seed

    set_seed(seed)
    dev = resolve_device(device)

    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels).astype(np.int64).reshape(-1)
    if X.ndim != 2:
        raise ValueError(f"features must be 2-D [n, d_model], got {X.shape}")
    if len(X) != len(y):
        raise ValueError(f"features/labels length mismatch: {len(X)} vs {len(y)}")

    if valid_features is None or valid_labels is None:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(X))
        n_valid = max(1, int(round(len(X) * valid_ratio)))
        valid_idx, train_idx = idx[:n_valid], idx[n_valid:]
        Xtr, ytr = X[train_idx], y[train_idx]
        Xva, yva = X[valid_idx], y[valid_idx]
    else:
        Xtr, ytr = X, y
        Xva = np.asarray(valid_features, dtype=np.float32)
        yva = np.asarray(valid_labels).astype(np.int64).reshape(-1)

    W = torch.zeros(Xtr.shape[1], N_TOXIC_CLASSES, device=dev, dtype=torch.float32)
    nn.init.normal_(W, std=1e-3)
    W.requires_grad_(True)

    optim = torch.optim.AdamW([W], lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=_class_weights(ytr, class_weight))
    if loss_fn.weight is not None:
        loss_fn.weight = loss_fn.weight.to(dev)

    Xtr_t = torch.as_tensor(Xtr, device=dev)
    ytr_t = torch.as_tensor(ytr, device=dev)
    Xva_t = torch.as_tensor(Xva, device=dev)

    n = len(Xtr_t)
    best_acc = -1.0
    best_state = W.detach().clone()
    best_epoch = 0
    patience_left = max(1, int(patience))
    history: List[Dict[str, float]] = []

    def _valid_logits() -> np.ndarray:
        if len(Xva_t) == 0:
            return np.zeros((0, N_TOXIC_CLASSES), dtype=np.float32)
        return (Xva_t @ W).detach().cpu().numpy()

    for epoch in range(1, int(epochs) + 1):
        perm = torch.randperm(n, device=dev)
        running, seen = 0.0, 0
        for start in range(0, n, batch_size):
            sel = perm[start : start + batch_size]
            xb, yb = Xtr_t[sel], ytr_t[sel]
            logits = xb @ W
            loss = loss_fn(logits, yb)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            running += float(loss.detach()) * len(sel)
            seen += len(sel)
        train_loss = running / max(1, seen)
        train_metrics = _metrics((Xtr_t @ W).detach().cpu().numpy(), ytr)
        valid_logits = _valid_logits()
        valid_metrics = _metrics(valid_logits, yva)
        v_loss = float("nan")
        if len(valid_logits):
            with torch.no_grad():
                v_loss = float(nn.functional.cross_entropy(valid_logits and torch.as_tensor(valid_logits, device=dev), torch.as_tensor(yva, device=dev)))
        entry = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "valid_loss": float(v_loss),
            "train_accuracy": float(train_metrics["accuracy"]),
            "valid_accuracy": float(valid_metrics["accuracy"]),
            "valid_f1": float(valid_metrics["f1"]),
            "valid_auc": float(valid_metrics["auc"]),
        }
        history.append(entry)
        if verbose:
            print(
                f"[probe] epoch {epoch:02d} train_loss={train_loss:.4f} "
                f"train_acc={entry['train_accuracy']:.4f} "
                f"valid_acc={entry['valid_accuracy']:.4f} "
                f"valid_f1={entry['valid_f1']:.4f}"
            )
        score = entry["valid_accuracy"] if len(valid_logits) else train_metrics["accuracy"]
        if score > best_acc + 1e-6:
            best_acc = score
            best_state = W.detach().clone()
            best_epoch = epoch
            patience_left = max(1, int(patience))
        else:
            patience_left -= 1
            if patience_left <= 0:
                if verbose:
                    print(f"[probe] early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    probe = ToxicityProbe(
        best_state.detach().cpu(),
        model_name=model_name,
        layer=int(layer),
        n_layers=int(n_layers),
        position=position,
        pooling=pooling,
        valid_accuracy=float(best_acc),
        meta=dict(meta or {}),
    )

    final_train = _metrics((Xtr_t @ probe.W.to(dev)).detach().cpu().numpy(), ytr)
    final_valid = _metrics(_valid_logits() if False else (Xva_t @ probe.W.to(dev)).detach().cpu().numpy(), yva)
    v_loss_final = float("nan")
    if len(Xva):
        with torch.no_grad():
            v_loss_final = float(
                nn.functional.cross_entropy(
                    (Xva_t @ probe.W.to(dev)), torch.as_tensor(yva, device=dev)
                )
            )
    return ProbeTrainResult(
        probe=probe,
        train_accuracy=float(final_train["accuracy"]),
        valid_accuracy=float(final_valid["accuracy"]),
        valid_balanced_accuracy=float(final_valid["balanced_accuracy"]),
        valid_f1=float(final_valid["f1"]),
        valid_auc=float(final_valid["auc"]),
        valid_loss=float(v_loss_final),
        train_loss=float(history[-1]["train_loss"]) if history else float("nan"),
        epochs_run=len(history),
        n_train=int(len(Xtr)),
        n_valid=int(len(Xva)),
        history=history,
        meta=dict(meta or {}),
    )


def evaluate_probe(
    probe: ToxicityProbe,
    features: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """Classification metrics of a trained probe on given features."""

    logits = probe.forward(_to_tensor(features)).detach().cpu().numpy()
    return _metrics(logits, np.asarray(labels))


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def default_probe_path(path: Optional[str] = None) -> str:
    return path or PROBE_PATH


def save_probe(path: str, result, verbose: bool = False) -> str:
    """Save a probe (or a :class:`ProbeTrainResult`) to ``path`` (``.pt``)."""

    import torch

    probe = result.probe if isinstance(result, ProbeTrainResult) else result
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {"state_dict": probe.state_dict(), "config": probe.to_dict()}
    if isinstance(result, ProbeTrainResult):
        payload["train_result"] = result.to_dict()
        payload["history"] = result.history
    torch.save(payload, path)

    # human-readable sidecar
    json_path = os.path.splitext(path)[0] + ".json"
    try:
        meta = {"config": probe.to_dict()}
        if isinstance(result, ProbeTrainResult):
            meta["train_result"] = result.to_dict()
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, default=str)
    except Exception:
        pass
    if verbose:
        print(f"[probe] saved W_Toxic to {path}")
    return path


def load_probe(path: Optional[str] = None, map_location: str = "cpu") -> ToxicityProbe:
    """Load a probe saved by :func:`save_probe` (or a bare state dict)."""

    import torch

    path = default_probe_path(path)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        cfg = dict(payload.get("config") or {})
        return ToxicityProbe.from_state_dict(payload["state_dict"], **{
            k: cfg[k] for k in ("model_name", "layer", "n_layers", "position", "pooling", "valid_accuracy")
            if k in cfg
        })
    if isinstance(payload, dict) and "W" in payload:
        return ToxicityProbe(payload["W"], bias=payload.get("bias"))
    raise ValueError(f"unrecognised probe payload in {path}")


def probe_exists(path: Optional[str] = None) -> bool:
    return os.path.exists(default_probe_path(path))


# --------------------------------------------------------------------------- #
# High level: Jigsaw -> probe
# --------------------------------------------------------------------------- #


def train_probe_on_jigsaw(
    model=None,
    tokenizer=None,
    data=None,
    model_name: str = "gpt2",
    limit: Optional[int] = None,
    max_valid: Optional[int] = None,
    text_column: str = "comment_text",
    valid_ratio: float = 0.1,
    seed: int = 0,
    **train_kwargs,
) -> ProbeTrainResult:
    """Full Section 3.1 pipeline: Jigsaw -> last-layer residual -> linear probe.

    Parameters
    ----------
    model, tokenizer:
        Loaded GPT2 model/tokenizer.  ``None`` triggers
        :func:`src.model_utils.load_model` with ``model_name``.
    data:
        Optional :class:`data.jigsaw.JigsawData`.  ``None`` loads Jigsaw from
        the HuggingFace mirror with a 90:10 split.
    """

    from .model_utils import load_model, model_info

    if model is None or tokenizer is None:
        model, tokenizer = load_model(model_name)

    if data is None:
        from data.jigsaw import load_jigsaw

        data = load_jigsaw(valid_ratio=valid_ratio, seed=seed, limit=limit)

    train_texts = list(getattr(data.train, "texts"))
    train_labels = np.asarray(getattr(data.train, "labels"))
    valid_texts = list(getattr(data.valid, "texts"))
    valid_labels = np.asarray(getattr(data.valid, "labels"))
    if max_valid is not None:
        valid_texts = valid_texts[: int(max_valid)]
        valid_labels = valid_labels[: int(max_valid)]

    info = model_info(model)
    device = str(next(model.parameters()).device)

    feats_tr = collect_probe_features(
        model, tokenizer, train_texts, device=device, verbose=train_kwargs.pop("verbose", False)
    )
    feats_va = collect_probe_features(model, tokenizer, valid_texts, device=device)

    return train_probe(
        feats_tr,
        train_labels,
        feats_va,
        valid_labels,
        model_name=model_name,
        layer=-1,
        n_layers=info.n_layers,
        seed=seed,
        **train_kwargs,
    )


def load_or_train_probe(
    path: Optional[str] = None,
    model=None,
    tokenizer=None,
    model_name: str = "gpt2",
    force: bool = False,
    verbose: bool = False,
    **kwargs,
) -> ToxicityProbe:
    """Return the saved probe at ``path``, training it first if missing."""

    path = default_probe_path(path)
    if probe_exists(path) and not force:
        return load_probe(path)
    result = train_probe_on_jigsaw(model, tokenizer, model_name=model_name, verbose=verbose, **kwargs)
    save_probe(path, result, verbose=verbose)
    return result.probe


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #


def _main() -> None:  # pragma: no cover - manual smoke test
    import argparse

    parser = argparse.ArgumentParser(description="Train W_Toxic on Jigsaw (smoke test).")
    parser.add_argument("--model", default="openai-community/gpt2-medium")
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--out", default=PROBE_PATH)
    args = parser.parse_args()

    from .model_utils import load_model, model_info

    model, tokenizer = load_model(args.model)
    info = model_info(model)
    print(f"[probe] {info}")
    result = train_probe_on_jigsaw(
        model,
        tokenizer,
        model_name=args.model,
        limit=args.limit,
        epochs=args.epochs,
        verbose=True,
    )
    print("[probe] summary:", json.dumps(result.summary(), indent=2, default=str))
    save_probe(args.out, result, verbose=True)


if __name__ == "__main__":  # pragma: no cover
    _main()
