"""Section 3.1 -- the toxicity probe ``W_toxic``.

The paper trains a linear probe on the *mean residual stream of the last layer*
of GPT2-medium::

    P(Toxic | xbar^{L-1}) = softmax(W_toxic xbar^{L-1}),  W_toxic in R^{d x 2}

with a 90:10 split of the Jigsaw dataset, reaching 94% validation accuracy.
``W_toxic[:, 1]`` is the toxic class row; per the addendum this is the vector
that all cosine similarities in the paper are computed against.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .utils import batches, get_device, mean_pool, save_json, set_seed


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_mean_residuals(
    model: torch.nn.Module,
    tokenizer,
    texts: Sequence[str],
    layer: int = -1,
    max_length: int = 128,
    batch_size: int = 16,
    device: Optional[torch.device] = None,
    show_progress: bool = True,
) -> np.ndarray:
    """Mean-pooled residual stream features ``xbar`` of shape ``[N, d_model]``.

    ``layer=-1`` is the residual stream after the final transformer block, which
    is the "last layer" residual stream of the paper.  ``layer=-2`` is the
    residual stream after the second-to-last block (identical hidden state, since
    no module sits between two blocks).
    """
    device = device or next(model.parameters()).device
    feats: List[np.ndarray] = []
    iterator = batches(list(texts), batch_size)
    if show_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, total=(len(texts) + batch_size - 1) // batch_size)
        except ImportError:  # pragma: no cover
            pass
    for batch in iterator:
        enc = tokenizer(list(batch), return_tensors="pt", padding=True, truncation=True,
                        max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True)
        hidden = out.hidden_states[layer]  # (B, T, d)
        pooled = mean_pool(hidden, enc["attention_mask"])
        feats.append(pooled.float().cpu().numpy())
    return np.concatenate(feats, axis=0)


# --------------------------------------------------------------------------- #
# Probe model
# --------------------------------------------------------------------------- #
@dataclass
class ProbeConfig:
    layer: int = -1
    max_length: int = 128
    batch_size: int = 16
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    val_fraction: float = 0.1
    seed: int = 0
    max_train: Optional[int] = None   # cap for CPU-friendly runs
    max_val: Optional[int] = None
    verbose: bool = True


class ToxicityProbe(nn.Module):
    """Linear probe; ``weight`` has shape ``[2, d_model]`` (row 1 = toxic)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, 2, bias=False)

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight  # [2, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    @torch.no_grad()
    def toxic_direction(self) -> torch.Tensor:
        """``W_toxic[:, 1]`` (the toxic class row) in the paper's convention."""
        return self.linear.weight[1].detach().clone()

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x).argmax(dim=-1)

    # ------------------------------------------------------------- (de)serialise
    def save(self, path: str | os.PathLike) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"weight": self.linear.weight.detach().cpu()}, path)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "ToxicityProbe":
        blob = torch.load(path, map_location="cpu")
        probe = cls(d_model=int(blob["weight"].shape[1]))
        probe.linear.weight.data.copy_(blob["weight"])
        probe.eval()
        return probe


def train_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    cfg: ProbeConfig,
    device: Optional[torch.device] = None,
) -> Tuple[ToxicityProbe, Dict]:
    """Fit the linear probe with Adam + cross-entropy, keeping the best val checkpoint."""
    set_seed(cfg.seed)
    device = device or get_device()
    d_model = train_features.shape[1]
    probe = ToxicityProbe(d_model).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    lossf = nn.CrossEntropyLoss()

    xtr = torch.tensor(train_features, dtype=torch.float32, device=device)
    ytr = torch.tensor(train_labels, dtype=torch.long, device=device)
    xva = torch.tensor(val_features, dtype=torch.float32, device=device)
    yva = torch.tensor(val_labels, dtype=torch.long, device=device)

    bs = cfg.batch_size
    history: List[Dict] = []
    best_acc, best_state = -1.0, None
    for epoch in range(cfg.epochs):
        probe.train()
        perm = torch.randperm(xtr.shape[0], device=device)
        total, correct, loss_sum = 0, 0, 0.0
        for i in range(0, xtr.shape[0], bs):
            idx = perm[i : i + bs]
            logits = probe(xtr[idx])
            loss = lossf(logits, ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += float(loss) * len(idx)
            correct += int((logits.argmax(-1) == ytr[idx]).sum())
            total += len(idx)
        probe.eval()
        with torch.no_grad():
            val_logits = probe(xva)
            val_loss = float(lossf(val_logits, yva))
            val_acc = float((val_logits.argmax(-1) == yva).float().mean())
        rec = {"epoch": epoch, "train_loss": loss_sum / max(total, 1),
               "train_acc": correct / max(total, 1), "val_loss": val_loss, "val_acc": val_acc}
        history.append(rec)
        if cfg.verbose:
            print(f"[probe] epoch {epoch}: train_acc={rec['train_acc']:.4f} "
                  f"train_loss={rec['train_loss']:.4f} val_acc={val_acc:.4f} val_loss={val_loss:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}
    if best_state is not None:
        probe.load_state_dict(best_state)
    probe.eval()
    metrics = {"best_val_acc": best_acc, "history": history, "d_model": d_model}
    return probe, metrics


def run_probe_training(
    model_name: str = "gpt2-medium",
    cfg: Optional[ProbeConfig] = None,
    out_dir: str = "artifacts/probe",
    cache_dir: Optional[str] = None,
    hf_dataset_id: Optional[str] = None,
    device: Optional[str] = None,
    feature_cache: Optional[str] = None,
    features_dir: Optional[str] = None,
) -> Tuple[ToxicityProbe, Dict]:
    """End-to-end Section 3.1: Jigsaw -> residual features -> linear probe ``W_toxic``.

    ``features_dir`` loads mean-pooled residual features from the resumable
    shards produced by ``scripts/extract_features.py`` instead of running the
    LM over the whole dataset again.
    """
    from .data.jigsaw import JIGSAW_HF_ID, build_jigsaw_dataset
    from .utils import load_tokenizer

    cfg = cfg or ProbeConfig()
    dev = get_device(device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_texts, train_labels, val_texts, val_labels, stats = build_jigsaw_dataset(
        hf_id=hf_dataset_id or JIGSAW_HF_ID, cache_dir=cache_dir,
        max_train=cfg.max_train, max_val=cfg.max_val, val_fraction=cfg.val_fraction, seed=cfg.seed)

    from transformers import AutoModelForCausalLM

    tokenizer = load_tokenizer(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(dev)
    model.eval()

    if features_dir is not None:
        import glob

        def _load(split: str):
            xs, ys = [], []
            for f in sorted(glob.glob(str(Path(features_dir) / f"{split}_*.npz"))):
                blob = np.load(f)
                xs.append(blob["features"])
                ys.append(blob["labels"])
            if not xs:
                raise FileNotFoundError(f"no feature shards found in {features_dir} for split {split}")
            return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

        xtr, ytr = _load("train")
        xva, yva = _load("val")
    elif feature_cache and Path(feature_cache).exists():
        blob = np.load(feature_cache)
        xtr, ytr, xva, yva = blob["xtr"], blob["ytr"], blob["xva"], blob["yva"]
    else:
        xtr = extract_mean_residuals(model, tokenizer, train_texts, layer=cfg.layer,
                                     max_length=cfg.max_length, batch_size=cfg.batch_size, device=dev)
        xva = extract_mean_residuals(model, tokenizer, val_texts, layer=cfg.layer,
                                     max_length=cfg.max_length, batch_size=cfg.batch_size, device=dev)
        ytr, yva = train_labels, val_labels
        if feature_cache:
            Path(feature_cache).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(feature_cache, xtr=xtr, ytr=ytr, xva=xva, yva=yva)

    probe, metrics = train_probe(xtr, ytr, xva, yva, cfg, device=dev)
    probe.save(out / "toxic_probe.pt")
    torch.save(probe.toxic_direction().cpu(), out / "w_toxic.pt")
    save_json({"config": cfg.__dict__, "metrics": metrics, "data": stats,
               "model_name": model_name, "probe_path": str(out / "toxic_probe.pt"),
               "w_toxic_path": str(out / "w_toxic.pt")}, out / "probe_report.json")
    return probe, metrics


def load_probe(path: str, device: Optional[str] = None) -> ToxicityProbe:
    probe = ToxicityProbe.load(path)
    return probe.to(get_device(device))
