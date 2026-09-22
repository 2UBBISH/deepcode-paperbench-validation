"""Section 3.1 -- extracting toxic MLP vectors and their SVD decomposition.

Given the probe direction ``W_toxic[:, 1]`` we rank every MLP *value vector* of
the model (columns of ``W_V``, i.e. rows of our ``[d_mlp, d_model]`` view) by
cosine similarity with the probe direction.  The ``N = 128`` most similar value
vectors -- together with their key vectors -- are the "toxic vectors"
``MLP.v_toxic`` / ``MLP.k_toxic`` of the paper.

Stacking the ``N`` value vectors into an ``N x d`` matrix and taking the SVD of
its transpose (as clarified in the addendum) yields ``d``-dimensional singular
value vectors ``SVD.U_toxic[i]`` that span the toxicity representation space.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from .architecture import TransformerInternals
from .utils import save_json


@dataclass
class ToxicVectorConfig:
    top_n: int = 128
    branch: str = "k"              # GPT2 has a single branch; GLUs use "k" (gate) by default
    n_svd_components: int = 10
    exclude_layers: Tuple[int, ...] = ()


def load_w_toxic(path: str | os.PathLike, d_model: Optional[int] = None) -> torch.Tensor:
    """Load the toxic probe direction ``W_toxic[:, 1]`` (unit-normalised)."""
    blob = torch.load(path, map_location="cpu")
    if isinstance(blob, dict):
        for key in ("w_toxic", "weight", "toxic_direction"):
            if key in blob:
                blob = blob[key]
                break
    w = torch.as_tensor(blob, dtype=torch.float32)
    if w.ndim == 2:  # [2, d] probe matrix -> toxic row
        w = w[1]
    return w


@torch.no_grad()
def rank_value_vectors(internals: TransformerInternals,
                       w_toxic: torch.Tensor,
                       layers: Optional[List[int]] = None) -> Dict:
    """Cosine similarity between every value vector and ``W_toxic``."""
    layers = layers if layers is not None else list(range(internals.n_layers))
    w = w_toxic.float()
    w = w / w.norm().clamp(min=1e-8)
    sims, locs, vecs = [], [], []
    for l in layers:
        v = internals.value_weight(l).detach().float().cpu()          # [d_mlp, d_model]
        vn = v / v.norm(dim=1, keepdim=True).clamp(min=1e-8)
        sim = vn @ w
        sims.append(sim)
        locs.extend([(l, i) for i in range(v.shape[0])])
        vecs.append(vn)
    sims = torch.cat(sims, dim=0)
    vecs = torch.cat(vecs, dim=0)
    return {"similarities": sims, "locations": locs, "unit_vectors": vecs}


def select_toxic_vectors(internals: TransformerInternals, w_toxic: torch.Tensor,
                         cfg: ToxicVectorConfig) -> Dict:
    """Top-``N`` toxic value vectors + their key vectors (and location metadata)."""
    ranked = rank_value_vectors(internals, w_toxic)
    sims, locs = ranked["similarities"], ranked["locations"]
    order = torch.argsort(sims, descending=True)
    if cfg.exclude_layers:
        keep = [i for i, idx in enumerate(order.tolist()) if locs[idx][0] not in cfg.exclude_layers]
        order = order[torch.tensor(keep, dtype=torch.long)]
    chosen = order[: cfg.top_n]

    selections = []
    value_vectors, key_vectors, raw_value_vectors = [], [], []
    for idx in chosen.tolist():
        layer, neuron = locs[idx]
        v = internals.value_weight(layer).detach().float().cpu()[neuron]
        k = internals.key_weight(layer, cfg.branch).detach().float().cpu()[neuron]
        value_vectors.append(v / v.norm().clamp(min=1e-8))
        raw_value_vectors.append(v)
        key_vectors.append(k)
        selections.append({"layer": int(layer), "index": int(neuron),
                           "cosine": float(sims[idx]), "flat_index": int(idx)})
    return {
        "selections": selections,
        "value_vectors": torch.stack(value_vectors),          # unit-normalised, [N, d]
        "raw_value_vectors": torch.stack(raw_value_vectors),  # [N, d]
        "key_vectors": torch.stack(key_vectors),              # [N, d]
        "all_similarities": sims,
        "all_locations": locs,
    }


def svd_toxic_basis(unit_value_vectors: torch.Tensor, n_components: int = 10) -> Dict:
    """SVD of the ``d x N`` matrix (transpose of the stacked ``N x d`` vectors).

    Returns the leading singular value vectors ``SVD.U_toxic[i]`` together with
    the singular values and the explained-variance ratios.
    """
    m = unit_value_vectors.double()          # [N, d]
    u, s, vh = torch.linalg.svd(m.T, full_matrices=False)  # m.T is [d, N]
    var = (s ** 2) / (s ** 2).sum().clamp(min=1e-12)
    return {
        "u": u[:, :n_components].float(),    # [d, k] columns = SVD.U_toxic[i]
        "u_full": u.float(),
        "singular_values": s.float(),
        "explained_variance_ratio": var.float(),
    }


def extract_toxic_vectors(model: torch.nn.Module,
                          w_toxic_path: str,
                          cfg: Optional[ToxicVectorConfig] = None,
                          out_dir: Optional[str] = "artifacts/toxic_vectors",
                          save: bool = True) -> Dict:
    """Full Section 3.1 pipeline: rank -> top-N -> SVD."""
    cfg = cfg or ToxicVectorConfig()
    internals = TransformerInternals(model)
    w = load_w_toxic(w_toxic_path)
    selection = select_toxic_vectors(internals, w, cfg)
    svd = svd_toxic_basis(selection["value_vectors"], cfg.n_svd_components)
    result = {"config": cfg.__dict__, "selection": selection, "svd": svd,
              "w_toxic": w.cpu()}
    if save and out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save({
            "value_vectors": selection["value_vectors"],
            "raw_value_vectors": selection["raw_value_vectors"],
            "key_vectors": selection["key_vectors"],
            "svd_u": svd["u"],
            "singular_values": svd["singular_values"],
            "explained_variance_ratio": svd["explained_variance_ratio"],
            "w_toxic": w.cpu(),
        }, out / "toxic_vectors.pt")
        save_json({"selections": selection["selections"],
                   "n_value_vectors": int(selection["value_vectors"].shape[0]),
                   "d_model": int(selection["value_vectors"].shape[1]),
                   "explained_variance_ratio": svd["explained_variance_ratio"].tolist(),
                   "config": cfg.__dict__}, out / "toxic_vectors.json")
    return result


def load_toxic_vectors(path: str | os.PathLike) -> Dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu")
