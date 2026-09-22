"""Section 3.2 -- projecting toxic vectors onto the vocabulary space.

Following Geva et al. (2022), a value vector ``v`` promotes the tokens whose
embeddings have the largest dot product with it: ``r = E v``.  Table 1 of the
paper lists the top tokens of ``W_toxic``, several ``MLP.v_toxic`` and of
``SVD.U_toxic[i]``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from .utils import save_json

# The paper censors the offensive tokens in Table 1 ("WARNING: THESE EXAMPLES
# ARE HIGHLY OFFENSIVE"): c*nt, f*ck, a**hole, ...  We reproduce the same
# censoring in the markdown rendering (the JSON keeps the raw tokens).
_PROFANITY = [
    "asshole", "arsehole", "shit", "fuck", "fucking", "cunt", "bitch", "whore",
    "dick", "cock", "pussy", "nigger", "nigga", "faggot", "rape", "bastard",
    "slut", "piss", "penis", "vagina", "boobs", "tits", "douche", "retard",
]


def censor_token(token: str) -> str:
    """Mask the middle of offensive tokens, as the paper does in Table 1."""
    prefix_len = len(token) - len(token.lstrip("ĠĊ "))
    prefix, stripped = token[:prefix_len], token[prefix_len:]
    low = stripped.lower()
    for word in _PROFANITY:
        if low.startswith(word):
            if len(word) <= 3:
                masked = word[0] + "*" * (len(word) - 1)
            else:
                masked = word[0] + "*" * (len(word) - 2) + word[-1]
            return prefix + masked
    return token


def embedding_matrix(model: torch.nn.Module) -> torch.Tensor:
    """The (untied) embedding matrix ``E`` of shape ``[|V|, d]``."""
    return model.get_input_embeddings().weight.detach().float().cpu()


def top_tokens(model: torch.nn.Module, tokenizer, vector: torch.Tensor, k: int = 10,
               skip_special: bool = True, min_token_len: int = 0) -> List[Dict]:
    """Tokens with the highest dot product with ``vector`` (Table 1 / Table 6)."""
    e = embedding_matrix(model)
    v = vector.detach().float().cpu().flatten()
    scores = e @ v
    order = torch.argsort(scores, descending=True)
    out: List[Dict] = []
    for idx in order.tolist():
        if len(out) >= k:
            break
        tok = tokenizer.convert_ids_to_tokens(int(idx))
        if tok is None:
            continue
        if skip_special and tok in tokenizer.all_special_tokens:
            continue
        if len(tok) < min_token_len:
            continue
        out.append({"token": tok, "token_id": int(idx), "score": float(scores[idx])})
    return out


def build_table(model: torch.nn.Module, tokenizer, w_toxic: torch.Tensor,
                selections: Sequence[Dict], value_vectors: torch.Tensor,
                svd_u: Optional[torch.Tensor] = None, k: int = 8) -> Dict:
    """Reproduce the layout of Table 1 for a set of toxic vectors."""
    table: Dict[str, List[Dict]] = {}
    table["W_toxic"] = top_tokens(model, tokenizer, w_toxic, k=k)
    for sel, vec in zip(selections, value_vectors):
        name = f"MLP.v_{sel['index']}^{sel['layer']}"
        table[name] = top_tokens(model, tokenizer, vec, k=k)
    if svd_u is not None:
        for i in range(svd_u.shape[1]):
            table[f"SVD.U_toxic[{i}]"] = top_tokens(model, tokenizer, svd_u[:, i], k=k)
    return table


def project_and_save(model: torch.nn.Module, tokenizer, w_toxic: torch.Tensor,
                     selections: Sequence[Dict], value_vectors: torch.Tensor,
                     svd_u: Optional[torch.Tensor], out_dir: str = "artifacts/vocab",
                     k: int = 8, n_svd: int = 5) -> Dict:
    svd_subset = svd_u[:, :n_svd] if svd_u is not None else None
    table = build_table(model, tokenizer, w_toxic, selections, value_vectors, svd_subset, k=k)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(table, out / "vocab_projection.json")
    with open(out / "vocab_projection.md", "w") as f:
        f.write("| Vector | top tokens |\n|---|---|\n")
        for name, row in table.items():
            toks = ", ".join(censor_token(r["token"]) for r in row)
            f.write(f"| {name} | {toks} |\n")
    return table
