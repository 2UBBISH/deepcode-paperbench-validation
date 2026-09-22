"""Sec. 3.2 -- Logit-change based forecasting.

Eqn. 2 relates the logit change of an upstream example x_j to the logit change of
the online learned example x_i through the NTK-like kernel
Theta(x_j, x_i) Theta(x_i, x_i)^-1.  The paper replaces that kernel with a
learnable one, Theta_tilde(x_j, x_i) = h(x_j, y_j) h(x_i, y_i)^T, so that

    f_hat_i(x_j) = Theta_tilde(x_j, x_i) @ [f_hat_i(x_i) - f_hat_0(x_i)] + f_hat_0(x_j)

Two variants are reported in Table 1:

* FixedLogitForecaster     -- h is the frozen representation of the base PTLM.
  This is the exact kernel of Eqn. 2 when only the LM head is tuned (Sec. 4.2).
* TrainableLogitForecaster -- h is a trainable LM + 2-layer MLP, learned with the
  margin loss of Eqn. 3.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from tqdm import tqdm

from ..config import FORECAST_TRAIN
from .base import BaseForecaster, ForecastContext, sample_pairs
from .encoders import PairEncoder, build_encoder
from .types import OnlineArtifact


# --------------------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------------------
def pad_to(arr: np.ndarray, T: int, axis: int = 0) -> np.ndarray:
    """Zero-pad ``arr`` along ``axis`` to length ``T`` (truncating if longer)."""
    arr = np.asarray(arr)
    if arr.shape[axis] >= T:
        return np.take(arr, np.arange(T), axis=axis)
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (0, T - arr.shape[axis])
    return np.pad(arr, pad_width, mode="constant")


def predict_forgetting_from_logits(
    pred_logits: np.ndarray,      # [N, T, C]
    gold_ids: np.ndarray,         # [N, T]
    mask: np.ndarray,             # [N, T] bool
    vocab_ids: np.ndarray,        # [C]
) -> np.ndarray:
    """Forgotten iff the arg-max predicted token differs from the gold token.

    Mirrors Algorithms 2 and 4 of the paper ("if arg max f_hat_i(x_j) != y_j then
    z_hat_ij = 1").  Only positions that belong to the gold output are compared.
    """
    best = pred_logits.argmax(axis=-1)               # [N, T] index into the candidate vocab
    predicted_ids = vocab_ids[best]                  # [N, T] original vocabulary ids
    mismatch = (predicted_ids != gold_ids) & mask
    return mismatch.any(axis=1).astype(int)


def margin_loss(
    scores_correct,
    scores_other,
    labels,
    margin: float = FORECAST_TRAIN["margin"],
    positive_weight: float = FORECAST_TRAIN["positive_loss_weight"],
):
    """Eqn. 3 of the paper.

    max(0, 1 + (-1)^z (max_{v != y_j} f_hat_i(x_j)[v] - f_hat_i(x_j)[y_j]))

    ``labels`` holds z_ij; positive (forgotten) pairs are down-weighted with
    alpha = 0.1 as described in Appendix B.
    """
    import torch

    sign = torch.where(labels > 0.5, torch.full_like(labels, -1.0), torch.ones_like(labels))
    loss = torch.relu(margin + sign * (scores_other - scores_correct))
    weights = torch.where(labels > 0.5, torch.full_like(loss, positive_weight), torch.ones_like(loss))
    return (loss * weights).sum() / weights.sum().clamp(min=1e-6)


class _LogitForecasterBase(BaseForecaster):
    """Common caching of the upstream logits / gold ids / masks."""

    def __init__(self, T: Optional[int] = None) -> None:
        self.T: Optional[int] = T
        self.vocab_ids: Optional[np.ndarray] = None
        self.upstream_logits: Optional[np.ndarray] = None
        self.upstream_gold: Optional[np.ndarray] = None
        self.upstream_mask: Optional[np.ndarray] = None

    def _cache_upstream(self, ctx: ForecastContext) -> None:
        self.T = self.T or ctx.cfg.max_target_len
        cache = ctx.upstream_cache
        self.vocab_ids = cache.vocab.vocab_ids
        self.upstream_logits = np.stack(
            [pad_to(item.reduced_logits.astype(np.float32), self.T) for item in cache.items], axis=0
        )
        # `reduced_logits` are indexed by the candidate vocabulary, so the gold
        # tokens have to be translated from vocabulary ids to candidate columns.
        column_of = {int(v): k for k, v in enumerate(self.vocab_ids)}
        gold_columns, masks = [], []
        for item in cache.items:
            columns = np.asarray([column_of.get(int(g), 0) for g in item.gold_ids], dtype=np.int64)
            present = np.asarray([int(g) in column_of for g in item.gold_ids], dtype=bool)
            gold_columns.append(pad_to(columns, self.T))
            masks.append(pad_to(item.mask & present, self.T))
        self.upstream_gold = np.stack(gold_columns, axis=0)
        self.upstream_mask = np.stack(masks, axis=0)

    def _predicted_logits(self, artifact: OnlineArtifact, kernel: np.ndarray) -> np.ndarray:
        delta = pad_to(np.asarray(artifact.delta_logits, dtype=np.float32), self.T)   # [T, C]
        return np.einsum("nt,tc->ntc", kernel, delta) + self.upstream_logits


# --------------------------------------------------------------------------------------
# fixed (non-trained) logit-based forecasting
# --------------------------------------------------------------------------------------
class FixedLogitForecaster(_LogitForecasterBase):
    """h = frozen final-layer representation of the base PTLM (Sec. 4.2)."""

    name = "fixed_logit"
    trainable = False

    def __init__(self, T: Optional[int] = None) -> None:
        super().__init__(T)
        self.upstream_reps: Optional[np.ndarray] = None

    def fit(self, ctx: ForecastContext) -> "FixedLogitForecaster":
        self._cache_upstream(ctx)
        self.upstream_reps = np.stack(
            [pad_to(item.decoder_reps.astype(np.float32), self.T) for item in ctx.upstream_cache.items], axis=0
        )
        return self

    def predict(self, artifact: OnlineArtifact) -> np.ndarray:
        if self.upstream_reps is None:
            raise RuntimeError("call fit() first")
        online_reps = getattr(artifact, "frozen_token_reps", None)
        if online_reps is None:
            raise RuntimeError(
                "the artifact does not carry the frozen representation of the online example; "
                "rebuild it with wwmf.forecasting.cache.build_online_artifacts(..., cache_frozen_reps=True)"
            )
        online_reps = pad_to(np.asarray(online_reps, dtype=np.float32), self.T)
        kernel = np.einsum("ntd,td->nt", self.upstream_reps, online_reps)
        pred = self._predicted_logits(artifact, kernel)
        return predict_forgetting_from_logits(pred, self.upstream_gold, self.upstream_mask, self.vocab_ids)

    def scores(self, artifact: OnlineArtifact) -> np.ndarray:
        return self.predict(artifact).astype(np.float64)


# --------------------------------------------------------------------------------------
# trainable logit-based forecasting
# --------------------------------------------------------------------------------------
class TrainableLogitForecaster(_LogitForecasterBase):
    """h = trainable LM + MLP, optimised with the margin loss of Eqn. 3."""

    name = "trainable_logit"
    trainable = True

    def __init__(
        self,
        encoder: Optional[PairEncoder] = None,
        T: Optional[int] = None,
        max_steps: int = FORECAST_TRAIN["max_steps"],
        batch_pos: int = FORECAST_TRAIN["n_positive_per_batch"],
        batch_neg: int = FORECAST_TRAIN["n_negative_per_batch"],
        margin: float = FORECAST_TRAIN["margin"],
        positive_weight: float = FORECAST_TRAIN["positive_loss_weight"],
        log_every: int = 500,
    ) -> None:
        super().__init__(T)
        self.encoder = encoder
        self.max_steps = max_steps
        self.batch_pos = batch_pos
        self.batch_neg = batch_neg
        self.margin = margin
        self.positive_weight = positive_weight
        self.log_every = log_every
        self.upstream_reps: Optional[np.ndarray] = None
        self.history: List[float] = []

    # ----------------------------------------------------------------------------------
    def _pad_tokens(self, tensor, T: int):
        import torch

        if tensor.shape[1] >= T:
            return tensor[:, :T]
        pad = torch.zeros(
            (tensor.shape[0], T - tensor.shape[1], tensor.shape[2]), device=tensor.device, dtype=tensor.dtype
        )
        return torch.cat([tensor, pad], dim=1)

    def fit(self, ctx: ForecastContext) -> "TrainableLogitForecaster":
        import torch

        self._cache_upstream(ctx)
        if self.encoder is None:
            self.encoder = build_encoder(
                ctx.cfg.model_spec().forecast_encoder_hf_name,
                device=ctx.device,
                hidden_dim=ctx.hidden_dim,
                trainable_lm=True,
                max_input_len=ctx.cfg.max_input_len,
                max_target_len=ctx.cfg.max_target_len,
            )
        groups = [g for g in self.encoder.parameter_groups(
            FORECAST_TRAIN["lm_learning_rate"], FORECAST_TRAIN["mlp_learning_rate"]) if g["params"]]
        optimizer = torch.optim.AdamW(groups)
        labels = ctx.train_labels.astype(np.int64)
        rng = np.random.default_rng(ctx.seed)
        examples = ctx.upstream_cache.examples
        online_examples = [a.example for a in ctx.train_artifacts]
        deltas = {
            a.example.key: np.asarray(a.delta_logits, dtype=np.float32) for a in ctx.train_artifacts
        }
        allowed = ctx.eval_mask

        self.encoder.lm.train()
        for step in tqdm(range(1, self.max_steps + 1), desc="train logit forecaster",
                         disable=not ctx.verbose):
            pos_pairs, neg_pairs = sample_pairs(
                labels, self.batch_pos, self.batch_neg, rng, allowed_upstream=allowed
            )
            pairs = np.concatenate([pos_pairs, neg_pairs], axis=0)
            if len(pairs) == 0:
                break
            pair_labels = torch.as_tensor(
                np.concatenate([np.ones(len(pos_pairs)), np.zeros(len(neg_pairs))]),
                dtype=torch.float32, device=ctx.device,
            )
            out_online = self.encoder(
                [online_examples[int(i)].input for i in pairs[:, 0]],
                [online_examples[int(i)].target for i in pairs[:, 0]],
            )
            out_up = self.encoder(
                [examples[int(j)].input for j in pairs[:, 1]], [examples[int(j)].target for j in pairs[:, 1]]
            )
            T = max(out_online.token_reps.shape[1], out_up.token_reps.shape[1])
            H_i = self._pad_tokens(out_online.token_reps, T)                # [B, T, d]
            H_j = self._pad_tokens(out_up.token_reps, T)                    # [B, T, d]
            delta = torch.as_tensor(
                np.stack([pad_to(deltas[online_examples[int(i)].key], T) for i in pairs[:, 0]]),
                device=ctx.device,
            )                                                               # [B, T, C]
            gold = torch.as_tensor(
                np.stack([pad_to(self.upstream_gold[int(j)], T) for j in pairs[:, 1]]), device=ctx.device
            )                                                               # [B, T]
            base = torch.as_tensor(
                np.stack([pad_to(self.upstream_logits[int(j)], T) for j in pairs[:, 1]]), device=ctx.device
            )                                                               # [B, T, C]
            mask = torch.as_tensor(
                np.stack([pad_to(self.upstream_mask[int(j)], T) for j in pairs[:, 1]]), device=ctx.device
            )                                                               # [B, T]

            kernel = torch.einsum("btd,bid->bti", H_j, H_i)                 # [B, T, T]
            pred = torch.einsum("bti,bic->btc", kernel, delta) + base       # [B, T, C]
            scores_correct = torch.gather(pred, 2, gold.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            other = pred.clone()
            other.scatter_(2, gold.clamp(min=0).unsqueeze(-1), float("-inf"))
            scores_other = other.max(dim=2).values

            keep = mask.reshape(-1)
            loss = margin_loss(
                scores_correct.reshape(-1)[keep],
                scores_other.reshape(-1)[keep],
                pair_labels.unsqueeze(1).expand(-1, mask.shape[1]).reshape(-1)[keep],
                margin=self.margin,
                positive_weight=self.positive_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.history.append(float(loss.detach().cpu()))
            if ctx.verbose and self.log_every and step % self.log_every == 0:
                print(f"[trainable-logit] step {step}: loss={np.mean(self.history[-self.log_every:]):.4f}")
        self.prepare_upstream(ctx)
        return self

    def prepare_upstream(self, ctx: ForecastContext, batch_size: int = 8) -> None:
        """Cache h(x_j, y_j) for every upstream example (one pass over D_PT)."""
        reps, _, _ = self.encoder.encode_many(
            [e.input for e in ctx.upstream_cache.examples],
            [e.target for e in ctx.upstream_cache.examples],
            batch_size=batch_size,
            pad_to=self.T,
        )
        self.upstream_reps = reps

    def predict(self, artifact: OnlineArtifact) -> np.ndarray:
        if self.upstream_reps is None:
            raise RuntimeError("call fit() (or prepare_upstream) first")
        import torch

        with torch.no_grad():
            out = self.encoder([artifact.example.input], [artifact.example.target])
            H_i = self._pad_tokens(out.token_reps, self.T)[0].float().cpu().numpy()   # [T, d]
        kernel = np.einsum("ntd,td->nt", self.upstream_reps, H_i)
        pred = self._predicted_logits(artifact, kernel)
        return predict_forgetting_from_logits(pred, self.upstream_gold, self.upstream_mask, self.vocab_ids)

    def scores(self, artifact: OnlineArtifact) -> np.ndarray:
        return self.predict(artifact).astype(np.float64)
