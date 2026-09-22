"""Sec. 3.3 -- Representation-based (black-box) forecasting.

    g(<x_i, y_i>, <x_j, y_j>) = sigma( h(x_j, y_j) h(x_i, y_i)^T + b_j )

where ``h`` maps the concatenation of inputs and outputs to an averaged
representation (Sec. 3.3), and ``b_j`` is the frequency prior of forgetting the
upstream example x_j in ``D_R^Train`` (Sec. 3.3, "Forecasting with Frequency
Priors").  The model is trained with a binary cross-entropy loss.

Setting ``use_prior=False`` gives the "w/o Prior" ablation of Table 1 / Table 2.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from tqdm import tqdm

from ..config import FORECAST_TRAIN
from .base import BaseForecaster, ForecastContext, frequency_prior, sample_pairs
from .encoders import PairEncoder, build_encoder
from .types import OnlineArtifact


class RepresentationForecaster(BaseForecaster):
    name = "representation"
    trainable = True

    def __init__(
        self,
        encoder: Optional[PairEncoder] = None,
        use_prior: bool = True,
        max_steps: int = FORECAST_TRAIN["max_steps"],
        batch_pos: int = FORECAST_TRAIN["n_positive_per_batch"],
        batch_neg: int = FORECAST_TRAIN["n_negative_per_batch"],
        positive_weight: float = FORECAST_TRAIN["positive_loss_weight"],
        log_every: int = 500,
    ) -> None:
        self.encoder = encoder
        self.use_prior = use_prior
        self.max_steps = max_steps
        self.batch_pos = batch_pos
        self.batch_neg = batch_neg
        self.positive_weight = positive_weight
        self.log_every = log_every
        self.prior: Optional[np.ndarray] = None
        self.upstream_reps: Optional[np.ndarray] = None
        self.history: List[float] = []

    # ----------------------------------------------------------------------------------
    def fit(self, ctx: ForecastContext) -> "RepresentationForecaster":
        import torch

        if self.encoder is None:
            self.encoder = build_encoder(
                ctx.cfg.model_spec().forecast_encoder_hf_name,
                device=ctx.device,
                hidden_dim=ctx.hidden_dim,
                trainable_lm=True,
                max_input_len=ctx.cfg.max_input_len,
                max_target_len=ctx.cfg.max_target_len,
            )
        labels = ctx.train_labels.astype(np.float32)
        self.prior = frequency_prior(labels) if self.use_prior else np.zeros(labels.shape[1], dtype=np.float64)
        prior_t = torch.as_tensor(self.prior, dtype=torch.float32, device=ctx.device)

        groups = [g for g in self.encoder.parameter_groups(
            FORECAST_TRAIN["lm_learning_rate"], FORECAST_TRAIN["mlp_learning_rate"]) if g["params"]]
        optimizer = torch.optim.AdamW(groups)
        rng = np.random.default_rng(ctx.seed)
        upstream_examples = ctx.upstream_cache.examples
        online_examples = [a.example for a in ctx.train_artifacts]
        allowed = ctx.eval_mask

        self.encoder.lm.train()
        for step in tqdm(range(1, self.max_steps + 1), desc="train representation forecaster",
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
                [upstream_examples[int(j)].input for j in pairs[:, 1]],
                [upstream_examples[int(j)].target for j in pairs[:, 1]],
            )
            score = (out_up.pooled * out_online.pooled).sum(dim=-1) + prior_t[pairs[:, 1]]
            weight = torch.where(
                pair_labels > 0.5, torch.full_like(pair_labels, self.positive_weight), torch.ones_like(pair_labels)
            )
            loss = torch.nn.functional.binary_cross_entropy_with_logits(score, pair_labels, weight=weight)
            loss = loss / weight.mean().clamp(min=1e-6)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.history.append(float(loss.detach().cpu()))
            if ctx.verbose and self.log_every and step % self.log_every == 0:
                print(f"[representation] step {step}: loss={np.mean(self.history[-self.log_every:]):.4f}")
        self.prepare_upstream(ctx)
        return self

    # ----------------------------------------------------------------------------------
    def prepare_upstream(self, ctx: ForecastContext, batch_size: int = 8) -> None:
        """Cache h(x_j, y_j) for every upstream example (one pass over D_PT)."""
        _, pooled, _ = self.encoder.encode_many(
            [e.input for e in ctx.upstream_cache.examples],
            [e.target for e in ctx.upstream_cache.examples],
            batch_size=batch_size,
        )
        self.upstream_reps = pooled

    def scores(self, artifact: OnlineArtifact) -> np.ndarray:
        if self.upstream_reps is None:
            raise RuntimeError("call fit() (or prepare_upstream) first")
        import torch

        with torch.no_grad():
            out = self.encoder([artifact.example.input], [artifact.example.target])
            h_i = out.pooled[0].float().cpu().numpy()
        return self.upstream_reps @ h_i + (self.prior if self.use_prior else 0.0)

    def predict(self, artifact: OnlineArtifact) -> np.ndarray:
        return (self.scores(artifact) > 0.0).astype(int)

    # ----------------------------------------------------------------------------------
    @classmethod
    def without_prior(cls, **kwargs) -> "RepresentationForecaster":
        kwargs["use_prior"] = False
        return cls(**kwargs)
