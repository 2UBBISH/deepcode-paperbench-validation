"""The Simformer: training (denoising score matching) and sampling.

This module implements Sec. 3 of the paper:

* the joint ``x_hat = (theta, x)`` is embedded with the tokenizer of Sec. 3.1,
* a transformer with an attention mask ``M_E`` predicts the score (Sec. 3.2),
* training uses denoising score matching where the *condition mask* ``M_C`` is
  sampled for every element of a batch (Sec. 3.3, Appendix A2.1),
* arbitrary conditional distributions are sampled by running the reverse SDE on
  the latent variables only, and intervals / arbitrary constraints can be
  enforced with diffusion guidance (Sec. 3.4, Algorithm 1).
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .masks import (dense_mask, inversion_attention_mask, moral_neighbour_mask,
                    undirected)
from .problem import Problem
from .sde import SDE, get_sde
from .tokenizer import IdentifierEmbedding, Tokenizer, TokenizerConfig
from .transformer import TransformerConfig, TransformerScoreNet


# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #
@dataclass
class SimformerConfig:
    # score model / diffusion
    sde: str = "vesde"
    mask_mode: str = "dense"          # "dense" | "undirected" | "directed"
    n_layers: int = 6
    d_model: int = 50
    n_heads: int = 4
    attention_size: int = 10
    widening_factor: float = 3.0
    time_embed_dim: int = 128
    token_mlp: bool = True
    dropout: float = 0.0
    metadata_dim: int = 0
    # condition mask distribution during training
    random_mask_probs: Tuple[float, float] = (0.3, 0.7)
    # which condition masks are sampled during training: "all" (joint, posterior,
    # likelihood and the two random masks, as in the paper) or "posterior" (only
    # the posterior mask, i.e. the "Simformer (posterior only)" variant of
    # Appendix A3.1).
    condition_mask_options: str = "all"
    # maximum number of distinct condition masks per batch for which the exact
    # graph inversion is computed (otherwise a vectorised variant is used),
    max_unique_masks: int = 32
    # sampling
    num_steps: int = 500
    self_recurrence: int = 0
    t_min: float = 1e-5
    t_max: float = 1.0
    seed: int = 0

    def transformer_config(self, n_variables: int) -> TransformerConfig:
        return TransformerConfig(
            n_variables=n_variables,
            d_model=self.d_model,
            n_heads=self.n_heads,
            attention_size=self.attention_size,
            n_layers=self.n_layers,
            widening_factor=self.widening_factor,
            time_embed_dim=self.time_embed_dim,
            dropout=self.dropout,
        )


# --------------------------------------------------------------------------- #
#  Guidance / constraints
# --------------------------------------------------------------------------- #
class Constraint:
    """A constraint ``c(x_hat) <= 0`` used for guided diffusion (Sec. 3.4).

    Guidance modifies the estimated score according to

        ``s(x_t, t | c) ~ s_phi(x_t, t) + grad_{x_t} log sigma(-s(t) c(x_hat_0))``

    where the constraint is evaluated on the denoised estimate
    ``x_hat_0 = (x_t + sigma(t)^2 s_phi(x_t, t)) / mu(t)`` (Bansal et al., 2023).
    """

    name = "constraint"

    def __call__(self, x0_estimate: torch.Tensor) -> torch.Tensor:
        """Return a ``(batch,)`` tensor of constraint values ``c(x_hat_0)``."""
        raise NotImplementedError


class IntervalUpperBound(Constraint):
    """``c(x_hat) = x_hat[i] - u <= 0`` for a set of variables ``i``."""

    name = "interval_upper_bound"

    def __init__(self, variable_indices: Sequence[int], upper: float):
        self.variable_indices = list(variable_indices)
        self.upper = float(upper)

    def __call__(self, x0_estimate: torch.Tensor) -> torch.Tensor:
        vals = x0_estimate[..., self.variable_indices]
        return (vals - self.upper).amax(dim=-1)


class IntervalLowerBound(Constraint):
    """``c(x_hat) = l - x_hat[i] <= 0`` for a set of variables ``i``."""

    name = "interval_lower_bound"

    def __init__(self, variable_indices: Sequence[int], lower: float):
        self.variable_indices = list(variable_indices)
        self.lower = float(lower)

    def __call__(self, x0_estimate: torch.Tensor) -> torch.Tensor:
        vals = x0_estimate[..., self.variable_indices]
        return (self.lower - vals).amax(dim=-1)


class Interval(Constraint):
    """``l <= x_hat[i] <= u``."""

    name = "interval"

    def __init__(self, variable_indices: Sequence[int], lower: float,
                 upper: float):
        self.lower = IntervalLowerBound(variable_indices, lower)
        self.upper = IntervalUpperBound(variable_indices, upper)

    def __call__(self, x0_estimate: torch.Tensor) -> torch.Tensor:
        return torch.maximum(self.lower(x0_estimate), self.upper(x0_estimate))


class LinearConstraint(Constraint):
    """Two sided linear constraint ``|a^T x_hat + b| <= eps``."""

    name = "linear"

    def __init__(self, weights: np.ndarray, bias: float = 0.0,
                 eps: float = 0.0):
        self.weights = torch.as_tensor(np.asarray(weights), dtype=torch.float32)
        self.bias = float(bias)
        self.eps = float(eps)

    def __call__(self, x0_estimate: torch.Tensor) -> torch.Tensor:
        w = self.weights.to(x0_estimate.device, x0_estimate.dtype)
        value = x0_estimate @ w + self.bias
        return value.abs() - self.eps


class CallableConstraint(Constraint):
    """Wrap an arbitrary (differentiable) function ``x_hat -> c(x_hat)``."""

    name = "callable"

    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor],
                 name: str = "callable"):
        self.fn = fn
        self.name = name

    def __call__(self, x0_estimate: torch.Tensor) -> torch.Tensor:
        return self.fn(x0_estimate)


# --------------------------------------------------------------------------- #
#  The model
# --------------------------------------------------------------------------- #
class Simformer(nn.Module):
    """Simformer for a given :class:`~simformer.problem.Problem`."""

    def __init__(self, problem: Problem, config: Optional[SimformerConfig] = None):
        super().__init__()
        self.problem = problem
        self.config = config or SimformerConfig()
        torch.manual_seed(self.config.seed)
        self.sde: SDE = get_sde(
            self.config.sde, t_min=self.config.t_min, t_max=self.config.t_max)

        use_fourier = torch.as_tensor(problem.use_fourier, dtype=torch.bool)
        tokenizer_config = TokenizerConfig(
            n_variables=problem.n_variables,
            d_model=self.config.d_model,
            id_dim=self.config.d_model,
            value_dim=self.config.d_model,
            cond_dim=self.config.d_model,
            metadata_dim=self.config.metadata_dim,
            token_mlp=self.config.token_mlp,
            learnable_identifiers=not bool(use_fourier.any()),
        )
        identifiers = None
        if bool(use_fourier.any()):
            identifiers = IdentifierEmbedding(
                n_variables=problem.n_variables,
                id_dim=self.config.d_model,
                n_kinds=int(problem.n_kinds),
                use_fourier=use_fourier,
                index_dim=problem.index_dim,
            )
        self.net = TransformerScoreNet(
            self.config.transformer_config(problem.n_variables),
            tokenizer=Tokenizer(tokenizer_config),
            identifiers=identifiers,
        )
        self.register_buffer("variable_kind",
                             torch.as_tensor(problem.variable_kind,
                                             dtype=torch.long))
        self.register_buffer("use_fourier",
                             torch.as_tensor(problem.use_fourier,
                                             dtype=torch.bool))
        self._mask_cache: Dict[bytes, np.ndarray] = {}
        self.history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------ problem
    def set_problem(self, problem: Problem) -> "Simformer":
        """Re-target a trained model to a problem with a different layout.

        The transformer is a set function over tokens: its parameters do not
        depend on the number of variables, so a model that was trained on, e.g.,
        10 time points of the contact rate of the SIRD task can be evaluated on
        any other number of (arbitrarily placed) time points.  Only the variable
        bookkeeping (which variable has which kind / index) has to be updated.
        """
        self.problem = problem
        self.variable_kind = torch.as_tensor(problem.variable_kind,
                                             dtype=torch.long)
        self.use_fourier = torch.as_tensor(problem.use_fourier,
                                           dtype=torch.bool)
        self._mask_cache = {}
        return self

    # ------------------------------------------------------------------ masks
    def build_attention_mask(self, condition_state: np.ndarray,
                             index: Optional[np.ndarray] = None,
                             metadata: Optional[dict] = None) -> torch.Tensor:
        """Attention mask ``(batch, n, n)`` for a batch of condition masks.

        For the "directed" variant, the base mask (the directed graphical model
        of the simulator) is adapted with graph inversion (Webb et al., 2018) for
        every condition state, as described in Sec. 3.2 / Appendix A1.1.
        """
        condition_state = np.asarray(condition_state, dtype=float)
        batch = condition_state.shape[0]
        base = np.asarray(self.problem.mask_builder(condition_state, index,
                                                    metadata), dtype=bool)
        if base.ndim == 2:
            base = np.broadcast_to(base[None], (batch,) + base.shape).copy()
        mode = self.config.mask_mode
        if mode == "dense":
            return torch.ones(batch, self.problem.n_variables,
                              self.problem.n_variables, dtype=torch.bool)
        if mode == "undirected":
            masks = np.stack([undirected(b) for b in base])
            return torch.as_tensor(masks, dtype=torch.bool)
        if mode != "directed":
            raise ValueError(f"Unknown mask mode '{mode}'.")

        # group the batch by the (base mask, condition mask) pair: graph
        # inversion only has to be run once per distinct pair.
        combined = np.concatenate(
            [base.reshape(batch, -1), (condition_state > 0.5)], axis=-1)
        unique, inverse = np.unique(combined, axis=0, return_inverse=True)
        masks = np.empty((batch, self.problem.n_variables,
                          self.problem.n_variables), dtype=bool)
        if len(unique) > self.config.max_unique_masks:
            # too many distinct masks in this batch to run the exact
            # (sequential) graph inversion for every one of them: use the
            # vectorised variant that adds the moral graph edges of the latent
            # variables (see simformer.masks.moral_neighbour_mask).
            for b in range(batch):
                masks[b] = moral_neighbour_mask(base[b],
                                                condition_state[b:b + 1])[0]
            return torch.as_tensor(masks, dtype=torch.bool)
        n_base = base[0].size
        for k, row in enumerate(unique):
            base_flat = row[:n_base].reshape(self.problem.n_variables,
                                             self.problem.n_variables)
            pattern = row[n_base:]
            key = row.tobytes()
            cached = self._mask_cache.get(key)
            if cached is None:
                cached = inversion_attention_mask(base_flat, pattern)
                self._mask_cache[key] = cached
            masks[inverse == k] = cached
        return torch.as_tensor(masks, dtype=torch.bool)

    # ------------------------------------------------------- condition masks
    def sample_condition_mask(self, batch_size: int,
                              rng: np.random.Generator) -> np.ndarray:
        """Sample the condition mask ``M_C`` for every element of a batch.

        Uniformly at random among (i) the joint mask (all False), (ii) the
        posterior mask (parameters latent, data conditioned), (iii) the
        likelihood mask (data latent, parameters conditioned) and (iv) two
        randomly sampled masks ``Ber(0.3)`` and ``Ber(0.7)`` (addendum).
        """
        n = self.problem.n_variables
        params = self.problem.param_indices
        data = self.problem.data_indices
        p1, p2 = self.config.random_mask_probs
        if self.config.condition_mask_options == "posterior":
            mask = np.zeros((batch_size, n), dtype=float)
            mask[np.ix_(np.ones(batch_size, dtype=bool), data)] = 1.0
            return mask
        categories = rng.integers(0, 5, size=batch_size)
        mask = np.zeros((batch_size, n), dtype=float)
        is_post = categories == 1
        is_like = categories == 2
        is_r1 = categories == 3
        is_r2 = categories == 4
        if is_post.any():
            mask[np.ix_(is_post, data)] = 1.0
        if is_like.any():
            mask[np.ix_(is_like, params)] = 1.0
        if is_r1.any():
            mask[is_r1] = (rng.random((int(is_r1.sum()), n)) < p1).astype(float)
        if is_r2.any():
            mask[is_r2] = (rng.random((int(is_r2.sum()), n)) < p2).astype(float)
        return mask

    # ------------------------------------------------------------------ loss
    def compute_loss(self, theta: torch.Tensor, x: torch.Tensor,
                     index: Optional[torch.Tensor] = None,
                     metadata: Optional[dict] = None,
                     rng: Optional[np.random.Generator] = None) -> torch.Tensor:
        """Denoising score matching loss with a sampled condition mask (Eq. 4-5)."""
        rng = rng if rng is not None else np.random.default_rng()
        device = theta.device
        batch = theta.shape[0]
        theta = theta.to(dtype=torch.float32)
        x = x.to(dtype=torch.float32)
        x0 = torch.cat([theta, x], dim=-1)

        cond = torch.as_tensor(self.sample_condition_mask(batch, rng),
                               dtype=x0.dtype, device=device)
        t = (torch.rand(batch, device=device) *
             (self.sde.t_max - self.sde.t_min) + self.sde.t_min)
        eps = torch.randn_like(x0)
        x_t, eps, sigma = self.sde.perturb(x0, t, eps)
        # variables we condition on remain clean
        x_t_c = (1.0 - cond) * x_t + cond * x0

        attn_mask = self.build_attention_mask(
            cond.detach().cpu().numpy(),
            None if index is None else index.detach().cpu().numpy(),
            metadata).to(device)
        score = self.net(x_t_c, cond, t, attn_mask, index=index,
                         variable_kind=self.variable_kind.to(device),
                         use_fourier=self.use_fourier.to(device),
                         metadata=None)
        # Analytic denoising score target ``grad log p_t(x_t | x_0)``; for a
        # linear SDE this equals ``-eps / sigma(t)`` which is numerically much
        # more stable than ``-(x_t - mu(t) x_0) / sigma(t)^2`` when ``t`` is
        # small (see simformer.sde.SDE.marginal_score for the analytic form).
        target = -eps / sigma
        weight = self.sde.loss_weight(t)
        residual = (1.0 - cond) * (score - target)
        per_sample = (residual ** 2).sum(dim=-1) * weight
        n_latent = (1.0 - cond).sum(dim=-1).clamp(min=1.0)
        return (per_sample / n_latent).mean()

    # ------------------------------------------------------------------- fit
    def fit(self, theta: np.ndarray, x: np.ndarray,
            index: Optional[np.ndarray] = None,
            metadata: Optional[dict] = None,
            batch_size: int = 1000,
            max_epochs: int = 200,
            lr: float = 1e-4,
            val_fraction: float = 0.1,
            patience: int = 20,
            max_steps: Optional[int] = None,
            max_wall_time: Optional[float] = None,
            seed: int = 0,
            verbose: bool = True,
            optimizer_cls=torch.optim.Adam,
            device: str = "cpu") -> "Simformer":
        """Train the Simformer (Adam, batch size 1000, early stopping on the
        validation loss -- as in Appendix A2.1)."""
        rng = np.random.default_rng(seed)
        self.to(device)
        theta_t = torch.as_tensor(np.asarray(theta), dtype=torch.float32)
        x_t = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        index_t = (None if index is None
                   else torch.as_tensor(np.asarray(index), dtype=torch.float32))
        n_all = theta_t.shape[0]
        perm = rng.permutation(n_all)
        n_val = int(round(val_fraction * n_all))
        val_idx, train_idx = perm[:n_val], perm[n_val:]
        optimizer = optimizer_cls(self.parameters(), lr=lr)
        best_val, best_state, bad_epochs = np.inf, None, 0
        step = 0
        start = time.time()
        for epoch in range(max_epochs):
            self.train()
            order = rng.permutation(len(train_idx))
            epoch_losses = []
            for start_i in range(0, len(order), batch_size):
                idx = train_idx[order[start_i:start_i + batch_size]]
                if len(idx) < 2:
                    continue
                batch_theta = theta_t[idx].to(device)
                batch_x = x_t[idx].to(device)
                batch_index = None if index_t is None else index_t[idx].to(device)
                loss = self.compute_loss(batch_theta, batch_x, batch_index,
                                         metadata, rng)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
                step += 1
                if max_steps is not None and step >= max_steps:
                    break
            # ---------------- validation
            self.eval()
            with torch.no_grad():
                if n_val > 0:
                    val_theta = theta_t[val_idx].to(device)
                    val_x = x_t[val_idx].to(device)
                    val_index = (None if index_t is None
                                 else index_t[val_idx].to(device))
                    val_loss = float(self.compute_loss(
                        val_theta, val_x, val_index, metadata, rng).cpu())
                else:
                    val_loss = float(np.mean(epoch_losses))
            record = {"epoch": epoch, "step": step,
                      "train_loss": float(np.mean(epoch_losses)) if epoch_losses
                      else float("nan"),
                      "val_loss": val_loss, "wall_time": time.time() - start}
            self.history.append(record)
            if verbose:
                print(f"[simformer] epoch {epoch:4d} step {step:6d} "
                      f"train {record['train_loss']:.4f} val {val_loss:.4f}")
            if val_loss < best_val - 1e-6:
                best_val, bad_epochs = val_loss, 0
                best_state = copy.deepcopy(self.state_dict())
            else:
                bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(f"[simformer] early stopping after {epoch + 1} epochs "
                          f"(best val {best_val:.4f})")
                break
            if max_steps is not None and step >= max_steps:
                break
            if (max_wall_time is not None
                    and time.time() - start > max_wall_time):
                break
        if best_state is not None:
            self.load_state_dict(best_state)
        self.eval()
        return self

    # ---------------------------------------------------------------- sampling
    def score(self, values: torch.Tensor, condition_state: torch.Tensor,
              t: torch.Tensor, attention_mask: torch.Tensor,
              index: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.net(values, condition_state, t, attention_mask, index=index,
                        variable_kind=self.variable_kind.to(values.device),
                        use_fourier=self.use_fourier.to(values.device))

    def _prepare_conditions(self, conditions: Sequence[Tuple[np.ndarray, np.ndarray]],
                            num_samples: int, rng: np.random.Generator,
                            device: str, index: Optional[np.ndarray] = None):
        """Build the initial (noisy) latent values and the condition mask."""
        n = self.problem.n_variables
        values = np.zeros((len(conditions) * num_samples, n), dtype=np.float32)
        states = np.zeros_like(values)
        if index is None:
            index_values = np.arange(n, dtype=np.float32)[None].repeat(
                len(conditions) * num_samples, axis=0)
        else:
            index_values = np.asarray(index, dtype=np.float32)
            if index_values.ndim == 1:
                index_values = index_values[None].repeat(
                    len(conditions) * num_samples, axis=0)
            elif index_values.shape[0] == len(conditions):
                index_values = np.repeat(index_values, num_samples, axis=0)
        for k, (value, state) in enumerate(conditions):
            value = np.asarray(value, dtype=np.float32).reshape(-1)
            state = np.asarray(state, dtype=np.float32).reshape(-1)
            sl = slice(k * num_samples, (k + 1) * num_samples)
            values[sl] = value[None]
            states[sl] = state[None]
        x = torch.as_tensor(values, device=device)
        state_t = torch.as_tensor(states, device=device)
        init = self.sde.prior_sample(x.shape, device=device)
        x = state_t * x + (1.0 - state_t) * init
        index_t = torch.as_tensor(index_values, device=device)
        return x, state_t, index_t

    @torch.no_grad()
    def sample(self,
               conditions: Sequence[Tuple[np.ndarray, np.ndarray]],
               num_samples: int = 1,
               index: Optional[np.ndarray] = None,
               metadata: Optional[dict] = None,
               num_steps: Optional[int] = None,
               guidance: Optional[Constraint] = None,
               self_recurrence: Optional[int] = None,
               return_samples: bool = True,
               device: str = "cpu",
               rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample arbitrary conditionals of the joint distribution.

        ``conditions`` is a list of ``(value, state)`` pairs, where ``state`` is
        a ``(n_variables,)`` 0/1 vector (1 = conditioned).  ``num_samples``
        samples are drawn for every condition by running the reverse SDE on the
        latent variables while keeping the conditioned variables fixed
        (Sec. 3.3).  If ``guidance`` is given, the reverse SDE is modified with
        the constraint score (Sec. 3.4 / Algorithm 1).
        """
        self.eval()
        rng = rng if rng is not None else np.random.default_rng()
        num_steps = num_steps or self.config.num_steps
        self_recurrence = (self.config.self_recurrence if self_recurrence is None
                           else self_recurrence)

        x, state, index_t = self._prepare_conditions(conditions, num_samples,
                                                     rng, device, index)

        x = self._reverse_diffusion(x, state, index_t, metadata, num_steps,
                                    guidance, self_recurrence)
        samples = x.detach().cpu().numpy()
        if not return_samples:
            return samples
        return samples.reshape(len(conditions), num_samples, -1)

    def _reverse_diffusion(self, x: torch.Tensor, condition_state: torch.Tensor,
                           index: Optional[torch.Tensor],
                           metadata: Optional[dict], num_steps: int,
                           guidance: Optional[Constraint],
                           self_recurrence: int) -> torch.Tensor:
        """Euler-Maruyama discretisation of the reverse SDE.

        Follows Algorithm 1: the sample is started at the terminal distribution
        and integrated from ``t_max`` to ``t_min``.  For self-recurrence
        (``r > 0``) the step is repeated ``r`` times and the future point is
        re-sampled with the forward SDE.
        """
        dt = (self.sde.t_max - self.sde.t_min) / num_steps
        latent = 1.0 - condition_state
        attn_mask = self.build_attention_mask(
            condition_state.detach().cpu().numpy(),
            None if index is None else index.detach().cpu().numpy(),
            metadata).to(x.device)
        n_inner = max(1, int(self_recurrence) + 1)
        x_cur = x
        for i in range(1, num_steps + 1):
            t_new = self.sde.t_max - i * dt
            t_cur = self.sde.t_max - (i - 1) * dt
            for j in range(n_inner):
                t_tensor = torch.full((x.shape[0],), t_cur, device=x.device)
                if guidance is None:
                    with torch.no_grad():
                        score = self.score(x_cur, condition_state, t_tensor,
                                           attn_mask, index)
                else:
                    score = self._guided_score(x_cur, condition_state, t_tensor,
                                               attn_mask, index, guidance,
                                               t_cur)
                g = self.sde.diffusion(torch.tensor(t_new, device=x.device))
                f = self.sde.drift(x_cur, torch.tensor(t_new, device=x.device))
                eps = torch.randn_like(x_cur) * latent
                update = (f - (g ** 2) * score) * dt + g * math.sqrt(dt) * eps
                x_new = x_cur - update * latent
                if self_recurrence > 0 and j < n_inner - 1:
                    # self recurrence (Lugmayr et al., 2022 / Algorithm 1):
                    # resample the future point with the forward SDE.
                    eps2 = torch.randn_like(x_cur) * latent
                    x_cur = x_new + (f * dt + g * math.sqrt(dt) * eps2) * latent
                else:
                    x_cur = x_new
        return x_cur

    def _guided_score(self, x: torch.Tensor, condition_state: torch.Tensor,
                      t: torch.Tensor, attn_mask: torch.Tensor,
                      index: Optional[torch.Tensor], guidance: Constraint,
                      t_scalar: float) -> torch.Tensor:
        """``s_phi(x_t, t) + grad log sigma(-s(t) c(x_hat_0))``."""
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            score = self.score(x_in, condition_state, t, attn_mask, index)
            mu, sigma = self.sde.marginal_coeff(torch.tensor(t_scalar))
            mu = float(mu)
            sigma = float(sigma)
            x0_estimate = (x_in + sigma ** 2 * score) / mu
            constraint_value = guidance(x0_estimate)
            scale = float(self.sde.guidance_scale(torch.tensor(t_scalar)))
            log_prob = torch.nn.functional.logsigmoid(-scale * constraint_value)
            grad = torch.autograd.grad(log_prob.sum(), x_in)[0]
        return (score + grad).detach()
