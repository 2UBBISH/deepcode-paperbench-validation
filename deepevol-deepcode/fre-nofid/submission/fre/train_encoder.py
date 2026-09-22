"""Phase 1 (Algorithm 1) driver: train the FRE encoder/decoder (``q_theta``).

This script trains the permutation-invariant transformer encoder together with the
reward decoder using the information-bottleneck objective of Eq. (6):

    L(theta) = E_{eta ~ p(eta), (s, eta(s)) ~ D} [ MSE( q_theta(eta(s) | s, z), eta(s) ) ]
               + beta * KL( p_theta(z | context) || N(0, I) )

with ``beta = 0.01`` and an encoder context of exactly ``K = 32`` (state, reward)
pairs sampled from the unsupervised reward prior ``p(eta)`` (mixture of
goal-reaching / linear / MLP families).  The decoder is evaluated on ``K' = 8``
held-out states that are disjoint from the encoder context.

Training schedule (paper Section 4 / Appendix):
    * AntMaze:        150,000 steps
    * ExORL / Kitchen: 1,000,000 steps
    * Adam, lr 1e-4, batch size 512 (512 reward functions x 32 context pairs)

The frozen encoder produced here is consumed by ``train_policy.py`` (Phase 2).

Usage
-----
    python -m fre.train_encoder --domain antmaze --steps 150000
    python -m fre.train_encoder --domain exorl --domain-name walker --steps 1000000
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # torch is required for training, but keep the import failure informative.
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "train_encoder.py requires PyTorch. Install it with `pip install torch`."
    ) from exc

# --------------------------------------------------------------------------------------
# Defensive imports: work both as `python -m fre.train_encoder` and standalone.
# --------------------------------------------------------------------------------------

try:  # pragma: no cover - package context
    from .fre.encoder import FREEncoder
except Exception:  # pragma: no cover
    try:
        from fre.fre.encoder import FREEncoder  # type: ignore
    except Exception:
        from fre.encoder import FREEncoder  # type: ignore

try:  # pragma: no cover
    from .fre.decoder import FREDecoder, FREModel
except Exception:  # pragma: no cover
    try:
        from fre.fre.decoder import FREDecoder, FREModel  # type: ignore
    except Exception:
        try:
            from fre.decoder import FREDecoder, FREModel  # type: ignore
        except Exception:  # pragma: no cover
            FREModel = None  # type: ignore
            from fre.decoder import FREDecoder  # type: ignore

try:  # pragma: no cover
    from .fre.vae_loss import (
        DEFAULT_BETA,
        LossOutput,
        VAELoss,
        kl_divergence,
        reward_prediction_loss,
    )
except Exception:  # pragma: no cover
    try:
        from fre.fre.vae_loss import (  # type: ignore
            DEFAULT_BETA,
            LossOutput,
            VAELoss,
            kl_divergence,
            reward_prediction_loss,
        )
    except Exception:
        from fre.vae_loss import (  # type: ignore
            DEFAULT_BETA,
            LossOutput,
            VAELoss,
            kl_divergence,
            reward_prediction_loss,
        )

try:  # pragma: no cover
    from .rewards.prior import (
        CONTEXT_SIZE,
        DECODER_SIZE,
        DEFAULT_BATCH_SIZE as PRIOR_BATCH_SIZE,
        MixturePrior,
        make_mixture_prior,
        make_prior,
    )
except Exception:  # pragma: no cover
    try:
        from fre.rewards.prior import (  # type: ignore
            CONTEXT_SIZE,
            DECODER_SIZE,
            MixturePrior,
            make_mixture_prior,
            make_prior,
        )
    except Exception:  # pragma: no cover
        CONTEXT_SIZE = 32
        DECODER_SIZE = 8
        MixturePrior = None  # type: ignore
        make_mixture_prior = None  # type: ignore
        make_prior = None  # type: ignore

try:  # pragma: no cover
    from .utils.logging import (
        MetricTracker,
        get_logger,
        progress,
        seed_everything,
        set_seed,
        write_json,
    )
except Exception:  # pragma: no cover
    try:
        from fre.utils.logging import (  # type: ignore
            MetricTracker,
            get_logger,
            progress,
            seed_everything,
            set_seed,
            write_json,
        )
    except Exception:  # pragma: no cover
        import logging as _logging

        def seed_everything(seed: int, deterministic: bool = False) -> int:  # type: ignore
            np.random.seed(seed)
            torch.manual_seed(seed)
            return seed

        set_seed = seed_everything  # type: ignore

        def get_logger(name: str = "fre"):  # type: ignore
            return _logging.getLogger(name)

        def write_json(path: str, obj: Any) -> str:  # type: ignore
            import json

            with open(path, "w") as fh:
                json.dump(obj, fh, indent=2, default=str)
            return path

        def progress(iterable=None, total=None, desc=None, **kwargs):  # type: ignore
            return iterable if iterable is not None else range(total or 0)

        class MetricTracker:  # type: ignore
            def __init__(self, *args, **kwargs):
                self._values: Dict[str, List[float]] = {}

            def update(self, values=None, **kwargs):
                data = dict(values or {})
                data.update(kwargs)
                for key, value in data.items():
                    try:
                        self._values.setdefault(key, []).append(float(value))
                    except (TypeError, ValueError):
                        pass

            def mean(self, key: Optional[str] = None):
                if key is None:
                    return {k: (sum(v) / max(len(v), 1)) for k, v in self._values.items()}
                values = self._values.get(key, [])
                return sum(values) / max(len(values), 1)

            def as_dict(self, recent: bool = False) -> Dict[str, float]:
                return {k: (sum(v) / max(len(v), 1)) for k, v in self._values.items()}

            def reset(self):
                self._values = {}

try:  # pragma: no cover - dataset loaders (optional)
    from .data.d4rl_loader import load_antmaze, load_kitchen_multitask
except Exception:  # pragma: no cover
    try:
        from fre.data.d4rl_loader import load_antmaze, load_kitchen_multitask  # type: ignore
    except Exception:  # pragma: no cover
        load_antmaze = None  # type: ignore
        load_kitchen_multitask = None  # type: ignore

try:  # pragma: no cover
    from .data.exorl_loader import load_exorl_dataset
except Exception:  # pragma: no cover
    try:
        from fre.data.exorl_loader import load_exorl_dataset  # type: ignore
    except Exception:  # pragma: no cover
        load_exorl_dataset = None  # type: ignore


__all__ = [
    "EncoderTrainConfig",
    "EncoderTrainer",
    "train_encoder",
    "load_prior_source",
    "main",
]


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

DEFAULT_STEPS = {
    "antmaze": 150_000,
    "exorl": 1_000_000,
    "kitchen": 1_000_000,
}

DEFAULT_LATENT_DIM = 128
DEFAULT_CONTEXT_SIZE = 32
DEFAULT_DECODER_SIZE = 8
DEFAULT_BETA_VALUE = 0.01


@dataclass
class EncoderTrainConfig:
    """Hyper-parameters for Phase-1 FRE encoder/decoder pretraining."""

    domain: str = "antmaze"
    domain_name: Optional[str] = None  # e.g. ExORL "walker"/"cheetah"
    steps: Optional[int] = None
    batch_size: int = 512
    lr: float = 1e-4
    beta: float = DEFAULT_BETA_VALUE
    context_size: int = DEFAULT_CONTEXT_SIZE
    decoder_size: int = DEFAULT_DECODER_SIZE
    latent_dim: int = DEFAULT_LATENT_DIM
    num_blocks: int = 4
    num_heads: int = 4
    mlp_dim: int = 256
    dropout: float = 0.0
    decoder_hidden: Tuple[int, int, int] = (512, 512, 512)
    grad_clip: float = 10.0
    log_interval: int = 1000
    eval_interval: int = 25_000
    checkpoint_interval: int = 50_000
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "runs/fre_encoder"
    dataset: Optional[str] = None
    data_root: Optional[str] = None
    families: Sequence[str] = ("goal_reaching", "linear", "mlp")
    exclude_xy: Optional[bool] = None  # defaults to True for AntMaze
    use_mean_latent_for_eval: bool = True
    log_file: Optional[str] = None
    resume: Optional[str] = None
    dry_run: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.steps is None:
            self.steps = DEFAULT_STEPS.get(self.domain, 150_000)
        if self.exclude_xy is None:
            self.exclude_xy = self.domain == "antmaze"

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["families"] = list(self.families)
        data["decoder_hidden"] = list(self.decoder_hidden)
        return data


# --------------------------------------------------------------------------------------
# Data / prior source
# --------------------------------------------------------------------------------------


def load_prior_source(
    cfg: EncoderTrainConfig,
) -> Tuple[Any, int]:
    """Load the offline dataset that supplies states for the reward prior.

    Returns ``(source, state_dim)`` where ``source`` is whatever object the prior's
    ``set_source`` / ``get_observations`` helpers understand (a dict of arrays or a
    replay buffer).
    """
    source: Any = None

    if cfg.domain == "antmaze":
        if load_antmaze is None:
            raise ImportError(
                "AntMaze training requires the d4rl loader (install d4rl from a "
                "commit dated before June 2024)."
            )
        dataset_name = cfg.dataset or "antmaze-large-diverse-v2"
        source = load_antmaze(dataset_name)
    elif cfg.domain == "exorl":
        if load_exorl_dataset is None:
            raise ImportError("ExORL training requires fre.data.exorl_loader.")
        domain_name = cfg.domain_name or "walker"
        payload = load_exorl_dataset(
            domain_name,
            root=cfg.data_root,
            append_physics_to_obs=True,
            normalise=True,
            verbose=False,
        )
        # The encoder observes raw obs + normalised physics, per the paper.
        source = payload
        if isinstance(payload, dict) and "encoder_observations" in payload:
            source = dict(payload)
            source["observations"] = np.asarray(payload["encoder_observations"])
    elif cfg.domain == "kitchen":
        if load_kitchen_multitask is None:
            raise ImportError("Kitchen training requires fre.data.d4rl_loader.")
        dataset_name = cfg.dataset or "kitchen-complete-v0"
        payload = load_kitchen_multitask(dataset_name)
        source = payload.get("dataset", payload) if isinstance(payload, dict) else payload
    else:
        raise ValueError(f"Unknown domain '{cfg.domain}'.")

    state_dim = _infer_state_dim(source)
    if state_dim is None:
        raise ValueError("Could not infer observation dimension from dataset source.")
    return source, int(state_dim)


def _infer_state_dim(source: Any) -> Optional[int]:
    """Best-effort inference of the observation dimension from a dataset object."""
    if source is None:
        return None
    if isinstance(source, dict):
        for key in ("observations", "encoder_observations", "obs", "states", "state"):
            if key in source:
                arr = np.asarray(source[key])
                if arr.ndim >= 2:
                    return int(arr.shape[-1])
        return None
    for name in ("observations", "obs", "states"):
        if hasattr(source, name):
            try:
                arr = np.asarray(getattr(source, name))
                if arr.ndim >= 2:
                    return int(arr.shape[-1])
            except Exception:
                continue
    for name in ("obs_dim", "observation_dim", "state_dim"):
        if hasattr(source, name):
            try:
                return int(getattr(source, name))
            except Exception:
                continue
    if hasattr(source, "get_observations"):
        try:
            return int(np.asarray(source.get_observations()).shape[-1])
        except Exception:
            return None
    return None


def build_prior(cfg: EncoderTrainConfig, source: Any, state_dim: int) -> Any:
    """Construct the 0.33/0.33/0.33 unsupervised reward prior ``p(eta)``."""
    kwargs: Dict[str, Any] = {
        "source": source,
        "state_dim": state_dim,
        "seed": cfg.seed,
    }
    if cfg.families and tuple(cfg.families) != ("goal_reaching", "linear", "mlp"):
        kwargs["families"] = tuple(cfg.families)
    if cfg.exclude_xy:
        # Paper: remove XY positions from linear reward generation on AntMaze.
        kwargs["exclude_dims"] = (0, 1)

    builder = make_mixture_prior or make_prior
    if builder is None:
        if MixturePrior is None:  # pragma: no cover
            raise ImportError("Could not import the reward prior (MixturePrior).")
        return MixturePrior(**kwargs)  # type: ignore
    try:
        return builder(**kwargs)  # type: ignore
    except TypeError:
        kwargs.pop("families", None)
        return builder(**kwargs)  # type: ignore


def sample_prior_batch(
    prior: Any,
    batch_size: int,
    context_size: int,
    decoder_size: int,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray]:
    """Draw a batch of reward functions with their context / decoder states.

    Returns a dict with ``context_states`` ``(B, K, D)``, ``context_rewards``
    ``(B, K)``, ``decoder_states`` ``(B, K', D)`` and ``decoder_rewards`` ``(B, K')``.
    Falls back to a manual loop over ``sample_functions`` if ``sample_batch`` is not
    available on the prior.
    """
    if hasattr(prior, "sample_batch"):
        result = prior.sample_batch(
            batch_size,
            num_context=context_size,
            num_decoder=decoder_size,
        )
        return _normalize_prior_batch(result, batch_size, context_size, decoder_size)

    # --- fallback: sample function-by-function and build contexts ----
    rng = rng if rng is not None else np.random.default_rng(0)
    functions = prior.sample_functions(batch_size)
    if not isinstance(functions, (list, tuple)):
        functions = [functions]

    ctx_states = np.zeros((len(functions), context_size, prior.state_dim), dtype=np.float32)
    ctx_rewards = np.zeros((len(functions), context_size), dtype=np.float32)
    dec_states = np.zeros((len(functions), decoder_size, prior.state_dim), dtype=np.float32)
    dec_rewards = np.zeros((len(functions), decoder_size), dtype=np.float32)

    for i, fn in enumerate(functions):
        builder = getattr(prior, "_context_builder", None)
        if builder is None:
            raise AttributeError(
                "Prior exposes neither `sample_batch` nor a context builder; cannot "
                "construct encoder contexts."
            )
        out = builder(fn, context_size, decoder_size, rng)  # type: ignore
        ctx_states[i], ctx_rewards[i], dec_states[i], dec_rewards[i] = out

    return {
        "context_states": ctx_states,
        "context_rewards": ctx_rewards,
        "decoder_states": dec_states,
        "decoder_rewards": dec_rewards,
        "functions": list(functions),
    }


def _normalize_prior_batch(
    result: Any,
    batch_size: int,
    context_size: int,
    decoder_size: int,
) -> Dict[str, np.ndarray]:
    """Coerce the many possible prior outputs into a consistent dict of arrays."""
    if isinstance(result, dict):
        ctx_states = result.get("context_states", result.get("encoder_states"))
        ctx_rewards = result.get("context_rewards", result.get("encoder_rewards"))
        dec_states = result.get("decoder_states", result.get("decoder_obs"))
        dec_rewards = result.get("decoder_rewards")
        functions = result.get("functions")
    elif isinstance(result, (tuple, list)):
        if len(result) == 4:
            ctx_states, ctx_rewards, dec_states, dec_rewards = result
            functions = None
        elif len(result) == 5:
            ctx_states, ctx_rewards, dec_states, dec_rewards, functions = result
        else:
            raise ValueError(f"Unexpected prior batch of length {len(result)}.")
    else:  # pragma: no cover
        raise TypeError(f"Unsupported prior batch type: {type(result)!r}")

    out = {
        "context_states": np.asarray(ctx_states, dtype=np.float32),
        "context_rewards": np.asarray(ctx_rewards, dtype=np.float32),
        "decoder_states": None if dec_states is None else np.asarray(dec_states, dtype=np.float32),
        "decoder_rewards": None if dec_rewards is None else np.asarray(dec_rewards, dtype=np.float32),
        "functions": functions,
    }

    if out["context_states"].ndim == 2:  # single function -> add batch dim
        out["context_states"] = out["context_states"][None]
        out["context_rewards"] = np.atleast_2d(out["context_rewards"])
        if out["decoder_states"] is not None and out["decoder_states"].ndim == 2:
            out["decoder_states"] = out["decoder_states"][None]
            out["decoder_rewards"] = np.atleast_2d(out["decoder_rewards"])
    return out


# --------------------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------------------


class EncoderTrainer:
    """Trains the FRE encoder + decoder with the Eq. (6) objective."""

    def __init__(
        self,
        cfg: EncoderTrainConfig,
        state_dim: int,
        prior: Optional[Any] = None,
        device: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self.state_dim = int(state_dim)
        self.device = torch.device(device or cfg.device)
        self.logger = get_logger("fre.train_encoder", log_file=cfg.log_file)

        self.encoder = FREEncoder(
            state_dim=self.state_dim,
            latent_dim=cfg.latent_dim,
            num_blocks=cfg.num_blocks,
            num_heads=cfg.num_heads,
            mlp_dim=cfg.mlp_dim,
            dropout=cfg.dropout,
        ).to(self.device)

        self.decoder = FREDecoder(
            state_dim=self.state_dim,
            latent_dim=cfg.latent_dim,
            hidden_sizes=tuple(cfg.decoder_hidden),
        ).to(self.device)

        self.model = (
            FREModel(self.encoder, self.decoder) if FREModel is not None else None
        )

        self.prior = prior
        self.rng = np.random.default_rng(cfg.seed)

        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = torch.optim.Adam(params, lr=cfg.lr)
        self.loss_fn = VAELoss(beta=cfg.beta)

        self.metrics = MetricTracker()
        self.global_step = 0
        self.history: List[Dict[str, float]] = []
        self._eval_stats: Dict[str, float] = {}

    # -- data ---------------------------------------------------------------------

    def sample_batch(self, batch_size: Optional[int] = None) -> Dict[str, Any]:
        if self.prior is None:
            raise RuntimeError("EncoderTrainer requires a reward prior (set `prior`).")
        batch_size = batch_size or self.cfg.batch_size
        raw = sample_prior_batch(
            self.prior,
            batch_size=batch_size,
            context_size=self.cfg.context_size,
            decoder_size=self.cfg.decoder_size,
            rng=self.rng,
        )
        return self._to_tensors(raw)

    def _to_tensors(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Move a numpy prior batch to the training device and fill decoder defaults."""
        ctx_states = torch.as_tensor(
            raw["context_states"], dtype=torch.float32, device=self.device
        )
        ctx_rewards = torch.as_tensor(
            raw["context_rewards"], dtype=torch.float32, device=self.device
        )

        dec_states = raw.get("decoder_states")
        if dec_states is None:
            # Degenerate fallback: reuse context states (should not normally happen).
            dec_states = raw["context_states"][:, : self.cfg.decoder_size]
        dec_states = torch.as_tensor(dec_states, dtype=torch.float32, device=self.device)

        dec_rewards = raw.get("decoder_rewards")
        if dec_rewards is None:
            dec_rewards = self._evaluate_functions(raw.get("functions"), raw.get("decoder_states"))
        if isinstance(dec_rewards, np.ndarray):
            dec_rewards = torch.as_tensor(
                dec_rewards, dtype=torch.float32, device=self.device
            )

        return {
            "context_states": ctx_states,
            "context_rewards": ctx_rewards,
            "decoder_states": dec_states,
            "decoder_rewards": dec_rewards,
        }

    def _evaluate_functions(
        self, functions: Optional[Sequence[Any]], states: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """Fallback: compute true rewards on decoder states from the reward functions."""
        if functions is None or states is None:
            return None
        rewards = np.zeros(states.shape[:2], dtype=np.float32)
        for i, fn in enumerate(functions):
            try:
                rewards[i] = np.asarray(fn(states[i]), dtype=np.float32).reshape(-1)
            except Exception:
                try:
                    rewards[i] = np.asarray(fn.compute_numpy(states[i]), dtype=np.float32).reshape(-1)
                except Exception:
                    pass
        return rewards

    # -- training -----------------------------------------------------------------

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        self.encoder.train()
        self.decoder.train()

        z, mu, log_sigma = self.encoder.sample(
            batch["context_states"], batch["context_rewards"]
        )
        pred_reward = self.decoder.predict_for_context(batch["decoder_states"], z)

        out = self.loss_fn(
            pred_reward,
            batch["decoder_rewards"],
            mu,
            log_sigma,
            step=self.global_step,
            return_metrics=True,
        )
        if isinstance(out, tuple):
            loss, metrics = out
            metrics = dict(metrics or {})
        else:  # LossOutput / tensor
            metrics = getattr(out, "as_dict", lambda: {})()
            loss = out.total if hasattr(out, "total") else out

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.decoder.parameters()),
                self.cfg.grad_clip,
            )
        self.optimizer.step()
        self.global_step += 1

        stats = {
            "loss": float(loss.detach()),
            "recon_loss": float(metrics.get("recon_loss", metrics.get("recon", float("nan")))),
            "kl_loss": float(metrics.get("kl_loss", metrics.get("kl", float("nan")))),
            "beta": float(metrics.get("beta", self.cfg.beta)),
        }
        self.metrics.update(stats)
        return stats

    def train(self, steps: Optional[int] = None) -> List[Dict[str, float]]:
        steps = steps or int(self.cfg.steps or 0)
        if self.cfg.dry_run:
            steps = min(steps, 5)
        if steps <= 0:
            raise ValueError("Number of training steps must be positive.")

        self.logger.info(
            f"Phase-1 encoder training: domain={self.cfg.domain} "
            f"steps={steps} batch={self.cfg.batch_size} lr={self.cfg.lr} "
            f"beta={self.cfg.beta} K={self.cfg.context_size} K'={self.cfg.decoder_size} "
            f"device={self.device}"
        )

        start = time.time()
        iterator = progress(range(steps), total=steps, desc="fre-encoder")
        for _ in iterator:
            batch = self.sample_batch()
            self.train_step(batch)

            if self.global_step % self.cfg.log_interval == 0:
                self._log_progress(start)

            if self.cfg.eval_interval and self.global_step % self.cfg.eval_interval == 0:
                self.evaluate()

            if (
                self.cfg.checkpoint_interval
                and self.global_step % self.cfg.checkpoint_interval == 0
            ):
                self.save_checkpoint()

        self.evaluate()
        self.save_checkpoint(tag="final")
        elapsed = time.time() - start
        self.logger.info(
            f"Phase-1 training finished after {self.global_step} steps "
            f"({elapsed / 60.0:.1f} min)."
        )
        return self.history

    # -- evaluation / checkpointing -----------------------------------------------

    @torch.no_grad()
    def evaluate(self, num_batches: int = 4, use_mean_latent: Optional[bool] = None) -> Dict[str, float]:
        """Held-out reconstruction check: MSE + KL on freshly sampled reward functions."""
        self.encoder.eval()
        self.decoder.eval()
        use_mean_latent = (
            self.cfg.use_mean_latent_for_eval if use_mean_latent is None else use_mean_latent
        )

        totals = {"eval_mse": 0.0, "eval_rmse": 0.0, "eval_kl": 0.0, "eval_z_std": 0.0}
        count = 0
        for _ in range(max(int(num_batches), 1)):
            batch = self.sample_batch()
            if use_mean_latent and hasattr(self.encoder, "encode"):
                try:
                    z = self.encoder.encode(
                        batch["context_states"], batch["context_rewards"], use_mean=True
                    )
                    _, mu, log_sigma = self.encoder.sample(
                        batch["context_states"], batch["context_rewards"]
                    )
                except TypeError:
                    z, mu, log_sigma = self.encoder.sample(
                        batch["context_states"], batch["context_rewards"]
                    )
            else:
                z, mu, log_sigma = self.encoder.sample(
                    batch["context_states"], batch["context_rewards"]
                )

            pred = self.decoder.predict_for_context(batch["decoder_states"], z)
            target = batch["decoder_rewards"]
            mse = float(torch.nn.functional.mse_loss(pred.reshape_as(target), target))
            kl = float(kl_divergence(mu, log_sigma).mean())

            totals["eval_mse"] += mse
            totals["eval_rmse"] += float(np.sqrt(max(mse, 0.0)))
            totals["eval_kl"] += kl
            totals["eval_z_std"] += float(z.detach().std(dim=0).mean())
            count += 1

        stats = {k: v / max(count, 1) for k, v in totals.items()}
        self._eval_stats = stats
        self.metrics.update(stats)
        self.logger.info(
            f"[step {self.global_step}] eval mse={stats['eval_mse']:.5f} "
            f"rmse={stats['eval_rmse']:.5f} kl={stats['eval_kl']:.3f} "
            f"z_std={stats['eval_z_std']:.3f}"
        )
        return stats

    def _log_progress(self, start_time: float) -> None:
        means = self.metrics.as_dict()
        elapsed = max(time.time() - start_time, 1e-6)
        sps = self.global_step / elapsed
        self.logger.info(
            f"[step {self.global_step}] loss={means.get('loss', float('nan')):.5f} "
            f"recon={means.get('recon_loss', float('nan')):.5f} "
            f"kl={means.get('kl_loss', float('nan')):.4f} "
            f"({sps:.1f} steps/s)"
        )
        self.history.append({"step": self.global_step, **means})
        self.metrics.reset()

    # -- serialization ------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        return {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "config": self.cfg.as_dict(),
            "state_dim": self.state_dim,
            "latent_dim": self.cfg.latent_dim,
            "context_size": self.cfg.context_size,
            "decoder_size": self.cfg.decoder_size,
            "beta": self.cfg.beta,
            "metrics": self._eval_stats,
        }

    def load_state_dict(self, state: Dict[str, Any], load_optimizer: bool = True) -> None:
        self.encoder.load_state_dict(state["encoder"])
        self.decoder.load_state_dict(state["decoder"])
        if load_optimizer and "optimizer" in state:
            try:
                self.optimizer.load_state_dict(state["optimizer"])
            except Exception:  # pragma: no cover
                self.logger.warning("Could not restore optimizer state; continuing fresh.")
        self.global_step = int(state.get("global_step", 0))
        self._eval_stats = dict(state.get("metrics", {}) or {})

    def save_checkpoint(self, tag: Optional[str] = None) -> str:
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        name = "encoder.pt" if tag is None else f"encoder_{tag}.pt"
        path = os.path.join(self.cfg.output_dir, name)
        torch.save(self.state_dict(), path)

        meta_path = os.path.join(self.cfg.output_dir, "encoder_latest.json")
        write_json(
            meta_path,
            {
                "step": self.global_step,
                "config": self.cfg.as_dict(),
                "metrics": self._eval_stats,
                "checkpoint": path,
            },
        )
        self.logger.info(f"Saved Phase-1 checkpoint -> {path}")
        return path

    # -- convenience --------------------------------------------------------------

    @property
    def frozen_encoder(self) -> nn.Module:
        """Encoder with gradients disabled — handed to Phase 2."""
        for param in self.encoder.parameters():
            param.requires_grad_(False)
        self.encoder.eval()
        return self.encoder

    def extra_repr(self) -> str:
        return (
            f"state_dim={self.state_dim}, latent_dim={self.cfg.latent_dim}, "
            f"K={self.cfg.context_size}, K'={self.cfg.decoder_size}"
        )


# --------------------------------------------------------------------------------------
# Top-level entry point
# --------------------------------------------------------------------------------------


def train_encoder(
    cfg: Optional[EncoderTrainConfig] = None,
    prior: Optional[Any] = None,
    source: Optional[Any] = None,
    state_dim: Optional[int] = None,
    **overrides: Any,
) -> EncoderTrainer:
    """Build and run the Phase-1 encoder/decoder training loop.

    Either pass a fully constructed ``prior`` (and ``state_dim``), or let the function
    load the dataset and build the 0.33/0.33/0.33 mixture prior automatically.
    """
    if cfg is None:
        cfg = EncoderTrainConfig(**overrides)
    elif overrides:
        for key, value in overrides.items():
            setattr(cfg, key, value)

    seed_everything(cfg.seed)

    if source is None or state_dim is None:
        source, inferred = load_prior_source(cfg)
        state_dim = state_dim or inferred

    if prior is None:
        prior = build_prior(cfg, source, int(state_dim))

    trainer = EncoderTrainer(cfg, state_dim=int(state_dim), prior=prior)

    if cfg.resume:
        checkpoint = torch.load(cfg.resume, map_location=cfg.device)
        trainer.load_state_dict(checkpoint)
        trainer.logger.info(f"Resumed Phase-1 training from {cfg.resume}")

    trainer.train()
    return trainer


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase-1 FRE encoder/decoder pretraining (Eq. 6)."
    )
    parser.add_argument("--domain", default="antmaze", choices=["antmaze", "exorl", "kitchen"])
    parser.add_argument("--domain-name", default=None, help="ExORL domain: walker | cheetah")
    parser.add_argument("--dataset", default=None, help="Dataset name override")
    parser.add_argument("--data-root", default=None, help="Dataset root dir")
    parser.add_argument("--steps", type=int, default=None, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA_VALUE)
    parser.add_argument("--context-size", type=int, default=DEFAULT_CONTEXT_SIZE)
    parser.add_argument("--decoder-size", type=int, default=DEFAULT_DECODER_SIZE)
    parser.add_argument("--latent-dim", type=int, default=DEFAULT_LATENT_DIM)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mlp-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="runs/fre_encoder")
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=25_000)
    parser.add_argument("--checkpoint-interval", type=int, default=50_000)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> EncoderTrainer:
    args = parse_args(argv)
    output_dir = args.output_dir
    if args.domain_name:
        output_dir = os.path.join(output_dir, args.domain_name)
    cfg = EncoderTrainConfig(
        domain=args.domain,
        domain_name=args.domain_name,
        dataset=args.dataset,
        data_root=args.data_root,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        beta=args.beta,
        context_size=args.context_size,
        decoder_size=args.decoder_size,
        latent_dim=args.latent_dim,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        seed=args.seed,
        device=args.device,
        output_dir=output_dir,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        log_file=args.log_file,
        resume=args.resume,
        dry_run=args.dry_run,
    )
    return train_encoder(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
