"""Strided training controller for Functional Reward Encodings (FRE).

Implements Algorithm 1 from the paper ("Functional Reward Encodings (FRE)",
Section 4.3 "Offline RL with FRE"):

    # Train encoder
    while not converged do
        Sample reward function eta ~ p(eta)
        Sample K states for encoder {s_k^e} ~ D
        Sample K' states for decoder {s_k^d} ~ D
        Train FRE by maximizing Equation (6)
    end while
    # Train policy
    while not converged do
        Sample reward function eta ~ p(eta)
        Sample K states for encoder {s_k^e} ~ D
        Encode into latent vector z ~ p_theta({(s_k^e, eta(s_k^e))})
        Train pi(a|s,z), Q(s,a,z), V(s,z) using IQL with r = eta(s)
    end while

The paper states: "We find that a strided training scheme leads to the most stable
performance. In the strided scheme, we first only train the FRE encoder with
gradients from the decoder (Equation (6)). During this time, the RL components are
not trained. After the encoder loss converges, we freeze the encoder and then start
the training of the RL networks using the frozen encoder's outputs. In this way, we
can make the mapping from eta to z stationary during policy learning, which we found
to be important to correctly estimate multitask Q values using TD learning."

Step counts (Appendix A / Table 3): encoder 150,000 steps (1,000,000 for
ExORL/Kitchen), policy 850,000 steps (1,000,000 for ExORL/Kitchen).
Optimizer: Adam, lr 1e-4, batch size 512.
"""

from __future__ import annotations

import copy
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from fre.fre.fre_model import FREModel, FRELoss
from fre.fre.prior import PriorBatch, PriorSampler, make_prior_sampler
from rl.iql import IQL, make_iql  # noqa: F401  (kept for API symmetry)
from fre.rl.iql import IQL as _IQL  # noqa: F401


# --------------------------------------------------------------------------------------
# Logging helpers
# --------------------------------------------------------------------------------------
class AverageMeter:
    """Tracks a running average (and count) of a scalar quantity."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        v = float(value)
        if not np.isfinite(v):
            return
        self.sum += v * n
        self.count += n

    @property
    def mean(self) -> float:
        return self.sum / max(self.count, 1)

    def __float__(self) -> float:  # pragma: no cover - convenience
        return self.mean


@dataclass
class PhaseStats:
    """Aggregate statistics for one training phase."""

    phase: str
    steps: int
    seconds: float
    metrics: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"phase": self.phase, "steps": self.steps, "seconds": self.seconds, **self.metrics}


def _to_tensor(value, device, dtype=None) -> torch.Tensor:
    """Convert numpy/list/array input to a tensor on ``device``."""
    if isinstance(value, torch.Tensor):
        t = value.to(device)
        return t if dtype is None else t.to(dtype)
    t = torch.as_tensor(np.asarray(value), device=device)
    return t if dtype is None else t.to(dtype)


# --------------------------------------------------------------------------------------
# Dataset adapter
# --------------------------------------------------------------------------------------
class DatasetAdapter:
    """Uniform access to the offline dataset for prior sampling.

    Accepts either a :class:`fre.rl.replay_buffer.ReplayBuffer` (duck-typed) or raw
    numpy arrays.  ``encoder_states`` optionally returns *physics-augmented* states
    (Appendix C.2) via ``ReplayBuffer.augmented_states``.
    """

    def __init__(self, dataset, use_physics: bool = False):
        self.dataset = dataset
        self.use_physics = bool(use_physics) and hasattr(dataset, "augmented_states")

    # -- properties proxied from the underlying buffer (duck typing) -------------------
    @property
    def num_states(self) -> int:
        if hasattr(self.dataset, "num_states"):
            return int(self.dataset.num_states)
        return int(len(self.dataset.observations))

    @property
    def state_std(self):
        return getattr(self.dataset, "state_std", None)

    @property
    def obs_std(self):
        return getattr(self.dataset, "obs_std", None)

    @property
    def states(self):
        return getattr(self.dataset, "states", None)

    # -- sampling --------------------------------------------------------------------
    def sample_raw_states(self, num: int, rng: Optional[np.random.Generator] = None):
        """Sample ``num`` base observation vectors from the dataset."""
        if hasattr(self.dataset, "sample_states"):
            return self.dataset.sample_states(num)
        obs = np.asarray(self.dataset.observations)
        idx = np.random.randint(0, len(obs), size=num)
        return obs[idx]

    def sample_states(self, num: int, rng: Optional[np.random.Generator] = None):
        """Sample ``num`` encoder states (physics-augmented when requested)."""
        raw = self.sample_raw_states(num, rng=rng)
        if self.use_physics:
            return self.dataset.augmented_states(raw)
        return raw

    def sample_trajectories(self, num: int, rng: Optional[np.random.Generator] = None):
        """Sample trajectories for goal-reaching (HER) prior functions."""
        if hasattr(self.dataset, "sample_trajectories"):
            try:
                trajs = self.dataset.sample_trajectories(num)
            except TypeError:  # pragma: no cover - signature drift tolerance
                trajs = self.dataset.sample_trajectories()
        else:  # pragma: no cover - raw-array fallback (single pseudo-episode)
            obs = np.asarray(self.dataset.observations)
            trajs = obs[None, :, :]
        if self.use_physics and trajs is not None:
            trajs = self.dataset.augmented_states(trajs)
        return trajs

    def sample_transitions(self, num: int):
        """Sample ``(obs, act, rew, next_obs, term)`` transition tuples for IQL."""
        return self.dataset.sample_transitions(num)


# --------------------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------------------
class FRETrainer:
    """Two-phase (strided) FRE trainer implementing Algorithm 1.

    Phase 1 ("encoder"): jointly train the encoder + decoder by maximizing Eq. (6),
    i.e. minimizing ``MSE(reward reconstruction) + beta * KL(q(z|L^e) || N(0, I))``.
    The RL components are explicitly *not* trained.

    Phase 2 ("policy"): freeze the encoder, then train the z-conditioned IQL agent
    ``pi(a|s,z), Q(s,a,z), V(s,z)`` with rewards ``r = eta(s)`` sampled from the same
    prior reward distribution.

    Parameters mirror Table 3 (Appendix A) of the paper.
    """

    def __init__(
        self,
        model: FREModel,
        prior: PriorSampler,
        dataset,
        config=None,
        iql: Optional[_IQL] = None,
        device: str = "cpu",
        encoder_steps: int = 150_000,
        policy_steps: int = 850_000,
        batch_size: int = 512,
        learning_rate: float = 1e-4,
        grad_clip_norm: float = 10.0,
        log_interval: int = 1000,
        eval_interval: int = 10_000,
        save_interval: int = 50_000,
        output_dir: Optional[str] = None,
        seed: int = 0,
        use_physics_augmentation: bool = False,
        train_ratio: float = 1.0,
        num_policy_rewards_per_batch: int = 1,
        beta_kl: Optional[float] = None,
        logger: Optional[Callable[[Dict[str, Any], int], None]] = None,
    ):
        self.model = model.to(device)
        self.prior = prior
        self.dataset = dataset
        self.data = DatasetAdapter(dataset, use_physics=use_physics_augmentation)
        self.config = config
        self.device = device
        self.seed = int(seed)

        # -- step budgets (Table 3) ---------------------------------------------------
        if config is not None:
            if hasattr(config, "encoder_steps") and encoder_steps is None:
                encoder_steps = config.encoder_steps()
            if hasattr(config, "policy_steps") and policy_steps is None:
                policy_steps = config.policy_steps()
        self.encoder_steps = int(encoder_steps)
        self.policy_steps = int(policy_steps)
        self.train_ratio = float(train_ratio)

        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.grad_clip_norm = float(grad_clip_norm)
        self.log_interval = int(log_interval)
        self.eval_interval = int(eval_interval)
        self.save_interval = int(save_interval)
        self.output_dir = output_dir
        self.logger = logger
        self.num_policy_rewards_per_batch = max(1, int(num_policy_rewards_per_batch))

        if beta_kl is not None:
            self.model.beta_kl = float(beta_kl)

        # -- optimizers ---------------------------------------------------------------
        betas = getattr(config, "adam_betas", (0.9, 0.999)) if config is not None else (0.9, 0.999)
        self.encoder_optimizer = torch.optim.Adam(
            list(self.model.trainable_parameters()), lr=self.learning_rate, betas=tuple(betas)
        )

        # -- RL agent (phase 2) -------------------------------------------------------
        self.iql = iql
        self.obs_dim = self._infer_obs_dim()
        self.action_dim = self._infer_action_dim()
        if self.iql is None and self.action_dim is not None:
            self.iql = _IQL(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                latent_dim=self.model.latent_dim,
                hidden_layers=getattr(config, "rl_hidden_layers", (512, 512, 512)),
                activation=getattr(config, "rl_activation", "relu"),
                expectile=getattr(config, "iql_expectile", 0.8),
                temperature=getattr(config, "iql_temperature", 3.0),
                discount=getattr(config, "discount", 0.88),
                target_update_rate=getattr(config, "target_update_rate", 0.001),
                learning_rate=self.learning_rate,
                grad_clip_norm=self.grad_clip_norm,
                log_std_min=getattr(config, "log_std_min", -5.0),
                log_std_max=getattr(config, "log_std_max", 2.0),
                num_qs=getattr(config, "rl_num_qs", 2),
                layernorm=getattr(config, "rl_layernorm", False),
                device=device,
            )

        self.frozen = False
        self.step = 0
        self.phase_stats: List[PhaseStats] = []
        self.history: List[Dict[str, Any]] = []
        self._rng = np.random.RandomState(self.seed)

    # ----------------------------------------------------------------------------------
    # Dimension inference
    # ----------------------------------------------------------------------------------
    def _infer_obs_dim(self) -> int:
        for attr in ("encoder_state_dim", "state_dim", "obs_dim"):
            if hasattr(self.dataset, attr):
                return int(getattr(self.dataset, attr))
        return int(self.data.sample_states(1).shape[-1])

    def _infer_action_dim(self) -> Optional[int]:
        for attr in ("action_dim",):
            if hasattr(self.dataset, attr):
                return int(getattr(self.dataset, attr))
        actions = getattr(self.dataset, "actions", None)
        if actions is not None:
            return int(np.asarray(actions).shape[-1])
        return None

    # ----------------------------------------------------------------------------------
    # Phase 1: encoder (+ decoder) training, Eq. (6)
    # ----------------------------------------------------------------------------------
    def encoder_loss(self, batch: Optional[PriorBatch] = None) -> FRELoss:
        """Compute one Eq. (6) loss for a freshly sampled batch of reward functions."""
        if batch is None:
            batch = self.sample_prior_batch(self.batch_size)
        enc_states = _to_tensor(batch.encoder_states, self.device)
        enc_rewards = _to_tensor(batch.encoder_rewards, self.device)
        dec_states = _to_tensor(batch.decoder_states, self.device)
        dec_rewards = _to_tensor(batch.decoder_rewards, self.device)
        return self.model.loss(enc_states, enc_rewards, dec_states, dec_rewards)

    def sample_prior_batch(self, batch_size: int) -> PriorBatch:
        """Sample eta ~ p(eta) and the K / K' state sets from D (Algorithm 1)."""
        try:
            return self.prior.sample_batch(dataset=self.data.dataset, batch_size=batch_size)
        except TypeError:  # pragma: no cover - older sampler signature
            return self.prior.sample_batch(batch_size=batch_size)

    def train_encoder(self, num_steps: Optional[int] = None) -> PhaseStats:
        """Phase 1 of Algorithm 1: train FRE with decoder gradients only."""
        num_steps = int(self.encoder_steps if num_steps is None else num_steps)
        self.model.train()
        if self.iql is not None:
            self.iql.eval()
        meters = {k: AverageMeter(k) for k in ("loss", "reconstruction", "kl", "reward_std")}
        t0 = time.time()
        for _ in range(num_steps):
            batch = self.sample_prior_batch(self.batch_size)
            out = self.encoder_loss(batch)

            self.encoder_optimizer.zero_grad(set_to_none=True)
            out.loss.backward()
            if self.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(
                    list(self.model.trainable_parameters()), self.grad_clip_norm
                )
            self.encoder_optimizer.step()

            meters["loss"].update(out.loss.detach().item())
            meters["reconstruction"].update(out.reconstruction.detach().item())
            meters["kl"].update(out.kl.detach().item())
            meters["reward_std"].update(np.asarray(batch.decoder_rewards).std())

            self.step += 1
            if self.log_interval and self.step % self.log_interval == 0:
                record = {"step": self.step, "phase": "encoder"}
                record.update({k: m.mean for k, m in meters.items()})
                self._log(record)
                for m in meters.values():
                    m.reset()

        stats = PhaseStats(
            phase="encoder",
            steps=num_steps,
            seconds=time.time() - t0,
            metrics={
                "loss": meters["loss"].mean,
                "reconstruction": meters["reconstruction"].mean,
                "kl": meters["kl"].mean,
            },
        )
        self.phase_stats.append(stats)
        return stats

    # ----------------------------------------------------------------------------------
    # Freezing
    # ----------------------------------------------------------------------------------
    def freeze_encoder(self) -> None:
        """Freeze the encoder after Phase 1 ("we freeze the encoder").

        ``trainable_parameters`` on the model yields only decoder parameters (and any
        parameters not belonging to the encoder), so Phase 2 touches only the IQL nets.
        """
        for p in self.model.encoder.parameters():
            p.requires_grad_(False)
        self.model.encoder.eval()
        self.frozen = True

    def unfreeze_encoder(self) -> None:  # pragma: no cover - ablation utility
        for p in self.model.encoder.parameters():
            p.requires_grad_(True)
        self.model.encoder.train()
        self.frozen = False

    # ----------------------------------------------------------------------------------
    # Phase 2: IQL on the frozen encoder, r = eta(s)
    # ----------------------------------------------------------------------------------
    @torch.no_grad()
    def encode_batch(self, transitions: Dict[str, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample eta, encode z from K encoding states, and evaluate r = eta(s) on the batch.

        Mirrors Algorithm 1 / Section 4.3:

            Sample reward function eta ~ p(eta)
            Sample K states for encoder {s_k^e} ~ D
            Encode into latent vector z ~ p_theta({(s_k^e, eta(s_k^e))})
            Train ... using IQL with r = eta(s)

        Returns ``(z, rewards)`` where ``z`` has shape ``(batch, latent_dim)`` and
        ``rewards`` has shape ``(batch,)`` for the sampled reward function(s).
        """
        n = int(transitions["observations"].shape[0])
        obs = np.asarray(transitions["observations"])
        next_obs = np.asarray(transitions["next_observations"])

        # K encoding states sampled from D (physics-augmented for ExORL, Appendix C.2)
        enc_states = self.data.sample_states(self.prior.num_encoder_samples)
        enc_states_t = _to_tensor(enc_states, self.device, dtype=torch.float32)

        zs, rewards = [], []
        for _ in range(self.num_policy_rewards_per_batch):
            batch = self.prior.sample_rewards(
                encoder_states=enc_states,
                decoder_states=enc_states,
            )
            enc_rewards_t = _to_tensor(batch.encoder_rewards, self.device, dtype=torch.float32)
            enc_rewards_t = enc_rewards_t.reshape(1, -1) if enc_rewards_t.dim() == 1 else enc_rewards_t
            z = self.model.encode(enc_states_t, enc_rewards_t, sample=True)  # z ~ p_theta(z | L^e)
            zs.append(z)
            # r = eta(s) evaluated on the *batch* transitions (raw, matching obs in the batch)
            r = self.prior.evaluate_params(
                params=batch.params, states=obs, function_types=batch.function_types
            )
            rewards.append(_to_tensor(r, self.device, dtype=torch.float32).reshape(-1))

        z_stack = torch.cat(zs, dim=0)  # (num_rewards * batch, latent_dim)
        r_stack = torch.cat(rewards, dim=0)  # (num_rewards * batch,)
        return z_stack, r_stack

    def train_policy(self, num_steps: Optional[int] = None) -> PhaseStats:
        """Phase 2 of Algorithm 1: train pi/Q/V with IQL on the frozen encoder."""
        num_steps = int(self.policy_steps if num_steps is None else num_steps)
        if self.iql is None:
            raise RuntimeError(
                "Cannot run the policy phase without an IQL agent; the dataset must "
                "expose actions (offline RL transitions)."
            )
        if not self.frozen:
            self.freeze_encoder()

        self.iql.train()
        meters = {
            k: AverageMeter(k) for k in ("q_loss", "v_loss", "policy_loss", "q_mean", "v_mean")
        }
        t0 = time.time()
        for _ in range(num_steps):
            n = self.batch_size
            transitions = self.data.sample_transitions(n)
            z, rewards = self.encode_batch(transitions)
            # z/rewards may be replicated over multiple sampled reward functions
            gen = z.shape[0]
            if gen != n:
                reps = gen // n
                transitions = {
                    k: np.repeat(np.asarray(v), reps, axis=0) for k, v in transitions.items()
                }
            info = self.iql.update(transitions, rewards=rewards, z=z)
            for k, m in meters.items():
                if k in info:
                    m.update(info[k])

            self.step += 1
            if self.log_interval and self.step % self.log_interval == 0:
                record = {"step": self.step, "phase": "policy"}
                record.update({k: m.mean for k, m in meters.items()})
                self._log(record)

        stats = PhaseStats(
            phase="policy",
            steps=num_steps,
            seconds=time.time() - t0,
            metrics={k: m.mean for k, m in meters.items()},
        )
        self.phase_stats.append(stats)
        return stats

    # ----------------------------------------------------------------------------------
    # Combined entry point
    # ----------------------------------------------------------------------------------
    def train(
        self,
        encoder_steps: Optional[int] = None,
        policy_steps: Optional[int] = None,
        freeze: bool = True,
    ) -> Dict[str, Any]:
        """Run the full strided scheme: encoder phase then frozen-encoder policy phase."""
        enc_stats = self.train_encoder(encoder_steps)
        if freeze:
            self.freeze_encoder()
        policy_stats = None
        if self.iql is not None:
            policy_stats = self.train_policy(policy_steps)
        return {
            "encoder": enc_stats.as_dict(),
            "policy": None if policy_stats is None else policy_stats.as_dict(),
            "frozen_encoder": self.frozen,
        }

    # ----------------------------------------------------------------------------------
    # Evaluation / diagnostics hooks
    # ----------------------------------------------------------------------------------
    @torch.no_grad()
    def reconstruction_mse(self, num_batches: int = 8, batch_size: Optional[int] = None) -> float:
        """Held-out reward-reconstruction MSE of the decoder (encoder diagnostics)."""
        self.model.eval()
        bs = int(batch_size or self.batch_size)
        vals = []
        for _ in range(num_batches):
            batch = self.sample_prior_batch(bs)
            vals.append(
                self.model.reconstruction_mse(
                    _to_tensor(batch.encoder_states, self.device),
                    _to_tensor(batch.encoder_rewards, self.device),
                    _to_tensor(batch.decoder_states, self.device),
                    _to_tensor(batch.decoder_rewards, self.device),
                )
            )
        self.model.train()
        return float(np.mean(vals))

    # ----------------------------------------------------------------------------------
    # Persistence & logging
    # ----------------------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        sd = {
            "model": self.model.state_dict(),
            "encoder_optimizer": self.encoder_optimizer.state_dict(),
            "step": self.step,
            "frozen": self.frozen,
            "seed": self.seed,
        }
        if self.iql is not None:
            sd["iql"] = self.iql.state_dict()
        return sd

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self.model.load_state_dict(sd["model"])
        if "encoder_optimizer" in sd:
            self.encoder_optimizer.load_state_dict(sd["encoder_optimizer"])
        self.step = int(sd.get("step", 0))
        self.frozen = bool(sd.get("frozen", False))
        if self.iql is not None and "iql" in sd:
            self.iql.load_state_dict(sd["iql"])

    def save(self, path: str, extras: Optional[Dict[str, Any]] = None) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = self.state_dict()
        if extras:
            payload.update(extras)
        torch.save(payload, path)

    def load(self, path: str, map_location: Optional[str] = None) -> "FRETrainer":
        payload = torch.load(path, map_location=map_location or self.device)
        self.load_state_dict(payload)
        return self

    def _log(self, record: Dict[str, Any]) -> None:
        self.history.append(record)
        if self.logger is not None:
            try:
                self.logger(record, self.step)
                return
            except TypeError:  # pragma: no cover - single-arg logger
                self.logger(record)
                return
        msg = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items())
        print(f"[FRE] {msg}", flush=True)

    # ----------------------------------------------------------------------------------
    # Factories
    # ----------------------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        config,
        dataset,
        iql: Optional[_IQL] = None,
        action_dim: Optional[int] = None,
        encoder: Optional[nn.Module] = None,
        **overrides,
    ) -> "FRETrainer":
        """Build a trainer from a :class:`fre.config.default.Config`-like object."""
        device = overrides.pop("device", getattr(config, "device", "cpu"))
        use_physics = overrides.pop(
            "use_physics_augmentation",
            bool(getattr(config, "use_physics_augmentation", False))
            or (str(getattr(config, "domain", "")).startswith("exorl")
                and str(getattr(config, "exorl_physics_augmentation", "")).lower() not in ("", "none", "false")),
        )

        # Encoder state dim = physics-augmented dim when ExORL physics augmentation is on.
        state_dim = getattr(dataset, "encoder_state_dim", None)
        if state_dim is None:
            state_dim = getattr(dataset, "obs_dim", None)
        if state_dim is None:
            state_dim = int(np.asarray(dataset.observations).shape[-1])
        if use_physics and hasattr(dataset, "augmented_states"):
            state_dim = int(dataset.augmented_states(np.zeros((1, dataset.obs_dim))).shape[-1])
        state_dim = overrides.pop("state_dim", int(state_dim))

        model = overrides.pop("model", None)
        if model is None:
            model = FREModel.from_config(config, state_dim=state_dim, encoder=encoder)

        prior = overrides.pop("prior", None)
        if prior is None:
            prior = make_prior_sampler(config, state_dim=state_dim)

        train_ratio = overrides.pop("train_ratio", 1.0)
        enc_steps = overrides.pop("encoder_steps", None)
        pol_steps = overrides.pop("policy_steps", None)
        if enc_steps is None:
            enc_steps = getattr(config, "encoder_steps", None)
            enc_steps = enc_steps() if callable(enc_steps) else enc_steps
        if pol_steps is None:
            pol_steps = getattr(config, "policy_steps", None)
            pol_steps = pol_steps() if callable(pol_steps) else pol_steps
        if enc_steps is None:
            enc_steps = int(round(150_000 * train_ratio))
        if pol_steps is None:
            pol_steps = int(round(850_000 * train_ratio))

        if action_dim is None:
            action_dim = getattr(dataset, "action_dim", None)
        if iql is None and action_dim is not None:
            iql = _IQL.from_config(
                config,
                obs_dim=state_dim,
                action_dim=action_dim,
                device=device,
                **overrides.pop("iql_kwargs", {}),
            )

        trainer = cls(
            model=model,
            prior=prior,
            dataset=dataset,
            config=config,
            iql=iql,
            device=device,
            encoder_steps=int(enc_steps),
            policy_steps=int(pol_steps),
            batch_size=overrides.pop("batch_size", getattr(config, "batch_size", 512)),
            learning_rate=overrides.pop("learning_rate", getattr(config, "learning_rate", 1e-4)),
            grad_clip_norm=overrides.pop("grad_clip_norm", getattr(config, "grad_clip_norm", 10.0)),
            log_interval=overrides.pop("log_interval", 1000),
            output_dir=overrides.pop("output_dir", None),
            seed=overrides.pop("seed", getattr(config, "seed", 0)),
            use_physics_augmentation=use_physics,
            train_ratio=train_ratio,
            **overrides,
        )
        return trainer


# --------------------------------------------------------------------------------------
# Convenience functions
# --------------------------------------------------------------------------------------
def make_trainer(config, dataset, **kwargs) -> FRETrainer:
    """Functional wrapper around :meth:`FRETrainer.from_config`."""
    return FRETrainer.from_config(config, dataset, **kwargs)


def train_fre(config, dataset, **kwargs) -> Dict[str, Any]:
    """Build and run a full FRE training run (Algorithm 1)."""
    trainer = make_trainer(config, dataset, **kwargs)
    return trainer.train(**_phase_overrides(config, kwargs))


def _phase_overrides(config, kwargs) -> Dict[str, Any]:  # pragma: no cover - tiny helper
    out = {}
    for k in ("encoder_steps", "policy_steps", "freeze"):
        if k in kwargs:
            out[k] = kwargs[k]
    return out


__all__ = [
    "FRETrainer",
    "DatasetAdapter",
    "PhaseStats",
    "AverageMeter",
    "make_trainer",
    "train_fre",
]
