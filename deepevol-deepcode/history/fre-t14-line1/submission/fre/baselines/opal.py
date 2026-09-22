"""OPAL baseline (Ajay et al., 2020) re-implemented inside the FRE codebase.

Paper references
----------------
* Section 5.2: "OPAL (Ajay et al., 2020), a representative offline unsupervised
  skill discovery method where latent skills are learned by auto-encoding
  trajectories."  Table 1 reports the ``OPAL-10`` column (privileged evaluation
  with 10 sampled skills).
* Addendum ("Additional Details on OPAL"):

  - "No manually designed rewards are used in OPAL."
  - "For the OPAL encoder, the same transformer architecture is used as in FRE."
  - "For the privileged execution evaluation described in the paper:
       * OPAL's task policy is not used
       * 10 random skills are sampled from a unit Gaussian,
       * for each skill ``z``, the policy is conditioned on it and evaluated for
         the entire episode,
       * and the best performing rollout is taken."

Design (faithful re-implementation)
-----------------------------------
OPAL learns a latent skill space by *auto-encoding trajectories* while maximising
the mutual information between the skill and the states visited under it:

    max_{q(z|tau), p(s'|s,z), pi}  I(tau; z) + I(s'; z)   (state-conditioned MI)

which, in the original paper, decomposes into

  (1) a trajectory variational auto-encoder  (encoder ``q(z | tau)`` -> N(0, I)
      prior + decoder ``p(s_{t+k} | s_t, z)``) whose reconstruction term is the
      *discrimination* objective  ``E[log q_theta(z | s_{t+k}) ]`` /
      ``E[log p_theta(s_{t+k} | s_t, z)]``, and
  (2) an off-policy RL phase that trains ``pi(a | s, z)`` and ``Q(s, a, z)`` to
      maximise the intrinsic skill reward ``r(s, z) = log q_theta(z | s) - log u(z)``,
      i.e. the mutual information between the skill and the current/future state.

Because the paper evaluates OPAL with a *privileged* protocol (10 skills sampled
from the unit Gaussian, best rollout kept, no task policy), the task-conditioned
policy is irrelevant at evaluation time; what matters is that a diverse set of
conditioned behaviours was learned.  We therefore instantiate exactly that
pipeline here:

* ``SkillEncoder``      : state-token transformer (identical architecture to
                          :class:`fre.fre.encoder.Encoder`, but without reward
                          tokens -- OPAL has no reward inputs) producing N(0, I)
                          regularised Gaussian skills ``z`` (128-dim).
* ``SkillDecoder``      : MLP ``p(s_{t+k} | s_t, z)`` predicting future states.
* ``OPALModel``         : joint encoder/decoder, loss = MSE + beta * KL(N(0, I)).
* ``OPALAgent``         : skill pretraining (phase 1) + z-conditioned IQL on the
                          intrinsic skill reward ``log q(z | s)`` (phase 2).
* ``evaluate_opal_privileged`` : 10 unit-Gaussian skills, full-episode rollouts,
                          best rollout taken (``OPAL-10`` in Table 1).

Reference Table 1 numbers (``OPAL-10``)::

    antmaze-all  45.6 +/- 17.0
    exorl-all    28.2 +/-  4.0
    kitchen      26   +/- 16
    all          33   +/- 12

"""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fre.fre.encoder import TransformerEncoder, diagonal_gaussian_kl
from fre.rl.iql import IQL, make_iql
from fre.rl.networks import MLP, GaussianPolicy, make_activation

__all__ = [
    # networks / model
    "SkillEncoder",
    "SkillDecoder",
    "OPALModel",
    "OPALLoss",
    # agent
    "OPALAgent",
    "make_opal",
    "train_opal",
    # trajectory helpers
    "TrajectoryWindows",
    "sample_trajectory_batch",
    # evaluation
    "make_opal_policy_fn",
    "evaluate_opal_privileged",
    "OPAL_NUM_SKILLS",
    "OPAL_TABLE1_REFERENCE",
]


# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

#: Number of skills sampled from the unit Gaussian for the privileged evaluation
#: (addendum: "10 random skills are sampled from a unit Gaussian").
OPAL_NUM_SKILLS = 10

#: Table 1 reference numbers for the OPAL-10 column (mean, std across 5 seeds).
OPAL_TABLE1_REFERENCE: Dict[str, Tuple[float, float]] = {
    "antmaze": (45.6, 17.0),
    "exorl": (28.2, 4.0),
    "kitchen": (26.0, 16.0),
    "all": (33.0, 12.0),
    # per-task AntMaze / ExORL rows (from Table 1)
    "ant-goal-reaching": (19.4, 12.0),
    "ant-directional": (39.4, 13.0),
    "ant-random-simplex": (27.3, 8.0),
    "ant-path-loop": (44.4, 22.0),
    "ant-path-edges": (85.0, 10.0),
    "ant-path-center": (58.1, 36.0),
    "exorl-walker-goals": (31.0, 6.0),
    "exorl-cheetah-goals": (26.0, 2.0),
    "exorl-walker-velocity": (17.0, 2.0),
    "exorl-cheetah-velocity": (38.0, 12.0),
}

#: Default number of future steps predicted by the skill decoder (state-conditioned MI).
OPAL_PREDICTION_HORIZON = 5

#: Default trajectory length used for skill encoding during pretraining.
OPAL_TRAJECTORY_LENGTH = 64


# --------------------------------------------------------------------------------------
# Trajectory helpers
# --------------------------------------------------------------------------------------


class TrajectoryWindows:
    """Flat-dataset trajectory bookkeeping for OPAL's trajectory auto-encoding.

    Provides

    * ``window(indices, length)`` : gather ``(len(indices), length, state_dim)``
      contiguous state windows starting at the given flat transition indices (the
      window is clamped to the end of the trajectory it belongs to), and
    * ``future_state(indices, k)`` : gather the state ``k`` steps ahead
      (clamped inside the same trajectory).

    Episode boundaries are inferred from ``timeouts``/``terminals`` masks, from an
    explicit ``trajectory_ids`` array, or from a fixed ``episode_length`` — the
    same convention used by :mod:`fre.rl.replay_buffer`.
    """

    def __init__(
        self,
        states: np.ndarray,
        traj_ids: Optional[np.ndarray] = None,
        terminals: Optional[np.ndarray] = None,
        timeouts: Optional[np.ndarray] = None,
        episode_length: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        states = np.asarray(states)
        if states.ndim != 2:
            raise ValueError(f"states must be 2-D, got shape {states.shape}")
        self.states = states
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        n = states.shape[0]
        if traj_ids is not None and len(np.asarray(traj_ids)) == n:
            self.traj_ids = np.asarray(traj_ids).astype(np.int64)
        elif episode_length is not None and episode_length > 0:
            self.traj_ids = (np.arange(n) // int(episode_length)).astype(np.int64)
        else:
            ends = np.zeros(n, dtype=bool)
            if terminals is not None and len(np.asarray(terminals)) == n:
                ends |= np.asarray(terminals).astype(bool)
            if timeouts is not None and len(np.asarray(timeouts)) == n:
                ends |= np.asarray(timeouts).astype(bool)
            if not ends.any():
                # No episode information at all: treat data as one long trajectory.
                self.traj_ids = np.zeros(n, dtype=np.int64)
            else:
                self.traj_ids = np.concatenate([[0], np.cumsum(ends[:-1])]).astype(np.int64)

        # trajectory end indices (exclusive) per trajectory id
        self._ends: Dict[int, int] = {}
        ids = self.traj_ids
        if n:
            change = np.nonzero(np.diff(ids))[0]
            starts = np.concatenate([[0], change + 1])
            stops = np.concatenate([change + 1, [n]])
            for s, e in zip(starts, stops):
                self._ends[int(ids[s])] = int(e)

    # -- basic properties -------------------------------------------------------------
    @property
    def num_transitions(self) -> int:
        return int(self.states.shape[0])

    @property
    def state_dim(self) -> int:
        return int(self.states.shape[1])

    @classmethod
    def from_dataset(cls, dataset: Any, seed: Optional[int] = None) -> "TrajectoryWindows":
        states = _dataset_states(dataset)
        return cls(
            states=states,
            traj_ids=getattr(dataset, "trajectory_ids", None),
            terminals=getattr(dataset, "terminals", None),
            timeouts=getattr(dataset, "timeouts", None),
            episode_length=getattr(dataset, "max_episode_steps", None),
            seed=seed,
        )

    # -- sampling ---------------------------------------------------------------------
    def sample_indices(self, num: int) -> np.ndarray:
        if self.num_transitions == 0:
            raise ValueError("cannot sample from an empty dataset")
        return self.rng.integers(0, self.num_transitions, size=int(num)).astype(np.int64)

    def window(self, indices: np.ndarray, length: int) -> np.ndarray:
        """Return ``(len(indices), length, state_dim)`` windows (clamped per trajectory)."""
        indices = np.atleast_1d(np.asarray(indices, dtype=np.int64))
        length = max(1, int(length))
        out = np.empty((indices.shape[0], length, self.state_dim), dtype=self.states.dtype)
        for row, idx in enumerate(indices):
            tid = int(self.traj_ids[idx])
            end = self._ends.get(tid, self.num_transitions)
            pos = np.clip(np.arange(idx, idx + length), idx, end - 1)
            out[row] = self.states[pos]
        return out

    def future_state(self, indices: np.ndarray, k: int = 1) -> np.ndarray:
        indices = np.atleast_1d(np.asarray(indices, dtype=np.int64))
        out = np.empty((indices.shape[0], self.state_dim), dtype=self.states.dtype)
        for row, idx in enumerate(indices):
            tid = int(self.traj_ids[idx])
            end = self._ends.get(tid, self.num_transitions)
            pos = min(int(idx) + int(k), end - 1)
            out[row] = self.states[pos]
        return out

    def trajectory_ids(self) -> np.ndarray:
        return np.unique(self.traj_ids)

    def sample_trajectories(self, num: int, length: int) -> np.ndarray:
        """Sample ``num`` contiguous trajectory segments of ``length`` states."""
        indices = self.sample_indices(num)
        return self.window(indices, length)


def sample_trajectory_batch(
    dataset: Any,
    batch_size: int,
    length: int = OPAL_TRAJECTORY_LENGTH,
    seed: Optional[int] = None,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Sample ``(batch_size, length, state_dim)`` trajectory segments as a tensor.

    Prefers a native ``dataset.sample_trajectories(num)`` implementation when it
    returns a suitable 3-D array; otherwise falls back to
    :class:`TrajectoryWindows`.
    """
    arr: Optional[np.ndarray] = None
    sampler = getattr(dataset, "sample_trajectories", None)
    if callable(sampler):
        try:
            arr = np.asarray(sampler(int(batch_size)))
        except Exception:  # pragma: no cover - defensive
            arr = None
        if arr is not None and (arr.ndim != 3 or arr.shape[0] != int(batch_size)):
            arr = None
    if arr is None:
        windows = TrajectoryWindows.from_dataset(dataset, seed=seed)
        arr = windows.sample_trajectories(int(batch_size), int(length))
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:  # (T, state_dim) -- a single trajectory
        arr = arr[None]
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


def _dataset_states(dataset: Any) -> np.ndarray:
    """Extract a flat ``(N, state_dim)`` state array from a dataset-like object."""
    for attr in ("observations", "obs", "states"):
        value = getattr(dataset, attr, None)
        if value is not None:
            return np.asarray(value, dtype=np.float32)
    sampler = getattr(dataset, "sample_states", None)
    if callable(sampler):
        return np.asarray(sampler(1), dtype=np.float32)
    raise ValueError("dataset does not expose observations/states or sample_states()")


# --------------------------------------------------------------------------------------
# Networks
# --------------------------------------------------------------------------------------


def _activation(name: str) -> nn.Module:
    try:
        return make_activation(name)
    except Exception:  # pragma: no cover - defensive
        return nn.GELU() if name == "gelu" else nn.ReLU()


class SkillEncoder(nn.Module):
    """Trajectory -> skill encoder ``q_theta(z | s_1, ..., s_T)``.

    Uses exactly the same transformer architecture as the FRE encoder (addendum:
    "For the OPAL encoder, the same transformer architecture is used as in FRE"):
    a learned state embedding, ``num_blocks`` transformer blocks with
    ``num_heads`` attention heads, width 128, MLP 128 -> 256 -> 128, no
    positional encodings and no causal masking (so the encoder is
    permutation-invariant over the trajectory tokens), followed by mean pooling
    and two linear heads producing the mean / log-std of a diagonal Gaussian.
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int = 128,
        state_embedding_dim: int = 128,
        transformer_width: int = 128,
        num_blocks: int = 4,
        num_heads: int = 4,
        transformer_mlp_dim: int = 256,
        activation: str = "gelu",
        dropout: float = 0.0,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.latent_dim = int(latent_dim)
        self.state_embedding_dim = int(state_embedding_dim)
        if self.state_embedding_dim != int(transformer_width):
            raise ValueError(
                "state_embedding_dim must equal transformer_width for the OPAL "
                f"encoder (got {self.state_embedding_dim} vs {transformer_width})."
            )
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        self.state_embedding = nn.Linear(self.state_dim, self.state_embedding_dim)
        self.transformer = TransformerEncoder(
            width=int(transformer_width),
            num_blocks=int(num_blocks),
            num_heads=int(num_heads),
            mlp_dim=int(transformer_mlp_dim),
            activation=activation,
            dropout=float(dropout),
            use_positional_encoding=False,  # permutation-invariant over the trajectory
            use_causal_mask=False,
        )
        self.mean_head = nn.Linear(int(transformer_width), self.latent_dim)
        self.log_std_head = nn.Linear(int(transformer_width), self.latent_dim)
        self.apply(self._init_weights)
        nn.init.normal_(self.log_std_head.weight, std=0.01)
        nn.init.zeros_(self.log_std_head.bias)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, states: torch.Tensor) -> torch.distributions.Normal:
        if states.dim() == 2:
            states = states.unsqueeze(0)
        tokens = self.state_embedding(states)
        hidden = self.transformer(tokens)
        mean = self.mean_head(hidden)
        log_std = torch.clamp(self.log_std_head(hidden), self.log_std_min, self.log_std_max)
        std = torch.exp(log_std) + 1e-6
        return torch.distributions.Normal(mean, std)

    def encode(self, states: torch.Tensor, sample: bool = False) -> torch.Tensor:
        dist = self.forward(states)
        return dist.rsample() if sample else dist.mean

    def log_prob(self, states: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """``log q_theta(z | states)`` -- the intrinsic skill reward."""
        dist = self.forward(states)
        z = z.reshape(1, -1) if z.dim() == 1 else z
        if z.shape[0] != dist.mean.shape[0]:
            z = z.expand(dist.mean.shape[0], -1)
        return dist.log_prob(z).sum(-1)


class SkillDecoder(nn.Module):
    """``p_theta(s_{t+k} | s_t, z)`` -- MLP predicting future states from a skill."""

    def __init__(
        self,
        state_dim: int,
        latent_dim: int = 128,
        hidden_layers: Sequence[int] = (512, 512, 512),
        activation: str = "gelu",
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.latent_dim = int(latent_dim)
        self.output_dim = int(output_dim if output_dim is not None else state_dim)
        self.net = MLP(
            input_dim=self.state_dim + self.latent_dim,
            hidden_layers=tuple(hidden_layers),
            output_dim=self.output_dim,
            activation=activation,
        )

    def forward(self, states: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if states.dim() == 3:  # (B, T, state_dim) -> predict for each state
            batch, length, dim = states.shape
            z_exp = z.reshape(z.shape[0], 1, -1).expand(batch, length, z.shape[-1])
            x = torch.cat([states, z_exp], dim=-1)
            out = self.net(x.reshape(batch * length, -1))
            return out.reshape(batch, length, self.output_dim)
        z = z.reshape(-1, z.shape[-1])
        if z.shape[0] != states.shape[0]:
            z = z.expand(states.shape[0], -1)
        return self.net(torch.cat([states, z], dim=-1))


class OPALLoss(torch.utils.data.dataset if False else object):  # pragma: no cover
    """Placeholder (kept out of the public API); see :class:`OPALModel`."""


@dataclass
class OPALTrainingLoss:
    """Container for the OPAL trajectory auto-encoding objective components."""

    loss: torch.Tensor
    reconstruction: torch.Tensor
    kl: torch.Tensor
    weighted_kl: torch.Tensor
    z: torch.Tensor
    pred: torch.Tensor
    target: torch.Tensor

    @property
    def mse(self) -> torch.Tensor:
        return self.reconstruction

    def as_dict(self) -> Dict[str, float]:
        return {
            "loss": float(self.loss.detach().cpu()),
            "reconstruction": float(self.reconstruction.detach().cpu()),
            "kl": float(self.kl.detach().cpu()),
            "weighted_kl": float(self.weighted_kl.detach().cpu()),
        }


class OPALModel(nn.Module):
    """Joint OPAL skill encoder + future-state decoder.

    Objective (trajectory variational auto-encoder with unit-Gaussian skill prior,
    OPAL Eq. 1-2)::

        L = E[ || s_{t+k} - p_theta(s_t, z) ||^2 ] + beta * KL(q_theta(z|tau) || N(0, I))

    Minimised jointly over the encoder and decoder.  No reward information is
    used anywhere (addendum: "No manually designed rewards are used in OPAL").
    """

    def __init__(
        self,
        encoder: SkillEncoder,
        decoder: SkillDecoder,
        beta_kl: float = 0.01,
        prediction_horizon: int = OPAL_PREDICTION_HORIZON,
        use_rsample: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.beta_kl = float(beta_kl)
        self.prediction_horizon = int(prediction_horizon)
        self.use_rsample = bool(use_rsample)

    # -- core ------------------------------------------------------------------------
    def loss(
        self,
        trajectories: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        beta_kl: Optional[float] = None,
    ) -> OPALTrainingLoss:
        """Trajectory auto-encoding loss.

        ``trajectories`` : ``(B, T, state_dim)`` states sampled from the offline
        dataset.  ``targets`` : ``(B, T, state_dim)`` future states
        ``s_{t+k}`` (defaults to the trajectories shifted by the prediction
        horizon, clamped to the last available state).
        """
        if trajectories.dim() == 2:
            trajectories = trajectories.unsqueeze(0)
        beta = float(self.beta_kl if beta_kl is None else beta_kl)

        posterior = self.encoder.forward(trajectories)
        z = posterior.rsample() if self.use_rsample else posterior.mean

        if targets is None:
            k = max(1, self.prediction_horizon)
            if trajectories.shape[1] > k:
                targets = trajectories[:, k:, :]
                inputs = trajectories[:, :-k, :]
            else:
                targets = trajectories[:, -1:, :]
                inputs = trajectories[:, :1, :]
        else:
            if targets.dim() == 2:
                targets = targets.unsqueeze(1)
            inputs = trajectories
            if inputs.shape[0] != targets.shape[0]:
                inputs = inputs[: targets.shape[0]]
            if inputs.shape[1] != targets.shape[1]:
                T = min(inputs.shape[1], targets.shape[1])
                inputs, targets = inputs[:, :T], targets[:, :T]

        pred = self.decoder(inputs, z[:, :1, :] if z.dim() == 3 else z)
        # broadcast z across the time axis if the decoder returned per-timestep outputs
        if pred.shape[1] != targets.shape[1]:
            pred = pred[:, : targets.shape[1]]
        reconstruction = F.mse_loss(pred, targets)
        kl = diagonal_gaussian_kl(posterior.loc, posterior.scale)
        total = reconstruction + beta * kl
        return OPALTrainingLoss(
            loss=total,
            reconstruction=reconstruction,
            kl=kl,
            weighted_kl=beta * kl,
            z=z,
            pred=pred,
            target=targets,
        )

    def forward(self, trajectories: torch.Tensor, targets: Optional[torch.Tensor] = None) -> OPALTrainingLoss:
        return self.loss(trajectories, targets)

    # -- convenience ------------------------------------------------------------------
    def encode(self, trajectories: torch.Tensor, sample: bool = False) -> torch.Tensor:
        with torch.no_grad():
            return self.encoder.encode(trajectories, sample=sample)

    def log_prob(self, states: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.encoder.log_prob(states, z)

    def trainable_parameters(self):
        for p in self.parameters():
            if p.requires_grad:
                yield p

    @classmethod
    def from_config(cls, config: Any, state_dim: int, **overrides: Any) -> "OPALModel":
        encoder_kwargs = dict(overrides.pop("encoder_kwargs", {}) or {})
        decoder_kwargs = dict(overrides.pop("decoder_kwargs", {}) or {})
        latent_dim = int(getattr(config, "latent_dim", 128))
        encoder = SkillEncoder(
            state_dim=state_dim,
            latent_dim=latent_dim,
            state_embedding_dim=int(encoder_kwargs.pop("state_embedding_dim", latent_dim)),
            transformer_width=int(encoder_kwargs.pop("transformer_width", 128)),
            num_blocks=int(encoder_kwargs.pop("num_blocks", getattr(config, "num_encoder_blocks", 4))),
            num_heads=int(encoder_kwargs.pop("num_heads", getattr(config, "num_attention_heads", 4))),
            transformer_mlp_dim=int(encoder_kwargs.pop("transformer_mlp_dim", getattr(config, "transformer_mlp_dim", 256))),
            activation=encoder_kwargs.pop("activation", getattr(config, "decoder_activation", "gelu")),
            dropout=float(encoder_kwargs.pop("dropout", 0.0)),
            **encoder_kwargs,
        )
        decoder = SkillDecoder(
            state_dim=state_dim,
            latent_dim=latent_dim,
            hidden_layers=tuple(decoder_kwargs.pop("hidden_layers", getattr(config, "decoder_layers", (512, 512, 512)))),
            activation=decoder_kwargs.pop("activation", getattr(config, "decoder_activation", "gelu")),
            **decoder_kwargs,
        )
        return cls(
            encoder=encoder,
            decoder=decoder,
            beta_kl=float(overrides.pop("beta_kl", getattr(config, "beta_kl", 0.01))),
            prediction_horizon=int(overrides.pop("prediction_horizon", OPAL_PREDICTION_HORIZON)),
            **overrides,
        )

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return f"latent_dim={self.encoder.latent_dim}, beta_kl={self.beta_kl}, horizon={self.prediction_horizon}"


# --------------------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------------------


class OPALAgent:
    """OPAL agent: skill pretraining (trajectory VAE) + intrinsic-reward IQL.

    Parameters mirror the FRE configuration (Table 3) so that the OPAL column of
    Table 1 is trained with the same budget and RL hyper-parameters as FRE::

        skill_steps  : phase-1 (auto-encode trajectories) -- defaults to the
                       FRE encoder budget (150k, or 1M for ExORL/Kitchen).
        policy_steps : phase-2 z-conditioned IQL on the intrinsic reward.

    The intrinsic reward is the state-skill mutual information
    ``r(s, z) = log q_theta(z | s) - log u(z)`` (estimated with the frozen skill
    encoder), and the policy/critic are the shared ``z``-conditioned IQL networks.
    """

    def __init__(
        self,
        model: OPALModel,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 128,
        hidden_layers: Sequence[int] = (512, 512, 512),
        activation: str = "relu",
        expectile: float = 0.8,
        temperature: float = 3.0,
        discount: float = 0.88,
        target_update_rate: float = 0.001,
        learning_rate: float = 1e-4,
        grad_clip_norm: float = 10.0,
        beta_kl: Optional[float] = None,
        prediction_horizon: int = OPAL_PREDICTION_HORIZON,
        trajectory_length: int = OPAL_TRAJECTORY_LENGTH,
        iql: Optional[IQL] = None,
        device: Union[str, torch.device] = "cpu",
        seed: int = 0,
        use_intrinsic_reward: bool = True,
        intrinsic_reward_kind: str = "log_prob_state",
    ):
        self.device = torch.device(device)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.beta_kl = float(beta_kl if beta_kl is not None else getattr(model, "beta_kl", 0.01))
        self.prediction_horizon = int(prediction_horizon)
        self.trajectory_length = int(trajectory_length)
        self.learning_rate = float(learning_rate)
        self.grad_clip_norm = float(grad_clip_norm)
        self.seed = int(seed)
        self.use_intrinsic_reward = bool(use_intrinsic_reward)
        self.intrinsic_reward_kind = str(intrinsic_reward_kind)

        self.model = model.to(self.device)
        self.iql = iql if iql is not None else make_iql(
            None,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            device=self.device,
            latent_dim=self.latent_dim,
            hidden_layers=tuple(hidden_layers),
            activation=activation,
            expectile=expectile,
            temperature=temperature,
            discount=discount,
            target_update_rate=target_update_rate,
            learning_rate=self.learning_rate,
        )
        if iql is None:
            self.iql.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.trainable_parameters(), lr=self.learning_rate)
        self._trajectory_index: Optional[TrajectoryWindows] = None
        self._dataset: Any = None

    # -- setup -------------------------------------------------------------------------
    def attach_dataset(self, dataset: Any) -> None:
        self._dataset = dataset
        self._trajectory_index = TrajectoryWindows.from_dataset(dataset, seed=self.seed)

    @property
    def trajectory_index(self) -> TrajectoryWindows:
        if self._trajectory_index is None:
            raise RuntimeError("no dataset attached; call attach_dataset() first")
        return self._trajectory_index

    def to(self, device: Union[str, torch.device]) -> "OPALAgent":
        self.device = torch.device(device)
        self.model.to(self.device)
        self.iql.to(self.device)
        return self

    def train(self) -> "OPALAgent":
        self.model.train()
        self.iql.train()
        return self

    def eval(self) -> "OPALAgent":
        self.model.eval()
        self.iql.eval()
        return self

    def parameters(self):
        return self.model.parameters()

    # -- phase 1: trajectory auto-encoding ("OPAL encoder") -----------------------------
    def skill_loss(self, trajectories: torch.Tensor) -> OPALTrainingLoss:
        return self.model.loss(trajectories, beta_kl=self.beta_kl)

    def train_skill_step(self, trajectories: torch.Tensor) -> Dict[str, float]:
        self.model.train()
        loss = self.model.loss(trajectories, beta_kl=self.beta_kl)
        self.optimizer.zero_grad(set_to_none=True)
        loss.loss.backward()
        if self.grad_clip_norm:
            nn.utils.clip_grad_norm_(list(self.model.trainable_parameters()), self.grad_clip_norm)
        self.optimizer.step()
        return loss.as_dict()

    def train_skill(
        self,
        dataset: Optional[Any] = None,
        steps: int = 150_000,
        batch_size: int = 512,
        log_interval: int = 1000,
        logger: Any = None,
        prefix: str = "opal_skill",
        close_logger: bool = False,
    ) -> List[Dict[str, float]]:
        """Phase 1: auto-encode trajectories to learn the skill encoder/decoder."""
        if dataset is not None:
            self.attach_dataset(dataset)
        if self._dataset is None:
            raise ValueError("train_skill requires a dataset (pass one or call attach_dataset)")

        history: List[Dict[str, float]] = []
        rng = np.random.default_rng(self.seed + 1234)
        for step in range(int(steps)):
            try:
                traj = sample_trajectory_batch(
                    self._dataset, batch_size, length=self.trajectory_length, seed=int(rng.integers(0, 2**31)), device=self.device
                )
            except Exception:
                # fall back to windows over randomly sampled states
                idx = self.trajectory_index.sample_indices(batch_size)
                traj = torch.as_tensor(
                    self.trajectory_index.window(idx, self.trajectory_length),
                    dtype=torch.float32,
                    device=self.device,
                )
            metrics = self.train_skill_step(traj)
            if log_interval and (step % log_interval == 0 or step == int(steps) - 1):
                metrics = dict(metrics, step=step)
                history.append(metrics)
                if logger is not None:
                    _log(logger, metrics, prefix)
        if close_logger and logger is not None:
            _safe_close(logger)
        return history

    # -- phase 2: intrinsic-reward IQL ---------------------------------------------------
    @torch.no_grad()
    def intrinsic_reward(
        self,
        observations: torch.Tensor,
        z: torch.Tensor,
        next_observations: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``r(s, z) = log q_theta(z | s) - log u(z)`` (state-skill mutual information).

        The skill encoder is applied to a single-state context here; when
        ``next_observations`` is provided and
        ``intrinsic_reward_kind == "log_prob_future"`` the reward uses the
        successor state instead, which is closer to the original
        state-conditioned MI objective ``I(s_{t+k}; z)``.
        """
        if not self.use_intrinsic_reward:
            return torch.zeros(observations.shape[0], device=observations.device)
        context = observations
        if next_observations is not None and self.intrinsic_reward_kind == "log_prob_future":
            context = next_observations
        # (1, N, state_dim) -> encoder returns (1, latent_dim)
        log_q = self.model.encoder.log_prob(context.unsqueeze(0), z)
        prior = torch.distributions.Normal(
            torch.zeros(self.latent_dim, device=z.device), torch.ones(self.latent_dim, device=z.device)
        )
        log_u = prior.log_prob(z.reshape(-1)).sum(-1)
        return log_q - log_u

    def _transition_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        sampler = getattr(self._dataset, "sample_transitions", None)
        if callable(sampler):
            batch = sampler(int(batch_size))
        elif callable(getattr(self._dataset, "sample_indices", None)):
            idx = self._dataset.sample_indices(int(batch_size))
            obs = _dataset_states(self._dataset)[idx]
            batch = {"observations": obs, "next_observations": obs, "actions": None, "terminals": None}
        else:  # pragma: no cover - defensive
            raise ValueError("dataset does not expose sample_transitions or sample_indices")
        return _to_tensor_batch(batch, self.device)

    def update_policy(self, batch: Dict[str, torch.Tensor], batch_size: Optional[int] = None, target_update: bool = True) -> Dict[str, float]:
        """One IQL update on the intrinsic skill reward."""
        obs = batch["observations"]
        z = torch.randn(obs.shape[0], self.latent_dim, device=self.device)
        rewards = self.intrinsic_reward(obs, z, batch.get("next_observations"))
        info = self.iql.update(
            batch,
            rewards=rewards,
            z=z,
            next_z=z,
            target_update=target_update,
        )
        return info if isinstance(info, dict) else {"loss": float(info)}

    def train_policy(
        self,
        dataset: Optional[Any] = None,
        steps: int = 850_000,
        batch_size: int = 512,
        log_interval: int = 1000,
        logger: Any = None,
        prefix: str = "opal_policy",
        target_update_every: int = 1,
    ) -> List[Dict[str, float]]:
        """Phase 2: z-conditioned IQL with the intrinsic skill reward."""
        if dataset is not None:
            self.attach_dataset(dataset)
        if self._dataset is None:
            raise ValueError("train_policy requires a dataset (pass one or call attach_dataset)")

        # freeze the skill encoder: the intrinsic reward must stay stationary
        self.freeze_encoder()

        history: List[Dict[str, float]] = []
        for step in range(int(steps)):
            batch = self._transition_batch(batch_size)
            info = self.update_policy(batch, target_update=bool(step % max(1, target_update_every) == 0))
            if log_interval and (step % log_interval == 0 or step == int(steps) - 1):
                metrics = dict(info, step=step)
                history.append(metrics)
                if logger is not None:
                    _log(logger, metrics, prefix)
        return history

    def freeze_encoder(self) -> None:
        for p in self.model.encoder.parameters():
            p.requires_grad_(False)
        self.model.encoder.eval()

    def unfreeze_encoder(self) -> None:
        for p in self.model.encoder.parameters():
            p.requires_grad_(True)

    # -- inference ----------------------------------------------------------------------
    def encode_skill(self, trajectories: torch.Tensor, sample: bool = False) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            if isinstance(trajectories, np.ndarray):
                trajectories = torch.as_tensor(trajectories, dtype=torch.float32)
            trajectories = trajectories.to(self.device).float()
            return self.model.encode(trajectories, sample=sample)

    def sample_skills(self, num_skills: int = OPAL_NUM_SKILLS, seed: Optional[int] = None) -> torch.Tensor:
        """Sample skills from the unit Gaussian prior (privileged evaluation)."""
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
            z = torch.randn(int(num_skills), self.latent_dim, generator=generator).to(self.device)
        else:
            z = torch.randn(int(num_skills), self.latent_dim, device=self.device)
        return z

    def select_action(
        self,
        obs: np.ndarray,
        skill: Union[np.ndarray, torch.Tensor],
        deterministic: bool = True,
        clip: Optional[bool] = True,
    ) -> np.ndarray:
        z = _as_tensor(skill, self.device)
        action = self.iql.select_action(obs, z, deterministic=deterministic, clip=bool(clip))
        return np.asarray(action, dtype=np.float32).reshape(-1)

    def value_of(self, obs: np.ndarray, skill: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        return self.iql.value_of(obs, _as_tensor(skill, self.device))

    # -- persistence ---------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "iql": self.iql.state_dict() if hasattr(self.iql, "state_dict") else None,
            "optimizer": self.optimizer.state_dict(),
            "latent_dim": self.latent_dim,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> "OPALAgent":
        self.model.load_state_dict(state["model"])
        if state.get("iql") is not None:
            self.iql.load_state_dict(state["iql"])
        if state.get("optimizer") is not None:
            try:
                self.optimizer.load_state_dict(state["optimizer"])
            except Exception:  # pragma: no cover - defensive
                pass
        return self

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)
        return path

    def load(self, path: str, map_location: Union[str, torch.device] = "cpu") -> "OPALAgent":
        state = torch.load(path, map_location=map_location)
        return self.load_state_dict(state)

    # -- factories ------------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        config: Any,
        obs_dim: int,
        action_dim: int,
        device: Union[str, torch.device, None] = None,
        **overrides: Any,
    ) -> "OPALAgent":
        device = device if device is not None else getattr(config, "device", "cpu")
        latent_dim = int(overrides.pop("latent_dim", getattr(config, "latent_dim", 128)))
        model = overrides.pop("model", None)
        if model is None:
            model = OPALModel.from_config(config, state_dim=obs_dim, **overrides.pop("model_kwargs", {}))
        iql = overrides.pop("iql", None)
        if iql is None:
            iql = make_iql(config, obs_dim=obs_dim, action_dim=action_dim, device=device, latent_dim=latent_dim)
        return cls(
            model=model,
            obs_dim=obs_dim,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_layers=tuple(overrides.pop("hidden_layers", getattr(config, "rl_hidden_layers", (512, 512, 512)))),
            activation=overrides.pop("activation", getattr(config, "rl_activation", "relu")),
            expectile=float(overrides.pop("expectile", getattr(config, "iql_expectile", 0.8))),
            temperature=float(overrides.pop("temperature", getattr(config, "iql_temperature", 3.0))),
            discount=float(overrides.pop("discount", getattr(config, "discount", 0.88))),
            target_update_rate=float(overrides.pop("target_update_rate", getattr(config, "target_update_rate", 0.001))),
            learning_rate=float(overrides.pop("learning_rate", getattr(config, "learning_rate", 1e-4))),
            grad_clip_norm=float(overrides.pop("grad_clip_norm", getattr(config, "grad_clip_norm", 10.0))),
            beta_kl=float(overrides.pop("beta_kl", getattr(config, "beta_kl", 0.01))),
            trajectory_length=int(overrides.pop("trajectory_length", OPAL_TRAJECTORY_LENGTH)),
            prediction_horizon=int(overrides.pop("prediction_horizon", OPAL_PREDICTION_HORIZON)),
            iql=iql,
            device=device,
            seed=int(overrides.pop("seed", getattr(config, "seed", 0))),
            **overrides,
        )


# --------------------------------------------------------------------------------------
# Convenience factories / training driver
# --------------------------------------------------------------------------------------


def make_opal(
    config: Any,
    obs_dim: int,
    action_dim: int,
    device: Union[str, torch.device, None] = None,
    **overrides: Any,
) -> OPALAgent:
    """Build an :class:`OPALAgent` from a FRE ``Config``-like object."""
    return OPALAgent.from_config(config, obs_dim=obs_dim, action_dim=action_dim, device=device, **overrides)


def train_opal(
    config: Any,
    dataset: Any,
    obs_dim: Optional[int] = None,
    action_dim: Optional[int] = None,
    skill_steps: Optional[int] = None,
    policy_steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    log_interval: Optional[int] = None,
    logger: Any = None,
    agent: Optional[OPALAgent] = None,
    device: Union[str, torch.device, None] = None,
    **overrides: Any,
) -> Tuple[OPALAgent, Dict[str, List[Dict[str, float]]]]:
    """Full OPAL training: skill pretraining then intrinsic-reward IQL.

    Step budgets default to the FRE schedule (``config.encoder_steps()`` /
    ``config.policy_steps()``) so OPAL-10 is compared at equal training cost.
    """
    if obs_dim is None:
        obs_dim = int(getattr(dataset, "obs_dim", None) or _dataset_states(dataset).shape[1])
    if action_dim is None:
        action_dim = int(getattr(dataset, "action_dim", None) or _infer_action_dim(dataset))
    device = device if device is not None else getattr(config, "device", "cpu")

    if agent is None:
        agent = make_opal(config, obs_dim=obs_dim, action_dim=action_dim, device=device, **overrides)
    agent.attach_dataset(dataset)

    if skill_steps is None:
        try:
            skill_steps = int(config.encoder_steps())
        except Exception:  # pragma: no cover - defensive
            skill_steps = 150_000
    if policy_steps is None:
        try:
            policy_steps = int(config.policy_steps())
        except Exception:  # pragma: no cover - defensive
            policy_steps = 850_000
    if batch_size is None:
        batch_size = int(getattr(config, "batch_size", 512))
    if log_interval is None:
        log_interval = int(getattr(config, "log_interval", 1000))

    skill_history = agent.train_skill(dataset, steps=skill_steps, batch_size=batch_size, log_interval=log_interval, logger=logger)
    policy_history = agent.train_policy(dataset, steps=policy_steps, batch_size=batch_size, log_interval=log_interval, logger=logger)
    return agent, {"skill": skill_history, "policy": policy_history}


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------


def make_opal_policy_fn(
    agent: OPALAgent,
    skill: Union[np.ndarray, torch.Tensor],
    deterministic: bool = True,
    clip: bool = True,
) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a skill-conditioned OPAL policy into an ``act_fn(obs) -> action`` callable."""

    def act_fn(obs: np.ndarray) -> np.ndarray:
        return agent.select_action(obs, skill, deterministic=deterministic, clip=clip)

    return act_fn


def make_skill_policy_fns(
    agent: OPALAgent,
    num_skills: int = OPAL_NUM_SKILLS,
    seed: int = 0,
    deterministic: bool = True,
    clip: bool = True,
) -> Tuple[torch.Tensor, List[Callable[[np.ndarray], np.ndarray]]]:
    """Sample ``num_skills`` unit-Gaussian skills and return one policy per skill."""
    skills = agent.sample_skills(num_skills, seed=seed)
    return skills, [make_opal_policy_fn(agent, skills[i], deterministic=deterministic, clip=clip) for i in range(skills.shape[0])]


def evaluate_opal_privileged(
    agent: OPALAgent,
    evaluate_fn: Callable[[Callable[[np.ndarray], np.ndarray], int], Union[float, Dict[str, Any]]],
    num_skills: int = OPAL_NUM_SKILLS,
    seed: int = 0,
    deterministic: bool = True,
    clip: bool = True,
    base_seed: Optional[int] = None,
    return_details: bool = False,
) -> Union[float, Dict[str, Any]]:
    """Privileged OPAL evaluation (addendum): sample ``num_skills`` skills from
    ``N(0, I)``, evaluate each for the entire episode, and take the *best*
    rollout (OPAL's task policy is not used).

    Parameters
    ----------
    agent : trained :class:`OPALAgent`.
    evaluate_fn : callable ``(act_fn, seed) -> score | dict`` that runs the full
        evaluation protocol (e.g. 20 episodes on one task suite) for a single
        policy and returns the normalized 0-100 score (or a dict containing
        ``"score"``/``"mean"``).
    num_skills : number of Gaussian skills sampled (10 in the paper -> "OPAL-10").
    seed : RNG seed controlling both the skill sample and per-skill rollouts.
    base_seed : explicit seed offset for the evaluation function.

    Returns
    -------
    ``float`` (best score) or, when ``return_details=True``, a dict with
    ``best_score``, ``best_skill_index``, ``skill_scores`` and ``skills``.
    """
    skills, act_fns = make_skill_policy_fns(agent, num_skills=num_skills, seed=seed, deterministic=deterministic, clip=clip)
    scores: List[float] = []
    seed_offset = int(base_seed) if base_seed is not None else int(seed) * 1000
    for i, act_fn in enumerate(act_fns):
        result = evaluate_fn(act_fn, seed_offset + i)
        scores.append(_score_of(result))
    scores_arr = np.asarray(scores, dtype=np.float64)
    best = int(np.argmax(scores_arr)) if scores_arr.size else -1
    best_score = float(scores_arr[best]) if best >= 0 else 0.0
    if not return_details:
        return best_score
    return {
        "best_score": best_score,
        "best_skill_index": best,
        "skill_scores": scores,
        "skill_mean": float(np.mean(scores_arr)) if scores_arr.size else 0.0,
        "skill_std": float(np.std(scores_arr)) if scores_arr.size else 0.0,
        "skills": skills.detach().cpu().numpy(),
        "num_skills": int(num_skills),
    }


def evaluate_opal_suite(
    agent: OPALAgent,
    suite_evaluate_fn: Callable[[Callable[[np.ndarray], np.ndarray], int], Dict[str, Any]],
    num_skills: int = OPAL_NUM_SKILLS,
    seed: int = 0,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Apply the privileged protocol to a suite-level evaluation function.

    ``suite_evaluate_fn(act_fn, seed)`` is expected to run a whole task suite for a
    single policy (as :func:`fre.envs.antmaze_eval.evaluate_antmaze_suite` does)
    and return ``{task_name: {"score": ...}}``.  Each of the ``num_skills`` skills
    is evaluated over the full suite and the *best* skill per task is retained.
    """
    skills, act_fns = make_skill_policy_fns(agent, num_skills=num_skills, seed=seed)
    per_skill: List[Dict[str, Any]] = []
    for i, act_fn in enumerate(act_fns):
        per_skill.append(suite_evaluate_fn(act_fn, int(seed) * 1000 + i))

    task_names: List[str] = []
    for result in per_skill:
        for key in result:
            if key not in task_names:
                task_names.append(key)

    out: Dict[str, Any] = {}
    for name in task_names:
        scores = [float(_score_of(r.get(name, 0.0))) for r in per_skill]
        best = int(np.argmax(scores)) if scores else -1
        out[name] = {
            "score": float(scores[best]) if best >= 0 else 0.0,
            "best_skill_index": best,
            "skill_scores": scores,
            "num_skills": int(num_skills),
        }
    return out


# --------------------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------------------


def _score_of(result: Union[float, Dict[str, Any], np.ndarray, torch.Tensor]) -> float:
    if isinstance(result, (int, float, np.floating, np.integer)):
        return float(result)
    if isinstance(result, dict):
        for key in ("score", "mean", "normalized_return", "total_return"):
            if key in result:
                return float(result[key])
        return 0.0
    arr = np.asarray(result, dtype=np.float64)
    return float(arr.mean()) if arr.size else 0.0


def _as_tensor(value: Union[np.ndarray, torch.Tensor, Sequence[float]], device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device).float().reshape(1, -1)
    arr = np.asarray(value, dtype=np.float32).reshape(1, -1)
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


def _to_tensor_batch(batch: Any, device: torch.device) -> Dict[str, torch.Tensor]:
    """Normalise a replay-buffer batch (dict or tuple) into a tensor dict."""
    if isinstance(batch, dict):
        out: Dict[str, torch.Tensor] = {}
        for key, value in batch.items():
            if value is None:
                out[key] = None
                continue
            out[key] = _to_tensor(value, device)
        return out
    if isinstance(batch, (tuple, list)):
        names = ["observations", "actions", "rewards", "next_observations", "terminals"]
        return {name: _to_tensor(value, device) for name, value in zip(names, batch) if value is not None}
    raise TypeError(f"unsupported batch type: {type(batch)}")


def _to_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.to(device)
        return tensor.float() if tensor.dtype in (torch.float64, torch.float16) else tensor
    arr = np.asarray(value)
    if arr.dtype == np.bool_:
        return torch.as_tensor(arr, dtype=torch.float32, device=device)
    if np.issubdtype(arr.dtype, np.integer):
        return torch.as_tensor(arr, dtype=torch.float32, device=device)
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


def _infer_action_dim(dataset: Any) -> int:
    for attr in ("action_dim",):
        value = getattr(dataset, attr, None)
        if value:
            return int(value)
    actions = getattr(dataset, "actions", None)
    if actions is not None:
        return int(np.asarray(actions).reshape(len(actions), -1).shape[1])
    raise ValueError("cannot infer action_dim from dataset")


def _log(logger: Any, metrics: Dict[str, Any], prefix: str) -> None:
    """Best-effort logging compatible with :class:`fre.utils.logging.MetricLogger`."""
    payload = {f"{prefix}/{k}": v for k, v in metrics.items()}
    for method_name in ("log_metrics", "log"):
        method = getattr(logger, method_name, None)
        if callable(method):
            try:
                method(payload)
                return
            except TypeError:
                try:
                    method(**payload)
                    return
                except Exception:  # pragma: no cover - defensive
                    return
    if callable(logger):
        try:
            logger(payload)
        except Exception:  # pragma: no cover - defensive
            pass


def _safe_close(logger: Any) -> None:
    close = getattr(logger, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # pragma: no cover - defensive
            pass
