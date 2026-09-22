"""Successor Features (SF) baseline for the FRE paper (Sec. 5.2).

Paper specification (Section 5.2, Table 1):

    "Successor Features (SF) (Barreto et al., 2017; Borsa et al., 2018), which
     utilize a set of pre-trained features to approximate a universal family of
     reward functions and their corresponding policies."

    "FB and SF are based on DDPG-based policies, and are run via the code
     provided from (Touati et al., 2022).  For the SF comparisons, we follow
     prior work (Touati et al., 2022) and learn features using ICM (Pathak et
     al., 2017), which is reported to be the strongest method in the ExORL
     Walker and Cheetah tasks (Touati et al., 2022)."

    "Note that FB/SF rely on linear regression to perform test time adaptation,
     whereas FRE uses a learned encoder network.  To be consistent with prior
     methodology, we give these methods 5120 reward samples during evaluation
     time (in comparison to only 32 for FRE)."

Math of the SF factorisation (Barreto et al. 2017; Borsa et al. 2018):

    reward features:        r(s) = phi(s)^T z                (linear in features)
    successor features:     psi_pi(s, a) = E[ sum_t gamma^t phi(s_t) ]
    value function:         Q_pi(s, a, z) = psi_pi(s, a)^T z
    test-time adaptation:   z* = argmin_z || Phi z - r ||^2   (ridge regression)

where ``Phi`` rows are ``phi(s_i)`` for the 5120 reward-annotated evaluation
samples (for FRE only 32 such samples are used).  As in FB, the task space is
explicitly *linear* - this is exactly the structural restriction FRE removes.

This module offers two execution paths, mirroring ``forward_backward.py``:

1.  A thin subprocess launcher for the official
    ``facebookresearch/controllable_agent`` codebase (``alg="sf"`` plus ICM
    feature learning), which is what the paper used.
2.  A self-contained in-house SF agent (ICM feature extractor + successor
    feature network + DDPG actor) so that evaluation can proceed locally when
    the external repository is not available.

Both paths share the same 5120-sample linear-regression test-time adaptation
from ``forward_backward.py``.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # torch is soft-imported so the launcher-only helpers stay importable
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch-free environments
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Shared helpers with the FB baseline (subprocess launcher, ridge regression,
# 5120-sample evaluation reward sampling).  Imported lazily/tolerantly so this
# module remains usable if the FB file is absent.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - trivial import plumbing
    from fre.baselines.forward_backward import (  # type: ignore
        CONTROLLABLE_AGENT_DIR,
        CONTROLLABLE_AGENT_REPO,
        ControllableAgentUnavailable,
        controllable_agent_available,
        find_controllable_agent,
        build_controllable_agent_command,
        run_controllable_agent,
        sample_eval_reward_samples,
        solve_task_vector,
        FB_EVAL_SAMPLES,
    )

    _SHARED_FB = True
except Exception:  # pragma: no cover
    _SHARED_FB = False
    CONTROLLABLE_AGENT_REPO = "https://github.com/facebookresearch/controllable_agent"
    CONTROLLABLE_AGENT_DIR = os.environ.get(
        "CONTROLLABLE_AGENT_DIR", os.path.join("third_party", "controllable_agent")
    )
    FB_EVAL_SAMPLES = 5120

    class ControllableAgentUnavailable(RuntimeError):  # type: ignore
        """Raised when the official controllable_agent checkout cannot be found."""

    def find_controllable_agent(root: Optional[str] = None) -> Optional[str]:  # type: ignore
        root = root or CONTROLLABLE_AGENT_DIR
        if os.path.isdir(root):
            return root
        return None

    def controllable_agent_available(root: Optional[str] = None) -> bool:  # type: ignore
        return find_controllable_agent(root) is not None

    def build_controllable_agent_command(*args: Any, **kwargs: Any) -> List[str]:  # type: ignore
        raise ControllableAgentUnavailable(
            "fre.baselines.forward_backward is unavailable; cannot build the "
            "controllable_agent command line."
        )

    def run_controllable_agent(*args: Any, **kwargs: Any) -> Dict[str, Any]:  # type: ignore
        raise ControllableAgentUnavailable(
            "fre.baselines.forward_backward is unavailable; cannot launch the "
            "controllable_agent baseline."
        )

    def sample_eval_reward_samples(  # type: ignore
        task: Any,
        dataset: Any = None,
        num_samples: int = 5120,
        rng: Optional[np.random.Generator] = None,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback: sample states from the dataset and evaluate the task reward."""
        rng = rng or np.random.default_rng(0)
        states = None
        if dataset is not None:
            for attr in ("states", "observations"):
                candidate = getattr(dataset, attr, None)
                if candidate is not None:
                    states = np.asarray(candidate)
                    break
            if states is None and hasattr(dataset, "sample_states"):
                states = np.asarray(dataset.sample_states(num_samples))
        if states is None:
            raise ValueError("Cannot sample evaluation reward samples without a dataset.")
        idx = rng.integers(0, len(states), size=num_samples)
        sampled = states[idx]
        rewards = np.asarray(
            [float(task.reward_from_state(s)) for s in sampled], dtype=np.float32
        )
        return sampled, rewards

    def solve_task_vector(  # type: ignore
        agent: Any,
        states: np.ndarray,
        rewards: np.ndarray,
        ridge: float = 1e-3,
        normalize_rewards: bool = True,
        features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fallback ridge regression z* = (Phi^T Phi + lambda I)^-1 Phi^T r."""
        if features is None:
            features = np.asarray(agent.reward_features(states), dtype=np.float64)
        else:
            features = np.asarray(features, dtype=np.float64)
        targets = np.asarray(rewards, dtype=np.float64).reshape(-1)
        if normalize_rewards and targets.std() > 1e-8:
            targets = (targets - targets.mean()) / (targets.std() + 1e-8)
        feat_mean = features.mean(axis=0, keepdims=True)
        feat_std = features.std(axis=0, keepdims=True) + 1e-8
        normed = (features - feat_mean) / feat_std
        dim = normed.shape[1]
        gram = normed.T @ normed + ridge * np.eye(dim)
        z = np.linalg.solve(gram, normed.T @ targets)
        return z.astype(np.float32)


# Early import of FRE network utilities (requires torch).
if _TORCH_AVAILABLE:  # pragma: no cover - import guard
    from fre.rl.networks import MLP, GaussianPolicy, make_activation  # type: ignore


__all__ = [
    "SF_EVAL_SAMPLES",
    "SF_DATASET",
    "SF_FEATURE_DIM",
    "SF_TABLE1_REFERENCE",
    "SF_DEFAULT_LATENT_DIM",
    "SF_DEFAULT_HIDDEN_LAYERS",
    "SF_DEFAULT_DISCOUNT",
    "SF_DEFAULT_LR",
    "SF_DEFAULT_TARGET_RATE",
    "SF_DEFAULT_BATCH_SIZE",
    "ICM_DEFAULT_HIDDEN_LAYERS",
    "ICM_DEFAULT_FEATURE_DIM",
    "ICMFeatureExtractor",
    "RandomFeatureExtractor",
    "SFModel",
    "SFTrainingStats",
    "SuccessorFeaturesAgent",
    "SFAgent",
    "build_sf_command",
    "run_sf_controllable_agent",
    "sample_eval_reward_samples",
    "solve_task_vector",
    "make_sf_policy_fn",
    "make_successor_features",
    "train_successor_features",
    "evaluate_sf_suite",
    "sf_reference_row",
    "dump_sf_config",
]


# ---------------------------------------------------------------------------
# Constants (Table 1 reference numbers for SF; Sec. 5.2 evaluation protocol)
# ---------------------------------------------------------------------------

#: FB/SF are given 5120 reward samples at evaluation time (FRE only 32).
SF_EVAL_SAMPLES = 5120 if not _SHARED_FB else FB_EVAL_SAMPLES

#: ExORL dataset used for SF/FB per Sec. 5.2 ("ExORL uses RND dataset").
SF_DATASET = "rnd"

#: ICM feature dimensionality (Touati et al., 2022 use 128-d features).
SF_FEATURE_DIM = 128
ICM_DEFAULT_HIDDEN_LAYERS: Tuple[int, ...] = (256, 256)
ICM_DEFAULT_FEATURE_DIM = 128

#: SF hyperparameters (DDPG-based; paper points at Touati et al. 2022 defaults).
SF_DEFAULT_LATENT_DIM = SF_FEATURE_DIM
SF_DEFAULT_HIDDEN_LAYERS: Tuple[int, ...] = (512, 512, 512)
SF_DEFAULT_DISCOUNT = 0.99
SF_DEFAULT_LR = 1e-4
SF_DEFAULT_TARGET_RATE = 0.005
SF_DEFAULT_BATCH_SIZE = 512

#: Table 1 reference scores for the SF baseline (mean +/- std across 5 seeds).
SF_TABLE1_REFERENCE: Dict[str, Tuple[float, float]] = {
    # domain aggregates
    "antmaze-all": (11.8, 12.6),
    "exorl-all": (40.9, 1.9),
    "kitchen": (1.0, 1.0),
    "all": (18.0, 5.0),
    # per-task rows (Table 1)
    "ant-goal-reaching": (10.7, 14.0),
    "ant-directional": (4.5, 3.5),
    "ant-random-simplex": (23.0, 19.5),
    "ant-path-loop": (18.0, 23.0),
    "ant-path-edges": (13.0, 10.0),
    "ant-path-center": (12.0, 17.0),
    "exorl-walker-goals": (44.0, 5.0),
    "exorl-cheetah-goals": (25.0, 4.0),
    "exorl-walker-velocity": (55.0, 6.0),
    "exorl-cheetah-velocity": (40.0, 4.0),
}


# ---------------------------------------------------------------------------
# Feature extractors: ICM (Pathak et al., 2017) and a random-projection
# fallback used when ICM training is skipped (documented default).
# ---------------------------------------------------------------------------


if _TORCH_AVAILABLE:

    class ICMFeatureExtractor(nn.Module):  # type: ignore[misc]
        """Intrinsic Curiosity Module feature extractor (Pathak et al., 2017).

        The forward/inverse dynamics losses shape ``encoder`` so that its output
        ``h(s)`` captures controllable aspects of the state.  Following
        (Touati et al., 2022) the SF baseline uses ``phi(s) = h(s)`` as the
        reward features, i.e. rewards are approximated linearly in ICM features.
        """

        def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            feature_dim: int = ICM_DEFAULT_FEATURE_DIM,
            hidden_layers: Sequence[int] = ICM_DEFAULT_HIDDEN_LAYERS,
            activation: str = "relu",
            learning_rate: float = 1e-4,
            inverse_coef: float = 1.0,
            forward_coef: float = 1.0,
            device: str = "cpu",
        ) -> None:
            super().__init__()
            self.obs_dim = int(obs_dim)
            self.action_dim = int(action_dim)
            self.feature_dim = int(feature_dim)
            self.learning_rate = float(learning_rate)
            self.inverse_coef = float(inverse_coef)
            self.forward_coef = float(forward_coef)
            hidden = tuple(int(h) for h in hidden_layers)

            self.encoder = MLP(
                self.obs_dim,
                hidden_layers=hidden,
                output_dim=self.feature_dim,
                activation=activation,
            )
            self.inverse_model = MLP(
                2 * self.feature_dim,
                hidden_layers=hidden,
                output_dim=self.action_dim,
                activation=activation,
            )
            self.forward_model = MLP(
                self.feature_dim + self.action_dim,
                hidden_layers=hidden,
                output_dim=self.feature_dim,
                activation=activation,
            )
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            self.to(device)
            self.device = device

        # -- features -----------------------------------------------------
        def features(self, obs: Any) -> "torch.Tensor":
            """Return ICM features ``h(s)`` as a torch tensor."""
            obs_t = _as_tensor(obs, device=self.device)
            single = obs_t.dim() == 1
            if single:
                obs_t = obs_t.unsqueeze(0)
            feats = self.encoder(obs_t)
            return feats[0] if single else feats

        @torch.no_grad()
        def numpy_features(self, obs: Any) -> np.ndarray:
            return self.features(obs).detach().cpu().numpy()

        # -- ICM losses ---------------------------------------------------
        def icm_loss(
            self, obs: "torch.Tensor", next_obs: "torch.Tensor", actions: "torch.Tensor"
        ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            h = self.encoder(obs)
            h_next = self.encoder(next_obs)
            pred_actions = self.inverse_model(torch.cat([h, h_next], dim=-1))
            inverse_loss = F.mse_loss(pred_actions, actions)
            pred_next = self.forward_model(torch.cat([h, actions], dim=-1))
            forward_loss = F.mse_loss(pred_next, h_next.detach())
            total = self.inverse_coef * inverse_loss + self.forward_coef * forward_loss
            return total, inverse_loss, forward_loss

        def update(self, obs: Any, next_obs: Any, actions: Any) -> Dict[str, float]:
            obs_t = _as_tensor(obs, device=self.device)
            next_t = _as_tensor(next_obs, device=self.device)
            act_t = _as_tensor(actions, device=self.device)
            if act_t.dim() == 1:
                act_t = act_t.unsqueeze(-1)
            total, inv, fwd = self.icm_loss(obs_t, next_t, act_t)
            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            self.optimizer.step()
            return {
                "icm_loss": float(total.detach().cpu()),
                "icm_inverse": float(inv.detach().cpu()),
                "icm_forward": float(fwd.detach().cpu()),
            }

        def train_icm(
            self,
            dataset: Any,
            steps: int = 50_000,
            batch_size: int = SF_DEFAULT_BATCH_SIZE,
            log_interval: int = 5000,
            logger: Any = None,
            seed: int = 0,
        ) -> List[Dict[str, float]]:
            """Pretrain the ICM features on offline transitions."""
            rng = np.random.default_rng(seed)
            history: List[Dict[str, float]] = []
            self.train()
            for step in range(1, int(steps) + 1):
                batch = _sample_transition_dict(dataset, batch_size, rng)
                if batch is None:
                    break
                metrics = self.update(
                    batch["observations"], batch["next_observations"], batch["actions"]
                )
                if log_interval and step % log_interval == 0:
                    metrics = dict(metrics)
                    metrics["step"] = step
                    history.append(metrics)
                    if logger is not None:
                        try:
                            logger.log_metrics(metrics, step=step, prefix="sf/icm")
                        except Exception:
                            pass
            return history

        def forward(self, obs: Any) -> "torch.Tensor":  # type: ignore[override]
            return self.features(obs)


    class RandomFeatureExtractor(nn.Module):  # type: ignore[misc]
        """Frozen random-projection features (fallback if ICM is not trained).

        Matches the "linear function" prior family used by FB/SF: ``phi(s)`` is
        a frozen random Gaussian projection of the state.
        """

        def __init__(
            self,
            obs_dim: int,
            feature_dim: int = SF_FEATURE_DIM,
            seed: int = 0,
            device: str = "cpu",
        ) -> None:
            super().__init__()
            self.obs_dim = int(obs_dim)
            self.feature_dim = int(feature_dim)
            self.device = device
            generator = torch.Generator().manual_seed(seed)
            weight = torch.randn(self.obs_dim, self.feature_dim, generator=generator) / np.sqrt(
                self.obs_dim
            )
            bias = torch.zeros(self.feature_dim)
            self.register_buffer("weight", weight)
            self.register_buffer("bias", bias)
            for param in self.parameters():
                param.requires_grad_(False)
            self.to(device)

        def features(self, obs: Any) -> "torch.Tensor":
            obs_t = _as_tensor(obs, device=self.device)
            single = obs_t.dim() == 1
            if single:
                obs_t = obs_t.unsqueeze(0)
            feats = torch.tanh(obs_t @ self.weight + self.bias)
            return feats[0] if single else feats

        @torch.no_grad()
        def numpy_features(self, obs: Any) -> np.ndarray:
            return self.features(obs).detach().cpu().numpy()

        def forward(self, obs: Any) -> "torch.Tensor":  # type: ignore[override]
            return self.features(obs)


# ---------------------------------------------------------------------------
# Successor-feature model
# ---------------------------------------------------------------------------


if _TORCH_AVAILABLE:

    class SFModel(nn.Module):  # type: ignore[misc]
        """Successor features + reward features (SF factorisation).

        ``psi(s, a)`` is the successor-feature vector and ``phi(s)`` the reward
        feature vector, so that ``r(s) = phi(s)^T z`` and
        ``Q(s, a, z) = psi(s, a)^T z`` (Barreto et al., 2017; Borsa et al., 2018).
        When an external ICM feature extractor is provided, ``phi`` is that frozen
        network's output and only ``psi`` is learned.
        """

        def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            feature_dim: int = SF_FEATURE_DIM,
            hidden_layers: Sequence[int] = SF_DEFAULT_HIDDEN_LAYERS,
            activation: str = "relu",
            layernorm: bool = False,
            feature_extractor: Optional["nn.Module"] = None,
            learn_reward_features: bool = False,
        ) -> None:
            super().__init__()
            self.obs_dim = int(obs_dim)
            self.action_dim = int(action_dim)
            self.feature_dim = int(feature_dim)
            self.hidden_layers = tuple(int(h) for h in hidden_layers)
            self.activation = activation
            self.layernorm = layernorm
            self.learn_reward_features = bool(learn_reward_features)

            self.external_features = feature_extractor
            if feature_extractor is None or self.learn_reward_features:
                self.phi: Optional[nn.Module] = MLP(
                    self.obs_dim,
                    hidden_layers=self.hidden_layers,
                    output_dim=self.feature_dim,
                    activation=activation,
                    layernorm=layernorm,
                )
            else:
                self.phi = None

            self.psi = MLP(
                self.obs_dim + self.action_dim,
                hidden_layers=self.hidden_layers,
                output_dim=self.feature_dim,
                activation=activation,
                layernorm=layernorm,
            )

        # -- reward features ----------------------------------------------
        def reward_features(self, obs: "torch.Tensor") -> "torch.Tensor":
            obs_t = obs
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0)
            if self.external_features is not None and not self.learn_reward_features:
                feats = self.external_features.features(obs_t)
            else:
                assert self.phi is not None
                feats = self.phi(obs_t)
            return feats

        # -- successor features -------------------------------------------
        def successor_features(self, obs: "torch.Tensor", action: "torch.Tensor") -> "torch.Tensor":
            obs_t = obs
            act_t = action
            if obs_t.dim() == 1:
                obs_t = obs_t.unsqueeze(0)
            if act_t.dim() == 1:
                act_t = act_t.unsqueeze(0) if obs_t.dim() == 2 else act_t
            if act_t.dim() == obs_t.dim() - 1:
                act_t = act_t.unsqueeze(-1)
            if act_t.dim() == 1 and obs_t.dim() == 2:
                act_t = act_t.unsqueeze(0).expand(obs_t.shape[0], -1)
            return self.psi(torch.cat([obs_t, act_t], dim=-1))

        def forward_values(
            self, obs: "torch.Tensor", action: "torch.Tensor", z: "torch.Tensor"
        ) -> "torch.Tensor":
            feats = self.successor_features(obs, action)
            z_t = _as_tensor(z, device=feats.device)
            if z_t.dim() == 1:
                z_t = z_t.unsqueeze(0)
            return (feats * z_t).sum(dim=-1)

        def forward(
            self, obs: "torch.Tensor", action: "torch.Tensor", z: "torch.Tensor"
        ) -> "torch.Tensor":  # type: ignore[override]
            return self.forward_values(obs, action, z)


    class SFTrainingStats:
        """Aggregated diagnostics for an in-house SF training run."""

        __slots__ = (
            "steps",
            "seconds",
            "sf_loss",
            "reward_loss",
            "actor_loss",
            "bc_loss",
            "icm_metrics",
            "history",
        )

        def __init__(self) -> None:
            self.steps = 0
            self.seconds = 0.0
            self.sf_loss = 0.0
            self.reward_loss = 0.0
            self.actor_loss = 0.0
            self.bc_loss = 0.0
            self.icm_metrics: Dict[str, float] = {}
            self.history: List[Dict[str, float]] = []

        def as_dict(self) -> Dict[str, Any]:
            return {
                "steps": self.steps,
                "seconds": self.seconds,
                "sf_loss": self.sf_loss,
                "reward_loss": self.reward_loss,
                "actor_loss": self.actor_loss,
                "bc_loss": self.bc_loss,
                **{f"icm/{k}": v for k, v in self.icm_metrics.items()},
            }


# ---------------------------------------------------------------------------
# Self-contained SF agent (DDPG actor + successor features)
# ---------------------------------------------------------------------------


if _TORCH_AVAILABLE:

    class SuccessorFeaturesAgent:
        """In-house successor-features baseline.

        Structure mirrors ``forward_backward.ForwardBackwardAgent`` so the two
        can be swapped in evaluation scripts:

            * ``learned_features``: reward features ``phi(s)`` (ICM or learned MLP)
            * ``psi``: successor features ``psi(s, a)``
            * ``actor``: deterministic DDPG actor ``pi(s, z)``

        Training losses (Touati & Ollivier 2021 style SF-DDPG):

            L_psi   = || psi(s, a) - (phi(s) + gamma * psi_bar(s', pi(s', z))) ||^2
            L_phi   = || phi(s)^T z - r ||^2                     (task-vector regression)
            L_actor = - psi(s, pi(s, z))^T z  (+ optional BC term)

        Task vectors ``z`` are sampled from the random-function prior family
        (coordinates scaled by the reward-feature scale), and at test time they
        are recovered by *linear* ridge regression over 5120 (state, reward)
        samples - the structural restriction of SF that FRE removes.
        """

        def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            latent_dim: int = SF_DEFAULT_LATENT_DIM,
            hidden_layers: Sequence[int] = SF_DEFAULT_HIDDEN_LAYERS,
            activation: str = "relu",
            layernorm: bool = False,
            discount: float = SF_DEFAULT_DISCOUNT,
            learning_rate: float = SF_DEFAULT_LR,
            target_update_rate: float = SF_DEFAULT_TARGET_RATE,
            grad_clip_norm: float = 10.0,
            batch_size: int = SF_DEFAULT_BATCH_SIZE,
            bc_coef: float = 0.0,
            actor_learning_rate: Optional[float] = None,
            feature_learning_rate: float = 1e-4,
            device: str = "cpu",
            seed: int = 0,
            task_reward_scale: float = 1.0,
            task_mask_prob: float = 0.0,
            use_icm_features: bool = True,
            icm_steps: int = 50_000,
            icm_hidden_layers: Sequence[int] = ICM_DEFAULT_HIDDEN_LAYERS,
            learn_reward_features: bool = False,
            freeze_features: bool = True,
            action_low: Optional[np.ndarray] = None,
            action_high: Optional[np.ndarray] = None,
        ) -> None:
            self.obs_dim = int(obs_dim)
            self.action_dim = int(action_dim)
            self.latent_dim = int(latent_dim)
            self.hidden_layers = tuple(int(h) for h in hidden_layers)
            self.discount = float(discount)
            self.learning_rate = float(learning_rate)
            self.target_update_rate = float(target_update_rate)
            self.grad_clip_norm = float(grad_clip_norm)
            self.batch_size = int(batch_size)
            self.bc_coef = float(bc_coef)
            self.device = device
            self.seed = int(seed)
            self.task_reward_scale = float(task_reward_scale)
            self.task_mask_prob = float(task_mask_prob)
            self.use_icm_features = bool(use_icm_features)
            self.icm_steps = int(icm_steps)
            self.icm_hidden_layers = tuple(int(h) for h in icm_hidden_layers)
            self.learn_reward_features = bool(learn_reward_features)
            self.freeze_features = bool(freeze_features)
            self.action_low = None if action_low is None else np.asarray(action_low, dtype=np.float32)
            self.action_high = None if action_high is None else np.asarray(action_high, dtype=np.float32)

            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

            if self.use_icm_features:
                self.features = ICMFeatureExtractor(
                    obs_dim=self.obs_dim,
                    action_dim=self.action_dim,
                    feature_dim=self.latent_dim,
                    hidden_layers=self.icm_hidden_layers,
                    learning_rate=feature_learning_rate,
                    device=device,
                )
            else:
                self.features = RandomFeatureExtractor(
                    obs_dim=self.obs_dim, feature_dim=self.latent_dim, seed=self.seed, device=device
                )
            #: Optional externally injected reward-feature matrix (e.g. precomputed ICM).
            self._external_reward_features: Optional[np.ndarray] = None

            self.model = SFModel(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                feature_dim=self.latent_dim,
                hidden_layers=self.hidden_layers,
                activation=activation,
                layernorm=layernorm,
                feature_extractor=self.features,
                learn_reward_features=self.learn_reward_features,
            ).to(device)
            self.target_model = SFModel(
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                feature_dim=self.latent_dim,
                hidden_layers=self.hidden_layers,
                activation=activation,
                layernorm=layernorm,
                feature_extractor=self.features,
                learn_reward_features=self.learn_reward_features,
            ).to(device)
            self.target_model.load_state_dict(self.model.state_dict())

            self.actor = MLP(
                self.obs_dim + self.latent_dim,
                hidden_layers=self.hidden_layers,
                output_dim=self.action_dim,
                activation=activation,
                layernorm=layernorm,
                output_activation="tanh",
            ).to(device)

            self.model_optimizer = torch.optim.Adam(
                [p for p in self.model.parameters() if p.requires_grad], lr=self.learning_rate
            )
            self.actor_optimizer = torch.optim.Adam(
                self.actor.parameters(),
                lr=float(actor_learning_rate) if actor_learning_rate else self.learning_rate,
            )

            self.dataset: Any = None
            self._state_dim: Optional[int] = None
            self.history: List[Dict[str, float]] = []

            if self.freeze_features:
                for param in self.features.parameters():
                    param.requires_grad_(False)

        # ------------------------------------------------------------------
        # Feature / reward plumbing
        # ------------------------------------------------------------------
        def register_reward_features(self, features: Any) -> None:
            """Inject external reward features (e.g. precomputed ICM features).

            ``features`` may be a callable ``states -> (N, d)``, a
            ``(obs_dim, d)`` matrix, or a ``(d,)``-output torch module.
            """
            if callable(features):
                self._reward_feature_fn = features  # type: ignore[attr-defined]
                self._external_reward_features = None
            else:
                array = np.asarray(features, dtype=np.float32)
                self._external_reward_features = array
            self.model.external_features = None  # type: ignore[assignment]

        def _reward_feature_fn_call(self, states: np.ndarray) -> np.ndarray:
            fn = getattr(self, "_reward_feature_fn", None)
            if fn is not None:
                return np.asarray(fn(states), dtype=np.float32)
            if self._external_reward_features is not None:
                feats = np.asarray(states, dtype=np.float32) @ self._external_reward_features
                return feats.astype(np.float32)
            if self.use_icm_features:
                return self.features.numpy_features(states)
            # learned MLP features
            with torch.no_grad():
                feats = self.model.reward_features(_as_tensor(states, device=self.device))
            return feats.detach().cpu().numpy()

        def features_numpy(self, states: np.ndarray) -> np.ndarray:
            """Reward features ``phi(s)`` as numpy (for ridge regression)."""
            return self._reward_feature_fn_call(np.asarray(states, dtype=np.float32))

        # Alias matching the shared ``solve_task_vector`` helper.
        def reward_features(self, states: np.ndarray) -> np.ndarray:
            return self.features_numpy(states)

        def sample_task_vectors(self, num: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
            """Sample task vectors ``z`` from the random-function prior family.

            Uses an isotropic Gaussian scaled by ``task_reward_scale``; an
            optional Bernoulli mask (``task_mask_prob``) mirrors the sparse
            linear prior used for FB/SF.
            """
            rng = rng or np.random.default_rng(self.seed)
            z = rng.normal(0.0, self.task_reward_scale, size=(int(num), self.latent_dim))
            if self.task_mask_prob > 0.0:
                mask = (rng.random(size=(int(num), self.latent_dim)) > self.task_mask_prob).astype(np.float32)
                z = z * mask
            return z.astype(np.float32)

        def task_rewards(self, states: np.ndarray, z: np.ndarray) -> np.ndarray:
            """Reward of a task: ``r(s) = phi(s)^T z`` (linear in features)."""
            feats = self.features_numpy(states)
            z = np.asarray(z, dtype=np.float32).reshape(-1)
            return (feats * z[None, :]).sum(axis=-1)

        # Alias used by OPAL-style suite evaluators.
        def intrinsic_reward(self, obs: np.ndarray, z: np.ndarray, next_obs: Any = None) -> np.ndarray:
            return self.task_rewards(np.asarray(obs, dtype=np.float32), z)

        # ------------------------------------------------------------------
        # Dataset
        # ------------------------------------------------------------------
        def attach_dataset(self, dataset: Any) -> None:
            self.dataset = dataset
            self._state_dim = _infer_state_dim(dataset)

        def _sample_batch(self, batch_size: Optional[int] = None) -> Dict[str, np.ndarray]:
            size = int(batch_size or self.batch_size)
            rng = np.random.default_rng(self.seed + len(self.history))
            batch = _sample_transition_dict(self.dataset, size, rng)
            if batch is None:
                raise ValueError("SuccessorFeaturesAgent has no dataset attached.")
            return batch

        # ------------------------------------------------------------------
        # Training
        # ------------------------------------------------------------------
        def _sample_z_for_batch(self, batch: Dict[str, np.ndarray]) -> np.ndarray:
            """Sample task vector(s) whose reward scale matches the batch targets."""
            z = self.sample_task_vectors(batch["observations"].shape[0])[:, None, :]
            return np.repeat(z, 1, axis=1)

        def update(
            self,
            batch: Optional[Dict[str, np.ndarray]] = None,
            batch_size: Optional[int] = None,
            target_update: bool = True,
            z: Optional[np.ndarray] = None,
        ) -> Dict[str, float]:
            """One SF-DDPG update: successor-feature TD + reward regression + actor."""
            if batch is None:
                batch = self._sample_batch(batch_size)
            obs = _as_tensor(batch["observations"], device=self.device)
            next_obs = _as_tensor(batch["next_observations"], device=self.device)
            actions = _as_tensor(batch["actions"], device=self.device)
            if actions.dim() == 1:
                actions = actions.unsqueeze(-1)
            terminals = _as_tensor(
                batch.get("terminals", np.zeros(len(batch["observations"]))), device=self.device
            ).float()
            if terminals.dim() == 0:
                terminals = terminals.unsqueeze(0)

            if z is None:
                z_np = self.sample_task_vectors(obs.shape[0])
            else:
                z_np = np.asarray(z, dtype=np.float32)
                if z_np.ndim == 1:
                    z_np = np.tile(z_np[None, :], (obs.shape[0], 1))
            z_t = _as_tensor(z_np, device=self.device)

            # --- successor-feature fixed point -----------------------------
            with torch.no_grad():
                next_actions = self.actor(torch.cat([next_obs, z_t], dim=-1))
                next_psi = self.target_model.successor_features(next_obs, next_actions)
                feats = self.model.reward_features(obs)
                target_psi = feats + self.discount * (1.0 - terminals.unsqueeze(-1)) * next_psi
            pred_psi = self.model.successor_features(obs, actions)
            sf_loss = F.mse_loss(pred_psi, target_psi)

            # --- reward-feature regression (only when features are learned) --
            reward_loss = torch.zeros((), device=obs.device)
            if self.learn_reward_features:
                pred_rewards = (feats * z_t).sum(dim=-1)
                target_rewards = (target_psi * z_t).sum(dim=-1).detach()
                reward_loss = F.mse_loss(pred_rewards, target_rewards)

            # --- DDPG actor: maximize psi(s, pi(s,z))^T z -------------------
            actor_actions = self.actor(torch.cat([obs, z_t], dim=-1))
            actor_loss = -(self.model.successor_features(obs, actor_actions) * z_t).sum(dim=-1).mean()
            if self.bc_coef > 0.0:
                bc_loss = F.mse_loss(actor_actions, actions)
                actor_loss = actor_loss + self.bc_coef * bc_loss
            else:
                bc_loss = torch.zeros((), device=obs.device)

            self.model_optimizer.zero_grad(set_to_none=True)
            (sf_loss + reward_loss).backward(retain_graph=True)
            if self.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.model_optimizer.step()

            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            if self.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
            self.actor_optimizer.step()

            if target_update:
                self.update_targets()

            return {
                "sf_loss": float(sf_loss.detach().cpu()),
                "reward_loss": float(reward_loss.detach().cpu()),
                "actor_loss": float(actor_loss.detach().cpu()),
                "bc_loss": float(bc_loss.detach().cpu()),
                "q_mean": float((pred_psi.detach() * z_t).sum(dim=-1).mean().cpu()),
            }

        @torch.no_grad()
        def update_targets(self, rate: Optional[float] = None) -> None:
            rate = self.target_update_rate if rate is None else float(rate)
            for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.mul_(1.0 - rate).add_(param.data, alpha=rate)

        def train_representation(
            self,
            dataset: Any = None,
            steps: int = 50_000,
            batch_size: Optional[int] = None,
            log_interval: int = 5000,
            logger: Any = None,
            seed: Optional[int] = None,
        ) -> List[Dict[str, float]]:
            """Phase 1a: pretrain ICM features on offline transitions."""
            if dataset is not None:
                self.attach_dataset(dataset)
            if not self.use_icm_features or steps <= 0:
                return []
            history = self.features.train_icm(
                self.dataset,
                steps=int(steps),
                batch_size=int(batch_size or self.batch_size),
                log_interval=log_interval,
                logger=logger,
                seed=self.seed if seed is None else seed,
            )
            self.history.extend(history)
            return history

        def train_policy(
            self,
            dataset: Any = None,
            steps: int = 200_000,
            batch_size: Optional[int] = None,
            log_interval: int = 5000,
            logger: Any = None,
            stats: Optional["SFTrainingStats"] = None,
            seed: Optional[int] = None,
        ) -> "SFTrainingStats":
            """Phase 1b: train successor features + DDPG actor."""
            if dataset is not None:
                self.attach_dataset(dataset)
            stats = stats or SFTrainingStats()
            rng = np.random.default_rng(self.seed if seed is None else seed)
            start = time.time()
            self.model.train()
            self.actor.train()
            for step in range(1, int(steps) + 1):
                batch = _sample_transition_dict(self.dataset, int(batch_size or self.batch_size), rng)
                metrics = self.update(batch=batch, target_update=True)
                stats.steps += 1
                if log_interval and step % log_interval == 0:
                    record = dict(metrics)
                    record["step"] = step
                    stats.history.append(record)
                    self.history.append(record)
                    stats.sf_loss = metrics.get("sf_loss", stats.sf_loss)
                    stats.reward_loss = metrics.get("reward_loss", stats.reward_loss)
                    stats.actor_loss = metrics.get("actor_loss", stats.actor_loss)
                    stats.bc_loss = metrics.get("bc_loss", stats.bc_loss)
                    if logger is not None:
                        try:
                            logger.log_metrics(metrics, step=step, prefix="sf")
                        except Exception:
                            pass
            stats.seconds += time.time() - start
            return stats

        # ------------------------------------------------------------------
        # Inference
        # ------------------------------------------------------------------
        @torch.no_grad()
        def select_action(
            self,
            obs: np.ndarray,
            z: np.ndarray,
            deterministic: bool = True,
            clip: bool = True,
        ) -> np.ndarray:
            obs_np = np.asarray(obs, dtype=np.float32).reshape(-1)
            z_np = np.asarray(z, dtype=np.float32).reshape(-1)
            inp = _as_tensor(np.concatenate([obs_np, z_np])[None, :], device=self.device)
            action = self.actor(inp)[0].detach().cpu().numpy().astype(np.float32)
            action = _scale_action(action, self.action_low, self.action_high, clip=clip)
            return action

        # Alias for API compatibility with IQL/OPAL agents.
        act = select_action

        @torch.no_grad()
        def value_of(self, obs: np.ndarray, z: np.ndarray) -> float:
            obs_np = np.asarray(obs, dtype=np.float32).reshape(-1)
            z_np = np.asarray(z, dtype=np.float32).reshape(-1)
            action = self.select_action(obs_np, z_np)
            q = self.model.forward_values(
                _as_tensor(obs_np[None, :], device=self.device),
                _as_tensor(action[None, :], device=self.device),
                _as_tensor(z_np[None, :], device=self.device),
            )
            return float(q[0].detach().cpu())

        # ------------------------------------------------------------------
        # Persistence / device
        # ------------------------------------------------------------------
        def state_dict(self) -> Dict[str, Any]:
            return {
                "model": self.model.state_dict(),
                "target_model": self.target_model.state_dict(),
                "actor": self.actor.state_dict(),
            }

        def load_state_dict(self, state: Dict[str, Any]) -> None:
            self.model.load_state_dict(state["model"])
            self.target_model.load_state_dict(state.get("target_model", state["model"]))
            self.actor.load_state_dict(state["actor"])

        def save(self, path: str) -> str:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
            torch.save(self.state_dict(), path)
            return path

        def load(self, path: str, map_location: Optional[str] = None) -> None:
            state = torch.load(path, map_location=map_location or self.device)
            self.load_state_dict(state)

        def to(self, device: str) -> "SuccessorFeaturesAgent":
            self.device = device
            self.model.to(device)
            self.target_model.to(device)
            self.actor.to(device)
            if self.features is not None:
                self.features.to(device)
            return self

        def parameters(self):
            yield from self.model.parameters()
            yield from self.actor.parameters()

        def train(self) -> None:
            self.model.train()
            self.target_model.train()
            self.actor.train()

        def eval(self) -> None:
            self.model.eval()
            self.target_model.eval()
            self.actor.eval()

        # ------------------------------------------------------------------
        # Factory
        # ------------------------------------------------------------------
        @classmethod
        def from_config(
            cls,
            config: Any,
            obs_dim: int,
            action_dim: int,
            device: Optional[str] = None,
            **overrides: Any,
        ) -> "SuccessorFeaturesAgent":
            device = device or getattr(config, "device", "cpu")
            kwargs: Dict[str, Any] = {
                "latent_dim": getattr(config, "sf_feature_dim", SF_FEATURE_DIM),
                "hidden_layers": getattr(config, "rl_hidden_layers", SF_DEFAULT_HIDDEN_LAYERS),
                "activation": getattr(config, "rl_activation", "relu"),
                "discount": getattr(config, "discount", SF_DEFAULT_DISCOUNT),
                "learning_rate": getattr(config, "learning_rate", SF_DEFAULT_LR),
                "batch_size": getattr(config, "batch_size", SF_DEFAULT_BATCH_SIZE),
                "seed": getattr(config, "seed", 0),
                "use_icm_features": True,
                "icm_steps": getattr(config, "sf_icm_steps", 50_000),
            }
            kwargs.update(overrides)
            return cls(obs_dim=obs_dim, action_dim=action_dim, device=device, **kwargs)


    SFAgent = SuccessorFeaturesAgent


# ---------------------------------------------------------------------------
# Launcher for the official controllable_agent SF implementation
# ---------------------------------------------------------------------------


def build_sf_command(
    env: str,
    dataset: str = SF_DATASET,
    train_steps: Optional[int] = None,
    ca_dir: Optional[str] = None,
    python: str = "python",
    output_dir: Optional[str] = None,
    seed: int = 0,
    icm_features: bool = True,
    extra_args: Optional[Sequence[str]] = None,
) -> List[str]:
    """Build the ``controllable_agent`` CLI invocation for the SF baseline.

    Mirrors the FB command builder but forces ``--alg sf`` and enables the ICM
    feature learning path used for the SF comparison (Sec. 5.2).
    """
    kwargs: Dict[str, Any] = {
        "env": env,
        "alg": "sf",
        "dataset": dataset,
        "seed": seed,
        "num_train_steps": train_steps,
        "output_dir": output_dir,
        "ca_dir": ca_dir,
        "python": python,
    }
    try:
        command = build_controllable_agent_command(**{k: v for k, v in kwargs.items() if v is not None})
    except ControllableAgentUnavailable as exc:  # pragma: no cover
        raise exc
    if icm_features:
        command.extend(["--icm", "--icm_features"])
    if extra_args:
        command.extend([str(a) for a in extra_args])
    return command


def run_sf_controllable_agent(
    env: str,
    ca_dir: Optional[str] = None,
    dataset: str = SF_DATASET,
    train_steps: Optional[int] = None,
    output_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    timeout: Optional[float] = None,
    dry_run: bool = False,
    icm_features: bool = True,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run the official controllable_agent SF baseline in a subprocess."""
    command = build_sf_command(
        env=env,
        dataset=dataset,
        train_steps=train_steps,
        ca_dir=ca_dir,
        output_dir=output_dir,
        icm_features=icm_features,
        **kwargs,
    )
    if dry_run:
        return {"command": command, "dry_run": True}
    return run_controllable_agent(
        env,
        ca_dir=ca_dir,
        output_dir=output_dir,
        log_file=log_file,
        timeout=timeout,
        extra_args=command[3:] if len(command) > 3 else None,
    )


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def make_sf_policy_fn(
    agent: "SuccessorFeaturesAgent",
    z: np.ndarray,
    deterministic: bool = True,
    clip: bool = True,
) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap an SF agent + recovered task vector into ``act_fn(obs) -> action``."""

    def act_fn(obs: np.ndarray) -> np.ndarray:
        return agent.select_action(obs, z, deterministic=deterministic, clip=clip)

    return act_fn


def make_successor_features(
    config: Any,
    obs_dim: int,
    action_dim: int,
    device: Optional[str] = None,
    **overrides: Any,
) -> "SuccessorFeaturesAgent":
    """Build an SF agent from a FRE ``Config``-like object."""
    return SuccessorFeaturesAgent.from_config(
        config, obs_dim=obs_dim, action_dim=action_dim, device=device, **overrides
    )


def train_successor_features(
    config: Any,
    dataset: Any,
    obs_dim: Optional[int] = None,
    action_dim: Optional[int] = None,
    icm_steps: Optional[int] = None,
    sf_steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    log_interval: int = 5000,
    logger: Any = None,
    agent: Optional["SuccessorFeaturesAgent"] = None,
    device: Optional[str] = None,
    prefer_official: bool = False,
    ca_dir: Optional[str] = None,
    **overrides: Any,
) -> Tuple["SuccessorFeaturesAgent", List[Dict[str, float]]]:
    """Train the SF baseline (ICM features then successor features + actor).

    The paper ran SF through the official ``controllable_agent`` codebase; when
    that checkout is available and ``prefer_official=True`` the subprocess is
    launched in addition to the in-house agent (which is what evaluation uses
    locally).
    """
    device = device or getattr(config, "device", "cpu")
    obs_dim = obs_dim or _infer_state_dim(dataset) or getattr(config, "obs_dim", 0)
    action_dim = action_dim or _infer_action_dim(dataset) or getattr(config, "action_dim", 0)

    official_log: Dict[str, Any] = {"launched": False}
    if prefer_official and controllable_agent_available(ca_dir):
        env_id = getattr(config, "env_id", None) or getattr(config, "domain", "antmaze")
        try:
            official_log = run_sf_controllable_agent(
                env=env_id,
                ca_dir=ca_dir,
                train_steps=sf_steps or getattr(config, "policy_train_steps", None),
                dry_run=False,
            )
            official_log["launched"] = True
        except Exception as exc:  # pragma: no cover - external dependency
            official_log = {"launched": False, "error": str(exc)}
    elif prefer_official:
        official_log = {"launched": False, "error": "controllable_agent checkout not found"}

    if agent is None:
        agent = make_successor_features(
            config, obs_dim=obs_dim, action_dim=action_dim, device=device, **overrides
        )
    agent.attach_dataset(dataset)

    icm_steps = int(
        icm_steps if icm_steps is not None else getattr(config, "sf_icm_steps", 50_000)
    )
    sf_steps = int(
        sf_steps
        if sf_steps is not None
        else getattr(config, "policy_steps", lambda: 850_000)()
        if callable(getattr(config, "policy_steps", None))
        else getattr(config, "policy_train_steps", 850_000)
    )
    history: List[Dict[str, float]] = []
    history.extend(
        agent.train_representation(
            dataset=dataset, steps=icm_steps, batch_size=batch_size, log_interval=log_interval, logger=logger
        )
    )
    stats = agent.train_policy(
        dataset=dataset, steps=sf_steps, batch_size=batch_size, log_interval=log_interval, logger=logger
    )
    history.extend(stats.history)
    if official_log.get("launched"):
        history.append({"official_launch": 1.0})
    return agent, history


def evaluate_sf_suite(
    agent: "SuccessorFeaturesAgent",
    suite_evaluate_fn: Callable[[Callable[[np.ndarray], np.ndarray], int], Dict[str, Any]],
    dataset: Any = None,
    num_samples: int = SF_EVAL_SAMPLES,
    seed: int = 0,
    tasks: Optional[Sequence[Any]] = None,
    task_names: Optional[Sequence[str]] = None,
    base_seed: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Zero-shot evaluation of SF across a task suite.

    For every task, ``z`` is recovered by *linear ridge regression* over
    ``num_samples`` (5120) reward-annotated states - the test-time adaptation
    protocol used for FB/SF in Sec. 5.2 - before rolling out the DDPG actor.
    """
    if dataset is None:
        dataset = agent.dataset
    rng = np.random.default_rng(seed)
    results: Dict[str, Any] = {}
    task_list = list(tasks) if tasks is not None else list(kwargs.pop("task_list", []))

    scores: List[float] = []
    for index, task in enumerate(task_list):
        name = task_names[index] if task_names is not None else getattr(task, "name", f"task-{index}")
        states, rewards = sample_eval_reward_samples(
            task, dataset, num_samples=num_samples, rng=rng, **kwargs
        )
        z = solve_task_vector(agent, states, rewards, ridge=1e-3, normalize_rewards=True)
        act_fn = make_sf_policy_fn(agent, z)
        task_seed = (base_seed if base_seed is not None else seed) + index
        try:
            task_result = suite_evaluate_fn(act_fn, task_seed)
        except TypeError:
            task_result = suite_evaluate_fn(act_fn)
        results[name] = task_result
        score = task_result.get("score", task_result.get("mean")) if isinstance(task_result, dict) else task_result
        if score is not None:
            scores.append(float(score))

    if scores:
        results["sf-all"] = {
            "score": float(np.mean(scores)),
            "score_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
            "num_tasks": len(scores),
        }
    return results


def sf_reference_row(domain: str) -> Tuple[float, float]:
    """Return the Table 1 reference ``(mean, std)`` for the SF baseline."""
    key = domain if domain in SF_TABLE1_REFERENCE else f"{domain}-all"
    return SF_TABLE1_REFERENCE.get(key, (float("nan"), float("nan")))


def dump_sf_config(config: Any, path: str) -> str:
    """Persist SF hyperparameters together with the Table 1 reference row."""
    payload: Dict[str, Any] = {
        "baseline": "successor_features",
        "eval_samples": SF_EVAL_SAMPLES,
        "dataset": SF_DATASET,
        "feature_dim": SF_FEATURE_DIM,
        "table1_reference": {k: list(v) for k, v in SF_TABLE1_REFERENCE.items()},
    }
    if config is not None and hasattr(config, "to_dict"):
        try:
            payload["config"] = config.to_dict()
        except Exception:
            pass
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _as_tensor(value: Any, device: str = "cpu") -> "torch.Tensor":
    """Convert numpy/list/torch input into a float32 tensor on ``device``."""
    if _TORCH_AVAILABLE and isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float32)
    array = np.asarray(value, dtype=np.float32)
    return torch.as_tensor(array, device=device)


def _scale_action(
    action: np.ndarray,
    low: Optional[np.ndarray],
    high: Optional[np.ndarray],
    clip: bool = True,
) -> np.ndarray:
    """Squash a tanh-scaled action in [-1, 1] onto the environment action range."""
    if clip:
        action = np.clip(action, -1.0, 1.0)
    if low is not None and high is not None:
        low = np.asarray(low, dtype=np.float32)
        high = np.asarray(high, dtype=np.float32)
        action = low + 0.5 * (action + 1.0) * (high - low)
    return action.astype(np.float32)


def _infer_state_dim(dataset: Any) -> Optional[int]:
    if dataset is None:
        return None
    for attr in ("states", "observations"):
        value = getattr(dataset, attr, None)
        if value is not None:
            array = np.asarray(value)
            return int(array.shape[-1])
    return None


def _infer_action_dim(dataset: Any) -> Optional[int]:
    if dataset is None:
        return None
    for attr in ("actions",):
        value = getattr(dataset, attr, None)
        if value is not None:
            array = np.asarray(value)
            return int(array.shape[-1])
    return None


def _sample_transition_dict(
    dataset: Any, batch_size: int, rng: np.random.Generator
) -> Optional[Dict[str, np.ndarray]]:
    """Return a dict of transition arrays from a duck-typed offline dataset."""
    if dataset is None:
        return None
    if hasattr(dataset, "sample_transitions"):
        batch = dataset.sample_transitions(int(batch_size))
        if isinstance(batch, dict):
            return {k: np.asarray(v) for k, v in batch.items()}
    states = None
    for attr in ("observations", "states"):
        candidate = getattr(dataset, attr, None)
        if candidate is not None:
            states = np.asarray(candidate)
            break
    if states is None:
        return None
    actions = getattr(dataset, "actions", None)
    if actions is None:
        return None
    actions = np.asarray(actions)
    next_states = getattr(dataset, "next_observations", None)
    if next_states is None:
        index = np.arange(len(states) - 1)
        next_states = states[index + 1]
        states = states[index]
        actions = actions[index]
        terminals = np.zeros(len(index), dtype=np.float32)
        keep = (index + 1) % max(len(states), 1) != 0
        terminals[~keep] = 1.0
    else:
        next_states = np.asarray(next_states)
        terminals = np.asarray(getattr(dataset, "terminals", np.zeros(len(states), dtype=np.float32)))
    num = len(states)
    idx = rng.integers(0, num, size=int(batch_size))
    return {
        "observations": states[idx].astype(np.float32),
        "actions": actions[idx].astype(np.float32),
        "next_observations": next_states[idx].astype(np.float32),
        "terminals": terminals[idx].astype(np.float32),
    }
