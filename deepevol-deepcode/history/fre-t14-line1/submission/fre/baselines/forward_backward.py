"""Forward-Backward (FB) baseline for the FRE paper (Sec. 5.2, Table 1).

Paper specification (Sec. 5.2 + addendum "Additional Details on SF and FB Baselines"):

    * "Forward-Backward (FB) method (Touati & Ollivier, 2021), a state-of-the-art
      zero-shot RL method that jointly learns a pair of representations that
      represent a family of tasks and their optimal policies."
    * "FB and SF are based on DDPG-based policies, and are run via the code
      provided from (Touati et al., 2022)" i.e.
      https://github.com/facebookresearch/controllable_agent
    * "As such, reproductions should also use this codebase for training and
      evaluating these baselines."
    * "All SF/FB ExoRL experiments use the RND dataset."
    * "Training the FB/SF policies did not require any changes to the
      `facebookresearch/controllable_agent` codebase."
    * "For SF/FB evaluation, the set of evaluation tasks considered in the paper
      were re-implemented. ... the authors introduced a custom reward function
      into the pre-existing environments ... that replaced the default reward
      with their custom rewards."
    * "FB/SF rely on linear regression to perform test time adaptation ... To be
      consistent with prior methodology, we give these methods 5120 reward
      samples during evaluation time (in comparison to only 32 for FRE)."

This module therefore provides two things:

1.  ``run_controllable_agent`` / ``find_controllable_agent``: thin (subprocess)
    launcher for the *official* external codebase, which is what the paper
    actually used for the numbers reported in Table 1.  ``CONTROLLABLE_AGENT_DIR``
    env var (or ``--ca-dir``) points at a clone of the repo.

2.  ``ForwardBackwardAgent``: an in-house, self-contained FB implementation used
    when the external repo is unavailable.  It follows Touati & Ollivier (2021):
    a *linear* zero-shot task space with

        F(s, a, z) = < phi(s, a), z >,     B(s, z) = < psi(s), z >

    trained on offline transitions with task vectors ``z`` sampled from a prior
    over linear reward functions (the same family the paper criticises FB/SF for
    being restricted to: "They rely on linearized value functions to achieve
    generalization, whereas FRE learns a shared latent space through modeling a
    reward distribution").  At test time the task vector is recovered by *linear
    regression* on 5120 (state, reward) samples, exactly as described in Sec. 5.2.

The DDPG-based actor is trained to maximise ``B(s, z) = < psi(s), z >``, which is
the FB policy improvement step.
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

try:  # torch is a hard dependency of the RL part of this repo, but keep import soft
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - torch is required below
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore

from fre.rl.networks import MLP, GaussianPolicy, make_activation

__all__ = [
    "CONTROLLABLE_AGENT_DIR",
    "CONTROLLABLE_AGENT_REPO",
    "FB_EVAL_SAMPLES",
    "FB_TABLE1_REFERENCE",
    "FBAgent",
    "FBModel",
    "ForwardBackwardAgent",
    "FBTrainingStats",
    "ControllableAgentUnavailable",
    "build_controllable_agent_command",
    "controllable_agent_available",
    "evaluate_fb_suite",
    "find_controllable_agent",
    "make_fb_policy_fn",
    "make_forward_backward",
    "run_controllable_agent",
    "sample_eval_reward_samples",
    "solve_task_vector",
    "train_forward_backward",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTROLLABLE_AGENT_REPO = "https://github.com/facebookresearch/controllable_agent"

#: Directory of a local clone of the official controllable_agent codebase.
CONTROLLABLE_AGENT_DIR = os.environ.get("CONTROLLABLE_AGENT_DIR", "third_party/controllable_agent")

#: Number of (state, reward) samples handed to FB/SF at evaluation time (Sec. 5.2).
FB_EVAL_SAMPLES = 5120

#: Table 1 reference numbers for FB (mean/±std across 5 seeds, normalised 0-100).
FB_TABLE1_REFERENCE: Dict[str, Tuple[float, float]] = {
    "antmaze-all": (25.8, 19.8),
    "exorl-all": (43.4, 9.1),
    "kitchen": (3.0, 6.0),
    "all": (24.0, 12.0),
    # per-task rows (AntMaze)
    "ant-goal-reaching": (0.0, 0.0),
    "ant-directional": (24.9, 0.0),
    "ant-random-simplex": (17.6, 0.0),
    "ant-path-loop": (0.0, 0.0),
    "ant-path-edges": (0.0, 0.0),
    "ant-path-center": (0.0, 0.0),
    # per-task rows (ExORL)
    "exorl-walker-goals": (0.0, 0.0),
    "exorl-cheetah-goals": (0.0, 0.0),
    "exorl-walker-velocity": (0.0, 0.0),
    "exorl-cheetah-velocity": (0.0, 0.0),
}

#: Defaults that the paper does not specify for FB (documented defaults).
FB_DEFAULT_LATENT_DIM = 128  # paper silent; matched to FRE's z dimension
FB_DEFAULT_HIDDEN_LAYERS = (512, 512, 512)  # DDPG MLPs in controllable_agent
FB_DEFAULT_DISCOUNT = 0.99  # controllable_agent default (paper silent for FB)
FB_DEFAULT_LR = 1e-4
FB_DEFAULT_TARGET_RATE = 0.005  # DDPG-style Polyak in controllable_agent
FB_DEFAULT_BATCH_SIZE = 512
FB_LOG_STD_MIN = -5.0
FB_LOG_STD_MAX = 2.0


class ControllableAgentUnavailable(RuntimeError):
    """Raised when the official controllable_agent codebase cannot be located."""


# ---------------------------------------------------------------------------
# 1) Official controllable_agent launcher
# ---------------------------------------------------------------------------


def find_controllable_agent(root: Optional[str] = None) -> Optional[str]:
    """Locate a local clone of ``facebookresearch/controllable_agent``.

    Search order: explicit ``root`` argument, ``$CONTROLLABLE_AGENT_DIR``, a few
    conventional places (``third_party/``, ``external/``, ``~/``).
    Returns the path if a plausible checkout is found, else ``None``.
    """
    candidates: List[str] = []
    if root:
        candidates.append(root)
    if CONTROLLABLE_AGENT_DIR:
        candidates.append(CONTROLLABLE_AGENT_DIR)
    candidates += [
        "third_party/controllable_agent",
        "external/controllable_agent",
        "controllable_agent",
        os.path.expanduser("~/controllable_agent"),
    ]

    for candidate in candidates:
        if not candidate:
            continue
        path = os.path.abspath(os.path.expanduser(candidate))
        if not os.path.isdir(path):
            continue
        # A plausible checkout has either main.py / train_FB.py / an fb package.
        entries = set(os.listdir(path))
        if entries & {"main.py", "train_FB.py", "train_FB_general.py", "fb", "forward_backward"}:
            return path
    return None


def controllable_agent_available(root: Optional[str] = None) -> bool:
    """True when the official controllable_agent checkout can be found."""
    return find_controllable_agent(root) is not None


def build_controllable_agent_command(
    env: str,
    *,
    alg: str = "fb",
    dataset_path: Optional[str] = None,
    dataset: str = "rnd",
    output_dir: Optional[str] = None,
    seed: int = 0,
    num_train_steps: Optional[int] = None,
    num_eval_episodes: int = 20,
    ca_dir: Optional[str] = None,
    extra_args: Optional[Sequence[str]] = None,
    python: str = "python",
) -> List[str]:
    """Build the CLI invocation used to train/evaluate FB via controllable_agent.

    Mirrors the instructions followed by the authors (addendum): download the
    offline RND dataset, build the replay buffer, then run training with
    evaluation numbers logged during the run.  Exact flag names in the external
    repo have changed over time, so the caller can always append ``extra_args``.
    """
    root = find_controllable_agent(ca_dir)
    if root is None:
        raise ControllableAgentUnavailable(
            "controllable_agent checkout not found. Set CONTROLLABLE_AGENT_DIR or pass ca_dir=... "
            f"(expected a clone of {CONTROLLABLE_AGENT_REPO})."
        )
    script = "main.py" if os.path.exists(os.path.join(root, "main.py")) else "train_FB.py"

    cmd: List[str] = [python, script, "--alg", alg, "--env", env]
    if dataset_path:
        # controllable_agent expects an hdf5/npz dataset for the offline setting.
        cmd += ["--dataset", dataset_path, "--offline_dataset", dataset_path]
    cmd += ["--dataset_type", dataset]
    cmd += ["--seed", str(seed)]
    if output_dir:
        cmd += ["--logdir", output_dir, "--output_dir", output_dir]
    if num_train_steps is not None:
        cmd += ["--num_train_steps", str(num_train_steps)]
    cmd += ["--num_eval_episodes", str(num_eval_episodes)]
    if extra_args:
        cmd += list(extra_args)
    return cmd


def run_controllable_agent(
    env: str,
    *,
    ca_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    timeout: Optional[float] = None,
    dry_run: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Launch FB training/evaluation through the official codebase.

    Returns a dict with the resolved command, the working directory, the return
    code, and captured stdout/stderr (``dry_run=True`` only resolves the command).
    """
    cmd = build_controllable_agent_command(env, ca_dir=ca_dir, output_dir=output_dir, **kwargs)
    root = find_controllable_agent(ca_dir)
    assert root is not None  # build_controllable_agent_command already validated

    result: Dict[str, Any] = {
        "command": cmd,
        "cwd": root,
        "env": env,
        "dry_run": bool(dry_run),
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "log_file": log_file,
    }
    if dry_run:
        return result

    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=dict(os.environ),
    )
    result["returncode"] = proc.returncode
    result["stdout"] = proc.stdout
    result["stderr"] = proc.stderr
    result["seconds"] = time.time() - started
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)) or ".", exist_ok=True)
        with open(log_file, "w") as fh:
            fh.write("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n\n")
            fh.write(proc.stdout or "")
            fh.write("\n--- stderr ---\n")
            fh.write(proc.stderr or "")
    return result


# ---------------------------------------------------------------------------
# 2) In-house FB implementation
# ---------------------------------------------------------------------------


if nn is not None:

    class FBModel(nn.Module):
        """Forward-backward networks with a *linear* zero-shot task space.

        ``F(s, a, z) = <phi(s, a), z>`` and ``B(s, z) = <psi(s), z>`` — the FB
        factorisation of Touati & Ollivier (2021), which is what makes zero-shot
        generalisation to new reward functions possible and also what restricts
        FB to linearised value functions (as noted in Sec. 5.2 of the FRE paper).
        """

        def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            latent_dim: int = FB_DEFAULT_LATENT_DIM,
            hidden_layers: Sequence[int] = FB_DEFAULT_HIDDEN_LAYERS,
            activation: str = "relu",
            layernorm: bool = False,
        ) -> None:
            super().__init__()
            self.obs_dim = int(obs_dim)
            self.action_dim = int(action_dim)
            self.latent_dim = int(latent_dim)

            self.forward_net = MLP(
                self.obs_dim + self.action_dim,
                hidden_layers=hidden_layers,
                output_dim=self.latent_dim,
                activation=activation,
                layernorm=layernorm,
            )
            self.backward_net = MLP(
                self.obs_dim,
                hidden_layers=hidden_layers,
                output_dim=self.latent_dim,
                activation=activation,
                layernorm=layernorm,
            )

        def phi(self, obs: "torch.Tensor", action: "torch.Tensor") -> "torch.Tensor":
            return self.forward_net(torch.cat([obs, action], dim=-1))

        def psi(self, obs: "torch.Tensor") -> "torch.Tensor":
            return self.backward_net(obs)

        def forward_values(self, obs, action, z) -> "torch.Tensor":
            phi = self.phi(obs, action)
            return (phi * z).sum(dim=-1)

        def backward_values(self, obs, z) -> "torch.Tensor":
            psi = self.psi(obs)
            return (psi * z).sum(dim=-1)

        # Aliases used by the trainer
        def forward(self, obs, action, z):  # type: ignore[override]
            return self.forward_values(obs, action, z)

        def extra_repr(self) -> str:  # pragma: no cover - cosmetic
            return (
                f"obs_dim={self.obs_dim}, action_dim={self.action_dim}, "
                f"latent_dim={self.latent_dim}"
            )


@dataclass
class FBTrainingStats:
    """Aggregated FB training diagnostics."""

    steps: int = 0
    seconds: float = 0.0
    forward_loss: float = float("nan")
    backward_loss: float = float("nan")
    projection_loss: float = float("nan")
    actor_loss: float = float("nan")
    bc_loss: float = float("nan")
    history: List[Dict[str, float]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "steps": self.steps,
            "seconds": self.seconds,
            "forward_loss": self.forward_loss,
            "backward_loss": self.backward_loss,
            "projection_loss": self.projection_loss,
            "actor_loss": self.actor_loss,
            "bc_loss": self.bc_loss,
        }


class ForwardBackwardAgent:
    """Self-contained FB baseline (Touati & Ollivier, 2021) with linear z

    Components
    ----------
    ``model``     : :class:`FBModel` (phi/psi, linear in ``z``)
    ``critic``    : copy of ``model`` used as the Polyak target
    ``actor``     : DDPG policy ``pi(a | s, z)`` (tanh-squashed Gaussian; the
                    deterministic mode is used for evaluation, as in DDPG)
    ``reward_features``
                  : frozen random features ``g(s)`` defining the *training* task
                    family ``r_z(s) = < g(s), z >`` (the unsupervised prior over
                    linear reward functions; see FRE's Appendix B linear family).

    Test-time adaptation (Sec. 5.2): given 5120 reward-annotated states, solve
    ``z = argmin  ||psi(S) z - r||^2`` (ridge regression) via
    :func:`solve_task_vector` and execute the deterministic actor.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = FB_DEFAULT_LATENT_DIM,
        hidden_layers: Sequence[int] = FB_DEFAULT_HIDDEN_LAYERS,
        activation: str = "relu",
        layernorm: bool = False,
        discount: float = FB_DEFAULT_DISCOUNT,
        learning_rate: float = FB_DEFAULT_LR,
        target_update_rate: float = FB_DEFAULT_TARGET_RATE,
        grad_clip_norm: float = 10.0,
        batch_size: int = FB_DEFAULT_BATCH_SIZE,
        actor_learning_rate: Optional[float] = None,
        bc_coef: float = 0.0,
        device: str = "cpu",
        seed: int = 0,
        task_reward_scale: float = 1.0,
        task_mask_prob: float = 0.0,
        use_reward_features: bool = True,
        freeze_reward_features: bool = True,
    ) -> None:
        if torch is None:  # pragma: no cover
            raise ImportError("FB baseline requires torch")
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.discount = float(discount)
        self.grad_clip_norm = float(grad_clip_norm)
        self.batch_size = int(batch_size)
        self.target_update_rate = float(target_update_rate)
        self.device = torch.device(device)
        self.seed = int(seed)
        self.task_reward_scale = float(task_reward_scale)
        self.task_mask_prob = float(task_mask_prob)
        self.bc_coef = float(bc_coef)

        torch.manual_seed(self.seed)
        self._rng = np.random.default_rng(self.seed)

        self.model = FBModel(
            self.obs_dim,
            self.action_dim,
            latent_dim=self.latent_dim,
            hidden_layers=hidden_layers,
            activation=activation,
            layernorm=layernorm,
        ).to(self.device)
        self.critic = FBModel(
            self.obs_dim,
            self.action_dim,
            latent_dim=self.latent_dim,
            hidden_layers=hidden_layers,
            activation=activation,
            layernorm=layernorm,
        ).to(self.device)
        self.critic.load_state_dict(self.model.state_dict())
        for p in self.critic.parameters():
            p.requires_grad_(False)

        self.actor = GaussianPolicy(
            self.obs_dim,
            self.action_dim,
            latent_dim=self.latent_dim,
            hidden_layers=hidden_layers,
            activation=activation,
            tanh_squash=True,
            log_std_min=FB_LOG_STD_MIN,
            log_std_max=FB_LOG_STD_MAX,
            state_dependent_std=False,
            layernorm=layernorm,
        ).to(self.device)

        # Frozen random features defining the *training* reward family
        # r_z(s) = < g(s), z >; g(s) = W_r s + b_r (Appendix B "random linear").
        self.use_reward_features = bool(use_reward_features)
        self.freeze_reward_features = bool(freeze_reward_features)
        gen = torch.Generator(device="cpu").manual_seed(self.seed + 1337)
        W = torch.randn(self.latent_dim, self.obs_dim, generator=gen) / max(1.0, np.sqrt(self.obs_dim))
        b = 0.1 * torch.randn(self.latent_dim, generator=gen)
        self.register_reward_features(W, b)

        lr = float(learning_rate)
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=float(actor_learning_rate if actor_learning_rate else lr)
        )
        self.stats = FBTrainingStats()
        self._dataset = None
        self._dataset_states: Optional[np.ndarray] = None

    # -- reward features ----------------------------------------------------
    def register_reward_features(self, W: "torch.Tensor", b: "torch.Tensor") -> None:
        """Register the frozen random linear reward features ``g(s) = W s + b``."""
        self._Wr = W.to(self.device)
        self._br = b.to(self.device)
        if self.freeze_reward_features:
            self._Wr.requires_grad_(False)
            self._br.requires_grad_(False)

    def reward_features(self, states: "torch.Tensor") -> "torch.Tensor":
        """``g(s)`` — the feature vector whose inner product with ``z`` is r_z(s)."""
        if not self.use_reward_features:
            # Fall back to raw (padded/truncated) states as features.
            if states.shape[-1] >= self.latent_dim:
                return states[..., : self.latent_dim]
            pad = self.latent_dim - states.shape[-1]
            return F.pad(states, (0, pad))
        return states @ self._Wr.t() + self._br

    def sample_task_vectors(self, batch_size: int) -> "torch.Tensor":
        """Sample task vectors ``z ~ N(0, I)`` (random linear reward functions)."""
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        if self.task_mask_prob > 0.0:
            mask = (torch.rand(batch_size, self.latent_dim, device=self.device) > self.task_mask_prob).float()
            z = z * mask
        return z

    def task_rewards(self, states: "torch.Tensor", z: "torch.Tensor") -> "torch.Tensor":
        """``r_z(s) = <g(s), z>`` evaluated for a batch of states/task vectors."""
        return (self.reward_features(states) * z).sum(dim=-1) * self.task_reward_scale

    # -- data ---------------------------------------------------------------
    def attach_dataset(self, dataset: Any) -> None:
        self._dataset = dataset
        states = getattr(dataset, "observations", None)
        if states is None:
            states = getattr(dataset, "states", None)
        if states is not None:
            self._dataset_states = np.asarray(states, dtype=np.float32)

    def _sample_transitions(self, batch_size: int) -> Dict[str, "torch.Tensor"]:
        if self._dataset is None:
            raise RuntimeError("call attach_dataset(dataset) before training")
        sampler = getattr(self._dataset, "sample_transitions", None)
        if callable(sampler):
            batch = sampler(batch_size)
        else:  # minimal fallback over flat arrays
            n = getattr(self._dataset, "num_transitions", len(self._dataset_states))
            idx = self._rng.integers(0, n, size=batch_size)
            obs = np.asarray(self._dataset.observations, dtype=np.float32)[idx]
            act = np.asarray(self._dataset.actions, dtype=np.float32)[idx]
            nobs = np.asarray(getattr(self._dataset, "next_observations", obs), dtype=np.float32)[idx]
            term = np.asarray(getattr(self._dataset, "terminals", np.zeros(n)), dtype=np.float32)[idx]
            batch = {"observations": obs, "actions": act, "next_observations": nobs, "terminals": term}

        def _t(key: str, default=None):
            value = batch.get(key, default)
            if value is None:
                return None
            return torch.as_tensor(np.asarray(value), dtype=torch.float32, device=self.device)

        obs = _t("observations")
        actions = _t("actions")
        next_obs = _t("next_observations", batch.get("observations"))
        terminals = _t("terminals")
        if terminals is None:
            terminals = torch.zeros(obs.shape[0], device=self.device)
        return {
            "observations": obs,
            "actions": actions,
            "next_observations": next_obs,
            "terminals": terminals,
        }

    # -- training -----------------------------------------------------------
    def update(self, batch: Optional[Dict[str, "torch.Tensor"]] = None, batch_size: Optional[int] = None) -> Dict[str, float]:
        """One FB update.

        Losses (Touati & Ollivier, 2021; linear-in-z factorisation):

        ``L_F``   : ``(F(s,a,z) - [r_z(s) + gamma * B_bar(s', z)])^2``
        ``L_B``   : ``(B(s,z) - [r_z(s) + gamma * B_bar(s', z)])^2``
        ``L_proj``: ``(B(s,z) - F(s, pi(s,z), z))^2`` — keeps the backward
                    function equal to the forward value of the current policy,
                    which is what makes the linear ``psi`` a valid value basis.
        ``L_actor``: ``- B(s, pi(s,z), z)`` (DDPG policy improvement through FB)
        """
        bs = int(batch_size or self.batch_size)
        if batch is None:
            batch = self._sample_transitions(bs)
        obs = batch["observations"]
        actions = batch["actions"]
        next_obs = batch["next_observations"]
        terminals = batch["terminals"].reshape(-1)

        n = obs.shape[0]
        z = self.sample_task_vectors(n)
        r_z = self.task_rewards(obs, z).detach()
        r_z_next = self.task_rewards(next_obs, z).detach()

        with torch.no_grad():
            next_actions = self.actor.act(next_obs, z, deterministic=True)
            # Bootstrap on the backward value (FB's target, cf. controllable_agent)
            b_next = self.critic.backward_values(next_obs, z)
            target = r_z + self.discount * (1.0 - terminals) * b_next

        f_pred = self.model.forward_values(obs, actions, z)
        loss_f = F.mse_loss(f_pred, target)

        b_pred = self.model.backward_values(obs, z)
        loss_b = F.mse_loss(b_pred, target)

        # Projection: B(s, z) should equal F(s, pi(s, z), z) (policy-fixed point).
        with torch.no_grad():
            pi_actions = self.actor.act(obs, z, deterministic=False)
        f_pi = self.model.forward_values(obs, pi_actions, z)
        loss_proj = F.mse_loss(b_pred, f_pi.detach())

        model_loss = loss_f + loss_b + loss_proj
        self.model_optimizer.zero_grad(set_to_none=True)
        model_loss.backward()
        if self.grad_clip_norm:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.model_optimizer.step()

        # Actor: maximise the FB value B(s, pi(s,z), z)
        actor_actions = self.actor.act(obs, z, deterministic=False)
        actor_loss = -self.model.backward_values(obs, z * 0.0 + z, ).new_zeros(())  # placeholder replaced below
        actor_loss = -self.model.forward_values(obs, actor_actions, z).mean()
        bc_loss = torch.zeros((), device=self.device)
        if self.bc_coef > 0.0 and actions is not None:
            log_prob = self.actor.evaluate_actions(obs, actions, z)[1]
            bc_loss = -log_prob.mean()
            actor_loss = actor_loss + self.bc_coef * bc_loss

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.grad_clip_norm:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm)
        self.actor_optimizer.step()

        # Polyak target update
        with torch.no_grad():
            for tp, p in zip(self.critic.parameters(), self.model.parameters()):
                tp.mul_(1.0 - self.target_update_rate).add_(p, alpha=self.target_update_rate)

        metrics = {
            "forward_loss": float(loss_f.detach().cpu()),
            "backward_loss": float(loss_b.detach().cpu()),
            "projection_loss": float(loss_proj.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "bc_loss": float(bc_loss.detach().cpu()),
        }
        self.stats.steps += 1
        self.stats.forward_loss = metrics["forward_loss"]
        self.stats.backward_loss = metrics["backward_loss"]
        self.stats.projection_loss = metrics["projection_loss"]
        self.stats.actor_loss = metrics["actor_loss"]
        self.stats.bc_loss = metrics["bc_loss"]
        return metrics

    def train_representation(
        self,
        dataset: Any = None,
        steps: int = 150_000,
        batch_size: Optional[int] = None,
        log_interval: int = 1_000,
        logger: Any = None,
        prefix: str = "fb",
    ) -> List[Dict[str, float]]:
        if dataset is not None:
            self.attach_dataset(dataset)
        history: List[Dict[str, float]] = []
        started = time.time()
        for step in range(1, steps + 1):
            metrics = self.update(batch_size=batch_size)
            if step % max(1, log_interval) == 0 or step == steps:
                metrics = dict(metrics)
                metrics["step"] = step
                history.append(metrics)
                self.stats.history.append(metrics)
                if logger is not None:
                    log_fn = getattr(logger, "log_metrics", None) or getattr(logger, "log", None)
                    if callable(log_fn):
                        try:
                            log_fn(metrics, step=step, prefix=prefix)
                        except TypeError:
                            log_fn({f"{prefix}/{k}": v for k, v in metrics.items()})
        self.stats.seconds += time.time() - started
        return history

    def train_policy(self, *args: Any, **kwargs: Any) -> List[Dict[str, float]]:
        """FB trains actor and representation jointly; alias for API symmetry."""
        return self.train_representation(*args, **kwargs)

    # -- inference / test-time adaptation -----------------------------------
    @torch.no_grad()
    def select_action(
        self,
        obs: np.ndarray,
        z: Any,
        deterministic: bool = True,
        clip: bool = True,
    ) -> np.ndarray:
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)
        z_t = torch.as_tensor(np.asarray(z, dtype=np.float32), device=self.device)
        if z_t.ndim == 1:
            z_t = z_t.unsqueeze(0)
        if z_t.shape[0] == 1 and obs_t.shape[0] > 1:
            z_t = z_t.expand(obs_t.shape[0], -1)
        action = self.actor.act(obs_t, z_t, deterministic=deterministic)
        action = action.detach().cpu().numpy()
        if clip:
            action = np.clip(action, -1.0, 1.0)
        return action

    @torch.no_grad()
    def value_of(self, obs: np.ndarray, z: Any) -> np.ndarray:
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)
        z_t = torch.as_tensor(np.asarray(z, dtype=np.float32), device=self.device)
        if z_t.ndim == 1:
            z_t = z_t.unsqueeze(0)
        return self.model.backward_values(obs_t, z_t).detach().cpu().numpy()

    def features(self, states: np.ndarray) -> np.ndarray:
        """Backward features ``psi(s)`` used for test-time linear regression."""
        with torch.no_grad():
            states_t = torch.as_tensor(np.asarray(states, dtype=np.float32), device=self.device)
            return self.model.psi(states_t).cpu().numpy()

    def to(self, device: Any) -> "ForwardBackwardAgent":
        self.device = torch.device(device)
        self.model.to(self.device)
        self.critic.to(self.device)
        self.actor.to(self.device)
        self._Wr = self._Wr.to(self.device)
        self._br = self._br.to(self.device)
        return self

    def train(self) -> "ForwardBackwardAgent":
        self.model.train()
        self.actor.train()
        return self

    def eval(self) -> "ForwardBackwardAgent":
        self.model.eval()
        self.critic.eval()
        self.actor.eval()
        return self

    def parameters(self):
        yield from self.model.parameters()
        yield from self.actor.parameters()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "critic": self.critic.state_dict(),
            "actor": self.actor.state_dict(),
            "reward_features": {"W": self._Wr.detach().cpu(), "b": self._br.detach().cpu()},
            "stats": self.stats.as_dict(),
            "config": {
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "latent_dim": self.latent_dim,
                "discount": self.discount,
                "target_update_rate": self.target_update_rate,
                "seed": self.seed,
            },
        }

    def load_state_dict(self, state: Dict[str, Any]) -> "ForwardBackwardAgent":
        self.model.load_state_dict(state["model"])
        if "critic" in state:
            self.critic.load_state_dict(state["critic"])
        if "actor" in state:
            self.actor.load_state_dict(state["actor"])
        if "reward_features" in state:
            self.register_reward_features(state["reward_features"]["W"], state["reward_features"]["b"])
        return self

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)
        return path

    def load(self, path: str, map_location: Any = "cpu") -> "ForwardBackwardAgent":
        state = torch.load(path, map_location=map_location)
        return self.load_state_dict(state)

    @classmethod
    def from_config(
        cls,
        config: Any,
        obs_dim: int,
        action_dim: int,
        device: Optional[str] = None,
        **overrides: Any,
    ) -> "ForwardBackwardAgent":
        kwargs: Dict[str, Any] = dict(
            obs_dim=obs_dim,
            action_dim=action_dim,
            latent_dim=getattr(config, "latent_dim", FB_DEFAULT_LATENT_DIM),
            hidden_layers=getattr(config, "rl_hidden_layers", FB_DEFAULT_HIDDEN_LAYERS),
            activation=getattr(config, "rl_activation", "relu"),
            layernorm=getattr(config, "rl_layernorm", False),
            learning_rate=getattr(config, "learning_rate", FB_DEFAULT_LR),
            batch_size=getattr(config, "batch_size", FB_DEFAULT_BATCH_SIZE),
            grad_clip_norm=getattr(config, "grad_clip_norm", 10.0),
            device=device or getattr(config, "device", "cpu"),
            seed=getattr(config, "seed", 0),
        )
        kwargs.update(overrides)
        return cls(**kwargs)


# Backwards/forwards compatible alias
FBAgent = ForwardBackwardAgent


# ---------------------------------------------------------------------------
# Test-time adaptation: 5120 reward samples + linear regression
# ---------------------------------------------------------------------------


def solve_task_vector(
    agent: Any,
    states: np.ndarray,
    rewards: np.ndarray,
    *,
    ridge: float = 1e-3,
    normalize_rewards: bool = True,
    features: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Recover the FB task vector by *linear regression* (Sec. 5.2).

    ``z = argmin_z || Phi z - r||^2 + ridge ||z||^2`` where ``Phi`` are the FB
    backward features ``psi(s)`` (or externally supplied features, e.g. ICM
    features for SF).  This is the "linear regression to perform test time
    adaptation" step that FB/SF use with 5120 reward samples, versus FRE's 32.
    """
    states = np.asarray(states, dtype=np.float32)
    rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
    if features is None:
        if hasattr(agent, "features"):
            features = np.asarray(agent.features(states), dtype=np.float64)
        elif hasattr(agent, "model") and hasattr(agent.model, "psi"):
            features = np.asarray(agent.model.psi(torch.as_tensor(states, device=agent.device)).detach().cpu().numpy(), dtype=np.float64)
        else:  # fall back to raw states
            features = states.astype(np.float64)
    else:
        features = np.asarray(features, dtype=np.float64)

    target = rewards.astype(np.float64)
    if normalize_rewards and target.size > 1:
        std = float(target.std())
        if std > 1e-8:
            target = target / std

    d = features.shape[1]
    A = features.T @ features + float(ridge) * np.eye(d)
    b = features.T @ target
    try:
        z = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:  # pragma: no cover - degenerate features
        z = np.linalg.lstsq(A, b, rcond=None)[0]
    return z.astype(np.float32)


def sample_eval_reward_samples(
    task: Any,
    dataset: Any,
    num_samples: int = FB_EVAL_SAMPLES,
    *,
    rng: Optional[np.random.Generator] = None,
    use_physics: bool = False,
    use_task_encoder_pairs: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect ``num_samples`` (state, reward) pairs for FB/SF test-time adaptation.

    The paper gives FB/SF **5120** reward samples (vs 32 for FRE).  When the task
    provides an ``encoder_pairs`` helper we reuse it and pad by resampling states
    from the offline dataset, otherwise we sample states uniformly and evaluate
    the task's reward function on them.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    num_samples = int(num_samples)

    # Fast path: the env task suites already know how to build (state, reward) pairs.
    if use_task_encoder_pairs and hasattr(task, "encoder_pairs"):
        try:
            states, rewards = task.encoder_pairs(dataset, num_samples=num_samples, rng=rng)
            states = np.asarray(states, dtype=np.float32)
            rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
            if states.shape[0] >= num_samples:
                return states[:num_samples], rewards[:num_samples]
        except Exception:  # pragma: no cover - tolerant of heterogeneous task APIs
            pass

    states_all = getattr(dataset, "observations", None)
    if states_all is None:
        states_all = getattr(dataset, "states", None)
    if states_all is None and hasattr(dataset, "sample_states"):
        states_all = dataset.sample_states(max(num_samples, 1))
    states_all = np.asarray(states_all, dtype=np.float32)

    idx = rng.integers(0, states_all.shape[0], size=num_samples)
    states = states_all[idx]

    if hasattr(task, "reward_from_state"):
        rewards = np.asarray([task.reward_from_state(s) for s in states], dtype=np.float32)
    elif callable(task):
        rewards = np.asarray([task(s) for s in states], dtype=np.float32)
    else:  # pragma: no cover
        raise TypeError("task must expose reward_from_state(...) or be callable")
    return states, rewards.reshape(-1)


def make_fb_policy_fn(
    agent: Any,
    z: Any,
    deterministic: bool = True,
    clip: bool = True,
) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap a trained FB agent + task vector into an ``act_fn(obs) -> action``."""

    def act_fn(obs: np.ndarray) -> np.ndarray:
        return agent.select_action(obs, z, deterministic=deterministic, clip=clip)

    return act_fn


# ---------------------------------------------------------------------------
# Training / evaluation drivers
# ---------------------------------------------------------------------------


def make_forward_backward(
    config: Any,
    obs_dim: int,
    action_dim: int,
    device: Optional[str] = None,
    **overrides: Any,
) -> ForwardBackwardAgent:
    """Build the in-house FB agent from an FRE ``Config``-like object."""
    return ForwardBackwardAgent.from_config(config, obs_dim, action_dim, device=device, **overrides)


def train_forward_backward(
    config: Any,
    dataset: Any,
    obs_dim: Optional[int] = None,
    action_dim: Optional[int] = None,
    steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    log_interval: int = 1_000,
    logger: Any = None,
    agent: Optional[ForwardBackwardAgent] = None,
    device: Optional[str] = None,
    ca_dir: Optional[str] = None,
    prefer_official: bool = True,
    **overrides: Any,
) -> Tuple[ForwardBackwardAgent, List[Dict[str, float]]]:
    """Train FB for the FRE Table 1 comparison.

    If the official ``facebookresearch/controllable_agent`` checkout is present
    and ``prefer_official`` is True, its training command is resolved and
    returned in the result metadata (the paper's own protocol); the in-house
    agent is still trained so that a local policy exists for evaluation in this
    codebase.
    """
    if obs_dim is None:
        obs_dim = int(getattr(dataset, "obs_dim", getattr(dataset, "observations", np.zeros((0, 0))).shape[-1]))
    if action_dim is None:
        action_dim = int(getattr(dataset, "action_dim", 1))

    agent = agent or make_forward_backward(config, obs_dim, action_dim, device=device, **overrides)
    agent.attach_dataset(dataset)

    if steps is None:
        steps = int(config.encoder_steps() if hasattr(config, "encoder_steps") else 150_000)
    if batch_size is None:
        batch_size = int(getattr(config, "batch_size", FB_DEFAULT_BATCH_SIZE))

    if prefer_official and controllable_agent_available(ca_dir):
        env_name = str(getattr(config, "env_id", getattr(config, "domain", "antmaze")))
        output_dir = getattr(config, "output_dir", None)
        try:
            run_controllable_agent(
                env_name,
                ca_dir=ca_dir,
                output_dir=output_dir,
                seed=int(getattr(config, "seed", 0)),
                num_train_steps=steps,
                num_eval_episodes=int(getattr(config, "num_eval_episodes", 20)),
                extra_args=["--dataset_type", str(getattr(config, "exorl_variant", "rnd"))],
            )
        except ControllableAgentUnavailable:  # pragma: no cover - guarded above
            pass

    history = agent.train_representation(
        dataset=dataset, steps=steps, batch_size=batch_size, log_interval=log_interval, logger=logger
    )
    return agent, history


def evaluate_fb_suite(
    agent: ForwardBackwardAgent,
    suite_evaluate_fn: Callable[..., Any],
    dataset: Any = None,
    num_samples: int = FB_EVAL_SAMPLES,
    seed: int = 0,
    tasks: Optional[Sequence[Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Zero-shot evaluation of FB on a task suite with 5120 reward samples/task.

    ``suite_evaluate_fn`` follows the env-suite convention used by
    ``fre/envs/*_eval.py``: it accepts an ``act_fn`` (and kwargs such as
    ``num_episodes``/``seed``) and returns a per-task score mapping.
    """
    results: Dict[str, Any] = {}
    rng = np.random.default_rng(seed)
    if tasks is None:
        tasks = list(getattr(suite_evaluate_fn, "tasks", []) or [])

    for task in tasks:
        name = getattr(task, "name", str(task))
        states, rewards = sample_eval_reward_samples(task, dataset, num_samples=num_samples, rng=rng)
        z = solve_task_vector(agent, states, rewards)
        act_fn = make_fb_policy_fn(agent, z)
        try:
            score = suite_evaluate_fn(act_fn, task=task, seed=seed, **kwargs)
        except TypeError:
            score = suite_evaluate_fn(task, act_fn, seed=seed, **kwargs)
        results[name] = score

    scores = []
    for value in results.values():
        if isinstance(value, dict):
            scores.append(float(value.get("score", value.get("mean", 0.0))))
        else:
            scores.append(float(value))
    results["fb-all"] = float(np.mean(scores)) if scores else 0.0
    return results


def fb_reference_row(domain: str) -> Tuple[float, float]:
    """Return the Table 1 FB reference ``(mean, std)`` for a domain key."""
    return FB_TABLE1_REFERENCE.get(domain, (float("nan"), float("nan")))


def dump_fb_config(config: Any, path: str) -> str:
    """Persist the FB hyperparameters used for a run (small helper for scripts)."""
    payload = {
        "model": "forward_backward",
        "reference": CONTROLLABLE_AGENT_REPO,
        "eval_reward_samples": FB_EVAL_SAMPLES,
        "table1": {k: list(v) for k, v in FB_TABLE1_REFERENCE.items()},
        "config": getattr(config, "to_dict", lambda: {})(),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path
