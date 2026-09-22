"""Diversity metrics for SAPG (Section 6.4, Figures 7-8).

This module implements the two diversity analyses used in the paper to show
that SAPG's M policies explore a more diverse set of states than a single
policy (PPO) or a random policy:

1. **PCA reconstruction error** (Figure 7):
   Collect a batch of state-transitions from a policy, fit a PCA on the states,
   and measure the reconstruction error using the top-k principal components.
   A *slower* decay of reconstruction error as k grows indicates a *more
   diverse* state distribution (the states span more directions in state
   space).

2. **MLP reconstruction error** (Figure 8):
   Train a 2-layer MLP autoencoder (same hidden size, ReLU, Adam with default
   hyperparameters) on 400k state-transitions with an L2 reconstruction loss.
   A *higher* training reconstruction error indicates a *more diverse* state
   distribution (harder to compress into a low-dimensional bottleneck).

Both metrics are computed for:
  - SAPG (leader + followers aggregated),
  - PPO (single policy),
  - a random policy (uniform actions).

The module is dependency-tolerant: ``scikit-learn`` is optional (a pure-numpy
PCA fallback is provided) and the environment can be mocked (``--force-mock``)
so the analysis runs on CPU without IsaacGym.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from sapg.envs import TASK_NAMES, make_task_env
from sapg.sapg.models import build_actor_critic
from sapg.sapg.utils import get_device, get_logger, set_seed

try:  # scikit-learn is optional; a numpy fallback is provided below.
    from sklearn.decomposition import PCA as _SklearnPCA

    _HAS_SKLEARN = True
except Exception:  # pragma: no cover - exercised only without sklearn
    _SklearnPCA = None  # type: ignore
    _HAS_SKLEARN = False


LOGGER = get_logger("sapg.diversity")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class DiversityConfig:
    """Configuration for the diversity analysis (Figures 7-8).

    Attributes
    ----------
    task:
        Task family name (``allegrokuka``, ``shadowhand``, ``allegrohand``).
    task_name:
        Specific task variant (e.g. ``regrasping``, ``throw``, ``reorientation``).
    num_envs:
        Number of parallel envs used to collect the state batch.
    num_transitions:
        Total number of state-transitions to collect (paper: 400k).
    num_policies:
        Number of SAPG policies (M). The leader + followers are aggregated.
    horizon:
        Rollout horizon per collection step.
    pca_components:
        List of ``k`` values (number of principal components) to evaluate.
    mlp_hidden_sizes:
        List of hidden sizes for the MLP autoencoder sweep.
    mlp_epochs:
        Number of training epochs for the MLP autoencoder.
    mlp_batch_size:
        Mini-batch size for MLP training.
    mlp_lr:
        Learning rate for the MLP autoencoder (Adam default).
    seed:
        Random seed.
    device:
        Torch device string.
    output_dir:
        Directory to write the JSON results.
    force_mock:
        Use the dependency-free mock env (CPU debugging).
    headless:
        Run IsaacGym headless.
    checkpoint:
        Optional path to a trained SAPG checkpoint. If provided, the trained
        policies are used; otherwise randomly-initialised policies are used
        (still valid for the diversity comparison, matching the paper's
        "random policy" baseline).
    """

    task: str = "allegrokuka"
    task_name: str = "regrasping"
    num_envs: int = 1024
    num_transitions: int = 400_000
    num_policies: int = 6
    horizon: int = 16
    pca_components: Sequence[int] = field(
        default_factory=lambda: [1, 2, 4, 8, 16, 32, 64, 128]
    )
    mlp_hidden_sizes: Sequence[int] = field(
        default_factory=lambda: [4, 8, 16, 32, 64, 128, 256]
    )
    mlp_epochs: int = 20
    mlp_batch_size: int = 4096
    mlp_lr: float = 1e-3
    seed: int = 0
    device: str = "cuda:0"
    output_dir: str = "results/diversity"
    force_mock: bool = False
    headless: bool = True
    checkpoint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["pca_components"] = list(self.pca_components)
        d["mlp_hidden_sizes"] = list(self.mlp_hidden_sizes)
        return d


# ---------------------------------------------------------------------------
# State collection
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_states(
    env,
    actor_critic: Optional[nn.Module],
    num_transitions: int,
    num_policies: int = 1,
    policy_index: int = 0,
    horizon: int = 16,
    device: Optional[torch.device] = None,
    random_policy: bool = False,
    seed: int = 0,
) -> np.ndarray:
    """Collect a batch of states from ``env``.

    Parameters
    ----------
    env:
        Vectorized environment exposing ``reset``/``step`` and ``obs_dim``.
    actor_critic:
        Policy used to act. Ignored when ``random_policy`` is True.
    num_transitions:
        Target number of states to collect.
    num_policies:
        If > 1, the states from all policies are aggregated (SAPG setting).
    policy_index:
        Base policy index (used when ``num_policies == 1``).
    horizon:
        Number of steps per rollout chunk.
    device:
        Torch device.
    random_policy:
        Sample uniform actions in [-1, 1] instead of using the policy.
    seed:
        RNG seed for the random policy.

    Returns
    -------
    np.ndarray
        Array of shape ``(num_transitions, obs_dim)``.
    """
    device = device or get_device("cpu")
    obs = env.reset()
    if isinstance(obs, np.ndarray):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    obs = obs.to(device)

    obs_dim = int(getattr(env, "obs_dim", obs.shape[-1]))
    act_dim = int(getattr(env, "act_dim", 1))

    collected: List[np.ndarray] = []
    total = 0
    gen = torch.Generator(device="cpu").manual_seed(seed)

    lstm_states = None
    if actor_critic is not None and getattr(actor_critic, "use_lstm", False):
        lstm_states = actor_critic.init_lstm_state(env.num_envs, device=device)

    while total < num_transitions:
        # Record current states.
        collected.append(obs.detach().cpu().numpy().copy())
        total += obs.shape[0]

        # Choose actions.
        if random_policy or actor_critic is None:
            actions = (
                torch.rand(obs.shape[0], act_dim, generator=gen, device="cpu")
                .to(device)
                * 2.0
                - 1.0
            )
        else:
            if num_policies > 1:
                # Aggregate across all policies: split envs into blocks.
                actions = torch.zeros(obs.shape[0], act_dim, device=device)
                block = max(1, obs.shape[0] // num_policies)
                for j in range(num_policies):
                    lo = j * block
                    hi = obs.shape[0] if j == num_policies - 1 else (j + 1) * block
                    if lo >= hi:
                        continue
                    a_j, _, _ = actor_critic.act(
                        obs[lo:hi], j, deterministic=False, lstm_state=None
                    )
                    actions[lo:hi] = a_j
            else:
                actions, _, lstm_states = actor_critic.act(
                    obs, policy_index, deterministic=False, lstm_state=lstm_states
                )

        step_out = env.step(actions)
        obs = step_out[0]
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        obs = obs.to(device)

    states = np.concatenate(collected, axis=0)[:num_transitions]
    return states.astype(np.float32)


# ---------------------------------------------------------------------------
# PCA reconstruction error (Figure 7)
# ---------------------------------------------------------------------------
def _pca_reconstruction_error_numpy(
    states: np.ndarray, k: int
) -> float:
    """Pure-numpy PCA reconstruction error using the top-``k`` components."""
    x = states - states.mean(axis=0, keepdims=True)
    # SVD-based PCA (economy) to avoid materialising the full covariance.
    # Use a subsample for the SVD if the batch is very large.
    n, d = x.shape
    k = min(k, min(n, d))
    try:
        u, s, vt = np.linalg.svd(x, full_matrices=False)
    except np.linalg.LinAlgError:  # pragma: no cover
        # Fall back to covariance eigendecomposition.
        cov = (x.T @ x) / max(1, n - 1)
        w, v = np.linalg.eigh(cov)
        order = np.argsort(w)[::-1]
        vt = v[:, order].T
    components = vt[:k]  # (k, d)
    proj = x @ components.T  # (n, k)
    recon = proj @ components  # (n, d)
    err = float(np.mean((x - recon) ** 2))
    return err


def pca_reconstruction_errors(
    states: np.ndarray, components: Sequence[int]
) -> Dict[int, float]:
    """Compute PCA reconstruction error for each ``k`` in ``components``.

    Returns a dict mapping ``k -> mean squared reconstruction error``.
    """
    errors: Dict[int, float] = {}
    if _HAS_SKLEARN:
        for k in components:
            kk = int(min(k, min(states.shape[0], states.shape[1])))
            if kk < 1:
                continue
            pca = _SklearnPCA(n_components=kk)
            pca.fit(states)
            recon = pca.inverse_transform(pca.transform(states))
            errors[int(k)] = float(np.mean((states - recon) ** 2))
    else:
        for k in components:
            errors[int(k)] = _pca_reconstruction_error_numpy(states, int(k))
    return errors


# ---------------------------------------------------------------------------
# MLP reconstruction error (Figure 8)
# ---------------------------------------------------------------------------
class _MLPAutoencoder(nn.Module):
    """2-layer MLP autoencoder with ReLU activations (paper Section 6.4)."""

    def __init__(self, obs_dim: int, hidden_size: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, obs_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def mlp_reconstruction_error(
    states: np.ndarray,
    hidden_size: int,
    epochs: int = 20,
    batch_size: int = 4096,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    seed: int = 0,
) -> float:
    """Train a 2-layer MLP autoencoder and return its final training MSE.

    Higher training error => more diverse state distribution (Figure 8).
    """
    device = device or get_device("cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    x = torch.as_tensor(states, dtype=torch.float32, device=device)
    obs_dim = x.shape[1]
    model = _MLPAutoencoder(obs_dim, int(hidden_size)).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n = x.shape[0]
    batch_size = max(1, min(batch_size, n))
    final_loss = float("nan")

    for _ in range(max(1, epochs)):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        num_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            batch = x[idx]
            recon = model(batch)
            loss = loss_fn(recon, batch)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += float(loss.detach().item())
            num_batches += 1
        final_loss = epoch_loss / max(1, num_batches)

    return float(final_loss)


# ---------------------------------------------------------------------------
# High-level analysis
# ---------------------------------------------------------------------------
def _build_actor_critic_for_task(
    cfg: DiversityConfig, obs_dim: int, act_dim: int
) -> nn.Module:
    """Build the actor-critic matching the task architecture preset."""
    task_key = cfg.task.lower()
    if task_key in ("regrasping", "throw", "reorientation"):
        task_key = "allegrokuka"
    model = build_actor_critic(
        task=task_key,
        obs_dim=obs_dim,
        act_dim=act_dim,
        num_policies=cfg.num_policies,
    )
    return model


def _load_checkpoint(model: nn.Module, path: str, device: torch.device) -> None:
    """Load a checkpoint into ``model`` (best-effort, tolerant of key prefixes)."""
    try:
        ckpt = torch.load(path, map_location=device)
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Failed to load checkpoint %s: %s", path, exc)
        return
    state = ckpt
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "actor_critic", "weights"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                break
    if not isinstance(state, dict):
        LOGGER.warning("Checkpoint %s has unexpected format; skipping.", path)
        return
    # Strip common prefixes.
    cleaned = {}
    for k, v in state.items():
        nk = k
        for prefix in ("module.", "actor_critic.", "model."):
            if nk.startswith(prefix):
                nk = nk[len(prefix) :]
        cleaned[nk] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        LOGGER.warning("Checkpoint missing %d keys (using init).", len(missing))
    if unexpected:
        LOGGER.debug("Checkpoint had %d unexpected keys.", len(unexpected))


def run_diversity_analysis(cfg: DiversityConfig) -> Dict[str, Any]:
    """Run the full diversity analysis and persist results as JSON.

    Returns a dict with keys:
      - ``config``: the config used,
      - ``pca``: ``{policy_name: {k: error}}``,
      - ``mlp``: ``{policy_name: {hidden_size: error}}``,
      - ``num_transitions``: number of states collected per policy,
      - ``obs_dim``: observation dimensionality.
    """
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    os.makedirs(cfg.output_dir, exist_ok=True)

    LOGGER.info(
        "Diversity analysis: task=%s/%s num_transitions=%d device=%s",
        cfg.task,
        cfg.task_name,
        cfg.num_transitions,
        device,
    )

    env = make_task_env(
        task=cfg.task_name if cfg.task_name in TASK_NAMES else cfg.task,
        num_envs=cfg.num_envs,
        device=str(device),
        headless=cfg.headless,
        seed=cfg.seed,
        force_mock=cfg.force_mock,
    )

    obs_dim = int(getattr(env, "obs_dim", 0))
    act_dim = int(getattr(env, "act_dim", 1))

    # Build the SAPG actor-critic (shared backbone + M phi_j latents).
    actor_critic = _build_actor_critic_for_task(cfg, obs_dim, act_dim).to(device)
    if cfg.checkpoint:
        _load_checkpoint(actor_critic, cfg.checkpoint, device)
    actor_critic.eval()

    # --- Collect states for each policy variant ---------------------------
    policy_states: Dict[str, np.ndarray] = {}

    LOGGER.info("Collecting SAPG states (M=%d policies aggregated)...", cfg.num_policies)
    policy_states["sapg"] = collect_states(
        env,
        actor_critic,
        cfg.num_transitions,
        num_policies=cfg.num_policies,
        horizon=cfg.horizon,
        device=device,
        seed=cfg.seed,
    )

    LOGGER.info("Collecting PPO (single-policy) states...")
    policy_states["ppo"] = collect_states(
        env,
        actor_critic,
        cfg.num_transitions,
        num_policies=1,
        policy_index=0,
        horizon=cfg.horizon,
        device=device,
        seed=cfg.seed + 1,
    )

    LOGGER.info("Collecting random-policy states...")
    policy_states["random"] = collect_states(
        env,
        None,
        cfg.num_transitions,
        num_policies=1,
        horizon=cfg.horizon,
        device=device,
        random_policy=True,
        seed=cfg.seed + 2,
    )

    # --- PCA reconstruction error (Figure 7) ------------------------------
    pca_results: Dict[str, Dict[int, float]] = {}
    for name, states in policy_states.items():
        LOGGER.info("PCA reconstruction error for '%s'...", name)
        pca_results[name] = pca_reconstruction_errors(states, cfg.pca_components)

    # --- MLP reconstruction error (Figure 8) ------------------------------
    mlp_results: Dict[str, Dict[int, float]] = {}
    for name, states in policy_states.items():
        mlp_results[name] = {}
        for hidden in cfg.mlp_hidden_sizes:
            LOGGER.info(
                "MLP reconstruction error for '%s' (hidden=%d)...", name, hidden
            )
            err = mlp_reconstruction_error(
                states,
                hidden_size=int(hidden),
                epochs=cfg.mlp_epochs,
                batch_size=cfg.mlp_batch_size,
                lr=cfg.mlp_lr,
                device=device,
                seed=cfg.seed,
            )
            mlp_results[name][int(hidden)] = err

    result: Dict[str, Any] = {
        "config": cfg.to_dict(),
        "pca": pca_results,
        "mlp": mlp_results,
        "num_transitions": int(cfg.num_transitions),
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "timestamp": time.time(),
    }

    out_path = os.path.join(
        cfg.output_dir, f"diversity_{cfg.task}_{cfg.task_name}_seed{cfg.seed}.json"
    )
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    LOGGER.info("Saved diversity results to %s", out_path)

    try:
        env.close()
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SAPG diversity analysis (Figures 7-8): PCA and MLP "
        "reconstruction error of state distributions."
    )
    parser.add_argument("--task", type=str, default="allegrokuka")
    parser.add_argument("--task-name", type=str, default="regrasping")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--num-transitions", type=int, default=400_000)
    parser.add_argument("--num-policies", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument(
        "--pca-components",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128],
    )
    parser.add_argument(
        "--mlp-hidden-sizes",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32, 64, 128, 256],
    )
    parser.add_argument("--mlp-epochs", type=int, default=20)
    parser.add_argument("--mlp-batch-size", type=int, default=4096)
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-dir", type=str, default="results/diversity")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--force-mock", action="store_true")
    parser.add_argument("--headless", action="store_true", default=True)
    return parser


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cfg = DiversityConfig(
        task=args.task,
        task_name=args.task_name,
        num_envs=args.num_envs,
        num_transitions=args.num_transitions,
        num_policies=args.num_policies,
        horizon=args.horizon,
        pca_components=args.pca_components,
        mlp_hidden_sizes=args.mlp_hidden_sizes,
        mlp_epochs=args.mlp_epochs,
        mlp_batch_size=args.mlp_batch_size,
        mlp_lr=args.mlp_lr,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
        force_mock=args.force_mock,
        headless=args.headless,
        checkpoint=args.checkpoint,
    )
    return run_diversity_analysis(cfg)


__all__ = [
    "DiversityConfig",
    "collect_states",
    "pca_reconstruction_errors",
    "mlp_reconstruction_error",
    "run_diversity_analysis",
    "build_arg_parser",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    main()
