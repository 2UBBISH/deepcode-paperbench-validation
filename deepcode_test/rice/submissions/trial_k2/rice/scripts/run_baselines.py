#!/usr/bin/env python3
"""
Run comparison baselines for the RICE reproduction.

Supported baselines
-------------------
* ppo-finetune : continue training the frozen target policy with a lower LR.
* sil          : self-imitation learning that prioritises high-return past
                 trajectories in an on-policy-style buffer.
* jsrl         : jump-start RL by initialising refinement episodes from
                 high-return target-policy trajectories.
* statemask-r  : proxy for StateMask-R: train a mask network with the same
                 architecture as the target policy and refine from its critical
                 states (uses the RICE refining pipeline).
* random-exp   : fixed refining pipeline + Random explanation critical states.
* ig-exp       : fixed refining pipeline + Integrated Gradients critical states.
* airs-exp     : fixed refining pipeline + AIRS stub critical states.

The script reuses the RICE environment factories, target-policy loaders, and
refinement trainer so that all baselines are evaluated under the same harness.
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Allow running without installing the package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rice.agents import PPOConfig, PPOTrainer, load_target_policy
from rice.agents.target_policy import BaseTargetPolicy, MLPActorCritic, TorchTargetPolicy
from rice.envs import make_mujoco_env, make_sparse_mujoco_env
from rice.envs.cage_env import make_cage_env
from rice.envs.malware_env import make_malware_env
from rice.envs.metadrive_env import make_metadrive_env
from rice.envs.selfish_mining_env import make_selfish_mining_env
from rice.explain import RandomExplanation, IntegratedGradients, AIRSStub
from rice.masknet import MaskNetwork, MaskTrainer, build_mask_network, train_mask_network
from rice.refine import (
    CriticalStateBuffer,
    RefineTrainer,
    build_critical_buffer_from_trajectories,
    refine_policy,
)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RICE baselines")
    parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        choices=[
            "ppo-finetune",
            "sil",
            "jsrl",
            "statemask-r",
            "random-exp",
            "ig-exp",
            "airs-exp",
        ],
        help="Baseline method to run.",
    )
    parser.add_argument(
        "--target-policy",
        type=str,
        required=True,
        help="Path to the frozen target-policy checkpoint.",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default=None,
        help="Environment/task id. Inferred from metadata if omitted.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="mujoco",
        choices=["mujoco", "sparse_mujoco", "selfish_mining", "cage", "metadrive", "malware"],
        help="Task family.",
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="Path to a trained mask checkpoint (required for statemask-r).",
    )
    parser.add_argument(
        "--critical-buffer",
        type=str,
        default=None,
        help="Path to a saved critical-state buffer .npz (optional).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/baselines",
        help="Directory to save baseline checkpoints and metadata.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="Total environment steps for the baseline run.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="Rollout length per PPO update.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="PPO mini-batch size.",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="PPO epochs per update.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (default depends on baseline).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor.",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="GAE lambda.",
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=0.2,
        help="PPO clip range.",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.0,
        help="Entropy coefficient.",
    )
    parser.add_argument(
        "--vf-coef",
        type=float,
        default=0.5,
        help="Value-function coefficient.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="Gradient clipping.",
    )
    parser.add_argument(
        "--normalize-advantage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalise advantages in PPO.",
    )
    parser.add_argument(
        "--normalize-obs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Observation normalisation for MuJoCo (default True for Walker2d/HalfCheetah).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="PyTorch device.",
    )
    parser.add_argument(
        "--n-trajectories",
        type=int,
        default=100,
        help="Number of target-policy trajectories used to build critical buffers / demos.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.25,
        help="Top-p percentile for critical-state selection.",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.5,
        help="Mixed-reset probability for refining baselines.",
    )
    parser.add_argument(
        "--lambda-rnd",
        type=float,
        default=0.01,
        help="RND bonus scale for refining baselines.",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="Save intermediate checkpoints every N updates.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="Logging interval in PPO updates.",
    )
    parser.add_argument(
        "--use-sb3",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force Stable-Baselines3 backend for target policy loading.",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Optional metadata.txt path next to the target checkpoint.",
    )
    parser.add_argument(
        "--demo-dir",
        type=str,
        default=None,
        help="Directory containing demonstration trajectories for JSRL.",
    )
    parser.add_argument(
        "--sil-buffer-size",
        type=int,
        default=100_000,
        help="Max number of transitions stored by the SIL buffer.",
    )
    parser.add_argument(
        "--sil-batch-size",
        type=int,
        default=256,
        help="SIL update batch size.",
    )
    parser.add_argument(
        "--sil-epochs",
        type=int,
        default=4,
        help="SIL update epochs per PPO update.",
    )
    parser.add_argument(
        "--sil-tau",
        type=float,
        default=0.95,
        help="Minimum return quantile threshold for SIL sampling (higher = more elite).",
    )
    args = parser.parse_args()

    # Resolve backend from checkpoint extension if not explicit.
    if args.use_sb3 is None:
        args.use_sb3 = str(args.target_policy).endswith(".zip")

    # Default learning rates per baseline.
    if args.learning_rate is None:
        if args.baseline == "ppo-finetune":
            args.learning_rate = 3e-5
        else:
            args.learning_rate = 3e-4

    # Resolve observation normalisation default.
    if args.normalize_obs is None and args.env_id is not None:
        args.normalize_obs = "Walker2d" in args.env_id or "HalfCheetah" in args.env_id
    elif args.normalize_obs is None:
        args.normalize_obs = False

    return args


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _auto_detect_metadata(checkpoint_path: str, metadata_path: Optional[str]) -> Optional[Path]:
    if metadata_path is not None:
        return Path(metadata_path)
    cp = Path(checkpoint_path)
    candidate = cp.parent / "metadata.txt"
    if candidate.exists():
        return candidate
    return None


def _read_metadata(metadata_path: Optional[Path]) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    if metadata_path is None or not metadata_path.exists():
        return meta
    with open(metadata_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
    return meta


def _should_normalize_obs(env_id: str, explicit: Optional[bool]) -> bool:
    if explicit is not None:
        return explicit
    return "Walker2d" in env_id or "HalfCheetah" in env_id


def make_env(args: argparse.Namespace, seed: int) -> Any:
    """Create the task-specific training/evaluation environment."""
    env_id = args.env_id
    if args.task == "mujoco":
        if env_id is None:
            raise ValueError("--env-id is required for MuJoCo tasks")
        return make_mujoco_env(env_id, normalize_obs=args.normalize_obs, seed=seed)
    if args.task == "sparse_mujoco":
        if env_id is None:
            raise ValueError("--env-id is required for sparse MuJoCo tasks")
        return make_sparse_mujoco_env(env_id, normalize_obs=args.normalize_obs, seed=seed)
    if args.task == "selfish_mining":
        return make_selfish_mining_env(seed=seed)
    if args.task == "cage":
        return make_cage_env(seed=seed)
    if args.task == "metadrive":
        return make_metadrive_env(seed=seed)
    if args.task == "malware":
        return make_malware_env(seed=seed)
    raise ValueError(f"Unknown task: {args.task}")


def build_ppo_config(args: argparse.Namespace) -> PPOConfig:
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return PPOConfig(
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm,
        normalize_advantage=args.normalize_advantage,
        device=device,
        seed=args.seed,
    )


def save_checkpoint(
    policy: BaseTargetPolicy,
    args: argparse.Namespace,
    output_dir: Path,
    elapsed: float,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = output_dir / "policy.pt"
    policy.save(str(policy_path))
    meta = {
        "baseline": args.baseline,
        "task": args.task,
        "env_id": str(args.env_id),
        "seed": str(args.seed),
        "total_timesteps": str(args.total_timesteps),
        "learning_rate": str(args.learning_rate),
        "elapsed_sec": f"{elapsed:.2f}",
        "policy_path": str(policy_path),
    }
    if extra_meta:
        meta.update({k: str(v) for k, v in extra_meta.items()})
    with open(output_dir / "metadata.txt", "w") as f:
        for k, v in meta.items():
            f.write(f"{k}: {v}\n")
    print(f"Saved baseline checkpoint to {output_dir}")
    return meta


# --------------------------------------------------------------------------- #
# Baseline implementations
# --------------------------------------------------------------------------- #


def _warm_start_policy(target_policy: BaseTargetPolicy, device: torch.device) -> TorchTargetPolicy:
    """Return a trainable copy of the target policy for refinement/fine-tuning."""
    if isinstance(target_policy, TorchTargetPolicy):
        model = target_policy.model
        obs_space = target_policy.observation_space
        act_space = target_policy.action_space
        cloned = MLPActorCritic(
            obs_dim=int(np.prod(obs_space.shape)),
            action_dim=int(np.prod(act_space.shape)) if hasattr(act_space, "shape") else act_space.n,
            hidden_sizes=getattr(model, "hidden_sizes", (64, 64)),
            discrete=isinstance(act_space, (gym.spaces.Discrete,)),
        )
        cloned.load_state_dict(model.state_dict())
        return TorchTargetPolicy(cloned, obs_space, act_space, device=device)
    # SB3 target: fall back to a fresh MLP actor-critic with default sizes.
    obs_space = target_policy.observation_space
    act_space = target_policy.action_space
    discrete = isinstance(act_space, (gym.spaces.Discrete,))
    model = MLPActorCritic(
        obs_dim=int(np.prod(obs_space.shape)),
        action_dim=act_space.n if discrete else int(np.prod(act_space.shape)),
        hidden_sizes=(64, 64),
        discrete=discrete,
    )
    return TorchTargetPolicy(model, obs_space, act_space, device=device)


def _collect_target_trajectories(
    target_policy: BaseTargetPolicy,
    env: Any,
    n_trajectories: int,
    max_steps: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Collect target-policy trajectories for JSRL / SIL / buffer construction."""
    trajectories: List[Dict[str, Any]] = []
    for _ in range(n_trajectories):
        obs, info = env.reset(), {}
        if isinstance(obs, tuple):
            obs, info = obs
        traj = {"observations": [], "actions": [], "rewards": [], "infos": []}
        done = False
        step = 0
        while not done:
            action, _ = target_policy.predict(obs, deterministic=False)
            traj["observations"].append(np.array(obs, copy=True))
            traj["actions"].append(np.array(action, copy=True))
            result = env.step(action)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                obs, reward, done, info = result
            traj["rewards"].append(float(reward))
            traj["infos"].append(info)
            step += 1
            if max_steps is not None and step >= max_steps:
                done = True
        traj["return"] = float(np.sum(traj["rewards"]))
        trajectories.append(traj)
    return trajectories


# --------------------------------------------------------------------------- #
# PPO fine-tuning
# --------------------------------------------------------------------------- #


def run_ppo_finetune(args: argparse.Namespace, env: Any, target_policy: BaseTargetPolicy) -> Dict[str, Any]:
    """Continue training the target policy with a lower learning rate."""
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    refined_policy = _warm_start_policy(target_policy, torch.device(device))
    config = build_ppo_config(args)
    trainer = PPOTrainer(refined_policy, env, config=config)
    start = time.time()
    stats = trainer.learn(
        total_timesteps=args.total_timesteps,
        log_interval=args.log_interval,
        save_path=str(Path(args.output_dir) / "checkpoints" / "policy") if args.save_interval else None,
        save_interval=args.save_interval,
    )
    elapsed = time.time() - start
    save_checkpoint(refined_policy, args, Path(args.output_dir), elapsed, extra_meta=stats)
    return stats


# --------------------------------------------------------------------------- #
# Self-Imitation Learning (SIL)
# --------------------------------------------------------------------------- #


class SILBuffer:
    """Simple fixed-size replay buffer that stores elite transitions."""

    def __init__(self, capacity: int, obs_shape: Tuple[int, ...], action_shape: Tuple[int, ...]):
        self.capacity = capacity
        self.obs = np.zeros((capacity,) + obs_shape, dtype=np.float32)
        self.actions = np.zeros((capacity,) + action_shape, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.pos = 0

    def add(self, obs: np.ndarray, action: np.ndarray, ret: float) -> None:
        idx = self.pos % self.capacity
        self.obs[idx] = obs
        self.actions[idx] = action
        self.returns[idx] = ret
        self.pos += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, min_return: float) -> Optional[Dict[str, torch.Tensor]]:
        elite_mask = self.returns[: self.size] >= min_return
        idxs = np.where(elite_mask)[0]
        if len(idxs) == 0:
            return None
        chosen = np.random.choice(idxs, size=min(batch_size, len(idxs)), replace=False)
        return {
            "obs": torch.as_tensor(self.obs[chosen], dtype=torch.float32),
            "actions": torch.as_tensor(self.actions[chosen], dtype=torch.float32),
            "returns": torch.as_tensor(self.returns[chosen], dtype=torch.float32),
        }


class SILPPOTrainer(PPOTrainer):
    """PPO trainer augmented with a SIL update step on elite transitions."""

    def __init__(
        self,
        policy: TorchTargetPolicy,
        env: Any,
        config: PPOConfig,
        sil_buffer: SILBuffer,
        sil_batch_size: int,
        sil_epochs: int,
        sil_tau: float,
    ):
        super().__init__(policy, env, config=config)
        self.sil_buffer = sil_buffer
        self.sil_batch_size = sil_batch_size
        self.sil_epochs = sil_epochs
        self.sil_tau = sil_tau

    def update_policy(self) -> Dict[str, float]:
        stats = super().update_policy()
        # SIL update on elite transitions.
        if self.sil_buffer.size > 0:
            min_return = float(np.quantile(self.sil_buffer.returns[: self.sil_buffer.size], self.sil_tau))
            for _ in range(self.sil_epochs):
                batch = self.sil_buffer.sample(self.sil_batch_size, min_return)
                if batch is None:
                    break
                obs = batch["obs"].to(self.device)
                actions = batch["actions"].to(self.device)
                returns = batch["returns"].to(self.device)
                _, log_prob, entropy, values = self.policy.evaluate_actions(obs, actions)
                advantages = returns - values.squeeze(-1)
                policy_loss = -(log_prob * advantages.detach()).mean()
                value_loss = 0.5 * advantages.pow(2).mean()
                loss = policy_loss + self.config.vf_coef * value_loss - self.config.ent_coef * entropy.mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
        return stats


def run_sil(args: argparse.Namespace, env: Any, target_policy: BaseTargetPolicy) -> Dict[str, Any]:
    """Self-imitation learning baseline."""
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    refined_policy = _warm_start_policy(target_policy, torch.device(device))
    config = build_ppo_config(args)

    obs_space = env.observation_space
    action_space = env.action_space
    obs_shape = obs_space.shape
    action_shape = () if isinstance(action_space, gym.spaces.Discrete) else action_space.shape
    sil_buffer = SILBuffer(args.sil_buffer_size, obs_shape, action_shape)

    trainer = SILPPOTrainer(
        refined_policy,
        env,
        config,
        sil_buffer,
        args.sil_batch_size,
        args.sil_epochs,
        args.sil_tau,
    )

    # Pre-fill SIL buffer with target-policy trajectories.
    print(f"Collecting {args.n_trajectories} target-policy trajectories for SIL buffer...")
    trajectories = _collect_target_trajectories(target_policy, env, args.n_trajectories)
    for traj in trajectories:
        ret = traj["return"]
        for obs, action in zip(traj["observations"], traj["actions"]):
            sil_buffer.add(obs, action, ret)

    start = time.time()
    stats = trainer.learn(
        total_timesteps=args.total_timesteps,
        log_interval=args.log_interval,
        save_path=str(Path(args.output_dir) / "checkpoints" / "policy") if args.save_interval else None,
        save_interval=args.save_interval,
    )
    elapsed = time.time() - start
    save_checkpoint(refined_policy, args, Path(args.output_dir), elapsed, extra_meta=stats)
    return stats


# --------------------------------------------------------------------------- #
# Jump-Start RL (JSRL)
# --------------------------------------------------------------------------- #


def run_jsrl(args: argparse.Namespace, env: Any, target_policy: BaseTargetPolicy) -> Dict[str, Any]:
    """Jump-start RL: refine from high-return target-policy trajectories.

    We approximate JSRL by building a critical-state buffer from the top
    target-policy trajectories and running the RICE refinement pipeline with
    p=1.0 (always reset from the demonstration buffer).
    """
    print(f"Collecting {args.n_trajectories} target-policy trajectories for JSRL...")
    trajectories = _collect_target_trajectories(target_policy, env, args.n_trajectories)
    # Keep only the top half by return.
    trajectories = sorted(trajectories, key=lambda t: t["return"], reverse=True)
    demo_trajectories = trajectories[: max(1, len(trajectories) // 2)]

    buffer = build_critical_buffer_from_trajectories(
        demo_trajectories,
        capacity=None,
        selection_mode="top_p",
        top_p=1.0,  # use all demo states
        threshold=0.0,
    )

    # Run refinement with p=1.0 (always from demos) and no RND bonus.
    jsrl_args = argparse.Namespace(**vars(args))
    jsrl_args.p = 1.0
    jsrl_args.lambda_rnd = 0.0
    return run_refining_baseline(jsrl_args, env, target_policy, buffer, baseline_name="jsrl")


# --------------------------------------------------------------------------- #
# StateMask-R proxy and alternative-explanation refining baselines
# --------------------------------------------------------------------------- #


def _load_or_build_mask_network(
    args: argparse.Namespace,
    env: Any,
    target_policy: BaseTargetPolicy,
    device: torch.device,
) -> MaskNetwork:
    if args.mask is not None:
        mask_path = Path(args.mask)
        if mask_path.suffix == ".pt" or mask_path.suffix == ".pth":
            mask = build_mask_network(env.observation_space)
            mask.load_state_dict(torch.load(mask_path, map_location="cpu"))
            return mask
    # Train a mask network from scratch (proxy for StateMask-R mask stage).
    print("Training a mask network for the StateMask-R proxy...")
    mask_trainer = train_mask_network(
        env,
        target_policy,
        total_timesteps=min(args.total_timesteps, 500_000),
        alpha=1e-4,
        device=str(device),
        save_path=str(Path(args.output_dir) / "mask.pt"),
    )
    return mask_trainer.mask_network


def _build_buffer_from_explainer(
    explainer: Any,
    env: Any,
    target_policy: BaseTargetPolicy,
    n_trajectories: int,
    top_p: float,
) -> CriticalStateBuffer:
    trajectories = _collect_target_trajectories(target_policy, env, n_trajectories)
    for traj in trajectories:
        obs_batch = np.stack(traj["observations"])
        scores = explainer.predict(obs_batch)
        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)
        traj["xi"] = scores.squeeze(-1).tolist()
    return build_critical_buffer_from_trajectories(
        trajectories,
        capacity=None,
        selection_mode="top_p",
        top_p=top_p,
        threshold=0.5,
    )


def run_refining_baseline(
    args: argparse.Namespace,
    env: Any,
    target_policy: BaseTargetPolicy,
    buffer: CriticalStateBuffer,
    baseline_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the RICE refinement pipeline with a fixed critical-state buffer."""
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    config = build_ppo_config(args)
    start = time.time()
    trainer = refine_policy(
        env,
        target_policy,
        critical_buffer=buffer,
        total_timesteps=args.total_timesteps,
        p=args.p,
        lambda_rnd=args.lambda_rnd,
        ppo_config=config,
        device=device,
        save_path=str(Path(args.output_dir) / "policy.pt"),
    )
    elapsed = time.time() - start
    extra = {"buffer_size": len(buffer)}
    save_checkpoint(trainer.refined_policy, args, Path(args.output_dir), elapsed, extra_meta=extra)
    return {"elapsed": elapsed, **extra}


def run_statemask_r(args: argparse.Namespace, env: Any, target_policy: BaseTargetPolicy) -> Dict[str, Any]:
    """Proxy for StateMask-R: train mask + refine from its critical states."""
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    mask_net = _load_or_build_mask_network(args, env, target_policy, device)
    buffer = _build_buffer_from_explainer(mask_net, env, target_policy, args.n_trajectories, args.top_p)
    return run_refining_baseline(args, env, target_policy, buffer, baseline_name="statemask-r")


def run_random_exp(args: argparse.Namespace, env: Any, target_policy: BaseTargetPolicy) -> Dict[str, Any]:
    """Fixed refining pipeline + Random explanation."""
    explainer = RandomExplanation(seed=args.seed)
    buffer = _build_buffer_from_explainer(explainer, env, target_policy, args.n_trajectories, args.top_p)
    return run_refining_baseline(args, env, target_policy, buffer, baseline_name="random-exp")


def run_ig_exp(args: argparse.Namespace, env: Any, target_policy: BaseTargetPolicy) -> Dict[str, Any]:
    """Fixed refining pipeline + Integrated Gradients explanation."""
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if not isinstance(target_policy, TorchTargetPolicy):
        raise ValueError("Integrated-Gradients baseline requires a TorchTargetPolicy")
    explainer = IntegratedGradients(
        target_model=lambda obs: target_policy.model.get_value(obs),
        n_steps=50,
        aggregator="sum",
    )
    buffer = _build_buffer_from_explainer(explainer, env, target_policy, args.n_trajectories, args.top_p)
    return run_refining_baseline(args, env, target_policy, buffer, baseline_name="ig-exp")


def run_airs_exp(args: argparse.Namespace, env: Any, target_policy: BaseTargetPolicy) -> Dict[str, Any]:
    """Fixed refining pipeline + AIRS stub explanation."""
    explainer = AIRSStub(score=0.5, randomize=True, seed=args.seed)
    buffer = _build_buffer_from_explainer(explainer, env, target_policy, args.n_trajectories, args.top_p)
    return run_refining_baseline(args, env, target_policy, buffer, baseline_name="airs-exp")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # Resolve env_id from metadata if needed.
    metadata_path = _auto_detect_metadata(args.target_policy, args.metadata)
    metadata = _read_metadata(metadata_path)
    if args.env_id is None and "env_id" in metadata:
        args.env_id = metadata["env_id"]
    if args.task == "mujoco" and args.env_id is None and "env_id" in metadata:
        args.env_id = metadata["env_id"]

    env = make_env(args, args.seed)

    # Load target policy.
    target_policy = load_target_policy(
        args.target_policy,
        backend="sb3" if args.use_sb3 else "torch",
        device=args.device,
    )

    baseline_dispatch = {
        "ppo-finetune": run_ppo_finetune,
        "sil": run_sil,
        "jsrl": run_jsrl,
        "statemask-r": run_statemask_r,
        "random-exp": run_random_exp,
        "ig-exp": run_ig_exp,
        "airs-exp": run_airs_exp,
    }

    runner = baseline_dispatch[args.baseline]
    print(f"Running baseline: {args.baseline}")
    stats = runner(args, env, target_policy)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "stats.json", "w") as f:
        json.dump({k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in stats.items()}, f, indent=2)
    print("Done.")


if __name__ == "__main__":
    try:
        import gymnasium as gym
    except ImportError:
        import gym  # type: ignore
    main()
