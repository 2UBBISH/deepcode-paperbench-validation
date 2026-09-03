#!/usr/bin/env python3
"""
Train a base PPO agent on a given environment until convergence.

This script implements the agent training phase of the RICE pipeline:
1. Creates the environment (with optional wrappers)
2. Configures and trains a PPO agent using Stable-Baselines3
3. Saves the trained model and training curves

Supports:
- MuJoCo environments (Hopper, Walker2d, Reacher, HalfCheetah)
- Sparse variants of MuJoCo environments
- Custom environments (selfish mining, CAGE, auto driving, malware)

Usage:
    python experiments/train_agent.py --env Hopper-v3 --total_timesteps 1000000
    python experiments/train_agent.py --env selfish_mining --config config/env_specific/selfish_mining.yaml
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rice.utils import (
    load_config,
    set_seed,
    make_env,
    make_vec_env,
    evaluate_policy,
    ensure_dir,
    get_device,
    Logger,
    format_time,
)

# Stable-Baselines3 imports
try:
    import stable_baselines3
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CheckpointCallback,
        EvalCallback,
    )
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import (
        VecNormalize,
        DummyVecEnv,
        SubprocVecEnv,
        VecMonitor,
    )
    from stable_baselines3.common.utils import set_random_seed
except ImportError:
    raise ImportError(
        "Stable-Baselines3 is required. Install with: pip install stable-baselines3>=2.0.0"
    )


# ==============================================================================
# Custom Callbacks
# ==============================================================================


class TrainingLoggerCallback(BaseCallback):
    """
    Custom callback that logs training metrics to our Logger.
    Records: episode rewards, value loss, policy loss, entropy, etc.
    """

    def __init__(
        self,
        logger: Logger,
        log_interval: int = 1000,
        eval_env=None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 10,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.logger = logger
        self.log_interval = log_interval
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0.0
        self.current_episode_length = 0
        self.start_time = time.time()

    def _on_step(self) -> bool:
        # Track episode rewards from infos
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep_info = info["episode"]
                self.episode_rewards.append(ep_info["r"])
                self.episode_lengths.append(ep_info["l"])
                self.logger.log("train/episode_reward", ep_info["r"], self.num_timesteps)
                self.logger.log("train/episode_length", ep_info["l"], self.num_timesteps)

        # Log training metrics at intervals
        if self.num_timesteps % self.log_interval == 0:
            if hasattr(self.model, "logger") and self.model.logger is not None:
                for key in self.model.logger.name_to_value:
                    value = self.model.logger.name_to_value[key]
                    self.logger.log(f"train/{key}", value, self.num_timesteps)

            # Log FPS
            elapsed = time.time() - self.start_time
            fps = self.num_timesteps / max(elapsed, 1e-8)
            self.logger.log("train/fps", fps, self.num_timesteps)
            self.logger.log("train/elapsed_time", elapsed, self.num_timesteps)

        # Periodic evaluation
        if self.eval_env is not None and self.num_timesteps % self.eval_freq == 0:
            eval_result = evaluate_policy(
                self.eval_env,
                self.model,
                n_episodes=self.n_eval_episodes,
                deterministic=True,
            )
            self.logger.log(
                "eval/mean_reward", eval_result["mean_reward"], self.num_timesteps
            )
            self.logger.log(
                "eval/std_reward", eval_result["std_reward"], self.num_timesteps
            )

        return True

    def _on_training_end(self) -> None:
        elapsed = time.time() - self.start_time
        self.logger.log("train/total_time", elapsed, self.num_timesteps)


class ProgressBarCallback(BaseCallback):
    """Simple progress bar callback using tqdm."""

    def __init__(self, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.pbar = None

    def _on_training_start(self) -> None:
        try:
            from tqdm import tqdm

            self.pbar = tqdm(total=self.total_timesteps, desc="Training", unit="steps")
        except ImportError:
            self.pbar = None

    def _on_step(self) -> bool:
        if self.pbar is not None:
            self.pbar.update(self.locals.get("n_steps", 1))
        return True

    def _on_training_end(self) -> None:
        if self.pbar is not None:
            self.pbar.close()


# ==============================================================================
# Environment Creation
# ==============================================================================


def create_environment(
    env_id: str,
    seed: int = 0,
    n_envs: int = 1,
    normalize: bool = True,
    normalize_obs: bool = True,
    normalize_reward: bool = True,
    gamma: float = 0.99,
    **kwargs,
) -> Tuple[Any, Optional[VecNormalize]]:
    """
    Create and configure the environment for training.

    Args:
        env_id: Gymnasium environment ID or custom env name.
        seed: Random seed.
        n_envs: Number of parallel environments.
        normalize: Whether to use VecNormalize wrapper.
        normalize_obs: Normalize observations.
        normalize_reward: Normalize rewards.
        gamma: Discount factor for reward normalization.
        **kwargs: Additional arguments passed to make_env.

    Returns:
        Tuple of (vec_env, vec_normalize_instance or None).
    """
    if n_envs == 1:
        env = make_env(env_id, seed=seed, **kwargs)
        env = Monitor(env)
        vec_env = DummyVecEnv([lambda: env])
    else:
        vec_env = make_vec_env(
            env_id,
            n_envs=n_envs,
            seed=seed,
            **kwargs,
        )
        vec_env = VecMonitor(vec_env)

    vec_norm = None
    if normalize:
        vec_norm = VecNormalize(
            vec_env,
            norm_obs=normalize_obs,
            norm_reward=normalize_reward,
            gamma=gamma,
        )
        vec_env = vec_norm

    return vec_env, vec_norm


# ==============================================================================
# Policy Configuration
# ==============================================================================


def build_policy_kwargs(
    config: Dict[str, Any],
    env_id: str,
    observation_space: Any,
    action_space: Any,
) -> Dict[str, Any]:
    """
    Build policy_kwargs for PPO based on config and environment type.

    Args:
        config: Agent configuration dictionary.
        env_id: Environment identifier.
        observation_space: Observation space of the environment.
        action_space: Action space of the environment.

    Returns:
        Dictionary of policy keyword arguments.
    """
    policy_kwargs = config.get("policy_kwargs", {}).copy()

    # Determine network architecture
    net_arch = policy_kwargs.get("net_arch", None)
    if net_arch is None:
        # Default: 2-layer MLP with 64 units each (MuJoCo default)
        hidden_sizes = policy_kwargs.get("hidden_sizes", [64, 64])
        activation_fn = policy_kwargs.get("activation_fn", "tanh")

        # For custom environments, use larger networks if specified
        if "selfish_mining" in env_id.lower():
            hidden_sizes = policy_kwargs.get("hidden_sizes", [128, 128, 128, 128])
        elif "cage" in env_id.lower():
            hidden_sizes = policy_kwargs.get("hidden_sizes", [128, 128, 128, 128])
        elif "metadrive" in env_id.lower() or "auto_driving" in env_id.lower():
            hidden_sizes = policy_kwargs.get("hidden_sizes", [256, 256])
        elif "malware" in env_id.lower():
            hidden_sizes = policy_kwargs.get("hidden_sizes", [128, 128])

        net_arch = dict(pi=hidden_sizes, vf=hidden_sizes)

    policy_kwargs["net_arch"] = net_arch

    # Activation function
    if "activation_fn" in policy_kwargs:
        act_name = policy_kwargs["activation_fn"]
        if act_name == "tanh":
            policy_kwargs["activation_fn"] = torch.nn.Tanh
        elif act_name == "relu":
            policy_kwargs["activation_fn"] = torch.nn.ReLU
        elif act_name == "elu":
            policy_kwargs["activation_fn"] = torch.nn.ELU

    # Remove keys not recognized by SB3
    for key in ["hidden_sizes"]:
        policy_kwargs.pop(key, None)

    return policy_kwargs


# ==============================================================================
# Main Training Function
# ==============================================================================


def train_agent(
    env_id: str,
    config: Dict[str, Any],
    output_dir: str,
    seed: int = 0,
    total_timesteps: Optional[int] = None,
    n_envs: Optional[int] = None,
    device: str = "auto",
    save_freq: int = 100000,
    eval_freq: int = 10000,
    n_eval_episodes: int = 10,
    verbose: int = 1,
    resume_from: Optional[str] = None,
    **env_kwargs,
) -> Tuple[PPO, Logger, str]:
    """
    Train a PPO agent on the specified environment.

    Args:
        env_id: Environment ID (e.g., 'Hopper-v3', 'selfish_mining').
        config: Configuration dictionary (agent section).
        output_dir: Directory to save model and logs.
        seed: Random seed.
        total_timesteps: Override total training timesteps.
        n_envs: Override number of parallel environments.
        device: Device string ('auto', 'cpu', 'cuda').
        save_freq: Frequency of model checkpointing.
        eval_freq: Frequency of evaluation.
        n_eval_episodes: Number of evaluation episodes.
        verbose: Verbosity level.
        resume_from: Path to checkpoint to resume from.
        **env_kwargs: Additional keyword arguments for environment creation.

    Returns:
        Tuple of (trained PPO model, Logger, model_save_path).
    """
    # Set random seeds
    set_seed(seed)

    # Get device
    torch_device = get_device(device)
    print(f"[train_agent] Using device: {torch_device}")

    # Extract agent config
    agent_config = config.get("agent", config)

    # Override total_timesteps if provided
    if total_timesteps is None:
        total_timesteps = agent_config.get("total_timesteps", 1_000_000)
    if n_envs is None:
        n_envs = agent_config.get("n_envs", 1)

    # Create output directories
    ensure_dir(output_dir)
    log_dir = os.path.join(output_dir, "logs")
    model_dir = os.path.join(output_dir, "models")
    ensure_dir(log_dir)
    ensure_dir(model_dir)

    # Initialize logger
    logger = Logger(log_dir=log_dir)

    # Create training environment
    print(f"[train_agent] Creating environment: {env_id}")
    train_env, train_vec_norm = create_environment(
        env_id=env_id,
        seed=seed,
        n_envs=n_envs,
        normalize=agent_config.get("normalize", True),
        normalize_obs=agent_config.get("normalize_obs", True),
        normalize_reward=agent_config.get("normalize_reward", True),
        gamma=agent_config.get("gamma", 0.99),
        **env_kwargs,
    )

    # Create evaluation environment (separate instance)
    eval_env = None
    if eval_freq > 0:
        eval_env, _ = create_environment(
            env_id=env_id,
            seed=seed + 1000,  # Different seed for eval
            n_envs=1,
            normalize=agent_config.get("normalize", True),
            normalize_obs=agent_config.get("normalize_obs", True),
            normalize_reward=False,  # Don't normalize rewards for eval
            gamma=agent_config.get("gamma", 0.99),
            **env_kwargs,
        )

    # Build policy kwargs
    policy_kwargs = build_policy_kwargs(
        agent_config,
        env_id,
        train_env.observation_space,
        train_env.action_space,
    )

    # PPO hyperparameters
    ppo_kwargs = {
        "policy": agent_config.get("policy", "MlpPolicy"),
        "env": train_env,
        "learning_rate": agent_config.get("learning_rate", 3e-4),
        "n_steps": agent_config.get("n_steps", 2048),
        "batch_size": agent_config.get("batch_size", 64),
        "n_epochs": agent_config.get("n_epochs", 10),
        "gamma": agent_config.get("gamma", 0.99),
        "gae_lambda": agent_config.get("gae_lambda", 0.95),
        "clip_range": agent_config.get("clip_range", 0.2),
        "clip_range_vf": agent_config.get("clip_range_vf", None),
        "normalize_advantage": agent_config.get("normalize_advantage", True),
        "ent_coef": agent_config.get("ent_coef", 0.0),
        "vf_coef": agent_config.get("vf_coef", 0.5),
        "max_grad_norm": agent_config.get("max_grad_norm", 0.5),
        "use_sde": agent_config.get("use_sde", False),
        "sde_sample_freq": agent_config.get("sde_sample_freq", -1),
        "target_kl": agent_config.get("target_kl", None),
        "policy_kwargs": policy_kwargs,
        "verbose": verbose,
        "seed": seed,
        "device": torch_device,
        "tensorboard_log": log_dir if agent_config.get("tensorboard", False) else None,
    }

    # Remove None values
    ppo_kwargs = {k: v for k, v in ppo_kwargs.items() if v is not None}

    print(f"[train_agent] PPO configuration:")
    print(f"  - Total timesteps: {total_timesteps}")
    print(f"  - n_steps: {ppo_kwargs['n_steps']}")
    print(f"  - batch_size: {ppo_kwargs['batch_size']}")
    print(f"  - learning_rate: {ppo_kwargs['learning_rate']}")
    print(f"  - gamma: {ppo_kwargs['gamma']}")
    print(f"  - n_envs: {n_envs}")
    print(f"  - policy_kwargs: {policy_kwargs}")

    # Create or load model
    if resume_from and os.path.exists(resume_from):
        print(f"[train_agent] Resuming from: {resume_from}")
        model = PPO.load(resume_from, env=train_env, device=str(torch_device))
    else:
        model = PPO(**ppo_kwargs)

    # Setup callbacks
    callbacks = []

    # Training logger callback
    train_logger_callback = TrainingLoggerCallback(
        logger=logger,
        log_interval=agent_config.get("log_interval", 1000),
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
    )
    callbacks.append(train_logger_callback)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=model_dir,
        name_prefix=f"{env_id}_ppo",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)

    # Progress bar
    if verbose > 0:
        try:
            progress_callback = ProgressBarCallback(total_timesteps)
            callbacks.append(progress_callback)
        except Exception:
            pass

    # Train
    print(f"[train_agent] Starting training for {total_timesteps} timesteps...")
    start_time = time.time()

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=1,  # We handle logging in our callback
            progress_bar=False,  # We use our own progress bar
        )
    except KeyboardInterrupt:
        print("[train_agent] Training interrupted by user. Saving model...")

    training_time = time.time() - start_time
    print(f"[train_agent] Training completed in {format_time(training_time)}")

    # Save final model
    model_save_path = os.path.join(model_dir, f"{env_id}_ppo_final")
    model.save(model_save_path)
    print(f"[train_agent] Model saved to: {model_save_path}.zip")

    # Save VecNormalize statistics if used
    if train_vec_norm is not None:
        vec_norm_path = os.path.join(model_dir, f"{env_id}_vecnormalize.pkl")
        train_vec_norm.save(vec_norm_path)
        print(f"[train_agent] VecNormalize stats saved to: {vec_norm_path}")

    # Save logger data
    logger.save(os.path.join(log_dir, "training_metrics.json"))
    logger.save_csv(os.path.join(log_dir, "training_metrics.csv"))

    # Final evaluation
    if eval_env is not None:
        print("[train_agent] Running final evaluation...")
        final_eval = evaluate_policy(
            eval_env,
            model,
            n_episodes=n_eval_episodes * 10,  # More episodes for final eval
            deterministic=True,
        )
        print(f"  Final mean reward: {final_eval['mean_reward']:.2f} ± {final_eval['std_reward']:.2f}")
        logger.log("final/mean_reward", final_eval["mean_reward"], total_timesteps)
        logger.log("final/std_reward", final_eval["std_reward"], total_timesteps)

    # Clean up
    train_env.close()
    if eval_env is not None:
        eval_env.close()

    return model, logger, model_save_path


# ==============================================================================
# CLI Entry Point
# ==============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a PPO agent for the RICE pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on Hopper-v3 with default config
  python experiments/train_agent.py --env Hopper-v3

  # Train with environment-specific config
  python experiments/train_agent.py --env selfish_mining --config config/env_specific/selfish_mining.yaml

  # Train with custom timesteps and seed
  python experiments/train_agent.py --env Walker2d-v3 --total_timesteps 2000000 --seed 42

  # Resume from checkpoint
  python experiments/train_agent.py --env Hopper-v3 --resume models/hopper_ppo_1000000_steps.zip
        """,
    )

    parser.add_argument(
        "--env",
        type=str,
        required=True,
        help="Environment ID (e.g., Hopper-v3, Walker2d-v3, selfish_mining, cage, auto_driving, malware)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file. If not provided, uses default config with env-specific overrides.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for models and logs. Default: ./output/<env_id>/",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Total training timesteps (overrides config).",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=None,
        help="Number of parallel environments (overrides config).",
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
        choices=["auto", "cpu", "cuda", "cuda:0", "cuda:1"],
        help="Device to use for training.",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=100000,
        help="Save model checkpoint every N steps.",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10000,
        help="Evaluate policy every N steps.",
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=10,
        help="Number of episodes for evaluation.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to model checkpoint to resume training from.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable VecNormalize wrapper.",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load configuration
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = load_config(args.env)

    # Determine output directory
    if args.output_dir is None:
        output_dir = os.path.join(
            os.getcwd(), "output", args.env.replace("/", "_"), f"seed_{args.seed}"
        )
    else:
        output_dir = args.output_dir

    print(f"[train_agent] Output directory: {output_dir}")
    print(f"[train_agent] Environment: {args.env}")
    print(f"[train_agent] Seed: {args.seed}")

    # Override normalize if requested
    if args.no_normalize:
        if "agent" not in config:
            config["agent"] = {}
        config["agent"]["normalize"] = False

    # Train agent
    model, logger, model_path = train_agent(
        env_id=args.env,
        config=config,
        output_dir=output_dir,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        device=args.device,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        verbose=args.verbose,
        resume_from=args.resume,
    )

    print(f"\n[ train_agent] Training complete!")
    print(f"  Model saved to: {model_path}.zip")
    print(f"  Logs saved to: {output_dir}/logs/")

    return model, logger, model_path


if __name__ == "__main__":
    main()