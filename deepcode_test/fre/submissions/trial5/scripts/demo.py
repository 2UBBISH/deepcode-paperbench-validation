#!/usr/bin/env python3
"""
Demo script for Functional Reward Encodings (FRE).

This script demonstrates zero-shot offline RL using a trained FRE agent:
  1. Load a trained checkpoint (encoder, decoder, IQL agent).
  2. Define a downstream reward function (goal-reaching, directional, etc.).
  3. Encode the reward function from a few example states via the frozen encoder.
  4. Roll out the conditioned policy in the environment and visualize results.

Usage:
    python scripts/demo.py --checkpoint path/to/checkpoint.pt --domain antmaze \
        --task goal-reaching --render --record_video

    python scripts/demo.py --checkpoint path/to/checkpoint.pt --domain kitchen \
        --task subtask_0 --num_episodes 5 --render

    python scripts/demo.py --checkpoint path/to/checkpoint.pt --domain exorl_walker \
        --task velocity --target_velocity 2.0 --num_episodes 10
"""

import argparse
import logging
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import OfflineDataset, ReplayBuffer, load_dataset, create_replay_buffer
from evaluation import (
    FREEvaluator,
    EvaluationTask,
    EvaluationResult,
    build_tasks_for_domain,
    normalize_returns,
    compute_normalized_score,
    get_domain_normalization,
    make_antmaze_goal_reaching_reward,
    make_antmaze_directional_reward,
    make_antmaze_random_simplex_reward,
    make_antmaze_path_reward,
    make_exorl_goal_reaching_reward,
    make_exorl_velocity_reward,
    make_kitchen_subtask_reward,
)
from models import FREEncoder, FREDecoder, IQLAgent
from utils import set_seed, get_device, configure_logging, to_tensor, to_numpy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FRE Demo: Zero-shot offline RL with functional reward encodings."
    )

    # Required
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained checkpoint (.pt file).",
    )
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        choices=["antmaze", "exorl_walker", "exorl_cheetah", "kitchen"],
        help="Domain for evaluation.",
    )

    # Task specification
    parser.add_argument(
        "--task",
        type=str,
        default="goal-reaching",
        help="Task type: goal-reaching, directional, random-simplex, path-loop, "
        "path-edges, path-center, velocity, subtask_0..subtask_6.",
    )
    parser.add_argument(
        "--goal",
        type=float,
        nargs="+",
        default=None,
        help="Goal state coordinates (space-separated). If not provided, a random "
        "goal is sampled from the dataset.",
    )
    parser.add_argument(
        "--direction",
        type=float,
        nargs="+",
        default=None,
        help="Direction vector for directional task (space-separated).",
    )
    parser.add_argument(
        "--target_velocity",
        type=float,
        default=1.0,
        help="Target velocity for ExORL velocity task.",
    )
    parser.add_argument(
        "--subtask_idx",
        type=int,
        default=0,
        help="Subtask index for Kitchen (0-6).",
    )
    parser.add_argument(
        "--path_type",
        type=str,
        default="loop",
        choices=["loop", "edges", "center"],
        help="Path type for AntMaze path tasks.",
    )

    # Encoding
    parser.add_argument(
        "--K_enc",
        type=int,
        default=32,
        help="Number of encoding states for reward function encoding.",
    )

    # Rollout
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=5,
        help="Number of episodes to run.",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=1000,
        help="Maximum steps per episode.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic policy (mean action).",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        default=False,
        help="Use stochastic policy (sample from Gaussian).",
    )

    # Visualization
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Render the environment during rollouts.",
    )
    parser.add_argument(
        "--record_video",
        action="store_true",
        default=False,
        help="Record video of rollouts (requires gym wrappers).",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="./videos",
        help="Directory to save recorded videos.",
    )
    parser.add_argument(
        "--print_latent",
        action="store_true",
        default=False,
        help="Print the latent encoding vector.",
    )
    parser.add_argument(
        "--compare_latents",
        action="store_true",
        default=False,
        help="Compare latent encodings of multiple reward functions.",
    )

    # Config & data
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: configs/<domain>.yaml).",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to ExORL data directory (if needed).",
    )

    # Misc
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
        help="Device: 'auto', 'cpu', 'cuda', 'cuda:0', etc.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_configs(base: Dict, override: Dict) -> Dict:
    """Recursively merge two config dictionaries (override wins)."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Build configuration from default + domain-specific + CLI overrides."""
    # Load default config
    default_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs",
        "default.yaml",
    )
    if os.path.exists(default_path):
        config = load_yaml(default_path)
    else:
        config = {}

    # Load domain-specific config
    domain_config_map = {
        "antmaze": "antmaze.yaml",
        "exorl_walker": "exorl.yaml",
        "exorl_cheetah": "exorl.yaml",
        "kitchen": "kitchen.yaml",
    }
    domain_config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs",
        domain_config_map.get(args.domain, "default.yaml"),
    )
    if os.path.exists(domain_config_path):
        domain_config = load_yaml(domain_config_path)
        config = merge_configs(config, domain_config)

    # Load user-specified config
    if args.config and os.path.exists(args.config):
        user_config = load_yaml(args.config)
        config = merge_configs(config, user_config)

    # Override with CLI arguments
    if args.data_path:
        config.setdefault("dataset", {})["exorl_data_path"] = args.data_path

    return config


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models_from_checkpoint(
    checkpoint_path: str,
    state_dim: int,
    action_dim: int,
    config: Dict[str, Any],
    device: torch.device,
) -> Tuple[FREEncoder, FREDecoder, IQLAgent]:
    """Reconstruct encoder, decoder, and IQL agent from a checkpoint."""
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model configs
    enc_cfg = config.get("encoder", {})
    dec_cfg = config.get("decoder", {})
    agent_cfg = config.get("agent", {})

    # Build encoder
    encoder = FREEncoder(
        state_dim=state_dim,
        embed_dim=enc_cfg.get("embed_dim", 256),
        latent_dim=enc_cfg.get("latent_dim", 64),
        num_layers=enc_cfg.get("num_layers", 3),
        num_heads=enc_cfg.get("num_heads", 4),
        dropout=enc_cfg.get("dropout", 0.1),
        num_bins=enc_cfg.get("num_bins", 64),
        reward_min=enc_cfg.get("reward_min"),
        reward_max=enc_cfg.get("reward_max"),
    ).to(device)

    # Build decoder
    decoder = FREDecoder(
        state_dim=state_dim,
        latent_dim=enc_cfg.get("latent_dim", 64),
        hidden_dims=dec_cfg.get("hidden_dims", [256, 256]),
    ).to(device)

    # Build IQL agent
    agent = IQLAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=enc_cfg.get("latent_dim", 64),
        hidden_dims=agent_cfg.get("hidden_dims", [256, 256]),
        expectile=agent_cfg.get("expectile", 0.7),
        temperature=agent_cfg.get("temperature", 3.0),
        discount=agent_cfg.get("discount", 0.99),
        target_tau=agent_cfg.get("target_tau", 0.005),
    ).to(device)

    # Load state dicts
    if "encoder_state_dict" in checkpoint:
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
    elif "encoder" in checkpoint:
        encoder.load_state_dict(checkpoint["encoder"])

    if "decoder_state_dict" in checkpoint:
        decoder.load_state_dict(checkpoint["decoder_state_dict"])
    elif "decoder" in checkpoint:
        decoder.load_state_dict(checkpoint["decoder"])

    if "agent_state_dict" in checkpoint:
        agent.load_state_dict(checkpoint["agent_state_dict"])
    elif "agent" in checkpoint:
        agent.load_state_dict(checkpoint["agent"])

    # Freeze models for evaluation
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad = False

    agent.eval()
    for p in agent.parameters():
        p.requires_grad = False

    logger.info("Models loaded and frozen for evaluation.")
    return encoder, decoder, agent


# ---------------------------------------------------------------------------
# Reward function construction
# ---------------------------------------------------------------------------

def build_demo_reward_function(
    args: argparse.Namespace,
    replay_buffer: ReplayBuffer,
    rng: np.random.RandomState,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a reward function based on CLI arguments."""
    domain = args.domain
    all_states = replay_buffer.get_all_states()

    if domain == "antmaze":
        if args.task == "goal-reaching":
            if args.goal is not None:
                goal = np.array(args.goal, dtype=np.float32)
            else:
                # Sample random goal from dataset
                idx = rng.randint(0, len(all_states))
                goal = all_states[idx].copy()
                logger.info(f"Sampled random goal (first 5 dims): {goal[:5]}")
            return make_antmaze_goal_reaching_reward(goal)

        elif args.task == "directional":
            if args.direction is not None:
                direction = np.array(args.direction, dtype=np.float32)
            else:
                # Random direction
                direction = rng.randn(2).astype(np.float32)
                direction /= np.linalg.norm(direction) + 1e-8
                logger.info(f"Random direction: {direction}")
            return make_antmaze_directional_reward(direction)

        elif args.task == "random-simplex":
            return make_antmaze_random_simplex_reward(
                state_dim=all_states.shape[1], rng=rng
            )

        elif args.task.startswith("path-"):
            path_type = args.task.split("-", 1)[1]
            # Generate synthetic path points
            if path_type == "loop":
                t = np.linspace(0, 2 * np.pi, 20)
                path_points = np.stack([np.cos(t), np.sin(t)], axis=1) * 5.0
            elif path_type == "edges":
                path_points = np.array(
                    [[-5, -5], [5, -5], [5, 5], [-5, 5]], dtype=np.float32
                )
            elif path_type == "center":
                path_points = np.array([[0, 0]], dtype=np.float32)
            else:
                path_points = np.array([[0, 0]], dtype=np.float32)
            return make_antmaze_path_reward(path_points)

        else:
            raise ValueError(f"Unknown AntMaze task: {args.task}")

    elif domain in ("exorl_walker", "exorl_cheetah"):
        if args.task == "goal-reaching":
            if args.goal is not None:
                goal = np.array(args.goal, dtype=np.float32)
            else:
                idx = rng.randint(0, len(all_states))
                goal = all_states[idx].copy()
                logger.info(f"Sampled random goal (first 5 dims): {goal[:5]}")
            return make_exorl_goal_reaching_reward(goal)

        elif args.task == "velocity":
            return make_exorl_velocity_reward(
                target_velocity=args.target_velocity,
                velocity_idx=0,  # Default: first velocity component
            )

        else:
            raise ValueError(f"Unknown ExORL task: {args.task}")

    elif domain == "kitchen":
        if args.task.startswith("subtask_"):
            subtask_idx = int(args.task.split("_")[1])
        else:
            subtask_idx = args.subtask_idx
        return make_kitchen_subtask_reward(subtask_idx)

    else:
        raise ValueError(f"Unknown domain: {domain}")


# ---------------------------------------------------------------------------
# Environment creation
# ---------------------------------------------------------------------------

def make_env(
    domain: str,
    render: bool = False,
    record_video: bool = False,
    video_dir: str = "./videos",
) -> Any:
    """Create a gym environment for the given domain."""
    try:
        import gym
    except ImportError:
        logger.error("gym is required for environment interaction. Install with: pip install gym")
        raise

    env_name_map = {
        "antmaze": "antmaze-large-diverse-v2",
        "exorl_walker": "walker-walk-v0",  # Approximate; may need adjustment
        "exorl_cheetah": "cheetah-run-v0",  # Approximate
        "kitchen": "kitchen-complete-v0",
    }

    env_name = env_name_map.get(domain, domain)

    try:
        env = gym.make(env_name)
    except Exception:
        logger.warning(f"Could not create env '{env_name}', trying '{domain}' directly.")
        env = gym.make(domain)

    if record_video:
        os.makedirs(video_dir, exist_ok=True)
        try:
            from gym.wrappers import RecordVideo
            env = RecordVideo(
                env,
                video_dir,
                episode_trigger=lambda ep: True,
                name_prefix=f"fre-demo-{domain}",
            )
            logger.info(f"Video recording enabled, saving to {video_dir}")
        except ImportError:
            logger.warning("RecordVideo wrapper not available; install gym[all] for video support.")

    return env


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def run_episode(
    env: Any,
    encoder: FREEncoder,
    agent: IQLAgent,
    latent_z: np.ndarray,
    replay_buffer: ReplayBuffer,
    max_steps: int = 1000,
    deterministic: bool = True,
    render: bool = False,
    device: torch.device = None,
) -> Tuple[float, int]:
    """Run a single episode with the conditioned policy."""
    obs = env.reset()
    if hasattr(obs, "__len__") and not isinstance(obs, np.ndarray):
        obs = np.array(obs, dtype=np.float32)
    else:
        obs = np.array(obs, dtype=np.float32).flatten()

    total_reward = 0.0
    steps = 0

    latent_tensor = torch.as_tensor(latent_z, dtype=torch.float32, device=device)

    for step in range(max_steps):
        if render:
            env.render()

        # Normalize state
        norm_obs = replay_buffer.normalize_states(obs.reshape(1, -1))
        state_tensor = torch.as_tensor(norm_obs, dtype=torch.float32, device=device)

        # Get action
        with torch.no_grad():
            action = agent.get_action(state_tensor, latent_tensor, deterministic=deterministic)

        action_np = action.flatten()

        # Step environment
        result = env.step(action_np)
        if len(result) == 4:
            next_obs, reward, done, info = result
        else:
            next_obs, reward, terminated, truncated, info = result
            done = terminated or truncated

        if hasattr(next_obs, "__len__") and not isinstance(next_obs, np.ndarray):
            next_obs = np.array(next_obs, dtype=np.float32)
        else:
            next_obs = np.array(next_obs, dtype=np.float32).flatten()

        total_reward += float(reward)
        steps += 1
        obs = next_obs

        if done:
            break

    return total_reward, steps


# ---------------------------------------------------------------------------
# Latent comparison
# ---------------------------------------------------------------------------

def compare_latent_encodings(
    encoder: FREEncoder,
    replay_buffer: ReplayBuffer,
    reward_fns: Dict[str, Callable],
    K_enc: int,
    device: torch.device,
) -> None:
    """Compare latent encodings of multiple reward functions."""
    logger.info("\n" + "=" * 60)
    logger.info("Latent Encoding Comparison")
    logger.info("=" * 60)

    all_states = replay_buffer.get_all_states()
    rng = np.random.RandomState(42)

    latents = {}
    for name, reward_fn in reward_fns.items():
        # Sample encoding states
        idx = rng.choice(len(all_states), size=min(K_enc, len(all_states)), replace=False)
        enc_states = all_states[idx]
        rewards = reward_fn(enc_states)

        # Encode
        states_t = torch.as_tensor(
            replay_buffer.normalize_states(enc_states), dtype=torch.float32, device=device
        )
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device)

        with torch.no_grad():
            z = encoder.encode_deterministic(states_t, rewards_t)

        latents[name] = to_numpy(z).flatten()
        logger.info(f"  {name}: latent norm = {np.linalg.norm(latents[name]):.4f}")

    # Compute pairwise cosine similarities
    logger.info("\nPairwise Cosine Similarities:")
    names = list(latents.keys())
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if i < j:
                sim = np.dot(latents[n1], latents[n2]) / (
                    np.linalg.norm(latents[n1]) * np.linalg.norm(latents[n2]) + 1e-8
                )
                logger.info(f"  {n1} vs {n2}: {sim:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Setup
    set_seed(args.seed)
    device = get_device(args.device != "cpu") if args.device == "auto" else torch.device(args.device)
    configure_logging(log_dir=None, level=logging.DEBUG if args.verbose else logging.INFO)

    logger.info("=" * 60)
    logger.info("FRE Demo: Zero-Shot Offline RL")
    logger.info("=" * 60)
    logger.info(f"Domain: {args.domain}")
    logger.info(f"Task: {args.task}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Device: {device}")

    # Load config
    config = build_config(args)

    # Determine dataset name
    dataset_name_map = {
        "antmaze": "antmaze-large-diverse-v2",
        "exorl_walker": "walker",
        "exorl_cheetah": "cheetah",
        "kitchen": "kitchen-complete-v0",
    }
    dataset_name = dataset_name_map.get(args.domain, args.domain)

    # Load dataset
    logger.info(f"Loading dataset: {dataset_name}")
    try:
        dataset = load_dataset(
            dataset_name,
            normalize_states=True,
            data_path=args.data_path or config.get("dataset", {}).get("exorl_data_path"),
        )
    except Exception as e:
        logger.warning(f"Could not load dataset '{dataset_name}': {e}")
        logger.info("Attempting to create a minimal dataset for demo purposes...")
        # Create a dummy dataset for demo if real data unavailable
        state_dim = config.get("domain", {}).get("state_dim", 29)
        action_dim = config.get("domain", {}).get("action_dim", 8)
        dummy_states = np.random.randn(1000, state_dim).astype(np.float32)
        dummy_actions = np.random.randn(1000, action_dim).astype(np.float32)
        dummy_next_states = np.random.randn(1000, state_dim).astype(np.float32)
        dummy_terminals = np.zeros(1000, dtype=np.float32)
        from data import create_dataset_from_arrays
        dataset = create_dataset_from_arrays(
            dummy_states, dummy_actions, dummy_next_states, dummy_terminals,
            normalize_states=True,
        )

    replay_buffer = create_replay_buffer(dataset, device=device)

    # Infer dimensions
    sample = replay_buffer.sample(1)
    state_dim = sample["states"].shape[1]
    action_dim = sample["actions"].shape[1]
    logger.info(f"State dim: {state_dim}, Action dim: {action_dim}")

    # Load models
    encoder, decoder, agent = load_models_from_checkpoint(
        args.checkpoint, state_dim, action_dim, config, device
    )

    # Build reward function
    rng = np.random.RandomState(args.seed + 1)
    reward_fn = build_demo_reward_function(args, replay_buffer, rng)
    logger.info(f"Reward function: {args.task}")

    # Encode reward function
    logger.info(f"Encoding reward function using K={args.K_enc} states...")
    all_states = replay_buffer.get_all_states()
    idx = rng.choice(len(all_states), size=min(args.K_enc, len(all_states)), replace=False)
    enc_states = all_states[idx]
    rewards = reward_fn(enc_states)

    states_t = torch.as_tensor(
        replay_buffer.normalize_states(enc_states), dtype=torch.float32, device=device
    )
    rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=device)

    with torch.no_grad():
        latent_z = encoder.encode_deterministic(states_t, rewards_t)

    latent_np = to_numpy(latent_z).flatten()
    logger.info(f"Latent encoding shape: {latent_np.shape}")
    if args.print_latent:
        logger.info(f"Latent vector (first 10 dims): {latent_np[:10]}")
        logger.info(f"Latent norm: {np.linalg.norm(latent_np):.4f}")

    # Compare latents if requested
    if args.compare_latents:
        # Build multiple reward functions for comparison
        compare_fns = {}
        if args.domain == "antmaze":
            compare_fns["goal-reaching"] = make_antmaze_goal_reaching_reward(
                all_states[rng.randint(0, len(all_states))]
            )
            compare_fns["directional"] = make_antmaze_directional_reward(
                np.array([1.0, 0.0], dtype=np.float32)
            )
            compare_fns["random-simplex"] = make_antmaze_random_simplex_reward(
                state_dim=state_dim, rng=rng
            )
        elif args.domain in ("exorl_walker", "exorl_cheetah"):
            compare_fns["goal-reaching"] = make_exorl_goal_reaching_reward(
                all_states[rng.randint(0, len(all_states))]
            )
            compare_fns["velocity-1.0"] = make_exorl_velocity_reward(1.0)
            compare_fns["velocity-3.0"] = make_exorl_velocity_reward(3.0)
        elif args.domain == "kitchen":
            for i in range(min(3, 7)):
                compare_fns[f"subtask_{i}"] = make_kitchen_subtask_reward(i)

        compare_latent_encodings(encoder, replay_buffer, compare_fns, args.K_enc, device)

    # Create environment
    deterministic = not args.stochastic
    logger.info(f"Creating environment (render={args.render}, record_video={args.record_video})...")
    env = make_env(args.domain, render=args.render, record_video=args.record_video, video_dir=args.video_dir)

    # Run episodes
    logger.info(f"\nRunning {args.num_episodes} episodes...")
    logger.info("-" * 60)

    episode_returns = []
    episode_lengths = []

    for ep in range(args.num_episodes):
        ep_return, ep_length = run_episode(
            env=env,
            encoder=encoder,
            agent=agent,
            latent_z=latent_np,
            replay_buffer=replay_buffer,
            max_steps=args.max_episode_steps,
            deterministic=deterministic,
            render=args.render,
            device=device,
        )
        episode_returns.append(ep_return)
        episode_lengths.append(ep_length)
        logger.info(f"  Episode {ep + 1}: return = {ep_return:.2f}, length = {ep_length}")

    env.close()

    # Summary
    returns_arr = np.array(episode_returns)
    lengths_arr = np.array(episode_lengths)

    logger.info("-" * 60)
    logger.info("Summary:")
    logger.info(f"  Mean return: {returns_arr.mean():.2f} ± {returns_arr.std():.2f}")
    logger.info(f"  Min return: {returns_arr.min():.2f}")
    logger.info(f"  Max return: {returns_arr.max():.2f}")
    logger.info(f"  Mean episode length: {lengths_arr.mean():.1f}")

    # Normalize if possible
    try:
        min_ret, max_ret = get_domain_normalization(args.domain)
        norm_mean, norm_std = compute_normalized_score(
            episode_returns, min_ret, max_ret
        )
        logger.info(f"  Normalized score: {norm_mean:.1f} ± {norm_std:.1f} (scale 0-100)")
    except Exception:
        logger.info("  Normalized score: N/A (domain normalization not available)")

    logger.info("=" * 60)
    logger.info("Demo complete!")


if __name__ == "__main__":
    main()