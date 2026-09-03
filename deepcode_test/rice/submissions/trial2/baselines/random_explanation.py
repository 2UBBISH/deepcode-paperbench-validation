"""
Random Explanation Baseline for RICE

This module implements the random explanation baseline: instead of using a trained
mask network to identify critical states, it randomly selects states from collected
trajectories. The selected states are then used in the same refining pipeline as RICE
(mixed initial state distribution + RND exploration bonus).

This serves as a control to demonstrate that RICE's mask-based critical state
identification provides meaningful improvement over random selection.

Paper Reference: Section 5 (Experiments), baseline comparison.
"""

import os
import time
import argparse
import json
import pickle
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from rice.utils import (
    load_config,
    set_seed,
    CriticalStateBuffer,
    Logger,
    collect_trajectories,
    evaluate_policy,
    ensure_dir,
    get_device,
    make_env,
    set_env_state,
    format_time,
)
from rice.refining import refine_agent


# ==============================================================================
# Random Explanation Extraction
# ==============================================================================

def extract_random_critical_states(
    agent_policy: Any,
    env_id: str,
    config: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    num_trajectories: int = 100,
    max_steps: int = 1000,
    buffer_size: int = 10000,
    seed: int = 0,
    device: str = "auto",
    deterministic_agent: bool = True,
    top_k_per_trajectory: int = 1,
    save_buffer: bool = True,
    verbose: int = 1,
    **env_kwargs,
) -> Tuple[CriticalStateBuffer, List[Dict]]:
    """
    Extract "critical" states by randomly selecting states from trajectories.

    This is the random explanation baseline: instead of using a mask network
    to compute importance scores, we uniformly sample states from collected
    trajectories and treat them as critical.

    Args:
        agent_policy: Pre-trained agent policy (SB3 model or callable).
        env_id: Gym environment ID.
        config: Configuration dictionary (optional).
        output_dir: Directory to save outputs.
        num_trajectories: Number of trajectories to collect.
        max_steps: Maximum steps per trajectory.
        buffer_size: Maximum size of the critical state buffer.
        seed: Random seed.
        device: Device string ("auto", "cpu", "cuda").
        deterministic_agent: Whether to use deterministic agent actions.
        top_k_per_trajectory: Number of states to select per trajectory.
        save_buffer: Whether to save the buffer to disk.
        verbose: Verbosity level.
        **env_kwargs: Additional keyword arguments for environment creation.

    Returns:
        Tuple of (CriticalStateBuffer, list of selected state dicts).
    """
    if config is None:
        config = load_config()

    set_seed(seed)
    device = get_device(device)

    if output_dir is not None:
        ensure_dir(output_dir)

    # Create environment for trajectory collection
    env = make_env(env_id, seed=seed, **env_kwargs)

    if verbose >= 1:
        print(f"[RandomExplanation] Collecting {num_trajectories} trajectories "
              f"from {env_id}...")

    # Collect trajectories using the agent policy
    trajectories = collect_trajectories(
        env=env,
        policy=agent_policy,
        num_trajectories=num_trajectories,
        max_steps=max_steps,
        deterministic=deterministic_agent,
    )

    if verbose >= 1:
        total_steps = sum(len(t["observations"]) for t in trajectories)
        print(f"[RandomExplanation] Collected {len(trajectories)} trajectories "
              f"with {total_steps} total steps.")

    # Create buffer
    buffer = CriticalStateBuffer(max_size=buffer_size)

    # Randomly select states from each trajectory
    selected_states = []
    rng = np.random.RandomState(seed)

    for traj_idx, traj in enumerate(trajectories):
        obs_list = traj["observations"]
        actions = traj.get("actions", [])
        rewards = traj.get("rewards", [])
        next_obs_list = traj.get("next_observations", [])
        dones = traj.get("dones", [])

        n_steps = len(obs_list)
        if n_steps == 0:
            continue

        # Randomly select up to top_k_per_trajectory states
        k = min(top_k_per_trajectory, n_steps)
        selected_indices = rng.choice(n_steps, size=k, replace=False)

        for step_idx in selected_indices:
            obs = obs_list[step_idx]
            # Assign random importance score (uniform [0, 1])
            importance = rng.uniform(0.0, 1.0)

            action = actions[step_idx] if step_idx < len(actions) else None
            next_obs = next_obs_list[step_idx] if step_idx < len(next_obs_list) else None
            done = dones[step_idx] if step_idx < len(dones) else False

            buffer.add(
                state=obs,
                action=action,
                next_state=next_obs,
                importance=importance,
                trajectory_id=traj_idx,
                step=step_idx,
            )

            selected_states.append({
                "trajectory_id": traj_idx,
                "step": step_idx,
                "state": obs,
                "action": action,
                "next_state": next_obs,
                "importance": importance,
                "done": done,
            })

    env.close()

    if verbose >= 1:
        print(f"[RandomExplanation] Selected {len(selected_states)} random states "
              f"from {len(trajectories)} trajectories.")
        print(f"[RandomExplanation] Buffer size: {len(buffer)}")

    # Save buffer if requested
    if save_buffer and output_dir is not None:
        buffer_path = os.path.join(output_dir, "random_critical_states.pkl")
        buffer.save(buffer_path)
        if verbose >= 1:
            print(f"[RandomExplanation] Saved buffer to {buffer_path}")

        # Also save selected states as JSON for inspection
        states_path = os.path.join(output_dir, "random_selected_states.json")
        # Convert numpy arrays to lists for JSON serialization
        serializable_states = []
        for s in selected_states:
            ss = {}
            for k, v in s.items():
                if isinstance(v, np.ndarray):
                    ss[k] = v.tolist()
                elif isinstance(v, (np.integer,)):
                    ss[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    ss[k] = float(v)
                else:
                    ss[k] = v
            serializable_states.append(ss)

        with open(states_path, "w") as f:
            json.dump(serializable_states, f, indent=2)
        if verbose >= 1:
            print(f"[RandomExplanation] Saved selected states to {states_path}")

    return buffer, selected_states


# ==============================================================================
# Full Random Explanation Pipeline
# ==============================================================================

def run_random_explanation_pipeline(
    env_id: str,
    agent_path: str,
    config: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    seed: int = 0,
    num_trajectories: int = 100,
    max_steps: int = 1000,
    buffer_size: int = 10000,
    top_k_per_trajectory: int = 1,
    # Refining parameters
    total_timesteps: Optional[int] = None,
    p: Optional[float] = None,
    lambda_rnd: Optional[float] = None,
    use_rnd: bool = True,
    use_mixed_init: bool = True,
    rnd_embedding_dim: int = 128,
    rnd_hidden_sizes: Optional[List[int]] = None,
    rnd_learning_rate: float = 1e-4,
    ppo_learning_rate: Optional[float] = None,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.0,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    eval_freq: int = 10000,
    n_eval_episodes: int = 10,
    rnd_update_freq: int = 1000,
    rnd_batch_size: int = 64,
    rnd_n_epochs: int = 1,
    state_buffer_size: int = 100000,
    device: str = "auto",
    verbose: int = 1,
    save_freq: int = 100000,
    **env_kwargs,
) -> Dict[str, Any]:
    """
    Run the full random explanation baseline pipeline:

    1. Load the pre-trained agent.
    2. Collect trajectories and randomly select "critical" states.
    3. Run the RICE refining process using the randomly selected states.

    Args:
        env_id: Gym environment ID.
        agent_path: Path to the pre-trained agent model.
        config: Configuration dictionary.
        output_dir: Directory to save outputs.
        seed: Random seed.
        num_trajectories: Number of trajectories for state selection.
        max_steps: Maximum steps per trajectory.
        buffer_size: Maximum buffer size for critical states.
        top_k_per_trajectory: Number of random states per trajectory.
        total_timesteps: Total timesteps for refining.
        p: Probability of sampling from critical states.
        lambda_rnd: RND bonus coefficient.
        use_rnd: Whether to use RND exploration bonus.
        use_mixed_init: Whether to use mixed initial state distribution.
        rnd_embedding_dim: RND embedding dimension.
        rnd_hidden_sizes: RND hidden layer sizes.
        rnd_learning_rate: RND predictor learning rate.
        ppo_learning_rate: PPO learning rate for refining.
        n_steps: PPO n_steps.
        batch_size: PPO batch size.
        n_epochs: PPO n_epochs.
        gamma: Discount factor.
        gae_lambda: GAE lambda.
        clip_range: PPO clip range.
        ent_coef: Entropy coefficient.
        vf_coef: Value function coefficient.
        max_grad_norm: Maximum gradient norm.
        eval_freq: Evaluation frequency.
        n_eval_episodes: Number of evaluation episodes.
        rnd_update_freq: RND update frequency.
        rnd_batch_size: RND update batch size.
        rnd_n_epochs: RND update epochs.
        state_buffer_size: State buffer size for RND.
        device: Device string.
        verbose: Verbosity level.
        save_freq: Checkpoint save frequency.
        **env_kwargs: Additional environment kwargs.

    Returns:
        Dictionary with results including refined model path, metrics, etc.
    """
    if config is None:
        config = load_config(env_id)

    set_seed(seed)
    device = get_device(device)

    if output_dir is None:
        output_dir = os.path.join(
            config.get("general", {}).get("log_dir", "./logs"),
            "random_explanation",
            env_id,
            f"seed_{seed}",
        )
    ensure_dir(output_dir)

    logger = Logger(log_dir=output_dir)

    start_time = time.time()

    # ------------------------------------------------------------------
    # Step 1: Load pre-trained agent
    # ------------------------------------------------------------------
    if verbose >= 1:
        print(f"[RandomExplanation] Loading agent from {agent_path}...")

    # Try loading as SB3 model first
    try:
        from stable_baselines3 import PPO
        agent = PPO.load(agent_path, device=device)
        if verbose >= 1:
            print(f"[RandomExplanation] Loaded SB3 PPO agent.")
    except Exception as e:
        if verbose >= 1:
            print(f"[RandomExplanation] Could not load as SB3 PPO: {e}")
            print(f"[RandomExplanation] Trying torch.load...")
        agent = torch.load(agent_path, map_location=device)
        if verbose >= 1:
            print(f"[RandomExplanation] Loaded agent via torch.load.")

    # ------------------------------------------------------------------
    # Step 2: Extract random critical states
    # ------------------------------------------------------------------
    if verbose >= 1:
        print(f"[RandomExplanation] Extracting random critical states...")

    buffer, selected_states = extract_random_critical_states(
        agent_policy=agent,
        env_id=env_id,
        config=config,
        output_dir=output_dir,
        num_trajectories=num_trajectories,
        max_steps=max_steps,
        buffer_size=buffer_size,
        seed=seed,
        device=device,
        top_k_per_trajectory=top_k_per_trajectory,
        save_buffer=True,
        verbose=verbose,
        **env_kwargs,
    )

    # Save buffer path for refining
    critical_states_path = os.path.join(output_dir, "random_critical_states.pkl")

    extraction_time = time.time() - start_time
    logger.log("extraction_time", extraction_time, 0)
    logger.log("num_critical_states", len(buffer), 0)

    if verbose >= 1:
        print(f"[RandomExplanation] Extraction completed in {format_time(extraction_time)}")
        print(f"[RandomExplanation] Buffer contains {len(buffer)} states.")

    # ------------------------------------------------------------------
    # Step 3: Run refining with random critical states
    # ------------------------------------------------------------------
    if verbose >= 1:
        print(f"[RandomExplanation] Starting refining with random critical states...")

    refine_output_dir = os.path.join(output_dir, "refined")

    refined_model, refine_logger, refined_model_path = refine_agent(
        env_id=env_id,
        agent_path=agent_path,
        critical_states_path=critical_states_path,
        config=config,
        output_dir=refine_output_dir,
        seed=seed,
        total_timesteps=total_timesteps,
        p=p,
        lambda_rnd=lambda_rnd,
        use_rnd=use_rnd,
        use_mixed_init=use_mixed_init,
        rnd_embedding_dim=rnd_embedding_dim,
        rnd_hidden_sizes=rnd_hidden_sizes,
        rnd_learning_rate=rnd_learning_rate,
        ppo_learning_rate=ppo_learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        rnd_update_freq=rnd_update_freq,
        rnd_batch_size=rnd_batch_size,
        rnd_n_epochs=rnd_n_epochs,
        state_buffer_size=state_buffer_size,
        device=device,
        verbose=verbose,
        save_freq=save_freq,
        **env_kwargs,
    )

    total_time = time.time() - start_time
    logger.log("total_time", total_time, 0)

    # ------------------------------------------------------------------
    # Step 4: Final evaluation
    # ------------------------------------------------------------------
    if verbose >= 1:
        print(f"[RandomExplanation] Running final evaluation...")

    eval_env = make_env(env_id, seed=seed + 1000, **env_kwargs)
    eval_results = evaluate_policy(
        env=eval_env,
        policy=refined_model,
        n_episodes=n_eval_episodes,
        deterministic=True,
    )
    eval_env.close()

    logger.log("final_mean_return", eval_results["mean_reward"], 0)
    logger.log("final_std_return", eval_results["std_reward"], 0)

    if verbose >= 1:
        print(f"[RandomExplanation] Final evaluation: "
              f"mean={eval_results['mean_reward']:.4f}, "
              f"std={eval_results['std_reward']:.4f}")

    # Save results summary
    results = {
        "env_id": env_id,
        "method": "random_explanation",
        "seed": seed,
        "num_critical_states": len(buffer),
        "extraction_time": extraction_time,
        "total_time": total_time,
        "final_mean_return": float(eval_results["mean_reward"]),
        "final_std_return": float(eval_results["std_reward"]),
        "refined_model_path": refined_model_path,
        "critical_states_path": critical_states_path,
        "config": {
            "num_trajectories": num_trajectories,
            "top_k_per_trajectory": top_k_per_trajectory,
            "p": p,
            "lambda_rnd": lambda_rnd,
            "use_rnd": use_rnd,
            "use_mixed_init": use_mixed_init,
            "total_timesteps": total_timesteps,
        },
    }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save logger
    logger_path = os.path.join(output_dir, "logger.pkl")
    logger.save(logger_path)

    if verbose >= 1:
        print(f"[RandomExplanation] Results saved to {results_path}")
        print(f"[RandomExplanation] Total time: {format_time(total_time)}")

    return results


# ==============================================================================
# Fidelity Computation for Random Explanation
# ==============================================================================

def compute_random_fidelity_score(
    agent_policy: Any,
    env_id: str,
    critical_states: List[Dict],
    num_episodes: int = 100,
    max_steps: int = 1000,
    seed: int = 0,
    device: str = "auto",
    verbose: int = 1,
    **env_kwargs,
) -> Dict[str, float]:
    """
    Compute fidelity score for random explanation baseline.

    Fidelity measures how much the return drops when we randomize the agent's
    action at the identified "critical" state vs. at a random state.
    For random explanation, we expect low fidelity (small difference).

    Args:
        agent_policy: Pre-trained agent policy.
        env_id: Gym environment ID.
        critical_states: List of selected critical state dicts.
        num_episodes: Number of evaluation episodes.
        max_steps: Maximum steps per episode.
        seed: Random seed.
        device: Device string.
        verbose: Verbosity level.
        **env_kwargs: Additional environment kwargs.

    Returns:
        Dictionary with fidelity metrics.
    """
    set_seed(seed)
    device = get_device(device)

    env = make_env(env_id, seed=seed, **env_kwargs)

    # Build lookup: (trajectory_id, step) -> state info
    state_lookup = {}
    for s in critical_states:
        key = (s.get("trajectory_id", -1), s.get("step", -1))
        state_lookup[key] = s

    # Evaluate with randomization at critical states
    critical_returns = []
    random_returns = []

    rng = np.random.RandomState(seed)

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_return = 0.0
        step = 0

        # Determine if this episode will have a critical state intervention
        # Match trajectory_id to episode index
        traj_id = ep

        while not done and not truncated and step < max_steps:
            # Check if current step matches a critical state for this trajectory
            key = (traj_id, step)
            if key in state_lookup:
                # Randomize action at critical state
                action = env.action_space.sample()
            else:
                # Use agent action
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                with torch.no_grad():
                    if hasattr(agent_policy, "predict"):
                        action, _ = agent_policy.predict(obs, deterministic=True)
                    else:
                        action = agent_policy(obs_tensor.unsqueeze(0)).squeeze(0).cpu().numpy()

            obs, reward, done, truncated, info = env.step(action)
            ep_return += reward
            step += 1

        critical_returns.append(ep_return)

    # Evaluate with randomization at random states (10% of steps)
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_return = 0.0
        step = 0

        while not done and not truncated and step < max_steps:
            # Randomize with 10% probability
            if rng.random() < 0.1:
                action = env.action_space.sample()
            else:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                with torch.no_grad():
                    if hasattr(agent_policy, "predict"):
                        action, _ = agent_policy.predict(obs, deterministic=True)
                    else:
                        action = agent_policy(obs_tensor.unsqueeze(0)).squeeze(0).cpu().numpy()

            obs, reward, done, truncated, info = env.step(action)
            ep_return += reward
            step += 1

        random_returns.append(ep_return)

    env.close()

    mean_critical = np.mean(critical_returns)
    std_critical = np.std(critical_returns)
    mean_random = np.mean(random_returns)
    std_random = np.std(random_returns)

    # Fidelity: drop from random baseline to critical intervention
    # Higher fidelity = larger drop for critical states
    fidelity = mean_random - mean_critical

    results = {
        "mean_return_critical": float(mean_critical),
        "std_return_critical": float(std_critical),
        "mean_return_random": float(mean_random),
        "std_return_random": float(std_random),
        "fidelity": float(fidelity),
        "num_episodes": num_episodes,
    }

    if verbose >= 1:
        print(f"[RandomExplanation] Fidelity Results:")
        print(f"  Return with critical randomization: {mean_critical:.4f} ± {std_critical:.4f}")
        print(f"  Return with random randomization:   {mean_random:.4f} ± {std_random:.4f}")
        print(f"  Fidelity (drop):                     {fidelity:.4f}")

    return results


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main():
    """Command-line entry point for random explanation baseline."""
    parser = argparse.ArgumentParser(
        description="Random Explanation Baseline for RICE"
    )

    # Required arguments
    parser.add_argument(
        "--env_id", type=str, required=True,
        help="Gym environment ID (e.g., Hopper-v3)"
    )
    parser.add_argument(
        "--agent_path", type=str, required=True,
        help="Path to pre-trained agent model"
    )

    # Extraction arguments
    parser.add_argument(
        "--num_trajectories", type=int, default=100,
        help="Number of trajectories for state selection"
    )
    parser.add_argument(
        "--max_steps", type=int, default=1000,
        help="Maximum steps per trajectory"
    )
    parser.add_argument(
        "--buffer_size", type=int, default=10000,
        help="Maximum buffer size"
    )
    parser.add_argument(
        "--top_k_per_trajectory", type=int, default=1,
        help="Number of random states per trajectory"
    )

    # Refining arguments
    parser.add_argument(
        "--total_timesteps", type=int, default=None,
        help="Total timesteps for refining"
    )
    parser.add_argument(
        "--p", type=float, default=None,
        help="Probability of sampling from critical states"
    )
    parser.add_argument(
        "--lambda_rnd", type=float, default=None,
        help="RND bonus coefficient"
    )
    parser.add_argument(
        "--no_rnd", action="store_true",
        help="Disable RND exploration bonus"
    )
    parser.add_argument(
        "--no_mixed_init", action="store_true",
        help="Disable mixed initial state distribution"
    )

    # PPO arguments
    parser.add_argument(
        "--n_steps", type=int, default=2048,
        help="PPO n_steps"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="PPO batch size"
    )
    parser.add_argument(
        "--n_epochs", type=int, default=10,
        help="PPO n_epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=None,
        help="PPO learning rate"
    )

    # RND arguments
    parser.add_argument(
        "--rnd_embedding_dim", type=int, default=128,
        help="RND embedding dimension"
    )
    parser.add_argument(
        "--rnd_lr", type=float, default=1e-4,
        help="RND predictor learning rate"
    )

    # General arguments
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device (auto, cpu, cuda)"
    )
    parser.add_argument(
        "--verbose", type=int, default=1,
        help="Verbosity level"
    )
    parser.add_argument(
        "--eval_freq", type=int, default=10000,
        help="Evaluation frequency"
    )
    parser.add_argument(
        "--n_eval_episodes", type=int, default=10,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--save_freq", type=int, default=100000,
        help="Checkpoint save frequency"
    )

    # Fidelity-only mode
    parser.add_argument(
        "--fidelity_only", action="store_true",
        help="Only compute fidelity score (skip refining)"
    )
    parser.add_argument(
        "--critical_states_path", type=str, default=None,
        help="Path to pre-extracted critical states (for fidelity-only mode)"
    )

    args = parser.parse_args()

    # Load config
    if args.config is not None:
        with open(args.config, "r") as f:
            import yaml
            config = yaml.safe_load(f)
    else:
        config = load_config(args.env_id)

    # Override config with CLI arguments
    if args.total_timesteps is not None:
        if "refining" not in config:
            config["refining"] = {}
        config["refining"]["total_timesteps"] = args.total_timesteps
    if args.p is not None:
        if "refining" not in config:
            config["refining"] = {}
        config["refining"]["p"] = args.p
    if args.lambda_rnd is not None:
        if "refining" not in config:
            config["refining"] = {}
        config["refining"]["lambda"] = args.lambda_rnd

    if args.fidelity_only:
        # Load agent
        from stable_baselines3 import PPO
        agent = PPO.load(args.agent_path, device=args.device)

        # Load critical states
        if args.critical_states_path is not None:
            buffer = CriticalStateBuffer()
            buffer.load(args.critical_states_path)
            # Convert buffer to list of dicts
            critical_states = []
            for i in range(len(buffer)):
                # Access internal buffer
                if hasattr(buffer, '_buffer'):
                    critical_states.append(buffer._buffer[i])
                else:
                    break
        else:
            # Extract random states
            _, critical_states = extract_random_critical_states(
                agent_policy=agent,
                env_id=args.env_id,
                config=config,
                output_dir=args.output_dir,
                num_trajectories=args.num_trajectories,
                max_steps=args.max_steps,
                buffer_size=args.buffer_size,
                seed=args.seed,
                device=args.device,
                top_k_per_trajectory=args.top_k_per_trajectory,
                save_buffer=False,
                verbose=args.verbose,
            )

        # Compute fidelity
        results = compute_random_fidelity_score(
            agent_policy=agent,
            env_id=args.env_id,
            critical_states=critical_states,
            num_episodes=args.n_eval_episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            device=args.device,
            verbose=args.verbose,
        )

        # Save results
        if args.output_dir is not None:
            ensure_dir(args.output_dir)
            results_path = os.path.join(args.output_dir, "fidelity_results.json")
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Fidelity results saved to {results_path}")

    else:
        # Run full pipeline
        results = run_random_explanation_pipeline(
            env_id=args.env_id,
            agent_path=args.agent_path,
            config=config,
            output_dir=args.output_dir,
            seed=args.seed,
            num_trajectories=args.num_trajectories,
            max_steps=args.max_steps,
            buffer_size=args.buffer_size,
            top_k_per_trajectory=args.top_k_per_trajectory,
            total_timesteps=args.total_timesteps,
            p=args.p,
            lambda_rnd=args.lambda_rnd,
            use_rnd=not args.no_rnd,
            use_mixed_init=not args.no_mixed_init,
            rnd_embedding_dim=args.rnd_embedding_dim,
            rnd_learning_rate=args.rnd_lr,
            ppo_learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            device=args.device,
            verbose=args.verbose,
            save_freq=args.save_freq,
        )

        print(f"\nRandom Explanation pipeline completed!")
        print(f"Final mean return: {results['final_mean_return']:.4f} ± "
              f"{results['final_std_return']:.4f}")


if __name__ == "__main__":
    main()