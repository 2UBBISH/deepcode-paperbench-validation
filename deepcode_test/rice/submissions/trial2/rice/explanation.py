"""
Explanation Extraction Module for RICE
======================================
Implements critical state identification from a trained mask network.

Algorithm:
  1. Collect trajectories by running the agent policy π (or perturbed policy π̄).
  2. For each state sₜ in trajectory τ, compute importance:
     I(sₜ) = 1 - ξ(aᵉ=0 | sₜ)
     where ξ(aᵉ=0 | sₜ) is the mask network's probability of trusting the agent.
     Higher I(sₜ) indicates a more critical state.
  3. Select the state with maximum I in each trajectory as the critical state s*.
  4. Store s* in a buffer D (with optional metadata: action, next state, etc.).

Theoretical basis:
  I(s) ∝ Q_diff = Q^π(s,a) - E_a'[Q^π(s,a')]
  Larger Q_diff indicates higher criticality — the agent's specific action
  matters significantly more than a random action at this state.

Usage:
  from rice.explanation import ExplanationExtractor, extract_critical_states
"""

import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from rice.mask_network import compute_importance, get_mask_probability, load_mask_network
from rice.perturbed_env import PerturbedEnvWrapper
from rice.utils import (
    CriticalStateBuffer,
    Logger,
    collect_trajectories,
    ensure_dir,
    evaluate_policy,
    get_device,
    load_config,
    make_env,
    set_seed,
)


class ExplanationExtractor:
    """
    Extracts critical state explanations from a trained mask network.

    This class handles:
      - Collecting trajectories using the agent policy (or perturbed policy)
      - Computing importance scores for each state via the mask network
      - Identifying critical states (highest importance per trajectory)
      - Building and managing a critical state buffer

    Attributes:
        mask_network: Trained mask network (SB3 PPO model or callable).
        agent_policy: Frozen agent policy (SB3 model or callable).
        buffer: CriticalStateBuffer storing identified critical states.
        config: Configuration dictionary.
        device: Torch device for computation.
        logger: Logger instance for tracking metrics.
    """

    def __init__(
        self,
        mask_network: Any,
        agent_policy: Any,
        config: Optional[Dict[str, Any]] = None,
        buffer_size: int = 10000,
        device: str = "auto",
        logger: Optional[Logger] = None,
    ):
        """
        Initialize the ExplanationExtractor.

        Args:
            mask_network: Trained mask network (SB3 PPO model).
            agent_policy: Frozen agent policy (SB3 model or callable).
            config: Configuration dictionary (optional).
            buffer_size: Maximum size of the critical state buffer.
            device: Torch device string or "auto".
            logger: Optional Logger instance.
        """
        self.mask_network = mask_network
        self.agent_policy = agent_policy
        self.config = config or {}
        self.device = get_device(device)
        self.logger = logger or Logger()

        # Extract explanation-specific config
        expl_config = self.config.get("explanation", {})
        self.num_trajectories = expl_config.get("num_trajectories", 100)
        self.max_steps = expl_config.get("max_steps", 1000)
        self.use_perturbed_policy = expl_config.get("use_perturbed_policy", False)
        self.deterministic_agent = expl_config.get("deterministic_agent", True)
        self.top_k_per_trajectory = expl_config.get("top_k_per_trajectory", 1)
        self.min_importance_threshold = expl_config.get("min_importance_threshold", 0.0)

        # Initialize critical state buffer
        self.buffer = CriticalStateBuffer(max_size=buffer_size)

        # Statistics
        self._n_trajectories_processed = 0
        self._n_states_evaluated = 0
        self._mean_importance = 0.0

    def compute_importance_batch(
        self, observations: np.ndarray
    ) -> np.ndarray:
        """
        Compute importance scores for a batch of observations.

        I(s) = 1 - ξ(aᵉ=0 | s)

        Args:
            observations: Array of shape (n_states, obs_dim) or (obs_dim,).

        Returns:
            Importance scores array of shape (n_states,) or scalar.
        """
        return compute_importance(self.mask_network, observations)

    def compute_mask_probability_batch(
        self, observations: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        """
        Compute mask probability of trusting the agent for a batch.

        Args:
            observations: Array of shape (n_states, obs_dim) or (obs_dim,).
            deterministic: If True, return deterministic action (0 or 1).

        Returns:
            Probability array of shape (n_states,) or scalar.
        """
        return get_mask_probability(self.mask_network, observations, deterministic)

    def extract_from_trajectories(
        self,
        trajectories: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Extract critical states from a list of pre-collected trajectories.

        For each trajectory, computes importance scores for all states,
        selects the top-k states with highest importance, and adds them
        to the critical state buffer.

        Args:
            trajectories: List of trajectory dicts, each containing:
                - 'observations': np.ndarray (T, obs_dim)
                - 'actions': np.ndarray (T, act_dim)
                - 'rewards': np.ndarray (T,)
                - 'next_observations': np.ndarray (T, obs_dim) [optional]
                - 'dones': np.ndarray (T,)
                - 'infos': list of info dicts [optional]

        Returns:
            List of critical state dicts extracted, each containing:
                - 'state': np.ndarray (obs_dim,)
                - 'action': np.ndarray (act_dim,)
                - 'next_state': np.ndarray (obs_dim,) [if available]
                - 'importance': float
                - 'trajectory_id': int
                - 'step': int
        """
        critical_states = []

        for traj_id, traj in enumerate(trajectories):
            obs = traj["observations"]
            actions = traj.get("actions", None)
            next_obs = traj.get("next_observations", None)
            rewards = traj.get("rewards", None)
            dones = traj.get("dones", None)

            n_steps = len(obs)

            if n_steps == 0:
                continue

            # Compute importance for all states in trajectory
            importances = self.compute_importance_batch(obs)

            self._n_states_evaluated += n_steps
            self._mean_importance = (
                self._mean_importance * (self._n_states_evaluated - n_steps)
                + np.sum(importances)
            ) / max(1, self._n_states_evaluated)

            # Select top-k states by importance
            if self.top_k_per_trajectory >= n_steps:
                top_indices = np.arange(n_steps)
            else:
                top_indices = np.argpartition(
                    -importances, self.top_k_per_trajectory
                )[: self.top_k_per_trajectory]
                # Sort the top-k by importance (descending)
                top_indices = top_indices[
                    np.argsort(-importances[top_indices])
                ]

            for idx in top_indices:
                imp = float(importances[idx])

                # Skip if below threshold
                if imp < self.min_importance_threshold:
                    continue

                state_dict = {
                    "state": obs[idx].copy(),
                    "importance": imp,
                    "trajectory_id": traj_id,
                    "step": int(idx),
                }

                if actions is not None:
                    state_dict["action"] = (
                        actions[idx].copy()
                        if isinstance(actions, np.ndarray)
                        else actions[idx]
                    )

                if next_obs is not None and idx < len(next_obs):
                    state_dict["next_state"] = next_obs[idx].copy()

                if rewards is not None:
                    state_dict["reward"] = float(rewards[idx])

                if dones is not None:
                    state_dict["done"] = bool(dones[idx])

                # Add to buffer
                self.buffer.add(
                    state=state_dict["state"],
                    action=state_dict.get("action"),
                    next_state=state_dict.get("next_state"),
                    importance=imp,
                    trajectory_id=traj_id,
                    step=int(idx),
                )

                critical_states.append(state_dict)

        self._n_trajectories_processed += len(trajectories)

        # Log statistics
        self.logger.log("n_trajectories_processed", self._n_trajectories_processed, 0)
        self.logger.log("n_states_evaluated", self._n_states_evaluated, 0)
        self.logger.log("buffer_size", len(self.buffer), 0)
        self.logger.log("mean_importance", self._mean_importance, 0)

        return critical_states

    def collect_and_extract(
        self,
        env_id: str,
        num_trajectories: Optional[int] = None,
        max_steps: Optional[int] = None,
        seed: int = 0,
        **env_kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Collect trajectories using the agent policy and extract critical states.

        This is the main high-level method: creates an environment, collects
        trajectories (optionally using the perturbed policy wrapper), computes
        importance scores, and populates the critical state buffer.

        Args:
            env_id: Gymnasium environment ID.
            num_trajectories: Number of trajectories to collect (default from config).
            max_steps: Maximum steps per trajectory (default from config).
            seed: Random seed for environment.
            **env_kwargs: Additional environment keyword arguments.

        Returns:
            List of critical state dicts extracted.
        """
        num_traj = num_trajectories or self.num_trajectories
        max_st = max_steps or self.max_steps

        set_seed(seed)

        # Create environment
        if self.use_perturbed_policy:
            # Use perturbed policy wrapper for collection
            env = make_env(env_id, seed=seed, **env_kwargs)
            env = PerturbedEnvWrapper(
                env=env,
                agent_policy=self.agent_policy,
                mask_network=self.mask_network,
                deterministic_agent=self.deterministic_agent,
                device=self.device,
            )
        else:
            env = make_env(env_id, seed=seed, **env_kwargs)

        # Collect trajectories
        trajectories = collect_trajectories(
            env=env,
            policy=self.agent_policy if not self.use_perturbed_policy else None,
            num_trajectories=num_traj,
            max_steps=max_st,
            deterministic=self.deterministic_agent,
        )

        # If using perturbed wrapper, the wrapper handles action selection;
        # trajectories are collected via the wrapper's step method.
        # For standard collection, we pass the agent policy directly.

        env.close()

        # Extract critical states
        critical_states = self.extract_from_trajectories(trajectories)

        return critical_states

    def get_top_critical_states(self, k: int = 10) -> List[Dict[str, Any]]:
        """
        Get the top-k critical states from the buffer by importance.

        Args:
            k: Number of states to return.

        Returns:
            List of state dicts sorted by importance (descending).
        """
        return self.buffer.get_top_k(k)

    def sample_critical_states(self, n: int = 1) -> List[Dict[str, Any]]:
        """
        Sample critical states uniformly from the buffer.

        Args:
            n: Number of states to sample.

        Returns:
            List of state dicts.
        """
        return self.buffer.sample(n)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics of the extraction process.

        Returns:
            Dictionary with statistics.
        """
        all_importances = [
            entry.get("importance", 0.0) for entry in self.buffer.buffer
        ]
        return {
            "n_trajectories_processed": self._n_trajectories_processed,
            "n_states_evaluated": self._n_states_evaluated,
            "buffer_size": len(self.buffer),
            "mean_importance": self._mean_importance,
            "max_importance": float(np.max(all_importances)) if all_importances else 0.0,
            "min_importance": float(np.min(all_importances)) if all_importances else 0.0,
            "std_importance": float(np.std(all_importances)) if all_importances else 0.0,
        }

    def save_buffer(self, path: str) -> None:
        """
        Save the critical state buffer to disk.

        Args:
            path: File path for saving (pickle format).
        """
        self.buffer.save(path)

    def load_buffer(self, path: str) -> None:
        """
        Load a critical state buffer from disk.

        Args:
            path: File path to load from.
        """
        self.buffer.load(path)


def extract_critical_states(
    mask_network: Any,
    agent_policy: Any,
    env_id: str,
    config: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    num_trajectories: int = 100,
    max_steps: int = 1000,
    buffer_size: int = 10000,
    seed: int = 0,
    device: str = "auto",
    use_perturbed_policy: bool = False,
    deterministic_agent: bool = True,
    top_k_per_trajectory: int = 1,
    save_buffer: bool = True,
    verbose: bool = True,
    **env_kwargs,
) -> Tuple[ExplanationExtractor, List[Dict[str, Any]]]:
    """
    Convenience function to extract critical states from a trained mask network.

    This is the main entry point for explanation extraction. It:
      1. Creates an ExplanationExtractor with the given mask and agent.
      2. Collects trajectories from the specified environment.
      3. Computes importance scores and identifies critical states.
      4. Optionally saves the critical state buffer to disk.

    Args:
        mask_network: Trained mask network (SB3 PPO model).
        agent_policy: Frozen agent policy (SB3 model or callable).
        env_id: Gymnasium environment ID.
        config: Configuration dictionary.
        output_dir: Directory to save outputs.
        num_trajectories: Number of trajectories to collect.
        max_steps: Maximum steps per trajectory.
        buffer_size: Maximum buffer size.
        seed: Random seed.
        device: Torch device.
        use_perturbed_policy: Whether to use perturbed policy for collection.
        deterministic_agent: Whether to use deterministic agent actions.
        top_k_per_trajectory: Number of top states to extract per trajectory.
        save_buffer: Whether to save the buffer to disk.
        verbose: Whether to print progress.
        **env_kwargs: Additional environment arguments.

    Returns:
        Tuple of (ExplanationExtractor, list of critical state dicts).
    """
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "outputs", "explanation")
    ensure_dir(output_dir)

    logger = Logger(log_dir=output_dir)

    if verbose:
        print(f"[Explanation] Extracting critical states from {env_id}")
        print(f"[Explanation] Collecting {num_trajectories} trajectories...")

    start_time = time.time()

    extractor = ExplanationExtractor(
        mask_network=mask_network,
        agent_policy=agent_policy,
        config=config,
        buffer_size=buffer_size,
        device=device,
        logger=logger,
    )

    # Override config with function arguments
    extractor.num_trajectories = num_trajectories
    extractor.max_steps = max_steps
    extractor.use_perturbed_policy = use_perturbed_policy
    extractor.deterministic_agent = deterministic_agent
    extractor.top_k_per_trajectory = top_k_per_trajectory

    # Collect and extract
    critical_states = extractor.collect_and_extract(
        env_id=env_id,
        num_trajectories=num_trajectories,
        max_steps=max_steps,
        seed=seed,
        **env_kwargs,
    )

    elapsed = time.time() - start_time

    if verbose:
        stats = extractor.get_statistics()
        print(f"[Explanation] Completed in {elapsed:.1f}s")
        print(f"  - Trajectories processed: {stats['n_trajectories_processed']}")
        print(f"  - States evaluated: {stats['n_states_evaluated']}")
        print(f"  - Critical states extracted: {len(critical_states)}")
        print(f"  - Buffer size: {stats['buffer_size']}")
        print(f"  - Mean importance: {stats['mean_importance']:.4f}")
        print(f"  - Max importance: {stats['max_importance']:.4f}")

    # Save buffer
    if save_buffer and output_dir:
        buffer_path = os.path.join(output_dir, "critical_state_buffer.pkl")
        extractor.save_buffer(buffer_path)
        if verbose:
            print(f"[Explanation] Buffer saved to {buffer_path}")

        # Save statistics
        stats_path = os.path.join(output_dir, "explanation_stats.json")
        import json
        stats = extractor.get_statistics()
        stats["elapsed_time"] = elapsed
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

    # Save logger
    logger.save(os.path.join(output_dir, "explanation_logger.pkl"))

    return extractor, critical_states


def compute_fidelity_score(
    mask_network: Any,
    agent_policy: Any,
    env_id: str,
    critical_states: List[Dict[str, Any]],
    num_episodes: int = 100,
    max_steps: int = 1000,
    seed: int = 0,
    device: str = "auto",
    verbose: bool = True,
    **env_kwargs,
) -> Dict[str, float]:
    """
    Compute fidelity score: drop in return when randomizing actions at
    critical states vs. random states.

    Fidelity = (R_original - R_critical_randomized) / (R_original - R_random_randomized + ε)

    A higher fidelity score indicates that the identified critical states
    are truly more important than random states.

    Args:
        mask_network: Trained mask network.
        agent_policy: Agent policy.
        env_id: Environment ID.
        critical_states: List of critical state dicts (must contain 'state' and 'step').
        num_episodes: Number of evaluation episodes.
        max_steps: Maximum steps per episode.
        seed: Random seed.
        device: Torch device.
        verbose: Whether to print progress.
        **env_kwargs: Additional environment arguments.

    Returns:
        Dictionary with fidelity metrics:
            - 'original_return': Mean return of original policy.
            - 'critical_randomized_return': Mean return when randomizing at critical states.
            - 'random_randomized_return': Mean return when randomizing at random states.
            - 'fidelity_score': Computed fidelity score.
            - 'critical_drop': Absolute drop from critical randomization.
            - 'random_drop': Absolute drop from random randomization.
    """
    set_seed(seed)
    device = get_device(device)

    # Evaluate original policy
    env = make_env(env_id, seed=seed, **env_kwargs)
    original_result = evaluate_policy(
        env, agent_policy, n_episodes=num_episodes, deterministic=True
    )
    env.close()
    original_return = original_result["mean_reward"]

    if verbose:
        print(f"[Fidelity] Original return: {original_return:.4f}")

    # Evaluate with randomization at critical states
    critical_return = _evaluate_with_randomization(
        env_id=env_id,
        agent_policy=agent_policy,
        mask_network=mask_network,
        critical_states=critical_states,
        randomize_critical=True,
        num_episodes=num_episodes,
        max_steps=max_steps,
        seed=seed + 1,
        device=device,
        **env_kwargs,
    )

    if verbose:
        print(f"[Fidelity] Critical-randomized return: {critical_return:.4f}")

    # Evaluate with randomization at random states
    random_return = _evaluate_with_randomization(
        env_id=env_id,
        agent_policy=agent_policy,
        mask_network=mask_network,
        critical_states=critical_states,
        randomize_critical=False,
        num_episodes=num_episodes,
        max_steps=max_steps,
        seed=seed + 2,
        device=device,
        **env_kwargs,
    )

    if verbose:
        print(f"[Fidelity] Random-randomized return: {random_return:.4f}")

    # Compute fidelity score
    eps = 1e-8
    critical_drop = original_return - critical_return
    random_drop = original_return - random_return
    fidelity = critical_drop / (random_drop + eps)

    # Clamp fidelity to reasonable range
    fidelity = max(0.0, min(fidelity, 10.0))

    if verbose:
        print(f"[Fidelity] Critical drop: {critical_drop:.4f}")
        print(f"[Fidelity] Random drop: {random_drop:.4f}")
        print(f"[Fidelity] Fidelity score: {fidelity:.4f}")

    return {
        "original_return": original_return,
        "critical_randomized_return": critical_return,
        "random_randomized_return": random_return,
        "fidelity_score": fidelity,
        "critical_drop": critical_drop,
        "random_drop": random_drop,
    }


def _evaluate_with_randomization(
    env_id: str,
    agent_policy: Any,
    mask_network: Any,
    critical_states: List[Dict[str, Any]],
    randomize_critical: bool,
    num_episodes: int = 100,
    max_steps: int = 1000,
    seed: int = 0,
    device: str = "auto",
    **env_kwargs,
) -> float:
    """
    Evaluate policy with randomization at specified states.

    Args:
        env_id: Environment ID.
        agent_policy: Agent policy.
        mask_network: Mask network.
        critical_states: List of critical state dicts.
        randomize_critical: If True, randomize at critical states;
                           if False, randomize at random states.
        num_episodes: Number of episodes.
        max_steps: Maximum steps per episode.
        seed: Random seed.
        device: Torch device.
        **env_kwargs: Additional environment arguments.

    Returns:
        Mean return over episodes.
    """
    set_seed(seed)
    device = get_device(device)

    env = make_env(env_id, seed=seed, **env_kwargs)

    # Build set of critical step indices (trajectory_id, step) for matching
    critical_step_set = set()
    for cs in critical_states:
        critical_step_set.add((cs.get("trajectory_id", -1), cs.get("step", -1)))

    episode_returns = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_return = 0.0
        step = 0

        while not done and not truncated and step < max_steps:
            # Determine if we should randomize at this step
            should_randomize = False
            if randomize_critical:
                # Randomize if this step matches a critical state
                should_randomize = (ep, step) in critical_step_set
            else:
                # Randomize at a random step (e.g., 10% of steps)
                should_randomize = (np.random.random() < 0.1)

            if should_randomize:
                # Take random action
                action = env.action_space.sample()
            else:
                # Take agent action
                action = _get_agent_action(agent_policy, obs, deterministic=True)

            obs, reward, done, truncated, info = env.step(action)
            ep_return += reward
            step += 1

        episode_returns.append(ep_return)

    env.close()

    return float(np.mean(episode_returns))


def _get_agent_action(
    agent_policy: Any,
    observation: np.ndarray,
    deterministic: bool = True,
) -> np.ndarray:
    """
    Get action from agent policy for a single observation.

    Args:
        agent_policy: SB3 model or callable.
        observation: Single observation array.
        deterministic: Whether to use deterministic action.

    Returns:
        Action array.
    """
    # Handle SB3 model
    if hasattr(agent_policy, "predict"):
        action, _ = agent_policy.predict(observation, deterministic=deterministic)
        return action
    # Handle callable
    elif callable(agent_policy):
        return agent_policy(observation)
    # Handle torch module
    elif isinstance(agent_policy, nn.Module):
        with torch.no_grad():
            obs_tensor = torch.as_tensor(
                observation, dtype=torch.float32, device=get_device()
            ).unsqueeze(0)
            action = agent_policy(obs_tensor)
            if deterministic:
                action = action.mean if hasattr(action, "mean") else action
            return action.cpu().numpy().squeeze(0)
    else:
        raise ValueError(f"Unsupported agent policy type: {type(agent_policy)}")


def visualize_importance(
    extractor: ExplanationExtractor,
    env_id: str,
    num_episodes: int = 5,
    max_steps: int = 1000,
    seed: int = 0,
    output_dir: Optional[str] = None,
    **env_kwargs,
) -> Dict[str, Any]:
    """
    Collect trajectories and record per-step importance for visualization.

    Args:
        extractor: ExplanationExtractor instance with trained mask.
        env_id: Environment ID.
        num_episodes: Number of episodes to visualize.
        max_steps: Maximum steps per episode.
        seed: Random seed.
        output_dir: Directory to save visualization data.
        **env_kwargs: Additional environment arguments.

    Returns:
        Dictionary with per-trajectory importance data.
    """
    set_seed(seed)

    env = make_env(env_id, seed=seed, **env_kwargs)

    trajectory_data = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        step = 0

        ep_data = {
            "episode": ep,
            "observations": [],
            "actions": [],
            "rewards": [],
            "importances": [],
            "mask_probs": [],
        }

        while not done and not truncated and step < max_steps:
            action = _get_agent_action(
                extractor.agent_policy, obs, deterministic=True
            )

            # Compute importance and mask probability
            imp = float(extractor.compute_importance_batch(obs))
            mask_prob = float(extractor.compute_mask_probability_batch(obs))

            ep_data["observations"].append(obs.copy())
            ep_data["actions"].append(action.copy() if isinstance(action, np.ndarray) else action)
            ep_data["importances"].append(imp)
            ep_data["mask_probs"].append(mask_prob)

            obs, reward, done, truncated, info = env.step(action)
            ep_data["rewards"].append(float(reward))
            step += 1

        # Convert lists to arrays
        for key in ["observations", "actions", "rewards", "importances", "mask_probs"]:
            ep_data[key] = np.array(ep_data[key])

        trajectory_data.append(ep_data)

    env.close()

    # Save if output directory provided
    if output_dir:
        ensure_dir(output_dir)
        import pickle
        save_path = os.path.join(output_dir, "importance_visualization.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(trajectory_data, f)

    return {"trajectories": trajectory_data, "n_episodes": num_episodes}


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main():
    """Command-line interface for explanation extraction."""
    import argparse

    parser = argparse.ArgumentParser(
        description="RICE Explanation Extraction - Identify critical states"
    )
    parser.add_argument(
        "--env-id", type=str, default="Hopper-v3",
        help="Gymnasium environment ID"
    )
    parser.add_argument(
        "--mask-path", type=str, required=True,
        help="Path to trained mask network"
    )
    parser.add_argument(
        "--agent-path", type=str, required=True,
        help="Path to trained agent policy"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--num-trajectories", type=int, default=100,
        help="Number of trajectories to collect"
    )
    parser.add_argument(
        "--max-steps", type=int, default=1000,
        help="Maximum steps per trajectory"
    )
    parser.add_argument(
        "--buffer-size", type=int, default=10000,
        help="Maximum critical state buffer size"
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Torch device (cpu, cuda, auto)"
    )
    parser.add_argument(
        "--use-perturbed-policy", action="store_true",
        help="Use perturbed policy for trajectory collection"
    )
    parser.add_argument(
        "--top-k", type=int, default=1,
        help="Number of top states to extract per trajectory"
    )
    parser.add_argument(
        "--compute-fidelity", action="store_true",
        help="Compute fidelity score after extraction"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Generate importance visualization data"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Print progress information"
    )

    args = parser.parse_args()

    # Load configuration
    config = {}
    if args.config:
        config = load_config(env_name=None, base_config_path=args.config)
    else:
        try:
            config = load_config(env_name=args.env_id)
        except Exception:
            pass

    # Set output directory
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(
            os.getcwd(), "outputs", "explanation", args.env_id
        )

    # Load agent policy
    from stable_baselines3 import PPO
    print(f"[Explanation] Loading agent policy from {args.agent_path}")
    agent_policy = PPO.load(args.agent_path)

    # Load mask network
    print(f"[Explanation] Loading mask network from {args.mask_path}")
    env_temp = make_env(args.env_id, seed=args.seed)
    mask_network = load_mask_network(args.mask_path, env_temp, device=args.device)
    env_temp.close()

    # Extract critical states
    extractor, critical_states = extract_critical_states(
        mask_network=mask_network,
        agent_policy=agent_policy,
        env_id=args.env_id,
        config=config,
        output_dir=output_dir,
        num_trajectories=args.num_trajectories,
        max_steps=args.max_steps,
        buffer_size=args.buffer_size,
        seed=args.seed,
        device=args.device,
        use_perturbed_policy=args.use_perturbed_policy,
        top_k_per_trajectory=args.top_k,
        verbose=args.verbose,
    )

    # Compute fidelity if requested
    if args.compute_fidelity:
        print("\n[Explanation] Computing fidelity score...")
        fidelity_result = compute_fidelity_score(
            mask_network=mask_network,
            agent_policy=agent_policy,
            env_id=args.env_id,
            critical_states=critical_states,
            num_episodes=100,
            seed=args.seed + 100,
            device=args.device,
            verbose=args.verbose,
        )

        # Save fidelity results
        import json
        fidelity_path = os.path.join(output_dir, "fidelity_score.json")
        with open(fidelity_path, "w") as f:
            json.dump(fidelity_result, f, indent=2)
        print(f"[Explanation] Fidelity results saved to {fidelity_path}")

    # Generate visualization data if requested
    if args.visualize:
        print("\n[Explanation] Generating importance visualization data...")
        viz_data = visualize_importance(
            extractor=extractor,
            env_id=args.env_id,
            num_episodes=5,
            max_steps=args.max_steps,
            seed=args.seed + 200,
            output_dir=output_dir,
        )
        print(f"[Explanation] Visualization data saved to {output_dir}")

    print("\n[Explanation] Done!")


if __name__ == "__main__":
    main()