"""
Jump-Start RL (JSRL) Baseline Implementation

JSRL uses a pre-trained guide policy to initialize episodes, allowing the
student policy to start from states that the guide would visit. A curriculum
gradually reduces the number of guide steps, transitioning from full guidance
to autonomous learning.

Reference: "Jump-Start Reinforcement Learning" (Uchendu et al., 2022)
Paper: https://arxiv.org/abs/2204.02372
Original implementation: https://github.com/steventango/jumpstart-rl

In the RICE paper, JSRL serves as a baseline for the refining phase:
- RICE: identifies critical states via mask network, mixed init + RND bonus
- JSRL: uses guide policy rollouts to initialize episodes with curriculum

Algorithm:
1. Load pre-trained agent as guide policy π_g
2. Initialize student policy π_s (copy of π_g or fresh)
3. For each episode:
   a. Sample guide steps k ~ curriculum schedule
   b. Reset environment
   c. Run π_g for k steps, collecting trajectory
   d. Student π_s takes over from state s_k
   e. Update π_s using PPO on collected transitions
4. Gradually decrease k over training (e.g., linear decay)
"""

import os
import time
import argparse
import json
import pickle
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

# Internal imports
from rice.utils import (
    load_config, set_seed, Logger, ensure_dir, get_device,
    evaluate_policy, make_env, make_vec_env, format_time,
    get_project_root, CriticalStateBuffer
)


class JSCurriculum:
    """
    Curriculum schedule for Jump-Start RL.
    
    Controls the number of guide steps k at each training iteration.
    Supports linear, exponential, and step decay schedules.
    """
    
    def __init__(
        self,
        initial_guide_steps: int = 100,
        final_guide_steps: int = 0,
        total_timesteps: int = 1_000_000,
        schedule_type: str = "linear",
        decay_start_fraction: float = 0.0,
    ):
        """
        Args:
            initial_guide_steps: Starting number of guide steps (k_max)
            final_guide_steps: Final number of guide steps (k_min)
            total_timesteps: Total training timesteps for the schedule
            schedule_type: "linear", "exponential", or "step"
            decay_start_fraction: Fraction of total_timesteps before decay begins
        """
        self.initial_guide_steps = initial_guide_steps
        self.final_guide_steps = final_guide_steps
        self.total_timesteps = total_timesteps
        self.schedule_type = schedule_type
        self.decay_start_fraction = decay_start_fraction
        
        self._current_timestep = 0
    
    def get_guide_steps(self, current_timestep: Optional[int] = None) -> int:
        """
        Get the number of guide steps for the current training progress.
        
        Args:
            current_timestep: Current global timestep (uses internal if None)
            
        Returns:
            Number of guide steps k to use
        """
        if current_timestep is not None:
            self._current_timestep = current_timestep
        
        t = self._current_timestep
        T = self.total_timesteps
        k_max = self.initial_guide_steps
        k_min = self.final_guide_steps
        
        # Compute progress fraction
        decay_start = int(self.decay_start_fraction * T)
        if t < decay_start:
            progress = 0.0
        else:
            progress = min(1.0, (t - decay_start) / max(1, T - decay_start))
        
        if self.schedule_type == "linear":
            k = k_max - progress * (k_max - k_min)
        elif self.schedule_type == "exponential":
            # Exponential decay: k = k_max * (k_min/k_max)^progress
            if k_min == 0:
                k_min = 1  # Avoid log(0)
            k = k_max * (k_min / k_max) ** progress
        elif self.schedule_type == "step":
            # Step decay: halve every 25% of training
            num_halvings = int(progress * 4)
            k = max(k_min, k_max / (2 ** num_halvings))
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
        
        return max(0, int(round(k)))
    
    def update(self, timestep: int) -> int:
        """Update internal timestep and return guide steps."""
        self._current_timestep = timestep
        return self.get_guide_steps()
    
    def get_stats(self) -> Dict[str, Any]:
        """Return current curriculum statistics."""
        return {
            "current_timestep": self._current_timestep,
            "guide_steps": self.get_guide_steps(),
            "progress": min(1.0, self._current_timestep / max(1, self.total_timesteps)),
            "schedule_type": self.schedule_type,
            "initial_guide_steps": self.initial_guide_steps,
            "final_guide_steps": self.final_guide_steps,
        }
    
    def save(self, path: str) -> None:
        """Save curriculum state."""
        with open(path, "w") as f:
            json.dump(self.get_stats(), f, indent=2)
    
    def load(self, path: str) -> None:
        """Load curriculum state."""
        with open(path, "r") as f:
            data = json.load(f)
        self._current_timestep = data.get("current_timestep", 0)


class JSRLEnvWrapper(gym.Wrapper):
    """
    Environment wrapper for Jump-Start RL.
    
    This wrapper handles the guide policy rollout at the beginning of each
    episode. On reset, it optionally runs the guide policy for k steps,
    then returns control to the student policy.
    
    The wrapper exposes the same interface as a standard Gym environment,
    so it can be used with standard RL algorithms like PPO.
    """
    
    def __init__(
        self,
        env: gym.Env,
        guide_policy: Any,
        curriculum: JSCurriculum,
        deterministic_guide: bool = False,
        device: str = "auto",
    ):
        """
        Args:
            env: Base Gym environment
            guide_policy: Pre-trained policy used as guide (SB3 model or callable)
            curriculum: JSCurriculum instance controlling guide steps
            deterministic_guide: Whether guide acts deterministically
            device: Torch device
        """
        super().__init__(env)
        self.guide_policy = guide_policy
        self.curriculum = curriculum
        self.deterministic_guide = deterministic_guide
        self.device = get_device(device)
        
        # Episode state
        self._guide_steps_remaining = 0
        self._total_guide_steps_this_episode = 0
        self._episode_step = 0
        self._guide_actions_taken = 0
        
        # Statistics
        self._episode_rewards = []
        self._episode_guide_rewards = []
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment and optionally run guide policy."""
        self._episode_step = 0
        self._guide_actions_taken = 0
        
        # Determine guide steps for this episode
        self._guide_steps_remaining = self.curriculum.get_guide_steps()
        self._total_guide_steps_this_episode = self._guide_steps_remaining
        
        # Reset base environment
        obs, info = self.env.reset(seed=seed, options=options)
        
        # If guide steps > 0, run guide policy
        if self._guide_steps_remaining > 0:
            obs, info = self._run_guide(obs)
        
        return obs, info
    
    def step(self, action):
        """Take a step in the environment."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._episode_step += 1
        
        # If episode ended but we still have guide steps for next episode,
        # that's fine - the next reset will handle it
        
        return obs, reward, terminated, truncated, info
    
    def _run_guide(self, initial_obs: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Run the guide policy for the remaining guide steps.
        
        Args:
            initial_obs: Initial observation from env.reset()
            
        Returns:
            (final_obs, info) after guide steps
        """
        obs = initial_obs
        info = {}
        
        for i in range(self._guide_steps_remaining):
            # Get guide action
            if isinstance(self.guide_policy, BaseAlgorithm):
                action, _ = self.guide_policy.predict(
                    obs, deterministic=self.deterministic_guide
                )
            elif callable(self.guide_policy):
                action = self.guide_policy(obs)
            else:
                raise TypeError(f"Unsupported guide policy type: {type(self.guide_policy)}")
            
            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._guide_actions_taken += 1
            self._episode_step += 1
            
            # Store guide rewards for logging
            self._episode_guide_rewards.append(reward)
            
            if terminated or truncated:
                # Episode ended during guide phase; reset and continue
                obs, info = self.env.reset()
                self._episode_guide_rewards = []
        
        self._guide_steps_remaining = 0
        return obs, info
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """Get statistics for the current/last episode."""
        return {
            "guide_steps_taken": self._guide_actions_taken,
            "total_guide_steps_planned": self._total_guide_steps_this_episode,
            "episode_step": self._episode_step,
            "guide_reward_sum": sum(self._episode_guide_rewards) if self._episode_guide_rewards else 0.0,
        }


class JSRLCallback(BaseCallback):
    """
    Stable-Baselines3 callback for JSRL training.
    
    Updates the curriculum schedule, logs metrics, and periodically
    evaluates the student policy.
    """
    
    def __init__(
        self,
        logger: Logger,
        curriculum: JSCurriculum,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 10,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self._logger = logger
        self.curriculum = curriculum
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        
        self._last_eval_step = 0
        self._episode_count = 0
        self._episode_rewards = []
    
    def _on_step(self) -> bool:
        """Called at each step of training."""
        # Update curriculum
        guide_steps = self.curriculum.update(self.num_timesteps)
        
        # Log curriculum info
        self._logger.log("jsrl/guide_steps", guide_steps, self.num_timesteps)
        self._logger.log("jsrl/progress", 
                         min(1.0, self.num_timesteps / max(1, self.curriculum.total_timesteps)),
                         self.num_timesteps)
        
        # Track episode rewards from info dicts
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._episode_rewards.append(info["episode"]["r"])
                self._episode_count += 1
                self._logger.log("jsrl/episode_reward", info["episode"]["r"], self.num_timesteps)
                self._logger.log("jsrl/episode_length", info["episode"]["l"], self.num_timesteps)
        
        # Periodic evaluation
        if (self.eval_env is not None and 
            self.num_timesteps - self._last_eval_step >= self.eval_freq):
            self._run_evaluation()
            self._last_eval_step = self.num_timesteps
        
        return True
    
    def _run_evaluation(self) -> None:
        """Evaluate the current student policy."""
        if self.eval_env is None:
            return
        
        # Temporarily set model to eval mode
        self.model.policy.set_training_mode(False)
        
        eval_result = evaluate_policy(
            self.eval_env,
            self.model,
            n_episodes=self.n_eval_episodes,
            deterministic=True,
        )
        
        self.model.policy.set_training_mode(True)
        
        self._logger.log("jsrl/eval/mean_reward", eval_result["mean_reward"], self.num_timesteps)
        self._logger.log("jsrl/eval/std_reward", eval_result["std_reward"], self.num_timesteps)
        
        if self.verbose > 0:
            print(f"[JSRL] Step {self.num_timesteps}: "
                  f"Guide steps={self.curriculum.get_guide_steps()}, "
                  f"Eval reward={eval_result['mean_reward']:.2f} ± {eval_result['std_reward']:.2f}")
    
    def _on_training_end(self) -> None:
        """Called when training ends."""
        if self.eval_env is not None:
            self._run_evaluation()


def create_jsrl_model(
    observation_space: gym.Space,
    action_space: gym.Space,
    guide_policy: Any,
    config: Optional[Dict] = None,
    device: str = "auto",
) -> PPO:
    """
    Create a PPO model for JSRL student policy.
    
    The student policy can be initialized from the guide policy weights
    or trained from scratch.
    
    Args:
        observation_space: Environment observation space
        action_space: Environment action space
        guide_policy: Pre-trained guide policy (for architecture reference)
        config: Configuration dictionary
        device: Torch device
        
    Returns:
        PPO model for the student policy
    """
    if config is None:
        config = {}
    
    agent_config = config.get("agent", {})
    jsrl_config = config.get("baselines", {}).get("jsrl", {})
    
    # Build policy kwargs matching guide policy architecture
    policy_kwargs = agent_config.get("policy_kwargs", {})
    if not policy_kwargs:
        # Default: same as guide
        if hasattr(guide_policy, "policy"):
            guide_arch = guide_policy.policy
            if hasattr(guide_arch, "net_arch"):
                policy_kwargs["net_arch"] = guide_arch.net_arch
            else:
                policy_kwargs["net_arch"] = dict(pi=[64, 64], vf=[64, 64])
        else:
            policy_kwargs["net_arch"] = dict(pi=[64, 64], vf=[64, 64])
    
    model = PPO(
        "MlpPolicy",
        DummyVecEnv([lambda: gym.spaces.utils.flatten_space(observation_space)]),  # Temporary
        learning_rate=jsrl_config.get("learning_rate", agent_config.get("learning_rate", 3e-4)),
        n_steps=jsrl_config.get("n_steps", agent_config.get("n_steps", 2048)),
        batch_size=jsrl_config.get("batch_size", agent_config.get("batch_size", 64)),
        n_epochs=jsrl_config.get("n_epochs", agent_config.get("n_epochs", 10)),
        gamma=jsrl_config.get("gamma", agent_config.get("gamma", 0.99)),
        gae_lambda=jsrl_config.get("gae_lambda", agent_config.get("gae_lambda", 0.95)),
        clip_range=jsrl_config.get("clip_range", agent_config.get("clip_range", 0.2)),
        ent_coef=jsrl_config.get("ent_coef", agent_config.get("ent_coef", 0.0)),
        vf_coef=jsrl_config.get("vf_coef", agent_config.get("vf_coef", 0.5)),
        max_grad_norm=jsrl_config.get("max_grad_norm", agent_config.get("max_grad_norm", 0.5)),
        policy_kwargs=policy_kwargs,
        device=get_device(device),
        verbose=0,
    )
    
    return model


def train_jsrl(
    env_id: str,
    guide_policy: Any,
    config: Optional[Dict] = None,
    output_dir: Optional[str] = None,
    seed: int = 0,
    total_timesteps: Optional[int] = None,
    initial_guide_steps: Optional[int] = None,
    final_guide_steps: Optional[int] = None,
    schedule_type: str = "linear",
    decay_start_fraction: float = 0.0,
    deterministic_guide: bool = False,
    learning_rate: Optional[float] = None,
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
    device: str = "auto",
    verbose: int = 1,
    save_freq: int = 100000,
    resume_from: Optional[str] = None,
    **env_kwargs,
) -> Tuple[PPO, Logger, str]:
    """
    Train a student policy using Jump-Start RL.
    
    Args:
        env_id: Gym environment ID
        guide_policy: Pre-trained guide policy
        config: Configuration dictionary
        output_dir: Directory to save outputs
        seed: Random seed
        total_timesteps: Total training timesteps
        initial_guide_steps: Starting number of guide steps
        final_guide_steps: Final number of guide steps
        schedule_type: Curriculum schedule type
        decay_start_fraction: When to start decaying guide steps
        deterministic_guide: Whether guide acts deterministically
        learning_rate: PPO learning rate
        n_steps: PPO n_steps
        batch_size: PPO batch size
        n_epochs: PPO n_epochs
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_range: PPO clip range
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Max gradient norm
        eval_freq: Evaluation frequency
        n_eval_episodes: Number of evaluation episodes
        device: Torch device
        verbose: Verbosity level
        save_freq: Model saving frequency
        resume_from: Path to resume training from
        **env_kwargs: Additional environment arguments
        
    Returns:
        (trained_model, logger, model_save_path)
    """
    # Load config
    if config is None:
        config = load_config()
    
    jsrl_config = config.get("baselines", {}).get("jsrl", {})
    agent_config = config.get("agent", {})
    
    # Set defaults from config
    if total_timesteps is None:
        total_timesteps = jsrl_config.get("total_timesteps", agent_config.get("total_timesteps", 1_000_000))
    if initial_guide_steps is None:
        initial_guide_steps = jsrl_config.get("initial_guide_steps", 100)
    if final_guide_steps is None:
        final_guide_steps = jsrl_config.get("final_guide_steps", 0)
    
    # Setup
    set_seed(seed)
    device = get_device(device)
    
    if output_dir is None:
        output_dir = os.path.join(get_project_root(), "outputs", "jsrl", env_id)
    output_dir = ensure_dir(output_dir)
    
    logger = Logger(log_dir=os.path.join(output_dir, "logs"))
    
    # Create curriculum
    curriculum = JSCurriculum(
        initial_guide_steps=initial_guide_steps,
        final_guide_steps=final_guide_steps,
        total_timesteps=total_timesteps,
        schedule_type=schedule_type,
        decay_start_fraction=decay_start_fraction,
    )
    
    # Create base environment
    base_env = make_env(env_id, seed=seed, **env_kwargs)
    
    # Wrap with JSRL wrapper
    jsrl_env = JSRLEnvWrapper(
        base_env,
        guide_policy=guide_policy,
        curriculum=curriculum,
        deterministic_guide=deterministic_guide,
        device=device,
    )
    
    # Wrap for monitoring and vectorization
    jsrl_env = Monitor(jsrl_env)
    vec_env = DummyVecEnv([lambda: jsrl_env])
    
    # Create evaluation environment (without guide)
    eval_env = make_env(env_id, seed=seed + 1000, **env_kwargs)
    eval_env = Monitor(eval_env)
    
    # Create or load model
    if resume_from and os.path.exists(resume_from):
        if verbose > 0:
            print(f"[JSRL] Resuming from {resume_from}")
        model = PPO.load(resume_from, env=vec_env, device=device)
    else:
        # Build policy kwargs
        policy_kwargs = agent_config.get("policy_kwargs", {})
        if not policy_kwargs:
            policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
        
        # Use learning rate from args or config
        lr = learning_rate if learning_rate is not None else jsrl_config.get(
            "learning_rate", agent_config.get("learning_rate", 3e-4)
        )
        
        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=lr,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=0,
        )
    
    # Create callback
    callback = JSRLCallback(
        logger=logger,
        curriculum=curriculum,
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        verbose=verbose,
    )
    
    # Train
    if verbose > 0:
        print(f"[JSRL] Starting training on {env_id}")
        print(f"  Total timesteps: {total_timesteps}")
        print(f"  Initial guide steps: {initial_guide_steps}")
        print(f"  Final guide steps: {final_guide_steps}")
        print(f"  Schedule: {schedule_type}")
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=verbose > 0,
        )
    except KeyboardInterrupt:
        if verbose > 0:
            print("\n[JSRL] Training interrupted, saving model...")
    
    elapsed = time.time() - start_time
    
    # Save model
    model_path = os.path.join(output_dir, "jsrl_model")
    model.save(model_path)
    
    # Save curriculum state
    curriculum.save(os.path.join(output_dir, "curriculum.json"))
    
    # Save config
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump({
            "env_id": env_id,
            "seed": seed,
            "total_timesteps": total_timesteps,
            "initial_guide_steps": initial_guide_steps,
            "final_guide_steps": final_guide_steps,
            "schedule_type": schedule_type,
            "elapsed_time": elapsed,
            "elapsed_time_formatted": format_time(elapsed),
        }, f, indent=2)
    
    # Final evaluation
    final_eval = evaluate_policy(eval_env, model, n_episodes=n_eval_episodes, deterministic=True)
    
    if verbose > 0:
        print(f"[JSRL] Training completed in {format_time(elapsed)}")
        print(f"  Final eval reward: {final_eval['mean_reward']:.2f} ± {final_eval['std_reward']:.2f}")
        print(f"  Model saved to: {model_path}.zip")
    
    # Save final results
    results = {
        "env_id": env_id,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "elapsed_time": elapsed,
        "final_mean_reward": final_eval["mean_reward"],
        "final_std_reward": final_eval["std_reward"],
        "final_min_reward": final_eval.get("min_reward", None),
        "final_max_reward": final_eval.get("max_reward", None),
        "curriculum_stats": curriculum.get_stats(),
    }
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Clean up
    vec_env.close()
    eval_env.close()
    
    return model, logger, model_path


def run_jsrl_pipeline(
    env_id: str,
    agent_path: str,
    config: Optional[Dict] = None,
    output_dir: Optional[str] = None,
    seed: int = 0,
    total_timesteps: Optional[int] = None,
    initial_guide_steps: Optional[int] = None,
    final_guide_steps: Optional[int] = None,
    schedule_type: str = "linear",
    decay_start_fraction: float = 0.0,
    deterministic_guide: bool = False,
    eval_freq: int = 10000,
    n_eval_episodes: int = 10,
    device: str = "auto",
    verbose: int = 1,
    **env_kwargs,
) -> Dict[str, Any]:
    """
    Run the full JSRL pipeline: load guide agent, train student, evaluate.
    
    This is the main entry point for using JSRL as a baseline in RICE experiments.
    
    Args:
        env_id: Gym environment ID
        agent_path: Path to pre-trained agent (guide policy)
        config: Configuration dictionary
        output_dir: Output directory
        seed: Random seed
        total_timesteps: Total training timesteps
        initial_guide_steps: Starting guide steps
        final_guide_steps: Final guide steps
        schedule_type: Curriculum schedule type
        decay_start_fraction: When to start decay
        deterministic_guide: Whether guide is deterministic
        eval_freq: Evaluation frequency
        n_eval_episodes: Number of eval episodes
        device: Torch device
        verbose: Verbosity
        **env_kwargs: Additional env kwargs
        
    Returns:
        Dictionary with results and paths
    """
    # Load config
    if config is None:
        config = load_config(env_id)
    
    if output_dir is None:
        output_dir = os.path.join(
            get_project_root(), "outputs", "jsrl", env_id, f"seed_{seed}"
        )
    output_dir = ensure_dir(output_dir)
    
    # Load guide policy
    if verbose > 0:
        print(f"[JSRL Pipeline] Loading guide policy from {agent_path}")
    
    guide_policy = PPO.load(agent_path, device=get_device(device))
    
    # Train JSRL
    model, logger, model_path = train_jsrl(
        env_id=env_id,
        guide_policy=guide_policy,
        config=config,
        output_dir=output_dir,
        seed=seed,
        total_timesteps=total_timesteps,
        initial_guide_steps=initial_guide_steps,
        final_guide_steps=final_guide_steps,
        schedule_type=schedule_type,
        decay_start_fraction=decay_start_fraction,
        deterministic_guide=deterministic_guide,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        device=device,
        verbose=verbose,
        **env_kwargs,
    )
    
    # Load results
    results_path = os.path.join(output_dir, "results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
    else:
        results = {}
    
    results["model_path"] = model_path + ".zip"
    results["output_dir"] = output_dir
    
    return results


def main():
    """CLI entry point for JSRL baseline training."""
    parser = argparse.ArgumentParser(
        description="Jump-Start RL Baseline Training"
    )
    
    # Required arguments
    parser.add_argument("--env-id", type=str, required=True,
                        help="Gym environment ID")
    parser.add_argument("--agent-path", type=str, required=True,
                        help="Path to pre-trained guide agent (.zip)")
    
    # Output
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    
    # Training
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Total training timesteps")
    parser.add_argument("--initial-guide-steps", type=int, default=None,
                        help="Initial number of guide steps")
    parser.add_argument("--final-guide-steps", type=int, default=0,
                        help="Final number of guide steps")
    parser.add_argument("--schedule-type", type=str, default="linear",
                        choices=["linear", "exponential", "step"],
                        help="Curriculum schedule type")
    parser.add_argument("--decay-start-fraction", type=float, default=0.0,
                        help="Fraction of training before decay starts")
    parser.add_argument("--deterministic-guide", action="store_true",
                        help="Use deterministic guide policy")
    
    # PPO hyperparameters
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="PPO learning rate")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="PPO n_steps")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="PPO batch size")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="PPO n_epochs")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="GAE lambda")
    parser.add_argument("--clip-range", type=float, default=0.2,
                        help="PPO clip range")
    parser.add_argument("--ent-coef", type=float, default=0.0,
                        help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="Value function coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="Max gradient norm")
    
    # Evaluation
    parser.add_argument("--eval-freq", type=int, default=10000,
                        help="Evaluation frequency")
    parser.add_argument("--n-eval-episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    
    # Other
    parser.add_argument("--device", type=str, default="auto",
                        help="Torch device")
    parser.add_argument("--verbose", type=int, default=1,
                        help="Verbosity level")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Resume training from checkpoint")
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        with open(args.config, "r") as f:
            import yaml
            config = yaml.safe_load(f)
    else:
        config = load_config(args.env_id)
    
    # Run pipeline
    results = run_jsrl_pipeline(
        env_id=args.env_id,
        agent_path=args.agent_path,
        config=config,
        output_dir=args.output_dir,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        initial_guide_steps=args.initial_guide_steps,
        final_guide_steps=args.final_guide_steps,
        schedule_type=args.schedule_type,
        decay_start_fraction=args.decay_start_fraction,
        deterministic_guide=args.deterministic_guide,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        device=args.device,
        verbose=args.verbose,
    )
    
    print(f"\n[JSRL] Pipeline complete!")
    print(f"  Results: {json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()