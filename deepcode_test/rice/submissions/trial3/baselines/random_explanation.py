"""
Random Explanation Baseline for RICE

This baseline replaces the trained mask network's importance scores with
uniformly random scores. It follows the same critical state collection and
refining pipeline as RICE, but uses random importance scores instead of
learned ones. This serves as a negative control to demonstrate that the
mask network's learned explanations are essential for effective refinement.

Paper Reference:
    Section 5.2 (Ablation Studies): "Random Explanation" baseline replaces
    the mask network with random importance scores. Expected: random
    explanation degrades performance compared to learned mask network.

Usage:
    from baselines.random_explanation import RandomExplanation, run_random_refine
    
    # Create random explanation baseline
    random_exp = RandomExplanation(state_dim=11)
    
    # Or run full random refine pipeline
    results = run_random_refine(
        env_name="Hopper-v4",
        model_dir="./trained_agents",
        output_dir="./random_refine_results",
        ...
    )
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List, Callable
import os
import pickle
import json
import time
from pathlib import Path


class RandomExplanation:
    """
    Generates random importance scores for states, mimicking the mask network
    interface but without any learning. Used as a baseline to validate that
    the mask network's learned explanations are meaningful.
    
    The random scores are drawn from a uniform distribution [0, 1] and can be
    optionally fixed per state (deterministic) or re-sampled each time.
    """
    
    def __init__(
        self,
        state_dim: int,
        seed: int = 42,
        deterministic: bool = True,
        device: str = "cpu"
    ):
        """
        Args:
            state_dim: Dimension of the state space.
            seed: Random seed for reproducibility.
            deterministic: If True, uses a hash-based deterministic mapping
                          from state to score (same state → same score).
                          If False, samples fresh random score each time.
            device: Device for tensor operations.
        """
        self.state_dim = state_dim
        self.seed = seed
        self.deterministic = deterministic
        self.device = device
        
        # Internal RNG for reproducibility
        self._rng = np.random.RandomState(seed)
        
        # Cache for deterministic mode: maps state hash to score
        self._score_cache: Dict[int, float] = {}
        
    def get_importance_score(self, state: np.ndarray) -> float:
        """
        Compute a random importance score for a given state.
        
        Args:
            state: State vector of shape (state_dim,) or (batch, state_dim).
            
        Returns:
            Random importance score in [0, 1] (scalar or array).
        """
        state = np.asarray(state)
        
        if state.ndim == 1:
            return self._get_single_score(state)
        else:
            return np.array([self._get_single_score(s) for s in state])
    
    def _get_single_score(self, state: np.ndarray) -> float:
        """Get random score for a single state."""
        if self.deterministic:
            # Use hash of state bytes for deterministic mapping
            state_hash = hash(state.tobytes())
            if state_hash not in self._score_cache:
                # Use a secondary RNG seeded by the hash for consistency
                local_rng = np.random.RandomState(abs(state_hash) % (2**31))
                self._score_cache[state_hash] = float(local_rng.uniform(0.0, 1.0))
            return self._score_cache[state_hash]
        else:
            return float(self._rng.uniform(0.0, 1.0))
    
    def get_importance_scores_batch(
        self, 
        states: np.ndarray,
        device: str = "cpu"
    ) -> np.ndarray:
        """
        Compute random importance scores for a batch of states.
        
        Args:
            states: Array of shape (N, state_dim).
            device: Device (unused, kept for interface compatibility).
            
        Returns:
            Array of shape (N,) with random scores in [0, 1].
        """
        states = np.asarray(states)
        if states.ndim == 1:
            states = states.reshape(1, -1)
        
        scores = np.array([self._get_single_score(s) for s in states])
        return scores
    
    def to(self, device: str):
        """No-op for interface compatibility with MaskNetwork."""
        self.device = device
        return self
    
    def eval(self):
        """No-op for interface compatibility."""
        return self
    
    def train(self):
        """No-op for interface compatibility."""
        return self
    
    def state_dict(self) -> Dict[str, Any]:
        """Return a serializable state dict (for saving)."""
        return {
            "state_dim": self.state_dim,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "score_cache_size": len(self._score_cache),
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state (for interface compatibility)."""
        self.state_dim = state_dict.get("state_dim", self.state_dim)
        self.seed = state_dict.get("seed", self.seed)
        self.deterministic = state_dict.get("deterministic", self.deterministic)
    
    def save(self, path: str):
        """Save the random explanation configuration."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.state_dict(), f)
    
    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "RandomExplanation":
        """Load a random explanation from disk."""
        with open(path, "rb") as f:
            state_dict = pickle.load(f)
        instance = cls(
            state_dim=state_dict["state_dim"],
            seed=state_dict.get("seed", 42),
            deterministic=state_dict.get("deterministic", True),
            device=device
        )
        return instance


def collect_critical_states_random(
    env,
    target_policy_fn: Callable[[np.ndarray], np.ndarray],
    random_explanation: RandomExplanation,
    num_episodes: int = 100,
    top_k_per_episode: int = 1,
    max_episode_steps: int = 1000,
    verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], List[float]]:
    """
    Collect critical states using random importance scores instead of
    a trained mask network.
    
    For each episode:
      1. Run the target policy to collect states.
      2. Compute random importance scores for each state.
      3. Select the top-k states with highest random scores as "critical".
      4. Save the full environment state for each selected critical state.
    
    Args:
        env: The environment (should support state save/restore).
        target_policy_fn: Function mapping state -> action.
        random_explanation: RandomExplanation instance.
        num_episodes: Number of episodes to collect.
        top_k_per_episode: Number of critical states to select per episode.
        max_episode_steps: Maximum steps per episode.
        verbose: Whether to print progress.
        
    Returns:
        Tuple of (critical_states, importance_scores):
            critical_states: List of dicts with 'env_state', 'obs', 'score', 'episode'.
            importance_scores: List of all importance scores collected.
    """
    from rice.env_wrappers import save_env_state
    
    critical_states = []
    all_scores = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        step = 0
        
        episode_states = []
        episode_scores = []
        
        while not (done or truncated) and step < max_episode_steps:
            # Save environment state
            env_state = save_env_state(env)
            
            # Get random importance score
            score = random_explanation.get_importance_score(obs)
            
            episode_states.append({
                "env_state": env_state,
                "obs": obs.copy(),
                "score": score,
                "episode": ep,
                "step": step,
            })
            episode_scores.append(score)
            
            # Take action using target policy
            action = target_policy_fn(obs)
            if isinstance(action, tuple):
                action = action[0]  # Handle (action, ...) tuples
            
            # Step environment
            result = env.step(action)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                obs, reward, done, info = result
                truncated = False
            
            step += 1
        
        # Select top-k states by random score
        if len(episode_states) > 0:
            episode_states.sort(key=lambda x: x["score"], reverse=True)
            for i in range(min(top_k_per_episode, len(episode_states))):
                critical_states.append(episode_states[i])
        
        all_scores.extend(episode_scores)
        
        if verbose and (ep + 1) % 10 == 0:
            print(f"  [Random Explanation] Collected {ep + 1}/{num_episodes} episodes, "
                  f"{len(critical_states)} critical states so far")
    
    if verbose:
        print(f"  [Random Explanation] Total: {len(critical_states)} critical states "
              f"from {num_episodes} episodes")
        print(f"  [Random Explanation] Mean random score: {np.mean(all_scores):.4f} "
              f"(std: {np.std(all_scores):.4f})")
    
    return critical_states, all_scores


def run_random_refine(
    env_name: str,
    model_dir: str,
    output_dir: str,
    config_path: Optional[str] = None,
    total_steps: int = 1_000_000,
    p_mixed: float = 0.25,
    lambda_rnd: float = 0.01,
    seed: int = 42,
    device: str = "cuda",
    num_critical_episodes: int = 100,
    top_k_per_episode: int = 1,
    verbose: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Run the full random explanation baseline: collect critical states using
    random importance scores, then refine the policy using the same RICE
    pipeline (mixed initial distribution + RND exploration bonus).
    
    This function mirrors `run_refine` from `experiments/mujoco/refine.py`
    but replaces the mask network with a RandomExplanation.
    
    Args:
        env_name: Name of the environment (e.g., "Hopper-v4").
        model_dir: Directory containing the pre-trained target agent.
        output_dir: Directory to save results.
        config_path: Optional path to a YAML config file.
        total_steps: Total training steps for refining.
        p_mixed: Probability of resetting to a critical state.
        lambda_rnd: RND exploration bonus coefficient.
        seed: Random seed.
        device: Device for computation.
        num_critical_episodes: Number of episodes for critical state collection.
        top_k_per_episode: Number of critical states per episode.
        verbose: Whether to print progress.
        **kwargs: Additional arguments passed to the refine function.
        
    Returns:
        Dictionary with results: refined_policy, history, eval_rewards, config.
    """
    import gym
    import yaml
    import torch
    
    from rice.utils import set_seed, evaluate_policy
    from rice.env_wrappers import make_state_saveable, StateSaveWrapper
    from rice.refine import RICERefine, refine_policy
    
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configuration
    config = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    
    # Determine state_dim from environment
    temp_env = gym.make(env_name)
    if hasattr(temp_env.observation_space, 'shape'):
        state_dim = temp_env.observation_space.shape[0]
    else:
        state_dim = temp_env.observation_space.n
    if hasattr(temp_env.action_space, 'shape'):
        action_dim = temp_env.action_space.shape[0]
        discrete_action = False
        num_discrete_actions = None
    else:
        action_dim = temp_env.action_space.n
        discrete_action = True
        num_discrete_actions = action_dim
    temp_env.close()
    
    # Create random explanation
    random_exp = RandomExplanation(
        state_dim=state_dim,
        seed=seed,
        deterministic=True,
        device=device
    )
    
    # Load target policy
    try:
        from stable_baselines3 import PPO
        model_path = os.path.join(model_dir, f"{env_name}_ppo_final.zip")
        if os.path.exists(model_path):
            model = PPO.load(model_path, device=device)
        else:
            # Try alternative path
            model_path = os.path.join(model_dir, "ppo_final.zip")
            model = PPO.load(model_path, device=device)
        
        # Load VecNormalize if available
        vec_normalize = None
        vn_path = os.path.join(model_dir, f"{env_name}_vecnormalize.pkl")
        if os.path.exists(vn_path):
            with open(vn_path, "rb") as f:
                vec_normalize = pickle.load(f)
        
        def target_policy_fn(obs: np.ndarray) -> np.ndarray:
            """Deterministic target policy function."""
            if vec_normalize is not None:
                obs = vec_normalize.normalize_obs(obs)
            action, _ = model.predict(obs, deterministic=True)
            return action
        
    except ImportError:
        # Fallback: try loading a PyTorch state dict
        policy_path = os.path.join(model_dir, f"{env_name}_policy.pt")
        if os.path.exists(policy_path):
            policy_state = torch.load(policy_path, map_location=device)
            # Build a simple MLP policy
            policy_net = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
            )
            policy_net.load_state_dict(policy_state)
            policy_net.to(device)
            policy_net.eval()
            
            def target_policy_fn(obs: np.ndarray) -> np.ndarray:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                    if obs_t.ndim == 1:
                        obs_t = obs_t.unsqueeze(0)
                    action = policy_net(obs_t).cpu().numpy()
                    if action.shape[0] == 1:
                        action = action[0]
                    return action
        else:
            raise FileNotFoundError(f"No target policy found in {model_dir}")
    
    # Create environment with state save wrapper
    env = gym.make(env_name)
    env = make_state_saveable(env)
    
    # Collect critical states using random scores
    if verbose:
        print(f"\n{'='*60}")
        print(f"Random Explanation Baseline: {env_name}")
        print(f"{'='*60}")
        print(f"Collecting critical states with random scores...")
    
    start_time = time.time()
    critical_states, all_scores = collect_critical_states_random(
        env=env,
        target_policy_fn=target_policy_fn,
        random_explanation=random_exp,
        num_episodes=num_critical_episodes,
        top_k_per_episode=top_k_per_episode,
        verbose=verbose,
    )
    collection_time = time.time() - start_time
    
    if verbose:
        print(f"Critical state collection completed in {collection_time:.1f}s")
    
    # Save critical states
    critical_states_path = os.path.join(output_dir, "random_critical_states.pkl")
    with open(critical_states_path, "wb") as f:
        pickle.dump(critical_states, f)
    
    # Save random explanation
    random_exp_path = os.path.join(output_dir, "random_explanation.pkl")
    random_exp.save(random_exp_path)
    
    # Run refining using the same pipeline as RICE
    # We need to create a dummy mask network that returns random scores
    # for the RICERefine class to use.
    # Actually, we'll modify the approach: directly use refine_policy
    # but override the critical state collection.
    
    if verbose:
        print(f"\nStarting refining with random explanation...")
        print(f"  p_mixed: {p_mixed}, lambda_rnd: {lambda_rnd}")
        print(f"  Total steps: {total_steps}")
    
    # Create a wrapper mask network that returns random scores
    class RandomMaskWrapper:
        """Wraps RandomExplanation to match MaskNetwork interface."""
        def __init__(self, random_exp: RandomExplanation):
            self.random_exp = random_exp
        
        def get_importance_score(self, state):
            if isinstance(state, torch.Tensor):
                state = state.cpu().numpy()
            return self.random_exp.get_importance_score(state)
        
        def to(self, device):
            return self
        
        def eval(self):
            return self
        
        def state_dict(self):
            return self.random_exp.state_dict()
        
        def load_state_dict(self, d):
            self.random_exp.load_state_dict(d)
    
    random_mask = RandomMaskWrapper(random_exp)
    
    # Run refine
    refined_policy, history = refine_policy(
        env=env,
        target_policy=target_policy_fn,
        mask_network=random_mask,
        state_dim=state_dim,
        action_dim=action_dim,
        discrete_action=discrete_action,
        num_discrete_actions=num_discrete_actions,
        device=device,
        p_mixed=p_mixed,
        lambda_rnd=lambda_rnd,
        total_steps=total_steps,
        num_critical_episodes=num_critical_episodes,
        top_k_per_episode=top_k_per_episode,
        verbose=verbose,
        save_path=os.path.join(output_dir, "random_refined_policy.pt"),
        **kwargs
    )
    
    # Evaluate refined policy
    if verbose:
        print(f"\nEvaluating refined policy...")
    
    def refined_policy_fn(obs: np.ndarray) -> np.ndarray:
        """Deterministic refined policy function."""
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            if obs_t.ndim == 1:
                obs_t = obs_t.unsqueeze(0)
            if hasattr(refined_policy, 'get_action'):
                action = refined_policy.get_action(obs_t, deterministic=True)
                if isinstance(action, tuple):
                    action = action[0]
            else:
                action = refined_policy(obs_t)
            action = action.cpu().numpy()
            if action.ndim == 2 and action.shape[0] == 1:
                action = action[0]
            return action
    
    eval_results = evaluate_policy(
        env=env,
        policy_fn=refined_policy_fn,
        num_episodes=100,
        deterministic=True,
        verbose=verbose,
    )
    
    # Also evaluate original target policy for comparison
    target_eval = evaluate_policy(
        env=env,
        policy_fn=target_policy_fn,
        num_episodes=100,
        deterministic=True,
        verbose=False,
    )
    
    # Compile results
    results = {
        "env_name": env_name,
        "baseline": "random_explanation",
        "seed": seed,
        "p_mixed": p_mixed,
        "lambda_rnd": lambda_rnd,
        "total_steps": total_steps,
        "num_critical_episodes": num_critical_episodes,
        "top_k_per_episode": top_k_per_episode,
        "collection_time": collection_time,
        "num_critical_states": len(critical_states),
        "mean_random_score": float(np.mean(all_scores)),
        "std_random_score": float(np.std(all_scores)),
        "target_eval": {
            "mean_reward": float(target_eval["mean_reward"]),
            "std_reward": float(target_eval["std_reward"]),
        },
        "refined_eval": {
            "mean_reward": float(eval_results["mean_reward"]),
            "std_reward": float(eval_results["std_reward"]),
        },
        "improvement": float(eval_results["mean_reward"] - target_eval["mean_reward"]),
        "improvement_pct": float(
            (eval_results["mean_reward"] - target_eval["mean_reward"]) 
            / max(abs(target_eval["mean_reward"]), 1e-8) * 100
        ),
        "history": history,
    }
    
    # Save results
    results_path = os.path.join(output_dir, "random_refine_results.json")
    # Convert non-serializable items
    serializable_results = {}
    for k, v in results.items():
        if k == "history":
            serializable_results[k] = v  # history is already a list of dicts
        elif isinstance(v, (int, float, str, bool, list, dict, type(None))):
            serializable_results[k] = v
        else:
            serializable_results[k] = str(v)
    
    with open(results_path, "w") as f:
        json.dump(serializable_results, f, indent=2, default=str)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Random Explanation Baseline Results for {env_name}")
        print(f"{'='*60}")
        print(f"  Target policy mean reward:    {target_eval['mean_reward']:.2f} ± {target_eval['std_reward']:.2f}")
        print(f"  Refined policy mean reward:   {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
        print(f"  Improvement:                  {results['improvement']:.2f} ({results['improvement_pct']:.1f}%)")
        print(f"  Results saved to: {results_path}")
    
    env.close()
    
    return results


# ============================================================================
# Command-line Interface
# ============================================================================

def parse_args():
    """Parse command-line arguments for the random explanation baseline."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Random Explanation Baseline for RICE"
    )
    parser.add_argument(
        "--env", type=str, default="Hopper-v4",
        help="Environment name (default: Hopper-v4)"
    )
    parser.add_argument(
        "--model-dir", type=str, default="./trained_agents",
        help="Directory containing pre-trained target agent"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./random_refine_results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--total-steps", type=int, default=1_000_000,
        help="Total training steps for refining"
    )
    parser.add_argument(
        "--p-mixed", type=float, default=0.25,
        help="Probability of resetting to a critical state"
    )
    parser.add_argument(
        "--lambda-rnd", type=float, default=0.01,
        help="RND exploration bonus coefficient"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device: 'cuda' or 'cpu'"
    )
    parser.add_argument(
        "--num-critical-episodes", type=int, default=100,
        help="Number of episodes for critical state collection"
    )
    parser.add_argument(
        "--top-k", type=int, default=1,
        help="Number of critical states per episode"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output"
    )
    return parser.parse_args()


def main():
    """Main entry point for the random explanation baseline."""
    args = parse_args()
    
    results = run_random_refine(
        env_name=args.env,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        total_steps=args.total_steps,
        p_mixed=args.p_mixed,
        lambda_rnd=args.lambda_rnd,
        seed=args.seed,
        device=args.device,
        num_critical_episodes=args.num_critical_episodes,
        top_k_per_episode=args.top_k,
        verbose=not args.quiet,
    )
    
    return results


if __name__ == "__main__":
    main()