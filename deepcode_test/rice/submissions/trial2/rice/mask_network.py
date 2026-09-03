"""
Mask Network Module for RICE (Refining via Critical State Explanation).

This module implements the mask network ξ(s) that learns a binary policy:
- aᵉ = 0: take the agent's action (trust the agent)
- aᵉ = 1: take a random action (blind/override the agent)

The mask network is trained via PPO on a PerturbedEnv to maximize the perturbed
policy's performance while receiving an intrinsic reward for blinding (α bonus).

Architecture: MLP with same hidden structure as the target agent's policy network.
Input: state vector s. Output: 2 logits for binary action (Discrete(2)).

Training Algorithm (simplified from StateMask):
1. Wrap the environment with PerturbedEnv (handles action selection and reward augmentation)
2. Train mask network as a PPO agent with Discrete(2) action space
3. Reward: r_mask = r_env + α * I(aᵉ = 1), where α = 0.0001
4. Update ξ using PPO clipped loss on cumulative discounted r_mask

Reference: Paper Section 3.2, Algorithm 1
"""

import os
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from rice.perturbed_env import PerturbedEnv, make_perturbed_env, make_perturbed_vec_env
from rice.utils import (
    build_mlp,
    ensure_dir,
    evaluate_policy,
    get_device,
    init_weights,
    load_config,
    Logger,
    set_seed,
)


class MaskFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor for the mask network.
    
    Uses an MLP with the same hidden structure as the agent's policy network.
    For MuJoCo: [64, 64]; for other envs: configurable.
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 64,
        hidden_sizes: Tuple[int, ...] = (64, 64),
        activation_fn: nn.Module = nn.ReLU,
        use_layer_norm: bool = False,
    ):
        super().__init__(observation_space, features_dim)
        
        # Determine input dimension from observation space
        if isinstance(observation_space, gym.spaces.Box):
            input_dim = int(np.prod(observation_space.shape))
        elif isinstance(observation_space, gym.spaces.Discrete):
            input_dim = int(observation_space.n)
        else:
            raise ValueError(f"Unsupported observation space: {type(observation_space)}")
        
        # Build MLP: input_dim -> hidden_sizes -> features_dim
        layers = []
        prev_dim = input_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_size))
            layers.append(activation_fn())
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            prev_dim = hidden_size
        
        layers.append(nn.Linear(prev_dim, features_dim))
        layers.append(activation_fn())
        
        self.mlp = nn.Sequential(*layers)
        
        # Apply orthogonal initialization
        self.apply(lambda m: init_weights(m, gain=np.sqrt(2)))
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.mlp(observations)


class MaskPolicyNetwork(ActorCriticPolicy):
    """
    Actor-Critic policy for the mask network.
    
    Extends SB3's ActorCriticPolicy to use a custom feature extractor
    and produce 2 logits for the binary mask action (Discrete(2)).
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule: Callable,
        net_arch: Optional[Union[Dict[str, Any], list]] = None,
        activation_fn: type = nn.ReLU,
        features_extractor_class: type = MaskFeatureExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: type = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}
        
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            **kwargs,
        )


def create_mask_network(
    observation_space: gym.Space,
    action_space: gym.Space,
    hidden_sizes: Tuple[int, ...] = (64, 64),
    features_dim: int = 64,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.0,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    use_sde: bool = False,
    device: str = "auto",
    verbose: int = 0,
) -> PPO:
    """
    Create a PPO agent for the mask network.
    
    The mask network is a PPO agent with Discrete(2) action space.
    It uses the same MLP architecture as the target agent's policy.
    
    Args:
        observation_space: Environment observation space
        action_space: Should be Discrete(2) for binary mask action
        hidden_sizes: Hidden layer sizes for the MLP feature extractor
        features_dim: Output dimension of the feature extractor
        learning_rate: PPO learning rate
        n_steps: Number of steps per rollout
        batch_size: Minibatch size
        n_epochs: Number of epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_range: PPO clip range
        ent_coef: Entropy coefficient
        vf_coef: Value function coefficient
        max_grad_norm: Max gradient norm
        use_sde: Whether to use State Dependent Exploration
        device: Device string
        verbose: Verbosity level
    
    Returns:
        PPO agent configured for mask network training
    """
    device_obj = get_device(device)
    
    # Build policy kwargs
    policy_kwargs = {
        "features_extractor_class": MaskFeatureExtractor,
        "features_extractor_kwargs": {
            "features_dim": features_dim,
            "hidden_sizes": hidden_sizes,
            "activation_fn": nn.ReLU,
            "use_layer_norm": False,
        },
        "net_arch": [],  # No additional layers beyond feature extractor + value/policy heads
    }
    
    mask_ppo = PPO(
        policy=MaskPolicyNetwork,
        env=None,  # Will be set later via set_env or during training
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        use_sde=use_sde,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log=None,
        policy_kwargs=policy_kwargs,
        verbose=verbose,
        seed=None,
        device=device_obj,
    )
    
    return mask_ppo


class MaskTrainingCallback(BaseCallback):
    """
    Callback for mask network training that logs metrics and evaluates
    the perturbed policy performance.
    """
    
    def __init__(
        self,
        logger: Logger,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 10,
        agent_policy: Optional[Any] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.logger = logger
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.agent_policy = agent_policy
        self.start_time = time.time()
        self.last_eval_step = 0
    
    def _on_step(self) -> bool:
        # Log training metrics
        if len(self.model.ep_info_buffer) > 0:
            recent_episodes = list(self.model.ep_info_buffer)[-10:]
            if recent_episodes:
                avg_reward = np.mean([ep.get('r', 0) for ep in recent_episodes])
                avg_length = np.mean([ep.get('l', 0) for ep in recent_episodes])
                self.logger.log('mask/train_reward', avg_reward, self.num_timesteps)
                self.logger.log('mask/train_length', avg_length, self.num_timesteps)
        
        # Log losses
        if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
            for key, value in self.model.logger.name_to_value.items():
                self.logger.log(f'mask/{key}', value, self.num_timesteps)
        
        # Periodic evaluation
        if self.eval_env is not None and (self.num_timesteps - self.last_eval_step) >= self.eval_freq:
            self._evaluate()
            self.last_eval_step = self.num_timesteps
        
        return True
    
    def _evaluate(self):
        """Evaluate the perturbed policy performance."""
        if self.eval_env is None:
            return
        
        # We need to evaluate the perturbed policy: agent + mask
        # The eval_env should be a PerturbedEnvWrapper for this purpose
        try:
            total_rewards = []
            total_lengths = []
            
            for _ in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                truncated = False
                ep_reward = 0.0
                ep_length = 0
                
                while not done and not truncated:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = self.eval_env.step(action)
                    ep_reward += reward
                    ep_length += 1
                
                total_rewards.append(ep_reward)
                total_lengths.append(ep_length)
            
            mean_reward = np.mean(total_rewards)
            std_reward = np.std(total_rewards)
            mean_length = np.mean(total_lengths)
            
            self.logger.log('mask/eval_reward', mean_reward, self.num_timesteps)
            self.logger.log('mask/eval_std_reward', std_reward, self.num_timesteps)
            self.logger.log('mask/eval_length', mean_length, self.num_timesteps)
            
            if self.verbose > 0:
                elapsed = time.time() - self.start_time
                print(f"[Mask Eval] Step: {self.num_timesteps} | "
                      f"Reward: {mean_reward:.2f} ± {std_reward:.2f} | "
                      f"Length: {mean_length:.1f} | "
                      f"Time: {elapsed:.1f}s")
        except Exception as e:
            if self.verbose > 0:
                print(f"[Mask Eval] Evaluation failed: {e}")


def train_mask_network(
    env_id: str,
    agent_policy: Any,
    config: Dict[str, Any],
    output_dir: str,
    seed: int = 0,
    total_timesteps: Optional[int] = None,
    alpha: Optional[float] = None,
    hidden_sizes: Optional[Tuple[int, ...]] = None,
    learning_rate: Optional[float] = None,
    n_steps: Optional[int] = None,
    batch_size: Optional[int] = None,
    n_epochs: Optional[int] = None,
    gamma: Optional[float] = None,
    ent_coef: Optional[float] = None,
    device: str = "auto",
    eval_freq: int = 10000,
    n_eval_episodes: int = 10,
    verbose: int = 1,
    resume_from: Optional[str] = None,
    **env_kwargs,
) -> Tuple[PPO, Logger, str]:
    """
    Train the mask network ξ(s) using PPO on the PerturbedEnv.
    
    The mask network learns to decide whether to use the agent's action (aᵉ=0)
    or a random action (aᵉ=1) at each state, maximizing the perturbed policy's
    performance while receiving an intrinsic reward α for blinding.
    
    Args:
        env_id: Gymnasium environment ID
        agent_policy: Frozen agent policy π (SB3 model, callable, or nn.Module)
        config: Configuration dictionary (from load_config)
        output_dir: Directory to save model and logs
        seed: Random seed
        total_timesteps: Total training timesteps (default from config)
        alpha: Intrinsic reward coefficient for blinding (default from config)
        hidden_sizes: Hidden layer sizes for mask MLP
        learning_rate: PPO learning rate
        n_steps: Rollout steps per update
        batch_size: Minibatch size
        n_epochs: Number of epochs per update
        gamma: Discount factor
        ent_coef: Entropy coefficient
        device: Device string
        eval_freq: Evaluation frequency in timesteps
        n_eval_episodes: Number of evaluation episodes
        verbose: Verbosity level
        resume_from: Path to resume training from
        **env_kwargs: Additional environment arguments
    
    Returns:
        Tuple of (trained mask PPO model, Logger, model save path)
    """
    set_seed(seed)
    device_obj = get_device(device)
    
    # Load config defaults
    mask_config = config.get('mask', {})
    agent_config = config.get('agent', {})
    
    # Set hyperparameters with fallback chain: argument > config > default
    if total_timesteps is None:
        total_timesteps = mask_config.get('total_timesteps', 300000)
    if alpha is None:
        alpha = mask_config.get('alpha', 0.0001)
    if hidden_sizes is None:
        hidden_sizes = tuple(mask_config.get('hidden_sizes', 
                            agent_config.get('policy_kwargs', {}).get('net_arch', {}).get('pi', [64, 64])))
    if learning_rate is None:
        learning_rate = mask_config.get('learning_rate', 3e-4)
    if n_steps is None:
        n_steps = mask_config.get('n_steps', 2048)
    if batch_size is None:
        batch_size = mask_config.get('batch_size', 64)
    if n_epochs is None:
        n_epochs = mask_config.get('n_epochs', 10)
    if gamma is None:
        gamma = mask_config.get('gamma', 0.99)
    if ent_coef is None:
        ent_coef = mask_config.get('ent_coef', 0.0)
    
    ensure_dir(output_dir)
    logger = Logger(log_dir=output_dir)
    
    # Create perturbed environment for training
    # The PerturbedEnv handles: mask action -> agent/random action -> env step -> augmented reward
    train_env = make_perturbed_env(
        env_id=env_id,
        agent_policy=agent_policy,
        alpha=alpha,
        deterministic_agent=False,
        mask_network=None,  # Will be set after model creation
        device=device,
        seed=seed,
        **env_kwargs,
    )
    
    # Create evaluation environment (separate instance)
    eval_env = make_perturbed_env(
        env_id=env_id,
        agent_policy=agent_policy,
        alpha=alpha,
        deterministic_agent=True,
        mask_network=None,
        device=device,
        seed=seed + 1000,  # Different seed for eval
        **env_kwargs,
    )
    
    # Create mask network PPO agent
    mask_ppo = create_mask_network(
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,  # Should be Discrete(2)
        hidden_sizes=hidden_sizes,
        features_dim=hidden_sizes[-1] if hidden_sizes else 64,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        ent_coef=ent_coef,
        device=device,
        verbose=verbose,
    )
    
    # Set the mask network in the environments
    train_env.set_mask_network(mask_ppo.policy)
    eval_env.set_mask_network(mask_ppo.policy)
    
    # Set the environment for PPO
    mask_ppo.set_env(train_env)
    
    # Resume from checkpoint if specified
    if resume_from is not None and os.path.exists(resume_from):
        if verbose > 0:
            print(f"Resuming mask training from {resume_from}")
        mask_ppo = PPO.load(resume_from, env=train_env, device=device_obj)
        train_env.set_mask_network(mask_ppo.policy)
        eval_env.set_mask_network(mask_ppo.policy)
    
    # Create callback
    callback = MaskTrainingCallback(
        logger=logger,
        eval_env=eval_env,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        agent_policy=agent_policy,
        verbose=verbose,
    )
    
    # Train the mask network
    if verbose > 0:
        print(f"\n{'='*60}")
        print(f"Training Mask Network")
        print(f"{'='*60}")
        print(f"Environment: {env_id}")
        print(f"Total timesteps: {total_timesteps}")
        print(f"Alpha (blinding bonus): {alpha}")
        print(f"Hidden sizes: {hidden_sizes}")
        print(f"Learning rate: {learning_rate}")
        print(f"Device: {device_obj}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}\n")
    
    start_time = time.time()
    
    try:
        mask_ppo.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=1,
            progress_bar=(verbose > 0),
        )
    except KeyboardInterrupt:
        if verbose > 0:
            print("\nTraining interrupted. Saving current model...")
    
    elapsed = time.time() - start_time
    
    # Save the trained mask network
    model_path = os.path.join(output_dir, "mask_network.zip")
    mask_ppo.save(model_path)
    
    # Save training metadata
    metadata = {
        'env_id': env_id,
        'seed': seed,
        'total_timesteps': total_timesteps,
        'alpha': alpha,
        'hidden_sizes': list(hidden_sizes),
        'learning_rate': learning_rate,
        'training_time_seconds': elapsed,
        'final_eval_reward': logger.get_stats('mask/eval_reward') if logger else None,
    }
    
    import json
    metadata_path = os.path.join(output_dir, "mask_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    # Save logger data
    logger.save(os.path.join(output_dir, "mask_logger.pkl"))
    
    if verbose > 0:
        print(f"\n{'='*60}")
        print(f"Mask Network Training Complete")
        print(f"{'='*60}")
        print(f"Training time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"Model saved to: {model_path}")
        print(f"Metadata saved to: {metadata_path}")
        
        # Print final evaluation stats
        eval_stats = logger.get_stats('mask/eval_reward')
        if eval_stats:
            print(f"Final eval reward: {eval_stats['mean']:.2f} ± {eval_stats['std']:.2f}")
        print(f"{'='*60}\n")
    
    return mask_ppo, logger, model_path


def load_mask_network(
    path: str,
    env: Optional[gym.Env] = None,
    device: str = "auto",
    **kwargs,
) -> PPO:
    """
    Load a trained mask network from disk.
    
    Args:
        path: Path to the saved mask network (.zip file)
        env: Optional environment to attach
        device: Device string
        **kwargs: Additional arguments for PPO.load
    
    Returns:
        Loaded PPO mask network
    """
    device_obj = get_device(device)
    
    if env is not None:
        mask_ppo = PPO.load(path, env=env, device=device_obj, **kwargs)
    else:
        mask_ppo = PPO.load(path, device=device_obj, **kwargs)
    
    return mask_ppo


def get_mask_probability(
    mask_network: PPO,
    observations: np.ndarray,
    deterministic: bool = False,
) -> np.ndarray:
    """
    Get the mask network's probability of NOT randomizing (aᵉ=0) for given states.
    
    This is used for computing importance scores: I(s) = 1 - ξ(aᵉ=0|s).
    
    Args:
        mask_network: Trained mask PPO model
        observations: Batch of observations (numpy array)
        deterministic: If True, return deterministic action probabilities
    
    Returns:
        Array of probabilities for action 0 (trust agent) for each observation
    """
    if isinstance(observations, np.ndarray) and observations.ndim == 1:
        observations = observations[np.newaxis, :]
    
    # Get action probabilities from the policy
    obs_tensor = torch.as_tensor(observations, dtype=torch.float32, 
                                  device=mask_network.device)
    
    with torch.no_grad():
        # Get distribution over actions
        dist = mask_network.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs  # Shape: (batch, 2)
        
        if deterministic:
            # Return 1.0 for the most likely action
            actions = probs.argmax(dim=-1)
            result = (actions == 0).float().cpu().numpy()
        else:
            # Return probability of action 0
            result = probs[:, 0].cpu().numpy()
    
    if result.ndim == 0:
        result = np.array([result])
    
    return result


def compute_importance(
    mask_network: PPO,
    observations: np.ndarray,
) -> np.ndarray:
    """
    Compute importance scores for states.
    
    I(s) = 1 - ξ(aᵉ=0 | s) = probability of randomizing.
    Higher I(s) means the state is more critical (mask wants to randomize more).
    
    Args:
        mask_network: Trained mask PPO model
        observations: Batch of observations
    
    Returns:
        Importance scores for each observation
    """
    prob_trust = get_mask_probability(mask_network, observations, deterministic=False)
    importance = 1.0 - prob_trust
    return importance


# ==============================================================================
# Command-line Interface
# ==============================================================================

def main():
    """CLI entry point for mask network training."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train RICE Mask Network",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train mask on Hopper with default config
  python -m rice.mask_network --env Hopper-v3 --agent-path agent.zip
  
  # Train mask with custom parameters
  python -m rice.mask_network --env Walker2d-v3 --agent-path agent.zip \\
      --total-timesteps 300000 --alpha 0.0001 --hidden-sizes 64 64
        """,
    )
    
    parser.add_argument('--env', type=str, required=True,
                        help='Gymnasium environment ID')
    parser.add_argument('--agent-path', type=str, required=True,
                        help='Path to pre-trained agent model (.zip)')
    parser.add_argument('--output-dir', type=str, default='./outputs/mask',
                        help='Output directory for mask model and logs')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to base config YAML')
    parser.add_argument('--env-config', type=str, default=None,
                        help='Path to environment-specific config YAML')
    parser.add_argument('--total-timesteps', type=int, default=None,
                        help='Total training timesteps')
    parser.add_argument('--alpha', type=float, default=None,
                        help='Blinding bonus coefficient')
    parser.add_argument('--hidden-sizes', type=int, nargs='+', default=None,
                        help='Hidden layer sizes for mask MLP')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='PPO learning rate')
    parser.add_argument('--n-steps', type=int, default=None,
                        help='Rollout steps per update')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Minibatch size')
    parser.add_argument('--n-epochs', type=int, default=None,
                        help='Number of epochs per update')
    parser.add_argument('--gamma', type=float, default=None,
                        help='Discount factor')
    parser.add_argument('--ent-coef', type=float, default=None,
                        help='Entropy coefficient')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cpu, cuda, auto)')
    parser.add_argument('--eval-freq', type=int, default=10000,
                        help='Evaluation frequency')
    parser.add_argument('--n-eval-episodes', type=int, default=10,
                        help='Number of evaluation episodes')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbosity level')
    parser.add_argument('--resume-from', type=str, default=None,
                        help='Resume training from checkpoint')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config is not None:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = load_config(env_name=None)
    
    # Merge environment-specific config if provided
    if args.env_config is not None:
        import yaml
        with open(args.env_config, 'r') as f:
            env_config = yaml.safe_load(f)
        from rice.utils import deep_merge
        config = deep_merge(config, env_config)
    
    # Load agent policy
    print(f"Loading agent policy from {args.agent_path}")
    agent_policy = PPO.load(args.agent_path)
    
    # Train mask network
    mask_ppo, logger, model_path = train_mask_network(
        env_id=args.env,
        agent_policy=agent_policy,
        config=config,
        output_dir=args.output_dir,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        alpha=args.alpha,
        hidden_sizes=tuple(args.hidden_sizes) if args.hidden_sizes else None,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        device=args.device,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        verbose=args.verbose,
        resume_from=args.resume_from,
    )
    
    return mask_ppo, logger, model_path


if __name__ == '__main__':
    main()