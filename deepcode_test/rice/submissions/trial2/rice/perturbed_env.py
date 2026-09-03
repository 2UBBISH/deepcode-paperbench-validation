"""
Perturbed Environment Wrapper for RICE Mask Network Training.

This module implements a Gymnasium wrapper that interposes a mask network
between a frozen agent policy and the environment. The mask network decides
at each step whether to use the agent's action or a random action.

The perturbed policy is:
    π̄(a|s) = ξ(aᵉ=0|s) · π(a|s) + (1 - ξ(aᵉ=0|s)) · πʳ(a|s)

where:
    - ξ is the mask network (binary policy)
    - π is the frozen agent policy
    - πʳ is the random (uniform) policy

The mask network receives an augmented reward:
    r_mask = r_env + α · I(aᵉ = 1)

where α encourages the mask to "blind" (randomize) actions, and the mask
must learn to balance between following the agent (for performance) and
randomizing (for exploration/explanation).

Reference: RICE paper (Section 3.2, Mask Network Training)
"""

from typing import Any, Callable, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from gymnasium import spaces


class PerturbedEnv(gym.Wrapper):
    """
    A Gymnasium wrapper that creates a perturbed environment for mask network training.

    The wrapper:
    1. Takes a frozen agent policy π and a mask network ξ.
    2. On each step, samples mask action aᵉ ~ ξ(·|s):
       - If aᵉ = 0: use agent action a ~ π(·|s)
       - If aᵉ = 1: use random action a ~ Uniform(A)
    3. Returns augmented reward: r_mask = r_env + α · I(aᵉ = 1)
    4. Exposes a Discrete(2) action space for the mask network.

    This wrapper is designed to be used with Stable-Baselines3 PPO for training
    the mask network as a standard RL agent.
    """

    def __init__(
        self,
        env: gym.Env,
        agent_policy: Any,
        alpha: float = 0.0001,
        deterministic_agent: bool = False,
        mask_network: Optional[nn.Module] = None,
        device: str = "auto",
    ):
        """
        Initialize the perturbed environment wrapper.

        Args:
            env: The base Gymnasium environment.
            agent_policy: The frozen agent policy π. Can be:
                - A Stable-Baselines3 model (with .predict() method)
                - A callable that takes (obs) and returns (action, ...)
                - A PyTorch module that takes (obs_tensor) and returns action
            alpha: Coefficient for the intrinsic reward for randomizing (default: 0.0001).
            deterministic_agent: Whether to use deterministic actions from the agent policy.
            mask_network: Optional pre-initialized mask network (for evaluation mode).
            device: Device string for tensor operations ("auto", "cpu", "cuda").
        """
        super().__init__(env)

        self.agent_policy = agent_policy
        self.alpha = alpha
        self.deterministic_agent = deterministic_agent
        self.mask_network = mask_network

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Store original action space for random action sampling
        self._original_action_space = env.action_space

        # The mask network action space is Discrete(2):
        #   0 = follow agent (aᵉ = 0)
        #   1 = randomize (aᵉ = 1)
        self.action_space = spaces.Discrete(2)

        # Observation space remains the same as the base environment
        # (the mask network sees the same state as the agent)

        # Track last mask action for reward computation
        self._last_mask_action: Optional[int] = None
        self._last_env_reward: float = 0.0

        # Cache for agent action computation
        self._last_obs: Optional[np.ndarray] = None

    def step(
        self, action: Union[int, np.ndarray]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the perturbed environment.

        Args:
            action: The mask network action aᵉ ∈ {0, 1}.
                - 0: Use agent policy action
                - 1: Use random action

        Returns:
            observation: Next state observation.
            reward: Augmented reward r_mask = r_env + α · I(aᵉ = 1).
            terminated: Whether the episode terminated.
            truncated: Whether the episode was truncated.
            info: Additional information dictionary.
        """
        # Extract mask action (handle both scalar and array inputs)
        if isinstance(action, (np.ndarray, list)):
            mask_action = int(action[0]) if len(action) > 0 else int(action)
        else:
            mask_action = int(action)

        mask_action = np.clip(mask_action, 0, 1)
        self._last_mask_action = mask_action

        # Get the current observation
        obs = self._last_obs if self._last_obs is not None else self.unwrapped._get_obs()

        # Determine the actual environment action
        if mask_action == 0:
            # Follow agent policy
            env_action = self._get_agent_action(obs)
        else:
            # Use random action
            env_action = self._get_random_action()

        # Step the base environment with the selected action
        next_obs, env_reward, terminated, truncated, info = self.env.step(env_action)

        self._last_env_reward = env_reward
        self._last_obs = next_obs

        # Compute augmented reward for the mask network
        # r_mask = r_env + α · I(aᵉ = 1)
        intrinsic_reward = self.alpha * float(mask_action == 1)
        mask_reward = env_reward + intrinsic_reward

        # Add mask-specific info
        info["mask_action"] = mask_action
        info["env_action"] = env_action
        info["env_reward"] = env_reward
        info["intrinsic_reward"] = intrinsic_reward
        info["mask_reward"] = mask_reward

        return next_obs, mask_reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment and return the initial observation.

        Args:
            seed: Random seed for reproducibility.
            options: Additional reset options.

        Returns:
            observation: Initial state observation.
            info: Additional information dictionary.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_obs = obs
        self._last_mask_action = None
        self._last_env_reward = 0.0
        return obs, info

    def _get_agent_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Get the action from the frozen agent policy.

        Args:
            obs: Current observation.

        Returns:
            Action sampled from the agent policy.
        """
        # Handle different agent policy types
        if hasattr(self.agent_policy, "predict"):
            # Stable-Baselines3 model
            action, _states = self.agent_policy.predict(
                obs, deterministic=self.deterministic_agent
            )
            return action
        elif callable(self.agent_policy):
            # Callable function
            result = self.agent_policy(obs)
            if isinstance(result, tuple):
                return result[0]
            return result
        elif isinstance(self.agent_policy, nn.Module):
            # PyTorch module
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                if obs_tensor.ndim == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)
                action_tensor = self.agent_policy(obs_tensor)
                if self.deterministic_agent:
                    action_tensor = action_tensor.argmax(dim=-1)
                action = action_tensor.squeeze(0).cpu().numpy()
            return action
        else:
            raise TypeError(
                f"Unsupported agent policy type: {type(self.agent_policy)}. "
                "Expected SB3 model, callable, or nn.Module."
            )

    def _get_random_action(self) -> np.ndarray:
        """
        Sample a random action from the original action space.

        Returns:
            Random action sampled uniformly from the action space.
        """
        return self._original_action_space.sample()

    def get_mask_probability(self, obs: np.ndarray) -> float:
        """
        Compute the probability that the mask network would randomize for a given state.

        This is used for explanation extraction: I(s) = 1 - ξ(aᵉ=0 | s).

        Args:
            obs: State observation.

        Returns:
            Probability of randomizing (aᵉ = 1).
        """
        if self.mask_network is None:
            return 0.0

        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            if obs_tensor.ndim == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            logits = self.mask_network(obs_tensor)
            probs = torch.softmax(logits, dim=-1)
            # Probability of action 1 (randomize)
            randomize_prob = probs[0, 1].item()
        return randomize_prob

    def get_importance(self, obs: np.ndarray) -> float:
        """
        Compute the importance score for a state.

        I(s) = 1 - ξ(aᵉ=0 | s) = probability of NOT following the agent.
        Higher I(s) means the state is more critical (the mask thinks it should randomize).

        Args:
            obs: State observation.

        Returns:
            Importance score in [0, 1].
        """
        return self.get_mask_probability(obs)


class PerturbedEnvWrapper(gym.Wrapper):
    """
    Alternative wrapper that uses a pre-trained mask network for evaluation.

    This wrapper is used during the explanation extraction phase, where we
    have a trained mask network and want to evaluate its decisions on trajectories.

    Unlike PerturbedEnv (which is for training the mask), this wrapper:
    - Uses the mask network to decide actions (no random exploration needed)
    - Records mask decisions for later analysis
    - Does not modify rewards
    """

    def __init__(
        self,
        env: gym.Env,
        agent_policy: Any,
        mask_network: nn.Module,
        deterministic_agent: bool = True,
        device: str = "auto",
    ):
        """
        Initialize the perturbed environment wrapper for evaluation.

        Args:
            env: The base Gymnasium environment.
            agent_policy: The frozen agent policy π.
            mask_network: The trained mask network ξ.
            deterministic_agent: Whether to use deterministic agent actions.
            device: Device string for tensor operations.
        """
        super().__init__(env)

        self.agent_policy = agent_policy
        self.mask_network = mask_network
        self.deterministic_agent = deterministic_agent

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._original_action_space = env.action_space

        # Track mask decisions
        self.mask_history: list = []
        self._last_obs: Optional[np.ndarray] = None

    def step(self, action: Any = None) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step using the mask network to decide the action.

        Note: The action parameter is ignored; the mask network decides.

        Args:
            action: Ignored (mask network decides).

        Returns:
            observation, reward, terminated, truncated, info
        """
        obs = self._last_obs

        # Get mask decision
        mask_action, mask_prob = self._get_mask_decision(obs)

        # Determine actual action
        if mask_action == 0:
            env_action = self._get_agent_action(obs)
        else:
            env_action = self._get_random_action()

        # Step environment
        next_obs, reward, terminated, truncated, info = self.env.step(env_action)

        self._last_obs = next_obs

        # Record mask decision
        self.mask_history.append({
            "observation": obs,
            "mask_action": mask_action,
            "mask_prob_randomize": mask_prob,
            "env_action": env_action,
            "reward": reward,
        })

        info["mask_action"] = mask_action
        info["mask_prob_randomize"] = mask_prob

        return next_obs, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_obs = obs
        self.mask_history = []
        return obs, info

    def _get_mask_decision(self, obs: np.ndarray) -> Tuple[int, float]:
        """
        Get the mask network's decision.

        Args:
            obs: Current observation.

        Returns:
            Tuple of (mask_action, probability_of_randomizing).
        """
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            if obs_tensor.ndim == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            logits = self.mask_network(obs_tensor)
            probs = torch.softmax(logits, dim=-1)
            # Sample action
            action_dist = torch.distributions.Categorical(probs)
            mask_action = action_dist.sample().item()
            randomize_prob = probs[0, 1].item()
        return mask_action, randomize_prob

    def _get_agent_action(self, obs: np.ndarray) -> np.ndarray:
        """Get action from the frozen agent policy."""
        if hasattr(self.agent_policy, "predict"):
            action, _states = self.agent_policy.predict(
                obs, deterministic=self.deterministic_agent
            )
            return action
        elif callable(self.agent_policy):
            result = self.agent_policy(obs)
            if isinstance(result, tuple):
                return result[0]
            return result
        elif isinstance(self.agent_policy, nn.Module):
            with torch.no_grad():
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                if obs_tensor.ndim == 1:
                    obs_tensor = obs_tensor.unsqueeze(0)
                action_tensor = self.agent_policy(obs_tensor)
                if self.deterministic_agent:
                    action_tensor = action_tensor.argmax(dim=-1)
                action = action_tensor.squeeze(0).cpu().numpy()
            return action
        else:
            raise TypeError(f"Unsupported agent policy type: {type(self.agent_policy)}")

    def _get_random_action(self) -> np.ndarray:
        """Sample a random action."""
        return self._original_action_space.sample()

    def get_importance_scores(self) -> list:
        """
        Get importance scores for all states in the trajectory.

        Returns:
            List of dicts with 'observation', 'importance' (prob of randomizing).
        """
        scores = []
        for entry in self.mask_history:
            scores.append({
                "observation": entry["observation"],
                "importance": entry["mask_prob_randomize"],
                "mask_action": entry["mask_action"],
            })
        return scores


def make_perturbed_env(
    env_id: str,
    agent_policy: Any,
    alpha: float = 0.0001,
    deterministic_agent: bool = False,
    mask_network: Optional[nn.Module] = None,
    device: str = "auto",
    seed: int = 0,
    **env_kwargs,
) -> PerturbedEnv:
    """
    Factory function to create a perturbed environment.

    Args:
        env_id: Gymnasium environment ID (e.g., "Hopper-v3").
        agent_policy: Frozen agent policy.
        alpha: Intrinsic reward coefficient.
        deterministic_agent: Whether agent uses deterministic actions.
        mask_network: Optional pre-initialized mask network.
        device: Device for tensor operations.
        seed: Random seed.
        **env_kwargs: Additional arguments for environment creation.

    Returns:
        PerturbedEnv instance.
    """
    import gymnasium as gym

    env = gym.make(env_id, **env_kwargs)
    env.reset(seed=seed)

    perturbed_env = PerturbedEnv(
        env=env,
        agent_policy=agent_policy,
        alpha=alpha,
        deterministic_agent=deterministic_agent,
        mask_network=mask_network,
        device=device,
    )

    return perturbed_env


def make_perturbed_vec_env(
    env_id: str,
    agent_policy: Any,
    n_envs: int = 1,
    alpha: float = 0.0001,
    deterministic_agent: bool = False,
    device: str = "auto",
    seed: int = 0,
    **env_kwargs,
) -> Any:
    """
    Factory function to create a vectorized perturbed environment for SB3 training.

    Args:
        env_id: Gymnasium environment ID.
        agent_policy: Frozen agent policy.
        n_envs: Number of parallel environments.
        alpha: Intrinsic reward coefficient.
        deterministic_agent: Whether agent uses deterministic actions.
        device: Device for tensor operations.
        seed: Random seed.
        **env_kwargs: Additional arguments for environment creation.

    Returns:
        Vectorized environment compatible with SB3.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    def _make_env(rank: int):
        def _init():
            import gymnasium as gym

            env = gym.make(env_id, **env_kwargs)
            env.reset(seed=seed + rank)

            perturbed_env = PerturbedEnv(
                env=env,
                agent_policy=agent_policy,
                alpha=alpha,
                deterministic_agent=deterministic_agent,
                device=device,
            )
            return perturbed_env

        return _init

    vec_env = DummyVecEnv([_make_env(i) for i in range(n_envs)])
    return vec_env