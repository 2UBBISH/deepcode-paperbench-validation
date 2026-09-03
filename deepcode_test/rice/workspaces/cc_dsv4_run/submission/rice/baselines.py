"""
Baseline refinement methods compared against RICE in the paper.

Baselines:
1. PPO Fine-tuning: Lower learning rate, continue training with PPO.
2. StateMask-R: Reset to critical state identified by explanation,
   then fine-tune from that state (Cheng et al., 2023).
3. Jump-Start Reinforcement Learning (JSRL): Curriculum-based
   refinement using a guide policy and exploration policy
   (Uchendu et al., 2023).
4. SAC Fine-tuning: Continue training with SAC (for SAC agents).
5. Self-Imitation Learning (SIL): Prioritize good past experiences
   (Oh et al., 2018).

Also includes the "Random" explanation baseline: selects a random
visited state as the "critical" state.
"""

from typing import Callable, Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Baseline Explanation Methods
# ---------------------------------------------------------------------------

def random_explanation_importance(
    states: np.ndarray,
) -> np.ndarray:
    """
    Random explanation baseline: assigns random importance scores to states.
    This is equivalent to randomly selecting a visited state as critical.

    Args:
        states: Array of shape (T, state_dim).

    Returns:
        importances: Array of shape (T,) with random scores in [0, 1].
    """
    return np.random.random(len(states)).astype(np.float32)


def random_critical_state(
    states: np.ndarray,
) -> Tuple[int, np.ndarray]:
    """
    Random explanation: pick a random state from the trajectory as critical.

    Args:
        states: Array of shape (T, state_dim).

    Returns:
        idx: Random index.
        state: The state at that index.
    """
    idx = np.random.randint(0, len(states))
    return idx, states[idx]


# ---------------------------------------------------------------------------
# PPO Fine-tuning Baseline
# ---------------------------------------------------------------------------

def ppo_finetune(
    policy_net: nn.Module,
    env_reset_fn: Callable[[], np.ndarray],
    env_step_fn: Callable[[np.ndarray], Tuple],
    n_iterations: int,
    rollout_length: int = 2048,
    lr: float = 1e-4,  # Lower learning rate for fine-tuning
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    update_epochs: int = 10,
    batch_size: int = 64,
    normalize_advantages: bool = True,
    verbose: bool = True,
    evaluate_fn: Optional[Callable[[nn.Module], float]] = None,
) -> List[Dict[str, float]]:
    """
    PPO Fine-tuning baseline.
    Lower the learning rate and continue training with standard PPO.
    No explanation, no mixed initial distribution, no exploration bonus.

    Args:
        policy_net: Policy network to fine-tune (must support
                    get_action_mean, get_action_std, get_value).
        env_reset_fn: Environment reset function.
        env_step_fn: Environment step function.
        n_iterations: Number of PPO iterations.
        rollout_length: Steps per rollout.
        lr: Learning rate (lower than training from scratch).
        gamma: Discount factor.
        gae_lambda: GAE lambda.
        clip_epsilon: PPO clip epsilon.
        value_coef: Value loss coefficient.
        entropy_coef: Entropy bonus coefficient.
        max_grad_norm: Max gradient norm.
        update_epochs: PPO epochs per iteration.
        batch_size: Mini-batch size.
        normalize_advantages: Whether to normalize advantages.
        verbose: Print progress.
        evaluate_fn: Optional evaluation function.

    Returns:
        Training history.
    """
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    history = []

    for iteration in range(n_iterations):
        # Collect rollout from default initial distribution
        states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
        state = env_reset_fn()

        for _ in range(rollout_length):
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_mean = policy_net.get_action_mean(state_t)
                action_std = policy_net.get_action_std(state_t)
                dist = torch.distributions.Normal(action_mean, action_std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
                entropy = dist.entropy().sum(dim=-1, keepdim=True)
                value = policy_net.get_value(state_t)

            action_np = action.squeeze(0).numpy()
            next_state, reward, done, info = env_step_fn(action_np)

            states.append(state)
            actions.append(action_np)
            rewards.append(reward)
            values.append(float(value.item()))
            log_probs.append(float(log_prob.item()))
            dones.append(float(done))

            state = next_state
            if done:
                state = env_reset_fn()

        # Compute GAE
        advantages, returns = _compute_gae(
            rewards, values, dones, gamma, gae_lambda
        )

        # PPO update
        stats = _ppo_update(
            policy_net, optimizer,
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(log_probs, dtype=np.float32),
            advantages, returns,
            clip_epsilon, value_coef, entropy_coef,
            max_grad_norm, update_epochs, batch_size,
            normalize_advantages,
        )

        stats["iteration"] = iteration
        stats["mean_reward"] = float(np.mean(rewards))
        if evaluate_fn is not None:
            stats["eval_reward"] = evaluate_fn(policy_net)
        history.append(stats)

        if verbose and iteration % 10 == 0:
            eval_str = ""
            if "eval_reward" in stats:
                eval_str = f"eval_reward={stats['eval_reward']:.2f} "
            print(
                f"[PPO-FT] iter={iteration} {eval_str}"
                f"policy_loss={stats['policy_loss']:.4f} "
                f"value_loss={stats['value_loss']:.4f}"
            )

    return history


# ---------------------------------------------------------------------------
# StateMask-R Baseline
# ---------------------------------------------------------------------------

def statemask_r_refine(
    policy_net: nn.Module,
    mask_trainer,
    env_reset_fn: Callable[[], np.ndarray],
    env_step_fn: Callable[[np.ndarray], Tuple],
    n_iterations: int,
    rollout_length: int = 2048,
    target_policy_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    lr: float = 1e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    update_epochs: int = 10,
    batch_size: int = 64,
    normalize_advantages: bool = True,
    verbose: bool = True,
    evaluate_fn: Optional[Callable[[nn.Module], float]] = None,
) -> List[Dict[str, float]]:
    """
    StateMask-R baseline: Refine by resetting to critical states and
    continuing training from those states.

    This is the refinement method from Cheng et al. (2023).
    Unlike RICE, it:
    - Initializes ONLY from critical states (no mixture with ρ)
    - Does NOT use an exploration bonus

    Args:
        policy_net: Policy network to refine.
        mask_trainer: Trained MaskNetworkTrainer for finding critical states.
        env_reset_fn: Environment reset.
        env_step_fn: Environment step.
        n_iterations: Number of refinement iterations.
        rollout_length: Steps per rollout.
        target_policy_fn: Pre-trained policy (for collecting trajectory).
        lr: Learning rate.
        gamma, gae_lambda, clip_epsilon, value_coef, entropy_coef,
        max_grad_norm, update_epochs, batch_size: PPO hyper-parameters.
        normalize_advantages: Whether to normalize advantages.
        verbose: Print progress.
        evaluate_fn: Evaluation function.

    Returns:
        Training history.
    """
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    history = []

    for iteration in range(n_iterations):
        states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []

        # Always start from a critical state (no mixing in StateMask-R)
        state = _get_critical_start_state(
            mask_trainer=mask_trainer,
            target_policy_fn=target_policy_fn,
            env_reset_fn=env_reset_fn,
            env_step_fn=env_step_fn,
        )

        for _ in range(rollout_length):
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_mean = policy_net.get_action_mean(state_t)
                action_std = policy_net.get_action_std(state_t)
                dist = torch.distributions.Normal(action_mean, action_std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
                entropy = dist.entropy().sum(dim=-1, keepdim=True)
                value = policy_net.get_value(state_t)

            action_np = action.squeeze(0).numpy()
            next_state, reward, done, info = env_step_fn(action_np)

            states.append(state)
            actions.append(action_np)
            rewards.append(reward)
            values.append(float(value.item()))
            log_probs.append(float(log_prob.item()))
            dones.append(float(done))

            state = next_state
            if done:
                # Start new episode from critical state
                state = _get_critical_start_state(
                    mask_trainer=mask_trainer,
                    target_policy_fn=target_policy_fn,
                    env_reset_fn=env_reset_fn,
                    env_step_fn=env_step_fn,
                )

        # Compute GAE
        advantages, returns = _compute_gae(
            rewards, values, dones, gamma, gae_lambda
        )

        # PPO update
        stats = _ppo_update(
            policy_net, optimizer,
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(log_probs, dtype=np.float32),
            advantages, returns,
            clip_epsilon, value_coef, entropy_coef,
            max_grad_norm, update_epochs, batch_size,
            normalize_advantages,
        )

        stats["iteration"] = iteration
        stats["mean_reward"] = float(np.mean(rewards))
        if evaluate_fn is not None:
            stats["eval_reward"] = evaluate_fn(policy_net)
        history.append(stats)

        if verbose and iteration % 10 == 0:
            eval_str = ""
            if "eval_reward" in stats:
                eval_str = f"eval_reward={stats['eval_reward']:.2f} "
            print(
                f"[StateMask-R] iter={iteration} {eval_str}"
                f"policy_loss={stats['policy_loss']:.4f}"
            )

    return history


# ---------------------------------------------------------------------------
# Jump-Start Reinforcement Learning (JSRL) Baseline
# ---------------------------------------------------------------------------

class JSRLRefiner:
    """
    Jump-Start Reinforcement Learning (Uchendu et al., 2023).

    JSRL uses a guide policy π_g to design a curriculum for training
    an exploration policy π_e. The curriculum gradually shifts the
    exploration frontier from near the end of episodes toward the beginning.

    In the refinement setting:
    - π_g = π (the pre-trained policy)
    - π_e is initialized to π_g
    - The curriculum randomly selects a step U in [0, H] where H is
      the episode horizon. π_g runs for U steps, then π_e takes over.
    - Over time, the curriculum reduces U, shifting more responsibility
      to π_e.

    Key difference from RICE: JSRL's exploration frontiers are random
    (not identified by explanation), which cannot guarantee positive returns.
    """

    def __init__(
        self,
        guide_policy_net: nn.Module,
        exploration_policy_net: nn.Module,
        env_reset_fn: Callable[[], np.ndarray],
        env_step_fn: Callable[[np.ndarray], Tuple],
        max_episode_steps: int = 1000,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        batch_size: int = 64,
        normalize_advantages: bool = True,
    ):
        """
        Args:
            guide_policy_net: Pre-trained policy network (π_g = π).
            exploration_policy_net: Policy network to train (π_e),
                                    initialized to π_g weights.
            env_reset_fn: Environment reset.
            env_step_fn: Environment step.
            max_episode_steps: Maximum steps per episode (H).
            lr, gamma, gae_lambda, clip_epsilon, value_coef, entropy_coef,
            max_grad_norm, update_epochs, batch_size: PPO hyper-parameters.
            normalize_advantages: Whether to normalize advantages.
        """
        self.guide_policy = guide_policy_net
        self.explore_policy = exploration_policy_net
        self.env_reset_fn = env_reset_fn
        self.env_step_fn = env_step_fn
        self.max_episode_steps = max_episode_steps

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.normalize_advantages = normalize_advantages

        self.optimizer = torch.optim.Adam(
            exploration_policy_net.parameters(), lr=lr
        )

    def _get_rollout_step(
        self,
        step_in_episode: int,
        state: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
        """
        Sample rollout step using JSRL's curriculum approach.

        The guide policy runs for U steps (randomly chosen), then the
        exploration policy takes over.
        """
        # Curriculum: random U in [0, H]
        U = np.random.randint(0, self.max_episode_steps)

        if step_in_episode < U:
            # Use guide policy
            policy = self.guide_policy
        else:
            # Use exploration policy
            policy = self.explore_policy

        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_mean = policy.get_action_mean(state_t)
            action_std = policy.get_action_std(state_t)
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().sum(dim=-1, keepdim=True)
            value = self.explore_policy.get_value(state_t)

        return (
            action.squeeze(0).numpy(),
            log_prob,
            entropy,
            value,
        )

    def refine(
        self,
        n_iterations: int,
        rollout_length: int = 2048,
        verbose: bool = True,
        evaluate_fn: Optional[Callable[[nn.Module], float]] = None,
    ) -> List[Dict[str, float]]:
        """Run JSRL refinement."""
        history = []

        for iteration in range(n_iterations):
            states, actions, rewards, values, log_probs, dones = (
                [], [], [], [], [], []
            )
            state = self.env_reset_fn()
            step_in_episode = 0

            for _ in range(rollout_length):
                action_np, lp, ent, val = self._get_rollout_step(
                    step_in_episode, state
                )
                next_state, reward, done, info = self.env_step_fn(action_np)

                states.append(state)
                actions.append(action_np)
                rewards.append(reward)
                values.append(float(val.item()))
                log_probs.append(float(lp.item()))
                dones.append(float(done))

                state = next_state
                step_in_episode += 1
                if done:
                    state = self.env_reset_fn()
                    step_in_episode = 0

            advantages, returns = _compute_gae(
                rewards, values, dones, self.gamma, self.gae_lambda
            )

            stats = _ppo_update(
                self.explore_policy, self.optimizer,
                np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.float32),
                np.array(log_probs, dtype=np.float32),
                advantages, returns,
                self.clip_epsilon, self.value_coef, self.entropy_coef,
                self.max_grad_norm, self.update_epochs, self.batch_size,
                self.normalize_advantages,
            )

            stats["iteration"] = iteration
            stats["mean_reward"] = float(np.mean(rewards))
            if evaluate_fn is not None:
                stats["eval_reward"] = evaluate_fn(self.explore_policy)
            history.append(stats)

            if verbose and iteration % 10 == 0:
                eval_str = ""
                if "eval_reward" in stats:
                    eval_str = f"eval_reward={stats['eval_reward']:.2f} "
                print(
                    f"[JSRL] iter={iteration} {eval_str}"
                    f"policy_loss={stats['policy_loss']:.4f}"
                )

        return history


# ---------------------------------------------------------------------------
# Self-Imitation Learning (SIL) Baseline
# ---------------------------------------------------------------------------

class SILBuffer:
    """
    Self-Imitation Learning replay buffer.
    Stores (state, action, cumulative_return) tuples, prioritizing
    experiences with higher returns.
    """

    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.states = []
        self.actions = []
        self.returns = []

    def add(self, state, action, ret):
        self.states.append(state)
        self.actions.append(action)
        self.returns.append(ret)
        if len(self.states) > self.capacity:
            # Remove lowest return
            idx = np.argmin(self.returns)
            del self.states[idx]
            del self.actions[idx]
            del self.returns[idx]

    def sample(self, batch_size: int):
        if len(self.states) < batch_size:
            return None
        # Sample proportionally to return (prioritize good experiences)
        probs = np.array(self.returns)
        probs = np.maximum(probs, 0)  # Only positive returns
        if probs.sum() == 0:
            probs = np.ones(len(self.returns)) / len(self.returns)
        else:
            probs = probs / probs.sum()
        indices = np.random.choice(len(self.states), batch_size, p=probs)
        return (
            np.array([self.states[i] for i in indices], dtype=np.float32),
            np.array([self.actions[i] for i in indices], dtype=np.float32),
            np.array([self.returns[i] for i in indices], dtype=np.float32),
        )

    def __len__(self):
        return len(self.states)


def sil_refine(
    policy_net: nn.Module,
    env_reset_fn: Callable[[], np.ndarray],
    env_step_fn: Callable[[np.ndarray], Tuple],
    n_iterations: int,
    rollout_length: int = 2048,
    sil_batch_size: int = 64,
    sil_coef: float = 0.01,
    max_episode_steps: int = 1000,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
    update_epochs: int = 10,
    batch_size: int = 64,
    normalize_advantages: bool = True,
    verbose: bool = True,
    evaluate_fn: Optional[Callable[[nn.Module], float]] = None,
) -> List[Dict[str, float]]:
    """
    Self-Imitation Learning baseline (Oh et al., 2018).

    Adds a SIL loss to standard PPO that encourages the agent to imitate
    its own past successful experiences. Experiences with higher cumulative
    returns are prioritized.

    Args:
        policy_net: Policy network.
        env_reset_fn, env_step_fn: Environment functions.
        n_iterations: Number of iterations.
        rollout_length: Steps per rollout.
        sil_batch_size: Batch size for SIL updates.
        sil_coef: Weight of SIL loss.
        max_episode_steps: Max episode length.
        lr, gamma, gae_lambda, clip_epsilon, value_coef, entropy_coef,
        max_grad_norm, update_epochs, batch_size: PPO hyper-parameters.
        normalize_advantages: Whether to normalize advantages.
        verbose: Print progress.
        evaluate_fn: Evaluation function.

    Returns:
        Training history.
    """
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    sil_buffer = SILBuffer(capacity=100000)
    history = []

    for iteration in range(n_iterations):
        states, actions, rewards, values, log_probs, dones = (
            [], [], [], [], [], []
        )
        episode_states, episode_actions, episode_rewards = [], [], []

        state = env_reset_fn()

        for step in range(rollout_length):
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_mean = policy_net.get_action_mean(state_t)
                action_std = policy_net.get_action_std(state_t)
                dist = torch.distributions.Normal(action_mean, action_std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
                entropy = dist.entropy().sum(dim=-1, keepdim=True)
                value = policy_net.get_value(state_t)

            action_np = action.squeeze(0).numpy()
            next_state, reward, done, info = env_step_fn(action_np)

            states.append(state)
            actions.append(action_np)
            rewards.append(reward)
            values.append(float(value.item()))
            log_probs.append(float(log_prob.item()))
            dones.append(float(done))

            episode_states.append(state)
            episode_actions.append(action_np)
            episode_rewards.append(reward)

            state = next_state
            if done:
                # Store episode experiences with cumulative returns
                cum_returns = []
                ret = 0.0
                for r in reversed(episode_rewards):
                    ret = r + gamma * ret
                    cum_returns.append(ret)
                cum_returns.reverse()

                for s, a, r in zip(episode_states, episode_actions, cum_returns):
                    sil_buffer.add(s, a, r)

                episode_states, episode_actions, episode_rewards = [], [], []
                state = env_reset_fn()

        # Compute GAE
        advantages, returns = _compute_gae(
            rewards, values, dones, gamma, gae_lambda
        )

        # Standard PPO update
        stats = _ppo_update(
            policy_net, optimizer,
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(log_probs, dtype=np.float32),
            advantages, returns,
            clip_epsilon, value_coef, entropy_coef,
            max_grad_norm, update_epochs, batch_size,
            normalize_advantages,
        )

        # SIL update
        sil_loss_val = 0.0
        sil_sample = sil_buffer.sample(sil_batch_size)
        if sil_sample is not None:
            sil_states, sil_actions, sil_returns = sil_sample
            sil_states_t = torch.FloatTensor(sil_states)
            sil_actions_t = torch.FloatTensor(sil_actions)
            sil_returns_t = torch.FloatTensor(sil_returns)

            action_mean = policy_net.get_action_mean(sil_states_t)
            action_std = policy_net.get_action_std(sil_states_t)
            dist = torch.distributions.Normal(action_mean, action_std)
            sil_log_probs = dist.log_prob(sil_actions_t).sum(dim=-1)
            sil_values = policy_net.get_value(sil_states_t).squeeze(-1)

            # SIL advantage: max(0, R - V(s))
            sil_adv = torch.clamp(sil_returns_t - sil_values, min=0)
            sil_loss = -(sil_adv * sil_log_probs).mean()

            optimizer.zero_grad()
            (sil_coef * sil_loss).backward()
            nn.utils.clip_grad_norm_(policy_net.parameters(), max_grad_norm)
            optimizer.step()

            sil_loss_val = float(sil_loss.item())

        stats["iteration"] = iteration
        stats["mean_reward"] = float(np.mean(rewards))
        stats["sil_loss"] = sil_loss_val
        if evaluate_fn is not None:
            stats["eval_reward"] = evaluate_fn(policy_net)
        history.append(stats)

        if verbose and iteration % 10 == 0:
            eval_str = ""
            if "eval_reward" in stats:
                eval_str = f"eval_reward={stats['eval_reward']:.2f} "
            print(
                f"[SIL] iter={iteration} {eval_str}"
                f"policy_loss={stats['policy_loss']:.4f} "
                f"sil_loss={stats['sil_loss']:.4f}"
            )

    return history


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[float],
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute GAE advantages and returns."""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    returns_arr = np.zeros(T, dtype=np.float32)
    gae = 0.0
    next_value = 0.0

    for t in reversed(range(T)):
        if t == T - 1:
            delta = rewards[t] + gamma * next_value * (1.0 - dones[t]) - values[t]
        else:
            delta = rewards[t] + gamma * values[t + 1] * (1.0 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * gae
        advantages[t] = gae
        returns_arr[t] = gae + values[t]

    return advantages, returns_arr


def _ppo_update(
    policy_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    states: np.ndarray,
    actions: np.ndarray,
    old_log_probs: np.ndarray,
    advantages: np.ndarray,
    returns: np.ndarray,
    clip_epsilon: float,
    value_coef: float,
    entropy_coef: float,
    max_grad_norm: float,
    update_epochs: int,
    batch_size: int,
    normalize_advantages: bool,
) -> Dict[str, float]:
    """Standard PPO update step. Returns loss statistics."""
    states_t = torch.FloatTensor(states)
    actions_t = torch.FloatTensor(actions)
    old_log_probs_t = torch.FloatTensor(old_log_probs)
    advantages_t = torch.FloatTensor(advantages)
    returns_t = torch.FloatTensor(returns)

    if normalize_advantages:
        advantages_t = (advantages_t - advantages_t.mean()) / (
            advantages_t.std() + 1e-8
        )

    n = len(states_t)
    indices = np.arange(n)
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    n_updates = 0

    for _ in range(update_epochs):
        np.random.shuffle(indices)
        for start in range(0, n, batch_size):
            batch_idx = indices[start : start + batch_size]
            batch_states = states_t[batch_idx]
            batch_actions = actions_t[batch_idx]
            batch_old_log_probs = old_log_probs_t[batch_idx]
            batch_advantages = advantages_t[batch_idx]
            batch_returns = returns_t[batch_idx]

            action_mean = policy_net.get_action_mean(batch_states)
            action_std = policy_net.get_action_std(batch_states)
            dist = torch.distributions.Normal(action_mean, action_std)
            new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            new_values = policy_net.get_value(batch_states).squeeze(-1)

            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = (
                torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
                * batch_advantages
            )
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(new_values, batch_returns)
            entropy_loss = -entropy.mean()

            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy_net.parameters(), max_grad_norm)
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
            n_updates += 1

    return {
        "policy_loss": total_policy_loss / max(n_updates, 1),
        "value_loss": total_value_loss / max(n_updates, 1),
        "entropy": total_entropy / max(n_updates, 1),
    }


def _get_critical_start_state(
    mask_trainer,
    target_policy_fn: Callable[[np.ndarray], np.ndarray],
    env_reset_fn: Callable[[], np.ndarray],
    env_step_fn: Callable[[np.ndarray], Tuple],
    trajectory_length: int = 200,
) -> np.ndarray:
    """
    Run policy to get a trajectory, find the most critical state via
    mask network, and return it as the starting state.
    """
    states = []
    state = env_reset_fn()

    for _ in range(trajectory_length):
        action = target_policy_fn(state)
        next_state, _, done, _ = env_step_fn(action)
        states.append(state)
        if done:
            state = env_reset_fn()
        else:
            state = next_state

    if len(states) < 2:
        return env_reset_fn()

    states_arr = np.array(states, dtype=np.float32)
    idx, critical_state = mask_trainer.find_most_critical_state(states_arr)
    return critical_state