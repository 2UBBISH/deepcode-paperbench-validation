"""
Fidelity Score computation for evaluating explanation methods.

The fidelity score measures how accurately an explanation method identifies
the time steps that are truly critical to the agent's final reward.

Pipeline:
1. The explanation method generates step-level importance scores for a trajectory.
2. A sliding window of size l = L * K (where L is trajectory length, K is fraction)
   finds the segment with the highest average importance score.
3. The agent is fast-forwarded to the start of the critical segment, and random
   actions are taken for l steps.
4. After random actions, the agent's policy continues normally until episode end.
5. Fidelity Score = log(d / d_max) - log(l / L)
   where d = |R' - R| (reward change), d_max = maximum possible reward change.

A higher fidelity score indicates higher fidelity.
"""

from typing import Callable, Dict, List, Optional, Tuple
import numpy as np


def compute_fidelity_score(
    d: float,
    d_max: float,
    l: int,
    L: int,
    eps: float = 1e-8,
) -> float:
    """
    Compute the fidelity score for a single trajectory.

    Args:
        d: Absolute reward change |R' - R| after randomizing critical segment.
        d_max: Maximum possible reward change in the environment.
        l: Length of the critical segment (window size).
        L: Total trajectory length.
        eps: Small epsilon to prevent log(0).

    Returns:
        Fidelity score (higher = better explanation).
    """
    return np.log(d / (d_max + eps) + eps) - np.log(l / L + eps)


def find_critical_segment(
    importances: np.ndarray,
    L: int,
    K: float,
) -> Tuple[int, int]:
    """
    Find the segment of consecutive steps with the highest average importance.

    Uses a sliding window approach: a window of size l = L * K slides across
    the trajectory, and we select the window position with the highest mean
    importance score.

    Args:
        importances: Array of step-level importance scores, shape (L,).
        L: Total trajectory length.
        K: Fraction defining window size (e.g., 0.1, 0.2, 0.3, 0.4).

    Returns:
        (start_idx, end_idx): Start and end indices (inclusive) of the
                              most critical segment.
    """
    L_actual = len(importances)
    l = max(1, int(L_actual * K))

    if l >= L_actual:
        return 0, L_actual - 1

    best_start = 0
    best_avg = -float("inf")

    for start in range(L_actual - l + 1):
        window_avg = np.mean(importances[start : start + l])
        if window_avg > best_avg:
            best_avg = window_avg
            best_start = start

    return best_start, best_start + l - 1


def evaluate_fidelity(
    env_reset_fn: Callable[[], np.ndarray],
    env_step_fn: Callable[[np.ndarray], Tuple],
    env_set_state_fn: Optional[Callable[[np.ndarray], None]],
    policy_fn: Callable[[np.ndarray], np.ndarray],
    importance_fn: Callable[[np.ndarray], np.ndarray],
    action_space_sample_fn: Callable[[], np.ndarray],
    d_max: float,
    n_trajectories: int = 500,
    K: float = 0.1,
    max_episode_steps: int = 1000,
) -> Dict[str, float]:
    """
    Evaluate the fidelity of an explanation method over multiple trajectories.

    For each trajectory:
    1. Run the policy to get a full trajectory and original reward R.
    2. Apply the explanation method to get importance scores.
    3. Find the critical segment using sliding window.
    4. Fast-forward to the critical segment start.
    5. Take random actions for l steps.
    6. Continue with policy to end.
    7. Compute fidelity score.

    Args:
        env_reset_fn: Reset environment, returns initial state.
        env_step_fn: Step environment, returns (next_state, reward, done, info).
        env_set_state_fn: Set environment to a specific state (for fast-forward).
                           If None, fast-forward by replaying actions.
        policy_fn: Function (state) -> action for the target agent.
        importance_fn: Function (trajectory_states) -> importance_scores.
        action_space_sample_fn: Function () -> random action.
        d_max: Maximum possible reward change in this environment.
        n_trajectories: Number of trajectories to evaluate (default 500).
        K: Fraction of trajectory for window size (e.g., 0.1, 0.2).
        max_episode_steps: Maximum steps per episode.

    Returns:
        Dict with mean fidelity score and standard deviation.
    """
    fidelity_scores = []

    for traj_idx in range(n_trajectories):
        # Collect original trajectory
        states = []
        actions = []
        rewards = []

        state = env_reset_fn()
        done = False
        step = 0

        while not done and step < max_episode_steps:
            action = policy_fn(state)
            next_state, reward, done, info = env_step_fn(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            step += 1

        L = len(states)
        if L < 2:
            continue

        original_reward = sum(rewards)
        states_arr = np.array(states, dtype=np.float32)

        # Get importance scores
        importances = importance_fn(states_arr)

        # Find critical segment
        l = max(1, int(L * K))
        start_idx, end_idx = find_critical_segment(importances, L, K)

        # Fast-forward to critical segment start
        state = env_reset_fn()
        for i in range(start_idx):
            next_state, _, done, _ = env_step_fn(actions[i])
            if done:
                break
            state = next_state

        if env_set_state_fn is not None:
            env_set_state_fn(states[start_idx])
        else:
            # Re-execute from start to reach the critical state
            state = env_reset_fn()
            for i in range(start_idx):
                action = policy_fn(state)
                next_state, _, done, _ = env_step_fn(action)
                if done:
                    break
                state = next_state

        # Take random actions during critical segment
        perturbed_reward = 0.0
        for i in range(l):
            random_action = action_space_sample_fn()
            next_state, reward, done, info = env_step_fn(random_action)
            perturbed_reward += reward
            if done:
                break
            state = next_state

        # Continue with policy
        if not done:
            while not done and step < max_episode_steps:
                action = policy_fn(state)
                next_state, reward, done, info = env_step_fn(action)
                perturbed_reward += reward
                state = next_state
                step += 1

        # Compute fidelity
        d = abs(perturbed_reward - original_reward)
        score = compute_fidelity_score(d, d_max, l, L)
        fidelity_scores.append(score)

    mean_fidelity = float(np.mean(fidelity_scores)) if fidelity_scores else 0.0
    std_fidelity = float(np.std(fidelity_scores)) if fidelity_scores else 0.0

    return {
        "mean_fidelity": mean_fidelity,
        "std_fidelity": std_fidelity,
        "n_trajectories": len(fidelity_scores),
        "K": K,
    }


def evaluate_fidelity_multiple_K(
    env_reset_fn: Callable[[], np.ndarray],
    env_step_fn: Callable[[np.ndarray], Tuple],
    env_set_state_fn: Optional[Callable[[np.ndarray], None]],
    policy_fn: Callable[[np.ndarray], np.ndarray],
    importance_fn: Callable[[np.ndarray], np.ndarray],
    action_space_sample_fn: Callable[[], np.ndarray],
    d_max: float,
    K_values: List[float] = (0.1, 0.2, 0.3, 0.4),
    n_trajectories: int = 500,
    n_repeats: int = 3,
    max_episode_steps: int = 1000,
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Evaluate fidelity across multiple K values with multiple repeats.

    This implements Experiment I from the paper.

    Args:
        Same as evaluate_fidelity, plus:
        K_values: List of K fractions to evaluate (e.g., [0.1, 0.2, 0.3, 0.4]).
        n_repeats: Number of repetitions with different random seeds.

    Returns:
        Dict mapping K -> (mean, std) across repeats.
    """
    results = {}

    for K in K_values:
        K_repeats = []
        for repeat in range(n_repeats):
            np.random.seed(repeat)
            result = evaluate_fidelity(
                env_reset_fn=env_reset_fn,
                env_step_fn=env_step_fn,
                env_set_state_fn=env_set_state_fn,
                policy_fn=policy_fn,
                importance_fn=importance_fn,
                action_space_sample_fn=action_space_sample_fn,
                d_max=d_max,
                n_trajectories=n_trajectories,
                K=K,
                max_episode_steps=max_episode_steps,
            )
            K_repeats.append(result["mean_fidelity"])

        results[str(K)] = (float(np.mean(K_repeats)), float(np.std(K_repeats)))

    return results