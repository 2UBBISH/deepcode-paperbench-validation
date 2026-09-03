"""
CAGE Challenge 2 Environment for RICE Framework

Implements a Gymnasium environment simulating the CAGE (Cyber Autonomy Gym
for Experimentation) Challenge 2 scenario: a blue agent defends a network
against a red agent ("B-line" strategy). The blue agent selects defensive
actions while the red agent attempts to compromise hosts and gain admin access.

Based on the champion scheme from Cardiff University (CAGE Challenge 2 winner)
as referenced in the RICE paper.

State Space:
    Vector representing network state including:
    - Host compromise statuses
    - Service availability
    - Red agent activity indicators
    - Defense cooldown timers

Action Space (Discrete(5)):
    0: Monitor   - Gather intelligence, no direct effect
    1: Analyze   - Identify compromised hosts/services
    2: Decoy     - Deploy decoy services to trap red agent
    3: Remove    - Remove red agent from compromised host
    4: Restore   - Restore compromised services

Reward:
    - Negative reward when red agent gains admin access
    - Small positive reward for successful defense actions
    - Bonus for keeping services available
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import random


class CageEnv(gym.Env):
    """
    CAGE Challenge 2 environment.

    Simulates a network defense scenario where a blue agent must protect
    a network of hosts from a red agent following the "B-line" strategy.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    # Action constants
    ACTION_MONITOR = 0
    ACTION_ANALYZE = 1
    ACTION_DECOY = 2
    ACTION_REMOVE = 3
    ACTION_RESTORE = 4

    ACTION_NAMES = ["Monitor", "Analyze", "Decoy", "Remove", "Restore"]

    # Host status constants
    HOST_CLEAN = 0
    HOST_COMPROMISED = 1
    HOST_ADMIN_ACCESS = 2  # Red has admin/root access
    HOST_DECOYED = 3

    # Service status
    SERVICE_UP = 0
    SERVICE_DOWN = 1
    SERVICE_COMPROMISED = 2

    def __init__(
        self,
        num_hosts: int = 5,
        num_services: int = 3,
        max_steps: int = 100,
        red_activity_prob: float = 0.3,
        red_success_prob: float = 0.4,
        analyze_detection_prob: float = 0.7,
        remove_success_prob: float = 0.6,
        restore_success_prob: float = 0.8,
        decoy_effectiveness: float = 0.5,
        monitor_intel_gain: float = 0.1,
        reward_scale: float = 1.0,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the CAGE environment.

        Args:
            num_hosts: Number of hosts in the network.
            num_services: Number of services per host.
            max_steps: Maximum steps per episode.
            red_activity_prob: Probability red agent acts each step.
            red_success_prob: Base probability red succeeds in compromise.
            analyze_detection_prob: Probability Analyze detects compromise.
            remove_success_prob: Probability Remove succeeds.
            restore_success_prob: Probability Restore succeeds.
            decoy_effectiveness: Probability decoy traps red agent.
            monitor_intel_gain: Information gain from Monitor action.
            reward_scale: Scaling factor for rewards.
            render_mode: Rendering mode.
        """
        super().__init__()

        self.num_hosts = num_hosts
        self.num_services = num_services
        self.max_steps = max_steps
        self.red_activity_prob = red_activity_prob
        self.red_success_prob = red_success_prob
        self.analyze_detection_prob = analyze_detection_prob
        self.remove_success_prob = remove_success_prob
        self.restore_success_prob = restore_success_prob
        self.decoy_effectiveness = decoy_effectiveness
        self.monitor_intel_gain = monitor_intel_gain
        self.reward_scale = reward_scale
        self.render_mode = render_mode

        # State components:
        # For each host: [status, num_compromised_services, red_activity_flag, decoy_active]
        # Global: [steps_remaining, total_compromised_hosts, red_admin_count]
        # Total state size: num_hosts * 4 + 3
        state_dim = num_hosts * 4 + 3
        self.observation_space = spaces.Box(
            low=-1.0,
            high=float(num_hosts),
            shape=(state_dim,),
            dtype=np.float32,
        )

        # Action space: 5 discrete actions
        self.action_space = spaces.Discrete(5)

        # Internal state
        self.host_statuses = None  # shape: (num_hosts,)
        self.service_statuses = None  # shape: (num_hosts, num_services)
        self.red_activity = None  # shape: (num_hosts,) - bool
        self.decoy_active = None  # shape: (num_hosts,) - bool
        self.steps_taken = 0
        self.current_host = 0  # Currently focused host for actions
        self.red_admin_count = 0
        self.total_compromised = 0
        self.episode_reward = 0.0

        # Red agent state
        self.red_cooldown = 0
        self.red_targets = []

        # Action cooldowns
        self.remove_cooldown = 0
        self.analyze_cooldown = 0

        # Random state
        self.np_random = None
        self.seed_value = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed.
            options: Additional options (unused).

        Returns:
            observation: Initial state vector.
            info: Empty info dict.
        """
        super().reset(seed=seed)
        self.np_random = np.random.RandomState(seed)
        self.seed_value = seed

        # Initialize hosts as clean
        self.host_statuses = np.zeros(self.num_hosts, dtype=np.int32)
        self.service_statuses = np.full(
            (self.num_hosts, self.num_services), self.SERVICE_UP, dtype=np.int32
        )
        self.red_activity = np.zeros(self.num_hosts, dtype=bool)
        self.decoy_active = np.zeros(self.num_hosts, dtype=bool)

        # Randomly compromise 1-2 hosts initially
        num_initial_compromised = self.np_random.integers(1, min(3, self.num_hosts + 1))
        initial_targets = self.np_random.choice(
            self.num_hosts, size=num_initial_compromised, replace=False
        )
        for host in initial_targets:
            self.host_statuses[host] = self.HOST_COMPROMISED
            # Compromise some services
            num_comp = self.np_random.integers(1, self.num_services + 1)
            comp_services = self.np_random.choice(
                self.num_services, size=num_comp, replace=False
            )
            self.service_statuses[host, comp_services] = self.SERVICE_COMPROMISED

        self.steps_taken = 0
        self.current_host = 0
        self.red_admin_count = np.sum(
            self.host_statuses == self.HOST_ADMIN_ACCESS
        )
        self.total_compromised = np.sum(
            self.host_statuses >= self.HOST_COMPROMISED
        )
        self.episode_reward = 0.0
        self.red_cooldown = 0
        self.red_targets = []
        self.remove_cooldown = 0
        self.analyze_cooldown = 0

        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Integer action in {0,1,2,3,4}.

        Returns:
            observation: Next state.
            reward: Reward for this step.
            terminated: Whether episode ended.
            truncated: Whether episode was truncated (max steps).
            info: Additional info dict.
        """
        reward = 0.0
        info = {"action_name": self.ACTION_NAMES[action]}

        # --- Blue agent action ---
        if action == self.ACTION_MONITOR:
            reward += self._do_monitor()
        elif action == self.ACTION_ANALYZE:
            reward += self._do_analyze()
        elif action == self.ACTION_DECOY:
            reward += self._do_decoy()
        elif action == self.ACTION_REMOVE:
            reward += self._do_remove()
        elif action == self.ACTION_RESTORE:
            reward += self._do_restore()
        else:
            raise ValueError(f"Invalid action: {action}")

        # --- Red agent action ---
        red_reward = self._red_agent_step()
        reward += red_reward

        # --- Update state ---
        self.steps_taken += 1

        # Decrease cooldowns
        if self.remove_cooldown > 0:
            self.remove_cooldown -= 1
        if self.analyze_cooldown > 0:
            self.analyze_cooldown -= 1
        if self.red_cooldown > 0:
            self.red_cooldown -= 1

        # Update counts
        self.red_admin_count = np.sum(
            self.host_statuses == self.HOST_ADMIN_ACCESS
        )
        self.total_compromised = np.sum(
            self.host_statuses >= self.HOST_COMPROMISED
        )

        # Small penalty for each compromised host
        reward -= 0.01 * self.total_compromised * self.reward_scale

        # Large penalty for admin access
        reward -= 0.5 * self.red_admin_count * self.reward_scale

        # Small bonus for clean hosts
        clean_hosts = np.sum(self.host_statuses == self.HOST_CLEAN)
        reward += 0.005 * clean_hosts * self.reward_scale

        self.episode_reward += reward

        # Check termination
        terminated = False
        truncated = self.steps_taken >= self.max_steps

        # Episode ends if all hosts have admin access (total failure)
        if self.red_admin_count >= self.num_hosts:
            terminated = True
            reward -= 1.0 * self.reward_scale  # Extra penalty for total loss

        observation = self._get_observation()
        info["red_admin_count"] = self.red_admin_count
        info["total_compromised"] = self.total_compromised
        info["episode_reward"] = self.episode_reward

        return observation, reward, terminated, truncated, info

    def _do_monitor(self) -> float:
        """
        Monitor action: gather intelligence on network state.
        Increases detection probability for next Analyze.
        """
        # Monitor gives intel bonus - improves future detection
        self.monitor_intel_gain = min(0.3, self.monitor_intel_gain + 0.02)
        return 0.01 * self.reward_scale

    def _do_analyze(self) -> float:
        """
        Analyze action: scan current host for compromise.
        """
        reward = 0.0
        host = self.current_host

        if self.analyze_cooldown > 0:
            return -0.01 * self.reward_scale  # Penalty for spamming

        self.analyze_cooldown = 2  # 2-step cooldown

        if self.host_statuses[host] >= self.HOST_COMPROMISED:
            # Detection chance depends on intel
            detection_prob = self.analyze_detection_prob + self.monitor_intel_gain
            if self.np_random.random() < detection_prob:
                # Successfully identified compromise
                reward += 0.05 * self.reward_scale
                info_extra = {"detected": True, "host": host}
            else:
                reward += 0.01 * self.reward_scale
        else:
            reward += 0.02 * self.reward_scale  # Confirmed clean

        # Cycle to next host
        self.current_host = (self.current_host + 1) % self.num_hosts

        # Reset intel gain after use
        self.monitor_intel_gain = max(0.1, self.monitor_intel_gain - 0.05)

        return reward

    def _do_decoy(self) -> float:
        """
        Decoy action: deploy decoy services on current host to trap red agent.
        """
        reward = 0.0
        host = self.current_host

        if self.decoy_active[host]:
            return -0.01 * self.reward_scale  # Already decoyed

        self.decoy_active[host] = True

        # If red is active on this host, chance to trap
        if self.red_activity[host]:
            if self.np_random.random() < self.decoy_effectiveness:
                # Red trapped! Downgrade host status
                if self.host_statuses[host] == self.HOST_ADMIN_ACCESS:
                    self.host_statuses[host] = self.HOST_COMPROMISED
                    reward += 0.3 * self.reward_scale
                elif self.host_statuses[host] == self.HOST_COMPROMISED:
                    self.host_statuses[host] = self.HOST_CLEAN
                    # Restore services
                    self.service_statuses[host, :] = self.SERVICE_UP
                    reward += 0.2 * self.reward_scale
                self.red_activity[host] = False
                self.host_statuses[host] = self.HOST_DECOYED
            else:
                reward += 0.02 * self.reward_scale
        else:
            reward += 0.01 * self.reward_scale

        # Cycle to next host
        self.current_host = (self.current_host + 1) % self.num_hosts

        return reward

    def _do_remove(self) -> float:
        """
        Remove action: attempt to remove red agent from current host.
        """
        reward = 0.0
        host = self.current_host

        if self.remove_cooldown > 0:
            return -0.01 * self.reward_scale

        self.remove_cooldown = 3  # 3-step cooldown

        if self.host_statuses[host] >= self.HOST_COMPROMISED:
            if self.np_random.random() < self.remove_success_prob:
                # Successfully removed red
                old_status = self.host_statuses[host]
                self.host_statuses[host] = self.HOST_CLEAN
                self.red_activity[host] = False
                self.decoy_active[host] = False
                # Restore services
                self.service_statuses[host, :] = self.SERVICE_UP

                if old_status == self.HOST_ADMIN_ACCESS:
                    reward += 0.4 * self.reward_scale
                else:
                    reward += 0.15 * self.reward_scale
            else:
                reward += 0.01 * self.reward_scale  # Attempted but failed
        else:
            reward += 0.02 * self.reward_scale  # Host was clean

        # Cycle to next host
        self.current_host = (self.current_host + 1) % self.num_hosts

        return reward

    def _do_restore(self) -> float:
        """
        Restore action: restore compromised services on current host.
        """
        reward = 0.0
        host = self.current_host

        compromised_services = np.sum(
            self.service_statuses[host] == self.SERVICE_COMPROMISED
        )

        if compromised_services > 0:
            # Attempt to restore each compromised service
            restored = 0
            for s in range(self.num_services):
                if self.service_statuses[host, s] == self.SERVICE_COMPROMISED:
                    if self.np_random.random() < self.restore_success_prob:
                        self.service_statuses[host, s] = self.SERVICE_UP
                        restored += 1

            reward += 0.03 * restored * self.reward_scale

            # If all services restored and host was only compromised (not admin),
            # downgrade status
            if (
                np.all(self.service_statuses[host] == self.SERVICE_UP)
                and self.host_statuses[host] == self.HOST_COMPROMISED
            ):
                self.host_statuses[host] = self.HOST_CLEAN
                reward += 0.1 * self.reward_scale
        else:
            reward += 0.01 * self.reward_scale  # Nothing to restore

        # Cycle to next host
        self.current_host = (self.current_host + 1) % self.num_hosts

        return reward

    def _red_agent_step(self) -> float:
        """
        Execute red agent (B-line) actions.

        B-line strategy: systematically targets hosts, attempts to escalate
        privileges, and moves laterally.

        Returns:
            Reward impact from red actions (negative).
        """
        reward = 0.0

        if self.red_cooldown > 0:
            return reward

        # Red acts with some probability
        if self.np_random.random() > self.red_activity_prob:
            return reward

        self.red_cooldown = 1  # 1-step cooldown

        # Red selects target: prefer compromised hosts for escalation,
        # otherwise target clean hosts
        compromised_hosts = np.where(
            self.host_statuses == self.HOST_COMPROMISED
        )[0]
        clean_hosts = np.where(self.host_statuses == self.HOST_CLEAN)[0]
        admin_hosts = np.where(self.host_statuses == self.HOST_ADMIN_ACCESS)[0]

        # Strategy: escalate on compromised, spread from admin, attack clean
        if len(compromised_hosts) > 0 and self.np_random.random() < 0.6:
            # Escalate on compromised host
            target = self.np_random.choice(compromised_hosts)
            if self.decoy_active[target]:
                # Decoy traps red!
                self.decoy_active[target] = False
                self.red_activity[target] = False
                reward += 0.1 * self.reward_scale  # Positive for blue
                return reward

            if self.np_random.random() < self.red_success_prob:
                self.host_statuses[target] = self.HOST_ADMIN_ACCESS
                self.red_activity[target] = True
                reward -= 0.2 * self.reward_scale
        elif len(admin_hosts) > 0 and self.np_random.random() < 0.4:
            # Lateral movement from admin host
            source = self.np_random.choice(admin_hosts)
            # Target a neighboring host
            target = (source + 1) % self.num_hosts
            if self.host_statuses[target] == self.HOST_CLEAN:
                if self.np_random.random() < self.red_success_prob * 0.8:
                    self.host_statuses[target] = self.HOST_COMPROMISED
                    self.red_activity[target] = True
                    # Compromise some services
                    num_comp = self.np_random.integers(1, self.num_services + 1)
                    comp_services = self.np_random.choice(
                        self.num_services, size=num_comp, replace=False
                    )
                    self.service_statuses[target, comp_services] = self.SERVICE_COMPROMISED
                    reward -= 0.1 * self.reward_scale
        elif len(clean_hosts) > 0:
            # Initial compromise
            target = self.np_random.choice(clean_hosts)
            if self.np_random.random() < self.red_success_prob:
                self.host_statuses[target] = self.HOST_COMPROMISED
                self.red_activity[target] = True
                # Compromise some services
                num_comp = self.np_random.integers(1, self.num_services + 1)
                comp_services = self.np_random.choice(
                    self.num_services, size=num_comp, replace=False
                )
                self.service_statuses[target, comp_services] = self.SERVICE_COMPROMISED
                reward -= 0.1 * self.reward_scale

        return reward

    def _get_observation(self) -> np.ndarray:
        """
        Build the observation vector from internal state.

        Returns:
            Normalized state vector.
        """
        obs = []

        # Per-host features
        for h in range(self.num_hosts):
            # Host status (normalized)
            obs.append(self.host_statuses[h] / 3.0)
            # Number of compromised services (normalized)
            num_comp = np.sum(
                self.service_statuses[h] == self.SERVICE_COMPROMISED
            )
            obs.append(num_comp / max(1, self.num_services))
            # Red activity flag
            obs.append(1.0 if self.red_activity[h] else 0.0)
            # Decoy active flag
            obs.append(1.0 if self.decoy_active[h] else 0.0)

        # Global features
        obs.append(self.steps_taken / max(1, self.max_steps))
        obs.append(self.total_compromised / max(1, self.num_hosts))
        obs.append(self.red_admin_count / max(1, self.num_hosts))

        return np.array(obs, dtype=np.float32)

    def get_state(self) -> np.ndarray:
        """
        Get the full internal state for saving/restoring.

        Returns:
            State vector representing all internal variables.
        """
        state = np.concatenate([
            self.host_statuses.astype(np.float32),
            self.service_statuses.flatten().astype(np.float32),
            self.red_activity.astype(np.float32),
            self.decoy_active.astype(np.float32),
            np.array([
                self.steps_taken,
                self.current_host,
                self.red_admin_count,
                self.total_compromised,
                self.red_cooldown,
                self.remove_cooldown,
                self.analyze_cooldown,
                self.monitor_intel_gain,
            ], dtype=np.float32),
        ])
        return state

    def set_state(self, state: np.ndarray) -> None:
        """
        Restore the environment to a given state.

        Args:
            state: State vector from get_state().
        """
        idx = 0
        n = self.num_hosts
        m = self.num_services

        self.host_statuses = state[idx:idx + n].astype(np.int32)
        idx += n

        self.service_statuses = state[idx:idx + n * m].reshape(n, m).astype(np.int32)
        idx += n * m

        self.red_activity = state[idx:idx + n].astype(bool)
        idx += n

        self.decoy_active = state[idx:idx + n].astype(bool)
        idx += n

        self.steps_taken = int(state[idx])
        self.current_host = int(state[idx + 1])
        self.red_admin_count = int(state[idx + 2])
        self.total_compromised = int(state[idx + 3])
        self.red_cooldown = int(state[idx + 4])
        self.remove_cooldown = int(state[idx + 5])
        self.analyze_cooldown = int(state[idx + 6])
        self.monitor_intel_gain = float(state[idx + 7])

    def render(self) -> Optional[str]:
        """
        Render the environment state.

        Returns:
            String representation if mode is 'ansi', None otherwise.
        """
        if self.render_mode == "ansi":
            output = f"\n=== CAGE Challenge 2 | Step {self.steps_taken}/{self.max_steps} ===\n"
            output += f"Current focus host: {self.current_host}\n"
            output += f"Red admin count: {self.red_admin_count}/{self.num_hosts}\n"
            output += f"Total compromised: {self.total_compromised}/{self.num_hosts}\n"
            output += f"Episode reward: {self.episode_reward:.3f}\n\n"

            output += "Host Status:\n"
            for h in range(self.num_hosts):
                status_names = {
                    self.HOST_CLEAN: "CLEAN",
                    self.HOST_COMPROMISED: "COMPROMISED",
                    self.HOST_ADMIN_ACCESS: "ADMIN_ACCESS",
                    self.HOST_DECOYED: "DECOYED",
                }
                status_str = status_names.get(
                    self.host_statuses[h], "UNKNOWN"
                )
                red_str = "🔴" if self.red_activity[h] else "  "
                decoy_str = "🪤" if self.decoy_active[h] else "  "
                comp_svc = np.sum(
                    self.service_statuses[h] == self.SERVICE_COMPROMISED
                )
                output += (
                    f"  Host {h}: {status_str:<14} {red_str} {decoy_str} "
                    f"| Services: {comp_svc}/{self.num_services} compromised\n"
                )

            output += f"\nCooldowns: Remove={self.remove_cooldown}, "
            output += f"Analyze={self.analyze_cooldown}, Red={self.red_cooldown}\n"
            output += f"Intel gain: {self.monitor_intel_gain:.2f}\n"

            return output
        elif self.render_mode == "human":
            print(self.render())
            return None
        return None

    def close(self) -> None:
        """Clean up resources."""
        pass


class CageEnvWrapper(gym.Wrapper):
    """
    Wrapper for CAGE environment that adds save_state/restore_state
    convenience methods for RICE integration.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._saved_state = None

    def save_state(self) -> np.ndarray:
        """Save current environment state."""
        self._saved_state = self.env.get_state()
        return self._saved_state

    def restore_state(self, state: np.ndarray) -> None:
        """Restore environment to a saved state."""
        self.env.set_state(state)
        self._saved_state = state

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def make_cage_env(
    num_hosts: int = 5,
    num_services: int = 3,
    max_steps: int = 100,
    red_activity_prob: float = 0.3,
    red_success_prob: float = 0.4,
    analyze_detection_prob: float = 0.7,
    remove_success_prob: float = 0.6,
    restore_success_prob: float = 0.8,
    decoy_effectiveness: float = 0.5,
    monitor_intel_gain: float = 0.1,
    reward_scale: float = 1.0,
    render_mode: Optional[str] = None,
    seed: Optional[int] = None,
    **kwargs,
) -> CageEnv:
    """
    Factory function to create a CAGE environment.

    Args:
        num_hosts: Number of hosts.
        num_services: Number of services per host.
        max_steps: Maximum steps per episode.
        red_activity_prob: Red agent activity probability.
        red_success_prob: Red agent success probability.
        analyze_detection_prob: Analyze detection probability.
        remove_success_prob: Remove success probability.
        restore_success_prob: Restore success probability.
        decoy_effectiveness: Decoy effectiveness.
        monitor_intel_gain: Monitor intelligence gain.
        reward_scale: Reward scaling factor.
        render_mode: Rendering mode.
        seed: Random seed.

    Returns:
        CageEnv instance.
    """
    env = CageEnv(
        num_hosts=num_hosts,
        num_services=num_services,
        max_steps=max_steps,
        red_activity_prob=red_activity_prob,
        red_success_prob=red_success_prob,
        analyze_detection_prob=analyze_detection_prob,
        remove_success_prob=remove_success_prob,
        restore_success_prob=restore_success_prob,
        decoy_effectiveness=decoy_effectiveness,
        monitor_intel_gain=monitor_intel_gain,
        reward_scale=reward_scale,
        render_mode=render_mode,
    )

    if seed is not None:
        env.reset(seed=seed)

    return env


# Register the environment with Gymnasium
try:
    gym.register(
        id="Cage-v0",
        entry_point="envs.cage_env:CageEnv",
        max_episode_steps=100,
    )
except gym.error.Error:
    # Already registered
    pass


if __name__ == "__main__":
    # Quick test
    env = make_cage_env(render_mode="ansi")
    obs, info = env.reset(seed=42)
    print(env.render())

    total_reward = 0.0
    for step in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step {step}: Action={CageEnv.ACTION_NAMES[action]}, "
              f"Reward={reward:.3f}, Terminated={terminated}")

        if terminated or truncated:
            break

    print(f"\nTotal reward: {total_reward:.3f}")
    print(f"Final state:\n{env.render()}")

    # Test state save/restore
    state = env.get_state()
    print(f"State vector shape: {state.shape}")

    env2 = make_cage_env()
    env2.reset(seed=42)
    env2.set_state(state)
    print(f"Restored state matches: {np.allclose(env.get_state(), env2.get_state())}")