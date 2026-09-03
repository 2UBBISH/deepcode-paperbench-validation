"""
RICE Refining Process - Core algorithm from "RICE: Refining via Critical State Explanation"

Improves a pre-trained agent by:
1. Mixed initial state distribution (sampling from critical states with probability p)
2. RND exploration bonus added to environment reward
3. Periodic RND predictor updates on collected states
"""

import os, time, argparse, json, pickle
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.type_aliases import GymEnv
from rice.utils import (
    load_config, set_seed, CriticalStateBuffer, Logger,
    set_env_state, get_mujoco_state, evaluate_policy, ensure_dir,
    get_device, build_mlp, make_env, make_vec_env, format_time, get_project_root
)
from rice.rnd import RNDModule, create_rnd_module, RunningMeanStd


class RefiningEnvWrapper(gym.Wrapper):
    """Wrapper: reset to critical states, augmented reward r' = r_env + lambda * r_rnd(s)."""
    def __init__(self, env, rnd_module=None, lambda_rnd=0.01, normalize_obs=True, obs_rms=None):
        super().__init__(env)
        self.rnd_module = rnd_module
        self.lambda_rnd = lambda_rnd
        self.normalize_obs = normalize_obs
        self.obs_rms = obs_rms if obs_rms is not None else RunningMeanStd()
        self._started_from_critical = False
        self._critical_state_info = None
        self._current_obs = None
        self._episode_rnd_bonus = []
        self._episode_env_reward = []
        self._initial_state = None

    def reset(self, *, seed=None, options=None):
        self._started_from_critical = False
        self._critical_state_info = None
        self._episode_rnd_bonus = []
        self._episode_env_reward = []
        critical_state = options.get("critical_state") if options else None
        if critical_state is not None:
            state_vector = critical_state.get("state")
            if state_vector is not None:
                try:
                    set_env_state(self.env, state_vector)
                    self._started_from_critical = True
                    self._critical_state_info = critical_state
                    obs, info = self.env.reset(seed=seed)
                    self._current_obs = obs
                    self._initial_state = get_mujoco_state(self.env) if hasattr(self.env.unwrapped, 'sim') else state_vector
                    if self.normalize_obs: self.obs_rms.update(obs.reshape(1, -1))
                    return obs, info
                except Exception as e:
                    print(f"Warning: Failed to set critical state: {e}. Using default reset.")
        obs, info = self.env.reset(seed=seed)
        self._current_obs = obs
        if hasattr(self.env.unwrapped, 'sim'): self._initial_state = get_mujoco_state(self.env)
        if self.normalize_obs: self.obs_rms.update(obs.reshape(1, -1))
        return obs, info

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)
        self._episode_env_reward.append(float(env_reward))
        rnd_bonus = 0.0
        if self.rnd_module is not None:
            obs_for_rnd = self.obs_rms.normalize(obs.reshape(1, -1)).flatten() if self.normalize_obs else obs
            rnd_bonus = float(self.rnd_module.compute_bonus(obs_for_rnd, normalize=True))
        self._episode_rnd_bonus.append(rnd_bonus)
        augmented_reward = env_reward + self.lambda_rnd * rnd_bonus
        info["rnd_bonus"] = rnd_bonus
        info["env_reward"] = float(env_reward)
        info["augmented_reward"] = float(augmented_reward)
        info["started_from_critical"] = self._started_from_critical
        self._current_obs = obs
        return obs, augmented_reward, terminated, truncated, info

    def get_episode_stats(self):
        return {"total_env_reward": sum(self._episode_env_reward),
                "total_rnd_bonus": sum(self._episode_rnd_bonus),
                "mean_rnd_bonus": np.mean(self._episode_rnd_bonus) if self._episode_rnd_bonus else 0.0,
                "started_from_critical": self._started_from_critical}


class MixedInitialStateSampler:
    """With probability p, samples from critical state buffer D; otherwise default reset."""
    def __init__(self, critical_state_buffer, p=0.5, seed=None):
        self.buffer = critical_state_buffer
        self.p = p
        self.rng = np.random.RandomState(seed)
        self.stats = {"n_critical_samples": 0, "n_default_samples": 0}

    def sample_initial_state(self):
        if len(self.buffer) > 0 and self.rng.random() < self.p:
            critical_states = self.buffer.sample(1)
            if critical_states:
                self.stats["n_critical_samples"] += 1
                return critical_states[0], True
        self.stats["n_default_samples"] += 1
        return None, False

    def get_stats(self):
        total = self.stats["n_critical_samples"] + self.stats["n_default_samples"]
        return {**self.stats, "total_samples": total,
                "critical_fraction": self.stats["n_critical_samples"] / max(total, 1)}


class RefiningCallback(BaseCallback):
    """Logs metrics, evaluates policy, updates RND predictor during refining."""
    def __init__(self, logger, eval_env=None, eval_freq=10000, n_eval_episodes=10,
                 rnd_module=None, rnd_update_freq=1000, rnd_batch_size=64, rnd_n_epochs=1,
                 state_buffer=None, verbose=1):
        super().__init__(verbose)
        self.logger = logger; self.eval_env = eval_env; self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes; self.rnd_module = rnd_module
        self.rnd_update_freq = rnd_update_freq; self.rnd_batch_size = rnd_batch_size
        self.rnd_n_epochs = rnd_n_epochs
        self.state_buffer = state_buffer if state_buffer is not None else []
        self._last_eval_step = 0; self._last_rnd_update_step = 0
        self._episode_rewards = []; self._episode_lengths = []
        self._current_episode_reward = 0.0; self._current_episode_length = 0

    def _on_step(self):
        rewards = self.locals.get("rewards", np.zeros(1))
        dones = self.locals.get("dones", np.zeros(1, dtype=bool))
        self._current_episode_reward += float(rewards[0]) if len(rewards) > 0 else 0.0
        self._current_episode_length += 1
        if dones[0] if len(dones) > 0 else False:
            self._episode_rewards.append(self._current_episode_reward)
            self._episode_lengths.append(self._current_episode_length)
            self._current_episode_reward = 0.0; self._current_episode_length = 0
        if self.n_calls % 100 == 0:
            self.logger.log("refining/timesteps", self.num_timesteps, self.num_timesteps)
            if len(self._episode_rewards) > 0:
                self.logger.log("refining/mean_episode_reward", np.mean(self._episode_rewards[-10:]), self.num_timesteps)
                self.logger.log("refining/mean_episode_length", np.mean(self._episode_lengths[-10:]), self.num_timesteps)
            for key in ["clip_fraction","approx_kl","value_loss","policy_gradient_loss"]:
                if key in self.locals: self.logger.log(f"refining/{key}", float(self.locals[key]), self.num_timesteps)
        if self.eval_env and (self.num_timesteps - self._last_eval_step) >= self.eval_freq:
            self._run_evaluation(); self._last_eval_step = self.num_timesteps
        if (self.rnd_module and len(self.state_buffer) >= self.rnd_batch_size and
            (self.num_timesteps - self._last_rnd_update_step) >= self.rnd_update_freq):
            self._update_rnd(); self._last_rnd_update_step = self.num_timesteps
        return True

    def _run_evaluation(self):
        if not self.eval_env or not self.model: return
        try:
            r = evaluate_policy(self.eval_env, self.model, n_episodes=self.n_eval_episodes, deterministic=True)
            self.logger.log("refining/eval_mean_reward", r["mean_reward"], self.num_timesteps)
            self.logger.log("refining/eval_std_reward", r["std_reward"], self.num_timesteps)
            if self.verbose > 0: print(f"\n[Refining] Step {self.num_timesteps}: Eval = {r['mean_reward']:.2f} +/- {r['std_reward']:.2f}")
        except Exception as e: print(f"Warning: Evaluation failed: {e}")

    def _update_rnd(self):
        if not self.rnd_module or len(self.state_buffer) < self.rnd_batch_size: return
        try:
            idx = np.random.choice(len(self.state_buffer), size=min(self.rnd_batch_size*10, len(self.state_buffer)), replace=False)
            states = np.stack([self.state_buffer[i] for i in idx])
            info = self.rnd_module.update_predictor(states, batch_size=self.rnd_batch_size, n_epochs=self.rnd_n_epochs)
            self.logger.log("refining/rnd_loss", info.get("mean_loss",0.0), self.num_timesteps)
            if self.verbose > 0: print(f"\n[RND] Step {self.num_timesteps}: Loss = {info.get('mean_loss',0.0):.6f}")
        except Exception as e: print(f"Warning: RND update failed: {e}")

    def _on_training_end(self):
        if self.verbose > 0:
            print(f"\n[Refining] Training completed at step {self.num_timesteps}")
            if len(self._episode_rewards) > 0: print(f"  Final mean reward: {np.mean(self._episode_rewards[-10:]):.2f}")


class StateCollectionCallback(BaseCallback):
    """Collects observations during training for RND updates."""
    def __init__(self, state_buffer, max_buffer_size=100000):
        super().__init__(verbose=0); self.state_buffer = state_buffer; self.max_buffer_size = max_buffer_size

    def _on_step(self):
        obs = self.locals.get("obs_tensor", None)
        if obs is not None:
            obs_np = obs.cpu().numpy() if hasattr(obs,'cpu') else np.array(obs)
            if obs_np.ndim == 2:
                for i in range(obs_np.shape[0]):
                    if len(self.state_buffer) < self.max_buffer_size: self.state_buffer.append(obs_np[i].copy())
                    else: self.state_buffer[np.random.randint(0,len(self.state_buffer))] = obs_np[i].copy()
            else:
                if len(self.state_buffer) < self.max_buffer_size: self.state_buffer.append(obs_np.copy())
                else: self.state_buffer[np.random.randint(0,len(self.state_buffer))] = obs_np.copy()
        return True


class MixedInitTrainer:
    """Custom training loop: resets to critical states with prob p, collects rollouts, trains, updates RND."""
    def __init__(self, model, env, sampler, rnd_module=None, lambda_rnd=0.01, total_timesteps=1_000_000,
                 n_steps=2048, callbacks=None, state_buffer=None, eval_env=None, eval_freq=10000,
                 n_eval_episodes=10, rnd_update_freq=1000, rnd_batch_size=64, rnd_n_epochs=1,
                 save_freq=100000, save_dir=None, logger=None, verbose=1):
        self.model = model; self.env = env; self.sampler = sampler; self.rnd_module = rnd_module
        self.lambda_rnd = lambda_rnd; self.total_timesteps = total_timesteps; self.n_steps = n_steps
        self.callbacks = callbacks or []; self.state_buffer = state_buffer or []
        self.eval_env = eval_env; self.eval_freq = eval_freq; self.n_eval_episodes = n_eval_episodes
        self.rnd_update_freq = rnd_update_freq; self.rnd_batch_size = rnd_batch_size
        self.rnd_n_epochs = rnd_n_epochs; self.save_freq = save_freq; self.save_dir = save_dir
        self.logger = logger; self.verbose = verbose
        self._current_timesteps = 0; self._critical_init_count = 0; self._default_init_count = 0
        self._last_eval_step = 0; self._last_rnd_update_step = 0; self._last_save_step = 0

    def train(self):
        start_time = time.time()
        n_envs = self.env.num_envs if hasattr(self.env,'num_envs') else 1
        print(f"\n{'='*60}\nStarting RICE Refining Training\n{'='*60}")
        print(f"Total: {self.total_timesteps} | n_steps: {self.n_steps} | n_envs: {n_envs} | p: {self.sampler.p} | lambda: {self.lambda_rnd}\n{'='*60}\n")
        for cb in self.callbacks: cb.init_callback(self.model)
        while self._current_timesteps < self.total_timesteps:
            observations = self._reset_with_mixed_init(n_envs)
            self.model._last_obs = observations
            self.model.collect_rollouts(self.env, callback=self.callbacks,
                                        rollout_buffer=self.model.rollout_buffer, n_rollout_steps=self.n_steps)
            if hasattr(self.model.rollout_buffer,'observations'):
                obs = self.model.rollout_buffer.observations
                obs_np = obs.cpu().numpy() if hasattr(obs,'cpu') else np.array(obs)
                for i in range(min(obs_np.shape[0],1000)):
                    if len(self.state_buffer) < 100000: self.state_buffer.append(obs_np[i].copy())
            self.model.train()
            self._current_timesteps += self.n_steps * n_envs
            if self.eval_env and self._current_timesteps - self._last_eval_step >= self.eval_freq:
                self._run_evaluation(); self._last_eval_step = self._current_timesteps
            if (self.rnd_module and len(self.state_buffer) >= self.rnd_batch_size and
                self._current_timesteps - self._last_rnd_update_step >= self.rnd_update_freq):
                self._update_rnd(); self._last_rnd_update_step = self._current_timesteps
            if self.save_dir and self._current_timesteps - self._last_save_step >= self.save_freq:
                self._save_checkpoint(); self._last_save_step = self._current_timesteps
            elapsed = time.time() - start_time; fps = self._current_timesteps / max(elapsed,1)
            if self.logger:
                self.logger.log("refining/total_timesteps", self._current_timesteps, self._current_timesteps)
                self.logger.log("refining/fps", fps, self._current_timesteps)
            if self.verbose > 0:
                print(f"\r[Refining] {self._current_timesteps}/{self.total_timesteps} ({100*self._current_timesteps/self.total_timesteps:.1f}%) - {fps:.0f} fps - {format_time(elapsed)}", end="", flush=True)
        elapsed = time.time() - start_time
        print(f"\n\nRefining completed in {format_time(elapsed)}")
        if self.save_dir: self._save_checkpoint(final=True)
        for cb in self.callbacks: cb.on_training_end()
        return {"total_timesteps": self._current_timesteps, "elapsed_time": elapsed,
                "critical_init_count": self._critical_init_count, "default_init_count": self._default_init_count,
                "sampler_stats": self.sampler.get_stats()}

    def _reset_with_mixed_init(self, n_envs):
        obs = np.zeros((n_envs,) + self.env.observation_space.shape, dtype=np.float32)
        for i in range(n_envs):
            cs, is_crit = self.sampler.sample_initial_state()
            if is_crit:
                self._critical_init_count += 1
                o = self.env.env_method('reset', indices=[i], options={"critical_state": cs})[0] if hasattr(self.env,'env_method') else self.env.reset(options={"critical_state": cs})[0]
            else:
                self._default_init_count += 1
                o = self.env.env_method('reset', indices=[i])[0] if hasattr(self.env,'env_method') else self.env.reset()[0]
            obs[i] = o
        return obs

    def _run_evaluation(self):
        if not self.eval_env: return
        try:
            r = evaluate_policy(self.eval_env, self.model, n_episodes=self.n_eval_episodes, deterministic=True)
            if self.logger:
                self.logger.log("refining/eval_mean_reward", r["mean_reward"], self._current_timesteps)
                self.logger.log("refining/eval_std_reward", r["std_reward"], self._current_timesteps)
            if self.verbose > 0: print(f"\n[Eval] Step {self._current_timesteps}: {r['mean_reward']:.2f} +/- {r['std_reward']:.2f}")
        except Exception as e: print(f"\nWarning: Evaluation failed: {e}")

    def _update_rnd(self):
        if not self.rnd_module or len(self.state_buffer) < self.rnd_batch_size: return
        try:
            idx = np.random.choice(len(self.state_buffer), size=min(self.rnd_batch_size*10, len(self.state_buffer)), replace=False)
            states = np.stack([self.state_buffer[i] for i in idx])
            info = self.rnd_module.update_predictor(states, batch_size=self.rnd_batch_size, n_epochs=self.rnd_n_epochs)
            if self.logger: self.logger.log("refining/rnd_loss", info.get("mean_loss",0.0), self._current_timesteps)
            if self.verbose > 0: print(f"\n[RND] Step {self._current_timesteps}: Loss = {info.get('mean_loss',0.0):.6f}")
        except Exception as e: print(f"\nWarning: RND update failed: {e}")

    def _save_checkpoint(self, final=False):
        if not self.save_dir: return
        suffix = "final" if final else str(self._current_timesteps)
        self.model.save(os.path.join(self.save_dir, f"refined_agent_{suffix}.zip"))
        if self.rnd_module: self.rnd_module.save(os.path.join(self.save_dir, f"rnd_{suffix}.pt"))
        if len(self.state_buffer) > 0: np.save(os.path.join(self.save_dir, f"state_buffer_{suffix}.npy"), np.stack(self.state_buffer[-10000:]))
        if self.logger: self.logger.save(os.path.join(self.save_dir, f"metrics_{suffix}.json"))
        if self.verbose > 0: print(f"\n[Save] Checkpoint saved at step {self._current_timesteps}")


def refine_agent(
    env_id, agent_path, critical_states_path, config, output_dir, seed=0,
    total_timesteps=None, p=None, lambda_rnd=None, use_rnd=True, use_mixed_init=True,
    rnd_embedding_dim=128, rnd_hidden_sizes=None, rnd_learning_rate=1e-4,
    ppo_learning_rate=None, n_steps=2048, batch_size=64, n_epochs=10,
    gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, vf_coef=0.5,
    max_grad_norm=0.5, eval_freq=10000, n_eval_episodes=10, rnd_update_freq=1000,
    rnd_batch_size=64, rnd_n_epochs=1, state_buffer_size=100000, device="auto",
    verbose=1, save_freq=100000, **env_kwargs,
):
    """Run the RICE refining process. Returns (refined_model, logger, save_path)."""
    refining_config = config.get("refining", {})
    rnd_config = config.get("rnd", {})
    agent_config = config.get("agent", {})
    if total_timesteps is None: total_timesteps = refining_config.get("total_timesteps", 1_000_000)
    if p is None: p = refining_config.get("p", 0.5)
    if lambda_rnd is None: lambda_rnd = refining_config.get("lambda", 0.01)
    if rnd_hidden_sizes is None: rnd_hidden_sizes = rnd_config.get("hidden_sizes", [64, 64])
    if ppo_learning_rate is None: ppo_learning_rate = agent_config.get("learning_rate", 3e-4)

    set_seed(seed)
    device_obj = get_device(device)
    output_dir = ensure_dir(output_dir)
    save_dir = ensure_dir(os.path.join(output_dir, "checkpoints"))
    log_dir = ensure_dir(os.path.join(output_dir, "logs"))
    logger = Logger(log_dir)

    print(f"\n{'='*60}\nRICE Refining Process\n{'='*60}")
    print(f"Env: {env_id} | p: {p} | lambda: {lambda_rnd} | RND: {use_rnd} | Mixed init: {use_mixed_init}")
    print(f"Total timesteps: {total_timesteps}\n{'='*60}\n")

    print("Loading pre-trained agent...")
    agent = PPO.load(agent_path, device=device_obj)
    print(f"  Loaded from {agent_path}")

    print("Loading critical states buffer...")
    critical_buffer = CriticalStateBuffer()
    critical_buffer.load(critical_states_path)
    print(f"  Loaded {len(critical_buffer)} critical states")

    eval_env = make_vec_env(env_id, n_envs=1, seed=seed + 1000, **env_kwargs)

    rnd_module = None
    if use_rnd:
        print("Creating RND module...")
        temp_env = make_env(env_id, seed=seed, **env_kwargs)
        obs_dim = temp_env.observation_space.shape[0]
        temp_env.close()
        rnd_module = RNDModule(input_dim=obs_dim, embedding_dim=rnd_embedding_dim,
                               hidden_sizes=rnd_hidden_sizes, learning_rate=rnd_learning_rate,
                               device=device_obj, normalize_obs=True)
        print(f"  RND: input_dim={obs_dim}, embedding_dim={rnd_embedding_dim}")

    state_buffer = []
    sampler = MixedInitialStateSampler(critical_buffer, p=p if use_mixed_init else 0.0, seed=seed)

    def make_refining_env():
        env = make_env(env_id, seed=seed, **env_kwargs)
        env = RefiningEnvWrapper(env, rnd_module=rnd_module, lambda_rnd=lambda_rnd, normalize_obs=True)
        env = Monitor(env)
        return env

    vec_env = DummyVecEnv([make_refining_env])
    policy_kwargs = agent.policy_kwargs if hasattr(agent, 'policy_kwargs') else {}

    print("Setting up refined agent...")
    refined_agent = PPO(
        policy=agent.policy_class, env=vec_env, learning_rate=ppo_learning_rate,
        n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, gamma=gamma,
        gae_lambda=gae_lambda, clip_range=clip_range, clip_range_vf=None,
        normalize_advantage=True, ent_coef=ent_coef, vf_coef=vf_coef,
        max_grad_norm=max_grad_norm, use_sde=False, sde_sample_freq=-1,
        target_kl=None, tensorboard_log=None, policy_kwargs=policy_kwargs,
        verbose=verbose, seed=seed, device=device_obj,
    )
    refined_agent.policy.load_state_dict(agent.policy.state_dict())
    refined_agent.policy.optimizer.load_state_dict(agent.policy.optimizer.state_dict())
    print("  Initialized with pre-trained weights")

    callbacks = [
        StateCollectionCallback(state_buffer, max_buffer_size=state_buffer_size),
        RefiningCallback(logger, eval_env, eval_freq, n_eval_episodes, rnd_module,
                         rnd_update_freq, rnd_batch_size, rnd_n_epochs, state_buffer, verbose),
    ]

    trainer = MixedInitTrainer(
        model=refined_agent, env=vec_env, sampler=sampler, rnd_module=rnd_module,
        lambda_rnd=lambda_rnd, total_timesteps=total_timesteps, n_steps=n_steps,
        callbacks=callbacks, state_buffer=state_buffer, eval_env=eval_env,
        eval_freq=eval_freq, n_eval_episodes=n_eval_episodes,
        rnd_update_freq=rnd_update_freq, rnd_batch_size=rnd_batch_size,
        rnd_n_epochs=rnd_n_epochs, save_freq=save_freq, save_dir=save_dir,
        logger=logger, verbose=verbose,
    )

    train_info = trainer.train()

    # Final save
    final_model_path = os.path.join(output_dir, "refined_agent_final.zip")
    refined_agent.save(final_model_path)
    logger.save(os.path.join(log_dir, "final_metrics.json"))
    if rnd_module: rnd_module.save(os.path.join(output_dir, "rnd_final.pt"))

    with open(os.path.join(output_dir, "train_info.json"), "w") as f:
        json.dump(train_info, f, indent=2, default=str)

    print(f"\nRefining complete! Model saved to {final_model_path}")
    print(f"Critical init count: {train_info['critical_init_count']}")
    print(f"Default init count: {train_info['default_init_count']}")
    print(f"Sampler stats: {train_info['sampler_stats']}")

    return refined_agent, logger, final_model_path


def main():
    """CLI entry point for refining."""
    parser = argparse.ArgumentParser(description="RICE Refining Process")
    parser.add_argument("--env-id", type=str, required=True, help="Gymnasium environment ID")
    parser.add_argument("--agent-path", type=str, required=True, help="Path to pre-trained agent")
    parser.add_argument("--critical-states-path", type=str, required=True, help="Path to critical states buffer")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--output-dir", type=str, default="./refining_output", help="Output directory")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Total timesteps")
    parser.add_argument("--p", type=float, default=None, help="Mixed init probability")
    parser.add_argument("--lambda-rnd", type=float, default=None, help="RND bonus weight")
    parser.add_argument("--no-rnd", action="store_true", help="Disable RND")
    parser.add_argument("--no-mixed-init", action="store_true", help="Disable mixed init")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity")
    parser.add_argument("--env-name", type=str, default=None, help="Environment name for config override")
    args = parser.parse_args()

    config = load_config(args.env_name) if args.env_name else load_config()
    if args.config:
        import yaml
        with open(args.config, "r") as f:
            config.update(yaml.safe_load(f))

    refine_agent(
        env_id=args.env_id,
        agent_path=args.agent_path,
        critical_states_path=args.critical_states_path,
        config=config,
        output_dir=args.output_dir,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        p=args.p,
        lambda_rnd=args.lambda_rnd,
        use_rnd=not args.no_rnd,
        use_mixed_init=not args.no_mixed_init,
        device=args.device,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()