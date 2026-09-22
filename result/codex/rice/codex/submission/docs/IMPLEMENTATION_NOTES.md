# Where every line of the two algorithms lives

Line-by-line mapping between the pseudo-code of the paper and this repository,
so that the correctness of the implementation can be checked quickly.

## Algorithm 1 — training the mask network (`rice/explanation/mask_trainer.py`)

| Pseudo-code | Code |
|---|---|
| `Input: Target agent's policy pi` | `target_policy` argument of `train_mask_network` (any object with `act(obs, deterministic)`) |
| `Initialize the weights theta for the mask net` | `MaskActorCritic(obs_dim, hidden=spec.mask_hidden)` (built by `rice/experiments/explanation.train_explanation`) |
| `theta_old <- theta` | the sampling log-probabilities stored in the `RolloutBuffer` are those of the current policy at the moment of collection |
| `Set the initial state s0 ~ rho` | `obs, _ = env.reset()` at the beginning of every iteration |
| `Sample a_t ~ pi(a_t | s_t)` | `action_t = target_policy.act(obs, deterministic=False)` |
| `Sample a_t^m ~ pi_theta_old(a_t^m | s_t)` | `dist = mask_net.dist(obs)`, `mask_action = dist.sample()` |
| `a <- a_t ⊙ a_t^m` | `executed = env.random_action() if mask == 1 else action_t` |
| `(s_{t+1}, R'_t) <- env.step(a)`, record in `D` | `next_obs, env_reward, ... = env.step(executed)` then `buffer.add(obs, [mask], reward, value, log_prob, done, info)` |
| `update theta_old <- theta using D by PPO` | `buffer.compute_returns_and_advantage(0)` + `PPOUpdater.update(buffer)` (`rice/ppo_core.py`) |
| `R'(s,a) = R(s,a) + alpha * a^m` (Eq. 4) | `reward = config.reward_scale * env_reward + config.alpha * mask` |
| importance = `P(a^m = 0 | s)` | `MaskNet.importance(obs)` = `Categorical(logits).probs[..., 0]` |
| Theorem 3.3 (`J(theta) = max eta(pi_bar)`) | we maximise `eta(pi_bar)` with PPO, which the theorem shows is equivalent to StateMask's `min |eta(pi) - eta(pi_bar)|`; the `alpha` bonus prevents the trivial "never blind" solution |

Sampling budget: `config.rollout_length` steps per iteration × `config.iterations`
is exactly the number of samples of Table 4 (`spec.mask_samples` in
`rice/envs/registry.py`), and `MaskTrainingResult.wall_time` / `.samples` give
the efficiency columns.

## Algorithm 2 — refining the agent (`rice/refining/*.py`)

| Pseudo-code | Code |
|---|---|
| `Input: pre-trained pi, mask pi_tilde, rho, reset probability p` | `refine_rice(env, policy, mask_net, config)` with `config.p` |
| `RAND_NUM <- RAND(0,1)`, `if RAND_NUM < p` | `CriticalStateProvider.__call__` (`rice/refining/critical_state_provider.py`) |
| `Run pi to obtain a trajectory tau of length K` | `CriticalStateProvider.sample_critical_state` roll-out of `config.rollin_length` steps |
| `Identify the most critical state s_t in tau via the state mask` | `importance_scores(mask_net, states)` + `most_critical_state(scores)` |
| `Set the initial state s0 <- s_t` | `env.set_state(snapshot)` — the simulator snapshot taken before that step (`rice/envs/adapters.py`) |
| `else s0 ~ rho` | provider returns `None` → `env.reset()` |
| `Sample a_t ~ pi(a_t|s_t)` | `self.policy.act(policy_obs)` inside `RefiningTrainer.train` |
| `R_t^RND = ||f(s_{t+1}) - f_hat(s_{t+1})||^2 with normalization` | `RND.error` (running mean/std normalisation of the intrinsic reward) |
| `Add (s_t, s_{t+1}, a_t, R_t + lambda R_t^RND) to D` | `RNDRewardShaper.shape` returns `reward + lambda * bonus`, the buffer stores it |
| `Optimize pi_theta w.r.t PPO loss on D` | `PPOUpdater.update` |
| `Optimize f_hat_theta w.r.t. MSE loss on D using Adam` | `RND.update` (called from `RNDRewardShaper.after_iteration`) |
| `pi' <- pi_theta` | returned `RefineResult.policy` |

The same loop produces the three baselines, only the hooks differ
(`rice/refining/methods.py`):

| Baseline | initial states | exploration | roll-in |
|---|---|---|---|
| PPO fine-tuning | `rho` | none | none |
| StateMask-R | critical states (`p = 1`) | none | none |
| JSRL | `rho` | none | pre-trained guide for `N ~ U{0..N_max(progress)}` steps, data of the guided phase is discarded |
| RICE (ours) | mixed (`p`) | RND (`lambda`) | — |

## Fidelity score (`rice/fidelity.py`)

| Step of the metric | Code |
|---|---|
| importance of every step | `importance_fn(episode.obs_array())` |
| sliding window `l = K*L`, highest average importance | `most_critical_window(scores, l)` (`rice/explanation/critical_states.py`) |
| fast-forward to the beginning of the window | `rollout_episode(..., snapshot=episode.snapshots[start])` |
| `l` random actions, then the policy until the episode ends | `perturbed_fn` (random while `t < l`, then `policy.act`) |
| `d = abs(R' - R)` | `d = abs(perturbed.total_reward - original_reward)` |
| `score = log(d / d_max) - log(l / L)` | final two lines of `fidelity_scores_for_trajectory` |
| 500 trajectories, 3 seeds, `K in {10,20,30,40}%` | `FidelityConfig` defaults, `compute_fidelity(..., n_seeds=3)` |

`d_max` is a per-application constant in `rice/envs/registry.py`
(`EnvSpec.d_max`); it only shifts all fidelity scores of an application by the
same constant, so the comparison between explanation methods is unaffected.

## Experiment IV (`rice/imitation.py`, `rice/experiments/sac_refine.py`)

| Paper step | Code |
|---|---|
| pre-train a SAC agent | `train_sb3_agent(..., algo="sac")` (Stable-Baselines3, as in the paper) |
| GAIL imitation of the SAC agent | `train_gail`: discriminator `D(s,a)` + PPO generator with reward `-log(1 - D(s,a))` |
| refine the imitated policy | `run_experiment_iv` → `run_single_refine` for `ours`, `ppo`, `statemask_r`, `jsrl` |
| SAC fine-tuning baseline | `sac_finetune` (SAC `learn()` with a lowered learning rate) |

## Where the random actions come from

Algorithm 1 blinds the agent with `env.random_action()`, and the fidelity metric
randomises actions the same way.  `StatefulEnv.random_action` delegates to the
action space of the environment (`space.sample()`), and `set_global_seeds`
seeds that space, which makes both procedures reproducible.
