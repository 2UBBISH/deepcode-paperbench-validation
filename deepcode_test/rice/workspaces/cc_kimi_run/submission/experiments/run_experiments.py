"""Unified experiment runner for RICE reproduction."""
import argparse
import json
import os
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from rice.baselines import jsrl_finetune, ppo_finetune, statemask_r_finetune
from rice.env_utils import make_env
from rice.explanations import (
    ExplanationMethod,
    MaskExplanation,
    RandomExplanation,
    StateMaskExplanation,
)
from rice.fidelity import compute_fidelity_score
from rice.mask_network import MaskNetworkTrainer
from rice.refining import refine_rice
from rice.utils import set_seed


HYPERPARAMETERS: Dict[str, Dict[str, float]] = {
    "Hopper-v3": {"p": 0.25, "lambda": 0.001, "alpha": 0.0001},
    "Walker2d-v3": {"p": 0.25, "lambda": 0.01, "alpha": 0.0001},
    "Reacher-v2": {"p": 0.50, "lambda": 0.001, "alpha": 0.0001},
    "HalfCheetah-v3": {"p": 0.50, "lambda": 0.01, "alpha": 0.0001},
    "SparseHopper-v3": {"p": 0.25, "lambda": 0.001, "alpha": 0.0001},
    "SparseHalfCheetah-v3": {"p": 0.50, "lambda": 0.01, "alpha": 0.0001},
}


def load_or_train_policy(
    env_id: str,
    model_path: Optional[str],
    total_timesteps: int = 1_000_000,
    seed: int = 0,
    sparse: bool = False,
    normalize_obs: bool = False,
) -> PPO:
    """Load a pre-trained policy or train a new one."""
    if model_path is not None and os.path.exists(model_path):
        env = make_env(env_id, seed=seed, sparse=sparse, normalize_obs=normalize_obs)
        return PPO.load(model_path, env=env)
    env = make_env(env_id, seed=seed, sparse=sparse, normalize_obs=normalize_obs)
    model = PPO("MlpPolicy", env, verbose=1, seed=seed)
    model.learn(total_timesteps=total_timesteps)
    if model_path is not None:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
    return model


def train_mask_network(
    env_id: str,
    policy: PPO,
    alpha: float,
    total_timesteps: int = 300_000,
    seed: int = 0,
    sparse: bool = False,
    normalize_obs: bool = False,
    save_path: Optional[str] = None,
) -> MaskNetworkTrainer:
    """Train the RICE mask network for the given policy."""
    env = make_env(env_id, seed=seed, sparse=sparse, normalize_obs=normalize_obs)
    obs_dim = int(np.prod(env.observation_space.shape))
    trainer = MaskNetworkTrainer(
        env=env,
        target_policy=policy,
        obs_dim=obs_dim,
        alpha=alpha,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        hidden_sizes=(64, 64),
    )
    trainer.train(total_timesteps=total_timesteps, steps_per_iter=2048)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        trainer.save(save_path)
    return trainer


def run_experiment_i(
    env_id: str,
    policy: PPO,
    mask_trainer: MaskNetworkTrainer,
    output_dir: str,
    d_max: float = 1000.0,
    k_values: tuple = (0.1, 0.2, 0.3, 0.4),
    seed: int = 0,
) -> Dict[str, Any]:
    """Experiment I: compare fidelity scores of explanation methods."""
    env = make_env(env_id, seed=seed)
    ours = MaskExplanation(mask_trainer.mask_net)
    random_exp = RandomExplanation(seed=seed)
    statemask = StateMaskExplanation(mask_trainer.mask_net)

    results: Dict[str, Any] = {}
    for name, expl in [("Ours", ours), ("Random", random_exp), ("StateMask", statemask)]:
        results[name] = {}
        for k in k_values:
            score = compute_fidelity_score(
                env=env,
                policy=policy,
                explanation=expl,
                d_max=d_max,
                k=k,
                n_trajectories=100,  # reduced for speed; paper uses 500
                max_steps=1000,
            )
            results[name][f"k={k}"] = score
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "experiment_i.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def run_experiment_ii(
    env_id: str,
    policy: PPO,
    mask_trainer: MaskNetworkTrainer,
    output_dir: str,
    refine_timesteps: int = 500_000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Experiment II: compare refining methods."""
    set_seed(seed)
    env = make_env(env_id, seed=seed)
    hparams = HYPERPARAMETERS.get(env_id, {"p": 0.25, "lambda": 0.01, "alpha": 0.0001})

    results: Dict[str, Any] = {}

    # PPO fine-tuning.
    ppo_refined = ppo_finetune(env, clone_policy(policy, env), total_timesteps=refine_timesteps)
    results["PPO"] = evaluate_policy(env, ppo_refined, seed=seed)

    # StateMask-R.
    ours_exp = MaskExplanation(mask_trainer.mask_net)
    sm_refined = statemask_r_finetune(
        env, clone_policy(policy, env), explanation=ours_exp, total_timesteps=refine_timesteps
    )
    results["StateMask-R"] = evaluate_policy(env, sm_refined, seed=seed)

    # JSRL.
    jsrl_refined = jsrl_finetune(env, clone_policy(policy, env), total_timesteps=refine_timesteps)
    results["JSRL"] = evaluate_policy(env, jsrl_refined, seed=seed)

    # RICE (Ours).
    rice_refined = refine_rice(
        env,
        clone_policy(policy, env),
        mask_trainer.mask_net,
        total_timesteps=refine_timesteps,
        p=hparams["p"],
        lambda_rnd=hparams["lambda"],
        alpha=hparams["alpha"],
    )
    results["Ours"] = evaluate_policy(env, rice_refined, seed=seed)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "experiment_ii.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def run_experiment_iii(
    env_id: str,
    policy: PPO,
    mask_trainer: MaskNetworkTrainer,
    output_dir: str,
    refine_timesteps: int = 500_000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Experiment III: refining with different explanation methods."""
    set_seed(seed)
    env = make_env(env_id, seed=seed)
    hparams = HYPERPARAMETERS.get(env_id, {"p": 0.25, "lambda": 0.01, "alpha": 0.0001})

    explanations = {
        "Random": RandomExplanation(seed=seed),
        "StateMask": StateMaskExplanation(mask_trainer.mask_net),
        "Ours": MaskExplanation(mask_trainer.mask_net),
    }

    results: Dict[str, Any] = {}
    for name, expl in explanations.items():
        refined = refine_rice(
            env,
            clone_policy(policy, env),
            mask_trainer.mask_net,
            total_timesteps=refine_timesteps,
            p=hparams["p"],
            lambda_rnd=hparams["lambda"],
            alpha=hparams["alpha"],
        )
        results[name] = evaluate_policy(env, refined, seed=seed)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "experiment_iii.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def run_experiment_v(
    env_id: str,
    policy: PPO,
    mask_trainer: MaskNetworkTrainer,
    output_dir: str,
    refine_timesteps: int = 500_000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Experiment V: hyperparameter sensitivity."""
    set_seed(seed)
    env = make_env(env_id, seed=seed)
    hparams = HYPERPARAMETERS.get(env_id, {"p": 0.25, "lambda": 0.01, "alpha": 0.0001})

    p_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    lambda_values = [0.0, 0.1, 0.01, 0.001]
    alpha_values = [0.01, 0.001, 0.0001]

    results: Dict[str, Any] = {"p": {}, "lambda": {}, "alpha": {}}

    for p in p_values:
        refined = refine_rice(
            env,
            clone_policy(policy, env),
            mask_trainer.mask_net,
            total_timesteps=refine_timesteps,
            p=p,
            lambda_rnd=hparams["lambda"],
            alpha=hparams["alpha"],
        )
        results["p"][str(p)] = evaluate_policy(env, refined, seed=seed)

    for lam in lambda_values:
        refined = refine_rice(
            env,
            clone_policy(policy, env),
            mask_trainer.mask_net,
            total_timesteps=refine_timesteps,
            p=hparams["p"],
            lambda_rnd=lam,
            alpha=hparams["alpha"],
        )
        results["lambda"][str(lam)] = evaluate_policy(env, refined, seed=seed)

    for alpha in alpha_values:
        # For alpha sensitivity, re-train mask network with each alpha.
        alpha_trainer = train_mask_network(
            env_id, policy, alpha=alpha, total_timesteps=300_000, seed=seed
        )
        refined = refine_rice(
            env,
            clone_policy(policy, env),
            alpha_trainer.mask_net,
            total_timesteps=refine_timesteps,
            p=hparams["p"],
            lambda_rnd=hparams["lambda"],
            alpha=alpha,
        )
        results["alpha"][str(alpha)] = evaluate_policy(env, refined, seed=seed)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "experiment_v.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


def clone_policy(policy: PPO, env: gym.Env) -> PPO:
    """Create a copy of an SB3 PPO policy on a new environment."""
    new_policy = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=policy.learning_rate,
        n_steps=policy.n_steps,
        batch_size=policy.batch_size,
        n_epochs=policy.n_epochs,
        gamma=policy.gamma,
        gae_lambda=policy.gae_lambda,
        clip_range=policy.clip_range,
        ent_coef=policy.ent_coef,
        vf_coef=policy.vf_coef,
        max_grad_norm=policy.max_grad_norm,
        verbose=policy.verbose,
        seed=policy.seed,
        device=policy.device,
    )
    new_policy.set_parameters(policy.get_parameters(), exact_match=True)
    return new_policy


def evaluate_policy(
    env: gym.Env,
    policy: PPO,
    n_eval_episodes: int = 10,
    seed: int = 0,
) -> Dict[str, float]:
    """Evaluate a policy by running several episodes."""
    rewards = []
    for ep in range(n_eval_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        done = False
        steps = 0
        max_steps = 1000
        while not done and steps < max_steps:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1
        rewards.append(total_reward)
    return {
        "mean": float(np.mean(rewards)),
        "std": float(np.std(rewards)),
        "rewards": [float(r) for r in rewards],
    }


def main():
    parser = argparse.ArgumentParser(description="Run RICE experiments")
    parser.add_argument("--env-id", type=str, default="Hopper-v3")
    parser.add_argument("--experiment", type=str, default="all", choices=["i", "ii", "iii", "v", "all"])
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--target-timesteps", type=int, default=1_000_000)
    parser.add_argument("--mask-timesteps", type=int, default=300_000)
    parser.add_argument("--refine-timesteps", type=int, default=500_000)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--normalize-obs", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    env_id = args.env_id
    output_dir = os.path.join(args.output_dir, env_id.replace("/", "_"))

    # Load or train target policy.
    policy = load_or_train_policy(
        env_id=env_id,
        model_path=args.model_path,
        total_timesteps=args.target_timesteps,
        seed=args.seed,
        sparse=args.sparse,
        normalize_obs=args.normalize_obs,
    )

    # Train mask network.
    hparams = HYPERPARAMETERS.get(env_id, {"p": 0.25, "lambda": 0.01, "alpha": 0.0001})
    mask_save_path = os.path.join(output_dir, "mask_net.pt")
    mask_trainer = train_mask_network(
        env_id=env_id,
        policy=policy,
        alpha=hparams["alpha"],
        total_timesteps=args.mask_timesteps,
        seed=args.seed,
        sparse=args.sparse,
        normalize_obs=args.normalize_obs,
        save_path=mask_save_path,
    )

    if args.experiment in ("i", "all"):
        run_experiment_i(env_id, policy, mask_trainer, output_dir, seed=args.seed)
    if args.experiment in ("ii", "all"):
        run_experiment_ii(env_id, policy, mask_trainer, output_dir, refine_timesteps=args.refine_timesteps, seed=args.seed)
    if args.experiment in ("iii", "all"):
        run_experiment_iii(env_id, policy, mask_trainer, output_dir, refine_timesteps=args.refine_timesteps, seed=args.seed)
    if args.experiment in ("v", "all"):
        run_experiment_v(env_id, policy, mask_trainer, output_dir, refine_timesteps=args.refine_timesteps, seed=args.seed)

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
