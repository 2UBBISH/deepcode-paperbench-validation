"""Train target agents used by RICE experiments."""
import argparse
import os

from stable_baselines3 import PPO, SAC

from rice.env_utils import make_env
from rice.utils import set_seed


def train_target_agent(
    env_id: str,
    algorithm: str = "PPO",
    total_timesteps: int = 1_000_000,
    save_path: str = "models/target.pt",
    seed: int = 0,
    normalize_obs: bool = False,
    sparse: bool = False,
    **algo_kwargs,
):
    """Train a target RL agent and save the policy."""
    set_seed(seed)
    env = make_env(env_id, seed=seed, sparse=sparse, normalize_obs=normalize_obs)
    if algorithm.upper() == "PPO":
        model = PPO("MlpPolicy", env, verbose=1, seed=seed, **algo_kwargs)
    elif algorithm.upper() == "SAC":
        model = SAC("MlpPolicy", env, verbose=1, seed=seed, **algo_kwargs)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    model.learn(total_timesteps=total_timesteps)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Saved target agent to {save_path}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, required=True)
    parser.add_argument("--algorithm", type=str, default="PPO")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--save-path", type=str, default="models/target_ppo.pt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--normalize-obs", action="store_true")
    parser.add_argument("--sparse", action="store_true")
    args = parser.parse_args()
    train_target_agent(
        env_id=args.env_id,
        algorithm=args.algorithm,
        total_timesteps=args.timesteps,
        save_path=args.save_path,
        seed=args.seed,
        normalize_obs=args.normalize_obs,
        sparse=args.sparse,
    )


if __name__ == "__main__":
    main()
