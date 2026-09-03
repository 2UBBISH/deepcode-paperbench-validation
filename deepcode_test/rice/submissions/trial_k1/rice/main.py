"""Single entry point for the RICE reproduction pipeline.

This script provides a unified command-line interface for:

* training target agents (`train-target`)
* training mask networks (`train-mask`)
* refining agents with RICE (`refine`)
* running the paper's experiments I-V (`exp-i`, `exp-ii`, `exp-iii`, `exp-iv`, `exp-v`)
* evaluating a saved policy (`eval`)

Examples
--------
Train a target PPO agent on Hopper-v3:

    python main.py train-target --domain mujoco --env-id Hopper-v3 \
        --save-dir results/targets/hopper --seed 0

Train a RICE mask network from a saved target agent:

    python main.py train-mask --domain mujoco --env-id Hopper-v3 \
        --target-path results/targets/hopper/agent.zip \
        --save-dir results/masks/hopper --seed 0

Refine the target agent with RICE:

    python main.py refine --domain mujoco --env-id Hopper-v3 \
        --target-path results/targets/hopper/agent.zip \
        --mask-path results/masks/hopper/mask_net.pt \
        --save-dir results/refined/hopper --seed 0

Run Experiment I (fidelity):

    python main.py exp-i --domain mujoco --env-id Hopper-v3 \
        --target-path results/targets/hopper/agent.zip \
        --mask-path results/masks/hopper/mask_net.pt

Run all experiments for a domain:

    python main.py run-all --domain mujoco --env-id Hopper-v3 \
        --target-path results/targets/hopper/agent.zip
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from rice.utils.config import get_domain_config, list_available_configs
from rice.utils.logger import make_logger


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by most subcommands."""
    parser.add_argument("--domain", type=str, required=True,
                        help="Domain name (e.g. mujoco, selfish_mining, cage, metadrive, malware).")
    parser.add_argument("--env-id", type=str, default=None,
                        help="Gym environment id (required for mujoco).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--device", type=str, default="auto",
                        help="PyTorch device (auto/cpu/cuda).")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory for checkpoints and logs.")
    parser.add_argument("--verbose", type=int, default=1,
                        help="Verbosity level (0=quiet, 1=info, 2=debug).")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional YAML config override path.")
    parser.add_argument("--use-tensorboard", action="store_true",
                        help="Enable TensorBoard logging.")
    parser.add_argument("--no-csv", action="store_true",
                        help="Disable CSV logging.")


def _add_target_args(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "train-target", help="Train a pre-trained target agent.")
    _add_common_args(parser)
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Override target training budget.")
    parser.add_argument("--sparse", action="store_true",
                        help="Use sparse reward variant (MuJoCo only).")
    parser.add_argument("--normalize-obs", action="store_true",
                        help="Force observation normalization.")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Override PPO learning rate.")


def _add_mask_args(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "train-mask", help="Train the RICE mask network.")
    _add_common_args(parser)
    parser.add_argument("--target-path", type=str, required=True,
                        help="Path to a saved target agent checkpoint.")
    parser.add_argument("--sample-budget", type=int, default=None,
                        help="Fixed sample budget for mask training (Table 4).")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Blinding coefficient override.")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Number of critical states to extract.")
    parser.add_argument("--sparse", action="store_true",
                        help="Use sparse reward variant (MuJoCo only).")


def _add_refine_args(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "refine", help="Refine a target agent with RICE.")
    _add_common_args(parser)
    parser.add_argument("--target-path", type=str, required=True,
                        help="Path to a saved target agent checkpoint.")
    parser.add_argument("--mask-path", type=str, default=None,
                        help="Path to a saved RICE mask network.")
    parser.add_argument("--critical-buffer", type=str, default=None,
                        help="Path to a pickled critical-state buffer.")
    parser.add_argument("--p", type=float, default=None,
                        help="Probability of starting from a critical state.")
    parser.add_argument("--lambda-coef", type=float, default=None,
                        help="RND bonus coefficient.")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Override refining budget.")
    parser.add_argument("--sparse", action="store_true",
                        help="Use sparse reward variant (MuJoCo only).")


def _add_eval_args(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "eval", help="Evaluate a saved target/refined agent.")
    _add_common_args(parser)
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to a saved agent checkpoint.")
    parser.add_argument("--n-eval-episodes", type=int, default=50,
                        help="Number of evaluation episodes.")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic policy evaluation.")
    parser.add_argument("--sparse", action="store_true",
                        help="Use sparse reward variant (MuJoCo only).")


def _add_experiment_args(subparsers: argparse._SubParsersAction) -> None:
    for name, help_text in [
        ("exp-i", "Experiment I: explanation fidelity (Figure 5)."),
        ("exp-ii", "Experiment II: mask-training efficiency (Table 4)."),
        ("exp-iii", "Experiment III: agent refining on dense MuJoCo (Tables 5/6)."),
        ("exp-iv", "Experiment IV: sparse-reward MuJoCo refining (Figures 10-13)."),
        ("exp-v", "Experiment V: case studies (malware, MetaDrive, MountainCar)."),
    ]:
        parser = subparsers.add_parser(name, help=help_text)
        _add_common_args(parser)
        parser.add_argument("--target-path", type=str, required=True,
                            help="Path to a saved target agent checkpoint.")
        parser.add_argument("--mask-path", type=str, default=None,
                            help="Path to a saved mask network (optional).")
        parser.add_argument("--train-mask", action="store_true",
                            help="Train a mask on demand if --mask-path is not provided.")
        parser.add_argument("--sample-budget", type=int, default=None,
                            help="Fixed sample budget for on-demand mask training.")
        parser.add_argument("--n-eval", type=int, default=50,
                            help="Number of evaluation episodes.")
        if name in ("exp-iii", "exp-iv"):
            parser.add_argument(
                "--methods", type=str, default=None,
                help="Comma-separated list of refining methods to run.")
        if name == "exp-iv":
            parser.add_argument("--p-grid", type=str, default="0,0.25,0.5,0.75,1",
                                help="Grid of critical-reset probabilities.")
            parser.add_argument("--lambda-grid", type=str,
                                default="0,0.001,0.01,0.1",
                                help="Grid of RND bonus coefficients.")
        if name == "exp-v":
            parser.add_argument("--study", type=str, default="malware",
                                choices=["malware", "metadrive", "mountaincar"],
                                help="Case-study sub-domain.")
            parser.add_argument("--reward-fix-scale", type=float, default=3.0,
                                help="Reward-design fix scale for malware study.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rice",
        description="RICE: Refining RL Agents via Critical Explanations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_target_args(subparsers)
    _add_mask_args(subparsers)
    _add_refine_args(subparsers)
    _add_eval_args(subparsers)
    _add_experiment_args(subparsers)

    run_all = subparsers.add_parser(
        "run-all", help="Run target training, mask training, refining, and experiments.")
    _add_common_args(run_all)
    run_all.add_argument("--target-path", type=str, default=None,
                         help="Optional pre-trained target path; otherwise train one.")
    run_all.add_argument("--mask-path", type=str, default=None,
                         help="Optional pre-trained mask path; otherwise train one.")
    run_all.add_argument("--skip-target", action="store_true",
                         help="Skip target-agent training.")
    run_all.add_argument("--skip-mask", action="store_true",
                         help="Skip mask training.")
    run_all.add_argument("--skip-refine", action="store_true",
                         help="Skip refining.")
    run_all.add_argument("--experiments", type=str, default="i,ii,iii,iv,v",
                         help="Comma-separated experiments to run.")
    run_all.add_argument("--sparse", action="store_true",
                         help="Use sparse reward variant (MuJoCo only).")

    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    return parser


def _make_save_dir(args: argparse.Namespace, default_name: str) -> Path:
    if args.save_dir is not None:
        return Path(args.save_dir)
    domain = args.domain
    env_id = getattr(args, "env_id", None) or domain
    return Path("results") / default_name / f"{domain}_{env_id}" / f"seed_{args.seed}"


def _make_logger(args: argparse.Namespace, experiment_name: str) -> Any:
    save_dir = _make_save_dir(args, experiment_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    return make_logger(
        log_dir=str(save_dir),
        experiment_name=experiment_name,
        use_tensorboard=args.use_tensorboard,
        use_csv=not args.no_csv,
        verbose=args.verbose,
    )


def _resolve_env_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """Build environment kwargs from CLI flags and optional config."""
    kwargs: Dict[str, Any] = {}
    if args.command in ("train-target", "train-mask", "refine", "eval", "run-all"):
        if hasattr(args, "sparse") and args.sparse:
            kwargs["sparse"] = True
        if hasattr(args, "normalize_obs") and args.normalize_obs:
            kwargs["normalize_obs"] = True
    if args.config is not None:
        from rice.utils.config import load_yaml_config
        cfg = load_yaml_config(args.config)
        if "env_kwargs" in cfg:
            kwargs.update(cfg["env_kwargs"])
    return kwargs


def _cmd_train_target(args: argparse.Namespace) -> int:
    from rice.training.train_target import train_target_agent

    env_kwargs = _resolve_env_kwargs(args)
    extra_kwargs: Dict[str, Any] = {"seed": args.seed, "device": args.device}
    if args.total_timesteps is not None:
        extra_kwargs["total_timesteps"] = args.total_timesteps
    if args.learning_rate is not None:
        extra_kwargs["learning_rate"] = args.learning_rate

    save_dir = _make_save_dir(args, "targets")
    logger = _make_logger(args, "train_target")
    logger.log_hyperparams({"command": args.command, "domain": args.domain,
                            "env_id": args.env_id, **extra_kwargs, **env_kwargs})

    agent = train_target_agent(
        domain=args.domain,
        save_dir=str(save_dir),
        **env_kwargs,
        **extra_kwargs,
    )
    mean, std = agent.evaluate(n_eval_episodes=10)
    logger.log({"target/mean_return": mean, "target/std_return": std}, step=0)
    print(f"Target agent saved to {save_dir}. Mean return: {mean:.2f} +/- {std:.2f}")
    logger.close()
    return 0


def _cmd_train_mask(args: argparse.Namespace) -> int:
    from rice.training.train_mask import train_mask

    env_kwargs = _resolve_env_kwargs(args)
    extra_kwargs: Dict[str, Any] = {
        "target_path": args.target_path,
        "seed": args.seed,
        "device": args.device,
    }
    if args.sample_budget is not None:
        extra_kwargs["total_timesteps"] = args.sample_budget
    if args.alpha is not None:
        extra_kwargs["alpha"] = args.alpha
    if args.top_k is not None:
        extra_kwargs["top_k"] = args.top_k

    save_dir = _make_save_dir(args, "masks")
    logger = _make_logger(args, "train_mask")
    logger.log_hyperparams({"command": args.command, "domain": args.domain,
                            "env_id": args.env_id, **extra_kwargs, **env_kwargs})

    result = train_mask(
        domain=args.domain,
        save_dir=str(save_dir),
        **env_kwargs,
        **extra_kwargs,
    )
    print(f"Mask network saved to {save_dir}. Critical states: "
          f"{len(result.get('critical_buffer', []))}")
    logger.close()
    return 0


def _cmd_refine(args: argparse.Namespace) -> int:
    from rice.agents.target_agent import TargetAgent
    from rice.training.refine_agent import RefineConfig, default_refine_config, refine_agent
    from rice.agents.mask_network import load_mask_network
    from rice.envs import make_mujoco_env, make_selfish_mining_env, make_cage_env, make_metadrive_env, make_malware_env

    env_kwargs = _resolve_env_kwargs(args)
    env_factories = {
        "mujoco": make_mujoco_env,
        "selfish_mining": make_selfish_mining_env,
        "cage": make_cage_env,
        "metadrive": make_metadrive_env,
        "malware": make_malware_env,
    }
    make_env = env_factories.get(args.domain)
    if make_env is None:
        raise ValueError(f"Unknown domain: {args.domain}")

    env = make_env(env_id=args.env_id, **env_kwargs) if args.domain == "mujoco" else make_env(**env_kwargs)
    target_agent = TargetAgent.load(args.target_path, env=env, device=args.device)

    mask_net = None
    if args.mask_path is not None:
        mask_net = load_mask_network(args.mask_path, env.observation_space, env.action_space, device=args.device)

    config = default_refine_config(domain=args.domain, env_id=args.env_id)
    if args.p is not None:
        config.p = args.p
    if args.lambda_coef is not None:
        config.lambda_coef = args.lambda_coef
    if args.total_timesteps is not None:
        config.total_timesteps = args.total_timesteps

    save_dir = _make_save_dir(args, "refined")
    logger = _make_logger(args, "refine")
    logger.log_hyperparams({"command": args.command, "domain": args.domain,
                            "env_id": args.env_id, **config.to_dict(), **env_kwargs})

    refined = refine_agent(
        env=env,
        target_agent=target_agent,
        mask_net=mask_net,
        config=config,
        save_dir=str(save_dir),
    )
    mean, std = refined.evaluate(n_eval_episodes=10)
    logger.log({"refined/mean_return": mean, "refined/std_return": std}, step=0)
    print(f"Refined agent saved to {save_dir}. Mean return: {mean:.2f} +/- {std:.2f}")
    logger.close()
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from rice.agents.target_agent import TargetAgent
    from rice.envs import make_mujoco_env, make_selfish_mining_env, make_cage_env, make_metadrive_env, make_malware_env

    env_kwargs = _resolve_env_kwargs(args)
    env_factories = {
        "mujoco": make_mujoco_env,
        "selfish_mining": make_selfish_mining_env,
        "cage": make_cage_env,
        "metadrive": make_metadrive_env,
        "malware": make_malware_env,
    }
    make_env = env_factories.get(args.domain)
    if make_env is None:
        raise ValueError(f"Unknown domain: {args.domain}")

    env = make_env(env_id=args.env_id, **env_kwargs) if args.domain == "mujoco" else make_env(**env_kwargs)
    agent = TargetAgent.load(args.model_path, env=env, device=args.device)
    mean, std = agent.evaluate(n_eval_episodes=args.n_eval_episodes,
                               deterministic=args.deterministic)
    print(f"Mean return: {mean:.2f} +/- {std:.2f}")
    return 0


def _cmd_exp_i(args: argparse.Namespace) -> int:
    from scripts.run_exp_i_fidelity import main as exp_i_main

    argv = [
        "--domain", args.domain,
        "--env-id", str(args.env_id or ""),
        "--target-path", args.target_path,
        "--seed", str(args.seed),
        "--device", args.device,
        "--n-eval", str(args.n_eval),
    ]
    if args.mask_path:
        argv += ["--mask-path", args.mask_path]
    if args.train_mask:
        argv += ["--train-mask"]
    if args.sample_budget is not None:
        argv += ["--sample-budget", str(args.sample_budget)]
    if args.save_dir:
        argv += ["--save-dir", args.save_dir]
    return exp_i_main(argv)


def _cmd_exp_ii(args: argparse.Namespace) -> int:
    from scripts.run_exp_ii_efficiency import main as exp_ii_main

    argv = [
        "--domain", args.domain,
        "--env-id", str(args.env_id or ""),
        "--target-path", args.target_path,
        "--seed", str(args.seed),
        "--device", args.device,
    ]
    if args.sample_budget is not None:
        argv += ["--sample-budget", str(args.sample_budget)]
    if args.save_dir:
        argv += ["--save-dir", args.save_dir]
    return exp_ii_main(argv)


def _cmd_exp_iii(args: argparse.Namespace) -> int:
    from scripts.run_exp_iii_refining import main as exp_iii_main

    argv = [
        "--domain", args.domain,
        "--env-id", str(args.env_id or ""),
        "--target-path", args.target_path,
        "--seed", str(args.seed),
        "--device", args.device,
        "--n-eval", str(args.n_eval),
    ]
    if args.mask_path:
        argv += ["--mask-path", args.mask_path]
    if args.train_mask:
        argv += ["--train-mask"]
    if args.methods:
        argv += ["--methods", args.methods]
    if args.save_dir:
        argv += ["--save-dir", args.save_dir]
    return exp_iii_main(argv)


def _cmd_exp_iv(args: argparse.Namespace) -> int:
    from scripts.run_exp_iv_sparse import main as exp_iv_main

    argv = [
        "--domain", args.domain,
        "--env-id", str(args.env_id or ""),
        "--target-path", args.target_path,
        "--seed", str(args.seed),
        "--device", args.device,
        "--n-eval", str(args.n_eval),
        "--p-grid", args.p_grid,
        "--lambda-grid", args.lambda_grid,
    ]
    if args.mask_path:
        argv += ["--mask-path", args.mask_path]
    if args.train_mask:
        argv += ["--train-mask"]
    if args.methods:
        argv += ["--methods", args.methods]
    if args.save_dir:
        argv += ["--save-dir", args.save_dir]
    return exp_iv_main(argv)


def _cmd_exp_v(args: argparse.Namespace) -> int:
    from scripts.run_exp_v_case_study import main as exp_v_main

    argv = [
        "--study", args.study,
        "--target-path", args.target_path,
        "--seed", str(args.seed),
        "--device", args.device,
        "--n-eval", str(args.n_eval),
        "--reward-fix-scale", str(args.reward_fix_scale),
    ]
    if args.mask_path:
        argv += ["--mask-path", args.mask_path]
    if args.train_mask:
        argv += ["--train-mask"]
    if args.save_dir:
        argv += ["--save-dir", args.save_dir]
    return exp_v_main(argv)


def _cmd_run_all(args: argparse.Namespace) -> int:
    """Run the full pipeline: target, mask, refine, and selected experiments."""
    experiments = [e.strip() for e in args.experiments.split(",")]
    target_path = args.target_path
    mask_path = args.mask_path

    if not args.skip_target and target_path is None:
        target_args = argparse.Namespace(**vars(args))
        target_args.command = "train-target"
        target_args.save_dir = str(_make_save_dir(args, "targets"))
        ret = _cmd_train_target(target_args)
        if ret != 0:
            return ret
        target_path = str(Path(target_args.save_dir) / "agent.zip")

    if not args.skip_mask and mask_path is None:
        mask_args = argparse.Namespace(**vars(args))
        mask_args.command = "train-mask"
        mask_args.target_path = target_path
        mask_args.save_dir = str(_make_save_dir(args, "masks"))
        ret = _cmd_train_mask(mask_args)
        if ret != 0:
            return ret
        mask_path = str(Path(mask_args.save_dir) / "mask_net.pt")

    if not args.skip_refine:
        refine_args = argparse.Namespace(**vars(args))
        refine_args.command = "refine"
        refine_args.target_path = target_path
        refine_args.mask_path = mask_path
        refine_args.save_dir = str(_make_save_dir(args, "refined"))
        ret = _cmd_refine(refine_args)
        if ret != 0:
            return ret

    for exp in experiments:
        exp_args = argparse.Namespace(**vars(args))
        exp_args.command = f"exp-{exp}"
        exp_args.target_path = target_path
        exp_args.mask_path = mask_path
        exp_args.save_dir = str(_make_save_dir(args, f"exp_{exp}"))
        handler = COMMAND_HANDLERS.get(exp_args.command)
        if handler is None:
            warnings.warn(f"Unknown experiment {exp}; skipping.")
            continue
        ret = handler(exp_args)
        if ret != 0:
            return ret

    return 0


COMMAND_HANDLERS = {
    "train-target": _cmd_train_target,
    "train-mask": _cmd_train_mask,
    "refine": _cmd_refine,
    "eval": _cmd_eval,
    "exp-i": _cmd_exp_i,
    "exp-ii": _cmd_exp_ii,
    "exp-iii": _cmd_exp_iii,
    "exp-iv": _cmd_exp_iv,
    "exp-v": _cmd_exp_v,
    "run-all": _cmd_run_all,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    handler = COMMAND_HANDLERS.get(args.command)
    if handler is None:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1

    try:
        return handler(args)
    except Exception as exc:
        print(f"Error executing command '{args.command}': {exc}", file=sys.stderr)
        if args.verbose > 1:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
