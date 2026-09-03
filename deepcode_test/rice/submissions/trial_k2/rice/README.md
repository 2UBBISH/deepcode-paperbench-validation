# RICE: Refining with Explanation for Reinforcement Learning Agents

This repository contains a reproduction implementation of **RICE**, a post-hoc policy-improvement framework for RL agents. RICE (1) trains a lightweight mask network to identify critical decision steps of a pre-trained agent, (2) refines the agent by restarting episodes from a mixed initial-state distribution (default resets + sampled critical states), and (3) adds a Random Network Distillation (RND) exploration bonus during refinement.

## Repository Structure

```
rice/
├── configs/                  # Task-specific hyper-parameter configs
│   ├── mujoco.yaml
│   ├── sparse_mujoco.yaml
│   ├── selfish_mining.yaml
│   ├── cage.yaml
│   ├── metadrive.yaml
│   └── malware.yaml
├── rice/
│   ├── agents/               # Target-policy wrappers and generic PPO trainer
│   ├── masknet/              # Mask network, masked env, mask trainer
│   ├── refine/               # Critical-state buffer, mixed reset, RND, refine trainer
│   ├── explain/              # Alternative explanation baselines (Random, IG, AIRS stub)
│   └── envs/                 # Task environment wrappers
├── scripts/                  # Executable entry points
│   ├── train_target.py       # Train the base target policy
│   ├── train_mask.py         # Train the MaskNet explanation module
│   ├── refine.py             # Run RICE refinement
│   ├── evaluate.py           # Evaluate a policy and compute fidelity
│   └── run_baselines.py      # Run comparison baselines
├── tests/                    # Unit tests
│   ├── test_mask.py
│   └── test_refine.py
├── requirements.txt
└── README.md
```

## Installation

### Requirements

- Python >= 3.8, < 3.11 (recommended)
- PyTorch >= 1.12
- Stable-Baselines3 >= 1.6
- Gymnasium >= 0.26 or Gym >= 0.21
- MuJoCo 2.1.0+ (modern `mujoco` bindings or legacy `mujoco-py`)
- Optional task-specific packages:
  - MetaDrive >= 0.3
  - CybORG / CAGE Challenge 2
  - `malware_rl` / MalConv gym
  - DI-drive (MetaDrive PPO)
  - Tianshou >= 0.4

### Quick Install

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install core dependencies
pip install -r requirements.txt
```

Some optional dependencies (DI-drive, CybORG, `malware_rl`, `mujoco-py`) are commented out in `requirements.txt` because they require external git repositories or system build tools. Install them manually if you need the corresponding tasks.

## Quick Start

The RICE pipeline consists of three stages:

1. **Train a target policy** for the task.
2. **Train a MaskNet** to identify critical states.
3. **Refine** the policy using mixed critical-state resets + RND.

### 1. Train Target Policy

```bash
python scripts/train_target.py \
  --task mujoco \
  --env-id Hopper-v3 \
  --total-timesteps 1000000 \
  --output-dir checkpoints/hopper
```

### 2. Train Mask Network

```bash
python scripts/train_mask.py \
  --target-policy checkpoints/hopper/policy.zip \
  --task mujoco \
  --env-id Hopper-v3 \
  --total-timesteps 500000 \
  --alpha 0.0001 \
  --output-dir checkpoints/hopper_mask
```

### 3. Refine Policy

```bash
python scripts/refine.py \
  --target-policy checkpoints/hopper/policy.zip \
  --critical-buffer checkpoints/hopper_mask/critical_buffer.npz \
  --task mujoco \
  --env-id Hopper-v3 \
  --total-timesteps 1000000 \
  --p 0.5 \
  --lambda-rnd 0.01 \
  --output-dir checkpoints/hopper_refined
```

### Evaluate

```bash
python scripts/evaluate.py \
  --policy checkpoints/hopper_refined/policy.pt \
  --task mujoco \
  --env-id Hopper-v3 \
  --n-episodes 500
```

## Scripts

### `scripts/train_target.py`

Trains the base policy for a task. Supports Stable-Baselines3 PPO (default for MuJoCo, sparse MuJoCo, CAGE, selfish mining) and a custom PyTorch PPO trainer (default for malware, MetaDrive, selfish mining when `--no-sb3`).

Key arguments:
- `--task`: `mujoco`, `sparse_mujoco`, `selfish_mining`, `cage`, `metadrive`, `malware`
- `--env-id`: Gym environment ID (e.g., `Hopper-v3`)
- `--total-timesteps`: training budget
- `--normalize-obs`: enable observation normalization (auto-enabled for Walker2d/HalfCheetah)
- `--output-dir`: checkpoint output directory

### `scripts/train_mask.py`

Trains the MaskNet explanation module on the `MaskedEnv` wrapper and extracts a critical-state buffer.

Key arguments:
- `--target-policy`: path to trained target policy
- `--alpha`: blinding bonus coefficient (default `1e-4`)
- `--top-p`: percentile for critical-state selection (default `0.25`)
- `--threshold`: alternative hard threshold for critical-state selection
- `--output-dir`: checkpoint output directory

Outputs:
- `mask.pt`: trained MaskNet
- `critical_buffer.npz`: critical-state buffer
- `metadata.txt`: run metadata

### `scripts/refine.py`

Runs the RICE refinement stage: mixed-reset environment + RND bonus + PPO training.

Key arguments:
- `--target-policy`: path to frozen target policy
- `--critical-buffer`: path to critical-state buffer `.npz`
- `--p`: probability of resetting from a critical state (default `0.5`)
- `--lambda-rnd`: RND bonus scale (default `0.01`)
- `--warm-start`: initialize refined policy from target policy
- `--output-dir`: checkpoint output directory

Outputs:
- `policy.pt`: refined policy
- `rnd.pt`: RND predictor checkpoint
- `metadata.txt`: run metadata

### `scripts/evaluate.py`

Evaluates a trained policy and optionally computes explanation fidelity.

Key arguments:
- `--policy`: path to target or refined policy
- `--mask`: path to trained MaskNet (optional, for fidelity)
- `--n-episodes`: number of evaluation episodes (default `500`)
- `--compute-fidelity`: enable fidelity computation

### `scripts/run_baselines.py`

Runs comparison baselines: PPO fine-tuning, Self-Imitation Learning (SIL), Jump-Start RL (JSRL), StateMask-R proxy, and alternative explanation methods (Random, Integrated Gradients, AIRS).

Example:

```bash
python scripts/run_baselines.py \
  --baseline sil \
  --target-policy checkpoints/hopper/policy.zip \
  --task mujoco \
  --env-id Hopper-v3 \
  --total-timesteps 1000000 \
  --output-dir checkpoints/hopper_sil
```

## Configuration Files

Task-specific hyper-parameters are stored in `configs/*.yaml`. You can load them in your own experiment runner or pass individual flags to the scripts. Key config files:

- `configs/mujoco.yaml`: dense MuJoCo (Hopper, Walker2d, Reacher, HalfCheetah)
- `configs/sparse_mujoco.yaml`: sparse-reward MuJoCo variants
- `configs/selfish_mining.yaml`: selfish mining blockchain task
- `configs/cage.yaml`: CAGE Challenge 2 / CybORG
- `configs/metadrive.yaml`: MetaDrive autonomous driving
- `configs/malware.yaml`: malware mutation / MalConv

## Reproducing Paper Results

### Experiment I – Explanation Fidelity

Train the proposed mask network and StateMask proxy on each task, then evaluate fidelity:

```bash
python scripts/train_mask.py --target-policy <policy> --task <task> --output-dir <mask_dir>
python scripts/evaluate.py --policy <policy> --mask <mask_dir>/mask.pt --task <task> --compute-fidelity
```

### Experiment II – Mask-Training Efficiency

Measure wall-clock time reported in `metadata.txt` after `train_mask.py` completes.

### Experiment III – Dense MuJoCo Refining

Run RICE, PPO fine-tuning, SIL, JSRL, and StateMask-R on Hopper, Walker2d, Reacher, HalfCheetah:

```bash
for method in rice ppo_finetune sil jsrl statemask_r; do
  python scripts/run_baselines.py --baseline $method \
    --target-policy checkpoints/<env>/policy.zip \
    --task mujoco --env-id <Env>-v3 \
    --output-dir results/<env>_$method
done
```

### Experiment IV – Sparse MuJoCo Refining

Use `--task sparse_mujoco` and the corresponding sparse environment IDs.

### Experiment V – Downstream Applications

- **Selfish mining**: use `--task selfish_mining`
- **CAGE Challenge 2**: use `--task cage`
- **MetaDrive**: use `--task metadrive`
- **Malware**: use `--task malware`

### Experiment F – Explanation Source Ablation

Run refinement with different explanation sources:

```bash
python scripts/run_baselines.py --baseline random_exp ...
python scripts/run_baselines.py --baseline ig_exp ...
python scripts/run_baselines.py --baseline airs_exp ...
python scripts/run_baselines.py --baseline statemask_r ...
```

### Experiment G – Hyper-Parameter Sensitivity

Vary `p`, `lambda-rnd`, and `alpha` via command-line flags and compare evaluation returns.

### Experiment H – Malware Reward-Design Debugging

Follow the malware case-study ablation in `configs/malware.yaml` and adjust the reward function in `rice/envs/malware_env.py` to make it Markovian and scale intermediate rewards by 3.

### Experiment I – Negative Control (MountainCarContinuous)

Train a weak PPO agent on `MountainCarContinuous-v0`, then apply RICE and RND. Expected outcome: no improvement.

## Running Tests

```bash
pytest tests/
```

The test suite uses lightweight synthetic environments and does not require MuJoCo, MetaDrive, CybORG, or `malware_rl`.

## Key Formulas

- **Perturbed policy mixture:**
  ```
  π̄(a|s) = ξ(s) π(a|s) + (1 - ξ(s)) π^r(a|s)
  ```
- **Mask-network reward:**
  ```
  r_mask = r_env + α (1 - ξ(s))
  ```
- **RND bonus:**
  ```
  b_RND(s) = || φ_target(s) - φ_pred(s) ||^2
  ```
- **Mixed reset:**
  ```
  s_0 ~ μ,  μ = p · Uniform(critical_states) + (1 - p) · ρ(s)
  ```
- **Refining reward:**
  ```
  r_refine = r_env + λ b_RND(s)
  ```

## Citation

If you use this code, please cite the original RICE paper:

```bibtex
@article{rice2023,
  title={RICE: Refining with Explanation for Reinforcement Learning Agents},
  journal={},
  year={2023}
}
```

## License

This reproduction is provided for academic research purposes. Please refer to the original paper and associated code licenses for task-specific dependencies.
