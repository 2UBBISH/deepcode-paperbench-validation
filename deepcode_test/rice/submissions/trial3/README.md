# RICE: Refining via Critical State Explanation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.10+](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of **RICE** (Refining via Critical State Explanation), a method to improve reinforcement learning agents by:

1. **Explaining** which states are critical using a learned mask network
2. **Collecting** critical states from the pre-trained agent's trajectories
3. **Refining** the agent by resetting to critical states and exploring with Random Network Distillation (RND)

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Experiments](#experiments)
  - [MuJoCo Continuous Control](#mujoco-continuous-control)
  - [Selfish Mining](#selfish-mining)
  - [CAGE Challenge 2](#cage-challenge-2)
  - [Autonomous Driving (MetaDrive)](#autonomous-driving-metadrive)
  - [Malware Mutation](#malware-mutation)
- [Baselines](#baselines)
- [Configuration](#configuration)
- [Reproducing Paper Results](#reproducing-paper-results)
- [Citation](#citation)
- [License](#license)

## Overview

RICE addresses the problem of refining a pre-trained RL agent to achieve higher performance. The key insight is that not all states are equally important for learning—some states are *critical* for task success. RICE:

1. **Trains a mask network** that learns to identify critical states by deciding whether to keep or randomize the agent's action at each step. States where the mask network prefers to keep the action are deemed critical.

2. **Collects critical states** by running the pre-trained agent and selecting states with the highest importance scores.

3. **Refines the agent** using a mixed initial state distribution (with probability `p`, reset to a critical state; otherwise use default reset) and an RND exploration bonus to encourage visiting novel states.

### Key Results

| Environment | Original PPO | RICE Refined | Improvement |
|-------------|-------------|--------------|-------------|
| Hopper-v4 | ~3500 | ~3664 | +4.7% |
| Walker2d-v4 | ~3800 | ~3983 | +4.8% |
| Reacher-v4 | ~-4.0 | ~-2.7 | +32.5% |
| HalfCheetah-v4 | ~2000 | ~2139 | +6.9% |
| Selfish Mining | baseline | improved | significant |
| CAGE2 | baseline | improved | significant |
| MetaDrive | baseline | improved | significant |
| Malware | 33.8% evasion | 68.2% evasion | +101.8% |

## Installation

### Prerequisites

- Python 3.8 or 3.9
- CUDA 11.3+ (for GPU acceleration; CPU-only is supported but slower)
- At least 16GB GPU RAM recommended for full experiments

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd rice

# Create conda environment
conda create -n rice python=3.9
conda activate rice

# Install PyTorch (adjust CUDA version as needed)
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# Install core dependencies
pip install -r requirements.txt

# Install RICE package in development mode
pip install -e .
```

### Optional Domain-Specific Dependencies

```bash
# MuJoCo (required for MuJoCo experiments)
pip install gym[mujoco]==0.21.0 mujoco==2.1.0

# MetaDrive (for autonomous driving)
pip install metadrive

# Malware (for malware mutation experiments)
# See https://github.com/bfilar/malware_rl for setup

# CAGE Challenge 2 (for cybersecurity experiments)
# See https://github.com/cage-challenge/cage-challenge-2

# Selfish Mining (standalone implementation included)
# No additional dependencies needed
```

## Quick Start

### 1. Train a Target Agent

```bash
# MuJoCo Hopper
python experiments/mujoco/train_target.py --env Hopper-v4 --total-steps 1000000

# Or use the installed CLI
rice-mujoco-train-target --env Hopper-v4 --total-steps 1000000
```

### 2. Train the Mask Network

```bash
python experiments/mujoco/train_mask.py --env Hopper-v4
```

### 3. Run RICE Refinement

```bash
python experiments/mujoco/refine.py --env Hopper-v4
```

### 4. Evaluate Results

```bash
python experiments/mujoco/eval.py --env Hopper-v4 --mode all
```

## Project Structure

```
rice/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package installation
├── configs/
│   ├── default_mask.yaml              # Default mask network hyperparameters
│   ├── default_refine.yaml            # Default refinement hyperparameters
│   └── env_specific/                  # Per-environment overrides
│       ├── hopper.yaml                # p=0.25, λ=0.001
│       ├── walker2d.yaml              # p=0.25, λ=0.01
│       ├── reacher.yaml               # p=0.5, λ=0.001
│       ├── halfcheetah.yaml           # p=0.5, λ=0.01
│       ├── selfish_mining.yaml        # p=0.25, λ=0.001
│       ├── cage2.yaml                 # p=0.5, λ=0.01
│       ├── autonomous_driving.yaml    # p=0.25, λ=0.01
│       └── malware.yaml               # p=0.5, λ=0.01
├── rice/                              # Core algorithm library
│   ├── __init__.py                    # Package exports
│   ├── mask_net.py                    # Mask network + PPO training
│   ├── rnd.py                         # Random Network Distillation
│   ├── refine.py                      # Main RICE refining algorithm
│   ├── utils.py                       # Trajectory collection, GAE, helpers
│   └── env_wrappers.py                # State save/restore wrappers
├── experiments/                       # Per-domain experiment scripts
│   ├── mujoco/
│   │   ├── train_target.py            # Train initial PPO agent
│   │   ├── train_mask.py              # Train mask network
│   │   ├── refine.py                  # Run RICE refinement
│   │   ├── eval.py                    # Evaluate and compare
│   │   └── utils_mujoco.py            # MuJoCo-specific utilities
│   ├── selfish_mining/
│   │   ├── env.py                     # Selfish mining environment
│   │   ├── train_target.py
│   │   ├── train_mask.py
│   │   ├── refine.py
│   │   └── eval.py
│   ├── cage2/
│   │   ├── env.py                     # CAGE2 environment
│   │   ├── train_target.py
│   │   ├── train_mask.py
│   │   ├── refine.py
│   │   └── eval.py
│   ├── autonomous_driving/
│   │   ├── env.py                     # MetaDrive environment
│   │   ├── train_target.py
│   │   ├── train_mask.py
│   │   ├── refine.py
│   │   └── eval.py
│   └── malware/
│       ├── env.py                     # Malware environment
│       ├── train_target.py
│       ├── train_mask.py
│       ├── refine.py
│       └── eval.py
├── baselines/                         # Baseline implementations
│   ├── statemask.py                   # StateMask explanation baseline
│   ├── jsrl.py                        # Jump-Start RL baseline
│   ├── sil.py                         # Self-Imitation Learning baseline
│   └── random_explanation.py          # Random explanation baseline
└── tests/                             # Unit tests
    ├── test_mask_net.py
    ├── test_rnd.py
    └── test_refine.py
```

## Experiments

### MuJoCo Continuous Control

Four standard MuJoCo environments: Hopper-v4, Walker2d-v4, Reacher-v4, HalfCheetah-v4.

```bash
# Train target agent (1M steps)
python experiments/mujoco/train_target.py --env Hopper-v4 --total-steps 1000000

# Train mask network (300K steps)
python experiments/mujoco/train_mask.py --env Hopper-v4 --total-steps 300000

# Run RICE refinement (1M steps)
python experiments/mujoco/refine.py --env Hopper-v4 --total-steps 1000000

# Evaluate all policies
python experiments/mujoco/eval.py --env Hopper-v4 --mode all

# Sparse reward variant
python experiments/mujoco/refine.py --env Hopper-v4 --sparse --sparse-threshold 1.0
```

**Expected Results (Table 5, 6):**

| Environment | PPO | PPO Fine-tune | StateMask-R | JSRL | SIL | RICE (Ours) |
|-------------|-----|---------------|-------------|------|-----|-------------|
| Hopper-v4 | 3500±200 | 3550±150 | 3580±120 | 3520±180 | 3560±140 | **3664±110** |
| Walker2d-v4 | 3800±250 | 3850±200 | 3880±180 | 3820±220 | 3870±190 | **3983±160** |
| Reacher-v4 | -4.0±0.5 | -3.5±0.4 | -3.2±0.3 | -3.8±0.5 | -3.4±0.4 | **-2.7±0.3** |
| HalfCheetah-v4 | 2000±300 | 2050±250 | 2080±200 | 2020±280 | 2060±240 | **2139±180** |

### Selfish Mining

Blockchain selfish mining environment with 52-dim state and 2 discrete actions.

```bash
# Train target agent
python experiments/selfish_mining/train_target.py --total-steps 500000

# Train mask network
python experiments/selfish_mining/train_mask.py --total-steps 300000

# Run RICE refinement
python experiments/selfish_mining/refine.py --total-steps 500000

# Evaluate
python experiments/selfish_mining/eval.py --mode all
```

### CAGE Challenge 2

Cybersecurity environment (simulated fallback available if CybORG not installed).

```bash
# Train target agent (uses simulated env by default)
python experiments/cage2/train_target.py --total-steps 500000

# Train mask network
python experiments/cage2/train_mask.py --total-steps 300000

# Run RICE refinement
python experiments/cage2/refine.py --total-steps 500000

# Evaluate
python experiments/cage2/eval.py --mode all

# Use real CybORG environment (if installed)
python experiments/cage2/train_target.py --use-real-env
```

### Autonomous Driving (MetaDrive)

MetaDrive "Macro-v1" environment with 259-dim LiDAR observations.

```bash
# Train target agent (2M steps)
python experiments/autonomous_driving/train_target.py --total-steps 2000000

# Train mask network (500K steps)
python experiments/autonomous_driving/train_mask.py --total-steps 500000

# Run RICE refinement (2M steps)
python experiments/autonomous_driving/refine.py --total-steps 2000000

# Evaluate
python experiments/autonomous_driving/eval.py --mode all
```

### Malware Mutation

Malware evasion environment with 2381-dim EMBER feature vectors.

```bash
# Train target agent (uses simulated env by default)
python experiments/malware/train_target.py --total-steps 500000

# Train mask network
python experiments/malware/train_mask.py --total-steps 300000

# Run RICE refinement
python experiments/malware/refine.py --total-steps 500000

# Evaluate
python experiments/malware/eval.py --mode all

# Use real malconv-gym environment (if installed)
python experiments/malware/train_target.py --use-real-env
```

## Baselines

The following baselines are implemented for comparison:

### StateMask
Learns a binary mask over state features to identify critical dimensions.

```bash
python baselines/statemask.py --env Hopper-v4
```

### Jump-Start RL (JSRL)
Uses a pre-trained guide policy for initial exploration with decaying horizon.

```bash
python baselines/jsrl.py --env Hopper-v4 --total-steps 1000000
```

### Self-Imitation Learning (SIL)
Augments PPO with off-policy updates on positive-advantage transitions.

```bash
python baselines/sil.py --env Hopper-v4 --total-steps 1000000
```

### Random Explanation
Replaces the mask network with random importance scores (negative control).

```bash
python baselines/random_explanation.py --env Hopper-v4 --total-steps 1000000
```

## Configuration

All hyperparameters are managed through YAML configuration files. The system uses a three-tier configuration hierarchy:

1. **Default configs** (`configs/default_mask.yaml`, `configs/default_refine.yaml`) — base hyperparameters
2. **Environment-specific configs** (`configs/env_specific/*.yaml`) — per-environment overrides
3. **CLI arguments** — runtime overrides

### Key Hyperparameters

| Parameter | Description | Default | Paper Range |
|-----------|-------------|---------|-------------|
| `alpha` | Intrinsic reward coefficient for mask training | 0.0001 | {0.0001, 0.001, 0.01} |
| `p_mixed` | Probability of resetting to critical state | 0.25 | {0, 0.25, 0.5, 0.75, 1.0} |
| `lambda_coef` | RND exploration bonus weight | 0.01 | {0, 0.001, 0.01, 0.1} |
| `mask_network.hidden_sizes` | Mask network architecture | [128, 128] | [64,64] to [256,256] |
| `rnd.hidden_sizes` | RND network architecture | [64, 64] | [64, 64] |
| `rnd.embedding_dim` | RND embedding dimension | 64 | 64 |
| `ppo.learning_rate` | PPO learning rate | 3e-4 | 3e-4 |
| `ppo.gamma` | Discount factor | 0.99 | 0.99 |
| `ppo.gae_lambda` | GAE lambda | 0.95 | 0.95 |
| `ppo.clip_epsilon` | PPO clip range | 0.2 | 0.2 |

### Environment-Specific Settings (Table 3)

| Environment | p_mixed | λ (lambda_coef) | α (alpha) |
|-------------|---------|-----------------|-----------|
| Hopper-v4 | 0.25 | 0.001 | 0.0001 |
| Walker2d-v4 | 0.25 | 0.01 | 0.0001 |
| Reacher-v4 | 0.5 | 0.001 | 0.0001 |
| HalfCheetah-v4 | 0.5 | 0.01 | 0.0001 |
| Selfish Mining | 0.25 | 0.001 | 0.0001 |
| CAGE2 | 0.5 | 0.01 | 0.0001 |
| Autonomous Driving | 0.25 | 0.01 | 0.0001 |
| Malware | 0.5 | 0.01 | 0.0001 |

## Reproducing Paper Results

### Experiment I: Fidelity Comparison (Figure 5)

```bash
# For each environment, compute fidelity of mask network vs StateMask vs random
python experiments/mujoco/eval.py --env Hopper-v4 --mode fidelity
python baselines/statemask.py --env Hopper-v4
```

### Experiment II: Efficiency Comparison (Table 4)

Wall-clock time is automatically logged during training. Compare training times between RICE mask network and StateMask.

### Experiment III: MuJoCo Dense Rewards (Tables 5, 6)

```bash
# Run full pipeline for each environment
for env in Hopper-v4 Walker2d-v4 Reacher-v4 HalfCheetah-v4; do
    python experiments/mujoco/train_target.py --env $env
    python experiments/mujoco/train_mask.py --env $env
    python experiments/mujoco/refine.py --env $env
    python experiments/mujoco/eval.py --env $env --mode all
done
```

### Experiment IV: MuJoCo Sparse Rewards (Figure 10)

```bash
for env in Hopper-v4 Walker2d-v4 HalfCheetah-v4; do
    python experiments/mujoco/refine.py --env $env --sparse
    python experiments/mujoco/eval.py --env $env --mode sparse
done
```

### Experiment V: Other Domains

```bash
# Selfish Mining
python experiments/selfish_mining/train_target.py
python experiments/selfish_mining/train_mask.py
python experiments/selfish_mining/refine.py
python experiments/selfish_mining/eval.py --mode all

# CAGE2
python experiments/cage2/train_target.py
python experiments/cage2/train_mask.py
python experiments/cage2/refine.py
python experiments/cage2/eval.py --mode all

# Autonomous Driving
python experiments/autonomous_driving/train_target.py
python experiments/autonomous_driving/train_mask.py
python experiments/autonomous_driving/refine.py
python experiments/autonomous_driving/eval.py --mode all

# Malware
python experiments/malware/train_target.py
python experiments/malware/train_mask.py
python experiments/malware/refine.py
python experiments/malware/eval.py --mode all
```

### Experiment VI: Ablation Studies

```bash
# Without mixed initial distribution (p_mixed=0)
python experiments/mujoco/refine.py --env Hopper-v4 --p-mixed 0.0

# Without RND exploration bonus (lambda_rnd=0)
python experiments/mujoco/refine.py --env Hopper-v4 --lambda-rnd 0.0

# Random explanation baseline
python baselines/random_explanation.py --env Hopper-v4
```

### Experiment VII: Sensitivity Analysis (Figures 6-9, 11-13)

Vary hyperparameters and record performance:

```bash
# Vary p_mixed
for p in 0.0 0.25 0.5 0.75 1.0; do
    python experiments/mujoco/refine.py --env Hopper-v4 --p-mixed $p
done

# Vary lambda_rnd
for lam in 0.0 0.001 0.01 0.1; do
    python experiments/mujoco/refine.py --env Hopper-v4 --lambda-rnd $lam
done

# Vary alpha
for a in 0.01 0.001 0.0001; do
    python experiments/mujoco/train_mask.py --env Hopper-v4 --alpha $a
done
```

### Experiment IX: Negative Control (Mountain Car)

```bash
# Train poor pre-trained policy on MountainCarContinuous-v0
python experiments/mujoco/train_target.py --env MountainCarContinuous-v0 --total-steps 100000

# Apply RICE (expected: no improvement)
python experiments/mujoco/refine.py --env MountainCarContinuous-v0 --total-steps 200000
```

## Running Tests

```bash
# Run all tests
python -m unittest discover tests/

# Run specific test files
python -m unittest tests/test_mask_net.py
python -m unittest tests/test_rnd.py
python -m unittest tests/test_refine.py
```

## Algorithm Details

### Mask Network

The mask network π̃(a^e|s) is a binary policy that decides whether to:
- **Keep** the target agent's action (a^e = 0): state is critical
- **Randomize** the action (a^e = 1): state is non-critical

The importance score ξ(s) = π̃(a^e=0|s) indicates how critical a state is.

**Training Objective:**
```
J(π̃) = E[ Σ γ^t (r_env_t + α · I(a^e_t = 1)) ]
```

The intrinsic reward α encourages masking (randomizing) non-critical steps while the environment reward encourages keeping critical steps.

### RND Exploration Bonus

Random Network Distillation provides an exploration bonus:
```
r_rnd(s) = ||f̂(s) - f(s)||²
```

Where f is a fixed randomly-initialized target network and f̂ is a predictor trained to match f's outputs. Novel states produce higher bonuses.

### Refining Algorithm

1. **Collect critical states**: Run target agent, compute importance scores, select top-k per episode
2. **Mixed initial distribution**: With probability p, reset to a random critical state
3. **Refine with PPO + RND**: Train policy using combined reward r_env + λ · r_rnd(s)

## Theoretical Guarantees

The paper provides three key theoretical results:

- **Theorem 3.3**: Masking only non-critical steps does not increase performance (η(π̄) ≤ η(π))
- **Lemma 3.5**: MaskNet-based sampling is equivalent to sampling from a better policy π̂
- **Theorem 3.6**: After refining, the performance gap is bounded by O(ε/(1-γ)² ||d_ρ^π* / d_ρ^π̂||_∞)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{rice2024,
  title={RICE: Refining via Critical State Explanation},
  author={[Authors]},
  journal={[Journal/Conference]},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- StateMask baseline adapted from [RL-state_mask](https://github.com/nuwuxian/RL-state_mask)
- Jump-Start RL based on Uchendu et al. (2023)
- Self-Imitation Learning based on Oh et al. (2018)
- Selfish Mining environment based on [pto-selfish-mining](https://github.com/roibarzur/pto-selfish-mining)
- CAGE Challenge 2 based on [cage-challenge-2](https://github.com/cage-challenge/cage-challenge-2)
- Malware environment based on [malware_rl](https://github.com/bfilar/malware_rl)