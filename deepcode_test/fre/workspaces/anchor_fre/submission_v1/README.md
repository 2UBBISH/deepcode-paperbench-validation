# Functional Reward Encodings (FRE) - Reproduction

This repository contains a reproduction of the paper "Unsupervised Zero-Shot Reinforcement Learning via Functional Reward Encodings" (ICML 2024).

## Paper Summary

The paper presents FRE (Functional Reward Encoding), a method for zero-shot reinforcement learning that:
1. Learns to encode arbitrary reward functions into a latent space using a transformer-based variational auto-encoder
2. Trains a multi-task policy conditioned on these encodings using offline RL (IQL)
3. Enables zero-shot adaptation to new tasks by encoding their reward functions from a small number of (state, reward) samples

## Implementation Overview

This reproduction implements the core contributions of the paper:

### 1. FRE Architecture (`fre/`)

#### Encoder (`fre/encoder.py`)
- **Permutation-invariant transformer** that encodes K (state, reward) pairs
- Reward discretization into 32 bins with learned embeddings (64-dim)
- State embeddings via linear projection (64-dim)
- Transformer with 4 attention heads and 4 layers
- Outputs mean and log std of Gaussian distribution over 128-dim latent z
- Implements variational information bottleneck objective

#### Decoder (`fre/decoder.py`)
- **MLP decoder** with 3 hidden layers (512 units each)
- Takes state and latent z as input
- Predicts reward for the state
- Trained jointly with encoder using MSE loss

#### Reward Functions (`fre/reward_functions.py`)
Implements the three types of random reward functions used as the prior distribution:

1. **Goal-Reaching Rewards**: Singleton rewards that return -1 until goal reached, 0 when reached
   - Goals sampled using hindsight relabeling (0.2 current, 0.5 future, 0.3 random)

2. **Random Linear Functions**: Inner product between random sparse vector and state
   - 90% sparsity to bias towards simpler functions
   - Excludes XY positions for AntMaze (as specified in paper)

3. **Random MLP Functions**: 2-layer MLPs with random initialization
   - Architecture: (state_dim, 32, 1)
   - Tanh activation between layers
   - Clipped to [-1, 1]

The `RewardFunctionSampler` samples from a mixture (default: 33% each type).

### 2. Offline RL with FRE (`fre/iql.py`)

Implements **Implicit Q-Learning (IQL)** conditioned on latent z:
- Q-networks: Q(s, a, z) - two networks for stability
- Value network: V(s, z)
- Policy network: π(a|s, z) - Gaussian policy

Training procedure follows Algorithm 1:
1. **Phase 1** (150k steps): Train encoder-decoder on random reward functions
2. **Phase 2** (850k steps): Freeze encoder, train IQL policy on sampled tasks

Key features:
- Expectile regression for value function (τ = 0.8)
- Advantage-weighted regression for policy (temperature = 3.0)
- Discount factor γ = 0.88
- All networks conditioned on latent z

### 3. Evaluation Environments (`environments/`)

#### AntMaze Tasks (`environments/antmaze_tasks.py`)
Implements all evaluation tasks from the paper:

1. **Goal-Reaching** (5 tasks): Fixed goal locations
   - goal-bottom (28, 0)
   - goal-left (0, 15)
   - goal-top (35, 24)
   - goal-center (12, 24)
   - goal-right (33, 16)

2. **Directional** (4 tasks): Velocity-based rewards
   - vel_left, vel_up, vel_down, vel_right
   - Reward = dot product of velocity with target direction

3. **Random Simplex** (5 tasks): Procedural noise-based rewards
   - Uses opensimplex noise to create height maps and velocity preferences
   - 5 different seeds for 5 evaluation tasks

4. **Path Tasks** (3 tasks): Hand-crafted corridor following
   - path-center: Follow central corridor
   - path-loop: Move in loop around grid
   - path-edges: Stay near edges

### 4. Baseline Methods (`baselines/`)

#### Goal-Conditioned IQL (`baselines/gc_iql.py`)
- IQL with observations and goals concatenated
- Hindsight relabeling: 0.5 future (geometric), 0.3 random, 0.2 current
- Reward: 0 if |s - g| < threshold, else -1
- Same network architecture as FRE for fair comparison

#### Goal-Conditioned BC (`baselines/gc_bc.py`)
- MLP with 3 hidden layers (512 units) and layer normalization
- Maximum likelihood estimation loss
- Only geometric sampling for hindsight relabeling (no random/current goals)
- Outputs Gaussian distribution over actions

### 5. Training and Evaluation Scripts

#### Training (`train_fre.py`)
Main training script implementing the two-phase procedure:
```bash
python train_fre.py --env antmaze-large-diverse-v2 \
                    --encoder_steps 150000 \
                    --policy_steps 850000 \
                    --batch_size 512 \
                    --K 32 \
                    --K_prime 8
```

#### Evaluation (`evaluate_fre.py`)
Zero-shot evaluation on downstream tasks:
```bash
python evaluate_fre.py --checkpoint checkpoints/fre_full_antmaze-large-diverse-v2.pt \
                       --env antmaze-large-diverse-v2 \
                       --num_eval_samples 32
```

## Key Implementation Details

Based on the paper and addendum:

### Architecture Specifications
- **State embedding**: 64 dimensions
- **Reward embedding**: 64 dimensions (concatenated → 128-dim input to transformer)
- **Latent z**: 128 dimensions
- **Encoder transformer**: 4 heads, 4 layers, MLP hidden dim = 256
- **Decoder MLP**: [512, 512, 512]
- **RL networks**: [512, 512, 512]
- **Reward discretization**: 32 bins, rescaled from [-1, 1] to [0, 1]

### Hyperparameters
- **Learning rate**: 1e-4 (Adam)
- **Batch size**: 512
- **Encoder samples (K)**: 32 state-reward pairs
- **Decoder samples (K')**: 8 state-reward pairs
- **β (KL weight)**: 0.01
- **Discount factor (γ)**: 0.88
- **Target update rate (τ)**: 0.001
- **IQL expectile**: 0.8
- **AWR temperature**: 3.0

### Training Procedure
1. Train encoder-decoder for 150k steps (1M for ExORL/Kitchen)
2. Freeze encoder weights
3. Train IQL policy for 850k steps (1M for ExORL/Kitchen)
4. At each policy training step:
   - Sample reward function from prior
   - Encode it using K random states → get z
   - Sample batch of transitions
   - Compute rewards using sampled function
   - Update IQL with (s, a, r, s', d, z)

### Evaluation
- Use only 32 (state, reward) samples to encode downstream tasks
- No fine-tuning or additional training
- Evaluate on 20 episodes per task, max 2000 steps per episode (AntMaze)

## Main Results (from Paper)

The paper reports that FRE achieves:
- **AntMaze-all**: 52.8 ± 18.2 (normalized score)
- **ExORL-all**: 51.5 ± 6.3
- **Kitchen**: 66 ± 3

Compared to baselines:
- Outperforms FB (Forward-Backward): 24 ± 12 overall
- Outperforms SF (Successor Features): 18 ± 5 overall
- Matches or outperforms GC-IQL on goal-reaching tasks
- More general than GC-IQL/GC-BC which only handle goal-reaching

## Dependencies

```
torch >= 1.12.0
numpy >= 1.21.0
gym >= 0.21.0
d4rl >= 1.1
mujoco-py >= 2.1.2.14
dm-control >= 1.0.0
```

For random simplex tasks (optional):
```
opensimplex
```

## Project Structure

```
submission/
├── fre/
│   ├── __init__.py
│   ├── encoder.py          # Transformer-based encoder
│   ├── decoder.py          # MLP decoder
│   ├── reward_functions.py # Random reward generators
│   └── iql.py              # IQL with FRE conditioning
├── environments/
│   ├── __init__.py
│   └── antmaze_tasks.py    # AntMaze evaluation tasks
├── baselines/
│   ├── __init__.py
│   ├── gc_iql.py           # Goal-conditioned IQL
│   └── gc_bc.py            # Goal-conditioned BC
├── train_fre.py            # Main training script
├── evaluate_fre.py         # Evaluation script
├── requirements.txt
└── README.md
```

## What Was Implemented

### Core Contributions (Fully Implemented)
✅ Functional Reward Encoder (transformer-based VAE)
✅ Functional Reward Decoder (MLP)
✅ Random reward function generators (goal, linear, MLP)
✅ IQL conditioned on FRE latent z
✅ Two-phase training procedure (encoder → policy)
✅ AntMaze evaluation tasks (all 6 task categories)
✅ GC-IQL baseline
✅ GC-BC baseline

### Partially Implemented
⚠️ ExORL tasks (structure defined but not fully tested)
⚠️ Kitchen tasks (structure defined but not fully tested)

### Not Implemented (Out of Scope or Lower Priority)
❌ FB (Forward-Backward) baseline - requires external codebase (facebookresearch/controllable_agent)
❌ SF (Successor Features) baseline - requires external codebase
❌ OPAL baseline - requires trajectory auto-encoding
❌ Online fine-tuning experiments
❌ Ablation studies from Section 5.3 (different reward mixtures)
❌ Domain-specific priors from Section 5.4 (FRE-hint)

## Notes on Reproduction

1. **Dataset**: Uses D4RL's `antmaze-large-diverse-v2` dataset as specified
2. **Evaluation**: Implements zero-shot evaluation with 32 encoding samples (vs 5120 for FB/SF in paper)
3. **Normalization**: Score normalization needs task-specific ranges (simplified in current implementation)
4. **Random Functions**: MLP and linear functions use random initialization - results may vary across runs
5. **Opensimplex**: Random simplex tasks require opensimplex library (fallback to numpy if not available)

## Key Differences from Paper

1. **Simplified score normalization**: Paper normalizes scores based on task-specific min/max performance
2. **No ExORL physics augmentation**: Paper appends physics info (velocity, height) to states for ExORL
3. **FB/SF baselines not included**: These require the facebookresearch/controllable_agent codebase
4. **Limited to AntMaze**: Full reproduction focuses on AntMaze as the primary benchmark

## References

Frans, K., Park, S., Abbeel, P., & Levine, S. (2024). Unsupervised Zero-Shot Reinforcement Learning via Functional Reward Encodings. In Proceedings of the 41st International Conference on Machine Learning (ICML 2024).

## Citation

```bibtex
@inproceedings{frans2024fre,
  title={Unsupervised Zero-Shot Reinforcement Learning via Functional Reward Encodings},
  author={Frans, Kevin and Park, Seohong and Abbeel, Pieter and Levine, Sergey},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```
