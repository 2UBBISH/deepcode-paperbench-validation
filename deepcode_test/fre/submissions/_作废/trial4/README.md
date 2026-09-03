# Functional Reward Encodings (FRE)

**Zero-Shot Offline Reinforcement Learning via Functional Reward Encodings**

This repository implements the Functional Reward Encodings (FRE) method described in the paper *"Functional Reward Encodings for Zero-Shot Offline Reinforcement Learning"*. FRE enables zero-shot generalization of offline RL policies to novel downstream tasks by learning a variational auto-encoder with a permutation-invariant transformer that encodes arbitrary reward functions from a small set of (state, reward) pairs.

---

## Overview

FRE addresses the problem of zero-shot offline RL: given an offline dataset of transitions collected by an unsupervised exploration policy, can we train a single policy that can solve any downstream task at test time without additional training?

The key idea is to learn a latent representation `z` of reward functions using a permutation-invariant transformer VAE. During training, diverse random reward functions are sampled from a prior distribution, encoded into `z`, and used to condition an Implicit Q-Learning (IQL) agent. At test time, a new reward function is encoded with just K=32 (state, reward) pairs, and the conditioned policy is deployed zero-shot.

### Architecture

- **Encoder**: Permutation-invariant transformer (3 layers, 4 heads, hidden dim 256) that maps K (state, reward) pairs → latent vector z ~ N(μ, σ²I)
- **Decoder**: MLP [256, 256] that predicts reward η(s) given (s, z)
- **IQL Agent**: Policy π(a|s,z), Q-function Q(s,a,z), Value function V(s,z), all conditioned on z via concatenation
- **Reward Prior**: Mixture of goal-reaching, random linear, and random MLP reward functions

### Key Results

| Domain | Task | FRE (Ours) |
|--------|------|------------|
| AntMaze | Goal-reaching | ~48.8 |
| AntMaze | Directional | ~55.2 |
| AntMaze | Overall mean | ~52.8 |
| ExORL Walker | Goal-reaching | ~94 |
| ExORL Cheetah | Goal-reaching | ~58 |
| Kitchen | 7 subtasks | ~66 |

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.12+ (CUDA 11.3+ recommended)
- MuJoCo 2.1+
- 1 GPU with ≥16GB RAM (NVIDIA RTX 2080 Ti or better)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd fre

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Install D4RL (for AntMaze and Kitchen datasets)
pip install git+https://github.com/Farama-Foundation/D4RL.git@master

# Install MuJoCo
pip install mujoco
```

### ExORL Dataset Setup

For ExORL experiments (Walker, Cheetah), download the unsupervised exploration datasets from the [ExORL repository](https://github.com/denisyarats/exorl):

```bash
# Download ExORL datasets to data/exorl/
mkdir -p data/exorl
# Follow instructions at https://github.com/denisyarats/exorl
# Place walker and cheetah datasets in data/exorl/walker/ and data/exorl/cheetah/
```

---

## Project Structure

```
project_root/
├── fre/                          # Core FRE implementation
│   ├── __init__.py               # Package initialization
│   ├── encoder.py                # Transformer VAE encoder (permutation-invariant)
│   ├── decoder.py                # MLP reward decoder
│   ├── reward_prior.py           # Random reward function distributions
│   ├── iql.py                    # Implicit Q-Learning agent (conditioned on z)
│   ├── fre_agent.py              # Main training loop (Algorithm 1)
│   └── utils.py                  # Replay buffer, data loading, helpers
├── experiments/                  # Experiment scripts
│   ├── train.py                  # Train FRE on a domain
│   ├── evaluate.py               # Zero-shot evaluation on downstream tasks
│   └── configs/                  # YAML configs per domain
│       ├── antmaze.yaml
│       ├── exorl_walker.yaml
│       ├── exorl_cheetah.yaml
│       └── kitchen.yaml
├── data/                         # Dataset storage (D4RL, ExORL)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## Training

Train FRE on a specific domain using the provided configuration files:

### AntMaze

```bash
python experiments/train.py --config experiments/configs/antmaze.yaml
```

### ExORL Walker

```bash
python experiments/train.py --config experiments/configs/exorl_walker.yaml
```

### ExORL Cheetah

```bash
python experiments/train.py --config experiments/configs/exorl_cheetah.yaml
```

### Kitchen

```bash
python experiments/train.py --config experiments/configs/kitchen.yaml
```

### Custom Configuration

Override any config parameter via command line:

```bash
python experiments/train.py --config experiments/configs/antmaze.yaml \
    --training.vae_steps 50000 \
    --training.iql_steps 500000 \
    --seed 42
```

### Training Phases

Training follows a **strided scheme** (Algorithm 1 in the paper):

1. **Phase 1 (VAE Training)**: Train encoder + decoder on diverse random reward functions for ~100k gradient steps. The encoder learns to map (state, reward) pairs to a compact latent vector z.
2. **Phase 2 (IQL Training)**: Freeze the encoder; train policy, Q, and V networks conditioned on z for ~1M gradient steps. Rewards are relabeled on-the-fly using sampled reward functions.

---

## Evaluation

Run zero-shot evaluation on a trained checkpoint:

```bash
# AntMaze evaluation
python experiments/evaluate.py --config experiments/configs/antmaze.yaml \
    --checkpoint checkpoints/antmaze/fre_agent_final.pt

# ExORL Walker evaluation
python experiments/evaluate.py --config experiments/configs/exorl_walker.yaml \
    --checkpoint checkpoints/walker/fre_agent_final.pt

# ExORL Cheetah evaluation
python experiments/evaluate.py --config experiments/configs/exorl_cheetah.yaml \
    --checkpoint checkpoints/cheetah/fre_agent_final.pt

# Kitchen evaluation
python experiments/evaluate.py --config experiments/configs/kitchen.yaml \
    --checkpoint checkpoints/kitchen/fre_agent_final.pt
```

### Evaluation Tasks

- **AntMaze**: Goal-reaching (random goals), directional (move in (x,y) direction), random-simplex (procedural noise), path tasks (edges, loop, center)
- **ExORL Walker/Cheetah**: Goal-reaching (to random states), velocity (forward/backward)
- **Kitchen**: 7 subtasks (microwave, kettle, light, slide cabinet, hinge cabinet, top burner, bottom burner)

### Metrics

- **Normalized return** (0-100 scale)
- **Success rate** (for goal-reaching and subtask completion)
- Results saved as JSON (full returns) and CSV (summary) in the checkpoint directory

---

## Reproducing Paper Results

### Experiment 1: Zero-shot transfer on AntMaze (Table 1, Figure 3)

```bash
# Train on antmaze-large-diverse-v2
python experiments/train.py --config experiments/configs/antmaze.yaml --seed 0
python experiments/train.py --config experiments/configs/antmaze.yaml --seed 1
python experiments/train.py --config experiments/configs/antmaze.yaml --seed 2
python experiments/train.py --config experiments/configs/antmaze.yaml --seed 3
python experiments/train.py --config experiments/configs/antmaze.yaml --seed 4

# Evaluate each seed
for seed in 0 1 2 3 4; do
    python experiments/evaluate.py --config experiments/configs/antmaze.yaml \
        --checkpoint checkpoints/antmaze_seed${seed}/fre_agent_final.pt
done
```

### Experiment 2: Zero-shot on ExORL (Table 1)

```bash
# Walker
for seed in 0 1 2 3 4; do
    python experiments/train.py --config experiments/configs/exorl_walker.yaml --seed $seed
    python experiments/evaluate.py --config experiments/configs/exorl_walker.yaml \
        --checkpoint checkpoints/walker_seed${seed}/fre_agent_final.pt
done

# Cheetah
for seed in 0 1 2 3 4; do
    python experiments/train.py --config experiments/configs/exorl_cheetah.yaml --seed $seed
    python experiments/evaluate.py --config experiments/configs/exorl_cheetah.yaml \
        --checkpoint checkpoints/cheetah_seed${seed}/fre_agent_final.pt
done
```

### Experiment 3: Zero-shot on Kitchen (Table 1)

```bash
for seed in 0 1 2 3 4; do
    python experiments/train.py --config experiments/configs/kitchen.yaml --seed $seed
    python experiments/evaluate.py --config experiments/configs/kitchen.yaml \
        --checkpoint checkpoints/kitchen_seed${seed}/fre_agent_final.pt
done
```

### Experiment 4: Scaling with reward diversity (Figure 5, Table 4)

Modify the `reward_prior.families` field in the config to test subsets:
- `["goal"]` — only goal-reaching
- `["linear"]` — only random linear
- `["mlp"]` — only random MLP
- `["goal", "linear"]` — goals + linear
- `["goal", "mlp"]` — goals + MLP
- `["linear", "mlp"]` — linear + MLP
- `["goal", "linear", "mlp"]` — all three (full FRE)

### Experiment 5: Domain knowledge augmentation (Figure 6)

For AntMaze, set `reward_prior.use_xy_prior: true` in the config to use XY-position-only reward functions, demonstrating improved performance through domain-specific priors.

---

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| d_z | 64 | Latent dimension |
| K | 32 | Number of (state, reward) pairs for encoding |
| Transformer layers | 3 | Encoder depth |
| Transformer heads | 4 | Attention heads |
| Hidden dim | 256 | Model width |
| Reward bins | 50 | Discretization bins for reward embedding |
| β (KL weight) | 1.0 | VAE regularization strength |
| τ (expectile) | 0.7 | IQL expectile parameter |
| α (AWR temperature) | 3.0 | Advantage-weighted regression temperature |
| γ (discount) | 0.99 | Discount factor |
| Learning rate | 3e-4 | Adam learning rate |
| VAE steps | 100k | Phase 1 training steps |
| IQL steps | 1M | Phase 2 training steps |
| Batch size | 256 | Training batch size |

---

## Monitoring Training

Training progress is logged to TensorBoard (and optionally Weights & Biases):

```bash
# View TensorBoard logs
tensorboard --logdir logs/

# Enable Weights & Biases logging
python experiments/train.py --config experiments/configs/antmaze.yaml --use_wandb true
```

---

## Checkpoints

Checkpoints are saved automatically during training:

- `checkpoints/<domain>/fre_agent_step_<N>.pt` — periodic checkpoints
- `checkpoints/<domain>/fre_agent_final.pt` — final checkpoint after training

Each checkpoint contains:
- Encoder, decoder, and IQL network weights
- Optimizer states
- Training step counters
- Configuration metadata

---

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{fransen2024functional,
  title={Functional Reward Encodings for Zero-Shot Offline Reinforcement Learning},
  author={Fransen, Kevin and others},
  journal={arXiv preprint},
  year={2024}
}
```

---

## Acknowledgments

This implementation builds upon:
- [D4RL](https://github.com/Farama-Foundation/D4RL) for offline RL datasets
- [ExORL](https://github.com/denisyarats/exorl) for unsupervised exploration data
- [IQL](https://github.com/ikostrikov/implicit_q_learning) for the base offline RL algorithm

---

## License

This project is provided for research purposes. See the original paper for licensing details.

---

## Troubleshooting

### D4RL Installation Issues

If `pip install d4rl` fails, try:
```bash
pip install git+https://github.com/Farama-Foundation/D4RL.git@master
```

### MuJoCo Rendering

For headless rendering on servers:
```bash
export MUJOCO_GL=egl
# or
export MUJOCO_GL=osmesa
```

### Out of Memory

Reduce batch size or model dimensions:
```bash
python experiments/train.py --config experiments/configs/antmaze.yaml \
    --training.vae_batch_size 128 \
    --training.iql_batch_size 128
```

### ExORL Dataset Not Found

Ensure datasets are placed in the correct directory structure:
```
data/exorl/walker/  # Walker dataset files
data/exorl/cheetah/ # Cheetah dataset files
```