# Functional Reward Encodings (FRE)

**Zero-Shot Offline Reinforcement Learning via Functional Reward Encodings**

This repository implements the Functional Reward Encodings (FRE) method described in the paper:

> *"Functional Reward Encodings (FRE) for Zero-Shot Offline Reinforcement Learning"*

FRE pre-trains a generalist agent from unlabeled offline trajectories using a transformer-based variational auto-encoder that encodes arbitrary reward functions into a latent space. The agent can then solve novel downstream tasks **zero-shot**, given only a few reward-annotated state samples, without any fine-tuning.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Reproducing Results](#reproducing-results)
  - [AntMaze](#antmaze)
  - [ExORL (Walker & Cheetah)](#exorl-walker--cheetah)
  - [Kitchen](#kitchen)
- [Configuration](#configuration)
- [Evaluation Tasks](#evaluation-tasks)
- [Expected Results](#expected-results)
- [Citation](#citation)
- [License](#license)

---

## Overview

FRE consists of two training phases:

1. **Phase 1 — Encoder Pre-training (VAE):** A permutation-invariant transformer VAE is trained on random unsupervised reward functions (goal-reaching, linear, MLP) to learn a latent encoding `z` that captures reward function structure. The encoder maps a set of `(state, reward)` pairs to a latent vector `z`, and a decoder reconstructs rewards from `(state, z)`.

2. **Phase 2 — IQL Agent Training:** With the encoder frozen, an Implicit Q-Learning (IQL) agent — with Q, V, and policy networks all conditioned on `z` — is trained on the offline dataset using the same random reward prior. The agent learns to maximize arbitrary reward functions encoded by `z`.

At **evaluation time**, given only ~32 reward-labeled states for a new downstream task, the frozen encoder produces a latent `z`, and the IQL policy conditioned on `z` solves the task zero-shot.

---

## Installation

### Prerequisites

- **Python 3.8+** (3.9 recommended)
- **MuJoCo 2.1+** (license required for D4RL/ExORL environments)
- **GPU** with at least 8 GB VRAM (NVIDIA RTX 2080 or better recommended)

### Step 1: Create a Conda Environment

```bash
conda create -n fre python=3.9
conda activate fre
```

### Step 2: Install PyTorch

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or CPU-only (very slow for training)
pip install torch torchvision torchaudio
```

### Step 3: Install D4RL

```bash
pip install d4rl
```

D4RL provides the AntMaze and Kitchen datasets. Datasets are automatically downloaded on first use.

### Step 4: Install ExORL

```bash
git clone https://github.com/denisyarats/exorl.git
cd exorl
pip install -e .
cd ..
```

ExORL provides the Walker and Cheetah datasets from the DeepMind Control Suite. Datasets must be downloaded separately (see [ExORL documentation](https://github.com/denisyarats/exorl)).

### Step 5: Install Other Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Verify Installation

```bash
python -c "import d4rl; import exorl; import torch; print('All dependencies OK')"
```

---

## Project Structure

```
fre/
├── models/
│   ├── __init__.py          # Model exports
│   ├── encoder.py           # Transformer VAE encoder (permutation-invariant)
│   ├── decoder.py           # Feedforward reward decoder
│   └── iql.py               # IQL networks: Q, V, policy (z-conditioned)
├── reward_functions/
│   ├── __init__.py          # Reward function exports
│   ├── base.py              # Abstract reward function class
│   ├── singleton.py         # Goal-reaching singleton rewards
│   ├── linear.py            # Random linear functions with sparse mask
│   ├── mlp.py               # Random 2-layer MLP functions
│   ├── mixture.py           # Uniform mixture of the three types
│   └── eval_rewards.py      # Evaluation reward functions for all benchmark tasks
├── training/
│   ├── __init__.py          # Training exports
│   ├── train_encoder.py     # Phase 1: train FRE encoder+decoder (VAE)
│   ├── train_iql.py         # Phase 2: train IQL with frozen encoder
│   └── utils.py             # Data sampling, reward computation, logging
├── evaluation/
│   ├── __init__.py          # Evaluation exports
│   └── evaluate.py          # Zero-shot evaluation on downstream tasks
├── data/
│   ├── __init__.py          # Data exports
│   └── dataset.py           # Offline dataset loading (D4RL, ExORL) and replay buffer
├── scripts/
│   ├── run_antmaze.sh       # Full pipeline for AntMaze
│   ├── run_exorl.sh         # Full pipeline for ExORL (Walker/Cheetah)
│   └── run_kitchen.sh       # Full pipeline for Kitchen
├── main.py                  # Entry point: orchestrates training and evaluation
├── config.py                # All hyperparameters and configuration
└── requirements.txt         # Python dependencies
```

---

## Quick Start

### Train and Evaluate on AntMaze (Single Command)

```bash
bash fre/scripts/run_antmaze.sh --gpu 0 --seed 0
```

This will:
1. Train the FRE encoder (Phase 1) — ~2–4 hours
2. Train the IQL agent (Phase 2) — ~6–12 hours
3. Evaluate zero-shot on all 16 AntMaze tasks — ~30 minutes

Results are saved to `./results/antmaze/`.

### Train and Evaluate on ExORL Walker

```bash
bash fre/scripts/run_exorl.sh --gpu 0 --seed 0 --domain exorl_walker
```

### Train and Evaluate on Kitchen

```bash
bash fre/scripts/run_kitchen.sh --gpu 0 --seed 0
```

---

## Reproducing Results

### AntMaze

The AntMaze domain uses the D4RL `antmaze-large-diverse-v2` dataset.

**Step 1: Train Encoder (Phase 1)**

```bash
python -m fre.main --mode train_encoder --domain antmaze --gpu 0 --seed 0
```

This trains the VAE encoder+decoder for 200,000 steps. Checkpoints are saved to `./checkpoints/antmaze/`.

**Step 2: Train IQL Agent (Phase 2)**

```bash
python -m fre.main --mode train_iql --domain antmaze --gpu 0 --seed 0 \
    --encoder_checkpoint ./checkpoints/antmaze/encoder_best.pt
```

This trains the IQL agent for 1,000,000 steps with the frozen encoder.

**Step 3: Evaluate Zero-Shot**

```bash
python -m fre.main --mode evaluate_multi --domain antmaze --gpu 0 --seed 0 \
    --encoder_checkpoint ./checkpoints/antmaze/encoder_best.pt \
    --iql_checkpoint ./checkpoints/antmaze/iql_best.pt
```

Evaluates on all 16 AntMaze tasks across 5 random seeds.

**Or run all steps together:**

```bash
python -m fre.main --mode train_all --domain antmaze --gpu 0 --seed 0
```

### ExORL (Walker & Cheetah)

The ExORL domain uses datasets from the [ExORL benchmark](https://github.com/denisyarats/exorl).

**Walker:**

```bash
# Phase 1
python -m fre.main --mode train_encoder --domain exorl_walker --gpu 0 --seed 0

# Phase 2
python -m fre.main --mode train_iql --domain exorl_walker --gpu 0 --seed 0 \
    --encoder_checkpoint ./checkpoints/exorl_walker/encoder_best.pt

# Evaluation
python -m fre.main --mode evaluate_multi --domain exorl_walker --gpu 0 --seed 0 \
    --encoder_checkpoint ./checkpoints/exorl_walker/encoder_best.pt \
    --iql_checkpoint ./checkpoints/exorl_walker/iql_best.pt
```

**Cheetah:**

Replace `exorl_walker` with `exorl_cheetah` in the commands above.

### Kitchen

The Kitchen domain uses the D4RL `kitchen-complete-v0` dataset.

```bash
# Phase 1
python -m fre.main --mode train_encoder --domain kitchen --gpu 0 --seed 0

# Phase 2
python -m fre.main --mode train_iql --domain kitchen --gpu 0 --seed 0 \
    --encoder_checkpoint ./checkpoints/kitchen/encoder_best.pt

# Evaluation
python -m fre.main --mode evaluate_multi --domain kitchen --gpu 0 --seed 0 \
    --encoder_checkpoint ./checkpoints/kitchen/encoder_best.pt \
    --iql_checkpoint ./checkpoints/kitchen/iql_best.pt
```

---

## Configuration

All hyperparameters are centralized in `fre/config.py`. Key defaults:

### Encoder (VAE)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `K` | 32 | Number of encoder states |
| `K_prime` | 32 | Number of decoder states |
| `d_embed` | 128 | Reward embedding dimension |
| `d_model` | 256 | Transformer hidden dimension |
| `num_layers` | 2 | Transformer encoder layers |
| `num_heads` | 4 | Attention heads |
| `d_latent` | 64 | Latent z dimension |
| `num_reward_bins` | 64 | Reward discretization bins |
| `r_max` | 10.0 | Reward clipping range |
| `beta_kl` | 0.1 | KL divergence weight |
| `encoder_steps` | 200,000 | Phase 1 training steps |
| `encoder_lr` | 1e-4 | Learning rate |

### IQL Agent
| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau` | 0.7 | Expectile for value regression |
| `beta` | 3.0 | Temperature for AWR policy update |
| `gamma` | 0.99 | Discount factor |
| `iql_lr` | 3e-4 | Learning rate |
| `batch_size` | 256 | Training batch size |
| `target_update_rate` | 0.005 | Soft target update rate |
| `iql_steps` | 1,000,000 | Phase 2 training steps |

### Evaluation
| Parameter | Default | Description |
|-----------|---------|-------------|
| `K_eval` | 32 | Encoder states for evaluation |
| `num_episodes` | 20 | Episodes per task per seed |
| `num_seeds` | 5 | Random seeds for evaluation |

Override any parameter via command line:

```bash
python -m fre.main --mode train_encoder --domain antmaze --K 64 --beta_kl 0.5
```

---

## Evaluation Tasks

### AntMaze (16 tasks)
- **Goal-reaching (4 tasks):** Reach specific (x, y) locations in the maze
- **Directional (4 tasks):** Maximize velocity in a given direction
- **Random Simplex (4 tasks):** Random Fourier-feature reward functions over the maze
- **Path (4 tasks):** Follow specific paths (edges, loop, center, waypoints)

### ExORL Walker (3 tasks)
- **Goal-reaching:** Reach a randomly sampled goal state
- **Velocity forward:** Maximize forward velocity
- **Velocity backward:** Maximize backward velocity

### ExORL Cheetah (3 tasks)
- **Goal-reaching:** Reach a randomly sampled goal state
- **Velocity forward:** Maximize forward velocity
- **Velocity backward:** Maximize backward velocity

### Kitchen (8 tasks)
- **7 individual subtasks:** Microwave, Kettle, Light, Slide Cabinet, Hinge Cabinet, Top Burner, Bottom Burner
- **All subtasks:** Complete all 7 subtasks simultaneously

---

## Expected Results

Results are normalized between 0 and 100 (higher is better). The table below shows approximate expected performance from the paper (mean ± std over 5 seeds):

| Domain | Task | FRE Score |
|--------|------|-----------|
| AntMaze | Goal-reaching | 48.8 ± 6 |
| AntMaze | Directional | 55.2 ± 8 |
| AntMaze | Random Simplex | 21.3 ± 4 |
| AntMaze | Path (loop) | 67.2 ± 36 |
| AntMaze | Path (edges) | 60.0 ± 17 |
| AntMaze | Path (center) | 64.4 ± 38 |
| ExORL Walker | Goals | 94 ± 2 |
| ExORL Walker | Velocity | 34 ± 13 |
| ExORL Cheetah | Goals | 58 ± 8 |
| ExORL Cheetah | Velocity | 20 ± 2 |
| Kitchen | All subtasks | 66 ± 3 |
| **Overall Average** | | **~57 ± 9** |

**Note:** Exact results may vary due to random seeds, hardware, and software versions. For best reproduction, use multiple seeds (5+) and report mean ± std.

---

## Ablation Studies

To reproduce the ablation studies from the paper (Figures 5, 6):

### Reward Family Ablation (Figure 5)

Train FRE agents using subsets of the prior reward distribution:

```bash
# Only goal-reaching rewards
python -m fre.main --mode train_all --domain antmaze --reward_types singleton

# Only linear rewards
python -m fre.main --mode train_all --domain antmaze --reward_types linear

# Only MLP rewards
python -m fre.main --mode train_all --domain antmaze --reward_types mlp

# Goals + Linear
python -m fre.main --mode train_all --domain antmaze --reward_types singleton,linear

# Full mixture (all three)
python -m fre.main --mode train_all --domain antmaze --reward_types singleton,linear,mlp
```

### Domain Knowledge Augmentation (Figure 6)

Add domain-specific reward functions (e.g., XY-position-only for AntMaze) to the prior:

```bash
python -m fre.main --mode train_all --domain antmaze --augment_domain_knowledge
```

---

## Logging and Monitoring

Training progress is logged via TensorBoard:

```bash
tensorboard --logdir ./logs/
```

Metrics logged include:
- **Phase 1:** Total VAE loss, reconstruction loss, KL divergence
- **Phase 2:** Q loss, V loss, policy loss, average Q-value, average return (offline proxy)

Evaluation results are saved as JSON files in `./results/`.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 8 GB | 16+ GB |
| RAM | 16 GB | 32 GB |
| Disk | 10 GB | 50 GB (for datasets) |

**Training Time Estimates (NVIDIA RTX 2080 Ti):**
- Phase 1 (Encoder): ~2–4 hours per domain
- Phase 2 (IQL): ~6–12 hours per domain
- Total per domain: ~8–16 hours

---

## Troubleshooting

### D4RL Import Errors
If you encounter `ImportError` for D4RL, ensure MuJoCo is properly installed:
```bash
pip install mujoco==2.3.7
```

### ExORL Dataset Not Found
ExORL datasets must be downloaded separately. See the [ExORL repository](https://github.com/denisyarats/exorl) for download instructions. By default, datasets are expected in `~/.exorl/`.

### Out of Memory (OOM)
Reduce batch size or model dimensions:
```bash
python -m fre.main --mode train_encoder --domain antmaze --batch_size 128 --d_model 128
```

### Slow Training
- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Use mixed precision training (not yet implemented; consider adding AMP)

---

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{fransen2024fre,
  title   = {Functional Reward Encodings (FRE) for Zero-Shot Offline Reinforcement Learning},
  author  = {Fransen, ...},
  journal = {...},
  year    = {2024}
}
```

---

## License

This implementation is provided for research purposes. See the original paper and dataset licenses (D4RL, ExORL, MuJoCo) for terms of use.

---

## Acknowledgments

This implementation builds upon:
- [D4RL](https://github.com/Farama-Foundation/D4RL) — Offline datasets for AntMaze and Kitchen
- [ExORL](https://github.com/denisyarats/exorl) — Offline datasets for Walker and Cheetah
- [Implicit Q-Learning (IQL)](https://github.com/ikostrikov/implicit_q_learning) — Base offline RL algorithm
- [Forward-Backward (FB) Representations](https://github.com/facebookresearch/controllable_agent) — Related zero-shot RL work