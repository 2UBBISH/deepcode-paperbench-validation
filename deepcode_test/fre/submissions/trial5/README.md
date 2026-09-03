# Functional Reward Encodings (FRE) for Zero-Shot Offline Reinforcement Learning

This repository contains a reproduction of the paper **"Functional Reward Encodings (FRE) for Zero-Shot Offline Reinforcement Learning"**. FRE is a general unsupervised method that learns a functional representation of arbitrary reward functions via a transformer-based variational auto-encoder, enabling a single pre-trained agent to solve novel downstream tasks given only a few reward-annotated state samples.

## Overview

The key idea behind FRE is to train an encoder that maps a set of (state, reward) pairs to a latent vector `z` that captures the structure of an arbitrary reward function. This latent encoding is then used to condition an offline RL agent (Implicit Q-Learning, IQL) so that at test time, given only a few examples of a new reward function, the agent can zero-shot generalize to solve the corresponding task.

### Architecture

- **FRE Encoder**: A permutation-invariant Transformer that encodes a set of `(state, reward)` pairs into a latent Gaussian distribution `p(z | {(s_i, η(s_i))})`.
- **FRE Decoder**: A feedforward MLP that predicts the reward `η(s)` for a state `s` given the latent code `z`.
- **IQL Agent**: An Implicit Q-Learning agent (Q-network, V-network, Gaussian policy) conditioned on the latent `z`.
- **Prior Reward Distribution**: A uniform mixture of three unsupervised reward families:
  - **Singleton (Goal-reaching)**: Reward = -1 until goal reached.
  - **Random Linear**: `reward(s) = dot(w, s)` with sparse random weights.
  - **Random MLP**: A randomly initialized 2-layer MLP.

### Training

Training proceeds in two phases:

1. **Phase 1 (Encoder Training)**: The encoder and decoder are trained jointly using a β-VAE objective on randomly sampled reward functions from the prior distribution.
2. **Phase 2 (RL Training)**: The encoder is frozen, and the IQL agent is trained on offline data with rewards computed by randomly sampled reward functions, conditioned on the latent encoding `z`.

Optionally, a **strided training** scheme alternates between encoder and RL updates to mitigate catastrophic forgetting.

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.12+
- MuJoCo 2.0/2.1 (for D4RL environments)
- CUDA 11.3+ (recommended for GPU training)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd fre

# Create a conda environment (recommended)
conda create -n fre python=3.8
conda activate fre

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### D4RL and MuJoCo

D4RL requires MuJoCo. Install MuJoCo following the [official instructions](https://github.com/deepmind/mujoco), then:

```bash
pip install d4rl>=1.1
```

### ExORL (Optional)

For ExORL benchmark experiments, download datasets from the [controllable_agent repository](https://github.com/facebookresearch/controllable_agent) or install the `exorl` package:

```bash
pip install exorl dm_control
```

## Project Structure

```
project_root/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package installation script
├── configs/                           # Configuration files
│   ├── default.yaml                   # Default hyperparameters
│   ├── antmaze.yaml                   # AntMaze-specific config
│   ├── exorl.yaml                     # ExORL-specific config
│   └── kitchen.yaml                   # Kitchen-specific config
├── data/                              # Data loading and processing
│   ├── __init__.py
│   ├── dataset.py                     # Offline dataset loader (D4RL, ExORL)
│   └── replay_buffer.py              # Replay buffer for offline RL
├── models/                            # Neural network models
│   ├── __init__.py
│   ├── fre_encoder.py                # Transformer-based VAE encoder
│   ├── fre_decoder.py                # Feedforward decoder for reward prediction
│   ├── iql_agent.py                  # IQL agent (Q, V, policy networks)
│   └── reward_embedding.py           # Reward discretization and embedding
├── rewards/                           # Prior reward distribution
│   ├── __init__.py
│   ├── base.py                       # Abstract reward function class
│   ├── singleton.py                  # Goal-reaching singleton rewards
│   ├── linear.py                     # Random linear reward functions
│   ├── mlp.py                        # Random MLP reward functions
│   └── mixture.py                    # Mixture distribution over reward families
├── training/                          # Training loops
│   ├── __init__.py
│   ├── train_encoder.py              # Phase 1: Train FRE encoder only
│   ├── train_rl.py                   # Phase 2: Train IQL with frozen encoder
│   └── trainer.py                    # Main orchestrator (strided training)
├── evaluation/                        # Evaluation and metrics
│   ├── __init__.py
│   ├── evaluator.py                  # Zero-shot evaluation on downstream tasks
│   └── metrics.py                    # Normalized return computation
├── utils/                             # Utilities
│   ├── __init__.py
│   ├── logger.py                     # Logging and checkpointing
│   └── helpers.py                    # Miscellaneous helpers
└── scripts/                           # Executable entry points
    ├── train.py                      # Main training script
    ├── evaluate.py                   # Evaluation script
    └── demo.py                       # Optional visualization/demo
```

## Usage

### Training

Train an FRE agent on a specific domain:

```bash
# Train on AntMaze
python scripts/train.py --domain antmaze

# Train on ExORL Walker
python scripts/train.py --domain exorl_walker

# Train on ExORL Cheetah
python scripts/train.py --domain exorl_cheetah

# Train on Kitchen
python scripts/train.py --domain kitchen
```

Key command-line arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--domain` | Domain to train on (`antmaze`, `exorl_walker`, `exorl_cheetah`, `kitchen`) | `antmaze` |
| `--config` | Path to base config file | `configs/default.yaml` |
| `--domain_config` | Path to domain-specific config (auto-inferred if not set) | auto |
| `--encoder_steps` | Number of Phase 1 encoder training steps | from config |
| `--rl_steps` | Number of Phase 2 RL training steps | from config |
| `--strided` | Enable strided training | `False` |
| `--seed` | Random seed | `0` |
| `--device` | Device (`cuda` or `cpu`) | `cuda` if available |
| `--resume` | Path to checkpoint to resume from | `None` |
| `--eval_during_training` | Run zero-shot evaluation periodically | `False` |
| `--use_wandb` | Enable Weights & Biases logging | `False` |
| `--wandb_project` | W&B project name | `fre` |

### Evaluation

Evaluate a trained checkpoint on downstream tasks:

```bash
# Evaluate a single checkpoint
python scripts/evaluate.py --checkpoint path/to/checkpoint.pt --domain antmaze

# Multi-seed evaluation (all checkpoints in a directory)
python scripts/evaluate.py --checkpoint_dir path/to/checkpoints/ --domain antmaze --multi_seed
```

Key evaluation arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--checkpoint` | Path to a single checkpoint file | required |
| `--checkpoint_dir` | Directory with multiple seed checkpoints | `None` |
| `--multi_seed` | Run multi-seed evaluation | `False` |
| `--num_episodes` | Episodes per task | `20` |
| `--K_enc` | Number of encoding states | `32` |
| `--seed` | Evaluation seed | `0` |

### Demo

Run a demonstration of zero-shot transfer:

```bash
# Demo goal-reaching on AntMaze
python scripts/demo.py --checkpoint path/to/checkpoint.pt --domain antmaze --task goal_reaching

# Demo with rendering
python scripts/demo.py --checkpoint path/to/checkpoint.pt --domain antmaze --task directional --render

# Compare latent encodings of different reward functions
python scripts/demo.py --checkpoint path/to/checkpoint.pt --domain antmaze --compare_latents
```

## Reproducing Paper Results

### Experiment 1: Zero-Shot Transfer on AntMaze (Table 1)

```bash
# Train for 5 seeds
for seed in 0 1 2 3 4; do
    python scripts/train.py --domain antmaze --seed $seed
done

# Evaluate all seeds
python scripts/evaluate.py --checkpoint_dir checkpoints/antmaze/ --domain antmaze --multi_seed
```

**Expected Results** (normalized returns, mean ± std):

| Task | Expected |
|------|----------|
| goal-reaching | 48.8 ± 6 |
| directional | 55.2 ± 8 |
| random-simplex | 21.3 ± 4 |
| path-loop | 67.2 ± 36 |
| path-edges | 60.0 ± 17 |
| path-center | 64.4 ± 38 |
| **Average** | **52.8 ± 18.2** |

### Experiment 2: Zero-Shot Transfer on ExORL (Table 1)

```bash
# Train on Walker
for seed in 0 1 2 3 4; do
    python scripts/train.py --domain exorl_walker --seed $seed
done

# Train on Cheetah
for seed in 0 1 2 3 4; do
    python scripts/train.py --domain exorl_cheetah --seed $seed
done

# Evaluate
python scripts/evaluate.py --checkpoint_dir checkpoints/exorl_walker/ --domain exorl_walker --multi_seed
python scripts/evaluate.py --checkpoint_dir checkpoints/exorl_cheetah/ --domain exorl_cheetah --multi_seed
```

**Expected Results**:

| Task | Expected |
|------|----------|
| walker-goals | 94 ± 2 |
| cheetah-goals | 58 ± 8 |
| walker-velocity | 34 ± 13 |
| cheetah-velocity | 20 ± 2 |
| **Average** | **51.5 ± 6.3** |

### Experiment 3: Zero-Shot Transfer on Kitchen (Table 1)

```bash
for seed in 0 1 2 3 4; do
    python scripts/train.py --domain kitchen --seed $seed
done

python scripts/evaluate.py --checkpoint_dir checkpoints/kitchen/ --domain kitchen --multi_seed
```

**Expected Result**: Kitchen average: **66 ± 3**

### Experiment 4: Scaling Properties (Section 5.3)

Train agents with subsets of reward families by modifying the reward mixture weights in the config:

```yaml
# Singleton only
reward:
  singleton_weight: 1.0
  linear_weight: 0.0
  mlp_weight: 0.0

# Linear only
reward:
  singleton_weight: 0.0
  linear_weight: 1.0
  mlp_weight: 0.0

# MLP only
reward:
  singleton_weight: 0.0
  linear_weight: 0.0
  mlp_weight: 1.0

# All three (default)
reward:
  singleton_weight: 0.333
  linear_weight: 0.333
  mlp_weight: 0.333
```

### Experiment 5: Domain Knowledge Augmentation (Section 5.4)

Add domain-specific reward functions to the prior by extending the mixture distribution (modify `rewards/mixture.py` or create custom reward families).

## Configuration

All hyperparameters are managed through YAML configuration files:

- `configs/default.yaml`: Base configuration with all parameters and defaults.
- `configs/antmaze.yaml`: AntMaze-specific overrides.
- `configs/exorl.yaml`: ExORL-specific overrides.
- `configs/kitchen.yaml`: Kitchen-specific overrides.

Configuration is built by merging `default.yaml` → domain-specific config → CLI arguments (highest priority).

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `encoder.latent_dim` | 64 | Dimension of latent reward encoding |
| `encoder.embed_dim` | 256 | Transformer hidden dimension |
| `encoder.num_layers` | 3 | Number of transformer layers |
| `encoder.num_heads` | 4 | Number of attention heads |
| `encoder.num_bins` | 64 | Reward discretization bins |
| `encoder_training.K_enc` | 64 | Number of encoding states |
| `encoder_training.K_dec` | 64 | Number of decoding states |
| `encoder_training.beta_kl` | 0.1 | KL divergence penalty weight |
| `encoder_training.num_steps` | 100000 | Phase 1 training steps |
| `agent.expectile` | 0.7 | IQL expectile parameter |
| `agent.temperature` | 3.0 | AWR temperature |
| `agent.discount` | 0.99 | Discount factor |
| `rl_training.num_steps` | 1000000 | Phase 2 training steps |
| `rl_training.batch_size` | 256 | RL batch size |
| `evaluation.K_enc` | 32 | Encoding states for evaluation |

## Datasets

### D4RL

- **AntMaze**: `antmaze-large-diverse-v2` (loaded via `d4rl`)
- **Kitchen**: `kitchen-complete-v0` (loaded via `d4rl`)

### ExORL

- **Walker**: ExORL Walker dataset (requires manual download or `exorl` package)
- **Cheetah**: ExORL Cheetah dataset (requires manual download or `exorl` package)

ExORL datasets can be downloaded from the [FB controllable_agent repository](https://github.com/facebookresearch/controllable_agent).

## Logging

Training metrics are logged to the console and optionally to files in the log directory. Weights & Biases integration is available via `--use_wandb`.

Checkpoints are saved periodically and can be used for resumption or evaluation.

## Citation

If you use this code in your research, please cite the original paper:

```
@article{frans2024functional,
  title={Functional Reward Encodings for Zero-Shot Offline Reinforcement Learning},
  author={Frans, Kevin and Park, Seohong and Levine, Sergey},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is provided for research purposes. See the original paper for licensing details.

## Acknowledgements

This implementation is based on the FRE paper. The IQL implementation follows Kostrikov et al. (2021). D4RL datasets are from Fu et al. (2020). ExORL datasets are from Yarats et al. (2022).