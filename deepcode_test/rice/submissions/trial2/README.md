# RICE: Refining via Critical State Explanation

Official implementation of the paper **"RICE: Refining via Critical State Explanation"** — a method to refine reinforcement learning agents by identifying critical states via a mask network (explanation) and using them to create a mixed initial state distribution with exploration bonus, improving performance and overcoming training bottlenecks.

## Overview

RICE (Refining via CrItical state Explanation) addresses a fundamental challenge in deep reinforcement learning: agents often plateau during training due to insufficient exploration of critical states. RICE tackles this by:

1. **Training a mask network** ξ(s) that learns to identify critical states — states where the agent's action is essential for success.
2. **Extracting critical states** from trajectories using the mask network's importance scores.
3. **Refining the agent** by continuing training with:
   - A **mixed initial state distribution**: with probability `p`, episodes start from a critical state (sampled from the buffer); otherwise, from the default initial distribution.
   - An **RND exploration bonus**: intrinsic reward encouraging the agent to visit novel states.

This approach provides theoretical guarantees (Theorem 3.6) and empirically outperforms baselines including StateMask, Jump-Start RL (JSRL), Self-Imitation Learning (SIL), and random explanation across MuJoCo, blockchain, cybersecurity, autonomous driving, and malware domains.

## Repository Structure

```
project_root/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
├── config/                      # Hyperparameter configuration files
│   ├── default.yaml             # Default hyperparameters
│   └── env_specific/            # Per-environment overrides
│       ├── hopper.yaml
│       ├── walker2d.yaml
│       ├── reacher.yaml
│       ├── halfcheetah.yaml
│       ├── selfish_mining.yaml
│       ├── cage.yaml
│       ├── auto_driving.yaml
│       └── malware.yaml
├── rice/                        # Core RICE implementation
│   ├── __init__.py
│   ├── mask_network.py          # Mask network model and training
│   ├── explanation.py           # Critical state identification
│   ├── refining.py              # Refining process with mixed init dist
│   ├── rnd.py                   # Random Network Distillation module
│   ├── perturbed_env.py         # Environment wrapper for mask training
│   └── utils.py                 # Helpers (set_state, buffers, etc.)
├── envs/                        # Custom environment integrations
│   ├── __init__.py
│   ├── selfish_mining_env.py    # Selfish mining blockchain env
│   ├── cage_env.py              # CAGE Challenge 2 env
│   ├── auto_driving_env.py      # MetaDrive autonomous driving env
│   └── malware_env.py           # MalConv malware mutation env
├── experiments/                 # Experiment scripts
│   ├── train_agent.py           # Train initial PPO agent
│   ├── train_mask.py            # Train mask network
│   ├── refine.py                # Run RICE refining
│   ├── evaluate.py              # Evaluate and compare methods
│   └── run_experiments.sh       # Shell script to reproduce all experiments
└── baselines/                   # Baseline method implementations
    ├── __init__.py
    ├── statemask.py             # StateMask explanation baseline
    ├── jsrl.py                  # Jump-Start RL baseline
    ├── sil.py                   # Self-Imitation Learning baseline
    └── random_explanation.py    # Random explanation baseline
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU with >= 16GB VRAM (recommended; CPU-only feasible for small-scale tests)
- MuJoCo (for MuJoCo environments)

### Quick Install

```bash
# Clone the repository
git clone <repository-url>
cd rice

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Environment-Specific Setup

#### MuJoCo
```bash
pip install mujoco>=2.3.0
# Follow official MuJoCo installation if needed: https://mujoco.org/
```

#### MetaDrive (Autonomous Driving)
```bash
pip install metadrive>=0.4.0
```

#### Malware (MalConv)
```bash
pip install tianshou>=0.5.0
# Optional: install MalConv model from https://github.com/bfilar/malware_rl
```

#### Selfish Mining
```bash
# Clone the reference implementation
git clone https://github.com/roibarzur/pto-selfish-mining
# Our envs/selfish_mining_env.py provides a standalone implementation
```

#### CAGE Challenge 2
```bash
# Reference: https://github.com/john-cardiff/-cyborg-cage-2
# Our envs/cage_env.py provides a standalone simulation
```

## Quick Start

### 1. Train a Base Agent

```bash
# Train PPO on Hopper-v3
python experiments/train_agent.py --env Hopper-v3 --total-timesteps 1000000 --output-dir ./outputs/hopper

# Or use the installed entry point
rice-train-agent --env Hopper-v3 --total-timesteps 1000000 --output-dir ./outputs/hopper
```

### 2. Train the Mask Network

```bash
# Train mask network using the pre-trained agent
python experiments/train_mask.py --env Hopper-v3 --agent-path ./outputs/hopper/final_model.zip --output-dir ./outputs/hopper/mask

# Or use the entry point
rice-train-mask --env Hopper-v3 --agent-path ./outputs/hopper/final_model.zip --output-dir ./outputs/hopper/mask
```

### 3. Run RICE Refining

```bash
# Refine the agent using critical states and RND exploration
python experiments/refine.py --env Hopper-v3 --agent-path ./outputs/hopper/final_model.zip --critical-states-path ./outputs/hopper/mask/critical_states.pkl --output-dir ./outputs/hopper/refined

# Or use the entry point
rice-refine --env Hopper-v3 --agent-path ./outputs/hopper/final_model.zip --critical-states-path ./outputs/hopper/mask/critical_states.pkl --output-dir ./outputs/hopper/refined
```

### 4. Evaluate and Compare

```bash
# Run all experiments and comparisons
python experiments/evaluate.py --env Hopper-v3 --experiment all --output-dir ./outputs/hopper/eval

# Or use the entry point
rice-evaluate --env Hopper-v3 --experiment all --output-dir ./outputs/hopper/eval
```

## Reproducing Paper Experiments

### One-Command Reproduction

The shell script `experiments/run_experiments.sh` automates the entire reproduction pipeline:

```bash
# Reproduce all experiments for all environments
bash experiments/run_experiments.sh --env all --experiment all

# Reproduce only MuJoCo experiments
bash experiments/run_experiments.sh --env hopper,walker2d,reacher,halfcheetah --experiment all

# Reproduce a specific experiment
bash experiments/run_experiments.sh --env hopper --experiment refining

# With multiple seeds
bash experiments/run_experiments.sh --env hopper --experiment all --n-seeds 5 --seed 0
```

### Experiment I: Fidelity Comparison

Evaluates the quality of critical state identification by measuring the drop in return when randomizing actions at identified critical states vs. random states.

```bash
bash experiments/run_experiments.sh --env hopper --experiment fidelity
```

**Expected Results (Figure 5):** RICE achieves fidelity scores similar to StateMask, significantly higher than random explanation.

### Experiment II: Efficiency Comparison

Measures wall-clock time to train the mask network, comparing RICE's simplified training vs. StateMask's Lagrangian method.

```bash
bash experiments/run_experiments.sh --env hopper --experiment efficiency
```

**Expected Results (Table 4):** RICE is ~16.8% faster on average. E.g., Hopper: ~12,426s vs ~15,393s for StateMask.

### Experiment III: Refining Performance (MuJoCo)

Compares RICE-refined agents against baselines (PPO fine-tuning, StateMask-R, JSRL, SIL, random explanation) on MuJoCo environments.

```bash
bash experiments/run_experiments.sh --env hopper,walker2d,reacher,halfcheetah --experiment refining
```

**Expected Results (Tables 5-6):**
| Environment | RICE (Ours) | SIL | Random |
|-------------|-------------|-----|--------|
| Hopper | 3663.91±20.98 | 3646.46±23.12 | 3648.98±39.06 |
| Walker2d | 3982.79±3.15 | 3967.66±1.53 | — |
| Reacher | -2.66±0.03 | -2.87±0.09 | — |
| HalfCheetah | 2138.89±3.22 | 2069.80±3.44 | — |

### Experiment IV: Other Applications

Applies RICE to Selfish Mining, CAGE Challenge 2, Autonomous Driving, and Malware Mutation.

```bash
bash experiments/run_experiments.sh --env selfish_mining,cage,auto_driving,malware --experiment applications
```

**Expected Results (Table 1, Table 7):** RICE achieves highest average return / evasion probability in each domain. Malware evasion: 68.2% (full RICE) vs 33.8% (original).

### Experiment V: Case Study & Sensitivity Analysis

```bash
# Malware case study
bash experiments/run_experiments.sh --env malware --experiment applications

# Sensitivity analysis
bash experiments/run_experiments.sh --env hopper --experiment sensitivity
```

**Expected Results (Figures 6-9, Table 7):**
- p=0.25-0.5 robust across environments
- λ=0.01 best for exploration bonus
- α insensitive (0.0001 works well)
- Full RICE yields 68.2% malware evasion with diverse action distribution

## Configuration

All hyperparameters are managed through YAML configuration files:

- `config/default.yaml`: Base defaults for all environments
- `config/env_specific/*.yaml`: Per-environment overrides

Key hyperparameters (from paper Tables 3-4):

| Parameter | Description | Default | MuJoCo | Other Envs |
|-----------|-------------|---------|--------|------------|
| α | Mask intrinsic reward coefficient | 0.0001 | 0.0001 | 0.0001 |
| p | Critical state reset probability | 0.5 | 0.25-0.5 | 0.5 |
| λ | RND exploration bonus coefficient | 0.01 | 0.001-0.01 | 0.01 |
| Mask training samples | — | — | 3×10⁵ | varies |
| Agent network | Hidden layers | [64,64] | [64,64] | [128,128,128,128] |
| RND embedding dim | — | 128 | 128 | 128 |

Override via command line:
```bash
python experiments/refine.py --env Hopper-v3 ... --p 0.25 --lambda-rnd 0.01
```

## Baselines

The following baseline methods are implemented for comparison:

| Baseline | Description | Entry Point |
|----------|-------------|-------------|
| **StateMask** | Original mask network with Lagrangian optimization | `rice-statemask` |
| **Jump-Start RL (JSRL)** | Guide policy curriculum for exploration | `rice-jsrl` |
| **Self-Imitation Learning (SIL)** | Prioritize past successful experiences | `rice-sil` |
| **Random Explanation** | Random critical state selection (ablation) | `rice-random-explanation` |
| **PPO Fine-tuning** | Continue training without resetting or bonus | Built into refine |

Usage:
```bash
# Run JSRL baseline
rice-jsrl --env Hopper-v3 --agent-path ./outputs/hopper/final_model.zip --output-dir ./outputs/hopper/jsrl

# Run SIL baseline
rice-sil --env Hopper-v3 --agent-path ./outputs/hopper/final_model.zip --output-dir ./outputs/hopper/sil
```

## Custom Environments

### Selfish Mining (`SelfishMining-v0`)
Blockchain mining MDP where an attacker strategically withholds blocks.
- **State**: [honest_chain_len, private_chain_len, fork_state, gamma]
- **Actions**: Adopt (0), Reveal (1), Mine (2)
- **Reward**: Positive for accepted blocks, negative for orphaned

### CAGE Challenge 2 (`Cage-v0`)
Network defense scenario: blue agent defends against red "B-line" attacker.
- **State**: Host statuses, services, cooldowns
- **Actions**: Monitor (0), Analyze (1), Decoy (2), Remove (3), Restore (4)
- **Reward**: Negative when red gains admin access

### Autonomous Driving (`AutoDriving-v0`)
MetaDrive "Macro-v1" environment for autonomous driving.
- **State**: BEV + sensor info (flattened vector)
- **Actions**: [steering, acceleration/brake] ∈ [-1,1]²
- **Reward**: Forward motion, speed maintenance, collision penalties

### Malware Mutation (`Malware-v0`)
Adversarial malware modification to evade MalConv classifier.
- **State**: Feature vector (dim 256)
- **Actions**: 16 mutation operations
- **Reward**: 10 for evasion, otherwise score difference

## API Reference

### Core RICE Pipeline

```python
from rice.utils import load_config, set_seed
from rice.mask_network import train_mask_network
from rice.explanation import extract_critical_states
from rice.refining import refine_agent

# Load config
config = load_config("hopper")

# Train mask network
mask_model, mask_logger, mask_path = train_mask_network(
    env_id="Hopper-v3",
    agent_policy=agent,
    config=config,
    output_dir="./outputs/hopper/mask"
)

# Extract critical states
extractor, critical_states = extract_critical_states(
    mask_network=mask_model,
    agent_policy=agent,
    env_id="Hopper-v3",
    config=config,
    output_dir="./outputs/hopper/explanation"
)

# Refine agent
refined_model, refine_logger, refine_path = refine_agent(
    env_id="Hopper-v3",
    agent_path="./outputs/hopper/final_model.zip",
    critical_states_path="./outputs/hopper/explanation/critical_states.pkl",
    config=config,
    output_dir="./outputs/hopper/refined"
)
```

### Key Modules

- **`rice.mask_network`**: `train_mask_network()`, `compute_importance()`, `MaskPolicyNetwork`
- **`rice.explanation`**: `ExplanationExtractor`, `extract_critical_states()`, `compute_fidelity_score()`
- **`rice.refining`**: `refine_agent()`, `MixedInitTrainer`, `RefiningEnvWrapper`
- **`rice.rnd`**: `RNDModule`, `create_rnd_module()`, `RunningMeanStd`
- **`rice.perturbed_env`**: `PerturbedEnv`, `PerturbedEnvWrapper`
- **`rice.utils`**: `load_config()`, `CriticalStateBuffer`, `Logger`, `evaluate_policy()`

## Hardware Requirements

The paper used 8 NVIDIA A100 GPUs. For reproduction:

- **GPU**: Any CUDA-capable GPU with >= 16GB VRAM (A100, V100, A40, RTX 3090/4090)
- **CPU**: 16+ cores recommended for vectorized environments
- **RAM**: 32GB+ recommended
- **Storage**: ~50GB for models and logs across all experiments

CPU-only execution is possible but significantly slower (10-50x).

## Troubleshooting

### Common Issues

1. **MuJoCo rendering errors**: Set `render_mode=None` or use headless rendering.
   ```bash
   export MUJOCO_GL=osmesa  # or egl, glfw
   ```

2. **Out of memory**: Reduce `n_envs`, `batch_size`, or use gradient accumulation.

3. **MetaDrive not found**: Install with `pip install metadrive>=0.4.0`.

4. **State setting not supported**: Some environments don't support `set_state`. The code falls back to storing observations and using `reset` with controlled seeds.

### Verification

To verify your installation is correct:

```bash
# Run a quick test on a simple environment
python -c "
from rice.utils import make_env, evaluate_policy
from stable_baselines3 import PPO

env = make_env('Hopper-v3', seed=0)
model = PPO('MlpPolicy', env, verbose=0)
model.learn(total_timesteps=10000)
results = evaluate_policy(env, model, n_episodes=5)
print(f'Mean reward: {results[\"mean_reward\"]:.2f}')
"
```

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{rice2024,
  title={RICE: Refining via Critical State Explanation},
  author={...},
  journal={...},
  year={2024}
}
```

## License

MIT License. See `setup.py` for details.

## Acknowledgments

This implementation builds upon:
- Stable-Baselines3 (https://github.com/DLR-RM/stable-baselines3)
- StateMask (for the mask network concept)
- Random Network Distillation (Burda et al., 2019)
- Jump-Start RL (Uchendu et al., 2023)
- Self-Imitation Learning (Oh et al., 2018)