# RICE: Refining Reinforcement Learning Agents via Critical Explanations

This repository contains a reproduction of the paper **"RICE: Refining Reinforcement Learning Agents via Critical Explanations"**. RICE is a unified framework that (1) trains a lightweight mask network to identify critical decision steps of a pre-trained RL agent, and (2) refines the agent by restarting episodes from a mixed initial-state distribution while adding an RND-based exploration bonus.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Reproducing the Paper's Experiments](#reproducing-the-papers-experiments)
  - [Experiment I: Explanation Fidelity](#experiment-i-explanation-fidelity)
  - [Experiment II: Efficiency of Mask Training](#experiment-ii-efficiency-of-mask-training)
  - [Experiment III: Agent Refining on Dense MuJoCo](#experiment-iii-agent-refining-on-dense-mujoco)
  - [Experiment IV: Sparse-Reward MuJoCo](#experiment-iv-sparse-reward-mujoco)
  - [Experiment V: Case Studies](#experiment-v-case-studies)
- [Configuration](#configuration)
- [Testing](#testing)
- [Expected Results and Known Limitations](#expected-results-and-known-limitations)
- [Citation](#citation)

## Overview

RICE consists of three main stages:

1. **Target-agent training**: Train a pre-trained policy π for each domain.
2. **Mask network training**: Train a lightweight mask network ξ(s) that identifies critical states by learning a perturbed policy π̄ = ξπ + (1-ξ)πʳ with a blinding intrinsic reward.
3. **Agent refining**: Refine π by training from a mixed initial-state distribution (default initial states + sampled critical states) with an RND exploration bonus.

The framework is evaluated on:
- Dense and sparse MuJoCo continuous-control tasks
- Selfish mining blockchain MDP
- CAGE Challenge 2 cyber defense
- MetaDrive autonomous driving
- Malware mutation (MalConv gym)

## Repository Structure

```
rice/
├── rice/
│   ├── agents/           # Target agent, mask network, RND bonus
│   ├── envs/             # Environment adapters and wrappers
│   ├── training/         # Training loops for target, mask, and refining
│   ├── evaluation/       # Fidelity, efficiency, evaluation, visualization
│   └── utils/            # Config, logging, replay buffers
├── scripts/              # Experiment scripts (Exp I-V)
├── configs/              # Domain-specific YAML configurations
├── tests/                # Unit and sanity tests
├── main.py               # Unified CLI entry point
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Installation

### Prerequisites

- Python >= 3.8 (3.9 or 3.10 recommended)
- MuJoCo 2.1+ system libraries
- CUDA drivers (optional, for GPU acceleration)

### Install Core Dependencies

```bash
pip install -r requirements.txt
```

This installs PyTorch, Stable-Baselines3, Gymnasium, Tianshou, and other core utilities.

### Install Domain-Specific Repositories (Optional)

For full reproduction of all domains, install the external simulators from source:

```bash
# Selfish mining
pip install git+https://github.com/psyhtest/pto-selfish-mining.git

# CAGE Challenge 2 / CybORG
pip install git+https://github.com/cage-challenge/cage-challenge-2.git

# MetaDrive + DI-drive
pip install git+https://github.com/metadriverse/metadrive.git
pip install git+https://github.com/decisionintelligence/DI-drive.git

# Malware mutation
pip install git+https://github.com/endgameinc/malware_rl.git
```

> **Note**: These repositories may have their own system dependencies. The RICE codebase uses soft imports, so missing optional domains will not break the core MuJoCo experiments.

## Quick Start

### Train a Target Agent

```bash
python -m rice.main train-target --domain mujoco --env-id Hopper-v3 \
    --save-dir results/targets/hopper --seed 0
```

### Train a Mask Network

```bash
python -m rice.main train-mask --domain mujoco --env-id Hopper-v3 \
    --target-path results/targets/hopper/target_agent.zip \
    --save-dir results/masks/hopper --seed 0
```

### Refine the Agent with RICE

```bash
python -m rice.main refine --domain mujoco --env-id Hopper-v3 \
    --target-path results/targets/hopper/target_agent.zip \
    --mask-path results/masks/hopper/mask_net.pt \
    --save-dir results/refined/hopper --seed 0
```

### Evaluate a Saved Agent

```bash
python -m rice.main eval --domain mujoco --env-id Hopper-v3 \
    --target-path results/refined/hopper/refined_model.zip \
    --n-eval 50 --seed 0
```

## Reproducing the Paper's Experiments

Each experiment has a dedicated script under `scripts/` and a corresponding subcommand in `main.py`.

### Experiment I: Explanation Fidelity

Reproduces Figure 5: compares RICE, StateMask, random masking, Integrated Gradients, and AIRS.

```bash
python -m rice.main exp-i --domain mujoco --env-id Hopper-v3 \
    --target-path results/targets/hopper/target_agent.zip \
    --mask-path results/masks/hopper/mask_net.pt \
    --save-dir results/exp_i/hopper --seed 0
```

Or directly:

```bash
python -m rice.scripts.run_exp_i_fidelity --domain mujoco --env-id Hopper-v3 \
    --target-path results/targets/hopper/target_agent.zip \
    --mask-path results/masks/hopper/mask_net.pt \
    --save-dir results/exp_i/hopper --seed 0
```

### Experiment II: Efficiency of Mask Training

Reproduces Table 4: wall-clock training time of RICE mask vs. StateMask on fixed sample budgets.

```bash
python -m rice.main exp-ii --domain mujoco --env-id Hopper-v3 \
    --target-path results/targets/hopper/target_agent.zip \
    --save-dir results/exp_ii/hopper --seed 0
```

### Experiment III: Agent Refining on Dense MuJoCo

Reproduces Tables 5 and 6: compares RICE with JSRL, SIL, StateMask-R, and vanilla PPO fine-tuning.

```bash
python -m rice.main exp-iii --env-id Hopper-v3 \
    --target-path results/targets/hopper/target_agent.zip \
    --mask-path results/masks/hopper/mask_net.pt \
    --save-dir results/exp_iii/hopper --seed 0
```

### Experiment IV: Sparse-Reward MuJoCo

Reproduces Figures 10-13: refining on sparse Hopper, Walker2d, and HalfCheetah with sensitivity sweeps over `p` and `λ`.

```bash
python -m rice.main exp-iv --env-id Hopper-v3 \
    --target-path results/targets/hopper_sparse/target_agent.zip \
    --mask-path results/masks/hopper_sparse/mask_net.pt \
    --save-dir results/exp_iv/hopper --seed 0
```

Use `--sparse` when training the target and mask for sparse environments:

```bash
python -m rice.main train-target --domain mujoco --env-id Hopper-v3 \
    --sparse --save-dir results/targets/hopper_sparse --seed 0
```

### Experiment V: Case Studies

Reproduces Table 7 (malware), Figure 14 (MetaDrive), and Figure 15 (MountainCar negative control).

```bash
# Malware case study
python -m rice.main exp-v --study malware \
    --target-path results/targets/malware/target_agent.zip \
    --save-dir results/exp_v/malware --seed 0

# MetaDrive case study
python -m rice.main exp-v --study metadrive \
    --target-path results/targets/metadrive/target_agent.zip \
    --save-dir results/exp_v/metadrive --seed 0

# MountainCar negative control
python -m rice.main exp-v --study mountaincar \
    --target-path results/targets/mountaincar/target_agent.zip \
    --save-dir results/exp_v/mountaincar --seed 0
```

### Run the Full Pipeline

```bash
python -m rice.main run-all --domain mujoco --env-id Hopper-v3 \
    --save-dir results/full_pipeline/hopper --seed 0
```

This trains the target agent, mask network, refines the agent, and runs Experiments I, II, and III.

## Configuration

Domain-specific hyper-parameters are stored in `configs/`:

- `configs/mujoco.yaml`
- `configs/selfish_mining.yaml`
- `configs/cage.yaml`
- `configs/metadrive.yaml`
- `configs/malware.yaml`

You can override config values from the command line using dotted notation:

```bash
python -m rice.main refine --domain mujoco --env-id Hopper-v3 \
    --config configs/mujoco.yaml \
    --refine.p 0.5 --refine.lambda_coef 0.01 \
    --target-path results/targets/hopper/target_agent.zip \
    --mask-path results/masks/hopper/mask_net.pt \
    --save-dir results/refined/hopper_p0.5
```

## Testing

Run the unit and sanity tests with pytest:

```bash
pytest rice/tests/ -v
```

Key tests:

- `test_mask.py`: verifies mask network training increases blinding while keeping perturbed-policy return near the target return.
- `test_reset_wrapper.py`: verifies resetting to a stored critical state reproduces the stored observation and simulator state.
- `test_rnd.py`: verifies RND bonus decreases for visited states and remains positive for novel states.

## Expected Results and Known Limitations

### Expected Dense MuJoCo Refining Results (Table 5)

| Environment   | Expected Return | Std Error |
|---------------|----------------:|----------:|
| Hopper-v3     | ~3663.91        | 20.98     |
| Walker2d-v3   | ~3982.79        | 3.15      |
| Reacher-v2    | ~ -2.66         | 0.03      |
| HalfCheetah-v3| ~2138.89        | 3.22      |

### Expected Malware Evasion Rates (Table 7)

| Condition                              | Expected Evasion Rate |
|----------------------------------------|----------------------:|
| Original pre-trained policy            | ~33.8%                |
| Continue training                      | ~38.8%                |
| Critical states only (overfitting)     | ~50.8%                |
| Mixed initial distribution             | ~58.4%                |
| Mixed + RND (full RICE)                | ~68.2%                |
| Full RICE + reward fix (scale ×3)      | ~72.0%                |

### Known Limitations

- **External repository APIs**: Domain-specific adapters (`selfish_mining_env.py`, `cage_env.py`, `metadrive_env.py`, `malware_env.py`) use soft imports and best-effort state capture/restore. API changes in upstream repositories may require minor compatibility updates.
- **Computational cost**: The paper was trained on 8× NVIDIA A100 GPUs. Full reproduction of all domains is computationally expensive; MuJoCo experiments are feasible on a single GPU or CPU.
- **MountainCar negative control**: RICE is not expected to improve performance when the pre-trained policy has almost no state coverage (Figure 15). Performance should be similar to RND-only fine-tuning.
- **StateMask baseline**: The fidelity and refining baselines approximate StateMask with a state-ablation importance ranking. For exact reproduction, install the authors' official `RL-state_mask` repository and adapt the ranking functions in `rice/evaluation/fidelity.py`.
- **Hyper-parameter sensitivity**: The paper reports robustness to many hyper-parameters, but exact numerical matches may require tuning `p`, `λ`, and `α` for your specific hardware and random seeds.

## Citation

If you use this code, please cite the original RICE paper:

```bibtex
@inproceedings{rice2023,
  title={RICE: Refining Reinforcement Learning Agents via Critical Explanations},
  author={RICE Authors},
  booktitle={Conference},
  year={2023}
}
```

---

For questions or issues, please open a GitHub issue or refer to the reproduction plan in the repository documentation.
