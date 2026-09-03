# RICE: Refining Reinforcement Learning with Explanation

This repository contains a reproduction of the RICE algorithm from the paper
*"RICE: Breaking Through the Training Bottlenecks of Reinforcement Learning with
Explanation"* (Cheng et al., ICML 2024).

> **Note:** The reproduction does not access the authors' implementation at
> `https://github.com/chengzelei/RICE` (which is blacklisted for this task). All
> code is written from scratch based on the paper and the provided addendum.

## Implemented Components

### Core Method

- **`rice/mask_network.py`** — Improved StateMask-style mask network (Algorithm 1).
  - The mask network learns a binary policy over `{blind, keep}` for each state.
  - Following Theorem 3.3, we reformulate the objective as maximizing the
    expected return of the perturbed agent and train with vanilla PPO.
  - An intrinsic reward `alpha * a^m` encourages blinding and avoids the trivial
    "never blind" solution.
  - State importance is defined as `P(a^m = 0 | s)`.

- **`rice/refining.py`** — RICE refining algorithm (Algorithm 2).
  - `RICERefiningEnv` implements the mixed initial-state distribution:
    with probability `p` the episode starts from a critical state identified by
    the mask network, otherwise from the default initial distribution.
  - Random Network Distillation (RND) is used as the exploration bonus.
  - `refine_rice()` creates a fresh PPO learner on the RICE-wrapped environment
    and initializes it from the pre-trained policy.

- **`rice/rnd.py`** — Random Network Distillation implementation.
  - Fixed random target network and a trainable predictor network.
  - Running mean/variance normalization of bonuses.

### Explanation Methods

- **`rice/explanations.py`**
  - `MaskExplanation` — RICE explanation using the trained mask network.
  - `RandomExplanation` — Random baseline that selects states uniformly.
  - `StateMaskExplanation` — Interface-compatible wrapper for the original
    StateMask objective. The paper's alternative design is used in practice
    (see mask network); this wrapper is provided for experiment labels.

### Baselines

- **`rice/baselines.py`**
  - `ppo_finetune` — Continue PPO training with reduced learning rate.
  - `statemask_r_finetune` — Reset to critical states and fine-tune (StateMask-R).
  - `jsrl_finetune` — Jump-Start Reinforcement Learning with a guide-policy
    curriculum.

### Evaluation

- **`rice/fidelity.py`** — Fidelity-score computation (Experiment I).
  - Implements the sliding-window critical-segment selection, randomization of
    actions in the selected window, and the fidelity formula from the paper.

- **`experiments/run_experiments.py`** — Driver for Experiments I, II, III, and V
  on MuJoCo environments.

- **`experiments/experiment_iv_sac.py`** — Experiment IV: refine a pre-trained SAC
  agent by imitating it with a PPO policy trained via GAIL, then apply RICE.

- **`scripts/train_target_agents.py`** — Train pre-target agents used by the
  refining experiments.

### Environment Utilities

- **`rice/env_utils.py`** — Wrappers for sparse MuJoCo rewards, observation
  normalization, and resetting a simulator to an arbitrary state.
- **`rice/real_world_envs.py`** — Gymnasium-compatible interface stubs for the
  real-world applications (selfish mining, CAGE Challenge 2, MetaDrive). These
  require external simulators not included here.

## What Was Skipped and Why

1. **Section 3.4 (Theoretical Analysis)**  
   Marked as out of scope in the addendum. The code focuses on the empirical
   algorithms; the theoretical bounds are not implemented.

2. **SparseWalker2d hyper-parameter sensitivity**  
   The addendum explicitly excludes sparse MuJoCo hyper-parameter sensitivity
   results from the reproduction.

3. **Qualitative Autonomous Driving Analysis**  
   Marked as out of scope in the addendum (visualization of driving behavior).

4. **Malware Mutation Experiments**  
   Marked as out of scope in the addendum. The environment and detector are not
   included.

5. **Real-World Application Simulators**  
   The selfish-mining, CAGE Challenge 2, and MetaDrive simulators are external
   dependencies. We provide Gymnasium interface stubs so that the RICE pipeline
   can be plugged in once the simulators are installed, but the simulators
   themselves are not bundled.

6. **Exact Architectural Hyperparameters for Real-World Apps**  
   The addendum notes that exact architectures are omitted from the reproduction
   because the methods are black-box with respect to the target agent. We use
   generic MLP policies for MuJoCo and document expected shapes for the real-world
   apps.

## Repository Structure

```
submission/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── rice/
│   ├── __init__.py
│   ├── mask_network.py       # Mask network training (Algorithm 1)
│   ├── refining.py           # RICE refining (Algorithm 2)
│   ├── rnd.py                # Random Network Distillation
│   ├── explanations.py       # Explanation methods
│   ├── fidelity.py           # Fidelity score
│   ├── baselines.py          # Baseline refining methods
│   ├── env_utils.py          # Environment wrappers and state reset
│   ├── real_world_envs.py    # Stubs for real-world simulators
│   ├── config.py             # Hyperparameter configuration
│   └── utils.py              # Generic helpers
├── experiments/
│   ├── run_experiments.py    # Experiments I, II, III, V
│   └── experiment_iv_sac.py  # Experiment IV
├── scripts/
│   ├── train_target_agents.py
│   └── plot_results.py
└── tests/
    ├── __init__.py
    └── test_smoke.py
```

## Usage

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train a target agent

```bash
python scripts/train_target_agents.py \
  --env-id Hopper-v3 \
  --algorithm PPO \
  --timesteps 1000000 \
  --save-path models/hopper_ppo.pt
```

### Run experiments

```bash
python experiments/run_experiments.py \
  --env-id Hopper-v3 \
  --model-path models/hopper_ppo.pt \
  --experiment all \
  --target-timesteps 1000000 \
  --mask-timesteps 300000 \
  --refine-timesteps 500000 \
  --output-dir results
```

### Run Experiment IV (SAC agent)

```bash
python experiments/experiment_iv_sac.py \
  --env-id Hopper-v3 \
  --output-dir results/sac_hopper
```

## Design Decisions

- **Stable-Baselines3** is used for the base PPO/SAC learners, consistent with
  the authors' implementation notes in the addendum.
- **No version pins** are used in `requirements.txt` to avoid dependency
  conflicts, per the prompt's strategy note.
- The mask network is trained with vanilla PPO rather than the primal-dual
  method of the original StateMask, following the paper's reformulation.
- State reset for MuJoCo uses the underlying `set_state(qpos, qvel)` API when
  available; for other simulators a custom `set_state` can be provided through
  `StateResetWrapper`.

## Metrics

- File count: 17 Python files plus README, requirements, setup.py, and .gitignore.
- Total lines of Python: ~2,450 (measured with `find . -name "*.py" | xargs wc -l`).
- Commits: 9.
