# Functional Reward Encodings for Zero-Shot Offline Reinforcement Learning

This repository reproduces the method, baselines, experiments, and evaluation
protocols described in **"Functional Reward Encodings for Zero-Shot Offline
Reinforcement Learning"**. It implements:

1. A random reward-function prior over goal-reaching, sparse linear, and random
   MLP reward functions.
2. A permutation-invariant transformer variational autoencoder (**FRE VAE**)
   that encodes state–reward pairs into a latent reward-function code `z`.
3. A frozen-FRE-conditioned **IQL** offline RL agent that uses the latent reward
   code for zero-shot policy conditioning.
4. Zero-shot evaluation on **AntMaze-large-diverse-v2**, **ExORL
   walker/cheetah**, and **D4RL Franka Kitchen**.
5. Reproducible implementations of the comparison baselines **FB**, **SF**,
   **GC-IQL**, **GC-BC**, and **OPAL**.
6. Ablations and visualization scripts from the paper, including prior-mixture
   scaling and domain-knowledge reward-prior augmentation.

---

## Repository layout

```
fre-reproduction/
├── fre/
│   ├── reward_prior.py          # random reward function families
│   ├── reward_embedding.py      # reward discretization and learned token embeddings
│   ├── encoder.py               # permutation-invariant transformer VAE encoder
│   ├── decoder.py               # reward decoder MLP
│   ├── fre_vae.py               # FRE VAE model and training helpers
│   ├── iql.py                   # IQL Q/V/policy networks and losses
│   ├── agent.py                 # FRE-conditioned IQL agent
│   ├── dataset.py               # offline dataset wrappers and sampling
│   ├── config.py                # dataclasses and hyperparameter defaults
│   └── utils.py                 # normalization, logging, seed helpers
├── baselines/
│   ├── fb.py                    # Forward-Backward baseline
│   ├── sf.py                    # Successor Features baseline
│   ├── gc_iql.py                # Goal-Conditioned IQL baseline
│   ├── gc_bc.py                 # Goal-Conditioned Behavioral Cloning baseline
│   ├── opal.py                  # OPAL unsupervised skill-discovery baseline
│   └── baseline_utils.py        # shared baseline building blocks
├── envs/
│   ├── antmaze_wrapper.py       # AntMaze-large-diverse-v2 tasks and evaluation
│   ├── exorl_wrapper.py         # ExORL walker/cheetah tasks and evaluation
│   └── kitchen_wrapper.py       # D4RL Kitchen subtask rewards and evaluation
├── scripts/
│   ├── train_fre_encoder.py     # Phase 1: train FRE encoder/decoder only
│   ├── train_rl.py              # Phase 2: train IQL with frozen FRE
│   ├── train_baselines.py       # train FB, SF, GC-IQL, GC-BC, OPAL
│   ├── eval_zero_shot.py        # FRE zero-shot evaluation from 32 examples
│   ├── eval_baselines.py        # consistent baseline evaluation
│   ├── ablation_prior_mixture.py
│   ├── domain_prior_augmentation.py
│   ├── visualize_antmaze.py
│   └── run_experiments.py       # full benchmark sweep launcher
├── configs/
│   ├── antmaze.yaml
│   ├── exorl.yaml
│   └── kitchen.yaml
├── tests/
│   ├── test_reward_priors.py
│   ├── test_encoder_invariance.py
│   ├── test_decoder_reconstruction.py
│   ├── test_iql_losses.py
│   ├── test_eval_tasks.py
│   └── test_baselines.py
├── requirements.txt
└── README.md
```

---

## Installation

A Python 3.8 or 3.9 environment is recommended.

### 1. Python dependencies

```bash
cd fre-reproduction
pip install -r requirements.txt
```

The core dependencies are PyTorch, NumPy, Gym, D4RL, and optional MuJoCo/ExORL
bindings. If you plan to run only CPU unit tests and small smoke runs, MuJoCo is
not required.

### 2. MuJoCo and D4RL

The exact MuJoCo installation depends on your platform. A typical setup is:

```bash
pip install mujoco
pip install git+https://github.com/Farama-Foundation/d4rl@master
```

AntMaze and Kitchen are loaded through D4RL. For visualization and online
evaluation, make sure the MuJoCo binary used by D4RL matches your installed
`mujoco` package.

### 3. ExORL

ExORL datasets are optional for the walker/cheetah experiments. Clone the ExORL
repository and add it to `PYTHONPATH`:

```bash
git clone https://github.com/denisyarats/exorl
export PYTHONPATH="$PWD/exorl:$PYTHONPATH"
```

The ExORL loader in `fre/dataset.py` can read HDF5 exploratory datasets
containing `observations`, `actions`, `rewards`, and either `next_observations`
or shifted observations.

---

## Core architecture and defaults

### Reward prior

`fre/reward_prior.py` samples uniformly from three reward-function families:

- **Singleton goal-reaching rewards**: `0` when a state is within `epsilon` of
  a goal sampled from the offline state pool, otherwise `-1`.
- **Random sparse linear rewards**: `states @ (weights * mask)`, with
  `weights ~ Uniform(-1, 1)` and Bernoulli masking probability `0.75`.
- **Random MLP rewards**: two hidden layers of width 256, ReLU activation,
  scalar output, random PyTorch initialization.

All sampled rewards are clipped to `[-1, 1]`.

### FRE VAE

The FRE encoder receives a set of `(state, scalar reward)` pairs. Rewards are
discretized into 64 learned bins, embedded into 64 dimensions, concatenated with
a 192-dimensional state projection, and mapped to a 256-dimensional transformer
token. The transformer encoder has:

- `d_model = 256`
- `nhead = 4`
- `num_layers = 4`
- GELU activation
- no positional encoding and no causal mask

Averaging over final token representations produces a permutation-invariant
summary, which is projected to Gaussian posterior parameters `mu` and `logvar`.
The latent dimension is `128`.

The decoder is a 256-256 MLP from `[state, z]` to a scalar reward. The VAE loss
is

```
L = mean(MSE(decoded_reward, true_reward)) + beta * mean(KL)
```

with `beta = 1.0`.

### FRE-conditioned IQL

The RL agent uses the frozen FRE VAE to encode each sampled reward function.
IQL networks are all 256-256 MLPs:

- Q networks: input `[state, action, z]`
- V network: input `[state, z]`
- policy: input `[state, z]`, Gaussian with tanh-squashed actions

IQL hyperparameters:

- `gamma = 0.99`
- expectile `tau = 0.9`
- advantage-weighted regression temperature `beta = 3.0`
- advantage clipping `[-5.0, 2.0]`
- Polyak averaging `tau = 0.005`
- Adam learning rate `3e-4`

---

## Training and evaluation

### Phase 1: FRE encoder/decoder

Train the reward encoder VAE without RL:

```bash
python scripts/train_fre_encoder.py \
  --domain antmaze \
  --dataset_name antmaze-large-diverse-v2 \
  --vae_steps 100000 \
  --encoder_states 32 \
  --decoder_states 256 \
  --latent_dim 128 \
  --device auto \
  --output_dir checkpoints/antmaze/encoder
```

The script samples reward functions from the prior, samples encoder and decoder
state sets from the offline state pool, and minimizes reconstruction plus KL
loss. Checkpoints contain the model state, optimizer state, and architecture
metadata.

### Phase 2: FRE-conditioned IQL

Train IQL while keeping the FRE encoder frozen:

```bash
python scripts/train_rl.py \
  --domain antmaze \
  --dataset_name antmaze-large-diverse-v2 \
  --vae_checkpoint checkpoints/antmaze/encoder/encoder_final.pt \
  --rl_steps 1000000 \
  --batch_size 256 \
  --encoder_states 32 \
  --device auto \
  --output_dir checkpoints/antmaze/rl
```

The agent samples random reward functions, encodes them with the frozen FRE VAE,
computes the corresponding batch rewards, and updates Q/V/policy networks using
IQL losses.

### Zero-shot evaluation

Evaluate a trained FRE agent using exactly 32 state–reward examples:

```bash
python scripts/eval_zero_shot.py \
  --domain antmaze \
  --dataset_name antmaze-large-diverse-v2 \
  --checkpoint checkpoints/antmaze/rl/agent_final.pt \
  --num_examples 32 \
  --num_episodes 20 \
  --seeds 0 1 2 3 4 \
  --device auto \
  --output_dir eval_results/antmaze
```

Results are saved as JSON with per-seed and aggregate mean/standard-deviation
scores. The default policy uses the posterior mean latent code; use
`--stochastic` to sample from the encoder posterior.

### Baselines

Train a baseline:

```bash
python scripts/train_baselines.py \
  --baseline fb \
  --domain antmaze \
  --dataset_name antmaze-large-diverse-v2 \
  --steps 1000000 \
  --device auto \
  --output_dir checkpoints/baselines
```

Available baseline names are `fb`, `sf`, `gc_iql`, `gc_bc`, and `opal`.

Evaluate baselines:

```bash
python scripts/eval_baselines.py \
  --baseline fb \
  --domain antmaze \
  --dataset_name antmaze-large-diverse-v2 \
  --checkpoint checkpoints/baselines/fb_antmaze_seed0_final.pt \
  --num_reward_samples 5120 \
  --num_episodes 20 \
  --seeds 0 1 2 3 4 \
  --device auto \
  --output_dir eval_results/baselines
```

Evaluation protocols match the paper:

- **FB** and **SF** infer task vectors from 5120 reward samples using ridge
  regression.
- **GC-IQL** and **GC-BC** are conditioned directly on the goal selected from
  sampled task states.
- **OPAL** uses privileged evaluation: 10 sampled skills are evaluated online,
  and the best downstream score is reported.

### Full benchmark sweep

To run the complete multi-domain, multi-seed reproduction:

```bash
python scripts/run_experiments.py \
  --domains antmaze kitchen exorl-walker exorl-cheetah \
  --methods fre fb sf gc_iql gc_bc opal \
  --seeds 0 1 2 3 4 \
  --output_root results \
  --skip_existing
```

Use `--dry_run` to inspect the commands that would be executed. A
`sweep_summary.json` is written to the output root.

---

## Domain-specific tasks

### AntMaze-large-diverse-v2

Implemented tasks:

- `ant-goal-reaching`
- `ant-directional`
- `ant-random-simplex`
- `ant-path-loop`
- `ant-path-edges`
- `ant-path-center`

AntMaze observations are D4RL qpos+velocity states with the first two qpos
entries treated as XY position.

### ExORL walker/cheetah

Implemented tasks:

- `walker-goals`
- `cheetah-goals`
- `walker-velocity`
- `cheetah-velocity`

Goal rewards are sparse `{0, -1}`. Velocity rewards use a clipped normalized
velocity.

### Kitchen

Seven D4RL subtasks are evaluated:

- `microwave`
- `kettle`
- `light_switch`
- `slide_cabinet`
- `hinge_cabinet`
- `bottom_burner`
- `top_burner`

Rewards are sparse subtask-completion rewards extracted from flattened
observations.

---

## Ablations and visualizations

### Prior-mixture ablation (Figure 5)

Train FRE variants restricted to subsets of the reward prior:

```bash
python scripts/ablation_prior_mixture.py \
  --domain antmaze \
  --dataset_name antmaze-large-diverse-v2 \
  --mixtures goals linear mlp all \
  --vae_steps 50000 \
  --rl_steps 300000 \
  --device auto \
  --output_dir ablation_results
```

### Domain-knowledge reward-prior augmentation (Figure 6)

Add XY-position rewards on AntMaze or velocity rewards on ExORL to the base
prior:

```bash
python scripts/domain_prior_augmentation.py \
  --domain antmaze \
  --dataset_name antmaze-large-diverse-v2 \
  --variants base xy_augmented \
  --device auto \
  --output_dir augmentation_results
```

For ExORL:

```bash
python scripts/domain_prior_augmentation.py \
  --domain exorl-walker \
  --variants base velocity_augmented \
  --device auto \
  --output_dir augmentation_results
```

### AntMaze visualization (Figure 3)

```bash
python scripts/visualize_antmaze.py \
  --checkpoint checkpoints/antmaze/rl/agent_final.pt \
  --domain antmaze \
  --dataset_name antmaze-large-diverse-v2 \
  --env_name antmaze-large-diverse-v2 \
  --task ant-goal-reaching \
  --num_examples 32 \
  --grid_resolution 200 \
  --output_dir figures
```

The script renders true reward, encoding states, decoded reward, policy rollout
positions, and predicted value function panels.

---

## Unit tests

Run the full CPU-friendly test suite:

```bash
pytest tests -q
```

Individual test groups:

```bash
pytest tests/test_reward_priors.py
pytest tests/test_encoder_invariance.py
pytest tests/test_decoder_reconstruction.py
pytest tests/test_iql_losses.py
pytest tests/test_eval_tasks.py
pytest tests/test_baselines.py
```

The tests verify:

- reward-family semantics and clipping
- transformer encoder permutation invariance
- FRE VAE reconstruction and KL boundedness
- IQL loss behavior and policy action bounds
- downstream task reward definitions and 32-example sampling
- baseline training-loop stability and action validity

---

## Configuration files

YAML configurations in `configs/` mirror the dataclass defaults in
`fre/config.py`:

- `configs/antmaze.yaml`
- `configs/exorl.yaml`
- `configs/kitchen.yaml`

They can be used with OmegaConf or Hydra-based launchers and provide
paper-aligned defaults for each benchmark.

---

## Expected experimental setup

The reproduction plan recommends the following phase-2 gradient budgets:

| Domain  | RL gradient steps |
|---------|-------------------|
| AntMaze | 1,000,000         |
| ExORL   | 500,000           |
| Kitchen | 1,000,000         |

FRE phase-1 encoder training continues until reconstruction converges across
the three reward families. Evaluation uses 20 episodes per task and aggregates
over 5 random seeds.

Approximate paper targets from Table 1 include:

- AntMaze all tasks: `52.8 ± 18.2`
- ExORL all tasks: `51.5 ± 6.3`
- Kitchen: `66 ± 3`

Reproduction is considered successful when scores are within approximately one
standard deviation of the reported results or confidently above random/naive
baselines on most tasks.

---

## Handling missing details

Where the paper leaves implementation details unspecified, this repository uses
the following paper-informed defaults:

- `K = 32` encoder states
- `K' = 256` decoder states
- `d_model = 256`
- `latent_dim = 128`
- reward bins `B = 64`
- IQL expectile `tau = 0.9`
- AWR temperature `beta = 3.0`
- discount `gamma = 0.99`
- Adam learning rate `3e-4` for IQL and `1e-4` for FRE VAE

If training is unstable, the implementation supports searching over encoder
layers `{3, 4}`, latent dimensions `{64, 128}`, reward clipping, and expectile
values `{0.7, 0.9}`.
