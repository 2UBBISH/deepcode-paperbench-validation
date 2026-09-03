# Functional Reward Encodings for Zero-Shot Offline Reinforcement Learning

This repository reproduces **Functional Reward Encodings (FRE)**, a method for
zero-shot offline reinforcement learning. FRE learns a permutation-invariant
transformer variational autoencoder over reward functions, encodes a downstream
task from a small set of `(state, reward)` examples, and conditions an offline
Implicit Q-Learning (IQL) agent on the resulting latent vector.

The implementation covers:

- The FRE reward-prior distribution (singleton goal-reaching, random linear, and
  random MLP reward functions).
- The permutation-invariant transformer VAE reward encoder/decoder.
- FRE-conditioned offline IQL training with a frozen reward encoder.
- Zero-shot downstream evaluation on AntMaze, ExORL (Walker/Cheetah), and Kitchen.
- Baselines: GC-IQL, GC-BC, OPAL, Forward-Backward (FB), and Successor Features (SF).
- Visualization and ablation utilities.

---

## Repository layout

```
fre/
  __init__.py
  config.py                     # configuration dataclasses and YAML helpers
  main.py                       # top-level CLI dispatcher
  data/
    dataset.py                  # offline dataset wrapper and sampling utilities
    reward_sampler.py           # prior reward-function mixture
    d4rl_loader.py              # AntMaze and Kitchen loading
    exorl_loader.py             # ExORL Walker/Cheetah loading
  modeling/
    reward_embedding.py         # reward discretization and learned embedding
    transformer_encoder.py      # permutation-invariant transformer encoder
    fre_vae.py                  # FRE variational autoencoder
    decoder.py                  # reward decoder
  rl/
    networks.py                 # V, Q, policy MLPs conditioned on z or goal
    iql.py                      # implicit Q-learning losses and update code
    rl_trainer.py               # FRE-conditioned offline RL trainer
    gc_iql.py                   # goal-conditioned IQL baseline
    gc_bc.py                    # goal-conditioned behavioral cloning baseline
  baselines/
    fb.py                       # Forward-Backward baseline
    sf.py                       # Successor Features baseline
    opal.py                     # OPAL skill-discovery baseline
    baseline_eval.py            # baseline evaluation facade
  pipeline/
    pretrain_encoder.py         # Phase 1 FRE VAE pretraining
    train_agent.py              # Phase 2 strided IQL training
    evaluate.py                 # FRE zero-shot evaluation
    evaluate_baselines.py       # baseline evaluation utilities
    visualize.py                # reward/value/policy visualizations
  envs/
    antmaze.py
    kitchen.py
    dmc.py
  utils/
    metrics.py
    seeds.py
configs/
  antmaze.yaml
  exorl.yaml
  kitchen.yaml
scripts/
  run_pretrain.sh
  run_train.sh
  run_eval.sh
  run_baselines.sh
requirements.txt
README.md
```

---

## Installation

Python 3.8 or 3.9 is recommended.

```bash
pip install -r requirements.txt
```

MuJoCo-related dependencies are environment-specific and are intentionally
listed separately/commented in `requirements.txt` because they may require
platform binaries. For D4RL datasets:

```bash
pip install d4rl>=1.1
```

For ExORL evaluation, install the DeepMind Control Suite and dmc2gym:

```bash
pip install dm-control>=1.0.0 dmc2gym
```

Set dataset directories before running:

```bash
export D4RL_DATA_PATH=/path/to/d4rl
export EXORL_DATA_PATH=/path/to/exorl
```

By default the code also checks `D4RL_DATASET_DIR` and local
`./exorl_data` / `./d4rl_data` fallbacks.

---

## Quick start

The scripts accept an optional YAML config path and forward extra arguments
after `--`.

### 1. Pretrain the FRE reward encoder

```bash
bash scripts/run_pretrain.sh configs/antmaze.yaml
```

Equivalent Python call:

```bash
python -m fre.pipeline.pretrain_encoder --config configs/antmaze.yaml
```

### 2. Train the FRE-conditioned IQL agent

```bash
bash scripts/run_train.sh configs/antmaze.yaml
```

The training phase freezes the pretrained FRE encoder so the latent task vector
`z` remains stationary during TD learning.

### 3. Evaluate FRE zero-shot

```bash
bash scripts/run_eval.sh configs/antmaze.yaml
```

This encodes each downstream task with 32 reward examples and rolls out 20
episodes per task.

### 4. Evaluate baselines

```bash
bash scripts/run_baselines.sh configs/antmaze.yaml
```

Baseline evaluation uses 20 episodes and averages over 5 seeds. FB and SF use
5120 reward samples for test-time reward regression; OPAL uses privileged
online skill selection with 10 candidate skills.

### 5. Visualizations

```bash
python -m fre.pipeline.visualize --config configs/antmaze.yaml antmaze --task ant-goal-reaching
```

Or, alternatively, use the top-level dispatcher:

```bash
python -m fre.main visualize --config configs/antmaze.yaml
```

---

## Configuration

Config files are YAML and are loaded through `fre.config.get_config`.
Important sections:

| Section | Description |
| --- | --- |
| `data` | environment/dataset names, paths, normalization, batch size |
| `reward_sampler` | prior reward families, encoder/decoder state counts, reward range |
| `fre` | transformer VAE architecture, latent dimension, pretraining optimizer |
| `iql` | IQL hyperparameters and FRE-conditioned RL training settings |
| `baseline` | GC-IQL, GC-BC, OPAL, FB, SF hyperparameters |
| `eval` | evaluation episodes, reward samples, seeds, normalization |

Flat overrides can be supplied on the CLI using `section__field` syntax,
for example `--override iql__tau=0.9`.

---

## Default hyperparameters for missing details

Where the original paper leaves a detail implicit, the following defaults are
used throughout this reproduction:

| Parameter | Default |
| --- | --- |
| Encoder context size `K` | 32 |
| Decoder context size `K'` | 1024 |
| Reward embedding bins `M` | 128 |
| Latent dimension `d_z` | 64 |
| Transformer layers | 2 |
| Transformer heads | 4 |
| Transformer model dimension `d_model` | 128 |
| Transformer feedforward dimension `d_ff` | 256 |
| Encoder learning rate | 1e-4 |
| RL learning rate | 3e-4 |
| IQL expectile `tau` | 0.7 |
| IQL temperature `beta` | 3.0 |
| Discount `gamma` | 0.99 |
| Target soft-update `tau` | 0.005 |

These defaults are also recorded in the YAML files under `configs/`.

---

## Tasks

### AntMaze large-diverse-v2

- ant-goal-reaching
- ant-directional
- ant-random-simplex
- ant-path-loop
- ant-path-edges
- ant-path-center

### ExORL

- Walker goal-reaching
- Cheetah goal-reaching
- Walker forward/backward velocity
- Cheetah forward/backward velocity

### Kitchen

Seven standard subtask reward functions:
- microwave
- kettle
- light switch
- slide cabinet
- hinge cabinet
- bottom burner
- top burner

---

## Expected results

Normalized scores are reported in the paper on a 0–100 scale. The implementation
includes utilities in `fre/utils/metrics.py` for this normalization and for
aggregating `mean ± std` over seeds.

Reference FRE values:

| Task | Expected |
| --- | --- |
| ant-goal-reaching | 48.8 ± 6 |
| ant-directional | 55.2 ± 8 |
| ant-random-simplex | 21.3 ± 4 |
| ant-path-loop | 67.2 ± 36 |
| ant-path-edges | 60.0 ± 17 |
| ant-path-center | 64.4 ± 38 |
| antmaze-all | 52.8 ± 18.2 |
| exorl-walker-goals | 94 ± 2 |
| exorl-cheetah-goals | 58 ± 8 |
| exorl-walker-velocity | 34 ± 13 |
| exorl-cheetah-velocity | 20 ± 2 |
| exorl-all | 51.5 ± 6.3 |
| kitchen | 66 ± 3 |
| all | 57 ± 9 |

Baseline scores from Table 1 are also expected to be broadly consistent with
the paper. Exact values may vary with environment versions, dataset versions,
and random seeds.

---

## Sanity checks

Before running full experiments, verify:

1. **FRE reconstruction**: Phase 1 reconstruction MSE decreases and KL remains
   bounded.
2. **Permutation invariance**: permuting the encoder context order should not
   change the encoded latent `z`.
3. **Single-task IQL**: plain IQL without `z` should reach reasonable offline RL
   performance on a single D4RL or goal-reaching task.
4. **Frozen encoder stability**: after freezing FRE, Q/V/policy losses should
   remain stable during Phase 2.

---

## Ablations

### Reward-prior scaling

The reward sampler supports enabling or disabling individual reward families
through `reward_sampler.singleton_enabled`, `reward_sampler.linear_enabled`, and
`reward_sampler.mlp_enabled`. Train separate agents for singleton-only,
linear-only, MLP-only, singleton+linear, singleton+MLP, linear+MLP, and all
three, then evaluate all downstream tasks. The full uniform mixture should be
the most broadly competitive.

### Domain-prior augmentation

Domain-specific random functions can be added to the prior without changing the
architecture. For AntMaze, use XY-position-only rewards; for ExORL, use
velocity-only rewards. The reward sampler can be extended with these families
and the same training pipeline reused.

---

## Reproducibility

Use seeds `0, 1, 2, 3, 4` for final averaged results. `fre/utils/seeds.py`
provides `set_seed`, `seed_worker`, and `worker_init_fn` for deterministic runs.

---

## Troubleshooting

- **D4RL import errors**: ensure `d4rl` is installed and `D4RL_DATA_PATH` points
  to a valid dataset directory.
- **MuJoCo errors**: install a compatible MuJoCo binary version and ensure
  `LD_LIBRARY_PATH` includes its library path.
- **ExORL loading failures**: provide a D4RL-style HDF5, NPZ, or directory of
  transitions and set `EXORL_DATA_PATH`.
- **GPU memory**: reduce `data.batch_size`, `fre.num_decoder_states`, or
  `eval.num_episodes` if memory is limited.
- **Policy instability**: increase `iql.tau` or decrease `iql.beta`; ensure the
  FRE encoder is frozen before Phase 2.

---

## Citation

This is a reproduction of:

> *Functional Reward Encodings for Zero-Shot Offline Reinforcement Learning*

Please cite the original paper when using this code.
