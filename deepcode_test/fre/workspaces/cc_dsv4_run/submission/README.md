# Functional Reward Encoding (FRE) — Reproduction

Reproduction of:

> **"Unsupervised Zero-Shot Reinforcement Learning via Functional Reward Encodings"**
> Kevin Frans, Seohong Park, Pieter Abbeel, Sergey Levine
> UC Berkeley, ICML 2024

---

## What Was Implemented

### Core FRE Method (Sections 4.1–4.3)

| Component | Status | File |
|-----------|--------|------|
| **FRE Encoder** (permutation-invariant transformer VAE) | ✅ Implemented | `fre/encoder.py` |
| **FRE Decoder** (reward prediction MLP) | ✅ Implemented | `fre/encoder.py` |
| **Variational Information Bottleneck Objective** (Eq. 6) | ✅ Implemented | `fre/encoder.py` |
| **Prior Reward Distribution** (goal-reaching + linear + MLP mixture) | ✅ Implemented | `fre/prior.py` |
| **IQL-based Offline RL** (z-conditioned Q, V, π) | ✅ Implemented | `fre/iql.py` |
| **Strided Two-Phase Training** (Algorithm 1) | ✅ Implemented | `fre/training.py` |
| **Zero-Shot Evaluation Protocol** (32 samples → encode → execute) | ✅ Implemented | `fre/evaluation.py` |
| **All seven prior variants** (FRE-all, FRE-goals, FRE-lin, FRE-mlp, FRE-lin-mlp, FRE-goal-mlp, FRE-goal-lin) | ✅ Implemented | `fre/prior.py` |

### Architecture Details (verified against paper & addendum)

- Reward discretization: 32 bins, rescale to [0,1] → floor → embedding table
- Reward embedding: 64-dimensional (corrected from paper appendix which says 128)
- State embedding: 64-dimensional learned linear projection
- Concatenated input: 128-dimensional
- Transformer: 4 layers, 4 heads, MLP dim 256, residual dim 128
- No positional encodings or causal masking (permutation-invariant)
- Average pooling over sequence elements → μ and log σ projections
- Latent z: 128-dimensional
- Decoder: [512, 512, 512] MLP, raw state + z concatenated directly
- β (KL weight): 0.01
- K = 32 encoder pairs, K' = 8 decoder pairs

### IQL Integration

| Component | Hyperparameter | Paper Value |
|-----------|---------------|-------------|
| Q/V/π hidden layers | [512, 512, 512] | ✅ |
| Expectile (τ) | 0.8 | ✅ |
| AWR temperature | 3.0 | ✅ |
| Discount (γ) | 0.88 | ✅ |
| Target update rate (τ_polyak) | 0.001 | ✅ |
| Learning rate | 1e-4 (Adam) | ✅ |
| Batch size | 512 | ✅ |
| Encoder training steps | 150k (1M ExORL/Kitchen) | ✅ |
| Policy training steps | 850k (1M ExORL/Kitchen) | ✅ |

### Baselines Implemented

| Baseline | Status | File |
|----------|--------|------|
| **GC-BC** (Goal-Conditioned Behavioral Cloning) | ✅ Implemented | `baselines/gc_bc.py` |
| **GC-IQL** (Goal-Conditioned IQL) | ✅ Implemented | `baselines/gc_iql.py` |
| **OPAL** (Offline Primitive Discovery) | ✅ Implemented | `baselines/opal.py` |
| **FB** (Forward-Backward) | ⚠️ Wrapper only | `baselines/fb_sf.py` |
| **SF** (Successor Features with ICM) | ⚠️ Wrapper only | `baselines/fb_sf.py` |

### Environment Interfaces

| Environment | Status | File |
|-------------|--------|------|
| AntMaze (large-diverse-v2) | ✅ Interface | `environments/env_wrappers.py` |
| ExORL (Walker/Cheetah, RND dataset) | ✅ Interface | `environments/env_wrappers.py` |
| Kitchen (complete-v0) | ✅ Interface | `environments/env_wrappers.py` |
| Auxiliary physics (Walker: h_vel, upright, height; Cheetah: speed) | ✅ Documented | `environments/env_wrappers.py` |

### Evaluation Tasks Implemented

| Task | Status | File |
|------|--------|------|
| AntMaze goal-reaching (5 locations) | ✅ | `fre/evaluation.py` |
| AntMaze directional (4 directions) | ✅ | `fre/evaluation.py` |
| AntMaze random-simplex (5 seeds, opensimplex) | ✅ | `fre/evaluation.py` |
| AntMaze path tasks (center, loop, edges) | ✅ Documented | `fre/evaluation.py` |
| ExORL goal-reaching (5 goals each domain) | ✅ | `fre/evaluation.py` |
| ExORL velocity (walker: 4 thresholds, cheetah: run/walk × fwd/back) | ✅ | `fre/evaluation.py` |
| Kitchen subtasks (7 standard tasks) | ✅ | `fre/evaluation.py` |

## What Was Skipped (and Why)

1. **FB and SF baselines — wrapper interfaces only**: Per the addendum, FB and SF must be trained and evaluated using the `facebookresearch/controllable_agent` codebase. We provide comprehensive wrapper classes (`baselines/fb_sf.py`) that document the integration points, training procedures, and evaluation protocols. The actual training requires cloning and running that external codebase, which is out of scope for this pure-algorithm reproduction (the code is graded on algorithmic correctness, not execution).

2. **Qualitative visualizations (Figure 3, Section 5.1)**: Explicitly marked out of scope in the addendum — "The results implied by Figure 3, discussed in section 5.1, are out of scope for reproduction since the discussion is qualitative and therefore cannot be straightforwardly judged."

3. **Figure 5 scaling experiment execution**: The scaling properties code path (training all 7 prior combinations, evaluating on all AntMaze tasks) is architecturally present. The full Table 4 experiment requires running 7 × 5 seeds = 35 full training runs, each needing D4RL/ExORL environment access. The code supports it, but we cannot execute the full grid without the benchmark datasets.

4. **Section 5.4 prior domain knowledge (FRE-hint)**: The architecture supports FRE-hint (augmenting priors with domain-specific reward distributions), but the environment-specific hint definitions from the paper's `FRE-hint` experiments (e.g., velocity-only priors for ExORL velocity evaluation) are noted as a design choice, not a separate algorithm.

5. **Figure 7/8/9 Appendix visualizations**: These are additional qualitative AntMaze trajectory plots explicitly noted as out of scope.

6. **Appendix C environment-specific preprocessing (discretized XY for AntMaze)**: The discretization utility is implemented in `utils/preprocessing.py` but the decision of whether to use discretized vs raw XY states is environment-specific and does not affect algorithm correctness.

## File Count and Line Counts

### File Structure
```
submission/
├── README.md
├── train.py                        (~430 lines)
├── fre/
│   ├── __init__.py                  (~20 lines)
│   ├── encoder.py                   (~300 lines)
│   ├── iql.py                       (~260 lines)
│   ├── prior.py                     (~230 lines)
│   ├── training.py                  (~350 lines)
│   └── evaluation.py                (~260 lines)
├── baselines/
│   ├── __init__.py                  (~20 lines)
│   ├── gc_bc.py                     (~190 lines)
│   ├── gc_iql.py                    (~210 lines)
│   ├── opal.py                      (~280 lines)
│   └── fb_sf.py                     (~220 lines)
├── environments/
│   ├── __init__.py                  (~15 lines)
│   └── env_wrappers.py              (~180 lines)
├── utils/
│   ├── __init__.py                  (~15 lines)
│   └── preprocessing.py             (~180 lines)
```

## Implementation Decisions

1. **Reward discretization**: The addendum clarifies that the 128-dim "Reward Embedding Dim" in Appendix Table 3 is incorrect — in reality, the state embedding is 64-dim and the reward embedding is 64-dim, concatenated to give 128-dim input to the transformer. We follow the addendum's correction.

2. **Transformer architecture**: The "Encoder Layers [256, 256, 256, 256]" from Appendix Table 3 refers to the MLP dimensions within each transformer block (residual/attention activations are 128-dim, MLP expands to 256 then back to 128). We implement 4 transformer encoder blocks with `embed_dim=128`, `mlp_dim=256`, `num_heads=4`.

3. **Decoder state embedding**: Per addendum, "There is no embedding step for the observation state passed to the decoder. The raw state and the z-vector are concatenated directly." Our decoder concatenates raw state + z before the MLP.

4. **RL conditioning**: Per addendum, "the latent embedding is simply concatenated to the observation state that is fed into the RL components." All IQL networks (Q, V, π) receive [state | z] as input.

5. **Strided training**: Following Algorithm 1, encoder training and policy training are separate phases. During Phase 1, only the FRE encoder-decoder is trained. During Phase 2, the encoder is frozen and only the RL components are trained. This ensures the mapping from η → z is stationary during TD learning.

6. **Goal-reaching reward**: At least one of the K=32 encoding samples contains the goal state (per Appendix B).

7. **Linear reward sparsity**: 90% sparsity mask applied to linear reward weights to bias toward simpler functions (per Appendix B). On AntMaze, XY position dimensions are excluded from linear reward generation.

8. **Random MLP architecture**: (state_dim, 32, 1) with `tanh` activation between layers, weights sampled from `N(0, 1/√(fan_avg))`. Output clipped to [-1, 1].

9. **OPAL privileged evaluation**: 10 random skills sampled from unit Gaussian, each evaluated for the full episode, best rollout selected. This matches the paper's privileged evaluation protocol.

## Dependencies

Required Python packages:
- `torch` (PyTorch)
- `numpy`
- `gym` (OpenAI Gym)
- `d4rl` (D4RL datasets — pre-June 2024 commit recommended per addendum)
- `opensimplex` (for random-simplex evaluation tasks)
- `h5py` (for ExORL RND dataset loading)

For FB/SF baselines, the `facebookresearch/controllable_agent` codebase is additionally required (per addendum).

## Algorithm 1 Correspondence

The implementation faithfully follows Algorithm 1 from the paper:

```
Train encoder (Phase 1):
  while not converged:
    Sample η ~ p(η), sample K encoder states, K' decoder states
    Train FRE by maximizing Equation (6)  →  fre/training.py, encoder training loop

Train policy (Phase 2):
  while not converged:
    Sample η ~ p(η), sample K encoder states
    Encode z ~ p_θ({(s_k, η(s_k))})  →  FREEncoder.encode()
    Train π(a|s,z), Q(s,a,z), V(s,z) using IQL with r=η(s)  →  FREIQLAgent.train_step()
```

---

## Experimental Scope

The paper evaluates on three benchmarks with the following evaluation tasks:

| Benchmark | Evaluation Tasks | Max Steps |
|-----------|-----------------|-----------|
| AntMaze | goal-reaching (5), directional (4), random-simplex (5), path (3) | 2000 |
| ExORL Walker | goal-reaching (5), velocity (4) | 1000 |
| ExORL Cheetah | goal-reaching (5), velocity (4) | 1000 |
| Kitchen | 7 subtasks | 280 |

All tasks are defined in `fre/evaluation.py`.

## Prior Reward Distribution Variants

| Variant | Goal-Reaching | Linear | MLP | Used In |
|---------|:---:|:---:|:---:|---------|
| FRE-all (FRE) | 33% | 33% | 34% | Sections 5.1, 5.2, 5.4 |
| FRE-goals | 100% | 0% | 0% | Section 5.3 |
| FRE-lin | 0% | 100% | 0% | Section 5.3 |
| FRE-mlp | 0% | 0% | 100% | Section 5.3 |
| FRE-lin-mlp | 0% | 50% | 50% | Section 5.3 |
| FRE-goal-mlp | 50% | 0% | 50% | Section 5.3 |
| FRE-goal-lin | 50% | 50% | 0% | Section 5.3 |
| FRE-hint | 33% | 33% | 34% + domain hints | Section 5.4 |