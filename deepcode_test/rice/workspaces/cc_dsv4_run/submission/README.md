# RICE: Breaking Through Training Bottlenecks of Reinforcement Learning with Explanation

Reproduction of the RICE paper (Cheng et al., ICML 2024).

## Paper Reference

Cheng, Z., Wu, X., Yu, J., Yang, S., Wang, G., & Xing, X. (2024). RICE: Breaking Through the Training Bottlenecks of Reinforcement Learning with Explanation. *Proceedings of the 41st International Conference on Machine Learning*, PMLR 235.

## Implemented Components

### Core Methods (Fully Implemented)

1. **Mask Network (Algorithm 1)** — `rice/mask_network.py`
   - Improved StateMask explanation method
   - Reformulated objective: J(θ) = max η(π̄) with vanilla PPO (replaces primal-dual optimization)
   - Theorem 3.3: η(π̄) ≤ η(π) justification
   - Reward bonus α for encouraging blinding (prevents trivial "never blind" solution)
   - Trajectory-wise importance scoring
   - Most-critical-state identification for exploration frontiers

2. **RICE Refinement (Algorithm 2)** — `rice/refiner.py`
   - Mixed initial state distribution: μ(s) = β·d_ρ^π̂(s) + (1-β)·ρ(s)
   - Controlled by hyper-parameter p (reset probability threshold)
   - Random Network Distillation (RND) exploration bonus
   - Modified reward: R'(s,a) = R(s,a) + λ·||f(s') - f̂(s')||²
   - PPO-based policy optimization
   - RND predictor normalization (running mean/std)

3. **RND Exploration** — `rice/rnd.py`
   - Target network f: fixed, randomly initialized
   - Predictor network f̂: trained to match target output via MSE
   - Exploration bonus = ||f(s) - f̂(s)||² (normalized)
   - Implements running statistics normalization from Burda et al. (2018)

4. **Fidelity Score Computation** — `rice/fidelity.py`
   - Complete fidelity evaluation pipeline matching paper methodology
   - Sliding window selection of most critical segment
   - Fast-forward + random action randomization
   - Score formula: log(d/d_max) - log(l/L)
   - Support for multiple K values and repeated trials

### Baseline Methods (Fully Implemented) — `rice/baselines.py`

5. **PPO Fine-tuning Baseline**: Lower learning rate, continue training
6. **StateMask-R Baseline**: Reset to critical states only, fine-tune (no mixing, no exploration)
7. **Jump-Start RL (JSRL)**: Curriculum-based refinement with guide policy
8. **Self-Imitation Learning (SIL)**: Prioritize high-return past experiences
9. **Random Explanation Baseline**: Random importance scores

### Policy Networks — `rice/policy.py`

10. **MlpPolicy**: Standard SB3-style actor-critic for continuous action spaces (MuJoCo)
11. **DiscreteMlpPolicy**: Actor-critic for discrete action spaces (Selfish Mining, CAGE, Malware)
12. **SACPolicy**: Soft Actor-Critic policy for SAC pre-trained agents
13. **GAILDiscriminator**: For imitation learning from SAC (Experiment IV)

### Experiments — `rice/experiments.py`

14. **Experiment I**: Fidelity and efficiency evaluation framework
15. **Experiment II**: Refining method comparison (PPO-FT, JSRL, StateMask-R, RICE)
16. **Experiment III**: Explanation quality impact (Random vs Ours)
17. **Experiment IV**: SAC agent refining via GAIL
18. **Experiment V**: Hyper-parameter sensitivity (p, λ, α)
19. **SIL Comparison**: RICE vs Self-Imitation Learning

### Supporting Modules

20. **Replay Buffers** — `rice/buffer.py`
    - RolloutBuffer: On-policy PPO rollout storage
    - TrajectoryBuffer: Episode-level storage
    - PriorityBuffer: Return-prioritized buffer for SIL

21. **Utilities** — `rice/utils.py`
    - Random action samplers (continuous/discrete)
    - Policy evaluation harness
    - Episode runners
    - Random seed management

## Components Skipped

| Component | Reason |
|-----------|--------|
| Section 3.4 (Theoretical Analysis proofs) | Out of scope per addendum |
| Malware Mutation experiments | Out of scope per addendum |
| SparseWalker2d hyper-parameter sensitivity | Out of scope per addendum |
| Autonomous driving qualitative analysis | Out of scope per addendum |
| Environment-specific implementations (MuJoCo wrappers, Selfish Mining env, Cage Challenge env, Auto Driving env, Malware env) | These require proprietary/specific environment code. The RICE algorithm interfaces with any environment via abstract callbacks (env_reset_fn, env_step_fn, policy_fn). |
| Exact neural network architecture for Cage Challenge ([64,64,64]) and Autonomous Driving (DI-drive VAC) | Architecture parameters are exposed as configuration; specific env wrappers are not in scope |
| Experiment results/plots | The code implements the experimental framework but does not run experiments (requires pre-trained agents + environments) |
| Table 3 exact hyper-parameter selection | Default hyper-parameters provided; per-environment tuning is done at experiment run time |

## Architecture

```
rice/
├── __init__.py          # Package metadata
├── mask_network.py      # Algorithm 1: Mask Network training
├── rnd.py               # RND exploration bonus module
├── refiner.py           # Algorithm 2: RICE refinement + RICEAgent
├── fidelity.py          # Fidelity score computation
├── baselines.py         # PPO-FT, StateMask-R, JSRL, SIL baselines
├── policy.py            # Policy network architectures
├── buffer.py            # Replay buffers
├── experiments.py       # Experiment I-V runners
└── utils.py             # Utility functions
```

## Key Design Decisions

1. **Black-box policy assumption**: All RICE components interact with policies through abstract interfaces (get_action, get_value), making them independent of internal network architecture.

2. **State restoration**: The code supports both simulator-based state restoration (via set_state) and replay-based restoration (re-executing the trajectory). This is noted in Appendix C.1.

3. **RND normalization**: Running mean/std normalization of RND bonuses is implemented (following the RND paper), since raw prediction errors can vary by orders of magnitude.

4. **Abstract environment interface**: Environments are accessed through function callbacks (env_reset_fn, env_step_fn, etc.) rather than hard-coded wrappers, making the RICE algorithm environment-agnostic.

5. **No pinned package versions**: Per the strategy guidance, package requirements are minimal (numpy, torch with flexible version ranges).

## Hyper-parameter Defaults

Following the paper's Table 3 recommendations:
- `p` (reset probability): 0.25 (Hopper, Walker2d, Selfish, Auto), 0.50 (Reacher, HalfCheetah, Cage)
- `λ` (RND weight): 0.001 (Hopper, Selfish), 0.01 (others)
- `α` (blinding bonus): 0.0001 (all applications)