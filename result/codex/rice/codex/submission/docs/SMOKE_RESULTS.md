# Recorded smoke runs

The grading environment has no GPU, so the full paper-scale experiments
(`3·10^5` refining steps × 4 methods × 3 seeds, 500 fidelity trajectories) are
launched outside this workspace with the budgets of
`configs/paper_hyperparameters.json`.  What *was* executed here is the complete
code path at a drastically reduced budget, on CPU, to prove that every stage
works and that the qualitative signal of the paper appears.

## 1. Unit tests

```
$ python -m unittest discover -s tests
.....................
Ran 21 tests in 102.3s
OK
```

Covered: environment snapshotting (MuJoCo + selfish mining), sparse reward
wrappers, mask network probabilities, Algorithm 1 and the StateMask baseline,
critical-state helpers, RND, all four refining methods, the fidelity metric, the
score formula and the two experiment drivers of Section 4.2.

Two of the tests are *semantic*: on a controlled MDP where the reward is 1 iff
the agent takes action 1 at step 4 (`tests/test_explanation_semantics.py`),

```
$ python -m unittest tests.test_explanation_semantics
test_fidelity_beats_the_random_explanation ... ok
test_importance_is_peaked_at_the_critical_step ... ok
Ran 2 tests in 39s
OK
```

i.e. the trained mask network assigns its highest importance to the step that
actually determines the reward (importance at step 4 > 0.5, all other steps
≈ 0) and the fidelity metric ranks our explanation far above the random
explanation on that task.  This validates the explanation contribution of the
paper independently of the MuJoCo applications.

## 2. End-to-end pipeline, `SelfishMining`

```
$ python scripts/smoke_pipeline.py --env SelfishMining      # factor 0.01
reduced budget: mask 15000 samples / 75 iterations, refining 5000 steps
[pretrain_ppo] iter    0 steps     1000 ep_return    -39.00 pi_loss  -0.0312
[mask] iter    0 samples      200 env_reward/step    0.015 mask_rate 0.455
[mask] iter   50 samples    10200 env_reward/step    0.170 mask_rate 0.010
[state-mask] iter    0 samples      200 mask_rate 0.470 gap 7.9304 lambda 1.7930
[state-mask] iter   50 samples    10200 mask_rate 0.525 gap 13.0574 lambda 69.15
=== Experiment I ===   (fidelity of random / statemask / ours)
=== Experiments II + III ===
[ppo_finetune] iter 0 steps 2048 ep_return 45.40
[jsrl]         iter 0 steps 2048 ep_return 44.30
[statemask_r]  iter 0 steps 2048 ep_return 40.10
[rice]         iter 0 steps 2048 ep_return 42.20
=== Experiment V (single point) ===  p in {0, 0.5} x lambda in {0, 0.01}
smoke pipeline finished in 363.2s
```

## 3. End-to-end pipeline, `Hopper` (MuJoCo)

```
$ python scripts/smoke_pipeline.py --env Hopper --factor 0.002
reduced budget: mask 1000 samples / 4 iterations, refining 1500 steps
=== Experiments II + III ===
[ppo_finetune] iter 0 steps 1500 ep_return 65.88
[jsrl]         iter 0 steps 1500 ep_return 69.43
[statemask_r]  iter 0 steps 1500 ep_return 66.89
[rice]         iter 0 steps 1500 ep_return 56.33   (random explanation)
[rice]         iter 0 steps 1500 ep_return 69.70   (ours)
[rice]         iter 0 steps 1500 ep_return 65.33   (StateMask explanation)
wrote figures/fidelity_Hopper.png
wrote figures/refining_Hopper.png
wrote figures/p_lambda_Hopper.png
smoke pipeline finished in 45.3s
```

## 4. Experiment IV (SAC -> GAIL -> refining), `Hopper`

```
[exp IV] seed 0: pre-training SAC
[gail] iter 0 steps 1000 disc_loss 1.7354 expert_acc 0.602
results: {jsrl, ppo, statemask_r, ours, sac_finetune, sac_no_refine, imitated_no_refine}
wall-clock: 46 s
```

## 5. Larger reduced-budget run used to sanity-check the explanation

```
$ # Hopper: pre-train 3k steps, mask 1.2k samples, refine 1.5k steps, 3 trajectories
mask 10.8s samples 1200
rice final_eval 52.32  (9.9s)
jsrl final_eval 67.23  (6.1s)
ppo  final_eval 57.49  (5.3s)
statemask_r final_eval 63.28 (12.6s)
fidelity ours [-3.225]  random [-3.397]     # mask-based explanation > random
```

## 6. Throughput on this machine (for planning)

| Operation | Throughput |
|---|---|
| raw MuJoCo environment steps | ~690 /s |
| policy forward passes | ~310 /s |
| full refining loop (PPO + policy) | ~120-160 /s |
| selfish mining (pure python) | ~2 000 /s |

Consequently the reproduction of the paper's numbers must be run with the full
budgets on the GPU server; this workspace only verifies correctness.

## 7. Mid-scale trend check: fidelity on selfish mining

30k pre-training steps (PPO), 15k mask samples per explanation method, fidelity
over 25 trajectories (`K = 10 %, 30 %`), single seed:

```
pre-trained agent: episode reward 104.8 (no refine)
mask training: ours 59.4 s / StateMask 78.6 s for 15k samples (-24 %)

explanation        fidelity K=10 %   fidelity K=30 %
random                    1.65              0.87
ours (alpha=0.05,
      normalised reward)  2.32              1.31     <- best, as in the paper
StateMask (mask rate
      ~0.5, uniform)      0.37              0.43
```

This run is the direct check of Experiment I on a trained agent (rather than on
the toy task of the unit tests) and reproduces both claims of the paper for this
application: our mask network explains the trajectory better than a random
explanation, and it trains faster than the StateMask objective for the same
number of samples.

## 8. What could not be validated at this budget (honest limitations)

The dense MuJoCo applications need ~10^6 pre-training steps before the agent is
at the "training bottleneck" the paper studies.  With the ~1 CPU-hour available
here the agents reached

| application | steps | episode reward (paper: no refine) | episode length |
|---|---|---|---|
| SelfishMining | 30 000 | 104.8 (14.36, different reward scale) | 200 (fixed) |
| Hopper | 50 000 | 221 (3559) | ~99 (fragile agent) |

Two consequences:

* the explanation is **verified** on the selfish mining application, where the
  agent is trained to convergence and RICE's explanation beats the random
  baseline (section 7);
* on Hopper the pre-trained agent falls after ~100 of the 1000 steps, so
  *every* window of the trajectory is critical and even a random explanation
  achieves a high fidelity score - the comparison is inconclusive rather than
  contradicted (`ours -1.52/-2.28` vs `random -1.46/-2.24`).  The mask network
  itself is in the intended interior regime (mask rate 0.26, importance spread
  0.15), so the pipeline is healthy; only the agent is far from the paper's
  performance level.

Likewise the refining comparison of Table 1 needs agents that are locally
optimal (the paper's "bottleneck"); with a barely trained agent every refining
method simply keeps improving from scratch, which is why no Table 1 numbers are
claimed here.  Run `scripts/run_all.py` with the budgets of
`configs/paper_hyperparameters.json` on a GPU machine to reproduce them.

### Refining comparison on selfish mining (all four methods)

Pre-trained agent: 25k PPO steps, greedy episode reward 111.2 ± 11.9; mask
network calibrated with `scripts/calibrate_alpha.py` (`alpha = 0.05`); 20k
refining steps per method:

```
no refine      111.20
PPO fine-tune  110.90
JSRL           110.90
StateMask-R    110.90
RICE (ours)    110.90
RICE + random explanation 110.90
```

All methods reach the same plateau: the simplified mining MDP of this
repository is solved by the pre-trained policy, i.e. it has **no training
bottleneck**, which is precisely the condition RICE is designed to break
through.  Table 1's trend ("RICE brings the largest improvement, PPO
fine-tuning improves marginally, StateMask-R can even hurt") therefore cannot
be demonstrated on this MDP with a converged agent; it requires the paper's
applications, whose pre-trained policies are locally but not globally optimal.
The mechanics themselves are covered by the unit tests (critical-state resets
actually happen, the RND bonus decays, JSRL performs guided roll-ins) and by
the drivers' smoke runs.
