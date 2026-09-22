# Fine-tuning RL Models is Secretly a Forgetting Mitigation Problem — reproduction

This repository is a reproduction attempt of

> M. Wołczyk, B. Cupiał, M. Ostaszewski, M. Bortkiewicz, M. Zając, R. Pascanu,
> Ł. Kuciński, P. Miłoś.
> **Fine-tuning Reinforcement Learning Models is Secretly a Forgetting
> Mitigation Problem.** ICML 2024.

The paper's thesis is that fine-tuning a pre-trained RL agent fails mainly
because the agent **forgets pre-trained capabilities (FPC)** in the parts of the
state space that are not visited during the early phase of fine-tuning. The
paper names two instances of FPC — the **state coverage gap** and the
**imperfect cloning gap** — and shows that standard knowledge-retention methods
(EWC, behavioral cloning, kickstarting, episodic memory) fix the problem in
NetHack, Montezuma's Revenge and a multi-stage robotic task (`RoboticSequence`).

The repository contains a complete, runnable implementation of every component
described in the main text and the appendices that the main text depends on: the
training loops for the three environments, the four knowledge-retention methods,
the pre-training / data-collection procedures and the evaluations that produce
the paper's figures.

---

## 1. How to use this repository

```bash
pip install -r requirements.txt          # numpy, scipy, torch, matplotlib (+pytest)

python scripts/smoke_test.py             # end-to-end check of every training loop (CPU, ~30s)
python -m pytest tests -q                # unit tests
python scripts/run_toy_experiments.py    # Appendix-A toy experiments (CPU, ~25s)
```

The full experiments are **not** runnable in this environment (no GPU, and the
individual runs take tens of hours); the scripts below are written to be
launched on a machine with the required environments installed:

```bash
# NetHack: APPO fine-tuning with kickstarting / behavioral cloning / EWC
python scripts/train_nethack.py --compute-fisher            # 10 000 NLD-AA batches
python scripts/train_nethack.py --method ks --steps 500000000

# Montezuma's Revenge: PPO + RND
python scripts/train_montezuma.py --phase pretrain  --steps 100000000
python scripts/train_montezuma.py --phase collect
python scripts/train_montezuma.py --phase finetune  --method bc --steps 50000000

# RoboticSequence (Meta-World, SAC)
python scripts/train_metaworld.py --phase grid --seeds 20

# figures
python scripts/plot_figures.py --results results --out results/figures
```

Optional dependencies for the real environments are listed (commented out) in
`requirements.txt`. Every environment is imported lazily, so the library and the
toy experiments work with `numpy/scipy/torch/matplotlib` alone.

---

## 2. Map of the codebase to the paper

| Paper element | Where it lives |
|---|---|
| FPC, state coverage gap, imperfect cloning gap (Sec. 2, Fig. 1–2) | `fpc/toy/two_state_mdp.py`, `fpc/toy/apple_retrieval.py` (analytic demonstrations), `fpc/metaworld/robotic_sequence.py` (the two-stage instantiation) |
| Knowledge retention: EWC (Sec. 2, App. C.1) | `fpc/retention/ewc.py`, `fpc/retention/fisher.py` |
| BC / kickstarting KL losses (Sec. 2, App. C.2) | `fpc/retention/distillation.py` |
| Episodic memory (App. C.3) | `fpc/retention/episodic_memory.py` |
| Actor-only regularization (App. C.5) | `retention` is attached to the actor/policy in every trainer |
| NetHack model + APPO + retention (App. B.1) | `fpc/nethack/model.py`, `fpc/nethack/appo.py`, `fpc/nethack/config.py` |
| NLD-AA dataset, BC buffer, 10 000 Fisher batches | `fpc/nethack/dataset.py` |
| Montezuma PPO + RND + BC/EWC (App. B.2) | `fpc/montezuma/{ppo,rnd,model,env,config}.py` |
| RoboticSequence + SAC + EWC/BC/EM (App. B.3, Alg. 1) | `fpc/metaworld/{robotic_sequence,sac,train,config}.py` |
| CKA analysis (App. F, Fig. 20) | `fpc/analysis/cka.py` |
| Forward transfer / AUC (App. F, Table 6) | `fpc/analysis/forward_transfer.py` |
| Expert log-likelihoods + PCA (Fig. 8) | `fpc/analysis/forgetting.py`, `fpc/metaworld/analysis.py` |
| Level / room visitation densities (Fig. 4, 16, 18) | `fpc/analysis/forgetting.py`, `fpc/nethack/eval.py`, `fpc/montezuma/eval.py` |
| Evaluation protocols (Sec. 3, App. B.1–B.2, addendum) | `fpc/nethack/eval.py`, `fpc/montezuma/eval.py`, `fpc/metaworld/eval.py` |
| Figures | `scripts/plot_figures.py` |

---

## 3. What was reproduced, and how far

### 3.1 Knowledge-retention core — **implemented and verified**

All four methods of Section 2 / Appendix C are implemented from the paper's
equations:

* **EWC** — `L_aux(θ) = Σ_i F_i (θ*_i − θ_i)²` with the **diagonal empirical
  Fisher** `F_i = E_{s~D, a~π*}[ (∂ log π*(a|s) / ∂θ_i)² ]`
  (`fpc/retention/fisher.py`). The estimator supports the 10 000-batch NLD-AA
  sampling required by the addendum, and EWC never regularizes the critic
  (Appendix C.5).
* **Behavioral cloning** — `L_BC(θ) = E_{s~B_BC}[ D_KL(π* || π_θ) ]` on the
  expert buffer (`fpc/retention/distillation.py`). Because
  `D_KL(π*||π_θ) = H(π*) − E_{π*}[log π_θ]`, this is exactly the cross-entropy
  BC objective; the reverse direction is available too since the main text and
  Appendix C.2 write the KL in opposite orders.
* **Kickstarting** — the same KL but on states sampled by the online policy,
  with the NetHack schedule (coefficient `0.5`, exponential decay `0.99998`
  per training step) — `fpc/retention/schedules.py`.
* **Episodic memory** — a replay buffer with a **protected region** that is
  never overwritten (10 % of the buffer, i.e. 10 000 of 100 000 transitions in
  Meta-World), `fpc/retention/episodic_memory.py`.

`tests/test_retention.py` verifies the KL identities, the decay schedule, that
the EWC penalty is exactly zero at the anchor and grows away from it, that the
Fisher is non-negative and matches a manual gradient-square computation, and
that the protected replay region is never overwritten.

### 3.2 The three experimental domains — **implemented** (runs require the real environments)

**NetHack (Section 3–5, Figure 3a, Figure 5, Table 4/5).**
`fpc/nethack/model.py` implements the Tuyls et al. (2023) architecture: per-cell
`(character, colour)` embeddings processed by a ResNet trunk for the dungeon
screen, two-layer MLPs for `blstats` and `message`, a shared LSTM core, and
policy/baseline heads (hidden size 1738, Table 1). `fpc/nethack/appo.py`
implements asynchronous PPO (Petrenko et al., 2020) with worker threads, a
batcher, clipped PPO and the APPO/importance corrections, together with the
retention hooks: kickstarting on online data (`0.5`, decay `0.99998`),
behavioral cloning on AutoAscend trajectories (`2.0`, no decay), EWC with
coefficient `2e6`, entropy disabled when retention is on, and **frozen
encoders**. The 500M-step baseline-head pre-training of Appendix B.1 is
implemented as `APPO.pretrain_baseline_head`. `fpc/nethack/dataset.py` contains
the NLD-AA plumbing (Human Monk, ~8000 games), the `B_BC = {(s, π*(s))}` buffer
and the 10 000-batch Fisher iterator.

**Montezuma's Revenge (Section 3–5, Figure 3b, Figure 6).**
`fpc/montezuma/rnd.py` implements Random Network Distillation (frozen target +
trained predictor, 512-dim features, whitened intrinsic reward);
`fpc/montezuma/ppo.py` implements PPO + RND with the Table 2 hyperparameters
(128 envs, 128 steps, 4 epochs, GAE, sticky actions 0.25, clipping 0.5,
`MaxStepPerEpisode = 4500`, `UpdateProportion = 0.25`) plus BC/EWC retainers;
`fpc/montezuma/env.py` contains the frame-stacking/room-restriction wrappers and
the 500-trajectory BC collection; `fpc/montezuma/eval.py` implements the
**Room-7 success-rate metric** of Figure 6 (coin, new item, or exit through a
different passage) evaluated **every 5M steps** as required by the addendum, and
the room-visitation accounting of Figure 18.

**RoboticSequence (Section 3–5, Figure 3c, Figure 7, Appendix B.3/F).**
`fpc/metaworld/robotic_sequence.py` implements Algorithm 1 together with every
modification listed in Appendix B.3: randomised start/goal, termination on
success *or* time limit with no bootstrapping, the normalized timestep appended
to the observation, and the augmented success reward `r' = β·r·(T − t)` with
`β = 1.5`. `fpc/metaworld/sac.py` implements SAC with 4×256 Leaky-ReLU layers,
layer normalization after the first layer, a **separate head per stage**,
automatic entropy tuning, twin critics and Polyak-averaged targets (lr `1e-3`,
batch 128). EWC (`100`), BC (`1`, 10 000 samples) and EM (10 000 protected
transitions) follow Table 3. `fpc/metaworld/train.py` contains the SAC
pre-training of `π*` on the last two stages, the BC/EM buffer collection and the
multi-seed fine-tuning grid.

### 3.3 Toy environments (Appendix A) — **reproduced with numbers**

These are cheap enough to run here, and they *do* reproduce the paper's numbers.
Running `scripts/run_toy_experiments.py` (results in `results/toy/`) gives:

**Two-state MDPs (Appendix A.1).** Implementing the paper's closed form

```
v0(θ) = 1/(1−γ) · [θ + r0(1−θ)(1−γ f_θ) + γ θ r1 (1−f_θ)] / [1 − γ f_θ + γθ]
```

with `r0 = 0`, `r1 = −1`, `γ = 0.9` and the `f_θ = 2|θ − 0.5|` parametrisation,
gradient ascent on `v0` started at `θ = 0` is trapped at the suboptimal local
maximum

| quantity | paper | this reproduction |
|---|---|---|
| fixed point `θ*` | 0.11 | **0.1111** |
| value at `θ*` | 2.22 | **2.2222** |
| optimal value (`θ = 1`) | 10 | **10.0** |

The qualitative claims also hold: `f_0 = 1` so a policy pre-trained on `s1`
stays there; the state-coverage-gap parametrisation
`f_θ = (−ε/(1−ε/2))θ + 1` (for `θ ≤ 1−ε/2`, then `2θ−1`) is monotone in this
reward configuration, and the imperfect-cloning-gap landscape has an interior
local maximum that fits a pre-trained policy which is only slightly perturbed.

**AppleRetrieval (Appendix A.2).** A 1D grid-world with a CLOSE phase
(walk to the apple) and a FAR phase (walk home), trained with REINFORCE on the
two-parameter policy `π(o) = σ(w·o + b)` initialised from a Phase-2-only
pre-training. The two effects reported in the paper appear:

* *Figure 11 (impact of `c`)* — smaller `c` pushes the pre-trained model towards
  the bias solution (higher `|b|/|w|`) and produces strong forgetting; larger
  `c` keeps the weight solution and no forgetting occurs:

| `c` | early `|b|/|w|` | Phase-2 (FAR) success after fine-tuning |
|---|---|---|
| 0.1 | 3.24 | **0.00** |
| 0.3 | 1.07 | **0.00** |
| 0.5 | 0.61 | **0.00** |
| 1.0 | 0.22 | 1.00 |
| 5.0 | 0.14 | 1.00 |

* *Figure 10 (impact of `M`)* — with `c = 0.5` the FAR-phase skill is retained
  for short distances and collapses for long ones (`M = 5, 15` → success 1.00;
  `M = 30, 50` → success 0.00), i.e. forgetting becomes more severe the longer
  the CLOSE phase lasts.

### 3.4 Analysis tools (Section 5, Appendix D/F) — **implemented**

`fpc/analysis/cka.py` implements CKA with a linear kernel
`CKA(K,L) = HSIC(K,L)/√(HSIC(K,K)HSIC(L,L))` and the layer-wise tracking used in
Figure 20. `fpc/analysis/forward_transfer.py` implements
`(AUC − AUC_b)/(1 − AUC_b)` used in Table 6, plus 90 % confidence intervals as in
Appendix B.3. `fpc/analysis/forgetting.py` implements the expert-action
log-likelihoods and the 2D PCA projection of Figure 8, and the
`(turns, max-level)` density of Figures 4/16. `fpc/metaworld/analysis.py` wires
these to the SAC agent (Gaussian log-density with tanh correction) and computes
the log-likelihoods **every 50K steps** as required by the addendum.

### 3.5 A CPU-runnable harness for the SAC + retention pipeline — **provided, not converged**

`fpc/metaworld/point_chain.py` and
`scripts/run_robotic_sequence_standin.py` implement a self-contained
multi-stage continuous-control task (`PointReachChain`) with *exactly* the
structure of RoboticSequence — a chain of stages where the agent only advances on
success, a separate head per stage, the normalised timestep in the observation,
no bootstrapping at the time limit and the augmented success reward
`r' = β·r·(T − t)`.  `π*` is pre-trained on the last two stages so that the first
stages form the CLOSE set and the pre-trained stages the FAR set.

This exists purely because Meta-World requires Python ≥ 3.10 (this environment
has Python 3.9) and MuJoCo, so the real RoboticSequence cannot be executed here.
**It is included as a harness, not as evidence.** Running it confirms that the
harness itself is correct — SAC reaches 100 % success on the pre-trained FAR
chain and then on the full chain — but the four stages are too *homogeneous* to
produce the paper's forgetting signal: once the agent learns the reaching
behaviour it transfers to every stage, so there is nothing to forget. The
paper's stages (hammer, push, peg-unplug-side, push-wall) are genuinely
different skills, which is what makes the state coverage gap appear. No numbers
are therefore claimed from the stand-in; it is useful for exercising the full
`SAC + EWC/BC/EM` code path end-to-end on CPU in a few minutes.

---

## 4. What is *not* reproduced here, and why

* **The full training runs.** They require a GPU, hundreds of millions to tens
  of billions of environment steps (`500M` steps ≈ 24h on an A100 for NetHack,
  `48h` per Meta-World run on 8 CPU cores) and the external environments
  (`nle`, `sample-factory`, `ale-py` + Atari ROMs, `metaworld` + MuJoCo). The
  code is written for those runs; `scripts/smoke_test.py` exercises every loop
  with mock environments instead so that the implementations are verified.
* **Meta-World execution here.** The Farama Meta-World release requires
  Python ≥ 3.10 and this environment has Python 3.9, so RoboticSequence could
  not be executed; see Section 3.5 for the stand-in harness.
* **The 115B-transition pre-trained NetHack checkpoint** (Tuyls et al., 2023)
  and the AutoAscend expert. `fpc/nethack/eval.py:collect_expert_saves`
  documents how the 200 per-level save states of Figure 5 are generated with
  AutoAscend, but the bot itself is not vendored.
* **Appendix-only experiments** that are not needed by the main text
  (network-size sweep, prefix-length sweep, translated-observation sequence,
  head reset, alternative sequences). The code contains the switches for them
  (`--reset-last-layer`, `--translate`, `MetaworldConfig.prefix_tasks`) and the
  metrics (`forward_transfer_table`), but they are not run.

### Ambiguities encountered

1. **Two-state MDP rewards.** Appendix A.1 prints the closed form of `v0` but
   never states `r0`, `r1` or `γ`. With `r0 = r1 = 1` the printed formula is
   degenerate (numerator ≡ denominator, so `v0 ≡ 1/(1−γ)` for every `θ`). We
   therefore expose `r0`, `r1`, `γ` as arguments and report the parameter set
   (`r0 = 0`, `r1 = −1`, `γ = 0.9`) that reproduces the paper's stated fixed
   point and value (0.11 / 2.22) exactly against an optimum of 10.
2. **KL direction in the BC loss.** The main text writes
   `D_KL(π*(s) ‖ π_θ(s))` while Appendix C.2 writes `D_KL^s(π_θ ‖ π_*)`.
   The implementation defaults to the forward direction (which is equivalent to
   the cross-entropy BC objective used to pre-train the NetHack model) and
   exposes `kl_direction="reverse"` as an option.
3. **Montezuma's Revenge room index.** ALE does not expose the room index; the
   room is read from the emulator RAM. The precise byte and the RAM→room map
   depend on the emulator build, which is why the room detector is isolated in
   `fpc/montezuma/env.py:room_transition_detector`.

---

## 5. Repository layout

```
fpc/
  retention/      knowledge-retention methods (EWC, BC, KS, EM) + Fisher + schedules
  nethack/        NetHack model, APPO loop, NLD-AA dataset, evaluation
  montezuma/      PPO+RND, Atari wrappers, BC collection, Room-7 evaluation
  metaworld/      RoboticSequence, SAC, retention glue, analyses, CPU stand-in
  toy/            two-state MDPs, AppleRetrieval
  analysis/       CKA, forward transfer, log-likelihoods, visitation densities
  testing/        mock environments for the smoke tests
scripts/          entry points (training, plotting, toy experiments, smoke test)
tests/            unit tests
results/toy/      outputs of scripts/run_toy_experiments.py
```

## 6. Verification status

| check | command | status |
|---|---|---|
| unit tests (retention, GAE, Algorithm 1, forward transfer, SAC, toys) | `python -m pytest tests -q` | 22 passed |
| all training loops (mock envs, all retainers) | `python scripts/smoke_test.py` | passed |
| toy experiments + figures | `python scripts/run_toy_experiments.py` | passed, numbers match the paper |
